use async_trait::async_trait;
use futures_util::StreamExt;
use tokio::sync::mpsc::UnboundedSender;

use super::{ChatMessage, ChatOptions, ChatRole, ChatStreamEvent, EngineError, InferenceBackend, ModelInfo};

const DEFAULT_MODEL: &str = "gemini-3.6-flash";

pub struct GeminiBackend {
    api_key: String,
}

impl GeminiBackend {
    pub fn new(api_key: String) -> Self {
        Self { api_key }
    }
}

#[async_trait]
impl InferenceBackend for GeminiBackend {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, EngineError> {
        Ok(vec![ModelInfo {
            name: DEFAULT_MODEL.to_string(),
            size_bytes: None,
            parameter_size: None,
            quantization: None,
        }])
    }

    async fn chat_stream(
        &self,
        request_id: String,
        model: &str,
        messages: &[ChatMessage],
        _options: ChatOptions,
        sender: UnboundedSender<ChatStreamEvent>,
    ) -> Result<(), EngineError> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse"
        );
        let client = reqwest::Client::new();

        let system_text = messages
            .iter()
            .filter(|m| matches!(m.role, ChatRole::System))
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");

        let contents: Vec<serde_json::Value> = messages
            .iter()
            .filter(|m| !matches!(m.role, ChatRole::System))
            .map(|m| {
                let role = match m.role {
                    ChatRole::User => "user",
                    _ => "model",
                };
                serde_json::json!({ "role": role, "parts": [{ "text": m.content }] })
            })
            .collect();

        let mut body = serde_json::json!({ "contents": contents });
        if !system_text.is_empty() {
            body["systemInstruction"] = serde_json::json!({ "parts": [{ "text": system_text }] });
        }

        let resp = client
            .post(&url)
            .header("x-goog-api-key", &self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| EngineError::Unreachable(url.clone(), e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(EngineError::Response(format!("{status}: {text}")));
        }

        let mut byte_stream = resp.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk) = byte_stream.next().await {
            let chunk = chunk.map_err(|e| EngineError::Unreachable(url.clone(), e.to_string()))?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(newline_pos) = buffer.find('\n') {
                let line = buffer[..newline_pos].trim().to_string();
                buffer.drain(..=newline_pos);

                let Some(payload) = line.strip_prefix("data:").map(str::trim) else {
                    continue;
                };
                if payload.is_empty() {
                    continue;
                }

                let value: serde_json::Value = match serde_json::from_str(payload) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                if let Some(err) = value.get("error") {
                    let msg = err
                        .get("message")
                        .and_then(|m| m.as_str())
                        .unwrap_or("unknown error");
                    let _ = sender.send(ChatStreamEvent::Error {
                        request_id: request_id.clone(),
                        message: msg.to_string(),
                    });
                    return Ok(());
                }

                if let Some(text) = value
                    .get("candidates")
                    .and_then(|c| c.get(0))
                    .and_then(|c| c.get("content"))
                    .and_then(|c| c.get("parts"))
                    .and_then(|p| p.get(0))
                    .and_then(|p| p.get("text"))
                    .and_then(|t| t.as_str())
                {
                    if !text.is_empty() {
                        let _ = sender.send(ChatStreamEvent::Token {
                            request_id: request_id.clone(),
                            content: text.to_string(),
                        });
                    }
                }
            }
        }

        let _ = sender.send(ChatStreamEvent::Done { request_id });
        Ok(())
    }
}
