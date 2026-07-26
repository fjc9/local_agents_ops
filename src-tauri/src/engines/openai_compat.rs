use async_trait::async_trait;
use futures_util::StreamExt;
use serde::Serialize;
use tokio::sync::mpsc::UnboundedSender;

use super::{ChatMessage, ChatOptions, ChatRole, ChatStreamEvent, EngineError, InferenceBackend, ModelInfo};

/// Client for any OpenAI-compatible `/chat/completions` API. Used for both
/// OpenAI itself and xAI's Grok, which intentionally mirrors the same wire
/// format so existing OpenAI clients work against it unchanged.
pub struct OpenAiCompatBackend {
    base_url: String,
    api_key: String,
    default_model: String,
}

impl OpenAiCompatBackend {
    pub fn new(
        base_url: impl Into<String>,
        api_key: String,
        default_model: impl Into<String>,
    ) -> Self {
        Self {
            base_url: base_url.into(),
            api_key,
            default_model: default_model.into(),
        }
    }
}

#[derive(Serialize)]
struct OaiMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct OaiRequest {
    model: String,
    messages: Vec<OaiMessage>,
    stream: bool,
}

#[async_trait]
impl InferenceBackend for OpenAiCompatBackend {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, EngineError> {
        Ok(vec![ModelInfo {
            name: self.default_model.clone(),
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
        let url = format!("{}/chat/completions", self.base_url);
        let client = reqwest::Client::new();

        let body = OaiRequest {
            model: model.to_string(),
            messages: messages
                .iter()
                .map(|m| OaiMessage {
                    role: match m.role {
                        ChatRole::System => "system",
                        ChatRole::User => "user",
                        ChatRole::Assistant => "assistant",
                    }
                    .to_string(),
                    content: m.content.clone(),
                })
                .collect(),
            stream: true,
        };

        let resp = client
            .post(&url)
            .bearer_auth(&self.api_key)
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
                if payload == "[DONE]" {
                    let _ = sender.send(ChatStreamEvent::Done {
                        request_id: request_id.clone(),
                    });
                    return Ok(());
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

                if let Some(content) = value
                    .get("choices")
                    .and_then(|c| c.get(0))
                    .and_then(|c| c.get("delta"))
                    .and_then(|d| d.get("content"))
                    .and_then(|t| t.as_str())
                {
                    if !content.is_empty() {
                        let _ = sender.send(ChatStreamEvent::Token {
                            request_id: request_id.clone(),
                            content: content.to_string(),
                        });
                    }
                }
            }
        }

        let _ = sender.send(ChatStreamEvent::Done { request_id });
        Ok(())
    }
}
