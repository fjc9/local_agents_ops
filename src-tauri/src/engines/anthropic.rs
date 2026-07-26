use async_trait::async_trait;
use futures_util::StreamExt;
use serde::Serialize;
use tokio::sync::mpsc::UnboundedSender;

use super::{ChatMessage, ChatOptions, ChatRole, ChatStreamEvent, EngineError, InferenceBackend, ModelInfo};

const API_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u32 = 4096;

/// Raw-HTTP client for Anthropic's Messages API. There's no official Rust
/// SDK, so this talks the wire format directly (see claude-api skill).
pub struct AnthropicBackend {
    api_key: String,
    client: reqwest::Client,
}

impl AnthropicBackend {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            client: reqwest::Client::new(),
        }
    }
}

#[derive(Serialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct AnthropicThinking {
    #[serde(rename = "type")]
    kind: &'static str,
    display: &'static str,
}

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<AnthropicMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<AnthropicThinking>,
}

#[async_trait]
impl InferenceBackend for AnthropicBackend {
    async fn list_models(&self) -> Result<Vec<ModelInfo>, EngineError> {
        Ok(vec![ModelInfo {
            name: "claude-opus-4-8".to_string(),
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
        options: ChatOptions,
        sender: UnboundedSender<ChatStreamEvent>,
    ) -> Result<(), EngineError> {
        let url = "https://api.anthropic.com/v1/messages";

        let system = {
            let parts: Vec<&str> = messages
                .iter()
                .filter(|m| matches!(m.role, ChatRole::System))
                .map(|m| m.content.as_str())
                .collect();
            if parts.is_empty() {
                None
            } else {
                Some(parts.join("\n\n"))
            }
        };

        let anthropic_messages = messages
            .iter()
            .filter(|m| !matches!(m.role, ChatRole::System))
            .map(|m| AnthropicMessage {
                role: match m.role {
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                    ChatRole::System => unreachable!("filtered above"),
                }
                .to_string(),
                content: m.content.clone(),
            })
            .collect();

        let body = AnthropicRequest {
            model: model.to_string(),
            max_tokens: DEFAULT_MAX_TOKENS,
            messages: anthropic_messages,
            stream: true,
            system,
            thinking: options.think.then_some(AnthropicThinking {
                kind: "adaptive",
                display: "summarized",
            }),
        };

        let resp = self
            .client
            .post(url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
            .json(&body)
            .send()
            .await
            .map_err(|e| EngineError::Unreachable(url.to_string(), e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(EngineError::Response(format!("{status}: {text}")));
        }

        let mut byte_stream = resp.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk) = byte_stream.next().await {
            let chunk = chunk.map_err(|e| EngineError::Unreachable(url.to_string(), e.to_string()))?;
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

                match value.get("type").and_then(|t| t.as_str()) {
                    Some("content_block_delta") => {
                        if let Some(delta) = value.get("delta") {
                            match delta.get("type").and_then(|t| t.as_str()) {
                                Some("text_delta") => {
                                    if let Some(text) = delta.get("text").and_then(|t| t.as_str()) {
                                        let _ = sender.send(ChatStreamEvent::Token {
                                            request_id: request_id.clone(),
                                            content: text.to_string(),
                                        });
                                    }
                                }
                                Some("thinking_delta") => {
                                    if let Some(text) = delta.get("thinking").and_then(|t| t.as_str()) {
                                        let _ = sender.send(ChatStreamEvent::Thinking {
                                            request_id: request_id.clone(),
                                            content: text.to_string(),
                                        });
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    Some("message_stop") => {
                        let _ = sender.send(ChatStreamEvent::Done {
                            request_id: request_id.clone(),
                        });
                        return Ok(());
                    }
                    Some("error") => {
                        let msg = value
                            .get("error")
                            .and_then(|e| e.get("message"))
                            .and_then(|m| m.as_str())
                            .unwrap_or("unknown error");
                        let _ = sender.send(ChatStreamEvent::Error {
                            request_id: request_id.clone(),
                            message: msg.to_string(),
                        });
                        return Ok(());
                    }
                    _ => {}
                }
            }
        }

        let _ = sender.send(ChatStreamEvent::Done { request_id });
        Ok(())
    }
}
