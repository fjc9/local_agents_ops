use async_trait::async_trait;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::UnboundedSender;

use super::{
    ChatMessage, ChatOptions, ChatStreamEvent, EngineError, InferenceBackend, ModelInfo,
    PullProgressEvent,
};

pub struct OllamaBackend {
    base_url: String,
    client: reqwest::Client,
}

impl OllamaBackend {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            client: reqwest::Client::new(),
        }
    }

    pub fn default_local() -> Self {
        Self::new("http://localhost:11434")
    }

    /// Version string reported by the running Ollama server, e.g. "0.31.1".
    /// Used both for a startup health check and to detect when the engine
    /// binary itself is out of date.
    pub async fn version(&self) -> Result<String, EngineError> {
        let url = format!("{}/api/version", self.base_url);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| EngineError::Unreachable(url.clone(), e.to_string()))?;

        let body: VersionResponse = resp
            .json()
            .await
            .map_err(|e| EngineError::Parse(e.to_string()))?;
        Ok(body.version)
    }

    /// Pulls a model, forwarding Ollama's own download-progress lines
    /// (manifest -> per-layer digest download -> verify -> success) as
    /// they arrive rather than waiting for the whole multi-GB transfer.
    pub async fn pull_model(&self, model: String, sender: UnboundedSender<PullProgressEvent>) {
        let url = format!("{}/api/pull", self.base_url);
        let resp = match self
            .client
            .post(&url)
            .json(&serde_json::json!({ "name": &model, "stream": true }))
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                let _ = sender.send(PullProgressEvent::Error {
                    model,
                    message: e.to_string(),
                });
                return;
            }
        };

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            let _ = sender.send(PullProgressEvent::Error {
                model,
                message: format!("{status}: {text}"),
            });
            return;
        }

        let mut byte_stream = resp.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk) = byte_stream.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(e) => {
                    let _ = sender.send(PullProgressEvent::Error {
                        model,
                        message: e.to_string(),
                    });
                    return;
                }
            };
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(newline_pos) = buffer.find('\n') {
                let line = buffer[..newline_pos].trim().to_string();
                buffer.drain(..=newline_pos);
                if line.is_empty() {
                    continue;
                }

                let value: serde_json::Value = match serde_json::from_str(&line) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                if let Some(err) = value.get("error").and_then(|e| e.as_str()) {
                    let _ = sender.send(PullProgressEvent::Error {
                        model,
                        message: err.to_string(),
                    });
                    return;
                }

                let status = value
                    .get("status")
                    .and_then(|s| s.as_str())
                    .unwrap_or("")
                    .to_string();
                let completed = value.get("completed").and_then(|v| v.as_u64());
                let total = value.get("total").and_then(|v| v.as_u64());

                let is_success = status == "success";
                let _ = sender.send(PullProgressEvent::Progress {
                    model: model.clone(),
                    status,
                    completed,
                    total,
                });
                if is_success {
                    let _ = sender.send(PullProgressEvent::Done { model });
                    return;
                }
            }
        }

        let _ = sender.send(PullProgressEvent::Done { model });
    }
}

#[derive(Debug, Deserialize)]
struct VersionResponse {
    version: String,
}

#[derive(Debug, Deserialize)]
struct TagsResponse {
    models: Vec<TagModel>,
}

#[derive(Debug, Deserialize)]
struct TagModel {
    name: String,
    size: Option<u64>,
    details: Option<TagModelDetails>,
}

#[derive(Debug, Deserialize)]
struct TagModelDetails {
    parameter_size: Option<String>,
    quantization_level: Option<String>,
}

#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    stream: bool,
    think: bool,
}

#[derive(Debug, Deserialize)]
struct ChatStreamLine {
    #[serde(default)]
    message: Option<ChatStreamMessage>,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatStreamMessage {
    #[serde(default)]
    content: String,
    /// Reasoning trace for "thinking" models (e.g. Qwen3.x). Empty/absent
    /// when `think` wasn't requested or the model doesn't support it.
    #[serde(default)]
    thinking: String,
}

#[async_trait]
impl InferenceBackend for OllamaBackend {
    fn engine_name(&self) -> &'static str {
        "ollama"
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, EngineError> {
        let url = format!("{}/api/tags", self.base_url);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| EngineError::Unreachable(url.clone(), e.to_string()))?;

        let body: TagsResponse = resp
            .json()
            .await
            .map_err(|e| EngineError::Parse(e.to_string()))?;

        Ok(body
            .models
            .into_iter()
            .map(|m| ModelInfo {
                name: m.name,
                size_bytes: m.size,
                parameter_size: m.details.as_ref().and_then(|d| d.parameter_size.clone()),
                quantization: m.details.and_then(|d| d.quantization_level),
            })
            .collect())
    }

    async fn chat_stream(
        &self,
        request_id: String,
        model: &str,
        messages: &[ChatMessage],
        options: ChatOptions,
        sender: UnboundedSender<ChatStreamEvent>,
    ) -> Result<(), EngineError> {
        let url = format!("{}/api/chat", self.base_url);
        let req_body = ChatRequest {
            model,
            messages,
            stream: true,
            think: options.think,
        };

        let resp = self
            .client
            .post(&url)
            .json(&req_body)
            .send()
            .await
            .map_err(|e| EngineError::Unreachable(url.clone(), e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(EngineError::Response(format!("{status}: {text}")));
        }

        // Ollama streams newline-delimited JSON objects (not SSE), so we
        // buffer raw bytes and split on '\n' as they arrive.
        let mut byte_stream = resp.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk) = byte_stream.next().await {
            let chunk = chunk.map_err(|e| EngineError::Unreachable(url.clone(), e.to_string()))?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(newline_pos) = buffer.find('\n') {
                let line = buffer[..newline_pos].trim().to_string();
                buffer.drain(..=newline_pos);
                if line.is_empty() {
                    continue;
                }

                let parsed: ChatStreamLine = match serde_json::from_str(&line) {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = sender.send(ChatStreamEvent::Error {
                            request_id: request_id.clone(),
                            message: format!("parse error: {e}"),
                        });
                        continue;
                    }
                };

                if let Some(err) = parsed.error {
                    let _ = sender.send(ChatStreamEvent::Error {
                        request_id: request_id.clone(),
                        message: err,
                    });
                    return Ok(());
                }

                if let Some(msg) = parsed.message {
                    if !msg.thinking.is_empty() {
                        let _ = sender.send(ChatStreamEvent::Thinking {
                            request_id: request_id.clone(),
                            content: msg.thinking,
                        });
                    }
                    if !msg.content.is_empty() {
                        let _ = sender.send(ChatStreamEvent::Token {
                            request_id: request_id.clone(),
                            content: msg.content,
                        });
                    }
                }

                if parsed.done {
                    let _ = sender.send(ChatStreamEvent::Done {
                        request_id: request_id.clone(),
                    });
                    return Ok(());
                }
            }
        }

        let _ = sender.send(ChatStreamEvent::Done { request_id });
        Ok(())
    }
}
