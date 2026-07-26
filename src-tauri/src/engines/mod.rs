pub mod ollama;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::UnboundedSender;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub size_bytes: Option<u64>,
    pub parameter_size: Option<String>,
    pub quantization: Option<String>,
}

/// One increment of a streaming chat response, forwarded to the frontend
/// via a Tauri event as it arrives. `request_id` lets the UI route chunks
/// to the right chat panel when several backends are streaming at once.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatStreamEvent {
    Token {
        request_id: String,
        content: String,
    },
    Done {
        request_id: String,
    },
    Error {
        request_id: String,
        message: String,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("could not reach engine at {0}: {1}")]
    Unreachable(String, String),
    #[error("engine returned an error: {0}")]
    Response(String),
    #[error("failed to parse engine response: {0}")]
    Parse(String),
}

/// Common surface every local inference engine (Ollama, LM Studio, ...)
/// must implement so the rest of the app never branches on which engine
/// a given model is actually running on.
#[async_trait]
pub trait InferenceBackend: Send + Sync {
    /// Human-readable identifier for this backend, e.g. "ollama".
    fn engine_name(&self) -> &'static str;

    /// Models this backend currently has available to serve.
    async fn list_models(&self) -> Result<Vec<ModelInfo>, EngineError>;

    /// Stream a chat completion. Chunks are pushed onto `sender` as they
    /// arrive rather than returned, so the caller can forward them to the
    /// frontend as Tauri events without buffering the whole response.
    async fn chat_stream(
        &self,
        request_id: String,
        model: &str,
        messages: &[ChatMessage],
        sender: UnboundedSender<ChatStreamEvent>,
    ) -> Result<(), EngineError>;
}
