pub mod anthropic;
pub mod gemini;
pub mod ollama;
pub mod openai_compat;

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
    /// Reasoning-model "thinking" trace, delivered separately from the
    /// final answer so the UI can show it as a distinct (collapsible)
    /// section instead of leaving the user staring at a blank bubble.
    Thinking {
        request_id: String,
        content: String,
    },
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

/// Per-request knobs that apply across backends. Kept as a struct (rather
/// than growing the `chat_stream` signature) so new options don't require
/// touching every implementor.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ChatOptions {
    /// Ask reasoning-capable models to emit a thinking trace before the
    /// final answer. Off by default: for a "thinking" model this can mean
    /// thousands of extra tokens (tens of seconds) before anything is
    /// visible, so callers opt in when they want the deeper reasoning.
    #[serde(default)]
    pub think: bool,
}

/// Progress for a model download, forwarded to the frontend as a
/// `pull-progress` Tauri event, tagged by model name rather than a
/// request id since pulls aren't tied to a chat turn.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PullProgressEvent {
    Progress {
        model: String,
        status: String,
        completed: Option<u64>,
        total: Option<u64>,
    },
    Done {
        model: String,
    },
    Error {
        model: String,
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
        options: ChatOptions,
        sender: UnboundedSender<ChatStreamEvent>,
    ) -> Result<(), EngineError>;
}
