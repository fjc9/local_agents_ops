use std::sync::Arc;
use tauri::{AppHandle, Emitter, State};
use tokio::sync::mpsc;

use crate::engines::ollama::OllamaBackend;
use crate::engines::{ChatMessage, ChatStreamEvent, InferenceBackend, ModelInfo};

pub struct AppState {
    pub ollama: Arc<OllamaBackend>,
}

#[tauri::command]
pub async fn list_ollama_models(state: State<'_, AppState>) -> Result<Vec<ModelInfo>, String> {
    state.ollama.list_models().await.map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn ollama_version(state: State<'_, AppState>) -> Result<String, String> {
    state.ollama.version().await.map_err(|e| e.to_string())
}

/// Streams a chat completion. Tokens are pushed to the frontend as
/// `chat-stream` events as they arrive; the returned Result only signals
/// whether the request was sent successfully, not the content itself.
#[tauri::command]
pub async fn send_chat(
    app: AppHandle,
    state: State<'_, AppState>,
    request_id: String,
    model: String,
    messages: Vec<ChatMessage>,
) -> Result<(), String> {
    let (tx, mut rx) = mpsc::unbounded_channel::<ChatStreamEvent>();
    let backend = state.ollama.clone();

    let forward_app = app.clone();
    tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            let _ = forward_app.emit("chat-stream", &event);
        }
    });

    backend
        .chat_stream(request_id, &model, &messages, tx)
        .await
        .map_err(|e| e.to_string())
}
