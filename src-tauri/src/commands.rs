use std::sync::Arc;
use tauri::{AppHandle, Emitter, State};
use tokio::sync::mpsc;

use crate::credentials;
use crate::engines::anthropic::AnthropicBackend;
use crate::engines::gemini::GeminiBackend;
use crate::engines::ollama::OllamaBackend;
use crate::engines::openai_compat::OpenAiCompatBackend;
use crate::engines::{ChatMessage, ChatOptions, ChatStreamEvent, InferenceBackend, ModelInfo};

pub struct AppState {
    pub ollama: Arc<OllamaBackend>,
}

#[derive(serde::Serialize)]
pub struct ProviderInfo {
    pub id: String,
    pub label: String,
    pub default_model: String,
    pub configured: bool,
}

/// Static registry of online providers. `default_model` is a best-effort
/// current flagship id -- these move fast, so it's editable per-provider
/// from the settings UI rather than hardcoded deeper into the app.
const ONLINE_PROVIDERS: &[(&str, &str, &str)] = &[
    ("anthropic", "Claude", "claude-opus-4-8"),
    ("openai", "ChatGPT", "gpt-5.6"),
    ("gemini", "Gemini", "gemini-3.6-flash"),
    ("xai", "Grok", "grok-4.3"),
];

#[tauri::command]
pub fn list_providers() -> Vec<ProviderInfo> {
    ONLINE_PROVIDERS
        .iter()
        .map(|(id, label, default_model)| ProviderInfo {
            id: id.to_string(),
            label: label.to_string(),
            default_model: default_model.to_string(),
            configured: credentials::has_key(id),
        })
        .collect()
}

#[tauri::command]
pub fn save_api_key(provider: String, key: String) -> Result<(), String> {
    credentials::save_key(&provider, &key)
}

#[tauri::command]
pub fn clear_api_key(provider: String) -> Result<(), String> {
    credentials::clear_key(&provider)
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
    provider: String,
    model: String,
    messages: Vec<ChatMessage>,
    think: Option<bool>,
) -> Result<(), String> {
    let (tx, mut rx) = mpsc::unbounded_channel::<ChatStreamEvent>();
    let options = ChatOptions {
        think: think.unwrap_or(false),
    };

    let forward_app = app.clone();
    tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            let _ = forward_app.emit("chat-stream", &event);
        }
    });

    match provider.as_str() {
        "ollama" => {
            let backend = state.ollama.clone();
            backend
                .chat_stream(request_id, &model, &messages, options, tx)
                .await
                .map_err(|e| e.to_string())
        }
        "anthropic" => {
            let key = credentials::get_key("anthropic")
                .ok_or_else(|| "Anthropic APIキーが未設定です".to_string())?;
            AnthropicBackend::new(key)
                .chat_stream(request_id, &model, &messages, options, tx)
                .await
                .map_err(|e| e.to_string())
        }
        "openai" => {
            let key = credentials::get_key("openai")
                .ok_or_else(|| "OpenAI APIキーが未設定です".to_string())?;
            OpenAiCompatBackend::new("openai", "https://api.openai.com/v1", key, "gpt-5.6")
                .chat_stream(request_id, &model, &messages, options, tx)
                .await
                .map_err(|e| e.to_string())
        }
        "xai" => {
            let key = credentials::get_key("xai").ok_or_else(|| "xAI APIキーが未設定です".to_string())?;
            OpenAiCompatBackend::new("xai", "https://api.x.ai/v1", key, "grok-4.3")
                .chat_stream(request_id, &model, &messages, options, tx)
                .await
                .map_err(|e| e.to_string())
        }
        "gemini" => {
            let key = credentials::get_key("gemini")
                .ok_or_else(|| "Gemini APIキーが未設定です".to_string())?;
            GeminiBackend::new(key)
                .chat_stream(request_id, &model, &messages, options, tx)
                .await
                .map_err(|e| e.to_string())
        }
        other => Err(format!("unknown provider: {other}")),
    }
}
