mod commands;
mod engines;

use std::sync::Arc;

use commands::AppState;
use engines::ollama::OllamaBackend;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(AppState {
            ollama: Arc::new(OllamaBackend::default_local()),
        })
        .invoke_handler(tauri::generate_handler![
            commands::list_ollama_models,
            commands::ollama_version,
            commands::send_chat,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
