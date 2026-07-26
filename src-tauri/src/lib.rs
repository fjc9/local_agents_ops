mod catalog;
mod commands;
mod credentials;
mod engines;
mod router;

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
            commands::unload_model,
            commands::send_chat,
            commands::list_providers,
            commands::save_api_key,
            commands::clear_api_key,
            commands::detect_hardware,
            commands::recommend_models,
            commands::pull_model,
            commands::route_and_compress,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
