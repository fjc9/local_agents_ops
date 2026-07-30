mod catalog;
mod commands;
mod credentials;
mod engines;
mod router;
mod updater;

use std::sync::Arc;

use commands::AppState;
use engines::ollama::OllamaBackend;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let ram_budget_gb = catalog::total_ram_gb() * catalog::RAM_BUDGET_FRACTION;

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(AppState::new(
            Arc::new(OllamaBackend::default_local()),
            ram_budget_gb,
        ))
        .invoke_handler(tauri::generate_handler![
            commands::list_ollama_models,
            commands::ollama_version,
            commands::update_ollama,
            commands::unload_model,
            commands::send_chat,
            commands::cancel_chat,
            commands::set_serialize_local,
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
