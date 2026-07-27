use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use tauri::{AppHandle, Emitter, State};
use tokio::sync::{mpsc, oneshot, Semaphore};

use crate::catalog::{self, CatalogEntry, HardwareProfile};
use crate::credentials;
use crate::engines::anthropic::AnthropicBackend;
use crate::engines::gemini::GeminiBackend;
use crate::engines::ollama::{self, OllamaBackend};
use crate::engines::openai_compat::OpenAiCompatBackend;
use crate::engines::{ChatMessage, ChatOptions, ChatStreamEvent, InferenceBackend, ModelInfo};
use crate::router::{self, RouterDecision};
use crate::updater;

/// Granularity of the RAM budget, in permits per gigabyte. Fine enough that a
/// model's reservation is close to its real footprint, coarse enough that the
/// permit counts stay small.
const PERMITS_PER_GB: f64 = 8.0;

pub struct AppState {
    pub ollama: Arc<OllamaBackend>,
    /// The RAM budget for loaded local models, expressed as semaphore permits.
    ///
    /// Measurements on the CPU-only reference machine showed that running two
    /// models at once is *not* slower in wall-clock terms -- generation
    /// throughput held up and the pair finished sooner than back-to-back runs.
    /// So parallelism isn't the problem; committing more RAM than the machine
    /// has is. Requests reserve permits proportional to their model's expected
    /// footprint and queue when the budget is spoken for, which lets small
    /// models run side by side while stopping several large ones from making
    /// Ollama evict and reload in a loop.
    pub local_slots: Arc<Semaphore>,
    pub total_local_permits: u32,
    /// When set, every local request reserves the entire budget, which turns the
    /// admission control above into strict one-at-a-time execution.
    ///
    /// Off by default because measurement didn't support serialising: two models
    /// at once finished sooner than back to back. It exists for the case the
    /// numbers don't capture -- wanting the *first* answer as fast as possible
    /// rather than all of them as soon as possible.
    pub serialize_local: AtomicBool,
    /// Cancellation channels for in-flight requests, keyed by request id.
    pub cancels: Mutex<HashMap<String, oneshot::Sender<()>>>,
}

impl AppState {
    pub fn new(ollama: Arc<OllamaBackend>, ram_budget_gb: f64) -> Self {
        // At least one permit's worth, or nothing could ever run.
        let permits = ((ram_budget_gb * PERMITS_PER_GB).floor() as u32).max(1);
        Self {
            ollama,
            local_slots: Arc::new(Semaphore::new(permits as usize)),
            total_local_permits: permits,
            serialize_local: AtomicBool::new(false),
            cancels: Mutex::new(HashMap::new()),
        }
    }

    /// How much of the budget a model should reserve.
    async fn permits_for(&self, model: &str) -> u32 {
        if self.serialize_local.load(Ordering::Relaxed) {
            return self.total_local_permits;
        }
        let Some(size_bytes) = self.ollama.model_size_bytes(model).await else {
            // Unknown size: reserve a single permit rather than guessing high
            // and needlessly serialising everything.
            return 1;
        };
        permits_for_size_gb(
            size_bytes as f64 / 1024f64.powi(3),
            self.total_local_permits,
        )
    }

    /// The permits a model needs, and whether they're free right now.
    ///
    /// Split from the acquire so the caller can announce the wait *before*
    /// blocking on it: a model that silently sits there with no output reads as
    /// broken rather than as queued.
    async fn plan_reservation(&self, model: &str) -> (u32, bool) {
        let needed = self.permits_for(model).await;
        let immediate = self.local_slots.available_permits() >= needed as usize;
        (needed, immediate)
    }
}

/// Permits a model of this on-disk size should reserve.
///
/// Clamped to the whole budget at the top: a model larger than the budget still
/// has to be allowed to run on its own, or it would wait forever for permits
/// that do not exist. Clamped to 1 at the bottom so every request reserves
/// something and the budget can actually fill up.
fn permits_for_size_gb(size_gb: f64, total_permits: u32) -> u32 {
    let needed = (catalog::resident_gb(size_gb) * PERMITS_PER_GB).ceil() as u32;
    needed.clamp(1, total_permits)
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

#[derive(serde::Serialize)]
pub struct EngineVersion {
    pub version: String,
    pub supported: bool,
    pub minimum: String,
    /// Newest published release, when it could be looked up. `None` offline.
    pub latest: Option<String>,
    pub update_available: bool,
    /// How an in-app update would be performed, if it can be at all.
    pub package_manager: Option<updater::PackageManager>,
    pub download_page: String,
}

/// The engine's version alongside whether it's new enough for the API surface
/// this app uses. Previously the version was fetched and thrown away, so an
/// engine too old for `think` or for capability detection failed silently at
/// each call site instead of once, visibly, at startup.
#[tauri::command]
pub async fn ollama_version(state: State<'_, AppState>) -> Result<EngineVersion, String> {
    let version = state.ollama.version().await.map_err(|e| e.to_string())?;
    let (major, minor, patch) = ollama::MINIMUM_VERSION;

    // Neither of these should be able to fail the whole call: the installed
    // version is the useful part, and being offline shouldn't hide it.
    let latest = updater::latest_release(&crate::engines::http_client()).await.ok();
    let update_available = match (&latest, ollama::parse_version(&version)) {
        (Some(latest), Some(installed)) => ollama::parse_version(latest)
            .is_some_and(|newest| newest > installed),
        _ => false,
    };

    Ok(EngineVersion {
        supported: ollama::version_is_supported(&version),
        version,
        minimum: format!("{major}.{minor}.{patch}"),
        latest,
        update_available,
        package_manager: updater::detect_package_manager().await,
        download_page: updater::DOWNLOAD_PAGE_URL.to_string(),
    })
}

/// Upgrades the Ollama engine through the OS package manager, streaming its
/// output as `ollama-update` events.
///
/// Refuses while local work is in flight. The upgrade stops the engine, so
/// running it mid-generation would kill answers the user is waiting on -- and the
/// RAM budget already knows whether anything is running.
#[tauri::command]
pub async fn update_ollama(app: AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    if state.local_slots.available_permits() < state.total_local_permits as usize {
        return Err(
            "ローカルモデルの実行中です。完了するか中断してから更新してください。".to_string(),
        );
    }

    let manager = updater::detect_package_manager()
        .await
        .ok_or_else(|| {
            format!(
                "このマシンで自動更新に使えるパッケージマネージャが見つかりませんでした。{} から手動で更新してください。",
                updater::DOWNLOAD_PAGE_URL
            )
        })?;

    let (tx, mut rx) = mpsc::unbounded_channel();
    let forward_app = app.clone();
    tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            let _ = forward_app.emit("ollama-update", &event);
        }
    });

    updater::upgrade(manager, tx).await;
    Ok(())
}

#[tauri::command]
pub async fn unload_model(state: State<'_, AppState>, model: String) -> Result<(), String> {
    state.ollama.unload_model(&model).await.map_err(|e| e.to_string())
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

    let (cancel_tx, cancel_rx) = oneshot::channel::<()>();
    state
        .cancels
        .lock()
        .map_err(|e| e.to_string())?
        .insert(request_id.clone(), cancel_tx);

    // Only local models draw on the RAM budget. Online ones cost network and
    // money, not memory, so they never wait behind anything.
    let _permit = if provider == "ollama" {
        let (needed, immediate) = state.plan_reservation(&model).await;
        if !immediate {
            let _ = app.emit(
                "chat-stream",
                &ChatStreamEvent::Queued {
                    request_id: request_id.clone(),
                },
            );
        }
        Some(
            state
                .local_slots
                .clone()
                .acquire_many_owned(needed)
                .await
                .map_err(|e| e.to_string())?,
        )
    } else {
        None
    };

    // Pull what the streaming future needs out of `state` first, so the guard
    // itself isn't moved into the future and stays usable for cleanup below.
    let ollama = state.ollama.clone();
    let stream_id = request_id.clone();
    let work = async move {
        match provider.as_str() {
            "ollama" => {
                ollama
                    .chat_stream(stream_id, &model, &messages, options, tx)
                    .await
                    .map_err(|e| e.to_string())
            }
            "anthropic" => {
                let key = credentials::get_key("anthropic")
                    .ok_or_else(|| "Anthropic APIキーが未設定です".to_string())?;
                AnthropicBackend::new(key)
                    .chat_stream(stream_id, &model, &messages, options, tx)
                    .await
                    .map_err(|e| e.to_string())
            }
            "openai" => {
                let key = credentials::get_key("openai")
                    .ok_or_else(|| "OpenAI APIキーが未設定です".to_string())?;
                OpenAiCompatBackend::new("openai", "https://api.openai.com/v1", key, "gpt-5.6")
                    .chat_stream(stream_id, &model, &messages, options, tx)
                    .await
                    .map_err(|e| e.to_string())
            }
            "xai" => {
                let key =
                    credentials::get_key("xai").ok_or_else(|| "xAI APIキーが未設定です".to_string())?;
                OpenAiCompatBackend::new("xai", "https://api.x.ai/v1", key, "grok-4.3")
                    .chat_stream(stream_id, &model, &messages, options, tx)
                    .await
                    .map_err(|e| e.to_string())
            }
            "gemini" => {
                let key = credentials::get_key("gemini")
                    .ok_or_else(|| "Gemini APIキーが未設定です".to_string())?;
                GeminiBackend::new(key)
                    .chat_stream(stream_id, &model, &messages, options, tx)
                    .await
                    .map_err(|e| e.to_string())
            }
            other => Err(format!("unknown provider: {other}")),
        }
    };

    let cancelled = async move {
        // A dropped sender means the request finished and cleaned itself up,
        // not that the user asked to stop. Never resolve in that case, so the
        // work branch is the one that wins.
        if cancel_rx.await.is_err() {
            std::future::pending::<()>().await;
        }
    };

    // Dropping `work` closes the connection to the engine mid-generation,
    // which is what actually gives the machine back: on a CPU-only box a large
    // model otherwise keeps every core busy for minutes after the user has
    // stopped caring about the answer.
    let outcome = tokio::select! {
        res = work => res,
        () = cancelled => {
            let _ = app.emit(
                "chat-stream",
                &ChatStreamEvent::Cancelled {
                    request_id: request_id.clone(),
                },
            );
            Ok(())
        }
    };

    if let Ok(mut cancels) = state.cancels.lock() {
        cancels.remove(&request_id);
    }
    outcome
}

/// Turns strict one-at-a-time local execution on or off. Takes effect for
/// requests started after the call; anything already running keeps its slot.
#[tauri::command]
pub fn set_serialize_local(state: State<'_, AppState>, serialize: bool) {
    state.serialize_local.store(serialize, Ordering::Relaxed);
}

/// Stops an in-flight request. The reply keeps whatever text already arrived --
/// a partial answer is usually why the user hit stop in the first place.
#[tauri::command]
pub fn cancel_chat(state: State<'_, AppState>, request_id: String) -> Result<(), String> {
    let sender = state
        .cancels
        .lock()
        .map_err(|e| e.to_string())?
        .remove(&request_id);
    if let Some(tx) = sender {
        let _ = tx.send(());
    }
    Ok(())
}

/// Whether the engine is offloading to a GPU has to be asked of the engine,
/// not of the OS -- see `OllamaBackend::gpu_offload_detected`.
#[tauri::command]
pub async fn detect_hardware(state: State<'_, AppState>) -> Result<HardwareProfile, String> {
    let accelerated = state.ollama.gpu_offload_detected().await;
    Ok(catalog::detect_hardware(
        accelerated,
        state.ollama.observed_gb_per_sec(),
    ))
}

#[tauri::command]
pub async fn recommend_models(state: State<'_, AppState>) -> Result<Vec<CatalogEntry>, String> {
    let accelerated = state.ollama.gpu_offload_detected().await;
    let profile = catalog::detect_hardware(accelerated, state.ollama.observed_gb_per_sec());
    Ok(catalog::recommend(&profile, 10))
}

/// Streams `ollama pull` progress as `pull-progress` events. Returns once
/// the pull finishes or errors, mirroring the fire-and-forget shape of
/// `send_chat` -- the frontend listens for the events, not the return value.
#[tauri::command]
pub async fn pull_model(app: AppHandle, state: State<'_, AppState>, model: String) -> Result<(), String> {
    let (tx, mut rx) = mpsc::unbounded_channel();

    let forward_app = app.clone();
    tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            let _ = forward_app.emit("pull-progress", &event);
        }
    });

    state.ollama.clone().pull_model(model, tx).await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference machine's budget: 31.7GB installed, 70% of it.
    const REFERENCE_BUDGET_GB: f64 = 22.2;
    /// A model these tests can count on being installed, and its on-disk size:
    ///   ollama pull llama3.2:3b
    const TEST_MODEL: &str = "llama3.2:3b";
    const TEST_MODEL_GB: f64 = 2.0;

    fn state_with_budget(budget_gb: f64) -> AppState {
        AppState::new(Arc::new(OllamaBackend::default_local()), budget_gb)
    }

    #[test]
    fn a_models_reservation_tracks_its_expected_footprint() {
        let total = (REFERENCE_BUDGET_GB * PERMITS_PER_GB) as u32;
        let small = permits_for_size_gb(TEST_MODEL_GB, total);
        let large = permits_for_size_gb(8.0, total);
        assert!(small < large);
        // 8GB on disk is ~12.8GB resident, i.e. over half of a 22.2GB budget.
        assert!(large as f64 > total as f64 * 0.5);
    }

    /// A model too big for the budget still has to be runnable on its own.
    /// Reserving more permits than exist would park it forever.
    #[test]
    fn a_model_larger_than_the_budget_still_fits_exactly_once() {
        let total = (2.0 * PERMITS_PER_GB) as u32;
        assert_eq!(permits_for_size_gb(500.0, total), total);
    }

    #[test]
    fn every_request_reserves_at_least_something() {
        let total = (REFERENCE_BUDGET_GB * PERMITS_PER_GB) as u32;
        assert_eq!(permits_for_size_gb(0.0, total), 1);
    }

    /// Budgets are never zero-permit, or the first request would deadlock.
    #[test]
    fn a_budget_too_small_to_measure_still_admits_one_request() {
        assert_eq!(state_with_budget(0.0).total_local_permits, 1);
    }

    /// Serialising reuses the budget rather than adding a second mechanism: a
    /// request that reserves everything leaves nothing for a second one.
    #[tokio::test]
    #[ignore]
    async fn serialising_makes_one_request_consume_the_whole_budget() {
        let state = state_with_budget(REFERENCE_BUDGET_GB);
        let (parallel_share, _) = state.plan_reservation(TEST_MODEL).await;
        assert!(
            parallel_share < state.total_local_permits,
            "a small model should leave room for others by default"
        );

        state.serialize_local.store(true, Ordering::Relaxed);
        let (whole_budget, _) = state.plan_reservation(TEST_MODEL).await;
        assert_eq!(whole_budget, state.total_local_permits);

        let _held = state
            .local_slots
            .clone()
            .acquire_many_owned(whole_budget)
            .await
            .expect("reservation");
        let (_, immediate) = state.plan_reservation(TEST_MODEL).await;
        assert!(!immediate, "a second request must wait while serialising");
    }

    /// The queue actually firing. Needs a live Ollama with the small model:
    ///   ollama pull llama3.2:3b
    ///   cargo test --lib -- --ignored queues_a_second_model
    ///
    /// Budget is set so exactly one copy of the model fits, which is what the
    /// reference machine can't demonstrate on its own -- 22.2GB of budget
    /// swallows every model small enough to have been installed for testing.
    #[tokio::test]
    #[ignore]
    async fn queues_a_second_model_when_the_budget_is_already_committed() {
        let one_model_only = catalog::resident_gb(TEST_MODEL_GB);
        let state = state_with_budget(one_model_only);

        let (needed, immediate) = state.plan_reservation(TEST_MODEL).await;
        assert!(immediate, "the first request should not have to wait");
        assert!(needed > 1, "expected a real reservation, got {needed}");

        let held = state
            .local_slots
            .clone()
            .acquire_many_owned(needed)
            .await
            .expect("first reservation");

        // With the budget committed, a second request has to be announced as
        // queued rather than started.
        let (_, immediate_again) = state.plan_reservation(TEST_MODEL).await;
        assert!(!immediate_again, "second request should have to queue");

        // And it must actually get through once the first one finishes, rather
        // than waiting on permits that never come back.
        drop(held);
        let waited = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            state.local_slots.clone().acquire_many_owned(needed),
        )
        .await
        .expect("releasing the first reservation should unblock the second");
        assert!(waited.is_ok());
    }

    /// Upgrading the engine stops it, which would kill an answer the user is
    /// waiting on -- so `update_ollama` refuses while local work is in flight,
    /// reading that from the same budget the admission control uses rather than
    /// tracking it a second time. This pins the signal it relies on.
    #[tokio::test]
    #[ignore]
    async fn an_update_can_tell_when_local_work_is_in_flight() {
        let state = state_with_budget(REFERENCE_BUDGET_GB);
        assert_eq!(
            state.local_slots.available_permits(),
            state.total_local_permits as usize,
            "an idle machine should show the whole budget free"
        );

        let (needed, _) = state.plan_reservation(TEST_MODEL).await;
        let _held = state
            .local_slots
            .clone()
            .acquire_many_owned(needed)
            .await
            .expect("reservation");

        assert!(
            state.local_slots.available_permits() < state.total_local_permits as usize,
            "a running model has to be visible to the update guard"
        );
    }

    /// Online providers cost network and money, not memory, so they must not
    /// be held up behind a local model that is hogging the budget.
    #[tokio::test]
    #[ignore]
    async fn a_committed_budget_does_not_hold_up_online_providers() {
        let state = state_with_budget(catalog::resident_gb(TEST_MODEL_GB));
        let (needed, _) = state.plan_reservation(TEST_MODEL).await;
        let _held = state
            .local_slots
            .clone()
            .acquire_many_owned(needed)
            .await
            .expect("reservation");

        // `send_chat` only reserves for provider == "ollama"; nothing about the
        // online path consults the budget. Assert the budget really is empty,
        // so this stays a live check rather than a comment.
        assert_eq!(state.local_slots.available_permits(), 0);
    }
}

/// Asks the parent local model which online providers are worth calling
/// for this turn, and for a compressed version of the context to send
/// them. Errors (including a parent model that didn't return valid JSON)
/// are surfaced as-is -- the frontend decides whether to fall back to an
/// uncompressed direct send rather than silently guessing here.
#[tauri::command]
pub async fn route_and_compress(
    state: State<'_, AppState>,
    parent_model: String,
    candidate_providers: Vec<String>,
    history: Vec<ChatMessage>,
    new_message: String,
) -> Result<RouterDecision, String> {
    // The parent model occupies RAM like any other, so it queues on the same
    // budget rather than loading on top of whatever is already resident.
    let (needed, _) = state.plan_reservation(&parent_model).await;
    let _permit = state
        .local_slots
        .clone()
        .acquire_many_owned(needed)
        .await
        .map_err(|e| e.to_string())?;

    router::route_and_compress(
        &state.ollama,
        &parent_model,
        &candidate_providers,
        &history,
        &new_message,
    )
    .await
    .map_err(|e| e.to_string())
}
