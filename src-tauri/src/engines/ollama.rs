use async_trait::async_trait;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::UnboundedSender;

use super::{
    ChatMessage, ChatOptions, ChatStreamEvent, EngineError, InferenceBackend, ModelInfo,
    PullProgressEvent,
};

/// Oldest engine this app's API usage is known to work against. Both things it
/// relies on -- `think` on `/api/chat` and `capabilities` on `/api/show` --
/// arrived in Ollama 0.9. Below that, thinking mode and the per-model capability
/// check silently do nothing useful.
///
/// Deliberately not raised to whatever version is current: a newer engine is
/// needed to *resolve newer model tags* from the registry, but that's a registry
/// question this check can't answer, and refusing to run on a working engine
/// would be worse than letting a pull fail with its own message.
pub const MINIMUM_VERSION: (u32, u32, u32) = (0, 9, 0);

/// Parses an Ollama version string such as `"0.23.2"`. Trailing pre-release
/// junk (`"0.24.0-rc1"`) keeps the numeric prefix.
pub fn parse_version(raw: &str) -> Option<(u32, u32, u32)> {
    let mut parts = raw.trim().trim_start_matches('v').split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch = parts
        .next()
        .map(|p| {
            p.split(|c: char| !c.is_ascii_digit())
                .next()
                .unwrap_or("0")
                .parse()
                .unwrap_or(0)
        })
        .unwrap_or(0);
    Some((major, minor, patch))
}

/// Unparseable versions count as supported: a version string this code doesn't
/// recognise is more likely a format change in a *newer* engine than an ancient
/// one, and a false alarm on every launch is worse than a missed warning.
pub fn version_is_supported(raw: &str) -> bool {
    parse_version(raw).is_none_or(|version| version >= MINIMUM_VERSION)
}

pub struct OllamaBackend {
    base_url: String,
    client: reqwest::Client,
    /// Weight-read throughput in GB/s, learned from generation this engine has
    /// actually completed here: model size times its reported tokens/sec.
    /// `None` until the first local answer finishes.
    observed_gb_per_sec: std::sync::Mutex<Option<f64>>,
}

impl OllamaBackend {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            client: super::http_client(),
            observed_gb_per_sec: std::sync::Mutex::new(None),
        }
    }

    /// The calibration figure, if anything has been generated yet.
    pub fn observed_gb_per_sec(&self) -> Option<f64> {
        *self.observed_gb_per_sec.lock().ok()?
    }

    /// Folds one completed generation into the calibration.
    ///
    /// Uses on-disk size as a stand-in for bytes-read-per-token, which is right
    /// for a dense model and an over-estimate for a mixture-of-experts one
    /// (only its active experts get read). Smoothed rather than replaced, so a
    /// single odd sample -- a thermally throttled run, a stray long pause --
    /// doesn't swing the recommendations.
    fn record_generation(&self, size_gb: f64, eval_count: u64, eval_duration_ns: u64) {
        /// A handful of tokens is mostly setup cost, so the implied throughput
        /// is noise rather than a measurement.
        const MIN_TOKENS: u64 = 16;
        /// Weight of a new sample against the running figure.
        const SMOOTHING: f64 = 0.3;
        /// Models below this size make the machine look faster than it is, and
        /// the estimates built on the result are optimistic -- the wrong
        /// direction for a gate that decides what is usable.
        ///
        /// Measured on the reference machine: a 270MB model implied 49.6 GB/s,
        /// while 2.0GB and 4.9GB models implied 36.4 and 37.7. Small weights sit
        /// largely in cache, and a small model's file is proportionally more
        /// vocabulary and embeddings, which aren't all read per token -- so
        /// file size overstates the per-token read badly at that end.
        const MIN_SIZE_GB: f64 = 1.0;

        if eval_count < MIN_TOKENS || eval_duration_ns == 0 || size_gb < MIN_SIZE_GB {
            return;
        }
        let tokens_per_sec = eval_count as f64 / (eval_duration_ns as f64 / 1e9);
        let sample = size_gb * tokens_per_sec;

        if let Ok(mut current) = self.observed_gb_per_sec.lock() {
            *current = Some(match *current {
                Some(previous) => previous * (1.0 - SMOOTHING) + sample * SMOOTHING,
                None => sample,
            });
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

    /// Whether Ollama is currently holding any loaded model in VRAM.
    ///
    /// `/api/ps` is the honest signal here: it reports what the engine
    /// actually did with the hardware, not what hardware exists. A machine can
    /// have a GPU that Ollama won't use (unsupported vendor, missing driver),
    /// and that distinction is exactly what the model recommendations hinge
    /// on. With nothing loaded the answer is unknown, and unknown is reported
    /// as `false`: over-estimating the machine is what produces
    /// recommendations it can't actually run.
    pub async fn gpu_offload_detected(&self) -> bool {
        let url = format!("{}/api/ps", self.base_url);
        let Ok(resp) = self.client.get(&url).send().await else {
            return false;
        };
        if !resp.status().is_success() {
            return false;
        }
        let Ok(value) = resp.json::<serde_json::Value>().await else {
            return false;
        };
        value
            .get("models")
            .and_then(|m| m.as_array())
            .map(|models| {
                models.iter().any(|m| {
                    m.get("size_vram").and_then(|v| v.as_u64()).unwrap_or(0) > 0
                })
            })
            .unwrap_or(false)
    }

    /// On-disk size of an installed model, for deciding how much of the RAM
    /// budget a request should reserve before it starts. Cheaper than
    /// `list_models`, which also fans out to `/api/show` per model.
    pub async fn model_size_bytes(&self, model: &str) -> Option<u64> {
        let url = format!("{}/api/tags", self.base_url);
        let resp = self.client.get(&url).send().await.ok()?;
        let body: TagsResponse = resp.json().await.ok()?;
        body.models
            .into_iter()
            .find(|m| m.name == model)
            .and_then(|m| m.size)
    }

    /// Capabilities Ollama advertises for a model, e.g. `["completion"]` or
    /// `["completion", "thinking"]`. Metadata only -- this does not load the
    /// model or run inference, so it's cheap enough to call per request.
    pub async fn capabilities(&self, model: &str) -> Result<Vec<String>, EngineError> {
        let url = format!("{}/api/show", self.base_url);
        let resp = self
            .client
            .post(&url)
            .json(&serde_json::json!({ "model": model }))
            .send()
            .await
            .map_err(|e| EngineError::Unreachable(url.clone(), e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(EngineError::Response(format!("{status}: {text}")));
        }

        let value: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| EngineError::Parse(e.to_string()))?;

        Ok(value
            .get("capabilities")
            .and_then(|c| c.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default())
    }

    /// Whether asking this model to think is allowed at all. Treats an
    /// unreachable or unparseable answer as "no": guessing wrong in that
    /// direction costs a plain response, guessing wrong the other way makes
    /// Ollama reject the whole chat request.
    pub async fn supports_thinking(&self, model: &str) -> bool {
        self.capabilities(model)
            .await
            .map(|caps| caps.iter().any(|c| c == "thinking"))
            .unwrap_or(false)
    }

    /// Forces Ollama to drop a model from memory immediately, rather than
    /// waiting out its normal keep_alive idle timeout. `keep_alive: 0` on a
    /// prompt-less generate request is Ollama's documented unload signal;
    /// the actual memory free happens a moment after this call returns.
    pub async fn unload_model(&self, model: &str) -> Result<(), EngineError> {
        let url = format!("{}/api/generate", self.base_url);
        let resp = self
            .client
            .post(&url)
            .json(&serde_json::json!({ "model": model, "keep_alive": 0 }))
            .send()
            .await
            .map_err(|e| EngineError::Unreachable(url.clone(), e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(EngineError::Response(format!("{status}: {text}")));
        }
        Ok(())
    }

    /// Single-shot (non-streaming) chat call that returns the complete
    /// response text. Used for internal decision-making calls (e.g. the
    /// router/compression step) where the caller needs to parse the whole
    /// answer at once rather than render it token by token. `force_json`
    /// asks Ollama to constrain output to valid JSON.
    pub async fn chat_once(
        &self,
        model: &str,
        messages: &[ChatMessage],
        force_json: bool,
    ) -> Result<String, EngineError> {
        let url = format!("{}/api/chat", self.base_url);
        let mut body = serde_json::json!({
            "model": model,
            "messages": messages,
            "stream": false,
            // This call is for fast internal decisions, not user-facing answers --
            // a thinking-capable model left to its default would rather spend
            // tens of seconds deliberating than emit the short JSON we asked for.
            "think": false,
        });
        if force_json {
            body["format"] = serde_json::Value::String("json".to_string());
        }

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| EngineError::Unreachable(url.clone(), e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(EngineError::Response(format!("{status}: {text}")));
        }

        let value: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| EngineError::Parse(e.to_string()))?;

        value
            .get("message")
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .map(str::to_string)
            .ok_or_else(|| EngineError::Parse("no message.content in response".to_string()))
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

        // Falling off the end of the stream is not the same as finishing. Ollama
        // marks a completed pull with a `success` status line, and the loop above
        // returns as soon as it sees one. Reaching here means the connection
        // ended first -- a dropped network, a restarted engine, a cancelled
        // request -- and reporting that as Done left a half-downloaded model
        // looking installed in the UI.
        let _ = sender.send(PullProgressEvent::Error {
            model,
            message: "ダウンロードが完了前に中断されました（接続が切れた可能性があります）".to_string(),
        });
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
    /// Left out of the payload entirely unless thinking was both requested and
    /// advertised by the model. Ollama rejects the whole request with
    /// `"<model>" does not support thinking` otherwise, and most of the
    /// catalog (gemma3, llama3.x, phi4, mistral) is non-thinking.
    #[serde(skip_serializing_if = "Option::is_none")]
    think: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct ChatStreamLine {
    #[serde(default)]
    message: Option<ChatStreamMessage>,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    error: Option<String>,
    /// Present on the final line only. Tokens generated, and the nanoseconds
    /// spent generating them -- the machine's real throughput, straight from
    /// the engine.
    #[serde(default)]
    eval_count: Option<u64>,
    #[serde(default)]
    eval_duration: Option<u64>,
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

        // /api/tags doesn't report capabilities, so ask /api/show per model.
        // Concurrent and metadata-only, and a model list is a handful of
        // entries, so this stays well inside the cost of the list refresh.
        let thinking = futures_util::future::join_all(
            body.models.iter().map(|m| self.supports_thinking(&m.name)),
        )
        .await;

        Ok(body
            .models
            .into_iter()
            .zip(thinking)
            .map(|(m, supports_thinking)| ModelInfo {
                name: m.name,
                size_bytes: m.size,
                parameter_size: m.details.as_ref().and_then(|d| d.parameter_size.clone()),
                quantization: m.details.and_then(|d| d.quantization_level),
                supports_thinking,
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
        // 「じっくり」 is a single toggle for the whole comparison, so a
        // parallel run routinely mixes thinking and non-thinking models.
        // Downgrade the ones that can't think to a plain response instead of
        // letting Ollama fail their half of the comparison outright.
        let think = if options.think && self.supports_thinking(model).await {
            Some(true)
        } else {
            None
        };
        let req_body = ChatRequest {
            model,
            messages,
            stream: true,
            think,
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
                    // Calibrate off the answer we just produced. The size
                    // lookup is one metadata call at the end of a generation
                    // that took seconds to minutes.
                    if let (Some(count), Some(duration)) = (parsed.eval_count, parsed.eval_duration) {
                        if let Some(size_bytes) = self.model_size_bytes(model).await {
                            let size_gb = size_bytes as f64 / 1024f64.powi(3);
                            self.record_generation(size_gb, count, duration);
                        }
                    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::ChatRole;

    /// These talk to a real Ollama, so they're ignored by default. Setup:
    ///   ollama serve
    ///   ollama pull llama3.2:3b
    ///   ollama pull qwen3:0.6b
    ///   cargo test --lib -- --ignored engines::ollama
    ///
    /// The two sides of the capability gate. Both are models someone would
    /// actually keep installed -- an English-only toy model was used here at
    /// first, which made the suite depend on a model nobody wants on the machine.
    const NON_THINKING_MODEL: &str = "llama3.2:3b";
    const THINKING_MODEL: &str = "qwen3:0.6b";

    /// Under the 1GB calibration floor, for the test that the floor holds.
    const TINY_MODEL: &str = "qwen3:0.6b";

    #[tokio::test]
    #[ignore]
    async fn reports_a_non_thinking_model_as_unable_to_think() {
        let ollama = OllamaBackend::default_local();
        let caps = ollama.capabilities(NON_THINKING_MODEL).await.unwrap();
        assert!(
            !caps.iter().any(|c| c == "thinking"),
            "expected no thinking capability, got {caps:?}"
        );
        assert!(!ollama.supports_thinking(NON_THINKING_MODEL).await);
    }

    /// The other half of the capability gate, and the one that gives the
    /// downgrade test its meaning: a gate that simply never asked for thinking
    /// would satisfy `thinking_request_downgrades_instead_of_failing` just as
    /// well. This checks the gate discriminates rather than just suppresses.
    #[tokio::test]
    #[ignore]
    async fn a_thinking_capable_model_still_gets_its_trace() {
        let ollama = OllamaBackend::default_local();
        assert!(
            ollama.supports_thinking(THINKING_MODEL).await,
            "{THINKING_MODEL} should advertise thinking"
        );

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        ollama
            .chat_stream(
                "thinking".to_string(),
                THINKING_MODEL,
                &[ChatMessage {
                    role: ChatRole::User,
                    content: "2+2は?".to_string(),
                }],
                ChatOptions { think: true },
                tx,
            )
            .await
            .expect("chat_stream");

        let mut thinking = String::new();
        let mut errors = Vec::new();
        while let Some(event) = rx.recv().await {
            match event {
                ChatStreamEvent::Thinking { content, .. } => thinking.push_str(&content),
                ChatStreamEvent::Error { message, .. } => errors.push(message),
                _ => {}
            }
        }

        assert!(errors.is_empty(), "expected no errors, got {errors:?}");
        assert!(!thinking.is_empty(), "expected a reasoning trace to come through");
    }

    async fn generate_once(ollama: &OllamaBackend, model: &str) {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        ollama
            .chat_stream(
                "calibration".to_string(),
                model,
                &[ChatMessage {
                    role: ChatRole::User,
                    content: "Write a paragraph about the sea.".to_string(),
                }],
                ChatOptions { think: false },
                tx,
            )
            .await
            .expect("chat_stream");
        while rx.recv().await.is_some() {}
    }

    /// The calibration path, end to end: generate something real and check the
    /// engine learned a throughput figure from it. This is the measurement the
    /// model recommendations rest on, and it replaced a synthetic memory
    /// benchmark that reported anywhere from 3.4 to 80 GB/s on one machine
    /// depending on optimisation level.
    #[tokio::test]
    #[ignore]
    async fn learns_its_throughput_from_a_real_generation() {
        let ollama = OllamaBackend::default_local();
        assert!(
            ollama.observed_gb_per_sec().is_none(),
            "a fresh backend should admit it hasn't measured anything"
        );

        generate_once(&ollama, NON_THINKING_MODEL).await;

        let observed = ollama
            .observed_gb_per_sec()
            .expect("calibrated after one answer");
        // Loose bounds: enough to catch a zero, a unit error, or a figure no
        // real hardware could produce.
        assert!(observed > 0.5 && observed < 500.0, "implausible: {observed} GB/s");
        println!("observed weight-read throughput: {observed:.1} GB/s");
    }

    /// Tiny models are excluded from calibration on purpose.
    ///
    /// Measured on the reference machine: a 270MB model implied 49.6 GB/s where
    /// 2.0GB and 4.9GB models implied 36.4 and 37.7. Calibrating off the small
    /// one made every estimate optimistic by up to 27% -- and optimistic is the
    /// harmful direction, because the speed gate then recommends models that
    /// turn out to be slower than promised.
    #[tokio::test]
    #[ignore]
    async fn a_tiny_model_does_not_get_to_set_the_calibration() {
        let ollama = OllamaBackend::default_local();
        generate_once(&ollama, TINY_MODEL).await;
        assert!(
            ollama.observed_gb_per_sec().is_none(),
            "a sub-1GB model should not be treated as representative"
        );
    }

    /// The regression this pins down: asking a model that can't think to think
    /// used to put `think: true` on the wire, and Ollama answers that with
    /// `"<model>" does not support thinking` -- failing the model's whole turn
    /// instead of just leaving out the reasoning trace. Since 「じっくり」 is
    /// one toggle for every selected model, that took out each non-thinking
    /// model in a parallel comparison.
    #[tokio::test]
    #[ignore]
    async fn thinking_request_downgrades_instead_of_failing() {
        let ollama = OllamaBackend::default_local();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

        ollama
            .chat_stream(
                "test-request".to_string(),
                NON_THINKING_MODEL,
                &[ChatMessage {
                    role: ChatRole::User,
                    content: "Say OK".to_string(),
                }],
                ChatOptions { think: true },
                tx,
            )
            .await
            .expect("chat_stream should succeed for a non-thinking model");

        let mut errors = Vec::new();
        let mut tokens = 0;
        while let Some(event) = rx.recv().await {
            match event {
                ChatStreamEvent::Error { message, .. } => errors.push(message),
                ChatStreamEvent::Token { .. } => tokens += 1,
                _ => {}
            }
        }

        assert!(errors.is_empty(), "expected no errors, got {errors:?}");
        assert!(tokens > 0, "expected at least one token");
    }
}
