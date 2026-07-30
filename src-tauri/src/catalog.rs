use serde::Serialize;
use sysinfo::System;

#[derive(Debug, Clone, Serialize)]
pub struct HardwareProfile {
    pub os: String,
    pub total_ram_gb: f64,
    pub logical_cores: usize,
    /// Whether Ollama was observed actually placing weights in VRAM. Unknown
    /// counts as false -- see `recommend`.
    pub accelerated: bool,
    /// Effective weight-read throughput, in GB/s, derived from generation this
    /// engine has actually done on this machine: model size times its measured
    /// tokens/sec.
    ///
    /// Deliberately not a microbenchmark. A synthetic memory benchmark was
    /// tried first and abandoned: the same code reported 3.4 GB/s unoptimised
    /// and 45 GB/s optimised, and a memcpy variant reported 26 vs 80 GB/s --
    /// above this machine's theoretical peak. An app that recommends different
    /// models under `tauri dev` than under `tauri build` is worse than one that
    /// admits it doesn't know yet, which is what `None` means here.
    pub observed_gb_per_sec: Option<f64>,
}

/// Loaded model weights need meaningfully more RAM than the on-disk file
/// (KV cache, context buffers, runtime overhead) -- this is a rough
/// multiplier, not a precise per-context-length estimate. Conservative for
/// MoE models, whose KV cache scales with active rather than total size.
pub const RUNTIME_OVERHEAD_MULTIPLIER: f64 = 1.6;
/// Leave headroom for the OS, this app, and other running models.
pub const RAM_BUDGET_FRACTION: f64 = 0.7;

/// Below this, a model is not worth recommending on a machine with no GPU
/// offload: a paragraph-length answer takes minutes and the comparison feature
/// stops being usable.
const MIN_USABLE_TOKENS_PER_SEC: f64 = 5.0;

/// However slow the machine turns out to be, always offer at least this many
/// models that fit in RAM. An empty catalog is a dead end -- the user can't
/// even install the small model that would work -- so the speed gate trims the
/// list, it never empties it.
const MIN_RECOMMENDATIONS: usize = 3;

pub fn total_ram_gb() -> f64 {
    let mut sys = System::new();
    sys.refresh_memory();
    sys.total_memory() as f64 / 1024f64.powi(3)
}

pub fn detect_hardware(accelerated: bool, observed_gb_per_sec: Option<f64>) -> HardwareProfile {
    HardwareProfile {
        os: std::env::consts::OS.to_string(),
        total_ram_gb: total_ram_gb(),
        logical_cores: std::thread::available_parallelism()
            .map(|v| v.get())
            .unwrap_or(1),
        accelerated,
        observed_gb_per_sec,
    }
}

/// RAM this app is willing to see committed to loaded models at once.
pub fn ram_budget_gb(profile: &HardwareProfile) -> f64 {
    profile.total_ram_gb * RAM_BUDGET_FRACTION
}

/// What a model is expected to occupy once loaded, including KV cache and
/// runtime buffers.
pub fn resident_gb(total_size_gb: f64) -> f64 {
    total_size_gb * RUNTIME_OVERHEAD_MULTIPLIER
}

#[derive(Debug, Clone, Serialize)]
pub struct CatalogEntry {
    pub tag: String,
    pub origin: String,
    pub label: String,
    pub size_gb: f64,
    /// Weights actually read per token. Equal to `size_gb` for a dense model;
    /// far smaller for a mixture-of-experts one, which is why a 19GB MoE can
    /// generate faster than a 9GB dense model while needing more RAM.
    pub active_size_gb: f64,
    pub role: String,
    pub description: String,
    /// Estimated generation speed on the current machine. `None` when a GPU is
    /// doing the work, since a bandwidth-derived guess would be meaningless.
    pub est_tokens_per_sec: Option<f64>,
}

fn full_catalog() -> Vec<CatalogEntry> {
    fn e(
        tag: &str,
        origin: &str,
        label: &str,
        size_gb: f64,
        active_size_gb: f64,
        role: &str,
        description: &str,
    ) -> CatalogEntry {
        CatalogEntry {
            tag: tag.to_string(),
            origin: origin.to_string(),
            label: label.to_string(),
            size_gb,
            active_size_gb,
            role: role.to_string(),
            description: description.to_string(),
            est_tokens_per_sec: None,
        }
    }

    // `active_size_gb` for a mixture-of-experts entry is derived as
    // `size_gb * (active_params / total_params)` -- the same bytes-per-parameter
    // the file already implies, applied to the subset actually read per token.
    // For a dense model it equals `size_gb`. Sizes are the model layer as
    // published on the Ollama registry, not estimates.
    vec![
        e("qwen3.5:9b", "Alibaba", "Qwen3.5 9B", 6.6, 6.6, "general", "軽量な汎用モデル"),
        e("qwen3:4b", "Alibaba", "Qwen3 4B", 2.3, 2.3, "reasoning", "軽量ながらthinking対応"),
        e("qwen3.6:35b-a3b", "Alibaba", "Qwen3.6 35B-A3B", 23.0, 2.0, "general", "MoEで効率的な大型モデル"),
        e("qwen3-coder:30b", "Alibaba", "Qwen3 Coder 30B", 19.0, 2.0, "code", "コード生成特化のMoEモデル"),
        e("gemma3:4b", "Google", "Gemma3 4B", 3.3, 3.3, "router", "超軽量、ルーター役に最適"),
        e("gemma3:12b", "Google", "Gemma3 12B", 8.1, 8.1, "general", "バランス型・マルチモーダル対応"),
        e("gemma3:27b", "Google", "Gemma3 27B", 17.0, 17.0, "general", "マルチモーダル対応の大型モデル"),
        e("mistral-small:22b", "Mistral AI", "Mistral Small 22B", 13.0, 13.0, "general", "フランスMistral AI製、128K文脈"),
        e("mistral-nemo", "Mistral AI", "Mistral Nemo 12B", 7.1, 7.1, "general", "軽量・128K文脈"),
        e("llama3.1:8b", "Meta", "Llama 3.1 8B", 4.9, 4.9, "general", "定番の汎用モデル"),
        e("llama3.2:3b", "Meta", "Llama 3.2 3B", 2.0, 2.0, "router", "超軽量、ルーター役に最適"),
        e("llama3.3:70b", "Meta", "Llama 3.3 70B", 43.0, 43.0, "general", "大規模・高精度"),
        e("deepseek-r1:8b", "DeepSeek", "DeepSeek-R1 8B", 5.2, 5.2, "reasoning", "推論特化・軽量"),
        e("deepseek-r1:32b", "DeepSeek", "DeepSeek-R1 32B", 20.0, 20.0, "reasoning", "推論特化・高精度"),
        // 15.7B total / 2.4B active, so it reads about a sixth of its file per
        // token: the only code model in this catalog that a machine without a
        // GPU can run at a usable speed.
        e("deepseek-coder-v2:16b", "DeepSeek", "DeepSeek-Coder-V2 16B", 8.3, 1.3, "code", "コード特化のMoE・CPUでも高速"),
        e("phi4:14b", "Microsoft", "Phi-4 14B", 9.1, 9.1, "general", "コンパクトながら高性能"),
        e("phi4-mini", "Microsoft", "Phi-4 mini 3.8B", 2.3, 2.3, "general", "超軽量な汎用モデル"),
        e("gpt-oss:20b", "OpenAI", "gpt-oss 20B", 13.0, 3.6, "general", "OpenAIのオープンウェイトモデル"),
        e("granite3.3:8b", "IBM", "Granite 3.3 8B", 4.6, 4.6, "general", "IBM製・商用利用しやすいライセンス"),
    ]
}

/// Tokens per second this machine can be expected to sustain for a model.
/// `None` until the engine has generated something here to measure, and when a
/// GPU is doing the work -- a figure calibrated against one VRAM-resident model
/// says little about a model that won't fit in VRAM.
pub fn estimate_tokens_per_sec(profile: &HardwareProfile, active_size_gb: f64) -> Option<f64> {
    if profile.accelerated || active_size_gb <= 0.0 {
        return None;
    }
    Some(profile.observed_gb_per_sec? / active_size_gb)
}

/// Recommends up to `limit` models the machine can actually run, cheapest
/// first, round-robining across vendors so the result isn't dominated by
/// whichever origin happens to have the most catalog entries -- diversity of
/// training lineage is the whole point of comparing multiple models.
///
/// Two independent gates, because they fail for different reasons:
///   * RAM, against total size -- a model that doesn't fit gets evicted and
///     reloaded, or pushes the machine into paging. Always applied.
///   * Speed, against *active* size and the throughput this engine was actually
///     measured doing here -- a model that fits can still be too slow to be
///     worth offering. Applied only once there's a measurement, and never to
///     the point of leaving fewer than `MIN_RECOMMENDATIONS` entries.
pub fn recommend(profile: &HardwareProfile, limit: usize) -> Vec<CatalogEntry> {
    let budget = ram_budget_gb(profile);

    let mut fits: Vec<CatalogEntry> = full_catalog()
        .into_iter()
        .filter(|entry| resident_gb(entry.size_gb) <= budget)
        .map(|mut entry| {
            entry.est_tokens_per_sec = estimate_tokens_per_sec(profile, entry.active_size_gb);
            entry
        })
        .collect();

    // Fastest first, so a trim keeps the usable end of the list.
    fits.sort_by(|a, b| {
        a.active_size_gb
            .partial_cmp(&b.active_size_gb)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let usable_count = fits
        .iter()
        .filter(|e| {
            e.est_tokens_per_sec
                .is_none_or(|est| est >= MIN_USABLE_TOKENS_PER_SEC)
        })
        .count();
    fits.truncate(usable_count.max(MIN_RECOMMENDATIONS));

    let mut by_origin: Vec<(String, Vec<CatalogEntry>)> = Vec::new();
    for entry in fits {
        match by_origin.iter_mut().find(|(origin, _)| origin == &entry.origin) {
            Some((_, bucket)) => bucket.push(entry),
            None => by_origin.push((entry.origin.clone(), vec![entry])),
        }
    }

    // Ascending: the first pick from each vendor should be its fastest, not
    // its largest. The previous ordering put a machine's least usable models
    // at the top of the list.
    for (_, bucket) in by_origin.iter_mut() {
        bucket.sort_by(|a, b| {
            a.active_size_gb
                .partial_cmp(&b.active_size_gb)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    let mut result = Vec::new();
    let mut round = 0;
    loop {
        let mut added_any = false;
        for (_, bucket) in &by_origin {
            if let Some(entry) = bucket.get(round) {
                result.push(entry.clone());
                added_any = true;
                if result.len() >= limit {
                    return result;
                }
            }
        }
        if !added_any {
            break;
        }
        round += 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference machine this port targets: 4 cores / 8 threads, 32GB,
    /// no GPU offload.
    ///
    /// Measured, not guessed: llama3.2:3b generated 18.2 tok/s from 2.0GB of
    /// weights there, and llama3.1:8b 7.7 tok/s from 4.9GB -- 36.4 and 37.7 GB/s
    /// respectively. An earlier figure of 49.6 came from a 270MB model and made
    /// every estimate optimistic; see `record_generation`'s size floor.
    const REFERENCE_GB_PER_SEC: f64 = 36.4;

    fn cpu_only_laptop() -> HardwareProfile {
        HardwareProfile {
            os: "windows".to_string(),
            total_ram_gb: 31.7,
            logical_cores: 8,
            accelerated: false,
            observed_gb_per_sec: Some(REFERENCE_GB_PER_SEC),
        }
    }

    /// Before the engine has generated anything there is nothing to calibrate
    /// against, so the speed gate has to stay out of the way rather than run on
    /// a made-up number.
    fn uncalibrated() -> HardwareProfile {
        HardwareProfile {
            observed_gb_per_sec: None,
            ..cpu_only_laptop()
        }
    }

    #[test]
    fn makes_no_speed_claim_before_it_has_measured_anything() {
        let picks = recommend(&uncalibrated(), 20);
        assert!(!picks.is_empty());
        assert!(picks.iter().all(|e| e.est_tokens_per_sec.is_none()));
        // No estimate means no speed gate: anything that fits stays on offer.
        assert!(picks.iter().any(|e| e.tag == "mistral-small:22b"));
    }

    /// The failure this pins down: an implausibly low calibration once filtered
    /// out every single model, leaving a catalog with nothing in it -- so the
    /// user couldn't even install the small model that would have worked.
    #[test]
    fn never_returns_an_empty_catalog_however_slow_the_machine_looks() {
        let crawling = HardwareProfile {
            observed_gb_per_sec: Some(0.5),
            ..cpu_only_laptop()
        };
        let picks = recommend(&crawling, 20);
        assert!(picks.len() >= MIN_RECOMMENDATIONS, "got {} picks", picks.len());
        // And what survives is the fast end of the list, not an arbitrary slice.
        assert!(picks.iter().any(|e| e.tag == "llama3.2:3b"));
    }

    #[test]
    fn recommends_the_fastest_model_of_each_vendor_first() {
        let picks = recommend(&cpu_only_laptop(), 10);
        assert!(!picks.is_empty());

        // Regression: the old ordering sorted descending, so the first pick
        // from each vendor was its largest -- on this machine that put two
        // 13GB models and a 9.1GB model at the top of the list.
        let mut seen: Vec<&str> = Vec::new();
        for entry in &picks {
            if seen.contains(&entry.origin.as_str()) {
                continue;
            }
            seen.push(&entry.origin);
            let vendor_min = picks
                .iter()
                .filter(|e| e.origin == entry.origin)
                .map(|e| e.active_size_gb)
                .fold(f64::INFINITY, f64::min);
            assert_eq!(
                entry.active_size_gb, vendor_min,
                "{} led with {}, not its lightest",
                entry.origin, entry.label
            );
        }
    }

    #[test]
    fn drops_models_too_slow_to_use_even_when_they_fit_in_ram() {
        let profile = cpu_only_laptop();
        let picks = recommend(&profile, 20);

        // mistral-small:22b is 13GB: it fits the 22.2GB budget but generates
        // around 3 tok/s here, which is what the speed gate exists for.
        assert!(
            !picks.iter().any(|e| e.tag == "mistral-small:22b"),
            "13GB dense model should be filtered out on a CPU-only machine"
        );
        assert!(picks.iter().any(|e| e.tag == "llama3.2:3b"));

        for entry in &picks {
            let est = entry.est_tokens_per_sec.expect("estimate on a calibrated CPU machine");
            assert!(est >= MIN_USABLE_TOKENS_PER_SEC, "{} too slow", entry.label);
            assert!(resident_gb(entry.size_gb) <= ram_budget_gb(&profile));
        }
    }

    /// The catalog has to offer this machine a code model it can actually run.
    /// Every dense coder is too large, so this depends on MoE being judged by
    /// active size -- without that, the code role is empty here.
    #[test]
    fn offers_a_code_model_this_machine_can_run() {
        let picks = recommend(&cpu_only_laptop(), 12);
        assert!(
            picks.iter().any(|e| e.role == "code"),
            "no code model survived the gates: {:?}",
            picks.iter().map(|e| &e.tag).collect::<Vec<_>>()
        );
    }

    /// A MoE model reads only its active experts per token, so it can clear
    /// the speed gate at a size where a dense model can't. gpt-oss:20b is
    /// 13GB on disk with ~3.6GB active: same footprint as mistral-small:22b,
    /// several times the speed.
    #[test]
    fn judges_moe_models_by_active_size_not_total() {
        let profile = cpu_only_laptop();
        let dense = estimate_tokens_per_sec(&profile, 13.0).unwrap();
        let moe = estimate_tokens_per_sec(&profile, 3.6).unwrap();
        assert!(moe > dense * 3.0);
        assert!(recommend(&profile, 20).iter().any(|e| e.tag == "gpt-oss:20b"));
    }

    /// A calibration taken from a VRAM-resident model says nothing about a
    /// model that won't fit in VRAM, so the speed gate has to step aside rather
    /// than filter on a number that doesn't transfer.
    #[test]
    fn skips_the_speed_gate_when_a_gpu_is_doing_the_work() {
        let mut profile = cpu_only_laptop();
        profile.accelerated = true;
        let picks = recommend(&profile, 20);
        assert!(picks.iter().any(|e| e.tag == "mistral-small:22b"));
        assert!(picks.iter().all(|e| e.est_tokens_per_sec.is_none()));
    }

    /// Diagnostic, not an assertion: prints what *this* machine gets, both
    /// before and after calibration. The first thing worth looking at when the
    /// app is moved somewhere new.
    ///   cargo test --lib -- --ignored --nocapture whats_recommended_here
    #[test]
    #[ignore]
    fn whats_recommended_here() {
        for (name, observed) in [
            ("uncalibrated", None),
            ("calibrated", Some(REFERENCE_GB_PER_SEC)),
        ] {
            let profile = detect_hardware(false, observed);
            println!(
                "\n=== {name} === {} / {:.1}GB RAM / {} threads / budget {:.1}GB",
                profile.os,
                profile.total_ram_gb,
                profile.logical_cores,
                ram_budget_gb(&profile),
            );
            for entry in recommend(&profile, 10) {
                match entry.est_tokens_per_sec {
                    Some(est) => println!(
                        "  {:<20} {:>6.1}G active {:>5.1}G  ~{:>5.1} tok/s",
                        entry.tag, entry.size_gb, entry.active_size_gb, est
                    ),
                    None => println!(
                        "  {:<20} {:>6.1}G active {:>5.1}G  (未測定)",
                        entry.tag, entry.size_gb, entry.active_size_gb
                    ),
                }
            }
        }
    }

    #[test]
    fn never_recommends_a_model_that_cannot_fit() {
        let profile = cpu_only_laptop();
        let picks = recommend(&profile, 20);
        // 43GB of weights cannot fit 31.7GB of RAM under any speed rule.
        assert!(!picks.iter().any(|e| e.tag == "llama3.3:70b"));
    }
}
