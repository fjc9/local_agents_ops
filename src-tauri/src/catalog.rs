use serde::Serialize;
use sysinfo::System;

#[derive(Debug, Clone, Serialize)]
pub struct HardwareProfile {
    pub os: String,
    pub total_ram_gb: f64,
}

pub fn detect_hardware() -> HardwareProfile {
    let mut sys = System::new();
    sys.refresh_memory();
    HardwareProfile {
        os: std::env::consts::OS.to_string(),
        total_ram_gb: sys.total_memory() as f64 / 1024f64.powi(3),
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CatalogEntry {
    pub tag: String,
    pub origin: String,
    pub label: String,
    pub size_gb: f64,
    pub role: String,
    pub description: String,
}

/// Loaded model weights need meaningfully more RAM than the on-disk file
/// (KV cache, context buffers, runtime overhead) -- this is a rough
/// multiplier, not a precise per-context-length estimate.
const RUNTIME_OVERHEAD_MULTIPLIER: f64 = 1.6;
/// Leave headroom for the OS, this app, and other running models.
const RAM_BUDGET_FRACTION: f64 = 0.7;

fn full_catalog() -> Vec<CatalogEntry> {
    fn e(tag: &str, origin: &str, label: &str, size_gb: f64, role: &str, description: &str) -> CatalogEntry {
        CatalogEntry {
            tag: tag.to_string(),
            origin: origin.to_string(),
            label: label.to_string(),
            size_gb,
            role: role.to_string(),
            description: description.to_string(),
        }
    }

    vec![
        e("qwen3.5:9b", "Alibaba", "Qwen3.5 9B", 6.6, "general", "軽量な汎用モデル"),
        e("qwen3.6:35b-a3b", "Alibaba", "Qwen3.6 35B-A3B", 23.0, "general", "MoEで効率的な大型モデル"),
        e("qwen3-coder:30b", "Alibaba", "Qwen3 Coder 30B", 19.0, "code", "コード生成特化のMoEモデル"),
        e("gemma3:4b", "Google", "Gemma3 4B", 3.3, "router", "超軽量、ルーター役に最適"),
        e("gemma3:12b", "Google", "Gemma3 12B", 8.1, "general", "バランス型・マルチモーダル対応"),
        e("gemma3:27b", "Google", "Gemma3 27B", 17.0, "general", "マルチモーダル対応の大型モデル"),
        e("mistral-small:22b", "Mistral AI", "Mistral Small 22B", 13.0, "general", "フランスMistral AI製、128K文脈"),
        e("mistral-nemo", "Mistral AI", "Mistral Nemo 12B", 7.1, "general", "軽量・128K文脈"),
        e("llama3.1:8b", "Meta", "Llama 3.1 8B", 4.9, "general", "定番の汎用モデル"),
        e("llama3.2:3b", "Meta", "Llama 3.2 3B", 2.0, "router", "超軽量、ルーター役に最適"),
        e("llama3.3:70b", "Meta", "Llama 3.3 70B", 43.0, "general", "大規模・高精度"),
        e("deepseek-r1:8b", "DeepSeek", "DeepSeek-R1 8B", 5.2, "reasoning", "推論特化・軽量"),
        e("deepseek-r1:32b", "DeepSeek", "DeepSeek-R1 32B", 20.0, "reasoning", "推論特化・高精度"),
        e("phi4:14b", "Microsoft", "Phi-4 14B", 9.1, "general", "コンパクトながら高性能"),
        e("gpt-oss:20b", "OpenAI", "gpt-oss 20B", 13.0, "general", "OpenAIのオープンウェイトモデル"),
    ]
}

/// Recommends up to `limit` models that fit the machine's RAM budget,
/// round-robining across vendors first so the result isn't dominated by
/// whichever origin happens to have the most catalog entries -- diversity
/// of training lineage is the whole point of comparing multiple models.
pub fn recommend(profile: &HardwareProfile, limit: usize) -> Vec<CatalogEntry> {
    let budget = profile.total_ram_gb * RAM_BUDGET_FRACTION;

    let mut by_origin: Vec<(String, Vec<CatalogEntry>)> = Vec::new();
    for entry in full_catalog() {
        if entry.size_gb * RUNTIME_OVERHEAD_MULTIPLIER > budget {
            continue;
        }
        match by_origin.iter_mut().find(|(origin, _)| origin == &entry.origin) {
            Some((_, bucket)) => bucket.push(entry),
            None => by_origin.push((entry.origin.clone(), vec![entry])),
        }
    }
    for (_, bucket) in by_origin.iter_mut() {
        bucket.sort_by(|a, b| b.size_gb.partial_cmp(&a.size_gb).unwrap());
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
