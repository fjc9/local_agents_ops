use serde::Serialize;

use crate::engines::ollama::OllamaBackend;
use crate::engines::{ChatMessage, ChatRole, EngineError};

#[derive(Debug, Serialize)]
pub struct RouterDecision {
    pub providers: Vec<String>,
    pub compressed_prompt: String,
}

#[derive(Debug, thiserror::Error)]
pub enum RouterError {
    #[error("engine error: {0}")]
    Engine(#[from] EngineError),
    #[error("could not parse router output: {0}")]
    Parse(String),
}

/// Keep roughly the last few exchanges -- enough for the parent model to
/// follow the thread without it (or us) resending an ever-growing
/// transcript to a paid API on every turn.
const MAX_HISTORY_MESSAGES: usize = 6;

/// Cheap first pass, no model call: shrink history before the expensive
/// semantic compression step even sees it.
fn deterministic_trim(history: &[ChatMessage]) -> Vec<ChatMessage> {
    let start = history.len().saturating_sub(MAX_HISTORY_MESSAGES);
    history[start..]
        .iter()
        .map(|m| ChatMessage {
            role: m.role,
            content: m.content.split_whitespace().collect::<Vec<_>>().join(" "),
        })
        .collect()
}

fn provider_label(id: &str) -> &str {
    match id {
        "anthropic" => "Claude",
        "openai" => "ChatGPT",
        "gemini" => "Gemini",
        "xai" => "Grok",
        other => other,
    }
}

fn role_label(role: ChatRole) -> &'static str {
    match role {
        ChatRole::User => "ユーザー",
        ChatRole::Assistant => "アシスタント",
        ChatRole::System => "システム",
    }
}

/// Asks the parent local model to (a) pick which of the candidate online
/// providers are actually worth calling for this turn and (b) compress the
/// context down to what's needed -- the two moves that make fanning a
/// prompt out to paid APIs cheaper than sending everything to everything.
pub async fn route_and_compress(
    ollama: &OllamaBackend,
    parent_model: &str,
    candidate_providers: &[String],
    history: &[ChatMessage],
    new_message: &str,
) -> Result<RouterDecision, RouterError> {
    let trimmed = deterministic_trim(history);

    let candidates_desc = candidate_providers
        .iter()
        .map(|id| format!("- {id} ({})", provider_label(id)))
        .collect::<Vec<_>>()
        .join("\n");

    let history_desc = if trimmed.is_empty() {
        "(まだ会話の最初です)".to_string()
    } else {
        trimmed
            .iter()
            .map(|m| format!("{}: {}", role_label(m.role), m.content))
            .collect::<Vec<_>>()
            .join("\n")
    };

    let prompt = format!(
        "あなたは複数の有料オンラインAIサービスへの問い合わせを管理する司令塔です。コストを抑えるため以下を厳守してください。\n\
        1. 候補サービスのうち、この質問に答える上で本当に呼び出す価値があるものだけを選ぶ(全部選ぶ必要はない)\n\
        2. compressed_promptには実際にサービスへ送信する全文を必ず書く(絶対に空にしない)\n\
        3. 新しい質問がすでに短く簡潔な場合は、言い換えずに新しい質問をそのままcompressed_promptにコピーする(固有名詞や主題を落とすと意味が変わってしまうため)\n\
        4. 会話が長い場合のみ、要点を保ったまま過去のやり取りを圧縮してcompressed_promptに含める(新しい質問自体の言葉は変えない)\n\
        5. 出力は次のJSON形式のみ。前置きや説明は一切書かない:\n\
        {{\"providers\": [\"id\", ...], \"compressed_prompt\": \"(実際に送信する全文。空にしない)\"}}\n\n\
        候補サービス:\n{candidates_desc}\n\n\
        これまでの会話:\n{history_desc}\n\n\
        新しい質問:\n{new_message}"
    );

    let messages = [ChatMessage {
        role: ChatRole::User,
        content: prompt,
    }];

    let raw = ollama.chat_once(parent_model, &messages, true).await?;
    parse_decision(&raw, candidate_providers)
}

fn parse_decision(raw: &str, candidates: &[String]) -> Result<RouterDecision, RouterError> {
    let value: serde_json::Value =
        serde_json::from_str(raw.trim()).map_err(|e| RouterError::Parse(format!("{e}: {raw}")))?;

    let providers: Vec<String> = value
        .get("providers")
        .and_then(|p| p.as_array())
        .ok_or_else(|| RouterError::Parse("missing providers array".to_string()))?
        .iter()
        .filter_map(|v| v.as_str())
        .map(str::to_string)
        .filter(|p| candidates.contains(p))
        .collect();

    let compressed_prompt = value
        .get("compressed_prompt")
        .and_then(|p| p.as_str())
        .ok_or_else(|| RouterError::Parse("missing compressed_prompt".to_string()))?
        .to_string();

    if providers.is_empty() || compressed_prompt.trim().is_empty() {
        return Err(RouterError::Parse(
            "empty providers or compressed_prompt".to_string(),
        ));
    }

    Ok(RouterDecision {
        providers,
        compressed_prompt,
    })
}
