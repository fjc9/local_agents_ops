use serde::Serialize;

use crate::engines::ollama::OllamaBackend;
use crate::engines::{ChatMessage, ChatRole, EngineError};

#[derive(Debug, Serialize)]
pub struct RouterDecision {
    pub providers: Vec<String>,
    pub compressed_prompt: String,
    /// True when this was decided without a model call, so the UI doesn't
    /// claim the parent model optimised something it never looked at.
    pub shortcut: bool,
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

/// A question at or under this length has nothing to compress: the router
/// prompt itself is already several times longer than the payload it would be
/// deciding about.
const SHORTCUT_MAX_CHARS: usize = 400;

/// Returns a decision without calling the parent model, when there is
/// demonstrably nothing for it to do: no prior conversation to compress, and a
/// question short enough that the instructions would have it copied verbatim
/// anyway. Every candidate provider is kept -- the user picked them
/// explicitly, and dropping one on no evidence is the worse error.
fn shortcut_decision(
    candidate_providers: &[String],
    history: &[ChatMessage],
    new_message: &str,
) -> Option<RouterDecision> {
    if !history.is_empty() || new_message.chars().count() > SHORTCUT_MAX_CHARS {
        return None;
    }
    Some(RouterDecision {
        providers: candidate_providers.to_vec(),
        compressed_prompt: new_message.to_string(),
        shortcut: true,
    })
}

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
/// providers are actually worth calling for this turn and (b) phrase the
/// question to send them -- the two moves that make fanning a prompt out to
/// paid APIs cheaper than sending everything to everything.
///
/// `history` should carry the user's own turns only. Each provider keeps its own
/// thread and is sent its own prior answers directly, so folding one provider's
/// replies in here would have the router judge every provider by what a
/// different one happened to say. User turns are identical across providers,
/// which makes the decision symmetric.
pub async fn route_and_compress(
    ollama: &OllamaBackend,
    parent_model: &str,
    candidate_providers: &[String],
    history: &[ChatMessage],
    new_message: &str,
) -> Result<RouterDecision, RouterError> {
    // The prompt already tells the parent model to copy a short question
    // through unchanged -- so on a short question the whole model call buys
    // nothing but latency, and on a CPU-only machine that latency is the
    // dominant cost of the turn: prompt processing alone runs tens of seconds
    // for a 9B parent. Decide the trivial case without asking.
    if let Some(decision) = shortcut_decision(candidate_providers, history, new_message) {
        return Ok(decision);
    }

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
        2. どのサービスも呼び出す価値がないと判断したら、providersを空配列[]にする(それが最も安い結果であり、正しい判断です)\n\
        3. compressed_promptには「新しい質問」を送信する形で書く。短く簡潔ならそのままコピーする(固有名詞や主題を落とすと意味が変わってしまうため)\n\
        4. 過去のやり取りは各サービスへ別途そのまま送られる。compressed_promptに過去の内容を混ぜてはいけない\n\
        5. 出力は次のJSON形式のみ。前置きや説明は一切書かない:\n\
        {{\"providers\": [\"id\", ...], \"compressed_prompt\": \"(送信する質問文)\"}}\n\n\
        候補サービス:\n{candidates_desc}\n\n\
        これまでのユーザーの発言:\n{history_desc}\n\n\
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
        .unwrap_or_default()
        .to_string();

    // An empty provider list is a legitimate answer -- "none of these are worth
    // paying for this turn" is exactly what the parent model is asked to decide,
    // and it is the cheapest possible outcome. Treating it as a parse failure
    // made the caller fall back to sending everything to everyone, turning the
    // cheapest decision into the most expensive behaviour the app has.
    if providers.is_empty() {
        return Ok(RouterDecision {
            providers,
            compressed_prompt: String::new(),
            shortcut: false,
        });
    }

    // With providers chosen, there has to be something to send them.
    if compressed_prompt.trim().is_empty() {
        return Err(RouterError::Parse(format!(
            "chose {} provider(s) but left compressed_prompt empty",
            providers.len()
        )));
    }

    Ok(RouterDecision {
        providers,
        compressed_prompt,
        shortcut: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn providers() -> Vec<String> {
        vec!["anthropic".to_string(), "openai".to_string()]
    }

    #[test]
    fn short_opening_question_skips_the_model_call() {
        let decision = shortcut_decision(&providers(), &[], "Rustの所有権とは？")
            .expect("a short opening question should shortcut");
        assert!(decision.shortcut);
        // The question has to survive verbatim: paraphrasing a short prompt is
        // how proper nouns and the actual subject get lost.
        assert_eq!(decision.compressed_prompt, "Rustの所有権とは？");
        assert_eq!(decision.providers, providers());
    }

    #[test]
    fn existing_conversation_still_goes_to_the_parent_model() {
        let history = [ChatMessage {
            role: ChatRole::User,
            content: "前の質問".to_string(),
        }];
        assert!(shortcut_decision(&providers(), &history, "短い質問").is_none());
    }

    #[test]
    fn long_question_still_goes_to_the_parent_model() {
        let long = "あ".repeat(SHORTCUT_MAX_CHARS + 1);
        assert!(shortcut_decision(&providers(), &[], &long).is_none());
    }

    /// Counted in characters, not bytes: a Japanese prompt is ~3 bytes per
    /// character, so a byte-based threshold would send everything to the
    /// parent model at a third of the intended length.
    #[test]
    fn threshold_counts_characters_not_bytes() {
        let multibyte = "あ".repeat(SHORTCUT_MAX_CHARS);
        assert!(multibyte.len() > SHORTCUT_MAX_CHARS);
        assert!(shortcut_decision(&providers(), &[], &multibyte).is_some());
    }

    /// "Send this to nobody" is the cheapest outcome the router can produce and
    /// has to survive as a decision. It used to be reported as a parse failure,
    /// and the caller's failure path is an uncompressed send to every selected
    /// provider -- so the cheapest judgement turned into the most expensive
    /// behaviour in the app.
    #[test]
    fn declining_every_provider_is_a_decision_not_an_error() {
        let decision = parse_decision(r#"{"providers": [], "compressed_prompt": ""}"#, &providers())
            .expect("an empty provider list is a legitimate answer");
        assert!(decision.providers.is_empty());
    }

    /// Same when the model names only providers that weren't on offer: after
    /// filtering there is nobody left to call, which is still "send to nobody"
    /// rather than a reason to bill every provider.
    #[test]
    fn unknown_provider_names_filter_down_to_sending_nothing() {
        let decision = parse_decision(
            r#"{"providers": ["some-service-we-do-not-have"], "compressed_prompt": "x"}"#,
            &providers(),
        )
        .expect("filtered-out providers should not be an error");
        assert!(decision.providers.is_empty());
    }

    /// Choosing providers but leaving nothing to send them is still broken
    /// output -- there's no message to bill for.
    #[test]
    fn choosing_providers_with_no_prompt_is_still_an_error() {
        assert!(parse_decision(
            r#"{"providers": ["openai"], "compressed_prompt": "   "}"#,
            &providers()
        )
        .is_err());
    }

    #[test]
    fn malformed_output_is_still_an_error() {
        assert!(parse_decision("not json at all", &providers()).is_err());
        assert!(parse_decision(r#"{"compressed_prompt": "x"}"#, &providers()).is_err());
    }
}
