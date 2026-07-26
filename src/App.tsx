import { useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type {
  ChatMessage,
  ChatStreamEvent,
  ModelInfo,
  ModelTarget,
  ProviderInfo,
  RouterDecision,
} from "./types";
import Settings from "./Settings";
import Catalog from "./Catalog";
import "./App.css";

type OllamaStatus = "checking" | "connected" | "unreachable";
type ReplyStatus = "streaming" | "done" | "error" | "skipped";

const ONLINE_PROVIDER_IDS = new Set(["anthropic", "openai", "gemini", "xai"]);
const PARENT_MODEL_STORAGE_KEY = "localAgentsOps.parentModel";

interface ModelReply {
  target: ModelTarget;
  requestId: string;
  content: string;
  thinking?: string;
  status: ReplyStatus;
  /** What was actually sent, when the parent model rewrote/compressed the
   * prompt for this online target -- shown so routing isn't a black box. */
  sentContent?: string;
}

interface Turn {
  id: string;
  userText: string;
  replies: ModelReply[];
}

/** Each model gets its own conversation thread: its own past answers only,
 * not what the other models in a parallel-comparison turn said. */
function buildHistoryForTarget(turns: Turn[], targetId: string): ChatMessage[] {
  const history: ChatMessage[] = [];
  for (const turn of turns) {
    const reply = turn.replies.find((r) => r.target.id === targetId);
    if (!reply) continue;
    history.push({ role: "user", content: turn.userText });
    history.push({ role: "assistant", content: reply.content });
  }
  return history;
}

function App() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [status, setStatus] = useState<OllamaStatus>("checking");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [thinkMode, setThinkMode] = useState(false);
  const [onlineMode, setOnlineMode] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [catalogOpen, setCatalogOpen] = useState(false);
  const [unloadingModels, setUnloadingModels] = useState<Set<string>>(new Set());
  const [parentModel, setParentModel] = useState(() => {
    try {
      return localStorage.getItem(PARENT_MODEL_STORAGE_KEY) ?? "";
    } catch {
      return "";
    }
  });
  const [routing, setRouting] = useState(false);
  const isComposingRef = useRef(false);

  function updateParentModel(model: string) {
    setParentModel(model);
    try {
      localStorage.setItem(PARENT_MODEL_STORAGE_KEY, model);
    } catch {
      // Persistence is a nice-to-have; a blocked/unavailable localStorage
      // (private context, storage quota, etc.) shouldn't break selection.
    }
  }

  const isStreaming =
    turns.length > 0 && turns[turns.length - 1].replies.some((r) => r.status === "streaming");

  const localTargets: ModelTarget[] = useMemo(
    () =>
      models.map((m) => ({
        id: `ollama:${m.name}`,
        provider: "ollama",
        model: m.name,
        label: m.name,
      })),
    [models],
  );

  const onlineTargets: ModelTarget[] = useMemo(
    () =>
      providers
        .filter((p) => p.configured)
        .map((p) => ({
          id: `${p.id}:${p.default_model}`,
          provider: p.id,
          model: p.default_model,
          label: p.label,
        })),
    [providers],
  );

  const availableTargets = onlineMode ? [...localTargets, ...onlineTargets] : localTargets;
  const targetsById = useMemo(() => {
    const map = new Map<string, ModelTarget>();
    for (const t of availableTargets) map.set(t.id, t);
    return map;
  }, [availableTargets]);

  function refreshProviders() {
    invoke<ProviderInfo[]>("list_providers").then(setProviders).catch(() => {});
  }

  function refreshModels() {
    invoke<ModelInfo[]>("list_ollama_models")
      .then((list) => {
        setModels(list);
        setStatus("connected");
        setSelectedIds((prev) => {
          const known = new Set(prev);
          const additions = list.map((m) => `ollama:${m.name}`).filter((id) => !known.has(id));
          return [...prev, ...additions];
        });
      })
      .catch(() => setStatus("unreachable"));
  }

  useEffect(() => {
    invoke<ModelInfo[]>("list_ollama_models")
      .then((list) => {
        setModels(list);
        setStatus("connected");
        setSelectedIds(list.map((m) => `ollama:${m.name}`));
      })
      .catch(() => setStatus("unreachable"));
    refreshProviders();
  }, []);

  // Default the parent (router) model to the smallest installed one, so
  // there's a sensible pick without asking the user to configure it first --
  // they can still change it in Settings.
  useEffect(() => {
    if (models.length === 0) return;
    if (parentModel && models.some((m) => m.name === parentModel)) return;
    const smallest = [...models].sort((a, b) => (a.size_bytes ?? 0) - (b.size_bytes ?? 0))[0];
    updateParentModel(smallest.name);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [models]);

  function updateReplyByRequestId(requestId: string, updater: (r: ModelReply) => ModelReply) {
    setTurns((prev) =>
      prev.map((turn) => {
        if (!turn.replies.some((r) => r.requestId === requestId)) return turn;
        return {
          ...turn,
          replies: turn.replies.map((r) => (r.requestId === requestId ? updater(r) : r)),
        };
      }),
    );
  }

  useEffect(() => {
    const unlisten = listen<ChatStreamEvent>("chat-stream", (event) => {
      const payload = event.payload;

      if (payload.type === "thinking") {
        updateReplyByRequestId(payload.request_id, (r) => ({
          ...r,
          thinking: (r.thinking ?? "") + payload.content,
        }));
      } else if (payload.type === "token") {
        updateReplyByRequestId(payload.request_id, (r) => ({ ...r, content: r.content + payload.content }));
      } else if (payload.type === "done") {
        updateReplyByRequestId(payload.request_id, (r) =>
          !r.content && r.thinking
            ? { ...r, status: "done", content: "（思考の途中で応答が終了しました。もう一度お試しください）" }
            : { ...r, status: "done" },
        );
      } else if (payload.type === "error") {
        updateReplyByRequestId(payload.request_id, (r) => ({
          ...r,
          status: "error",
          content: `[error: ${payload.message}]`,
        }));
      }
    });

    return () => {
      unlisten.then((fn) => fn());
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function toggleTarget(id: string) {
    setSelectedIds((prev) => (prev.includes(id) ? prev.filter((t) => t !== id) : [...prev, id]));
  }

  async function handleUnload(model: string) {
    setUnloadingModels((prev) => new Set(prev).add(model));
    try {
      await invoke("unload_model", { model });
    } catch (err) {
      console.error(`failed to unload ${model}:`, err);
    } finally {
      setUnloadingModels((prev) => {
        const next = new Set(prev);
        next.delete(model);
        return next;
      });
    }
  }

  function sendToTarget(reply: ModelReply, messages: ChatMessage[]) {
    invoke("send_chat", {
      requestId: reply.requestId,
      provider: reply.target.provider,
      model: reply.target.model,
      messages,
      think: thinkMode,
    }).catch((err) => {
      updateReplyByRequestId(reply.requestId, (r) => ({
        ...r,
        status: "error",
        content: `[error: ${String(err)}]`,
      }));
    });
  }

  async function handleSend() {
    const text = input.trim();
    const targets = selectedIds.map((id) => targetsById.get(id)).filter((t): t is ModelTarget => !!t);
    if (!text || targets.length === 0 || isStreaming) return;

    const priorTurns = turns;
    const localSelected = targets.filter((t) => t.provider === "ollama");
    const onlineSelected = targets.filter((t) => ONLINE_PROVIDER_IDS.has(t.provider));

    const replies: ModelReply[] = targets.map((target) => ({
      target,
      requestId: crypto.randomUUID(),
      content: "",
      status: "streaming",
    }));
    const replyByTargetId = new Map(replies.map((r) => [r.target.id, r]));

    setTurns((prev) => [...prev, { id: crypto.randomUUID(), userText: text, replies }]);
    setInput("");

    // Local models always get their own full history, sent immediately --
    // the parent-model routing/compression step is about protecting paid
    // API calls from waste, not about how local models are prompted.
    for (const target of localSelected) {
      const reply = replyByTargetId.get(target.id);
      if (!reply) continue;
      const outgoing: ChatMessage[] = [
        ...buildHistoryForTarget(priorTurns, target.id),
        { role: "user", content: text },
      ];
      sendToTarget(reply, outgoing);
    }

    if (onlineSelected.length === 0) return;

    const sendOnlineDirect = () => {
      for (const target of onlineSelected) {
        const reply = replyByTargetId.get(target.id);
        if (!reply) continue;
        const outgoing: ChatMessage[] = [
          ...buildHistoryForTarget(priorTurns, target.id),
          { role: "user", content: text },
        ];
        sendToTarget(reply, outgoing);
      }
    };

    if (!parentModel) {
      sendOnlineDirect();
      return;
    }

    setRouting(true);
    try {
      const representativeHistory = buildHistoryForTarget(priorTurns, onlineSelected[0].id);
      const decision = await invoke<RouterDecision>("route_and_compress", {
        parentModel,
        candidateProviders: onlineSelected.map((t) => t.provider),
        history: representativeHistory,
        newMessage: text,
      });

      for (const target of onlineSelected) {
        const reply = replyByTargetId.get(target.id);
        if (!reply) continue;
        if (decision.providers.includes(target.provider)) {
          updateReplyByRequestId(reply.requestId, (r) => ({ ...r, sentContent: decision.compressed_prompt }));
          sendToTarget(reply, [{ role: "user", content: decision.compressed_prompt }]);
        } else {
          updateReplyByRequestId(reply.requestId, (r) => ({ ...r, status: "skipped" }));
        }
      }
    } catch (err) {
      console.error("routing failed, falling back to a direct send:", err);
      sendOnlineDirect();
    } finally {
      setRouting(false);
    }
  }

  return (
    <div className="app">
      <header className="toolbar">
        <span className="app-title">LocalAgentsOps</span>
        {status === "checking" && <span className="status status-checking">Ollama: 確認中…</span>}
        {status === "unreachable" && (
          <span className="status status-error">
            Ollama未接続 — ターミナルで `ollama serve` を実行してください
          </span>
        )}
        {status === "connected" && (
          <>
            <div className="think-toggle" role="group" aria-label="思考モード">
              <button
                className={!thinkMode ? "active" : ""}
                onClick={() => setThinkMode(false)}
                disabled={isStreaming}
                title="thinkingを使わず即座に回答"
              >
                ⚡ 高速
              </button>
              <button
                className={thinkMode ? "active" : ""}
                onClick={() => setThinkMode(true)}
                disabled={isStreaming}
                title="thinkingを使ってじっくり推論してから回答"
              >
                🧠 じっくり
              </button>
            </div>
            <div className="think-toggle" role="group" aria-label="オンライン連携">
              <button
                className={onlineMode ? "active" : ""}
                onClick={() => setOnlineMode((v) => !v)}
                disabled={isStreaming}
                title="Claude/ChatGPT/Gemini/Grokも比較対象に含める"
              >
                🌐 オンライン
              </button>
            </div>
            <div className="model-select" role="group" aria-label="比較するモデル">
              {availableTargets.map((t) => (
                <div key={t.id} className="model-chip">
                  <button
                    className={selectedIds.includes(t.id) ? "active" : ""}
                    onClick={() => toggleTarget(t.id)}
                    disabled={isStreaming}
                    title={t.provider}
                  >
                    {t.label}
                  </button>
                  {t.provider === "ollama" && (
                    <button
                      className="unload-button"
                      onClick={() => handleUnload(t.model)}
                      disabled={isStreaming || unloadingModels.has(t.model)}
                      title="メモリから解放 (unload)"
                    >
                      {unloadingModels.has(t.model) ? "…" : "⏏"}
                    </button>
                  )}
                </div>
              ))}
            </div>
            <button className="settings-button" onClick={() => setCatalogOpen(true)} title="モデルカタログ">
              📦
            </button>
            <button className="settings-button" onClick={() => setSettingsOpen(true)} title="APIキー設定">
              ⚙️
            </button>
          </>
        )}
      </header>

      {onlineMode && onlineTargets.length === 0 && (
        <div className="online-hint">
          オンラインで比較するには⚙️からAPIキーを設定してください
        </div>
      )}
      {onlineMode && onlineTargets.length > 0 && !parentModel && (
        <div className="online-hint">
          ⚙️で親モデルを設定すると、送信前に内容を最適化して不要なサービスへの送信を減らせます
        </div>
      )}
      {routing && (
        <div className="online-hint routing-hint">🧠 親モデルが送信内容を最適化中…</div>
      )}

      <main className="turns">
        {turns.length === 0 && (
          <div className="empty-state">
            比較したいモデルを選んでメッセージを送信してください（複数選択で並列比較）
          </div>
        )}
        {turns.map((turn) => (
          <div key={turn.id} className="turn">
            <div className="message message-user">
              <div className="message-role">USER</div>
              <div className="message-content">{turn.userText}</div>
            </div>
            <div className="reply-row">
              {turn.replies.map((r) => {
                const stillThinking = r.status === "streaming" && !r.content && !!r.thinking;
                const waiting = r.status === "streaming" && !r.content && !r.thinking;
                if (r.status === "skipped") {
                  return (
                    <div key={r.requestId} className="reply-panel reply-skipped">
                      <div className="reply-header">
                        <span className="reply-model">{r.target.label}</span>
                        <span className="reply-status">親モデルが不要と判断</span>
                      </div>
                      <p className="catalog-description">
                        今回の質問には呼び出す価値がないと親モデルが判断したため、送信をスキップしました。
                      </p>
                    </div>
                  );
                }
                return (
                  <div key={r.requestId} className={`reply-panel reply-${r.status}`}>
                    <div className="reply-header">
                      <span className="reply-model">{r.target.label}</span>
                      <span className={`reply-status reply-status-${r.status}`}>
                        {r.status === "streaming" ? (stillThinking ? "思考中…" : "生成中…") : r.status}
                      </span>
                    </div>
                    {r.sentContent && (
                      <details className="message-thinking">
                        <summary>親モデルが最適化して送信</summary>
                        <div className="message-thinking-content">{r.sentContent}</div>
                      </details>
                    )}
                    {r.thinking && (
                      <details className="message-thinking" open={stillThinking}>
                        <summary>{stillThinking ? "思考中…" : "思考の過程"}</summary>
                        <div className="message-thinking-content">{r.thinking}</div>
                      </details>
                    )}
                    <div className="message-content">{r.content || (waiting ? "…" : "")}</div>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </main>

      <footer className="composer">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onCompositionStart={() => {
            isComposingRef.current = true;
          }}
          onCompositionEnd={() => {
            isComposingRef.current = false;
          }}
          onKeyDown={(e) => {
            if (e.key !== "Enter" || e.shiftKey) return;
            // IME conversion-confirm also fires an Enter keydown. Guard with
            // all three signals -- WebKit (Tauri's macOS webview) doesn't
            // reliably mark that keydown as isComposing, so the ref from
            // compositionstart/end and the legacy keyCode 229 check both
            // matter, not just nativeEvent.isComposing.
            if (isComposingRef.current || e.nativeEvent.isComposing || e.keyCode === 229) return;
            e.preventDefault();
            handleSend();
          }}
          placeholder="プロンプトを入力 (Enterで送信 / Shift+Enterで改行)"
          disabled={status !== "connected" || isStreaming}
        />
        <button
          onClick={handleSend}
          disabled={status !== "connected" || isStreaming || !input.trim() || selectedIds.length === 0}
        >
          {isStreaming ? "生成中…" : `送信 (${selectedIds.length})`}
        </button>
      </footer>

      {settingsOpen && (
        <Settings
          providers={providers}
          models={models}
          parentModel={parentModel}
          onParentModelChange={updateParentModel}
          onClose={() => setSettingsOpen(false)}
          onChanged={refreshProviders}
        />
      )}

      {catalogOpen && (
        <Catalog
          installedModels={models}
          onClose={() => setCatalogOpen(false)}
          onModelsChanged={refreshModels}
        />
      )}
    </div>
  );
}

export default App;
