import { useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type { ChatMessage, ChatStreamEvent, ModelInfo, ModelTarget, ProviderInfo } from "./types";
import Settings from "./Settings";
import Catalog from "./Catalog";
import "./App.css";

type OllamaStatus = "checking" | "connected" | "unreachable";
type ReplyStatus = "streaming" | "done" | "error";

interface ModelReply {
  target: ModelTarget;
  requestId: string;
  content: string;
  thinking?: string;
  status: ReplyStatus;
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
  const isComposingRef = useRef(false);

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

  useEffect(() => {
    function updateReply(requestId: string, updater: (r: ModelReply) => ModelReply) {
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

    const unlisten = listen<ChatStreamEvent>("chat-stream", (event) => {
      const payload = event.payload;

      if (payload.type === "thinking") {
        updateReply(payload.request_id, (r) => ({
          ...r,
          thinking: (r.thinking ?? "") + payload.content,
        }));
      } else if (payload.type === "token") {
        updateReply(payload.request_id, (r) => ({ ...r, content: r.content + payload.content }));
      } else if (payload.type === "done") {
        updateReply(payload.request_id, (r) =>
          !r.content && r.thinking
            ? { ...r, status: "done", content: "（思考の途中で応答が終了しました。もう一度お試しください）" }
            : { ...r, status: "done" },
        );
      } else if (payload.type === "error") {
        updateReply(payload.request_id, (r) => ({
          ...r,
          status: "error",
          content: `[error: ${payload.message}]`,
        }));
      }
    });

    return () => {
      unlisten.then((fn) => fn());
    };
  }, []);

  function toggleTarget(id: string) {
    setSelectedIds((prev) => (prev.includes(id) ? prev.filter((t) => t !== id) : [...prev, id]));
  }

  async function handleSend() {
    const text = input.trim();
    const targets = selectedIds.map((id) => targetsById.get(id)).filter((t): t is ModelTarget => !!t);
    if (!text || targets.length === 0 || isStreaming) return;

    const priorTurns = turns;
    const replies: ModelReply[] = targets.map((target) => ({
      target,
      requestId: crypto.randomUUID(),
      content: "",
      status: "streaming",
    }));

    setTurns((prev) => [...prev, { id: crypto.randomUUID(), userText: text, replies }]);
    setInput("");

    for (const reply of replies) {
      const outgoing: ChatMessage[] = [
        ...buildHistoryForTarget(priorTurns, reply.target.id),
        { role: "user", content: text },
      ];
      invoke("send_chat", {
        requestId: reply.requestId,
        provider: reply.target.provider,
        model: reply.target.model,
        messages: outgoing,
        think: thinkMode,
      }).catch((err) => {
        setTurns((prev) =>
          prev.map((turn) => ({
            ...turn,
            replies: turn.replies.map((r) =>
              r.requestId === reply.requestId
                ? { ...r, status: "error" as const, content: `[error: ${String(err)}]` }
                : r,
            ),
          })),
        );
      });
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
                <button
                  key={t.id}
                  className={selectedIds.includes(t.id) ? "active" : ""}
                  onClick={() => toggleTarget(t.id)}
                  disabled={isStreaming}
                  title={t.provider}
                >
                  {t.label}
                </button>
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
                return (
                  <div key={r.requestId} className={`reply-panel reply-${r.status}`}>
                    <div className="reply-header">
                      <span className="reply-model">{r.target.label}</span>
                      <span className={`reply-status reply-status-${r.status}`}>
                        {r.status === "streaming" ? (stillThinking ? "思考中…" : "生成中…") : r.status}
                      </span>
                    </div>
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
