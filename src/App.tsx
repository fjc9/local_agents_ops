import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type { ChatMessage, ChatStreamEvent, ModelInfo } from "./types";
import "./App.css";

type OllamaStatus = "checking" | "connected" | "unreachable";
type ReplyStatus = "streaming" | "done" | "error";

interface ModelReply {
  model: string;
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
function buildHistoryForModel(turns: Turn[], model: string): ChatMessage[] {
  const history: ChatMessage[] = [];
  for (const turn of turns) {
    const reply = turn.replies.find((r) => r.model === model);
    if (!reply) continue;
    history.push({ role: "user", content: turn.userText });
    history.push({ role: "assistant", content: reply.content });
  }
  return history;
}

function App() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [status, setStatus] = useState<OllamaStatus>("checking");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [thinkMode, setThinkMode] = useState(false);

  const isStreaming =
    turns.length > 0 && turns[turns.length - 1].replies.some((r) => r.status === "streaming");

  useEffect(() => {
    invoke<ModelInfo[]>("list_ollama_models")
      .then((list) => {
        setModels(list);
        setStatus("connected");
        setSelectedModels(list.map((m) => m.name));
      })
      .catch(() => setStatus("unreachable"));
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

  function toggleModel(name: string) {
    setSelectedModels((prev) =>
      prev.includes(name) ? prev.filter((m) => m !== name) : [...prev, name],
    );
  }

  async function handleSend() {
    const text = input.trim();
    if (!text || selectedModels.length === 0 || isStreaming) return;

    const priorTurns = turns;
    const replies: ModelReply[] = selectedModels.map((model) => ({
      model,
      requestId: crypto.randomUUID(),
      content: "",
      status: "streaming",
    }));

    setTurns((prev) => [...prev, { id: crypto.randomUUID(), userText: text, replies }]);
    setInput("");

    for (const reply of replies) {
      const outgoing: ChatMessage[] = [
        ...buildHistoryForModel(priorTurns, reply.model),
        { role: "user", content: text },
      ];
      invoke("send_chat", {
        requestId: reply.requestId,
        model: reply.model,
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
            <div className="model-select" role="group" aria-label="比較するモデル">
              {models.map((m) => (
                <button
                  key={m.name}
                  className={selectedModels.includes(m.name) ? "active" : ""}
                  onClick={() => toggleModel(m.name)}
                  disabled={isStreaming}
                  title={m.parameter_size ?? ""}
                >
                  {m.name}
                </button>
              ))}
            </div>
          </>
        )}
      </header>

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
                      <span className="reply-model">{r.model}</span>
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
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSend();
            }
          }}
          placeholder="プロンプトを入力 (Enterで送信 / Shift+Enterで改行)"
          disabled={status !== "connected" || isStreaming}
        />
        <button
          onClick={handleSend}
          disabled={status !== "connected" || isStreaming || !input.trim() || selectedModels.length === 0}
        >
          {isStreaming ? "生成中…" : `送信 (${selectedModels.length})`}
        </button>
      </footer>
    </div>
  );
}

export default App;
