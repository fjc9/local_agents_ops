import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type { ChatMessage, ChatStreamEvent, ModelInfo } from "./types";
import "./App.css";

type OllamaStatus = "checking" | "connected" | "unreachable";
type DisplayMessage = ChatMessage & { thinking?: string };

function App() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [status, setStatus] = useState<OllamaStatus>("checking");
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [thinkMode, setThinkMode] = useState(false);

  const activeRequestId = useRef<string | null>(null);

  useEffect(() => {
    invoke<ModelInfo[]>("list_ollama_models")
      .then((list) => {
        setModels(list);
        setStatus("connected");
        if (list.length > 0) setSelectedModel(list[0].name);
      })
      .catch(() => setStatus("unreachable"));
  }, []);

  useEffect(() => {
    const unlisten = listen<ChatStreamEvent>("chat-stream", (event) => {
      const payload = event.payload;
      if (payload.request_id !== activeRequestId.current) return;

      if (payload.type === "thinking") {
        setMessages((prev) => {
          const next = [...prev];
          const last = next[next.length - 1];
          if (last && last.role === "assistant") {
            next[next.length - 1] = {
              ...last,
              thinking: (last.thinking ?? "") + payload.content,
            };
          }
          return next;
        });
      } else if (payload.type === "token") {
        setMessages((prev) => {
          const next = [...prev];
          const last = next[next.length - 1];
          if (last && last.role === "assistant") {
            next[next.length - 1] = {
              ...last,
              content: last.content + payload.content,
            };
          }
          return next;
        });
      } else if (payload.type === "done") {
        setMessages((prev) => {
          const next = [...prev];
          const last = next[next.length - 1];
          if (last && last.role === "assistant" && !last.content && last.thinking) {
            next[next.length - 1] = {
              ...last,
              content: "（思考の途中で応答が終了しました。もう一度お試しください）",
            };
          }
          return next;
        });
        setIsStreaming(false);
        activeRequestId.current = null;
      } else if (payload.type === "error") {
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: `[error: ${payload.message}]` },
        ]);
        setIsStreaming(false);
        activeRequestId.current = null;
      }
    });

    return () => {
      unlisten.then((fn) => fn());
    };
  }, []);

  async function handleSend() {
    const text = input.trim();
    if (!text || !selectedModel || isStreaming) return;

    const requestId = crypto.randomUUID();
    activeRequestId.current = requestId;

    // Only role/content go back to the model — thinking traces are for
    // display only and shouldn't bloat the conversation history we resend.
    const outgoing: ChatMessage[] = [
      ...messages.map(({ role, content }) => ({ role, content })),
      { role: "user", content: text },
    ];
    setMessages([...outgoing, { role: "assistant", content: "" }]);
    setInput("");
    setIsStreaming(true);

    try {
      await invoke("send_chat", {
        requestId,
        model: selectedModel,
        messages: outgoing,
        think: thinkMode,
      });
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `[error: ${String(err)}]` },
      ]);
      setIsStreaming(false);
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
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              disabled={isStreaming}
            >
              {models.map((m) => (
                <option key={m.name} value={m.name}>
                  {m.name}
                  {m.parameter_size ? ` (${m.parameter_size})` : ""}
                </option>
              ))}
            </select>
          </>
        )}
      </header>

      <main className="messages">
        {messages.length === 0 && (
          <div className="empty-state">モデルを選んでメッセージを送信してください</div>
        )}
        {messages.map((m, i) => {
          const isLast = i === messages.length - 1;
          const stillThinking = isStreaming && isLast && !m.content;
          return (
            <div key={i} className={`message message-${m.role}`}>
              <div className="message-role">{m.role}</div>
              {m.thinking && (
                <details className="message-thinking" open={stillThinking}>
                  <summary>{stillThinking ? "思考中…" : "思考の過程"}</summary>
                  <div className="message-thinking-content">{m.thinking}</div>
                </details>
              )}
              <div className="message-content">
                {m.content || (stillThinking && !m.thinking ? "…" : "")}
              </div>
            </div>
          );
        })}
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
          disabled={status !== "connected" || isStreaming || !input.trim()}
        >
          {isStreaming ? "生成中…" : "送信"}
        </button>
      </footer>
    </div>
  );
}

export default App;
