import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type { ChatMessage, ChatStreamEvent, ModelInfo } from "./types";
import "./App.css";

type OllamaStatus = "checking" | "connected" | "unreachable";

function App() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [status, setStatus] = useState<OllamaStatus>("checking");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);

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

      if (payload.type === "token") {
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

    const outgoing: ChatMessage[] = [...messages, { role: "user", content: text }];
    setMessages([...outgoing, { role: "assistant", content: "" }]);
    setInput("");
    setIsStreaming(true);

    try {
      await invoke("send_chat", {
        requestId,
        model: selectedModel,
        messages: outgoing,
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
        )}
      </header>

      <main className="messages">
        {messages.length === 0 && (
          <div className="empty-state">モデルを選んでメッセージを送信してください</div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`message message-${m.role}`}>
            <div className="message-role">{m.role}</div>
            <div className="message-content">
              {m.content || (isStreaming && i === messages.length - 1 ? "…" : "")}
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
          disabled={status !== "connected" || isStreaming || !input.trim()}
        >
          {isStreaming ? "生成中…" : "送信"}
        </button>
      </footer>
    </div>
  );
}

export default App;
