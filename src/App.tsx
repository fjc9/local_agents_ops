import { useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type {
  ChatMessage,
  ChatStreamEvent,
  EngineVersion,
  HardwareProfile,
  ModelInfo,
  ModelTarget,
  ProviderInfo,
  RouterDecision,
} from "./types";
import Settings from "./Settings";
import Catalog from "./Catalog";
import "./App.css";

type OllamaStatus = "checking" | "connected" | "unreachable";
type ReplyStatus = "queued" | "streaming" | "done" | "error" | "skipped" | "cancelled";

const ONLINE_PROVIDER_IDS = new Set(["anthropic", "openai", "gemini", "xai"]);
const PARENT_MODEL_STORAGE_KEY = "localAgentsOps.parentModel";
const SERIALIZE_LOCAL_STORAGE_KEY = "localAgentsOps.serializeLocal";
const DISABLED_MODELS_STORAGE_KEY = "localAgentsOps.disabledModels";

/** Mirrors the Rust catalog's estimates so the UI can warn before a
 * selection is sent, rather than after Ollama starts thrashing:
 * loaded weights need meaningfully more RAM than the on-disk size, and
 * the OS plus this app need headroom of their own. */
const RUNTIME_OVERHEAD_MULTIPLIER = 1.6;
const RAM_BUDGET_FRACTION = 0.7;

/** Exchanges of a model's own history to resend each turn.
 *
 * Prompt processing is compute-bound rather than bandwidth-bound, so on a
 * machine without a GPU it costs tens of seconds per thousand tokens -- an
 * uncapped transcript makes every turn slower than the last before the model
 * has generated anything. Applied to online targets too, both to bound paid
 * tokens and to keep the comparison fair: models answering the same question
 * from different amounts of context aren't comparable. Mirrors the bound the
 * router already puts on the history it forwards. */
const MAX_HISTORY_EXCHANGES = 6;

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

/** The user's side of the conversation, which is the same whichever model
 * answered. What the router needs to judge relevance, without inheriting one
 * provider's answers as if they were everyone's. */
function userTurnsOnly(turns: Turn[], maxTurns: number): ChatMessage[] {
  return turns.slice(-maxTurns).map((turn) => ({ role: "user", content: turn.userText }));
}

/** Header text for each reply state. `queued` in particular has to explain
 * itself -- a panel sitting there with no label reads as broken rather than as
 * waiting for its turn. */
function replyStatusLabel(status: ReplyStatus, stillThinking: boolean): string {
  switch (status) {
    case "queued":
      return "待機中（メモリ空き待ち）";
    case "streaming":
      return stillThinking ? "思考中…" : "生成中…";
    case "cancelled":
      return "中断しました";
    case "error":
      return "エラー";
    case "done":
      return "完了";
    default:
      return status;
  }
}

/** Smallest installed model by on-disk size. Used both for the router role
 * and for the initial selection: with no GPU, generation speed is bounded by
 * model size over memory bandwidth, so the smallest model is the only one
 * guaranteed to feel responsive before the user has picked anything. */
function smallestModel(models: ModelInfo[]): ModelInfo | undefined {
  return [...models].sort((a, b) => (a.size_bytes ?? 0) - (b.size_bytes ?? 0))[0];
}

/** Each model gets its own conversation thread: its own past answers only,
 * not what the other models in a parallel-comparison turn said.
 *
 * Replays `sentContent` in place of the raw user text when the router
 * rewrote that turn -- the target's own prior answer was to whatever it
 * actually received, so resending the original would show it a "user" turn
 * that doesn't match what it already replied to.
 *
 * `maxExchanges` bounds how far back that thread reaches, so the prompt --
 * and with it the per-turn prefill cost -- stops growing without limit.
 * Replies that never produced text (skipped, cancelled, failed) are left out
 * rather than sent as empty assistant turns. */
function buildHistoryForTarget(
  turns: Turn[],
  targetId: string,
  maxExchanges = Infinity,
): ChatMessage[] {
  const exchanges: ChatMessage[][] = [];
  for (const turn of turns) {
    const reply = turn.replies.find((r) => r.target.id === targetId);
    if (!reply || !reply.content) continue;
    exchanges.push([
      { role: "user", content: reply.sentContent ?? turn.userText },
      { role: "assistant", content: reply.content },
    ]);
  }
  return exchanges.slice(-maxExchanges).flat();
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
  /** Models the user has explicitly unloaded, hidden from the comparison strip
   * until the next model list refresh. Ejecting is a statement that the model is
   * done for now, and leaving its chip up invites a click that quietly reloads
   * several GB.
   *
   * Kept separate from `models`, which means *installed*: dropping it from there
   * would have the catalog offer to download something already on disk, and
   * remove it from the parent-model choices in settings. */
  const [ejectedModels, setEjectedModels] = useState<Set<string>>(new Set());

  /** Models installed but switched off in the catalog, so they stay off the
   * comparison strip until switched back on.
   *
   * Three different notions of "not using this model" now coexist, and they
   * answer different questions:
   *   - not in `selectedIds`: not part of *this* turn's comparison.
   *   - in `ejectedModels`: free its memory now; back on the next list refresh.
   *   - here: don't offer it at all. Survives restarts.
   * Downloading a model and wanting it in every comparison are separate
   * decisions, which is what this one exists for. */
  const [disabledModels, setDisabledModels] = useState<Set<string>>(() => {
    try {
      const stored = localStorage.getItem(DISABLED_MODELS_STORAGE_KEY);
      return new Set<string>(stored ? JSON.parse(stored) : []);
    } catch {
      return new Set<string>();
    }
  });
  const [parentModel, setParentModel] = useState(() => {
    try {
      return localStorage.getItem(PARENT_MODEL_STORAGE_KEY) ?? "";
    } catch {
      return "";
    }
  });
  const [routing, setRouting] = useState(false);
  /** What the routing step decided, when it's something the user should know
   * about: nothing was sent, or money was spent on a fallback. */
  const [routingNotice, setRoutingNotice] = useState<string | null>(null);
  const [hardware, setHardware] = useState<HardwareProfile | null>(null);
  const [engineVersion, setEngineVersion] = useState<EngineVersion | null>(null);
  const [serializeLocal, setSerializeLocal] = useState(() => {
    try {
      return localStorage.getItem(SERIALIZE_LOCAL_STORAGE_KEY) === "true";
    } catch {
      return false;
    }
  });
  const isComposingRef = useRef(false);

  /** The backend owns the actual behaviour, so push the preference through on
   * every change and once at startup -- a value restored from localStorage that
   * the backend never heard about would be a lie in the settings panel. */
  function updateSerializeLocal(serialize: boolean) {
    setSerializeLocal(serialize);
    try {
      localStorage.setItem(SERIALIZE_LOCAL_STORAGE_KEY, String(serialize));
    } catch {
      // Persistence is a nice-to-have; the in-session setting still applies.
    }
    invoke("set_serialize_local", { serialize }).catch((err) => {
      console.error("failed to apply the serialize setting:", err);
    });
  }

  function toggleModelEnabled(model: string) {
    setDisabledModels((prev) => {
      const next = new Set(prev);
      if (next.has(model)) {
        next.delete(model);
      } else {
        next.add(model);
        // Switching a model off shouldn't leave it queued up for the next send.
        setSelectedIds((ids) => ids.filter((id) => id !== `ollama:${model}`));
      }
      try {
        localStorage.setItem(DISABLED_MODELS_STORAGE_KEY, JSON.stringify([...next]));
      } catch {
        // Persistence is a nice-to-have; the setting still applies this session.
      }
      return next;
    });
  }

  function updateParentModel(model: string) {
    setParentModel(model);
    try {
      localStorage.setItem(PARENT_MODEL_STORAGE_KEY, model);
    } catch {
      // Persistence is a nice-to-have; a blocked/unavailable localStorage
      // (private context, storage quota, etc.) shouldn't break selection.
    }
  }

  // Queued counts as in-flight: the request is accepted and holding a place in
  // the RAM budget, it just hasn't started generating yet.
  const isStreaming =
    turns.length > 0 &&
    turns[turns.length - 1].replies.some((r) => r.status === "streaming" || r.status === "queued");

  const localTargets: ModelTarget[] = useMemo(
    () =>
      models
        .filter((m) => !ejectedModels.has(m.name) && !disabledModels.has(m.name))
        .map((m) => ({
          id: `ollama:${m.name}`,
          provider: "ollama",
          model: m.name,
          label: m.name,
        })),
    [models, ejectedModels, disabledModels],
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

  // Both modifiers work everywhere; the label just follows the local habit.
  const sendShortcutLabel = hardware?.os === "macos" ? "⌘+Enter" : "Ctrl+Enter";

  const availableTargets = onlineMode ? [...localTargets, ...onlineTargets] : localTargets;
  const targetsById = useMemo(() => {
    const map = new Map<string, ModelTarget>();
    for (const t of availableTargets) map.set(t.id, t);
    return map;
  }, [availableTargets]);

  /** Selected chips first, so the strip reflects what the next send will use
   * rather than whatever order Ollama happens to list models in.
   *
   * A stable partition: the underlying order still shows through inside each
   * group. Note that chips do move as you toggle them -- that's the point of
   * ordering by selection, but it means a rapid series of clicks lands on
   * targets that have shifted under the cursor. */
  const orderedTargets = useMemo(() => {
    const selected = new Set(selectedIds);
    return [
      ...availableTargets.filter((t) => selected.has(t.id)),
      ...availableTargets.filter((t) => !selected.has(t.id)),
    ];
  }, [availableTargets, selectedIds]);

  /** The selections that still resolve to something sendable.
   *
   * `selectedIds` can outlive what it points at: turning off online mode, or a
   * model list refresh, leaves ids behind that `handleSend` silently drops.
   * Everything user-facing counts these instead, so the send button can't
   * promise more requests than it will make. */
  const sendableTargets = useMemo(
    () => selectedIds.map((id) => targetsById.get(id)).filter((t): t is ModelTarget => !!t),
    [selectedIds, targetsById],
  );

  /** What the current selection would cost in RAM if Ollama held every one of
   * those models at once, measured against the same budget the Rust catalog
   * uses. Ollama accepts the requests either way and only then starts
   * evicting and reloading, so the warning has to come from here. */
  const memoryPressure = useMemo(() => {
    if (!hardware) return null;
    let onDiskGb = 0;
    let count = 0;
    for (const target of sendableTargets) {
      if (target.provider !== "ollama") continue;
      onDiskGb += (models.find((m) => m.name === target.model)?.size_bytes ?? 0) / 1024 ** 3;
      count += 1;
    }
    const residentGb = onDiskGb * RUNTIME_OVERHEAD_MULTIPLIER;
    const budgetGb = hardware.total_ram_gb * RAM_BUDGET_FRACTION;
    return { count, residentGb, budgetGb, overBudget: residentGb > budgetGb };
  }, [sendableTargets, models, hardware]);

  /** Selected local models that can't think, so 「じっくり」 can say what it
   * will and won't change instead of looking like it applies to everything. */
  const thinkUnsupported = useMemo(
    () =>
      sendableTargets
        .filter((t) => t.provider === "ollama")
        .filter((t) => models.find((m) => m.name === t.model)?.supports_thinking === false)
        .map((t) => t.label),
    [sendableTargets, models],
  );

  function refreshEngineVersion() {
    invoke<EngineVersion>("ollama_version").then(setEngineVersion).catch(() => {});
  }

  function refreshProviders() {
    invoke<ProviderInfo[]>("list_providers").then(setProviders).catch(() => {});
  }

  function refreshModels() {
    invoke<ModelInfo[]>("list_ollama_models")
      .then((list) => {
        setModels(list);
        setStatus("connected");
        // Drop selections whose model is gone, but deliberately don't select
        // freshly pulled ones: a model just downloaded from the catalog is
        // usually the largest thing on the machine, and silently adding it to
        // the comparison set is how you end up loading several GB by accident.
        const available = new Set(list.map((m) => `ollama:${m.name}`));
        setSelectedIds((prev) => prev.filter((id) => !id.startsWith("ollama:") || available.has(id)));
        // An explicit refresh is the user asking to see the installed set again,
        // so previously ejected models come back onto the strip.
        setEjectedModels(new Set());
      })
      .catch(() => setStatus("unreachable"));
  }

  useEffect(() => {
    invoke<ModelInfo[]>("list_ollama_models")
      .then((list) => {
        setModels(list);
        setStatus("connected");
        // Start with a single model rather than every installed one. Models
        // running in parallel share one memory bus, so on a machine without a
        // GPU an all-models default makes the first turn as slow as it can be.
        const smallest = smallestModel(list);
        setSelectedIds(smallest ? [`ollama:${smallest.name}`] : []);
      })
      .catch(() => setStatus("unreachable"));
    refreshProviders();
    invoke<HardwareProfile>("detect_hardware").then(setHardware).catch(() => {});
    refreshEngineVersion();
    // Re-apply the restored preference, which the backend starts without.
    if (serializeLocal) {
      invoke("set_serialize_local", { serialize: true }).catch(() => {});
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
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

      if (payload.type === "queued") {
        updateReplyByRequestId(payload.request_id, (r) => ({ ...r, status: "queued" }));
      } else if (payload.type === "thinking") {
        updateReplyByRequestId(payload.request_id, (r) => ({
          ...r,
          status: "streaming",
          thinking: (r.thinking ?? "") + payload.content,
        }));
      } else if (payload.type === "token") {
        updateReplyByRequestId(payload.request_id, (r) => ({
          ...r,
          status: "streaming",
          content: r.content + payload.content,
        }));
      } else if (payload.type === "cancelled") {
        updateReplyByRequestId(payload.request_id, (r) => ({ ...r, status: "cancelled" }));
      } else if (payload.type === "done") {
        updateReplyByRequestId(payload.request_id, (r) => {
          // A cancelled request can still see a trailing Done from the
          // stream it was reading; don't relabel a stop as a completion.
          if (r.status === "cancelled") return r;
          return !r.content && r.thinking
            ? { ...r, status: "done", content: "（思考の途中で応答が終了しました。もう一度お試しください）" }
            : { ...r, status: "done" };
        });
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
      // Only hide it once the unload actually succeeded -- a chip that vanished
      // while the model was still resident would misreport the machine's state.
      setEjectedModels((prev) => new Set(prev).add(model));
      setSelectedIds((prev) => prev.filter((id) => id !== `ollama:${model}`));
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

  function handleCancel(requestId: string) {
    invoke("cancel_chat", { requestId }).catch((err) => {
      console.error("cancel failed:", err);
    });
  }

  function handleCancelAll() {
    const current = turns[turns.length - 1];
    if (!current) return;
    for (const reply of current.replies) {
      if (reply.status === "streaming" || reply.status === "queued") {
        handleCancel(reply.requestId);
      }
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
    const targets = sendableTargets;
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
    setRoutingNotice(null);

    // Local models always get their own full history, sent immediately --
    // the parent-model routing/compression step is about protecting paid
    // API calls from waste, not about how local models are prompted.
    for (const target of localSelected) {
      const reply = replyByTargetId.get(target.id);
      if (!reply) continue;
      const outgoing: ChatMessage[] = [
        ...buildHistoryForTarget(priorTurns, target.id, MAX_HISTORY_EXCHANGES),
        { role: "user", content: text },
      ];
      sendToTarget(reply, outgoing);
    }

    if (onlineSelected.length === 0) return;

    /** Sends `message` to one online target on top of that target's own thread.
     * The history has to come from the target itself: each provider answered
     * the earlier turns differently, and handing one provider another's replies
     * would make it continue a conversation it never had. */
    const sendOnline = (target: ModelTarget, message: string) => {
      const reply = replyByTargetId.get(target.id);
      if (!reply) return;
      sendToTarget(reply, [
        ...buildHistoryForTarget(priorTurns, target.id, MAX_HISTORY_EXCHANGES),
        { role: "user", content: message },
      ]);
    };

    const sendOnlineDirect = () => {
      for (const target of onlineSelected) sendOnline(target, text);
    };

    if (!parentModel) {
      sendOnlineDirect();
      return;
    }

    setRouting(true);
    try {
      // The user's own turns only. They're identical for every provider, so the
      // routing decision doesn't depend on which provider happens to be first
      // in the selection -- and each provider's own replies reach it directly
      // through `sendOnline` anyway.
      const decision = await invoke<RouterDecision>("route_and_compress", {
        parentModel,
        candidateProviders: onlineSelected.map((t) => t.provider),
        history: userTurnsOnly(priorTurns, MAX_HISTORY_EXCHANGES),
        newMessage: text,
      });

      if (decision.providers.length === 0) {
        setRoutingNotice(
          "🧠 親モデルが、今回はどのオンラインサービスも呼び出す価値がないと判断しました。送信していません。",
        );
      }

      for (const target of onlineSelected) {
        const reply = replyByTargetId.get(target.id);
        if (!reply) continue;
        if (decision.providers.includes(target.provider)) {
          // Only claim the parent model rewrote something when it actually
          // ran, and when the wording actually changed.
          if (!decision.shortcut && decision.compressed_prompt !== text) {
            updateReplyByRequestId(reply.requestId, (r) => ({
              ...r,
              sentContent: decision.compressed_prompt,
            }));
          }
          sendOnline(target, decision.compressed_prompt);
        } else {
          updateReplyByRequestId(reply.requestId, (r) => ({ ...r, status: "skipped" }));
        }
      }
    } catch (err) {
      // The user did pick these providers, so honouring the selection is the
      // right fallback -- but it spends money, so say so rather than only
      // logging it to a console nobody has open.
      console.error("routing failed, falling back to a direct send:", err);
      setRoutingNotice(
        `⚠️ 親モデルによる最適化に失敗したため、選択した${onlineSelected.length}サービスへそのまま送信しました（${String(err)}）`,
      );
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
            {hardware?.os === "windows"
              ? "Ollama未接続 — Ollamaを起動してください（タスクトレイに常駐します）"
              : "Ollama未接続 — ターミナルで `ollama serve` を実行してください"}
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
              {orderedTargets.map((t) => (
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
      {engineVersion && !engineVersion.supported && (
        <div className="online-hint">
          ⚠️ Ollama {engineVersion.version} は古すぎます（必要: {engineVersion.minimum} 以上）。
          じっくりモードとモデルごとの対応判定が正しく動きません。⚙️から更新してください。
        </div>
      )}
      {engineVersion?.supported && engineVersion.update_available && (
        <div className="online-hint">
          Ollama {engineVersion.latest} が利用できます（現在 {engineVersion.version}）。⚙️から更新できます。
        </div>
      )}
      {thinkMode && thinkUnsupported.length > 0 && (
        <div className="online-hint">
          🧠 じっくりモードに対応していないモデルがあります（{thinkUnsupported.join(", ")}）。
          これらは通常の応答になります。
        </div>
      )}
      {memoryPressure?.overBudget && (
        <div className="online-hint">
          ⚠️ 選択中の{memoryPressure.count}モデルを同時に読み込むと約
          {memoryPressure.residentGb.toFixed(1)}GB必要で、この端末の予算
          {memoryPressure.budgetGb.toFixed(1)}GBを超えます。並列実行はメモリ帯域も分割するため、
          モデルを減らしたほうが速く終わります。
        </div>
      )}
      {routing && (
        <div className="online-hint routing-hint">🧠 親モデルが送信内容を最適化中…</div>
      )}
      {routingNotice && <div className="online-hint routing-hint">{routingNotice}</div>}

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
                const inFlight = r.status === "streaming" || r.status === "queued";
                const waiting = inFlight && !r.content && !r.thinking;
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
                        {replyStatusLabel(r.status, stillThinking)}
                      </span>
                      {inFlight && (
                        <button
                          className="unload-button"
                          onClick={() => handleCancel(r.requestId)}
                          title="このモデルの生成を中断"
                        >
                          ⏹
                        </button>
                      )}
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
            // Ctrl+Enter sends (Cmd+Enter too, for the macOS habit); plain Enter
            // inserts a newline.
            //
            // Enter-to-send cannot be made safe alongside an IME: the keypress
            // that confirms a conversion is the same keypress that would submit.
            // Guarding on isComposing, the legacy keyCode 229, a
            // compositionstart/end ref and a grace window after compositionend
            // still let it through -- on Windows compositionend can land before
            // the confirming keydown, at which point every one of those signals
            // reads "not composing" and the keypress is indistinguishable from a
            // deliberate Enter. Moving the shortcut is the only real fix.
            if (e.key !== "Enter") return;
            if (!e.ctrlKey && !e.metaKey) return;
            // Still not mid-conversion, in case an IME maps this combination.
            if (isComposingRef.current || e.nativeEvent.isComposing || e.keyCode === 229) return;
            e.preventDefault();
            handleSend();
          }}
          placeholder={`プロンプトを入力 (${sendShortcutLabel}で送信 / Enterで改行)`}
          disabled={status !== "connected" || isStreaming}
        />
        {isStreaming ? (
          <button onClick={handleCancelAll} title="生成を中断してマシンを解放">
            ⏹ 中断
          </button>
        ) : (
          <button
            onClick={handleSend}
            disabled={status !== "connected" || !input.trim() || sendableTargets.length === 0}
            title={`${sendShortcutLabel}でも送信できます`}
          >
            送信 ({sendableTargets.length})
          </button>
        )}
      </footer>

      {settingsOpen && (
        <Settings
          providers={providers}
          models={models}
          engine={engineVersion}
          onEngineUpdated={() => {
            // The engine restarts, so both the version and what it has loaded
            // are stale.
            refreshEngineVersion();
            refreshModels();
          }}
          hardware={hardware}
          parentModel={parentModel}
          onParentModelChange={updateParentModel}
          serializeLocal={serializeLocal}
          onSerializeLocalChange={updateSerializeLocal}
          onClose={() => setSettingsOpen(false)}
          onChanged={refreshProviders}
        />
      )}

      {catalogOpen && (
        <Catalog
          installedModels={models}
          disabledModels={disabledModels}
          onToggleEnabled={toggleModelEnabled}
          onClose={() => setCatalogOpen(false)}
          onModelsChanged={refreshModels}
        />
      )}
    </div>
  );
}

export default App;
