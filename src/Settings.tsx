import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import EngineUpdate from "./EngineUpdate";
import type { EngineVersion, HardwareProfile, ModelInfo, ProviderInfo } from "./types";

interface Props {
  providers: ProviderInfo[];
  models: ModelInfo[];
  engine: EngineVersion | null;
  onEngineUpdated: () => void;
  /** Used to name the real credential store rather than guessing, and to
   * estimate what the chosen parent model will cost per turn. */
  hardware: HardwareProfile | null;
  parentModel: string;
  onParentModelChange: (model: string) => void;
  serializeLocal: boolean;
  onSerializeLocalChange: (serialize: boolean) => void;
  onClose: () => void;
  onChanged: () => void;
}

/** Below this, the parent model turns the routing step into the slowest part of
 * the turn. The router prompt runs a few hundred tokens, and prompt processing
 * on a machine without a GPU is compute-bound -- so a large parent model can
 * spend tens of seconds deciding before a single paid request goes out. */
const SLOW_PARENT_TOKENS_PER_SEC = 10;

/** Rough generation speed for one installed model, from the throughput the
 * engine was measured at. null when there's nothing to base it on. */
function estimateTokensPerSec(
  model: ModelInfo | undefined,
  hardware: HardwareProfile | null,
): number | null {
  if (!model?.size_bytes || !hardware || hardware.accelerated) return null;
  if (hardware.observed_gb_per_sec == null) return null;
  return hardware.observed_gb_per_sec / (model.size_bytes / 1024 ** 3);
}

/** Where the `keyring` crate actually puts the secret on each platform.
 * Worth naming precisely rather than saying "stored locally": it's the whole
 * reason a user is willing to paste a paid API key into this box. */
function credentialStoreName(os?: string): string {
  switch (os) {
    case "windows":
      return "Windowsの資格情報マネージャー";
    case "macos":
      return "このMacのKeychain";
    default:
      return "OSの認証情報ストア";
  }
}

function Settings({
  providers,
  models,
  engine,
  onEngineUpdated,
  hardware,
  parentModel,
  onParentModelChange,
  serializeLocal,
  onSerializeLocalChange,
  onClose,
  onChanged,
}: Props) {
  const os = hardware?.os;
  const parentEstimate = estimateTokensPerSec(
    models.find((m) => m.name === parentModel),
    hardware,
  );
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function save(providerId: string) {
    const key = (drafts[providerId] ?? "").trim();
    if (!key) return;
    setBusy(providerId);
    setError(null);
    try {
      await invoke("save_api_key", { provider: providerId, key });
      setDrafts((prev) => ({ ...prev, [providerId]: "" }));
      onChanged();
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(null);
    }
  }

  async function clear(providerId: string) {
    setBusy(providerId);
    setError(null);
    try {
      await invoke("clear_api_key", { provider: providerId });
      onChanged();
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="settings-overlay" onClick={onClose}>
      <div className="settings-panel" onClick={(e) => e.stopPropagation()}>
        <div className="settings-header">
          <span>設定</span>
          <button onClick={onClose}>✕</button>
        </div>

        <EngineUpdate engine={engine} onUpdated={onEngineUpdated} />

        <div className="settings-row">
          <div className="settings-row-header">
            <span className="settings-label">親モデル（オンライン送信の最適化に使用）</span>
          </div>
          <p className="settings-note" style={{ margin: "0 0 8px" }}>
            オンライン送信前に、このモデルが「今回どのサービスを呼ぶ価値があるか」を判断し、送信する質問文を整えます。
            どこにも送る価値がないと判断すれば送信しません。過去のやり取りは各サービスへそのまま送られます。
          </p>
          <select
            value={parentModel}
            onChange={(e) => onParentModelChange(e.target.value)}
            disabled={models.length === 0}
          >
            <option value="">(なし — 最適化せず直接送信)</option>
            {models.map((m) => (
              <option key={m.name} value={m.name}>
                {m.name}
              </option>
            ))}
          </select>
          {parentEstimate != null && parentEstimate < SLOW_PARENT_TOKENS_PER_SEC && (
            <p className="settings-error" style={{ margin: "8px 0 0" }}>
              ⚠️ このモデルは推定 約{parentEstimate.toFixed(0)} tok/s
              で、親モデルには重すぎます。判断のためのプロンプト処理だけで数十秒かかり、
              オンライン送信の前に毎回待たされます。より軽いモデルを選んでください。
            </p>
          )}
        </div>

        <div className="settings-row">
          <div className="settings-row-header">
            <span className="settings-label">ローカルモデルを1つずつ実行</span>
            <span className={`settings-status ${serializeLocal ? "ok" : ""}`}>
              {serializeLocal ? "有効" : "無効"}
            </span>
          </div>
          <p className="settings-note" style={{ margin: "0 0 8px" }}>
            通常は空きメモリの範囲で並列に実行します（実測では2モデル同時のほうが全体は早く終わります）。
            有効にすると1つずつ順番に実行し、最初の回答が最速で読めるようになります。
          </p>
          <div className="settings-row-controls">
            <button onClick={() => onSerializeLocalChange(!serializeLocal)}>
              {serializeLocal ? "並列実行に戻す" : "1つずつ実行する"}
            </button>
          </div>
        </div>
        <p className="settings-note">
          各サービス自身のAPIキーを入力してください。キーは{credentialStoreName(os)}に保存され、外部には送信されません。
        </p>
        {error && <p className="settings-error">{error}</p>}
        {providers.map((p) => (
          <div key={p.id} className="settings-row">
            <div className="settings-row-header">
              <span className="settings-label">{p.label}</span>
              <span className={`settings-status ${p.configured ? "ok" : ""}`}>
                {p.configured ? "設定済み" : "未設定"}
              </span>
            </div>
            <div className="settings-row-controls">
              <input
                type="password"
                placeholder={p.configured ? "新しいキーで上書き" : "APIキーを入力"}
                value={drafts[p.id] ?? ""}
                onChange={(e) => setDrafts((prev) => ({ ...prev, [p.id]: e.target.value }))}
                disabled={busy === p.id}
              />
              <button
                onClick={() => save(p.id)}
                disabled={busy === p.id || !(drafts[p.id] ?? "").trim()}
              >
                保存
              </button>
              {p.configured && (
                <button onClick={() => clear(p.id)} disabled={busy === p.id}>
                  削除
                </button>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default Settings;
