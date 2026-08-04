import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import EngineUpdate from "./EngineUpdate";
import {
  PARAM_DESCRIPTORS,
  formatParamValue,
  hasOverrides,
  type GenerationParamKey,
} from "./generationParams";
import type {
  EngineVersion,
  GenerationParams,
  HardwareProfile,
  ModelInfo,
  ProviderInfo,
} from "./types";

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
  /** Sampling settings keyed by model name. Absent model, or absent key within
   * one, means that knob is on the engine's default. */
  modelParams: Record<string, GenerationParams>;
  onModelParamsChange: (model: string, params: GenerationParams) => void;
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

/** Keeps a typed number inside the knob's usable range. Applied when editing
 * finishes rather than per keystroke: clamping mid-type turns "1" on its way to
 * "16384" into the minimum and eats the rest of the input. */
function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

/** The nine sampling knobs for one chosen local model.
 *
 * Split out from `Settings` because it owns editing state of its own (which
 * model is being tuned, and the half-typed text in each box) that nothing else
 * in the panel needs to know about. */
function GenerationParamsSection({
  models,
  modelParams,
  onModelParamsChange,
}: Pick<Props, "models" | "modelParams" | "onModelParamsChange">) {
  const [target, setTarget] = useState(() => models[0]?.name ?? "");
  /** Text as typed, per knob, so an in-progress "-" or "0." isn't reparsed into
   * something else under the cursor. Dropped once the field is left. */
  const [drafts, setDrafts] = useState<Partial<Record<GenerationParamKey, string>>>({});

  // A model can be uninstalled while this panel is open, and a picker pointing
  // at something gone would silently write settings nothing will ever read.
  useEffect(() => {
    if (models.length === 0) {
      setTarget("");
    } else if (!models.some((m) => m.name === target)) {
      setTarget(models[0].name);
    }
  }, [models, target]);

  const params: GenerationParams = modelParams[target] ?? {};

  function setValue(key: GenerationParamKey, value: number | null) {
    const next = { ...params };
    if (value == null) {
      delete next[key];
    } else {
      next[key] = value;
    }
    onModelParamsChange(target, next);
  }

  function clearDraft(key: GenerationParamKey) {
    setDrafts((prev) => {
      const next = { ...prev };
      delete next[key];
      return next;
    });
  }

  /** Applies what's in the box on the way out: blank means "back to the
   * engine's default", anything unparseable is treated the same way. */
  function finishEditing(key: GenerationParamKey, min: number, max: number) {
    const raw = drafts[key];
    clearDraft(key);
    if (raw === undefined) return;
    const parsed = Number(raw.trim());
    if (raw.trim() === "" || !Number.isFinite(parsed)) {
      setValue(key, null);
    } else {
      setValue(key, clamp(parsed, min, max));
    }
  }

  if (models.length === 0) {
    return (
      <div className="settings-row">
        <div className="settings-row-header">
          <span className="settings-label">回答の生成パラメータ</span>
        </div>
        <p className="settings-note" style={{ margin: 0 }}>
          ローカルモデルが1つも入っていないため、調整できる項目がありません。
          モデルカタログからモデルを追加してください。
        </p>
      </div>
    );
  }

  return (
    <div className="settings-row">
      <div className="settings-row-header">
        <span className="settings-label">回答の生成パラメータ（モデルごと）</span>
        {hasOverrides(modelParams[target]) && (
          <button className="param-reset" onClick={() => onModelParamsChange(target, {})}>
            このモデルを既定に戻す
          </button>
        )}
      </div>
      <p className="settings-note" style={{ margin: "0 0 8px" }}>
        選んだローカルモデルの回答の作り方を調整します。空欄のままの項目はモデル本来の既定値で動きます。
        オンラインの各サービスには適用されません（受け付ける項目が異なり、一部だけ効くと比較が正しく読めなくなるため）。
      </p>
      <select value={target} onChange={(e) => setTarget(e.target.value)}>
        {models.map((m) => (
          <option key={m.name} value={m.name}>
            {m.name}
            {hasOverrides(modelParams[m.name]) ? "（調整済み）" : ""}
          </option>
        ))}
      </select>

      {PARAM_DESCRIPTORS.map((d) => {
        const current = params[d.key];
        const overridden = current != null;
        return (
          <div key={d.key} className="param-row">
            <div className="param-row-header">
              <span className="param-label">
                {d.label}
                <span className="param-english">{d.englishLabel}</span>
              </span>
              <span className={`param-value ${overridden ? "set" : ""}`}>
                {overridden
                  ? formatParamValue(current, d.step)
                  : `既定 ${formatParamValue(d.fallback, d.step)}`}
              </span>
            </div>
            <div className="param-controls">
              <input
                type="range"
                min={d.min}
                max={d.max}
                step={d.step}
                value={current ?? d.fallback}
                onChange={(e) => {
                  clearDraft(d.key);
                  setValue(d.key, Number(e.target.value));
                }}
              />
              <input
                className="param-number"
                type="number"
                min={d.min}
                max={d.max}
                step={d.step}
                placeholder={formatParamValue(d.fallback, d.step)}
                value={drafts[d.key] ?? (current != null ? String(current) : "")}
                onChange={(e) => {
                  const text = e.target.value;
                  setDrafts((prev) => ({ ...prev, [d.key]: text }));
                  const parsed = Number(text.trim());
                  if (text.trim() !== "" && Number.isFinite(parsed)) {
                    setValue(d.key, parsed);
                  }
                }}
                onBlur={() => finishEditing(d.key, d.min, d.max)}
              />
              <button
                className="param-reset"
                onClick={() => {
                  clearDraft(d.key);
                  setValue(d.key, null);
                }}
                disabled={!overridden}
                title="この項目を既定値に戻す"
              >
                既定
              </button>
            </div>
            <p className="param-help">{d.help}</p>
          </div>
        );
      })}
    </div>
  );
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
  modelParams,
  onModelParamsChange,
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

        <GenerationParamsSection
          models={models}
          modelParams={modelParams}
          onModelParamsChange={onModelParamsChange}
        />

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
