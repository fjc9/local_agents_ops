import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type { CatalogEntry, HardwareProfile, ModelInfo, PullProgressEvent } from "./types";

interface Props {
  installedModels: ModelInfo[];
  /** Installed but switched off, so it stays off the comparison strip. */
  disabledModels: Set<string>;
  onToggleEnabled: (model: string) => void;
  onClose: () => void;
  onModelsChanged: () => void;
}

interface PullState {
  status: string;
  completed: number | null;
  total: number | null;
  done: boolean;
  error?: string;
}

/** Generation speed for an installed model, from the throughput the engine was
 * measured at here. null when there's nothing to base it on -- the same rule the
 * Rust catalog follows, rather than inventing a number for the UI. */
function estimateTokensPerSec(
  model: ModelInfo,
  hardware: HardwareProfile | null,
): number | null {
  if (!model.size_bytes || !hardware || hardware.accelerated) return null;
  if (hardware.observed_gb_per_sec == null) return null;
  return hardware.observed_gb_per_sec / (model.size_bytes / 1024 ** 3);
}

function Catalog({
  installedModels,
  disabledModels,
  onToggleEnabled,
  onClose,
  onModelsChanged,
}: Props) {
  const [hardware, setHardware] = useState<HardwareProfile | null>(null);
  const [entries, setEntries] = useState<CatalogEntry[]>([]);
  const [pulls, setPulls] = useState<Record<string, PullState>>({});

  const installedTags = new Set(installedModels.map((m) => m.name));
  // Recommendations still worth offering, i.e. not already on disk.
  const notInstalled = entries.filter((entry) => !installedTags.has(entry.tag));
  const entryByTag = new Map(entries.map((entry) => [entry.tag, entry]));

  useEffect(() => {
    invoke<HardwareProfile>("detect_hardware").then(setHardware).catch(() => {});
    invoke<CatalogEntry[]>("recommend_models").then(setEntries).catch(() => {});
  }, []);

  useEffect(() => {
    const unlisten = listen<PullProgressEvent>("pull-progress", (event) => {
      const payload = event.payload;
      setPulls((prev) => {
        const next = { ...prev };
        if (payload.type === "progress") {
          next[payload.model] = {
            status: payload.status,
            completed: payload.completed,
            total: payload.total,
            done: false,
          };
        } else if (payload.type === "done") {
          next[payload.model] = {
            status: "done",
            completed: null,
            total: null,
            done: true,
          };
          onModelsChanged();
        } else if (payload.type === "error") {
          next[payload.model] = {
            status: "error",
            completed: null,
            total: null,
            done: true,
            error: payload.message,
          };
        }
        return next;
      });
    });
    return () => {
      unlisten.then((fn) => fn());
    };
  }, [onModelsChanged]);

  function startPull(tag: string) {
    setPulls((prev) => ({ ...prev, [tag]: { status: "starting", completed: null, total: null, done: false } }));
    invoke("pull_model", { model: tag }).catch((err) => {
      setPulls((prev) => ({
        ...prev,
        [tag]: { status: "error", completed: null, total: null, done: true, error: String(err) },
      }));
    });
  }

  return (
    <div className="settings-overlay" onClick={onClose}>
      <div className="settings-panel" onClick={(e) => e.stopPropagation()}>
        <div className="settings-header">
          <span>モデルカタログ</span>
          <button onClick={onClose}>✕</button>
        </div>
        {hardware && (
          <p className="settings-note">
            検出したハードウェア: {hardware.os} / 総メモリ約{hardware.total_ram_gb.toFixed(0)}GB /{" "}
            {hardware.logical_cores}スレッド
            {hardware.accelerated ? " / GPUオフロード検出" : " / GPUオフロードなし"}
            {hardware.observed_gb_per_sec != null &&
              ` / 実測スループット ${hardware.observed_gb_per_sec.toFixed(0)}GB/s`}
            <br />
            {hardware.accelerated
              ? "このマシンに収まる、出所の異なるモデルを優先しておすすめしています。"
              : hardware.observed_gb_per_sec == null
                ? "まだこのマシンでの生成実績がないため速度は未測定です。収まるモデルを軽い順に出しています。一度ローカルモデルに回答させると、以降は実測値で絞り込みます。"
                : "GPUを使っていないため、生成速度は実測スループット÷モデルサイズで決まります。実用速度に届くものだけを、速い順におすすめしています。"}
          </p>
        )}

        <div className="settings-header" style={{ marginTop: 4 }}>
          <span>導入済み（{installedModels.length}）</span>
        </div>
        {installedModels.length === 0 && (
          <p className="settings-note">まだモデルがありません。下からダウンロードしてください。</p>
        )}
        {installedModels.map((model) => {
          const enabled = !disabledModels.has(model.name);
          const entry = entryByTag.get(model.name);
          const estimate = estimateTokensPerSec(model, hardware);
          const sizeGb = model.size_bytes ? model.size_bytes / 1024 ** 3 : null;
          return (
            <div key={model.name} className="settings-row">
              <div className="settings-row-header">
                <span className="settings-label">
                  {model.name}
                  {entry && <span className="catalog-origin"> {entry.origin}</span>}
                </span>
                <span className={`settings-status ${enabled ? "ok" : ""}`}>
                  {enabled ? "使用する" : "使用しない"}
                </span>
              </div>
              <p className="catalog-description">
                {sizeGb != null && `約${sizeGb.toFixed(1)}GB`}
                {model.supports_thinking && " / じっくり対応"}
                {estimate != null && ` / このマシンで推定 約${estimate.toFixed(0)} tok/s`}
                {entry && ` — ${entry.description}`}
              </p>
              <div className="settings-row-controls">
                <button onClick={() => onToggleEnabled(model.name)}>
                  {enabled ? "使用しない" : "使用する"}
                </button>
                {!enabled && (
                  <span className="catalog-installed">比較の選択肢から外しています</span>
                )}
              </div>
            </div>
          );
        })}

        {notInstalled.length > 0 && (
          <div className="settings-header" style={{ marginTop: 4 }}>
            <span>このマシンで動かせるおすすめ（{notInstalled.length}）</span>
          </div>
        )}
        {notInstalled.map((entry) => {
          const pull = pulls[entry.tag];
          const pct =
            pull?.completed != null && pull?.total ? Math.round((pull.completed / pull.total) * 100) : null;
          return (
            <div key={entry.tag} className="settings-row">
              <div className="settings-row-header">
                <span className="settings-label">
                  {entry.label} <span className="catalog-origin">{entry.origin}</span>
                </span>
                <span className="settings-status">約{entry.size_gb.toFixed(1)}GB</span>
              </div>
              <p className="catalog-description">
                {entry.description}
                {entry.est_tokens_per_sec != null && (
                  <>
                    {" — このマシンで推定 約"}
                    {entry.est_tokens_per_sec.toFixed(0)}
                    {" tok/s"}
                    {/* A MoE model reads only its active experts per token, so
                        its speed comes from a much smaller number than its
                        download size suggests. Say so, or the estimate looks
                        like a mistake. */}
                    {entry.active_size_gb < entry.size_gb &&
                      `（MoE: 1トークンあたり約${entry.active_size_gb.toFixed(1)}GBのみ読み出し）`}
                  </>
                )}
              </p>
              <div className="settings-row-controls">
                {pull && !pull.done ? (
                  <span className="catalog-progress">
                    {pull.status}
                    {pct != null ? ` ${pct}%` : ""}
                  </span>
                ) : pull?.error ? (
                  <span className="settings-error">{pull.error}</span>
                ) : (
                  <button onClick={() => startPull(entry.tag)}>ダウンロード</button>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default Catalog;
