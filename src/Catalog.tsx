import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type { CatalogEntry, HardwareProfile, ModelInfo, PullProgressEvent } from "./types";

interface Props {
  installedModels: ModelInfo[];
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

function Catalog({ installedModels, onClose, onModelsChanged }: Props) {
  const [hardware, setHardware] = useState<HardwareProfile | null>(null);
  const [entries, setEntries] = useState<CatalogEntry[]>([]);
  const [pulls, setPulls] = useState<Record<string, PullState>>({});

  const installedTags = new Set(installedModels.map((m) => m.name));

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
            検出したハードウェア: {hardware.os} / 総メモリ約{hardware.total_ram_gb.toFixed(0)}GB
            — このマシンに収まりそうな、出所の異なるモデルを優先しておすすめしています。
          </p>
        )}
        {entries.map((entry) => {
          const installed = installedTags.has(entry.tag);
          const pull = pulls[entry.tag];
          const pct =
            pull?.completed != null && pull?.total ? Math.round((pull.completed / pull.total) * 100) : null;
          return (
            <div key={entry.tag} className="settings-row">
              <div className="settings-row-header">
                <span className="settings-label">
                  {entry.label} <span className="catalog-origin">{entry.origin}</span>
                </span>
                <span className={`settings-status ${installed ? "ok" : ""}`}>
                  {installed ? "導入済み" : `約${entry.size_gb.toFixed(1)}GB`}
                </span>
              </div>
              <p className="catalog-description">{entry.description}</p>
              <div className="settings-row-controls">
                {installed ? (
                  <span className="catalog-installed">✓ 選択候補に表示されます</span>
                ) : pull && !pull.done ? (
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
