import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { openUrl } from "@tauri-apps/plugin-opener";
import type { EngineVersion, UpdateProgressEvent } from "./types";

interface Props {
  engine: EngineVersion | null;
  /** Called after a successful update so the caller can re-read the version and
   * the model list -- the engine restarts, so both are stale. */
  onUpdated: () => void;
}

/** Upgrade output arrives a line at a time and a package manager redrawing a
 * download bar produces a lot of them. Keep the tail; nobody scrolls back
 * through a progress log. */
const MAX_LOG_LINES = 200;

const MANAGER_LABEL: Record<string, string> = {
  winget: "winget",
  homebrew: "Homebrew",
};

/** Ollama version and, when there is one, an in-app path to updating it.
 *
 * The update runs through the OS package manager rather than by downloading an
 * installer here -- see `src-tauri/src/updater.rs` for why. When no package
 * manager is available this falls back to opening the official download page,
 * which is the honest end of the road for an app that won't fetch binaries. */
function EngineUpdate({ engine, onUpdated }: Props) {
  const [running, setRunning] = useState(false);
  const [log, setLog] = useState<string[]>([]);
  const [result, setResult] = useState<{ success: boolean; message: string } | null>(null);
  const logEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const unlisten = listen<UpdateProgressEvent>("ollama-update", (event) => {
      const payload = event.payload;
      if (payload.type === "line") {
        setLog((prev) => [...prev, payload.text].slice(-MAX_LOG_LINES));
      } else {
        setRunning(false);
        setResult({ success: payload.success, message: payload.message });
        if (payload.success) onUpdated();
      }
    });
    return () => {
      unlisten.then((fn) => fn());
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ block: "end" });
  }, [log]);

  async function startUpdate() {
    setRunning(true);
    setLog([]);
    setResult(null);
    try {
      await invoke("update_ollama");
      // The `done` event normally clears this and carries the outcome. Clearing
      // it here too means a lost event leaves the button usable rather than
      // stuck on 更新中… forever.
      setRunning(false);
    } catch (err) {
      // A rejection here means the update never started -- work in flight, or no
      // package manager. The backend's message says which.
      setRunning(false);
      setResult({ success: false, message: String(err) });
    }
  }

  if (!engine) {
    return (
      <div className="settings-row">
        <div className="settings-row-header">
          <span className="settings-label">Ollamaのバージョン</span>
          <span className="settings-status">取得できていません</span>
        </div>
        <p className="settings-note">
          Ollamaに接続できていないため、バージョンを確認できません。
        </p>
      </div>
    );
  }

  const manager = engine.package_manager;

  return (
    <div className="settings-row">
      <div className="settings-row-header">
        <span className="settings-label">Ollamaのバージョン</span>
        <span className={`settings-status ${engine.update_available ? "" : "ok"}`}>
          {engine.version}
          {engine.update_available && engine.latest ? ` → ${engine.latest} が利用可能` : " (最新)"}
        </span>
      </div>

      {engine.latest == null && (
        <p className="settings-note" style={{ margin: "0 0 8px" }}>
          最新版の情報を取得できませんでした（オフラインの可能性があります）。
        </p>
      )}

      {!engine.supported && (
        <p className="settings-error" style={{ margin: "0 0 8px" }}>
          ⚠️ このバージョンは古すぎます（必要: {engine.minimum} 以上）。
          じっくりモードとモデルごとの対応判定が正しく動きません。
        </p>
      )}

      {engine.update_available && (
        <p className="settings-note" style={{ margin: "0 0 8px" }}>
          {manager
            ? `更新は${MANAGER_LABEL[manager] ?? manager}経由で実行します（インストーラを直接ダウンロードせず、署名済みマニフェストで検証された配布物を使います）。`
            : "このマシンには自動更新に使えるパッケージマネージャが見つかりませんでした。公式ページから手動で更新してください。"}
          <br />
          更新中はOllamaが再起動するため、<strong>読み込み済みモデルは解放され、実行中の生成は中断されます</strong>。
          {manager === "winget" &&
            "（winget側のマニフェスト反映が遅れている場合、最新版より少し古いバージョンが入ることがあります）"}
        </p>
      )}

      <div className="settings-row-controls">
        {engine.update_available && manager && (
          <button onClick={startUpdate} disabled={running}>
            {running ? "更新中…" : `${MANAGER_LABEL[manager] ?? manager}で更新する`}
          </button>
        )}
        <button onClick={() => openUrl(engine.download_page)} disabled={running}>
          公式ページを開く
        </button>
      </div>

      {log.length > 0 && (
        <details className="message-thinking" open>
          <summary>更新ログ</summary>
          <div className="message-thinking-content">
            {log.map((line, i) => (
              <div key={i}>{line}</div>
            ))}
            <div ref={logEndRef} />
          </div>
        </details>
      )}

      {result && (
        <p className={result.success ? "settings-note" : "settings-error"} style={{ margin: "8px 0 0" }}>
          {result.success ? "✓ " : "✕ "}
          {result.message}
        </p>
      )}
    </div>
  );
}

export default EngineUpdate;
