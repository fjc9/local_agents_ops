import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import type { ProviderInfo } from "./types";

interface Props {
  providers: ProviderInfo[];
  onClose: () => void;
  onChanged: () => void;
}

function Settings({ providers, onClose, onChanged }: Props) {
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
          <span>APIキー設定</span>
          <button onClick={onClose}>✕</button>
        </div>
        <p className="settings-note">
          各サービス自身のAPIキーを入力してください。キーはこのMac上のKeychainに保存され、外部には送信されません。
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
