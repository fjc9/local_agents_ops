# Windows対応 引き継ぎメモ

macOS (Apple Silicon, 32GB RAM, Ollama + MLXバックエンド) で実装・動作確認したプロジェクトを、Windows (32GB RAM, CPU-only) へ移植するための申し送り事項。

## 構成

- Tauri 2 (Rust) + React/TypeScript。Rust側が `src-tauri/src`、フロントエンドが `src`。
- ローカルLLMはOllama (`http://localhost:11434`) 経由。オンライン4サービス (Claude/ChatGPT/Gemini/Grok) はBYOKで、APIキーはOSの認証情報ストア (`keyring`クレート) に保存。
- 主要機能: 複数ローカルモデルの並列比較、think高速/じっくり切替、オンライン4サービスへのファンアウト、ハードウェア連動のモデルカタログ、明示的unloadボタン、IME対応、Token Killer (親モデルによるオンライン送信の要否判断・圧縮)。

## Windows移植で確認・対応が必要な点

1. **`keyring`クレートの`windows-native`機能**
   `src-tauri/Cargo.toml`で`features = ["apple-native", "windows-native"]`は設定済み。Windows Credential Managerを使う実装のはずだが、実機での動作は未検証。`save_api_key`/`clear_api_key` (Settings画面) で保存・削除・再起動後の永続化を確認すること。

2. **CPU-onlyでの推論速度とモデル推奨ロジック**
   `src-tauri/src/catalog.rs`の`recommend()`はRAM容量だけを見てモデルを推奨しており(`total_ram_gb * 0.7`を予算に、モデルサイズ×1.6が収まるか判定)、CPUかGPU/MLXかは一切考慮していない。32GB RAMなら`mistral-small:22b`(13GB)や`gpt-oss:20b`(13GB)も「収まる」と判定されるが、CPU-onlyだと実用速度が出ない可能性が高い。実機で体感速度を確認し、必要なら軽量モデル寄りに推奨ロジックを調整すること。

3. **Token Killerの親モデル呼び出し速度**
   `src-tauri/src/router.rs`が親モデルに投げるルーティング判断は`think: false`を明示しているが、CPU-onlyだと9B程度のモデルでも数秒〜十数秒かかる可能性がある。体感が悪ければ、より軽量なモデル(`gemma3:4b`など、カタログ上は"router"役として用意済み)を親モデルに選ぶ運用を想定。

4. **パッケージング**
   macOS側は`bundle_dmg.sh`が毎回失敗する既知の問題があるが(`.app`自体は正常動作)、これはDMG固有の問題でWindowsには関係ない。Windowsでは`tauri.conf.json`の`bundle.targets: "all"`によりNSIS/MSIが生成されるはずだが未検証。`npm run tauri build`で問題なくインストーラが出力されるか確認すること。

5. **コード署名**
   macOSは未署名(ad-hoc)のまま運用している。Windowsでも未署名バイナリになるため、初回起動時にSmartScreenの警告が出る想定(想定内の挙動)。社内配布のみであれば許容範囲かどうかは要判断。

6. **IME入力**
   日本語入力Enter誤送信バグは`src/App.tsx`の`onKeyDown`で`isComposingRef` / `e.nativeEvent.isComposing` / `e.keyCode === 229`の三重チェックで対応済み。macOSのWKWebViewでの不具合(変換確定Enterが`isComposing`扱いされない)を踏まえた実装だが、WindowsのWebView2 (Chromium系) では`isComposing`の挙動が異なる可能性があるため、実機のIMEで変換確定Enterが誤送信されないか一度確認すること。

7. **アイコン**
   `src-tauri/icons/icon.ico`は用意済みなので追加対応は不要。

## 既知の未解決事項(Windowsに限らない)

- `bundle_dmg.sh`がmacOSビルドで毎回失敗する(上記4参照、Windowsには影響しない見込み)。
- ClaudeのAPIキーはアカウントの残高不足でエラーになる状態(コード自体は正常、認証も通っている)。ChatGPTのキーは実際に動作確認済み。

## ビルド・起動方法

```
npm install
npm run tauri dev    # 開発時
npm run tauri build  # 配布用ビルド
```

Rustツールチェイン、Node.js、Ollama (`ollama serve`) がそれぞれインストール・起動済みであることが前提。
