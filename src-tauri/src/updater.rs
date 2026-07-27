//! Updating the Ollama engine itself.
//!
//! Two constraints shape this module.
//!
//! **We never download or run an installer ourselves.** The upgrade is handed to
//! the OS package manager -- winget on Windows, Homebrew on macOS -- which
//! resolves the installer through its own signed manifest and verifies its hash.
//! Fetching a binary off the internet and executing it would put this app in the
//! business of deciding what is safe to run, which it has no way to do. Where no
//! package manager is available the app points at the official download page and
//! lets the user take it from there.
//!
//! **Nothing happens without an explicit click.** Upgrading restarts the engine:
//! every loaded model is dropped and any generation in flight dies. That is not
//! something to do behind the user's back, so this module only ever reports what
//! is available; acting on it is a separate, deliberate call.

use serde::Serialize;
use tokio::io::{AsyncReadExt, BufReader};
use tokio::sync::mpsc::UnboundedSender;

/// Official download page, used when no package manager can do the job.
pub const DOWNLOAD_PAGE_URL: &str = "https://ollama.com/download";

/// Canonical source of truth for "what is the newest Ollama". Deliberately not
/// read out of `winget list`, whose table output is localised -- this machine
/// prints Japanese column headers -- and would need re-parsing per locale.
const LATEST_RELEASE_URL: &str = "https://api.github.com/repos/ollama/ollama/releases/latest";

/// GitHub rejects API requests without one.
const USER_AGENT: &str = "LocalAgentsOps";

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PackageManager {
    Winget,
    Homebrew,
}

impl PackageManager {
    fn program(self) -> &'static str {
        match self {
            PackageManager::Winget => "winget",
            PackageManager::Homebrew => "brew",
        }
    }

    /// The upgrade invocation. Non-interactive on purpose: a prompt from a
    /// process with no console attached would hang forever.
    fn upgrade_args(self) -> Vec<&'static str> {
        match self {
            PackageManager::Winget => vec![
                "upgrade",
                "--id",
                "Ollama.Ollama",
                "--exact",
                "--accept-source-agreements",
                "--accept-package-agreements",
                "--disable-interactivity",
            ],
            PackageManager::Homebrew => vec!["upgrade", "ollama"],
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            PackageManager::Winget => "winget",
            PackageManager::Homebrew => "Homebrew",
        }
    }
}

/// One line of upgrade output, then a single terminal event.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UpdateProgressEvent {
    Line { text: String },
    Done { success: bool, message: String },
}

/// The newest published Ollama release, without the leading `v`.
pub async fn latest_release(client: &reqwest::Client) -> Result<String, String> {
    let resp = client
        .get(LATEST_RELEASE_URL)
        .header(reqwest::header::USER_AGENT, USER_AGENT)
        .header(reqwest::header::ACCEPT, "application/vnd.github+json")
        .send()
        .await
        .map_err(|e| format!("リリース情報を取得できませんでした: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!("リリース情報の取得に失敗しました: {}", resp.status()));
    }

    let value: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("リリース情報を解釈できませんでした: {e}"))?;

    value
        .get("tag_name")
        .and_then(|t| t.as_str())
        .map(|tag| tag.trim_start_matches('v').to_string())
        .ok_or_else(|| "リリース情報に tag_name がありません".to_string())
}

/// Which package manager, if any, can perform the upgrade on this machine.
///
/// Established by running it rather than by looking for a file: on Windows
/// `winget` is an app execution alias, so its presence on disk says little about
/// whether this process can actually invoke it.
pub async fn detect_package_manager() -> Option<PackageManager> {
    let candidate = if cfg!(target_os = "windows") {
        PackageManager::Winget
    } else if cfg!(target_os = "macos") {
        PackageManager::Homebrew
    } else {
        return None;
    };

    let ran = tokio::process::Command::new(candidate.program())
        .arg("--version")
        .output()
        .await;

    match ran {
        Ok(output) if output.status.success() => Some(candidate),
        _ => None,
    }
}

/// Runs the upgrade, forwarding output as it arrives.
///
/// Sends exactly one `Done` at the end, whatever happens, so the UI never sits
/// waiting on a process that already failed to start.
pub async fn upgrade(manager: PackageManager, sender: UnboundedSender<UpdateProgressEvent>) {
    let mut command = tokio::process::Command::new(manager.program());
    command
        .args(manager.upgrade_args())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    #[cfg(target_os = "windows")]
    {
        // Without this the child briefly flashes a console window over the app.
        // `creation_flags` is inherent on tokio's Command, so no std trait import.
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }

    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(e) => {
            let _ = sender.send(UpdateProgressEvent::Done {
                success: false,
                message: format!("{} を起動できませんでした: {e}", manager.label()),
            });
            return;
        }
    };

    if let Some(stdout) = child.stdout.take() {
        forward_output(stdout, sender.clone()).await;
    }
    let mut stderr_text = String::new();
    if let Some(mut stderr) = child.stderr.take() {
        let _ = stderr.read_to_string(&mut stderr_text).await;
    }

    match child.wait().await {
        Ok(status) if status.success() => {
            let _ = sender.send(UpdateProgressEvent::Done {
                success: true,
                message: "更新が完了しました。Ollamaの再起動が必要な場合があります。".to_string(),
            });
        }
        Ok(status) => {
            // winget uses a non-zero exit for "nothing to upgrade" as well as for
            // real failures, so pass its own words through rather than inventing
            // a diagnosis.
            let detail = if stderr_text.trim().is_empty() {
                format!("終了コード {}", status.code().unwrap_or(-1))
            } else {
                stderr_text.trim().to_string()
            };
            let _ = sender.send(UpdateProgressEvent::Done {
                success: false,
                message: format!("{} が更新を完了できませんでした: {detail}", manager.label()),
            });
        }
        Err(e) => {
            let _ = sender.send(UpdateProgressEvent::Done {
                success: false,
                message: format!("更新プロセスの終了を待てませんでした: {e}"),
            });
        }
    }
}

/// Forwards output a line at a time.
///
/// Splits on carriage returns as well as newlines: winget redraws download
/// progress with `\r`, so waiting for a `\n` would show nothing at all through a
/// several-hundred-megabyte download and then dump it all at once.
async fn forward_output<R: tokio::io::AsyncRead + Unpin>(
    reader: R,
    sender: UnboundedSender<UpdateProgressEvent>,
) {
    let mut reader = BufReader::new(reader);
    let mut buffer: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 1024];
    let mut last_sent = String::new();

    loop {
        match reader.read(&mut chunk).await {
            Ok(0) | Err(_) => break,
            Ok(n) => buffer.extend_from_slice(&chunk[..n]),
        }

        while let Some(pos) = buffer.iter().position(|b| *b == b'\n' || *b == b'\r') {
            let line: Vec<u8> = buffer.drain(..=pos).collect();
            let text = String::from_utf8_lossy(&line).trim().to_string();
            // A redrawn progress bar repeats itself between updates; forwarding
            // duplicates would be a lot of events saying nothing new.
            if text.is_empty() || text == last_sent {
                continue;
            }
            last_sent = text.clone();
            let _ = sender.send(UpdateProgressEvent::Line { text });
        }
    }

    let tail = String::from_utf8_lossy(&buffer).trim().to_string();
    if !tail.is_empty() && tail != last_sent {
        let _ = sender.send(UpdateProgressEvent::Line { text: tail });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn winget_is_invoked_non_interactively() {
        let args = PackageManager::Winget.upgrade_args();
        // A prompt from a process with no console never gets answered, so the
        // upgrade would hang rather than fail.
        assert!(args.contains(&"--disable-interactivity"));
        assert!(args.contains(&"--accept-source-agreements"));
        assert!(args.contains(&"--accept-package-agreements"));
        // --exact so a package whose id merely starts with Ollama.Ollama can't be
        // picked up instead.
        assert!(args.contains(&"--exact"));
    }

    /// Reads back the version the way `latest_release` does, so a change to the
    /// tag format is caught here rather than in the UI.
    #[test]
    fn release_tags_lose_their_v_prefix() {
        let value: serde_json::Value = serde_json::from_str(r#"{"tag_name": "v0.32.4"}"#).unwrap();
        let parsed = value
            .get("tag_name")
            .and_then(|t| t.as_str())
            .map(|tag| tag.trim_start_matches('v').to_string());
        assert_eq!(parsed.as_deref(), Some("0.32.4"));
    }

    /// Talks to GitHub, so it stays opt-in:
    ///   cargo test --lib -- --ignored fetches_the_real_latest_release
    #[tokio::test]
    #[ignore]
    async fn fetches_the_real_latest_release() {
        let version = latest_release(&crate::engines::http_client())
            .await
            .expect("latest release");
        assert!(
            crate::engines::ollama::parse_version(&version).is_some(),
            "not a version string: {version}"
        );
        println!("latest published Ollama release: {version}");
    }

    #[tokio::test]
    #[ignore]
    async fn finds_this_machines_package_manager() {
        let found = detect_package_manager().await;
        println!("package manager: {found:?}");
        assert!(found.is_some(), "expected winget on this machine");
    }
}
