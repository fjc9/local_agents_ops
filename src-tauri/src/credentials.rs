use keyring::Entry;

const SERVICE: &str = "com.fjdev.localagentsops";

fn entry_for(provider: &str) -> Result<Entry, String> {
    Entry::new(SERVICE, &format!("api_key_{provider}")).map_err(|e| e.to_string())
}

pub fn save_key(provider: &str, key: &str) -> Result<(), String> {
    entry_for(provider)?.set_password(key).map_err(|e| e.to_string())
}

pub fn get_key(provider: &str) -> Option<String> {
    entry_for(provider).ok()?.get_password().ok()
}

pub fn has_key(provider: &str) -> bool {
    get_key(provider).is_some()
}

pub fn clear_key(provider: &str) -> Result<(), String> {
    let entry = entry_for(provider)?;
    match entry.delete_credential() {
        Ok(()) => Ok(()),
        Err(keyring::Error::NoEntry) => Ok(()),
        Err(e) => Err(e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A provider id no real provider would claim, so a run that dies partway
    /// through can't leave a user's actual key clobbered.
    const TEST_PROVIDER: &str = "__test_provider__";
    const PERSIST_PROVIDER: &str = "__test_persist__";
    const PERSIST_VALUE: &str = "survives-a-process-restart";

    /// Round-trips a value through the real OS credential store. This is the
    /// check that matters when moving to a new platform: `keyring` compiling
    /// says nothing about whether the backing store actually accepts and
    /// returns the secret.
    #[test]
    fn round_trips_through_the_os_credential_store() {
        // Leftover state from an interrupted earlier run would let the
        // assertions below pass for the wrong reason.
        clear_key(TEST_PROVIDER).expect("clear should tolerate a missing entry");
        assert!(!has_key(TEST_PROVIDER));

        save_key(TEST_PROVIDER, "not-a-real-key-0123456789").expect("save_key");
        assert_eq!(
            get_key(TEST_PROVIDER).as_deref(),
            Some("not-a-real-key-0123456789")
        );
        assert!(has_key(TEST_PROVIDER));

        // Settings offers "overwrite with a new key" rather than making the
        // user delete first, so set_password has to replace, not fail.
        save_key(TEST_PROVIDER, "rotated-key").expect("overwrite");
        assert_eq!(get_key(TEST_PROVIDER).as_deref(), Some("rotated-key"));

        clear_key(TEST_PROVIDER).expect("clear_key");
        assert!(!has_key(TEST_PROVIDER));
        // Clicking 削除 twice hits this path, so it must stay a no-op.
        clear_key(TEST_PROVIDER).expect("second clear should be a no-op");
    }

    /// Marks the re-invocation of this test binary as the reading half.
    const CHILD_ENV: &str = "LAO_CREDENTIAL_PERSISTENCE_CHILD";

    /// Cross-process persistence, checked for real: the parent writes a probe,
    /// then re-runs *this same test binary* so a fresh process reads it back.
    /// A value that survives into another process is a value that reached the
    /// OS store rather than any in-process cache, which is what the round-trip
    /// test above cannot show on its own.
    ///
    /// One test rather than a write half and a read half, so it holds up under
    /// `cargo test -- --include-ignored`: as two tests they ran in alphabetical
    /// order, which put the read before the write.
    ///
    ///   cargo test --lib -- --ignored persists_across_processes
    #[test]
    #[ignore]
    fn persists_across_processes() {
        if std::env::var(CHILD_ENV).is_ok() {
            assert_eq!(get_key(PERSIST_PROVIDER).as_deref(), Some(PERSIST_VALUE));
            return;
        }

        clear_key(PERSIST_PROVIDER).expect("start from a clean slate");
        save_key(PERSIST_PROVIDER, PERSIST_VALUE).expect("save_key");

        let exe = std::env::current_exe().expect("path to this test binary");
        let child = std::process::Command::new(exe)
            .args([
                "--ignored",
                "--exact",
                "credentials::tests::persists_across_processes",
            ])
            .env(CHILD_ENV, "1")
            .output();

        // Clean up before asserting, so a failing read doesn't leave the probe
        // behind in the user's credential store.
        clear_key(PERSIST_PROVIDER).expect("cleanup");

        let child = child.expect("re-run the test binary");
        assert!(
            child.status.success(),
            "a fresh process could not read the stored credential:\n{}",
            String::from_utf8_lossy(&child.stdout)
        );
    }
}
