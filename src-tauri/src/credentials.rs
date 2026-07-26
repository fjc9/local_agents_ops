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
