//! Persisted application settings.

use std::{fs, io, path::Path};

use serde::{Deserialize, Serialize};

const SETTINGS_FILE: &str = "settings.json";
const SETTINGS_TEMP_FILE: &str = "settings.json.tmp";

/// Persisted user preferences under `<data_dir>/settings.json`.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Settings {
	/// Model key selected as the default for interactive chat.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub default_model: Option<String>,
}

impl Settings {
	/// Loads settings from `data_dir`, falling back to defaults when absent or
	/// corrupt.
	#[must_use]
	pub fn load(data_dir: &Path) -> Self {
		fs::read(data_dir.join(SETTINGS_FILE))
			.ok()
			.and_then(|data| serde_json::from_slice(&data).ok())
			.unwrap_or_default()
	}

	/// Atomically saves settings to `<data_dir>/settings.json`.
	pub fn save(&self, data_dir: &Path) -> io::Result<()> {
		fs::create_dir_all(data_dir)?;
		let data = serde_json::to_vec_pretty(self).map_err(io::Error::other)?;
		let temporary = data_dir.join(SETTINGS_TEMP_FILE);
		fs::write(&temporary, data)?;
		fs::rename(temporary, data_dir.join(SETTINGS_FILE))
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn settings_round_trip() {
		let data_dir = tempfile::tempdir().expect("create temporary data directory");
		let settings = Settings { default_model: Some("anthropic/claude-sonnet-4".to_owned()) };

		settings.save(data_dir.path()).expect("save settings");

		let loaded = Settings::load(data_dir.path());
		assert_eq!(loaded.default_model, settings.default_model);
	}

	#[test]
	fn corrupt_settings_fall_back_to_default() {
		let data_dir = tempfile::tempdir().expect("create temporary data directory");
		fs::write(data_dir.path().join(SETTINGS_FILE), b"not valid json")
			.expect("write corrupt settings");

		let loaded = Settings::load(data_dir.path());
		assert!(loaded.default_model.is_none());
	}
}
