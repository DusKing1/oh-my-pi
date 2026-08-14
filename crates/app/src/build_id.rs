//! Build identity of the running executable.
//!
//! Project daemons advertise this identity in their hello frames so clients
//! from a different build can detect a stale daemon and replace it. A content
//! hash — unlike a package version or git revision — changes on every rebuild
//! and is identical for byte-identical binaries regardless of path.

use std::sync::OnceLock;

/// Returns the memoized blake3 content hash of the current executable, or an
/// empty string when the executable cannot be read.
///
/// An empty identity means "unknown": callers must never initiate daemon
/// replacement from an unknown identity, and must treat an empty advertised
/// identity as stale only when their own identity is known.
pub fn current() -> &'static str {
	static BUILD_ID: OnceLock<String> = OnceLock::new();
	BUILD_ID.get_or_init(compute)
}

/// Returns whether a daemon advertising `theirs` should be replaced by a
/// client whose identity is `ours`.
///
/// Replacement requires a known local identity; a daemon with an unknown
/// (empty) identity predates build identification and counts as stale.
#[must_use]
pub fn is_stale(ours: &str, theirs: &str) -> bool {
	!ours.is_empty() && ours != theirs
}

fn compute() -> String {
	std::env::current_exe()
		.and_then(std::fs::read)
		.map(|bytes| blake3::hash(&bytes).to_hex().to_string())
		.unwrap_or_default()
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn current_is_stable_and_hex() {
		let first = current();
		assert_eq!(first, current());
		assert!(!first.is_empty(), "test executable must be readable");
		assert_eq!(first.len(), 64);
		assert!(first.bytes().all(|byte| byte.is_ascii_hexdigit()));
	}

	#[test]
	fn staleness_requires_known_local_identity() {
		assert!(!is_stale("", "abc"));
		assert!(!is_stale("", ""));
		assert!(is_stale("abc", ""));
		assert!(is_stale("abc", "def"));
		assert!(!is_stale("abc", "abc"));
	}
}
