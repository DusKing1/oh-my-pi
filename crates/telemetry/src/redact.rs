//! Opt-in credential scrubbing with the same token grammar as `pi`.
//!
//! The JavaScript source uses lookbehind and lookahead. Rust's `regex` crate
//! intentionally does not implement lookaround, so matches are found with the
//! unchanged alternation and its two boundary assertions are checked manually.

use std::sync::{
	LazyLock,
	atomic::{AtomicBool, Ordering},
};

use regex::Regex;

const REDACTED: &str = "[REDACTED]";

static CREDENTIAL_REDACTION_ENABLED: AtomicBool = AtomicBool::new(false);
static SENSITIVE_TOKEN_RE: LazyLock<Regex> = LazyLock::new(|| {
	Regex::new(
		r"(?i-u)(gh[opusr]_[a-zA-Z0-9_*]{36,}|github_pat_[a-zA-Z0-9_*]{36,}|glpat-[a-zA-Z0-9_*-]{20,}|sk-proj-[a-zA-Z0-9_*-]{36,}|sk-ant-[a-zA-Z0-9_*-]{36,}|sk-[a-zA-Z0-9_*-]{48,})",
	)
	.expect("the static sensitive-token expression is valid")
});

/// Enables or disables credential redaction.
///
/// This mirrors `configureCredentialRedaction(secrets.enabled)` in `pi` and is
/// deliberately off until the host opts in.
pub fn configure_credential_redaction(enabled: bool) {
	CREDENTIAL_REDACTION_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Returns whether credential redaction is currently enabled.
pub fn credential_redaction_enabled() -> bool {
	CREDENTIAL_REDACTION_ENABLED.load(Ordering::Relaxed)
}

/// Replaces credential-shaped tokens with the literal `[REDACTED]`.
///
/// When redaction is disabled (the default), this skips all regex work.
/// Matching is ASCII-case-insensitive, exactly like the `gi`
/// flags on `pi`'s expression.
pub fn redact_sensitive_credentials(text: &str) -> String {
	if !credential_redaction_enabled() {
		return text.to_owned();
	}

	let mut output = String::new();
	let mut copied_through = 0;
	for candidate in SENSITIVE_TOKEN_RE.find_iter(text) {
		if !has_sensitive_boundaries(text, candidate.start(), candidate.end()) {
			continue;
		}
		output.push_str(&text[copied_through..candidate.start()]);
		output.push_str(REDACTED);
		copied_through = candidate.end();
	}
	if copied_through == 0 {
		return text.to_owned();
	}
	output.push_str(&text[copied_through..]);
	output
}

fn has_sensitive_boundaries(text: &str, start: usize, end: usize) -> bool {
	let left = text[..start].chars().next_back();
	let right = text[end..].chars().next();
	!left.is_some_and(is_sensitive_token_char) && !right.is_some_and(is_sensitive_token_char)
}

const fn is_sensitive_token_char(character: char) -> bool {
	character.is_ascii_alphanumeric() || matches!(character, '_' | '*' | '-')
}

#[cfg(test)]
mod tests {
	use parking_lot::{Mutex, MutexGuard};

	use super::*;

	static TEST_LOCK: Mutex<()> = Mutex::new(());

	fn enabled() -> MutexGuard<'static, ()> {
		let guard = TEST_LOCK.lock();
		configure_credential_redaction(true);
		guard
	}

	#[test]
	fn redaction_is_off_by_default() {
		let _guard = TEST_LOCK.lock();
		configure_credential_redaction(false);
		let token = format!("gho_{}", "A".repeat(36));
		assert_eq!(redact_sensitive_credentials(&token), token);
	}

	#[test]
	fn redacts_every_token_family_case_insensitively() {
		let _guard = enabled();
		for token in [
			format!("gho_{}", "A".repeat(36)),
			format!("ghp_{}", "A".repeat(36)),
			format!("ghu_{}", "A".repeat(36)),
			format!("ghs_{}", "A".repeat(36)),
			format!("ghr_{}", "A".repeat(36)),
			format!("github_pat_{}", "A".repeat(36)),
			format!("glpat-{}", "A".repeat(20)),
			format!("sk-proj-{}", "A".repeat(36)),
			format!("sk-ant-{}", "A".repeat(36)),
			format!("SK-{}", "A".repeat(48)),
		] {
			assert_eq!(redact_sensitive_credentials(&token), REDACTED, "{token}");
		}
	}

	#[test]
	fn embedded_token_replaces_only_the_token() {
		let _guard = enabled();
		let token = format!("github_pat_{}", "Ab1_".repeat(9));
		let text = format!("before: {token}; after");
		assert_eq!(redact_sensitive_credentials(&text), "before: [REDACTED]; after");
	}

	#[test]
	fn short_prefix_is_not_redacted() {
		let _guard = enabled();
		let token = format!("sk-proj-{}", "A".repeat(35));
		assert_eq!(redact_sensitive_credentials(&token), token);
	}

	#[test]
	fn sensitive_left_boundary_prevents_a_match() {
		let _guard = enabled();
		let token = format!("ghp_{}", "A".repeat(36));
		for adjacent in ['a', 'Z', '0', '_', '*', '-'] {
			let text = format!("{adjacent}{token}");
			assert_eq!(redact_sensitive_credentials(&text), text, "{adjacent:?}");
		}
	}

	#[test]
	fn sensitive_right_boundary_prevents_a_match() {
		let _guard = enabled();
		// `gh*` bodies do not consume `-`, while the original lookahead still
		// treats it as a credential character and therefore rejects the match.
		let token = format!("ghp_{}", "A".repeat(36));
		let text = format!("{token}-");
		assert_eq!(redact_sensitive_credentials(&text), text);
	}
}
