//! OAuth refresh-failure triage.
//!
//! Killing a grant on a transient failure forces a needless interactive
//! re-login; keeping a dead grant alive silently disables the credential.
//! The split: an explicit grant-death token is definitive, a bare `401` is
//! definitive only when nothing in the message smells transient.

use crate::patterns;

/// Verdict on a failed OAuth token refresh.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OAuthFailure {
	/// The stored grant/client is dead (`invalid_grant`, `invalid_token`,
	/// `unauthorized_client`, revoked, refresh token expired). Refresh will
	/// never succeed again; interactive re-login is required.
	Definitive,
	/// Transport/throttle/upstream trouble; the grant may still be valid.
	/// Retry the refresh later.
	Transient,
}

/// Classifies an OAuth refresh error message.
pub fn classify_refresh(message: &str) -> OAuthFailure {
	if patterns::OAUTH_DEFINITIVE.is_match(message) {
		return OAuthFailure::Definitive;
	}
	if patterns::OAUTH_HTTP_401.is_match(message) && !patterns::OAUTH_TRANSIENT.is_match(message) {
		return OAuthFailure::Definitive;
	}
	OAuthFailure::Transient
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn grant_death_tokens() {
		assert_eq!(
			classify_refresh(r#"HTTP 400 invalid_grant {"error":"invalid_grant"}"#),
			OAuthFailure::Definitive
		);
		assert_eq!(
			classify_refresh("OAuth refresh failed: 400 invalid_grant: refresh token revoked"),
			OAuthFailure::Definitive
		);
		assert_eq!(
			classify_refresh("refresh_token expired after rotation"),
			OAuthFailure::Definitive
		);
	}

	#[test]
	fn bare_401_definitive_unless_transient_smell() {
		assert_eq!(classify_refresh("token endpoint returned 401"), OAuthFailure::Definitive);
		assert_eq!(
			classify_refresh("401 behind cloudflare captcha challenge"),
			OAuthFailure::Transient
		);
		assert_eq!(
			classify_refresh("fetch failed: ETIMEDOUT after 401 retry"),
			OAuthFailure::Transient
		);
	}

	#[test]
	fn transport_is_transient() {
		assert_eq!(classify_refresh("fetch failed: ECONNRESET"), OAuthFailure::Transient);
		assert_eq!(classify_refresh("HTTP 503 upstream unavailable"), OAuthFailure::Transient);
	}
}
