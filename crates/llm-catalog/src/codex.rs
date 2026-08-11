//! ChatGPT Codex client identity.
//!
//! The Codex backend fingerprints a client by the `originator` it presents.
//! Authorization mints credentials for the originator named in the authorize
//! URL, so every later request — inference, model discovery, and media — must
//! present the same value or the first authenticated call is rejected with a
//! 401. These constants are therefore the single source of that identity:
//! `oauth.toml` carries the only data copy, and
//! [`codex_login_matches_request_identity`] pins the two together.

use crate::oauth_params::OAuthParams;

/// Client identity presented to the Codex backend on every surface.
pub const CODEX_ORIGINATOR: &str = "omp";

/// Pinned Codex client version reported alongside [`CODEX_ORIGINATOR`].
pub const CODEX_CLIENT_VERSION: &str = "0.144.1";

/// Authorization parameter naming the client the credential is minted for.
pub const CODEX_ORIGINATOR_PARAM: &str = "originator";

/// Returns whether a Codex login row authorizes [`CODEX_ORIGINATOR`].
#[must_use]
pub fn codex_login_matches_request_identity(params: &OAuthParams) -> bool {
	params
		.extra_auth_params
		.get(CODEX_ORIGINATOR_PARAM)
		.is_some_and(|value| value == CODEX_ORIGINATOR)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::oauth_params::load_embedded;

	#[test]
	fn every_codex_login_row_authorizes_the_request_originator() {
		let params = load_embedded().expect("bundled OAuth rows");
		let codex = params
			.iter()
			.filter(|row| row.credential_provider.starts_with("openai-codex"))
			.collect::<Vec<_>>();
		assert!(!codex.is_empty(), "bundled catalog must define a Codex login flow");
		for row in codex {
			assert!(
				codex_login_matches_request_identity(row),
				"login row `{}` authorizes a different client than requests present",
				row.provider
			);
		}
	}
}
