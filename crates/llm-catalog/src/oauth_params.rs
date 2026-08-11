//! Data-driven OAuth and login-flow parameters.
//!
//! The TypeScript implementation accumulated eighteen near-identical provider
//! flow modules whose control flow differed mostly in constants. OMP keeps the
//! three engines (`Pkce`, `DeviceCode`, and `CustomExchange`) in code and
//! describes provider variance as table rows in `oauth.toml`.
//!
//! `CustomExchange` is intentionally narrow.  Codex must recover the `ChatGPT`
//! account id from namespaced JWT claims, GitHub Copilot historically exchanged
//! a GitHub device token for an approximately 25-minute Copilot session token,
//! Z.ai mints a durable API key through its business API, and Cursor, Devin,
//! Perplexity, and paste-key providers use protocols that are not standard
//! OAuth grants.  Those genuinely bespoke steps cannot be represented by
//! endpoint and scope constants alone.

use std::collections::{BTreeMap, BTreeSet};

use omp_core::Str;
use serde::Deserialize;
use smallvec::SmallVec;
use thiserror::Error;

use crate::provider::{AuthSpec, ProviderCatalog};

/// The reusable authorization engine selected by a provider row.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum FlowKind {
	/// Authorization-code flow with PKCE and a loopback callback.
	Pkce,
	/// RFC 8628-style device authorization and polling, without a callback
	/// listener.
	DeviceCode,
	/// A provider-specific exchange or interactive credential acquisition.
	CustomExchange,
}

/// Provider-specific work that cannot be expressed as a standard OAuth grant.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum CustomExchange {
	/// Decode Codex JWT claims and retain the `ChatGPT` account/workspace id.
	OpenAiCodexClaims,
	/// Exchange a GitHub token for Copilot's short-lived (roughly 25-minute)
	/// session token.
	GithubCopilotSessionToken,
	/// Poll Cursor's `auth/poll` endpoint and exchange its user API key on
	/// refresh.
	CursorPoll,
	/// Turn Z.ai OAuth and business-login tokens into a durable `id.secret` API
	/// key.
	ZaiApiKey,
	/// Exchange Devin's callback code and PKCE verifier using its JSON CLI
	/// endpoint.
	DevinCliToken,
	/// Acquire and refresh Perplexity's JWT with email OTP and Socket.IO.
	PerplexityEmailOtp,
	/// Ask the user to paste an API key obtained from the provider's web
	/// console.
	ApiKeyPaste,
	/// Complete PKCE through a registered non-HTTP redirect and pasted callback
	/// URL.
	ExternalRedirectPkce,
}

/// One provider's data-only OAuth or login-flow configuration.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, bon::Builder)]
#[non_exhaustive]
pub struct OAuthParams {
	/// Stable flow identifier used by provider authentication policy.
	pub provider:            Str,
	/// Canonical provider row that stores and consumes credentials from this
	/// flow.
	pub credential_provider: Str,
	/// Engine choice; Cursor and Z.ai forced the explicit custom-exchange case.
	pub kind:                FlowKind,
	/// Public OAuth application id; empty only for non-OAuth paste-key/custom
	/// flows.
	pub client_id:           Str,
	/// Browser authorization or device-code initiation URL.
	pub authorize_url:       Str,
	/// Token endpoint, or the bespoke exchange endpoint for custom flows.
	pub token_url:           Str,
	/// Requested grants; Google requires more than the common four-scope inline
	/// case.
	#[serde(default)]
	pub scopes:              SmallVec<Str, 4>,
	/// Preferred loopback port; fixed allowlists force ports such as Codex's
	/// 1455.
	pub callback_port:       Option<u16>,
	/// Provider-only authorization parameters such as Google's
	/// `access_type=offline`.
	#[serde(default)]
	pub extra_auth_params:   BTreeMap<Str, Str>,
	/// Bespoke post-authorization operation; required by
	/// [`FlowKind::CustomExchange`].
	pub exchange:            Option<CustomExchange>,
}

/// Failure to parse or validate an OAuth parameter table.
#[derive(Debug, Error)]
pub enum OAuthParamsError {
	/// The TOML document is malformed or has a field of the wrong type.
	#[error("invalid OAuth parameter TOML: {0}")]
	Toml(#[from] toml::de::Error),
	/// Two rows use the same provider id.
	#[error("duplicate OAuth provider `{0}`")]
	DuplicateProvider(Str),
	/// A row violates the requirements of its selected flow engine.
	#[error("invalid OAuth provider `{provider}`: {detail}")]
	InvalidFlow {
		/// Provider id of the invalid row.
		provider: Str,
		/// Human-readable invariant that was violated.
		detail:   &'static str,
	},
}

/// Failure to join provider authentication policy to the OAuth flow table.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum OAuthLinkError {
	/// A provider references a flow absent from `oauth.toml`.
	#[error("provider `{provider}` references unknown OAuth flow `{flow}`")]
	UnknownFlow {
		/// Provider catalog identifier.
		provider: Str,
		/// Missing flow identifier.
		flow:     Str,
	},
	/// A provider's request-auth flow and supplemental login flow disagree.
	#[error("provider `{provider}` has conflicting OAuth flows `{auth_flow}` and `{login_flow}`")]
	ConflictingFlow {
		/// Provider catalog identifier.
		provider:   Str,
		/// Flow selected by [`AuthSpec::OAuth`].
		auth_flow:  Str,
		/// Supplemental login flow on the provider row.
		login_flow: Str,
	},
	/// One or more flow rows have no provider reference.
	#[error("orphan OAuth flows: {0:?}")]
	OrphanFlows(Box<[Str]>),
	/// A flow names a canonical provider absent from the provider catalog.
	#[error("OAuth flow `{flow}` names missing credential provider `{provider}`")]
	MissingCredentialProvider {
		/// OAuth flow identifier.
		flow:     Str,
		/// Missing provider identifier.
		provider: Str,
	},
	/// A flow's canonical provider does not point back to it.
	#[error("OAuth flow `{flow}` is not referenced by credential provider `{provider}`")]
	UnlinkedCredentialProvider {
		/// OAuth flow identifier.
		flow:     Str,
		/// Provider identifier.
		provider: Str,
	},
}

#[derive(Deserialize)]
struct OAuthFile {
	provider: Vec<OAuthParams>,
}

/// Parse and validate an `oauth.toml` document.
///
/// # Errors
///
/// Returns [`OAuthParamsError`] for malformed TOML, duplicate provider ids, or
/// a row that does not satisfy its [`FlowKind`] invariants.
pub fn load(input: &str) -> Result<Box<[OAuthParams]>, OAuthParamsError> {
	let file: OAuthFile = toml::from_str(input)?;
	let mut seen = BTreeSet::new();
	for params in &file.provider {
		if !seen.insert(params.provider.clone()) {
			return Err(OAuthParamsError::DuplicateProvider(params.provider.clone()));
		}
		validate(params)?;
	}
	Ok(file.provider.into_boxed_slice())
}

/// Load the table shipped with `omp-llm-catalog`.
///
/// # Errors
///
/// Returns [`OAuthParamsError`] if the embedded table is malformed or invalid.
pub fn load_embedded() -> Result<Box<[OAuthParams]>, OAuthParamsError> {
	load(include_str!("../oauth.toml"))
}

/// Find a provider row by its stable id.
#[must_use]
pub fn lookup<'a>(params: &'a [OAuthParams], provider: &str) -> Option<&'a OAuthParams> {
	params.iter().find(|params| params.provider == provider)
}

/// Validates the complete provider-to-flow join.
///
/// Both request-auth OAuth and supplemental login flows count as references.
/// The latter permits a provider to retain API-key environment fallback while
/// also accepting a broker-managed login. Every reference must resolve and
/// every flow row must have at least one referencing provider.
///
/// # Errors
///
/// Returns [`OAuthLinkError`] for a missing, conflicting, or orphaned flow.
pub fn validate_provider_links(
	providers: &ProviderCatalog,
	params: &[OAuthParams],
) -> Result<(), OAuthLinkError> {
	for row in params {
		let Some(provider) = providers.get(row.credential_provider.as_str()) else {
			return Err(OAuthLinkError::MissingCredentialProvider {
				flow:     row.provider.clone(),
				provider: row.credential_provider.clone(),
			});
		};
		let auth_matches =
			matches!(&provider.auth, AuthSpec::OAuth { flow } if flow == &row.provider);
		if !auth_matches && provider.oauth_flow.as_ref() != Some(&row.provider) {
			return Err(OAuthLinkError::UnlinkedCredentialProvider {
				flow:     row.provider.clone(),
				provider: row.credential_provider.clone(),
			});
		}
	}

	let known: BTreeSet<&str> = params.iter().map(|row| row.provider.as_str()).collect();
	let mut referenced = BTreeSet::new();

	for provider in providers.values() {
		let auth_flow = match &provider.auth {
			AuthSpec::OAuth { flow } => Some(flow),
			_ => None,
		};
		if let (Some(auth_flow), Some(login_flow)) = (auth_flow, provider.oauth_flow.as_ref()) {
			if auth_flow != login_flow {
				return Err(OAuthLinkError::ConflictingFlow {
					provider:   provider.id.clone(),
					auth_flow:  auth_flow.clone(),
					login_flow: login_flow.clone(),
				});
			}
		}

		for flow in [auth_flow, provider.oauth_flow.as_ref()]
			.into_iter()
			.flatten()
		{
			if !known.contains(flow.as_str()) {
				return Err(OAuthLinkError::UnknownFlow {
					provider: provider.id.clone(),
					flow:     flow.clone(),
				});
			}
			referenced.insert(flow.as_str());
		}
	}

	let orphans: Box<[Str]> = params
		.iter()
		.filter(|row| !referenced.contains(row.provider.as_str()))
		.map(|row| row.provider.clone())
		.collect();
	if orphans.is_empty() {
		Ok(())
	} else {
		Err(OAuthLinkError::OrphanFlows(orphans))
	}
}

fn validate(params: &OAuthParams) -> Result<(), OAuthParamsError> {
	let detail = match params.kind {
		FlowKind::Pkce if params.callback_port.is_none() => Some("PKCE requires a callback port"),
		FlowKind::DeviceCode if params.callback_port.is_some() => {
			Some("device-code must not bind a callback port")
		},
		FlowKind::CustomExchange if params.exchange.is_none() => {
			Some("custom-exchange requires an exchange")
		},
		_ => None,
	};
	match detail {
		Some(detail) => {
			Err(OAuthParamsError::InvalidFlow { provider: params.provider.clone(), detail })
		},
		None => Ok(()),
	}
}

#[cfg(test)]
mod tests {
	use super::{FlowKind, load, load_embedded, lookup};

	#[test]
	fn shipped_table_parses_and_satisfies_flow_invariants() {
		let params = load_embedded().expect("shipped oauth.toml must parse");
		assert_eq!(params.len(), 18);
		for row in &params {
			match row.kind {
				FlowKind::Pkce => assert!(row.callback_port.is_some()),
				FlowKind::DeviceCode => assert!(row.callback_port.is_none()),
				FlowKind::CustomExchange => assert!(row.exchange.is_some()),
			}
		}
	}

	#[test]
	fn lookup_uses_provider_id() {
		let params = load_embedded().expect("shipped oauth.toml must parse");
		let codex = lookup(&params, "openai-codex").expect("Codex row must exist");
		assert_eq!(codex.callback_port, Some(1455));
		assert!(lookup(&params, "not-a-provider").is_none());
	}

	#[test]
	fn rejects_flow_kind_mismatch() {
		let invalid = r#"
[[provider]]
provider = "broken"
credential_provider = "broken"
kind = "pkce"
client_id = "client"
authorize_url = "https://example.test/authorize"
token_url = "https://example.test/token"
scopes = []
extra_auth_params = {}
"#;
		assert!(load(invalid).is_err());
	}
}
