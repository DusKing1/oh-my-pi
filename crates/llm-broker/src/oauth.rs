//! Data-driven OAuth login and refresh engines.
//!
//! Provider constants live in `omp-llm-catalog`'s OAuth parameter table. This
//! module contains only the three reusable control flows: PKCE, device code,
//! and the small set of genuinely custom exchanges named by that table.

use std::{
	collections::HashMap,
	future::Future,
	pin::Pin,
	sync::Arc,
	time::{Duration, SystemTime, UNIX_EPOCH},
};

use bytes::Bytes;
use http::{
	Method,
	header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue, USER_AGENT},
};
use omp_core::{Str, USER_AGENT as OMP_USER_AGENT};
use omp_llm_catalog::oauth_params::{self, CustomExchange, FlowKind, OAuthParams};
use parking_lot::Mutex;
use serde_json::{Map, Value};
use tokio::{
	io::{AsyncReadExt, AsyncWriteExt},
	net::TcpListener,
};
use url::Url;

use crate::{
	sealed::Secret,
	store::{CredentialMeta, Store, StoreError},
};

const DEFAULT_DEVICE_INTERVAL_SECS: u64 = 5;
const COPILOT_TOKEN_URL: &str = "https://api.github.com/copilot_internal/v2/token";
const CALLBACK_LIMIT: usize = 16 * 1024;
const IPV6_COMPANION_ATTEMPTS: usize = 4;
const FORM_CONTENT_TYPE: &str = "application/x-www-form-urlencoded";
const JSON_CONTENT_TYPE: &str = "application/json";
const CURSOR_MAX_POLLS: u16 = 150;
const CURSOR_INITIAL_INTERVAL_MS: u64 = 1_000;
const CURSOR_MAX_INTERVAL_MS: u64 = 10_000;
const CUSTOM_FLOW_TIMEOUT_MS: u64 = 5 * 60 * 1_000;
const CURSOR_TIMEOUT_MS: u64 = 25 * 60 * 1_000;
const NEVER_EXPIRES_MS: u64 = 8_640_000_000_000_000;

/// Boxed future returned by the injected OAuth HTTP transport.
pub type HttpFuture<'a> =
	Pin<Box<dyn Future<Output = Result<HttpResponse, HttpError>> + Send + 'a>>;

/// An OAuth HTTP request passed to the gateway-owned egress stack.
///
/// This type intentionally has no `Debug` implementation: token and refresh
/// requests contain secret material in their headers or body.
pub struct HttpRequest {
	/// HTTP method.
	pub method:  Method,
	/// Absolute destination URL.
	pub url:     Str,
	/// Request headers.
	pub headers: HeaderMap,
	/// Complete request body.
	pub body:    Bytes,
}

/// A complete response returned by the injected OAuth HTTP transport.
pub struct HttpResponse {
	/// HTTP status code.
	pub status: u16,
	/// Complete response body.
	pub body:   Bytes,
}

/// Transport failure reported by the injected OAuth HTTP implementation.
#[derive(Clone, Debug, thiserror::Error)]
#[error("OAuth transport failed: {detail}")]
pub struct HttpError {
	/// Client-safe failure detail.
	pub detail:    Str,
	/// Whether retrying the same operation may succeed.
	pub transient: bool,
}

/// Minimal HTTP capability required by the OAuth engines.
pub trait HttpClient: Send + Sync {
	/// Executes one cancellable HTTP request.
	fn execute(&self, request: HttpRequest) -> HttpFuture<'_>;
}

/// Optional desktop browser integration.
pub trait BrowserOpener: Send + Sync {
	/// Opens the authorization URL in the user's browser.
	fn open(&self, url: &str) -> Result<(), Str>;
}

/// Injectable time and sleeping used by device-code polling.
pub trait TimeSource: Send + Sync {
	/// Returns Unix epoch milliseconds.
	fn now_ms(&self) -> u64;
	/// Sleeps without blocking an executor thread.
	fn sleep(&self, duration: std::time::Duration) -> Pin<Box<dyn Future<Output = ()> + Send + '_>>;
}

struct SystemTimeSource;

impl TimeSource for SystemTimeSource {
	fn now_ms(&self) -> u64 {
		SystemTime::now()
			.duration_since(UNIX_EPOCH)
			.unwrap_or_default()
			.as_millis()
			.try_into()
			.unwrap_or(u64::MAX)
	}

	fn sleep(&self, duration: std::time::Duration) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
		Box::pin(tokio::time::sleep(duration))
	}
}

/// Initial instruction returned to a login client.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoginStart {
	/// Opaque id used for `submit_code` or `wait_login`.
	pub flow_id:       Str,
	/// Provider catalog id.
	pub provider:      Str,
	/// Interaction the client should present.
	pub prompt:        LoginPrompt,
	/// Absolute flow expiry in Unix epoch milliseconds, or zero when
	/// unspecified.
	pub expires_at_ms: u64,
}

/// User interaction required by an OAuth flow.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LoginPrompt {
	/// Open a browser and complete an authorization redirect.
	Browse {
		/// Fully parameterized authorization URL.
		url:      Str,
		/// Whether the broker successfully bound the configured loopback port.
		loopback: bool,
	},
	/// Enter a device code at the provider's verification page.
	Device {
		/// Short code shown to the user.
		user_code:        Str,
		/// Browser verification URL.
		verification_url: Str,
		/// Initial provider-requested polling interval.
		interval_secs:    u64,
	},
	/// Paste a code, callback URL, API key, or documented structured prompt
	/// value.
	Paste {
		/// Page at which the value is acquired.
		url: Str,
	},
}

/// Failures produced by an OAuth flow.
#[derive(Debug, thiserror::Error)]
pub enum OAuthError {
	/// The shipped parameter table is invalid.
	#[error("invalid OAuth parameter table: {0}")]
	Params(#[from] oauth_params::OAuthParamsError),
	/// No table row exists for the provider.
	#[error("OAuth provider is not configured")]
	UnknownProvider,
	/// The flow id is unknown or has already completed.
	#[error("OAuth flow `{0}` is not pending")]
	UnknownFlow(Str),
	/// A method was used with the wrong flow kind.
	#[error("OAuth flow `{0}` does not support this operation")]
	WrongFlow(Str),
	/// A URL in the parameter table or callback is invalid.
	#[error("invalid OAuth URL: {0}")]
	InvalidUrl(Str),
	/// The loopback callback was malformed.
	#[error("invalid OAuth callback: {0}")]
	InvalidCallback(Str),
	/// The callback state did not match the initiating request.
	#[error("OAuth state mismatch")]
	StateMismatch,
	/// The provider rejected an OAuth operation.
	#[error("OAuth provider error `{code}` (HTTP {status})")]
	Provider {
		/// OAuth error code, without provider response details that may contain
		/// secrets.
		code:      Str,
		/// HTTP status, or 200 for an OAuth error in a success response.
		status:    u16,
		/// Whether retrying may succeed.
		transient: bool,
	},
	/// The injected HTTP stack failed.
	#[error(transparent)]
	Transport(#[from] HttpError),
	/// A provider response was not valid for the selected flow.
	#[error("invalid OAuth response: {0}")]
	InvalidResponse(Str),
	/// A bounded interactive or polling exchange expired.
	#[error("OAuth login expired")]
	LoginExpired,
	/// Credential persistence failed.
	#[error(transparent)]
	Store(#[from] StoreError),
}

impl OAuthError {
	const fn is_transient(&self) -> bool {
		match self {
			Self::Transport(error) => error.transient,
			Self::Provider { transient, .. } => *transient,
			_ => false,
		}
	}
}

/// Data-driven OAuth flow orchestrator.
pub struct OAuthEngine {
	store:   Arc<Store>,
	http:    Arc<dyn HttpClient>,
	time:    Arc<dyn TimeSource>,
	browser: Option<Arc<dyn BrowserOpener>>,
	params:  Box<[OAuthParams]>,
	pending: Mutex<HashMap<Str, PendingFlow>>,
}

impl OAuthEngine {
	/// Creates an engine using the shipped provider table and system clock.
	///
	/// # Errors
	///
	/// Returns an error if the embedded parameter table is invalid.
	pub fn new(store: Arc<Store>, http: Arc<dyn HttpClient>) -> Result<Self, OAuthError> {
		Self::with_time_source(store, http, Arc::new(SystemTimeSource))
	}

	/// Creates an engine with an injected clock, primarily for deterministic
	/// polling.
	///
	/// # Errors
	///
	/// Returns an error if the embedded parameter table is invalid.
	pub fn with_time_source(
		store: Arc<Store>,
		http: Arc<dyn HttpClient>,
		time: Arc<dyn TimeSource>,
	) -> Result<Self, OAuthError> {
		Ok(Self {
			store,
			http,
			time,
			browser: None,
			params: oauth_params::load_embedded()?,
			pending: Mutex::new(HashMap::new()),
		})
	}

	/// Adds optional best-effort desktop browser opening.
	#[must_use]
	pub fn with_browser(mut self, browser: Arc<dyn BrowserOpener>) -> Self {
		self.browser = Some(browser);
		self
	}

	/// Returns the configured provider identifiers in catalog order.
	pub fn providers(&self) -> impl Iterator<Item = &str> {
		self.params.iter().map(|params| params.provider.as_str())
	}

	/// Begins the provider's configured login flow.
	///
	/// PKCE binds the configured loopback address before returning. If binding
	/// is unavailable, the same flow remains usable through `submit_code`.
	///
	/// # Errors
	///
	/// Returns an error for an unknown provider, malformed response, or HTTP
	/// failure.
	pub async fn begin_login(&self, provider: &str, now_ms: u64) -> Result<LoginStart, OAuthError> {
		let params = self.params(provider)?.clone();
		let flow_id = random_urlsafe(24);
		let (prompt, expires_at_ms, pending) = match params.kind {
			FlowKind::Pkce => self.begin_pkce(&params, now_ms).await?,
			FlowKind::DeviceCode => self.begin_device(&params, now_ms).await?,
			FlowKind::CustomExchange => self.begin_custom(&params, now_ms).await?,
		};
		self.pending.lock().insert(flow_id.clone(), pending);
		Ok(LoginStart { flow_id, provider: params.provider, prompt, expires_at_ms })
	}

	/// Completes a PKCE or paste-driven flow with user-provided input.
	///
	/// The sealed `code` may contain a raw authorization code, callback URL, or
	/// custom-exchange input. A non-empty explicit `state` is always verified;
	/// callback URLs always have their embedded state verified. Raw-code
	/// submission with an empty state is the no-loopback paste fallback.
	///
	/// # Errors
	///
	/// Returns an error for an unknown flow, state mismatch, exchange failure,
	/// or credential-store failure.
	pub async fn submit_code(
		&self,
		flow_id: &str,
		code: &Secret,
		state: &str,
		now_ms: u64,
	) -> Result<CredentialMeta, OAuthError> {
		let code = secret_str(code)?;
		let pending = self.take_pending(flow_id)?;
		match pending {
			PendingFlow::Pkce(pkce) => {
				let (code, callback_state) = parse_pasted_code(code, pkce.state.as_str())?;
				let supplied_state = callback_state.as_deref().unwrap_or(state);
				if !supplied_state.is_empty() && supplied_state != pkce.state {
					return Err(OAuthError::StateMismatch);
				}
				self.exchange_pkce(pkce, &code, now_ms).await
			},
			PendingFlow::Paste { params } => {
				if params.exchange == Some(CustomExchange::PerplexityEmailOtp) {
					self.exchange_perplexity(&params, code, now_ms).await
				} else {
					let secret = Secret::new(code.trim().as_bytes());
					if secret.expose().is_empty() {
						return Err(OAuthError::InvalidResponse("empty API key".into()));
					}
					self.persist_api_key(&params, secret, now_ms)
				}
			},
			PendingFlow::Device(_) | PendingFlow::Cursor(_) => {
				Err(OAuthError::WrongFlow(flow_id.into()))
			},
		}
	}

	/// Waits for a loopback PKCE callback or polls a device-code grant.
	///
	/// The broker, not the RPC client, owns device polling. Dropping this future
	/// structurally cancels the outstanding HTTP request or listener wait.
	///
	/// # Errors
	///
	/// Returns an error for an unknown flow, state mismatch, expiry, provider
	/// rejection, or credential-store failure.
	pub async fn wait_login(
		&self,
		flow_id: &str,
		_now_ms: u64,
	) -> Result<CredentialMeta, OAuthError> {
		let pending = self.take_pending(flow_id)?;
		match pending {
			PendingFlow::Pkce(pkce) => {
				let listener = pkce
					.listener
					.as_ref()
					.ok_or_else(|| OAuthError::WrongFlow(flow_id.into()))?;
				let callback = receive_callback(listener, pkce.state.as_str());
				let (code, state) = if pkce.expires_at_ms == 0 {
					callback.await?
				} else {
					let remaining = pkce.expires_at_ms.saturating_sub(self.time.now_ms());
					if remaining == 0 {
						return Err(OAuthError::LoginExpired);
					}
					tokio::time::timeout(Duration::from_millis(remaining), callback)
						.await
						.map_err(|_| OAuthError::LoginExpired)??
				};
				debug_assert_eq!(state, pkce.state);
				self.exchange_pkce(pkce, &code, self.time.now_ms()).await
			},
			PendingFlow::Device(device) => self.poll_device(device).await,
			PendingFlow::Cursor(cursor) => self.poll_cursor(cursor).await,
			PendingFlow::Paste { .. } => Err(OAuthError::WrongFlow(flow_id.into())),
		}
	}

	/// Imports OAuth material already acquired by a trusted broker RPC.
	///
	/// # Errors
	///
	/// Returns an error for an unknown provider or persistence failure.
	pub(crate) fn import_oauth(
		&self,
		provider: &str,
		identity: &str,
		access: &Secret,
		refresh: Option<&Secret>,
		props: &Value,
		expires_at_ms: u64,
		now_ms: u64,
	) -> Result<CredentialMeta, OAuthError> {
		self.params(provider)?;
		self
			.store
			.upsert_oauth_material(
				provider,
				identity,
				access.expose(),
				refresh.map(Secret::expose),
				props,
				expires_at_ms,
				now_ms,
			)
			.map_err(Into::into)
	}

	/// Refreshes one credential under the store's in-process single-flight.
	///
	/// Permanent failures mark the credential expired without deleting it.
	///
	/// # Errors
	///
	/// Returns the refresh or persistence failure. Transient failures preserve
	/// the active credential; permanent failures first persist `Expired` state.
	pub async fn refresh_credential(
		&self,
		credential_id: u64,
		now_ms: u64,
	) -> Result<CredentialMeta, OAuthError> {
		let before = self
			.store
			.get_credential(credential_id, now_ms)?
			.ok_or_else(|| OAuthError::InvalidResponse("credential not found".into()))?;
		let _guard = self.store.refresh_singleflight(credential_id).await;
		let current = self
			.store
			.get_credential(credential_id, now_ms)?
			.ok_or_else(|| OAuthError::InvalidResponse("credential not found".into()))?;
		if (current.updated_at_ms != before.updated_at_ms
			|| current.expires_at_ms != before.expires_at_ms)
			&& current.expires_at_ms > now_ms
		{
			return Ok(current);
		}
		let params = self.params(current.provider.as_str())?.clone();
		let result = self.refresh_inner(&params, &current, now_ms).await;
		if result.as_ref().is_err_and(|error| !error.is_transient()) {
			self.store.expire_credential(credential_id, now_ms)?;
		}
		result
	}

	fn params(&self, provider: &str) -> Result<&OAuthParams, OAuthError> {
		self
			.params
			.iter()
			.find(|params| params.provider == provider || params.credential_provider == provider)
			.ok_or(OAuthError::UnknownProvider)
	}

	fn take_pending(&self, flow_id: &str) -> Result<PendingFlow, OAuthError> {
		self
			.pending
			.lock()
			.remove(flow_id)
			.ok_or_else(|| OAuthError::UnknownFlow(flow_id.into()))
	}

	async fn begin_pkce(
		&self,
		params: &OAuthParams,
		now_ms: u64,
	) -> Result<(LoginPrompt, u64, PendingFlow), OAuthError> {
		let verifier = Secret::new(random_urlsafe(32).as_bytes());
		let challenge = base64_url_encode(&sha256(verifier.expose()));
		let state = random_urlsafe(24);
		let port = params
			.callback_port
			.ok_or_else(|| OAuthError::InvalidResponse("PKCE callback port missing".into()))?;
		let callback_host = optional_extra_param(params, "callback_host").unwrap_or("localhost");
		let callback_path = optional_extra_param(params, "callback_path").unwrap_or("/callback");
		let listener = Arc::new(start_callback_listeners(callback_host, port).await?);
		let callback_port = listener.port();
		let redirect_uri = format!("http://{callback_host}:{callback_port}{callback_path}");
		let url = authorization_url(params, &redirect_uri, &state, &challenge)?;
		if let Some(browser) = self.browser.as_ref() {
			let _ = browser.open(url.as_str());
		}
		let prompt = LoginPrompt::Browse { url: url.as_str().into(), loopback: true };
		let expires_at_ms = now_ms.saturating_add(CUSTOM_FLOW_TIMEOUT_MS);
		Ok((
			prompt,
			expires_at_ms,
			PendingFlow::Pkce(PkcePending {
				params: params.clone(),
				verifier,
				state,
				redirect_uri: redirect_uri.into(),
				listener: Some(listener),
				expires_at_ms,
			}),
		))
	}

	async fn begin_device(
		&self,
		params: &OAuthParams,
		now_ms: u64,
	) -> Result<(LoginPrompt, u64, PendingFlow), OAuthError> {
		let body = form(&[("client_id", params.client_id.as_str()), ("scope", &scope(params))]);
		let value = self.post_form(params.authorize_url.as_str(), body).await?;
		provider_error(&value, 200)?;
		let device_code = required_string(&value, "device_code")?;
		let user_code = required_string(&value, "user_code")?;
		let verification_url = value
			.get("verification_uri_complete")
			.or_else(|| value.get("verification_uri"))
			.or_else(|| value.get("verification_url"))
			.and_then(Value::as_str)
			.ok_or_else(|| OAuthError::InvalidResponse("missing verification URL".into()))?;
		let expires_in = number(&value, "expires_in").unwrap_or(900);
		let interval = number(&value, "interval").unwrap_or(DEFAULT_DEVICE_INTERVAL_SECS);
		let expires_at_ms = now_ms.saturating_add(expires_in.saturating_mul(1_000));
		let prompt = LoginPrompt::Device {
			user_code:        user_code.into(),
			verification_url: verification_url.into(),
			interval_secs:    interval,
		};
		Ok((
			prompt,
			expires_at_ms,
			PendingFlow::Device(DevicePending {
				params: params.clone(),
				device_code: Secret::new(device_code.as_bytes()),
				interval_secs: interval,
				expires_at_ms,
			}),
		))
	}

	async fn begin_custom(
		&self,
		params: &OAuthParams,
		now_ms: u64,
	) -> Result<(LoginPrompt, u64, PendingFlow), OAuthError> {
		match params.exchange {
			Some(CustomExchange::ApiKeyPaste) => {
				Ok((LoginPrompt::Paste { url: params.authorize_url.clone() }, 0, PendingFlow::Paste {
					params: params.clone(),
				}))
			},
			Some(CustomExchange::PerplexityEmailOtp) => {
				let mut page = parsed_url(params.authorize_url.as_str())?;
				page.set_path("/");
				page.set_query(None);
				page.set_fragment(None);
				Ok((LoginPrompt::Paste { url: page.as_str().into() }, 0, PendingFlow::Paste {
					params: params.clone(),
				}))
			},
			Some(CustomExchange::CursorPoll) => {
				let verifier = Secret::new(random_urlsafe(32).as_bytes());
				let challenge = base64_url_encode(&sha256(verifier.expose()));
				let uuid = random_uuid();
				let mut url = parsed_url(params.authorize_url.as_str())?;
				{
					let mut query = url.query_pairs_mut();
					query.append_pair("challenge", &challenge);
					query.append_pair("uuid", uuid.as_str());
					for (key, value) in &params.extra_auth_params {
						if key != "refresh_url" {
							query.append_pair(key.as_str(), value.as_str());
						}
					}
				}
				self.open_browser(url.as_str());
				let expires_at_ms = now_ms.saturating_add(CURSOR_TIMEOUT_MS);
				Ok((
					LoginPrompt::Browse { url: url.as_str().into(), loopback: false },
					expires_at_ms,
					PendingFlow::Cursor(CursorPending {
						params: params.clone(),
						verifier,
						uuid,
						interval_ms: CURSOR_INITIAL_INTERVAL_MS,
						attempts: 0,
						consecutive_errors: 0,
						expires_at_ms,
					}),
				))
			},
			Some(CustomExchange::ZaiApiKey) => self.begin_custom_callback(params, now_ms, false).await,
			Some(CustomExchange::DevinCliToken) => {
				self.begin_custom_callback(params, now_ms, true).await
			},
			Some(CustomExchange::ExternalRedirectPkce) => self.begin_external_redirect(params, now_ms),
			Some(CustomExchange::OpenAiCodexClaims | CustomExchange::GithubCopilotSessionToken) => {
				Err(OAuthError::InvalidResponse(
					"custom exchange is attached to wrong flow kind".into(),
				))
			},
			None => Err(OAuthError::InvalidResponse("custom exchange selector missing".into())),
		}
	}

	async fn begin_custom_callback(
		&self,
		params: &OAuthParams,
		now_ms: u64,
		with_pkce: bool,
	) -> Result<(LoginPrompt, u64, PendingFlow), OAuthError> {
		let verifier = if with_pkce {
			Secret::new(random_urlsafe(32).as_bytes())
		} else {
			Secret::new(&[])
		};
		let challenge = with_pkce.then(|| base64_url_encode(&sha256(verifier.expose())));
		let state = random_urlsafe(24);
		let port = params
			.callback_port
			.ok_or_else(|| OAuthError::InvalidResponse("callback port missing".into()))?;
		let callback_host = optional_extra_param(params, "callback_host").unwrap_or("localhost");
		let callback_path = optional_extra_param(params, "callback_path").unwrap_or("/callback");
		let listener = Arc::new(start_callback_listeners(callback_host, port).await?);
		let callback_port = listener.port();
		let redirect_uri = format!("http://{callback_host}:{callback_port}{callback_path}");
		let mut url = parsed_url(params.authorize_url.as_str())?;
		{
			let mut query = url.query_pairs_mut();
			query.append_pair("redirect_uri", &redirect_uri);
			query.append_pair("state", state.as_str());
			if with_pkce {
				query.append_pair("prompt", extra_param(params, "prompt")?);
				let challenge = challenge
					.as_deref()
					.ok_or_else(|| OAuthError::InvalidResponse("PKCE challenge missing".into()))?;
				query.append_pair("code_challenge", challenge);
				query.append_pair("code_challenge_method", "S256");
			} else {
				query.append_pair("response_type", "code");
				query.append_pair("client_id", params.client_id.as_str());
			}
		}
		self.open_browser(url.as_str());
		let expires_at_ms = now_ms.saturating_add(CUSTOM_FLOW_TIMEOUT_MS);
		Ok((
			LoginPrompt::Browse { url: url.as_str().into(), loopback: true },
			expires_at_ms,
			PendingFlow::Pkce(PkcePending {
				params: params.clone(),
				verifier,
				state,
				redirect_uri: redirect_uri.into(),
				listener: Some(listener),
				expires_at_ms,
			}),
		))
	}

	fn begin_external_redirect(
		&self,
		params: &OAuthParams,
		now_ms: u64,
	) -> Result<(LoginPrompt, u64, PendingFlow), OAuthError> {
		let verifier = Secret::new(random_urlsafe(32).as_bytes());
		let challenge = base64_url_encode(&sha256(verifier.expose()));
		let state = random_urlsafe(24);
		let redirect_uri = extra_param(params, "redirect_uri")?;
		let url = authorization_url(params, redirect_uri, &state, &challenge)?;
		self.open_browser(url.as_str());
		let expires_at_ms = now_ms.saturating_add(CUSTOM_FLOW_TIMEOUT_MS);
		Ok((
			LoginPrompt::Browse { url: url.as_str().into(), loopback: false },
			expires_at_ms,
			PendingFlow::Pkce(PkcePending {
				params: params.clone(),
				verifier,
				state,
				redirect_uri: redirect_uri.into(),
				listener: None,
				expires_at_ms,
			}),
		))
	}

	fn open_browser(&self, url: &str) {
		if let Some(browser) = self.browser.as_ref() {
			let _ = browser.open(url);
		}
	}

	async fn exchange_pkce(
		&self,
		pkce: PkcePending,
		code: &str,
		now_ms: u64,
	) -> Result<CredentialMeta, OAuthError> {
		if pkce.expires_at_ms != 0 && now_ms >= pkce.expires_at_ms {
			return Err(OAuthError::LoginExpired);
		}
		match pkce.params.exchange {
			Some(CustomExchange::ZaiApiKey) => self.exchange_zai(pkce, code, now_ms).await,
			Some(CustomExchange::DevinCliToken) => self.exchange_devin(pkce, code, now_ms).await,
			_ => {
				let verifier = secret_str(&pkce.verifier)?;
				let value = if optional_extra_param(&pkce.params, "token_encoding") == Some("json") {
					self
						.post_json(
							pkce.params.token_url.as_str(),
							serde_json::json!({
								"grant_type": "authorization_code",
								"client_id": pkce.params.client_id.as_str(),
								"code": code,
								"state": pkce.state.as_str(),
								"redirect_uri": pkce.redirect_uri.as_str(),
								"code_verifier": verifier,
							}),
							HeaderMap::new(),
						)
						.await?
				} else {
					let body = if let Some(client_secret) =
						optional_extra_param(&pkce.params, "client_secret")
					{
						form(&[
							("grant_type", "authorization_code"),
							("client_id", pkce.params.client_id.as_str()),
							("client_secret", client_secret),
							("code", code),
							("redirect_uri", pkce.redirect_uri.as_str()),
							("code_verifier", verifier),
						])
					} else {
						form(&[
							("grant_type", "authorization_code"),
							("client_id", pkce.params.client_id.as_str()),
							("code", code),
							("redirect_uri", pkce.redirect_uri.as_str()),
							("code_verifier", verifier),
						])
					};
					self.post_form(pkce.params.token_url.as_str(), body).await?
				};
				let tokens = parse_tokens(value)?;
				self.persist_tokens(&pkce.params, tokens, now_ms).await
			},
		}
	}

	async fn exchange_devin(
		&self,
		pkce: PkcePending,
		code: &str,
		now_ms: u64,
	) -> Result<CredentialMeta, OAuthError> {
		let value = self
			.post_json(
				pkce.params.token_url.as_str(),
				serde_json::json!({
					"code": code,
					"code_verifier": secret_str(&pkce.verifier)?,
				}),
				HeaderMap::new(),
			)
			.await?;
		let token = required_string(&value, "token")?;
		let secret = Secret::new(token.as_bytes());
		let identity =
			jwt_subject(&secret).unwrap_or_else(|| pkce.params.credential_provider.clone());
		self
			.store
			.upsert_api_key(
				pkce.params.credential_provider.as_str(),
				identity.as_str(),
				secret.expose(),
				now_ms,
			)
			.map_err(Into::into)
	}

	async fn exchange_zai(
		&self,
		pkce: PkcePending,
		code: &str,
		now_ms: u64,
	) -> Result<CredentialMeta, OAuthError> {
		let token_value = self
			.post_json(
				pkce.params.token_url.as_str(),
				serde_json::json!({
					"provider": "zai",
					"code": code,
					"redirect_uri": pkce.redirect_uri,
					"state": pkce.state,
				}),
				HeaderMap::new(),
			)
			.await?;
		let token_data = unwrap_zai(token_value)?;
		let oauth_access = token_data
			.pointer("/zai/access_token")
			.and_then(Value::as_str)
			.filter(|value| !value.is_empty())
			.ok_or_else(|| {
				OAuthError::InvalidResponse("Z.ai token response missing access token".into())
			})?;
		let identity = token_data
			.pointer("/user/email")
			.or_else(|| token_data.pointer("/user/id"))
			.and_then(value_string)
			.unwrap_or_else(|| pkce.params.credential_provider.clone());
		let business_url = extra_param(&pkce.params, "business_login_url")?;
		let business_value = self
			.post_json(business_url, serde_json::json!({ "token": oauth_access }), HeaderMap::new())
			.await?;
		let business_data = unwrap_zai(business_value)?;
		let business_token = business_data
			.get("access_token")
			.or_else(|| business_data.get("accessToken"))
			.and_then(Value::as_str)
			.filter(|value| !value.is_empty())
			.ok_or_else(|| {
				OAuthError::InvalidResponse("Z.ai business login missing access token".into())
			})?;
		let headers = bearer_headers(business_token)?;
		let business_base = parsed_url(business_url)?;
		let origin = format!(
			"{}://{}",
			business_base.scheme(),
			business_base
				.host_str()
				.ok_or_else(|| OAuthError::InvalidUrl("Z.ai business host missing".into()))?
		);
		let customer = unwrap_zai(
			self
				.get_json(&format!("{origin}/api/biz/customer/getCustomerInfo"), headers.clone())
				.await?,
		)?;
		let (organization, project) = zai_default_scope(&customer)?;
		let keys_url =
			format!("{origin}/api/biz/v1/organization/{organization}/projects/{project}/api_keys");
		let keys = unwrap_zai(self.get_json(&keys_url, headers.clone()).await?)?;
		let key_name = extra_param(&pkce.params, "key_name")?;
		let existing = zai_keys(&keys)
			.iter()
			.find(|key| key.get("name").and_then(Value::as_str) == Some(key_name));
		let key_record = if let Some(key) = existing {
			key.clone()
		} else {
			unwrap_zai(
				self
					.post_json(&keys_url, serde_json::json!({ "name": key_name }), headers.clone())
					.await?,
			)?
		};
		let api_key = required_string(&key_record, "apiKey")?;
		let copied = unwrap_zai(
			self
				.get_json(&format!("{keys_url}/copy/{}", percent_encode_segment(api_key)), headers)
				.await?,
		)?;
		let secret_key = required_string(&copied, "secretKey")?;
		let minted = Secret::new(format!("{api_key}.{secret_key}").as_bytes());
		self
			.store
			.upsert_api_key(
				pkce.params.credential_provider.as_str(),
				identity.as_str(),
				minted.expose(),
				now_ms,
			)
			.map_err(Into::into)
	}

	async fn exchange_perplexity(
		&self,
		params: &OAuthParams,
		input: &str,
		now_ms: u64,
	) -> Result<CredentialMeta, OAuthError> {
		let (email, otp) = parse_email_otp(input)?;
		let otp = secret_str(&otp)?;
		let mut headers = HeaderMap::new();
		headers.insert(
			USER_AGENT,
			HeaderValue::from_static("Perplexity/641 CFNetwork/1568 Darwin/25.2.0"),
		);
		headers.insert("x-app-apiversion", HeaderValue::from_static("2.18"));
		let csrf_url = extra_param(params, "csrf_url")?;
		let csrf = self.get_json(csrf_url, headers.clone()).await?;
		let csrf_token = required_string(&csrf, "csrfToken")?;
		self
			.post_json(
				params.authorize_url.as_str(),
				serde_json::json!({ "email": email.as_str(), "csrfToken": csrf_token }),
				headers.clone(),
			)
			.await?;
		let verified = self
			.post_json(
				params.token_url.as_str(),
				serde_json::json!({ "email": email.as_str(), "otp": otp, "csrfToken": csrf_token }),
				headers,
			)
			.await?;
		let token = required_string(&verified, "token")?;
		let access = Secret::new(token.as_bytes());
		let expires_at_ms = jwt_expiry_ms(&access).unwrap_or(NEVER_EXPIRES_MS);
		self
			.store
			.upsert_oauth_material(
				params.credential_provider.as_str(),
				email.as_str(),
				access.expose(),
				Some(access.expose()),
				&Value::Null,
				expires_at_ms,
				now_ms,
			)
			.map_err(Into::into)
	}

	async fn poll_device(&self, mut pending: DevicePending) -> Result<CredentialMeta, OAuthError> {
		loop {
			if self.time.now_ms() >= pending.expires_at_ms {
				return Err(OAuthError::LoginExpired);
			}
			self
				.time
				.sleep(std::time::Duration::from_secs(pending.interval_secs))
				.await;
			if self.time.now_ms() >= pending.expires_at_ms {
				return Err(OAuthError::LoginExpired);
			}
			let device_code = secret_str(&pending.device_code)?;
			let body = form(&[
				("client_id", pending.params.client_id.as_str()),
				("device_code", device_code),
				("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
			]);
			let response = self
				.post_form_raw(pending.params.token_url.as_str(), body)
				.await?;
			let value = decode_response(&response.body)?;
			let error = value.get("error").and_then(Value::as_str);
			match error {
				Some("authorization_pending") => {},
				Some("slow_down") => pending.interval_secs = pending.interval_secs.saturating_add(5),
				Some(_) => return Err(provider_error_value(&value, response.status)),
				None if !(200..300).contains(&response.status) => {
					return Err(http_status_error(response.status));
				},
				None => {
					let tokens = parse_tokens(value)?;
					return self
						.persist_tokens(&pending.params, tokens, self.time.now_ms())
						.await;
				},
			}
		}
	}

	async fn poll_cursor(&self, mut pending: CursorPending) -> Result<CredentialMeta, OAuthError> {
		while pending.attempts < CURSOR_MAX_POLLS {
			if self.time.now_ms() >= pending.expires_at_ms {
				return Err(OAuthError::LoginExpired);
			}
			self
				.time
				.sleep(Duration::from_millis(pending.interval_ms))
				.await;
			if self.time.now_ms() >= pending.expires_at_ms {
				return Err(OAuthError::LoginExpired);
			}
			pending.attempts = pending.attempts.saturating_add(1);
			let verifier = secret_str(&pending.verifier)?;
			let mut url = parsed_url(pending.params.token_url.as_str())?;
			url.query_pairs_mut()
				.append_pair("uuid", pending.uuid.as_str())
				.append_pair("verifier", verifier);
			let response = self
				.http
				.execute(HttpRequest {
					method:  Method::GET,
					url:     url.as_str().into(),
					headers: json_headers(),
					body:    Bytes::new(),
				})
				.await;
			match response {
				Ok(response) if response.status == 404 => {
					pending.consecutive_errors = 0;
					pending.interval_ms = pending
						.interval_ms
						.saturating_mul(6)
						.div_ceil(5)
						.min(CURSOR_MAX_INTERVAL_MS);
				},
				Ok(response) if (200..300).contains(&response.status) => {
					let value = decode_response(&response.body)?;
					let access = value
						.get("accessToken")
						.and_then(Value::as_str)
						.filter(|value| !value.is_empty())
						.ok_or_else(|| {
							OAuthError::InvalidResponse("Cursor response missing access token".into())
						})?;
					let refresh = value
						.get("refreshToken")
						.and_then(Value::as_str)
						.filter(|value| !value.is_empty())
						.ok_or_else(|| {
							OAuthError::InvalidResponse("Cursor response missing refresh token".into())
						})?;
					let access = Secret::new(access.as_bytes());
					let identity = jwt_subject(&access)
						.unwrap_or_else(|| pending.params.credential_provider.clone());
					let completed_at_ms = self.time.now_ms();
					let expires_at_ms = jwt_expiry_ms(&access)
						.unwrap_or_else(|| completed_at_ms.saturating_add(3_600_000));
					return self
						.store
						.upsert_oauth_material(
							pending.params.credential_provider.as_str(),
							identity.as_str(),
							access.expose(),
							Some(refresh.as_bytes()),
							&Value::Null,
							expires_at_ms,
							completed_at_ms,
						)
						.map_err(Into::into);
				},
				Ok(_) | Err(_) => {
					pending.consecutive_errors = pending.consecutive_errors.saturating_add(1);
					if pending.consecutive_errors >= 3 {
						return Err(OAuthError::Provider {
							code:      "cursor_poll_failed".into(),
							status:    0,
							transient: true,
						});
					}
				},
			}
		}
		Err(OAuthError::LoginExpired)
	}

	async fn persist_tokens(
		&self,
		params: &OAuthParams,
		tokens: TokenSet,
		now_ms: u64,
	) -> Result<CredentialMeta, OAuthError> {
		let mut props = Map::new();
		let mut identity =
			token_identity(&tokens).unwrap_or_else(|| params.credential_provider.clone());
		if params.exchange == Some(CustomExchange::OpenAiCodexClaims) {
			let id_token = tokens
				.id_token
				.as_ref()
				.ok_or_else(|| OAuthError::InvalidResponse("Codex response missing id_token".into()))?;
			identity = codex_account_id(id_token)?;
			props.insert("account_id".into(), Value::String(identity.to_string()));
		}
		if params.exchange == Some(CustomExchange::GithubCopilotSessionToken) {
			let source = tokens.access;
			let session = self.github_copilot_exchange(&source).await?;
			let expires_at_ms = session.expires_at_ms(now_ms);
			return self
				.store
				.upsert_oauth_material(
					params.credential_provider.as_str(),
					identity.as_str(),
					session.access.expose(),
					Some(source.expose()),
					&Value::Object(props),
					expires_at_ms,
					now_ms,
				)
				.map_err(Into::into);
		}
		let expires_at_ms = tokens.expires_at_ms(now_ms);
		self
			.store
			.upsert_oauth_material(
				params.credential_provider.as_str(),
				identity.as_str(),
				tokens.access.expose(),
				tokens.refresh.as_ref().map(Secret::expose),
				&Value::Object(props),
				expires_at_ms,
				now_ms,
			)
			.map_err(Into::into)
	}

	fn persist_api_key(
		&self,
		params: &OAuthParams,
		secret: Secret,
		now_ms: u64,
	) -> Result<CredentialMeta, OAuthError> {
		self
			.store
			.upsert_api_key(
				params.credential_provider.as_str(),
				params.credential_provider.as_str(),
				secret.expose(),
				now_ms,
			)
			.map_err(Into::into)
	}

	async fn refresh_inner(
		&self,
		params: &OAuthParams,
		meta: &CredentialMeta,
		now_ms: u64,
	) -> Result<CredentialMeta, OAuthError> {
		let refresh = self
			.store
			.redeem_refresh(meta.id)?
			.ok_or_else(|| OAuthError::InvalidResponse("credential has no refresh material".into()))?;
		if params.exchange == Some(CustomExchange::CursorPoll) {
			let refresh_url = extra_param(params, "refresh_url")?;
			let headers = bearer_headers(secret_str(&refresh)?)?;
			let value = self
				.post_json(refresh_url, Value::Object(Map::new()), headers)
				.await?;
			let access = value
				.get("accessToken")
				.and_then(Value::as_str)
				.filter(|value| !value.is_empty())
				.ok_or_else(|| {
					OAuthError::InvalidResponse("Cursor refresh missing access token".into())
				})?;
			let next_refresh = value
				.get("refreshToken")
				.and_then(Value::as_str)
				.filter(|value| !value.is_empty());
			let access = Secret::new(access.as_bytes());
			let expires_at_ms =
				jwt_expiry_ms(&access).unwrap_or_else(|| now_ms.saturating_add(3_600_000));
			return self
				.store
				.upsert_oauth_material(
					params.credential_provider.as_str(),
					meta.identity.as_str(),
					access.expose(),
					Some(next_refresh.map_or_else(|| refresh.expose(), str::as_bytes)),
					&self.store.oauth_props(meta.id)?.unwrap_or(Value::Null),
					expires_at_ms,
					now_ms,
				)
				.map_err(Into::into);
		}
		if params.exchange == Some(CustomExchange::GithubCopilotSessionToken) {
			let session = self.github_copilot_exchange(&refresh).await?;
			return self
				.store
				.upsert_oauth_material(
					params.credential_provider.as_str(),
					meta.identity.as_str(),
					session.access.expose(),
					Some(refresh.expose()),
					&self.store.oauth_props(meta.id)?.unwrap_or(Value::Null),
					session.expires_at_ms(now_ms),
					now_ms,
				)
				.map_err(Into::into);
		}
		let refresh_value = secret_str(&refresh)?;
		let value = if optional_extra_param(params, "token_encoding") == Some("json") {
			let mut headers = HeaderMap::new();
			if let Some(beta) = optional_extra_param(params, "refresh_beta") {
				headers.insert(
					"anthropic-beta",
					HeaderValue::try_from(beta)
						.map_err(|_| OAuthError::InvalidResponse("invalid OAuth beta header".into()))?,
				);
			}
			if let Some(user_agent) = optional_extra_param(params, "refresh_user_agent") {
				headers.insert(
					USER_AGENT,
					HeaderValue::try_from(user_agent)
						.map_err(|_| OAuthError::InvalidResponse("invalid OAuth user agent".into()))?,
				);
			}
			self
				.post_json(
					params.token_url.as_str(),
					serde_json::json!({
						"grant_type": "refresh_token",
						"client_id": params.client_id.as_str(),
						"refresh_token": refresh_value,
					}),
					headers,
				)
				.await?
		} else {
			let body = if params.exchange == Some(CustomExchange::ExternalRedirectPkce) {
				form(&[
					("grant_type", "refresh_token"),
					("client_id", params.client_id.as_str()),
					("redirect_uri", extra_param(params, "redirect_uri")?),
					("refresh_token", refresh_value),
				])
			} else if let Some(client_secret) = optional_extra_param(params, "client_secret") {
				form(&[
					("grant_type", "refresh_token"),
					("client_id", params.client_id.as_str()),
					("client_secret", client_secret),
					("refresh_token", refresh_value),
				])
			} else {
				form(&[
					("grant_type", "refresh_token"),
					("client_id", params.client_id.as_str()),
					("refresh_token", refresh_value),
				])
			};
			self.post_form(params.token_url.as_str(), body).await?
		};
		let mut tokens = parse_tokens(value)?;
		if tokens.refresh.is_none() {
			tokens.refresh = Some(refresh);
		}
		let props = self.store.oauth_props(meta.id)?.unwrap_or(Value::Null);
		self
			.store
			.upsert_oauth_material(
				params.credential_provider.as_str(),
				meta.identity.as_str(),
				tokens.access.expose(),
				tokens.refresh.as_ref().map(Secret::expose),
				&props,
				tokens.expires_at_ms(now_ms),
				now_ms,
			)
			.map_err(Into::into)
	}

	async fn github_copilot_exchange(&self, github_token: &Secret) -> Result<TokenSet, OAuthError> {
		let mut headers = HeaderMap::new();
		headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
		headers.insert(USER_AGENT, HeaderValue::from_static(OMP_USER_AGENT));
		let mut bearer = Vec::with_capacity(7 + github_token.expose().len());
		bearer.extend_from_slice(b"Bearer ");
		bearer.extend_from_slice(github_token.expose());
		headers.insert(
			AUTHORIZATION,
			HeaderValue::from_bytes(&bearer)
				.map_err(|_| OAuthError::InvalidResponse("invalid GitHub token bytes".into()))?,
		);
		let response = self
			.http
			.execute(HttpRequest {
				method: Method::GET,
				url: COPILOT_TOKEN_URL.into(),
				headers,
				body: Bytes::new(),
			})
			.await?;
		if !(200..300).contains(&response.status) {
			return Err(http_status_error(response.status));
		}
		parse_tokens(decode_response(&response.body)?)
	}

	async fn post_form(&self, url: &str, body: Bytes) -> Result<Value, OAuthError> {
		let response = self.post_form_raw(url, body).await?;
		let value = decode_response(&response.body)?;
		if !(200..300).contains(&response.status) {
			if value.get("error").is_some() {
				return Err(provider_error_value(&value, response.status));
			}
			return Err(http_status_error(response.status));
		}
		provider_error(&value, response.status)?;
		Ok(value)
	}

	async fn post_json(
		&self,
		url: &str,
		value: Value,
		mut headers: HeaderMap,
	) -> Result<Value, OAuthError> {
		headers.insert(CONTENT_TYPE, HeaderValue::from_static(JSON_CONTENT_TYPE));
		headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
		let response = self
			.http
			.execute(HttpRequest {
				method: Method::POST,
				url: url.into(),
				headers,
				body: Bytes::from(serde_json::to_vec(&value).map_err(|_| {
					OAuthError::InvalidResponse("could not encode OAuth request".into())
				})?),
			})
			.await?;
		decode_http_response(response)
	}

	async fn get_json(&self, url: &str, mut headers: HeaderMap) -> Result<Value, OAuthError> {
		headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
		let response = self
			.http
			.execute(HttpRequest { method: Method::GET, url: url.into(), headers, body: Bytes::new() })
			.await?;
		decode_http_response(response)
	}

	async fn post_form_raw(&self, url: &str, body: Bytes) -> Result<HttpResponse, OAuthError> {
		let mut headers = HeaderMap::new();
		headers.insert(CONTENT_TYPE, HeaderValue::from_static(FORM_CONTENT_TYPE));
		headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
		self
			.http
			.execute(HttpRequest { method: Method::POST, url: url.into(), headers, body })
			.await
			.map_err(Into::into)
	}
}

struct CallbackListeners {
	primary:   TcpListener,
	companion: Option<TcpListener>,
	port:      u16,
}

impl CallbackListeners {
	const fn port(&self) -> u16 {
		self.port
	}
}

struct PkcePending {
	params:        OAuthParams,
	verifier:      Secret,
	state:         Str,
	redirect_uri:  Str,
	listener:      Option<Arc<CallbackListeners>>,
	expires_at_ms: u64,
}

struct DevicePending {
	params:        OAuthParams,
	device_code:   Secret,
	interval_secs: u64,
	expires_at_ms: u64,
}
struct CursorPending {
	params:             OAuthParams,
	verifier:           Secret,
	uuid:               Str,
	interval_ms:        u64,
	attempts:           u16,
	consecutive_errors: u8,
	expires_at_ms:      u64,
}

enum PendingFlow {
	Pkce(PkcePending),
	Device(DevicePending),
	Cursor(CursorPending),
	Paste { params: OAuthParams },
}

struct TokenSet {
	access:                Secret,
	refresh:               Option<Secret>,
	id_token:              Option<Secret>,
	expires_in_secs:       u64,
	expires_at_epoch_secs: u64,
	identity:              Option<Str>,
}

impl TokenSet {
	const fn expires_at_ms(&self, now_ms: u64) -> u64 {
		if self.expires_at_epoch_secs != 0 {
			self.expires_at_epoch_secs.saturating_mul(1_000)
		} else {
			now_ms.saturating_add(self.expires_in_secs.saturating_mul(1_000))
		}
	}
}

fn decode_http_response(response: HttpResponse) -> Result<Value, OAuthError> {
	let value = if response.body.is_empty() {
		Value::Null
	} else {
		decode_response(&response.body)?
	};
	if !(200..300).contains(&response.status) {
		if value.get("error").is_some() {
			return Err(provider_error_value(&value, response.status));
		}
		return Err(http_status_error(response.status));
	}
	provider_error(&value, response.status)?;
	Ok(value)
}

fn parsed_url(url: &str) -> Result<Url, OAuthError> {
	Url::parse(url).map_err(|error| OAuthError::InvalidUrl(error.to_string().into()))
}

fn extra_param<'a>(params: &'a OAuthParams, key: &str) -> Result<&'a str, OAuthError> {
	params
		.extra_auth_params
		.get(key)
		.map(Str::as_str)
		.filter(|value| !value.is_empty())
		.ok_or_else(|| OAuthError::InvalidResponse(format!("missing custom parameter {key}").into()))
}
fn optional_extra_param<'a>(params: &'a OAuthParams, key: &str) -> Option<&'a str> {
	params
		.extra_auth_params
		.get(key)
		.map(Str::as_str)
		.filter(|value| !value.is_empty())
}
fn internal_param(key: &str) -> bool {
	matches!(
		key,
		"redirect_uri"
			| "callback_host"
			| "callback_path"
			| "client_secret"
			| "token_encoding"
			| "refresh_beta"
			| "refresh_user_agent"
	)
}

fn json_headers() -> HeaderMap {
	let mut headers = HeaderMap::new();
	headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
	headers
}

fn bearer_headers(token: &str) -> Result<HeaderMap, OAuthError> {
	let mut bytes = Vec::with_capacity(7 + token.len());
	bytes.extend_from_slice(b"Bearer ");
	bytes.extend_from_slice(token.as_bytes());
	let mut headers = json_headers();
	headers.insert(
		AUTHORIZATION,
		HeaderValue::from_bytes(&bytes)
			.map_err(|_| OAuthError::InvalidResponse("invalid bearer token bytes".into()))?,
	);
	Ok(headers)
}

fn parse_email_otp(input: &str) -> Result<(Str, Secret), OAuthError> {
	let input = input.trim();
	let parsed = serde_json::from_str::<Value>(input).ok();
	let pair = parsed.as_ref().and_then(|value| {
		Some((value.get("email")?.as_str()?.trim(), value.get("otp")?.as_str()?.trim()))
	});
	let (email, otp) = pair.or_else(|| input.split_once('\n')).ok_or_else(|| {
		OAuthError::InvalidResponse(
			"Perplexity login requires JSON with non-secret `email` and sealed `otp` fields".into(),
		)
	})?;
	if email.is_empty() || !email.contains('@') || otp.is_empty() {
		return Err(OAuthError::InvalidResponse("Perplexity email or OTP is invalid".into()));
	}
	Ok((email.into(), Secret::new(otp.as_bytes())))
}

fn random_uuid() -> Str {
	let mut bytes: [u8; 16] = rand::random();
	bytes[6] = (bytes[6] & 0x0f) | 0x40;
	bytes[8] = (bytes[8] & 0x3f) | 0x80;
	format!(
		"{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:\
		 02x}{:02x}",
		bytes[0],
		bytes[1],
		bytes[2],
		bytes[3],
		bytes[4],
		bytes[5],
		bytes[6],
		bytes[7],
		bytes[8],
		bytes[9],
		bytes[10],
		bytes[11],
		bytes[12],
		bytes[13],
		bytes[14],
		bytes[15],
	)
	.into()
}

fn jwt_payload(token: &Secret) -> Option<Value> {
	let token = secret_str(token).ok()?;
	let payload = token.split('.').nth(1)?;
	let decoded = base64_url_decode(payload).ok()?;
	serde_json::from_slice(&decoded).ok()
}

fn jwt_subject(token: &Secret) -> Option<Str> {
	let payload = jwt_payload(token)?;
	let subject = payload.get("sub")?.as_str()?.trim();
	let subject = subject.rsplit('|').next()?.trim();
	(!subject.is_empty()).then(|| subject.into())
}

fn jwt_expiry_ms(token: &Secret) -> Option<u64> {
	let expiry = jwt_payload(token)?.get("exp")?.as_u64()?;
	Some(expiry.saturating_mul(1_000).saturating_sub(5 * 60 * 1_000))
}

fn unwrap_zai(value: Value) -> Result<Value, OAuthError> {
	let Some(object) = value.as_object() else {
		return Ok(value);
	};
	let has_envelope = object.contains_key("code") || object.contains_key("success");
	if !has_envelope {
		return Ok(value);
	}
	let code_ok = match object.get("code") {
		None | Some(Value::Null) => true,
		Some(Value::Number(code)) => matches!(code.as_u64(), Some(0 | 200)),
		Some(Value::String(code)) => matches!(code.as_str(), "0" | "200"),
		Some(_) => false,
	};
	let success = object
		.get("success")
		.and_then(Value::as_bool)
		.unwrap_or(true);
	if !code_ok || !success {
		return Err(OAuthError::Provider {
			code:      "zai_exchange_failed".into(),
			status:    200,
			transient: false,
		});
	}
	Ok(object.get("data").cloned().unwrap_or(value))
}

fn value_string(value: &Value) -> Option<Str> {
	match value {
		Value::String(value) if !value.is_empty() => Some(value.as_str().into()),
		Value::Number(value) => Some(value.to_string().into()),
		_ => None,
	}
}

fn zai_default_scope(value: &Value) -> Result<(Str, Str), OAuthError> {
	let organizations = value
		.get("organizations")
		.and_then(Value::as_array)
		.ok_or_else(|| OAuthError::InvalidResponse("Z.ai account has no organization".into()))?;
	let organization = organizations
		.iter()
		.find(|item| item.get("isDefault").and_then(Value::as_bool) == Some(true))
		.or_else(|| organizations.first())
		.ok_or_else(|| OAuthError::InvalidResponse("Z.ai account has no organization".into()))?;
	let organization_id = organization
		.get("organizationId")
		.and_then(value_string)
		.ok_or_else(|| OAuthError::InvalidResponse("Z.ai organization is missing an id".into()))?;
	let projects = organization
		.get("projects")
		.and_then(Value::as_array)
		.ok_or_else(|| OAuthError::InvalidResponse("Z.ai organization has no project".into()))?;
	let project = projects
		.iter()
		.find(|item| item.get("isDefault").and_then(Value::as_bool) == Some(true))
		.or_else(|| projects.first())
		.ok_or_else(|| OAuthError::InvalidResponse("Z.ai organization has no project".into()))?;
	let project_id = project
		.get("projectId")
		.and_then(value_string)
		.ok_or_else(|| OAuthError::InvalidResponse("Z.ai project is missing an id".into()))?;
	Ok((organization_id, project_id))
}

fn zai_keys(value: &Value) -> &[Value] {
	if let Some(keys) = value.as_array() {
		return keys;
	}
	for field in ["list", "keys", "apiKeys", "records"] {
		if let Some(keys) = value.get(field).and_then(Value::as_array) {
			return keys;
		}
	}
	&[]
}

fn percent_encode_segment(value: &str) -> String {
	const HEX: &[u8; 16] = b"0123456789ABCDEF";
	let mut output = String::with_capacity(value.len());
	for byte in value.bytes() {
		if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
			output.push(char::from(byte));
		} else {
			output.push('%');
			output.push(char::from(HEX[usize::from(byte >> 4)]));
			output.push(char::from(HEX[usize::from(byte & 0x0f)]));
		}
	}
	output
}

fn authorization_url(
	params: &OAuthParams,
	redirect_uri: &str,
	state: &str,
	challenge: &str,
) -> Result<Url, OAuthError> {
	let mut url = Url::parse(params.authorize_url.as_str())
		.map_err(|error| OAuthError::InvalidUrl(error.to_string().into()))?;
	{
		let mut query = url.query_pairs_mut();
		query.append_pair("response_type", "code");
		query.append_pair("client_id", params.client_id.as_str());
		query.append_pair("redirect_uri", redirect_uri);
		query.append_pair("scope", &scope(params));
		query.append_pair("state", state);
		query.append_pair("code_challenge", challenge);
		query.append_pair("code_challenge_method", "S256");
		for (key, value) in &params.extra_auth_params {
			if !internal_param(key.as_str()) {
				query.append_pair(key.as_str(), value.as_str());
			}
		}
	}
	Ok(url)
}

fn scope(params: &OAuthParams) -> String {
	params
		.scopes
		.iter()
		.map(Str::as_str)
		.collect::<Vec<_>>()
		.join(" ")
}

fn form(pairs: &[(&str, &str)]) -> Bytes {
	let mut serializer = url::form_urlencoded::Serializer::new(String::new());
	for (key, value) in pairs {
		serializer.append_pair(key, value);
	}
	Bytes::from(serializer.finish())
}

async fn bind_callback_listeners(
	hostname: &str,
	requested_port: u16,
) -> std::io::Result<CallbackListeners> {
	if hostname != "localhost" {
		let primary = TcpListener::bind((hostname, requested_port)).await?;
		let port = primary.local_addr()?.port();
		return Ok(CallbackListeners { primary, companion: None, port });
	}

	for attempt in 0..=IPV6_COMPANION_ATTEMPTS {
		let primary = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, requested_port)).await?;
		let port = primary.local_addr()?.port();
		match TcpListener::bind((std::net::Ipv6Addr::LOCALHOST, port)).await {
			Ok(companion) => {
				return Ok(CallbackListeners { primary, companion: Some(companion), port });
			},
			// When IPv6 itself is unavailable, IPv4 is the only reachable
			// localhost family and remains valid on the preferred port.
			Err(error) if error.kind() != std::io::ErrorKind::AddrInUse => {
				return Ok(CallbackListeners { primary, companion: None, port });
			},
			// An exact-address collision would leave localhost half-reachable.
			// Random ports can be redrawn; fixed ports fall through to fallback.
			Err(error) if requested_port != 0 || attempt == IPV6_COMPANION_ATTEMPTS => {
				return Err(error);
			},
			Err(_) => {},
		}
	}
	unreachable!()
}

async fn start_callback_listeners(
	hostname: &str,
	preferred_port: u16,
) -> Result<CallbackListeners, OAuthError> {
	let listeners = match bind_callback_listeners(hostname, preferred_port).await {
		Ok(listeners) => Ok(listeners),
		Err(_) => bind_callback_listeners(hostname, 0).await,
	};
	listeners.map_err(|error| {
		OAuthError::InvalidCallback(format!("failed to bind OAuth callback listener: {error}").into())
	})
}

async fn receive_callback(
	listeners: &CallbackListeners,
	expected_state: &str,
) -> Result<(Str, Str), OAuthError> {
	let accepted = if let Some(companion) = listeners.companion.as_ref() {
		tokio::select! {
			accepted = listeners.primary.accept() => accepted,
			accepted = companion.accept() => accepted,
		}
	} else {
		listeners.primary.accept().await
	};
	let (mut stream, _) =
		accepted.map_err(|error| OAuthError::InvalidCallback(error.to_string().into()))?;
	let mut request = Vec::with_capacity(1_024);
	loop {
		let mut chunk = [0_u8; 1_024];
		let read = stream
			.read(&mut chunk)
			.await
			.map_err(|error| OAuthError::InvalidCallback(error.to_string().into()))?;
		if read == 0 {
			break;
		}
		request.extend_from_slice(&chunk[..read]);
		if request.windows(4).any(|window| window == b"\r\n\r\n") {
			break;
		}
		if request.len() > CALLBACK_LIMIT {
			return Err(OAuthError::InvalidCallback("callback headers too large".into()));
		}
	}
	let parsed = parse_callback_request(&request, expected_state);
	let (status, message) = if parsed.is_ok() {
		("200 OK", "Authorization received. You may close this window.")
	} else {
		("400 Bad Request", "Invalid authorization callback.")
	};
	let response = format!(
		"HTTP/1.1 {status}\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: \
		 {}\r\nConnection: close\r\n\r\n{message}",
		message.len()
	);
	let _ = stream.write_all(response.as_bytes()).await;
	parsed
}

fn parse_callback_request(request: &[u8], expected_state: &str) -> Result<(Str, Str), OAuthError> {
	let request = std::str::from_utf8(request)
		.map_err(|_| OAuthError::InvalidCallback("callback is not UTF-8".into()))?;
	let target = request
		.lines()
		.next()
		.and_then(|line| line.split_whitespace().nth(1))
		.ok_or_else(|| OAuthError::InvalidCallback("missing request target".into()))?;
	let url = Url::parse(&format!("http://127.0.0.1{target}"))
		.map_err(|error| OAuthError::InvalidCallback(error.to_string().into()))?;
	callback_values(&url, expected_state)
}

fn parse_pasted_code(input: &str, expected_state: &str) -> Result<(Str, Option<Str>), OAuthError> {
	if let Ok(url) = Url::parse(input) {
		let (code, state) = callback_values(&url, expected_state)?;
		Ok((code, Some(state)))
	} else {
		Ok((input.into(), None))
	}
}

fn callback_values(url: &Url, expected_state: &str) -> Result<(Str, Str), OAuthError> {
	let values: HashMap<_, _> = url.query_pairs().collect();
	let state = values
		.get("state")
		.filter(|value| !value.is_empty())
		.ok_or_else(|| OAuthError::InvalidCallback("missing state".into()))?;
	if state.as_ref() != expected_state {
		return Err(OAuthError::StateMismatch);
	}
	if let Some(error) = values.get("error") {
		return Err(OAuthError::Provider {
			code:      safe_error_code(error.as_ref()).into(),
			status:    200,
			transient: false,
		});
	}
	let code = values
		.get("code")
		.filter(|value| !value.is_empty())
		.ok_or_else(|| OAuthError::InvalidCallback("missing code".into()))?;
	Ok((code.as_ref().into(), state.as_ref().into()))
}

fn decode_response(body: &[u8]) -> Result<Value, OAuthError> {
	if let Ok(value) = serde_json::from_slice(body) {
		return Ok(value);
	}
	let text = std::str::from_utf8(body)
		.map_err(|_| OAuthError::InvalidResponse("response is neither JSON nor form data".into()))?;
	let mut object = Map::new();
	for (key, value) in url::form_urlencoded::parse(text.as_bytes()) {
		object.insert(key.into_owned(), Value::String(value.into_owned()));
	}
	if object.is_empty() {
		Err(OAuthError::InvalidResponse("empty OAuth response".into()))
	} else {
		Ok(Value::Object(object))
	}
}

fn parse_tokens(value: Value) -> Result<TokenSet, OAuthError> {
	provider_error(&value, 200)?;
	let access = value
		.get("access_token")
		.or_else(|| value.get("token"))
		.and_then(Value::as_str)
		.filter(|token| !token.is_empty())
		.ok_or_else(|| OAuthError::InvalidResponse("missing access token".into()))?;
	let expires_in_secs = number(&value, "expires_in").unwrap_or(0);
	let expires_at_epoch_secs = number(&value, "expires_at")
		.or_else(|| {
			number(&value, "created_at").map(|created| created.saturating_add(expires_in_secs))
		})
		.unwrap_or(0);
	Ok(TokenSet {
		access: Secret::new(access.as_bytes()),
		refresh: value
			.get("refresh_token")
			.and_then(Value::as_str)
			.map(|token| Secret::new(token.as_bytes())),
		id_token: value
			.get("id_token")
			.and_then(Value::as_str)
			.map(|token| Secret::new(token.as_bytes())),
		expires_in_secs,
		expires_at_epoch_secs,
		identity: value
			.get("account_id")
			.or_else(|| value.get("user_id"))
			.or_else(|| value.get("login"))
			.and_then(Value::as_str)
			.map(Into::into),
	})
}

fn token_identity(tokens: &TokenSet) -> Option<Str> {
	tokens.identity.clone()
}

fn required_string<'a>(value: &'a Value, key: &str) -> Result<&'a str, OAuthError> {
	value
		.get(key)
		.and_then(Value::as_str)
		.filter(|value| !value.is_empty())
		.ok_or_else(|| OAuthError::InvalidResponse(format!("missing {key}").into()))
}

fn number(value: &Value, key: &str) -> Option<u64> {
	value
		.get(key)
		.and_then(|value| value.as_u64().or_else(|| value.as_str()?.parse().ok()))
}

fn provider_error(value: &Value, status: u16) -> Result<(), OAuthError> {
	if value.get("error").is_some() {
		Err(provider_error_value(value, status))
	} else {
		Ok(())
	}
}

fn provider_error_value(value: &Value, status: u16) -> OAuthError {
	let code = safe_error_code(
		value
			.get("error")
			.and_then(Value::as_str)
			.unwrap_or("provider_error"),
	);
	OAuthError::Provider {
		code: code.into(),
		status,
		transient: matches!(code, "temporarily_unavailable" | "server_error")
			|| transient_status(status),
	}
}

fn safe_error_code(code: &str) -> &'static str {
	match code {
		"authorization_pending" => "authorization_pending",
		"slow_down" => "slow_down",
		"access_denied" => "access_denied",
		"expired_token" => "expired_token",
		"invalid_grant" => "invalid_grant",
		"invalid_token" => "invalid_token",
		"unauthorized_client" => "unauthorized_client",
		"temporarily_unavailable" => "temporarily_unavailable",
		"server_error" => "server_error",
		"authorization_declined" => "authorization_declined",
		"bad_verification_code" => "bad_verification_code",
		"incorrect_device_code" => "incorrect_device_code",
		"device_flow_expired" => "device_flow_expired",
		_ => "provider_error",
	}
}

fn http_status_error(status: u16) -> OAuthError {
	OAuthError::Provider { code: "http_error".into(), status, transient: transient_status(status) }
}

const fn transient_status(status: u16) -> bool {
	matches!(status, 408 | 425 | 429 | 500..=599)
}

fn secret_str(secret: &Secret) -> Result<&str, OAuthError> {
	std::str::from_utf8(secret.expose())
		.map_err(|_| OAuthError::InvalidResponse("OAuth secret is not UTF-8".into()))
}

fn codex_account_id(id_token: &Secret) -> Result<Str, OAuthError> {
	let token = secret_str(id_token)?;
	let payload = token
		.split('.')
		.nth(1)
		.ok_or_else(|| OAuthError::InvalidResponse("id_token is not a JWT".into()))?;
	let bytes = base64_url_decode(payload)?;
	let claims: Value = serde_json::from_slice(&bytes)
		.map_err(|_| OAuthError::InvalidResponse("id_token claims are not JSON".into()))?;
	claims
		.pointer("/https:~1~1api.openai.com~1auth/chatgpt_account_id")
		.or_else(|| claims.get("chatgpt_account_id"))
		.and_then(Value::as_str)
		.filter(|id| !id.is_empty())
		.map(Into::into)
		.ok_or_else(|| OAuthError::InvalidResponse("id_token missing ChatGPT account id".into()))
}

fn random_urlsafe(bytes: usize) -> Str {
	let mut random = vec![0_u8; bytes];
	for chunk in random.chunks_mut(32) {
		let value: [u8; 32] = rand::random();
		chunk.copy_from_slice(&value[..chunk.len()]);
	}
	base64_url_encode(&random).into()
}

fn base64_url_encode(input: &[u8]) -> String {
	const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
	let mut output = String::with_capacity(input.len().div_ceil(3) * 4);
	for chunk in input.chunks(3) {
		let bits = (u32::from(chunk[0]) << 16)
			| (u32::from(*chunk.get(1).unwrap_or(&0)) << 8)
			| u32::from(*chunk.get(2).unwrap_or(&0));
		output.push(char::from(TABLE[((bits >> 18) & 63) as usize]));
		output.push(char::from(TABLE[((bits >> 12) & 63) as usize]));
		if chunk.len() > 1 {
			output.push(char::from(TABLE[((bits >> 6) & 63) as usize]));
		}
		if chunk.len() > 2 {
			output.push(char::from(TABLE[(bits & 63) as usize]));
		}
	}
	output
}

fn base64_url_decode(input: &str) -> Result<Vec<u8>, OAuthError> {
	const fn value(byte: u8) -> Option<u8> {
		match byte {
			b'A'..=b'Z' => Some(byte - b'A'),
			b'a'..=b'z' => Some(byte - b'a' + 26),
			b'0'..=b'9' => Some(byte - b'0' + 52),
			b'-' | b'+' => Some(62),
			b'_' | b'/' => Some(63),
			_ => None,
		}
	}
	let mut output = Vec::with_capacity(input.len() * 3 / 4);
	let mut accumulator = 0_u32;
	let mut bits = 0_u8;
	for byte in input.bytes().take_while(|byte| *byte != b'=') {
		let digit =
			value(byte).ok_or_else(|| OAuthError::InvalidResponse("invalid JWT base64".into()))?;
		accumulator = (accumulator << 6) | u32::from(digit);
		bits += 6;
		if bits >= 8 {
			bits -= 8;
			output.push((accumulator >> bits) as u8);
			accumulator &= (1_u32 << bits).saturating_sub(1);
		}
	}
	Ok(output)
}

#[allow(
	clippy::many_single_char_names,
	reason = "SHA-256's standard compression state is conventionally named a through h"
)]
fn sha256(input: &[u8]) -> [u8; 32] {
	const INITIAL: [u32; 8] = [
		0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
		0x5be0cd19,
	];
	const K: [u32; 64] = [
		0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
		0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
		0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
		0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
		0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
		0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
		0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
		0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
		0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
		0xc67178f2,
	];
	let bit_len = (input.len() as u64).wrapping_mul(8);
	let padded_len = (input.len() + 9).div_ceil(64) * 64;
	let mut message = Vec::with_capacity(padded_len);
	message.extend_from_slice(input);
	message.push(0x80);
	message.resize(padded_len - 8, 0);
	message.extend_from_slice(&bit_len.to_be_bytes());
	let mut state = INITIAL;
	for block in message.as_chunks::<64>().0 {
		let mut words = [0_u32; 64];
		for (index, word) in words[..16].iter_mut().enumerate() {
			*word = u32::from_be_bytes(
				block[index * 4..index * 4 + 4]
					.try_into()
					.expect("four-byte SHA word"),
			);
		}
		for index in 16..64 {
			let s0 = words[index - 15].rotate_right(7)
				^ words[index - 15].rotate_right(18)
				^ (words[index - 15] >> 3);
			let s1 = words[index - 2].rotate_right(17)
				^ words[index - 2].rotate_right(19)
				^ (words[index - 2] >> 10);
			words[index] = words[index - 16]
				.wrapping_add(s0)
				.wrapping_add(words[index - 7])
				.wrapping_add(s1);
		}

		let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
		for index in 0..64 {
			let sum1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
			let choice = (e & f) ^ (!e & g);
			let temp1 = h
				.wrapping_add(sum1)
				.wrapping_add(choice)
				.wrapping_add(K[index])
				.wrapping_add(words[index]);
			let sum0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
			let majority = (a & b) ^ (a & c) ^ (b & c);
			let temp2 = sum0.wrapping_add(majority);
			h = g;
			g = f;
			f = e;
			e = d.wrapping_add(temp1);
			d = c;
			c = b;
			b = a;
			a = temp1.wrapping_add(temp2);
		}
		for (slot, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
			*slot = slot.wrapping_add(value);
		}
	}
	let mut output = [0_u8; 32];
	for (chunk, word) in output.as_chunks_mut::<4>().0.iter_mut().zip(state) {
		chunk.copy_from_slice(&word.to_be_bytes());
	}
	output
}

#[cfg(test)]
mod tests {
	use std::{
		collections::VecDeque,
		sync::atomic::{AtomicBool, AtomicU64, Ordering},
	};

	use super::*;
	use crate::store::CredentialState;

	struct ScriptedHttp {
		responses: Mutex<VecDeque<HttpResponse>>,
		requests:  Mutex<Vec<HttpRequest>>,
	}

	impl ScriptedHttp {
		fn new(responses: impl IntoIterator<Item = HttpResponse>) -> Self {
			Self {
				responses: Mutex::new(responses.into_iter().collect()),
				requests:  Mutex::new(Vec::new()),
			}
		}
	}

	impl HttpClient for ScriptedHttp {
		fn execute(&self, request: HttpRequest) -> HttpFuture<'_> {
			self.requests.lock().push(request);
			let response = self.responses.lock().pop_front();
			Box::pin(async move {
				response.ok_or_else(|| HttpError {
					detail:    "unexpected request".into(),
					transient: false,
				})
			})
		}
	}
	struct PendingHttp {
		started:  tokio::sync::Notify,
		dropped:  AtomicBool,
		requests: AtomicU64,
	}

	impl HttpClient for PendingHttp {
		fn execute(&self, _request: HttpRequest) -> HttpFuture<'_> {
			struct DropMarker<'a>(&'a AtomicBool);
			impl Drop for DropMarker<'_> {
				fn drop(&mut self) {
					self.0.store(true, Ordering::Relaxed);
				}
			}

			self.requests.fetch_add(1, Ordering::Relaxed);
			self.started.notify_one();
			Box::pin(async move {
				let _marker = DropMarker(&self.dropped);
				std::future::pending::<Result<HttpResponse, HttpError>>().await
			})
		}
	}

	struct FakeTime(AtomicU64);

	impl TimeSource for FakeTime {
		fn now_ms(&self) -> u64 {
			self.0.load(Ordering::Relaxed)
		}

		fn sleep(
			&self,
			duration: std::time::Duration,
		) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
			Box::pin(async move {
				self
					.0
					.fetch_add(duration.as_millis().try_into().unwrap_or(u64::MAX), Ordering::Relaxed);
			})
		}
	}

	fn response(status: u16, value: Value) -> HttpResponse {
		HttpResponse { status, body: Bytes::from(serde_json::to_vec(&value).expect("JSON response")) }
	}

	fn store() -> (tempfile::TempDir, Arc<Store>) {
		let directory = tempfile::tempdir().expect("temporary directory");
		let store =
			Arc::new(Store::open(directory.path().join("broker.sqlite")).expect("credential store"));
		(directory, store)
	}

	fn bearer(store: &Store, meta: &CredentialMeta) -> HeaderValue {
		let lease = store
			.lease(meta.id)
			.expect("lease query")
			.expect("active lease");
		store
			.redeem_with(lease.provider(), lease.credential_id(), lease.generation(), |auth| {
				let mut headers = HeaderMap::new();
				auth.apply_bearer_to_headers(&mut headers)?;
				Ok::<_, http::header::InvalidHeaderValue>(
					headers.remove(AUTHORIZATION).expect("authorization header"),
				)
			})
			.expect("redeem query")
			.expect("redeem lease")
			.expect("valid bearer value")
	}
	fn sealed(value: &str) -> Secret {
		Secret::new(value.as_bytes())
	}
	fn prompt_state(prompt: &LoginPrompt) -> String {
		let LoginPrompt::Browse { url, .. } = prompt else {
			panic!("expected browser prompt");
		};
		Url::parse(url)
			.expect("authorization URL")
			.query_pairs()
			.find_map(|(key, value)| (key == "state").then(|| value.into_owned()))
			.expect("state")
	}

	#[test]
	fn verifier_challenge_matches_rfc_7636_vector() {
		let verifier = b"dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";
		assert_eq!(
			base64_url_encode(&sha256(verifier)),
			"E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM",
		);
	}

	#[tokio::test]
	async fn callback_listener_accepts_both_loopback_families() {
		async fn callback_through(address: std::net::IpAddr) {
			let listeners = Arc::new(
				bind_callback_listeners("localhost", 0)
					.await
					.expect("IPv4 callback listener"),
			);
			if address.is_ipv6() && listeners.companion.is_none() {
				return;
			}
			let callback = {
				let listeners = listeners.clone();
				tokio::spawn(async move { receive_callback(&listeners, "expected-state").await })
			};
			let mut stream =
				tokio::net::TcpStream::connect(std::net::SocketAddr::new(address, listeners.port()))
					.await
					.expect("connect to callback listener");
			stream
				.write_all(
					b"GET /callback?code=accepted&state=expected-state HTTP/1.1\r\nHost: \
					 localhost\r\n\r\n",
				)
				.await
				.expect("write callback");
			let mut response = Vec::new();
			stream
				.read_to_end(&mut response)
				.await
				.expect("read callback response");
			assert!(response.starts_with(b"HTTP/1.1 200 OK"));
			let (code, state) = callback
				.await
				.expect("callback task")
				.expect("valid callback");
			assert_eq!(code, "accepted");
			assert_eq!(state, "expected-state");
		}

		callback_through(std::net::Ipv4Addr::LOCALHOST.into()).await;
		callback_through(std::net::Ipv6Addr::LOCALHOST.into()).await;
	}

	#[tokio::test]
	async fn exact_ipv6_loopback_collision_moves_from_the_fixed_port() {
		let Ok(squatter) = TcpListener::bind((std::net::Ipv6Addr::LOCALHOST, 0)).await else {
			return;
		};
		let port = squatter.local_addr().expect("IPv6 squatter address").port();
		let Err(error) = bind_callback_listeners("localhost", port).await else {
			panic!("fixed port must not remain IPv4-only after an IPv6 collision");
		};
		assert_eq!(error.kind(), std::io::ErrorKind::AddrInUse);
		let listeners = start_callback_listeners("localhost", port)
			.await
			.expect("random-port fallback");
		assert_ne!(listeners.port(), port);
		assert!(listeners.companion.is_some());
		TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, port))
			.await
			.expect("failed dual bind must release the IPv4 listener");
	}

	#[tokio::test]
	async fn pkce_happy_path_persists_token_and_rejects_state_mismatch() {
		let (_directory, store) = store();
		let http = Arc::new(ScriptedHttp::new([response(
			200,
			serde_json::json!({
				"access_token": "pkce-access",
				"refresh_token": "pkce-refresh",
				"expires_in": 3600,
			}),
		)]));
		let engine = OAuthEngine::new(store.clone(), http.clone()).expect("OAuth engine");
		let login = engine
			.begin_login("anthropic", 1_000)
			.await
			.expect("begin PKCE");
		let state = match &login.prompt {
			LoginPrompt::Browse { url, .. } => Url::parse(url)
				.expect("authorize URL")
				.query_pairs()
				.find_map(|(key, value)| (key == "state").then(|| value.into_owned()))
				.expect("state"),
			_ => panic!("expected browser prompt"),
		};
		let meta = engine
			.submit_code(login.flow_id.as_str(), &sealed("authorization-code"), &state, 1_000)
			.await
			.expect("exchange code");
		assert_eq!(bearer(&store, &meta), "Bearer pkce-access");

		let login = engine
			.begin_login("anthropic", 2_000)
			.await
			.expect("begin second PKCE");
		let error = engine
			.submit_code(login.flow_id.as_str(), &sealed("authorization-code"), "wrong-state", 2_000)
			.await
			.expect_err("state mismatch must fail");
		assert!(matches!(error, OAuthError::StateMismatch));
		assert_eq!(http.requests.lock().len(), 1);
	}

	#[tokio::test]
	async fn device_polling_honors_pending_then_succeeds_and_stops_at_expiry() {
		let (_directory, store) = store();
		let http = Arc::new(ScriptedHttp::new([
			response(
				200,
				serde_json::json!({
					"device_code": "device-secret",
					"user_code": "ABCD-EFGH",
					"verification_uri": "https://idp.test/device",
					"expires_in": 60,
					"interval": 1,
				}),
			),
			response(200, serde_json::json!({"error": "authorization_pending"})),
			response(
				200,
				serde_json::json!({
					"access_token": "device-access",
					"refresh_token": "device-refresh",
					"expires_in": 3600,
				}),
			),
			response(
				200,
				serde_json::json!({
					"device_code": "expiring-secret",
					"user_code": "IJKL-MNOP",
					"verification_uri": "https://idp.test/device",
					"expires_in": 1,
					"interval": 1,
				}),
			),
		]));
		let time = Arc::new(FakeTime(AtomicU64::new(10_000)));
		let engine = OAuthEngine::with_time_source(store.clone(), http.clone(), time.clone())
			.expect("OAuth engine");
		let login = engine
			.begin_login("xai", 10_000)
			.await
			.expect("begin device flow");
		let meta = engine
			.wait_login(login.flow_id.as_str(), 10_000)
			.await
			.expect("device authorization");
		assert_eq!(bearer(&store, &meta), "Bearer device-access");
		assert_eq!(time.now_ms(), 12_000);

		let login = engine
			.begin_login("xai", 12_000)
			.await
			.expect("begin expiring flow");
		let error = engine
			.wait_login(login.flow_id.as_str(), 12_000)
			.await
			.expect_err("expired device grant");
		assert!(matches!(error, OAuthError::LoginExpired));
		assert_eq!(http.requests.lock().len(), 4, "expired grant must not be polled");
	}

	#[tokio::test]
	async fn github_custom_exchange_persists_second_stage_token() {
		let (_directory, store) = store();
		let http = Arc::new(ScriptedHttp::new([response(
			200,
			serde_json::json!({"token": "copilot-session", "expires_at": 2_500}),
		)]));
		let engine = OAuthEngine::new(store.clone(), http.clone()).expect("OAuth engine");
		let params = engine
			.params("github-copilot")
			.expect("GitHub params")
			.clone();
		let source = TokenSet {
			access:                Secret::new(b"github-access"),
			refresh:               None,
			id_token:              None,
			expires_in_secs:       0,
			expires_at_epoch_secs: 0,
			identity:              Some("octocat".into()),
		};
		let meta = engine
			.persist_tokens(&params, source, 1_000)
			.await
			.expect("Copilot exchange");
		assert_eq!(bearer(&store, &meta), "Bearer copilot-session");
		assert_eq!(meta.expires_at_ms, 2_500_000);
		assert_eq!(http.requests.lock()[0].url.as_str(), COPILOT_TOKEN_URL);
	}

	#[tokio::test]
	async fn all_standard_pkce_rows_complete_against_mock_idp() {
		let (_directory, store) = store();
		let claims =
			base64_url_encode(br#"{"https://api.openai.com/auth":{"chatgpt_account_id":"acct-42"}}"#);
		let http = Arc::new(ScriptedHttp::new([
			response(
				200,
				serde_json::json!({
					"access_token": "anthropic-access",
					"refresh_token": "anthropic-refresh",
					"expires_in": 3600,
				}),
			),
			response(
				200,
				serde_json::json!({
					"access_token": "codex-access",
					"refresh_token": "codex-refresh",
					"id_token": format!("h.{claims}.s"),
					"expires_in": 3600,
				}),
			),
			response(
				200,
				serde_json::json!({
					"access_token": "gemini-access",
					"refresh_token": "gemini-refresh",
					"expires_in": 3600,
				}),
			),
			response(
				200,
				serde_json::json!({
					"access_token": "antigravity-access",
					"refresh_token": "antigravity-refresh",
					"expires_in": 3600,
				}),
			),
			response(
				200,
				serde_json::json!({
					"access_token": "gitlab-access",
					"refresh_token": "gitlab-refresh",
					"expires_in": 3600,
				}),
			),
		]));
		let engine = OAuthEngine::new(store.clone(), http.clone()).expect("OAuth engine");
		for (provider, expected) in [
			("anthropic", "Bearer anthropic-access"),
			("openai-codex", "Bearer codex-access"),
			("google-gemini-cli", "Bearer gemini-access"),
			("google-antigravity", "Bearer antigravity-access"),
			("gitlab-duo", "Bearer gitlab-access"),
		] {
			let login = engine
				.begin_login(provider, 1_000)
				.await
				.expect("PKCE start");
			let state = prompt_state(&login.prompt);
			let meta = engine
				.submit_code(login.flow_id.as_str(), &sealed("authorization-code"), &state, 1_000)
				.await
				.expect("PKCE exchange");
			assert_eq!(bearer(&store, &meta), expected);
		}
		for request in http.requests.lock().iter() {
			let body = std::str::from_utf8(&request.body).expect("PKCE body");
			if request.headers.get(CONTENT_TYPE) == Some(&HeaderValue::from_static(JSON_CONTENT_TYPE))
			{
				let value: Value = serde_json::from_slice(&request.body).expect("PKCE JSON");
				assert!(value.get("code_verifier").is_some());
				assert!(value.get("redirect_uri").is_some());
				assert!(value.get("state").is_some());
			} else {
				assert!(body.contains("code_verifier="));
				assert!(body.contains("redirect_uri="));
			}
		}
	}

	#[tokio::test]
	async fn all_device_rows_complete_against_mock_idp() {
		let (_directory, store) = store();
		let http = Arc::new(ScriptedHttp::new([
			response(
				200,
				serde_json::json!({
					"device_code": "github-device",
					"user_code": "GITHUB",
					"verification_uri": "https://idp.test/github",
					"expires_in": 60,
					"interval": 1,
				}),
			),
			response(
				200,
				serde_json::json!({
					"access_token": "github-source",
					"login": "octocat",
				}),
			),
			response(
				200,
				serde_json::json!({
					"token": "copilot-session",
					"expires_at": 2500,
				}),
			),
			response(
				200,
				serde_json::json!({
					"device_code": "xai-device",
					"user_code": "XAI",
					"verification_uri": "https://idp.test/xai",
					"expires_in": 60,
					"interval": 1,
				}),
			),
			response(
				200,
				serde_json::json!({
					"access_token": "xai-access",
					"refresh_token": "xai-refresh",
					"expires_in": 3600,
				}),
			),
			response(
				200,
				serde_json::json!({
					"device_code": "kimi-device",
					"user_code": "KIMI",
					"verification_uri": "https://idp.test/kimi",
					"expires_in": 60,
					"interval": 1,
				}),
			),
			response(
				200,
				serde_json::json!({
					"access_token": "kimi-access",
					"refresh_token": "kimi-refresh",
					"expires_in": 3600,
				}),
			),
		]));
		let time = Arc::new(FakeTime(AtomicU64::new(1_000)));
		let engine = OAuthEngine::with_time_source(store.clone(), http, time).expect("OAuth engine");
		for (provider, expected) in [
			("github-copilot", "Bearer copilot-session"),
			("xai", "Bearer xai-access"),
			("kimi", "Bearer kimi-access"),
		] {
			let login = engine
				.begin_login(provider, 1_000)
				.await
				.expect("device start");
			let meta = engine
				.wait_login(login.flow_id.as_str(), 1_000)
				.await
				.expect("device exchange");
			assert_eq!(bearer(&store, &meta), expected);
		}
	}

	#[tokio::test]
	async fn every_api_key_exchange_completes_without_echoing_input() {
		let (_directory, store) = store();
		let engine = OAuthEngine::new(store.clone(), Arc::new(ScriptedHttp::new(std::iter::empty())))
			.expect("OAuth engine");
		let api_key_providers = engine
			.params
			.iter()
			.filter(|params| matches!(params.exchange, Some(CustomExchange::ApiKeyPaste)))
			.map(|params| (params.provider.clone(), params.credential_provider.clone()))
			.collect::<Vec<_>>();
		assert_ne!(api_key_providers, [] as [(omp_core::Str, omp_core::Str); 0]);
		for (flow_provider, credential_provider) in api_key_providers {
			let login = engine
				.begin_login(flow_provider.as_str(), 1_000)
				.await
				.expect("paste start");
			assert_eq!(login.provider, flow_provider);
			let meta = engine
				.submit_code(login.flow_id.as_str(), &sealed("paste-key-secret"), "", 1_000)
				.await
				.expect("paste exchange");
			assert_eq!(meta.provider, credential_provider);
			assert_eq!(bearer(&store, &meta), "Bearer paste-key-secret");
			assert!(!format!("{meta:?}").contains("paste-key-secret"));
		}
	}

	#[tokio::test]
	async fn every_declared_login_engine_can_start() {
		let (_directory, store) = store();
		let http = Arc::new(ScriptedHttp::new([
			response(
				200,
				serde_json::json!({
					"device_code": "xai-device",
					"user_code": "XAI-CODE",
					"verification_uri": "https://idp.test/xai",
					"expires_in": 60,
				}),
			),
			response(
				200,
				serde_json::json!({
					"device_code": "kimi-device",
					"user_code": "KIMI-CODE",
					"verification_uri": "https://idp.test/kimi",
					"expires_in": 60,
				}),
			),
			response(
				200,
				serde_json::json!({
					"device_code": "github-device",
					"user_code": "GITHUB-CODE",
					"verification_uri": "https://idp.test/github",
					"expires_in": 60,
				}),
			),
		]));
		let engine = OAuthEngine::new(store, http).expect("OAuth engine");
		let providers = engine.providers().map(str::to_owned).collect::<Vec<_>>();
		assert_eq!(providers.len(), 18);
		for provider in providers {
			engine
				.begin_login(&provider, 1_000)
				.await
				.unwrap_or_else(|error| panic!("{provider} did not start: {error}"));
		}
	}

	#[tokio::test]
	async fn cursor_poll_and_expiry_follow_the_bounded_exchange() {
		let (_directory, store) = store();
		let payload = base64_url_encode(br#"{"sub":"auth0|cursor-user","exp":3600}"#);
		let access = format!("header.{payload}.signature");
		let http = Arc::new(ScriptedHttp::new([
			response(404, Value::Null),
			response(
				200,
				serde_json::json!({
					"accessToken": access,
					"refreshToken": "cursor-refresh-secret",
				}),
			),
			response(
				200,
				serde_json::json!({
					"accessToken": format!(
						"header.{}.signature",
						base64_url_encode(br#"{"sub":"auth0|cursor-user","exp":7200}"#),
					),
					"refreshToken": "cursor-refresh-rotated",
				}),
			),
		]));
		let time = Arc::new(FakeTime(AtomicU64::new(1_000)));
		let engine = OAuthEngine::with_time_source(store.clone(), http.clone(), time.clone())
			.expect("OAuth engine");
		let login = engine
			.begin_login("cursor", 1_000)
			.await
			.expect("Cursor login");
		let LoginPrompt::Browse { url: auth_url, loopback: false } = login.prompt else {
			panic!("Cursor must use browser polling");
		};
		let auth_url = Url::parse(&auth_url).expect("Cursor auth URL");
		assert!(auth_url.query_pairs().any(|(key, _)| key == "challenge"));
		assert!(auth_url.query_pairs().any(|(key, _)| key == "uuid"));
		let meta = engine
			.wait_login(login.flow_id.as_str(), 1_000)
			.await
			.expect("Cursor poll");
		assert!(
			bearer(&store, &meta)
				.to_str()
				.expect("bearer")
				.starts_with("Bearer header.")
		);
		assert_eq!(http.requests.lock().len(), 2);
		let refreshed = engine
			.refresh_credential(meta.id, meta.expires_at_ms)
			.await
			.expect("Cursor refresh");
		assert!(
			bearer(&store, &refreshed)
				.to_str()
				.expect("bearer")
				.starts_with("Bearer header.")
		);
		{
			let requests = http.requests.lock();
			assert_eq!(requests.len(), 3);
			assert_eq!(
				requests[2]
					.headers
					.get(AUTHORIZATION)
					.and_then(|value| value.to_str().ok()),
				Some("Bearer cursor-refresh-secret"),
			);
		}

		let expired = engine
			.begin_login("cursor", time.now_ms())
			.await
			.expect("Cursor login");
		time.0.store(expired.expires_at_ms, Ordering::Relaxed);
		let error = engine
			.wait_login(expired.flow_id.as_str(), time.now_ms())
			.await
			.expect_err("expired Cursor login");
		assert!(matches!(error, OAuthError::LoginExpired));
		assert_eq!(http.requests.lock().len(), 3, "expiry must not poll or fall back");
	}

	#[tokio::test]
	async fn devin_and_external_redirect_pkce_use_their_distinct_wire_shapes() {
		let (_directory, store) = store();
		let subject = base64_url_encode(br#"{"sub":"devin-user","exp":9999999999}"#);
		let http = Arc::new(ScriptedHttp::new([
			response(200, serde_json::json!({"token": format!("h.{subject}.s")})),
			response(
				200,
				serde_json::json!({
					"access_token": "gitlab-access",
					"refresh_token": "gitlab-refresh",
					"expires_in": 3600,
					"created_at": 100,
				}),
			),
		]));
		let engine = OAuthEngine::new(store.clone(), http.clone()).expect("OAuth engine");

		let devin = engine
			.begin_login("devin", 1_000)
			.await
			.expect("Devin login");
		let devin_state = prompt_state(&devin.prompt);
		let devin_meta = engine
			.submit_code(devin.flow_id.as_str(), &sealed("devin-code"), &devin_state, 1_000)
			.await
			.expect("Devin exchange");
		assert!(
			bearer(&store, &devin_meta)
				.to_str()
				.expect("bearer")
				.starts_with("Bearer h.")
		);
		{
			let requests = http.requests.lock();
			let devin_request = &requests[0];
			assert_eq!(
				devin_request
					.headers
					.get(CONTENT_TYPE)
					.expect("content type"),
				JSON_CONTENT_TYPE,
			);
			let devin_body: Value = serde_json::from_slice(&devin_request.body).expect("Devin JSON");
			assert_eq!(devin_body.get("code").and_then(Value::as_str), Some("devin-code"));
			assert!(
				devin_body
					.get("code_verifier")
					.and_then(Value::as_str)
					.is_some()
			);
		}

		let gitlab = engine
			.begin_login("gitlab-duo-workflow", 2_000)
			.await
			.expect("GitLab external redirect");
		let state = prompt_state(&gitlab.prompt);
		let callback =
			format!("vscode://gitlab.gitlab-workflow/authentication?code=gitlab-code&state={state}");
		let meta = engine
			.submit_code(gitlab.flow_id.as_str(), &sealed(&callback), "", 2_000)
			.await
			.expect("GitLab exchange");
		assert_eq!(meta.provider, "gitlab-duo-agent");
		assert_eq!(bearer(&store, &meta), "Bearer gitlab-access");
		{
			let requests = http.requests.lock();
			let body = std::str::from_utf8(&requests[1].body).expect("GitLab form");
			assert!(body.contains("redirect_uri=vscode%3A%2F%2Fgitlab.gitlab-workflow"));
			assert!(body.contains("code_verifier="));
		}
	}

	#[tokio::test]
	async fn zai_mints_durable_key_without_exposing_intermediate_tokens() {
		let (_directory, store) = store();
		let http = Arc::new(ScriptedHttp::new([
			response(
				200,
				serde_json::json!({
					"code": 0,
					"data": {
						"zai": { "access_token": "zai-oauth-intermediate-secret" },
						"user": { "email": "user@example.test" },
					},
				}),
			),
			response(
				200,
				serde_json::json!({
					"code": 200,
					"data": { "access_token": "zai-business-intermediate-secret" },
				}),
			),
			response(
				200,
				serde_json::json!({
					"code": 200,
					"data": { "organizations": [{
						"organizationId": "org",
						"isDefault": true,
						"projects": [{ "projectId": "project", "isDefault": true }],
					}] },
				}),
			),
			response(
				200,
				serde_json::json!({
					"code": 200,
					"data": [{ "name": "omp", "apiKey": "key-id" }],
				}),
			),
			response(
				200,
				serde_json::json!({
					"code": 200,
					"data": { "secretKey": "key-secret" },
				}),
			),
		]));
		let engine = OAuthEngine::new(store.clone(), http.clone()).expect("OAuth engine");
		let login = engine.begin_login("zai", 1_000).await.expect("Z.ai login");
		let state = prompt_state(&login.prompt);
		let meta = engine
			.submit_code(login.flow_id.as_str(), &sealed("zai-code"), &state, 1_000)
			.await
			.expect("Z.ai key mint");
		assert_eq!(bearer(&store, &meta), "Bearer key-id.key-secret");
		assert_eq!(http.requests.lock().len(), 5);
		for request in http.requests.lock().iter().skip(1) {
			assert_ne!(
				request
					.headers
					.get(AUTHORIZATION)
					.and_then(|value| value.to_str().ok()),
				Some("Bearer zai-oauth-intermediate-secret"),
			);
		}
	}

	#[tokio::test]
	async fn perplexity_email_otp_is_sealed_and_provider_rejections_are_redacted() {
		let (_directory, store) = store();
		let http = Arc::new(ScriptedHttp::new([
			response(200, serde_json::json!({"csrfToken": "csrf-intermediate-secret"})),
			response(200, Value::Null),
			response(200, serde_json::json!({"token": "perplexity-jwt-secret"})),
			response(
				400,
				serde_json::json!({
					"error": "access_denied",
					"error_description": "must-not-leak-token-value",
				}),
			),
		]));
		let engine = OAuthEngine::new(store.clone(), http).expect("OAuth engine");
		let login = engine
			.begin_login("perplexity", 1_000)
			.await
			.expect("Perplexity login");
		let meta = engine
			.submit_code(
				login.flow_id.as_str(),
				&sealed(r#"{"email":"user@example.test","otp":"123456"}"#),
				"",
				1_000,
			)
			.await
			.expect("Perplexity OTP");
		assert_eq!(bearer(&store, &meta), "Bearer perplexity-jwt-secret");

		let gitlab = engine
			.begin_login("gitlab-duo-workflow", 2_000)
			.await
			.expect("GitLab login");
		let state = prompt_state(&gitlab.prompt);
		let error = engine
			.submit_code(gitlab.flow_id.as_str(), &sealed("rejected-code"), &state, 2_000)
			.await
			.expect_err("provider rejection");
		let rendered = format!("{error:?} {error}");
		assert!(!rendered.contains("must-not-leak-token-value"));
		assert!(!rendered.contains("perplexity-jwt-secret"));
		let unknown = engine
			.begin_login("provider-id-must-not-be-echoed", 2_000)
			.await
			.expect_err("unknown provider");
		assert!(!format!("{unknown:?} {unknown}").contains("provider-id-must-not-be-echoed"));
	}

	#[tokio::test]
	async fn dropping_custom_poll_cancels_without_fallback() {
		let (_directory, store) = store();
		let http = Arc::new(PendingHttp {
			started:  tokio::sync::Notify::new(),
			dropped:  AtomicBool::new(false),
			requests: AtomicU64::new(0),
		});
		let engine = Arc::new(
			OAuthEngine::with_time_source(
				store,
				http.clone(),
				Arc::new(FakeTime(AtomicU64::new(1_000))),
			)
			.expect("OAuth engine"),
		);
		let login = engine
			.begin_login("cursor", 1_000)
			.await
			.expect("Cursor login");
		let worker = {
			let engine = engine.clone();
			tokio::spawn(async move { engine.wait_login(login.flow_id.as_str(), 1_000).await })
		};
		http.started.notified().await;
		worker.abort();
		let _ = worker.await;
		tokio::task::yield_now().await;
		assert!(http.dropped.load(Ordering::Relaxed));
		assert_eq!(http.requests.load(Ordering::Relaxed), 1, "cancellation must not fall back");
	}

	#[tokio::test]
	async fn permanent_refresh_failure_marks_credential_expired() {
		let (_directory, store) = store();
		let http = Arc::new(ScriptedHttp::new([response(
			400,
			serde_json::json!({"error": "invalid_grant"}),
		)]));
		let engine = OAuthEngine::new(store.clone(), http).expect("OAuth engine");
		let access = Secret::new(b"old-access");
		let refresh = Secret::new(b"dead-refresh");
		let meta = engine
			.import_oauth("anthropic", "user", &access, Some(&refresh), &Value::Null, 1_000, 2_000)
			.expect("import credential");
		let error = engine
			.refresh_credential(meta.id, 2_000)
			.await
			.expect_err("invalid grant");
		assert!(matches!(error, OAuthError::Provider { transient: false, .. }));
		let meta = store
			.get_credential(meta.id, 2_000)
			.expect("credential query")
			.expect("credential remains");
		assert_eq!(meta.state, CredentialState::Expired);
	}

	#[test]
	fn codex_claim_extracts_namespaced_account_id() {
		let claims =
			base64_url_encode(br#"{"https://api.openai.com/auth":{"chatgpt_account_id":"acct-42"}}"#);
		let token = Secret::new(format!("header.{claims}.signature").as_bytes());
		assert_eq!(codex_account_id(&token).expect("account id"), "acct-42");
	}
}
