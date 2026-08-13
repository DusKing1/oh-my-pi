//! Generic OAuth PKCE, device, paste, and refresh protocol engines.

use std::{
	sync::Arc,
	time::{Duration, SystemTime},
};

use bytes::{Bytes, BytesMut};
use futures::{
	FutureExt,
	future::{BoxFuture, Either, select},
};
use http::{
	HeaderMap, HeaderValue, Method, Request,
	header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE},
};
use http_body_util::{BodyExt as _, Full};
use hyper_rustls::{HttpsConnector, HttpsConnectorBuilder};
use hyper_util::{
	client::legacy::{Client, connect::HttpConnector},
	rt::TokioExecutor,
};
use omp_core::{Str, base64_url};
use omp_llm_catalog::provider::PrincipalResolution;
use ring::rand::{SecureRandom, SystemRandom};
use secrecy::{ExposeSecret, SecretBox, SecretString};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use url::Url;
use zeroize::Zeroizing;

use super::{
	lease::{AuthRejection, AuthRejectionKind, CredentialLease, LeaseMeta},
	login::{LoginChannelError, LoginDriver},
	spec::{
		OAuthClientSpec, OAuthCustomSpec, OAuthDeviceSpec, OAuthParameter, OAuthPasteSpec,
		OAuthPkceSpec, OAuthRefreshSpec, PkceCompletion,
	},
	store::{CredentialOrigin, CredentialStore, OAuthCredentialWrite, StoreError},
};
use crate::{
	account::{
		CredentialFreshness, RefreshCoordinator, RefreshError, RefreshOperationError, RefreshOutcome,
		RefreshRequest, RefreshedCredential,
	},
	answer::{AuthEvent, AuthPrompt, AuthPromptKind},
	call::AuthInput,
	id::{AccountId, PrincipalId},
};

const FORM_CONTENT_TYPE: &str = "application/x-www-form-urlencoded";
const DEVICE_GRANT: &str = "urn:ietf:params:oauth:grant-type:device_code";

/// Secret-bearing OAuth request handed directly to an injected HTTP transport.
pub struct OAuthHttpRequest {
	method:  Method,
	url:     Url,
	headers: HeaderMap,
	body:    Option<SecretString>,
}

impl OAuthHttpRequest {
	/// Consumes the request into transport-ready parts while preserving body
	/// secrecy.
	#[must_use]
	pub fn into_parts(self) -> (Method, Url, HeaderMap, Option<SecretString>) {
		(self.method, self.url, self.headers, self.body)
	}

	/// Creates a secret-bearing request at a protocol-engine boundary.
	pub(crate) fn new(
		method: Method,
		url: &str,
		mut headers: HeaderMap,
		body: Option<SecretString>,
	) -> Result<Self, OAuthError> {
		let url = parse_http_url(url)?;
		headers
			.entry(ACCEPT)
			.or_insert(HeaderValue::from_static("application/json"));
		Ok(Self { method, url, headers, body })
	}

	/// Creates a form-encoded secret POST request for another auth engine.
	pub(crate) fn secret_form(url: &str, body: SecretString) -> Result<Self, OAuthError> {
		let mut headers = HeaderMap::new();
		headers.insert(CONTENT_TYPE, HeaderValue::from_static(FORM_CONTENT_TYPE));
		Self::new(Method::POST, url, headers, Some(body))
	}
}

/// Secret-bearing OAuth response returned only to the protocol engine.
pub struct OAuthHttpResponse {
	/// HTTP-like response status.
	pub status:  u16,
	/// Response headers used only by protocol-specific transport policy.
	pub headers: HeaderMap,
	/// Bounded response body, which may contain access and refresh tokens.
	pub body:    SecretString,
}

/// Cold I/O boundary used by every generic OAuth engine.
pub trait OAuthHttpClient: Send + Sync {
	/// Executes one OAuth request without logging its secret body.
	fn execute(
		&self,
		request: OAuthHttpRequest,
	) -> BoxFuture<'_, Result<OAuthHttpResponse, OAuthTransportError>>;
}

const MAX_OAUTH_RESPONSE_BYTES: usize = 1024 * 1024;
type PooledOAuthClient = Client<HttpsConnector<HttpConnector>, Full<Bytes>>;

/// Production OAuth client using the workspace rustls root set and bounded
/// responses.
#[derive(Clone)]
pub struct SystemOAuthHttpClient {
	inner: PooledOAuthClient,
}

impl SystemOAuthHttpClient {
	/// Constructs a pooled HTTP/1.1 and HTTP/2 OAuth client.
	#[must_use]
	pub fn new() -> Self {
		let _ = rustls::crypto::ring::default_provider().install_default();
		let connector = HttpsConnectorBuilder::new()
			.with_webpki_roots()
			.https_or_http()
			.enable_http1()
			.enable_http2()
			.build();
		let inner = Client::builder(TokioExecutor::new()).build(connector);
		Self { inner }
	}
}

impl Default for SystemOAuthHttpClient {
	fn default() -> Self {
		Self::new()
	}
}

impl std::fmt::Debug for SystemOAuthHttpClient {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		formatter.write_str("SystemOAuthHttpClient(..)")
	}
}

impl OAuthHttpClient for SystemOAuthHttpClient {
	fn execute(
		&self,
		request: OAuthHttpRequest,
	) -> BoxFuture<'_, Result<OAuthHttpResponse, OAuthTransportError>> {
		let client = self.inner.clone();
		async move {
			let (method, url, headers, body) = request.into_parts();
			let body = body.as_ref().map_or_else(Bytes::new, |body| {
				Bytes::copy_from_slice(body.expose_secret().as_bytes())
			});
			let mut outbound = Request::builder()
				.method(method)
				.uri(url.as_str())
				.body(Full::new(body))
				.map_err(|_| OAuthTransportError)?;
			*outbound.headers_mut() = headers;
			let response = client
				.request(outbound)
				.await
				.map_err(|_| OAuthTransportError)?;
			let status = response.status().as_u16();
			let headers = response.headers().clone();
			let mut incoming = response.into_body();
			let mut bytes = BytesMut::new();
			while let Some(frame) = incoming.frame().await {
				let frame = frame.map_err(|_| OAuthTransportError)?;
				if let Some(data) = frame.data_ref() {
					if bytes.len().saturating_add(data.len()) > MAX_OAUTH_RESPONSE_BYTES {
						return Err(OAuthTransportError);
					}
					bytes.extend_from_slice(data);
				}
			}
			let body = String::from_utf8(bytes.to_vec()).map_err(|_| OAuthTransportError)?;
			Ok(OAuthHttpResponse { status, headers, body: SecretString::from(body) })
		}
		.boxed()
	}
}

/// Production wall clock and bounded asynchronous sleeper for OAuth polling.
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemOAuthClock;

impl OAuthClock for SystemOAuthClock {
	fn now(&self) -> SystemTime {
		SystemTime::now()
	}

	fn sleep(&self, duration: Duration) -> BoxFuture<'_, ()> {
		async move { tokio::time::sleep(duration).await }.boxed()
	}
}

/// Injectable clock and bounded sleeping used by device polling.
pub trait OAuthClock: Send + Sync {
	/// Current wall clock used for expiry calculations.
	fn now(&self) -> SystemTime;
	/// Sleeps for one server-bounded polling interval.
	fn sleep(&self, duration: Duration) -> BoxFuture<'_, ()>;
}

/// Injectable cryptographic entropy used by PKCE and state generation.
pub trait OAuthEntropy: Send + Sync {
	/// Fills the destination with cryptographically secure bytes.
	fn fill(&self, destination: &mut [u8]) -> Result<(), OAuthError>;
}

/// Operating-system cryptographic entropy source.
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemEntropySource;

impl OAuthEntropy for SystemEntropySource {
	fn fill(&self, destination: &mut [u8]) -> Result<(), OAuthError> {
		SystemRandom::new()
			.fill(destination)
			.map_err(|_| OAuthError::Entropy)
	}
}

/// One typed custom OAuth exchange implementation.
pub trait OAuthCustomHandler: Send + Sync {
	/// Exact catalog engine discriminator handled by this implementation.
	fn exchange_kind(&self) -> omp_llm_catalog::provider::OAuthExchangeKind;

	/// Runs the typed exchange over the bounded login channel.
	fn exchange<'a>(
		&'a self,
		spec: &'a OAuthCustomSpec,
		driver: &'a LoginDriver,
	) -> BoxFuture<'a, Result<OAuthTokenSet, OAuthError>>;
}

/// Registry dispatching custom OAuth strictly by catalog exchange enum.
#[derive(Default)]
pub struct OAuthCustomDispatcher {
	handlers: Vec<Arc<dyn OAuthCustomHandler>>,
}

impl OAuthCustomDispatcher {
	/// Constructs an empty dispatcher.
	#[must_use]
	pub const fn new() -> Self {
		Self { handlers: Vec::new() }
	}

	/// Registers one handler, rejecting duplicate typed discriminators.
	pub fn register(
		&mut self,
		handler: Arc<dyn OAuthCustomHandler>,
	) -> Result<(), OAuthCustomDispatchError> {
		let kind = handler.exchange_kind();
		if self
			.handlers
			.iter()
			.any(|candidate| candidate.exchange_kind() == kind)
		{
			return Err(OAuthCustomDispatchError::Duplicate(kind));
		}
		self.handlers.push(handler);
		Ok(())
	}

	/// Dispatches exactly the catalog-selected exchange or fails planning
	/// safely.
	pub async fn exchange(
		&self,
		spec: &OAuthCustomSpec,
		driver: &LoginDriver,
	) -> Result<OAuthTokenSet, OAuthCustomDispatchError> {
		let handler = self
			.handlers
			.iter()
			.find(|handler| handler.exchange_kind() == spec.exchange)
			.ok_or(OAuthCustomDispatchError::Unavailable(spec.exchange))?;
		handler
			.exchange(spec, driver)
			.await
			.map_err(OAuthCustomDispatchError::Protocol)
	}
}

impl std::fmt::Debug for OAuthCustomDispatcher {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		formatter
			.debug_struct("OAuthCustomDispatcher")
			.field(
				"exchange_kinds",
				&self
					.handlers
					.iter()
					.map(|handler| handler.exchange_kind())
					.collect::<Vec<_>>(),
			)
			.finish()
	}
}

/// Typed custom OAuth dispatch failure.
#[derive(Debug, thiserror::Error)]
pub enum OAuthCustomDispatchError {
	/// A handler for the exact catalog exchange was already registered.
	#[error("duplicate custom OAuth exchange handler for {0}")]
	Duplicate(omp_llm_catalog::provider::OAuthExchangeKind),
	/// No handler constructs the exact advertised exchange.
	#[error("custom OAuth exchange handler is unavailable for {0}")]
	Unavailable(omp_llm_catalog::provider::OAuthExchangeKind),
	/// The selected exchange failed with secret-free protocol evidence.
	#[error(transparent)]
	Protocol(#[from] OAuthError),
}

/// Data-driven OAuth protocol engine.
pub struct OAuthEngine<'a, C, K, R = SystemEntropySource> {
	http:    &'a C,
	clock:   &'a K,
	entropy: R,
}

impl<'a, C, K> OAuthEngine<'a, C, K, SystemEntropySource> {
	/// Constructs an engine using operating-system cryptographic entropy.
	#[must_use]
	pub fn new(http: &'a C, clock: &'a K) -> Self {
		Self { http, clock, entropy: SystemEntropySource }
	}
}

impl<'a, C, K, R> OAuthEngine<'a, C, K, R>
where
	C: OAuthHttpClient,
	K: OAuthClock,
	R: OAuthEntropy,
{
	/// Constructs an engine with deterministic injectable entropy.
	#[must_use]
	pub fn with_entropy(http: &'a C, clock: &'a K, entropy: R) -> Self {
		Self { http, clock, entropy }
	}

	/// Starts a PKCE flow and emits browser/prompt events.
	pub async fn begin_pkce(
		&self,
		spec: &OAuthPkceSpec,
		driver: &LoginDriver,
	) -> Result<PkcePending, OAuthError> {
		let mut verifier_bytes = Zeroizing::new([0_u8; 32]);
		let mut state_bytes = Zeroizing::new([0_u8; 24]);
		self.entropy.fill(&mut verifier_bytes[..])?;
		self.entropy.fill(&mut state_bytes[..])?;
		let verifier = SecretString::from(base64_url::encode_raw(&verifier_bytes[..]).into_string());
		let state = Str::from(base64_url::encode_raw(&state_bytes[..]).into_string());
		let challenge =
			base64_url::encode_raw(&Sha256::digest(verifier.expose_secret().as_bytes())).into_string();
		let mut url = parse_http_url(&spec.authorize_url)?;
		{
			let mut query = url.query_pairs_mut();
			query
				.append_pair("response_type", "code")
				.append_pair("client_id", &spec.client.client_id)
				.append_pair("redirect_uri", &spec.redirect_uri)
				.append_pair("code_challenge", &challenge)
				.append_pair("code_challenge_method", "S256")
				.append_pair("state", &state);
			if !spec.client.scopes.is_empty() {
				let scope = spec
					.client
					.scopes
					.iter()
					.map(Str::as_str)
					.collect::<Vec<_>>()
					.join(" ");
				query.append_pair("scope", &scope);
			}
			if let Some(audience) = &spec.client.audience {
				query.append_pair("audience", audience);
			}
			for parameter in &spec.authorize_params {
				query.append_pair(&parameter.name, &parameter.value);
			}
		}
		driver.emit(AuthEvent::OpenUrl(url.as_str().into())).await?;
		let (id, message, input) = match spec.completion {
			PkceCompletion::CallbackUrl => (
				"oauth-callback",
				"Complete authorization in the opened browser",
				AuthPromptKind::Confirmation,
			),
			PkceCompletion::PasteCallbackUrl => (
				"oauth-callback-url",
				"Paste the complete authorization callback URL",
				AuthPromptKind::AuthorizationCode,
			),
			PkceCompletion::PasteCode => {
				("oauth-code", "Paste the authorization code", AuthPromptKind::AuthorizationCode)
			},
		};
		driver
			.emit(AuthEvent::Prompt(AuthPrompt { id: id.into(), message: message.into(), input }))
			.await?;
		Ok(PkcePending {
			verifier,
			state,
			redirect_uri: spec.redirect_uri.clone(),
			completion: spec.completion,
		})
	}

	/// Completes a PKCE exchange from typed login input.
	pub async fn complete_pkce(
		&self,
		spec: &OAuthPkceSpec,
		pending: PkcePending,
		input: AuthInput,
	) -> Result<OAuthTokenSet, OAuthError> {
		let code = match (pending.completion, input) {
			(PkceCompletion::PasteCode, AuthInput::AuthorizationCode(code)) => code,
			(
				PkceCompletion::CallbackUrl | PkceCompletion::PasteCallbackUrl,
				AuthInput::CallbackUrl(callback),
			) => callback_code(&callback, &pending.state)?,
			(_, AuthInput::Cancel) => return Err(OAuthError::Cancelled),
			_ => return Err(OAuthError::UnexpectedInput),
		};
		let fields = vec![
			("grant_type", FormValue::Public("authorization_code")),
			("client_id", FormValue::Public(&spec.client.client_id)),
			("code", FormValue::Secret(code.expose_secret())),
			("redirect_uri", FormValue::Public(&pending.redirect_uri)),
			("code_verifier", FormValue::Secret(pending.verifier.expose_secret())),
		];
		self.exchange(&spec.client, fields, None).await
	}

	/// Starts device authorization and emits its typed device-code timeline.
	pub async fn begin_device(
		&self,
		spec: &OAuthDeviceSpec,
		driver: &LoginDriver,
	) -> Result<DevicePending, OAuthError> {
		let fields = vec![("client_id", FormValue::Public(&spec.client.client_id))];
		let response = self
			.http
			.execute(form_request(&spec.device_authorization_url, &fields, &spec.client.token_params)?)
			.await?;
		if !(200..300).contains(&response.status) {
			return Err(provider_error(response.status, &response.body, false));
		}
		let parsed: DeviceAuthorizationResponse = decode(&response.body)?;
		let device_code = parsed.device_code.ok_or(OAuthError::MalformedResponse)?;
		let user_code = parsed.user_code.ok_or(OAuthError::MalformedResponse)?;
		let verification_url = parsed
			.verification_uri
			.or(parsed.verification_url)
			.ok_or(OAuthError::MalformedResponse)?;
		parse_http_url(&verification_url)?;
		driver
			.emit(AuthEvent::ShowDeviceCode {
				code:             SecretString::from(user_code),
				verification_url: verification_url.into(),
			})
			.await?;
		if let Some(complete) = parsed.verification_uri_complete {
			parse_http_url(&complete)?;
			driver.emit(AuthEvent::OpenUrl(complete.into())).await?;
		}
		let interval =
			Duration::from_secs(parsed.interval.unwrap_or(spec.default_interval.as_secs()));
		let interval = interval.max(spec.default_interval).min(spec.max_interval);
		let expires_in = Duration::from_secs(parsed.expires_in.unwrap_or(600));
		let expires_at = self
			.clock
			.now()
			.checked_add(expires_in)
			.ok_or(OAuthError::InvalidExpiry)?;
		Ok(DevicePending {
			device_code: SecretString::from(device_code),
			interval,
			expires_at,
			polls: 0,
		})
	}

	/// Polls a device grant with catalog bounds, server slow-down, and
	/// cancellation.
	pub async fn poll_device(
		&self,
		spec: &OAuthDeviceSpec,
		mut pending: DevicePending,
		driver: &LoginDriver,
	) -> Result<OAuthTokenSet, OAuthError> {
		loop {
			driver.check_cancelled()?;
			match driver.try_receive()? {
				None | Some(AuthInput::DeviceConfirmed) => {},
				Some(_) => return Err(OAuthError::UnexpectedInput),
			}
			if pending.polls >= spec.max_polls || self.clock.now() >= pending.expires_at {
				return Err(OAuthError::PollingExhausted { polls: pending.polls });
			}
			driver.emit(AuthEvent::Waiting).await?;
			let sleep = self.clock.sleep(pending.interval).fuse();
			let cancelled = driver.wait_cancelled().fuse();
			futures::pin_mut!(sleep, cancelled);
			if matches!(select(sleep, cancelled).await, Either::Right(_)) {
				return Err(OAuthError::Cancelled);
			}
			driver.check_cancelled()?;
			pending.polls = pending.polls.saturating_add(1);
			let fields = vec![
				("grant_type", FormValue::Public(DEVICE_GRANT)),
				("device_code", FormValue::Secret(pending.device_code.expose_secret())),
				("client_id", FormValue::Public(&spec.client.client_id)),
			];
			let response = self
				.http
				.execute(form_request(&spec.client.token_url, &fields, &spec.client.token_params)?)
				.await?;
			if (200..300).contains(&response.status) {
				return token_response(response, None);
			}
			match provider_code(&response.body) {
				OAuthProviderCode::AuthorizationPending => continue,
				OAuthProviderCode::SlowDown => {
					pending.interval = pending
						.interval
						.saturating_add(Duration::from_secs(5))
						.min(spec.max_interval);
				},
				_ => return Err(provider_error(response.status, &response.body, false)),
			}
		}
	}

	/// Starts a browser-assisted paste flow.
	pub async fn begin_paste(
		&self,
		spec: &OAuthPasteSpec,
		driver: &LoginDriver,
	) -> Result<(), OAuthError> {
		parse_http_url(&spec.authorization_url)?;
		driver
			.emit(AuthEvent::OpenUrl(spec.authorization_url.clone()))
			.await?;
		driver
			.emit(AuthEvent::Prompt(AuthPrompt {
				id:      "oauth-paste-code".into(),
				message: spec.prompt.clone(),
				input:   AuthPromptKind::AuthorizationCode,
			}))
			.await?;
		Ok(())
	}

	/// Exchanges a pasted authorization code using standard OAuth form fields.
	pub async fn complete_paste(
		&self,
		spec: &OAuthPasteSpec,
		input: AuthInput,
	) -> Result<OAuthTokenSet, OAuthError> {
		let AuthInput::AuthorizationCode(code) = input else {
			return if matches!(input, AuthInput::Cancel) {
				Err(OAuthError::Cancelled)
			} else {
				Err(OAuthError::UnexpectedInput)
			};
		};
		let fields = vec![
			("grant_type", FormValue::Public("authorization_code")),
			("client_id", FormValue::Public(&spec.client.client_id)),
			("code", FormValue::Secret(code.expose_secret())),
		];
		self.exchange(&spec.client, fields, None).await
	}

	/// Refreshes an access token while preserving refresh-token continuity.
	pub async fn refresh(
		&self,
		client: &OAuthClientSpec,
		refresh_token: SecretString,
	) -> Result<OAuthTokenSet, OAuthError> {
		let (url, parameters) = match &client.refresh {
			OAuthRefreshSpec::Unsupported => return Err(OAuthError::RefreshUnsupported),
			OAuthRefreshSpec::TokenEndpoint => (&client.token_url, client.token_params.as_slice()),
			OAuthRefreshSpec::Endpoint { url, parameters } => (url, parameters.as_slice()),
		};
		let fields = vec![
			("grant_type", FormValue::Public("refresh_token")),
			("client_id", FormValue::Public(&client.client_id)),
			("refresh_token", FormValue::Secret(refresh_token.expose_secret())),
		];
		let response = self
			.http
			.execute(form_request(url, &fields, parameters)?)
			.await?;
		if !(200..300).contains(&response.status) {
			return Err(provider_error(response.status, &response.body, true));
		}
		token_response(response, Some(refresh_token))
	}

	async fn exchange(
		&self,
		client: &OAuthClientSpec,
		fields: Vec<(&str, FormValue<'_>)>,
		fallback_refresh: Option<SecretString>,
	) -> Result<OAuthTokenSet, OAuthError> {
		let response = self
			.http
			.execute(form_request(&client.token_url, &fields, &client.token_params)?)
			.await?;
		if !(200..300).contains(&response.status) {
			return Err(provider_error(response.status, &response.body, fallback_refresh.is_some()));
		}
		token_response(response, fallback_refresh)
	}
}

/// Pending state for one PKCE login; formatting is always redacted.
pub struct PkcePending {
	verifier:     SecretString,
	state:        Str,
	redirect_uri: Str,
	completion:   PkceCompletion,
}

impl<'a, C, K, R> OAuthEngine<'a, C, K, R>
where
	C: OAuthHttpClient,
	K: OAuthClock,
	R: OAuthEntropy,
{
	/// Persists a successful interactive OAuth result as one opaque renewable
	/// bundle.
	pub fn persist_login(
		&self,
		store: &CredentialStore,
		tokens: OAuthTokenSet,
		meta: &LeaseMeta,
		origin: CredentialOrigin,
		issued_at: SystemTime,
	) -> Result<CredentialFreshness, OAuthCredentialManagerError> {
		let expires_at = tokens
			.expires_in()
			.and_then(|duration| issued_at.checked_add(duration));
		let bundle = tokens.into_renewable_bundle()?.encode()?;
		let now_ms = unix_millis(issued_at)?;
		let expires_at_ms = expires_at.map(unix_millis).transpose()?;
		let stored = store.put_oauth_bundle(OAuthCredentialWrite {
			account_id: &meta.account,
			principal_id: &meta.principal,
			bundle: &bundle,
			expires_at_ms,
			origin,
			now_ms,
			expected_generation: None,
		})?;
		Ok(CredentialFreshness {
			generation: stored.generation,
			issued_at: Some(issued_at),
			expires_at,
			observed_at: issued_at,
		})
	}

	/// Loads a persisted renewable access token as an opaque request lease.
	pub fn lease_persisted(
		&self,
		store: &CredentialStore,
		account: &AccountId,
		now: SystemTime,
	) -> Result<CredentialLease, OAuthCredentialManagerError> {
		let stored = store.load_oauth_bundle(account)?;
		let bundle = RenewableCredentialBundle::decode(&stored.bundle)?;
		let expires_at = stored
			.metadata
			.expires_at_ms
			.map(system_time_from_millis)
			.transpose()?;
		if expires_at.is_some_and(|expires_at| expires_at <= now) {
			return Err(OAuthCredentialManagerError::Expired);
		}
		let meta = LeaseMeta {
			account: stored.metadata.account_id,
			principal: stored.metadata.principal_id,
			generation: stored.metadata.generation,
			expires_at,
		};
		Ok(CredentialLease::bearer(meta, bundle.access_token))
	}

	/// Refreshes and fenced-persists one rejected OAuth generation through the
	/// shared process/cross-process coordinator.
	pub async fn refresh_persisted(
		&self,
		coordinator: &RefreshCoordinator,
		store: std::sync::Arc<CredentialStore>,
		client: OAuthClientSpec,
		request: RefreshRequest,
		origin: CredentialOrigin,
	) -> Result<RefreshOutcome, OAuthCredentialManagerError> {
		let engine = self;
		let account = request.account.clone();
		let principal = request.principal.clone();
		let rejected_generation = request.rejected.generation;
		let requested_at = request.requested_at;
		coordinator
			.refresh(store.clone(), request, move |refresh_lease| {
				let store = store.clone();
				async move {
					let stored = store
						.load_oauth_bundle(&account)
						.map_err(refresh_store_operation)?;
					if stored.metadata.generation != rejected_generation {
						return Err(RefreshOperationError {
							code:    "generation-changed".into(),
							summary: "credential generation changed before refresh".into(),
						});
					}
					if stored.metadata.principal_id != principal {
						return Err(RefreshOperationError {
							code:    "principal-changed".into(),
							summary: "credential principal changed before refresh".into(),
						});
					}
					let bundle = RenewableCredentialBundle::decode(&stored.bundle)
						.map_err(refresh_oauth_operation)?;
					let tokens = engine
						.refresh(&client, bundle.into_refresh())
						.await
						.map_err(refresh_oauth_operation)?;
					let expires_at = tokens
						.expires_in()
						.and_then(|duration| requested_at.checked_add(duration));
					let bundle = tokens
						.into_renewable_bundle()
						.map_err(refresh_oauth_operation)?
						.encode()
						.map_err(refresh_oauth_operation)?;
					let write = OAuthCredentialWrite {
						account_id: &account,
						principal_id: &principal,
						bundle: &bundle,
						expires_at_ms: expires_at
							.map(unix_millis)
							.transpose()
							.map_err(refresh_manager_operation)?,
						origin,
						now_ms: unix_millis(requested_at).map_err(refresh_manager_operation)?,
						expected_generation: Some(rejected_generation),
					};
					let metadata = store
						.put_oauth_bundle_under_refresh_lease(write, &refresh_lease, requested_at)
						.map_err(refresh_store_operation)?;
					Ok(RefreshedCredential {
						account:   metadata.account_id,
						principal: metadata.principal_id,
						freshness: CredentialFreshness {
							generation: metadata.generation,
							issued_at: Some(requested_at),
							expires_at,
							observed_at: requested_at,
						},
					})
				}
			})
			.await
			.map_err(OAuthCredentialManagerError::Refresh)
	}
}

impl std::fmt::Debug for PkcePending {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		formatter
			.debug_struct("PkcePending")
			.field("verifier", &"[REDACTED]")
			.field("state", &"[REDACTED]")
			.field("redirect_uri", &self.redirect_uri)
			.field("completion", &self.completion)
			.finish()
	}
}

/// Pending state for bounded device-code polling; formatting is redacted.
pub struct DevicePending {
	device_code: SecretString,
	interval:    Duration,
	expires_at:  SystemTime,
	polls:       u16,
}

impl std::fmt::Debug for DevicePending {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		formatter
			.debug_struct("DevicePending")
			.field("device_code", &"[REDACTED]")
			.field("interval", &self.interval)
			.field("expires_at", &self.expires_at)
			.field("polls", &self.polls)
			.finish()
	}
}

/// Secret-bearing OAuth result with no plaintext accessor or serialization.
pub struct OAuthTokenSet {
	access_token:      SecretString,
	refresh_token:     Option<SecretString>,
	token_type:        Str,
	expires_in:        Option<Duration>,
	identity_response: SecretString,
}

impl OAuthTokenSet {
	/// Returns whether the response contains a renewable grant.
	#[must_use]
	pub fn is_refreshable(&self) -> bool {
		self.refresh_token.is_some()
	}

	/// Returns the non-secret token type evidence.
	#[must_use]
	pub fn token_type(&self) -> &str {
		&self.token_type
	}

	/// Returns the relative lifetime reported by the token endpoint.
	#[must_use]
	pub const fn expires_in(&self) -> Option<Duration> {
		self.expires_in
	}

	/// Resolves the authenticated principal using only the catalog-selected
	/// rule.
	pub async fn resolve_principal<C: OAuthHttpClient>(
		&self,
		resolution: &PrincipalResolution,
		http: &C,
	) -> Result<PrincipalId, OAuthError> {
		let value = match resolution {
			PrincipalResolution::StaticLabel { label } => label.clone(),
			PrincipalResolution::TokenResponseField { pointer } => {
				json_string_at(self.identity_response.expose_secret(), pointer)?
			},
			PrincipalResolution::IdTokenClaim { claim } => {
				let id_token = json_string_at(self.identity_response.expose_secret(), "/id_token")?;
				let payload = id_token
					.split(".")
					.nth(1)
					.ok_or(OAuthError::PrincipalUnresolved)?;
				let decoded = Zeroizing::new(
					base64_url::decode_raw(payload.as_bytes())
						.into_vec()
						.map_err(|_| OAuthError::PrincipalUnresolved)?,
				);
				let claims =
					std::str::from_utf8(&decoded).map_err(|_| OAuthError::PrincipalUnresolved)?;
				json_object_string(claims, claim)?
			},
			PrincipalResolution::UserinfoEndpoint { url, field } => {
				let mut headers = HeaderMap::new();
				let mut authorization = Zeroizing::new(String::with_capacity(
					self.token_type.len() + self.access_token.expose_secret().len() + 1,
				));
				authorization.push_str(&self.token_type);
				authorization.push(' ');
				authorization.push_str(self.access_token.expose_secret());
				let mut header = HeaderValue::from_str(&authorization)
					.map_err(|_| OAuthError::PrincipalUnresolved)?;
				header.set_sensitive(true);
				headers.insert(AUTHORIZATION, header);
				let response = http
					.execute(OAuthHttpRequest::new(Method::GET, url, headers, None)?)
					.await?;
				if !(200..300).contains(&response.status) {
					return Err(OAuthError::PrincipalUnresolved);
				}
				json_object_string(response.body.expose_secret(), field)?
			},
		};
		if value.is_empty() {
			return Err(OAuthError::PrincipalUnresolved);
		}
		Ok(PrincipalId::from(value))
	}

	/// Converts a non-renewable access token into an ephemeral opaque lease.
	///
	/// Renewable results must use [`Self::into_renewable_bundle`] so their
	/// refresh grant is persisted atomically instead of being discarded.
	pub fn into_ephemeral_lease(
		self,
		mut meta: LeaseMeta,
		issued_at: SystemTime,
	) -> Result<CredentialLease, OAuthError> {
		if self.refresh_token.is_some() {
			return Err(OAuthError::RenewableCredentialRequiresPersistence);
		}
		if meta.expires_at.is_none() {
			meta.expires_at = self
				.expires_in
				.and_then(|duration| issued_at.checked_add(duration));
		}
		Ok(CredentialLease::bearer(meta, self.access_token))
	}

	/// Moves a renewable result into the opaque persistence bundle.
	pub(crate) fn into_renewable_bundle(self) -> Result<RenewableCredentialBundle, OAuthError> {
		let refresh_token = self.refresh_token.ok_or(OAuthError::MissingRefreshToken)?;
		Ok(RenewableCredentialBundle {
			access_token: self.access_token,
			refresh_token,
			token_type: self.token_type,
			expires_in: self.expires_in,
		})
	}
}

/// Move-only renewable OAuth material owned inside the auth boundary.
pub(crate) struct RenewableCredentialBundle {
	access_token:  SecretString,
	refresh_token: SecretString,
	token_type:    Str,
	expires_in:    Option<Duration>,
}

impl RenewableCredentialBundle {
	/// Encodes the bundle into an opaque zeroizing store payload.
	pub(crate) fn encode(&self) -> Result<SecretBox<Vec<u8>>, OAuthError> {
		let access = self.access_token.expose_secret().as_bytes();
		let refresh = self.refresh_token.expose_secret().as_bytes();
		let token_type = self.token_type.as_bytes();
		let mut encoded =
			Zeroizing::new(Vec::with_capacity(24 + access.len() + refresh.len() + token_type.len()));
		encoded.extend_from_slice(b"ORCB1");
		encode_field(&mut encoded, access)?;
		encode_field(&mut encoded, refresh)?;
		encode_field(&mut encoded, token_type)?;
		encoded.extend_from_slice(
			&self
				.expires_in
				.map_or(u64::MAX, |value| value.as_secs())
				.to_be_bytes(),
		);
		Ok(SecretBox::new(Box::new(std::mem::take(&mut *encoded))))
	}

	/// Decodes an authenticated store payload without exposing token text.
	pub(crate) fn decode(encoded: &SecretBox<Vec<u8>>) -> Result<Self, OAuthError> {
		let mut input = encoded.expose_secret().as_slice();
		if !input.starts_with(b"ORCB1") {
			return Err(OAuthError::MalformedRenewableCredential);
		}
		input = &input[5..];
		let access = Zeroizing::new(decode_field(&mut input)?);
		let refresh = Zeroizing::new(decode_field(&mut input)?);
		let token_type = Zeroizing::new(decode_field(&mut input)?);
		if input.len() != 8 {
			return Err(OAuthError::MalformedRenewableCredential);
		}
		let expires = u64::from_be_bytes(
			input
				.try_into()
				.map_err(|_| OAuthError::MalformedRenewableCredential)?,
		);
		let access = String::from_utf8(access.to_vec())
			.map_err(|_| OAuthError::MalformedRenewableCredential)?;
		let refresh = String::from_utf8(refresh.to_vec())
			.map_err(|_| OAuthError::MalformedRenewableCredential)?;
		let token_type = String::from_utf8(token_type.to_vec())
			.map_err(|_| OAuthError::MalformedRenewableCredential)?;
		if access.is_empty() || refresh.is_empty() || token_type.is_empty() {
			return Err(OAuthError::MalformedRenewableCredential);
		}
		Ok(Self {
			access_token:  SecretString::from(access),
			refresh_token: SecretString::from(refresh),
			token_type:    token_type.into(),
			expires_in:    (expires != u64::MAX).then(|| Duration::from_secs(expires)),
		})
	}

	pub(crate) fn into_refresh(self) -> SecretString {
		self.refresh_token
	}
}

impl std::fmt::Debug for RenewableCredentialBundle {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		formatter.write_str("RenewableCredentialBundle([REDACTED])")
	}
}

fn encode_field(output: &mut Vec<u8>, value: &[u8]) -> Result<(), OAuthError> {
	let length = u32::try_from(value.len()).map_err(|_| OAuthError::MalformedRenewableCredential)?;
	output.extend_from_slice(&length.to_be_bytes());
	output.extend_from_slice(value);
	Ok(())
}

fn decode_field(input: &mut &[u8]) -> Result<Vec<u8>, OAuthError> {
	let length_bytes: [u8; 4] = input
		.get(..4)
		.ok_or(OAuthError::MalformedRenewableCredential)?
		.try_into()
		.map_err(|_| OAuthError::MalformedRenewableCredential)?;
	let length = usize::try_from(u32::from_be_bytes(length_bytes))
		.map_err(|_| OAuthError::MalformedRenewableCredential)?;
	let value = input
		.get(4..4 + length)
		.ok_or(OAuthError::MalformedRenewableCredential)?;
	*input = &input[4 + length..];
	Ok(value.to_vec())
}

impl std::fmt::Debug for OAuthTokenSet {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		formatter.write_str("OAuthTokenSet([REDACTED])")
	}
}

/// Closed, sanitized OAuth provider error vocabulary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OAuthProviderCode {
	/// Device authorization remains pending.
	AuthorizationPending,
	/// Device polling must slow down.
	SlowDown,
	/// Authorization was declined by the resource owner.
	AccessDenied,
	/// Device or authorization grant expired.
	ExpiredToken,
	/// Refresh or authorization grant is invalid/revoked.
	InvalidGrant,
	/// Public client declaration is invalid.
	InvalidClient,
	/// Request shape is invalid.
	InvalidRequest,
	/// Requested scope is invalid.
	InvalidScope,
	/// Provider failed transiently.
	ServerError,
	/// Provider is temporarily unavailable.
	TemporarilyUnavailable,
	/// Provider returned an unknown code; raw text is deliberately discarded.
	Unknown,
}

impl OAuthProviderCode {
	/// Stable, secret-free code suitable for rejection evidence.
	#[must_use]
	pub const fn as_str(self) -> &'static str {
		match self {
			Self::AuthorizationPending => "authorization_pending",
			Self::SlowDown => "slow_down",
			Self::AccessDenied => "access_denied",
			Self::ExpiredToken => "expired_token",
			Self::InvalidGrant => "invalid_grant",
			Self::InvalidClient => "invalid_client",
			Self::InvalidRequest => "invalid_request",
			Self::InvalidScope => "invalid_scope",
			Self::ServerError => "server_error",
			Self::TemporarilyUnavailable => "temporarily_unavailable",
			Self::Unknown => "unknown",
		}
	}
}

/// OAuth transport failure stripped of request/response material.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[error("OAuth HTTP transport failed")]
pub struct OAuthTransportError;

/// OAuth engine failure with typed, secret-free evidence.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum OAuthError {
	/// Catalog endpoint is not a valid absolute HTTP(S) URL.
	#[error("OAuth endpoint URL is invalid")]
	InvalidUrl,
	/// Authorization callback state does not match the pending flow.
	#[error("OAuth authorization state did not match")]
	StateMismatch,
	/// Authorization callback omits a code or has an invalid shape.
	#[error("OAuth authorization callback is malformed")]
	MalformedCallback,
	/// Token or device response has an invalid typed shape.
	#[error("OAuth response is malformed")]
	MalformedResponse,
	/// HTTP transport failed without retaining source text.
	#[error(transparent)]
	Transport(#[from] OAuthTransportError),

	/// Provider returned sanitized protocol evidence.
	#[error("OAuth provider rejected the request")]
	Provider { status: u16, code: OAuthProviderCode, retryable: bool },
	/// Refresh grant rejection must reach credential-source rejection policy.
	#[error("OAuth refresh grant was rejected")]
	RefreshRejected(AuthRejection),
	/// Catalog explicitly declares that this flow cannot refresh.
	#[error("OAuth flow does not support refresh")]
	RefreshUnsupported,
	/// Caller supplied input for a different login step.
	#[error("OAuth login received unexpected input")]
	UnexpectedInput,
	/// Login was cancelled.
	#[error("OAuth login was cancelled")]
	Cancelled,
	/// Device polling reached its catalog or expiry bound.
	#[error("OAuth device polling exhausted its bound")]
	PollingExhausted { polls: u16 },
	/// Cryptographic random generation failed.
	#[error("OAuth cryptographic entropy is unavailable")]
	Entropy,
	/// A token expiry cannot be represented.
	#[error("OAuth token expiry is invalid")]
	InvalidExpiry,
	/// Renewable token was routed to the ephemeral lease path.
	#[error("renewable OAuth credential requires encrypted persistence")]
	RenewableCredentialRequiresPersistence,
	/// OAuth result did not include a refresh grant.
	#[error("OAuth response did not include a refresh token")]
	MissingRefreshToken,
	/// Authenticated renewable bundle had an invalid internal shape.
	#[error("stored renewable OAuth credential is malformed")]
	MalformedRenewableCredential,
	/// Catalog-selected identity evidence was absent or invalid.
	#[error("OAuth principal identity could not be resolved")]
	PrincipalUnresolved,
	/// Login event/input channel failed.
	#[error(transparent)]
	Login(LoginChannelError),
}

impl From<LoginChannelError> for OAuthError {
	fn from(error: LoginChannelError) -> Self {
		match error {
			LoginChannelError::Cancelled => Self::Cancelled,
			error => Self::Login(error),
		}
	}
}
/// Converts a stored opaque renewable bundle into a lease without exposing its
/// encoding or token material to the store-backed credential source.
pub(crate) fn lease_stored_bundle(
	stored: super::store::StoredOAuthCredential,
	valid_after: SystemTime,
) -> Result<CredentialLease, super::lease::CredentialError> {
	let bundle = RenewableCredentialBundle::decode(&stored.bundle)
		.map_err(|_| super::lease::CredentialError::SourceFailure)?;
	let expires_at = stored
		.metadata
		.expires_at_ms
		.map(system_time_from_millis)
		.transpose()
		.map_err(|_| super::lease::CredentialError::SourceFailure)?;
	if expires_at.is_some_and(|expires_at| expires_at <= valid_after) {
		return Err(super::lease::CredentialError::Expired);
	}
	let meta = LeaseMeta {
		account: stored.metadata.account_id,
		principal: stored.metadata.principal_id,
		generation: stored.metadata.generation,
		expires_at,
	};
	Ok(CredentialLease::bearer(meta, bundle.access_token))
}

/// OAuth persistence/refresh failure with secret-free evidence.
#[derive(Debug, thiserror::Error)]
pub enum OAuthCredentialManagerError {
	/// OAuth protocol or bundle processing failed.
	#[error(transparent)]
	OAuth(#[from] OAuthError),
	/// Encrypted credential store failed.
	#[error(transparent)]
	Store(#[from] StoreError),
	/// Shared refresh coordination failed.
	#[error(transparent)]
	Refresh(RefreshError),
	/// Persisted access token is expired and requires coordinated refresh.
	#[error("persisted OAuth access token is expired")]
	Expired,
	/// Wall-clock timestamp cannot be represented as Unix milliseconds.
	#[error("OAuth credential timestamp is invalid")]
	InvalidTime,
}

fn unix_millis(time: SystemTime) -> Result<u64, OAuthCredentialManagerError> {
	let millis = time
		.duration_since(SystemTime::UNIX_EPOCH)
		.map_err(|_| OAuthCredentialManagerError::InvalidTime)?
		.as_millis();
	u64::try_from(millis).map_err(|_| OAuthCredentialManagerError::InvalidTime)
}

fn system_time_from_millis(millis: u64) -> Result<SystemTime, OAuthCredentialManagerError> {
	SystemTime::UNIX_EPOCH
		.checked_add(Duration::from_millis(millis))
		.ok_or(OAuthCredentialManagerError::InvalidTime)
}

fn refresh_store_operation(_: StoreError) -> RefreshOperationError {
	RefreshOperationError {
		code:    "credential-store".into(),
		summary: "encrypted credential persistence failed".into(),
	}
}

fn refresh_oauth_operation(error: OAuthError) -> RefreshOperationError {
	let code = match error {
		OAuthError::RefreshRejected(_) => "refresh-rejected",
		OAuthError::Cancelled => "cancelled",
		OAuthError::Transport(_) => "transport",
		_ => "oauth-protocol",
	};
	RefreshOperationError { code: code.into(), summary: "OAuth credential refresh failed".into() }
}

fn refresh_manager_operation(_: OAuthCredentialManagerError) -> RefreshOperationError {
	RefreshOperationError {
		code:    "credential-time".into(),
		summary: "OAuth credential timestamp is invalid".into(),
	}
}

#[derive(Deserialize)]
struct DeviceAuthorizationResponse {
	device_code:               Option<String>,
	user_code:                 Option<String>,
	verification_uri:          Option<String>,
	verification_url:          Option<String>,
	verification_uri_complete: Option<String>,
	expires_in:                Option<u64>,
	interval:                  Option<u64>,
}

#[derive(Deserialize)]
struct TokenResponse {
	access_token:  Option<String>,
	refresh_token: Option<String>,
	token_type:    Option<String>,
	expires_in:    Option<u64>,
	error:         Option<String>,
}

fn token_response(
	response: OAuthHttpResponse,
	fallback_refresh: Option<SecretString>,
) -> Result<OAuthTokenSet, OAuthError> {
	let parsed: TokenResponse = decode(&response.body)?;
	if parsed.error.is_some() {
		return Err(provider_error(response.status, &response.body, fallback_refresh.is_some()));
	}
	let access_token = parsed
		.access_token
		.filter(|value| !value.is_empty())
		.ok_or(OAuthError::MalformedResponse)?;
	Ok(OAuthTokenSet {
		access_token:      SecretString::from(access_token),
		refresh_token:     parsed
			.refresh_token
			.map(SecretString::from)
			.or(fallback_refresh),
		token_type:        parsed
			.token_type
			.unwrap_or_else(|| "Bearer".to_owned())
			.into(),
		expires_in:        parsed.expires_in.map(Duration::from_secs),
		identity_response: response.body,
	})
}

fn json_string_at(document: &str, pointer: &str) -> Result<Str, OAuthError> {
	if !pointer.starts_with('/') {
		return Err(OAuthError::PrincipalUnresolved);
	}
	let value: serde_json::Value =
		serde_json::from_str(document).map_err(|_| OAuthError::PrincipalUnresolved)?;
	value
		.pointer(pointer)
		.and_then(serde_json::Value::as_str)
		.filter(|value| !value.is_empty())
		.map(Str::from)
		.ok_or(OAuthError::PrincipalUnresolved)
}

fn json_object_string(document: &str, field: &str) -> Result<Str, OAuthError> {
	if field.is_empty() {
		return Err(OAuthError::PrincipalUnresolved);
	}
	let value: serde_json::Value =
		serde_json::from_str(document).map_err(|_| OAuthError::PrincipalUnresolved)?;
	value
		.as_object()
		.and_then(|object| object.get(field))
		.and_then(serde_json::Value::as_str)
		.filter(|value| !value.is_empty())
		.map(Str::from)
		.ok_or(OAuthError::PrincipalUnresolved)
}

fn provider_error(status: u16, body: &SecretString, refresh: bool) -> OAuthError {
	let code = provider_code(body);
	let retryable = matches!(status, 408 | 425 | 429 | 500..=599)
		|| matches!(code, OAuthProviderCode::ServerError | OAuthProviderCode::TemporarilyUnavailable);
	if refresh
		&& matches!(
			code,
			OAuthProviderCode::InvalidGrant
				| OAuthProviderCode::InvalidClient
				| OAuthProviderCode::AccessDenied
		) {
		return OAuthError::RefreshRejected(AuthRejection {
			kind:        AuthRejectionKind::RefreshRejected,
			status:      Some(status),
			code:        Some(code.as_str().into()),
			refreshable: false,
		});
	}
	OAuthError::Provider { status, code, retryable }
}

fn provider_code(body: &SecretString) -> OAuthProviderCode {
	let Ok(parsed) = serde_json::from_str::<TokenResponse>(body.expose_secret()) else {
		return OAuthProviderCode::Unknown;
	};
	match parsed.error.as_deref() {
		Some("authorization_pending") => OAuthProviderCode::AuthorizationPending,
		Some("slow_down") => OAuthProviderCode::SlowDown,
		Some("access_denied" | "authorization_declined") => OAuthProviderCode::AccessDenied,
		Some("expired_token") => OAuthProviderCode::ExpiredToken,
		Some("invalid_grant" | "bad_verification_code") => OAuthProviderCode::InvalidGrant,
		Some("invalid_client") => OAuthProviderCode::InvalidClient,
		Some("invalid_request") => OAuthProviderCode::InvalidRequest,
		Some("invalid_scope") => OAuthProviderCode::InvalidScope,
		Some("server_error") => OAuthProviderCode::ServerError,
		Some("temporarily_unavailable") => OAuthProviderCode::TemporarilyUnavailable,
		_ => OAuthProviderCode::Unknown,
	}
}

fn callback_code(
	callback: &SecretString,
	expected_state: &str,
) -> Result<SecretString, OAuthError> {
	let callback = callback.expose_secret();
	if !(callback.starts_with("http://") || callback.starts_with("https://")) {
		return Err(OAuthError::MalformedCallback);
	}
	let query = callback
		.split_once('?')
		.map(|(_, query)| query.split('#').next().unwrap_or_default())
		.ok_or(OAuthError::MalformedCallback)?;
	let mut state_seen = false;
	let mut code = None;
	for field in query.split('&').filter(|field| !field.is_empty()) {
		let (name, value) = field.split_once('=').unwrap_or((field, ""));
		let name = decode_form_component(name)?;
		if name.as_str() == "state" {
			if state_seen {
				return Err(OAuthError::MalformedCallback);
			}
			state_seen = true;
			let state = decode_form_component(value)?;
			if state.as_str() != expected_state {
				return Err(OAuthError::StateMismatch);
			}
		} else if name.as_str() == "code" {
			if code.is_some() {
				return Err(OAuthError::MalformedCallback);
			}
			let mut decoded = decode_form_component(value)?;
			if decoded.is_empty() {
				return Err(OAuthError::MalformedCallback);
			}
			code = Some(SecretString::from(std::mem::take(&mut *decoded)));
		}
	}
	if !state_seen {
		return Err(OAuthError::MalformedCallback);
	}
	code.ok_or(OAuthError::MalformedCallback)
}

fn decode_form_component(value: &str) -> Result<Zeroizing<String>, OAuthError> {
	let bytes = value.as_bytes();
	let mut decoded = Zeroizing::new(Vec::with_capacity(bytes.len()));
	let mut index = 0;
	while index < bytes.len() {
		match bytes[index] {
			b'+' => {
				decoded.push(b' ');
				index += 1;
			},
			b'%' if index + 2 < bytes.len() => {
				let high = hex_nibble(bytes[index + 1]).ok_or(OAuthError::MalformedCallback)?;
				let low = hex_nibble(bytes[index + 2]).ok_or(OAuthError::MalformedCallback)?;
				decoded.push((high << 4) | low);
				index += 3;
			},
			b'%' => return Err(OAuthError::MalformedCallback),
			byte => {
				decoded.push(byte);
				index += 1;
			},
		}
	}
	let decoded = String::from_utf8(std::mem::take(&mut *decoded))
		.map_err(|_| OAuthError::MalformedCallback)?;
	Ok(Zeroizing::new(decoded))
}

const fn hex_nibble(byte: u8) -> Option<u8> {
	match byte {
		b'0'..=b'9' => Some(byte - b'0'),
		b'a'..=b'f' => Some(byte - b'a' + 10),
		b'A'..=b'F' => Some(byte - b'A' + 10),
		_ => None,
	}
}

fn form_request(
	url: &str,
	fields: &[(&str, FormValue<'_>)],
	extra: &[OAuthParameter],
) -> Result<OAuthHttpRequest, OAuthError> {
	let url = parse_http_url(url)?;
	let mut serializer = url::form_urlencoded::Serializer::new(String::new());
	for (name, value) in fields {
		serializer.append_pair(name, value.expose());
	}
	for parameter in extra {
		serializer.append_pair(&parameter.name, &parameter.value);
	}
	let body = SecretString::from(serializer.finish());
	let mut headers = HeaderMap::new();
	headers.insert(CONTENT_TYPE, HeaderValue::from_static(FORM_CONTENT_TYPE));
	headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
	Ok(OAuthHttpRequest { method: Method::POST, url, headers, body: Some(body) })
}

enum FormValue<'a> {
	Public(&'a str),
	Secret(&'a str),
}

impl FormValue<'_> {
	const fn expose(&self) -> &str {
		match self {
			Self::Public(value) | Self::Secret(value) => value,
		}
	}
}

fn parse_http_url(value: &str) -> Result<Url, OAuthError> {
	let parsed = Url::parse(value).map_err(|_| OAuthError::InvalidUrl)?;
	if matches!(parsed.scheme(), "http" | "https") && parsed.has_host() {
		Ok(parsed)
	} else {
		Err(OAuthError::InvalidUrl)
	}
}

fn decode<T: for<'de> Deserialize<'de>>(body: &SecretString) -> Result<T, OAuthError> {
	serde_json::from_str(body.expose_secret()).map_err(|_| OAuthError::MalformedResponse)
}

#[cfg(test)]
mod tests {
	use std::{collections::VecDeque, sync::Arc};

	use futures::FutureExt;
	use parking_lot::Mutex;
	use tempfile::tempdir;

	use super::*;
	use crate::{
		account::{CredentialFreshness, RefreshCoordinator, RefreshPolicy, RefreshRequest},
		auth::{
			CredentialOrigin, CredentialSourceSpec, CredentialStore, HeadlessKeySource, KeyId,
			OAuthRefreshSpec, login::default_login_channels, spec::HeaderPlacement,
		},
		id::{LoginSessionId, PrincipalId},
	};

	struct FixedEntropy;
	impl OAuthEntropy for FixedEntropy {
		fn fill(&self, destination: &mut [u8]) -> Result<(), OAuthError> {
			for (index, byte) in destination.iter_mut().enumerate() {
				*byte = index as u8;
			}
			Ok(())
		}
	}

	struct TestClock(SystemTime);
	impl OAuthClock for TestClock {
		fn now(&self) -> SystemTime {
			self.0
		}

		fn sleep(&self, _: Duration) -> BoxFuture<'_, ()> {
			async {}.boxed()
		}
	}

	struct TestHttp(Mutex<VecDeque<OAuthHttpResponse>>);
	impl OAuthHttpClient for TestHttp {
		fn execute(
			&self,
			_: OAuthHttpRequest,
		) -> BoxFuture<'_, Result<OAuthHttpResponse, OAuthTransportError>> {
			async move { Ok(self.0.lock().pop_front().expect("fixture response")) }.boxed()
		}
	}

	fn client() -> OAuthClientSpec {
		OAuthClientSpec {
			sources:      vec![CredentialSourceSpec::Interactive],
			client_id:    "client".into(),
			refresh:      OAuthRefreshSpec::TokenEndpoint,
			token_url:    "https://auth.example/token".into(),
			scopes:       vec!["openid".into(), "profile".into()],
			audience:     None,
			token_params: Vec::new(),
			placement:    HeaderPlacement::bearer().into(),
		}
	}

	#[tokio::test]
	async fn pkce_timeline_validates_callback_state_and_redacts_pending_state() {
		let http = TestHttp(Mutex::new(VecDeque::from([OAuthHttpResponse {
			status:  200,
			headers: HeaderMap::new(),
			body:    SecretString::from(
				r#"{"access_token":"access","refresh_token":"refresh","expires_in":3600}"#.to_owned(),
			),
		}])));
		let clock = TestClock(SystemTime::UNIX_EPOCH);
		let engine = OAuthEngine::with_entropy(&http, &clock, FixedEntropy);
		let spec = OAuthPkceSpec {
			client:           client(),
			authorize_url:    "https://auth.example/authorize".into(),
			redirect_uri:     "http://127.0.0.1:1455/callback".into(),
			completion:       PkceCompletion::PasteCallbackUrl,
			authorize_params: Vec::new(),
		};
		let (session, driver, _) = default_login_channels(LoginSessionId::from("login"));
		let pending = engine.begin_pkce(&spec, &driver).await.expect("begin");
		let first = session
			.events
			.recv_async()
			.await
			.expect("event")
			.expect("ok");
		let AuthEvent::OpenUrl(url) = first else {
			panic!("open URL")
		};
		let state = Url::parse(&url)
			.expect("url")
			.query_pairs()
			.find(|(name, _)| name == "state")
			.expect("state")
			.1
			.into_owned();
		assert!(!format!("{pending:?}").contains(&state));
		let callback = format!("http://127.0.0.1:1455/callback?code=code&state={state}");
		let tokens = engine
			.complete_pkce(&spec, pending, AuthInput::CallbackUrl(SecretString::from(callback)))
			.await
			.expect("tokens");
		assert!(tokens.is_refreshable());
		assert!(!format!("{tokens:?}").contains("access"));
	}

	#[tokio::test]
	async fn refresh_rejection_returns_typed_evidence_without_provider_text() {
		let http = TestHttp(Mutex::new(VecDeque::from([OAuthHttpResponse {
			status:  400,
			headers: HeaderMap::new(),
			body:    SecretString::from(
				r#"{"error":"invalid_grant","error_description":"leaked-secret"}"#.to_owned(),
			),
		}])));
		let clock = TestClock(SystemTime::UNIX_EPOCH);
		let engine = OAuthEngine::with_entropy(&http, &clock, FixedEntropy);
		let error = engine
			.refresh(&client(), SecretString::from("refresh-secret".to_owned()))
			.await
			.expect_err("rejected");
		let OAuthError::RefreshRejected(evidence) = error else {
			panic!("evidence")
		};
		assert_eq!(evidence.code.as_deref(), Some("invalid_grant"));
		assert!(!format!("{evidence:?}").contains("leaked-secret"));
	}
	#[tokio::test]
	async fn custom_oauth_dispatch_is_typed_and_fails_closed_when_unregistered() {
		let spec = OAuthCustomSpec {
			client:        client(),
			authorize_url: "https://auth.example/custom".into(),
			exchange:      omp_llm_catalog::provider::OAuthExchangeKind::ApiKeyPaste,
			parameters:    Vec::new(),
			polling:       None,
		};
		let (_, driver, _) = default_login_channels(LoginSessionId::from("custom"));
		let error = OAuthCustomDispatcher::new()
			.exchange(&spec, &driver)
			.await
			.expect_err("missing handler");
		assert!(matches!(
			error,
			OAuthCustomDispatchError::Unavailable(
				omp_llm_catalog::provider::OAuthExchangeKind::ApiKeyPaste
			)
		));
	}

	#[test]
	fn renewable_bundle_round_trips_opaque_bytes_and_redacts_debug() {
		let access = "access-secret-marker";
		let refresh = "refresh-secret-marker";
		let tokens = OAuthTokenSet {
			access_token:      SecretString::from(access.to_owned()),
			refresh_token:     Some(SecretString::from(refresh.to_owned())),
			token_type:        "Bearer".into(),
			expires_in:        Some(Duration::from_secs(3600)),
			identity_response: SecretString::from("{}".to_owned()),
		};
		let bundle = tokens.into_renewable_bundle().expect("renewable");
		let encoded = bundle.encode().expect("encode");
		assert!(!format!("{bundle:?} {encoded:?}").contains(access));
		assert!(!format!("{bundle:?} {encoded:?}").contains(refresh));
		let decoded = RenewableCredentialBundle::decode(&encoded).expect("decode");
		let debug = format!("{decoded:?}");
		assert!(!debug.contains(access));
		assert!(!debug.contains(refresh));
	}

	#[test]
	fn renewable_token_cannot_enter_ephemeral_lease_path() {
		let tokens = OAuthTokenSet {
			access_token:      SecretString::from("access".to_owned()),
			refresh_token:     Some(SecretString::from("refresh".to_owned())),
			token_type:        "Bearer".into(),
			expires_in:        None,
			identity_response: SecretString::from("{}".to_owned()),
		};
		let meta = LeaseMeta {
			account:    AccountId::from("account"),
			principal:  PrincipalId::from("principal"),
			generation: 0,
			expires_at: None,
		};
		assert!(matches!(
			tokens.into_ephemeral_lease(meta, SystemTime::UNIX_EPOCH),
			Err(OAuthError::RenewableCredentialRequiresPersistence)
		));
	}

	#[tokio::test]
	async fn persisted_login_refreshes_once_and_increments_generation() {
		let directory = tempdir().expect("temporary directory");
		let keys = Arc::new(HeadlessKeySource::new(KeyId::new("oauth-test-key"), [0x5a; 32]));
		let store = Arc::new(
			CredentialStore::open(directory.path().join("credentials.sqlite"), keys)
				.expect("credential store"),
		);
		let issued_at = SystemTime::UNIX_EPOCH + Duration::from_secs(100);
		let meta = LeaseMeta {
			account:    AccountId::from("renewable-account"),
			principal:  PrincipalId::from("renewable-principal"),
			generation: 0,
			expires_at: None,
		};
		let initial = OAuthTokenSet {
			access_token:      SecretString::from("old-access-marker".to_owned()),
			refresh_token:     Some(SecretString::from("refresh-marker".to_owned())),
			token_type:        "Bearer".into(),
			expires_in:        Some(Duration::from_secs(1)),
			identity_response: SecretString::from("{}".to_owned()),
		};
		let http = TestHttp(Mutex::new(VecDeque::from([OAuthHttpResponse {
			status: 200,
			headers: HeaderMap::new(),
			body: SecretString::from(
				r#"{"access_token":"new-access-marker","refresh_token":"new-refresh-marker","expires_in":3600}"#
					.to_owned(),
			),
		}])));
		let clock = TestClock(issued_at);
		let engine = OAuthEngine::with_entropy(&http, &clock, FixedEntropy);
		let freshness = engine
			.persist_login(&store, initial, &meta, CredentialOrigin::Persistent, issued_at)
			.expect("persist login");
		assert_eq!(freshness.generation, 1);
		assert!(matches!(
			engine.lease_persisted(&store, &meta.account, issued_at + Duration::from_secs(2)),
			Err(OAuthCredentialManagerError::Expired)
		));
		let coordinator = RefreshCoordinator::new("oauth-test-owner", RefreshPolicy::default())
			.expect("coordinator");
		let requested_at = issued_at + Duration::from_secs(2);
		let outcome = engine
			.refresh_persisted(
				&coordinator,
				store.clone(),
				client(),
				RefreshRequest {
					account: meta.account.clone(),
					principal: meta.principal.clone(),
					rejected: CredentialFreshness {
						generation:  1,
						issued_at:   Some(issued_at),
						expires_at:  freshness.expires_at,
						observed_at: requested_at,
					},
					requested_at,
				},
				CredentialOrigin::Persistent,
			)
			.await
			.expect("refresh");
		assert_eq!(outcome.result.freshness.generation, 2);
		let lease = engine
			.lease_persisted(&store, &meta.account, requested_at)
			.expect("renewed lease");
		let debug = format!("{lease:?} {outcome:?} {store:?}");
		for marker in ["refresh-marker", "new-refresh-marker", "new-access-marker"] {
			assert!(!debug.contains(marker));
		}
	}

	#[tokio::test]
	async fn principal_resolution_uses_only_catalog_selected_evidence() {
		let claim = "https://api.example/account";
		let claims = format!(r#"{{"{claim}":"claim-principal"}}"#);
		let payload = base64_url::encode_raw(claims.as_bytes()).into_string();
		let identity = format!(
			r#"{{"profile":{{"id":"response-principal"}},"id_token":"e30.{payload}.signature"}}"#,
		);
		let tokens = OAuthTokenSet {
			access_token:      SecretString::from("access-secret".to_owned()),
			refresh_token:     Some(SecretString::from("refresh-secret".to_owned())),
			token_type:        "Bearer".into(),
			expires_in:        Some(Duration::from_secs(3600)),
			identity_response: SecretString::from(identity),
		};
		let http = TestHttp(Mutex::new(VecDeque::from([OAuthHttpResponse {
			status:  200,
			headers: HeaderMap::new(),
			body:    SecretString::from(r#"{"subject":"userinfo-principal"}"#.to_owned()),
		}])));
		assert_eq!(
			tokens
				.resolve_principal(
					&PrincipalResolution::TokenResponseField { pointer: "/profile/id".into() },
					&http,
				)
				.await
				.expect("token response principal")
				.as_str(),
			"response-principal",
		);
		assert_eq!(
			tokens
				.resolve_principal(&PrincipalResolution::IdTokenClaim { claim: claim.into() }, &http,)
				.await
				.expect("ID token principal")
				.as_str(),
			"claim-principal",
		);
		assert_eq!(
			tokens
				.resolve_principal(
					&PrincipalResolution::StaticLabel { label: "configured-principal".into() },
					&http,
				)
				.await
				.expect("static principal")
				.as_str(),
			"configured-principal",
		);
		assert_eq!(
			tokens
				.resolve_principal(
					&PrincipalResolution::UserinfoEndpoint {
						url:   "https://auth.example/userinfo".into(),
						field: "subject".into(),
					},
					&http,
				)
				.await
				.expect("userinfo principal")
				.as_str(),
			"userinfo-principal",
		);
		assert!(!format!("{tokens:?}").contains("access-secret"));
	}

	#[tokio::test]
	async fn cancelled_device_poll_never_sends_a_request() {
		let http = TestHttp(Mutex::new(VecDeque::new()));
		let clock = TestClock(SystemTime::UNIX_EPOCH);
		let engine = OAuthEngine::with_entropy(&http, &clock, FixedEntropy);
		let (_, driver, cancellation) = default_login_channels(LoginSessionId::from("device"));
		cancellation.cancel();
		let spec = OAuthDeviceSpec {
			client:                   client(),
			device_authorization_url: "https://auth.example/device".into(),
			max_polls:                2,
			default_interval:         Duration::from_secs(1),
			max_interval:             Duration::from_secs(5),
		};
		let pending = DevicePending {
			device_code: SecretString::from("device-secret".to_owned()),
			interval:    Duration::from_secs(1),
			expires_at:  SystemTime::UNIX_EPOCH + Duration::from_secs(30),
			polls:       0,
		};
		assert!(matches!(
			engine.poll_device(&spec, pending, &driver).await,
			Err(OAuthError::Cancelled)
		));
	}
}
