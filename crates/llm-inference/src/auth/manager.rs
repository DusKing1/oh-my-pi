//! Direct authentication-operation manager shared by every registry route.

use std::{
	collections::{BTreeMap, BTreeSet},
	fmt,
	sync::{
		Arc,
		atomic::{AtomicU64, Ordering},
	},
	time::{SystemTime, UNIX_EPOCH},
};

use flume::Sender;
use futures::future::{BoxFuture, FutureExt as _};
use omp_core::Str;
use omp_llm_catalog::{
	AuthSpecId, Catalog,
	provider::{AuthSpecKind, OAuthFlowSpec},
};
use parking_lot::Mutex;
use secrecy::{ExposeSecret as _, SecretBox};

use super::{
	AuthSpec, CredentialBroker, CredentialError, CredentialNeed, CredentialOrigin, CredentialSource,
	CredentialStore, CredentialWrite, KeyError, LoginChannelError, OAuthClientSpec, OAuthClock,
	OAuthCredentialManagerError, OAuthCustomDispatchError, OAuthCustomDispatcher, OAuthCustomSpec,
	OAuthEngine, OAuthError, OAuthHttpClient, StoreError, default_login_channels,
};
use crate::{
	account::{AccountPool, AccountRecord, CredentialFreshness, RefreshCoordinator, RefreshRequest},
	answer::{
		AccountState, AccountSummary, AuthAnswer, AuthEvent, AuthPrompt, AuthPromptKind,
		AuthResponse, AuthSession,
	},
	call::{AccountRoutingContext, AuthInput, AuthMethod, AuthRequest, LoginRequest},
	error::{Error, ErrorDetail, ErrorKind, ErrorPhase, RetryAction},
	id::{AccountId, LoginSessionId, PrincipalId},
	receipt::ExecutionReceipt,
};
/// One constructed engine for a typed public login method.
pub trait AuthLoginEngine: Send + Sync {
	/// Public login method implemented by this engine.
	fn method(&self) -> AuthMethod;
	/// Returns whether this engine supports the provider.
	///
	/// Provider-scoped engines must be registered before generic engines for
	/// the same method because dispatch selects the first supporting engine.
	fn supports(&self, provider: &omp_llm_catalog::ProviderId) -> bool;

	/// Begins the exact catalog-selected authentication specification.
	fn begin(
		&self,
		request: LoginRequest,
		spec: AuthSpecId,
	) -> BoxFuture<'_, Result<AuthSession, Error>>;
}

/// Constructed credential refresher used by direct authentication operations.
pub trait AuthRefreshEngine: Send + Sync {
	/// Refreshes one exact account and returns its secret-free state.
	fn refresh(&self, account: AccountId) -> BoxFuture<'_, Result<AccountSummary, Error>>;
}

/// Construction failure for a static secret login engine.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("secret login engine supports only API-key or session-token methods")]
pub struct SecretLoginEngineError;

/// Concrete bounded login engine for caller-labeled API keys and session
/// tokens.
#[derive(Clone)]
pub struct SecretLoginEngine {
	method:          AuthMethod,
	principal_label: Str,
	catalog:         Arc<Catalog>,
	store:           Arc<CredentialStore>,
	accounts:        AccountPool,
}

impl SecretLoginEngine {
	/// Constructs a persistent secret login engine with an explicit non-secret
	/// principal label.
	pub fn new(
		method: AuthMethod,
		principal_label: Str,
		catalog: Arc<Catalog>,
		store: Arc<CredentialStore>,
		accounts: AccountPool,
	) -> Result<Self, SecretLoginEngineError> {
		if !matches!(method, AuthMethod::ApiKey | AuthMethod::SessionToken)
			|| principal_label.is_empty()
		{
			return Err(SecretLoginEngineError);
		}
		Ok(Self { method, principal_label, catalog, store, accounts })
	}
}

impl AuthLoginEngine for SecretLoginEngine {
	fn method(&self) -> AuthMethod {
		self.method
	}

	fn supports(&self, _provider: &omp_llm_catalog::ProviderId) -> bool {
		true
	}

	fn begin(
		&self,
		request: LoginRequest,
		spec: AuthSpecId,
	) -> BoxFuture<'_, Result<AuthSession, Error>> {
		let catalog = Arc::clone(&self.catalog);
		let store = Arc::clone(&self.store);
		let accounts = self.accounts.clone();
		let principal_label = self.principal_label.clone();
		let method = self.method;
		async move {
			let auth = catalog.auth_spec(&spec).ok_or_else(auth_not_found)?;
			let credential_kind = match (method, auth.kind) {
				(AuthMethod::ApiKey, AuthSpecKind::ApiKey) => "api-key",
				(AuthMethod::ApiKey, AuthSpecKind::Bearer) => "bearer",
				(AuthMethod::SessionToken, AuthSpecKind::OmpSession) => "session-token",
				_ => return Err(auth_unavailable()),
			};
			let provider = catalog
				.provider(&request.provider)
				.ok_or_else(auth_not_found)?;
			let session_id = next_login_session_id();
			let (session, driver, _) = default_login_channels(session_id);
			let provider_id = request.provider;
			let routes = provider.routes.iter().cloned().collect();
			tokio::spawn(async move {
				let result = async {
					let prompt = AuthPrompt {
						id:      match method {
							AuthMethod::ApiKey => "api-key".into(),
							_ => "session-token".into(),
						},
						message: match method {
							AuthMethod::ApiKey => "Enter the API key".into(),
							_ => "Enter the session token".into(),
						},
						input:   match method {
							AuthMethod::ApiKey => AuthPromptKind::ApiKey,
							_ => AuthPromptKind::SessionToken,
						},
					};
					driver
						.emit(AuthEvent::Prompt(prompt))
						.await
						.map_err(login_channel_error)?;
					let input = driver.receive().await.map_err(login_channel_error)?;
					let ((AuthMethod::ApiKey, AuthInput::ApiKey(secret))
					| (AuthMethod::SessionToken, AuthInput::SessionToken(secret))) = (method, input)
					else {
						return Err(auth_invalid_request());
					};
					let principal = PrincipalId::from(principal_label.clone());
					let account = AccountId::from(format!("{provider_id}:{principal_label}"));
					let bytes = SecretBox::new(Box::new(secret.expose_secret().as_bytes().to_vec()));
					let metadata = store
						.put(CredentialWrite {
							account_id:          &account,
							principal_id:        &principal,
							kind:                credential_kind,
							secret:              &bytes,
							expires_at_ms:       None,
							origin:              CredentialOrigin::Persistent,
							now_ms:              unix_millis(SystemTime::now())?,
							expected_generation: None,
						})
						.map_err(auth_store_error)?;
					accounts
						.upsert(AccountRecord {
							account: account.clone(),
							principal: principal.clone(),
							provider: provider_id.clone(),
							routes,
							enabled: true,
							credential_generation: metadata.generation,
							routing: AccountRoutingContext::default(),
						})
						.map_err(|_| auth_store_failure())?;
					let summary = AccountSummary {
						account,
						provider: provider_id,
						principal: Some(principal),
						label: Some(principal_label),
						state: AccountState::Active,
					};
					driver
						.emit(AuthEvent::Complete(summary))
						.await
						.map_err(login_channel_error)
				}
				.await;
				if let Err(error) = result {
					let _ = driver.emit_error(error).await;
				}
			});
			Ok(session)
		}
		.boxed()
	}
}

/// Construction failure for a non-interactive credential acquisition engine.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("credential acquisition engine supports only ADC or AWS-chain methods")]
pub struct CredentialAcquisitionLoginEngineError;

/// Concrete login adapter for application-default and AWS credential chains.
#[derive(Clone)]
pub struct CredentialAcquisitionLoginEngine {
	method:          AuthMethod,
	principal_label: Str,
	catalog:         Arc<Catalog>,
	broker:          CredentialBroker,
	accounts:        AccountPool,
}

impl CredentialAcquisitionLoginEngine {
	/// Constructs one catalog-driven non-interactive acquisition adapter.
	pub fn new(
		method: AuthMethod,
		principal_label: Str,
		catalog: Arc<Catalog>,
		broker: CredentialBroker,
		accounts: AccountPool,
	) -> Result<Self, CredentialAcquisitionLoginEngineError> {
		if !matches!(method, AuthMethod::ApplicationDefault | AuthMethod::AwsCredentialChain)
			|| principal_label.is_empty()
		{
			return Err(CredentialAcquisitionLoginEngineError);
		}
		Ok(Self { method, principal_label, catalog, broker, accounts })
	}
}

impl AuthLoginEngine for CredentialAcquisitionLoginEngine {
	fn method(&self) -> AuthMethod {
		self.method
	}

	fn supports(&self, _provider: &omp_llm_catalog::ProviderId) -> bool {
		true
	}

	fn begin(
		&self,
		request: LoginRequest,
		spec: AuthSpecId,
	) -> BoxFuture<'_, Result<AuthSession, Error>> {
		let catalog = Arc::clone(&self.catalog);
		let broker = self.broker.clone();
		let accounts = self.accounts.clone();
		let label = self.principal_label.clone();
		let method = self.method;
		async move {
			let auth = catalog.auth_spec(&spec).ok_or_else(auth_not_found)?;
			let expected = match method {
				AuthMethod::ApplicationDefault => AuthSpecKind::GcpAdc,
				AuthMethod::AwsCredentialChain => AuthSpecKind::AwsSigv4,
				_ => return Err(auth_unavailable()),
			};
			if auth.kind != expected {
				return Err(auth_unavailable());
			}
			let provider = catalog
				.provider(&request.provider)
				.ok_or_else(auth_not_found)?;
			let provider_id = request.provider;
			let routes = provider.routes.iter().cloned().collect();
			let principal = PrincipalId::from(label.clone());
			let account = AccountId::from(format!("{provider_id}:{label}"));
			let (session, driver, _) = default_login_channels(next_login_session_id());
			tokio::spawn(async move {
				let result = async {
					let lease = broker
						.lease(CredentialNeed {
							spec,
							account: Some(account.clone()),
							principal: Some(principal.clone()),
							valid_after: SystemTime::now(),
						})
						.await
						.map_err(credential_error)?;
					accounts
						.upsert(AccountRecord {
							account: account.clone(),
							principal: principal.clone(),
							provider: provider_id.clone(),
							routes,
							enabled: true,
							credential_generation: lease.meta().generation,
							routing: AccountRoutingContext::default(),
						})
						.map_err(|_| auth_store_failure())?;
					driver
						.emit(AuthEvent::Complete(AccountSummary {
							account,
							provider: provider_id,
							principal: Some(principal),
							label: Some(label),
							state: AccountState::Active,
						}))
						.await
						.map_err(login_channel_error)
				}
				.await;
				if let Err(error) = result {
					let _ = driver.emit_error(error).await;
				}
			});
			Ok(session)
		}
		.boxed()
	}
}

/// Construction failure for a concrete OAuth login adapter.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("OAuth login engine supports only PKCE/paste or device methods")]
pub struct OAuthLoginEngineError;

/// Owned OAuth login adapter over the catalog protocol engine.
pub struct OAuthLoginEngine<C, K> {
	method:   AuthMethod,
	catalog:  Arc<Catalog>,
	store:    Arc<CredentialStore>,
	accounts: AccountPool,
	http:     Arc<C>,
	clock:    Arc<K>,
	custom:   Arc<OAuthCustomDispatcher>,
}

impl<C, K> OAuthLoginEngine<C, K> {
	/// Constructs one method-specific owned OAuth login adapter.
	pub fn new(
		method: AuthMethod,
		catalog: Arc<Catalog>,
		store: Arc<CredentialStore>,
		accounts: AccountPool,
		http: Arc<C>,
		clock: Arc<K>,
		custom: Arc<OAuthCustomDispatcher>,
	) -> Result<Self, OAuthLoginEngineError> {
		if !matches!(method, AuthMethod::OAuthPkce | AuthMethod::OAuthDevice) {
			return Err(OAuthLoginEngineError);
		}
		Ok(Self { method, catalog, store, accounts, http, clock, custom })
	}
}

impl<C, K> AuthLoginEngine for OAuthLoginEngine<C, K>
where
	C: OAuthHttpClient + 'static,
	K: OAuthClock + 'static,
{
	fn method(&self) -> AuthMethod {
		self.method
	}

	fn supports(&self, _provider: &omp_llm_catalog::ProviderId) -> bool {
		true
	}

	fn begin(
		&self,
		request: LoginRequest,
		spec_id: AuthSpecId,
	) -> BoxFuture<'_, Result<AuthSession, Error>> {
		let catalog = Arc::clone(&self.catalog);
		let store = Arc::clone(&self.store);
		let accounts = self.accounts.clone();
		let http = Arc::clone(&self.http);
		let clock = Arc::clone(&self.clock);
		let custom = Arc::clone(&self.custom);
		let method = self.method;
		async move {
			let auth = catalog.auth_spec(&spec_id).ok_or_else(auth_not_found)?;
			let oauth_id = auth.oauth.as_ref().ok_or_else(auth_unavailable)?;
			let oauth = catalog.oauth_spec(oauth_id).ok_or_else(auth_unavailable)?;
			let resolution = oauth
				.principal_resolution
				.clone()
				.ok_or_else(principal_unresolved)?;
			let runtime =
				AuthSpec::from_catalog(auth, Some(oauth), None).map_err(|_| auth_unavailable())?;
			let provider = catalog
				.provider(&request.provider)
				.ok_or_else(auth_not_found)?;
			let routes = provider.routes.iter().cloned().collect();
			let provider_id = request.provider;
			let session_id = next_login_session_id();
			let (session, driver, _) = default_login_channels(session_id);
			tokio::spawn(async move {
				let result = async {
					let engine = OAuthEngine::new(http.as_ref(), clock.as_ref());
					let tokens = match runtime {
						AuthSpec::OAuthPkce(spec) if method == AuthMethod::OAuthPkce => {
							let pending = engine
								.begin_pkce(&spec, &driver)
								.await
								.map_err(oauth_error)?;
							let input = driver.receive().await.map_err(login_channel_error)?;
							engine
								.complete_pkce(&spec, pending, input)
								.await
								.map_err(oauth_error)?
						},
						AuthSpec::OAuthPaste(spec) if method == AuthMethod::OAuthPkce => {
							engine
								.begin_paste(&spec, &driver)
								.await
								.map_err(oauth_error)?;
							let input = driver.receive().await.map_err(login_channel_error)?;
							engine
								.complete_paste(&spec, input)
								.await
								.map_err(oauth_error)?
						},
						AuthSpec::OAuthDevice(spec) if method == AuthMethod::OAuthDevice => {
							let pending = engine
								.begin_device(&spec, &driver)
								.await
								.map_err(oauth_error)?;
							engine
								.poll_device(&spec, pending, &driver)
								.await
								.map_err(oauth_error)?
						},
						AuthSpec::OAuthCustom(spec) => custom
							.exchange(&spec, &driver)
							.await
							.map_err(oauth_custom_error)?,
						_ => return Err(auth_unavailable()),
					};
					let principal = tokens
						.resolve_principal(&resolution, http.as_ref())
						.await
						.map_err(oauth_error)?;
					let account = AccountId::from(format!("{provider_id}:{principal}"));
					let issued_at = clock.now();
					let meta = super::LeaseMeta {
						account:    account.clone(),
						principal:  principal.clone(),
						generation: 0,
						expires_at: None,
					};
					let freshness = engine
						.persist_login(&store, tokens, &meta, CredentialOrigin::Persistent, issued_at)
						.map_err(oauth_manager_error)?;
					accounts
						.upsert(AccountRecord {
							account: account.clone(),
							principal: principal.clone(),
							provider: provider_id.clone(),
							routes,
							enabled: true,
							credential_generation: freshness.generation,
							routing: AccountRoutingContext::default(),
						})
						.map_err(|_| auth_store_failure())?;
					let summary = AccountSummary {
						account,
						provider: provider_id,
						principal: Some(principal.clone()),
						label: Some(Str::from(principal.as_str())),
						state: AccountState::Active,
					};
					driver
						.emit(AuthEvent::Complete(summary))
						.await
						.map_err(login_channel_error)
				}
				.await;
				if let Err(error) = result {
					let _ = driver.emit_error(error).await;
				}
			});
			Ok(session)
		}
		.boxed()
	}
}

enum OAuthRefreshRuntime {
	Standard(OAuthClientSpec),
	Custom(OAuthCustomSpec),
}

/// Owned refresh adapter for encrypted OAuth credentials.
pub struct StoredOAuthRefreshEngine<C, K> {
	catalog:     Arc<Catalog>,
	store:       Arc<CredentialStore>,
	accounts:    AccountPool,
	http:        Arc<C>,
	clock:       Arc<K>,
	custom:      Arc<OAuthCustomDispatcher>,
	coordinator: Arc<RefreshCoordinator>,
}

impl<C, K> StoredOAuthRefreshEngine<C, K> {
	/// Constructs an OAuth refresh adapter over one shared coordinator.
	#[must_use]
	pub const fn new(
		catalog: Arc<Catalog>,
		store: Arc<CredentialStore>,
		accounts: AccountPool,
		http: Arc<C>,
		clock: Arc<K>,
		custom: Arc<OAuthCustomDispatcher>,
		coordinator: Arc<RefreshCoordinator>,
	) -> Self {
		Self { catalog, store, accounts, http, clock, custom, coordinator }
	}
}

impl<C, K> AuthRefreshEngine for StoredOAuthRefreshEngine<C, K>
where
	C: OAuthHttpClient + 'static,
	K: OAuthClock + 'static,
{
	fn refresh(&self, account: AccountId) -> BoxFuture<'_, Result<AccountSummary, Error>> {
		let catalog = Arc::clone(&self.catalog);
		let store = Arc::clone(&self.store);
		let accounts = self.accounts.clone();
		let http = Arc::clone(&self.http);
		let clock = Arc::clone(&self.clock);
		let custom = Arc::clone(&self.custom);
		let coordinator = Arc::clone(&self.coordinator);
		async move {
			let record = accounts.account(&account).ok_or_else(auth_not_found)?;
			let provider = catalog
				.provider(&record.provider)
				.ok_or_else(auth_not_found)?;
			let mut runtime = None;
			for id in &provider.auth {
				let auth = catalog.auth_spec(id).ok_or_else(auth_not_found)?;
				if auth.kind != AuthSpecKind::Oauth {
					continue;
				}
				let oauth_id = auth.oauth.as_ref().ok_or_else(auth_unavailable)?;
				let oauth = catalog.oauth_spec(oauth_id).ok_or_else(auth_unavailable)?;
				let candidate =
					AuthSpec::from_catalog(auth, Some(oauth), None).map_err(|_| auth_unavailable())?;
				runtime = match candidate {
					AuthSpec::OAuthCustom(spec) => Some(OAuthRefreshRuntime::Custom(spec)),
					candidate => oauth_client(&candidate).map(OAuthRefreshRuntime::Standard),
				};
				if runtime.is_some() {
					break;
				}
			}
			let runtime = runtime.ok_or_else(auth_unavailable)?;
			let metadata = store
				.metadata(&account)
				.map_err(|_| auth_store_failure())?
				.ok_or_else(auth_not_found)?;
			if metadata.principal_id != record.principal {
				return Err(auth_store_failure());
			}
			let requested_at = clock.now();
			let expires_at = metadata
				.expires_at_ms
				.map(system_time_from_millis)
				.transpose()?;
			let engine = OAuthEngine::new(http.as_ref(), clock.as_ref());
			let request = RefreshRequest {
				account: account.clone(),
				principal: record.principal.clone(),
				rejected: CredentialFreshness {
					generation: metadata.generation,
					issued_at: Some(system_time_from_millis(metadata.updated_at_ms)?),
					expires_at,
					observed_at: requested_at,
				},
				requested_at,
			};
			let outcome = match runtime {
				OAuthRefreshRuntime::Standard(client) => {
					engine
						.refresh_persisted(
							&coordinator,
							Arc::clone(&store),
							client,
							request,
							CredentialOrigin::Persistent,
						)
						.await
				},
				OAuthRefreshRuntime::Custom(spec) => {
					engine
						.refresh_custom_persisted(
							&coordinator,
							Arc::clone(&store),
							custom,
							spec,
							request,
							CredentialOrigin::Persistent,
						)
						.await
				},
			}
			.map_err(oauth_manager_error)?;
			if !accounts
				.update_credential_generation(
					&account,
					&record.principal,
					outcome.result.freshness.generation,
				)
				.map_err(|_| auth_store_failure())?
			{
				return Err(auth_store_failure());
			}
			Ok(AccountSummary {
				account,
				provider: record.provider,
				principal: Some(record.principal),
				label: None,
				state: AccountState::Active,
			})
		}
		.boxed()
	}
}

/// Typed failure constructing a complete direct authentication service.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum AuthManagerBuildError {
	/// The catalog advertises a login method with no constructed engine.
	#[error("catalog authentication method has no constructed login engine")]
	MissingLoginEngine(AuthMethod),
	/// A provider references an authentication specification absent from the
	/// catalog.
	#[error("provider references an unknown authentication specification")]
	UnknownAuthSpec(AuthSpecId),
}

/// Direct, route-independent authentication and account-management service.
///
/// Login engine selection is derived exclusively from typed catalog records.
/// Secret inputs move through bounded session channels and are never retained
/// by the manager. List, refresh, and logout bypass model routing and wire
/// codecs.
#[derive(Clone)]
pub struct AuthManager {
	catalog:  Arc<Catalog>,
	store:    Arc<CredentialStore>,
	broker:   CredentialBroker,
	accounts: AccountPool,
	login:    Arc<BTreeMap<AuthMethodKey, Vec<Arc<dyn AuthLoginEngine>>>>,
	refresh:  Arc<dyn AuthRefreshEngine>,
	sessions: Arc<Mutex<BTreeMap<LoginSessionId, Sender<AuthResponse>>>>,
}

impl AuthManager {
	/// Constructs a complete manager, preserving registration order among
	/// engines for the same public method.
	pub fn new(
		catalog: Arc<Catalog>,
		store: Arc<CredentialStore>,
		broker: CredentialBroker,
		accounts: AccountPool,
		login_engines: Vec<Arc<dyn AuthLoginEngine>>,
		refresh: Arc<dyn AuthRefreshEngine>,
	) -> Result<Self, AuthManagerBuildError> {
		let mut login: BTreeMap<AuthMethodKey, Vec<Arc<dyn AuthLoginEngine>>> = BTreeMap::new();
		for engine in login_engines {
			login
				.entry(AuthMethodKey::from(engine.method()))
				.or_default()
				.push(engine);
		}
		let required = required_login_methods(&catalog)?;
		for method in required {
			if !login.contains_key(&method) {
				return Err(AuthManagerBuildError::MissingLoginEngine(method.into()));
			}
		}
		Ok(Self {
			catalog,
			store,
			broker,
			accounts,
			login: Arc::new(login),
			refresh,
			sessions: Arc::new(Mutex::new(BTreeMap::new())),
		})
	}

	/// Executes one route-independent authentication operation.
	pub async fn execute(&self, request: AuthRequest) -> Result<AuthAnswer, Error> {
		match request {
			AuthRequest::Login(request) => self.login(request).await,
			AuthRequest::Submit { session, input } => {
				let sender = self
					.sessions
					.lock()
					.get(&session)
					.cloned()
					.ok_or_else(auth_not_found)?;
				let cancelled = matches!(input, crate::call::AuthInput::Cancel);
				if sender
					.send_async(AuthResponse { session: session.clone(), input })
					.await
					.is_err()
				{
					self.sessions.lock().remove(&session);
					return Err(auth_not_found());
				}
				if cancelled {
					self.sessions.lock().remove(&session);
				}
				Ok(AuthAnswer::Submitted(session))
			},
			AuthRequest::ListAccounts { provider } => {
				let accounts = self
					.accounts
					.accounts()
					.into_iter()
					.filter(|record| {
						provider
							.as_ref()
							.is_none_or(|provider| provider == &record.provider)
					})
					.map(|record| AccountSummary {
						account:   record.account,
						provider:  record.provider,
						principal: Some(record.principal),
						label:     None,
						state:     if record.enabled {
							AccountState::Active
						} else {
							AccountState::Disabled
						},
					})
					.collect();
				Ok(AuthAnswer::Accounts(accounts))
			},
			AuthRequest::Refresh { account } => self
				.refresh
				.refresh(account)
				.await
				.map(AuthAnswer::Refreshed),
			AuthRequest::Logout { account } => {
				let stored = self
					.store
					.delete(&account)
					.map_err(|_| auth_store_failure())?;
				let pooled = self.accounts.remove(&account).is_some();
				if !stored && !pooled {
					return Err(auth_not_found());
				}
				Ok(AuthAnswer::LoggedOut(account))
			},
		}
	}

	/// Returns the shared catalog-aware credential source used by route
	/// execution.
	#[must_use]
	pub const fn credential_broker(&self) -> &CredentialBroker {
		&self.broker
	}

	async fn login(&self, request: LoginRequest) -> Result<AuthAnswer, Error> {
		let provider = self
			.catalog
			.provider(&request.provider)
			.ok_or_else(auth_not_found)?;
		let (spec, method) = select_auth_spec(&self.catalog, &provider.auth, request.method)?
			.ok_or_else(auth_not_found)?;
		let engines = self
			.login
			.get(&AuthMethodKey::from(method))
			.ok_or_else(auth_unavailable)?;
		let engine = select_login_engine(engines, &request.provider).ok_or_else(auth_unavailable)?;
		let session = engine.begin(request, spec).await?;
		self
			.sessions
			.lock()
			.insert(session.id.clone(), session.responses.clone());
		Ok(AuthAnswer::Session(session))
	}
}

impl fmt::Debug for AuthManager {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter
			.debug_struct("AuthManager")
			.field("login_engines", &self.login.keys())
			.field("active_sessions", &self.sessions.lock().len())
			.field("credential_broker", &self.broker)
			.finish_non_exhaustive()
	}
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum AuthMethodKey {
	ApiKey,
	OAuthPkce,
	OAuthDevice,
	ApplicationDefault,
	AwsCredentialChain,
	SessionToken,
}

impl From<AuthMethod> for AuthMethodKey {
	fn from(method: AuthMethod) -> Self {
		match method {
			AuthMethod::ApiKey => Self::ApiKey,
			AuthMethod::OAuthPkce => Self::OAuthPkce,
			AuthMethod::OAuthDevice => Self::OAuthDevice,
			AuthMethod::ApplicationDefault => Self::ApplicationDefault,
			AuthMethod::AwsCredentialChain => Self::AwsCredentialChain,
			AuthMethod::SessionToken => Self::SessionToken,
		}
	}
}

fn select_auth_spec(
	catalog: &Catalog,
	auth: &[AuthSpecId],
	requested: Option<AuthMethod>,
) -> Result<Option<(AuthSpecId, AuthMethod)>, Error> {
	let mut fallback = None;
	for id in auth {
		let spec = catalog.auth_spec(id).ok_or_else(auth_not_found)?;
		let method = auth_method(catalog, spec)?;
		if let Some(requested) = requested {
			if requested == method {
				return Ok(Some((id.clone(), method)));
			}
		} else {
			fallback.get_or_insert_with(|| (id.clone(), method));
			if matches!(method, AuthMethod::OAuthPkce | AuthMethod::OAuthDevice) {
				return Ok(Some((id.clone(), method)));
			}
		}
	}
	Ok(fallback)
}

impl From<AuthMethodKey> for AuthMethod {
	fn from(method: AuthMethodKey) -> Self {
		match method {
			AuthMethodKey::ApiKey => Self::ApiKey,
			AuthMethodKey::OAuthPkce => Self::OAuthPkce,
			AuthMethodKey::OAuthDevice => Self::OAuthDevice,
			AuthMethodKey::ApplicationDefault => Self::ApplicationDefault,
			AuthMethodKey::AwsCredentialChain => Self::AwsCredentialChain,
			AuthMethodKey::SessionToken => Self::SessionToken,
		}
	}
}

fn required_login_methods(
	catalog: &Catalog,
) -> Result<BTreeSet<AuthMethodKey>, AuthManagerBuildError> {
	let mut required = BTreeSet::new();
	for provider in catalog.providers() {
		for id in &provider.auth {
			let spec = catalog
				.auth_spec(id)
				.ok_or_else(|| AuthManagerBuildError::UnknownAuthSpec(id.clone()))?;
			if spec.kind == AuthSpecKind::None {
				continue;
			}
			required.insert(AuthMethodKey::from(
				auth_method(catalog, spec)
					.map_err(|_| AuthManagerBuildError::UnknownAuthSpec(id.clone()))?,
			));
		}
	}
	Ok(required)
}
fn select_login_engine<'a>(
	engines: &'a [Arc<dyn AuthLoginEngine>],
	provider: &omp_llm_catalog::ProviderId,
) -> Option<&'a Arc<dyn AuthLoginEngine>> {
	engines.iter().find(|engine| engine.supports(provider))
}

fn auth_method(
	catalog: &Catalog,
	spec: &omp_llm_catalog::provider::AuthSpec,
) -> Result<AuthMethod, Error> {
	match spec.kind {
		AuthSpecKind::None => Err(auth_unavailable()),
		AuthSpecKind::ApiKey | AuthSpecKind::Bearer => Ok(AuthMethod::ApiKey),
		AuthSpecKind::AzureAd | AuthSpecKind::GithubApp => Ok(AuthMethod::SessionToken),
		AuthSpecKind::GcpAdc => Ok(AuthMethod::ApplicationDefault),
		AuthSpecKind::AwsSigv4 => Ok(AuthMethod::AwsCredentialChain),
		AuthSpecKind::OmpSession => Ok(AuthMethod::SessionToken),
		AuthSpecKind::Oauth => {
			let id = spec.oauth.as_ref().ok_or_else(auth_unavailable)?;
			let oauth = catalog.oauth_spec(id).ok_or_else(auth_unavailable)?;
			Ok(match &oauth.flow {
				OAuthFlowSpec::DeviceCode { .. } => AuthMethod::OAuthDevice,
				OAuthFlowSpec::Pkce { .. } | OAuthFlowSpec::Paste { .. } => AuthMethod::OAuthPkce,
				OAuthFlowSpec::Custom { polling: Some(_), .. } => AuthMethod::OAuthDevice,
				OAuthFlowSpec::Custom { polling: None, .. } => AuthMethod::OAuthPkce,
			})
		},
	}
}

fn auth_not_found() -> Error {
	Error::new(
		ErrorKind::TargetNotFound,
		ErrorPhase::Authentication,
		RetryAction::Never,
		ExecutionReceipt::default(),
	)
}

fn auth_unavailable() -> Error {
	Error::new(
		ErrorKind::Authentication,
		ErrorPhase::Authentication,
		RetryAction::Never,
		ExecutionReceipt::default(),
	)
}

fn auth_store_failure() -> Error {
	Error::new(
		ErrorKind::InternalInvariant,
		ErrorPhase::Authentication,
		RetryAction::Never,
		ExecutionReceipt::default(),
	)
}

fn auth_store_error(error: StoreError) -> Error {
	if matches!(error, StoreError::Key(KeyError::Unavailable | KeyError::OsCredential)) {
		Error::new(
			ErrorKind::CredentialStorageUnavailable,
			ErrorPhase::Authentication,
			RetryAction::Never,
			ExecutionReceipt::default(),
		)
	} else {
		auth_store_failure()
	}
}

static LOGIN_SESSION_SEQUENCE: AtomicU64 = AtomicU64::new(1);

fn next_login_session_id() -> LoginSessionId {
	let sequence = LOGIN_SESSION_SEQUENCE.fetch_add(1, Ordering::Relaxed);
	LoginSessionId::from(format!("auth-{sequence}"))
}

fn unix_millis(time: SystemTime) -> Result<u64, Error> {
	let millis = time
		.duration_since(UNIX_EPOCH)
		.map_err(|_| auth_store_failure())?
		.as_millis();
	u64::try_from(millis).map_err(|_| auth_store_failure())
}

fn login_channel_error(error: LoginChannelError) -> Error {
	match error {
		LoginChannelError::Cancelled => Error::new(
			ErrorKind::Cancelled,
			ErrorPhase::Authentication,
			RetryAction::Never,
			ExecutionReceipt::default(),
		),
		_ => auth_unavailable(),
	}
}

fn auth_invalid_request() -> Error {
	Error::new(
		ErrorKind::InvalidRequest,
		ErrorPhase::Authentication,
		RetryAction::Never,
		ExecutionReceipt::default(),
	)
}

fn oauth_client(spec: &AuthSpec) -> Option<OAuthClientSpec> {
	match spec {
		AuthSpec::OAuthPkce(spec) => Some(spec.client.clone()),
		AuthSpec::OAuthDevice(spec) => Some(spec.client.clone()),
		AuthSpec::OAuthPaste(spec) => Some(spec.client.clone()),
		AuthSpec::OAuthCustom(spec) => Some(spec.client.clone()),
		_ => None,
	}
}

fn system_time_from_millis(millis: u64) -> Result<SystemTime, Error> {
	UNIX_EPOCH
		.checked_add(std::time::Duration::from_millis(millis))
		.ok_or_else(auth_store_failure)
}

fn principal_unresolved() -> Error {
	Error::new(
		ErrorKind::Authentication,
		ErrorPhase::Authentication,
		RetryAction::Never,
		ExecutionReceipt::default(),
	)
}

fn oauth_error(error: OAuthError) -> Error {
	let detail = ErrorDetail::provider(Str::from(error.to_string()));
	match error {
		OAuthError::Cancelled => Error::new(
			ErrorKind::Cancelled,
			ErrorPhase::Authentication,
			RetryAction::Never,
			ExecutionReceipt::default(),
		),
		OAuthError::PrincipalUnresolved => principal_unresolved().detail(detail),
		OAuthError::Provider { status, code, .. } => auth_unavailable()
			.status(Some(status))
			.code(Str::from(code.as_str()))
			.detail(detail),
		_ => auth_unavailable().detail(detail),
	}
}

fn oauth_custom_error(error: OAuthCustomDispatchError) -> Error {
	match error {
		OAuthCustomDispatchError::Protocol(error) => oauth_error(error),
		OAuthCustomDispatchError::Duplicate(_) | OAuthCustomDispatchError::Unavailable(_) => {
			auth_unavailable()
		},
	}
}

fn oauth_manager_error(error: OAuthCredentialManagerError) -> Error {
	match error {
		OAuthCredentialManagerError::OAuth(error) => oauth_error(*error),
		OAuthCredentialManagerError::Refresh(_) => auth_unavailable(),
		OAuthCredentialManagerError::Expired => Error::new(
			ErrorKind::Authentication,
			ErrorPhase::Authentication,
			RetryAction::RefreshCredential,
			ExecutionReceipt::default(),
		),
		OAuthCredentialManagerError::Store(error) => auth_store_error(*error),
		OAuthCredentialManagerError::InvalidTime => auth_store_failure(),
	}
}

fn credential_error(error: CredentialError) -> Error {
	match error {
		CredentialError::Cancelled => Error::new(
			ErrorKind::Cancelled,
			ErrorPhase::Authentication,
			RetryAction::Never,
			ExecutionReceipt::default(),
		),
		CredentialError::Expired => Error::new(
			ErrorKind::Authentication,
			ErrorPhase::Authentication,
			RetryAction::RefreshCredential,
			ExecutionReceipt::default(),
		),
		CredentialError::Unavailable
		| CredentialError::StaleGeneration
		| CredentialError::InvalidSource
		| CredentialError::SourceFailure => auth_unavailable(),
	}
}
#[cfg(test)]
mod tests {
	use std::{
		collections::VecDeque,
		sync::Arc,
		time::{Duration, SystemTime, UNIX_EPOCH},
	};

	use futures::future::{BoxFuture, FutureExt as _};
	use http::HeaderMap;
	use omp_llm_catalog::{ProviderId, provider::AuthSpecKind, snapshot::Catalog};
	use parking_lot::Mutex;
	use secrecy::{ExposeSecret as _, SecretString};

	use super::{
		AuthLoginEngine, AuthRefreshEngine, OAuthLoginEngine, StoredOAuthRefreshEngine, auth_method,
		auth_store_error, select_auth_spec, select_login_engine,
	};
	use crate::{
		account::{AccountPool, RefreshCoordinator, RefreshPolicy},
		answer::AuthEvent,
		auth::{
			AlibabaTokenPlanLoginEngine, CredentialStore, HeadlessKeySource, KeyError, KeyId,
			OAuthClock, OAuthCustomDispatcher, OAuthHttpClient, OAuthHttpRequest, OAuthHttpResponse,
			OAuthTransportError, SecretLoginEngine, StoreError,
		},
		call::{AuthMethod, LoginRequest},
		error::ErrorKind,
	};

	#[test]
	fn unavailable_credential_key_has_distinct_error_kind() {
		let error = auth_store_error(StoreError::Key(KeyError::Unavailable));
		assert_eq!(error.kind, ErrorKind::CredentialStorageUnavailable);
	}

	struct ImmediateClock(SystemTime);

	impl OAuthClock for ImmediateClock {
		fn now(&self) -> SystemTime {
			self.0
		}

		fn sleep(&self, _duration: Duration) -> BoxFuture<'_, ()> {
			async {}.boxed()
		}
	}

	#[test]
	fn embedded_copilot_bearer_then_device_prefers_interactive_login() {
		let catalog = Catalog::embedded();
		let provider = catalog
			.provider(&ProviderId::from("github-copilot"))
			.expect("GitHub Copilot provider");
		let spec_for = |method| {
			provider
				.auth
				.iter()
				.find(|id| {
					catalog.auth_spec(id).is_some_and(|spec| {
						auth_method(catalog, spec).is_ok_and(|actual| actual == method)
					})
				})
				.cloned()
				.expect("Copilot auth method")
		};
		let bearer = spec_for(AuthMethod::ApiKey);
		let device = spec_for(AuthMethod::OAuthDevice);
		let auth = [bearer.clone(), device.clone()];

		let default = select_auth_spec(catalog, &auth, None)
			.expect("valid Copilot auth specs")
			.expect("default Copilot auth spec");
		assert_eq!(default, (device, AuthMethod::OAuthDevice));

		let api_key = select_auth_spec(catalog, &auth, Some(AuthMethod::ApiKey))
			.expect("valid Copilot auth specs")
			.expect("Copilot bearer auth spec");
		assert_eq!(api_key, (bearer, AuthMethod::ApiKey));
	}

	#[test]
	fn plain_bearer_auth_is_an_api_key_login_method() {
		let catalog = Catalog::embedded();
		let provider = catalog
			.provider(&ProviderId::from("alibaba-token-plan"))
			.expect("Alibaba Token Plan provider");
		let spec = catalog
			.auth_spec(provider.auth.first().expect("Alibaba auth spec id"))
			.expect("Alibaba auth spec");
		assert_eq!(spec.kind, AuthSpecKind::Bearer);
		assert_eq!(auth_method(catalog, spec).expect("login method"), AuthMethod::ApiKey);
	}

	#[test]
	fn alibaba_scoped_api_key_engine_precedes_generic_engine() {
		let catalog = Arc::new(Catalog::embedded().clone());
		let provider = ProviderId::from("alibaba-token-plan");
		let suffix = SystemTime::now()
			.duration_since(UNIX_EPOCH)
			.expect("current timestamp")
			.as_nanos();
		let path = std::env::temp_dir()
			.join(format!("omp-alibaba-dispatch-{}-{suffix}.sqlite", std::process::id()));
		let store = Arc::new(
			CredentialStore::open(
				&path,
				Arc::new(HeadlessKeySource::new(KeyId::new("alibaba-dispatch"), [8; 32])),
			)
			.expect("credential store"),
		);
		let http: Arc<dyn OAuthHttpClient> = Arc::new(FixtureHttp {
			responses: Mutex::new(VecDeque::new()),
			requests:  Mutex::new(Vec::new()),
		});
		let scoped: Arc<dyn AuthLoginEngine> = Arc::new(AlibabaTokenPlanLoginEngine::new(
			Arc::clone(&catalog),
			Arc::clone(&store),
			AccountPool::new(),
			http,
		));
		let generic: Arc<dyn AuthLoginEngine> = Arc::new(
			SecretLoginEngine::new(
				AuthMethod::ApiKey,
				"generic".into(),
				catalog,
				store,
				AccountPool::new(),
			)
			.expect("generic API-key engine"),
		);
		let engines = vec![Arc::clone(&scoped), generic];
		let selected = select_login_engine(&engines, &provider).expect("supporting engine");
		assert!(Arc::ptr_eq(selected, &scoped));
		drop(engines);
		drop(scoped);
		let _ = std::fs::remove_file(path);
	}

	struct FixtureHttp {
		responses: Mutex<VecDeque<OAuthHttpResponse>>,
		requests:  Mutex<Vec<(String, String)>>,
	}

	impl OAuthHttpClient for FixtureHttp {
		fn execute(
			&self,
			request: OAuthHttpRequest,
		) -> BoxFuture<'_, Result<OAuthHttpResponse, OAuthTransportError>> {
			let (_, url, _, body) = request.into_parts();
			self.requests.lock().push((
				url.to_string(),
				body.map_or_else(String::new, |body| body.expose_secret().to_owned()),
			));
			let response = self.responses.lock().pop_front().expect("fixture response");
			async move { Ok(response) }.boxed()
		}
	}

	#[tokio::test]
	async fn embedded_kimi_login_starts_and_resolves_its_principal() {
		let catalog = Arc::new(Catalog::embedded().clone());
		let provider = ProviderId::from("kimi-code");
		let auth = catalog
			.provider(&provider)
			.and_then(|provider| provider.auth.first())
			.cloned()
			.expect("Kimi OAuth auth spec");
		let suffix = SystemTime::now()
			.duration_since(UNIX_EPOCH)
			.unwrap()
			.as_nanos();
		let store_path = std::env::temp_dir()
			.join(format!("omp-kimi-login-{}-{suffix}.sqlite", std::process::id()));
		let store = Arc::new(
			CredentialStore::open(
				&store_path,
				Arc::new(HeadlessKeySource::new(KeyId::new("kimi-login-test"), [9; 32])),
			)
			.unwrap(),
		);
		let http = Arc::new(FixtureHttp {
			responses: Mutex::new(VecDeque::from([
				OAuthHttpResponse {
					status:  200,
					headers: HeaderMap::new(),
					body:    SecretString::from(
						r#"{"device_code":"device","user_code":"ABCD-EFGH","verification_uri":"https://www.kimi.com/code/authorize_device","verification_uri_complete":"https://www.kimi.com/code/authorize_device?user_code=ABCD-EFGH","expires_in":1800,"interval":5}"#
							.to_owned(),
					),
				},
				OAuthHttpResponse {
					status:  200,
					headers: HeaderMap::new(),
					body:    SecretString::from(
						r#"{"access_token":"header.eyJ1c2VyX2lkIjoia2ltaS11c2VyLTQyIiwic3ViIjoiZmFsbGJhY2sifQ.signature","refresh_token":"refresh","token_type":"Bearer","expires_in":3600}"#
							.to_owned(),
					),
				},
				OAuthHttpResponse {
					status:  200,
					headers: HeaderMap::new(),
					body:    SecretString::from(
						r#"{"access_token":"header.eyJ1c2VyX2lkIjoia2ltaS11c2VyLTQyIiwic3ViIjoiZmFsbGJhY2sifQ.signature","refresh_token":"refresh-2","token_type":"Bearer","expires_in":3600}"#
							.to_owned(),
					),
				},
			])),
			requests:  Mutex::new(Vec::new()),
		});
		let accounts = AccountPool::new();
		let custom = Arc::new(OAuthCustomDispatcher::new());
		let clock = Arc::new(ImmediateClock(SystemTime::UNIX_EPOCH));
		let engine = OAuthLoginEngine::new(
			AuthMethod::OAuthDevice,
			Arc::clone(&catalog),
			Arc::clone(&store),
			accounts.clone(),
			Arc::clone(&http),
			Arc::clone(&clock),
			Arc::clone(&custom),
		)
		.unwrap();
		let session = engine
			.begin(LoginRequest { provider, method: None }, auth)
			.await
			.unwrap();
		let mut saw_code = false;
		let completed = loop {
			let event = tokio::time::timeout(Duration::from_secs(1), session.events.recv_async())
				.await
				.expect("Kimi login event")
				.expect("Kimi login event channel")
				.expect("successful Kimi login event");
			match event {
				AuthEvent::ShowDeviceCode { code, verification_url } => {
					assert_eq!(code.expose_secret(), "ABCD-EFGH");
					assert_eq!(verification_url, "https://www.kimi.com/code/authorize_device");
					saw_code = true;
				},
				AuthEvent::Complete(account) => {
					assert_eq!(account.account.as_str(), "kimi-code:kimi-user-42");
					assert_eq!(
						account
							.principal
							.as_ref()
							.map(|principal| principal.as_str()),
						Some("kimi-user-42")
					);
					break account;
				},
				AuthEvent::OpenUrl(_) | AuthEvent::Waiting => {},
				AuthEvent::Prompt(_) => panic!("Kimi device flow must not request private input"),
			}
		};
		let refreshed = StoredOAuthRefreshEngine::new(
			catalog,
			store,
			accounts,
			http.clone(),
			clock,
			custom,
			Arc::new(
				RefreshCoordinator::new("kimi-refresh-test", RefreshPolicy::default())
					.expect("refresh coordinator"),
			),
		)
		.refresh(completed.account.clone())
		.await
		.expect("Kimi refresh");
		assert_eq!(refreshed.account, completed.account);
		assert_eq!(refreshed.principal, completed.principal);
		assert!(saw_code);
		let requests = http.requests.lock();
		assert_eq!(requests.len(), 3);
		assert_eq!(requests[0].0, "https://auth.kimi.com/api/oauth/device_authorization");
		assert!(
			requests[0]
				.1
				.contains("client_id=17e5f671-d194-4dfb-9706-5516cb48c098")
		);
		assert!(!requests[0].1.contains("scope="));
		assert!(
			requests[1]
				.1
				.contains("grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Adevice_code")
		);
		assert!(requests[1].1.contains("device_code=device"));
		assert!(requests[2].1.contains("grant_type=refresh_token"));
		assert!(requests[2].1.contains("refresh_token=refresh"));
		drop(requests);
		drop(session);
		drop(engine);
		let _ = std::fs::remove_file(store_path);
	}
}
