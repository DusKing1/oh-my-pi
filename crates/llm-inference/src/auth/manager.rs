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
use secrecy::{ExposeSecret as _, SecretBox, SecretString};

use super::{
	AuthSpec, CredentialBroker, CredentialError, CredentialNeed, CredentialOrigin, CredentialSource,
	CredentialStore, CredentialWrite, LoginChannelError, OAuthClientSpec, OAuthClock,
	OAuthCredentialManagerError, OAuthCustomDispatchError, OAuthCustomDispatcher, OAuthEngine,
	OAuthError, OAuthHttpClient, default_login_channels,
};
use crate::{
	account::{
		AccountPool, AccountRecord, CredentialFreshness, RefreshCoordinator, RefreshPolicy,
		RefreshRequest,
	},
	answer::{
		AccountState, AccountSummary, AuthAnswer, AuthEvent, AuthPrompt, AuthPromptKind,
		AuthResponse, AuthSession,
	},
	call::{AccountRoutingContext, AuthInput, AuthMethod, AuthRequest, LoginRequest},
	error::{Error, ErrorKind, ErrorPhase, RetryAction},
	id::{AccountId, LoginSessionId, PrincipalId},
	receipt::ExecutionReceipt,
};
/// One constructed engine for a typed public login method.
pub trait AuthLoginEngine: Send + Sync {
	/// Public login method implemented by this engine.
	fn method(&self) -> AuthMethod;

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
			let expected = match method {
				AuthMethod::ApiKey => AuthSpecKind::ApiKey,
				AuthMethod::SessionToken => AuthSpecKind::OmpSession,
				_ => return Err(auth_unavailable()),
			};
			if auth.kind != expected {
				return Err(auth_unavailable());
			}
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
					let secret = match (method, input) {
						(AuthMethod::ApiKey, AuthInput::ApiKey(secret))
						| (AuthMethod::SessionToken, AuthInput::SessionToken(secret)) => secret,
						_ => return Err(auth_invalid_request()),
					};
					let principal = PrincipalId::from(principal_label.clone());
					let account = AccountId::from(format!("{provider_id}:{principal_label}"));
					let bytes = SecretBox::new(Box::new(secret.expose_secret().as_bytes().to_vec()));
					let metadata = store
						.put(CredentialWrite {
							account_id:          &account,
							principal_id:        &principal,
							kind:                match method {
								AuthMethod::ApiKey => "api-key",
								_ => "session-token",
							},
							secret:              &bytes,
							expires_at_ms:       None,
							origin:              CredentialOrigin::Persistent,
							now_ms:              unix_millis(SystemTime::now())?,
							expected_generation: None,
						})
						.map_err(|_| auth_store_failure())?;
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
							.map_err(|error| oauth_custom_error(error))?,
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

/// Owned refresh adapter for encrypted OAuth credentials.
pub struct StoredOAuthRefreshEngine<C, K> {
	catalog:     Arc<Catalog>,
	store:       Arc<CredentialStore>,
	accounts:    AccountPool,
	http:        Arc<C>,
	clock:       Arc<K>,
	coordinator: Arc<RefreshCoordinator>,
}

impl<C, K> StoredOAuthRefreshEngine<C, K> {
	/// Constructs an OAuth refresh adapter over one shared coordinator.
	#[must_use]
	pub fn new(
		catalog: Arc<Catalog>,
		store: Arc<CredentialStore>,
		accounts: AccountPool,
		http: Arc<C>,
		clock: Arc<K>,
		coordinator: Arc<RefreshCoordinator>,
	) -> Self {
		Self { catalog, store, accounts, http, clock, coordinator }
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
		let coordinator = Arc::clone(&self.coordinator);
		async move {
			let record = accounts.account(&account).ok_or_else(auth_not_found)?;
			let provider = catalog
				.provider(&record.provider)
				.ok_or_else(auth_not_found)?;
			let mut client = None;
			for id in &provider.auth {
				let auth = catalog.auth_spec(id).ok_or_else(auth_not_found)?;
				if auth.kind != AuthSpecKind::Oauth {
					continue;
				}
				let oauth_id = auth.oauth.as_ref().ok_or_else(auth_unavailable)?;
				let oauth = catalog.oauth_spec(oauth_id).ok_or_else(auth_unavailable)?;
				let runtime =
					AuthSpec::from_catalog(auth, Some(oauth), None).map_err(|_| auth_unavailable())?;
				client = oauth_client(&runtime);
				if client.is_some() {
					break;
				}
			}
			let client = client.ok_or_else(auth_unavailable)?;
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
			let outcome = engine
				.refresh_persisted(
					&coordinator,
					Arc::clone(&store),
					client,
					RefreshRequest {
						account: account.clone(),
						principal: record.principal.clone(),
						rejected: CredentialFreshness {
							generation: metadata.generation,
							issued_at: Some(system_time_from_millis(metadata.updated_at_ms)?),
							expires_at,
							observed_at: requested_at,
						},
						requested_at,
					},
					CredentialOrigin::Persistent,
				)
				.await
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
	/// Two engines claim the same public login method.
	#[error("duplicate authentication login engine")]
	DuplicateLoginEngine(AuthMethod),
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
	login:    Arc<BTreeMap<AuthMethodKey, Arc<dyn AuthLoginEngine>>>,
	refresh:  Arc<dyn AuthRefreshEngine>,
	sessions: Arc<Mutex<BTreeMap<LoginSessionId, Sender<AuthResponse>>>>,
}

impl AuthManager {
	/// Constructs a complete manager, rejecting duplicate or missing advertised
	/// engines.
	pub fn new(
		catalog: Arc<Catalog>,
		store: Arc<CredentialStore>,
		broker: CredentialBroker,
		accounts: AccountPool,
		login_engines: Vec<Arc<dyn AuthLoginEngine>>,
		refresh: Arc<dyn AuthRefreshEngine>,
	) -> Result<Self, AuthManagerBuildError> {
		let mut login = BTreeMap::new();
		for engine in login_engines {
			let method = engine.method();
			if login.insert(AuthMethodKey::from(method), engine).is_some() {
				return Err(AuthManagerBuildError::DuplicateLoginEngine(method));
			}
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
		let mut selected = None;
		for id in &provider.auth {
			let spec = self.catalog.auth_spec(id).ok_or_else(auth_not_found)?;
			let method = auth_method(&self.catalog, spec)?;
			if request.method.is_none_or(|requested| requested == method) {
				selected = Some((id.clone(), method));
				break;
			}
		}
		let (spec, method) = selected.ok_or_else(auth_not_found)?;
		let engine = self
			.login
			.get(&AuthMethodKey::from(method))
			.ok_or_else(auth_unavailable)?;
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

fn auth_method(
	catalog: &Catalog,
	spec: &omp_llm_catalog::provider::AuthSpec,
) -> Result<AuthMethod, Error> {
	match spec.kind {
		AuthSpecKind::None => Err(auth_unavailable()),
		AuthSpecKind::ApiKey => Ok(AuthMethod::ApiKey),
		AuthSpecKind::Bearer | AuthSpecKind::AzureAd | AuthSpecKind::GithubApp => {
			Ok(AuthMethod::SessionToken)
		},
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
	match error {
		OAuthError::Cancelled => Error::new(
			ErrorKind::Cancelled,
			ErrorPhase::Authentication,
			RetryAction::Never,
			ExecutionReceipt::default(),
		),
		OAuthError::PrincipalUnresolved => principal_unresolved(),
		_ => auth_unavailable(),
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
		OAuthCredentialManagerError::OAuth(error) => oauth_error(error),
		OAuthCredentialManagerError::Refresh(_) => auth_unavailable(),
		OAuthCredentialManagerError::Expired => Error::new(
			ErrorKind::Authentication,
			ErrorPhase::Authentication,
			RetryAction::RefreshCredential,
			ExecutionReceipt::default(),
		),
		OAuthCredentialManagerError::Store(_) | OAuthCredentialManagerError::InvalidTime => {
			auth_store_failure()
		},
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
