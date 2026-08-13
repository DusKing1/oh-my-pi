//! Production typed inference registry construction and daemon lifecycle.

use std::{
	collections::BTreeMap,
	path::{Path, PathBuf},
	sync::Arc,
	time::Duration,
};

use omp_core::Str;
use omp_llm_inference::{
	Client, ProviderService, Registry,
	account::{
		AccountPool, AccountStateStore, AccountStateStoreError, RefreshCoordinator, RefreshPolicy,
	},
	auth::{
		AuthLoginEngine, AuthManager, AuthManagerBuildError, CredentialAcquisitionLoginEngine,
		CredentialAcquisitionLoginEngineError, CredentialBroker, CredentialBrokerEngines,
		CredentialStore, KeyError, KeySource, OAuthCustomDispatcher, OAuthLoginEngine,
		OAuthLoginEngineError, OsCredentialKeySource, SecretLoginEngine, SecretLoginEngineError,
		StoreError, StoredCredentialSource, StoredOAuthRefreshEngine, SystemOAuthClock,
		SystemOAuthHttpClient, UnavailableKeySource,
	},
	call::AuthMethod,
	codec::google_cca::{AntigravityPolicy, CcaHeaders},
	layer::{
		admission::AdmissionController,
		observe::{ExecutionFinished, ExecutionStarted, Observer},
		stack::BuiltinConfig,
	},
	provider::builtin::{
		AuthApplicationConfig, GoogleCcaConfig, LocalRouteBackend, ProductionDependencies,
	},
	router::Router,
	session::{ConversationError, ConversationSessionPlanner},
	transport::{http::HttpTransport, websocket_transport::WebSocketTransport},
};
use omp_proto::{
	auth::v1::auth_server::AuthServer, blob::v1::blob_server::BlobServer,
	inference::v1::inference_server::InferenceServer,
};
use omp_storage::blob::BlobStore;
use tokio::{sync::watch, task::JoinHandle};
use tonic::transport::Server;

use crate::{
	auth_rpc::AuthRpc, blob_rpc::BlobRpc, endpoint::LocalEndpoint, rpc_adapter::InferenceRpc,
};

const DATA_DIR_ENV: &str = "OMP_DATA_DIR";
const KEYCHAIN_OPT_IN_ENV: &str = "OMP_LLM_KEYCHAIN";
const KEYCHAIN_SERVICE: &str = "dev.omp.llm";
const KEYCHAIN_ACCOUNT: &str = "credential-store-master";

/// Selection of the credential encryption-key source.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CredentialKeyMode {
	/// Fail closed without contacting an operating-system credential service.
	#[default]
	Unavailable,
	/// Use the operating-system credential service after explicit application
	/// opt-in.
	OsKeychain,
}

impl CredentialKeyMode {
	/// Selects the OS keychain only for exact `OMP_LLM_KEYCHAIN=1`; unset or any
	/// other value fails closed without OS access.
	#[must_use]
	pub fn from_environment() -> Self {
		Self::from_value(std::env::var_os(KEYCHAIN_OPT_IN_ENV).as_deref())
	}

	fn from_value(value: Option<&std::ffi::OsStr>) -> Self {
		match value {
			Some(value) if value == "1" => Self::OsKeychain,
			_ => Self::Unavailable,
		}
	}
}

/// Production daemon construction options.
pub struct DaemonConfig {
	data_dir: Option<PathBuf>,
	endpoint: LocalEndpoint,
}

impl DaemonConfig {
	/// Creates the standard owner-local daemon configuration.
	#[must_use]
	pub fn local(endpoint: impl Into<LocalEndpoint>) -> Self {
		let data_dir = std::env::var_os(DATA_DIR_ENV)
			.map(PathBuf::from)
			.or_else(|| {
				std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share/omp"))
			});
		Self { data_dir, endpoint: endpoint.into() }
	}

	/// Overrides the directory containing encrypted credentials and session
	/// state.
	#[must_use]
	pub fn with_data_dir(mut self, data_dir: PathBuf) -> Self {
		self.data_dir = Some(data_dir);
		self
	}
}

/// Runtime facts available once registry construction succeeds.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DaemonReadiness {
	/// Requested owner-local endpoint.
	pub endpoint: LocalEndpoint,
	/// Number of catalog routes backed by constructed services.
	pub routes:   usize,
}

/// A production daemon startup or lifecycle failure.
#[derive(Debug, thiserror::Error)]
pub enum DaemonError {
	/// Neither an explicit data directory nor `OMP_DATA_DIR`/`HOME` was
	/// available.
	#[error("daemon data directory is unavailable; set OMP_DATA_DIR or HOME")]
	MissingDataDirectory,
	/// Durable state directory could not be prepared.
	#[error("could not prepare daemon state directory")]
	PrepareState(#[source] std::io::Error),
	/// The checked-in catalog snapshot is invalid.
	#[error("embedded catalog snapshot is invalid")]
	Catalog(#[source] &'static omp_llm_catalog::snapshot::SnapshotError),
	/// Registry construction or route service failed.
	#[error(transparent)]
	Inference(#[from] omp_llm_inference::Error),
	/// Encrypted credential state could not be opened.
	#[error(transparent)]
	CredentialStore(#[from] StoreError),
	/// Credential encryption key provisioning failed.
	#[error(transparent)]
	CredentialKey(#[from] KeyError),
	/// Durable account state could not be opened.
	#[error(transparent)]
	AccountState(#[from] AccountStateStoreError),
	/// A static secret login engine was configured with an unsupported method.
	#[error(transparent)]
	SecretLogin(#[from] SecretLoginEngineError),
	/// A credential acquisition engine was configured with an unsupported
	/// method.
	#[error(transparent)]
	CredentialAcquisitionLogin(#[from] CredentialAcquisitionLoginEngineError),
	/// An OAuth login engine was configured with an unsupported method.
	#[error(transparent)]
	OAuthLogin(#[from] OAuthLoginEngineError),
	/// Refresh coordination policy was invalid.
	#[error(transparent)]
	RefreshPolicy(#[from] omp_llm_inference::account::RefreshPolicyError),
	/// The catalog advertised an authentication method without a concrete
	/// engine.
	#[error(transparent)]
	AuthManager(#[from] AuthManagerBuildError),
	/// Durable conversation state could not be opened.
	#[error(transparent)]
	Conversation(#[from] ConversationError),
	/// Content-addressed blob state could not be opened.
	#[error(transparent)]
	BlobStore(#[from] omp_storage::blob::Error),
	/// Owner-local RPC listener could not bind.
	#[error("could not bind owner-local RPC endpoint")]
	RpcListen(#[source] omp_rpc::Error),
	/// Tonic RPC serving failed.
	#[error("owner-local inference RPC server failed")]
	RpcServe(#[source] tonic::transport::Error),
	/// The daemon RPC task failed to join.
	#[error("owner-local inference RPC task failed")]
	RpcTask(#[source] tokio::task::JoinError),
	/// The RPC server exited before a shutdown request.
	#[error("owner-local inference RPC server stopped unexpectedly")]
	RpcStopped,
	/// Signal handling failed.
	#[error("shutdown signal handling failed")]
	Signal(#[source] std::io::Error),
}

/// Opens encrypted production credential state, contacting the OS keychain only
/// for exact `OMP_LLM_KEYCHAIN=1`.
pub fn open_credential_store(
	database: impl AsRef<Path>,
) -> Result<Arc<CredentialStore>, DaemonError> {
	match CredentialKeyMode::from_environment() {
		CredentialKeyMode::Unavailable => {
			open_credential_store_with_key_source(database, Arc::new(UnavailableKeySource))
		},
		CredentialKeyMode::OsKeychain => {
			let key_source = OsCredentialKeySource::new(KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNT);
			if key_source.active_key().is_err() {
				key_source.rotate()?;
			}
			open_credential_store_with_key_source(database, Arc::new(key_source))
		},
	}
}

/// Opens encrypted credential state with an explicitly supplied non-secret key
/// source.
pub fn open_credential_store_with_key_source(
	database: impl AsRef<Path>,
	key_source: Arc<dyn KeySource>,
) -> Result<Arc<CredentialStore>, DaemonError> {
	Ok(Arc::new(CredentialStore::open(database.as_ref(), key_source)?))
}

pub async fn production_registry(
	data_dir: &Path,
	credential_store: Arc<CredentialStore>,
) -> Result<Registry, DaemonError> {
	std::fs::create_dir_all(data_dir).map_err(DaemonError::PrepareState)?;
	let catalog = Arc::new(
		omp_llm_catalog::snapshot::Catalog::try_embedded()
			.map_err(DaemonError::Catalog)?
			.clone(),
	);
	#[cfg(feature = "local-applefm")]
	let apple_routes = catalog
		.routes()
		.iter()
		.filter(|route| {
			route.codec_profile == omp_llm_catalog::CodecProfile::AppleFm
				&& route.transport == omp_llm_catalog::TransportKind::Local
		})
		.map(|route| route.id.clone())
		.collect::<Vec<_>>();
	let stored = Arc::new(StoredCredentialSource::new(credential_store.clone()));
	let credentials = CredentialBroker::system(&catalog, CredentialBrokerEngines {
		stored: Some(stored),
		..CredentialBrokerEngines::default()
	})
	.map_err(|_| {
		DaemonError::Inference(omp_llm_inference::Error::planning(
			omp_llm_inference::ErrorKind::InvalidRequest,
			omp_llm_inference::ErrorDetail::Target {
				selector: Str::from("catalog-credential-broker-invalid"),
			},
			omp_llm_inference::ExecutionReceipt::default(),
		))
	})?;
	let database = data_dir.join("credentials.db");
	let accounts = AccountPool::with_store(Arc::new(AccountStateStore::open(&database)?))?;
	let oauth_http = Arc::new(SystemOAuthHttpClient::new());
	let oauth_clock = Arc::new(SystemOAuthClock);
	let oauth_custom = Arc::new(OAuthCustomDispatcher::new());
	let refresh_coordinator =
		Arc::new(RefreshCoordinator::new("omp-auth-refresh", RefreshPolicy::default())?);
	let login_engines: Vec<Arc<dyn AuthLoginEngine>> = vec![
		Arc::new(SecretLoginEngine::new(
			AuthMethod::ApiKey,
			Str::from("api-key"),
			catalog.clone(),
			credential_store.clone(),
			accounts.clone(),
		)?),
		Arc::new(SecretLoginEngine::new(
			AuthMethod::SessionToken,
			Str::from("session-token"),
			catalog.clone(),
			credential_store.clone(),
			accounts.clone(),
		)?),
		Arc::new(CredentialAcquisitionLoginEngine::new(
			AuthMethod::ApplicationDefault,
			Str::from("application-default"),
			catalog.clone(),
			credentials.clone(),
			accounts.clone(),
		)?),
		Arc::new(CredentialAcquisitionLoginEngine::new(
			AuthMethod::AwsCredentialChain,
			Str::from("aws-credential-chain"),
			catalog.clone(),
			credentials.clone(),
			accounts.clone(),
		)?),
		Arc::new(OAuthLoginEngine::new(
			AuthMethod::OAuthPkce,
			catalog.clone(),
			credential_store.clone(),
			accounts.clone(),
			oauth_http.clone(),
			oauth_clock.clone(),
			oauth_custom.clone(),
		)?),
		Arc::new(OAuthLoginEngine::new(
			AuthMethod::OAuthDevice,
			catalog.clone(),
			credential_store.clone(),
			accounts.clone(),
			oauth_http.clone(),
			oauth_clock.clone(),
			oauth_custom,
		)?),
	];
	let refresh = Arc::new(StoredOAuthRefreshEngine::new(
		catalog.clone(),
		credential_store.clone(),
		accounts.clone(),
		oauth_http,
		oauth_clock,
		refresh_coordinator,
	));
	let auth_manager = AuthManager::new(
		catalog.clone(),
		credential_store,
		credentials.clone(),
		accounts.clone(),
		login_engines,
		refresh,
	)?;
	let sessions = ConversationSessionPlanner::open(&database, catalog.clone())?;
	let auth_application = AuthApplicationConfig { signing_regions: Arc::new(BTreeMap::new()) };
	let google_cca = GoogleCcaConfig {
		gemini_cli_platform: Str::from(std::env::consts::OS),
		gemini_cli_arch:     Str::from(std::env::consts::ARCH),
		antigravity_headers: CcaHeaders::antigravity(
			std::env::consts::OS,
			std::env::consts::ARCH,
			false,
			None,
		),
		antigravity_policy:  AntigravityPolicy::default(),
	};
	let dependencies = ProductionDependencies::new(
		credentials,
		auth_manager,
		accounts,
		sessions,
		WebSocketTransport::new(),
		google_cca,
		HttpTransport::new(),
		auth_application,
		AdmissionController::new(32, 128),
		Duration::from_secs(60),
		Arc::new(BTreeMap::new()),
	);
	#[cfg(feature = "local-applefm")]
	let dependencies = {
		use omp_llm_inference::local::applefm::{AppleFmCodec, AppleFmTransport, FRAMEWORK_TIMEOUT};
		match AppleFmTransport::new() {
			Ok(transport) => {
				let backend =
					LocalRouteBackend::new(Arc::new(AppleFmCodec), transport, FRAMEWORK_TIMEOUT);
				dependencies.with_local_routes(
					apple_routes
						.into_iter()
						.map(|route| (route, backend.clone())),
				)
			},
			Err(evidence) => {
				let reason = ReasonId(Str::from(evidence.state.code()));
				dependencies.with_local_unavailable(
					apple_routes
						.into_iter()
						.map(|route| (route, reason.clone())),
				)
			},
		}
	};
	Ok(Registry::builder(catalog)
		.with_builtins(BuiltinConfig::production(dependencies))?
		.build()?)
}

#[derive(Clone, Copy)]
struct TracingObservation;

impl Observer for TracingObservation {
	fn started(&self, event: ExecutionStarted) {
		tracing::debug!(execution = ?event, "inference execution started");
	}

	fn finished(&self, event: ExecutionFinished) {
		tracing::debug!(execution = ?event, "inference execution finished");
	}
}

/// Running comprehensive inference registry.
pub struct DaemonHandle {
	readiness: DaemonReadiness,
	registry:  Registry,
	shutdown:  watch::Sender<bool>,
	rpc_task:  JoinHandle<Result<(), tonic::transport::Error>>,
}

impl DaemonHandle {
	/// Loads the immutable catalog and constructs every built-in route service.
	pub async fn start(config: DaemonConfig) -> Result<Self, DaemonError> {
		let data_dir = config.data_dir.ok_or(DaemonError::MissingDataDirectory)?;
		std::fs::create_dir_all(&data_dir).map_err(DaemonError::PrepareState)?;
		let credential_store = open_credential_store(data_dir.join("credentials.db"))?;
		let registry = production_registry(&data_dir, credential_store).await?;
		let routes = registry
			.catalog()
			.routes()
			.iter()
			.filter(|route| registry.contains_service(&route.id))
			.count();
		let incoming = omp_rpc::uds::listen(config.endpoint.as_path())
			.await
			.map_err(DaemonError::RpcListen)?;
		let (shutdown, mut rpc_shutdown) = watch::channel(false);
		let blobs = Arc::new(BlobStore::open(&data_dir)?);
		let inference = InferenceServer::new(InferenceRpc::new(registry.clone()));
		let auth = AuthServer::new(AuthRpc::new(registry.clone()));
		let blobs = BlobServer::new(BlobRpc::new(blobs));
		let rpc_task = tokio::spawn(async move {
			Server::builder()
				.add_service(inference)
				.add_service(blobs)
				.add_service(auth)
				.serve_with_incoming_shutdown(incoming, async move {
					while !*rpc_shutdown.borrow() && rpc_shutdown.changed().await.is_ok() {}
				})
				.await
		});
		Ok(Self {
			readiness: DaemonReadiness { endpoint: config.endpoint, routes },
			registry,
			shutdown,
			rpc_task,
		})
	}

	/// Returns registry readiness facts.
	#[must_use]
	pub const fn readiness(&self) -> &DaemonReadiness {
		&self.readiness
	}

	/// Returns a clone-cheap comprehensive operation service.
	#[must_use]
	pub fn service(&self) -> ProviderService {
		self.registry.service_with_observer(TracingObservation)
	}

	/// Creates a typed client using caller-provided call metadata.
	#[must_use]
	pub fn client(&self, meta: omp_llm_inference::CallMeta) -> Client<ProviderService, Router> {
		Client::new(self.service(), Router::new(self.registry.clone(), Duration::from_secs(30)), meta)
	}

	/// Waits for process shutdown and then signals daemon-owned tasks.
	pub async fn wait(mut self) -> Result<(), DaemonError> {
		tokio::select! {
			signal = shutdown_signal() => signal.map_err(DaemonError::Signal)?,
			result = &mut self.rpc_task => {
				result.map_err(DaemonError::RpcTask)?.map_err(DaemonError::RpcServe)?;
				return Err(DaemonError::RpcStopped);
			},
		}
		self.finish_shutdown().await
	}

	/// Initiates graceful shutdown.
	pub async fn shutdown(self) -> Result<(), DaemonError> {
		self.finish_shutdown().await
	}

	async fn finish_shutdown(mut self) -> Result<(), DaemonError> {
		let _ = self.shutdown.send(true);
		(&mut self.rpc_task)
			.await
			.map_err(DaemonError::RpcTask)?
			.map_err(DaemonError::RpcServe)?;
		#[cfg(unix)]
		match tokio::fs::remove_file(self.readiness.endpoint.as_path()).await {
			Ok(()) => {},
			Err(error) if error.kind() == std::io::ErrorKind::NotFound => {},
			Err(error) => return Err(DaemonError::PrepareState(error)),
		}

		Ok(())
	}
}
#[cfg(test)]
mod credential_key_mode_tests {
	use std::ffi::OsStr;

	use super::CredentialKeyMode;

	#[test]
	fn only_exact_one_opts_into_os_keychain() {
		assert_eq!(
			CredentialKeyMode::from_value(Some(OsStr::new("1"))),
			CredentialKeyMode::OsKeychain
		);
		for value in [None, Some(OsStr::new("")), Some(OsStr::new("true")), Some(OsStr::new("0"))] {
			assert_eq!(CredentialKeyMode::from_value(value), CredentialKeyMode::Unavailable);
		}
	}
}

#[cfg(all(test, feature = "local-applefm"))]
mod tests {
	use super::*;

	#[tokio::test]
	async fn every_catalog_apple_route_has_backend_or_unavailability_evidence() {
		let state = tempfile::tempdir().expect("temporary daemon state");
		let store = open_credential_store_with_key_source(
			state.path().join("credentials.db"),
			Arc::new(omp_llm_inference::auth::HeadlessKeySource::new(
				omp_llm_inference::auth::KeyId::new("apple-route-test"),
				[0x34; 32],
			)),
		)
		.expect("credential store");
		let registry = production_registry(state.path(), store)
			.await
			.expect("production registry");
		for route in registry.catalog().routes().iter().filter(|route| {
			route.codec_profile == omp_llm_catalog::CodecProfile::AppleFm
				&& route.transport == omp_llm_catalog::TransportKind::Local
		}) {
			assert!(
				registry.contains_service(&route.id) || registry.unavailability(&route.id).is_some(),
				"Apple route {} lacks a backend and typed unavailability",
				route.id
			);
		}
	}
}

#[cfg(unix)]
async fn shutdown_signal() -> Result<(), std::io::Error> {
	use tokio::signal::unix::{SignalKind, signal};
	let mut terminate = signal(SignalKind::terminate())?;
	tokio::select! { result = tokio::signal::ctrl_c() => result, _ = terminate.recv() => Ok(()) }
}

#[cfg(windows)]
async fn shutdown_signal() -> Result<(), std::io::Error> {
	tokio::signal::ctrl_c().await
}
