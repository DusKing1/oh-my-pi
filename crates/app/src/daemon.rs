//! Production inference daemon assembly and listener lifecycle.

use std::{
	collections::BTreeMap,
	future::Future,
	path::{Path, PathBuf},
	sync::Arc,
	time::{Duration, SystemTime, UNIX_EPOCH},
};

use futures::{StreamExt as _, future::join_all, stream::FuturesUnordered};
use omp_core::Str;
use omp_llm_broker::{
	BrokerCliBackend,
	oauth::{OAuthEngine, OAuthError},
	service::AuthService,
	source::{BrokerCredentialSource, SpecializedCredentialAuth},
	store::{ClientUsage, CredentialFilter, CredentialKind, CredentialState, Store, StoreError},
	usage::{BrokerObserver, UsageHttp, UsageManager},
};
use omp_llm_catalog::{
	models::{Availability, embedded_catalog},
	overlay::{OverlayError, load_with_overlays},
	provider::{AuthSpec, Facet, ProviderCatalog, ProviderEntry, RegistryMapping, TransportId},
	registry::{CredentialView, Registry},
};
use omp_llm_cursor::CursorChat;
use omp_llm_devin::DevinChat;
use omp_llm_egress::{
	auth_inject::AuthInjectLayer,
	client::EgressClient,
	limits::{KeyedLimitsLayer, LimitConfig},
};
use omp_llm_error::{BlockTable, Classification, Feature};
use omp_llm_fm::AppleFmChat;
use omp_llm_gateway::{
	audio_backends::{AudioRouteRegistrationError, register_production_audio_routes},
	blob::BlobService,
	context::ContextStore,
	discovery::DiscoveryService,
	facade::{FacadeAuth, FacadeConfig, FacadeState, Router},
	image_backends::EgressImageBackend,
	images::{ImageRegistry, LeasedImageCredentials},
	inference::InferenceService,
	listener::{ListenerControl, ListenerError, RemoteTls, Services},
	local::LocalEndpoint,
	media::{MediaFacets, RejectingDownloader},
	routes::{RouteRegistrationError, SpecializedChats, register_production_routes},
	search::SearchRegistry,
	turn::{ChatResolver, RoutedChat, TurnEngine},
	videos::{OpenAiVideoBackend, VideoCredentialLeases, VideoError, VideoInitError},
};
use omp_llm_google::adc::{AdcEngine, AdcError, AdcIntoError, AdcRoute, AdcTokenSink};
use omp_llm_tower::{
	cache::CachePolicy,
	codex_websocket::CodexWebSocketEgress,
	learn::RequestRepair,
	preflight::{Admission, UsageOracle},
	provider::ProviderRoute,
	refresh::{CredentialRefresher as RouteCredentialRefresher, RefreshFailure},
	select::{CredentialCandidates, CredentialPool, LeaseSource},
	stack::{
		builder::{RouteDependencies, RouteStackConfig},
		meter::UsageObserver,
		routing::{
			CountRouter, EmbedRouter, RemoteCountBuildError, RemoteEmbedBuildError,
			remote_count_route, remote_embed_route,
		},
	},
	tap::FrameSink,
};
use omp_llm_types::facet::{Chat, Facets, VideoGen};
use omp_proto::inference::v1::{TurnEvent, TurnRequest};
use omp_rpc::HelloService;
use omp_storage::blob::BlobStore;
use parking_lot::Mutex;
use tokio::task::JoinHandle;
use tower::Layer as _;
use zeroize::Zeroizing;

use crate::auth_backend::{BrokerDiscoveryHttp, BrokerHttp};

const FIRST_BYTE_TIMEOUT: Duration = Duration::from_secs(30);
const GATEWAY_TOKEN_ENV: &str = "OMP_GATEWAY_TOKEN";
const DATA_DIR_ENV: &str = "OMP_DATA_DIR";
const APPLE_PROVIDER_ID: &str = "apple-intelligence";

/// One listener requested from the production daemon.
#[derive(Clone)]
pub enum DaemonListener {
	/// Platform-native local transport: an owner-only Unix-domain socket or a
	/// local-user-only Windows named pipe.
	Local(LocalEndpoint),
	/// TLS TCP listener protected by the authentication policy in [`RemoteTls`].
	Tcp {
		/// Address to bind. Port zero requests an operating-system-selected port.
		addr: std::net::SocketAddr,
		/// Server identity plus mandatory bearer or client-certificate policy.
		tls:  RemoteTls,
	},
}

/// Production daemon construction options.
///
/// Use [`DaemonConfig::local`] for the normal `omp serve --endpoint` path.
/// Builder methods permit an embedding host to add authenticated TCP listeners
/// and specialized transports without introducing a second daemon
/// implementation.
pub struct DaemonConfig {
	data_dir:           Option<PathBuf>,
	project_dir:        PathBuf,
	gateway_token:      Option<Zeroizing<String>>,
	listeners:          Vec<DaemonListener>,
	first_byte_timeout: Duration,
	specialized:        SpecializedChats,
	facets:             Facets,
}

impl DaemonConfig {
	/// Creates the standard owner-only platform-local daemon configuration.
	#[must_use]
	pub fn local(endpoint: impl Into<LocalEndpoint>) -> Self {
		let data_dir = std::env::var_os(DATA_DIR_ENV)
			.map(PathBuf::from)
			.or_else(|| {
				std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share/omp"))
			});
		Self {
			data_dir,
			project_dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
			gateway_token: std::env::var(GATEWAY_TOKEN_ENV)
				.ok()
				.filter(|token| !token.is_empty())
				.map(Zeroizing::new),
			listeners: vec![DaemonListener::Local(endpoint.into())],
			first_byte_timeout: FIRST_BYTE_TIMEOUT,
			specialized: SpecializedChats::default(),
			facets: Facets::default(),
		}
	}

	/// Creates a daemon with one authenticated TLS TCP listener.
	#[must_use]
	pub fn remote(addr: std::net::SocketAddr, tls: RemoteTls) -> Self {
		let mut config = Self::local(PathBuf::new());
		config.listeners = vec![DaemonListener::Tcp { addr, tls }];
		config
	}

	/// Overrides the directory containing `broker.db` and the blob store.
	#[must_use]
	pub fn with_data_dir(mut self, data_dir: PathBuf) -> Self {
		self.data_dir = Some(data_dir);
		self
	}

	/// Overrides the project directory used for `.omp/providers.toml` overlays.
	#[must_use]
	pub fn with_project_dir(mut self, project_dir: PathBuf) -> Self {
		self.project_dir = project_dir;
		self
	}

	/// Sets the client-to-gateway bearer used by foreign HTTP facades.
	#[must_use]
	pub fn with_gateway_token(mut self, token: impl Into<String>) -> Self {
		self.gateway_token = Some(Zeroizing::new(token.into()));
		self
	}

	/// Adds another local or authenticated remote listener.
	#[must_use]
	pub fn with_listener(mut self, listener: DaemonListener) -> Self {
		self.listeners.push(listener);
		self
	}

	/// Installs real implementations for non-HTTP chat transports.
	#[must_use]
	pub fn with_specialized_chats(mut self, specialized: SpecializedChats) -> Self {
		self.specialized = specialized;
		self
	}

	/// Installs a fully assembled GitLab Duo Workflow transport in the
	/// production route registry. The chat owns its lease-backed authentication
	/// boundary; this API never accepts or returns a token.
	#[must_use]
	pub fn with_gitlab_duo_workflow(mut self, chat: omp_llm_gitlab::GitLabDuoChat) -> Self {
		self.specialized.gitlab_duo_workflow = Some(Arc::new(chat));
		self
	}

	/// Installs production implementations for non-chat inference facets.
	#[must_use]
	pub fn with_facets(mut self, facets: Facets) -> Self {
		self.facets = facets;
		self
	}
}

/// Listener addresses and route count made ready by daemon startup.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DaemonReadiness {
	/// Bound platform-native local endpoints.
	pub local_endpoints: Vec<LocalEndpoint>,
	/// Effective bound TCP addresses.
	pub tcp_addresses:   Vec<std::net::SocketAddr>,
	/// Catalog-driven chat routes registered before listener binding.
	pub chat_routes:     usize,
}

/// A production daemon startup or serving failure.
#[derive(Debug, thiserror::Error)]
pub enum DaemonError {
	/// Neither an explicit data directory nor `OMP_DATA_DIR`/`HOME` was
	/// available.
	#[error("daemon data directory is unavailable; set OMP_DATA_DIR or HOME")]
	MissingDataDirectory,
	/// No listener was requested.
	#[error("daemon requires at least one listener")]
	MissingListener,
	/// Foreign facade authentication was not configured.
	#[error("daemon requires a non-empty OMP_GATEWAY_TOKEN")]
	MissingGatewayAuthentication,
	/// The bundled model catalog could not be loaded.
	#[error("bundled model catalog is empty")]
	EmptyModelCatalog,
	/// Provider overlays produced no provider rows.
	#[error("provider catalog is empty")]
	EmptyProviderCatalog,
	/// Durable daemon state could not be prepared.
	#[error("could not prepare daemon state at {path}: {source}")]
	PrepareState {
		/// Path whose directory could not be created.
		path:   PathBuf,
		/// Filesystem failure.
		#[source]
		source: std::io::Error,
	},
	/// The broker store could not be opened or queried.
	#[error(transparent)]
	Store(#[from] StoreError),
	/// Provider catalog overlays could not be loaded.
	#[error(transparent)]
	Providers(#[from] OverlayError),
	/// OAuth configuration could not be constructed.
	#[error(transparent)]
	OAuth(#[from] OAuthError),
	/// Google ADC discovery or route resolution failed.
	#[error(transparent)]
	Adc(#[from] AdcError),
	/// A minted ADC token could not be persisted at the one-way broker ingress.
	#[error(transparent)]
	AdcStore(#[from] AdcIntoError<StoreError>),
	/// Catalog routes could not be assembled.
	#[error(transparent)]
	Routes(#[from] RouteRegistrationError),
	/// A catalog specialized transport endpoint was invalid.
	#[error("chat provider `{provider}` has an invalid specialized endpoint: {source}")]
	SpecializedEndpoint {
		/// Catalog provider identifier.
		provider: Str,
		/// Endpoint parsing failure.
		#[source]
		source:   tonic::transport::Error,
	},
	/// Advertised audio routes could not be assembled.
	#[error(transparent)]
	AudioRoutes(#[from] AudioRouteRegistrationError),
	/// Advertised embedding routes could not be assembled.
	#[error(transparent)]
	EmbeddingRoute(#[from] RemoteEmbedBuildError),
	/// Advertised provider token-count routes could not be assembled.
	#[error(transparent)]
	CountRoute(#[from] RemoteCountBuildError),
	/// Durable video state could not be initialized.
	#[error(transparent)]
	Video(#[from] VideoInitError),
	/// The durable blob store could not be opened.
	#[error(transparent)]
	Storage(#[from] omp_storage::blob::Error),
	/// A listener could not bind or serve.
	#[error(transparent)]
	Listener(#[from] ListenerError),
	/// A listener task panicked or was cancelled.
	#[error("daemon listener task failed: {0}")]
	Task(#[from] tokio::task::JoinError),
	/// Process signal handling could not be installed.
	#[error("could not install shutdown signal handler: {0}")]
	Signal(#[from] std::io::Error),
}

/// Running daemon ownership, readiness, and graceful-shutdown control.

/// Prompt-cache policy for one transport.
///
/// `TailTwo` placement and short retention are Anthropic's breakpoint
/// semantics, measured cheapest over a 1.2k-session replay. Other transports
/// read the same [`omp_proto::inference::v1::CacheHint`] fields differently —
/// the `OpenAI` Responses dialect projects `session_key` into
/// `prompt_cache_key` — so they keep the inert default and behave exactly as
/// before.
///
/// Keep-alive refreshes stay off. They are worth about a thirtieth of what
/// placement is worth, and they are only cheap if dropping the response stream
/// closes the upstream request, which nothing here proves. Enable with
/// `.with_keepalive(3)` once a recording proxy has shown the abort lands.
fn cache_policy(transport: TransportId) -> CachePolicy {
	match transport {
		TransportId::AnthropicMessages => CachePolicy::tail_two(),
		_ => CachePolicy::default(),
	}
}

/// Running daemon listeners and their shared broker state.
///
/// Dropping the handle releases listener controls and background tasks.
pub struct DaemonHandle {
	readiness: DaemonReadiness,
	controls:  Vec<ListenerControl>,
	tasks:     Vec<JoinHandle<Result<(), ListenerError>>>,
	state:     Arc<DaemonState>,
}

struct DaemonState {
	store:      Arc<Store>,
	oauth:      Arc<OAuthEngine>,
	usage_http: Arc<dyn UsageHttp>,
	observer:   BrokerObserver,
}

impl DaemonHandle {
	/// Builds the complete inference runtime, binds every requested listener,
	/// and returns only after all reported endpoints are ready.
	///
	/// Provider-owned discovery protocols are registered here as the single
	/// application wiring point for new discovery transports.
	pub async fn start(config: DaemonConfig) -> Result<Self, DaemonError> {
		if config.listeners.is_empty() {
			return Err(DaemonError::MissingListener);
		}
		let data_dir = config.data_dir.ok_or(DaemonError::MissingDataDirectory)?;
		let token = config
			.gateway_token
			.filter(|token| !token.is_empty())
			.ok_or(DaemonError::MissingGatewayAuthentication)?;
		prepare_dir(&data_dir)?;

		let providers = Arc::new(load_with_overlays(&config.project_dir)?);
		if providers.is_empty() {
			return Err(DaemonError::EmptyProviderCatalog);
		}
		let models = embedded_catalog();
		if models.is_empty() {
			return Err(DaemonError::EmptyModelCatalog);
		}

		let store = Arc::new(Store::open(data_dir.join("broker.db"))?);
		bootstrap_catalog_credentials(&store, &providers)?;
		let client = Arc::new(EgressClient::new(config.first_byte_timeout));
		let adc = Arc::new(AdcEngine::new(client.as_ref().clone()));
		let adc_routes = bootstrap_adc(&adc, &store, &providers).await?;
		let broker_http = Arc::new(BrokerHttp::with_client(client.as_ref().clone()));
		let oauth = Arc::new(OAuthEngine::new(Arc::clone(&store), broker_http.clone())?);
		let usage_http: Arc<dyn UsageHttp> = broker_http;
		let observer = BrokerObserver::new(Arc::clone(&store));
		let state = Arc::new(DaemonState {
			store:      Arc::clone(&store),
			oauth:      Arc::clone(&oauth),
			usage_http: Arc::clone(&usage_http),
			observer:   observer.clone(),
		});
		let apple_chat = match AppleFmChat::load().await {
			Ok(chat) => Some(Arc::new(chat) as Arc<dyn Chat>),
			Err(error) => {
				tracing::debug!(%error, "Apple Intelligence is unavailable; embedded route disabled");
				None
			},
		};
		let apple_fm_available = apple_chat.is_some();

		let credential_view: Arc<dyn CredentialView> = Arc::new(BrokerAvailability {
			store: Arc::clone(&store),
			providers: Arc::clone(&providers),
			apple_fm_available,
		});
		let model_catalog = Arc::new(models.clone());
		let premium_multipliers = Arc::new(
			model_catalog
				.models()
				.iter()
				.filter_map(|card| {
					card.behavior.premium_multiplier.as_ref().map(|multiplier| {
						((card.provider.clone(), card.model.clone()), multiplier.millionths)
					})
				})
				.collect::<BTreeMap<_, _>>(),
		);
		let resolver_registry =
			Arc::new(parking_lot::RwLock::new(Registry::new(models, Arc::clone(&credential_view))));
		let resolver = Arc::new(ChatResolver::new(Arc::clone(&resolver_registry)));
		let credential_source = BrokerCredentialSource::new(
			Arc::clone(&store),
			Arc::clone(&providers),
			Arc::clone(&oauth),
		);
		let limits = KeyedLimitsLayer::with_block_sink(LimitConfig::default(), observer.clone())
			.layer(client.as_ref().clone());
		let egress = AuthInjectLayer::new(credential_source.clone()).layer(limits);
		let chat_egress = CodexWebSocketEgress::new(egress.clone(), credential_source.clone());
		let discovery = crate::discovery::register(Arc::new(BrokerDiscoveryHttp::new(
			egress.clone(),
			Arc::clone(&store),
			credential_source.clone(),
		)));
		let route_providers = usable_providers(&providers, &store, &adc_routes, apple_fm_available);
		let mut specialized = config.specialized;
		mount_apple_chat(&mut specialized, apple_chat);
		for provider in &route_providers {
			match provider.transport {
				TransportId::Cursor
					if specialized.cursor.is_none()
						&& !specialized.by_provider.contains_key(&provider.id) =>
				{
					let auth =
						SpecializedCredentialAuth::new(credential_source.clone(), provider.id.clone());
					specialized.by_provider.insert(
						provider.id.clone(),
						Arc::new(CursorChat::new(provider.base_url.clone(), auth)),
					);
				},
				TransportId::Devin
					if specialized.devin.is_none()
						&& !specialized.by_provider.contains_key(&provider.id) =>
				{
					let auth =
						SpecializedCredentialAuth::new(credential_source.clone(), provider.id.clone());
					let endpoint = tonic::transport::Endpoint::from_shared(
						provider.base_url.as_str().trim_end_matches('/').to_owned(),
					)
					.map_err(|source| DaemonError::SpecializedEndpoint {
						provider: provider.id.clone(),
						source,
					})?;
					specialized.by_provider.insert(
						provider.id.clone(),
						Arc::new(DevinChat::new(endpoint.connect_lazy(), auth)),
					);
				},
				_ => {},
			}
		}
		let blocks = Arc::new(Mutex::new(BlockTable::default()));
		let audio = register_production_audio_routes(
			providers
				.values()
				.filter(|provider| provider_is_routable(provider, &store, &adc_routes)),
			egress.clone(),
			|provider| provider_route(provider, &adc_routes),
		)?;
		let mut embed = EmbedRouter::new(Arc::clone(&model_catalog), Arc::clone(&providers));
		let mut has_embed_route = false;
		for provider in providers.values().filter(|provider| {
			provider.facets.contains(&Facet::Embeddings)
				&& provider_is_routable(provider, &store, &adc_routes)
		}) {
			let route = remote_embed_route(
				provider.clone(),
				provider_route(provider, &adc_routes),
				egress.clone(),
			)?;
			embed.insert_route(provider.id.clone(), route);
			has_embed_route = true;
		}
		let mut count_tokens = CountRouter::new(Arc::clone(&model_catalog), Arc::clone(&providers));
		for provider in providers.values().filter(|provider| {
			provider.transport == TransportId::AnthropicMessages
				&& provider_is_routable(provider, &store, &adc_routes)
		}) {
			let route = remote_count_route(
				provider.clone(),
				provider_route(provider, &adc_routes),
				egress.clone(),
			)?;
			count_tokens.insert_provider_endpoint(provider.id.clone(), route);
		}
		let dependency_store = Arc::clone(&store);
		let dependency_oauth = Arc::clone(&oauth);
		let dependency_blocks = Arc::clone(&blocks);
		let dependency_adc = Arc::clone(&adc);
		let dependency_adc_projects = Arc::new(
			adc_routes
				.iter()
				.map(|(provider, route)| (provider.clone(), route.project.clone()))
				.collect::<BTreeMap<_, _>>(),
		);
		let registration_adc_routes = adc_routes.clone();
		let dependency_observer = observer.clone();
		let dependency_premium_multipliers = Arc::clone(&premium_multipliers);
		let registration = register_production_routes(
			&resolver,
			route_providers,
			chat_egress,
			move |provider| {
				route_dependencies(
					provider,
					Arc::clone(&dependency_store),
					Arc::clone(&dependency_oauth),
					Arc::clone(&dependency_adc),
					dependency_adc_projects.get(provider.id.as_str()).cloned(),
					Arc::clone(&dependency_blocks),
					dependency_observer.clone(),
					Arc::clone(&dependency_premium_multipliers),
				)
			},
			|provider| RouteStackConfig {
				cache: cache_policy(provider.transport),
				..RouteStackConfig::default()
			},
			move |provider| provider_route(provider, &registration_adc_routes),
			specialized,
		)?;

		let blobs = Arc::new(BlobStore::open(data_dir.join("blobs"))?);
		let mut facets = config.facets;
		facets.chat = Some(Arc::new(RoutedChat::new(Arc::clone(&resolver))));
		if facets.embed.is_none() && has_embed_route {
			facets.embed = Some(Arc::new(embed));
		}
		if facets.count_tokens.is_none() {
			facets.count_tokens = Some(Arc::new(count_tokens));
		}
		if facets.speak.is_none() {
			facets.speak = audio.speak;
		}
		if facets.transcribe.is_none() {
			facets.transcribe = audio.transcribe;
		}
		if facets.search.is_none() {
			facets.search = Some(Arc::new(SearchRegistry::production(
				credential_source.clone(),
				Arc::clone(&client),
			)));
		}
		if facets.image_gen.is_none()
			&& providers.values().any(|provider| {
				provider.facets.contains(&Facet::ImageGeneration)
					&& provider_is_routable(provider, &store, &adc_routes)
			}) {
			let credentials = Arc::new(LeasedImageCredentials::new(credential_source.clone()));
			let backend = Arc::new(EgressImageBackend::new(egress.clone(), Arc::clone(&providers)));
			facets.image_gen = Some(Arc::new(ImageRegistry::new(credentials, backend)));
		}
		if facets.video_gen.is_none()
			&& let Some(openai) = providers.get("openai").filter(|provider| {
				provider.facets.contains(&Facet::VideoGeneration)
					&& provider_is_routable(provider, &store, &adc_routes)
			}) {
			let leases = Arc::new(BrokerVideoLeases { store: Arc::clone(&store) });
			let video = OpenAiVideoBackend::new(
				egress,
				leases,
				Arc::clone(&blobs),
				data_dir.join("video-jobs"),
				Some(openai.base_url.as_str()),
			)?;
			facets.video_gen = Some(Arc::new(video) as Arc<dyn VideoGen>);
		}
		let facets = Arc::new(facets);
		let contexts = Arc::new(ContextStore::default());
		let turn = TurnEngine::new(Arc::clone(&contexts), resolver);
		let discovery = DiscoveryService::from_shared(
			Arc::clone(&resolver_registry),
			Arc::clone(&providers),
			discovery,
		);
		let media = MediaFacets::from_facets(
			Arc::clone(&blobs),
			facets.as_ref(),
			Arc::new(RejectingDownloader),
		);
		let inference = InferenceService::new(turn, contexts, discovery, Arc::clone(&facets), media);
		let facade = Router::new(Arc::new(FacadeState {
			facets,
			registry: resolver_registry,
			blobs: Arc::clone(&blobs),
			auth: FacadeAuth::new(token.as_str()),
			config: FacadeConfig::default(),
		}));
		let mut capabilities = vec!["auth".into(), "blob.v1".into(), "foreign-facades".into()];
		capabilities.extend(inference.capabilities());
		let hello = HelloService::new(env!("CARGO_PKG_VERSION"), capabilities);

		let mut readiness = DaemonReadiness {
			local_endpoints: Vec::new(),
			tcp_addresses:   Vec::new(),
			chat_routes:     registration.registered,
		};
		let mut bound = Vec::with_capacity(config.listeners.len());
		for requested in config.listeners {
			match requested {
				DaemonListener::Local(endpoint) => {
					let listener =
						omp_llm_gateway::listener::LocalListener::bind(endpoint.as_path()).await?;
					readiness.local_endpoints.push(endpoint);
					bound.push(BoundListener::Local(listener));
				},
				DaemonListener::Tcp { addr, tls } => {
					let listener = omp_llm_gateway::listener::RemoteListener::bind(addr, tls).await?;
					readiness.tcp_addresses.push(listener.local_addr());
					bound.push(BoundListener::Tcp(listener));
				},
			}
		}

		let mut controls = Vec::with_capacity(bound.len());
		let mut tasks = Vec::with_capacity(bound.len());
		for listener in bound {
			let auth = AuthService::with_oauth(
				Arc::clone(&store),
				Arc::clone(&oauth),
				Arc::clone(&usage_http),
			);
			let blob = BlobService::new(Arc::clone(&blobs));
			let services = Services::new(inference.clone(), auth, blob, facade.clone(), hello.clone());
			controls.push(services.control());
			tasks.push(match listener {
				BoundListener::Local(listener) => tokio::spawn(listener.serve(services)),
				BoundListener::Tcp(listener) => tokio::spawn(listener.serve(services)),
			});
		}

		Ok(Self { readiness, controls, tasks, state })
	}

	/// Returns endpoints that were bound before startup completed.
	#[must_use]
	pub const fn readiness(&self) -> &DaemonReadiness {
		&self.readiness
	}

	/// Builds an in-process auth CLI backend over this daemon's exact broker
	/// store, OAuth engine, and usage transport.
	#[must_use]
	pub fn broker_backend(&self) -> BrokerCliBackend {
		BrokerCliBackend::with_shared_oauth(
			Arc::clone(&self.state.store),
			Some(Arc::clone(&self.state.oauth)),
			UsageManager::new(Arc::clone(&self.state.store), Arc::clone(&self.state.usage_http)),
		)
	}

	/// Returns the observer shared by egress blocking and broker persistence.
	#[must_use]
	pub fn usage_observer(&self) -> BrokerObserver {
		self.state.observer.clone()
	}

	/// Waits for SIGINT/SIGTERM or listener completion, then drains every
	/// listener and removes bound platform-local endpoints.
	pub async fn wait(mut self) -> Result<(), DaemonError> {
		let signal = shutdown_signal();
		tokio::pin!(signal);
		let mut tasks: FuturesUnordered<_> = std::mem::take(&mut self.tasks).into_iter().collect();
		loop {
			tokio::select! {
				signal = &mut signal => {
					signal?;
					self.begin_shutdown();
					return drain_listener_tasks(&mut tasks).await;
				},
				result = tasks.next() => {
					let Some(result) = result else {
						return Ok(());
					};
					match result? {
						Ok(()) => {
							self.begin_shutdown();
							return drain_listener_tasks(&mut tasks).await;
						},
						Err(error) => {
							self.begin_shutdown();
							let _ = drain_listener_tasks(&mut tasks).await;
							return Err(error.into());
						},
					}
				},
			}
		}
	}

	/// Initiates graceful shutdown and waits for every active response body to
	/// drain before returning.
	pub async fn shutdown(mut self) -> Result<(), DaemonError> {
		self.begin_shutdown();
		join_listener_tasks(std::mem::take(&mut self.tasks)).await
	}

	fn begin_shutdown(&self) {
		for control in &self.controls {
			control.shutdown();
		}
	}
}

impl Drop for DaemonHandle {
	fn drop(&mut self) {
		self.begin_shutdown();
	}
}

enum BoundListener {
	Local(omp_llm_gateway::listener::LocalListener),
	Tcp(omp_llm_gateway::listener::RemoteListener),
}

fn prepare_dir(path: &Path) -> Result<(), DaemonError> {
	std::fs::create_dir_all(path)
		.map_err(|source| DaemonError::PrepareState { path: path.to_owned(), source })
}

fn usable_providers<'a>(
	providers: &'a ProviderCatalog,
	store: &Store,
	adc_routes: &BTreeMap<Str, AdcRoute>,
	apple_fm_available: bool,
) -> Vec<&'a ProviderEntry> {
	providers
		.values()
		.filter(|provider| provider.facets.contains(&Facet::Chat))
		.filter(|provider| runtime_provider_is_usable(provider, apple_fm_available))
		.filter(|provider| provider_is_routable(provider, store, adc_routes))
		.collect()
}

fn mount_apple_chat(specialized: &mut SpecializedChats, chat: Option<Arc<dyn Chat>>) -> bool {
	let available = chat.is_some();
	if specialized.embedded.is_none() {
		specialized.embedded = chat;
	}
	available
}

fn runtime_provider_is_usable(provider: &ProviderEntry, apple_fm_available: bool) -> bool {
	provider.id.as_str() != APPLE_PROVIDER_ID || apple_fm_available
}

fn provider_is_configured(provider: &ProviderEntry, store: &Store) -> bool {
	if !matches!(provider.transport, TransportId::Cursor | TransportId::Devin)
		&& matches!(&provider.auth, AuthSpec::None | AuthSpec::OptionalBearer { .. })
	{
		return true;
	}
	let states = [CredentialState::Active];
	store
		.list_credentials(&CredentialFilter {
			provider: Some(provider.id.as_str()),
			states:   &states,
			now_ms:   now_ms(),
		})
		.is_ok_and(|credentials| !credentials.is_empty())
}

fn provider_is_routable(
	provider: &ProviderEntry,
	store: &Store,
	adc_routes: &BTreeMap<Str, AdcRoute>,
) -> bool {
	provider_is_configured(provider, store)
		&& provider_route_is_complete(provider, &provider_route(provider, adc_routes))
}

fn provider_route_is_complete(provider: &ProviderEntry, route: &ProviderRoute) -> bool {
	let base_url = provider.base_url.as_str();
	if matches!(provider.transport, TransportId::GoogleVertex | TransportId::AnthropicVertex)
		&& (route.project.is_empty() || route.region.is_empty())
	{
		return false;
	}
	if base_url.contains("{project}") && route.project.is_empty() {
		return false;
	}
	if (base_url.contains("{region}") || base_url.contains("{location}")) && route.region.is_empty()
	{
		return false;
	}
	if base_url.contains("{account}") && route.account.is_empty() {
		return false;
	}
	!base_url.contains("{gateway}") || !route.gateway.is_empty()
}

fn bootstrap_catalog_credentials(
	store: &Store,
	providers: &ProviderCatalog,
) -> Result<(), StoreError> {
	let observed_at_ms = now_ms();
	for provider in providers.values() {
		match &provider.auth {
			AuthSpec::Bearer { env }
			| AuthSpec::OptionalBearer { env }
			| AuthSpec::Header { env, .. }
			| AuthSpec::Query { env, .. } => {
				if let Some(secret) = first_catalog_env(env) {
					store.upsert_api_key(
						provider.id.as_str(),
						"environment",
						secret.as_bytes(),
						observed_at_ms,
					)?;
				}
			},
			AuthSpec::GoogleAdc { api_key_env, .. } => {
				if let Some(secret) = first_catalog_env(api_key_env) {
					store.upsert_api_key(
						provider.id.as_str(),
						"environment-api-key",
						secret.as_bytes(),
						observed_at_ms,
					)?;
				}
			},
			AuthSpec::AwsSigV4 => {
				if let (Ok(access), Ok(secret)) =
					(std::env::var("AWS_ACCESS_KEY_ID"), std::env::var("AWS_SECRET_ACCESS_KEY"))
					&& !access.is_empty()
					&& !secret.is_empty()
				{
					let session = std::env::var("AWS_SESSION_TOKEN")
						.ok()
						.filter(|token| !token.is_empty());
					store.upsert_aws(
						provider.id.as_str(),
						"environment",
						access.as_bytes(),
						secret.as_bytes(),
						session.as_deref().map(str::as_bytes),
						observed_at_ms,
					)?;
				}
			},
			AuthSpec::None | AuthSpec::OAuth { .. } => {},
		}
	}
	Ok(())
}

fn first_catalog_env(names: &[Str]) -> Option<String> {
	names.iter().find_map(|name| {
		std::env::var(name.as_str())
			.ok()
			.filter(|value| !value.is_empty())
	})
}

async fn bootstrap_adc(
	engine: &Arc<AdcEngine<EgressClient>>,
	store: &Arc<Store>,
	providers: &ProviderCatalog,
) -> Result<BTreeMap<Str, AdcRoute>, DaemonError> {
	let mut routes = BTreeMap::new();
	if !adc_source_configured() {
		return Ok(routes);
	}
	for provider in providers.values() {
		let AuthSpec::GoogleAdc { project_env, location_env, .. } = &provider.auth else {
			continue;
		};
		let route = if matches!(
			provider.transport,
			TransportId::GoogleVertex | TransportId::AnthropicVertex
		) {
			let project = first_catalog_env(project_env);
			let location = first_catalog_env(location_env);
			Some(
				engine
					.resolve_route(project.as_deref(), location.as_deref())
					.await?,
			)
		} else {
			None
		};
		let sink = BrokerAdcSink {
			store:    Arc::clone(store),
			provider: provider.id.clone(),
			project:  route.as_ref().map(|route| route.project.clone()),
		};
		engine.authorize_into(&sink).await?;
		if let Some(route) = route {
			routes.insert(provider.id.clone(), route);
		}
	}
	Ok(routes)
}

fn adc_source_configured() -> bool {
	if [
		"GOOGLE_CLOUD_ACCESS_TOKEN",
		"CLOUDSDK_AUTH_ACCESS_TOKEN",
		"GOOGLE_SERVICE_ACCOUNT_JSON",
		"GOOGLE_APPLICATION_CREDENTIALS",
		"GCE_METADATA_HOST",
	]
	.into_iter()
	.any(|name| std::env::var_os(name).is_some_and(|value| !value.is_empty()))
	{
		return true;
	}
	let directory = std::env::var_os("CLOUDSDK_CONFIG")
		.map(PathBuf::from)
		.or_else(|| {
			std::env::var_os("HOME")
				.map(PathBuf::from)
				.map(|home| home.join(".config/gcloud"))
		});
	directory.is_some_and(|path| path.join("application_default_credentials.json").is_file())
}

#[derive(Clone)]
struct BrokerAdcSink {
	store:    Arc<Store>,
	provider: Str,
	project:  Option<String>,
}

impl AdcTokenSink for BrokerAdcSink {
	type Error = StoreError;

	fn accept(&self, token: &[u8], expires_at: SystemTime) -> Result<(), Self::Error> {
		let expires_at_ms = expires_at
			.duration_since(UNIX_EPOCH)
			.unwrap_or_default()
			.as_millis()
			.try_into()
			.unwrap_or(u64::MAX);
		let props = self.project.as_ref().map_or_else(
			|| serde_json::json!({}),
			|project| {
				if self.provider.contains("antigravity") {
					serde_json::json!({"antigravity": {"project_id": project}})
				} else {
					serde_json::json!({"google": {"project_id": project}})
				}
			},
		);
		self
			.store
			.upsert_minted_bearer(
				self.provider.as_str(),
				"application-default",
				token,
				expires_at_ms,
				&props,
				now_ms(),
			)
			.map(|_| ())
	}
}

fn route_dependencies(
	provider: &ProviderEntry,
	store: Arc<Store>,
	oauth: Arc<OAuthEngine>,
	adc: Arc<AdcEngine<EgressClient>>,
	adc_project: Option<String>,
	blocks: Arc<Mutex<BlockTable>>,
	observer: BrokerObserver,
	premium_multipliers: Arc<BTreeMap<(Str, Str), u64>>,
) -> RouteDependencies {
	let provider_id = provider.id.clone();
	let anonymous = matches!(&provider.auth, AuthSpec::None | AuthSpec::OptionalBearer { .. });
	let refresh = if matches!(&provider.auth, AuthSpec::GoogleAdc { .. }) {
		RouteRefresh::Adc {
			engine: adc,
			sink:   BrokerAdcSink {
				store:    Arc::clone(&store),
				provider: provider_id.clone(),
				project:  adc_project,
			},
		}
	} else if matches!(&provider.auth, AuthSpec::OAuth { .. }) || provider.oauth_flow.is_some() {
		RouteRefresh::OAuth(oauth)
	} else {
		RouteRefresh::Static
	};
	RouteDependencies {
		usage: Arc::new(BrokerUsageOracle {
			store: Arc::clone(&store),
			provider: provider_id.clone(),
			anonymous,
		}),
		credentials: Arc::new(BrokerCredentialPool {
			store:    Arc::clone(&store),
			provider: provider_id.clone(),
		}),
		leases: Arc::new(BrokerLeases { store: Arc::clone(&store) }),
		refresher: Arc::new(BrokerRouteRefresher { store, provider: provider_id, refresh }),
		repair: Arc::new(CatalogRepair),
		observer: Arc::new(NoopFrameSink),
		usage_observer: Arc::new(BrokerTerminalUsage { observer, premium_multipliers }),
		blocks,
	}
}

fn provider_route(provider: &ProviderEntry, adc_routes: &BTreeMap<Str, AdcRoute>) -> ProviderRoute {
	let mut route = ProviderRoute {
		project:    env_str(&["GOOGLE_CLOUD_PROJECT", "CLOUDSDK_CORE_PROJECT"]),
		region:     env_str(&[
			"GOOGLE_CLOUD_LOCATION",
			"AZURE_OPENAI_REGION",
			"AWS_REGION",
			"AWS_DEFAULT_REGION",
		]),
		deployment: env_str(&["AZURE_OPENAI_DEPLOYMENT"]),
		account:    env_str(&["CLOUDFLARE_ACCOUNT_ID"]),
		gateway:    env_str(&["CLOUDFLARE_AI_GATEWAY_ID"]),
	};
	if matches!(&provider.auth, AuthSpec::AwsSigV4) {
		route.region = env_str(&["AWS_REGION", "AWS_DEFAULT_REGION"]);
	}
	if let AuthSpec::GoogleAdc { project_env, location_env, .. } = &provider.auth {
		if let Some(project) = first_catalog_env(project_env) {
			route.project = Str::from(project);
		}
		if let Some(location) = first_catalog_env(location_env) {
			route.region = Str::from(location);
		}
	}
	if let Some(adc) = adc_routes.get(provider.id.as_str()) {
		route.project = Str::from(adc.project.as_str());
		route.region = Str::from(adc.location.as_str());
	}
	route
}

fn env_str(names: &[&str]) -> Str {
	names
		.iter()
		.find_map(|name| std::env::var(name).ok().filter(|value| !value.is_empty()))
		.map_or_else(|| Str::new_static(""), Str::from)
}

struct BrokerAvailability {
	store:              Arc<Store>,
	providers:          Arc<ProviderCatalog>,
	apple_fm_available: bool,
}

impl CredentialView for BrokerAvailability {
	fn availability(&self, provider: &str) -> Availability {
		if provider == APPLE_PROVIDER_ID && !self.apple_fm_available {
			return Availability::Disabled;
		}
		let Some(entry) = self.providers.get(provider) else {
			return Availability::Disabled;
		};
		if matches!(&entry.auth, AuthSpec::None | AuthSpec::OptionalBearer { .. }) {
			return Availability::Available;
		}
		let credential_provider = match &entry.mapping {
			RegistryMapping::Alias { target, .. } => target.as_str(),
			_ => provider,
		};
		let Ok(credentials) = self.store.list_credentials(&CredentialFilter {
			provider: Some(credential_provider),
			now_ms: now_ms(),
			..CredentialFilter::default()
		}) else {
			return Availability::Disabled;
		};
		if credentials
			.iter()
			.any(|credential| credential.state == CredentialState::Active)
		{
			Availability::Available
		} else if credentials
			.iter()
			.any(|credential| credential.state == CredentialState::Blocked)
		{
			Availability::Blocked
		} else {
			Availability::LoginRequired
		}
	}

	fn availability_for(&self, provider: &str, account: &str) -> Availability {
		if provider == APPLE_PROVIDER_ID && !self.apple_fm_available {
			return Availability::Disabled;
		}
		let Ok(credential_id) = account.parse::<u64>() else {
			return self.availability(provider);
		};
		let Some(entry) = self.providers.get(provider) else {
			return Availability::Disabled;
		};
		let expected_provider = match &entry.mapping {
			RegistryMapping::Alias { target, .. } => target.as_str(),
			_ => entry.id.as_str(),
		};
		let Ok(Some(credential)) = self.store.get_credential(credential_id, now_ms()) else {
			return Availability::LoginRequired;
		};
		if credential.provider != expected_provider {
			return Availability::LoginRequired;
		}
		match credential.state {
			CredentialState::Active => Availability::Available,
			CredentialState::Blocked => Availability::Blocked,
			_ => Availability::LoginRequired,
		}
	}
}

struct BrokerCredentialPool {
	store:    Arc<Store>,
	provider: Str,
}

impl CredentialPool for BrokerCredentialPool {
	fn candidates(&self, model: &str) -> CredentialCandidates {
		self
			.store
			.ranked_credential_ids(
				self.provider.as_str(),
				(!model.is_empty()).then_some(model),
				None,
				now_ms(),
			)
			.unwrap_or_default()
	}
}

struct BrokerLeases {
	store: Arc<Store>,
}

impl LeaseSource for BrokerLeases {
	fn lease(&self, id: u64) -> Option<omp_llm_egress::auth_inject::CredentialLease> {
		self.store.lease(id).ok().flatten()
	}
}

struct BrokerVideoLeases {
	store: Arc<Store>,
}

impl VideoCredentialLeases for BrokerVideoLeases {
	fn select(&self) -> Result<omp_llm_egress::auth_inject::CredentialLease, VideoError> {
		self
			.store
			.lease_provider("openai", now_ms())
			.map_err(|error| VideoError::Credential(error.to_string().into()))?
			.ok_or_else(|| VideoError::Credential("no active OpenAI credential".into()))
	}

	fn by_id(
		&self,
		credential_id: u64,
	) -> Result<omp_llm_egress::auth_inject::CredentialLease, VideoError> {
		let lease = self
			.store
			.lease(credential_id)
			.map_err(|error| VideoError::Credential(error.to_string().into()))?
			.ok_or_else(|| VideoError::Credential("video credential is unavailable".into()))?;
		if lease.provider() != "openai" {
			return Err(VideoError::Credential("video credential belongs to another provider".into()));
		}
		Ok(lease)
	}
}

struct BrokerUsageOracle {
	store:     Arc<Store>,
	provider:  Str,
	anonymous: bool,
}

impl UsageOracle for BrokerUsageOracle {
	fn admit(&self, _model: &str) -> Admission {
		if self.anonymous {
			return Admission::Allow;
		}
		match self.store.list_credentials(&CredentialFilter {
			provider: Some(self.provider.as_str()),
			now_ms: now_ms(),
			..CredentialFilter::default()
		}) {
			Ok(credentials)
				if credentials
					.iter()
					.any(|item| item.state == CredentialState::Active) =>
			{
				Admission::Allow
			},
			Ok(credentials)
				if credentials
					.iter()
					.any(|item| item.state == CredentialState::Blocked) =>
			{
				let retry_after_ms = credentials
					.iter()
					.flat_map(|item| item.blocks.iter())
					.map(|block| block.until_ms.saturating_sub(now_ms()))
					.min()
					.unwrap_or_default();
				Admission::DenyQuota {
					detail: format!("all {} credentials are temporarily blocked", self.provider),
					retry_after_ms,
				}
			},
			Ok(_) => Admission::DenyAuth {
				detail: format!("provider {} requires a usable credential", self.provider),
			},
			Err(error) => Admission::Unknown { detail: error.to_string() },
		}
	}
}

enum RouteRefresh {
	OAuth(Arc<OAuthEngine>),
	Adc { engine: Arc<AdcEngine<EgressClient>>, sink: BrokerAdcSink },
	Static,
}

struct BrokerRouteRefresher {
	store:    Arc<Store>,
	provider: Str,
	refresh:  RouteRefresh,
}

impl RouteCredentialRefresher for BrokerRouteRefresher {
	fn expires_at_ms(&self) -> Option<u64> {
		self
			.credentials()
			.into_iter()
			.map(|credential| credential.expires_at_ms)
			.find(|expires| *expires != 0)
	}

	fn refresh(
		&self,
		_force: bool,
	) -> std::pin::Pin<Box<dyn Future<Output = Result<(), RefreshFailure>> + Send + '_>> {
		Box::pin(async move {
			match &self.refresh {
				RouteRefresh::OAuth(oauth) => {
					let credential = self
						.credentials()
						.into_iter()
						.find(|credential| credential.kind == CredentialKind::OAuth)
						.ok_or_else(|| RefreshFailure::new("provider has no refreshable credential"))?;
					oauth
						.refresh_credential(credential.id, now_ms())
						.await
						.map(|_| ())
						.map_err(|error| RefreshFailure::new(error.to_string()))
				},
				RouteRefresh::Adc { engine, sink } => engine
					.refresh_into(sink)
					.await
					.map_err(|error| RefreshFailure::new(error.to_string())),
				RouteRefresh::Static => {
					Err(RefreshFailure::new("provider credential is not refreshable"))
				},
			}
		})
	}
}

impl BrokerRouteRefresher {
	fn credentials(&self) -> Vec<omp_llm_broker::store::CredentialMeta> {
		self
			.store
			.list_credentials(&CredentialFilter {
				provider: Some(self.provider.as_str()),
				now_ms: now_ms(),
				..CredentialFilter::default()
			})
			.unwrap_or_default()
	}
}

struct CatalogRepair;

impl RequestRepair for CatalogRepair {
	fn strip(
		&self,
		_request: &TurnRequest,
		_feature: Feature,
		_classification: &Classification,
	) -> Option<TurnRequest> {
		None
	}
}

struct NoopFrameSink;

impl FrameSink for NoopFrameSink {
	fn on_request(&self, _request: &TurnRequest) {}

	fn on_frame(&self, _frame: &TurnEvent) {}

	fn on_end(&self) {}
}

struct BrokerTerminalUsage {
	observer:            BrokerObserver,
	premium_multipliers: Arc<BTreeMap<(Str, Str), u64>>,
}

impl UsageObserver for BrokerTerminalUsage {
	fn record_usage(
		&self,
		lease: &omp_llm_egress::auth_inject::CredentialLease,
		turn_id: &str,
		model: &str,
		initiator: &str,
		premium_multiplier_millionths: Option<u64>,
		client_id: &str,
		client_label: &str,
		usage: &omp_proto::inference::v1::Usage,
		cost: &omp_proto::inference::v1::Cost,
	) {
		let observed_at_ms = now_ms();
		let usage = ClientUsage {
			client_id:          client_id.into(),
			label:              client_label.into(),
			input_tokens:       usage.input_tokens,
			output_tokens:      usage.output_tokens,
			cache_read_tokens:  usage.cache_read_tokens,
			cache_write_tokens: usage.cache_write_tokens,
			nanos_usd:          cost.nanos_usd,
			last_seen_ms:       observed_at_ms,
		};
		let premium_multiplier_millionths = premium_multiplier_millionths.or_else(|| {
			self
				.premium_multipliers
				.get(&(Str::new(lease.provider()), Str::new(model)))
				.copied()
		});
		if let Err(error) = self.observer.record_terminal_observation(
			lease,
			turn_id,
			model,
			initiator,
			premium_multiplier_millionths,
			Some(&usage),
			observed_at_ms,
		) {
			tracing::warn!(
				credential_id = lease.credential_id(),
				provider = lease.provider(),
				%error,
				"failed to persist terminal provider usage"
			);
		}
	}
}

async fn join_listener_tasks(
	tasks: Vec<JoinHandle<Result<(), ListenerError>>>,
) -> Result<(), DaemonError> {
	for result in join_all(tasks).await {
		result??;
	}
	Ok(())
}

async fn drain_listener_tasks(
	tasks: &mut FuturesUnordered<JoinHandle<Result<(), ListenerError>>>,
) -> Result<(), DaemonError> {
	while let Some(result) = tasks.next().await {
		result??;
	}
	Ok(())
}

#[cfg(unix)]
async fn shutdown_signal() -> Result<(), std::io::Error> {
	use tokio::signal::unix::{SignalKind, signal};
	let mut terminate = signal(SignalKind::terminate())?;
	tokio::select! {
		result = tokio::signal::ctrl_c() => result,
		_ = terminate.recv() => Ok(()),
	}
}

#[cfg(not(unix))]
async fn shutdown_signal() -> Result<(), std::io::Error> {
	tokio::signal::ctrl_c().await
}

fn now_ms() -> u64 {
	SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.unwrap_or_default()
		.as_millis()
		.try_into()
		.unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn apple_embedded_route_is_registered_only_after_a_successful_probe() {
		let providers = Arc::new(omp_llm_catalog::provider::load_builtin().unwrap());
		let apple = &providers[APPLE_PROVIDER_ID];

		let mut available = SpecializedChats::default();
		let chat: Arc<dyn Chat> = Arc::new(AppleFmChat::new(omp_llm_fm::AppleFm));
		assert!(mount_apple_chat(&mut available, Some(chat)));
		assert!(available.embedded.is_some());
		assert!(runtime_provider_is_usable(apple, true));

		let mut unavailable = SpecializedChats::default();
		assert!(!mount_apple_chat(&mut unavailable, None));
		assert!(unavailable.embedded.is_none());
		assert!(!runtime_provider_is_usable(apple, false));

		let temp = tempfile::tempdir().unwrap();
		let store = Arc::new(Store::open(temp.path().join("broker.db")).unwrap());
		let available_view = BrokerAvailability {
			store:              Arc::clone(&store),
			providers:          Arc::clone(&providers),
			apple_fm_available: true,
		};
		assert_eq!(available_view.availability(APPLE_PROVIDER_ID), Availability::Available);
		let unavailable_view = BrokerAvailability { store, providers, apple_fm_available: false };
		assert_eq!(unavailable_view.availability(APPLE_PROVIDER_ID), Availability::Disabled);
	}

	#[test]
	fn cursor_and_devin_routes_require_broker_credentials() {
		let providers = omp_llm_catalog::provider::load_builtin().unwrap();
		let temp = tempfile::tempdir().unwrap();
		let store = Store::open(temp.path().join("broker.db")).unwrap();

		for provider_id in ["cursor", "devin"] {
			let provider = &providers[provider_id];
			assert!(!provider_is_configured(provider, &store));
			store
				.upsert_api_key(provider_id, "test-account", b"sealed-route-secret", now_ms())
				.unwrap();
			assert!(provider_is_configured(provider, &store));
		}
	}
}

#[cfg(test)]
mod cache_policy_tests {
	use omp_llm_catalog::provider::TransportId;

	use super::cache_policy;

	/// Route registration hands one config closure every provider, so the
	/// Anthropic-shaped policy must be selected by transport rather than
	/// defaulted globally.
	#[test]
	fn only_anthropic_routes_receive_the_tail_two_policy() {
		let anthropic = cache_policy(TransportId::AnthropicMessages);
		assert_eq!(anthropic.breakpoint, omp_proto::inference::v1::cache_hint::Breakpoint::TailTwo);
		assert_eq!(anthropic.retention, omp_proto::inference::v1::cache_hint::Retention::Short);
		// Refreshes stay off until the upstream abort path is proven: an
		// unaborted ping buys a whole generation, and placement is worth far
		// more than refreshes are.
		assert_eq!(anthropic.pings, 0, "keep-alive shipped before the abort path was proven");

		// Every other transport keeps the inert default: OpenAI Responses turns
		// the same hint into `prompt_cache_key`, and Bedrock reads `breakpoint`
		// with its own meaning.
		for transport in [
			TransportId::OpenAiResponses,
			TransportId::BedrockConverse,
			TransportId::AnthropicBedrock,
			TransportId::Cursor,
		] {
			let policy = cache_policy(transport);
			assert_eq!(
				policy.breakpoint,
				omp_proto::inference::v1::cache_hint::Breakpoint::Unspecified,
				"{transport:?} inherited Anthropic placement"
			);
			assert_eq!(policy.pings, 0, "{transport:?} scheduled refreshes");
		}
	}
}
