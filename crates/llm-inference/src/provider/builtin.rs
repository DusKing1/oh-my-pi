//! Canonical production construction for catalog-backed provider routes.

use std::{
	collections::BTreeMap,
	future::{Ready, ready},
	num::NonZeroU32,
	sync::Arc,
	time::{Duration, Instant, SystemTime},
};

use omp_core::Str;
use omp_llm_catalog::{
	OperationBits, OperationKind,
	provider::{AuthSpecKind, CodecProfile, DiscoveryKind, RouteDef, TransportKind},
	snapshot::Catalog,
};
use secrecy::ExposeSecret as _;
use tower::{Service, util::BoxCloneSyncService};

use crate::{
	account::{
		AccountPool, AccountSelection, AccountSelectionRequest, RateAvailability, RotationPolicy,
	},
	auth::{
		AuthManager, CredentialBroker, CredentialNeed, CredentialSource, OAuthHttpClient,
		OAuthHttpRequest,
	},
	call::{Call, NativeResponseFraming, OperationCall, Setting, ToolChoice},
	codec::{
		Cancellation, Codec, DecodeContext, DecoderState, EncodeAttempt, EncodeContext,
		EncodedRequest, HandshakenResponse, NativeResponseFormat, TransportAttempt, TransportRequest,
		anthropic::AnthropicCodec,
		bedrock::BedrockConverseCodec,
		cursor::CursorCodec,
		devin::DevinCodec,
		discovery::{
			AccountModelsDiscoveryCodec, GoogleModelsDiscoveryCodec, OllamaTagsDiscoveryCodec,
			OpenAiModelsDiscoveryCodec,
		},
		gemini::GeminiCodec,
		gitlab::GitLabWorkflowCodec,
		google_cca::{AntigravityPolicy, CcaHeaders, GoogleCcaCodec},
		ollama::OllamaCodec,
		openai::OpenAiCodec,
		openai_codex::OpenAiCodexCodec,
		openai_embedding::OpenAiEmbeddingCodec,
		openai_responses::OpenAiResponsesCodec,
		search_exa::ExaSearchCodec,
		search_kagi::KagiSearchCodec,
		search_parallel::ParallelSearchCodec,
		search_perplexity::PerplexitySearchCodec,
		search_tavily::TavilySearchCodec,
	},
	error::{Error, ErrorDetail, ErrorKind, ErrorPhase, RetryAction},
	gate::GateCondition,
	layer::{
		AttemptAction, ExecutionContext,
		account::{AccountPoolLayer, AccountSelector},
		admission::{AdmissionController, AdmissionLayer},
		auth::{AuthLeaseLayer, LeaseProvider},
		encode::{AttemptEncoder, CredentialApplier, CredentialApplyLayer},
		intent::{IntentLayer, IntentPlanner},
		operation::{EmbeddingRoutePolicy, OperationPolicyConfig, OperationPolicyLayer},
		rate::{RateLayer, RateLimiter},
		recover::{DiscoveryProjector, RecoveryLayer},
		retry::TransportRetryLayer,
		semantic::{SemanticLayer, SemanticPolicy},
		session::SessionLayer,
		stack::{RouteComposer, RouteProviderService, RouteStackLayers, build_route_stack},
	},
	operation::{
		discovery::CatalogDiscoveryProjector, embedding::NormalizationSupport,
		usage::UsageServiceConfig,
	},
	receipt::{ExecutionReceipt, ReasonId},
	registry::RouteUnavailable,
	transport::{http::HttpTransport, websocket_transport::WebSocketTransport},
};

/// Explicit route construction settings for the two Cloud Code Assist clients.
#[derive(Clone)]
pub struct GoogleCcaConfig {
	/// Platform coordinate used in Gemini CLI's public model-bearing
	/// fingerprint.
	pub gemini_cli_platform: Str,
	/// Architecture coordinate used in Gemini CLI's public model-bearing
	/// fingerprint.
	pub gemini_cli_arch:     Str,
	/// Public Antigravity fingerprint supplied by application policy.
	pub antigravity_headers: CcaHeaders,
	/// Typed Antigravity lowering policy.
	pub antigravity_policy:  AntigravityPolicy,
}

/// Fetches the latest Antigravity client version from the official update
/// manifest.
///
/// Returns `None` on any transport, status, or parse failure; callers keep
/// the pinned [`DEFAULT_ANTIGRAVITY_VERSION`] fallback valid. The request
/// mimics electron-builder's update probe so the endpoint serves the same
/// manifest the real client sees.
///
/// [`DEFAULT_ANTIGRAVITY_VERSION`]: crate::codec::google_cca::DEFAULT_ANTIGRAVITY_VERSION
pub async fn discover_antigravity_version(client: &dyn OAuthHttpClient) -> Option<Str> {
	let mut headers = http::HeaderMap::new();
	headers.insert(http::header::CACHE_CONTROL, http::HeaderValue::from_static("no-cache"));
	headers.insert(http::header::USER_AGENT, http::HeaderValue::from_static("electron-builder"));
	let request = OAuthHttpRequest::new(
		http::Method::GET,
		crate::codec::google_cca::ANTIGRAVITY_VERSION_MANIFEST_URL,
		headers,
		None,
	)
	.ok()?;
	let response = client.execute(request).await.ok()?;
	if response.status != 200 {
		return None;
	}
	crate::codec::google_cca::parse_antigravity_manifest_version(response.body.expose_secret())
}

/// Resolved non-secret signing regions supplied by the application.
#[derive(Clone)]
pub struct AuthApplicationConfig {
	/// Resolved environment/endpoint signing region keyed by route.
	pub signing_regions: Arc<BTreeMap<omp_llm_catalog::RouteId, Str>>,
}

type WireService = BoxCloneSyncService<TransportRequest, HandshakenResponse, Error>;

/// Feature-gated in-process backend inserted beneath the canonical fixed stack.
#[derive(Clone)]
pub struct LocalRouteBackend {
	codec:             Arc<dyn Codec>,
	wire:              WireService,
	framework_timeout: Duration,
}

impl LocalRouteBackend {
	/// Erases a concrete local codec and transport once at application
	/// construction.
	pub fn new<S>(codec: Arc<dyn Codec>, wire: S, framework_timeout: Duration) -> Self
	where
		S: Service<TransportRequest, Response = HandshakenResponse, Error = Error>
			+ Clone
			+ Send
			+ Sync
			+ 'static,
		S::Future: Send + 'static,
	{
		Self { codec, wire: WireService::new(wire), framework_timeout }
	}
}
/// Complete dependencies required to construct production catalog routes.
#[derive(Clone)]
pub struct ProductionDependencies {
	credentials:          CredentialBroker,
	auth_manager:         AuthManager,
	accounts:             AccountPool,
	sessions:             crate::session::ConversationSessionPlanner,
	websocket:            WebSocketTransport,
	http:                 HttpTransport,
	admission:            AdmissionController,
	google_cca:           GoogleCcaConfig,
	transport_timeout:    Duration,
	auth_application:     AuthApplicationConfig,
	local_routes:         Arc<BTreeMap<crate::catalog::RouteId, LocalRouteBackend>>,
	discovery_projectors: Arc<BTreeMap<crate::catalog::RouteId, Arc<dyn DiscoveryProjector>>>,
	local_unavailable:    Arc<BTreeMap<crate::catalog::RouteId, ReasonId>>,
}

impl ProductionDependencies {
	/// Creates production dependencies with explicit policy and shared state.
	pub fn new(
		credentials: CredentialBroker,
		auth_manager: AuthManager,
		accounts: AccountPool,
		sessions: crate::session::ConversationSessionPlanner,
		websocket: WebSocketTransport,
		google_cca: GoogleCcaConfig,
		http: HttpTransport,
		auth_application: AuthApplicationConfig,
		admission: AdmissionController,
		transport_timeout: Duration,
		discovery_projectors: Arc<BTreeMap<crate::catalog::RouteId, Arc<dyn DiscoveryProjector>>>,
	) -> Self {
		Self {
			credentials,
			auth_manager,
			accounts,
			sessions,
			websocket,
			http,
			admission,
			google_cca,
			auth_application,
			transport_timeout,
			discovery_projectors,
			local_routes: Arc::new(BTreeMap::new()),
			local_unavailable: Arc::new(BTreeMap::new()),
		}
	}

	pub(crate) fn auth_manager(&self) -> AuthManager {
		self.auth_manager.clone()
	}

	/// Adds feature-gated local codec/transport pairs keyed by exact catalog
	/// route.
	pub fn with_local_routes(
		mut self,
		routes: impl IntoIterator<Item = (crate::RouteId, LocalRouteBackend)>,
	) -> Self {
		self.local_routes = Arc::new(routes.into_iter().collect());
		self
	}

	/// Adds precise platform/feature availability evidence for unconstructed
	/// local routes.
	pub fn with_local_unavailable(
		mut self,
		routes: impl IntoIterator<Item = (crate::RouteId, ReasonId)>,
	) -> Self {
		self.local_unavailable = Arc::new(routes.into_iter().collect());
		self
	}
}
/// Concrete route composer used by [`crate::layer::stack::BuiltinConfig`].
#[derive(Clone)]
pub struct ProductionRouteComposer {
	dependencies: ProductionDependencies,
}

impl ProductionRouteComposer {
	/// Creates a composer owning all shared production dependencies.
	pub fn new(dependencies: ProductionDependencies) -> Self {
		Self { dependencies }
	}
}

impl RouteComposer for ProductionRouteComposer {
	fn compose(
		&self,
		catalog: &Catalog,
		route: &RouteDef,
	) -> Result<RouteProviderService, RouteUnavailable> {
		let (mut binding, wire, framework_timeout) = match route.transport {
			TransportKind::Local => {
				let backend = self
					.dependencies
					.local_routes
					.get(&route.id)
					.ok_or_else(|| {
						self
							.dependencies
							.local_unavailable
							.get(&route.id)
							.cloned()
							.map_or_else(
								|| {
									unavailable(
										route,
										"local-route-not-constructed-for-current-platform-or-feature",
									)
								},
								|reason| RouteUnavailable {
									route: route.id.clone(),
									reason,
									operation: None,
								},
							)
					})?;
				(
					local_codec_binding(route, backend.codec.clone())?,
					backend.wire.clone(),
					backend.framework_timeout,
				)
			},
			TransportKind::Http | TransportKind::AwsEventStream | TransportKind::Connect => (
				codec_binding(route, &self.dependencies.google_cca)?,
				WireService::new(self.dependencies.http.clone()),
				self.dependencies.transport_timeout,
			),
			TransportKind::Websocket => (
				codec_binding(route, &self.dependencies.google_cca)?,
				WireService::new(self.dependencies.websocket.clone()),
				self.dependencies.transport_timeout,
			),
			TransportKind::Webrtc => return Err(unavailable(route, "transport-not-implemented")),
		};
		let discovery = discovery_codec(catalog, route, &binding)?;
		if discovery.is_some() {
			binding.supported.insert_kind(OperationKind::DiscoverModels);
		}
		let advertised = advertised_operations(catalog, route);
		let operation = operation_policy(&binding, advertised);
		let codec = Arc::new(RouteCodecSet::for_route(route, advertised, binding, discovery)?);
		let recovery = if advertised.contains_kind(OperationKind::DiscoverModels) {
			let projector = match self
				.dependencies
				.discovery_projectors
				.get(&route.id)
				.cloned()
			{
				Some(projector) => projector,
				None => Arc::new(
					CatalogDiscoveryProjector::for_route(catalog, route)
						.map_err(|_| unavailable(route, "catalog-discovery-projector-invalid"))?,
				),
			};
			RecoveryLayer::new(projector)
		} else {
			RecoveryLayer::without_discovery()
		};
		let auth = catalog
			.auth_spec(&route.auth)
			.ok_or_else(|| unavailable(route, "catalog-auth-spec-missing"))?;
		let authenticated = auth.kind != AuthSpecKind::None;
		let oauth = auth.oauth.as_ref().and_then(|id| catalog.oauth_spec(id));
		let signing_region = self
			.dependencies
			.auth_application
			.signing_regions
			.get(&route.id)
			.cloned()
			.or_else(|| route.endpoint.region.clone());
		let runtime_auth = crate::auth::spec::AuthSpec::from_catalog(auth, oauth, signing_region)
			.map_err(|_| unavailable(route, "catalog-auth-spec-invalid"))?;
		let account = RouteAccountSelector {
			pool: self.dependencies.accounts.clone(),
			provider: route.provider.clone(),
			route: route.id.clone(),
			authenticated,
		};
		let leases = RouteLeaseProvider {
			source: self.dependencies.credentials.clone(),
			spec: route.auth.clone(),
			authenticated,
		};
		let encoder = RouteEncoder {
			route: route.clone(),
			headers: catalog
				.header_profile(&route.headers)
				.map(|profile| {
					profile
						.headers
						.iter()
						.map(|header| crate::codec::RequestHeader {
							name:  header.name.clone(),
							value: header.value.clone(),
						})
						.collect::<Vec<_>>()
						.into_boxed_slice()
				})
				.unwrap_or_default(),
			codec,
			transport_timeout: self.dependencies.transport_timeout.min(framework_timeout),
		};
		let stack = build_route_stack(wire, RouteStackLayers {
			intent: IntentLayer::new(PlannedIntent { route: route.id.clone() }),
			session: SessionLayer::new(self.dependencies.sessions.clone()),
			semantic: SemanticLayer::new(CanonicalSemantic),
			operation: OperationPolicyLayer::new(operation),
			recovery,
			admission: AdmissionLayer::new(self.dependencies.admission.clone()),
			account: AccountPoolLayer::new(account),
			auth: AuthLeaseLayer::new(leases),
			retry: TransportRetryLayer::new(u32::MAX),
			rate: RateLayer::new(PoolRateLimiter { pool: self.dependencies.accounts.clone() }),
			encode: crate::layer::encode::EncodeLayer::new(encoder, false),
			credential_apply: CredentialApplyLayer::new(RouteCredentialApplier { auth: runtime_auth }),
		});
		Ok(RouteProviderService::new(stack))
	}
}

#[derive(Clone)]
struct CodecBinding {
	primary:                   Arc<dyn Codec>,
	supported:                 OperationBits,
	embedding:                 Option<EmbeddingRoutePolicy>,
	openai_embedding_override: bool,
}

fn operation_bits(kinds: &[OperationKind]) -> OperationBits {
	let mut bits = OperationBits::empty();
	for kind in kinds {
		bits.insert_kind(*kind);
	}
	bits
}

fn operation_policy(binding: &CodecBinding, advertised: OperationBits) -> OperationPolicyConfig {
	OperationPolicyConfig {
		embedding:              advertised
			.contains_kind(OperationKind::Embed)
			.then(|| binding.embedding)
			.flatten(),
		native:                 None,
		usage:                  UsageServiceConfig::new(Duration::MAX),
		discovery_maximum_page: advertised
			.contains_kind(OperationKind::DiscoverModels)
			.then_some(NonZeroU32::MAX),
		exact_token_count:      binding.supported.contains_kind(OperationKind::CountTokens),
	}
}

fn local_codec_binding(
	route: &RouteDef,
	primary: Arc<dyn Codec>,
) -> Result<CodecBinding, RouteUnavailable> {
	match (route.codec.as_str(), route.codec_profile) {
		("local", CodecProfile::AppleFm) => Ok(CodecBinding {
			primary,
			supported: operation_bits(&[OperationKind::Chat, OperationKind::DiscoverModels]),
			embedding: None,
			openai_embedding_override: false,
		}),
		_ => Err(unavailable(route, "codec-or-profile-not-implemented")),
	}
}

fn codec_binding(
	route: &RouteDef,
	cca: &GoogleCcaConfig,
) -> Result<CodecBinding, RouteUnavailable> {
	let (primary, supported, embedding, openai_embedding_override): (
		Arc<dyn Codec>,
		OperationBits,
		Option<EmbeddingRoutePolicy>,
		bool,
	) = match (route.codec.as_str(), route.codec_profile) {
		("anthropic", CodecProfile::Standard) => (
			Arc::new(AnthropicCodec::direct()),
			operation_bits(&[OperationKind::Chat, OperationKind::CountTokens]),
			None,
			false,
		),
		("bedrock-converse", CodecProfile::Standard) => (
			Arc::new(BedrockConverseCodec::default()),
			operation_bits(&[OperationKind::Chat, OperationKind::DiscoverModels]),
			None,
			false,
		),
		("cursor", CodecProfile::Standard) => (
			Arc::new(CursorCodec::new()),
			operation_bits(&[OperationKind::Chat, OperationKind::DiscoverModels]),
			None,
			false,
		),
		("devin", CodecProfile::Standard) => (
			Arc::new(DevinCodec::new()),
			operation_bits(&[OperationKind::Chat, OperationKind::DiscoverModels]),
			None,
			false,
		),
		("gitlab-duo", CodecProfile::Standard) => (
			Arc::new(GitLabWorkflowCodec::new()),
			operation_bits(&[OperationKind::Chat, OperationKind::DiscoverModels]),
			None,
			false,
		),
		("google-genai", CodecProfile::Standard) => (
			Arc::new(GeminiCodec::generative_language(None)),
			operation_bits(&[OperationKind::Chat, OperationKind::CountTokens, OperationKind::Embed]),
			Some(EmbeddingRoutePolicy {
				normalization:          NormalizationSupport::Never,
				maximum_input_tokens:   None,
				native_text_truncation: false,
			}),
			false,
		),
		("google-vertex", CodecProfile::Standard) => (
			Arc::new(GeminiCodec::vertex(None)),
			operation_bits(&[OperationKind::Chat, OperationKind::CountTokens, OperationKind::Embed]),
			Some(EmbeddingRoutePolicy {
				normalization:          NormalizationSupport::Never,
				maximum_input_tokens:   None,
				native_text_truncation: false,
			}),
			false,
		),
		("google-cca", CodecProfile::GoogleCcaGeminiCli) => (
			Arc::new(GoogleCcaCodec::gemini_cli_for_route(
				None,
				cca.gemini_cli_platform.clone(),
				cca.gemini_cli_arch.clone(),
			)),
			operation_bits(&[OperationKind::Chat, OperationKind::DiscoverModels]),
			None,
			false,
		),
		("google-cca", CodecProfile::GoogleCcaAntigravity) => (
			Arc::new(GoogleCcaCodec::antigravity(
				None,
				cca.antigravity_headers.clone(),
				cca.antigravity_policy.clone(),
			)),
			operation_bits(&[OperationKind::Chat, OperationKind::DiscoverModels]),
			None,
			false,
		),
		("ollama", CodecProfile::Standard) => (
			Arc::new(OllamaCodec),
			operation_bits(&[
				OperationKind::Chat,
				OperationKind::Embed,
				OperationKind::DiscoverModels,
			]),
			Some(EmbeddingRoutePolicy {
				normalization:          NormalizationSupport::Never,
				maximum_input_tokens:   None,
				native_text_truncation: true,
			}),
			false,
		),
		("openai-chat", CodecProfile::Standard) => (
			Arc::new(OpenAiCodec::default()),
			operation_bits(&[
				OperationKind::Chat,
				OperationKind::Embed,
				OperationKind::GenerateImage,
				OperationKind::Transcribe,
				OperationKind::Realtime,
			]),
			Some(EmbeddingRoutePolicy {
				normalization:          NormalizationSupport::Never,
				maximum_input_tokens:   None,
				native_text_truncation: false,
			}),
			true,
		),
		("openai-codex", CodecProfile::Standard) => (
			Arc::new(OpenAiCodexCodec::default()),
			operation_bits(&[OperationKind::Chat, OperationKind::DiscoverModels]),
			None,
			false,
		),
		("openai-responses", CodecProfile::Standard) => (
			Arc::new(OpenAiResponsesCodec::default()),
			operation_bits(&[OperationKind::Chat]),
			None,
			false,
		),
		("openai-embedding", CodecProfile::Standard) => (
			Arc::new(OpenAiEmbeddingCodec::for_openai_protocol()),
			operation_bits(&[OperationKind::Embed]),
			Some(EmbeddingRoutePolicy {
				normalization:          NormalizationSupport::Never,
				maximum_input_tokens:   None,
				native_text_truncation: false,
			}),
			false,
		),
		("search-exa", CodecProfile::Standard) => {
			(Arc::new(ExaSearchCodec), operation_bits(&[OperationKind::Search]), None, false)
		},
		("search-kagi", CodecProfile::Standard) => {
			(Arc::new(KagiSearchCodec), operation_bits(&[OperationKind::Search]), None, false)
		},
		("search-parallel", CodecProfile::Standard) => {
			(Arc::new(ParallelSearchCodec), operation_bits(&[OperationKind::Search]), None, false)
		},
		("search-perplexity", CodecProfile::Standard) => {
			(Arc::new(PerplexitySearchCodec), operation_bits(&[OperationKind::Search]), None, false)
		},
		("search-tavily", CodecProfile::Standard) => {
			(Arc::new(TavilySearchCodec), operation_bits(&[OperationKind::Search]), None, false)
		},
		_ => return Err(unavailable(route, "codec-or-profile-not-implemented")),
	};
	Ok(CodecBinding { primary, supported, embedding, openai_embedding_override })
}

fn discovery_codec(
	catalog: &Catalog,
	route: &RouteDef,
	binding: &CodecBinding,
) -> Result<Option<Arc<dyn Codec>>, RouteUnavailable> {
	let Some(discovery) = route.discovery.as_ref() else {
		return Ok(None);
	};
	let spec = catalog
		.discovery_spec(discovery)
		.ok_or_else(|| unavailable(route, "catalog-discovery-spec-missing"))?;
	let codec: Arc<dyn Codec> = match spec.kind {
		DiscoveryKind::OpenAiModels => Arc::new(
			OpenAiModelsDiscoveryCodec::from_spec(spec)
				.map_err(|_| unavailable(route, "openai-models-discovery-codec-invalid"))?,
		),
		DiscoveryKind::OllamaTags => Arc::new(
			OllamaTagsDiscoveryCodec::from_spec(spec)
				.map_err(|_| unavailable(route, "ollama-tags-discovery-codec-invalid"))?,
		),
		DiscoveryKind::AccountModels => Arc::new(
			AccountModelsDiscoveryCodec::from_spec(spec)
				.map_err(|_| unavailable(route, "account-models-discovery-codec-invalid"))?,
		),
		DiscoveryKind::GoogleModels => Arc::new(
			GoogleModelsDiscoveryCodec::from_spec(spec)
				.map_err(|_| unavailable(route, "google-models-discovery-codec-invalid"))?,
		),
		DiscoveryKind::Specialized => {
			if !binding
				.supported
				.contains_kind(OperationKind::DiscoverModels)
			{
				return Err(RouteUnavailable {
					route:     route.id.clone(),
					reason:    ReasonId(Str::from("specialized-discovery-codec-not-implemented")),
					operation: Some(OperationKind::DiscoverModels),
				});
			}
			binding.primary.clone()
		},
	};
	Ok(Some(codec))
}

const OPERATION_COUNT: usize = OperationKind::Native as usize + 1;
const OPERATIONS: [OperationKind; OPERATION_COUNT] = [
	OperationKind::Chat,
	OperationKind::CountTokens,
	OperationKind::Tokenize,
	OperationKind::Detokenize,
	OperationKind::Embed,
	OperationKind::GenerateImage,
	OperationKind::GenerateVideo,
	OperationKind::Speak,
	OperationKind::Transcribe,
	OperationKind::Realtime,
	OperationKind::Search,
	OperationKind::Usage,
	OperationKind::DiscoverModels,
	OperationKind::Auth,
	OperationKind::Native,
];

fn advertised_operations(catalog: &Catalog, route: &RouteDef) -> OperationBits {
	let mut advertised = OperationBits::empty();
	for model in catalog.models() {
		if model.routes.iter().any(|candidate| candidate == &route.id) {
			advertised |= model.capabilities.operations;
		}
	}
	if let Some(limits) = route.capability_limits.operations {
		advertised = OperationBits::from_bits(advertised.bits() & limits.bits());
	}
	if route.discovery.is_some() {
		advertised.insert_kind(OperationKind::DiscoverModels);
	}
	advertised
}

struct RouteCodecSet {
	operations: [Option<Arc<dyn Codec>>; OPERATION_COUNT],
}

impl RouteCodecSet {
	fn for_route(
		route: &RouteDef,
		advertised: OperationBits,
		binding: CodecBinding,
		discovery: Option<Arc<dyn Codec>>,
	) -> Result<Self, RouteUnavailable> {
		let embedding: Arc<dyn Codec> = Arc::new(OpenAiEmbeddingCodec::for_openai_protocol());
		let mut operations: [Option<Arc<dyn Codec>>; OPERATION_COUNT] = std::array::from_fn(|_| None);
		for operation in OPERATIONS {
			if !advertised.contains_kind(operation) {
				continue;
			}
			if !binding.supported.contains_kind(operation) {
				return Err(RouteUnavailable {
					route:     route.id.clone(),
					reason:    ReasonId(Str::from("advertised-operation-codec-not-implemented")),
					operation: Some(operation),
				});
			}
			operations[operation as usize] = Some(if operation == OperationKind::DiscoverModels {
				discovery
					.clone()
					.ok_or_else(|| unavailable(route, "discovery-codec-not-constructed"))?
			} else if binding.openai_embedding_override && operation == OperationKind::Embed {
				embedding.clone()
			} else {
				binding.primary.clone()
			});
		}
		Ok(Self { operations })
	}

	fn codec(&self, operation: OperationKind) -> Result<&Arc<dyn Codec>, Error> {
		self.operations[operation as usize].as_ref().ok_or_else(|| {
			Error::planning(
				ErrorKind::CapabilityMismatch,
				ErrorDetail::Capability {
					feature: Str::from(operation.to_string()),
					reason:  ReasonId(Str::from("operation-not-advertised-on-route")),
				},
				ExecutionReceipt::default(),
			)
		})
	}
}

impl Codec for RouteCodecSet {
	fn encode(
		&self,
		context: &EncodeContext<'_>,
		operation: &OperationCall,
	) -> Result<EncodedRequest, Error> {
		self.codec(operation.kind())?.encode(context, operation)
	}

	fn encode_realtime_handshake(
		&self,
		context: &EncodeContext<'_>,
		operation: &OperationCall,
	) -> Result<Option<EncodedRequest>, Error> {
		self
			.codec(operation.kind())?
			.encode_realtime_handshake(context, operation)
	}

	fn decoder(&self, context: &DecodeContext<'_>) -> Result<DecoderState, Error> {
		self.codec(context.operation)?.decoder(context)
	}

	fn realtime(
		&self,
		context: &DecodeContext<'_>,
	) -> Result<Option<crate::codec::RealtimeWireCodecState>, Error> {
		self.codec(context.operation)?.realtime(context)
	}
}

#[derive(Clone)]
struct RouteEncoder {
	route:             RouteDef,
	headers:           Box<[crate::codec::RequestHeader]>,
	codec:             Arc<dyn Codec>,
	transport_timeout: Duration,
}

fn encode_wire_request(
	codec: &dyn Codec,
	context: &EncodeContext<'_>,
	operation: &OperationCall,
	execution: &ExecutionContext,
) -> Result<EncodedRequest, Error> {
	if operation.kind() == OperationKind::Realtime {
		codec
			.encode_realtime_handshake(context, operation)?
			.ok_or_else(|| contract_error(execution, "realtime-handshake-codec-not-constructed"))
	} else {
		codec.encode(context, operation)
	}
}

impl AttemptEncoder<Call> for RouteEncoder {
	fn encode(
		&self,
		call: &Call,
		execution: &ExecutionContext,
		attempt: u32,
		provisional: bool,
		cancel: Cancellation,
	) -> Result<TransportRequest, Error> {
		let plan = call
			.execution
			.as_ref()
			.ok_or_else(|| contract_error(execution, "execution-plan-missing"))?;
		if plan.route != self.route.id || plan.codec != self.route.codec {
			return Err(contract_error(execution, "route-codec-does-not-match-plan"));
		}
		let account = execution.account_routing();
		let server_state = execution.session_state();
		let encode_context = EncodeContext {
			request_id:         &call.id,
			route:              &self.route,
			target:             plan.wire_target(),
			policy_model:       plan.policy_model.as_deref(),
			policy:             &plan.wire_policy,
			thinking_policy:    plan.thinking_policy.as_deref(),
			thinking_selection: plan.thinking_selection.as_ref(),
			session:            call.session.as_ref(),
			server_state:       server_state.as_ref(),
			account:            account.as_ref(),
			attempt:            EncodeAttempt { index: attempt, provisional },
		};
		let mut encoded =
			encode_wire_request(self.codec.as_ref(), &encode_context, &call.operation, execution)?;
		merge_static_headers(&mut encoded.headers, &self.headers, execution)?;
		let mut timeout = self.transport_timeout;
		if let Some(deadline) = call.deadline {
			timeout = timeout.min(deadline.saturating_duration_since(Instant::now()));
		}
		if let Some(max_elapsed) = call.budget.max_elapsed {
			timeout = timeout.min(max_elapsed.saturating_sub(execution.elapsed()));
		}
		if timeout.is_zero() {
			return Err(Error::new(
				ErrorKind::DeadlineExceeded,
				ErrorPhase::Connecting,
				RetryAction::Never,
				execution.receipt(),
			));
		}
		let decode_context = DecodeContext {
			request_id: &call.id,
			provider: &plan.provider,
			route: &plan.route,
			target: plan.wire_target(),
			policy_model: plan.policy_model.as_deref(),
			policy: &plan.wire_policy,
			thinking_policy: plan.thinking_policy.as_deref(),
			thinking_selection: plan.thinking_selection.as_ref(),
			operation: call.operation.kind(),
			operation_call: &call.operation,
			framing: encoded.framing,
			native_response: native_response(&call.operation),
			attempt,
		};
		decode_context.debug_assert_valid();
		let realtime = self.codec.realtime(&decode_context)?;
		let decoder = if realtime.is_none() {
			Some(self.codec.decoder(&decode_context)?)
		} else {
			None
		};
		if (call.operation.kind() == OperationKind::Realtime) != realtime.is_some() {
			return Err(contract_error(execution, "realtime-wire-codec-contract-mismatch"));
		}
		Ok(TransportRequest {
			encoded,
			credentials: None,
			decoder,
			realtime,
			cancel,
			attempt: TransportAttempt {
				request_id: call.id.clone(),
				provider: plan.provider.clone(),
				route: plan.route.clone(),
				account: account.as_ref().and_then(|routing| routing.account.clone()),
				principal: account
					.as_ref()
					.and_then(|routing| routing.principal.clone()),
				index: attempt,
				provisional,
				capture_limit: call.budget.max_staging_bytes,
				timeout,
			},
		})
	}
}

fn merge_static_headers(
	destination: &mut Box<[crate::codec::RequestHeader]>,
	configured: &[crate::codec::RequestHeader],
	execution: &ExecutionContext,
) -> Result<(), Error> {
	let mut values = BTreeMap::new();
	let mut merged = Vec::with_capacity(destination.len() + configured.len());
	for header in destination.iter().chain(configured) {
		let name = header.name.to_ascii_lowercase();
		match values.get(&name) {
			Some(value) if value == &header.value => continue,
			Some(_) => return Err(contract_error(execution, "conflicting-public-request-header")),
			None => {
				values.insert(name, header.value.clone());
				merged.push(header.clone());
			},
		}
	}
	*destination = merged.into_boxed_slice();
	Ok(())
}

fn native_response(operation: &OperationCall) -> Option<NativeResponseFormat> {
	let OperationCall::Native(request) = operation else {
		return None;
	};
	Some(match request.response_framing {
		NativeResponseFraming::Json => NativeResponseFormat::Json,
		NativeResponseFraming::Sse => NativeResponseFormat::Sse,
		NativeResponseFraming::Bytes => NativeResponseFormat::Bytes,
	})
}

#[derive(Clone, Debug)]
enum RouteAccount {
	Anonymous { _account: AnonymousAccount },
	Brokered { _account: BrokeredAccount },
	Authenticated(AccountSelection),
}
#[derive(Clone, Debug)]
struct AnonymousAccount {
	_provider: crate::catalog::ProviderId,
	_route:    crate::catalog::RouteId,
}
#[derive(Clone, Debug)]
struct BrokeredAccount {
	_provider: crate::catalog::ProviderId,
	_route:    crate::catalog::RouteId,
}

#[derive(Clone)]
struct RouteAccountSelector {
	pool:          AccountPool,
	provider:      crate::catalog::ProviderId,
	route:         crate::catalog::RouteId,
	authenticated: bool,
}

impl AccountSelector<Call> for RouteAccountSelector {
	type Account = RouteAccount;

	fn select(&self, _: &Call, context: &ExecutionContext) -> Result<Self::Account, Error> {
		if !self.authenticated {
			return Ok(RouteAccount::Anonymous {
				_account: AnonymousAccount {
					_provider: self.provider.clone(),
					_route:    self.route.clone(),
				},
			});
		}
		let (previous_account, rotate) = match context.attempt_action() {
			AttemptAction::Initial => (None, false),
			AttemptAction::RefreshCredential { previous_account } => (previous_account, false),
			AttemptAction::RotateAccount { previous_account } => (previous_account, true),
		};
		let affinity = context.session_affinity();
		let preserve_principal = affinity.is_some();
		let request = AccountSelectionRequest {
			provider: self.provider.clone(),
			route: self.route.clone(),
			affinity: affinity.as_ref().map(|binding| binding.principal.clone()),
			previous_account,
			previous_principal: affinity.as_ref().map(|binding| binding.principal.clone()),
			rotate,
			rotation: RotationPolicy { allow_account_change: true, preserve_principal },
			now: SystemTime::now(),
		};
		match self.pool.select(&request) {
			Ok(selection) => Ok(RouteAccount::Authenticated(selection)),
			Err(error) if error.receipt.candidates.is_empty() => Ok(RouteAccount::Brokered {
				_account: BrokeredAccount {
					_provider: self.provider.clone(),
					_route:    self.route.clone(),
				},
			}),
			Err(_) => Err(Error::new(
				ErrorKind::Authentication,
				ErrorPhase::Authentication,
				RetryAction::Never,
				context.receipt(),
			)),
		}
	}

	fn routing(&self, account: &Self::Account) -> Option<crate::call::AccountRoutingContext> {
		match account {
			RouteAccount::Anonymous { .. } | RouteAccount::Brokered { .. } => None,
			RouteAccount::Authenticated(selection) => Some(selection.routing.clone()),
		}
	}
}

#[derive(Clone)]
struct RouteLeaseProvider {
	source:        CredentialBroker,
	spec:          crate::catalog::AuthSpecId,
	authenticated: bool,
}

impl LeaseProvider<Call, RouteAccount> for RouteLeaseProvider {
	type Lease = Option<crate::auth::CredentialLease>;

	type Future<'a> = impl Future<Output = Result<Self::Lease, Error>> + Send + 'a;

	fn acquire<'a>(
		&'a self,
		_: &'a Call,
		account: &'a RouteAccount,
		context: &'a ExecutionContext,
	) -> Self::Future<'a> {
		async move {
			if !self.authenticated {
				return match account {
					RouteAccount::Anonymous { .. } => Ok(None),
					RouteAccount::Brokered { .. } | RouteAccount::Authenticated(_) => {
						Err(contract_error(context, "authenticated-account-on-anonymous-route"))
					},
				};
			}
			let (account, principal) = match account {
				RouteAccount::Brokered { .. } => (None, None),
				RouteAccount::Authenticated(selection) => {
					(Some(selection.record.account.clone()), Some(selection.record.principal.clone()))
				},
				RouteAccount::Anonymous { .. } => {
					return Err(contract_error(context, "anonymous-account-on-authenticated-route"));
				},
			};
			let need = CredentialNeed {
				spec: self.spec.clone(),
				account,
				principal,
				valid_after: SystemTime::now(),
			};
			self.source.lease(need).await.map(Some).map_err(|_| {
				Error::new(
					ErrorKind::Authentication,
					ErrorPhase::Authentication,
					RetryAction::Never,
					context.receipt(),
				)
			})
		}
	}
}

#[derive(Clone)]
struct RouteCredentialApplier {
	auth: crate::auth::AuthSpec,
}

impl CredentialApplier<RouteAccount, Option<crate::auth::CredentialLease>>
	for RouteCredentialApplier
{
	fn apply(
		&self,
		_: &RouteAccount,
		lease: Option<crate::auth::CredentialLease>,
		request: &mut TransportRequest,
		context: &ExecutionContext,
	) -> Result<(), Error> {
		match (&self.auth, lease) {
			(crate::auth::AuthSpec::None, None) => Ok(()),
			(crate::auth::AuthSpec::None, Some(_)) => {
				Err(authentication_error(context, "credential-on-anonymous-route"))
			},
			(_, None) => Err(authentication_error(context, "credential-lease-missing")),
			(_, Some(lease)) => {
				let credentials = lease
					.prepare(&self.auth, SystemTime::now())
					.map_err(|_| authentication_error(context, "credential-application-failed"))?;
				request.credentials = Some(credentials);
				Ok(())
			},
		}
	}
}

#[derive(Clone)]
struct PoolRateLimiter {
	pool: AccountPool,
}

impl<R> RateLimiter<R> for PoolRateLimiter {
	type Future<'a>
		= Ready<Result<(), Error>>
	where
		R: 'a;

	fn reserve<'a>(&'a self, _: &'a R, context: &'a ExecutionContext) -> Self::Future<'a> {
		let result = context.checkpoint(ErrorPhase::Readiness).and_then(|()| {
			let Some(account) = context
				.account_routing()
				.and_then(|routing| routing.account)
			else {
				return Ok(());
			};
			match self
				.pool
				.rate_state(&account)
				.availability(SystemTime::now())
			{
				RateAvailability::Available => Ok(()),
				RateAvailability::Delayed { until } => Err(Error::new(
					ErrorKind::RateLimited,
					ErrorPhase::Admission,
					RetryAction::SameRoute {
						after: until.duration_since(SystemTime::now()).unwrap_or_default(),
					},
					context.receipt(),
				)),
				RateAvailability::ExhaustedUnknownReset => Err(Error::new(
					ErrorKind::RateLimited,
					ErrorPhase::Admission,
					RetryAction::Never,
					context.receipt(),
				)),
			}
		});
		ready(result)
	}
}

#[derive(Clone)]
struct PlannedIntent {
	route: crate::catalog::RouteId,
}
impl IntentPlanner for PlannedIntent {
	fn negotiate(&self, call: &mut Call, _: &mut ExecutionReceipt) -> Result<(), Error> {
		let Some(plan) = &call.execution else {
			return Err(Error::planning(
				ErrorKind::ProviderContractMismatch,
				ErrorDetail::Protocol { reason: ReasonId(Str::from("execution-plan-missing")) },
				ExecutionReceipt::default(),
			));
		};
		if plan.route != self.route {
			return Err(Error::planning(
				ErrorKind::ProviderContractMismatch,
				ErrorDetail::Protocol { reason: ReasonId(Str::from("planned-route-mismatch")) },
				ExecutionReceipt::default(),
			));
		}
		Ok(())
	}
}
#[derive(Clone, Copy)]
struct CanonicalSemantic;
impl SemanticPolicy<Call> for CanonicalSemantic {
	fn condition(&self, call: &Call) -> Option<GateCondition> {
		let OperationCall::Chat(chat) = &call.operation else {
			return None;
		};
		if let Setting::Require(ToolChoice::Named(tool)) = &chat.tool_choice {
			return Some(GateCondition::ToolCallReady { tool: tool.clone() });
		}
		if matches!(chat.output, Setting::Require(_)) {
			return Some(GateCondition::ValidStructuredOutput);
		}
		if matches!(chat.tool_choice, Setting::Require(ToolChoice::Required)) {
			return Some(GateCondition::WholeAttempt);
		}
		None
	}

	fn max_retries(&self, call: &Call) -> u32 {
		call.budget.max_attempts.saturating_sub(1)
	}
}

fn unavailable(route: &RouteDef, reason: &'static str) -> RouteUnavailable {
	RouteUnavailable {
		route:     route.id.clone(),
		reason:    ReasonId(Str::from(reason)),
		operation: None,
	}
}

fn authentication_error(context: &ExecutionContext, reason: &'static str) -> Error {
	let mut error = Error::new(
		ErrorKind::Authentication,
		ErrorPhase::Authentication,
		RetryAction::Never,
		context.receipt(),
	);
	error.detail = Some(ErrorDetail::Protocol { reason: ReasonId(Str::from(reason)) });
	error
}

fn contract_error(context: &ExecutionContext, reason: &'static str) -> Error {
	let mut error = Error::new(
		ErrorKind::ProviderContractMismatch,
		ErrorPhase::Encoding,
		RetryAction::Never,
		context.receipt(),
	);
	error.detail = Some(ErrorDetail::Protocol { reason: ReasonId(Str::from(reason)) });
	error
}

#[cfg(test)]
mod tests {
	use omp_llm_catalog::{PolicyModel, WireTarget};

	use super::*;
	use crate::{
		call::{NegotiationPolicy, RealtimeRequest},
		id::RequestId,
		receipt::ExecutionBudget,
	};

	#[test]
	fn realtime_route_encoder_constructs_websocket_handshake_before_http_encode() {
		let catalog = Catalog::try_embedded().expect("embedded catalog");
		let (model, route, wire_model) = catalog
			.models()
			.iter()
			.find_map(|model| {
				model.routes.iter().find_map(|route_id| {
					let route = catalog.route(route_id)?;
					if route.codec.as_str() != "openai-chat" {
						return None;
					}
					let wire_model = model
						.wire_ids
						.iter()
						.find(|(candidate, _)| candidate == route_id)
						.map(|(_, wire_model)| wire_model.clone())?;
					Some((model, route.clone(), wire_model))
				})
			})
			.expect("catalog OpenAI route");
		let mut route = route;
		route.transport = TransportKind::Websocket;
		let cca = GoogleCcaConfig {
			gemini_cli_platform: Str::from("test"),
			gemini_cli_arch:     Str::from("test"),
			antigravity_headers: CcaHeaders::antigravity(
				&crate::codec::google_cca::AntigravityFingerprint::default(),
				false,
				None,
			),
			antigravity_policy:  AntigravityPolicy::default(),
		};
		let binding = codec_binding(&route, &cca).expect("route codec binding");
		let codec = RouteCodecSet::for_route(
			&route,
			OperationBits::for_kind(OperationKind::Realtime),
			binding,
			None,
		)
		.expect("realtime codec slot");
		let policy_model = PolicyModel::from(model);
		let wire_policy = catalog
			.wire_policy(&model.wire_policy)
			.expect("wire policy");
		let target = WireTarget {
			route: route.id.clone(),
			codec: route.codec.clone(),
			endpoint: route.endpoint.clone(),
			wire_model,
		};
		let operation = OperationCall::Realtime(Arc::new(RealtimeRequest {
			instructions:   None,
			modalities:     Arc::from([]),
			voice:          None,
			input_audio:    Setting::Unset,
			output_audio:   Setting::Unset,
			turn_detection: Setting::Unset,
			tools:          Arc::from([]),
			negotiation:    NegotiationPolicy::default(),
		}));
		let request_id = RequestId::new("realtime-handshake-test");
		let context = EncodeContext {
			request_id:         &request_id,
			route:              &route,
			target:             Some(&target),
			policy_model:       Some(&policy_model),
			policy:             wire_policy,
			thinking_policy:    None,
			thinking_selection: None,
			session:            None,
			server_state:       None,
			account:            None,
			attempt:            EncodeAttempt { index: 0, provisional: false },
		};
		let execution = ExecutionContext::new(ExecutionBudget::default());
		let encoded =
			encode_wire_request(&codec, &context, &operation, &execution).expect("realtime handshake");
		assert_eq!(encoded.operation, OperationKind::Realtime);
		assert_eq!(encoded.method, crate::codec::RequestMethod::Get);
		assert_eq!(encoded.framing, crate::transport::FramingProtocol::WebSocket);
		assert!(encoded.uri.as_str().contains("/v1/realtime?model="));
	}
}
