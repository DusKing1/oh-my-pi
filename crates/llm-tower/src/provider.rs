//! Catalog-driven HTTP provider attempt adaptation.
//!
//! This module is the typed commit boundary between replayable HTTP egress and
//! canonical turn events. Request bodies are completely buffered before egress,
//! while response headers and empty/control frames remain pre-commit. The
//! service future resolves only after the first canonical event has decoded.

use std::{
	collections::{HashMap, VecDeque},
	fmt,
	future::Future,
	pin::Pin,
	sync::{Arc, LazyLock},
	task::{Context, Poll},
	time::{SystemTime, UNIX_EPOCH},
};

use bytes::{Bytes, BytesMut};
use futures::{Stream, StreamExt};
use http::{HeaderMap, Request, Response, StatusCode, header};
use http_body_util::{BodyExt, Full};
use hyper::body::Body as HttpBody;
use omp_core::{Str, fmts};
use omp_llm_anthropic::{
	AnthropicCodec,
	bedrock::{self, BedrockCodec, BedrockEventStreamDecoder},
	vertex::{self, VertexCodec},
};
use omp_llm_bedrock::BedrockConverseCodec;
use omp_llm_catalog::{
	compat::{Compat, StreamProtocol},
	provider::{
		AuthSpec, BaseUrlVars, CodexTransportPreference, ProviderEntry, TransportId, expand_base_url,
	},
};
use omp_llm_egress::{
	auth_inject::{AuthContext, CredentialAuthKind, CredentialMetadata},
	client::Body,
	retry::parse_retry_after,
};
use omp_llm_google::{
	GoogleCodec,
	cca::{AntigravityRequestMetadata, CcaCodec, CcaEndpointPlan},
	stream::cca_first_event_timeout,
	vertex as google_vertex,
};
use omp_llm_ollama::OllamaChatCodec;
use omp_llm_openai::{
	CodexAttestor, CodexCredentialMetadata, CodexHeaderContext, CodexRequestIdentity,
	CodexWireTransport, OpenAiChatCodec, OpenAiCodexCodec, OpenAiResponsesCodec,
	apply_codex_client_metadata, build_codex_header_plan,
};
use omp_llm_transport::{DecodeState, Frame, Transport, ndjson::NdjsonDecoder, sse::SseDecoder};
use omp_llm_types::{
	Chat, ChatRequest, Effort, Error as ChatError, Executor, ItemKind, Part, ResolvedModelPolicy,
	TurnError, TurnErrorKind, TurnEvent as NativeTurnEvent, Unsupported, UnsupportedAction,
};
use omp_proto::inference::v1::{TurnEvent, TurnRequest};
use parking_lot::Mutex;
use smallvec::SmallVec;
use tower::{Service, ServiceExt};

use crate::{codex_websocket::CodexWebSocketRequest, select::Routed};

/// Route values needed by catalog URL templates and Google endpoint families.
///
/// These values are endpoint metadata, never credential material. `project` is
/// used by Vertex endpoint paths and as the unleased CCA default; a selected
/// credential's validated project overrides it before encoding. `deployment`
/// defaults to the requested model when omitted.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ProviderRoute {
	/// Google Cloud project for Vertex or unleased Cloud Code Assist routes.
	pub project:    Str,
	/// Region substituted into `{region}` and Vertex publisher paths.
	pub region:     Str,
	/// Deployment substituted into `{deployment}`; the model is used if empty.
	pub deployment: Str,
	/// Account identifier substituted into gateway endpoint templates.
	pub account:    Str,
	/// Gateway identifier substituted into gateway endpoint templates.
	pub gateway:    Str,
}

/// The statically dispatched set of HTTP provider codecs.
///
/// The enum deliberately avoids a `dyn Transport` call on every frame. Only
/// transports whose wire protocol is ordinary HTTP can be represented here.
#[derive(Debug)]
#[allow(
	clippy::large_enum_variant,
	reason = "provider codecs are constructed once behind Arc; keeping concrete variants preserves \
	          allocation-free static dispatch on every streamed frame"
)]
pub enum HttpCodec {
	/// OpenAI-compatible Chat Completions.
	OpenAiChat(OpenAiChatCodec),
	/// OpenAI-compatible Responses.
	OpenAiResponses(OpenAiResponsesCodec),
	/// `ChatGPT` subscription Codex Responses.
	OpenAiCodex(OpenAiCodexCodec),
	/// Anthropic Messages.
	Anthropic(AnthropicCodec),
	/// Anthropic Messages adapted to AWS Bedrock.
	AnthropicBedrock(BedrockCodec),
	/// Amazon Bedrock model-independent Converse Stream.
	BedrockConverse(BedrockConverseCodec),
	/// Anthropic Messages adapted to Google Vertex AI.
	AnthropicVertex(VertexCodec),
	/// Public Google Generative Language API.
	GoogleGenAi(GoogleCodec),
	/// Google Vertex AI publisher-model API.
	GoogleVertex(GoogleCodec),
	/// Google Cloud Code Assist with the Gemini CLI wire identity.
	GoogleCca(CcaCodec),
	/// Ollama native `/api/chat`.
	Ollama(OllamaChatCodec),
	/// Google Cloud Code Assist with Antigravity request-scoped identity.
	///
	/// The immutable `CcaCodec` snapshot is constructed from canonical metadata
	/// immediately before encoding; only its route project is retained here.
	GoogleCcaAntigravity(Str),
}

impl HttpCodec {
	fn from_catalog(
		provider: &ProviderEntry,
		route: &ProviderRoute,
	) -> Result<Self, ProviderBuildError> {
		match provider.transport {
			TransportId::OpenAiChat => Ok(Self::OpenAiChat(OpenAiChatCodec)),
			TransportId::OpenAiResponses => Ok(Self::OpenAiResponses(OpenAiResponsesCodec::new())),
			TransportId::OpenAiCodex => Ok(Self::OpenAiCodex(if provider.codex_responses_lite {
				OpenAiCodexCodec::responses_lite()
			} else {
				OpenAiCodexCodec::new()
			})),
			TransportId::AnthropicMessages => {
				Ok(Self::Anthropic(if matches!(&provider.auth, AuthSpec::OAuth { .. }) {
					AnthropicCodec::claude_oauth()
				} else {
					AnthropicCodec::new()
				}))
			},
			TransportId::AnthropicBedrock => Ok(Self::AnthropicBedrock(BedrockCodec::new())),
			TransportId::BedrockConverse => Ok(Self::BedrockConverse(BedrockConverseCodec)),
			TransportId::AnthropicVertex => Ok(Self::AnthropicVertex(VertexCodec::new())),
			TransportId::GoogleGenAi => Ok(Self::GoogleGenAi(GoogleCodec::gen_ai())),
			TransportId::GoogleVertex => Ok(Self::GoogleVertex(GoogleCodec::vertex())),
			TransportId::GoogleCca if provider.id == "google-antigravity" => {
				Ok(Self::GoogleCcaAntigravity(route.project.clone()))
			},
			TransportId::GoogleCca => Ok(Self::GoogleCca(CcaCodec::new(route.project.clone()))),
			TransportId::OllamaChat => Ok(Self::Ollama(OllamaChatCodec)),
			TransportId::Cursor
			| TransportId::Devin
			| TransportId::GitLabDuoWorkflow
			| TransportId::Omp
			| TransportId::Embedded => Err(ProviderBuildError::SpecializedTransport(provider.transport)),
		}
	}

	/// Returns the catalog transport identifier handled by this codec.
	#[must_use]
	pub const fn id(&self) -> TransportId {
		match self {
			Self::OpenAiChat(_) => TransportId::OpenAiChat,
			Self::OpenAiResponses(_) => TransportId::OpenAiResponses,
			Self::OpenAiCodex(_) => TransportId::OpenAiCodex,
			Self::Anthropic(_) => TransportId::AnthropicMessages,
			Self::AnthropicBedrock(_) => TransportId::AnthropicBedrock,
			Self::BedrockConverse(_) => TransportId::BedrockConverse,
			Self::AnthropicVertex(_) => TransportId::AnthropicVertex,
			Self::GoogleGenAi(_) => TransportId::GoogleGenAi,
			Self::GoogleVertex(_) => TransportId::GoogleVertex,
			Self::GoogleCca(_) | Self::GoogleCcaAntigravity(_) => TransportId::GoogleCca,
			Self::Ollama(_) => TransportId::OllamaChat,
		}
	}

	fn encode(
		&self,
		request: &ChatRequest,
		compat: &Compat,
	) -> Result<(Bytes, Vec<Unsupported>), ChatError> {
		match self {
			Self::OpenAiChat(codec) => codec.encode(request, compat),
			Self::OpenAiResponses(codec) => codec.encode(request, compat),
			Self::OpenAiCodex(codec) => codec.encode(request, compat),
			Self::Anthropic(codec) => codec.encode(request, compat),
			Self::AnthropicBedrock(codec) => codec.encode(request, compat),
			Self::BedrockConverse(codec) => codec.encode(request, compat),
			Self::AnthropicVertex(codec) => codec.encode(request, compat),
			Self::GoogleGenAi(codec) | Self::GoogleVertex(codec) => codec.encode(request, compat),
			Self::GoogleCca(codec) => codec.encode(request, compat),
			Self::Ollama(codec) => codec.encode(request, compat),
			Self::GoogleCcaAntigravity(_) => Err(ChatError::Provider(Str::new_static(
				"Antigravity codec requires canonical request metadata",
			))),
		}
	}

	/// Computes non-secret, request-selected HTTP headers for this codec.
	///
	/// Authentication headers are never returned here; they remain exclusively
	/// owned by the downstream egress authentication layer.
	pub fn request_headers(
		&self,
		request: &ChatRequest,
		compat: &Compat,
	) -> Result<HeaderMap, ChatError> {
		let mut headers = HeaderMap::new();
		let selected = match self {
			Self::Anthropic(_) => omp_llm_anthropic::request_headers(request, compat),
			Self::OpenAiChat(_)
			| Self::OpenAiResponses(_)
			| Self::OpenAiCodex(_)
			| Self::AnthropicBedrock(_)
			| Self::BedrockConverse(_)
			| Self::AnthropicVertex(_)
			| Self::GoogleGenAi(_)
			| Self::GoogleVertex(_)
			| Self::GoogleCca(_)
			| Self::Ollama(_)
			| Self::GoogleCcaAntigravity(_) => Vec::new(),
		};
		for selected in selected {
			let name: http::header::HeaderName = selected.name.parse().map_err(|_| {
				ChatError::Provider(fmts!(
					"provider selected an invalid HTTP header name: {}",
					selected.name
				))
			})?;
			let value: http::header::HeaderValue = selected.value.parse().map_err(|_| {
				ChatError::Provider(fmts!(
					"provider selected an invalid value for HTTP header {}",
					selected.name
				))
			})?;
			headers.insert(name, value);
		}
		Ok(headers)
	}

	fn decode(
		&self,
		frame: Frame<'_>,
		state: &mut DecodeState,
	) -> Result<SmallVec<NativeTurnEvent, 2>, ChatError> {
		match self {
			Self::OpenAiChat(codec) => codec.decode(frame, state),
			Self::OpenAiResponses(codec) => codec.decode(frame, state),
			Self::OpenAiCodex(codec) => codec.decode(frame, state),
			Self::Anthropic(codec) => codec.decode(frame, state),
			Self::AnthropicBedrock(codec) => codec.decode(frame, state),
			Self::BedrockConverse(codec) => codec.decode(frame, state),
			Self::AnthropicVertex(codec) => codec.decode(frame, state),
			Self::GoogleGenAi(codec) | Self::GoogleVertex(codec) => codec.decode(frame, state),
			Self::GoogleCca(codec) => codec.decode(frame, state),
			Self::Ollama(codec) => codec.decode(frame, state),
			Self::GoogleCcaAntigravity(_) => Err(ChatError::Provider(Str::new_static(
				"Antigravity codec snapshot was not prepared before decode",
			))),
		}
	}
}

/// Failure to construct an HTTP provider adapter from a catalog row.
#[derive(Clone, Debug, thiserror::Error, Eq, PartialEq)]
pub enum ProviderBuildError {
	/// The transport is owned by a specialized non-HTTP adapter.
	#[error("transport {0:?} requires its specialized adapter")]
	SpecializedTransport(TransportId),
	/// A catalog static header violates the selected transport's wire contract.
	#[error("transport {transport:?} forbids static header {name}")]
	ForbiddenStaticHeader {
		/// Transport whose wire contract rejected the header.
		transport: TransportId,
		/// Case-preserving rejected header name.
		name:      Str,
	},
}

/// A failure occurring before a provider attempt commits its first event.
///
/// Egress and body errors remain typed so outer policy can classify them. No
/// request, header map, URI query, or authentication material is retained.
#[derive(Debug, thiserror::Error)]
pub enum ProviderAttemptError<E, B> {
	/// The native request could not be recovered from its protobuf envelope.
	#[error("invalid provider turn request: {0}")]
	Request(omp_llm_types::ConvertError),
	/// The selected codec could not encode the request.
	#[error("provider request encoding failed: {0}")]
	Encode(ChatError),
	/// Required validated credential metadata was unavailable for this
	/// transport.
	#[error("provider credential metadata is missing {0}")]
	CredentialMetadata(&'static str),
	/// The catalog URL or static headers could not form an HTTP request.
	#[error("invalid provider HTTP request: {0}")]
	HttpRequest(http::Error),
	/// Catalog base-URL expansion failed.
	#[error("invalid provider endpoint: {0}")]
	Endpoint(Str),
	/// Egress failed before response headers arrived.
	#[error("provider egress failed: {0}")]
	Egress(E),
	/// The provider returned unsuccessful response headers.
	#[error("provider returned HTTP status {0}")]
	HttpStatus(StatusCode),
	/// The streaming body failed before the first canonical event.
	#[error("provider response body failed: {0}")]
	Body(B),
	/// A frame failed codec validation before the first canonical event.
	#[error("provider response decoding failed: {0}")]
	Decode(ChatError),
	/// A decoded provider terminal failure arrived before any output committed.
	#[error("provider rejected the attempt before commit: {0}")]
	Rejected(Str),
	/// A provider returned a classified terminal failure before commit.
	#[error("provider rejected the attempt before commit: {}", .0.detail)]
	Classified(TurnError),
	/// The response ended without a canonical event.
	#[error("provider response ended before its first event")]
	Empty,
	/// Successful response headers arrived, but the first SSE event stalled.
	#[error("provider response timed out before its first event")]
	FirstEventTimeout,
}

/// Catalog-driven service performing one routed provider attempt, including
/// catalog-defined CCA endpoint failover before the first SSE event.
///
/// The sole request contract is [`Routed`], supplied by the completed selection
/// stack so leases and validated non-secret metadata remain out-of-band from
/// [`TurnRequest`]. Its `poll_ready` and `call` methods delegate to the same
/// egress instance, allowing reservation-based services underneath it.
/// Authentication is represented only by request extensions; the downstream
/// egress stack is responsible for mutating the request in place.
#[derive(Clone)]
pub struct ProviderAttempt<S> {
	shared: Arc<ProviderShared>,
	egress: S,
}

type EndpointPolicyPair = (Arc<ResolvedModelPolicy>, Arc<ResolvedModelPolicy>);
type EndpointPolicies = HashMap<usize, EndpointPolicyPair>;
type PreparedProviderRequest = (Request<Body>, Vec<Unsupported>, Arc<HttpCodec>, Str);

struct ProviderShared {
	provider:          ProviderEntry,
	route:             ProviderRoute,
	codec:             Arc<HttpCodec>,
	endpoint_policies: Mutex<EndpointPolicies>,
}

impl<S> ProviderAttempt<S> {
	/// Constructs one attempt service from a provider catalog row.
	///
	/// Cursor, Devin, OMP federation, and embedded inference are rejected
	/// because their existing specialized adapters own those protocols.
	pub fn new(
		provider: ProviderEntry,
		route: ProviderRoute,
		egress: S,
	) -> Result<Self, ProviderBuildError> {
		if provider.transport == TransportId::AnthropicVertex
			&& let Some(name) = provider
				.headers
				.keys()
				.find(|name| name.eq_ignore_ascii_case("anthropic-beta"))
		{
			return Err(ProviderBuildError::ForbiddenStaticHeader {
				transport: provider.transport,
				name:      name.clone(),
			});
		}
		let codec = Arc::new(HttpCodec::from_catalog(&provider, &route)?);
		Ok(Self {
			shared: Arc::new(ProviderShared {
				provider,
				route,
				codec,
				endpoint_policies: Mutex::new(HashMap::new()),
			}),
			egress,
		})
	}

	/// Returns the selected provider catalog row.
	#[must_use]
	pub fn provider(&self) -> &ProviderEntry {
		&self.shared.provider
	}

	/// Returns the concrete HTTP codec selected at construction.
	#[must_use]
	pub fn codec(&self) -> &HttpCodec {
		self.shared.codec.as_ref()
	}
}

impl<S, B> Service<Routed> for ProviderAttempt<S>
where
	S: Service<Request<Body>, Response = Response<B>> + Clone + Send + 'static,
	S::Future: Send + 'static,
	S::Error: fmt::Display + Send + 'static,
	B: HttpBody<Data = Bytes> + Send + Unpin + 'static,
	B::Error: fmt::Display + Send + 'static,
{
	type Error = ProviderAttemptError<S::Error, B::Error>;
	type Response = ProviderStream<B>;

	type Future = impl Future<Output = Result<Self::Response, Self::Error>> + Send + 'static;

	fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		self
			.egress
			.poll_ready(cx)
			.map_err(ProviderAttemptError::Egress)
	}

	fn call(&mut self, routed: Routed) -> Self::Future {
		let prepared = prepare_request(&self.shared, routed);
		let replacement = self.egress.clone();
		let mut egress = std::mem::replace(&mut self.egress, replacement);
		let shared = Arc::clone(&self.shared);
		async move {
			let (mut request, unsupported, mut codec, model) = match prepared {
				Ok(prepared) => prepared,
				Err(error) => return Err(error),
			};
			let attestation_eligible = shared.provider.transport == TransportId::OpenAiCodex
				&& request
					.extensions()
					.get::<CredentialMetadata>()
					.is_some_and(|metadata| metadata.account_id.is_some());
			if attestation_eligible
				&& let Some(attestation) = CodexAttestor::default().generate().await
				&& let Ok(mut value) = http::HeaderValue::from_bytes(attestation.as_bytes())
			{
				value.set_sensitive(true);
				request
					.headers_mut()
					.insert(http::HeaderName::from_static("x-oai-attestation"), value);
			}
			let first_event_timeout = matches!(codec.as_ref(), HttpCodec::GoogleCca(_))
				.then(|| cca_first_event_timeout(model.as_str()));
			let mut fallback_attempts: SmallVec<(Request<Body>, Arc<HttpCodec>), 2> = SmallVec::new();
			if let HttpCodec::GoogleCca(cca) = codec.as_ref()
				&& cca.is_antigravity()
			{
				for endpoint in &shared.provider.fallback_base_urls {
					let url = CcaEndpointPlan::stream_url(endpoint);
					let fallback_request = clone_request_for_endpoint(&request, url.as_str())
						.map_err(ProviderAttemptError::HttpRequest)?;
					let fallback_codec = Arc::new(HttpCodec::GoogleCca(
						cca.clone().with_served_endpoint(endpoint.clone()),
					));
					fallback_attempts.push((fallback_request, fallback_codec));
				}
			}
			let mut response = egress
				.call(request)
				.await
				.map_err(ProviderAttemptError::Egress)?;
			if let Some(first_event_timeout) = first_event_timeout {
				let mut fallback_attempts = fallback_attempts.into_iter();
				loop {
					if !response.status().is_success() {
						break;
					}
					let framing = Framing::for_response(
						shared.provider.transport,
						&shared.provider.compat,
						response.headers(),
					);
					let machine = DecodeMachine::new(
						response.into_body(),
						Arc::clone(&codec),
						framing,
						unsupported.clone(),
						shared.provider.id.clone(),
						model.clone(),
					);
					match tokio::time::timeout(first_event_timeout, establish_first_wire_event(machine))
						.await
					{
						Ok(Ok(machine)) => return establish_commit(machine).await,
						Ok(Err(error)) => return Err(error),
						Err(_) => {
							let Some((fallback_request, fallback_codec)) = fallback_attempts.next() else {
								return Err(ProviderAttemptError::FirstEventTimeout);
							};
							egress.ready().await.map_err(ProviderAttemptError::Egress)?;
							response = egress
								.call(fallback_request)
								.await
								.map_err(ProviderAttemptError::Egress)?;
							codec = fallback_codec;
						},
					}
				}
			}
			if !response.status().is_success() {
				if shared.provider.transport == TransportId::AnthropicVertex {
					const MAX_ERROR_BODY: usize = 64 * 1024;
					let status = response.status();
					let mut body = response.into_body();
					let mut bytes = BytesMut::new();
					while let Some(frame) = body.frame().await {
						let frame = frame.map_err(ProviderAttemptError::Body)?;
						let Ok(data) = frame.into_data() else {
							continue;
						};
						let remaining = MAX_ERROR_BODY.saturating_sub(bytes.len());
						bytes.extend_from_slice(&data[..data.len().min(remaining)]);
						if bytes.len() == MAX_ERROR_BODY {
							break;
						}
					}
					let error = vertex::classify_error(status, &bytes);
					return Err(ProviderAttemptError::Classified(error));
				}
				if shared.provider.transport == TransportId::GoogleVertex
					&& matches!(response.status(), StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN)
				{
					let status = response.status();
					let error = TurnError::builder()
						.kind(google_vertex::classify_status(status.as_u16()))
						.detail(fmts!("Google Vertex returned HTTP status {status}"))
						.unsupported(Vec::new())
						.retry_after_ms(0)
						.build();
					return Err(ProviderAttemptError::Classified(error));
				}
				const MAX_ERROR_BODY: usize = 64 * 1024;
				let status = response.status();
				let retry_after_ms = response
					.headers()
					.get(header::RETRY_AFTER)
					.and_then(|value| value.to_str().ok())
					.and_then(|value| parse_retry_after(value, SystemTime::now()))
					.map_or(0, |duration| u64::try_from(duration.as_millis()).unwrap_or(u64::MAX));
				let mut body = response.into_body();
				let mut bytes = BytesMut::new();
				while let Some(frame) = body.frame().await {
					let frame = frame.map_err(ProviderAttemptError::Body)?;
					let Ok(data) = frame.into_data() else {
						continue;
					};
					let remaining = MAX_ERROR_BODY.saturating_sub(bytes.len());
					bytes.extend_from_slice(&data[..data.len().min(remaining)]);
					if bytes.len() == MAX_ERROR_BODY {
						break;
					}
				}
				let kind = match status {
					StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => TurnErrorKind::Auth,
					StatusCode::TOO_MANY_REQUESTS => TurnErrorKind::RateLimited,
					status if status.as_u16() == 529 => TurnErrorKind::Overloaded,
					_ => TurnErrorKind::Upstream,
				};
				let error = TurnError::builder()
					.kind(kind)
					.detail(fmts!("HTTP {status} {}", String::from_utf8_lossy(&bytes)))
					.unsupported(Vec::new())
					.retry_after_ms(retry_after_ms)
					.build();
				let mut machine = DecodeMachine::new(
					body,
					codec,
					Framing::Raw(BytesMut::new()),
					unsupported,
					shared.provider.id.clone(),
					model,
				);
				machine.body = None;
				machine.push_native([NativeTurnEvent::Error(error)]);
				return Ok(ProviderStream { machine, ended: false });
			}
			let framing = Framing::for_response(
				shared.provider.transport,
				&shared.provider.compat,
				response.headers(),
			);
			let machine = DecodeMachine::new(
				response.into_body(),
				codec,
				framing,
				unsupported,
				shared.provider.id.clone(),
				model,
			);
			establish_commit(machine).await
		}
	}
}

#[allow(
	clippy::result_large_err,
	reason = "the public attempt error keeps rich typed provider failures unboxed for \
	          classification by retry policy"
)]
fn prepare_request<E, B>(
	shared: &ProviderShared,
	routed: Routed,
) -> Result<PreparedProviderRequest, ProviderAttemptError<E, B>> {
	let Routed { request: turn_request, model_policy, lease, credential_metadata } = routed;
	let mut native = ChatRequest::try_from(turn_request).map_err(ProviderAttemptError::Request)?;
	native.model_policy = model_policy;
	let model = native.model.clone();
	let api_version = if shared.provider.id == "azure"
		&& matches!(shared.provider.transport, TransportId::OpenAiChat | TransportId::OpenAiResponses)
	{
		take_azure_api_version(&mut native)
	} else {
		None
	};
	let endpoint = endpoint(shared, model.as_str(), api_version.as_deref())
		.map_err(ProviderAttemptError::Endpoint)?;
	let proxy_computer_use_demoted =
		demote_proxy_computer_use(shared, &mut native, endpoint.as_str());
	let model_headers =
		model_headers(&native, shared.provider.transport).map_err(ProviderAttemptError::Encode)?;
	let cca_codec = match shared.codec.as_ref() {
		HttpCodec::GoogleCca(codec) => Some((codec.clone(), false, None)),
		HttpCodec::GoogleCcaAntigravity(project) => {
			let (metadata, project_override) =
				take_antigravity_metadata(&mut native).map_err(ProviderAttemptError::Encode)?;
			let served_endpoint = endpoint
				.strip_suffix(omp_llm_google::cca::STREAM_GENERATE_PATH)
				.unwrap_or(endpoint.as_str());
			let mut codec = CcaCodec::antigravity(project.clone(), metadata)
				.with_served_endpoint(Str::new(served_endpoint));
			if model.to_ascii_lowercase().contains("flash") {
				codec =
					codec.with_planning_leak_filter(native.tools.iter().map(|tool| tool.name.clone()));
			}
			Some((codec, true, project_override))
		},
		_ => None,
	};
	let codec = if let Some((mut codec, antigravity, project_override)) = cca_codec {
		if (lease.is_some() && credential_metadata.is_none()) || (antigravity && lease.is_none()) {
			return Err(ProviderAttemptError::CredentialMetadata("project_id"));
		}
		if let Some(metadata) = credential_metadata.as_ref() {
			let project = if antigravity {
				Some(
					project_override
						.or_else(|| metadata.project_id.clone())
						.ok_or(ProviderAttemptError::CredentialMetadata("project_id"))?,
				)
			} else {
				metadata.project_id.clone()
			};
			if let Some(project) = project {
				codec = codec.with_project_id(project);
			}
			codec = codec.with_identity(metadata.identity.clone());
			if let Some(account_id) = &metadata.account_id {
				codec = codec.with_account_id(account_id.clone());
			}
			if let Some(organization_id) = &metadata.organization_id {
				codec = codec.with_organization_id(organization_id.clone());
			}
		}
		Arc::new(HttpCodec::GoogleCca(codec))
	} else if let HttpCodec::Anthropic(shared_codec) = shared.codec.as_ref() {
		let tool_codec = match credential_metadata
			.as_ref()
			.map(|metadata| metadata.auth_kind)
		{
			Some(CredentialAuthKind::OAuth) => AnthropicCodec::claude_oauth(),
			Some(
				CredentialAuthKind::ApiKey | CredentialAuthKind::Aws | CredentialAuthKind::GoogleAdc,
			) => AnthropicCodec::new(),
			None => *shared_codec,
		};
		Arc::new(HttpCodec::Anthropic(tool_codec))
	} else {
		Arc::clone(&shared.codec)
	};
	let responses_lite = match codec.as_ref() {
		HttpCodec::OpenAiCodex(codec) => codec
			.request_uses_responses_lite(&native)
			.map_err(ProviderAttemptError::Encode)?,
		_ => false,
	};
	let codex_turn_state = native
		.provider_options
		.as_ref()
		.and_then(|props| props.get_ns("openai-codex", "turn_state"))
		.and_then(serde_json::Value::as_str)
		.map(Str::new);
	let (mut body, mut unsupported) = codec
		.encode(&native, &shared.provider.compat)
		.map_err(ProviderAttemptError::Encode)?;
	if proxy_computer_use_demoted {
		unsupported.push(
			Unsupported::builder()
				.what(Str::new_static("computer_use"))
				.detail(Str::new_static(
					"inferred computer-use support is not trusted through a proxy endpoint",
				))
				.action(UnsupportedAction::Dropped)
				.build(),
		);
	}
	let mut codex_marker = None;
	let mut codex_identity = None;
	if matches!(codec.as_ref(), HttpCodec::OpenAiCodex(_)) {
		let mut value: serde_json::Value = serde_json::from_slice(&body).map_err(|error| {
			ProviderAttemptError::Encode(ChatError::Provider(Str::from(error.to_string())))
		})?;
		let identity = codex_request_identity(&shared.provider, &native, &value);
		apply_codex_client_metadata(
			&mut value,
			&identity,
			CodexWireTransport::Http,
			responses_lite,
			codex_turn_state.as_deref(),
		)
		.map_err(ProviderAttemptError::Encode)?;
		body = Bytes::from(serde_json::to_vec(&value).map_err(|error| {
			ProviderAttemptError::Encode(ChatError::Provider(Str::from(error.to_string())))
		})?);
		let websocket_preferred = native
			.model_policy
			.as_deref()
			.and_then(|policy| policy.prefer_websockets)
			.unwrap_or(
				shared.provider.codex_transport == CodexTransportPreference::WebsocketPreferred,
			);
		if websocket_preferred {
			codex_marker = Some(CodexWebSocketRequest {
				session_key: codex_session_key(&shared.provider, &native, &value),
				identity: identity.clone(),
				account_id: credential_metadata
					.as_ref()
					.and_then(|metadata| metadata.account_id.clone()),
				responses_lite,
			});
		}
		codex_identity = Some(identity);
	}
	let mut dynamic_headers = codec
		.request_headers(&native, &shared.provider.compat)
		.map_err(ProviderAttemptError::Encode)?;
	if let HttpCodec::GoogleCca(codec) = codec.as_ref() {
		let reasoning = native
			.thinking
			.as_ref()
			.is_some_and(|thinking| thinking.value.effort != Some(Effort::Off));
		for (name, value) in codec
			.request_headers(native.model.as_str(), reasoning)
			.entries()
		{
			let name: http::header::HeaderName = name.parse().map_err(|_| {
				ProviderAttemptError::Encode(ChatError::Provider(fmts!(
					"CCA selected an invalid HTTP header name: {name}"
				)))
			})?;
			let value: http::header::HeaderValue = value.parse().map_err(|_| {
				ProviderAttemptError::Encode(ChatError::Provider(fmts!(
					"CCA selected an invalid value for HTTP header {name}"
				)))
			})?;
			dynamic_headers.insert(name, value);
		}
	}
	if shared.provider.id == "github-copilot" {
		apply_copilot_headers(&shared.provider, &native, &mut dynamic_headers);
	}
	let accept = match shared.provider.compat.stream_protocol {
		StreamProtocol::Ndjson => "application/x-ndjson",
		StreamProtocol::SseData | StreamProtocol::SseEvents => "text/event-stream",
		StreamProtocol::Connect => "application/octet-stream",
	};
	let mut builder = Request::post(endpoint)
		.header(header::CONTENT_TYPE, "application/json")
		.header(header::ACCEPT, accept);
	for (name, value) in &shared.provider.headers {
		builder = builder.header(name.as_str(), value.as_str());
	}
	let mut request = builder
		.body(Full::new(body))
		.map_err(ProviderAttemptError::HttpRequest)?;
	for (name, value) in &model_headers {
		request.headers_mut().insert(name.clone(), value.clone());
	}
	if shared.provider.transport == TransportId::OpenAiCodex {
		let credential = CodexCredentialMetadata {
			account_id: credential_metadata
				.as_ref()
				.and_then(|metadata| metadata.account_id.clone()),
		};
		let plan = build_codex_header_plan(&CodexHeaderContext {
			transport: CodexWireTransport::Http,
			identity: codex_identity.as_ref(),
			credential: &credential,
			attestation: None,
			turn_state: codex_turn_state.as_deref(),
			models_etag: None,
			responses_lite,
		});
		for (name, value) in plan.iter() {
			let header_name: http::HeaderName = name.parse().map_err(|_| {
				ProviderAttemptError::Encode(ChatError::Provider(fmts!(
					"Codex selected an invalid HTTP header name: {name}"
				)))
			})?;
			let mut header_value = http::HeaderValue::from_bytes(value.as_bytes()).map_err(|_| {
				ProviderAttemptError::Encode(ChatError::Provider(fmts!(
					"Codex selected an invalid value for HTTP header {name}"
				)))
			})?;
			header_value.set_sensitive(value.is_sensitive());
			request.headers_mut().insert(header_name, header_value);
		}
	}
	for (name, value) in &dynamic_headers {
		request.headers_mut().insert(name.clone(), value.clone());
	}
	if matches!(
		shared.provider.transport,
		TransportId::AnthropicBedrock | TransportId::BedrockConverse
	) {
		let region = bedrock::resolve_region(
			shared.route.region.as_str(),
			model.as_str(),
			shared.provider.base_url.as_str(),
		);
		bedrock::attach_sigv4(&mut request, region, SystemTime::now());
	}
	if shared.provider.transport == TransportId::AnthropicVertex {
		vertex::attach_adc(&mut request, shared.provider.id.as_str());
	} else {
		request
			.extensions_mut()
			.insert(AuthContext::new(shared.provider.id.as_str()));
	}
	if let Some(lease) = lease {
		request.extensions_mut().insert(lease);
	}
	if let Some(marker) = codex_marker {
		request.extensions_mut().insert(marker);
	}
	if let Some(metadata) = credential_metadata {
		request.extensions_mut().insert(metadata);
	}
	Ok((request, unsupported, codec, model))
}

fn clone_request_for_endpoint(
	request: &Request<Body>,
	endpoint: &str,
) -> Result<Request<Body>, http::Error> {
	let mut cloned = Request::builder()
		.method(request.method().clone())
		.version(request.version())
		.uri(endpoint)
		.body(request.body().clone())?;
	*cloned.headers_mut() = request.headers().clone();
	*cloned.extensions_mut() = request.extensions().clone();
	Ok(cloned)
}

fn model_headers(request: &ChatRequest, transport: TransportId) -> Result<HeaderMap, ChatError> {
	let mut headers = HeaderMap::new();
	let Some(policy) = request.model_policy.as_deref() else {
		return Ok(headers);
	};
	for (raw_name, raw_value) in policy.headers.iter() {
		let name: http::header::HeaderName = raw_name.parse().map_err(|_| {
			ChatError::Provider(fmts!("model policy contains an invalid HTTP header name: {raw_name}"))
		})?;
		// Bedrock caller headers are added to the request before the sealed egress
		// layer signs it. Drop fields whose wire value is generated or replaced by
		// SigV4/fetch so the signer cannot cover bytes different from those sent.
		if matches!(transport, TransportId::AnthropicBedrock | TransportId::BedrockConverse)
			&& bedrock_transport_owned_header(name.as_str())
		{
			continue;
		}
		if unsafe_model_header(name.as_str()) {
			return Err(ChatError::Provider(fmts!(
				"model policy forbids unsafe HTTP header: {raw_name}"
			)));
		}
		let value: http::header::HeaderValue = raw_value.parse().map_err(|_| {
			ChatError::Provider(fmts!(
				"model policy contains an invalid value for HTTP header {raw_name}"
			))
		})?;
		headers.insert(name, value);
	}
	Ok(headers)
}

fn bedrock_transport_owned_header(name: &str) -> bool {
	matches!(
		name,
		"host"
			| "x-amz-date"
			| "x-amz-content-sha256"
			| "x-amz-security-token"
			| "content-length"
			| "content-type"
			| "accept"
			| "authorization"
	)
}

fn unsafe_model_header(name: &str) -> bool {
	let name = name.to_ascii_lowercase();
	matches!(
		name.as_str(),
		"authorization"
			| "proxy-authorization"
			| "proxy-authenticate"
			| "www-authenticate"
			| "cookie"
			| "set-cookie"
			| "connection"
			| "keep-alive"
			| "te" | "trailer"
			| "transfer-encoding"
			| "upgrade"
			| "host"
			| "content-length"
			| "content-type"
			| "chatgpt-account-id"
			| "openai-organization"
			| "openai-project"
			| "x-goog-user-project"
			| "x-goog-quota-project"
			| "content-encoding"
			| "content-range"
			| "expect"
			| "x-api-key"
			| "api-key"
			| "x-goog-api-key"
			| "x-amz-security-token"
			| "x-aws-security-token"
			| "private-token"
	) || name.ends_with("-api-key")
		|| name.ends_with("-access-token")
		|| name.ends_with("-auth-token")
		|| name.contains("credential")
		|| name.contains("secret")
}

fn demote_proxy_computer_use(
	shared: &ProviderShared,
	request: &mut ChatRequest,
	endpoint: &str,
) -> bool {
	let Some(policy) = request.model_policy.as_ref() else {
		return false;
	};
	if policy.capabilities.computer_use != Some(true)
		|| policy.capabilities.computer_use_config == Some(true)
		|| verified_computer_use_endpoint(endpoint)
	{
		return false;
	}
	let key = Arc::as_ptr(policy) as usize;
	let demoted = {
		let mut endpoint_policies = shared.endpoint_policies.lock();
		let (_, demoted) = endpoint_policies.entry(key).or_insert_with(|| {
			let mut demoted = policy.as_ref().clone();
			demoted.capabilities.computer_use = Some(false);
			(Arc::clone(policy), Arc::new(demoted))
		});
		Arc::clone(demoted)
	};
	request.model_policy = Some(demoted);
	true
}

fn verified_computer_use_endpoint(endpoint: &str) -> bool {
	let Ok(uri) = endpoint.parse::<http::Uri>() else {
		return false;
	};
	let Some(host) = uri.host() else {
		return false;
	};
	let host = host.to_ascii_lowercase();
	matches!(
		host.as_str(),
		"api.anthropic.com" | "api.openai.com" | "chatgpt.com" | "generativelanguage.googleapis.com"
	) || host.ends_with(".openai.azure.com")
		|| host.ends_with("-aiplatform.googleapis.com")
		|| host.ends_with(".bedrock-runtime.amazonaws.com")
		|| host.ends_with(".api.aws")
}

fn codex_session_key(
	provider: &ProviderEntry,
	request: &ChatRequest,
	body: &serde_json::Value,
) -> Str {
	if let Some(key) = request
		.cache
		.as_ref()
		.map(|cache| cache.session_key.as_str())
		.filter(|key| !key.is_empty())
	{
		return Str::new(key);
	}
	if let Some(key) = request
		.meta
		.as_ref()
		.map(|meta| meta.session_id.as_str())
		.filter(|key| !key.is_empty())
	{
		return Str::new(key);
	}
	let mut hasher = blake3::Hasher::new();
	hasher.update(provider.id.as_bytes());
	hasher.update(&[0]);
	hasher.update(request.model.as_bytes());
	hasher.update(&[0]);
	hasher.update(serde_json::to_string(body).unwrap_or_default().as_bytes());
	fmts!("codex_ephemeral_{}", hasher.finalize().to_hex())
}

fn codex_request_identity(
	provider: &ProviderEntry,
	request: &ChatRequest,
	body: &serde_json::Value,
) -> CodexRequestIdentity {
	let key = codex_session_key(provider, request, body);
	let installation_id = codex_installation_id();
	let session_id = codex_uuid("session", key.as_bytes());
	let thread_id = codex_uuid("thread", key.as_bytes());
	let window_id = codex_uuid("window", key.as_bytes());
	let body_bytes = serde_json::to_vec(body).unwrap_or_default();
	let turn_id = codex_uuid("turn", &body_bytes);
	let turn_metadata = Str::from(
		serde_json::json!({
			"request_kind": "turn",
			"session_id": session_id.as_str(),
			"thread_id": thread_id.as_str(),
			"turn_id": turn_id.as_str(),
			"window_id": window_id.as_str(),
		})
		.to_string(),
	);
	CodexRequestIdentity {
		installation_id,
		session_id,
		thread_id,
		window_id,
		turn_id,
		turn_metadata,
	}
}

fn codex_installation_id() -> Str {
	static INSTALLATION_ID: LazyLock<Str> = LazyLock::new(|| {
		let created = SystemTime::now()
			.duration_since(UNIX_EPOCH)
			.unwrap_or_default()
			.as_nanos();
		let seed = fmts!("{}:{created}", std::process::id());
		codex_uuid("installation", seed.as_bytes())
	});
	(*INSTALLATION_ID).clone()
}

fn codex_uuid(domain: &str, value: &[u8]) -> Str {
	let mut hasher = blake3::Hasher::new();
	hasher.update(domain.as_bytes());
	hasher.update(&[0]);
	hasher.update(value);
	let mut bytes = [0_u8; 16];
	bytes.copy_from_slice(&hasher.finalize().as_bytes()[..16]);
	bytes[6] = (bytes[6] & 0x0f) | 0x40;
	bytes[8] = (bytes[8] & 0x3f) | 0x80;
	fmts!(
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
}
fn take_antigravity_metadata(
	request: &mut ChatRequest,
) -> Result<(AntigravityRequestMetadata, Option<Str>), ChatError> {
	const NAMESPACE: &str = "google-antigravity";
	const KEYS: [&str; 6] =
		["agent_id", "request_id", "trajectory_id", "step_index", "last_execution_id", "model_enum"];
	let session_id = request
		.meta
		.as_ref()
		.map(|meta| meta.session_id.clone())
		.filter(|value| {
			value.as_str().strip_prefix('-').is_some_and(|digits| {
				!digits.is_empty() && digits.bytes().all(|byte| byte.is_ascii_digit())
			})
		})
		.ok_or_else(|| antigravity_metadata_error("RequestMeta.session_id"))?;
	let options = request
		.provider_options
		.as_ref()
		.ok_or_else(|| antigravity_metadata_error("provider_options"))?;
	let string = |key: &'static str| {
		options
			.get_ns(NAMESPACE, key)
			.and_then(serde_json::Value::as_str)
			.filter(|value| !value.is_empty())
			.map(Str::new)
			.ok_or_else(|| antigravity_metadata_error(key))
	};
	let agent_id = string("agent_id")?;
	let request_id = string("request_id")?;
	let trajectory_id = string("trajectory_id")?;
	let step_index = options
		.get_ns(NAMESPACE, "step_index")
		.and_then(serde_json::Value::as_u64)
		.filter(|step| *step >= 2)
		.ok_or_else(|| antigravity_metadata_error("step_index"))?;
	let last_execution_id = options
		.get_ns(NAMESPACE, "last_execution_id")
		.and_then(serde_json::Value::as_str)
		.filter(|value| !value.is_empty())
		.map(Str::new);
	let model_enum = options
		.get_ns(NAMESPACE, "model_enum")
		.and_then(serde_json::Value::as_str)
		.filter(|value| !value.is_empty())
		.map(Str::new);
	let project_override = options
		.get_ns("antigravity", "project_id")
		.and_then(serde_json::Value::as_str)
		.filter(|value| !value.is_empty())
		.map(Str::new);

	let mut parts = request_id.as_str().split('/');
	let valid_request_id = parts.next() == Some("agent")
		&& parts.next() == Some(agent_id.as_str())
		&& parts.next().is_some_and(|millis| {
			!millis.is_empty() && millis.bytes().all(|byte| byte.is_ascii_digit())
		}) && parts.next() == Some(trajectory_id.as_str())
		&& parts.next().and_then(|step| step.parse::<u64>().ok()) == Some(step_index)
		&& parts.next().is_none();
	if !valid_request_id {
		return Err(antigravity_metadata_error("request_id"));
	}

	let mut metadata =
		AntigravityRequestMetadata::new(session_id, request_id, trajectory_id, step_index);
	if let Some(last_execution_id) = last_execution_id {
		metadata = metadata.with_last_execution_id(last_execution_id);
	}
	if let Some(model_enum) = model_enum {
		metadata = metadata.with_model_enum(model_enum);
	}
	let empty = {
		let options = request
			.provider_options
			.as_mut()
			.expect("validated provider options above");
		for key in KEYS {
			options.0.remove(fmts!("{NAMESPACE}/{key}").as_str());
		}
		options.0.remove("antigravity/project_id");
		options.is_empty()
	};
	if empty {
		request.provider_options = None;
	}
	Ok((metadata, project_override))
}

fn antigravity_metadata_error(field: &'static str) -> ChatError {
	ChatError::Provider(fmts!("invalid Google Antigravity metadata field: {field}"))
}

fn endpoint(
	shared: &ProviderShared,
	model: &str,
	api_version_override: Option<&str>,
) -> Result<String, Str> {
	let deployment = if shared.route.deployment.is_empty() {
		model
	} else {
		shared.route.deployment.as_str()
	};
	let bedrock_region = matches!(
		shared.provider.transport,
		TransportId::AnthropicBedrock | TransportId::BedrockConverse
	)
	.then(|| {
		bedrock::resolve_region(
			shared.route.region.as_str(),
			model,
			shared.provider.base_url.as_str(),
		)
	});
	let region = bedrock_region
		.as_deref()
		.unwrap_or(shared.route.region.as_str());
	let base = expand_base_url(
		&shared.provider.base_url,
		BaseUrlVars::builder()
			.region(region)
			.location(region)
			.project(shared.route.project.as_str())
			.deployment(deployment)
			.model(model)
			.account(shared.route.account.as_str())
			.gateway(shared.route.gateway.as_str())
			.build(),
	)
	.map_err(|error| fmts!("{error}"))?;
	let base = base.trim_end_matches('/');
	let suffix = match shared.provider.transport {
		TransportId::OpenAiChat => "/chat/completions".to_owned(),
		TransportId::OpenAiResponses => "/responses".to_owned(),
		TransportId::OpenAiCodex => "/codex/responses".to_owned(),
		TransportId::AnthropicMessages => "/v1/messages".to_owned(),
		TransportId::AnthropicBedrock => {
			return bedrock::endpoint(base, region, model)
				.map(|endpoint| endpoint.to_string())
				.map_err(|error| fmts!("{error}"));
		},
		TransportId::BedrockConverse => {
			return bedrock::converse_endpoint(base, region, model)
				.map(|endpoint| endpoint.to_string())
				.map_err(|error| fmts!("{error}"));
		},
		TransportId::AnthropicVertex => {
			return vertex::endpoint(
				base,
				shared.route.project.as_str(),
				shared.route.region.as_str(),
				model,
			)
			.map(|endpoint| endpoint.to_string())
			.map_err(|error| fmts!("{error}"));
		},
		TransportId::GoogleGenAi => format!("/models/{model}:streamGenerateContent?alt=sse"),
		TransportId::GoogleVertex => {
			return google_vertex::vertex_stream_url(
				&shared.provider,
				shared.route.project.as_str(),
				shared.route.region.as_str(),
				model,
			)
			.map(|endpoint| endpoint.to_string())
			.map_err(|error| fmts!("{error}"));
		},
		TransportId::GoogleCca => omp_llm_google::cca::STREAM_GENERATE_PATH.to_owned(),
		TransportId::OllamaChat => if base.ends_with("/api") {
			"/chat"
		} else {
			"/api/chat"
		}
		.to_owned(),
		TransportId::Cursor
		| TransportId::Devin
		| TransportId::GitLabDuoWorkflow
		| TransportId::Omp
		| TransportId::Embedded => {
			unreachable!("non-HTTP transports are rejected during construction")
		},
	};
	if shared.provider.id == "azure"
		&& matches!(shared.provider.transport, TransportId::OpenAiChat | TransportId::OpenAiResponses)
	{
		return Ok(azure_endpoint(
			base,
			suffix.as_str(),
			api_version_override.or(shared.provider.api_version.as_deref()),
		));
	}
	Ok(format!("{base}{suffix}"))
}

fn take_azure_api_version(request: &mut ChatRequest) -> Option<Str> {
	let options = request.provider_options.as_mut()?;
	let value = options.0.remove("azure/api_version")?;
	match value {
		serde_json::Value::String(value) if !value.is_empty() => Some(value.into()),
		value => {
			options.0.insert("azure/api_version".into(), value);
			None
		},
	}
}

fn azure_endpoint(base: &str, suffix: &str, api_version: Option<&str>) -> String {
	let (base, query) = base
		.split_once('?')
		.map_or((base, None), |(base, query)| (base, Some(query)));
	let mut endpoint = format!("{}{suffix}", base.trim_end_matches('/'));
	if let Some(query) = query.filter(|query| !query.is_empty()) {
		endpoint.push('?');
		endpoint.push_str(query);
	}
	if endpoint.split_once('?').is_some_and(|(_, query)| {
		query.split('&').any(|field| {
			field
				.split_once('=')
				.map_or(field, |(name, _)| name)
				.eq_ignore_ascii_case("api-version")
		})
	}) {
		return endpoint;
	}
	let Some(api_version) = api_version.filter(|version| !version.is_empty()) else {
		return endpoint;
	};
	endpoint.push(if endpoint.contains('?') { '&' } else { '?' });
	endpoint.push_str("api-version=");
	push_query_component(&mut endpoint, api_version);
	endpoint
}

fn push_query_component(output: &mut String, value: &str) {
	const HEX: &[u8; 16] = b"0123456789ABCDEF";
	for &byte in value.as_bytes() {
		if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
			output.push(char::from(byte));
		} else {
			output.push('%');
			output.push(char::from(HEX[usize::from(byte >> 4)]));
			output.push(char::from(HEX[usize::from(byte & 0x0f)]));
		}
	}
}

fn apply_copilot_headers(provider: &ProviderEntry, request: &ChatRequest, headers: &mut HeaderMap) {
	let explicit = request
		.meta
		.as_ref()
		.and_then(|meta| copilot_initiator(meta.initiator.as_str()))
		.or_else(|| {
			provider
				.headers
				.iter()
				.rev()
				.find(|(name, _)| name.eq_ignore_ascii_case("x-initiator"))
				.and_then(|(_, value)| copilot_initiator(value.as_str()))
		});
	let initiator = explicit.unwrap_or_else(|| match request.thread.items.last() {
		Some(item) if matches!(&item.kind, ItemKind::Message(message) if message.role == omp_llm_types::Role::User) => {
			"user"
		},
		Some(_) => "agent",
		None => "user",
	});
	headers.insert("x-initiator", http::HeaderValue::from_static(initiator));
	headers.insert("openai-intent", http::HeaderValue::from_static("conversation-edits"));
	if request.thread.items.iter().any(|item| {
		let parts = match &item.kind {
			ItemKind::Message(message) if message.role == omp_llm_types::Role::User => {
				Some(message.parts.as_slice())
			},
			ItemKind::ToolResult(result) => Some(result.parts.as_slice()),
			_ => None,
		};
		parts.is_some_and(|parts| {
			parts
				.iter()
				.any(|part| matches!(part, Part::Blob(blob) if blob.mime.starts_with("image/")))
		})
	}) {
		headers.insert("copilot-vision-request", http::HeaderValue::from_static("true"));
	}
}

fn copilot_initiator(value: &str) -> Option<&'static str> {
	let value = value.trim();
	if value.eq_ignore_ascii_case("user") {
		Some("user")
	} else if value.eq_ignore_ascii_case("agent") {
		Some("agent")
	} else {
		None
	}
}

#[derive(Default)]
enum Framing {
	#[default]
	Sse,
	Ndjson(NdjsonDecoder),
	Bedrock(BedrockEventStreamDecoder),
	Raw(BytesMut),
}

impl Framing {
	fn for_response(transport: TransportId, compat: &Compat, headers: &http::HeaderMap) -> Self {
		if matches!(transport, TransportId::AnthropicBedrock | TransportId::BedrockConverse) {
			return Self::Bedrock(BedrockEventStreamDecoder::new());
		}
		let content_type = headers
			.get(header::CONTENT_TYPE)
			.and_then(|value| value.to_str().ok())
			.unwrap_or("")
			.to_ascii_lowercase();
		if content_type.contains("application/x-ndjson")
			|| content_type.contains("application/ndjson")
		{
			return Self::Ndjson(NdjsonDecoder::new());
		}
		if content_type.contains("application/json") && !content_type.contains("event-stream") {
			return Self::Raw(BytesMut::new());
		}
		match compat.stream_protocol {
			StreamProtocol::Ndjson => Self::Ndjson(NdjsonDecoder::new()),
			StreamProtocol::SseData | StreamProtocol::SseEvents => Self::Sse,
			StreamProtocol::Connect => Self::Raw(BytesMut::new()),
		}
	}
}

struct DecodeMachine<B> {
	body:              Option<B>,
	codec:             Arc<HttpCodec>,
	framing:           FramingState,
	state:             DecodeState,
	queue:             VecDeque<TurnEvent>,
	unsupported:       Vec<Unsupported>,
	selected_provider: Str,
	selected_model:    Str,
	terminal:          bool,
	wire_event_seen:   bool,
}

enum FramingState {
	Sse(SseDecoder),
	Ndjson(NdjsonDecoder),
	Raw(BytesMut),
	Bedrock(BedrockEventStreamDecoder),
}

impl<B> DecodeMachine<B> {
	fn new(
		body: B,
		codec: Arc<HttpCodec>,
		framing: Framing,
		unsupported: Vec<Unsupported>,
		selected_provider: Str,
		selected_model: Str,
	) -> Self {
		let framing = match framing {
			Framing::Sse => FramingState::Sse(SseDecoder::new()),
			Framing::Ndjson(decoder) => FramingState::Ndjson(decoder),
			Framing::Raw(buffer) => FramingState::Raw(buffer),
			Framing::Bedrock(decoder) => FramingState::Bedrock(decoder),
		};
		Self {
			body: Some(body),
			codec,
			framing,
			state: DecodeState::default(),
			queue: VecDeque::new(),
			unsupported,
			selected_provider,
			selected_model,
			terminal: false,
			wire_event_seen: false,
		}
	}

	fn push_chunk(&mut self, chunk: Bytes) -> Result<(), ChatError> {
		enum WireFrame {
			Data(Bytes),
			Event(Option<Str>, Bytes),
		}
		let (frames, done): (SmallVec<WireFrame, 4>, bool) = match &mut self.framing {
			FramingState::Sse(decoder) => {
				let frames = decoder
					.push(chunk)
					.map(|event| WireFrame::Event(event.name, event.data))
					.collect();
				(frames, decoder.is_done())
			},
			FramingState::Ndjson(decoder) => {
				(decoder.push(chunk).map(WireFrame::Data).collect(), false)
			},
			FramingState::Raw(buffer) => {
				buffer.extend_from_slice(&chunk);
				(SmallVec::new(), false)
			},
			FramingState::Bedrock(decoder) => {
				let frames = decoder
					.push(chunk)?
					.into_iter()
					.map(WireFrame::Data)
					.collect();
				(frames, decoder.is_terminal())
			},
		};
		if !frames.is_empty() {
			self.wire_event_seen = true;
		}
		for frame in frames {
			let events = match &frame {
				WireFrame::Data(data) => self.codec.decode(Frame::Data(data), &mut self.state)?,
				WireFrame::Event(name, data) => self
					.codec
					.decode(Frame::Event { name: name.as_deref(), data }, &mut self.state)?,
			};
			self.push_native(events);
			if self.terminal {
				break;
			}
		}
		if done && !self.terminal {
			let events = self.codec.decode(Frame::Done, &mut self.state)?;
			self.push_native(events);
		}
		Ok(())
	}

	fn finish(&mut self) -> Result<(), ChatError> {
		if self.terminal {
			return Ok(());
		}
		if let FramingState::Raw(buffer) = &mut self.framing
			&& !buffer.is_empty()
		{
			let data = buffer.split().freeze();
			let events = self.codec.decode(Frame::Data(&data), &mut self.state)?;
			self.push_native(events);
		}
		if let FramingState::Bedrock(decoder) = &mut self.framing {
			decoder.finish()?;
		}
		if !self.terminal {
			let events = self.codec.decode(Frame::Done, &mut self.state)?;
			self.push_native(events);
		}
		Ok(())
	}

	fn push_native(&mut self, events: impl IntoIterator<Item = NativeTurnEvent>) {
		for mut event in events {
			if self.terminal {
				break;
			}
			let terminal = match &mut event {
				NativeTurnEvent::Outcome(outcome) => {
					outcome.unsupported.append(&mut self.unsupported);
					if outcome.provider.is_empty() {
						outcome.provider = self.selected_provider.clone();
					}
					if outcome.model.is_empty() {
						outcome.model = self.selected_model.clone();
					}
					true
				},
				NativeTurnEvent::Error(_) => true,
				_ => false,
			};
			self.queue.push_back(event.into());
			if terminal {
				self.terminal = true;
				self.body = None;
			}
		}
	}

	fn fail_in_band(&mut self, detail: impl fmt::Display) {
		if self.terminal {
			return;
		}
		let error = TurnError::builder()
			.kind(TurnErrorKind::Upstream)
			.detail(fmts!("{detail}"))
			.unsupported(Vec::new())
			.retry_after_ms(0)
			.build();
		self.push_native([NativeTurnEvent::Error(error)]);
	}
}

#[allow(
	clippy::result_large_err,
	reason = "the public attempt error keeps rich typed provider failures unboxed for \
	          classification by retry policy"
)]
async fn establish_first_wire_event<E, B>(
	mut machine: DecodeMachine<B>,
) -> Result<DecodeMachine<B>, ProviderAttemptError<E, B::Error>>
where
	B: HttpBody<Data = Bytes> + Send + Unpin + 'static,
	B::Error: fmt::Display + Send + 'static,
{
	loop {
		if machine.wire_event_seen {
			return Ok(machine);
		}
		let frame = futures::future::poll_fn(|cx| {
			let body = machine
				.body
				.as_mut()
				.expect("body remains before first event");
			Pin::new(body).poll_frame(cx)
		})
		.await;
		match frame {
			Some(Ok(frame)) => {
				if let Ok(data) = frame.into_data() {
					machine
						.push_chunk(data)
						.map_err(ProviderAttemptError::Decode)?;
				}
			},
			Some(Err(error)) => return Err(ProviderAttemptError::Body(error)),
			None => return Err(ProviderAttemptError::Empty),
		}
	}
}

#[allow(
	clippy::result_large_err,
	reason = "the public attempt error keeps rich typed provider failures unboxed for \
	          classification by retry policy"
)]
async fn establish_commit<E, B>(
	mut machine: DecodeMachine<B>,
) -> Result<ProviderStream<B>, ProviderAttemptError<E, B::Error>>
where
	B: HttpBody<Data = Bytes> + Send + Unpin + 'static,
	B::Error: fmt::Display + Send + 'static,
{
	loop {
		if machine.queue.front().is_some() {
			return Ok(ProviderStream { machine, ended: false });
		}
		let frame = futures::future::poll_fn(|cx| {
			let body = machine.body.as_mut().expect("body remains before commit");
			Pin::new(body).poll_frame(cx)
		})
		.await;
		match frame {
			Some(Ok(frame)) => {
				if let Ok(data) = frame.into_data() {
					machine
						.push_chunk(data)
						.map_err(ProviderAttemptError::Decode)?;
				}
			},
			Some(Err(error)) => return Err(ProviderAttemptError::Body(error)),
			None => {
				machine.finish().map_err(ProviderAttemptError::Decode)?;
				if machine.queue.is_empty() {
					return Err(ProviderAttemptError::Empty);
				}
			},
		}
	}
}

/// Concrete provider event stream.
///
/// A first terminal error remains repairable pre-commit while the outer route
/// policy inspects it. The stream owns the live HTTP body; dropping it cancels
/// further reads immediately.
pub struct ProviderStream<B> {
	machine: DecodeMachine<B>,
	ended:   bool,
}

impl<B> Stream for ProviderStream<B>
where
	B: HttpBody<Data = Bytes> + Unpin,
	B::Error: fmt::Display,
{
	type Item = TurnEvent;

	fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
		loop {
			if let Some(event) = self.machine.queue.pop_front() {
				if matches!(
					event.event,
					Some(
						omp_proto::inference::v1::turn_event::Event::Outcome(_)
							| omp_proto::inference::v1::turn_event::Event::Error(_)
					)
				) {
					self.ended = true;
				}
				return Poll::Ready(Some(event));
			}
			if self.ended || self.machine.body.is_none() {
				return Poll::Ready(None);
			}
			let poll = {
				let body = self.machine.body.as_mut().expect("checked above");
				Pin::new(body).poll_frame(cx)
			};
			match poll {
				Poll::Pending => return Poll::Pending,
				Poll::Ready(Some(Ok(frame))) => {
					if let Ok(data) = frame.into_data()
						&& let Err(error) = self.machine.push_chunk(data)
					{
						self.machine.fail_in_band(error);
					}
				},
				Poll::Ready(Some(Err(error))) => self.machine.fail_in_band(error),
				Poll::Ready(None) => {
					if let Err(error) = self.machine.finish() {
						self.machine.fail_in_band(error);
					} else if !self.machine.terminal {
						self
							.machine
							.fail_in_band("provider response ended without a terminal event");
					}
				},
			}
		}
	}
}

/// Object-safe native [`Chat`] facade over a once-built typed turn service.
///
/// The mutex supplies the mutable Tower receiver without cloning a service
/// between readiness and dispatch. Native/protobuf conversion occurs once at
/// each edge; only the public `Chat`/`BoxStream` cold-I/O boundary is erased.
pub struct ServiceChat<S> {
	service: tokio::sync::Mutex<S>,
}

impl<S> ServiceChat<S> {
	/// Wraps a fully composed `Service<ProviderRequest>` as a native chat facet.
	#[must_use]
	pub fn new(service: S) -> Self {
		Self { service: tokio::sync::Mutex::new(service) }
	}

	/// Recovers the built Tower service.
	pub fn into_inner(self) -> S {
		self.service.into_inner()
	}
}

#[async_trait::async_trait]
impl<S, St> Chat for ServiceChat<S>
where
	S: Service<crate::envelope::ProviderRequest, Response = St> + Send,
	S::Future: Send,
	S::Error: fmt::Display + Send,
	St: Stream<Item = TurnEvent> + Send + 'static,
{
	async fn turn(
		&self,
		request: ChatRequest,
		_executor: Option<Arc<dyn Executor>>,
	) -> Result<futures::stream::BoxStream<'static, NativeTurnEvent>, ChatError> {
		let model_policy = request.model_policy.clone();
		let request = crate::envelope::ProviderRequest::new(TurnRequest::from(request), model_policy);
		let mut service = self.service.lock().await;
		let stream = service
			.ready()
			.await
			.map_err(|error| ChatError::Transport(fmts!("{error}")))?
			.call(request)
			.await
			.map_err(|error| ChatError::Transport(fmts!("{error}")))?;
		Ok(NativeTurnStream { inner: stream, ended: false }.boxed())
	}
}

pin_project_lite::pin_project! {
	struct NativeTurnStream<St> {
		#[pin]
		inner: St,
		ended: bool,
	}
}

impl<St> Stream for NativeTurnStream<St>
where
	St: Stream<Item = TurnEvent>,
{
	type Item = NativeTurnEvent;

	fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
		let mut this = self.project();
		if *this.ended {
			return Poll::Ready(None);
		}
		match this.inner.as_mut().poll_next(cx) {
			Poll::Pending => Poll::Pending,
			Poll::Ready(None) => Poll::Ready(None),
			Poll::Ready(Some(event)) => match NativeTurnEvent::try_from(event) {
				Ok(event) => {
					*this.ended =
						matches!(event, NativeTurnEvent::Outcome(_) | NativeTurnEvent::Error(_));
					Poll::Ready(Some(event))
				},
				Err(error) => {
					*this.ended = true;
					Poll::Ready(Some(NativeTurnEvent::Error(
						TurnError::builder()
							.kind(TurnErrorKind::Upstream)
							.detail(fmts!("invalid canonical provider event: {error}"))
							.unsupported(Vec::new())
							.retry_after_ms(0)
							.build(),
					)))
				},
			},
		}
	}
}
