//! Catalog-driven routing for native inference facets.

use std::{collections::BTreeMap, fmt, sync::Arc};

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use http::{Request, Response, header};
use http_body_util::BodyExt;
use hyper::body::Body as HttpBody;
use omp_core::{Str, fmts};
use omp_llm_anthropic::AnthropicCodec;
use omp_llm_catalog::{
	models::{ModelCard, ModelCatalog, PriceUnit},
	provider::{
		BaseUrlVars, Facet as CatalogFacet, ProviderCatalog, ProviderEntry, TransportId,
		expand_base_url,
	},
};
use omp_llm_egress::{auth_inject::AuthContext, client::Body};
use omp_llm_google::embeddings::{self as google_embeddings, GoogleEmbeddingVariant};
use omp_llm_openai::embeddings as openai_embeddings;
use omp_llm_transport::embedded::{Embedded, LocalEngine};
use omp_llm_types::{
	Accuracy, Chat, ChatRequest, CountInput, CountRequest, CountResponse, Embed, EmbedRequest,
	EmbedResponse, Executor, Facets, ItemKind, Part, Speak, SpeakEvent, SpeakRequest, Thread,
	Transcribe, TranscribeRequest, TranscribeResponse, TurnEvent, Unsupported, UnsupportedAction,
	Usage,
	facet::{CountTokens, Error},
};
use serde_json::Value;
use smallvec::SmallVec;
use tower::{Service, ServiceExt};

use super::tokenizer::{OpenAiTokenizer, Tokenizer};
use crate::provider::ProviderRoute;

/// One ordered, already-built provider implementation.
///
/// A candidate may implement any subset of native facets. Registration order
/// is preserved and the first candidate implementing a requested facet wins.
#[derive(Clone)]
pub struct ProviderCandidate {
	/// Native implementations exposed by this provider attempt.
	pub facets: Facets,
}

/// Catalog-checked router shared by remote and embedded provider candidates.
///
/// Local engines enter through [`Self::register_local`], which wraps the engine
/// in [`Embedded`] and installs the same canonical facet traits used by remote
/// transports. Requests are rewritten from their catalog selector to the
/// provider-local model id immediately before dispatch.
pub struct ProviderRouter {
	models:     Arc<ModelCatalog>,
	providers:  Arc<ProviderCatalog>,
	candidates: BTreeMap<Str, SmallVec<ProviderCandidate, 2>>,
}

impl ProviderRouter {
	/// Creates an empty candidate registry over immutable catalogs.
	#[must_use]
	pub fn new(models: Arc<ModelCatalog>, providers: Arc<ProviderCatalog>) -> Self {
		Self { models, providers, candidates: BTreeMap::new() }
	}

	/// Appends one already-built provider candidate.
	pub fn register(&mut self, provider: impl Into<Str>, candidate: ProviderCandidate) {
		self
			.candidates
			.entry(provider.into())
			.or_default()
			.push(candidate);
	}

	/// Registers one embedded local engine for chat, embeddings, speech, and
	/// transcription.
	pub fn register_local<E: LocalEngine>(&mut self, provider: impl Into<Str>, engine: Arc<E>) {
		let transport = Arc::new(Embedded::new(engine));
		self.register(provider, ProviderCandidate {
			facets: Facets {
				chat: Some(transport.clone()),
				embed: Some(transport.clone()),
				speak: Some(transport.clone()),
				transcribe: Some(transport),
				..Facets::default()
			},
		});
	}

	fn resolve(
		&self,
		selector: &str,
		facet: CatalogFacet,
	) -> Result<(&ModelCard, &ProviderEntry, &[ProviderCandidate]), Error> {
		let (provider_id, model_id) = selector
			.split_once('/')
			.ok_or_else(|| Error::Provider(Str::from("model selector must be provider/model")))?;
		let model = self
			.models
			.get(provider_id, model_id)
			.ok_or_else(|| Error::Provider(fmts!("unknown catalog model {selector}")))?;
		let provider = self.providers.get(model.provider.as_str()).ok_or_else(|| {
			Error::Provider(fmts!("unknown catalog provider {}", model.provider))
		})?;
		if !model.facets.contains(&facet) || !provider.facets.contains(&facet) {
			return Err(Error::Unsupported(vec![unsupported(
				"model",
				"the catalog route does not advertise the requested facet",
			)]));
		}
		let candidates = self.candidates.get(provider.id.as_str()).ok_or_else(|| {
			Error::Provider(fmts!("provider route unavailable for {}", provider.id))
		})?;
		Ok((model, provider, candidates))
	}
}

#[async_trait]
impl Chat for ProviderRouter {
	async fn turn(
		&self,
		mut request: ChatRequest,
		executor: Option<Arc<dyn Executor>>,
	) -> Result<futures::stream::BoxStream<'static, TurnEvent>, Error> {
		let (model, _, candidates) = self.resolve(request.model.as_str(), CatalogFacet::Chat)?;
		request.model = model.model.clone();
		let backend = candidates
			.iter()
			.find_map(|candidate| candidate.facets.chat.clone())
			.ok_or_else(|| Error::Provider(Str::from("chat candidate unavailable")))?;
		backend.turn(request, executor).await
	}
}

#[async_trait]
impl Embed for ProviderRouter {
	async fn embed(&self, mut request: EmbedRequest) -> Result<EmbedResponse, Error> {
		let (model, _, candidates) =
			self.resolve(request.model.as_str(), CatalogFacet::Embeddings)?;
		request.model = model.model.clone();
		let backend = candidates
			.iter()
			.find_map(|candidate| candidate.facets.embed.clone())
			.ok_or_else(|| Error::Provider(Str::from("embedding candidate unavailable")))?;
		backend.embed(request).await
	}
}

#[async_trait]
impl Speak for ProviderRouter {
	async fn speak(
		&self,
		mut request: SpeakRequest,
	) -> Result<futures::stream::BoxStream<'static, SpeakEvent>, Error> {
		let (model, _, candidates) =
			self.resolve(request.model.as_str(), CatalogFacet::AudioSpeech)?;
		request.model = model.model.clone();
		let backend = candidates
			.iter()
			.find_map(|candidate| candidate.facets.speak.clone())
			.ok_or_else(|| Error::Provider(Str::from("speech candidate unavailable")))?;
		backend.speak(request).await
	}
}

#[async_trait]
impl Transcribe for ProviderRouter {
	async fn transcribe(&self, mut request: TranscribeRequest) -> Result<TranscribeResponse, Error> {
		let (model, _, candidates) =
			self.resolve(request.model.as_str(), CatalogFacet::AudioTranscription)?;
		request.model = model.model.clone();
		let backend = candidates
			.iter()
			.find_map(|candidate| candidate.facets.transcribe.clone())
			.ok_or_else(|| Error::Provider(Str::from("transcription candidate unavailable")))?;
		backend.transcribe(request).await
	}
}

/// Catalog-driven three-tier token-count router.
///
/// Exact provider endpoints take precedence over exact local tokenizers. When
/// neither exists, inline threads use a conservative four-characters-per-token
/// estimate and explicitly report [`Accuracy::Estimated`].
pub struct CountRouter {
	models:             Arc<ModelCatalog>,
	providers:          Arc<ProviderCatalog>,
	provider_endpoints: BTreeMap<Str, Arc<dyn CountTokens>>,
	tokenizers:         BTreeMap<Str, Arc<dyn Tokenizer>>,
}

impl CountRouter {
	/// Creates an empty runtime router over immutable catalogs.
	#[must_use]
	pub fn new(models: Arc<ModelCatalog>, providers: Arc<ProviderCatalog>) -> Self {
		Self { models, providers, provider_endpoints: BTreeMap::new(), tokenizers: BTreeMap::new() }
	}

	/// Installs the exact count endpoint belonging to `provider`.
	pub fn insert_provider_endpoint(
		&mut self,
		provider: impl Into<Str>,
		endpoint: Arc<dyn CountTokens>,
	) {
		self.provider_endpoints.insert(provider.into(), endpoint);
	}

	/// Installs an exact custom tokenizer keyed by [`ModelCard::family`].
	pub fn insert_tokenizer(&mut self, family: impl Into<Str>, tokenizer: Arc<dyn Tokenizer>) {
		self.tokenizers.insert(family.into(), tokenizer);
	}

	fn model_and_provider(&self, selector: &str) -> Result<(&ModelCard, &ProviderEntry), Error> {
		let (provider_id, model_id) = selector
			.split_once('/')
			.ok_or_else(|| Error::Provider(Str::from("model selector must be provider/model")))?;
		let model = self
			.models
			.get(provider_id, model_id)
			.ok_or_else(|| Error::Provider(fmts!("unknown catalog model {selector}")))?;
		let provider = self.providers.get(model.provider.as_str()).ok_or_else(|| {
			Error::Provider(fmts!("unknown catalog provider {}", model.provider))
		})?;
		Ok((model, provider))
	}
}

#[async_trait]
impl CountTokens for CountRouter {
	async fn count(&self, mut request: CountRequest) -> Result<CountResponse, Error> {
		let (model, provider) = self.model_and_provider(request.model.as_str())?;
		let provider_has_chat = provider.facets.contains(&CatalogFacet::Chat)
			&& model.facets.contains(&CatalogFacet::Chat);
		let has_exact_endpoint = provider_has_chat;

		if has_exact_endpoint
			&& let Some(endpoint) = self.provider_endpoints.get(provider.id.as_str())
		{
			request.model = model.model.clone();
			let mut response = endpoint.count(request).await?;
			response.accuracy = Accuracy::Exact;
			return Ok(response);
		}

		if let Some(tokenizer) = self.tokenizers.get(model.family.as_str()) {
			return Ok(CountResponse::builder()
				.tokens(tokenizer.count(&request)?)
				.accuracy(Accuracy::Exact)
				.build());
		}

		if let Some(tokenizer) = OpenAiTokenizer::for_model(model) {
			return Ok(CountResponse::builder()
				.tokens(tokenizer.count(&request)?)
				.accuracy(Accuracy::Exact)
				.build());
		}

		Ok(CountResponse::builder()
			.tokens(estimate_tokens(&request)?)
			.accuracy(Accuracy::Estimated)
			.build())
	}
}

/// Failure to construct a production provider token-count route.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum RemoteCountBuildError {
	/// The catalog row does not advertise chat prompts that can be counted.
	#[error("provider {0} does not advertise chat")]
	FacetNotAdvertised(Str),
	/// The provider transport has no native count endpoint owned by this stack.
	#[error("transport {0:?} has no production token-count adapter")]
	UnsupportedTransport(TransportId),
	/// A URL-template route value is unavailable.
	#[error("provider {provider} requires route field {field}")]
	MissingRouteField {
		/// Catalog provider identifier.
		provider: Str,
		/// Missing non-secret route field.
		field:    &'static str,
	},
}

struct RemoteCountShared {
	provider: ProviderEntry,
	route:    ProviderRoute,
	codec:    AnthropicCodec,
}

/// Production unary provider token-count adapter over authenticated egress.
///
/// The outbound request contains an [`AuthContext`] only. Credential lease
/// selection and redemption remain downstream in the shared egress stack.
/// Dropping the count future drops the in-flight egress future.
pub struct RemoteCount<S> {
	shared: Arc<RemoteCountShared>,
	egress: tokio::sync::Mutex<S>,
}

impl<S> RemoteCount<S> {
	/// Constructs the native count adapter supported by a catalog provider.
	pub fn new(
		provider: ProviderEntry,
		route: ProviderRoute,
		egress: S,
	) -> Result<Self, RemoteCountBuildError> {
		if !provider.facets.contains(&CatalogFacet::Chat) {
			return Err(RemoteCountBuildError::FacetNotAdvertised(provider.id));
		}
		if provider.transport != TransportId::AnthropicMessages {
			return Err(RemoteCountBuildError::UnsupportedTransport(provider.transport));
		}
		if let Some(field) = missing_remote_route_field(&provider, &route) {
			return Err(RemoteCountBuildError::MissingRouteField { provider: provider.id, field });
		}
		Ok(Self {
			shared: Arc::new(RemoteCountShared { provider, route, codec: AnthropicCodec::new() }),
			egress: tokio::sync::Mutex::new(egress),
		})
	}
}

/// Builds an object-safe native provider count route for [`CountRouter`].
pub fn remote_count_route<S, B>(
	provider: ProviderEntry,
	route: ProviderRoute,
	egress: S,
) -> Result<Arc<dyn CountTokens>, RemoteCountBuildError>
where
	S: Service<Request<Body>, Response = Response<B>> + Send + 'static,
	S::Future: Send + 'static,
	S::Error: fmt::Display + Send + 'static,
	B: HttpBody<Data = Bytes> + Send + Unpin + 'static,
	B::Error: fmt::Display + Send + 'static,
{
	Ok(Arc::new(RemoteCount::new(provider, route, egress)?))
}

#[async_trait]
impl<S, B> CountTokens for RemoteCount<S>
where
	S: Service<Request<Body>, Response = Response<B>> + Send + 'static,
	S::Future: Send + 'static,
	S::Error: fmt::Display + Send + 'static,
	B: HttpBody<Data = Bytes> + Send + Unpin + 'static,
	B::Error: fmt::Display + Send + 'static,
{
	async fn count(&self, request: CountRequest) -> Result<CountResponse, Error> {
		const MAX_COUNT_RESPONSE_BYTES: usize = 64 * 1024;

		let endpoint = remote_count_endpoint(&self.shared, request.model.as_str())?;
		let (encoded, unsupported) = self
			.shared
			.codec
			.encode_count(&request, &self.shared.provider.compat)?;
		if !unsupported.is_empty() {
			return Err(Error::Unsupported(unsupported));
		}
		let mut builder = Request::post(endpoint)
			.header(header::CONTENT_TYPE, "application/json")
			.header(header::ACCEPT, "application/json");
		for (name, value) in &self.shared.provider.headers {
			builder = builder.header(name.as_str(), value.as_str());
		}
		let mut outbound = builder.body(Body::new(encoded)).map_err(|error| {
			Error::Provider(fmts!("invalid token-count HTTP request: {error}"))
		})?;
		outbound
			.extensions_mut()
			.insert(AuthContext::new(self.shared.provider.id.as_str()));

		let response = {
			let mut egress = self.egress.lock().await;
			egress
				.ready()
				.await
				.map_err(|error| {
					Error::Transport(fmts!("token-count egress not ready: {error}"))
				})?
				.call(outbound)
				.await
				.map_err(|error| Error::Transport(fmts!("token-count egress failed: {error}")))?
		};
		let status = response.status();
		let mut response_body = response.into_body();
		let mut body = BytesMut::new();
		while let Some(frame) = response_body.frame().await {
			let frame = frame.map_err(|error| {
				Error::Transport(fmts!("token-count response body failed: {error}"))
			})?;
			let Some(data) = frame.data_ref() else {
				continue;
			};
			if body.len().saturating_add(data.len()) > MAX_COUNT_RESPONSE_BYTES {
				return Err(Error::Transport(Str::from("token-count response exceeded 64 KiB")));
			}
			body.extend_from_slice(data);
		}
		if !status.is_success() {
			let message = serde_json::from_slice::<Value>(&body)
				.ok()
				.and_then(|value| {
					value
						.pointer("/error/message")
						.and_then(Value::as_str)
						.map(Str::from)
				})
				.unwrap_or_else(|| Str::from("provider returned an error response"));
			return Err(Error::Provider(fmts!(
				"token-count provider returned HTTP {}: {message}",
				status.as_u16()
			)));
		}
		self.shared.codec.decode_count(&body)
	}
}

fn remote_count_endpoint(shared: &RemoteCountShared, model: &str) -> Result<String, Error> {
	let deployment = if shared.route.deployment.is_empty() {
		model
	} else {
		shared.route.deployment.as_str()
	};
	let base = expand_base_url(
		&shared.provider.base_url,
		BaseUrlVars::builder()
			.region(shared.route.region.as_str())
			.location(shared.route.region.as_str())
			.project(shared.route.project.as_str())
			.deployment(deployment)
			.model(model)
			.account(shared.route.account.as_str())
			.gateway(shared.route.gateway.as_str())
			.build(),
	)
	.map_err(|error| Error::Provider(fmts!("invalid token-count endpoint: {error}")))?;
	Ok(format!("{}/v1/messages/count_tokens", base.trim_end_matches('/')))
}

/// Verified per-request limits for one remote embedding endpoint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RemoteEmbedPolicy {
	/// Maximum number of input texts accepted by one provider request.
	pub max_batch_size:      usize,
	/// Whether the endpoint accepts a requested output dimensionality.
	pub supports_dimensions: bool,
}

/// Failure to construct a production remote embedding route.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum RemoteEmbedBuildError {
	/// The catalog row does not advertise embeddings.
	#[error("provider {0} does not advertise embeddings")]
	FacetNotAdvertised(Str),
	/// The provider transport has no verified embedding wire.
	#[error("transport {0:?} has no production embedding adapter")]
	UnsupportedTransport(TransportId),
	/// The transport is compatible, but this provider has no verified limits.
	#[error("provider {0} has no verified embedding policy")]
	UnverifiedProvider(Str),
	/// A URL-template or Vertex route field is unavailable.
	#[error("provider {provider} requires route field {field}")]
	MissingRouteField {
		/// Catalog provider identifier.
		provider: Str,
		/// Missing non-secret route field.
		field:    &'static str,
	},
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RemoteEmbedCodec {
	OpenAi,
	Google(GoogleEmbeddingVariant),
}

impl RemoteEmbedCodec {
	fn encode(self, request: &EmbedRequest) -> Result<Bytes, Error> {
		match self {
			Self::OpenAi => openai_embeddings::encode(request),
			Self::Google(variant) => google_embeddings::encode(request, variant),
		}
	}

	fn decode(self, body: &[u8], estimated_input_tokens: u64) -> Result<EmbedResponse, Error> {
		match self {
			Self::OpenAi => openai_embeddings::decode(body, estimated_input_tokens),
			Self::Google(variant) => google_embeddings::decode(body, variant, estimated_input_tokens),
		}
	}

	fn error_message(self, body: &[u8]) -> Option<Str> {
		let value: Value = serde_json::from_slice(body).ok()?;
		match self {
			Self::OpenAi => openai_embeddings::error_message(&value),
			Self::Google(_) => google_embeddings::error_message(&value),
		}
	}
}

struct RemoteEmbedShared {
	provider: ProviderEntry,
	route:    ProviderRoute,
	codec:    RemoteEmbedCodec,
	policy:   RemoteEmbedPolicy,
}

/// Production unary HTTP embeddings adapter over the shared egress/auth stack.
///
/// The adapter inserts only [`AuthContext`], never credential bytes. The
/// downstream egress stack selects and redeems the provider's sealed credential
/// lease. Dropping the returned `embed` future drops the in-flight egress
/// future and therefore cancels the provider request.
pub struct RemoteEmbed<S> {
	shared: Arc<RemoteEmbedShared>,
	egress: tokio::sync::Mutex<S>,
}

impl<S> RemoteEmbed<S> {
	/// Constructs an adapter only for a catalog row with a verified wire.
	pub fn new(
		provider: ProviderEntry,
		route: ProviderRoute,
		egress: S,
	) -> Result<Self, RemoteEmbedBuildError> {
		if !provider.facets.contains(&CatalogFacet::Embeddings) {
			return Err(RemoteEmbedBuildError::FacetNotAdvertised(provider.id));
		}
		let (codec, policy) = remote_embed_config(&provider)?;
		require_remote_route(&provider, &route)?;
		Ok(Self {
			shared: Arc::new(RemoteEmbedShared { provider, route, codec, policy }),
			egress: tokio::sync::Mutex::new(egress),
		})
	}

	/// Returns the verified batching and dimensions policy.
	#[must_use]
	pub fn policy(&self) -> RemoteEmbedPolicy {
		self.shared.policy
	}

	/// Returns the selected provider catalog row.
	#[must_use]
	pub fn provider(&self) -> &ProviderEntry {
		&self.shared.provider
	}
}

/// Builds an [`EmbedRoute`] backed by a production remote HTTP adapter.
pub fn remote_embed_route<S, B>(
	provider: ProviderEntry,
	route: ProviderRoute,
	egress: S,
) -> Result<EmbedRoute, RemoteEmbedBuildError>
where
	S: Service<Request<Body>, Response = Response<B>> + Send + 'static,
	S::Future: Send + 'static,
	S::Error: fmt::Display + Send + 'static,
	B: HttpBody<Data = Bytes> + Send + Unpin + 'static,
	B::Error: fmt::Display + Send + 'static,
{
	let backend = RemoteEmbed::new(provider, route, egress)?;
	let policy = backend.policy();
	Ok(EmbedRoute {
		backend:             Arc::new(backend),
		max_batch_size:      policy.max_batch_size,
		supports_dimensions: policy.supports_dimensions,
	})
}

#[async_trait]
impl<S, B> Embed for RemoteEmbed<S>
where
	S: Service<Request<Body>, Response = Response<B>> + Send + 'static,
	S::Future: Send + 'static,
	S::Error: fmt::Display + Send + 'static,
	B: HttpBody<Data = Bytes> + Send + Unpin + 'static,
	B::Error: fmt::Display + Send + 'static,
{
	async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse, Error> {
		if request.texts.len() > self.shared.policy.max_batch_size {
			return Err(Error::Provider(fmts!(
				"embedding batch exceeds provider limit of {} inputs",
				self.shared.policy.max_batch_size
			)));
		}
		if request.dimensions == Some(0) {
			return Err(Error::Unsupported(vec![unsupported(
				"dimensions",
				"embedding dimensions must be greater than zero",
			)]));
		}
		if request.dimensions.is_some() && !self.shared.policy.supports_dimensions {
			return Err(Error::Unsupported(vec![unsupported(
				"dimensions",
				"the selected embedding provider does not support custom dimensions",
			)]));
		}
		if request.texts.is_empty() {
			return Ok(EmbedResponse::builder().vectors(Vec::new()).build());
		}
		let endpoint = remote_embed_endpoint(&self.shared, request.model.as_str())?;
		let encoded = self.shared.codec.encode(&request)?;
		let estimated_tokens = estimate_embedding_tokens(&request.texts);
		let mut builder = Request::post(endpoint)
			.header(header::CONTENT_TYPE, "application/json")
			.header(header::ACCEPT, "application/json");
		for (name, value) in &self.shared.provider.headers {
			builder = builder.header(name.as_str(), value.as_str());
		}
		let mut outbound = builder.body(Body::new(encoded)).map_err(|error| {
			Error::Provider(fmts!("invalid embedding HTTP request: {error}"))
		})?;
		outbound
			.extensions_mut()
			.insert(AuthContext::new(self.shared.provider.id.as_str()));

		let response = {
			let mut egress = self.egress.lock().await;
			egress
				.ready()
				.await
				.map_err(|error| Error::Transport(fmts!("embedding egress not ready: {error}")))?
				.call(outbound)
				.await
				.map_err(|error| Error::Transport(fmts!("embedding egress failed: {error}")))?
		};
		let status = response.status();
		let body = response
			.into_body()
			.collect()
			.await
			.map_err(|error| {
				Error::Transport(fmts!("embedding response body failed: {error}"))
			})?
			.to_bytes();
		if !status.is_success() {
			let message = self
				.shared
				.codec
				.error_message(&body)
				.unwrap_or_else(|| Str::from("provider returned an error response"));
			return Err(Error::Provider(fmts!(
				"embedding provider returned HTTP {}: {message}",
				status.as_u16()
			)));
		}
		self.shared.codec.decode(&body, estimated_tokens)
	}
}

fn remote_embed_config(
	provider: &ProviderEntry,
) -> Result<(RemoteEmbedCodec, RemoteEmbedPolicy), RemoteEmbedBuildError> {
	let configured = match provider.transport {
		TransportId::OpenAiChat | TransportId::OpenAiResponses => {
			let policy = match provider.id.as_str() {
				"openai" | "litellm" | "vllm" => {
					RemoteEmbedPolicy { max_batch_size: 2_048, supports_dimensions: true }
				},
				"mistral" => RemoteEmbedPolicy { max_batch_size: 32, supports_dimensions: false },
				// These endpoints expose the OpenAI-compatible wire, but their
				// published batch/dimensionality guarantees are model-specific.
				// A one-item request is the conservative verified common contract.
				"fireworks" | "together" | "nvidia" | "ollama" | "lm-studio" => {
					RemoteEmbedPolicy { max_batch_size: 1, supports_dimensions: false }
				},
				_ => {
					return Err(RemoteEmbedBuildError::UnverifiedProvider(provider.id.clone()));
				},
			};
			(RemoteEmbedCodec::OpenAi, policy)
		},
		TransportId::GoogleGenAi => {
			(RemoteEmbedCodec::Google(GoogleEmbeddingVariant::GenAi), RemoteEmbedPolicy {
				max_batch_size:      100,
				supports_dimensions: true,
			})
		},
		TransportId::GoogleVertex => {
			(RemoteEmbedCodec::Google(GoogleEmbeddingVariant::Vertex), RemoteEmbedPolicy {
				max_batch_size:      250,
				supports_dimensions: true,
			})
		},
		transport => return Err(RemoteEmbedBuildError::UnsupportedTransport(transport)),
	};
	Ok(configured)
}

fn require_remote_route(
	provider: &ProviderEntry,
	route: &ProviderRoute,
) -> Result<(), RemoteEmbedBuildError> {
	if let Some(field) = missing_remote_route_field(provider, route) {
		return Err(RemoteEmbedBuildError::MissingRouteField {
			provider: provider.id.clone(),
			field,
		});
	}
	Ok(())
}

fn missing_remote_route_field(
	provider: &ProviderEntry,
	route: &ProviderRoute,
) -> Option<&'static str> {
	if provider.transport == TransportId::GoogleVertex && route.project.is_empty() {
		Some("project")
	} else if (provider.transport == TransportId::GoogleVertex
		|| provider.base_url.contains("{region}")
		|| provider.base_url.contains("{location}"))
		&& route.region.is_empty()
	{
		Some("region")
	} else if provider.base_url.contains("{account}") && route.account.is_empty() {
		Some("account")
	} else if provider.base_url.contains("{gateway}") && route.gateway.is_empty() {
		Some("gateway")
	} else {
		None
	}
}

fn remote_embed_endpoint(shared: &RemoteEmbedShared, model: &str) -> Result<String, Error> {
	let deployment = if shared.route.deployment.is_empty() {
		model
	} else {
		shared.route.deployment.as_str()
	};
	let base = expand_base_url(
		&shared.provider.base_url,
		BaseUrlVars::builder()
			.region(shared.route.region.as_str())
			.location(shared.route.region.as_str())
			.project(shared.route.project.as_str())
			.deployment(deployment)
			.model(model)
			.account(shared.route.account.as_str())
			.gateway(shared.route.gateway.as_str())
			.build(),
	)
	.map_err(|error| Error::Provider(fmts!("invalid embedding endpoint: {error}")))?;
	let base = base.trim_end_matches('/');
	let suffix = match shared.codec {
		RemoteEmbedCodec::OpenAi => "/embeddings".to_owned(),
		RemoteEmbedCodec::Google(GoogleEmbeddingVariant::GenAi) => {
			format!("/models/{model}:batchEmbedContents")
		},
		RemoteEmbedCodec::Google(GoogleEmbeddingVariant::Vertex) => format!(
			"/projects/{}/locations/{}/publishers/google/models/{model}:predict",
			shared.route.project, shared.route.region
		),
	};
	Ok(format!("{base}{suffix}"))
}

fn estimate_embedding_tokens(texts: &[Str]) -> u64 {
	texts.iter().fold(0u64, |total, text| {
		let chars = u64::try_from(text.chars().count()).unwrap_or(u64::MAX);
		total.saturating_add(chars.saturating_add(3) / 4)
	})
}

/// Runtime configuration for one HTTP or in-process embedding provider.
#[derive(Clone)]
pub struct EmbedRoute {
	/// Provider or embedded-engine implementation.
	pub backend:             Arc<dyn Embed>,
	/// Maximum texts admitted in one backend request; values below one are
	/// treated as one.
	pub max_batch_size:      usize,
	/// Whether the backend honors [`EmbedRequest::dimensions`].
	pub supports_dimensions: bool,
}

/// Catalog-driven embedding router for HTTP providers and embedded local
/// engines.
///
/// Both kinds install their existing [`Embed`] implementation in an
/// [`EmbedRoute`]; [`omp_llm_transport::embedded::Embedded`] therefore supplies
/// the LocalEngine-style seam without coupling routing to a particular local
/// runtime.
pub struct EmbedRouter {
	models:    Arc<ModelCatalog>,
	providers: Arc<ProviderCatalog>,
	routes:    BTreeMap<Str, EmbedRoute>,
}

impl EmbedRouter {
	/// Creates an empty runtime router over immutable catalogs.
	#[must_use]
	pub const fn new(models: Arc<ModelCatalog>, providers: Arc<ProviderCatalog>) -> Self {
		Self { models, providers, routes: BTreeMap::new() }
	}

	/// Installs an HTTP provider or embedded local-engine route.
	pub fn insert_route(&mut self, provider: impl Into<Str>, route: EmbedRoute) {
		self.routes.insert(provider.into(), route);
	}

	fn resolve(&self, selector: &str) -> Result<(&ModelCard, &ProviderEntry, &EmbedRoute), Error> {
		let (provider_id, model_id) = selector
			.split_once('/')
			.ok_or_else(|| Error::Provider(Str::from("model selector must be provider/model")))?;
		let model = self
			.models
			.get(provider_id, model_id)
			.ok_or_else(|| Error::Provider(fmts!("unknown catalog model {selector}")))?;
		let provider = self.providers.get(model.provider.as_str()).ok_or_else(|| {
			Error::Provider(fmts!("unknown catalog provider {}", model.provider))
		})?;
		if !model.facets.contains(&CatalogFacet::Embeddings)
			|| !provider.facets.contains(&CatalogFacet::Embeddings)
		{
			return Err(Error::Unsupported(vec![unsupported(
				"model",
				"the catalog route does not advertise embeddings",
			)]));
		}
		let route = self.routes.get(provider.id.as_str()).ok_or_else(|| {
			Error::Provider(fmts!("embedding route unavailable for {}", provider.id))
		})?;
		Ok((model, provider, route))
	}
}

#[async_trait]
impl Embed for EmbedRouter {
	async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse, Error> {
		let (model, _, route) = self.resolve(request.model.as_str())?;
		if request.dimensions == Some(0) {
			return Err(Error::Unsupported(vec![unsupported(
				"dimensions",
				"embedding dimensions must be greater than zero",
			)]));
		}
		if request.dimensions.is_some() && !route.supports_dimensions {
			return Err(Error::Unsupported(vec![unsupported(
				"dimensions",
				"the selected embedding provider does not support custom dimensions",
			)]));
		}
		if request.texts.is_empty() {
			return Ok(EmbedResponse::builder().vectors(Vec::new()).build());
		}

		let mut vectors = Vec::with_capacity(request.texts.len());
		let mut usage = None;
		let mut vector_width = None;
		let mut request_count = 0u64;
		for texts in request.texts.chunks(route.max_batch_size.max(1)) {
			request_count = request_count.saturating_add(1);
			let expected = texts.len();
			let response = route
				.backend
				.embed(
					EmbedRequest::builder()
						.model(model.model.clone())
						.texts(texts.to_vec())
						.maybe_dimensions(request.dimensions)
						.props(request.props.clone())
						.build(),
				)
				.await?;
			if response.vectors.len() != expected {
				return Err(Error::Provider(Str::from(
					"embedding provider returned a vector count different from its input count",
				)));
			}
			validate_vector_dimensions(&response.vectors, request.dimensions, &mut vector_width)?;
			vectors.extend(response.vectors);
			merge_usage(&mut usage, response.usage);
		}
		attach_embedding_cost(model, &request.texts, request_count, &mut usage);
		Ok(EmbedResponse::builder()
			.vectors(vectors)
			.maybe_usage(usage)
			.build())
	}
}

fn validate_vector_dimensions(
	vectors: &[omp_llm_types::EmbeddingVector],
	requested: Option<u32>,
	observed: &mut Option<usize>,
) -> Result<(), Error> {
	let requested = requested.and_then(|width| usize::try_from(width).ok());
	for vector in vectors {
		let width = vector.values.len();
		if width == 0 {
			return Err(Error::Provider(Str::from("embedding provider returned an empty vector")));
		}
		if requested.is_some_and(|requested| requested != width) {
			return Err(Error::Provider(Str::from(
				"embedding provider returned a vector with the wrong requested dimensions",
			)));
		}
		if observed.is_some_and(|observed| observed != width) {
			return Err(Error::Provider(Str::from(
				"embedding provider returned inconsistent native dimensions",
			)));
		}
		*observed = Some(width);
	}
	Ok(())
}

fn attach_embedding_cost(
	model: &ModelCard,
	texts: &[Str],
	request_count: u64,
	usage: &mut Option<Usage>,
) {
	let Some(usage) = usage else { return };
	let characters = texts.iter().fold(0u64, |total, text| {
		total.saturating_add(u64::try_from(text.chars().count()).unwrap_or(u64::MAX))
	});
	let mut nanos = 0u128;
	let mut priced = false;
	for price in &model.pricing {
		match price.unit {
			PriceUnit::MtokInput | PriceUnit::McharInput | PriceUnit::Request => {
				priced = true;
			},
			_ => {},
		};
		let quantity = match price.unit {
			PriceUnit::MtokInput => u128::from(usage.input_tokens),
			PriceUnit::McharInput => u128::from(characters),
			PriceUnit::Request => {
				nanos = nanos.saturating_add(
					u128::from(price.nanos_usd).saturating_mul(u128::from(request_count)),
				);
				continue;
			},
			_ => continue,
		};
		let component = u128::from(price.nanos_usd)
			.saturating_mul(quantity)
			.saturating_add(999_999)
			/ 1_000_000;
		nanos = nanos.saturating_add(component);
	}
	if !priced {
		return;
	}
	let nanos = u64::try_from(nanos).unwrap_or(u64::MAX);
	usage
		.detail
		.insert_ns("omp", "cost_nanos_usd", Value::from(nanos));
	usage
		.detail
		.insert_ns("omp", "cost_estimated", Value::Bool(true));
}

fn unsupported(what: &str, detail: &str) -> Unsupported {
	Unsupported::builder()
		.what(Str::from(what))
		.detail(Str::from(detail))
		.action(UnsupportedAction::Dropped)
		.build()
}

// Four characters per token is intentionally only a fallback. Structural
// overhead keeps empty/short messages from being reported as free, while every
// family without a real tokenizer remains honestly Estimated.
fn estimate_tokens(request: &CountRequest) -> Result<u64, Error> {
	let CountInput::Thread(thread) = &request.input else {
		return Err(Error::Unsupported(vec![unsupported(
			"input.context",
			"heuristic counting requires the resolved inline thread",
		)]));
	};
	let tool_chars = request.tools.iter().fold(0usize, |total, tool| {
		total.saturating_add(
			tool
				.name
				.chars()
				.count()
				.saturating_add(tool.description.chars().count())
				.saturating_add(tool.schema_json.len())
				.saturating_add(8),
		)
	});
	let chars = thread_chars(thread).saturating_add(tool_chars);
	Ok(u64::try_from(chars.saturating_add(3) / 4).unwrap_or(u64::MAX))
}

fn thread_chars(thread: &Thread) -> usize {
	thread.items.iter().fold(0usize, |total, item| {
		let item_chars = match &item.kind {
			ItemKind::Message(message) => message
				.parts
				.iter()
				.fold(0usize, |sum, part| sum.saturating_add(part_chars(part))),
			ItemKind::ToolCall(call) => call
				.name
				.chars()
				.count()
				.saturating_add(call.args_json.len()),
			ItemKind::ToolResult(result) => result
				.parts
				.iter()
				.fold(0usize, |sum, part| sum.saturating_add(part_chars(part))),
			_ => 0,
		};
		total.saturating_add(4usize.saturating_add(item_chars))
	})
}

fn part_chars(part: &Part) -> usize {
	match part {
		Part::Text(text) => text.chars().count(),
		Part::Thinking(thinking) => thinking.text.chars().count(),
		Part::Blob(blob) => usize::try_from(blob.size / 4).unwrap_or(usize::MAX),
		_ => 0,
	}
}

fn merge_usage(total: &mut Option<Usage>, next: Option<Usage>) {
	let Some(next) = next else { return };
	if let Some(total) = total {
		total.input_tokens = total.input_tokens.saturating_add(next.input_tokens);
		total.output_tokens = total.output_tokens.saturating_add(next.output_tokens);
		total.cache_read_tokens = total
			.cache_read_tokens
			.saturating_add(next.cache_read_tokens);
		total.cache_write_tokens = total
			.cache_write_tokens
			.saturating_add(next.cache_write_tokens);
		if matches!(next.accuracy, Accuracy::Estimated) {
			total.accuracy = Accuracy::Estimated;
		}
		total.detail.0.extend(next.detail.0);
	} else {
		*total = Some(next);
	}
}

#[cfg(test)]
mod tests {
	use std::sync::atomic::{AtomicUsize, Ordering};

	use http_body_util::Full;
	use omp_llm_catalog::{
		compat::Compat,
		models::{Availability, Modality, Source},
		provider::AuthSpec,
	};
	use omp_llm_types::{EmbeddingVector, Item, Message, Props, Role};
	use parking_lot::Mutex;
	use smallvec::{SmallVec, smallvec};
	use tower::service_fn;

	use super::*;

	fn catalogs(
		transport: TransportId,
		facet: CatalogFacet,
	) -> (Arc<ModelCatalog>, Arc<ProviderCatalog>) {
		catalogs_for_model(transport, facet, "model", "family")
	}

	fn catalogs_for_model(
		transport: TransportId,
		facet: CatalogFacet,
		model_id: &str,
		family: &str,
	) -> (Arc<ModelCatalog>, Arc<ProviderCatalog>) {
		let model = ModelCard::builder()
			.id(Str::new(format!("vendor/{model_id}")))
			.provider(Str::from("vendor"))
			.model(Str::new(model_id))
			.name(Str::new(model_id))
			.family(Str::new(family))
			.facets(smallvec![facet])
			.inputs(smallvec![Modality::Text])
			.outputs(SmallVec::new())
			.reasoning(false)
			.efforts(SmallVec::new())
			.context_window(1_000)
			.max_output_tokens(0)
			.pricing(SmallVec::new())
			.availability(Availability::Available)
			.source(Source::Bundled)
			.blocked_until_ms(0)
			.deprecated(false)
			.updated_at_ms(0)
			.props(Props::default())
			.effort_routing(BTreeMap::new())
			.build();
		let provider = ProviderEntry::builder()
			.id(Str::from("vendor"))
			.transport(transport)
			.base_url(Str::from("https://example.test"))
			.fallback_base_urls(SmallVec::new())
			.auth(AuthSpec::default())
			.facets(smallvec![facet])
			.headers(BTreeMap::new())
			.compat(Compat::default())
			.build();
		let mut providers = ProviderCatalog::new();
		providers.insert(provider.id.clone(), provider);
		(Arc::new(ModelCatalog::new(vec![model])), Arc::new(providers))
	}

	fn count_request() -> CountRequest {
		count_request_for_model("model")
	}

	fn count_request_for_model(model_id: &str) -> CountRequest {
		CountRequest::builder()
			.model(Str::new(format!("vendor/{model_id}")))
			.input(CountInput::Thread(
				Thread::builder()
					.items(vec![
						Item::builder()
							.seq(0)
							.kind(ItemKind::Message(
								Message::builder()
									.role(Role::User)
									.parts(vec![Part::Text(Str::from("hello world"))])
									.build(),
							))
							.props(Props::default())
							.build(),
					])
					.build(),
			))
			.tools(Vec::new())
			.build()
	}

	struct FixedCount {
		calls:  AtomicUsize,
		tokens: u64,
	}
	#[async_trait]
	impl CountTokens for FixedCount {
		async fn count(&self, _: CountRequest) -> Result<CountResponse, Error> {
			self.calls.fetch_add(1, Ordering::Relaxed);
			Ok(CountResponse::builder()
				.tokens(self.tokens)
				.accuracy(Accuracy::Estimated)
				.build())
		}
	}
	struct FixedTokenizer;
	impl Tokenizer for FixedTokenizer {
		fn count(&self, _: &CountRequest) -> Result<u64, Error> {
			Ok(23)
		}
	}

	#[tokio::test]
	async fn provider_count_endpoint_is_exact() {
		let (models, providers) = catalogs(TransportId::AnthropicMessages, CatalogFacet::Chat);
		let endpoint = Arc::new(FixedCount { calls: AtomicUsize::new(0), tokens: 17 });
		let mut router = CountRouter::new(models, providers);
		router.insert_provider_endpoint("vendor", endpoint.clone());
		let result = router.count(count_request()).await.unwrap();
		assert_eq!((result.tokens, result.accuracy), (17, Accuracy::Exact));
		assert_eq!(endpoint.calls.load(Ordering::Relaxed), 1);
	}

	#[tokio::test]
	async fn missing_endpoint_and_tokenizer_is_estimated() {
		let (models, providers) = catalogs(TransportId::OpenAiResponses, CatalogFacet::Chat);
		let result = CountRouter::new(models, providers)
			.count(count_request())
			.await
			.unwrap();
		assert!(result.tokens > 0);
		assert_eq!(result.accuracy, Accuracy::Estimated);
	}

	#[tokio::test]
	async fn catalog_selected_openai_ranks_are_exact() {
		for (model_id, expected_tokens) in [
			("gpt-3.5-turbo-0301", 10),
			("gpt-4-0613", 9),
			("gpt-4o-2024-08-06", 9),
			("codex-mini-latest", 9),
		] {
			let (models, providers) = catalogs_for_model(
				TransportId::OpenAiResponses,
				CatalogFacet::Chat,
				model_id,
				"openai",
			);
			let response = CountRouter::new(models, providers)
				.count(count_request_for_model(model_id))
				.await
				.unwrap();
			assert_eq!(
				(response.tokens, response.accuracy),
				(expected_tokens, Accuracy::Exact),
				"{model_id}",
			);
		}
	}

	#[tokio::test]
	async fn unknown_and_foreign_families_remain_estimated() {
		for (model_id, family) in
			[("future-openai-model", "openai"), ("gpt-4o", "anthropic"), ("gpt-4o", "gemini")]
		{
			let (models, providers) =
				catalogs_for_model(TransportId::OpenAiResponses, CatalogFacet::Chat, model_id, family);
			let response = CountRouter::new(models, providers)
				.count(count_request_for_model(model_id))
				.await
				.unwrap();
			assert_eq!(response.accuracy, Accuracy::Estimated, "{family}/{model_id}");
		}
	}

	#[tokio::test]
	async fn provider_precedes_available_tokenizer() {
		let (models, providers) = catalogs(TransportId::GoogleGenAi, CatalogFacet::Chat);
		let endpoint = Arc::new(FixedCount { calls: AtomicUsize::new(0), tokens: 11 });
		let mut router = CountRouter::new(models, providers);
		router.insert_provider_endpoint("vendor", endpoint);
		router.insert_tokenizer("family", Arc::new(FixedTokenizer));
		assert_eq!(router.count(count_request()).await.unwrap().tokens, 11);
	}

	#[tokio::test]
	async fn anthropic_remote_count_uses_authenticated_native_endpoint() {
		let (models, providers) = catalogs_for_model(
			TransportId::AnthropicMessages,
			CatalogFacet::Chat,
			"model",
			"anthropic",
		);
		let mut provider = providers.get("vendor").unwrap().clone();
		provider
			.headers
			.insert(Str::from("anthropic-version"), Str::from("2023-06-01"));
		let observed = Arc::new(Mutex::new(None));
		let observed_request = Arc::clone(&observed);
		let egress = service_fn(move |request: Request<Body>| {
			let observed = Arc::clone(&observed_request);
			async move {
				let authenticated = request.extensions().get::<AuthContext>().is_some();
				let version = request
					.headers()
					.get("anthropic-version")
					.and_then(|value| value.to_str().ok())
					.map(str::to_owned);
				let uri = request.uri().to_string();
				let body = request.into_body().collect().await.unwrap().to_bytes();
				let json: Value = serde_json::from_slice(&body).unwrap();
				*observed.lock() = Some((uri, authenticated, version, json));
				Ok::<_, std::convert::Infallible>(
					Response::builder()
						.status(200)
						.body(Full::new(Bytes::from_static(br#"{"input_tokens":7}"#)))
						.unwrap(),
				)
			}
		});
		let endpoint = remote_count_route(provider, ProviderRoute::default(), egress).unwrap();
		let mut router = CountRouter::new(models, providers);
		router.insert_provider_endpoint("vendor", endpoint);
		let response = router.count(count_request()).await.unwrap();
		assert_eq!((response.tokens, response.accuracy), (7, Accuracy::Exact));

		let (uri, authenticated, version, body) = observed.lock().take().unwrap();
		assert_eq!(uri, "https://example.test/v1/messages/count_tokens");
		assert!(authenticated);
		assert_eq!(version.as_deref(), Some("2023-06-01"));
		assert_eq!(body["model"], "model");
		assert_eq!(body["messages"][0]["role"], "user");
	}

	#[derive(Default)]
	struct RecordingEmbed {
		batches: Mutex<Vec<Vec<Str>>>,
	}
	#[async_trait]
	impl Embed for RecordingEmbed {
		async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse, Error> {
			self.batches.lock().push(request.texts.clone());
			Ok(EmbedResponse::builder()
				.vectors(
					request
						.texts
						.iter()
						.map(|text| {
							EmbeddingVector::builder()
								.values(vec![text.parse::<f32>().unwrap()])
								.build()
						})
						.collect(),
				)
				.build())
		}
	}

	#[tokio::test]
	async fn embedding_chunks_and_preserves_input_order() {
		let (models, providers) = catalogs(TransportId::OpenAiResponses, CatalogFacet::Embeddings);
		let backend = Arc::new(RecordingEmbed::default());
		let mut router = EmbedRouter::new(models, providers);
		router.insert_route("vendor", EmbedRoute {
			backend:             backend.clone(),
			max_batch_size:      2,
			supports_dimensions: true,
		});
		let response = router
			.embed(
				EmbedRequest::builder()
					.model(Str::from("vendor/model"))
					.texts(["0", "1", "2", "3", "4"].map(Str::from).to_vec())
					.dimensions(1)
					.props(Props::default())
					.build(),
			)
			.await
			.unwrap();
		assert_eq!(
			backend
				.batches
				.lock()
				.iter()
				.map(Vec::len)
				.collect::<Vec<_>>(),
			vec![2, 2, 1]
		);
		assert_eq!(
			response
				.vectors
				.iter()
				.map(|vector| vector.values[0])
				.collect::<Vec<_>>(),
			vec![0.0, 1.0, 2.0, 3.0, 4.0]
		);
	}

	#[tokio::test]
	async fn unsupported_dimensions_are_not_silently_ignored() {
		let (models, providers) = catalogs(TransportId::OpenAiResponses, CatalogFacet::Embeddings);
		let backend = Arc::new(RecordingEmbed::default());
		let mut router = EmbedRouter::new(models, providers);
		router.insert_route("vendor", EmbedRoute {
			backend:             backend.clone(),
			max_batch_size:      8,
			supports_dimensions: false,
		});
		let error = router
			.embed(
				EmbedRequest::builder()
					.model(Str::from("vendor/model"))
					.texts(vec![Str::from("1")])
					.dimensions(8)
					.props(Props::default())
					.build(),
			)
			.await
			.unwrap_err();
		assert!(matches!(error, Error::Unsupported(records) if records[0].what == "dimensions"));
		assert!(backend.batches.lock().is_empty());
	}
}
