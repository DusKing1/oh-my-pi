//! Catalog-driven production registration for remote audio facets.

use std::{collections::BTreeMap, fmt, sync::Arc};

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::BoxStream;
use http::{Request, Response};
use hyper::body::Body as HttpBody;
use omp_core::Str;
use omp_llm_catalog::provider::{Facet as CatalogFacet, ProviderEntry};
use omp_llm_egress::client::Body;
use omp_llm_tower::{
	audio::{AudioAttemptBuildError, AudioProviderAttempt},
	provider::ProviderRoute,
};
use omp_llm_types::{
	SpeakEvent, SpeakRequest, TranscribeRequest, TranscribeResponse,
	facet::{self, Speak, Transcribe},
};
use tower::Service;

/// Remote audio facet implementations assembled from advertised catalog rows.
#[derive(Clone, Default)]
pub struct ProductionAudioFacets {
	/// Speech synthesis when at least one production route was registered.
	pub speak:                Option<Arc<dyn Speak>>,
	/// Transcription when at least one production route was registered.
	pub transcribe:           Option<Arc<dyn Transcribe>>,
	/// Number of speech-capable catalog rows registered.
	pub speech_routes:        usize,
	/// Number of transcription-capable catalog rows registered.
	pub transcription_routes: usize,
}

/// Failure to assemble an advertised production audio route.
#[derive(Clone, Debug, thiserror::Error, Eq, PartialEq)]
pub enum AudioRouteRegistrationError {
	/// A catalog row advertised audio without a production codec.
	#[error("audio provider {provider} could not be assembled: {source}")]
	Provider {
		/// Provider catalog id with the unsupported claim.
		provider: Str,
		/// Concrete adapter construction failure.
		source:   AudioAttemptBuildError,
	},
}

#[derive(Clone)]
struct RegisteredAudioRoute<S> {
	attempt:    AudioProviderAttempt<S>,
	speech:     bool,
	transcribe: bool,
}

struct ProductionAudioRouter<S> {
	routes: BTreeMap<Str, RegisteredAudioRoute<S>>,
}

/// Registers every advertised remote audio catalog row.
///
/// Rows are admitted only when a production codec exists for their transport;
/// an advertised but unsupported wire is a startup error rather than an empty
/// stream. The supplied egress service must include credential injection and
/// deployment policy. Each request executes exactly one selected provider and
/// never falls back after cancellation or transport commit.
pub fn register_production_audio_routes<'a, I, S, B, R>(
	providers: I,
	egress: S,
	mut route: R,
) -> Result<ProductionAudioFacets, AudioRouteRegistrationError>
where
	I: IntoIterator<Item = &'a ProviderEntry>,
	S: Service<Request<Body>, Response = Response<B>> + Clone + Send + Sync + 'static,
	S::Future: Send + 'static,
	S::Error: fmt::Display + Send + 'static,
	B: HttpBody<Data = Bytes> + Send + Unpin + 'static,
	B::Error: fmt::Display + Send + 'static,
	R: FnMut(&ProviderEntry) -> ProviderRoute,
{
	let mut routes = BTreeMap::new();
	let mut speech_routes = 0;
	let mut transcription_routes = 0;
	for provider in providers {
		let speech = provider.facets.contains(&CatalogFacet::AudioSpeech);
		let transcribe = provider.facets.contains(&CatalogFacet::AudioTranscription);
		if !speech && !transcribe {
			continue;
		}
		let attempt = AudioProviderAttempt::new(provider.clone(), route(provider), egress.clone())
			.map_err(|source| AudioRouteRegistrationError::Provider {
				provider: provider.id.clone(),
				source,
			})?;
		speech_routes += usize::from(speech);
		transcription_routes += usize::from(transcribe);
		routes.insert(provider.id.clone(), RegisteredAudioRoute { attempt, speech, transcribe });
	}
	if routes.is_empty() {
		return Ok(ProductionAudioFacets::default());
	}
	let router = Arc::new(ProductionAudioRouter { routes });
	let speak = (speech_routes != 0).then(|| Arc::clone(&router) as Arc<dyn Speak>);
	let transcribe = (transcription_routes != 0).then(|| Arc::clone(&router) as Arc<dyn Transcribe>);
	Ok(ProductionAudioFacets { speak, transcribe, speech_routes, transcription_routes })
}

#[async_trait]
impl<S, B> Speak for ProductionAudioRouter<S>
where
	S: Service<Request<Body>, Response = Response<B>> + Clone + Send + Sync + 'static,
	S::Future: Send + 'static,
	S::Error: fmt::Display + Send + 'static,
	B: HttpBody<Data = Bytes> + Send + Unpin + 'static,
	B::Error: fmt::Display + Send + 'static,
{
	async fn speak(
		&self,
		mut request: SpeakRequest,
	) -> Result<BoxStream<'static, SpeakEvent>, facet::Error> {
		let (provider, local_model) =
			select_route(&self.routes, &request.model, &request.props, |route| route.speech)?;
		request.model = local_model;
		provider
			.attempt
			.speak(request)
			.await
			.map_err(|error| facet::Error::Provider(error.to_string().into()))
	}
}

#[async_trait]
impl<S, B> Transcribe for ProductionAudioRouter<S>
where
	S: Service<Request<Body>, Response = Response<B>> + Clone + Send + Sync + 'static,
	S::Future: Send + 'static,
	S::Error: fmt::Display + Send + 'static,
	B: HttpBody<Data = Bytes> + Send + Unpin + 'static,
	B::Error: fmt::Display + Send + 'static,
{
	async fn transcribe(
		&self,
		mut request: TranscribeRequest,
	) -> Result<TranscribeResponse, facet::Error> {
		let (provider, local_model) =
			select_route(&self.routes, &request.model, &request.props, |route| route.transcribe)?;
		request.model = local_model;
		provider
			.attempt
			.transcribe(request)
			.await
			.map_err(|error| facet::Error::Provider(error.to_string().into()))
	}
}

fn select_route<'a, S>(
	routes: &'a BTreeMap<Str, RegisteredAudioRoute<S>>,
	model: &str,
	props: &omp_llm_types::Props,
	supports: impl Fn(&RegisteredAudioRoute<S>) -> bool,
) -> Result<(&'a RegisteredAudioRoute<S>, Str), facet::Error> {
	let pinned = props
		.get_ns("audio", "provider")
		.or_else(|| props.get_ns("omp", "provider"))
		.and_then(serde_json::Value::as_str);
	if let Some(provider) = pinned {
		let route = routes
			.get(provider)
			.filter(|route| supports(route))
			.ok_or_else(|| {
				facet::Error::Provider(format!("audio provider {provider} is unavailable").into())
			})?;
		return Ok((route, strip_provider(model, provider)));
	}
	if let Some((provider, local)) = model.split_once('/')
		&& let Some(route) = routes.get(provider).filter(|route| supports(route))
	{
		return Ok((route, local.into()));
	}
	if let Some(route) = routes.get("openai").filter(|route| supports(route)) {
		return Ok((route, model.into()));
	}
	let mut candidates = routes.values().filter(|route| supports(route));
	let Some(route) = candidates.next() else {
		return Err(facet::Error::Provider("no configured audio provider".into()));
	};
	if candidates.next().is_some() {
		return Err(facet::Error::Provider(
			"audio model must include a provider prefix when multiple providers are configured".into(),
		));
	}
	Ok((route, model.into()))
}

fn strip_provider(model: &str, provider: &str) -> Str {
	model
		.strip_prefix(provider)
		.and_then(|model| model.strip_prefix('/'))
		.unwrap_or(model)
		.into()
}
