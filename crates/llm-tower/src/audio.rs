//! Catalog-backed HTTP attempts for remote speech and transcription facets.
//!
//! Requests pass through the caller-supplied egress stack. This module installs
//! only non-secret [`AuthContext`] metadata; credential selection, redemption,
//! retry policy, limits, and socket ownership remain in egress.

use std::fmt;

use bytes::{Bytes, BytesMut};
use futures::{StreamExt, stream::BoxStream};
use http::{Request, Response, StatusCode, header};
use http_body_util::{BodyExt, Full};
use hyper::body::Body as HttpBody;
use omp_core::{SmolStr, format_smol};
use omp_llm_catalog::{
	compat::AudioApiVersion,
	provider::{BaseUrlVars, ProviderEntry, TransportId, expand_base_url},
};
use omp_llm_egress::{auth_inject::AuthContext, client::Body};
use omp_llm_openai::{EncodedAudioRequest, OpenAiAudioCodec, OpenAiAudioError, OpenAiAudioProfile};
use omp_llm_types::{
	AudioEncoding, BlobPart, Props, SpeakChunk, SpeakDone, SpeakEvent, SpeakRequest,
	TranscribeRequest, TranscribeResponse,
};
use tower::{Service, ServiceExt};

use crate::provider::ProviderRoute;

/// Construction failure for a catalog audio route.
#[derive(Clone, Debug, thiserror::Error, Eq, PartialEq)]
pub enum AudioAttemptBuildError {
	/// Only OpenAI-compatible HTTP transports own production audio codecs.
	#[error("transport {0:?} has no production audio codec")]
	UnsupportedTransport(TransportId),
}

/// Classified failure from one remote audio provider attempt.
#[derive(Clone, Debug, thiserror::Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum AudioAttemptError {
	/// The provider codec rejected a canonical request.
	#[error("audio request encoding failed: {0}")]
	Encode(OpenAiAudioError),
	/// Catalog endpoint template expansion failed.
	#[error("invalid audio provider endpoint: {0}")]
	Endpoint(SmolStr),
	/// The encoded request could not form an HTTP request.
	#[error("invalid audio provider request: {0}")]
	HttpRequest(SmolStr),
	/// The authenticated egress stack failed before response headers.
	#[error("audio provider egress failed: {0}")]
	Egress(SmolStr),
	/// The upstream rejected the request with an HTTP status.
	#[error("audio provider returned HTTP status {0}")]
	HttpStatus(StatusCode),
	/// The upstream response media type did not match the operation.
	#[error("audio provider returned invalid Content-Type: {0}")]
	ContentType(SmolStr),
	/// The upstream response body failed or was empty.
	#[error("audio provider response body failed: {0}")]
	Body(SmolStr),
	/// The codec rejected a complete transcription response.
	#[error("audio provider response decoding failed: {0}")]
	Decode(OpenAiAudioError),
}

/// One catalog-selected OpenAI-compatible audio provider attempt.
///
/// The adapter never stores credential bytes and never retries. Dropping a
/// returned speech stream drops its upstream HTTP body, structurally closing
/// the in-flight response.
#[derive(Clone)]
pub struct AudioProviderAttempt<S> {
	provider: ProviderEntry,
	route:    ProviderRoute,
	codec:    OpenAiAudioCodec,
	egress:   S,
}

impl<S> AudioProviderAttempt<S> {
	/// Builds an audio attempt for an advertised OpenAI-compatible catalog row.
	pub fn new(
		provider: ProviderEntry,
		route: ProviderRoute,
		egress: S,
	) -> Result<Self, AudioAttemptBuildError> {
		if !matches!(provider.transport, TransportId::OpenAiChat | TransportId::OpenAiResponses) {
			return Err(AudioAttemptBuildError::UnsupportedTransport(provider.transport));
		}
		Ok(Self { provider, route, codec: OpenAiAudioCodec, egress })
	}

	/// Returns the catalog provider identifier used for auth isolation.
	#[must_use]
	pub fn provider_id(&self) -> &str {
		self.provider.id.as_str()
	}

	/// Starts one streamed speech attempt.
	pub async fn speak<B>(
		&self,
		request: SpeakRequest,
	) -> Result<BoxStream<'static, SpeakEvent>, AudioAttemptError>
	where
		S: Service<Request<Body>, Response = Response<B>> + Clone + Send + 'static,
		S::Future: Send + 'static,
		S::Error: fmt::Display + Send + 'static,
		B: HttpBody<Data = Bytes> + Send + Unpin + 'static,
		B::Error: fmt::Display + Send + 'static,
	{
		let profile = match self.provider.compat.audio_api_version {
			AudioApiVersion::None => OpenAiAudioProfile::Standard,
			AudioApiVersion::V2025_04_01Preview => OpenAiAudioProfile::DeploymentPath,
		};
		let encoded = self
			.codec
			.encode_speech_for(&request, profile)
			.map_err(AudioAttemptError::Encode)?;
		let fallback_mime = encoding_mime(request.encoding);
		let response = self.dispatch(&request.model, encoded).await?;
		let status = response.status();
		if !status.is_success() {
			return Err(AudioAttemptError::HttpStatus(status));
		}
		let mime = response_mime(response.headers(), fallback_mime, true)?;
		let mut body = response.into_body();
		let first = next_data(&mut body).await?;
		if first.is_empty() {
			return Err(AudioAttemptError::Body("empty speech response".into()));
		}
		let provider_id = self.provider.id.clone();
		let model = request.model.clone();
		let stream = async_stream::stream! {
			let mut utterance = BytesMut::new();
			utterance.extend_from_slice(&first);
			yield SpeakEvent::Chunk(SpeakChunk::builder()
				.audio(first)
				.transcript_delta(SmolStr::default())
				.build());
			while let Some(frame) = body.frame().await {
				let Ok(frame) = frame else { return };
				let Ok(chunk) = frame.into_data() else { continue };
				if chunk.is_empty() { continue; }
				utterance.extend_from_slice(&chunk);
				yield SpeakEvent::Chunk(SpeakChunk::builder()
					.audio(chunk)
					.transcript_delta(SmolStr::default())
					.build());
			}
			let utterance = utterance.freeze();
			let mut props = Props::default();
			props.insert_ns("audio", "provider", provider_id.as_str().into());
			props.insert_ns("audio", "model", model.as_str().into());
			yield SpeakEvent::Done(SpeakDone::builder()
				.audio(BlobPart::builder()
					.hash(*blake3::hash(&utterance).as_bytes())
					.mime(mime)
					.size(utterance.len() as u64)
					.inline(utterance)
					.build())
				.duration_ms(0)
				.maybe_usage(None)
				.maybe_cost(None)
				.unsupported(Vec::new())
				.props(props)
				.build());
		};
		Ok(stream.boxed())
	}

	/// Executes one multipart transcription attempt.
	pub async fn transcribe<B>(
		&self,
		request: TranscribeRequest,
	) -> Result<TranscribeResponse, AudioAttemptError>
	where
		S: Service<Request<Body>, Response = Response<B>> + Clone + Send + 'static,
		S::Future: Send + 'static,
		S::Error: fmt::Display + Send + 'static,
		B: HttpBody<Data = Bytes> + Send + Unpin + 'static,
		B::Error: fmt::Display + Send + 'static,
	{
		let encoded = self
			.codec
			.encode_transcription(&request)
			.map_err(AudioAttemptError::Encode)?;
		let response = self.dispatch(&request.model, encoded).await?;
		let status = response.status();
		if !status.is_success() {
			return Err(AudioAttemptError::HttpStatus(status));
		}
		response_mime(response.headers(), "application/json", false)?;
		let body = response
			.into_body()
			.collect()
			.await
			.map_err(|error| AudioAttemptError::Body(error.to_string().into()))?
			.to_bytes();
		if body.is_empty() {
			return Err(AudioAttemptError::Body("empty transcription response".into()));
		}
		let mut decoded = self
			.codec
			.decode_transcription(&body)
			.map_err(AudioAttemptError::Decode)?;
		decoded
			.props
			.insert_ns("audio", "provider", self.provider.id.as_str().into());
		decoded
			.props
			.insert_ns("audio", "model", request.model.as_str().into());
		if decoded.language.is_empty() {
			decoded.language = request.language;
		}
		Ok(decoded)
	}

	async fn dispatch<B>(
		&self,
		model: &str,
		encoded: EncodedAudioRequest,
	) -> Result<Response<B>, AudioAttemptError>
	where
		S: Service<Request<Body>, Response = Response<B>> + Clone + Send + 'static,
		S::Future: Send + 'static,
		S::Error: fmt::Display + Send + 'static,
	{
		let endpoint = endpoint(&self.provider, &self.route, model, encoded.path)?;
		let mut builder = Request::post(endpoint)
			.header(header::CONTENT_TYPE, encoded.content_type.as_str())
			.header(header::ACCEPT, encoded.accept);
		for (name, value) in &self.provider.headers {
			builder = builder.header(name.as_str(), value.as_str());
		}
		let mut request = builder
			.body(Full::new(encoded.body))
			.map_err(|error| AudioAttemptError::HttpRequest(error.to_string().into()))?;
		request
			.extensions_mut()
			.insert(AuthContext::new(self.provider.id.as_str()));
		self
			.egress
			.clone()
			.oneshot(request)
			.await
			.map_err(|error| AudioAttemptError::Egress(error.to_string().into()))
	}
}

fn endpoint(
	provider: &ProviderEntry,
	route: &ProviderRoute,
	model: &str,
	path: &str,
) -> Result<String, AudioAttemptError> {
	let deployment = if route.deployment.is_empty() {
		model
	} else {
		route.deployment.as_str()
	};
	let base = expand_base_url(
		&provider.base_url,
		BaseUrlVars::builder()
			.region(route.region.as_str())
			.location(route.region.as_str())
			.project(route.project.as_str())
			.deployment(deployment)
			.model(model)
			.account(route.account.as_str())
			.gateway(route.gateway.as_str())
			.build(),
	)
	.map_err(|error| AudioAttemptError::Endpoint(format_smol!("{error}")))?;
	let separator = match provider.compat.audio_api_version {
		AudioApiVersion::None => "",
		AudioApiVersion::V2025_04_01Preview => "?api-version=2025-04-01-preview",
	};
	Ok(format!("{}{path}{separator}", base.trim_end_matches('/')))
}

async fn next_data<B>(body: &mut B) -> Result<Bytes, AudioAttemptError>
where
	B: HttpBody<Data = Bytes> + Unpin,
	B::Error: fmt::Display,
{
	while let Some(frame) = body.frame().await {
		let frame = frame.map_err(|error| AudioAttemptError::Body(error.to_string().into()))?;
		if let Ok(data) = frame.into_data()
			&& !data.is_empty()
		{
			return Ok(data);
		}
	}
	Ok(Bytes::new())
}

fn response_mime(
	headers: &http::HeaderMap,
	fallback: &'static str,
	audio: bool,
) -> Result<SmolStr, AudioAttemptError> {
	let value = headers
		.get(header::CONTENT_TYPE)
		.and_then(|value| value.to_str().ok())
		.unwrap_or(fallback)
		.split(';')
		.next()
		.unwrap_or(fallback)
		.trim();
	let valid = if audio {
		value == fallback
			|| value == "application/octet-stream"
			|| (fallback == "audio/wav" && value == "audio/x-wav")
			|| (fallback == "audio/mpeg" && value == "audio/mp3")
	} else {
		value == "application/json" || value.ends_with("+json")
	};
	if !valid {
		return Err(AudioAttemptError::ContentType(value.into()));
	}
	Ok(if audio && value != fallback {
		fallback.into()
	} else {
		value.into()
	})
}

const fn encoding_mime(encoding: AudioEncoding) -> &'static str {
	match encoding {
		AudioEncoding::Mp3 => "audio/mpeg",
		AudioEncoding::Pcm16 => "audio/pcm",
		AudioEncoding::Wav => "audio/wav",
		AudioEncoding::Opus => "audio/ogg",
		AudioEncoding::Aac => "audio/aac",
		AudioEncoding::Flac => "audio/flac",
		_ => "application/octet-stream",
	}
}
