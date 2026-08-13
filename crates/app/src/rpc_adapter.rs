//! Tonic transport projection over the typed inference registry.

use std::{
	collections::BTreeMap,
	pin::Pin,
	sync::Arc,
	time::{Duration, SystemTime, UNIX_EPOCH},
};

use bytes::Bytes;
use futures::{Stream, StreamExt as _, stream};
use omp_core::Str;
use omp_llm_catalog::{
	ModelAvailability, ModelKey, ModelSpec, OperationKind, ProviderDef, ProviderId,
};
use omp_llm_inference::{
	Client, Registry,
	answer::{
		Artifact, ArtifactBody, AudioChunk, GenerationEvent, ImageArtifact, NativeResponse,
		NativeResponseBody, RealtimeEvent as CanonicalRealtimeEvent, RealtimeInput, SearchResults,
		TranscriptEvent, UsageReport, UsageWindowKind, VideoArtifact,
	},
	call::{
		AudioFormat, Background, CallMeta, ContentPart, CountAccuracy, CountTokensRequest,
		DetokenizeRequest, Dimensions, EmbedRequest, EmbeddingInput, ImageFormat, ImageQuality,
		ImageRequest, MediaInput, Message, NativeMethod, NativePath, NativePayload, NativeRequest,
		NativeResponseFraming, NegotiationPolicy, OpaqueJson, RawJson, RealtimeModality,
		RealtimeRequest, Role, Sampling, SearchRecency, SearchRequest, Setting, SpeechRequest,
		Target, TimestampGranularity, TokenizeRequest, ToolChoice, ToolDefinition,
		TranscriptionRequest, TruncationPolicy, UsageRequest, UsageScope, VideoRequest,
	},
	error::{Error, ErrorKind},
	event::{BlockKind, ChatEvent, FinishReason},
	id::{AccountId, RequestId, ToolCallId},
	operation::job::{JobCancelError, JobCancellationReceipt},
	receipt::{Cost, ExecutionBudget, Usage, UsageSource},
	router::Router,
};
use omp_proto::{inference::v1 as pb, thread::v1 as thread_pb};
use parking_lot::Mutex;
use tonic::{Request, Response, Status};

/// Stream returned by RPC methods whose typed operation produces events.
pub type RpcStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;

/// Projects the canonical catalog and typed operation service onto the retained
/// OMP RPC schema.
#[derive(Clone)]
pub struct InferenceRpc {
	registry:    Registry,
	epoch:       Arc<[u8]>,
	contexts:    Arc<Mutex<BTreeMap<String, RpcContext>>>,
	generations: Arc<Mutex<BTreeMap<String, RpcGeneration>>>,
}

#[derive(Clone, Default)]
struct RpcContext {
	revision: u64,
	messages: Vec<Message>,
}

#[derive(Clone)]
struct RpcGeneration {
	status:  Arc<Mutex<pb::GenerationStatus>>,
	updates: tokio::sync::broadcast::Sender<pb::GenerationStatus>,
	cancel:
		flume::Sender<tokio::sync::oneshot::Sender<Result<JobCancellationReceipt, JobCancelError>>>,
}

impl InferenceRpc {
	/// Creates an RPC projection over one immutable registry generation.
	#[must_use]
	pub fn new(registry: Registry) -> Self {
		let epoch = format!("{}:{}", registry.catalog_revision(), registry.generation()).into_bytes();
		Self {
			registry,
			epoch: epoch.into(),
			contexts: Arc::new(Mutex::new(BTreeMap::new())),
			generations: Arc::new(Mutex::new(BTreeMap::new())),
		}
	}

	fn cursor(&self) -> pb::Cursor {
		pb::Cursor {
			epoch:      self.epoch.as_ref().to_vec().into(),
			generation: self.registry.generation(),
		}
	}

	fn list_models_response(&self, request: &pb::ListModelsRequest) -> pb::ListModelsResponse {
		let requested_facet = pb::Facet::try_from(request.facet).unwrap_or(pb::Facet::Unspecified);
		let models = self
			.registry
			.catalog()
			.models()
			.iter()
			.filter_map(|model| {
				let provider = model
					.routes
					.first()
					.and_then(|route| self.registry.catalog().route(route))
					.map(|route| route.provider.clone())?;
				if !request.provider.is_empty() && provider.as_str() != request.provider {
					return None;
				}
				if request.available_only && !matches!(model.availability, ModelAvailability::Available)
				{
					return None;
				}
				let facets = model_facets(model);
				if requested_facet != pb::Facet::Unspecified
					&& !facets.contains(&(requested_facet as i32))
				{
					return None;
				}
				Some(model_card(model, provider.as_str(), facets))
			})
			.collect();
		pb::ListModelsResponse { models, cursor: Some(self.cursor()), roles: Default::default() }
	}

	fn target(&self, selector: &str, operation: OperationKind) -> Result<Target, Status> {
		if !selector.is_empty() {
			return Ok(Target::Model(ModelKey::from(selector)));
		}
		self
			.registry
			.catalog()
			.models()
			.iter()
			.find(|model| model.capabilities.operations.contains_kind(operation))
			.map(|model| Target::Model(model.key.clone()))
			.ok_or_else(|| {
				Status::failed_precondition(format!("no catalog target serves {operation}"))
			})
	}

	fn client(
		&self,
		target: Target,
		request: RequestId,
	) -> Client<omp_llm_inference::ProviderService, Router> {
		Client::new(
			self.registry.service(),
			Router::new(self.registry.clone(), Duration::from_secs(30)),
			CallMeta {
				id: request,
				target,
				deadline: None,
				budget: ExecutionBudget::default(),
				session: None,
			},
		)
	}

	fn management_target(
		&self,
		provider: Option<&ProviderId>,
		operation: OperationKind,
	) -> Result<Target, Status> {
		if let Some(provider) = provider {
			return Ok(Target::ProviderService(provider.clone()));
		}
		self
			.registry
			.catalog()
			.providers()
			.iter()
			.find(|provider| provider.management.supports(operation))
			.map(|provider| Target::ProviderService(provider.id.clone()))
			.ok_or_else(|| {
				Status::failed_precondition(format!("no provider service serves {operation}"))
			})
	}

	fn resolve_turn_input(
		&self,
		input: Option<&pb::turn_request::Input>,
	) -> Result<(Vec<Message>, Option<String>, u64), Status> {
		match input {
			Some(pb::turn_request::Input::Seed(seed)) => {
				let thread = seed
					.thread
					.as_ref()
					.ok_or_else(|| Status::invalid_argument("Seed.thread is required"))?;
				let messages = thread_messages(thread)?;
				Ok((messages, (!seed.context_id.is_empty()).then(|| seed.context_id.clone()), 0))
			},
			Some(pb::turn_request::Input::Incremental(incremental)) => {
				let context = incremental
					.context
					.as_ref()
					.ok_or_else(|| Status::invalid_argument("Incremental.context is required"))?;
				let held = self
					.contexts
					.lock()
					.get(&context.context_id)
					.cloned()
					.ok_or_else(|| Status::not_found("context is not held"))?;
				validate_revision(context, held.revision)?;
				let delta = incremental
					.delta
					.as_ref()
					.ok_or_else(|| Status::invalid_argument("Incremental.delta is required"))?;
				let retained = delta.truncate_to.unwrap_or(held.revision);
				if retained > held.revision {
					return Err(Status::invalid_argument("truncate_to exceeds context head"));
				}
				let mut messages = held
					.messages
					.into_iter()
					.take(retained as usize)
					.collect::<Vec<_>>();
				messages.extend(items_messages(&delta.append)?);
				Ok((messages, Some(context.context_id.clone()), retained))
			},
			None => Err(Status::invalid_argument("TurnRequest.input is required")),
		}
	}

	fn generation(&self, generation_id: &str) -> Result<RpcGeneration, Status> {
		if generation_id.is_empty() {
			return Err(Status::invalid_argument("generation_id is required"));
		}
		self
			.generations
			.lock()
			.get(generation_id)
			.cloned()
			.ok_or_else(|| Status::not_found("generation is not held by this daemon"))
	}
}

#[tonic::async_trait]
impl pb::inference_server::Inference for InferenceRpc {
	type AttachGenerationStream = RpcStream<pb::GenerationStatus>;
	type GenerateImageStream = RpcStream<pb::ImageEvent>;
	type NativeStream = RpcStream<pb::NativeChunk>;
	type RealtimeStream = RpcStream<pb::RealtimeEvent>;
	type SpeakStream = RpcStream<pb::SpeakEvent>;
	type TurnStream = RpcStream<pb::TurnEvent>;
	type WatchModelsStream = RpcStream<pb::ModelEvent>;

	async fn turn(
		&self,
		request: Request<tonic::Streaming<pb::TurnFrame>>,
	) -> Result<Response<Self::TurnStream>, Status> {
		let mut incoming = request.into_inner();
		let first = incoming
			.message()
			.await?
			.ok_or_else(|| Status::invalid_argument("Turn requires an opening frame"))?;
		let open = match first.frame {
			Some(pb::turn_frame::Frame::Open(open)) => open,
			_ => return Err(Status::invalid_argument("the first Turn frame must be open")),
		};
		if open.turn_id.is_empty() {
			return Err(Status::invalid_argument("TurnRequest.turn_id is required"));
		}
		let params = open
			.params
			.as_ref()
			.ok_or_else(|| Status::invalid_argument("TurnRequest.params is required"))?;
		let (messages, context_id, base_revision) = self.resolve_turn_input(open.input.as_ref())?;
		let committed_messages = messages.clone();
		let chat = chat_request(messages, params)?;
		let target = self.target(&params.model, OperationKind::Chat)?;
		let mut client = self.client(target, RequestId::from(open.turn_id.as_str()));
		let events = client.execute(chat).await.map_err(inference_status)?;
		let contexts = Arc::clone(&self.contexts);
		let output = turn_events(events, contexts, context_id, base_revision, committed_messages);
		Ok(Response::new(Box::pin(output)))
	}

	async fn realtime(
		&self,
		request: Request<tonic::Streaming<pb::RealtimeFrame>>,
	) -> Result<Response<Self::RealtimeStream>, Status> {
		let mut incoming = request.into_inner();
		let first = incoming
			.message()
			.await?
			.ok_or_else(|| Status::invalid_argument("Realtime requires an opening frame"))?;
		let open = match first.frame {
			Some(pb::realtime_frame::Frame::Open(open)) => open,
			_ => return Err(Status::invalid_argument("the first Realtime frame must be open")),
		};
		if open.request_id.is_empty() || open.model.is_empty() {
			return Err(Status::invalid_argument("RealtimeOpen.request_id and model are required"));
		}
		let operation = RealtimeRequest {
			instructions:   (!open.instructions.is_empty()).then(|| open.instructions.as_str().into()),
			modalities:     open
				.modalities
				.iter()
				.map(|modality| {
					match pb::realtime_open::Modality::try_from(*modality)
						.unwrap_or(pb::realtime_open::Modality::Unspecified)
					{
						pb::realtime_open::Modality::Text => Ok(RealtimeModality::Text),
						pb::realtime_open::Modality::Audio => Ok(RealtimeModality::Audio),
						pb::realtime_open::Modality::Unspecified => {
							Err(Status::invalid_argument("RealtimeOpen modality is required"))
						},
					}
				})
				.collect::<Result<Vec<_>, _>>()?
				.into(),
			voice:          (!open.voice.is_empty()).then(|| open.voice.as_str().into()),
			input_audio:    realtime_audio_format(open.input_audio),
			output_audio:   realtime_audio_format(open.output_audio),
			turn_detection: Setting::Unset,
			tools:          open
				.tools
				.iter()
				.map(tool_definition)
				.collect::<Result<Vec<_>, _>>()?
				.into(),
			negotiation:    NegotiationPolicy::default(),
		};
		let target = self.target(&open.model, OperationKind::Realtime)?;
		let mut client = self.client(target, RequestId::from(open.request_id.as_str()));
		let session = Arc::new(client.execute(operation).await.map_err(inference_status)?);
		let (input_errors, errors) = flume::bounded(1);
		let input_session = Arc::clone(&session);
		tokio::spawn(async move {
			while let Ok(Some(frame)) = incoming.message().await {
				let close = matches!(frame.frame, Some(pb::realtime_frame::Frame::Close(_)));
				let input = match realtime_input(frame) {
					Ok(input) => input,
					Err(error) => {
						let _ = input_errors.send_async(error).await;
						break;
					},
				};
				if let Err(error) = input_session.send(input).await {
					let _ = input_errors
						.send_async(Status::failed_precondition(format!(
							"realtime input was rejected: {error:?}"
						)))
						.await;
					break;
				}
				if close {
					break;
				}
			}
		});
		let mut errors_open = true;
		let output = async_stream::try_stream! {
			loop {
				let event = tokio::select! {
					error = errors.recv_async(), if errors_open => match error {
						Ok(error) => Err(error),
						Err(_) => {
							errors_open = false;
							continue;
						},
					},
					event = session.recv() => match event {
						Ok(Ok(event)) => Ok(event),
						Ok(Err(error)) => Err(inference_status(error)),
						Err(error) => Err(Status::failed_precondition(format!(
							"realtime session receive failed: {error:?}"
						))),
					},
				}?;
				let terminal = matches!(event, CanonicalRealtimeEvent::Closed);
				yield realtime_event(event)?;
				if terminal { break; }
			}
		};
		Ok(Response::new(Box::pin(output)))
	}

	async fn fork(
		&self,
		request: Request<pb::ForkRequest>,
	) -> Result<Response<pb::ForkResponse>, Status> {
		let request = request.into_inner();
		let parent = request
			.parent
			.ok_or_else(|| Status::invalid_argument("ForkRequest.parent is required"))?;
		if request.context_id.is_empty() {
			return Err(Status::invalid_argument("ForkRequest.context_id is required"));
		}
		let mut contexts = self.contexts.lock();
		let source = contexts
			.get(&parent.context_id)
			.cloned()
			.ok_or_else(|| Status::not_found("parent context is not held"))?;
		validate_revision(&parent, source.revision)?;
		let at = request.at.unwrap_or(source.revision);
		if at > source.revision {
			return Err(Status::invalid_argument("fork revision exceeds parent head"));
		}
		if contexts.contains_key(&request.context_id) {
			return Err(Status::already_exists("fork context already exists"));
		}
		let fork = RpcContext {
			revision: at,
			messages: source.messages.into_iter().take(at as usize).collect(),
		};
		contexts.insert(request.context_id.clone(), fork);
		Ok(Response::new(pb::ForkResponse { revision: Some(revision(&request.context_id, at)) }))
	}

	async fn drop(
		&self,
		request: Request<pb::DropRequest>,
	) -> Result<Response<pb::DropResponse>, Status> {
		let context_id = request.into_inner().context_id;
		if context_id.is_empty() {
			return Err(Status::invalid_argument("DropRequest.context_id is required"));
		}
		if self.contexts.lock().remove(&context_id).is_none() {
			return Err(Status::not_found("context is not held"));
		}
		Ok(Response::new(pb::DropResponse {}))
	}

	async fn count_tokens(
		&self,
		request: Request<pb::CountTokensRequest>,
	) -> Result<Response<pb::CountTokensResponse>, Status> {
		let request = request.into_inner();
		let messages = match request.input {
			Some(pb::count_tokens_request::Input::Thread(thread)) => thread_messages(&thread)?,
			Some(pb::count_tokens_request::Input::Context(context)) => {
				let held = self
					.contexts
					.lock()
					.get(&context.context_id)
					.cloned()
					.ok_or_else(|| Status::not_found("context is not held"))?;
				validate_revision(&context, held.revision)?;
				held.messages
			},
			None => return Err(Status::invalid_argument("CountTokensRequest.input is required")),
		};
		let operation = CountTokensRequest {
			messages: messages.into(),
			tools:    request
				.tools
				.iter()
				.map(tool_definition)
				.collect::<Result<Vec<_>, _>>()?
				.into(),

			accuracy: CountAccuracy::AllowEstimate,
		};
		let target = self.target(&request.model, OperationKind::CountTokens)?;
		let mut client = self.client(target, rpc_request_id("count"));
		let answer = client.execute(operation).await.map_err(inference_status)?;
		Ok(Response::new(pb::CountTokensResponse {
			tokens:     answer.tokens,
			accuracy:   if answer.provenance.exact {
				pb::usage::Accuracy::Exact as i32
			} else {
				pb::usage::Accuracy::Estimated as i32
			},
			provenance: Some(tokenizer_provenance(answer.provenance)),
		}))
	}

	async fn tokenize(
		&self,
		request: Request<pb::TokenizeRequest>,
	) -> Result<Response<pb::TokenizeResponse>, Status> {
		let request = request.into_inner();
		let operation = TokenizeRequest {
			text:          request.text.as_str().into(),
			allow_special: request.allow_special,
		};
		let target = self.target(&request.model, OperationKind::Tokenize)?;
		let mut client = self.client(target, rpc_request_id("tokenize"));
		let answer = client.execute(operation).await.map_err(inference_status)?;
		Ok(Response::new(pb::TokenizeResponse {
			tokens:     answer.tokens,
			provenance: Some(tokenizer_provenance(answer.provenance)),
		}))
	}

	async fn detokenize(
		&self,
		request: Request<pb::DetokenizeRequest>,
	) -> Result<Response<pb::DetokenizeResponse>, Status> {
		let request = request.into_inner();
		let operation = DetokenizeRequest { tokens: request.tokens.into(), strict: request.strict };
		let target = self.target(&request.model, OperationKind::Detokenize)?;
		let mut client = self.client(target, rpc_request_id("detokenize"));
		let answer = client.execute(operation).await.map_err(inference_status)?;
		Ok(Response::new(pb::DetokenizeResponse {
			text:       answer.text.as_str().to_owned(),
			provenance: Some(tokenizer_provenance(answer.provenance)),
		}))
	}

	async fn embed(
		&self,
		request: Request<pb::EmbedRequest>,
	) -> Result<Response<pb::EmbedResponse>, Status> {
		let request = request.into_inner();
		if request.texts.is_empty() {
			return Err(Status::invalid_argument("EmbedRequest.texts must not be empty"));
		}
		let operation = EmbedRequest {
			inputs:      request
				.texts
				.iter()
				.map(|text| EmbeddingInput::Text(Str::from(text.as_str())))
				.collect::<Vec<_>>()
				.into(),
			dimensions:  request.dimensions.map_or(Setting::Unset, Setting::Prefer),
			normalize:   Setting::Unset,
			truncation:  TruncationPolicy::Reject,
			negotiation: NegotiationPolicy::default(),
		};
		let target = self.target(&request.model, OperationKind::Embed)?;
		let mut client = self.client(target, rpc_request_id("embed"));
		let answer = client.execute(operation).await.map_err(inference_status)?;
		Ok(Response::new(pb::EmbedResponse {
			vectors: answer
				.embeddings
				.into_iter()
				.map(|embedding| pb::embed_response::Vector { values: embedding.values })
				.collect(),
			usage:   Some(proto_usage(answer.usage)),
		}))
	}

	async fn generate_image(
		&self,
		request: Request<pb::GenerateImageRequest>,
	) -> Result<Response<Self::GenerateImageStream>, Status> {
		let request = request.into_inner();
		if request.prompt.is_empty() {
			return Err(Status::invalid_argument("GenerateImageRequest.prompt is required"));
		}
		let dimensions = request
			.size
			.map(|size| Setting::Prefer(Dimensions { width: size.width, height: size.height }))
			.unwrap_or(Setting::Unset);
		let quality = match pb::generate_image_request::Quality::try_from(request.quality)
			.unwrap_or(pb::generate_image_request::Quality::Unspecified)
		{
			pb::generate_image_request::Quality::Low => Setting::Prefer(ImageQuality::Draft),
			pb::generate_image_request::Quality::Medium => Setting::Prefer(ImageQuality::Standard),
			pb::generate_image_request::Quality::High => Setting::Prefer(ImageQuality::High),
			pb::generate_image_request::Quality::Unspecified => Setting::Unset,
		};
		let format = match pb::generate_image_request::Format::try_from(request.format)
			.unwrap_or(pb::generate_image_request::Format::Unspecified)
		{
			pb::generate_image_request::Format::Png => Setting::Prefer(ImageFormat::Png),
			pb::generate_image_request::Format::Webp => Setting::Prefer(ImageFormat::Webp),
			pb::generate_image_request::Format::Jpeg => Setting::Prefer(ImageFormat::Jpeg),
			pb::generate_image_request::Format::Svg => {
				return Err(Status::invalid_argument("SVG is not a canonical generated image format"));
			},
			pb::generate_image_request::Format::Unspecified => Setting::Unset,
		};
		let background = match pb::generate_image_request::Background::try_from(request.background)
			.unwrap_or(pb::generate_image_request::Background::Unspecified)
		{
			pb::generate_image_request::Background::Opaque => Setting::Prefer(Background::Opaque),
			pb::generate_image_request::Background::Transparent => {
				Setting::Prefer(Background::Transparent)
			},
			pb::generate_image_request::Background::Unspecified => Setting::Unset,
		};
		let operation = ImageRequest {
			prompt: request.prompt.as_str().into(),
			references: request
				.input_images
				.iter()
				.map(media_input)
				.collect::<Result<Vec<_>, _>>()?
				.into(),
			mask: None,
			count: request.n.max(1),
			dimensions,
			quality,
			background,
			format,
			style: Setting::Unset,
			safety: Arc::from([]),
			seed: request.seed,
			negotiation: NegotiationPolicy::default(),
		};
		let target = self.target(&request.model, OperationKind::GenerateImage)?;
		let mut client = self.client(target, rpc_request_id("image"));
		let events = client.execute(operation).await.map_err(inference_status)?;
		Ok(Response::new(Box::pin(image_events(events))))
	}

	async fn speak(
		&self,
		request: Request<pb::SpeakRequest>,
	) -> Result<Response<Self::SpeakStream>, Status> {
		let request = request.into_inner();
		if request.text.is_empty() || request.voice.is_empty() {
			return Err(Status::invalid_argument("SpeakRequest.text and voice are required"));
		}
		let format = match pb::AudioEncoding::try_from(request.encoding)
			.unwrap_or(pb::AudioEncoding::Unspecified)
		{
			pb::AudioEncoding::Mp3 => Setting::Prefer(omp_llm_inference::call::AudioFormat::Mp3),
			pb::AudioEncoding::Pcm16 => Setting::Prefer(omp_llm_inference::call::AudioFormat::Pcm16),
			pb::AudioEncoding::Wav => Setting::Prefer(omp_llm_inference::call::AudioFormat::Wav),
			pb::AudioEncoding::Opus => Setting::Prefer(omp_llm_inference::call::AudioFormat::Opus),
			pb::AudioEncoding::Aac => Setting::Prefer(omp_llm_inference::call::AudioFormat::Aac),
			pb::AudioEncoding::Flac => Setting::Prefer(omp_llm_inference::call::AudioFormat::Flac),
			pb::AudioEncoding::Unspecified => Setting::Unset,
		};
		let operation = SpeechRequest {
			text: request.text.as_str().into(),
			voice: request.voice.as_str().into(),
			format,
			sample_rate_hz: request
				.sample_rate_hz
				.map_or(Setting::Unset, Setting::Prefer),
			speed: request
				.speed
				.map_or(Setting::Unset, |speed| Setting::Prefer(speed as f32)),
			timestamps: Setting::Unset,
			negotiation: NegotiationPolicy::default(),
		};
		let target = self.target(&request.model, OperationKind::Speak)?;
		let mut client = self.client(target, rpc_request_id("speak"));
		let events = client.execute(operation).await.map_err(inference_status)?;
		let output = events.map(|event| event.map(speak_event).map_err(inference_status));
		Ok(Response::new(Box::pin(output)))
	}

	async fn transcribe(
		&self,
		request: Request<pb::TranscribeRequest>,
	) -> Result<Response<pb::TranscribeResponse>, Status> {
		let request = request.into_inner();
		let audio = request
			.audio
			.as_ref()
			.ok_or_else(|| Status::invalid_argument("TranscribeRequest.audio is required"))
			.and_then(media_input)?;
		let granularity = if request.granularities.iter().any(|value| {
			pb::transcribe_request::Granularity::try_from(*value)
				.is_ok_and(|value| value == pb::transcribe_request::Granularity::Word)
		}) {
			Setting::Prefer(TimestampGranularity::Word)
		} else if request.granularities.iter().any(|value| {
			pb::transcribe_request::Granularity::try_from(*value)
				.is_ok_and(|value| value == pb::transcribe_request::Granularity::Segment)
		}) {
			Setting::Prefer(TimestampGranularity::Segment)
		} else {
			Setting::Unset
		};
		let operation = TranscriptionRequest {
			audio,
			language: (!request.language.is_empty()).then(|| request.language.as_str().into()),
			translate_to_english: request.translate,
			diarization: request
				.diarize
				.then_some(Setting::Require(true))
				.unwrap_or(Setting::Unset),
			timestamps: granularity,
			prompt: (!request.prompt.is_empty()).then(|| request.prompt.as_str().into()),
			negotiation: NegotiationPolicy::default(),
		};
		let target = self.target(&request.model, OperationKind::Transcribe)?;
		let mut client = self.client(target, rpc_request_id("transcribe"));
		let mut events = client.execute(operation).await.map_err(inference_status)?;
		let mut response = pb::TranscribeResponse::default();
		while let Some(event) = events.next().await {
			match event.map_err(inference_status)? {
				TranscriptEvent::Started { language } => {
					if let Some(language) = language {
						response.language = language.as_str().to_owned();
					}
				},
				TranscriptEvent::TextDelta { .. } => {},
				TranscriptEvent::Segment { text, start_ms, end_ms, speaker, .. } => {
					response.segments.push(pb::transcribe_response::Segment {
						start_ms,
						end_ms,
						text: text.as_str().to_owned(),
						speaker: speaker.map(|speaker| speaker.index),
						confidence: None,
					});
				},
				TranscriptEvent::Word { text, start_ms, end_ms, speaker, .. } => {
					response.words.push(pb::transcribe_response::Word {
						start_ms,
						end_ms,
						word: text.as_str().to_owned(),
						speaker: speaker.map(|speaker| speaker.index),
					});
				},
				TranscriptEvent::Completed { text, usage } => {
					response.text = text.as_str().to_owned();
					response.usage = Some(proto_usage(usage));
				},
			}
		}
		Ok(Response::new(response))
	}

	async fn search(
		&self,
		request: Request<pb::SearchRequest>,
	) -> Result<Response<pb::SearchResponse>, Status> {
		let request = request.into_inner();
		if request.query.is_empty() {
			return Err(Status::invalid_argument("SearchRequest.query is required"));
		}
		let recency = match pb::search_request::Recency::try_from(request.recency)
			.unwrap_or(pb::search_request::Recency::Unspecified)
		{
			pb::search_request::Recency::Day => Some(SearchRecency::Day),
			pb::search_request::Recency::Week => Some(SearchRecency::Week),
			pb::search_request::Recency::Month => Some(SearchRecency::Month),
			pb::search_request::Recency::Year => Some(SearchRecency::Year),
			pb::search_request::Recency::Unspecified => None,
		};
		let locale = match (request.language.is_empty(), request.country.is_empty()) {
			(false, false) => Some(Str::from(format!("{}-{}", request.language, request.country))),
			(false, true) => Some(request.language.as_str().into()),
			(true, false) => Some(request.country.as_str().into()),
			(true, true) => None,
		};
		let operation = SearchRequest {
			query: request.query.as_str().into(),
			include_domains: request
				.allowed_domains
				.iter()
				.map(|value| Str::from(value.as_str()))
				.collect::<Vec<_>>()
				.into(),
			exclude_domains: request
				.excluded_domains
				.iter()
				.map(|value| Str::from(value.as_str()))
				.collect::<Vec<_>>()
				.into(),
			recency,
			locale,
			max_results: request.limit,
			synthesize_answer: Setting::Prefer(true),
			negotiation: NegotiationPolicy::default(),
		};
		let target = if request.engine.is_empty() {
			self.target("", OperationKind::Search)?
		} else {
			Target::ProviderService(ProviderId::from(request.engine.as_str()))
		};
		let mut client = self.client(target, rpc_request_id("search"));
		let answer = client.execute(operation).await.map_err(inference_status)?;
		Ok(Response::new(search_response(answer)))
	}

	async fn generate_video(
		&self,
		request: Request<pb::GenerateVideoRequest>,
	) -> Result<Response<pb::GenerationStatus>, Status> {
		let request = request.into_inner();
		if request.prompt.is_empty() {
			return Err(Status::invalid_argument("GenerateVideoRequest.prompt is required"));
		}
		if request.end_frame.is_some() || !request.references.is_empty() {
			return Err(Status::invalid_argument(
				"end-frame and multi-reference video inputs have no canonical VideoRequest projection",
			));
		}
		let operation = VideoRequest {
			prompt:            request.prompt.as_str().into(),
			reference:         request.start_frame.as_ref().map(media_input).transpose()?,
			duration_ms:       request
				.duration_seconds
				.map_or(Setting::Unset, |seconds| Setting::Prefer(u64::from(seconds) * 1_000)),
			dimensions:        video_dimensions(request.resolution, request.aspect_ratio),
			frames_per_second: Setting::Unset,
			audio:             request.audio.map_or(Setting::Unset, Setting::Prefer),
			safety:            Arc::from([]),
			seed:              request.seed,
			negotiation:       NegotiationPolicy::default(),
		};
		let target = self.target(&request.model, OperationKind::GenerateVideo)?;
		let mut client = self.client(target, rpc_request_id("video"));
		let session = client.execute(operation).await.map_err(inference_status)?;
		let checkpoint = session.checkpoint();
		let generation_id = checkpoint.job.handle.as_str().to_owned();
		let created_at_ms = system_time_ms(checkpoint.created_at);
		let initial = pb::GenerationStatus {
			generation_id: generation_id.clone(),
			state: pb::generation_status::State::Queued as i32,
			progress_percent: 0.0,
			detail: String::new(),
			artifacts: Vec::new(),
			usage: None,
			cost: None,
			unsupported: Vec::new(),
			created_at_ms,
			updated_at_ms: created_at_ms,
			props: None,
		};
		let status = Arc::new(Mutex::new(initial.clone()));
		let (updates, _) = tokio::sync::broadcast::channel(32);
		let (cancel, cancel_rx) = flume::bounded(1);
		self
			.generations
			.lock()
			.insert(generation_id, RpcGeneration {
				status: Arc::clone(&status),
				updates: updates.clone(),
				cancel,
			});
		tokio::spawn(run_generation(session, status, updates, cancel_rx));
		Ok(Response::new(initial))
	}

	async fn get_generation(
		&self,
		request: Request<pb::GetGenerationRequest>,
	) -> Result<Response<pb::GenerationStatus>, Status> {
		let generation = self.generation(&request.into_inner().generation_id)?;
		let status = generation.status.lock().clone();
		Ok(Response::new(status))
	}

	async fn attach_generation(
		&self,
		request: Request<pb::AttachGenerationRequest>,
	) -> Result<Response<Self::AttachGenerationStream>, Status> {
		let generation = self.generation(&request.into_inner().generation_id)?;
		let initial = generation.status.lock().clone();
		let mut receiver = generation.updates.subscribe();
		let output = async_stream::try_stream! {
			yield initial;
			loop {
				match receiver.recv().await {
					Ok(status) => {
						let terminal = matches!(
							pb::generation_status::State::try_from(status.state),
							Ok(pb::generation_status::State::Completed
								| pb::generation_status::State::Failed
								| pb::generation_status::State::Cancelled)
						);
						yield status;
						if terminal { break; }
					},
					Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
						Err(Status::resource_exhausted("generation attachment fell behind"))?
					},
					Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
				}
			}
		};
		Ok(Response::new(Box::pin(output)))
	}

	async fn cancel_generation(
		&self,
		request: Request<pb::CancelGenerationRequest>,
	) -> Result<Response<pb::GenerationStatus>, Status> {
		let generation = self.generation(&request.into_inner().generation_id)?;
		let (reply, result) = tokio::sync::oneshot::channel();
		generation
			.cancel
			.send_async(reply)
			.await
			.map_err(|_| Status::failed_precondition("generation actor has stopped"))?;
		result
			.await
			.map_err(|_| {
				Status::failed_precondition("generation cancellation acknowledgement closed")
			})?
			.map_err(|error| {
				Status::failed_precondition(format!("generation cancellation failed: {error:?}"))
			})?;
		let status = generation.status.lock().clone();
		Ok(Response::new(status))
	}

	async fn usage(
		&self,
		request: Request<pb::UsageRequest>,
	) -> Result<Response<pb::UsageResponse>, Status> {
		let request = request.into_inner();
		let provider =
			(!request.provider.is_empty()).then(|| ProviderId::from(request.provider.as_str()));
		let operation = UsageRequest {
			provider:    provider.clone(),
			account:     (!request.account.is_empty())
				.then(|| AccountId::from(request.account.as_str())),
			scope:       match pb::usage_request::Scope::try_from(request.scope)
				.unwrap_or(pb::usage_request::Scope::Unspecified)
			{
				pb::usage_request::Scope::Unspecified | pb::usage_request::Scope::Current => {
					UsageScope::Current
				},
				pb::usage_request::Scope::Billing => UsageScope::Billing,
				pb::usage_request::Scope::RateLimit => UsageScope::RateLimit,
				pb::usage_request::Scope::All => UsageScope::All,
			},
			allow_stale: request.allow_stale,
		};
		let target = self.management_target(provider.as_ref(), OperationKind::Usage)?;
		let mut client = self.client(target, rpc_request_id("usage"));
		let answer = client.execute(operation).await.map_err(inference_status)?;
		Ok(Response::new(usage_response(answer)))
	}

	async fn native(
		&self,
		request: Request<pb::NativeRequest>,
	) -> Result<Response<Self::NativeStream>, Status> {
		let request = request.into_inner();
		let method = match pb::native_request::Method::try_from(request.method)
			.unwrap_or(pb::native_request::Method::Unspecified)
		{
			pb::native_request::Method::Get => NativeMethod::Get,
			pb::native_request::Method::Post => NativeMethod::Post,
			pb::native_request::Method::Delete => NativeMethod::Delete,
			pb::native_request::Method::Unspecified => {
				return Err(Status::invalid_argument("NativeRequest.method is required"));
			},
		};
		let path = match pb::native_request::Path::try_from(request.path)
			.unwrap_or(pb::native_request::Path::Unspecified)
		{
			pb::native_request::Path::ChatCompletions => NativePath::ChatCompletions,
			pb::native_request::Path::Responses => NativePath::Responses,
			pb::native_request::Path::Messages => NativePath::Messages,
			pb::native_request::Path::MessageTokenCounts => NativePath::MessageTokenCounts,
			pb::native_request::Path::Embeddings => NativePath::Embeddings,
			pb::native_request::Path::ImageGenerations => NativePath::ImageGenerations,
			pb::native_request::Path::AudioSpeech => NativePath::AudioSpeech,
			pb::native_request::Path::AudioTranscriptions => NativePath::AudioTranscriptions,
			pb::native_request::Path::RealtimeSessions => NativePath::RealtimeSessions,
			pb::native_request::Path::Models => NativePath::Models,
			pb::native_request::Path::Usage => NativePath::Usage,
			pb::native_request::Path::Unspecified => {
				return Err(Status::invalid_argument("NativeRequest.path is required"));
			},
		};
		let maximum = request.max_response_bytes.max(1);
		let payload = match request.payload {
			Some(pb::native_request::Payload::Json(bytes)) => Some(NativePayload::Json(
				RawJson::new(bytes, maximum)
					.map_err(|error| Status::invalid_argument(error.to_string()))?,
			)),
			Some(pb::native_request::Payload::Bytes(bytes)) => Some(NativePayload::Bytes(bytes)),
			None => None,
		};
		let response_framing = match pb::native_request::Framing::try_from(request.framing)
			.unwrap_or(pb::native_request::Framing::Unspecified)
		{
			pb::native_request::Framing::Json => NativeResponseFraming::Json,
			pb::native_request::Framing::Sse => NativeResponseFraming::Sse,
			pb::native_request::Framing::Bytes => NativeResponseFraming::Bytes,
			pb::native_request::Framing::Unspecified => {
				return Err(Status::invalid_argument("NativeRequest.framing is required"));
			},
		};
		let operation =
			NativeRequest { method, path, payload, response_framing, max_response_bytes: maximum };
		let target = self.target(&request.model, OperationKind::Native)?;
		let mut client = self.client(target, rpc_request_id("native"));
		let answer = client.execute(operation).await.map_err(inference_status)?;
		Ok(Response::new(Box::pin(native_response_stream(answer))))
	}

	async fn list_providers(
		&self,
		request: Request<pb::ListProvidersRequest>,
	) -> Result<Response<pb::ListProvidersResponse>, Status> {
		let requested_facet =
			pb::Facet::try_from(request.into_inner().facet).unwrap_or(pb::Facet::Unspecified);
		let providers = self
			.registry
			.catalog()
			.providers()
			.iter()
			.filter_map(|provider| {
				let card = provider_card(&self.registry, provider);
				(requested_facet == pb::Facet::Unspecified
					|| card.facets.contains(&(requested_facet as i32)))
				.then_some(card)
			})
			.collect();
		Ok(Response::new(pb::ListProvidersResponse { providers, cursor: Some(self.cursor()) }))
	}

	async fn list_models(
		&self,
		request: Request<pb::ListModelsRequest>,
	) -> Result<Response<pb::ListModelsResponse>, Status> {
		Ok(Response::new(self.list_models_response(&request.into_inner())))
	}

	async fn watch_models(
		&self,
		_request: Request<pb::WatchModelsRequest>,
	) -> Result<Response<Self::WatchModelsStream>, Status> {
		let event = pb::ModelEvent {
			cursor: Some(self.cursor()),
			event:  Some(pb::model_event::Event::Reset(pb::model_event::Reset {})),
		};
		Ok(Response::new(Box::pin(stream::once(async move { Ok(event) }))))
	}

	async fn refresh_models(
		&self,
		request: Request<pb::RefreshModelsRequest>,
	) -> Result<Response<pb::ListModelsResponse>, Status> {
		let provider = request.into_inner().provider;
		Ok(Response::new(self.list_models_response(&pb::ListModelsRequest {
			provider,
			facet: pb::Facet::Unspecified as i32,
			available_only: false,
		})))
	}
}

fn provider_card(registry: &Registry, provider: &ProviderDef) -> pb::ProviderCard {
	let models = registry
		.catalog()
		.models()
		.iter()
		.filter(|model| {
			model.routes.iter().any(|route| {
				registry
					.catalog()
					.route(route)
					.is_some_and(|route| route.provider == provider.id)
			})
		})
		.collect::<Vec<_>>();
	let facets = models
		.iter()
		.flat_map(|model| model_facets(model))
		.collect::<std::collections::BTreeSet<_>>()
		.into_iter()
		.collect();
	pb::ProviderCard {
		id: provider.id.as_str().to_owned(),
		name: provider.name.as_str().to_owned(),
		facets,
		auth: Vec::new(),
		credentialed: provider
			.routes
			.iter()
			.any(|route| registry.contains_service(route)),
		model_count: models.len().try_into().unwrap_or(u32::MAX),
		props: None,
	}
}

fn model_card(model: &ModelSpec, provider: &str, facets: Vec<i32>) -> pb::ModelCard {
	pb::ModelCard {
		id: model.key.as_str().to_owned(),
		provider: provider.to_owned(),
		model: model.key.as_str().to_owned(),
		name: model.display_name.as_str().to_owned(),
		family: model.family.as_str().to_owned(),
		facets,
		inputs: Vec::new(),
		outputs: Vec::new(),
		reasoning: model.thinking.is_some(),
		efforts: Vec::new(),
		context_window: model.limits.context_window.unwrap_or_default(),
		max_output_tokens: model.limits.maximum_output_tokens.unwrap_or_default(),
		pricing: Vec::new(),
		availability: match model.availability {
			ModelAvailability::Unspecified => pb::Availability::Unspecified,
			ModelAvailability::Available => pb::Availability::Available,
			ModelAvailability::LoginRequired => pb::Availability::LoginRequired,
			ModelAvailability::Blocked => pb::Availability::Blocked,
			ModelAvailability::Disabled => pb::Availability::Disabled,
		} as i32,
		source: pb::model_card::Source::Bundled as i32,
		blocked_until_ms: model.provenance.blocked_until_ms.unwrap_or_default(),
		deprecated: model.provenance.deprecated,
		updated_at_ms: model.provenance.updated_at_ms.unwrap_or_default(),
		props: None,
	}
}

fn model_facets(model: &ModelSpec) -> Vec<i32> {
	[
		(OperationKind::Chat, pb::Facet::Chat),
		(OperationKind::Embed, pb::Facet::Embed),
		(OperationKind::GenerateImage, pb::Facet::ImageGen),
		(OperationKind::GenerateVideo, pb::Facet::VideoGen),
		(OperationKind::Speak, pb::Facet::Speak),
		(OperationKind::Transcribe, pb::Facet::Transcribe),
		(OperationKind::Realtime, pb::Facet::Realtime),
		(OperationKind::Search, pb::Facet::Search),
	]
	.into_iter()
	.filter_map(|(operation, facet)| {
		model
			.capabilities
			.operations
			.contains_kind(operation)
			.then_some(facet as i32)
	})
	.collect()
}

fn rpc_request_id(prefix: &str) -> RequestId {
	use std::sync::atomic::{AtomicU64, Ordering};
	static NEXT: AtomicU64 = AtomicU64::new(1);
	RequestId::from(format!("{prefix}-{}", NEXT.fetch_add(1, Ordering::Relaxed)))
}

fn inference_status(error: Error) -> Status {
	let request = error
		.request_id
		.as_ref()
		.map_or("<unassigned>", |request| request.as_str());
	let message = format!("{:?} during {:?} (request {request})", error.kind, error.phase,);
	match error.kind {
		ErrorKind::Cancelled => Status::cancelled(message),
		ErrorKind::DeadlineExceeded => Status::deadline_exceeded(message),
		ErrorKind::InvalidRequest
		| ErrorKind::CodecMismatch
		| ErrorKind::CapabilityMismatch
		| ErrorKind::NativeRequestRejected => Status::invalid_argument(message),
		ErrorKind::TargetNotFound => Status::not_found(message),
		ErrorKind::Authentication => Status::unauthenticated(message),
		ErrorKind::Authorization | ErrorKind::AccountDisabled | ErrorKind::PaymentRequired => {
			Status::permission_denied(message)
		},
		ErrorKind::RateLimited
		| ErrorKind::QuotaExhausted
		| ErrorKind::BudgetExhausted
		| ErrorKind::ResourceExhausted => Status::resource_exhausted(message),
		ErrorKind::SessionConflict | ErrorKind::StalePlan => Status::aborted(message),
		ErrorKind::RouteUnavailable
		| ErrorKind::LocalModelUnavailable
		| ErrorKind::CapabilityUnknown => Status::failed_precondition(message),
		_ => Status::internal(message),
	}
}

fn validate_revision(context: &pb::ContextRef, actual: u64) -> Result<(), Status> {
	let expected = context
		.expected
		.as_ref()
		.ok_or_else(|| Status::invalid_argument("ContextRef.expected is required"))?;
	if expected.head != actual
		|| expected.token.as_ref() != revision_token(&context.context_id, actual).as_slice()
	{
		return Err(Status::aborted(format!(
			"context revision conflict: expected {}, actual {actual}",
			expected.head
		)));
	}
	Ok(())
}

fn revision(context: &str, head: u64) -> thread_pb::Revision {
	thread_pb::Revision { head, token: revision_token(context, head).into() }
}

fn revision_token(context: &str, head: u64) -> Vec<u8> {
	let mut token = Vec::with_capacity(context.len() + 8);
	token.extend_from_slice(context.as_bytes());
	token.extend_from_slice(&head.to_be_bytes());
	token
}

fn thread_messages(thread: &thread_pb::Thread) -> Result<Vec<Message>, Status> {
	items_messages(&thread.items)
}

fn items_messages(items: &[thread_pb::Item]) -> Result<Vec<Message>, Status> {
	items
		.iter()
		.map(|item| match item.kind.as_ref() {
			Some(thread_pb::item::Kind::Message(message)) => message_from_proto(message),
			Some(thread_pb::item::Kind::ToolCall(call)) => Ok(Message {
				role:    Role::Assistant,
				content: Arc::from([ContentPart::ToolCall {
					call:      ToolCallId::from(call.id.as_str()),
					name:      call.name.as_str().into(),
					arguments: opaque_json(&call.args_json, "ToolCall.args_json")?,
					proof:     None,
				}]),
				name:    None,
			}),
			Some(thread_pb::item::Kind::ToolResult(result)) => {
				let content = result
					.parts
					.iter()
					.map(tool_result_part)
					.collect::<Result<Vec<_>, _>>()?;
				Ok(Message {
					role:    Role::Tool,
					content: Arc::from([ContentPart::ToolResult {
						call:     ToolCallId::from(result.call_id.as_str()),
						name:     (!result.name.is_empty()).then(|| result.name.as_str().into()),
						content:  content.into(),
						is_error: result.is_error,
					}]),
					name:    None,
				})
			},
			None => Err(Status::invalid_argument("thread item kind is required")),
		})
		.collect()
}

fn message_from_proto(message: &thread_pb::Message) -> Result<Message, Status> {
	let role = match thread_pb::Role::try_from(message.role).unwrap_or(thread_pb::Role::Unspecified)
	{
		thread_pb::Role::System => Role::System,
		thread_pb::Role::User => Role::User,
		thread_pb::Role::Assistant => Role::Assistant,
		thread_pb::Role::Unspecified => {
			return Err(Status::invalid_argument("message role is required"));
		},
	};
	let content = message
		.parts
		.iter()
		.map(content_part)
		.collect::<Result<Vec<_>, _>>()?;
	Ok(Message { role, content: content.into(), name: None })
}

fn content_part(part: &thread_pb::Part) -> Result<ContentPart, Status> {
	match part.kind.as_ref() {
		Some(thread_pb::part::Kind::Text(text)) => {
			Ok(ContentPart::Text { text: text.as_str().into(), proof: None })
		},
		Some(thread_pb::part::Kind::Thinking(thinking)) if thinking.signature.is_empty() => {
			Ok(ContentPart::Reasoning { text: thinking.text.as_str().into(), proof: None })
		},
		Some(thread_pb::part::Kind::Thinking(_)) => Err(Status::invalid_argument(
			"unscoped reasoning signatures cannot enter canonical inference",
		)),
		Some(thread_pb::part::Kind::Blob(blob)) => Ok(ContentPart::Image(media_input(blob)?)),
		Some(thread_pb::part::Kind::Fallback(_)) | Some(thread_pb::part::Kind::ServerTool(_)) => {
			Err(Status::invalid_argument(
				"legacy fallback/server-tool parts require an explicit canonical projection",
			))
		},
		None => Err(Status::invalid_argument("message part kind is required")),
	}
}

fn tool_result_part(
	part: &thread_pb::Part,
) -> Result<omp_llm_inference::call::ToolResultContent, Status> {
	match part.kind.as_ref() {
		Some(thread_pb::part::Kind::Text(text)) => {
			Ok(omp_llm_inference::call::ToolResultContent::Text(text.as_str().into()))
		},
		Some(thread_pb::part::Kind::Blob(blob)) => {
			Ok(omp_llm_inference::call::ToolResultContent::Document(media_input(blob)?))
		},
		_ => Err(Status::invalid_argument(
			"tool result contains a part that has no canonical projection",
		)),
	}
}

fn media_input(blob: &thread_pb::Blob) -> Result<MediaInput, Status> {
	if blob.mime.is_empty() {
		return Err(Status::invalid_argument("Blob.mime is required"));
	}
	if !blob.inline.is_empty() {
		return Ok(MediaInput::Bytes {
			media_type: blob.mime.as_str().into(),
			data:       Bytes::copy_from_slice(&blob.inline),
		});
	}
	if blob.hash.is_empty() {
		return Err(Status::invalid_argument("Blob requires inline bytes or a content hash"));
	}
	let id = blob
		.hash
		.iter()
		.map(|byte| format!("{byte:02x}"))
		.collect::<String>();
	Ok(MediaInput::Stored(omp_llm_inference::answer::ArtifactRef {
		store:    Str::from("omp-rpc-blobs"),
		id:       id.as_str().into(),
		revision: id.as_str().into(),
	}))
}

fn opaque_json(bytes: &[u8], field: &'static str) -> Result<OpaqueJson, Status> {
	serde_json::from_slice(bytes)
		.map(OpaqueJson::new)
		.map_err(|error| Status::invalid_argument(format!("{field} is invalid JSON: {error}")))
}

fn tool_definition(tool: &pb::ToolDef) -> Result<ToolDefinition, Status> {
	if tool.name.is_empty() {
		return Err(Status::invalid_argument("ToolDef.name is required"));
	}
	Ok(ToolDefinition {
		name:        tool.name.as_str().into(),
		description: (!tool.description.is_empty()).then(|| tool.description.as_str().into()),
		parameters:  opaque_json(&tool.schema_json, "ToolDef.schema_json")?,
		strict:      tool.strict.unwrap_or(false),
	})
}

fn chat_request(
	messages: Vec<Message>,
	params: &pb::ChatParams,
) -> Result<omp_llm_inference::call::ChatRequest, Status> {
	let tools = params
		.tools
		.iter()
		.map(tool_definition)
		.collect::<Result<Vec<_>, _>>()?;
	let tool_choice = params
		.tool_choice
		.as_ref()
		.map_or(Ok(Setting::Unset), |choice| {
			let choice = match pb::tool_choice::Mode::try_from(choice.mode)
				.unwrap_or(pb::tool_choice::Mode::Unspecified)
			{
				pb::tool_choice::Mode::Unspecified | pb::tool_choice::Mode::Auto => ToolChoice::Auto,
				pb::tool_choice::Mode::None => ToolChoice::Disabled,
				pb::tool_choice::Mode::Required => ToolChoice::Required,
				pb::tool_choice::Mode::Named if !choice.name.is_empty() => {
					ToolChoice::Named(choice.name.as_str().into())
				},
				pb::tool_choice::Mode::Named => {
					return Err(Status::invalid_argument("named tool choice requires a name"));
				},
			};
			Ok(Setting::Require(choice))
		})?;
	let sampling = params
		.sampling
		.as_ref()
		.map_or_else(Sampling::default, |sampling| Sampling {
			temperature:       sampling.temperature.map(|value| value as f32),
			top_p:             sampling.top_p.map(|value| value as f32),
			top_k:             sampling.top_k,
			seed:              None,
			stop:              sampling
				.stop
				.iter()
				.map(|value| Str::from(value.as_str()))
				.collect::<Vec<_>>()
				.into(),
			presence_penalty:  sampling.presence_penalty.map(|value| value as f32),
			frequency_penalty: sampling.frequency_penalty.map(|value| value as f32),
		});
	Ok(omp_llm_inference::call::ChatRequest {
		messages: messages.into(),
		tools: tools.into(),
		hosted_tools: Arc::from([]),
		tool_choice,
		output: Setting::Unset,
		reasoning: Setting::Unset,
		verbosity: Setting::Unset,
		cache_retention: Setting::Unset,
		service_tier: Setting::Unset,
		sampling,
		max_output_tokens: params
			.sampling
			.as_ref()
			.and_then(|sampling| sampling.max_output_tokens),
		top_logprobs: None,
		safety: Arc::from([]),
		negotiation: NegotiationPolicy::default(),
	})
}

fn proto_usage(usage: Usage) -> pb::Usage {
	pb::Usage {
		input_tokens:       usage.input_tokens,
		output_tokens:      usage.output_tokens,
		cache_read_tokens:  usage.cache_read_tokens,
		cache_write_tokens: usage.cache_write_tokens,
		accuracy:           match usage.source {
			UsageSource::Provider | UsageSource::Measured => pb::usage::Accuracy::Exact,
			UsageSource::Estimated => pb::usage::Accuracy::Estimated,
			UsageSource::Mixed => pb::usage::Accuracy::Mixed,
			UsageSource::Unknown => pb::usage::Accuracy::Unspecified,
		} as i32,
		detail:             None,
		total_tokens:       Some(usage.total_tokens()),
		context_tokens:     None,
		orchestration:      None,
		premium_requests:   None,
		reasoning_tokens:   Some(usage.reasoning_tokens),
		cache_ttl:          None,
		server_tools:       (usage.search_calls != 0).then(|| pb::ServerToolUsage {
			web_search_requests: Some(u64::from(usage.search_calls)),
			web_fetch_requests:  None,
		}),
	}
}

fn tokenizer_provenance(
	provenance: omp_llm_inference::answer::TokenizerProvenance,
) -> pb::TokenizerProvenance {
	pb::TokenizerProvenance {
		tokenizer: provenance.tokenizer.as_str().to_owned(),
		revision:  provenance.revision.as_str().to_owned(),
		exact:     provenance.exact,
	}
}

fn proto_cost(cost: Cost) -> pb::Cost {
	pb::Cost {
		nanos_usd:             cost
			.micro_usd
			.max(0)
			.saturating_mul(1_000)
			.try_into()
			.unwrap_or(u64::MAX),
		estimated:             false,
		input_nanos_usd:       None,
		output_nanos_usd:      None,
		cache_read_nanos_usd:  None,
		cache_write_nanos_usd: None,
	}
}

fn turn_events(
	mut events: omp_llm_inference::answer::ChatStream,
	contexts: Arc<Mutex<BTreeMap<String, RpcContext>>>,
	context_id: Option<String>,
	_base_revision: u64,
	input_messages: Vec<Message>,
) -> impl Stream<Item = Result<pb::TurnEvent, Status>> + Send + 'static {
	async_stream::try_stream! {
		yield pb::TurnEvent {
			event: Some(pb::turn_event::Event::Accepted(pb::Accepted { replay: false })),
		};
		let mut assistant_text = String::new();
		let mut assistant_parts = Vec::<ContentPart>::new();
		let mut output = Vec::<thread_pb::Item>::new();
		while let Some(event) = events.next().await {
			match event.map_err(inference_status)? {
				ChatEvent::Started(_) => {},
				ChatEvent::BlockStarted { index, kind } => {
					let kind = match kind {
						BlockKind::Text => pb::part_start::Kind::Text,
						BlockKind::Thinking => pb::part_start::Kind::Thinking,
						BlockKind::ToolCall => continue,
						BlockKind::Artifact => {
							Err(Status::failed_precondition(
								"chat artifacts must be staged before RPC projection",
							))?
						},
					};
					yield pb::TurnEvent {
						event: Some(pb::turn_event::Event::PartStart(pb::PartStart {
							index,
							kind: kind as i32,
							tool_call_id: String::new(),
							tool_name: String::new(),
						})),
					};
				},
				ChatEvent::TextDelta { index, text } => {
					assistant_text.push_str(text.as_str());
					yield pb::TurnEvent {
						event: Some(pb::turn_event::Event::PartDelta(pb::PartDelta {
							index,
							chunk: Bytes::copy_from_slice(text.as_bytes()),
						})),
					};
				},
				ChatEvent::ThinkingDelta { index, text } => {
					yield pb::TurnEvent {
						event: Some(pb::turn_event::Event::PartDelta(pb::PartDelta {
							index,
							chunk: Bytes::copy_from_slice(text.as_bytes()),
						})),
					};
				},
				ChatEvent::ToolCallStarted { index, id, name } => {
					yield pb::TurnEvent {
						event: Some(pb::turn_event::Event::PartStart(pb::PartStart {
							index,
							kind: pb::part_start::Kind::ToolCall as i32,
							tool_call_id: id.as_str().to_owned(),
							tool_name: name.as_str().to_owned(),
						})),
					};
				},
				ChatEvent::ToolArgumentsDelta { index, bytes } => {
					yield pb::TurnEvent {
						event: Some(pb::turn_event::Event::PartDelta(pb::PartDelta {
							index,
							chunk: bytes,
						})),
					};
				},
				ChatEvent::ToolCallReady { index, call } => {
					let arguments = serde_json::to_vec(call.arguments.as_value())
						.map_err(|error| Status::internal(format!("tool arguments serialization failed: {error}")))?;
					assistant_parts.push(ContentPart::ToolCall {
						call: call.id.clone(),
						name: call.name.clone(),
						arguments: call.arguments.clone(),
						proof: None,
					});
					output.push(thread_pb::Item {
						seq: 0,
						created_at_ms: 0,
						kind: Some(thread_pb::item::Kind::ToolCall(thread_pb::ToolCall {
							id: call.id.as_str().to_owned(),
							name: call.name.as_str().to_owned(),
							args_json: arguments.into(),
							thought_signature: Bytes::new(),
							intent: None,
							raw: None,
							custom_wire_name: None,
							provider_metadata: None,
						})),
						props: None,
					});
					yield pb::TurnEvent {
						event: Some(pb::turn_event::Event::PartEnd(pb::PartEnd {
							index,
							signature: Bytes::new(),
						})),
					};
				},
				ChatEvent::Artifact { .. } => {
					Err(Status::failed_precondition(
						"chat artifacts must be staged before RPC projection",
					))?
				},
				ChatEvent::Usage(_) => {},
				ChatEvent::Completed(completion) => {
					if !assistant_text.is_empty() {
						assistant_parts.insert(0, ContentPart::Text {
							text: assistant_text.as_str().into(),
							proof: None,
						});
						output.insert(0, thread_pb::Item {
							seq: 0,
							created_at_ms: 0,
							kind: Some(thread_pb::item::Kind::Message(thread_pb::Message {
								role: thread_pb::Role::Assistant as i32,
								parts: vec![thread_pb::Part {
									kind: Some(thread_pb::part::Kind::Text(assistant_text.clone())),
								}],
							})),
							props: None,
						});
					}
					let next_revision = if let Some(context_id) = context_id.as_ref() {
						let mut contexts = contexts.lock();
						let held = contexts.entry(context_id.clone()).or_default();
						held.messages = input_messages.clone();
						held.messages.push(Message {
							role: Role::Assistant,
							content: assistant_parts.clone().into(),
							name: None,
						});
						held.revision = held.messages.len() as u64;
						Some(revision(context_id, held.revision))
					} else {
						None
					};
					yield pb::TurnEvent {
						event: Some(pb::turn_event::Event::Outcome(pb::Outcome {
							output: std::mem::take(&mut output),
							stop: match completion.reason {
								FinishReason::Stop | FinishReason::Other(_) => 1,
								FinishReason::Length => 3,
								FinishReason::ToolCalls => 2,
								FinishReason::ContentFilter => 4,
								FinishReason::Cancelled => 0,
							} as i32,
							usage: Some(proto_usage(completion.usage)),
							cost: Some(proto_cost(completion.receipt.cost)),
							unsupported: Vec::new(),
							revision: next_revision,
							provider: completion.receipt.plan.provider.as_ref().map_or_else(String::new, |value| value.as_str().to_owned()),
							model: completion.receipt.plan.model.as_ref().map_or_else(String::new, |value| value.as_str().to_owned()),
							diagnostics: Vec::new(),
							upstream_provider: None,
							duration_ms: Some(completion.receipt.timings.total.as_millis().try_into().unwrap_or(u64::MAX)),
							ttft_ms: completion.receipt.timings.first_frame.map(|value| value.as_millis().try_into().unwrap_or(u64::MAX)),
							context_snapshot: None,
							props: None,
						})),
					};
				},
			}
		}
	}
}

fn image_events(
	mut events: omp_llm_inference::answer::GenerationStream<ImageArtifact>,
) -> impl Stream<Item = Result<pb::ImageEvent, Status>> + Send + 'static {
	async_stream::try_stream! {
		let mut images = Vec::new();
		let mut revised_prompt = None::<String>;
		let mut preview_index = 0_u32;
		while let Some(event) = events.next().await {
			match event.map_err(inference_status)? {
				GenerationEvent::Queued { .. } | GenerationEvent::Progress { .. } => {},
				GenerationEvent::Preview(image) => {
					let blob = artifact_blob(image.artifact)?;
					yield pb::ImageEvent {
						event: Some(pb::image_event::Event::Partial(pb::image_event::Partial {
							index: preview_index,
							preview: Some(blob),
						})),
					};
					preview_index = preview_index.saturating_add(1);
				},
				GenerationEvent::Artifact(image) => {
					if revised_prompt.is_none() {
						revised_prompt = image.revised_prompt.map(|value| value.as_str().to_owned());
					}
					images.push(artifact_blob(image.artifact)?);
				},
				GenerationEvent::Completed(summary) => {
					yield pb::ImageEvent {
						event: Some(pb::image_event::Event::Done(pb::image_event::Done {
							images,
							revised_prompt: revised_prompt.unwrap_or_default(),
							text: String::new(),
							usage: Some(proto_usage(summary.usage)),
							cost: Some(proto_cost(summary.cost)),
							unsupported: Vec::new(),
							props: None,
						})),
					};
					break;
				},
			}
		}
	}
}

fn artifact_blob(artifact: Artifact) -> Result<thread_pb::Blob, Status> {
	let (hash, inline) = match artifact.body {
		ArtifactBody::Bytes(bytes) => (
			artifact
				.digest
				.map_or_else(Bytes::new, |digest| digest.value),
			bytes,
		),
		ArtifactBody::Stored(reference) => {
			(Bytes::copy_from_slice(reference.revision.as_bytes()), Bytes::new())
		},
		ArtifactBody::Stream(_) => {
			return Err(Status::failed_precondition(
				"streamed artifacts must be persisted before RPC projection",
			));
		},
	};
	Ok(thread_pb::Blob {
		hash,
		mime: artifact.media_type.as_str().to_owned(),
		size: artifact.size.unwrap_or(inline.len() as u64),
		inline,
		detail: thread_pb::blob::Detail::Original as i32,
	})
}

fn speak_event(chunk: AudioChunk) -> pb::SpeakEvent {
	if chunk.final_chunk {
		pb::SpeakEvent {
			event: Some(pb::speak_event::Event::Done(pb::speak_event::Done {
				audio:       Some(thread_pb::Blob {
					hash:   Bytes::new(),
					mime:   String::new(),
					size:   chunk.bytes.len() as u64,
					inline: chunk.bytes,
					detail: thread_pb::blob::Detail::Original as i32,
				}),
				duration_ms: chunk.end_ms.unwrap_or_default(),
				usage:       None,
				cost:        None,
				unsupported: Vec::new(),
				props:       None,
			})),
		}
	} else {
		pb::SpeakEvent {
			event: Some(pb::speak_event::Event::Chunk(pb::speak_event::Chunk {
				audio:            chunk.bytes,
				transcript_delta: String::new(),
			})),
		}
	}
}

fn search_response(answer: SearchResults) -> pb::SearchResponse {
	pb::SearchResponse {
		engine:         String::new(),
		answer:         answer
			.answer
			.map_or_else(String::new, |answer| answer.as_str().to_owned()),
		sources:        answer
			.results
			.into_iter()
			.map(|result| pb::search_response::Source {
				url:          result.url.as_str().to_owned(),
				title:        result.title.as_str().to_owned(),
				snippet:      result
					.snippet
					.map_or_else(String::new, |snippet| snippet.as_str().to_owned()),
				published_at: result
					.published_at
					.and_then(|time| time.duration_since(UNIX_EPOCH).ok())
					.map_or_else(String::new, |duration| duration.as_secs().to_string()),
				author:       String::new(),
				score:        result.score.map(f64::from),
			})
			.collect(),
		citations:      Vec::new(),
		search_queries: Vec::new(),

		related:     Vec::new(),
		warnings:    Vec::new(),
		usage:       Some(proto_usage(answer.usage)),
		cost:        None,
		unsupported: Vec::new(),
		props:       None,
	}
}
fn usage_response(report: UsageReport) -> pb::UsageResponse {
	pb::UsageResponse {
		provider:  report.provider.as_str().to_owned(),
		account:   report.account.as_str().to_owned(),
		principal: report
			.principal
			.map_or_else(String::new, |value| value.as_str().to_owned()),
		windows:   report
			.windows
			.into_iter()
			.map(|window| pb::usage_response::Window {
				kind:           match window.kind {
					UsageWindowKind::RateLimit => pb::usage_response::window::Kind::RateLimit,
					UsageWindowKind::Quota => pb::usage_response::window::Kind::Quota,
					UsageWindowKind::Billing => pb::usage_response::window::Kind::Billing,
					UsageWindowKind::Balance => pb::usage_response::window::Kind::Balance,
				} as i32,
				dimension:      window.dimension.as_str().to_owned(),
				consumed:       window.consumed,
				remaining:      window.remaining,
				limit:          window.limit,
				resets_at_ms:   window.resets_at.map(system_time_ms),
				accuracy:       match window.source {
					UsageSource::Provider | UsageSource::Measured => pb::usage::Accuracy::Exact,
					UsageSource::Estimated => pb::usage::Accuracy::Estimated,
					UsageSource::Mixed => pb::usage::Accuracy::Mixed,
					UsageSource::Unknown => pb::usage::Accuracy::Unspecified,
				} as i32,
				observed_at_ms: system_time_ms(window.observed_at),
			})
			.collect(),
	}
}
fn realtime_audio_format(encoding: i32) -> Setting<AudioFormat> {
	match pb::AudioEncoding::try_from(encoding).unwrap_or(pb::AudioEncoding::Unspecified) {
		pb::AudioEncoding::Mp3 => Setting::Prefer(AudioFormat::Mp3),
		pb::AudioEncoding::Pcm16 => Setting::Prefer(AudioFormat::Pcm16),
		pb::AudioEncoding::Wav => Setting::Prefer(AudioFormat::Wav),
		pb::AudioEncoding::Opus => Setting::Prefer(AudioFormat::Opus),
		pb::AudioEncoding::Aac => Setting::Prefer(AudioFormat::Aac),
		pb::AudioEncoding::Flac => Setting::Prefer(AudioFormat::Flac),
		pb::AudioEncoding::Unspecified => Setting::Unset,
	}
}

fn realtime_input(frame: pb::RealtimeFrame) -> Result<RealtimeInput, Status> {
	match frame.frame {
		Some(pb::realtime_frame::Frame::Audio(bytes)) => Ok(RealtimeInput::Audio(bytes)),
		Some(pb::realtime_frame::Frame::Text(text)) => Ok(RealtimeInput::Text(text.into())),
		Some(pb::realtime_frame::Frame::ToolResult(result)) => Ok(RealtimeInput::ToolResult {
			call:     ToolCallId::from(result.call_id.as_str()),
			name:     (!result.name.is_empty()).then(|| result.name.as_str().into()),
			content:  result
				.parts
				.iter()
				.map(tool_result_part)
				.collect::<Result<Vec<_>, _>>()?
				.into(),
			is_error: result.is_error,
		}),
		Some(pb::realtime_frame::Frame::Commit(_)) => Ok(RealtimeInput::Commit),
		Some(pb::realtime_frame::Frame::CancelResponse(_)) => Ok(RealtimeInput::CancelResponse),
		Some(pb::realtime_frame::Frame::Close(_)) => Ok(RealtimeInput::Close),
		Some(pb::realtime_frame::Frame::Open(_)) => {
			Err(Status::invalid_argument("Realtime open may appear only once"))
		},
		None => Err(Status::invalid_argument("Realtime frame variant is required")),
	}
}

fn realtime_event(event: CanonicalRealtimeEvent) -> Result<pb::RealtimeEvent, Status> {
	let event = match event {
		CanonicalRealtimeEvent::Ready => pb::realtime_event::Event::Ready(pb::RealtimeReady {}),
		CanonicalRealtimeEvent::Audio(chunk) => pb::realtime_event::Event::Audio(pb::RealtimeAudio {
			audio:    chunk.bytes,
			start_ms: chunk.start_ms,
			end_ms:   chunk.end_ms,
			r#final:  chunk.final_chunk,
		}),
		CanonicalRealtimeEvent::InputCommitted => {
			pb::realtime_event::Event::InputCommitted(pb::RealtimeInputCommitted {})
		},
		CanonicalRealtimeEvent::Closed => pb::realtime_event::Event::Closed(pb::RealtimeClosed {}),
		CanonicalRealtimeEvent::Chat(chat) => {
			pb::realtime_event::Event::Chat(realtime_chat_event(chat)?)
		},
	};
	Ok(pb::RealtimeEvent { event: Some(event) })
}

fn realtime_chat_event(event: ChatEvent) -> Result<pb::TurnEvent, Status> {
	let event = match event {
		ChatEvent::Started(_) => pb::turn_event::Event::Accepted(pb::Accepted { replay: false }),
		ChatEvent::BlockStarted { index, kind } => {
			let kind = match kind {
				BlockKind::Text => pb::part_start::Kind::Text,
				BlockKind::Thinking => pb::part_start::Kind::Thinking,
				BlockKind::ToolCall => pb::part_start::Kind::ToolCall,
				BlockKind::Artifact => {
					return Err(Status::failed_precondition(
						"realtime chat artifacts require an explicit RPC artifact projection",
					));
				},
			};
			pb::turn_event::Event::PartStart(pb::PartStart {
				index,
				kind: kind as i32,
				tool_call_id: String::new(),
				tool_name: String::new(),
			})
		},
		ChatEvent::TextDelta { index, text } | ChatEvent::ThinkingDelta { index, text } => {
			pb::turn_event::Event::PartDelta(pb::PartDelta {
				index,
				chunk: Bytes::copy_from_slice(text.as_bytes()),
			})
		},
		ChatEvent::ToolCallStarted { index, id, name } => {
			pb::turn_event::Event::PartStart(pb::PartStart {
				index,
				kind: pb::part_start::Kind::ToolCall as i32,
				tool_call_id: id.as_str().to_owned(),
				tool_name: name.as_str().to_owned(),
			})
		},
		ChatEvent::ToolArgumentsDelta { index, bytes } => {
			pb::turn_event::Event::PartDelta(pb::PartDelta { index, chunk: bytes })
		},
		ChatEvent::ToolCallReady { index, .. } => {
			pb::turn_event::Event::PartEnd(pb::PartEnd { index, signature: Bytes::new() })
		},
		ChatEvent::Artifact { .. } => {
			return Err(Status::failed_precondition(
				"realtime chat artifacts require an explicit RPC artifact projection",
			));
		},
		ChatEvent::Usage(update) => pb::turn_event::Event::Attempt(pb::Attempt {
			number: 0,
			reason: format!("usage:{}:{}", update.usage.input_tokens, update.usage.output_tokens),
		}),
		ChatEvent::Completed(completion) => pb::turn_event::Event::Outcome(pb::Outcome {
			output:            Vec::new(),
			stop:              match completion.reason {
				FinishReason::Stop | FinishReason::Other(_) => 1,
				FinishReason::Length => 3,
				FinishReason::ToolCalls => 2,
				FinishReason::ContentFilter => 4,
				FinishReason::Cancelled => 0,
			},
			usage:             Some(proto_usage(completion.usage)),
			cost:              Some(proto_cost(completion.receipt.cost)),
			unsupported:       Vec::new(),
			revision:          None,
			provider:          completion
				.receipt
				.plan
				.provider
				.as_ref()
				.map_or_else(String::new, |value| value.as_str().to_owned()),
			model:             completion
				.receipt
				.plan
				.model
				.as_ref()
				.map_or_else(String::new, |value| value.as_str().to_owned()),
			diagnostics:       Vec::new(),
			upstream_provider: None,
			duration_ms:       Some(
				completion
					.receipt
					.timings
					.total
					.as_millis()
					.try_into()
					.unwrap_or(u64::MAX),
			),
			ttft_ms:           completion
				.receipt
				.timings
				.first_frame
				.map(|value| value.as_millis().try_into().unwrap_or(u64::MAX)),
			context_snapshot:  None,
			props:             None,
		}),
	};
	Ok(pb::TurnEvent { event: Some(event) })
}

fn native_response_stream(
	response: NativeResponse,
) -> impl Stream<Item = Result<pb::NativeChunk, Status>> + Send + 'static {
	async_stream::try_stream! {
		let status = u32::from(response.status);
		let media_type =
			response.media_type.map_or_else(String::new, |value| value.as_str().to_owned());
		let provider_request_id = response
			.provider_request_id
			.map_or_else(String::new, |value| value.as_str().to_owned());
		match response.body {
			NativeResponseBody::Json(value) => {
				yield pb::NativeChunk {
					status,
					media_type,
					provider_request_id,
					data: value.into_bytes(),
					r#final: true,
				};
			},
			NativeResponseBody::Bytes(bytes) => {
				yield pb::NativeChunk {
					status,
					media_type,
					provider_request_id,
					data: bytes,
					r#final: true,
				};
			},
			NativeResponseBody::Stream(mut stream) => {
				yield pb::NativeChunk {
					status,
					media_type,
					provider_request_id,
					data: Bytes::new(),
					r#final: false,
				};
				while let Some(chunk) = stream.next().await {
					yield pb::NativeChunk {
						status: 0,
						media_type: String::new(),
						provider_request_id: String::new(),
						data: chunk.map_err(inference_status)?,
						r#final: false,
					};
				}
				yield pb::NativeChunk {
					status: 0,
					media_type: String::new(),
					provider_request_id: String::new(),
					data: Bytes::new(),
					r#final: true,
				};
			},
		}
	}
}

fn video_dimensions(resolution: i32, aspect_ratio: i32) -> Setting<Dimensions> {
	let height = match resolution {
		1 => 480,
		2 => 720,
		3 => 1_080,
		4 => 2_160,
		_ => return Setting::Unset,
	};
	let width = match aspect_ratio {
		1 => height,
		3 => height * 9 / 16,
		4 => height * 4 / 3,
		5 => height * 3 / 4,
		6 => height * 3 / 2,
		7 => height * 2 / 3,
		8 => height * 21 / 9,
		2 => height * 16 / 9,
		_ => return Setting::Unset,
	};
	Setting::Prefer(Dimensions { width, height })
}

async fn run_generation(
	mut session: omp_llm_inference::answer::GenerationSession<VideoArtifact>,
	status: Arc<Mutex<pb::GenerationStatus>>,
	updates: tokio::sync::broadcast::Sender<pb::GenerationStatus>,
	cancel: flume::Receiver<
		tokio::sync::oneshot::Sender<Result<JobCancellationReceipt, JobCancelError>>,
	>,
) {
	let mut cancel_open = true;
	loop {
		tokio::select! {
			command = cancel.recv_async(), if cancel_open => {
				let Ok(command) = command else {
					cancel_open = false;
					continue;
				};
				let result = session.cancel().await;
				if result.as_ref().is_ok_and(|receipt| receipt.acknowledged) {
					publish_generation(&status, &updates, |status| {
						status.state = pb::generation_status::State::Cancelled as i32;
					});
				}
				let terminal = result.as_ref().is_ok_and(|receipt| receipt.acknowledged);
				let _ = command.send(result);
				if terminal { break; }
			},
			event = session.next() => {
				let Some(event) = event else {
					if !generation_terminal(status.lock().state) {
						publish_generation(&status, &updates, |status| {
							status.state = pb::generation_status::State::Failed as i32;
							status.detail = "generation stream ended before a terminal event".to_owned();
						});
					}
					break;
				};
				match event {
					Ok(GenerationEvent::Queued { .. }) => publish_generation(&status, &updates, |status| {
						status.state = pb::generation_status::State::Queued as i32;
					}),
					Ok(GenerationEvent::Progress { completed, total }) => publish_generation(&status, &updates, |status| {
						status.state = pb::generation_status::State::Running as i32;
						status.progress_percent = total
							.filter(|total| *total != 0)
							.map_or(0.0, |total| completed as f64 * 100.0 / total as f64);
					}),
					Ok(GenerationEvent::Preview(_)) => {},
					Ok(GenerationEvent::Artifact(video)) => match artifact_blob(video.artifact) {
						Ok(blob) => publish_generation(&status, &updates, |status| {
							status.artifacts.push(pb::generation_status::Artifact {
								blob: Some(blob),
								variant: "video".to_owned(),
								url: String::new(),
								url_expires_at_ms: 0,
							});
						}),
						Err(error) => {
							publish_generation(&status, &updates, |status| {
								status.state = pb::generation_status::State::Failed as i32;
								status.detail = error.message().to_owned();
							});
							break;
						},
					},
					Ok(GenerationEvent::Completed(summary)) => {
						publish_generation(&status, &updates, |status| {
							status.state = pb::generation_status::State::Completed as i32;
							status.progress_percent = 100.0;
							status.usage = Some(proto_usage(summary.usage));
							status.cost = Some(proto_cost(summary.cost));
						});
						break;
					},
					Err(error) => {
						publish_generation(&status, &updates, |status| {
							status.state = pb::generation_status::State::Failed as i32;
							status.detail = format!("{:?}", error.kind);
						});
						break;
					},
				}
			},
		}
	}
}

fn publish_generation(
	status: &Mutex<pb::GenerationStatus>,
	updates: &tokio::sync::broadcast::Sender<pb::GenerationStatus>,
	update: impl FnOnce(&mut pb::GenerationStatus),
) {
	let snapshot = {
		let mut status = status.lock();
		update(&mut status);
		status.updated_at_ms = system_time_ms(SystemTime::now());
		status.clone()
	};
	let _ = updates.send(snapshot);
}

fn generation_terminal(state: i32) -> bool {
	matches!(
		pb::generation_status::State::try_from(state),
		Ok(pb::generation_status::State::Completed
			| pb::generation_status::State::Failed
			| pb::generation_status::State::Cancelled)
	)
}

fn system_time_ms(time: SystemTime) -> u64 {
	time
		.duration_since(UNIX_EPOCH)
		.unwrap_or_default()
		.as_millis()
		.try_into()
		.unwrap_or(u64::MAX)
}
