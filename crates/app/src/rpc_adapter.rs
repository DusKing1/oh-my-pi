//! Tonic transport projection over the typed inference registry.

use std::{
	collections::BTreeMap,
	pin::Pin,
	sync::Arc,
	time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use bytes::Bytes;
use futures::{Stream, StreamExt as _, stream};
use omp_agent::project_thread_history;
use omp_core::Str;
use omp_llm_catalog::{
	GrammarBits, ModelAvailability, ModelKey, ModelSpec, OperationKind, ProviderDef, ProviderId,
};
use omp_llm_inference::{
	Client, Registry,
	answer::{
		Artifact, ArtifactBody, AudioChunk, ChatControl, ChatControlError, GenerationEvent,
		ImageArtifact, NativeResponse, NativeResponseBody, RealtimeEvent as CanonicalRealtimeEvent,
		RealtimeInput, SearchResults, TranscriptEvent, UsageReport, UsageWindowKind, VideoArtifact,
	},
	call::{
		AudioFormat, Background, CallMeta, ContentPart, ContextStrategy, CountAccuracy,
		CountTokensRequest, DetokenizeRequest, Dimensions, EmbedRequest, EmbeddingInput, ImageFormat,
		ImageQuality, ImageRequest, MediaInput, Message, NativeMethod, NativePath, NativePayload,
		NativeRequest, NativeResponseFraming, NegotiationPolicy, OpaqueJson, RawJson,
		RealtimeModality, RealtimeRequest, Role, Sampling, SearchRecency, SearchRequest,
		SessionRequest, Setting, SpeechRequest, Target, TimestampGranularity, TokenizeRequest,
		ToolChoice, ToolDefinition, ToolInputConstraint, TranscriptionRequest, TruncationPolicy,
		UsageRequest, UsageScope, VideoRequest,
	},
	error::{Error, ErrorKind},
	event::{
		BlockKind, ChatEvent, Completion, FinishReason, InvokeComplete, InvokeInput,
		WorkflowActionResponse, WorkflowResponse, WorkflowResponseKind,
	},
	id::{
		AccountId, ConversationId, RequestId, Revision as ProviderRevision, ToolCallId,
		TurnId as ProviderTurnId,
	},
	operation::job::{JobCancelError, JobCancellationReceipt},
	receipt::{Cost, ExecutionBudget, Usage, UsageSource},
	router::Router,
	session::{ConversationSessionPlanner, TurnReplay},
};
use omp_proto::{inference::v1 as pb, prost::Message as _, thread::v1 as thread_pb};
use omp_tool::{LoweringCaps, PromptCaps, Registry as ToolRegistry, TOOL_REV_PROP};
use parking_lot::Mutex;
use tonic::{Request, Response, Status};

// env/v1/turn carries no per-model projection caps; this bounded text-only
// fallback is valid for every transport and never silently exposes media.
const RPC_HISTORY_PROMPT_CAPS: PromptCaps =
	PromptCaps { maximum_parts: 1, maximum_text_bytes: 64 * 1024, media: false };

/// Stream returned by RPC methods whose typed operation produces events.
pub type RpcStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;

/// Projects the canonical catalog and typed operation service onto the retained
/// OMP RPC schema.
#[derive(Clone)]
pub struct InferenceRpc {
	registry:            Registry,
	tool_registry:       Arc<ToolRegistry>,
	sessions:            ConversationSessionPlanner,
	epoch:               Arc<[u8]>,
	provider_sessions:   bool,
	test_live_responses: Option<flume::Sender<WorkflowResponse>>,
	contexts:            Arc<Mutex<BTreeMap<String, RpcContext>>>,
	generations:         Arc<Mutex<BTreeMap<String, RpcGeneration>>>,
}

#[derive(Clone, Default)]
struct RpcContext {
	revision:              u64,
	messages:              Vec<Message>,
	provider_conversation: Option<ConversationId>,
	provider_revision:     Option<ProviderRevision>,
	provider_heads:        BTreeMap<u64, ProviderRevision>,
}

struct ResolvedTurn {
	request_messages:      Vec<Message>,
	committed_messages:    Vec<Message>,
	context_id:            Option<String>,
	provider_session:      Option<SessionRequest>,
	provider_conversation: Option<ConversationId>,
	provider_heads:        BTreeMap<u64, ProviderRevision>,
}

#[derive(Default)]
struct TurnProjection {
	assistant_text: String,
	output:         Vec<thread_pb::Item>,
}

#[derive(Clone)]
struct RpcGeneration {
	status:  Arc<Mutex<pb::GenerationStatus>>,
	updates: tokio::sync::broadcast::Sender<pb::GenerationStatus>,
	cancel:
		flume::Sender<tokio::sync::oneshot::Sender<Result<JobCancellationReceipt, JobCancelError>>>,
}

impl InferenceRpc {
	/// Creates an RPC projection over one immutable registry generation and the
	/// same provider-conversation planner installed in its route stack.
	#[must_use]
	pub fn new(
		registry: Registry,
		sessions: ConversationSessionPlanner,
		tool_registry: Arc<ToolRegistry>,
	) -> Self {
		Self::with_provider_sessions(registry, sessions, tool_registry, true, None)
	}

	/// Constructs the production RPC projection around a deterministic route
	/// registry whose route stack does not install provider-session middleware.
	///
	/// This is an integration-test seam only. Gateway context, turn replay, and
	/// duplex projection remain owned by this service.
	#[doc(hidden)]
	#[must_use]
	pub fn new_for_test(
		registry: Registry,
		sessions: ConversationSessionPlanner,
		tool_registry: Arc<ToolRegistry>,
		live_responses: flume::Sender<WorkflowResponse>,
	) -> Self {
		Self::with_provider_sessions(registry, sessions, tool_registry, false, Some(live_responses))
	}

	fn with_provider_sessions(
		registry: Registry,
		sessions: ConversationSessionPlanner,
		tool_registry: Arc<ToolRegistry>,
		provider_sessions: bool,
		test_live_responses: Option<flume::Sender<WorkflowResponse>>,
	) -> Self {
		let epoch = format!("{}:{}", registry.catalog_revision(), registry.generation()).into_bytes();
		Self {
			registry,
			tool_registry,
			sessions,
			provider_sessions,
			test_live_responses,
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

	fn turn_client(
		&self,
		target: Target,
		request: RequestId,
		session: Option<SessionRequest>,
	) -> Client<omp_llm_inference::ProviderService, Router> {
		Client::new(
			self.registry.service(),
			Router::new(self.registry.clone(), Duration::from_secs(30)),
			CallMeta {
				id: request,
				target,
				deadline: None,
				budget: ExecutionBudget::default(),
				session,
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
		turn: ProviderTurnId,
		input: Option<&pb::turn_request::Input>,
	) -> Result<ResolvedTurn, Status> {
		let strategy = ContextStrategy::Replay;
		match input {
			Some(pb::turn_request::Input::Seed(seed)) => {
				let thread = seed
					.thread
					.as_ref()
					.ok_or_else(|| Status::invalid_argument("Seed.thread is required"))?;
				let projected =
					project_thread_history(thread, &self.tool_registry, &RPC_HISTORY_PROMPT_CAPS)
					.map_err(|error| Status::invalid_argument(error.to_string()))?;
				let messages = thread_messages(&projected)?;
				if seed.context_id.is_empty() {
					return Ok(ResolvedTurn {
						request_messages:      messages.clone(),
						committed_messages:    messages,
						context_id:            None,
						provider_session:      None,
						provider_conversation: None,
						provider_heads:        BTreeMap::new(),
					});
				}
				if self.contexts.lock().contains_key(&seed.context_id) {
					return Err(Status::aborted("seed context is already held"));
				}
				if !self.provider_sessions {
					return Ok(ResolvedTurn {
						request_messages:      messages.clone(),
						committed_messages:    messages,
						context_id:            Some(seed.context_id.clone()),
						provider_session:      None,
						provider_conversation: None,
						provider_heads:        BTreeMap::new(),
					});
				}
				let root = self
					.sessions
					.create_conversation()
					.map_err(conversation_status)?;
				let conversation = root.conversation().clone();
				let revision = root.revision().clone();
				let provider_session = SessionRequest {
					conversation: conversation.clone(),
					revision: revision.clone(),
					turn,
					strategy,
				};
				Ok(ResolvedTurn {
					request_messages:      messages.clone(),
					committed_messages:    messages,
					context_id:            Some(seed.context_id.clone()),
					provider_session:      Some(provider_session),
					provider_conversation: Some(conversation),
					provider_heads:        BTreeMap::from([(0, revision)]),
				})
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
				let projected = project_thread_history(
					&thread_pb::Thread { items: delta.append.clone() },
					&self.tool_registry,
					&RPC_HISTORY_PROMPT_CAPS,
				)
				.map_err(|error| Status::invalid_argument(error.to_string()))?;
				let appended = thread_messages(&projected)?;
				let mut committed_messages = held
					.messages
					.iter()
					.take(retained as usize)
					.cloned()
					.collect::<Vec<_>>();
				committed_messages.extend(appended.iter().cloned());
				if !self.provider_sessions {
					return Ok(ResolvedTurn {
						request_messages: committed_messages.clone(),
						committed_messages,
						context_id: Some(context.context_id.clone()),
						provider_session: None,
						provider_conversation: None,
						provider_heads: BTreeMap::new(),
					});
				}
				let (request_messages, conversation, revision, provider_heads) =
					if (delta.truncate_to.is_none() || retained == held.revision)
						&& held.provider_heads.contains_key(&retained)
					{
						(
							appended,
							held.provider_conversation.clone().ok_or_else(|| {
								Status::internal("held context has no provider conversation")
							})?,
							held
								.provider_revision
								.clone()
								.ok_or_else(|| Status::internal("held context has no provider revision"))?,
							held.provider_heads,
						)
					} else if let Some(revision) = held.provider_heads.get(&retained).cloned() {
						let conversation = self
							.sessions
							.fork_conversation(&revision)
							.map_err(conversation_status)?;
						(
							appended,
							conversation,
							revision,
							held
								.provider_heads
								.into_iter()
								.filter(|(head, _)| *head <= retained)
								.collect(),
						)
					} else {
						let root = self
							.sessions
							.create_conversation()
							.map_err(conversation_status)?;
						(
							committed_messages.clone(),
							root.conversation().clone(),
							root.revision().clone(),
							BTreeMap::from([(0, root.revision().clone())]),
						)
					};
				let provider_session =
					SessionRequest { conversation: conversation.clone(), revision, turn, strategy };
				Ok(ResolvedTurn {
					request_messages,
					committed_messages,
					context_id: Some(context.context_id.clone()),
					provider_session: Some(provider_session),
					provider_conversation: Some(conversation),
					provider_heads,
				})
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
		let turn = ProviderTurnId::from(open.turn_id.as_str());
		if let Some(replay) = self
			.sessions
			.turn_replay(&turn)
			.map_err(conversation_status)?
		{
			let output = turn_replay_events(replay, &open)?;
			return Ok(Response::new(Box::pin(output)));
		}
		let request_bytes = Bytes::from(open.encode_to_vec());
		let params = open
			.params
			.as_ref()
			.ok_or_else(|| Status::invalid_argument("TurnRequest.params is required"))?;
		let mut resolved = match self.resolve_turn_input(turn.clone(), open.input.as_ref()) {
			Ok(resolved) => resolved,
			Err(status) => {
				let Some(event) = turn_recovery_event(&status, open.input.as_ref(), &self.contexts)
				else {
					return Err(status);
				};
				return Ok(Response::new(Box::pin(stream::once(async move { Ok(event) }))));
			},
		};
		let projection = Arc::new(Mutex::new(TurnProjection::default()));
		let request_id = RequestId::from(open.turn_id.as_str());
		if resolved.provider_session.is_some() {
			let replay_projection = Arc::clone(&projection);
			let replay_context = resolved.context_id.clone();
			let committed_len = resolved.committed_messages.len();
			self.sessions.stage_turn_replay(
				request_id.clone(),
				turn.clone(),
				request_bytes.clone(),
				move |completion| {
					Ok(Bytes::from(
						build_turn_outcome(
							&replay_projection.lock(),
							completion,
							replay_context.as_deref(),
							committed_len,
						)
						.encode_to_vec(),
					))
				},
			);
		}
		let chat = chat_request(
			std::mem::take(&mut resolved.request_messages),
			params,
			&self.tool_registry,
		)?;
		let target = self.target(&params.model, OperationKind::Chat)?;
		let mut client = self.turn_client(target, request_id, resolved.provider_session.clone());
		let events = match client.execute(chat).await {
			Ok(events) => events,
			Err(error) => {
				let event = inference_turn_error(error);
				return Ok(Response::new(Box::pin(stream::once(async move { Ok(event) }))));
			},
		};
		let output = turn_events(
			events,
			incoming,
			Arc::clone(&self.contexts),
			self.sessions.clone(),
			resolved,
			turn,
			request_bytes,
			projection,
			Arc::clone(&self.tool_registry),
			self.test_live_responses.clone(),
		);
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
		let provider_revision = source.provider_heads.get(&at).cloned();
		let provider_conversation = provider_revision
			.as_ref()
			.map(|revision| self.sessions.fork_conversation(revision))
			.transpose()
			.map_err(conversation_status)?;
		let provider_heads = if provider_revision.is_some() {
			source
				.provider_heads
				.iter()
				.filter(|(head, _)| **head <= at)
				.map(|(head, revision)| (*head, revision.clone()))
				.collect()
		} else {
			BTreeMap::new()
		};
		let fork = RpcContext {
			revision: at,
			messages: source.messages.into_iter().take(at as usize).collect(),
			provider_conversation,
			provider_revision,
			provider_heads,
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
fn inference_turn_error(error: Error) -> pb::TurnEvent {
	let kind = match error.kind {
		ErrorKind::Authentication
		| ErrorKind::Authorization
		| ErrorKind::AccountDisabled
		| ErrorKind::PaymentRequired => pb::turn_error::Kind::Auth,
		ErrorKind::RateLimited | ErrorKind::QuotaExhausted => pb::turn_error::Kind::RateLimited,
		ErrorKind::BudgetExhausted | ErrorKind::ResourceExhausted => pb::turn_error::Kind::Overloaded,
		_ => pb::turn_error::Kind::Upstream,
	};
	let retry_after_ms = match error.action {
		omp_llm_inference::RetryAction::SameRoute { after } => {
			after.as_millis().try_into().unwrap_or(u64::MAX)
		},
		_ => 0,
	};
	pb::TurnEvent {
		event: Some(pb::turn_event::Event::Error(pb::TurnError {
			kind: kind as i32,
			detail: format!("{:?} during {:?}", error.kind, error.phase),
			actual: None,
			unsupported: Vec::new(),
			retry_after_ms,
			diagnostics: Vec::new(),
			error_id: None,
		})),
	}
}

fn conversation_status(error: omp_llm_inference::session::ConversationError) -> Status {
	match error {
		omp_llm_inference::session::ConversationError::RevisionConflict { .. }
		| omp_llm_inference::session::ConversationError::TurnConflict(_) => {
			Status::aborted(error.to_string())
		},
		omp_llm_inference::session::ConversationError::UnknownConversation(_)
		| omp_llm_inference::session::ConversationError::UnknownRevision(_) => {
			Status::not_found(error.to_string())
		},
		_ => Status::internal(error.to_string()),
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

/// Exercises the same canonical history and live-definition projection used by
/// [`InferenceRpc::turn`] without opening a transport.
#[doc(hidden)]
pub fn project_provider_turn_for_test(
	thread: &thread_pb::Thread,
	params: &pb::ChatParams,
	tool_registry: &ToolRegistry,
) -> Result<(thread_pb::Thread, omp_llm_inference::call::ChatRequest), Status> {
	let projected = project_thread_history(thread, tool_registry, &RPC_HISTORY_PROMPT_CAPS)
		.map_err(|error| Status::invalid_argument(error.to_string()))?;
	let request = chat_request(thread_messages(&projected)?, params, tool_registry)?;
	Ok((projected, request))
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
		input:       ToolInputConstraint::JsonSchema {
			parameters: opaque_json(&tool.schema_json, "ToolDef.schema_json")?,
			strict:     tool.strict.unwrap_or(false),
		},
	})
}

fn chat_request(
	messages: Vec<Message>,
	params: &pb::ChatParams,
	tool_registry: &ToolRegistry,
) -> Result<omp_llm_inference::call::ChatRequest, Status> {
	let advertised = tool_registry
		.advertise(LoweringCaps { strict_schema: false, grammar: GrammarBits::empty() });
	if let Some(tool) = params
		.tools
		.iter()
		.find(|tool| tool_registry.live_identity(&tool.name).is_none())
	{
		return Err(Status::failed_precondition(format!(
			"executable harness tool `{}` has no live registry identity",
			tool.name
		)));
	}
	let tools: Vec<ToolDefinition> = advertised
		.into_iter()
		.filter(|tool| {
			params.tools.is_empty()
				|| params.tools.iter().any(|requested| requested.name == tool.identity.name.as_str())
		})
		.map(|tool| tool.definition)
		.collect();
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

fn tool_revision_props(registry: &ToolRegistry, name: &str) -> Option<pb::ValueMap> {
	let (_, revision) = registry.live_identity(name)?;
	Some(pb::ValueMap {
		fields: BTreeMap::from([(TOOL_REV_PROP.to_owned(), pb::Value {
			kind: Some(pb::value::Kind::String(revision.to_string())),
		})]),
	})
}

fn build_turn_outcome(
	projection: &TurnProjection,
	completion: &Completion,
	context_id: Option<&str>,
	committed_len: usize,
) -> pb::Outcome {
	let mut output = projection.output.clone();
	if !projection.assistant_text.is_empty() {
		output.insert(0, thread_pb::Item {
			seq:           0,
			created_at_ms: 0,
			kind:          Some(thread_pb::item::Kind::Message(thread_pb::Message {
				role:  thread_pb::Role::Assistant as i32,
				parts: vec![thread_pb::Part {
					kind: Some(thread_pb::part::Kind::Text(projection.assistant_text.clone())),
				}],
			})),
			props:         None,
		});
	}
	let mut head = u64::try_from(committed_len).unwrap_or(u64::MAX);
	for item in &mut output {
		head = head.saturating_add(1);
		item.seq = head;
	}
	pb::Outcome {
		output,
		stop: match &completion.reason {
			FinishReason::Stop | FinishReason::Other(_) => 1,
			FinishReason::Length => 3,
			FinishReason::ToolCalls => 2,
			FinishReason::ContentFilter => 4,
			FinishReason::Cancelled => 0,
		},
		usage: Some(proto_usage(completion.usage)),
		cost: Some(proto_cost(completion.receipt.cost)),
		unsupported: Vec::new(),
		revision: context_id.map(|context| revision(context, head)),
		provider: completion
			.receipt
			.plan
			.provider
			.as_ref()
			.map_or_else(String::new, |value| value.as_str().to_owned()),
		model: completion
			.receipt
			.plan
			.model
			.as_ref()
			.map_or_else(String::new, |value| value.as_str().to_owned()),
		diagnostics: Vec::new(),
		upstream_provider: None,
		duration_ms: Some(
			completion
				.receipt
				.timings
				.total
				.as_millis()
				.try_into()
				.unwrap_or(u64::MAX),
		),
		ttft_ms: completion
			.receipt
			.timings
			.first_frame
			.map(|value| value.as_millis().try_into().unwrap_or(u64::MAX)),
		context_snapshot: None,
		props: None,
	}
}

fn turn_replay_events(
	replay: TurnReplay,
	request: &pb::TurnRequest,
) -> Result<impl Stream<Item = Result<pb::TurnEvent, Status>> + Send + 'static, Status> {
	let stored_request = pb::TurnRequest::decode(replay.request)
		.map_err(|_| Status::internal("stored turn request is corrupt"))?;
	if stored_request != *request {
		return Err(Status::already_exists(
			"turn_id already committed with a different opening request",
		));
	}
	let outcome = pb::Outcome::decode(replay.outcome)
		.map_err(|_| Status::internal("stored turn outcome is corrupt"))?;
	Ok(stream::iter([
		Ok(pb::TurnEvent {
			event: Some(pb::turn_event::Event::Accepted(pb::Accepted { replay: true })),
		}),
		Ok(pb::TurnEvent { event: Some(pb::turn_event::Event::Outcome(outcome)) }),
	]))
}

#[derive(Clone)]
struct PendingInvocation {
	kind:       WorkflowResponseKind,
	deadline:   Option<Instant>,
	tool_call:  Option<thread_pb::ToolCall>,
	tool_props: Option<pb::ValueMap>,
}

enum TurnMux {
	Event(Option<Result<ChatEvent, Error>>),
	Frame(Result<Option<pb::TurnFrame>, Status>),
	Timeout(String),
}
async fn route_live_turn_frame(
	frame: pb::TurnFrame,
	control: Option<&ChatControl>,
	test_live_responses: Option<&flume::Sender<WorkflowResponse>>,
	pending: &mut BTreeMap<String, PendingInvocation>,
	projection: &Arc<Mutex<TurnProjection>>,
) -> Result<(), Status> {
	let mut completion_result = None;
	let response = match frame.frame {
		Some(pb::turn_frame::Frame::Input(input)) if !input.invocation_id.is_empty() => {
			let Some(invocation) = pending.get(&input.invocation_id) else {
				return Err(Status::invalid_argument("unknown or late invocation_id"));
			};
			if invocation.kind != WorkflowResponseKind::Invoke {
				return Err(Status::invalid_argument(
					"provider action does not accept incremental invocation input",
				));
			}
			WorkflowResponse::InvokeInput(InvokeInput {
				invocation: Str::from(input.invocation_id.as_str()),
				payload:    Bytes::from(input.encode_to_vec()),
			})
		},
		Some(pb::turn_frame::Frame::Complete(complete)) if !complete.invocation_id.is_empty() => {
			let Some(invocation) = pending.get(&complete.invocation_id).cloned() else {
				return Err(Status::invalid_argument("unknown or late invocation_id"));
			};
			if let (Some(call), Some(result)) =
				(invocation.tool_call.as_ref(), complete.tool_result.as_ref())
				&& !result.call_id.is_empty()
				&& result.call_id != call.id
			{
				return Err(Status::invalid_argument(
					"tool_result.call_id does not match invocation tool_call",
				));
			}
			completion_result = complete.tool_result.clone();
			match invocation.kind {
				WorkflowResponseKind::Action => {
					let (response, is_error) = workflow_action_result(&complete)?;
					WorkflowResponse::WorkflowActionResponse(WorkflowActionResponse {
						invocation: Str::from(complete.invocation_id.as_str()),
						response,
						is_error,
					})
				},
				WorkflowResponseKind::Invoke => WorkflowResponse::InvokeComplete(InvokeComplete {
					invocation: Str::from(complete.invocation_id.as_str()),
					payload:    Bytes::from(complete.encode_to_vec()),
				}),
			}
		},
		Some(pb::turn_frame::Frame::Open(_)) => {
			return Err(Status::invalid_argument("Turn open frame may only appear first"));
		},
		Some(_) => return Err(Status::invalid_argument("invocation_id is required")),
		None => return Err(Status::invalid_argument("Turn frame body is required")),
	};
	let terminal = response.is_terminal();
	let invocation_id = response.invocation().as_str().to_owned();
	if let Some(control) = control {
		control
			.submit(response)
			.await
			.map_err(|error| match error {
				ChatControlError::DeadlineExceeded => {
					Status::deadline_exceeded("invoke deadline exceeded")
				},
				ChatControlError::UnknownInvocation => {
					Status::invalid_argument("unknown or late invocation_id")
				},
				ChatControlError::Closed => {
					Status::failed_precondition("live invocation path is closed")
				},
			})?;
	} else if let Some(responses) = test_live_responses {
		responses
			.send_async(response)
			.await
			.map_err(|_| Status::failed_precondition("test live invocation observer closed"))?;
	} else {
		return Err(Status::failed_precondition(
			"selected provider does not accept live invocation responses",
		));
	}
	if terminal
		&& let Some(invocation) = pending.remove(&invocation_id)
		&& let (Some(call), Some(result)) = (invocation.tool_call, completion_result)
	{
		let mut projection = projection.lock();
		projection.output.push(thread_pb::Item {
			seq:           0,
			created_at_ms: 0,
			kind:          Some(thread_pb::item::Kind::ToolCall(call)),
			props:         invocation.tool_props,
		});
		projection.output.push(thread_pb::Item {
			seq:           0,
			created_at_ms: 0,
			kind:          Some(thread_pb::item::Kind::ToolResult(result)),
			props:         None,
		});
	}
	Ok(())
}

fn workflow_action_result(complete: &pb::InvokeComplete) -> Result<(Bytes, bool), Status> {
	if let Some(result) = complete.tool_result.as_ref() {
		let mut text = String::new();
		for part in &result.parts {
			match part.kind.as_ref() {
				Some(thread_pb::part::Kind::Text(part)) => text.push_str(part),
				_ => {
					return Err(Status::invalid_argument(
						"workflow action results accept text parts only",
					));
				},
			}
		}
		return Ok((Bytes::from(text), result.is_error));
	}
	if !complete.vendor.is_empty() {
		let is_error = complete
			.status
			.as_ref()
			.is_some_and(|status| status.outcome() != pb::exec_status::Outcome::Exited);
		return Ok((complete.vendor.clone(), is_error));
	}
	Err(Status::invalid_argument(
		"workflow action completion requires tool_result or vendor payload",
	))
}

fn turn_recovery_event(
	status: &Status,
	input: Option<&pb::turn_request::Input>,
	contexts: &Mutex<BTreeMap<String, RpcContext>>,
) -> Option<pb::TurnEvent> {
	let kind = match status.code() {
		tonic::Code::Aborted => pb::turn_error::Kind::Conflict,
		tonic::Code::NotFound => pb::turn_error::Kind::NeedFull,
		_ => return None,
	};
	let context_id = match input {
		Some(pb::turn_request::Input::Seed(seed)) => Some(seed.context_id.as_str()),
		Some(pb::turn_request::Input::Incremental(incremental)) => incremental
			.context
			.as_ref()
			.map(|context| context.context_id.as_str()),
		None => None,
	};
	let actual = (kind == pb::turn_error::Kind::Conflict)
		.then(|| {
			let context_id = context_id?;
			let held = contexts.lock();
			let context = held.get(context_id)?;
			Some(revision(context_id, context.revision))
		})
		.flatten();
	Some(pb::TurnEvent {
		event: Some(pb::turn_event::Event::Error(pb::TurnError {
			kind: kind as i32,
			detail: status.message().to_owned(),
			actual,
			unsupported: Vec::new(),
			retry_after_ms: 0,
			diagnostics: Vec::new(),
			error_id: None,
		})),
	})
}
fn invoke_timeout(invocation_id: &str) -> pb::TurnEvent {
	pb::TurnEvent {
		event: Some(pb::turn_event::Event::Error(pb::TurnError {
			kind:           pb::turn_error::Kind::InvokeTimeout as i32,
			detail:         format!("invocation {invocation_id} exceeded its completion deadline"),
			actual:         None,
			unsupported:    Vec::new(),
			retry_after_ms: 0,
			diagnostics:    Vec::new(),
			error_id:       None,
		})),
	}
}

fn turn_events(
	mut events: omp_llm_inference::answer::ChatStream,
	mut incoming: tonic::Streaming<pb::TurnFrame>,
	contexts: Arc<Mutex<BTreeMap<String, RpcContext>>>,
	sessions: ConversationSessionPlanner,
	mut resolved: ResolvedTurn,
	turn: ProviderTurnId,
	request_bytes: Bytes,
	projection: Arc<Mutex<TurnProjection>>,
	tool_registry: Arc<ToolRegistry>,
	test_live_responses: Option<flume::Sender<WorkflowResponse>>,
) -> impl Stream<Item = Result<pb::TurnEvent, Status>> + Send + 'static {
	let control = events.control();
	async_stream::try_stream! {
		yield pb::TurnEvent {
			event: Some(pb::turn_event::Event::Accepted(pb::Accepted { replay: false })),
		};
		let mut pending = BTreeMap::<String, PendingInvocation>::new();
		let mut incoming_open = true;
		loop {
			let event = loop {
				let next_timeout = pending
					.iter()
					.filter_map(|(id, invocation)| invocation.deadline.map(|deadline| (id.clone(), deadline)))
					.min_by_key(|(_, deadline)| *deadline);
				let mux = tokio::select! {
					event = events.next(), if pending.is_empty() || test_live_responses.is_none() => TurnMux::Event(event),
					frame = incoming.message(), if incoming_open => TurnMux::Frame(frame),
					invocation_id = async {
						match next_timeout {
							Some((invocation_id, deadline)) => {
								tokio::time::sleep_until(deadline.into()).await;
								invocation_id
							},
							None => std::future::pending().await,
						}
					} => TurnMux::Timeout(invocation_id),
				};
				match mux {
					TurnMux::Event(event) => break event,
					TurnMux::Frame(frame) => match frame? {
						Some(frame) => {
							let frame_id = match frame.frame.as_ref() {
								Some(pb::turn_frame::Frame::Input(input)) => input.invocation_id.clone(),
								Some(pb::turn_frame::Frame::Complete(complete)) => complete.invocation_id.clone(),
								_ => String::new(),
							};
							if let Err(status) = route_live_turn_frame(
								frame,
								control.as_ref(),
								test_live_responses.as_ref(),
								&mut pending,
								&projection,
							).await {
								if status.code() == tonic::Code::DeadlineExceeded {
									yield invoke_timeout(&frame_id);
									return;
								}
								Err(status)?;
							}
						},
						None => incoming_open = false,
					},
					TurnMux::Timeout(invocation_id) => {
						yield invoke_timeout(&invocation_id);
						return;
					},
				}
			};
			let Some(event) = event else { break };
			let event = match event {
				Ok(event) => event,
				Err(error) if !pending.is_empty() && error.kind == ErrorKind::DeadlineExceeded => {
					let invocation_id = pending.keys().next().expect("pending invocation").clone();
					yield invoke_timeout(&invocation_id);
					return;
				},
				Err(error) => {
					yield inference_turn_error(error);
					return;
				},
			};
			match event {
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
					projection.lock().assistant_text.push_str(text.as_str());
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
					let props = tool_revision_props(tool_registry.as_ref(), call.name.as_str());
					projection.lock().output.push(thread_pb::Item {
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
						props,
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
				ChatEvent::WorkflowAction(action) => {
					if control.is_none() && test_live_responses.is_none() {
						Err(Status::failed_precondition(
							"provider emitted a workflow action without a live response path",
						))?;
					}
					let invocation_id = action.invocation.as_str().to_owned();
					if pending.contains_key(&invocation_id) {
						Err(Status::failed_precondition("provider reused a live invocation_id"))?;
					}
					let deadline = action.timeout.map(|timeout| Instant::now() + timeout);
					let vendor = action.call.is_none().then(|| action.arguments.clone()).unwrap_or_default();
					let tool_props = action
						.call
						.as_ref()
						.and_then(|_| tool_revision_props(tool_registry.as_ref(), action.name.as_str()));
					let tool_call = action.call.map(|call| thread_pb::ToolCall {
						id: call.as_str().to_owned(),
						name: action.name.as_str().to_owned(),
						args_json: action.arguments,
						thought_signature: Bytes::new(),
						intent: None,
						raw: None,
						custom_wire_name: None,
						provider_metadata: None,
					});
					pending.insert(
						invocation_id.clone(),
						PendingInvocation {
							kind: action.response_kind,
							deadline,
							tool_call: tool_call.clone(),
							tool_props,
						},
					);
					yield pb::TurnEvent {
						event: Some(pb::turn_event::Event::Invoke(pb::Invoke {
							invocation_id: invocation_id,
							name: action.name.as_str().to_owned(),
							tool_call,
							vendor,
							timeout_ms: action.timeout.map_or(0, |value| value.as_millis().try_into().unwrap_or(u64::MAX)),
							props: None,
						})),
					};
				},
				ChatEvent::WorkflowResume(_) => {},
				ChatEvent::WorkflowCancelled { invocation } => {
					let invocation_id = invocation.as_str().to_owned();
					pending.remove(&invocation_id);
					yield pb::TurnEvent {
						event: Some(pb::turn_event::Event::InvokeCancel(pb::InvokeCancel {
							invocation_id,
						})),
					};
				},
				ChatEvent::Completed(completion) => {
					if !pending.is_empty() {
						Err(Status::failed_precondition(
							"provider completed with live invocations outstanding",
						))?;
					}
					let outcome = build_turn_outcome(
						&projection.lock(),
						&completion,
						resolved.context_id.as_deref(),
						resolved.committed_messages.len(),
					);
					let provider_revision = if let Some(conversation) =
						resolved.provider_conversation.as_ref()
					{
						Some(
							sessions
								.committed_turn(conversation, &turn)
								.map_err(conversation_status)?
								.ok_or_else(|| {
									Status::internal("completed provider turn has no committed revision")
								})?
								.revision()
								.clone(),
						)
					} else {
						None
					};
					let committed_context =
						if let (Some(context_id), Some(next_revision)) =
							(resolved.context_id.as_ref(), outcome.revision.as_ref())
						{
							let mut messages = resolved.committed_messages.clone();
							messages.extend(items_messages(&outcome.output)?);
							let head = next_revision.head;
							if let Some(provider_revision) = provider_revision.as_ref() {
								resolved.provider_heads.insert(head, provider_revision.clone());
							}
							Some((
								context_id.clone(),
								RpcContext {
									revision: head,
									messages,
									provider_conversation: resolved.provider_conversation.clone(),
									provider_revision,
									provider_heads: std::mem::take(&mut resolved.provider_heads),
								},
							))
						} else {
							None
						};
					if resolved.provider_session.is_none() {
						sessions
							.commit_turn_replay(
								turn.clone(),
								request_bytes.clone(),
								Bytes::from(outcome.encode_to_vec()),
							)
							.map_err(conversation_status)?;
					}
					if let Some((context_id, context)) = committed_context {
						contexts.lock().insert(context_id, context);
					}
					yield pb::TurnEvent {
						event: Some(pb::turn_event::Event::Outcome(outcome)),
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
		ChatEvent::WorkflowAction(_)
		| ChatEvent::WorkflowResume(_)
		| ChatEvent::WorkflowCancelled { .. } => {
			return Err(Status::failed_precondition(
				"workflow control events require the duplex Turn RPC",
			));
		},
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

#[cfg(test)]
mod tests {
	use futures::StreamExt as _;

	use super::*;

	#[test]
	fn invocation_timeout_projects_the_dedicated_turn_error_kind() {
		let event = invoke_timeout("invoke-9");
		assert!(matches!(
			event.event,
			Some(pb::turn_event::Event::Error(pb::TurnError {
				kind,
				detail,
				..
			})) if kind == pb::turn_error::Kind::InvokeTimeout as i32
				&& detail.contains("invoke-9")
		));
	}

	#[test]
	fn provider_owned_calls_without_registry_identity_remain_unstamped() {
		let registry = ToolRegistry::new();
		assert_eq!(tool_revision_props(&registry, "provider.search"), None);
	}

	#[test]
	fn workflow_action_completion_preserves_text_and_error_classification() {
		let complete = pb::InvokeComplete {
			invocation_id: "invoke-1".to_owned(),
			tool_result: Some(thread_pb::ToolResult {
				parts: vec![
					thread_pb::Part { kind: Some(thread_pb::part::Kind::Text("first".to_owned())) },
					thread_pb::Part { kind: Some(thread_pb::part::Kind::Text(" second".to_owned())) },
				],
				is_error: true,
				..Default::default()
			}),
			..Default::default()
		};
		let (payload, is_error) = workflow_action_result(&complete).expect("text workflow response");
		assert_eq!(payload.as_ref(), b"first second");
		assert!(is_error);
	}

	#[tokio::test]
	async fn committed_turn_replay_is_exact_and_mismatched_open_is_rejected() {
		let request = pb::TurnRequest { turn_id: "turn-1".to_owned(), ..Default::default() };
		let outcome = pb::Outcome { model: "recorded-model".to_owned(), ..Default::default() };
		let replay = TurnReplay {
			request: Bytes::from(request.encode_to_vec()),
			outcome: Bytes::from(outcome.encode_to_vec()),
		};
		let events = turn_replay_events(replay.clone(), &request)
			.expect("matching replay")
			.collect::<Vec<_>>()
			.await;
		assert!(matches!(
			events.as_slice(),
			[
				Ok(pb::TurnEvent {
					event: Some(pb::turn_event::Event::Accepted(pb::Accepted {
						replay: true
					}))
				}),
				Ok(pb::TurnEvent {
					event: Some(pb::turn_event::Event::Outcome(actual))
				}),
			] if actual == &outcome
		));
		let mismatched = pb::TurnRequest {
			turn_id: "turn-1".to_owned(),
			params: Some(pb::ChatParams { model: "different".to_owned(), ..Default::default() }),
			..Default::default()
		};
		let status = match turn_replay_events(replay, &mismatched) {
			Ok(_) => panic!("mismatched replay payload must be rejected"),
			Err(status) => status,
		};
		assert_eq!(status.code(), tonic::Code::AlreadyExists);
	}
}
