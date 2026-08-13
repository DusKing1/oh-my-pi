//! Sans-I/O wire-codec contracts shared by every inference transport.

use std::{
	fmt,
	pin::Pin,
	sync::{
		Arc,
		atomic::{AtomicBool, Ordering},
	},
	task::{Context, Poll},
	time::Duration,
};

use bytes::Bytes;
use futures::{Stream, task::AtomicWaker};
use omp_core::Str;
use omp_llm_catalog::{
	DiscoveredModel, OperationKind, PolicyModel, ProviderId, RouteDef, RouteId, ThinkingPolicy,
	ThinkingSelection, WireTarget, policy::WirePolicy,
};
use smallvec::SmallVec;

use crate::{
	answer::{
		AnswerBody, AudioChunk, GenerationEvent, ImageArtifact, RealtimeEvent, RealtimeInput,
		RealtimeSession, TranscriptEvent, VideoArtifact,
	},
	auth::{BodyPlacement, lease::AppliedCredentials},
	body::{AttemptEvidenceHandle, BodySource},
	call::{AccountRoutingContext, OperationCall, SessionRequest},
	error::Error,
	event::{ChatEvent, FinishReason},
	id::{AccountId, PrincipalId, RequestId, ToolCallId},
	receipt::Usage,
	transport::{Frame, FramingProtocol},
};

pub mod anthropic;
pub mod cursor;
pub mod discovery;
pub mod gemini;
pub mod google_cca;
pub mod ollama;
pub mod openai;
pub mod openai_chat;
pub mod openai_codex;
pub mod openai_embedding;
pub mod openai_media;
pub mod openai_realtime;
pub mod openai_responses;

pub mod bedrock;
pub mod devin;
pub mod gitlab;
pub mod native;
pub mod omp_native;
pub mod search_exa;
pub mod search_kagi;
pub mod search_parallel;
pub mod search_perplexity;
pub mod search_tavily;
/// HTTP method used by a wire request without pulling policy into the
/// transport.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestMethod {
	/// Read a resource.
	Get,
	/// Create or invoke a resource.
	Post,
	/// Replace a resource.
	Put,
	/// Partially update a resource.
	Patch,
	/// Delete a resource.
	Delete,
}

/// A public, non-secret request header produced by a codec.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestHeader {
	/// Header name.
	pub name:  Str,
	/// Header value; credentials are prohibited here.
	pub value: Str,
}

/// Explicit request and response byte limits enforced by the transport.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SizeBounds {
	/// Maximum encoded request body size.
	pub request_body: u64,
	/// Maximum individual framed payload size.
	pub frame:        u64,
	/// Maximum aggregate response bytes.
	pub response:     u64,
}

/// Crate-private typed request body awaiting credential binding.
///
/// Templates contain no credential material. They are consumed at the
/// innermost transport boundary and deliberately have no serialization or
/// public inspection surface.
pub(crate) enum SealedBodyTemplate {
	Devin(devin::DevinSealedBody),
}

impl SealedBodyTemplate {
	pub(crate) const fn placement(&self) -> BodyPlacement {
		match self {
			Self::Devin(_) => BodyPlacement::DevinMetadata,
		}
	}

	pub(crate) fn bind(self, secret: &str) -> Result<Bytes, crate::auth::CredentialApplyError> {
		match self {
			Self::Devin(template) => template.bind(secret),
		}
	}
}

impl fmt::Debug for SealedBodyTemplate {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter
			.debug_struct("SealedBodyTemplate")
			.field("placement", &self.placement())
			.field("body", &"[REDACTED]")
			.finish()
	}
}

/// Secret-free request emitted by a codec and finalized by credential
/// middleware.
pub struct EncodedRequest {
	/// Operation represented by the request.
	pub operation:          OperationKind,
	/// Wire method.
	pub method:             RequestMethod,
	/// Absolute endpoint URI including non-secret query parameters.
	pub uri:                Str,
	/// Public headers. Credential middleware owns all sensitive headers.
	pub headers:            Box<[RequestHeader]>,
	/// Fresh or one-shot request body with explicit replay semantics.
	pub body:               BodySource,
	/// Response framing selected by the codec.
	pub framing:            FramingProtocol,
	/// Enforced byte limits.
	pub bounds:             SizeBounds,
	pub(crate) sealed_body: Option<SealedBodyTemplate>,
}
impl EncodedRequest {
	/// Constructs an ordinary credential-free encoded request.
	#[must_use]
	pub fn new(
		operation: OperationKind,
		method: RequestMethod,
		uri: Str,
		headers: Box<[RequestHeader]>,
		body: BodySource,
		framing: FramingProtocol,
		bounds: SizeBounds,
	) -> Self {
		Self { operation, method, uri, headers, body, framing, bounds, sealed_body: None }
	}

	pub(crate) fn with_sealed_body(mut self, template: SealedBodyTemplate) -> Self {
		self.sealed_body = Some(template);
		self
	}

	pub(crate) fn take_sealed_body(&mut self) -> Option<SealedBodyTemplate> {
		self.sealed_body.take()
	}
}

/// Attempt identity visible to pure encoding without account or credential
/// data.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EncodeAttempt {
	/// Zero-based attempt index.
	pub index:       u32,
	/// Whether output from this attempt is held transactionally.
	pub provisional: bool,
}

/// Credential-free context for canonical-to-wire lowering.
pub struct EncodeContext<'a> {
	/// Logical request identity used by protocols with stable reconnect keys.
	pub request_id:         &'a RequestId,
	/// Complete selected route definition.
	pub route:              &'a RouteDef,
	/// Optional codec-facing target. Model-less management operations carry
	/// none.
	pub target:             Option<&'a WireTarget>,
	/// Exact capability, limit, and pricing evidence selected by the immutable
	/// plan.
	///
	/// Model-less management operations carry none.
	pub policy_model:       Option<&'a PolicyModel>,
	/// Interned lowering policy selected during planning.
	pub policy:             &'a WirePolicy,
	/// Exact model thinking policy resolved during planning.
	pub thinking_policy:    Option<&'a ThinkingPolicy>,
	/// Per-request effort, budget, mode, and wire-model selection resolved by
	/// the immutable plan.
	pub thinking_selection: Option<&'a ThinkingSelection>,
	/// Optional canonical session identity and revision.
	pub session:            Option<&'a SessionRequest>,
	/// Compatible typed provider-side state selected by session planning.
	pub server_state:       Option<&'a crate::session::ServerStateBinding>,
	/// Non-secret account/project/tenant routing metadata.
	pub account:            Option<&'a AccountRoutingContext>,
	/// Attempt metadata that may affect idempotency fields.
	pub attempt:            EncodeAttempt,
}

/// Lossless response representation requested by a native operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeResponseFormat {
	/// One typed JSON body.
	Json,
	/// One opaque binary body.
	Bytes,
	/// Incremental SSE payload bytes.
	Sse,
}

/// Credential-free context for decoding one provider attempt.
pub struct DecodeContext<'a> {
	/// Logical request identity.
	pub request_id:         &'a RequestId,
	/// Selected provider domain.
	pub provider:           &'a ProviderId,
	/// Selected route.
	pub route:              &'a RouteId,
	/// Optional codec-facing wire target selected by the immutable plan.
	///
	/// Model-less management operations carry none.
	pub target:             Option<&'a WireTarget>,
	/// Exact capability, limit, and pricing evidence selected by the immutable
	/// plan.
	///
	/// Model-less management operations carry none.
	pub policy_model:       Option<&'a PolicyModel>,
	/// Interned lowering policy used to encode the request.
	pub policy:             &'a WirePolicy,
	/// Exact model thinking policy used for this response.
	pub thinking_policy:    Option<&'a ThinkingPolicy>,
	/// Per-request thinking selection used to interpret this response.
	pub thinking_selection: Option<&'a ThinkingSelection>,
	/// Exact credential-free canonical operation used to interpret response
	/// fields omitted on wire.
	pub operation_call:     &'a OperationCall,
	/// Operation being decoded.
	pub operation:          OperationKind,
	/// Framing selected by the encoded request.
	pub framing:            FramingProtocol,
	/// Explicit lossless native representation, when decoding a native
	/// operation.
	pub native_response:    Option<NativeResponseFormat>,
	/// Zero-based attempt index.
	pub attempt:            u32,
}

impl DecodeContext<'_> {
	/// Debug-checks that the fast discriminator matches the exact canonical
	/// operation.
	///
	/// Central context constructors call this before handing the context to a
	/// decoder.
	#[inline]
	pub fn debug_assert_valid(&self) {
		debug_assert_eq!(
			self.operation,
			self.operation_call.kind(),
			"decode operation discriminator must match canonical operation",
		);
	}
}
/// Syntax category of a complete provider tool input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ToolInputKind {
	/// JSON arguments requiring schema validation.
	Json,
	/// Arbitrary freeform text accepted only by a declared freeform tool.
	Freeform,
}

/// Complete provider tool-call syntax awaiting canonical schema validation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnvalidatedToolCall {
	/// Canonical call identity.
	pub id:         ToolCallId,
	/// Provider-emitted tool name.
	pub name:       Str,
	/// Input syntax category.
	pub input_kind: ToolInputKind,
	/// Exact assembled input bytes.
	pub arguments:  Bytes,
}

/// Provider-state evidence that must survive canonical event projection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProviderStateEvent {
	/// An authoritative continuation handle was observed.
	Continuation { handle: Str },
	/// Opaque encrypted reasoning material associated with a content block.
	ReasoningSignature { index: u32, signature: Bytes },
	/// Provider-scoped proof required to replay a canonicalized tool call.
	ToolCallProof { index: u32, value: Bytes },
	/// Codec-scoped opaque canonical-history proof for hosted server blocks.
	HistoryBlock { index: u32, data: Bytes },
	/// Stable server output-item identity used by continuation protocols.
	OutputItem { index: u32, id: Str },
	/// Provider checkpoint identity and its authoritative opaque state bytes.
	Checkpoint { id: Option<Str>, data: Bytes },
}
/// Provider response metadata that is neither session state nor accounting
/// telemetry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProviderMetadataEvent {
	/// Stable response identity.
	ResponseId(Str),
	/// Candidate grounding metadata.
	Grounding { candidate: u32, data: Bytes },
	/// Candidate citation metadata.
	Citations { candidate: u32, data: Bytes },
	/// Candidate safety ratings.
	SafetyRatings { candidate: u32, data: Bytes },
	/// Provider finish explanation.
	FinishMessage { candidate: u32, message: Str },
	/// Typed auxiliary candidate part whose provider kind is preserved without
	/// interpretation.
	AuxiliaryPart { index: u32, kind: Str, label: Option<Str> },
}

/// Normalized provider safety action.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SafetyAction {
	/// No provider action was taken.
	None,
	/// Output was blocked.
	Blocked,
	/// A guardrail intervened without a full block.
	Intervened,
}

/// Category of one provider safety finding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SafetyFindingKind {
	/// Content classifier finding.
	Content,
	/// Sensitive-information finding.
	SensitiveInformation,
	/// Topic-policy finding.
	Topic,
	/// Word or phrase-policy finding.
	Word,
	/// Contextual-grounding finding.
	ContextualGrounding,
}

/// Typed confidence vocabulary retained from provider safety evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SafetyConfidence {
	/// Low confidence.
	Low,
	/// Medium confidence.
	Medium,
	/// High confidence.
	High,
}

/// Provider safety filter strength.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SafetyStrength {
	/// Low strength.
	Low,
	/// Medium strength.
	Medium,
	/// High strength.
	High,
}

/// One ordered provider safety finding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SafetyFinding {
	/// Finding category.
	pub kind:                 SafetyFindingKind,
	/// Provider category or policy label.
	pub label:                Str,
	/// Optional concrete policy identifier.
	pub policy:               Option<Str>,
	/// Action taken for this finding.
	pub action:               SafetyAction,
	/// Whether the provider reports an actual detection.
	pub detected:             bool,
	/// Optional typed confidence.
	pub confidence:           Option<SafetyConfidence>,
	/// Optional typed filter strength.
	pub strength:             Option<SafetyStrength>,
	/// Optional threshold represented in millionths.
	pub threshold_millionths: Option<u32>,
	/// Optional score represented in millionths.
	pub score_millionths:     Option<u32>,
	/// Optional matched word, regex text, or entity label.
	pub matched:              Option<Str>,
}

/// Typed telemetry emitted by provider decoders for receipts and observability.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProviderTelemetryEvent {
	/// Model-side latency reported by the provider.
	ModelLatency(Duration),
	/// Ordered safety assessment and guardrail latency.
	SafetyAssessment {
		/// Aggregate provider action.
		action:            SafetyAction,
		/// Ordered normalized findings.
		findings:          Box<[SafetyFinding]>,
		/// Provider-reported guardrail invocation latency.
		guardrail_latency: Option<Duration>,
	},
}

/// Typed protocol control emitted by codecs whose wire protocol is
/// bidirectional.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProviderControlEvent {
	/// Provider requests a correlated shell command through an already declared
	/// tool call.
	ShellInvoke {
		/// Provider invocation identity.
		invocation: Str,
		/// Optional execution identity used across streamed updates.
		exec:       Option<Str>,
		/// Canonical tool-call identity.
		call:       ToolCallId,
		/// Command text.
		command:    Str,
		/// Optional working directory.
		cwd:        Option<Str>,
		/// Optional provider deadline.
		timeout_ms: Option<u64>,
		/// Whether incremental execution updates are expected.
		streaming:  bool,
	},
	/// Provider requests a correlated interactive answer.
	InteractionQuery { id: u32, kind: Str, payload: Bytes },
	/// Provider cancels an outstanding tool or control call.
	Cancel { call: ToolCallId },
	/// Provider acknowledges incremental session state.
	StateAccepted { sequence: u64 },
	/// Provider asks the client to replay from a sequence boundary.
	ReplayFrom { sequence: u64 },
	/// Provider reports an optimistic-concurrency conflict.
	Conflict { expected: u64, actual: u64 },
	/// Provider rolls back uncommitted incremental state.
	Rollback { sequence: u64 },
	/// Provider requests an externally executed workflow action.
	WorkflowAction { request_id: Str, name: Str, arguments: Bytes, timeout_ms: Option<u64> },
	/// Provider supplies a reconnect/resume cursor for a workflow.
	WorkflowResume { workflow_id: Str, session_id: Str, last_event_id: Option<Str> },
	/// Internal envelope accepted a request and reports whether it is replayed.
	Accepted { replay: bool },
	/// Internal envelope reports an opaque revision conflict.
	RevisionConflict { actual_revision: Str },
	/// Internal envelope rolled back to an opaque revision.
	RolledBack { revision: Option<Str> },
	/// Internal envelope confirms cancellation.
	Cancelled,
}

/// Client response accepted only by a live bidirectional provider attempt.
pub type ProviderControlInput = crate::event::WorkflowResponse;

/// Codec-emitted terminal facts before final accounting is merged.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RawCompletion {
	/// Normalized provider finish reason.
	pub reason: FinishReason,
	/// Number of canonical content blocks emitted by the attempt.
	pub blocks: u32,
	/// Final attempt usage known at codec completion.
	pub usage:  Usage,
}

/// Closed, typed output vocabulary produced by sans-I/O decoders.
pub enum RawEvent {
	/// Non-terminal canonical generative chat event.
	Chat(ChatEvent),
	/// Terminal chat facts retained internally until final receipt accounting is
	/// complete.
	Completion(RawCompletion),
	/// Syntactically complete tool input that recovery must validate before
	/// authorization.
	ToolCallComplete { index: u32, call: UnvalidatedToolCall },
	/// Typed unary or operation-specific output.
	Answer(AnswerBody),
	/// Provider-side state evidence consumed by session middleware.
	ProviderState(ProviderStateEvent),
	/// Typed bidirectional protocol control.
	Control(ProviderControlEvent),
	/// Incremental image-generation output.
	ImageGeneration(GenerationEvent<ImageArtifact>),
	/// Incremental video-generation output.
	VideoGeneration(GenerationEvent<VideoArtifact>),
	/// Incremental encoded speech output.
	Audio(AudioChunk),
	/// Incremental transcription output.
	Transcript(TranscriptEvent),
	/// Lossless native response bytes emitted incrementally.
	NativeChunk(Bytes),
	/// Typed provider response metadata.
	Metadata(ProviderMetadataEvent),
	/// Typed provider telemetry consumed by receipt and observation layers.
	Telemetry(ProviderTelemetryEvent),
	/// Conservative runtime discovery rows awaiting catalog normalization.
	DiscoveredModels { rows: Vec<DiscoveredModel>, next_cursor: Option<Str> },
	/// Structured provider failure. No raw secret-bearing source text is
	/// retained.
	Failure(Error),
}

/// Provider-specific incremental decoder constructed once per attempt.
pub trait Decoder: Send {
	/// Consumes one already-framed transport payload and emits zero or more
	/// typed events.
	fn push(&mut self, frame: Frame, emit: &mut dyn FnMut(RawEvent)) -> Result<(), Error>;

	/// Completes the stream, flushing partial state or returning a typed
	/// truncation error.
	fn finish(&mut self, emit: &mut dyn FnMut(RawEvent)) -> Result<(), Error>;

	/// Returns whether this ordinary decoder owns a live provider response path.
	fn supports_control(&self) -> bool {
		false
	}

	/// Encodes one correlated client response for the same provider session.
	fn encode_control(&mut self, _input: ProviderControlInput) -> Result<Option<Bytes>, Error> {
		Ok(None)
	}
}

/// Construction-time decoder erasure at the transport I/O boundary.
pub type DecoderState = Box<dyn Decoder>;
/// Short bounded batch of provider frames produced by one canonical realtime
/// input.
///
/// A canonical action may require more than one ordered wire message, such as
/// committing buffered audio and then requesting response creation. The
/// transport enforces the frame-size bound on every element.
pub type RealtimeWireFrames = SmallVec<Bytes, 2>;

/// Short bounded batch of canonical events decoded from one realtime provider
/// payload.
///
/// Empty batches represent provider acknowledgements with no canonical meaning.
/// Multiple events preserve semantic ordering when one payload starts a block
/// and its tool call.
pub type RealtimeEvents = SmallVec<RealtimeEvent, 4>;

/// Sans-I/O provider codec for one bidirectional realtime session.
///
/// The transport owns the bounded channel pump and enforces
/// `EncodedRequest::bounds.frame` on every encoded and received frame.
pub trait RealtimeWireCodec: Send + 'static {
	/// Encodes ordered provider initialization frames after upgrade and before
	/// the session is ready.
	///
	/// The transport sends every returned frame before accepting caller input or
	/// emitting [`RealtimeEvent::Ready`].
	fn initial_frames(&mut self) -> Result<RealtimeWireFrames, Error>;

	/// Encodes one canonical caller message into a short ordered batch of
	/// provider wire frames.
	fn encode(&mut self, input: RealtimeInput) -> Result<RealtimeWireFrames, Error>;

	/// Decodes one bounded provider payload into zero or more ordered canonical
	/// events.
	fn decode(&mut self, payload: Bytes) -> Result<RealtimeEvents, Error>;
}

/// Construction-time erasure for one provider realtime wire codec.
pub type RealtimeWireCodecState = Box<dyn RealtimeWireCodec>;

/// Pure wire codec: no network, authentication, account selection, or retry
/// behavior.
pub trait Codec: Send + Sync + 'static {
	/// Lowers one canonical operation into a secret-free wire request.
	fn encode(
		&self,
		context: &EncodeContext<'_>,
		operation: &OperationCall,
	) -> Result<EncodedRequest, Error>;

	/// Lowers a realtime operation into its secret-free bidirectional transport
	/// handshake.
	///
	/// Ordinary codecs return `None`; realtime-capable codecs return the
	/// complete planned handshake without deriving protocol support from
	/// provider or model names.
	fn encode_realtime_handshake(
		&self,
		_context: &EncodeContext<'_>,
		_operation: &OperationCall,
	) -> Result<Option<EncodedRequest>, Error> {
		Ok(None)
	}

	/// Constructs fresh incremental state for one ordinary response attempt.
	fn decoder(&self, context: &DecodeContext<'_>) -> Result<DecoderState, Error>;

	/// Constructs fresh bidirectional wire state for a realtime operation.
	///
	/// Ordinary codecs return `None`; a realtime-capable codec returns `Some`
	/// and the transport request must leave its ordinary decoder absent.
	fn realtime(
		&self,
		_context: &DecodeContext<'_>,
	) -> Result<Option<RealtimeWireCodecState>, Error> {
		Ok(None)
	}
}
/// Clone-cheap cooperative cancellation shared by a transport and its response
/// stream.
#[derive(Clone, Default)]
pub struct Cancellation {
	state: Arc<CancellationState>,
}

#[derive(Default)]
struct CancellationState {
	cancelled: AtomicBool,
	waker:     AtomicWaker,
}

impl Cancellation {
	/// Requests cancellation and wakes a pending transport poll.
	pub fn cancel(&self) {
		self.state.cancelled.store(true, Ordering::Release);
		self.state.waker.wake();
	}

	/// Returns whether cancellation has been requested.
	pub fn is_cancelled(&self) -> bool {
		self.state.cancelled.load(Ordering::Acquire)
	}

	/// Registers a transport waker and observes cancellation without a lost
	/// wakeup.
	pub fn poll_cancelled(&self, context: &mut Context<'_>) -> Poll<()> {
		if self.is_cancelled() {
			return Poll::Ready(());
		}
		self.state.waker.register(context.waker());
		if self.is_cancelled() {
			Poll::Ready(())
		} else {
			Poll::Pending
		}
	}
}

impl fmt::Debug for Cancellation {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter
			.debug_struct("Cancellation")
			.field("cancelled", &self.is_cancelled())
			.finish()
	}
}

/// Attempt metadata required by the wire transport.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransportAttempt {
	/// Logical request identity.
	pub request_id:    RequestId,
	/// Provider selected for this attempt.
	pub provider:      ProviderId,
	/// Route selected for this attempt.
	pub route:         RouteId,
	/// Account selected without credential material.
	pub account:       Option<AccountId>,
	/// Principal selected for affinity.
	pub principal:     Option<PrincipalId>,
	/// Zero-based attempt index.
	pub index:         u32,
	/// Whether events remain provisional behind an output gate.
	pub provisional:   bool,
	/// Attempt-local timeout after composing the call deadline, remaining
	/// execution budget, and transport bound.
	pub timeout:       Duration,
	/// Maximum sanitized capture bytes for observability or cassettes.
	pub capture_limit: u64,
}

/// Fully encoded transport call with a fresh decoder and cancellation handle.
pub struct TransportRequest {
	/// Secret-free encoded request, never mutated by credential application.
	pub encoded:     EncodedRequest,
	/// Credentials applied at the innermost boundary and ignored by logs and
	/// cassettes.
	pub credentials: Option<AppliedCredentials>,
	/// Fresh ordinary provider decoder, present exactly when `realtime` is
	/// absent.
	pub decoder:     Option<DecoderState>,
	/// Provider realtime wire codec, present exactly when `decoder` is absent.
	pub realtime:    Option<RealtimeWireCodecState>,
	/// Cooperative cancellation handle.
	pub cancel:      Cancellation,
	/// Attempt identity and capture policy.
	pub attempt:     TransportAttempt,
}

/// Sanitized response handshake metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HandshakeMeta {
	/// HTTP-like status when the transport has one.
	pub status:              Option<u16>,
	/// Public response headers retained after allowlisting.
	pub headers:             Box<[RequestHeader]>,
	/// Provider request identifier, if present.
	pub provider_request_id: Option<Str>,
}

/// Raw event stream returned after the first decodable event or typed error is
/// known.
pub type RawEventStream = Pin<Box<dyn Stream<Item = Result<RawEvent, Error>> + Send + 'static>>;

/// Response returned only after transport handshake and first codec output.
pub struct HandshakenResponse {
	/// Sanitized handshake metadata.
	pub meta:     HandshakeMeta,
	/// Live request-body evidence retained until response stream termination.
	pub body:     AttemptEvidenceHandle,
	/// Ordinary decoded event stream.
	pub events:   Option<RawEventStream>,
	/// Same-session response path for an ordinary bidirectional stream.
	pub control:  Option<flume::Sender<ProviderControlInput>>,
	/// Owned realtime session; present exactly when `events` is absent.
	pub realtime: Option<RealtimeSession>,
}
