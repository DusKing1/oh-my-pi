//! Typed unary, streaming, artifact, and session answers.

use std::{
	fmt,
	pin::Pin,
	sync::{
		Arc,
		atomic::{AtomicBool, Ordering},
	},
	time::{Duration, SystemTime},
};

use bytes::Bytes;
use futures::Stream;
use omp_core::Str;
use secrecy::SecretString;

use crate::{
	body::ByteStream,
	catalog::{ModelKey, ModelSpec, ProviderId, RouteId},
	error::Error,
	event::ChatEvent,
	id::{AccountId, GenerationHandle, LoginSessionId, PrincipalId, RequestId, ToolCallId},
	operation::job::{
		JobCancelError, JobCancelHandle, JobCancellationReceipt, JobCheckpoint, JobCheckpointHandle,
		JobRef,
	},
	receipt::{Cost, ExecutionReceipt, Usage, UsageSource},
};

/// Owned asynchronous stream of fallible values.
pub type OutputStream<T> = Pin<Box<dyn Stream<Item = Result<T, Error>> + Send + 'static>>;

/// Canonical chat event stream.
pub type ChatStream = OutputStream<ChatEvent>;

/// Stream of long-running generation state and artifacts.
pub type GenerationStream<T> = OutputStream<GenerationEvent<T>>;

/// Stream of encoded audio chunks.
pub type AudioStream = OutputStream<AudioChunk>;

/// Stream of incremental transcript events.
pub type TranscriptStream = OutputStream<TranscriptEvent>;

/// Metadata common to every successful answer.
#[derive(Clone, Debug)]
pub struct ResponseMeta {
	/// Logical request identity.
	pub request_id:          RequestId,
	/// Provider selected for the successful attempt.
	pub provider:            ProviderId,
	/// Concrete selected route.
	pub route:               RouteId,
	/// Normalized selected model for model-scoped operations.
	pub model:               Option<ModelKey>,
	/// Sanitized provider request identifier.
	pub provider_request_id: Option<Str>,
	/// Wall-clock time at which the answer handshake completed.
	pub created_at:          SystemTime,
}

/// Successful erased service response.
pub struct Answer {
	/// Response identity and selected route metadata.
	pub meta:    ResponseMeta,
	/// Execution accounting available when the answer handshake completes.
	///
	/// For streaming chat, [`crate::event::Completion::receipt`] is the
	/// authoritative final receipt.
	pub receipt: ExecutionReceipt,
	/// Operation-specific response body.
	pub body:    AnswerBody,
}

impl fmt::Debug for Answer {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter
			.debug_struct("Answer")
			.field("meta", &self.meta)
			.field("receipt", &self.receipt)
			.field("body", &self.body.kind())
			.finish()
	}
}

/// One normalized page of runtime model discovery.
#[derive(Clone, Debug)]
pub struct ModelDiscoveryPage {
	/// Normalized model specifications in provider order.
	pub models:      Vec<ModelSpec>,
	/// Opaque cursor for the next page, when more rows are available.
	pub next_cursor: Option<Str>,
}

/// Closed response body produced by the erased service center.
pub enum AnswerBody {
	/// Canonical chat event stream.
	Chat(ChatStream),
	/// Exact or estimated prompt-token count.
	Tokens(TokenCount),
	/// Token identifier sequence.
	TokenIds(TokenSequence),
	/// Detokenized text with tokenizer provenance.
	Text(DetokenizedText),
	/// Batch of embedding vectors.
	Embeddings(EmbeddingBatch),
	/// Image generation progress and artifacts.
	Images(GenerationStream<ImageArtifact>),
	/// Owned asynchronous video job session with resumable checkpoint and
	/// cancellation.
	Video(GenerationSession<VideoArtifact>),
	/// Encoded audio stream.
	Speech(AudioStream),
	/// Incremental transcript stream.
	Transcript(TranscriptStream),
	/// Owned bidirectional realtime session.
	Realtime(RealtimeSession),
	/// Ranked standalone search results.
	Search(SearchResults),
	/// Account-scoped usage and quota report.
	Usage(UsageReport),
	/// Runtime-discovered normalized model page.
	Models(ModelDiscoveryPage),
	/// Authentication or account-management result.
	Auth(AuthAnswer),
	/// Bounded allowlisted native response.
	Native(NativeResponse),
}

impl AnswerBody {
	/// Returns the body variant without consuming stream or session state.
	pub const fn kind(&self) -> AnswerKind {
		match self {
			Self::Chat(_) => AnswerKind::Chat,
			Self::Tokens(_) => AnswerKind::Tokens,
			Self::TokenIds(_) => AnswerKind::TokenIds,
			Self::Text(_) => AnswerKind::Text,
			Self::Embeddings(_) => AnswerKind::Embeddings,
			Self::Images(_) => AnswerKind::Images,
			Self::Video(_) => AnswerKind::Video,
			Self::Speech(_) => AnswerKind::Speech,
			Self::Transcript(_) => AnswerKind::Transcript,
			Self::Realtime(_) => AnswerKind::Realtime,
			Self::Search(_) => AnswerKind::Search,
			Self::Usage(_) => AnswerKind::Usage,
			Self::Models(_) => AnswerKind::Models,
			Self::Auth(_) => AnswerKind::Auth,
			Self::Native(_) => AnswerKind::Native,
		}
	}
}

/// Discriminant used in structured body-variant mismatch errors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AnswerKind {
	/// Chat stream.
	Chat,
	/// Token count.
	Tokens,
	/// Token sequence.
	TokenIds,
	/// Detokenized text.
	Text,
	/// Embedding batch.
	Embeddings,
	/// Image generation stream.
	Images,
	/// Video generation stream.
	Video,
	/// Speech audio stream.
	Speech,
	/// Transcript stream.
	Transcript,
	/// Realtime session.
	Realtime,
	/// Search results.
	Search,
	/// Usage report.
	Usage,
	/// Discovered models.
	Models,
	/// Authentication result.
	Auth,
	/// Native response.
	Native,
}

/// Provenance of a token-count answer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenizerProvenance {
	/// Stable tokenizer identity.
	pub tokenizer: Str,
	/// Immutable tokenizer revision or digest.
	pub revision:  Str,
	/// Whether the count is exact for the selected wire encoding.
	pub exact:     bool,
}

/// Prompt token count with mandatory provenance.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenCount {
	/// Number of tokens.
	pub tokens:     u64,
	/// Tokenizer or provider provenance.
	pub provenance: TokenizerProvenance,
}

/// Tokenization output with tokenizer provenance.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenSequence {
	/// Ordered token identifiers.
	pub tokens:     Vec<u32>,
	/// Tokenizer provenance.
	pub provenance: TokenizerProvenance,
}

/// Detokenized text with tokenizer identity and immutable revision.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DetokenizedText {
	/// Reconstructed UTF-8 text.
	pub text:       Str,
	/// Tokenizer provenance.
	pub provenance: TokenizerProvenance,
}

/// One embedding vector and its input position.
#[derive(Clone, Debug, PartialEq)]
pub struct Embedding {
	/// Zero-based input index.
	pub index:  u32,
	/// Dense floating-point vector.
	pub values: Vec<f32>,
}

/// Ordered embedding batch with dimensions and usage.
#[derive(Clone, Debug, PartialEq)]
pub struct EmbeddingBatch {
	/// Vector dimensions after any declared adjustment.
	pub dimensions: u32,
	/// One vector per input.
	pub embeddings: Vec<Embedding>,
	/// Resource usage for embedding generation.
	pub usage:      Usage,
}

/// Stable reference to immutable artifact-store content.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactRef {
	/// Artifact store namespace.
	pub store:    Str,
	/// Store-local immutable object identifier.
	pub id:       Str,
	/// Content revision used to guarantee repeatable reads.
	pub revision: Str,
}

/// Digest algorithm attached to generated artifacts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DigestAlgorithm {
	/// SHA-256.
	Sha256,
	/// BLAKE3.
	Blake3,
}

/// Content digest for an artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Digest {
	/// Digest algorithm.
	pub algorithm: DigestAlgorithm,
	/// Raw digest bytes.
	pub value:     Bytes,
}

/// Storage representation for a generated or referenced artifact.
pub enum ArtifactBody {
	/// Small inline immutable bytes.
	Bytes(Bytes),
	/// Owned asynchronous byte stream.
	Stream(ByteStream),
	/// Immutable object in an artifact store.
	Stored(ArtifactRef),
}

impl fmt::Debug for ArtifactBody {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::Bytes(bytes) => formatter.debug_tuple("Bytes").field(&bytes.len()).finish(),
			Self::Stream(_) => formatter.write_str("Stream(..)"),
			Self::Stored(reference) => formatter.debug_tuple("Stored").field(reference).finish(),
		}
	}
}

/// Large or small media artifact with integrity metadata.
#[derive(Debug)]
pub struct Artifact {
	/// MIME media type.
	pub media_type: Str,
	/// Declared size in bytes when known.
	pub size:       Option<u64>,
	/// Optional content digest.
	pub digest:     Option<Digest>,
	/// Owned or stored content body.
	pub body:       ArtifactBody,
}

/// Generated image artifact and dimensions.
#[derive(Debug)]
pub struct ImageArtifact {
	/// Generated content.
	pub artifact:       Artifact,
	/// Width in pixels.
	pub width:          u32,
	/// Height in pixels.
	pub height:         u32,
	/// Revised prompt disclosed by the provider, if any.
	pub revised_prompt: Option<Str>,
}

/// Generated video artifact and media metadata.
#[derive(Debug)]
pub struct VideoArtifact {
	/// Generated content.
	pub artifact:          Artifact,
	/// Width in pixels.
	pub width:             u32,
	/// Height in pixels.
	pub height:            u32,
	/// Duration in milliseconds.
	pub duration_ms:       u64,
	/// Frames per second when known.
	pub frames_per_second: Option<u32>,
}

/// Progress or output event for a long-running media generation.
#[derive(Debug)]
pub enum GenerationEvent<T> {
	/// Provider accepted and queued the job.
	Queued {
		/// Provider job handle.
		job: GenerationHandle,
	},
	/// Monotonic progress counters.
	Progress {
		/// Completed work units.
		completed: u64,
		/// Total work units when known.
		total:     Option<u64>,
	},
	/// Provisional preview that is not a final artifact.
	Preview(T),
	/// Final generated artifact.
	Artifact(T),
	/// Job completed with aggregate accounting.
	Completed(GenerationSummary),
}

/// Owned asynchronous generation job with live events, resume state, and
/// cancellation.
///
/// Dropping the session sends one nonblocking cancellation command to the
/// shared job-controller actor unless explicit cancellation was already
/// requested.
pub struct GenerationSession<T> {
	events:     GenerationStream<T>,
	checkpoint: JobCheckpointHandle,
	cancel:     JobCancelHandle,
}

impl<T> GenerationSession<T> {
	/// Creates a session after verifying that the checkpoint and cancellation
	/// handle identify the same job.
	pub fn new(
		events: GenerationStream<T>,
		checkpoint: JobCheckpointHandle,
		mut cancel: JobCancelHandle,
	) -> Result<Self, GenerationSessionError> {
		if &checkpoint.snapshot().job != cancel.job() {
			cancel.disarm();
			return Err(GenerationSessionError::JobMismatch);
		}
		Ok(Self { events, checkpoint, cancel })
	}

	/// Borrows the stable provider-qualified job identity.
	pub fn job(&self) -> &JobRef {
		self.cancel.job()
	}

	/// Returns a consistent current checkpoint suitable for explicit resume.
	pub fn checkpoint(&self) -> JobCheckpoint {
		self.checkpoint.snapshot()
	}

	/// Borrows the owned event stream for direct stream combinators.
	pub fn events_mut(&mut self) -> &mut GenerationStream<T> {
		&mut self.events
	}

	/// Requests provider cancellation exactly once and waits for typed
	/// acknowledgement evidence.
	pub async fn cancel(&mut self) -> Result<JobCancellationReceipt, JobCancelError> {
		self.cancel.cancel().await
	}

	/// Returns whether this session already dispatched its single cancellation
	/// command.
	pub const fn cancellation_requested(&self) -> bool {
		self.cancel.cancellation_requested()
	}
}

impl<T> fmt::Debug for GenerationSession<T> {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter
			.debug_struct("GenerationSession")
			.field("checkpoint", &self.checkpoint)
			.field("cancellation_requested", &self.cancel.cancellation_requested())
			.finish_non_exhaustive()
	}
}

impl<T> Stream for GenerationSession<T> {
	type Item = Result<GenerationEvent<T>, Error>;

	fn poll_next(
		mut self: Pin<&mut Self>,
		context: &mut std::task::Context<'_>,
	) -> std::task::Poll<Option<Self::Item>> {
		self.events.as_mut().poll_next(context)
	}
}

/// Construction failure for an owned generation session.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GenerationSessionError {
	/// The event checkpoint and cancellation handle identify different jobs.
	JobMismatch,
}

impl fmt::Display for GenerationSessionError {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter.write_str("generation checkpoint and cancellation handle identify different jobs")
	}
}

impl std::error::Error for GenerationSessionError {}

/// Final summary for a long-running generation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenerationSummary {
	/// Number of final artifacts emitted.
	pub artifacts: u32,
	/// Total job duration.
	pub elapsed:   Duration,
	/// Usage across polling and generation.
	pub usage:     Usage,
	/// Integer job cost.
	pub cost:      Cost,
}

/// One encoded output audio chunk.
#[derive(Clone, Debug)]
pub struct AudioChunk {
	/// Encoded immutable bytes.
	pub bytes:       Bytes,
	/// Start timestamp in milliseconds.
	pub start_ms:    Option<u64>,
	/// End timestamp in milliseconds.
	pub end_ms:      Option<u64>,
	/// Whether this is the final audio chunk.
	pub final_chunk: bool,
}

/// One identified speaker in a transcript.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Speaker {
	/// Stable speaker index within the transcript.
	pub index: u32,
	/// Optional provider label.
	pub label: Option<Str>,
}

/// Incremental speech transcript event.
#[derive(Clone, Debug, PartialEq)]
pub enum TranscriptEvent {
	/// A transcript stream was accepted.
	Started {
		/// Detected or requested language.
		language: Option<Str>,
	},
	/// Provisional text that may be superseded before segment finalization.
	TextDelta {
		/// Provisional transcript text.
		text: Str,
	},
	/// Final timestamped segment.
	Segment {
		/// Monotonic segment index.
		index:    u32,
		/// Final segment text.
		text:     Str,
		/// Inclusive start time in milliseconds.
		start_ms: u64,
		/// Exclusive end time in milliseconds.
		end_ms:   u64,
		/// Identified speaker when available.
		speaker:  Option<Speaker>,
	},
	/// Final timestamped word.
	Word {
		/// Final word text.
		text:       Str,
		/// Inclusive start time in milliseconds.
		start_ms:   u64,
		/// Exclusive end time in milliseconds.
		end_ms:     u64,
		/// Provider confidence when exposed.
		confidence: Option<f32>,
		/// Identified speaker when available.
		speaker:    Option<Speaker>,
	},
	/// Final transcript and usage.
	Completed {
		/// Complete transcript text.
		text:  Str,
		/// Final transcription usage.
		usage: Usage,
	},
}

/// Caller-to-provider message in an owned realtime session.
#[derive(Clone, Debug)]
pub enum RealtimeInput {
	/// Append encoded audio bytes.
	Audio(Bytes),
	/// Append user text.
	Text(Str),
	/// Submit typed ordered content for a completed tool call.
	ToolResult {
		/// Stable completed tool-call identity.
		call:     ToolCallId,
		/// Tool name when required by the wire protocol.
		name:     Option<Str>,
		/// Ordered typed result content.
		content:  Arc<[crate::call::ToolResultContent]>,
		/// Whether tool execution failed.
		is_error: bool,
	},
	/// Commit current input and request a response.
	Commit,
	/// Cancel the active response.
	CancelResponse,
	/// Close the session.
	Close,
}

/// Provider-to-caller message from an owned realtime session.
#[derive(Debug)]
pub enum RealtimeEvent {
	/// Session handshake completed.
	Ready,
	/// Canonical chat event.
	Chat(ChatEvent),
	/// Encoded output audio.
	Audio(AudioChunk),
	/// Input audio was committed.
	InputCommitted,
	/// Session closed cleanly.
	Closed,
}

/// Owned bounded bidirectional realtime session.
///
/// Channels are private so all caller input observes backpressure and the
/// single terminal close transition.
pub struct RealtimeSession {
	pub(crate) outbound: flume::Sender<RealtimeInput>,
	pub(crate) inbound:  flume::Receiver<Result<RealtimeEvent, Error>>,
	pub(crate) closed:   Arc<AtomicBool>,
}

impl RealtimeSession {
	/// Creates a session from one bounded channel pair and its shared terminal
	/// state.
	pub(crate) fn from_channels(
		outbound: flume::Sender<RealtimeInput>,
		inbound: flume::Receiver<Result<RealtimeEvent, Error>>,
		closed: Arc<AtomicBool>,
	) -> Self {
		Self { outbound, inbound, closed }
	}

	/// Returns whether close was requested or the session reached terminal
	/// closure.
	pub fn is_closed(&self) -> bool {
		self.closed.load(Ordering::Acquire)
	}
}

impl fmt::Debug for RealtimeSession {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter
			.debug_struct("RealtimeSession")
			.field("closed", &self.is_closed())
			.field("outbound_disconnected", &self.outbound.is_disconnected())
			.field("inbound_disconnected", &self.inbound.is_disconnected())
			.finish()
	}
}

impl Drop for RealtimeSession {
	fn drop(&mut self) {
		if !self.closed.swap(true, Ordering::AcqRel) {
			let _ = self.outbound.try_send(RealtimeInput::Close);
		}
	}
}

/// One ranked standalone search result.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult {
	/// Rank, starting at one.
	pub rank:         u32,
	/// Result URL.
	pub url:          Str,
	/// Result title.
	pub title:        Str,
	/// Provider-produced snippet.
	pub snippet:      Option<Str>,
	/// Relevance score when exposed.
	pub score:        Option<f32>,
	/// Publication time when known.
	pub published_at: Option<SystemTime>,
}

/// Ranked standalone search answer and optional synthesis.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchResults {
	/// Ordered ranked results.
	pub results: Vec<SearchResult>,
	/// Optional provider-generated answer synthesis.
	pub answer:  Option<Str>,
	/// Search resource usage.
	pub usage:   Usage,
}

/// Type of an account usage or quota window.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UsageWindowKind {
	/// Request or token rate limit.
	RateLimit,
	/// Account quota.
	Quota,
	/// Billing-period consumption.
	Billing,
	/// Remaining monetary or unit balance.
	Balance,
}

/// Account-scoped usage or quota window.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UsageWindow {
	/// Window category.
	pub kind:        UsageWindowKind,
	/// Dimension name, such as requests or input tokens.
	pub dimension:   Str,
	/// Consumed integer units when exposed or safely derived.
	pub consumed:    Option<u64>,
	/// Remaining integer units when exposed.
	pub remaining:   Option<u64>,
	/// Total integer limit when exposed.
	pub limit:       Option<u64>,
	/// Window reset time when exposed.
	pub resets_at:   Option<SystemTime>,
	/// Observation provenance.
	pub source:      UsageSource,
	/// Time at which this observation was recorded.
	pub observed_at: SystemTime,
}

/// Account-scoped usage, quota, and balance answer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UsageReport {
	/// Provider to which the report belongs.
	pub provider:  ProviderId,
	/// Account to which the report belongs.
	pub account:   AccountId,
	/// Principal owning account affinity.
	pub principal: Option<PrincipalId>,
	/// Typed windows returned by the provider or local runtime.
	pub windows:   Vec<UsageWindow>,
}

/// Non-secret account metadata safe for display and receipts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AccountSummary {
	/// Opaque account identity.
	pub account:   AccountId,
	/// Provider domain.
	pub provider:  ProviderId,
	/// Authenticated principal identity when known.
	pub principal: Option<PrincipalId>,
	/// Caller-facing label.
	pub label:     Option<Str>,
	/// Current lifecycle state.
	pub state:     AccountState,
}

/// Public lifecycle state of an account.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AccountState {
	/// Account may serve requests.
	Active,
	/// Account must refresh before serving.
	RefreshRequired,
	/// Account is administratively or provider disabled.
	Disabled,
	/// Account credentials were removed.
	LoggedOut,
}

/// Public prompt emitted by an interactive authentication flow.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuthPrompt {
	/// Stable prompt identity.
	pub id:      Str,
	/// Caller-facing prompt text without secret content.
	pub message: Str,
	/// Expected input form.
	pub input:   AuthPromptKind,
}

/// Expected response form for an authentication prompt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AuthPromptKind {
	/// OAuth authorization code.
	AuthorizationCode,
	/// Static API key.
	ApiKey,
	/// Provider session token.
	SessionToken,
	/// Yes-or-no confirmation.
	Confirmation,
}

/// Authentication flow event; secret-bearing variants deliberately omit
/// `Debug`.
pub enum AuthEvent {
	/// Open a browser at the public authorization URL.
	OpenUrl(Str),
	/// Display a short-lived secret device code and public verification URL.
	ShowDeviceCode {
		/// Secret short-lived device code.
		code:             SecretString,
		/// Public verification URL.
		verification_url: Str,
	},
	/// Ask the caller for typed input.
	Prompt(AuthPrompt),
	/// The provider is waiting for external completion.
	Waiting,
	/// Authentication completed with a non-secret account summary.
	Complete(AccountSummary),
}

/// Caller response routed back to an interactive authentication flow.
pub struct AuthResponse {
	/// Login session receiving the response.
	pub session: LoginSessionId,
	/// Secret or control input.
	pub input:   crate::call::AuthInput,
}

/// Owned channels for an interactive authentication flow.
pub struct AuthSession {
	/// Stable login session identity.
	pub id:        LoginSessionId,
	/// Stream-like channel of authentication events or structured errors.
	pub events:    flume::Receiver<Result<AuthEvent, Error>>,
	/// Response channel back to the authentication engine.
	pub responses: flume::Sender<AuthResponse>,
}

/// Authentication or account-management operation answer.
pub enum AuthAnswer {
	/// Newly started or resumed interactive login session.
	Session(AuthSession),
	/// Non-secret account listing.
	Accounts(Vec<AccountSummary>),
	/// Refreshed account metadata.
	Refreshed(AccountSummary),
	/// Account removed from active use.
	LoggedOut(AccountId),
	/// Input was accepted and flow progress continues on the session channel.
	Submitted(LoginSessionId),
}

/// Bounded native response body.
pub enum NativeResponseBody {
	/// Exact validated JSON response bytes preserved without reserialization.
	Json(crate::call::RawJson),
	/// Immutable binary response.
	Bytes(Bytes),
	/// Incremental opaque response bytes, including native SSE framing.
	Stream(ByteStream),
}

impl fmt::Debug for NativeResponseBody {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::Json(value) => formatter
				.debug_tuple("Json")
				.field(&value.as_bytes().len())
				.finish(),
			Self::Bytes(bytes) => formatter.debug_tuple("Bytes").field(&bytes.len()).finish(),
			Self::Stream(_) => formatter.write_str("Stream(..)"),
		}
	}
}

/// Response from one allowlisted native wire operation.
#[derive(Debug)]
pub struct NativeResponse {
	/// HTTP-like response status.
	pub status:              u16,
	/// Response media type when supplied.
	pub media_type:          Option<Str>,
	/// Bounded response body.
	pub body:                NativeResponseBody,
	/// Sanitized provider request identifier.
	pub provider_request_id: Option<Str>,
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{catalog::OperationKind, operation::job::JobCommand};

	fn job(handle: &'static str) -> JobRef {
		JobRef {
			provider:  ProviderId::from("provider"),
			route:     RouteId::from("route"),
			operation: OperationKind::GenerateVideo,
			handle:    GenerationHandle::from(handle),
		}
	}

	fn checkpoint(job: JobRef) -> JobCheckpointHandle {
		JobCheckpointHandle::new(JobCheckpoint {
			job,
			completed: 0,
			total: None,
			polls: 0,
			expires_at: None,
			created_at: SystemTime::UNIX_EPOCH,
		})
	}

	#[test]
	fn generation_session_rejects_mismatched_job_authority() {
		let (cancel, commands) =
			JobCancelHandle::bounded(job("cancel"), 1).expect("bounded cancellation");
		let result = GenerationSession::<VideoArtifact>::new(
			Box::pin(futures::stream::empty()),
			checkpoint(job("checkpoint")),
			cancel,
		);
		assert!(matches!(result, Err(GenerationSessionError::JobMismatch)));
		assert!(commands.try_recv().is_err(), "rejected ownership must not cancel another job");
	}

	#[test]
	fn dropping_generation_session_dispatches_one_nonblocking_cancel() {
		let job = job("video");
		let (cancel, commands) =
			JobCancelHandle::bounded(job.clone(), 1).expect("bounded cancellation");
		let session = GenerationSession::<VideoArtifact>::new(
			Box::pin(futures::stream::empty()),
			checkpoint(job),
			cancel,
		)
		.expect("matching session");
		drop(session);
		assert!(matches!(commands.try_recv(), Ok(JobCommand::Cancel { acknowledgement: None })));
		assert!(commands.try_recv().is_err());
	}
}
