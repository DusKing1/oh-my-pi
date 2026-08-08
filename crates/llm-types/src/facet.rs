//! Object-safe inference capabilities and their native request contracts.
//!
//! These traits are intentionally erased behind [`std::sync::Arc`] in
//! [`Facets`]: each async call is a cold boundary dominated by provider I/O.
//! `async_trait` therefore boxes once per remote operation, and streaming
//! methods return one [`BoxStream`] per operation, never per event or chunk.

use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use omp_core::SmolStr;

use crate::{
	ChatRequest, CountRequest, CountResponse, EmbedRequest, EmbedResponse, GenerateImageRequest,
	GenerateVideoRequest, GenerationStatus, ImageEvent, Invoke, InvokeComplete, InvokeInput,
	SearchRequest, SearchResponse, SpeakEvent, SpeakRequest, TranscribeRequest, TranscribeResponse,
	TurnEvent, Unsupported,
};

/// An inference operation independently advertised by a provider.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum Facet {
	/// Streaming conversational inference.
	Chat,
	/// Prompt token counting.
	CountTokens,
	/// Text embeddings.
	Embed,
	/// Image generation.
	ImageGen,
	/// Speech synthesis.
	Speak,
	/// Audio transcription.
	Transcribe,
	/// Asynchronous video generation.
	VideoGen,
	/// Web search.
	Search,
	/// Account quota inspection.
	Quota,
}

/// Failure to admit or execute a native facet request.
///
/// Provider response classification remains in `omp_llm_error`; that crate
/// intentionally exposes evidence and policy rather than an owning error type.
// TODO(errors): integrate a future owning provider error from `omp_llm_error`.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
	/// The selected provider path cannot honor required capabilities.
	#[error("unsupported inference request")]
	Unsupported(Vec<Unsupported>),
	/// The provider rejected or failed an admitted operation.
	#[error("provider failure: {0}")]
	Provider(SmolStr),
	/// The upstream transport failed.
	#[error("transport failure: {0}")]
	Transport(SmolStr),
}

/// Executes tools that a transport must answer while a turn remains open.
///
/// The executor receives the fully materialized invocation and a channel for
/// streaming zero or more [`InvokeInput`] frames back to the transport. Its
/// return value is the single terminal completion frame. Transports such as
/// Cursor that require this facility fail admission with [`Error::Unsupported`]
/// when it is absent. Normal chat transports never call it: a tool call ends
/// the turn and the agent supplies its result in a later turn.
#[async_trait]
pub trait Executor: Send + Sync {
	/// Executes one provider-initiated invocation.
	async fn invoke(&self, invocation: Invoke, inputs: flume::Sender<InvokeInput>)
	-> InvokeComplete;
}

/// Streaming conversational inference.
///
/// Dropping the returned stream structurally aborts upstream work; there is no
/// cooperative cancellation token that an implementation may ignore.
#[async_trait]
pub trait Chat: Send + Sync {
	/// Starts one model turn after admission checks have completed.
	async fn turn(
		&self,
		request: ChatRequest,
		executor: Option<Arc<dyn Executor>>,
	) -> Result<BoxStream<'static, TurnEvent>, Error>;
}

/// Counts prompt tokens without running a model turn.
#[async_trait]
pub trait CountTokens: Send + Sync {
	/// Counts the complete request, including tool definitions.
	async fn count(&self, request: CountRequest) -> Result<CountResponse, Error>;
}

/// Produces vector embeddings for text inputs.
#[async_trait]
pub trait Embed: Send + Sync {
	/// Embeds every input in request order.
	async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse, Error>;
}

/// Streaming image generation.
///
/// Dropping the stream structurally aborts the provider request.
#[async_trait]
pub trait ImageGen: Send + Sync {
	/// Starts generation and yields progressive previews followed by completion.
	async fn generate(
		&self,
		request: GenerateImageRequest,
	) -> Result<BoxStream<'static, ImageEvent>, Error>;
}

/// Streaming speech synthesis.
///
/// Dropping the stream structurally aborts synthesis.
#[async_trait]
pub trait Speak: Send + Sync {
	/// Synthesizes speech as ordered encoded chunks.
	async fn speak(&self, request: SpeakRequest) -> Result<BoxStream<'static, SpeakEvent>, Error>;
}

/// Unary transcription of a recorded blob.
#[async_trait]
pub trait Transcribe: Send + Sync {
	/// Transcribes one complete recording.
	async fn transcribe(&self, request: TranscribeRequest) -> Result<TranscribeResponse, Error>;
}

/// Stable gateway handle for a long-running generation.
#[derive(Clone, Debug, Eq, Hash, PartialEq, bon::Builder)]
#[non_exhaustive]
pub struct GenerationHandle {
	/// Gateway-scoped generation identifier.
	pub id: SmolStr,
}

/// Job-shaped video generation.
///
/// Every surveyed provider uses asynchronous submit/poll: renders take roughly
/// 20 seconds to five minutes, and returned artifact URLs expire. The gateway
/// therefore returns a stable handle immediately and ingests artifacts before
/// expiry. Dropping an [`Self::attach`] stream only detaches the observer; it
/// does not cancel the durable job. Cancellation is explicit.
#[async_trait]
pub trait VideoGen: Send + Sync {
	/// Submits a render and returns immediately after durable admission.
	async fn submit(&self, request: GenerateVideoRequest) -> Result<GenerationHandle, Error>;
	/// Fetches the latest lifecycle snapshot.
	async fn get(&self, handle: GenerationHandle) -> Result<GenerationStatus, Error>;
	/// Attaches to lifecycle changes for an existing render.
	async fn attach(
		&self,
		handle: GenerationHandle,
	) -> Result<BoxStream<'static, GenerationStatus>, Error>;
	/// Explicitly cancels an existing render.
	async fn cancel(&self, handle: GenerationHandle) -> Result<GenerationStatus, Error>;
}

/// Unary web search.
#[async_trait]
pub trait Search: Send + Sync {
	/// Executes one search, including server-side engine fallback.
	async fn search(&self, request: SearchRequest) -> Result<SearchResponse, Error>;
}

/// Request for provider account usage windows.
#[derive(Clone, Debug, bon::Builder)]
#[non_exhaustive]
pub struct QuotaRequest {
	/// Provider whose credential quota should be inspected.
	pub provider: SmolStr,
}

/// One bounded provider usage window.
#[derive(Clone, Debug, bon::Builder)]
#[non_exhaustive]
pub struct QuotaWindow {
	/// Provider-defined window name.
	pub name:         SmolStr,
	/// Consumed units.
	pub used:         u64,
	/// Maximum units, when the provider discloses one.
	pub limit:        Option<u64>,
	/// Unix millisecond at which the window resets, when known.
	pub resets_at_ms: Option<u64>,
}

/// Provider quota snapshot.
#[derive(Clone, Debug, bon::Builder)]
#[non_exhaustive]
pub struct QuotaResponse {
	/// Reported usage windows.
	#[builder(default)]
	pub windows: Vec<QuotaWindow>,
}

/// Unary provider account quota inspection.
#[async_trait]
pub trait Quota: Send + Sync {
	/// Reads the current usage windows for the selected provider account.
	async fn quota(&self, request: QuotaRequest) -> Result<QuotaResponse, Error>;
}

/// Runtime registry of independently available inference facets.
#[derive(Clone, Default)]
pub struct Facets {
	/// Conversational inference implementation.
	pub chat:         Option<Arc<dyn Chat>>,
	/// Token counting implementation.
	pub count_tokens: Option<Arc<dyn CountTokens>>,
	/// Embedding implementation.
	pub embed:        Option<Arc<dyn Embed>>,
	/// Image-generation implementation.
	pub image_gen:    Option<Arc<dyn ImageGen>>,
	/// Speech-synthesis implementation.
	pub speak:        Option<Arc<dyn Speak>>,
	/// Transcription implementation.
	pub transcribe:   Option<Arc<dyn Transcribe>>,
	/// Video-generation implementation.
	pub video_gen:    Option<Arc<dyn VideoGen>>,
	/// Search implementation.
	pub search:       Option<Arc<dyn Search>>,
	/// Quota implementation.
	pub quota:        Option<Arc<dyn Quota>>,
}

impl Facets {
	/// Reports whether the named facet has an implementation.
	#[must_use]
	pub fn supports(&self, facet: Facet) -> bool {
		match facet {
			Facet::Chat => self.chat.is_some(),
			Facet::CountTokens => self.count_tokens.is_some(),
			Facet::Embed => self.embed.is_some(),
			Facet::ImageGen => self.image_gen.is_some(),
			Facet::Speak => self.speak.is_some(),
			Facet::Transcribe => self.transcribe.is_some(),
			Facet::VideoGen => self.video_gen.is_some(),
			Facet::Search => self.search.is_some(),
			Facet::Quota => self.quota.is_some(),
		}
	}
}

/// Recorder for feature degradation performed by a transport encoder.
///
/// The law is simple: nothing drops silently. Every ignored, emulated, or
/// clamped request feature must add exactly one record, allowing every encoder
/// to return `(wire_body, sink.into_vec())` with the same honest shape.
#[derive(Clone, Debug, Default)]
pub struct UnsupportedSink {
	records: Vec<Unsupported>,
}

impl UnsupportedSink {
	/// Creates an empty recorder.
	#[must_use]
	pub const fn new() -> Self {
		Self { records: Vec::new() }
	}

	/// Records a feature omitted under its fallback policy.
	pub fn drop_feature(&mut self, what: impl Into<SmolStr>, detail: impl Into<SmolStr>) {
		self.record(what, detail, crate::UnsupportedAction::Dropped);
	}

	/// Records a feature approximated by a softer strategy.
	pub fn emulate(&mut self, what: impl Into<SmolStr>, detail: impl Into<SmolStr>) {
		self.record(what, detail, crate::UnsupportedAction::Emulated);
	}

	/// Records a requested value constrained to provider limits.
	pub fn clamp(&mut self, what: impl Into<SmolStr>, detail: impl Into<SmolStr>) {
		self.record(what, detail, crate::UnsupportedAction::Clamped);
	}

	/// Consumes the sink and returns records in codec discovery order.
	#[must_use]
	pub fn into_vec(self) -> Vec<Unsupported> {
		self.records
	}

	fn record(
		&mut self,
		what: impl Into<SmolStr>,
		detail: impl Into<SmolStr>,
		action: crate::UnsupportedAction,
	) {
		self
			.records
			.push(Unsupported { what: what.into(), detail: detail.into(), action });
	}
}
