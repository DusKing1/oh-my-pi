//! In-process inference bridge for local model engines.
//!
//! Local models are represented as a transport rather than a gateway special
//! case. Consequently role routing (`tiny`, `smol`, and `slow`), capability
//! admission, metering, and provider selection remain identical for local and
//! remote provider entries; a local model is simply another provider entry.

use std::{future::Future, sync::Arc};

use async_trait::async_trait;
use futures::stream::BoxStream;
use omp_llm_catalog::TransportId;
use omp_llm_types::{
	Chat, ChatRequest, Embed, EmbedRequest, EmbedResponse, Error, Executor, Speak, SpeakEvent,
	SpeakRequest, Transcribe, TranscribeRequest, TranscribeResponse, TurnEvent,
};

/// Narrow interface implemented by an in-process local inference runtime.
///
/// Keeping this interface in the transport crate prevents provider routing from
/// depending on a particular local runtime implementation.
pub trait LocalEngine: Send + Sync + 'static {
	/// Runs a complete streaming chat turn.
	fn chat(
		&self,
		request: ChatRequest,
		executor: Option<Arc<dyn Executor>>,
	) -> impl Future<Output = Result<BoxStream<'static, TurnEvent>, Error>> + Send + '_;

	/// Embeds every input in request order.
	fn embed(
		&self,
		request: EmbedRequest,
	) -> impl Future<Output = Result<EmbedResponse, Error>> + Send + '_;

	/// Synthesizes speech into ordered encoded chunks.
	fn speak(
		&self,
		request: SpeakRequest,
	) -> impl Future<Output = Result<BoxStream<'static, SpeakEvent>, Error>> + Send + '_;

	/// Transcribes one complete recording.
	fn transcribe(
		&self,
		request: TranscribeRequest,
	) -> impl Future<Output = Result<TranscribeResponse, Error>> + Send + '_;
}

/// In-process transport backed by a local inference engine.
#[derive(Clone)]
pub struct Embedded<E> {
	engine: Arc<E>,
}

impl<E> Embedded<E> {
	/// Creates an embedded transport backed by `engine`.
	#[must_use]
	pub const fn new(engine: Arc<E>) -> Self {
		Self { engine }
	}

	/// Returns the catalog transport selected by this bridge.
	#[must_use]
	pub const fn id(&self) -> TransportId {
		TransportId::Embedded
	}

	/// Returns the backing engine.
	#[must_use]
	pub const fn engine(&self) -> &Arc<E> {
		&self.engine
	}
}

#[async_trait]
impl<E: LocalEngine> Chat for Embedded<E> {
	async fn turn(
		&self,
		request: ChatRequest,
		executor: Option<Arc<dyn Executor>>,
	) -> Result<BoxStream<'static, TurnEvent>, Error> {
		self.engine.chat(request, executor).await
	}
}

#[async_trait]
impl<E: LocalEngine> Embed for Embedded<E> {
	async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse, Error> {
		self.engine.embed(request).await
	}
}

#[async_trait]
impl<E: LocalEngine> Speak for Embedded<E> {
	async fn speak(&self, request: SpeakRequest) -> Result<BoxStream<'static, SpeakEvent>, Error> {
		self.engine.speak(request).await
	}
}

#[async_trait]
impl<E: LocalEngine> Transcribe for Embedded<E> {
	async fn transcribe(&self, request: TranscribeRequest) -> Result<TranscribeResponse, Error> {
		self.engine.transcribe(request).await
	}
}
