//! Native OMP federation client.
//!
//! This client speaks `omp.inference.v1` to an upstream gateway. Credentials
//! therefore remain on the gateway that owns them: a remote agent can traverse
//! a remote daemon and a home gateway without credential bytes crossing that
//! boundary.
//!
//! Context is deliberately not cached here. Context state belongs to the
//! terminal gateway, and protocol errors are converted without
//! reinterpretation; notably an upstream `NEED_FULL` remains `NEED_FULL` so the
//! downstream caller knows to reseed the terminal gateway.

use std::sync::Arc;

use async_trait::async_trait;
use futures::{StreamExt, stream::BoxStream};
use omp_core::Str;
use omp_llm_catalog::TransportId;
use omp_llm_types::{
	Chat, ChatRequest, CountRequest, CountResponse, CountTokens, Embed, EmbedRequest, EmbedResponse,
	Error, Executor, Invoke, InvokeInput, Search, SearchRequest, SearchResponse, Speak, SpeakEvent,
	SpeakRequest, Transcribe, TranscribeRequest, TranscribeResponse, TurnEvent,
};
use omp_proto::inference::v1::{self as pb, inference_client::InferenceClient};
use tonic::transport::Channel;

/// Client for a federated upstream OMP gateway.
#[derive(Clone)]
pub struct OmpFederation {
	client: InferenceClient<Channel>,
}

impl OmpFederation {
	/// Wraps an established generated gRPC client.
	#[must_use]
	pub const fn new(client: InferenceClient<Channel>) -> Self {
		Self { client }
	}

	/// Returns the catalog transport spoken by this client.
	#[must_use]
	pub const fn id(&self) -> TransportId {
		TransportId::Omp
	}

	/// Connects to an upstream gateway endpoint.
	pub async fn connect<D>(endpoint: D) -> Result<Self, Error>
	where
		D: TryInto<tonic::transport::Endpoint>,
		D::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
	{
		InferenceClient::connect(endpoint)
			.await
			.map(Self::new)
			.map_err(transport)
	}
}

fn transport(_error: impl std::fmt::Display) -> Error {
	// Upstream status messages and conversion diagnostics are untrusted text.
	// They may echo provider headers, query parameters, or response bodies, so
	// only the stable transport classification crosses the federation boundary.
	Error::Transport(Str::new_static("upstream transport failed"))
}

fn request_stream(
	receiver: flume::Receiver<pb::TurnFrame>,
) -> impl futures::Stream<Item = pb::TurnFrame> + Send + 'static {
	futures::stream::unfold(receiver, |receiver| async move {
		receiver
			.recv_async()
			.await
			.ok()
			.map(|frame| (frame, receiver))
	})
}

#[async_trait]
impl Chat for OmpFederation {
	async fn turn(
		&self,
		request: ChatRequest,
		executor: Option<Arc<dyn Executor>>,
	) -> Result<BoxStream<'static, TurnEvent>, Error> {
		let executor_tools = executor.as_ref().map(|_| {
			request
				.tools
				.iter()
				.map(|tool| tool.name.to_string())
				.collect()
		});
		let mut open: pb::TurnRequest = request.into();
		open.turn_id = ulid::Ulid::generate().to_string();
		open.executor = executor_tools.map(|tools| pb::Executor { tools });

		let (frames_tx, frames_rx) = flume::bounded(16);
		frames_tx
			.send_async(pb::TurnFrame { frame: Some(pb::turn_frame::Frame::Open(open)) })
			.await
			.map_err(|_| Error::Transport(Str::new("failed to open upstream turn stream")))?;
		let interactive_tx = executor.as_ref().map(|_| frames_tx.clone());
		drop(frames_tx); // Half-close before awaiting response headers when non-interactive.

		let mut client = self.client.clone();
		let response = client
			.turn(request_stream(frames_rx))
			.await
			.map_err(transport)?;
		let mut upstream = response.into_inner();

		let events = async_stream::stream! {
			while let Some(wire) = upstream.next().await {
				let Ok(wire) = wire else {
					break;
				};
				let event: TurnEvent = match wire.try_into() {
					Ok(event) => event,
					Err(_) => break,
				};

				if let (TurnEvent::Invoke(invocation), Some(executor), Some(frames_tx)) =
					(&event, executor.as_ref(), interactive_tx.as_ref())
				{
					spawn_invocation(invocation.clone(), Arc::clone(executor), frames_tx.clone());
				}
				yield event;
			}
		};
		Ok(Box::pin(events))
	}
}

fn spawn_invocation(
	invocation: Invoke,
	executor: Arc<dyn Executor>,
	frames: flume::Sender<pb::TurnFrame>,
) {
	tokio::spawn(async move {
		let (inputs_tx, inputs_rx) = flume::bounded::<InvokeInput>(16);
		let input_frames = frames.clone();
		let forward = tokio::spawn(async move {
			while let Ok(input) = inputs_rx.recv_async().await {
				if input_frames
					.send_async(pb::TurnFrame {
						frame: Some(pb::turn_frame::Frame::Input(input.into())),
					})
					.await
					.is_err()
				{
					break;
				}
			}
		});
		let complete = executor.invoke(invocation, inputs_tx).await;
		let _ = forward.await;
		let _ = frames
			.send_async(pb::TurnFrame {
				frame: Some(pb::turn_frame::Frame::Complete(complete.into())),
			})
			.await;
	});
}

#[async_trait]
impl CountTokens for OmpFederation {
	async fn count(&self, request: CountRequest) -> Result<CountResponse, Error> {
		let mut client = self.client.clone();
		client
			.count_tokens(pb::CountTokensRequest::from(request))
			.await
			.map_err(transport)?
			.into_inner()
			.try_into()
			.map_err(transport)
	}
}

#[async_trait]
impl Embed for OmpFederation {
	async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse, Error> {
		let mut client = self.client.clone();
		client
			.embed(pb::EmbedRequest::from(request))
			.await
			.map_err(transport)?
			.into_inner()
			.try_into()
			.map_err(transport)
	}
}

#[async_trait]
impl Speak for OmpFederation {
	async fn speak(&self, request: SpeakRequest) -> Result<BoxStream<'static, SpeakEvent>, Error> {
		let mut client = self.client.clone();
		let mut upstream = client
			.speak(pb::SpeakRequest::from(request))
			.await
			.map_err(transport)?
			.into_inner();
		let stream = async_stream::stream! {
			while let Some(Ok(event)) = upstream.next().await {
				if let Ok(event) = event.try_into() {
					yield event;
				} else {
					break;
				}
			}
		};
		Ok(Box::pin(stream))
	}
}

#[async_trait]
impl Transcribe for OmpFederation {
	async fn transcribe(&self, request: TranscribeRequest) -> Result<TranscribeResponse, Error> {
		let mut client = self.client.clone();
		client
			.transcribe(pb::TranscribeRequest::from(request))
			.await
			.map_err(transport)?
			.into_inner()
			.try_into()
			.map_err(transport)
	}
}

#[async_trait]
impl Search for OmpFederation {
	async fn search(&self, request: SearchRequest) -> Result<SearchResponse, Error> {
		let mut client = self.client.clone();
		client
			.search(pb::SearchRequest::from(request))
			.await
			.map_err(transport)?
			.into_inner()
			.try_into()
			.map_err(transport)
	}
}

#[cfg(test)]
mod tests {
	use omp_llm_types::{Props, Thread, TurnError, TurnErrorKind};

	use super::*;

	#[test]
	fn stateless_turn_round_trips_through_canonical_conversions() {
		let original = ChatRequest::builder()
			.model(Str::new("slow"))
			.thread(Thread::default())
			.tools(Vec::new())
			.provider_options(Props::default())
			.build();
		let wire: pb::TurnRequest = original.clone().into();
		let decoded = ChatRequest::try_from(wire).expect("valid stateless turn");
		assert_eq!(decoded, original);
	}

	#[test]
	fn upstream_error_kind_round_trips_unchanged() {
		let original = TurnEvent::Error(
			TurnError::builder()
				.kind(TurnErrorKind::NeedFull)
				.detail(Str::new("terminal gateway needs a seed"))
				.unsupported(Vec::new())
				.retry_after_ms(0)
				.build(),
		);
		let wire: pb::TurnEvent = original.clone().into();
		let decoded = TurnEvent::try_from(wire).expect("valid canonical event");
		assert_eq!(decoded, original);
	}

	#[test]
	fn upstream_transport_diagnostics_never_cross_federation() {
		const CANARY: &str = "canary-signed-authorization";
		let error = transport(std::io::Error::other(CANARY));
		assert_eq!(error.to_string(), "transport failure: upstream transport failed");
		assert!(!error.to_string().contains(CANARY));
		assert!(!format!("{error:?}").contains(CANARY));
	}
}
