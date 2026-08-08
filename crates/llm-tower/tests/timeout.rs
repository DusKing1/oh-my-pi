//! Phase-deadline acceptance tests.

use std::{
	convert::Infallible,
	sync::Arc,
	task::{Context, Poll},
	time::Duration,
};

use futures::{StreamExt, channel::mpsc, stream};
use omp_llm_error::Kind;
use omp_llm_tower::{
	recovery::classify_turn_error,
	testing::{kind_of, outcome, part_delta, part_start},
	timeout::{PhaseTimeout, PhaseTimeoutConfig},
};
use omp_proto::inference::v1::{TurnError, TurnEvent, TurnRequest, turn_event};
use parking_lot::Mutex;
use tokio::time::{advance, pause, sleep};
use tower::{Service, ServiceExt};

#[derive(Clone)]
struct OneStream {
	stream: Arc<Mutex<Option<mpsc::UnboundedReceiver<TurnEvent>>>>,
}

impl OneStream {
	fn new() -> (Self, mpsc::UnboundedSender<TurnEvent>) {
		let (sender, receiver) = mpsc::unbounded();
		(Self { stream: Arc::new(Mutex::new(Some(receiver))) }, sender)
	}
}

impl Service<TurnRequest> for OneStream {
	type Error = Infallible;
	type Future = std::future::Ready<Result<mpsc::UnboundedReceiver<TurnEvent>, Infallible>>;
	type Response = mpsc::UnboundedReceiver<TurnEvent>;

	fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		Poll::Ready(Ok(()))
	}

	fn call(&mut self, _req: TurnRequest) -> Self::Future {
		let receiver = self.stream.lock().take().unwrap_or_else(|| {
			let (_sender, receiver) = mpsc::unbounded();
			receiver
		});
		std::future::ready(Ok(receiver))
	}
}

const fn config() -> PhaseTimeoutConfig {
	PhaseTimeoutConfig { call_ms: 10, first_event_ms: 10, idle_ms: 10 }
}

fn error_from(frame: TurnEvent) -> TurnError {
	match frame.event {
		Some(turn_event::Event::Error(error)) => error,
		other => panic!("expected error frame, got {other:?}"),
	}
}

#[tokio::test]
async fn call_future_timeout_is_connect_error_and_classifies_as_timeout() {
	pause();
	let inner = tower::service_fn(|_req: TurnRequest| async {
		sleep(Duration::from_millis(50)).await;
		Ok::<_, Infallible>(stream::empty::<TurnEvent>())
	});
	let service = PhaseTimeout::new(inner, config());
	let response = tokio::spawn(service.oneshot(TurnRequest::default()));
	tokio::task::yield_now().await;

	advance(Duration::from_millis(11)).await;
	let mut response = Box::pin(
		response
			.await
			.expect("call task panicked")
			.expect("infallible service"),
	);
	let error = error_from(response.next().await.expect("deadline frame"));

	assert_eq!(error.detail, "connect deadline: timed out after 10ms");
	assert!(response.next().await.is_none(), "deadline must be terminal");
	let classification = classify_turn_error(&error);
	assert!(classification.kinds.has(Kind::Timeout));
	assert!(classification.kinds.has(Kind::Transient));
}

#[tokio::test]
async fn silence_before_first_frame_is_first_event_error() {
	pause();
	let (inner, _sender) = OneStream::new();
	let mut response = Box::pin(
		PhaseTimeout::new(inner, config())
			.oneshot(TurnRequest::default())
			.await
			.expect("infallible service"),
	);
	let waiting = tokio::spawn(async move {
		let deadline = response.next().await;
		let eof = response.next().await;
		(deadline, eof)
	});
	tokio::task::yield_now().await;

	advance(Duration::from_millis(11)).await;
	let (deadline, eof) = waiting.await.expect("stream task panicked");
	let error = error_from(deadline.expect("deadline frame"));
	assert_eq!(error.detail, "first-event deadline: timed out after 10ms");
	assert!(eof.is_none(), "deadline must be terminal");
}

#[tokio::test]
async fn gap_after_a_frame_is_idle_error() {
	pause();
	let (inner, sender) = OneStream::new();
	sender.unbounded_send(part_start()).expect("receiver alive");
	let mut response = Box::pin(
		PhaseTimeout::new(inner, config())
			.oneshot(TurnRequest::default())
			.await
			.expect("infallible service"),
	);
	assert_eq!(kind_of(&response.next().await.expect("first frame")), "part_start");
	let waiting = tokio::spawn(async move {
		let deadline = response.next().await;
		let eof = response.next().await;
		(deadline, eof)
	});
	tokio::task::yield_now().await;

	advance(Duration::from_millis(11)).await;
	let (deadline, eof) = waiting.await.expect("stream task panicked");
	let error = error_from(deadline.expect("deadline frame"));
	assert_eq!(error.detail, "idle deadline: timed out after 10ms");
	assert!(eof.is_none(), "deadline must be terminal");
}

#[tokio::test]
async fn many_timely_frames_have_no_total_deadline() {
	pause();
	let (inner, sender) = OneStream::new();
	let mut response = Box::pin(
		PhaseTimeout::new(inner, config())
			.oneshot(TurnRequest::default())
			.await
			.expect("infallible service"),
	);

	for _ in 0..32 {
		let waiting = tokio::spawn(async move {
			let frame = response.next().await;
			(response, frame)
		});
		tokio::task::yield_now().await;
		advance(Duration::from_millis(9)).await;
		sender.unbounded_send(part_delta()).expect("receiver alive");
		let (returned, frame) = waiting.await.expect("stream task panicked");
		response = returned;
		assert_eq!(kind_of(&frame.expect("timely frame")), "part_delta");
	}

	let waiting = tokio::spawn(async move {
		let frame = response.next().await;
		(response, frame)
	});
	tokio::task::yield_now().await;
	advance(Duration::from_millis(9)).await;
	sender.unbounded_send(outcome()).expect("receiver alive");
	let (mut response, frame) = waiting.await.expect("stream task panicked");
	assert_eq!(kind_of(&frame.expect("timely outcome")), "outcome");

	drop(sender);
	assert!(response.next().await.is_none(), "clean EOF must pass through");
}
