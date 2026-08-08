//! Semantic re-sampling acceptance tests.

use std::{
	convert::Infallible,
	task::{Context, Poll},
};

use futures::StreamExt;
use omp_llm_tower::{
	resample::{AttemptEvent, AttemptStream, Resample, ResampleConfig},
	testing::{Script, error, ev, kind_of, outcome, part_delta},
};
use omp_proto::inference::v1::{Outcome, TurnRequest, turn_error, turn_event};
use tower::{Service, ServiceExt};

/// Marks the scripted stream as pre-commit — the assertion [`Resample`]'s
/// frame type exists to demand.
#[derive(Clone)]
struct Scripted(Script);

impl Service<TurnRequest> for Scripted {
	type Error = Infallible;
	type Future = std::future::Ready<Result<AttemptStream, Infallible>>;
	type Response = AttemptStream;

	fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Infallible>> {
		Poll::Ready(Ok(()))
	}

	fn call(&mut self, req: TurnRequest) -> Self::Future {
		let stream = self.0.call(req).into_inner().unwrap();
		std::future::ready(Ok(
			// Scripted streams are synthetic: nothing here is committed.
			Box::pin(stream.map(AttemptEvent::new)) as AttemptStream,
		))
	}
}

const fn config(max_attempts: u32) -> ResampleConfig {
	ResampleConfig { max_attempts, base_delay_ms: 0 }
}

fn real_outcome() -> omp_proto::inference::v1::TurnEvent {
	ev(turn_event::Event::Outcome(Outcome {
		output: vec![Default::default()],
		..Outcome::default()
	}))
}

#[tokio::test]
async fn empty_completion_is_suppressed_and_redispatched() {
	let script = Script::new([vec![outcome()], vec![real_outcome()]]);
	let calls = script.calls.clone();
	let stream = Resample::new(Scripted(script), config(3))
		.oneshot(TurnRequest::default())
		.await
		.unwrap();
	let frames = stream
		.map(AttemptEvent::into_inner)
		.collect::<Vec<_>>()
		.await;

	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["attempt", "outcome"]);
	assert_eq!(calls.lock().len(), 2);
	match frames[0].event.as_ref() {
		Some(turn_event::Event::Attempt(attempt)) => {
			assert_eq!(attempt.number, 2);
			assert_eq!(attempt.reason, "empty completion");
		},
		_ => panic!("expected attempt frame"),
	}
}

#[tokio::test]
async fn empty_completion_cap_forwards_last_outcome() {
	let script = Script::new([vec![outcome()], vec![outcome()], vec![outcome()]]);
	let calls = script.calls.clone();
	let stream = Resample::new(Scripted(script), config(2))
		.oneshot(TurnRequest::default())
		.await
		.unwrap();
	let frames = stream
		.map(AttemptEvent::into_inner)
		.collect::<Vec<_>>()
		.await;

	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["attempt", "attempt", "outcome"]);
	assert!(
		matches!(frames.last().and_then(|frame| frame.event.as_ref()), Some(turn_event::Event::Outcome(outcome)) if outcome.output.is_empty())
	);
	assert_eq!(calls.lock().len(), 3);
}

#[tokio::test]
async fn part_output_bars_empty_completion_resampling() {
	let script = Script::new([vec![part_delta(), outcome()], vec![real_outcome()]]);
	let calls = script.calls.clone();
	let stream = Resample::new(Scripted(script), config(3))
		.oneshot(TurnRequest::default())
		.await
		.unwrap();
	let frames = stream
		.map(AttemptEvent::into_inner)
		.collect::<Vec<_>>()
		.await;

	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["part_delta", "outcome"]);
	assert_eq!(calls.lock().len(), 1);
}

#[tokio::test]
async fn thinking_loop_is_suppressed_and_redispatched() {
	let script = Script::new([
		vec![error(turn_error::Kind::Upstream, "Thinking loop detected")],
		vec![real_outcome()],
	]);
	let stream = Resample::new(Scripted(script), config(2))
		.oneshot(TurnRequest::default())
		.await
		.unwrap();
	let frames = stream
		.map(AttemptEvent::into_inner)
		.collect::<Vec<_>>()
		.await;

	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["attempt", "outcome"]);
	assert!(
		matches!(frames[0].event.as_ref(), Some(turn_event::Event::Attempt(attempt)) if attempt.reason == "thinking loop")
	);
}

#[tokio::test]
async fn thinking_loop_cap_forwards_last_error() {
	let script = Script::new([
		vec![error(turn_error::Kind::Upstream, "Thinking loop detected")],
		vec![error(turn_error::Kind::Upstream, "Thinking loop detected")],
	]);
	let stream = Resample::new(Scripted(script), config(1))
		.oneshot(TurnRequest::default())
		.await
		.unwrap();
	let frames = stream
		.map(AttemptEvent::into_inner)
		.collect::<Vec<_>>()
		.await;

	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["attempt", "error"]);
	assert!(
		matches!(frames.last().and_then(|frame| frame.event.as_ref()), Some(turn_event::Event::Error(error)) if error.detail == "Thinking loop detected")
	);
}
