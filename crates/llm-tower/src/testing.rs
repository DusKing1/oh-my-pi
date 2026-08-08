//! Shared test harness: scripted provider-attempt services and frame
//! constructors. `#[doc(hidden)]` — test infrastructure, not API.

use std::{
	collections::VecDeque,
	convert::Infallible,
	sync::Arc,
	task::{Context, Poll},
};

use omp_proto::inference::v1::{
	Invoke, Outcome, PartDelta, PartStart, TurnError, TurnEvent, TurnRequest, turn_error, turn_event,
};
use parking_lot::Mutex;
use tower::Service;

/// Stream type produced by [`Script`].
pub type ScriptStream = futures::stream::Iter<std::vec::IntoIter<TurnEvent>>;

/// Service that answers each call with the next scripted event stream and
/// records every request it received.
#[derive(Clone, Default)]
pub struct Script {
	streams:   Arc<Mutex<VecDeque<Vec<TurnEvent>>>>,
	/// Every request the service saw, in order.
	pub calls: Arc<Mutex<Vec<TurnRequest>>>,
}

impl Script {
	/// Scripted service; each call consumes the next stream (empty stream
	/// once the script is exhausted).
	pub fn new(streams: impl IntoIterator<Item = Vec<TurnEvent>>) -> Self {
		Self {
			streams: Arc::new(Mutex::new(streams.into_iter().collect())),
			calls:   Arc::new(Mutex::new(Vec::new())),
		}
	}
}

impl Service<TurnRequest> for Script {
	type Error = Infallible;
	type Future = std::future::Ready<Result<ScriptStream, Infallible>>;
	type Response = ScriptStream;

	fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Infallible>> {
		Poll::Ready(Ok(()))
	}

	fn call(&mut self, req: TurnRequest) -> Self::Future {
		self.calls.lock().push(req);
		let next = self.streams.lock().pop_front().unwrap_or_default();
		std::future::ready(Ok(futures::stream::iter(next)))
	}
}

/// Wraps an event into a frame.
pub const fn ev(event: turn_event::Event) -> TurnEvent {
	TurnEvent { event: Some(event) }
}

/// Default `PartStart` frame.
pub fn part_start() -> TurnEvent {
	ev(turn_event::Event::PartStart(PartStart::default()))
}

/// Default `PartDelta` frame.
pub fn part_delta() -> TurnEvent {
	ev(turn_event::Event::PartDelta(PartDelta::default()))
}

/// Default success `Outcome` frame (no output items).
pub fn outcome() -> TurnEvent {
	ev(turn_event::Event::Outcome(Outcome::default()))
}

/// Default `Invoke` frame.
pub fn invoke() -> TurnEvent {
	ev(turn_event::Event::Invoke(Invoke::default()))
}

/// Terminal error frame.
pub fn error(kind: turn_error::Kind, detail: &str) -> TurnEvent {
	ev(turn_event::Event::Error(TurnError {
		kind: kind as i32,
		detail: detail.to_owned(),
		..TurnError::default()
	}))
}

/// Frame discriminant label for terse test assertions.
pub const fn kind_of(frame: &TurnEvent) -> &'static str {
	match frame.event {
		Some(turn_event::Event::Accepted(_)) => "accepted",
		Some(turn_event::Event::Attempt(_)) => "attempt",
		Some(turn_event::Event::PartStart(_)) => "part_start",
		Some(turn_event::Event::PartDelta(_)) => "part_delta",
		Some(turn_event::Event::PartEnd(_)) => "part_end",
		Some(turn_event::Event::Outcome(_)) => "outcome",
		Some(turn_event::Event::Error(_)) => "error",
		Some(turn_event::Event::Invoke(_)) => "invoke",
		Some(turn_event::Event::InvokeCancel(_)) => "invoke_cancel",
		_ => "other",
	}
}
