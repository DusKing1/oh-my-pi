//! Phase deadlines with cancellation provenance.
//!
//! This layer deliberately has no total deadline: a healthy slow stream may
//! run for arbitrarily long as long as it produces each frame within the idle
//! budget.

use std::{
	fmt,
	future::{Ready, ready},
	task::{Context, Poll},
	time::Duration,
};

use futures::{Stream, StreamExt};
use omp_proto::inference::v1::{TurnError, TurnEvent, turn_error, turn_event};
use tokio::time::{Instant, timeout, timeout_at};
use tower::{Layer, Service, ServiceExt};

use crate::envelope::TurnRequestEnvelope;

/// Deadlines for the three independently timed phases of a provider attempt.
///
/// There is intentionally no total deadline. Streams that keep producing
/// frames within [`Self::idle_ms`] are allowed to run indefinitely.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PhaseTimeoutConfig {
	/// Deadline for the inner call future to yield its event stream.
	pub call_ms:        u64,
	/// Deadline for the stream to yield its first event.
	pub first_event_ms: u64,
	/// Deadline between every pair of events after the first.
	pub idle_ms:        u64,
}

impl Default for PhaseTimeoutConfig {
	fn default() -> Self {
		Self { call_ms: 60_000, first_event_ms: 300_000, idle_ms: 300_000 }
	}
}

/// [`Layer`] producing [`PhaseTimeout`] services.
#[derive(Clone, Copy, Debug, Default)]
pub struct PhaseTimeoutLayer {
	config: PhaseTimeoutConfig,
}

impl PhaseTimeoutLayer {
	/// Creates a timeout layer with the given phase deadlines.
	pub const fn new(config: PhaseTimeoutConfig) -> Self {
		Self { config }
	}
}

impl<S> Layer<S> for PhaseTimeoutLayer {
	type Service = PhaseTimeout<S>;

	fn layer(&self, inner: S) -> Self::Service {
		PhaseTimeout { inner, config: self.config }
	}
}

/// Provider-attempt service with connect, first-event, and idle deadlines.
///
/// Deadline expiry is surfaced in-band as one terminal `TurnError` frame so
/// downstream classification retains cancellation provenance.
#[derive(Clone, Debug)]
pub struct PhaseTimeout<S> {
	inner:  S,
	config: PhaseTimeoutConfig,
}

impl<S> PhaseTimeout<S> {
	/// Wraps `inner` with the given phase deadlines.
	pub const fn new(inner: S, config: PhaseTimeoutConfig) -> Self {
		Self { inner, config }
	}
}

impl<S, St, R> Service<R> for PhaseTimeout<S>
where
	R: TurnRequestEnvelope,
	S: Service<R, Response = St> + Clone + Send + 'static,
	S::Future: Send,
	S::Error: fmt::Display + Send + 'static,
	St: Stream<Item = TurnEvent> + Send + 'static,
{
	type Error = S::Error;
	type Future = Ready<Result<Self::Response, S::Error>>;
	type Response = PhasedStream<S, St, R>;

	fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		Poll::Ready(Ok(()))
	}

	fn call(&mut self, req: R) -> Self::Future {
		// Move the service whose readiness was observed into the call future;
		// the clone only replaces it for the next readiness cycle.
		let clone = self.inner.clone();
		let inner = std::mem::replace(&mut self.inner, clone);
		let config = self.config;
		let deadline = Instant::now() + Duration::from_millis(config.call_ms);
		ready(Ok(timeout_stream(inner, req, config, deadline)))
	}
}

/// Concrete phase-deadline stream.
///
/// One heap-pinned generator per call: the single allocation keeps this
/// layer's state behind a pointer, so composed stacks stay flat. Fully
/// inline generator nesting embeds every inner layer's state in the
/// parent's and was measured to overflow the thread stack at this
/// composition depth; a hand-written pin-projected state machine is the
/// box-free replacement if this layer ever gets hot. Erase to a boxed-dyn
/// stream only
/// at the outer boundary.
pub type PhasedStream<
	S: Service<R, Response = St> + Send + 'static,
	St: Stream<Item = TurnEvent> + Send + 'static,
	R: TurnRequestEnvelope,
>
	= impl Stream<Item = TurnEvent> + Send + Unpin
where
	S::Future: Send,
	S::Error: fmt::Display + Send + 'static;

#[define_opaque(PhasedStream)]
fn timeout_stream<S, St, R>(
	mut svc: S,
	req: R,
	config: PhaseTimeoutConfig,
	call_deadline: Instant,
) -> PhasedStream<S, St, R>
where
	R: TurnRequestEnvelope,
	S: Service<R, Response = St> + Send + 'static,
	S::Future: Send,
	S::Error: fmt::Display + Send + 'static,
	St: Stream<Item = TurnEvent> + Send + 'static,
{
	Box::pin(async_stream::stream! {
		let stream = match timeout_at(call_deadline, async {
			svc.ready().await?.call(req).await
		})
		.await
		{
			Ok(Ok(stream)) => stream,
			Ok(Err(error)) => {
				yield service_error(&error);
				return;
			},
			Err(_) => {
				yield deadline_error("connect", config.call_ms);
				return;
			},
		};
		let mut stream = std::pin::pin!(stream);
		let mut first = true;
		loop {
			let (phase, budget_ms) = if first {
				("first-event", config.first_event_ms)
			} else {
				("idle", config.idle_ms)
			};
			match timeout(Duration::from_millis(budget_ms), stream.next()).await {
				Ok(Some(frame)) => {
					first = false;
					yield frame;
				},
				Ok(None) => return,
				Err(_) => {
					yield deadline_error(phase, budget_ms);
					return;
				},
			}
		}
	})
}

fn service_error(error: &impl fmt::Display) -> TurnEvent {
	TurnEvent {
		event: Some(turn_event::Event::Error(TurnError {
			kind: turn_error::Kind::Upstream as i32,
			detail: error.to_string(),
			..TurnError::default()
		})),
	}
}

fn deadline_error(phase: &'static str, budget_ms: u64) -> TurnEvent {
	TurnEvent {
		event: Some(turn_event::Event::Error(TurnError {
			kind: turn_error::Kind::Upstream as i32,
			detail: format!("{phase} deadline: timed out after {budget_ms}ms"),
			..TurnError::default()
		})),
	}
}
