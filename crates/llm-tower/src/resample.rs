//! Semantic re-sampling for empty completions and detected thinking loops.
//!
//! This layer suppresses empty terminal `Outcome` frames and re-dispatches,
//! which is only legal BELOW the turn coordinator's commit/idempotency
//! machinery — above commit, `Outcome` is the authoritative record and a
//! reused `turn_id` replays it instead of re-sampling. The service
//! therefore types on [`AttemptEvent`], a distinct pre-commit domain type:
//! committed `TurnEvent` streams do not unify with it implicitly, and the
//! explicit conversion site is where the pre-commit claim lives and gets
//! reviewed. This is a domain boundary, not compiler-enforced authority —
//! keep construction confined to the coordinator's attempt producer.
//!
//! This layer only repeats attempts that have not emitted a part or an
//! invocation. Thinking-loop exhaustion is forwarded unchanged; disabling the
//! upstream loop guard for a final "cook" pass is the coordinator's concern.

use std::{
	fmt,
	future::{Ready, ready},
	pin::Pin,
	sync::Arc,
	task::{Context, Poll},
	time::Duration,
};

use futures::{Stream, StreamExt};
use omp_llm_error::Kind;
use omp_proto::inference::v1::{
	Attempt, Outcome, StopReason, TurnError, TurnEvent, turn_error, turn_event,
};
use tower::{Layer, Service, ServiceExt};

use crate::{envelope::TurnRequestEnvelope, recovery::classify_turn_error};

/// A turn event observed BEFORE the coordinator commits the attempt.
///
/// A distinct domain type, not a transparent alias: wrapping a frame is the
/// producer's explicit, reviewable assertion that no wrapped terminal
/// `Outcome` has advanced a context revision. The field is private so the
/// conversion cannot happen implicitly; it is NOT unforgeable — restraint
/// lives at the construction site, which should be the coordinator's
/// attempt producer and nowhere else.
#[derive(Clone, Debug)]
pub struct AttemptEvent(TurnEvent);

impl AttemptEvent {
	/// Asserts `frame` precedes commit and enters the attempt domain.
	pub const fn new(frame: TurnEvent) -> Self {
		Self(frame)
	}

	/// The wrapped frame.
	pub const fn frame(&self) -> &TurnEvent {
		&self.0
	}

	/// Unwraps into the plain frame (e.g. for the coordinator to commit).
	pub fn into_inner(self) -> TurnEvent {
		self.0
	}
}

/// Type-erased pre-commit attempt stream for the stack's OUTER boundary
/// only; layers return the concrete [`ResampleStream`].
pub type AttemptStream = Pin<Box<dyn Stream<Item = AttemptEvent> + Send>>;

/// Concrete re-sampling stream.
///
/// One heap-pinned generator per call: the single allocation keeps this
/// layer's state behind a pointer, so composed stacks stay flat. Fully
/// inline generator nesting embeds every inner layer's state in the
/// parent's and was measured to overflow the thread stack at this
/// composition depth; a hand-written pin-projected state machine is the
/// box-free replacement if this layer ever gets hot. Erase to a boxed-dyn
/// stream only
/// at the outer boundary.
pub type ResampleStream<
	S: Service<R, Response = St> + Send + 'static,
	St: Stream<Item = AttemptEvent> + Send + 'static,
	R: TurnRequestEnvelope,
>
	= impl Stream<Item = AttemptEvent> + Send + Unpin
where
	S::Future: Send,
	S::Error: fmt::Display + Send + 'static;

/// Policy for semantic re-sampling.
#[derive(Clone, Debug)]
pub struct ResampleConfig {
	/// Maximum number of re-dispatches after the initial attempt.
	pub max_attempts:  u32,
	/// Initial delay before a re-dispatch, doubled after each re-sample.
	pub base_delay_ms: u64,
}

impl Default for ResampleConfig {
	fn default() -> Self {
		Self { max_attempts: 3, base_delay_ms: 500 }
	}
}

/// [`Layer`] producing [`Resample`] services.
#[derive(Clone, Debug, Default)]
pub struct ResampleLayer {
	config: Arc<ResampleConfig>,
}

impl ResampleLayer {
	/// Creates a layer using `config`.
	pub fn new(config: ResampleConfig) -> Self {
		Self { config: Arc::new(config) }
	}
}

impl<S> Layer<S> for ResampleLayer {
	type Service = Resample<S>;

	fn layer(&self, inner: S) -> Self::Service {
		Resample { inner, config: Arc::clone(&self.config) }
	}
}

/// Re-sampling wrapper around an inference-attempt service.
#[derive(Clone, Debug)]
pub struct Resample<S> {
	inner:  S,
	config: Arc<ResampleConfig>,
}

impl<S> Resample<S> {
	/// Wraps `inner` with the supplied semantic re-sampling policy.
	pub fn new(inner: S, config: ResampleConfig) -> Self {
		Self { inner, config: Arc::new(config) }
	}
}

impl<S, St, R> Service<R> for Resample<S>
where
	R: TurnRequestEnvelope,
	S: Service<R, Response = St> + Clone + Send + 'static,
	S::Future: Send,
	S::Error: fmt::Display + Send + 'static,
	St: Stream<Item = AttemptEvent> + Send + 'static,
{
	type Error = S::Error;
	type Future = Ready<Result<Self::Response, S::Error>>;
	type Response = ResampleStream<S, St, R>;

	fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		Poll::Ready(Ok(()))
	}

	fn call(&mut self, req: R) -> Self::Future {
		let clone = self.inner.clone();
		let inner = std::mem::replace(&mut self.inner, clone);
		let config = Arc::clone(&self.config);
		ready(Ok(resample_stream(inner, req, config)))
	}
}

#[define_opaque(ResampleStream)]
fn resample_stream<S, St, R>(
	svc: S,
	req: R,
	config: Arc<ResampleConfig>,
) -> ResampleStream<S, St, R>
where
	R: TurnRequestEnvelope,
	S: Service<R, Response = St> + Send + 'static,
	S::Future: Send,
	S::Error: fmt::Display + Send + 'static,
	St: Stream<Item = AttemptEvent> + Send + 'static,
{
	Box::pin(async_stream::stream! {
		let mut svc = svc;
		let first = match svc.ready().await {
			Ok(svc) => match svc.call(req.clone()).await {
				Ok(stream) => stream,
				Err(error) => {
					yield AttemptEvent::new(service_error(&error));
					return;
				},
			},
			Err(error) => {
				yield AttemptEvent::new(service_error(&error));
				return;
			},
		};
		let mut current = std::pin::pin!(first);
		let mut retries: u32 = 0;
		let mut saw_part = false;
		let mut invoked = false;
		loop {
			let Some(event) = current.next().await else {
				return;
			};

			let reason = match event.0.event.as_ref() {
				Some(
					turn_event::Event::PartStart(_)
					| turn_event::Event::PartDelta(_)
					| turn_event::Event::PartEnd(_),
				) => {
					saw_part = true;
					yield event;
					continue;
				},
				Some(turn_event::Event::Invoke(_) | turn_event::Event::InvokeCancel(_)) => {
					invoked = true;
					yield event;
					continue;
				},
				Some(turn_event::Event::Outcome(outcome)) => {
					if is_empty(outcome) && !saw_part && !invoked {
						"empty completion"
					} else {
						yield event;
						return;
					}
				},
				Some(turn_event::Event::Error(err)) => {
					let classification = classify_turn_error(err);
					if classification.kinds.has(Kind::ThinkingLoop) && !saw_part && !invoked {
						"thinking loop"
					} else {
						yield event;
						return;
					}
				},
				_ => {
					yield event;
					continue;
				},
			};

			if retries >= config.max_attempts {
				yield event;
				return;
			}

			let delay_ms = backoff_ms(config.base_delay_ms, retries);
			tokio::time::sleep(Duration::from_millis(delay_ms)).await;
			let next = if let Ok(ready) = svc.ready().await {
				ready.call(req.clone()).await
			} else {
				yield event;
				return;
			};
			if let Ok(stream) = next {
				retries += 1;
				current.set(stream);
				saw_part = false;
				invoked = false;
				yield AttemptEvent(TurnEvent {
					event: Some(turn_event::Event::Attempt(Attempt {
						number: retries + 1,
						reason: reason.to_owned(),
					})),
				});
			} else {
				yield event;
				return;
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

fn is_empty(outcome: &Outcome) -> bool {
	outcome.output.is_empty() && outcome.stop() != StopReason::StopToolUse
}

fn backoff_ms(base: u64, retry: u32) -> u64 {
	base
		.saturating_mul(1_u64.checked_shl(retry).unwrap_or(u64::MAX))
		.min(8_000)
}
