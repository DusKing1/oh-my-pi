//! Tower recovery middleware for `omp.inference.v1` provider attempts.
//!
//! [`Recovery`] wraps a `Service<TurnRequest>` whose response is a stream
//! of [`TurnEvent`]s. That is NOT the public `Turn` RPC — the RPC is a
//! bidirectional frame stream whose later inbound frames carry live client
//! traffic (tool-invocation input, completion). This layer belongs INSIDE
//! the turn coordinator, around its provider-attempt dispatch: the
//! coordinator owns the client half of the conversation and re-drives
//! interactive exchanges itself; what it hands this layer is the
//! half-closed "one provider attempt in, event stream out" boundary, which
//! IS transparently replayable.
//!
//! On a terminal [`TurnError`] frame the layer classifies the failure and —
//! when the verdict and budget allow — re-dispatches the SAME request
//! (same `turn_id`, so idempotency dedupes) after the advised backoff,
//! surfacing each restart as an honest [`Attempt`] frame. Terminal
//! failures pass through, optionally normalized (kind corrected to
//! `KIND_RATE_LIMITED`, `retry_after_ms` filled from parsed evidence).
//!
//! Retry is strictly pre-commit. Once any part or invocation frame is visible,
//! the client owns it and no layer may replay the request. Cancellation is
//! likewise terminal and never spends retry budget or triggers fallback.

use std::{
	fmt,
	future::{Ready, ready},
	hash::{BuildHasher, Hasher},
	pin::Pin,
	sync::Arc,
	task::{Context, Poll},
};

use futures::{Stream, StreamExt};
use omp_llm_error::{
	Classification, Evidence, Kind, Kinds, RateLimit, RateLimitReason, RetryBudget, RetryDecision,
	RetryHint, WireApi, classify_at,
};
use omp_proto::{
	inference::v1::{Attempt, TurnError, TurnEvent, TurnRequest, turn_error, turn_event, value},
	thread::v1::Item,
};
use tower::{Layer, Service, ServiceExt};

use crate::{envelope::TurnRequestEnvelope, single_turn};

/// Recovery policy configuration shared by all turns through one layer.
#[derive(Clone, Debug)]
pub struct RecoveryConfig {
	/// Per-turn retry budget template.
	pub budget:    RetryBudget,
	/// Rewrite terminal error frames with classified facts: upgrade
	/// `KIND_UPSTREAM` to `KIND_RATE_LIMITED` when the body proves a limit,
	/// and fill `retry_after_ms` from parsed evidence when the gateway left
	/// it zero.
	pub normalize: bool,
}

impl Default for RecoveryConfig {
	fn default() -> Self {
		Self { budget: RetryBudget::default(), normalize: true }
	}
}

/// [`Layer`] producing [`Recovery`] services.
#[derive(Clone, Debug, Default)]
pub struct RecoveryLayer {
	config: Arc<RecoveryConfig>,
}

impl RecoveryLayer {
	/// Layer with the given policy.
	pub fn new(config: RecoveryConfig) -> Self {
		Self { config: Arc::new(config) }
	}
}

impl<S> Layer<S> for RecoveryLayer {
	type Service = Recovery<S>;

	fn layer(&self, inner: S) -> Self::Service {
		Recovery { inner, config: Arc::clone(&self.config) }
	}
}

/// Type-erased turn stream for the stack's OUTER boundary only.
///
/// Middleware layers return concrete stream types; erase at most once, at
/// the edge that genuinely needs `dyn` (RPC surface, heterogeneous
/// registries) — never between layers.
pub type TurnStream = Pin<Box<dyn Stream<Item = TurnEvent> + Send>>;

/// Concrete retrying stream produced by [`Recovery`].
///
/// Opaque alias so the stack
/// composes with static dispatch. One heap-pinned generator per call: the
/// single allocation keeps this layer's state behind a pointer, so composed
/// stacks stay flat. Fully inline generator nesting embeds every inner
/// layer's state in the parent's and was measured to overflow the thread
/// stack at this composition depth; a hand-written pin-projected state
/// machine is the box-free replacement if this layer ever gets hot.
/// Erase into [`TurnStream`] only at the outer boundary.
pub type RecoveryStream<
	S: Service<R, Response = St> + Send + 'static,
	St: Stream<Item = TurnEvent> + Send + 'static,
	R: TurnRequestEnvelope,
>
	= impl Stream<Item = TurnEvent> + Send + Unpin
where
	S::Future: Send,
	S::Error: fmt::Display + Send + 'static;

/// Retrying, classifying wrapper around an inference turn service.
#[derive(Clone, Debug)]
pub struct Recovery<S> {
	inner:  S,
	config: Arc<RecoveryConfig>,
}

impl<S> Recovery<S> {
	/// Wraps `inner` with the given policy.
	pub fn new(inner: S, config: RecoveryConfig) -> Self {
		Self { inner, config: Arc::new(config) }
	}
}

impl<S, St, R> Service<R> for Recovery<S>
where
	R: TurnRequestEnvelope,
	S: Service<R, Response = St> + Clone + Send + 'static,
	S::Future: Send,
	S::Error: fmt::Display + Send + 'static,
	St: Stream<Item = TurnEvent> + Send + 'static,
{
	type Error = S::Error;
	type Future = Ready<Result<Self::Response, S::Error>>;
	type Response = RecoveryStream<S, St, R>;

	fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), S::Error>> {
		Poll::Ready(Ok(()))
	}

	fn call(&mut self, req: R) -> Self::Future {
		// Standard tower pattern: take the ready service, leave a fresh clone.
		let clone = self.inner.clone();
		let inner = std::mem::replace(&mut self.inner, clone);
		let config = Arc::clone(&self.config);
		ready(Ok(recovery_stream(inner, req, config)))
	}
}

#[define_opaque(RecoveryStream)]
fn recovery_stream<S, St, R>(
	svc: S,
	req: R,
	config: Arc<RecoveryConfig>,
) -> RecoveryStream<S, St, R>
where
	R: TurnRequestEnvelope,
	S: Service<R, Response = St> + Send + 'static,
	S::Future: Send,
	S::Error: fmt::Display + Send + 'static,
	St: Stream<Item = TurnEvent> + Send + 'static,
{
	Box::pin(async_stream::stream! {
		let mut svc = svc;
		let responses_continuation = has_responses_continuation(req.request());
		let first = match svc.ready().await {
			Ok(svc) => match svc.call(req.clone()).await {
				Ok(stream) => futures::future::Either::Left(stream),
				Err(error) => futures::future::Either::Right(single_turn(service_error(&error))),
			},
			Err(error) => futures::future::Either::Right(single_turn(service_error(&error))),
		};
		let mut budget = config.budget.clone();
		let mut saw_output = false;
		let mut current = std::pin::pin!(first);
		// An `Invoke` frame passed through an attempt — hard replay bar.
		let mut invoked = false;
		let mut replay_req = req;
		let mut stale_session_replayed = false;
		loop {
			// Everything except a terminal error frame passes straight
			// through; a terminal error — including the synthetic one for a
			// stream that ended without any terminal frame — falls into the
			// shared recovery decision below.
			let err = match current.next().await {
				// Premature EOF violates the protocol's terminal-frame
				// contract: treat it as upstream stream corruption, never as
				// clean completion.
				None => TurnError {
					kind: turn_error::Kind::Upstream as i32,
					detail: "provider stream ended before a terminal frame".to_owned(),
					..TurnError::default()
				},
				Some(event) => match event.event {
					Some(turn_event::Event::Error(err)) => err,
					Some(turn_event::Event::Outcome(_)) => {
						yield event;
						return;
					},
					Some(
						turn_event::Event::PartStart(_)
						| turn_event::Event::PartDelta(_)
						| turn_event::Event::PartEnd(_),
					) => {
						saw_output = true;
						yield event;
						continue;
					},
					Some(turn_event::Event::Invoke(_) | turn_event::Event::InvokeCancel(_)) => {
						// Latched for the rest of the turn — even a cancelled
						// invocation may have executed partially, and a fresh
						// attempt cannot replay the client half.
						invoked = true;
						saw_output = true;
						yield event;
						continue;
					},
					_ => {
						yield event;
						continue;
					},
				},
			};

			let cls = classify_turn_error_with_api(
				&err,
				responses_continuation.then_some(WireApi::OpenAiResponses),
			);
			if cls.kinds.has(Kind::Aborted) {
				yield finalize(err, &cls, &config);
				return;
			}
			// Any visible part or invocation is a hard commit boundary. The
			// client may already have observed output or performed side effects.
			let replay_safe = !saw_output && !invoked;
			let repaired_stale = replay_safe
				&& cls.kinds.has(Kind::StaleSessionItem)
				&& !stale_session_replayed;
			let decision = if repaired_stale {
				reset_responses_continuation(replay_req.request_mut());
				stale_session_replayed = true;
				budget.decide_repaired(&cls, true, jitter01())
			} else {
				budget.decide(&cls, replay_safe, jitter01())
			};
			match decision {
				RetryDecision::Retry { delay_ms, attempt } => {
					tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
					let redispatch = match svc.ready().await {
						Ok(svc) => svc.call(replay_req.clone()).await,
						Err(e) => Err(e),
					};
					current.set(match redispatch {
						Ok(next) => futures::future::Either::Left(next),
						Err(error) => futures::future::Either::Right(single_turn(service_error(&error))),
					});
					saw_output = false;
					yield TurnEvent {
						event: Some(turn_event::Event::Attempt(Attempt {
							// Attempt numbers are 1-based and count
							// dispatches, so retry N announces N+1.
							number: attempt + 1,
							reason: truncate(&err.detail, 256),
						})),
					};
				},
				RetryDecision::GiveUp(_) => {
					yield finalize(err, &cls, &config);
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

/// Clears Responses continuation state before a one-shot full-context replay.
///
/// The canonical thread remains intact. Only the server-held anchor, its delta
/// boundary, and provider-issued item ids are removed; portable call ids and
/// encrypted reasoning remain available to reconstruct the conversation.
fn reset_responses_continuation(req: &mut TurnRequest) {
	if let Some(params) = &mut req.params
		&& let Some(options) = &mut params.provider_options
	{
		options.fields.remove("openai/previous_response_id");
		options.fields.remove("openai/previous_response_item_count");
	}
	match req.input.as_mut() {
		Some(omp_proto::inference::v1::turn_request::Input::Seed(seed)) => {
			if let Some(thread) = &mut seed.thread {
				strip_server_item_ids(&mut thread.items);
			}
		},
		Some(omp_proto::inference::v1::turn_request::Input::Incremental(incremental)) => {
			if let Some(delta) = &mut incremental.delta {
				strip_server_item_ids(&mut delta.append);
			}
		},
		None => {},
	}
}

fn strip_server_item_ids(items: &mut [Item]) {
	for item in items {
		let Some(props) = &mut item.props else {
			continue;
		};
		props.fields.remove("openai/item_id");
		if let Some(server_item) = props.fields.get_mut("openai/server_tool_item")
			&& let Some(value::Kind::Map(server_item)) = &mut server_item.kind
		{
			server_item.fields.remove("id");
		}
	}
}

fn has_responses_continuation(req: &TurnRequest) -> bool {
	req.params
		.as_ref()
		.and_then(|params| params.provider_options.as_ref())
		.is_some_and(|options| options.fields.contains_key("openai/previous_response_id"))
}

/// Maps a terminal protocol error to a classification.
///
/// The proto kind is a coarse prior; `detail` prose refines it (a
/// `KIND_UPSTREAM` whose detail says "usage limit reached" classifies as a
/// rotatable usage limit).
pub fn classify_turn_error(err: &TurnError) -> Classification {
	classify_turn_error_with_api(err, None)
}

fn classify_turn_error_with_api(err: &TurnError, api: Option<WireApi>) -> Classification {
	let now_ms = std::time::SystemTime::now()
		.duration_since(std::time::UNIX_EPOCH)
		.map_or(0, |d| d.as_millis() as u64);
	// `detail` often embeds the flattened provider body; expose it as body
	// too so envelope parsing recovers machine codes.
	let mut evidence = Evidence { body: Some(&err.detail), ..Evidence::live(&err.detail) };
	evidence.api = api;
	let mut cls = classify_at(&evidence, now_ms);
	match err.kind() {
		turn_error::Kind::RateLimited => {
			cls.kinds |= Kind::RateThrottle | Kind::Transient;
			// The protocol already asserts this IS a rate limit; an
			// unrecognized detail must not leave the lane verdict empty.
			if cls.rate_limit.is_none() {
				cls.rate_limit =
					Some(RateLimit { reason: RateLimitReason::RateLimitExceeded, rotate: false });
			}
		},
		turn_error::Kind::Overloaded => cls.kinds |= Kind::ModelCapacity | Kind::Transient,
		turn_error::Kind::InvokeTimeout => cls.kinds |= Kind::Timeout | Kind::Transient,
		turn_error::Kind::Auth => cls.kinds |= Kind::AuthFailed,
		turn_error::Kind::Upstream => {},
		// Client-actionable protocol outcomes: never auto-retried here.
		turn_error::Kind::Conflict
		| turn_error::Kind::NeedFull
		| turn_error::Kind::Unsupported
		| turn_error::Kind::Unspecified => {
			cls.kinds = Kinds::EMPTY;
		},
	}
	// Auth at this layer means "no usable credential" — the gateway already
	// exhausted rotation. Not retryable by waiting.
	if err.kind() == turn_error::Kind::Auth {
		cls.kinds.remove(Kind::Transient);
	}
	if err.retry_after_ms > 0 && cls.retry.is_none() && cls.kinds.has(Kind::Transient) {
		cls.retry = Some(RetryHint { delay_ms: err.retry_after_ms, reset_at_ms: None });
	}
	cls
}

/// Applies terminal normalization per config.
fn finalize(mut err: TurnError, cls: &Classification, config: &RecoveryConfig) -> TurnEvent {
	if config.normalize {
		if err.kind() == turn_error::Kind::Upstream && cls.rate_limit.is_some() {
			err.set_kind(turn_error::Kind::RateLimited);
		}
		if err.retry_after_ms == 0
			&& let Some((min, _)) = cls.suggested_backoff_ms()
		{
			err.retry_after_ms = min;
		}
	}
	TurnEvent { event: Some(turn_event::Event::Error(err)) }
}

fn truncate(s: &str, max: usize) -> String {
	if s.len() <= max {
		return s.to_owned();
	}
	let mut end = max;
	while !s.is_char_boundary(end) {
		end -= 1;
	}
	format!("{}…", &s[..end])
}

/// Cheap entropy in `[0, 1)` for backoff jitter; quality is irrelevant,
/// decorrelation is the only goal.
fn jitter01() -> f64 {
	let h = std::collections::hash_map::RandomState::new();
	let mut hasher = h.build_hasher();
	hasher.write_u128(
		std::time::SystemTime::now()
			.duration_since(std::time::UNIX_EPOCH)
			.map_or(0, |d| d.as_nanos()),
	);
	(hasher.finish() >> 11) as f64 / (1u64 << 53) as f64
}
