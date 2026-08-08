//! Canonical stream policy for once-built provider routes.
//!
//! Watchdog, healing, and loop detection wrap each live provider stream.
//! Forced-tool emulation buffers an uncommitted candidate attempt so a
//! non-compliant result can be discarded and retried without leaking partial
//! output. Dropping any adapter drops the currently live upstream stream.

use std::{
	fmt,
	future::{Future, Ready, ready},
	task::{Context, Poll},
	time::Duration,
};

use async_stream::stream;
use bytes::{Buf, Bytes, BytesMut};
use futures::{Stream, StreamExt, TryFutureExt, pin_mut};
use omp_core::SmolStr;
use omp_llm_catalog::compat::{Compat, LeakedThinkingHealer};
use omp_llm_types::{StreamPartKind, TurnError, TurnErrorKind, TurnEvent, ids::CallId};
use omp_proto::{
	inference::v1::{
		Fallback, StopReason, TurnError as ProtoTurnError, TurnEvent as ProtoTurnEvent, tool_choice,
		turn_error, turn_event as proto_turn_event,
	},
	thread::v1::{Item, Message, Part, Role, item, part},
};
use rustc_hash::FxHashSet;
use smallvec::SmallVec;
use tower::{Layer, Service, ServiceExt};

use crate::{
	envelope::TurnRequestEnvelope,
	stack::capability::{ForcedToolEscalation, ForcedToolStrategy},
};

/// Heap-pinned so this policy contributes only one pointer-sized state field
/// when composed with the rest of the production route.
pub type ProductionPolicyStream<S: Stream<Item = ProtoTurnEvent> + Send + 'static> =
	impl Stream<Item = ProtoTurnEvent> + Send + Unpin;

/// Heap-pinned canonical-event conversion used inside [`production_policy`].
type CanonicalEventStream<S: Stream<Item = ProtoTurnEvent> + Send + 'static> =
	impl Stream<Item = TurnEvent> + Send + Unpin;

/// Applies every catalog-selected production stream guard to a canonical
/// protobuf attempt stream.
///
/// The protobuf/native conversion is lossless for canonical events. A corrupt
/// internal frame becomes one terminal upstream error rather than silently
/// disabling watchdog, healing, or loop detection.

#[define_opaque(ProductionPolicyStream)]
pub fn production_policy<S>(stream: S, compat: Compat) -> ProductionPolicyStream<S>
where
	S: Stream<Item = ProtoTurnEvent> + Send + 'static,
{
	Box::pin(stream! {
		let enabled = compat.leaked_thinking_healer != LeakedThinkingHealer::None
			|| compat.thinking_loop_guard
			|| compat.stream_watchdog.first_event_ms.is_some()
			|| compat.stream_watchdog.idle_ms.is_some();
		if !enabled {
			pin_mut!(stream);
			while let Some(event) = stream.next().await {
				yield event;
			}
			return;
		}
		let guarded =
			guard_thinking_loop(heal(watchdog(canonical_events(stream), compat), compat), compat);
		pin_mut!(guarded);
		while let Some(event) = guarded.next().await {
			yield ProtoTurnEvent::from(event);
		}
	})
}

#[define_opaque(CanonicalEventStream)]
fn canonical_events<S>(stream: S) -> CanonicalEventStream<S>
where
	S: Stream<Item = ProtoTurnEvent> + Send + 'static,
{
	Box::pin(stream! {
		pin_mut!(stream);
		while let Some(event) = stream.next().await {
			match TurnEvent::try_from(event) {
				Ok(event) => yield event,
				Err(error) => {
					yield TurnEvent::Error(
						TurnError::builder()
							.kind(TurnErrorKind::Upstream)
							.detail(SmolStr::new(format!(
								"invalid canonical stream event: {error}"
							)))
							.unsupported(Vec::new())
							.retry_after_ms(0)
							.build(),
					);
					return;
				},
			}
		}
	})
}

/// Once-built catalog-selected watchdog, healer, and loop-guard layer.
#[derive(Clone, Copy, Debug)]
pub struct ProductionPolicyLayer {
	compat: Compat,
}

impl ProductionPolicyLayer {
	/// Creates the production stream policy for one provider catalog row.
	#[must_use]
	pub const fn new(compat: Compat) -> Self {
		Self { compat }
	}
}

impl<S> Layer<S> for ProductionPolicyLayer {
	type Service = ProductionPolicy<S>;

	fn layer(&self, inner: S) -> Self::Service {
		ProductionPolicy { inner, compat: self.compat }
	}
}

/// Service whose every response stream runs the production stream policy.
#[derive(Clone, Debug)]
pub struct ProductionPolicy<S> {
	inner:  S,
	compat: Compat,
}

impl<S, St, R> Service<R> for ProductionPolicy<S>
where
	R: Send + 'static,
	S: Service<R, Response = St>,
	S::Future: Send,
	St: Stream<Item = ProtoTurnEvent> + Send + 'static,
{
	type Error = S::Error;
	type Response = ProductionPolicyStream<St>;

	type Future = impl Future<Output = Result<Self::Response, Self::Error>> + Send;

	fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		self.inner.poll_ready(cx)
	}

	fn call(&mut self, req: R) -> Self::Future {
		let compat = self.compat;
		self
			.inner
			.call(req)
			.map_ok(move |stream| production_policy(stream, compat))
	}
}

/// Once-built layer which applies forced-tool emulation around one provider
/// attempt service.
#[derive(Clone, Copy, Debug)]
pub struct ForcedToolLayer {
	compat:       Compat,
	max_attempts: u32,
}

impl ForcedToolLayer {
	/// Creates a bounded forced-tool escalation layer.
	#[must_use]
	pub const fn new(compat: Compat, max_attempts: u32) -> Self {
		Self { compat, max_attempts }
	}
}

impl<S> Layer<S> for ForcedToolLayer {
	type Service = ForcedTool<S>;

	fn layer(&self, inner: S) -> Self::Service {
		ForcedTool { inner, compat: self.compat, max_attempts: self.max_attempts }
	}
}

/// Provider service with cache-friendly soft forcing and bounded native
/// escalation.
#[derive(Clone, Debug)]
pub struct ForcedTool<S> {
	inner:        S,
	compat:       Compat,
	max_attempts: u32,
}

/// Concrete stream returned by [`ForcedTool`].
pub type ForcedToolStream<
	S: Service<R, Response = St> + Send + 'static,
	St: Stream<Item = ProtoTurnEvent> + Send + 'static,
	R: TurnRequestEnvelope,
>
	= impl Stream<Item = ProtoTurnEvent> + Send + Unpin
where
	S::Future: Send,
	S::Error: fmt::Display + Send + 'static;

impl<S, St, R> Service<R> for ForcedTool<S>
where
	R: TurnRequestEnvelope,
	S: Service<R, Response = St> + Clone + Send + 'static,
	S::Future: Send,
	S::Error: fmt::Display + Send + 'static,
	St: Stream<Item = ProtoTurnEvent> + Send + 'static,
{
	type Error = S::Error;
	type Future = Ready<Result<Self::Response, Self::Error>>;
	type Response = ForcedToolStream<S, St, R>;

	fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		Poll::Ready(Ok(()))
	}

	fn call(&mut self, req: R) -> Self::Future {
		let clone = self.inner.clone();
		let inner = std::mem::replace(&mut self.inner, clone);
		ready(Ok(forced_tool_stream(inner, req, self.compat, self.max_attempts)))
	}
}

#[define_opaque(ForcedToolStream)]
fn forced_tool_stream<S, St, R>(
	svc: S,
	original: R,
	compat: Compat,
	max_attempts: u32,
) -> ForcedToolStream<S, St, R>
where
	R: TurnRequestEnvelope,
	S: Service<R, Response = St> + Send + 'static,
	S::Future: Send,
	S::Error: fmt::Display + Send + 'static,
	St: Stream<Item = ProtoTurnEvent> + Send + 'static,
{
	Box::pin(stream! {
		let mut svc = svc;
		let original = original;
		let mut ladder = ForcedToolEscalation::new(&compat, max_attempts);
		let active = forced_tool_emulation_requested(original.request());
		let (first_request, first_event) = if active {
			let plan = ladder.start().expect("a fresh forced-tool ladder always starts");
			match request_for_strategy(&original, plan.strategy) {
				Some(request) => (request, Some(ProtoTurnEvent::from(plan.event))),
				None => (original.clone(), None),
			}
		} else {
			(original.clone(), None)
		};
		let first = match svc.ready().await {
			Ok(svc) => match svc.call(first_request).await {
				Ok(stream) => stream,
				Err(error) => {
					yield service_error(&error);
					return;
				},
			},
			Err(error) => {
				yield service_error(&error);
				return;
			},
		};
		let mut current = std::pin::pin!(first);
		if !active {
			while let Some(event) = current.next().await {
				yield event;
			}
			return;
		}
		if let Some(event) = first_event {
			yield event;
		}
		loop {
			let mut buffered = Vec::new();
			let mut complied = false;
			let mut successful = false;
			while let Some(event) = current.next().await {
				match event.event.as_ref() {
					Some(proto_turn_event::Event::PartStart(part))
						if part.kind() == omp_proto::inference::v1::part_start::Kind::ToolCall =>
					{
						complied = true;
					},
					Some(proto_turn_event::Event::Invoke(_)) => complied = true,
					Some(proto_turn_event::Event::Outcome(outcome)) => {
						successful = true;
						complied |= outcome.stop() == StopReason::StopToolUse;
					},
					Some(proto_turn_event::Event::Error(_)) => {
						for frame in buffered.drain(..) {
							yield frame;
						}
						yield event;
						return;
					},
					_ => {},
				}
				buffered.push(event);
				if successful {
					break;
				}
			}
			if !successful || complied {
				for event in buffered {
					yield event;
				}
				return;
			}

			let plan = match ladder.verify(false) {
				Ok(Some(plan)) => plan,
				Ok(None) => {
					for event in buffered {
						yield event;
					}
					return;
				},
				Err(error) => {
					yield ProtoTurnEvent::from(TurnEvent::Error(error));
					return;
				},
			};
			let Some(next_request) = request_for_strategy(&original, plan.strategy) else {
				for event in buffered {
					yield event;
				}
				return;
			};
			yield ProtoTurnEvent::from(plan.event);
			let next = match svc.ready().await {
				Ok(ready) => ready.call(next_request).await,
				Err(error) => Err(error),
			};
			let Ok(next) = next else {
				for event in buffered {
					yield event;
				}
				return;
			};
			current.set(next);
		}
	})
}

fn service_error(error: &impl fmt::Display) -> ProtoTurnEvent {
	ProtoTurnEvent {
		event: Some(proto_turn_event::Event::Error(ProtoTurnError {
			kind: turn_error::Kind::Upstream as i32,
			detail: error.to_string(),
			..ProtoTurnError::default()
		})),
	}
}

fn forced_tool_emulation_requested(req: &omp_proto::inference::v1::TurnRequest) -> bool {
	req.params
		.as_ref()
		.and_then(|params| params.tool_choice.as_ref())
		.is_some_and(|choice| {
			matches!(choice.mode(), tool_choice::Mode::Required | tool_choice::Mode::Named)
				&& choice.on_unsupported() == Fallback::Emulate
		})
}

fn request_for_strategy<R: TurnRequestEnvelope>(
	original: &R,
	strategy: ForcedToolStrategy,
) -> Option<R> {
	let mut request = original.clone();
	if strategy == ForcedToolStrategy::ForceNative {
		return Some(request);
	}
	let params = request.request_mut().params.as_mut()?;
	let choice = params.tool_choice.as_mut()?;
	let tool_name = if choice.mode() == tool_choice::Mode::Named {
		choice.name.clone()
	} else {
		"one of the available tools".to_owned()
	};
	choice.set_mode(tool_choice::Mode::Auto);
	choice.name.clear();
	let instruction = Item {
		kind: Some(item::Kind::Message(Message {
			role:  Role::System as i32,
			parts: vec![Part {
				kind: Some(part::Kind::Text(format!(
					"You must call {tool_name}. Do not answer with ordinary text."
				))),
			}],
		})),
		..Item::default()
	};
	match request.request_mut().input.as_mut()? {
		omp_proto::inference::v1::turn_request::Input::Seed(seed) => {
			seed.thread.get_or_insert_default().items.push(instruction)
		},
		omp_proto::inference::v1::turn_request::Input::Incremental(incremental) => incremental
			.delta
			.get_or_insert_default()
			.append
			.push(instruction),
	}
	Some(request)
}

const THINKING_LOOP_MAX_GUARDED_ATTEMPTS: u32 = 3;
const VERBATIM_TAIL_WINDOW: usize = 250;
const VERBATIM_MIN_REPEATED_BYTES: usize = 180;
const VERBATIM_MAX_UNIT: usize = 60;
const SEGMENT_BYTE_CAP: usize = 700;
const SEGMENT_MIN_NORMALIZED_BYTES: usize = 60;
const SEGMENT_WINDOW: usize = 16;
const SEGMENT_MIN_COUNT: usize = 8;
const SEGMENT_MIN_CLUSTER: usize = 4;
const SEGMENT_SIMILARITY: f32 = 0.8;
const NOVELTY_WINDOW: usize = 8;
const NOVELTY_FLOOR: f32 = 0.2;
const NOVELTY_STALL_RUN: usize = 8;

/// Backoff policy for repeated-thinking re-sampling.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThinkingLoopRetry {
	/// Delay before the first guarded re-sample.
	pub base_delay: Duration,
	/// Upper bound for exponential retry delays.
	pub max_delay:  Duration,
}

impl Default for ThinkingLoopRetry {
	fn default() -> Self {
		Self { base_delay: Duration::from_millis(500), max_delay: Duration::from_secs(8) }
	}
}

/// Heals leaked reasoning and tool markup found in visible text deltas.
///
/// Repair is selected exclusively by [`Compat::leaked_thinking_healer`]. With
/// [`LeakedThinkingHealer::None`] the input events, indexes, and byte buffers
/// pass through untouched. Enabled streams are re-indexed because one visible
/// input part may expand into several canonical parts.

/// Heap-pinned stream returned by [`heal`].
pub type HealStream<S: Stream<Item = TurnEvent>> = impl Stream<Item = TurnEvent> + Unpin;
/// Repairs leaked reasoning and tool markup in `stream` according to `compat`.
#[define_opaque(HealStream)]
pub fn heal<S>(stream: S, compat: Compat) -> HealStream<S>
where
	S: Stream<Item = TurnEvent>,
{
	Box::pin(stream! {
		if compat.leaked_thinking_healer == LeakedThinkingHealer::None {
			pin_mut!(stream);
			while let Some(event) = stream.next().await {
				yield event;
			}
			return;
		}

		let mut healer = StreamHealer::new(compat.leaked_thinking_healer);
		pin_mut!(stream);
		while let Some(event) = stream.next().await {
			for healed in healer.push(event) {
				yield healed;
			}
		}
		for healed in healer.finish() {
			yield healed;
		}
	})
}

/// Stops one stream attempt with a classified error when reasoning makes no
/// progress.
///
/// The detector is active only when [`Compat::thinking_loop_guard`] is true.
/// Detection drops the wrapped stream before yielding the terminal error,
/// structurally aborting the upstream request.

/// Heap-pinned stream returned by [`guard_thinking_loop`].
pub type ThinkingLoopGuardStream<S: Stream<Item = TurnEvent>> =
	impl Stream<Item = TurnEvent> + Unpin;
/// Stops `stream` with one terminal error when guarded reasoning stops
/// progressing.
#[define_opaque(ThinkingLoopGuardStream)]
pub fn guard_thinking_loop<S>(stream: S, compat: Compat) -> ThinkingLoopGuardStream<S>
where
	S: Stream<Item = TurnEvent>,
{
	Box::pin(stream! {
		if !compat.thinking_loop_guard {
			pin_mut!(stream);
			while let Some(event) = stream.next().await {
				yield event;
			}
			return;
		}

		let mut detector = EventLoopDetector::default();
		let detail = {
			pin_mut!(stream);
			let mut hit = None;
			while let Some(event) = stream.next().await {
				if let Some(reason) = detector.push(&event) {
					hit = Some(reason);
					break;
				}
				yield event;
			}
			hit
		};
		if let Some(detail) = detail {
			yield loop_error(&detail);
		}
	})
}

/// Re-samples a guarded stream after repeated-thinking stalls, then performs a
/// final cook pass.
///
/// `resample` receives whether the newly dispatched attempt must be guarded. At
/// most three attempts (including `stream`) are aborted by the guard. If all
/// three loop, the third call to `resample` receives `false`; that final
/// attempt is forwarded without detection rather than failing the turn.
/// Dropping the returned stream drops whichever attempt is currently live.

/// Heap-pinned stream returned by [`guard_thinking_loop_with_resampling`].
pub type ThinkingLoopResamplingStream<S: Stream<Item = TurnEvent>, F: FnMut(bool) -> S> =
	impl Stream<Item = TurnEvent> + Unpin;
/// Re-samples guarded attempts after repeated-thinking stalls.
#[define_opaque(ThinkingLoopResamplingStream)]
pub fn guard_thinking_loop_with_resampling<S, F>(
	stream: S,
	compat: Compat,
	retry: ThinkingLoopRetry,
	mut resample: F,
) -> ThinkingLoopResamplingStream<S, F>
where
	S: Stream<Item = TurnEvent>,
	F: FnMut(bool) -> S,
{
	Box::pin(stream! {
		if !compat.thinking_loop_guard {
			pin_mut!(stream);
			while let Some(event) = stream.next().await {
				yield event;
			}
			return;
		}

		let mut current = stream;
		let mut guarded_attempt = 1_u32;
		loop {
			let guarded = guarded_attempt <= THINKING_LOOP_MAX_GUARDED_ATTEMPTS;
			let hit = {
				let attempt = current;
				pin_mut!(attempt);
				let mut detector = EventLoopDetector::default();
				let mut reason = None;
				while let Some(event) = attempt.next().await {
					if guarded
						&& let Some(detail) = detector.push(&event) {
							reason = Some(detail);
							break;
						}
					yield event;
				}
				reason
			};

			let Some(detail) = hit else { break };
			let next_is_guarded = guarded_attempt < THINKING_LOOP_MAX_GUARDED_ATTEMPTS;
			let next_number = guarded_attempt + 1;
			yield TurnEvent::Attempt {
				number: next_number,
				reason: SmolStr::new(format!("thinking loop: {detail}")),
			};
			if next_is_guarded {
				let shift = guarded_attempt.saturating_sub(1).min(31);
				let factor = 1_u32 << shift;
				let delay = retry.base_delay.saturating_mul(factor).min(retry.max_delay);
				if !delay.is_zero() {
					tokio::time::sleep(delay).await;
				}
			}
			current = resample(next_is_guarded);
			guarded_attempt = next_number;
		}
	})
}

/// Enforces provider-specific first-event and inter-event timeout bounds.
///
/// Both bounds come from [`Compat::stream_watchdog`]. `None` disables that
/// phase. A timeout drops the upstream stream before yielding one classified
/// terminal error.

/// Heap-pinned stream returned by [`watchdog`].
pub type WatchdogStream<S: Stream<Item = TurnEvent>> = impl Stream<Item = TurnEvent> + Unpin;
/// Applies provider-specific first-event and inter-event timeout bounds.
#[define_opaque(WatchdogStream)]
pub fn watchdog<S>(stream: S, compat: Compat) -> WatchdogStream<S>
where
	S: Stream<Item = TurnEvent>,
{
	Box::pin(stream! {
		let policy = compat.stream_watchdog;
		if policy.first_event_ms.is_none() && policy.idle_ms.is_none() {
			pin_mut!(stream);
			while let Some(event) = stream.next().await {
				yield event;
			}
			return;
		}

		pin_mut!(stream);
		let mut first = true;
		loop {
			let bound = if first { policy.first_event_ms } else { policy.idle_ms };
			let next = if let Some(ms) = bound {
				if let Ok(event) = tokio::time::timeout(Duration::from_millis(ms), stream.next()).await { event } else {
							 let detail = if first {
								 "first-event watchdog timeout"
							 } else {
								 "inter-event idle watchdog timeout"
							 };
							 yield timeout_error(detail);
							 break;
						 }
			} else {
				stream.next().await
			};
			let Some(event) = next else { break };
			first = false;
			yield event;
		}
	})
}

fn timeout_error(detail: &'static str) -> TurnEvent {
	TurnEvent::Error(
		TurnError::builder()
			.kind(TurnErrorKind::Upstream)
			.detail(SmolStr::new_static(detail))
			.unsupported(Vec::new())
			.retry_after_ms(0)
			.build(),
	)
}

fn loop_error(detail: &str) -> TurnEvent {
	TurnEvent::Error(
		TurnError::builder()
			.kind(TurnErrorKind::Upstream)
			.detail(SmolStr::new(format!(
				"thinking loop detected: {detail}; treating as a stream stall"
			)))
			.unsupported(Vec::new())
			.retry_after_ms(0)
			.build(),
	)
}

#[derive(Debug)]
struct StreamHealer {
	mode:        LeakedThinkingHealer,
	next_index:  u32,
	text:        SmallVec<TextProjection, 4>,
	passthrough: SmallVec<(u32, u32), 8>,
}

impl StreamHealer {
	const fn new(mode: LeakedThinkingHealer) -> Self {
		Self { mode, next_index: 0, text: SmallVec::new(), passthrough: SmallVec::new() }
	}

	fn push(&mut self, event: TurnEvent) -> SmallVec<TurnEvent, 8> {
		let mut out = SmallVec::new();
		match event {
			TurnEvent::PartStart { index, kind: StreamPartKind::Text, .. } => {
				self.text.push(TextProjection::new(index, self.mode));
			},
			TurnEvent::PartStart { index, kind, tool_call_id, tool_name } => {
				let projected = self.allocate_index();
				self.passthrough.push((index, projected));
				out.push(TurnEvent::PartStart { index: projected, kind, tool_call_id, tool_name });
			},
			TurnEvent::PartDelta { index, chunk } => {
				if let Some(position) = self.text.iter().position(|part| part.input_index == index) {
					let next = &mut self.next_index;
					self.text[position].push(chunk, next, &mut out);
				} else {
					out.push(TurnEvent::PartDelta { index: self.projected_index(index), chunk });
				}
			},
			TurnEvent::PartEnd { index, signature } => {
				if let Some(position) = self.text.iter().position(|part| part.input_index == index) {
					let mut part = self.text.remove(position);
					part.finish(&mut self.next_index, &mut out);
				} else {
					out.push(TurnEvent::PartEnd { index: self.projected_index(index), signature });
					if let Some(position) = self
						.passthrough
						.iter()
						.position(|(source, _)| *source == index)
					{
						self.passthrough.remove(position);
					}
				}
			},
			terminal @ (TurnEvent::Outcome(_) | TurnEvent::Error(_)) => {
				self.flush_text(&mut out);
				out.push(terminal);
			},
			other => out.push(other),
		}
		out
	}

	fn finish(&mut self) -> SmallVec<TurnEvent, 8> {
		let mut out = SmallVec::new();
		self.flush_text(&mut out);
		out
	}

	fn flush_text(&mut self, out: &mut SmallVec<TurnEvent, 8>) {
		for mut part in self.text.drain(..) {
			part.finish(&mut self.next_index, out);
		}
	}

	const fn allocate_index(&mut self) -> u32 {
		let index = self.next_index;
		self.next_index = self.next_index.saturating_add(1);
		index
	}

	fn projected_index(&self, source: u32) -> u32 {
		self
			.passthrough
			.iter()
			.find_map(|(input, output)| (*input == source).then_some(*output))
			.unwrap_or(source)
	}
}

#[derive(Debug)]
struct TextProjection {
	input_index: u32,
	scanner:     MarkupScanner,
	open:        Option<(u32, StreamPartKind)>,
}

impl TextProjection {
	fn new(input_index: u32, mode: LeakedThinkingHealer) -> Self {
		Self { input_index, scanner: MarkupScanner::new(mode), open: None }
	}

	fn push(&mut self, chunk: Bytes, next: &mut u32, out: &mut SmallVec<TurnEvent, 8>) {
		let events = self.scanner.push(chunk);
		self.emit(events, next, out);
	}

	fn finish(&mut self, next: &mut u32, out: &mut SmallVec<TurnEvent, 8>) {
		let events = self.scanner.finish();
		self.emit(events, next, out);
		self.close(out);
	}

	fn emit(
		&mut self,
		events: SmallVec<MarkupEvent, 8>,
		next: &mut u32,
		out: &mut SmallVec<TurnEvent, 8>,
	) {
		for event in events {
			match event {
				MarkupEvent::Text(chunk) => self.delta(StreamPartKind::Text, chunk, next, out),
				MarkupEvent::ThinkingStart => {
					self.close(out);
					self.start(
						StreamPartKind::Thinking,
						SmolStr::default(),
						SmolStr::default(),
						next,
						out,
					);
				},
				MarkupEvent::Thinking(chunk) => self.delta(StreamPartKind::Thinking, chunk, next, out),
				MarkupEvent::ThinkingEnd => self.close(out),
				MarkupEvent::Tool { name, arguments } => {
					self.close(out);
					let id = SmolStr::new(CallId::new().to_string());
					self.start(StreamPartKind::ToolCall, id, name, next, out);
					if !arguments.is_empty() {
						let index = self.open.expect("tool part was just opened").0;
						out.push(TurnEvent::PartDelta { index, chunk: arguments });
					}
					self.close(out);
				},
			}
		}
	}

	fn delta(
		&mut self,
		kind: StreamPartKind,
		chunk: Bytes,
		next: &mut u32,
		out: &mut SmallVec<TurnEvent, 8>,
	) {
		if chunk.is_empty() {
			return;
		}
		if self.open.is_none_or(|(_, current)| current != kind) {
			self.close(out);
			self.start(kind, SmolStr::default(), SmolStr::default(), next, out);
		}
		let index = self.open.expect("part was just opened").0;
		out.push(TurnEvent::PartDelta { index, chunk });
	}

	fn start(
		&mut self,
		kind: StreamPartKind,
		tool_call_id: SmolStr,
		tool_name: SmolStr,
		next: &mut u32,
		out: &mut SmallVec<TurnEvent, 8>,
	) {
		let index = *next;
		*next = next.saturating_add(1);
		self.open = Some((index, kind));
		out.push(TurnEvent::PartStart { index, kind, tool_call_id, tool_name });
	}

	fn close(&mut self, out: &mut SmallVec<TurnEvent, 8>) {
		if let Some((index, _)) = self.open.take() {
			out.push(TurnEvent::PartEnd { index, signature: Default::default() });
		}
	}
}

#[derive(Debug)]
struct MarkupScanner {
	mode:   LeakedThinkingHealer,
	buffer: BytesMut,
	state:  ScannerState,
}

#[derive(Debug)]
enum ScannerState {
	Outside,
	Code { ticks: usize },
	Thinking { close: &'static str },
	Tool { open: Bytes, close: &'static str, flavor: ToolFlavor },
}

#[derive(Clone, Copy, Debug)]
enum ToolFlavor {
	Json,
	XmlToolCall,
	XmlInvoke,
	Kimi,
	DeepSeek,
	Gemma,
}

#[derive(Debug)]
enum MarkupEvent {
	Text(Bytes),
	ThinkingStart,
	Thinking(Bytes),
	ThinkingEnd,
	Tool { name: SmolStr, arguments: Bytes },
}

#[derive(Clone, Copy, Debug)]
enum OutsideHit {
	Thinking { index: usize, open_len: usize, close: &'static str },
	Tool { index: usize, open_len: usize, close: &'static str, flavor: ToolFlavor },
	Skip { index: usize, len: usize },
	Code { index: usize, ticks: usize },
	Hold { index: usize },
	None,
}

impl MarkupScanner {
	fn new(mode: LeakedThinkingHealer) -> Self {
		Self { mode, buffer: BytesMut::new(), state: ScannerState::Outside }
	}

	fn push(&mut self, chunk: Bytes) -> SmallVec<MarkupEvent, 8> {
		self.buffer.extend_from_slice(&chunk);
		self.consume(false)
	}

	fn finish(&mut self) -> SmallVec<MarkupEvent, 8> {
		self.consume(true)
	}

	fn consume(&mut self, final_chunk: bool) -> SmallVec<MarkupEvent, 8> {
		let mut out = SmallVec::new();
		loop {
			match self.state {
				ScannerState::Outside => {
					if self.buffer.is_empty() {
						break;
					}
					match find_outside_hit(&self.buffer, self.mode, final_chunk) {
						OutsideHit::Thinking { index, open_len, close } => {
							self.emit_prefix(index, &mut out);
							self.buffer.advance(open_len);
							self.state = ScannerState::Thinking { close };
							out.push(MarkupEvent::ThinkingStart);
						},
						OutsideHit::Tool { index, open_len, close, flavor } => {
							self.emit_prefix(index, &mut out);
							let open = self.buffer.split_to(open_len).freeze();
							self.state = ScannerState::Tool { open, close, flavor };
						},
						OutsideHit::Skip { index, len } => {
							self.emit_prefix(index, &mut out);
							self.buffer.advance(len);
						},
						OutsideHit::Code { index, ticks } => {
							self.emit_prefix(index, &mut out);
							out.push(MarkupEvent::Text(self.buffer.split_to(ticks).freeze()));
							self.state = ScannerState::Code { ticks };
						},
						OutsideHit::Hold { index } => {
							self.emit_prefix(index, &mut out);
							break;
						},
						OutsideHit::None => {
							out.push(MarkupEvent::Text(self.buffer.split().freeze()));
							break;
						},
					}
				},
				ScannerState::Code { ticks } => {
					let delimiter = vec![b'`'; ticks];
					if let Some(index) = find_bytes(&self.buffer, &delimiter) {
						out.push(MarkupEvent::Text(self.buffer.split_to(index + ticks).freeze()));
						self.state = ScannerState::Outside;
						continue;
					}
					if final_chunk {
						if !self.buffer.is_empty() {
							out.push(MarkupEvent::Text(self.buffer.split().freeze()));
						}
						self.state = ScannerState::Outside;
						continue;
					}
					let hold = trailing_byte_run(&self.buffer, b'`').min(ticks.saturating_sub(1));
					let emit = self.buffer.len().saturating_sub(hold);
					self.emit_prefix(emit, &mut out);
					break;
				},
				ScannerState::Thinking { close } => {
					if let Some(index) = find_bytes(&self.buffer, close.as_bytes()) {
						if index > 0 {
							out.push(MarkupEvent::Thinking(self.buffer.split_to(index).freeze()));
						}
						self.buffer.advance(close.len());
						out.push(MarkupEvent::ThinkingEnd);
						self.state = ScannerState::Outside;
						continue;
					}
					let hold = if final_chunk {
						0
					} else {
						partial_suffix(&self.buffer, close.as_bytes())
					};
					let emit = self.buffer.len().saturating_sub(hold);
					if emit > 0 {
						out.push(MarkupEvent::Thinking(self.buffer.split_to(emit).freeze()));
					}
					if final_chunk {
						out.push(MarkupEvent::ThinkingEnd);
						self.state = ScannerState::Outside;
						continue;
					}
					break;
				},
				ScannerState::Tool { ref open, close, flavor } => {
					let Some(index) = find_bytes(&self.buffer, close.as_bytes()) else {
						if final_chunk {
							let mut literal = BytesMut::with_capacity(open.len() + self.buffer.len());
							literal.extend_from_slice(open);
							literal.extend_from_slice(&self.buffer.split());
							out.push(MarkupEvent::Text(literal.freeze()));
							self.state = ScannerState::Outside;
							continue;
						}
						break;
					};
					let body = self.buffer.split_to(index).freeze();
					self.buffer.advance(close.len());
					let open = open.clone();
					if let Some((name, arguments)) = parse_tool(flavor, &open, &body) {
						out.push(MarkupEvent::Tool { name, arguments });
					} else {
						let mut literal = BytesMut::with_capacity(open.len() + body.len() + close.len());
						literal.extend_from_slice(&open);
						literal.extend_from_slice(&body);
						literal.extend_from_slice(close.as_bytes());
						out.push(MarkupEvent::Text(literal.freeze()));
					}
					self.state = ScannerState::Outside;
				},
			}
		}
		out
	}

	fn emit_prefix(&mut self, len: usize, out: &mut SmallVec<MarkupEvent, 8>) {
		if len > 0 {
			out.push(MarkupEvent::Text(self.buffer.split_to(len).freeze()));
		}
	}
}

fn find_outside_hit(buffer: &[u8], mode: LeakedThinkingHealer, final_chunk: bool) -> OutsideHit {
	const THINKING: &[(&str, &str)] = &[
		("<think>", "</think>"),
		("<thinking>", "</thinking>"),
		("<scratchpad>", "</scratchpad>"),
		("```thinking\n", "```"),
		("<|channel>thought\n", "<channel|>"),
		("<|start|>assistant<|channel|>analysis<|message|>", "<|end|>"),
		("<|channel|>analysis<|message|>", "<|end|>"),
	];
	const TOOLS: &[(&str, &str, ToolFlavor)] = &[
		("```xml\n<tool_call>", "</tool_call>\n```", ToolFlavor::XmlToolCall),
		("<tool_call>", "</tool_call>", ToolFlavor::XmlToolCall),
		("```tool_call\n", "```", ToolFlavor::Json),
		("```tool\n", "```", ToolFlavor::Json),
		("<|tool_call>", "<tool_call|>", ToolFlavor::Gemma),
	];
	const KIMI_CALL: (&str, &str, ToolFlavor) =
		("<|tool_call_begin|>", "<|tool_call_end|>", ToolFlavor::Kimi);
	const DEEPSEEK_CALL: (&str, &str, ToolFlavor) =
		("<｜tool▁call▁begin｜>", "<｜tool▁call▁end｜>", ToolFlavor::DeepSeek);
	const SKIPS: &[&str] = &[
		"<|tool_calls_section_begin|>",
		"<|tool_calls_section_end|>",
		"<｜tool▁calls▁begin｜>",
		"<｜tool▁calls▁end｜>",
		"<｜DSML｜tool_calls>",
		"</｜DSML｜tool_calls>",
		"<|DSML|tool_calls>",
		"</|DSML|tool_calls>",
	];

	for index in 0..buffer.len() {
		let rest = &buffer[index..];
		for &(open, close) in THINKING {
			if rest.starts_with(open.as_bytes()) {
				return OutsideHit::Thinking { index, open_len: open.len(), close };
			}
		}
		for &(open, close, flavor) in TOOLS {
			if rest.starts_with(open.as_bytes()) {
				return OutsideHit::Tool { index, open_len: open.len(), close, flavor };
			}
		}
		if mode == LeakedThinkingHealer::Kimi && rest.starts_with(KIMI_CALL.0.as_bytes()) {
			return OutsideHit::Tool {
				index,
				open_len: KIMI_CALL.0.len(),
				close: KIMI_CALL.1,
				flavor: KIMI_CALL.2,
			};
		}
		if mode == LeakedThinkingHealer::Dsml && rest.starts_with(DEEPSEEK_CALL.0.as_bytes()) {
			return OutsideHit::Tool {
				index,
				open_len: DEEPSEEK_CALL.0.len(),
				close: DEEPSEEK_CALL.1,
				flavor: DEEPSEEK_CALL.2,
			};
		}
		if rest.starts_with(b"<invoke") && invoke_open_len(rest).is_some() {
			return OutsideHit::Tool {
				index,
				open_len: invoke_open_len(rest).expect("checked above"),
				close: "</invoke>",
				flavor: ToolFlavor::XmlInvoke,
			};
		}
		if mode == LeakedThinkingHealer::Dsml {
			for (open, close) in
				[("<｜DSML｜invoke", "</｜DSML｜invoke>"), ("<|DSML|invoke", "</|DSML|invoke>")]
			{
				if rest.starts_with(open.as_bytes())
					&& let Some(open_len) = tag_open_len(rest, open.len())
				{
					return OutsideHit::Tool { index, open_len, close, flavor: ToolFlavor::XmlInvoke };
				}
			}
		}
		if mode != LeakedThinkingHealer::Thinking {
			for skip in SKIPS {
				if rest.starts_with(skip.as_bytes()) {
					return OutsideHit::Skip { index, len: skip.len() };
				}
			}
		}
		if !final_chunk && is_partial_open(rest, mode) {
			return OutsideHit::Hold { index };
		}
		if buffer[index] == b'`' {
			let ticks = byte_run(buffer, index, b'`');
			if !final_chunk && index + ticks == buffer.len() {
				return OutsideHit::Hold { index };
			}
			return OutsideHit::Code { index, ticks };
		}
	}
	OutsideHit::None
}

fn is_partial_open(rest: &[u8], mode: LeakedThinkingHealer) -> bool {
	const COMMON: &[&str] = &[
		"<think>",
		"<thinking>",
		"<scratchpad>",
		"```thinking\n",
		"<|channel>thought\n",
		"<|start|>assistant<|channel|>analysis<|message|>",
		"<|channel|>analysis<|message|>",
		"```xml\n<tool_call>",
		"<tool_call>",
		"```tool_call\n",
		"```tool\n",
		"<|tool_call>",
		"<invoke",
	];
	if COMMON
		.iter()
		.any(|candidate| candidate.len() > rest.len() && candidate.as_bytes().starts_with(rest))
	{
		return true;
	}
	let transport: &[&str] = match mode {
		LeakedThinkingHealer::Kimi => {
			&["<|tool_calls_section_begin|>", "<|tool_calls_section_end|>", "<|tool_call_begin|>"]
		},
		LeakedThinkingHealer::Dsml => &[
			"<｜tool▁calls▁begin｜>",
			"<｜tool▁calls▁end｜>",
			"<｜tool▁call▁begin｜>",
			"<｜DSML｜tool_calls>",
			"</｜DSML｜tool_calls>",
			"<|DSML|tool_calls>",
			"</|DSML|tool_calls>",
			"<｜DSML｜invoke",
			"<|DSML|invoke",
		],
		_ => &[],
	};
	transport
		.iter()
		.any(|candidate| candidate.len() > rest.len() && candidate.as_bytes().starts_with(rest))
		|| ((rest.starts_with(b"<invoke")
			|| rest.starts_with("<｜DSML｜invoke".as_bytes())
			|| rest.starts_with(b"<|DSML|invoke"))
			&& !rest.contains(&b'>'))
}

fn invoke_open_len(rest: &[u8]) -> Option<usize> {
	if !rest.starts_with(b"<invoke") {
		return None;
	}
	if rest
		.get(7)
		.is_some_and(|byte| !byte.is_ascii_whitespace() && *byte != b'>')
	{
		return None;
	}
	tag_open_len(rest, 7)
}

fn tag_open_len(rest: &[u8], minimum: usize) -> Option<usize> {
	rest
		.get(minimum..)?
		.iter()
		.position(|byte| *byte == b'>')
		.map(|offset| minimum + offset + 1)
}

fn parse_tool(flavor: ToolFlavor, open: &Bytes, body: &Bytes) -> Option<(SmolStr, Bytes)> {
	match flavor {
		ToolFlavor::Json => parse_json_call(body),
		ToolFlavor::XmlToolCall => parse_json_call(body).or_else(|| parse_xml_tool_call(body)),
		ToolFlavor::XmlInvoke => {
			let name = parse_name_attribute(open)?;
			let range = trim_range(body);
			serde_json::from_slice::<serde_json::Value>(&body[range.clone()]).ok()?;
			Some((name, body.slice(range)))
		},
		ToolFlavor::Kimi => {
			const ARGUMENT: &[u8] = b"<|tool_call_argument_begin|>";
			let split = find_bytes(body, ARGUMENT)?;
			let raw_name = std::str::from_utf8(&body[..split]).ok()?.trim();
			let raw_name = raw_name.split(':').next().unwrap_or(raw_name);
			let name = raw_name.rsplit('.').next().unwrap_or(raw_name).trim();
			if name.is_empty() {
				return None;
			}
			let args = body.slice(split + ARGUMENT.len()..);
			let range = trim_range(&args);
			Some((SmolStr::new(name), args.slice(range)))
		},
		ToolFlavor::DeepSeek => {
			const SEPARATOR: &[u8] = "<｜tool▁sep｜>".as_bytes();
			let split = find_bytes(body, SEPARATOR)?;
			let name = std::str::from_utf8(&body[..split]).ok()?.trim();
			if name.is_empty() {
				return None;
			}
			let args = body.slice(split + SEPARATOR.len()..);
			let range = trim_range(&args);
			Some((SmolStr::new(name), args.slice(range)))
		},
		ToolFlavor::Gemma => {
			let range = trim_range(body);
			let trimmed = &body[range.clone()];
			let rest = trimmed.strip_prefix(b"call:")?;
			let brace = rest.iter().position(|byte| *byte == b'{')?;
			let name = std::str::from_utf8(&rest[..brace]).ok()?.trim();
			if name.is_empty() {
				return None;
			}
			let args_start = range.start + b"call:".len() + brace;
			let args_end = range.end;
			Some((SmolStr::new(name), body.slice(args_start..args_end)))
		},
	}
}

fn parse_xml_tool_call(body: &Bytes) -> Option<(SmolStr, Bytes)> {
	const NAME_OPEN: &[u8] = b"<name>";
	const NAME_CLOSE: &[u8] = b"</name>";
	const ARGUMENTS_OPEN: &[u8] = b"<arguments>";
	const ARGUMENTS_CLOSE: &[u8] = b"</arguments>";

	let name_start = find_bytes(body, NAME_OPEN)? + NAME_OPEN.len();
	let name_end = name_start + find_bytes(&body[name_start..], NAME_CLOSE)?;
	let name = std::str::from_utf8(&body[name_start..name_end])
		.ok()?
		.trim();
	if name.is_empty() {
		return None;
	}
	let arguments_start = name_end
		+ NAME_CLOSE.len()
		+ find_bytes(&body[name_end + NAME_CLOSE.len()..], ARGUMENTS_OPEN)?
		+ ARGUMENTS_OPEN.len();
	let arguments_end = arguments_start + find_bytes(&body[arguments_start..], ARGUMENTS_CLOSE)?;
	serde_json::from_slice::<serde_json::Value>(&body[arguments_start..arguments_end]).ok()?;
	Some((SmolStr::new(name), body.slice(arguments_start..arguments_end)))
}

fn parse_json_call(body: &Bytes) -> Option<(SmolStr, Bytes)> {
	let trimmed = trim_range(body);
	let value: serde_json::Value = serde_json::from_slice(&body[trimmed.clone()]).ok()?;
	let name = value.get("name")?.as_str()?;
	if name.is_empty() {
		return None;
	}
	let arguments = find_json_member_value(&body[trimmed.clone()], b"arguments")?;
	Some((
		SmolStr::new(name),
		body.slice(trimmed.start + arguments.start..trimmed.start + arguments.end),
	))
}

fn find_json_member_value(object: &[u8], wanted: &[u8]) -> Option<std::ops::Range<usize>> {
	let mut at = skip_ws(object, 0);
	if object.get(at)? != &b'{' {
		return None;
	}
	at += 1;
	loop {
		at = skip_ws(object, at);
		if object.get(at)? == &b'}' {
			return None;
		}
		let key_start = at;
		let key_end = scan_json_string(object, key_start)?;
		let key: SmolStr = serde_json::from_slice(&object[key_start..key_end]).ok()?;
		at = skip_ws(object, key_end);
		if object.get(at)? != &b':' {
			return None;
		}
		at = skip_ws(object, at + 1);
		let value_start = at;
		let value_end = scan_json_value(object, value_start)?;
		if key.as_bytes() == wanted {
			return Some(value_start..value_end);
		}
		at = skip_ws(object, value_end);
		match object.get(at)? {
			b',' => at += 1,
			b'}' => return None,
			_ => return None,
		}
	}
}

fn scan_json_value(input: &[u8], start: usize) -> Option<usize> {
	match *input.get(start)? {
		b'"' => scan_json_string(input, start),
		b'{' | b'[' => {
			let mut stack: SmallVec<u8, 8> = SmallVec::new();
			let mut at = start;
			while at < input.len() {
				match input[at] {
					b'"' => at = scan_json_string(input, at)?,
					b'{' => {
						stack.push(b'}');
						at += 1;
					},
					b'[' => {
						stack.push(b']');
						at += 1;
					},
					byte if stack.last() == Some(&byte) => {
						stack.pop();
						at += 1;
						if stack.is_empty() {
							return Some(at);
						}
					},
					_ => at += 1,
				}
			}
			None
		},
		_ => {
			let mut at = start;
			while at < input.len() && !matches!(input[at], b',' | b'}' | b']') {
				at += 1;
			}
			let mut end = at;
			while end > start && input[end - 1].is_ascii_whitespace() {
				end -= 1;
			}
			(end > start).then_some(end)
		},
	}
}

fn scan_json_string(input: &[u8], start: usize) -> Option<usize> {
	if input.get(start)? != &b'"' {
		return None;
	}
	let mut at = start + 1;
	while at < input.len() {
		match input[at] {
			b'\\' => at = at.checked_add(2)?,
			b'"' => return Some(at + 1),
			_ => at += 1,
		}
	}
	None
}

fn parse_name_attribute(open: &[u8]) -> Option<SmolStr> {
	let text = std::str::from_utf8(open).ok()?;
	let name = text.find("name=")? + "name=".len();
	let quote = *text.as_bytes().get(name)?;
	if !matches!(quote, b'\'' | b'"') {
		return None;
	}
	let value_start = name + 1;
	let value_end = text.as_bytes()[value_start..]
		.iter()
		.position(|byte| *byte == quote)?
		+ value_start;
	let value = &text[value_start..value_end];
	(!value.is_empty()).then(|| SmolStr::new(value))
}

fn trim_range(bytes: &[u8]) -> std::ops::Range<usize> {
	let mut start = 0;
	while bytes.get(start).is_some_and(u8::is_ascii_whitespace) {
		start += 1;
	}
	let mut end = bytes.len();
	while end > start && bytes[end - 1].is_ascii_whitespace() {
		end -= 1;
	}
	start..end
}

fn skip_ws(bytes: &[u8], mut at: usize) -> usize {
	while bytes.get(at).is_some_and(u8::is_ascii_whitespace) {
		at += 1;
	}
	at
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
	if needle.is_empty() {
		return Some(0);
	}
	haystack
		.windows(needle.len())
		.position(|window| window == needle)
}

fn partial_suffix(buffer: &[u8], delimiter: &[u8]) -> usize {
	let maximum = buffer.len().min(delimiter.len().saturating_sub(1));
	(1..=maximum)
		.rev()
		.find(|&length| delimiter.starts_with(&buffer[buffer.len() - length..]))
		.unwrap_or(0)
}

fn byte_run(bytes: &[u8], start: usize, byte: u8) -> usize {
	bytes[start..]
		.iter()
		.take_while(|candidate| **candidate == byte)
		.count()
}

fn trailing_byte_run(bytes: &[u8], byte: u8) -> usize {
	bytes
		.iter()
		.rev()
		.take_while(|candidate| **candidate == byte)
		.count()
}

#[derive(Debug, Default)]
struct EventLoopDetector {
	thinking:       LoopDetector,
	text:           LoopDetector,
	parts:          SmallVec<(u32, StreamPartKind), 8>,
	thinking_armed: bool,
	text_armed:     bool,
}

impl EventLoopDetector {
	fn push(&mut self, event: &TurnEvent) -> Option<SmolStr> {
		match event {
			TurnEvent::PartStart { index, kind, .. } => {
				self.parts.push((*index, *kind));
				match kind {
					StreamPartKind::Thinking => {
						self.thinking_armed = true;
						self.thinking = LoopDetector::default();
					},
					StreamPartKind::Text => {
						self.thinking_armed = false;
						self.text_armed = true;
					},
					StreamPartKind::ToolCall => {
						self.thinking_armed = false;
						self.text_armed = false;
					},
					_ => {
						self.thinking_armed = false;
						self.text_armed = false;
					},
				}
				None
			},
			TurnEvent::PartDelta { index, chunk } => match self.kind(*index) {
				Some(StreamPartKind::Thinking) if self.thinking_armed => self.thinking.push(chunk),
				Some(StreamPartKind::Text) if self.text_armed => self.text.push(chunk),
				_ => None,
			},
			TurnEvent::PartEnd { index, .. } => {
				let kind = self.kind(*index);
				if let Some(position) = self.parts.iter().position(|(part, _)| part == index) {
					self.parts.remove(position);
				}
				match kind {
					Some(StreamPartKind::Thinking) if self.thinking_armed => {
						self.thinking_armed = false;
						self.thinking.flush()
					},
					Some(StreamPartKind::Text) if self.text_armed => self.text.flush(),
					_ => None,
				}
			},
			TurnEvent::Outcome(_) => self.thinking.flush().or_else(|| self.text.flush()),
			_ => None,
		}
	}

	fn kind(&self, index: u32) -> Option<StreamPartKind> {
		self
			.parts
			.iter()
			.rev()
			.find_map(|(part, kind)| (*part == index).then_some(*kind))
	}
}

#[derive(Debug, Default)]
struct LoopDetector {
	tail:          String,
	pending:       String,
	recent:        SmallVec<FxHashSet<SmolStr>, 16>,
	vocab:         SmallVec<FxHashSet<SmolStr>, 8>,
	segment_count: usize,
	stall_run:     usize,
}

impl LoopDetector {
	fn push(&mut self, delta: &[u8]) -> Option<SmolStr> {
		let delta = std::str::from_utf8(delta).ok()?;
		if delta.is_empty() {
			return None;
		}
		self.tail.push_str(delta);
		if self.tail.len() > VERBATIM_TAIL_WINDOW {
			let mut start = self.tail.len() - VERBATIM_TAIL_WINDOW;
			while !self.tail.is_char_boundary(start) {
				start += 1;
			}
			self.tail.drain(..start);
		}
		if let Some((unit, count)) = detect_verbatim_repetition(self.tail.as_bytes()) {
			return Some(SmolStr::new(format!(
				"repeated {:?} {count} times back-to-back",
				String::from_utf8_lossy(unit).trim()
			)));
		}

		self.pending.push_str(delta);
		loop {
			let boundary = blank_line_boundary(self.pending.as_bytes());
			let take = boundary.as_ref().map_or_else(
				|| (self.pending.len() > SEGMENT_BYTE_CAP).then_some(SEGMENT_BYTE_CAP),
				|range| Some(range.start),
			);
			let mut take = take?;
			while take > 0 && !self.pending.is_char_boundary(take) {
				take -= 1;
			}
			let segment: String = self.pending.drain(..take).collect();
			if let Some(range) = boundary {
				let delimiter_len = range.end - range.start;
				self.pending.drain(..delimiter_len);
			}
			if let Some(reason) = self.consume_segment(&segment) {
				return Some(reason);
			}
		}
	}

	fn flush(&mut self) -> Option<SmolStr> {
		if self.pending.is_empty() {
			return None;
		}
		let pending = std::mem::take(&mut self.pending);
		for chunk in pending.as_bytes().chunks(SEGMENT_BYTE_CAP) {
			let segment = std::str::from_utf8(chunk).ok()?;
			if let Some(reason) = self.consume_segment(segment) {
				return Some(reason);
			}
		}
		None
	}

	fn consume_segment(&mut self, raw: &str) -> Option<SmolStr> {
		let normalized = normalize_segment(raw);
		if normalized.len() < SEGMENT_MIN_NORMALIZED_BYTES {
			return None;
		}
		let words: SmallVec<&str, 96> = normalized.split_ascii_whitespace().collect();
		let mut fingerprint = FxHashSet::default();
		if words.len() < 3 {
			fingerprint.insert(SmolStr::new(normalized.as_str()));
		} else {
			for triple in words.windows(3) {
				fingerprint.insert(SmolStr::new(format!("{} {} {}", triple[0], triple[1], triple[2])));
			}
		}
		let cluster = 1
			+ self
				.recent
				.iter()
				.filter(|prior| jaccard(&fingerprint, prior) >= SEGMENT_SIMILARITY)
				.count();

		let word_set: FxHashSet<SmolStr> = words.iter().map(SmolStr::new).collect();
		let mut prior_vocab = FxHashSet::default();
		for prior in &self.vocab {
			prior_vocab.extend(prior.iter().cloned());
		}
		let unseen = word_set
			.iter()
			.filter(|word| !prior_vocab.contains(*word))
			.count();
		let novelty = if prior_vocab.is_empty() {
			1.0
		} else {
			unseen as f32 / word_set.len().max(1) as f32
		};
		if novelty <= NOVELTY_FLOOR {
			self.stall_run += 1;
		} else {
			self.stall_run = 0;
		}

		self.recent.push(fingerprint);
		if self.recent.len() > SEGMENT_WINDOW {
			self.recent.remove(0);
		}
		self.vocab.push(word_set);
		if self.vocab.len() > NOVELTY_WINDOW {
			self.vocab.remove(0);
		}
		self.segment_count += 1;
		if self.segment_count >= SEGMENT_MIN_COUNT {
			if cluster >= SEGMENT_MIN_CLUSTER {
				return Some(SmolStr::new(format!(
					"{cluster} near-identical segments within the last {SEGMENT_WINDOW}"
				)));
			}
			if self.stall_run >= NOVELTY_STALL_RUN {
				return Some(SmolStr::new(format!(
					"{} low-information segments recycling recent wording",
					self.stall_run
				)));
			}
		}
		None
	}
}

fn detect_verbatim_repetition(text: &[u8]) -> Option<(&[u8], usize)> {
	if text.len() < VERBATIM_MIN_REPEATED_BYTES {
		return None;
	}
	for length in 2..=VERBATIM_MAX_UNIT {
		if text.len() < length * 4 {
			continue;
		}
		let unit = &text[text.len() - length..];
		if !unit.iter().any(u8::is_ascii_alphabetic) {
			continue;
		}
		let mut count = 0;
		let mut end = text.len();
		while end >= length && &text[end - length..end] == unit {
			count += 1;
			end -= length;
		}
		if count >= 4 && count * length >= VERBATIM_MIN_REPEATED_BYTES {
			return Some((unit, count));
		}
	}
	None
}

fn blank_line_boundary(text: &[u8]) -> Option<std::ops::Range<usize>> {
	let mut at = 0;
	while at < text.len() {
		if text[at] == b'\n' {
			let start = at;
			at += 1;
			while text
				.get(at)
				.is_some_and(|byte| matches!(byte, b' ' | b'\t' | b'\r'))
			{
				at += 1;
			}
			if text.get(at) == Some(&b'\n') {
				return Some(start..at + 1);
			}
		}
		at += 1;
	}
	None
}

fn normalize_segment(segment: &str) -> String {
	let mut normalized = String::with_capacity(segment.len());
	let mut prior_space = true;
	for byte in segment.bytes() {
		if byte.is_ascii_alphanumeric() {
			if byte.is_ascii_alphabetic() {
				normalized.push(byte.to_ascii_lowercase() as char);
			} else {
				normalized.push(byte as char);
			}
			prior_space = false;
		} else if !prior_space {
			normalized.push(' ');
			prior_space = true;
		}
	}
	if normalized.ends_with(' ') {
		normalized.pop();
	}
	normalized
}

fn jaccard(left: &FxHashSet<SmolStr>, right: &FxHashSet<SmolStr>) -> f32 {
	if left.is_empty() || right.is_empty() {
		return 0.0;
	}
	let intersection = left.iter().filter(|item| right.contains(*item)).count();
	let union = left.len() + right.len() - intersection;
	intersection as f32 / union as f32
}

#[cfg(test)]
mod tests {
	use std::sync::Arc;

	use futures::{StreamExt, stream};
	use parking_lot::Mutex;

	use super::*;

	fn text_stream(
		chunks: impl IntoIterator<Item = &'static [u8]>,
	) -> impl Stream<Item = TurnEvent> {
		let mut events = Vec::new();
		events.push(TurnEvent::PartStart {
			index:        7,
			kind:         StreamPartKind::Text,
			tool_call_id: SmolStr::default(),
			tool_name:    SmolStr::default(),
		});
		for chunk in chunks {
			events.push(TurnEvent::PartDelta { index: 7, chunk: Bytes::from_static(chunk) });
		}
		events.push(TurnEvent::PartEnd { index: 7, signature: Default::default() });
		stream::iter(events)
	}

	fn healer_compat() -> Compat {
		let mut compat = Compat::default();
		compat.leaked_thinking_healer = LeakedThinkingHealer::Thinking;
		compat
	}

	#[tokio::test]
	async fn heals_think_tag_split_across_three_chunks() {
		let events: Vec<_> =
			heal(text_stream([b"<th".as_slice(), b"ink", b">reason</think>"]), healer_compat())
				.collect()
				.await;
		assert!(matches!(
			events.as_slice(),
			[
				TurnEvent::PartStart { kind: StreamPartKind::Thinking, .. },
				TurnEvent::PartDelta { chunk, .. },
				TurnEvent::PartEnd { .. }
			] if chunk.as_ref() == b"reason"
		));
	}

	#[tokio::test]
	async fn passes_literal_partial_think_tag_untouched() {
		let events: Vec<_> =
			heal(text_stream([b"literal <th".as_slice(), b"ink is text"]), healer_compat())
				.collect()
				.await;
		let bytes: Vec<u8> = events
			.iter()
			.filter_map(|event| match event {
				TurnEvent::PartDelta { chunk, .. } => Some(chunk.as_ref()),
				_ => None,
			})
			.flatten()
			.copied()
			.collect();
		assert_eq!(bytes, b"literal <think is text");
	}

	#[tokio::test]
	async fn heals_xml_wrapped_tool_call_with_verbatim_arguments() {
		let raw = b"before <tool_call>{\"name\":\"search\",\"arguments\": { \"q\": \"rust\" }}</tool_call> after";
		let events: Vec<_> = heal(text_stream([raw.as_slice()]), healer_compat())
			.collect()
			.await;
		let tool = events.iter().position(|event| {
			matches!(event, TurnEvent::PartStart { kind: StreamPartKind::ToolCall, tool_name, .. } if tool_name.as_str() == "search")
		});
		let tool = tool.expect("structured tool-call start");
		assert!(matches!(
			&events[tool + 1],
			TurnEvent::PartDelta { chunk, .. } if chunk.as_ref() == b"{ \"q\": \"rust\" }"
		));
	}

	#[tokio::test]
	async fn heals_fenced_xml_tool_call_without_fence_residue() {
		let events: Vec<_> = heal(
			text_stream([
				b"before\n``".as_slice(),
				b"`xml\n<tool_".as_slice(),
				b"call><name>read</name><arguments>{ \"path\": \"README.md\" }</arguments></tool_call>\n``"
					.as_slice(),
				b"`after".as_slice(),
			]),
			healer_compat(),
		)
		.collect()
		.await;
		let deltas: Vec<_> = events
			.iter()
			.filter_map(|event| match event {
				TurnEvent::PartDelta { chunk, .. } => Some(chunk.as_ref()),
				_ => None,
			})
			.collect();
		assert_eq!(deltas, vec![
			b"before\n".as_slice(),
			b"{ \"path\": \"README.md\" }".as_slice(),
			b"after".as_slice()
		]);
		assert!(events.iter().any(|event| {
			matches!(event, TurnEvent::PartStart { kind: StreamPartKind::ToolCall, tool_name, .. } if tool_name.as_str() == "read")
		}));
	}

	#[tokio::test]
	async fn passes_non_tool_xml_fence_untouched() {
		let raw =
			b"```xml\n<tool_call><name>read</name><arguments>not JSON</arguments></tool_call>\n```";
		let events: Vec<_> = heal(text_stream([raw.as_slice()]), healer_compat())
			.collect()
			.await;
		let bytes: Vec<u8> = events
			.iter()
			.filter_map(|event| match event {
				TurnEvent::PartDelta { chunk, .. } => Some(chunk.as_ref()),
				_ => None,
			})
			.flatten()
			.copied()
			.collect();
		assert_eq!(bytes, raw);
	}

	#[tokio::test]
	async fn loop_guard_fires_on_verbatim_repeat() {
		let unit = "reasoning-loop-unit-";
		let repeated = unit.repeat(12);
		let events = stream::iter([
			TurnEvent::PartStart {
				index:        0,
				kind:         StreamPartKind::Thinking,
				tool_call_id: SmolStr::default(),
				tool_name:    SmolStr::default(),
			},
			TurnEvent::PartDelta { index: 0, chunk: Bytes::from(repeated) },
		]);
		let mut compat = Compat::default();
		compat.thinking_loop_guard = true;
		let guarded: Vec<_> = guard_thinking_loop(events, compat).collect().await;
		assert!(matches!(
			guarded.last(),
			Some(TurnEvent::Error(error)) if error.detail.as_str().contains("thinking loop detected")
		));
	}

	#[tokio::test]
	async fn third_resample_disables_guard_and_cooks() {
		fn looping() -> impl Stream<Item = TurnEvent> {
			stream::iter([
				TurnEvent::PartStart {
					index:        0,
					kind:         StreamPartKind::Thinking,
					tool_call_id: SmolStr::default(),
					tool_name:    SmolStr::default(),
				},
				TurnEvent::PartDelta { index: 0, chunk: Bytes::from("thinking-loop-unit-".repeat(12)) },
			])
		}
		let calls = Arc::new(Mutex::new(Vec::new()));
		let observed = calls.clone();
		let mut compat = Compat::default();
		compat.thinking_loop_guard = true;
		let result: Vec<_> = guard_thinking_loop_with_resampling(
			looping(),
			compat,
			ThinkingLoopRetry { base_delay: Duration::ZERO, max_delay: Duration::ZERO },
			move |enabled| {
				observed.lock().push(enabled);
				looping()
			},
		)
		.collect()
		.await;
		assert_eq!(&*calls.lock(), &[true, true, false]);
		assert!(
			!result
				.iter()
				.any(|event| matches!(event, TurnEvent::Error(_)))
		);
	}

	#[tokio::test]
	async fn idle_watchdog_yields_classified_error() {
		let input = stream::once(async { TurnEvent::Accepted { replay: false } })
			.chain(stream::pending::<TurnEvent>());
		let mut compat = Compat::default();
		compat.stream_watchdog.idle_ms = Some(1);
		let events: Vec<_> = watchdog(input, compat).collect().await;
		assert!(matches!(
			events.as_slice(),
			[TurnEvent::Accepted { .. }, TurnEvent::Error(error)]
				if error.kind == TurnErrorKind::Upstream && error.detail.as_str().contains("idle")
		));
	}

	#[tokio::test]
	async fn healer_is_exact_noop_when_compat_is_off() {
		let original = vec![
			TurnEvent::PartStart {
				index:        7,
				kind:         StreamPartKind::Text,
				tool_call_id: SmolStr::default(),
				tool_name:    SmolStr::default(),
			},
			TurnEvent::PartDelta { index: 7, chunk: Bytes::from_static(b"<think>visible</think>") },
			TurnEvent::PartEnd { index: 7, signature: Default::default() },
		];
		let healed: Vec<_> = heal(stream::iter(original.clone()), Compat::default())
			.collect()
			.await;
		assert_eq!(healed, original);
	}
}
