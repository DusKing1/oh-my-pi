//! Prompt-cache policy: breakpoint injection and keep-alive refreshes.
//!
//! Two concerns, one layer, because they are the same decision seen from two
//! sides: what to cache, and how long to hold it.
//!
//! **Injection.** Placement is a cross-provider policy, but only a dialect can
//! act on it, so the layer writes a [`CacheHint`] onto the canonical request
//! and the codec turns it into markers. A hint the client supplied is
//! authoritative and is never overwritten.
//!
//! **Keep-alive.** A turn that stops on `STOP_TOOL_USE` is the one case where a
//! follow-up request is known to be coming: the agent is blocked on its own
//! tool, not on a human. Retention can then be rented in TTL-sized increments
//! by replaying the request and dropping the stream once the provider has
//! prefilled — the cache read is billed, the generation is not.
//!
//! Measured over a 1.2k-session replay: tool-loop returns average 5.7s and
//! 99.5% land inside a five-minute window, so pings only ever bridge the tail.
//! A ping repays itself while the conditional return rate inside the next
//! window clears `read / (write - read)`; past roughly seven pings a long-TTL
//! write is strictly cheaper, which is why [`CachePolicy::pings`] is capped.
//!
//! Keep-alive is OFF unless a route calls [`CachePolicy::with_keepalive`]: it
//! is only cheap if dropping the response stream actually cancels the upstream
//! request, which is a property of the transport beneath this layer, not of
//! this layer.

use std::{
	pin::Pin,
	sync::Arc,
	task::{Context, Poll},
	time::Duration,
};

use futures::{Stream, StreamExt, TryFutureExt};
use omp_core::Str;
use omp_proto::inference::v1::{
	CacheHint, StopReason, TurnEvent, TurnRequest, cache_hint, turn_event, turn_request,
};
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use tokio::{task::AbortHandle, time::Instant};
use tower::{Layer, Service};

use crate::envelope::TurnRequestEnvelope;

/// Largest useful ping budget: beyond this a long-retention write costs less
/// than the reads needed to reach the same lifetime.
pub const MAX_PINGS: u8 = 7;

/// Prompt-cache policy applied to requests that carry no explicit hint.
#[derive(Clone, Copy, Debug)]
pub struct CachePolicy {
	/// Breakpoint placement requested from the dialect.
	pub breakpoint: cache_hint::Breakpoint,
	/// Retention class requested from the dialect.
	pub retention:  cache_hint::Retention,
	/// Keep-alive refreshes per idle gap, clamped to [`MAX_PINGS`]. Zero
	/// disables keep-alive entirely.
	pub pings:      u8,
	/// Entry lifetime the refreshes race against.
	pub ttl:        Duration,
	/// How early to refresh before the entry lapses.
	pub lead:       Duration,
}

impl Default for CachePolicy {
	/// Inert. One [`crate::stack::builder::RouteStackConfig`] serves every
	/// provider route, so a tuned default here would push Anthropic-shaped
	/// placement onto dialects that read the same fields differently. Routes
	/// opt in with [`CachePolicy::tail_two`].
	fn default() -> Self {
		Self {
			breakpoint: cache_hint::Breakpoint::Unspecified,
			retention:  cache_hint::Retention::Unspecified,
			pings:      0,
			ttl:        Duration::from_secs(300),
			lead:       Duration::from_secs(1),
		}
	}
}

impl CachePolicy {
	/// The measured-cheapest Anthropic placement: two tail breakpoints at short
	/// retention. Keep-alive is off — see [`CachePolicy::with_keepalive`].
	///
	/// Only mount this on routes whose dialect reads
	/// [`omp_llm_types::PromptCacheBreakpoint::TailTwo`] as breakpoint
	/// placement; it is not a portable default.
	#[must_use]
	pub fn tail_two() -> Self {
		Self {
			breakpoint: cache_hint::Breakpoint::TailTwo,
			retention: cache_hint::Retention::Short,
			pings: 0,
			..Self::default()
		}
	}

	/// Enables `pings` keep-alive refreshes per idle gap. Opt-in, and kept
	/// separate from placement on purpose.
	///
	/// A refresh is only cheap if dropping the response stream after the first
	/// provider frame actually closes the upstream request — a property of the
	/// transport below this layer, which this crate does not test. If it does
	/// not hold, every ping buys a whole generation rather than a cache read,
	/// and nothing here bounds it: the Anthropic codec raises `max_tokens` to
	/// `thinking_budget + 1024`.
	///
	/// Placement is worth roughly thirty times what refreshes are worth on the
	/// replay this policy was tuned against, so this is a small optimization
	/// sitting behind an unverified cost risk. Prove the abort with a recording
	/// proxy before enabling; `1..=7` is the useful range.
	#[must_use]
	pub const fn with_keepalive(mut self, pings: u8) -> Self {
		self.pings = pings;
		self
	}

	/// Interval between refreshes, never zero even under a pathological lead.
	fn interval(self) -> Duration {
		self
			.ttl
			.checked_sub(self.lead)
			.unwrap_or(self.ttl)
			.max(Duration::from_secs(1))
	}

	fn budget(self) -> u8 {
		self.pings.min(MAX_PINGS)
	}

	/// Whether this route opted into a policy at all. The default is inert, so
	/// a route that never enabled caching is left byte-identical.
	const fn active(self) -> bool {
		!matches!(self.breakpoint, cache_hint::Breakpoint::Unspecified)
			|| !matches!(self.retention, cache_hint::Retention::Unspecified)
			|| self.pings > 0
	}
}

/// Writes the policy onto a request that did not already carry one.
///
/// Returns the conversation key the keep-alive registry is keyed by. A request
/// without one cannot be kept warm: there is nothing to cancel it against.
/// The conversation this turn belongs to, if it belongs to one.
///
/// `context_id` is a client-minted ULID namespaced per authenticated client, so
/// it is stable across the turns of one conversation and cannot collide across
/// clients — exactly the identity a prompt cache wants. A fully stateless turn
/// has none.
fn conversation(request: &TurnRequest) -> Option<Str> {
	let id = match request.input.as_ref()? {
		turn_request::Input::Incremental(incremental) => {
			incremental.context.as_ref()?.context_id.as_str()
		},
		turn_request::Input::Seed(seed) => seed.context_id.as_str(),
	};
	(!id.is_empty()).then(|| Str::new(id))
}

fn apply(request: &mut TurnRequest, policy: CachePolicy) -> Option<Str> {
	// An inert policy must not even create the hint: its mere presence is
	// meaningful to other dialects, and this layer is mounted on every route.
	if !policy.active() {
		return None;
	}
	let conversation = conversation(request);
	let params = request.params.as_mut()?;
	let hint = params.cache.get_or_insert_with(CacheHint::default);
	// An explicit client policy wins: it may know the conversation is about to
	// be discarded, or that this prefix is not worth a breakpoint at all.
	if hint.breakpoint == cache_hint::Breakpoint::Unspecified as i32 {
		hint.breakpoint = policy.breakpoint as i32;
	}
	if hint.retention == cache_hint::Retention::Unspecified as i32 {
		hint.retention = policy.retention as i32;
	}
	// The client owns conversation identity when it supplies one; otherwise the
	// turn's own context id is the same thing under a different name.
	if hint.session_key.is_empty()
		&& let Some(id) = &conversation
	{
		hint.session_key = id.to_string();
	}
	// A stateless turn still gets its breakpoints, but nothing to key a refresh
	// loop on — and nothing that would be worth keeping warm anyway.
	(!hint.session_key.is_empty()).then(|| Str::new(&hint.session_key))
}
/// Whether this frame ends the turn because the model called a tool.
fn stopped_on_tool_use(frame: &TurnEvent) -> bool {
	matches!(
		&frame.event,
		Some(turn_event::Event::Outcome(outcome)) if outcome.stop == StopReason::StopToolUse as i32
	)
}

/// Whether this frame could only have been authored by the provider, which
/// means the prompt was prefilled and the cache read is already billed.
fn provider_answered(frame: &TurnEvent) -> bool {
	matches!(
		&frame.event,
		Some(
			turn_event::Event::PartStart(_)
				| turn_event::Event::PartDelta(_)
				| turn_event::Event::PartEnd(_)
				| turn_event::Event::Outcome(_)
				| turn_event::Event::Error(_)
				| turn_event::Event::Invoke(_)
		)
	)
}

/// Keep-alive tasks in flight, keyed by conversation.
type Registry = Arc<Mutex<FxHashMap<Str, AbortHandle>>>;

/// [`Layer`] producing [`Cache`] services.
#[derive(Clone)]
pub struct CacheLayer {
	policy:   CachePolicy,
	registry: Registry,
}

impl CacheLayer {
	/// Layer applying `policy` to requests without an explicit cache hint.
	#[must_use]
	pub fn new(policy: CachePolicy) -> Self {
		Self { policy, registry: Registry::default() }
	}
}

impl<S> Layer<S> for CacheLayer {
	type Service = Cache<S>;

	fn layer(&self, inner: S) -> Self::Service {
		Cache { inner, policy: self.policy, registry: Arc::clone(&self.registry) }
	}
}

/// Cache-policy wrapper around an attempt service.
#[derive(Clone)]
pub struct Cache<S> {
	inner:    S,
	policy:   CachePolicy,
	registry: Registry,
}

impl<S> Cache<S> {
	/// Wraps `inner`, applying `policy`.
	pub fn new(inner: S, policy: CachePolicy) -> Self {
		Self { inner, policy, registry: Registry::default() }
	}
}

pin_project_lite::pin_project! {
	/// Stream that arms a keep-alive once the turn stops on a tool call.
	pub struct Warmed<St, S, R> {
		#[pin]
		inner: St,
		armer: Option<Armer<S, R>>,
		prefill: Option<Instant>,
	}
}

impl<St, S, R> Stream for Warmed<St, S, R>
where
	St: Stream<Item = TurnEvent> + Send + 'static,
	S: Service<R, Response = St> + Clone + Send + 'static,
	S::Future: Send,
	R: TurnRequestEnvelope,
{
	type Item = TurnEvent;

	fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<TurnEvent>> {
		let this = self.project();
		let polled = this.inner.poll_next(cx);
		if let Poll::Ready(Some(frame)) = &polled {
			// The entry's lifetime starts when the provider wrote it, which the
			// first provider-authored frame dates. Anchoring the refresh to the
			// terminal frame instead would lose the whole generation window:
			// a turn that thinks for a minute would schedule its first ping a
			// minute after the entry had already lapsed.
			if this.prefill.is_none() && provider_answered(frame) {
				*this.prefill = Some(Instant::now());
			}
			// `Outcome` is terminal, so the idle gap starts on this frame.
			// Arming here rather than at stream end covers consumers that stop
			// reading as soon as they have the outcome.
			if stopped_on_tool_use(frame)
				&& let Some(armer) = this.armer.take()
			{
				armer.arm(*this.prefill);
			}
		}
		polled
	}
}

/// Everything a finished turn needs to keep its prefix warm.
struct Armer<S, R> {
	inner:    S,
	request:  R,
	key:      Str,
	policy:   CachePolicy,
	registry: Registry,
}

impl<S, St, R> Armer<S, R>
where
	S: Service<R, Response = St> + Clone + Send + 'static,
	S::Future: Send,
	St: Stream<Item = TurnEvent> + Send + 'static,
	R: TurnRequestEnvelope,
{
	/// Spawns the refresh loop and registers it for cancellation.
	///
	/// `prefill` dates the cache entry. The first refresh is due `interval`
	/// after *that*, not after the turn finished, so the generation window is
	/// subtracted; a turn that outran the TTL refreshes immediately.
	fn arm(self, prefill: Option<Instant>) {
		let Self { mut inner, request, key, policy, registry } = self;
		let interval = policy.interval();
		let budget = policy.budget();
		let mut delay = prefill.map_or(interval, |at| interval.saturating_sub(at.elapsed()));
		let handle = tokio::spawn(async move {
			for _ in 0..budget {
				tokio::time::sleep(delay).await;
				// Each refresh re-dates the entry at its own prefill, and it
				// generates nothing, so later gaps are the full interval.
				delay = interval;
				if std::future::poll_fn(|cx| inner.poll_ready(cx))
					.await
					.is_err()
				{
					return;
				}
				// Replaying the request verbatim is what makes the prefix match.
				let Ok(stream) = inner.call(request.clone()).await else {
					return;
				};
				// Hit and cancel. The cache read is billed, and the entry's
				// lifetime extended, once the provider has prefilled — which
				// the first provider-authored frame proves. Dropping the stream
				// there drops the hyper body, which aborts the upstream
				// request, so the generation is never paid for.
				let mut stream = std::pin::pin!(stream);
				while let Some(frame) = stream.next().await {
					if provider_answered(&frame) {
						break;
					}
				}
			}
		})
		.abort_handle();
		if let Some(previous) = registry.lock().insert(key, handle) {
			previous.abort();
		}
	}
}

impl<S, St, R> Service<R> for Cache<S>
where
	R: TurnRequestEnvelope,
	S: Service<R, Response = St> + Clone + Send + 'static,
	S::Future: Send,
	S::Error: Send + 'static,
	St: Stream<Item = TurnEvent> + Send + 'static,
{
	type Error = S::Error;
	type Response = Warmed<St, S, R>;

	type Future = impl Future<Output = Result<Self::Response, S::Error>> + Send;

	fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), S::Error>> {
		self.inner.poll_ready(cx)
	}

	fn call(&mut self, mut req: R) -> Self::Future {
		let key = apply(req.request_mut(), self.policy);
		if let Some(key) = &key {
			// A real request supersedes any refresh loop for this conversation:
			// it re-reads the prefix itself, and a stale loop would race it.
			if let Some(handle) = self.registry.lock().remove(key) {
				handle.abort();
			}
		}

		let clone = self.inner.clone();
		let mut inner = std::mem::replace(&mut self.inner, clone);
		// The refresh loop dispatches through a clone of the INNER service, so
		// a ping never re-enters this layer, never re-arms itself, and never
		// cancels the loop it belongs to.
		let armer = key.filter(|_| self.policy.budget() > 0).map(|key| Armer {
			inner: inner.clone(),
			request: req.clone(),
			key,
			policy: self.policy,
			registry: Arc::clone(&self.registry),
		});
		inner
			.call(req)
			.map_ok(move |stream| Warmed { inner: stream, armer, prefill: None })
	}
}

#[cfg(test)]
mod tests {
	use std::time::Duration;

	use futures::StreamExt;
	use omp_proto::inference::v1::{
		ChatParams, Outcome, PartDelta, Seed, StopReason, cache_hint, turn_request,
	};
	use tower::{Service, ServiceExt};

	use super::*;
	use crate::testing::{Script, ev};

	fn request(session: &str, breakpoint: cache_hint::Breakpoint) -> TurnRequest {
		TurnRequest {
			params: Some(ChatParams {
				cache: Some(CacheHint {
					session_key: session.into(),
					breakpoint: breakpoint as i32,
					..CacheHint::default()
				}),
				..ChatParams::default()
			}),
			..TurnRequest::default()
		}
	}

	fn seen(script: &Script, index: usize) -> CacheHint {
		script.calls.lock()[index]
			.params
			.as_ref()
			.unwrap()
			.cache
			.clone()
			.unwrap()
	}

	fn stopped(stop: StopReason) -> Vec<TurnEvent> {
		vec![ev(turn_event::Event::Outcome(Outcome { stop: stop as i32, ..Outcome::default() }))]
	}

	async fn drain<S>(service: &mut S, req: TurnRequest)
	where
		S: Service<TurnRequest>,
		S::Response: Stream<Item = TurnEvent> + Unpin,
		S::Error: std::fmt::Debug,
	{
		let mut stream = service.ready().await.unwrap().call(req).await.unwrap();
		while stream.next().await.is_some() {}
	}

	#[tokio::test]
	async fn policy_fills_gaps_without_overriding_an_explicit_client_choice() {
		let script =
			Script::new([stopped(StopReason::StopEndTurn), stopped(StopReason::StopEndTurn)]);
		let mut service = Cache::new(script.clone(), CachePolicy::tail_two());

		drain(&mut service, request("a", cache_hint::Breakpoint::Unspecified)).await;
		drain(&mut service, request("b", cache_hint::Breakpoint::None)).await;

		let calls = script.calls.lock();
		let hint = |index: usize| {
			calls[index]
				.params
				.as_ref()
				.unwrap()
				.cache
				.as_ref()
				.unwrap()
				.clone()
		};
		assert_eq!(hint(0).breakpoint, cache_hint::Breakpoint::TailTwo as i32);
		assert_eq!(hint(0).retention, cache_hint::Retention::Short as i32);
		// An explicit suppression survives: the client may know this prefix is
		// about to be discarded.
		assert_eq!(hint(1).breakpoint, cache_hint::Breakpoint::None as i32);
	}

	#[tokio::test(start_paused = true)]
	async fn tool_use_arms_refreshes_and_a_real_request_cancels_them() {
		let script = Script::new([stopped(StopReason::StopToolUse)]);
		let policy =
			CachePolicy { pings: 2, ttl: Duration::from_secs(300), ..CachePolicy::default() };
		let mut service = Cache::new(script.clone(), policy);

		drain(&mut service, request("session", cache_hint::Breakpoint::Unspecified)).await;
		assert_eq!(script.calls.lock().len(), 1, "no refresh before the window closes");

		tokio::time::sleep(Duration::from_secs(300)).await;
		tokio::task::yield_now().await;
		assert_eq!(script.calls.lock().len(), 2, "one refresh once the window lapsed");

		// The next real turn re-reads the prefix itself, so the loop must stop.
		drain(&mut service, request("session", cache_hint::Breakpoint::Unspecified)).await;
		let after_real = script.calls.lock().len();
		tokio::time::sleep(Duration::from_secs(600)).await;
		tokio::task::yield_now().await;
		assert_eq!(script.calls.lock().len(), after_real, "cancelled loop kept pinging");
	}

	#[tokio::test(start_paused = true)]
	async fn a_turn_that_did_not_call_a_tool_is_never_kept_warm() {
		let script = Script::new([stopped(StopReason::StopEndTurn)]);
		let policy = CachePolicy { pings: 3, ..CachePolicy::tail_two() };
		let mut service = Cache::new(script.clone(), policy);

		drain(&mut service, request("session", cache_hint::Breakpoint::Unspecified)).await;
		tokio::time::sleep(Duration::from_secs(1_800)).await;
		tokio::task::yield_now().await;

		// Waiting on a human has no known return time; pinging through it burns
		// reads into the void.
		assert_eq!(script.calls.lock().len(), 1);
	}

	/// A refresh must stop at the first provider-authored frame. Draining it
	/// would pay for the generation the hit-and-cancel exists to avoid.
	#[tokio::test(start_paused = true)]
	async fn a_refresh_stops_polling_once_the_provider_has_answered() {
		use std::sync::atomic::{AtomicUsize, Ordering};

		type Boxed = Pin<Box<dyn Stream<Item = TurnEvent> + Send>>;

		#[derive(Clone)]
		struct Counting {
			polled: Arc<AtomicUsize>,
			calls:  Arc<AtomicUsize>,
		}

		impl Service<TurnRequest> for Counting {
			type Error = std::convert::Infallible;
			type Future = std::future::Ready<Result<Boxed, Self::Error>>;
			type Response = Boxed;

			fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
				Poll::Ready(Ok(()))
			}

			fn call(&mut self, _req: TurnRequest) -> Self::Future {
				let first = self.calls.fetch_add(1, Ordering::SeqCst) == 0;
				let frames = if first {
					vec![ev(turn_event::Event::Outcome(Outcome {
						stop: StopReason::StopToolUse as i32,
						..Outcome::default()
					}))]
				} else {
					// Only the first of these is needed to prove prefill; the
					// rest are the generation a drain would buy.
					vec![
						ev(turn_event::Event::PartDelta(PartDelta::default())),
						ev(turn_event::Event::PartDelta(PartDelta::default())),
						ev(turn_event::Event::PartDelta(PartDelta::default())),
						ev(turn_event::Event::Outcome(Outcome::default())),
					]
				};
				let polled = Arc::clone(&self.polled);
				let stream = futures::stream::iter(frames).inspect(move |_| {
					polled.fetch_add(1, Ordering::SeqCst);
				});
				std::future::ready(Ok(Box::pin(stream) as Boxed))
			}
		}

		let polled = Arc::new(AtomicUsize::new(0));
		let inner = Counting { polled: Arc::clone(&polled), calls: Arc::new(AtomicUsize::new(0)) };
		let policy =
			CachePolicy { pings: 1, ttl: Duration::from_secs(300), ..CachePolicy::tail_two() };
		let mut service = Cache::new(inner, policy);

		let mut stream = service
			.ready()
			.await
			.unwrap()
			.call(request("session", cache_hint::Breakpoint::Unspecified))
			.await
			.unwrap();
		while stream.next().await.is_some() {}
		assert_eq!(polled.load(Ordering::SeqCst), 1, "the real turn yields one frame");

		tokio::time::sleep(Duration::from_secs(300)).await;
		tokio::task::yield_now().await;
		assert_eq!(
			polled.load(Ordering::SeqCst),
			2,
			"the refresh drained past prefill instead of cancelling"
		);
	}

	/// An enabled route is the injecting half of the middleware: an ordinary
	/// request carries no hint, and must leave with one keyed by the
	/// conversation it belongs to.
	#[tokio::test(start_paused = true)]
	async fn an_enabled_route_synthesizes_the_hint_and_keys_it_by_context() {
		let script = Script::new([stopped(StopReason::StopToolUse)]);
		let mut service = Cache::new(script.clone(), CachePolicy::tail_two().with_keepalive(1));

		drain(&mut service, TurnRequest {
			input: Some(turn_request::Input::Seed(Seed {
				context_id: "01JCONTEXT".into(),
				..Seed::default()
			})),
			params: Some(ChatParams::default()),
			..TurnRequest::default()
		})
		.await;

		let hint = seen(&script, 0);
		assert_eq!(hint.breakpoint, cache_hint::Breakpoint::TailTwo as i32);
		assert_eq!(hint.retention, cache_hint::Retention::Short as i32);
		assert_eq!(hint.session_key, "01JCONTEXT", "refreshes need a stable key");

		// Keep-alive was opted into, and it is a real conversation, so the tool
		// gap is kept warm.
		tokio::time::sleep(Duration::from_secs(300)).await;
		tokio::task::yield_now().await;
		assert_eq!(script.calls.lock().len(), 2);
	}

	/// A stateless turn has no conversation to key on. It still gets its
	/// breakpoints — the prefix is billed either way — but nothing is armed,
	/// because there is no next turn to keep the prefix warm for.
	#[tokio::test(start_paused = true)]
	async fn a_stateless_turn_gets_placement_but_no_refresh() {
		let script = Script::new([stopped(StopReason::StopToolUse)]);
		let mut service = Cache::new(script.clone(), CachePolicy::tail_two());

		drain(&mut service, TurnRequest {
			input: Some(turn_request::Input::Seed(Seed::default())),
			params: Some(ChatParams::default()),
			..TurnRequest::default()
		})
		.await;

		let hint = seen(&script, 0);
		assert_eq!(hint.breakpoint, cache_hint::Breakpoint::TailTwo as i32);
		assert!(hint.session_key.is_empty());

		tokio::time::sleep(Duration::from_secs(1_800)).await;
		tokio::task::yield_now().await;
		assert_eq!(script.calls.lock().len(), 1, "a keyless turn armed a refresh");
	}

	/// The inert default must not even create the hint: its presence alone
	/// changes other dialects, and this layer sits on every route.
	#[tokio::test]
	async fn an_inert_route_never_invents_a_hint() {
		let script = Script::new([stopped(StopReason::StopEndTurn)]);
		let mut service = Cache::new(script.clone(), CachePolicy::default());

		drain(&mut service, TurnRequest {
			input: Some(turn_request::Input::Seed(Seed {
				context_id: "01JCONTEXT".into(),
				..Seed::default()
			})),
			params: Some(ChatParams::default()),
			..TurnRequest::default()
		})
		.await;

		assert!(
			script.calls.lock()[0]
				.params
				.as_ref()
				.unwrap()
				.cache
				.is_none()
		);
	}

	/// The default is inert: one route config serves every provider, so opting
	/// in has to be explicit.
	#[tokio::test]
	async fn the_default_policy_touches_nothing() {
		let script = Script::new([stopped(StopReason::StopToolUse)]);
		let mut service = Cache::new(script.clone(), CachePolicy::default());

		drain(&mut service, request("session", cache_hint::Breakpoint::Unspecified)).await;

		let hint = seen(&script, 0);
		assert_eq!(hint.breakpoint, cache_hint::Breakpoint::Unspecified as i32);
		assert_eq!(hint.retention, cache_hint::Retention::Unspecified as i32);
		assert_eq!(script.calls.lock().len(), 1, "inert policy scheduled a refresh");
	}

	/// The entry is dated at prefill, so a turn that spends a long time
	/// generating must still refresh before the original window closes — not
	/// one generation-length later, by which point the entry has lapsed.
	#[tokio::test(start_paused = true)]
	async fn the_refresh_deadline_is_measured_from_prefill_not_from_the_outcome() {
		use std::sync::atomic::{AtomicUsize, Ordering};

		type Boxed = Pin<Box<dyn Stream<Item = TurnEvent> + Send>>;
		const THINKING: Duration = Duration::from_secs(120);

		#[derive(Clone)]
		struct SlowTurn {
			calls: Arc<AtomicUsize>,
		}

		impl Service<TurnRequest> for SlowTurn {
			type Error = std::convert::Infallible;
			type Future = std::future::Ready<Result<Boxed, Self::Error>>;
			type Response = Boxed;

			fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
				Poll::Ready(Ok(()))
			}

			fn call(&mut self, _req: TurnRequest) -> Self::Future {
				let first = self.calls.fetch_add(1, Ordering::SeqCst) == 0;
				let stream = async_stream::stream! {
					// Prefill: the entry exists from here.
					yield ev(turn_event::Event::PartDelta(PartDelta::default()));
					if first {
						tokio::time::sleep(THINKING).await;
						yield ev(turn_event::Event::Outcome(Outcome {
							stop: StopReason::StopToolUse as i32,
							..Outcome::default()
						}));
					}
				};
				std::future::ready(Ok(Box::pin(stream) as Boxed))
			}
		}

		let calls = Arc::new(AtomicUsize::new(0));
		let inner = SlowTurn { calls: Arc::clone(&calls) };
		let policy = CachePolicy {
			pings: 1,
			ttl: Duration::from_secs(300),
			lead: Duration::from_secs(1),
			..CachePolicy::tail_two()
		};
		let mut service = Cache::new(inner, policy);

		let mut stream = service
			.ready()
			.await
			.unwrap()
			.call(request("session", cache_hint::Breakpoint::Unspecified))
			.await
			.unwrap();
		while stream.next().await.is_some() {}
		assert_eq!(calls.load(Ordering::SeqCst), 1);

		// 299s after prefill is 179s after the outcome. Anchored to the
		// outcome, nothing would have fired yet and the entry would be dead.
		tokio::time::sleep(Duration::from_secs(170)).await;
		tokio::task::yield_now().await;
		assert_eq!(calls.load(Ordering::SeqCst), 1, "refreshed too early");

		tokio::time::sleep(Duration::from_secs(20)).await;
		tokio::task::yield_now().await;
		assert_eq!(
			calls.load(Ordering::SeqCst),
			2,
			"refresh was scheduled from the outcome, so it missed the window"
		);
	}
}
