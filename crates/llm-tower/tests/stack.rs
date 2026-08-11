//! Once-built route-stack proofs over a scripted provider attempt.

use std::{
	convert::Infallible,
	future::Future,
	pin::Pin,
	sync::{
		Arc,
		atomic::{AtomicBool, AtomicU64, Ordering},
	},
	task::{Context, Poll},
	time::Duration,
};

use bytes::Bytes;
use futures::{Stream, StreamExt};
use omp_llm_catalog::compat::{Compat, LeakedThinkingHealer, StreamWatchdog};
use omp_llm_error::{Classification, Feature, RetryBudget, policy::BlockTable};
use omp_llm_tower::{
	cache::CachePolicy,
	envelope::ProviderRequest,
	learn::RequestRepair,
	preflight::{Admission, UsageOracle},
	recovery::RecoveryConfig,
	refresh::{CredentialRefresher, RefreshFailure},
	resample::ResampleConfig,
	select::{CredentialCandidates, CredentialLease, CredentialPool, LeaseSource, Routed},
	stack::builder::{RouteDependencies, RouteStack, RouteStackBuilder, RouteStackConfig},
	tap::FrameSink,
	testing::{Script, error, ev, kind_of},
};
use omp_proto::{
	inference::v1::{
		CacheHint, ChatParams, Effort, Fallback, Outcome, PartDelta, PartEnd, PartStart, Reasoning,
		Seed, StopReason, ToolChoice, ToolDef, TurnEvent, TurnRequest, cache_hint, part_start,
		tool_choice, turn_error, turn_event, turn_request,
	},
	thread::v1::{Item, Message, Part, Role, item, part},
};
use parking_lot::Mutex;
use tower::{Service, ServiceExt};

/// Outcome carrying one item, the representative success shape.
fn full_outcome() -> TurnEvent {
	ev(turn_event::Event::Outcome(Outcome {
		output: vec![Item {
			kind: Some(item::Kind::Message(Message {
				role:  Role::Assistant as i32,
				parts: vec![Part { kind: Some(part::Kind::Text("ok".to_owned())) }],
			})),
			..Item::default()
		}],
		stop: StopReason::StopEndTurn as i32,
		..Outcome::default()
	}))
}

fn empty_outcome() -> TurnEvent {
	ev(turn_event::Event::Outcome(Outcome::default()))
}

fn request_with_params(params: ChatParams) -> TurnRequest {
	TurnRequest {
		turn_id: "turn".to_owned(),
		input: Some(turn_request::Input::Seed(Seed {
			thread: Some(Default::default()),
			..Seed::default()
		})),
		params: Some(params),
		..TurnRequest::default()
	}
}

struct AllowAll;
impl UsageOracle for AllowAll {
	fn admit(&self, _model: &str) -> Admission {
		Admission::Allow
	}
}

struct DenyQuota;
impl UsageOracle for DenyQuota {
	fn admit(&self, _model: &str) -> Admission {
		Admission::DenyQuota {
			detail:         "monthly quota exhausted".to_owned(),
			retry_after_ms: 60_000,
		}
	}
}

struct OnePool;
impl CredentialPool for OnePool {
	fn candidates(&self, _model: &str) -> CredentialCandidates {
		std::iter::once(1).collect()
	}
}

struct TwoPool;
impl CredentialPool for TwoPool {
	fn candidates(&self, _model: &str) -> CredentialCandidates {
		[1, 2].into_iter().collect()
	}
}

struct SwitchPool(AtomicU64);
impl CredentialPool for SwitchPool {
	fn candidates(&self, _model: &str) -> CredentialCandidates {
		std::iter::once(self.0.load(Ordering::Relaxed)).collect()
	}
}

struct Leases;
impl LeaseSource for Leases {
	fn lease(&self, id: u64) -> Option<CredentialLease> {
		Some(CredentialLease::new("provider", id, 1))
	}
}

struct NoRepair;
impl RequestRepair for NoRepair {
	fn strip(
		&self,
		_req: &TurnRequest,
		_feature: Feature,
		_cls: &Classification,
	) -> Option<TurnRequest> {
		None
	}
}

struct FreshForever;
impl CredentialRefresher for FreshForever {
	fn expires_at_ms(&self) -> Option<u64> {
		None
	}

	fn refresh(
		&self,
		_force: bool,
	) -> std::pin::Pin<Box<dyn Future<Output = Result<(), RefreshFailure>> + Send + '_>> {
		Box::pin(std::future::ready(Ok(())))
	}
}

struct OrderedObserver(Arc<Mutex<Vec<&'static str>>>);
impl FrameSink for OrderedObserver {
	fn on_request(&self, _req: &TurnRequest) {
		self.0.lock().push("tap");
	}

	fn on_frame(&self, _frame: &TurnEvent) {}

	fn on_end(&self) {}
}

struct OrderedOracle(Arc<Mutex<Vec<&'static str>>>);
impl UsageOracle for OrderedOracle {
	fn admit(&self, _model: &str) -> Admission {
		self.0.lock().push("preflight");
		Admission::Allow
	}
}

struct OrderedPool(Arc<Mutex<Vec<&'static str>>>);
impl CredentialPool for OrderedPool {
	fn candidates(&self, _model: &str) -> CredentialCandidates {
		self.0.lock().push("select");
		std::iter::once(1).collect()
	}
}

struct OrderedRefresh(Arc<Mutex<Vec<&'static str>>>);
impl CredentialRefresher for OrderedRefresh {
	fn expires_at_ms(&self) -> Option<u64> {
		self.0.lock().push("refresh");
		None
	}

	fn refresh(
		&self,
		_force: bool,
	) -> std::pin::Pin<Box<dyn Future<Output = Result<(), RefreshFailure>> + Send + '_>> {
		Box::pin(std::future::ready(Ok(())))
	}
}

struct NoopObserver;
impl FrameSink for NoopObserver {
	fn on_request(&self, _req: &TurnRequest) {}

	fn on_frame(&self, _frame: &TurnEvent) {}

	fn on_end(&self) {}
}

/// Innermost provider attempt: records the out-of-band lease, then hands only
/// the canonical request to the scripted transport.
#[derive(Clone)]
struct RecordingAttempt<S> {
	inner:       S,
	leases_seen: Arc<Mutex<Vec<Option<CredentialLease>>>>,
}

impl<S, St> Service<Routed> for RecordingAttempt<S>
where
	S: Service<TurnRequest, Response = St, Error = Infallible>,
{
	type Error = Infallible;
	type Future = S::Future;
	type Response = St;

	fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Infallible>> {
		self.inner.poll_ready(cx)
	}

	fn call(&mut self, routed: Routed) -> Self::Future {
		self.leases_seen.lock().push(routed.lease);
		self.inner.call(routed.request)
	}
}

#[derive(Clone)]
struct OrderedAttempt {
	inner: Script,
	order: Arc<Mutex<Vec<&'static str>>>,
}

impl Service<Routed> for OrderedAttempt {
	type Error = Infallible;
	type Future = <Script as Service<TurnRequest>>::Future;
	type Response = <Script as Service<TurnRequest>>::Response;

	fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		self.inner.poll_ready(cx)
	}

	fn call(&mut self, routed: Routed) -> Self::Future {
		self.order.lock().push("provider");
		self.inner.call(routed.request)
	}
}

struct Stack {
	svc:         RouteStack<RecordingAttempt<Script>>,
	script:      Script,
	leases_seen: Arc<Mutex<Vec<Option<CredentialLease>>>>,
	blocks:      Arc<Mutex<BlockTable>>,
}

/// Constructs the documented altitude exactly once, then reuses its concrete
/// service for every call.
fn stack(oracle: Arc<dyn UsageOracle>, streams: Vec<Vec<TurnEvent>>) -> Stack {
	stack_with_pool(oracle, Arc::new(OnePool), streams)
}

fn stack_with_pool(
	oracle: Arc<dyn UsageOracle>,
	pool: Arc<dyn CredentialPool>,
	streams: Vec<Vec<TurnEvent>>,
) -> Stack {
	stack_with_config(oracle, pool, streams, RouteStackConfig {
		recovery: RecoveryConfig {
			budget: RetryBudget::new(3, 1, 2, 1_000_000),
			..RecoveryConfig::default()
		},
		resample: ResampleConfig { max_attempts: 0, base_delay_ms: 0 },
		..RouteStackConfig::default()
	})
}

fn stack_with_config(
	oracle: Arc<dyn UsageOracle>,
	pool: Arc<dyn CredentialPool>,
	streams: Vec<Vec<TurnEvent>>,
	config: RouteStackConfig,
) -> Stack {
	let script = Script::new(streams);
	let leases_seen = Arc::new(Mutex::new(Vec::new()));
	let blocks = Arc::new(Mutex::new(BlockTable::new()));
	let attempt =
		RecordingAttempt { inner: script.clone(), leases_seen: Arc::clone(&leases_seen) };
	let builder = RouteStackBuilder::new(
		RouteDependencies {
			usage:          oracle,
			credentials:    pool,
			leases:         Arc::new(Leases),
			refresher:      Arc::new(FreshForever),
			repair:         Arc::new(NoRepair),
			observer:       Arc::new(NoopObserver),
			usage_observer: Arc::new(omp_llm_tower::stack::meter::NoopUsageObserver),
			blocks:         Arc::clone(&blocks),
		},
		config,
	);
	let svc = builder.build(attempt);
	Stack { svc, script, leases_seen, blocks }
}

const fn assert_provider_request_service<S, St>(_service: &S)
where
	S: Service<ProviderRequest, Response = St>,
{
}

async fn drive(stack: &mut Stack) -> Vec<TurnEvent> {
	drive_request(stack, TurnRequest::default()).await
}

async fn drive_request(stack: &mut Stack, request: TurnRequest) -> Vec<TurnEvent> {
	let stream = stack
		.svc
		.ready()
		.await
		.unwrap()
		.call(ProviderRequest::new(request, None))
		.await
		.unwrap();
	stream.collect().await
}

#[tokio::test]
async fn happy_path_traverses_all_layers() {
	let mut stack = stack(Arc::new(AllowAll), vec![vec![full_outcome()]]);
	assert_provider_request_service(&stack.svc);
	let frames = drive(&mut stack).await;
	let kinds: Vec<_> = frames.iter().map(kind_of).collect();
	assert_eq!(kinds, ["outcome"]);
	assert_eq!(stack.script.calls.lock().len(), 1);
	// The lease crossed the bridge out of band; the request payload is
	// untouched default.
	assert_eq!(stack.leases_seen.lock().as_slice(), &[Some(CredentialLease::new("provider", 1, 1))]);
	assert_eq!(stack.script.calls.lock()[0], TurnRequest::default());
	assert!(stack.blocks.lock().earliest_unblock_ms(0).is_none());
}

#[tokio::test]
async fn transient_upstream_error_recovers_via_recovery_layer() {
	let mut stack = stack(Arc::new(AllowAll), vec![
		vec![error(turn_error::Kind::Upstream, "connection error, retry your request")],
		vec![full_outcome()],
	]);
	let frames = drive(&mut stack).await;
	let kinds: Vec<_> = frames.iter().map(kind_of).collect();
	assert_eq!(kinds, ["attempt", "outcome"]);
	assert_eq!(stack.script.calls.lock().len(), 2);
}

#[tokio::test]
async fn preflight_denial_stops_before_any_dispatch() {
	let mut stack = stack(Arc::new(DenyQuota), vec![vec![full_outcome()]]);
	let frames = drive(&mut stack).await;
	assert_eq!(frames.len(), 1);
	let Some(omp_proto::inference::v1::turn_event::Event::Error(err)) = &frames[0].event else {
		panic!("expected terminal error frame, got {:?}", frames[0]);
	};
	assert_eq!(err.kind(), turn_error::Kind::RateLimited);
	assert_eq!(err.retry_after_ms, 60_000);
	assert!(stack.script.calls.lock().is_empty(), "no bytes may leave on a quota denial");
	assert!(stack.leases_seen.lock().is_empty());
}

#[tokio::test]
async fn route_is_built_in_documented_altitude_order() {
	let order = Arc::new(Mutex::new(Vec::new()));
	let script = Script::new(vec![vec![full_outcome()]]);
	let learn_order = Arc::clone(&order);
	let builder = RouteStackBuilder::new(
		RouteDependencies {
			usage:          Arc::new(OrderedOracle(Arc::clone(&order))),
			credentials:    Arc::new(OrderedPool(Arc::clone(&order))),
			leases:         Arc::new(Leases),
			refresher:      Arc::new(OrderedRefresh(Arc::clone(&order))),
			repair:         Arc::new(NoRepair),
			observer:       Arc::new(OrderedObserver(Arc::clone(&order))),
			usage_observer: Arc::new(omp_llm_tower::stack::meter::NoopUsageObserver),
			blocks:         Arc::new(Mutex::new(BlockTable::new())),
		},
		RouteStackConfig {
			resample: ResampleConfig { max_attempts: 0, base_delay_ms: 0 },
			learn_scope: Some(Arc::new(move |_request| {
				learn_order.lock().push("learn");
				None
			})),
			..RouteStackConfig::default()
		},
	);
	let mut service = builder.build(OrderedAttempt { inner: script, order: Arc::clone(&order) });
	let stream = service
		.ready()
		.await
		.unwrap()
		.call(ProviderRequest::new(TurnRequest::default(), None))
		.await
		.unwrap();
	let _: Vec<_> = stream.collect().await;
	assert_eq!(order.lock().as_slice(), &[
		"tap",
		"preflight",
		"select",
		"refresh",
		"learn",
		"provider"
	],);
}

#[tokio::test]
async fn pre_commit_refresh_redispatch_preserves_the_out_of_band_lease() {
	let mut stack = stack_with_pool(Arc::new(AllowAll), Arc::new(TwoPool), vec![
		vec![error(turn_error::Kind::Auth, "credential rejected")],
		vec![full_outcome()],
	]);
	let frames = drive(&mut stack).await;
	let kinds: Vec<_> = frames.iter().map(kind_of).collect();
	assert_eq!(kinds, ["attempt", "outcome"]);
	assert_eq!(stack.leases_seen.lock().as_slice(), &[
		Some(CredentialLease::new("provider", 1, 1)),
		Some(CredentialLease::new("provider", 1, 1)),
	],);
	assert!(
		stack
			.script
			.calls
			.lock()
			.iter()
			.all(|request| request == &TurnRequest::default()),
		"lease identity must never enter the protobuf request",
	);
}

#[tokio::test]
async fn first_committed_event_prevents_replay_and_credential_rotation() {
	let mut stack = stack_with_pool(Arc::new(AllowAll), Arc::new(TwoPool), vec![vec![
		ev(turn_event::Event::PartDelta(Default::default())),
		error(turn_error::Kind::Auth, "late credential rejection"),
	]]);
	let frames = drive(&mut stack).await;
	let kinds: Vec<_> = frames.iter().map(kind_of).collect();
	assert_eq!(kinds, ["part_delta", "error"]);
	assert_eq!(stack.script.calls.lock().len(), 1);
	assert_eq!(stack.leases_seen.lock().as_slice(), &[Some(CredentialLease::new("provider", 1, 1))],);
}

fn repair_config() -> RouteStackConfig {
	RouteStackConfig {
		recovery: RecoveryConfig {
			budget: RetryBudget::new(0, 0, 0, 0),
			..RecoveryConfig::default()
		},
		resample: ResampleConfig { max_attempts: 0, base_delay_ms: 0 },
		..RouteStackConfig::default()
	}
}

fn strict_request(model: &str) -> TurnRequest {
	request_with_params(ChatParams {
		model: model.to_owned(),
		tools: vec![ToolDef { name: "search".to_owned(), strict: Some(true), ..ToolDef::default() }],
		..ChatParams::default()
	})
}

#[tokio::test]
async fn built_stack_downgrades_strict_grammar_before_commit_only() {
	const STRICT: &str = r#"HTTP 400 {"error":{"type":"invalid_request_error","message":"the compiled grammar is too large"}}"#;
	let mut stack = stack_with_config(
		Arc::new(AllowAll),
		Arc::new(OnePool),
		vec![vec![error(turn_error::Kind::Upstream, STRICT)], vec![full_outcome()]],
		repair_config(),
	);
	let frames = drive_request(&mut stack, strict_request("provider/model")).await;
	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["attempt", "outcome"]);
	let calls = stack.script.calls.lock();
	assert_eq!(calls.len(), 2);
	assert_eq!(calls[0].params.as_ref().unwrap().tools[0].strict, Some(true));
	assert_eq!(calls[1].params.as_ref().unwrap().tools[0].strict, Some(false));
	drop(calls);

	let mut late = stack_with_config(
		Arc::new(AllowAll),
		Arc::new(OnePool),
		vec![vec![
			ev(turn_event::Event::PartDelta(PartDelta {
				index: 0,
				chunk: Bytes::from_static(b"visible"),
			})),
			error(turn_error::Kind::Upstream, STRICT),
		]],
		repair_config(),
	);
	let frames = drive_request(&mut late, strict_request("provider/model")).await;
	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["part_delta", "error"]);
	assert_eq!(late.script.calls.lock().len(), 1, "post-commit repair must not replay");
}

#[tokio::test]
async fn built_stack_selects_an_allowed_reasoning_effort() {
	let detail = r#"HTTP 400 {"error":{"message":"[reasoning.effort] Invalid type: expected one of 'low', 'medium', or 'high'"}}"#;
	let request = request_with_params(ChatParams {
		model: "provider/model".to_owned(),
		thinking: Some(Reasoning {
			effort: Effort::Max as i32,
			budget_tokens: Some(4096),
			..Reasoning::default()
		}),
		..ChatParams::default()
	});
	let mut stack = stack_with_config(
		Arc::new(AllowAll),
		Arc::new(OnePool),
		vec![vec![error(turn_error::Kind::Upstream, detail)], vec![full_outcome()]],
		repair_config(),
	);
	let frames = drive_request(&mut stack, request).await;
	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["attempt", "outcome"]);
	let calls = stack.script.calls.lock();
	let thinking = calls[1].params.as_ref().unwrap().thinking.as_ref().unwrap();
	assert_eq!(thinking.effort(), Effort::Low);
	assert_eq!(thinking.budget_tokens, None);
}

#[tokio::test]
async fn built_stack_resolves_thinking_and_forced_tool_choice_conflict() {
	let detail =
		r#"HTTP 400 {"error":{"message":"tool_choice: only 'auto' is supported by this model"}}"#;
	let request = request_with_params(ChatParams {
		model: "provider/model".to_owned(),
		thinking: Some(Reasoning { effort: Effort::High as i32, ..Reasoning::default() }),
		tool_choice: Some(ToolChoice {
			mode: tool_choice::Mode::Required as i32,
			..ToolChoice::default()
		}),
		..ChatParams::default()
	});
	let mut stack = stack_with_config(
		Arc::new(AllowAll),
		Arc::new(OnePool),
		vec![vec![error(turn_error::Kind::Upstream, detail)], vec![full_outcome()]],
		repair_config(),
	);
	let frames = drive_request(&mut stack, request).await;
	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["attempt", "outcome"]);
	let calls = stack.script.calls.lock();
	assert!(calls[1].params.as_ref().unwrap().thinking.is_none());
	assert_eq!(
		calls[1]
			.params
			.as_ref()
			.unwrap()
			.tool_choice
			.as_ref()
			.unwrap()
			.mode(),
		tool_choice::Mode::Required
	);
}

#[tokio::test]
async fn built_stack_rewrites_deterministic_llamacpp_tool_parse_failure_once() {
	let mut stack = stack_with_config(
		Arc::new(AllowAll),
		Arc::new(OnePool),
		vec![
			vec![error(
				turn_error::Kind::Upstream,
				"HTTP 500 failed to parse tool call arguments as json",
			)],
			vec![full_outcome()],
		],
		repair_config(),
	);
	let frames = drive_request(
		&mut stack,
		request_with_params(ChatParams {
			model: "llama/model".to_owned(),
			tools: vec![ToolDef { name: "search".to_owned(), ..ToolDef::default() }],
			..ChatParams::default()
		}),
	)
	.await;
	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["attempt", "outcome"]);
	let calls = stack.script.calls.lock();
	let turn_request::Input::Seed(seed) = calls[1].input.as_ref().unwrap() else {
		panic!("expected seeded request")
	};
	assert_eq!(seed.thread.as_ref().unwrap().items.len(), 1);
}

#[tokio::test]
async fn built_stack_resamples_empty_completion_to_its_bound() {
	let mut config = repair_config();
	config.resample = ResampleConfig { max_attempts: 1, base_delay_ms: 0 };
	let mut stack = stack_with_config(
		Arc::new(AllowAll),
		Arc::new(OnePool),
		vec![vec![empty_outcome()], vec![full_outcome()]],
		config,
	);
	let frames = drive(&mut stack).await;
	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["attempt", "outcome"]);
	assert_eq!(stack.script.calls.lock().len(), 2);
}

#[tokio::test]
async fn built_stack_heals_leaked_thinking_and_guards_gemini_loops() {
	let mut compat = Compat::default();
	compat.leaked_thinking_healer = LeakedThinkingHealer::Thinking;
	let mut config = repair_config();
	config.compat = compat;
	let leaked = vec![
		ev(turn_event::Event::PartStart(PartStart {
			index: 0,
			kind: part_start::Kind::Text as i32,
			..PartStart::default()
		})),
		ev(turn_event::Event::PartDelta(PartDelta {
			index: 0,
			chunk: Bytes::from_static(b"<think>reason</think>answer"),
		})),
		ev(turn_event::Event::PartEnd(PartEnd { index: 0, ..PartEnd::default() })),
		full_outcome(),
	];
	let mut stack = stack_with_config(Arc::new(AllowAll), Arc::new(OnePool), vec![leaked], config);
	let frames = drive(&mut stack).await;
	assert!(frames.iter().any(|event| matches!(
		event.event.as_ref(),
		Some(turn_event::Event::PartStart(part))
			if part.kind() == part_start::Kind::Thinking
	)));

	let mut compat = Compat::default();
	compat.thinking_loop_guard = true;
	let mut config = repair_config();
	config.compat = compat;
	let repeated = Bytes::from("reasoning-loop-unit-".repeat(12));
	let mut stack = stack_with_config(
		Arc::new(AllowAll),
		Arc::new(OnePool),
		vec![vec![
			ev(turn_event::Event::PartStart(PartStart {
				index: 0,
				kind: part_start::Kind::Thinking as i32,
				..PartStart::default()
			})),
			ev(turn_event::Event::PartDelta(PartDelta { index: 0, chunk: repeated })),
		]],
		config,
	);
	let frames = drive(&mut stack).await;
	assert_eq!(kind_of(frames.last().unwrap()), "error");
	assert_eq!(stack.script.calls.lock().len(), 1, "visible reasoning crosses the replay cutoff");
}

#[tokio::test]
async fn built_stack_forced_tool_escalation_is_visible_and_bounded() {
	let mut config = repair_config();
	config.forced_tool_attempts = 2;
	config.compat.forced_tool_choice = true;
	let request = request_with_params(ChatParams {
		model: "provider/model".to_owned(),
		tools: vec![ToolDef { name: "search".to_owned(), ..ToolDef::default() }],
		tool_choice: Some(ToolChoice {
			mode:           tool_choice::Mode::Named as i32,
			name:           "search".to_owned(),
			on_unsupported: Fallback::Emulate as i32,
		}),
		..ChatParams::default()
	});
	let mut stack = stack_with_config(
		Arc::new(AllowAll),
		Arc::new(OnePool),
		vec![vec![full_outcome()], vec![full_outcome()]],
		config,
	);
	let frames = drive_request(&mut stack, request).await;
	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["attempt", "attempt", "error"]);
	assert_eq!(stack.script.calls.lock().len(), 2);
	let calls = stack.script.calls.lock();
	assert_eq!(
		calls[0]
			.params
			.as_ref()
			.unwrap()
			.tool_choice
			.as_ref()
			.unwrap()
			.mode(),
		tool_choice::Mode::Auto
	);
	assert_eq!(
		calls[1]
			.params
			.as_ref()
			.unwrap()
			.tool_choice
			.as_ref()
			.unwrap()
			.mode(),
		tool_choice::Mode::Named
	);
}

#[tokio::test]
async fn learned_repairs_expire_and_do_not_cross_accounts() {
	const STRICT: &str = r#"HTTP 400 {"error":{"type":"invalid_request_error","message":"the compiled grammar is too large"}}"#;
	let pool = Arc::new(SwitchPool(AtomicU64::new(1)));
	let mut config = repair_config();
	config.learn_expiry = Duration::from_millis(2);
	let mut stack = stack_with_config(
		Arc::new(AllowAll),
		pool.clone(),
		vec![
			vec![error(turn_error::Kind::Upstream, STRICT)],
			vec![full_outcome()],
			vec![full_outcome()],
			vec![full_outcome()],
		],
		config,
	);
	drive_request(&mut stack, strict_request("provider/model")).await;
	pool.0.store(2, Ordering::Relaxed);
	let mut second = strict_request("provider/model");
	second.turn_id = "account-two".to_owned();
	drive_request(&mut stack, second).await;
	assert_eq!(
		stack.script.calls.lock()[2].params.as_ref().unwrap().tools[0].strict,
		Some(true),
		"account two must not inherit account one's downgrade"
	);

	pool.0.store(1, Ordering::Relaxed);
	tokio::time::sleep(Duration::from_millis(5)).await;
	let mut expired = strict_request("provider/model");
	expired.turn_id = "expired".to_owned();
	drive_request(&mut stack, expired).await;
	assert_eq!(
		stack.script.calls.lock()[3].params.as_ref().unwrap().tools[0].strict,
		Some(true),
		"expired learning must be probed again"
	);
}

#[derive(Clone)]
struct PendingAttempt {
	dropped: Arc<AtomicBool>,
}

struct PendingStream {
	dropped: Arc<AtomicBool>,
}

impl Stream for PendingStream {
	type Item = TurnEvent;

	fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
		Poll::Pending
	}
}

impl Drop for PendingStream {
	fn drop(&mut self) {
		self.dropped.store(true, Ordering::Relaxed);
	}
}

impl Service<Routed> for PendingAttempt {
	type Error = Infallible;
	type Future = std::future::Ready<Result<Self::Response, Self::Error>>;
	type Response = PendingStream;

	fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		Poll::Ready(Ok(()))
	}

	fn call(&mut self, _request: Routed) -> Self::Future {
		std::future::ready(Ok(PendingStream { dropped: Arc::clone(&self.dropped) }))
	}
}

fn pending_stack(dropped: Arc<AtomicBool>, compat: Compat) -> RouteStack<PendingAttempt> {
	RouteStackBuilder::new(
		RouteDependencies {
			usage:          Arc::new(AllowAll),
			credentials:    Arc::new(OnePool),
			leases:         Arc::new(Leases),
			refresher:      Arc::new(FreshForever),
			repair:         Arc::new(NoRepair),
			observer:       Arc::new(NoopObserver),
			usage_observer: Arc::new(omp_llm_tower::stack::meter::NoopUsageObserver),
			blocks:         Arc::new(Mutex::new(BlockTable::new())),
		},
		RouteStackConfig {
			compat,
			recovery: RecoveryConfig {
				budget: RetryBudget::new(0, 0, 0, 0),
				..RecoveryConfig::default()
			},
			resample: ResampleConfig { max_attempts: 0, base_delay_ms: 0 },
			..RouteStackConfig::default()
		},
	)
	.build(PendingAttempt { dropped })
}

#[tokio::test]
async fn built_stack_watchdog_aborts_upstream_and_stream_drop_cancels_without_fallback() {
	let dropped = Arc::new(AtomicBool::new(false));
	let mut compat = Compat::default();
	compat.stream_watchdog = StreamWatchdog { first_event_ms: Some(1), idle_ms: None };
	let mut stack = pending_stack(Arc::clone(&dropped), compat);
	let frames: Vec<_> = stack
		.ready()
		.await
		.unwrap()
		.call(ProviderRequest::new(TurnRequest::default(), None))
		.await
		.unwrap()
		.collect()
		.await;
	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["error"]);
	assert!(dropped.load(Ordering::Relaxed), "watchdog must drop the live upstream");

	let dropped = Arc::new(AtomicBool::new(false));
	let mut stack = pending_stack(Arc::clone(&dropped), Compat::default());
	let mut stream = Box::pin(
		stack
			.ready()
			.await
			.unwrap()
			.call(ProviderRequest::new(TurnRequest::default(), None))
			.await
			.unwrap(),
	);
	std::future::poll_fn(|cx| {
		assert!(matches!(stream.as_mut().poll_next(cx), Poll::Pending));
		Poll::Ready(())
	})
	.await;
	drop(stream);
	assert!(dropped.load(Ordering::Relaxed), "caller cancellation must drop the live attempt");
}

/// Outcome that ends the turn on a tool call: the one gap a keep-alive may
/// bridge, because the agent is blocked on its own tool rather than a human.
fn tool_use_outcome() -> TurnEvent {
	ev(turn_event::Event::Outcome(Outcome {
		stop: StopReason::StopToolUse as i32,
		..Outcome::default()
	}))
}

fn session_request(session: &str) -> TurnRequest {
	TurnRequest {
		params: Some(ChatParams {
			cache: Some(CacheHint { session_key: session.into(), ..CacheHint::default() }),
			..ChatParams::default()
		}),
		..TurnRequest::default()
	}
}

fn seen_hint(stack: &Stack, index: usize) -> CacheHint {
	stack.script.calls.lock()[index]
		.params
		.as_ref()
		.unwrap()
		.cache
		.clone()
		.unwrap()
}

/// The composed route stack must both inject the policy the codec reads and
/// own the refresh lifecycle: arm on a tool-use turn, cancel on the next real
/// request for the same conversation.
#[tokio::test(start_paused = true)]
async fn route_stack_injects_cache_policy_and_owns_refresh_lifecycle() {
	let mut stack = stack_with_config(
		Arc::new(AllowAll),
		Arc::new(OnePool),
		vec![vec![tool_use_outcome()], vec![full_outcome()], vec![full_outcome()]],
		RouteStackConfig {
			cache: CachePolicy { pings: 1, ttl: Duration::from_secs(300), ..CachePolicy::tail_two() },
			resample: ResampleConfig { max_attempts: 0, base_delay_ms: 0 },
			..RouteStackConfig::default()
		},
	);

	drive_request(&mut stack, session_request("conversation")).await;
	let hint = seen_hint(&stack, 0);
	assert_eq!(hint.breakpoint, cache_hint::Breakpoint::TailTwo as i32);
	assert_eq!(hint.retention, cache_hint::Retention::Short as i32);
	assert_eq!(stack.script.calls.lock().len(), 1);

	// The tool gap lapses: exactly one refresh reaches the provider, carrying
	// the same prefix policy so it lands on the same cache entry.
	tokio::time::sleep(Duration::from_secs(300)).await;
	tokio::task::yield_now().await;
	assert_eq!(stack.script.calls.lock().len(), 2, "refresh never dispatched");
	assert_eq!(seen_hint(&stack, 1).breakpoint, cache_hint::Breakpoint::TailTwo as i32);

	// A real turn re-reads the prefix itself, so the loop must not outlive it.
	drive_request(&mut stack, session_request("conversation")).await;
	let after_real = stack.script.calls.lock().len();
	tokio::time::sleep(Duration::from_mins(20)).await;
	tokio::task::yield_now().await;
	assert_eq!(stack.script.calls.lock().len(), after_real, "refresh outlived its turn");
}

/// A route left on the default config must be indistinguishable from before
/// the cache layer existed: one `RouteStackConfig` serves every provider, so a
/// non-Anthropic route inherits whatever the default does.
#[tokio::test(start_paused = true)]
async fn a_default_route_neither_rewrites_the_hint_nor_refreshes() {
	let mut stack = stack(Arc::new(AllowAll), vec![vec![tool_use_outcome()]]);

	drive_request(&mut stack, session_request("conversation")).await;

	let hint = seen_hint(&stack, 0);
	assert_eq!(hint.breakpoint, cache_hint::Breakpoint::Unspecified as i32);
	assert_eq!(hint.retention, cache_hint::Retention::Unspecified as i32);
	assert_eq!(hint.session_key, "conversation", "the client's key must survive");

	tokio::time::sleep(Duration::from_mins(30)).await;
	tokio::task::yield_now().await;
	assert_eq!(stack.script.calls.lock().len(), 1, "default route scheduled a refresh");
}

/// And a route that never sent a hint keeps not having one, so dialects that
/// project `req.cache` into wire fields see no change.
#[tokio::test]
async fn a_default_route_does_not_invent_a_cache_hint() {
	let mut stack = stack(Arc::new(AllowAll), vec![vec![full_outcome()]]);

	drive_request(&mut stack, TurnRequest {
		params: Some(ChatParams::default()),
		..TurnRequest::default()
	})
	.await;

	assert!(
		stack.script.calls.lock()[0]
			.params
			.as_ref()
			.unwrap()
			.cache
			.is_none()
	);
}
