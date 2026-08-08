//! Recovery middleware behavior over scripted provider-attempt streams.

use std::{
	collections::{BTreeMap, VecDeque},
	convert::Infallible,
	sync::Arc,
	task::{Context, Poll},
};

use futures::StreamExt;
use omp_llm_error::RetryBudget;
use omp_llm_tower::{
	envelope::ProviderRequest,
	recovery::{Recovery, RecoveryConfig},
};
use omp_llm_types::ResolvedModelPolicy;
use omp_proto::{
	inference::v1::{
		ChatParams, Invoke, Outcome, PartDelta, PartStart, Seed, TurnError, TurnEvent, TurnRequest,
		Value, ValueMap, turn_error, turn_event, turn_request, value,
	},
	thread::v1::{Item, Thread},
};
use parking_lot::Mutex;
use tower::{Service, ServiceExt};

/// Service that answers each call with the next scripted event stream.
#[derive(Clone)]
struct Script {
	streams: Arc<Mutex<VecDeque<Vec<TurnEvent>>>>,
	calls:   Arc<Mutex<Vec<TurnRequest>>>,
}

impl Script {
	fn new(streams: impl IntoIterator<Item = Vec<TurnEvent>>) -> Self {
		Self {
			streams: Arc::new(Mutex::new(streams.into_iter().collect())),
			calls:   Arc::new(Mutex::new(Vec::new())),
		}
	}
}

type ScriptStream = futures::stream::Iter<std::vec::IntoIter<TurnEvent>>;

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

#[derive(Clone)]
struct PolicyScript {
	inner:    Script,
	policies: Arc<Mutex<Vec<Option<Arc<ResolvedModelPolicy>>>>>,
}

impl Service<ProviderRequest> for PolicyScript {
	type Error = Infallible;
	type Future = std::future::Ready<Result<ScriptStream, Infallible>>;
	type Response = ScriptStream;

	fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Infallible>> {
		self.inner.poll_ready(cx)
	}

	fn call(&mut self, req: ProviderRequest) -> Self::Future {
		self.policies.lock().push(req.model_policy);
		self.inner.call(req.request)
	}
}

const fn ev(event: turn_event::Event) -> TurnEvent {
	TurnEvent { event: Some(event) }
}

fn part_start() -> TurnEvent {
	ev(turn_event::Event::PartStart(PartStart::default()))
}

fn part_delta() -> TurnEvent {
	ev(turn_event::Event::PartDelta(PartDelta::default()))
}

fn outcome() -> TurnEvent {
	ev(turn_event::Event::Outcome(Outcome::default()))
}

fn invoke() -> TurnEvent {
	ev(turn_event::Event::Invoke(Invoke::default()))
}

fn error(kind: turn_error::Kind, detail: &str) -> TurnEvent {
	ev(turn_event::Event::Error(TurnError {
		kind: kind as i32,
		detail: detail.to_owned(),
		..TurnError::default()
	}))
}

fn fast_config() -> RecoveryConfig {
	RecoveryConfig { budget: RetryBudget::new(3, 1, 2, 1_000_000), ..RecoveryConfig::default() }
}

fn string_value(value: &str) -> Value {
	Value { kind: Some(value::Kind::String(value.to_owned())) }
}

fn stale_responses_request() -> TurnRequest {
	let server_item = ValueMap {
		fields: BTreeMap::from([
			("id".to_owned(), string_value("rs_server")),
			("type".to_owned(), string_value("web_search_call")),
		]),
	};
	let item = Item {
		props: Some(ValueMap {
			fields: BTreeMap::from([
				("openai/item_id".to_owned(), string_value("rs_message")),
				("openai/server_tool_item".to_owned(), Value {
					kind: Some(value::Kind::Map(server_item)),
				}),
				("test/retained".to_owned(), string_value("yes")),
			]),
		}),
		..Item::default()
	};
	TurnRequest {
		turn_id: "stale-turn".to_owned(),
		input: Some(turn_request::Input::Seed(Seed {
			thread: Some(Thread { items: vec![item] }),
			..Seed::default()
		})),
		params: Some(ChatParams {
			provider_options: Some(ValueMap {
				fields: BTreeMap::from([
					("openai/previous_response_id".to_owned(), string_value("resp_stale")),
					("openai/previous_response_item_count".to_owned(), Value {
						kind: Some(value::Kind::Uint(1)),
					}),
				]),
			}),
			..ChatParams::default()
		}),
		..TurnRequest::default()
	}
}

async fn run(config: RecoveryConfig, streams: Vec<Vec<TurnEvent>>) -> Vec<TurnEvent> {
	let mut svc = Recovery::new(Script::new(streams), config);
	let stream = svc
		.ready()
		.await
		.unwrap()
		.call(TurnRequest::default())
		.await
		.unwrap();
	stream.collect().await
}

const fn kind_of(frame: &TurnEvent) -> &'static str {
	match frame.event {
		Some(turn_event::Event::Attempt(_)) => "attempt",
		Some(turn_event::Event::PartStart(_)) => "part_start",
		Some(turn_event::Event::PartDelta(_)) => "part_delta",
		Some(turn_event::Event::Outcome(_)) => "outcome",
		Some(turn_event::Event::Error(_)) => "error",
		Some(turn_event::Event::Invoke(_)) => "invoke",
		_ => "other",
	}
}

#[tokio::test]
async fn pre_output_eof_retries_with_attempt_frame() {
	// First attempt dies without any frame; retry succeeds.
	let frames = run(fast_config(), vec![vec![], vec![outcome()]]).await;
	let kinds: Vec<_> = frames.iter().map(kind_of).collect();
	assert_eq!(kinds, ["attempt", "outcome"]);
	let Some(turn_event::Event::Attempt(att)) = &frames[0].event else {
		unreachable!()
	};
	assert_eq!(att.number, 2, "retry 1 announces dispatch 2");
	assert!(att.reason.contains("terminal frame"), "synthetic EOF reason: {}", att.reason);
}

#[tokio::test]
async fn post_output_eof_surfaces_terminal_error() {
	// Output already streamed, then EOF: never a silent clean end, and no
	// replay by default (partials belong to the client).
	let frames = run(fast_config(), vec![vec![part_start(), part_delta()]]).await;
	let kinds: Vec<_> = frames.iter().map(kind_of).collect();
	assert_eq!(kinds, ["part_start", "part_delta", "error"]);
	let Some(turn_event::Event::Error(err)) = &frames[2].event else {
		unreachable!()
	};
	assert!(err.detail.contains("terminal frame"));
}

#[tokio::test]
async fn exact_replay_preserves_model_policy_arc_identity() {
	let policy = Arc::new(ResolvedModelPolicy {
		request_model_id: Some("wire-model".into()),
		..ResolvedModelPolicy::default()
	});
	let policies = Arc::new(Mutex::new(Vec::new()));
	let script = PolicyScript {
		inner:    Script::new([
			vec![error(turn_error::Kind::Upstream, "connection error, retry your request")],
			vec![outcome()],
		]),
		policies: Arc::clone(&policies),
	};
	let stream = Recovery::new(script, fast_config())
		.oneshot(ProviderRequest::new(TurnRequest::default(), Some(Arc::clone(&policy))))
		.await
		.expect("dispatch");
	let _frames: Vec<_> = stream.collect().await;
	let policies = policies.lock();
	assert_eq!(policies.len(), 2);
	assert!(
		policies
			.iter()
			.all(|candidate| { Arc::ptr_eq(candidate.as_ref().expect("model policy"), &policy) })
	);
}

#[tokio::test]
async fn transient_error_frame_retries() {
	let frames = run(fast_config(), vec![
		vec![error(turn_error::Kind::Upstream, "connection error, retry your request")],
		vec![outcome()],
	])
	.await;
	let kinds: Vec<_> = frames.iter().map(kind_of).collect();
	assert_eq!(kinds, ["attempt", "outcome"]);
}

#[tokio::test]
async fn oversized_overload_retry_after_fails_fast_without_redispatch() {
	let overloaded = ev(turn_event::Event::Error(TurnError {
		kind: turn_error::Kind::Overloaded as i32,
		detail: "provider overloaded".to_owned(),
		retry_after_ms: 301_000,
		..TurnError::default()
	}));
	let script = Script::new([vec![overloaded], vec![outcome()]]);
	let config =
		RecoveryConfig { budget: RetryBudget::new(3, 1, 2, 300_000), ..RecoveryConfig::default() };
	let stream = Recovery::new(script.clone(), config)
		.oneshot(TurnRequest::default())
		.await
		.expect("dispatch");
	let frames = stream.collect::<Vec<_>>().await;
	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["error"]);
	assert_eq!(script.calls.lock().len(), 1, "oversized provider hint must not be retried");
	let Some(turn_event::Event::Error(error)) = &frames[0].event else {
		panic!("expected terminal overload")
	};
	assert_eq!(error.kind(), turn_error::Kind::Overloaded);
	assert_eq!(error.retry_after_ms, 301_000);
}

#[tokio::test]
async fn stale_responses_replays_full_thread_without_server_identifiers() {
	let stale = error(turn_error::Kind::Upstream, "HTTP 404 Item with id 'rs_message' not found.");
	let script = Script::new([vec![stale], vec![outcome()]]);
	let mut svc = Recovery::new(script.clone(), fast_config());
	let frames = svc
		.ready()
		.await
		.unwrap()
		.call(stale_responses_request())
		.await
		.unwrap()
		.collect::<Vec<_>>()
		.await;
	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["attempt", "outcome"]);

	let calls = script.calls.lock();
	assert_eq!(calls.len(), 2);
	let replay = &calls[1];
	let options = replay
		.params
		.as_ref()
		.and_then(|params| params.provider_options.as_ref())
		.expect("provider options");
	assert!(!options.fields.contains_key("openai/previous_response_id"));
	assert!(
		!options
			.fields
			.contains_key("openai/previous_response_item_count")
	);
	let Some(turn_request::Input::Seed(seed)) = &replay.input else {
		panic!("full seed replay");
	};
	let items = &seed.thread.as_ref().expect("thread").items;
	assert_eq!(items.len(), 1, "full canonical context is retained");
	let props = items[0].props.as_ref().expect("item props");
	assert!(!props.fields.contains_key("openai/item_id"));
	assert_eq!(
		props
			.fields
			.get("test/retained")
			.and_then(|value| match &value.kind {
				Some(value::Kind::String(value)) => Some(value.as_str()),
				_ => None,
			}),
		Some("yes"),
	);
	let server_item = props
		.fields
		.get("openai/server_tool_item")
		.and_then(|value| match &value.kind {
			Some(value::Kind::Map(value)) => Some(value),
			_ => None,
		})
		.expect("server item");
	assert!(!server_item.fields.contains_key("id"));
	assert!(server_item.fields.contains_key("type"));
}

#[tokio::test]
async fn stale_previous_response_code_uses_the_same_one_shot_full_replay() {
	let stale = error(
		turn_error::Kind::Upstream,
		r#"HTTP 400 {"error":{"code":"previous_response_not_found","message":"Previous response expired."}}"#,
	);
	let script = Script::new([vec![stale], vec![outcome()]]);
	let mut svc = Recovery::new(script.clone(), fast_config());
	let frames = svc
		.ready()
		.await
		.unwrap()
		.call(stale_responses_request())
		.await
		.unwrap()
		.collect::<Vec<_>>()
		.await;
	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["attempt", "outcome"]);
	assert_eq!(script.calls.lock().len(), 2);
}

#[tokio::test]
async fn stale_responses_full_replay_is_attempted_only_once() {
	let stale = || {
		error(turn_error::Kind::Upstream, "previous_response_not_found: previous response expired")
	};
	let script = Script::new([vec![stale()], vec![stale()], vec![outcome()]]);
	let mut svc = Recovery::new(script.clone(), fast_config());
	let frames = svc
		.ready()
		.await
		.unwrap()
		.call(stale_responses_request())
		.await
		.unwrap()
		.collect::<Vec<_>>()
		.await;
	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["attempt", "error"]);
	assert_eq!(script.calls.lock().len(), 2);
}

#[tokio::test]
async fn stale_responses_does_not_cross_output_replay_barrier() {
	let frames = run(fast_config(), vec![
		vec![
			part_delta(),
			error(
				turn_error::Kind::Upstream,
				"previous_response_not_found: Item with id 'rs_late' not found.",
			),
		],
		vec![outcome()],
	])
	.await;
	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["part_delta", "error"]);
}

#[tokio::test]
async fn invoke_latch_blocks_retry_after_commit() {
	let frames = run(fast_config(), vec![
		vec![
			invoke(),
			error(
				turn_error::Kind::Upstream,
				"previous_response_not_found: Item with id 'rs_invoke' not found.",
			),
		],
		vec![outcome()],
	])
	.await;
	let kinds: Vec<_> = frames.iter().map(kind_of).collect();
	assert_eq!(kinds, ["invoke", "error"], "invocation side effects bar transparent replay");
}

#[tokio::test]
async fn giveup_normalizes_upstream_rate_limit() {
	// A usage limit is not same-route retryable; the terminal frame is
	// normalized: kind upgraded and retry_after_ms filled from the lane.
	let frames = run(fast_config(), vec![vec![error(
		turn_error::Kind::Upstream,
		"429 The usage limit has been reached (usage_limit_reached)",
	)]])
	.await;
	assert_eq!(frames.len(), 1);
	let Some(turn_event::Event::Error(err)) = &frames[0].event else {
		unreachable!()
	};
	assert_eq!(err.kind(), turn_error::Kind::RateLimited);
	assert_eq!(err.retry_after_ms, 1_800_000);
}

#[tokio::test]
async fn budget_exhaustion_surfaces_last_error() {
	// Three attempts allowed, all EOF: terminal synthetic error, no hang.
	let frames = run(fast_config(), vec![vec![], vec![], vec![], vec![]]).await;
	let kinds: Vec<_> = frames.iter().map(kind_of).collect();
	assert_eq!(kinds, ["attempt", "attempt", "attempt", "error"]);
}
