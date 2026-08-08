//! Credential selection and rotation acceptance tests.

use std::{
	convert::Infallible,
	sync::Arc,
	task::{Context, Poll},
	time::{SystemTime, UNIX_EPOCH},
};

use futures::StreamExt;
use omp_llm_error::{BlockKey, BlockTable};
use omp_llm_tower::{
	envelope::ProviderRequest,
	select::{
		CredentialCandidates, CredentialLease, CredentialMetadata, CredentialPool, LeaseSource,
		Routed, Select,
	},
	testing::{Script, ScriptStream, error, kind_of, outcome, part_delta},
};
use omp_llm_types::ResolvedModelPolicy;
use omp_proto::inference::v1::{CacheHint, ChatParams, TurnRequest, turn_error, turn_event};
use parking_lot::Mutex;
use tower::{Service, ServiceExt};

const FIRST_ID: u64 = 10;
const SECOND_ID: u64 = 20;

#[derive(Clone)]
struct TwoCredentials;

impl CredentialPool for TwoCredentials {
	fn candidates(&self, _model: &str) -> CredentialCandidates {
		[FIRST_ID, SECOND_ID].into_iter().collect()
	}
}

#[derive(Clone)]
struct RankedCredentials(Arc<Mutex<CredentialCandidates>>);

impl CredentialPool for RankedCredentials {
	fn candidates(&self, _model: &str) -> CredentialCandidates {
		self.0.lock().clone()
	}
}

#[derive(Clone)]
struct FixedLeases;

fn credential_metadata(id: u64) -> CredentialMetadata {
	CredentialMetadata {
		auth_kind:       omp_llm_egress::auth_inject::CredentialAuthKind::ApiKey,
		identity:        format!("credential-{id}").into(),
		account_id:      Some(format!("account-{id}").into()),
		project_id:      None,
		organization_id: None,
	}
}

impl LeaseSource for FixedLeases {
	fn lease(&self, id: u64) -> Option<CredentialLease> {
		Some(CredentialLease::new("provider", id, id + 1))
	}

	fn metadata(&self, lease: &CredentialLease) -> Option<CredentialMetadata> {
		Some(credential_metadata(lease.credential_id()))
	}
}

#[derive(Clone)]
struct RoutedScript {
	inner: Script,
	calls: Arc<Mutex<Vec<Routed>>>,
}

impl RoutedScript {
	fn new(inner: Script) -> Self {
		Self { inner, calls: Arc::new(Mutex::new(Vec::new())) }
	}
}

impl Service<Routed> for RoutedScript {
	type Error = Infallible;
	type Future = std::future::Ready<Result<ScriptStream, Infallible>>;
	type Response = ScriptStream;

	fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		self.inner.poll_ready(cx)
	}

	fn call(&mut self, routed: Routed) -> Self::Future {
		self.calls.lock().push(routed.clone());
		self.inner.call(routed.request)
	}
}

fn request(session: &str) -> TurnRequest {
	TurnRequest {
		params: Some(ChatParams {
			model: "provider/model".to_owned(),
			cache: Some(CacheHint { session_key: session.to_owned(), ..CacheHint::default() }),
			..ChatParams::default()
		}),
		..TurnRequest::default()
	}
}

fn service(script: RoutedScript, blocks: Arc<Mutex<BlockTable>>) -> Select<RoutedScript> {
	Select::new(script, Arc::new(TwoCredentials), Arc::new(FixedLeases), blocks)
}

fn lease_id(routed: &Routed) -> u64 {
	routed
		.lease
		.as_ref()
		.expect("selected request has a lease")
		.credential_id()
}

fn block_key(id: u64) -> BlockKey {
	BlockKey::credential(&id.to_string())
}

fn now_ms() -> u64 {
	SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.unwrap()
		.as_millis() as u64
}

#[tokio::test]
async fn usage_limit_rotates_and_blocks_first_credential() {
	let script = Script::new([
		vec![error(
			turn_error::Kind::RateLimited,
			"The usage limit has been reached (code: usage_limit_reached)",
		)],
		vec![outcome()],
	]);
	let payload_calls = script.calls.clone();
	let script = RoutedScript::new(script);
	let routed_calls = script.calls.clone();
	let blocks = Arc::new(Mutex::new(BlockTable::new()));
	let policy = Arc::new(ResolvedModelPolicy {
		request_model_id: Some("wire-model".into()),
		..ResolvedModelPolicy::default()
	});
	let stream = service(script, Arc::clone(&blocks))
		.oneshot(ProviderRequest::new(request("session-1"), Some(Arc::clone(&policy))))
		.await
		.unwrap();
	let frames = stream.collect::<Vec<_>>().await;

	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["attempt", "outcome"]);
	let calls = routed_calls.lock();
	assert_eq!(calls.len(), 2);
	assert_eq!(lease_id(&calls[0]), FIRST_ID);
	assert_eq!(lease_id(&calls[1]), SECOND_ID);
	assert_eq!(calls[0].credential_metadata.as_ref(), Some(&credential_metadata(FIRST_ID)));
	assert_eq!(
		calls[1].credential_metadata.as_ref(),
		Some(&credential_metadata(SECOND_ID)),
		"credential rotation must resolve metadata for the replacement lease"
	);
	assert!(
		calls
			.iter()
			.all(|call| { Arc::ptr_eq(call.model_policy.as_ref().expect("model policy"), &policy) })
	);
	assert!(
		payload_calls
			.lock()
			.iter()
			.all(|request| request.props.is_none())
	);
	assert!(
		blocks
			.lock()
			.blocked_for_ms(&block_key(FIRST_ID), now_ms())
			.is_some()
	);
	match frames[0].event.as_ref() {
		Some(turn_event::Event::Attempt(attempt)) => assert_eq!(attempt.number, 2),
		other => panic!("expected attempt frame, got {other:?}"),
	}
}

#[tokio::test]
async fn concurrency_cap_does_not_rotate() {
	let script = RoutedScript::new(Script::new([vec![error(
		turn_error::Kind::RateLimited,
		"Too many concurrent requests",
	)]]));
	let calls = script.calls.clone();
	let blocks = Arc::new(Mutex::new(BlockTable::new()));
	let stream = service(script, Arc::clone(&blocks))
		.oneshot(ProviderRequest::new(request("session-2"), None))
		.await
		.unwrap();
	let frames = stream.collect::<Vec<_>>().await;

	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["error"]);
	assert_eq!(calls.lock().len(), 1);
	assert!(
		blocks
			.lock()
			.blocked_for_ms(&block_key(FIRST_ID), now_ms())
			.is_none()
	);
}

#[tokio::test]
async fn sticky_session_reuses_successful_credential() {
	let script = RoutedScript::new(Script::new([vec![outcome()], vec![outcome()]]));
	let calls = script.calls.clone();
	let blocks = Arc::new(Mutex::new(BlockTable::new()));
	let mut select = service(script, blocks);

	let first = select
		.ready()
		.await
		.unwrap()
		.call(ProviderRequest::new(request("sticky-session"), None))
		.await
		.unwrap();
	assert_eq!(
		first
			.collect::<Vec<_>>()
			.await
			.iter()
			.map(kind_of)
			.collect::<Vec<_>>(),
		["outcome"]
	);
	let second = select
		.ready()
		.await
		.unwrap()
		.call(ProviderRequest::new(request("sticky-session"), None))
		.await
		.unwrap();
	assert_eq!(
		second
			.collect::<Vec<_>>()
			.await
			.iter()
			.map(kind_of)
			.collect::<Vec<_>>(),
		["outcome"]
	);

	let calls = calls.lock();
	assert_eq!(calls.len(), 2);
	assert_eq!(lease_id(&calls[0]), FIRST_ID);
	assert_eq!(lease_id(&calls[1]), FIRST_ID);
}

#[tokio::test]
async fn unpolled_cancel_dispatches_nothing_and_preserves_committed_session_pin() {
	let script = RoutedScript::new(Script::new([vec![outcome()], vec![outcome()], vec![outcome()]]));
	let calls = script.calls.clone();
	let ranking = Arc::new(Mutex::new([FIRST_ID, SECOND_ID].into_iter().collect()));
	let mut select = Select::new(
		script,
		Arc::new(RankedCredentials(Arc::clone(&ranking))),
		Arc::new(FixedLeases),
		Arc::new(Mutex::new(BlockTable::new())),
	);

	let first = select
		.ready()
		.await
		.unwrap()
		.call(ProviderRequest::new(request("transactional-session"), None))
		.await
		.unwrap();
	assert_eq!(first.collect::<Vec<_>>().await.len(), 1);

	*ranking.lock() = [SECOND_ID].into_iter().collect();
	let cancelled = select
		.ready()
		.await
		.unwrap()
		.call(ProviderRequest::new(request("transactional-session"), None))
		.await
		.unwrap();
	drop(cancelled);
	assert_eq!(
		calls.lock().len(),
		1,
		"dropping an unpolled selection must not dispatch its candidate",
	);

	*ranking.lock() = [SECOND_ID, FIRST_ID].into_iter().collect();
	let resumed = select
		.ready()
		.await
		.unwrap()
		.call(ProviderRequest::new(request("transactional-session"), None))
		.await
		.unwrap();
	assert_eq!(resumed.collect::<Vec<_>>().await.len(), 1);

	let calls = calls.lock();
	assert_eq!(calls.len(), 2);
	assert_eq!(lease_id(&calls[0]), FIRST_ID);
	assert_eq!(
		lease_id(&calls[1]),
		FIRST_ID,
		"an unpolled candidate must neither dispatch nor replace the authoritative pin"
	);
}

#[tokio::test]
async fn all_candidates_blocked_returns_typed_auth_without_dispatch() {
	let script = RoutedScript::new(Script::new([]));
	let calls = script.calls.clone();
	let blocks = Arc::new(Mutex::new(BlockTable::new()));
	{
		let mut table = blocks.lock();
		table.block(block_key(FIRST_ID), 0, u64::MAX);
		table.block(block_key(SECOND_ID), 0, u64::MAX);
	}
	let stream = service(script, blocks)
		.oneshot(ProviderRequest::new(request("blocked-session"), None))
		.await
		.unwrap();
	let frames = stream.collect::<Vec<_>>().await;

	assert_eq!(calls.lock().len(), 0);
	assert_eq!(frames.len(), 1);
	match frames[0].event.as_ref() {
		Some(turn_event::Event::Error(err)) => {
			assert_eq!(err.kind(), turn_error::Kind::Auth);
			assert!(
				err.detail
					.starts_with("no usable credential for provider/model: 2 blocked until ")
			);
		},
		other => panic!("expected typed auth error, got {other:?}"),
	}
}

#[tokio::test]
async fn output_before_usage_limit_bars_redispatch() {
	let script = RoutedScript::new(Script::new([vec![
		part_delta(),
		error(
			turn_error::Kind::RateLimited,
			"The usage limit has been reached (code: usage_limit_reached)",
		),
	]]));
	let calls = script.calls.clone();
	let blocks = Arc::new(Mutex::new(BlockTable::new()));
	let stream = service(script, blocks)
		.oneshot(ProviderRequest::new(request("session-3"), None))
		.await
		.unwrap();
	let frames = stream.collect::<Vec<_>>().await;

	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["part_delta", "error"]);
	assert_eq!(calls.lock().len(), 1);
}

#[tokio::test]
async fn stale_session_401_neither_rotates_nor_blocks() {
	// The credential is fine; the replayed server-side state is not.
	// Rotating or blocking here wastes a healthy account.
	let script = RoutedScript::new(Script::new([vec![error(
		turn_error::Kind::Auth,
		"HTTP 401 {\"error\":{\"code\":\"previous_response_not_found\",\"message\":\"Item with id \
		 'rs_123' not found.\"}}",
	)]]));
	let routed_calls = script.calls.clone();
	let blocks = Arc::new(Mutex::new(BlockTable::new()));
	let stream = service(script, Arc::clone(&blocks))
		.oneshot(ProviderRequest::new(request("session-stale"), None))
		.await
		.unwrap();
	let frames = stream.collect::<Vec<_>>().await;

	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["error"]);
	assert_eq!(routed_calls.lock().len(), 1, "no rotation re-dispatch");
	assert!(blocks.lock().earliest_unblock_ms(0).is_none(), "no block recorded");
}
