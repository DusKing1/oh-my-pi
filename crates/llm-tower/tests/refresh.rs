//! OAuth refresh layer acceptance tests.

use std::{
	future::Future,
	pin::Pin,
	sync::{
		Arc,
		atomic::{AtomicU64, AtomicUsize, Ordering},
	},
	time::{Duration, SystemTime, UNIX_EPOCH},
};

use futures::StreamExt;
use omp_llm_tower::{
	refresh::{CredentialRefresher, RefreshConfig, RefreshFailure, RefreshLayer},
	testing::{Script, error, kind_of, outcome, part_delta},
};
use omp_proto::inference::v1::{TurnEvent, TurnRequest, turn_error, turn_event};
use parking_lot::Mutex;
use tower::{Layer, Service, ServiceExt};

struct MockRefresher {
	expires_at_ms: AtomicU64,
	calls:         AtomicUsize,
	active:        AtomicUsize,
	max_active:    AtomicUsize,
	force_flags:   Mutex<Vec<bool>>,
	delay_ms:      u64,
}

impl MockRefresher {
	fn new(expires_at_ms: u64) -> Arc<Self> {
		Arc::new(Self {
			expires_at_ms: AtomicU64::new(expires_at_ms),
			calls:         AtomicUsize::new(0),
			active:        AtomicUsize::new(0),
			max_active:    AtomicUsize::new(0),
			force_flags:   Mutex::new(Vec::new()),
			delay_ms:      0,
		})
	}

	fn delayed(expires_at_ms: u64, delay_ms: u64) -> Arc<Self> {
		Arc::new(Self {
			expires_at_ms: AtomicU64::new(expires_at_ms),
			calls: AtomicUsize::new(0),
			active: AtomicUsize::new(0),
			max_active: AtomicUsize::new(0),
			force_flags: Mutex::new(Vec::new()),
			delay_ms,
		})
	}
}

impl CredentialRefresher for MockRefresher {
	fn expires_at_ms(&self) -> Option<u64> {
		Some(self.expires_at_ms.load(Ordering::SeqCst))
	}

	fn refresh(
		&self,
		force: bool,
	) -> Pin<Box<dyn Future<Output = Result<(), RefreshFailure>> + Send + '_>> {
		self.calls.fetch_add(1, Ordering::SeqCst);
		self.force_flags.lock().push(force);
		Box::pin(async move {
			let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
			self.max_active.fetch_max(active, Ordering::SeqCst);
			if self.delay_ms > 0 {
				tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
			}
			self
				.expires_at_ms
				.store(now_ms().saturating_add(3_600_000), Ordering::SeqCst);
			self.active.fetch_sub(1, Ordering::SeqCst);
			Ok(())
		})
	}
}

fn now_ms() -> u64 {
	SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.unwrap()
		.as_millis() as u64
}

async fn run<S, St>(mut service: S) -> Vec<TurnEvent>
where
	S: Service<TurnRequest, Response = St>,
	S::Error: std::fmt::Debug,
	St: futures::Stream<Item = TurnEvent>,
{
	let stream = service
		.ready()
		.await
		.unwrap()
		.call(TurnRequest::default())
		.await
		.unwrap();
	stream.collect().await
}

#[tokio::test]
async fn proactive_refreshes_inside_skew() {
	let refresher = MockRefresher::new(now_ms().saturating_add(30_000));
	let script = Script::new([vec![outcome()]]);
	let service =
		RefreshLayer::new(refresher.clone(), RefreshConfig::default()).layer(script.clone());

	let frames = run(service).await;

	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["outcome"]);
	assert_eq!(script.calls.lock().len(), 1);
	assert_eq!(refresher.calls.load(Ordering::SeqCst), 1);
	assert_eq!(&*refresher.force_flags.lock(), &[false]);
}

#[tokio::test]
async fn proactive_does_not_refresh_far_from_expiry() {
	let refresher = MockRefresher::new(now_ms().saturating_add(3_600_000));
	let script = Script::new([vec![outcome()]]);
	let service = RefreshLayer::new(refresher.clone(), RefreshConfig::default()).layer(script);

	let frames = run(service).await;

	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["outcome"]);
	assert_eq!(refresher.calls.load(Ordering::SeqCst), 0);
	assert!(refresher.force_flags.lock().is_empty());
}

#[tokio::test]
async fn concurrent_proactive_calls_delegate_coalescing_to_refresher() {
	let refresher = MockRefresher::delayed(now_ms().saturating_add(30_000), 5);
	let script = Script::new([vec![outcome()], vec![outcome()]]);
	let layer = RefreshLayer::new(refresher.clone(), RefreshConfig::default());
	let first = layer.layer(script.clone());
	let second = layer.layer(script.clone());

	let (first_frames, second_frames) = tokio::join!(run(first), run(second));

	assert_eq!(first_frames.iter().map(kind_of).collect::<Vec<_>>(), ["outcome"]);
	assert_eq!(second_frames.iter().map(kind_of).collect::<Vec<_>>(), ["outcome"]);
	assert_eq!(script.calls.lock().len(), 2);
	assert_eq!(refresher.calls.load(Ordering::SeqCst), 2);
	assert_eq!(refresher.max_active.load(Ordering::SeqCst), 2);
	assert_eq!(&*refresher.force_flags.lock(), &[false, false]);
}

#[tokio::test]
async fn reactive_auth_failure_forces_refresh_and_redispatches() {
	let refresher = MockRefresher::new(now_ms().saturating_add(3_600_000));
	let script = Script::new([
		vec![error(turn_error::Kind::Auth, "HTTP 401 credential rejected")],
		vec![outcome()],
	]);
	let service =
		RefreshLayer::new(refresher.clone(), RefreshConfig::default()).layer(script.clone());

	let frames = run(service).await;

	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["attempt", "outcome"]);
	match frames[0].event.as_ref() {
		Some(turn_event::Event::Attempt(attempt)) => {
			assert_eq!(attempt.number, 2);
			assert_eq!(attempt.reason, "OAuth credential refreshed");
		},
		other => panic!("expected attempt frame, got {other:?}"),
	}
	assert_eq!(script.calls.lock().len(), 2);
	assert_eq!(refresher.calls.load(Ordering::SeqCst), 1);
	assert_eq!(&*refresher.force_flags.lock(), &[true]);
}

#[tokio::test]
async fn reactive_auth_failure_after_output_is_forwarded() {
	let refresher = MockRefresher::new(now_ms().saturating_add(3_600_000));
	let script = Script::new([vec![
		part_delta(),
		error(turn_error::Kind::Auth, "HTTP 401 credential rejected"),
	]]);
	let service =
		RefreshLayer::new(refresher.clone(), RefreshConfig::default()).layer(script.clone());

	let frames = run(service).await;

	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["part_delta", "error"]);
	assert_eq!(script.calls.lock().len(), 1);
	assert_eq!(refresher.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn oauth_expired_error_is_never_refreshed() {
	let refresher = MockRefresher::new(now_ms().saturating_add(3_600_000));
	let script = Script::new([vec![error(
		turn_error::Kind::Auth,
		"OAuth token endpoint returned invalid_grant: refresh token revoked",
	)]]);
	let service =
		RefreshLayer::new(refresher.clone(), RefreshConfig::default()).layer(script.clone());

	let frames = run(service).await;

	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["error"]);
	assert_eq!(script.calls.lock().len(), 1);
	assert_eq!(refresher.calls.load(Ordering::SeqCst), 0);
	assert!(refresher.force_flags.lock().is_empty());
}

#[tokio::test]
async fn stale_session_401_never_touches_the_refresher() {
	// Server-side replay state rides in on a 401; the token is valid and
	// refreshing it here would churn a healthy credential.
	let refresher = MockRefresher::new(now_ms().saturating_add(3_600_000));
	let script = Script::new([vec![error(
		turn_error::Kind::Auth,
		"HTTP 401 {\"error\":{\"code\":\"previous_response_not_found\",\"message\":\"Item with id \
		 'rs_123' not found.\"}}",
	)]]);
	let service =
		RefreshLayer::new(refresher.clone(), RefreshConfig::default()).layer(script.clone());

	let frames = run(service).await;

	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["error"]);
	assert_eq!(refresher.calls.load(Ordering::SeqCst), 0);
	assert_eq!(script.calls.lock().len(), 1);
}
