//! Tower readiness-contract tests.
//!
//! A readiness-sensitive inner service (concurrency limit, load balancer)
//! reserves capacity in `poll_ready` and releases it when `call` consumes
//! the reservation. A layer that polls inner readiness and then short-
//! circuits without calling — or parks on its own gate while holding the
//! reservation — leaks that slot. These tests use a reserving mock that
//! counts leaked reservations on drop.

use std::{
	convert::Infallible,
	sync::{
		Arc,
		atomic::{AtomicUsize, Ordering},
	},
	task::{Context, Poll},
};

use futures::StreamExt;
use omp_llm_error::policy::BlockTable;
use omp_llm_tower::{
	admission::AdmissionLayer,
	envelope::ProviderRequest,
	preflight::{Admission, PreflightConfig, PreflightLayer, UsageOracle},
	select::{
		CredentialCandidates, CredentialLease, CredentialPool, LeaseSource, Routed, SelectLayer,
	},
	testing::{Script, ScriptStream, kind_of, outcome},
};
use omp_proto::inference::v1::{TurnEvent, TurnRequest};
use parking_lot::Mutex;
use tower::{Layer, Service, ServiceExt};

/// Shared ledger of reservations taken, released, and leaked.
#[derive(Default)]
struct Ledger {
	reserved: AtomicUsize,
	released: AtomicUsize,
	leaked:   AtomicUsize,
}

/// Readiness-sensitive mock: `poll_ready` reserves capacity, `call`
/// releases it; dropping the service while a reservation is outstanding
/// counts as a leak (exactly what happens to a swapped-out clone that was
/// polled ready but never called).
struct Reserving<Req> {
	inner:   Script,
	ledger:  Arc<Ledger>,
	holding: bool,
	_req:    std::marker::PhantomData<fn(Req)>,
}

impl<Req> Reserving<Req> {
	fn new(inner: Script, ledger: Arc<Ledger>) -> Self {
		Self { inner, ledger, holding: false, _req: std::marker::PhantomData }
	}
}

impl<Req> Clone for Reserving<Req> {
	fn clone(&self) -> Self {
		// Clones start without a reservation, like tower's own
		// readiness-holding middlewares.
		Self::new(self.inner.clone(), Arc::clone(&self.ledger))
	}
}

impl<Req> Drop for Reserving<Req> {
	fn drop(&mut self) {
		if self.holding {
			self.ledger.leaked.fetch_add(1, Ordering::SeqCst);
		}
	}
}

trait IntoTurnRequest {
	fn into_turn_request(self) -> TurnRequest;
}
impl IntoTurnRequest for TurnRequest {
	fn into_turn_request(self) -> TurnRequest {
		self
	}
}
impl IntoTurnRequest for Routed {
	fn into_turn_request(self) -> TurnRequest {
		self.request
	}
}

impl<Req: IntoTurnRequest + 'static> Service<Req> for Reserving<Req> {
	type Error = Infallible;
	type Future = std::future::Ready<Result<ScriptStream, Infallible>>;
	type Response = ScriptStream;

	fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Infallible>> {
		if !self.holding {
			self.holding = true;
			self.ledger.reserved.fetch_add(1, Ordering::SeqCst);
		}
		Poll::Ready(Ok(()))
	}

	fn call(&mut self, req: Req) -> Self::Future {
		assert!(self.holding, "call without poll_ready reservation violates the tower contract");
		self.holding = false;
		self.ledger.released.fetch_add(1, Ordering::SeqCst);
		self.inner.call(req.into_turn_request())
	}
}

struct DenyAll;
impl UsageOracle for DenyAll {
	fn admit(&self, _model: &str) -> Admission {
		Admission::DenyAuth { detail: "denied".to_owned() }
	}
}

struct AllowAll;
impl UsageOracle for AllowAll {
	fn admit(&self, _model: &str) -> Admission {
		Admission::Allow
	}
}

struct EmptyPool;
impl CredentialPool for EmptyPool {
	fn candidates(&self, _model: &str) -> CredentialCandidates {
		CredentialCandidates::new()
	}
}

struct NoLeases;
impl LeaseSource for NoLeases {
	fn lease(&self, _id: u64) -> Option<CredentialLease> {
		None
	}
}

#[tokio::test]
async fn preflight_denial_reserves_nothing() {
	let ledger = Arc::new(Ledger::default());
	let inner = Reserving::<TurnRequest>::new(Script::new([vec![outcome()]]), Arc::clone(&ledger));
	{
		let mut svc = PreflightLayer::new(Arc::new(DenyAll), PreflightConfig::default()).layer(inner);
		let frames: Vec<TurnEvent> = svc
			.ready()
			.await
			.unwrap()
			.call(TurnRequest::default())
			.await
			.unwrap()
			.collect()
			.await;
		assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["error"]);
	}
	assert_eq!(ledger.reserved.load(Ordering::SeqCst), 0, "denial must not touch inner readiness");
	assert_eq!(ledger.leaked.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn preflight_allow_balances_reservations() {
	let ledger = Arc::new(Ledger::default());
	let inner = Reserving::<TurnRequest>::new(Script::new([vec![outcome()]]), Arc::clone(&ledger));
	{
		let mut svc =
			PreflightLayer::new(Arc::new(AllowAll), PreflightConfig::default()).layer(inner);
		let frames: Vec<TurnEvent> = svc
			.ready()
			.await
			.unwrap()
			.call(TurnRequest::default())
			.await
			.unwrap()
			.collect()
			.await;
		assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["outcome"]);
	}
	assert_eq!(ledger.reserved.load(Ordering::SeqCst), ledger.released.load(Ordering::SeqCst));
	assert_eq!(ledger.leaked.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn select_without_credentials_reserves_nothing() {
	let ledger = Arc::new(Ledger::default());
	let inner = Reserving::<Routed>::new(Script::new([vec![outcome()]]), Arc::clone(&ledger));
	{
		let mut svc = SelectLayer::new(
			Arc::new(EmptyPool),
			Arc::new(NoLeases),
			Arc::new(Mutex::new(BlockTable::new())),
		)
		.layer(inner);
		let frames: Vec<TurnEvent> = svc
			.ready()
			.await
			.unwrap()
			.call(ProviderRequest::new(TurnRequest::default(), None))
			.await
			.unwrap()
			.collect()
			.await;
		assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["error"]);
	}
	assert_eq!(ledger.reserved.load(Ordering::SeqCst), 0);
	assert_eq!(ledger.leaked.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn admission_acquires_permit_before_inner_readiness() {
	let ledger = Arc::new(Ledger::default());
	let inner = Reserving::<TurnRequest>::new(
		Script::new([vec![outcome()], vec![outcome()]]),
		Arc::clone(&ledger),
	);
	let layer = AdmissionLayer::new(1);
	{
		// Hold the only permit with a live stream, then start a second call:
		// while it queues, inner readiness must remain untouched.
		let mut first = layer.layer(inner.clone());
		let mut held = Box::pin(
			first
				.ready()
				.await
				.unwrap()
				.call(TurnRequest::default())
				.await
				.unwrap(),
		);
		let after_first = ledger.reserved.load(Ordering::SeqCst);

		let mut second = layer.layer(inner);
		let queued = tokio::spawn(async move {
			let stream = second
				.ready()
				.await
				.unwrap()
				.call(TurnRequest::default())
				.await
				.unwrap();
			stream.collect::<Vec<_>>().await
		});
		tokio::time::sleep(std::time::Duration::from_millis(50)).await;
		assert_eq!(
			ledger.reserved.load(Ordering::SeqCst),
			after_first,
			"queued call must not reserve inner readiness while parked on the semaphore"
		);

		// Drain the first stream to release the permit; the queued call
		// then reserves, dispatches, and completes.
		while held.next().await.is_some() {}
		drop(held);
		let frames = queued.await.unwrap();
		assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["outcome"]);
	}
	assert_eq!(ledger.reserved.load(Ordering::SeqCst), ledger.released.load(Ordering::SeqCst));
	assert_eq!(ledger.leaked.load(Ordering::SeqCst), 0);
}
