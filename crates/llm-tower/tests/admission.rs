//! In-flight admission acceptance tests.

use std::{
	sync::{
		Arc,
		atomic::{AtomicUsize, Ordering},
	},
	task::{Context, Poll},
};

use futures::StreamExt;
use omp_llm_tower::{
	admission::AdmissionLayer,
	testing::{kind_of, outcome},
};
use omp_proto::inference::v1::{TurnEvent, TurnRequest};
use tower::{Layer, Service, ServiceExt};

/// Service whose streams stay open until released, counting concurrency.
#[derive(Clone)]
struct Gated {
	active:  Arc<AtomicUsize>,
	peak:    Arc<AtomicUsize>,
	release: Arc<tokio::sync::Notify>,
}

/// Stream yielding one outcome only after the shared release fires.
struct GatedStream {
	active:  Arc<AtomicUsize>,
	release: Arc<tokio::sync::Notify>,
	state:   u8,
}

impl futures::Stream for GatedStream {
	type Item = TurnEvent;

	fn poll_next(
		mut self: std::pin::Pin<&mut Self>,
		cx: &mut Context<'_>,
	) -> Poll<Option<TurnEvent>> {
		match self.state {
			0 => {
				let release = Arc::clone(&self.release);
				let waker = cx.waker().clone();
				self.state = 1;
				tokio::spawn(async move {
					release.notified().await;
					waker.wake();
				});
				Poll::Pending
			},
			1 => {
				self.state = 2;
				Poll::Ready(Some(outcome()))
			},
			_ => Poll::Ready(None),
		}
	}
}

impl Drop for GatedStream {
	fn drop(&mut self) {
		self.active.fetch_sub(1, Ordering::SeqCst);
	}
}

impl Service<TurnRequest> for Gated {
	type Error = std::convert::Infallible;
	type Future = std::future::Ready<Result<GatedStream, Self::Error>>;
	type Response = GatedStream;

	fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		Poll::Ready(Ok(()))
	}

	fn call(&mut self, _req: TurnRequest) -> Self::Future {
		let now = self.active.fetch_add(1, Ordering::SeqCst) + 1;
		self.peak.fetch_max(now, Ordering::SeqCst);
		std::future::ready(Ok(GatedStream {
			active:  Arc::clone(&self.active),
			release: Arc::clone(&self.release),
			state:   0,
		}))
	}
}

#[tokio::test]
async fn permit_bounds_concurrency_and_lives_for_the_stream() {
	let gated = Gated {
		active:  Arc::new(AtomicUsize::new(0)),
		peak:    Arc::new(AtomicUsize::new(0)),
		release: Arc::new(tokio::sync::Notify::new()),
	};
	let peak = Arc::clone(&gated.peak);
	let release = Arc::clone(&gated.release);
	let layer = AdmissionLayer::new(2);

	let mut tasks = Vec::new();
	for _ in 0..4 {
		let mut svc = layer.layer(gated.clone());
		tasks.push(tokio::spawn(async move {
			let stream = svc
				.ready()
				.await
				.unwrap()
				.call(TurnRequest::default())
				.await
				.unwrap();
			stream
				.map(|f| kind_of(&f).to_owned())
				.collect::<Vec<_>>()
				.await
		}));
	}

	// Let the first two acquire permits and park in their streams.
	tokio::time::sleep(std::time::Duration::from_millis(50)).await;
	assert_eq!(peak.load(Ordering::SeqCst), 2, "third and fourth attempts must queue");

	// Release everyone; queued attempts proceed as permits free up.
	for _ in 0..8 {
		release.notify_waiters();
		tokio::time::sleep(std::time::Duration::from_millis(20)).await;
	}
	for task in tasks {
		assert_eq!(task.await.unwrap(), ["outcome"]);
	}
	assert!(peak.load(Ordering::SeqCst) <= 2, "permit pool must never be exceeded");
}
