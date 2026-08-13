//! In-flight admission: bounded concurrent provider attempts.
//!
//! One [`Admission`] instance guards one pool (typically per provider or
//! per endpoint; the gateway decides the granularity by where it installs
//! the layer). A permit is held until the provider emits a terminal frame or
//! ends its stream. Releasing before either point would let N+1 streams run
//! against an N-slot provider; retaining it afterward needlessly blocks the
//! next attempt.
//!
//! Waiting callers queue on the semaphore (backpressure, FIFO within
//! tokio's fairness); there is deliberately no reject mode — shedding is
//! the rate-limit lane's job when the PROVIDER says no, and queueing is
//! correct when WE are the constraint.

use std::{
	pin::Pin,
	sync::Arc,
	task::{Context, Poll},
};

use futures::Stream;
use omp_proto::inference::v1::{TurnEvent, turn_event};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tower::{Layer, Service, ServiceExt};

use crate::envelope::TurnRequestEnvelope;

/// [`Layer`] producing [`Admission`] services sharing one permit pool.
#[derive(Clone, Debug)]
pub struct AdmissionLayer {
	permits: Arc<Semaphore>,
}

impl AdmissionLayer {
	/// Pool admitting at most `max_inflight` concurrent attempt streams.
	pub fn new(max_inflight: usize) -> Self {
		Self { permits: Arc::new(Semaphore::new(max_inflight)) }
	}
}

impl<S> Layer<S> for AdmissionLayer {
	type Service = Admission<S>;

	fn layer(&self, inner: S) -> Self::Service {
		Admission { inner, permits: Arc::clone(&self.permits) }
	}
}

/// Attempt service gated by a shared in-flight permit pool.
#[derive(Clone, Debug)]
pub struct Admission<S> {
	inner:   S,
	permits: Arc<Semaphore>,
}

impl<S> Admission<S> {
	/// Wraps `inner` behind `permits`.
	pub const fn new(inner: S, permits: Arc<Semaphore>) -> Self {
		Self { inner, permits }
	}
}

pin_project_lite::pin_project! {
	/// Stream holding an admission permit until its terminal frame or end.
	pub struct Permitted<St> {
		#[pin]
		inner: St,
		permit: Option<OwnedSemaphorePermit>,
	}
}

impl<St: Stream<Item = TurnEvent>> Stream for Permitted<St> {
	type Item = TurnEvent;

	fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<TurnEvent>> {
		let this = self.project();
		match this.inner.poll_next(cx) {
			Poll::Ready(item) => {
				let terminal = item.as_ref().is_none_or(|frame| {
					matches!(
						&frame.event,
						Some(turn_event::Event::Outcome(_) | turn_event::Event::Error(_))
					)
				});
				if terminal {
					drop(this.permit.take());
				}
				Poll::Ready(item)
			},
			Poll::Pending => Poll::Pending,
		}
	}
}

impl<S, St, R> Service<R> for Admission<S>
where
	R: TurnRequestEnvelope,
	S: Service<R, Response = St> + Clone + Send + 'static,
	S::Future: Send,
	S::Error: Send + 'static,
	St: Stream<Item = TurnEvent> + Send + 'static,
{
	type Error = S::Error;
	type Response = Permitted<St>;

	type Future = impl Future<Output = Result<Self::Response, S::Error>> + Send;

	fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), S::Error>> {
		// Permit acquisition must precede inner readiness: polling a
		// readiness-sensitive inner service ready and then parking on the
		// semaphore would hold its reserved slot for the whole queue wait.
		Poll::Ready(Ok(()))
	}

	fn call(&mut self, req: R) -> Self::Future {
		let clone = self.inner.clone();
		let mut inner = std::mem::replace(&mut self.inner, clone);
		let permits = Arc::clone(&self.permits);
		async move {
			// A closed semaphore is impossible here (we never close it), so
			// acquire can only pend, which is exactly the backpressure we
			// want.
			#[allow(clippy::expect_used, reason = "semaphore is never closed")]
			let permit = permits
				.acquire_owned()
				.await
				.expect("admission semaphore closed");
			let stream = inner.ready().await?.call(req).await?;
			Ok(Permitted { inner: stream, permit: Some(permit) })
		}
	}
}
