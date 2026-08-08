//! Diagnostics tap: observe requests and frames without touching them.
//!
//! The tap is pure observation — it never mutates, reorders, delays, or
//! drops frames, and a panicking sink is the sink author's bug, not a
//! stream failure. Redaction is the SINK's obligation: frames at this
//! boundary carry user content (part deltas) and provider error prose, so
//! any sink that persists or ships them off-process MUST scrub before the
//! bytes leave (the transport research documented verbatim-authorization
//! logging as a real production leak; do not recreate it).

use std::{
	pin::Pin,
	sync::Arc,
	task::{Context, Poll},
};

use futures::{Stream, TryFutureExt};
use omp_proto::inference::v1::{TurnEvent, TurnRequest};
use tower::{Layer, Service};

use crate::envelope::TurnRequestEnvelope;

/// Observer for one attempt's request and frames.
///
/// Callbacks run inline on the stream path — keep them cheap and
/// non-blocking; hand heavy work to a channel.
pub trait FrameSink: Send + Sync + 'static {
	/// Called once per dispatch, before the inner call.
	fn on_request(&self, req: &TurnRequest);
	/// Called for every frame as it passes through.
	fn on_frame(&self, frame: &TurnEvent);
	/// Called when the stream ends (after the terminal frame or EOF).
	fn on_end(&self);
}

/// [`Layer`] producing observing [`Tap`] services.
#[derive(Clone)]
pub struct TapLayer {
	sink: Arc<dyn FrameSink>,
}

impl TapLayer {
	/// Layer observing through `sink`.
	pub fn new(sink: Arc<dyn FrameSink>) -> Self {
		Self { sink }
	}
}

impl<S> Layer<S> for TapLayer {
	type Service = Tap<S>;

	fn layer(&self, inner: S) -> Self::Service {
		Tap { inner, sink: Arc::clone(&self.sink) }
	}
}

/// Observing wrapper around an attempt service.
#[derive(Clone)]
pub struct Tap<S> {
	inner: S,
	sink:  Arc<dyn FrameSink>,
}

impl<S> Tap<S> {
	/// Wraps `inner`, observing through `sink`.
	pub fn new(inner: S, sink: Arc<dyn FrameSink>) -> Self {
		Self { inner, sink }
	}
}

pin_project_lite::pin_project! {
	/// Stream that mirrors every frame into its [`FrameSink`].
	pub struct Tapped<St> {
		#[pin]
		inner: St,
		sink: Arc<dyn FrameSink>,
		ended: bool,
	}

	impl<St> PinnedDrop for Tapped<St> {
		fn drop(this: Pin<&mut Self>) {
			// A consumer that drops the stream early (cancellation) still
			// ends the observation window.
			let this = this.project();
			if !*this.ended {
				*this.ended = true;
				this.sink.on_end();
			}
		}
	}
}

impl<St: Stream<Item = TurnEvent>> Stream for Tapped<St> {
	type Item = TurnEvent;

	fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<TurnEvent>> {
		let this = self.project();
		match this.inner.poll_next(cx) {
			Poll::Ready(Some(frame)) => {
				this.sink.on_frame(&frame);
				Poll::Ready(Some(frame))
			},
			Poll::Ready(None) => {
				if !*this.ended {
					*this.ended = true;
					this.sink.on_end();
				}
				Poll::Ready(None)
			},
			Poll::Pending => Poll::Pending,
		}
	}
}

impl<S, St, R> Service<R> for Tap<S>
where
	R: TurnRequestEnvelope,
	S: Service<R, Response = St> + Clone + Send + 'static,
	S::Future: Send,
	S::Error: Send + 'static,
	St: Stream<Item = TurnEvent> + Send + 'static,
{
	type Error = S::Error;
	type Response = Tapped<St>;

	type Future = impl Future<Output = Result<Self::Response, S::Error>> + Send;

	fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), S::Error>> {
		self.inner.poll_ready(cx)
	}

	fn call(&mut self, req: R) -> Self::Future {
		let clone = self.inner.clone();
		let mut inner = std::mem::replace(&mut self.inner, clone);
		let sink = Arc::clone(&self.sink);
		sink.on_request(req.request());
		inner
			.call(req)
			.map_ok(move |stream| Tapped { inner: stream, sink, ended: false })
	}
}
