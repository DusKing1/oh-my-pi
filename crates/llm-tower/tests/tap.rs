//! Diagnostics tap acceptance tests.

use std::sync::{
	Arc,
	atomic::{AtomicUsize, Ordering},
};

use futures::StreamExt;
use omp_llm_tower::{
	tap::{FrameSink, TapLayer},
	testing::{Script, error, kind_of, outcome, part_delta, part_start},
};
use omp_proto::inference::v1::{TurnEvent, TurnRequest, turn_error};
use parking_lot::Mutex;
use tower::{Layer, Service, ServiceExt};

#[derive(Default)]
struct Recorder {
	requests: AtomicUsize,
	frames:   Mutex<Vec<&'static str>>,
	ends:     AtomicUsize,
}

impl FrameSink for Recorder {
	fn on_request(&self, _req: &TurnRequest) {
		self.requests.fetch_add(1, Ordering::SeqCst);
	}

	fn on_frame(&self, frame: &TurnEvent) {
		self.frames.lock().push(kind_of(frame));
	}

	fn on_end(&self) {
		self.ends.fetch_add(1, Ordering::SeqCst);
	}
}

#[tokio::test]
async fn observes_everything_and_alters_nothing() {
	let sink = Arc::new(Recorder::default());
	let script =
		Script::new([vec![part_start(), part_delta(), error(turn_error::Kind::Upstream, "boom")]]);
	let mut svc = TapLayer::new(sink.clone()).layer(script);

	let frames: Vec<_> = svc
		.ready()
		.await
		.unwrap()
		.call(TurnRequest::default())
		.await
		.unwrap()
		.collect()
		.await;

	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), [
		"part_start",
		"part_delta",
		"error"
	]);
	assert_eq!(sink.requests.load(Ordering::SeqCst), 1);
	assert_eq!(&*sink.frames.lock(), &["part_start", "part_delta", "error"]);
	assert_eq!(sink.ends.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn early_drop_still_closes_the_observation_window() {
	let sink = Arc::new(Recorder::default());
	let script = Script::new([vec![part_start(), outcome()]]);
	let mut svc = TapLayer::new(sink.clone()).layer(script);

	let mut stream = Box::pin(
		svc.ready()
			.await
			.unwrap()
			.call(TurnRequest::default())
			.await
			.unwrap(),
	);
	let first = stream.next().await.unwrap();
	assert_eq!(kind_of(&first), "part_start");
	drop(stream); // consumer cancels mid-stream

	assert_eq!(sink.ends.load(Ordering::SeqCst), 1, "drop must fire on_end exactly once");
}
