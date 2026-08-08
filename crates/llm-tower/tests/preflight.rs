//! Preflight admission-gate acceptance tests.

use std::sync::Arc;

use futures::StreamExt;
use omp_llm_tower::{
	preflight::{Admission, PreflightConfig, PreflightLayer, UsageOracle},
	recovery::classify_turn_error,
	testing::{Script, kind_of, outcome, part_start},
};
use omp_proto::inference::v1::{ChatParams, TurnError, TurnRequest, turn_error, turn_event};
use tower::{Layer, ServiceExt};

#[derive(Clone)]
struct FixedOracle(Admission);

impl UsageOracle for FixedOracle {
	fn admit(&self, _model: &str) -> Admission {
		self.0.clone()
	}
}

fn request() -> TurnRequest {
	TurnRequest {
		turn_id: "turn-preflight".to_owned(),
		params: Some(ChatParams { model: "provider/model".to_owned(), ..ChatParams::default() }),
		..TurnRequest::default()
	}
}

fn layer(admission: Admission, fail_closed: bool) -> PreflightLayer {
	PreflightLayer::new(Arc::new(FixedOracle(admission)), PreflightConfig { fail_closed })
}

fn error_from(frames: &[omp_proto::inference::v1::TurnEvent]) -> &TurnError {
	assert_eq!(frames.len(), 1);
	match frames[0].event.as_ref() {
		Some(turn_event::Event::Error(error)) => error,
		other => panic!("expected one error frame, got {other:?}"),
	}
}

#[tokio::test]
async fn allow_passes_frames_and_request_through() {
	let script = Script::new([vec![part_start(), outcome()]]);
	let calls = Arc::clone(&script.calls);
	let frames = layer(Admission::Allow, true)
		.layer(script)
		.oneshot(request())
		.await
		.unwrap()
		.collect::<Vec<_>>()
		.await;

	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["part_start", "outcome"]);
	let calls = calls.lock();
	assert_eq!(calls.len(), 1);
	assert_eq!(calls[0].params.as_ref().unwrap().model, "provider/model");
}

#[tokio::test]
async fn quota_denial_is_one_typed_frame_without_dispatch() {
	let script = Script::default();
	let calls = Arc::clone(&script.calls);
	let frames = layer(
		Admission::DenyQuota {
			detail:         "Monthly usage limit reached".to_owned(),
			retry_after_ms: 2_500,
		},
		true,
	)
	.layer(script)
	.oneshot(request())
	.await
	.unwrap()
	.collect::<Vec<_>>()
	.await;

	let error = error_from(&frames);
	assert_eq!(error.kind(), turn_error::Kind::RateLimited);
	assert_eq!(error.retry_after_ms, 2_500);
	assert_eq!(error.detail, "Monthly usage limit reached");
	assert!(calls.lock().is_empty());
	assert!(classify_turn_error(error).rate_limit.is_some());
}

#[tokio::test]
async fn auth_denial_is_typed_auth_without_dispatch() {
	let script = Script::default();
	let calls = Arc::clone(&script.calls);
	let frames =
		layer(Admission::DenyAuth { detail: "account credentials expired".to_owned() }, true)
			.layer(script)
			.oneshot(request())
			.await
			.unwrap()
			.collect::<Vec<_>>()
			.await;

	let error = error_from(&frames);
	assert_eq!(error.kind(), turn_error::Kind::Auth);
	assert_eq!(error.detail, "account credentials expired");
	assert!(calls.lock().is_empty());
}

#[tokio::test]
async fn unknown_fails_closed_by_default() {
	let script = Script::default();
	let calls = Arc::clone(&script.calls);
	let frames = PreflightLayer::new(
		Arc::new(FixedOracle(Admission::Unknown { detail: "usage service timed out".to_owned() })),
		PreflightConfig::default(),
	)
	.layer(script)
	.oneshot(request())
	.await
	.unwrap()
	.collect::<Vec<_>>()
	.await;

	let error = error_from(&frames);
	assert_eq!(error.kind(), turn_error::Kind::Auth);
	assert_eq!(error.detail, "preflight unavailable: usage service timed out");
	assert!(calls.lock().is_empty());
}

#[tokio::test]
async fn unknown_can_fail_open() {
	let script = Script::new([vec![outcome()]]);
	let calls = Arc::clone(&script.calls);
	let frames = layer(Admission::Unknown { detail: "usage service timed out".to_owned() }, false)
		.layer(script)
		.oneshot(request())
		.await
		.unwrap()
		.collect::<Vec<_>>()
		.await;

	assert_eq!(frames.iter().map(kind_of).collect::<Vec<_>>(), ["outcome"]);
	assert_eq!(calls.lock().len(), 1);
}
