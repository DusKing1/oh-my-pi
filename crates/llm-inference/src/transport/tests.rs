use std::sync::{
	Arc,
	atomic::{AtomicUsize, Ordering},
};

use bytes::Bytes;
use futures::{StreamExt as _, stream};
use omp_core::Str;
use omp_llm_catalog::{OperationKind, ProviderId, RouteId};
use serde::Deserialize;
use tower::{Service as _, ServiceExt as _};

use super::{Frame, FramingProtocol, WebSocketTransport, cassette::*, http::HttpTransport};
use crate::{
	answer::{RealtimeEvent, RealtimeInput},
	body::{
		BodyFactoryHandle, BodySource, ByteStream, OneShotBody, RetryDecision, RetryDecisionReason,
	},
	call::{RealtimeModality, RealtimeRequest, Setting},
	codec::{
		Cancellation, Decoder, EncodedRequest, ProviderMetadataEvent, ProviderStateEvent,
		RawCompletion, RawEvent, RealtimeEvents, RealtimeWireCodec, RealtimeWireFrames,
		RequestMethod, SizeBounds, TransportAttempt, TransportRequest,
		openai_realtime::OpenAiRealtimeWireCodec,
	},
	error::{Error, ErrorDetail, ErrorKind, ErrorPhase, RetryAction},
	event::{BlockKind, ChatEvent, FinishReason},
	id::RequestId,
	receipt::{AttemptOutcome, ExecutionReceipt, ReasonId, Usage},
};

#[derive(Deserialize)]
struct LifecycleFixture {
	commit_rule: String,
	cases:       Vec<LifecycleCase>,
}
#[derive(Deserialize)]
struct LifecycleCase {
	id:       String,
	expected: Option<LifecycleExpected>,
}

#[derive(Deserialize)]
struct LifecycleExpected {
	commit_state: Option<String>,
	retry_action: Option<String>,
}

#[derive(Deserialize)]
struct RetryFixture {
	policy: String,
	cases:  Vec<RetryCase>,
}

#[derive(Deserialize)]
struct RetryCase {
	id:       String,
	expected: Option<RetryExpected>,
}

#[derive(Deserialize)]
struct RetryExpected {
	attempt_count: Option<u32>,
	retry_action:  Option<String>,
}

struct EmitDecoder;

impl Decoder for EmitDecoder {
	fn push(&mut self, _frame: Frame, emit: &mut dyn FnMut(RawEvent)) -> Result<(), Error> {
		emit(RawEvent::Chat(ChatEvent::TextDelta { index: 0, text: Str::from("visible") }));
		Ok(())
	}

	fn finish(&mut self, _emit: &mut dyn FnMut(RawEvent)) -> Result<(), Error> {
		Ok(())
	}
}

struct PreambleThenVisibleDecoder;

impl Decoder for PreambleThenVisibleDecoder {
	fn push(&mut self, _frame: Frame, emit: &mut dyn FnMut(RawEvent)) -> Result<(), Error> {
		emit(RawEvent::Metadata(ProviderMetadataEvent::ResponseId(Str::from("response"))));
		emit(RawEvent::Chat(ChatEvent::TextDelta { index: 0, text: Str::from("visible") }));
		Ok(())
	}

	fn finish(&mut self, _emit: &mut dyn FnMut(RawEvent)) -> Result<(), Error> {
		Ok(())
	}
}

struct MetadataOnlyDecoder;

impl Decoder for MetadataOnlyDecoder {
	fn push(&mut self, _frame: Frame, emit: &mut dyn FnMut(RawEvent)) -> Result<(), Error> {
		emit(RawEvent::Metadata(ProviderMetadataEvent::ResponseId(Str::from("response"))));
		Ok(())
	}

	fn finish(&mut self, _emit: &mut dyn FnMut(RawEvent)) -> Result<(), Error> {
		Ok(())
	}
}

struct StateThenExpiredDecoder;

impl Decoder for StateThenExpiredDecoder {
	fn push(&mut self, _frame: Frame, emit: &mut dyn FnMut(RawEvent)) -> Result<(), Error> {
		emit(RawEvent::ProviderState(ProviderStateEvent::Checkpoint {
			id:   Some(Str::from("checkpoint")),
			data: Bytes::from_static(b"opaque"),
		}));
		let error = Error::new(
			ErrorKind::SessionExpired,
			ErrorPhase::Handshake,
			RetryAction::ReseedSession,
			ExecutionReceipt::default(),
		);
		emit(RawEvent::Failure(error));
		Ok(())
	}

	fn finish(&mut self, _emit: &mut dyn FnMut(RawEvent)) -> Result<(), Error> {
		Ok(())
	}
}

struct CompletionOnlyDecoder;

impl Decoder for CompletionOnlyDecoder {
	fn push(&mut self, _frame: Frame, emit: &mut dyn FnMut(RawEvent)) -> Result<(), Error> {
		emit(RawEvent::Completion(RawCompletion {
			reason: FinishReason::Stop,
			blocks: 0,
			usage:  Usage::default(),
		}));
		Ok(())
	}

	fn finish(&mut self, _emit: &mut dyn FnMut(RawEvent)) -> Result<(), Error> {
		Ok(())
	}
}

struct FailDecoder;

impl Decoder for FailDecoder {
	fn push(&mut self, _frame: Frame, _emit: &mut dyn FnMut(RawEvent)) -> Result<(), Error> {
		let mut error = Error::new(
			ErrorKind::Protocol,
			ErrorPhase::Handshake,
			RetryAction::Never,
			ExecutionReceipt::default(),
		);
		error.detail =
			Some(ErrorDetail::Protocol { reason: ReasonId(Str::from("fixture-first-frame")) });
		Err(error)
	}

	fn finish(&mut self, _emit: &mut dyn FnMut(RawEvent)) -> Result<(), Error> {
		Ok(())
	}
}

fn request(
	body: BodySource,
	decoder: impl Decoder + 'static,
	cancel: Cancellation,
) -> TransportRequest {
	TransportRequest {
		encoded: EncodedRequest {
			operation: OperationKind::Chat,
			method: RequestMethod::Post,
			uri: Str::from("https://provider.invalid/v1/stream"),
			headers: Box::new([]),
			body,
			framing: FramingProtocol::Raw,
			bounds: SizeBounds { request_body: 1024, frame: 1024, response: 1024 },
			sealed_body: None,
		},
		credentials: None,
		decoder: Some(Box::new(decoder)),
		realtime: None,
		cancel,
		attempt: TransportAttempt {
			request_id:    RequestId::new("request"),
			provider:      ProviderId::new("provider"),
			route:         RouteId::new("route"),
			account:       None,
			principal:     None,
			index:         0,
			provisional:   false,
			timeout:       std::time::Duration::from_secs(5),
			capture_limit: 10,
		},
	}
}

fn attempt(
	body: CassetteBodyAction,
	terminal: CassetteTerminal,
	frame_count: usize,
) -> CassetteAttempt {
	CassetteAttempt {
		status: Some(200),
		headers: Box::new([]),
		provider_request_id: Some(Str::from("provider-request")),
		body,
		frames: (0..frame_count)
			.map(|_| Frame::Raw(Bytes::from_static(b"secret-frame")))
			.collect::<Vec<_>>()
			.into_boxed_slice(),
		terminal,
	}
}

#[tokio::test]
async fn first_frame_error_remains_precommit_with_exact_body_evidence() {
	let mut service = CassetteTransport::new(Arc::from([attempt(
		CassetteBodyAction::PollChunks(1),
		CassetteTerminal::Complete,
		1,
	)]));
	service.ready().await.expect("cassette ready");
	let error = service
		.call(request(
			BodySource::Bytes(Bytes::from_static(b"request")),
			FailDecoder,
			Cancellation::default(),
		))
		.await
		.err()
		.expect("first frame fails handshake");
	assert!(!error.committed);
	assert_eq!(error.phase, ErrorPhase::Handshake);
	let receipt = error.receipt.attempts.last().expect("attempt receipt");
	assert_eq!(receipt.outcome, AttemptOutcome::FailedPreCommit);
	assert!(receipt.body.opened && receipt.body.consumed);
}

#[tokio::test]
async fn disconnect_after_first_event_is_a_committed_partial_error() {
	let mut service = CassetteTransport::new(Arc::from([attempt(
		CassetteBodyAction::Drain,
		CassetteTerminal::Disconnect,
		1,
	)]));
	service.ready().await.expect("cassette ready");
	let response = service
		.call(request(
			BodySource::Bytes(Bytes::from_static(b"request")),
			EmitDecoder,
			Cancellation::default(),
		))
		.await
		.expect("handshake");
	let mut events = response.events.expect("ordinary event stream");
	assert!(events.next().await.expect("first event").is_ok());
	let error = match events.next().await.expect("partial error") {
		Err(error) => error,
		Ok(_) => panic!("disconnect must surface as an error"),
	};
	assert!(error.committed);
	assert_eq!(
		error
			.receipt
			.attempts
			.last()
			.expect("attempt receipt")
			.outcome,
		AttemptOutcome::FailedCommitted
	);
	assert!(events.next().await.is_none(), "committed failure terminates the response stream");
}

#[tokio::test]
async fn readiness_cancellation_and_capture_are_deterministic() {
	let cancel = Cancellation::default();
	let mut service = CassetteTransport::new(Arc::from([attempt(
		CassetteBodyAction::Opened,
		CassetteTerminal::Complete,
		2,
	)]))
	.with_pending_ready_polls(1);
	service
		.ready()
		.await
		.expect("readiness forwards pending then ready");
	let response = service
		.call(request(BodySource::Bytes(Bytes::from_static(b"request")), EmitDecoder, cancel.clone()))
		.await
		.expect("handshake");
	drop(response.events.expect("ordinary event stream"));
	assert!(cancel.is_cancelled());
	let captures = service.captures();
	assert_eq!(captures.len(), 1);
	assert!(captures[0].body.opened && !captures[0].body.consumed);
	assert_eq!(captures[0].frames[0].redaction, Bytes::from_static(b"<redacted>"));
	assert_eq!(captures, service.captures());
}

#[tokio::test]
async fn replayable_factory_opens_a_fresh_body_for_every_attempt() {
	let opens = Arc::new(AtomicUsize::new(0));
	let factory_opens = Arc::clone(&opens);
	let factory = BodyFactoryHandle::new(move || {
		factory_opens.fetch_add(1, Ordering::SeqCst);
		let body: ByteStream =
			Box::pin(stream::iter([Ok(Bytes::from_static(b"a")), Ok(Bytes::from_static(b"b"))]));
		std::future::ready(Ok(body))
	});
	let source = BodySource::Factory(factory);
	let attempts: Arc<[CassetteAttempt]> = Arc::from([
		attempt(CassetteBodyAction::PollChunks(1), CassetteTerminal::Complete, 1),
		attempt(CassetteBodyAction::Drain, CassetteTerminal::Complete, 1),
	]);
	let mut service = CassetteTransport::new(attempts);
	for index in 0..2 {
		service.ready().await.expect("cassette ready");
		let mut request = request(source.clone(), EmitDecoder, Cancellation::default());
		request.attempt.index = index;
		let response = service.call(request).await.expect("handshake");
		drop(response.events.expect("ordinary event stream"));
	}
	assert_eq!(opens.load(Ordering::SeqCst), 2);
}

#[test]
fn lifecycle_and_retry_fixtures_bind_the_service_contract() {
	let lifecycle: LifecycleFixture = serde_json::from_str(include_str!(
		"../../../../fixtures/llm-oracle/transport/lifecycle.json"
	))
	.expect("typed lifecycle fixture");
	assert_eq!(lifecycle.commit_rule, "commit-after-first-decodable-meaningful-frame");
	let first = lifecycle
		.cases
		.iter()
		.find(|case| case.id == "lifecycle.first-frame-error.v1")
		.and_then(|case| case.expected.as_ref())
		.expect("first-frame case");
	assert_eq!(first.commit_state.as_deref(), Some("uncommitted"));
	assert_eq!(first.retry_action.as_deref(), Some("retry-exact-request"));
	let partial = lifecycle
		.cases
		.iter()
		.find(|case| case.id == "lifecycle.post-commit-disconnect.v1")
		.and_then(|case| case.expected.as_ref())
		.expect("partial case");
	assert_eq!(partial.commit_state.as_deref(), Some("committed"));
	assert_eq!(partial.retry_action.as_deref(), Some("surface-partial-stream-error"));

	let retry: RetryFixture =
		serde_json::from_str(include_str!("../../../../fixtures/llm-oracle/transport/retry.json"))
			.expect("typed retry fixture");
	assert_eq!(retry.policy, "retry-only-before-commit-and-only-when-exact-replay-is-allowed");
	let replay = retry
		.cases
		.iter()
		.find(|case| case.id == "retry.replayable-503-then-success.v1")
		.and_then(|case| case.expected.as_ref())
		.expect("replay case");
	assert_eq!(replay.attempt_count, Some(2));
	let committed = retry
		.cases
		.iter()
		.find(|case| case.id == "retry.post-commit-error.v1")
		.and_then(|case| case.expected.as_ref())
		.expect("committed retry case");
	assert_eq!(committed.retry_action.as_deref(), Some("do-not-retry"));
}

#[tokio::test]
async fn factory_error_is_preserved_and_captured_with_exact_evidence() {
	let expected = {
		let mut error = Error::new(
			ErrorKind::InvalidRequest,
			ErrorPhase::Encoding,
			RetryAction::Never,
			ExecutionReceipt::default(),
		);
		error.detail =
			Some(ErrorDetail::Protocol { reason: ReasonId(Str::from("factory-terminal")) });
		error
	};
	let factory = BodyFactoryHandle::new(move || {
		let error = expected.clone();
		async move { Err::<ByteStream, Error>(error) }
	});
	let mut service = CassetteTransport::new(Arc::from([attempt(
		CassetteBodyAction::Opened,
		CassetteTerminal::Complete,
		1,
	)]));
	service.ready().await.expect("cassette ready");
	let error = service
		.call(request(BodySource::Factory(factory), EmitDecoder, Cancellation::default()))
		.await
		.err()
		.expect("factory failure");
	assert_eq!(error.kind, ErrorKind::InvalidRequest);
	assert_eq!(error.action, RetryAction::Never);
	assert_eq!(error.phase, ErrorPhase::Encoding);
	assert!(matches!(error.detail, Some(ErrorDetail::Protocol { .. })));
	assert_eq!(service.captures().len(), 1);
	assert!(!service.captures()[0].body.opened);
}

#[tokio::test]
async fn retryable_factory_error_keeps_its_retry_action() {
	let factory = BodyFactoryHandle::new(move || async move {
		Err::<ByteStream, Error>(Error::new(
			ErrorKind::Connectivity,
			ErrorPhase::Connecting,
			RetryAction::SameRoute { after: std::time::Duration::from_millis(7) },
			ExecutionReceipt::default(),
		))
	});
	let mut service = CassetteTransport::new(Arc::from([attempt(
		CassetteBodyAction::Opened,
		CassetteTerminal::Complete,
		1,
	)]));
	service.ready().await.expect("cassette ready");
	let error = service
		.call(request(BodySource::Factory(factory), EmitDecoder, Cancellation::default()))
		.await
		.err()
		.expect("factory failure");
	assert_eq!(error.kind, ErrorKind::Connectivity);
	assert_eq!(error.action, RetryAction::SameRoute { after: std::time::Duration::from_millis(7) });
	assert_eq!(service.captures().len(), 1);
}

#[tokio::test]
async fn every_precommit_terminal_path_appends_one_capture() {
	let cases = [
		(attempt(CassetteBodyAction::Unopened, CassetteTerminal::Complete, 0), false),
		(attempt(CassetteBodyAction::Unopened, CassetteTerminal::Disconnect, 0), false),
		(attempt(CassetteBodyAction::Unopened, CassetteTerminal::Complete, 1), true),
	];
	for (script, decoder_fails) in cases {
		let mut service = CassetteTransport::new(Arc::from([script]));
		service.ready().await.expect("cassette ready");
		let result = if decoder_fails {
			service
				.call(request(BodySource::Bytes(Bytes::new()), FailDecoder, Cancellation::default()))
				.await
		} else {
			service
				.call(request(BodySource::Bytes(Bytes::new()), EmitDecoder, Cancellation::default()))
				.await
		};
		assert!(result.is_err());
		assert_eq!(service.captures().len(), 1);
	}

	let cancel = Cancellation::default();
	cancel.cancel();
	let mut service = CassetteTransport::new(Arc::from([attempt(
		CassetteBodyAction::Unopened,
		CassetteTerminal::Complete,
		1,
	)]));
	service.ready().await.expect("cassette ready");
	assert!(
		service
			.call(request(BodySource::Bytes(Bytes::new()), EmitDecoder, cancel))
			.await
			.is_err()
	);
	assert_eq!(service.captures().len(), 1);
}

#[tokio::test]
async fn metadata_then_disconnect_remains_precommit_retryable() {
	let mut service = CassetteTransport::new(Arc::from([attempt(
		CassetteBodyAction::Drain,
		CassetteTerminal::Disconnect,
		1,
	)]));
	service.ready().await.expect("cassette ready");
	let error = service
		.call(request(
			BodySource::Bytes(Bytes::from_static(b"request")),
			MetadataOnlyDecoder,
			Cancellation::default(),
		))
		.await
		.err()
		.expect("metadata does not commit");
	assert!(!error.committed);
	assert!(matches!(error.action, RetryAction::SameRoute { .. }));
}

#[tokio::test]
async fn provider_state_then_expiry_preserves_reseed_action() {
	let mut service = CassetteTransport::new(Arc::from([attempt(
		CassetteBodyAction::Unopened,
		CassetteTerminal::Complete,
		1,
	)]));
	service.ready().await.expect("cassette ready");
	let error = service
		.call(request(
			BodySource::Bytes(Bytes::new()),
			StateThenExpiredDecoder,
			Cancellation::default(),
		))
		.await
		.err()
		.expect("state is private preamble");
	assert!(!error.committed);
	assert_eq!(error.kind, ErrorKind::SessionExpired);
	assert_eq!(error.action, RetryAction::ReseedSession);
}

#[tokio::test]
async fn metadata_before_visible_event_is_returned_in_order() {
	let mut service = CassetteTransport::new(Arc::from([attempt(
		CassetteBodyAction::Unopened,
		CassetteTerminal::Complete,
		1,
	)]));
	service.ready().await.expect("cassette ready");
	let response = service
		.call(request(
			BodySource::Bytes(Bytes::new()),
			PreambleThenVisibleDecoder,
			Cancellation::default(),
		))
		.await
		.expect("visible commit candidate");
	let events: Vec<_> = response
		.events
		.expect("ordinary event stream")
		.collect()
		.await;
	assert!(matches!(events.first(), Some(Ok(RawEvent::Metadata(_)))));
	assert!(matches!(events.get(1), Some(Ok(RawEvent::Chat(_)))));
}

#[tokio::test]
async fn consumed_one_shot_preamble_failure_suppresses_retry_evidence() {
	let one_shot = Arc::new(OneShotBody::new(Box::pin(stream::once(std::future::ready(Ok(
		Bytes::from_static(b"live"),
	))))));
	let mut service = CassetteTransport::new(Arc::from([attempt(
		CassetteBodyAction::PollChunks(1),
		CassetteTerminal::Disconnect,
		1,
	)]));
	service.ready().await.expect("cassette ready");
	let error = service
		.call(request(BodySource::OneShot(one_shot), MetadataOnlyDecoder, Cancellation::default()))
		.await
		.err()
		.expect("metadata then disconnect");
	let body = error.receipt.attempts.last().expect("attempt receipt").body;
	assert_eq!(body.retry_decision, RetryDecision::Suppress);
	assert_eq!(body.reason, RetryDecisionReason::ConsumedOneShot);
}

#[tokio::test]
async fn completion_only_response_completes_transport_handshake() {
	let mut service = CassetteTransport::new(Arc::from([attempt(
		CassetteBodyAction::Unopened,
		CassetteTerminal::Complete,
		1,
	)]));
	service.ready().await.expect("cassette ready");
	let response = service
		.call(request(
			BodySource::Bytes(Bytes::new()),
			CompletionOnlyDecoder,
			Cancellation::default(),
		))
		.await
		.expect("completion is a terminal success candidate");
	let events: Vec<_> = response
		.events
		.expect("ordinary event stream")
		.collect()
		.await;
	assert!(matches!(events.first(), Some(Ok(RawEvent::Completion(_)))));
}

#[tokio::test]
async fn private_preamble_stall_times_out_precommit() {
	let mut service = CassetteTransport::new(Arc::from([attempt(
		CassetteBodyAction::Unopened,
		CassetteTerminal::Stall,
		1,
	)]));
	service.ready().await.expect("cassette ready");
	let mut call =
		request(BodySource::Bytes(Bytes::new()), MetadataOnlyDecoder, Cancellation::default());
	call.attempt.timeout = std::time::Duration::from_millis(5);
	let error = service
		.call(call)
		.await
		.err()
		.expect("preamble stall timeout");
	assert_eq!(error.kind, ErrorKind::DeadlineExceeded);
	assert!(!error.committed);
}

#[tokio::test]
async fn postcommit_stall_times_out_as_partial() {
	let mut service = CassetteTransport::new(Arc::from([attempt(
		CassetteBodyAction::Unopened,
		CassetteTerminal::Stall,
		1,
	)]));
	service.ready().await.expect("cassette ready");
	let mut call = request(BodySource::Bytes(Bytes::new()), EmitDecoder, Cancellation::default());
	call.attempt.timeout = std::time::Duration::from_millis(5);
	let response = service.call(call).await.expect("visible event commits");
	let mut events = response.events.expect("ordinary event stream");
	assert!(matches!(events.next().await, Some(Ok(RawEvent::Chat(_)))));
	let error = match events.next().await.expect("deadline error") {
		Err(error) => error,
		Ok(_) => panic!("stalled committed stream must time out"),
	};
	assert_eq!(error.kind, ErrorKind::DeadlineExceeded);
	assert!(error.committed);
}

#[tokio::test]
async fn stalled_body_preserves_factory_replay_and_suppresses_consumed_one_shot() {
	let factory =
		BodyFactoryHandle::new(|| async { Ok::<ByteStream, Error>(Box::pin(stream::pending())) });
	let mut replayable = CassetteTransport::new(Arc::from([attempt(
		CassetteBodyAction::PollChunks(1),
		CassetteTerminal::Complete,
		1,
	)]));
	replayable.ready().await.expect("cassette ready");
	let mut call = request(BodySource::Factory(factory), EmitDecoder, Cancellation::default());
	call.attempt.timeout = std::time::Duration::from_millis(5);
	let error = replayable.call(call).await.err().expect("body timeout");
	let body = error.receipt.attempts.last().expect("attempt receipt").body;
	assert_eq!(body.retry_decision, RetryDecision::Allow);

	let one_shot = Arc::new(OneShotBody::new(Box::pin(stream::pending())));
	let mut consumed = CassetteTransport::new(Arc::from([attempt(
		CassetteBodyAction::PollChunks(1),
		CassetteTerminal::Complete,
		1,
	)]));
	consumed.ready().await.expect("cassette ready");
	let mut call = request(BodySource::OneShot(one_shot), EmitDecoder, Cancellation::default());
	call.attempt.timeout = std::time::Duration::from_millis(5);
	let error = consumed.call(call).await.err().expect("body timeout");
	let body = error.receipt.attempts.last().expect("attempt receipt").body;
	assert_eq!(body.retry_decision, RetryDecision::Suppress);
	assert_eq!(body.reason, RetryDecisionReason::ConsumedOneShot);
}

struct RealtimeEchoCodec;

impl RealtimeWireCodec for RealtimeEchoCodec {
	fn initial_frames(&mut self) -> Result<RealtimeWireFrames, Error> {
		let mut frames = RealtimeWireFrames::new();
		frames.push(Bytes::from_static(b"session"));
		Ok(frames)
	}

	fn encode(&mut self, _input: RealtimeInput) -> Result<RealtimeWireFrames, Error> {
		Ok(RealtimeWireFrames::new())
	}

	fn decode(&mut self, _payload: Bytes) -> Result<RealtimeEvents, Error> {
		let mut events = RealtimeEvents::new();
		events.push(RealtimeEvent::InputCommitted);
		Ok(events)
	}
}

#[tokio::test]
async fn cassette_transfers_owned_realtime_session_only_after_first_frame() {
	let mut service = CassetteTransport::new(Arc::from([attempt(
		CassetteBodyAction::Unopened,
		CassetteTerminal::Complete,
		1,
	)]));
	service.ready().await.expect("cassette ready");
	let mut call = request(BodySource::Bytes(Bytes::new()), EmitDecoder, Cancellation::default());
	call.encoded.operation = OperationKind::Realtime;
	call.encoded.framing = FramingProtocol::WebSocket;
	call.decoder = None;
	call.realtime = Some(Box::new(RealtimeEchoCodec));
	let response = service.call(call).await.expect("realtime handshake");
	assert!(response.events.is_none());
	assert!(response.realtime.is_some());
}
#[tokio::test]
async fn openai_realtime_cassette_preserves_normal_response_through_done() {
	let provider_frames = [
		br#"{"type":"session.created"}"#.as_slice(),
		br#"{"type":"session.updated"}"#,
		br#"{"type":"input_audio_buffer.committed"}"#,
		br#"{"type":"response.created"}"#,
		br#"{"type":"response.output_item.added","output_index":0,"item":{"type":"message"}}"#,
		br#"{"type":"response.content_part.added","item_id":"item_1","output_index":0,"part":{"type":"text"}}"#,
		br#"{"type":"response.text.delta","item_id":"item_1","output_index":0,"delta":"hi"}"#,
		br#"{"type":"response.text.done"}"#,
		br#"{"type":"response.content_part.done"}"#,
		br#"{"type":"response.output_item.done"}"#,
		br#"{"type":"rate_limits.updated","rate_limits":[]}"#,
		br#"{"type":"response.done"}"#,
	];
	let scripted = CassetteAttempt {
		status:              Some(101),
		headers:             Box::new([]),
		provider_request_id: Some(Str::from("realtime-request")),
		body:                CassetteBodyAction::Unopened,
		frames:              provider_frames
			.into_iter()
			.map(|payload| Frame::Raw(Bytes::copy_from_slice(payload)))
			.collect::<Vec<_>>()
			.into_boxed_slice(),
		terminal:            CassetteTerminal::Complete,
	};
	let codec = OpenAiRealtimeWireCodec::new(RealtimeRequest {
		instructions:   None,
		modalities:     Arc::from([RealtimeModality::Text]),
		voice:          None,
		input_audio:    Setting::Unset,
		output_audio:   Setting::Unset,
		turn_detection: Setting::Unset,
		tools:          Arc::from([]),
		negotiation:    crate::call::NegotiationPolicy::default(),
	});
	let mut service = CassetteTransport::new(Arc::from([scripted]));
	service.ready().await.expect("cassette ready");
	let mut call = request(BodySource::Bytes(Bytes::new()), EmitDecoder, Cancellation::default());
	call.encoded.operation = OperationKind::Realtime;
	call.encoded.framing = FramingProtocol::WebSocket;
	call.decoder = None;
	call.realtime = Some(Box::new(codec));
	let response = service.call(call).await.expect("full realtime handshake");
	let session = response.realtime.expect("owned realtime session");
	let mut events = Vec::new();
	for _ in 0..5 {
		events.push(
			session
				.inbound
				.recv_async()
				.await
				.expect("realtime event")
				.expect("successful realtime event"),
		);
	}
	assert!(matches!(events[0], RealtimeEvent::Ready));
	assert!(matches!(events[1], RealtimeEvent::InputCommitted));
	assert!(matches!(
		events[2],
		RealtimeEvent::Chat(ChatEvent::BlockStarted { index: 0, kind: BlockKind::Text })
	));
	assert!(
		matches!(&events[3], RealtimeEvent::Chat(ChatEvent::TextDelta { index: 0, text }) if text.as_str() == "hi")
	);
	assert!(matches!(events[4], RealtimeEvent::Chat(ChatEvent::Completed(_))));
}

#[tokio::test]
async fn websocket_upgrade_sends_initial_frame_before_first_decodable_event() {
	let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
		.await
		.expect("bind websocket fixture");
	let address = listener.local_addr().expect("websocket fixture address");
	let server = tokio::spawn(async move {
		let (socket, _) = listener.accept().await.expect("accept websocket fixture");
		let mut socket = tokio_tungstenite::accept_async(socket)
			.await
			.expect("upgrade websocket fixture");
		let initial = socket
			.next()
			.await
			.expect("initial websocket frame")
			.expect("valid websocket frame");
		assert_eq!(initial.into_data(), Bytes::from_static(b"session"));
		use futures::SinkExt as _;
		socket
			.send(tokio_tungstenite::tungstenite::Message::text("provider-ready"))
			.await
			.expect("send provider frame");
	});
	let mut service = WebSocketTransport::new();
	service.ready().await.expect("websocket ready");
	let mut call = request(BodySource::Bytes(Bytes::new()), EmitDecoder, Cancellation::default());
	call.encoded.operation = OperationKind::Realtime;
	call.encoded.framing = FramingProtocol::WebSocket;
	call.encoded.uri = Str::from(format!("ws://{address}/realtime"));
	call.decoder = None;
	call.realtime = Some(Box::new(RealtimeEchoCodec));
	let response = service.call(call).await.expect("websocket handshake");
	let session = response.realtime.expect("owned realtime session");
	assert!(matches!(session.inbound.recv_async().await, Ok(Ok(RealtimeEvent::Ready))));
	assert!(matches!(session.inbound.recv_async().await, Ok(Ok(RealtimeEvent::InputCommitted))));
	server.await.expect("websocket fixture");
	let captures = service.captures();
	assert_eq!(captures.len(), 1);
	assert_eq!(captures[0].frames.len(), 1);
	assert_eq!(captures[0].frames[0].redaction, Bytes::from_static(b"<redacted>"));
}

#[tokio::test]
async fn stalled_http_connect_or_headers_honors_attempt_timeout() {
	let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
		.await
		.expect("bind fixture");
	let address = listener.local_addr().expect("fixture address");
	let server = tokio::spawn(async move {
		let (_socket, _) = listener.accept().await.expect("accept fixture");
		std::future::pending::<()>().await;
	});
	let mut service = HttpTransport::new();
	service.ready().await.expect("http ready");
	let mut call = request(
		BodySource::Bytes(Bytes::from_static(b"request")),
		EmitDecoder,
		Cancellation::default(),
	);
	call.encoded.uri = Str::from(format!("http://{address}/stall"));
	call.attempt.timeout = std::time::Duration::from_millis(10);
	let error = service.call(call).await.err().expect("headers timeout");
	assert_eq!(error.kind, ErrorKind::DeadlineExceeded);
	assert!(!error.committed);
	assert_eq!(error.receipt.attempts.len(), 1);
	server.abort();
}

#[tokio::test]
async fn stalled_http_headers_honor_in_flight_cancellation() {
	let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
		.await
		.expect("bind fixture");
	let address = listener.local_addr().expect("fixture address");
	let (accepted_tx, accepted_rx) = tokio::sync::oneshot::channel();
	let server = tokio::spawn(async move {
		let (_socket, _) = listener.accept().await.expect("accept fixture");
		let _ = accepted_tx.send(());
		std::future::pending::<()>().await;
	});
	let mut service = HttpTransport::new();
	service.ready().await.expect("http ready");
	let cancellation = Cancellation::default();
	let mut call =
		request(BodySource::Bytes(Bytes::from_static(b"request")), EmitDecoder, cancellation.clone());
	call.encoded.uri = Str::from(format!("http://{address}/stall"));
	call.attempt.timeout = std::time::Duration::from_secs(5);
	let response = service.call(call);
	tokio::pin!(response);
	tokio::select! {
		_ = &mut response => panic!("request ended before cancellation"),
		result = accepted_rx => result.expect("fixture accepted request"),
	}
	cancellation.cancel();
	let error = tokio::time::timeout(std::time::Duration::from_secs(1), response)
		.await
		.expect("cancellation bound")
		.err()
		.expect("cancelled request");
	assert_eq!(error.kind, ErrorKind::Cancelled);
	assert!(!error.committed);
	assert_eq!(error.receipt.attempts.len(), 1);
	server.abort();
}
