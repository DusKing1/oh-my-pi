//! Deterministic in-memory transport for lifecycle, handshake, and replay
//! tests.

use std::{
	collections::VecDeque,
	future::Future,
	pin::Pin,
	sync::{
		Arc,
		atomic::{AtomicBool, Ordering},
	},
	task::{Context, Poll},
	time::Instant,
};

use bytes::Bytes;
use futures::{Stream, StreamExt as _, future::poll_fn};
use omp_core::Str;
use parking_lot::Mutex;
use tower::Service;

use crate::{
	answer::{RealtimeEvent, RealtimeInput, RealtimeSession},
	body::{AttemptBodyEvidence, AttemptEvidenceHandle, BodyOpenError},
	catalog::OperationKind,
	codec::{
		Cancellation, HandshakeMeta, HandshakenResponse, RawEvent, RawEventStream, RequestHeader,
		TransportAttempt, TransportRequest,
	},
	error::{Error, ErrorDetail, ErrorKind, ErrorPhase, RetryAction},
	receipt::{ExecutionReceipt, ReasonId},
	transport::{Frame, http::record_failure},
};

/// Request-body behavior performed by one scripted attempt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CassetteBodyAction {
	/// Do not acquire the request body.
	Unopened,
	/// Acquire the body without polling it.
	Opened,
	/// Poll at most this many body chunks.
	PollChunks(usize),
	/// Consume the complete body stream.
	Drain,
}

/// Terminal behavior following the scripted provider frames.
#[derive(Clone, Debug)]
pub enum CassetteTerminal {
	/// Finish the codec normally.
	Complete,
	/// End the connection without a protocol-complete response.
	Disconnect,
	/// Keep the scripted connection open without another frame until the attempt
	/// timeout.
	Stall,
	/// Surface a preconstructed structured transport failure.
	Error(Error),
}

/// One deterministic provider attempt.
#[derive(Clone, Debug)]
pub struct CassetteAttempt {
	/// HTTP-like status exposed at handshake.
	pub status:              Option<u16>,
	/// Sanitized public response headers.
	pub headers:             Box<[RequestHeader]>,
	/// Provider request identifier.
	pub provider_request_id: Option<Str>,
	/// Request-body acquisition and polling behavior.
	pub body:                CassetteBodyAction,
	/// Already-framed provider input presented to the codec in order.
	pub frames:              Box<[Frame]>,
	/// Terminal behavior after all frames.
	pub terminal:            CassetteTerminal,
}

/// Sanitized structural frame record. Payload bytes are never retained.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapturedFrame {
	/// Zero-based frame ordinal.
	pub ordinal:        u64,
	/// Stable protocol label.
	pub protocol:       &'static str,
	/// Original payload length.
	pub observed_bytes: u64,
	/// Fixed redaction token, truncated to the configured capture budget.
	pub redaction:      Bytes,
}

/// Deterministic evidence retained for one cassette attempt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CassetteCapture {
	/// Zero-based scripted attempt index.
	pub attempt: usize,
	/// Request URI; credential middleware must never place secrets here.
	pub uri:     Str,
	/// Exact request-body lifecycle evidence.
	pub body:    AttemptBodyEvidence,
	/// Bounded, payload-free provider frame records.
	pub frames:  Box<[CapturedFrame]>,
}

#[derive(Clone, Default)]
struct CaptureLog(Arc<Mutex<Vec<CassetteCapture>>>);

/// Deterministic Tower service whose frames are decoded by each request's real
/// decoder.
///
/// Every clone receives an independent script cursor. Captures are shared so a
/// test can inspect all calls without moving the service used for `poll_ready`
/// and `call`.
#[derive(Clone)]
pub struct CassetteTransport {
	attempts:            Arc<[CassetteAttempt]>,
	cursor:              usize,
	pending_ready_polls: usize,
	ready_permit:        bool,
	captures:            CaptureLog,
}

struct CassetteCaptureFinalizer {
	log:      CaptureLog,
	attempt:  usize,
	uri:      Str,
	evidence: AttemptEvidenceHandle,
	frames:   Vec<CapturedFrame>,
}

impl Drop for CassetteCaptureFinalizer {
	fn drop(&mut self) {
		self.log.0.lock().push(CassetteCapture {
			attempt: self.attempt,
			uri:     self.uri.clone(),
			body:    self.evidence.evidence(),
			frames:  std::mem::take(&mut self.frames).into_boxed_slice(),
		});
	}
}

impl CassetteTransport {
	/// Creates a cassette with no artificial readiness delay.
	#[must_use]
	pub fn new(attempts: impl Into<Arc<[CassetteAttempt]>>) -> Self {
		Self {
			attempts:            attempts.into(),
			cursor:              0,
			pending_ready_polls: 0,
			ready_permit:        false,
			captures:            CaptureLog::default(),
		}
	}

	/// Makes the next readiness cycle return `Pending` this many times.
	#[must_use]
	pub fn with_pending_ready_polls(mut self, polls: usize) -> Self {
		self.pending_ready_polls = polls;
		self
	}

	/// Returns a stable snapshot of sanitized captures.
	#[must_use]
	pub fn captures(&self) -> Vec<CassetteCapture> {
		let mut captures = self.captures.0.lock().clone();
		captures.sort_by_key(|capture| capture.attempt);
		captures
	}
}

impl Service<TransportRequest> for CassetteTransport {
	type Error = Error;
	type Response = HandshakenResponse;

	type Future = impl Future<Output = Result<Self::Response, Self::Error>> + Send + 'static;

	fn poll_ready(&mut self, context: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		if self.pending_ready_polls > 0 {
			self.pending_ready_polls -= 1;
			context.waker().wake_by_ref();
			return Poll::Pending;
		}
		if self.cursor >= self.attempts.len() {
			return Poll::Ready(Err(transport_error(
				ErrorPhase::Readiness,
				false,
				"cassette-exhausted",
			)));
		}
		self.ready_permit = true;
		Poll::Ready(Ok(()))
	}

	fn call(&mut self, request: TransportRequest) -> Self::Future {
		let permit = std::mem::take(&mut self.ready_permit);
		let index = self.cursor;
		if permit {
			self.cursor += 1;
		}
		let attempt = self.attempts.get(index).cloned();
		let captures = self.captures.clone();
		async move {
			if !permit {
				return Err(transport_error(ErrorPhase::Readiness, false, "call-without-readiness"));
			}
			let attempt = attempt
				.ok_or_else(|| transport_error(ErrorPhase::Readiness, false, "cassette-exhausted"))?;
			run_attempt(index, attempt, request, captures).await
		}
	}
}

async fn run_attempt(
	index: usize,
	attempt: CassetteAttempt,
	mut request: TransportRequest,
	captures: CaptureLog,
) -> Result<HandshakenResponse, Error> {
	match (request.decoder.is_some(), request.realtime.is_some()) {
		(true, false) => {},
		(false, true) => return run_realtime_attempt(index, attempt, request, captures).await,
		_ => {
			let started = Instant::now();
			let body_attempt = request.encoded.body.begin_attempt();
			let evidence = body_attempt.evidence_handle();
			return Err(record_failure(
				transport_error(ErrorPhase::Handshake, false, "transport-decoder-cardinality"),
				&request.attempt,
				&evidence,
				attempt.status,
				attempt.provider_request_id.as_ref(),
				started,
				false,
			));
		},
	}
	let started = Instant::now();
	let transport_attempt = request.attempt.clone();
	let mut body_attempt = request.encoded.body.begin_attempt();
	let evidence = body_attempt.evidence_handle();
	let deadline = tokio::time::Instant::now() + transport_attempt.timeout;
	let mut capture = CassetteCaptureFinalizer {
		log:      captures,
		attempt:  index,
		uri:      request.encoded.uri.clone(),
		evidence: evidence.clone(),
		frames:   Vec::new(),
	};
	consume_body(&mut body_attempt, attempt.body, &request.cancel, deadline)
		.await
		.map_err(|error| {
			record_failure(
				error,
				&transport_attempt,
				&evidence,
				attempt.status,
				attempt.provider_request_id.as_ref(),
				started,
				false,
			)
		})?;
	let mut decoder = request.decoder.take().ok_or_else(|| {
		record_failure(
			transport_error(ErrorPhase::Handshake, false, "ordinary-decoder-missing"),
			&transport_attempt,
			&evidence,
			attempt.status,
			attempt.provider_request_id.as_ref(),
			started,
			false,
		)
	})?;

	let mut output = VecDeque::new();
	let mut capture_remaining = request.attempt.capture_limit;
	for (ordinal, frame) in attempt.frames.into_vec().into_iter().enumerate() {
		if request.cancel.is_cancelled() {
			return Err(record_failure(
				cancelled(false),
				&transport_attempt,
				&evidence,
				attempt.status,
				attempt.provider_request_id.as_ref(),
				started,
				false,
			));
		}
		capture_frame(&mut capture.frames, ordinal as u64, &frame, &mut capture_remaining);
		let mut emitted = |event| output.push_back(event);
		if let Err(error) = decoder.push(frame, &mut emitted) {
			return Err(record_failure(
				precommit(error),
				&transport_attempt,
				&evidence,
				attempt.status,
				attempt.provider_request_id.as_ref(),
				started,
				false,
			));
		}
		if let Some(position) = output
			.iter()
			.position(|event| matches!(event, RawEvent::Failure(_)))
			&& !output.iter().take(position).any(is_commit_candidate)
		{
			let Some(RawEvent::Failure(error)) = output.remove(position) else {
				unreachable!()
			};
			return Err(record_failure(
				precommit(error),
				&transport_attempt,
				&evidence,
				attempt.status,
				attempt.provider_request_id.as_ref(),
				started,
				false,
			));
		}
	}

	let mut stall_after_commit = false;
	match attempt.terminal {
		CassetteTerminal::Complete => {
			let mut emitted = |event| output.push_back(event);
			if let Err(error) = decoder.finish(&mut emitted) {
				if output.iter().any(is_commit_candidate) {
					output.push_back(RawEvent::Failure(committed(error)));
				} else {
					return Err(record_failure(
						precommit(error),
						&transport_attempt,
						&evidence,
						attempt.status,
						attempt.provider_request_id.as_ref(),
						started,
						false,
					));
				}
			}
		},
		CassetteTerminal::Disconnect if !output.iter().any(is_commit_candidate) => {
			return Err(record_failure(
				transport_error(ErrorPhase::Handshake, false, "disconnect-before-commit-event"),
				&transport_attempt,
				&evidence,
				attempt.status,
				attempt.provider_request_id.as_ref(),
				started,
				false,
			));
		},
		CassetteTerminal::Disconnect => output.push_back(RawEvent::Failure(transport_error(
			ErrorPhase::Streaming,
			true,
			"disconnect-after-partial-output",
		))),
		CassetteTerminal::Error(error) if !output.iter().any(is_commit_candidate) => {
			return Err(record_failure(
				precommit(error),
				&transport_attempt,
				&evidence,
				attempt.status,
				attempt.provider_request_id.as_ref(),
				started,
				false,
			));
		},
		CassetteTerminal::Stall if !output.iter().any(is_commit_candidate) => {
			tokio::select! {
				() = tokio::time::sleep_until(deadline) => {
					request.cancel.cancel();
					return Err(record_failure(deadline_exceeded(false), &transport_attempt, &evidence, attempt.status, attempt.provider_request_id.as_ref(), started, false));
				},
				() = poll_fn(|context| request.cancel.poll_cancelled(context)) => {
					return Err(record_failure(cancelled(false), &transport_attempt, &evidence, attempt.status, attempt.provider_request_id.as_ref(), started, false));
				},
			}
		},
		CassetteTerminal::Stall => stall_after_commit = true,
		CassetteTerminal::Error(error) => output.push_back(RawEvent::Failure(committed(error))),
	}
	if let Some(position) = output
		.iter()
		.position(|event| matches!(event, RawEvent::Failure(_)))
		&& !output.iter().take(position).any(is_commit_candidate)
	{
		let Some(RawEvent::Failure(error)) = output.remove(position) else {
			unreachable!()
		};
		return Err(record_failure(
			precommit(error),
			&transport_attempt,
			&evidence,
			attempt.status,
			attempt.provider_request_id.as_ref(),
			started,
			false,
		));
	}
	if !output.iter().any(is_commit_candidate) {
		return Err(record_failure(
			transport_error(ErrorPhase::Handshake, false, "no-committing-provider-event"),
			&transport_attempt,
			&evidence,
			attempt.status,
			attempt.provider_request_id.as_ref(),
			started,
			false,
		));
	}
	let stream: RawEventStream = Box::pin(CassetteEventStream::new(
		output,
		request.cancel,
		transport_attempt,
		evidence.clone(),
		attempt.status,
		attempt.provider_request_id.clone(),
		stall_after_commit,
		deadline,
		started,
	));
	Ok(HandshakenResponse {
		meta:     HandshakeMeta {
			status:              attempt.status,
			headers:             attempt.headers,
			provider_request_id: attempt.provider_request_id,
		},
		body:     evidence,
		events:   Some(stream),
		realtime: None,
	})
}

async fn run_realtime_attempt(
	index: usize,
	attempt: CassetteAttempt,
	mut request: TransportRequest,
	captures: CaptureLog,
) -> Result<HandshakenResponse, Error> {
	let started = Instant::now();
	let transport_attempt = request.attempt.clone();
	let deadline = tokio::time::Instant::now() + transport_attempt.timeout;
	let mut body_attempt = request.encoded.body.begin_attempt();
	let evidence = body_attempt.evidence_handle();
	let mut capture = CassetteCaptureFinalizer {
		log:      captures,
		attempt:  index,
		uri:      request.encoded.uri.clone(),
		evidence: evidence.clone(),
		frames:   Vec::new(),
	};
	consume_body(&mut body_attempt, attempt.body, &request.cancel, deadline)
		.await
		.map_err(|error| {
			record_failure(
				error,
				&transport_attempt,
				&evidence,
				attempt.status,
				attempt.provider_request_id.as_ref(),
				started,
				false,
			)
		})?;
	if request.encoded.operation != OperationKind::Realtime {
		return Err(record_failure(
			transport_error(ErrorPhase::Handshake, false, "realtime-codec-on-non-realtime-operation"),
			&transport_attempt,
			&evidence,
			attempt.status,
			attempt.provider_request_id.as_ref(),
			started,
			false,
		));
	}
	let mut codec = request.realtime.take().ok_or_else(|| {
		record_failure(
			transport_error(ErrorPhase::Handshake, false, "realtime-codec-missing"),
			&transport_attempt,
			&evidence,
			attempt.status,
			attempt.provider_request_id.as_ref(),
			started,
			false,
		)
	})?;
	let initial_frames = codec.initial_frames().map_err(|error| {
		record_failure(
			precommit(error),
			&transport_attempt,
			&evidence,
			attempt.status,
			attempt.provider_request_id.as_ref(),
			started,
			false,
		)
	})?;
	if initial_frames
		.iter()
		.any(|frame| frame.len() as u64 > request.encoded.bounds.frame)
	{
		return Err(record_failure(
			transport_error(ErrorPhase::Handshake, false, "realtime-initial-frame-limit"),
			&transport_attempt,
			&evidence,
			attempt.status,
			attempt.provider_request_id.as_ref(),
			started,
			false,
		));
	}
	let mut capture_remaining = request.attempt.capture_limit;
	let mut initial = Vec::new();
	let mut decoded_frame = false;
	for (ordinal, frame) in attempt.frames.into_vec().into_iter().enumerate() {
		capture_frame(&mut capture.frames, ordinal as u64, &frame, &mut capture_remaining);
		if let Some(payload) = realtime_payload(frame).map_err(|error| {
			record_failure(
				error,
				&transport_attempt,
				&evidence,
				attempt.status,
				attempt.provider_request_id.as_ref(),
				started,
				false,
			)
		})? {
			let events = codec.decode(payload).map_err(|error| {
				record_failure(
					error,
					&transport_attempt,
					&evidence,
					attempt.status,
					attempt.provider_request_id.as_ref(),
					started,
					false,
				)
			})?;
			decoded_frame = true;
			initial.extend(events);
		}
	}
	if !decoded_frame {
		return Err(record_failure(
			transport_error(ErrorPhase::Handshake, false, "realtime-no-decodable-provider-frame"),
			&transport_attempt,
			&evidence,
			attempt.status,
			attempt.provider_request_id.as_ref(),
			started,
			false,
		));
	}
	match attempt.terminal {
		CassetteTerminal::Disconnect => {
			return Err(record_failure(
				transport_error(ErrorPhase::Handshake, false, "realtime-disconnect-during-handshake"),
				&transport_attempt,
				&evidence,
				attempt.status,
				attempt.provider_request_id.as_ref(),
				started,
				false,
			));
		},
		CassetteTerminal::Error(error) => {
			return Err(record_failure(
				precommit(error),
				&transport_attempt,
				&evidence,
				attempt.status,
				attempt.provider_request_id.as_ref(),
				started,
				false,
			));
		},
		CassetteTerminal::Complete | CassetteTerminal::Stall => {},
	}
	let (outbound, outbound_rx) = flume::bounded(16);
	let (inbound_tx, inbound) = flume::bounded(16);
	let closed = Arc::new(AtomicBool::new(false));
	inbound_tx
		.send_async(Ok(RealtimeEvent::Ready))
		.await
		.map_err(|_| {
			record_failure(
				cancelled(false),
				&transport_attempt,
				&evidence,
				attempt.status,
				attempt.provider_request_id.as_ref(),
				started,
				false,
			)
		})?;
	for event in initial {
		inbound_tx.send_async(Ok(event)).await.map_err(|_| {
			record_failure(
				cancelled(false),
				&transport_attempt,
				&evidence,
				attempt.status,
				attempt.provider_request_id.as_ref(),
				started,
				false,
			)
		})?;
	}
	let cancel = request.cancel.clone();
	let status = attempt.status;
	let provider_request_id = attempt.provider_request_id.clone();
	let pump_evidence = evidence.clone();
	let pump_closed = Arc::clone(&closed);
	tokio::spawn(async move {
		let _closed = RealtimeClosedGuard(pump_closed);
		loop {
			let input = tokio::select! {
				input = outbound_rx.recv_async() => match input {
					Ok(input) => input,
					Err(_) => break,
				},
				() = poll_fn(|context| cancel.poll_cancelled(context)) => break,
				() = tokio::time::sleep_until(deadline) => {
					cancel.cancel();
					let error = record_failure(deadline_exceeded(true), &transport_attempt, &pump_evidence, status, provider_request_id.as_ref(), started, true);
					let _ = inbound_tx.send_async(Err(error)).await;
					break;
				},
			};
			if matches!(input, RealtimeInput::Close) {
				let _ = inbound_tx.send_async(Ok(RealtimeEvent::Closed)).await;
				break;
			}
			let frames = match codec.encode(input) {
				Ok(frames)
					if frames
						.iter()
						.all(|frame| frame.len() as u64 <= request.encoded.bounds.frame) =>
				{
					frames
				},
				Ok(_) => {
					let error = record_failure(
						transport_error(ErrorPhase::Streaming, true, "realtime-outbound-frame-limit"),
						&transport_attempt,
						&pump_evidence,
						status,
						provider_request_id.as_ref(),
						started,
						true,
					);
					let _ = inbound_tx.send_async(Err(error)).await;
					break;
				},
				Err(error) => {
					let error = record_failure(
						committed(error),
						&transport_attempt,
						&pump_evidence,
						status,
						provider_request_id.as_ref(),
						started,
						true,
					);
					let _ = inbound_tx.send_async(Err(error)).await;
					break;
				},
			};
			for encoded in frames {
				match codec.decode(encoded) {
					Ok(events) => {
						for event in events {
							if inbound_tx.send_async(Ok(event)).await.is_err() {
								return;
							}
						}
					},
					Err(error) => {
						let error = record_failure(
							committed(error),
							&transport_attempt,
							&pump_evidence,
							status,
							provider_request_id.as_ref(),
							started,
							true,
						);
						let _ = inbound_tx.send_async(Err(error)).await;
						return;
					},
				}
			}
		}
	});
	Ok(HandshakenResponse {
		meta:     HandshakeMeta {
			status:              attempt.status,
			headers:             attempt.headers,
			provider_request_id: attempt.provider_request_id,
		},
		body:     evidence,
		events:   None,
		realtime: Some(RealtimeSession::from_channels(outbound, inbound, closed)),
	})
}
struct RealtimeClosedGuard(Arc<AtomicBool>);

impl Drop for RealtimeClosedGuard {
	fn drop(&mut self) {
		self.0.store(true, Ordering::Release);
	}
}
fn realtime_payload(frame: Frame) -> Result<Option<Bytes>, Error> {
	match frame {
		Frame::Raw(payload) | Frame::Ndjson(payload) => Ok(Some(payload)),
		Frame::WebSocket(
			crate::transport::WebSocketMessage::Text(payload)
			| crate::transport::WebSocketMessage::Binary(payload),
		) => Ok(Some(payload)),
		Frame::WebSocket(
			crate::transport::WebSocketMessage::Ping(_) | crate::transport::WebSocketMessage::Pong(_),
		) => Ok(None),
		Frame::WebSocket(crate::transport::WebSocketMessage::Close { .. }) => {
			Err(transport_error(ErrorPhase::Handshake, false, "realtime-close-before-handshake"))
		},
		Frame::Sse(_) | Frame::Connect(_) | Frame::EventStream(_) => {
			Err(transport_error(ErrorPhase::Handshake, false, "realtime-invalid-provider-frame"))
		},
	}
}

async fn consume_body(
	attempt: &mut crate::body::BodyAttempt,
	action: CassetteBodyAction,
	cancel: &Cancellation,
	deadline: tokio::time::Instant,
) -> Result<(), Error> {
	if action == CassetteBodyAction::Unopened {
		return Ok(());
	}
	let reader = tokio::select! {
		result = attempt.open() => result.map_err(|error| match error {
			BodyOpenError::Factory(error) => error,
			BodyOpenError::AttemptAlreadyOpened => transport_error(ErrorPhase::Connecting, false, "body-attempt-already-opened"),
			BodyOpenError::ConcurrentReader => transport_error(ErrorPhase::Connecting, false, "body-concurrent-reader"),
			BodyOpenError::Consumed => transport_error(ErrorPhase::Connecting, false, "body-consumed"),
			BodyOpenError::ReacquisitionUnavailable => transport_error(ErrorPhase::Connecting, false, "body-reacquisition-unavailable"),
		})?,
		() = tokio::time::sleep_until(deadline) => {
			cancel.cancel();
			return Err(deadline_exceeded(false));
		},
	};
	let mut reader = reader;
	if action == CassetteBodyAction::Opened {
		return Ok(());
	}
	let limit = match action {
		CassetteBodyAction::PollChunks(limit) => Some(limit),
		CassetteBodyAction::Drain => None,
		_ => unreachable!(),
	};
	let mut polled = 0;
	while limit.is_none_or(|limit| polled < limit) {
		let next = tokio::select! {
			next = reader.next() => next,
			() = poll_fn(|context| cancel.poll_cancelled(context)) => return Err(cancelled(false)),
			() = tokio::time::sleep_until(deadline) => {
				cancel.cancel();
				return Err(deadline_exceeded(false));
			},
		};
		match next {
			Some(Ok(_)) => polled += 1,
			Some(Err(error)) => return Err(precommit(error)),
			None => break,
		}
	}
	Ok(())
}

pub(crate) fn capture_frame(
	output: &mut Vec<CapturedFrame>,
	ordinal: u64,
	frame: &Frame,
	remaining: &mut u64,
) {
	if *remaining == 0 {
		return;
	}
	let (protocol, observed) = frame_metadata(frame);
	const REDACTED: &[u8] = b"<redacted>";
	let retained = (*remaining).min(REDACTED.len() as u64) as usize;
	output.push(CapturedFrame {
		ordinal,
		protocol,
		observed_bytes: observed as u64,
		redaction: Bytes::from_static(REDACTED).slice(..retained),
	});
	*remaining -= retained as u64;
}

fn frame_metadata(frame: &Frame) -> (&'static str, usize) {
	match frame {
		Frame::Raw(data) => ("raw", data.len()),
		Frame::Sse(event) => ("sse", event.data.len()),
		Frame::Ndjson(data) => ("ndjson", data.len()),
		Frame::WebSocket(message) => ("websocket", websocket_payload_len(message)),
		Frame::Connect(envelope) => ("connect", envelope.payload.len()),
		Frame::EventStream(message) => ("aws-eventstream", message.payload.len()),
	}
}

fn websocket_payload_len(message: &crate::transport::WebSocketMessage) -> usize {
	match message {
		crate::transport::WebSocketMessage::Text(data)
		| crate::transport::WebSocketMessage::Binary(data)
		| crate::transport::WebSocketMessage::Ping(data)
		| crate::transport::WebSocketMessage::Pong(data) => data.len(),
		crate::transport::WebSocketMessage::Close { reason, .. } => reason.len(),
	}
}

pub(crate) fn is_commit_candidate(event: &RawEvent) -> bool {
	matches!(
		event,
		RawEvent::Chat(_)
			| RawEvent::Completion(_)
			| RawEvent::Answer(_)
			| RawEvent::Control(_)
			| RawEvent::NativeChunk(_)
			| RawEvent::DiscoveredModels { .. }
	)
}

fn precommit(mut error: Error) -> Error {
	error.committed = false;
	error.phase = ErrorPhase::Handshake;
	error
}

fn committed(mut error: Error) -> Error {
	error.committed = true;
	error.phase = ErrorPhase::Streaming;
	error.action = RetryAction::Never;
	error
}

fn deadline_exceeded(committed: bool) -> Error {
	let mut error = Error::new(
		ErrorKind::DeadlineExceeded,
		if committed {
			ErrorPhase::Streaming
		} else {
			ErrorPhase::Handshake
		},
		RetryAction::Never,
		ExecutionReceipt::default(),
	);
	error.committed = committed;
	error
}

fn cancelled(committed: bool) -> Error {
	let phase = if committed {
		ErrorPhase::Streaming
	} else {
		ErrorPhase::Handshake
	};
	let mut error =
		Error::new(ErrorKind::Cancelled, phase, RetryAction::Never, ExecutionReceipt::default());
	error.committed = committed;
	error
}

fn transport_error(phase: ErrorPhase, committed: bool, reason: &'static str) -> Error {
	let action = if committed {
		RetryAction::Never
	} else {
		RetryAction::SameRoute { after: std::time::Duration::ZERO }
	};
	let mut error = Error::new(ErrorKind::Connectivity, phase, action, ExecutionReceipt::default());
	error.committed = committed;
	error.detail = Some(ErrorDetail::Protocol { reason: ReasonId(Str::from(reason)) });
	error
}

struct CassetteEventStream {
	items:               VecDeque<RawEvent>,
	cancel:              Cancellation,
	attempt:             TransportAttempt,
	evidence:            AttemptEvidenceHandle,
	status:              Option<u16>,
	provider_request_id: Option<Str>,
	stall_after_commit:  bool,
	deadline:            Pin<Box<tokio::time::Sleep>>,
	started:             Instant,
	emitted:             bool,
	finished:            bool,
}

impl CassetteEventStream {
	fn new(
		items: VecDeque<RawEvent>,
		cancel: Cancellation,
		attempt: TransportAttempt,
		evidence: AttemptEvidenceHandle,
		status: Option<u16>,
		provider_request_id: Option<Str>,
		stall_after_commit: bool,
		deadline: tokio::time::Instant,
		started: Instant,
	) -> Self {
		Self {
			items,
			cancel,
			attempt,
			evidence,
			status,
			provider_request_id,
			stall_after_commit,
			deadline: Box::pin(tokio::time::sleep_until(deadline)),
			started,
			emitted: false,
			finished: false,
		}
	}
}

impl Stream for CassetteEventStream {
	type Item = Result<RawEvent, Error>;

	fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
		if self.finished {
			return Poll::Ready(None);
		}
		if self.cancel.is_cancelled() {
			self.finished = true;
			let committed = self.emitted;
			let error = record_failure(
				cancelled(committed),
				&self.attempt,
				&self.evidence,
				self.status,
				self.provider_request_id.as_ref(),
				self.started,
				committed,
			);
			return Poll::Ready(Some(Err(error)));
		}
		match self.items.pop_front() {
			Some(RawEvent::Failure(error)) => {
				self.finished = true;
				let committed = self.emitted;
				let error = record_failure(
					error,
					&self.attempt,
					&self.evidence,
					self.status,
					self.provider_request_id.as_ref(),
					self.started,
					committed,
				);
				Poll::Ready(Some(Err(error)))
			},
			Some(event) => {
				self.emitted |= is_commit_candidate(&event);
				Poll::Ready(Some(Ok(event)))
			},
			None if self.stall_after_commit => {
				if self.deadline.as_mut().poll(context).is_pending() {
					return Poll::Pending;
				}
				self.finished = true;
				self.cancel.cancel();
				let error = record_failure(
					deadline_exceeded(true),
					&self.attempt,
					&self.evidence,
					self.status,
					self.provider_request_id.as_ref(),
					self.started,
					true,
				);
				Poll::Ready(Some(Err(error)))
			},
			None => {
				self.finished = true;
				Poll::Ready(None)
			},
		}
	}
}

impl Drop for CassetteEventStream {
	fn drop(&mut self) {
		if !self.finished && !self.items.is_empty() {
			self.cancel.cancel();
		}
	}
}
