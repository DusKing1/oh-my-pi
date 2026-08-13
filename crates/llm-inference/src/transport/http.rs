//! Production pooled HTTP/1.1 and HTTP/2 streaming transport over rustls.

use std::{
	collections::VecDeque,
	convert::Infallible,
	future::Future,
	pin::Pin,
	sync::Arc,
	task::{Context, Poll},
	time::Instant,
};

use bytes::{Bytes, BytesMut};
use futures::{Stream, StreamExt as _, future::poll_fn, stream};
use http::{HeaderMap, HeaderName, HeaderValue, Method, Request, Uri, header};
use http_body_util::{BodyExt as _, Full, StreamBody, combinators::UnsyncBoxBody};
use hyper::body::{Frame as BodyFrame, Incoming};
use hyper_rustls::{HttpsConnector, HttpsConnectorBuilder};
use hyper_util::{
	client::legacy::{Client, connect::HttpConnector},
	rt::TokioExecutor,
};
use omp_core::Str;
use parking_lot::Mutex;
use smallvec::SmallVec;
use tower::{Service, ServiceExt as _};

use crate::{
	body::{
		AttemptBodyEvidence, AttemptEvidenceHandle, BodyFactoryHandle, BodyOpenError, BodyReader,
		BodySource, byte_stream,
	},
	codec::{
		Cancellation, HandshakeMeta, HandshakenResponse, RawEvent, RawEventStream, RequestHeader,
		RequestMethod, TransportAttempt, TransportRequest,
	},
	error::{Error, ErrorDetail, ErrorKind, ErrorPhase, RetryAction},
	receipt::{
		AttemptOutcome, AttemptReceipt, Cost, ExecutionReceipt, ProviderEvidence, ReasonId, Usage,
	},
	transport::{
		ConnectDecoder, EventStreamDecoder, Frame, FramingError, FramingProtocol, NdjsonDecoder,
		RawChunkFramer, SseDecoder,
		cassette::{CapturedFrame, capture_frame, is_commit_candidate},
	},
};

const MAX_CAPTURED_HEADERS: usize = 32;
const MAX_CAPTURED_HEADER_BYTES: usize = 32;
const PUBLIC_NUMERIC_HEADERS: [&str; 13] = [
	"content-length",
	"retry-after",
	"ratelimit-limit",
	"ratelimit-remaining",
	"ratelimit-reset",
	"x-ratelimit-limit",
	"x-ratelimit-remaining",
	"x-ratelimit-reset",
	"x-ratelimit-limit-requests",
	"x-ratelimit-remaining-requests",
	"x-ratelimit-limit-tokens",
	"x-ratelimit-remaining-tokens",
	"x-ratelimit-reset-tokens",
];

const MAX_REQUEST_ID_BYTES: usize = 128;
const PUBLIC_REQUEST_ID_HEADERS: [&str; 5] =
	["x-request-id", "request-id", "x-amzn-requestid", "x-goog-request-id", "cf-ray"];

type Connector = HttpsConnector<HttpConnector>;
type RequestBody = UnsyncBoxBody<Bytes, Error>;
type PooledClient = Client<Connector, RequestBody>;

/// Sanitized bounded evidence retained from a live HTTP attempt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HttpCapture {
	/// Zero-based attempt index.
	pub attempt:             u32,
	/// HTTP response status.
	pub status:              u16,
	/// Provider request identifier, if present.
	pub provider_request_id: Option<Str>,
	/// Exact body evidence observed when response headers arrived.
	pub body:                AttemptBodyEvidence,
	/// Payload-free frame records retained within the attempt capture budget.
	pub frames:              Vec<CapturedFrame>,
}

struct HttpCaptureRecord {
	snapshot: Arc<Mutex<HttpCapture>>,
	evidence: AttemptEvidenceHandle,
}

/// Pooled production HTTP transport using the workspace rustls stack.
///
/// `poll_ready` is run on the same pooled client value moved into the following
/// `call`; a clone replaces it only for the next readiness cycle. Request-body
/// factories are opened inside that call, exactly once per attempt.
pub struct HttpTransport {
	inner:        Option<PooledClient>,
	ready_permit: bool,
	captures:     Arc<Mutex<Vec<HttpCaptureRecord>>>,
}

impl Clone for HttpTransport {
	fn clone(&self) -> Self {
		Self {
			inner:        self.inner.as_ref().cloned(),
			ready_permit: false,
			captures:     Arc::clone(&self.captures),
		}
	}
}

impl Default for HttpTransport {
	fn default() -> Self {
		Self::new()
	}
}

impl HttpTransport {
	/// Constructs a pooled rustls client supporting HTTP/1.1 and HTTP/2.
	#[must_use]
	pub fn new() -> Self {
		let _ = rustls::crypto::ring::default_provider().install_default();
		let connector = HttpsConnectorBuilder::new()
			.with_webpki_roots()
			.https_or_http()
			.enable_http1()
			.enable_http2()
			.build();
		let inner = Client::builder(TokioExecutor::new()).build(connector);
		Self {
			inner:        Some(inner),
			ready_permit: false,
			captures:     Arc::new(Mutex::new(Vec::new())),
		}
	}

	/// Returns deterministic snapshots of completed and in-flight sanitized
	/// captures.
	#[must_use]
	pub fn captures(&self) -> Vec<HttpCapture> {
		let mut captures: Vec<_> = self
			.captures
			.lock()
			.iter()
			.map(|record| {
				let mut capture = record.snapshot.lock().clone();
				capture.body = record.evidence.evidence();
				capture
			})
			.collect();
		captures.sort_by_key(|capture| capture.attempt);
		captures
	}
}

impl Service<TransportRequest> for HttpTransport {
	type Error = Error;
	type Response = HandshakenResponse;

	type Future = impl Future<Output = Result<Self::Response, Self::Error>> + Send + 'static;

	fn poll_ready(&mut self, context: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		let Some(client) = self.inner.as_mut() else {
			return Poll::Ready(Err(protocol_error(
				ErrorPhase::Readiness,
				false,
				"http-readiness-state",
			)));
		};
		match client.poll_ready(context) {
			Poll::Ready(Ok(())) => {
				self.ready_permit = true;
				Poll::Ready(Ok(()))
			},
			Poll::Ready(Err(_)) => {
				Poll::Ready(Err(connectivity(ErrorPhase::Readiness, false, "http-client-not-ready")))
			},
			Poll::Pending => Poll::Pending,
		}
	}

	fn call(&mut self, request: TransportRequest) -> Self::Future {
		let permit = std::mem::take(&mut self.ready_permit);
		let client = if permit { self.inner.take() } else { None };
		if let Some(client) = &client {
			self.inner = Some(client.clone());
		}
		let captures = Arc::clone(&self.captures);
		async move {
			let client = client.ok_or_else(|| {
				protocol_error(ErrorPhase::Readiness, false, "call-without-readiness")
			})?;
			execute(client, request, captures).await
		}
	}
}

async fn execute(
	client: PooledClient,
	mut transport: TransportRequest,
	captures: Arc<Mutex<Vec<HttpCaptureRecord>>>,
) -> Result<HandshakenResponse, Error> {
	let started = Instant::now();
	let attempt = transport.attempt.clone();
	let mut body_attempt = transport.encoded.body.begin_attempt();
	let mut evidence = body_attempt.evidence_handle();
	let deadline = tokio::time::Instant::now() + attempt.timeout;
	if !matches!((transport.decoder.is_some(), transport.realtime.is_some()), (true, false)) {
		return Err(record_failure(
			protocol_error(ErrorPhase::Handshake, false, "http-decoder-cardinality"),
			&attempt,
			&evidence,
			None,
			None,
			started,
			false,
		));
	}
	if transport.encoded.framing == FramingProtocol::WebSocket {
		return Err(record_failure(
			protocol_error(ErrorPhase::Connecting, false, "websocket-requires-socket-transport"),
			&attempt,
			&evidence,
			None,
			None,
			started,
			false,
		));
	}
	if transport.cancel.is_cancelled() {
		return Err(record_failure(
			cancelled(false),
			&attempt,
			&evidence,
			None,
			None,
			started,
			false,
		));
	}

	let mut sealed_bound = false;
	if let Some(template) = transport.encoded.take_sealed_body() {
		let Some(credentials) = transport.credentials.as_ref() else {
			return Err(record_failure(
				authentication_error("sealed-body-credentials-missing"),
				&attempt,
				&evidence,
				None,
				None,
				started,
				false,
			));
		};
		let bytes = credentials
			.finalize_sealed_body(template, &transport.cancel, transport.encoded.bounds.request_body)
			.map_err(|_| {
				let error = if transport.cancel.is_cancelled() {
					cancelled(false)
				} else {
					authentication_error("sealed-body-finalization")
				};
				record_failure(error, &attempt, &evidence, None, None, started, false)
			})?;
		let finalized = bytes;
		transport.encoded.body = BodySource::Factory(BodyFactoryHandle::new(move || {
			let bytes = finalized.clone();
			async move { Ok(byte_stream(bytes)) }
		}));
		body_attempt = transport.encoded.body.begin_attempt();
		evidence = body_attempt.evidence_handle();
		sealed_bound = true;
	} else if transport
		.credentials
		.as_ref()
		.is_some_and(|credentials| credentials.requires_sealed_body())
	{
		return Err(record_failure(
			authentication_error("sealed-body-template-missing"),
			&attempt,
			&evidence,
			None,
			None,
			started,
			false,
		));
	}

	let reader = tokio::select! {
		result = body_attempt.open() => result.map_err(|error| {
			record_failure(map_body_open_error(error), &attempt, &evidence, None, None, started, false)
		})?,
		() = poll_fn(|context| transport.cancel.poll_cancelled(context)) => {
			return Err(record_failure(cancelled(false), &attempt, &evidence, None, None, started, false));
		},
		() = tokio::time::sleep_until(deadline) => {
			transport.cancel.cancel();
			return Err(record_failure(deadline_exceeded(false), &attempt, &evidence, None, None, started, false));
		},
	};
	let request = if transport
		.credentials
		.as_ref()
		.is_some_and(|credentials| credentials.requires_buffered_body())
	{
		let bytes = collect_request_body(
			reader,
			transport.encoded.bounds.request_body,
			&transport.cancel,
			deadline,
		)
		.await
		.map_err(|error| record_failure(error, &attempt, &evidence, None, None, started, false))?;
		let mut request = build_request(&transport, bytes)
			.map_err(|error| record_failure(error, &attempt, &evidence, None, None, started, false))?;
		transport
			.credentials
			.as_ref()
			.expect("buffered credentials checked")
			.finalize_buffered(&mut request)
			.map_err(|_| {
				record_failure(
					authentication_error("credential-finalization"),
					&attempt,
					&evidence,
					None,
					None,
					started,
					false,
				)
			})?;
		let (parts, bytes) = request.into_parts();
		let body = Full::new(bytes)
			.map_err(|never: Infallible| -> Error { match never {} })
			.boxed_unsync();
		Request::from_parts(parts, body)
	} else {
		let body = StreamBody::new(LimitedBodyStream {
			reader,
			cancel: transport.cancel.clone(),
			seen: 0,
			limit: transport.encoded.bounds.request_body,
			done: false,
		})
		.boxed_unsync();
		let mut request = build_request(&transport, body)
			.map_err(|error| record_failure(error, &attempt, &evidence, None, None, started, false))?;
		if !sealed_bound
			&& let Some(credentials) = &transport.credentials
			&& credentials.finalize_streaming(&mut request).is_err()
		{
			drop(request);
			return Err(record_failure(
				authentication_error("credential-finalization"),
				&attempt,
				&evidence,
				None,
				None,
				started,
				false,
			));
		}
		request
	};
	let response = tokio::select! {
		result = client.oneshot(request) => result.map_err(|_| {
			record_failure(connectivity(ErrorPhase::Connecting, false, "http-dispatch"), &attempt, &evidence, None, None, started, false)
		})?,
		() = poll_fn(|context| transport.cancel.poll_cancelled(context)) => {
			return Err(record_failure(cancelled(false), &attempt, &evidence, None, None, started, false));
		},
		() = tokio::time::sleep_until(deadline) => {
			transport.cancel.cancel();
			return Err(record_failure(deadline_exceeded(false), &attempt, &evidence, None, None, started, false));
		},
	};
	let (parts, incoming) = response.into_parts();
	let status = parts.status.as_u16();
	let headers = sanitize_headers(&parts.headers);
	let provider_request_id = request_id(&parts.headers);
	let capture = Arc::new(Mutex::new(HttpCapture {
		attempt: transport.attempt.index,
		status,
		provider_request_id: provider_request_id.clone(),
		body: evidence.evidence(),
		frames: Vec::new(),
	}));
	captures
		.lock()
		.push(HttpCaptureRecord { snapshot: Arc::clone(&capture), evidence: evidence.clone() });

	let framing = transport.encoded.framing;
	let frame_limit = usize::try_from(transport.encoded.bounds.frame).unwrap_or(usize::MAX);
	let response_limit = transport.encoded.bounds.response;
	let event_stream = decode_stream(
		incoming,
		framing,
		frame_limit,
		response_limit,
		transport.decoder.take().ok_or_else(|| {
			record_failure(
				protocol_error(ErrorPhase::Handshake, false, "ordinary-decoder-missing"),
				&attempt,
				&evidence,
				None,
				None,
				started,
				false,
			)
		})?,
		transport.cancel.clone(),
		transport.attempt.capture_limit,
		capture,
		attempt.clone(),
		evidence.clone(),
		status,
		provider_request_id.clone(),
		deadline,
		started,
	);
	let mut event_stream: RawEventStream = Box::pin(event_stream);
	let mut preamble = VecDeque::new();
	let first_visible = loop {
		match event_stream.next().await {
			Some(Ok(event)) if is_commit_candidate(&event) => break event,
			Some(Ok(event)) => preamble.push_back(event),
			Some(Err(mut error)) => {
				error.status = Some(status);
				error.committed = false;
				error.phase = ErrorPhase::Handshake;
				return Err(error);
			},
			None => {
				return Err(record_failure(
					protocol_error(ErrorPhase::Handshake, false, "response-ended-before-commit-event"),
					&attempt,
					&evidence,
					Some(status),
					provider_request_id.as_ref(),
					started,
					false,
				));
			},
		}
	};
	preamble.push_back(first_visible);
	let events: RawEventStream =
		Box::pin(stream::iter(preamble.into_iter().map(Ok)).chain(event_stream));
	Ok(HandshakenResponse {
		meta:     HandshakeMeta { status: Some(status), headers, provider_request_id },
		body:     evidence,
		events:   Some(events),
		realtime: None,
	})
}

fn build_request<B>(transport: &TransportRequest, body: B) -> Result<Request<B>, Error> {
	let method = match transport.encoded.method {
		RequestMethod::Get => Method::GET,
		RequestMethod::Post => Method::POST,
		RequestMethod::Put => Method::PUT,
		RequestMethod::Patch => Method::PATCH,
		RequestMethod::Delete => Method::DELETE,
	};
	let uri = transport
		.encoded
		.uri
		.as_str()
		.parse::<Uri>()
		.map_err(|_| protocol_error(ErrorPhase::Encoding, false, "invalid-http-uri"))?;
	let mut request = Request::builder()
		.method(method)
		.uri(uri)
		.body(body)
		.map_err(|_| protocol_error(ErrorPhase::Encoding, false, "invalid-http-request"))?;
	for item in &transport.encoded.headers {
		insert_header(request.headers_mut(), item.name.as_str(), item.value.as_str())?;
	}
	request
		.headers_mut()
		.entry(header::USER_AGENT)
		.or_insert(HeaderValue::from_static(omp_core::USER_AGENT));
	Ok(request)
}

fn insert_header(headers: &mut HeaderMap, name: &str, value: &str) -> Result<(), Error> {
	let name = HeaderName::from_bytes(name.as_bytes())
		.map_err(|_| protocol_error(ErrorPhase::Encoding, false, "invalid-header-name"))?;
	let value = HeaderValue::from_str(value)
		.map_err(|_| protocol_error(ErrorPhase::Encoding, false, "invalid-header-value"))?;
	headers.insert(name, value);
	Ok(())
}

fn map_body_open_error(error: BodyOpenError) -> Error {
	match error {
		BodyOpenError::Factory(error) => error,
		BodyOpenError::AttemptAlreadyOpened => {
			protocol_error(ErrorPhase::Connecting, false, "request-body-attempt-already-opened")
		},
		BodyOpenError::ConcurrentReader => {
			protocol_error(ErrorPhase::Connecting, false, "request-body-concurrent-reader")
		},
		BodyOpenError::Consumed => {
			protocol_error(ErrorPhase::Connecting, false, "request-body-consumed")
		},
		BodyOpenError::ReacquisitionUnavailable => {
			protocol_error(ErrorPhase::Connecting, false, "request-body-reacquisition-unavailable")
		},
	}
}

/// Extracts bounded provider-controlled correlation data from a closed header
/// allowlist. These values are never credential material or copied to outgoing
/// requests; the strict opaque-ID alphabet prevents header-shaped reflections.
pub(crate) fn request_id(headers: &HeaderMap) -> Option<Str> {
	let value = PUBLIC_REQUEST_ID_HEADERS
		.iter()
		.find_map(|name| headers.get(*name))?;
	let value = value.to_str().ok()?;
	(!value.is_empty()
		&& value.len() <= MAX_REQUEST_ID_BYTES
		&& value
			.bytes()
			.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.')))
	.then(|| Str::from(value))
}

pub(crate) fn sanitize_headers(headers: &HeaderMap) -> Box<[RequestHeader]> {
	headers
		.iter()
		.filter_map(|(name, value)| {
			if value.as_bytes().len() > MAX_CAPTURED_HEADER_BYTES
				|| !PUBLIC_NUMERIC_HEADERS.contains(&name.as_str())
			{
				return None;
			}
			let value = value.to_str().ok()?.trim().parse::<u64>().ok()?;
			Some(RequestHeader {
				name:  Str::from(name.as_str()),
				value: Str::from(value.to_string()),
			})
		})
		.take(MAX_CAPTURED_HEADERS)
		.collect::<Vec<_>>()
		.into_boxed_slice()
}

struct LimitedBodyStream {
	reader: BodyReader,
	cancel: Cancellation,
	seen:   u64,
	limit:  u64,
	done:   bool,
}

impl Stream for LimitedBodyStream {
	type Item = Result<BodyFrame<Bytes>, Error>;

	fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
		if self.done {
			return Poll::Ready(None);
		}
		if self.cancel.poll_cancelled(context).is_ready() {
			self.done = true;
			return Poll::Ready(Some(Err(cancelled(false))));
		}
		match Pin::new(&mut self.reader).poll_next(context) {
			Poll::Ready(Some(Ok(chunk))) => {
				self.seen = self.seen.saturating_add(chunk.len() as u64);
				if self.seen > self.limit {
					self.done = true;
					return Poll::Ready(Some(Err(protocol_error(
						ErrorPhase::Connecting,
						false,
						"request-body-limit",
					))));
				}
				Poll::Ready(Some(Ok(BodyFrame::data(chunk))))
			},
			Poll::Ready(Some(Err(error))) => {
				self.done = true;
				Poll::Ready(Some(Err(error)))
			},
			Poll::Ready(None) => {
				self.done = true;
				Poll::Ready(None)
			},
			Poll::Pending => Poll::Pending,
		}
	}
}

async fn collect_request_body(
	mut reader: BodyReader,
	limit: u64,
	cancel: &Cancellation,
	deadline: tokio::time::Instant,
) -> Result<Bytes, Error> {
	let capacity = usize::try_from(limit.min(64 * 1024)).unwrap_or(64 * 1024);
	let mut output = BytesMut::with_capacity(capacity);
	loop {
		let next = tokio::select! {
			next = reader.next() => next,
			() = poll_fn(|context| cancel.poll_cancelled(context)) => return Err(cancelled(false)),
			() = tokio::time::sleep_until(deadline) => {
				cancel.cancel();
				return Err(deadline_exceeded(false));
			},
		};
		match next {
			Some(Ok(chunk)) => {
				let observed = (output.len() as u64).saturating_add(chunk.len() as u64);
				if observed > limit {
					return Err(protocol_error(ErrorPhase::Connecting, false, "request-body-limit"));
				}
				output.extend_from_slice(&chunk);
			},
			Some(Err(error)) => return Err(error),
			None => return Ok(output.freeze()),
		}
	}
}

fn decode_stream(
	mut incoming: Incoming,
	protocol: FramingProtocol,
	frame_limit: usize,
	response_limit: u64,
	mut decoder: crate::codec::DecoderState,
	cancel: Cancellation,
	capture_limit: u64,
	capture: Arc<Mutex<HttpCapture>>,
	attempt: TransportAttempt,
	evidence: AttemptEvidenceHandle,
	status: u16,
	provider_request_id: Option<Str>,
	deadline: tokio::time::Instant,
	started: Instant,
) -> impl Stream<Item = Result<RawEvent, Error>> + Send + 'static {
	async_stream::stream! {
		let mut guard = CancelOnDrop::new(cancel.clone());
		let mut framer = ResponseFramer::new(protocol, frame_limit);
		let mut response_bytes = 0_u64;
		let mut capture_remaining = capture_limit;
		let mut ordinal = 0_u64;
		let mut emitted = false;
		loop {
			let next = tokio::select! {
				next = incoming.frame() => next,
				() = poll_fn(|context| cancel.poll_cancelled(context)) => {
					yield Err(record_failure(cancelled(emitted), &attempt, &evidence, Some(status), provider_request_id.as_ref(), started, emitted));
					break;
				},
				() = tokio::time::sleep_until(deadline) => {
					cancel.cancel();
					yield Err(record_failure(deadline_exceeded(emitted), &attempt, &evidence, Some(status), provider_request_id.as_ref(), started, emitted));
					break;
				},
			};
			let Some(next) = next else {
				match framer.finish() {
					Ok(frames) => {
						for frame in frames {
							capture_http_frame(&capture, ordinal, &frame, &mut capture_remaining);
							ordinal += 1;
							let mut events = VecDeque::new();
							if let Err(error) = decoder.push(frame, &mut |event| events.push_back(event)) {
								yield Err(record_failure(error, &attempt, &evidence, Some(status), provider_request_id.as_ref(), started, emitted));
								return;
							}
							for event in events {
								match event {
									RawEvent::Failure(error) => {
										yield Err(record_failure(error, &attempt, &evidence, Some(status), provider_request_id.as_ref(), started, emitted));
										return;
									},
									event => {
										emitted |= is_commit_candidate(&event);
										yield Ok(event);
									},
								}
							}
						}
					},
					Err(error) => {
						yield Err(record_failure(framing_error(error, emitted), &attempt, &evidence, Some(status), provider_request_id.as_ref(), started, emitted));
						break;
					},
				}
				let mut events = VecDeque::new();
				match decoder.finish(&mut |event| events.push_back(event)) {
					Ok(()) => for event in events {
						match event {
							RawEvent::Failure(error) => {
								yield Err(record_failure(error, &attempt, &evidence, Some(status), provider_request_id.as_ref(), started, emitted));
								return;
							},
							event => {
								emitted |= is_commit_candidate(&event);
								yield Ok(event);
							},
						}
					},
					Err(error) => {
						yield Err(record_failure(error, &attempt, &evidence, Some(status), provider_request_id.as_ref(), started, emitted));
					},
				}
				guard.disarm();
				break;
			};
			let body_frame = match next {
				Ok(frame) => frame,
				Err(_) => {
					let error = connectivity(if emitted { ErrorPhase::Streaming } else { ErrorPhase::Handshake }, emitted, "http-response-body");
					yield Err(record_failure(error, &attempt, &evidence, Some(status), provider_request_id.as_ref(), started, emitted));
					break;
				},
			};
			let Ok(chunk) = body_frame.into_data() else { continue };
			response_bytes = response_bytes.saturating_add(chunk.len() as u64);
			if response_bytes > response_limit {
				let error = protocol_error(if emitted { ErrorPhase::Streaming } else { ErrorPhase::Handshake }, emitted, "response-body-limit");
				yield Err(record_failure(error, &attempt, &evidence, Some(status), provider_request_id.as_ref(), started, emitted));
				break;
			}
			let frames = match framer.push(chunk) {
				Ok(frames) => frames,
				Err(error) => {
					yield Err(record_failure(framing_error(error, emitted), &attempt, &evidence, Some(status), provider_request_id.as_ref(), started, emitted));
					break;
				},
			};
			for frame in frames {
				capture_http_frame(&capture, ordinal, &frame, &mut capture_remaining);
				ordinal += 1;
				let mut events = VecDeque::new();
				if let Err(error) = decoder.push(frame, &mut |event| events.push_back(event)) {
					yield Err(record_failure(error, &attempt, &evidence, Some(status), provider_request_id.as_ref(), started, emitted));
					return;
				}
				for event in events {
					match event {
						RawEvent::Failure(error) => {
							yield Err(record_failure(error, &attempt, &evidence, Some(status), provider_request_id.as_ref(), started, emitted));
							return;
						},
						event => {
							emitted |= is_commit_candidate(&event);
							yield Ok(event);
						},
					}
				}
			}
		}
	}
}

fn capture_http_frame(
	capture: &Arc<Mutex<HttpCapture>>,
	ordinal: u64,
	frame: &Frame,
	remaining: &mut u64,
) {
	let mut capture = capture.lock();
	capture_frame(&mut capture.frames, ordinal, frame, remaining);
}

enum ResponseFramer {
	Raw { buffer: BytesMut, limit: usize },
	RawChunks(RawChunkFramer),
	Sse(SseDecoder),
	Ndjson(NdjsonDecoder),
	Connect(ConnectDecoder),
	EventStream(EventStreamDecoder),
}

impl ResponseFramer {
	fn new(protocol: FramingProtocol, limit: usize) -> Self {
		match protocol {
			FramingProtocol::Raw => Self::Raw { buffer: BytesMut::new(), limit },
			FramingProtocol::RawChunks => Self::RawChunks(RawChunkFramer::new(limit)),
			FramingProtocol::Sse => Self::Sse(SseDecoder::with_max_frame_bytes(limit)),
			FramingProtocol::Ndjson => Self::Ndjson(NdjsonDecoder::with_max_frame_bytes(limit)),
			FramingProtocol::Connect => Self::Connect(ConnectDecoder::with_max_payload_bytes(limit)),
			FramingProtocol::AwsEventStream => {
				Self::EventStream(EventStreamDecoder::with_limits(limit, limit.min(128 * 1024)))
			},
			FramingProtocol::WebSocket => unreachable!("rejected before response framing"),
		}
	}

	fn push(&mut self, chunk: Bytes) -> Result<SmallVec<Frame, 4>, FramingError> {
		match self {
			Self::Raw { buffer, limit } => {
				let observed = buffer.len().saturating_add(chunk.len());
				if observed > *limit {
					return Err(FramingError::LimitExceeded {
						protocol: FramingProtocol::Raw,
						limit: *limit,
						observed,
					});
				}
				buffer.extend_from_slice(&chunk);
				Ok(SmallVec::new())
			},
			Self::RawChunks(framer) => framer
				.push(chunk)
				.map(|frames| frames.into_iter().map(Frame::Raw).collect()),
			Self::Sse(framer) => framer
				.push(chunk)
				.map(|frames| frames.into_iter().map(Frame::Sse).collect()),
			Self::Ndjson(framer) => framer
				.push(chunk)
				.map(|frames| frames.into_iter().map(Frame::Ndjson).collect()),
			Self::Connect(framer) => framer
				.push(chunk)
				.map(|frames| frames.into_iter().map(Frame::Connect).collect()),
			Self::EventStream(framer) => framer
				.push(chunk)
				.map(|frames| frames.into_iter().map(Frame::EventStream).collect()),
		}
	}

	fn finish(&mut self) -> Result<SmallVec<Frame, 4>, FramingError> {
		match self {
			Self::Raw { buffer, .. } => {
				let mut frames = SmallVec::new();
				frames.push(Frame::Raw(buffer.split().freeze()));
				Ok(frames)
			},
			Self::RawChunks(framer) => framer
				.finish()
				.map(|frames| frames.into_iter().map(Frame::Raw).collect()),
			Self::Sse(framer) => framer
				.finish()
				.map(|frames| frames.into_iter().map(Frame::Sse).collect()),
			Self::Ndjson(framer) => framer
				.finish()
				.map(|frames| frames.into_iter().map(Frame::Ndjson).collect()),
			Self::Connect(framer) => framer
				.finish()
				.map(|frames| frames.into_iter().map(Frame::Connect).collect()),
			Self::EventStream(framer) => framer
				.finish()
				.map(|frames| frames.into_iter().map(Frame::EventStream).collect()),
		}
	}
}

struct CancelOnDrop {
	cancel: Cancellation,
	armed:  bool,
}

impl CancelOnDrop {
	fn new(cancel: Cancellation) -> Self {
		Self { cancel, armed: true }
	}

	fn disarm(&mut self) {
		self.armed = false;
	}
}

impl Drop for CancelOnDrop {
	fn drop(&mut self) {
		if self.armed {
			self.cancel.cancel();
		}
	}
}

fn framing_error(error: FramingError, committed: bool) -> Error {
	let (kind, reason) = match error {
		FramingError::AfterEnd { .. } => (ErrorKind::Protocol, "framing-after-end"),
		FramingError::Cancelled { .. } => (ErrorKind::Cancelled, "framing-cancelled"),
		FramingError::LimitExceeded { .. } => (ErrorKind::StreamCorruption, "framing-limit"),
		FramingError::UnexpectedEof { .. } => (ErrorKind::StreamCorruption, "framing-truncated"),
		FramingError::InvalidFlags { .. } => (ErrorKind::StreamCorruption, "framing-invalid-flags"),
		FramingError::InvalidWebSocketOpcode { .. } => {
			(ErrorKind::StreamCorruption, "websocket-opcode")
		},
		FramingError::NonCanonicalWebSocketLength { .. } => {
			(ErrorKind::StreamCorruption, "websocket-noncanonical-length")
		},
		FramingError::InvalidWebSocketControl => (ErrorKind::StreamCorruption, "websocket-control"),
		FramingError::InvalidWebSocketClose => (ErrorKind::StreamCorruption, "websocket-close"),
		FramingError::InvalidUtf8 { .. } => (ErrorKind::StreamCorruption, "framing-invalid-utf8"),
		FramingError::CrcMismatch { scope: crate::transport::CrcScope::Prelude, .. } => {
			(ErrorKind::StreamCorruption, "eventstream-prelude-crc")
		},
		FramingError::CrcMismatch { scope: crate::transport::CrcScope::Message, .. } => {
			(ErrorKind::StreamCorruption, "eventstream-message-crc")
		},
		FramingError::InvalidEventStreamLengths { .. } => {
			(ErrorKind::StreamCorruption, "eventstream-lengths")
		},
		FramingError::InvalidEventStreamHeader { .. } => {
			(ErrorKind::StreamCorruption, "eventstream-header")
		},
		FramingError::UnknownEventStreamHeaderType { .. } => {
			(ErrorKind::StreamCorruption, "eventstream-header-type")
		},
	};
	let phase = if committed {
		ErrorPhase::Streaming
	} else {
		ErrorPhase::Handshake
	};
	let mut error = structured_error(kind, phase, committed, reason);
	error.code = Some(Str::from(reason));
	if !committed && error.kind != ErrorKind::Cancelled {
		error.action = RetryAction::SameRoute { after: std::time::Duration::ZERO };
	}
	error
}

pub(crate) fn record_failure(
	mut error: Error,
	attempt: &TransportAttempt,
	evidence: &AttemptEvidenceHandle,
	status: Option<u16>,
	provider_request_id: Option<&Str>,
	started: Instant,
	committed: bool,
) -> Error {
	error.committed = committed;
	if committed {
		error.phase = ErrorPhase::Streaming;
		error.action = RetryAction::Never;
	}
	error.provider = Some(attempt.provider.clone());
	error.route = Some(attempt.route.clone());
	error.request_id = Some(attempt.request_id.clone());
	if error.status.is_none() {
		error.status = status;
	}
	let outcome = if error.kind == ErrorKind::Cancelled {
		AttemptOutcome::Cancelled
	} else if committed {
		AttemptOutcome::FailedCommitted
	} else {
		AttemptOutcome::FailedPreCommit
	};
	error.receipt.record_attempt(AttemptReceipt {
		index: attempt.index,
		hidden: attempt.provisional,
		provider: Some(attempt.provider.clone()),
		route: Some(attempt.route.clone()),
		account: attempt.account.clone(),
		principal: attempt.principal.clone(),
		body: evidence.evidence(),
		outcome,
		usage: Usage::default(),
		cost: Cost::default(),
		provider_evidence: ProviderEvidence {
			request_id: provider_request_id.cloned(),
			status:     error.status,
			code:       error.code.clone(),
			summary:    None,
		},
		elapsed: started.elapsed(),
	});
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
	let mut error = Error::new(
		ErrorKind::Cancelled,
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

fn authentication_error(reason: &'static str) -> Error {
	structured_error(ErrorKind::Authentication, ErrorPhase::Authentication, false, reason)
}

fn connectivity(phase: ErrorPhase, committed: bool, reason: &'static str) -> Error {
	let mut error = structured_error(ErrorKind::Connectivity, phase, committed, reason);
	if !committed {
		error.action = RetryAction::SameRoute { after: std::time::Duration::ZERO };
	}
	error
}

fn protocol_error(phase: ErrorPhase, committed: bool, reason: &'static str) -> Error {
	structured_error(ErrorKind::Protocol, phase, committed, reason)
}

fn structured_error(
	kind: ErrorKind,
	phase: ErrorPhase,
	committed: bool,
	reason: &'static str,
) -> Error {
	let mut error = Error::new(kind, phase, RetryAction::Never, ExecutionReceipt::default());
	error.committed = committed;
	error.detail = Some(ErrorDetail::Protocol { reason: ReasonId(Str::from(reason)) });
	error
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn reflected_secret_is_rejected_for_every_public_header_surface() {
		const SECRET: &str = "Bearer reflected-super-secret";
		let mut headers = HeaderMap::new();
		for name in PUBLIC_NUMERIC_HEADERS {
			headers.insert(HeaderName::from_static(name), HeaderValue::from_static(SECRET));
		}
		for name in [
			"content-type",
			"date",
			"traceparent",
			"x-request-id",
			"x-amzn-requestid",
			"x-ratelimit-reflection",
		] {
			headers.insert(HeaderName::from_static(name), HeaderValue::from_static(SECRET));
		}
		let sanitized = sanitize_headers(&headers);
		assert!(sanitized.is_empty());
		assert!(!format!("{sanitized:?}").contains(SECRET));
	}

	#[test]
	fn request_id_uses_closed_names_and_bounded_opaque_values() {
		let mut headers = HeaderMap::new();
		headers.insert("x-request-id", HeaderValue::from_static("req_01HZX-abc.7"));
		assert_eq!(request_id(&headers).as_deref(), Some("req_01HZX-abc.7"));
		headers.insert("x-request-id", HeaderValue::from_static("Bearer reflected-super-secret"));
		assert!(request_id(&headers).is_none());
		headers.clear();
		headers.insert("x-provider-request-id", HeaderValue::from_static("looks-safe"));
		assert!(request_id(&headers).is_none());
	}

	#[test]
	fn only_closed_numeric_fields_survive_in_canonical_form() {
		let mut headers = HeaderMap::new();
		for name in PUBLIC_NUMERIC_HEADERS {
			headers.insert(HeaderName::from_static(name), HeaderValue::from_static("0007"));
		}
		let sanitized = sanitize_headers(&headers);
		assert_eq!(sanitized.len(), PUBLIC_NUMERIC_HEADERS.len());
		assert!(sanitized.iter().all(|header| header.value.as_str() == "7"));
	}

	#[test]
	fn request_id_uses_first_present_allowlisted_header_only() {
		let mut headers = HeaderMap::new();
		headers.insert("x-request-id", HeaderValue::from_static("invalid value"));
		headers.insert("request-id", HeaderValue::from_static("would-be-valid"));
		assert!(request_id(&headers).is_none());

		headers.clear();
		for name in PUBLIC_REQUEST_ID_HEADERS {
			headers
				.insert(HeaderName::from_static(name), HeaderValue::from_static("provider_01.test-id"));
			assert_eq!(request_id(&headers).as_deref(), Some("provider_01.test-id"));
			headers.clear();
		}

		let oversized = "a".repeat(MAX_REQUEST_ID_BYTES + 1);
		headers.insert("x-request-id", HeaderValue::from_str(&oversized).expect("valid header"));
		assert!(request_id(&headers).is_none());
	}

	#[test]
	fn factory_open_error_preserves_typed_failure() {
		let mut inner = Error::new(
			ErrorKind::RateLimited,
			ErrorPhase::Admission,
			RetryAction::SameRoute { after: std::time::Duration::from_secs(3) },
			ExecutionReceipt::default(),
		);
		inner.detail =
			Some(ErrorDetail::Protocol { reason: ReasonId(Str::new_static("factory-rate-window")) });
		let mapped = map_body_open_error(BodyOpenError::Factory(inner.clone()));
		assert_eq!(mapped.kind, inner.kind);
		assert_eq!(mapped.phase, inner.phase);
		assert_eq!(mapped.action, inner.action);
		assert_eq!(mapped.detail, inner.detail);
		assert_eq!(mapped.receipt, inner.receipt);
	}
}
