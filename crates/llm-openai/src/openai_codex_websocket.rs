//! ChatGPT Codex WebSocket framing, continuation, and HTTP fallback policy.
//!
//! Network I/O is deliberately absent. The production egress service owns the
//! socket and credential lease; these types shape frames and enforce the replay
//! invariants at that boundary.

use std::fmt;

use omp_core::Str;
use serde_json::{Map, Value};

/// Codex WebSocket request discriminator.
pub const RESPONSE_CREATE: &str = "response.create";

/// Converts an HTTP(S) Codex Responses endpoint into its WebSocket endpoint.
pub fn codex_websocket_url(url: &str) -> Result<String, CodexWebSocketProtocolError> {
	if let Some(rest) = url.strip_prefix("https://") {
		Ok(format!("wss://{rest}"))
	} else if let Some(rest) = url.strip_prefix("http://") {
		Ok(format!("ws://{rest}"))
	} else if url.starts_with("wss://") || url.starts_with("ws://") {
		Ok(url.to_owned())
	} else {
		Err(CodexWebSocketProtocolError(Str::new(
			"Codex WebSocket endpoint requires http, https, ws, or wss",
		)))
	}
}

/// Produces the full-context body used for replay-safe HTTP fallback.
///
/// Callers must retain the authoritative transformed request rather than trying
/// to expand a delta-only WebSocket frame. This helper removes the two
/// WebSocket-only continuation fields that the HTTP endpoint rejects.
pub fn codex_http_fallback_body(
	full_request: &Value,
) -> Result<Value, CodexWebSocketProtocolError> {
	let mut body = full_request.as_object().cloned().ok_or_else(|| {
		CodexWebSocketProtocolError(Str::new("Codex HTTP fallback body must be a JSON object"))
	})?;
	body.remove("type");
	body.remove("previous_response_id");
	Ok(Value::Object(body))
}

/// Result of routing one inbound WebSocket frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CodexFrameDisposition {
	/// Deliver the frame to the Responses decoder.
	Deliver,
	/// Deliver this sole terminal frame, then stop reading the turn.
	Terminal,
	/// Discard a stale, post-terminal, or cancelled frame.
	Drop,
}

/// Protocol violation that makes continued use of a socket unsafe.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CodexWebSocketProtocolError(Str);

impl fmt::Display for CodexWebSocketProtocolError {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter.write_str(&self.0)
	}
}

impl std::error::Error for CodexWebSocketProtocolError {}

/// Per-request inbound-frame correlator for a reused Codex WebSocket.
///
/// Lifecycle frames are correlated by `response.id`. Once a response id is
/// observed, a different id or a regressing sequence closes the replay window
/// instead of risking cross-turn attribution. Cancellation drops all later
/// frames and tells the socket owner to drop the upstream body.
#[derive(Clone, Debug, Default)]
pub struct CodexFrameRouter {
	prior_response_id:  Option<Str>,
	active_response_id: Option<Str>,
	last_sequence:      Option<i64>,
	terminal:           bool,
	cancelled:          bool,
}

impl CodexFrameRouter {
	/// Creates a router that can recognize a late frame from the prior response.
	#[must_use]
	pub fn after_response(response_id: impl Into<Str>) -> Self {
		Self { prior_response_id: Some(response_id.into()), ..Self::default() }
	}

	/// Marks the request cancelled. The egress owner must drop the upstream body
	/// immediately; subsequent frames are discarded by [`Self::route`].
	pub fn cancel(&mut self) {
		self.cancelled = true;
	}

	/// Returns whether cancellation or a terminal event has ended this turn.
	#[must_use]
	pub const fn is_finished(&self) -> bool {
		self.cancelled || self.terminal
	}

	/// Correlates and classifies one decoded JSON frame.
	pub fn route(
		&mut self,
		frame: &Value,
	) -> Result<CodexFrameDisposition, CodexWebSocketProtocolError> {
		if self.cancelled || self.terminal {
			return Ok(CodexFrameDisposition::Drop);
		}
		let object = frame.as_object().ok_or_else(|| {
			CodexWebSocketProtocolError(Str::new("Codex WebSocket frame must be a JSON object"))
		})?;
		if let Some(response_id) = response_id(object) {
			if self.active_response_id.is_none()
				&& self.prior_response_id.as_deref() == Some(response_id)
			{
				return Ok(CodexFrameDisposition::Drop);
			}
			if let Some(active) = &self.active_response_id {
				if active != response_id {
					return Err(CodexWebSocketProtocolError(Str::from(format!(
						"Codex WebSocket response {response_id} interleaved into active response \
						 {active}",
					))));
				}
			} else {
				self.active_response_id = Some(Str::new(response_id));
			}
		}
		if let Some(sequence) = object.get("sequence_number").and_then(Value::as_i64) {
			if self.last_sequence.is_some_and(|last| sequence < last) {
				return Err(CodexWebSocketProtocolError(Str::from(format!(
					"Codex WebSocket sequence {sequence} regressed",
				))));
			}
			self.last_sequence = Some(sequence);
		}
		if is_terminal(
			object
				.get("type")
				.and_then(Value::as_str)
				.unwrap_or_default(),
		) {
			self.terminal = true;
			return Ok(CodexFrameDisposition::Terminal);
		}
		Ok(CodexFrameDisposition::Deliver)
	}

	/// Returns the response id authoritatively selected for this request.
	#[must_use]
	pub fn active_response_id(&self) -> Option<&str> {
		self.active_response_id.as_deref()
	}
}

/// Replay state retained after an authoritative successful terminal response.
#[derive(Clone, Debug, Default)]
pub struct CodexContinuationState {
	previous_request:        Option<Value>,
	previous_response_id:    Option<Str>,
	previous_response_items: Vec<Value>,
	turn_state:              Option<Str>,
	models_etag:             Option<Str>,
}

impl CodexContinuationState {
	/// Records a successful terminal response as the next append baseline.
	pub fn commit(
		&mut self,
		request: Value,
		response_id: impl Into<Str>,
		response_items: Vec<Value>,
	) {
		self.previous_request = Some(request);
		self.previous_response_id = Some(response_id.into());
		self.previous_response_items = response_items;
	}

	/// Clears all history-dependent state after cancellation, stale anchors, or
	/// a fresh socket handshake.
	pub fn reset(&mut self) {
		self.previous_request = None;
		self.previous_response_id = None;
		self.previous_response_items.clear();
		self.turn_state = None;
		self.models_etag = None;
	}

	/// Updates response metadata learned from handshake or `response.metadata`
	/// headers. Missing fields preserve the last authoritative values.
	pub fn update_metadata(&mut self, turn_state: Option<&str>, models_etag: Option<&str>) {
		if let Some(value) = turn_state {
			self.turn_state = Some(Str::new(value));
		}
		if let Some(value) = models_etag {
			self.models_etag = Some(Str::new(value));
		}
	}

	/// Returns the current per-turn continuation header value.
	#[must_use]
	pub fn turn_state(&self) -> Option<&str> {
		self.turn_state.as_deref()
	}

	/// Returns the current models ETag learned from the service.
	#[must_use]
	pub fn models_etag(&self) -> Option<&str> {
		self.models_etag.as_deref()
	}

	/// Starts a new user turn while preserving the append baseline. Codex turn
	/// state is scoped to one turn; the models ETag remains connection-scoped.
	pub fn start_new_turn(&mut self) {
		self.turn_state = None;
	}

	/// Shapes a `response.create` frame, using `previous_response_id` only when
	/// the current history is a strict append and all non-input options match.
	/// A mismatch clears continuation metadata and sends the full transcript.
	pub fn response_create(
		&mut self,
		current: &Value,
	) -> Result<Value, CodexWebSocketProtocolError> {
		let current_object = current.as_object().ok_or_else(|| {
			CodexWebSocketProtocolError(Str::new(
				"Codex response.create body must be a JSON object",
			))
		})?;
		let mut wire = current_object.clone();
		if let (Some(previous), Some(response_id)) =
			(self.previous_request.as_ref(), self.previous_response_id.as_ref())
		{
			if let Some(delta) = strict_delta(previous, &self.previous_response_items, current) {
				wire.insert("previous_response_id".into(), Value::String(response_id.to_string()));
				wire.insert("input".into(), Value::Array(delta));
			} else {
				self.reset();
			}
		}
		wire.insert("type".into(), Value::String(RESPONSE_CREATE.into()));
		Ok(Value::Object(wire))
	}
}

/// WebSocket failure class used by the replay policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CodexWebSocketFailure {
	/// Handshake failure known to be connection-fatal.
	ConnectionFatal,
	/// Service rejected the account's concurrent WebSocket connection count.
	ConnectionLimit,
	/// Socket close, send failure, timeout, malformed JSON, or sequence failure.
	RetryableTransport,
	/// Retryable in-band provider failure before output.
	RetryableProvider,
	/// Non-retryable in-band provider failure.
	Provider,
	/// Caller cancellation.
	Cancelled,
}

/// Classifies a WebSocket or in-band failure without broad string fallback.
///
/// Only Pi-proven transport phrases and provider codes are retryable. Unknown
/// failures remain provider failures and therefore never trigger HTTP replay.
#[must_use]
pub fn classify_codex_websocket_failure(
	code: Option<&str>,
	message: &str,
	cancelled: bool,
) -> CodexWebSocketFailure {
	if cancelled {
		return CodexWebSocketFailure::Cancelled;
	}
	if code == Some("websocket_connection_limit_reached") {
		return CodexWebSocketFailure::ConnectionLimit;
	}
	if matches!(code, Some("model_error" | "server_error" | "internal_error")) {
		return CodexWebSocketFailure::RetryableProvider;
	}
	let message = message.to_ascii_lowercase();
	if ["websocket error:", "websocket closed before open", "connection timeout"]
		.iter()
		.any(|pattern| message.contains(pattern))
	{
		return CodexWebSocketFailure::ConnectionFatal;
	}
	if [
		"websocket closed (",
		"websocket closed before response completion",
		"websocket connection is unavailable",
		"websocket send failed",
		"websocket ping failed",
		"websocket pong timeout",
		"websocket message queue exceeded",
		"websocket request already in progress",
		"idle timeout waiting for websocket",
		"timeout waiting for first websocket event",
		"syntaxerror",
		"json",
	]
	.iter()
	.any(|pattern| message.contains(pattern))
	{
		return CodexWebSocketFailure::RetryableTransport;
	}
	CodexWebSocketFailure::Provider
}

/// Delivery facts that decide whether replay could duplicate observable work.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CodexReplaySafety {
	/// Text or thinking has been buffered for this attempt.
	pub has_output:          bool,
	/// A tool call has already reached the caller and therefore cannot be
	/// replayed.
	pub delivered_tool_call: bool,
	/// A terminal event has already been accepted.
	pub terminal:            bool,
}

/// Action selected after a Codex WebSocket failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CodexFallbackAction {
	/// Open a fresh WebSocket and replay the full request.
	ReconnectWebSocket,
	/// Replay the full request through HTTP SSE.
	ReplayOverHttp,
	/// Surface the failure without replay.
	Surface,
	/// Propagate caller cancellation without fallback.
	Cancelled,
}

/// Applies Codex's replay-safe WebSocket-to-HTTP fallback rules.
///
/// HTTP replay is never selected after terminal delivery or a delivered tool
/// call. Retryable empty attempts first consume the WebSocket retry budget;
/// connection-fatal or output-buffered failures move directly to HTTP.
#[must_use]
pub const fn classify_codex_fallback(
	failure: CodexWebSocketFailure,
	safety: CodexReplaySafety,
	retries: u32,
	retry_budget: u32,
) -> CodexFallbackAction {
	if matches!(failure, CodexWebSocketFailure::Cancelled) {
		return CodexFallbackAction::Cancelled;
	}
	if safety.terminal || safety.delivered_tool_call {
		return CodexFallbackAction::Surface;
	}
	if matches!(failure, CodexWebSocketFailure::Provider) {
		return CodexFallbackAction::Surface;
	}
	if matches!(failure, CodexWebSocketFailure::RetryableProvider) {
		return if !safety.has_output && retries < retry_budget {
			CodexFallbackAction::ReconnectWebSocket
		} else {
			CodexFallbackAction::Surface
		};
	}
	if matches!(failure, CodexWebSocketFailure::ConnectionFatal) || safety.has_output {
		return CodexFallbackAction::ReplayOverHttp;
	}
	if retries < retry_budget {
		CodexFallbackAction::ReconnectWebSocket
	} else {
		CodexFallbackAction::ReplayOverHttp
	}
}

fn response_id(object: &Map<String, Value>) -> Option<&str> {
	object
		.get("response")
		.and_then(Value::as_object)
		.and_then(|response| response.get("id"))
		.and_then(Value::as_str)
}

fn is_terminal(kind: &str) -> bool {
	matches!(
		kind,
		"response.completed" | "response.done" | "response.incomplete" | "response.failed" | "error"
	)
}

fn strict_delta(previous: &Value, response_items: &[Value], current: &Value) -> Option<Vec<Value>> {
	let previous_object = previous.as_object()?;
	let current_object = current.as_object()?;
	if !objects_equal_without(previous_object, current_object, &["input", "client_metadata"]) {
		return None;
	}
	let previous_input = previous_object.get("input")?.as_array()?;
	let current_input = current_object.get("input")?.as_array()?;
	let baseline_len = previous_input.len().checked_add(response_items.len())?;
	if current_input.len() <= baseline_len {
		return None;
	}
	for (expected, actual) in previous_input
		.iter()
		.chain(response_items.iter())
		.zip(current_input.iter())
	{
		if !values_equal_without(expected, actual, &["status"]) {
			return None;
		}
	}
	Some(current_input[baseline_len..].to_vec())
}

fn objects_equal_without(
	left: &Map<String, Value>,
	right: &Map<String, Value>,
	omitted: &[&str],
) -> bool {
	left
		.iter()
		.filter(|(key, _)| !omitted.contains(&key.as_str()))
		.all(|(key, value)| right.get(key) == Some(value))
		&& right
			.iter()
			.filter(|(key, _)| !omitted.contains(&key.as_str()))
			.all(|(key, value)| left.get(key) == Some(value))
}

fn values_equal_without(left: &Value, right: &Value, omitted: &[&str]) -> bool {
	match (left.as_object(), right.as_object()) {
		(Some(left), Some(right)) => objects_equal_without(left, right, omitted),
		_ => left == right,
	}
}
