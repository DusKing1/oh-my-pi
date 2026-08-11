//! `ChatGPT` subscription Codex Responses codec and request fingerprint.
//!
//! Bearer credentials never enter this crate. The codec produces JSON and an
//! authorization-free header plan; the production egress/auth lease attaches
//! authorization at dispatch time.

use std::{collections::BTreeMap, fmt};

use bytes::Bytes;
use omp_core::Str;
use omp_llm_catalog::{
	TransportId,
	codex::{CODEX_CLIENT_VERSION, CODEX_ORIGINATOR},
	compat::Compat,
};
use omp_llm_transport::{DecodeState, Frame, Transport};
use omp_llm_types::{ChatRequest, Error, Unsupported};
use serde_json::Value;
use smallvec::SmallVec;
use zeroize::Zeroizing;

use crate::{
	model_policy::OpenAiModelPolicy,
	openai_codex_responses_lite::{
		CODEX_PROVIDER_NAMESPACE, RESPONSES_LITE_OPTION, transform_codex_request,
	},
	openai_responses::OpenAiResponsesCodec,
};

/// Resolves the `ChatGPT` Codex Responses endpoint from a configured base URL.
#[must_use]
pub fn resolve_codex_responses_url(base_url: &str) -> String {
	let normalized = base_url.trim().trim_end_matches('/');
	if normalized.ends_with("/codex/responses") {
		normalized.to_owned()
	} else if normalized.ends_with("/codex") {
		format!("{normalized}/responses")
	} else {
		format!("{normalized}/codex/responses")
	}
}

/// Codex wire transport selected for one attempt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CodexWireTransport {
	/// HTTP Server-Sent Events transport.
	Http,
	/// Responses WebSocket v2 transport.
	WebSocket,
}

/// Opaque `DeviceCheck` attestation envelope.
///
/// Debug output is always redacted. The egress layer may access the bytes only
/// while applying a just-in-time request header.
#[derive(Clone, Eq, PartialEq)]
pub struct CodexAttestation(Zeroizing<Vec<u8>>);

impl CodexAttestation {
	/// Accepts a non-empty complete attestation envelope.
	#[must_use]
	pub fn new(value: impl AsRef<[u8]>) -> Option<Self> {
		let value = value.as_ref();
		(!value.is_empty()).then(|| Self(Zeroizing::new(value.to_vec())))
	}

	/// Borrows the wire bytes for immediate insertion by egress.
	#[must_use]
	pub fn as_bytes(&self) -> &[u8] {
		&self.0
	}
}

impl fmt::Debug for CodexAttestation {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter.write_str("CodexAttestation([redacted])")
	}
}

/// One header value with sensitivity metadata preserved through egress.
///
/// The backing buffer zeroizes on drop, including for session and account
/// values which are not OAuth secrets but must not survive diagnostics.
#[derive(Clone, Eq, PartialEq)]
pub struct CodexHeaderValue {
	bytes:     Zeroizing<Vec<u8>>,
	sensitive: bool,
}

impl CodexHeaderValue {
	fn plain(value: impl Into<Bytes>) -> Self {
		let value = value.into();
		Self { bytes: Zeroizing::new(value.to_vec()), sensitive: false }
	}

	fn sensitive(value: impl Into<Bytes>) -> Self {
		let value = value.into();
		Self { bytes: Zeroizing::new(value.to_vec()), sensitive: true }
	}

	/// Borrows the header bytes for request construction.
	#[must_use]
	pub fn as_bytes(&self) -> &[u8] {
		&self.bytes
	}

	/// Returns whether diagnostics must redact this value.
	#[must_use]
	pub const fn is_sensitive(&self) -> bool {
		self.sensitive
	}
}

impl fmt::Debug for CodexHeaderValue {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		if self.sensitive {
			formatter.write_str("[redacted]")
		} else {
			write!(formatter, "{:?}", String::from_utf8_lossy(&self.bytes))
		}
	}
}

/// Authorization-free headers for one Codex attempt.
///
/// The plan contains subscription fingerprint and session headers, but never an
/// `authorization`, `cookie`, or API-key value.
#[derive(Clone, Default, Eq, PartialEq)]
pub struct CodexHeaderPlan(BTreeMap<Str, CodexHeaderValue>);

impl CodexHeaderPlan {
	/// Iterates deterministic header names and values for egress application.
	pub fn iter(&self) -> impl Iterator<Item = (&str, &CodexHeaderValue)> {
		self.0.iter().map(|(name, value)| (name.as_str(), value))
	}

	/// Returns one planned value using an ASCII-case-insensitive name lookup.
	#[must_use]
	pub fn get(&self, name: &str) -> Option<&CodexHeaderValue> {
		self
			.0
			.iter()
			.find_map(|(candidate, value)| candidate.eq_ignore_ascii_case(name).then_some(value))
	}

	fn insert_plain(&mut self, name: &'static str, value: impl Into<Bytes>) {
		self
			.0
			.insert(Str::new(name), CodexHeaderValue::plain(value));
	}

	fn insert_sensitive(&mut self, name: &'static str, value: impl Into<Bytes>) {
		self
			.0
			.insert(Str::new(name), CodexHeaderValue::sensitive(value));
	}
}

impl fmt::Debug for CodexHeaderPlan {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter.debug_map().entries(self.0.iter()).finish()
	}
}

/// Stable session and turn identity supplied by the application layer.
///
/// Identifiers are generated outside the codec so retrying an attempt never
/// rotates identity accidentally.
#[derive(Clone, Eq, PartialEq)]
pub struct CodexRequestIdentity {
	/// Installation id shared across sessions.
	pub installation_id: Str,
	/// Normalized conversation/session id.
	pub session_id:      Str,
	/// Stable thread id for the session.
	pub thread_id:       Str,
	/// Window id rotated by history compaction.
	pub window_id:       Str,
	/// Turn id retained across tool-result continuations.
	pub turn_id:         Str,
	/// Canonical ASCII JSON `x-codex-turn-metadata` value.
	pub turn_metadata:   Str,
}

impl fmt::Debug for CodexRequestIdentity {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter.write_str("CodexRequestIdentity([redacted])")
	}
}

/// Credential metadata released by the broker without releasing bearer bytes.
#[derive(Clone, Default, Eq, PartialEq)]
pub struct CodexCredentialMetadata {
	/// `ChatGPT` account/workspace id recovered from broker-held OAuth claims.
	pub account_id: Option<Str>,
}

impl fmt::Debug for CodexCredentialMetadata {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter
			.debug_struct("CodexCredentialMetadata")
			.field("account_id", &self.account_id.as_ref().map(|_| "[redacted]"))
			.finish()
	}
}

/// Inputs for constructing a Codex request fingerprint.
#[derive(Clone)]
pub struct CodexHeaderContext<'a> {
	/// Wire transport for this attempt.
	pub transport:      CodexWireTransport,
	/// Optional stable request identity.
	pub identity:       Option<&'a CodexRequestIdentity>,
	/// Broker-released non-secret credential metadata.
	pub credential:     &'a CodexCredentialMetadata,
	/// Just-in-time attestation; ignored for non-ChatGPT credentials.
	pub attestation:    Option<&'a CodexAttestation>,
	/// Per-turn continuation value learned from the server.
	pub turn_state:     Option<&'a str>,
	/// Models `ETag` learned from handshake or response metadata.
	pub models_etag:    Option<&'a str>,
	/// Whether this attempt uses Responses Lite.
	pub responses_lite: bool,
}

impl fmt::Debug for CodexHeaderContext<'_> {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter
			.debug_struct("CodexHeaderContext")
			.field("transport", &self.transport)
			.field("identity", &self.identity.map(|_| "[redacted]"))
			.field("credential", self.credential)
			.field("attestation", &self.attestation)
			.field("turn_state", &self.turn_state.map(|_| "[redacted]"))
			.field("models_etag", &self.models_etag.map(|_| "[redacted]"))
			.field("responses_lite", &self.responses_lite)
			.finish()
	}
}

/// Builds the authorization-free Codex request fingerprint for egress.
#[must_use]
pub fn build_codex_header_plan(context: &CodexHeaderContext<'_>) -> CodexHeaderPlan {
	let mut headers = CodexHeaderPlan::default();
	let beta = match context.transport {
		CodexWireTransport::Http => "responses=experimental",
		CodexWireTransport::WebSocket => "responses_websockets=2026-02-06",
	};
	headers.insert_plain("openai-beta", beta);
	headers.insert_plain("originator", CODEX_ORIGINATOR);
	headers.insert_plain("version", CODEX_CLIENT_VERSION);
	if matches!(context.transport, CodexWireTransport::Http) {
		headers.insert_plain("accept", "text/event-stream");
		headers.insert_plain("content-type", "application/json");
	}
	if let Some(identity) = context.identity {
		for (name, value) in [
			("conversation_id", identity.session_id.as_str()),
			("session_id", identity.session_id.as_str()),
			("x-client-request-id", identity.session_id.as_str()),
			("session-id", identity.session_id.as_str()),
			("thread-id", identity.thread_id.as_str()),
			("x-codex-window-id", identity.window_id.as_str()),
			("x-codex-turn-metadata", identity.turn_metadata.as_str()),
		] {
			headers.insert_sensitive(name, Bytes::copy_from_slice(value.as_bytes()));
		}
	}
	if let Some(account_id) = &context.credential.account_id {
		headers.insert_sensitive("chatgpt-account-id", Bytes::copy_from_slice(account_id.as_bytes()));
		if let Some(attestation) = context.attestation {
			headers
				.insert_sensitive("x-oai-attestation", Bytes::copy_from_slice(attestation.as_bytes()));
		}
	}
	if let Some(turn_state) = context.turn_state {
		headers.insert_sensitive("x-codex-turn-state", Bytes::copy_from_slice(turn_state.as_bytes()));
	}
	if let Some(models_etag) = context.models_etag {
		headers.insert_sensitive("x-models-etag", Bytes::copy_from_slice(models_etag.as_bytes()));
	}
	if context.responses_lite {
		headers.insert_plain("x-openai-internal-codex-responses-lite", "true");
	}
	headers
}

/// Applies canonical Codex identity to a request body's `client_metadata`.
///
/// Reserved identity fields always replace caller values. WebSocket-only
/// connection headers that may rotate between requests are mirrored into the
/// frame because an upgraded socket cannot carry new HTTP headers.
pub fn apply_codex_client_metadata(
	body: &mut Value,
	identity: &CodexRequestIdentity,
	transport: CodexWireTransport,
	responses_lite: bool,
	turn_state: Option<&str>,
) -> Result<(), Error> {
	let body = body
		.as_object_mut()
		.ok_or_else(|| Error::Provider(Str::new("Codex request body must be a JSON object")))?;
	let metadata = body
		.entry("client_metadata")
		.or_insert_with(|| Value::Object(Default::default()));
	if !metadata.is_object() {
		*metadata = Value::Object(Default::default());
	}
	let metadata = metadata
		.as_object_mut()
		.expect("client_metadata was normalized to an object");
	for (name, value) in [
		("x-codex-installation-id", identity.installation_id.as_str()),
		("session_id", identity.session_id.as_str()),
		("thread_id", identity.thread_id.as_str()),
		("x-codex-window-id", identity.window_id.as_str()),
		("turn_id", identity.turn_id.as_str()),
		("x-codex-turn-metadata", identity.turn_metadata.as_str()),
	] {
		metadata.insert(name.into(), Value::String(value.to_owned()));
	}
	if matches!(transport, CodexWireTransport::WebSocket) {
		if responses_lite {
			metadata.insert(
				"ws_request_header_x_openai_internal_codex_responses_lite".into(),
				Value::String("true".into()),
			);
		}
		if let Some(turn_state) = turn_state {
			metadata.insert("x-codex-turn-state".into(), Value::String(turn_state.to_owned()));
		}
	}
	Ok(())
}

/// `OpenAI` Responses codec with Codex subscription request transformation.
#[derive(Debug)]
pub struct OpenAiCodexCodec {
	responses:              OpenAiResponsesCodec,
	default_responses_lite: bool,
}

impl OpenAiCodexCodec {
	/// Creates a full Codex Responses codec.
	#[must_use]
	pub fn new() -> Self {
		Self { responses: OpenAiResponsesCodec::new(), default_responses_lite: false }
	}

	/// Creates a codec whose default request shape is Responses Lite.
	#[must_use]
	pub fn responses_lite() -> Self {
		Self { responses: OpenAiResponsesCodec::new(), default_responses_lite: true }
	}

	/// Resolves the Responses Lite marker for one request.
	///
	/// A malformed provider option is rejected before any egress attempt.
	pub fn request_uses_responses_lite(&self, req: &ChatRequest) -> Result<bool, Error> {
		match req
			.provider_options
			.as_ref()
			.and_then(|props| props.get_ns(CODEX_PROVIDER_NAMESPACE, RESPONSES_LITE_OPTION))
		{
			Some(Value::Bool(value)) => Ok(*value),
			Some(_) => Err(Error::Provider(Str::new("openai-codex/responses_lite must be a boolean"))),
			None => Ok(req
				.model_policy
				.as_deref()
				.and_then(|policy| policy.use_responses_lite)
				.unwrap_or(self.default_responses_lite)),
		}
	}

	/// Encodes one request with an already-resolved server-side Responses Lite
	/// policy. This is the typed application point for catalog model behavior;
	/// callers do not need to synthesize a provider option.
	pub fn encode_with_responses_lite(
		&self,
		req: &ChatRequest,
		compat: &Compat,
		responses_lite: bool,
	) -> Result<(Bytes, Vec<Unsupported>), Error> {
		let mut delegated = req.clone();
		if let Some(options) = delegated.provider_options.as_mut() {
			options.0.remove("openai-codex/responses_lite");
		}
		let (body, mut unsupported) = self.responses.encode(&delegated, compat)?;
		let mut body: Value = serde_json::from_slice(&body)
			.map_err(|error| Error::Provider(Str::from(error.to_string())))?;
		unsupported.extend(transform_codex_request(&mut body, responses_lite)?);
		let mut ignored = Vec::new();
		let policy = OpenAiModelPolicy::resolve(req, compat, &mut ignored);
		let explicit_encrypted_reasoning = req
			.provider_options
			.as_ref()
			.and_then(|props| props.get_ns("openai", "include"))
			.and_then(Value::as_array)
			.is_some_and(|include| {
				include
					.iter()
					.any(|value| value.as_str() == Some("reasoning.encrypted_content"))
			});
		if let Some(object) = body.as_object_mut() {
			if !policy.include_encrypted_reasoning
				&& !explicit_encrypted_reasoning
				&& let Some(include) = object.get_mut("include").and_then(Value::as_array_mut)
			{
				include.retain(|value| value.as_str() != Some("reasoning.encrypted_content"));
				if include.is_empty() {
					object.remove("include");
				}
			}
			if !policy.supports_store {
				object.remove("store");
			}
		}
		serde_json::to_vec(&body)
			.map(|body| (Bytes::from(body), unsupported))
			.map_err(|error| Error::Provider(Str::from(error.to_string())))
	}
}

impl Default for OpenAiCodexCodec {
	fn default() -> Self {
		Self::new()
	}
}

impl Transport for OpenAiCodexCodec {
	fn id(&self) -> TransportId {
		TransportId::OpenAiCodex
	}

	fn encode(
		&self,
		req: &ChatRequest,
		compat: &Compat,
	) -> Result<(Bytes, Vec<Unsupported>), Error> {
		let responses_lite = self.request_uses_responses_lite(req)?;
		self.encode_with_responses_lite(req, compat, responses_lite)
	}

	fn decode(
		&self,
		frame: Frame<'_>,
		state: &mut DecodeState,
	) -> Result<SmallVec<omp_llm_types::TurnEvent, 2>, Error> {
		self.responses.decode(frame, state)
	}
}

#[cfg(test)]
mod tests {
	use std::sync::Arc;

	use omp_core::Str;
	use omp_llm_types::{ChatRequest, Props, ResolvedModelPolicy, Thread};
	use serde_json::json;

	use super::{OpenAiCodexCodec, RESPONSES_LITE_OPTION};
	use crate::openai_codex_responses_lite::CODEX_PROVIDER_NAMESPACE;

	fn request() -> ChatRequest {
		ChatRequest::builder()
			.model(Str::new_static("gpt-5.6-sol"))
			.thread(Thread::builder().items(Vec::new()).build())
			.tools(Vec::new())
			.build()
	}

	#[test]
	fn model_policy_selects_lite_but_explicit_option_keeps_precedence() {
		let codec = OpenAiCodexCodec::new();
		let mut req = request();
		let mut policy = ResolvedModelPolicy::default();
		policy.use_responses_lite = Some(true);
		req.model_policy = Some(Arc::new(policy));
		assert!(codec.request_uses_responses_lite(&req).unwrap());

		let mut options = Props::default();
		options.insert_ns(CODEX_PROVIDER_NAMESPACE, RESPONSES_LITE_OPTION, json!(false));
		req.provider_options = Some(options);
		assert!(!codec.request_uses_responses_lite(&req).unwrap());
	}
}
