//! Anthropic Messages over Amazon Bedrock Runtime.
//!
//! This module owns Bedrock's model route, cloud body projection, AWS
//! EventStream deframing, and non-secret SigV4 request context. Secret AWS
//! material never enters the codec: the shared egress credential source redeems
//! the request's lease and signs the fully buffered request in place.

use std::time::SystemTime;

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use bytes::{Buf, Bytes, BytesMut};
use omp_core::{Str, StrMut};
use omp_llm_catalog::{compat::Compat, provider::TransportId};
use omp_llm_egress::auth_inject::AwsSigV4Context;
use omp_llm_transport::{DecodeState, Frame, Transport};
use omp_llm_types::{ChatRequest, Error, TurnEvent, Unsupported};
use serde::Deserialize;
use smallvec::SmallVec;

use crate::{
	AnthropicCodec,
	compat::{CloudMessages, cloud_betas, project_cloud_request},
};

/// SigV4 service name for Amazon Bedrock Runtime requests.
pub const SIGV4_SERVICE: &str = "bedrock";
/// Bedrock response media type carrying AWS EventStream messages.
pub const EVENTSTREAM_CONTENT_TYPE: &str = "application/vnd.amazon.eventstream";
/// Bedrock model response payload type nested inside EventStream chunks.
pub const MODEL_CONTENT_TYPE: &str = "application/json";

const MAX_EVENTSTREAM_MESSAGE_LEN: usize = 16 * 1024 * 1024;
const MAX_EVENTSTREAM_HEADERS_LEN: usize = 128 * 1024;

/// Codec for Anthropic Messages hosted by Amazon Bedrock Runtime.
#[derive(Debug, Default)]
pub struct BedrockCodec {
	inner: AnthropicCodec,
}

impl BedrockCodec {
	/// Constructs a stateless Anthropic-on-Bedrock codec.
	#[must_use]
	pub const fn new() -> Self {
		Self { inner: AnthropicCodec::new() }
	}
}

impl Transport for BedrockCodec {
	fn id(&self) -> TransportId {
		TransportId::AnthropicBedrock
	}

	fn encode(
		&self,
		req: &ChatRequest,
		compat: &Compat,
	) -> Result<(Bytes, Vec<Unsupported>), Error> {
		let (body, mut unsupported) = self.inner.encode(req, compat)?;
		let betas = cloud_betas(req, compat, "amazon-bedrock")?;
		unsupported.retain(|item| item.what != "amazon-bedrock/betas");
		project_cloud_request(&body, CloudMessages::Bedrock, &betas).map(|body| (body, unsupported))
	}

	fn decode(
		&self,
		frame: Frame<'_>,
		state: &mut DecodeState,
	) -> Result<SmallVec<TurnEvent, 2>, Error> {
		let mut events = self.inner.decode(frame, state)?;
		for event in &mut events {
			if let TurnEvent::Outcome(outcome) = event {
				outcome.provider = Str::new_static("amazon-bedrock");
			}
		}
		Ok(events)
	}
}

/// Resolves the signing region for a Bedrock model.
///
/// Inference-profile ARNs are authoritative because their partition and region
/// identify the resource being signed. Geo-prefixed cross-region profiles use
/// a concrete route only when that route can serve the profile; otherwise they
/// select the geo's stable default. An explicit route is next, followed by a
/// standard Bedrock Runtime endpoint host and the SDK-compatible `us-east-1`
/// fallback.
#[must_use]
pub fn resolve_region(explicit: &str, model: &str, base_url: &str) -> Str {
	if let Some(region) = arn_region(model) {
		return Str::new(region);
	}
	if let Some((geo, fallback)) = inference_profile_geo(model) {
		if !explicit.is_empty() && region_serves_geo(explicit, geo) {
			return Str::new(explicit);
		}
		return Str::new_static(fallback);
	}
	if !explicit.is_empty() {
		return Str::new(explicit);
	}
	if let Some(region) = endpoint_region(base_url) {
		return Str::new(region);
	}
	Str::new_static("us-east-1")
}

/// Builds a Bedrock `InvokeModelWithResponseStream` endpoint.
///
/// An empty `base_url` selects the standard regional Bedrock Runtime host.
/// Custom endpoints are retained verbatim apart from a trailing slash. Model
/// identifiers, including inference-profile ARNs, are encoded as one path
/// segment so embedded `/` cannot alter the invocation route.
pub fn endpoint(base_url: &str, region: &str, model: &str) -> Result<Str, Error> {
	endpoint_for(base_url, region, model, "invoke-with-response-stream")
}
/// Builds a model-independent Bedrock `ConverseStream` endpoint.
pub fn converse_endpoint(base_url: &str, region: &str, model: &str) -> Result<Str, Error> {
	endpoint_for(base_url, region, model, "converse-stream")
}
fn endpoint_for(
	base_url: &str,
	region: &str,
	model: &str,
	operation: &str,
) -> Result<Str, Error> {
	if model.is_empty() {
		return Err(provider_error("Bedrock model must not be empty"));
	}
	let region = resolve_region(region, model, base_url);
	if !region
		.bytes()
		.all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
	{
		return Err(provider_error("Bedrock region contains invalid hostname characters"));
	}
	let mut target = if base_url.is_empty() {
		let mut target = StrMut::new("https://bedrock-runtime.");
		target.push_str(&region);
		target.push_str(if region.starts_with("cn-") {
			".amazonaws.com.cn"
		} else {
			".amazonaws.com"
		});
		target
	} else {
		StrMut::new(base_url.trim_end_matches('/'))
	};
	target.push_str("/model/");
	push_path_segment(&mut target, model);
	target.push('/');
	target.push_str(operation);
	Ok(target.freeze())
}

fn arn_region(model: &str) -> Option<&str> {
	let mut fields = model.split(':');
	(fields.next()? == "arn").then_some(())?;
	let partition = fields.next()?;
	if !partition.starts_with("aws") || fields.next()? != "bedrock" {
		return None;
	}
	let region = fields.next()?;
	(!region.is_empty()).then_some(region)
}

fn inference_profile_geo(model: &str) -> Option<(&str, &'static str)> {
	let (prefix, _) = model.split_once('.')?;
	match prefix {
		"us" => Some(("us", "us-east-1")),
		"us-gov" => Some(("us-gov", "us-gov-west-1")),
		"eu" => Some(("eu", "eu-west-1")),
		"apac" => Some(("apac", "ap-southeast-1")),
		"au" => Some(("au", "ap-southeast-2")),
		"jp" => Some(("jp", "ap-northeast-1")),
		_ => None,
	}
}

fn region_serves_geo(region: &str, geo: &str) -> bool {
	match geo {
		"us-gov" => region.starts_with("us-gov-"),
		"us" => region.starts_with("us-") && !region.starts_with("us-gov-"),
		"eu" => region.starts_with("eu-"),
		"apac" => region.starts_with("ap-"),
		"au" => matches!(region, "ap-southeast-2" | "ap-southeast-4"),
		"jp" => matches!(region, "ap-northeast-1" | "ap-northeast-3"),
		_ => false,
	}
}

fn endpoint_region(base_url: &str) -> Option<&str> {
	let host = base_url
		.strip_prefix("https://")
		.or_else(|| base_url.strip_prefix("http://"))?
		.split(['/', ':'])
		.next()?;
	let region = host
		.strip_prefix("bedrock-runtime.")
		.or_else(|| host.strip_prefix("bedrock-runtime-fips."))?
		.strip_suffix(".amazonaws.com")
		.or_else(|| {
			host
				.strip_prefix("bedrock-runtime.")
				.or_else(|| host.strip_prefix("bedrock-runtime-fips."))?
				.strip_suffix(".amazonaws.com.cn")
		})?;
	(!region.is_empty()).then_some(region)
}

/// Adds Bedrock media headers and non-secret SigV4 metadata to a request.
///
/// This function does not sign and cannot access credentials. The production
/// auth layer observes [`AwsSigV4Context`] after routing has attached a
/// [`omp_llm_egress::auth_inject::CredentialLease`], then signs the final URI,
/// headers, and buffered body immediately before dispatch.
pub fn attach_sigv4<B>(
	request: &mut http::Request<B>,
	region: impl Into<Str>,
	signed_at: SystemTime,
) {
	request.headers_mut().remove("anthropic-beta");
	request.headers_mut().remove("anthropic-version");
	request
		.headers_mut()
		.insert(http::header::CONTENT_TYPE, http::HeaderValue::from_static(MODEL_CONTENT_TYPE));
	request
		.headers_mut()
		.insert(http::header::ACCEPT, http::HeaderValue::from_static(EVENTSTREAM_CONTENT_TYPE));
	request.headers_mut().insert(
		http::HeaderName::from_static("x-amzn-bedrock-accept"),
		http::HeaderValue::from_static(MODEL_CONTENT_TYPE),
	);
	request.extensions_mut().insert(AwsSigV4Context {
		service: SIGV4_SERVICE.into(),
		region: region.into(),
		signed_at,
	});
}

/// Incremental decoder for AWS EventStream response framing.
///
/// Every returned byte string is one model JSON event. Bedrock exception and
/// error messages are converted to Anthropic-shaped typed error events so each
/// codec receives the same canonical terminal error classes. Once a terminal
/// event is observed, later wire messages are ignored.
#[derive(Debug, Default)]
pub struct BedrockEventStreamDecoder {
	buffer:   BytesMut,
	terminal: bool,
}

impl BedrockEventStreamDecoder {
	/// Constructs an empty incremental decoder.
	#[must_use]
	pub fn new() -> Self {
		Self { buffer: BytesMut::new(), terminal: false }
	}

	/// Appends response bytes and returns every complete model JSON event.
	pub fn push(&mut self, chunk: Bytes) -> Result<SmallVec<Bytes, 4>, Error> {
		if self.terminal {
			return Ok(SmallVec::new());
		}
		self.buffer.extend_from_slice(&chunk);
		let mut output = SmallVec::new();
		loop {
			if self.buffer.len() < 12 {
				break;
			}
			let total_len = usize::try_from(u32::from_be_bytes(
				self.buffer[..4].try_into().expect("four-byte prefix"),
			))
			.expect("u32 fits usize");
			let headers_len = usize::try_from(u32::from_be_bytes(
				self.buffer[4..8].try_into().expect("four-byte prefix"),
			))
			.expect("u32 fits usize");
			if total_len < 16
				|| total_len > MAX_EVENTSTREAM_MESSAGE_LEN
				|| headers_len > MAX_EVENTSTREAM_HEADERS_LEN
				|| headers_len > total_len - 16
			{
				return Err(provider_error("invalid AWS EventStream message lengths"));
			}
			if self.buffer.len() < total_len {
				break;
			}
			let message = self.buffer.split_to(total_len).freeze();
			validate_crc(&message)?;
			let headers_end = 12 + headers_len;
			let headers = parse_headers(&message[12..headers_end])?;
			let payload = &message[headers_end..total_len - 4];
			if let Some(event) = decode_message(headers, payload)? {
				self.terminal = is_terminal_event(&event);
				output.push(event);
				if self.terminal {
					self.buffer.clear();
					break;
				}
			}
		}
		Ok(output)
	}

	/// Validates that the response ended on an EventStream frame boundary.
	pub fn finish(&self) -> Result<(), Error> {
		if self.buffer.is_empty() || self.terminal {
			Ok(())
		} else {
			Err(provider_error("Bedrock response ended inside an AWS EventStream message"))
		}
	}

	/// Returns whether a message-stop or exception terminal was decoded.
	#[must_use]
	pub const fn is_terminal(&self) -> bool {
		self.terminal
	}
}

#[derive(Default)]
struct EventHeaders<'a> {
	message_type:   &'a str,
	event_type:     &'a str,
	exception_type: &'a str,
	error_code:     &'a str,
	error_message:  &'a str,
}

fn validate_crc(message: &[u8]) -> Result<(), Error> {
	let expected_prelude = u32::from_be_bytes(message[8..12].try_into().expect("prelude CRC"));
	let actual_prelude = crc32fast::hash(&message[..8]);
	if expected_prelude != actual_prelude {
		return Err(provider_error("AWS EventStream prelude CRC mismatch"));
	}
	let crc_offset = message.len() - 4;
	let expected_message =
		u32::from_be_bytes(message[crc_offset..].try_into().expect("message CRC"));
	let actual_message = crc32fast::hash(&message[..crc_offset]);
	if expected_message != actual_message {
		return Err(provider_error("AWS EventStream message CRC mismatch"));
	}
	Ok(())
}

fn parse_headers(mut bytes: &[u8]) -> Result<EventHeaders<'_>, Error> {
	let mut result = EventHeaders::default();
	while !bytes.is_empty() {
		let name_len = usize::from(take_u8(&mut bytes)?);
		let name = take(&mut bytes, name_len)?;
		let kind = take_u8(&mut bytes)?;
		let string = match kind {
			0 | 1 => None,
			2 => {
				take(&mut bytes, 1)?;
				None
			},
			3 => {
				take(&mut bytes, 2)?;
				None
			},
			4 => {
				take(&mut bytes, 4)?;
				None
			},
			5 | 8 => {
				take(&mut bytes, 8)?;
				None
			},
			6 => {
				let len = usize::from(take_u16(&mut bytes)?);
				take(&mut bytes, len)?;
				None
			},
			7 => {
				let len = usize::from(take_u16(&mut bytes)?);
				Some(
					std::str::from_utf8(take(&mut bytes, len)?)
						.map_err(|_| provider_error("AWS EventStream string header is not UTF-8"))?,
				)
			},
			9 => {
				take(&mut bytes, 16)?;
				None
			},
			_ => return Err(provider_error("unknown AWS EventStream header value type")),
		};
		let Some(value) = string else { continue };
		match name {
			b":message-type" => result.message_type = value,
			b":event-type" => result.event_type = value,
			b":exception-type" => result.exception_type = value,
			b":error-code" => result.error_code = value,
			b":error-message" => result.error_message = value,
			_ => {},
		}
	}
	Ok(result)
}

fn decode_message(headers: EventHeaders<'_>, payload: &[u8]) -> Result<Option<Bytes>, Error> {
	if headers.message_type == "exception" || !headers.exception_type.is_empty() {
		return exception_event(headers.exception_type, payload).map(Some);
	}
	if headers.message_type == "error" {
		let synthesized;
		let payload = if payload.is_empty() && !headers.error_message.is_empty() {
			synthesized = serde_json::to_vec(&serde_json::json!({
				"message": headers.error_message
			}))
			.map_err(json_error)?;
			synthesized.as_slice()
		} else {
			payload
		};
		return exception_event(headers.error_code, payload).map(Some);
	}
	if headers.message_type != "event" || headers.event_type.is_empty() {
		return Err(provider_error("unsupported AWS EventStream message or event type"));
	}
	if headers.event_type != "chunk" {
		return if serde_json::from_slice::<serde_json::Value>(payload).is_ok() {
			Ok(Some(Bytes::copy_from_slice(payload)))
		} else {
			Err(provider_error("Bedrock event payload is not JSON"))
		};
	}
	#[derive(Deserialize)]
	struct Chunk<'a> {
		#[serde(borrow)]
		bytes: Option<&'a str>,
	}
	if let Ok(chunk) = serde_json::from_slice::<Chunk<'_>>(payload)
		&& let Some(encoded) = chunk.bytes
	{
		return BASE64
			.decode(encoded)
			.map(Bytes::from)
			.map(Some)
			.map_err(|_| provider_error("Bedrock chunk contains invalid base64"));
	}
	if serde_json::from_slice::<serde_json::Value>(payload).is_ok() {
		return Ok(Some(Bytes::copy_from_slice(payload)));
	}
	Err(provider_error("Bedrock chunk payload is not JSON"))
}

fn exception_event(kind: &str, payload: &[u8]) -> Result<Bytes, Error> {
	#[derive(Deserialize)]
	struct Exception<'a> {
		#[serde(default, borrow, alias = "Message")]
		message:          &'a str,
		#[serde(default, borrow, rename = "originalMessage")]
		original_message: &'a str,
	}
	let exception = serde_json::from_slice::<Exception<'_>>(payload)
		.unwrap_or(Exception { message: "", original_message: "" });
	let message = if !exception.message.is_empty() {
		exception.message
	} else if !exception.original_message.is_empty() {
		exception.original_message
	} else if payload.is_empty() {
		"Bedrock stream exception"
	} else {
		std::str::from_utf8(payload).unwrap_or("Bedrock stream exception")
	};
	let anthropic_kind = match kind {
		"accessDeniedException" | "AccessDeniedException" | "notAuthorized" => "authentication_error",
		"throttlingException" | "ThrottlingException" | "modelTimeoutException" => "rate_limit_error",
		"serviceUnavailableException"
		| "ServiceUnavailableException"
		| "internalServerException"
		| "InternalServerException"
		| "modelStreamErrorException"
		| "ModelStreamErrorException" => "overloaded_error",
		_ => "api_error",
	};
	serde_json::to_vec(&serde_json::json!({
		"type": "error",
		"error": { "type": anthropic_kind, "message": message }
	}))
	.map(Bytes::from)
	.map_err(json_error)
}

fn is_terminal_event(event: &[u8]) -> bool {
	#[derive(Deserialize)]
	struct Tag<'a> {
		#[serde(default, rename = "type")]
		kind: &'a str,
	}
	serde_json::from_slice::<Tag<'_>>(event)
		.is_ok_and(|tag| matches!(tag.kind, "message_stop" | "error"))
}

fn take_u8(input: &mut &[u8]) -> Result<u8, Error> {
	if input.is_empty() {
		return Err(provider_error("truncated AWS EventStream header"));
	}
	Ok(input.get_u8())
}

fn take_u16(input: &mut &[u8]) -> Result<u16, Error> {
	if input.len() < 2 {
		return Err(provider_error("truncated AWS EventStream header"));
	}
	Ok(input.get_u16())
}

fn take<'a>(input: &mut &'a [u8], len: usize) -> Result<&'a [u8], Error> {
	if input.len() < len {
		return Err(provider_error("truncated AWS EventStream header"));
	}
	let (value, rest) = input.split_at(len);
	*input = rest;
	Ok(value)
}

fn push_path_segment(target: &mut StrMut, value: &str) {
	const HEX: &[u8; 16] = b"0123456789ABCDEF";
	for byte in value.bytes() {
		if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
			target.push(char::from(byte));
		} else {
			target.push('%');
			target.push(char::from(HEX[usize::from(byte >> 4)]));
			target.push(char::from(HEX[usize::from(byte & 0x0f)]));
		}
	}
}

fn json_error(error: serde_json::Error) -> Error {
	provider_error(error.to_string())
}

fn provider_error(detail: impl Into<Str>) -> Error {
	Error::Provider(detail.into())
}

#[cfg(test)]
mod tests {
	use super::*;

	fn string_header(output: &mut Vec<u8>, name: &str, value: &str) {
		output.push(u8::try_from(name.len()).expect("short header name"));
		output.extend_from_slice(name.as_bytes());
		output.push(7);
		output.extend_from_slice(
			&u16::try_from(value.len())
				.expect("short value")
				.to_be_bytes(),
		);
		output.extend_from_slice(value.as_bytes());
	}

	fn framed(headers: &[u8], payload: &[u8]) -> Bytes {
		let total_len = 16 + headers.len() + payload.len();
		let mut message = Vec::with_capacity(total_len);
		message.extend_from_slice(
			&u32::try_from(total_len)
				.expect("fixture length")
				.to_be_bytes(),
		);
		message.extend_from_slice(
			&u32::try_from(headers.len())
				.expect("header length")
				.to_be_bytes(),
		);
		message.extend_from_slice(&crc32fast::hash(&message).to_be_bytes());
		message.extend_from_slice(headers);
		message.extend_from_slice(payload);
		message.extend_from_slice(&crc32fast::hash(&message).to_be_bytes());
		Bytes::from(message)
	}

	fn eventstream(event_type: &str, payload: &[u8]) -> Bytes {
		let mut headers = Vec::new();
		string_header(&mut headers, ":message-type", "event");
		string_header(&mut headers, ":event-type", event_type);
		string_header(&mut headers, ":content-type", "application/json");
		framed(&headers, payload)
	}

	fn exceptionstream(exception_type: &str, payload: &[u8]) -> Bytes {
		let mut headers = Vec::new();
		string_header(&mut headers, ":message-type", "exception");
		string_header(&mut headers, ":exception-type", exception_type);
		string_header(&mut headers, ":content-type", "application/json");
		framed(&headers, payload)
	}

	fn errorstream(code: &str, message: &str) -> Bytes {
		let mut headers = Vec::new();
		string_header(&mut headers, ":message-type", "error");
		string_header(&mut headers, ":error-code", code);
		string_header(&mut headers, ":error-message", message);
		framed(&headers, &[])
	}

	#[test]
	fn decodes_fragmented_eventstream_and_stops_after_terminal() {
		let inner = br#"{"type":"message_stop"}"#;
		let payload = serde_json::to_vec(&serde_json::json!({ "bytes": BASE64.encode(inner) }))
			.expect("fixture JSON");
		let frame = eventstream("chunk", &payload);
		let split = frame.len() / 2;
		let mut decoder = BedrockEventStreamDecoder::new();
		assert!(
			decoder
				.push(frame.slice(..split))
				.expect("first fragment")
				.is_empty()
		);
		let events = decoder.push(frame.slice(split..)).expect("second fragment");
		assert_eq!(events.as_slice(), &[Bytes::from_static(inner)]);
		assert!(decoder.is_terminal());
		assert!(
			decoder
				.push(frame)
				.expect("ignored after terminal")
				.is_empty()
		);
	}

	#[test]
	fn decodes_raw_converse_event_payload() {
		let payload = Bytes::from_static(br#"{"role":"assistant"}"#);
		let events = BedrockEventStreamDecoder::new()
			.push(eventstream("messageStart", &payload))
			.expect("Converse event");
		assert_eq!(events.as_slice(), &[payload]);
	}
	#[test]
	fn classifies_stream_exception_once() {
		let frame =
			exceptionstream("serviceUnavailableException", br#"{"message":"capacity unavailable"}"#);
		let mut decoder = BedrockEventStreamDecoder::new();
		let events = decoder.push(frame.clone()).expect("exception frame");
		assert_eq!(events.len(), 1);
		let value: serde_json::Value =
			serde_json::from_slice(&events[0]).expect("canonical error JSON");
		assert_eq!(value["type"], "error");
		assert_eq!(value["error"]["type"], "overloaded_error");
		assert_eq!(value["error"]["message"], "capacity unavailable");
		assert!(decoder.is_terminal());
		assert!(decoder.push(frame).expect("post-terminal frame").is_empty());

		let mut decoder = BedrockEventStreamDecoder::new();
		let events = decoder
			.push(errorstream("throttlingException", "retry later"))
			.expect("error frame");
		let value: serde_json::Value =
			serde_json::from_slice(&events[0]).expect("canonical error JSON");
		assert_eq!(value["error"]["type"], "rate_limit_error");
		assert_eq!(value["error"]["message"], "retry later");
	}

	#[test]
	fn endpoint_and_sigv4_context_are_request_metadata_only() {
		let uri = endpoint("", "us-east-1", "anthropic.claude-v2:1").expect("endpoint");
		assert_eq!(
			uri,
			"https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-v2%3A1/invoke-with-response-stream"
		);
		let signed_at = SystemTime::UNIX_EPOCH;
		let mut request = http::Request::post(uri.as_str())
			.header("anthropic-beta", "pdfs-2024-09-25")
			.header("anthropic-version", "2023-06-01")
			.body(())
			.expect("request");
		attach_sigv4(&mut request, "us-east-1", signed_at);
		let context = request
			.extensions()
			.get::<AwsSigV4Context>()
			.expect("SigV4 context");
		assert_eq!(context.service, SIGV4_SERVICE);
		assert_eq!(context.region, "us-east-1");
		assert_eq!(context.signed_at, signed_at);
		assert!(request.headers().get(http::header::AUTHORIZATION).is_none());
		assert!(request.headers().get("anthropic-beta").is_none());
		assert!(request.headers().get("anthropic-version").is_none());
	}

	#[test]
	fn converse_endpoint_geo_routing_and_crc_failure_are_shared() {
		assert_eq!(
			resolve_region("us-east-1", "eu.anthropic.claude-sonnet-4-6-v1:0", ""),
			"eu-west-1"
		);
		assert_eq!(
			resolve_region("eu-central-1", "eu.anthropic.claude-sonnet-4-6-v1:0", ""),
			"eu-central-1"
		);
		assert_eq!(
			resolve_region(
				"us-east-1",
				"arn:aws:bedrock:ap-southeast-2:123456789012:inference-profile/test",
				"",
			),
			"ap-southeast-2"
		);
		assert_eq!(
			converse_endpoint("", "eu-west-1", "eu.anthropic.claude:1").expect("endpoint"),
			"https://bedrock-runtime.eu-west-1.amazonaws.com/model/eu.anthropic.claude%3A1/converse-stream"
		);

		let mut corrupt = eventstream("contentBlockDelta", br#"{"contentBlockIndex":0}"#).to_vec();
		let last = corrupt.len() - 1;
		corrupt[last] ^= 1;
		let error = BedrockEventStreamDecoder::new()
			.push(Bytes::from(corrupt))
			.expect_err("CRC mismatch");
		assert!(error.to_string().contains("CRC mismatch"));
	}
}
