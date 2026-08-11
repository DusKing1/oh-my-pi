//! Anthropic Messages over Google Vertex AI.
//!
//! The codec keeps Claude's Messages schema and SSE events while projecting
//! cloud-only body fields and Vertex publisher-model routes. Authentication is
//! deliberately represented by the ordinary egress [`AuthContext`]; ADC token
//! resolution and sealed bearer injection happen at the credential boundary.

use bytes::Bytes;
use omp_core::{Str, StrMut};
use omp_llm_catalog::{compat::Compat, provider::TransportId};
use omp_llm_egress::auth_inject::AuthContext;
use omp_llm_transport::{DecodeState, Frame, Transport};
use omp_llm_types::{ChatRequest, Error, TurnError, TurnErrorKind, TurnEvent, Unsupported};
use serde::Deserialize;
use smallvec::SmallVec;

use crate::{
	AnthropicCodec,
	compat::{CloudMessages, cloud_betas, project_cloud_request},
};

/// OAuth scope requested by the shared Google ADC credential source.
pub const ADC_SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";

/// Codec for Anthropic Messages hosted by Vertex AI.
#[derive(Debug, Default)]
pub struct VertexCodec {
	inner: AnthropicCodec,
}

impl VertexCodec {
	/// Constructs a stateless Anthropic-on-Vertex codec.
	#[must_use]
	pub const fn new() -> Self {
		Self { inner: AnthropicCodec::new() }
	}
}

impl Transport for VertexCodec {
	fn id(&self) -> TransportId {
		TransportId::AnthropicVertex
	}

	fn encode(
		&self,
		req: &ChatRequest,
		compat: &Compat,
	) -> Result<(Bytes, Vec<Unsupported>), Error> {
		let (body, mut unsupported) = self.inner.encode(req, compat)?;
		let betas = cloud_betas(req, compat, "anthropic-vertex")?;
		unsupported.retain(|item| item.what != "anthropic-vertex/betas");
		project_cloud_request(&body, CloudMessages::Vertex, &betas).map(|body| (body, unsupported))
	}

	fn decode(
		&self,
		frame: Frame<'_>,
		state: &mut DecodeState,
	) -> Result<SmallVec<TurnEvent, 2>, Error> {
		let mut events = self.inner.decode(frame, state)?;
		for event in &mut events {
			if let TurnEvent::Outcome(outcome) = event {
				outcome.provider = Str::new_static("anthropic-vertex");
			}
		}
		Ok(events)
	}
}

/// Builds a Vertex Anthropic `streamRawPredict` endpoint.
///
/// `base_url` may be empty, in which case a regional location uses its
/// location-prefixed host and `global` uses `aiplatform.googleapis.com`.
pub fn endpoint(base_url: &str, project: &str, location: &str, model: &str) -> Result<Str, Error> {
	for (name, value) in [("project", project), ("location", location), ("model", model)] {
		if value.is_empty() {
			return Err(provider_error(format!("Vertex Anthropic {name} must not be empty")));
		}
	}
	if base_url.is_empty() {
		return regional_endpoint(project, location, model);
	}
	let mut target = StrMut::new(base_url.trim_end_matches('/'));
	append_path(&mut target, project, location, model);
	Ok(target.freeze())
}

fn regional_endpoint(project: &str, location: &str, model: &str) -> Result<Str, Error> {
	if !location
		.bytes()
		.all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
	{
		return Err(provider_error("Vertex location contains invalid hostname characters"));
	}
	let mut target = if location == "global" {
		StrMut::new("https://aiplatform.googleapis.com")
	} else {
		let mut target = StrMut::new("https://");
		target.push_str(location);
		target.push_str("-aiplatform.googleapis.com");
		target
	};
	append_path(&mut target, project, location, model);
	Ok(target.freeze())
}

fn append_path(target: &mut StrMut, project: &str, location: &str, model: &str) {
	target.push_str("/v1/projects/");
	push_path_segment(target, project);
	target.push_str("/locations/");
	push_path_segment(target, location);
	target.push_str("/publishers/anthropic/models/");
	push_path_segment(target, model);
	target.push_str(":streamRawPredict");
}

/// Attaches the non-secret ADC redemption context to a prepared HTTP request.
///
/// The selected [`omp_llm_egress::auth_inject::CredentialLease`] remains on the
/// request. The production auth layer resolves ADC and injects the bearer token
/// immediately before dispatch, without returning token bytes to this crate.
/// Direct Anthropic protocol headers are removed defensively because Vertex
/// accepts the protocol version and betas only in the JSON body.
pub fn attach_adc<B>(request: &mut http::Request<B>, provider: &str) {
	request.headers_mut().remove("anthropic-beta");
	request.headers_mut().remove("anthropic-version");
	request.extensions_mut().insert(AuthContext::new(provider));
}

/// Converts a non-success Vertex response into the canonical terminal error.
///
/// Google RPC status names take precedence over numeric status when present.
/// Safety-policy rejections retain their provider detail rather than being
/// mistaken for authentication or retryable capacity failures.
#[must_use]
pub fn classify_error(status: http::StatusCode, body: &[u8]) -> TurnError {
	#[derive(Deserialize)]
	struct Envelope<'a> {
		#[serde(borrow)]
		error: Option<GoogleError<'a>>,
	}
	#[derive(Deserialize)]
	struct GoogleError<'a> {
		#[serde(default)]
		status:  &'a str,
		#[serde(default)]
		message: &'a str,
	}

	let parsed = serde_json::from_slice::<Envelope<'_>>(body)
		.ok()
		.and_then(|value| value.error);
	let rpc_status = parsed.as_ref().map_or("", |error| error.status);
	let message = parsed
		.as_ref()
		.map(|error| error.message)
		.filter(|message| !message.is_empty())
		.unwrap_or_else(|| std::str::from_utf8(body).unwrap_or("Vertex request failed"));
	let safety = contains_ascii_case_insensitive(message, b"safety")
		|| contains_ascii_case_insensitive(message, b"content policy");
	let kind = if safety {
		TurnErrorKind::Upstream
	} else {
		match rpc_status {
			"UNAUTHENTICATED" | "PERMISSION_DENIED" => TurnErrorKind::Auth,
			"RESOURCE_EXHAUSTED" => TurnErrorKind::RateLimited,
			"UNAVAILABLE" => TurnErrorKind::Overloaded,
			_ if status == http::StatusCode::UNAUTHORIZED || status == http::StatusCode::FORBIDDEN => {
				TurnErrorKind::Auth
			},
			_ if status == http::StatusCode::TOO_MANY_REQUESTS => TurnErrorKind::RateLimited,
			_ if status == http::StatusCode::SERVICE_UNAVAILABLE => TurnErrorKind::Overloaded,
			_ => TurnErrorKind::Upstream,
		}
	};
	TurnError::builder()
		.kind(kind)
		.detail(Str::new(message))
		.unsupported(Vec::new())
		.retry_after_ms(0)
		.build()
}

fn contains_ascii_case_insensitive(haystack: &str, needle: &[u8]) -> bool {
	haystack
		.as_bytes()
		.windows(needle.len())
		.any(|window| window.eq_ignore_ascii_case(needle))
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

fn provider_error(detail: impl Into<Str>) -> Error {
	Error::Provider(detail.into())
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn regional_endpoint_uses_anthropic_publisher_and_escaped_model() {
		let endpoint =
			endpoint("", "my-project", "us-east5", "claude:sonnet/4").expect("valid endpoint");
		assert_eq!(
			endpoint,
			"https://us-east5-aiplatform.googleapis.com/v1/projects/my-project/locations/us-east5/publishers/anthropic/models/claude%3Asonnet%2F4:streamRawPredict"
		);
	}

	#[test]
	fn global_location_uses_canonical_vertex_host() {
		let endpoint =
			endpoint("", "my-project", "global", "claude-sonnet").expect("global endpoint");
		assert_eq!(
			endpoint,
			"https://aiplatform.googleapis.com/v1/projects/my-project/locations/global/publishers/anthropic/models/claude-sonnet:streamRawPredict"
		);
	}

	#[test]
	fn adc_context_strips_direct_anthropic_headers() {
		let mut request = http::Request::post("https://aiplatform.googleapis.com")
			.header("anthropic-beta", "pdfs-2024-09-25")
			.header("anthropic-version", "2023-06-01")
			.body(())
			.expect("request");
		attach_adc(&mut request, "anthropic-vertex");
		assert!(request.headers().get("anthropic-beta").is_none());
		assert!(request.headers().get("anthropic-version").is_none());
		assert_eq!(
			request.extensions().get::<AuthContext>(),
			Some(&AuthContext::new("anthropic-vertex"))
		);
	}

	#[test]
	fn google_rpc_error_classification_preserves_safety_detail() {
		let auth = classify_error(
			http::StatusCode::UNAUTHORIZED,
			br#"{"error":{"status":"UNAUTHENTICATED","message":"expired ADC token"}}"#,
		);
		assert_eq!(auth.kind, TurnErrorKind::Auth);
		let safety = classify_error(
			http::StatusCode::BAD_REQUEST,
			br#"{"error":{"status":"INVALID_ARGUMENT","message":"request blocked by safety policy"}}"#,
		);
		assert_eq!(safety.kind, TurnErrorKind::Upstream);
		assert!(safety.detail.contains("safety policy"));
	}
}
