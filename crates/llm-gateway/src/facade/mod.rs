//! Foreign-wire HTTP facades for stock vendor SDKs.
//!
//! Authentication at this boundary is deliberately gateway authentication:
//! `Authorization: Bearer <gateway token>` and Anthropic's equivalent
//! `x-api-key: <gateway token>` authenticate the client to omp. They are never
//! provider credentials. Provider secrets are selected and injected by the
//! server-side egress stack and no facade request may override them.

pub mod audio;
pub mod chat;
pub mod embeddings;
pub mod images;
pub mod messages;
pub mod models;
pub mod responses;
pub mod videos;

use std::{collections::BTreeMap, convert::Infallible, fmt::Display, sync::Arc};

use bytes::Bytes;
use futures::{Stream, StreamExt};
use http::{Method, Request, Response, StatusCode, header};
use http_body_util::{BodyExt, Full, StreamBody, combinators::UnsyncBoxBody};
use hyper::body::{Body, Frame};
use omp_core::Str;
use omp_llm_types::{Props, RequestMeta, TurnError, TurnErrorKind, facet::Facets, ids::CallId};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::{Value, json};
use ulid::Ulid;
use zeroize::Zeroizing;

/// Response body shared by buffered JSON and streaming facade routes.
pub type FacadeBody = UnsyncBoxBody<Bytes, Infallible>;

/// Response returned by a foreign-wire facade route.
pub type FacadeResponse = Response<FacadeBody>;

/// Vendor envelope and streaming conventions used by a route.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Vendor {
	/// `OpenAI` Chat Completions error and SSE conventions.
	OpenAi,
	/// `OpenAI` Responses error and SSE conventions.
	Responses,
	/// Anthropic Messages error and SSE conventions.
	Anthropic,
}

/// Failure normalized at the facade boundary.
pub enum FacadeError {
	/// Malformed or semantically invalid vendor input.
	Invalid(Str),
	/// Client-to-gateway authentication failed.
	Unauthorized,
	/// A canonical turn failed after admission.
	Turn(TurnError),
	/// A native facet could not execute the request.
	Facet(omp_llm_types::facet::Error),
}

impl std::fmt::Debug for FacadeError {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			Self::Invalid(_) => formatter.write_str("FacadeError::Invalid"),
			Self::Unauthorized => formatter.write_str("FacadeError::Unauthorized"),
			Self::Turn(error) => formatter
				.debug_tuple("FacadeError::Turn")
				.field(&error.kind)
				.finish(),
			Self::Facet(_) => formatter.write_str("FacadeError::Facet"),
		}
	}
}

/// Gateway-client authentication material for one facade listener.
///
/// This token is never an upstream provider key. Provider credentials remain
/// private to server-side egress and cannot be supplied through facade headers.
///
/// ```compile_fail
/// let auth = omp_llm_gateway::facade::FacadeAuth::new("gateway-token");
/// let _token = auth.gateway_token();
/// ```
#[derive(Clone)]
pub struct FacadeAuth {
	gateway_token: Zeroizing<String>,
}

impl std::fmt::Debug for FacadeAuth {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		formatter.write_str("FacadeAuth([redacted])")
	}
}

impl FacadeAuth {
	/// Creates authentication policy for a gateway bearer token.
	#[must_use]
	pub fn new(gateway_token: impl AsRef<str>) -> Self {
		Self { gateway_token: Zeroizing::new(gateway_token.as_ref().to_owned()) }
	}

	fn authenticated<B>(&self, request: &Request<B>) -> bool {
		let bearer = request
			.headers()
			.get(header::AUTHORIZATION)
			.and_then(|value| value.to_str().ok())
			.and_then(|value| value.strip_prefix("Bearer "));
		let api_key = request
			.headers()
			.get("x-api-key")
			.and_then(|value| value.to_str().ok());
		!self.gateway_token.is_empty()
			&& [bearer, api_key]
				.into_iter()
				.flatten()
				.any(|token| constant_time_eq(token.as_bytes(), self.gateway_token.as_bytes()))
	}
}

/// Wire representation used by the models facade.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ModelsRepresentation {
	/// Select Anthropic when `anthropic-version` is present, otherwise `OpenAI`.
	#[default]
	Auto,
	/// Always emit `OpenAI` model objects.
	OpenAi,
	/// Always emit Anthropic model objects.
	Anthropic,
}

/// Per-listener foreign-facade behavior.
#[derive(Clone, Debug, Default)]
pub struct FacadeConfig {
	/// Representation emitted by `GET /v1/models`.
	pub models_representation: ModelsRepresentation,
}

/// Dependencies shared by all foreign-wire handlers.
pub struct FacadeState {
	/// The same facet services used by native inference RPC.
	pub facets:   Arc<Facets>,
	/// Live joined catalog shared with routing and native discovery.
	pub registry: Arc<parking_lot::RwLock<omp_llm_catalog::registry::Registry>>,
	/// Durable content-addressed media artifacts.
	pub blobs:    Arc<omp_storage::blob::BlobStore>,
	/// Client-to-gateway authentication policy.
	pub auth:     FacadeAuth,
	/// Per-listener facade configuration.
	pub config:   FacadeConfig,
}

/// Authenticated router for all official foreign-wire endpoints.
#[derive(Clone)]
pub struct Router {
	state: Arc<FacadeState>,
}

impl Router {
	/// Creates a router from its shared facade dependencies.
	#[must_use]
	pub const fn new(state: Arc<FacadeState>) -> Self {
		Self { state }
	}

	/// Returns the dependencies used by this router.
	#[must_use]
	pub const fn state(&self) -> &Arc<FacadeState> {
		&self.state
	}

	/// Authenticates and dispatches one Hyper request without requiring a
	/// socket.
	pub async fn route<B>(&self, request: Request<B>) -> FacadeResponse
	where
		B: Body<Data = Bytes> + Send + 'static,
		B::Error: Display,
	{
		let vendor = vendor_for_path(request.uri().path());
		if !self.state.auth.authenticated(&request) {
			return error_response(vendor, FacadeError::Unauthorized);
		}
		match (request.method(), request.uri().path()) {
			(&Method::POST, "/v1/chat/completions") => {
				chat::handle(request, Arc::clone(&self.state)).await
			},
			(&Method::POST, "/v1/responses") => {
				responses::handle(request, Arc::clone(&self.state)).await
			},
			(&Method::POST, "/v1/messages" | "/v1/messages/count_tokens") => {
				messages::handle(request, Arc::clone(&self.state)).await
			},
			(&Method::POST, "/v1/embeddings") => {
				embeddings::handle(request, Arc::clone(&self.state)).await
			},
			(&Method::POST, "/v1/images/generations" | "/v1/images/edits") => {
				images::handle(request, Arc::clone(&self.state)).await
			},
			(
				&Method::POST,
				"/v1/audio/speech" | "/v1/audio/transcriptions" | "/v1/audio/translations",
			) => audio::handle(request, Arc::clone(&self.state)).await,
			(&Method::POST, "/v1/videos") => videos::handle(request, Arc::clone(&self.state)).await,
			(method @ (&Method::GET | &Method::DELETE), path) if path.starts_with("/v1/videos/") => {
				let _ = method;
				videos::handle(request, Arc::clone(&self.state)).await
			},
			(&Method::GET, "/v1/models") => models::handle(request, Arc::clone(&self.state)),
			_ => json_response(
				StatusCode::NOT_FOUND,
				&json!({"error":{"message":"route not found","type":"invalid_request_error"}}),
			),
		}
	}
}

fn vendor_for_path(path: &str) -> Vendor {
	if path.starts_with("/v1/messages") {
		Vendor::Anthropic
	} else if path == "/v1/responses" {
		Vendor::Responses
	} else {
		Vendor::OpenAi
	}
}

/// Retains foreign request extensions under the transport vendor's namespace.
///
/// Transport adapters either consume these values or report them in the
/// canonical outcome's unsupported diagnostics; the facade never discards an
/// accepted extension silently.
pub(crate) fn provider_options(namespace: &str, fields: &BTreeMap<Str, Value>) -> Props {
	let mut options = Props::default();
	for (name, value) in fields {
		options.insert_ns(namespace, name, value.clone());
	}
	options
}

/// Projects a vendor end-user attribution value into canonical request
/// metadata.
pub(crate) fn request_meta(initiator: Option<&Str>) -> Option<RequestMeta> {
	let initiator = initiator?;
	Some(
		RequestMeta::builder()
			.initiator(initiator.clone())
			.session_id(Str::default())
			.telemetry(BTreeMap::new())
			.build(),
	)
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
	if left.len() != right.len() {
		return false;
	}
	left
		.iter()
		.zip(right)
		.fold(0_u8, |diff, (a, b)| diff | (a ^ b))
		== 0
}

pub(crate) async fn read_json<B, T>(
	request: Request<B>,
	vendor: Vendor,
) -> Result<T, Box<FacadeResponse>>
where
	B: Body<Data = Bytes> + Send + 'static,
	B::Error: Display,
	T: DeserializeOwned,
{
	let body = request
		.into_body()
		.collect()
		.await
		.map_err(|error| {
			Box::new(error_response(
				vendor,
				FacadeError::Invalid(Str::from(format!("failed to read request body: {error}"))),
			))
		})?
		.to_bytes();
	serde_json::from_slice(&body).map_err(|error| {
		Box::new(error_response(
			vendor,
			FacadeError::Invalid(Str::from(format!("invalid JSON request: {error}"))),
		))
	})
}

pub(crate) fn json_response<T: Serialize + ?Sized>(
	status: StatusCode,
	value: &T,
) -> FacadeResponse {
	let body = serde_json::to_vec(value)
		.unwrap_or_else(|_| b"{\"error\":{\"message\":\"response serialization failed\"}}".to_vec());
	Response::builder()
		.status(status)
		.header(header::CONTENT_TYPE, "application/json")
		.body(
			Full::new(Bytes::from(body))
				.map_err(|never| match never {})
				.boxed_unsync(),
		)
		.expect("static facade response is valid")
}

pub(crate) fn error_response(vendor: Vendor, error: FacadeError) -> FacadeResponse {
	let (status, message, code, retry_after) = match error {
		FacadeError::Invalid(detail) => (StatusCode::BAD_REQUEST, detail, "invalid_request_error", 0),
		FacadeError::Unauthorized => (
			StatusCode::UNAUTHORIZED,
			Str::from("invalid gateway credential"),
			"authentication_error",
			0,
		),
		FacadeError::Facet(_) => (
			StatusCode::INTERNAL_SERVER_ERROR,
			Str::new_static("provider operation failed"),
			"api_error",
			0,
		),
		FacadeError::Turn(error) => {
			let (status, code) = match error.kind {
				TurnErrorKind::Auth => (StatusCode::UNAUTHORIZED, "authentication_error"),
				TurnErrorKind::RateLimited | TurnErrorKind::Overloaded => {
					(StatusCode::TOO_MANY_REQUESTS, "rate_limit_error")
				},
				TurnErrorKind::Conflict | TurnErrorKind::NeedFull | TurnErrorKind::Unsupported => {
					(StatusCode::BAD_REQUEST, "invalid_request_error")
				},
				_ => (StatusCode::INTERNAL_SERVER_ERROR, "api_error"),
			};
			(status, public_turn_message(error.kind), code, error.retry_after_ms)
		},
	};
	let envelope = match vendor {
		Vendor::Anthropic => json!({"type":"error","error":{"type":code,"message":message}}),
		Vendor::OpenAi | Vendor::Responses => {
			json!({"error":{"message":message,"type":code,"param":Value::Null,"code":Value::Null}})
		},
	};
	let mut response = json_response(status, &envelope);
	if retry_after > 0 {
		let seconds = retry_after.div_ceil(1000).to_string();
		if let Ok(value) = seconds.parse() {
			response.headers_mut().insert(header::RETRY_AFTER, value);
		}
	}
	response
}

const fn public_turn_message(kind: TurnErrorKind) -> Str {
	match kind {
		TurnErrorKind::Auth => Str::new_static("provider authentication failed"),
		TurnErrorKind::RateLimited | TurnErrorKind::Overloaded => {
			Str::new_static("provider is temporarily unavailable")
		},
		TurnErrorKind::Conflict => Str::new_static("context revision conflict"),
		TurnErrorKind::NeedFull => Str::new_static("full context is required"),
		TurnErrorKind::Unsupported => Str::new_static("requested capability is unavailable"),
		_ => Str::new_static("provider operation failed"),
	}
}

pub(crate) fn sse_response<S>(events: S) -> FacadeResponse
where
	S: Stream<Item = Bytes> + Send + 'static,
{
	let frames = events.map(|bytes| Ok::<_, Infallible>(Frame::data(bytes)));
	Response::builder()
		.status(StatusCode::OK)
		.header(header::CONTENT_TYPE, "text/event-stream")
		.header(header::CACHE_CONTROL, "no-cache")
		.body(BodyExt::boxed_unsync(StreamBody::new(frames)))
		.expect("static facade SSE response is valid")
}

pub(crate) fn sse_data(value: &Value) -> Bytes {
	let mut bytes = Vec::from(b"data: ");
	serde_json::to_writer(&mut bytes, value).expect("JSON values serialize");
	bytes.extend_from_slice(b"\n\n");
	Bytes::from(bytes)
}

pub(crate) fn sse_named(name: &str, value: &Value) -> Bytes {
	let mut bytes = Vec::from(b"event: ");
	bytes.extend_from_slice(name.as_bytes());
	bytes.extend_from_slice(b"\ndata: ");
	serde_json::to_writer(&mut bytes, value).expect("JSON values serialize");
	bytes.extend_from_slice(b"\n\n");
	Bytes::from(bytes)
}

pub(crate) fn canonical_call_id(wire: &str) -> CallId {
	let digest = blake3::hash(wire.as_bytes());
	let mut bytes = [0_u8; 16];
	bytes.copy_from_slice(&digest.as_bytes()[..16]);
	CallId::from_ulid(Ulid::from_bytes(bytes))
}

#[cfg(test)]
mod tests {
	use http_body_util::{BodyExt, Full};

	use super::*;

	#[test]
	fn accepts_bearer_and_anthropic_api_key_only_when_correct() {
		let auth = FacadeAuth::new("gateway-secret");
		let bearer = Request::builder()
			.header(header::AUTHORIZATION, "Bearer gateway-secret")
			.body(Full::new(Bytes::new()))
			.expect("valid request");
		let api_key = Request::builder()
			.header("x-api-key", "gateway-secret")
			.body(Full::new(Bytes::new()))
			.expect("valid request");
		let missing = Request::new(Full::new(Bytes::new()));
		let wrong = Request::builder()
			.header("x-api-key", "provider-secret")
			.body(Full::new(Bytes::new()))
			.expect("valid request");
		assert!(auth.authenticated(&bearer));
		assert!(auth.authenticated(&api_key));
		assert!(!auth.authenticated(&missing));
		assert!(!auth.authenticated(&wrong));
	}

	#[test]
	fn gateway_auth_debug_is_redacted() {
		const CANARY: &str = "canary-gateway-api-key";
		let auth = FacadeAuth::new(CANARY);
		let debug = format!("{auth:?}");
		assert_eq!(debug, "FacadeAuth([redacted])");
		assert!(!debug.contains(CANARY));
	}

	#[tokio::test]
	async fn canonical_errors_use_vendor_envelopes_status_and_retry_header() {
		for vendor in [Vendor::OpenAi, Vendor::Responses, Vendor::Anthropic] {
			let error = TurnError::builder()
				.kind(TurnErrorKind::RateLimited)
				.detail(Str::from("slow down"))
				.unsupported(Vec::new())
				.retry_after_ms(1_250)
				.build();
			let response = error_response(vendor, FacadeError::Turn(error));
			assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
			assert_eq!(
				response
					.headers()
					.get(header::RETRY_AFTER)
					.and_then(|value| value.to_str().ok()),
				Some("2")
			);
			let body = response
				.into_body()
				.collect()
				.await
				.expect("infallible body")
				.to_bytes();
			let envelope: Value = serde_json::from_slice(&body).expect("JSON error envelope");
			let error_type = envelope.pointer("/error/type");
			assert_eq!(error_type.and_then(Value::as_str), Some("rate_limit_error"));
		}
	}

	#[tokio::test]
	async fn provider_diagnostics_never_enter_facade_error_payloads() {
		const CANARY: &str = "canary-provider-cookie-and-authorization";
		for vendor in [Vendor::OpenAi, Vendor::Responses, Vendor::Anthropic] {
			let turn = TurnError::builder()
				.kind(TurnErrorKind::Upstream)
				.detail(Str::new_static(CANARY))
				.unsupported(Vec::new())
				.retry_after_ms(0)
				.build();
			for error in [
				FacadeError::Turn(turn.clone()),
				FacadeError::Facet(omp_llm_types::facet::Error::Provider(CANARY.into())),
			] {
				let debug = format!("{error:?}");
				assert!(!debug.contains(CANARY));
				let body = error_response(vendor, error)
					.into_body()
					.collect()
					.await
					.expect("infallible body")
					.to_bytes();
				assert!(!String::from_utf8_lossy(&body).contains(CANARY));
			}
		}
	}
}
