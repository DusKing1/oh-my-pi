//! Provider error-envelope walker.
//!
//! Provider bodies are structurally inconsistent: OpenAI-compatible hosts
//! nest `{error:{code,type,message}}` up to three `.error` layers deep, some
//! send flat `{error:"..."}` or `{message:"..."}` or `{detail:"..."}`,
//! Anthropic wraps as `{type:"error",error:{type,message}}`, Google uses
//! `{error:{code,message,status}}` with gRPC status words and sometimes
//! double-wraps another JSON error inside `error.message`, and proxies
//! return HTML. One walker normalizes all of it.

use omp_core::Str;
use serde_json::Value;

/// Fields recovered from a provider error body.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Envelope {
	/// Machine code (`error.code`, preferred over `error.type`).
	pub code:        Option<Str>,
	/// Error type token (`error.type`), when distinct from `code`.
	pub error_type:  Option<Str>,
	/// Human-readable message.
	pub message:     Option<Str>,
	/// gRPC-style status word (`"RESOURCE_EXHAUSTED"`, `"NOT_FOUND"`, ...).
	pub status_word: Option<Str>,
	/// Numeric status embedded in the body (Google `error.code`).
	pub status_code: Option<u16>,
	/// Provider request id when present (`request_id` / `requestId`).
	pub request_id:  Option<Str>,
	/// Offending parameter name (`error.param`).
	pub param:       Option<Str>,
	/// Body was HTML/non-JSON — treat content as opaque.
	pub opaque:      bool,
}

/// Parses a response body into an [`Envelope`].
///
/// Returns `None` for empty bodies. Non-JSON bodies yield an envelope with
/// `opaque: true` and the trimmed text as `message` so prose classification
/// still sees it.
pub fn parse(body: &str) -> Option<Envelope> {
	let trimmed = body.trim();
	if trimmed.is_empty() {
		return None;
	}
	if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
		let mut env = Envelope::default();
		walk(&value, &mut env, 0);
		// A JSON body that yielded nothing actionable is still opaque.
		if env.code.is_none()
			&& env.error_type.is_none()
			&& env.message.is_none()
			&& env.status_word.is_none()
		{
			env.opaque = true;
			env.message = Some(Str::new(trimmed));
		}
		return Some(env);
	}
	Some(Envelope { message: Some(Str::new(trimmed)), opaque: true, ..Envelope::default() })
}

/// Maximum `.error` nesting to descend. SDK gateways have been observed
/// burying the actionable code up to three envelopes deep.
const MAX_DEPTH: usize = 3;

fn walk(value: &Value, env: &mut Envelope, depth: usize) {
	let Value::Object(obj) = value else { return };

	// Anthropic outer `{type:"error", error:{...}}` — the outer `type` is
	// the frame discriminator, not the error type.
	let type_is_frame = obj.get("type").and_then(Value::as_str) == Some("error");

	if let Some(id) = str_field(obj, &["request_id", "requestId"]) {
		env.request_id.get_or_insert_with(|| Str::new(id));
	}

	// Descend nested error objects first: deepest fields are most specific.
	if depth < MAX_DEPTH
		&& let Some(inner) = obj.get("error")
	{
		match inner {
			Value::Object(_) => walk(inner, env, depth + 1),
			Value::String(s) => {
				env.message.get_or_insert_with(|| Str::new(s));
			},
			_ => {},
		}
	}

	if let Some(code) = obj.get("code") {
		match code {
			Value::String(s) => {
				env.code.get_or_insert_with(|| Str::new(s));
			},
			Value::Number(n) => {
				if let Some(status) = n.as_u64().and_then(|n| u16::try_from(n).ok())
					&& (100..=599).contains(&status)
				{
					env.status_code.get_or_insert(status);
				}
			},
			_ => {},
		}
	}
	if !type_is_frame && let Some(t) = obj.get("type").and_then(Value::as_str) {
		env.error_type.get_or_insert_with(|| Str::new(t));
	}
	if let Some(status) = obj.get("status").and_then(Value::as_str) {
		// gRPC status words are SCREAMING_SNAKE; HTTP reason phrases are not.
		if status.chars().all(|c| c.is_ascii_uppercase() || c == '_') && !status.is_empty() {
			env.status_word.get_or_insert_with(|| Str::new(status));
		}
	}
	if let Some(param) = obj.get("param").and_then(Value::as_str)
		&& !param.is_empty()
	{
		env.param.get_or_insert_with(|| Str::new(param));
	}
	if let Some(msg) = str_field(obj, &["message", "detail", "error_description"]) {
		// Google double-wrap: `error.message` may itself be a JSON error
		// document. Recurse once so the inner code/status surface.
		let inner_trim = msg.trim();
		if inner_trim.starts_with('{')
			&& depth < MAX_DEPTH
			&& let Ok(inner) = serde_json::from_str::<Value>(inner_trim)
		{
			walk(&inner, env, depth + 1);
		}
		env.message.get_or_insert_with(|| Str::new(msg));
	}
}

fn str_field<'v>(obj: &'v serde_json::Map<String, Value>, names: &[&str]) -> Option<&'v str> {
	names
		.iter()
		.find_map(|n| obj.get(*n).and_then(Value::as_str))
		.filter(|s| !s.is_empty())
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn openai_nested() {
		let env = parse(
			r#"{"error":{"message":"boom","type":"server_error","code":"internal_server_error"}}"#,
		)
		.unwrap();
		assert_eq!(env.code.as_deref(), Some("internal_server_error"));
		assert_eq!(env.error_type.as_deref(), Some("server_error"));
		assert_eq!(env.message.as_deref(), Some("boom"));
	}

	#[test]
	fn anthropic_frame() {
		let env = parse(
			r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"},"request_id":"req_011CW"}"#,
		)
		.unwrap();
		assert_eq!(env.error_type.as_deref(), Some("overloaded_error"));
		assert_eq!(env.message.as_deref(), Some("Overloaded"));
		assert_eq!(env.request_id.as_deref(), Some("req_011CW"));
	}

	#[test]
	fn google_double_wrap() {
		let body = r#"{"error":{"message":"{\n \"error\": {\n \"code\": 404,\n \"message\": \"models/gemini-live is not found\",\n \"status\": \"NOT_FOUND\"\n }\n}\n","code":404,"status":"Not Found"}}"#;
		let env = parse(body).unwrap();
		assert_eq!(env.status_word.as_deref(), Some("NOT_FOUND"));
		assert_eq!(env.status_code, Some(404));
	}

	#[test]
	fn flat_error_string() {
		let env = parse(r#"{"error":"invalid_grant"}"#).unwrap();
		assert_eq!(env.message.as_deref(), Some("invalid_grant"));
	}

	#[test]
	fn html_is_opaque() {
		let env = parse("<html><body>502 Bad Gateway</body></html>").unwrap();
		assert!(env.opaque);
		assert!(env.message.as_deref().unwrap().contains("502"));
	}

	#[test]
	fn grpc_status_word_only() {
		let env =
			parse(r#"{"error":{"status":"RESOURCE_EXHAUSTED","message":"Quota exceeded"}}"#).unwrap();
		assert_eq!(env.status_word.as_deref(), Some("RESOURCE_EXHAUSTED"));
	}

	#[test]
	fn reason_phrase_not_status_word() {
		let env = parse(r#"{"error":{"status":"Not Found","message":"x"}}"#).unwrap();
		assert_eq!(env.status_word, None);
	}
}
