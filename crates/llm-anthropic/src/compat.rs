//! Shared Anthropic Messages projection for cloud-hosted Claude endpoints.
//!
//! Bedrock and Vertex accept the Messages schema, but select the model in the
//! URL and require a cloud-specific `anthropic_version` in the JSON body.  This
//! module owns only that identical transformation; endpoint and framing rules
//! remain in their provider modules.

use bytes::Bytes;
use omp_core::SmolStr;
use omp_llm_catalog::compat::Compat;
use omp_llm_types::{ChatRequest, Error};
use serde_json::Value;
use smallvec::SmallVec;

use crate::request_headers;

/// Cloud-hosted Anthropic Messages wire family.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CloudMessages {
	/// Amazon Bedrock's native Anthropic Messages invocation body.
	Bedrock,
	/// Google Vertex AI's Anthropic `streamRawPredict` body.
	Vertex,
}

impl CloudMessages {
	/// Provider-required Anthropic protocol version.
	#[must_use]
	pub const fn anthropic_version(self) -> &'static str {
		match self {
			Self::Bedrock => "bedrock-2023-05-31",
			Self::Vertex => "vertex-2023-10-16",
		}
	}
}

/// Collects the Anthropic betas accepted by a cloud-hosted Messages body.
///
/// The direct transport's [`request_headers`] negotiation is authoritative for
/// request features shared by direct Anthropic, Bedrock, and Vertex. Explicit
/// cloud-namespace betas are merged after those negotiated values. The
/// cloud-only prompt-caching beta remains projected from the encoded
/// `cache_control` blocks in [`project_cloud_request`], because it is not a
/// direct Anthropic HTTP-header requirement.
pub(crate) fn cloud_betas(
	request: &ChatRequest,
	compat: &Compat,
	namespace: &str,
) -> Result<SmallVec<SmolStr, 12>, Error> {
	let mut betas = SmallVec::new();
	for header in request_headers(request, compat)
		.into_iter()
		.filter(|header| header.name.eq_ignore_ascii_case("anthropic-beta"))
	{
		for beta in header.value.split(",") {
			push_unique(&mut betas, &beta);
		}
	}

	if let Some(value) = request
		.provider_options
		.as_ref()
		.and_then(|options| options.get_ns(namespace, "betas"))
	{
		let values = value
			.as_array()
			.ok_or_else(|| provider_error(format!("{namespace}/betas must be an array of strings")))?;
		for value in values {
			let beta = value
				.as_str()
				.filter(|value| !value.is_empty())
				.ok_or_else(|| {
					provider_error(format!("{namespace}/betas contains a non-string value"))
				})?;
			push_unique(&mut betas, beta);
		}
	}

	Ok(betas)
}

/// Projects an ordinary Anthropic Messages request into a cloud-hosted body.
///
/// `model` and `stream` are removed because both cloud APIs select streaming
/// and the model in the request path. Active Anthropic betas travel in
/// `anthropic_beta`; cloud endpoints do not consume the direct API's
/// `anthropic-beta` HTTP header.
pub fn project_cloud_request(
	body: &[u8],
	family: CloudMessages,
	betas: &[SmolStr],
) -> Result<Bytes, Error> {
	let mut value: Value = serde_json::from_slice(body).map_err(json_error)?;
	let object = value
		.as_object_mut()
		.ok_or_else(|| provider_error("Anthropic request body must be a JSON object"))?;

	object.remove("model");
	object.remove("stream");
	object
		.insert("anthropic_version".to_owned(), Value::String(family.anthropic_version().to_owned()));

	let has_cache_control = contains_key(&value, "cache_control");
	let mut projected_betas =
		SmallVec::<SmolStr, 12>::with_capacity(betas.len() + usize::from(has_cache_control));
	for beta in betas {
		push_unique(&mut projected_betas, beta.as_str());
	}
	if has_cache_control {
		push_unique(&mut projected_betas, "prompt-caching-2024-07-31");
	}
	if !projected_betas.is_empty() {
		let object = value.as_object_mut().expect("validated object above");
		object.insert(
			"anthropic_beta".to_owned(),
			Value::Array(
				projected_betas
					.into_iter()
					.map(|beta| Value::String(beta.as_str().to_owned()))
					.collect(),
			),
		);
	}

	serde_json::to_vec(&value)
		.map(Bytes::from)
		.map_err(json_error)
}

fn push_unique(values: &mut SmallVec<SmolStr, 12>, value: &str) {
	let value = value.trim();
	if !value.is_empty() && !values.iter().any(|candidate| candidate.as_str() == value) {
		values.push(SmolStr::new(value));
	}
}

fn contains_key(value: &Value, needle: &str) -> bool {
	match value {
		Value::Array(values) => values.iter().any(|value| contains_key(value, needle)),
		Value::Object(values) => {
			values.contains_key(needle) || values.values().any(|value| contains_key(value, needle))
		},
		Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => false,
	}
}

fn json_error(error: serde_json::Error) -> Error {
	provider_error(error.to_string())
}

fn provider_error(detail: impl Into<SmolStr>) -> Error {
	Error::Provider(detail.into())
}

#[cfg(test)]
mod tests {
	use bytes::Bytes;
	use omp_llm_catalog::compat::ReasoningWireFormat;
	use omp_llm_transport::Transport;
	use omp_llm_types::{
		BlobPart, CacheHint, CacheRetention, Effort, Fallback, Feature, Item, ItemKind, JsonSchema,
		Message, Part, Props, Reasoning, ResponseFormat, ResponseFormatKind, Role, Thread, ToolDef,
	};
	use serde_json::json;

	use super::*;
	use crate::{AnthropicCodec, bedrock::BedrockCodec, vertex::VertexCodec};

	#[test]
	fn projection_moves_cloud_controls_into_body_and_preserves_cache_points() {
		let body = br#"{"model":"claude","stream":true,"messages":[{"role":"user","content":[{"type":"text","text":"hello","cache_control":{"type":"ephemeral"}}]}]}"#;
		let projected = project_cloud_request(body, CloudMessages::Vertex, &[SmolStr::new_static(
			"interleaved-thinking-2025-05-14",
		)])
		.expect("project request");
		let value: Value = serde_json::from_slice(&projected).expect("projected JSON");
		assert_eq!(value["anthropic_version"], "vertex-2023-10-16");
		assert!(value.get("model").is_none());
		assert!(value.get("stream").is_none());
		assert_eq!(
			value["anthropic_beta"],
			serde_json::json!(["interleaved-thinking-2025-05-14", "prompt-caching-2024-07-31"])
		);
		assert_eq!(value["messages"][0]["content"][0]["cache_control"]["type"], "ephemeral");
	}

	fn canonical_beta_request(namespace: &str) -> ChatRequest {
		let pdf = BlobPart::builder()
			.hash([1; 32])
			.mime("application/pdf".into())
			.size(4)
			.inline(Bytes::from_static(b"%PDF"))
			.build();
		let image = BlobPart::builder()
			.hash([2; 32])
			.mime("image/png".into())
			.size(3)
			.inline(Bytes::from_static(b"png"))
			.build();
		let pdf_item = Item::builder()
			.seq(0)
			.kind(ItemKind::Message(
				Message::builder()
					.role(Role::User)
					.parts(vec![Part::Blob(pdf)])
					.build(),
			))
			.props(Props::default())
			.build();
		let mut file_props = Props::default();
		file_props.insert_ns(
			"anthropic",
			"image_sources",
			json!([{"type":"file","file_id":"file_cloud_beta"}]),
		);
		let file_item = Item::builder()
			.seq(1)
			.kind(ItemKind::Message(
				Message::builder()
					.role(Role::User)
					.parts(vec![Part::Blob(image)])
					.build(),
			))
			.props(file_props)
			.build();
		let mut options = Props::default();
		options.insert_ns(
			"anthropic",
			"server_tools",
			json!([{"type":"web_search_20250305","name":"web_search"}]),
		);
		options.insert_ns("anthropic", "eager_input_streaming", json!(true));
		options.insert_ns("anthropic", "context_management", json!({"edits":[]}));
		options.insert_ns("anthropic", "service_tier", json!("priority"));
		options.insert_ns("anthropic", "betas", json!(["canonical-explicit-2026-08-10"]));
		options.insert_ns(
			namespace,
			"betas",
			json!(["pdfs-2024-09-25", "cloud-explicit-2026-08-10"]),
		);

		ChatRequest::builder()
			.model("claude-sonnet-4-6".into())
			.thread(Thread::builder().items(vec![pdf_item, file_item]).build())
			.tools(vec![
				ToolDef::builder()
					.name("lookup".into())
					.description("Look up a value".into())
					.schema_json(Bytes::from_static(br#"{"type":"object"}"#))
					.strict(false)
					.build(),
			])
			.thinking(
				Feature::builder()
					.value(Reasoning::builder().effort(Effort::High).build())
					.on_unsupported(Fallback::Ignore)
					.build(),
			)
			.cache(
				CacheHint::builder()
					.session_key("cloud-beta".into())
					.retention(CacheRetention::Long)
					.build(),
			)
			.response_format(
				Feature::builder()
					.value(
						ResponseFormat::builder()
							.kind(ResponseFormatKind::JsonSchema(
								JsonSchema::builder()
									.name("answer".into())
									.schema_json(Bytes::from_static(br#"{"type":"object"}"#))
									.strict(true)
									.build(),
							))
							.build(),
					)
					.on_unsupported(Fallback::Ignore)
					.build(),
			)
			.provider_options(options)
			.build()
	}

	fn anthropic_compat() -> Compat {
		let mut compat = Compat::default();
		compat.reasoning_wire_format = ReasoningWireFormat::Anthropic;
		compat
	}

	fn assert_cloud_body(family: CloudMessages, namespace: &str) {
		let request = canonical_beta_request(namespace);
		let compat = anthropic_compat();
		let (body, _) = match family {
			CloudMessages::Bedrock => BedrockCodec::new().encode(&request, &compat),
			CloudMessages::Vertex => VertexCodec::new().encode(&request, &compat),
		}
		.expect("encode cloud request");
		let value: Value = serde_json::from_slice(&body).expect("cloud body");
		let betas = value["anthropic_beta"]
			.as_array()
			.expect("anthropic_beta array");
		for expected in [
			"pdfs-2024-09-25",
			"files-api-2025-04-14",
			"extended-cache-ttl-2025-04-11",
			"prompt-caching-2024-07-31",
			"interleaved-thinking-2025-05-14",
			"effort-2025-11-24",
			"structured-outputs-2025-12-15",
			"web-search-2025-03-05",
			"fine-grained-tool-streaming-2025-05-14",
			"context-management-2025-06-27",
			"fast-mode-2026-02-01",
			"canonical-explicit-2026-08-10",
			"cloud-explicit-2026-08-10",
		] {
			assert!(
				betas.iter().any(|beta| beta.as_str() == Some(expected)),
				"{namespace} body omitted {expected}: {betas:?}"
			);
		}
		assert_eq!(
			betas
				.iter()
				.filter(|beta| beta.as_str() == Some("pdfs-2024-09-25"))
				.count(),
			1,
			"canonical and explicit betas must be deduplicated"
		);
		assert_eq!(value["anthropic_version"], family.anthropic_version());
		assert!(value.get("model").is_none());
		assert!(value.get("stream").is_none());
	}

	#[test]
	fn bedrock_body_projects_every_canonical_anthropic_beta() {
		assert_cloud_body(CloudMessages::Bedrock, "amazon-bedrock");
	}

	#[test]
	fn vertex_body_projects_every_canonical_anthropic_beta() {
		assert_cloud_body(CloudMessages::Vertex, "anthropic-vertex");
	}

	#[test]
	fn direct_anthropic_keeps_betas_in_headers_only() {
		let request = canonical_beta_request("amazon-bedrock");
		let compat = anthropic_compat();
		let (body, _) = AnthropicCodec::new()
			.encode(&request, &compat)
			.expect("encode direct request");
		let value: Value = serde_json::from_slice(&body).expect("direct body");
		assert!(value.get("anthropic_beta").is_none());
		let headers = request_headers(&request, &compat);
		assert!(
			headers
				.iter()
				.any(|header| header.name == "anthropic-version")
		);
		let beta = headers
			.iter()
			.find(|header| header.name == "anthropic-beta")
			.expect("direct beta header");
		assert!(
			beta
				.value
				.split(",")
				.any(|value| value == "pdfs-2024-09-25")
		);
		assert!(!beta.value.contains("cloud-explicit-2026-08-10"));
		assert!(!beta.value.contains("prompt-caching-2024-07-31"));
	}
}
