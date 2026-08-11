//! Gemini request projection helpers shared by public `GenAI` and Vertex codecs.

use bytes::Bytes;
use omp_core::Str;
use omp_llm_catalog::compat::{Compat, ToolSchemaFlavor};
use omp_llm_transport::normalize;
use omp_llm_types::{
	BlobPart, ChatRequest, Error, Fallback, Props, ResponseFormatKind, Unsupported,
	UnsupportedAction,
};
use serde_json::{Map, Value, json};

pub const NAMESPACE: &str = "google";

const CACHED_CONTENT: &str = "cached_content";
const SAFETY_SETTINGS: &str = "safety_settings";
const RESPONSE_MODALITIES: &str = "response_modalities";
const GOOGLE_SEARCH: &str = "google_search";
const CODE_EXECUTION: &str = "code_execution";
const FILE_DATA: &str = "file_data";

pub const REQUEST_OPTION_NAMES: &[&str] =
	&[CACHED_CONTENT, SAFETY_SETTINGS, RESPONSE_MODALITIES, GOOGLE_SEARCH, CODE_EXECUTION];

pub fn project_response_format(
	req: &ChatRequest,
	_compat: &Compat,
	generation: &mut Map<String, Value>,
	unsupported: &mut Vec<Unsupported>,
) -> Result<(), Error> {
	let Some(format) = &req.response_format else {
		return Ok(());
	};
	match &format.value.kind {
		ResponseFormatKind::JsonSchema(schema) => {
			let value =
				serde_json::from_slice::<Value>(&schema.schema_json).map_err(provider_error)?;
			let (mut schema, mut reports) = normalize::google(&value);
			normalize_google_schema(&mut schema);
			unsupported.append(&mut reports);
			generation.insert("responseMimeType".into(), Value::String("application/json".into()));
			generation.insert("responseJsonSchema".into(), schema);
		},
		ResponseFormatKind::Grammar(_) => report(
			unsupported,
			"response_format.grammar",
			"Gemini GenerateContent does not accept portable grammar response formats",
			format.on_unsupported,
		)?,
		_ => report(
			unsupported,
			"response_format",
			"unknown response format cannot be projected to Gemini",
			format.on_unsupported,
		)?,
	}

	Ok(())
}

pub fn project_provider_options(
	req: &ChatRequest,
	body: &mut Map<String, Value>,
	unsupported: &mut Vec<Unsupported>,
) -> Result<(), Error> {
	let Some(options) = &req.provider_options else {
		return Ok(());
	};
	if let Some(cached) = options.get_ns(NAMESPACE, CACHED_CONTENT) {
		let cached = cached.as_str().ok_or_else(|| {
			Error::Provider("google/cached_content must be a string resource name".into())
		})?;
		if cached.trim().is_empty() {
			return Err(Error::Provider("google/cached_content must not be blank".into()));
		}
		let incompatible = ["systemInstruction", "tools", "toolConfig"]
			.into_iter()
			.filter(|key| body.contains_key(*key))
			.collect::<Vec<_>>();
		if !incompatible.is_empty() {
			return Err(Error::Provider(Str::from(format!(
				"google/cached_content cannot be combined with request-level {}",
				incompatible.join(", ")
			))));
		}
		body.insert("cachedContent".into(), Value::String(cached.into()));
	}
	if let Some(settings) = options.get_ns(NAMESPACE, SAFETY_SETTINGS) {
		if !settings.is_array() {
			return Err(Error::Provider("google/safety_settings must be an array".into()));
		}
		body.insert("safetySettings".into(), settings.clone());
	}
	if let Some(modalities) = options.get_ns(NAMESPACE, RESPONSE_MODALITIES) {
		let values = modalities.as_array().ok_or_else(|| {
			Error::Provider("google/response_modalities must be an array of strings".into())
		})?;
		if values.iter().any(|value| !value.is_string()) {
			return Err(Error::Provider(
				"google/response_modalities must contain only strings".into(),
			));
		}
		generation_config(body).insert("responseModalities".into(), modalities.clone());
	}
	for (name, wire) in [(GOOGLE_SEARCH, "googleSearch"), (CODE_EXECUTION, "codeExecution")] {
		let Some(enabled) = options.get_ns(NAMESPACE, name) else {
			continue;
		};
		let enabled = enabled
			.as_bool()
			.ok_or_else(|| Error::Provider(Str::from(format!("google/{name} must be a boolean"))))?;
		if enabled {
			body
				.entry("tools")
				.or_insert_with(|| Value::Array(Vec::new()));
			body
				.get_mut("tools")
				.and_then(Value::as_array_mut)
				.expect("Gemini tools is always an array")
				.push(json!({ (wire): {} }));
		}
	}
	if options.get_ns(NAMESPACE, CACHED_CONTENT).is_some() && body.contains_key("tools") {
		return Err(Error::Provider(
			"google/cached_content cannot be combined with request-level tools".into(),
		));
	}
	for key in options.0.keys() {
		let recognized = key
			.strip_prefix("google/")
			.is_some_and(|name| REQUEST_OPTION_NAMES.contains(&name.as_str()));
		if !recognized {
			report(
				unsupported,
				key.clone(),
				"Google codec does not recognize this provider option",
				Fallback::Ignore,
			)?;
		}
	}
	Ok(())
}

pub fn normalize_tool_schema(compat: &Compat, schema: &Value) -> (Value, Vec<Unsupported>) {
	let (mut schema, reports) = normalize::normalize(compat.tool_schema_flavor, schema);
	if compat.tool_schema_flavor == ToolSchemaFlavor::Google {
		normalize_google_schema(&mut schema);
	}
	(schema, reports)
}

fn normalize_google_schema(schema: &mut Value) {
	let Some(object) = schema.as_object_mut() else {
		return;
	};
	if let Some(types) = object.get("type").and_then(Value::as_array) {
		let non_null = types
			.iter()
			.filter_map(Value::as_str)
			.filter(|kind| *kind != "null")
			.collect::<Vec<_>>();
		if types.iter().any(|kind| kind.as_str() == Some("null")) && non_null.len() == 1 {
			let kind = non_null[0].to_owned();
			object.insert("type".into(), Value::String(kind));
			object.insert("nullable".into(), Value::Bool(true));
		}
	}
	if object.get("type").is_none()
		&& let Some(values) = object.get("enum").and_then(Value::as_array)
		&& let Some(first) = values.first()
	{
		let inferred = if first.is_string() {
			Some("string")
		} else if first.is_boolean() {
			Some("boolean")
		} else if first.is_i64() || first.is_u64() {
			Some("integer")
		} else if first.is_number() {
			Some("number")
		} else {
			None
		};
		if let Some(inferred) = inferred {
			object.insert("type".into(), Value::String(inferred.into()));
		}
	}
	if object.get("type").and_then(Value::as_str) == Some("object") {
		object.entry("properties").or_insert_with(|| json!({}));
	}
	if let Some(properties) = object.get_mut("properties").and_then(Value::as_object_mut) {
		for child in properties.values_mut() {
			normalize_google_schema(child);
		}
	}
	if let Some(items) = object.get_mut("items") {
		normalize_google_schema(items);
	}
}

pub fn project_file_data(
	blob: &BlobPart,
	index: usize,
	props: &Props,
) -> Result<Option<Value>, Error> {
	let Some(entries) = props.get_ns(NAMESPACE, FILE_DATA).and_then(Value::as_array) else {
		return Ok(None);
	};
	let Some(entry) = entries.get(index) else {
		return Ok(None);
	};
	if entry.is_null() {
		return Ok(None);
	}
	let object = entry
		.as_object()
		.ok_or_else(|| Error::Provider("google/file_data entries must be objects or null".into()))?;
	let uri = object
		.get("file_uri")
		.and_then(Value::as_str)
		.ok_or_else(|| Error::Provider("google/file_data entry requires file_uri".into()))?;
	if uri.trim().is_empty() {
		return Err(Error::Provider("google/file_data file_uri must not be blank".into()));
	}
	let mime = object
		.get("mime_type")
		.and_then(Value::as_str)
		.unwrap_or(blob.mime.as_str());
	Ok(Some(json!({ "fileData": { "mimeType": mime, "fileUri": uri } })))
}

pub fn encode_inline(blob: &BlobPart) -> Value {
	json!({ "inlineData": { "mimeType": blob.mime, "data": base64(&blob.inline) } })
}

fn generation_config(body: &mut Map<String, Value>) -> &mut Map<String, Value> {
	body
		.entry("generationConfig")
		.or_insert_with(|| Value::Object(Map::new()))
		.as_object_mut()
		.expect("Gemini generationConfig is always an object")
}

fn report(
	unsupported: &mut Vec<Unsupported>,
	what: impl Into<Str>,
	detail: impl Into<Str>,
	fallback: Fallback,
) -> Result<(), Error> {
	let action = match fallback {
		Fallback::Ignore => UnsupportedAction::Dropped,
		Fallback::Emulate => UnsupportedAction::Emulated,
		Fallback::Error => {
			return Err(Error::Unsupported(vec![
				Unsupported::builder()
					.what(what.into())
					.detail(detail.into())
					.action(UnsupportedAction::Dropped)
					.build(),
			]));
		},
		_ => UnsupportedAction::Dropped,
	};
	unsupported.push(
		Unsupported::builder()
			.what(what.into())
			.detail(detail.into())
			.action(action)
			.build(),
	);
	Ok(())
}

#[cold]
fn provider_error(error: impl std::fmt::Display) -> Error {
	Error::Provider(Str::from(error.to_string()))
}

fn base64(input: &Bytes) -> String {
	const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
	let mut output = String::with_capacity(input.len().div_ceil(3) * 4);
	for chunk in input.chunks(3) {
		let bits = (u32::from(chunk[0]) << 16)
			| (u32::from(*chunk.get(1).unwrap_or(&0)) << 8)
			| u32::from(*chunk.get(2).unwrap_or(&0));
		output.push(char::from(ALPHABET[((bits >> 18) & 63) as usize]));
		output.push(char::from(ALPHABET[((bits >> 12) & 63) as usize]));
		output.push(if chunk.len() > 1 {
			char::from(ALPHABET[((bits >> 6) & 63) as usize])
		} else {
			'='
		});
		output.push(if chunk.len() > 2 {
			char::from(ALPHABET[(bits & 63) as usize])
		} else {
			'='
		});
	}
	output
}

#[cfg(test)]
mod tests {
	use bytes::Bytes;
	use omp_llm_catalog::compat::{Compat, ToolSchemaFlavor};
	use omp_llm_types::{BlobPart, ChatRequest, Props, Thread};
	use serde_json::{Map, Value};

	use super::{normalize_tool_schema, project_file_data, project_provider_options};

	fn fixture() -> Value {
		serde_json::from_str(include_str!(
			"../tests/fixtures/google_genai/request.semantic_parity.json"
		))
		.unwrap()
	}

	fn request(provider_options: Props) -> ChatRequest {
		ChatRequest::builder()
			.model("gemini-3-flash".into())
			.thread(Thread::builder().items(Vec::new()).build())
			.tools(Vec::new())
			.provider_options(provider_options)
			.build()
	}

	#[test]
	fn recorded_schema_translation_collapses_nullable_type_arrays() {
		let fixture = fixture();
		let schema = fixture["canonical"]["response_format"]["schema"].clone();
		let mut compat = Compat::default();
		compat.tool_schema_flavor = ToolSchemaFlavor::Google;
		let (schema, reports) = normalize_tool_schema(&compat, &schema);
		assert_eq!(reports, [] as [omp_llm_types::Unsupported; 0]);
		assert_eq!(schema, fixture["wire"]["generationConfig"]["responseJsonSchema"]);
	}

	#[test]
	fn recorded_file_data_and_builtin_tools_project_exactly() {
		let fixture = fixture();
		let file_props: Props =
			serde_json::from_value(fixture["file_data"]["canonical_props"].clone()).unwrap();
		let blob = BlobPart::builder()
			.hash([0; 32])
			.mime("application/octet-stream".into())
			.size(0)
			.inline(Bytes::new())
			.build();
		assert_eq!(
			project_file_data(&blob, 0, &file_props).unwrap().unwrap(),
			fixture["file_data"]["wire"]
		);

		let options: Props =
			serde_json::from_value(fixture["canonical"]["provider_options"].clone()).unwrap();
		let mut body = Map::new();
		let mut unsupported = Vec::new();
		project_provider_options(&request(options), &mut body, &mut unsupported).unwrap();
		assert_eq!(unsupported, [] as [omp_llm_types::Unsupported; 0]);
		assert_eq!(
			body["generationConfig"]["responseModalities"],
			fixture["wire"]["generationConfig"]["responseModalities"]
		);
		assert_eq!(body["tools"], fixture["wire"]["tools"]);
	}

	#[test]
	fn recorded_cached_content_resource_is_opaque_and_exclusive() {
		let fixture = fixture();
		let options: Props =
			serde_json::from_value(fixture["cached_content"]["canonical"].clone()).unwrap();
		let mut body = Map::new();
		project_provider_options(&request(options.clone()), &mut body, &mut Vec::new()).unwrap();
		assert_eq!(body["cachedContent"], fixture["cached_content"]["wire"]["cachedContent"]);

		body.insert("systemInstruction".into(), serde_json::json!({ "parts": [] }));
		assert!(project_provider_options(&request(options), &mut body, &mut Vec::new()).is_err());
	}
}
