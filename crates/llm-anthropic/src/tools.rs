use std::borrow::Cow;

use bytes::Bytes;
use omp_llm_types::{Error, Props};
use serde::Serialize;
use serde_json::{Map, Value};

const BUILTIN_TOOL_NAMES: [&str; 4] = ["web_search", "code_execution", "text_editor", "computer"];

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum ToolNamePolicy {
	#[default]
	Unchanged,
	ClaudeOauth,
}

impl ToolNamePolicy {
	pub(crate) fn encode<'a>(self, name: &'a str, escape_all: bool) -> Cow<'a, str> {
		let escape = match self {
			Self::ClaudeOauth => !is_builtin(name),
			Self::Unchanged => escape_all,
		};
		if escape {
			Cow::Owned(format!("_{name}"))
		} else {
			Cow::Borrowed(name)
		}
	}

	pub(crate) fn decode<'a>(self, name: &'a str) -> &'a str {
		if self == Self::ClaudeOauth && !is_builtin(name) {
			name.strip_prefix('_').unwrap_or(name)
		} else {
			name
		}
	}
}

fn is_builtin(name: &str) -> bool {
	BUILTIN_TOOL_NAMES.contains(&name)
}

/// One client-defined tool on Anthropic's Messages wire.
#[derive(Serialize)]
pub(crate) struct ClientTool<'a> {
	pub(crate) name:                  Cow<'a, str>,
	pub(crate) description:           &'a str,
	pub(crate) input_schema:          Value,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub(crate) strict:                Option<bool>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub(crate) eager_input_streaming: Option<bool>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub(crate) cache_control:         Option<super::CacheControl>,
}

/// A client-defined or Anthropic-hosted tool definition.
#[derive(Serialize)]
#[serde(untagged)]
pub(crate) enum WireTool<'a> {
	Client(ClientTool<'a>),
	Server(Value),
}

impl WireTool<'_> {
	pub(crate) fn set_cache(&mut self, value: super::CacheControl) -> bool {
		match self {
			Self::Client(tool) => tool.cache_control = Some(value),
			Self::Server(Value::Object(tool)) => {
				let mut cache = Map::new();
				cache.insert("type".into(), Value::String(value.r#type.into()));
				cache.insert("ttl".into(), Value::String(value.ttl.into()));
				if let Some(scope) = value.scope {
					cache.insert("scope".into(), Value::String(scope.into()));
				}
				tool.insert("cache_control".into(), Value::Object(cache));
			},
			Self::Server(_) => return false,
		}
		true
	}
}

/// Anthropic's native tool selection policy.
#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum WireToolChoice<'a> {
	Auto {
		#[serde(skip_serializing_if = "Option::is_none")]
		disable_parallel_tool_use: Option<bool>,
	},
	None,
	Any {
		#[serde(skip_serializing_if = "Option::is_none")]
		disable_parallel_tool_use: Option<bool>,
	},
	Tool {
		name: Cow<'a, str>,
		#[serde(skip_serializing_if = "Option::is_none")]
		disable_parallel_tool_use: Option<bool>,
	},
}

/// Normalizes a JSON Schema to the keyword subset accepted by Messages tools.
pub(crate) fn normalize_schema(bytes: &Bytes) -> Result<Value, Error> {
	let Value::Object(mut object) =
		serde_json::from_slice(bytes).map_err(|error| Error::Provider(error.to_string().into()))?
	else {
		return Err(provider_error("Anthropic tool input_schema must be a JSON object"));
	};
	object.insert("type".into(), Value::String("object".into()));
	if !object.get("properties").is_some_and(Value::is_object) {
		object.insert("properties".into(), Value::Object(Map::new()));
	}
	let required = object
		.remove("required")
		.and_then(|value| value.as_array().cloned())
		.unwrap_or_default()
		.into_iter()
		.filter(Value::is_string)
		.collect();
	object.insert("required".into(), Value::Array(required));
	Ok(normalize_object(object, true))
}

fn normalize_node(value: Value, root: bool) -> Value {
	match value {
		Value::Array(values) => Value::Array(
			values
				.into_iter()
				.map(|value| normalize_node(value, false))
				.collect(),
		),
		Value::Object(object) => normalize_object(object, root),
		value => value,
	}
}

fn normalize_object(object: Map<String, Value>, root: bool) -> Value {
	let scalar = effective_type(&object);
	let mut kept = Map::new();
	let mut spilled = Vec::new();
	for (key, value) in object {
		let keep = universal_key(&key)
			&& !(root && matches!(key.as_str(), "anyOf" | "allOf"))
			&& key != "oneOf"
			|| match scalar {
				SchemaKind::Object => {
					matches!(key.as_str(), "properties" | "required" | "additionalProperties")
				},
				SchemaKind::Array => {
					matches!(key.as_str(), "items" | "prefixItems")
						|| key == "minItems" && matches!(value.as_u64(), Some(0 | 1))
				},
				SchemaKind::String => {
					key == "format" && value.as_str().is_some_and(supported_string_format)
				},
				SchemaKind::Other => false,
			};
		if keep {
			let value = normalize_kept_value(&key, value);
			kept.insert(key, value);
		} else {
			spilled.push((key, value));
		}
	}
	if scalar == SchemaKind::Object && !kept.contains_key("additionalProperties") {
		kept.insert("additionalProperties".into(), Value::Bool(false));
	}
	if !spilled.is_empty() {
		let suffix = spilled
			.into_iter()
			.map(|(key, value)| format!("{key}: {value}"))
			.collect::<Vec<_>>()
			.join(", ");
		let description = kept
			.remove("description")
			.and_then(|value| value.as_str().map(str::to_owned))
			.map_or_else(|| format!("{{{suffix}}}"), |value| format!("{value}\\n\\n{{{suffix}}}"));
		kept.insert("description".into(), Value::String(description));
	}
	Value::Object(kept)
}

fn normalize_kept_value(key: &str, value: Value) -> Value {
	if matches!(key, "properties" | "$defs" | "definitions") {
		return match value {
			Value::Object(children) => Value::Object(
				children
					.into_iter()
					.map(|(name, child)| (name, normalize_node(child, false)))
					.collect(),
			),
			value => value,
		};
	}
	if key == "additionalProperties" {
		let normalized = normalize_node(value, false);
		return if normalized
			.as_object()
			.is_some_and(|object| object.is_empty())
		{
			Value::Bool(true)
		} else {
			normalized
		};
	}
	if key == "items" {
		return normalize_node(value, false);
	}
	if matches!(key, "prefixItems" | "anyOf" | "allOf") {
		return match value {
			Value::Array(values) => Value::Array(
				values
					.into_iter()
					.map(|value| normalize_node(value, false))
					.collect(),
			),
			value => value,
		};
	}
	value
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum SchemaKind {
	Object,
	Array,
	String,
	Other,
}

fn effective_type(object: &Map<String, Value>) -> SchemaKind {
	let explicit = object.get("type").and_then(|value| {
		value.as_str().or_else(|| {
			value
				.as_array()?
				.iter()
				.filter_map(Value::as_str)
				.find(|kind| *kind != "null")
		})
	});
	match explicit {
		Some("object") => SchemaKind::Object,
		Some("array") => SchemaKind::Array,
		Some("string") => SchemaKind::String,
		Some(_) => SchemaKind::Other,
		None if object.get("properties").is_some_and(Value::is_object) => SchemaKind::Object,
		None
			if object.contains_key("items")
				|| object.get("prefixItems").is_some_and(Value::is_array) =>
		{
			SchemaKind::Array
		},
		None => SchemaKind::Other,
	}
}

fn universal_key(key: &str) -> bool {
	matches!(
		key,
		"$ref"
			| "$defs"
			| "$schema"
			| "definitions"
			| "type"
			| "anyOf"
			| "allOf"
			| "enum"
			| "const"
			| "description"
			| "title"
			| "default"
			| "nullable"
	)
}

fn supported_string_format(value: &str) -> bool {
	matches!(
		value,
		"date-time"
			| "time"
			| "date"
			| "duration"
			| "email"
			| "hostname"
			| "uri"
			| "ipv4"
			| "ipv6"
			| "uuid"
	)
}

/// Extracts and validates native server tool definitions.
pub(crate) fn server_tools(props: &Props) -> Result<Vec<&Value>, Error> {
	let Some(value) = props.get_ns("anthropic", "server_tools") else {
		return Ok(Vec::new());
	};
	let tools = value.as_array().ok_or_else(|| {
		provider_error("anthropic/server_tools must be an array of native tool definitions")
	})?;
	let mut projected = Vec::with_capacity(tools.len());
	for tool in tools {
		validate_server_tool(tool)?;
		projected.push(tool);
	}
	Ok(projected)
}

fn validate_server_tool(tool: &Value) -> Result<(), Error> {
	let object = tool
		.as_object()
		.ok_or_else(|| provider_error("every anthropic/server_tools entry must be an object"))?;
	let kind = object
		.get("type")
		.and_then(Value::as_str)
		.ok_or_else(|| provider_error("every anthropic/server_tools entry requires a string type"))?;
	let name = object
		.get("name")
		.and_then(Value::as_str)
		.ok_or_else(|| provider_error("every anthropic/server_tools entry requires a string name"))?;
	let expected_name = if kind.starts_with("web_search_") {
		"web_search"
	} else if kind.starts_with("web_fetch_") {
		"web_fetch"
	} else if kind.starts_with("code_execution_") {
		"code_execution"
	} else if kind.starts_with("bash_") {
		"bash"
	} else if kind.starts_with("text_editor_") {
		"str_replace_based_edit_tool"
	} else {
		return Err(provider_error(
			"anthropic/server_tools type must be web_search, web_fetch, code_execution, bash, or \
			 text_editor",
		));
	};
	if name != expected_name {
		return Err(provider_error(
			"anthropic/server_tools name does not match its native tool type",
		));
	}
	Ok(())
}

fn provider_error(message: &'static str) -> Error {
	Error::Provider(message.into())
}
