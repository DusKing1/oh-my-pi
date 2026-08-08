//! Schema-derived argument coercion and delimiter matching primitives.

use std::{
	collections::{BTreeMap, BTreeSet},
	sync::atomic::{AtomicU64, Ordering},
};

use omp_core::SmolStr;
use serde_json::{Map, Value};

use crate::types::InbandTool;

/// Owned argument shape derived once when a scanner is constructed.
#[derive(Clone, Debug, Default)]
pub struct ToolArgShape {
	/// Properties whose schema admits only a string (optionally plus `null`).
	pub string_args:     BTreeSet<SmolStr>,
	/// Property schemas used by keyed XML-family scanners.
	pub properties:      BTreeMap<SmolStr, Value>,
	/// Source-schema property order retained for deterministic rendering.
	pub parameter_order: Vec<SmolStr>,
}

/// Tool-name lookup of owned, schema-derived argument shapes.
pub type ArgShapes = BTreeMap<SmolStr, ToolArgShape>;

/// Builds the owned coercion lookup used throughout one scanner lifetime.
#[must_use]
pub fn build_arg_shapes(tools: &[InbandTool<'_>]) -> ArgShapes {
	let mut shapes = ArgShapes::new();
	for tool in tools {
		let mut shape = ToolArgShape::default();
		if let Some(properties) = tool.parameters.get("properties").and_then(Value::as_object) {
			for (name, schema) in properties {
				let name = SmolStr::from(name.as_str());
				shape.parameter_order.push(name.clone());
				if is_string_only_schema(schema) {
					shape.string_args.insert(name.clone());
				}
				shape.properties.insert(name, schema.clone());
			}
		}
		shapes.insert(SmolStr::from(tool.name), shape);
	}
	shapes
}

/// Returns whether a JSON Schema admits strings and no non-null alternative.
#[must_use]
pub fn is_string_only_schema(schema: &Value) -> bool {
	let mut types = BTreeSet::new();
	collect_schema_types(schema, &mut types, 0);
	types.remove("null");
	types.len() == 1 && types.contains("string")
}

/// Collects primitive JSON types reachable through common schema combinators.
pub fn collect_schema_types(schema: &Value, out: &mut BTreeSet<&'static str>, depth: usize) {
	if depth > 8 {
		return;
	}
	let Some(node) = schema.as_object() else {
		return;
	};
	match node.get("type") {
		Some(Value::String(kind)) => insert_type(kind, out),
		Some(Value::Array(kinds)) => {
			for kind in kinds.iter().filter_map(Value::as_str) {
				insert_type(kind, out);
			}
		},
		None => {
			if let Some(values) = node.get("enum").and_then(Value::as_array) {
				for value in values {
					out.insert(json_type_of(value));
				}
			}
			if let Some(value) = node.get("const") {
				out.insert(json_type_of(value));
			}
		},
		Some(_) => {},
	}
	for key in ["anyOf", "oneOf", "allOf"] {
		if let Some(branches) = node.get(key).and_then(Value::as_array) {
			for branch in branches {
				collect_schema_types(branch, out, depth + 1);
			}
		}
	}
}

fn insert_type(kind: &str, out: &mut BTreeSet<&'static str>) {
	if let Some(kind) = match kind {
		"null" => Some("null"),
		"string" => Some("string"),
		"number" | "integer" => Some("number"),
		"boolean" => Some("boolean"),
		"array" => Some("array"),
		"object" => Some("object"),
		_ => None,
	} {
		out.insert(kind);
	}
}

/// Returns the JSON primitive category of a concrete value.
#[must_use]
pub const fn json_type_of(value: &Value) -> &'static str {
	match value {
		Value::Null => "null",
		Value::Bool(_) => "boolean",
		Value::Number(_) => "number",
		Value::String(_) => "string",
		Value::Array(_) => "array",
		Value::Object(_) => "object",
	}
}

/// Decodes ordinary JSON, retaining malformed or unquoted input as a string.
#[must_use]
pub fn decode_value(raw: &str) -> Value {
	let trimmed = raw.trim();
	if trimmed.is_empty() {
		return Value::String(trimmed.to_owned());
	}
	serde_json::from_str(trimmed).unwrap_or_else(|_| Value::String(raw.to_owned()))
}

/// Coerces a raw XML-family parameter according to its advertised schema.
///
/// String-only properties deliberately retain whitespace and JSON-looking text;
/// every other property receives tolerant JSON decoding.
#[must_use]
pub fn coerce_value(raw: &str, schema: Option<&Value>) -> Value {
	if schema.is_some_and(is_string_only_schema) {
		Value::String(raw.to_owned())
	} else {
		decode_value(raw)
	}
}

/// Coerces a named property using a precomputed tool shape.
#[must_use]
pub fn coerce_named_value(shape: Option<&ToolArgShape>, key: &str, raw: &str) -> Value {
	coerce_value(raw, shape.and_then(|shape| shape.properties.get(key)))
}

/// Returns whether a schema admits an array.
#[must_use]
pub fn is_array_schema(schema: &Value) -> bool {
	let mut types = BTreeSet::new();
	collect_schema_types(schema, &mut types, 0);
	types.contains("array")
}

/// Returns whether a schema admits an object.
#[must_use]
pub fn is_object_schema(schema: &Value) -> bool {
	let mut types = BTreeSet::new();
	collect_schema_types(schema, &mut types, 0);
	types.contains("object")
}

/// Returns an object's `properties`, or an empty map for another schema shape.
#[must_use]
pub fn object_properties(schema: &Value) -> Option<&Map<String, Value>> {
	schema.get("properties").and_then(Value::as_object)
}

/// Returns an array schema's item schema when present.
#[must_use]
pub fn array_item_schema(schema: &Value) -> Option<&Value> {
	schema.get("items")
}

static TOOL_CALL_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Mints a process-unique scanner-local call identifier without random state.
#[must_use]
pub fn mint_tool_call_id() -> SmolStr {
	let id = TOOL_CALL_COUNTER.fetch_add(1, Ordering::Relaxed);
	SmolStr::from(format!("ptc_{id:x}"))
}

/// Length of the longest suffix that is a proper prefix of `tag`.
#[must_use]
pub fn partial_suffix_overlap(text: &[u8], tag: &[u8]) -> usize {
	let max = text.len().min(tag.len().saturating_sub(1));
	(1..=max)
		.rev()
		.find(|&length| text.ends_with(&tag[..length]))
		.unwrap_or(0)
}

/// Greatest partial-suffix overlap across a delimiter set.
#[must_use]
pub fn partial_suffix_overlap_any(text: &[u8], tags: &[&[u8]]) -> usize {
	tags
		.iter()
		.map(|tag| partial_suffix_overlap(text, tag))
		.max()
		.unwrap_or(0)
}

/// Removes Kimi's optional namespace and ordinal suffix from a call header.
#[must_use]
pub fn normalize_kimi_function_name(raw_id: &str) -> &str {
	let before_index = raw_id.split(':').next().unwrap_or(raw_id);
	before_index
		.rsplit('.')
		.next()
		.unwrap_or(before_index)
		.trim()
}

/// Converts a JSON value to an object, replacing every other shape with `{}`.
#[must_use]
pub fn as_object(value: Value) -> Map<String, Value> {
	match value {
		Value::Object(object) => object,
		_ => Map::new(),
	}
}
