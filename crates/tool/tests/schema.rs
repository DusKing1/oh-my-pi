//! Integration coverage for model-facing schema generation.

use omp_tool::{ToolParam, schema};
use serde::Deserialize;
use serde_json::{Value, json};

/// Optional nested settings.
#[derive(Deserialize, ToolParam)]
struct Nested {
	/// toggles the nested behavior
	enabled: bool,
}

/// Selector vocabulary.
#[derive(Deserialize, ToolParam)]
#[serde(rename_all = "lowercase")]
enum Mode {
	Fast,
	Thorough,
}

#[derive(Deserialize, ToolParam)]
#[serde(deny_unknown_fields)]
#[param(extend({ "allOf": [{ "required": ["required"] }] }))]
#[expect(dead_code, reason = "deserialization targets are constructed by serde only")]
struct Params {
	/// Required input value.
	required:  String,
	/// Optional nested settings.
	optional:  Option<Nested>,
	/// Selector with enum vocabulary.
	mode:      Option<Mode>,
	/// Defaulted, so never required.
	#[serde(default)]
	defaulted: bool,
	/// Renamed on the wire.
	#[serde(rename = "wire")]
	local:     Option<String>,
	/// Nullable pagination cursor.
	#[param(nullable, minimum = 0)]
	cursor:    Option<f64>,
	/// doc suppressed on the wire
	#[param(description = "")]
	silent:    Option<String>,
	/// Bounded free text.
	#[param(min_length = 1, max_length = 80)]
	label:     Option<String>,
}

#[test]
fn generated_schema_is_compact_inlined_and_model_facing() {
	let first = schema::<Params>();
	let second = schema::<Params>();
	assert_eq!(first, second, "schema generation must be deterministic");
	assert!(!first.contains(&b'\n'), "schema must use compact JSON encoding");

	let value: Value = serde_json::from_slice(&first).expect("generated schema is valid JSON");
	assert_eq!(value["type"], "object");
	assert_eq!(value["additionalProperties"], false);
	assert_eq!(value["required"], json!(["required"]), "only non-optional fields are required");
	assert_eq!(value["allOf"], json!([{ "required": ["required"] }]), "container extend merges");

	let properties = value["properties"]
		.as_object()
		.expect("properties is an object");
	assert_eq!(
		properties.keys().map(String::as_str).collect::<Vec<_>>(),
		["required", "optional", "mode", "defaulted", "wire", "cursor", "silent", "label"],
		"property declaration order must be preserved and renames applied"
	);

	assert_eq!(
		properties["required"],
		json!({ "description": "Required input value.", "type": "string" })
	);
	assert_eq!(
		properties["optional"],
		json!({
			"description": "Optional nested settings.",
			"type": "object",
			"properties": {
				"enabled": { "description": "toggles the nested behavior", "type": "boolean" }
			},
			"required": ["enabled"]
		}),
		"nested schemas inline, and the field doc overrides the container doc"
	);
	assert_eq!(
		properties["mode"],
		json!({
			"description": "Selector with enum vocabulary.",
			"type": "string",
			"enum": ["fast", "thorough"]
		})
	);
	assert_eq!(
		properties["cursor"],
		json!({
			"description": "Nullable pagination cursor.",
			"type": ["number", "null"],
			"minimum": 0
		}),
		"nullable widens the scalar type after bounds apply"
	);
	assert_eq!(properties["silent"], json!({ "type": "string" }), "empty override drops the doc");
	assert_eq!(
		properties["label"],
		json!({
			"description": "Bounded free text.",
			"type": "string",
			"minLength": 1,
			"maxLength": 80
		})
	);

	let parsed: Params = serde_json::from_value(json!({
		"required": "value",
		"optional": { "enabled": true },
		"mode": "fast",
		"cursor": null,
	}))
	.expect("schema example deserializes");
	assert_eq!(parsed.required, "value");
	assert!(parsed.optional.expect("optional settings").enabled);

	let encoded = std::str::from_utf8(&first).expect("JSON is UTF-8");
	for forbidden in ["$schema", "$ref", "$defs", "title", "format"] {
		assert!(!encoded.contains(forbidden), "schema must not contain {forbidden}");
	}
}

#[test]
fn unsigned_integers_carry_their_lower_bound() {
	#[derive(Deserialize, ToolParam)]
	#[expect(dead_code, reason = "deserialization target")]
	struct Counted {
		/// bounded count
		count: u64,
	}
	let value: Value =
		serde_json::from_slice(&schema::<Counted>()).expect("generated schema is valid JSON");
	assert_eq!(
		value["properties"]["count"],
		json!({ "description": "bounded count", "type": "integer", "minimum": 0 })
	);
}

#[test]
fn fallback_map_schema_stays_permissive() {
	use std::collections::BTreeMap;
	let bytes = schema::<BTreeMap<String, Value>>();
	assert_eq!(bytes.as_ref(), br#"{"type":"object","additionalProperties":true}"#);
}

#[test]
fn option_values_stay_nullable_outside_properties() {
	assert_eq!(
		<Vec<Option<f64>> as ToolParam>::schema(),
		json!({ "type": "array", "items": { "anyOf": [{ "type": "number" }, { "type": "null" }] } }),
		"value-position Option keeps accepting null"
	);
}
