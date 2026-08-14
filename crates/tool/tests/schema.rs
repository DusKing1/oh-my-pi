//! Typed schema generation contract tests.

use omp_tool::schema;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

#[allow(dead_code, reason = "fields are inspected by schema generation tests")]
#[derive(Deserialize, JsonSchema)]
struct Nested {
	enabled: bool,
}

#[allow(dead_code, reason = "fields are inspected by schema generation tests")]
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Params {
	/// Required input value.
	required: String,
	/// Optional nested settings.
	#[schemars(default, skip_serializing_if = "Option::is_none", with = "Nested")]
	optional: Option<Nested>,
}

#[test]
fn generated_schema_is_compact_inlined_and_model_facing() {
	let first = schema::<Params>();
	let second = schema::<Params>();
	assert_eq!(first, second, "schema generation must be deterministic");
	assert!(!first.contains(&b'\n'), "schema must use compact JSON encoding");

	let value: Value = serde_json::from_slice(&first).expect("generated schema is valid JSON");
	assert_eq!(
		value,
		json!({
			"type": "object",
			"properties": {
				"required": {
					"description": "Required input value.",
					"type": "string"
				},
				"optional": {
					"description": "Optional nested settings.",
					"type": "object",
					"properties": {
						"enabled": {"type": "boolean"}
					},
					"required": ["enabled"]
				}
			},
			"required": ["required"],
			"additionalProperties": false
		}),
		"generator settings and serde annotations must project exactly"
	);

	let encoded = std::str::from_utf8(&first).expect("JSON is UTF-8");
	for forbidden in ["$schema", "$ref", "$defs", "title"] {
		assert!(!encoded.contains(forbidden), "schema must not contain {forbidden}");
	}
}
