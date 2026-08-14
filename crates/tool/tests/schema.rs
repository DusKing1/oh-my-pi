use omp_tool::schema;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(JsonSchema)]
struct Nested {
	enabled: bool,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Params {
	/// Required input value.
	required: String,
	/// Optional nested settings.
	optional: Option<Nested>,
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
	assert_eq!(value["required"], json!(["required"]));
	assert_eq!(value["properties"]["required"]["description"], "Required input value.");
	assert_eq!(value["properties"]["optional"]["description"], "Optional nested settings.");
	assert_eq!(value["properties"]["optional"]["type"], "object");
	assert_eq!(
		value["properties"]
			.as_object()
			.expect("properties is an object")
			.keys()
			.map(String::as_str)
			.collect::<Vec<_>>(),
		["required", "optional"],
		"property declaration order must be preserved"
	);

	let encoded = std::str::from_utf8(&first).expect("JSON is UTF-8");
	for forbidden in ["$schema", "$ref", "$defs", "title"] {
		assert!(!encoded.contains(forbidden), "schema must not contain {forbidden}");
	}
}
