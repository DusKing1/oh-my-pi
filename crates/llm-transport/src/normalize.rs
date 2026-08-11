//! Cross-provider terminal and tool-schema normalization.

use std::collections::BTreeSet;

use bytes::Bytes;
use omp_core::Str;
use omp_llm_catalog::compat::ToolSchemaFlavor;
use omp_llm_types::{StopReason, ToolDef, Unsupported, UnsupportedAction};
use serde_json::{Map, Value, json};

const MAX_ANTHROPIC_STRICT_TOOLS: usize = 20;
const MAX_ANTHROPIC_STRICT_OPTIONAL_PARAMETERS: usize = 24;
const MAX_ANTHROPIC_STRICT_UNION_PARAMETERS: usize = 16;
const ANTHROPIC_STRICT_TOOL_ALLOWLIST: &[&str] = &["bash", "python", "edit", "find"];
const GOOGLE_KEYWORDS: &[&str] =
	&["type", "format", "description", "nullable", "enum", "items", "properties", "required"];
const CCA_EXTRA_STRIPPED: &[&str] =
	&["$schema", "additionalProperties", "patternProperties", "propertyNames", "title"];
const ANTHROPIC_KEYWORDS: &[&str] = &[
	"$defs",
	"definitions",
	"type",
	"description",
	"enum",
	"const",
	"default",
	"format",
	"items",
	"prefixItems",
	"properties",
	"required",
	"additionalProperties",
	"anyOf",
	"minimum",
	"maximum",
	"exclusiveMinimum",
	"exclusiveMaximum",
	"multipleOf",
	"minLength",
	"maxLength",
	"pattern",
	"minItems",
	"maxItems",
	"uniqueItems",
	"minProperties",
	"maxProperties",
];

fn report(what: impl Into<Str>, detail: impl Into<Str>) -> Unsupported {
	Unsupported::builder()
		.what(what.into())
		.detail(detail.into())
		.action(UnsupportedAction::Dropped)
		.build()
}

/// Gives canonical tool use precedence over benign provider terminals.
///
/// A safety filter remains authoritative. Tool calls take precedence over
/// natural completion and output length because callers must execute emitted
/// calls before continuing the turn.
#[inline]
#[must_use]
pub const fn with_tool_use_precedence(mapped: StopReason, has_tool_calls: bool) -> StopReason {
	if has_tool_calls && matches!(mapped, StopReason::EndTurn | StopReason::MaxTokens) {
		StopReason::ToolUse
	} else {
		mapped
	}
}

/// Merges successful provider terminals using canonical severity precedence.
///
/// This is used for multi-choice and multi-candidate protocols where a single
/// canonical outcome must retain the strongest terminal: content filtering,
/// tool use, output length, then natural completion.
#[inline]
#[must_use]
pub const fn merge_stop_reason(left: StopReason, right: StopReason) -> StopReason {
	if matches!(left, StopReason::ContentFilter) || matches!(right, StopReason::ContentFilter) {
		StopReason::ContentFilter
	} else if matches!(left, StopReason::ToolUse) || matches!(right, StopReason::ToolUse) {
		StopReason::ToolUse
	} else if matches!(left, StopReason::MaxTokens) || matches!(right, StopReason::MaxTokens) {
		StopReason::MaxTokens
	} else {
		StopReason::EndTurn
	}
}

fn path(parent: &str, key: &str) -> String {
	if parent == "#" {
		format!("#/{key}")
	} else {
		format!("{parent}/{key}")
	}
}

fn strict_impossible(value: &Value, anthropic: bool) -> Option<String> {
	let object = value.as_object()?;
	for key in object.keys() {
		let incompatible = if anthropic {
			matches!(key.as_str(), "oneOf" | "allOf" | "$ref" | "patternProperties" | "propertyNames")
				|| !ANTHROPIC_KEYWORDS.contains(&key.as_str())
		} else {
			matches!(
				key.as_str(),
				"$ref" | "patternProperties" | "propertyNames" | "unevaluatedProperties"
			)
		};
		if incompatible {
			return Some(key.clone());
		}
	}
	for (key, child) in object {
		let hit = match key.as_str() {
			"properties" | "$defs" | "definitions" => child.as_object().and_then(|map| {
				map.values()
					.find_map(|schema| strict_impossible(schema, anthropic))
			}),
			"items" | "additionalProperties" | "not" => strict_impossible(child, anthropic),
			"anyOf" | "oneOf" | "allOf" | "prefixItems" => child.as_array().and_then(|values| {
				values
					.iter()
					.find_map(|schema| strict_impossible(schema, anthropic))
			}),
			_ => None,
		};
		if hit.is_some() {
			return hit;
		}
	}
	None
}

fn enforce_strict(value: &Value, optional: &mut usize, unions: &mut usize) -> Option<Value> {
	let object = value.as_object()?;
	let mut out = object.clone();
	for key in ["anyOf", "oneOf", "allOf"] {
		if let Some(branches) = object.get(key).and_then(Value::as_array) {
			if key == "anyOf" {
				*unions += branches.len();
			}
			let normalized = branches
				.iter()
				.map(|branch| enforce_strict(branch, optional, unions))
				.collect::<Option<Vec<_>>>()?;
			out.insert(key.into(), Value::Array(normalized));
		}
	}
	if let Some(items) = object.get("items") {
		out.insert("items".into(), enforce_strict(items, optional, unions)?);
	}
	if let Some(prefix_items) = object.get("prefixItems").and_then(Value::as_array) {
		let normalized = prefix_items
			.iter()
			.map(|item| enforce_strict(item, optional, unions))
			.collect::<Option<Vec<_>>>()?;
		out.insert("prefixItems".into(), Value::Array(normalized));
	}
	for key in ["$defs", "definitions"] {
		if let Some(definitions) = object.get(key).and_then(Value::as_object) {
			let normalized = definitions
				.iter()
				.map(|(name, schema)| Some((name.clone(), enforce_strict(schema, optional, unions)?)))
				.collect::<Option<Map<_, _>>>()?;
			out.insert(key.into(), Value::Object(normalized));
		}
	}
	if object.get("type") == Some(&Value::String("object".into()))
		|| object.contains_key("properties")
	{
		let properties = object
			.get("properties")
			.and_then(Value::as_object)
			.cloned()
			.unwrap_or_default();
		let originally_required = object
			.get("required")
			.and_then(Value::as_array)
			.into_iter()
			.flatten()
			.filter_map(Value::as_str)
			.collect::<BTreeSet<_>>();
		let mut normalized = Map::new();
		for (name, property) in &properties {
			let property = enforce_strict(property, optional, unions)?;
			if originally_required.contains(name.as_str()) {
				normalized.insert(name.clone(), property);
			} else {
				*optional += 1;
				let already_nullable =
					property
						.get("anyOf")
						.and_then(Value::as_array)
						.is_some_and(|branches| {
							branches
								.iter()
								.any(|branch| branch.get("type") == Some(&Value::String("null".into())))
						});
				if already_nullable {
					normalized.insert(name.clone(), property);
				} else {
					normalized.insert(name.clone(), json!({ "anyOf": [property, { "type": "null" }] }));
				}
			}
		}
		out.insert("properties".into(), Value::Object(normalized));
		out.insert(
			"required".into(),
			Value::Array(properties.keys().cloned().map(Value::String).collect()),
		);
		out.insert("additionalProperties".into(), Value::Bool(false));
	}
	let has_shape = out.contains_key("type")
		|| ["anyOf", "oneOf", "allOf"]
			.iter()
			.any(|key| out.contains_key(*key));
	if !has_shape {
		return None;
	}
	Some(Value::Object(out))
}

/// Normalizes a schema for `OpenAI` strict function-tool validation.
pub fn openai_strict(schema: &Value) -> (Value, Vec<Unsupported>) {
	if let Some(keyword) = strict_impossible(schema, false) {
		return (schema.clone(), vec![report(
			"tool.schema.strict",
			format!(
				"`{keyword}` cannot be represented in OpenAI strict mode; emitted the non-strict \
				 schema"
			),
		)]);
	}
	let (mut optional, mut unions) = (0, 0);
	match enforce_strict(schema, &mut optional, &mut unions) {
		Some(schema) => (schema, Vec::new()),
		None => (schema.clone(), vec![report(
			"tool.schema.strict",
			"schema has an untyped or non-object node; emitted the non-strict schema",
		)]),
	}
}

/// Normalizes a schema for Anthropic strict tool validation.
pub fn anthropic_strict(schema: &Value) -> (Value, Vec<Unsupported>) {
	if let Some(keyword) = strict_impossible(schema, true) {
		return (schema.clone(), vec![report(
			"tool.schema.anthropic_strict",
			format!(
				"`{keyword}` is outside Anthropic's strict-schema allowlist; emitted the non-strict \
				 schema"
			),
		)]);
	}
	let (mut optional, mut unions) = (0, 0);
	let Some(normalized) = enforce_strict(schema, &mut optional, &mut unions) else {
		return (schema.clone(), vec![report(
			"tool.schema.anthropic_strict",
			"schema cannot be represented by Anthropic strict tools; emitted the non-strict schema",
		)]);
	};
	if optional > MAX_ANTHROPIC_STRICT_OPTIONAL_PARAMETERS
		|| unions > MAX_ANTHROPIC_STRICT_UNION_PARAMETERS
	{
		return (schema.clone(), vec![report(
			"tool.schema.anthropic_strict",
			"schema exceeds Anthropic's strict optional-parameter or union budget; emitted the \
			 non-strict schema",
		)]);
	}
	(normalized, Vec::new())
}

fn subset(value: &Value, allowed: &[&str], at: &str, reports: &mut Vec<Unsupported>) -> Value {
	let Some(object) = value.as_object() else {
		reports.push(report(
			at,
			"boolean and scalar subschemas are not supported by this tool-schema flavor",
		));
		return json!({});
	};
	let mut out = Map::new();
	for (key, child) in object {
		if !allowed.contains(&key.as_str()) {
			reports.push(report(
				path(at, key),
				format!("`{key}` is not supported by this tool-schema flavor"),
			));
			continue;
		}
		let normalized = match key.as_str() {
			"properties" => Value::Object(
				child
					.as_object()
					.map(|properties| {
						properties
							.iter()
							.map(|(name, schema)| {
								(name.clone(), subset(schema, allowed, &path(at, name), reports))
							})
							.collect()
					})
					.unwrap_or_default(),
			),
			"items" => subset(child, allowed, &path(at, key), reports),
			_ => child.clone(),
		};
		out.insert(key.clone(), normalized);
	}
	Value::Object(out)
}

/// Normalizes to the JSON Schema subset accepted by Google `GenAI` function
/// declarations.
pub fn google(schema: &Value) -> (Value, Vec<Unsupported>) {
	let mut reports = Vec::new();
	let value = subset(schema, GOOGLE_KEYWORDS, "#", &mut reports);
	(value, reports)
}
fn cca_node(value: &Value, at: &str, reports: &mut Vec<Unsupported>) -> Value {
	let Some(object) = value.as_object() else {
		return value.clone();
	};
	let mut out = Map::new();
	for (key, child) in object {
		if CCA_EXTRA_STRIPPED.contains(&key.as_str()) {
			reports.push(report(path(at, key), format!("Cloud Code Assist strips `{key}`")));
			continue;
		}
		let normalized = match key.as_str() {
			"properties" | "$defs" | "definitions" => Value::Object(
				child
					.as_object()
					.map(|schemas| {
						schemas
							.iter()
							.map(|(name, schema)| {
								(name.clone(), cca_node(schema, &path(at, name), reports))
							})
							.collect()
					})
					.unwrap_or_default(),
			),
			"items" | "not" | "contains" | "if" | "then" | "else" => {
				cca_node(child, &path(at, key), reports)
			},
			"oneOf" | "anyOf" | "allOf" | "prefixItems" => Value::Array(
				child
					.as_array()
					.map(|schemas| {
						schemas
							.iter()
							.map(|schema| cca_node(schema, &path(at, key), reports))
							.collect()
					})
					.unwrap_or_default(),
			),
			_ => child.clone(),
		};
		out.insert(key.clone(), normalized);
	}
	Value::Object(out)
}

/// Removes the five schema keywords rejected by Cloud Code Assist.
pub fn cca(schema: &Value) -> (Value, Vec<Unsupported>) {
	let mut reports = Vec::new();
	let value = cca_node(schema, "#", &mut reports);
	(value, reports)
}

fn moonshot_node(value: &Value, at: &str, reports: &mut Vec<Unsupported>) -> Value {
	let Some(object) = value.as_object() else {
		if value.is_boolean() {
			reports.push(report(
				at,
				"Moonshot MFJS has no boolean subschemas; widened it to an empty schema",
			));
			return json!({});
		}
		return value.clone();
	};
	let mut out = Map::new();
	for (key, child) in object {
		let allowed =
			matches!(
				key.as_str(),
				"type"
					| "description"
					| "default"
					| "properties"
					| "required"
					| "items" | "enum"
					| "anyOf" | "additionalProperties"
					| "const" | "oneOf"
			);
		if !allowed {
			reports.push(report(path(at, key), format!("`{key}` is not part of Moonshot MFJS")));
			continue;
		}
		let normalized = match key.as_str() {
			"properties" => Value::Object(
				child
					.as_object()
					.map(|map| {
						map.iter()
							.map(|(name, schema)| {
								(name.clone(), moonshot_node(schema, &path(at, name), reports))
							})
							.collect()
					})
					.unwrap_or_default(),
			),
			"items" => moonshot_node(child, &path(at, key), reports),
			"additionalProperties" if child.is_boolean() => child.clone(),
			"additionalProperties" => moonshot_node(child, &path(at, key), reports),
			"anyOf" => Value::Array(
				child
					.as_array()
					.map(|values| {
						values
							.iter()
							.map(|value| moonshot_node(value, &path(at, key), reports))
							.collect()
					})
					.unwrap_or_default(),
			),
			"oneOf" => continue,
			"type" if child.is_array() => {
				reports.push(report(
					path(at, key),
					"Moonshot MFJS requires a scalar `type`; removed the nullable union branch",
				));
				child
					.as_array()
					.and_then(|types| types.iter().find(|kind| kind.as_str() != Some("null")))
					.cloned()
					.unwrap_or_else(|| Value::String("null".into()))
			},
			"const" => continue,
			_ => child.clone(),
		};
		out.insert(key.clone(), normalized);
	}
	if let Some(one_of) = object.get("oneOf").and_then(Value::as_array) {
		reports.push(report(path(at, "oneOf"), "Moonshot MFJS uses `anyOf`; converted `oneOf`"));
		let branches = one_of
			.iter()
			.map(|value| moonshot_node(value, &path(at, "oneOf"), reports));
		let existing = out
			.remove("anyOf")
			.and_then(|value| value.as_array().cloned())
			.unwrap_or_default();
		out.insert("anyOf".into(), Value::Array(existing.into_iter().chain(branches).collect()));
	}
	if let Some(constant) = object.get("const") {
		reports.push(report(path(at, "const"), "Moonshot MFJS uses `enum`; converted `const`"));
		out.insert("enum".into(), Value::Array(vec![constant.clone()]));
	}
	if let Some(values) = out.get("enum").and_then(Value::as_array) {
		if values
			.iter()
			.any(|value| !value.is_string() && !value.is_number())
		{
			reports.push(report(
				path(at, "enum"),
				"Moonshot MFJS enums contain only strings or numbers; dropped this enum",
			));
			out.remove("enum");
		} else if !out.contains_key("type") {
			let all_strings = values.iter().all(Value::is_string);
			let all_numbers = values.iter().all(Value::is_number);
			let all_integers = values.iter().all(|value| {
				value
					.as_number()
					.is_some_and(|number| number.is_i64() || number.is_u64())
			});
			let kind = if all_strings {
				Some("string")
			} else if all_integers {
				Some("integer")
			} else if all_numbers {
				Some("number")
			} else {
				None
			};
			if let Some(kind) = kind {
				out.insert("type".into(), Value::String(kind.into()));
			}
		}
	}
	Value::Object(out)
}

/// Normalizes to Moonshot Flavored JSON Schema.
pub fn moonshot_mfjs(schema: &Value) -> (Value, Vec<Unsupported>) {
	let mut reports = Vec::new();
	let value = moonshot_node(schema, "#", &mut reports);
	(value, reports)
}

struct Grammar {
	reports: Vec<Unsupported>,
}

impl Grammar {
	fn rule(&mut self, schema: &Value, at: &str) -> Option<String> {
		let Some(object) = schema.as_object() else {
			self
				.reports
				.push(report(at, "GBNF conversion requires an object-form JSON Schema node"));
			return None;
		};
		if let Some(values) = object.get("enum").and_then(Value::as_array) {
			if object
				.keys()
				.any(|key| !matches!(key.as_str(), "enum" | "type" | "description"))
			{
				self.reports.push(report(
					at,
					"GBNF cannot safely combine `enum` with other validation keywords",
				));
			}
			let choices = values
				.iter()
				.map(|value| {
					serde_json::to_string(value)
						.ok()
						.map(|literal| format!("{literal:?}"))
				})
				.collect::<Option<Vec<_>>>()?;
			return Some(choices.join(" | "));
		}
		if let Some(branches) = object
			.get("oneOf")
			.or_else(|| object.get("anyOf"))
			.and_then(Value::as_array)
		{
			let combiner_count =
				usize::from(object.contains_key("oneOf")) + usize::from(object.contains_key("anyOf"));
			if combiner_count != 1
				|| object
					.keys()
					.any(|key| !matches!(key.as_str(), "oneOf" | "anyOf" | "description"))
			{
				self.reports.push(report(
					at,
					"GBNF cannot safely combine a union with sibling validation keywords",
				));
			}
			return Some(
				branches
					.iter()
					.enumerate()
					.map(|(index, branch)| self.rule(branch, &format!("{at}/{index}")))
					.collect::<Option<Vec<_>>>()?
					.join(" | "),
			);
		}
		for key in object.keys() {
			if !matches!(
				key.as_str(),
				"type"
					| "description"
					| "properties"
					| "required"
					| "additionalProperties"
					| "items" | "enum"
					| "oneOf" | "anyOf"
			) {
				self.reports.push(report(
					path(at, key),
					format!(
						"GBNF conversion cannot express `{key}` without changing validation semantics"
					),
				));
			}
		}
		match object.get("type").and_then(Value::as_str) {
			Some("string") => Some(r#""\"" ([^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\"""#.into()),
			Some("integer") => Some(r#""-"? ("0" | [1-9] [0-9]*)"#.into()),
			Some("number") => Some(r#""-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?"#.into()),
			Some("boolean") => Some(r#""true" | "false""#.into()),
			Some("null") => Some(r#""null""#.into()),
			Some("array") => {
				let item = self.rule(object.get("items")?, &path(at, "items"))?;
				Some(format!(r#""[" ws ({item} (ws "," ws {item})*)? ws "]""#))
			}
			Some("object") => self.object_rule(object, at),
			_ => {
				self.reports.push(report(at, "GBNF conversion requires an explicit supported `type`"));
				None
			}
		}
	}

	fn object_rule(&mut self, object: &Map<String, Value>, at: &str) -> Option<String> {
		let properties = object.get("properties").and_then(Value::as_object)?;
		let required = object
			.get("required")
			.and_then(Value::as_array)
			.into_iter()
			.flatten()
			.filter_map(Value::as_str)
			.collect::<BTreeSet<_>>();
		if properties
			.keys()
			.any(|name| !required.contains(name.as_str()))
			|| object.get("additionalProperties") != Some(&Value::Bool(false))
		{
			self.reports.push(report(
				at,
				"GBNF object conversion requires every property and `additionalProperties: false`",
			));
			return None;
		}
		let mut fields = Vec::new();
		for (name, schema) in properties {
			let body = self.rule(schema, &path(at, name))?;
			let literal = serde_json::to_string(name).ok()?;
			fields.push(format!(r#"{literal:?} ws ":" ws {body}"#));
		}
		Some(format!(r#""{{" ws {} ws "}}""#, fields.join(r#" ws "," ws "#)))
	}
}

/// Converts a supported JSON Schema into a GBNF grammar string.
pub fn gbnf(schema: &Value) -> (Value, Vec<Unsupported>) {
	let mut grammar = Grammar { reports: Vec::new() };
	let root = grammar.rule(schema, "#");
	if root.is_none() || !grammar.reports.is_empty() {
		return (Value::String("root ::= \"__unsupported_tool_schema__\"\n".into()), grammar.reports);
	}
	let output = format!("root ::= {}\nws ::= [ \\t\\n\\r]*\n", root.expect("checked above"));
	(Value::String(output), Vec::new())
}

/// Dispatches normalization from the provider's compatibility data.
pub fn normalize(flavor: ToolSchemaFlavor, schema: &Value) -> (Value, Vec<Unsupported>) {
	match flavor {
		ToolSchemaFlavor::JsonSchema | ToolSchemaFlavor::Anthropic => (schema.clone(), Vec::new()),
		ToolSchemaFlavor::Google => google(schema),
		ToolSchemaFlavor::MoonshotMfjs => moonshot_mfjs(schema),
		ToolSchemaFlavor::Grammar => gbnf(schema),
		ToolSchemaFlavor::Cca => cca(schema),
	}
}

/// Normalizes one canonical tool definition while preserving its other fields.
pub fn normalize_tool(flavor: ToolSchemaFlavor, tool: &ToolDef) -> (ToolDef, Vec<Unsupported>) {
	let Ok(schema) = serde_json::from_slice::<Value>(&tool.schema_json) else {
		let mut normalized = tool.clone();
		if matches!(flavor, ToolSchemaFlavor::JsonSchema | ToolSchemaFlavor::Anthropic)
			&& normalized.strict == Some(true)
		{
			normalized.strict = Some(false);
		}
		return (normalized, vec![report(
			"tool.schema",
			"tool schema is not valid JSON; emitted it unchanged and disabled strict validation",
		)]);
	};
	let (schema, reports) = if tool.strict == Some(true) {
		match flavor {
			ToolSchemaFlavor::JsonSchema => openai_strict(&schema),
			ToolSchemaFlavor::Anthropic => anthropic_strict(&schema),
			_ => normalize(flavor, &schema),
		}
	} else {
		normalize(flavor, &schema)
	};
	let mut normalized = tool.clone();
	if let Ok(bytes) = serde_json::to_vec(&schema) {
		normalized.schema_json = Bytes::from(bytes);
	}
	if !reports.is_empty()
		&& matches!(flavor, ToolSchemaFlavor::JsonSchema | ToolSchemaFlavor::Anthropic)
	{
		normalized.strict = Some(false);
	}
	(normalized, reports)
}

/// Normalizes a tool list and applies Anthropic's strict tool-count/name
/// budget.
pub fn normalize_tools(
	flavor: ToolSchemaFlavor,
	tools: &[ToolDef],
) -> (Vec<ToolDef>, Vec<Unsupported>) {
	let mut normalized = Vec::with_capacity(tools.len());
	let mut reports = Vec::new();
	let mut anthropic_strict_count = 0;
	for tool in tools {
		if flavor == ToolSchemaFlavor::Anthropic
			&& tool.strict == Some(true)
			&& (!ANTHROPIC_STRICT_TOOL_ALLOWLIST.contains(&tool.name.as_str())
				|| anthropic_strict_count >= MAX_ANTHROPIC_STRICT_TOOLS)
		{
			let mut non_strict = tool.clone();
			non_strict.strict = Some(false);
			normalized.push(non_strict);
			reports.push(report(
				"tool.schema.anthropic_strict",
				"tool is outside Anthropic's strict name/count budget; emitted the non-strict schema",
			));
			continue;
		}
		let (tool, mut tool_reports) = normalize_tool(flavor, tool);
		if flavor == ToolSchemaFlavor::Anthropic
			&& tool.strict == Some(true)
			&& tool_reports.is_empty()
		{
			anthropic_strict_count += 1;
		}
		normalized.push(tool);
		reports.append(&mut tool_reports);
	}
	(normalized, reports)
}

#[cfg(test)]
mod tests {
	#[test]
	fn terminal_precedence_is_content_filter_then_tools_then_length() {
		assert_eq!(
			merge_stop_reason(StopReason::ToolUse, StopReason::ContentFilter),
			StopReason::ContentFilter
		);
		assert_eq!(
			merge_stop_reason(StopReason::MaxTokens, StopReason::ToolUse),
			StopReason::ToolUse
		);
		assert_eq!(
			merge_stop_reason(StopReason::EndTurn, StopReason::MaxTokens),
			StopReason::MaxTokens
		);
	}

	use super::*;
	fn matrix_schema() -> Value {
		json!({
			"type": "object",
			"description": "matrix",
			"$schema": "https://json-schema.org/draft/2020-12/schema",
			"properties": {
				"name": { "type": "string", "enum": ["a", "b"], "pattern": "^[ab]$" },
				"nested": {
					"type": "object",
					"properties": {
						"tags": { "type": "array", "items": { "type": "string", "enum": ["x", "y"] } }
					},
					"required": ["tags"],
					"additionalProperties": false
				},
				"choice": {
					"oneOf": [{ "type": "integer" }, { "$ref": "#/$defs/named" }]
				}
			},
			"required": ["name", "nested", "choice"],
			"$defs": { "named": { "type": "string" } },
			"unevaluatedProperties": false
		})
	}

	#[test]
	fn shared_schema_has_expected_result_for_every_flavor() {
		let input = matrix_schema();
		let google_expected = json!({
			"type": "object",
			"description": "matrix",
			"properties": {
				"name": { "type": "string", "enum": ["a", "b"] },
				"nested": {
					"type": "object",
					"properties": {
						"tags": { "type": "array", "items": { "type": "string", "enum": ["x", "y"] } }
					},
					"required": ["tags"]
				},
				"choice": {}
			},
			"required": ["name", "nested", "choice"]
		});
		let cca_expected = json!({
			"type": "object",
			"description": "matrix",
			"properties": {
				"name": { "type": "string", "enum": ["a", "b"], "pattern": "^[ab]$" },
				"nested": {
					"type": "object",
					"properties": {
						"tags": { "type": "array", "items": { "type": "string", "enum": ["x", "y"] } }
					},
					"required": ["tags"]
				},
				"choice": {
					"oneOf": [{ "type": "integer" }, { "$ref": "#/$defs/named" }]
				}
			},
			"required": ["name", "nested", "choice"],
			"$defs": { "named": { "type": "string" } },
			"unevaluatedProperties": false
		});
		for flavor in [
			ToolSchemaFlavor::JsonSchema,
			ToolSchemaFlavor::Anthropic,
			ToolSchemaFlavor::Google,
			ToolSchemaFlavor::MoonshotMfjs,
			ToolSchemaFlavor::Grammar,
			ToolSchemaFlavor::Cca,
		] {
			let (actual, reports) = normalize(flavor, &input);
			if matches!(flavor, ToolSchemaFlavor::JsonSchema | ToolSchemaFlavor::Anthropic) {
				assert_eq!(reports, [] as [omp_llm_types::Unsupported; 0]);
			} else {
				assert!(!reports.is_empty(), "{flavor:?} silently discarded an incompatible construct");
				assert!(
					reports
						.iter()
						.all(|report| report.action == UnsupportedAction::Dropped)
				);
			}
			match flavor {
				ToolSchemaFlavor::JsonSchema | ToolSchemaFlavor::Anthropic => assert_eq!(actual, input),
				ToolSchemaFlavor::Google => assert_eq!(actual, google_expected),
				ToolSchemaFlavor::Cca => assert_eq!(actual, cca_expected),
				ToolSchemaFlavor::MoonshotMfjs => {
					assert!(actual.pointer("/properties/choice/anyOf").is_some());
					assert!(actual.pointer("/properties/name/pattern").is_none());
					assert!(actual.pointer("/properties/choice/anyOf/1/$ref").is_none());
				},
				ToolSchemaFlavor::Grammar => {
					assert_eq!(
						actual,
						Value::String("root ::= \"__unsupported_tool_schema__\"\n".into())
					);
				},
			}
		}
	}

	#[test]
	fn strict_impossible_degrades_to_original_schema() {
		let schema = json!({
			"type": "object",
			"properties": { "recursive": { "$ref": "#" } },
			"required": ["recursive"]
		});
		for normalize in [openai_strict as fn(&Value) -> (Value, Vec<Unsupported>), anthropic_strict]
		{
			let (actual, reports) = normalize(&schema);
			assert_eq!(actual, schema);
			assert_eq!(reports.len(), 1);
			assert_eq!(reports[0].action, UnsupportedAction::Dropped);
		}
	}

	#[test]
	fn strict_requires_all_properties_and_closes_nested_objects() {
		let schema = json!({
			"type": "object",
			"properties": {
				"required": { "type": "string" },
				"optional": {
					"type": "object",
					"properties": { "leaf": { "type": "integer" } }
				}
			},
			"required": ["required"]
		});
		let (actual, reports) = openai_strict(&schema);
		assert_eq!(reports, [] as [omp_llm_types::Unsupported; 0]);
		assert_eq!(actual.get("additionalProperties"), Some(&Value::Bool(false)));
		assert_eq!(actual.get("required"), Some(&json!(["required", "optional"])));
		assert_eq!(
			actual.pointer("/properties/optional/anyOf/0/additionalProperties"),
			Some(&Value::Bool(false))
		);
		assert_eq!(actual.pointer("/properties/optional/anyOf/0/required"), Some(&json!(["leaf"])));
	}

	#[test]
	fn cca_removes_five_named_fields_and_preserves_google_fields() {
		let schema = json!({
			"type": "string",
			"format": "date",
			"description": "date",
			"nullable": true,
			"enum": ["today"],
			"items": { "type": "string" },
			"properties": { "x": { "type": "integer" } },
			"required": ["x"],
			"$schema": "draft",
			"additionalProperties": false,
			"patternProperties": {},
			"propertyNames": {},
			"title": "title",
			"x-extra": { "kept": true }
		});
		let (actual, reports) = cca(&schema);
		for key in GOOGLE_KEYWORDS {
			assert_eq!(actual.get(*key), schema.get(*key), "supported key `{key}` changed");
		}
		for key in CCA_EXTRA_STRIPPED {
			assert!(actual.get(*key).is_none(), "CCA retained `{key}`");
		}
		assert_eq!(actual.get("x-extra"), schema.get("x-extra"));
		assert_eq!(reports.len(), CCA_EXTRA_STRIPPED.len());
	}
	#[test]
	fn anthropic_strict_budgets_degrade_instead_of_overflowing() {
		let mut properties = Map::new();
		for index in 0..=MAX_ANTHROPIC_STRICT_OPTIONAL_PARAMETERS {
			properties.insert(format!("p{index}"), json!({ "type": "string" }));
		}
		let schema = json!({
			"type": "object",
			"properties": properties,
			"required": []
		});
		let (actual, reports) = anthropic_strict(&schema);
		assert_eq!(actual, schema);
		assert_eq!(reports.len(), 1);

		let tools = (0..=MAX_ANTHROPIC_STRICT_TOOLS)
			.map(|_| {
				ToolDef::builder()
					.name(Str::new("bash"))
					.description(Str::new(""))
					.schema_json(Bytes::from_static(b"{\"type\":\"object\",\"properties\":{}}"))
					.strict(true)
					.build()
			})
			.collect::<Vec<_>>();
		let (normalized, reports) = normalize_tools(ToolSchemaFlavor::Anthropic, &tools);
		assert!(
			normalized[..MAX_ANTHROPIC_STRICT_TOOLS]
				.iter()
				.all(|tool| tool.strict == Some(true))
		);
		assert_eq!(normalized[MAX_ANTHROPIC_STRICT_TOOLS].strict, Some(false));
		assert_eq!(reports.len(), 1);
	}

	#[test]
	fn simple_closed_object_has_stable_gbnf_round_trip() {
		let schema = json!({
			"type": "object",
			"properties": { "x": { "type": "integer" } },
			"required": ["x"],
			"additionalProperties": false
		});
		let (grammar, reports) = gbnf(&schema);
		assert_eq!(reports, [] as [omp_llm_types::Unsupported; 0]);
		assert_eq!(
			grammar,
			Value::String(
				"root ::= \"{\" ws \"\\\"x\\\"\" ws \":\" ws \"-\"? (\"0\" | [1-9] [0-9]*) ws \
				 \"}\"\nws ::= [ \\t\\n\\r]*\n"
					.into()
			)
		);
	}
}
