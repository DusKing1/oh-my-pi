//! Borrowed JSON-line and Harmony tool inventories with schema examples.

use std::{fmt, io};

use serde_json::{Map, Value};

use crate::{
	rendering::{write_json_value, write_py_call_value},
	types::{InbandTool, ToolExample},
};

/// Writes one OpenAI-shaped function definition per line without building a
/// temporary catalog or serialized schema string.
pub fn write_json_line_catalog<W: fmt::Write + ?Sized>(
	out: &mut W,
	tools: &[InbandTool<'_>],
) -> fmt::Result {
	for tool in tools {
		out.write_str("{\"type\":\"function\",\"function\":{\"name\":")?;
		write_json_string(out, tool.name)?;
		out.write_str(",\"description\":")?;
		write_json_string(out, tool.description.unwrap_or_default())?;
		out.write_str(",\"parameters\":")?;
		write_json_value(out, tool.parameters)?;
		out.write_str("}}\n")?;
	}
	Ok(())
}

/// Writes the verbose OpenAI-Harmony `namespace functions` inventory.
///
/// Descriptions and examples are comment-prefixed and each declaration uses a
/// TypeScript projection of the borrowed JSON Schema.
pub fn write_harmony_inventory<W: fmt::Write + ?Sized>(
	out: &mut W,
	tools: &[InbandTool<'_>],
) -> fmt::Result {
	if tools.is_empty() {
		return Ok(());
	}
	out.write_str("## functions\n\nnamespace functions {\n")?;
	for tool in tools {
		out.write_str("\n")?;
		if let Some(description) = tool.description.filter(|text| !text.is_empty()) {
			write_comment_lines(out, description)?;
		}
		if !tool.examples.is_empty() {
			if tool.description.is_some_and(|text| !text.is_empty()) {
				out.write_str("//\n")?;
			}
			write_tool_examples_jsdoc(out, tool)?;
		}
		write!(out, "type {} = (", tool.name)?;
		if !schema_is_empty_object(tool.parameters) {
			out.write_str("_: ")?;
			write_schema_type(out, tool.parameters)?;
		}
		out.write_str(");\n")?;
	}
	out.write_str("\n} // namespace functions")
}

/// Writes portable `<examples>` guidance for one tool.
///
/// Python keyword syntax is intentionally stable across target dialects. When
/// `intent_field` is present, its placeholder is emitted first.
pub fn write_tool_examples<W: fmt::Write + ?Sized>(
	out: &mut W,
	tool: &InbandTool<'_>,
	intent_field: Option<&str>,
) -> fmt::Result {
	if tool.examples.is_empty() {
		return Ok(());
	}
	out.write_str("<examples>\n")?;
	for (index, example) in tool.examples.iter().enumerate() {
		if index != 0 {
			out.write_str("\n")?;
		}
		match example {
			ToolExample::Call { caption, arguments } => {
				write_caption(out, *caption)?;
				write_example_call(out, tool.name, arguments, intent_field)?;
			},
			ToolExample::Contrast { caption, bad, good } => {
				write_caption(out, *caption)?;
				out.write_str("WRONG:\n")?;
				write_example_call(out, tool.name, bad, intent_field)?;
				out.write_str("\nRIGHT:\n")?;
				write_example_call(out, tool.name, good, intent_field)?;
			},
			ToolExample::Note { caption, note } => {
				write_caption(out, *caption)?;
				out.write_str(note)?;
			},
		}
	}
	out.write_str("\n</examples>")
}

fn write_tool_examples_jsdoc<W: fmt::Write + ?Sized>(
	out: &mut W,
	tool: &InbandTool<'_>,
) -> fmt::Result {
	for example in tool.examples {
		let caption = match example {
			ToolExample::Call { caption, .. }
			| ToolExample::Contrast { caption, .. }
			| ToolExample::Note { caption, .. } => *caption,
		};
		out.write_str("// @example")?;
		if let Some(caption) = caption {
			out.write_str(" ")?;
			write_json_string(out, caption)?;
		}
		out.write_str("\n")?;
		match example {
			ToolExample::Call { arguments, .. } => {
				out.write_str("// ")?;
				write_bare_or_py_call(out, tool.name, arguments)?;
				out.write_str("\n")?;
			},
			ToolExample::Contrast { bad, good, .. } => {
				out.write_str("// WRONG:\n// ")?;
				write_bare_or_py_call(out, tool.name, bad)?;
				out.write_str("\n// RIGHT:\n// ")?;
				write_bare_or_py_call(out, tool.name, good)?;
				out.write_str("\n")?;
			},
			ToolExample::Note { note, .. } => write_comment_lines(out, note)?,
		}
	}
	Ok(())
}

fn write_example_call<W: fmt::Write + ?Sized>(
	out: &mut W,
	name: &str,
	arguments: &Value,
	intent_field: Option<&str>,
) -> fmt::Result {
	if let Some(text) = sole_string_arg(arguments) {
		out.write_str("<example")?;
		if let Some(field) = intent_field {
			write!(out, " {field}=\"…\"")?;
		}
		out.write_str(">\n")?;
		out.write_str(text)?;
		return out.write_str("\n</example>");
	}
	out.write_str("<example>\n")?;
	write_py_call_value(out, name, arguments, intent_field.map(|field| (field, "…")))?;
	out.write_str("\n</example>")
}

fn write_bare_or_py_call<W: fmt::Write + ?Sized>(
	out: &mut W,
	name: &str,
	arguments: &Value,
) -> fmt::Result {
	if let Some(text) = sole_string_arg(arguments) {
		out.write_str(text)
	} else {
		write_py_call_value(out, name, arguments, None)
	}
}

fn sole_string_arg(arguments: &Value) -> Option<&str> {
	let object = arguments.as_object()?;
	if object.len() != 1 {
		return None;
	}
	object.values().next()?.as_str()
}

fn write_caption<W: fmt::Write + ?Sized>(out: &mut W, caption: Option<&str>) -> fmt::Result {
	if let Some(caption) = caption {
		writeln!(out, "# {caption}")?;
	}
	Ok(())
}

fn write_comment_lines<W: fmt::Write + ?Sized>(out: &mut W, text: &str) -> fmt::Result {
	for line in text.split('\n') {
		out.write_str("//")?;
		if !line.is_empty() {
			out.write_str(" ")?;
			out.write_str(line.trim_end())?;
		}
		out.write_str("\n")?;
	}
	Ok(())
}

fn schema_is_empty_object(schema: &Value) -> bool {
	let Some(object) = schema.as_object() else {
		return false;
	};
	matches!(object.get("type").and_then(Value::as_str), Some("object"))
		&& object
			.get("properties")
			.and_then(Value::as_object)
			.is_none_or(Map::is_empty)
}

fn write_schema_type<W: fmt::Write + ?Sized>(out: &mut W, schema: &Value) -> fmt::Result {
	if let Some(values) = schema.get("enum").and_then(Value::as_array) {
		for (index, value) in values.iter().enumerate() {
			if index != 0 {
				out.write_str(" | ")?;
			}
			write_json_value(out, value)?;
		}
		return Ok(());
	}
	for keyword in ["oneOf", "anyOf"] {
		if let Some(branches) = schema.get(keyword).and_then(Value::as_array) {
			for (index, branch) in branches.iter().enumerate() {
				if index != 0 {
					out.write_str(" | ")?;
				}
				write_schema_type(out, branch)?;
			}
			return Ok(());
		}
	}
	if let Some(branches) = schema.get("allOf").and_then(Value::as_array) {
		for (index, branch) in branches.iter().enumerate() {
			if index != 0 {
				out.write_str(" & ")?;
			}
			write_schema_type(out, branch)?;
		}
		return Ok(());
	}
	match schema.get("type").and_then(Value::as_str) {
		Some("null") => out.write_str("null"),
		Some("boolean") => out.write_str("boolean"),
		Some("integer" | "number") => out.write_str("number"),
		Some("string") => out.write_str("string"),
		Some("array") => {
			out.write_str("Array<")?;
			if let Some(items) = schema.get("items") {
				write_schema_type(out, items)?;
			} else {
				out.write_str("unknown")?;
			}
			out.write_str(">")
		},
		Some("object") | None if schema.get("properties").is_some() => {
			write_object_schema(out, schema)
		},
		_ => out.write_str("unknown"),
	}
}

fn write_object_schema<W: fmt::Write + ?Sized>(out: &mut W, schema: &Value) -> fmt::Result {
	let properties = schema.get("properties").and_then(Value::as_object);
	let required = schema.get("required").and_then(Value::as_array);
	out.write_str("{")?;
	if let Some(properties) = properties {
		for (index, (name, property)) in properties.iter().enumerate() {
			if index != 0 {
				out.write_str("; ")?;
			}
			if is_identifier(name) {
				out.write_str(name)?;
			} else {
				write_json_string(out, name)?;
			}
			let is_required =
				required.is_some_and(|items| items.iter().any(|item| item.as_str() == Some(name)));
			if !is_required {
				out.write_str("?")?;
			}
			out.write_str(": ")?;
			write_schema_type(out, property)?;
		}
	}
	out.write_str(" }")
}

fn is_identifier(name: &str) -> bool {
	let mut chars = name.chars();
	matches!(chars.next(), Some('_' | '$' | 'a'..='z' | 'A'..='Z'))
		&& chars.all(|ch| matches!(ch, '_' | '$' | 'a'..='z' | 'A'..='Z' | '0'..='9'))
}

fn write_json_string<W: fmt::Write + ?Sized>(out: &mut W, value: &str) -> fmt::Result {
	let mut serializer = serde_json::Serializer::new(FmtIo(out));
	serde::Serializer::serialize_str(&mut serializer, value).map_err(|_| fmt::Error)
}

struct FmtIo<'a, W: fmt::Write + ?Sized>(&'a mut W);

impl<W: fmt::Write + ?Sized> io::Write for FmtIo<'_, W> {
	fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
		let text = std::str::from_utf8(bytes).map_err(io::Error::other)?;
		self
			.0
			.write_str(text)
			.map_err(|_| io::Error::other("format destination failed"))?;
		Ok(bytes.len())
	}

	fn flush(&mut self) -> io::Result<()> {
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use serde_json::json;

	use super::*;

	#[test]
	fn json_lines_and_harmony_inventory_include_schema_examples() {
		let schema = json!({
			"type": "object",
			"properties": {
				"path": {"type": "string"},
				"count": {"type": "integer"}
			},
			"required": ["path"]
		});
		let arguments = json!({"path": "src/lib.rs", "count": 2});
		let examples = [ToolExample::Call { caption: Some("read source"), arguments: &arguments }];
		let tools = [InbandTool {
			name:        "read",
			description: Some("Read a file."),
			parameters:  &schema,
			examples:    &examples,
		}];

		let mut lines = String::new();
		write_json_line_catalog(&mut lines, &tools).unwrap();
		assert_eq!(
			lines,
			concat!(
				r#"{"type":"function","function":{"name":"read","description":"Read a file.","parameters":{"type":"object","properties":{"path":{"type":"string"},"count":{"type":"integer"}},"required":["path"]}}}"#,
				"\n"
			)
		);

		let mut harmony = String::new();
		write_harmony_inventory(&mut harmony, &tools).unwrap();
		assert!(harmony.contains("namespace functions"));
		assert!(harmony.contains("// @example \"read source\""));
		assert!(harmony.contains("read(path=\"src/lib.rs\", count=2)"));
		assert!(harmony.contains("type read = (_: {path: string; count?: number });"));
	}
}
