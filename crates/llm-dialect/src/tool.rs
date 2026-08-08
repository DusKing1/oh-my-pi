//! Allocation-bounded parsers for model-authored tool-call bodies.

use omp_core::SmolStr;
use serde_json::{Map, Number, Value};
use smallvec::SmallVec;

/// Parses every complete Python call expression in a Gemini `tool_code` body.
pub(crate) fn parse_python_calls(body: &[u8]) -> SmallVec<(SmolStr, Map<String, Value>), 2> {
	let Ok(body) = std::str::from_utf8(body) else {
		return SmallVec::new();
	};
	let bytes = body.as_bytes();
	let mut calls = SmallVec::new();
	let mut index = 0;
	while index < bytes.len() {
		match bytes[index] {
			b'\'' | b'"' => index = skip_python_string(bytes, index).unwrap_or(bytes.len()),
			b'#' => index = skip_comment(bytes, index),
			b'(' => {
				let Some(name) = identifier_before(bytes, index) else {
					index += 1;
					continue;
				};
				if name == "print" {
					index += 1;
					continue;
				}
				let Some(end) = matching_python_delimiter(bytes, index, b'(', b')') else {
					index += 1;
					continue;
				};
				if let Some(arguments) = parse_python_arguments(&body[index + 1..end]) {
					calls.push((SmolStr::from(name), arguments));
				}
				index = end + 1;
			},
			_ => index += 1,
		}
	}
	calls
}

fn parse_python_arguments(text: &str) -> Option<Map<String, Value>> {
	let text = strip_python_comments(text);
	let mut arguments = Map::new();
	for segment in split_top_level(&text, b',')? {
		let segment = segment.trim();
		if segment.is_empty() {
			continue;
		}
		let Some(equal) = top_level_index(segment, b'=')? else {
			continue;
		};
		let name = segment[..equal].trim();
		if !is_identifier(name.as_bytes()) {
			continue;
		}
		arguments.insert(name.to_owned(), parse_python_value(segment[equal + 1..].trim())?);
	}
	Some(arguments)
}

fn parse_python_value(text: &str) -> Option<Value> {
	let text = text.trim();
	match text {
		"True" | "true" => return Some(Value::Bool(true)),
		"False" | "false" => return Some(Value::Bool(false)),
		"None" | "null" => return Some(Value::Null),
		_ => {},
	}
	if python_string_prefix(text).is_some() {
		return decode_python_string(text).map(Value::String);
	}
	if text.starts_with('[') {
		if matching_python_delimiter(text.as_bytes(), 0, b'[', b']')? != text.len() - 1 {
			return None;
		}
		let mut values = Vec::new();
		for item in split_top_level(&text[1..text.len() - 1], b',')? {
			if !item.trim().is_empty() {
				values.push(parse_python_value(item)?);
			}
		}
		return Some(Value::Array(values));
	}
	if text.starts_with('{') {
		if matching_python_delimiter(text.as_bytes(), 0, b'{', b'}')? != text.len() - 1 {
			return None;
		}
		let mut values = Map::new();
		for item in split_top_level(&text[1..text.len() - 1], b',')? {
			let item = item.trim();
			if item.is_empty() {
				continue;
			}
			let colon = top_level_index(item, b':')??;
			let raw_key = item[..colon].trim();
			let key = if python_string_prefix(raw_key).is_some() {
				decode_python_string(raw_key)?
			} else if is_identifier(raw_key.as_bytes()) {
				raw_key.to_owned()
			} else {
				return None;
			};
			values.insert(key, parse_python_value(item[colon + 1..].trim())?);
		}
		return Some(Value::Object(values));
	}
	parse_number(text).map(Value::Number)
}

fn parse_number(text: &str) -> Option<Number> {
	if text.is_empty() || !matches!(text.as_bytes()[0], b'+' | b'-' | b'.' | b'0'..=b'9') {
		return None;
	}
	if !text
		.bytes()
		.all(|byte| matches!(byte, b'+' | b'-' | b'.' | b'e' | b'E' | b'0'..=b'9'))
	{
		return None;
	}
	if !text.bytes().any(|byte| matches!(byte, b'.' | b'e' | b'E')) {
		if let Ok(value) = text.parse::<i64>() {
			return Some(value.into());
		}
		if let Ok(value) = text.parse::<u64>() {
			return Some(value.into());
		}
	}
	text.parse::<f64>().ok().and_then(Number::from_f64)
}

fn python_string_prefix(text: &str) -> Option<usize> {
	[2, 1, 0].into_iter().find(|&length| {
		let Some(prefix) = text.get(..length) else {
			return false;
		};
		matches!(prefix.to_ascii_lowercase().as_str(), "" | "r" | "u" | "b" | "br" | "rb")
			&& text
				.as_bytes()
				.get(length)
				.is_some_and(|byte| matches!(*byte, b'\'' | b'"'))
	})
}

fn decode_python_string(text: &str) -> Option<String> {
	let prefix = python_string_prefix(text)?;
	let raw = text[..prefix]
		.bytes()
		.any(|byte| matches!(byte, b'r' | b'R'));
	let quote = *text.as_bytes().get(prefix)?;
	let triple = text
		.as_bytes()
		.get(prefix..prefix + 3)
		.is_some_and(|run| run.iter().all(|byte| *byte == quote));
	let width = if triple { 3 } else { 1 };
	if skip_python_string(text.as_bytes(), prefix)? != text.len() {
		return None;
	}
	if text.len() < prefix + width * 2
		|| !text.as_bytes()[text.len() - width..]
			.iter()
			.all(|byte| *byte == quote)
	{
		return None;
	}
	let inner = &text[prefix + width..text.len() - width];
	if raw {
		return Some(inner.to_owned());
	}
	let bytes = inner.as_bytes();
	let mut output = String::with_capacity(inner.len());
	let mut index = 0;
	while index < bytes.len() {
		if bytes[index] != b'\\' {
			let ch = inner[index..].chars().next()?;
			output.push(ch);
			index += ch.len_utf8();
			continue;
		}
		index += 1;
		let Some(&escape) = bytes.get(index) else {
			output.push('\\');
			break;
		};
		if matches!(escape, b'0'..=b'7') {
			let start = index;
			while index < bytes.len() && index < start + 3 && matches!(bytes[index], b'0'..=b'7') {
				index += 1;
			}
			let value = u32::from_str_radix(&inner[start..index], 8).ok()?;
			output.push(char::from_u32(value)?);
			continue;
		}
		match escape {
			b'n' => output.push('\n'),
			b'r' => output.push('\r'),
			b't' => output.push('\t'),
			b'\\' => output.push('\\'),
			b'\'' => output.push('\''),
			b'"' => output.push('"'),
			b'x' | b'u' | b'U' => {
				let digits = match escape {
					b'x' => 2,
					b'u' => 4,
					_ => 8,
				};
				let start = index + 1;
				let end = start + digits;
				let hex = inner.get(start..end)?;
				if !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
					return None;
				}
				output.push(char::from_u32(u32::from_str_radix(hex, 16).ok()?)?);
				index = end;
				continue;
			},
			other => output.push(other as char),
		}
		index += 1;
	}
	Some(output)
}

fn skip_python_string(bytes: &[u8], start: usize) -> Option<usize> {
	let quote = *bytes.get(start)?;
	let triple = bytes
		.get(start..start + 3)
		.is_some_and(|run| run.iter().all(|byte| *byte == quote));
	let width = if triple { 3 } else { 1 };
	let mut index = start + width;
	while index < bytes.len() {
		if !triple && bytes[index] == b'\\' {
			index = (index + 2).min(bytes.len());
			continue;
		}
		if bytes
			.get(index..index + width)
			.is_some_and(|run| run.iter().all(|byte| *byte == quote))
		{
			return Some(index + width);
		}
		index += 1;
	}
	None
}

fn matching_python_delimiter(bytes: &[u8], open: usize, opening: u8, closing: u8) -> Option<usize> {
	let mut depth = 0usize;
	let mut index = open;
	while index < bytes.len() {
		match bytes[index] {
			b'\'' | b'"' => index = skip_python_string(bytes, index)?,
			b'#' => index = skip_comment(bytes, index),
			byte if byte == opening => {
				depth += 1;
				index += 1;
			},
			byte if byte == closing => {
				depth = depth.checked_sub(1)?;
				if depth == 0 {
					return Some(index);
				}
				index += 1;
			},
			_ => index += 1,
		}
	}
	None
}

fn skip_comment(bytes: &[u8], start: usize) -> usize {
	bytes[start..]
		.iter()
		.position(|byte| *byte == b'\n')
		.map_or(bytes.len(), |offset| start + offset + 1)
}

fn strip_python_comments(text: &str) -> String {
	let bytes = text.as_bytes();
	let mut output = String::with_capacity(text.len());
	let mut index = 0;
	while index < bytes.len() {
		match bytes[index] {
			b'\'' | b'"' => {
				let end = skip_python_string(bytes, index).unwrap_or(bytes.len());
				output.push_str(&text[index..end]);
				index = end;
			},
			b'#' => {
				let end = skip_comment(bytes, index);
				if end <= bytes.len() && bytes.get(end.wrapping_sub(1)) == Some(&b'\n') {
					output.push('\n');
				}
				index = end;
			},
			_ => {
				let ch = text[index..]
					.chars()
					.next()
					.expect("index is at a UTF-8 boundary");
				output.push(ch);
				index += ch.len_utf8();
			},
		}
	}
	output
}

fn split_top_level(text: &str, separator: u8) -> Option<SmallVec<&str, 8>> {
	let bytes = text.as_bytes();
	let mut parts = SmallVec::new();
	let mut stack = SmallVec::<u8, 8>::new();
	let mut start = 0;
	let mut index = 0;
	while index < bytes.len() {
		match bytes[index] {
			b'\'' | b'"' => index = skip_python_string(bytes, index)?,
			b'(' | b'[' | b'{' => {
				stack.push(bytes[index]);
				index += 1;
			},
			b')' | b']' | b'}' => {
				let expected = match bytes[index] {
					b')' => b'(',
					b']' => b'[',
					_ => b'{',
				};
				if stack.pop() != Some(expected) {
					return None;
				}
				index += 1;
			},
			byte if byte == separator && stack.is_empty() => {
				parts.push(&text[start..index]);
				start = index + 1;
				index += 1;
			},
			_ => index += 1,
		}
	}
	if !stack.is_empty() {
		return None;
	}
	parts.push(&text[start..]);
	Some(parts)
}

fn top_level_index(text: &str, target: u8) -> Option<Option<usize>> {
	let bytes = text.as_bytes();
	let mut stack = SmallVec::<u8, 8>::new();
	let mut index = 0;
	while index < bytes.len() {
		match bytes[index] {
			b'\'' | b'"' => index = skip_python_string(bytes, index)?,
			b'(' | b'[' | b'{' => {
				stack.push(bytes[index]);
				index += 1;
			},
			b')' | b']' | b'}' => {
				let expected = match bytes[index] {
					b')' => b'(',
					b']' => b'[',
					_ => b'{',
				};
				if stack.pop() != Some(expected) {
					return None;
				}
				index += 1;
			},
			byte if byte == target && stack.is_empty() => return Some(Some(index)),
			_ => index += 1,
		}
	}
	if stack.is_empty() { Some(None) } else { None }
}

fn identifier_before(bytes: &[u8], open: usize) -> Option<&str> {
	let mut end = open;
	while end > 0 && bytes[end - 1].is_ascii_whitespace() {
		end -= 1;
	}
	let mut start = end;
	while start > 0 && (bytes[start - 1].is_ascii_alphanumeric() || bytes[start - 1] == b'_') {
		start -= 1;
	}
	let name = std::str::from_utf8(&bytes[start..end]).ok()?;
	is_identifier(name.as_bytes()).then_some(name)
}

fn is_identifier(bytes: &[u8]) -> bool {
	bytes
		.first()
		.is_some_and(|byte| byte.is_ascii_alphabetic() || *byte == b'_')
		&& bytes
			.iter()
			.all(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
}

/// Parses one complete Gemma token-delimited tool body.
pub(crate) fn parse_gemma_call(body: &[u8]) -> Option<(SmolStr, Map<String, Value>)> {
	let Ok(body) = std::str::from_utf8(body) else {
		return None;
	};
	let body = body.trim();
	let body = body.strip_prefix("call:")?.trim_start();
	let brace = body.find('{')?;
	let name = body[..brace].trim();
	if !is_identifier(name.as_bytes()) {
		return None;
	}
	let end = matching_gemma_delimiter(body.as_bytes(), brace, b'{', b'}')?;
	if !body[end + 1..].trim().is_empty() {
		return None;
	}
	Some((SmolStr::from(name), parse_gemma_arguments(&body[brace + 1..end])?))
}

fn parse_gemma_arguments(text: &str) -> Option<Map<String, Value>> {
	let mut arguments = Map::new();
	for segment in split_gemma_top_level(text, b',')? {
		let segment = segment.trim();
		if segment.is_empty() {
			continue;
		}
		let colon = gemma_top_level_index(segment, b':')?;
		let name = segment[..colon].trim();
		if !is_identifier(name.as_bytes()) {
			return None;
		}
		arguments.insert(name.to_owned(), parse_gemma_value(segment[colon + 1..].trim())?);
	}
	Some(arguments)
}

const GEMMA_STRING: &[u8] = b"<|\"|>";

fn parse_gemma_value(text: &str) -> Option<Value> {
	let text = text.trim();
	if text.as_bytes().starts_with(GEMMA_STRING) {
		let rest = &text[GEMMA_STRING.len()..];
		let close = rest
			.as_bytes()
			.windows(GEMMA_STRING.len())
			.position(|window| window == GEMMA_STRING)?;
		if !rest[close + GEMMA_STRING.len()..].trim().is_empty() {
			return None;
		}
		return Some(Value::String(rest[..close].to_owned()));
	}
	match text {
		"true" => return Some(Value::Bool(true)),
		"false" => return Some(Value::Bool(false)),
		"null" | "none" | "None" => return Some(Value::Null),
		_ => {},
	}
	if text.starts_with('[') {
		let end = matching_gemma_delimiter(text.as_bytes(), 0, b'[', b']')?;
		if end != text.len() - 1 {
			return None;
		}
		let mut values = Vec::new();
		for item in split_gemma_top_level(&text[1..end], b',')? {
			if !item.trim().is_empty() {
				values.push(parse_gemma_value(item)?);
			}
		}
		return Some(Value::Array(values));
	}
	if text.starts_with('{') {
		let end = matching_gemma_delimiter(text.as_bytes(), 0, b'{', b'}')?;
		if end != text.len() - 1 {
			return None;
		}
		return parse_gemma_arguments(&text[1..end]).map(Value::Object);
	}
	parse_number(text).map(Value::Number)
}

fn skip_gemma_string(bytes: &[u8], start: usize) -> Option<usize> {
	if !bytes.get(start..)?.starts_with(GEMMA_STRING) {
		return None;
	}
	let content = start + GEMMA_STRING.len();
	let close = bytes[content..]
		.windows(GEMMA_STRING.len())
		.position(|window| window == GEMMA_STRING)?;
	Some(content + close + GEMMA_STRING.len())
}

fn matching_gemma_delimiter(bytes: &[u8], open: usize, opening: u8, closing: u8) -> Option<usize> {
	let mut depth = 0usize;
	let mut index = open;
	while index < bytes.len() {
		if bytes[index..].starts_with(GEMMA_STRING) {
			index = skip_gemma_string(bytes, index)?;
			continue;
		}
		if bytes[index] == opening {
			depth += 1;
		} else if bytes[index] == closing {
			depth = depth.checked_sub(1)?;
			if depth == 0 {
				return Some(index);
			}
		}
		index += 1;
	}
	None
}

fn split_gemma_top_level(text: &str, separator: u8) -> Option<SmallVec<&str, 8>> {
	let bytes = text.as_bytes();
	let mut parts = SmallVec::new();
	let mut stack = SmallVec::<u8, 8>::new();
	let mut start = 0;
	let mut index = 0;
	while index < bytes.len() {
		if bytes[index..].starts_with(GEMMA_STRING) {
			index = skip_gemma_string(bytes, index)?;
			continue;
		}
		match bytes[index] {
			b'(' | b'[' | b'{' => {
				stack.push(bytes[index]);
				index += 1;
			},
			b')' | b']' | b'}' => {
				let expected = match bytes[index] {
					b')' => b'(',
					b']' => b'[',
					_ => b'{',
				};
				if stack.pop() != Some(expected) {
					return None;
				}
				index += 1;
			},
			byte if byte == separator && stack.is_empty() => {
				parts.push(&text[start..index]);
				start = index + 1;
				index += 1;
			},
			_ => index += 1,
		}
	}
	if !stack.is_empty() {
		return None;
	}
	parts.push(&text[start..]);
	Some(parts)
}

fn gemma_top_level_index(text: &str, target: u8) -> Option<usize> {
	let bytes = text.as_bytes();
	let mut stack = SmallVec::<u8, 8>::new();
	let mut index = 0;
	while index < bytes.len() {
		if bytes[index..].starts_with(GEMMA_STRING) {
			index = skip_gemma_string(bytes, index)?;
			continue;
		}
		match bytes[index] {
			b'(' | b'[' | b'{' => {
				stack.push(bytes[index]);
				index += 1;
			},
			b')' | b']' | b'}' => {
				let expected = match bytes[index] {
					b')' => b'(',
					b']' => b'[',
					_ => b'{',
				};
				if stack.pop() != Some(expected) {
					return None;
				}
				index += 1;
			},
			byte if byte == target && stack.is_empty() => return Some(index),
			_ => index += 1,
		}
	}
	None
}
