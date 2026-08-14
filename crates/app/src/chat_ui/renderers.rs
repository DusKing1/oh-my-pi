use std::{fmt::Write as _, time::Instant};

use bytes::Bytes;
use omp_core::Str;
use omp_proto::thread::v1::{Item, item};
use omp_tool::Rev;
use omp_tui::{
	Ui,
	components::{Col, DiffKind, DiffView, TextLeaf, ToolCard, ToolState},
};
use serde::de::DeserializeOwned;
use xutf::{IntoAnsiStripped as _, TextBuf as _, Utf8};

/// The retained state for a single tool call.
#[derive(Debug)]
pub struct ToolFold {
	pub(crate) call_id:     Str,
	pub(crate) name:        Str,
	pub(crate) rev:         Rev,
	pub(crate) parsed_args: omp_slopjson::Value,
	pub(crate) updates:     Vec<serde_json::Value>,
	pub(crate) item:        Option<Item>,
	pub(crate) start_time:  Instant,
	pub(crate) state:       ToolState,
}

impl ToolFold {
	pub(crate) fn new(call_id: Str, name: Str, rev: Rev) -> Self {
		Self {
			call_id,
			name,
			rev,
			parsed_args: omp_slopjson::Value::Object(omp_slopjson::Object::new()),
			updates: Vec::new(),
			item: None,
			start_time: Instant::now(),
			state: ToolState::Streaming,
		}
	}

	pub(crate) fn set_args_view(&mut self, view: omp_slopjson::Value) {
		self.parsed_args = view;
	}

	pub(crate) fn push_update(&mut self, json: Bytes) {
		if let Ok(val) = serde_json::from_slice(&json) {
			self.updates.push(val);
		}
	}
}

pub type RendererFn = fn(&mut Ui, &ToolFold) -> bool;

pub struct RendererRegistry {
	renderers: &'static [(&'static str, &'static str, u16, RendererFn)],
}

impl RendererRegistry {
	pub(crate) const fn new() -> Self {
		Self {
			renderers: &[
				("read", "", 1, render_read),
				("edit", "hl", 1, render_edit),
				("shell", "", 1, render_shell),
				("eval", "", 1, render_eval),
				("grep", "", 1, render_grep),
				("glob", "", 1, render_glob),
				("write", "", 1, render_write),
			],
		}
	}

	pub(crate) fn update(&self, ui: &mut Ui, fold: &ToolFold) -> bool {
		for (name, family, n, f) in self.renderers {
			if fold.name.as_str() == *name && fold.rev.family.as_str() == *family && fold.rev.n == *n {
				return f(ui, fold);
			}
		}
		render_generic(ui, fold)
	}
}

fn proto_to_json(v: &omp_proto::inference::v1::Value) -> Option<serde_json::Value> {
	match v.kind.as_ref()? {
		omp_proto::inference::v1::value::Kind::Null(_) => Some(serde_json::Value::Null),
		omp_proto::inference::v1::value::Kind::Int(n) => Some(serde_json::Value::from(*n)),
		omp_proto::inference::v1::value::Kind::Uint(n) => Some(serde_json::Value::from(*n)),
		omp_proto::inference::v1::value::Kind::Double(n) => {
			serde_json::Number::from_f64(*n).map(serde_json::Value::Number)
		},
		omp_proto::inference::v1::value::Kind::Bool(b) => Some(serde_json::Value::Bool(*b)),
		omp_proto::inference::v1::value::Kind::String(s) => {
			Some(serde_json::Value::String(s.clone()))
		},
		omp_proto::inference::v1::value::Kind::List(list) => {
			let mut out = Vec::with_capacity(list.values.len());
			for i in &list.values {
				out.push(proto_to_json(i)?);
			}
			Some(serde_json::Value::Array(out))
		},
		omp_proto::inference::v1::value::Kind::Map(m) => {
			let mut map = serde_json::Map::with_capacity(m.fields.len());
			for (k, v) in &m.fields {
				map.insert(k.clone(), proto_to_json(v)?);
			}
			Some(serde_json::Value::Object(map))
		},
	}
}

fn get_result_json(item: Option<&Item>) -> Option<serde_json::Value> {
	let i = item?;
	let item::Kind::ToolResult(res) = &i.kind.as_ref()? else {
		return None;
	};
	let details = res.details.as_ref()?;
	proto_to_json(details)
}

fn get_verdict<P: DeserializeOwned, F: DeserializeOwned>(
	item: Option<&Item>,
) -> Option<omp_tool::Verdict<P, F>> {
	let json_val = get_result_json(item)?;
	serde_json::from_value(json_val).ok()
}

fn render_verdict_fallback<P, F: serde::Serialize>(
	c: &mut ToolCard,
	verdict: &omp_tool::Verdict<P, F>,
) {
	match verdict {
		omp_tool::Verdict::Fault(f) => {
			let text = serde_json::to_string_pretty(f).unwrap_or_default();
			c.replace_body(Col::new().child(TextLeaf::new().text(format!("Fault:\n{text}"))));
		},
		omp_tool::Verdict::Args(a) => {
			let text = serde_json::to_string_pretty(a).unwrap_or_default();
			c.replace_body(Col::new().child(TextLeaf::new().text(format!("Arg Issue:\n{text}"))));
		},
		omp_tool::Verdict::Aborted(a) => {
			let text = serde_json::to_string_pretty(a).unwrap_or_default();
			c.replace_body(Col::new().child(TextLeaf::new().text(format!("Aborted:\n{text}"))));
		},
		_ => {},
	}
}

fn parse_diff_lines(diff: &mut DiffView, preview: &str) {
	for line in preview.lines() {
		if line.starts_with('+') && !line.starts_with("+++") {
			diff.push(DiffKind::Add, line);
		} else if line.starts_with('-') && !line.starts_with("---") {
			diff.push(DiffKind::Remove, line);
		} else if line.starts_with('@') || line.starts_with("---") || line.starts_with("+++") {
			diff.push(DiffKind::Header, line);
		} else {
			diff.push(DiffKind::Context, line);
		}
	}
}
fn edit_input_path(input: &str) -> Option<&str> {
	for line in input.lines() {
		let header = line.trim();
		let Some(body) = header
			.strip_prefix('[')
			.and_then(|value| value.strip_suffix(']'))
		else {
			continue;
		};
		let Some((path, tag)) = body.rsplit_once('#') else {
			continue;
		};
		if !path.is_empty() && !tag.is_empty() {
			return Some(path);
		}
	}
	None
}
fn append_edit_section(diff: &mut DiffView, section: &omp_tools::edit::SectionPayload) {
	diff.push(DiffKind::Header, section.header.as_deref().unwrap_or(section.path.as_str()));
	if let Some(destination) = &section.move_dest {
		diff.push(DiffKind::Header, format!("Moved to {destination}"));
	}
	parse_diff_lines(diff, section.diff.as_str());
	for warning in &section.warnings {
		diff.push(DiffKind::Context, format!("Warning: {warning}"));
	}
}

pub fn render_read(ui: &mut Ui, fold: &ToolFold) -> bool {
	ui.update_component::<ToolCard>(&fold.call_id, |c| {
		c.set_name(fold.name.clone());
		c.set_state(fold.state);
		if fold.state != ToolState::Streaming {
			c.set_badge(format!("{}ms", fold.start_time.elapsed().as_millis()));
		}

		if let Some(path) = fold.parsed_args.get("path").and_then(|v| v.as_str()) {
			c.set_intent(format!("read {path}"));
		}

		if fold.state != ToolState::Streaming
			&& let Some(verdict) =
				get_verdict::<omp_tools::read::Payload, serde_json::Value>(fold.item.as_ref())
		{
			match &verdict {
				omp_tool::Verdict::Ok(payload) => {
					let mut col = Col::new();
					for part in &payload.parts {
						let text = match part {
							omp_tools::read::PayloadPart::Text { text } => text,
							omp_tools::read::PayloadPart::Blob { alt, .. } => alt,
						};
						col = col.child(TextLeaf::new().text(text.as_str()));
					}
					c.replace_body(col);
				},
				_ => render_verdict_fallback(c, &verdict),
			}
		}
		true
	})
}

pub fn render_edit(ui: &mut Ui, fold: &ToolFold) -> bool {
	ui.update_component::<ToolCard>(&fold.call_id, |c| {
		c.set_name(fold.name.clone());
		c.set_state(fold.state);
		if fold.state != ToolState::Streaming {
			c.set_badge(format!("{}ms", fold.start_time.elapsed().as_millis()));
		}

		if let Some(path) = fold
			.parsed_args
			.get("input")
			.and_then(|value| value.as_str())
			.and_then(edit_input_path)
		{
			c.set_intent(format!("edit {path}"));
		}

		if fold.state == ToolState::Streaming {
			let preview = fold
				.updates
				.last()
				.and_then(|update| update.get("preview"))
				.and_then(|value| value.as_str())
				.or_else(|| {
					fold
						.parsed_args
						.get("input")
						.and_then(|value| value.as_str())
				});
			if let Some(preview) = preview {
				let mut diff = DiffView::new();
				parse_diff_lines(&mut diff, preview);
				c.replace_body(diff);
			}
		} else if let Some(verdict) =
			get_verdict::<omp_tools::edit::Payload, serde_json::Value>(fold.item.as_ref())
		{
			match &verdict {
				omp_tool::Verdict::Ok(payload) => {
					let mut diff = DiffView::new();
					for section in &payload.sections {
						append_edit_section(&mut diff, section);
					}
					c.replace_body(diff);
				},
				_ => render_verdict_fallback(c, &verdict),
			}
		}
		true
	})
}

fn append_shell_update(output: &mut Vec<u8>, update: &serde_json::Value) {
	if let Some(text) = update.get("text").and_then(serde_json::Value::as_str) {
		output.extend_from_slice(text.as_bytes());
		return;
	}
	let Some(values) = update.get("data").and_then(serde_json::Value::as_array) else {
		return;
	};
	let start = output.len();
	for value in values {
		let Some(byte) = value.as_u64().and_then(|value| u8::try_from(value).ok()) else {
			output.truncate(start);
			return;
		};
		output.push(byte);
	}
}

pub fn render_shell(ui: &mut Ui, fold: &ToolFold) -> bool {
	ui.update_component::<ToolCard>(&fold.call_id, |c| {
		c.set_name(fold.name.clone());
		c.set_state(fold.state);

		if fold.state != ToolState::Streaming {
			c.set_badge(format!("{}ms", fold.start_time.elapsed().as_millis()));
		}

		if let Some(command) = fold.parsed_args.get("command").and_then(|v| v.as_str()) {
			c.set_intent(command.split('\n').next().unwrap_or(command));
		}

		let mut tail_bytes = Vec::new();
		let start = fold.updates.len().saturating_sub(10);
		for update in &fold.updates[start..] {
			append_shell_update(&mut tail_bytes, update);
		}
		let tail_text =
			String::from_units(xutf::transcode::<Utf8, Utf8>(&tail_bytes)).into_ansi_stripped();

		let mut rendered_fallback = false;
		if fold.state != ToolState::Streaming
			&& let Some(verdict) =
				get_verdict::<omp_tools::shell::Payload, serde_json::Value>(fold.item.as_ref())
		{
			match &verdict {
				omp_tool::Verdict::Ok(payload) => match payload.status.outcome {
					omp_tools::shell::ExecOutcome::Exited => {
						// Keep default/pre-set success state
					},
					omp_tools::shell::ExecOutcome::Failed => {
						c.set_state(ToolState::Failure);
						if let Some(code) = payload.status.exit_code {
							c.set_badge(format!("exit {code}"));
						} else {
							c.set_badge("failed");
						}
					},
					omp_tools::shell::ExecOutcome::Cancelled => {
						c.set_state(ToolState::Failure);
						c.set_badge("interrupted");
					},
					omp_tools::shell::ExecOutcome::Timeout => {
						c.set_state(ToolState::Failure);
						c.set_badge("timeout");
					},
					_ => {},
				},
				_ => {
					render_verdict_fallback(c, &verdict);
					rendered_fallback = true;
				},
			}
		}

		if tail_text.is_empty() {
			if !rendered_fallback {
				c.replace_body(Col::new());
			}
		} else {
			let lines = tail_text.lines();
			let first_preview_line = lines.clone().count().saturating_sub(5);
			let mut tail_preview = String::new();
			for line in lines.skip(first_preview_line) {
				if !tail_preview.is_empty() {
					tail_preview.push('\n');
				}
				tail_preview.push_str(line);
			}
			let col = Col::new().child(TextLeaf::new().text(tail_preview));
			c.replace_body(col);
		}
		true
	})
}

fn eval_payload_text(payload: &omp_tools::eval::Payload) -> String {
	let mut bytes = Vec::new();
	for frame in &payload.frames {
		bytes.extend_from_slice(&frame.data);
	}
	let mut text = String::from_units(xutf::transcode::<Utf8, Utf8>(&bytes)).into_ansi_stripped();
	if let Some(result) = &payload.result
		&& !result.text.is_empty()
	{
		text.push_str(&result.text);
		if !result.text.ends_with('\n') {
			text.push('\n');
		}
	}
	let json_count = payload
		.display_outputs
		.iter()
		.filter(|output| matches!(output, omp_tools::eval::DisplayOutput::Json { .. }))
		.count();
	let mut json_index = 0usize;
	for output in &payload.display_outputs {
		match output {
			omp_tools::eval::DisplayOutput::Json { data } => {
				json_index += 1;
				if !text.is_empty() && !text.ends_with('\n') {
					text.push('\n');
				}
				if json_count > 1 {
					writeln!(text, "display[{json_index}]").expect("writing to a String cannot fail");
				}
				text.push_str(&serde_json::to_string_pretty(data).unwrap_or_else(|_| data.to_string()));
				text.push('\n');
			},
			omp_tools::eval::DisplayOutput::Markdown { text: markdown } => {
				text.push_str(markdown);
				if !markdown.ends_with('\n') {
					text.push('\n');
				}
			},
			omp_tools::eval::DisplayOutput::Image { .. }
			| omp_tools::eval::DisplayOutput::Status { .. } => {},
		}
	}
	if let Some(exception) = &payload.status.exception {
		if exception.traceback.is_empty() {
			writeln!(text, "{}: {}", exception.name, exception.message)
				.expect("writing to a String cannot fail");
		} else {
			for line in &exception.traceback {
				text.push_str(line);
				if !line.ends_with('\n') {
					text.push('\n');
				}
			}
		}
	}
	if payload.truncated {
		if let Some(blob) = &payload.spilled_output {
			writeln!(text, "… output truncated; full output in blob {}", blob.hash)
				.expect("writing to a String cannot fail");
		} else {
			text.push_str("… output truncated\n");
		}
	}
	text.trim_end().to_owned()
}

fn eval_preview(text: &str, maximum_lines: usize) -> String {
	let lines = text.lines().collect::<Vec<_>>();
	let start = lines.len().saturating_sub(maximum_lines);
	let mut preview = lines[start..].join("\n");
	if start != 0 {
		preview = format!("… {start} earlier lines\n{preview}");
	}
	preview
}

pub fn render_eval(ui: &mut Ui, fold: &ToolFold) -> bool {
	ui.update_component::<ToolCard>(&fold.call_id, |c| {
		c.set_name(fold.name.clone());
		c.set_state(fold.state);
		let title = fold
			.parsed_args
			.get("title")
			.and_then(|value| value.as_str());
		let code = fold
			.parsed_args
			.get("code")
			.and_then(|value| value.as_str());
		if let Some(intent) = title.or_else(|| code.and_then(|code| code.lines().next())) {
			c.set_intent(intent);
		} else {
			c.set_intent("running Python");
		}

		if fold.state == ToolState::Streaming {
			let mut bytes = Vec::new();
			for update in &fold.updates {
				append_shell_update(&mut bytes, update);
			}
			let text = String::from_units(xutf::transcode::<Utf8, Utf8>(&bytes)).into_ansi_stripped();
			if !text.is_empty() {
				c.replace_body(Col::new().child(TextLeaf::new().text(eval_preview(&text, 10))));
			}
			return true;
		}

		if let Some(verdict) =
			get_verdict::<omp_tools::eval::Payload, omp_tools::eval::Fault>(fold.item.as_ref())
		{
			match &verdict {
				omp_tool::Verdict::Ok(payload) => {
					match payload.status.outcome {
						omp_tools::eval::CellOutcome::Complete => {
							c.set_badge(format!("{}ms", payload.status.duration_ms));
						},
						omp_tools::eval::CellOutcome::Error => {
							c.set_state(ToolState::Failure);
							if let Some(code) = payload.status.exit_code {
								c.set_badge(format!("exit {code}"));
							} else {
								c.set_badge("failed");
							}
						},
						omp_tools::eval::CellOutcome::Timeout => {
							c.set_state(ToolState::Failure);
							c.set_badge("timeout");
						},
						omp_tools::eval::CellOutcome::Cancelled => {
							c.set_state(ToolState::Failure);
							c.set_badge("interrupted");
						},
					}
					let text = eval_payload_text(payload);
					if text.is_empty() {
						c.replace_body(Col::new());
					} else {
						c.replace_body(Col::new().child(TextLeaf::new().text(eval_preview(&text, 10))));
					}
				},
				_ => render_verdict_fallback(c, &verdict),
			}
		}
		true
	})
}

pub fn render_grep(ui: &mut Ui, fold: &ToolFold) -> bool {
	ui.update_component::<ToolCard>(&fold.call_id, |c| {
		c.set_name(fold.name.clone());
		c.set_state(fold.state);
		if fold.state != ToolState::Streaming {
			c.set_badge(format!("{}ms", fold.start_time.elapsed().as_millis()));
		}

		if let Some(pattern) = fold.parsed_args.get("pattern").and_then(|v| v.as_str()) {
			c.set_intent(format!("grep {pattern}"));
		}

		if fold.state != ToolState::Streaming
			&& let Some(verdict) =
				get_verdict::<omp_tools::grep::Payload, serde_json::Value>(fold.item.as_ref())
		{
			match &verdict {
				omp_tool::Verdict::Ok(payload) => {
					let mut col = Col::new();
					for f in &payload.files {
						for m in &f.matches {
							col = col.child(TextLeaf::new().text(format!(
								"{}:{}",
								f.path.as_str(),
								m.line_number
							)));
						}
					}
					if payload.file_limit_reached || payload.per_file_limit_reached {
						col = col.child(TextLeaf::new().text("..."));
					}
					c.replace_body(col);
				},
				_ => render_verdict_fallback(c, &verdict),
			}
		}
		true
	})
}

pub fn render_glob(ui: &mut Ui, fold: &ToolFold) -> bool {
	ui.update_component::<ToolCard>(&fold.call_id, |c| {
		c.set_name(fold.name.clone());
		c.set_state(fold.state);
		if fold.state != ToolState::Streaming {
			c.set_badge(format!("{}ms", fold.start_time.elapsed().as_millis()));
		}

		if let Some(path) = fold.parsed_args.get("path").and_then(|v| v.as_str()) {
			c.set_intent(format!("glob {path}"));
		}

		if fold.state != ToolState::Streaming
			&& let Some(verdict) =
				get_verdict::<omp_tools::glob::Payload, serde_json::Value>(fold.item.as_ref())
		{
			match &verdict {
				omp_tool::Verdict::Ok(payload) => {
					let mut col = Col::new();
					for m in &payload.matches {
						col = col.child(TextLeaf::new().text(m.path.as_str()));
					}
					if payload.truncated {
						col = col.child(TextLeaf::new().text("..."));
					}
					c.replace_body(col);
				},
				_ => render_verdict_fallback(c, &verdict),
			}
		}
		true
	})
}

pub fn render_write(ui: &mut Ui, fold: &ToolFold) -> bool {
	ui.update_component::<ToolCard>(&fold.call_id, |c| {
		c.set_name(fold.name.clone());
		c.set_state(fold.state);

		if let Some(path) = fold
			.parsed_args
			.get("path")
			.and_then(|value| value.as_str())
		{
			c.set_intent(format!("write {path}"));
		}
		if let Some(content) = fold
			.parsed_args
			.get("content")
			.and_then(|value| value.as_str())
		{
			let lines = content.lines().count();
			c.set_badge(format!("{lines} lines"));
		} else if fold.state != ToolState::Streaming {
			c.set_badge(format!("{}ms", fold.start_time.elapsed().as_millis()));
		}

		if fold.state != ToolState::Streaming
			&& let Some(verdict) =
				get_verdict::<omp_tools::write::Payload, serde_json::Value>(fold.item.as_ref())
		{
			match &verdict {
				omp_tool::Verdict::Ok(payload) => {
					let action = match payload.disposition {
						omp_tools::write::WriteDisposition::Created => "created",
						omp_tools::write::WriteDisposition::Overwrote => "overwrote",
					};
					c.replace_body(Col::new().child(TextLeaf::new().text(format!(
						"{action} {} ({} bytes)",
						payload.display_path, payload.reported_len
					))));
				},
				_ => render_verdict_fallback(c, &verdict),
			}
		}
		true
	})
}

pub fn render_generic(ui: &mut Ui, fold: &ToolFold) -> bool {
	ui.update_component::<ToolCard>(&fold.call_id, |c| {
		c.set_name(fold.name.clone());
		c.set_state(fold.state);

		if let Some(intent) = fold
			.parsed_args
			.get("path")
			.or_else(|| fold.parsed_args.get("command"))
			.or_else(|| fold.parsed_args.get("pattern"))
			.and_then(|v| v.as_str())
		{
			c.set_intent(intent);
		}

		if fold.state != ToolState::Streaming {
			c.set_badge(format!("{}ms", fold.start_time.elapsed().as_millis()));
		}

		if fold.state != ToolState::Streaming
			&& let Some(json_val) = get_result_json(fold.item.as_ref())
		{
			let text = serde_json::to_string_pretty(&json_val).unwrap_or_default();
			let lines: Vec<&str> = text.lines().take(10).collect();
			let mut preview = lines.join("\n");
			if text.lines().count() > 10 {
				preview.push_str("\n...");
			}
			c.replace_body(Col::new().child(TextLeaf::new().text(preview)));
		}
		true
	})
}

#[cfg(test)]
mod tests {
	use omp_tool::Verdict;
	use omp_tui::{Prop, UiContext, test_support::frame_row_text};
	use serde::Serialize;

	use super::*;

	fn tool_ui() -> Ui {
		Ui::from_root(ToolCard::new().with(Prop::Id, "call"), 80, UiContext::default())
	}

	fn frame_text(ui: &Ui) -> String {
		(0..ui.frame().size().height)
			.map(|row| frame_row_text(ui.frame(), row))
			.collect::<Vec<_>>()
			.join("\n")
	}

	fn json_to_proto(value: serde_json::Value) -> omp_proto::inference::v1::Value {
		use omp_proto::inference::v1::value::Kind;

		let kind = match value {
			serde_json::Value::Null => Kind::Null(false),
			serde_json::Value::Bool(value) => Kind::Bool(value),
			serde_json::Value::Number(value) => {
				if let Some(value) = value.as_i64() {
					Kind::Int(value)
				} else if let Some(value) = value.as_u64() {
					Kind::Uint(value)
				} else {
					Kind::Double(value.as_f64().expect("JSON number is representable"))
				}
			},
			serde_json::Value::String(value) => Kind::String(value),
			serde_json::Value::Array(values) => Kind::List(omp_proto::inference::v1::ValueList {
				values: values.into_iter().map(json_to_proto).collect(),
			}),
			serde_json::Value::Object(fields) => Kind::Map(omp_proto::inference::v1::ValueMap {
				fields: fields
					.into_iter()
					.map(|(key, value)| (key, json_to_proto(value)))
					.collect(),
			}),
		};
		omp_proto::inference::v1::Value { kind: Some(kind) }
	}

	fn result_item<P: Serialize, F: Serialize>(name: &str, verdict: &Verdict<P, F>) -> Item {
		let is_error = !matches!(verdict, Verdict::Ok(_));
		let details = serde_json::to_value(verdict).expect("verdict serializes");
		Item {
			seq:           1,
			created_at_ms: 0,
			kind:          Some(item::Kind::ToolResult(omp_proto::thread::v1::ToolResult {
				call_id: "call".to_owned(),
				parts: Vec::new(),
				is_error,
				name: name.to_owned(),
				details: Some(json_to_proto(details)),
				attribution: 0,
				pruned_at_ms: None,
				useless: None,
				provider_metadata: None,
			})),
			props:         None,
		}
	}
	fn edit_payload(diff: &str) -> omp_tools::edit::Payload {
		omp_tools::edit::Payload {
			sections: vec![omp_tools::edit::SectionPayload {
				path:               "test.rs".into(),
				canonical_path:     "/workspace/test.rs".into(),
				op:                 omp_tools::edit::SectionOp::Update,
				move_dest:          None,
				old_revision:       "a".into(),
				new_revision:       Some("b".into()),
				applied_ops:        Vec::new(),
				rebased:            false,
				before:             Bytes::from_static(b"old\n"),
				after:              Bytes::from_static(b"new\n"),
				header:             Some("[test.rs#A1B2]".into()),
				diff:               diff.into(),
				preview:            diff.into(),
				first_changed_line: Some(1),
				block_resolutions:  Vec::new(),
				warnings:           Vec::new(),
			}],
		}
	}

	fn eval_payload(
		outcome: omp_tools::eval::CellOutcome,
		output: &'static [u8],
	) -> omp_tools::eval::Payload {
		omp_tools::eval::Payload {
			session_id:      Bytes::from_static(b"session"),
			cell_id:         Bytes::from_static(b"cell"),
			language:        omp_tools::eval::Language::Py,
			title:           Some("load config".into()),
			code:            "print('loaded')".into(),
			reset:           false,
			frames:          vec![omp_tools::eval::OutputFrame {
				channel:  omp_tools::eval::OutputChannel::Stdout,
				data:     output.to_vec().into(),
				sequence: 1,
			}],
			result:          Some(omp_tools::eval::CellValue {
				text: "42".into(),
				json: Some(serde_json::json!(42)),
			}),
			display_outputs: Vec::new(),
			status:          omp_tools::eval::CellStatus {
				outcome,
				exit_code: if outcome == omp_tools::eval::CellOutcome::Error {
					Some(1)
				} else {
					Some(0)
				},
				duration_ms: 17,
				exception: None,
			},
			truncated:       false,
			spilled_output:  None,
			total_lines:     2,
			total_bytes:     output.len(),
		}
	}

	#[test]
	fn generic_card_renders_name_and_argument_intent() {
		let mut ui = tool_ui();
		let mut fold =
			ToolFold::new("call".into(), "unknown".into(), Rev { family: "".into(), n: 1 });
		fold.set_args_view(omp_slopjson::parse_streaming(r#"{"path":"foo.txt"}"#));

		assert!(RendererRegistry::new().update(&mut ui, &fold));
		let frame = frame_text(&ui);
		assert!(frame.contains("unknown"));
		assert!(frame.contains("foo.txt"));
	}

	#[test]
	fn unknown_revision_degrades_to_generic_card() {
		let mut ui = tool_ui();
		let verdict: Verdict<omp_tools::edit::Payload, omp_tools::edit::Fault> =
			Verdict::Ok(edit_payload("+added"));
		let mut fold =
			ToolFold::new("call".into(), "edit".into(), Rev { family: "hl".into(), n: 2 });
		fold.state = ToolState::Success;
		fold.item = Some(result_item("edit", &verdict));

		assert!(RendererRegistry::new().update(&mut ui, &fold));
		assert!(frame_text(&ui).contains("\"kind\": \"ok\""));
	}

	#[test]
	fn read_success_renders_canonical_payload_text() {
		let mut ui = tool_ui();
		let verdict: Verdict<omp_tools::read::Payload, omp_tools::read::Fault> =
			Verdict::Ok(omp_tools::read::Payload {
				parts: vec![
					omp_tools::read::PayloadPart::Text { text: "before blob".into() },
					omp_tools::read::PayloadPart::Blob {
						blob: omp_tool::BlobRef {
							hash:       "blob-hash".into(),
							media_type: "image/png".into(),
							byte_len:   12,
						},
						alt:  "image description".into(),
					},
					omp_tools::read::PayloadPart::Text { text: "after blob".into() },
				],
			});
		let mut fold =
			ToolFold::new("call".into(), "read".into(), Rev { family: "".into(), n: 1 });
		fold.state = ToolState::Success;
		fold.item = Some(result_item("read", &verdict));

		assert!(RendererRegistry::new().update(&mut ui, &fold));
		let frame = frame_text(&ui);
		let before = frame.find("before blob").expect("leading text rendered");
		let blob = frame.find("image description").expect("blob alt rendered");
		let after = frame.find("after blob").expect("trailing text rendered");
		assert!(before < blob && blob < after);
	}
	#[test]
	fn edit_intent_comes_from_input_section_header() {
		let mut ui = tool_ui();
		let mut fold =
			ToolFold::new("call".into(), "edit".into(), Rev { family: "hl".into(), n: 1 });
		fold.set_args_view(omp_slopjson::parse_streaming(
			r#"{"input":"[src/lib.rs#A1B2]\nPUT 1.=1:\n+replacement"}"#,
		));

		assert!(RendererRegistry::new().update(&mut ui, &fold));
		assert!(frame_text(&ui).contains("edit src/lib.rs"));
	}
	#[test]
	fn write_success_renders_typed_payload_and_content_line_badge() {
		let mut ui = tool_ui();
		let verdict: Verdict<omp_tools::write::Payload, omp_tools::write::Fault> =
			Verdict::Ok(omp_tools::write::Payload {
				resolved_path:    "/workspace/new.txt".into(),
				display_path:     "new.txt".into(),
				byte_len:         11,
				reported_len:     11,
				disposition:      omp_tools::write::WriteDisposition::Created,
				stripped_wrapper: false,
				made_executable:  false,
				snapshot_tag:     Some("A1B2".into()),
				operation:        omp_tools::write::WriteOperation::Plain,
			});
		let mut fold =
			ToolFold::new("call".into(), "write".into(), Rev { family: "".into(), n: 1 });
		fold.set_args_view(omp_slopjson::parse_streaming(
			r#"{"path":"new.txt","content":"first\nsecond"}"#,
		));
		fold.state = ToolState::Success;
		fold.item = Some(result_item("write", &verdict));

		assert!(RendererRegistry::new().update(&mut ui, &fold));
		let frame = frame_text(&ui);
		assert!(frame.contains("write new.txt"));
		assert!(frame.contains("2 lines"));
		assert!(frame.contains("created new.txt (11 bytes)"));
	}

	#[test]
	fn eval_success_card_renders_title_stdout_and_repl_result() {
		let mut ui = tool_ui();
		let verdict: Verdict<omp_tools::eval::Payload, omp_tools::eval::Fault> =
			Verdict::Ok(eval_payload(omp_tools::eval::CellOutcome::Complete, b"loaded\n"));
		let mut fold =
			ToolFold::new("call".into(), "eval".into(), Rev { family: "".into(), n: 1 });
		fold.set_args_view(omp_slopjson::parse_streaming(
			r#"{"language":"py","title":"load config","code":"print('loaded')"}"#,
		));
		fold.state = ToolState::Success;
		fold.item = Some(result_item("eval", &verdict));

		assert!(RendererRegistry::new().update(&mut ui, &fold));
		let frame = frame_text(&ui);
		assert!(frame.contains("eval"));
		assert!(frame.contains("load config"));
		assert!(frame.contains("17ms"));
		assert!(frame.contains("loaded"));
		assert!(frame.contains("42"));
	}

	#[test]
	fn eval_error_card_renders_traceback_and_exit_badge() {
		let mut ui = tool_ui();
		let mut payload = eval_payload(omp_tools::eval::CellOutcome::Error, b"before\n");
		payload.result = None;
		payload.status.exception = Some(omp_tools::eval::PythonException {
			name:      "ValueError".into(),
			message:   "bad value".into(),
			traceback: vec![
				"Traceback (most recent call last):".into(),
				"ValueError: bad value".into(),
			],
		});
		let verdict: Verdict<omp_tools::eval::Payload, omp_tools::eval::Fault> = Verdict::Ok(payload);
		let mut fold =
			ToolFold::new("call".into(), "eval".into(), Rev { family: "".into(), n: 1 });
		fold.set_args_view(omp_slopjson::parse_streaming(
			r#"{"language":"py","code":"raise ValueError('bad value')"}"#,
		));
		fold.state = ToolState::Success;
		fold.item = Some(result_item("eval", &verdict));

		assert!(RendererRegistry::new().update(&mut ui, &fold));
		let frame = frame_text(&ui);
		assert!(frame.contains("exit 1"));
		assert!(frame.contains("ValueError: bad value"));
	}

	#[test]
	fn edit_success_renders_diff_lines() {
		let mut ui = tool_ui();
		let verdict: Verdict<omp_tools::edit::Payload, omp_tools::edit::Fault> =
			Verdict::Ok(edit_payload("+added line\n-removed line\n context line"));
		let mut fold =
			ToolFold::new("call".into(), "edit".into(), Rev { family: "hl".into(), n: 1 });
		fold.state = ToolState::Success;
		fold.item = Some(result_item("edit", &verdict));

		assert!(RendererRegistry::new().update(&mut ui, &fold));
		let frame = frame_text(&ui);
		assert!(frame.contains("+added line"));
		assert!(frame.contains("-removed line"));
	}

	#[test]
	fn shell_renders_canonical_live_bytes_and_exit_badge() {
		let mut ui = tool_ui();
		let verdict: Verdict<omp_tools::shell::Payload, omp_tools::shell::Fault> =
			Verdict::Ok(omp_tools::shell::Payload {
				session_id:           Bytes::from_static(b"session"),
				exec_id:              Bytes::from_static(b"exec"),
				command:              "printf tail".into(),
				transcript:           Vec::new(),
				transcript_truncated: false,
				status:               omp_tools::shell::ExecStatus {
					outcome:         omp_tools::shell::ExecOutcome::Failed,
					exit_code:       Some(42),
					signal:          None,
					wall_clock_ms:   100,
					spilled_output:  None,
					aborted:         false,
					effects_unknown: false,
				},
			});
		let mut fold =
			ToolFold::new("call".into(), "shell".into(), Rev { family: "".into(), n: 1 });
		fold.set_args_view(omp_slopjson::parse_streaming(r#"{"command":"printf tail"}"#));
		fold
			.push_update(Bytes::from_static(br#"{"channel":"stdout","data":[231,149],"sequence":1}"#));
		fold.push_update(Bytes::from_static(br#"{"channel":"stdout","data":[140,27],"sequence":2}"#));
		fold.push_update(Bytes::from_static(
			br#"{"channel":"stdout","data":[91,51,49,109,108,105,118,101,45,116,97,105,108,27,91,48,109,10],"sequence":3}"#,
		));
		fold.state = ToolState::Success;
		fold.item = Some(result_item("shell", &verdict));

		assert!(RendererRegistry::new().update(&mut ui, &fold));
		let frame = frame_text(&ui);
		assert!(frame.contains("exit 42"));
		assert!(frame.contains("live-tail"));
		assert!(frame.contains("界"));
		assert!(!frame.contains('\u{1b}'));
	}

	#[test]
	fn skipped_shell_retains_abort_reason() {
		let mut ui = tool_ui();
		let verdict: Verdict<omp_tools::shell::Payload, omp_tools::shell::Fault> =
			Verdict::Aborted(omp_tool::Abort::Skipped {
				reason: "interrupted before execution".into(),
			});
		let mut fold =
			ToolFold::new("call".into(), "shell".into(), Rev { family: "".into(), n: 1 });
		fold.state = ToolState::Failure;
		fold.item = Some(result_item("shell", &verdict));

		assert!(RendererRegistry::new().update(&mut ui, &fold));
		let frame = frame_text(&ui);
		assert!(frame.contains("skipped"));
		assert!(frame.contains("interrupted before execution"));
	}

	#[test]
	fn fault_renders_canonical_detail() {
		let mut ui = tool_ui();
		let verdict: Verdict<omp_tools::edit::Payload, omp_tools::edit::Fault> =
			Verdict::Fault(omp_tools::edit::Fault {
				reason:    omp_tools::edit::RejectionReason::InvalidPatch {
					message: "unrecoverable conflict".into(),
				},
				conflicts: Vec::new(),
			});
		let mut fold =
			ToolFold::new("call".into(), "edit".into(), Rev { family: "hl".into(), n: 1 });
		fold.state = ToolState::Failure;
		fold.item = Some(result_item("edit", &verdict));

		assert!(RendererRegistry::new().update(&mut ui, &fold));
		assert!(frame_text(&ui).contains("unrecoverable conflict"));
	}
}
