use std::time::Instant;

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
pub(crate) struct ToolFold {
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

pub(crate) type RendererFn = fn(&mut Ui, &ToolFold) -> bool;

pub(crate) struct RendererRegistry {
	renderers: &'static [(&'static str, &'static str, u16, RendererFn)],
}

impl RendererRegistry {
	pub(crate) const fn new() -> Self {
		Self {
			renderers: &[
				("read", "", 1, render_read),
				("edit", "hl", 1, render_edit),
				("shell", "", 1, render_shell),
				("grep", "", 1, render_grep),
				("glob", "", 1, render_glob),
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

fn get_result_json(item: &Option<Item>) -> Option<serde_json::Value> {
	let i = item.as_ref()?;
	let item::Kind::ToolResult(res) = &i.kind.as_ref()? else {
		return None;
	};
	let details = res.details.as_ref()?;
	proto_to_json(details)
}

fn get_verdict<P: DeserializeOwned, F: DeserializeOwned>(
	item: &Option<Item>,
) -> Option<omp_tool::Verdict<P, F>> {
	let json_val = get_result_json(item)?;
	serde_json::from_value(json_val).ok()
}

fn render_verdict_fallback<P: DeserializeOwned>(
	c: &mut ToolCard,
	verdict: &omp_tool::Verdict<P, serde_json::Value>,
) {
	match verdict {
		omp_tool::Verdict::Fault(f) => {
			let text = serde_json::to_string_pretty(f).unwrap_or_default();
			c.replace_body(Col::new().child(TextLeaf::new().text(format!("Fault:\n{}", text))));
		},
		omp_tool::Verdict::Args(a) => {
			let text = serde_json::to_string_pretty(a).unwrap_or_default();
			c.replace_body(Col::new().child(TextLeaf::new().text(format!("Arg Issue:\n{}", text))));
		},
		omp_tool::Verdict::Aborted(a) => {
			let text = serde_json::to_string_pretty(a).unwrap_or_default();
			c.replace_body(Col::new().child(TextLeaf::new().text(format!("Aborted:\n{}", text))));
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

pub(crate) fn render_read(ui: &mut Ui, fold: &ToolFold) -> bool {
	ui.update_component::<ToolCard>(&fold.call_id, |c| {
		c.set_name(fold.name.clone());
		c.set_state(fold.state);
		if fold.state != ToolState::Streaming {
			c.set_badge(format!("{}ms", fold.start_time.elapsed().as_millis()));
		}

		if let Some(path) = fold.parsed_args.get("path").and_then(|v| v.as_str()) {
			c.set_intent(format!("read {}", path));
		}

		if fold.state != ToolState::Streaming {
			if let Some(verdict) =
				get_verdict::<omp_tools::read::Payload, serde_json::Value>(&fold.item)
			{
				match &verdict {
					omp_tool::Verdict::Ok(payload) => {
						let mut col = Col::new();
						match &payload.content {
							omp_tools::read::Content::Text { slices, .. } => {
								for slice in slices {
									col = col.child(TextLeaf::new().text(slice.text.as_str()));
								}
							},
							omp_tools::read::Content::Summary { segments, .. } => {
								for segment in segments {
									col = col.child(TextLeaf::new().text(segment.text.as_str()));
								}
							},
							omp_tools::read::Content::Blob { fallback, .. } => {
								col = col.child(TextLeaf::new().text(fallback.as_str()));
							},
						}
						c.replace_body(col);
					},
					_ => render_verdict_fallback(c, &verdict),
				}
			}
		}
		true
	})
}

pub(crate) fn render_edit(ui: &mut Ui, fold: &ToolFold) -> bool {
	ui.update_component::<ToolCard>(&fold.call_id, |c| {
		c.set_name(fold.name.clone());
		c.set_state(fold.state);
		if fold.state != ToolState::Streaming {
			c.set_badge(format!("{}ms", fold.start_time.elapsed().as_millis()));
		}

		if let Some(path) = fold.parsed_args.get("path").and_then(|v| v.as_str()) {
			c.set_intent(format!("edit {}", path));
		}

		if fold.state == ToolState::Streaming {
			if let Some(last_update) = fold.updates.last() {
				if let Some(preview) = last_update.get("preview").and_then(|v| v.as_str()) {
					let mut diff = DiffView::new();
					parse_diff_lines(&mut diff, preview);
					c.replace_body(diff);
				}
			}
		} else {
			if let Some(verdict) =
				get_verdict::<omp_tools::edit::Payload, serde_json::Value>(&fold.item)
			{
				match &verdict {
					omp_tool::Verdict::Ok(payload) => {
						let mut diff = DiffView::new();
						parse_diff_lines(&mut diff, payload.diff.as_str());
						c.replace_body(diff);
					},
					_ => render_verdict_fallback(c, &verdict),
				}
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

pub(crate) fn render_shell(ui: &mut Ui, fold: &ToolFold) -> bool {
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

		if fold.state != ToolState::Streaming {
			if let Some(verdict) =
				get_verdict::<omp_tools::shell::Payload, serde_json::Value>(&fold.item)
			{
				match &verdict {
					omp_tool::Verdict::Ok(payload) => match payload.status.outcome {
						omp_tools::shell::ExecOutcome::Exited => {
							if let Some(code) = payload.status.exit_code {
								if code != 0 {
									c.set_badge(format!("exit {}", code));
								}
							}
						},
						omp_tools::shell::ExecOutcome::Cancelled => {
							c.set_badge("interrupted");
						},
						omp_tools::shell::ExecOutcome::Timeout => {
							c.set_badge("timeout");
						},
						_ => {},
					},
					_ => render_verdict_fallback(c, &verdict),
				}
			}
		}

		if !tail_text.is_empty() {
			let lines: Vec<&str> = tail_text.lines().rev().take(5).collect();
			let tail_preview = lines.into_iter().rev().collect::<Vec<_>>().join("\n");
			let col = Col::new().child(TextLeaf::new().text(tail_preview));
			c.replace_body(col);
		} else {
			c.replace_body(Col::new());
		}
		true
	})
}

pub(crate) fn render_grep(ui: &mut Ui, fold: &ToolFold) -> bool {
	ui.update_component::<ToolCard>(&fold.call_id, |c| {
		c.set_name(fold.name.clone());
		c.set_state(fold.state);
		if fold.state != ToolState::Streaming {
			c.set_badge(format!("{}ms", fold.start_time.elapsed().as_millis()));
		}

		if let Some(pattern) = fold.parsed_args.get("pattern").and_then(|v| v.as_str()) {
			c.set_intent(format!("grep {}", pattern));
		}

		if fold.state != ToolState::Streaming {
			if let Some(verdict) =
				get_verdict::<omp_tools::grep::Payload, serde_json::Value>(&fold.item)
			{
				match &verdict {
					omp_tool::Verdict::Ok(payload) => {
						let mut col = Col::new();
						for m in &payload.matches {
							col =
								col.child(TextLeaf::new().text(format!("{}:{}", m.path.as_str(), m.line)));
						}
						if payload.truncated {
							col = col.child(TextLeaf::new().text("..."));
						}
						c.replace_body(col);
					},
					_ => render_verdict_fallback(c, &verdict),
				}
			}
		}
		true
	})
}

pub(crate) fn render_glob(ui: &mut Ui, fold: &ToolFold) -> bool {
	ui.update_component::<ToolCard>(&fold.call_id, |c| {
		c.set_name(fold.name.clone());
		c.set_state(fold.state);
		if fold.state != ToolState::Streaming {
			c.set_badge(format!("{}ms", fold.start_time.elapsed().as_millis()));
		}

		if let Some(pattern) = fold
			.parsed_args
			.get("pattern")
			.or_else(|| fold.parsed_args.get("path"))
			.and_then(|v| v.as_str())
		{
			c.set_intent(format!("glob {}", pattern));
		}

		if fold.state != ToolState::Streaming {
			if let Some(verdict) =
				get_verdict::<omp_tools::glob::Payload, serde_json::Value>(&fold.item)
			{
				match &verdict {
					omp_tool::Verdict::Ok(payload) => {
						let mut col = Col::new();
						for p in &payload.paths {
							col = col.child(TextLeaf::new().text(p.as_str()));
						}
						if payload.truncated {
							col = col.child(TextLeaf::new().text("..."));
						}
						c.replace_body(col);
					},
					_ => render_verdict_fallback(c, &verdict),
				}
			}
		}
		true
	})
}

pub(crate) fn render_generic(ui: &mut Ui, fold: &ToolFold) -> bool {
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

		if fold.state != ToolState::Streaming {
			if let Some(json_val) = get_result_json(&fold.item) {
				let text = serde_json::to_string_pretty(&json_val).unwrap_or_default();
				let lines: Vec<&str> = text.lines().take(10).collect();
				let mut preview = lines.join("\n");
				if text.lines().count() > 10 {
					preview.push_str("\n...");
				}
				c.replace_body(Col::new().child(TextLeaf::new().text(preview)));
			}
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
			Verdict::Ok(omp_tools::edit::Payload {
				path:         "test.rs".into(),
				old_revision: "a".into(),
				new_revision: "b".into(),
				applied_ops:  Vec::new(),
				rebased:      false,
				diff:         "+added".into(),
			});
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
				path:       "test.txt".into(),
				revision:   "hash".into(),
				ranges:     Vec::new(),
				structural: false,
				elided:     Vec::new(),
				content:    omp_tools::read::Content::Text {
					slices: vec![omp_tools::read::TextSlice {
						start_line: 1,
						text:       "hello file content".into(),
					}],
				},
			});
		let mut fold =
			ToolFold::new("call".into(), "read".into(), Rev { family: "".into(), n: 1 });
		fold.state = ToolState::Success;
		fold.item = Some(result_item("read", &verdict));

		assert!(RendererRegistry::new().update(&mut ui, &fold));
		assert!(frame_text(&ui).contains("hello file content"));
	}

	#[test]
	fn edit_success_renders_diff_lines() {
		let mut ui = tool_ui();
		let verdict: Verdict<omp_tools::edit::Payload, omp_tools::edit::Fault> =
			Verdict::Ok(omp_tools::edit::Payload {
				path:         "test.rs".into(),
				old_revision: "a".into(),
				new_revision: "b".into(),
				applied_ops:  Vec::new(),
				rebased:      false,
				diff:         "+added line\n-removed line\n context line".into(),
			});
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
					outcome:         omp_tools::shell::ExecOutcome::Exited,
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
