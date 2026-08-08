//! In-band tool rendering and scanner round-trip fixtures.

use bytes::Bytes;
use omp_llm_dialect::{
	Dialect, DialectRenderOptions, InbandTool, ScanEvent, ScannerOptions,
	factory::create_scanner,
	rendering::{render_assistant_tool_calls, render_thinking},
};
use omp_llm_types::{ToolCall, ids::CallId};
use serde_json::{Value, json};

#[derive(Debug, Eq, PartialEq)]
struct Observed {
	text:            String,
	thinking:        String,
	thinking_starts: usize,
	thinking_ends:   usize,
	calls:           Vec<(String, Value)>,
}

fn tool_call() -> ToolCall {
	ToolCall::builder()
		.id(CallId::new())
		.name("echo".into())
		.args_json(Bytes::from_static(r#"{"msg":"héllo 🦀","count":7,"ok":true}"#.as_bytes()))
		.thought_signature(Bytes::new())
		.build()
}

fn render_call(dialect: Dialect, tool: InbandTool<'_>) -> String {
	let mut rendered = String::new();
	render_assistant_tool_calls(
		&mut rendered,
		dialect,
		&[tool_call()],
		DialectRenderOptions::new(&[tool]),
	)
	.expect("tool call renders");
	rendered
}

fn observe(dialect: Dialect, input: &[u8], chunks: &[usize], tool: InbandTool<'_>) -> Observed {
	let mut scanner = create_scanner(dialect, ScannerOptions::new(&[tool]));
	let mut events = Vec::new();
	let mut offset = 0;
	for &length in chunks {
		events.extend(scanner.feed(Bytes::copy_from_slice(&input[offset..offset + length])));
		offset += length;
	}
	assert_eq!(offset, input.len());
	events.extend(scanner.flush());

	let mut observed = Observed {
		text:            String::new(),
		thinking:        String::new(),
		thinking_starts: 0,
		thinking_ends:   0,
		calls:           Vec::new(),
	};
	for event in events {
		match event {
			ScanEvent::Text(bytes) => observed.text.push_str(std::str::from_utf8(&bytes).unwrap()),
			ScanEvent::ThinkingStart => observed.thinking_starts += 1,
			ScanEvent::ThinkingDelta(bytes) => {
				observed
					.thinking
					.push_str(std::str::from_utf8(&bytes).unwrap());
			},
			ScanEvent::ThinkingEnd { .. } => observed.thinking_ends += 1,
			ScanEvent::ToolEnd { name, args_json, .. } => {
				observed
					.calls
					.push((name.to_string(), serde_json::from_slice(&args_json).unwrap()));
			},
			ScanEvent::ToolStart { .. } | ScanEvent::ToolArgumentDelta { .. } => {},
			_ => {},
		}
	}
	observed
}

fn assert_every_segmentation(
	dialect: Dialect,
	input: &str,
	expected: &Observed,
	tool: InbandTool<'_>,
) {
	let bytes = input.as_bytes();
	assert_eq!(observe(dialect, bytes, &[bytes.len()], tool), *expected, "{dialect} whole");
	assert_eq!(
		observe(dialect, bytes, &vec![1; bytes.len()], tool),
		*expected,
		"{dialect} byte-at-a-time"
	);
	for split in 0..=bytes.len() {
		let chunks = if split == 0 || split == bytes.len() {
			vec![bytes.len()]
		} else {
			vec![split, bytes.len() - split]
		};
		assert_eq!(observe(dialect, bytes, &chunks, tool), *expected, "{dialect} split {split}");
	}
}

#[test]
fn all_eleven_dialects_round_trip_calls_for_every_byte_segmentation() {
	let schema = json!({
		"type": "object",
		"properties": {
			"msg": {"type": "string"},
			"count": {"type": "integer"},
			"ok": {"type": "boolean"}
		},
		"required": ["msg", "count", "ok"]
	});
	let tool = InbandTool::new("echo", Some("Echo input"), &schema, &[]);
	let expected = Observed {
		text:            String::new(),
		thinking:        String::new(),
		thinking_starts: 0,
		thinking_ends:   0,
		calls:           vec![("echo".into(), json!({"msg":"héllo 🦀","count":7,"ok":true}))],
	};
	for dialect in Dialect::ALL {
		let rendered = render_call(dialect, tool);
		assert_every_segmentation(dialect, &rendered, &expected, tool);
	}
}

#[test]
fn all_eleven_dialects_preserve_thinking_across_utf8_and_delimiter_splits() {
	let schema = json!({"type":"object"});
	let tool = InbandTool::new("echo", None, &schema, &[]);
	for dialect in Dialect::ALL {
		let mut rendered = String::new();
		render_thinking(&mut rendered, dialect, "first 🦀 second").unwrap();
		let expected = observe(dialect, rendered.as_bytes(), &[rendered.len()], tool);
		assert_eq!(expected.thinking_starts, 1, "{dialect}");
		assert_eq!(expected.thinking_ends, 1, "{dialect}");
		assert!(expected.thinking.contains("first 🦀 second"), "{dialect}: {expected:?}");
		assert!(!expected.text.contains("first 🦀 second"), "{dialect}: {expected:?}");
		assert_every_segmentation(dialect, &rendered, &expected, tool);
	}
}

#[test]
fn malformed_and_unterminated_inputs_flush_deterministically_for_every_dialect() {
	let schema = json!({"type":"object","properties":{"msg":{"type":"string"}}});
	let tool = InbandTool::new("echo", None, &schema, &[]);
	for dialect in Dialect::ALL {
		let rendered = render_call(dialect, tool);
		let malformed = &rendered.as_bytes()[..rendered.len() - 1];
		let whole = observe(dialect, malformed, &[malformed.len()], tool);
		let bytewise = observe(dialect, malformed, &vec![1; malformed.len()], tool);
		assert_eq!(bytewise, whole, "{dialect} malformed byte stream");
		for split in 1..malformed.len() {
			assert_eq!(
				observe(dialect, malformed, &[split, malformed.len() - split], tool),
				whole,
				"{dialect} malformed split {split}"
			);
		}

		let mut thinking = String::new();
		render_thinking(&mut thinking, dialect, "unfinished thought").unwrap();
		let cut = thinking.len().saturating_sub(3);
		let unterminated = &thinking.as_bytes()[..cut];
		let flushed = observe(dialect, unterminated, &vec![1; unterminated.len()], tool);
		assert_eq!(flushed.thinking_starts, flushed.thinking_ends, "{dialect} closes thinking");
	}
}
