//! Scanner boundary and chunk-segmentation regression fixtures.

use bytes::Bytes;
use omp_llm_dialect::{Dialect, InbandTool, ScanEvent, ScannerOptions, factory::create_scanner};
use serde_json::{Value, json};

#[derive(Clone, Copy)]
struct Case {
	dialect:       Dialect,
	tool:          &'static str,
	raw:           &'static str,
	thinking_open: &'static str,
	incomplete:    &'static str,
	malformed:     &'static str,
	raw_arguments: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum CanonicalEvent {
	Text(Vec<u8>),
	Thinking {
		body:      Vec<u8>,
		signature: Vec<u8>,
	},
	Tool {
		name:            String,
		arguments:       Value,
		argument_stream: Vec<u8>,
		raw:             Option<Vec<u8>>,
	},
}

#[derive(Debug)]
struct OpenTool {
	id:              String,
	name:            String,
	argument_stream: Vec<u8>,
}

const fn cases() -> [Case; 11] {
	[
		Case {
			dialect:       Dialect::Glm,
			tool:          concat!(
				"<tool_call>echo\n",
				"<arg_key>msg</arg_key><arg_value>{\"snow\":\"雪\"}</arg_value>",
				"<arg_key>count</arg_key><arg_value>7</arg_value>",
				"</tool_call>",
			),
			raw:           concat!(
				"<tool_call>echo\n",
				"<arg_key>msg</arg_key><arg_value>{\"snow\":\"雪\"}</arg_value>",
				"<arg_key>count</arg_key><arg_value>7</arg_value>",
				"</tool_call>",
			),
			thinking_open: "<think>",
			incomplete:    "<tool_call>",
			malformed:     "<tool_call>\n</tool_call>",
			raw_arguments: true,
		},
		Case {
			dialect:       Dialect::Hermes,
			tool:          r#"<tool_call>{"name":"echo","arguments":{"msg":"{\"snow\":\"雪\"}","count":7}}</tool_call>"#,
			raw:           r#"<tool_call>{"name":"echo","arguments":{"msg":"{\"snow\":\"雪\"}","count":7}}</tool_call>"#,
			thinking_open: "<think>",
			incomplete:    "<tool_call>",
			malformed:     r#"<tool_call>{"name":"echo","arguments":42}</tool_call>"#,
			raw_arguments: false,
		},
		Case {
			dialect:       Dialect::Kimi,
			tool:          concat!(
				"<|tool_calls_section_begin|>",
				"<|tool_call_begin|>functions.echo:0<|tool_call_argument_begin|>",
				r#"{"msg":"{\"snow\":\"雪\"}","count":7}"#,
				"<|tool_call_end|><|tool_calls_section_end|>",
			),
			raw:           concat!(
				"<|tool_call_begin|>functions.echo:0<|tool_call_argument_begin|>",
				r#"{"msg":"{\"snow\":\"雪\"}","count":7}"#,
				"<|tool_call_end|>",
			),
			thinking_open: "<think>",
			incomplete:    "<|tool_calls_section_begin|><|tool_call_begin|>",
			malformed:     concat!(
				"<|tool_calls_section_begin|>",
				"<|tool_call_begin|>functions.echo:0<|tool_call_argument_begin|>{bad}",
				"<|tool_call_end|><|tool_calls_section_end|>",
			),
			raw_arguments: false,
		},
		Case {
			dialect:       Dialect::Xml,
			tool:          concat!(
				"<invoke name=\"echo\">",
				"<parameter name=\"msg\">{\"snow\":\"雪\"}</parameter>",
				"<parameter name=\"count\">7</parameter>",
				"</invoke>",
			),
			raw:           concat!(
				"<invoke name=\"echo\">",
				"<parameter name=\"msg\">{\"snow\":\"雪\"}</parameter>",
				"<parameter name=\"count\">7</parameter>",
				"</invoke>",
			),
			thinking_open: "<scratchpad>",
			incomplete:    "<invoke",
			malformed:     "<invoke></invoke>",
			raw_arguments: true,
		},
		Case {
			dialect:       Dialect::Anthropic,
			tool:          concat!(
				"<function_calls><invoke name=\"echo\">",
				"<parameter name=\"msg\">{\"snow\":\"雪\"}</parameter>",
				"<parameter name=\"count\">7</parameter>",
				"</invoke></function_calls>",
			),
			raw:           concat!(
				"<invoke name=\"echo\">",
				"<parameter name=\"msg\">{\"snow\":\"雪\"}</parameter>",
				"<parameter name=\"count\">7</parameter>",
				"</invoke>",
			),
			thinking_open: "<antml:thinking>",
			incomplete:    "<function_calls><invoke",
			malformed:     "<function_calls><invoke></invoke></function_calls>",
			raw_arguments: true,
		},
		Case {
			dialect:       Dialect::DeepSeek,
			tool:          concat!(
				"<｜tool▁call▁begin｜>echo<｜tool▁sep｜>",
				r#"{"msg":"{\"snow\":\"雪\"}","count":7}"#,
				"<｜tool▁call▁end｜>",
			),
			raw:           concat!(
				"<｜tool▁call▁begin｜>echo<｜tool▁sep｜>",
				r#"{"msg":"{\"snow\":\"雪\"}","count":7}"#,
				"<｜tool▁call▁end｜>",
			),
			thinking_open: "<think>",
			incomplete:    "<｜tool▁call▁begin｜>",
			malformed:     "<｜tool▁call▁begin｜>echo<｜tool▁sep｜>{bad}<｜tool▁call▁end｜>",
			raw_arguments: false,
		},
		Case {
			dialect:       Dialect::Harmony,
			tool:          concat!(
				"<\x7cstart\x7c>assistant<\x7cchannel\x7c>analysis to=functions.echo<\x7cmessage\x7c>",
				r#"{"msg":"{\"snow\":\"雪\"}","count":7}"#,
				"<\x7ccall\x7c>",
			),
			raw:           concat!(
				"<\x7cstart\x7c>assistant<\x7cchannel\x7c>analysis to=functions.echo<\x7cmessage\x7c>",
				r#"{"msg":"{\"snow\":\"雪\"}","count":7}"#,
				"<\x7ccall\x7c>",
			),
			thinking_open: "<\x7cstart\x7c>assistant<\x7cchannel\x7c>analysis<\x7cmessage\x7c>",
			incomplete:    "<\x7cstart\x7c>",
			malformed:     "<\x7cstart\x7c>tool<\x7cchannel\x7c>analysis \
			                to=functions.echo<\x7cmessage\x7c>{bad}<\x7ccall\x7c>",
			raw_arguments: false,
		},
		Case {
			dialect:       Dialect::Qwen3,
			tool:          r#"<tool_call>{"name":"echo","arguments":{"msg":"{\"snow\":\"雪\"}","count":7}}</tool_call>"#,
			raw:           r#"<tool_call>{"name":"echo","arguments":{"msg":"{\"snow\":\"雪\"}","count":7}}</tool_call>"#,
			thinking_open: "<think>",
			incomplete:    "<tool_call>",
			malformed:     r#"<tool_call>{"name":"echo","arguments":42}</tool_call>"#,
			raw_arguments: false,
		},
		Case {
			dialect:       Dialect::Gemini,
			tool:          "```tool_code\ndefault_api.echo(msg='{\"snow\":\"雪\"}', count=7)\n```",
			raw:           "```tool_code\ndefault_api.echo(msg='{\"snow\":\"雪\"}', count=7)\n```",
			thinking_open: "```thinking\n",
			incomplete:    "```tool_code",
			malformed:     "```tool_code\ndefault_api.echo(msg=unknown)\n```",
			raw_arguments: false,
		},
		Case {
			dialect:       Dialect::Gemma,
			tool:          "<|tool_call>call:echo{msg:<|\"|>{\"snow\":\"雪\"}<|\"|>,count:\
			                7}<tool_call|>",
			raw:           "<|tool_call>call:echo{msg:<|\"|>{\"snow\":\"雪\"}<|\"|>,count:\
			                7}<tool_call|>",
			thinking_open: "<|channel>thought\n",
			incomplete:    "<|tool_call>",
			malformed:     "<|tool_call>call:echo{msg:[1,}<tool_call|>",
			raw_arguments: false,
		},
		Case {
			dialect:       Dialect::MiniMax,
			tool:          concat!(
				"<minimax:tool_call><invoke name=\"echo\">",
				"<parameter name=\"msg\">{\"snow\":\"雪\"}</parameter>",
				"<parameter name=\"count\">7</parameter>",
				"</invoke></minimax:tool_call>",
			),
			raw:           concat!(
				"<invoke name=\"echo\">",
				"<parameter name=\"msg\">{\"snow\":\"雪\"}</parameter>",
				"<parameter name=\"count\">7</parameter>",
				"</invoke>",
			),
			thinking_open: "<thinking>",
			incomplete:    "<minimax:tool_call><invoke",
			malformed:     "<minimax:tool_call><invoke></invoke></minimax:tool_call>",
			raw_arguments: true,
		},
	]
}

fn scan(dialect: Dialect, wire: &[u8], cuts: &[usize], include_raw_tool: bool) -> Vec<ScanEvent> {
	let schema = json!({
		"type": "object",
		"properties": {
			"msg": { "type": "string" },
			"count": { "type": "integer" },
		},
	});
	let tools = [InbandTool::new("echo", None, &schema, &[])];
	let mut options = ScannerOptions::new(&tools);
	options.parse_thinking = true;
	options.include_raw_tool = include_raw_tool;
	let mut scanner = create_scanner(dialect, options);
	let mut events = Vec::new();
	let mut start = 0;
	for &end in cuts {
		assert!((start..=wire.len()).contains(&end));
		events.extend(scanner.feed(Bytes::copy_from_slice(&wire[start..end])));
		start = end;
	}
	events.extend(scanner.feed(Bytes::copy_from_slice(&wire[start..])));
	events.extend(scanner.flush());
	events
}

fn flush_text(text: &mut Vec<u8>, output: &mut Vec<CanonicalEvent>) {
	if !text.is_empty() {
		output.push(CanonicalEvent::Text(std::mem::take(text)));
	}
}

fn canonicalize(events: Vec<ScanEvent>) -> Vec<CanonicalEvent> {
	let mut output = Vec::new();
	let mut text = Vec::new();
	let mut thinking: Option<Vec<u8>> = None;
	let mut tool: Option<OpenTool> = None;

	for event in events {
		match event {
			ScanEvent::Text(delta) => {
				assert!(thinking.is_none(), "visible text emitted inside thinking");
				assert!(tool.is_none(), "visible text emitted inside a tool call");
				text.extend_from_slice(&delta);
			},
			ScanEvent::ThinkingStart => {
				flush_text(&mut text, &mut output);
				assert!(tool.is_none(), "thinking started inside a tool call");
				assert!(thinking.replace(Vec::new()).is_none(), "duplicated thinking start");
			},
			ScanEvent::ThinkingDelta(delta) => thinking
				.as_mut()
				.expect("thinking delta without a start")
				.extend_from_slice(&delta),
			ScanEvent::ThinkingEnd { signature } => {
				let body = thinking.take().expect("thinking end without a start");
				output.push(CanonicalEvent::Thinking { body, signature: signature.to_vec() });
			},
			ScanEvent::ToolStart { id, name } => {
				flush_text(&mut text, &mut output);
				assert!(thinking.is_none(), "tool started inside thinking");
				let next = OpenTool {
					id:              id.to_string(),
					name:            name.to_string(),
					argument_stream: Vec::new(),
				};
				assert!(tool.replace(next).is_none(), "duplicated tool start");
			},
			ScanEvent::ToolArgumentDelta { id, delta } => {
				let active = tool.as_mut().expect("tool arguments without a start");
				assert_eq!(id.as_str(), active.id, "tool argument id changed");
				active.argument_stream.extend_from_slice(&delta);
			},
			ScanEvent::ToolEnd { id, name, args_json, raw_block } => {
				let active = tool.take().expect("tool end without a start");
				assert_eq!(id.as_str(), active.id, "tool end id changed");
				assert_eq!(name.as_str(), active.name, "tool end name changed");
				output.push(CanonicalEvent::Tool {
					name:            active.name,
					arguments:       serde_json::from_slice(&args_json).expect("canonical tool JSON"),
					argument_stream: active.argument_stream,
					raw:             raw_block.map(|bytes| bytes.to_vec()),
				});
			},
			_ => panic!("scanner emitted an unknown event variant"),
		}
	}
	flush_text(&mut text, &mut output);
	assert!(thinking.is_none(), "thinking did not close on flush");
	assert!(tool.is_none(), "tool did not close on flush");
	output
}

#[test]
fn scanner_boundaries_are_canonical_for_every_dialect() {
	let expected_arguments = json!({ "msg": "{\"snow\":\"雪\"}", "count": 7 });
	let json_argument_stream = serde_json::to_vec(&expected_arguments).unwrap();

	for case in cases() {
		let wire = format!("前{}后{}理🙂", case.tool, case.thinking_open);
		let whole = canonicalize(scan(case.dialect, wire.as_bytes(), &[], true));
		let [
			CanonicalEvent::Text(prefix),
			CanonicalEvent::Tool { name, arguments, argument_stream, raw },
			CanonicalEvent::Text(suffix),
			CanonicalEvent::Thinking { body, signature },
		] = whole.as_slice()
		else {
			panic!(
				"{:?}: leaked a marker or emitted a duplicated/missing event: {whole:?}",
				case.dialect
			);
		};
		assert_eq!(prefix, "前".as_bytes(), "{:?}", case.dialect);
		assert_eq!(suffix, "后".as_bytes(), "{:?}", case.dialect);
		assert_eq!(body, "理🙂".as_bytes(), "{:?}", case.dialect);
		assert!(
			signature.is_empty(),
			"{:?}: in-band thinking must close with an empty signature",
			case.dialect
		);
		assert_eq!(name, "echo", "{:?}", case.dialect);
		assert_eq!(arguments, &expected_arguments, "{:?}", case.dialect);
		let expected_stream = if case.raw_arguments {
			b"{\"snow\":\"\xE9\x9B\xAA\"}7".as_slice()
		} else {
			json_argument_stream.as_slice()
		};
		assert_eq!(
			argument_stream, expected_stream,
			"{:?}: duplicated or missing argument delta",
			case.dialect
		);
		assert_eq!(raw.as_deref(), Some(case.raw.as_bytes()), "{:?}", case.dialect);

		let byte_cuts: Vec<usize> = (1..wire.len()).collect();
		assert_eq!(
			canonicalize(scan(case.dialect, wire.as_bytes(), &byte_cuts, true)),
			whole,
			"{:?}: byte-at-a-time chunks changed canonical events",
			case.dialect,
		);
		for split in 0..=wire.len() {
			assert_eq!(
				canonicalize(scan(case.dialect, wire.as_bytes(), &[split], true)),
				whole,
				"{:?}: split at byte {split} changed canonical events",
				case.dialect,
			);
		}

		let mut without_raw = whole.clone();
		for event in &mut without_raw {
			if let CanonicalEvent::Tool { raw, .. } = event {
				*raw = None;
			}
		}
		assert_eq!(
			canonicalize(scan(case.dialect, wire.as_bytes(), &byte_cuts, false)),
			without_raw,
			"{:?}: disabling raw capture changed scanner semantics",
			case.dialect,
		);

		let incomplete_whole =
			canonicalize(scan(case.dialect, case.incomplete.as_bytes(), &[], true));
		let incomplete_cuts: Vec<usize> = (1..case.incomplete.len()).collect();
		assert_eq!(
			canonicalize(scan(case.dialect, case.incomplete.as_bytes(), &incomplete_cuts, true,)),
			incomplete_whole,
			"{:?}: chunking changed incomplete-envelope flush behavior",
			case.dialect,
		);
		assert!(
			!incomplete_whole
				.iter()
				.any(|event| matches!(event, CanonicalEvent::Tool { .. })),
			"{:?}: an incomplete envelope completed a tool call during flush",
			case.dialect,
		);

		let malformed_cuts: Vec<usize> = (1..case.malformed.len()).collect();
		let malformed_whole = canonicalize(scan(case.dialect, case.malformed.as_bytes(), &[], true));
		assert_eq!(
			canonicalize(scan(case.dialect, case.malformed.as_bytes(), &malformed_cuts, true,)),
			malformed_whole,
			"{:?}: chunking changed malformed-envelope recovery",
			case.dialect,
		);
		assert!(
			!malformed_whole
				.iter()
				.any(|event| matches!(event, CanonicalEvent::Tool { .. })),
			"{:?}: a malformed envelope produced a tool call",
			case.dialect,
		);
	}

	let xml = "前<b data-x='1'>雪</b><ordinary";
	let xml_cuts: Vec<usize> = (1..xml.len()).collect();
	assert_eq!(
		canonicalize(scan(Dialect::Xml, xml.as_bytes(), &xml_cuts, true)),
		vec![CanonicalEvent::Text(xml.as_bytes().to_vec())],
		"non-tool XML or an incomplete ordinary tag did not pass through verbatim",
	);
}
