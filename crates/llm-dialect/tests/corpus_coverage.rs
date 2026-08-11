//! Exact cross-dialect corpus coverage for rendering and streaming scanners.

use std::collections::HashSet;

use bytes::Bytes;
use omp_llm_dialect::{
	Dialect, DialectRenderOptions, InbandTool, ScanEvent, ScannerOptions,
	demotion::render_demoted_thinking,
	factory::create_scanner,
	prompt::{dialect_guide, write_inband_tool_prompt},
	rendering::{render_assistant_tool_calls, render_thinking, render_transcript},
};
use omp_llm_types::{
	Item, ItemKind, Message, Part, Role, Thinking, Thread, ToolCall, ToolResult, ids::CallId,
};
use serde_json::{Value, json};
const PRIVATE_THOUGHT: &str = "private 🦀 matrix thought";
const TOOL_ARGS: &str =
	r#"{"path":"folder/é.txt","count":7,"enabled":true,"tags":["alpha","beta"],"meta":{"depth":2}}"#;

#[derive(Clone, Copy)]
struct DialectCase {
	dialect:           Dialect,
	prompt_marker:     &'static str,
	transcript_marker: &'static str,
	demotion_open:     &'static str,
	demotion_close:    &'static str,
	thinking_close:    &'static str,
}

const CASES: [DialectCase; 11] = [
	DialectCase {
		dialect:           Dialect::Anthropic,
		prompt_marker:     "<function_calls>",
		transcript_marker: "<function_calls>",
		demotion_open:     "",
		demotion_close:    "",
		thinking_close:    "</thinking>",
	},
	DialectCase {
		dialect:           Dialect::DeepSeek,
		prompt_marker:     "<｜tool▁calls▁begin｜>",
		transcript_marker: "<｜tool▁calls▁begin｜>",
		demotion_open:     "<think>\n",
		demotion_close:    "\n</think>",
		thinking_close:    "</think>",
	},
	DialectCase {
		dialect:           Dialect::Gemini,
		prompt_marker:     "default_api.function_name",
		transcript_marker: "```tool_code",
		demotion_open:     "```thinking\n",
		demotion_close:    "\n```",
		thinking_close:    "```",
	},
	DialectCase {
		dialect:           Dialect::Gemma,
		prompt_marker:     "<|tool_call>",
		transcript_marker: "<|tool_call>",
		demotion_open:     "<think>\n",
		demotion_close:    "\n</think>",
		thinking_close:    "<channel|>",
	},
	DialectCase {
		dialect:           Dialect::Glm,
		prompt_marker:     "<arg_key>",
		transcript_marker: "<tool_call>",
		demotion_open:     "<think>\n",
		demotion_close:    "\n</think>",
		thinking_close:    "</think>",
	},
	DialectCase {
		dialect:           Dialect::Harmony,
		prompt_marker:     "functions.function_name",
		transcript_marker: "to=functions.matrix_probe",
		demotion_open:     "<think>\n",
		demotion_close:    "\n</think>",
		thinking_close:    "<\u{7c}end\u{7c}>",
	},
	DialectCase {
		dialect:           Dialect::Hermes,
		prompt_marker:     "{\"name\":\"function_name\"",
		transcript_marker: "<tool_call>",
		demotion_open:     "<thinking>\n",
		demotion_close:    "\n</thinking>",
		thinking_close:    "</thinking>",
	},
	DialectCase {
		dialect:           Dialect::Kimi,
		prompt_marker:     "<|tool_calls_section_begin|>",
		transcript_marker: "<|tool_calls_section_begin|>",
		demotion_open:     "<think>\n",
		demotion_close:    "\n</think>",
		thinking_close:    "</think>",
	},
	DialectCase {
		dialect:           Dialect::MiniMax,
		prompt_marker:     "<minimax:tool_call>",
		transcript_marker: "<minimax:tool_call>",
		demotion_open:     "<thinking>\n",
		demotion_close:    "\n</thinking>",
		thinking_close:    "</thinking>",
	},
	DialectCase {
		dialect:           Dialect::Qwen3,
		prompt_marker:     "{\"name\":\"function_name\"",
		transcript_marker: "<tool_call>",
		demotion_open:     "<think>\n",
		demotion_close:    "\n</think>",
		thinking_close:    "</think>",
	},
	DialectCase {
		dialect:           Dialect::Xml,
		prompt_marker:     "<invoke name=\"fn\">",
		transcript_marker: "<invoke name=\"matrix_probe\">",
		demotion_open:     "<thinking>\n",
		demotion_close:    "\n</thinking>",
		thinking_close:    "</thinking>",
	},
];

#[derive(Debug)]
struct ObservedCall {
	name: String,
	args: Value,
	raw:  Option<Bytes>,
}

#[derive(Debug)]
struct Observation {
	text:            String,
	thinking:        String,
	thinking_starts: usize,
	thinking_ends:   usize,
	argument_deltas: usize,
	calls:           Vec<ObservedCall>,
}

const fn matrix_tool(schema: &Value) -> InbandTool<'_> {
	InbandTool::new(
		"matrix_probe",
		Some("Probe every dialect with structured arguments."),
		schema,
		&[],
	)
}

fn tool_schema() -> Value {
	json!({
		"type": "object",
		"properties": {
			"path": {"type": "string"},
			"count": {"type": "integer"},
			"enabled": {"type": "boolean"},
			"tags": {"type": "array", "items": {"type": "string"}},
			"meta": {
				"type": "object",
				"properties": {"depth": {"type": "integer"}},
				"required": ["depth"]
			}
		},
		"required": ["path", "count", "enabled", "tags", "meta"]
	})
}

fn tool_call(id: CallId) -> ToolCall {
	ToolCall::builder()
		.id(id)
		.name("matrix_probe".into())
		.args_json(Bytes::from_static(TOOL_ARGS.as_bytes()))
		.thought_signature(Bytes::new())
		.build()
}

fn item(kind: ItemKind) -> Item {
	Item::builder()
		.seq(0)
		.kind(kind)
		.props(Default::default())
		.build()
}

fn transcript() -> Thread {
	let call_id = CallId::new();
	Thread::builder()
		.items(vec![
			item(ItemKind::Message(
				Message::builder()
					.role(Role::System)
					.parts(vec![Part::Text("matrix system policy".into())])
					.build(),
			)),
			item(ItemKind::Message(
				Message::builder()
					.role(Role::User)
					.parts(vec![Part::Text("matrix user question".into())])
					.build(),
			)),
			item(ItemKind::Message(
				Message::builder()
					.role(Role::Assistant)
					.parts(vec![
						Part::Thinking(
							Thinking::builder()
								.text(PRIVATE_THOUGHT.into())
								.signature(Bytes::new())
								.redacted(false)
								.build(),
						),
						Part::Text("matrix assistant prelude".into()),
					])
					.build(),
			)),
			item(ItemKind::ToolCall(tool_call(call_id))),
			item(ItemKind::ToolResult(
				ToolResult::builder()
					.call_id(call_id)
					.name("matrix_probe".into())
					.parts(vec![Part::Text("matrix tool output".into())])
					.is_error(false)
					.build(),
			)),
			item(ItemKind::Message(
				Message::builder()
					.role(Role::Assistant)
					.parts(vec![Part::Text("matrix final answer".into())])
					.build(),
			)),
		])
		.build()
}

fn scan_chunks(
	dialect: Dialect,
	input: &[u8],
	chunks: &[usize],
	tool: InbandTool<'_>,
	include_raw_tool: bool,
) -> Observation {
	let tools = [tool];
	let mut options = ScannerOptions::new(&tools);
	options.include_raw_tool = include_raw_tool;
	let mut scanner = create_scanner(dialect, options);
	let mut events = Vec::new();
	let mut offset = 0;
	for &length in chunks {
		assert!(length > 0, "{dialect}: matrix chunks must be nonempty");
		events.extend(scanner.feed(Bytes::copy_from_slice(&input[offset..offset + length])));
		offset += length;
	}
	assert_eq!(offset, input.len(), "{dialect}: chunks must consume the fixture");
	events.extend(scanner.flush());

	let mut observation = Observation {
		text:            String::new(),
		thinking:        String::new(),
		thinking_starts: 0,
		thinking_ends:   0,
		argument_deltas: 0,
		calls:           Vec::new(),
	};
	let mut active = None;
	for event in events {
		match event {
			ScanEvent::Text(bytes) => {
				observation
					.text
					.push_str(std::str::from_utf8(&bytes).expect("scanner text is UTF-8"));
			},
			ScanEvent::ThinkingStart => observation.thinking_starts += 1,
			ScanEvent::ThinkingDelta(bytes) => observation
				.thinking
				.push_str(std::str::from_utf8(&bytes).expect("scanner thinking is UTF-8")),
			ScanEvent::ThinkingEnd { signature } => {
				assert!(signature.is_empty(), "{dialect}: rendered fixture has no thinking signature");
				observation.thinking_ends += 1;
			},
			ScanEvent::ToolStart { id, name } => {
				assert!(
					active.replace((id, name.to_string())).is_none(),
					"{dialect}: nested tool start"
				);
			},
			ScanEvent::ToolArgumentDelta { id, delta } => {
				let (active_id, _) = active
					.as_ref()
					.expect("argument delta must follow tool start");
				assert_eq!(&id, active_id, "{dialect}: argument delta correlation");
				assert!(!delta.is_empty(), "{dialect}: empty argument delta");
				observation.argument_deltas += 1;
			},
			ScanEvent::ToolEnd { id, name, args_json, raw_block } => {
				let (active_id, active_name) = active.take().expect("tool end must follow tool start");
				assert_eq!(id, active_id, "{dialect}: tool end correlation");
				assert_eq!(name.as_str(), active_name, "{dialect}: tool name correlation");
				observation.calls.push(ObservedCall {
					name: name.to_string(),
					args: serde_json::from_slice(&args_json).expect("tool arguments are JSON"),
					raw:  raw_block,
				});
			},
			_ => {},
		}
	}
	assert!(active.is_none(), "{dialect}: scanner left an active call after flush");
	observation
}

fn assert_tool_cell(
	case: DialectCase,
	mode: &str,
	observation: Observation,
	expected_args: &Value,
	expect_raw: bool,
) {
	let dialect = case.dialect;
	assert!(
		observation.text.trim().is_empty(),
		"{dialect} {mode}: leaked tool text: {:?}",
		observation.text
	);
	assert!(observation.thinking.is_empty(), "{dialect} {mode}: fabricated thinking");
	assert_eq!(observation.thinking_starts, 0, "{dialect} {mode}: fabricated thinking start");
	assert_eq!(observation.thinking_ends, 0, "{dialect} {mode}: fabricated thinking end");
	assert!(observation.argument_deltas > 0, "{dialect} {mode}: no argument deltas");
	assert_eq!(observation.calls.len(), 1, "{dialect} {mode}: completed calls");
	let call = &observation.calls[0];
	assert_eq!(call.name, "matrix_probe", "{dialect} {mode}: call name");
	assert_eq!(&call.args, expected_args, "{dialect} {mode}: structured arguments");
	if expect_raw {
		let raw = call
			.raw
			.as_ref()
			.unwrap_or_else(|| panic!("{dialect} {mode}: missing raw block"));
		assert!(!raw.is_empty(), "{dialect} {mode}: empty raw block");
		assert!(
			String::from_utf8_lossy(raw).contains("matrix_probe"),
			"{dialect} {mode}: raw block omitted tool name"
		);
	} else {
		assert!(call.raw.is_none(), "{dialect} {mode}: raw block ignored disabled option");
	}
}

#[test]
fn explicit_eleven_dialect_matrix_covers_prompt_transcript_demotion_and_scanner_modes() {
	let schema = tool_schema();
	let tool = matrix_tool(&schema);
	let expected_args: Value = serde_json::from_str(TOOL_ARGS).unwrap();
	let thread = transcript();
	let dialects: HashSet<_> = CASES.iter().map(|case| case.dialect).collect();
	assert_eq!(dialects.len(), 11, "matrix rows must be unique");
	assert_eq!(
		dialects,
		Dialect::ALL.into_iter().collect(),
		"matrix must name every public dialect"
	);

	for case in CASES {
		let dialect = case.dialect;

		let mut prompt = String::new();
		write_inband_tool_prompt(&mut prompt, &[tool], dialect).expect("prompt inventory renders");
		assert!(prompt.starts_with("# Tools\n"), "{dialect}: prompt prefix");
		assert!(prompt.contains("\"name\":\"matrix_probe\""), "{dialect}: prompt tool name");
		assert!(
			prompt.contains("Probe every dialect with structured arguments."),
			"{dialect}: prompt description"
		);
		assert!(prompt.contains("\"enabled\":{\"type\":\"boolean\"}"), "{dialect}: prompt schema");
		assert!(prompt.contains(case.prompt_marker), "{dialect}: dialect guide marker");
		assert!(prompt.ends_with(dialect_guide(dialect).trim()), "{dialect}: complete dialect guide");

		let mut rendered_transcript = String::new();
		render_transcript(
			&mut rendered_transcript,
			dialect,
			&thread,
			DialectRenderOptions::new(&[tool]),
		)
		.expect("transcript renders");
		for required in [
			"matrix system policy",
			"matrix user question",
			PRIVATE_THOUGHT,
			"matrix assistant prelude",
			"matrix_probe",
			"matrix tool output",
			"matrix final answer",
			case.transcript_marker,
		] {
			assert!(
				rendered_transcript.contains(required),
				"{dialect}: transcript omitted {required:?}"
			);
		}

		let mut demoted = String::new();
		render_demoted_thinking(&mut demoted, dialect, PRIVATE_THOUGHT).expect("thinking demotes");
		assert!(demoted.contains(PRIVATE_THOUGHT), "{dialect}: demotion lost content");
		if dialect == Dialect::Anthropic {
			assert_eq!(demoted, PRIVATE_THOUGHT, "{dialect}: demotion must be bare prose");
		} else {
			assert!(demoted.starts_with(case.demotion_open), "{dialect}: demotion opening");
			assert!(demoted.ends_with(case.demotion_close), "{dialect}: demotion closing");
		}

		let call = tool_call(CallId::new());
		let mut rendered_call = String::new();
		render_assistant_tool_calls(
			&mut rendered_call,
			dialect,
			&[call],
			DialectRenderOptions::new(&[tool]),
		)
		.expect("assistant tool call renders");
		assert!(!rendered_call.is_empty(), "{dialect}: empty tool rendering");
		let bytes = rendered_call.as_bytes();

		assert_tool_cell(
			case,
			"whole",
			scan_chunks(dialect, bytes, &[bytes.len()], tool, true),
			&expected_args,
			true,
		);
		assert_tool_cell(
			case,
			"byte",
			scan_chunks(dialect, bytes, &vec![1; bytes.len()], tool, true),
			&expected_args,
			true,
		);
		for split in 1..bytes.len() {
			assert_tool_cell(
				case,
				&format!("split-{split}"),
				scan_chunks(dialect, bytes, &[split, bytes.len() - split], tool, true),
				&expected_args,
				true,
			);
		}
		assert_tool_cell(
			case,
			"raw-disabled",
			scan_chunks(dialect, bytes, &[bytes.len()], tool, false),
			&expected_args,
			false,
		);

		let mut unterminated = String::new();
		render_thinking(&mut unterminated, dialect, PRIVATE_THOUGHT).expect("thinking renders");
		assert!(unterminated.ends_with(case.thinking_close), "{dialect}: native thinking closer");
		unterminated.truncate(unterminated.len() - case.thinking_close.len());
		let tools = [tool];
		let mut scanner = create_scanner(dialect, ScannerOptions::new(&tools));
		let before_flush = scanner.feed(Bytes::copy_from_slice(unterminated.as_bytes()));
		assert!(
			!before_flush
				.iter()
				.any(|event| matches!(event, ScanEvent::ThinkingEnd { .. })),
			"{dialect}: unterminated thought closed before flush"
		);
		let after_flush = scanner.flush();
		assert_eq!(
			after_flush
				.iter()
				.filter(|event| matches!(event, ScanEvent::ThinkingEnd { .. }))
				.count(),
			1,
			"{dialect}: flush must close exactly one thought"
		);
		let thinking = before_flush
			.into_iter()
			.chain(after_flush)
			.filter_map(|event| match event {
				ScanEvent::ThinkingDelta(bytes) => Some(bytes),
				_ => None,
			})
			.fold(String::new(), |mut text, bytes| {
				text.push_str(std::str::from_utf8(&bytes).expect("thinking is UTF-8"));
				text
			});
		assert!(thinking.contains(PRIVATE_THOUGHT), "{dialect}: flush lost unterminated thought");
	}
}
