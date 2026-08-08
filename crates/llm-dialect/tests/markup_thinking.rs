//! Markup-based thinking scanner and history projection fixtures.

use bytes::Bytes;
use omp_llm_dialect::{
	Dialect, ScanEvent, ScannerOptions, demotion::render_demoted_thinking, factory::create_scanner,
	history::project_inband_history, thinking::ThinkingScanner, types::DialectRenderOptions,
};
use omp_llm_types::{Item, ItemKind, Message, Part, Props, Role, Thinking, Thread};

#[derive(Debug, Default, Eq, PartialEq)]
struct Recovered {
	text:     String,
	thinking: String,
	starts:   usize,
	ends:     usize,
}

fn collect(events: impl IntoIterator<Item = ScanEvent>) -> Recovered {
	let mut recovered = Recovered::default();
	for event in events {
		match event {
			ScanEvent::Text(delta) => recovered
				.text
				.push_str(std::str::from_utf8(&delta).unwrap()),
			ScanEvent::ThinkingStart => recovered.starts += 1,
			ScanEvent::ThinkingDelta(delta) => {
				recovered
					.thinking
					.push_str(std::str::from_utf8(&delta).unwrap());
			},
			ScanEvent::ThinkingEnd { signature } => {
				assert!(signature.is_empty());
				recovered.ends += 1;
			},
			_ => {},
		}
	}
	recovered
}

fn heal(chunks: &[&[u8]]) -> Recovered {
	let mut scanner = ThinkingScanner::new();
	let mut events = Vec::new();
	for chunk in chunks {
		events.extend(scanner.feed(Bytes::copy_from_slice(chunk)));
	}
	events.extend(scanner.flush());
	collect(events)
}

fn assert_healed_whole_and_at_every_split(input: &str, text: &str, thinking: &str) {
	let expected =
		Recovered { text: text.into(), thinking: thinking.into(), starts: 1, ends: 1 };
	assert_eq!(heal(&[input.as_bytes()]), expected, "whole input: {input:?}");

	for split in 0..=input.len() {
		assert_eq!(
			heal(&[&input.as_bytes()[..split], &input.as_bytes()[split..]]),
			expected,
			"split {split}: {input:?}",
		);
	}

	let bytes = input.as_bytes();
	let one_byte_chunks = bytes.iter().map(std::slice::from_ref).collect::<Vec<_>>();
	assert_eq!(heal(&one_byte_chunks), expected, "one-byte chunks: {input:?}");
}

#[test]
fn leaked_reasoning_markup_is_repaired_whole_and_across_every_marker_split() {
	for (input, text, thinking) in [
		("before <think>plan</think> after", "before  after", "plan"),
		("before <thinking>plan</thinking> after", "before  after", "plan"),
		("before <scratchpad>plan</scratchpad> after", "before  after", "plan"),
		("before ```thinking\nplan\n```after", "before after", "plan\n"),
		("before <|channel>thought\nplan<channel|> after", "before  after", "plan"),
		(
			concat!("before <|start|>assistant<|channel|>analysis", "<|message|>plan<|end|> after"),
			"before  after",
			"plan",
		),
		("before <|channel|>analysis<|message|>plan<|end|> after", "before  after", "plan"),
	] {
		assert_healed_whole_and_at_every_split(input, text, thinking);
		assert!(!text.contains("think"));
		assert!(!text.contains("channel"));
		assert!(!text.contains("scratchpad"));
	}
}

#[test]
fn literal_reasoning_markup_in_markdown_never_becomes_private_reasoning() {
	for literal in [
		"prefix `<think>` suffix",
		"```md\n<think>literal</think>\n```\nafter",
		"   ```md\nconst fence = '```';\n<think>literal</think>\n   ```\nafter",
		"see <div>content</div> end",
		"if a < b:\n    return a",
	] {
		for split in 0..=literal.len() {
			assert_eq!(
				heal(&[&literal.as_bytes()[..split], &literal.as_bytes()[split..]]),
				Recovered { text: literal.into(), ..Recovered::default() },
				"split {split}: {literal:?}",
			);
		}
	}
}

fn scan_dialect(dialect: Dialect, input: &str, parse_thinking: bool, split: usize) -> Recovered {
	let mut options = ScannerOptions::default();
	options.parse_thinking = parse_thinking;
	let mut scanner = create_scanner(dialect, options);
	let mut events = scanner.feed(Bytes::copy_from_slice(&input.as_bytes()[..split]));
	events.extend(scanner.feed(Bytes::copy_from_slice(&input.as_bytes()[split..])));
	events.extend(scanner.flush());
	collect(events)
}

#[test]
fn reasoning_parser_policy_has_correct_enabled_and_disabled_outcome_for_every_dialect() {
	let cases = [
		(Dialect::Anthropic, "<thinking>private</thinking>visible", true),
		(Dialect::DeepSeek, "<think>private</think>visible", true),
		(Dialect::Gemini, "```thinking\nprivate\n```visible", true),
		(Dialect::Gemma, "<|channel>thought\nprivate<channel|>visible", true),
		(Dialect::Glm, "<think>private</think>visible", true),
		(
			Dialect::Harmony,
			concat!(
				"<\x7cstart\x7c>assistant<\x7cchannel\x7c>analysis",
				"<\x7cmessage\x7c>private<\x7cend\x7c>",
				"<\x7cstart\x7c>assistant<\x7cchannel\x7c>final",
				"<\x7cmessage\x7c>visible<\x7cend\x7c>"
			),
			false,
		),
		(Dialect::Hermes, "<think>private</think>visible", true),
		(Dialect::Kimi, "<think>private</think>visible", true),
		(Dialect::MiniMax, "<thinking>private</thinking>visible", true),
		(Dialect::Qwen3, "<think>private</think>visible", true),
		(Dialect::Xml, "<thinking>private</thinking>visible", true),
	];

	for (dialect, input, disabled_is_literal) in cases {
		for split in 0..=input.len() {
			let enabled = scan_dialect(dialect, input, true, split);
			assert_eq!(enabled.text, "visible", "enabled {dialect}, split {split}");
			assert_eq!(enabled.thinking.trim(), "private", "enabled {dialect}, split {split}");
			assert_eq!((enabled.starts, enabled.ends), (1, 1), "enabled {dialect}, split {split}");
			assert!(!enabled.text.contains("private"));
			assert!(!enabled.text.contains("<|"));

			let disabled = scan_dialect(dialect, input, false, split);
			if disabled_is_literal {
				assert_eq!(disabled.text, input, "disabled {dialect}, split {split}");
				assert!(disabled.thinking.is_empty(), "disabled {dialect}, split {split}");
				assert_eq!(
					(disabled.starts, disabled.ends),
					(0, 0),
					"disabled {dialect}, split {split}"
				);
			} else {
				assert_eq!(disabled, enabled, "native channel {dialect}, split {split}");
			}
		}
	}
}

#[test]
fn cross_model_reasoning_demotion_uses_target_safe_history_syntax() {
	const REASONING: &str = "The plan uses a private chain of thought.";
	for dialect in Dialect::ALL {
		let mut rendered = String::new();
		render_demoted_thinking(&mut rendered, dialect, REASONING).unwrap();
		assert!(rendered.contains(REASONING), "{dialect}: {rendered:?}");
		match dialect {
			Dialect::Anthropic => {
				assert_eq!(rendered, REASONING);
				assert!(!rendered.contains("<think"));
			},
			Dialect::Harmony | Dialect::Gemma => {
				assert_eq!(rendered, format!("<think>\n{REASONING}\n</think>"));
				assert!(!rendered.contains("<|"));
			},
			Dialect::Gemini => assert_eq!(rendered, format!("```thinking\n{REASONING}\n```")),
			_ => assert_ne!(rendered, REASONING),
		}
	}

	let mut escaped = String::new();
	render_demoted_thinking(&mut escaped, Dialect::Harmony, "<\x7cchannel\x7c>analysis").unwrap();
	assert_eq!(escaped, "<think>\n<\\|channel\\|>analysis\n</think>");
	assert!(!escaped.contains("<\x7cchannel\x7c>"));
}

fn assistant_history() -> Thread {
	Thread::builder()
		.items(vec![
			Item::builder()
				.seq(7)
				.kind(ItemKind::Message(
					Message::builder()
						.role(Role::Assistant)
						.parts(vec![
							Part::Thinking(
								Thinking::builder()
									.text("inspect the forecast".into())
									.signature(Bytes::from_static(b"foreign-signature"))
									.redacted(false)
									.build(),
							),
							Part::Text("Checking the forecast.".into()),
						])
						.build(),
				))
				.props(Props::default())
				.build(),
		])
		.build()
}

#[test]
fn inband_history_transformation_extracts_reasoning_before_visible_reply() {
	let projected = project_inband_history(
		&assistant_history(),
		Dialect::Gemini,
		DialectRenderOptions::default(),
	)
	.unwrap();
	assert_eq!(projected.items.len(), 1);
	let ItemKind::Message(message) = &projected.items[0].kind else {
		panic!("projected history item was not a message");
	};
	assert_eq!(message.role, Role::Assistant);
	assert_eq!(message.parts.len(), 1);
	let Part::Text(text) = &message.parts[0] else {
		panic!("projected assistant history was not flattened to text");
	};
	assert_eq!(text.as_str(), "```thinking\ninspect the forecast\n```Checking the forecast.");
	assert!(
		!message
			.parts
			.iter()
			.any(|part| matches!(part, Part::Thinking(_)))
	);
}
