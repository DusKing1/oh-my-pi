//! Thinking-channel scanning and demotion behavior across dialects.

use bytes::Bytes;
use omp_llm_dialect::{
	Dialect, ScanEvent, ScannerOptions, demotion::render_demoted_thinking, factory::create_scanner,
};

fn scan(dialect: Dialect, input: &str, parse_thinking: bool) -> (String, String, usize, usize) {
	let mut options = ScannerOptions::default();
	options.parse_thinking = parse_thinking;
	let mut scanner = create_scanner(dialect, options);
	let mut events = Vec::new();
	for byte in input.as_bytes() {
		events.extend(scanner.feed(Bytes::copy_from_slice(std::slice::from_ref(byte))));
	}
	events.extend(scanner.flush());
	let mut visible = String::new();
	let mut thinking = String::new();
	let mut starts = 0;
	let mut ends = 0;
	for event in events {
		match event {
			ScanEvent::Text(value) => visible.push_str(std::str::from_utf8(&value).unwrap()),
			ScanEvent::ThinkingStart => starts += 1,
			ScanEvent::ThinkingDelta(value) => thinking.push_str(std::str::from_utf8(&value).unwrap()),
			ScanEvent::ThinkingEnd { .. } => ends += 1,
			_ => {},
		}
	}
	(visible, thinking, starts, ends)
}

#[test]
fn gemma_gemini_and_kimi_route_thought_channels_without_leaking() {
	let cases = [
		(
			Dialect::Gemma,
			"<|channel>thought\nlet me reason\n<channel|>The answer.",
			"let me reason",
			"The answer.",
		),
		(
			Dialect::Gemini,
			"```thinking\nlet me reason\n```\nThe answer.",
			"let me reason",
			"The answer.",
		),
		(Dialect::Kimi, "<think>let me reason</think>The answer.", "let me reason", "The answer."),
	];
	for (dialect, input, private, answer) in cases {
		let (visible, thinking, starts, ends) = scan(dialect, input, true);
		assert!(thinking.contains(private), "{dialect}: {thinking:?}");
		assert_eq!(visible.trim(), answer, "{dialect}");
		assert!(!visible.contains(private), "{dialect}");
		assert_eq!((starts, ends), (1, 1), "{dialect}");
	}
}

#[test]
fn disabled_thinking_parsing_keeps_literal_envelopes_visible() {
	for (dialect, input, marker) in [
		(Dialect::Gemma, "<|channel>thought\nx\n<channel|>reply", "<|channel>thought"),
		(Dialect::Gemini, "```thinking\nx\n```\nreply", "```thinking"),
		(Dialect::Kimi, "<think>x</think>reply", "<think>"),
	] {
		let (visible, thinking, starts, ends) = scan(dialect, input, false);
		assert!(visible.contains(marker), "{dialect}: {visible:?}");
		assert_eq!(thinking, "");
		assert_eq!((starts, ends), (0, 0));
	}
}

#[test]
fn flush_closes_each_unterminated_thinking_channel_once() {
	for (dialect, input) in [
		(Dialect::DeepSeek, "<think>partial"),
		(Dialect::Gemini, "```thinking\npartial"),
		(Dialect::Gemma, "<|channel>thought\npartial"),
		(Dialect::Glm, "<think>partial"),
		(Dialect::Kimi, "<think>partial"),
		(Dialect::Qwen3, "<think>partial"),
	] {
		let (visible, thinking, starts, ends) = scan(dialect, input, true);
		assert!(visible.is_empty(), "{dialect}: {visible:?}");
		assert_eq!(thinking, "partial", "{dialect}");
		assert_eq!((starts, ends), (1, 1), "{dialect}");
	}
}

#[test]
fn cross_provider_demotion_uses_target_safe_syntax() {
	for dialect in Dialect::ALL {
		let mut rendered = String::new();
		render_demoted_thinking(&mut rendered, dialect, "private reasoning").unwrap();
		assert!(rendered.contains("private reasoning"), "{dialect}");
		match dialect {
			Dialect::Anthropic => assert_eq!(rendered, "private reasoning"),
			Dialect::Harmony | Dialect::Gemma => {
				assert_eq!(rendered, "<think>\nprivate reasoning\n</think>");
			},
			_ => assert_ne!(rendered, "private reasoning"),
		}
	}

	let mut escaped = String::new();
	render_demoted_thinking(&mut escaped, Dialect::Harmony, "<|channel|>analysis").unwrap();
	assert_eq!(escaped, "<think>\n<\\|channel\\|>analysis\n</think>");
}
