//! Exact transcript rendering fixtures for every supported dialect.

use bytes::Bytes;
use omp_llm_dialect::{
	Dialect, DialectRenderOptions, DialectToolResult,
	demotion::render_demoted_thinking,
	history::project_inband_history,
	rendering::{
		render_thinking, render_tool_results, render_transcript, write_escaped_harmony_json,
		write_escaped_harmony_text,
	},
};
use omp_llm_types::{
	BlobPart, Item, ItemKind, Message, Part, Props, Role, Thinking, ToolCall, ToolResult,
	ids::CallId,
};

fn item(seq: u64, kind: ItemKind) -> Item {
	Item::builder()
		.seq(seq)
		.kind(kind)
		.props(Props::default())
		.build()
}

fn message(seq: u64, role: Role, parts: Vec<Part>) -> Item {
	item(seq, ItemKind::Message(Message::builder().role(role).parts(parts).build()))
}

fn text_message(seq: u64, role: Role, text: &str) -> Item {
	message(seq, role, vec![Part::Text(text.into())])
}

fn thinking(text: &str) -> Part {
	Part::Thinking(
		Thinking::builder()
			.text(text.into())
			.signature(Bytes::new())
			.redacted(false)
			.build(),
	)
}

fn render(dialect: Dialect, thread: &omp_llm_types::Thread) -> String {
	let mut output = String::new();
	render_transcript(&mut output, dialect, thread, DialectRenderOptions::default()).unwrap();
	output
}

#[test]
fn complete_transcripts_use_the_target_dialects_exact_turn_framing() {
	let thread = omp_llm_types::Thread::builder()
		.items(vec![text_message(1, Role::User, "hi"), text_message(2, Role::Assistant, "ok")])
		.build();
	let cases = [
		(Dialect::Anthropic, "\n\nHuman: hi\n\nAssistant: ok"),
		(Dialect::MiniMax, "\n\nHuman: hi\n\nAssistant: ok"),
		(Dialect::Xml, "\n\nHuman: hi\n\nAssistant: ok"),
		(
			Dialect::Hermes,
			"<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nok<|im_end|>\n",
		),
		(
			Dialect::Qwen3,
			"<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nok<|im_end|>\n",
		),
		(
			Dialect::DeepSeek,
			"<｜begin▁of▁sentence｜><｜User｜>hi<｜Assistant｜>ok<｜end▁of▁sentence｜>",
		),
		(
			Dialect::Gemini,
			"<bos><start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\nok<end_of_turn>\n",
		),
		(Dialect::Gemma, "<bos><|turn>user\nhi<turn|><|turn>model\nok<turn|>"),
		(Dialect::Glm, "[gMASK]<sop><|user|>\nhi<|assistant|>\nok"),
		(
			Dialect::Harmony,
			"<\u{7c}start\u{7c}>user<\u{7c}message\u{7c}>hi<\u{7c}end\u{7c}><\u{7c}start\u{7c}>assistant<\u{7c}channel\u{7c}>final<\u{7c}message\u{7c}>ok<\u{7c}end\u{7c}>",
		),
		(
			Dialect::Kimi,
			"<|im_user|>user<|im_middle|>hi<|im_end|><|im_assistant|>assistant<|im_middle|>ok<|im_end|>",
		),
	];

	for (dialect, expected) in cases {
		assert_eq!(render(dialect, &thread), expected, "{dialect}");
	}
}

#[test]
fn full_reasoning_tool_round_trip_matches_borrowed_pi_transcripts() {
	let call_id = CallId::new();
	let thread = omp_llm_types::Thread::builder()
		.items(vec![
			text_message(1, Role::User, "Find pi"),
			message(2, Role::Assistant, vec![
				thinking("I should search."),
				Part::Text("Searching.".into()),
			]),
			item(
				3,
				ItemKind::ToolCall(
					ToolCall::builder()
						.id(call_id)
						.name("search".into())
						.args_json(Bytes::from_static(br#"{"query":"pi"}"#))
						.thought_signature(Bytes::new())
						.build(),
				),
			),
			item(
				4,
				ItemKind::ToolResult(
					ToolResult::builder()
						.call_id(call_id)
						.name("search".into())
						.parts(vec![Part::Text("result".into())])
						.is_error(false)
						.build(),
				),
			),
			text_message(5, Role::Assistant, "Done."),
		])
		.build();
	let cases = [
		(
			Dialect::Harmony,
			concat!(
				"<\u{7c}start\u{7c}>user<\u{7c}message\u{7c}>Find pi<\u{7c}end\u{7c}>",
				"<\u{7c}start\u{7c}>assistant<\u{7c}channel\u{7c}>analysis<\u{7c}message\u{7c}>I \
				 should search.<\u{7c}end\u{7c}>",
				"<\u{7c}start\u{7c}>assistant<\u{7c}channel\u{7c}>final<\u{7c}message\u{7c}>Searching.\
				 <\u{7c}end\u{7c}>",
				"<\u{7c}start\u{7c}>assistant<\u{7c}channel\u{7c}>commentary to=functions.search",
				"<\u{7c}message\u{7c}>{\"query\":\"pi\"}<\u{7c}call\u{7c}>",
				"<\u{7c}start\u{7c}>functions.search to=assistant<\u{7c}channel\u{7c}>commentary",
				"<\u{7c}message\u{7c}>result<\u{7c}end\u{7c}>",
				"<\u{7c}start\u{7c}>assistant<\u{7c}channel\u{7c}>final<\u{7c}message\u{7c}>Done.",
				"<\u{7c}end\u{7c}>"
			),
		),
		(
			Dialect::Qwen3,
			concat!(
				"<|im_start|>user\nFind pi<|im_end|>\n",
				"<|im_start|>assistant\n<think>\nI should search.\n</think>Searching.",
				"<tool_call>\n{\"name\":\"search\",\"arguments\":{\"query\":\"pi\"}}\n</tool_call>",
				"<|im_end|>\n<|im_start|>user\n<tool_response>\nresult\n</tool_response>",
				"<|im_end|>\n<|im_start|>assistant\nDone.<|im_end|>\n"
			),
		),
		(
			Dialect::Glm,
			concat!(
				"[gMASK]<sop><|user|>\nFind pi<|assistant|>\n<think>\n",
				"I should search.\n</think>Searching.<tool_call>search\n",
				"<arg_key>query</arg_key>\n<arg_value>\"pi\"</arg_value>\n",
				"</tool_call><|observation|>\n<tool_response>\nresult\n</tool_response>",
				"<|assistant|>\nDone."
			),
		),
		(
			Dialect::Anthropic,
			concat!(
				"\n\nHuman: Find pi\n\nAssistant: <thinking>\nI should search.\n</thinking>",
				"Searching.<function_calls>\n<invoke name=\"search\"><parameter name=\"query\">",
				"\"pi\"</parameter></invoke>\n</function_calls>\n\nHuman: <function_results>\n",
				"<result>\n<tool_name>search</tool_name>\n<stdout>result</stdout>\n</result>\n",
				"</function_results>\n\nAssistant: Done."
			),
		),
	];

	for (dialect, expected) in cases {
		assert_eq!(render(dialect, &thread), expected, "{dialect}");
	}
}

#[test]
fn empty_transcripts_and_empty_reasoning_render_nothing() {
	let thread = omp_llm_types::Thread::default();
	for dialect in Dialect::ALL {
		assert_eq!(render(dialect, &thread), "", "empty transcript for {dialect}");
		let mut output = String::new();
		render_thinking(&mut output, dialect, "").unwrap();
		assert_eq!(output, "", "empty reasoning for {dialect}");
	}

	let empty_assistant = omp_llm_types::Thread::builder()
		.items(vec![message(1, Role::Assistant, Vec::new())])
		.build();
	assert_eq!(
		render(Dialect::Harmony, &empty_assistant),
		concat!(
			"<\u{7c}start\u{7c}>assistant<\u{7c}channel\u{7c}>final",
			"<\u{7c}message\u{7c}><\u{7c}end\u{7c}>"
		)
	);
}

#[test]
fn multiline_reasoning_uses_each_targets_exact_native_envelope() {
	let cases = [
		(Dialect::Anthropic, "<thinking>\nfirst\nsecond\n</thinking>"),
		(Dialect::Hermes, "<thinking>\nfirst\nsecond\n</thinking>"),
		(Dialect::MiniMax, "<thinking>\nfirst\nsecond\n</thinking>"),
		(Dialect::Xml, "<thinking>\nfirst\nsecond\n</thinking>"),
		(Dialect::DeepSeek, "<think>\nfirst\nsecond\n</think>"),
		(Dialect::Glm, "<think>\nfirst\nsecond\n</think>"),
		(Dialect::Kimi, "<think>\nfirst\nsecond\n</think>"),
		(Dialect::Qwen3, "<think>\nfirst\nsecond\n</think>"),
		(Dialect::Gemini, "```thinking\nfirst\nsecond\n```"),
		(Dialect::Gemma, "<|channel>thoughtfirst\nsecond<channel|>"),
		(
			Dialect::Harmony,
			concat!(
				"<\u{7c}start\u{7c}>assistant<\u{7c}channel\u{7c}>analysis",
				"<\u{7c}message\u{7c}>first\nsecond<\u{7c}end\u{7c}>"
			),
		),
	];

	for (dialect, expected) in cases {
		let mut output = String::new();
		render_thinking(&mut output, dialect, "first\nsecond").unwrap();
		assert_eq!(output, expected, "{dialect}");
	}
}

#[test]
fn consecutive_results_keep_exact_grouping_and_multiline_content() {
	let results = [
		DialectToolResult::new("call-a", "first", 0, "one", false),
		DialectToolResult::new("call-b", "second", 1, "two\nlines", true),
	];
	let cases = [
		(
			Dialect::Anthropic,
			concat!(
				"<function_results>\n<result>\n<tool_name>first</tool_name>\n",
				"<stdout>one</stdout>\n</result>\n<error>\n<tool_name>second</tool_name>\n",
				"<stderr>two\nlines</stderr>\n</error>\n</function_results>"
			),
		),
		(
			Dialect::MiniMax,
			concat!(
				"<function_results>\n<result>\n<tool_name>first</tool_name>\n",
				"<stdout>one</stdout>\n</result>\n<error>\n<tool_name>second</tool_name>\n",
				"<stderr>two\nlines</stderr>\n</error>\n</function_results>"
			),
		),
		(
			Dialect::DeepSeek,
			concat!(
				"<｜tool▁output▁begin｜>one<｜tool▁output▁end｜>\n",
				"<｜tool▁output▁begin｜>two\nlines<｜tool▁output▁end｜>"
			),
		),
		(Dialect::Gemini, "```tool_outputs\none\n```\n```tool_outputs\ntwo\nlines\n```"),
		(
			Dialect::Gemma,
			"<|tool_response>response:first{output:<|\"|>one<|\"\
			 |>}<tool_response|><|tool_response>response:second{output:<|\"|>two\nlines<|\"\
			 |>}<tool_response|>",
		),
		(
			Dialect::Glm,
			"<observation>\n<tool_response>\none\n</tool_response>\n<tool_response>\ntwo\nlines\n</\
			 tool_response>\n</observation>",
		),
		(
			Dialect::Hermes,
			"<tool_response>\none\n</tool_response>\n<tool_response>\ntwo\nlines\n</tool_response>",
		),
		(
			Dialect::Qwen3,
			"<tool_response>\none\n</tool_response>\n<tool_response>\ntwo\nlines\n</tool_response>",
		),
		(
			Dialect::Xml,
			"<tool_response>\none\n</tool_response>\n<tool_response>\ntwo\nlines\n</tool_response>",
		),
		(
			Dialect::Harmony,
			concat!(
				"<\u{7c}start\u{7c}>functions.first to=assistant<\u{7c}channel\u{7c}>",
				"commentary<\u{7c}message\u{7c}>one<\u{7c}end\u{7c}>",
				"<\u{7c}start\u{7c}>functions.second to=assistant<\u{7c}channel\u{7c}>",
				"commentary<\u{7c}message\u{7c}>two\nlines<\u{7c}end\u{7c}>"
			),
		),
		(
			Dialect::Kimi,
			"<|im_system|>first<|im_middle|>## Return of \
			 functions.first:0\none<|im_end|><|im_system|>second<|im_middle|>## Return of \
			 functions.second:1\ntwo\nlines<|im_end|>",
		),
	];

	for (dialect, expected) in cases {
		let mut output = String::new();
		render_tool_results(&mut output, dialect, &results, DialectRenderOptions::default()).unwrap();
		assert_eq!(output, expected, "{dialect}");
	}
}

#[test]
fn empty_tool_results_keep_their_target_envelope() {
	let results = [DialectToolResult::new("call-empty", "empty", 0, "", false)];
	let cases = [
		(
			Dialect::Anthropic,
			"<function_results>\n<result>\n<tool_name>empty</tool_name>\n<stdout></stdout>\n</\
			 result>\n</function_results>",
		),
		(
			Dialect::MiniMax,
			"<function_results>\n<result>\n<tool_name>empty</tool_name>\n<stdout></stdout>\n</\
			 result>\n</function_results>",
		),
		(Dialect::DeepSeek, "<｜tool▁output▁begin｜><｜tool▁output▁end｜>"),
		(Dialect::Gemini, "```tool_outputs\n\n```"),
		(Dialect::Gemma, "<|tool_response>response:empty{output:<|\"|><|\"|>}<tool_response|>"),
		(Dialect::Glm, "<observation>\n<tool_response>\n\n</tool_response>\n</observation>"),
		(Dialect::Hermes, "<tool_response>\n\n</tool_response>"),
		(Dialect::Qwen3, "<tool_response>\n\n</tool_response>"),
		(Dialect::Xml, "<tool_response>\n\n</tool_response>"),
		(
			Dialect::Harmony,
			"<\u{7c}start\u{7c}>functions.empty \
			 to=assistant<\u{7c}channel\u{7c}>commentary<\u{7c}message\u{7c}><\u{7c}end\u{7c}>",
		),
		(Dialect::Kimi, "<|im_system|>empty<|im_middle|>## Return of functions.empty:0\n<|im_end|>"),
	];

	for (dialect, expected) in cases {
		let mut output = String::new();
		render_tool_results(&mut output, dialect, &results, DialectRenderOptions::default()).unwrap();
		assert_eq!(output, expected, "{dialect}");
	}
}

#[test]
fn projected_grouped_results_preserve_images_in_source_order() {
	let first_image = BlobPart::builder()
		.hash([1; 32])
		.mime("image/png".into())
		.size(1)
		.inline(Bytes::from_static(b"a"))
		.build();
	let second_image = BlobPart::builder()
		.hash([2; 32])
		.mime("image/jpeg".into())
		.size(1)
		.inline(Bytes::from_static(b"b"))
		.build();
	let first_id = CallId::new();
	let second_id = CallId::new();
	let thread = omp_llm_types::Thread::builder()
		.items(vec![
			item(
				1,
				ItemKind::ToolResult(
					ToolResult::builder()
						.call_id(first_id)
						.name("first".into())
						.parts(vec![Part::Text("one".into()), Part::Blob(first_image.clone())])
						.is_error(false)
						.build(),
				),
			),
			item(
				2,
				ItemKind::ToolResult(
					ToolResult::builder()
						.call_id(second_id)
						.name("second".into())
						.parts(vec![Part::Text("two".into()), Part::Blob(second_image.clone())])
						.is_error(false)
						.build(),
				),
			),
		])
		.build();

	let projected =
		project_inband_history(&thread, Dialect::Gemini, DialectRenderOptions::default()).unwrap();
	assert_eq!(projected.items.len(), 1);
	let ItemKind::Message(projected_message) = &projected.items[0].kind else {
		panic!("grouped results were not projected as one message");
	};
	assert_eq!(projected_message.role, Role::User);
	assert_eq!(projected_message.parts, vec![
		Part::Text("```tool_outputs\none\n```\n```tool_outputs\ntwo\n```".into()),
		Part::Blob(first_image),
		Part::Blob(second_image),
	]);
}

#[test]
fn cross_model_reasoning_is_demoted_with_exact_target_safe_syntax() {
	let cases = [
		(Dialect::Anthropic, "private\nplan"),
		(Dialect::Hermes, "<thinking>\nprivate\nplan\n</thinking>"),
		(Dialect::MiniMax, "<thinking>\nprivate\nplan\n</thinking>"),
		(Dialect::Xml, "<thinking>\nprivate\nplan\n</thinking>"),
		(Dialect::DeepSeek, "<think>\nprivate\nplan\n</think>"),
		(Dialect::Glm, "<think>\nprivate\nplan\n</think>"),
		(Dialect::Harmony, "<think>\nprivate\nplan\n</think>"),
		(Dialect::Kimi, "<think>\nprivate\nplan\n</think>"),
		(Dialect::Qwen3, "<think>\nprivate\nplan\n</think>"),
		(Dialect::Gemini, "```thinking\nprivate\nplan\n```"),
		(Dialect::Gemma, "<think>\nprivate\nplan\n</think>"),
	];

	for (dialect, expected) in cases {
		let mut output = String::new();
		render_demoted_thinking(&mut output, dialect, "private\nplan").unwrap();
		assert_eq!(output, expected, "{dialect}");
	}
}

#[test]
fn harmony_escapes_untrusted_controls_but_preserves_renderer_framing() {
	let mut text = String::new();
	write_escaped_harmony_text(
		&mut text,
		"<\u{7c}start\u{7c}> data <\u{7c}bogus\u{7c}> <\u{7c}end\u{7c}>",
	)
	.unwrap();
	assert_eq!(text, "<\\|start\\|> data <|bogus|> <\\|end\\|>");

	let mut json = String::new();
	write_escaped_harmony_json(
		&mut json,
		"{\"text\":\"<\u{7c}call\u{7c}>\",\"other\":\"<\u{7c}bogus\u{7c}>\"}",
	)
	.unwrap();
	assert_eq!(json, "{\"text\":\"<\\\\|call\\\\|>\",\"other\":\"<|bogus|>\"}");

	let thread = omp_llm_types::Thread::builder()
		.items(vec![
			text_message(1, Role::User, "<\u{7c}start\u{7c}>prompt"),
			message(2, Role::Assistant, vec![
				thinking("<\u{7c}channel\u{7c}>plan"),
				Part::Text("<\u{7c}end\u{7c}>answer".into()),
			]),
		])
		.build();
	assert_eq!(
		render(Dialect::Harmony, &thread),
		concat!(
			"<\u{7c}start\u{7c}>user<\u{7c}message\u{7c}><\\|start\\|>prompt<\u{7c}end\u{7c}>",
			"<\u{7c}start\u{7c}>assistant<\u{7c}channel\u{7c}>analysis<\u{7c}message\u{7c}>",
			"<\\|channel\\|>plan<\u{7c}end\u{7c}>",
			"<\u{7c}start\u{7c}>assistant<\u{7c}channel\u{7c}>final<\u{7c}message\u{7c}>",
			"<\\|end\\|>answer<\u{7c}end\u{7c}>"
		)
	);
}
