//! Gemini and Gemma dialect scanner fixtures.

use bytes::Bytes;
use omp_llm_dialect::{Dialect, ScanEvent, ScannerOptions, factory::create_scanner};
use serde_json::{Value, json};

fn calls(dialect: Dialect, input: &str, bytewise: bool) -> Vec<(String, Value)> {
	let mut scanner = create_scanner(dialect, ScannerOptions::default());
	let mut events = Vec::new();
	if bytewise {
		for byte in input.as_bytes() {
			events.extend(scanner.feed(Bytes::copy_from_slice(std::slice::from_ref(byte))));
		}
	} else {
		events.extend(scanner.feed(Bytes::copy_from_slice(input.as_bytes())));
	}
	events.extend(scanner.flush());
	events
		.into_iter()
		.filter_map(|event| match event {
			ScanEvent::ToolEnd { name, args_json, .. } => {
				Some((name.to_string(), serde_json::from_slice(&args_json).unwrap()))
			},
			_ => None,
		})
		.collect()
}

#[test]
fn gemini_pythonic_forms_literals_comments_and_parallel_calls_match_pi() {
	let cases = [
		("```tool_code\nprint(default_api.search(query='rust'))\n```", vec![(
			"search".into(),
			json!({"query":"rust"}),
		)]),
		(
			"```tool_code\ndefault_api.search(query=\"a,b (c)\")\nresult = \
			 default_api.read(path='/tmp/x')\n```",
			vec![
				("search".into(), json!({"query":"a,b (c)"})),
				("read".into(), json!({"path":"/tmp/x"})),
			],
		),
		(
			"```tool_code\nprint(default_api.run(ok=True, missing=None, n=-2.5, xs=[1, 'two'], \
			 cfg={'x': 3}))\n```",
			vec![(
				"run".into(),
				json!({"ok":true,"missing":null,"n":-2.5,"xs":[1,"two"],"cfg":{"x":3}}),
			)],
		),
		("```tool_code\n[default_api.a(x=1), default_api.b(y='two')]\n```", vec![
			("a".into(), json!({"x":1})),
			("b".into(), json!({"y":"two"})),
		]),
	];
	for (input, expected) in cases {
		assert_eq!(calls(Dialect::Gemini, input, false), expected);
		assert_eq!(calls(Dialect::Gemini, input, true), expected);
	}
}

#[test]
fn gemma_token_calls_accept_json5_shapes_and_stream_identically() {
	let cases = [
		("<|tool_call>call:search{query:<|\"|>rust<|\"|>}<tool_call|>", vec![(
			"search".into(),
			json!({"query":"rust"}),
		)]),
		("<|tool_call>call:run{x:1, ok:true, missing:null, nested:{items:[1,2]}}<tool_call|>", vec![
			("run".into(), json!({"x":1,"ok":true,"missing":null,"nested":{"items":[1,2]}})),
		]),
	];
	for (input, expected) in cases {
		assert_eq!(calls(Dialect::Gemma, input, false), expected);
		assert_eq!(calls(Dialect::Gemma, input, true), expected);
	}
}

#[test]
fn prose_outside_owned_blocks_is_not_fabricated_into_calls() {
	assert!(calls(Dialect::Gemini, "default_api.search(query='outside')", true).is_empty());
	assert!(calls(Dialect::Gemma, "call:search{query:'outside'}", true).is_empty());
}
