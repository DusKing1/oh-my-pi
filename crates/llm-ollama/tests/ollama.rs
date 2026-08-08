//! Ollama native request, stream, and policy behavior.

use std::{collections::BTreeMap, sync::Arc};

use bytes::Bytes;
use omp_core::SmolStr;
use omp_llm_catalog::{TransportId, compat::Compat};
use omp_llm_ollama::OllamaChatCodec;
use omp_llm_transport::{DecodeState, Frame, Transport, ndjson::NdjsonDecoder};
use omp_llm_types::{
	BlobPart, ChatRequest, Effort, Fallback, Feature, Item, ItemKind, Message, Part, Props,
	Reasoning, ResolvedModelPolicy, ResolvedThinkingMode, ResolvedThinkingPolicy, Role, Sampling,
	StopReason, StreamPartKind, Thinking, Thread, ToolCall, ToolChoice, ToolDef, ToolResult,
	TurnEvent, UnsupportedAction, ids::CallId,
};
use serde_json::{Value, json};
use smallvec::smallvec;

fn item(kind: ItemKind) -> Item {
	Item::builder()
		.seq(0)
		.kind(kind)
		.props(Props::default())
		.build()
}

fn message(role: Role, parts: Vec<Part>) -> Item {
	item(ItemKind::Message(Message::builder().role(role).parts(parts).build()))
}

fn fixture_request() -> ChatRequest {
	let call_id: CallId = "01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().unwrap();
	ChatRequest::builder()
		.model(SmolStr::new_static("qwen3.5-cloud"))
		.thread(
			Thread::builder()
				.items(vec![
					message(Role::System, vec![Part::Text("Be precise.".into())]),
					message(Role::User, vec![
						Part::Text("Describe this image".into()),
						Part::Blob(
							BlobPart::builder()
								.hash([7; 32])
								.mime("image/png".into())
								.size(3)
								.inline(Bytes::from_static(&[1, 2, 3]))
								.build(),
						),
					]),
					message(Role::Assistant, vec![
						Part::Thinking(
							Thinking::builder()
								.text("private chain".into())
								.signature(Bytes::new())
								.redacted(false)
								.build(),
						),
						Part::Text("I will check.".into()),
					]),
					item(ItemKind::ToolCall(
						ToolCall::builder()
							.id(call_id)
							.name("weather".into())
							.args_json(Bytes::from_static(br#"{"city":"Paris"}"#))
							.thought_signature(Bytes::new())
							.build(),
					)),
					item(ItemKind::ToolResult(
						ToolResult::builder()
							.call_id(call_id)
							.name("weather".into())
							.parts(vec![Part::Text("sunny".into())])
							.is_error(false)
							.build(),
					)),
				])
				.build(),
		)
		.tools(vec![
			ToolDef::builder()
				.name("weather".into())
				.description("Look up weather".into())
				.schema_json(Bytes::from_static(
					br#"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"#,
				))
				.build(),
			ToolDef::builder()
				.name("unused".into())
				.description("Filtered by named selection".into())
				.schema_json(Bytes::from_static(br#"{"type":"object"}"#))
				.build(),
		])
		.tool_choice(
			Feature::builder()
				.value(ToolChoice::Named("weather".into()))
				.on_unsupported(Fallback::Error)
				.build(),
		)
		.thinking(
			Feature::builder()
				.value(Reasoning::builder().effort(Effort::High).build())
				.on_unsupported(Fallback::Error)
				.build(),
		)
		.sampling(Sampling::builder().max_output_tokens(100_000).build())
		.build()
}

fn effort_policy(
	effort_map: BTreeMap<Effort, SmolStr>,
	omit_max_output_tokens: bool,
) -> Arc<ResolvedModelPolicy> {
	Arc::new(ResolvedModelPolicy {
		thinking: Some(ResolvedThinkingPolicy {
			mode: ResolvedThinkingMode::Effort,
			efforts: smallvec![
				Effort::Minimal,
				Effort::Low,
				Effort::Medium,
				Effort::High,
				Effort::XHigh,
				Effort::Max
			],
			default_effort: None,
			effort_map,
			effort_routing: BTreeMap::new(),
			effort_budgets: BTreeMap::new(),
			supports_display: None,
			suppress_when_off: None,
			requires_effort: None,
		}),
		omit_max_output_tokens: Some(omit_max_output_tokens),
		..ResolvedModelPolicy::default()
	})
}

#[test]
fn recorded_cloud_request_projects_history_tools_images_reasoning_and_cap() {
	let codec = OllamaChatCodec;
	assert_eq!(codec.id(), TransportId::OllamaChat);
	let (body, unsupported) = codec
		.encode(&fixture_request(), &Compat::default())
		.unwrap();
	let actual: Value = serde_json::from_slice(&body).unwrap();
	let expected: Value = serde_json::from_str(include_str!("fixtures/cloud_request.json")).unwrap();
	assert_eq!(actual, expected);
	assert!(
		unsupported
			.iter()
			.any(|entry| entry.what == "thread.assistant.thinking")
	);
	assert!(
		unsupported
			.iter()
			.any(|entry| entry.what == "thread.tool_result.call_id")
	);
	assert!(unsupported.iter().any(|entry| {
		entry.what == "sampling.max_output_tokens" && entry.action == UnsupportedAction::Clamped
	}));
}

#[test]
fn maps_every_canonical_reasoning_effort_to_ollama_think() {
	for (effort, expected) in [
		(Effort::Off, Value::Bool(false)),
		(Effort::Minimal, Value::String("low".into())),
		(Effort::Low, Value::String("low".into())),
		(Effort::Medium, Value::String("medium".into())),
		(Effort::High, Value::String("high".into())),
		(Effort::XHigh, Value::String("xhigh".into())),
		(Effort::Max, Value::String("max".into())),
	] {
		let mut request = fixture_request();
		request.thinking = Some(
			Feature::builder()
				.value(Reasoning::builder().effort(effort).build())
				.on_unsupported(Fallback::Error)
				.build(),
		);
		let (body, _) = OllamaChatCodec
			.encode(&request, &Compat::default())
			.unwrap();
		let body: Value = serde_json::from_slice(&body).unwrap();
		assert_eq!(body["think"], expected);
	}
}

#[test]
fn model_policy_controls_native_effort_spelling_and_num_predict_omission() {
	let mut request = fixture_request();
	request.model_policy = Some(effort_policy(
		BTreeMap::from([
			(Effort::High, SmolStr::new_static("turbo")),
			(Effort::XHigh, SmolStr::new_static("extra-high")),
			(Effort::Max, SmolStr::new_static("maximum")),
		]),
		true,
	));
	let (body, _) = OllamaChatCodec
		.encode(&request, &Compat::default())
		.unwrap();
	let body: Value = serde_json::from_slice(&body).unwrap();
	assert_eq!(body["think"], "turbo");
	assert!(body.get("options").is_none());

	request.model_policy = Some(effort_policy(
		BTreeMap::from([
			(Effort::High, SmolStr::new_static("careful")),
			(Effort::XHigh, SmolStr::new_static("extra-high")),
			(Effort::Max, SmolStr::new_static("maximum")),
		]),
		false,
	));
	let (opposite, _) = OllamaChatCodec
		.encode(&request, &Compat::default())
		.unwrap();
	let opposite: Value = serde_json::from_slice(&opposite).unwrap();
	assert_eq!(opposite["think"], "careful");
	assert_eq!(opposite["options"]["num_predict"], 65_536);

	for (effort, spelling) in [(Effort::XHigh, "extra-high"), (Effort::Max, "maximum")] {
		request.thinking = Some(
			Feature::builder()
				.value(Reasoning::builder().effort(effort).build())
				.on_unsupported(Fallback::Error)
				.build(),
		);
		let (body, _) = OllamaChatCodec
			.encode(&request, &Compat::default())
			.unwrap();
		let body: Value = serde_json::from_slice(&body).unwrap();
		assert_eq!(body["think"], spelling);
	}
}

#[test]
fn conditional_model_sampling_compat_applies_only_while_thinking() {
	let mut policy = (*effort_policy(BTreeMap::new(), true)).clone();
	policy
		.compat
		.insert_ns("wire", "supports_sampling_params", Value::Bool(true));
	policy
		.compat
		.insert_ns("wire", "when_thinking", json!({"supports_sampling_params": false}));
	let mut request = fixture_request();
	request.model_policy = Some(Arc::new(policy));
	request.sampling = Some(Sampling::builder().temperature(0.4).build());

	let (thinking_body, thinking_unsupported) = OllamaChatCodec
		.encode(&request, &Compat::default())
		.unwrap();
	let thinking_body: Value = serde_json::from_slice(&thinking_body).unwrap();
	assert!(thinking_body.get("options").is_none());
	assert!(
		thinking_unsupported
			.iter()
			.any(|entry| entry.what == "sampling.temperature")
	);

	request.thinking = Some(
		Feature::builder()
			.value(Reasoning::builder().effort(Effort::Off).build())
			.on_unsupported(Fallback::Error)
			.build(),
	);
	let (off_body, off_unsupported) = OllamaChatCodec
		.encode(&request, &Compat::default())
		.unwrap();
	let off_body: Value = serde_json::from_slice(&off_body).unwrap();
	assert_eq!(off_body["options"]["temperature"], 0.4);
	assert!(
		!off_unsupported
			.iter()
			.any(|entry| entry.what == "sampling.temperature")
	);
}

fn decode_fixture(fixture: &str) -> Vec<TurnEvent> {
	let codec = OllamaChatCodec;
	let mut framing = NdjsonDecoder::new();
	let mut state = DecodeState::default();
	let mut events = Vec::new();
	for chunk in fixture.as_bytes().chunks(7) {
		let records = framing
			.push(Bytes::copy_from_slice(chunk))
			.collect::<Vec<_>>();
		for record in records {
			events.extend(codec.decode(Frame::Data(&record), &mut state).unwrap());
		}
	}
	assert_eq!(framing.buffered_len(), 0);
	events.extend(codec.decode(Frame::Done, &mut state).unwrap());
	events
}

#[test]
fn recorded_ndjson_stream_emits_incremental_parts_usage_and_one_terminal() {
	let events = decode_fixture(include_str!("fixtures/cloud_stream.ndjson"));
	assert!(
		events
			.iter()
			.any(|event| matches!(event, TurnEvent::PartStart { kind: StreamPartKind::Thinking, .. }))
	);
	assert!(
		events
			.iter()
			.any(|event| matches!(event, TurnEvent::PartStart { kind: StreamPartKind::Text, .. }))
	);
	assert!(events.iter().any(|event| matches!(
		event,
		TurnEvent::PartStart { kind: StreamPartKind::ToolCall, tool_name, .. } if tool_name == "weather"
	)));
	let terminals: Vec<_> = events
		.iter()
		.filter(|event| matches!(event, TurnEvent::Outcome(_) | TurnEvent::Error(_)))
		.collect();
	assert_eq!(terminals.len(), 1);
	let TurnEvent::Outcome(outcome) = terminals[0] else {
		panic!("expected outcome")
	};
	assert_eq!(outcome.stop, StopReason::ToolUse);
	let usage = outcome.usage.as_ref().unwrap();
	assert_eq!(usage.input_tokens, 21);
	assert_eq!(usage.output_tokens, 8);
	assert_eq!(usage.total_tokens, Some(29));
	assert_eq!(outcome.output.len(), 3);
}

#[test]
fn maps_length_and_load_terminals_without_laundering_failures() {
	let codec = OllamaChatCodec;
	let mut length_state = DecodeState::default();
	let length = codec
		.decode(
			Frame::Data(
				br#"{"message":{"role":"assistant","content":""},"done":true,"done_reason":"length","prompt_eval_count":2,"eval_count":4}"#,
			),
			&mut length_state,
		)
		.unwrap();
	assert!(matches!(
		length.as_slice(),
		[TurnEvent::Outcome(outcome)] if outcome.stop == StopReason::MaxTokens
	));

	let mut load_state = DecodeState::default();
	let load = codec
		.decode(
			Frame::Data(
				br#"{"message":{"role":"assistant","content":""},"done":true,"done_reason":"load"}"#,
			),
			&mut load_state,
		)
		.unwrap();
	assert!(matches!(load.as_slice(), [TurnEvent::Error(_)]));
}

#[test]
fn recorded_stream_without_done_is_a_malformed_terminal_error() {
	let events = decode_fixture(include_str!("fixtures/malformed_terminal.ndjson"));
	let terminals: Vec<_> = events
		.iter()
		.filter(|event| matches!(event, TurnEvent::Outcome(_) | TurnEvent::Error(_)))
		.collect();
	assert_eq!(terminals.len(), 1);
	let TurnEvent::Error(error) = terminals[0] else {
		panic!("expected error")
	};
	assert!(error.detail.contains("before a done terminal chunk"));
}
