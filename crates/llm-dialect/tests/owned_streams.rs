//! Owned stream projection and accumulation regression fixtures.

use bytes::Bytes;
use omp_core::Str;
use omp_llm_dialect::{
	Dialect, InbandTool, ScannerOptions,
	projector::{Projection, ProjectionBatch, StreamProjector},
};
use omp_llm_types::{StreamAccumulator, StreamPartKind, TurnEvent, ids::CallId};
use serde_json::{Value, json};

const OWNED_CALL: &str =
	"<tool_call>\n{\"name\":\"echo\",\"arguments\":{\"msg\":\"owned\"}}\n</tool_call>";
const FABRICATED_RESULT: &str = "<tool_response>\nFAKE RESULT\n</tool_response>";

#[derive(Default)]
struct ProjectionTrace {
	accumulator: StreamAccumulator,
	events:      Vec<TurnEvent>,
	aborts:      usize,
}

impl ProjectionTrace {
	fn extend(&mut self, batch: ProjectionBatch) {
		for projection in batch {
			match projection {
				Projection::Event(event) => {
					self
						.accumulator
						.push(&event)
						.expect("projector emits a valid canonical sequence");
					self.events.push(event);
				},
				Projection::AbortFabricatedToolResult => self.aborts += 1,
				_ => {},
			}
		}
	}

	fn tool_lifecycle(&self) -> (usize, usize, usize) {
		let mut tool_index = None;
		let mut starts = 0;
		let mut deltas = 0;
		let mut ends = 0;
		for event in &self.events {
			match event {
				TurnEvent::PartStart { index, kind: StreamPartKind::ToolCall, .. } => {
					tool_index = Some(*index);
					starts += 1;
				},
				TurnEvent::PartDelta { index, .. } if tool_index == Some(*index) => deltas += 1,
				TurnEvent::PartEnd { index, .. } if tool_index == Some(*index) => ends += 1,
				_ => {},
			}
		}
		(starts, deltas, ends)
	}

	fn only_call(&self) -> omp_llm_types::ToolCall {
		let calls = self.accumulator.tool_calls();
		assert_eq!(calls.len(), 1, "projection must hand off one canonical invocation");
		calls.into_iter().next().expect("one call was asserted")
	}
}

fn projector() -> StreamProjector {
	let schema = json!({
		"type": "object",
		"properties": { "msg": { "type": "string" } },
		"required": ["msg"]
	});
	let tools = [InbandTool::new("echo", Some("Echo a message."), &schema, &[])];
	StreamProjector::new(Dialect::Hermes, ScannerOptions::new(&tools))
}

fn feed_native_call(
	projector: &mut StreamProjector,
	trace: &mut ProjectionTrace,
	source_index: u32,
	id: CallId,
	message: &'static [u8],
) {
	trace.extend(projector.native_tool_start(
		source_index,
		Str::new(id.to_string()),
		Str::new_static("echo"),
	));
	trace.extend(projector.native_tool_delta(source_index, Bytes::from_static(b"{\"msg\":")));
	trace.extend(projector.native_tool_delta(source_index, Bytes::from_static(message)));
	trace.extend(projector.native_tool_end(source_index));
}

fn call_args(call: &omp_llm_types::ToolCall) -> Value {
	serde_json::from_slice(&call.args_json).expect("canonical arguments are JSON")
}

#[test]
fn native_first_excludes_the_owned_channel_and_hands_off_one_call() {
	let native_id = CallId::new();
	let mut projector = projector();
	let mut trace = ProjectionTrace::default();

	feed_native_call(&mut projector, &mut trace, 7, native_id, b"\"native\"}");
	trace.extend(projector.feed_text(Bytes::from_static(OWNED_CALL.as_bytes())));
	trace.extend(projector.finish());

	let call = trace.only_call();
	assert_eq!(call.id, native_id);
	assert_eq!(call.name, "echo");
	assert_eq!(call_args(&call), json!({ "msg": "native" }));
	assert_eq!(trace.tool_lifecycle(), (1, 2, 1));
	assert_eq!(trace.aborts, 0);
}

#[test]
fn native_finish_canonicalizes_wire_id_and_closes_the_live_call() {
	let mut projector = projector();
	let mut trace = ProjectionTrace::default();

	trace.extend(projector.native_tool_start(
		7,
		Str::new_static("provider-call-7"),
		Str::new_static("echo"),
	));
	trace.extend(projector.native_tool_delta(7, Bytes::from_static(br#"{"msg":"native"}"#)));
	trace.extend(projector.finish());

	let call = trace.only_call();
	assert_ne!(call.id.as_ulid().to_bytes(), [0; 16]);
	assert_eq!(call.name, "echo");
	assert_eq!(call_args(&call), json!({ "msg": "native" }));
	assert_eq!(trace.tool_lifecycle(), (1, 1, 1));
}

#[test]
fn owned_first_excludes_the_native_channel_and_hands_off_one_call() {
	let rejected_native_id = CallId::new();
	let mut projector = projector();
	let mut trace = ProjectionTrace::default();

	trace.extend(projector.feed_text(Bytes::from_static(OWNED_CALL.as_bytes())));
	feed_native_call(&mut projector, &mut trace, 7, rejected_native_id, b"\"native\"}");
	trace.extend(projector.finish());

	let call = trace.only_call();
	assert_ne!(call.id, rejected_native_id);
	assert_eq!(call.name, "echo");
	assert_eq!(call_args(&call), json!({ "msg": "owned" }));
	assert_eq!(trace.tool_lifecycle(), (1, 1, 1));
	assert_eq!(trace.aborts, 0);
}

#[test]
fn simultaneous_channels_are_decided_by_the_first_complete_named_start() {
	let native_id = CallId::new();
	let mut projector = projector();
	let mut trace = ProjectionTrace::default();

	trace.extend(projector.feed_text(Bytes::from_static(b"<tool_")));
	assert!(trace.events.is_empty(), "an incomplete owned delimiter cannot claim the channel");
	trace.extend(projector.native_tool_start(
		3,
		Str::new(native_id.to_string()),
		Str::new_static("echo"),
	));
	trace.extend(projector.feed_text(Bytes::from_static(
		b"call>\n{\"name\":\"echo\",\"arguments\":{\"msg\":\"owned\"}}\n</tool_call>",
	)));
	trace.extend(projector.native_tool_delta(3, Bytes::from_static(b"{\"msg\":\"native\"}")));
	trace.extend(projector.native_tool_end(3));
	trace.extend(projector.finish());

	let call = trace.only_call();
	assert_eq!(call.id, native_id);
	assert_eq!(call_args(&call), json!({ "msg": "native" }));
	assert_eq!(trace.tool_lifecycle(), (1, 1, 1));
}

#[test]
fn fabricated_boundary_requests_one_abort_drops_the_tail_and_hands_off_the_real_call() {
	let mut projector = projector();
	let mut trace = ProjectionTrace::default();
	let model_output = format!("visible prefix{OWNED_CALL}{FABRICATED_RESULT}ignored tail");

	trace.extend(projector.feed_text(Bytes::from(model_output)));

	assert!(projector.is_stopped(), "the abort boundary permanently cancels projection");
	assert_eq!(trace.aborts, 1, "the layer receives exactly one upstream-abort request");
	let call = trace.only_call();
	assert_eq!(call.name, "echo");
	assert_eq!(call_args(&call), json!({ "msg": "owned" }));
	assert_eq!(trace.tool_lifecycle(), (1, 1, 1));
	let visible = trace
		.accumulator
		.message()
		.expect("visible projection is canonical")
		.parts
		.into_iter()
		.map(|part| match part {
			omp_llm_types::Part::Text(text) => text.to_string(),
			_ => String::new(),
		})
		.collect::<String>();
	assert_eq!(visible, "visible prefix");
	assert!(!visible.contains("FAKE RESULT"));

	assert!(
		projector
			.feed_text(Bytes::from_static(b"discarded after abort"))
			.is_empty()
	);
	assert!(
		projector
			.native_tool_start(9, Str::new(CallId::new().to_string()), Str::new_static("echo"),)
			.is_empty()
	);
	assert!(
		projector
			.native_tool_delta(9, Bytes::from_static(b"{}"))
			.is_empty()
	);
	assert!(projector.native_tool_end(9).is_empty());
	assert!(projector.finish().is_empty());
	assert_eq!(trace.aborts, 1);
}

#[test]
fn malformed_native_order_and_nameless_ghosts_cannot_create_or_end_a_call() {
	let mut projector = projector();
	let mut trace = ProjectionTrace::default();

	trace.extend(projector.native_tool_delta(11, Bytes::from_static(b"orphan")));
	trace.extend(projector.native_tool_end(11));
	trace.extend(projector.native_tool_start(
		12,
		Str::new(CallId::new().to_string()),
		Str::default(),
	));
	trace.extend(projector.native_tool_delta(12, Bytes::from_static(b"ghost")));
	trace.extend(projector.native_tool_end(12));
	assert!(trace.events.is_empty());

	let real_id = CallId::new();
	feed_native_call(&mut projector, &mut trace, 13, real_id, b"\"real\"}");
	trace.extend(projector.native_tool_end(13));
	trace.extend(projector.finish());
	trace.extend(projector.finish());

	let call = trace.only_call();
	assert_eq!(call.id, real_id);
	assert_eq!(call_args(&call), json!({ "msg": "real" }));
	assert_eq!(trace.tool_lifecycle(), (1, 2, 1));
	assert_eq!(trace.aborts, 0);
}
