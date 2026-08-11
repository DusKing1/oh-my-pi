//! Property tests for transcript replay capture and reconstruction.
use omp_core::Str;
use omp_storage::transcript::{
	block::{Block, BlockKind, CallId, Replay},
	capsule::{Ant, JoinMode, Oai, REV_1, split_markers},
	replay::{capture, emit, rebuild},
};
use pretty_assertions::assert_eq;
use proptest::prelude::*;
use serde_json::{Value, json};

fn string(value: &str) -> Str {
	Str::new(value)
}

const fn block(kind: BlockKind) -> Block {
	Block { kind, re: None }
}

fn text(value: &str) -> Block {
	block(BlockKind::Text { text: string(value) })
}

fn think(value: &str) -> Block {
	block(BlockKind::Think { text: string(value) })
}

fn tool(id: &str, name: &str, wire: Option<&str>, args: &str) -> Block {
	block(BlockKind::Tool {
		id:   CallId(string(id)),
		name: string(name),
		wire: wire.map(string),
		args: string(args),
	})
}

fn attach(blocks: &mut [Block], capsules: Vec<Option<Replay>>) {
	for (block, capsule) in blocks.iter_mut().zip(capsules) {
		block.re = capsule;
	}
}

#[test]
fn mixed_turn_rebuilds_every_residue_field_and_verbatim_arguments() {
	let odd_args = " { \"z\" : 1, \"a\":2, \"z\":3 } ";
	let mut blocks =
		vec![think("work it out"), text("The answer."), tool("call_7", "lookup", None, odd_args)];
	let native = vec![
		json!({
			"type": "reasoning",
			"summary": [{ "type": "summary_text", "text": "work it out" }],
			"id": "rs_1",
			"encrypted_content": "gAAAA-model-bound",
			"turn_id": "turn_9",
			"create_time": 1_723_456_789
		}),
		json!({
			"type": "message",
			"role": "assistant",
			"status": "completed",
			"phase": "final_answer",
			"content": [{
				"type": "output_text",
				"text": "The answer.",
				"annotations": [],
				"logprobs": []
			}],
			"id": "msg_1",
			"turn_id": "turn_9",
			"create_time": 1_723_456_790
		}),
		json!({
			"type": "function_call",
			"status": "completed",
			"call_id": "call_7",
			"name": "lookup",
			"arguments": odd_args,
			"id": "fc_1",
			"turn_id": "turn_9",
			"create_time": 1_723_456_791
		}),
	];

	let capsules = capture(&blocks, &native, &Oai, REV_1);
	attach(&mut blocks, capsules);
	let rebuilt = rebuild(&blocks, &Oai, REV_1);

	assert_eq!(rebuilt.len(), native.len());
	assert_eq!(rebuilt, native);
	assert_eq!(rebuilt[2]["arguments"].as_str(), Some(odd_args));
}

#[test]
fn payload_less_custom_tool_uses_the_wire_string_without_an_input_wrapper() {
	let args = "unstructured freeform patch\nwith a second line";
	let mut blocks = vec![tool("call_patch", "patch", Some("apply_patch"), args)];
	let native = vec![json!({
		"type": "custom_tool_call",
		"call_id": "call_patch",
		"name": "apply_patch",
		"input": args,
		"id": "ctc_1"
	})];

	let capsules = capture(&blocks, &native, &Oai, REV_1);
	assert!(
		capsules[0]
			.as_ref()
			.is_some_and(|capsule| capsule.f.contains_key("~omit"))
	);
	attach(&mut blocks, capsules);
	let rebuilt = rebuild(&blocks, &Oai, REV_1);

	assert_eq!(rebuilt, native);
	assert_eq!(rebuilt[0]["input"], Value::String(args.to_owned()));
	assert!(rebuilt[0]["input"].get("input").is_none());
}

#[test]
fn anthropic_thinking_capsule_contains_only_the_signature_residue() {
	let mut blocks = vec![think("private chain")];
	let native = vec![json!({
		"t": "think",
		"text": "private chain",
		"sig": "signed-model-bound-value"
	})];

	let capsules = capture(&blocks, &native, &Ant, REV_1);
	let capsule = capsules[0]
		.as_ref()
		.expect("thinking item has signature residue");
	assert_eq!(capsule.f.keys().map(Str::as_str).collect::<Vec<_>>(), vec!["sig"]);
	attach(&mut blocks, capsules);

	assert_eq!(rebuild(&blocks, &Ant, REV_1), native);
}

#[test]
fn multipart_markers_cover_split_blank_join_and_plain_join() {
	let mut blocks = vec![
		think("alpha\n\nbeta"),
		think("gamma"),
		think("left"),
		think("right"),
		text("plain"),
		text("join"),
	];
	let native = vec![
		json!({
			"type": "reasoning",
			"summary": [
				{ "type": "summary_text", "text": "alpha" },
				{ "type": "summary_text", "text": "beta" },
				{ "type": "summary_text", "text": "gamma" }
			],
			"id": "rs_split"
		}),
		json!({
			"type": "reasoning",
			"summary": [{ "type": "summary_text", "text": "left\n\nright" }],
			"id": "rs_jnn"
		}),
		json!({
			"type": "message",
			"role": "assistant",
			"status": "completed",
			"phase": "final_answer",
			"content": [{
				"type": "output_text",
				"text": "plainjoin",
				"annotations": [],
				"logprobs": []
			}],
			"id": "msg_j"
		}),
	];

	let capsules = capture(&blocks, &native, &Oai, REV_1);
	let markers = capsules
		.iter()
		.flatten()
		.map(|capsule| split_markers(&capsule.f).0)
		.collect::<Vec<_>>();
	assert_eq!(
		markers
			.iter()
			.map(|marker| (marker.np, marker.join))
			.collect::<Vec<_>>(),
		vec![
			(Some(2), Some(JoinMode::Split)),
			(Some(2), Some(JoinMode::Jnn)),
			(Some(2), Some(JoinMode::J)),
		]
	);
	attach(&mut blocks, capsules);

	assert_eq!(rebuild(&blocks, &Oai, REV_1), native);
}

#[test]
fn explicit_order_restores_opaque_item_in_its_wire_position() {
	let mut blocks = vec![text("after search"), block(BlockKind::Opaque)];
	let native = vec![
		json!({
			"type": "web_search_call",
			"id": "ws_1",
			"status": "completed",
			"query": "capsule format"
		}),
		json!({
			"type": "message",
			"role": "assistant",
			"status": "completed",
			"phase": "final_answer",
			"content": [{
				"type": "output_text",
				"text": "after search",
				"annotations": [],
				"logprobs": []
			}],
			"id": "msg_after"
		}),
	];

	let capsules = capture(&blocks, &native, &Oai, REV_1);
	assert!(
		capsules
			.iter()
			.flatten()
			.all(|capsule| capsule.f.contains_key("~ord"))
	);
	attach(&mut blocks, capsules);
	let rebuilt = rebuild(&blocks, &Oai, REV_1);

	assert_eq!(rebuilt, native);
	assert!(rebuilt.iter().all(|item| {
		item
			.as_object()
			.is_some_and(|object| object.keys().all(|key| !key.starts_with('~')))
	}));
}

proptest! {
	#[test]
	fn captured_random_blocks_rebuild_count_order_and_every_field(
		entries in prop::collection::vec(
			(0_u8..3, "[a-zA-Z0-9 \n]{0,32}", any::<u16>(), any::<bool>()),
			0..24,
		),
	) {
		let mut blocks = entries
			.iter()
			.enumerate()
			.map(|(index, (kind, payload, _, _))| match kind {
				0 => text(payload),
				1 => think(payload),
				_ => tool(
					&format!("call_{index}"),
					&format!("tool_{index}"),
					None,
					payload,
				),
			})
			.collect::<Vec<_>>();
		let mut native = emit(&blocks, &Oai, REV_1);
		for (index, (item, (_, _, residue, omit_status))) in
			native.iter_mut().zip(&entries).enumerate()
		{
			let object = item.as_object_mut().expect("dialect defaults are objects");
			object.insert("id".to_owned(), Value::String(format!("item_{index}")));
			object.insert("trace".to_owned(), Value::from(*residue));
			if *omit_status {
				object.remove("status");
			}
		}

		let capsules = capture(&blocks, &native, &Oai, REV_1);
		attach(&mut blocks, capsules);
		let rebuilt = rebuild(&blocks, &Oai, REV_1);

		prop_assert_eq!(rebuilt.len(), native.len());
		prop_assert_eq!(rebuilt, native);
	}
}
