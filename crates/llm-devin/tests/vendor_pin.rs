//! Pin tests for Devin's compiled protobuf contract.

use bytes::Bytes;
use omp_llm_devin::{
	State as DevinDecodeState, decode_response as decode_devin, finish as finish_devin,
	wire as devin_wire,
};
use omp_llm_types::{ItemKind, StopReason, TurnEvent};
use prost::Message as _;

#[test]
fn compiled_devin_schema_retains_codec_contract() {
	use devin_wire::exa::{
		api_server_pb::GetChatMessageResponse,
		codeium_common_pb::{ChatToolCall, ModelUsageStats, StopReason},
	};

	let response = GetChatMessageResponse {
		delta_text: "answer".to_owned(),
		delta_thinking: "thought".to_owned(),
		delta_signature: "signature".to_owned(),
		delta_tool_calls: vec![ChatToolCall {
			id:             "wire-call".to_owned(),
			name:           "task".to_owned(),
			arguments_json: "{}".to_owned(),
		}],
		stop_reason: StopReason::FunctionCall as i32,
		usage: Some(ModelUsageStats {
			input_tokens:       11,
			output_tokens:      7,
			cache_read_tokens:  5,
			cache_write_tokens: 3,
		}),
		..Default::default()
	};
	let decoded = GetChatMessageResponse::decode(response.encode_to_vec().as_slice())
		.expect("Devin's compiled GetChatMessageResponse pin must remain decodable");
	assert_eq!(decoded.delta_tool_calls[0].arguments_json, "{}");
	assert_eq!(
		decoded
			.usage
			.expect("usage field must remain present")
			.cache_read_tokens,
		5
	);
}

fn decode_devin_arguments(chunks: &[&str], stop: i32) -> (Bytes, StopReason) {
	use devin_wire::exa::{api_server_pb::GetChatMessageResponse, codeium_common_pb::ChatToolCall};
	let mut state = DevinDecodeState::default();
	for chunk in chunks {
		decode_devin(
			GetChatMessageResponse {
				delta_tool_calls: vec![ChatToolCall {
					id:             "call-1".to_owned(),
					name:           "task".to_owned(),
					arguments_json: (*chunk).to_owned(),
				}],
				stop_reason: stop,
				..Default::default()
			},
			&mut state,
		);
	}
	finish_devin(&mut state)
		.into_iter()
		.find_map(|event| match event {
			TurnEvent::Outcome(outcome) => {
				outcome.output.into_iter().find_map(|item| match item.kind {
					ItemKind::ToolCall(call) => Some((call.args_json, outcome.stop)),
					_ => None,
				})
			},
			_ => None,
		})
		.expect("Devin tool-call outcome")
}

#[test]
fn recorded_devin_stream_decodes_incremental_and_cumulative_arguments_identically() {
	use devin_wire::exa::{
		api_server_pb::GetChatMessageResponse,
		codeium_common_pb::{ChatToolCall, ModelUsageStats, StopReason as WireStopReason},
	};

	let fixture = include_str!("fixtures/devin/stream.tool_args_usage.jsonl");
	let rows: Vec<serde_json::Value> = fixture
		.lines()
		.map(|line| serde_json::from_str(line).expect("valid recorded Devin JSONL"))
		.collect();
	let chunks: Vec<&str> = rows
		.iter()
		.map(|row| row["deltaToolCalls"][0]["argumentsJson"].as_str().unwrap())
		.collect();
	let incremental = decode_devin_arguments(&chunks, WireStopReason::FunctionCall as i32);
	let cumulative = decode_devin_arguments(
		&[
			r#"{"agent":"#,
			r#"{"agent":"task","note":"initial","#,
			r#"{"agent":"task","note":"initial","step":12}"#,
		],
		WireStopReason::MaxTokens as i32,
	);
	assert_eq!(incremental.0, cumulative.0);
	assert_eq!(incremental.0, Bytes::from_static(br#"{"agent":"task","note":"initial","step":12}"#));
	assert_eq!(incremental.1, StopReason::ToolUse);
	assert_eq!(cumulative.1, StopReason::ToolUse, "tool use must beat MAX_TOKENS");

	let mut state = DevinDecodeState::default();
	let mut deltas = Vec::new();
	for (index, row) in rows.iter().enumerate() {
		let usage = row.get("usage").map(|usage| ModelUsageStats {
			input_tokens: usage["inputTokens"].as_str().unwrap().parse().unwrap(),
			output_tokens: usage["outputTokens"].as_str().unwrap().parse().unwrap(),
			cache_read_tokens: usage["cachedInputTokens"]
				.as_str()
				.unwrap()
				.parse()
				.unwrap(),
			..Default::default()
		});
		deltas.extend(decode_devin(
			GetChatMessageResponse {
				delta_tool_calls: vec![ChatToolCall {
					id:             row["deltaToolCalls"][0]["id"].as_str().unwrap().to_owned(),
					name:           row["deltaToolCalls"][0]["name"]
						.as_str()
						.unwrap()
						.to_owned(),
					arguments_json: chunks[index].to_owned(),
				}],
				stop_reason: if index + 1 == rows.len() {
					WireStopReason::FunctionCall as i32
				} else {
					WireStopReason::Unspecified as i32
				},
				usage,
				..Default::default()
			},
			&mut state,
		));
	}
	assert_eq!(
		deltas
			.iter()
			.filter_map(|event| match event {
				TurnEvent::PartDelta { chunk, .. } => Some(chunk.clone()),
				_ => None,
			})
			.collect::<Vec<_>>(),
		chunks
			.iter()
			.map(|chunk| Bytes::copy_from_slice(chunk.as_bytes()))
			.collect::<Vec<_>>()
	);
	let outcome = finish_devin(&mut state)
		.into_iter()
		.find_map(|event| match event {
			TurnEvent::Outcome(outcome) => Some(outcome),
			_ => None,
		})
		.unwrap();
	assert_eq!(outcome.stop, StopReason::ToolUse);
	let usage = outcome.usage.unwrap();
	assert_eq!((usage.input_tokens, usage.output_tokens, usage.cache_read_tokens), (11, 7, 5));
}
