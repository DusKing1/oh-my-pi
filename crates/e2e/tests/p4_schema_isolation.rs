use std::fmt::Write as _;

use bytes::Bytes;
use futures::{Stream, StreamExt as _};
use omp_agent::{
	Journal, TurnClient, TurnId, TurnInput, TurnOptions, TurnSession, project_journal,
};
use omp_e2e::support::{ScriptedTurn, ScriptedTurnClient};
use omp_core::Str;
use omp_llm_catalog::GrammarBits;
use omp_llm_inference::call::{ChatRequest, ContentPart, ToolResultContent};
use omp_proto::{inference::v1 as pb, thread::v1 as thread_pb};
use omp_storage::transcript::{Header, SessionId};
use omp_tool::{
	Constraint, Ev, IncomingParams, LiftedCall, LoweringCaps, Part, PromptCaps, RecordedCall,
	RecordedCallOwned, Registry, Rev, Tool, ToolIdentity, ToolSpec,
};
use prost::Message as _;
use serde_json::{Value, json};
use tempfile::TempDir;

const HL1_SCHEMA_FRAGMENT: &[u8] = b"historical_hl1_schema_poison";
const HL2_SCHEMA: &[u8] = br#"{"type":"object","properties":{"patch":{"type":"string"},"mode":{"const":"hl.2"},"hl2_schema_only":{"type":"boolean"}},"required":["patch","mode"],"additionalProperties":false}"#;
const PROMPT_CAPS: PromptCaps = PromptCaps {
	maximum_parts: 1,
	maximum_text_bytes: 65_536,
	media: false,
};

struct LiveEdit {
	spec:       ToolSpec,
	allow_lift: bool,
}

impl LiveEdit {
	fn new(allow_lift: bool) -> Self {
		Self {
			spec: ToolSpec {
				name:        Str::new_static("edit"),
				rev:         Rev { family: Str::new_static("hl"), n: 2 },
				description: Str::new_static("apply a hashline edit"),
				schema:      Bytes::from_static(HL2_SCHEMA),
				constraint:  Constraint::Schema { priority: 1 },
			},
			allow_lift,
		}
	}
}

impl Tool for LiveEdit {
	type Fault = Value;
	type Params = Value;
	type Payload = Value;
	type Update = Value;

	fn spec(&self) -> &ToolSpec {
		&self.spec
	}

	fn call<'c>(
		&'c self,
		_params: IncomingParams<'c>,
	) -> impl Stream<Item = Ev<Self::Update, Self::Payload, Self::Fault>> + Send + 'c {
		futures::stream::empty()
	}

	fn prompt(&self, view: Result<&Self::Payload, &Self::Fault>, _caps: &PromptCaps) -> Vec<Part> {
		let (branch, value) = match view {
			Ok(value) => ("ok", value),
			Err(value) => ("fault", value),
		};
		let recorded = value
			.get("recorded_verdict")
			.and_then(Value::as_str)
			.unwrap_or("missing-recorded-verdict");
		let lifted = value.get("lifted_to").and_then(Value::as_str).unwrap_or("not-lifted");
		vec![Part::Text { text: Str::from(format!("{branch}|{recorded}|{lifted}")) }]
	}

	fn lift(&self, from: &Rev, call: RecordedCall<'_>) -> Option<LiftedCall> {
		if !self.allow_lift || from.family.as_str() != "hl" || from.n != 1 {
			return None;
		}

		let args: Value = serde_json::from_slice(call.raw_args).ok()?;
		let legacy_patch = args.get("legacy_patch")?.as_str()?;
		let source_nonce = args.get("recorded_arg_nonce")?.as_str()?;
		let mut verdict: Value = serde_json::from_slice(call.verdict).ok()?;
		let value = verdict.get_mut("value")?.as_object_mut()?;
		if !value
			.get("recorded_verdict")
			.and_then(Value::as_str)
			.is_some_and(|marker| marker.starts_with("verdict-"))
		{
			return None;
		}
		value.insert("lifted_to".to_owned(), Value::String("hl.2".to_owned()));

		Some(LiftedCall {
			raw_args: Bytes::from(
				serde_json::to_vec(&json!({
					"patch": legacy_patch,
					"mode": "hl.2",
					"source_nonce": source_nonce,
				}))
				.ok()?,
			),
			verdict: Bytes::from(serde_json::to_vec(&verdict).ok()?),
		})
	}
}

fn registry(allow_lift: bool) -> Registry {
	let mut registry = Registry::new();
	registry.register(LiveEdit::new(allow_lift)).expect("live edit@hl.2 registers");
	registry
}

fn json_proto_value(value: Value) -> pb::Value {
	let kind = match value {
		Value::Null => pb::value::Kind::Null(true),
		Value::Bool(value) => pb::value::Kind::Bool(value),
		Value::Number(value) => {
			if let Some(value) = value.as_i64() {
				pb::value::Kind::Int(value)
			} else if let Some(value) = value.as_u64() {
				pb::value::Kind::Uint(value)
			} else {
				pb::value::Kind::Double(value.as_f64().expect("fixture numbers are finite"))
			}
		},
		Value::String(value) => pb::value::Kind::String(value),
		Value::Array(values) => pb::value::Kind::List(pb::ValueList {
			values: values.into_iter().map(json_proto_value).collect(),
		}),
		Value::Object(fields) => pb::value::Kind::Map(pb::ValueMap {
			fields: fields
				.into_iter()
				.map(|(key, value)| (key, json_proto_value(value)))
				.collect(),
		}),
	};
	pb::Value { kind: Some(kind) }
}

fn props(fields: impl IntoIterator<Item = (&'static str, pb::Value)>) -> pb::ValueMap {
	pb::ValueMap {
		fields: fields.into_iter().map(|(key, value)| (key.to_owned(), value)).collect(),
	}
}

fn string_value(value: &str) -> pb::Value {
	pb::Value { kind: Some(pb::value::Kind::String(value.to_owned())) }
}

fn recorded_verdict(branch: &str, marker: &str, line: i64) -> Value {
	json!({
		"kind": branch,
		"value": {
			"recorded_verdict": marker,
			"detail": { "line": line, "authority": "recorded" }
		}
	})
}

fn historical_items() -> Vec<thread_pb::Item> {
	[("edit-one", "alpha", "args-one", "fault", "verdict-one", 7_i64, true, true),
	 ("edit-two", "beta", "args-two", "ok", "verdict-two", 11_i64, false, false)]
		.into_iter()
		.flat_map(|(id, patch, nonce, branch, marker, line, is_error, useless)| {
			let verdict = recorded_verdict(branch, marker, line);
			[
				thread_pb::Item {
					seq: 0,
					created_at_ms: u64::try_from(line).expect("positive fixture line"),
					kind: Some(thread_pb::item::Kind::ToolCall(thread_pb::ToolCall {
						id: id.to_owned(),
						name: "edit".to_owned(),
						args_json: Bytes::from(
							serde_json::to_vec(&json!({
								"legacy_patch": patch,
								"recorded_arg_nonce": nonce,
							}))
							.expect("fixture arguments serialize"),
						),
						..Default::default()
					})),
					props: Some(props([
						("omp/tool-rev", string_value("hl.1")),
						("fixture/call-meta", string_value(nonce)),
					])),
				},
				thread_pb::Item {
					seq: 0,
					created_at_ms: u64::try_from(line + 1).expect("positive fixture line"),
					kind: Some(thread_pb::item::Kind::ToolResult(thread_pb::ToolResult {
						call_id: id.to_owned(),
						name: "edit".to_owned(),
						parts: vec![thread_pb::Part {
							kind: Some(thread_pb::part::Kind::Text(format!(
								"recorded-visible|{marker}"
							))),
						}],
						details: Some(json_proto_value(verdict)),
						is_error,
						useless: Some(useless),
						..Default::default()
					})),
					props: Some(props([("fixture/result-meta", string_value(marker))])),
				},
			]
		})
		.collect()
}

fn persisted_fixture() -> (TempDir, Journal, thread_pb::Thread) {
	let directory = tempfile::tempdir().expect("fixture directory");
	let path = directory.path().join("schema-isolation.jsonl");
	let header = Header {
		v: 4,
		id: SessionId(Str::new_static("p4-schema-isolation")),
		created: 1,
		cwd: directory.path().to_owned(),
	};
	let items = historical_items();
	let mut journal = Journal::create(&path, &header).expect("create real transcript fixture");
	for (offset, item) in items.iter().cloned().enumerate() {
		journal
			.append_optimistic(
				u64::try_from(offset + 2).expect("fixture offset fits u64"),
				item,
				None,
			)
			.expect("append recorded call or verdict");
	}
	drop(journal);
	let journal = Journal::open(&path).expect("reopen persisted transcript fixture");
	(directory, journal, thread_pb::Thread { items })
}

fn schema_bytes(request: &ChatRequest) -> Vec<Vec<u8>> {
	request
		.tools
		.iter()
		.map(|definition| {
			let (schema, _) = definition.input.json_schema().expect("edit uses JSON Schema");
			serde_json::to_vec(schema.as_value()).expect("provider schema serializes")
		})
		.collect()
}

fn render_owned_xml(request: &ChatRequest) -> Vec<u8> {
	let mut xml = String::from("<turn><tools>");
	for definition in request.tools.iter() {
		let (schema, strict) = definition.input.json_schema().expect("edit uses JSON Schema");
		let schema = serde_json::to_string(schema.as_value()).expect("schema serializes");
		write!(
			xml,
			"<tool name=\"{}\" strict=\"{strict}\"><schema>{schema}</schema></tool>",
			definition.name
		)
		.expect("write owned XML tool");
	}
	xml.push_str("</tools><thread>");
	for message in request.messages.iter() {
		for part in message.content.iter() {
			match part {
				ContentPart::ToolCall { call, name, arguments, .. } => {
					let arguments = serde_json::to_string(arguments.as_value())
						.expect("arguments serialize");
					write!(
						xml,
						"<invoke id=\"{}\" name=\"{name}\"><arguments>{arguments}</arguments></invoke>",
						call.as_str()
					)
					.expect("write owned XML invocation");
				},
				ContentPart::ToolResult { call, content, is_error, .. } => {
					write!(xml, "<result call=\"{}\" error=\"{is_error}\">", call.as_str())
						.expect("write owned XML result");
					for content in content.iter() {
						match content {
							ToolResultContent::Text(text) => xml.push_str(text.as_str()),
							ToolResultContent::Json(json) => xml.push_str(
								&serde_json::to_string(json.as_value()).expect("result JSON serializes"),
							),
							ToolResultContent::Image(_) | ToolResultContent::Document(_) => {
								xml.push_str("<media/>");
							},
						}
					}
					xml.push_str("</result>");
				},
				_ => {},
			}
		}
	}
	xml.push_str("</thread></turn>");
	xml.into_bytes()
}


#[tokio::test]
async fn historical_edit_schema_is_isolated_and_lifts_from_recorded_truth() {
	let (_directory, journal, original) = persisted_fixture();
	let log = journal.load().expect("load persisted transcript fixture");

	let without_lift = registry(false);
	let advertised = without_lift.advertise(LoweringCaps {
		strict_schema: true,
		grammar: GrammarBits::empty(),
	});
	let [advertised_edit] = advertised.as_slice() else {
		panic!("registry must advertise exactly one live edit definition")
	};
	assert_eq!(advertised_edit.identity.name.as_str(), "edit");
	assert_eq!(advertised_edit.identity.rev, Rev { family: Str::new_static("hl"), n: 2 });
	let (advertised_schema, strict) = advertised_edit
		.definition
		.input
		.json_schema()
		.expect("live edit advertises JSON Schema");
	let advertised_schema = serde_json::to_vec(advertised_schema.as_value())
		.expect("advertised schema serializes");
	assert!(strict, "the route supports exact strict schema advertisement");
	assert_eq!(advertised_schema, HL2_SCHEMA);
	assert!(!advertised_schema.windows(HL1_SCHEMA_FRAGMENT.len()).any(|window| {
		window == HL1_SCHEMA_FRAGMENT
	}));

	let recorded = RecordedCallOwned {
		identity: ToolIdentity {
			name: Str::new_static("edit"),
			rev: Rev { family: Str::new_static("hl"), n: 1 },
		},
		raw_args: Bytes::from_static(br#"{"legacy_patch":"alpha","recorded_arg_nonce":"args-one"}"#),
		verdict: Bytes::from(
			serde_json::to_vec(&recorded_verdict("fault", "verdict-one", 7))
				.expect("recorded verdict serializes"),
		),
	};
	assert!(matches!(without_lift.project(recorded.clone()), omp_tool::ProjectedCall::Data(data) if data == recorded));

	let data = project_journal(&log, &without_lift, &PROMPT_CAPS)
		.expect("unliftable historical revision projects as canonical data");
	assert_eq!(
		data.encode_to_vec(),
		original.encode_to_vec(),
		"unliftable calls, verdict details, error/useless bits, props, and field presence stay verbatim"
	);

	let params = pb::ChatParams {
		tools: vec![pb::ToolDef {
			name: advertised_edit.definition.name.to_string(),
			description: advertised_edit
				.definition
				.description
				.as_ref()
				.map_or_else(String::new, ToString::to_string),
			schema_json: Bytes::copy_from_slice(&advertised_schema),
			strict: Some(strict),
		}],
		..Default::default()
	};
	let (provider_data, provider_request) = omp_app::rpc_adapter::project_provider_turn_for_test(
		&data,
		&params,
		&without_lift,
	)
	.expect("owned provider dialect accepts unliftable canonical history");
	assert_eq!(provider_data.encode_to_vec(), original.encode_to_vec());
	assert_eq!(schema_bytes(&provider_request), vec![HL2_SCHEMA.to_vec()]);
	let owned_xml = render_owned_xml(&provider_request);
	assert!(owned_xml.windows(HL2_SCHEMA.len()).any(|window| window == HL2_SCHEMA));
	assert!(
		!owned_xml
			.windows(HL1_SCHEMA_FRAGMENT.len())
			.any(|window| window == HL1_SCHEMA_FRAGMENT),
		"provider rendering must not recover or inject a historical schema"
	);

	let with_lift = registry(true);
	let lifted_advertised = with_lift.advertise(LoweringCaps {
		strict_schema: true,
		grammar: GrammarBits::empty(),
	});
	let [lifted_advertised] = lifted_advertised.as_slice() else {
		panic!("adding a lift must not synthesize a historical registration")
	};
	let (lifted_schema, _) = lifted_advertised
		.definition
		.input
		.json_schema()
		.expect("lifted live edit advertises JSON Schema");
	assert_eq!(
		serde_json::to_vec(lifted_schema.as_value()).expect("lifted schema serializes"),
		HL2_SCHEMA
	);
	let first = project_journal(&log, &with_lift, &PROMPT_CAPS)
		.expect("recorded hl.1 truth lifts to live hl.2");
	let second = project_journal(&log, &with_lift, &PROMPT_CAPS)
		.expect("unchanged journal reprojects");
	assert_eq!(
		first.encode_to_vec(),
		second.encode_to_vec(),
		"projection of an unchanged transcript must be byte-identical"
	);
	assert_eq!(
		first.items.len(),
		original.items.len(),
		"projection must retain every canonical call/result item"
	);

	for (ordinal, pair) in first.items.chunks_exact(2).enumerate() {
		let call = match pair[0].kind.as_ref() {
			Some(thread_pb::item::Kind::ToolCall(call)) => call,
			other => panic!("expected lifted tool call, got {other:?}"),
		};
		let result = match pair[1].kind.as_ref() {
			Some(thread_pb::item::Kind::ToolResult(result)) => result,
			other => panic!("expected lifted tool result, got {other:?}"),
		};
		let expected_marker = if ordinal == 0 { "verdict-one" } else { "verdict-two" };
		let expected_nonce = if ordinal == 0 { "args-one" } else { "args-two" };
		let expected_patch = if ordinal == 0 { "alpha" } else { "beta" };
		let expected_branch = if ordinal == 0 { "fault" } else { "ok" };
		let expected_error = ordinal == 0;
		let expected_useless = ordinal == 0;

		assert_eq!(
			serde_json::from_slice::<Value>(&call.args_json).expect("lifted args are JSON"),
			json!({ "patch": expected_patch, "mode": "hl.2", "source_nonce": expected_nonce })
		);
		assert!(matches!(
			pair[0]
				.props
				.as_ref()
				.and_then(|props| props.fields.get("omp/tool-rev"))
				.and_then(|value| value.kind.as_ref()),
			Some(pb::value::Kind::String(rev)) if rev == "hl.2"
		));
		assert!(matches!(
			pair[0]
				.props
				.as_ref()
				.and_then(|props| props.fields.get("fixture/call-meta"))
				.and_then(|value| value.kind.as_ref()),
			Some(pb::value::Kind::String(nonce)) if nonce == expected_nonce
		));
		assert_eq!(result.is_error, expected_error, "error branch must be recomputed");
		assert_eq!(result.useless, Some(expected_useless), "recorded useless metadata survives");
		assert!(matches!(
			pair[1]
				.props
				.as_ref()
				.and_then(|props| props.fields.get("omp/tool-rev"))
				.and_then(|value| value.kind.as_ref()),
			Some(pb::value::Kind::String(rev)) if rev == "hl.2"
		));
		assert_eq!(
			result.details,
			Some(json_proto_value(json!({
				"kind": expected_branch,
				"value": {
					"recorded_verdict": expected_marker,
					"detail": {
						"line": if ordinal == 0 { 7 } else { 11 },
						"authority": "recorded"
					},
					"lifted_to": "hl.2"
				}
			}))),
			"lift must transform the recorded verdict bytes without dropping detail"
		);
		assert!(matches!(
			result.parts.as_slice(),
			[thread_pb::Part { kind: Some(thread_pb::part::Kind::Text(text)) }]
				if text == &format!("{expected_branch}|{expected_marker}|hl.2")
		));
		assert!(matches!(
			pair[1]
				.props
				.as_ref()
				.and_then(|props| props.fields.get("fixture/result-meta"))
				.and_then(|value| value.kind.as_ref()),
			Some(pb::value::Kind::String(marker)) if marker == expected_marker
		));
	}

	let expected_full = first.encode_to_vec();
	let client = ScriptedTurnClient::new([ScriptedTurn::events([pb::TurnEvent {
		event: Some(pb::turn_event::Event::Accepted(pb::Accepted { replay: false })),
	}])]);
	let options = TurnOptions::default();
	let mut session = client
		.turn(TurnId::new("p4-schema-isolation"), TurnInput::Full(first), &options)
		.await
		.expect("TurnClient accepts the full projected thread");
	assert!(matches!(
		session.events().next().await,
		Some(Ok(pb::TurnEvent {
			event: Some(pb::turn_event::Event::Accepted(pb::Accepted { replay: false }))
		}))
	));
	let captures = client.captures();
	let [captured] = captures.as_slice() else {
		panic!("TurnClient must capture exactly one accepted request")
	};
	assert_eq!(captured.turn_id.as_str(), "p4-schema-isolation");
	match &captured.input {
		TurnInput::Full(thread) => assert_eq!(
			thread.encode_to_vec(),
			expected_full,
			"the accepted full input is the byte-identical projected thread"
		),
		TurnInput::Delta(..) => panic!("projected replay must be submitted as TurnInput::Full"),
	}
}
