//! Round-trip tests for transcript persistence.

use std::{collections::BTreeMap, fs::OpenOptions, io::Write as _, path::PathBuf};

use omp_core::Str;
use omp_proto::{inference::v1 as pb, thread::v1 as thread_pb};
use omp_storage::{
	blob::BlobRef,
	transcript::{
		AmendPatch, Attribution, Block, BlockKind, CallId, CtxSnapshot, DialectId, Entry, Error,
		Event, FeatureId, Header, ItemRecord, Kind, ModelChange, ModelId, ModelRef, Msg, Patch, Pin,
		ProviderId, Replay, RequestError, SessionId, Stop, ThinkingSel, Timing, TitleSource,
		ToolBatchAuthorized, TurnInputItem, TurnInputRecord, TurnOptionsRecord, TurnReceipt, TurnStart,
		Usage, UserBlock, Writer, load, read_line,
		write_header, write_line,
	},
};
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use tempfile::tempdir;

fn text(value: &str) -> Str {
	Str::new(value)
}

fn raw(value: &str) -> Box<RawValue> {
	RawValue::from_string(value.to_owned()).expect("valid raw JSON")
}

const fn blob(byte: u8, size: u64) -> BlobRef {
	BlobRef { hash: [byte; 32], size }
}

fn model() -> ModelRef {
	ModelRef {
		provider: ProviderId(text("provider")),
		api:      text("responses"),
		model:    ModelId(text("model")),
	}
}

fn header() -> Header {
	Header {
		v:       4,
		id:      SessionId(text("session")),
		created: 123,
		cwd:     PathBuf::from("/tmp/work"),
	}
}

fn title(ts: u64, value: &str) -> Event {
	Event { ts, kind: Kind::Title { title: text(value), source: TitleSource::User } }
}

fn every_kind() -> Vec<Event> {
	let mut replay_fields = BTreeMap::new();
	replay_fields.insert(text("id"), raw(r#"{ "z":2, "a":1 }"#));
	let assistant = Msg::Assistant {
		content:     vec![Block {
			kind: BlockKind::Tool {
				id:   CallId(text("call")),
				name: text("tool"),
				wire: Some(text("wire_tool")),
				args: text(r#"{"b":2,"a":1}"#),
			},
			re:   Some(Replay { p: DialectId(text("oai")), f: replay_fields }),
		}],
		model:       model(),
		stop:        Stop::ToolUse,
		usage:       Usage { input: 10, output: 2, cache_read: 3, cache_write: 4 },
		response_id: Some(text("response")),
		upstream:    Some(text("route")),
		ctx:         Some(CtxSnapshot { tokens: 19, limit: 100 }),
		timing:      Timing { duration_ms: 50, ttft_ms: 10 },
		disabled:    vec![FeatureId(text("search"))],
	};
	vec![
		Event {
			ts:   1,
			kind: Kind::Init {
				system_prompt: blob(1, 8),
				tools:         vec![text("read"), text("write")],
				agent:         Some(text("worker")),
				output_schema: Some(raw(r#"{ "required" : ["x"], "type":"object" }"#)),
			},
		},
		Event {
			ts:   2,
			kind: Kind::Msg(Msg::User {
				content:     vec![UserBlock::Text { text: text("hello") }],
				synthetic:   false,
				steering:    false,
				attribution: Some(Attribution { source: text("human"), id: None }),
			}),
		},
		Event { ts: 3, kind: Kind::Msg(assistant) },
		Event {
			ts:   4,
			kind: Kind::Failed {
				error: RequestError {
					message: text("failed"),
					code:    Some(text("bad_request")),
					status:  Some(400),
					details: Some(raw(r#"{"b": 2, "a":1}"#)),
				},
				model: model(),
				usage: Some(Usage::default()),
			},
		},
		Event {
			ts:   5,
			kind: Kind::Infer {
				thinking: Patch::Set(ThinkingSel {
					effective:  text("high"),
					configured: text("auto"),
				}),
				model:    Patch::Set(ModelChange {
					role:     text("primary"),
					model:    model(),
					fallback: false,
				}),
				tier:     Patch::Clear,
				cred_pin: Patch::Set(Pin {
					provider:   ProviderId(text("provider")),
					credential: text("credential"),
				}),
			},
		},
		Event { ts: 6, kind: Kind::Rewind { to: Some(1) } },
		Event {
			ts:   7,
			kind: Kind::Compact {
				summary:       text("summary"),
				short:         Some(text("short")),
				first_kept:    2,
				tokens_before: 99,
				warning:       Some(text("warning")),
			},
		},
		Event { ts: 8, kind: Kind::Branch { from: 1, summary: text("branch") } },
		Event { ts: 9, kind: Kind::Reset },
		Event {
			ts:   10,
			kind: Kind::Title { title: text("title"), source: TitleSource::Assistant },
		},
		Event { ts: 11, kind: Kind::AddDirs { dirs: vec![PathBuf::from("/tmp/other")] } },
		Event {
			ts:   12,
			kind: Kind::ForkedFrom { session: SessionId(text("parent")), at: Some(4) },
		},
		Event {
			ts:   13,
			kind: Kind::NativeCheckpoint {
				provider: ProviderId(text("provider")),
				model:    ModelId(text("model")),
				items:    blob(2, 16),
			},
		},
		Event { ts: 14, kind: Kind::Aborted { tool_call_ids: vec![CallId(text("call"))] } },
		Event {
			ts:   15,
			kind: Kind::Amend { target: 2, patch: AmendPatch::Prune { keep_blocks: 1 } },
		},
		Event { ts: 16, kind: Kind::Label { target: 2, label: Some(text("good")) } },
		Event {
			ts:   17,
			kind: Kind::Custom {
				kind:    text("extension"),
				data:    Some(raw(r#"{ "z" : [3,2,1], "a":"x&y" }"#)),
				context: Some(vec![UserBlock::Image { blob: blob(3, 32) }]),
				display: true,
			},
		},
		Event {
			ts:   18,
			kind: Kind::Item(ItemRecord {
				item:        thread_pb::Item {
					seq:           0,
					created_at_ms: 18,
					kind:          Some(thread_pb::item::Kind::Message(thread_pb::Message {
						role:  thread_pb::Role::User as i32,
						parts: vec![thread_pb::Part {
							kind: Some(thread_pb::part::Kind::Text("canonical".to_owned())),
						}],
					})),
					props:         None,
				},
				turn_id:     Some(text("turn-1")),
				prompt_hash: Some([4; 32]),
			}),
		},
		Event { ts: 19, kind: Kind::Amend { target: 17, patch: AmendPatch::Seq { seq: 7 } } },
		Event {
			ts:   20,
			kind: Kind::TurnStart(TurnStart {
				turn_id:            text("turn-1"),
				item_events:        vec![17],
				prompt_hash:        [4; 32],
				prompt_head_events: vec![17],
				toolset_hash:       [5; 32],
				enabled_tools:      Vec::new(),
				sequence_targets:   vec![17],
				input:              TurnInputRecord::Delta {
					context: pb::ContextRef {
						context_id: "context".to_owned(),
						expected: Some(thread_pb::Revision { head: 6, token: vec![8; 32].into() }),
					},
					delta: pb::ThreadDelta { truncate_to: None, append: Vec::new() },
				},
				options: TurnOptionsRecord {
					context_id: None,
					params: pb::ChatParams::default(),
					executor: None,
					props: None,
				},
			}),
		},
		Event {
			ts:   21,
			kind: Kind::TurnReceipt(TurnReceipt {
				turn_id:            text("turn-1"),
				prompt_hash:        [4; 32],
				prompt_head_events: vec![17],
				item_events:        vec![17],
				outcome:            pb::Outcome {
					output: vec![thread_pb::Item {
						seq:           7,
						created_at_ms: 18,
						kind:          Some(thread_pb::item::Kind::Message(thread_pb::Message {
							role:  thread_pb::Role::Assistant as i32,
							parts: vec![thread_pb::Part {
								kind: Some(thread_pb::part::Kind::Text("done".to_owned())),
							}],
						})),
						props:         None,
					}],
					stop: pb::StopReason::StopEndTurn as i32,
					revision: Some(thread_pb::Revision { head: 7, token: vec![9; 32].into() }),
					provider: "fixture".to_owned(),
					model: "fixture-model".to_owned(),
					..Default::default()
				},
			}),
		},
		Event {
			ts: 22,
			kind: Kind::TurnInput(TurnInputItem {
				turn_id: text("turn-2"),
				item: thread_pb::Item {
					seq: 0,
					created_at_ms: 22,
					kind: Some(thread_pb::item::Kind::Message(thread_pb::Message {
						role: thread_pb::Role::User as i32,
						parts: vec![thread_pb::Part {
							kind: Some(thread_pb::part::Kind::Text("next".to_owned())),
						}],
					})),
					props: None,
				},
				prompt_hash: Some([4; 32]),
			}),
		},
		Event {
			ts: 23,
			kind: Kind::ToolBatchAuthorized(ToolBatchAuthorized {
				turn_id: text("turn-1"),
				call_ids: vec![text("call-1")],
			}),
		},
		Event {
			ts:   22,
			kind: Kind::Unknown(raw(r#"{ "foreign" : true, "ts" : 22, "k":"else" }"#)),
		},
	]
}

#[test]
fn unknown_line_is_byte_verbatim() {
	let source = br#"{  "foreign" : "a&b", "z" : [3, 2], "ts" : 77, "k" : "alien" }"#;
	let event = read_line(source).expect("foreign object is readable");
	let mut encoded = Vec::new();
	write_line(&event, &mut encoded).expect("foreign object is writable");
	assert_eq!(encoded.as_slice(), source);
}

#[test]
fn embedded_raw_value_is_spliced_verbatim() {
	let raw_data = r#"{ "z" : [3, 2], "a" : "x&y" }"#;
	let event = Event {
		ts:   4,
		kind: Kind::Custom {
			kind:    text("raw"),
			data:    Some(raw(raw_data)),
			context: None,
			display: false,
		},
	};
	let mut first = Vec::new();
	write_line(&event, &mut first).expect("custom event writes");
	assert!(
		std::str::from_utf8(&first)
			.expect("JSON is UTF-8")
			.contains(raw_data)
	);
	let decoded = read_line(&first).expect("custom event reads");
	let mut second = Vec::new();
	write_line(&decoded, &mut second).expect("custom event rewrites");
	assert_eq!(second, first);
}

#[test]
fn every_event_kind_is_idempotent() {
	for event in every_kind() {
		let mut first = Vec::new();
		write_line(&event, &mut first).expect("event writes");
		let decoded = read_line(&first).expect("event reads");
		assert_eq!(decoded, event);
		let mut second = Vec::new();
		write_line(&decoded, &mut second).expect("event rewrites");
		assert_eq!(second, first);
	}
}

#[test]
fn header_is_single_and_torn_tail_is_truncated() {
	let directory = tempdir().expect("temporary directory");
	let path = directory.path().join("session.jsonl");
	let mut writer = Writer::create(&path, &header()).expect("new transcript");
	assert!(matches!(writer.write_header(&header()), Err(Error::DuplicateHeader)));
	let mut duplicate = Vec::new();
	write_header(&header(), &mut duplicate).expect("duplicate header encodes");
	let duplicate = Event {
		ts:   0,
		kind: Kind::Unknown(raw(std::str::from_utf8(&duplicate).expect("header is UTF-8"))),
	};
	assert!(matches!(writer.append(&duplicate), Err(Error::DuplicateHeader)));
	assert_eq!(writer.append(&title(1, "first")).expect("first event"), 0);
	drop(writer);

	let mut file = OpenOptions::new()
		.append(true)
		.open(&path)
		.expect("append torn fragment");
	file
		.write_all(br#"{"ts":2,"k":"title","title":"tor"#)
		.expect("write torn fragment");
	drop(file);

	let mut writer = Writer::open_append(&path).expect("repair torn tail");
	assert_eq!(writer.append(&title(3, "second")).expect("second event"), 1);
	drop(writer);
	let log = load(&path).expect("repaired transcript loads");
	assert_eq!(log.len(), 2);
}

#[test]
fn malformed_middle_line_is_a_tombstone() {
	let directory = tempdir().expect("temporary directory");
	let path = directory.path().join("session.jsonl");
	let mut bytes = Vec::new();
	write_header(&header(), &mut bytes).expect("header writes");
	bytes.extend_from_slice(b"\n{\"ts\":1,\"k\":\"reset\"}\n{not json}\n{\"ts\":3,\"k\":\"title\",\"title\":\"later\",\"source\":\"user\"}\n");
	std::fs::write(&path, bytes).expect("fixture writes");
	let log = load(&path).expect("fixture loads");
	assert_eq!(log.len(), 3);
	assert!(matches!(log.get(1), Some(Entry::Tombstone(_))));
	assert!(matches!(
		log.get(2),
		Some(Entry::Ok(event)) if matches!(&event.kind, Kind::Title { title, .. } if title.as_str() == "later")
	));
}

#[test]
fn forward_fold_applies_rewind_reset_and_compact() {
	let directory = tempdir().expect("temporary directory");
	let path = directory.path().join("session.jsonl");
	let mut writer = Writer::create(&path, &header()).expect("new transcript");
	writer.append(&title(1, "zero")).expect("event zero");
	writer.append(&title(2, "one")).expect("event one");
	writer
		.append(&Event { ts: 3, kind: Kind::Rewind { to: Some(0) } })
		.expect("rewind");
	writer.append(&title(4, "three")).expect("event three");
	writer
		.append(&Event { ts: 5, kind: Kind::Reset })
		.expect("reset");
	writer.append(&title(6, "five")).expect("event five");
	writer.append(&title(7, "six")).expect("event six");
	writer
		.append(&Event {
			ts:   8,
			kind: Kind::Compact {
				summary:       text("summary"),
				short:         None,
				first_kept:    5,
				tokens_before: 50,
				warning:       None,
			},
		})
		.expect("compact");
	drop(writer);
	assert_eq!(load(&path).expect("transcript loads").live(), vec![7, 5, 6]);
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
struct PatchHolder {
	#[serde(default, skip_serializing_if = "Patch::is_unchanged")]
	value: Patch<u64>,
}

#[test]
fn patch_absent_null_and_value_are_distinct() {
	let absent: PatchHolder = serde_json::from_str("{}").expect("absent patch");
	let clear: PatchHolder = serde_json::from_str(r#"{"value":null}"#).expect("clear patch");
	let set: PatchHolder = serde_json::from_str(r#"{"value":9}"#).expect("set patch");
	assert_eq!(absent.value, Patch::Unchanged);
	assert_eq!(clear.value, Patch::Clear);
	assert_eq!(set.value, Patch::Set(9));
	assert_eq!(serde_json::to_string(&absent).expect("serialize absent patch"), "{}");
}

#[test]
fn writer_rejects_empty_infer_event() {
	let directory = tempdir().expect("temporary directory");
	let path = directory.path().join("session.jsonl");
	let mut writer = Writer::create(&path, &header()).expect("new transcript");
	let event = Event {
		ts:   1,
		kind: Kind::Infer {
			thinking: Patch::Unchanged,
			model:    Patch::Unchanged,
			tier:     Patch::Unchanged,
			cred_pin: Patch::Unchanged,
		},
	};
	assert!(matches!(writer.append(&event), Err(Error::EmptyInfer)));
}
