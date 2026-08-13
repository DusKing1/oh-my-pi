use std::collections::BTreeMap;

use bytes::Bytes;
use omp_agent::ProjectionError;
use omp_proto::{inference::v1 as inference, thread::v1 as thread};
use omp_tool::{TOOL_REV_PROP, ToolIdentity};

/// Builds one canonical text message item with no authority sequence stamp.
#[must_use]
pub fn message_item(role: thread::Role, text: impl Into<String>) -> thread::Item {
	thread::Item {
		kind: Some(thread::item::Kind::Message(thread::Message {
			role: role as i32,
			parts: vec![thread::Part { kind: Some(thread::part::Kind::Text(text.into())) }],
		})),
		..Default::default()
	}
}

/// Builds one canonical user text item.
#[must_use]
pub fn user_item(text: impl Into<String>) -> thread::Item {
	message_item(thread::Role::User, text)
}

/// Builds one canonical assistant text item.
#[must_use]
pub fn assistant_item(text: impl Into<String>) -> thread::Item {
	message_item(thread::Role::Assistant, text)
}

/// Builds a canonical revision-stamped tool-call item.
#[must_use]
pub fn tool_call_item(
	created_at_ms: u64,
	call_id: impl Into<String>,
	identity: &ToolIdentity,
	args_json: impl Into<Bytes>,
) -> thread::Item {
	thread::Item {
		seq: 0,
		created_at_ms,
		kind: Some(thread::item::Kind::ToolCall(thread::ToolCall {
			id: call_id.into(),
			name: identity.name.as_str().to_owned(),
			args_json: args_json.into(),
			..Default::default()
		})),
		props: Some(tool_revision_props(identity)),
	}
}

/// Builds a canonical revision-stamped tool-result item from retained truth.
pub fn tool_result_item(
	created_at_ms: u64,
	call_id: &str,
	identity: &ToolIdentity,
	details: &serde_json::Value,
	is_error: bool,
	useless: bool,
	parts: Vec<thread::Part>,
) -> Result<thread::Item, ProjectionError> {
	let details = serde_json::to_vec(details).expect("serializing a serde_json::Value cannot fail");
	omp_agent::tool_result_item_canonical_parts(
		created_at_ms,
		call_id,
		identity,
		&details,
		is_error,
		useless,
		parts,
	)
}

/// Builds the first successful event in an admitted turn.
#[must_use]
pub fn accepted_event(replay: bool) -> inference::TurnEvent {
	turn_event(inference::turn_event::Event::Accepted(inference::Accepted { replay }))
}

/// Builds one terminal canonical turn outcome event.
#[must_use]
pub fn outcome_event(outcome: inference::Outcome) -> inference::TurnEvent {
	turn_event(inference::turn_event::Event::Outcome(outcome))
}

/// Wraps a generated inference event body.
#[must_use]
pub fn turn_event(event: inference::turn_event::Event) -> inference::TurnEvent {
	inference::TurnEvent { event: Some(event) }
}

fn tool_revision_props(identity: &ToolIdentity) -> inference::ValueMap {
	inference::ValueMap {
		fields: BTreeMap::from([(TOOL_REV_PROP.to_owned(), inference::Value {
			kind: Some(inference::value::Kind::String(identity.rev.to_string())),
		})]),
	}
}
