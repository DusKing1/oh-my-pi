//! Pure transcript-to-thread projection and canonical tool-result lowering.

use std::collections::BTreeMap;

use bytes::Bytes;
use omp_core::{Str, encoding::hex};
use omp_proto::{inference::v1 as pb, thread::v1 as thread_pb};
use omp_storage::transcript::{AmendPatch, Entry, Kind, Log};
use omp_tool::{
	Abort, Part as ToolPart, ProjectedCall, PromptCaps, RecordedCallOwned, Registry as ToolRegistry,
	Rev, TOOL_REV_PROP, ToolIdentity, Verdict,
};
use thiserror::Error;

/// Canonical thread projection failure.
#[derive(Debug, Error)]
pub enum ProjectionError {
	/// A committed tool revision property had the wrong shape.
	#[error("omp/tool-rev must be a string")]
	RevisionType,
	/// A committed tool revision string was malformed.
	#[error("omp/tool-rev contains an invalid revision")]
	InvalidRevision,
	/// Structured tool verdict JSON was invalid.
	#[error("invalid tool verdict JSON: {0}")]
	VerdictJson(#[from] serde_json::Error),
	/// A model-facing JSON part was not UTF-8.
	#[error("tool JSON part is not UTF-8: {0}")]
	PartUtf8(#[from] std::str::Utf8Error),
	/// A model-facing blob hash was not hexadecimal.
	#[error("tool blob hash is not valid hexadecimal")]
	BlobHash,
	/// The live tool could not deterministically render a lifted verdict.
	#[error("tool projection failed: {0}")]
	Tool(#[from] omp_tool::RegistryError),
	/// A recovery target was not a canonical tool call.
	#[error("tool recovery target is not a tool call")]
	ExpectedToolCall,
	/// A committed tool call lacked its durable revision identity.
	#[error("committed tool call is missing omp/tool-rev")]
	MissingRevision,
}

/// Projects the live append-only journal chain into one canonical thread.
///
/// Rewinds are resolved by [`Log::live`]. Sequence amendments update only the
/// working copy; original item events remain untouched.
pub fn project_journal(
	log: &Log,
	tool_registry: &ToolRegistry,
	caps: &PromptCaps,
) -> Result<thread_pb::Thread, ProjectionError> {
	let mut items = Vec::new();
	let mut positions = BTreeMap::new();
	for index in log.live() {
		let Some(Entry::Ok(event)) = log.get(index) else {
			continue;
		};
		match &event.kind {
			Kind::Item(record) => {
				positions.insert(index, items.len());
				items.push(record.item.clone());
			},
			Kind::TurnInput(input) => {
				positions.insert(index, items.len());
				items.push(input.item.clone());
			},
			Kind::PromptRewriteStage(stage) => {
				positions.insert(index, items.len());
				items.push(stage.item.clone());
			},
			Kind::JobSettled(settled) => {
				positions.insert(index, items.len());
				items.push(settled.settlement.clone());
			},
			Kind::Amend { target, patch: AmendPatch::Seq { seq } } => {
				if let Some(position) = positions.get(target).copied() {
					items[position].seq = *seq;
				}
			},
			_ => {},
		}
	}
	project_thread_history(&thread_pb::Thread { items }, tool_registry, caps)
}

/// Re-expresses historical tool calls through complete live revision lifts.
///
/// Calls without a complete lift path are retained exactly. Calls already at
/// the live revision are not decoded or rewritten, preserving their bytes and
/// field presence exactly.
pub fn project_thread_history(
	thread: &thread_pb::Thread,
	tool_registry: &ToolRegistry,
	caps: &PromptCaps,
) -> Result<thread_pb::Thread, ProjectionError> {
	let mut projected = thread.clone();
	for call_index in 0..projected.items.len() {
		let Some(thread_pb::item::Kind::ToolCall(call)) = projected.items[call_index].kind.as_ref()
		else {
			continue;
		};
		let Some(rev) = tool_revision(&projected.items[call_index])? else {
			continue;
		};
		let call_id = call.id.clone();
		let name = call.name.clone();
		let Some((_, live_rev)) = tool_registry.live_identity(&name) else {
			continue;
		};
		if live_rev == &rev {
			continue;
		}
		let raw_args = call.args_json.clone();
		let Some(result_index) = projected
			.items
			.iter()
			.enumerate()
			.skip(call_index + 1)
			.find_map(|(index, item)| {
				matches!(
					item.kind.as_ref(),
					Some(thread_pb::item::Kind::ToolResult(result))
						if result.call_id == call_id && result.details.is_some()
				)
				.then_some(index)
			})
		else {
			continue;
		};
		let Some(thread_pb::item::Kind::ToolResult(result)) =
			projected.items[result_index].kind.as_ref()
		else {
			unreachable!("result index came from ToolResult items")
		};
		let recorded_useless = result.useless.unwrap_or(false);
		let Some(verdict) = proto_json_bytes(
			result
				.details
				.as_ref()
				.expect("selected result has structured details"),
		) else {
			continue;
		};
		let original = RecordedCallOwned {
			identity: ToolIdentity { name: Str::from(name.as_str()), rev: rev.clone() },
			raw_args: Bytes::copy_from_slice(&raw_args),
			verdict,
		};
		let ProjectedCall::Live(live) = tool_registry.project(original) else {
			continue;
		};
		let rendered = tool_registry.project_verdict(
			&live.identity,
			&live.verdict,
			recorded_useless,
			caps,
		)?;
		let lifted_verdict: serde_json::Value = serde_json::from_slice(&live.verdict)?;
		let lifted_details = json_proto_value(lifted_verdict);
		let lifted_parts = tool_parts(&rendered.parts)?;

		let Some(thread_pb::item::Kind::ToolCall(call)) = projected.items[call_index].kind.as_mut()
		else {
			unreachable!("call index came from ToolCall items")
		};
		call.args_json = live.raw_args.clone();
		let props = projected.items[call_index].props.get_or_insert_default();
		props.fields.insert(TOOL_REV_PROP.to_owned(), pb::Value {
			kind: Some(pb::value::Kind::String(live.identity.rev.to_string())),
		});
		let result_props = projected.items[result_index].props.get_or_insert_default();
		result_props
			.fields
			.insert(TOOL_REV_PROP.to_owned(), pb::Value {
				kind: Some(pb::value::Kind::String(live.identity.rev.to_string())),
			});

		let Some(thread_pb::item::Kind::ToolResult(result)) =
			projected.items[result_index].kind.as_mut()
		else {
			unreachable!("result index came from ToolResult items")
		};
		result.details = Some(lifted_details);
		result.parts = lifted_parts;
		result.is_error = rendered.is_error;
		result.useless = Some(rendered.useless);
	}
	Ok(projected)
}

pub(crate) fn recovery_tool_result_item(
	created_at_ms: u64,
	call_item: &thread_pb::Item,
	abort: Abort,
) -> Result<thread_pb::Item, ProjectionError> {
	let Some(thread_pb::item::Kind::ToolCall(call)) = call_item.kind.as_ref() else {
		return Err(ProjectionError::ExpectedToolCall);
	};
	let rev = tool_revision(call_item)?.ok_or(ProjectionError::MissingRevision)?;
	let identity = ToolIdentity { name: Str::from(call.name.as_str()), rev };
	let text = match &abort {
		Abort::Skipped { reason } => format!("skipped: {reason}"),
		Abort::Interrupted { reason } => format!("interrupted: {reason}"),
		Abort::EffectsUnknown { reason } => format!("aborted with effects unknown: {reason}"),
		Abort::InputDropped => "aborted: invocation input dropped before commit".to_owned(),
		Abort::MissingOutcome => "aborted: executor ended without a terminal outcome".to_owned(),
	};
	let verdict = Verdict::<serde_json::Value, serde_json::Value>::Aborted(abort);
	let raw = serde_json::to_vec(&verdict)?;
	tool_result_item(
		created_at_ms,
		&call.id,
		&identity,
		&raw,
		true,
		false,
		&[ToolPart::Text { text: Str::from(text) }],
	)
}

/// Builds one canonical optimistic tool-result item from durable tool truth.
pub fn tool_result_item(
	created_at_ms: u64,
	call_id: &str,
	identity: &ToolIdentity,
	verdict: &[u8],
	is_error: bool,
	useless: bool,
	parts: &[ToolPart],
) -> Result<thread_pb::Item, ProjectionError> {
	build_tool_result_item(
		created_at_ms,
		call_id,
		identity,
		verdict,
		is_error,
		useless,
		tool_parts(parts)?,
	)
}

/// Builds a canonical optimistic tool result while preserving wire-provided
/// canonical parts as an authoritative fallback.
pub fn tool_result_item_canonical_parts(
	created_at_ms: u64,
	call_id: &str,
	identity: &ToolIdentity,
	verdict: &[u8],
	is_error: bool,
	useless: bool,
	parts: Vec<thread_pb::Part>,
) -> Result<thread_pb::Item, ProjectionError> {
	build_tool_result_item(created_at_ms, call_id, identity, verdict, is_error, useless, parts)
}

fn build_tool_result_item(
	created_at_ms: u64,
	call_id: &str,
	identity: &ToolIdentity,
	verdict: &[u8],
	is_error: bool,
	useless: bool,
	parts: Vec<thread_pb::Part>,
) -> Result<thread_pb::Item, ProjectionError> {
	let details = json_proto_value(serde_json::from_slice(verdict)?);
	let props = pb::ValueMap {
		fields: BTreeMap::from([(TOOL_REV_PROP.to_owned(), pb::Value {
			kind: Some(pb::value::Kind::String(identity.rev.to_string())),
		})]),
	};
	Ok(thread_pb::Item {
		seq: 0,
		created_at_ms,
		kind: Some(thread_pb::item::Kind::ToolResult(thread_pb::ToolResult {
			call_id: call_id.to_owned(),
			parts,
			is_error,
			name: identity.name.as_str().to_owned(),
			details: Some(details),
			useless: Some(useless),
			..Default::default()
		})),
		props: Some(props),
	})
}

fn tool_revision(item: &thread_pb::Item) -> Result<Option<Rev>, ProjectionError> {
	let Some(value) = item
		.props
		.as_ref()
		.and_then(|props| props.fields.get(TOOL_REV_PROP))
	else {
		return Ok(None);
	};
	let Some(pb::value::Kind::String(value)) = value.kind.as_ref() else {
		return Err(ProjectionError::RevisionType);
	};
	let (family, number) = value
		.rsplit_once('.')
		.map_or(("", value.as_str()), |(family, number)| (family, number));
	if number.is_empty() {
		return Err(ProjectionError::InvalidRevision);
	}
	let n = number
		.parse::<u16>()
		.map_err(|_| ProjectionError::InvalidRevision)?;
	Ok(Some(Rev { family: Str::from(family), n }))
}

fn proto_json_bytes(value: &pb::Value) -> Option<Bytes> {
	serde_json::to_vec(&proto_json_value(value)?)
		.ok()
		.map(Bytes::from)
}

fn proto_json_value(value: &pb::Value) -> Option<serde_json::Value> {
	let value = match value.kind.as_ref()? {
		pb::value::Kind::Null(_) => serde_json::Value::Null,
		pb::value::Kind::Int(value) => serde_json::Value::from(*value),
		pb::value::Kind::Double(value) => {
			serde_json::Value::Number(serde_json::Number::from_f64(*value)?)
		},
		pb::value::Kind::Bool(value) => serde_json::Value::Bool(*value),
		pb::value::Kind::String(value) => serde_json::Value::String(value.clone()),
		pb::value::Kind::List(list) => serde_json::Value::Array(
			list
				.values
				.iter()
				.map(proto_json_value)
				.collect::<Option<Vec<_>>>()?,
		),
		pb::value::Kind::Map(map) => serde_json::Value::Object(
			map.fields
				.iter()
				.map(|(key, value)| Some((key.clone(), proto_json_value(value)?)))
				.collect::<Option<serde_json::Map<_, _>>>()?,
		),
		pb::value::Kind::Uint(value) => serde_json::Value::from(*value),
	};
	Some(value)
}

fn json_proto_value(value: serde_json::Value) -> pb::Value {
	let kind = match value {
		serde_json::Value::Null => pb::value::Kind::Null(true),
		serde_json::Value::Bool(value) => pb::value::Kind::Bool(value),
		serde_json::Value::Number(value) => {
			if let Some(value) = value.as_i64() {
				pb::value::Kind::Int(value)
			} else if let Some(value) = value.as_u64() {
				pb::value::Kind::Uint(value)
			} else {
				pb::value::Kind::Double(value.as_f64().expect("JSON numbers are finite"))
			}
		},
		serde_json::Value::String(value) => pb::value::Kind::String(value),
		serde_json::Value::Array(values) => pb::value::Kind::List(pb::ValueList {
			values: values.into_iter().map(json_proto_value).collect(),
		}),
		serde_json::Value::Object(fields) => pb::value::Kind::Map(pb::ValueMap {
			fields: fields
				.into_iter()
				.map(|(key, value)| (key, json_proto_value(value)))
				.collect(),
		}),
	};
	pb::Value { kind: Some(kind) }
}

fn tool_parts(parts: &[ToolPart]) -> Result<Vec<thread_pb::Part>, ProjectionError> {
	let mut projected = Vec::with_capacity(parts.len());
	for part in parts {
		match part {
			ToolPart::Text { text } => projected.push(thread_pb::Part {
				kind: Some(thread_pb::part::Kind::Text(text.as_str().to_owned())),
			}),
			ToolPart::Json { json } => projected.push(thread_pb::Part {
				kind: Some(thread_pb::part::Kind::Text(std::str::from_utf8(json)?.to_owned())),
			}),
			ToolPart::Blob { blob, alt } => {
				if let Some(alt) = alt {
					projected.push(thread_pb::Part {
						kind: Some(thread_pb::part::Kind::Text(alt.as_str().to_owned())),
					});
				}
				let hash = hex::decode(blob.hash.as_str())
					.into_vec()
					.map_err(|_| ProjectionError::BlobHash)?;
				if hash.len() != 32 {
					return Err(ProjectionError::BlobHash);
				}
				projected.push(thread_pb::Part {
					kind: Some(thread_pb::part::Kind::Blob(thread_pb::Blob {
						hash: hash.into(),
						mime: blob.media_type.as_str().to_owned(),
						size: blob.byte_len,
						..Default::default()
					})),
				});
			},
		}
	}
	Ok(projected)
}
