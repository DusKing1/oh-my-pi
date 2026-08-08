//! Hand-written JSON line codec preserving raw payload bytes.

use std::{path::PathBuf, str::Utf8Error};

use bytes::BufMut;
use omp_core::SmolStr;
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use thiserror::Error as ThisError;

use super::{
	event::{Event, Kind},
	msg::{Content, Msg},
	patch::Patch,
	types::{
		AmendPatch, CallId, ModelChange, ModelId, ModelRef, Pin, ProviderId, RequestError, SessionId,
		ThinkingSel, Tier, TitleSource, Usage,
	},
};
use crate::blob::BlobRef;

/// Transcript encoding, decoding, and file-integrity errors.
#[derive(Debug, ThisError)]
pub enum Error {
	/// A file-system operation failed.
	#[error("transcript I/O failed: {0}")]
	Io(#[from] std::io::Error),
	/// A JSON object could not be encoded or decoded.
	#[error("invalid transcript JSON: {0}")]
	Json(#[from] serde_json::Error),
	/// A journal line was not valid UTF-8.
	#[error("transcript line is not UTF-8: {0}")]
	Utf8(#[from] Utf8Error),
	/// A journal did not contain its required line-zero header.
	#[error("transcript header is missing")]
	MissingHeader,
	/// The line-zero header used an unsupported format version.
	#[error("unsupported transcript version {0}")]
	InvalidHeaderVersion(u8),
	/// A writer was asked to add a second header.
	#[error("a transcript may contain exactly one header")]
	DuplicateHeader,
	/// A recognized event did not contain its timestamp.
	#[error("recognized transcript event is missing `ts`")]
	MissingTimestamp,
	/// An inference update did not change or clear any field.
	#[error("an infer event must change or clear at least one field")]
	EmptyInfer,
}

/// The identity header stored at line zero of every transcript v4 file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Header {
	/// Transcript format version; transcript v4 requires the value `4`.
	pub v:       u8,
	/// Stable session identifier.
	pub id:      SessionId,
	/// Session creation time in epoch milliseconds.
	pub created: u64,
	/// Absolute working directory at session creation.
	pub cwd:     PathBuf,
}

struct Object<'a, B> {
	out:   &'a mut B,
	first: bool,
}

impl<'a, B: BufMut> Object<'a, B> {
	fn new(out: &'a mut B) -> Self {
		out.put_u8(b'{');
		Self { out, first: true }
	}

	fn field<T>(&mut self, name: &'static str, value: &T) -> Result<(), Error>
	where
		T: Serialize + ?Sized,
	{
		if self.first {
			self.first = false;
		} else {
			self.out.put_u8(b',');
		}
		serde_json::to_writer((&mut *self.out).writer(), name)?;
		self.out.put_u8(b':');
		serde_json::to_writer((&mut *self.out).writer(), value)?;
		Ok(())
	}

	fn finish(self) {
		self.out.put_u8(b'}');
	}
}

/// Writes a header object without a trailing newline.
pub fn write_header(header: &Header, out: &mut impl BufMut) -> Result<(), Error> {
	let mut object = Object::new(out);
	object.field("v", &header.v)?;
	object.field("id", &header.id)?;
	object.field("created", &header.created)?;
	object.field("cwd", &header.cwd)?;
	object.finish();
	Ok(())
}

/// Reads and validates a transcript v4 header object.
pub fn read_header(line: &[u8]) -> Result<Header, Error> {
	let header: Header = serde_json::from_slice(line)?;
	if header.v != 4 {
		return Err(Error::InvalidHeaderVersion(header.v));
	}
	Ok(header)
}

/// Writes one event object without a trailing newline.
///
/// Unknown event objects are copied byte-for-byte. Recognized objects are
/// emitted directly to the destination, and every [`RawValue`] is serialized as
/// a raw JSON fragment rather than buffered through an intermediate value tree.
pub fn write_line(event: &Event, out: &mut impl BufMut) -> Result<(), Error> {
	if let Kind::Unknown(raw) = &event.kind {
		out.put_slice(raw.get().as_bytes());
		return Ok(());
	}
	let mut object = Object::new(out);
	object.field("ts", &event.ts)?;
	match &event.kind {
		Kind::Init { system_prompt, tools, agent, output_schema } => {
			object.field("k", "init")?;
			object.field("system_prompt", system_prompt)?;
			object.field("tools", tools)?;
			object.field("agent", agent)?;
			object.field("output_schema", output_schema)?;
		},
		Kind::Msg(message) => write_msg_fields(&mut object, message)?,
		Kind::Failed { error, model, usage } => {
			object.field("k", "failed")?;
			object.field("error", error)?;
			object.field("model", model)?;
			object.field("usage", usage)?;
		},
		Kind::Infer { thinking, model, tier, cred_pin } => {
			object.field("k", "infer")?;
			if !thinking.is_unchanged() {
				object.field("thinking", thinking)?;
			}
			if !model.is_unchanged() {
				object.field("model", model)?;
			}
			if !tier.is_unchanged() {
				object.field("tier", tier)?;
			}
			if !cred_pin.is_unchanged() {
				object.field("cred_pin", cred_pin)?;
			}
		},
		Kind::Rewind { to } => {
			object.field("k", "rewind")?;
			object.field("to", to)?;
		},
		Kind::Compact { summary, short, first_kept, tokens_before, warning } => {
			object.field("k", "compact")?;
			object.field("summary", summary)?;
			object.field("short", short)?;
			object.field("first_kept", first_kept)?;
			object.field("tokens_before", tokens_before)?;
			object.field("warning", warning)?;
		},
		Kind::Branch { from, summary } => {
			object.field("k", "branch")?;
			object.field("from", from)?;
			object.field("summary", summary)?;
		},
		Kind::Reset => object.field("k", "reset")?,
		Kind::Title { title, source } => {
			object.field("k", "title")?;
			object.field("title", title)?;
			object.field("source", source)?;
		},
		Kind::AddDirs { dirs } => {
			object.field("k", "add_dirs")?;
			object.field("dirs", dirs)?;
		},
		Kind::ForkedFrom { session, at } => {
			object.field("k", "forked_from")?;
			object.field("session", session)?;
			object.field("at", at)?;
		},
		Kind::NativeCheckpoint { provider, model, items } => {
			object.field("k", "native_checkpoint")?;
			object.field("provider", provider)?;
			object.field("model", model)?;
			object.field("items", items)?;
		},
		Kind::Aborted { tool_call_ids } => {
			object.field("k", "aborted")?;
			object.field("tool_call_ids", tool_call_ids)?;
		},
		Kind::Amend { target, patch } => {
			object.field("k", "amend")?;
			object.field("target", target)?;
			object.field("patch", patch)?;
		},
		Kind::Label { target, label } => {
			object.field("k", "label")?;
			object.field("target", target)?;
			object.field("label", label)?;
		},
		Kind::Custom { kind, data, context, display } => {
			object.field("k", "custom")?;
			object.field("kind", kind)?;
			object.field("data", data)?;
			object.field("context", context)?;
			object.field("display", display)?;
		},
		Kind::Unknown(_) => unreachable!("unknown events return before object encoding"),
	}
	object.finish();
	Ok(())
}

fn write_msg_fields<B: BufMut>(object: &mut Object<'_, B>, message: &Msg) -> Result<(), Error> {
	object.field("k", "msg")?;
	match message {
		Msg::User { content, synthetic, steering, attribution } => {
			object.field("role", "user")?;
			object.field("content", content)?;
			object.field("synthetic", synthetic)?;
			object.field("steering", steering)?;
			object.field("attribution", attribution)?;
		},
		Msg::Developer { content, attribution } => {
			object.field("role", "developer")?;
			object.field("content", content)?;
			object.field("attribution", attribution)?;
		},
		Msg::Assistant {
			content,
			model,
			stop,
			usage,
			response_id,
			upstream,
			ctx,
			timing,
			disabled,
		} => {
			object.field("role", "assistant")?;
			object.field("content", content)?;
			object.field("model", model)?;
			object.field("stop", stop)?;
			object.field("usage", usage)?;
			object.field("response_id", response_id)?;
			object.field("upstream", upstream)?;
			object.field("ctx", ctx)?;
			object.field("timing", timing)?;
			object.field("disabled", disabled)?;
		},
		Msg::ToolResult { call, tool, content, details, error, useless, provider_meta } => {
			object.field("role", "tool_result")?;
			object.field("call", call)?;
			object.field("tool", tool)?;
			object.field("content", content)?;
			object.field("details", details)?;
			object.field("error", error)?;
			object.field("useless", useless)?;
			object.field("provider_meta", provider_meta)?;
		},
	}
	Ok(())
}

#[derive(Deserialize)]
struct Probe {
	#[serde(default)]
	ts: Option<u64>,
	#[serde(default)]
	k:  Option<SmolStr>,
}

macro_rules! payload {
	($name:ident { $($field:ident : $ty:ty),* $(,)? }) => {
		#[derive(Deserialize)]
		struct $name {
			$($field: $ty),*
		}
	};
}

payload!(InitPayload {
	system_prompt: BlobRef,
	tools: Vec<SmolStr>,
	agent: Option<SmolStr>,
	output_schema: Option<Box<RawValue>>,
});
payload!(FailedPayload {
	error: RequestError,
	model: ModelRef,
	usage: Option<Usage>,
});

#[derive(Serialize, Deserialize)]
struct InferPayload {
	#[serde(default, skip_serializing_if = "Patch::is_unchanged")]
	thinking: Patch<ThinkingSel>,
	#[serde(default, skip_serializing_if = "Patch::is_unchanged")]
	model:    Patch<ModelChange>,
	#[serde(default, skip_serializing_if = "Patch::is_unchanged")]
	tier:     Patch<Tier>,
	#[serde(default, skip_serializing_if = "Patch::is_unchanged")]
	cred_pin: Patch<Pin>,
}

payload!(RewindPayload { to: Option<u64> });
payload!(CompactPayload {
	summary: SmolStr,
	short: Option<SmolStr>,
	first_kept: u64,
	tokens_before: u64,
	warning: Option<SmolStr>,
});
payload!(BranchPayload { from: u64, summary: SmolStr });
payload!(TitlePayload { title: SmolStr, source: TitleSource });
payload!(AddDirsPayload { dirs: Vec<PathBuf> });
payload!(ForkedFromPayload { session: SessionId, at: Option<u64> });
payload!(CheckpointPayload { provider: ProviderId, model: ModelId, items: BlobRef });
payload!(AbortedPayload { tool_call_ids: Vec<CallId> });
payload!(AmendPayload { target: u64, patch: AmendPatch });
payload!(LabelPayload { target: u64, label: Option<SmolStr> });
payload!(CustomPayload {
	kind: SmolStr,
	data: Option<Box<RawValue>>,
	context: Option<Content>,
	display: bool,
});

/// Reads one event object, preserving an unrecognized object's complete source
/// bytes.
pub fn read_line(line: &[u8]) -> Result<Event, Error> {
	let probe: Probe = serde_json::from_slice(line)?;
	let Some(tag) = probe.k.as_ref().map(SmolStr::as_str) else {
		return unknown_line(line, probe.ts.unwrap_or_default());
	};
	let Some(ts) = probe.ts else {
		return Err(Error::MissingTimestamp);
	};

	let kind = match tag {
		"init" => {
			let payload: InitPayload = serde_json::from_slice(line)?;
			Kind::Init {
				system_prompt: payload.system_prompt,
				tools:         payload.tools,
				agent:         payload.agent,
				output_schema: payload.output_schema,
			}
		},
		"msg" => Kind::Msg(serde_json::from_slice::<Msg>(line)?),
		"failed" => {
			let payload: FailedPayload = serde_json::from_slice(line)?;
			Kind::Failed { error: payload.error, model: payload.model, usage: payload.usage }
		},
		"infer" => {
			let payload: InferPayload = serde_json::from_slice(line)?;
			Kind::Infer {
				thinking: payload.thinking,
				model:    payload.model,
				tier:     payload.tier,
				cred_pin: payload.cred_pin,
			}
		},
		"rewind" => {
			let payload: RewindPayload = serde_json::from_slice(line)?;
			Kind::Rewind { to: payload.to }
		},
		"compact" => {
			let payload: CompactPayload = serde_json::from_slice(line)?;
			Kind::Compact {
				summary:       payload.summary,
				short:         payload.short,
				first_kept:    payload.first_kept,
				tokens_before: payload.tokens_before,
				warning:       payload.warning,
			}
		},
		"branch" => {
			let payload: BranchPayload = serde_json::from_slice(line)?;
			Kind::Branch { from: payload.from, summary: payload.summary }
		},
		"reset" => Kind::Reset,
		"title" => {
			let payload: TitlePayload = serde_json::from_slice(line)?;
			Kind::Title { title: payload.title, source: payload.source }
		},
		"add_dirs" => {
			let payload: AddDirsPayload = serde_json::from_slice(line)?;
			Kind::AddDirs { dirs: payload.dirs }
		},
		"forked_from" => {
			let payload: ForkedFromPayload = serde_json::from_slice(line)?;
			Kind::ForkedFrom { session: payload.session, at: payload.at }
		},
		"native_checkpoint" => {
			let payload: CheckpointPayload = serde_json::from_slice(line)?;
			Kind::NativeCheckpoint {
				provider: payload.provider,
				model:    payload.model,
				items:    payload.items,
			}
		},
		"aborted" => {
			let payload: AbortedPayload = serde_json::from_slice(line)?;
			Kind::Aborted { tool_call_ids: payload.tool_call_ids }
		},
		"amend" => {
			let payload: AmendPayload = serde_json::from_slice(line)?;
			Kind::Amend { target: payload.target, patch: payload.patch }
		},
		"label" => {
			let payload: LabelPayload = serde_json::from_slice(line)?;
			Kind::Label { target: payload.target, label: payload.label }
		},
		"custom" => {
			let payload: CustomPayload = serde_json::from_slice(line)?;
			Kind::Custom {
				kind:    payload.kind,
				data:    payload.data,
				context: payload.context,
				display: payload.display,
			}
		},
		_ => return unknown_line(line, ts),
	};
	Ok(Event { ts, kind })
}

fn unknown_line(line: &[u8], ts: u64) -> Result<Event, Error> {
	let source = std::str::from_utf8(line)?.to_owned();
	let raw = RawValue::from_string(source)?;
	Ok(Event { ts, kind: Kind::Unknown(raw) })
}
