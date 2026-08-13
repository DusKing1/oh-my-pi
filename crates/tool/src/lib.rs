//! Typed, revisioned tool contracts for the agent/environment boundary.
//!
//! Execution is deliberately absent from this crate. A tool keeps concrete
//! parameter and result types until [`Registry::register`], while prompt
//! projection and revision lifting remain deterministic shared code.

mod incoming;
mod registry;

use std::{fmt, future::Future};

use bytes::Bytes;
use futures::Stream;
pub use incoming::{
	CommitError, IncomingParams, Interrupt, InterruptibleParams, InvocationEvent, InvocationFeed,
	InvocationSendError, ParamError,
};
use omp_core::Str;
pub use registry::{
	ConstraintDisposition, ErasedEv, ErasedOutcome, ErasedStream, LoweredTool, LoweringCaps,
	ProjectedCall, Registry, RegistryError,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
/// Namespaced thread-item property carrying a committed tool revision.
pub const TOOL_REV_PROP: &str = "omp/tool-rev";
use thiserror::Error;

/// One argument-dialect revision within a revision family.
#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct Rev {
	/// Argument-dialect family, such as `hl` or `rep`.
	pub family: Str,
	/// Monotonic revision within `family`.
	pub n:      u16,
}

impl fmt::Display for Rev {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		if self.family.is_empty() {
			write!(f, "{}", self.n)
		} else {
			write!(f, "{}.{}", self.family, self.n)
		}
	}
}

/// Durable identity of a tool call in a transcript.
#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct ToolIdentity {
	/// Stable model-facing name.
	pub name: Str,
	/// Argument and rendering revision.
	pub rev:  Rev,
}

/// Static description of one tool revision.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ToolSpec {
	/// Stable wire name exposed to models.
	pub name:        Str,
	/// Transcript revision.
	pub rev:         Rev,
	/// Model-facing purpose.
	pub description: Str,
	/// Complete JSON Schema bytes.
	pub schema:      Bytes,
	/// Requested constrained-sampling behavior.
	pub constraint:  Constraint,
}

impl ToolSpec {
	/// Returns the durable `(name, family/n)` identity.
	pub fn identity(&self) -> ToolIdentity {
		ToolIdentity { name: self.name.clone(), rev: self.rev.clone() }
	}
}

/// Requested argument-sampling constraint.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Constraint {
	/// Ordinary lenient JSON arguments.
	None,
	/// Strict JSON Schema sampling when supported.
	Schema {
		/// Relative request priority retained for upstream negotiation.
		priority: u8,
	},
	/// Freeform input constrained by a grammar.
	Grammar {
		/// Grammar language.
		syntax:     GrammarSyntax,
		/// Complete grammar definition.
		definition: Str,
		/// Relative request priority retained for upstream negotiation.
		priority:   u8,
	},
}

/// Grammar languages represented in the model catalog.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GrammarSyntax {
	/// Lark grammar.
	Lark,
	/// Regular expression.
	Regex,
	/// Extended Backus-Naur form.
	Ebnf,
}

/// Deterministic model-facing projection budget.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PromptCaps {
	/// Maximum number of parts a tool may emit.
	pub maximum_parts:      u16,
	/// Maximum aggregate UTF-8 text bytes.
	pub maximum_text_bytes: u32,
	/// Whether blob-backed media parts may be exposed to the model.
	pub media:              bool,
}

/// A content-addressed blob reference suitable for durable projection.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct BlobRef {
	/// Content hash in the environment blob namespace.
	pub hash:       Str,
	/// MIME type of the stored bytes.
	pub media_type: Str,
	/// Exact stored byte length.
	pub byte_len:   u64,
}

/// One model-facing tool-result part.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Part {
	/// UTF-8 model-visible text.
	Text {
		/// Model-visible text payload.
		text: Str,
	},
	/// Structured JSON retained as exact bytes.
	Json {
		/// Raw JSON byte payload.
		json: Bytes,
	},
	/// Blob-backed media; never inline base64.
	Blob {
		/// Durable blob reference.
		blob: BlobRef,
		/// Optional deterministic accessibility/model fallback.
		alt:  Option<Str>,
	},
}

/// One typed tool implementation.
pub trait Tool: Send + Sync + 'static {
	/// Declared whole-argument shape for tools which opt into whole validation.
	type Params: DeserializeOwned;
	/// Ephemeral progress payload.
	type Update: Serialize + Send;
	/// Durable successful result.
	type Payload: Serialize + DeserializeOwned + Send;
	/// Durable typed failure.
	type Fault: Serialize + DeserializeOwned + Send;

	/// Returns this implementation's immutable specification.
	fn spec(&self) -> &ToolSpec;

	/// Executes one invocation from its single linear argument/event stream.
	fn call<'c>(
		&'c self,
		params: IncomingParams<'c>,
	) -> impl Stream<Item = Ev<Self::Update, Self::Payload, Self::Fault>> + Send + 'c;

	/// Deterministically projects either durable tool branch for one model.
	fn prompt(&self, view: Result<&Self::Payload, &Self::Fault>, caps: &PromptCaps) -> Vec<Part>;

	/// Deterministically migrates one historical call toward this revision.
	fn lift(&self, _from: &Rev, _call: RecordedCall<'_>) -> Option<LiftedCall> {
		None
	}
}

/// One event emitted by a typed tool invocation.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum Ev<U, P, F> {
	/// Ephemeral progress, never transcript history.
	Update(U),
	/// Terminal structured failure of a parameter the tool pulled.
	Args(ArgIssue),
	/// Terminal structured cancellation or effect-uncertainty report.
	Aborted(Abort),
	/// Terminal outcome; supervisors fuse the stream after this event.
	Done(Outcome<P, F>),
}

/// Terminal executor outcome before journal verdict lowering.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Outcome<P, F> {
	/// A synchronous success or typed fault.
	Done {
		/// Tool-owned durable branch.
		result:  Result<P, F>,
		/// Whether model-facing parts may be compacted while truth survives.
		useless: bool,
	},
	/// Work continues outside the turn and will settle through the job board.
	Detached(JobRef),
}

/// Journaled truth for every completed call branch.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum Verdict<P, F> {
	/// Successful durable payload.
	Ok(P),
	/// Tool-owned durable fault.
	Fault(F),
	/// Structured failure of a parameter the tool actually pulled.
	Args(ArgIssue),
	/// Structured cancellation/skip/effect-uncertainty report.
	Aborted(Abort),
}

/// One segment in a pulled JSON path.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum ArgPath {
	/// Object key.
	Key(Str),
	/// Array index.
	Index(u64),
}

/// Stable class of parameter pull failure.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ArgIssueKind {
	/// Required pulled value was absent.
	Missing,
	/// Input ended before the pulled value completed.
	Incomplete,
	/// Input was explicitly or implicitly abandoned.
	Aborted,
	/// Complete input was malformed.
	Malformed,
	/// Pulled value had another JSON shape.
	TypeMismatch,
	/// Invocation framing violated the linear stream contract.
	Protocol,
}

/// Structured issue for one parameter the tool pulled.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ArgIssue {
	/// Full pulled key/index path.
	pub path:     Vec<ArgPath>,
	/// Requested shape.
	pub expected: Str,
	/// Stable failure class.
	pub kind:     ArgIssueKind,
	/// Optional valid example for model repair.
	pub example:  Option<Str>,
	/// Observed shape for [`ArgIssueKind::TypeMismatch`].
	pub found:    Option<Str>,
}

/// Structured reason an invocation did not produce a normal verdict.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Abort {
	/// Call was deliberately not started.
	Skipped {
		/// Explanation of why invocation execution was bypassed.
		reason: Str,
	},
	/// Owner observed interruption before effects could land.
	Interrupted {
		/// Explanation of the interruption event or signal.
		reason: Str,
	},
	/// Cancellation raced an effect and only the owner can report uncertainty.
	EffectsUnknown {
		/// Explanation of why side-effect state cannot be confirmed.
		reason: Str,
	},
	/// Invocation feed disappeared before explicit commitment.
	InputDropped,
	/// Executor stream ended without a terminal event.
	MissingOutcome,
}

/// Retention promise for an artifact produced by detached work.
///
/// This is a lifetime hint for artifact storage, not ownership of an
/// environment resource. Producers may retain an artifact longer than promised.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactLifetime {
	/// Retain only long enough to consume the settlement.
	Ephemeral,
	/// Retain for the current agent session.
	#[default]
	Session,
	/// Retain independently of the current agent session.
	Durable,
}

/// Detached work and its expected artifact.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct JobRef {
	/// Stable environment job identifier.
	pub id:       Str,
	/// Artifact expected when the job settles.
	pub artifact: ExpectedArtifact,
}

/// Expected output of a detached job.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ExpectedArtifact {
	/// Human-readable artifact role.
	pub description: Str,
	/// Expected MIME type, when known.
	pub media_type:  Option<Str>,
	/// Minimum retention promised by the artifact producer.
	pub lifetime:    ArtifactLifetime,
}

/// Borrowed durable call supplied to a pure revision lift.
#[derive(Clone, Copy, Debug)]
pub struct RecordedCall<'a> {
	/// Exact original model-emitted argument bytes.
	pub raw_args: &'a [u8],
	/// Exact structured verdict JSON bytes.
	pub verdict:  &'a [u8],
}

/// Owned result of one successful pure revision lift.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct LiftedCall {
	/// Arguments expressed in the target revision.
	pub raw_args: Bytes,
	/// Verdict expressed in the target revision.
	pub verdict:  Bytes,
}

/// Owned historical call retained when projecting a transcript.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RecordedCallOwned {
	/// Durable tool identity at recording time.
	pub identity: ToolIdentity,
	/// Exact original arguments.
	pub raw_args: Bytes,
	/// Exact original structured verdict.
	pub verdict:  Bytes,
}

impl RecordedCallOwned {
	/// Borrows the byte-stable lift input.
	pub fn as_recorded(&self) -> RecordedCall<'_> {
		RecordedCall { raw_args: &self.raw_args, verdict: &self.verdict }
	}
}

/// Serialized verdict details before or after blob spill.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "storage", rename_all = "snake_case")]
pub enum VerdictDetails {
	/// Small verdict retained inline as structured JSON bytes.
	Inline {
		/// Complete serialized verdict JSON bytes.
		json: Bytes,
	},
	/// Large verdict retained by content-addressed blob reference.
	Spilled {
		/// Durable blob reference.
		blob:     BlobRef,
		/// Original serialized byte length.
		byte_len: u64,
	},
}

/// Environment-provided hook for durable large-verdict storage.
pub trait VerdictSpill: Send + Sync {
	/// Storage error.
	type Error;

	/// Stores exact JSON bytes and returns their durable blob reference.
	fn spill(&self, json: Bytes) -> impl Future<Output = Result<BlobRef, Self::Error>> + Send + '_;
}

/// Failure while serializing or spilling a structured verdict.
#[derive(Debug, Error)]
pub enum VerdictDetailsError<E> {
	/// Structured verdict serialization failed.
	#[error("verdict serialization failed: {0}")]
	Serialize(#[from] serde_json::Error),
	/// Blob storage failed.
	#[error("verdict spill failed")]
	Spill(E),
}

/// Serializes a verdict deterministically and spills it above `inline_limit`.
pub async fn verdict_details<P, F, S>(
	verdict: &Verdict<P, F>,
	inline_limit: usize,
	spill: &S,
) -> Result<VerdictDetails, VerdictDetailsError<S::Error>>
where
	P: Serialize,
	F: Serialize,
	S: VerdictSpill,
{
	let json = Bytes::from(serde_json::to_vec(verdict)?);
	if json.len() <= inline_limit {
		return Ok(VerdictDetails::Inline { json });
	}
	let byte_len = json.len() as u64;
	let blob = spill
		.spill(json)
		.await
		.map_err(VerdictDetailsError::Spill)?;
	Ok(VerdictDetails::Spilled { blob, byte_len })
}
