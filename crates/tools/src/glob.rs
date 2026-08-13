//! Deterministic workspace path matching with structured truncation truth.

use std::fmt;

use async_stream::stream;
use bytes::Bytes;
use futures::{FutureExt, Stream, pin_mut, select_biased};
use omp_core::Str;
use omp_tool::{
	Abort, ArgIssue, ArgIssueKind, CommitError, Constraint, Ev, IncomingParams, InterruptWaitError,
	Outcome, ParamError, Part, PromptCaps, Rev, Tool, ToolSpec,
};
use serde::{Deserialize, Serialize};

use crate::{grep::WorkspaceSearch, render::TextProjection};

const SCHEMA: &[u8] = br#"{
  "type":"object",
  "additionalProperties":false,
  "properties":{
    "path":{"type":"string","default":".","description":"Workspace-relative traversal root."},
    "patterns":{"type":"array","items":{"type":"string","minLength":1},"minItems":1,"description":"Path globs matched with separator-aware workspace glob semantics."},
    "exclude":{"type":"array","items":{"type":"string"},"default":[],"description":"Path globs excluded from results and traversal."},
    "gitignore":{"type":"boolean","default":true,"description":"Honor gitignore and ignore files."},
    "hidden":{"type":"boolean","default":false,"description":"Include dot-prefixed paths."},
    "limit":{"type":"integer","minimum":0,"maximum":18446744073709551615,"description":"Hard maximum number of returned paths."}
  },
  "required":["patterns","limit"]
}"#;

fn default_path() -> Str {
	Str::from(".")
}

const fn default_true() -> bool {
	true
}

/// Model arguments for `glob@1`.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Params {
	/// Workspace-relative traversal root.
	#[serde(default = "default_path")]
	pub path:      Str,
	/// Separator-aware path glob patterns to include.
	pub patterns:  Vec<Str>,
	/// Path glob patterns to exclude.
	#[serde(default)]
	pub exclude:   Vec<Str>,
	/// Whether ignore files are honored.
	#[serde(default = "default_true")]
	pub gitignore: bool,
	/// Whether dot-prefixed paths are traversed.
	#[serde(default)]
	pub hidden:    bool,
	/// Hard maximum number of returned paths.
	pub limit:     u64,
}

/// Fully specified request passed to the workspace resource after commitment.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WalkRequest {
	/// Workspace-relative traversal root.
	pub path:      Str,
	/// Separator-aware path glob patterns to include.
	pub patterns:  Vec<Str>,
	/// Path glob patterns to exclude.
	pub exclude:   Vec<Str>,
	/// Whether ignore files are honored.
	pub gitignore: bool,
	/// Whether dot-prefixed paths are traversed.
	pub hidden:    bool,
	/// Maximum paths requested, including one lookahead used to prove truncation.
	pub limit:     u64,
}

/// Structured resource result before the tool applies its defensive hard cap.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WalkResult {
	/// Deterministically ordered workspace-relative paths.
	pub paths:     Vec<Str>,
	/// Whether additional matching paths exist beyond those returned.
	pub truncated: bool,
}

/// Durable successful `glob@1` result.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Payload {
	/// Deterministically ordered workspace-relative paths.
	pub paths:     Vec<Str>,
	/// Whether results were omitted by either the resource or the hard result
	/// limit.
	pub truncated: bool,
}

/// Ephemeral progress from `glob@1`; path traversal has no durable updates.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum Update {}

/// Durable typed `glob@1` failure.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Fault {
	/// A glob pattern could not be compiled by the workspace walker.
	InvalidPattern {
		/// Exact rejected pattern.
		pattern: Str,
		/// Resource-owned parser explanation.
		message: Str,
	},
	/// The workspace owner rejected or failed the request.
	Workspace {
		/// Stable resource-owned explanation.
		message: Str,
	},
	/// The resource itself observed cancellation without an invocation
	/// interrupt.
	Cancelled {
		/// Stable resource-owned cancellation reason.
		reason: Str,
	},
}

impl fmt::Display for Fault {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::InvalidPattern { pattern, message } => {
				write!(formatter, "invalid glob pattern {pattern}: {message}")
			},
			Self::Workspace { message } => write!(formatter, "workspace walk failed: {message}"),
			Self::Cancelled { reason } => {
				write!(formatter, "workspace walk was cancelled: {reason}")
			},
		}
	}
}

impl std::error::Error for Fault {}

/// Generic `glob@1` executor over an environment-owned workspace resource.
pub struct Glob<W> {
	workspace: W,
	spec:      ToolSpec,
}

/// Constructs the `glob@1` tool over `workspace`.
pub fn tool<W: WorkspaceSearch>(workspace: W) -> Glob<W> {
	Glob {
		workspace,
		spec: ToolSpec {
			name:        Str::from("glob"),
			rev:         Rev { family: Str::new(""), n: 1 },
			description: Str::from(
				"Match deterministic workspace-relative paths with include and exclude globs",
			),
			schema:      Bytes::from_static(SCHEMA),
			constraint:  Constraint::Schema { priority: 100 },
		},
	}
}

impl<W: WorkspaceSearch> Tool for Glob<W> {
	type Fault = Fault;
	type Params = Params;
	type Payload = Payload;
	type Update = Update;

	fn spec(&self) -> &ToolSpec {
		&self.spec
	}

	fn call<'c>(
		&'c self,
		mut params: IncomingParams<'c>,
	) -> impl Stream<Item = Ev<Self::Update, Self::Payload, Self::Fault>> + Send + 'c {
		stream! {
			let arguments = match params.whole::<Params>().await {
				Ok(arguments) => arguments,
				Err(error) => {
					yield param_event(error);
					return;
				},
			};
			match params.interruptable().committed().await {
				Ok(_) => {},
				Err(error) => {
					yield commit_event(error);
					return;
				},
			}

			let limit = arguments.limit;
			let request = WalkRequest {
				path: arguments.path,
				patterns: arguments.patterns,
				exclude: arguments.exclude,
				gitignore: arguments.gitignore,
				hidden: arguments.hidden,
				limit: limit.saturating_add(1),
			};
			let operation = self.workspace.glob(request).fuse();
			let interruption = params.next_interrupt().fuse();
			pin_mut!(operation, interruption);
			select_biased! {
				result = operation => yield done(result.map(|result| payload(result, limit))),
				interrupt = interruption => {
					yield interrupt_event(interrupt, "glob traversal owner disappeared");
				},
			}
		}
	}

	fn prompt(&self, view: Result<&Self::Payload, &Self::Fault>, caps: &PromptCaps) -> Vec<Part> {
		let Some(mut projection) = TextProjection::new(caps) else {
			return Vec::new();
		};
		match view {
			Ok(payload) => {
				if payload.paths.is_empty() && !projection.push("No paths matched.\n") {
					return projection.finish();
				}
				for path in &payload.paths {
					let record = format!("{path}\n");
					if !projection.push(&record) {
						return projection.finish();
					}
				}
				if payload.truncated {
					projection.push("Results truncated.\n");
				}
			},
			Err(fault) => {
				projection.push(&format!("glob fault: {fault}\n"));
			},
		}
		projection.finish()
	}
}

fn payload(mut result: WalkResult, limit: u64) -> Payload {
	result.paths.sort_unstable();
	let available = u64::try_from(result.paths.len()).unwrap_or(u64::MAX);
	let truncated = result.truncated || available > limit;
	let retain = usize::try_from(limit)
		.unwrap_or(usize::MAX)
		.min(result.paths.len());
	result.paths.truncate(retain);
	Payload { paths: result.paths, truncated }
}

fn done(result: Result<Payload, Fault>) -> Ev<Update, Payload, Fault> {
	Ev::Done(Outcome::Done { result, useless: false })
}

fn param_event(error: ParamError) -> Ev<Update, Payload, Fault> {
	match error {
		ParamError::Args(issue) if issue.kind == ArgIssueKind::Aborted => {
			Ev::Aborted(Abort::InputDropped)
		},
		ParamError::Args(issue) => Ev::Args(issue),
		ParamError::Interrupted(interrupt) => {
			Ev::Aborted(Abort::Interrupted { reason: interrupt.reason })
		},
		ParamError::Protocol(message) => Ev::Args(protocol_issue(message)),
	}
}

fn commit_event(error: CommitError) -> Ev<Update, Payload, Fault> {
	match error {
		CommitError::Aborted => Ev::Aborted(Abort::InputDropped),
		CommitError::Interrupted(interrupt) => {
			Ev::Aborted(Abort::Interrupted { reason: interrupt.reason })
		},
		CommitError::Protocol(message) => Ev::Args(protocol_issue(message)),
	}
}

fn interrupt_event(
	interrupt: Result<omp_tool::Interrupt, InterruptWaitError>,
	closed_reason: &'static str,
) -> Ev<Update, Payload, Fault> {
	match interrupt {
		Ok(interrupt) => Ev::Aborted(Abort::Interrupted { reason: interrupt.reason }),
		Err(InterruptWaitError::Closed) => {
			Ev::Aborted(Abort::Interrupted { reason: Str::from(closed_reason) })
		},
		Err(InterruptWaitError::Protocol(message)) => Ev::Args(protocol_issue(message)),
	}
}

fn protocol_issue(message: Str) -> ArgIssue {
	ArgIssue {
		path:     Vec::new(),
		expected: Str::from("one committed JSON argument object"),
		kind:     ArgIssueKind::Protocol,
		example:  Some(Str::from("{\"patterns\":[\"src/**/*.rs\"],\"limit\":100}")),
		found:    Some(message),
	}
}
