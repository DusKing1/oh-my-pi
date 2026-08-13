//! Exact workspace-content search with structured durable matches.

use std::{
	fmt::{self, Write as _},
	future::Future,
};

use async_stream::stream;
use bytes::Bytes;
use futures::{FutureExt, Stream, pin_mut, select_biased};
use omp_core::Str;
use omp_tool::{
	Abort, ArgIssue, ArgIssueKind, CommitError, Constraint, Ev, IncomingParams, InterruptWaitError,
	Outcome, ParamError, Part, PromptCaps, Rev, Tool, ToolSpec,
};
use serde::{Deserialize, Serialize};

use crate::render::TextProjection;

const SCHEMA: &[u8] = br#"{
  "type":"object",
  "additionalProperties":false,
  "properties":{
    "path":{"type":"string","default":".","description":"Workspace-relative traversal root."},
    "patterns":{"type":"array","items":{"type":"string","minLength":1},"minItems":1,"description":"Exact text strings to find."},
    "include":{"type":"array","items":{"type":"string"},"default":[],"description":"Path globs which candidates must match."},
    "exclude":{"type":"array","items":{"type":"string"},"default":[],"description":"Path globs excluded from candidate discovery."},
    "gitignore":{"type":"boolean","default":true,"description":"Honor gitignore and ignore files."},
    "hidden":{"type":"boolean","default":false,"description":"Include dot-prefixed paths."},
    "case_sensitive":{"type":"boolean","default":true,"description":"Match exact letter case; false is unsupported by the current workspace host."},
    "mode":{"type":"string","enum":["fixed","regex"],"default":"fixed","description":"Pattern interpretation; regex is recognized but unsupported."},
    "limit":{"type":"integer","minimum":0,"maximum":18446744073709551615,"description":"Hard maximum number of returned match records."}
  },
  "required":["patterns","limit"]
}"#;

fn default_path() -> Str {
	Str::from(".")
}

const fn default_true() -> bool {
	true
}

/// Interpretation requested for [`Params::patterns`].
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PatternMode {
	/// Match each pattern as exact UTF-8 bytes.
	#[default]
	Fixed,
	/// Regular expressions, recognized so unsupported semantics fail explicitly.
	Regex,
}

/// Model arguments for `grep@1`.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Params {
	/// Workspace-relative traversal root.
	#[serde(default = "default_path")]
	pub path:           Str,
	/// Exact text patterns to find.
	pub patterns:       Vec<Str>,
	/// Candidate path globs to include.
	#[serde(default)]
	pub include:        Vec<Str>,
	/// Candidate path globs to exclude.
	#[serde(default)]
	pub exclude:        Vec<Str>,
	/// Whether ignore files are honored.
	#[serde(default = "default_true")]
	pub gitignore:      bool,
	/// Whether dot-prefixed paths are traversed.
	#[serde(default)]
	pub hidden:         bool,
	/// Whether matching is case-sensitive.
	#[serde(default = "default_true")]
	pub case_sensitive: bool,
	/// Pattern interpretation.
	#[serde(default)]
	pub mode:           PatternMode,
	/// Hard maximum number of returned match records.
	pub limit:          u64,
}

/// Fully specified request passed to the workspace resource after commitment.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SearchRequest {
	/// Workspace-relative traversal root.
	pub path:      Str,
	/// Exact UTF-8 byte patterns.
	pub patterns:  Vec<Str>,
	/// Candidate path globs to include.
	pub include:   Vec<Str>,
	/// Candidate path globs to exclude.
	pub exclude:   Vec<Str>,
	/// Whether ignore files are honored.
	pub gitignore: bool,
	/// Whether dot-prefixed paths are traversed.
	pub hidden:    bool,
	/// Maximum records requested, including one lookahead used to prove truncation.
	pub limit:     u64,
}

/// Half-open byte range within a retained UTF-8 line.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct ByteSpan {
	/// Zero-based inclusive byte offset in [`SearchMatch::line_text`].
	pub start: u64,
	/// Zero-based exclusive byte offset in [`SearchMatch::line_text`].
	pub end:   u64,
}

/// One source line containing one or more exact matches.
#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct SearchMatch {
	/// Deterministic workspace-relative path using `/` separators.
	pub path:      Str,
	/// One-based source line number.
	pub line:      u64,
	/// Exact byte spans within `line_text`.
	pub spans:     Vec<ByteSpan>,
	/// Complete UTF-8 line text, retained without slicing around matches.
	pub line_text: Str,
}

/// Structured resource result before the tool applies its defensive hard cap.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SearchResult {
	/// Deterministically ordered source-line matches.
	pub matches:        Vec<SearchMatch>,
	/// Workspace-relative binary candidates deliberately skipped by the
	/// resource.
	pub binary_skipped: Vec<Str>,
	/// Whether additional matches exist beyond those returned.
	pub truncated:      bool,
}

/// Durable successful `grep@1` result.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Payload {
	/// Deterministically ordered source-line matches.
	pub matches:        Vec<SearchMatch>,
	/// Workspace-relative binary candidates deliberately skipped by the
	/// resource.
	pub binary_skipped: Vec<Str>,
	/// Whether results were omitted by either the resource or the hard result
	/// limit.
	pub truncated:      bool,
}

/// Ephemeral progress from `grep@1`; the current exact search has no updates.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum Update {}

/// Durable typed `grep@1` failure.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Fault {
	/// The selected pattern mode is recognized but unsupported by WorkspaceHost.
	UnsupportedPatternMode {
		/// Unsupported requested mode.
		mode: PatternMode,
	},
	/// Case-insensitive matching is not implemented by WorkspaceHost.
	CaseInsensitiveUnsupported,
	/// One exact pattern was empty.
	EmptyPattern {
		/// Zero-based position in the submitted pattern list.
		index: u64,
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
			Self::UnsupportedPatternMode { mode } => {
				write!(formatter, "pattern mode {mode:?} is unsupported")
			},
			Self::CaseInsensitiveUnsupported => {
				formatter.write_str("case-insensitive search is unsupported")
			},
			Self::EmptyPattern { index } => {
				write!(formatter, "search pattern at index {index} is empty")
			},
			Self::Workspace { message } => write!(formatter, "workspace search failed: {message}"),
			Self::Cancelled { reason } => {
				write!(formatter, "workspace search was cancelled: {reason}")
			},
		}
	}
}

impl std::error::Error for Fault {}

/// Zero-box workspace traversal boundary shared by `grep@1` and `glob@1`.
///
/// Implementations must return workspace-relative paths in deterministic order.
/// Dropping either returned future must structurally cancel its active
/// traversal.
pub trait WorkspaceSearch: Send + Sync + 'static {
	/// Search exact text in deterministic workspace candidates.
	fn search(
		&self,
		request: SearchRequest,
	) -> impl Future<Output = Result<SearchResult, Fault>> + Send + '_;

	/// Match paths in deterministic workspace traversal order.
	fn glob(
		&self,
		request: crate::glob::WalkRequest,
	) -> impl Future<Output = Result<crate::glob::WalkResult, crate::glob::Fault>> + Send + '_;
}

/// Generic `grep@1` executor over an environment-owned workspace resource.
pub struct Grep<W> {
	workspace: W,
	spec:      ToolSpec,
}

/// Constructs the `grep@1` tool over `workspace`.
pub fn tool<W: WorkspaceSearch>(workspace: W) -> Grep<W> {
	Grep {
		workspace,
		spec: ToolSpec {
			name:        Str::from("grep"),
			rev:         Rev { family: Str::new(""), n: 1 },
			description: Str::from(
				"Search deterministic workspace candidates for exact text and return line byte spans",
			),
			schema:      Bytes::from_static(SCHEMA),
			constraint:  Constraint::Schema { priority: 100 },
		},
	}
}

impl<W: WorkspaceSearch> Tool for Grep<W> {
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

			if arguments.mode != PatternMode::Fixed {
				yield done(Err(Fault::UnsupportedPatternMode { mode: arguments.mode }));
				return;
			}
			if !arguments.case_sensitive {
				yield done(Err(Fault::CaseInsensitiveUnsupported));
				return;
			}
			if let Some(index) = arguments.patterns.iter().position(Str::is_empty) {
				yield done(Err(Fault::EmptyPattern {
					index: u64::try_from(index).unwrap_or(u64::MAX),
				}));
				return;
			}

			let limit = arguments.limit;
			let request = SearchRequest {
				path: arguments.path,
				patterns: arguments.patterns,
				include: arguments.include,
				exclude: arguments.exclude,
				gitignore: arguments.gitignore,
				hidden: arguments.hidden,
				limit: limit.saturating_add(1),
			};
			let operation = self.workspace.search(request).fuse();
			let interruption = params.next_interrupt().fuse();
			pin_mut!(operation, interruption);
			select_biased! {
				result = operation => yield done(result.map(|result| payload(result, limit))),
				interrupt = interruption => {
					yield interrupt_event(interrupt, "grep traversal owner disappeared");
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
				if payload.matches.is_empty() && !projection.push("No matches.\n") {
					return projection.finish();
				}
				for matched in &payload.matches {
					let mut record = format!("{}:{}:", matched.path, matched.line);
					for (index, span) in matched.spans.iter().enumerate() {
						if index != 0 {
							record.push(',');
						}
						let _ = write!(record, "{}-{}", span.start, span.end);
					}
					record.push_str(": ");
					record.push_str(&matched.line_text);
					record.push('\n');
					if !projection.push(&record) {
						return projection.finish();
					}
				}
				for path in &payload.binary_skipped {
					let record = format!("binary skipped: {path}\n");
					if !projection.push(&record) {
						return projection.finish();
					}
				}
				if payload.truncated {
					projection.push("Results truncated.\n");
				}
			},
			Err(fault) => {
				projection.push(&format!("grep fault: {fault}\n"));
			},
		}
		projection.finish()
	}
}

fn payload(mut result: SearchResult, limit: u64) -> Payload {
	for matched in &mut result.matches {
		matched.spans.sort_unstable();
	}
	result.matches.sort_unstable();
	result.binary_skipped.sort_unstable();
	let available = u64::try_from(result.matches.len()).unwrap_or(u64::MAX);
	let truncated = result.truncated || available > limit;
	let retain = usize::try_from(limit)
		.unwrap_or(usize::MAX)
		.min(result.matches.len());
	result.matches.truncate(retain);
	Payload { matches: result.matches, binary_skipped: result.binary_skipped, truncated }
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
		example:  Some(Str::from("{\"patterns\":[\"needle\"],\"limit\":100}")),
		found:    Some(message),
	}
}
