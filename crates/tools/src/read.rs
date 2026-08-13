//! Revision-pinned document reads.

use async_stream::stream;
use bytes::Bytes;
use futures::{Future, FutureExt, Stream, pin_mut, select_biased};
use omp_core::Str;
use omp_tool::{
	Abort, ArgIssue, ArgIssueKind, BlobRef, CommitError, Constraint, Ev, IncomingParams, Outcome,
	ParamError, Part, PromptCaps, Rev, Tool, ToolSpec,
};
use serde::{Deserialize, Serialize};

use crate::render::TextProjection;

const SCHEMA: &[u8] = br#"{"type":"object","additionalProperties":false,"required":["path"],"properties":{"path":{"type":"string","minLength":1},"ranges":{"type":"array","items":{"type":"object","additionalProperties":false,"required":["start","end"],"properties":{"start":{"type":"integer","minimum":1},"end":{"type":"integer","minimum":1}}}},"structural":{"type":"boolean","default":false}}}"#;

/// One inclusive, one-based line selection.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct LineRange {
	/// One-based starting line number (inclusive).
	pub start: u64,
	/// One-based ending line number (inclusive).
	pub end:   u64,
}

/// Arguments accepted by `read@1`.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Params {
	/// Workspace-relative document path.
	pub path:       Str,
	/// Requested line ranges.
	#[serde(default)]
	pub ranges:     Vec<LineRange>,
	/// Whether to produce a structural summary instead of line slices.
	#[serde(default)]
	pub structural: bool,
}

/// Ephemeral read progress (reserved without exposing host details).
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Update {
	/// Progress phase description.
	pub phase: Str,
}

/// Classification pinned when the document lease opens.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DocumentKind {
	/// Plain UTF-8 text document.
	Text,
	/// Non-text binary media document.
	Binary {
		/// MIME type of the binary payload.
		media_type: Str,
		/// Model-visible fallback description.
		fallback:   Str,
	},
}

/// A numbered text slice returned by the document owner.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TextSlice {
	/// Starting line number for this slice.
	pub start_line: u64,
	/// Verbatim text lines.
	pub text:       Str,
}

/// Structural-summary recovery metadata.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SummarySegment {
	/// Starting line number of the segment.
	pub start_line: u64,
	/// Ending line number of the segment.
	pub end_line:   u64,
	/// Structural declaration or summary text.
	pub text:       Str,
	/// Whether code inside this segment was elided.
	pub elided:     bool,
}

/// Exact result of reading the pinned lease.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LeaseContent {
	/// Selected line slices and elided ranges.
	Text {
		/// Selected text slices.
		slices: Vec<TextSlice>,
		/// Elided line ranges.
		elided: Vec<LineRange>,
	},
	/// Structural summary segments and elided ranges.
	Summary {
		/// Summary segments.
		segments: Vec<SummarySegment>,
		/// Elided line ranges.
		elided:   Vec<LineRange>,
	},
	/// Raw binary payload bytes.
	Binary(Bytes),
}

/// Durable, dialect-neutral read truth.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Payload {
	/// Workspace-relative document path.
	pub path:       Str,
	/// Pinned document revision string.
	pub revision:   Str,
	/// Originally requested line ranges.
	pub ranges:     Vec<LineRange>,
	/// Whether structural summary mode was requested.
	pub structural: bool,
	/// Line ranges elided from the output.
	pub elided:     Vec<LineRange>,
	/// Read content payload.
	pub content:    Content,
}

/// Durable read content; binary bytes are represented only by their blob.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Content {
	/// Selected text slices.
	Text {
		/// Selected text slices.
		slices: Vec<TextSlice>,
	},
	/// Structural summary segments.
	Summary {
		/// Summary segments.
		segments: Vec<SummarySegment>,
	},
	/// Blob-backed binary media.
	Blob {
		/// Durable blob reference.
		blob:     BlobRef,
		/// Model-facing fallback text.
		fallback: Str,
	},
}

/// Typed document/blob failure without leaking host error types.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Fault {
	/// Document lease opening failed.
	Open {
		/// Failure message.
		message: Str,
	},
	/// Document content reading failed.
	Read {
		/// Failure message.
		message: Str,
	},
	/// Binary blob storage failed.
	Blob {
		/// Failure message.
		message: Str,
	},
	/// Document kind mismatch.
	KindMismatch {
		/// Failure message.
		message: Str,
	},
}

/// An opaque revision-pinned document lease.
pub trait ReadLease: Send + Sync {
	/// Returns the pinned revision string.
	fn revision(&self) -> &Str;
	/// Returns the document classification.
	fn kind(&self) -> &DocumentKind;
	/// Reads the requested line ranges or structural summary.
	fn read<'a>(
		&'a self,
		ranges: &'a [LineRange],
		structural: bool,
	) -> impl Future<Output = Result<LeaseContent, Fault>> + Send + 'a;
}

/// Opens revision-pinned document leases.
pub trait ReadDocuments: Send + Sync + 'static {
	/// Pinned lease type.
	type Lease: ReadLease;
	/// Opens a revision-pinned document lease.
	fn open(&self, path: Str) -> impl Future<Output = Result<Self::Lease, Fault>> + Send + '_;
}

/// Stores non-text bytes in the durable environment blob namespace.
pub trait ReadBlobs: Send + Sync + 'static {
	/// Stores binary bytes and returns a durable blob reference.
	fn store(
		&self,
		bytes: Bytes,
		media_type: Str,
	) -> impl Future<Output = Result<BlobRef, Fault>> + Send + '_;
}

/// `read@1` executor over document and blob resource adapters.
pub struct ReadTool<D, B> {
	documents: D,
	blobs:     B,
	spec:      ToolSpec,
}

/// Constructs the revision-pinned `read@1` tool.
pub fn tool<D: ReadDocuments, B: ReadBlobs>(documents: D, blobs: B) -> ReadTool<D, B> {
	ReadTool {
		documents,
		blobs,
		spec: ToolSpec {
			name:        Str::new_static("read"),
			rev:         Rev { family: Str::new_static(""), n: 1 },
			description: Str::new_static(
				"Read line ranges, whole documents, structural summaries, or binary media from a \
				 revision-pinned path.",
			),
			schema:      Bytes::from_static(SCHEMA),
			constraint:  Constraint::Schema { priority: 10 },
		},
	}
}

impl<D: ReadDocuments, B: ReadBlobs> Tool for ReadTool<D, B> {
	type Fault = Fault;
	type Params = Params;
	type Payload = Payload;
	type Update = Update;

	fn spec(&self) -> &ToolSpec {
		&self.spec
	}

	fn call<'c>(
		&'c self,
		mut incoming: IncomingParams<'c>,
	) -> impl Stream<Item = Ev<Update, Payload, Fault>> + Send + 'c {
		stream! {
			// The path is the first completed subtree. Opening is deliberately inside
			// this sole cursor session, before waiting for optional siblings.
			let pulled = incoming.pull(|mut doc| async move {
				let mut root = doc.json().object();
				let path = root.key("path").string().finish().await?;
				let open = self.documents.open(path.clone());
				let collect = root.collect();
				let (lease, object) = futures::join!(open, collect);
				Ok((path, lease, object?.to_string()))
			}).await;
			let (path, lease, raw) = match pulled {
				Ok(value) => value,
				Err(ParamError::Args(issue)) if issue.kind == ArgIssueKind::Aborted => { yield Ev::Aborted(Abort::InputDropped); return; },
				Err(ParamError::Args(issue)) => { yield Ev::Args(issue); return; },
				Err(ParamError::Interrupted(interrupt)) => { yield Ev::Aborted(Abort::Interrupted { reason: interrupt.reason }); return; },
				Err(ParamError::Protocol(reason)) => { yield Ev::Args(protocol_issue(reason)); return; },
			};
			let decoded: Params = match serde_json::from_str(&raw) {
				Ok(value) => value,
				Err(_) => { yield Ev::Args(args_issue()); return; },
			};
			let lease = match lease { Ok(value) => value, Err(fault) => { yield done(Err(fault)); return; } };
			match incoming.committed().await {
				Ok(_) => {},
				Err(CommitError::Aborted) => { yield Ev::Aborted(Abort::InputDropped); return; },
				Err(CommitError::Interrupted(interrupt)) => { yield Ev::Aborted(Abort::Interrupted { reason: interrupt.reason }); return; },
				Err(CommitError::Protocol(reason)) => { yield Ev::Args(protocol_issue(reason)); return; },
			}
			let waited: Result<Result<LeaseContent, Fault>, Str> = {
				let work = lease.read(&decoded.ranges, decoded.structural).fuse();
				let cancel = incoming.next_interrupt().fuse();
				pin_mut!(work, cancel);
				select_biased! {
					value = work => Ok(value),
					interrupt = cancel => Err(interrupt.map(|v| v.reason).unwrap_or_else(|_| Str::new_static("invocation owner dropped"))),
				}
			};
			let result = match waited {
				Ok(value) => value,
				Err(reason) => { yield Ev::Aborted(Abort::Interrupted { reason }); return; },
			};
			let read = match result { Ok(value) => value, Err(fault) => { yield done(Err(fault)); return; } };
			let revision = lease.revision().clone();
			let kind = lease.kind().clone();
			let (content, elided) = match (kind, read) {
				(DocumentKind::Text, LeaseContent::Text { slices, elided }) => (Content::Text { slices }, elided),
				(DocumentKind::Text, LeaseContent::Summary { segments, elided }) => (Content::Summary { segments }, elided),
				(DocumentKind::Binary { media_type, fallback }, LeaseContent::Binary(bytes)) => {
					let waited: Result<Result<BlobRef, Fault>, Str> = {
						let store = self.blobs.store(bytes, media_type).fuse();
						let cancel = incoming.next_interrupt().fuse();
						pin_mut!(store, cancel);
						select_biased! {
							value = store => Ok(value),
							interrupt = cancel => Err(interrupt.map(|v| v.reason).unwrap_or_else(|_| Str::new_static("invocation owner dropped"))),
						}
					};
					let blob = match waited {
						Ok(Ok(blob)) => blob,
						Ok(Err(fault)) => { yield done(Err(fault)); return; },
						Err(reason) => { yield Ev::Aborted(Abort::Interrupted { reason }); return; },
					};
					(Content::Blob { blob, fallback }, Vec::new())
				},
				_ => { yield done(Err(Fault::KindMismatch { message: Str::new_static("document classification changed within pinned lease") })); return; },
			};
			yield done(Ok(Payload { path, revision, ranges: decoded.ranges, structural: decoded.structural, elided, content }));
		}
	}

	fn prompt(&self, view: Result<&Payload, &Fault>, caps: &PromptCaps) -> Vec<Part> {
		let Ok(payload) = view else {
			let mut out = match TextProjection::new(caps) {
				Some(value) => value,
				None => return Vec::new(),
			};
			out.push(&format!("read failed: {:?}", view.unwrap_err()));
			return out.finish();
		};
		if let Content::Blob { blob, fallback } = &payload.content {
			if caps.media && caps.maximum_parts != 0 {
				return vec![Part::Blob { blob: blob.clone(), alt: Some(fallback.clone()) }];
			}
			let mut out = match TextProjection::new(caps) {
				Some(value) => value,
				None => return Vec::new(),
			};
			out.push(fallback);
			return out.finish();
		}
		let mut out = match TextProjection::new(caps) {
			Some(value) => value,
			None => return Vec::new(),
		};
		match &payload.content {
			Content::Text { slices } => {
				for slice in slices {
					for (offset, line) in slice.text.lines().enumerate() {
						if !out.push(&format!("{}:{}\n", slice.start_line + offset as u64, line)) {
							break;
						}
					}
				}
			},
			Content::Summary { segments } => {
				for segment in segments {
					let marker = if segment.elided { "elided" } else { "kept" };
					if !out.push(&format!(
						"[{}-{} {}]\n{}\n",
						segment.start_line, segment.end_line, marker, segment.text
					)) {
						break;
					}
				}
			},
			Content::Blob { .. } => unreachable!(),
		}
		out.finish()
	}
}

fn done(result: Result<Payload, Fault>) -> Ev<Update, Payload, Fault> {
	Ev::Done(Outcome::Done { result, useless: false })
}
fn args_issue() -> ArgIssue {
	ArgIssue {
		path:     Vec::new(),
		expected: Str::new_static("read@1 arguments"),
		kind:     ArgIssueKind::Malformed,
		example:  None,
		found:    None,
	}
}
fn protocol_issue(reason: Str) -> ArgIssue {
	ArgIssue {
		path:     Vec::new(),
		expected: Str::new_static("linear invocation frames"),
		kind:     ArgIssueKind::Protocol,
		example:  Some(reason),
		found:    None,
	}
}
