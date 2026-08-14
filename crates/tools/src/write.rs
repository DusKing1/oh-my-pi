//! Pi-compatible whole-file writes over the session document host.

use std::{fmt, future::Future};

use async_stream::stream;
use bytes::Bytes;
use futures::{FutureExt as _, Stream, pin_mut, select_biased};
use omp_core::Str;
use omp_hashline::format_hashline_header;
use omp_tool::{
	Abort, ArgIssue, ArgIssueKind, CommitError, Constraint, Ev, IncomingParams, InterruptWaitError,
	Outcome, ParamError, Part, PromptCaps, Rev, Tool, ToolSpec,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{read::selector::LiteralPathProbe, render::TextProjection};

/// Archive and SQLite write seams.
pub mod backends;

const DESCRIPTION: &str =
	"Creates or overwrites file at specified path.\n\n<conditions>\n- Creating new files \
	 explicitly required by task\n- Replacing entire file contents when editing would be more \
	 complex\n- Supports `.tar`, `.tar.gz`, `.tgz`, `.zip`, and ZIP-based \
	 `.jar`/`.war`/`.ear`/`.apk` archive entries via `archive.ext:path/inside/archive`\n- Supports \
	 SQLite row operations via `db.sqlite:table` (insert), `db.sqlite:table:key` (update with JSON \
	 content, delete with empty content)\n</conditions>\n\n<critical>\n- You SHOULD use Edit tool \
	 for modifying existing files\n- You NEVER create documentation files (*.md, README) unless \
	 explicitly requested\n- You NEVER use emojis unless requested\n</critical>";
const EXECUTABLE_NOTICE: &str = "[Notice: Made executable via chmod +x]";
const STRIPPED_NOTICE: &str =
	"Note: auto-stripped hashline display prefixes from content before writing.";

/// Model arguments for `write@1`.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[schemars(description = "")]
#[serde(deny_unknown_fields)]
pub struct Params {
	/// file path
	#[schemars(with = "String")]
	pub path:    Str,
	/// file content
	#[schemars(with = "String")]
	pub content: Str,
}

/// Ephemeral write progress. Plain writes do not emit speculative updates.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum Update {}

/// Whether the committed plain write created or replaced its target.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WriteDisposition {
	/// The target did not exist before the transaction.
	Created,
	/// The target existed and was atomically replaced.
	Overwrote,
}

/// Mutation family used to project pi's exact special-write response text.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WriteOperation {
	/// Plain whole-file create or overwrite.
	Plain,
	/// ZIP/TAR member create or replacement.
	ArchiveMember,
	/// SQLite row insertion.
	SqliteInsert { table: Str },
	/// SQLite row update, including a no-match result.
	SqliteUpdate { table: Str, key: Str, changed: bool },
	/// SQLite row deletion, including a no-match result.
	SqliteDelete { table: Str, key: Str, changed: bool },
}

/// Fully validated whole-file request passed to the document owner.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlainWriteRequest {
	/// Authored local path, after strict hashline-header unwrapping.
	pub path:    Str,
	/// Exact text to persist, after display-prefix stripping.
	pub content: Str,
}

/// Resource-owned truth returned after one atomic plain-file transaction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlainWriteResult {
	/// Canonical absolute path committed by the document owner.
	pub resolved_path:   Str,
	/// Stable workspace-relative or shortened model-facing path.
	pub display_path:    Str,
	/// Exact number of UTF-8 bytes persisted.
	pub byte_len:        u64,
	/// Whether the transaction created or replaced the target.
	pub disposition:     WriteDisposition,
	/// Whether a shebang caused at least one execute bit to be added.
	pub made_executable: bool,
	/// Four-character tag recorded in the shared session snapshot store.
	/// Absent for oversized or otherwise untaggable text.
	pub snapshot_tag:    Option<Str>,
}

/// Durable successful `write@1` result.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Payload {
	/// Canonical absolute committed path.
	pub resolved_path:    Str,
	/// Stable model-facing committed path.
	pub display_path:     Str,
	/// Exact UTF-8 byte length persisted.
	pub byte_len:         u64,
	/// Pi-compatible JavaScript string length (UTF-16 code units) reported in
	/// the model-facing success line.
	pub reported_len:     u64,
	/// Whether the transaction created or replaced the target.
	pub disposition:      WriteDisposition,
	/// Whether the content-copy guard stripped read/hashline decoration.
	pub stripped_wrapper: bool,
	/// Whether the host added execute bits for a leading shebang.
	pub made_executable:  bool,
	/// Four-character shared-session snapshot tag, when taggable.
	pub snapshot_tag:     Option<Str>,
	/// Typed mutation family and SQLite outcome details.
	pub operation:        WriteOperation,
}

/// Durable typed `write@1` failure.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Fault {
	/// A URI scheme has no writable resource implementation yet.
	UnsupportedScheme {
		/// Lowercase URI scheme without punctuation.
		scheme: Str,
	},
	/// A malformed URI-like path was refused instead of becoming a local file.
	UriLikeTarget {
		/// Exact model-facing diagnostic.
		message: Str,
	},
	/// An empty write was accidentally addressed to a read range.
	ReadSelectorMisfire {
		/// Original authored target.
		target:   Str,
		/// Selector without its leading colon.
		selector: Str,
	},
	/// A semicolon-joined multi-read expression was passed as one write target.
	ReadSelectorListMisfire {
		/// Original authored target.
		target: Str,
		/// Number of selector-bearing segments.
		count:  usize,
	},
	/// The document resource rejected the request without changing the target.
	Document {
		/// Exact resource-owned explanation.
		message: Str,
	},
}

impl fmt::Display for Fault {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::UnsupportedScheme { scheme } => {
				write!(formatter, "{scheme}:// targets are not supported yet")
			},
			Self::UriLikeTarget { message } | Self::Document { message } => {
				formatter.write_str(message)
			},
			Self::ReadSelectorMisfire { target, selector } => write!(
				formatter,
				"write target '{target}' ends with a read-tool selector ':{selector}' and no such \
				 file exists — refusing to create a literal file by that name. If you meant to read \
				 it, use read({{ path: \"{target}\" }}). If you truly intend to create this file, \
				 pass its contents in `content` (a non-empty write is never blocked)."
			),
			Self::ReadSelectorListMisfire { target, count } => write!(
				formatter,
				"write target '{target}' is a semicolon-joined list of {count} read-tool selectors, \
				 not a filesystem path — refusing to create it. write creates a single file; issue \
				 one read() per path to read these ranges (e.g. read({{ path: \"<one path>:<range>\" \
				 }}))."
			),
		}
	}
}

/// Resource failure classification for the effectful whole-file transaction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WriteCommitError {
	/// The resource proves the target was not changed.
	Rejected(Fault),
	/// The resource cannot prove whether the effect landed.
	EffectsUnknown {
		/// Stable explanation of the uncertainty.
		reason: Str,
	},
}

/// Session document boundary used by `write@1`.
///
/// Implementations MUST use the same transaction coordinator and hashline
/// snapshot store as read/edit. A successful `write_plain` atomically creates
/// parent directories and the target or replaces it while preserving existing
/// mode bits. For a new shebang file it applies the platform default mode and
/// adds `a+x`; for an existing shebang file it adds only missing execute bits.
pub trait WriteDocuments: Send + Sync + 'static {
	/// Probe the exact literal spelling without following a trailing read
	/// selector. Ambiguous errors return [`LiteralPathProbe::Unknown`].
	fn probe_literal(
		&self,
		path: Str,
	) -> impl Future<Output = Result<LiteralPathProbe, Fault>> + Send + '_;

	/// Atomically commit a plain whole-file request and record its fresh
	/// snapshot in the session-shared store before returning.
	fn write_plain(
		&self,
		request: PlainWriteRequest,
	) -> impl Future<Output = Result<PlainWriteResult, WriteCommitError>> + Send + '_;

	/// Attempts an archive-member write after commitment.
	fn write_archive_member(
		&self,
		_display_path: Str,
		_content: Bytes,
	) -> impl Future<Output = Result<Option<backends::ResultPayload>, backends::Fault>> + Send + '_
	{
		std::future::ready(Ok(None))
	}

	/// Attempts a SQLite-row mutation after archive dispatch.
	fn write_sqlite_row(
		&self,
		_display_path: Str,
		_content: Str,
	) -> impl Future<Output = Result<Option<backends::ResultPayload>, backends::Fault>> + Send + '_
	{
		std::future::ready(Ok(None))
	}
}

/// `write@1` executor.
pub struct WriteTool<D> {
	documents: D,
	spec:      ToolSpec,
}

/// Construct the built-in whole-file write tool.
pub fn tool<D: WriteDocuments>(documents: D) -> WriteTool<D> {
	WriteTool {
		documents,
		spec: ToolSpec {
			name:        "write".into(),
			rev:         Rev { family: Str::new(""), n: 1 },
			description: DESCRIPTION.into(),
			schema:      omp_tool::schema::<Params>(),
			constraint:  Constraint::Schema { priority: 100 },
		},
	}
}

impl<D: WriteDocuments> Tool for WriteTool<D> {
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
			let path = Str::from(unwrap_hashline_header_path(&arguments.path));
			if let Some(fault) = reject_uri_like_target(&path) {
				yield done(Err(fault));
				return;
			}
			let stripped = strip_write_content(&arguments.content);
			let reported_len =
				u64::try_from(stripped.text.encode_utf16().count()).unwrap_or(u64::MAX);

			match params.interruptable().committed().await {
				Ok(_) => {},
				Err(error) => {
					yield commit_event(error);
					return;
				},
			}

			let archive_result = {
				let operation = self.documents.write_archive_member(
					path.clone(),
					Bytes::copy_from_slice(stripped.text.as_bytes()),
				).fuse();
				let interruption = params.next_interrupt().fuse();
				pin_mut!(operation, interruption);
				select_biased! {
					result = operation => match result {
						Ok(result) => result,
						Err(fault) => {
							yield done(Err(Fault::Document { message: fault.message }));
							return;
						},
					},
					interrupt = interruption => {
						yield interrupt_event(interrupt, true);
						return;
					},
				}
			};
			if let Some(result) = archive_result {
				yield done(Ok(special_payload(result, stripped.stripped, reported_len)));
				return;
			}

			let sqlite_result = {
				let operation = self.documents.write_sqlite_row(
					path.clone(),
					stripped.text.clone(),
				).fuse();
				let interruption = params.next_interrupt().fuse();
				pin_mut!(operation, interruption);
				select_biased! {
					result = operation => match result {
						Ok(result) => result,
						Err(fault) => {
							yield done(Err(Fault::Document { message: fault.message }));
							return;
						},
					},
					interrupt = interruption => {
						yield interrupt_event(interrupt, true);
						return;
					},
				}
			};
			if let Some(result) = sqlite_result {
				yield done(Ok(special_payload(result, stripped.stripped, reported_len)));
				return;
			}

			let literal = {
				let probe = self.documents.probe_literal(path.clone()).fuse();
				let interruption = params.next_interrupt().fuse();
				pin_mut!(probe, interruption);
				select_biased! {
					result = probe => match result {
						Ok(result) => result,
						Err(fault) => {
							yield done(Err(fault));
							return;
						},
					},
					interrupt = interruption => {
						yield interrupt_event(interrupt, false);
						return;
					},
				}
			};
			if literal == LiteralPathProbe::Missing {
				if let Some(count) = read_selector_list_misfire(&path) {
					yield done(Err(Fault::ReadSelectorListMisfire { target: path, count }));
					return;
				}
				if stripped.text.is_empty() {
					let split = crate::read::selector::split_path_and_selector(&path);
					if let Some(selector) = split.selector.map(Str::from) {
						yield done(Err(Fault::ReadSelectorMisfire {
							target: path.clone(),
							selector,
						}));
						return;
					}
				}
			}

			let request = PlainWriteRequest { path, content: stripped.text };
			let operation = self.documents.write_plain(request).fuse();
			let interruption = params.next_interrupt().fuse();
			pin_mut!(operation, interruption);
			select_biased! {
				result = operation => match result {
					Ok(result) => yield done(Ok(Payload {
						resolved_path: result.resolved_path,
						display_path: result.display_path,
						byte_len: result.byte_len,
						reported_len,
						disposition: result.disposition,
						stripped_wrapper: stripped.stripped,
						made_executable: result.made_executable,
						snapshot_tag: result.snapshot_tag,
						operation: WriteOperation::Plain,
					})),
					Err(WriteCommitError::Rejected(fault)) => yield done(Err(fault)),
					Err(WriteCommitError::EffectsUnknown { reason }) => {
						yield Ev::Aborted(Abort::EffectsUnknown { reason });
					},
				},
				interrupt = interruption => {
					yield interrupt_event(interrupt, true);
				},
			}
		}
	}

	fn prompt(&self, view: Result<&Payload, &Fault>, caps: &PromptCaps) -> Vec<Part> {
		let Some(mut output) = TextProjection::new(caps) else {
			return Vec::new();
		};
		match view {
			Ok(payload) => {
				let rendered = render_payload(payload);
				output.push(&rendered);
			},
			Err(fault) => {
				let rendered = fault.to_string();
				output.push(&rendered);
			},
		}
		output.finish()
	}
}

#[derive(Debug)]
struct StrippedContent {
	text:     Str,
	stripped: bool,
}

fn strip_write_content(content: &str) -> StrippedContent {
	let lines: Vec<&str> = content.split('\n').collect();
	if let Some(cleaned) = strip_hashline_prefixes(&lines) {
		return StrippedContent { text: Str::from(cleaned.join("\n")), stripped: true };
	}
	let Some(header_index) = lines.iter().position(|line| !line.trim().is_empty()) else {
		return StrippedContent { text: Str::from(content), stripped: false };
	};
	if !is_loose_hashline_header(lines[header_index]) {
		return StrippedContent { text: Str::from(content), stripped: false };
	}
	let mut without_header = Vec::with_capacity(lines.len().saturating_sub(1));
	without_header.extend_from_slice(&lines[..header_index]);
	without_header.extend_from_slice(&lines[header_index + 1..]);
	if let Some(cleaned) = strip_hashline_prefixes(&without_header) {
		return StrippedContent { text: Str::from(cleaned.join("\n")), stripped: true };
	}
	StrippedContent { text: Str::from(content), stripped: false }
}

fn strip_hashline_prefixes(lines: &[&str]) -> Option<Vec<String>> {
	let mut content_lines = 0usize;
	let mut prefixed_lines = 0usize;
	for line in lines {
		if line.is_empty() || is_read_metadata_line(line) || is_strict_hashline_header(line) {
			continue;
		}
		content_lines += 1;
		if strip_one_hashline_prefix(line).is_some() {
			prefixed_lines += 1;
		}
	}
	if content_lines == 0 || content_lines != prefixed_lines {
		return None;
	}
	Some(
		lines
			.iter()
			.filter(|line| !is_read_metadata_line(line) && !is_strict_hashline_header(line))
			.map(|line| {
				let mut current = *line;
				while let Some(stripped) = strip_one_hashline_prefix(current) {
					current = stripped;
				}
				current.to_owned()
			})
			.collect(),
	)
}

fn strip_one_hashline_prefix(line: &str) -> Option<&str> {
	let mut rest = line.trim_start();
	if let Some(after) = rest.strip_prefix(">>>").or_else(|| rest.strip_prefix(">>")) {
		rest = after.trim_start();
	}
	if matches!(rest.as_bytes().first(), Some(b'+' | b'*' | b'-')) {
		rest = rest[1..].trim_start();
	}
	let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
	if digits == 0 || !matches!(rest.as_bytes().get(digits), Some(b':' | b'|')) {
		return None;
	}
	Some(&rest[digits + 1..])
}

fn is_strict_hashline_header(line: &str) -> bool {
	let Some(inner) = line
		.trim()
		.strip_prefix('[')
		.and_then(|line| line.strip_suffix(']'))
	else {
		return false;
	};
	let Some((path, tag)) = inner.rsplit_once('#') else {
		return false;
	};
	!path.is_empty()
		&& !path.contains(['#', '\r', '\n'])
		&& tag.len() == 4
		&& tag.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn is_loose_hashline_header(line: &str) -> bool {
	let Some(inner) = line
		.trim()
		.strip_prefix('[')
		.and_then(|line| line.strip_suffix(']'))
	else {
		return false;
	};
	let Some((path, tag)) = inner.rsplit_once('#') else {
		return false;
	};
	!path.is_empty()
		&& !path.contains(['#', '\r', '\n'])
		&& !tag.bytes().any(|byte| byte.is_ascii_whitespace())
}

fn is_read_metadata_line(line: &str) -> bool {
	let trimmed = line.trim();
	if matches!(trimmed, "…" | "...") {
		return true;
	}
	if trimmed.starts_with('[') && trimmed.ends_with(']') {
		let inner = &trimmed[1..trimmed.len() - 1];
		if (inner.starts_with("Showing lines ") || inner.contains("ln elided;"))
			&& (inner.contains("Use :") || inner.contains("re-read needed ranges"))
		{
			return true;
		}
	}
	let Some((range, body)) = trimmed.split_once(':') else {
		return false;
	};
	let Some((start, end)) = range.split_once('-') else {
		return false;
	};
	start.trim().bytes().all(|byte| byte.is_ascii_digit())
		&& end.trim().bytes().all(|byte| byte.is_ascii_digit())
		&& (body.contains('…') || body.contains("..."))
}

fn unwrap_hashline_header_path(path: &str) -> &str {
	let trimmed = path.trim_end();
	let Some(inner) = trimmed
		.strip_prefix('[')
		.and_then(|value| value.strip_suffix(']'))
	else {
		return path;
	};
	if inner.is_empty() {
		return path;
	}
	if let Some((path_part, tag)) = inner.rsplit_once('#') {
		if path_part.is_empty()
			|| path_part.contains('#')
			|| tag.len() != 4
			|| !tag.bytes().all(|byte| byte.is_ascii_hexdigit())
		{
			return path;
		}
		return path_part;
	}
	if inner.contains('#') { path } else { inner }
}

fn read_selector_list_misfire(target: &str) -> Option<usize> {
	if !target.contains(';') {
		return None;
	}
	let mut count = 0usize;
	for segment in target.split(';') {
		let trimmed = segment.trim();
		if trimmed.is_empty()
			|| crate::read::selector::split_path_and_selector(trimmed)
				.selector
				.is_none()
		{
			return None;
		}
		count += 1;
	}
	(count >= 2).then_some(count)
}

fn reject_uri_like_target(target: &str) -> Option<Fault> {
	let trimmed = target.trim();
	if windows_absolute(trimmed) {
		return None;
	}
	if trimmed
		.get(..3)
		.is_some_and(|prefix| prefix.eq_ignore_ascii_case("xd/"))
	{
		let rest = trimmed[3..].trim_start_matches('/');
		return Some(Fault::UriLikeTarget {
			message: Str::from(format!(
				"Unknown URI-like write target '{trimmed}'. Did you mean 'xd://{rest}'? Prefix the \
				 path with './' to write it as a filesystem path."
			)),
		});
	}
	let colon = trimmed.find(':')?;
	let scheme = &trimmed[..colon];
	if !valid_uri_scheme(scheme) {
		return None;
	}
	let suffix = &trimmed[colon + 1..];
	if let Some(_body) = suffix.strip_prefix("//") {
		return Some(Fault::UnsupportedScheme { scheme: Str::from(scheme.to_ascii_lowercase()) });
	}
	if !suffix.starts_with('/') {
		return None;
	}
	let canonical =
		matches!(scheme.to_ascii_lowercase().as_str(), "dx" | "xdd" | "xdt").then_some("xd");
	let suggestion = canonical.map_or_else(
		|| " Tool devices use 'xd://<tool>'.".to_owned(),
		|canonical| format!(" Did you mean '{canonical}://{}'?", suffix.trim_start_matches('/')),
	);
	Some(Fault::UriLikeTarget {
		message: Str::from(format!(
			"Unknown URI-like write target '{trimmed}'.{suggestion} Prefix the path with './' to \
			 write it as a filesystem path."
		)),
	})
}

fn valid_uri_scheme(scheme: &str) -> bool {
	let mut bytes = scheme.bytes();
	matches!(bytes.next(), Some(first) if first.is_ascii_alphabetic())
		&& bytes.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'.' | b'-'))
}

fn windows_absolute(path: &str) -> bool {
	path.as_bytes().get(1) == Some(&b':')
		&& path
			.as_bytes()
			.get(2)
			.is_some_and(|byte| matches!(byte, b'/' | b'\\'))
		&& path.as_bytes()[0].is_ascii_alphabetic()
}

fn special_payload(
	result: backends::ResultPayload,
	stripped_wrapper: bool,
	reported_len: u64,
) -> Payload {
	Payload {
		resolved_path: result.resolved_path,
		display_path: result.display_path,
		byte_len: result.byte_len,
		reported_len,
		disposition: result.disposition,
		stripped_wrapper,
		made_executable: false,
		snapshot_tag: result.snapshot_tag,
		operation: result.operation,
	}
}

fn render_payload(payload: &Payload) -> String {
	let mut output = String::new();
	if let Some(tag) = &payload.snapshot_tag {
		output.push_str(&format_hashline_header(&payload.display_path, tag));
		output.push('\n');
	}
	match &payload.operation {
		WriteOperation::Plain | WriteOperation::ArchiveMember => output.push_str(&format!(
			"Successfully wrote {} bytes to {}",
			payload.reported_len, payload.display_path
		)),
		WriteOperation::SqliteInsert { table } => {
			output.push_str(&format!("Inserted row into {table}"));
		},
		WriteOperation::SqliteUpdate { table, key, changed } => {
			if *changed {
				output.push_str(&format!("Updated row '{key}' in {table}"));
			} else {
				output.push_str(&format!("No row updated in {table} for key '{key}'"));
			}
		},
		WriteOperation::SqliteDelete { table, key, changed } => {
			if *changed {
				output.push_str(&format!("Deleted row '{key}' from {table}"));
			} else {
				output.push_str(&format!("No row deleted from {table} for key '{key}'"));
			}
		},
	}
	if payload.stripped_wrapper {
		output.push('\n');
		output.push_str(STRIPPED_NOTICE);
	}
	if payload.made_executable {
		output.push('\n');
		output.push_str(EXECUTABLE_NOTICE);
	}
	output
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
	effects_started: bool,
) -> Ev<Update, Payload, Fault> {
	let reason = match interrupt {
		Ok(interrupt) => interrupt.reason,
		Err(InterruptWaitError::Closed) if effects_started => {
			"invocation owner disappeared during write transaction".into()
		},
		Err(InterruptWaitError::Closed) => "write resource owner disappeared".into(),
		Err(InterruptWaitError::Protocol(message)) => return Ev::Args(protocol_issue(message)),
	};
	if effects_started {
		Ev::Aborted(Abort::EffectsUnknown { reason })
	} else {
		Ev::Aborted(Abort::Interrupted { reason })
	}
}

fn protocol_issue(message: Str) -> ArgIssue {
	ArgIssue {
		path:     Vec::new(),
		expected: "one committed JSON argument object".into(),
		kind:     ArgIssueKind::Protocol,
		example:  Some("{\"path\":\"src/main.rs\",\"content\":\"fn main() {}\\n\"}".into()),
		found:    Some(message),
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn strips_strict_hashline_read_echo() {
		let stripped = strip_write_content("[src/a.rs#A1B2]\n1:fn main() {\n2:}\n");
		assert!(stripped.stripped);
		assert_eq!(stripped.text, "fn main() {\n}\n");
	}

	#[test]
	fn strips_loose_header_only_when_body_is_prefixed() {
		let stripped = strip_write_content("[src/a.rs#stale]\n10:first\n11:second");
		assert!(stripped.stripped);
		assert_eq!(stripped.text, "first\nsecond");
		let literal = strip_write_content("[src/a.rs#stale]\nfirst\nsecond");
		assert!(!literal.stripped);
	}

	#[test]
	fn preserves_literal_numbered_content_when_not_uniform() {
		let stripped = strip_write_content("1:first\nliteral\n3:third");
		assert!(!stripped.stripped);
		assert_eq!(stripped.text, "1:first\nliteral\n3:third");
	}

	#[test]
	fn selector_misfire_messages_match_pi() {
		assert_eq!(
			Fault::ReadSelectorMisfire { target: "a.rs:1-2".into(), selector: "1-2".into() }
				.to_string(),
			"write target 'a.rs:1-2' ends with a read-tool selector ':1-2' and no such file exists — \
			 refusing to create a literal file by that name. If you meant to read it, use read({ \
			 path: \"a.rs:1-2\" }). If you truly intend to create this file, pass its contents in \
			 `content` (a non-empty write is never blocked)."
		);
		assert_eq!(
			Fault::ReadSelectorListMisfire { target: "a:1-2;b:3-4".into(), count: 2 }.to_string(),
			"write target 'a:1-2;b:3-4' is a semicolon-joined list of 2 read-tool selectors, not a \
			 filesystem path — refusing to create it. write creates a single file; issue one read() \
			 per path to read these ranges (e.g. read({ path: \"<one path>:<range>\" }))."
		);
	}

	#[test]
	fn renders_plain_write_exactly() {
		let payload = Payload {
			resolved_path:    "/repo/bin/run".into(),
			display_path:     "bin/run".into(),
			byte_len:         10,
			reported_len:     10,
			disposition:      WriteDisposition::Created,
			stripped_wrapper: true,
			made_executable:  true,
			snapshot_tag:     Some("A1B2".into()),
			operation:        WriteOperation::Plain,
		};
		assert_eq!(
			render_payload(&payload),
			"[bin/run#A1B2]\nSuccessfully wrote 10 bytes to bin/run\nNote: auto-stripped hashline \
			 display prefixes from content before writing.\n[Notice: Made executable via chmod +x]"
		);
	}

	#[test]
	fn renders_archive_count_with_pi_utf16_length() {
		let payload = Payload {
			resolved_path:    "/repo/a.zip".into(),
			display_path:     "a.zip:x.txt".into(),
			byte_len:         "é😀".len() as u64,
			reported_len:     "é😀".encode_utf16().count() as u64,
			disposition:      WriteDisposition::Created,
			stripped_wrapper: false,
			made_executable:  false,
			snapshot_tag:     None,
			operation:        WriteOperation::ArchiveMember,
		};
		assert_eq!(render_payload(&payload), "Successfully wrote 3 bytes to a.zip:x.txt");
	}

	#[test]
	fn renders_sqlite_row_outcomes_exactly() {
		let payload = Payload {
			resolved_path:    "/repo/data.db".into(),
			display_path:     "data.db:items:7".into(),
			byte_len:         14,
			reported_len:     14,
			disposition:      WriteDisposition::Overwrote,
			stripped_wrapper: false,
			made_executable:  false,
			snapshot_tag:     None,
			operation:        WriteOperation::SqliteUpdate {
				table:   "items".into(),
				key:     "7".into(),
				changed: false,
			},
		};
		assert_eq!(render_payload(&payload), "No row updated in items for key '7'");
	}

	#[test]
	fn rejects_uri_shapes_without_blocking_windows_paths() {
		assert_eq!(
			reject_uri_like_target("skill://x")
				.expect("fault")
				.to_string(),
			"skill:// targets are not supported yet"
		);
		assert!(reject_uri_like_target("C:\\tmp\\x").is_none());
	}
}
