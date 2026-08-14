//! Regex workspace search with pi-compatible grouping and pagination.

use std::{
	collections::{HashMap, HashSet},
	fmt::{self, Write as _},
	future::Future,
};

use async_stream::stream;
use futures::{FutureExt, Stream, pin_mut, select_biased};
use omp_core::Str;
use omp_hashline::format_hashline_header;
use omp_tool::{
	Abort, ArgIssue, ArgIssueKind, BlobRef, CommitError, Constraint, Ev, IncomingParams,
	InterruptWaitError, Outcome, ParamError, Part, PromptCaps, Rev, Tool, ToolSpec,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
	read::{
		ReadBlobs,
		selector::{
			LineRange, ParsedSelector, UriTarget, classify_uri_target, line_is_in_ranges,
			parse_selector, split_path_and_selector, split_semicolon_targets,
		},
	},
	render::{
		TextProjection,
		paths::{GroupedTreeEventKind, PathTreeInput, build_path_tree, walk_path_tree},
		truncate::spill_truncated_text,
	},
};

const DEFAULT_FILE_LIMIT: usize = 20;
const MULTI_FILE_PER_FILE_MATCHES: usize = 20;
const SINGLE_FILE_MATCHES: usize = 200;
const INTERNAL_TOTAL_CAP: u32 = 2_000;
const NATIVE_GREP_MAX_FILE_BYTES: u32 = 4 * 1024 * 1024;
const SEARCH_GREP_TIMEOUT_MS: u32 = 30_000;
const DEFAULT_MAX_COLUMN: u32 = 512;

// Model arguments for `grep@1`.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Params {
	/// regex pattern
	#[schemars(with = "String")]
	pub pattern:   Str,
	/// file, directory, glob, internal URL, or "<file>:<lines>" selector to
	/// search; pass several as a semicolon-delimited list ("src; tests").
	/// Omitted -> searches the workspace root (".")
	#[schemars(with = "Option<String>")]
	pub path:      Option<Str>,
	/// case-sensitive search
	#[serde(rename = "case")]
	pub case:      Option<bool>,
	/// respect gitignore
	pub gitignore: Option<bool>,
	/// files to skip before collecting results — use to paginate when the prior
	/// call hit the file limit
	#[schemars(with = "Option<Option<f64>>")]
	pub skip:      Option<f64>,
}

/// Kind of target supplied to the workspace search resource.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SearchRootKind {
	/// A local file, directory, or glob.
	Filesystem,
	/// A member-shaped archive target awaiting archive materialization.
	Archive,
	/// An HTTP(S) target awaiting URL materialization.
	Url,
}

/// One selector-peeled search target.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SearchRoot {
	/// Caller spelling, retained for diagnostics.
	pub original: Str,
	/// Target with a trailing line selector removed.
	pub path:     Str,
	/// I/O route the adapter must use.
	pub kind:     SearchRootKind,
	/// Inclusive one-based match ranges, empty for an unrestricted target.
	pub ranges:   Box<[LineRange]>,
}

/// Fully specified request passed to the workspace resource after commitment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SearchRequest {
	/// Regular expression preserved verbatim.
	pub pattern:               Str,
	/// Semicolon-expanded, selector-peeled roots.
	pub roots:                 Vec<SearchRoot>,
	/// Whether matching ignores case.
	pub ignore_case:           bool,
	/// Whether the expression is searched across lines.
	pub multiline:             bool,
	/// Whether ignore files are respected.
	pub gitignore:             bool,
	/// Dot-prefixed candidates are always included by grep.
	pub hidden:                bool,
	/// Global native safety ceiling.
	pub max_count:             u32,
	/// Native per-file fetch budget for a single-file scope.
	pub single_file_max_count: u32,
	/// Native per-file fetch budget for a multi-file scope.
	pub multi_file_max_count:  u32,
	/// Leading context line count.
	pub context_before:        u32,
	/// Trailing context line count.
	pub context_after:         u32,
	/// Maximum retained columns in one matching line.
	pub max_columns:           u32,
	/// Native wall-clock deadline.
	pub timeout_ms:            u32,
}

/// One context line adjacent to a regex match.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ContextLine {
	/// One-based source line number.
	pub line_number: u32,
	/// Retained line text.
	pub line:        Str,
}

/// One resource match before range filtering, overlap deduplication, and
/// grouping.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SearchMatch {
	/// Stable canonical identity used only for overlap deduplication.
	pub source_key:     Str,
	/// Workspace-relative model-facing path.
	pub path:           Str,
	/// Index of the request root that produced this match.
	pub root_index:     u64,
	/// One-based source line number.
	pub line_number:    u32,
	/// Retained matching line text.
	pub line:           Str,
	/// Whether the native engine clipped this line at the column cap.
	pub truncated:      bool,
	/// Leading context in source order.
	pub context_before: Vec<ContextLine>,
	/// Trailing context in source order.
	pub context_after:  Vec<ContextLine>,
	/// Whole-file snapshot tag, absent for immutable or oversized sources.
	pub snapshot_tag:   Option<Str>,
}

/// Structured resource result returned to the executor.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct SearchResult {
	/// Matches in deterministic traversal order.
	pub matches:            Vec<SearchMatch>,
	/// Whether the resolved scope can contain multiple files.
	pub multi_scope:        bool,
	/// Whether the native global match ceiling prevented a complete scan.
	pub limit_reached:      bool,
	/// Count of unreadable large candidates whose names were unavailable.
	pub skipped_oversized:  u32,
	/// Missing targets retained in caller order.
	pub missing_paths:      Vec<Str>,
	/// Archive members that could not be searched as UTF-8 text.
	pub archive_unreadable: Vec<Str>,
	/// Explicit files searched only through the leading 4MB window.
	pub oversized_files:    Vec<Str>,
}

/// One retained match in a grouped payload.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct FileMatch {
	/// One-based source line number.
	pub line_number:    u32,
	/// Retained matching line text.
	pub line:           Str,
	/// Whether the matching line was column-truncated.
	pub truncated:      bool,
	/// Leading context within the requested line ranges.
	pub context_before: Vec<ContextLine>,
	/// Trailing context within the requested line ranges.
	pub context_after:  Vec<ContextLine>,
}

/// One model-facing file section.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct FileGroup {
	/// Workspace-relative display path.
	pub path:         Str,
	/// Whole-file hashline snapshot tag when editable.
	pub snapshot_tag: Option<Str>,
	/// Retained matches in source order.
	pub matches:      Vec<FileMatch>,
}

/// Durable successful `grep@1` result.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Payload {
	/// Current page of grouped file matches.
	pub files:                   Vec<FileGroup>,
	/// Number of distinct matching files observed before pagination.
	pub total_files:             u64,
	/// Whether the total is a lower bound because native grep stopped early.
	pub total_files_lower_bound: bool,
	/// Whether the resolved scope can contain multiple files.
	pub multi_scope:             bool,
	/// Effective file offset for this page.
	pub skip:                    u64,
	/// Whether more matching files remain after this page.
	pub file_limit_reached:      bool,
	/// Whether any hot file was clipped at its diversity cap.
	pub per_file_limit_reached:  bool,
	/// Ordered model-facing diagnostic notes.
	pub notes:                   Vec<Str>,
	/// Exact bounded model-facing text prepared before prompt projection.
	pub projected_text:          Str,
	/// Durable complete output when `projected_text` was pre-truncated.
	pub output_blob:             Option<BlobRef>,
	/// Complete lines retained in `projected_text` before its footer.
	pub output_shown_lines:      u64,
	/// Complete line count in the pre-truncation output.
	pub output_total_lines:      u64,
}

/// Ephemeral progress from `grep@1`; grep has no durable updates.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum Update {}

/// Durable typed `grep@1` failure.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Fault {
	/// The expression was empty or whitespace-only.
	EmptyPattern,
	/// The requested file offset was negative or not finite.
	InvalidSkip,
	/// A path selector was invalid for grep.
	InvalidSelector {
		/// Exact model-facing diagnostic.
		message: Str,
	},
	/// A URI target uses a backend that has not landed yet.
	UnsupportedTarget {
		/// Exact model-facing diagnostic.
		message: Str,
	},
	/// Neither the Rust regex engine nor PCRE2 accepted the expression.
	InvalidRegex {
		/// Parser detail without the `Invalid regex:` prefix.
		message: Str,
	},
	/// The fixed 30-second native deadline elapsed.
	TimedOut,
	/// Every submitted path was missing.
	AllPathsMissing {
		/// Missing paths in caller order.
		paths: Vec<Str>,
	},
	/// The workspace owner rejected or failed the request.
	Workspace {
		/// Stable resource-owned explanation.
		message: Str,
	},
	/// Durable blob storage failed while preserving complete output.
	Blob {
		/// Stable blob-owned explanation.
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
			Self::EmptyPattern => formatter.write_str("Pattern must not be empty"),
			Self::InvalidSkip => formatter.write_str("Skip must be a non-negative number"),
			Self::InvalidSelector { message }
			| Self::UnsupportedTarget { message }
			| Self::Workspace { message }
			| Self::Blob { message } => formatter.write_str(message),
			Self::InvalidRegex { message } => write!(formatter, "Invalid regex: {message}"),
			Self::TimedOut => formatter.write_str(
				"Grep timed out after 30s; narrow paths or pattern, or scope with `glob` first",
			),
			Self::AllPathsMissing { paths } => write!(
				formatter,
				"Path not found: {}; list each target in the semicolon-delimited `path`",
				join_strs(paths)
			),
			Self::Cancelled { reason } => {
				write!(formatter, "workspace search was cancelled: {reason}")
			},
		}
	}
}

impl std::error::Error for Fault {}

/// Zero-box workspace traversal boundary shared by `grep@1` and `glob@1`.
pub trait WorkspaceSearch: Send + Sync + 'static {
	/// Execute a native regex search and mint snapshot tags for editable
	/// matches.
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

/// Generic `grep@1` executor over environment-owned workspace and blob
/// resources.
pub struct Grep<W, B> {
	workspace: W,
	blobs:     B,
	spec:      ToolSpec,
}

/// Construct `grep@1` over `workspace` and the shared durable blob namespace.
pub fn tool<W: WorkspaceSearch, B: ReadBlobs>(workspace: W, blobs: B) -> Grep<W, B> {
	Grep {
		workspace,
		blobs,
		spec: ToolSpec {
			name:        Str::from("grep"),
			rev:         Rev { family: Str::new(""), n: 1 },
			description: Str::new_static(
				"Searches files/internal URLs: Rust regex, PCRE2 fallback.\n\n<instruction>\n- \
				 `path`: known files, directories, globs, internal URLs; roots `;`-separated.\n- \
				 Broad searches may time out → narrow scope or use `glob` first.\n- One-file line \
				 selector: `src/foo.ts:50-100`; never selects search root.\n- Literal `\\n` or \
				 `\\\\n` enables cross-line patterns.\n</instruction>\n\n<critical>\n- MUST use \
				 instead of shell `grep`/`rg`.\n</critical>",
			),
			schema:      omp_tool::schema::<Params>(),
			constraint:  Constraint::Schema { priority: 100 },
		},
	}
}

impl<W: WorkspaceSearch, B: ReadBlobs> Tool for Grep<W, B> {
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

			if arguments.pattern.trim().is_empty() {
				yield done(Err(Fault::EmptyPattern));
				return;
			}
			let skip = match normalize_skip(arguments.skip) {
				Ok(skip) => skip,
				Err(fault) => {
					yield done(Err(fault));
					return;
				},
			};
			let roots = match parse_roots(arguments.path.as_deref()) {
				Ok(roots) => roots,
				Err(fault) => {
					yield done(Err(fault));
					return;
				},
			};
			let (single_file_max_count, multi_file_max_count, max_count) = fetch_budgets(&roots);
			let multiline = arguments.pattern.contains('\n') || arguments.pattern.contains("\\n");
			let request = SearchRequest {
				pattern: arguments.pattern,
				roots: roots.clone(),
				ignore_case: !arguments.case.unwrap_or(true),
				multiline,
				gitignore: arguments.gitignore.unwrap_or(true),
				hidden: true,
				max_count,
				single_file_max_count,
				multi_file_max_count,
				context_before: 0,
				context_after: 0,
				max_columns: DEFAULT_MAX_COLUMN,
				timeout_ms: SEARCH_GREP_TIMEOUT_MS,
			};
			let operation = async {
				let result = self.workspace.search(request).await?;
				prepare_payload(result, &roots, skip, &self.blobs).await
			}.fuse();
			let interruption = params.next_interrupt().fuse();
			pin_mut!(operation, interruption);
			select_biased! {
				result = operation => yield done(result),
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
		let text = match view {
			Ok(payload) => payload.projected_text.as_str(),
			Err(fault) => {
				let message = fault.to_string();
				projection.push(&message);
				return projection.finish();
			},
		};
		for fragment in text.split_inclusive('\n') {
			if !projection.push(fragment) {
				break;
			}
		}
		projection.finish()
	}
}

fn normalize_skip(skip: Option<f64>) -> Result<u64, Fault> {
	let skip = skip.unwrap_or(0.0);
	if !skip.is_finite() || skip < 0.0 {
		return Err(Fault::InvalidSkip);
	}
	Ok(skip.floor() as u64)
}

fn parse_roots(path: Option<&str>) -> Result<Vec<SearchRoot>, Fault> {
	let entries = path.map(split_semicolon_targets).unwrap_or_default();
	let entries = if entries.is_empty() {
		vec![Str::from(".")]
	} else {
		entries
	};
	entries.into_iter().map(parse_root).collect()
}

fn parse_root(original: Str) -> Result<SearchRoot, Fault> {
	let split = split_path_and_selector(&original);
	let mut ranges = Box::<[LineRange]>::default();
	if let Some(selector) = split.selector {
		let parsed = parse_selector(Some(selector)).map_err(|error| Fault::InvalidSelector {
			message: Str::from(format!(
				"path entry \"{original}\" has an invalid selector \":{selector}\" — {error}"
			)),
		})?;
		match parsed {
			ParsedSelector::Lines { ranges: selected, raw: false } => ranges = selected,
			ParsedSelector::Lines { raw: true, .. }
			| ParsedSelector::Raw
			| ParsedSelector::Conflicts => {
				return Err(Fault::InvalidSelector {
					message: Str::from(format!(
						"path entry \"{original}\" — only line-range selectors like \":50-100\" are \
						 supported (no \":raw\"/\":conflicts\")"
					)),
				});
			},
			ParsedSelector::None => {},
		}
	}
	let clean = split.path;
	if !ranges.is_empty() && has_glob_chars(clean) {
		return Err(Fault::InvalidSelector {
			message: Str::from(format!(
				"Line-range selector requires a single file, not a glob: {original}"
			)),
		});
	}
	let kind = match classify_uri_target(clean) {
		UriTarget::Http { .. } => SearchRootKind::Url,
		UriTarget::Unsupported { scheme } => {
			return Err(Fault::UnsupportedTarget {
				message: Str::from(format!(
					"{}:// targets are not supported yet",
					scheme.to_ascii_lowercase()
				)),
			});
		},
		UriTarget::LocalOrOther if looks_like_archive_member(clean) => SearchRootKind::Archive,
		UriTarget::LocalOrOther => SearchRootKind::Filesystem,
	};
	Ok(SearchRoot { original: Str::from(original.as_str()), path: Str::from(clean), kind, ranges })
}

fn has_glob_chars(path: &str) -> bool {
	path
		.bytes()
		.any(|byte| matches!(byte, b'*' | b'?' | b'[' | b']' | b'{' | b'}'))
}

fn looks_like_archive_member(path: &str) -> bool {
	let lower = path.to_ascii_lowercase();
	[".zip:", ".jar:", ".war:", ".ear:", ".apk:", ".tar:", ".tar.gz:", ".tgz:"]
		.iter()
		.any(|marker| lower.contains(marker))
}

fn fetch_budgets(roots: &[SearchRoot]) -> (u32, u32, u32) {
	let single_baseline = u32::try_from(SINGLE_FILE_MATCHES + 1).unwrap_or(u32::MAX);
	let multi_baseline = u32::try_from(MULTI_FILE_PER_FILE_MATCHES + 1).unwrap_or(u32::MAX);
	let has_ranges = roots.iter().any(|root| !root.ranges.is_empty());
	let range_cap = |per_file_keep: u32| {
		let cap = roots
			.iter()
			.flat_map(|root| root.ranges.iter())
			.map(|range| {
				range.end_line.unwrap_or_else(|| {
					range
						.start_line
						.saturating_sub(1)
						.saturating_add(u64::from(per_file_keep))
				})
			})
			.max()
			.unwrap_or(0)
			.min(u64::from(NATIVE_GREP_MAX_FILE_BYTES));
		u32::try_from(cap).unwrap_or(NATIVE_GREP_MAX_FILE_BYTES)
	};
	let single = single_baseline.max(range_cap(single_baseline));
	let multi = multi_baseline.max(range_cap(multi_baseline));
	let max_count = if has_ranges {
		INTERNAL_TOTAL_CAP
			.div_ceil(multi_baseline)
			.saturating_mul(multi)
	} else {
		INTERNAL_TOTAL_CAP
	};
	(single, multi, max_count)
}

fn make_payload(result: SearchResult, roots: &[SearchRoot], requested_skip: u64) -> Payload {
	let per_file_cap = if result.multi_scope {
		MULTI_FILE_PER_FILE_MATCHES
	} else {
		SINGLE_FILE_MATCHES
	};
	let mut groups = Vec::<FileGroup>::new();
	let mut group_by_path = HashMap::<Str, usize>::new();
	let mut seen = HashSet::<(Str, u32)>::new();
	let mut per_file_limit_reached = false;

	for matched in result.matches {
		let ranges = usize::try_from(matched.root_index)
			.ok()
			.and_then(|index| roots.get(index))
			.map(|root| root.ranges.as_ref())
			.unwrap_or_default();
		if !ranges.is_empty() && !line_is_in_ranges(u64::from(matched.line_number), ranges) {
			continue;
		}
		if !seen.insert((matched.source_key.clone(), matched.line_number)) {
			continue;
		}
		let group_index = match group_by_path.get(&matched.path).copied() {
			Some(index) => index,
			None => {
				let index = groups.len();
				group_by_path.insert(matched.path.clone(), index);
				groups.push(FileGroup {
					path:         matched.path.clone(),
					snapshot_tag: matched.snapshot_tag.clone(),
					matches:      Vec::new(),
				});
				index
			},
		};
		let group = &mut groups[group_index];
		if group.matches.len() >= per_file_cap {
			per_file_limit_reached = true;
			continue;
		}
		let filter_context = |context: Vec<ContextLine>| {
			if ranges.is_empty() {
				context
			} else {
				context
					.into_iter()
					.filter(|line| line_is_in_ranges(u64::from(line.line_number), ranges))
					.collect()
			}
		};
		group.matches.push(FileMatch {
			line_number:    matched.line_number,
			line:           matched.line,
			truncated:      matched.truncated,
			context_before: filter_context(matched.context_before),
			context_after:  filter_context(matched.context_after),
		});
	}

	let total_files = u64::try_from(groups.len()).unwrap_or(u64::MAX);
	let skip = if result.multi_scope {
		requested_skip
	} else {
		0
	};
	let start = usize::try_from(skip.min(total_files))
		.unwrap_or(usize::MAX)
		.min(groups.len());
	let end = start.saturating_add(DEFAULT_FILE_LIMIT).min(groups.len());
	let file_limit_reached = result.multi_scope && end < groups.len();
	let files = groups.drain(start..end).collect();
	let mut notes = Vec::new();
	if !result.missing_paths.is_empty() {
		notes.push(Str::from(format!("Skipped missing paths: {}", join_strs(&result.missing_paths))));
	}
	if !result.archive_unreadable.is_empty() {
		notes.push(Str::from(format!(
			"Skipped archive entries (search supports text members only): {}",
			join_strs(&result.archive_unreadable)
		)));
	}
	if !result.oversized_files.is_empty() {
		notes.push(Str::from(format!(
			"Searched only the first 4MB of large files (matches past the 4MB window are not shown; \
			 use `read` for the rest): {}",
			join_strs(&result.oversized_files)
		)));
	} else if result.skipped_oversized > 0 {
		notes.push(Str::from(format!(
			"Skipped {} unreadable large file(s); target them directly with `read`",
			result.skipped_oversized
		)));
	}
	Payload {
		files,
		total_files,
		total_files_lower_bound: result.limit_reached,
		multi_scope: result.multi_scope,
		skip,
		file_limit_reached,
		per_file_limit_reached,
		notes,
		projected_text: Str::new(""),
		output_blob: None,
		output_shown_lines: 0,
		output_total_lines: 0,
	}
}

async fn prepare_payload<B: ReadBlobs>(
	result: SearchResult,
	roots: &[SearchRoot],
	requested_skip: u64,
	blobs: &B,
) -> Result<Payload, Fault> {
	let mut payload = make_payload(result, roots, requested_skip);
	let output = spill_truncated_text(render_payload(&payload), blobs)
		.await
		.map_err(|fault| Fault::Blob { message: fault.message().clone() })?;
	payload.projected_text = output.content;
	payload.output_blob = output.blob;
	payload.output_shown_lines = output.shown_lines;
	payload.output_total_lines = output.total_lines;
	Ok(payload)
}

fn render_payload(payload: &Payload) -> String {
	let mut output = String::new();
	if payload.files.is_empty() {
		if payload.multi_scope
			&& payload.skip > 0
			&& payload.total_files > 0
			&& payload.skip >= payload.total_files
		{
			let suffix = if payload.total_files_lower_bound {
				"+"
			} else {
				""
			};
			let _ = write!(
				output,
				"No more results ({}{} files total; skip={} is past the end)",
				payload.total_files, suffix, payload.skip
			);
		} else {
			output.push_str("No matches found");
		}
		append_notes(&mut output, &payload.notes);
		return output;
	}

	if payload.multi_scope {
		render_grouped_files(&mut output, &payload.files);
	} else {
		for (index, file) in payload.files.iter().enumerate() {
			if index > 0 {
				output.push_str("\n\n");
			}
			if let Some(tag) = &file.snapshot_tag {
				let _ = writeln!(output, "{}", format_hashline_header(&file.path, tag));
			}
			render_file_matches(&mut output, &file.matches);
		}
	}
	if payload.file_limit_reached {
		let next_skip = payload
			.skip
			.saturating_add(u64::try_from(payload.files.len()).unwrap_or(u64::MAX));
		let suffix = if payload.total_files_lower_bound {
			"+"
		} else {
			""
		};
		let _ = write!(
			output,
			"\n\nShowing files {}-{} of {}{}. Use skip={} for the next page, or narrow paths/pattern.",
			payload.skip.saturating_add(1),
			next_skip,
			payload.total_files,
			suffix,
			next_skip
		);
	}
	append_notes(&mut output, &payload.notes);
	output
}

fn render_grouped_files(output: &mut String, files: &[FileGroup]) {
	let tree = build_path_tree(
		files
			.iter()
			.map(|file| PathTreeInput::with_key(&file.path, false, &file.path)),
	);
	let by_path: HashMap<&str, &FileGroup> = files
		.iter()
		.map(|file| (file.path.as_ref(), file))
		.collect();
	let mut emitted = false;
	for event in walk_path_tree(&tree) {
		if emitted {
			if event.starts_group() {
				output.push_str("\n\n");
			} else {
				output.push('\n');
			}
		}
		emitted = true;
		for _ in 0..event.heading_level() {
			output.push('#');
		}
		output.push(' ');
		output.push_str(event.name);
		match event.kind {
			GroupedTreeEventKind::Directory => output.push('/'),
			GroupedTreeEventKind::File => {
				let file = by_path[event.key];
				if let Some(tag) = &file.snapshot_tag {
					output.push('#');
					output.push_str(tag);
				}
				output.push('\n');
				render_file_matches(output, &file.matches);
			},
		}
	}
}

fn render_file_matches(output: &mut String, matches: &[FileMatch]) {
	let mut last_emitted = None;
	for matched in matches {
		for context in &matched.context_before {
			push_match_line(output, &mut last_emitted, context.line_number, &context.line, false);
		}
		push_match_line(output, &mut last_emitted, matched.line_number, &matched.line, true);
		for context in &matched.context_after {
			push_match_line(output, &mut last_emitted, context.line_number, &context.line, false);
		}
	}
	if output.ends_with('\n') {
		output.pop();
	}
}

fn push_match_line(
	output: &mut String,
	last: &mut Option<u32>,
	number: u32,
	line: &str,
	matched: bool,
) {
	if last.is_some_and(|previous| number > previous.saturating_add(1)) {
		output.push_str("...\n");
	}
	let marker = if matched { '*' } else { ' ' };
	let _ = writeln!(output, "{marker}{number}:{line}");
	*last = Some(number);
}

fn append_notes(output: &mut String, notes: &[Str]) {
	if notes.is_empty() {
		return;
	}
	output.push_str("\n\n");
	for (index, note) in notes.iter().enumerate() {
		if index > 0 {
			output.push('\n');
		}
		output.push_str(note);
	}
}

fn join_strs(values: &[Str]) -> String {
	values
		.iter()
		.map(Str::as_str)
		.collect::<Vec<_>>()
		.join(", ")
}

fn done(result: Result<Payload, Fault>) -> Ev<Update, Payload, Fault> {
	let useless = result
		.as_ref()
		.is_ok_and(|payload| payload.files.is_empty() && payload.total_files == 0);
	Ev::Done(Outcome::Done { result, useless })
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
		example:  Some(Str::from("{\"pattern\":\"TODO\",\"path\":\"src\"}")),
		found:    Some(message),
	}
}
