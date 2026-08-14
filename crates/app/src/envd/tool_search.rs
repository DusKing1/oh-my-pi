//! App-owned workspace adapter for the generic grep and glob executors.

use std::{
	collections::{HashMap, HashSet},
	fmt,
	future::Future,
	path::{Component, Path, PathBuf},
	time::{Duration, Instant, UNIX_EPOCH},
};

use bytes::Bytes;
use omp_core::Str;
use omp_hashline::{RevisionToken, SnapshotStore};
use omp_tools::{
	glob::{self, WalkMatch, WalkResult},
	grep::{self, SearchMatch, SearchResult, SearchRoot, SearchRootKind, WorkspaceSearch},
	read::{ReadSources as _, archive, web},
};
use omp_walker::{
	CompiledWalkGlob, FileType, FollowLinks, SizeHintPolicy, WalkDecision, WalkDetail, WalkError,
	WalkFilter, WalkOrder, WalkRequest,
};
use tokio_util::sync::CancellationToken;

use super::{docs::DocumentHost, tool_read_sources::ReadSourceAdapter, workspace::WorkspaceHost};

const CANCELLED_REASON: &str = "workspace traversal future was dropped";
const SNAPSHOT_MAX_BYTES: u64 = 4 * 1024 * 1024;

/// Cloneable bridge from generic search tools to the app-owned workspace and
/// session document state.
#[derive(Clone, Debug)]
pub(crate) struct WorkspaceSearchAdapter {
	host:         WorkspaceHost,
	documents:    DocumentHost,
	read_sources: ReadSourceAdapter,
}

impl WorkspaceSearchAdapter {
	/// Wraps the concrete workspace owner and session document host used by the
	/// environment daemon.
	pub(crate) fn new(host: WorkspaceHost, documents: DocumentHost) -> Self {
		let read_sources = ReadSourceAdapter::new(documents.clone(), host.clone());
		Self { host, documents, read_sources }
	}

	/// Returns the document host that owns this session's shared hashline state.
	pub(crate) const fn documents(&self) -> &DocumentHost {
		&self.documents
	}
}

impl WorkspaceSearch for WorkspaceSearchAdapter {
	fn search(
		&self,
		request: grep::SearchRequest,
	) -> impl Future<Output = Result<SearchResult, grep::Fault>> + Send + '_ {
		let host = self.host.clone();
		let documents = self.documents().clone();
		let read_sources = self.read_sources.clone();
		async move {
			let cancel = CancellationToken::new();
			let cancel_on_drop = CancelOnDrop(cancel.clone());
			let deadline = Instant::now()
				.checked_add(Duration::from_millis(u64::from(request.timeout_ms)))
				.unwrap_or_else(Instant::now);
			let external =
				materialize_external_roots(&host, &read_sources, &request, deadline, &cancel).await?;
			let operation = tokio::task::spawn_blocking(move || {
				search_blocking(&host, &documents, request, external, deadline, &cancel)
			});
			let result = operation.await.map_err(|error| grep::Fault::Workspace {
				message: Str::from(format!("workspace search task failed: {error}")),
			})?;
			drop(cancel_on_drop);
			result
		}
	}

	fn glob(
		&self,
		request: glob::WalkRequest,
	) -> impl Future<Output = Result<WalkResult, glob::Fault>> + Send + '_ {
		let host = self.host.clone();
		async move {
			let cancel = CancellationToken::new();
			let cancel_on_drop = CancelOnDrop(cancel.clone());
			let operation =
				tokio::task::spawn_blocking(move || glob_blocking(&host, request, &cancel));
			let result = operation.await.map_err(|error| glob::Fault::Workspace {
				message: Str::from(format!("workspace walk task failed: {error}")),
			})?;
			drop(cancel_on_drop);
			result
		}
	}
}

struct CancelOnDrop(CancellationToken);

impl Drop for CancelOnDrop {
	fn drop(&mut self) {
		self.0.cancel();
	}
}

#[derive(Debug)]
enum GrepTarget {
	Filesystem { root_index: u64, path: PathBuf, glob: Option<Str>, is_file: bool },
	Memory(MemorySearchTarget),
}

#[derive(Debug)]
struct MemorySearchTarget {
	root_index: u64,
	source_key: Str,
	path:       Str,
	content:    Bytes,
}

#[derive(Debug, Default)]
struct ExternalMaterialization {
	by_root:            HashMap<usize, Vec<MemorySearchTarget>>,
	archive_unreadable: Vec<Str>,
}

async fn materialize_external_roots(
	host: &WorkspaceHost,
	sources: &ReadSourceAdapter,
	request: &grep::SearchRequest,
	deadline: Instant,
	cancel: &CancellationToken,
) -> Result<ExternalMaterialization, grep::Fault> {
	let mut materialized = ExternalMaterialization::default();
	for (root_index, root) in request.roots.iter().enumerate() {
		check_grep_cancel(cancel)?;
		let remaining =
			Duration::from_millis(u64::from(remaining_millis(deadline).ok_or(grep::Fault::TimedOut)?));
		match root.kind {
			SearchRootKind::Filesystem => {},
			SearchRootKind::Archive => {
				let archive = tokio::time::timeout(
					remaining,
					materialize_archive_root(host, sources, root, root_index),
				)
				.await
				.map_err(|_| grep::Fault::TimedOut)?;
				match archive {
					Ok((targets, unreadable)) => {
						if !targets.is_empty() {
							materialized.by_root.insert(root_index, targets);
						}
						materialized.archive_unreadable.extend(unreadable);
					},
					Err(message) => materialized
						.archive_unreadable
						.push(Str::from(format!("{} ({message})", root.path))),
				}
			},
			SearchRootKind::Url => {
				let target =
					tokio::time::timeout(remaining, materialize_url_root(sources, root, root_index))
						.await
						.map_err(|_| grep::Fault::TimedOut)??;
				materialized.by_root.insert(root_index, vec![target]);
			},
		}
	}
	remaining_millis(deadline).ok_or(grep::Fault::TimedOut)?;
	Ok(materialized)
}

async fn materialize_archive_root(
	host: &WorkspaceHost,
	sources: &ReadSourceAdapter,
	root: &SearchRoot,
	root_index: usize,
) -> Result<(Vec<MemorySearchTarget>, Vec<Str>), String> {
	let candidates = archive::parse_archive_path_candidates(root.path.as_str());
	let mut last_error = None;
	for candidate in candidates
		.into_iter()
		.filter(|candidate| !candidate.sub_path.is_empty())
	{
		let archive_path = resolve_input_path(host.root(), &candidate.archive_path);
		let archive_path = tokio::fs::canonicalize(&archive_path)
			.await
			.unwrap_or(archive_path);
		let source_path = Str::from(archive_path.to_string_lossy().into_owned());
		let bytes = match sources.read_bytes(source_path).await {
			Ok(bytes) => bytes,
			Err(error) => {
				last_error = Some(error.message().to_string());
				continue;
			},
		};
		let archive_key = Str::from(archive_path.to_string_lossy().into_owned());
		let root = root.clone();
		return tokio::task::spawn_blocking(move || {
			materialize_archive_bytes(&root, root_index, candidate, archive_key, bytes)
		})
		.await
		.map_err(|error| format!("archive materialization task failed: {error}"))?;
	}
	Err(format!(
		"cannot open archive: {}",
		last_error.unwrap_or_else(|| "archive path could not be resolved".to_owned())
	))
}

fn materialize_archive_bytes(
	root: &SearchRoot,
	root_index: usize,
	candidate: archive::ArchivePathCandidate,
	archive_key: Str,
	bytes: Bytes,
) -> Result<(Vec<MemorySearchTarget>, Vec<Str>), String> {
	let format = archive::archive_format_from_path(&candidate.archive_path)
		.or_else(|| archive::sniff_archive_format(&bytes))
		.ok_or_else(|| "cannot determine archive format".to_owned())?;
	let contents = archive::open_archive_bytes(bytes, format)
		.map_err(|error| format!("cannot open archive: {error}"))?
		.materialize_text_members()
		.map_err(|error| format!("cannot read archive: {error}"))?;
	let selected = candidate.sub_path.trim_matches('/');
	let selected_prefix = format!("{selected}/");
	let matches_selected = |path: &str| path == selected || path.starts_with(&selected_prefix);
	let root_index = u64::try_from(root_index).unwrap_or(u64::MAX);
	let mut targets = Vec::new();
	for member in contents
		.members
		.into_iter()
		.filter(|member| matches_selected(&member.path))
	{
		let display = archive_display_path(root.path.as_str(), &candidate, selected, &member.path);
		targets.push(MemorySearchTarget {
			root_index,
			source_key: Str::from(format!("archive:{archive_key}:{}", member.path)),
			path: display,
			content: Bytes::from(member.text),
		});
	}
	let mut unreadable = Vec::new();
	for member in contents
		.binary_members
		.into_iter()
		.filter(|member| matches_selected(&member.node.path))
	{
		let display =
			archive_display_path(root.path.as_str(), &candidate, selected, &member.node.path);
		unreadable.push(Str::from(format!("{display} (binary archive entry)")));
	}
	if targets.is_empty() && unreadable.is_empty() {
		unreadable.push(Str::from(format!("{} (archive entry not found)", root.path)));
	}
	Ok((targets, unreadable))
}

fn archive_display_path(
	authored: &str,
	candidate: &archive::ArchivePathCandidate,
	selected: &str,
	member: &str,
) -> Str {
	if selected == member {
		Str::from(authored)
	} else {
		Str::from(format!("{}:{member}", candidate.archive_path))
	}
}

async fn materialize_url_root<C: web::types::HttpClient + Sync>(
	client: &C,
	root: &SearchRoot,
	root_index: usize,
) -> Result<MemorySearchTarget, grep::Fault> {
	let parsed = web::parse_target(root.path.as_str())
		.map_err(grep_workspace_message)?
		.ok_or_else(|| grep_workspace_message(format!("invalid URL: {}", root.path)))?;
	let rendered = web::read(client, &parsed.url, false)
		.await
		.map_err(grep_workspace_message)?;
	Ok(MemorySearchTarget {
		root_index: u64::try_from(root_index).unwrap_or(u64::MAX),
		source_key: Str::from(format!("url:{}", parsed.url)),
		path:       root.path.clone(),
		content:    Bytes::from(rendered.content),
	})
}

#[derive(Debug)]
struct PendingSnapshot {
	path:       PathBuf,
	seen_lines: HashSet<usize>,
}

fn search_blocking(
	host: &WorkspaceHost,
	documents: &DocumentHost,
	request: grep::SearchRequest,
	mut external: ExternalMaterialization,
	deadline: Instant,
	cancel: &CancellationToken,
) -> Result<SearchResult, grep::Fault> {
	check_grep_cancel(cancel)?;
	let mut targets = Vec::new();
	let mut missing_paths = Vec::new();
	for (root_index, root) in request.roots.iter().enumerate() {
		match root.kind {
			SearchRootKind::Archive | SearchRootKind::Url => {
				targets.extend(
					external
						.by_root
						.remove(&root_index)
						.unwrap_or_default()
						.into_iter()
						.map(GrepTarget::Memory),
				);
			},
			SearchRootKind::Filesystem => {
				let literal_original = if root.original != root.path {
					resolve_literal_grep_target(host, root.original.as_str())?
				} else {
					None
				};
				let literal_won = literal_original.is_some();
				match literal_original.or(resolve_grep_target(host, root.path.as_str())?) {
					Some(GrepTarget::Filesystem { path, glob, is_file, .. }) => {
						targets.push(GrepTarget::Filesystem {
							root_index: if literal_won {
								u64::MAX
							} else {
								u64::try_from(root_index).unwrap_or(u64::MAX)
							},
							path,
							glob,
							is_file,
						})
					},
					Some(GrepTarget::Memory(_)) => unreachable!("filesystem resolver returned memory"),
					None => missing_paths.push(root.original.clone()),
				}
			},
		}
	}
	if targets.is_empty() && external.archive_unreadable.is_empty() {
		return Err(grep::Fault::AllPathsMissing { paths: missing_paths });
	}

	let memory_targets = targets
		.iter()
		.filter(|target| matches!(target, GrepTarget::Memory(_)))
		.count();
	let multi_scope = request.roots.len() > 1
		|| memory_targets > 1
		|| targets.iter().any(|target| {
			matches!(
				target,
				GrepTarget::Filesystem { is_file: false, .. }
					| GrepTarget::Filesystem { glob: Some(_), .. }
			)
		});
	let per_file_cap = if multi_scope {
		request.multi_file_max_count
	} else {
		request.single_file_max_count
	};
	let mut remaining = request.max_count;
	let mut matches = Vec::new();
	let mut limit_reached = false;
	let mut skipped_oversized = 0_u32;
	let mut oversized_files = Vec::new();
	let mut pending_snapshots: HashMap<Str, PendingSnapshot> = HashMap::new();

	for target in &targets {
		check_grep_cancel(cancel)?;
		if remaining == 0 {
			limit_reached = true;
			break;
		}
		let timeout_ms = remaining_millis(deadline).ok_or(grep::Fault::TimedOut)?;
		let (display_path, glob) = match target {
			GrepTarget::Filesystem { path, glob, is_file, .. } => {
				if *is_file
					&& std::fs::metadata(path)
						.is_ok_and(|metadata| metadata.len() > omp_grep::MAX_FILE_BYTES)
				{
					oversized_files.push(workspace_relative(host.root(), path)?);
				}
				(Str::from(path.to_string_lossy().into_owned()), glob.clone())
			},
			GrepTarget::Memory(memory) => (memory.path.clone(), None),
		};
		let options = omp_grep::GrepOptions {
			pattern: request.pattern.clone(),
			path: display_path,
			glob,
			ignore_case: request.ignore_case,
			multiline: request.multiline,
			hidden: request.hidden,
			gitignore: request.gitignore,
			max_count: Some(remaining),
			max_count_per_file: Some(per_file_cap),
			context_before: request.context_before,
			context_after: request.context_after,
			max_columns: Some(request.max_columns),
			mode: omp_grep::GrepOutputMode::Content,
			timeout_ms: Some(timeout_ms),
		};
		let native = match target {
			GrepTarget::Filesystem { .. } => {
				omp_grep::grep(&options).map_err(map_native_grep_fault)?
			},
			GrepTarget::Memory(memory) => {
				omp_grep::search(&memory.content, &options).map_err(map_native_grep_fault)?
			},
		};
		skipped_oversized = skipped_oversized.saturating_add(native.skipped_oversized);
		limit_reached |= native.limit_reached;
		remaining = remaining.saturating_sub(u32::try_from(native.matches.len()).unwrap_or(u32::MAX));

		for matched in native.matches {
			check_grep_cancel(cancel)?;
			let context_before: Vec<_> = matched
				.context_before
				.into_iter()
				.map(|line| grep::ContextLine { line_number: line.line_number, line: line.line })
				.collect();
			let context_after: Vec<_> = matched
				.context_after
				.into_iter()
				.map(|line| grep::ContextLine { line_number: line.line_number, line: line.line })
				.collect();
			let (source_key, path, root_index) = match target {
				GrepTarget::Filesystem { root_index, path, is_file, .. } => {
					let source_path = if *is_file {
						path.clone()
					} else {
						path.join(matched.path.as_str())
					};
					let canonical = std::fs::canonicalize(&source_path).unwrap_or(source_path);
					let source_key = Str::from(canonical.to_string_lossy().into_owned());
					let pending = pending_snapshots
						.entry(source_key.clone())
						.or_insert_with(|| PendingSnapshot {
							path:       canonical.clone(),
							seen_lines: HashSet::new(),
						});
					retain_snapshot_lines(
						&mut pending.seen_lines,
						matched.line_number,
						&context_before,
						&context_after,
					);
					(source_key, workspace_relative(host.root(), &canonical)?, *root_index)
				},
				GrepTarget::Memory(memory) => {
					(memory.source_key.clone(), memory.path.clone(), memory.root_index)
				},
			};
			matches.push(SearchMatch {
				source_key,
				path,
				root_index,
				line_number: matched.line_number,
				line: matched.line,
				truncated: matched.truncated,
				context_before,
				context_after,
				snapshot_tag: None,
			});
		}
	}
	let snapshot_tags: HashMap<_, _> = pending_snapshots
		.into_iter()
		.map(|(source_key, pending)| {
			let tag = record_grep_snapshot(documents, &pending.path, pending.seen_lines.into_iter());
			(source_key, tag)
		})
		.collect();
	for matched in &mut matches {
		matched.snapshot_tag = snapshot_tags.get(&matched.source_key).cloned().flatten();
	}
	check_grep_cancel(cancel)?;
	oversized_files.sort_unstable();
	oversized_files.dedup();
	Ok(SearchResult {
		matches,
		multi_scope,
		limit_reached,
		skipped_oversized,
		missing_paths,
		archive_unreadable: external.archive_unreadable,
		oversized_files,
	})
}

fn resolve_literal_grep_target(
	host: &WorkspaceHost,
	input: &str,
) -> Result<Option<GrepTarget>, grep::Fault> {
	let input = normalize_input(input);
	let literal = resolve_input_path(host.root(), &input);
	let metadata = match std::fs::metadata(&literal) {
		Ok(metadata) => metadata,
		Err(error) if is_missing(&error) => return Ok(None),
		Err(error) => return Err(grep_workspace_message(error)),
	};
	let canonical = std::fs::canonicalize(&literal).map_err(grep_workspace_message)?;
	Ok(Some(GrepTarget::Filesystem {
		root_index: 0,
		path:       canonical,
		glob:       None,
		is_file:    metadata.is_file(),
	}))
}

fn resolve_grep_target(
	host: &WorkspaceHost,
	input: &str,
) -> Result<Option<GrepTarget>, grep::Fault> {
	let input = normalize_input(input);
	let literal = resolve_input_path(host.root(), &input);
	match std::fs::metadata(&literal) {
		Ok(metadata) => {
			let canonical = std::fs::canonicalize(&literal).map_err(grep_workspace_message)?;
			return Ok(Some(GrepTarget::Filesystem {
				root_index: 0,
				path:       canonical,
				glob:       None,
				is_file:    metadata.is_file(),
			}));
		},
		Err(error) if is_missing(&error) => {},
		Err(error) => return Err(grep_workspace_message(error)),
	}
	let Some(parsed) = parse_glob_path(&input) else {
		return Ok(None);
	};
	let base = resolve_input_path(host.root(), &parsed.base);
	let metadata = match std::fs::metadata(&base) {
		Ok(metadata) => metadata,
		Err(error) if is_missing(&error) => return Ok(None),
		Err(error) => return Err(grep_workspace_message(error)),
	};
	let canonical = std::fs::canonicalize(&base).map_err(grep_workspace_message)?;
	if !metadata.is_dir() {
		return Ok(None);
	}
	Ok(Some(GrepTarget::Filesystem {
		root_index: 0,
		path:       canonical,
		glob:       Some(parsed.pattern.into()),
		is_file:    false,
	}))
}

fn retain_snapshot_lines(
	seen_lines: &mut HashSet<usize>,
	match_line: u32,
	context_before: &[grep::ContextLine],
	context_after: &[grep::ContextLine],
) {
	seen_lines.extend(
		std::iter::once(match_line)
			.chain(context_before.iter().map(|line| line.line_number))
			.chain(context_after.iter().map(|line| line.line_number))
			.filter_map(|line| usize::try_from(line).ok())
			.filter(|line| *line != 0),
	);
}

fn record_grep_snapshot(
	documents: &DocumentHost,
	path: &Path,
	seen_lines: impl IntoIterator<Item = usize>,
) -> Option<Str> {
	let metadata = std::fs::metadata(path).ok()?;
	if metadata.len() > SNAPSHOT_MAX_BYTES {
		return None;
	}
	let bytes = std::fs::read(path).ok()?;
	let revision = RevisionToken::new(blake3::hash(&bytes).as_bytes());
	record_snapshot(
		&mut documents.snapshot_store().lock(),
		path,
		revision,
		Bytes::from(bytes),
		seen_lines,
	)
}

fn record_snapshot(
	store: &mut SnapshotStore,
	path: &Path,
	revision: RevisionToken,
	bytes: Bytes,
	seen_lines: impl IntoIterator<Item = usize>,
) -> Option<Str> {
	store
		.record(Str::from(path.to_string_lossy().into_owned()), revision, bytes, seen_lines)
		.ok()
}

fn map_native_grep_fault(error: omp_grep::GrepError) -> grep::Fault {
	match error {
		omp_grep::GrepError::InvalidRegex { regex, pcre2 } => {
			let message = format!("{regex}; PCRE2 fallback: {pcre2}");
			grep::Fault::InvalidRegex { message: Str::from(strip_regex_error_prefix(&message)) }
		},
		omp_grep::GrepError::Timeout { .. } => grep::Fault::TimedOut,
		omp_grep::GrepError::PathNotFound { path } => {
			grep::Fault::AllPathsMissing { paths: vec![path] }
		},
		omp_grep::GrepError::InvalidGlob { message }
		| omp_grep::GrepError::Walk { message }
		| omp_grep::GrepError::Search { message } => grep::Fault::Workspace { message },
	}
}
fn strip_regex_error_prefix(message: &str) -> &str {
	for prefix in ["regex parse error:", "regex error:"] {
		if message
			.get(..prefix.len())
			.is_some_and(|candidate| candidate.eq_ignore_ascii_case(prefix))
		{
			return message[prefix.len()..].trim_start();
		}
	}
	message
}

fn remaining_millis(deadline: Instant) -> Option<u32> {
	let remaining = deadline.checked_duration_since(Instant::now())?;
	let millis = remaining.as_millis().clamp(1, u128::from(u32::MAX));
	Some(u32::try_from(millis).unwrap_or(u32::MAX))
}

fn check_grep_cancel(cancel: &CancellationToken) -> Result<(), grep::Fault> {
	if cancel.is_cancelled() {
		Err(grep::Fault::Cancelled { reason: Str::from(CANCELLED_REASON) })
	} else {
		Ok(())
	}
}

#[derive(Debug)]
struct ParsedGlob {
	base:    String,
	pattern: String,
}

fn parse_glob_path(input: &str) -> Option<ParsedGlob> {
	let normalized = input.replace('\\', "/");
	let segments: Vec<&str> = normalized.split('/').collect();
	let first_glob = segments
		.iter()
		.position(|segment| has_glob_chars(segment))?;
	let base = if first_glob == 0 {
		".".to_owned()
	} else {
		segments[..first_glob].join("/")
	};
	Some(ParsedGlob { base, pattern: segments[first_glob..].join("/") })
}

#[derive(Debug)]
struct FindPattern {
	base:     String,
	pattern:  String,
	has_glob: bool,
}

fn parse_find_pattern(input: &str) -> FindPattern {
	let normalized = input.replace('\\', "/");
	let segments: Vec<&str> = normalized.split('/').collect();
	let Some(first_glob) = segments.iter().position(|segment| has_glob_chars(segment)) else {
		return FindPattern { base: normalized, pattern: "**/*".to_owned(), has_glob: false };
	};
	if first_glob == 0 {
		let pattern = if normalized.starts_with("**/") {
			normalized
		} else {
			format!("**/{normalized}")
		};
		return FindPattern { base: ".".to_owned(), pattern, has_glob: true };
	}
	FindPattern {
		base:     segments[..first_glob].join("/"),
		pattern:  segments[first_glob..].join("/"),
		has_glob: true,
	}
}

fn glob_blocking(
	host: &WorkspaceHost,
	request: glob::WalkRequest,
	cancel: &CancellationToken,
) -> Result<WalkResult, glob::Fault> {
	let inputs = split_glob_inputs(host, request.path.as_str())?;
	let multi_target = inputs.len() > 1;
	let mut missing_paths = Vec::new();
	let mut targets = Vec::new();
	let mut found_paths = 0_usize;

	for input in inputs {
		check_glob_cancel(cancel)?;
		if input.bytes().all(|byte| byte == b'/') {
			return Err(glob::Fault::RootSearch);
		}
		let literal_path = resolve_input_path(host.root(), &input);
		let (parsed, metadata, target_path) = match std::fs::metadata(&literal_path) {
			Ok(metadata) => (
				FindPattern { base: input.clone(), pattern: "**/*".to_owned(), has_glob: false },
				metadata,
				literal_path,
			),
			Err(error) if is_missing(&error) => {
				let parsed = parse_find_pattern(&input);
				if parsed.base.bytes().all(|byte| byte == b'/') {
					return Err(glob::Fault::RootSearch);
				}
				let target_path = resolve_input_path(host.root(), &parsed.base);
				match std::fs::metadata(&target_path) {
					Ok(metadata) => (parsed, metadata, target_path),
					Err(error) if is_missing(&error) => {
						missing_paths.push(Str::from(input));
						continue;
					},
					Err(error) => return Err(glob_workspace_message(error)),
				}
			},
			Err(error) => return Err(glob_workspace_message(error)),
		};
		let canonical = std::fs::canonicalize(&target_path).map_err(glob_workspace_message)?;
		found_paths = found_paths.saturating_add(1);
		if (!metadata.is_file() && !metadata.is_dir()) || (parsed.has_glob && !metadata.is_dir()) {
			if multi_target {
				continue;
			}
			return Err(glob::Fault::PathNotDirectory { path: Str::from(input) });
		}
		targets.push(GlobTarget { parsed, metadata, canonical });
	}
	if targets.is_empty() && found_paths == 0 {
		return Err(glob::Fault::PathNotFound { paths: missing_paths });
	}

	let deadline = Instant::now()
		.checked_add(Duration::from_millis(request.timeout_ms))
		.unwrap_or_else(Instant::now);
	let mut matches = Vec::new();
	let mut truncated = false;
	let mut timed_out = false;
	for target in targets {
		check_glob_cancel(cancel)?;
		if Instant::now() >= deadline {
			timed_out = true;
			break;
		}
		if !target.parsed.has_glob && target.metadata.is_file() {
			matches.push(WalkMatch {
				path:        workspace_relative_glob(host.root(), &target.canonical)?,
				modified_ms: modified_millis(&target.metadata),
				is_dir:      false,
			});
			continue;
		}
		let compiled = CompiledWalkGlob::new([target.parsed.pattern.as_str()]).map_err(|error| {
			glob::Fault::InvalidPattern {
				pattern: Str::from(target.parsed.pattern.clone()),
				message: Str::from(error.to_string()),
			}
		})?;
		let mentions_node_modules = target.parsed.pattern.contains("node_modules");
		let max_depth = glob_max_depth(&target.parsed.pattern);
		let walk = WalkRequest::new(&target.canonical)
			.hidden(request.hidden)
			.gitignore(request.gitignore)
			.skip_git(true)
			.skip_node_modules(!mentions_node_modules)
			.follow_links(FollowLinks::Never)
			.detail(WalkDetail::Full)
			.size_hints(SizeHintPolicy::Always)
			.order(WalkOrder::Unordered)
			.emit_root(false)
			.depth(1, max_depth)
			.filter(WalkFilter::all().glob(compiled))
			.cache(false);
		let outcome = walk_glob_target(host.root(), &walk, request.limit, deadline, cancel)?;
		matches.extend(outcome.matches);
		truncated |= outcome.truncated;
		if outcome.timed_out {
			timed_out = true;
			break;
		}
	}

	matches.sort_by(|left, right| {
		right
			.modified_ms
			.cmp(&left.modified_ms)
			.then_with(|| left.path.cmp(&right.path))
	});
	let mut seen = HashSet::with_capacity(matches.len());
	matches.retain(|entry| seen.insert(entry.path.clone()));
	let retain = usize::try_from(request.limit).unwrap_or(usize::MAX);
	if matches.len() > retain {
		truncated = true;
		matches.truncate(retain);
	}
	Ok(WalkResult { matches, missing_paths, timed_out, truncated })
}

#[derive(Debug)]
struct GlobTarget {
	parsed:    FindPattern,
	metadata:  std::fs::Metadata,
	canonical: PathBuf,
}

#[derive(Debug)]
struct GlobTargetOutcome {
	matches:   Vec<WalkMatch>,
	truncated: bool,
	timed_out: bool,
}

#[derive(Clone, Debug)]
enum WalkStop {
	Cancelled,
	TimedOut,
	Workspace(Str),
}

impl fmt::Display for WalkStop {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::Cancelled => formatter.write_str("cancelled"),
			Self::TimedOut => formatter.write_str("timed out"),
			Self::Workspace(message) => formatter.write_str(message),
		}
	}
}

fn walk_glob_target(
	workspace_root: &Path,
	request: &WalkRequest,
	limit: u64,
	deadline: Instant,
	cancel: &CancellationToken,
) -> Result<GlobTargetOutcome, glob::Fault> {
	let keep = usize::try_from(limit).unwrap_or(usize::MAX);
	let mut matches = Vec::with_capacity(keep.saturating_add(1).min(201));
	let result = request.for_each_entry_with_heartbeat(
		|| {
			if cancel.is_cancelled() {
				Err(WalkStop::Cancelled)
			} else if Instant::now() >= deadline {
				Err(WalkStop::TimedOut)
			} else {
				Ok(())
			}
		},
		|entry| {
			let mut path = workspace_relative_raw(workspace_root, &entry.absolute_path)
				.map_err(|error| WalkStop::Workspace(Str::from(error.to_string())))?;
			let is_dir = entry.file_type == FileType::Dir;
			if is_dir {
				path.push('/');
			}
			retain_ranked(
				&mut matches,
				WalkMatch {
					path: Str::from(path),
					modified_ms: entry.mtime.map_or(0, float_millis),
					is_dir,
				},
				keep,
			);
			Ok(WalkDecision::Include)
		},
		|_| Ok(WalkDecision::Include),
	);
	let timed_out = match result {
		Ok(_) => Instant::now() >= deadline,
		Err(WalkError::Interrupted(WalkStop::TimedOut)) => true,
		Err(WalkError::Interrupted(WalkStop::Cancelled)) => return Err(cancelled_glob()),
		Err(WalkError::Interrupted(WalkStop::Workspace(message))) => {
			return Err(glob::Fault::Workspace { message });
		},
		Err(error) => {
			return Err(glob::Fault::Workspace { message: Str::from(error.to_string()) });
		},
	};
	matches.sort_by(|left, right| {
		right
			.modified_ms
			.cmp(&left.modified_ms)
			.then_with(|| left.path.cmp(&right.path))
	});
	let truncated = matches.len() > keep;
	matches.truncate(keep);
	Ok(GlobTargetOutcome { matches, truncated, timed_out })
}

fn retain_ranked(matches: &mut Vec<WalkMatch>, candidate: WalkMatch, limit: usize) {
	let capacity = limit.saturating_add(1);
	if matches.len() < capacity {
		matches.push(candidate);
		return;
	}
	let Some((worst, _)) = matches
		.iter()
		.enumerate()
		.min_by(|(_, left), (_, right)| compare_glob_rank(left, right))
	else {
		return;
	};
	if compare_glob_rank(&candidate, &matches[worst]).is_gt() {
		matches[worst] = candidate;
	}
}

fn compare_glob_rank(left: &WalkMatch, right: &WalkMatch) -> std::cmp::Ordering {
	left
		.modified_ms
		.cmp(&right.modified_ms)
		.then_with(|| right.path.cmp(&left.path))
}

fn split_glob_inputs(host: &WorkspaceHost, raw: &str) -> Result<Vec<String>, glob::Fault> {
	let normalized = normalize_input(raw);
	if normalized.is_empty() {
		return Err(glob::Fault::EmptyPath);
	}
	if !normalized.contains(';')
		|| std::fs::metadata(resolve_input_path(host.root(), &normalized)).is_ok()
	{
		return Ok(vec![normalized]);
	}
	let inputs: Vec<String> = normalized
		.split(';')
		.map(normalize_input)
		.filter(|entry| !entry.is_empty())
		.collect();
	if inputs.is_empty() {
		Err(glob::Fault::EmptyPath)
	} else {
		Ok(inputs)
	}
}

fn glob_max_depth(pattern: &str) -> usize {
	if pattern.split('/').any(|segment| segment == "**") {
		usize::MAX
	} else {
		pattern
			.split('/')
			.filter(|segment| !segment.is_empty())
			.count()
			.max(1)
	}
}

fn has_glob_chars(segment: &str) -> bool {
	segment
		.bytes()
		.any(|byte| matches!(byte, b'*' | b'?' | b'[' | b'{'))
}

fn normalize_input(input: &str) -> String {
	let trimmed = input.trim();
	trimmed
		.strip_prefix('"')
		.and_then(|value| value.strip_suffix('"'))
		.unwrap_or(trimmed)
		.to_owned()
}

fn resolve_input_path(root: &Path, input: &str) -> PathBuf {
	let path = Path::new(input);
	if path.is_absolute() {
		path.to_path_buf()
	} else {
		root.join(path)
	}
}

fn is_missing(error: &std::io::Error) -> bool {
	matches!(
		error.kind(),
		std::io::ErrorKind::NotFound
			| std::io::ErrorKind::NotADirectory
			| std::io::ErrorKind::InvalidInput
	)
}

fn workspace_relative(root: &Path, path: &Path) -> Result<Str, grep::Fault> {
	workspace_relative_raw(root, path)
		.map(Str::from)
		.map_err(grep_workspace_message)
}

fn workspace_relative_glob(root: &Path, path: &Path) -> Result<Str, glob::Fault> {
	workspace_relative_raw(root, path)
		.map(Str::from)
		.map_err(glob_workspace_message)
}

fn workspace_relative_raw(root: &Path, path: &Path) -> Result<String, std::io::Error> {
	let Ok(relative) = path.strip_prefix(root) else {
		return Ok(path.to_string_lossy().replace('\\', "/"));
	};
	let mut normalized = String::new();
	for component in relative.components() {
		match component {
			Component::CurDir => {},
			Component::Normal(component) => {
				if !normalized.is_empty() {
					normalized.push('/');
				}
				normalized.push_str(&component.to_string_lossy());
			},
			Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
				return Err(std::io::Error::new(
					std::io::ErrorKind::PermissionDenied,
					"path is outside the workspace",
				));
			},
		}
	}
	Ok(normalized)
}

fn modified_millis(metadata: &std::fs::Metadata) -> u64 {
	metadata
		.modified()
		.ok()
		.and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
		.map_or(0, duration_millis)
}

fn duration_millis(duration: Duration) -> u64 {
	u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn float_millis(value: f64) -> u64 {
	if value.is_finite() && value > 0.0 {
		value.min(u64::MAX as f64) as u64
	} else {
		0
	}
}

fn grep_workspace_message(error: impl fmt::Display) -> grep::Fault {
	grep::Fault::Workspace { message: Str::from(error.to_string()) }
}

fn glob_workspace_message(error: impl fmt::Display) -> glob::Fault {
	glob::Fault::Workspace { message: Str::from(error.to_string()) }
}

fn check_glob_cancel(cancel: &CancellationToken) -> Result<(), glob::Fault> {
	if cancel.is_cancelled() {
		Err(cancelled_glob())
	} else {
		Ok(())
	}
}

fn cancelled_glob() -> glob::Fault {
	glob::Fault::Cancelled { reason: Str::from(CANCELLED_REASON) }
}

#[cfg(test)]
mod tests {
	use omp_docserver::{
		Environment, ServerConfig,
		connection::{ConnectionConfig, serve_connection},
	};
	use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

	use super::*;

	async fn connected_search_adapter(root: &Path) -> WorkspaceSearchAdapter {
		let config = ServerConfig::new(root).expect("docserver config");
		let environment = Environment::new(config).expect("document authority");
		let (client_stream, server_stream) = tokio::io::duplex(256 * 1024);
		tokio::spawn(serve_connection(environment, server_stream, ConnectionConfig::default()));
		let documents = DocumentHost::connect(client_stream)
			.await
			.expect("document hello");
		let workspace = WorkspaceHost::open(root).expect("workspace host");
		WorkspaceSearchAdapter::new(workspace, documents)
	}

	fn search_request(
		path: impl Into<Str>,
		kind: SearchRootKind,
		timeout_ms: u32,
	) -> grep::SearchRequest {
		let path = path.into();
		grep::SearchRequest {
			pattern: Str::from("needle"),
			roots: vec![SearchRoot { original: path.clone(), path, kind, ranges: Box::default() }],
			ignore_case: false,
			multiline: false,
			gitignore: false,
			hidden: true,
			max_count: 2_000,
			single_file_max_count: 200,
			multi_file_max_count: 20,
			context_before: 0,
			context_after: 0,
			max_columns: 512,
			timeout_ms,
		}
	}

	#[test]
	fn find_patterns_preserve_non_recursive_scopes() {
		let parsed = parse_find_pattern("src/*");
		assert_eq!(parsed.base, "src");
		assert_eq!(parsed.pattern, "*");
		assert!(parsed.has_glob);
	}

	#[test]
	fn leading_glob_patterns_become_recursive() {
		let parsed = parse_find_pattern("*.rs");
		assert_eq!(parsed.base, ".");
		assert_eq!(parsed.pattern, "**/*.rs");
	}

	#[test]
	fn ranked_retention_keeps_newest_then_lexical() {
		let mut matches = Vec::new();
		for (path, modified_ms) in [("b", 1), ("c", 2), ("a", 2)] {
			retain_ranked(
				&mut matches,
				WalkMatch { path: Str::from(path), modified_ms, is_dir: false },
				1,
			);
		}
		matches.sort_by(|left, right| compare_glob_rank(right, left));
		assert_eq!(matches.len(), 2);
		assert_eq!(matches[0].path, "a");
	}

	#[test]
	fn grep_accepts_an_external_file_and_uses_its_absolute_display_path() {
		let parent = tempfile::tempdir().expect("parent directory");
		let workspace = parent.path().join("workspace");
		let external = parent.path().join("external.txt");
		std::fs::create_dir(&workspace).expect("workspace directory");
		std::fs::write(&external, "alpha\nneedle\nomega\n").expect("external file");
		let host = WorkspaceHost::open(&workspace).expect("workspace host");
		let target = resolve_grep_target(&host, external.to_str().expect("UTF-8 external path"))
			.expect("resolve target")
			.expect("external target");
		let GrepTarget::Filesystem { path, .. } = target else {
			panic!("external file resolved to memory");
		};
		let result = omp_grep::grep(&omp_grep::GrepOptions {
			pattern: Str::from("needle"),
			path: Str::from(path.to_string_lossy().into_owned()),
			..omp_grep::GrepOptions::default()
		})
		.expect("grep external file");

		assert_eq!(result.matches.len(), 1);
		assert_eq!(
			workspace_relative(&host.root(), &path).expect("display path"),
			Str::from(path.to_string_lossy().replace('\\', "/")),
		);
	}

	#[test]
	fn glob_accepts_an_external_directory_and_returns_absolute_paths() {
		let parent = tempfile::tempdir().expect("parent directory");
		let workspace = parent.path().join("workspace");
		let external = parent.path().join("external");
		std::fs::create_dir(&workspace).expect("workspace directory");
		std::fs::create_dir(&external).expect("external directory");
		let source = external.join("lib.rs");
		std::fs::write(&source, "fn external() {}\n").expect("external source");
		let host = WorkspaceHost::open(&workspace).expect("workspace host");
		let result = glob_blocking(
			&host,
			glob::WalkRequest {
				path:       Str::from(format!("{}/**/*.rs", external.to_string_lossy())),
				hidden:     true,
				gitignore:  false,
				limit:      200,
				timeout_ms: 5_000,
			},
			&CancellationToken::new(),
		)
		.expect("glob external directory");

		assert_eq!(result.matches.len(), 1);
		assert_eq!(result.matches[0].path, Str::from(source.to_string_lossy().replace('\\', "/")),);
	}

	#[test]
	fn grep_snapshot_retains_match_and_context_line_provenance() {
		let before =
			[grep::ContextLine { line_number: 3, line: Str::from("before one") }, grep::ContextLine {
				line_number: 4,
				line:        Str::from("before two"),
			}];
		let after = [grep::ContextLine { line_number: 6, line: Str::from("after") }];
		let mut seen_lines = HashSet::new();
		retain_snapshot_lines(&mut seen_lines, 5, &before, &after);

		let path = Path::new("/workspace/file.rs");
		let bytes = Bytes::from_static(b"one\ntwo\nthree\nfour\nmatch\nsix\nseven\n");
		let revision = RevisionToken::new(blake3::hash(&bytes).as_bytes());
		let mut store = SnapshotStore::default();
		let tag = record_snapshot(&mut store, path, revision.clone(), bytes.clone(), seen_lines)
			.expect("snapshot tag");
		let snapshot = store
			.resolve(path.to_str().expect("UTF-8 test path"), &tag, Some(&revision))
			.expect("retained snapshot");

		assert_eq!(snapshot.path(), "/workspace/file.rs");
		assert_eq!(snapshot.revision(), &revision);
		assert_eq!(snapshot.bytes(), &bytes);
		assert!(snapshot.seen_lines().contains(&3));
		assert!(snapshot.seen_lines().contains(&4));
		assert!(snapshot.seen_lines().contains(&5));
		assert!(snapshot.seen_lines().contains(&6));
		assert!(!snapshot.seen_lines().contains(&7));
	}

	#[test]
	fn cancellation_guard_trips_the_walker_token() {
		let token = CancellationToken::new();
		{
			let _guard = CancelOnDrop(token.clone());
			assert!(!token.is_cancelled());
		}
		assert!(token.is_cancelled());
	}

	#[test]
	fn archive_root_round_trips_real_zip_members_into_memory_search() {
		let directory = tempfile::tempdir().expect("temp directory");
		let archive_path = directory.path().join("fixture.zip");
		std::fs::write(
			&archive_path,
			stored_zip(&[
				("docs/readme.txt", b"first\nneedle\nthird\n"),
				("docs/blob.bin", b"\0binary"),
			]),
		)
		.expect("write ZIP");
		let bytes = Bytes::from(std::fs::read(&archive_path).expect("read ZIP"));
		let root = SearchRoot {
			original: Str::from("fixture.zip:docs"),
			path:     Str::from("fixture.zip:docs"),
			kind:     SearchRootKind::Archive,
			ranges:   Box::default(),
		};
		let candidate = archive::parse_archive_path_candidates(root.path.as_str())
			.into_iter()
			.next()
			.expect("archive candidate");
		let archive_key = Str::from(
			std::fs::canonicalize(&archive_path)
				.expect("canonical ZIP")
				.to_string_lossy()
				.into_owned(),
		);
		let (targets, unreadable) =
			materialize_archive_bytes(&root, 0, candidate, archive_key, bytes)
				.expect("materialize ZIP");

		assert_eq!(targets.len(), 1);
		assert_eq!(targets[0].path, "fixture.zip:docs/readme.txt");
		assert_eq!(unreadable, vec![Str::from("fixture.zip:docs/blob.bin (binary archive entry)")]);
		let result = omp_grep::search(&targets[0].content, &omp_grep::GrepOptions {
			pattern: Str::from("needle"),
			path: targets[0].path.clone(),
			..omp_grep::GrepOptions::default()
		})
		.expect("search materialized member");
		assert_eq!(result.matches.len(), 1);
		assert_eq!(result.matches[0].path, "fixture.zip:docs/readme.txt");
		assert_eq!(result.matches[0].line_number, 2);
	}

	#[derive(Clone)]
	struct CannedHttpClient;

	impl web::types::HttpClient for CannedHttpClient {
		fn get(
			&self,
			request: web::types::HttpRequest,
		) -> impl Future<Output = Result<web::types::HttpResponse, web::types::WebError>> + Send + '_
		{
			async move {
				assert_eq!(request.url, "https://example.test/data.txt");
				Ok(web::types::HttpResponse {
					final_url:    request.url,
					status:       200,
					content_type: Some(Str::from("text/plain")),
					headers:      Default::default(),
					body:         Bytes::from_static(b"alpha\nneedle\nomega\n"),
				})
			}
		}
	}

	#[tokio::test]
	async fn url_root_round_trips_canned_http_into_memory_search() {
		let root = SearchRoot {
			original: Str::from("https://example.test/data.txt"),
			path:     Str::from("https://example.test/data.txt"),
			kind:     SearchRootKind::Url,
			ranges:   Box::default(),
		};
		let target = materialize_url_root(&CannedHttpClient, &root, 3)
			.await
			.expect("materialize URL");
		let result = omp_grep::search(&target.content, &omp_grep::GrepOptions {
			pattern: Str::from("needle"),
			path: target.path.clone(),
			..omp_grep::GrepOptions::default()
		})
		.expect("search URL");

		assert_eq!(target.root_index, 3);
		assert_eq!(result.matches.len(), 1);
		assert_eq!(result.matches[0].path, "https://example.test/data.txt");
		assert_eq!(result.matches[0].line_number, 2);
	}

	#[tokio::test]
	async fn archive_root_round_trips_through_the_real_search_adapter() {
		let directory = tempfile::tempdir().expect("temp directory");
		std::fs::write(
			directory.path().join("fixture.zip"),
			stored_zip(&[
				("docs/readme.txt", b"first\nneedle\nthird\n"),
				("docs/blob.bin", b"\0binary"),
			]),
		)
		.expect("write ZIP");
		let adapter = connected_search_adapter(directory.path()).await;
		let result = WorkspaceSearch::search(
			&adapter,
			search_request("fixture.zip:docs", SearchRootKind::Archive, 5_000),
		)
		.await
		.expect("search archive through adapter");

		assert_eq!(result.matches.len(), 1);
		assert_eq!(result.matches[0].path, "fixture.zip:docs/readme.txt");
		assert_eq!(result.matches[0].root_index, 0);
		assert_eq!(result.matches[0].line_number, 2);
		assert_eq!(result.matches[0].line, "needle");
		assert!(result.matches[0].source_key.starts_with("archive:"));
		assert!(result.matches[0].source_key.ends_with(":docs/readme.txt"));
		assert_eq!(result.archive_unreadable, vec![Str::from(
			"fixture.zip:docs/blob.bin (binary archive entry)"
		)]);
	}

	#[tokio::test]
	async fn external_archive_members_accept_absolute_and_parent_relative_roots() {
		let parent = tempfile::tempdir().expect("parent directory");
		let workspace = parent.path().join("workspace");
		std::fs::create_dir(&workspace).expect("workspace directory");
		let archive_path = parent.path().join("fixture.zip");
		std::fs::write(&archive_path, stored_zip(&[("docs/readme.txt", b"first\nneedle\nthird\n")]))
			.expect("write external ZIP");
		let canonical_archive = std::fs::canonicalize(&archive_path).expect("canonical external ZIP");
		let expected_source_key =
			format!("archive:{}:docs/readme.txt", canonical_archive.to_string_lossy());
		let adapter = connected_search_adapter(&workspace).await;
		let absolute = format!("{}:docs/readme.txt", archive_path.to_string_lossy());

		for authored in [absolute.as_str(), "../fixture.zip:docs/readme.txt"] {
			let result = WorkspaceSearch::search(
				&adapter,
				search_request(authored, SearchRootKind::Archive, 5_000),
			)
			.await
			.unwrap_or_else(|error| panic!("search external archive root {authored}: {error:?}"));

			assert_eq!(result.matches.len(), 1, "{authored}");
			assert_eq!(result.matches[0].path, authored);
			assert_eq!(result.matches[0].source_key, expected_source_key);
			assert_eq!(result.matches[0].root_index, 0);
			assert_eq!(result.matches[0].line_number, 2);
			assert_eq!(result.matches[0].line, "needle");
			assert_eq!(result.matches[0].snapshot_tag, None);
			assert!(result.archive_unreadable.is_empty());
		}
	}

	#[tokio::test]
	async fn external_parent_relative_glob_round_trips_through_the_real_adapter() {
		let parent = tempfile::tempdir().expect("parent directory");
		let workspace = parent.path().join("workspace");
		let external = parent.path().join("external");
		std::fs::create_dir(&workspace).expect("workspace directory");
		std::fs::create_dir(&external).expect("external directory");
		let source = external.join("lib.rs");
		std::fs::write(&source, "fn external() {}\n").expect("external source");
		let adapter = connected_search_adapter(&workspace).await;
		let result = WorkspaceSearch::glob(&adapter, glob::WalkRequest {
			path:       Str::from("../external/**/*.rs"),
			hidden:     true,
			gitignore:  false,
			limit:      200,
			timeout_ms: 5_000,
		})
		.await
		.expect("glob external parent-relative directory through adapter");

		assert_eq!(result.matches.len(), 1);
		assert_eq!(
			result.matches[0].path,
			std::fs::canonicalize(source)
				.expect("canonical external source")
				.to_string_lossy()
				.replace('\\', "/")
		);
		assert!(!result.matches[0].is_dir);
		assert!(!result.timed_out);
		assert!(!result.truncated);
	}

	#[tokio::test]
	async fn url_root_round_trips_local_http_through_the_real_search_adapter() {
		let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
			.await
			.expect("bind local HTTP fixture");
		let address = listener.local_addr().expect("fixture address");
		let url = format!("http://{address}/data.txt");
		let server = tokio::spawn(async move {
			let (mut socket, _) = listener.accept().await.expect("accept local request");
			let mut request = [0_u8; 4_096];
			let read = socket.read(&mut request).await.expect("read local request");
			let request = String::from_utf8_lossy(&request[..read]);
			assert!(request.starts_with("GET /data.txt "), "{request}");
			let body = b"alpha\nneedle\nomega\n";
			let response = format!(
				"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: \
				 close\r\n\r\n",
				body.len()
			);
			socket
				.write_all(response.as_bytes())
				.await
				.expect("write response headers");
			socket.write_all(body).await.expect("write response body");
			socket.shutdown().await.expect("close local response");
		});
		let directory = tempfile::tempdir().expect("temp directory");
		let adapter = connected_search_adapter(directory.path()).await;
		let result =
			WorkspaceSearch::search(&adapter, search_request(url.clone(), SearchRootKind::Url, 5_000))
				.await
				.expect("search local URL through adapter");
		tokio::time::timeout(Duration::from_secs(2), server)
			.await
			.expect("local HTTP fixture completed before its deadline")
			.expect("local HTTP fixture task succeeded");

		assert_eq!(result.matches.len(), 1);
		assert_eq!(result.matches[0].path, url);
		assert_eq!(result.matches[0].source_key, format!("url:{url}"));
		assert_eq!(result.matches[0].root_index, 0);
		assert_eq!(result.matches[0].line_number, 2);
		assert_eq!(result.matches[0].line, "needle");
		assert_eq!(result.matches[0].snapshot_tag, None);
	}

	#[tokio::test]
	async fn real_search_adapter_enforces_the_shared_external_materialization_deadline() {
		let directory = tempfile::tempdir().expect("temp directory");
		let adapter = connected_search_adapter(directory.path()).await;
		let error = WorkspaceSearch::search(
			&adapter,
			search_request("http://127.0.0.1:9/deadline", SearchRootKind::Url, 0),
		)
		.await
		.expect_err("zero-duration URL materialization deadline must expire before any request");
		assert_eq!(error, grep::Fault::TimedOut);
	}

	#[test]
	fn cancellation_and_glob_deadlines_are_observed_before_walking() {
		let directory = tempfile::tempdir().expect("temp directory");
		let host = WorkspaceHost::open(directory.path()).expect("workspace host");
		let cancelled = CancellationToken::new();
		cancelled.cancel();
		assert_eq!(
			check_grep_cancel(&cancelled),
			Err(grep::Fault::Cancelled { reason: Str::from(CANCELLED_REASON) })
		);
		assert_eq!(
			glob_blocking(
				&host,
				glob::WalkRequest {
					path:       Str::from("."),
					hidden:     true,
					gitignore:  false,
					limit:      200,
					timeout_ms: 5_000,
				},
				&cancelled,
			),
			Err(glob::Fault::Cancelled { reason: Str::from(CANCELLED_REASON) })
		);

		let timed_out = glob_blocking(
			&host,
			glob::WalkRequest {
				path:       Str::from("."),
				hidden:     true,
				gitignore:  false,
				limit:      200,
				timeout_ms: 0,
			},
			&CancellationToken::new(),
		)
		.expect("glob returns partial timeout metadata");
		assert!(timed_out.timed_out);
		assert!(timed_out.matches.is_empty());
	}

	fn stored_zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
		let mut output = Vec::new();
		let mut central = Vec::new();
		for (name, content) in entries {
			let offset = u32::try_from(output.len()).expect("small ZIP");
			let name = name.as_bytes();
			let size = u32::try_from(content.len()).expect("small member");
			let crc = crc32(content);
			push_u32(&mut output, 0x0403_4b50);
			push_u16(&mut output, 20);
			push_u16(&mut output, 0);
			push_u16(&mut output, 0);
			push_u16(&mut output, 0);
			push_u16(&mut output, 0);
			push_u32(&mut output, crc);
			push_u32(&mut output, size);
			push_u32(&mut output, size);
			push_u16(&mut output, u16::try_from(name.len()).expect("short name"));
			push_u16(&mut output, 0);
			output.extend_from_slice(name);
			output.extend_from_slice(content);

			push_u32(&mut central, 0x0201_4b50);
			push_u16(&mut central, 20);
			push_u16(&mut central, 20);
			push_u16(&mut central, 0);
			push_u16(&mut central, 0);
			push_u16(&mut central, 0);
			push_u16(&mut central, 0);
			push_u32(&mut central, crc);
			push_u32(&mut central, size);
			push_u32(&mut central, size);
			push_u16(&mut central, u16::try_from(name.len()).expect("short name"));
			push_u16(&mut central, 0);
			push_u16(&mut central, 0);
			push_u16(&mut central, 0);
			push_u16(&mut central, 0);
			push_u32(&mut central, 0);
			push_u32(&mut central, offset);
			central.extend_from_slice(name);
		}
		let central_offset = u32::try_from(output.len()).expect("small ZIP");
		let central_size = u32::try_from(central.len()).expect("small ZIP");
		output.extend_from_slice(&central);
		push_u32(&mut output, 0x0605_4b50);
		push_u16(&mut output, 0);
		push_u16(&mut output, 0);
		let count = u16::try_from(entries.len()).expect("few entries");
		push_u16(&mut output, count);
		push_u16(&mut output, count);
		push_u32(&mut output, central_size);
		push_u32(&mut output, central_offset);
		push_u16(&mut output, 0);
		output
	}

	fn crc32(bytes: &[u8]) -> u32 {
		let mut crc = u32::MAX;
		for byte in bytes {
			crc ^= u32::from(*byte);
			for _ in 0..8 {
				crc = (crc >> 1) ^ (0xedb8_8320 & 0_u32.wrapping_sub(crc & 1));
			}
		}
		!crc
	}

	fn push_u16(output: &mut Vec<u8>, value: u16) {
		output.extend_from_slice(&value.to_le_bytes());
	}

	fn push_u32(output: &mut Vec<u8>, value: u32) {
		output.extend_from_slice(&value.to_le_bytes());
	}
}
