//! App-owned workspace adapter for the generic grep and glob executors.

use std::{future::Future, path::{Component, Path}};

use omp_core::Str;
use omp_tools::{
	glob::{self, WalkResult},
	grep::{self, ByteSpan, SearchMatch, SearchResult, WorkspaceSearch},
};
use omp_walker::{CompiledWalkGlob, WalkFilter, WalkRequest};
use tokio_util::sync::CancellationToken;

use super::workspace::{WorkspaceError, WorkspaceHost};

const CANCELLED_REASON: &str = "workspace traversal future was dropped";

/// Cloneable bridge from generic search tools to the app-owned workspace.
#[derive(Clone, Debug)]
pub(crate) struct WorkspaceSearchAdapter {
	host: WorkspaceHost,
}

impl WorkspaceSearchAdapter {
	/// Wraps the concrete workspace owner used by the environment daemon.
	pub(crate) const fn new(host: WorkspaceHost) -> Self {
		Self { host }
	}
}

impl WorkspaceSearch for WorkspaceSearchAdapter {
	fn search(
		&self,
		request: grep::SearchRequest,
	) -> impl Future<Output = Result<SearchResult, grep::Fault>> + Send + '_ {
		let host = self.host.clone();
		async move {
			let cancel = CancellationToken::new();
			let cancel_on_drop = CancelOnDrop(cancel.clone());
			let operation = tokio::task::spawn_blocking(move || search_blocking(&host, request, &cancel));
			let result = operation
				.await
				.map_err(|error| grep::Fault::Workspace {
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
			let operation = tokio::task::spawn_blocking(move || glob_blocking(&host, request, &cancel));
			let result = operation
				.await
				.map_err(|error| glob::Fault::Workspace {
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

fn search_blocking(
	host: &WorkspaceHost,
	request: grep::SearchRequest,
	cancel: &CancellationToken,
) -> Result<SearchResult, grep::Fault> {
	if let Some(index) = request.patterns.iter().position(Str::is_empty) {
		return Err(grep::Fault::EmptyPattern {
			index: u64::try_from(index).unwrap_or(u64::MAX),
		});
	}
	if request.patterns.is_empty() {
		return Err(grep::Fault::EmptyPattern { index: 0 });
	}

	let include = compile_globs(&request.include).map_err(|(pattern, message)| {
		grep::Fault::Workspace {
			message: Str::from(format!("invalid include glob {pattern}: {message}")),
		}
	})?;
	let exclude = compile_globs(&request.exclude).map_err(|(pattern, message)| {
		grep::Fault::Workspace {
			message: Str::from(format!("invalid exclude glob {pattern}: {message}")),
		}
	})?;
	let mut walk = build_walk(host, &request.path, request.gitignore, request.hidden)
		.map_err(grep_workspace_fault)?
		.filter(match include {
			Some(include) => WalkFilter::files_only().glob(include),
			None => WalkFilter::files_only(),
		});
	walk = walk.no_limit();
	let candidates = host.candidates(&walk, cancel).map_err(grep_workspace_fault)?;
	let mut matches = Vec::new();
	let mut binary_skipped = Vec::new();

	for candidate in candidates {
		check_cancel(cancel).map_err(grep_workspace_fault)?;
		if exclude.as_ref().is_some_and(|glob| glob.is_match(&candidate.relative)) {
			continue;
		}
		let path = workspace_relative(host.root(), &candidate.path).map_err(grep_workspace_fault)?;
		let bytes = std::fs::read(&candidate.path).map_err(|source| {
			grep_workspace_fault(WorkspaceError::Read { path: path.clone(), source })
		})?;
		let Ok(text) = std::str::from_utf8(&bytes) else {
			binary_skipped.push(path);
			continue;
		};
		if text.as_bytes().contains(&0) {
			binary_skipped.push(path);
			continue;
		}
		for (line_index, line) in text.split('\n').enumerate() {
			check_cancel(cancel).map_err(grep_workspace_fault)?;
			let mut spans = exact_spans(line.as_bytes(), &request.patterns);
			if spans.is_empty() {
				continue;
			}
			spans.sort_unstable();
			spans.dedup();
			matches.push(SearchMatch {
				path: path.clone(),
				line: u64::try_from(line_index).unwrap_or(u64::MAX).saturating_add(1),
				spans,
				line_text: Str::from(line),
			});
		}
	}

	matches.sort_unstable();
	binary_skipped.sort_unstable();
	binary_skipped.dedup();
	let retain = usize::try_from(request.limit).unwrap_or(usize::MAX);
	let truncated = matches.len() > retain;
	matches.truncate(retain);
	Ok(SearchResult { matches, binary_skipped, truncated })
}

fn glob_blocking(
	host: &WorkspaceHost,
	request: glob::WalkRequest,
	cancel: &CancellationToken,
) -> Result<WalkResult, glob::Fault> {
	if request.patterns.is_empty() {
		return Err(glob::Fault::InvalidPattern {
			pattern: Str::new(""),
			message: Str::from("at least one glob pattern is required"),
		});
	}
	if let Some(pattern) = request.patterns.iter().find(|pattern| pattern.is_empty()) {
		return Err(glob::Fault::InvalidPattern {
			pattern: pattern.clone(),
			message: Str::from("glob pattern must not be empty"),
		});
	}
	let include = compile_globs(&request.patterns).map_err(glob_pattern_fault)?.ok_or_else(|| {
		glob::Fault::InvalidPattern {
			pattern: Str::new(""),
			message: Str::from("at least one glob pattern is required"),
		}
	})?;
	let exclude = compile_globs(&request.exclude).map_err(glob_pattern_fault)?;
	let walk = build_walk(host, &request.path, request.gitignore, request.hidden)
		.map_err(glob_workspace_fault)?
		.filter(WalkFilter::all().glob(include))
		.no_limit();
	let outcome = host.walk(&walk, cancel).map_err(glob_workspace_fault)?;
	let mut paths = Vec::with_capacity(outcome.entries.len());
	for entry in outcome.entries {
		check_cancel(cancel).map_err(glob_workspace_fault)?;
		if exclude.as_ref().is_some_and(|glob| glob.is_match(&entry.path)) {
			continue;
		}
		paths.push(
			workspace_relative(host.root(), &entry.absolute_path(walk.root()))
				.map_err(glob_workspace_fault)?,
		);
	}
	paths.sort_unstable();
	paths.dedup();
	let retain = usize::try_from(request.limit).unwrap_or(usize::MAX);
	let truncated = paths.len() > retain;
	paths.truncate(retain);
	Ok(WalkResult { paths, truncated })
}

fn build_walk(
	host: &WorkspaceHost,
	path: &Str,
	gitignore: bool,
	hidden: bool,
) -> Result<WalkRequest, WorkspaceError> {
	let root = std::fs::canonicalize(host.root().join(path.as_str()))
		.map_err(WorkspaceError::RequestRoot)?;
	if !root.starts_with(host.root()) {
		return Err(WorkspaceError::OutsideWorkspace);
	}
	Ok(WalkRequest::new(root).gitignore(gitignore).hidden(hidden))
}

fn compile_globs(patterns: &[Str]) -> Result<Option<CompiledWalkGlob>, (Str, Str)> {
	if patterns.is_empty() {
		return Ok(None);
	}
	for pattern in patterns {
		if let Err(error) = CompiledWalkGlob::new([pattern.as_str()]) {
			return Err((pattern.clone(), Str::from(error.to_string())));
		}
	}
	CompiledWalkGlob::new(patterns.iter().map(|pattern| pattern.as_str()))
		.map(Some)
		.map_err(|error| (patterns[0].clone(), Str::from(error.to_string())))
}

fn exact_spans(line: &[u8], patterns: &[Str]) -> Vec<ByteSpan> {
	let mut spans = Vec::new();
	for pattern in patterns {
		let pattern = pattern.as_bytes();
		for (start, window) in line.windows(pattern.len()).enumerate() {
			if window == pattern {
				spans.push(ByteSpan {
					start: u64::try_from(start).unwrap_or(u64::MAX),
					end: u64::try_from(start + pattern.len()).unwrap_or(u64::MAX),
				});
			}
		}
	}
	spans
}

fn workspace_relative(root: &Path, path: &Path) -> Result<Str, WorkspaceError> {
	let relative = path.strip_prefix(root).map_err(|_| WorkspaceError::OutsideWorkspace)?;
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
				return Err(WorkspaceError::OutsideWorkspace);
			},
		}
	}
	Ok(Str::from(normalized))
}

fn check_cancel(cancel: &CancellationToken) -> Result<(), WorkspaceError> {
	if cancel.is_cancelled() {
		Err(WorkspaceError::Cancelled)
	} else {
		Ok(())
	}
}

fn grep_workspace_fault(error: WorkspaceError) -> grep::Fault {
	match error {
		WorkspaceError::Cancelled => grep::Fault::Cancelled {
			reason: Str::from(CANCELLED_REASON),
		},
		other => grep::Fault::Workspace { message: Str::from(other.to_string()) },
	}
}

fn glob_workspace_fault(error: WorkspaceError) -> glob::Fault {
	match error {
		WorkspaceError::Cancelled => glob::Fault::Cancelled {
			reason: Str::from(CANCELLED_REASON),
		},
		other => glob::Fault::Workspace { message: Str::from(other.to_string()) },
	}
}

fn glob_pattern_fault((pattern, message): (Str, Str)) -> glob::Fault {
	glob::Fault::InvalidPattern { pattern, message }
}

#[cfg(test)]
mod tests {
	use std::fs;

	use super::*;

	#[test]
	fn conversion_roots_requests_and_preserves_walk_controls() {
		let workspace = tempfile::tempdir().expect("workspace tempdir");
		let nested = workspace.path().join("nested");
		fs::create_dir(&nested).expect("nested directory");
		let host = WorkspaceHost::open(workspace.path()).expect("workspace host");
		let walk = build_walk(&host, &Str::from("nested"), true, false).expect("walk request");
		assert_eq!(walk.root(), fs::canonicalize(nested).expect("canonical nested"));
		assert!(walk.options().use_gitignore);
		assert!(!walk.options().include_hidden);
	}

	#[test]
	fn conversion_rejects_a_canonical_escape() {
		let parent = tempfile::tempdir().expect("parent tempdir");
		let workspace = parent.path().join("workspace");
		let outside = parent.path().join("outside");
		fs::create_dir(&workspace).expect("workspace directory");
		fs::create_dir(&outside).expect("outside directory");
		let host = WorkspaceHost::open(&workspace).expect("workspace host");
		assert!(matches!(
			build_walk(&host, &Str::from("../outside"), true, false),
			Err(WorkspaceError::OutsideWorkspace)
		));
	}

	#[test]
	fn exact_patterns_merge_into_ordered_utf8_byte_spans() {
		let mut spans = exact_spans("λneedle needle".as_bytes(), &[
			Str::from("needle"),
			Str::from("λ"),
			Str::from("needle"),
		]);
		spans.sort_unstable();
		spans.dedup();
		assert_eq!(
			spans,
			[
				ByteSpan { start: 0, end: 2 },
				ByteSpan { start: 2, end: 8 },
				ByteSpan { start: 9, end: 15 },
			]
		);
	}

	#[test]
	fn search_merges_lines_and_reports_binary_candidates() {
		let workspace = tempfile::tempdir().expect("workspace tempdir");
		fs::write(workspace.path().join("text.txt"), "λneedle needle\n").expect("text file");
		fs::write(workspace.path().join("raw.bin"), [0xff, b'n', b'e', b'e', b'd', b'l', b'e'])
			.expect("binary file");
		let host = WorkspaceHost::open(workspace.path()).expect("workspace host");
		let result = search_blocking(
			&host,
			grep::SearchRequest {
				path: Str::from("."),
				patterns: vec![Str::from("needle"), Str::from("λ")],
				include: Vec::new(),
				exclude: Vec::new(),
				gitignore: false,
				hidden: false,
				limit: 2,
			},
			&CancellationToken::new(),
		)
		.expect("search result");
		assert_eq!(result.matches.len(), 1);
		assert_eq!(result.matches[0].path, "text.txt");
		assert_eq!(
			result.matches[0].spans,
			[
				ByteSpan { start: 0, end: 2 },
				ByteSpan { start: 2, end: 8 },
				ByteSpan { start: 9, end: 15 },
			]
		);
		assert_eq!(result.binary_skipped, [Str::from("raw.bin")]);
		assert!(!result.truncated);
	}

	#[test]
	fn search_limit_retains_one_record_and_reports_lookahead_truth() {
		let workspace = tempfile::tempdir().expect("workspace tempdir");
		fs::write(workspace.path().join("text.txt"), "needle\nneedle\n").expect("text file");
		let host = WorkspaceHost::open(workspace.path()).expect("workspace host");
		let result = search_blocking(
			&host,
			grep::SearchRequest {
				path: Str::from("."),
				patterns: vec![Str::from("needle")],
				include: Vec::new(),
				exclude: Vec::new(),
				gitignore: false,
				hidden: false,
				limit: 1,
			},
			&CancellationToken::new(),
		)
		.expect("search result");
		assert_eq!(result.matches.len(), 1);
		assert!(result.truncated);
	}

	#[test]
	fn glob_paths_are_workspace_relative_with_slashes() {
		let workspace = tempfile::tempdir().expect("workspace tempdir");
		let nested = workspace.path().join("src");
		fs::create_dir(&nested).expect("source directory");
		fs::write(nested.join("lib.rs"), "fn item() {}\n").expect("source file");
		let host = WorkspaceHost::open(workspace.path()).expect("workspace host");
		let result = glob_blocking(
			&host,
			glob::WalkRequest {
				path: Str::from("src"),
				patterns: vec![Str::from("*.rs")],
				exclude: Vec::new(),
				gitignore: false,
				hidden: false,
				limit: 2,
			},
			&CancellationToken::new(),
		)
		.expect("glob result");
		assert_eq!(result.paths, [Str::from("src/lib.rs")]);
		assert!(!result.truncated);
	}

	#[test]
	fn dropping_cancellation_guard_trips_the_walker_token() {
		let token = CancellationToken::new();
		{
			let _guard = CancelOnDrop(token.clone());
			assert!(!token.is_cancelled());
		}
		assert!(token.is_cancelled());
	}
}
