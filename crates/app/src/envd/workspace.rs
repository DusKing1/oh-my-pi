//! Workspace traversal, candidate discovery, and cancellable byte search.

use std::{
	io,
	path::{Path, PathBuf},
	sync::atomic::{AtomicUsize, Ordering},
};

use bytes::Bytes;
use omp_core::Str;
use omp_walker::{FileCandidate, WalkError, WalkOutcome, WalkRequest, execute_candidates};
use parking_lot::Mutex;
use thiserror::Error;
use tokio_util::sync::CancellationToken;

const CANCELLED: &str = "workspace operation cancelled";

/// One fixed-byte match in a workspace file.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SearchMatch {
	/// Walk-relative path using `/` separators.
	pub(crate) path:        Str,
	/// One-based source line containing the match.
	pub(crate) line:        u64,
	/// Zero-based byte offset in the complete file.
	pub(crate) byte_offset: u64,
	/// Exact bytes of the matching line, excluding its line-feed delimiter.
	pub(crate) line_bytes:  Bytes,
}

/// Workspace traversal or search failed.
#[derive(Debug, Error)]
pub enum WorkspaceError {
	/// The caller cancelled the operation.
	#[error("workspace operation was cancelled")]
	Cancelled,
	/// The requested walker root escapes the owned workspace.
	#[error("workspace request root is outside the owned workspace")]
	OutsideWorkspace,
	/// A byte search was requested with an empty pattern.
	#[error("search pattern must not be empty")]
	EmptyPattern,
	/// Workspace traversal failed.
	#[error("workspace walk failed: {0}")]
	Walk(Str),
	/// A discovered candidate could not be read.
	#[error("failed to read workspace candidate {path}: {source}")]
	Read {
		/// Workspace-relative candidate path.
		path:   Str,
		/// Underlying filesystem error.
		#[source]
		source: io::Error,
	},
	/// The owned workspace root could not be opened.
	#[error("workspace root cannot be opened: {0}")]
	Root(#[source] io::Error),
	/// The requested traversal root could not be opened.
	#[error("workspace request root cannot be opened: {0}")]
	RequestRoot(#[source] io::Error),
}

/// Concrete env-side owner of one canonical walker workspace.
#[derive(Clone, Debug)]
pub struct WorkspaceHost {
	root: PathBuf,
}

impl WorkspaceHost {
	/// Opens a workspace rooted at a canonical existing path.
	pub fn open(root: impl AsRef<Path>) -> Result<Self, WorkspaceError> {
		let root = std::fs::canonicalize(root).map_err(WorkspaceError::Root)?;
		Ok(Self { root })
	}

	/// Returns the canonical workspace root.
	pub fn root(&self) -> &Path {
		&self.root
	}

	/// Starts a walker request that cannot escape this host's workspace.
	pub fn request(&self) -> WalkRequest {
		WalkRequest::new(self.root.clone())
	}

	/// Runs a walker collection with a cancellation heartbeat.
	pub fn walk(
		&self,
		request: &WalkRequest,
		cancel: &CancellationToken,
	) -> Result<WalkOutcome, WorkspaceError> {
		self.check_request(request)?;
		request
			.collect_with_heartbeat(|| cancellation_heartbeat(cancel))
			.map_err(map_walk_error)
	}

	/// Collects regular-file candidates with a cancellation heartbeat.
	pub fn candidates(
		&self,
		request: &WalkRequest,
		cancel: &CancellationToken,
	) -> Result<Vec<FileCandidate>, WorkspaceError> {
		self.check_request(request)?;
		request
			.collect_file_candidates_with_heartbeat(|| cancellation_heartbeat(cancel))
			.map_err(map_walk_error)
	}

	/// Searches candidate contents for exact byte strings.
	///
	/// Candidate discovery and file scanning share the same cancellation token.
	/// `limit` is a process-wide bound across walker workers; `None` is
	/// unbounded.
	pub fn search(
		&self,
		request: &WalkRequest,
		pattern: &[u8],
		limit: Option<usize>,
		cancel: &CancellationToken,
	) -> Result<Vec<SearchMatch>, WorkspaceError> {
		if pattern.is_empty() {
			return Err(WorkspaceError::EmptyPattern);
		}
		if limit == Some(0) {
			return Ok(Vec::new());
		}
		let candidates = self.candidates(request, cancel)?;
		let matches = Mutex::new(Vec::new());
		let matched = AtomicUsize::new(0);
		execute_candidates(&candidates, |candidate| {
			cancellation_heartbeat(cancel).map_err(|_| WorkspaceError::Cancelled)?;
			if limit.is_some_and(|limit| matched.load(Ordering::Relaxed) >= limit) {
				return Ok(());
			}
			let relative_path = Str::new(&candidate.relative);
			let content = Bytes::from(
				std::fs::read(&candidate.path)
					.map_err(|source| WorkspaceError::Read { path: relative_path.clone(), source })?,
			);
			let mut local = Vec::new();
			let mut line_start = 0_usize;
			let mut line_number = 1_u64;
			while line_start <= content.len() {
				cancellation_heartbeat(cancel).map_err(|_| WorkspaceError::Cancelled)?;
				if limit.is_some_and(|limit| matched.load(Ordering::Relaxed) >= limit) {
					break;
				}
				let line_end = content[line_start..]
					.iter()
					.position(|byte| *byte == b'\n')
					.map_or(content.len(), |relative| line_start + relative);
				let line = &content[line_start..line_end];
				for relative in match_offsets(line, pattern) {
					let slot = matched.fetch_add(1, Ordering::Relaxed);
					if limit.is_some_and(|limit| slot >= limit) {
						break;
					}
					local.push(SearchMatch {
						path:        relative_path.clone(),
						line:        line_number,
						byte_offset: u64::try_from(line_start + relative).unwrap_or(u64::MAX),
						line_bytes:  content.slice(line_start..line_end),
					});
				}
				if line_end == content.len() {
					break;
				}
				line_start = line_end + 1;
				line_number += 1;
			}
			matches.lock().extend(local);
			Ok(())
		})?;
		if cancel.is_cancelled() {
			return Err(WorkspaceError::Cancelled);
		}
		let mut matches = matches.into_inner();
		matches.sort_unstable_by(|left, right| {
			left
				.path
				.cmp(&right.path)
				.then_with(|| left.byte_offset.cmp(&right.byte_offset))
		});
		if let Some(limit) = limit {
			matches.truncate(limit);
		}
		Ok(matches)
	}

	fn check_request(&self, request: &WalkRequest) -> Result<(), WorkspaceError> {
		let root = std::fs::canonicalize(request.root()).map_err(WorkspaceError::RequestRoot)?;
		if root.starts_with(&self.root) {
			Ok(())
		} else {
			Err(WorkspaceError::OutsideWorkspace)
		}
	}
}

fn cancellation_heartbeat(cancel: &CancellationToken) -> Result<(), &'static str> {
	if cancel.is_cancelled() {
		Err(CANCELLED)
	} else {
		Ok(())
	}
}

fn map_walk_error(error: WalkError<String>) -> WorkspaceError {
	match error {
		WalkError::Interrupted(message) if message == CANCELLED => WorkspaceError::Cancelled,
		other => WorkspaceError::Walk(Str::from(other.to_string())),
	}
}

fn match_offsets<'a>(line: &'a [u8], pattern: &'a [u8]) -> impl Iterator<Item = usize> + 'a {
	line
		.windows(pattern.len())
		.enumerate()
		.filter_map(move |(offset, window)| (window == pattern).then_some(offset))
}
