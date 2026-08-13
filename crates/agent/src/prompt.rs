//! Deterministic construction of the canonical system-prompt head.

use std::{
	fmt,
	path::{Path, PathBuf},
	sync::Arc,
};

use bytes::Bytes;
use omp_core::Str;
use omp_proto::thread::v1::{self as thread, Item};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Immutable bytes and identity for one workspace context file.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContextFile {
	/// Workspace-relative or absolute path presented to the model.
	pub path:    PathBuf,
	/// Exact file bytes captured for this snapshot.
	pub content: Bytes,
}

impl ContextFile {
	/// Creates an immutable context-file input.
	#[inline]
	pub fn new(path: impl Into<PathBuf>, content: impl Into<Bytes>) -> Self {
		Self { path: path.into(), content: content.into() }
	}
}

/// Stable source-control identity included in a workspace prompt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VcsIdentity {
	/// Repository root captured for this snapshot.
	pub root: PathBuf,
	/// Stable revision, branch, or ref identity supplied by the host.
	pub head: Str,
}

impl VcsIdentity {
	/// Creates a source-control identity.
	#[inline]
	pub fn new(root: impl Into<PathBuf>, head: impl Into<Str>) -> Self {
		Self { root: root.into(), head: head.into() }
	}
}

/// Immutable input used to render a workspace system prompt.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct WorkspaceInput {
	/// Current workspace directory captured by the host.
	pub cwd:           PathBuf,
	/// Optional source-control identity captured at the same boundary.
	pub vcs:           Option<VcsIdentity>,
	/// Ordered context files with exact, immutable contents.
	pub context_files: Arc<[ContextFile]>,
}

impl WorkspaceInput {
	/// Creates workspace input without source-control identity.
	#[inline]
	pub fn new(cwd: impl Into<PathBuf>, context_files: impl Into<Arc<[ContextFile]>>) -> Self {
		Self { cwd: cwd.into(), vcs: None, context_files: context_files.into() }
	}

	/// Attaches a stable source-control identity.
	#[inline]
	#[must_use]
	pub fn with_vcs(mut self, vcs: VcsIdentity) -> Self {
		self.vcs = Some(vcs);
		self
	}
}

/// Stable BLAKE3 digest of the canonical prompt items.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(transparent)]
pub struct PromptHash([u8; 32]);

impl PromptHash {
	/// Returns the digest bytes.
	#[inline]
	pub const fn as_bytes(&self) -> &[u8; 32] {
		&self.0
	}

	/// Consumes the digest and returns its bytes.
	#[inline]
	pub const fn into_bytes(self) -> [u8; 32] {
		self.0
	}
}

impl From<[u8; 32]> for PromptHash {
	#[inline]
	fn from(bytes: [u8; 32]) -> Self {
		Self(bytes)
	}
}

impl From<PromptHash> for [u8; 32] {
	#[inline]
	fn from(hash: PromptHash) -> Self {
		hash.0
	}
}

/// A checked canonical prompt head and its content hash.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderedPrompt {
	/// Ordered canonical system items.
	pub items: Arc<[Item]>,
	/// BLAKE3 digest of the canonical serialized items.
	pub hash:  PromptHash,
}

/// Synchronous source of canonical system-prompt items.
///
/// Implementations receive only immutable workspace input. Callers must use
/// [`render_prompt`] so the source is rendered twice and checked for volatile
/// output before its items enter a thread.
pub trait PromptSource: Send + Sync + 'static {
	/// Renders one candidate prompt head from immutable input.
	fn render(&self, workspace: &WorkspaceInput) -> Result<Vec<Item>, PromptError>;
}

/// Deterministic plain-text renderer for workspace identity and context files.
#[derive(Clone, Copy, Debug, Default)]
pub struct WorkspacePromptSource;

impl PromptSource for WorkspacePromptSource {
	fn render(&self, workspace: &WorkspaceInput) -> Result<Vec<Item>, PromptError> {
		let cwd = prompt_path(&workspace.cwd)?;
		let mut identity = String::with_capacity(cwd.len() + 96);
		identity.push_str("Workspace\nDirectory: ");
		identity.push_str(cwd);
		if let Some(vcs) = &workspace.vcs {
			identity.push_str("\nRepository: ");
			identity.push_str(prompt_path(&vcs.root)?);
			identity.push_str("\nRevision: ");
			identity.push_str(vcs.head.as_str());
		}

		let mut items = Vec::with_capacity(1 + workspace.context_files.len());
		items.push(system_text(identity));
		for file in workspace.context_files.iter() {
			let path = prompt_path(&file.path)?;
			let content = std::str::from_utf8(&file.content)
				.map_err(|source| PromptError::ContextEncoding { path: file.path.clone(), source })?;
			let mut text = String::with_capacity(path.len() + content.len() + 32);
			text.push_str("Context file: ");
			text.push_str(path);
			text.push('\n');
			text.push_str(content);
			items.push(system_text(text));
		}
		Ok(items)
	}
}

/// Prompt rendering or canonicalization failure.
#[derive(Debug, Error)]
pub enum PromptError {
	/// The source emitted different items for identical immutable input.
	#[error("prompt source emitted volatile output for identical workspace input")]
	Volatile,
	/// A prompt item was not a canonical, unstamped system message.
	#[error("prompt item {index} is not a canonical unstamped system message")]
	InvalidItem {
		/// Zero-based index of the invalid item.
		index: usize,
	},
	/// A workspace path could not be represented exactly as UTF-8.
	#[error("workspace path is not valid UTF-8: {0:?}")]
	PathEncoding(PathBuf),
	/// A context file was not valid UTF-8.
	#[error("context file is not valid UTF-8: {path:?}")]
	ContextEncoding {
		/// Path of the invalid context file.
		path:   PathBuf,
		/// UTF-8 decoding failure.
		#[source]
		source: std::str::Utf8Error,
	},
	/// Canonical item serialization failed.
	#[error("failed to serialize canonical prompt items")]
	Serialize(#[from] serde_json::Error),
	/// A custom prompt source rejected its input.
	#[error("prompt source failed: {0}")]
	Source(Str),
}

/// Renders, validates, volatility-checks, and hashes one prompt head.
///
/// The source is invoked twice against the same immutable input. Both complete
/// item sequences must be byte-for-byte equal before either is accepted.
pub fn render_prompt(
	source: &dyn PromptSource,
	workspace: &WorkspaceInput,
) -> Result<RenderedPrompt, PromptError> {
	let first = source.render(workspace)?;
	validate_items(&first)?;
	let second = source.render(workspace)?;
	validate_items(&second)?;
	if first != second {
		return Err(PromptError::Volatile);
	}
	drop(second);

	let mut hasher = blake3::Hasher::new();
	serde_json::to_writer(&mut hasher, &first)?;
	let hash = PromptHash(*hasher.finalize().as_bytes());
	Ok(RenderedPrompt { items: first.into(), hash })
}

fn validate_items(items: &[Item]) -> Result<(), PromptError> {
	for (index, item) in items.iter().enumerate() {
		let canonical = item.seq == 0
			&& item.created_at_ms == 0
			&& item.props.is_none()
			&& matches!(
				item.kind.as_ref(),
				Some(thread::item::Kind::Message(message))
					if message.role == thread::Role::System as i32
			);
		if !canonical {
			return Err(PromptError::InvalidItem { index });
		}
	}
	Ok(())
}

fn prompt_path(path: &Path) -> Result<&str, PromptError> {
	path
		.to_str()
		.ok_or_else(|| PromptError::PathEncoding(path.to_path_buf()))
}

fn system_text(text: String) -> Item {
	Item {
		seq:           0,
		created_at_ms: 0,
		kind:          Some(thread::item::Kind::Message(thread::Message {
			role:  thread::Role::System as i32,
			parts: vec![thread::Part { kind: Some(thread::part::Kind::Text(text)) }],
		})),
		props:         None,
	}
}

impl fmt::Display for PromptHash {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(formatter, "{}", blake3::Hash::from_bytes(self.0))
	}
}

#[cfg(test)]
mod tests {
	use std::sync::atomic::{AtomicBool, Ordering};

	use super::*;

	#[test]
	fn workspace_prompt_is_canonical_and_stable() {
		let workspace = WorkspaceInput::new(
			"/workspace",
			Arc::from([ContextFile::new("AGENTS.md", Bytes::from_static(b"rules"))]),
		)
		.with_vcs(VcsIdentity::new("/workspace", "abc123"));
		let first = render_prompt(&WorkspacePromptSource, &workspace).expect("first render");
		let second = render_prompt(&WorkspacePromptSource, &workspace).expect("second render");

		assert_eq!(first, second);
		assert_eq!(first.items.len(), 2);
		assert!(first.items.iter().all(|item| matches!(
			item.kind.as_ref(),
			Some(thread::item::Kind::Message(message))
				if message.role == thread::Role::System as i32
		)));
		let changed = WorkspaceInput::new(
			"/workspace",
			Arc::from([ContextFile::new("AGENTS.md", Bytes::from_static(b"changed"))]),
		)
		.with_vcs(VcsIdentity::new("/workspace", "abc123"));
		let changed = render_prompt(&WorkspacePromptSource, &changed).expect("changed render");
		assert_ne!(first.hash, changed.hash);
	}

	#[test]
	fn volatile_source_is_rejected() {
		struct VolatileSource(AtomicBool);

		impl PromptSource for VolatileSource {
			fn render(&self, _workspace: &WorkspaceInput) -> Result<Vec<Item>, PromptError> {
				let prior = self.0.fetch_xor(true, Ordering::Relaxed);
				Ok(vec![system_text(prior.to_string())])
			}
		}

		let source = VolatileSource(AtomicBool::new(false));
		assert!(matches!(
			render_prompt(&source, &WorkspaceInput::default()),
			Err(PromptError::Volatile)
		));
	}
}
