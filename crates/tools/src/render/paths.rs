//! Prefix-folded path trees shared by path listings and grouped file output.

use std::collections::{HashMap, HashSet};

use omp_core::Str;

/// One flat path supplied to [`build_path_tree`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PathTreeInput<'a> {
	/// An absolute, workspace-relative, or URL-like path.
	pub path:   &'a str,
	/// Whether the path itself is a directory leaf.
	pub is_dir: bool,
	/// An opaque lookup key carried by the resulting file event; defaults to
	/// `path`.
	pub key:    Option<&'a str>,
}

impl<'a> PathTreeInput<'a> {
	/// Creates a path input whose file-event key is the original path.
	#[must_use]
	pub const fn new(path: &'a str, is_dir: bool) -> Self {
		Self { path, is_dir, key: None }
	}

	/// Creates a path input with an explicit file-event lookup key.
	#[must_use]
	pub const fn with_key(path: &'a str, is_dir: bool, key: &'a str) -> Self {
		Self { path, is_dir, key: Some(key) }
	}
}

#[derive(Debug, Default)]
struct BuilderNode {
	files:      Vec<FileLeaf>,
	file_names: HashSet<Str>,
	subdirs:    Vec<BuilderDirectory>,
	dir_index:  HashMap<Str, usize>,
}

impl BuilderNode {
	fn add_file(&mut self, name: &str, key: &str) {
		let name = Str::from(name);
		if !self.file_names.insert(name.clone()) {
			return;
		}
		self.files.push(FileLeaf { name, key: Str::from(key) });
	}

	fn child_mut(&mut self, name: &str) -> &mut Self {
		let index = match self.dir_index.get(name) {
			Some(index) => *index,
			None => {
				let index = self.subdirs.len();
				let name = Str::from(name);
				self.dir_index.insert(name.clone(), index);
				self
					.subdirs
					.push(BuilderDirectory { name, node: Self::default() });
				index
			},
		};
		&mut self.subdirs[index].node
	}
}

#[derive(Debug)]
struct BuilderDirectory {
	name: Str,
	node: BuilderNode,
}

#[derive(Debug)]
struct FileLeaf {
	name: Str,
	key:  Str,
}

#[derive(Debug)]
struct Directory {
	name: Str,
	node: PathTreeNode,
}

#[derive(Debug)]
struct PathTreeNode {
	files:   Vec<FileLeaf>,
	subdirs: Vec<Directory>,
}

/// A directory tree built from flat paths in first-seen order.
///
/// The tree owns compact copies of path segments and keys, so callers may drop
/// their input collection before walking it. Its representation is private to
/// keep deduplication indexes and folded directory chains out of consumers.
#[derive(Debug)]
pub struct PathTree {
	root: PathTreeNode,
}

/// The kind of item emitted by [`walk_path_tree`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GroupedTreeEventKind {
	/// A prefix-folded directory chain.
	Directory,
	/// A direct file leaf.
	File,
}

/// One prefix-folded directory or file emitted by [`walk_path_tree`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GroupedTreeEvent<'a> {
	/// Whether this event represents a directory or file.
	pub kind:  GroupedTreeEventKind,
	/// Zero-based nesting depth; root children have depth zero.
	pub depth: usize,
	/// A folded chain without a trailing slash for directories, or a basename
	/// for files.
	pub name:  &'a str,
	/// The opaque file key, or an empty string for directories.
	pub key:   &'a str,
}

impl GroupedTreeEvent<'_> {
	/// Returns whether grouped file output starts a blank-line-delimited section
	/// here.
	///
	/// Consumers insert a separator only when something has already been
	/// emitted. Every directory and every root-level file starts a section;
	/// nested files stay attached to the directory header above them.
	#[must_use]
	pub const fn starts_group(self) -> bool {
		matches!(self.kind, GroupedTreeEventKind::Directory) || self.depth == 0
	}

	/// Returns the number of `#` characters in this event's grouped header.
	#[must_use]
	pub const fn heading_level(self) -> usize {
		self.depth + 1
	}
}

/// Returns whether `path` has a `scheme://` prefix.
///
/// URL-like paths are retained whole as root-level files because their slash
/// components do not represent workspace directories.
#[must_use]
pub fn is_url_like_path(path: &str) -> bool {
	let Some((scheme, _)) = path.split_once("://") else {
		return false;
	};
	let mut chars = scheme.bytes();
	matches!(chars.next(), Some(first) if first.is_ascii_alphabetic())
		&& chars.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'.' | b'-'))
}

/// Builds a path tree while preserving first-seen directory and file order.
///
/// Backslashes are normalized to slashes. Absolute paths retain their leading
/// empty segment, allowing the common `/` prefix to fold naturally. Duplicate
/// file basenames under the same node keep the first key, matching glob retry
/// deduplication behavior. Directory-only inputs remain visible as leaf
/// headers.
#[must_use]
pub fn build_path_tree<'a>(entries: impl IntoIterator<Item = PathTreeInput<'a>>) -> PathTree {
	let mut root = BuilderNode::default();
	for input in entries {
		let normalized_storage = input
			.path
			.contains('\\')
			.then(|| input.path.replace('\\', "/"));
		let normalized = normalized_storage.as_deref().unwrap_or(input.path);
		let key = input.key.unwrap_or(input.path);
		if is_url_like_path(normalized) {
			root.add_file(normalized, key);
			continue;
		}

		let trimmed = normalized.strip_suffix('/').unwrap_or(normalized);
		if trimmed.is_empty() {
			continue;
		}

		let mut node = &mut root;
		let mut segments = trimmed.split('/').peekable();
		while let Some(segment) = segments.next() {
			if !input.is_dir && segments.peek().is_none() {
				node.add_file(segment, key);
				break;
			}
			node = node.child_mut(segment);
		}
	}
	PathTree { root: finish_node(root) }
}

fn finish_node(node: BuilderNode) -> PathTreeNode {
	PathTreeNode {
		files:   node.files,
		subdirs: node.subdirs.into_iter().map(finish_directory).collect(),
	}
}

fn finish_directory(mut directory: BuilderDirectory) -> Directory {
	let mut folded = None;
	while directory.node.files.is_empty() && directory.node.subdirs.len() == 1 {
		let only = directory
			.node
			.subdirs
			.pop()
			.expect("single child checked above");
		let buffer = folded.get_or_insert_with(|| {
			let mut buffer = String::with_capacity(directory.name.len() + 1 + only.name.len());
			buffer.push_str(&directory.name);
			buffer
		});
		buffer.push('/');
		buffer.push_str(&only.name);
		directory.node = only.node;
	}
	if let Some(folded) = folded {
		directory.name = Str::from(folded);
	}
	Directory { name: directory.name, node: finish_node(directory.node) }
}

#[derive(Debug)]
struct WalkFrame<'a> {
	node:      &'a PathTreeNode,
	depth:     usize,
	next_file: usize,
	next_dir:  usize,
}

impl<'a> WalkFrame<'a> {
	const fn new(node: &'a PathTreeNode, depth: usize) -> Self {
		Self { node, depth, next_file: 0, next_dir: 0 }
	}
}

/// Allocation-light depth-first iterator returned by [`walk_path_tree`].
///
/// A node's direct files are emitted before its child directories. The iterator
/// allocates only its traversal stack; folded directory names are computed once
/// when the tree is built and borrowed by each event.
#[derive(Debug)]
pub struct PathTreeIter<'a> {
	stack: Vec<WalkFrame<'a>>,
}

impl<'a> Iterator for PathTreeIter<'a> {
	type Item = GroupedTreeEvent<'a>;

	fn next(&mut self) -> Option<Self::Item> {
		loop {
			let frame = self.stack.last_mut()?;
			if let Some(file) = frame.node.files.get(frame.next_file) {
				frame.next_file += 1;
				return Some(GroupedTreeEvent {
					kind:  GroupedTreeEventKind::File,
					depth: frame.depth,
					name:  &file.name,
					key:   &file.key,
				});
			}
			if frame.next_dir < frame.node.subdirs.len() {
				let node = frame.node;
				let directory_index = frame.next_dir;
				let depth = frame.depth;
				frame.next_dir += 1;
				let directory = &node.subdirs[directory_index];
				self.stack.push(WalkFrame::new(&directory.node, depth + 1));
				return Some(GroupedTreeEvent {
					kind: GroupedTreeEventKind::Directory,
					depth,
					name: &directory.name,
					key: "",
				});
			}
			self.stack.pop();
		}
	}
}

/// Walks `tree` depth-first with single-child directory chains already folded.
#[must_use]
pub fn walk_path_tree(tree: &PathTree) -> PathTreeIter<'_> {
	PathTreeIter { stack: vec![WalkFrame::new(&tree.root, 0)] }
}

/// Renders paths as a prefix-folded listing without per-file annotations.
///
/// Directory headers carry one `#` per depth and retain a trailing slash. File
/// leaves are bare. Unlike grouped grep sections, this compact listing contains
/// no blank separator lines.
#[must_use]
pub fn format_grouped_paths<P: AsRef<str>>(paths: &[P]) -> String {
	format_grouped_paths_annotated(paths, |_| "")
}

/// Renders paths as a prefix-folded listing with a suffix appended to each
/// file.
///
/// `annotate` receives the full original path key, not the displayed basename;
/// its returned text is appended verbatim.
#[must_use]
pub fn format_grouped_paths_annotated<P, F, A>(paths: &[P], mut annotate: F) -> String
where
	P: AsRef<str>,
	F: FnMut(&str) -> A,
	A: AsRef<str>,
{
	if paths.is_empty() {
		return String::new();
	}
	let tree = build_path_tree(paths.iter().map(|entry| {
		let path: &str = entry.as_ref();
		PathTreeInput::new(path, path.ends_with('/'))
	}));
	let estimated_bytes = paths.iter().map(|path| path.as_ref().len() + 2).sum();
	let mut output = String::with_capacity(estimated_bytes);
	for event in walk_path_tree(&tree) {
		if !output.is_empty() {
			output.push('\n');
		}
		if event.kind == GroupedTreeEventKind::Directory {
			for _ in 0..event.heading_level() {
				output.push('#');
			}
			output.push(' ');
			output.push_str(event.name);
			output.push('/');
		} else {
			output.push_str(event.name);
			output.push_str(annotate(event.key).as_ref());
		}
	}
	output
}

#[cfg(test)]
mod tests {
	use super::{
		GroupedTreeEventKind, PathTreeInput, build_path_tree, format_grouped_paths,
		format_grouped_paths_annotated, is_url_like_path, walk_path_tree,
	};

	#[test]
	fn folds_absolute_prefixes_and_preserves_first_seen_order() {
		let paths = [
			"/Users/me/proj/shared/wasm/llvm.hpp",
			"/Users/me/proj/shared/wasm/vm.hpp",
			"/Users/me/proj/shared/xstd.hpp",
			"/Users/me/proj/shared/apollo/details/hash.hpp",
			"/Users/me/proj/flash/main.cpp",
		];
		assert_eq!(
			format_grouped_paths(&paths),
			[
				"# /Users/me/proj/",
				"## shared/",
				"xstd.hpp",
				"### wasm/",
				"llvm.hpp",
				"vm.hpp",
				"### apollo/details/",
				"hash.hpp",
				"## flash/",
				"main.cpp",
			]
			.join("\n")
		);
	}

	#[test]
	fn emits_direct_files_before_subdirectories_and_keeps_directory_leaves() {
		assert_eq!(
			format_grouped_paths(&["pkg/sub/deep.txt", "pkg/top.txt"]),
			["# pkg/", "top.txt", "## sub/", "deep.txt"].join("\n")
		);
		assert_eq!(
			format_grouped_paths(&["alpha/tests/", "beta/tests/"]),
			["# alpha/tests/", "# beta/tests/"].join("\n")
		);
	}

	#[test]
	fn root_files_and_annotations_use_original_keys() {
		assert_eq!(format_grouped_paths(&["single.txt"]), "single.txt");
		assert_eq!(
			format_grouped_paths_annotated(&["src/a.ts", "src/b.ts"], |path| {
				if path == "src/a.ts" {
					" (RW)"
				} else {
					" (Read)"
				}
			}),
			["# src/", "a.ts (RW)", "b.ts (Read)"].join("\n")
		);
	}

	#[test]
	fn traversal_marks_group_boundaries_for_grouped_file_projection() {
		let tree = build_path_tree([
			PathTreeInput::with_key("root.rs", false, "root"),
			PathTreeInput::with_key("src/a.rs", false, "a"),
			PathTreeInput::with_key("src/deep/b.rs", false, "b"),
		]);
		let events: Vec<_> = walk_path_tree(&tree).collect();
		assert_eq!(events[0].kind, GroupedTreeEventKind::File);
		assert_eq!((events[0].depth, events[0].name, events[0].key), (0, "root.rs", "root"));
		assert!(events[0].starts_group());
		assert_eq!(events[1].kind, GroupedTreeEventKind::Directory);
		assert_eq!((events[1].depth, events[1].name), (0, "src"));
		assert!(events[1].starts_group());
		assert_eq!((events[2].depth, events[2].name, events[2].key), (1, "a.rs", "a"));
		assert!(!events[2].starts_group());
		assert_eq!((events[3].depth, events[3].name), (1, "deep"));
		assert!(events[3].starts_group());
		assert_eq!((events[4].depth, events[4].name, events[4].key), (2, "b.rs", "b"));
	}

	#[test]
	fn normalizes_backslashes_but_keeps_url_like_paths_whole() {
		assert_eq!(format_grouped_paths(&[r"src\nested\file.rs"]), "# src/nested/\nfile.rs");
		assert_eq!(format_grouped_paths(&["https://example.test/a/b"]), "https://example.test/a/b");
		assert!(is_url_like_path("git+ssh://example.test/repo"));
		assert!(!is_url_like_path("1http://example.test"));
	}
}
