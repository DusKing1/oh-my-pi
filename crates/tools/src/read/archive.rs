//! Archive read behavior for selectors, text classification, and presentation.
//!
//! ZIP, TAR, and TAR.GZ parsing, limits, link resolution, and member decoding
//! live in `omp-ar`. This module translates that format-neutral API into the
//! read tool's selector and model-facing contracts.

use std::{
	fs::File,
	io::{BufReader, Cursor, Read, Seek},
	path::{Path, PathBuf},
};

use bytes::Bytes;
use omp_ar::{Archive, Entry, Error as ArError, Limits};

use super::selector::{ParsedSelector, SelectorError, parse_selector, split_path_and_selector};

/// Default number of archive-directory entries returned by one read.
pub const DEFAULT_ARCHIVE_LIST_LIMIT: usize = 500;
/// Maximum decoded size of one tar or tar.gz archive.
pub const MAX_TAR_ARCHIVE_BYTES: u64 = 256 * 1024 * 1024;
/// Maximum decoded size of one archive member.
pub const MAX_ARCHIVE_MEMBER_BYTES: u64 = 64 * 1024 * 1024;

/// Formats supported by archive reads.
pub use omp_ar::Format as ArchiveFormat;

/// One possible split of `archive.ext:member/path`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArchivePathCandidate {
	/// Authored path through the recognized archive extension.
	pub archive_path: String,
	/// Authored member path after the extension and colon separators.
	pub sub_path:     String,
}

/// Metadata for one archive node.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArchiveNode {
	/// Normalized slash-separated member path, or empty for the archive root.
	pub path:         String,
	/// Whether this node is a directory.
	pub is_directory: bool,
	/// Uncompressed size in bytes. Directories have size zero.
	pub size:         u64,
	/// Modification time in milliseconds since the Unix epoch, when available.
	pub mtime_ms:     Option<u64>,
}

/// One immediate child in an archive-directory listing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArchiveDirectoryEntry {
	/// Child basename.
	pub name: String,
	/// Child metadata.
	pub node: ArchiveNode,
}

/// Result-limit metadata matching read's one-based archive listing behavior.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArchiveListLimit {
	/// Number of entries returned at the limit boundary.
	pub reached:    usize,
	/// Suggested wider limit.
	pub suggestion: usize,
}

/// A sliced immediate-child archive-directory listing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArchiveListing {
	/// Normalized directory path, or empty for the root.
	pub path:          String,
	/// Entries after applying the one-based offset and result limit.
	pub entries:       Vec<ArchiveDirectoryEntry>,
	/// Number of immediate children before offset and limit.
	pub total_entries: usize,
	/// One-based offset used for this listing.
	pub offset:        usize,
	/// Result-limit truth for callers that render shared limit notices.
	pub result_limit:  Option<ArchiveListLimit>,
}

impl ArchiveListing {
	/// Renders entries using pi's archive-listing size suffixes.
	pub fn render(&self) -> String {
		if self.entries.is_empty() {
			return "(empty archive directory)".to_owned();
		}
		format_archive_entry_lines(&self.entries).join("\n")
	}
}

/// Extracted bytes and metadata for one archive member.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArchiveMember {
	/// Member metadata.
	pub node:  ArchiveNode,
	/// Uncompressed member bytes.
	pub bytes: Bytes,
}

/// A valid UTF-8, NUL-free archive text member.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArchiveTextMember {
	/// Member metadata.
	pub node: ArchiveNode,
	/// Decoded text.
	pub text: String,
}

/// Typed truth for a member that cannot enter the text pipeline.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArchiveBinaryMember {
	/// Member metadata, including its exact uncompressed byte size.
	pub node:   ArchiveNode,
	/// Model-facing binary notice.
	pub notice: String,
}

/// The content selected from an archive.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArchiveContent {
	/// An archive-root or member-directory listing.
	Directory(ArchiveListing),
	/// A member accepted by the UTF-8 text classifier.
	Text(ArchiveTextMember),
	/// A binary or invalid-UTF-8 member.
	Binary(ArchiveBinaryMember),
}

/// An archive read with the member selector preserved for the standard text
/// formatter.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArchiveRead {
	/// Selector composed after member-name resolution.
	pub selector: ParsedSelector,
	/// Selected archive content.
	pub content:  ArchiveContent,
}

/// One text member in deterministic whole-archive materialization.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaterializedTextMember {
	/// Normalized archive member path.
	pub path: String,
	/// Exact uncompressed byte size.
	pub size: u64,
	/// Decoded member text.
	pub text: String,
}

/// Deterministic archive materialization for grep and similar consumers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TextMemberMaterialization {
	/// Text members sorted by normalized archive path.
	pub members:        Vec<MaterializedTextMember>,
	/// Binary members sorted by normalized archive path.
	pub binary_members: Vec<ArchiveBinaryMember>,
}

/// Typed archive failure. Binary members are successful
/// `ArchiveContent::Binary` values rather than failures.
#[derive(Debug, thiserror::Error)]
pub enum ArchiveError {
	/// The path does not name a supported archive family.
	#[error("Unsupported archive format: {path}")]
	UnsupportedFormat {
		/// Authored archive path.
		path: String,
	},
	/// Filesystem archive access failed.
	#[error("Failed to {action} archive '{path}': {source}")]
	Io {
		/// Operation being attempted.
		action: &'static str,
		/// Filesystem path.
		path:   PathBuf,
		/// Underlying I/O error.
		#[source]
		source: std::io::Error,
	},
	/// Container metadata or compressed data is invalid.
	#[error("Invalid {format} archive: {message}")]
	InvalidArchive {
		/// Display format name.
		format:  &'static str,
		/// Parser detail.
		message: String,
	},
	/// A tar-family input exceeds the bounded in-memory parser limit.
	#[error("Archive is too large to read in memory ({size} > {limit} limit)")]
	ArchiveTooLarge {
		/// Formatted input or decoded size.
		size:  String,
		/// Formatted archive limit.
		limit: String,
	},
	/// A lookup attempted to escape the archive root.
	#[error("Archive path cannot contain '..'")]
	UnsafePath,
	/// A requested node was absent.
	#[error("Archive path '{path}' not found")]
	NotFound {
		/// Normalized requested path.
		path: String,
	},
	/// A requested file was a directory.
	#[error("Archive path '{path}' is a directory")]
	IsDirectory {
		/// Normalized requested path.
		path: String,
	},
	/// A requested directory was a file.
	#[error("Archive path '{path}' is not a directory")]
	NotDirectory {
		/// Normalized requested path.
		path: String,
	},
	/// A member's declared uncompressed size exceeds the extraction boundary.
	#[error("Archive member '{path}' is too large to extract in memory ({size} > {limit} limit)")]
	MemberTooLarge {
		/// Normalized member path.
		path:  String,
		/// Formatted declared size.
		size:  String,
		/// Formatted extraction limit.
		limit: String,
	},
	/// A tar sparse member cannot be represented by a contiguous byte slice.
	#[error("Archive member '{path}' is a sparse file and cannot be read")]
	SparseMember {
		/// Normalized member path.
		path: String,
	},
	/// A tar link has no safe, materializable target.
	#[error("Archive symlink '{path}' cannot be materialized from target '{target}'")]
	UnreadableLink {
		/// Link member path.
		path:   String,
		/// Authored normalized target.
		target: String,
	},
	/// A member selector was invalid.
	#[error("{0}")]
	Selector(#[from] SelectorError),
	/// Directory aliases formed a cycle.
	#[error("Archive path '{path}' crosses a cyclic symlink")]
	CyclicLink {
		/// Requested path.
		path: String,
	},
	/// Multi-range selectors do not map to directory entry slices.
	#[error("Multi-range line selectors are not supported for archive directory listings.")]
	DirectoryMultiRange,
}

/// Indexed archive reader backed by `omp-ar`.
pub struct ArchiveReader<R> {
	archive: Archive<R>,
}

/// Infers an archive format from a filename. Matching is ASCII
/// case-insensitive.
pub fn archive_format_from_path(path: impl AsRef<Path>) -> Option<ArchiveFormat> {
	ArchiveFormat::from_path(path)
}

/// Sniffs a ZIP, gzip, or ustar signature from in-memory bytes.
pub fn sniff_archive_format(bytes: &[u8]) -> Option<ArchiveFormat> {
	ArchiveFormat::sniff(bytes)
}

/// Split every plausible archive extension boundary, longest archive prefix
/// first. Only an extension followed by `:` or end-of-input is a boundary.
pub fn parse_archive_path_candidates(path: &str) -> Vec<ArchivePathCandidate> {
	const EXTENSIONS: [&str; 7] = [".tar.gz", ".tgz", ".zip", ".tar", ".jar", ".war", ".ear"];
	const APK: &str = ".apk";

	let normalized = path.replace('\\', "/");
	let lower = normalized.to_ascii_lowercase();
	let mut candidates = Vec::new();
	let mut seen = std::collections::BTreeSet::new();
	for (start, byte) in lower.bytes().enumerate() {
		if byte != b'.' {
			continue;
		}
		for extension in EXTENSIONS.iter().copied().chain(std::iter::once(APK)) {
			if !lower[start..].starts_with(extension) {
				continue;
			}
			let end = start + extension.len();
			if end != lower.len() && lower.as_bytes().get(end) != Some(&b':') {
				continue;
			}
			let archive_path = path[..end].to_owned();
			let sub_path = normalized[end..].trim_start_matches(':').to_owned();
			if seen.insert((archive_path.clone(), sub_path.clone())) {
				candidates.push(ArchivePathCandidate { archive_path, sub_path });
			}
			break;
		}
	}
	candidates.sort_by(|left, right| right.archive_path.len().cmp(&left.archive_path.len()));
	candidates
}

/// Opens an archive by filesystem path using the format implied by its name.
pub fn open_archive_path(
	path: impl AsRef<Path>,
) -> Result<ArchiveReader<BufReader<File>>, ArchiveError> {
	let path = path.as_ref();
	let format = archive_format_from_path(path).ok_or_else(|| ArchiveError::UnsupportedFormat {
		path: path.to_string_lossy().into_owned(),
	})?;
	let file = File::open(path).map_err(|source| ArchiveError::Io {
		action: "open",
		path: path.to_path_buf(),
		source,
	})?;
	let archive = Archive::with_format_and_limits(BufReader::new(file), format, archive_limits())
		.map_err(|error| archive_error(error, format))?;
	Ok(ArchiveReader { archive })
}

/// Opens format-tagged archive bytes. Members remain lazy over the retained
/// immutable byte buffer where the source format permits it.
pub fn open_archive_bytes(
	bytes: Bytes,
	format: ArchiveFormat,
) -> Result<ArchiveReader<Cursor<Bytes>>, ArchiveError> {
	let archive = Archive::with_format_and_limits(Cursor::new(bytes), format, archive_limits())
		.map_err(|error| archive_error(error, format))?;
	Ok(ArchiveReader { archive })
}

/// Reads one root, directory, or member target from an archive path.
pub fn read_archive_path(
	path: impl AsRef<Path>,
	target: &str,
	selector: ParsedSelector,
) -> Result<ArchiveRead, ArchiveError> {
	open_archive_path(path)?.read_target(target, selector)
}

/// Reads one root, directory, or member target from format-tagged bytes.
pub fn read_archive_bytes(
	bytes: Bytes,
	format: ArchiveFormat,
	target: &str,
	selector: ParsedSelector,
) -> Result<ArchiveRead, ArchiveError> {
	open_archive_bytes(bytes, format)?.read_target(target, selector)
}

impl<R: Read + Seek> ArchiveReader<R> {
	/// Container format backing this reader.
	pub const fn format(&self) -> ArchiveFormat {
		self.archive.format()
	}

	/// Returns metadata for one safe archive node. The empty path is the root.
	pub fn node(&self, path: &str) -> Option<ArchiveNode> {
		let normalized = normalize_lookup_path(path)?;
		if normalized.is_empty() {
			return Some(root_node());
		}
		let entry = self.archive.resolve_entry(&normalized).ok()?;
		Some(node_from_entry(entry, normalized))
	}

	/// Lists immediate children of one safe archive directory in stable,
	/// case-insensitive name order.
	pub fn list_directory(&self, path: &str) -> Result<Vec<ArchiveDirectoryEntry>, ArchiveError> {
		let normalized = normalize_lookup_path(path).ok_or(ArchiveError::UnsafePath)?;
		let format = self.format();
		let mut entries: Vec<_> = self
			.archive
			.list(&normalized)
			.map_err(|error| archive_error(error, format))?
			.into_iter()
			.map(|entry| {
				let name = entry.name().to_owned();
				let path = if normalized.is_empty() {
					name.clone()
				} else {
					format!("{normalized}/{name}")
				};
				ArchiveDirectoryEntry { node: node_from_entry(entry, path), name }
			})
			.collect();
		entries.sort_by(|left, right| {
			left
				.name
				.to_ascii_lowercase()
				.cmp(&right.name.to_ascii_lowercase())
				.then_with(|| left.name.cmp(&right.name))
		});
		Ok(entries)
	}

	/// Reads and decompresses one file member on demand.
	pub fn read_member(&mut self, path: &str) -> Result<ArchiveMember, ArchiveError> {
		let normalized = normalize_lookup_path(path).ok_or(ArchiveError::UnsafePath)?;
		if normalized.is_empty() {
			return Err(ArchiveError::IsDirectory { path: normalized });
		}
		let format = self.format();
		let node = {
			let entry = self
				.archive
				.resolve_entry(&normalized)
				.map_err(|error| archive_error(error, format))?;
			node_from_entry(entry, normalized.clone())
		};
		if node.is_directory {
			return Err(ArchiveError::IsDirectory { path: normalized });
		}
		let bytes = self
			.archive
			.read(&normalized)
			.map(Bytes::from)
			.map_err(|error| archive_error(error, format))?;
		Ok(ArchiveMember { node, bytes })
	}

	/// Reads one member through the strict NUL-free UTF-8 classifier.
	pub fn read_text_member(
		&mut self,
		path: &str,
	) -> Result<Result<ArchiveTextMember, ArchiveBinaryMember>, ArchiveError> {
		let member = self.read_member(path)?;
		match decode_utf8_text(&member.bytes) {
			Some(text) => Ok(Ok(ArchiveTextMember { node: member.node, text })),
			None => {
				let notice = binary_member_notice(&member.node.path, member.node.size);
				Ok(Err(ArchiveBinaryMember { node: member.node, notice }))
			},
		}
	}

	/// Resolves member-name precedence, composes an embedded selector when
	/// needed, then returns a directory listing or classified text/binary
	/// member.
	pub fn read_target(
		&mut self,
		target: &str,
		mut selector: ParsedSelector,
	) -> Result<ArchiveRead, ArchiveError> {
		let mut member_path = target;
		if self.node(member_path).is_none() && matches!(selector, ParsedSelector::None) {
			let split = split_path_and_selector(target);
			if split.selector.is_some() && self.node(split.path).is_some() {
				member_path = split.path;
				selector = parse_selector(split.selector)?;
			} else if let parsed @ (ParsedSelector::Raw
			| ParsedSelector::Conflicts
			| ParsedSelector::Lines { .. }) = parse_selector(Some(target))?
			{
				member_path = "";
				selector = parsed;
			}
		}
		let node = self
			.node(member_path)
			.ok_or_else(|| ArchiveError::NotFound { path: target.to_owned() })?;
		let content = if node.is_directory {
			if selector.is_multi_range() {
				return Err(ArchiveError::DirectoryMultiRange);
			}
			let (offset, limit) = selector.offset_limit();
			ArchiveContent::Directory(self.read_directory_slice(member_path, offset, limit)?)
		} else {
			match self.read_text_member(member_path)? {
				Ok(text) => ArchiveContent::Text(text),
				Err(binary) => ArchiveContent::Binary(binary),
			}
		};
		Ok(ArchiveRead { selector, content })
	}

	/// Materializes every readable text member in normalized path order while
	/// retaining typed binary-member truth. This is the shared grep seam.
	pub fn materialize_text_members(&mut self) -> Result<TextMemberMaterialization, ArchiveError> {
		let paths: Vec<_> = self
			.archive
			.entries()
			.filter(|entry| !entry.is_directory())
			.map(|entry| entry.path().to_owned())
			.collect();
		let mut members = Vec::new();
		let mut binary_members = Vec::new();
		for path in paths {
			match self.read_text_member(&path)? {
				Ok(member) => members.push(MaterializedTextMember {
					path: member.node.path,
					size: member.node.size,
					text: member.text,
				}),
				Err(binary) => binary_members.push(binary),
			}
		}
		Ok(TextMemberMaterialization { members, binary_members })
	}

	fn read_directory_slice(
		&self,
		path: &str,
		offset: Option<u64>,
		limit: Option<u64>,
	) -> Result<ArchiveListing, ArchiveError> {
		let all = self.list_directory(path)?;
		let total_entries = all.len();
		let offset = usize::try_from(offset.unwrap_or(1).max(1)).unwrap_or(usize::MAX);
		let start = offset.saturating_sub(1).min(all.len());
		let available = &all[start..];
		let limit =
			usize::try_from(limit.unwrap_or(DEFAULT_ARCHIVE_LIST_LIMIT as u64)).unwrap_or(usize::MAX);
		let limit = if limit == 0 {
			DEFAULT_ARCHIVE_LIST_LIMIT
		} else {
			limit
		};
		let reached = available.len() >= limit;
		let entries = available[..available.len().min(limit)].to_vec();
		Ok(ArchiveListing {
			path: normalize_lookup_path(path).ok_or(ArchiveError::UnsafePath)?,
			entries,
			total_entries,
			offset,
			result_limit: reached
				.then_some(ArchiveListLimit { reached: limit, suggestion: limit.saturating_mul(2) }),
		})
	}
}

/// Formats archive directory entries using pi's exact size suffix rules.
pub fn format_archive_entry_lines(entries: &[ArchiveDirectoryEntry]) -> Vec<String> {
	entries
		.iter()
		.map(|entry| {
			if entry.node.is_directory {
				format!("{}/", entry.name)
			} else if entry.node.size > 0 {
				format!("{} ({})", entry.name, format_bytes(entry.node.size))
			} else {
				entry.name.clone()
			}
		})
		.collect()
}

/// Strict archive text classifier: reject NUL bytes and malformed UTF-8.
pub fn decode_utf8_text(bytes: &[u8]) -> Option<String> {
	if bytes.contains(&0) {
		return None;
	}
	std::str::from_utf8(bytes).ok().map(ToOwned::to_owned)
}

/// Exact model-facing notice for a binary archive entry.
pub fn binary_member_notice(path: &str, size: u64) -> String {
	format!("[Cannot read binary archive entry '{path}' ({})]", format_bytes(size))
}

fn archive_limits() -> Limits {
	Limits::DEFAULT
		.with_max_archive_size(MAX_TAR_ARCHIVE_BYTES)
		.with_max_member_size(MAX_ARCHIVE_MEMBER_BYTES)
}

fn archive_error(error: ArError, format: ArchiveFormat) -> ArchiveError {
	match error {
		ArError::ArchiveTooLarge { actual, limit } => {
			ArchiveError::ArchiveTooLarge { size: format_bytes(actual), limit: format_bytes(limit) }
		},
		ArError::UnsafePath(_) => ArchiveError::UnsafePath,
		ArError::NotFound(path) => ArchiveError::NotFound { path: path.to_string() },
		ArError::IsDirectory(path) => ArchiveError::IsDirectory { path: path.to_string() },
		ArError::NotDirectory(path) => ArchiveError::NotDirectory { path: path.to_string() },
		ArError::MemberTooLarge { path, actual, limit } => ArchiveError::MemberTooLarge {
			path:  path.to_string(),
			size:  format_bytes(actual),
			limit: format_bytes(limit),
		},
		ArError::SparseMember(path) => ArchiveError::SparseMember { path: path.to_string() },
		ArError::UnreadableLink { path, target } => {
			ArchiveError::UnreadableLink { path: path.to_string(), target: target.to_string() }
		},
		ArError::LinkResolutionDepth { path, .. } => {
			ArchiveError::CyclicLink { path: path.to_string() }
		},
		other => {
			ArchiveError::InvalidArchive { format: format_name(format), message: other.to_string() }
		},
	}
}

const fn format_name(format: ArchiveFormat) -> &'static str {
	match format {
		ArchiveFormat::Zip => "ZIP",
		ArchiveFormat::Tar => "tar",
		ArchiveFormat::TarGz => "tar.gz",
	}
}

fn node_from_entry(entry: &Entry, path: String) -> ArchiveNode {
	ArchiveNode {
		path,
		is_directory: entry.is_directory(),
		size: entry.size(),
		mtime_ms: entry
			.modified_unix_seconds()
			.and_then(|seconds| seconds.checked_mul(1000)),
	}
}

fn normalize_lookup_path(path: &str) -> Option<String> {
	if path.starts_with(['/', '\\']) || path.contains('\0') {
		return None;
	}

	let mut normalized = String::with_capacity(path.len());
	let mut first = true;
	for part in path.split(['/', '\\']) {
		if part.is_empty() || part == "." {
			continue;
		}
		if part == ".." || first && is_windows_drive(part) {
			return None;
		}
		if !first {
			normalized.push('/');
		}
		normalized.push_str(part);
		first = false;
	}
	Some(normalized)
}

const fn is_windows_drive(component: &str) -> bool {
	let bytes = component.as_bytes();
	bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':'
}

fn root_node() -> ArchiveNode {
	ArchiveNode {
		path:         String::new(),
		is_directory: true,
		size:         0,
		mtime_ms:     None,
	}
}

fn format_bytes(bytes: u64) -> String {
	const KB: f64 = 1024.0;
	const MB: f64 = 1024.0 * 1024.0;
	const GB: f64 = 1024.0 * 1024.0 * 1024.0;
	if bytes < 1024 {
		format!("{bytes}B")
	} else if bytes < 1024 * 1024 {
		format!("{:.1}KB", bytes as f64 / KB)
	} else if bytes < 1024 * 1024 * 1024 {
		format!("{:.1}MB", bytes as f64 / MB)
	} else {
		format!("{:.1}GB", bytes as f64 / GB)
	}
}
