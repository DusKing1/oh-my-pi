//! Archive parsing, policy, codec, and I/O failures.

use std::io;

use omp_core::Str;

/// Result type used by `omp-ar` operations.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Failure produced while reading or writing an archive.
#[derive(Debug, thiserror::Error)]
pub enum Error {
	/// The underlying reader, writer, codec, or filesystem operation failed.
	#[error("archive I/O failed: {0}")]
	Io(#[from] io::Error),
	/// The input does not identify a supported archive format.
	#[error("unsupported or unrecognized archive format")]
	UnknownFormat,
	/// Container metadata is malformed, inconsistent, or truncated.
	#[error("invalid archive: {0}")]
	InvalidArchive(&'static str),
	/// Indexed records plus synthesized directories exceed the configured limit.
	#[error("archive has too many entries ({actual} > {limit})")]
	TooManyEntries {
		/// Indexed node count at rejection.
		actual: u64,
		/// Configured entry limit.
		limit:  u64,
	},
	/// A normalized member path exceeds the configured byte limit.
	#[error("archive member path is too long ({actual} bytes > {limit} byte limit)")]
	PathTooLong {
		/// Path byte length at rejection.
		actual: u64,
		/// Configured path byte limit.
		limit:  u64,
	},
	/// A normalized member path exceeds the configured component-depth limit.
	#[error(
		"archive member path is nested too deeply ({actual} components > {limit} component limit)"
	)]
	PathTooDeep {
		/// Normalized component count.
		actual: u64,
		/// Configured component limit.
		limit:  u64,
	},
	/// Format metadata exceeds the configured in-memory index limit.
	#[error("archive index is too large ({actual} bytes > {limit} byte limit)")]
	IndexTooLarge {
		/// Declared index size.
		actual: u64,
		/// Configured index limit.
		limit:  u64,
	},
	/// A TAR input or decoded TAR.GZ stream exceeds the configured archive
	/// limit.
	#[error("archive is too large ({actual} bytes > {limit} byte limit)")]
	ArchiveTooLarge {
		/// Input or decoded byte count at rejection.
		actual: u64,
		/// Configured archive limit.
		limit:  u64,
	},
	/// A member exceeds the configured per-member extraction limit.
	#[error("archive member '{path}' is too large ({actual} bytes > {limit} byte limit)")]
	MemberTooLarge {
		/// Normalized member path.
		path:   Str,
		/// Larger relevant stored or logical size.
		actual: u64,
		/// Configured member limit.
		limit:  u64,
	},
	/// Materializing all files would exceed the configured aggregate limit.
	#[error("archive is too large to materialize ({actual} bytes > {limit} byte limit)")]
	ArchiveTooLargeInMemory {
		/// Aggregate declared file size.
		actual: u64,
		/// Configured aggregate limit.
		limit:  u64,
	},
	/// A caller-supplied member path is absolute, empty, or contains `..`.
	#[error("unsafe archive member path: '{0}'")]
	UnsafePath(Str),
	/// No indexed member has the requested path.
	#[error("archive member '{0}' was not found")]
	NotFound(Str),
	/// A file operation targeted a directory.
	#[error("archive member '{0}' is a directory")]
	IsDirectory(Str),
	/// A directory operation targeted a file.
	#[error("archive member '{0}' is not a directory")]
	NotDirectory(Str),
	/// The requested ZIP member is encrypted.
	#[error("encrypted ZIP member '{0}' is not supported")]
	Encrypted(Str),
	/// The requested ZIP member uses an unsupported compression method.
	#[error("unsupported ZIP compression method {method} for member '{path}'")]
	UnsupportedCompression {
		/// Normalized member path.
		path:   Str,
		/// ZIP compression method number.
		method: u16,
	},
	/// Decoded bytes disagree with a member's declared size.
	#[error("archive member '{path}' has size {actual}, expected {expected}")]
	SizeMismatch {
		/// Normalized member path.
		path:     Str,
		/// Declared uncompressed size.
		expected: u64,
		/// Observed uncompressed size.
		actual:   u64,
	},
	/// Inflated ZIP bytes disagree with the central-directory CRC-32.
	#[error("ZIP member '{path}' has CRC-32 {actual:08x}, expected {expected:08x}")]
	ChecksumMismatch {
		/// Normalized member path.
		path:     Str,
		/// Declared CRC-32.
		expected: u32,
		/// Computed CRC-32.
		actual:   u32,
	},
	/// A sparse TAR member cannot be reconstructed by this reader.
	#[error("archive member '{0}' is a sparse file and cannot be read")]
	SparseMember(Str),
	/// A TAR symbolic link has no readable in-archive target.
	#[error("archive symlink '{path}' cannot be materialized from target '{target}'")]
	UnreadableLink {
		/// Link member path.
		path:   Str,
		/// Recorded link target.
		target: Str,
	},
	/// A TAR hard link targets an invalid member.
	#[error("archive hard link '{path}' targets {reason} '{target}'")]
	InvalidHardLink {
		/// Hard-link member path.
		path:   Str,
		/// Recorded link target.
		target: Str,
		/// Static target classification.
		reason: &'static str,
	},
	/// TAR links contain an unresolved dependency cycle.
	#[error("archive contains cyclic or unsupported links")]
	CyclicLinks,
	/// Resolving a TAR directory alias exceeded the configured rewrite bound.
	#[error("archive path '{path}' exceeds the {limit}-rewrite symbolic-link limit")]
	LinkResolutionDepth {
		/// Requested archive path.
		path:  Str,
		/// Configured rewrite limit.
		limit: u64,
	},
	/// A writer already contains the normalized member path.
	#[error("duplicate archive member path: '{0}'")]
	DuplicatePath(Str),
	/// A writer input exceeds an ordinary ZIP field and would require ZIP64.
	#[error("ZIP64 output is required but is not supported")]
	Zip64Required,
	/// A TAR value cannot fit its on-wire numeric field.
	#[error("TAR {0} value does not fit its header field")]
	TarFieldOverflow(&'static str),
	/// Raw DEFLATE compression failed while preparing a ZIP member.
	#[error("failed to compress ZIP member")]
	Compression(#[source] flate2::CompressError),
	/// Raw DEFLATE decompression failed for a ZIP member.
	#[error("invalid DEFLATE stream in ZIP member '{path}'")]
	Decompression {
		/// Normalized member path.
		path:   Str,
		/// Codec failure.
		#[source]
		source: flate2::DecompressError,
	},
}
