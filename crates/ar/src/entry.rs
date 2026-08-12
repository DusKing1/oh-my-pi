//! Format-neutral indexed archive member metadata.

use omp_core::Str;

/// Compression method recorded for a ZIP member.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionMethod {
	/// Bytes are stored verbatim.
	Stored,
	/// Bytes use raw DEFLATE compression.
	Deflate,
	/// The method is retained for listing but cannot be decoded.
	Unsupported(u16),
}

impl CompressionMethod {
	pub(crate) const fn from_code(code: u16) -> Self {
		match code {
			0 => Self::Stored,
			8 => Self::Deflate,
			other => Self::Unsupported(other),
		}
	}

	/// Returns the ZIP wire-format method number.
	pub const fn code(self) -> u16 {
		match self {
			Self::Stored => 0,
			Self::Deflate => 8,
			Self::Unsupported(code) => code,
		}
	}
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Storage {
	Synthetic,
	Zip {
		compressed_size:     u64,
		crc32:               u32,
		method:              CompressionMethod,
		flags:               u16,
		local_header_offset: u64,
	},
	Tar {
		data_offset: u64,
		stored_size: u64,
		sparse:      bool,
	},
	TarLink {
		target_path: Str,
	},
}

/// One normalized file, directory, or unresolved symbolic link in an archive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
	pub(crate) path:                  Str,
	pub(crate) directory:             bool,
	pub(crate) size:                  u64,
	pub(crate) modified_unix_seconds: Option<u64>,
	pub(crate) storage:               Storage,
}

impl Entry {
	pub(crate) const fn synthetic_directory(path: Str) -> Self {
		Self {
			path,
			directory: true,
			size: 0,
			modified_unix_seconds: None,
			storage: Storage::Synthetic,
		}
	}

	/// Returns the normalized archive-relative path.
	#[inline]
	pub fn path(&self) -> &str {
		self.path.as_str()
	}

	/// Returns the final component of the normalized path.
	#[inline]
	pub fn name(&self) -> &str {
		self.path.rsplit('/').next().unwrap_or(self.path.as_str())
	}

	/// Returns whether this entry represents a directory.
	#[inline]
	pub const fn is_directory(&self) -> bool {
		self.directory
	}

	/// Returns the declared logical size in bytes.
	#[inline]
	pub const fn size(&self) -> u64 {
		self.size
	}

	/// Returns the stored member size before ZIP decompression or TAR sparse
	/// expansion.
	#[inline]
	pub const fn compressed_size(&self) -> u64 {
		match &self.storage {
			Storage::Zip { compressed_size, .. } => *compressed_size,
			Storage::Tar { stored_size, .. } => *stored_size,
			Storage::Synthetic | Storage::TarLink { .. } => 0,
		}
	}

	/// Returns the ZIP compression method, or `None` for TAR members.
	#[inline]
	pub const fn zip_compression(&self) -> Option<CompressionMethod> {
		match &self.storage {
			Storage::Zip { method, .. } => Some(*method),
			Storage::Synthetic | Storage::Tar { .. } | Storage::TarLink { .. } => None,
		}
	}

	/// Returns the declared ZIP CRC-32, or `None` for TAR members.
	#[inline]
	pub const fn crc32(&self) -> Option<u32> {
		match &self.storage {
			Storage::Zip { crc32, .. } => Some(*crc32),
			Storage::Synthetic | Storage::Tar { .. } | Storage::TarLink { .. } => None,
		}
	}

	/// Returns whether this ZIP member declares traditional encryption.
	#[inline]
	pub fn is_encrypted(&self) -> bool {
		matches!(&self.storage, Storage::Zip { flags, .. } if flags & 1 != 0)
	}

	/// Returns whether this entry is an unresolved TAR symbolic-link node.
	#[inline]
	pub const fn is_link(&self) -> bool {
		matches!(&self.storage, Storage::TarLink { .. })
	}

	/// Returns an unresolved TAR link target.
	#[inline]
	pub fn link_target(&self) -> Option<&str> {
		match &self.storage {
			Storage::TarLink { target_path } => Some(target_path.as_str()),
			Storage::Synthetic | Storage::Zip { .. } | Storage::Tar { .. } => None,
		}
	}

	/// Returns the member modification time as Unix seconds when recorded.
	#[inline]
	pub const fn modified_unix_seconds(&self) -> Option<u64> {
		self.modified_unix_seconds
	}
}
