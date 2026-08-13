//! Verified, root-confined model artifacts for local inference.

use std::{
	fs::{self, File},
	io::Read,
	path::{Component, Path, PathBuf},
};

use omp_core::Str;
use sha2::{Digest, Sha256};

use super::runtime::{LocalCancellation, LocalError, LocalErrorKind, LocalResult};

/// Immutable expected identity of a local model artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactSpec {
	/// Root-relative artifact path.
	pub path:   PathBuf,
	/// Exact expected file length.
	pub bytes:  u64,
	/// Exact SHA-256 digest.
	pub sha256: [u8; 32],
}

/// Evidence produced after reading and hashing an artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactReceipt {
	/// Canonical verified path.
	pub path:   PathBuf,
	/// Observed file length.
	pub bytes:  u64,
	/// Observed SHA-256 digest.
	pub sha256: [u8; 32],
}

/// Root-confined storage boundary for model files.
#[derive(Clone, Debug)]
pub struct ArtifactStore {
	root: PathBuf,
}

impl ArtifactStore {
	/// Opens an existing artifact root and resolves it against symlinks.
	pub fn open(root: impl AsRef<Path>) -> LocalResult<Self> {
		let root = fs::canonicalize(root.as_ref()).map_err(|error| {
			LocalError::new(LocalErrorKind::Artifact, format!("artifact root is unavailable: {error}"))
		})?;
		if !root.is_dir() {
			return Err(LocalError::new(LocalErrorKind::Artifact, "artifact root is not a directory"));
		}
		Ok(Self { root })
	}

	/// Returns the canonical storage root.
	pub fn root(&self) -> &Path {
		&self.root
	}

	/// Verifies path confinement, size, and digest before returning a usable
	/// file.
	pub fn verify(
		&self,
		spec: &ArtifactSpec,
		cancel: &LocalCancellation,
	) -> LocalResult<VerifiedArtifact> {
		validate_relative(&spec.path)?;
		if cancel.is_cancelled() {
			return Err(LocalError::cancelled());
		}
		let candidate = self.root.join(&spec.path);
		let metadata = fs::symlink_metadata(&candidate).map_err(|error| {
			LocalError::new(LocalErrorKind::Artifact, format!("artifact metadata failed: {error}"))
		})?;
		if metadata.file_type().is_symlink() || !metadata.is_file() {
			return Err(LocalError::new(
				LocalErrorKind::Artifact,
				"artifact must be a regular file, not a symlink",
			));
		}
		let canonical = fs::canonicalize(&candidate).map_err(|error| {
			LocalError::new(LocalErrorKind::Artifact, format!("artifact path failed: {error}"))
		})?;
		if !canonical.starts_with(&self.root) {
			return Err(LocalError::new(
				LocalErrorKind::Artifact,
				"artifact escapes its storage root",
			));
		}
		if metadata.len() != spec.bytes {
			return Err(LocalError::new(
				LocalErrorKind::Artifact,
				format!(
					"artifact length mismatch: expected {}, observed {}",
					spec.bytes,
					metadata.len()
				),
			));
		}
		let mut file = File::open(&canonical).map_err(|error| {
			LocalError::new(LocalErrorKind::Artifact, format!("artifact open failed: {error}"))
		})?;
		let opened_metadata = file.metadata().map_err(|error| {
			LocalError::new(
				LocalErrorKind::Artifact,
				format!("opened artifact metadata failed: {error}"),
			)
		})?;
		if !opened_metadata.is_file() || opened_metadata.len() != metadata.len() {
			return Err(LocalError::new(
				LocalErrorKind::Artifact,
				"artifact changed while it was being opened",
			));
		}
		#[cfg(unix)]
		{
			use std::os::unix::fs::MetadataExt;
			if opened_metadata.dev() != metadata.dev() || opened_metadata.ino() != metadata.ino() {
				return Err(LocalError::new(
					LocalErrorKind::Artifact,
					"artifact changed while it was being opened",
				));
			}
		}
		let mut digest = Sha256::new();
		let mut buffer = [0_u8; 64 * 1024];
		loop {
			if cancel.is_cancelled() {
				return Err(LocalError::cancelled());
			}
			let read = file.read(&mut buffer).map_err(|error| {
				LocalError::new(LocalErrorKind::Artifact, format!("artifact read failed: {error}"))
			})?;
			if read == 0 {
				break;
			}
			digest.update(&buffer[..read]);
		}
		let observed: [u8; 32] = digest.finalize().into();
		if observed != spec.sha256 {
			return Err(LocalError::new(LocalErrorKind::Artifact, "artifact SHA-256 mismatch"));
		}
		Ok(VerifiedArtifact {
			file,
			receipt: ArtifactReceipt {
				path:   canonical,
				bytes:  opened_metadata.len(),
				sha256: observed,
			},
		})
	}
}

fn validate_relative(path: &Path) -> LocalResult<()> {
	if path.as_os_str().is_empty()
		|| path.is_absolute()
		|| path
			.components()
			.any(|component| !matches!(component, Component::Normal(_)))
	{
		return Err(LocalError::new(
			LocalErrorKind::Artifact,
			"artifact path must be a non-empty normalized relative path",
		));
	}
	Ok(())
}

/// Open file proven to match its declared immutable identity.
pub struct VerifiedArtifact {
	file:    File,
	receipt: ArtifactReceipt,
}

impl VerifiedArtifact {
	/// Borrows the verified open file.
	pub const fn file(&self) -> &File {
		&self.file
	}

	/// Returns verification evidence.
	pub const fn receipt(&self) -> &ArtifactReceipt {
		&self.receipt
	}

	/// Returns the canonical verified file path for engines requiring a path.
	pub fn path(&self) -> &Path {
		&self.receipt.path
	}
}

impl std::fmt::Debug for VerifiedArtifact {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		formatter
			.debug_struct("VerifiedArtifact")
			.field("receipt", &self.receipt)
			.finish()
	}
}

/// Describes an artifact failure without exposing untrusted path input.
pub fn artifact_failure(message: impl Into<Str>) -> LocalError {
	LocalError::new(LocalErrorKind::Artifact, message)
}
