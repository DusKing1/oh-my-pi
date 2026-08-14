//! Project-scoped runtime state kept outside tool-writable workspaces.

use std::{
	io,
	path::{Path, PathBuf},
};

use omp_core::encoding::hex;

/// Resolves the durable state directory for a project beneath the application
/// data directory.
///
/// Canonicalizing the project root gives aliases and symlinked paths one stable
/// state identity.
pub fn directory(data_dir: &Path, project_root: &Path) -> io::Result<PathBuf> {
	let root = std::fs::canonicalize(project_root)?;
	let digest = blake3::hash(root.as_os_str().as_encoded_bytes());
	Ok(data_dir
		.join("projects")
		.join(hex::encode_n(digest.as_bytes()).as_str()))
}
