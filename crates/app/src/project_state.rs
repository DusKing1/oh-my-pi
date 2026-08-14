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

/// Returns the short owner-local environment socket path for `state_dir`.
///
/// The path is keyed by the running executable's build identity: a rebuilt
/// `omp` binds its own listener immediately while stale-build listeners drain
/// and idle-exit, with no takeover protocol. The document socket stays
/// build-stable because its authority must remain singular per project.
#[cfg(unix)]
pub(crate) fn environment_socket(state_dir: &Path) -> PathBuf {
	let build = crate::build_id::current();
	let key = if build.is_empty() {
		"unknown"
	} else {
		&build[..8]
	};
	socket_path(state_dir, &format!("{key}-env"))
}

/// Returns the short owner-local document socket path for `state_dir`.
#[cfg(unix)]
pub fn document_socket(state_dir: &Path) -> PathBuf {
	socket_path(state_dir, "doc")
}

#[cfg(unix)]
fn socket_path(state_dir: &Path, kind: &str) -> PathBuf {
	let digest = blake3::hash(state_dir.as_os_str().as_encoded_bytes());
	let short: [u8; 16] = digest.as_bytes()[..16]
		.try_into()
		.expect("a Blake3 digest contains 16 prefix bytes");
	PathBuf::from("/tmp").join(format!(
		"omp-{}-{}-{kind}.sock",
		nix::unistd::geteuid().as_raw(),
		hex::encode_n(&short)
	))
}

#[cfg(all(test, unix))]
mod tests {
	use std::path::PathBuf;

	use super::{document_socket, environment_socket};

	#[test]
	fn socket_paths_fit_the_platform_address_limit() {
		let state_dir = PathBuf::from("/").join("long-project-state-segment".repeat(32));
		let env = environment_socket(&state_dir);
		let docs = document_socket(&state_dir);
		// SAFETY: every all-zero bit pattern is valid for libc's sockaddr_un
		// integer fields and fixed-size character array.
		let address: libc::sockaddr_un = unsafe { std::mem::zeroed() };
		let capacity = address.sun_path.len();

		assert_ne!(env, docs);
		assert!(env.as_os_str().as_encoded_bytes().len() < capacity);
		assert!(docs.as_os_str().as_encoded_bytes().len() < capacity);
	}
}
