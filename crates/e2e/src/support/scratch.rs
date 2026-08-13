use std::{fs, io, path::{Path, PathBuf}};

use tempfile::TempDir;
use url::Url;

/// Isolated project and daemon-state roots removed recursively on drop.
#[derive(Debug)]
pub struct Scratch {
	root: TempDir,
	project: PathBuf,
	state: PathBuf,
}

impl Scratch {
	/// Creates a fresh project/state pair below one private temporary root.
	pub fn new() -> io::Result<Self> {
		let root = tempfile::Builder::new().prefix("omp-e2e-").tempdir()?;
		let project = root.path().join("project");
		let state = root.path().join("state");
		fs::create_dir_all(&project)?;
		fs::create_dir_all(&state)?;
		Ok(Self { root, project, state })
	}

	/// Returns the private root containing all fixture-owned paths.
	#[must_use]
	pub fn root(&self) -> &Path {
		self.root.path()
	}

	/// Returns the scratch workspace root.
	#[must_use]
	pub fn project(&self) -> &Path {
		&self.project
	}

	/// Returns the state root kept outside the workspace.
	#[must_use]
	pub fn state(&self) -> &Path {
		&self.state
	}

	/// Returns the canonical file URI for the workspace root.
	pub fn project_uri(&self) -> io::Result<String> {
		Url::from_directory_path(self.project()).map(String::from).map_err(|()| {
			io::Error::new(io::ErrorKind::InvalidInput, "scratch project is not an absolute file path")
		})
	}

	/// Returns an endpoint path under the private state directory.
	#[must_use]
	pub fn socket(&self, name: &str) -> PathBuf {
		self.state.join(name)
	}

	/// Creates parent directories and writes one project-relative file.
	pub fn write(&self, relative: impl AsRef<Path>, bytes: impl AsRef<[u8]>) -> io::Result<PathBuf> {
		let path = self.project.join(relative);
		if let Some(parent) = path.parent() {
			fs::create_dir_all(parent)?;
		}
		fs::write(&path, bytes)?;
		Ok(path)
	}

	/// Reads one project-relative file.
	pub fn read(&self, relative: impl AsRef<Path>) -> io::Result<Vec<u8>> {
		fs::read(self.project.join(relative))
	}
}
