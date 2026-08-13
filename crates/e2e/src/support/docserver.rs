#![cfg(unix)]

use std::{io, path::{Path, PathBuf}, time::Duration};

use anyhow::{Context as _, Result};
use omp_app::envd::docs::DocumentHost;
use omp_docserver::daemon::{self, Transport};
use tokio::{net::UnixStream, task::JoinHandle};

use super::{DEFAULT_TIMEOUT, within};

/// Real document authority running on a private Unix socket in a cancellable task.
#[derive(Debug)]
pub struct DocServerTask {
	socket: PathBuf,
	task: JoinHandle<daemon::Result>,
}

impl DocServerTask {
	/// Starts a real docserver rooted at `project`, optionally with real LSP bindings.
	pub async fn spawn(
		project: impl Into<PathBuf>,
		socket: impl Into<PathBuf>,
		lsp_configs: Vec<PathBuf>,
	) -> Result<Self> {
		let project = project.into();
		let socket = socket.into();
		if let Some(parent) = socket.parent() {
			std::fs::create_dir_all(parent).context("creating docserver socket directory")?;
		}
		let task_socket = socket.clone();
		let task = tokio::spawn(async move {
			daemon::run(project, Transport::Socket(task_socket), lsp_configs).await
		});
		let mut server = Self { socket, task };
		within("docserver socket readiness", DEFAULT_TIMEOUT, async {
			loop {
				if server.task.is_finished() {
					let result = (&mut server.task).await.context("joining docserver startup task")?;
					result.context("docserver stopped during startup")?;
					return Err(anyhow::anyhow!("docserver stopped without a startup error"));
				}
				match UnixStream::connect(&server.socket).await {
					Ok(stream) => {
						drop(stream);
						return Ok(());
					},
					Err(error) if error.kind() == io::ErrorKind::NotFound
						|| error.kind() == io::ErrorKind::ConnectionRefused =>
					{
						tokio::time::sleep(Duration::from_millis(10)).await;
					},
					Err(error) => return Err(error).context("connecting to docserver socket"),
				}
			}
		}).await??;
		Ok(server)
	}

	/// Returns the owner-local document endpoint.
	#[must_use]
	pub fn socket(&self) -> &Path {
		&self.socket
	}

	/// Opens a typed, hello-complete framed client connection.
	pub async fn connect(&self) -> Result<DocumentHost> {
		let stream = within(
			"docserver connection",
			DEFAULT_TIMEOUT,
			UnixStream::connect(&self.socket),
		).await??;
		within("document hello", DEFAULT_TIMEOUT, DocumentHost::connect(stream)).await?
			.context("document hello failed")
	}

	/// Stops the task and removes its socket before returning.
	pub async fn shutdown(mut self) -> Result<()> {
		self.task.abort();
		match (&mut self.task).await {
			Ok(result) => result.context("docserver shutdown")?,
			Err(error) if error.is_cancelled() => {},
			Err(error) => return Err(error).context("joining docserver task"),
		}
		remove_socket(&self.socket)?;
		Ok(())
	}
}

impl Drop for DocServerTask {
	fn drop(&mut self) {
		self.task.abort();
		let _ = remove_socket(&self.socket);
	}
}

fn remove_socket(path: &Path) -> io::Result<()> {
	match std::fs::remove_file(path) {
		Ok(()) => Ok(()),
		Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
		Err(error) => Err(error),
	}
}
