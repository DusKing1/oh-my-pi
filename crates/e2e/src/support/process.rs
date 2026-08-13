use std::{io, path::PathBuf, process::Stdio, time::Duration};

use anyhow::{Context as _, Result};
use tokio::process::{Child, Command};

use super::within;

/// Resolves the worker-capable application binary Cargo builds with `omp-e2e` tests.
pub fn omp_binary() -> io::Result<PathBuf> {
	if let Some(path) = std::env::var_os("CARGO_BIN_EXE_omp_e2e_host") {
		let path = PathBuf::from(path);
		if path.is_file() {
			return Ok(path);
		}
	}
	let current = std::env::current_exe()?;
	if current.file_stem().is_some_and(|name| name == "omp_e2e_host") {
		return Ok(current);
	}
	let profile = current
		.parent()
		.and_then(|parent| (parent.file_name().is_some_and(|name| name == "deps")).then(|| parent.parent()))
		.flatten()
		.ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "test executable is not under Cargo's deps directory"))?;
	let binary = profile.join(format!("omp_e2e_host{}", std::env::consts::EXE_SUFFIX));
	if binary.is_file() {
		Ok(binary)
	} else {
		Err(io::Error::new(
			io::ErrorKind::NotFound,
			format!("Cargo-built omp_e2e_host is missing at {}", binary.display()),
		))
	}
}

/// Child process placed in its own process group and killed as a tree on drop.
#[derive(Debug)]
pub struct OwnedProcess {
	child: Child,
	group: Option<i32>,
	exited: bool,
}

impl OwnedProcess {
	/// Spawns one directly addressed executable without a shell.
	pub fn spawn(mut command: Command) -> io::Result<Self> {
		command.stdin(Stdio::null()).kill_on_drop(true);
		#[cfg(unix)]
		{
			use std::os::unix::process::CommandExt as _;
			command.as_std_mut().process_group(0);
		}
		let child = command.spawn()?;
		let group = child.id().and_then(|pid| i32::try_from(pid).ok());
		Ok(Self { child, group, exited: false })
	}

	/// Returns the operating-system child identifier while it is known.
	#[must_use]
	pub fn id(&self) -> Option<u32> {
		self.child.id()
	}

	/// Returns the dedicated Unix process-group identifier.
	#[must_use]
	pub const fn process_group(&self) -> Option<i32> {
		self.group
	}

	/// Waits for normal process exit within `limit`.
	pub async fn wait(&mut self, limit: Duration) -> Result<std::process::ExitStatus> {
		let status = within("owned child exit", limit, self.child.wait()).await??;
		self.exited = true;
		Ok(status)
	}

	/// Requests TERM, then escalates to KILL after `grace`, always targeting the tree.
	pub async fn terminate(mut self, grace: Duration) -> Result<()> {
		if self.exited {
			return Ok(());
		}
		self.signal_group_terminate();
		if tokio::time::timeout(grace, self.child.wait()).await.is_err() {
			self.signal_group_kill();
			self.child.wait().await.context("waiting for killed child")?;
		}
		self.exited = true;
		Ok(())
	}

	fn signal_group_terminate(&mut self) {
		#[cfg(unix)]
		if let Some(group) = self.group {
			let _ = nix::sys::signal::killpg(
				nix::unistd::Pid::from_raw(group),
				Some(nix::sys::signal::Signal::SIGTERM),
			);
			return;
		}
		let _ = self.child.start_kill();
	}

	fn signal_group_kill(&mut self) {
		#[cfg(unix)]
		if let Some(group) = self.group {
			let _ = nix::sys::signal::killpg(
				nix::unistd::Pid::from_raw(group),
				Some(nix::sys::signal::Signal::SIGKILL),
			);
			return;
		}
		let _ = self.child.start_kill();
	}
}

impl Drop for OwnedProcess {
	fn drop(&mut self) {
		if !self.exited {
			self.signal_group_kill();
		}
	}
}

/// Reports whether any process remains in a Unix process group.
#[cfg(unix)]
#[must_use]
pub fn process_group_alive(group: i32) -> bool {
	match nix::sys::signal::killpg(nix::unistd::Pid::from_raw(group), None) {
		Ok(()) | Err(nix::errno::Errno::EPERM) => true,
		Err(nix::errno::Errno::ESRCH) => false,
		Err(_) => true,
	}
}

/// Waits until a Unix process group disappears, with deterministic polling and a hard bound.
#[cfg(unix)]
pub async fn wait_process_group_dead(group: i32, limit: Duration) -> Result<()> {
	within("process-group death", limit, async move {
		while process_group_alive(group) {
			tokio::time::sleep(Duration::from_millis(10)).await;
		}
	}).await
}
