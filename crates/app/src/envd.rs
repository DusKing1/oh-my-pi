//! Project environment-daemon assembly and production serving.

pub mod blobs;
pub mod docs;
pub(crate) mod eval;
pub mod exec;
pub mod server;
mod tool_document;
mod tool_read_sources;
mod tool_search;
mod tool_shell;
mod tools;
pub mod worker;
pub mod workspace;
use std::{io, path::Path, sync::Arc};

#[doc(hidden)]
pub use eval::{EVAL_CHILD_ARG, ProcessError as EvalChildError, run_eval_child_entry};
use miette::IntoDiagnostic as _;
use omp_core::Str;
use omp_env::EnvClient;
use omp_proto::env::v1::{ClientHello, ServerHello};
use omp_tool::Registry;
pub use server::EnvdError;
use tokio_util::sync::CancellationToken;

use self::{
	server::EnvServer,
	worker::{PY_EVAL_MODULE, ToolWorkerConfig},
};
use crate::cli::EnvdArgs;

/// Starts the project environment daemon and serves until process shutdown.
pub async fn run(args: EnvdArgs) -> miette::Result<()> {
	server::run(args).await.into_diagnostic()
}

/// Client-side ownership of one project environment composition.
///
/// Dropping this value shuts down only servers and children started by this
/// composition. An existing owner environment remains untouched.
pub(crate) struct ProjectEnvironment {
	pub(crate) client:   EnvClient,
	pub(crate) registry: Arc<Registry>,
	eval_bridge:         Arc<eval::SessionBridgeHost>,
	eval_control:        omp_tools::eval::EvalSessionControl,
	_lifecycle:          ProjectLifecycle,
}

struct ProjectLifecycle {
	shutdown: Option<CancellationToken>,
	tasks:    Vec<tokio::task::JoinHandle<()>>,
	_server:  Arc<EnvServer>,
}

impl Drop for ProjectLifecycle {
	fn drop(&mut self) {
		if let Some(shutdown) = &self.shutdown {
			shutdown.cancel();
		}
		for task in &self.tasks {
			task.abort();
		}
	}
}

impl ProjectEnvironment {
	/// Connects an existing owner environment or starts one for this process.
	pub(crate) async fn connect_or_start(
		root: &Path,
		state_dir: &Path,
		socket: &Path,
		docserver_socket: &Path,
		py_eval: bool,
	) -> Result<Self, EnvdError> {
		match EnvServer::connect_owner_uds(socket).await {
			Ok((owner_probe, bridge)) => {
				match hello(&owner_probe).await {
					Ok(owner_hello)
						if crate::build_id::is_stale(
							crate::build_id::current(),
							&owner_hello.server_build,
						) =>
					{
						// Stale-build owners can only appear on explicitly
						// configured socket paths; the automatic path is keyed
						// by build identity. Ask the owner to retire, then wait
						// briefly for the endpoint to be released.
						let _ = owner_probe.retire().await;
						bridge.abort();
						let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
						loop {
							match tokio::net::UnixStream::connect(socket).await {
								Err(error)
									if matches!(
										error.kind(),
										io::ErrorKind::NotFound | io::ErrorKind::ConnectionRefused
									) =>
								{
									return Self::start(root, state_dir, socket, docserver_socket, py_eval)
										.await;
								},
								_ if tokio::time::Instant::now() >= deadline => break,
								_ => tokio::time::sleep(std::time::Duration::from_millis(50)).await,
							}
						}
						tracing::warn!(
							socket = %socket.display(),
							"stale-build environment daemon kept its socket; using an in-process environment"
						);
					},
					Ok(_) => bridge.abort(),
					Err(EnvdError::Client(omp_env::ClientError::Protocol(error))) => {
						// Owners from before the current schema revision reject
						// the hello outright; their endpoint drains with its
						// owner while this process stays in-process.
						bridge.abort();
						tracing::warn!(
							socket = %socket.display(),
							code = error.code,
							message = %error.message,
							"environment owner rejected the handshake; using an in-process environment"
						);
					},
					Err(error) => return Err(error),
				}
				Self::connect_peer(root, state_dir, docserver_socket, py_eval).await
			},
			Err(EnvdError::Io(error))
				if matches!(
					error.kind(),
					io::ErrorKind::NotFound | io::ErrorKind::ConnectionRefused
				) =>
			{
				// No owner: autostart a detached project daemon so the shared
				// authorities outlive this process, then join it as a peer.
				match spawn_project_daemon(root, state_dir, socket, docserver_socket).await {
					Ok(()) => Self::connect_peer(root, state_dir, docserver_socket, py_eval).await,
					Err(error) => {
						tracing::warn!(
							socket = %socket.display(),
							%error,
							"could not autostart the project daemon; running an embedded environment"
						);
						Self::start(root, state_dir, socket, docserver_socket, py_eval).await
					},
				}
			},
			Err(error) => Err(error),
		}
	}

	/// Joins the project as a peer of an already-running owner environment.
	///
	/// The composition serves tools in-process and holds only client
	/// connections to shared authorities, so dropping it never affects other
	/// connected apps.
	async fn connect_peer(
		root: &Path,
		state_dir: &Path,
		docserver_socket: &Path,
		py_eval: bool,
	) -> Result<Self, EnvdError> {
		let worker_config = worker_config(py_eval)?;
		let server = EnvServer::open_project(
			root,
			state_dir,
			docserver_socket,
			Registry::new(),
			worker_config,
			None,
		)
		.await?;
		let server = Arc::new(server);
		let registry = server.registry();
		let eval_bridge = server.eval_bridge();
		let eval_control = server.eval_control();
		let (client, transport) = EnvClient::in_process(64);
		let in_process_server = Arc::clone(&server);
		let in_process = tokio::spawn(async move {
			in_process_server.serve_in_process(transport).await;
		});
		hello(&client).await?;
		let lifecycle =
			ProjectLifecycle { shutdown: None, tasks: vec![in_process], _server: server };
		Ok(Self { client, registry, eval_bridge, eval_control, _lifecycle: lifecycle })
	}

	async fn start(
		root: &Path,
		state_dir: &Path,
		socket: &Path,
		docserver_socket: &Path,
		py_eval: bool,
	) -> Result<Self, EnvdError> {
		let worker_config = worker_config(py_eval)?;
		let server = EnvServer::open_project(
			root,
			state_dir,
			docserver_socket,
			Registry::new(),
			worker_config,
			None,
		)
		.await?;
		let server = Arc::new(server);
		let registry = server.registry();
		let eval_bridge = server.eval_bridge();
		let eval_control = server.eval_control();
		let (client, transport) = EnvClient::in_process(64);
		let in_process_server = Arc::clone(&server);
		let in_process = tokio::spawn(async move {
			in_process_server.serve_in_process(transport).await;
		});
		let shutdown = CancellationToken::new();
		let uds_server = Arc::clone(&server);
		let uds_shutdown = shutdown.clone();
		let socket = socket.to_path_buf();
		let uds = tokio::spawn(async move {
			if let Err(error) = uds_server.serve_uds(&socket, uds_shutdown, None).await {
				// A lost same-build bind race is benign: the winner serves the
				// endpoint while this composition stays fully in-process.
				tracing::debug!(
					socket = %socket.display(),
					%error,
					"environment socket is served by another process"
				);
			}
		});
		let lifecycle = ProjectLifecycle {
			shutdown: Some(shutdown),
			tasks:    vec![in_process, uds],
			_server:  server,
		};
		hello(&client).await?;
		Ok(Self { client, registry, eval_bridge, eval_control, _lifecycle: lifecycle })
	}

	#[must_use]
	pub(crate) const fn client(&self) -> &EnvClient {
		&self.client
	}

	#[must_use]
	pub(crate) fn registry(&self) -> Arc<Registry> {
		Arc::clone(&self.registry)
	}

	#[must_use]
	pub(crate) fn eval_bridge(&self) -> Arc<eval::SessionBridgeHost> {
		Arc::clone(&self.eval_bridge)
	}

	#[must_use]
	pub(crate) fn eval_control(&self) -> omp_tools::eval::EvalSessionControl {
		self.eval_control.clone()
	}
}

fn worker_config(py_eval: bool) -> Result<ToolWorkerConfig, EnvdError> {
	let mut config = ToolWorkerConfig::current()?;
	if py_eval {
		config.modules.push(Str::new_static(PY_EVAL_MODULE));
	}
	Ok(config)
}

async fn hello(client: &EnvClient) -> Result<ServerHello, EnvdError> {
	Ok(client
		.hello(ClientHello {
			client: "omp-chat".into(),
			schema_rev: omp_proto::SCHEMA_REV,
			..ClientHello::default()
		})
		.await?)
}

/// Launches a detached `omp envd` for this project and waits until its
/// environment socket answers a hello.
async fn spawn_project_daemon(
	root: &Path,
	state_dir: &Path,
	socket: &Path,
	docserver_socket: &Path,
) -> Result<(), EnvdError> {
	let executable = std::env::current_exe()?;
	spawn_project_daemon_with(
		&executable,
		root,
		state_dir,
		socket,
		docserver_socket,
		std::time::Duration::from_secs(10),
	)
	.await
}

/// Spawns `executable envd …` detached from this process and waits for
/// readiness on `socket` within `deadline`.
///
/// The daemon runs in its own process group with output appended to
/// `envd.log` in the state directory. A daemon that fails to become ready is
/// killed so it cannot linger half-initialized while the caller falls back
/// to an embedded environment.
async fn spawn_project_daemon_with(
	executable: &Path,
	root: &Path,
	state_dir: &Path,
	socket: &Path,
	docserver_socket: &Path,
	deadline: std::time::Duration,
) -> Result<(), EnvdError> {
	std::fs::create_dir_all(state_dir)?;
	let log = std::fs::OpenOptions::new()
		.create(true)
		.append(true)
		.open(state_dir.join("envd.log"))?;
	let errors = log.try_clone()?;
	let mut command = tokio::process::Command::new(executable);
	command
		.arg("envd")
		.arg("--root")
		.arg(root)
		.arg("--state-dir")
		.arg(state_dir)
		.arg("--socket")
		.arg(socket)
		.arg("--docserver-socket")
		.arg(docserver_socket)
		.stdin(std::process::Stdio::null())
		.stdout(log)
		.stderr(errors)
		.kill_on_drop(false);
	{
		use std::os::unix::process::CommandExt as _;
		command.as_std_mut().process_group(0);
	}
	let mut child = command.spawn()?;
	let deadline = tokio::time::Instant::now() + deadline;
	loop {
		if let Some(status) = child.try_wait()? {
			return Err(
				io::Error::other(format!("project daemon exited during startup: {status}")).into(),
			);
		}
		if let Ok((probe, bridge)) = EnvServer::connect_owner_uds(socket).await {
			let ready = hello(&probe).await;
			bridge.abort();
			if ready.is_ok() {
				// Reap in the background; the daemon's lifetime is its own.
				tokio::spawn(async move {
					let _ = child.wait().await;
				});
				return Ok(());
			}
		}
		if tokio::time::Instant::now() >= deadline {
			let _ = child.start_kill();
			tokio::spawn(async move {
				let _ = child.wait().await;
			});
			return Err(
				io::Error::new(io::ErrorKind::TimedOut, "project daemon did not become ready").into(),
			);
		}
		tokio::time::sleep(std::time::Duration::from_millis(50)).await;
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	async fn spawn_with(executable: &Path, deadline_ms: u64) -> Result<(), EnvdError> {
		let scratch = tempfile::tempdir().expect("scratch state directory");
		spawn_project_daemon_with(
			executable,
			scratch.path(),
			scratch.path(),
			&scratch.path().join("env.sock"),
			&scratch.path().join("doc.sock"),
			std::time::Duration::from_millis(deadline_ms),
		)
		.await
	}

	#[tokio::test]
	async fn spawn_reports_missing_daemon_executable() {
		let error = spawn_with(Path::new("/nonexistent/omp"), 1_000)
			.await
			.expect_err("missing executable must fail");
		assert!(matches!(error, EnvdError::Io(_)));
	}

	#[tokio::test]
	async fn spawn_reports_a_daemon_that_exits_during_startup() {
		let error = spawn_with(Path::new("/usr/bin/true"), 5_000)
			.await
			.expect_err("exiting daemon must fail");
		assert!(error.to_string().contains("exited during startup"), "unexpected error: {error}");
	}

	#[tokio::test]
	async fn spawn_kills_a_daemon_that_never_becomes_ready() {
		use std::os::unix::fs::PermissionsExt as _;

		let scratch = tempfile::tempdir().expect("scratch script directory");
		let script = scratch.path().join("hang.sh");
		std::fs::write(&script, "#!/bin/sh\nexec sleep 30\n").expect("write hang script");
		std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o700))
			.expect("mark script executable");

		let error = spawn_with(&script, 300)
			.await
			.expect_err("unready daemon must time out");
		let EnvdError::Io(error) = &error else {
			panic!("unexpected error: {error}");
		};
		assert_eq!(error.kind(), io::ErrorKind::TimedOut);
	}
}
