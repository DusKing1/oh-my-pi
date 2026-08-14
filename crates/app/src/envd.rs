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
use omp_proto::env::v1::ClientHello;
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
				hello(&owner_probe).await?;
				bridge.abort();
				let worker_config = worker_config(py_eval)?;
				let server = EnvServer::open_project(
					root,
					state_dir,
					docserver_socket,
					Registry::new(),
					worker_config,
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
			},
			Err(EnvdError::Io(error))
				if matches!(
					error.kind(),
					io::ErrorKind::NotFound | io::ErrorKind::ConnectionRefused
				) =>
			{
				Self::start(root, state_dir, socket, docserver_socket, py_eval).await
			},
			Err(error) => Err(error),
		}
	}

	async fn start(
		root: &Path,
		state_dir: &Path,
		socket: &Path,
		docserver_socket: &Path,
		py_eval: bool,
	) -> Result<Self, EnvdError> {
		let worker_config = worker_config(py_eval)?;
		let server =
			EnvServer::open_project(root, state_dir, docserver_socket, Registry::new(), worker_config)
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
			let _ = uds_server.serve_uds(&socket, uds_shutdown).await;
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

async fn hello(client: &EnvClient) -> Result<(), EnvdError> {
	client
		.hello(ClientHello {
			client: "omp-chat".into(),
			schema_rev: omp_proto::SCHEMA_REV,
			..ClientHello::default()
		})
		.await?;
	Ok(())
}
