//! Project environment-daemon assembly and production serving.

pub mod blobs;
pub mod docs;
pub mod exec;
pub mod server;
pub mod worker;
pub mod workspace;
mod tool_document;
mod tool_search;
mod tool_shell;
mod tools;
pub use server::EnvdError;

use std::{
	io,
	path::Path,
	sync::Arc,
};

use omp_core::Str;
use omp_env::EnvClient;
use omp_proto::env::v1::ClientHello;
use omp_tool::Registry;
use tokio_util::sync::CancellationToken;

use self::{
	server::EnvServer,
	worker::{PY_EVAL_MODULE, ToolWorkerConfig},
};

use crate::cli::EnvdArgs;

/// Starts the project environment daemon and serves until process shutdown.
pub async fn run(args: EnvdArgs) -> crate::Result<()> {
	Ok(server::run(args).await?)
}

/// Client-side ownership of one project environment composition.
///
/// Dropping this value shuts down only servers and children started by this
/// composition. An existing owner environment remains untouched.
pub(crate) struct ProjectEnvironment {
	pub(crate) client:   EnvClient,
	pub(crate) registry: Arc<Registry>,
	_lifecycle:          ProjectLifecycle,
}

struct ProjectLifecycle {
	shutdown:      Option<CancellationToken>,
	_tasks:        Vec<tokio::task::JoinHandle<()>>,
	remote_bridge: Option<tokio::task::JoinHandle<Result<(), EnvdError>>>,
	_server:       Arc<EnvServer>,
	_docserver:    Option<tokio::process::Child>,
}

impl Drop for ProjectLifecycle {
	fn drop(&mut self) {
		if let Some(shutdown) = &self.shutdown {
			shutdown.cancel();
		} else if let Some(bridge) = &self.remote_bridge {
			bridge.abort();
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
			Ok((client, bridge)) => {
				hello(&client).await?;
				let worker_config = worker_config(py_eval)?;
				let (server, docserver) = EnvServer::open_project(
					root,
					state_dir,
					docserver_socket,
					Registry::new(),
					worker_config,
				)
				.await?;
				let server = Arc::new(server);
				let registry = server.registry();
				let lifecycle = ProjectLifecycle {
					shutdown: None,
					_tasks: Vec::new(),
					remote_bridge: Some(bridge),
					_server: server,
					_docserver: docserver,
				};
				Ok(Self {
					client,
					registry,
					_lifecycle: lifecycle,
				})
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
		let (server, docserver) = EnvServer::open_project(
			root,
			state_dir,
			docserver_socket,
			Registry::new(),
			worker_config,
		)
		.await?;
		let server = Arc::new(server);
		let registry = server.registry();
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
			_tasks: vec![in_process, uds],
			remote_bridge: None,
			_server: server,
			_docserver: docserver,
		};
		hello(&client).await?;
		Ok(Self { client, registry, _lifecycle: lifecycle })
	}

	#[must_use]
	pub(crate) fn client(&self) -> &EnvClient {
		&self.client
	}

	#[must_use]
	pub(crate) fn registry(&self) -> Arc<Registry> {
		Arc::clone(&self.registry)
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
