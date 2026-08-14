//! Multi-client document authority over a local Unix socket or standard I/O.
//!
//! Embedding daemons can observe the socket connection gauge to drive idle
//! detection without inspecting document protocol traffic.

#[cfg(unix)]
use std::path::Path;
#[cfg(unix)]
use std::{
	fs::{File, OpenOptions},
	os::unix::fs::{FileTypeExt, MetadataExt, PermissionsExt},
};
use std::{path::PathBuf, time::Duration};

use omp_core::Str;
#[cfg(unix)]
use omp_core::hex;
#[cfg(unix)]
use rustix::fs::{FlockOperation, flock};
#[cfg(unix)]
use tokio::net::{UnixListener, UnixStream};
#[cfg(unix)]
use tokio::task::JoinSet;
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;

use crate::{
	Environment, LspProcess, LspProcessError, ServerConfig,
	connection::{ConnectionConfig, ConnectionError, serve_io_until},
	load_lsp_process_configs,
};

const SHUTDOWN_GRACE: Duration = Duration::from_secs(5);
#[cfg(unix)]
const MAX_SOCKET_CONNECTIONS: usize = 128;
#[cfg(unix)]
const ACCEPT_RETRY_DELAY: Duration = Duration::from_millis(100);

/// The transport on which the document authority accepts connections.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Transport {
	/// Serve one framed connection over standard input and standard output.
	Stdio,
	/// Serve concurrent framed connections over the Unix-domain socket at this
	/// path.
	Socket(PathBuf),
}

/// Options for serving the document authority.
pub struct ServeOptions {
	/// Language-server process configuration files loaded before serving.
	pub lsp_config_paths: Vec<PathBuf>,
	/// External shutdown; `None` installs signal handling.
	pub shutdown:         Option<CancellationToken>,
	/// Build identity advertised in `ServerHello`; empty means unknown.
	pub server_build:     Str,
	/// Socket-connection gauge receiving the live accepted-connection count.
	pub connections:      Option<tokio::sync::watch::Sender<usize>>,
}

/// An error that prevents the document authority from starting or serving its
/// transport.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
	/// The document Environment could not be configured.
	#[error(transparent)]
	Document(#[from] crate::Error),
	/// A standard-I/O connection failed.
	#[error(transparent)]
	Connection(#[from] ConnectionError),
	/// An operating-system operation failed.
	#[error(transparent)]
	Io(#[from] std::io::Error),
	/// A configured language-server process failed to start or stop.
	#[error(transparent)]
	LspProcess(#[from] LspProcessError),
	/// Document actors did not stop within the shutdown deadline.
	#[error("document actors did not stop within the shutdown deadline")]
	ShutdownDeadlineExceeded,

	/// Cannot open an authority lock file.
	#[error("cannot open authority lock {path:?}: {source}")]
	OpenAuthorityLock {
		/// Path to the lock file.
		path:   PathBuf,
		/// Underlying I/O error.
		#[source]
		source: std::io::Error,
	},

	/// Cannot set permissions on an authority lock file.
	#[error("cannot secure authority lock {path:?}: {source}")]
	SecureAuthorityLock {
		/// Path to the lock file.
		path:   PathBuf,
		/// Underlying I/O error.
		#[source]
		source: std::io::Error,
	},

	/// An authority lock is unavailable because another instance is running.
	#[cfg(unix)]
	#[error("another {kind} authority is already running or lock {path:?} is unavailable: {source}")]
	AcquireAuthorityLock {
		/// Kind of authority lock (e.g., "socket").
		kind:   &'static str,
		/// Path to the lock file.
		path:   PathBuf,
		/// Underlying file locking error.
		#[source]
		source: rustix::io::Errno,
	},

	/// Cannot set permissions on the lock directory.
	#[error("cannot secure lock directory {directory:?}: {source}")]
	SecureLockDirectory {
		/// Path to the lock directory.
		directory: PathBuf,
		/// Underlying I/O error.
		#[source]
		source:    std::io::Error,
	},

	/// Cannot create the lock directory.
	#[error("cannot create lock directory {directory:?}: {source}")]
	CreateLockDirectory {
		/// Path to the lock directory.
		directory: PathBuf,
		/// Underlying I/O error.
		#[source]
		source:    std::io::Error,
	},

	/// Cannot stat or inspect the lock directory.
	#[error("cannot inspect lock directory {directory:?}: {source}")]
	InspectLockDirectory {
		/// Path to the lock directory.
		directory: PathBuf,
		/// Underlying I/O error.
		#[source]
		source:    std::io::Error,
	},

	/// The lock directory has invalid ownership or permissions.
	#[error("lock directory {directory:?} must be an owner-only directory owned by uid {user_id}")]
	InvalidLockDirectoryPermissions {
		/// Path to the lock directory.
		directory: PathBuf,
		/// Effective user ID.
		user_id:   u32,
	},

	/// The requested Unix socket path lacks a file name component.
	#[error("socket path {path:?} has no file name")]
	SocketPathMissingFileName {
		/// Path to the socket.
		path: PathBuf,
	},

	/// Refusing to replace an existing non-socket file at the socket path.
	#[error("refusing to replace non-socket path {path:?}")]
	ReplaceNonSocketPath {
		/// Path to the existing non-socket entry.
		path: PathBuf,
	},

	/// Another docserver daemon is actively listening on the socket.
	#[error("another omp-docserver is listening on {path:?}")]
	SocketInUse {
		/// Active socket path.
		path: PathBuf,
	},

	/// Failed to probe whether an existing socket is active.
	#[error("cannot determine whether socket {path:?} is active: {source}")]
	ProbeActiveSocket {
		/// Socket path being probed.
		path:   PathBuf,
		/// Underlying I/O error.
		#[source]
		source: std::io::Error,
	},

	/// Environment root URI cannot be converted to a local file path.
	#[error("Environment root is not a local file URI")]
	NonFileUriRoot,

	/// The socket path is inside the writable Environment root.
	#[error("socket path {path:?} must be outside the writable Environment root {root:?}")]
	SocketInsideEnvironmentRoot {
		/// Requested socket path.
		path: PathBuf,
		/// Environment root path.
		root: PathBuf,
	},
	/// Unix-domain sockets were requested on a platform that does not support
	/// them.
	#[error("Unix-domain sockets are unsupported on this platform; use standard I/O")]
	UnsupportedSocket,
}

/// The result of a document-authority operation.
pub type Result<T = ()> = std::result::Result<T, Error>;

/// Serves the document authority rooted at `root` on `transport`.
///
/// Every LSP configuration path is parsed before authority is acquired. All
/// declared processes complete initialization and registry installation before
/// the client transport starts accepting requests. On Unix, socket endpoints
/// are protected by a separate instance lock and created sockets are
/// owner-only. When no external shutdown token is supplied, `SIGINT` or
/// `SIGTERM` starts graceful connection, LSP, and actor shutdown.
pub async fn serve(root: PathBuf, transport: Transport, options: ServeOptions) -> Result {
	run_with_shutdown(root, transport, options).await
}

async fn run_with_shutdown(root: PathBuf, transport: Transport, options: ServeOptions) -> Result {
	let process_configs = load_lsp_process_configs(&options.lsp_config_paths)?;
	let config = ServerConfig::new(root)?.with_server_build(options.server_build);
	let authority_lock = config.try_lock_authority()?;
	let environment = Environment::new(config)?;
	let mut processes = Vec::with_capacity(process_configs.len());
	for process_config in process_configs {
		match LspProcess::start(process_config, &environment, CancellationToken::new()).await {
			Ok(process) => processes.push(process),
			Err(error) => {
				let _ = stop_lsp_processes(&mut processes).await;
				let _ = timeout(SHUTDOWN_GRACE, environment.shutdown()).await;
				return Err(error.into());
			},
		}
	}
	let serve_result = match (transport, options.shutdown) {
		(Transport::Stdio, None) => serve_stdio(environment.clone()).await,
		(Transport::Stdio, Some(shutdown)) => serve_stdio_until(environment.clone(), shutdown).await,
		(Transport::Socket(path), None) => {
			serve_socket(environment.clone(), path, options.connections).await
		},
		(Transport::Socket(path), Some(shutdown)) => {
			serve_socket_until(environment.clone(), path, shutdown, options.connections).await
		},
	};
	let process_result = stop_lsp_processes(&mut processes).await;
	if timeout(SHUTDOWN_GRACE, environment.shutdown())
		.await
		.is_err()
	{
		// Keep the directory handle locked until process exit: returning a
		// reusable authority while an actor may still persist would permit a
		// split brain.
		std::mem::forget(authority_lock);
		return Err(Error::ShutdownDeadlineExceeded);
	}
	serve_result?;
	process_result
}

async fn stop_lsp_processes(processes: &mut Vec<LspProcess>) -> Result {
	let mut first_error = None;
	while let Some(process) = processes.pop() {
		if let Err(error) = process.shutdown().await
			&& first_error.is_none()
		{
			first_error = Some(Error::LspProcess(error));
		}
	}
	first_error.map_or(Ok(()), Err)
}

#[cfg(unix)]
struct InstanceLock {
	_file: File,
}

#[cfg(unix)]
impl InstanceLock {
	fn acquire(kind: &'static str, identity: &Path) -> Result<Self> {
		let directory = lock_directory()?;
		let digest = blake3::hash(identity.as_os_str().as_encoded_bytes());
		let encoded = hex::encode_n(digest.as_bytes());
		let path = directory.join(format!("{kind}-{encoded}.lock"));
		let file = OpenOptions::new()
			.read(true)
			.write(true)
			.create(true)
			.truncate(false)
			.open(&path)
			.map_err(|source| Error::OpenAuthorityLock { path: path.clone(), source })?;
		std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))
			.map_err(|source| Error::SecureAuthorityLock { path: path.clone(), source })?;
		flock(&file, FlockOperation::NonBlockingLockExclusive)
			.map_err(|source| Error::AcquireAuthorityLock { kind, path: path.clone(), source })?;
		Ok(Self { _file: file })
	}
}

#[cfg(unix)]
fn lock_directory() -> Result<PathBuf> {
	let user_id = rustix::process::geteuid().as_raw();
	let directory = match std::env::var_os("XDG_RUNTIME_DIR").map(PathBuf::from) {
		Some(runtime) if runtime.is_absolute() => runtime.join("omp-docserver"),
		_ => std::env::temp_dir().join(format!("omp-docserver-{user_id}")),
	};
	match std::fs::create_dir(&directory) {
		Ok(()) => std::fs::set_permissions(&directory, std::fs::Permissions::from_mode(0o700))
			.map_err(|source| Error::SecureLockDirectory { directory: directory.clone(), source })?,
		Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {},
		Err(source) => {
			return Err(Error::CreateLockDirectory { directory, source });
		},
	}
	let metadata = std::fs::symlink_metadata(&directory)
		.map_err(|source| Error::InspectLockDirectory { directory: directory.clone(), source })?;
	if !metadata.is_dir() || metadata.uid() != user_id || metadata.mode() & 0o077 != 0 {
		return Err(Error::InvalidLockDirectoryPermissions { directory, user_id });
	}
	Ok(directory)
}

async fn serve_stdio(environment: Environment) -> Result {
	let shutdown = CancellationToken::new();
	let signal_shutdown = shutdown.clone();
	let signal = tokio::spawn(async move {
		let _ = shutdown_signal().await;
		signal_shutdown.cancel();
	});
	let result = serve_stdio_until(environment, shutdown).await;
	signal.abort();
	result
}

async fn serve_stdio_until(environment: Environment, shutdown: CancellationToken) -> Result {
	serve_io_until(
		environment.session(),
		tokio::io::stdin(),
		tokio::io::stdout(),
		ConnectionConfig::default(),
		shutdown,
	)
	.await
	.map_err(Into::into)
}

#[cfg(unix)]
async fn bind_socket(path: PathBuf) -> Result<(UnixListener, SocketCleanup)> {
	let name = path
		.file_name()
		.map(std::ffi::OsStr::to_owned)
		.ok_or_else(|| Error::SocketPathMissingFileName { path: path.clone() })?;
	let parent = path
		.parent()
		.filter(|parent| !parent.as_os_str().is_empty())
		.unwrap_or_else(|| Path::new("."));
	let canonical_parent = std::fs::canonicalize(parent)?;
	let path = canonical_parent.join(name);
	let identity = path.clone();
	let lock = InstanceLock::acquire("socket", &identity)?;
	match std::fs::symlink_metadata(&path) {
		Ok(metadata) if !metadata.file_type().is_socket() => {
			return Err(Error::ReplaceNonSocketPath { path });
		},
		Ok(_) => match UnixStream::connect(&path).await {
			Ok(_) => {
				return Err(Error::SocketInUse { path });
			},
			Err(error)
				if matches!(
					error.kind(),
					std::io::ErrorKind::ConnectionRefused | std::io::ErrorKind::NotFound
				) =>
			{
				std::fs::remove_file(&path)?;
			},
			Err(error) => {
				return Err(Error::ProbeActiveSocket { path, source: error });
			},
		},
		Err(error) if error.kind() == std::io::ErrorKind::NotFound => {},
		Err(error) => return Err(error.into()),
	}
	let listener = UnixListener::bind(&path)?;
	std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
	let metadata = std::fs::symlink_metadata(&path)?;
	let cleanup = SocketCleanup { path, dev: metadata.dev(), ino: metadata.ino(), _lock: lock };
	Ok((listener, cleanup))
}

#[cfg(unix)]
fn validate_socket_location(path: &Path, root: &Path) -> Result {
	let parent = path
		.parent()
		.filter(|parent| !parent.as_os_str().is_empty())
		.unwrap_or_else(|| Path::new("."));
	let canonical_parent = std::fs::canonicalize(parent)?;
	if canonical_parent.starts_with(root) {
		Err(Error::SocketInsideEnvironmentRoot { path: path.to_owned(), root: root.to_owned() })
	} else {
		Ok(())
	}
}

#[cfg(unix)]
async fn serve_socket(
	environment: Environment,
	path: PathBuf,
	connections: Option<tokio::sync::watch::Sender<usize>>,
) -> Result {
	let shutdown = CancellationToken::new();
	let signal_shutdown = shutdown.clone();
	let signal = tokio::spawn(async move {
		let _ = shutdown_signal().await;
		signal_shutdown.cancel();
	});
	let result = serve_socket_until(environment, path, shutdown, connections).await;
	signal.abort();
	result
}

#[cfg(unix)]
async fn serve_socket_until(
	environment: Environment,
	path: PathBuf,
	shutdown: CancellationToken,
	connection_gauge: Option<tokio::sync::watch::Sender<usize>>,
) -> Result {
	let root = environment
		.root_uri()
		.to_file_path()
		.map_err(|()| Error::NonFileUriRoot)?;
	validate_socket_location(&path, &root)?;
	let (listener, socket) = bind_socket(path).await?;
	let mut connections = JoinSet::new();
	publish_connection_count(&connection_gauge, 0);

	loop {
		tokio::select! {
			() = shutdown.cancelled() => {
				break;
			},
			accepted = listener.accept(), if connections.len() < MAX_SOCKET_CONNECTIONS => {
				let (stream, _) = match accepted {
					Ok(connection) => connection,
					Err(error) => {
						eprintln!("omp-docserver: socket accept failed: {error}");
						tokio::time::sleep(ACCEPT_RETRY_DELAY).await;
						continue;
					},
				};
				match stream.peer_cred() {
					Ok(credentials) if credentials.uid() == rustix::process::geteuid().as_raw() => {},
					Ok(credentials) => {
						eprintln!(
							"omp-docserver: rejected socket peer owned by uid {}",
							credentials.uid()
						);
						continue;
					},
					Err(error) => {
						eprintln!("omp-docserver: cannot authenticate socket peer: {error}");
						continue;
					},
				}
				let session = environment.session();
				let connection_shutdown = shutdown.clone();
				connections.spawn(async move {
					// `into_split` (one Arc per connection, at setup) is required:
					// `serve_io_until` moves the writer half into its own task.
					let (reader, writer) = stream.into_split();
					serve_io_until(
						session,
						reader,
						writer,
						ConnectionConfig::default(),
						connection_shutdown,
					)
					.await
				});
				publish_connection_count(&connection_gauge, connections.len());
			},
			completed = connections.join_next(), if !connections.is_empty() => {
				if let Some(completed) = completed {
					report_connection(completed);
					publish_connection_count(&connection_gauge, connections.len());
				}
			},
		}
	}

	drop(listener);
	let drain = async {
		while let Some(completed) = connections.join_next().await {
			report_connection(completed);
			publish_connection_count(&connection_gauge, connections.len());
		}
	};
	if timeout(SHUTDOWN_GRACE, drain).await.is_err() {
		connections.shutdown().await;
		publish_connection_count(&connection_gauge, 0);
	}
	drop(socket);
	Ok(())
}

#[cfg(not(unix))]
async fn serve_socket(
	_environment: Environment,
	_path: PathBuf,
	_connections: Option<tokio::sync::watch::Sender<usize>>,
) -> Result {
	Err(Error::UnsupportedSocket)
}

#[cfg(not(unix))]
async fn serve_socket_until(
	_environment: Environment,
	_path: PathBuf,
	_shutdown: CancellationToken,
	_connections: Option<tokio::sync::watch::Sender<usize>>,
) -> Result {
	Err(Error::UnsupportedSocket)
}

#[cfg(unix)]
fn publish_connection_count(gauge: &Option<tokio::sync::watch::Sender<usize>>, count: usize) {
	if let Some(gauge) = gauge {
		gauge.send_replace(count);
	}
}

fn report_connection(
	result: std::result::Result<std::result::Result<(), ConnectionError>, tokio::task::JoinError>,
) {
	match result {
		Ok(Ok(())) => {},
		Ok(Err(error)) => {
			eprintln!("omp-docserver: connection failed: {error}");
		},
		Err(error) if error.is_cancelled() => {},
		Err(error) => {
			eprintln!("omp-docserver: connection task failed: {error}");
		},
	}
}

async fn shutdown_signal() -> std::io::Result<()> {
	#[cfg(unix)]
	{
		let mut terminate =
			tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
		tokio::select! {
			result = tokio::signal::ctrl_c() => result,
			_ = terminate.recv() => Ok(()),
		}
	}
	#[cfg(not(unix))]
	{
		tokio::signal::ctrl_c().await
	}
}

#[cfg(unix)]
struct SocketCleanup {
	path:  PathBuf,
	dev:   u64,
	ino:   u64,
	_lock: InstanceLock,
}

#[cfg(unix)]
impl Drop for SocketCleanup {
	fn drop(&mut self) {
		let Ok(metadata) = std::fs::symlink_metadata(&self.path) else {
			return;
		};
		if metadata.file_type().is_socket()
			&& metadata.dev() == self.dev
			&& metadata.ino() == self.ino
		{
			let _ = std::fs::remove_file(&self.path);
		}
	}
}

#[cfg(all(test, unix))]
mod tests {
	use bytes::{Bytes, BytesMut};
	use omp_proto::document::v1 as proto;
	use tempfile::TempDir;
	use tokio::sync::watch;

	use super::*;
	use crate::{
		connection::{PROTOCOL_MAJOR, PROTOCOL_MINOR},
		wire::{FrameConfig, read_server_frame, write_client_frame},
	};

	#[test]
	fn authority_lock_is_exclusive_and_released_on_drop() {
		let root = TempDir::new().expect("temporary directory");
		let identity = root.path().join("workspace");
		let first = InstanceLock::acquire("test-workspace", &identity).expect("first lock");
		assert!(
			InstanceLock::acquire("test-workspace", &identity).is_err(),
			"a second authority must be rejected"
		);
		drop(first);
		InstanceLock::acquire("test-workspace", &identity).expect("released lock is reusable");
	}

	#[test]
	fn workspace_authority_lock_is_exclusive_and_released_on_drop() {
		let root = TempDir::new().expect("temporary directory");
		let first_config = ServerConfig::new(root.path()).expect("first server config");
		let second_config = ServerConfig::new(root.path()).expect("second server config");
		let first = first_config
			.try_lock_authority()
			.expect("first workspace authority");
		assert!(
			second_config.try_lock_authority().is_err(),
			"a second workspace authority must be rejected"
		);
		drop(first);
		second_config
			.try_lock_authority()
			.expect("released workspace authority is reusable");
	}

	#[tokio::test]
	async fn socket_binding_replaces_stale_socket_but_not_live_listener() {
		let root = TempDir::new().expect("temporary directory");
		std::fs::set_permissions(root.path(), std::fs::Permissions::from_mode(0o700))
			.expect("secure socket parent");
		let path = root.path().join("document.sock");
		let stale = UnixListener::bind(&path).expect("stale listener");
		drop(stale);

		let (listener, cleanup) = bind_socket(path.clone())
			.await
			.expect("replace stale socket");
		assert_eq!(
			std::fs::metadata(&path)
				.expect("socket metadata")
				.permissions()
				.mode() & 0o777,
			0o600
		);
		drop(listener);
		drop(cleanup);
		assert!(!path.exists());

		let live = UnixListener::bind(&path).expect("live listener");
		assert!(bind_socket(path.clone()).await.is_err(), "a live listener must never be displaced");
		drop(live);
		std::fs::remove_file(path).expect("remove live test socket");
	}

	#[tokio::test]
	async fn socket_cleanup_preserves_a_replacement_entry() {
		let root = TempDir::new().expect("temporary directory");
		std::fs::set_permissions(root.path(), std::fs::Permissions::from_mode(0o700))
			.expect("secure socket parent");
		let path = root.path().join("document.sock");
		let (listener, cleanup) = bind_socket(path.clone()).await.expect("bind socket");
		drop(listener);
		std::fs::remove_file(&path).expect("remove original socket");
		std::fs::write(&path, b"replacement").expect("write replacement");

		drop(cleanup);
		assert_eq!(std::fs::read(path).expect("replacement remains"), b"replacement");
	}

	#[tokio::test]
	async fn socket_binding_accepts_standard_parent_permissions() {
		let root = TempDir::new().expect("temporary directory");
		let shared = root.path().join("shared");
		std::fs::create_dir(&shared).expect("create shared directory");
		std::fs::set_permissions(&shared, std::fs::Permissions::from_mode(0o755))
			.expect("set standard parent permissions");
		let path = shared.join("document.sock");

		let (listener, cleanup) = bind_socket(path.clone()).await.expect("bind socket");
		assert_eq!(
			std::fs::metadata(&path)
				.expect("socket metadata")
				.permissions()
				.mode() & 0o777,
			0o600
		);
		drop(listener);
		drop(cleanup);
		assert!(!path.exists());
	}

	#[tokio::test]
	async fn serve_until_cancellation_removes_socket() {
		let scratch = TempDir::new().expect("temporary directory");
		let project = scratch.path().join("project");
		let runtime = scratch.path().join("runtime");
		std::fs::create_dir(&project).expect("project directory");
		std::fs::create_dir(&runtime).expect("runtime directory");
		let socket = runtime.join("document.sock");
		let shutdown = CancellationToken::new();
		let task_shutdown = shutdown.clone();
		let task_project = project.clone();
		let task_socket = socket.clone();
		let task = tokio::spawn(async move {
			serve(task_project, Transport::Socket(task_socket), ServeOptions {
				lsp_config_paths: Vec::new(),
				shutdown:         Some(task_shutdown),
				server_build:     Str::default(),
				connections:      None,
			})
			.await
		});
		let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
		loop {
			if UnixStream::connect(&socket).await.is_ok() {
				break;
			}
			assert!(tokio::time::Instant::now() < deadline, "document socket did not start");
			tokio::time::sleep(Duration::from_millis(10)).await;
		}

		shutdown.cancel();
		timeout(Duration::from_secs(5), task)
			.await
			.expect("document authority stopped")
			.expect("document authority task")
			.expect("document authority result");
		assert!(!socket.exists(), "document socket must be removed after shutdown");
	}

	#[tokio::test]
	async fn socket_hello_advertises_configured_server_build() {
		let scratch = TempDir::new().expect("temporary directory");
		let project = scratch.path().join("project");
		let runtime = scratch.path().join("runtime");
		std::fs::create_dir(&project).expect("project directory");
		std::fs::create_dir(&runtime).expect("runtime directory");
		let socket = runtime.join("document.sock");
		let shutdown = CancellationToken::new();
		let (connection_tx, mut connection_rx) = watch::channel(usize::MAX);
		let task = tokio::spawn(serve(project, Transport::Socket(socket.clone()), ServeOptions {
			lsp_config_paths: Vec::new(),
			shutdown:         Some(shutdown.clone()),
			server_build:     Str::new_static("test-build"),
			connections:      Some(connection_tx),
		}));
		wait_for_connection_count(&mut connection_rx, 0).await;

		let mut stream = UnixStream::connect(&socket)
			.await
			.expect("connect document socket");
		let mut scratch = BytesMut::new();
		write_client_frame(
			&mut stream,
			&proto::ClientFrame {
				request_id: 0,
				body:       Some(proto::client_frame::Body::Hello(proto::ClientHello {
					protocol_major: PROTOCOL_MAJOR,
					protocol_minor: PROTOCOL_MINOR,
					client_id:      Bytes::from_static(b"daemon-test"),
				})),
			},
			FrameConfig::default(),
			&mut scratch,
		)
		.await
		.expect("write client hello");
		let response = read_server_frame(&mut stream, FrameConfig::default(), &mut scratch)
			.await
			.expect("read server hello")
			.expect("server hello frame");
		let Some(proto::server_frame::Body::Hello(hello)) = response.body else {
			panic!("expected server hello");
		};
		assert_eq!(hello.server_build, "test-build");

		drop(stream);
		shutdown.cancel();
		timeout(Duration::from_secs(5), task)
			.await
			.expect("document authority stopped")
			.expect("document authority task")
			.expect("document authority result");
	}

	#[tokio::test]
	async fn socket_connection_gauge_tracks_connect_and_disconnect() {
		let scratch = TempDir::new().expect("temporary directory");
		let project = scratch.path().join("project");
		let runtime = scratch.path().join("runtime");
		std::fs::create_dir(&project).expect("project directory");
		std::fs::create_dir(&runtime).expect("runtime directory");
		let socket = runtime.join("document.sock");
		let shutdown = CancellationToken::new();
		let (connection_tx, mut connection_rx) = watch::channel(usize::MAX);
		let task = tokio::spawn(serve(project, Transport::Socket(socket.clone()), ServeOptions {
			lsp_config_paths: Vec::new(),
			shutdown:         Some(shutdown.clone()),
			server_build:     Str::default(),
			connections:      Some(connection_tx),
		}));
		wait_for_connection_count(&mut connection_rx, 0).await;

		let stream = UnixStream::connect(&socket)
			.await
			.expect("connect document socket");
		wait_for_connection_count(&mut connection_rx, 1).await;
		drop(stream);
		wait_for_connection_count(&mut connection_rx, 0).await;

		shutdown.cancel();
		timeout(Duration::from_secs(5), task)
			.await
			.expect("document authority stopped")
			.expect("document authority task")
			.expect("document authority result");
	}

	async fn wait_for_connection_count(receiver: &mut watch::Receiver<usize>, expected: usize) {
		timeout(Duration::from_secs(5), async {
			loop {
				let current = *receiver.borrow_and_update();
				if current == expected {
					break;
				}
				receiver.changed().await.expect("connection gauge sender");
			}
		})
		.await
		.expect("connection gauge update");
	}

	#[test]
	fn socket_location_rejects_workspace_paths() {
		let scratch = TempDir::new().expect("temporary directory");
		let project = scratch.path().join("project");
		let metadata = project.join(".omp");
		let runtime = scratch.path().join("runtime");
		std::fs::create_dir_all(&metadata).expect("project metadata directory");
		std::fs::create_dir(&runtime).expect("runtime directory");
		let project = std::fs::canonicalize(project).expect("canonical project");

		let error = validate_socket_location(&metadata.join("document.sock"), &project)
			.expect_err("workspace socket must be rejected");
		assert!(matches!(error, Error::SocketInsideEnvironmentRoot { .. }));
		validate_socket_location(&runtime.join("document.sock"), &project)
			.expect("external runtime socket");
	}
}
