//! Multi-client document authority over a local Unix socket or standard I/O.

#[cfg(unix)]
use std::path::Path;
#[cfg(unix)]
use std::{
	fs::{File, OpenOptions},
	os::unix::fs::{FileTypeExt, MetadataExt, PermissionsExt},
};
use std::{path::PathBuf, time::Duration};

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

	/// The directory containing the socket has invalid ownership or permissions.
	#[error("socket directory {directory:?} must be an owner-only directory owned by uid {user_id}")]
	InvalidSocketDirectoryPermissions {
		/// Path to the socket directory.
		directory: PathBuf,
		/// Effective user ID.
		user_id:   u32,
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

/// Runs the document authority rooted at `root` on `transport` until shutdown.
///
/// Every `lsp_config_path` is parsed before authority is acquired. All declared
/// processes complete initialization and registry installation before the
/// client transport starts accepting requests. On Unix, socket endpoints are
/// protected by a separate instance lock and created sockets are owner-only.
/// `SIGINT` or `SIGTERM` starts graceful connection, LSP, and actor shutdown.
pub async fn run(root: PathBuf, transport: Transport, lsp_config_paths: Vec<PathBuf>) -> Result {
	let process_configs = load_lsp_process_configs(&lsp_config_paths)?;
	let config = ServerConfig::new(root)?;
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
	let serve_result = match transport {
		Transport::Stdio => serve_stdio(environment.clone()).await,
		Transport::Socket(path) => serve_socket(environment.clone(), path).await,
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
	let result = serve_io_until(
		environment.session(),
		tokio::io::stdin(),
		tokio::io::stdout(),
		ConnectionConfig::default(),
		shutdown,
	)
	.await;
	signal.abort();
	result.map_err(Into::into)
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
	let parent_metadata = std::fs::symlink_metadata(&canonical_parent)?;
	let user_id = rustix::process::geteuid().as_raw();
	if !parent_metadata.is_dir()
		|| parent_metadata.uid() != user_id
		|| parent_metadata.mode() & 0o077 != 0
	{
		return Err(Error::InvalidSocketDirectoryPermissions {
			directory: canonical_parent,
			user_id,
		});
	}
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
async fn serve_socket(environment: Environment, path: PathBuf) -> Result {
	let root = environment
		.root_uri()
		.to_file_path()
		.map_err(|()| Error::NonFileUriRoot)?;
	let parent = path
		.parent()
		.filter(|parent| !parent.as_os_str().is_empty())
		.unwrap_or_else(|| Path::new("."));
	let canonical_parent = std::fs::canonicalize(parent)?;
	if canonical_parent.starts_with(&root) {
		return Err(Error::SocketInsideEnvironmentRoot { path, root });
	}
	let (listener, socket) = bind_socket(path).await?;
	let shutdown = CancellationToken::new();
	let signal = shutdown_signal();
	tokio::pin!(signal);
	let mut connections = JoinSet::new();

	loop {
		tokio::select! {
			result = &mut signal => {
				result?;
				shutdown.cancel();
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
			},
			completed = connections.join_next(), if !connections.is_empty() => {
				if let Some(completed) = completed {
					report_connection(completed);
				}
			},
		}
	}

	drop(listener);
	let drain = async {
		while let Some(completed) = connections.join_next().await {
			report_connection(completed);
		}
	};
	if timeout(SHUTDOWN_GRACE, drain).await.is_err() {
		connections.shutdown().await;
	}
	drop(socket);
	Ok(())
}

#[cfg(not(unix))]
async fn serve_socket(_environment: Environment, _path: PathBuf) -> Result {
	Err(Error::UnsupportedSocket)
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
	use tempfile::TempDir;

	use super::*;

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
	async fn socket_binding_rejects_a_shared_parent_directory() {
		let root = TempDir::new().expect("temporary directory");
		let shared = root.path().join("shared");
		std::fs::create_dir(&shared).expect("create shared directory");
		std::fs::set_permissions(&shared, std::fs::Permissions::from_mode(0o755))
			.expect("make parent shared");
		let path = shared.join("document.sock");

		assert!(bind_socket(path.clone()).await.is_err());
		assert!(!path.exists());
	}
}
