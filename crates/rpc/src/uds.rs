//! Unix-domain-socket transport for local daemon connections.

#[cfg(unix)]
use std::os::unix::fs::{FileTypeExt, PermissionsExt};
use std::path::Path;

#[cfg(unix)]
use hyper_util::rt::TokioIo;
#[cfg(unix)]
use tokio::net::{UnixListener, UnixStream};
#[cfg(unix)]
use tokio_stream::wrappers::UnixListenerStream;
use tonic::transport::Channel;
#[cfg(unix)]
use tower::service_fn;

use crate::Error;

/// A stream of accepted local RPC connections.
#[cfg(unix)]
pub type Incoming = UnixListenerStream;

/// Placeholder incoming stream on platforms without Unix-domain sockets.
///
/// Named-pipe support is tracked separately. Until it lands, Windows clients
/// should use TCP on localhost with token authentication.
#[cfg(windows)]
#[derive(Debug)]
pub struct Incoming;

/// Bind an owner-only Unix-domain socket and return its incoming connection
/// stream.
///
/// Parent directories are created as needed. An existing path is removed only
/// when it is a socket that cannot be connected to; an active socket or a
/// non-socket path is left untouched.
#[cfg(unix)]
pub async fn listen(path: &Path) -> Result<Incoming, Error> {
	if let Some(parent) = path.parent()
		&& !parent.as_os_str().is_empty()
	{
		tokio::fs::create_dir_all(parent).await?;
	}

	match tokio::fs::symlink_metadata(path).await {
		Ok(metadata) if metadata.file_type().is_socket() => {
			if UnixStream::connect(path).await.is_ok() {
				return Err(
					std::io::Error::new(
						std::io::ErrorKind::AddrInUse,
						"Unix socket is already accepting connections",
					)
					.into(),
				);
			}
			tracing::debug!(socket = %path.display(), "removing stale Unix socket");
			tokio::fs::remove_file(path).await?;
		},
		Ok(_) => {},
		Err(error) if error.kind() == std::io::ErrorKind::NotFound => {},
		Err(error) => return Err(error.into()),
	}

	let listener = UnixListener::bind(path)?;
	tokio::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).await?;
	Ok(UnixListenerStream::new(listener))
}

/// Connect a tonic channel to a Unix-domain socket.
#[cfg(unix)]
pub async fn connect(path: &Path) -> Result<Channel, Error> {
	let path = path.to_owned();
	let endpoint = tonic::transport::Endpoint::from_static("http://[::]:50051");
	let channel = endpoint
		.connect_with_connector(service_fn(move |_| {
			let path = path.clone();
			async move { UnixStream::connect(path).await.map(TokioIo::new) }
		}))
		.await?;
	Ok(channel)
}

/// Return an unsupported error on Windows.
///
/// Named-pipe support is tracked separately. Until it lands, use TCP on
/// localhost with token authentication.
#[cfg(windows)]
pub async fn listen(_path: &Path) -> Result<Incoming, Error> {
	Err(Error::Unsupported("Unix-domain sockets are unavailable on Windows"))
}

/// Return an unsupported error on Windows.
///
/// Named-pipe support is tracked separately. Until it lands, use TCP on
/// localhost with token authentication.
#[cfg(windows)]
pub async fn connect(_path: &Path) -> Result<Channel, Error> {
	Err(Error::Unsupported("Unix-domain sockets are unavailable on Windows"))
}
