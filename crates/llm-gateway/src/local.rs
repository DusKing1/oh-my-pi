//! Platform-native local daemon endpoints and client connections.
//!
//! Unix endpoints are owner-only domain sockets. Windows endpoints are
//! byte-mode named pipes under `\\.\pipe\`, restricted to the creating user,
//! local administrators, and Local System. Neither platform falls back to TCP.

use std::{
	fmt,
	path::{Path, PathBuf},
	str::FromStr,
};

use tonic::transport::Channel;

/// A platform-native local daemon endpoint.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct LocalEndpoint(PathBuf);

impl LocalEndpoint {
	/// Creates an endpoint from its native operating-system spelling.
	///
	/// Validation is performed when the endpoint is parsed from the CLI or
	/// bound.
	#[must_use]
	pub fn native(endpoint: impl Into<PathBuf>) -> Self {
		Self(endpoint.into())
	}

	/// Returns the native endpoint spelling.
	#[must_use]
	pub fn as_path(&self) -> &Path {
		&self.0
	}

	#[cfg(windows)]
	pub(crate) fn validate_pipe(&self) -> Result<(), LocalEndpointError> {
		let Some(pipe) = self.0.to_str() else {
			return Err(LocalEndpointError::NonUnicodeNamedPipe);
		};
		validate_windows_pipe(pipe)
	}
}

impl AsRef<Path> for LocalEndpoint {
	fn as_ref(&self) -> &Path {
		self.as_path()
	}
}

impl From<PathBuf> for LocalEndpoint {
	fn from(endpoint: PathBuf) -> Self {
		Self::native(endpoint)
	}
}

impl fmt::Display for LocalEndpoint {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		self.0.display().fmt(formatter)
	}
}

/// A malformed platform-local endpoint.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum LocalEndpointError {
	/// The endpoint is empty.
	#[error("local endpoint cannot be empty")]
	Empty,
	/// The endpoint explicitly selects a local transport from another platform.
	#[error("local endpoint scheme is unavailable on this platform")]
	WrongPlatform,
	/// Windows named pipes require a Unicode Win32 object name.
	#[cfg(windows)]
	#[error("Windows named-pipe endpoints must be valid Unicode")]
	NonUnicodeNamedPipe,
	/// The endpoint is not a local Windows named-pipe name.
	#[cfg(windows)]
	#[error("Windows local endpoint must be \\\\.\\pipe\\NAME or npipe://./pipe/NAME")]
	InvalidNamedPipe,
}

impl FromStr for LocalEndpoint {
	type Err = LocalEndpointError;

	fn from_str(endpoint: &str) -> Result<Self, Self::Err> {
		if endpoint.is_empty() {
			return Err(LocalEndpointError::Empty);
		}
		#[cfg(unix)]
		{
			if endpoint.starts_with("npipe://") {
				return Err(LocalEndpointError::WrongPlatform);
			}
			let path = endpoint.strip_prefix("unix://").unwrap_or(endpoint);
			if path.is_empty() {
				return Err(LocalEndpointError::Empty);
			}
			Ok(Self(PathBuf::from(path)))
		}
		#[cfg(windows)]
		{
			if endpoint.starts_with("unix://") {
				return Err(LocalEndpointError::WrongPlatform);
			}
			let pipe = if let Some(uri) = endpoint.strip_prefix("npipe://") {
				let Some(name) = uri.trim_start_matches('/').strip_prefix("./pipe/") else {
					return Err(LocalEndpointError::InvalidNamedPipe);
				};
				if name.is_empty() {
					return Err(LocalEndpointError::InvalidNamedPipe);
				}
				format!(r"\\.\pipe\{}", name.replace('/', r"\"))
			} else {
				endpoint.to_owned()
			};
			validate_windows_pipe(&pipe)?;
			Ok(Self(PathBuf::from(pipe)))
		}
	}
}

/// Connects a tonic channel to a platform-native local daemon endpoint.
pub async fn connect(endpoint: &LocalEndpoint) -> Result<Channel, omp_rpc::Error> {
	#[cfg(unix)]
	{
		omp_rpc::uds::connect(endpoint.as_path()).await
	}
	#[cfg(windows)]
	{
		connect_pipe(endpoint).await
	}
}

#[cfg(windows)]
fn validate_windows_pipe(pipe: &str) -> Result<(), LocalEndpointError> {
	let Some(name) = pipe.strip_prefix(r"\\.\pipe\") else {
		return Err(LocalEndpointError::InvalidNamedPipe);
	};
	if name.is_empty() || name.contains('/') {
		return Err(LocalEndpointError::InvalidNamedPipe);
	}
	Ok(())
}

#[cfg(windows)]
use std::{
	ffi::{OsStr, c_void},
	io,
	pin::Pin,
	task::{Context, Poll},
};

#[cfg(windows)]
use hyper_util::rt::TokioIo;
#[cfg(windows)]
use tokio::{
	io::{AsyncRead, AsyncWrite, ReadBuf},
	net::windows::named_pipe::{ClientOptions, NamedPipeClient, NamedPipeServer, ServerOptions},
};
#[cfg(windows)]
use tokio_stream::Stream;
#[cfg(windows)]
use tonic::transport::server::Connected;
#[cfg(windows)]
use tower::service_fn;
#[cfg(windows)]
use windows_sys::Win32::{
	Foundation::{ERROR_PIPE_BUSY, LocalFree},
	Security::{
		Authorization::{ConvertStringSecurityDescriptorToSecurityDescriptorW, SDDL_REVISION_1},
		PSECURITY_DESCRIPTOR, SECURITY_ATTRIBUTES,
	},
};

/// A stream of accepted owner-only Windows named-pipe connections.
#[cfg(windows)]
pub(crate) type PipeIncoming = Pin<Box<dyn Stream<Item = io::Result<PipeConnection>> + Send>>;

/// Binds the first named-pipe instance and returns an incoming connection
/// stream.
#[cfg(windows)]
pub(crate) fn listen_pipe(endpoint: &LocalEndpoint) -> Result<PipeIncoming, omp_rpc::Error> {
	endpoint.validate_pipe().map_err(invalid_endpoint)?;
	let name = endpoint.as_path().to_owned();
	let descriptor = OwnerOnlyDescriptor::new()?;
	let first = create_pipe(&name, true, &descriptor)?;
	Ok(Box::pin(async_stream::try_stream! {
		let mut pending = first;
		loop {
			pending.connect().await?;
			let connected = pending;
			// Create the next listening instance before yielding this connection so
			// multiple clients can connect without serializing behind request work.
			pending = create_pipe(&name, false, &descriptor)?;
			yield PipeConnection(connected);
		}
	}))
}

#[cfg(windows)]
fn create_pipe(
	name: &OsStr,
	first: bool,
	descriptor: &OwnerOnlyDescriptor,
) -> io::Result<NamedPipeServer> {
	let mut attributes = SECURITY_ATTRIBUTES {
		nLength:              std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
		lpSecurityDescriptor: descriptor.0,
		bInheritHandle:       0,
	};
	let mut options = ServerOptions::new();
	options
		.first_pipe_instance(first)
		.reject_remote_clients(true);
	// SAFETY: `attributes` and its security descriptor remain alive for the
	// duration of CreateNamedPipeW. Windows copies the descriptor into the object.
	unsafe {
		options.create_with_security_attributes_raw(name, (&raw mut attributes).cast::<c_void>())
	}
}

#[cfg(windows)]
struct OwnerOnlyDescriptor(PSECURITY_DESCRIPTOR);

// SAFETY: the descriptor is uniquely owned, Windows treats it as read-only
// during each synchronous CreateNamedPipeW call, and LocalFree is thread-safe.
#[cfg(windows)]
unsafe impl Send for OwnerOnlyDescriptor {}

#[cfg(windows)]
impl OwnerOnlyDescriptor {
	fn new() -> io::Result<Self> {
		// OW is the creating token's owner. System and administrators retain the
		// access required to manage a wedged daemon, while Everyone and anonymous
		// receive no pipe access.
		let sddl: Vec<u16> = "D:P(A;;GA;;;OW)(A;;GA;;;SY)(A;;GA;;;BA)\0"
			.encode_utf16()
			.collect();
		let mut descriptor = std::ptr::null_mut();
		// SAFETY: `sddl` is NUL-terminated and `descriptor` is a valid out pointer.
		let converted = unsafe {
			ConvertStringSecurityDescriptorToSecurityDescriptorW(
				sddl.as_ptr(),
				SDDL_REVISION_1,
				&raw mut descriptor,
				std::ptr::null_mut(),
			)
		};
		if converted == 0 {
			return Err(io::Error::last_os_error());
		}
		Ok(Self(descriptor))
	}
}

#[cfg(windows)]
impl Drop for OwnerOnlyDescriptor {
	fn drop(&mut self) {
		// SAFETY: the descriptor was allocated by LocalAlloc inside the conversion
		// API and is freed exactly once here.
		unsafe {
			LocalFree(self.0);
		}
	}
}

#[cfg(windows)]
pub(crate) struct PipeConnection(NamedPipeServer);

#[cfg(windows)]
impl Connected for PipeConnection {
	type ConnectInfo = ();

	fn connect_info(&self) -> Self::ConnectInfo {}
}

#[cfg(windows)]
impl AsyncRead for PipeConnection {
	fn poll_read(
		mut self: Pin<&mut Self>,
		cx: &mut Context<'_>,
		buf: &mut ReadBuf<'_>,
	) -> Poll<io::Result<()>> {
		Pin::new(&mut self.0).poll_read(cx, buf)
	}
}

#[cfg(windows)]
impl AsyncWrite for PipeConnection {
	fn poll_write(
		mut self: Pin<&mut Self>,
		cx: &mut Context<'_>,
		buf: &[u8],
	) -> Poll<io::Result<usize>> {
		Pin::new(&mut self.0).poll_write(cx, buf)
	}

	fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
		Pin::new(&mut self.0).poll_flush(cx)
	}

	fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
		Pin::new(&mut self.0).poll_shutdown(cx)
	}
}

#[cfg(windows)]
async fn connect_pipe(endpoint: &LocalEndpoint) -> Result<Channel, omp_rpc::Error> {
	endpoint.validate_pipe().map_err(invalid_endpoint)?;
	let name = endpoint.as_path().to_owned();
	let channel = tonic::transport::Endpoint::from_static("http://[::]:50051")
		.connect_with_connector(service_fn(move |_| {
			let name = name.clone();
			async move { open_pipe(&name).await.map(TokioIo::new) }
		}))
		.await?;
	Ok(channel)
}

#[cfg(windows)]
async fn open_pipe(name: &Path) -> io::Result<NamedPipeClient> {
	loop {
		match ClientOptions::new().open(name) {
			Ok(client) => return Ok(client),
			Err(error) if error.raw_os_error() == Some(ERROR_PIPE_BUSY as i32) => {
				// ERROR_PIPE_BUSY: every instance is connected. Yield rather than
				// substituting an unrelated TCP transport.
				tokio::time::sleep(std::time::Duration::from_millis(10)).await;
			},
			Err(error) => return Err(error),
		}
	}
}

#[cfg(windows)]
fn invalid_endpoint(error: LocalEndpointError) -> omp_rpc::Error {
	omp_rpc::Error::Io(io::Error::new(io::ErrorKind::InvalidInput, error))
}

#[cfg(test)]
mod tests {
	use super::*;

	#[cfg(unix)]
	#[test]
	fn parses_unix_uri_without_changing_native_paths() {
		assert_eq!(
			"/tmp/omp.sock".parse::<LocalEndpoint>().unwrap().as_path(),
			Path::new("/tmp/omp.sock")
		);
		assert_eq!(
			"unix:///tmp/omp.sock"
				.parse::<LocalEndpoint>()
				.unwrap()
				.as_path(),
			Path::new("/tmp/omp.sock")
		);
		assert!("npipe://./pipe/omp".parse::<LocalEndpoint>().is_err());
	}

	#[cfg(windows)]
	#[test]
	fn parses_named_pipe_uri_to_native_name() {
		for uri in ["npipe://./pipe/omp/test", "npipe:////./pipe/omp/test"] {
			let endpoint = uri.parse::<LocalEndpoint>().unwrap();
			assert_eq!(endpoint.as_path(), Path::new(r"\\.\pipe\omp\test"));
		}
		assert!("http://127.0.0.1:1234".parse::<LocalEndpoint>().is_err());
		assert!("unix:///tmp/omp.sock".parse::<LocalEndpoint>().is_err());
	}
}
