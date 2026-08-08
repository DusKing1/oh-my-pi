//! One transport boundary for native gRPC and foreign vendor HTTP APIs.
//!
//! Both protocol families share every accepted connection. A request is native
//! gRPC only when it is HTTP/2 **and** its media type starts with
//! `application/grpc` (including `application/grpc+proto`); every other request
//! goes to [`crate::facade::Router`]. Requiring both signals prevents an HTTP/1
//! facade request with a spoofed content type from entering tonic.
//!
//! Graceful shutdown first stops accepting connections, asks Hyper to finish
//! active streams, and then waits for every response body to finish or detach.
//! Video renders outlive that drain: submit returns a durable generation
//! handle, and dropping `AttachGeneration` only detaches its observer. The job
//! itself is owned by the media service and only `CancelGeneration` can stop
//! it.

use std::{
	convert::Infallible,
	future::{Future, Ready, ready},
	net::SocketAddr,
	path::{Path, PathBuf},
	pin::Pin,
	sync::{
		Arc,
		atomic::{AtomicUsize, Ordering},
	},
	task::{Context, Poll},
};

use bytes::Bytes;
use http::{Request, Response, StatusCode, Version, header};
use hyper::body::{Body as HttpBody, Frame};
use omp_proto::{
	auth::v1::auth_server::{Auth, AuthServer},
	blob::v1::blob_server::BlobServer,
	gateway::v1::gateway_server::GatewayServer,
	inference::v1::inference_server::{Inference, InferenceServer},
};
use omp_rpc::{HelloService, TlsConfig};
use tokio::{
	net::TcpListener,
	sync::{Notify, watch},
};
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{body::Body, codegen::Service, service::Routes, transport::Server};

use crate::{blob::BlobService, facade};

/// A listener setup or serving failure.
#[derive(Debug, thiserror::Error)]
pub enum ListenerError {
	/// The shared RPC transport could not bind or configure the listener.
	#[error(transparent)]
	Rpc(#[from] omp_rpc::Error),
	/// Tonic could not configure or run the HTTP server.
	#[error("gateway HTTP transport failed")]
	Transport,
	/// Remote serving was configured without either mandatory mTLS or a bearer
	/// token.
	#[error("remote listener requires a client CA or a non-empty bearer token")]
	MissingRemoteAuthentication,
}

impl From<tonic::transport::Error> for ListenerError {
	fn from(_: tonic::transport::Error) -> Self {
		Self::Transport
	}
}

/// Cloneable control used to initiate graceful listener shutdown.
#[derive(Clone, Debug)]
pub struct ListenerControl {
	shutdown: watch::Sender<bool>,
}

impl Default for ListenerControl {
	fn default() -> Self {
		let (shutdown, _) = watch::channel(false);
		Self { shutdown }
	}
}

impl ListenerControl {
	/// Creates an untriggered shutdown control.
	#[must_use]
	pub fn new() -> Self {
		Self::default()
	}

	/// Stops acceptance and begins draining active requests.
	pub fn shutdown(&self) {
		let _ = self.shutdown.send_replace(true);
	}

	async fn cancelled(&self) {
		let mut receiver = self.shutdown.subscribe();
		if *receiver.borrow() {
			return;
		}
		while receiver.changed().await.is_ok() {
			if *receiver.borrow_and_update() {
				return;
			}
		}
	}
}

/// Native and foreign services mounted on one listener.
pub struct Services<I, A> {
	inference: I,
	auth:      A,
	blob:      BlobService,
	facade:    facade::Router,
	hello:     HelloService,
	control:   ListenerControl,
}

impl<I, A> Services<I, A> {
	/// Creates a service bundle with a fresh graceful-shutdown control.
	#[must_use]
	pub fn new(
		inference: I,
		auth: A,
		blob: BlobService,
		facade: facade::Router,
		hello: HelloService,
	) -> Self {
		Self { inference, auth, blob, facade, hello, control: ListenerControl::new() }
	}

	/// Returns a handle that can stop and drain the listener.
	#[must_use]
	pub fn control(&self) -> ListenerControl {
		self.control.clone()
	}
}

/// TLS identity and client authentication for a remote listener.
#[derive(Clone, Debug)]
pub struct RemoteTls {
	transport: TlsConfig,
	auth:      RemoteAuth,
}

#[derive(Clone, Debug)]
enum RemoteAuth {
	MutualTls,
	Bearer([u8; 32]),
}

impl RemoteTls {
	/// Uses mandatory client certificates signed by `transport.client_ca`.
	///
	/// # Errors
	///
	/// Returns [`ListenerError::MissingRemoteAuthentication`] when no client CA
	/// is configured.
	pub fn mutual_tls(transport: TlsConfig) -> Result<Self, ListenerError> {
		if transport.client_ca.is_none() {
			return Err(ListenerError::MissingRemoteAuthentication);
		}
		Ok(Self { transport, auth: RemoteAuth::MutualTls })
	}

	/// Uses a TLS server identity plus a gateway bearer token.
	///
	/// Only a BLAKE3 digest is retained by the listener configuration.
	///
	/// # Errors
	///
	/// Returns [`ListenerError::MissingRemoteAuthentication`] for an empty
	/// token.
	pub fn bearer(transport: TlsConfig, token: &[u8]) -> Result<Self, ListenerError> {
		if token.is_empty() {
			return Err(ListenerError::MissingRemoteAuthentication);
		}
		Ok(Self { transport, auth: RemoteAuth::Bearer(*blake3::hash(token).as_bytes()) })
	}
}
/// A bound owner-only Unix listener.
///
/// Binding is separated from serving so daemon startup can report readiness
/// only after the socket exists and has its final `0600` permissions.
#[cfg(unix)]
pub struct LocalListener {
	path:     PathBuf,
	incoming: Option<omp_rpc::uds::Incoming>,
}

#[cfg(unix)]
impl LocalListener {
	/// Binds an owner-only Unix socket without beginning request dispatch.
	pub async fn bind(path: impl AsRef<Path>) -> Result<Self, ListenerError> {
		let path = path.as_ref().to_owned();
		let incoming = omp_rpc::uds::listen(&path).await?;
		Ok(Self { path, incoming: Some(incoming) })
	}

	/// Returns the bound socket path.
	#[must_use]
	pub fn path(&self) -> &Path {
		&self.path
	}

	/// Serves the already-bound socket until graceful shutdown completes.
	pub async fn serve<I, A>(mut self, services: Services<I, A>) -> Result<(), ListenerError>
	where
		I: Inference,
		A: Auth,
	{
		let incoming = self
			.incoming
			.take()
			.expect("a bound local listener can only be served once");
		serve_local_incoming(&self.path, incoming, services).await
	}
}

#[cfg(unix)]
impl Drop for LocalListener {
	fn drop(&mut self) {
		if self.incoming.is_some() {
			let _ = std::fs::remove_file(&self.path);
		}
	}
}

/// A bound owner-only Windows named-pipe listener.
///
/// The first pipe instance is created during binding, so returning from
/// [`LocalListener::bind`] is the readiness boundary. Dropping the listener
/// closes the final server handle and removes its kernel object; Windows has no
/// filesystem entry that could remain stale.
#[cfg(windows)]
pub struct LocalListener {
	path:     PathBuf,
	incoming: Option<crate::local::PipeIncoming>,
}

#[cfg(windows)]
impl LocalListener {
	/// Binds a local-user-only Windows named pipe without beginning dispatch.
	pub async fn bind(path: impl AsRef<Path>) -> Result<Self, ListenerError> {
		let path = path.as_ref().to_owned();
		let endpoint = crate::local::LocalEndpoint::native(path.clone());
		let incoming = crate::local::listen_pipe(&endpoint)?;
		Ok(Self { path, incoming: Some(incoming) })
	}

	/// Returns the bound native named-pipe path.
	#[must_use]
	pub fn path(&self) -> &Path {
		&self.path
	}

	/// Serves the already-bound pipe until graceful shutdown completes.
	pub async fn serve<I, A>(mut self, services: Services<I, A>) -> Result<(), ListenerError>
	where
		I: Inference,
		A: Auth,
	{
		let incoming = self
			.incoming
			.take()
			.expect("a bound local listener can only be served once");
		serve_local_incoming(incoming, services).await
	}
}

/// A bound TLS TCP listener.
///
/// The effective address is available before serving, including an
/// operating-system-selected port when the configured port was zero.
pub struct RemoteListener {
	listener:   TcpListener,
	server_tls: tonic::transport::ServerTlsConfig,
	bearer:     Option<[u8; 32]>,
	addr:       SocketAddr,
}

impl RemoteListener {
	/// Binds a TCP address after validating the remote authentication policy.
	pub async fn bind(addr: SocketAddr, tls: RemoteTls) -> Result<Self, ListenerError> {
		let server_tls = omp_rpc::tls::server_tls(&tls.transport).await?;
		let bearer = match tls.auth {
			RemoteAuth::MutualTls => None,
			RemoteAuth::Bearer(digest) => Some(digest),
		};
		let listener = TcpListener::bind(addr)
			.await
			.map_err(omp_rpc::Error::from)?;
		let addr = listener.local_addr().map_err(omp_rpc::Error::from)?;
		Ok(Self { listener, server_tls, bearer, addr })
	}

	/// Returns the effective bound address.
	#[must_use]
	pub const fn local_addr(&self) -> SocketAddr {
		self.addr
	}

	/// Serves the already-bound TCP socket until graceful shutdown completes.
	pub async fn serve<I, A>(self, services: Services<I, A>) -> Result<(), ListenerError>
	where
		I: Inference,
		A: Auth,
	{
		serve_remote_incoming(self.listener, self.server_tls, self.bearer, services).await
	}
}

/// Serves the daemon protocol and vendor facades on an owner-only Unix socket.
///
/// Filesystem ownership and the socket's `0600` mode are the local security
/// boundary; no listener-level bearer token is required.
#[cfg(unix)]
pub async fn serve_local<I, A>(
	socket_path: &Path,
	services: Services<I, A>,
) -> Result<(), ListenerError>
where
	I: Inference,
	A: Auth,
{
	LocalListener::bind(socket_path)
		.await?
		.serve(services)
		.await
}

#[cfg(unix)]
async fn serve_local_incoming<I, A>(
	socket_path: &Path,
	incoming: omp_rpc::uds::Incoming,
	services: Services<I, A>,
) -> Result<(), ListenerError>
where
	I: Inference,
	A: Auth,
{
	let _ = omp_telemetry::export::init();
	let control = services.control.clone();
	let (multiplex, drain) = build_multiplex(services, None).await;
	let result = Server::builder()
		.accept_http1(true)
		.serve_with_incoming_shutdown(multiplex, incoming, control.cancelled())
		.await;
	drain.wait().await;
	omp_telemetry::export::flush();
	omp_telemetry::export::shutdown();
	match tokio::fs::remove_file(socket_path).await {
		Ok(()) => {},
		Err(error) if error.kind() == std::io::ErrorKind::NotFound => {},
		Err(error) if result.is_ok() => return Err(omp_rpc::Error::from(error).into()),
		Err(_) => {},
	}
	result.map_err(Into::into)
}

/// Serves the daemon protocol and vendor facades on a local-user-only Windows
/// named pipe.
#[cfg(windows)]
pub async fn serve_local<I, A>(
	pipe_path: &Path,
	services: Services<I, A>,
) -> Result<(), ListenerError>
where
	I: Inference,
	A: Auth,
{
	LocalListener::bind(pipe_path).await?.serve(services).await
}

#[cfg(windows)]
async fn serve_local_incoming<I, A>(
	incoming: crate::local::PipeIncoming,
	services: Services<I, A>,
) -> Result<(), ListenerError>
where
	I: Inference,
	A: Auth,
{
	let _ = omp_telemetry::export::init();
	let control = services.control.clone();
	let (multiplex, drain) = build_multiplex(services, None).await;
	let result = Server::builder()
		.accept_http1(true)
		.serve_with_incoming_shutdown(multiplex, incoming, control.cancelled())
		.await;
	drain.wait().await;
	omp_telemetry::export::flush();
	omp_telemetry::export::shutdown();
	result.map_err(Into::into)
}

/// Serves native gRPC and vendor facades on TCP with mTLS or bearer auth.
pub async fn serve_remote<I, A>(
	addr: SocketAddr,
	tls: RemoteTls,
	services: Services<I, A>,
) -> Result<(), ListenerError>
where
	I: Inference,
	A: Auth,
{
	RemoteListener::bind(addr, tls).await?.serve(services).await
}

async fn serve_remote_incoming<I, A>(
	listener: TcpListener,
	server_tls: tonic::transport::ServerTlsConfig,
	bearer: Option<[u8; 32]>,
	services: Services<I, A>,
) -> Result<(), ListenerError>
where
	I: Inference,
	A: Auth,
{
	let control = services.control.clone();

	let server = Server::builder()
		.accept_http1(true)
		.tls_config(server_tls)?;
	let _ = omp_telemetry::export::init();
	let (multiplex, drain) = build_multiplex(services, bearer).await;
	let result = server
		.serve_with_incoming_shutdown(
			multiplex,
			TcpListenerStream::new(listener),
			control.cancelled(),
		)
		.await;
	drain.wait().await;
	omp_telemetry::export::flush();
	omp_telemetry::export::shutdown();
	result.map_err(Into::into)
}

async fn build_multiplex<I, A>(
	services: Services<I, A>,
	bearer: Option<[u8; 32]>,
) -> (Multiplex<facade::Router>, Drain)
where
	I: Inference,
	A: Auth,
{
	let inference = InferenceServer::new(services.inference);
	let auth = AuthServer::new(services.auth);
	let blob = BlobServer::new(services.blob);
	let hello = GatewayServer::new(services.hello);
	let (health_reporter, health) = omp_rpc::health::health_service();
	health_reporter.set_serving::<InferenceServer<I>>().await;
	health_reporter.set_serving::<AuthServer<A>>().await;
	health_reporter
		.set_serving::<BlobServer<BlobService>>()
		.await;
	health_reporter
		.set_serving::<GatewayServer<HelloService>>()
		.await;

	let mut routes = Routes::builder();
	routes
		.add_service(inference)
		.add_service(blob)
		.add_service(auth)
		.add_service(hello)
		.add_service(health);
	let drain = Drain::default();
	(Multiplex::new(routes.routes().prepare(), services.facade, bearer, drain.clone()), drain)
}

trait FacadeDispatch: Clone + Send + Sync + 'static {
	fn dispatch(&self, request: Request<Body>) -> FacadeFuture;
}

type FacadeFuture = Pin<Box<dyn Future<Output = Result<Response<Body>, Infallible>> + Send>>;

impl FacadeDispatch for facade::Router {
	fn dispatch(&self, request: Request<Body>) -> FacadeFuture {
		let router = self.clone();
		// This is the cold foreign-wire/network boundary. Its single allocation
		// erases the many handler futures once, rather than boxing per body frame.
		Box::pin(async move { Ok(router.route(request).await.map(Body::new)) })
	}
}

#[derive(Clone)]
struct Multiplex<F> {
	grpc:   Routes,
	facade: F,
	bearer: Option<[u8; 32]>,
	drain:  Drain,
}

impl<F> Multiplex<F> {
	const fn new(grpc: Routes, facade: F, bearer: Option<[u8; 32]>, drain: Drain) -> Self {
		Self { grpc, facade, bearer, drain }
	}
}

impl<F> Service<Request<Body>> for Multiplex<F>
where
	F: FacadeDispatch,
{
	type Error = Infallible;
	type Future = DispatchFuture;
	type Response = Response<TrackedBody>;

	fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		// Tonic Routes is always cheaply ready; facade admission and provider
		// backpressure are enforced inside Router::route and the inference stack.
		Poll::Ready(Ok(()))
	}

	fn call(&mut self, request: Request<Body>) -> Self::Future {
		let guard = self.drain.begin();
		if let Some(expected) = self.bearer
			&& !valid_bearer(&request, &expected)
		{
			let response = unauthorized(is_grpc(&request));
			return DispatchFuture::new(DispatchKind::Ready(ready(Ok(response))), guard);
		}
		if is_grpc(&request) {
			DispatchFuture::new(DispatchKind::Grpc(self.grpc.call(request)), guard)
		} else {
			DispatchFuture::new(DispatchKind::Facade(self.facade.dispatch(request)), guard)
		}
	}
}

type GrpcFuture = <Routes as Service<Request<Body>>>::Future;

enum DispatchKind {
	Grpc(GrpcFuture),
	Facade(FacadeFuture),
	Ready(Ready<Result<Response<Body>, Infallible>>),
}

struct DispatchFuture {
	kind:  DispatchKind,
	guard: Option<InFlight>,
}

impl DispatchFuture {
	const fn new(kind: DispatchKind, guard: InFlight) -> Self {
		Self { kind, guard: Some(guard) }
	}
}

impl Future for DispatchFuture {
	type Output = Result<Response<TrackedBody>, Infallible>;

	fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
		// SAFETY: `kind` is never moved or replaced after `DispatchFuture` is
		// pinned. Each projected future therefore remains pinned for its lifetime.
		let this = unsafe { self.get_unchecked_mut() };
		let polled = match &mut this.kind {
			// SAFETY: justified by the invariant above.
			DispatchKind::Grpc(future) => unsafe { Pin::new_unchecked(future) }.poll(cx),
			DispatchKind::Facade(future) => future.as_mut().poll(cx),
			// SAFETY: justified by the invariant above.
			DispatchKind::Ready(future) => unsafe { Pin::new_unchecked(future) }.poll(cx),
		};
		polled.map(|result| {
			result.map(|response| {
				let guard = this
					.guard
					.take()
					.expect("dispatch future polled after completion");
				response.map(|body| TrackedBody { body, _guard: guard })
			})
		})
	}
}

struct TrackedBody {
	body:   Body,
	_guard: InFlight,
}

impl HttpBody for TrackedBody {
	type Data = Bytes;
	type Error = tonic::Status;

	fn poll_frame(
		self: Pin<&mut Self>,
		cx: &mut Context<'_>,
	) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
		// SAFETY: `body` is structurally pinned and never moved while `self` is pinned.
		unsafe { self.map_unchecked_mut(|tracked| &mut tracked.body) }.poll_frame(cx)
	}

	fn is_end_stream(&self) -> bool {
		self.body.is_end_stream()
	}

	fn size_hint(&self) -> hyper::body::SizeHint {
		self.body.size_hint()
	}
}

#[derive(Clone, Default)]
struct Drain {
	state: Arc<DrainState>,
}

#[derive(Default)]
struct DrainState {
	active: AtomicUsize,
	idle:   Notify,
}

impl Drain {
	fn begin(&self) -> InFlight {
		self.state.active.fetch_add(1, Ordering::AcqRel);
		InFlight { state: Arc::clone(&self.state) }
	}

	async fn wait(&self) {
		while self.state.active.load(Ordering::Acquire) != 0 {
			self.state.idle.notified().await;
		}
	}
}

struct InFlight {
	state: Arc<DrainState>,
}

impl Drop for InFlight {
	fn drop(&mut self) {
		if self.state.active.fetch_sub(1, Ordering::AcqRel) == 1 {
			self.state.idle.notify_one();
		}
	}
}

fn is_grpc<B>(request: &Request<B>) -> bool {
	request.version() == Version::HTTP_2
		&& request
			.headers()
			.get(header::CONTENT_TYPE)
			.and_then(|value| value.to_str().ok())
			.and_then(|value| value.split(';').next())
			.is_some_and(|media_type| {
				let media_type = media_type.trim();
				media_type == "application/grpc" || media_type.starts_with("application/grpc+")
			})
}

fn valid_bearer<B>(request: &Request<B>, expected: &[u8; 32]) -> bool {
	let Some(token) = request
		.headers()
		.get(header::AUTHORIZATION)
		.and_then(|value| value.to_str().ok())
		.and_then(|value| value.split_once(' '))
		.filter(|(scheme, token)| scheme.eq_ignore_ascii_case("bearer") && !token.is_empty())
		.map(|(_, token)| token)
	else {
		return false;
	};
	constant_time_eq(blake3::hash(token.as_bytes()).as_bytes(), expected)
}

fn constant_time_eq(left: &[u8; 32], right: &[u8; 32]) -> bool {
	left
		.iter()
		.zip(right)
		.fold(0_u8, |difference, (left, right)| difference | (left ^ right))
		== 0
}

fn unauthorized(grpc: bool) -> Response<Body> {
	if grpc {
		Response::builder()
			.status(StatusCode::OK)
			.header(header::CONTENT_TYPE, "application/grpc")
			.header("grpc-status", "16")
			.body(Body::empty())
			.expect("static gRPC authentication response is valid")
	} else {
		Response::builder()
			.status(StatusCode::UNAUTHORIZED)
			.body(Body::empty())
			.expect("static HTTP authentication response is valid")
	}
}

#[cfg(test)]
mod tests {
	#[cfg(unix)]
	use std::os::unix::fs::PermissionsExt;
	use std::{
		future::Ready,
		sync::{
			Arc,
			atomic::{AtomicUsize, Ordering},
		},
		task::{Context, Poll},
	};

	use http::{Request, Response, StatusCode, Version, header};
	use tonic::{body::Body, codegen::Service, server::NamedService, service::Routes};

	use super::{Drain, FacadeDispatch, FacadeFuture, Multiplex, TrackedBody};

	#[derive(Clone)]
	struct GrpcProbe(Arc<AtomicUsize>);

	impl NamedService for GrpcProbe {
		const NAME: &'static str = "test.Dispatch";
	}

	impl Service<Request<Body>> for GrpcProbe {
		type Error = std::convert::Infallible;
		type Future = Ready<Result<Self::Response, Self::Error>>;
		type Response = Response<Body>;

		fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
			Poll::Ready(Ok(()))
		}

		fn call(&mut self, _request: Request<Body>) -> Self::Future {
			self.0.fetch_add(1, Ordering::Relaxed);
			std::future::ready(Ok(Response::new(Body::empty())))
		}
	}

	#[derive(Clone)]
	struct FacadeProbe(Arc<AtomicUsize>);

	impl FacadeDispatch for FacadeProbe {
		fn dispatch(&self, _request: Request<Body>) -> FacadeFuture {
			self.0.fetch_add(1, Ordering::Relaxed);
			Box::pin(async {
				Ok(Response::builder()
					.status(StatusCode::NO_CONTENT)
					.body(Body::empty())
					.unwrap())
			})
		}
	}

	#[tokio::test]
	async fn grpc_and_facade_share_one_dispatcher() {
		let grpc_calls = Arc::new(AtomicUsize::new(0));
		let facade_calls = Arc::new(AtomicUsize::new(0));
		let grpc = Routes::new(GrpcProbe(Arc::clone(&grpc_calls))).prepare();
		let mut listener =
			Multiplex::new(grpc, FacadeProbe(Arc::clone(&facade_calls)), None, Drain::default());

		let grpc_request = Request::builder()
			.version(Version::HTTP_2)
			.uri("/test.Dispatch/Call")
			.header(header::CONTENT_TYPE, "application/grpc+proto")
			.body(Body::empty())
			.unwrap();
		let grpc_response = listener.call(grpc_request).await.unwrap();
		assert_eq!(grpc_response.status(), StatusCode::OK);

		let facade_request = Request::builder()
			.version(Version::HTTP_11)
			.uri("/v1/models")
			.body(Body::empty())
			.unwrap();
		let facade_response = listener.call(facade_request).await.unwrap();
		assert_eq!(facade_response.status(), StatusCode::NO_CONTENT);
		assert_eq!(grpc_calls.load(Ordering::Relaxed), 1);
		assert_eq!(facade_calls.load(Ordering::Relaxed), 1);
	}

	#[cfg(unix)]
	#[tokio::test]
	async fn local_socket_is_owner_only() {
		let directory = tempfile::tempdir().unwrap();
		let socket = directory.path().join("daemon.sock");
		let incoming = omp_rpc::uds::listen(&socket).await.unwrap();
		let mode = tokio::fs::metadata(&socket)
			.await
			.unwrap()
			.permissions()
			.mode()
			& 0o777;
		assert_eq!(mode, 0o600);
		drop(incoming);
	}

	#[tokio::test]
	async fn graceful_drain_waits_for_in_flight_body() {
		let drain = Drain::default();
		let body = TrackedBody { body: Body::empty(), _guard: drain.begin() };
		let waiter = tokio::spawn({
			let drain = drain.clone();
			async move { drain.wait().await }
		});
		tokio::task::yield_now().await;
		assert!(!waiter.is_finished());
		drop(body);
		waiter.await.unwrap();
	}
}
