//! Pooled HTTP transport for provider requests.
//!
//! This deliberately uses Hyper rather than `reqwest`: egress needs a proxy
//! decision per request, untouched streaming response bodies for SSE, HTTP/2,
//! and a path to provider-specific TLS profiles. The legacy Hyper client in
//! `hyper-util` supplies connection pooling while `hyper-rustls` supplies the
//! ring-backed `WebPKI` TLS connector.

use std::{
	future::{Future, Ready, ready},
	task::{Context, Poll},
	time::Duration,
};

use bytes::Bytes;
use futures::future::Either;
use http::{HeaderValue, Request, Response, header};
use http_body_util::Full;
use hyper::body::Incoming;
use hyper_rustls::{HttpsConnector, HttpsConnectorBuilder};
use hyper_util::{client::legacy::Client, rt::TokioExecutor};
use tower::{Layer, Service};

use super::{
	auth_inject::SensitiveQuery,
	proxy::{ProxyConnector, ProxyResolver},
};

/// Selects the workspace's ring-backed rustls provider before constructing a
/// TLS client.
///
/// The call is idempotent and preserves a provider explicitly installed by an
/// embedding process.
pub fn ensure_crypto_provider() {
	let _ = rustls::crypto::ring::default_provider().install_default();
}

/// A fully buffered provider request body.
///
/// Chat request encoding completes before egress begins. Keeping this body
/// cloneable is the invariant that makes retries before the commit point sound;
/// response bodies remain Hyper's streaming [`Incoming`] body.
pub type Body = Full<Bytes>;

type Connector = HttpsConnector<ProxyConnector>;
type PooledClient = Client<Connector, Body>;

/// Failure to receive an HTTP response from a provider.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum EgressError {
	/// Response headers did not arrive before the first-byte deadline.
	#[error("provider response headers timed out after {0:?}")]
	FirstByteTimeout(Duration),
	/// Hyper failed while dispatching the request or receiving its headers.
	///
	/// The underlying diagnostic is discarded because it may retain the
	/// credential-bearing request URI.
	#[error("provider HTTP transport failed")]
	Transport,
	/// Deferred query credential could not be represented in the outbound URI.
	#[error("provider query authentication could not be applied")]
	InvalidSensitiveQuery,
}

impl From<hyper_util::client::legacy::Error> for EgressError {
	fn from(_error: hyper_util::client::legacy::Error) -> Self {
		Self::Transport
	}
}

/// A Tower layer imposing a deadline on response headers.
#[derive(Clone, Copy, Debug)]
pub struct FirstByteTimeoutLayer {
	timeout: Duration,
}

impl FirstByteTimeoutLayer {
	/// Constructs a response-header timeout layer.
	#[must_use]
	pub const fn new(timeout: Duration) -> Self {
		Self { timeout }
	}
}

impl<S> Layer<S> for FirstByteTimeoutLayer {
	type Service = FirstByteTimeout<S>;

	fn layer(&self, inner: S) -> Self::Service {
		FirstByteTimeout { inner, timeout: self.timeout }
	}
}

/// A service whose request future fails unless response headers arrive in time.
#[derive(Clone, Debug)]
pub struct FirstByteTimeout<S> {
	inner:   S,
	timeout: Duration,
}

impl<S> FirstByteTimeout<S> {
	/// Returns the configured response-header timeout.
	#[must_use]
	pub const fn timeout(&self) -> Duration {
		self.timeout
	}
}

impl<S> Service<Request<Body>> for FirstByteTimeout<S>
where
	S: Service<Request<Body>, Response = Response<Incoming>>,
	S::Future: Send + 'static,
	EgressError: From<S::Error>,
{
	type Error = EgressError;
	type Response = Response<Incoming>;

	type Future = impl Future<Output = Result<Self::Response, Self::Error>> + Send + 'static;

	fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		self.inner.poll_ready(cx).map_err(EgressError::from)
	}

	fn call(&mut self, request: Request<Body>) -> Self::Future {
		let response = self.inner.call(request);
		let deadline = self.timeout;
		async move {
			match tokio::time::timeout(deadline, response).await {
				Ok(response) => response.map_err(EgressError::from),
				Err(_) => Err(EgressError::FirstByteTimeout(deadline)),
			}
		}
	}
}

/// A cloneable, pooled Hyper client with a response-header deadline.
///
/// “First byte” at this layer means that the response future has produced
/// headers. The first meaningful decoded event is validated by the typed stack
/// before its service future crosses the retry commit point. Idle and stream
/// watchdogs are separate stream-level concerns because they run after commit.
///
/// A deferred [`SensitiveQuery`] remains outside the URI through every generic
/// layer. This client consumes it after proxy preparation and immediately
/// before moving the final request into Hyper.
#[derive(Clone)]
pub struct EgressClient {
	inner: FirstByteTimeout<PooledClient>,
	proxy: ProxyConnector,
}

impl EgressClient {
	/// Constructs a pooled rustls client with the supplied response-header
	/// timeout.
	///
	/// Proxy policy is captured from the process environment at construction.
	#[must_use]
	pub fn new(first_byte_timeout: Duration) -> Self {
		Self::with_connector(first_byte_timeout, ProxyConnector::new(ProxyResolver::from_env()))
	}

	/// Constructs a client using an explicit provider-scoped proxy policy.
	#[must_use]
	pub fn for_provider(
		first_byte_timeout: Duration,
		resolver: ProxyResolver,
		provider: impl AsRef<str>,
	) -> Self {
		Self::with_connector(first_byte_timeout, ProxyConnector::for_provider(resolver, provider))
	}

	fn with_connector(first_byte_timeout: Duration, proxy: ProxyConnector) -> Self {
		ensure_crypto_provider();
		let connector = HttpsConnectorBuilder::new()
			.with_webpki_roots()
			.https_or_http()
			.enable_http1()
			.enable_http2()
			.wrap_connector(proxy.clone());
		let pooled = Client::builder(TokioExecutor::new()).build(connector);
		let inner = FirstByteTimeoutLayer::new(first_byte_timeout).layer(pooled);
		Self { inner, proxy }
	}

	/// Returns the configured response-header timeout.
	#[must_use]
	pub const fn first_byte_timeout(&self) -> Duration {
		self.inner.timeout()
	}
}

impl Service<Request<Body>> for EgressClient {
	type Error = EgressError;
	type Future = Either<
		Ready<Result<Self::Response, Self::Error>>,
		<FirstByteTimeout<PooledClient> as Service<Request<Body>>>::Future,
	>;
	type Response = Response<Incoming>;

	fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		self.inner.poll_ready(cx)
	}

	fn call(&mut self, mut request: Request<Body>) -> Self::Future {
		request
			.headers_mut()
			.entry(header::USER_AGENT)
			.or_insert(HeaderValue::from_static(omp_core::USER_AGENT));
		self.proxy.inject_proxy_auth(&mut request);
		if apply_sensitive_query(&mut request).is_err() {
			return Either::Left(ready(Err(EgressError::InvalidSensitiveQuery)));
		}
		Either::Right(self.inner.call(request))
	}
}

fn apply_sensitive_query(request: &mut Request<Body>) -> Result<(), ()> {
	let Some(query) = request.extensions_mut().remove::<SensitiveQuery>() else {
		return Ok(());
	};
	query.apply(request.uri_mut())
}

#[cfg(test)]
mod tests {
	use std::time::Duration;

	use bytes::Bytes;
	use http::Request;
	use http_body_util::Full;
	use tokio::io::{AsyncReadExt, AsyncWriteExt};
	use tower::ServiceExt as _;

	use super::{EgressClient, EgressError, SensitiveQuery, apply_sensitive_query};

	#[test]
	fn egress_failure_classes_have_opaque_observable_formatting() {
		let transport = EgressError::Transport;
		let timeout = EgressError::FirstByteTimeout(Duration::from_secs(3));
		assert_eq!(transport.to_string(), "provider HTTP transport failed");
		assert_eq!(timeout.to_string(), "provider response headers timed out after 3s");
		assert_eq!(format!("{transport:?}"), "Transport");
		assert_eq!(
			EgressError::InvalidSensitiveQuery.to_string(),
			"provider query authentication could not be applied"
		);
	}

	#[test]
	fn sensitive_query_is_invisible_until_final_dispatch() {
		const CANARY: &str = "canary-query-api-key";
		let mut request = Request::builder()
			.uri("https://provider.test/v1?existing=yes")
			.body(Full::new(Bytes::new()))
			.expect("request");
		request
			.extensions_mut()
			.insert(SensitiveQuery::new("key", CANARY.as_bytes()));

		let request_debug = format!("{request:?}");
		let extension_debug =
			format!("{:?}", request.extensions().get::<SensitiveQuery>().expect("query"));
		assert!(!request_debug.contains(CANARY));
		assert!(!extension_debug.contains(CANARY));
		assert_eq!(request.uri(), "https://provider.test/v1?existing=yes");

		apply_sensitive_query(&mut request).expect("late query placement");
		assert_eq!(request.uri(), "https://provider.test/v1?existing=yes&key=canary-query-api-key");
		assert!(request.extensions().get::<SensitiveQuery>().is_none());
	}

	#[tokio::test]
	async fn sensitive_query_reaches_only_the_real_wire_request() {
		const CANARY: &str = "canary-final-wire-query";
		let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
			.await
			.expect("listener");
		let address = listener.local_addr().expect("address");
		let server = tokio::spawn(async move {
			let (mut stream, _) = listener.accept().await.expect("accept");
			let mut bytes = Vec::new();
			loop {
				let mut chunk = [0_u8; 1024];
				let read = stream.read(&mut chunk).await.expect("read request");
				bytes.extend_from_slice(&chunk[..read]);
				if read == 0 || bytes.windows(4).any(|window| window == b"\r\n\r\n") {
					break;
				}
			}
			stream
				.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
				.await
				.expect("write response");
			String::from_utf8(bytes).expect("HTTP request")
		});
		let mut request = Request::builder()
			.uri(format!("http://{address}/v1"))
			.body(Full::new(Bytes::new()))
			.expect("request");
		request
			.extensions_mut()
			.insert(SensitiveQuery::new("api_key", CANARY.as_bytes()));

		let response = EgressClient::new(Duration::from_secs(2))
			.oneshot(request)
			.await
			.expect("egress response");
		assert_eq!(response.status(), http::StatusCode::OK);
		let wire = server.await.expect("server");
		assert!(wire.starts_with(&format!("GET /v1?api_key={CANARY} HTTP/1.1\r\n")));
	}
}
