//! Per-attempt credential selection, redemption, and refresh.
//!
//! This module deliberately defines a narrow credential-source interface rather
//! than depending on the broker crate. Secret bytes stay behind that interface:
//! redemption mutates a request in place and never returns a token.

use std::{
	error::Error as StdError,
	future::Future,
	sync::{
		Arc,
		atomic::{AtomicU64, Ordering},
	},
	task::{Context, Poll},
	time::SystemTime,
};

use http::{HeaderMap, Request, Response, StatusCode, Uri, uri::PathAndQuery};
use omp_core::Str;
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use tokio::sync::watch;
use tower::{Layer, Service, ServiceExt};
use zeroize::Zeroizing;

use crate::{client::Body, limits::EgressKey};

/// Opaque query credential deferred until the final HTTP dispatch.
///
/// Generic middleware, retries, telemetry, and request formatting see only this
/// redacted extension while the request URI remains credential-free. The final
/// egress client consumes the extension immediately before moving the request
/// into Hyper. Its owned bytes are zeroized on drop.
///
/// Secret exposure is deliberately absent:
///
/// ```compile_fail
/// let query = omp_llm_egress::auth_inject::SensitiveQuery::new("key", b"secret");
/// let _secret = query.value();
/// ```
///
/// ```compile_fail
/// let query = omp_llm_egress::auth_inject::SensitiveQuery::new("key", b"secret");
/// let _display = query.to_string();
/// ```
#[derive(Clone)]
pub struct SensitiveQuery {
	parameter: Str,
	value:     Zeroizing<Vec<u8>>,
}

impl SensitiveQuery {
	/// Seals one query parameter for late insertion by the final egress client.
	#[must_use]
	pub fn new(parameter: impl AsRef<str>, value: &[u8]) -> Self {
		Self {
			parameter: Str::new(parameter.as_ref()),
			value:     Zeroizing::new(value.to_vec()),
		}
	}

	pub(crate) fn apply(self, uri: &mut Uri) -> Result<(), ()> {
		let path_and_query = uri.path_and_query().map_or("/", PathAndQuery::as_str);
		let mut placed = Zeroizing::new(Vec::with_capacity(
			path_and_query.len() + self.parameter.len() + self.value.len() + 2,
		));
		placed.extend_from_slice(path_and_query.as_bytes());
		placed.push(if uri.query().is_some() { b'&' } else { b'?' });
		for segment in url::form_urlencoded::byte_serialize(self.parameter.as_bytes()) {
			placed.extend_from_slice(segment.as_bytes());
		}
		placed.push(b'=');
		for segment in url::form_urlencoded::byte_serialize(self.value.as_slice()) {
			placed.extend_from_slice(segment.as_bytes());
		}

		let path_and_query = std::str::from_utf8(placed.as_slice())
			.map_err(|_| ())?
			.parse::<PathAndQuery>()
			.map_err(|_| ())?;
		let mut parts = uri.clone().into_parts();
		parts.path_and_query = Some(path_and_query);
		*uri = Uri::from_parts(parts).map_err(|_| ())?;
		Ok(())
	}
}

impl std::fmt::Debug for SensitiveQuery {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		formatter
			.debug_struct("SensitiveQuery")
			.field("parameter", &self.parameter)
			.field("value", &"[redacted]")
			.finish()
	}
}

/// Request extension identifying the provider whose credential should be used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthContext {
	provider: Str,
}

impl AuthContext {
	/// Constructs authentication context for `provider`.
	#[must_use]
	pub fn new(provider: impl AsRef<str>) -> Self {
		Self { provider: Str::new(provider.as_ref()) }
	}

	/// Returns the provider identifier.
	#[must_use]
	pub fn provider(&self) -> &str {
		&self.provider
	}
}

/// Non-secret AWS Signature Version 4 parameters attached by a provider
/// transport before credential redemption.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AwsSigV4Context {
	/// AWS signing service, such as `bedrock`.
	pub service:   Str,
	/// AWS region used in the credential scope.
	pub region:    Str,
	/// Injectable signing time used for deterministic request signatures.
	pub signed_at: SystemTime,
}

/// An opaque, non-secret claim on one generation of a stored credential.
///
/// The generation is part of redemption: a [`CredentialSource`] must reject an
/// `apply` call if the stored generation has changed since this value was
/// issued. This prevents an in-flight request from applying a rotated-away
/// secret.
///
/// Secret exposure is not part of the lease API:
///
/// ```compile_fail
/// let lease = omp_llm_egress::auth_inject::CredentialLease::new("provider", 7, 3);
/// let _provider_token = lease.secret();
/// ```
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct CredentialLease {
	provider:      Str,
	credential_id: u64,
	generation:    u64,
}

impl CredentialLease {
	/// Constructs a lease containing identity metadata but no secret bytes.
	#[must_use]
	pub fn new(provider: impl AsRef<str>, credential_id: u64, generation: u64) -> Self {
		Self { provider: Str::new(provider.as_ref()), credential_id, generation }
	}

	/// Returns the provider identifier.
	#[must_use]
	pub fn provider(&self) -> &str {
		&self.provider
	}

	/// Returns the stable, non-secret numeric credential identifier.
	#[must_use]
	pub const fn credential_id(&self) -> u64 {
		self.credential_id
	}

	/// Returns the credential generation claimed by this lease.
	#[must_use]
	pub const fn generation(&self) -> u64 {
		self.generation
	}

	/// Returns the admission-control identity for this credential.
	#[must_use]
	pub fn egress_key(&self) -> EgressKey {
		EgressKey::from_str(self.provider.clone(), self.credential_id)
	}
}

/// Broker-facing credential operations needed by the egress layer.
///
/// `apply` is the atomic redemption operation: implementations must compare the
/// lease generation with the current stored generation before mutating headers.
/// Secret material must never be returned from any method.
pub trait CredentialSource: Clone + Send + Sync + 'static {
	/// Error returned by credential lookup, redemption, or refresh.
	type Error: StdError + Send + Sync + 'static;

	/// Leases the current credential for a provider, if one is available.
	fn lease(&self, provider: &str) -> Result<Option<CredentialLease>, Self::Error>;

	/// Atomically validates `lease` and applies its secret to `request`.
	fn apply(&self, lease: &CredentialLease, request: &mut Request<Body>)
	-> Result<(), Self::Error>;

	/// Atomically validates `lease` and mutates an outbound WebSocket handshake
	/// header map without returning credential bytes.
	///
	/// The default keeps the secret-bearing transfer inside this sealed API by
	/// applying the canonical request mutation to a temporary request and moving
	/// its headers into the actual handshake. Header-based credentials are the
	/// only supported WebSocket placement.
	fn apply_headers(
		&self,
		lease: &CredentialLease,
		headers: &mut HeaderMap,
	) -> Result<(), Self::Error> {
		let mut request = Request::new(Body::new(bytes::Bytes::new()));
		*request.headers_mut() = std::mem::take(headers);
		self.apply(lease, &mut request)?;
		*headers = request.into_parts().0.headers;
		Ok(())
	}

	/// Refresh is I/O-dominated, but its concrete future remains unboxed so
	/// implementations which can avoid allocation do so.
	fn refresh(
		&self,
		lease: CredentialLease,
	) -> impl Future<Output = Result<CredentialLease, Self::Error>> + Send + 'static;
}

/// Non-secret authentication category for a selected credential.
///
/// This is routing policy only. It never contains credential bytes or claims.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CredentialAuthKind {
	/// Provider API key.
	#[default]
	ApiKey,
	/// Broker-managed OAuth access token.
	OAuth,
	/// AWS access key used by SigV4.
	Aws,
	/// Google Application Default Credentials.
	GoogleAdc,
}

/// Validated, non-secret routing metadata associated with one credential
/// generation.
#[derive(Clone, Eq, PartialEq)]
pub struct CredentialMetadata {
	/// Non-secret account label stored with the credential.
	pub identity:        Str,
	/// Non-secret authentication category used to select provider wire policy.
	pub auth_kind:       CredentialAuthKind,
	/// Provider account identifier when known.
	pub account_id:      Option<Str>,
	/// Cloud project identifier when known.
	pub project_id:      Option<Str>,
	/// Provider organization identifier when known.
	pub organization_id: Option<Str>,
}

impl std::fmt::Debug for CredentialMetadata {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		formatter
			.debug_struct("CredentialMetadata")
			.field("auth_kind", &self.auth_kind)
			.field("identity", &"[redacted]")
			.field("account_id", &self.account_id.as_ref().map(|_| "[present]"))
			.field("project_id", &self.project_id.as_ref().map(|_| "[present]"))
			.field("organization_id", &self.organization_id.as_ref().map(|_| "[present]"))
			.finish()
	}
}

/// Non-secret metadata lookup paired with canonical credential redemption.
pub trait CredentialMetadataSource: CredentialSource {
	/// Atomically validates `lease` and returns its typed metadata projection.
	fn metadata(&self, lease: &CredentialLease) -> Result<CredentialMetadata, Self::Error>;
}
/// Failure from authentication injection or the wrapped egress service.
pub enum AuthInjectError<C, S> {
	/// Credential lookup, redemption, or refresh failed.
	Credential(Arc<C>),
	/// The wrapped service failed.
	Service(S),
}

impl<C, S> std::fmt::Debug for AuthInjectError<C, S> {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		std::fmt::Display::fmt(self, formatter)
	}
}

impl<C, S> std::fmt::Display for AuthInjectError<C, S> {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			Self::Credential(_) => formatter.write_str("credential operation failed"),
			Self::Service(_) => formatter.write_str("egress service failed"),
		}
	}
}

impl<C, S> StdError for AuthInjectError<C, S> {}

/// Layer applying credentials independently on every network attempt.
#[derive(Clone)]
pub struct AuthInjectLayer<C: CredentialSource> {
	shared: Arc<Shared<C>>,
}

impl<C> AuthInjectLayer<C>
where
	C: CredentialSource,
{
	/// Constructs an authentication layer backed by `source`.
	#[must_use]
	pub fn new(source: C) -> Self {
		Self {
			shared: Arc::new(Shared {
				source,
				flights: Arc::new(Mutex::new(FxHashMap::default())),
				next_flight: AtomicU64::new(1),
			}),
		}
	}
}

impl<C, S> Layer<S> for AuthInjectLayer<C>
where
	C: CredentialSource,
{
	type Service = AuthInject<C, S>;

	fn layer(&self, inner: S) -> Self::Service {
		AuthInject { inner, shared: Arc::clone(&self.shared) }
	}
}

/// Service redeeming a selected or provider-leased credential with one
/// refresh-on-401 retry.
#[derive(Clone)]
pub struct AuthInject<C: CredentialSource, S> {
	inner:  S,
	shared: Arc<Shared<C>>,
}

impl<C, S, R> Service<Request<Body>> for AuthInject<C, S>
where
	C: CredentialSource,
	S: Service<Request<Body>, Response = Response<R>> + Clone + Send + 'static,
	S::Future: Send + 'static,
	S::Error: Send + 'static,
	R: Send + 'static,
{
	type Error = AuthInjectError<C::Error, S::Error>;
	type Response = Response<R>;

	type Future = impl Future<Output = Result<Self::Response, Self::Error>> + Send + 'static;

	fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		self.inner.poll_ready(cx).map_err(AuthInjectError::Service)
	}

	fn call(&mut self, mut request: Request<Body>) -> Self::Future {
		let replacement = self.inner.clone();
		let mut inner = std::mem::replace(&mut self.inner, replacement);
		let shared = Arc::clone(&self.shared);
		async move {
			let selected = request.extensions().get::<CredentialLease>().cloned();
			let context = request.extensions().get::<AuthContext>().cloned();
			let lease = if let Some(selected) = selected {
				selected
			} else {
				let Some(context) = context.as_ref() else {
					return inner.call(request).await.map_err(AuthInjectError::Service);
				};
				let Some(lease) = shared
					.source
					.lease(context.provider())
					.map_err(|error| AuthInjectError::Credential(Arc::new(error)))?
				else {
					return inner.call(request).await.map_err(AuthInjectError::Service);
				};
				lease
			};
			let template = RequestTemplate::from_request(&request, context);
			apply_lease(&shared.source, &lease, &mut request)
				.map_err(|error| AuthInjectError::Credential(Arc::new(error)))?;
			let response = inner
				.call(request)
				.await
				.map_err(AuthInjectError::Service)?;
			if response.status() != StatusCode::UNAUTHORIZED {
				return Ok(response);
			}

			let refreshed = shared
				.refresh(lease)
				.await
				.map_err(AuthInjectError::Credential)?;
			let mut retry = template.build();
			apply_lease(&shared.source, &refreshed, &mut retry)
				.map_err(|error| AuthInjectError::Credential(Arc::new(error)))?;
			inner
				.ready()
				.await
				.map_err(AuthInjectError::Service)?
				.call(retry)
				.await
				.map_err(AuthInjectError::Service)
		}
	}
}

fn apply_lease<C: CredentialSource>(
	source: &C,
	lease: &CredentialLease,
	request: &mut Request<Body>,
) -> Result<(), C::Error> {
	source.apply(lease, request)?;
	request.extensions_mut().insert(lease.clone());
	request.extensions_mut().insert(lease.egress_key());
	Ok(())
}

type RefreshResult<E> = Result<CredentialLease, Arc<E>>;

struct Flight<E> {
	id:       u64,
	receiver: watch::Receiver<Option<RefreshResult<E>>>,
}

struct Shared<C: CredentialSource> {
	source:      C,
	flights:     Arc<Mutex<FxHashMap<CredentialLease, Flight<C::Error>>>>,
	next_flight: AtomicU64,
}

impl<C: CredentialSource> Shared<C> {
	async fn refresh(&self, lease: CredentialLease) -> RefreshResult<C::Error> {
		let key = lease.clone();
		let (mut receiver, start) = {
			let mut flights = self.flights.lock();
			if let Some(flight) = flights.get(&key) {
				(flight.receiver.clone(), None)
			} else {
				let id = self.next_flight.fetch_add(1, Ordering::Relaxed);
				let (sender, receiver) = watch::channel(None);
				flights.insert(key.clone(), Flight { id, receiver: receiver.clone() });
				(receiver, Some((id, sender)))
			}
		};

		if let Some((id, sender)) = start {
			let future = self.source.refresh(lease);
			let flights = Arc::clone(&self.flights);
			// The spawned task, rather than any caller, owns the refresh future.
			// Dropping a caller therefore only drops its receiver.
			tokio::spawn(async move {
				let result = future.await.map_err(Arc::new);
				let failed = result.is_err();
				sender.send_replace(Some(result));
				// Cache a successful refresh for the expired generation. This
				// closes the completion/removal race for late concurrent 401s.
				// Failures are evicted so a later attempt can try again.
				if failed {
					let mut flights = flights.lock();
					if flights.get(&key).is_some_and(|flight| flight.id == id) {
						flights.remove(&key);
					}
				}
			});
		}

		loop {
			let result = receiver.borrow().clone();
			if let Some(result) = result {
				return result;
			}
			let _ = receiver.changed().await;
		}
	}
}

#[derive(Clone)]
struct RequestTemplate {
	method:              http::Method,
	uri:                 http::Uri,
	version:             http::Version,
	headers:             http::HeaderMap,
	body:                Body,
	context:             Option<AuthContext>,
	aws:                 Option<AwsSigV4Context>,
	credential_metadata: Option<CredentialMetadata>,
}

impl RequestTemplate {
	fn from_request(request: &Request<Body>, context: Option<AuthContext>) -> Self {
		// `Body` is `Full<Bytes>`: cloning it and every `HeaderValue` is O(1).
		// Only the HeaderMap buckets are rebuilt for a possible 401 replay.
		Self {
			method: request.method().clone(),
			uri: request.uri().clone(),
			version: request.version(),
			headers: request.headers().clone(),
			body: request.body().clone(),
			context,
			aws: request.extensions().get::<AwsSigV4Context>().cloned(),
			credential_metadata: request.extensions().get::<CredentialMetadata>().cloned(),
		}
	}

	fn build(&self) -> Request<Body> {
		let mut request = Request::new(self.body.clone());
		*request.method_mut() = self.method.clone();
		*request.uri_mut() = self.uri.clone();
		*request.version_mut() = self.version;
		*request.headers_mut() = self.headers.clone();
		if let Some(context) = &self.context {
			request.extensions_mut().insert(context.clone());
		}
		if let Some(aws) = &self.aws {
			request.extensions_mut().insert(aws.clone());
		}
		if let Some(metadata) = &self.credential_metadata {
			request.extensions_mut().insert(metadata.clone());
		}
		request
	}
}

#[cfg(test)]
mod tests {
	use std::sync::atomic::{AtomicBool, AtomicUsize};

	use bytes::Bytes;
	use http_body_util::Full;
	use tokio::sync::Notify;
	use tower::service_fn;

	use super::*;

	#[derive(Debug, thiserror::Error)]
	#[error("stale credential generation")]
	struct TestError;

	#[derive(Clone)]
	struct Source {
		generation: Arc<AtomicU64>,
		leases:     Arc<AtomicUsize>,
		refreshes:  Arc<AtomicUsize>,
		gate:       Arc<Notify>,
		wait:       Arc<AtomicBool>,
	}

	impl CredentialSource for Source {
		type Error = TestError;

		fn lease(&self, provider: &str) -> Result<Option<CredentialLease>, Self::Error> {
			self.leases.fetch_add(1, Ordering::SeqCst);
			Ok(Some(CredentialLease::new(provider, 7, self.generation.load(Ordering::SeqCst))))
		}

		fn apply(
			&self,
			lease: &CredentialLease,
			_request: &mut Request<Body>,
		) -> Result<(), Self::Error> {
			(lease.generation() == self.generation.load(Ordering::SeqCst))
				.then_some(())
				.ok_or(TestError)
		}

		fn refresh(
			&self,
			lease: CredentialLease,
		) -> impl Future<Output = Result<CredentialLease, Self::Error>> + Send + 'static {
			self.refreshes.fetch_add(1, Ordering::SeqCst);
			let gate = Arc::clone(&self.gate);
			let wait = Arc::clone(&self.wait);
			async move {
				if wait.load(Ordering::SeqCst) {
					gate.notified().await;
				}
				Ok(lease)
			}
		}
	}

	fn source(wait: bool) -> Source {
		Source {
			generation: Arc::new(AtomicU64::new(1)),
			leases:     Arc::new(AtomicUsize::new(0)),
			refreshes:  Arc::new(AtomicUsize::new(0)),
			gate:       Arc::new(Notify::new()),
			wait:       Arc::new(AtomicBool::new(wait)),
		}
	}

	fn request() -> Request<Body> {
		let mut request = Request::new(Full::new(Bytes::new()));
		request
			.extensions_mut()
			.insert(AuthContext::new("provider"));
		request
	}

	#[tokio::test]
	async fn selected_extension_and_metadata_survive_refresh_retry_without_reselection() {
		let source = source(false);
		let leases = Arc::clone(&source.leases);
		let attempts = Arc::new(AtomicUsize::new(0));
		let observed = Arc::clone(&attempts);
		let selected = CredentialLease::new("provider", 41, 1);
		let metadata = CredentialMetadata {
			auth_kind:       CredentialAuthKind::ApiKey,
			identity:        "developer@example.com".into(),
			account_id:      Some("account-41".into()),
			project_id:      Some("project-41".into()),
			organization_id: None,
		};
		let expected_lease = selected.clone();
		let expected_metadata = metadata.clone();
		let mut service =
			AuthInjectLayer::new(source).layer(service_fn(move |request: Request<Body>| {
				assert_eq!(request.extensions().get::<CredentialLease>(), Some(&expected_lease));
				assert_eq!(request.extensions().get::<EgressKey>(), Some(&expected_lease.egress_key()));
				assert_eq!(request.extensions().get::<CredentialMetadata>(), Some(&expected_metadata));
				let status = if observed.fetch_add(1, Ordering::SeqCst) == 0 {
					StatusCode::UNAUTHORIZED
				} else {
					StatusCode::OK
				};
				async move {
					Ok::<_, std::convert::Infallible>(
						Response::builder().status(status).body(()).unwrap(),
					)
				}
			}));
		let mut request = request();
		request.extensions_mut().insert(selected);
		request.extensions_mut().insert(metadata);

		service.call(request).await.unwrap();

		assert_eq!(attempts.load(Ordering::SeqCst), 2);
		assert_eq!(leases.load(Ordering::SeqCst), 0);
	}

	#[tokio::test]
	async fn concurrent_unauthorized_responses_share_one_refresh() {
		let source = source(true);
		let refreshes = Arc::clone(&source.refreshes);
		let gate = Arc::clone(&source.gate);
		let service = AuthInjectLayer::new(source).layer(service_fn(|_request| async {
			Ok::<_, std::convert::Infallible>(
				Response::builder()
					.status(StatusCode::UNAUTHORIZED)
					.body(())
					.unwrap(),
			)
		}));
		let calls = (0..16).map(move |_| {
			let mut service = service.clone();
			async move { service.call(request()).await.unwrap() }
		});
		let joined = tokio::spawn(async move { futures::future::join_all(calls).await });
		while refreshes.load(Ordering::SeqCst) == 0 {
			tokio::task::yield_now().await;
		}
		tokio::task::yield_now().await;
		gate.notify_one();
		joined.await.unwrap();
		assert_eq!(refreshes.load(Ordering::SeqCst), 1);
	}

	#[tokio::test]
	async fn cancelling_waiter_does_not_cancel_shared_refresh() {
		let source = source(true);
		let refreshes = Arc::clone(&source.refreshes);
		let gate = Arc::clone(&source.gate);
		let service = AuthInjectLayer::new(source).layer(service_fn(|_request| async {
			Ok::<_, std::convert::Infallible>(
				Response::builder()
					.status(StatusCode::UNAUTHORIZED)
					.body(())
					.unwrap(),
			)
		}));
		let mut cancelled_service = service.clone();
		let cancelled = tokio::spawn(async move { cancelled_service.call(request()).await });
		while refreshes.load(Ordering::SeqCst) == 0 {
			tokio::task::yield_now().await;
		}
		cancelled.abort();
		let mut surviving_service = service;
		let surviving = tokio::spawn(async move { surviving_service.call(request()).await });
		tokio::task::yield_now().await;
		gate.notify_one();
		surviving.await.unwrap().unwrap();
		assert_eq!(refreshes.load(Ordering::SeqCst), 1);
	}

	#[tokio::test]
	async fn stale_generation_is_rejected_at_redemption() {
		let source = source(true);
		let generation = Arc::clone(&source.generation);
		let refreshes = Arc::clone(&source.refreshes);
		let gate = Arc::clone(&source.gate);
		let mut service = AuthInjectLayer::new(source).layer(service_fn(|_request| async {
			Ok::<_, std::convert::Infallible>(
				Response::builder()
					.status(StatusCode::UNAUTHORIZED)
					.body(())
					.unwrap(),
			)
		}));
		let task = tokio::spawn(async move { service.call(request()).await });
		while refreshes.load(Ordering::SeqCst) == 0 {
			tokio::task::yield_now().await;
		}
		generation.store(2, Ordering::SeqCst);
		gate.notify_one();
		assert!(matches!(task.await.unwrap(), Err(AuthInjectError::Credential(_))));
	}
	#[test]
	fn retry_template_preserves_aws_signing_context() {
		let mut request = request();
		let aws = AwsSigV4Context {
			service:   "bedrock".into(),
			region:    "us-east-1".into(),
			signed_at: SystemTime::UNIX_EPOCH,
		};
		request.extensions_mut().insert(aws.clone());
		let replay =
			RequestTemplate::from_request(&request, Some(AuthContext::new("bedrock"))).build();
		assert_eq!(replay.extensions().get::<AwsSigV4Context>(), Some(&aws));
	}
	#[test]
	fn credential_metadata_debug_redacts_identifiers() {
		let metadata = CredentialMetadata {
			auth_kind:       CredentialAuthKind::OAuth,
			identity:        "person@example.test".into(),
			account_id:      Some("account-secret-shaped".into()),
			project_id:      Some("project-7".into()),
			organization_id: Some("organization-9".into()),
		};
		let debug = format!("{metadata:?}");
		for value in [
			metadata.identity.as_str(),
			metadata.account_id.as_deref().unwrap(),
			metadata.project_id.as_deref().unwrap(),
			metadata.organization_id.as_deref().unwrap(),
		] {
			assert!(!debug.contains(value));
		}
	}

	#[test]
	fn auth_error_surfaces_never_format_or_chain_underlying_secrets() {
		const CANARY: &str = "canary-provider-token-must-not-escape";
		let credential = AuthInjectError::<std::io::Error, std::io::Error>::Credential(Arc::new(
			std::io::Error::other(CANARY),
		));
		let service =
			AuthInjectError::<std::io::Error, std::io::Error>::Service(std::io::Error::other(CANARY));

		for error in [&credential, &service] {
			assert!(!error.to_string().contains(CANARY));
			assert!(!format!("{error:?}").contains(CANARY));
			assert!(std::error::Error::source(error).is_none());
		}
		assert_eq!(credential.to_string(), "credential operation failed");
		assert_eq!(service.to_string(), "egress service failed");
	}
}
