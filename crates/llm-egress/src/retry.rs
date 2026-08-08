//! Retry policy for the replayable side of the egress commit point.
//!
//! The service request and response wrappers are intentionally asymmetric:
//! only [`Replayable`] requests can enter this layer and every successful call
//! produces [`Committed`]. Failures after that transition belong inside the
//! committed response stream and cannot inhabit the service error channel.

use std::{
	future::Future,
	sync::{
		Arc,
		atomic::{AtomicU64, Ordering},
	},
	task::{Context, Poll},
	time::{Duration, SystemTime, UNIX_EPOCH},
};

use http::Request;
use omp_llm_error::{Classification, extract::retry_hint_from_headers};
use thiserror::Error;
use tower::{Layer, Service, ServiceExt};

use crate::{
	auth_inject::{AuthContext, CredentialLease, CredentialMetadata, SensitiveQuery},
	client::Body,
	limits::EgressKey,
};

/// A factory for requests whose buffered bodies may still be replayed safely.
///
/// The factory is the only way retries obtain a request. Consequently a
/// non-cloneable `http::Request<Body>` can still be rebuilt for every attempt
/// without putting post-commit state in this type.
pub struct Replayable<T> {
	replay: Box<dyn Fn() -> T + Send + 'static>,
}

impl<T> Replayable<T> {
	/// Marks a cloneable request as being before the first-event commit point.
	#[must_use]
	pub fn new(request: T) -> Self
	where
		T: Clone + Send + 'static,
	{
		Self::from_factory(move || request.clone())
	}

	/// Constructs a replayable request from an exact-replay factory.
	#[must_use]
	pub fn from_factory(replay: impl Fn() -> T + Send + 'static) -> Self {
		Self { replay: Box::new(replay) }
	}

	/// Produces one request attempt.
	#[must_use]
	pub fn into_inner(self) -> T {
		(self.replay)()
	}

	fn replay(&self) -> T {
		(self.replay)()
	}
}

impl Replayable<Request<Body>> {
	/// Buffers an HTTP request for exact replay before credential injection.
	///
	/// HTTP protocol fields, headers, body bytes, [`AuthContext`], the selected
	/// [`CredentialLease`], its [`CredentialMetadata`], and [`EgressKey`] are
	/// retained. An already-sealed [`SensitiveQuery`] is replayed only as its
	/// redacted, zeroizing request extension; the buffered URI remains
	/// credential-free. Authentication still runs inside each attempt and
	/// overwrites any credential header.
	#[must_use]
	pub fn buffered(request: Request<Body>) -> Self {
		let template = BufferedRequest::new(&request);
		Self::from_factory(move || template.build())
	}
}

/// A response that crossed the first-meaningful-event commit point.
///
/// Stream failures after this point are values inside `T`; they are not Tower
/// service errors and therefore cannot be fed back into [`Retry`].
#[derive(Clone, Debug)]
pub struct Committed<T>(T);

impl<T> Committed<T> {
	/// Marks a response as validated through its first meaningful event.
	#[must_use]
	pub const fn new(inner: T) -> Self {
		Self(inner)
	}

	/// Returns a shared reference to the committed response.
	#[must_use]
	pub const fn get_ref(&self) -> &T {
		&self.0
	}

	/// Consumes the marker and returns the response or response stream.
	#[must_use]
	pub fn into_inner(self) -> T {
		self.0
	}
}

/// A classified failure known to have occurred before commit.
#[derive(Debug, Error)]
#[error("pre-commit egress attempt failed: {error}")]
pub struct PreCommitFailure<E> {
	error:          E,
	classification: Classification,
	retry_after:    Option<Duration>,
}

impl<E> PreCommitFailure<E> {
	/// Constructs a pre-commit failure using retry timing from its
	/// classification.
	#[must_use]
	pub const fn new(error: E, classification: Classification) -> Self {
		Self { error, classification, retry_after: None }
	}

	/// Sets a provider-mandated minimum retry delay.
	#[must_use]
	pub const fn with_retry_after(mut self, retry_after: Duration) -> Self {
		self.retry_after = Some(retry_after);
		self
	}

	/// Returns the classified provider-error taxonomy.
	#[must_use]
	pub const fn classification(&self) -> &Classification {
		&self.classification
	}

	/// Returns the underlying error.
	#[must_use]
	pub const fn source_error(&self) -> &E {
		&self.error
	}

	/// Consumes the failure and returns its underlying error.
	#[must_use]
	pub fn into_source(self) -> E {
		self.error
	}

	fn retry_delay(&self) -> Option<Duration> {
		self.retry_after.or_else(|| {
			self
				.classification
				.suggested_backoff_ms()
				.map(|(minimum, _)| Duration::from_millis(minimum))
		})
	}
}

/// Jittered exponential retry settings.
#[derive(Clone, Copy, Debug)]
pub struct RetryConfig {
	/// Delay used for the first retry.
	pub base_delay:  Duration,
	/// Upper bound applied before a larger provider `Retry-After` minimum.
	pub max_delay:   Duration,
	/// Maximum number of replays after the initial attempt.
	pub max_retries: u32,
	/// Symmetric jitter fraction in the inclusive range `0.0..=1.0`.
	pub jitter:      f64,
}

impl Default for RetryConfig {
	fn default() -> Self {
		Self {
			base_delay:  Duration::from_millis(250),
			max_delay:   Duration::from_secs(30),
			max_retries: 3,
			jitter:      0.2,
		}
	}
}

impl RetryConfig {
	/// Computes a deterministic jittered delay from a unit sample.
	///
	/// `retry` is zero for the first retry. `sample` is clamped to `0.0..=1.0`.
	#[must_use]
	pub fn backoff_for(self, retry: u32, sample: f64) -> Duration {
		let factor = 2_u32.checked_pow(retry).unwrap_or(u32::MAX);
		let exponential = self.base_delay.saturating_mul(factor).min(self.max_delay);
		let jitter = self.jitter.clamp(0.0, 1.0);
		let sample = sample.clamp(0.0, 1.0);
		let multiplier = (2.0 * jitter).mul_add(sample, 1.0 - jitter);
		Duration::from_secs_f64((exponential.as_secs_f64() * multiplier).max(0.0))
	}

	/// Computes the effective delay, treating a provider hint as a minimum.
	#[must_use]
	pub fn delay_for(self, retry: u32, sample: f64, provider_minimum: Option<Duration>) -> Duration {
		let backoff = self.backoff_for(retry, sample);
		provider_minimum.map_or(backoff, |minimum| minimum.max(backoff))
	}
}

/// Parses a `Retry-After` value in delta-seconds or IMF-fixdate form.
///
/// The parser is supplied by `omp-llm-error`, keeping header interpretation
/// consistent with the canonical [`Classification`] taxonomy.
#[must_use]
pub fn parse_retry_after(value: &str, now: SystemTime) -> Option<Duration> {
	let now_ms = u64::try_from(now.duration_since(UNIX_EPOCH).ok()?.as_millis()).ok()?;
	let headers = [("retry-after", value)];
	retry_hint_from_headers(&headers, now_ms).map(|hint| Duration::from_millis(hint.delay_ms))
}

/// Layer retrying only classified failures from replayable attempts.
#[derive(Clone, Debug)]
pub struct RetryLayer {
	config: RetryConfig,
	random: Arc<AtomicU64>,
}

impl RetryLayer {
	/// Constructs a retry layer with a process-local jitter seed.
	#[must_use]
	pub fn new(config: RetryConfig) -> Self {
		let seed = SystemTime::now()
			.duration_since(UNIX_EPOCH)
			.map_or(0x9e37_79b9_7f4a_7c15, |duration| {
				duration.as_secs() ^ u64::from(duration.subsec_nanos()).rotate_left(32)
			});
		Self::with_seed(config, seed)
	}

	/// Constructs a retry layer with a deterministic jitter seed.
	#[must_use]
	pub fn with_seed(config: RetryConfig, seed: u64) -> Self {
		Self { config, random: Arc::new(AtomicU64::new(seed.max(1))) }
	}
}

impl<S> Layer<S> for RetryLayer {
	type Service = Retry<S>;

	fn layer(&self, inner: S) -> Self::Service {
		Retry { inner, config: self.config, random: Arc::clone(&self.random) }
	}
}

/// Service whose error channel represents only failures before commit.
#[derive(Clone, Debug)]
pub struct Retry<S> {
	inner:  S,
	config: RetryConfig,
	random: Arc<AtomicU64>,
}

impl<S, T, R, E> Service<Replayable<T>> for Retry<S>
where
	S: Service<T, Response = Committed<R>, Error = PreCommitFailure<E>> + Clone + Send + 'static,
	S::Future: Send + 'static,
	T: Send + 'static,
	R: Send + 'static,
	E: std::fmt::Display + Send + 'static,
{
	type Error = PreCommitFailure<E>;
	type Response = Committed<R>;

	type Future = impl Future<Output = Result<Self::Response, Self::Error>> + Send + 'static;

	fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		self.inner.poll_ready(cx)
	}

	fn call(&mut self, request: Replayable<T>) -> Self::Future {
		let replacement = self.inner.clone();
		let mut inner = std::mem::replace(&mut self.inner, replacement);
		let config = self.config;
		let random = Arc::clone(&self.random);
		async move {
			let mut retries = 0;
			loop {
				let result = inner.call(request.replay()).await;
				match result {
					Ok(committed) => return Ok(committed),
					Err(error)
						if retries < config.max_retries
							&& error.classification.retryable_exact_request(true) =>
					{
						let delay = config.delay_for(retries, next_sample(&random), error.retry_delay());
						tokio::time::sleep(delay).await;
						retries += 1;
						inner.ready().await?;
					},
					Err(error) => {
						return Err(error);
					},
				}
			}
		}
	}
}

fn next_sample(state: &AtomicU64) -> f64 {
	let mut current = state.load(Ordering::Relaxed);
	loop {
		let mut next = current;
		next ^= next << 13;
		next ^= next >> 7;
		next ^= next << 17;
		match state.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed) {
			Ok(_) => {
				let bytes = next.to_be_bytes();
				let upper = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
				return f64::from(upper) / f64::from(u32::MAX);
			},
			Err(observed) => current = observed,
		}
	}
}

struct BufferedRequest {
	method:              http::Method,
	uri:                 http::Uri,
	version:             http::Version,
	headers:             http::HeaderMap,
	body:                Body,
	auth:                Option<AuthContext>,
	lease:               Option<CredentialLease>,
	credential_metadata: Option<CredentialMetadata>,
	sensitive_query:     Option<SensitiveQuery>,
	key:                 Option<EgressKey>,
}

impl BufferedRequest {
	fn new(request: &Request<Body>) -> Self {
		// `Full<Bytes>` and `HeaderValue` clones share their byte storage.
		// Rebuilding the HeaderMap itself is required for independent attempts.
		Self {
			method:              request.method().clone(),
			uri:                 request.uri().clone(),
			version:             request.version(),
			headers:             request.headers().clone(),
			body:                request.body().clone(),
			auth:                request.extensions().get::<AuthContext>().cloned(),
			lease:               request.extensions().get::<CredentialLease>().cloned(),
			credential_metadata: request.extensions().get::<CredentialMetadata>().cloned(),
			sensitive_query:     request.extensions().get::<SensitiveQuery>().cloned(),
			key:                 request.extensions().get::<EgressKey>().cloned(),
		}
	}

	fn build(&self) -> Request<Body> {
		let mut request = Request::new(self.body.clone());
		*request.method_mut() = self.method.clone();
		*request.uri_mut() = self.uri.clone();
		*request.version_mut() = self.version;
		*request.headers_mut() = self.headers.clone();
		if let Some(auth) = &self.auth {
			request.extensions_mut().insert(auth.clone());
		}
		if let Some(lease) = &self.lease {
			request.extensions_mut().insert(lease.clone());
		}
		if let Some(metadata) = &self.credential_metadata {
			request.extensions_mut().insert(metadata.clone());
		}
		if let Some(query) = &self.sensitive_query {
			request.extensions_mut().insert(query.clone());
		}
		if let Some(key) = &self.key {
			request.extensions_mut().insert(key.clone());
		}
		request
	}
}

#[cfg(test)]
mod tests {
	use std::{convert::Infallible, sync::atomic::AtomicUsize};

	use bytes::Bytes;
	use http_body_util::Full;
	use omp_llm_error::{Evidence, classify};
	use tower::service_fn;

	use super::*;

	#[test]
	fn retry_after_parses_both_forms_and_overrides_shorter_backoff() {
		let now = UNIX_EPOCH + Duration::from_secs(784_111_772);
		assert_eq!(parse_retry_after("5", now), Some(Duration::from_secs(5)));
		assert_eq!(
			parse_retry_after("Sun, 06 Nov 1994 08:49:37 GMT", now),
			Some(Duration::from_secs(5))
		);
		let provider_minimum = parse_retry_after("5", now);
		let config = RetryConfig {
			base_delay: Duration::from_millis(10),
			jitter: 0.0,
			..RetryConfig::default()
		};
		assert_eq!(config.delay_for(0, 0.5, provider_minimum), Duration::from_secs(5));
	}

	#[test]
	fn exponential_jitter_stays_within_configured_bounds() {
		let config = RetryConfig {
			base_delay:  Duration::from_secs(1),
			max_delay:   Duration::from_secs(20),
			max_retries: 5,
			jitter:      0.25,
		};
		for retry in 0..5 {
			let unjittered = Duration::from_secs(1_u64 << retry).min(config.max_delay);
			let low = config.backoff_for(retry, 0.0);
			let high = config.backoff_for(retry, 1.0);
			assert!(low >= unjittered.mul_f64(0.75));
			assert!(high <= unjittered.mul_f64(1.25));
			assert!(low <= high);
		}
	}

	#[tokio::test]
	async fn exact_replay_preserves_selected_credential_generation() {
		let selected = CredentialLease::new("provider", 41, 7);
		let auth = AuthContext::new("provider");
		let key = selected.egress_key();
		let metadata = CredentialMetadata {
			auth_kind:       crate::auth_inject::CredentialAuthKind::ApiKey,
			identity:        "developer@example.com".into(),
			account_id:      Some("account-41".into()),
			project_id:      Some("project-41".into()),
			organization_id: None,
		};
		let attempts = Arc::new(AtomicUsize::new(0));
		let observed = Arc::clone(&attempts);
		let expected_lease = selected.clone();
		let expected_auth = auth.clone();
		let expected_key = key.clone();
		let expected_metadata = metadata.clone();
		const QUERY_CANARY: &str = "canary-retry-query-secret";
		let inner = service_fn(move |request: Request<Body>| {
			assert_eq!(request.extensions().get::<CredentialLease>(), Some(&expected_lease));
			assert_eq!(request.extensions().get::<AuthContext>(), Some(&expected_auth));
			assert_eq!(request.extensions().get::<EgressKey>(), Some(&expected_key));
			assert_eq!(request.extensions().get::<CredentialMetadata>(), Some(&expected_metadata));
			let query = request
				.extensions()
				.get::<SensitiveQuery>()
				.expect("sensitive query survives replay");
			assert!(!format!("{request:?}").contains(QUERY_CANARY));
			assert!(!format!("{query:?}").contains(QUERY_CANARY));
			assert!(!request.uri().to_string().contains(QUERY_CANARY));
			let attempt = observed.fetch_add(1, Ordering::SeqCst);
			async move {
				if attempt == 0 {
					Err(PreCommitFailure::new(
						"retry",
						classify(&Evidence::http(503, "service unavailable")),
					))
				} else {
					Ok(Committed::new(()))
				}
			}
		});
		let mut request = Request::new(Full::new(Bytes::new()));
		request.extensions_mut().insert(auth);
		request.extensions_mut().insert(selected);
		request.extensions_mut().insert(key);
		request.extensions_mut().insert(metadata);
		request
			.extensions_mut()
			.insert(SensitiveQuery::new("key", QUERY_CANARY.as_bytes()));
		let mut retry = RetryLayer::with_seed(
			RetryConfig {
				base_delay: Duration::ZERO,
				max_retries: 1,
				jitter: 0.0,
				..RetryConfig::default()
			},
			1,
		)
		.layer(inner);

		retry.call(Replayable::buffered(request)).await.unwrap();

		assert_eq!(attempts.load(Ordering::SeqCst), 2);
	}

	#[tokio::test]
	async fn post_commit_failure_is_a_value_and_is_not_retried() {
		let attempts = Arc::new(AtomicUsize::new(0));
		let observed = Arc::clone(&attempts);
		let inner = service_fn(move |()| {
			observed.fetch_add(1, Ordering::SeqCst);
			async { Ok::<_, PreCommitFailure<Infallible>>(Committed::new(Err::<(), _>("stream"))) }
		});
		let mut retry = RetryLayer::with_seed(
			RetryConfig { base_delay: Duration::ZERO, ..RetryConfig::default() },
			1,
		)
		.layer(inner);
		let committed = retry.call(Replayable::new(())).await.unwrap();
		assert_eq!(committed.into_inner(), Err("stream"));
		assert_eq!(attempts.load(Ordering::SeqCst), 1);
	}

	#[test]
	fn classified_taxonomy_controls_exact_replay() {
		let retryable = classify(&Evidence::http(503, "service unavailable"));
		let terminal = classify(&Evidence::http(400, "invalid request"));
		assert!(retryable.retryable_exact_request(true));
		assert!(!terminal.retryable_exact_request(true));
	}
}
