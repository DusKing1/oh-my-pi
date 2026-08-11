//! Provider-and-credential keyed admission control.
//!
//! Tower's stock concurrency and rate layers scope state to a service value.
//! Credentials sharing that service would therefore throttle one another. This
//! layer instead stores one bounded admission stack for every [`EgressKey`].

use std::{
	future::Future,
	sync::Arc,
	task::{Context, Poll},
	time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use http::{HeaderMap, Request, Response, StatusCode};
use omp_core::Str;
use omp_llm_error::extract::retry_hint_from_headers;
use parking_lot::{Mutex, RwLock};
use rustc_hash::FxHashMap;
use thiserror::Error;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tower::{Layer, Service, ServiceExt};

use crate::client::Body;

/// Admission-control key for one provider credential.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EgressKey {
	provider:      Str,
	credential_id: u64,
}

impl EgressKey {
	/// Constructs a provider-and-credential key.
	#[must_use]
	pub fn new(provider: impl AsRef<str>, credential_id: u64) -> Self {
		Self { provider: Str::new(provider.as_ref()), credential_id }
	}

	pub(crate) const fn from_str(provider: Str, credential_id: u64) -> Self {
		Self { provider, credential_id }
	}

	/// Returns the provider identifier.
	#[must_use]
	pub fn provider(&self) -> &str {
		&self.provider
	}

	/// Returns the non-secret credential identifier.
	#[must_use]
	pub const fn credential_id(&self) -> u64 {
		self.credential_id
	}
}

/// Persistent credential block produced by a provider rate limit.
#[derive(Clone, Debug)]
pub struct CredentialBlock {
	/// Provider-and-credential scope of the block.
	pub key:           EgressKey,
	/// Wall-clock instant after which another attempt may be made.
	pub blocked_until: SystemTime,
	/// HTTP status which caused the block.
	pub status:        StatusCode,
}

impl CredentialBlock {
	/// Classifies a provider response into a non-secret, credential-scoped
	/// block.
	///
	/// Authentication and permission rejections (401/403) and provider throttles
	/// (429) are block observations. Reset and retry headers determine the
	/// expiry; `default_block` is used when the provider supplies no usable
	/// timing.
	#[must_use]
	pub fn from_response(
		key: EgressKey,
		status: StatusCode,
		headers: &HeaderMap,
		now: SystemTime,
		default_block: Duration,
	) -> Option<Self> {
		if !matches!(
			status,
			StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN | StatusCode::TOO_MANY_REQUESTS
		) {
			return None;
		}
		let now_ms = now
			.duration_since(UNIX_EPOCH)
			.unwrap_or(Duration::ZERO)
			.as_millis()
			.try_into()
			.unwrap_or(u64::MAX);
		let mut selected = [("", ""); 6];
		let mut count = 0;
		for name in [
			"retry-after-ms",
			"retry-after",
			"x-ratelimit-reset-ms",
			"x-ratelimit-reset",
			"x-codex-primary-reset-at",
			"x-codex-secondary-reset-at",
		] {
			if let Some(value) = headers.get(name).and_then(|value| value.to_str().ok()) {
				selected[count] = (name, value);
				count += 1;
			}
		}
		let duration = retry_hint_from_headers(&selected[..count], now_ms)
			.map_or(default_block, |hint| Duration::from_millis(hint.delay_ms));
		Some(Self { key, blocked_until: now + duration, status })
	}
}

/// Broker hand-off for persisting credential block metadata.
///
/// Implementations should update the credential's `blocks` metadata before
/// returning. The interface carries no broker types, preserving dependency
/// direction.
pub trait BlockSink: Clone + Send + Sync + 'static {
	/// Persists a newly observed credential block.
	fn record_block(&self, block: &CredentialBlock);
}

/// Block sink used when persistence is not configured.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopBlockSink;

impl BlockSink for NoopBlockSink {
	fn record_block(&self, _block: &CredentialBlock) {}
}

/// Per-key queue, concurrency, rate, and default-block settings.
#[derive(Clone, Copy, Debug)]
pub struct LimitConfig {
	/// Maximum requests executing concurrently for one key.
	pub concurrency:     usize,
	/// Maximum queued requests beyond the concurrency allowance.
	pub buffer:          usize,
	/// Sustained requests per second, or `None` for no rate limit.
	pub rate_per_second: Option<f64>,
	/// Token-bucket burst capacity.
	pub burst:           u32,
	/// Block duration used when a 429 omits `Retry-After`.
	pub default_block:   Duration,
}

impl Default for LimitConfig {
	fn default() -> Self {
		Self {
			concurrency:     8,
			buffer:          32,
			rate_per_second: None,
			burst:           1,
			default_block:   Duration::from_secs(60),
		}
	}
}

/// Shared keyed admission and block state.
///
/// Cloning this handle preserves limiter queues and provider-response blocks
/// across separately constructed service values.
#[derive(Clone, Debug)]
pub struct KeyedLimitState {
	config: LimitConfig,
	keys:   Arc<Mutex<FxHashMap<EgressKey, Arc<KeyState>>>>,
}

impl KeyedLimitState {
	/// Constructs empty keyed state.
	#[must_use]
	pub fn new(config: LimitConfig) -> Self {
		assert!(config.concurrency > 0, "per-key concurrency must be non-zero");
		assert!(config.burst > 0, "per-key burst must be non-zero");
		assert!(
			config
				.rate_per_second
				.is_none_or(|rate| rate.is_finite() && rate > 0.0),
			"per-key rate must be finite and positive",
		);
		Self { config, keys: Arc::new(Mutex::new(FxHashMap::default())) }
	}

	/// Records a block in the shared in-memory state.
	pub fn record_block(
		&self,
		key: EgressKey,
		status: StatusCode,
		duration: Duration,
	) -> CredentialBlock {
		let state = self.key_state(&key);
		*state.blocked_until.write() = Some(Instant::now() + duration);
		CredentialBlock { key, blocked_until: SystemTime::now() + duration, status }
	}

	/// Returns the remaining block duration for `key`, if it is still blocked.
	#[must_use]
	pub fn blocked_for(&self, key: &EgressKey) -> Option<Duration> {
		let state = self.keys.lock().get(key).cloned()?;
		state.blocked_for()
	}

	fn key_state(&self, key: &EgressKey) -> Arc<KeyState> {
		let mut keys = self.keys.lock();
		Arc::clone(
			keys
				.entry(key.clone())
				.or_insert_with(|| Arc::new(KeyState::new(self.config))),
		)
	}
}

/// Layer enforcing limits independently for each [`EgressKey`].
#[derive(Clone, Debug)]
pub struct KeyedLimitsLayer<B = NoopBlockSink> {
	state:      KeyedLimitState,
	block_sink: B,
}

impl KeyedLimitsLayer<NoopBlockSink> {
	/// Constructs a keyed layer without persistent block hand-off.
	#[must_use]
	pub fn new(config: LimitConfig) -> Self {
		Self { state: KeyedLimitState::new(config), block_sink: NoopBlockSink }
	}
}

impl<B> KeyedLimitsLayer<B>
where
	B: BlockSink,
{
	/// Constructs a keyed layer with a broker-facing block sink.
	#[must_use]
	pub fn with_block_sink(config: LimitConfig, block_sink: B) -> Self {
		Self { state: KeyedLimitState::new(config), block_sink }
	}

	/// Returns a handle to the shared keyed limit and block state.
	#[must_use]
	pub fn state(&self) -> KeyedLimitState {
		self.state.clone()
	}
}

impl<B, S> Layer<S> for KeyedLimitsLayer<B>
where
	B: BlockSink,
{
	type Service = KeyedLimits<B, S>;

	fn layer(&self, inner: S) -> Self::Service {
		KeyedLimits { inner, state: self.state.clone(), block_sink: self.block_sink.clone() }
	}
}

/// Service applying the buffered stack selected by a request's [`EgressKey`].
#[derive(Clone, Debug)]
pub struct KeyedLimits<B, S> {
	inner:      S,
	state:      KeyedLimitState,
	block_sink: B,
}

/// Admission-control or wrapped-service failure.
#[derive(Debug, Error)]
pub enum LimitError<E> {
	/// The credential is persistently blocked by an earlier provider response.
	#[error("credential {key:?} is blocked for another {retry_after:?}")]
	Blocked {
		/// Blocked provider-and-credential key.
		key:         EgressKey,
		/// Remaining block duration.
		retry_after: Duration,
	},
	/// The bounded per-key admission queue is full.
	#[error("credential {0:?} admission queue is full")]
	Overloaded(EgressKey),
	/// The wrapped egress service failed.
	#[error("egress service failed: {0}")]
	Service(E),
}

impl<B, S, R> Service<Request<Body>> for KeyedLimits<B, S>
where
	B: BlockSink,
	S: Service<Request<Body>, Response = Response<R>> + Clone + Send + 'static,
	S::Future: Send + 'static,
	S::Error: Send + 'static,
	R: Send + 'static,
{
	type Error = LimitError<S::Error>;
	type Response = Response<R>;

	type Future = impl Future<Output = Result<Self::Response, Self::Error>> + Send + 'static;

	fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		// Admission and inner readiness are keyed and enforced inside `call`.
		// Clones are cheap handles over the same Arc-backed limiter state.
		Poll::Ready(Ok(()))
	}

	fn call(&mut self, request: Request<Body>) -> Self::Future {
		let replacement = self.inner.clone();
		let mut inner = std::mem::replace(&mut self.inner, replacement);
		let state = self.state.clone();
		let block_sink = self.block_sink.clone();
		async move {
			let Some(key) = request.extensions().get::<EgressKey>().cloned() else {
				return inner
					.ready()
					.await
					.map_err(LimitError::Service)?
					.call(request)
					.await
					.map_err(LimitError::Service);
			};
			let key_state = state.key_state(&key);
			let _permit = key_state.admit(&key).await?;
			let response = inner
				.ready()
				.await
				.map_err(LimitError::Service)?
				.call(request)
				.await
				.map_err(LimitError::Service)?;
			if let Some(observation) = CredentialBlock::from_response(
				key.clone(),
				response.status(),
				response.headers(),
				SystemTime::now(),
				state.config.default_block,
			) {
				let duration = observation
					.blocked_until
					.duration_since(SystemTime::now())
					.unwrap_or(Duration::ZERO);
				state.record_block(key, observation.status, duration);
				block_sink.record_block(&observation);
			}
			Ok(response)
		}
	}
}

#[derive(Debug)]
struct KeyState {
	queue:         Arc<Semaphore>,
	running:       Arc<Semaphore>,
	bucket:        Mutex<TokenBucket>,
	blocked_until: RwLock<Option<Instant>>,
}

impl KeyState {
	fn new(config: LimitConfig) -> Self {
		Self {
			queue:         Arc::new(Semaphore::new(config.concurrency.saturating_add(config.buffer))),
			running:       Arc::new(Semaphore::new(config.concurrency)),
			bucket:        Mutex::new(TokenBucket::new(config)),
			blocked_until: RwLock::new(None),
		}
	}

	fn blocked_for(&self) -> Option<Duration> {
		let until = *self.blocked_until.read();
		let remaining = until?.checked_duration_since(Instant::now());
		if remaining.is_none() {
			*self.blocked_until.write() = None;
		}
		remaining
	}

	async fn admit<E>(&self, key: &EgressKey) -> Result<AdmissionPermit, LimitError<E>> {
		if let Some(retry_after) = self.blocked_for() {
			return Err(LimitError::Blocked { key: key.clone(), retry_after });
		}
		let queue = Arc::clone(&self.queue)
			.try_acquire_owned()
			.map_err(|_| LimitError::Overloaded(key.clone()))?;
		let delay = self.bucket.lock().reserve();
		if !delay.is_zero() {
			tokio::time::sleep(delay).await;
		}
		if let Some(retry_after) = self.blocked_for() {
			return Err(LimitError::Blocked { key: key.clone(), retry_after });
		}
		let running = Arc::clone(&self.running)
			.acquire_owned()
			.await
			.expect("per-key semaphore is never closed");
		Ok(AdmissionPermit { _queue: queue, _running: running })
	}
}

struct AdmissionPermit {
	_queue:   OwnedSemaphorePermit,
	_running: OwnedSemaphorePermit,
}

#[derive(Debug)]
struct TokenBucket {
	rate_per_second: Option<f64>,
	capacity:        f64,
	tokens:          f64,
	updated:         Instant,
}

impl TokenBucket {
	fn new(config: LimitConfig) -> Self {
		Self {
			rate_per_second: config.rate_per_second,
			capacity:        f64::from(config.burst),
			tokens:          f64::from(config.burst),
			updated:         Instant::now(),
		}
	}

	fn reserve(&mut self) -> Duration {
		let Some(rate) = self.rate_per_second else {
			return Duration::ZERO;
		};
		let now = Instant::now();
		let elapsed = now.duration_since(self.updated).as_secs_f64();
		self.tokens = elapsed.mul_add(rate, self.tokens).min(self.capacity);
		self.updated = now;
		self.tokens -= 1.0;
		if self.tokens >= 0.0 {
			Duration::ZERO
		} else {
			Duration::from_secs_f64(-self.tokens / rate)
		}
	}
}

#[cfg(test)]
mod tests {
	use std::{
		convert::Infallible,
		sync::atomic::{AtomicUsize, Ordering},
	};

	use bytes::Bytes;
	use http::header::RETRY_AFTER;
	use http_body_util::Full;
	use tokio::sync::Notify;
	use tower::service_fn;

	use super::*;

	fn request(credential: u64) -> Request<Body> {
		let mut request = Request::new(Full::new(Bytes::new()));
		request
			.extensions_mut()
			.insert(EgressKey::new("provider", credential));
		request
	}

	#[tokio::test]
	async fn same_provider_credentials_have_isolated_limiters() {
		let entered = Arc::new(Notify::new());
		let release = Arc::new(Notify::new());
		let inner_entered = Arc::clone(&entered);
		let inner_release = Arc::clone(&release);
		let inner = service_fn(move |request: Request<Body>| {
			let credential = request
				.extensions()
				.get::<EgressKey>()
				.unwrap()
				.credential_id();
			let entered = Arc::clone(&inner_entered);
			let release = Arc::clone(&inner_release);
			async move {
				if credential == 1 {
					entered.notify_one();
					release.notified().await;
				}
				Ok::<_, Infallible>(Response::new(()))
			}
		});
		let layer =
			KeyedLimitsLayer::new(LimitConfig { concurrency: 1, buffer: 1, ..LimitConfig::default() });
		let service = layer.layer(inner);
		let mut first_service = service.clone();
		let first = tokio::spawn(async move { first_service.call(request(1)).await });
		entered.notified().await;
		let mut second_service = service;
		let second =
			tokio::time::timeout(Duration::from_millis(100), second_service.call(request(2))).await;
		assert!(second.is_ok(), "credential b was blocked by credential a");
		release.notify_one();
		first.await.unwrap().unwrap();
	}

	#[tokio::test]
	async fn response_block_is_visible_to_later_service_clones() {
		let calls = Arc::new(AtomicUsize::new(0));
		let observed = Arc::clone(&calls);
		let inner = service_fn(move |_request: Request<Body>| {
			observed.fetch_add(1, Ordering::SeqCst);
			async {
				let mut response = Response::builder()
					.status(StatusCode::TOO_MANY_REQUESTS)
					.body(())
					.unwrap();
				response
					.headers_mut()
					.insert(RETRY_AFTER, http::HeaderValue::from_static("5"));
				Ok::<_, Infallible>(response)
			}
		});
		let layer = KeyedLimitsLayer::new(LimitConfig::default());
		let service = layer.layer(inner);
		let mut first = service.clone();
		first.call(request(1)).await.unwrap();
		let mut later = service;
		assert!(matches!(later.call(request(1)).await, Err(LimitError::Blocked { .. })));
		assert_eq!(calls.load(Ordering::SeqCst), 1);
	}

	#[tokio::test]
	async fn auth_rejection_blocks_only_the_selected_credential() {
		let calls = Arc::new(AtomicUsize::new(0));
		let observed = Arc::clone(&calls);
		let inner = service_fn(move |request: Request<Body>| {
			observed.fetch_add(1, Ordering::SeqCst);
			let credential = request
				.extensions()
				.get::<EgressKey>()
				.expect("egress key")
				.credential_id();
			async move {
				Ok::<_, Infallible>(
					Response::builder()
						.status(if credential == 1 {
							StatusCode::FORBIDDEN
						} else {
							StatusCode::OK
						})
						.body(())
						.unwrap(),
				)
			}
		});
		let layer = KeyedLimitsLayer::new(LimitConfig::default());
		let mut service = layer.layer(inner);
		service.call(request(1)).await.unwrap();
		assert!(matches!(service.call(request(1)).await, Err(LimitError::Blocked { .. })));
		assert!(service.call(request(2)).await.is_ok());
		assert_eq!(calls.load(Ordering::SeqCst), 2);
	}

	#[test]
	fn response_observation_classifies_auth_throttle_and_reset_headers() {
		let now = UNIX_EPOCH + Duration::from_millis(1_000_000);
		let mut headers = HeaderMap::new();
		headers.insert("x-ratelimit-reset-ms", http::HeaderValue::from_static("2500"));
		for status in [StatusCode::UNAUTHORIZED, StatusCode::FORBIDDEN, StatusCode::TOO_MANY_REQUESTS]
		{
			let block = CredentialBlock::from_response(
				EgressKey::new("provider", 17),
				status,
				&headers,
				now,
				Duration::from_secs(60),
			)
			.expect("classified block");
			assert_eq!(block.key, EgressKey::new("provider", 17));
			assert_eq!(block.status, status);
			assert_eq!(block.blocked_until, now + Duration::from_millis(2_500));
		}
		assert!(
			CredentialBlock::from_response(
				EgressKey::new("provider", 17),
				StatusCode::BAD_REQUEST,
				&headers,
				now,
				Duration::from_secs(60),
			)
			.is_none()
		);
	}
}
