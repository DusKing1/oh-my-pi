//! OAuth credential refresh middleware.
//!
//! The layer refreshes credentials nearing expiry before dispatch and gives
//! one replay-safe authentication failure a forced refresh and re-dispatch.
//! Definitive refresh failures are deliberately not invalidated here: the
//! caller that owns the credential store is responsible for invalidation.

use std::{
	fmt,
	future::{Ready, ready},
	pin::Pin,
	sync::Arc,
	task::{Context, Poll},
	time::{Duration, SystemTime, UNIX_EPOCH},
};

use futures::{Stream, StreamExt};
use omp_core::SmolStr;
use omp_llm_error::{Kind, OAuthFailure, oauth::classify_refresh};
use omp_proto::inference::v1::{Attempt, TurnError, TurnEvent, turn_error, turn_event};
use tower::{Layer, Service, ServiceExt};

use crate::{envelope::TurnRequestEnvelope, recovery::classify_turn_error};

/// Failure returned by a provider-specific OAuth token refresher.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RefreshFailure {
	/// Diagnostic returned by the token endpoint or credential store.
	pub message: SmolStr,
}

impl RefreshFailure {
	/// Creates a refresh failure with the given diagnostic.
	pub fn new(message: impl AsRef<str>) -> Self {
		Self { message: SmolStr::new(message) }
	}
}

impl std::fmt::Display for RefreshFailure {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.write_str(&self.message)
	}
}

impl std::error::Error for RefreshFailure {}

/// Provider-specific access-token refresh operation.
///
/// Implementations own token exchange, persistence, and coalescing concurrent
/// calls for the same credential. The gateway implementation delegates this
/// contract to the broker's credential-id-keyed `Store::refresh_singleflight`;
/// this layer intentionally adds no parallel single-flight mechanism. `force`
/// requests a new token even when the current expiry still appears usable.
pub trait CredentialRefresher: Send + Sync + 'static {
	/// Returns the current access-token expiry as Unix milliseconds.
	///
	/// `None` denotes a non-expiring credential.
	fn expires_at_ms(&self) -> Option<u64>;

	/// Refreshes and persists the credential.
	fn refresh(
		&self,
		force: bool,
	) -> Pin<Box<dyn Future<Output = Result<(), RefreshFailure>> + Send + '_>>;
}

/// OAuth refresh timing policy.
#[derive(Clone, Debug)]
pub struct RefreshConfig {
	/// Refresh before dispatch when expiry is within this many milliseconds.
	pub skew_ms:            u64,
	/// Maximum duration of one provider-owned refresh operation.
	pub refresh_timeout_ms: u64,
}

impl Default for RefreshConfig {
	fn default() -> Self {
		Self { skew_ms: 120_000, refresh_timeout_ms: 30_000 }
	}
}

/// [`Layer`] producing OAuth-refreshing [`Refresh`] services.
#[derive(Clone)]
pub struct RefreshLayer {
	refresher: Arc<dyn CredentialRefresher>,
	config:    Arc<RefreshConfig>,
}

impl RefreshLayer {
	/// Creates a layer for one credential and the given timing policy.
	pub fn new(refresher: Arc<dyn CredentialRefresher>, config: RefreshConfig) -> Self {
		Self { refresher, config: Arc::new(config) }
	}
}

impl<S> Layer<S> for RefreshLayer {
	type Service = Refresh<S>;

	fn layer(&self, inner: S) -> Self::Service {
		Refresh { inner, refresher: Arc::clone(&self.refresher), config: Arc::clone(&self.config) }
	}
}

/// OAuth-refreshing wrapper around a provider-attempt service.
#[derive(Clone)]
pub struct Refresh<S> {
	inner:     S,
	refresher: Arc<dyn CredentialRefresher>,
	config:    Arc<RefreshConfig>,
}

impl<S> Refresh<S> {
	/// Wraps `inner` for one credential and the given timing policy.
	pub fn new(inner: S, refresher: Arc<dyn CredentialRefresher>, config: RefreshConfig) -> Self {
		Self { inner, refresher, config: Arc::new(config) }
	}
}

impl<S, St, R> Service<R> for Refresh<S>
where
	R: TurnRequestEnvelope,
	S: Service<R, Response = St> + Clone + Send + 'static,
	S::Future: Send,
	S::Error: fmt::Display + Send + 'static,
	St: Stream<Item = TurnEvent> + Send + 'static,
{
	type Error = S::Error;
	type Future = Ready<Result<Self::Response, S::Error>>;
	type Response = RefreshStream<S, St, R>;

	fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		// A proactive refresh can hold the dispatch for many seconds;
		// reserving inner readiness across that wait would pin a slot in
		// readiness-sensitive inner services. Inner readiness is driven
		// after the refresh completes.
		Poll::Ready(Ok(()))
	}

	fn call(&mut self, req: R) -> Self::Future {
		let clone = self.inner.clone();
		let inner = std::mem::replace(&mut self.inner, clone);
		let refresher = Arc::clone(&self.refresher);
		let config = Arc::clone(&self.config);
		ready(Ok(refresh_stream(inner, req, refresher, config)))
	}
}

/// Concrete refresh-on-auth-failure stream.
///
/// One heap-pinned generator per call: the single allocation keeps this
/// layer's state behind a pointer, so composed stacks stay flat. Fully
/// inline generator nesting embeds every inner layer's state in the
/// parent's and was measured to overflow the thread stack at this
/// composition depth; a hand-written pin-projected state machine is the
/// box-free replacement if this layer ever gets hot. Erase to a boxed-dyn
/// stream only
/// at the outer boundary.
pub type RefreshStream<
	S: Service<R, Response = St> + Send + 'static,
	St: Stream<Item = TurnEvent> + Send + 'static,
	R: TurnRequestEnvelope,
>
	= impl Stream<Item = TurnEvent> + Send + Unpin
where
	S::Future: Send,
	S::Error: fmt::Display + Send + 'static;

#[define_opaque(RefreshStream)]
fn refresh_stream<S, St, R>(
	svc: S,
	req: R,
	refresher: Arc<dyn CredentialRefresher>,
	config: Arc<RefreshConfig>,
) -> RefreshStream<S, St, R>
where
	R: TurnRequestEnvelope,
	S: Service<R, Response = St> + Send + 'static,
	S::Future: Send,
	S::Error: fmt::Display + Send + 'static,
	St: Stream<Item = TurnEvent> + Send + 'static,
{
	Box::pin(async_stream::stream! {
		let mut svc = svc;
		if needs_proactive_refresh(refresher.as_ref(), config.skew_ms) {
			let _ = refresh_credential(refresher.as_ref(), false, config.refresh_timeout_ms).await;
		}
		let first = match svc.ready().await {
			Ok(svc) => match svc.call(req.clone()).await {
				Ok(stream) => stream,
				Err(error) => {
					yield service_error(&error);
					return;
				},
			},
			Err(error) => {
				yield service_error(&error);
				return;
			},
		};
		let mut current = std::pin::pin!(first);
		let mut saw_output = false;
		let mut invoked = false;
		let mut retried = false;
		loop {
			let Some(event) = current.next().await else {
				return;
			};
			match event.event.as_ref() {
				Some(
					turn_event::Event::PartStart(_)
					| turn_event::Event::PartDelta(_)
					| turn_event::Event::PartEnd(_),
				) => {
					saw_output = true;
					yield event;
					continue;
				},
				Some(turn_event::Event::Invoke(_) | turn_event::Event::InvokeCancel(_)) => {
					invoked = true;
					saw_output = true;
					yield event;
					continue;
				},
				Some(turn_event::Event::Outcome(_)) => {
					yield event;
					return;
				},
				Some(turn_event::Event::Error(err)) => {
					let cls = classify_turn_error(err);
					let refreshable = cls.kinds.has(Kind::AuthFailed)
						&& !cls.kinds.has(Kind::OAuthExpired)
						// A stale-session 401 is server-side replay state, not a
						// credential problem; refreshing on it churns a valid
						// token.
						&& !cls.kinds.has(Kind::StaleSessionItem)
						&& !saw_output
						&& !invoked
						&& !retried;
					if !refreshable {
						yield event;
						return;
					}
				},
				_ => {
					yield event;
					continue;
				},
			}

			retried = true;
			match refresh_credential(refresher.as_ref(), true, config.refresh_timeout_ms).await {
				Ok(()) => {
					let redispatch = match svc.ready().await {
						Ok(svc) => svc.call(req.clone()).await,
						Err(error) => Err(error),
					};
					if let Ok(next) = redispatch {
						current.set(next);
						yield TurnEvent {
							event: Some(turn_event::Event::Attempt(Attempt {
								number: 2,
								reason: "OAuth credential refreshed".to_owned(),
							})),
						};
					} else {
						yield event;
						return;
					}
				},
				Err(failure) => {
					if classify_refresh(&failure.message) == OAuthFailure::Definitive {
						// The credential-store owner invalidates dead grants.
						// This layer preserves the provider's original
						// terminal frame.
					}
					yield event;
					return;
				},
			}
		}
	})
}

fn service_error(error: &impl fmt::Display) -> TurnEvent {
	TurnEvent {
		event: Some(turn_event::Event::Error(TurnError {
			kind: turn_error::Kind::Upstream as i32,
			detail: error.to_string(),
			..TurnError::default()
		})),
	}
}

async fn refresh_credential(
	refresher: &dyn CredentialRefresher,
	force: bool,
	timeout_ms: u64,
) -> Result<(), RefreshFailure> {
	tokio::time::timeout(Duration::from_millis(timeout_ms), refresher.refresh(force))
		.await
		.unwrap_or_else(|_| Err(RefreshFailure::new("OAuth refresh timed out")))
}

fn needs_proactive_refresh(refresher: &dyn CredentialRefresher, skew_ms: u64) -> bool {
	refresher
		.expires_at_ms()
		.is_some_and(|expiry| now_ms().saturating_add(skew_ms) >= expiry)
}

fn now_ms() -> u64 {
	SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.map_or(0, |duration| duration.as_millis() as u64)
}
