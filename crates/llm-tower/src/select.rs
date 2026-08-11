//! Credential selection, sticky session affinity, and safe rotation.
//!
//! Credential identity rides beside the protobuf request in [`Routed`]. It
//! never enters `TurnRequest.props`: only the gateway may turn the opaque lease
//! into broker-owned credential material at the adapter boundary.

use std::{
	collections::HashMap,
	fmt,
	future::{Ready, ready},
	sync::Arc,
	task::{Context, Poll},
	time::{SystemTime, UNIX_EPOCH},
};

use futures::{Stream, StreamExt, future::Either};
use omp_core::{Str, fmts};
/// Canonical non-secret credential routing values selected for a provider
/// attempt.
pub use omp_llm_egress::auth_inject::{CredentialLease, CredentialMetadata};
use omp_llm_error::{BlockKey, BlockTable, Kind};
use omp_llm_types::ResolvedModelPolicy;
use omp_proto::inference::v1::{
	Attempt, TurnError, TurnEvent, TurnRequest, turn_error, turn_event,
};
use parking_lot::Mutex;
use smallvec::SmallVec;
use tower::{Layer, Service, ServiceExt};

use crate::{
	SingleTurn,
	envelope::{ProviderRequest, TurnRequestEnvelope},
	recovery::classify_turn_error,
	single_turn,
};

/// A provider request paired with out-of-band credential routing values.
#[derive(Clone, Debug)]
pub struct Routed {
	/// Original inference request, kept free of credential identity and secret
	/// material.
	pub request:             TurnRequest,
	/// Trusted model policy carried beside protobuf.
	pub model_policy:        Option<Arc<ResolvedModelPolicy>>,
	/// Broker lease selected for this provider attempt.
	pub lease:               Option<CredentialLease>,
	/// Validated non-secret metadata for the selected lease.
	///
	/// This remains beside the protobuf request and is never serialized into
	/// request properties.
	pub credential_metadata: Option<CredentialMetadata>,
}

impl Routed {
	/// Pairs a canonical request with its broker-owned routing values.
	#[must_use]
	pub const fn new(
		request: TurnRequest,
		lease: Option<CredentialLease>,
		credential_metadata: Option<CredentialMetadata>,
	) -> Self {
		Self { request, model_policy: None, lease, credential_metadata }
	}

	/// Attaches trusted model policy without placing it on the protobuf wire.
	#[must_use]
	pub fn with_model_policy(mut self, model_policy: Option<Arc<ResolvedModelPolicy>>) -> Self {
		self.model_policy = model_policy;
		self
	}

	fn from_envelope<R: TurnRequestEnvelope>(
		request: &R,
		lease: Option<CredentialLease>,
		credential_metadata: Option<CredentialMetadata>,
	) -> Self {
		Self::new(request.request().clone(), lease, credential_metadata)
			.with_model_policy(request.model_policy().cloned())
	}
}

/// Inline credential candidate list; providers normally expose only a handful
/// of accounts.
pub type CredentialCandidates = SmallVec<u64, 4>;

/// Ranked credential source consulted for each model dispatch.
pub trait CredentialPool: Send + Sync + 'static {
	/// Returns broker credential ids in preferred order for `model`.
	fn candidates(&self, model: &str) -> CredentialCandidates;
}

/// Resolves current broker leases and their non-secret metadata.
pub trait LeaseSource: Send + Sync + 'static {
	/// Returns a current lease for `id`, or `None` when it is unavailable.
	fn lease(&self, id: u64) -> Option<CredentialLease>;

	/// Returns validated non-secret metadata for `lease` when available.
	fn metadata(&self, _lease: &CredentialLease) -> Option<CredentialMetadata> {
		None
	}
}

/// [`Layer`] producing credential-selecting [`Select`] services.
#[derive(Clone)]
pub struct SelectLayer {
	pool:   Arc<dyn CredentialPool>,
	leases: Arc<dyn LeaseSource>,
	blocks: Arc<Mutex<BlockTable>>,
	pins:   Arc<Mutex<HashMap<Str, u64>>>,
}

impl SelectLayer {
	/// Creates a layer using ranked ids, broker lease resolution, and shared
	/// blocks.
	pub fn new(
		pool: Arc<dyn CredentialPool>,
		leases: Arc<dyn LeaseSource>,
		blocks: Arc<Mutex<BlockTable>>,
	) -> Self {
		Self { pool, leases, blocks, pins: Arc::new(Mutex::new(HashMap::new())) }
	}
}

impl<S> Layer<S> for SelectLayer {
	type Service = Select<S>;

	fn layer(&self, inner: S) -> Self::Service {
		Select {
			inner,
			pool: Arc::clone(&self.pool),
			leases: Arc::clone(&self.leases),
			blocks: Arc::clone(&self.blocks),
			pins: Arc::clone(&self.pins),
		}
	}
}

/// Credential-selecting wrapper around a routed inference-attempt service.
#[derive(Clone)]
pub struct Select<S> {
	inner:  S,
	pool:   Arc<dyn CredentialPool>,
	leases: Arc<dyn LeaseSource>,
	blocks: Arc<Mutex<BlockTable>>,
	pins:   Arc<Mutex<HashMap<Str, u64>>>,
}

impl<S> Select<S> {
	/// Wraps `inner` with ranked credential selection and shared blocks.
	pub fn new(
		inner: S,
		pool: Arc<dyn CredentialPool>,
		leases: Arc<dyn LeaseSource>,
		blocks: Arc<Mutex<BlockTable>>,
	) -> Self {
		Self { inner, pool, leases, blocks, pins: Arc::new(Mutex::new(HashMap::new())) }
	}
}

impl<S, St> Service<ProviderRequest> for Select<S>
where
	S: Service<Routed, Response = St> + Clone + Send + 'static,
	S::Future: Send,
	S::Error: fmt::Display + Send + 'static,
	St: Stream<Item = TurnEvent> + Send + 'static,
{
	type Error = S::Error;
	type Future = Ready<Result<Self::Response, S::Error>>;
	type Response = Either<SingleTurn, SelectStream<S, St>>;

	fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		// Request-dependent short-circuit layer: the no-usable-credential
		// branch never dispatches, so reserving inner readiness here would
		// leak the reservation. Inner readiness is driven in the dispatch
		// branch instead.
		Poll::Ready(Ok(()))
	}

	fn call(&mut self, req: ProviderRequest) -> Self::Future {
		let model = req
			.request()
			.params
			.as_ref()
			.map_or_else(|| Str::new_static(""), |params| Str::new(&params.model));
		let session = req
			.request()
			.params
			.as_ref()
			.and_then(|params| params.cache.as_ref())
			.map_or_else(|| Str::new_static(""), |cache| Str::new(&cache.session_key));
		let candidates = self.pool.candidates(model.as_str());
		let pinned = (!session.is_empty())
			.then(|| self.pins.lock().get(&session).copied())
			.flatten();
		let selection =
			select_credential(&candidates, pinned, &self.blocks, self.leases.as_ref(), now_ms());
		let Some((credential_id, lease, credential_metadata)) = selection.route else {
			return ready(Ok(Either::Left(no_credential_error(
				model.as_str(),
				selection.blocked,
				selection.earliest_unblock_ms,
			))));
		};

		let clone = self.inner.clone();
		let inner = std::mem::replace(&mut self.inner, clone);
		let leases = Arc::clone(&self.leases);
		let blocks = Arc::clone(&self.blocks);
		let pins = Arc::clone(&self.pins);
		ready(Ok(Either::Right(select_stream(
			inner,
			req,
			lease,
			credential_metadata,
			candidates,
			credential_id,
			session,
			leases,
			blocks,
			pins,
		))))
	}
}

struct Selection {
	route:               Option<(u64, CredentialLease, Option<CredentialMetadata>)>,
	blocked:             usize,
	earliest_unblock_ms: u64,
}

fn select_credential(
	candidates: &[u64],
	pinned: Option<u64>,
	blocks: &Arc<Mutex<BlockTable>>,
	leases: &dyn LeaseSource,
	now_ms: u64,
) -> Selection {
	let usable = |credential_id: u64| {
		let blocked = blocks
			.lock()
			.blocked_for_ms(&block_key(credential_id), now_ms)
			.is_some();
		(!blocked)
			.then(|| leases.lease(credential_id))
			.flatten()
			.map(|lease| {
				let metadata = leases.metadata(&lease);
				(lease, metadata)
			})
	};
	let route = pinned
		.filter(|pin| candidates.contains(pin))
		.and_then(|pin| usable(pin).map(|(lease, metadata)| (pin, lease, metadata)))
		.or_else(|| {
			candidates.iter().find_map(|&credential_id| {
				usable(credential_id).map(|(lease, metadata)| (credential_id, lease, metadata))
			})
		});

	let table = blocks.lock();
	let mut blocked = 0;
	let mut earliest_unblock_ms = u64::MAX;
	for &candidate in candidates {
		if let Some(remaining) = table.blocked_for_ms(&block_key(candidate), now_ms) {
			blocked += 1;
			earliest_unblock_ms = earliest_unblock_ms.min(now_ms.saturating_add(remaining));
		}
	}
	if earliest_unblock_ms == u64::MAX {
		earliest_unblock_ms = 0;
	}

	Selection { route, blocked, earliest_unblock_ms }
}

/// Concrete rotating-credential stream.
///
/// One heap-pinned generator per call: the single allocation keeps this
/// layer's state behind a pointer, so composed stacks stay flat. Fully
/// inline generator nesting embeds every inner layer's state in the
/// parent's and was measured to overflow the thread stack at this
/// composition depth; a hand-written pin-projected state machine is the
/// box-free replacement if this layer ever gets hot. Erase to a boxed-dyn
/// stream only
/// at the outer boundary.
pub type SelectStream<
	S: Service<Routed, Response = St> + Send + 'static,
	St: Stream<Item = TurnEvent> + Send + 'static,
>
	= impl Stream<Item = TurnEvent> + Send + Unpin
where
	S::Future: Send,
	S::Error: fmt::Display + Send + 'static;

#[define_opaque(SelectStream)]
fn select_stream<S, St>(
	svc: S,
	req: ProviderRequest,
	lease: CredentialLease,
	credential_metadata: Option<CredentialMetadata>,
	candidates: CredentialCandidates,
	credential_id: u64,
	session: Str,
	leases: Arc<dyn LeaseSource>,
	blocks: Arc<Mutex<BlockTable>>,
	pins: Arc<Mutex<HashMap<Str, u64>>>,
) -> SelectStream<S, St>
where
	S: Service<Routed, Response = St> + Send + 'static,
	S::Future: Send,
	S::Error: fmt::Display + Send + 'static,
	St: Stream<Item = TurnEvent> + Send + 'static,
{
	Box::pin(async_stream::stream! {
		let mut svc = svc;
		let first = match svc.ready().await {
			Ok(svc) => match svc
				.call(Routed::from_envelope(&req, Some(lease), credential_metadata))
				.await
			{
				Ok(stream) => Either::Left(stream),
				Err(error) => Either::Right(single_turn(service_error(&error))),
			},
			Err(error) => Either::Right(single_turn(service_error(&error))),
		};
		let mut current = std::pin::pin!(first);
		let mut credential_id = credential_id;
		let mut dispatches: u32 = 1;
		let mut saw_part = false;
		let mut invoked = false;
		loop {
			let Some(event) = current.next().await else {
				return;
			};
			let err = match event.event.as_ref() {
				Some(turn_event::Event::Outcome(_)) => {
					if !session.is_empty() {
						pins.lock().insert(session.clone(), credential_id);
					}
					yield event;
					return;
				},
				Some(
					turn_event::Event::PartStart(_)
					| turn_event::Event::PartDelta(_)
					| turn_event::Event::PartEnd(_),
				) => {
					saw_part = true;
					yield event;
					continue;
				},
				Some(turn_event::Event::Invoke(_) | turn_event::Event::InvokeCancel(_)) => {
					invoked = true;
					yield event;
					continue;
				},
				Some(turn_event::Event::Error(err)) => {
					err
				},
				_ => {
					yield event;
					continue;
				},
			};

			let cls = classify_turn_error(err);
			let rotatable = cls.credential_recoverable()
				// A stale-session failure rides on auth statuses but the
				// credential is fine; rotating or blocking it here is the
				// documented way to waste a healthy account.
				&& !cls.kinds.has(Kind::StaleSessionItem)
				&& (cls.rate_limit.is_some_and(|rate_limit| rate_limit.rotate)
					|| cls.kinds.has(Kind::AccountPolicy)
					|| cls.kinds.has(Kind::AuthFailed));
			if !rotatable {
				yield event;
				return;
			}

			let reason = truncate(&err.detail, 256);
			let now = now_ms();
			blocks.lock().block_for(block_key(credential_id), now, &cls);
			if saw_part || invoked {
				yield event;
				return;
			}

			let next = select_credential(&candidates, None, &blocks, leases.as_ref(), now).route;
			let Some((next_credential_id, next_lease, next_metadata)) = next else {
				yield event;
				return;
			};

			let next_stream = match svc.ready().await {
				Ok(ready) => ready
					.call(Routed::from_envelope(&req, Some(next_lease), next_metadata))
					.await,
				Err(error) => Err(error),
			};
			current.set(match next_stream {
				Ok(stream) => Either::Left(stream),
				Err(error) => Either::Right(single_turn(service_error(&error))),
			});
			credential_id = next_credential_id;
			dispatches += 1;
			saw_part = false;
			yield TurnEvent {
				event: Some(turn_event::Event::Attempt(Attempt { number: dispatches, reason })),
			};
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

fn block_key(credential_id: u64) -> BlockKey {
	BlockKey { credential: fmts!("{credential_id}"), scope: None }
}

fn no_credential_error(model: &str, blocked: usize, earliest_unblock_ms: u64) -> SingleTurn {
	let event = TurnEvent {
		event: Some(turn_event::Event::Error(TurnError {
			kind: turn_error::Kind::Auth as i32,
			detail: format!(
				"no usable credential for {model}: {blocked} blocked until {earliest_unblock_ms}"
			),
			..TurnError::default()
		})),
	};
	single_turn(event)
}

fn now_ms() -> u64 {
	SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.map_or(0, |duration| duration.as_millis() as u64)
}

fn truncate(detail: &str, max: usize) -> String {
	if detail.len() <= max {
		return detail.to_owned();
	}
	let mut end = max;
	while !detail.is_char_boundary(end) {
		end -= 1;
	}
	detail[..end].to_owned()
}
