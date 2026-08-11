//! Usage accounting, catalog pricing, and compatible telemetry emission.

use std::{
	future::Future,
	pin::Pin,
	sync::Arc,
	task::{Context, Poll},
};

use futures::{Stream, TryFutureExt};
use omp_core::Str;
use omp_llm_catalog::models::{CacheTtlUsage, CostUsage, ModelCard, calculate_cost};
use omp_llm_egress::auth_inject::CredentialLease;
use omp_llm_types::{Accuracy, Cost, Usage};
use omp_proto::inference::v1::{
	Cost as WireCost, TurnEvent, TurnRequest, Usage as WireUsage, turn_event, turn_request,
};
use omp_telemetry::{
	collector,
	metrics::{ChatUsageMetric, MetricRecorder},
};
use pin_project_lite::pin_project;
use tower::{Layer, Service};

use crate::select::Routed;

/// Credential-aware terminal usage callback installed below route selection.
pub trait UsageObserver: Send + Sync + 'static {
	/// Records one authoritative terminal outcome for the exact selected lease.
	fn record_usage(
		&self,
		lease: &CredentialLease,
		turn_id: &str,
		model: &str,
		initiator: &str,
		premium_multiplier_millionths: Option<u64>,
		client_id: &str,
		client_label: &str,
		usage: &WireUsage,
		cost: &WireCost,
	);
}

/// Usage observer that intentionally discards terminal observations.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopUsageObserver;

impl UsageObserver for NoopUsageObserver {
	fn record_usage(
		&self,
		_lease: &CredentialLease,
		_turn_id: &str,
		_model: &str,
		_initiator: &str,
		_premium_multiplier_millionths: Option<u64>,
		_client_id: &str,
		_client_label: &str,
		_usage: &WireUsage,
		_cost: &WireCost,
	) {
	}
}

/// Layer observing authoritative outcomes from one selected routed attempt.
#[derive(Clone)]
pub struct UsageObserverLayer {
	observer: Arc<dyn UsageObserver>,
}

impl UsageObserverLayer {
	/// Creates a terminal usage layer over `observer`.
	#[must_use]
	pub fn new(observer: Arc<dyn UsageObserver>) -> Self {
		Self { observer }
	}
}

impl<S> Layer<S> for UsageObserverLayer {
	type Service = ObserveUsage<S>;

	fn layer(&self, inner: S) -> Self::Service {
		ObserveUsage { inner, observer: Arc::clone(&self.observer) }
	}
}

/// Service capturing selected routing identity before driving an outcome
/// stream.
#[derive(Clone)]
pub struct ObserveUsage<S> {
	inner:    S,
	observer: Arc<dyn UsageObserver>,
}

impl<S, St> Service<Routed> for ObserveUsage<S>
where
	S: Service<Routed, Response = St>,
	S::Future: Send,
	St: Stream<Item = TurnEvent> + Send,
{
	type Error = S::Error;
	type Response = ObservedStream<St>;

	type Future = impl Future<Output = Result<Self::Response, Self::Error>> + Send;

	fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		self.inner.poll_ready(cx)
	}

	fn call(&mut self, routed: Routed) -> Self::Future {
		let identity = ObservedIdentity::from_request(
			&routed.request,
			routed.lease.clone(),
			routed
				.model_policy
				.as_deref()
				.and_then(|policy| policy.premium_millionths),
		);
		let observer = Arc::clone(&self.observer);
		let response = self.inner.call(routed);
		response.map_ok(move |inner| ObservedStream { inner, observer, identity, terminal: false })
	}
}

#[derive(Clone)]
struct ObservedIdentity {
	lease: Option<CredentialLease>,
	turn_id: Str,
	model: Str,
	initiator: Str,
	premium_multiplier_millionths: Option<u64>,
	client_id: Str,
	client_label: Str,
}

impl ObservedIdentity {
	fn from_request(
		request: &TurnRequest,
		lease: Option<CredentialLease>,
		premium_multiplier_millionths: Option<u64>,
	) -> Self {
		let context_id = match request.input.as_ref() {
			Some(turn_request::Input::Incremental(input)) => input
				.context
				.as_ref()
				.map_or("", |context| context.context_id.as_str()),
			Some(turn_request::Input::Seed(seed)) => seed.context_id.as_str(),
			None => "",
		};
		let client_id = if context_id.is_empty() {
			"stateless"
		} else {
			context_id
		};
		Self {
			lease,
			turn_id: Str::new(&request.turn_id),
			model: request
				.params
				.as_ref()
				.map_or_else(|| Str::new_static(""), |params| Str::new(&params.model)),
			initiator: request
				.params
				.as_ref()
				.and_then(|params| params.meta.as_ref())
				.map_or_else(|| Str::new_static(""), |meta| Str::new(&meta.initiator)),
			premium_multiplier_millionths,
			client_id: Str::new(client_id),
			client_label: Str::new(client_id),
		}
	}
}

pin_project! {
	/// Stream firing one usage callback only for an authoritative outcome.
	pub struct ObservedStream<St> {
		#[pin]
		inner: St,
		observer: Arc<dyn UsageObserver>,
		identity: ObservedIdentity,
		terminal: bool,
	}
}

impl<St> Stream for ObservedStream<St>
where
	St: Stream<Item = TurnEvent>,
{
	type Item = TurnEvent;

	fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
		let mut this = self.project();
		let item = futures::ready!(this.inner.as_mut().poll_next(cx));
		if !*this.terminal
			&& let Some(event) = item.as_ref()
			&& let Some(turn_event::Event::Outcome(outcome)) = event.event.as_ref()
		{
			*this.terminal = true;
			if let (Some(lease), Some(usage), Some(cost)) =
				(this.identity.lease.as_ref(), outcome.usage.as_ref(), outcome.cost.as_ref())
			{
				this.observer.record_usage(
					lease,
					&this.identity.turn_id,
					&this.identity.model,
					&this.identity.initiator,
					this.identity.premium_multiplier_millionths,
					&this.identity.client_id,
					&this.identity.client_label,
					usage,
					cost,
				);
			}
		}
		Poll::Ready(item)
	}
}

/// Catalog-pricing and telemetry layer for completed provider turns.
#[derive(Clone, Debug)]
pub struct Meter {
	recorder: MetricRecorder,
}

impl Default for Meter {
	fn default() -> Self {
		Self::new(MetricRecorder::new())
	}
}

impl Meter {
	/// Creates a meter using an injected telemetry recorder.
	#[must_use]
	pub const fn new(recorder: MetricRecorder) -> Self {
		Self { recorder }
	}

	/// Prices and records one completed chat turn.
	///
	/// `billed_nanos_usd` is the provider's in-band bill when present. Catalog
	/// pricing is always marked estimated, even when token counts are exact.
	pub fn record_chat(
		&self,
		model: &ModelCard,
		usage: &Usage,
		cache_ttl: Option<CacheTtlUsage>,
		billed_nanos_usd: Option<u64>,
		service_tier: Option<Str>,
	) -> Cost {
		let cost = cost_for_usage(model, usage, cache_ttl, billed_nanos_usd);
		let telemetry_usage = collector_usage(usage);
		self.recorder.record_chat_usage(&ChatUsageMetric {
			provider: Some(model.provider.clone()),
			model: model.model.clone(),
			service_tier,
			agent: None,
			usage: telemetry_usage,
			usage_accuracy: Str::new(match usage.accuracy {
				Accuracy::Exact => "provider",
				Accuracy::Estimated => "estimated",
				_ => "estimated",
			}),
			cost_usd: Some(cost.nanos_usd as f64 / 1_000_000_000.0),
		});
		cost
	}
}

/// Computes canonical monetary cost from usage and catalog pricing.
///
/// Cache-write TTL buckets are passed to the catalog helper unchanged so its
/// five-minute residual and one-hour input-rate multiplier remain
/// authoritative.
#[must_use]
pub fn cost_for_usage(
	model: &ModelCard,
	usage: &Usage,
	cache_ttl: Option<CacheTtlUsage>,
	billed_nanos_usd: Option<u64>,
) -> Cost {
	if let Some(nanos_usd) = billed_nanos_usd {
		return Cost::builder()
			.nanos_usd(nanos_usd)
			.estimated(false)
			.build();
	}
	let breakdown = calculate_cost(
		model,
		&CostUsage::builder()
			.input(usage.input_tokens)
			.output(usage.output_tokens)
			.cache_read(usage.cache_read_tokens)
			.cache_write(usage.cache_write_tokens)
			.maybe_cttl(cache_ttl)
			.build(),
	);
	Cost::builder()
		.nanos_usd((breakdown.total * 1_000_000_000.0).round() as u64)
		.estimated(true)
		.build()
}

/// Projects canonical usage into the telemetry collector's stable bucket shape.
#[must_use]
pub const fn collector_usage(usage: &Usage) -> collector::Usage {
	let input = usage
		.input_tokens
		.saturating_add(usage.cache_read_tokens)
		.saturating_add(usage.cache_write_tokens);
	collector::Usage {
		input,
		output: usage.output_tokens,
		cached_input: usage.cache_read_tokens,
		cache_write: usage.cache_write_tokens,
		reasoning_output: 0,
		total: input.saturating_add(usage.output_tokens),
	}
}

#[cfg(test)]
mod tests {
	use std::{collections::BTreeMap, convert::Infallible};

	use futures::{StreamExt, stream};
	use omp_llm_catalog::{
		models::{Availability, Price, PriceUnit, Source},
		provider::Facet,
	};
	use omp_llm_types::Props;
	use parking_lot::Mutex;
	use smallvec::smallvec;
	use tower::{ServiceExt, service_fn};

	use super::*;

	fn priced_model() -> ModelCard {
		ModelCard::builder()
			.id(Str::new("known/model"))
			.provider(Str::new("known"))
			.model(Str::new("model"))
			.name(Str::new("Known"))
			.family(Str::new("known"))
			.facets(smallvec![Facet::Chat])
			.inputs(smallvec![])
			.outputs(smallvec![])
			.reasoning(false)
			.efforts(smallvec![])
			.context_window(0)
			.max_output_tokens(0)
			.pricing(smallvec![
				Price::builder()
					.unit(PriceUnit::MtokInput)
					.nanos_usd(2_000_000_000)
					.build(),
				Price::builder()
					.unit(PriceUnit::MtokOutput)
					.nanos_usd(4_000_000_000)
					.build(),
				Price::builder()
					.unit(PriceUnit::MtokCacheRead)
					.nanos_usd(200_000_000)
					.build(),
				Price::builder()
					.unit(PriceUnit::MtokCacheWrite)
					.nanos_usd(2_500_000_000)
					.build(),
			])
			.availability(Availability::Available)
			.source(Source::Bundled)
			.blocked_until_ms(0)
			.deprecated(false)
			.updated_at_ms(0)
			.props(Props::default())
			.effort_routing(BTreeMap::new())
			.build()
	}

	fn known_usage() -> Usage {
		Usage::builder()
			.input_tokens(100)
			.output_tokens(50)
			.cache_read_tokens(25)
			.cache_write_tokens(100)
			.accuracy(Accuracy::Exact)
			.detail(Props::default())
			.build()
	}

	#[test]
	fn catalog_cost_matches_ttl_split_formula_exactly() {
		let cost = cost_for_usage(
			&priced_model(),
			&known_usage(),
			Some(
				CacheTtlUsage::builder()
					.ephemeral_5m(30)
					.ephemeral_1h(20)
					.build(),
			),
			None,
		);
		assert_eq!(cost, Cost::builder().nanos_usd(685_000).estimated(true).build());
	}

	#[test]
	fn in_band_bill_is_the_only_non_estimated_cost() {
		let cost = cost_for_usage(&priced_model(), &known_usage(), None, Some(42));
		assert_eq!(cost, Cost::builder().nanos_usd(42).estimated(false).build());
	}

	#[derive(Default)]
	struct CapturingUsage {
		records: Mutex<Vec<(u64, String, String, String, Option<u64>, u64)>>,
	}

	impl UsageObserver for CapturingUsage {
		fn record_usage(
			&self,
			lease: &CredentialLease,
			turn_id: &str,
			model: &str,
			initiator: &str,
			premium_multiplier_millionths: Option<u64>,
			_client_id: &str,
			_client_label: &str,
			_usage: &WireUsage,
			cost: &WireCost,
		) {
			self.records.lock().push((
				lease.credential_id(),
				turn_id.to_owned(),
				model.to_owned(),
				initiator.to_owned(),
				premium_multiplier_millionths,
				cost.nanos_usd,
			));
		}
	}
	fn routed(id: u64, turn_id: &str) -> Routed {
		Routed::new(
			TurnRequest {
				turn_id: turn_id.to_owned(),
				params: Some(omp_proto::inference::v1::ChatParams {
					model: "resolved-model".to_owned(),
					meta: Some(omp_proto::inference::v1::RequestMeta {
						initiator: "agent".to_owned(),
						..Default::default()
					}),
					..Default::default()
				}),
				..TurnRequest::default()
			},
			Some(CredentialLease::new("provider", id, 1)),
			None,
		)
		.with_model_policy(Some(Arc::new(omp_llm_types::ResolvedModelPolicy {
			premium_millionths: Some(330_000),
			..omp_llm_types::ResolvedModelPolicy::default()
		})))
	}

	fn wire_outcome() -> TurnEvent {
		TurnEvent {
			event: Some(turn_event::Event::Outcome(omp_proto::inference::v1::Outcome {
				usage: Some(WireUsage { input_tokens: 3, output_tokens: 2, ..WireUsage::default() }),
				cost: Some(WireCost { nanos_usd: 77, ..WireCost::default() }),
				..Default::default()
			})),
		}
	}

	#[tokio::test]
	async fn rotation_records_only_the_committing_lease_once() {
		let observer = Arc::new(CapturingUsage::default());
		let layer = UsageObserverLayer::new(observer.clone());
		let inner = service_fn(|routed: Routed| async move {
			let frames = if routed.lease.as_ref().map(CredentialLease::credential_id) == Some(1) {
				vec![TurnEvent { event: Some(turn_event::Event::Error(Default::default())) }]
			} else {
				vec![wire_outcome(), wire_outcome()]
			};
			Ok::<_, Infallible>(stream::iter(frames))
		});
		let mut service = layer.layer(inner);
		let _: Vec<_> = service
			.ready()
			.await
			.unwrap()
			.call(routed(1, "turn-1"))
			.await
			.unwrap()
			.collect()
			.await;
		let _: Vec<_> = service
			.ready()
			.await
			.unwrap()
			.call(routed(2, "turn-1"))
			.await
			.unwrap()
			.collect()
			.await;
		assert_eq!(observer.records.lock().as_slice(), &[(
			2,
			"turn-1".to_owned(),
			"resolved-model".to_owned(),
			"agent".to_owned(),
			Some(330_000),
			77,
		)]);
	}

	#[tokio::test]
	async fn cancellation_without_outcome_records_nothing() {
		let observer = Arc::new(CapturingUsage::default());
		let layer = UsageObserverLayer::new(observer.clone());
		let inner =
			service_fn(|_routed: Routed| async { Ok::<_, Infallible>(stream::empty::<TurnEvent>()) });
		let mut service = layer.layer(inner);
		let _: Vec<_> = service
			.ready()
			.await
			.unwrap()
			.call(routed(1, "turn-cancelled"))
			.await
			.unwrap()
			.collect()
			.await;
		assert!(observer.records.lock().is_empty());
	}
}
