//! Gateway-to-gateway catalog discovery and turn forwarding.
//!
//! A provider whose transport is [`TransportId::Omp`] is another gateway, not a
//! credential source. Its `ListModels` cards are copied into the downstream
//! view without rewriting identity, source, pricing, or availability. In
//! particular, upstream `LOGIN_REQUIRED` remains `LOGIN_REQUIRED`: login must
//! happen on the host that owns that provider credential.
//!
//! **Credential-locality invariant:** federation carries inference and
//! discovery requests, never provider credential bytes or credential metadata.
//! The generated [`InferenceClient`] is used without an auth-injection
//! interceptor; provider authentication remains inside the terminal gateway's
//! egress stack.
//!
//! Context is terminal-gateway state. [`FederatedProvider`] contains no shadow
//! context store and forwards chat through [`OmpFederation`]. Terminal events,
//! including `NEED_FULL`, cross each federation hop exactly once. A caller
//! repairs `NEED_FULL` by starting a new turn with the complete assembled
//! thread.

use std::{collections::BTreeMap, sync::Arc, time::Duration};

use futures::{StreamExt, stream::BoxStream};
use omp_core::Str;
use omp_llm_catalog::{
	models::{Availability, Modality, ModelCard, Price, PriceUnit, Source},
	provider::{Facet, ProviderEntry, TransportId},
};
use omp_llm_transport::omp::OmpFederation;
use omp_llm_types::{
	ChatRequest, ConvertError, Effort, Props, TurnError, TurnErrorKind, TurnEvent,
	facet::{Chat, Error as FacetError, Executor},
};
use omp_proto::inference::v1::{
	self as pb, inference_client::InferenceClient, model_event::Event as WireModelEvent,
};
use parking_lot::RwLock;
use smallvec::SmallVec;
use tokio::{sync::watch, task::JoinHandle, time::MissedTickBehavior};
use tonic::{Request, Status, transport::Channel};

/// Default interval used when an upstream does not implement `WatchModels`.
pub const DEFAULT_REFRESH_INTERVAL: Duration = Duration::from_secs(60);

/// A federation configuration, discovery, or wire-card failure.
#[derive(Debug, thiserror::Error)]
pub enum FederationError {
	/// The configured provider does not use the native OMP transport.
	#[error("provider {0} is not an omp federation provider")]
	NotOmp(Str),
	/// The refresh timer must make forward progress.
	#[error("federation refresh interval must be non-zero")]
	ZeroRefreshInterval,
	/// Only the gRPC classification is retained; status messages are untrusted
	/// upstream text and may echo request credentials.
	#[error("upstream discovery RPC failed ({0:?})")]
	Rpc(tonic::Code),
	/// A model card contained an enum value unknown to this schema revision.
	#[error("invalid upstream model-card {field} value {value}")]
	InvalidCard {
		/// Name of the invalid model-card field.
		field: &'static str,
		/// Unknown protobuf enum value.
		value: i32,
	},
	/// Vendor properties could not be converted to canonical properties.
	#[error("invalid upstream model-card properties")]
	Props,
}

impl From<Status> for FederationError {
	fn from(status: Status) -> Self {
		Self::Rpc(status.code())
	}
}

impl From<ConvertError> for FederationError {
	fn from(_error: ConvertError) -> Self {
		Self::Props
	}
}

/// A live OMP upstream: authoritative cards plus a context-transparent turn
/// client.
#[derive(Clone)]
pub struct FederatedProvider {
	provider_id: Str,
	cards:       Arc<RwLock<BTreeMap<Str, ModelCard>>>,
	discovery:   InferenceClient<Channel>,
	turns:       OmpFederation,
	lifetime:    Arc<RefreshLifetime>,
}

struct RefreshLifetime {
	stop: watch::Sender<bool>,
	task: JoinHandle<()>,
}

impl Drop for RefreshLifetime {
	fn drop(&mut self) {
		let _ = self.stop.send_replace(true);
		self.task.abort();
	}
}

impl FederatedProvider {
	/// Connects an OMP provider over an already authenticated gateway channel.
	///
	/// The initial `ListModels` completes before this returns. Afterwards the
	/// upstream watch stream drives changes; `UNIMPLEMENTED` falls back to the
	/// supplied periodic full-list interval.
	///
	/// The channel may authenticate the gateway peer (for example with mTLS),
	/// but this layer never reads or forwards a model-provider credential.
	///
	/// # Errors
	///
	/// Returns an error for a non-OMP provider, a zero interval, a failed
	/// initial list, or an invalid upstream card.
	pub async fn connect(
		provider: &ProviderEntry,
		channel: Channel,
		refresh_interval: Duration,
	) -> Result<Self, FederationError> {
		if provider.transport != TransportId::Omp {
			return Err(FederationError::NotOmp(provider.id.clone()));
		}
		if refresh_interval.is_zero() {
			return Err(FederationError::ZeroRefreshInterval);
		}

		let mut discovery = InferenceClient::new(channel.clone());
		let initial = discovery
			.list_models(federated_request(list_request()))
			.await?
			.into_inner();
		let cursor = initial.cursor.clone();
		let cards = Arc::new(RwLock::new(convert_snapshot(initial.models)?));
		let foreground_discovery = discovery.clone();
		let turns = OmpFederation::new(InferenceClient::new(channel));
		let (stop, receiver) = watch::channel(false);
		let refresh_cards = Arc::clone(&cards);
		let task = tokio::spawn(async move {
			refresh_loop(discovery, refresh_cards, cursor, refresh_interval, receiver).await;
		});
		let lifetime = Arc::new(RefreshLifetime { stop, task });
		Ok(Self {
			provider_id: provider.id.clone(),
			cards,
			discovery: foreground_discovery,
			turns,
			lifetime,
		})
	}

	/// Returns the downstream provider entry that owns this upstream connection.
	#[must_use]
	pub fn provider_id(&self) -> &str {
		self.provider_id.as_str()
	}

	/// Returns the latest complete upstream card snapshot.
	///
	/// Cards are semantically unchanged from the upstream wire values, including
	/// [`ModelCard::availability`]. Do not pass these through a local credential
	/// join that overwrites availability.
	#[must_use]
	pub fn cards(&self) -> Vec<ModelCard> {
		self.cards.read().values().cloned().collect()
	}

	/// Forwards one turn without creating downstream shadow context.
	///
	/// Terminal responses are never reinterpreted or retried. In particular,
	/// `NEED_FULL` ends this stream so the caller can explicitly open a new turn
	/// containing the complete assembled thread. Dropping the returned stream
	/// drops the active upstream RPC.
	pub async fn turn(
		&self,
		request: ChatRequest,
		executor: Option<Arc<dyn Executor>>,
	) -> Result<BoxStream<'static, TurnEvent>, FacetError> {
		let upstream = self.turns.turn(request, executor).await?;
		Ok(federated_turn_stream(upstream))
	}

	/// Refreshes the complete upstream snapshot immediately.
	///
	/// # Errors
	///
	/// Returns an upstream RPC or invalid-card error. The prior complete
	/// snapshot remains visible on failure.
	pub async fn refresh_now(&self) -> Result<(), FederationError> {
		let mut client = self.discovery.clone();
		let response = client
			.list_models(federated_request(list_request()))
			.await?
			.into_inner();
		let cards = convert_snapshot(response.models)?;
		*self.cards.write() = cards;
		Ok(())
	}

	/// Returns whether the background discovery task is still owned by this
	/// handle.
	#[must_use]
	pub fn refresh_is_live(&self) -> bool {
		!self.lifetime.task.is_finished()
	}
}

async fn refresh_loop(
	mut client: InferenceClient<Channel>,
	cards: Arc<RwLock<BTreeMap<Str, ModelCard>>>,
	mut cursor: Option<pb::Cursor>,
	refresh_interval: Duration,
	mut stop: watch::Receiver<bool>,
) {
	let mut watch_supported = true;
	loop {
		let mut refresh_immediately = false;
		if *stop.borrow() {
			return;
		}
		if watch_supported {
			match client
				.watch_models(federated_request(watch_request(cursor.clone())))
				.await
			{
				Ok(response) => {
					let mut stream = response.into_inner();
					loop {
						tokio::select! {
							changed = stop.changed() => {
								if changed.is_err() || *stop.borrow() {
									return;
								}
							},
							event = stream.message() => match event {
								Ok(Some(event)) => match apply_event(&cards, event) {
									Ok(EventAction::Continue(next)) => cursor = next,
									Ok(EventAction::Relist) => {
										refresh_immediately = true;
										break;
									},
									Err(error) => tracing::warn!(%error, "ignoring invalid federated model event"),
								},
								Ok(None) => break,
								Err(error) => {
									tracing::warn!(code = ?error.code(), "upstream model watch ended");
									break;
								},
							},
						}
					}
				},
				Err(error) if error.code() == tonic::Code::Unimplemented => {
					watch_supported = false;
				},
				Err(error) => {
					tracing::warn!(
						code = ?error.code(),
						"upstream model watch unavailable; retrying after refresh"
					);
				},
			}
		}

		if !refresh_immediately && wait_interval(refresh_interval, &mut stop).await {
			return;
		}
		match client.list_models(federated_request(list_request())).await {
			Ok(response) => {
				let response = response.into_inner();
				match convert_snapshot(response.models) {
					Ok(snapshot) => {
						*cards.write() = snapshot;
						cursor = response.cursor;
					},
					Err(error) => tracing::warn!(%error, "retaining prior federated model snapshot"),
				}
			},
			Err(error) => tracing::warn!(
				code = ?error.code(),
				"failed to refresh federated models"
			),
		}
	}
}

async fn wait_interval(interval: Duration, stop: &mut watch::Receiver<bool>) -> bool {
	let mut ticker = tokio::time::interval(interval);
	ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
	ticker.tick().await;
	tokio::select! {
		_ = ticker.tick() => false,
		changed = stop.changed() => changed.is_err() || *stop.borrow(),
	}
}

enum EventAction {
	Continue(Option<pb::Cursor>),
	Relist,
}

fn apply_event(
	cards: &RwLock<BTreeMap<Str, ModelCard>>,
	event: pb::ModelEvent,
) -> Result<EventAction, FederationError> {
	let cursor = event.cursor;
	match event.event {
		Some(WireModelEvent::Upserted(card)) => {
			let card = convert_card(card)?;
			cards.write().insert(card.id.clone(), card);
			Ok(EventAction::Continue(cursor))
		},
		Some(WireModelEvent::RemovedId(id)) => {
			cards.write().remove(id.as_str());
			Ok(EventAction::Continue(cursor))
		},
		Some(WireModelEvent::Reset(_)) | None => Ok(EventAction::Relist),
	}
}

const fn list_request() -> pb::ListModelsRequest {
	pb::ListModelsRequest {
		provider:       String::new(),
		facet:          pb::Facet::Unspecified as i32,
		available_only: false,
	}
}

const fn watch_request(cursor: Option<pb::Cursor>) -> pb::WatchModelsRequest {
	pb::WatchModelsRequest { since: cursor }
}

fn federated_request<T>(message: T) -> Request<T> {
	// Intentionally no metadata: federation never injects a provider credential.
	Request::new(message)
}

fn pass_through_event(event: TurnEvent) -> TurnEvent {
	match event {
		TurnEvent::Error(mut error) => {
			error.detail = Str::new_static("federated upstream turn failed");
			for diagnostic in &mut error.diagnostics {
				diagnostic.detail = Str::new_static("federated upstream attempt failed");
			}
			TurnEvent::Error(error)
		},
		event => event,
	}
}

fn federated_turn_stream(upstream: BoxStream<'static, TurnEvent>) -> BoxStream<'static, TurnEvent> {
	Box::pin(async_stream::stream! {
		let mut upstream = upstream;
		while let Some(event) = upstream.next().await {
			let terminal = matches!(event, TurnEvent::Outcome(_) | TurnEvent::Error(_));
			yield pass_through_event(event);
			if terminal {
				return;
			}
		}
		yield federation_stream_error("federated upstream ended without a terminal event");
	})
}

fn federation_stream_error(detail: impl AsRef<str>) -> TurnEvent {
	TurnEvent::Error(
		TurnError::builder()
			.kind(TurnErrorKind::Upstream)
			.detail(Str::new(detail.as_ref()))
			.unsupported(Vec::new())
			.retry_after_ms(0)
			.build(),
	)
}

fn convert_snapshot(
	cards: Vec<pb::ModelCard>,
) -> Result<BTreeMap<Str, ModelCard>, FederationError> {
	cards
		.into_iter()
		.map(convert_card)
		.map(|result| result.map(|card| (card.id.clone(), card)))
		.collect()
}

fn convert_card(card: pb::ModelCard) -> Result<ModelCard, FederationError> {
	let facets = card
		.facets
		.into_iter()
		.map(convert_facet)
		.collect::<Result<SmallVec<_, 4>, _>>()?;
	let inputs = card
		.inputs
		.into_iter()
		.map(convert_modality)
		.collect::<Result<SmallVec<_, 4>, _>>()?;
	let outputs = card
		.outputs
		.into_iter()
		.map(convert_modality)
		.collect::<Result<SmallVec<_, 4>, _>>()?;
	let efforts = card
		.efforts
		.into_iter()
		.map(convert_effort)
		.collect::<Result<SmallVec<_, 6>, _>>()?;
	let pricing = card
		.pricing
		.into_iter()
		.map(convert_price)
		.collect::<Result<SmallVec<_, 4>, _>>()?;
	let availability = convert_availability(card.availability)?;
	let source = convert_source(card.source)?;
	let props = match card.props {
		Some(props) => Props::try_from(props)?,
		None => Props::default(),
	};
	Ok(ModelCard::builder()
		.id(Str::from(card.id))
		.provider(Str::from(card.provider))
		.model(Str::from(card.model))
		.name(Str::from(card.name))
		.family(Str::from(card.family))
		.facets(facets)
		.inputs(inputs)
		.outputs(outputs)
		.reasoning(card.reasoning)
		.efforts(efforts)
		.context_window(card.context_window)
		.max_output_tokens(card.max_output_tokens)
		.pricing(pricing)
		.availability(availability)
		.source(source)
		.blocked_until_ms(card.blocked_until_ms)
		.deprecated(card.deprecated)
		.updated_at_ms(card.updated_at_ms)
		.props(props)
		.effort_routing(BTreeMap::new())
		.build())
}

fn convert_facet(value: i32) -> Result<Facet, FederationError> {
	match pb::Facet::try_from(value) {
		Ok(pb::Facet::Chat) => Ok(Facet::Chat),
		Ok(pb::Facet::Embed) => Ok(Facet::Embeddings),
		Ok(pb::Facet::ImageGen) => Ok(Facet::ImageGeneration),
		Ok(pb::Facet::VideoGen) => Ok(Facet::VideoGeneration),
		Ok(pb::Facet::Speak) => Ok(Facet::AudioSpeech),
		Ok(pb::Facet::Transcribe) => Ok(Facet::AudioTranscription),
		Ok(pb::Facet::Realtime) => Ok(Facet::Chat),
		Ok(pb::Facet::Search) => Ok(Facet::Chat),
		Ok(pb::Facet::Unspecified) | Err(_) => Err(invalid("facet", value)),
	}
}

fn convert_modality(value: i32) -> Result<Modality, FederationError> {
	match pb::Modality::try_from(value) {
		Ok(pb::Modality::Unspecified) => Ok(Modality::Unspecified),
		Ok(pb::Modality::Text) => Ok(Modality::Text),
		Ok(pb::Modality::Image) => Ok(Modality::Image),
		Ok(pb::Modality::Audio) => Ok(Modality::Audio),
		Ok(pb::Modality::Video) => Ok(Modality::Video),
		Ok(pb::Modality::Pdf) => Ok(Modality::Pdf),
		Err(_) => Err(invalid("modality", value)),
	}
}

fn convert_effort(value: i32) -> Result<Effort, FederationError> {
	match pb::Effort::try_from(value) {
		Ok(pb::Effort::Off) => Ok(Effort::Off),
		Ok(pb::Effort::Minimal) => Ok(Effort::Minimal),
		Ok(pb::Effort::Low) => Ok(Effort::Low),
		Ok(pb::Effort::Medium) => Ok(Effort::Medium),
		Ok(pb::Effort::High) => Ok(Effort::High),
		Ok(pb::Effort::Xhigh) => Ok(Effort::XHigh),
		Ok(pb::Effort::Max) => Ok(Effort::Max),
		Ok(pb::Effort::Unspecified) | Err(_) => Err(invalid("effort", value)),
	}
}

fn convert_price(price: pb::Price) -> Result<Price, FederationError> {
	let unit = match pb::price::Unit::try_from(price.unit) {
		Ok(pb::price::Unit::MtokInput) => PriceUnit::MtokInput,
		Ok(pb::price::Unit::MtokOutput) => PriceUnit::MtokOutput,
		Ok(pb::price::Unit::MtokCacheRead) => PriceUnit::MtokCacheRead,
		Ok(pb::price::Unit::MtokCacheWrite) => PriceUnit::MtokCacheWrite,
		Ok(pb::price::Unit::Image) => PriceUnit::Image,
		Ok(pb::price::Unit::VideoSecond) => PriceUnit::VideoSecond,
		Ok(pb::price::Unit::AudioSecond) => PriceUnit::AudioSecond,
		Ok(pb::price::Unit::McharInput) => PriceUnit::McharInput,
		Ok(pb::price::Unit::Request) => PriceUnit::Request,
		Ok(pb::price::Unit::Unspecified) | Err(_) => return Err(invalid("price.unit", price.unit)),
	};
	Ok(Price::builder()
		.unit(unit)
		.nanos_usd(price.nanos_usd)
		.build())
}

fn convert_availability(value: i32) -> Result<Availability, FederationError> {
	match pb::Availability::try_from(value) {
		Ok(pb::Availability::Unspecified) => Ok(Availability::Unspecified),
		Ok(pb::Availability::Available) => Ok(Availability::Available),
		Ok(pb::Availability::LoginRequired) => Ok(Availability::LoginRequired),
		Ok(pb::Availability::Blocked) => Ok(Availability::Blocked),
		Ok(pb::Availability::Disabled) => Ok(Availability::Disabled),
		Err(_) => Err(invalid("availability", value)),
	}
}

fn convert_source(value: i32) -> Result<Source, FederationError> {
	match pb::model_card::Source::try_from(value) {
		Ok(pb::model_card::Source::Unspecified) => Ok(Source::Unspecified),
		Ok(pb::model_card::Source::Bundled) => Ok(Source::Bundled),
		Ok(pb::model_card::Source::Discovered) => Ok(Source::Discovered),
		Ok(pb::model_card::Source::Configured) => Ok(Source::Configured),
		Err(_) => Err(invalid("source", value)),
	}
}

const fn invalid(field: &'static str, value: i32) -> FederationError {
	FederationError::InvalidCard { field, value }
}

#[cfg(test)]
mod tests {
	use omp_core::Str;
	use omp_llm_catalog::models::Availability;
	use omp_llm_types::{TurnError, TurnErrorKind, TurnEvent};
	use omp_proto::inference::v1 as pb;

	use super::{
		FederationError, convert_card, federated_request, list_request, pass_through_event,
		watch_request,
	};

	#[test]
	fn federated_list_models_preserves_login_required() {
		let wire = pb::ModelCard {
			id: "upstream/model".into(),
			provider: "upstream".into(),
			model: "model".into(),
			name: "Model".into(),
			family: "family".into(),
			availability: pb::Availability::LoginRequired as i32,
			..Default::default()
		};
		let card = convert_card(wire).unwrap();
		assert_eq!(card.availability, Availability::LoginRequired);
		assert_eq!(card.id.as_str(), "upstream/model");
	}

	#[test]
	fn federated_error_preserves_classification_but_redacts_upstream_detail() {
		const CANARY: &str = "canary-bearer-token-from-upstream";
		let event = TurnEvent::Error(
			TurnError::builder()
				.kind(TurnErrorKind::NeedFull)
				.detail(Str::new_static(CANARY))
				.unsupported(Vec::new())
				.retry_after_ms(37)
				.build(),
		);
		let forwarded = pass_through_event(event);
		let TurnEvent::Error(error) = forwarded else {
			panic!("error event must remain terminal");
		};
		assert_eq!(error.kind, TurnErrorKind::NeedFull);
		assert_eq!(error.retry_after_ms, 37);
		assert!(!error.detail.contains(CANARY));
		assert!(!format!("{error:?}").contains(CANARY));
	}

	#[test]
	fn federation_rpc_error_retains_code_without_status_message() {
		const CANARY: &str = "canary-refresh-token-in-status";
		let error = FederationError::from(tonic::Status::unauthenticated(CANARY));
		assert!(error.to_string().contains("Unauthenticated"));
		assert!(!error.to_string().contains(CANARY));
		assert!(!format!("{error:?}").contains(CANARY));
	}

	#[test]
	fn federated_requests_have_no_credential_metadata() {
		let list = federated_request(list_request());
		let watch = federated_request(watch_request(Some(pb::Cursor {
			epoch:      b"epoch".as_slice().into(),
			generation: 7,
		})));
		let turn = federated_request(pb::TurnFrame::default());
		assert!(list.metadata().is_empty());
		assert!(watch.metadata().is_empty());
		assert!(turn.metadata().is_empty());
		assert_eq!(list.get_ref().provider, "");
		assert!(!list.get_ref().available_only);
	}
}
