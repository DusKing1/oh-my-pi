//! Production route registration and session-affinity integration proofs.

use std::{
	collections::BTreeMap,
	convert::Infallible,
	future::Future,
	pin::Pin,
	sync::{
		Arc,
		atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
	},
	task::{Context as TaskContext, Poll},
};

use async_trait::async_trait;
use bytes::Bytes;
use futures::{Stream, StreamExt, stream};
use http::{Request, Response, header};
use http_body_util::{BodyExt, Full};
use omp_core::Str;
use omp_llm_catalog::{
	compat::Compat,
	models::{Availability, Modality, ModelCard, ModelCatalog, ModelWire, Source},
	provider::{AuthSpec, Facet, ProviderEntry, TransportId},
	registry::{CredentialView, Registry},
};
use omp_llm_egress::client::Body;
use omp_llm_error::{Classification, Feature, policy::BlockTable};
use omp_llm_gateway::{
	context::ContextStore,
	routes::{RouteRegistrationError, SpecializedChats, register_production_routes},
	turn::{ChatResolver, RoutedChat, TurnEngine, TurnStream},
};
use omp_llm_local::{Embedded, Inference};
use omp_llm_tower::{
	learn::RequestRepair,
	preflight::{Admission, UsageOracle},
	provider::ProviderRoute,
	refresh::{CredentialRefresher, RefreshFailure},
	select::{CredentialCandidates, CredentialLease, CredentialPool, LeaseSource},
	stack::builder::{RouteDependencies, RouteStackConfig},
	tap::FrameSink,
};
use omp_llm_types::{
	CacheHint, ChatOutcome, ChatParams, ChatRequest, ContextRef, Item, ItemKind, Message, Part,
	Props, Revision, Role, StopReason, Thread, ThreadDelta, TurnErrorKind, TurnEvent,
	facet::{Chat, Error, Executor},
};
use omp_proto::inference::{
	v1 as pb,
	v1::{TurnEvent as ProtoTurnEvent, TurnRequest as ProtoTurnRequest},
};
use parking_lot::{Mutex, RwLock};
use smallvec::smallvec;
use tower::service_fn;

struct Allow;

impl UsageOracle for Allow {
	fn admit(&self, _model: &str) -> Admission {
		Admission::Allow
	}
}

struct OneCredential;

impl CredentialPool for OneCredential {
	fn candidates(&self, _model: &str) -> CredentialCandidates {
		smallvec![7]
	}
}

struct RankedCredentials(Arc<Mutex<CredentialCandidates>>);

impl CredentialPool for RankedCredentials {
	fn candidates(&self, _model: &str) -> CredentialCandidates {
		self.0.lock().clone()
	}
}

struct Leases;

impl LeaseSource for Leases {
	fn lease(&self, id: u64) -> Option<CredentialLease> {
		Some(CredentialLease::new("openai", id, 1))
	}
}

struct DynamicLeases {
	generation: Arc<AtomicU64>,
}

impl LeaseSource for DynamicLeases {
	fn lease(&self, id: u64) -> Option<CredentialLease> {
		Some(CredentialLease::new("openai", id, self.generation.load(Ordering::SeqCst)))
	}
}

struct Fresh;

impl CredentialRefresher for Fresh {
	fn expires_at_ms(&self) -> Option<u64> {
		None
	}

	fn refresh(
		&self,
		_force: bool,
	) -> Pin<Box<dyn Future<Output = Result<(), RefreshFailure>> + Send + '_>> {
		Box::pin(async { Ok(()) })
	}
}

struct NoRepair;

impl RequestRepair for NoRepair {
	fn strip(
		&self,
		_req: &ProtoTurnRequest,
		_feature: Feature,
		_classification: &Classification,
	) -> Option<ProtoTurnRequest> {
		None
	}
}

struct NoopSink;

impl FrameSink for NoopSink {
	fn on_request(&self, _request: &ProtoTurnRequest) {}

	fn on_frame(&self, _frame: &ProtoTurnEvent) {}

	fn on_end(&self) {}
}

#[derive(Debug, thiserror::Error)]
#[error("stale credential generation")]
struct StaleGeneration;

struct Available;

impl CredentialView for Available {
	fn availability(&self, _provider: &str) -> Availability {
		Availability::Available
	}
}

struct RecordingChat {
	models: Arc<Mutex<Vec<Str>>>,
}

#[async_trait]
impl Chat for RecordingChat {
	async fn turn(
		&self,
		request: ChatRequest,
		_executor: Option<Arc<dyn Executor>>,
	) -> Result<futures::stream::BoxStream<'static, TurnEvent>, Error> {
		self.models.lock().push(request.model);
		Ok(Box::pin(stream::iter([TurnEvent::Outcome(
			ChatOutcome::builder()
				.output(Vec::new())
				.stop(StopReason::EndTurn)
				.unsupported(Vec::new())
				.provider(Str::new_static("embedded-provider"))
				.model(Str::new_static("canonical-model"))
				.props(Props::default())
				.build(),
		)])))
	}
}

struct FederatedScript {
	requests: Arc<Mutex<Vec<ChatRequest>>>,
	dispatch: Arc<AtomicUsize>,
	dropped:  Arc<AtomicBool>,
}

#[async_trait]
impl Chat for FederatedScript {
	async fn turn(
		&self,
		request: ChatRequest,
		_executor: Option<Arc<dyn Executor>>,
	) -> Result<futures::stream::BoxStream<'static, TurnEvent>, Error> {
		self.requests.lock().push(request);
		let dispatch = self.dispatch.fetch_add(1, Ordering::SeqCst);
		if dispatch == 1 {
			return Ok(Box::pin(TrackedPending { dropped: Arc::clone(&self.dropped) }));
		}
		Ok(Box::pin(stream::iter([TurnEvent::Outcome(
			ChatOutcome::builder()
				.output(Vec::new())
				.stop(StopReason::EndTurn)
				.unsupported(Vec::new())
				.provider(Str::new_static("federated"))
				.model(Str::new_static("remote-model"))
				.props(Props::default())
				.build(),
		)])))
	}
}

struct TrackedPending {
	dropped: Arc<AtomicBool>,
}

impl Stream for TrackedPending {
	type Item = TurnEvent;

	fn poll_next(self: Pin<&mut Self>, _cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
		Poll::Pending
	}
}

impl Drop for TrackedPending {
	fn drop(&mut self) {
		self.dropped.store(true, Ordering::SeqCst);
	}
}

#[tokio::test]
async fn injected_route_resolves_catalog_model_for_foreign_facets() {
	let resolver = resolver(vec![card("alias/model", "embedded-provider", "canonical-model")]);
	let seen = Arc::new(Mutex::new(Vec::new()));
	let provider = provider("embedded-provider", TransportId::Embedded);
	let result = register(&resolver, [&provider], SpecializedChats {
		embedded: Some(Arc::new(RecordingChat { models: Arc::clone(&seen) })),
		..SpecializedChats::default()
	})
	.expect("embedded route registers");
	assert_eq!(result.registered, 1);

	let facade = RoutedChat::new(Arc::clone(&resolver));
	let mut events = facade
		.turn(request("alias/model"), None)
		.await
		.expect("catalog alias resolves");
	assert!(matches!(events.next().await, Some(TurnEvent::Outcome(_))));
	assert_eq!(&*seen.lock(), &[Str::new_static("canonical-model")]);
}

#[tokio::test]
async fn production_embedded_bridge_constructs_portably_without_a_configured_text_facet() {
	let resolver = resolver(vec![card("embedded/model", "embedded", "matrix-model")]);
	let provider = provider("embedded", TransportId::Embedded);
	let inference = Inference::builder()
		.build()
		.await
		.expect("construct production local runtime without optional facets");
	register(&resolver, [&provider], SpecializedChats {
		embedded: Some(Arc::new(Embedded::new(Arc::new(inference)))),
		..SpecializedChats::default()
	})
	.expect("register production embedded bridge");

	let error = RoutedChat::new(resolver)
		.turn(request("embedded/model"), None)
		.await
		.err()
		.expect("an unconfigured local text facet is rejected");
	assert!(matches!(error, Error::Unsupported(_)));
}

#[test]
fn known_http_and_injected_specialized_rows_register_from_catalog() {
	let resolver = resolver(Vec::new());
	let http = provider("openai", TransportId::OpenAiChat);
	let cursor = provider("cursor", TransportId::Cursor);
	let devin = provider("devin", TransportId::Devin);
	let gitlab = provider("gitlab-duo-agent", TransportId::GitLabDuoWorkflow);
	let embedded = provider("embedded", TransportId::Embedded);
	let omp = provider("federated", TransportId::Omp);
	let chat =
		|| Arc::new(RecordingChat { models: Arc::new(Mutex::new(Vec::new())) }) as Arc<dyn Chat>;
	let result =
		register(&resolver, [&http, &cursor, &devin, &gitlab, &embedded, &omp], SpecializedChats {
			by_provider:         BTreeMap::new(),
			cursor:              Some(chat()),
			devin:               Some(chat()),
			gitlab_duo_workflow: Some(chat()),
			embedded:            Some(chat()),
			omp:                 Some(chat()),
		})
		.expect("HTTP and specialized transport families register");
	assert_eq!(result.registered, 6);
}

#[tokio::test]
async fn mixed_provider_cards_select_once_built_wire_stacks() {
	let cards = vec![
		wire_card(
			"github-copilot/claude-sonnet-4.5",
			"claude-sonnet-4.5",
			TransportId::AnthropicMessages,
			"https://api.githubcopilot.com",
		),
		wire_card(
			"github-copilot/gpt-4.1",
			"gpt-4.1",
			TransportId::OpenAiChat,
			"https://api.githubcopilot.com",
		),
		wire_card(
			"github-copilot/gpt-5",
			"gpt-5",
			TransportId::OpenAiResponses,
			"https://api.githubcopilot.com",
		),
		wire_card_for(
			"opencode-zen/claude-sonnet-4-5",
			"opencode-zen",
			"claude-sonnet-4-5",
			TransportId::AnthropicMessages,
			"https://opencode.ai/zen",
		),
		wire_card_for(
			"opencode-zen/big-pickle",
			"opencode-zen",
			"big-pickle",
			TransportId::OpenAiChat,
			"https://opencode.ai/zen/v1",
		),
		wire_card_for(
			"opencode-zen/gpt-5",
			"opencode-zen",
			"gpt-5",
			TransportId::OpenAiResponses,
			"https://opencode.ai/zen/v1",
		),
		wire_card_for(
			"opencode-zen/gemini-3-flash",
			"opencode-zen",
			"gemini-3-flash",
			TransportId::GoogleGenAi,
			"https://opencode.ai/zen/v1",
		),
	];
	let resolver = resolver(cards);
	let mut github = provider("github-copilot", TransportId::OpenAiResponses);
	github.base_url = "https://api.githubcopilot.com".into();
	let mut zen = provider("opencode-zen", TransportId::AnthropicMessages);
	zen.base_url = "https://opencode.ai/zen".into();
	let built = Arc::new(AtomicUsize::new(0));
	let requests = Arc::new(Mutex::new(Vec::new()));
	let capture = Arc::clone(&requests);
	let egress = service_fn(move |request: Request<Body>| {
		let capture = Arc::clone(&capture);
		async move {
			let path = request.uri().path().to_owned();
			let body = request.into_body().collect().await.unwrap().to_bytes();
			let body: serde_json::Value = serde_json::from_slice(&body).expect("JSON request body");
			capture.lock().push((path, body));
			Ok::<_, Infallible>(
				Response::builder()
					.header(header::CONTENT_TYPE, "text/event-stream")
					.body(Full::new(Bytes::new()))
					.expect("empty SSE fixture"),
			)
		}
	});
	let built_by_dependencies = Arc::clone(&built);
	let registration = register_production_routes(
		&resolver,
		[&github, &zen],
		egress,
		move |provider| {
			assert!(matches!(provider.id.as_str(), "github-copilot" | "opencode-zen"));
			built_by_dependencies.fetch_add(1, Ordering::SeqCst);
			unused_dependencies(provider)
		},
		|_| RouteStackConfig::default(),
		|_| ProviderRoute::default(),
		SpecializedChats::default(),
	)
	.expect("mixed provider routes register");
	assert_eq!(registration.registered, 7);
	assert_eq!(built.load(Ordering::SeqCst), 7);

	let chat = RoutedChat::new(Arc::clone(&resolver));
	for model in [
		"github-copilot/claude-sonnet-4.5",
		"github-copilot/gpt-4.1",
		"github-copilot/gpt-5",
		"github-copilot/gpt-5",
		"opencode-zen/claude-sonnet-4-5",
		"opencode-zen/big-pickle",
		"opencode-zen/gpt-5",
		"opencode-zen/gemini-3-flash",
	] {
		let stream = chat
			.turn(request(model), None)
			.await
			.expect("wire request resolves");
		let mut stream = Box::pin(stream);
		let _ = futures::poll!(stream.next());
		drop(stream);
	}
	assert_eq!(built.load(Ordering::SeqCst), 7, "dispatch must reuse prebuilt stacks");

	let requests = requests.lock();
	assert_eq!(requests.len(), 8);
	assert_eq!(requests[0].0, "/v1/messages");
	assert!(requests[0].1.get("messages").is_some());
	assert_eq!(requests[1].0, "/chat/completions");
	assert!(requests[1].1.get("messages").is_some());
	assert_eq!(requests[2].0, "/responses");
	assert!(requests[2].1.get("input").is_some());
	assert_eq!(requests[3].0, "/responses");
	assert_eq!(requests[4].0, "/zen/v1/messages");
	assert!(requests[4].1.get("messages").is_some());
	assert_eq!(requests[5].0, "/zen/v1/chat/completions");
	assert!(requests[5].1.get("messages").is_some());
	assert_eq!(requests[6].0, "/zen/v1/responses");
	assert!(requests[6].1.get("input").is_some());
	assert_eq!(requests[7].0, "/zen/v1/models/gemini-3-flash:streamGenerateContent");
	assert!(requests[7].1.get("contents").is_some());
}

#[test]
fn missing_specialized_and_empty_catalog_are_explicit_errors() {
	let resolver = resolver(Vec::new());
	let cursor = provider("cursor", TransportId::Cursor);
	assert_eq!(
		register(&resolver, [&cursor], SpecializedChats::default()),
		Err(RouteRegistrationError::MissingSpecialized {
			provider:  Str::new_static("cursor"),
			transport: TransportId::Cursor,
		})
	);
	assert_eq!(
		register(&resolver, std::iter::empty(), SpecializedChats::default()),
		Err(RouteRegistrationError::NoUsableRoutes)
	);
}

#[tokio::test]
async fn production_session_affinity_is_transactional_across_routes_and_reseeds() {
	let resolver = resolver(vec![
		card("openai/model", "openai", "model"),
		card("federated/model", "federated", "remote-model"),
	]);
	let mut http_provider = provider("openai", TransportId::OpenAiResponses);
	http_provider.compat.stateful_response_chaining = true;
	let federated_provider = provider("federated", TransportId::Omp);
	let ranking = Arc::new(Mutex::new(smallvec![7, 8]));
	let issued_generation = Arc::new(AtomicU64::new(1));
	let accepted_generation = Arc::new(AtomicU64::new(1));
	let observed = Arc::new(Mutex::new(Vec::new()));
	let dispatches = Arc::new(AtomicUsize::new(0));
	let inspect = Arc::clone(&observed);
	let accepted = Arc::clone(&accepted_generation);
	let count = Arc::clone(&dispatches);
	let egress = service_fn(move |request: Request<Body>| {
		let inspect = Arc::clone(&inspect);
		let accepted = Arc::clone(&accepted);
		let count = Arc::clone(&count);
		async move {
			let lease = request
				.extensions()
				.get::<CredentialLease>()
				.expect("selection installs a canonical lease")
				.clone();
			let body = request.into_body().collect().await.unwrap().to_bytes();
			inspect.lock().push((lease.clone(), body));
			if lease.generation() != accepted.load(Ordering::SeqCst) {
				return Err(StaleGeneration);
			}
			let response_id = count.fetch_add(1, Ordering::SeqCst) + 1;
			let added = serde_json::json!({
				"type": "response.output_item.added",
				"output_index": 0,
				"item": {
					"id": format!("msg_{response_id}"),
					"type": "message",
					"role": "assistant",
					"status": "in_progress",
					"content": [],
				},
			});
			let delta = serde_json::json!({
				"type": "response.output_text.delta",
				"output_index": 0,
				"item_id": format!("msg_{response_id}"),
				"content_index": 0,
				"delta": format!("completed {response_id}"),
			});
			let completed = serde_json::json!({
				"type": "response.completed",
				"response": {
					"id": format!("resp_{response_id}"),
					"status": "completed",
					"output": [{
						"id": format!("msg_{response_id}"),
						"type": "message",
						"role": "assistant",
						"status": "completed",
						"content": [{
							"type": "output_text",
							"text": format!("completed {response_id}"),
							"annotations": [],
						}],
					}],
				},
			});
			let payload = format!("data: {added}\n\ndata: {delta}\n\ndata: {completed}\n\n");
			Ok::<_, StaleGeneration>(
				Response::builder()
					.header(header::CONTENT_TYPE, "text/event-stream")
					.body(Full::new(Bytes::from(payload)))
					.expect("valid SSE response"),
			)
		}
	});
	let federation_requests = Arc::new(Mutex::new(Vec::new()));
	let federation_dispatch = Arc::new(AtomicUsize::new(0));
	let federation_dropped = Arc::new(AtomicBool::new(false));
	let dependency_ranking = Arc::clone(&ranking);
	let dependency_generation = Arc::clone(&issued_generation);
	let blocks = Arc::new(Mutex::new(BlockTable::default()));
	blocks
		.lock()
		.block(omp_llm_error::BlockKey::credential("8"), 0, u64::MAX);
	let dependency_blocks = Arc::clone(&blocks);
	register_production_routes(
		&resolver,
		[&http_provider, &federated_provider],
		egress,
		move |_| RouteDependencies {
			usage:          Arc::new(Allow),
			credentials:    Arc::new(RankedCredentials(Arc::clone(&dependency_ranking))),
			leases:         Arc::new(DynamicLeases { generation: Arc::clone(&dependency_generation) }),
			refresher:      Arc::new(Fresh),
			repair:         Arc::new(NoRepair),
			observer:       Arc::new(NoopSink),
			usage_observer: Arc::new(omp_llm_tower::stack::meter::NoopUsageObserver),
			blocks:         Arc::clone(&dependency_blocks),
		},
		|_| {
			let mut config = RouteStackConfig::default();
			config.recovery.budget = omp_llm_error::policy::RetryBudget::new(1, 0, 0, 0);
			config.compat.stateful_response_chaining = true;
			config
		},
		|_| ProviderRoute::default(),
		SpecializedChats {
			omp: Some(Arc::new(FederatedScript {
				requests: Arc::clone(&federation_requests),
				dispatch: Arc::clone(&federation_dispatch),
				dropped:  Arc::clone(&federation_dropped),
			})),
			..SpecializedChats::default()
		},
	)
	.expect("production routes register atomically");

	let contexts = Arc::new(ContextStore::default());
	let engine = TurnEngine::new(Arc::clone(&contexts), Arc::clone(&resolver));
	let (_, initial) = successful_turn(
		&engine,
		seed_open("turn-initial", "ctx-production", "openai/model", "session", thread("initial")),
	)
	.await;
	let initial_revision = initial.revision.expect("initial revision");

	*ranking.lock() = smallvec![8, 7];
	let (_, cached) = successful_turn(
		&engine,
		incremental_open(
			"turn-cached",
			"ctx-production",
			"openai/model",
			"session",
			initial_revision,
			"cached",
		),
	)
	.await;
	let cached_revision = cached.revision.expect("cached revision");

	accepted_generation.store(2, Ordering::SeqCst);
	let stale_open = incremental_open(
		"turn-stale-generation",
		"ctx-production",
		"openai/model",
		"session",
		cached_revision.clone(),
		"stale",
	);
	let stale = terminal_event(&engine, stale_open.clone()).await;
	assert!(matches!(stale, TurnEvent::Error(_)));
	assert_eq!(contexts.revision("ctx-production").unwrap(), cached_revision);

	issued_generation.store(2, Ordering::SeqCst);
	let (_, renewed) = successful_turn(&engine, stale_open).await;
	let renewed_revision = renewed.revision.expect("renewed revision");

	let (_, handoff) = successful_turn(
		&engine,
		incremental_open(
			"turn-handoff",
			"ctx-production",
			"federated/model",
			"session",
			renewed_revision,
			"handoff",
		),
	)
	.await;
	let handoff_revision = handoff.revision.expect("handoff revision");

	let mut cancelled = open_stream(
		&engine,
		incremental_open(
			"turn-cancelled",
			"ctx-production",
			"federated/model",
			"session",
			handoff_revision.clone(),
			"cancelled",
		),
	)
	.await;
	assert!(matches!(next_event(&mut cancelled).await, TurnEvent::Accepted { replay: false }));
	drop(cancelled);
	assert!(federation_dropped.load(Ordering::SeqCst));
	assert_eq!(contexts.revision("ctx-production").unwrap(), handoff_revision);

	assert!(contexts.evict("ctx-production"));
	let need_full = terminal_event(
		&engine,
		incremental_open(
			"turn-need-full",
			"ctx-production",
			"federated/model",
			"session",
			handoff_revision,
			"missing",
		),
	)
	.await;
	assert!(matches!(need_full, TurnEvent::Error(error) if error.kind == TurnErrorKind::NeedFull));

	let reseed = seed_open(
		"turn-reseed",
		"ctx-production",
		"federated/model",
		"session",
		thread("full authoritative history"),
	);
	let (replayed, final_outcome) = successful_turn(&engine, reseed.clone()).await;
	assert!(!replayed);
	let final_revision = final_outcome.revision.clone().expect("final revision");
	let (replayed, replayed_outcome) = successful_turn(&engine, reseed).await;
	assert!(replayed);
	assert_eq!(replayed_outcome, final_outcome);
	assert_eq!(contexts.revision("ctx-production").unwrap(), final_revision);
	assert_eq!(final_outcome.provider, "federated");
	assert_eq!(final_outcome.model, "remote-model");

	let observed = observed.lock();
	assert_eq!(
		observed
			.iter()
			.map(|(lease, _)| (lease.credential_id(), lease.generation()))
			.collect::<Vec<_>>(),
		vec![(7, 1), (7, 1), (7, 1), (7, 2)]
	);
	let initial_body: serde_json::Value = serde_json::from_slice(&observed[0].1).unwrap();
	let cached_body: serde_json::Value = serde_json::from_slice(&observed[1].1).unwrap();
	assert_eq!(cached_body["previous_response_id"], "resp_1");
	assert_eq!(initial_body["prompt_cache_key"], cached_body["prompt_cache_key"]);
	assert!(
		initial_body["prompt_cache_key"]
			.as_str()
			.is_some_and(|key| key.starts_with("omp_cache_"))
	);
	drop(observed);

	let federation = federation_requests.lock();
	assert_eq!(federation.len(), 3, "replay and NEED_FULL do not dispatch stale work");
	assert_eq!(
		federation[0].cache.as_ref().map(|cache| &cache.session_key),
		federation[2].cache.as_ref().map(|cache| &cache.session_key),
		"full reseed recovers the deterministic route-scoped cache identity"
	);
	assert!(
		federation[0]
			.provider_options
			.as_ref()
			.and_then(|options| options.get_ns("openai", "previous_response_id"))
			.is_none(),
		"provider state is cleared at the federation route handoff"
	);
}

#[tokio::test]
async fn unsupported_model_is_rejected_before_transport_dispatch() {
	let resolver = resolver(vec![card("known", "embedded-provider", "canonical-model")]);
	let calls = Arc::new(Mutex::new(Vec::new()));
	let provider = provider("embedded-provider", TransportId::Embedded);
	register(&resolver, [&provider], SpecializedChats {
		embedded: Some(Arc::new(RecordingChat { models: Arc::clone(&calls) })),
		..SpecializedChats::default()
	})
	.expect("route registers");
	let error = RoutedChat::new(resolver)
		.turn(request("unknown"), None)
		.await
		.err()
		.expect("unknown model is rejected");
	assert!(matches!(error, Error::Unsupported(_)));
	assert!(calls.lock().is_empty());
}

fn thread(text: &str) -> Thread {
	Thread::builder().items(vec![message(text)]).build()
}

fn message(text: &str) -> Item {
	Item::builder()
		.seq(0)
		.kind(ItemKind::Message(
			Message::builder()
				.role(Role::User)
				.parts(vec![Part::Text(Str::new(text))])
				.build(),
		))
		.props(Props::default())
		.build()
}

fn turn_params(model: &str, session: &str) -> pb::ChatParams {
	ChatParams::builder()
		.model(Str::new(model))
		.tools(Vec::new())
		.cache(
			CacheHint::builder()
				.session_key(Str::new(session))
				.build(),
		)
		.build()
		.into()
}

fn seed_open(
	turn_id: &str,
	context_id: &str,
	model: &str,
	session: &str,
	thread: Thread,
) -> pb::TurnFrame {
	pb::TurnFrame {
		frame: Some(pb::turn_frame::Frame::Open(pb::TurnRequest {
			turn_id:  turn_id.into(),
			input:    Some(pb::turn_request::Input::Seed(pb::Seed {
				context_id: context_id.into(),
				thread:     Some(thread.into()),
			})),
			params:   Some(turn_params(model, session)),
			executor: None,
			props:    None,
		})),
	}
}

fn incremental_open(
	turn_id: &str,
	context_id: &str,
	model: &str,
	session: &str,
	expected: Revision,
	append: &str,
) -> pb::TurnFrame {
	pb::TurnFrame {
		frame: Some(pb::turn_frame::Frame::Open(pb::TurnRequest {
			turn_id:  turn_id.into(),
			input:    Some(pb::turn_request::Input::Incremental(pb::Incremental {
				context: Some(
					ContextRef::builder()
						.context_id(Str::new(context_id))
						.expected(expected)
						.build()
						.into(),
				),
				delta:   Some(
					ThreadDelta::builder()
						.append(vec![message(append)])
						.build()
						.into(),
				),
			})),
			params:   Some(turn_params(model, session)),
			executor: None,
			props:    None,
		})),
	}
}

async fn open_stream(engine: &TurnEngine, open: pb::TurnFrame) -> TurnStream {
	engine
		.turn_frames(stream::iter([Ok(open)]))
		.await
		.expect("turn stream")
}

async fn next_event(stream: &mut TurnStream) -> TurnEvent {
	stream
		.next()
		.await
		.expect("event")
		.expect("transport event")
		.try_into()
		.expect("canonical event")
}

async fn terminal_event(engine: &TurnEngine, open: pb::TurnFrame) -> TurnEvent {
	let mut stream = open_stream(engine, open).await;
	loop {
		let event = next_event(&mut stream).await;
		if matches!(event, TurnEvent::Outcome(_) | TurnEvent::Error(_)) {
			assert!(
				stream.next().await.is_none(),
				"turn emitted another event after its terminal event"
			);
			return event;
		}
	}
}

async fn successful_turn(engine: &TurnEngine, open: pb::TurnFrame) -> (bool, ChatOutcome) {
	let mut stream = open_stream(engine, open).await;
	let accepted = next_event(&mut stream).await;
	let TurnEvent::Accepted { replay } = accepted else {
		panic!("turn was not accepted: {accepted:?}");
	};
	let outcome = loop {
		let event = next_event(&mut stream).await;
		match event {
			TurnEvent::Outcome(outcome) => break outcome,
			TurnEvent::Error(error) => panic!("turn did not commit: {error:?}"),
			_ => {},
		}
	};
	assert!(
		stream.next().await.is_none(),
		"successful turn emitted another event after its outcome"
	);
	(replay, outcome)
}

fn register<'a>(
	resolver: &ChatResolver,
	providers: impl IntoIterator<Item = &'a ProviderEntry>,
	specialized: SpecializedChats,
) -> Result<omp_llm_gateway::routes::RouteRegistration, RouteRegistrationError> {
	let egress = service_fn(|_request: Request<Body>| async {
		Ok::<_, Infallible>(Response::new(Full::new(Bytes::new())))
	});
	register_production_routes(
		resolver,
		providers,
		egress,
		unused_dependencies,
		|_| RouteStackConfig::default(),
		|_| ProviderRoute::default(),
		specialized,
	)
}

fn unused_dependencies(_provider: &ProviderEntry) -> RouteDependencies {
	RouteDependencies {
		usage:          Arc::new(Allow),
		credentials:    Arc::new(OneCredential),
		leases:         Arc::new(Leases),
		refresher:      Arc::new(Fresh),
		repair:         Arc::new(NoRepair),
		observer:       Arc::new(NoopSink),
		usage_observer: Arc::new(omp_llm_tower::stack::meter::NoopUsageObserver),
		blocks:         Arc::new(Mutex::new(BlockTable::default())),
	}
}

fn resolver(cards: Vec<ModelCard>) -> Arc<ChatResolver> {
	let catalog = ModelCatalog::new(cards);
	Arc::new(ChatResolver::new(Arc::new(RwLock::new(Registry::new(&catalog, Arc::new(Available))))))
}

fn provider(id: &'static str, transport: TransportId) -> ProviderEntry {
	ProviderEntry::builder()
		.id(Str::new_static(id))
		.transport(transport)
		.base_url(Str::new_static("https://example.invalid"))
		.auth(AuthSpec::None)
		.facets(smallvec![Facet::Chat])
		.headers(BTreeMap::new())
		.compat(Compat::default())
		.build()
}

fn card(id: &'static str, provider: &'static str, model: &'static str) -> ModelCard {
	ModelCard::builder()
		.id(Str::new_static(id))
		.provider(Str::new_static(provider))
		.model(Str::new_static(model))
		.name(Str::new_static(model))
		.family(Str::new_static("test"))
		.facets(smallvec![Facet::Chat])
		.inputs(smallvec![Modality::Text])
		.outputs(smallvec![Modality::Text])
		.reasoning(false)
		.efforts(smallvec![])
		.context_window(4096)
		.max_output_tokens(1024)
		.pricing(smallvec![])
		.availability(Availability::Available)
		.source(Source::Configured)
		.blocked_until_ms(0)
		.deprecated(false)
		.updated_at_ms(0)
		.props(Props::default())
		.effort_routing(BTreeMap::new())
		.build()
}

fn wire_card(
	id: &'static str,
	model: &'static str,
	transport: TransportId,
	base_url: &'static str,
) -> ModelCard {
	wire_card_for(id, "github-copilot", model, transport, base_url)
}

fn wire_card_for(
	id: &'static str,
	provider: &'static str,
	model: &'static str,
	transport: TransportId,
	base_url: &'static str,
) -> ModelCard {
	let mut card = card(id, provider, model);
	card.wire = Some(ModelWire { transport, base_url: Some(base_url.into()) });
	card
}

fn request(model: &'static str) -> ChatRequest {
	ChatRequest::builder()
		.model(Str::new_static(model))
		.thread(Thread::default())
		.tools(Vec::new())
		.provider_options(Props::default())
		.build()
}
