//! Protocol proof for a registered catalog provider over a real HTTP/2 socket.

use std::{
	collections::{BTreeMap, VecDeque},
	convert::Infallible,
	future::Future,
	pin::Pin,
	sync::{Arc, Mutex},
	task::{Context, Poll},
	time::Duration,
};

use bytes::Bytes;
use futures::StreamExt;
use http::{HeaderMap, Method, Request, Response, StatusCode, Uri, header};
use http_body_util::{BodyExt, Full};
use hyper::{
	body::{Body as HttpBody, Frame, Incoming, SizeHint},
	server::conn::http2,
	service::service_fn as hyper_service_fn,
};
use hyper_util::{
	client::legacy::{Client, connect::HttpConnector},
	rt::{TokioExecutor, TokioIo},
};
use omp_core::SmolStr;
use omp_llm_catalog::{
	models::{Availability, Modality, ModelCard, ModelCatalog, Source},
	provider::load_builtin,
	registry::{CredentialView, Registry},
};
use omp_llm_egress::{
	auth_inject::{AuthInjectLayer, CredentialLease, CredentialSource},
	client::Body,
	retry::{Committed, PreCommitFailure, Replayable, RetryConfig, RetryLayer},
};
use omp_llm_error::{Classification, Evidence, Feature, classify, policy::BlockTable};
use omp_llm_gateway::{
	facade::{FacadeAuth, FacadeConfig, FacadeState, ModelsRepresentation, Router},
	routes::{SpecializedChats, register_production_routes},
	turn::{ChatResolver, RoutedChat},
};
use omp_llm_tower::{
	learn::RequestRepair,
	preflight::{Admission, UsageOracle},
	provider::ProviderRoute,
	refresh::{CredentialRefresher, RefreshFailure},
	select::{CredentialCandidates, CredentialPool, LeaseSource},
	stack::builder::{RouteDependencies, RouteStackConfig},
	tap::FrameSink,
};
use omp_llm_types::{
	ChatRequest, Item, ItemKind, Message, Part, Props, Role, Thread, TurnEvent,
	facet::{Chat, Facets},
};
use omp_proto::inference::v1::{TurnEvent as ProtoTurnEvent, TurnRequest as ProtoTurnRequest};
use omp_storage::blob::BlobStore;
use parking_lot::RwLock;
use tokio::{net::TcpListener, sync::oneshot};
use tower::{Layer, Service, ServiceExt, service_fn};

const PROVIDER: &str = "openrouter";
const MODEL: &str = "fixture-model";
const SECRET: &str = "loopback-provider-secret";

#[derive(Clone, Debug, Eq, PartialEq)]
struct CapturedRequest {
	method:  Method,
	uri:     Uri,
	headers: HeaderMap,
	body:    Bytes,
}

struct FixtureState {
	requests:     Mutex<Vec<CapturedRequest>>,
	cancelled_tx: Mutex<Option<oneshot::Sender<()>>>,
}

impl FixtureState {
	async fn respond(
		self: Arc<Self>,
		request: Request<Incoming>,
	) -> Result<Response<FixtureBody>, Infallible> {
		let (parts, body) = request.into_parts();
		let body = body
			.collect()
			.await
			.expect("read HTTP/2 request body")
			.to_bytes();
		let attempt = {
			let mut requests = self.requests.lock().expect("request capture lock");
			requests.push(CapturedRequest {
				method: parts.method,
				uri: parts.uri,
				headers: parts.headers,
				body,
			});
			requests.len()
		};

		if attempt == 1 {
			return Ok(Response::builder()
				.status(StatusCode::SERVICE_UNAVAILABLE)
				.body(FixtureBody::complete())
				.expect("503 response"));
		}

		let chunks = [
			Bytes::from_static(
				b"data: {\"id\":\"chatcmpl_h2\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hel\"},\"finish_reason\":null}]}\n\n",
			),
			Bytes::from_static(
				b"data: {\"id\":\"chatcmpl_h2\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":null}]}\n\n",
			),
			Bytes::from_static(
				b"data: {\"id\":\"chatcmpl_h2\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
			),
			Bytes::from_static(
				b"data: {\"id\":\"chatcmpl_h2\",\"object\":\"chat.completion.chunk\",\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5,\"prompt_tokens_details\":{\"cached_tokens\":1}}}\n\n",
			),
			Bytes::from_static(b"data: [DONE]\n\n"),
		];
		let body = if attempt == 2 || attempt == 4 {
			FixtureBody::finite(chunks)
		} else {
			let cancelled = self
				.cancelled_tx
				.lock()
				.expect("cancellation sender lock")
				.take();
			FixtureBody::streaming([chunks[0].clone()], cancelled)
		};
		Ok(Response::builder()
			.status(StatusCode::OK)
			.header(header::CONTENT_TYPE, "text/event-stream")
			.body(body)
			.expect("streaming response"))
	}
}

struct FixtureBody {
	chunks:    VecDeque<Bytes>,
	hangs:     bool,
	cancelled: Option<oneshot::Sender<()>>,
}

impl FixtureBody {
	fn complete() -> Self {
		Self { chunks: VecDeque::new(), hangs: false, cancelled: None }
	}

	fn finite(chunks: impl IntoIterator<Item = Bytes>) -> Self {
		Self { chunks: chunks.into_iter().collect(), hangs: false, cancelled: None }
	}

	fn streaming(
		chunks: impl IntoIterator<Item = Bytes>,
		cancelled: Option<oneshot::Sender<()>>,
	) -> Self {
		Self { chunks: chunks.into_iter().collect(), hangs: true, cancelled }
	}
}

impl Drop for FixtureBody {
	fn drop(&mut self) {
		if let Some(cancelled) = self.cancelled.take() {
			let _ = cancelled.send(());
		}
	}
}

impl HttpBody for FixtureBody {
	type Data = Bytes;
	type Error = Infallible;

	fn poll_frame(
		mut self: Pin<&mut Self>,
		_cx: &mut Context<'_>,
	) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
		if let Some(chunk) = self.chunks.pop_front() {
			return Poll::Ready(Some(Ok(Frame::data(chunk))));
		}
		if self.hangs {
			Poll::Pending
		} else {
			Poll::Ready(None)
		}
	}

	fn is_end_stream(&self) -> bool {
		!self.hangs && self.chunks.is_empty()
	}

	fn size_hint(&self) -> SizeHint {
		SizeHint::default()
	}
}

#[derive(Clone)]
struct Credentials;

impl CredentialView for Credentials {
	fn availability(&self, _provider: &str) -> Availability {
		Availability::Available
	}
}

impl CredentialSource for Credentials {
	type Error = Infallible;

	fn lease(&self, _provider: &str) -> Result<Option<CredentialLease>, Self::Error> {
		panic!("the routed canonical lease must be redeemed without a second selection")
	}

	fn apply(
		&self,
		lease: &CredentialLease,
		request: &mut Request<Body>,
	) -> Result<(), Self::Error> {
		assert_eq!(lease.provider(), PROVIDER);
		assert_eq!(lease.credential_id(), 17);
		assert_eq!(lease.generation(), 4);
		request.headers_mut().insert(
			header::AUTHORIZATION,
			http::HeaderValue::from_static("Bearer loopback-provider-secret"),
		);
		Ok(())
	}

	fn refresh(
		&self,
		_lease: CredentialLease,
	) -> impl Future<Output = Result<CredentialLease, Self::Error>> + Send + 'static {
		async { unreachable!("the fixture never returns 401") }
	}
}

struct Allow;
impl UsageOracle for Allow {
	fn admit(&self, _model: &str) -> Admission {
		Admission::Allow
	}
}

struct OneCredential;
impl CredentialPool for OneCredential {
	fn candidates(&self, _model: &str) -> CredentialCandidates {
		let mut candidates = CredentialCandidates::new();
		candidates.push(17);
		candidates
	}
}

struct Leases;
impl LeaseSource for Leases {
	fn lease(&self, id: u64) -> Option<CredentialLease> {
		Some(CredentialLease::new(PROVIDER, id, 4))
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
		_request: &ProtoTurnRequest,
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

fn dependencies() -> RouteDependencies {
	RouteDependencies {
		usage:          Arc::new(Allow),
		credentials:    Arc::new(OneCredential),
		leases:         Arc::new(Leases),
		refresher:      Arc::new(Fresh),
		repair:         Arc::new(NoRepair),
		observer:       Arc::new(NoopSink),
		usage_observer: Arc::new(omp_llm_tower::stack::meter::NoopUsageObserver),
		blocks:         Arc::new(parking_lot::Mutex::new(BlockTable::default())),
	}
}

fn model_card() -> ModelCard {
	ModelCard::builder()
		.id(SmolStr::new_static(MODEL))
		.provider(SmolStr::new_static(PROVIDER))
		.model(SmolStr::new_static(MODEL))
		.name(SmolStr::new_static(MODEL))
		.family(SmolStr::new_static("fixture"))
		.facets(
			[omp_llm_catalog::provider::Facet::Chat]
				.into_iter()
				.collect(),
		)
		.inputs([Modality::Text].into_iter().collect())
		.outputs([Modality::Text].into_iter().collect())
		.reasoning(false)
		.efforts(Default::default())
		.context_window(4_096)
		.max_output_tokens(1_024)
		.pricing(Default::default())
		.availability(Availability::Available)
		.source(Source::Configured)
		.blocked_until_ms(0)
		.deprecated(false)
		.updated_at_ms(0)
		.props(Props::default())
		.effort_routing(BTreeMap::new())
		.build()
}

fn turn_request() -> ChatRequest {
	let message = Message::builder()
		.role(Role::User)
		.parts(vec![Part::Text("hello over h2".into())])
		.build();
	let item = Item::builder()
		.seq(0)
		.kind(ItemKind::Message(message))
		.props(Props::default())
		.build();
	ChatRequest::builder()
		.model(SmolStr::new_static(MODEL))
		.thread(Thread::builder().items(vec![item]).build())
		.tools(Vec::new())
		.provider_options(Props::default())
		.build()
}

#[tokio::test]
async fn registered_provider_replays_before_commit_decodes_usage_and_resets_http2_stream() {
	let listener = TcpListener::bind("127.0.0.1:0")
		.await
		.expect("bind loopback HTTP/2 fixture");
	let address = listener.local_addr().expect("loopback address");
	let (cancelled_tx, cancelled_rx) = oneshot::channel();
	let state = Arc::new(FixtureState {
		requests:     Mutex::new(Vec::new()),
		cancelled_tx: Mutex::new(Some(cancelled_tx)),
	});
	let server_state = Arc::clone(&state);
	let server = tokio::spawn(async move {
		let (socket, _) = listener.accept().await.expect("accept provider connection");
		http2::Builder::new(TokioExecutor::new())
			.serve_connection(
				TokioIo::new(socket),
				hyper_service_fn(move |request| Arc::clone(&server_state).respond(request)),
			)
			.await
	});

	let mut connector = HttpConnector::new();
	connector.enforce_http(true);
	let hyper = Client::builder(TokioExecutor::new())
		.http2_only(true)
		.build(connector);
	let authenticated = AuthInjectLayer::new(Credentials).layer(hyper);
	let classified = service_fn(move |request: Request<Body>| {
		let mut authenticated = authenticated.clone();
		async move {
			let response = authenticated
				.ready()
				.await
				.map_err(|error| PreCommitFailure::new(error.to_string(), Classification::default()))?
				.call(request)
				.await
				.map_err(|error| PreCommitFailure::new(error.to_string(), Classification::default()))?;
			if response.status() == StatusCode::SERVICE_UNAVAILABLE {
				Err(PreCommitFailure::new(
					"provider returned HTTP status 503".to_owned(),
					classify(&Evidence::http(503, "")),
				))
			} else {
				Ok(Committed::new(response))
			}
		}
	});
	let retry = RetryLayer::with_seed(
		RetryConfig {
			base_delay:  Duration::ZERO,
			max_delay:   Duration::ZERO,
			max_retries: 1,
			jitter:      0.0,
		},
		1,
	)
	.layer(classified);
	let egress = service_fn(move |request: Request<Body>| {
		let mut retry = retry.clone();
		async move {
			ServiceExt::ready(&mut retry).await?;
			Service::call(&mut retry, Replayable::buffered(request))
				.await
				.map(Committed::into_inner)
		}
	});

	let mut provider = load_builtin()
		.expect("built-in provider catalog")
		.remove(PROVIDER)
		.expect("OpenRouter catalog entry");
	provider.base_url = format!("http://{address}/v1").into();
	provider
		.headers
		.insert("x-catalog-proof".into(), "registered".into());
	let catalog = ModelCatalog::new(vec![model_card()]);
	let registry = Arc::new(RwLock::new(Registry::new(&catalog, Arc::new(Credentials))));
	let resolver = Arc::new(ChatResolver::new(registry));
	let registration = register_production_routes(
		&resolver,
		[&provider],
		egress,
		|_| dependencies(),
		|_| RouteStackConfig::default(),
		|_| ProviderRoute::default(),
		SpecializedChats::default(),
	)
	.expect("register catalog provider route");
	assert_eq!(registration.registered, 1);
	let chat = RoutedChat::new(resolver);
	let request = turn_request();

	let mut stream = chat
		.turn(request.clone(), None)
		.await
		.expect("503 is replayed before the first decoded event commits");
	let mut text = Vec::new();
	let mut outcome = None;
	while let Some(event) = stream.next().await {
		match event {
			TurnEvent::PartDelta { chunk, .. } => text.extend_from_slice(&chunk),
			TurnEvent::Outcome(value) => outcome = Some(value),
			TurnEvent::Error(error) => panic!("unexpected provider error: {}", error.detail),
			_ => {},
		}
	}
	assert_eq!(text, b"hello");
	let outcome = outcome.expect("terminal provider outcome");
	assert_eq!(outcome.provider, PROVIDER);
	assert_eq!(outcome.model, MODEL);
	let usage = outcome.usage.expect("provider usage");
	assert_eq!(usage.input_tokens, 3);
	assert_eq!(usage.output_tokens, 2);
	assert_eq!(usage.cache_read_tokens, 1);

	let mut cancelled_stream = chat
		.turn(request, None)
		.await
		.expect("cancellation probe starts");
	loop {
		match cancelled_stream
			.next()
			.await
			.expect("cancellation probe commits")
		{
			TurnEvent::PartStart { .. } | TurnEvent::PartDelta { .. } => break,
			TurnEvent::Outcome(_) | TurnEvent::Error(_) => {
				panic!("cancellation probe terminated before the response stream was live")
			},
			_ => {},
		}
	}
	drop(cancelled_stream);

	tokio::time::timeout(Duration::from_secs(1), cancelled_rx)
		.await
		.expect("client sent HTTP/2 cancellation/RST after the response stream was dropped")
		.expect("server observed dropped response stream");
	let directory = tempfile::tempdir().expect("temporary facade store");
	let facade_registry = Arc::new(RwLock::new(Registry::new(&catalog, Arc::new(Credentials))));
	let facade = Router::new(Arc::new(FacadeState {
		facets:   Arc::new(Facets { chat: Some(Arc::new(chat.clone())), ..Facets::default() }),
		registry: facade_registry,
		blobs:    Arc::new(BlobStore::open(directory.path()).expect("facade blob store")),
		auth:     FacadeAuth::new("gateway-token"),
		config:   FacadeConfig { models_representation: ModelsRepresentation::Auto },
	}));
	let facade_response = facade
		.route(
			Request::post("/v1/chat/completions")
				.header(header::AUTHORIZATION, "Bearer gateway-token")
				.header(header::CONTENT_TYPE, "application/json")
				.body(Full::new(Bytes::from_static(
					br#"{"model":"fixture-model","messages":[{"role":"user","content":"hello over h2"}]}"#,
				)))
				.expect("facade request"),
		)
		.await;
	assert_eq!(facade_response.status(), StatusCode::OK);
	let facade_json: serde_json::Value = serde_json::from_slice(
		&facade_response
			.into_body()
			.collect()
			.await
			.expect("facade response body")
			.to_bytes(),
	)
	.expect("facade JSON");
	assert_eq!(facade_json["object"], "chat.completion");
	assert_eq!(facade_json["choices"][0]["message"]["content"], "hello");
	assert_eq!(facade_json["model"], MODEL);
	assert_eq!(facade_json["usage"]["prompt_tokens"], 3);
	assert_eq!(facade_json["usage"]["completion_tokens"], 2);

	let requests = state.requests.lock().expect("request capture lock");
	assert_eq!(requests[0], requests[1], "retry must replay the exact encoded request");
	assert_eq!(requests[1], requests[2], "cancellation probe uses the same registered route");
	assert_eq!(requests.len(), 4);
	assert_eq!(requests[0].method, Method::POST);
	assert_eq!(requests[0].uri.path(), "/v1/chat/completions");
	assert_eq!(requests[0].headers[header::AUTHORIZATION], format!("Bearer {SECRET}"));
	assert_eq!(requests[0].headers["x-catalog-proof"], "registered");
	assert_eq!(requests[0].headers[header::ACCEPT], "text/event-stream");
	let encoded: serde_json::Value =
		serde_json::from_slice(&requests[0].body).expect("provider request JSON");
	assert_eq!(encoded["model"], MODEL);
	assert_eq!(encoded["stream"], true);
	assert_eq!(encoded["stream_options"]["include_usage"], true);

	drop(requests);
	server.abort();
}
