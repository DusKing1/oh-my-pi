//! Federation cancellation and credential-locality integration tests.
use std::{
	collections::BTreeMap,
	convert::Infallible,
	future::{self, Future},
	pin::Pin,
	sync::{
		Arc,
		atomic::{AtomicUsize, Ordering},
	},
	time::Duration,
};

use bytes::Bytes;
use futures::{Stream, StreamExt};
use http::Request as HttpRequest;
use omp_core::Str;
use omp_llm_broker::{
	source::{BrokerCredentialSource, CredentialRefresher},
	store::Store,
};
use omp_llm_catalog::{
	compat::Compat,
	models::{Availability, ModelCard, Source},
	provider::{AuthSpec, Facet, ProviderCatalog, ProviderEntry, TransportId},
	registry::{CredentialView, ListFilter, Registry},
};
use omp_llm_egress::{auth_inject::CredentialSource, client::Body};
use omp_llm_gateway::federation::FederatedProvider;
use omp_llm_transport::omp::OmpFederation;
use omp_llm_types::{ChatRequest, Props, Thread, TurnErrorKind, TurnEvent, facet::Chat};
use omp_proto::inference::v1::{
	self as pb,
	inference_client::InferenceClient,
	inference_server::{Inference, InferenceServer},
};
use parking_lot::Mutex;
use tokio::{
	io::{AsyncReadExt, AsyncWriteExt},
	net::{TcpListener, TcpStream},
	task::JoinHandle,
	time::timeout,
};
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{
	Request, Response, Status,
	transport::{Channel, Endpoint, Server},
};

const SENTINEL: &[u8] = b"OMP_TERMINAL_ONLY_4f9d8c3a_SECRET";

type RpcStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;

#[derive(Clone)]
struct TerminalGateway {
	credentials:        BrokerCredentialSource<NoRefresh>,
	turns:              Arc<AtomicUsize>,
	need_full_turns:    Arc<AtomicUsize>,
	credential_applied: Arc<AtomicUsize>,
	cancelled:          flume::Sender<String>,
}

struct AbortObserved {
	label:     String,
	cancelled: flume::Sender<String>,
}

impl Drop for AbortObserved {
	fn drop(&mut self) {
		let _ = self.cancelled.send(self.label.clone());
	}
}

struct AlwaysAvailable;
struct NoRefresh;

impl CredentialRefresher for NoRefresh {
	type Error = Infallible;

	#[allow(
		clippy::manual_async_fn,
		reason = "synchronous test refresher intentionally returns a zero-allocation ready future"
	)]
	fn refresh(
		&self,
		_credential_id: u64,
		_now_ms: u64,
	) -> impl Future<Output = Result<(), Self::Error>> + Send {
		future::ready(Ok(()))
	}
}

impl CredentialView for AlwaysAvailable {
	fn availability(&self, _provider: &str) -> Availability {
		Availability::Available
	}
}

#[tonic::async_trait]
impl Inference for TerminalGateway {
	type AttachGenerationStream = RpcStream<pb::GenerationStatus>;
	type GenerateImageStream = RpcStream<pb::ImageEvent>;
	type SpeakStream = RpcStream<pb::SpeakEvent>;
	type TurnStream = RpcStream<pb::TurnEvent>;
	type WatchModelsStream = RpcStream<pb::ModelEvent>;

	async fn turn(
		&self,
		request: Request<tonic::Streaming<pb::TurnFrame>>,
	) -> Result<Response<Self::TurnStream>, Status> {
		let mut request = request.into_inner();
		let Some(pb::turn_frame::Frame::Open(open)) =
			request.message().await?.and_then(|frame| frame.frame)
		else {
			return Err(Status::invalid_argument("first frame is not open"));
		};
		self.turns.fetch_add(1, Ordering::SeqCst);

		// Lease and apply the sealed credential only here, on the terminal host.
		// The resulting provider header is never placed on either gateway RPC.
		let lease = self
			.credentials
			.lease("terminal-provider")
			.map_err(|error| Status::internal(error.to_string()))?
			.ok_or_else(|| Status::internal("credential disappeared"))?;
		let mut provider_request = HttpRequest::builder()
			.uri("https://terminal.invalid/turn")
			.body(Body::new(Bytes::new()))
			.map_err(|error| Status::internal(error.to_string()))?;
		self
			.credentials
			.apply(&lease, &mut provider_request)
			.map_err(|error| Status::internal(error.to_string()))?;
		if provider_request
			.headers()
			.get(http::header::AUTHORIZATION)
			.is_some_and(|value| value.as_bytes().strip_prefix(b"Bearer ") == Some(SENTINEL))
		{
			self.credential_applied.fetch_add(1, Ordering::SeqCst);
		}

		let model = open
			.params
			.as_ref()
			.map(|params| params.model.clone())
			.ok_or_else(|| Status::invalid_argument("open frame has no chat params"))?;
		let cancelled = self.cancelled.clone();
		let need_full =
			model == "need-full" && self.need_full_turns.fetch_add(1, Ordering::SeqCst) == 0;
		let events = async_stream::try_stream! {
			// Install the observer before the first item. The client deliberately
			// drops immediately after that item, so placing it later would only
			// test a generator branch that cancellation never polled.
			let _abort_observed = model.starts_with("cancel-").then(|| AbortObserved {
				label: model.clone(),
				cancelled,
			});
			if need_full {
				yield pb::TurnEvent {
					event: Some(pb::turn_event::Event::Error(pb::TurnError {
						kind: pb::turn_error::Kind::NeedFull as i32,
						detail: "terminal gateway needs a seed".into(),
						..Default::default()
					})),
				};
				// A broken peer may continue after a terminal event. Federation
				// must stop polling it rather than forwarding a second turn.
				yield delta(b"must-not-cross-terminal");
			} else {
				yield delta(b"terminal-delta");
				if model.starts_with("cancel-") {
					loop {
						tokio::time::sleep(Duration::from_millis(5)).await;
						yield delta(b"x");
					}
				}
						 yield outcome();
			}
		};
		Ok(Response::new(Box::pin(events)))
	}

	async fn list_models(
		&self,
		_request: Request<pb::ListModelsRequest>,
	) -> Result<Response<pb::ListModelsResponse>, Status> {
		Ok(Response::new(pb::ListModelsResponse {
			models: vec![terminal_card(), login_required_card()],
			cursor: Some(pb::Cursor { epoch: Bytes::from_static(b"terminal"), generation: 1 }),
			roles:  BTreeMap::new(),
		}))
	}

	async fn watch_models(
		&self,
		_request: Request<pb::WatchModelsRequest>,
	) -> Result<Response<Self::WatchModelsStream>, Status> {
		Err(Status::unimplemented("test gateway uses polling"))
	}

	async fn fork(
		&self,
		_request: Request<pb::ForkRequest>,
	) -> Result<Response<pb::ForkResponse>, Status> {
		unimplemented_rpc()
	}

	async fn drop(
		&self,
		_request: Request<pb::DropRequest>,
	) -> Result<Response<pb::DropResponse>, Status> {
		unimplemented_rpc()
	}

	async fn count_tokens(
		&self,
		_request: Request<pb::CountTokensRequest>,
	) -> Result<Response<pb::CountTokensResponse>, Status> {
		unimplemented_rpc()
	}

	async fn embed(
		&self,
		_request: Request<pb::EmbedRequest>,
	) -> Result<Response<pb::EmbedResponse>, Status> {
		unimplemented_rpc()
	}

	async fn generate_image(
		&self,
		_request: Request<pb::GenerateImageRequest>,
	) -> Result<Response<Self::GenerateImageStream>, Status> {
		unimplemented_rpc()
	}

	async fn speak(
		&self,
		_request: Request<pb::SpeakRequest>,
	) -> Result<Response<Self::SpeakStream>, Status> {
		unimplemented_rpc()
	}

	async fn transcribe(
		&self,
		_request: Request<pb::TranscribeRequest>,
	) -> Result<Response<pb::TranscribeResponse>, Status> {
		unimplemented_rpc()
	}

	async fn generate_video(
		&self,
		_request: Request<pb::GenerateVideoRequest>,
	) -> Result<Response<pb::GenerationStatus>, Status> {
		unimplemented_rpc()
	}

	async fn get_generation(
		&self,
		_request: Request<pb::GetGenerationRequest>,
	) -> Result<Response<pb::GenerationStatus>, Status> {
		unimplemented_rpc()
	}

	async fn attach_generation(
		&self,
		_request: Request<pb::AttachGenerationRequest>,
	) -> Result<Response<Self::AttachGenerationStream>, Status> {
		unimplemented_rpc()
	}

	async fn cancel_generation(
		&self,
		_request: Request<pb::CancelGenerationRequest>,
	) -> Result<Response<pb::GenerationStatus>, Status> {
		unimplemented_rpc()
	}

	async fn search(
		&self,
		_request: Request<pb::SearchRequest>,
	) -> Result<Response<pb::SearchResponse>, Status> {
		unimplemented_rpc()
	}

	async fn list_providers(
		&self,
		_request: Request<pb::ListProvidersRequest>,
	) -> Result<Response<pb::ListProvidersResponse>, Status> {
		unimplemented_rpc()
	}

	async fn refresh_models(
		&self,
		_request: Request<pb::RefreshModelsRequest>,
	) -> Result<Response<pb::ListModelsResponse>, Status> {
		self
			.list_models(Request::new(pb::ListModelsRequest::default()))
			.await
	}
}

#[derive(Clone)]
struct RelayGateway {
	upstream: FederatedProvider,
	registry: Arc<Mutex<Registry>>,
}

#[tonic::async_trait]
impl Inference for RelayGateway {
	type AttachGenerationStream = RpcStream<pb::GenerationStatus>;
	type GenerateImageStream = RpcStream<pb::ImageEvent>;
	type SpeakStream = RpcStream<pb::SpeakEvent>;
	type TurnStream = RpcStream<pb::TurnEvent>;
	type WatchModelsStream = RpcStream<pb::ModelEvent>;

	async fn turn(
		&self,
		request: Request<tonic::Streaming<pb::TurnFrame>>,
	) -> Result<Response<Self::TurnStream>, Status> {
		let mut request = request.into_inner();
		let Some(pb::turn_frame::Frame::Open(mut open)) =
			request.message().await?.and_then(|frame| frame.frame)
		else {
			return Err(Status::invalid_argument("first frame is not open"));
		};
		// The relay has admitted the transport-level idempotency key. The
		// canonical conversion deliberately accepts only the stateless payload.
		open.turn_id.clear();
		let request = ChatRequest::try_from(open)
			.map_err(|error| Status::invalid_argument(error.to_string()))?;
		let stream = self
			.upstream
			.turn(request, None)
			.await
			.map_err(|error| Status::unavailable(error.to_string()))?;
		Ok(Response::new(Box::pin(stream.map(|event| Ok(event.into())))))
	}

	async fn list_models(
		&self,
		_request: Request<pb::ListModelsRequest>,
	) -> Result<Response<pb::ListModelsResponse>, Status> {
		let (models, cursor) = self.registry.lock().list(&ListFilter::default());
		Ok(Response::new(pb::ListModelsResponse {
			models: models.into_iter().map(card_to_wire).collect(),
			cursor: Some(pb::Cursor { epoch: cursor.epoch, generation: cursor.generation }),
			roles:  BTreeMap::new(),
		}))
	}

	async fn watch_models(
		&self,
		_request: Request<pb::WatchModelsRequest>,
	) -> Result<Response<Self::WatchModelsStream>, Status> {
		Err(Status::unimplemented("test gateway uses polling"))
	}

	async fn fork(
		&self,
		_request: Request<pb::ForkRequest>,
	) -> Result<Response<pb::ForkResponse>, Status> {
		unimplemented_rpc()
	}

	async fn drop(
		&self,
		_request: Request<pb::DropRequest>,
	) -> Result<Response<pb::DropResponse>, Status> {
		unimplemented_rpc()
	}

	async fn count_tokens(
		&self,
		_request: Request<pb::CountTokensRequest>,
	) -> Result<Response<pb::CountTokensResponse>, Status> {
		unimplemented_rpc()
	}

	async fn embed(
		&self,
		_request: Request<pb::EmbedRequest>,
	) -> Result<Response<pb::EmbedResponse>, Status> {
		unimplemented_rpc()
	}

	async fn generate_image(
		&self,
		_request: Request<pb::GenerateImageRequest>,
	) -> Result<Response<Self::GenerateImageStream>, Status> {
		unimplemented_rpc()
	}

	async fn speak(
		&self,
		_request: Request<pb::SpeakRequest>,
	) -> Result<Response<Self::SpeakStream>, Status> {
		unimplemented_rpc()
	}

	async fn transcribe(
		&self,
		_request: Request<pb::TranscribeRequest>,
	) -> Result<Response<pb::TranscribeResponse>, Status> {
		unimplemented_rpc()
	}

	async fn generate_video(
		&self,
		_request: Request<pb::GenerateVideoRequest>,
	) -> Result<Response<pb::GenerationStatus>, Status> {
		unimplemented_rpc()
	}

	async fn get_generation(
		&self,
		_request: Request<pb::GetGenerationRequest>,
	) -> Result<Response<pb::GenerationStatus>, Status> {
		unimplemented_rpc()
	}

	async fn attach_generation(
		&self,
		_request: Request<pb::AttachGenerationRequest>,
	) -> Result<Response<Self::AttachGenerationStream>, Status> {
		unimplemented_rpc()
	}

	async fn cancel_generation(
		&self,
		_request: Request<pb::CancelGenerationRequest>,
	) -> Result<Response<pb::GenerationStatus>, Status> {
		unimplemented_rpc()
	}

	async fn search(
		&self,
		_request: Request<pb::SearchRequest>,
	) -> Result<Response<pb::SearchResponse>, Status> {
		unimplemented_rpc()
	}

	async fn list_providers(
		&self,
		_request: Request<pb::ListProvidersRequest>,
	) -> Result<Response<pb::ListProvidersResponse>, Status> {
		unimplemented_rpc()
	}

	async fn refresh_models(
		&self,
		_request: Request<pb::RefreshModelsRequest>,
	) -> Result<Response<pb::ListModelsResponse>, Status> {
		self
			.list_models(Request::new(pb::ListModelsRequest::default()))
			.await
	}
}

struct Harness {
	direct:             Channel,
	relay:              Channel,
	capture:            Arc<Mutex<Vec<u8>>>,
	turns:              Arc<AtomicUsize>,
	credential_applied: Arc<AtomicUsize>,
	cancelled:          flume::Receiver<String>,
	_tasks:             Vec<JoinHandle<()>>,
	_temp:              tempfile::TempDir,
}

impl Harness {
	async fn start() -> Self {
		let temp = tempfile::tempdir().expect("temp credential store");
		let store = Arc::new(Store::open(temp.path().join("broker.sqlite")).expect("open store"));
		store
			.upsert_api_key("terminal-provider", "integration", SENTINEL, 1)
			.expect("plant sentinel credential");
		let turns = Arc::new(AtomicUsize::new(0));
		let need_full_turns = Arc::new(AtomicUsize::new(0));
		let credential_applied = Arc::new(AtomicUsize::new(0));
		let (cancelled_tx, cancelled) = flume::unbounded();
		let terminal_provider = terminal_provider();
		let providers: ProviderCatalog =
			std::iter::once((terminal_provider.id.clone(), terminal_provider)).collect();
		let terminal = TerminalGateway {
			credentials: BrokerCredentialSource::new(
				Arc::clone(&store),
				Arc::new(providers),
				Arc::new(NoRefresh),
			),
			turns: Arc::clone(&turns),
			need_full_turns,
			credential_applied: Arc::clone(&credential_applied),
			cancelled: cancelled_tx,
		};
		let (terminal_endpoint, terminal_task) = spawn_gateway(terminal).await;
		let terminal_addr = endpoint_addr(&terminal_endpoint);
		let (proxy_endpoint, capture, proxy_task) = spawn_capture_proxy(terminal_addr).await;
		let direct = connect(&proxy_endpoint).await;

		let provider = omp_provider(&proxy_endpoint);
		let federated =
			FederatedProvider::connect(&provider, direct.clone(), Duration::from_secs(3_600))
				.await
				.expect("connect federated provider");
		let mut registry = Registry::from_cards(&[], Arc::new(AlwaysAvailable));
		registry.apply_federated(federated.provider_id(), federated.cards());
		let (relay_endpoint, relay_task) = spawn_gateway(RelayGateway {
			upstream: federated,
			registry: Arc::new(Mutex::new(registry)),
		})
		.await;
		let relay = connect(&relay_endpoint).await;

		Self {
			direct,
			relay,
			capture,
			turns,
			credential_applied,
			cancelled,
			_tasks: vec![terminal_task, proxy_task, relay_task],
			_temp: temp,
		}
	}
}

#[tokio::test]
async fn two_gateway_chain_preserves_locality_availability_and_need_full() {
	let harness = Harness::start().await;
	let client = OmpFederation::new(InferenceClient::new(harness.relay.clone()));

	let mut normal = client
		.turn(chat("normal"), None)
		.await
		.expect("open federated turn");
	assert_eq!(
		normal.next().await,
		Some(TurnEvent::PartDelta { index: 0, chunk: Bytes::from_static(b"terminal-delta") })
	);
	assert!(matches!(normal.next().await, Some(TurnEvent::Outcome(_))));
	assert!(normal.next().await.is_none(), "events followed the terminal outcome");

	let mut discovery = InferenceClient::new(harness.relay.clone());
	let models = discovery
		.list_models(pb::ListModelsRequest::default())
		.await
		.expect("federated list models")
		.into_inner();
	assert_eq!(models.models.len(), 2);
	let login_required = models
		.models
		.iter()
		.find(|card| card.model == "login-required")
		.expect("federated login-required model");
	assert_eq!(login_required.availability, pb::Availability::LoginRequired as i32);
	let available = models
		.models
		.iter()
		.find(|card| card.model == "model")
		.expect("federated available model");
	assert_eq!(available.availability, pb::Availability::Available as i32);

	let mut compact = client
		.turn(chat("need-full"), None)
		.await
		.expect("open compact turn");
	assert!(matches!(
		compact.next().await,
		Some(TurnEvent::Error(error)) if error.kind == TurnErrorKind::NeedFull
	));
	assert!(compact.next().await.is_none(), "events followed terminal NEED_FULL");

	let mut full = client
		.turn(chat("need-full"), None)
		.await
		.expect("open full replay");
	assert!(matches!(full.next().await, Some(TurnEvent::PartDelta { .. })));
	assert!(matches!(full.next().await, Some(TurnEvent::Outcome(_))));
	assert!(full.next().await.is_none(), "events followed replay outcome");

	assert_eq!(harness.turns.load(Ordering::SeqCst), 3, "terminal provider saw each turn once");
	assert_eq!(
		harness.credential_applied.load(Ordering::SeqCst),
		3,
		"terminal provider received the sentinel credential"
	);
	let captured = harness.capture.lock();
	assert!(
		!captured
			.windows(SENTINEL.len())
			.any(|window| window == SENTINEL),
		"sentinel credential leaked into downstream-to-upstream gateway traffic"
	);
}

#[tokio::test]
async fn dropping_direct_and_federated_turns_aborts_terminal_streams() {
	let harness = Harness::start().await;
	let direct = OmpFederation::new(InferenceClient::new(harness.direct.clone()));
	let relay = OmpFederation::new(InferenceClient::new(harness.relay.clone()));

	let mut stream = direct
		.turn(chat("cancel-direct"), None)
		.await
		.expect("open direct turn");
	assert!(stream.next().await.is_some());
	drop(stream);
	assert_cancelled(&harness.cancelled, "cancel-direct").await;

	let mut stream = relay
		.turn(chat("cancel-federated"), None)
		.await
		.expect("open federated turn");
	assert!(stream.next().await.is_some());
	drop(stream);
	assert_cancelled(&harness.cancelled, "cancel-federated").await;
}

#[test]
fn delta_events_share_fixed_size_chunks_instead_of_accumulated_snapshots() {
	// A process-global counting allocator would make this parallel test suite
	// nondeterministic. Audit the stronger structural invariant instead: every
	// event owns only one fixed-size Bytes view, and cloning an event retains the
	// same backing allocation regardless of how many prior deltas exist.
	let backing = Bytes::from(vec![b'x'; 16 * 1024]);
	let mut events = Vec::with_capacity(4_096);
	for index in 0..4_096_u32 {
		let chunk = backing.slice((index as usize % 1_024)..(index as usize % 1_024 + 8));
		events.push(TurnEvent::PartDelta { index, chunk });
	}
	for event in &events {
		let cloned = event.clone();
		match (event, cloned) {
			(TurnEvent::PartDelta { chunk: original, .. }, TurnEvent::PartDelta { chunk, .. }) => {
				assert_eq!(chunk.len(), 8);
				assert_eq!(chunk.as_ptr(), original.as_ptr(), "Bytes clone copied delta payload");
			},
			_ => unreachable!("synthetic stream contains only deltas"),
		}
	}
	assert_eq!(
		events
			.iter()
			.map(|event| match event {
				TurnEvent::PartDelta { chunk, .. } => chunk.len(),
				_ => 0,
			})
			.sum::<usize>(),
		4_096 * 8
	);
}

const fn delta(chunk: &'static [u8]) -> pb::TurnEvent {
	pb::TurnEvent {
		event: Some(pb::turn_event::Event::PartDelta(pb::PartDelta {
			index: 0,
			chunk: Bytes::from_static(chunk),
		})),
	}
}

fn outcome() -> pb::TurnEvent {
	pb::TurnEvent {
		event: Some(pb::turn_event::Event::Outcome(pb::Outcome {
			stop: pb::StopReason::StopEndTurn as i32,
			..Default::default()
		})),
	}
}

fn terminal_card() -> pb::ModelCard {
	pb::ModelCard {
		id: "terminal-provider/model".into(),
		provider: "terminal-provider".into(),
		model: "model".into(),
		name: "Terminal model".into(),
		family: "test".into(),
		facets: vec![pb::Facet::Chat as i32],
		availability: pb::Availability::Available as i32,
		source: pb::model_card::Source::Configured as i32,
		..Default::default()
	}
}

fn login_required_card() -> pb::ModelCard {
	pb::ModelCard {
		id: "terminal-provider/login-required".into(),
		provider: "terminal-provider".into(),
		model: "login-required".into(),
		name: "Login-required model".into(),
		family: "test".into(),
		facets: vec![pb::Facet::Chat as i32],
		availability: pb::Availability::LoginRequired as i32,
		source: pb::model_card::Source::Configured as i32,
		..Default::default()
	}
}

fn card_to_wire(card: ModelCard) -> pb::ModelCard {
	pb::ModelCard {
		id: card.id.into(),
		provider: card.provider.into(),
		model: card.model.into(),
		name: card.name.into(),
		family: card.family.into(),
		facets: card
			.facets
			.into_iter()
			.map(|facet| match facet {
				Facet::Chat => pb::Facet::Chat as i32,
				_ => pb::Facet::Unspecified as i32,
			})
			.collect(),
		availability: match card.availability {
			Availability::Available => pb::Availability::Available as i32,
			Availability::LoginRequired => pb::Availability::LoginRequired as i32,
			Availability::Blocked => pb::Availability::Blocked as i32,
			Availability::Disabled => pb::Availability::Disabled as i32,
			_ => pb::Availability::Unspecified as i32,
		},
		source: match card.source {
			Source::Bundled => pb::model_card::Source::Bundled as i32,
			Source::Discovered => pb::model_card::Source::Discovered as i32,
			Source::Configured => pb::model_card::Source::Configured as i32,
			_ => pb::model_card::Source::Unspecified as i32,
		},
		..Default::default()
	}
}

fn chat(model: &str) -> ChatRequest {
	ChatRequest::builder()
		.model(Str::from(model))
		.thread(Thread::default())
		.tools(Vec::new())
		.provider_options(Props::default())
		.build()
}

fn terminal_provider() -> ProviderEntry {
	ProviderEntry::builder()
		.id(Str::new("terminal-provider"))
		.transport(TransportId::Omp)
		.base_url(Str::new("https://terminal.invalid"))
		.auth(AuthSpec::Bearer { env: smallvec::smallvec![] })
		.facets(smallvec::smallvec![Facet::Chat])
		.headers(BTreeMap::new())
		.compat(Compat::default())
		.build()
}

fn omp_provider(endpoint: &str) -> ProviderEntry {
	ProviderEntry::builder()
		.id(Str::new("relay-upstream"))
		.transport(TransportId::Omp)
		.base_url(Str::from(endpoint))
		.auth(AuthSpec::None)
		.facets(smallvec::smallvec![Facet::Chat])
		.headers(BTreeMap::new())
		.compat(Compat::default())
		.build()
}

async fn spawn_gateway<I>(service: I) -> (String, JoinHandle<()>)
where
	I: Inference,
{
	let listener = TcpListener::bind("127.0.0.1:0")
		.await
		.expect("bind gateway");
	let address = listener.local_addr().expect("gateway address");
	let task = tokio::spawn(async move {
		Server::builder()
			.add_service(InferenceServer::new(service))
			.serve_with_incoming(TcpListenerStream::new(listener))
			.await
			.expect("serve gateway");
	});
	(format!("http://{address}"), task)
}

async fn spawn_capture_proxy(
	target: std::net::SocketAddr,
) -> (String, Arc<Mutex<Vec<u8>>>, JoinHandle<()>) {
	let listener = TcpListener::bind("127.0.0.1:0")
		.await
		.expect("bind capture proxy");
	let address = listener.local_addr().expect("proxy address");
	let capture = Arc::new(Mutex::new(Vec::new()));
	let captured = Arc::clone(&capture);
	let task = tokio::spawn(async move {
		loop {
			let (client, _) = listener.accept().await.expect("accept gateway connection");
			let upstream = TcpStream::connect(target)
				.await
				.expect("connect terminal gateway");
			let bytes = Arc::clone(&captured);
			tokio::spawn(async move {
				let (mut client_read, mut client_write) = client.into_split();
				let (mut upstream_read, mut upstream_write) = upstream.into_split();
				let to_upstream = async {
					let mut buffer = [0_u8; 8 * 1024];
					loop {
						let count = client_read.read(&mut buffer).await?;
						if count == 0 {
							break;
						}
						bytes.lock().extend_from_slice(&buffer[..count]);
						upstream_write.write_all(&buffer[..count]).await?;
					}
					upstream_write.shutdown().await
				};
				let to_client = async {
					tokio::io::copy(&mut upstream_read, &mut client_write).await?;
					client_write.shutdown().await
				};
				let _: (std::io::Result<()>, std::io::Result<()>) =
					tokio::join!(to_upstream, to_client);
			});
		}
	});
	(format!("http://{address}"), capture, task)
}

async fn connect(endpoint: &str) -> Channel {
	Endpoint::from_shared(endpoint.to_owned())
		.expect("valid endpoint")
		.connect()
		.await
		.expect("connect gateway")
}

fn endpoint_addr(endpoint: &str) -> std::net::SocketAddr {
	endpoint
		.strip_prefix("http://")
		.expect("http endpoint")
		.parse()
		.expect("socket endpoint")
}

async fn assert_cancelled(cancelled: &flume::Receiver<String>, expected: &str) {
	let observed = timeout(Duration::from_secs(2), cancelled.recv_async())
		.await
		.expect("terminal server did not observe stream abort")
		.expect("cancellation observer closed");
	assert_eq!(observed, expected);
}

fn unimplemented_rpc<T>() -> Result<Response<T>, Status> {
	Err(Status::unimplemented("not used by federation integration test"))
}
