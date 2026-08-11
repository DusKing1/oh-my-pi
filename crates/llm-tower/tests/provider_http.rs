//! HTTP provider attempt behavior and wire-boundary fixtures.

use std::{
	collections::{BTreeMap, VecDeque},
	future::{Ready, ready},
	pin::Pin,
	sync::{
		Arc,
		atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
	},
	task::{Context, Poll},
	time::Duration,
};

use bytes::Bytes;
use futures::{StreamExt, task::AtomicWaker};
use http::{Request, Response, header};
use http_body_util::BodyExt;
use hyper::body::{Body as HttpBody, Frame};
use omp_llm_catalog::{
	compat::StreamProtocol,
	provider::{CodexTransportPreference, ProviderEntry, TransportId, load_builtin},
};
use omp_llm_egress::{
	auth_inject::{
		AuthContext, AwsSigV4Context, CredentialAuthKind, CredentialLease, CredentialMetadata,
	},
	client::Body,
};
use omp_llm_tower::{
	codex_websocket::CodexWebSocketRequest,
	provider::{ProviderAttempt, ProviderAttemptError, ProviderBuildError, ProviderRoute},
	select::Routed,
};
use omp_llm_types::{
	BlobPart, CacheHint, CacheRetention, CallId, ChatRequest, Item, ItemKind, Message, Part, Props,
	RequestMeta, ResolvedModelCapabilities, ResolvedModelHeaders, ResolvedModelPolicy, Role, Thread,
	ToolDef, ToolResult, TurnEvent as NativeTurnEvent,
};
use omp_proto::inference::v1::{TurnRequest, turn_event};
use parking_lot::Mutex;
use serde_json::Value;
use tower::{Service, ServiceExt};

#[derive(Debug, thiserror::Error)]
#[error("controlled body failure")]
struct BodyFailure;

#[derive(Debug, thiserror::Error)]
#[error("egress failure")]
struct EgressFailure;

enum BodyStep {
	Data(Bytes),
	Fail,
	End,
}

struct BodyState {
	steps:   Mutex<VecDeque<BodyStep>>,
	waker:   AtomicWaker,
	dropped: AtomicBool,
}

#[derive(Clone)]
struct BodyControl {
	state: Arc<BodyState>,
}

impl BodyControl {
	fn push(&self, data: impl Into<Bytes>) {
		self
			.state
			.steps
			.lock()
			.push_back(BodyStep::Data(data.into()));
		self.state.waker.wake();
	}

	fn fail(&self) {
		self.state.steps.lock().push_back(BodyStep::Fail);
		self.state.waker.wake();
	}

	fn finish(&self) {
		self.state.steps.lock().push_back(BodyStep::End);
		self.state.waker.wake();
	}

	fn dropped(&self) -> bool {
		self.state.dropped.load(Ordering::SeqCst)
	}
}

struct ControlledBody {
	state: Arc<BodyState>,
}

impl ControlledBody {
	fn new() -> (Self, BodyControl) {
		let state = Arc::new(BodyState {
			steps:   Mutex::new(VecDeque::new()),
			waker:   AtomicWaker::new(),
			dropped: AtomicBool::new(false),
		});
		(Self { state: Arc::clone(&state) }, BodyControl { state })
	}
}

impl Drop for ControlledBody {
	fn drop(&mut self) {
		self.state.dropped.store(true, Ordering::SeqCst);
		self.state.waker.wake();
	}
}

impl HttpBody for ControlledBody {
	type Data = Bytes;
	type Error = BodyFailure;

	fn poll_frame(
		self: Pin<&mut Self>,
		cx: &mut Context<'_>,
	) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
		if let Some(step) = self.state.steps.lock().pop_front() {
			return Poll::Ready(match step {
				BodyStep::Data(data) => Some(Ok(Frame::data(data))),
				BodyStep::Fail => Some(Err(BodyFailure)),
				BodyStep::End => None,
			});
		}
		self.state.waker.register(cx.waker());
		if let Some(step) = self.state.steps.lock().pop_front() {
			cx.waker().wake_by_ref();
			return Poll::Ready(match step {
				BodyStep::Data(data) => Some(Ok(Frame::data(data))),
				BodyStep::Fail => Some(Err(BodyFailure)),
				BodyStep::End => None,
			});
		}
		Poll::Pending
	}
}

struct CaptureShared {
	body:         Mutex<Option<ControlledBody>>,
	request:      Mutex<Option<Request<Body>>>,
	content_type: &'static str,
	ready_id:     AtomicU64,
	next_id:      AtomicU64,
	calls:        AtomicUsize,
}

struct CaptureService {
	shared: Arc<CaptureShared>,
	id:     u64,
}

impl CaptureService {
	fn new(body: ControlledBody, content_type: &'static str) -> Self {
		Self {
			shared: Arc::new(CaptureShared {
				body: Mutex::new(Some(body)),
				request: Mutex::new(None),
				content_type,
				ready_id: AtomicU64::new(0),
				next_id: AtomicU64::new(2),
				calls: AtomicUsize::new(0),
			}),
			id:     1,
		}
	}

	fn take_request(&self) -> Request<Body> {
		self.shared.request.lock().take().expect("request captured")
	}
}

impl Clone for CaptureService {
	fn clone(&self) -> Self {
		Self {
			shared: Arc::clone(&self.shared),
			id:     self.shared.next_id.fetch_add(1, Ordering::SeqCst),
		}
	}
}

impl Service<Request<Body>> for CaptureService {
	type Error = EgressFailure;
	type Future = Ready<Result<Self::Response, Self::Error>>;
	type Response = Response<ControlledBody>;

	fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		self.shared.ready_id.store(self.id, Ordering::SeqCst);
		Poll::Ready(Ok(()))
	}

	fn call(&mut self, request: Request<Body>) -> Self::Future {
		assert_eq!(
			self.shared.ready_id.load(Ordering::SeqCst),
			self.id,
			"poll_ready and call used different egress clones"
		);
		self.shared.calls.fetch_add(1, Ordering::SeqCst);
		*self.shared.request.lock() = Some(request);
		let body = self.shared.body.lock().take().expect("one response body");
		ready(Ok(Response::builder()
			.header(header::CONTENT_TYPE, self.shared.content_type)
			.body(body)
			.unwrap()))
	}
}

fn request() -> TurnRequest {
	let message = Message::builder()
		.role(Role::User)
		.parts(vec![Part::Text("hello".into())])
		.build();
	let item = Item::builder()
		.seq(0)
		.kind(ItemKind::Message(message))
		.props(Props::default())
		.build();
	ChatRequest::builder()
		.model("test-model".into())
		.thread(Thread::builder().items(vec![item]).build())
		.tools(Vec::new())
		.build()
		.into()
}

fn routed() -> Routed {
	Routed::new(request(), None, None)
}

fn routed_with_policy(policy: ResolvedModelPolicy) -> Routed {
	Routed::new(request(), None, None).with_model_policy(Some(Arc::new(policy)))
}

fn openai_service(
	body: ControlledBody,
	content_type: &'static str,
) -> (ProviderAttempt<CaptureService>, CaptureService) {
	let mut provider = load_builtin().unwrap().remove("openrouter").unwrap();
	provider.base_url = "https://example.test/v1".into();
	provider.headers.insert("x-static".into(), "catalog".into());
	let egress = CaptureService::new(body, content_type);
	let inspect = CaptureService { shared: Arc::clone(&egress.shared), id: egress.id };
	(ProviderAttempt::new(provider, ProviderRoute::default(), egress).unwrap(), inspect)
}

fn google_cca_service(body: ControlledBody) -> (ProviderAttempt<CaptureService>, CaptureService) {
	let mut provider = load_builtin().unwrap().remove("google-gemini-cli").unwrap();
	provider.base_url = "https://example.test".into();
	let egress = CaptureService::new(body, "text/event-stream");
	let inspect = CaptureService { shared: Arc::clone(&egress.shared), id: egress.id };
	let route = ProviderRoute { project: "catalog-placeholder".into(), ..ProviderRoute::default() };
	(ProviderAttempt::new(provider, route, egress).unwrap(), inspect)
}

async fn captured_provider_request(
	provider: ProviderEntry,
	route: ProviderRoute,
	request: TurnRequest,
) -> Request<Body> {
	let (body, control) = ControlledBody::new();
	control.fail();
	let egress = CaptureService::new(body, "text/event-stream");
	let capture = CaptureService { shared: Arc::clone(&egress.shared), id: egress.id };
	let mut adapter = ProviderAttempt::new(provider, route, egress).unwrap();
	let result = adapter
		.ready()
		.await
		.unwrap()
		.call(Routed::new(request, None, None))
		.await;
	assert!(matches!(result, Err(ProviderAttemptError::Body(BodyFailure))));
	capture.take_request()
}

fn content_chunk(text: &str) -> Bytes {
	Bytes::from(format!(
		"data: {{\"id\":\"chatcmpl_test\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":{text:?\
		 }}},\"finish_reason\":null}}]}}\n\n"
	))
}

fn terminal_chunk() -> Bytes {
	Bytes::from_static(
		b"data: {\"id\":\"chatcmpl_test\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
	)
}

#[test]
fn known_http_transports_construct_and_specialized_transports_are_rejected() {
	let catalog = load_builtin().unwrap();
	let template = catalog.get("openrouter").unwrap();
	let (body, _control) = ControlledBody::new();
	let egress = CaptureService::new(body, "text/event-stream");
	for transport in [
		TransportId::OpenAiChat,
		TransportId::OpenAiResponses,
		TransportId::OpenAiCodex,
		TransportId::AnthropicMessages,
		TransportId::AnthropicBedrock,
		TransportId::BedrockConverse,
		TransportId::AnthropicVertex,
		TransportId::GoogleGenAi,
		TransportId::GoogleVertex,
		TransportId::GoogleCca,
	] {
		let mut provider = template.clone();
		provider.transport = transport;
		let adapter = ProviderAttempt::new(
			provider,
			ProviderRoute {
				project:    "project".into(),
				region:     "us-central1".into(),
				deployment: "deployment".into(),
				account:    "account".into(),
				gateway:    "gateway".into(),
			},
			egress.clone(),
		)
		.unwrap();
		assert_eq!(adapter.codec().id(), transport);
	}
	for transport in [
		TransportId::Cursor,
		TransportId::Devin,
		TransportId::GitLabDuoWorkflow,
		TransportId::Embedded,
		TransportId::Omp,
	] {
		let mut provider = template.clone();
		provider.transport = transport;
		let result = ProviderAttempt::new(provider, ProviderRoute::default(), egress.clone());
		assert!(
			matches!(result, Err(ProviderBuildError::SpecializedTransport(id)) if id == transport)
		);
	}
}

#[test]
fn anthropic_vertex_rejects_beta_as_a_static_header() {
	let catalog = load_builtin().unwrap();
	let mut provider = catalog["openrouter"].clone();
	provider.transport = TransportId::AnthropicVertex;
	provider
		.headers
		.insert("AnThRoPiC-BeTa".into(), "prompt-caching".into());
	let (body, _control) = ControlledBody::new();
	let error = match ProviderAttempt::new(
		provider,
		ProviderRoute::default(),
		CaptureService::new(body, "text/event-stream"),
	) {
		Ok(_) => panic!("Anthropic Vertex accepted a forbidden beta header"),
		Err(error) => error,
	};
	assert!(matches!(
		error,
		ProviderBuildError::ForbiddenStaticHeader {
			transport: TransportId::AnthropicVertex,
			ref name,
		} if name == "AnThRoPiC-BeTa"
	));
}
#[tokio::test]
async fn anthropic_request_headers_override_statics_and_select_cache_betas() {
	let mut provider = load_builtin().unwrap().remove("anthropic").unwrap();
	provider
		.headers
		.insert("anthropic-version".into(), "static-version".into());
	provider
		.headers
		.insert("anthropic-beta".into(), "static-beta".into());
	let (body, control) = ControlledBody::new();
	control.fail();
	let egress = CaptureService::new(body, "text/event-stream");
	let capture = CaptureService { shared: Arc::clone(&egress.shared), id: egress.id };
	let mut adapter = ProviderAttempt::new(provider, ProviderRoute::default(), egress).unwrap();

	let mut native = ChatRequest::try_from(request()).unwrap();
	native.cache = Some(
		CacheHint::builder()
			.session_key("cache-session".into())
			.retention(CacheRetention::Long)
			.build(),
	);
	let mut options = Props::default();
	options.insert_ns("anthropic", "betas", Value::String("request-beta".to_owned()));
	options.insert_ns("anthropic", "version", Value::String("2025-01-01".to_owned()));
	native.provider_options = Some(options);
	let result = adapter
		.ready()
		.await
		.unwrap()
		.call(Routed::new(native.into(), None, None))
		.await;
	assert!(matches!(result, Err(ProviderAttemptError::Body(BodyFailure))));

	let request = capture.take_request();
	assert_eq!(request.headers()["anthropic-version"], "2025-01-01");
	assert_eq!(request.headers()["anthropic-beta"], "extended-cache-ttl-2025-04-11,request-beta");
	assert!(!request.headers().contains_key(header::AUTHORIZATION));
	assert!(!request.headers().contains_key("x-api-key"));
}

#[tokio::test]
async fn anthropic_selected_auth_kind_controls_tool_name_wire_policy() {
	for (auth_kind, expected) in [
		(CredentialAuthKind::ApiKey, "custom_tool"),
		(CredentialAuthKind::OAuth, "_custom_tool"),
		(CredentialAuthKind::Aws, "custom_tool"),
		(CredentialAuthKind::GoogleAdc, "custom_tool"),
	] {
		let provider = load_builtin().unwrap().remove("anthropic").unwrap();
		let (body, control) = ControlledBody::new();
		control.fail();
		let egress = CaptureService::new(body, "text/event-stream");
		let capture = CaptureService { shared: Arc::clone(&egress.shared), id: egress.id };
		let mut adapter = ProviderAttempt::new(provider, ProviderRoute::default(), egress).unwrap();
		let mut native = ChatRequest::try_from(request()).unwrap();
		native.tools.push(
			ToolDef::builder()
				.name("custom_tool".into())
				.description("test".into())
				.schema_json(Bytes::from_static(br#"{"type":"object"}"#))
				.build(),
		);
		let metadata = CredentialMetadata {
			identity: "routing-only".into(),
			auth_kind,
			account_id: None,
			project_id: None,
			organization_id: None,
		};
		let result = adapter
			.ready()
			.await
			.unwrap()
			.call(Routed::new(native.into(), None, Some(metadata)))
			.await;
		assert!(matches!(result, Err(ProviderAttemptError::Body(BodyFailure))));
		let request = capture.take_request();
		let encoded = request.into_body().collect().await.unwrap().to_bytes();
		let encoded: Value = serde_json::from_slice(&encoded).unwrap();
		assert_eq!(encoded["tools"][0]["name"], expected);
	}
}

#[tokio::test]
async fn safe_model_headers_override_statics_before_request_dynamic_headers() {
	let mut provider = load_builtin().unwrap().remove("github-copilot").unwrap();
	provider
		.headers
		.insert("x-precedence".into(), "provider".into());
	provider
		.headers
		.insert("x-initiator".into(), "agent".into());
	let (body, control) = ControlledBody::new();
	control.fail();
	let egress = CaptureService::new(body, "text/event-stream");
	let capture = CaptureService { shared: Arc::clone(&egress.shared), id: egress.id };
	let mut adapter = ProviderAttempt::new(provider, ProviderRoute::default(), egress).unwrap();
	let mut native = ChatRequest::try_from(request()).unwrap();
	native.meta = Some(
		RequestMeta::builder()
			.initiator("user".into())
			.session_id(omp_core::Str::new_static(""))
			.telemetry(BTreeMap::new())
			.build(),
	);
	let policy = Arc::new(ResolvedModelPolicy {
		headers: ResolvedModelHeaders(BTreeMap::from([
			("x-precedence".into(), "model".into()),
			("x-initiator".into(), "model".into()),
		])),
		..ResolvedModelPolicy::default()
	});
	let result = adapter
		.ready()
		.await
		.unwrap()
		.call(Routed::new(native.into(), None, None).with_model_policy(Some(policy)))
		.await;
	assert!(matches!(result, Err(ProviderAttemptError::Body(BodyFailure))));
	let request = capture.take_request();
	assert_eq!(request.headers()["x-precedence"], "model");
	assert_eq!(request.headers()["x-initiator"], "user");
}

#[tokio::test]
async fn unsafe_model_headers_are_rejected_before_http_egress() {
	for name in ["authorization", "connection", "content-length"] {
		let mut provider = load_builtin().unwrap().remove("openrouter").unwrap();
		provider.base_url = "https://proxy.example/v1".into();
		let (body, _control) = ControlledBody::new();
		let egress = CaptureService::new(body, "text/event-stream");
		let capture = CaptureService { shared: Arc::clone(&egress.shared), id: egress.id };
		let mut adapter = ProviderAttempt::new(provider, ProviderRoute::default(), egress).unwrap();
		let policy = ResolvedModelPolicy {
			headers: ResolvedModelHeaders(BTreeMap::from([(name.into(), "blocked".into())])),
			..ResolvedModelPolicy::default()
		};
		let result = adapter
			.ready()
			.await
			.unwrap()
			.call(routed_with_policy(policy))
			.await;
		assert!(matches!(result, Err(ProviderAttemptError::Encode(_))), "{name}");
		assert_eq!(capture.shared.calls.load(Ordering::SeqCst), 0, "{name}");
	}
}

#[tokio::test]
async fn inferred_computer_use_is_demoted_on_proxy_unless_explicitly_authored() {
	for (computer_use_config, expected_type) in [(None, "function"), (Some(true), "computer")] {
		let mut provider = load_builtin().unwrap().remove("openai").unwrap();
		provider.base_url = "https://unverified-proxy.example/v1".into();
		let (body, control) = ControlledBody::new();
		control.push(Bytes::from_static(
			b"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp\",\"status\":\"completed\",\"output\":[]}}\n\n",
		));
		let egress = CaptureService::new(body, "text/event-stream");
		let capture = CaptureService { shared: Arc::clone(&egress.shared), id: egress.id };
		let mut adapter = ProviderAttempt::new(provider, ProviderRoute::default(), egress).unwrap();
		let policy = Arc::new(ResolvedModelPolicy {
			capabilities: ResolvedModelCapabilities {
				computer_use: Some(true),
				computer_use_config,
				..ResolvedModelCapabilities::default()
			},
			..ResolvedModelPolicy::default()
		});
		let mut native = ChatRequest::try_from(request()).unwrap();
		let mut options = Props::default();
		options.insert_ns("openai", "hosted_tools", serde_json::json!([{ "type": "computer" }]));
		native.provider_options = Some(options);
		let stream = adapter
			.ready()
			.await
			.unwrap()
			.call(Routed::new(native.into(), None, None).with_model_policy(Some(policy)))
			.await
			.unwrap();
		let outbound = capture.take_request();
		let encoded = outbound.into_body().collect().await.unwrap().to_bytes();
		let encoded: Value = serde_json::from_slice(&encoded).unwrap();
		assert_eq!(encoded["tools"][0]["type"], expected_type);
		let events: Vec<_> = stream.collect().await;
		let outcome = events
			.into_iter()
			.find_map(|event| match event.event {
				Some(turn_event::Event::Outcome(outcome)) => Some(outcome),
				_ => None,
			})
			.expect("terminal outcome");
		assert_eq!(
			outcome
				.unsupported
				.iter()
				.any(|item| item.what == "computer_use"),
			computer_use_config.is_none()
		);
	}
}

#[tokio::test]
async fn request_encoding_uses_catalog_endpoint_headers_auth_and_routed_lease() {
	let (body, control) = ControlledBody::new();
	control.push(content_chunk("hello"));
	let (mut adapter, capture) = openai_service(body, "text/event-stream");
	let lease = CredentialLease::new("openai", 41, 7);
	let _stream = adapter
		.ready()
		.await
		.unwrap()
		.call(Routed::new(request(), Some(lease.clone()), None))
		.await
		.unwrap();
	let request = capture.take_request();
	assert_eq!(request.method(), http::Method::POST);
	assert_eq!(request.uri(), "https://example.test/v1/chat/completions");
	assert_eq!(request.headers()[header::CONTENT_TYPE], "application/json");
	assert_eq!(request.headers()[header::ACCEPT], "text/event-stream");
	assert_eq!(request.headers()["x-static"], "catalog");
	assert_eq!(
		request
			.extensions()
			.get::<AuthContext>()
			.unwrap()
			.provider(),
		"openrouter"
	);
	assert_eq!(request.extensions().get::<CredentialLease>(), Some(&lease));
	let encoded = request.into_body().collect().await.unwrap().to_bytes();
	let encoded: Value = serde_json::from_slice(&encoded).unwrap();
	assert_eq!(encoded["model"], "test-model");
	assert_eq!(encoded["stream"], true);
}

#[tokio::test]
async fn bedrock_converse_constructs_geo_endpoint_and_sealed_sigv4_metadata() {
	let provider = load_builtin()
		.unwrap()
		.remove("amazon-bedrock")
		.expect("Amazon Bedrock");
	assert_eq!(provider.transport, TransportId::BedrockConverse);
	let mut native = ChatRequest::try_from(request()).expect("canonical request");
	native.model = "eu.amazon.nova-pro-v1:0".into();
	let request = captured_provider_request(
		provider,
		ProviderRoute { region: "us-east-1".into(), ..ProviderRoute::default() },
		native.into(),
	)
	.await;
	assert_eq!(
		request.uri(),
		"https://bedrock-runtime.eu-west-1.amazonaws.com/model/eu.amazon.nova-pro-v1%3A0/converse-stream"
	);
	assert_eq!(request.headers()[header::ACCEPT], "application/vnd.amazon.eventstream");
	assert_eq!(request.headers()["x-amzn-bedrock-accept"], "application/json");
	assert!(!request.headers().contains_key(header::AUTHORIZATION));
	let context = request
		.extensions()
		.get::<AwsSigV4Context>()
		.expect("sealed SigV4 metadata");
	assert_eq!(context.service, "bedrock");
	assert_eq!(context.region, "eu-west-1");
	let body = request.into_body().collect().await.unwrap().to_bytes();
	let body: Value = serde_json::from_slice(&body).expect("Converse body");
	assert_eq!(body["messages"][0]["role"], "user");
	assert!(body.get("anthropic_version").is_none());
}

#[tokio::test]
async fn azure_chat_and_responses_append_one_catalog_selected_api_version() {
	let providers = load_builtin().unwrap();
	let route = ProviderRoute { region: "eastus".into(), ..ProviderRoute::default() };
	let chat = captured_provider_request(providers["azure"].clone(), route.clone(), request()).await;
	assert_eq!(
		chat.uri(),
		"https://eastus.openai.azure.com/openai/deployments/test-model/chat/completions?api-version=2024-10-21"
	);

	let mut responses_provider = providers["azure"].clone();
	responses_provider.transport = TransportId::OpenAiResponses;
	let mut responses_request = ChatRequest::try_from(request()).unwrap();
	let mut options = Props::default();
	options.insert_ns("azure", "api_version", Value::String("2025-01-01-preview".to_owned()));
	responses_request.provider_options = Some(options);
	let responses =
		captured_provider_request(responses_provider, route, responses_request.into()).await;
	assert_eq!(
		responses.uri(),
		"https://eastus.openai.azure.com/openai/deployments/test-model/responses?api-version=2025-01-01-preview"
	);

	let mut queried_provider = providers["azure"].clone();
	queried_provider.transport = TransportId::OpenAiResponses;
	queried_provider.base_url =
		"https://example.test/openai/deployments/test-model?api-version=existing&trace=1".into();
	let queried =
		captured_provider_request(queried_provider, ProviderRoute::default(), request()).await;
	assert_eq!(
		queried.uri(),
		"https://example.test/openai/deployments/test-model/responses?api-version=existing&trace=1"
	);
	assert_eq!(
		queried
			.uri()
			.query()
			.unwrap()
			.split('&')
			.filter(|field| field.starts_with("api-version="))
			.count(),
		1
	);
}

#[tokio::test]
async fn copilot_headers_follow_user_tool_result_vision_and_safe_overrides() {
	let provider = load_builtin().unwrap().remove("github-copilot").unwrap();
	let user =
		captured_provider_request(provider.clone(), ProviderRoute::default(), request()).await;
	assert_eq!(user.headers()["x-initiator"], "user");
	assert_eq!(user.headers()["openai-intent"], "conversation-edits");
	assert_eq!(user.headers()["user-agent"], "opencode/1.3.15");
	assert_eq!(user.headers()["x-github-api-version"], "2026-06-01");
	assert!(!user.headers().contains_key("copilot-vision-request"));
	assert!(!user.headers().contains_key("editor-version"));
	assert!(!user.headers().contains_key("copilot-integration-id"));

	let mut tool_result_request = ChatRequest::try_from(request()).unwrap();
	tool_result_request.thread.items.push(
		Item::builder()
			.seq(1)
			.kind(ItemKind::ToolResult(
				ToolResult::builder()
					.call_id(CallId::default())
					.name("read".into())
					.parts(vec![Part::Text("done".into())])
					.is_error(false)
					.build(),
			))
			.props(Props::default())
			.build(),
	);
	let tool_result = captured_provider_request(
		provider.clone(),
		ProviderRoute::default(),
		tool_result_request.into(),
	)
	.await;
	assert_eq!(tool_result.headers()["x-initiator"], "agent");

	let mut vision_request = ChatRequest::try_from(request()).unwrap();
	let ItemKind::Message(message) = &mut vision_request.thread.items[0].kind else {
		panic!("fixture starts with a user message");
	};
	message.parts.push(Part::Blob(
		BlobPart::builder()
			.hash([0; 32])
			.mime("image/png".into())
			.size(3)
			.inline(Bytes::from_static(b"png"))
			.build(),
	));
	let mut meta = RequestMeta::default();
	meta.initiator = "agent".into();
	vision_request.meta = Some(meta);
	let vision =
		captured_provider_request(provider.clone(), ProviderRoute::default(), vision_request.into())
			.await;
	assert_eq!(vision.headers()["x-initiator"], "agent");
	assert_eq!(vision.headers()["copilot-vision-request"], "true");

	let mut unsafe_provider = provider;
	unsafe_provider
		.headers
		.insert("X-Initiator".into(), "Bearer catalog-credential".into());
	let mut unsafe_request = ChatRequest::try_from(request()).unwrap();
	let mut meta = RequestMeta::default();
	meta.initiator = "runtime-credential".into();
	unsafe_request.meta = Some(meta);
	let safe =
		captured_provider_request(unsafe_provider, ProviderRoute::default(), unsafe_request.into())
			.await;
	assert_eq!(safe.headers()["x-initiator"], "user");
	assert!(
		safe
			.headers()
			.values()
			.all(|value| !value.to_str().unwrap().contains("credential"))
	);
}

#[tokio::test]
async fn cca_uses_selected_credential_metadata_for_envelope_headers_and_egress() {
	let (body, control) = ControlledBody::new();
	control.push(Bytes::from_static(
		b"data: {\"response\":{\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"hello\"}]}}]}}\n\n",
	));
	let (mut adapter, capture) = google_cca_service(body);
	let lease = CredentialLease::new("google-gemini-cli", 51, 4);
	let metadata = CredentialMetadata {
		auth_kind:       CredentialAuthKind::OAuth,
		identity:        "user@example.test".into(),
		account_id:      Some("account-51".into()),
		project_id:      Some("selected-project".into()),
		organization_id: Some("organization-51".into()),
	};
	let _stream = adapter
		.ready()
		.await
		.unwrap()
		.call(Routed::new(request(), Some(lease.clone()), Some(metadata.clone())))
		.await
		.unwrap();

	let request = capture.take_request();
	assert_eq!(request.headers()["x-goog-user-project"], "selected-project");
	assert_eq!(request.extensions().get::<CredentialLease>(), Some(&lease));
	assert_eq!(request.extensions().get::<CredentialMetadata>(), Some(&metadata));
	let encoded = request.into_body().collect().await.unwrap().to_bytes();
	let encoded: Value = serde_json::from_slice(&encoded).unwrap();
	assert_eq!(encoded["project"], "selected-project");
}
#[tokio::test]
async fn antigravity_envelope_and_flash_planning_filter_are_used_in_production_attempt() {
	let mut provider = load_builtin()
		.unwrap()
		.remove("google-antigravity")
		.unwrap();
	provider.base_url = "https://antigravity.example.test".into();
	let (body, control) = ControlledBody::new();
	control.push(Bytes::from_static(include_bytes!(
		"../../llm-google/tests/fixtures/google_cca/stream.antigravity_leak.sse"
	)));
	control.finish();
	let egress = CaptureService::new(body, "text/event-stream");
	let capture = CaptureService { shared: Arc::clone(&egress.shared), id: egress.id };
	let route = ProviderRoute { project: "catalog-placeholder".into(), ..ProviderRoute::default() };
	let mut adapter = ProviderAttempt::new(provider, route, egress).unwrap();

	let agent_id = "11111111-1111-1111-1111-111111111111";
	let trajectory_id = "22222222-2222-2222-2222-222222222222";
	let request_id = format!("agent/{agent_id}/1700000000000/{trajectory_id}/2");
	let mut native = ChatRequest::try_from(request()).unwrap();
	native.model = "gemini-3.5-flash-low".into();
	let mut meta = RequestMeta::default();
	meta.session_id = "-8392019482710394817".into();
	native.meta = Some(meta);
	let mut options = Props::default();
	for (name, value) in [
		("agent_id", Value::String(agent_id.to_owned())),
		("request_id", Value::String(request_id.clone())),
		("trajectory_id", Value::String(trajectory_id.to_owned())),
		("step_index", Value::from(2_u64)),
		("last_execution_id", Value::String("execution-before".to_owned())),
		("model_enum", Value::String("MODEL_PLACEHOLDER_M20".to_owned())),
	] {
		options.insert_ns("google-antigravity", name, value);
	}
	native.provider_options = Some(options);
	let lease = CredentialLease::new("google-antigravity", 71, 5);
	let credential = CredentialMetadata {
		auth_kind:       CredentialAuthKind::OAuth,
		identity:        "developer@example.test".into(),
		account_id:      Some("account-71".into()),
		project_id:      Some("selected-project".into()),
		organization_id: Some("organization-71".into()),
	};
	let mut stream = adapter
		.ready()
		.await
		.unwrap()
		.call(Routed::new(native.into(), Some(lease), Some(credential)))
		.await
		.unwrap();
	let mut visible = Vec::new();
	let mut outcome = None;
	while let Some(event) = stream.next().await {
		match NativeTurnEvent::try_from(event).unwrap() {
			NativeTurnEvent::PartDelta { chunk, .. } => visible.extend_from_slice(&chunk),
			NativeTurnEvent::Outcome(value) => outcome = Some(value),
			_ => {},
		}
	}
	let visible = String::from_utf8(visible).unwrap();
	assert!(visible.contains("正文 ✓"));
	assert!(!visible.contains("internal plan"));
	let outcome = outcome.expect("Antigravity stream has a terminal outcome");
	assert_eq!(
		outcome.props.get_ns("google", "response_id"),
		Some(&serde_json::json!("execution-after"))
	);
	assert_eq!(
		outcome.props.get_ns("google-cca", "served_endpoint"),
		Some(&serde_json::json!("https://antigravity.example.test"))
	);

	let request = capture.take_request();
	assert!(
		request.headers()["user-agent"]
			.to_str()
			.unwrap()
			.contains("antigravity")
	);
	assert_eq!(request.headers()["x-goog-user-project"], "selected-project");
	let encoded = request.into_body().collect().await.unwrap().to_bytes();
	let encoded: Value = serde_json::from_slice(&encoded).unwrap();
	assert_eq!(encoded["project"], "selected-project");
	assert_eq!(encoded["requestId"], request_id);
	assert_eq!(encoded["request"]["sessionId"], "-8392019482710394817");
	assert_eq!(encoded["request"]["labels"]["trajectory_id"], trajectory_id);
}

#[tokio::test]
async fn cca_selected_lease_without_metadata_is_rejected_before_egress() {
	let (body, _control) = ControlledBody::new();
	let (mut adapter, capture) = google_cca_service(body);
	let lease = CredentialLease::new("google-gemini-cli", 51, 4);
	let result = adapter
		.ready()
		.await
		.unwrap()
		.call(Routed::new(request(), Some(lease), None))
		.await;

	assert!(matches!(result, Err(ProviderAttemptError::CredentialMetadata("project_id"))));
	assert_eq!(capture.shared.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn codex_models_choose_sse_lite_and_websocket_full_independently() {
	let terminal = Bytes::from_static(
		b"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp\",\"status\":\"completed\",\"output\":[]}}\n\n",
	);
	let provider = load_builtin().unwrap().remove("openai-codex").unwrap();
	assert_eq!(provider.transport, TransportId::OpenAiCodex);
	let lease = CredentialLease::new("openai-codex", 91, 3);

	let (body, control) = ControlledBody::new();
	control.push(terminal.clone());
	let egress = CaptureService::new(body, "text/event-stream");
	let capture = CaptureService { shared: Arc::clone(&egress.shared), id: egress.id };
	let mut adapter =
		ProviderAttempt::new(provider.clone(), ProviderRoute::default(), egress).unwrap();
	let lite_policy = Arc::new(ResolvedModelPolicy {
		use_responses_lite: Some(true),
		prefer_websockets: Some(false),
		..ResolvedModelPolicy::default()
	});
	let _stream = adapter
		.ready()
		.await
		.unwrap()
		.call(Routed::new(request(), Some(lease.clone()), None).with_model_policy(Some(lite_policy)))
		.await
		.unwrap();
	let lite = capture.take_request();
	assert_eq!(lite.uri(), "https://chatgpt.com/backend-api/codex/responses");
	assert_eq!(lite.headers()["x-openai-internal-codex-responses-lite"], "true");
	assert!(lite.extensions().get::<CodexWebSocketRequest>().is_none());
	assert_eq!(lite.extensions().get::<CredentialLease>(), Some(&lease));
	let encoded = lite.into_body().collect().await.unwrap().to_bytes();
	let encoded: Value = serde_json::from_slice(&encoded).unwrap();
	assert_eq!(encoded["store"], false);

	let mut full_provider = provider;
	full_provider.codex_responses_lite = true;
	full_provider.codex_transport = CodexTransportPreference::HttpOnly;
	let (body, control) = ControlledBody::new();
	control.push(terminal);
	let egress = CaptureService::new(body, "text/event-stream");
	let capture = CaptureService { shared: Arc::clone(&egress.shared), id: egress.id };
	let mut adapter = ProviderAttempt::new(full_provider, ProviderRoute::default(), egress).unwrap();
	let full_policy = Arc::new(ResolvedModelPolicy {
		use_responses_lite: Some(false),
		prefer_websockets: Some(true),
		..ResolvedModelPolicy::default()
	});
	let _stream = adapter
		.ready()
		.await
		.unwrap()
		.call(Routed::new(request(), Some(lease), None).with_model_policy(Some(full_policy)))
		.await
		.unwrap();
	let full = capture.take_request();
	assert!(
		!full
			.headers()
			.contains_key("x-openai-internal-codex-responses-lite")
	);
	let websocket = full
		.extensions()
		.get::<CodexWebSocketRequest>()
		.expect("per-model websocket preference");
	assert!(!websocket.responses_lite);
}
#[tokio::test]
async fn vertex_global_and_regional_locations_expand_to_their_canonical_hosts() {
	for (location, expected) in [
		(
			"global",
			"https://aiplatform.googleapis.com/v1/projects/test-project/locations/global/publishers/google/models/test-model:streamGenerateContent?alt=sse",
		),
		(
			"us-central1",
			"https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1/publishers/google/models/test-model:streamGenerateContent?alt=sse",
		),
	] {
		let provider = load_builtin().unwrap().remove("google-vertex").unwrap();
		let (body, control) = ControlledBody::new();
		control.fail();
		let egress = CaptureService::new(body, "text/event-stream");
		let capture = CaptureService { shared: Arc::clone(&egress.shared), id: egress.id };
		let mut adapter = ProviderAttempt::new(
			provider,
			ProviderRoute {
				project: "test-project".into(),
				region: location.into(),
				..ProviderRoute::default()
			},
			egress,
		)
		.unwrap();
		let result = adapter.ready().await.unwrap().call(routed()).await;
		assert!(matches!(result, Err(ProviderAttemptError::Body(BodyFailure))));
		assert_eq!(capture.take_request().uri(), expected);
	}
}

#[tokio::test]
async fn headers_and_control_frames_do_not_commit_before_first_decoded_event() {
	let (body, control) = ControlledBody::new();
	control.push(Bytes::from_static(b": keep-alive\n\n"));
	let (mut adapter, _capture) = openai_service(body, "text/event-stream");
	let future = adapter.ready().await.unwrap().call(routed());
	futures::pin_mut!(future);
	assert!(
		tokio::time::timeout(Duration::from_millis(10), &mut future)
			.await
			.is_err()
	);
	control.push(content_chunk("committed"));
	let mut stream = future.await.unwrap();
	let first = stream.next().await.unwrap();
	assert!(matches!(first.event, Some(turn_event::Event::PartStart(_))));
}

#[tokio::test]
async fn decode_and_body_failures_before_first_event_are_service_errors() {
	let (body, control) = ControlledBody::new();
	control.push(Bytes::from_static(b"data: not-json\n\n"));
	let (mut adapter, _capture) = openai_service(body, "text/event-stream");
	assert!(matches!(
		adapter.ready().await.unwrap().call(routed()).await,
		Err(ProviderAttemptError::Decode(_))
	));

	let (body, control) = ControlledBody::new();
	control.fail();
	let (mut adapter, _capture) = openai_service(body, "text/event-stream");
	assert!(matches!(
		adapter.ready().await.unwrap().call(routed()).await,
		Err(ProviderAttemptError::Body(BodyFailure))
	));
}

#[tokio::test]
async fn post_commit_decode_failure_is_one_terminal_in_band_error() {
	let (body, control) = ControlledBody::new();
	control.push(content_chunk("visible"));
	let (mut adapter, _capture) = openai_service(body, "text/event-stream");
	let mut stream = adapter.ready().await.unwrap().call(routed()).await.unwrap();
	assert!(matches!(stream.next().await.unwrap().event, Some(turn_event::Event::PartStart(_))));
	assert!(matches!(stream.next().await.unwrap().event, Some(turn_event::Event::PartDelta(_))));
	control.push(Bytes::from_static(b"data: not-json\n\n"));
	control.push(terminal_chunk());
	assert!(matches!(stream.next().await.unwrap().event, Some(turn_event::Event::Error(_))));
	assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn successful_sentinel_produces_exactly_one_terminal_outcome() {
	let (body, control) = ControlledBody::new();
	control.push(content_chunk("visible"));
	control.push(Bytes::from(format!(
		"{}data: [DONE]\n\n",
		String::from_utf8(terminal_chunk().to_vec()).unwrap()
	)));
	let (mut adapter, _capture) = openai_service(body, "text/event-stream");
	let mut stream = adapter.ready().await.unwrap().call(routed()).await.unwrap();
	let mut outcomes = 0;
	while let Some(event) = stream.next().await {
		if matches!(event.event, Some(turn_event::Event::Outcome(_))) {
			outcomes += 1;
		}
	}
	assert_eq!(outcomes, 1);
}

#[tokio::test]
async fn dropping_committed_stream_drops_live_response_body() {
	let (body, control) = ControlledBody::new();
	control.push(content_chunk("visible"));
	let (mut adapter, _capture) = openai_service(body, "text/event-stream");
	let stream = adapter.ready().await.unwrap().call(routed()).await.unwrap();
	assert!(!control.dropped(), "non-terminal first event must retain the response body");
	drop(stream);
	assert!(control.dropped(), "dropping the caller stream must cancel the response body");
}

#[tokio::test]
async fn ndjson_and_raw_json_are_incrementally_framed() {
	for (content_type, protocol, first, second) in [
		(
			"application/x-ndjson",
			StreamProtocol::Ndjson,
			Bytes::from_static(b"{\"id\":\"chatcmpl_test\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ndjson\"},\"finish_reason\":null}]}\n"),
			Bytes::from_static(b"{\"id\":\"chatcmpl_test\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n"),
		),
		(
			"application/json",
			StreamProtocol::SseData,
			Bytes::from_static(b"{\"id\":\"chatcmpl_test\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"raw\"},\"finish_reason\":\"stop\"}]}"),
			Bytes::new(),
		),
	] {
		let (body, control) = ControlledBody::new();
		let (mut adapter, _capture) = openai_service(body, content_type);
		control.push(first);
		if !second.is_empty() {
			control.push(second);
		}
		control.finish();
		let mut stream = adapter.ready().await.unwrap().call(routed()).await.unwrap();
		let mut terminal = 0;
		while let Some(event) = stream.next().await {
			if matches!(
				event.event,
				Some(turn_event::Event::Outcome(_)) | Some(turn_event::Event::Error(_))
			) {
				terminal += 1;
			}
		}
		assert_eq!(terminal, 1, "{protocol:?} did not produce one terminal event");
	}
}
