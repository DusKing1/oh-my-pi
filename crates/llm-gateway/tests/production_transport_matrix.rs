#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

//! Production registration proof across every catalog chat transport.

use std::{
	collections::{BTreeMap, BTreeSet, VecDeque},
	convert::Infallible,
	future::Future,
	pin::Pin,
	sync::Arc,
	task::{Context, Poll},
	time::Duration,
};

use async_trait::async_trait;
use bytes::Bytes;
use futures::{SinkExt as _, StreamExt as _};
use http::{HeaderMap, HeaderValue, Request, Response, header};
use http_body_util::BodyExt as _;
use hyper::{
	body::{Body as HttpBody, Frame, Incoming, SizeHint},
	server::conn::http2,
	service::service_fn as hyper_service_fn,
};
use hyper_util::{
	client::legacy::{Client, connect::HttpConnector},
	rt::{TokioExecutor, TokioIo},
};
use omp_core::{Str, fmts};
use omp_llm_broker::{
	source::{
		BrokerCredentialSource, CredentialRefresher as BrokerCredentialRefresher,
		SpecializedCredentialAuth,
	},
	store::Store,
};
use omp_llm_catalog::{
	codex::{CODEX_CLIENT_VERSION, CODEX_ORIGINATOR},
	compat::{Compat, StreamProtocol},
	models::{Availability, Modality, ModelCard, ModelCatalog, Source},
	oauth_params,
	provider::{AuthSpec, Facet, ProviderCatalog, ProviderEntry, TransportId, load_builtin},
	registry::{CredentialView, Registry},
};
use omp_llm_cursor::{CursorChat, wire as cursor_wire};
use omp_llm_devin::{DevinChat, wire as devin_wire};
use omp_llm_egress::{
	auth_inject::{AuthInjectLayer, CredentialMetadataSource as _, CredentialSource},
	limits::{KeyedLimitsLayer, LimitConfig},
};
use omp_llm_error::{Classification, Feature, policy::BlockTable};
use omp_llm_gateway::{
	routes::{SpecializedChats, register_production_routes},
	turn::{ChatResolver, RoutedChat},
};
use omp_llm_gitlab::{GitLabDuoChat, WorkflowAuth, WorkflowConfig};
use omp_llm_local::{Embedded, Inference, TextSelection};
use omp_llm_tower::{
	learn::RequestRepair,
	preflight::{Admission, UsageOracle},
	provider::ProviderRoute,
	refresh::{CredentialRefresher, RefreshFailure},
	select::{
		CredentialCandidates, CredentialLease, CredentialMetadata, CredentialPool, LeaseSource,
	},
	stack::builder::{RouteDependencies, RouteStackConfig},
	tap::FrameSink,
};
use omp_llm_transport::omp::OmpFederation;
use omp_llm_types::{
	ChatRequest, Error, Executor, Invoke, InvokeComplete, InvokeInput, Item, ItemKind, Message,
	Part, Props, Role, Thread, TurnEvent, facet::Chat,
};
use omp_proto::inference::v1::{self as pb, inference_client::InferenceClient};
use parking_lot::{Mutex, RwLock};
use prost::Message as _;
use smallvec::smallvec;
use tokio::net::TcpListener;
use tokio_tungstenite::{accept_hdr_async, tungstenite::Message as WsMessage};
use tonic::transport::Endpoint;
use tower::Layer as _;

const SECRET: &str = "matrix-lease-secret";
const PROJECT: &str = "matrix-project";
const REGION: &str = "us-central1";

const HTTP_TRANSPORTS: [TransportId; 11] = [
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
	TransportId::OllamaChat,
];

const STREAMED_TRANSPORTS: [TransportId; 15] = [
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
	TransportId::OllamaChat,
	TransportId::Cursor,
	TransportId::Devin,
	TransportId::GitLabDuoWorkflow,
	TransportId::Omp,
];

#[derive(Clone)]
struct Credentials;

impl CredentialView for Credentials {
	fn availability(&self, _provider: &str) -> Availability {
		Availability::Available
	}
}

struct Allow;
impl UsageOracle for Allow {
	fn admit(&self, _model: &str) -> Admission {
		Admission::Allow
	}
}

struct OneCredential(u64);
impl CredentialPool for OneCredential {
	fn candidates(&self, _model: &str) -> CredentialCandidates {
		smallvec![self.0]
	}
}

#[derive(Clone)]
struct ProviderLease {
	lease:    CredentialLease,
	metadata: CredentialMetadata,
}

impl LeaseSource for ProviderLease {
	fn lease(&self, id: u64) -> Option<CredentialLease> {
		(id == self.lease.credential_id()).then(|| self.lease.clone())
	}

	fn metadata(&self, lease: &CredentialLease) -> Option<CredentialMetadata> {
		(lease == &self.lease).then(|| self.metadata.clone())
	}
}

struct NoBrokerRefresh;
impl BrokerCredentialRefresher for NoBrokerRefresh {
	type Error = Infallible;

	fn refresh(
		&self,
		_credential_id: u64,
		_now_ms: u64,
	) -> impl Future<Output = Result<(), Self::Error>> + Send {
		std::future::ready(Ok(()))
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
		_request: &pb::TurnRequest,
		_feature: Feature,
		_classification: &Classification,
	) -> Option<pb::TurnRequest> {
		None
	}
}

struct NoopSink;
impl FrameSink for NoopSink {
	fn on_request(&self, _request: &pb::TurnRequest) {}

	fn on_frame(&self, _frame: &pb::TurnEvent) {}

	fn on_end(&self) {}
}

fn dependencies(lease: CredentialLease, metadata: CredentialMetadata) -> RouteDependencies {
	RouteDependencies {
		usage:          Arc::new(Allow),
		credentials:    Arc::new(OneCredential(lease.credential_id())),
		leases:         Arc::new(ProviderLease { lease, metadata }),
		refresher:      Arc::new(Fresh),
		repair:         Arc::new(NoRepair),
		observer:       Arc::new(NoopSink),
		usage_observer: Arc::new(omp_llm_tower::stack::meter::NoopUsageObserver),
		blocks:         Arc::new(parking_lot::Mutex::new(BlockTable::default())),
	}
}

#[derive(Clone, Debug)]
struct Captured {
	transport: TransportId,
	path:      String,
	headers:   HeaderMap,
	body:      Bytes,
}

struct SocketState {
	calls:     Mutex<BTreeMap<TransportId, usize>>,
	captured:  Mutex<Vec<Captured>>,
	cancelled: flume::Sender<TransportId>,
}

impl SocketState {
	async fn respond(
		self: Arc<Self>,
		mut request: Request<Incoming>,
	) -> Result<Response<MatrixBody>, Infallible> {
		let path = request.uri().path().to_owned();
		if path == "/exa.auth_pb.AuthService/GetUserJwt" {
			let body = request
				.body_mut()
				.collect()
				.await
				.expect("read Devin auth request")
				.to_bytes();
			let auth = devin_wire::GetUserJwtRequest::decode(grpc_payload(&body))
				.expect("Devin auth protobuf request");
			let metadata = auth.metadata.expect("Devin auth metadata");
			assert_eq!(metadata.api_key, "devin-session-token$devin-key");
			let response = devin_wire::GetUserJwtResponse {
				user_jwt:              "devin-jwt".into(),
				custom_api_server_url: String::new(),
			};
			return Ok(Response::builder()
				.status(200)
				.header(header::CONTENT_TYPE, "application/grpc")
				.body(MatrixBody::finite(vec![grpc_message(&response)], Some(grpc_ok_trailers())))
				.expect("Devin auth fixture response"));
		}
		let transport = identify_transport(&request, &path);
		let body = if matches!(transport, TransportId::Cursor | TransportId::Omp) {
			request
				.body_mut()
				.frame()
				.await
				.and_then(Result::ok)
				.and_then(|frame| frame.into_data().ok())
				.unwrap_or_default()
		} else {
			request
				.body_mut()
				.collect()
				.await
				.expect("read fixture request")
				.to_bytes()
		};
		verify_request(transport, &path, request.headers(), &body);
		self.captured.lock().push(Captured {
			transport,
			path,
			headers: request.headers().clone(),
			body,
		});
		let call = {
			let mut calls = self.calls.lock();
			let call = calls.entry(transport).or_default();
			*call += 1;
			*call
		};
		let cancelling = call == 2;
		let (content_type, chunks, trailers) = response_frames(transport, cancelling);
		let response_body = if cancelling {
			MatrixBody::streaming(chunks, transport, self.cancelled.clone())
		} else {
			MatrixBody::finite(chunks, trailers)
		};
		Ok(Response::builder()
			.status(200)
			.header(header::CONTENT_TYPE, content_type)
			.body(response_body)
			.expect("fixture response"))
	}
}

fn identify_transport(request: &Request<Incoming>, path: &str) -> TransportId {
	if path == "/agent.v1.AgentService/Run" {
		return TransportId::Cursor;
	}
	if path == "/exa.api_server_pb.ApiServerService/GetChatMessage" {
		return TransportId::Devin;
	}
	if path == "/omp.inference.v1.Inference/Turn" {
		return TransportId::Omp;
	}
	let value = request.headers()["x-matrix-transport"]
		.to_str()
		.expect("transport header");
	transport_name(value).expect("known matrix transport")
}

fn verify_request(transport: TransportId, path: &str, headers: &HeaderMap, body: &[u8]) {
	match transport {
		TransportId::Cursor => {
			assert_eq!(headers[header::AUTHORIZATION], "Bearer cursor-secret");
			assert_eq!(headers[header::CONTENT_TYPE], "application/connect+proto");
			assert!(!body.is_empty(), "Cursor Connect open frame");
		},
		TransportId::Devin => {
			let message = grpc_payload(body);
			let request =
				devin_wire::GetChatMessageRequest::decode(message).expect("Devin protobuf request");
			let metadata = request.metadata.expect("Devin auth metadata");
			assert_eq!(metadata.api_key, "devin-session-token$devin-key");
			assert_eq!(metadata.user_jwt, "devin-jwt");
		},
		TransportId::Omp => {
			let frame = pb::TurnFrame::decode(grpc_payload(body)).expect("OMP open frame");
			let pb::turn_frame::Frame::Open(open) = frame.frame.expect("OMP frame") else {
				panic!("first OMP frame must open the turn")
			};
			assert!(open.params.is_some(), "OMP request encoding");
		},
		TransportId::AnthropicBedrock => {
			assert!(
				headers[header::AUTHORIZATION]
					.to_str()
					.expect("Bedrock authorization")
					.starts_with("AWS4-HMAC-SHA256 Credential=AKIDMATRIX/"),
				"Bedrock request is signed by BrokerCredentialSource"
			);
			assert!(headers.contains_key("x-amz-date"));
			assert_eq!(headers["x-amz-security-token"], "matrix-session");
			assert!(!body.is_empty(), "Bedrock encoded request body");
			assert!(path.contains("/matrix/model/matrix-model/invoke-with-response-stream"));
			serde_json::from_slice::<serde_json::Value>(body).expect("Bedrock provider request JSON");
		},
		TransportId::BedrockConverse => {
			assert!(
				headers[header::AUTHORIZATION]
					.to_str()
					.expect("Bedrock authorization")
					.starts_with("AWS4-HMAC-SHA256 Credential=AKIDMATRIX/"),
				"Bedrock Converse request is signed by BrokerCredentialSource"
			);
			assert!(headers.contains_key("x-amz-date"));
			assert_eq!(headers["x-amz-security-token"], "matrix-session");
			assert!(path.contains("/matrix/model/matrix-model/converse-stream"));
			let body: serde_json::Value =
				serde_json::from_slice(body).expect("Bedrock Converse request JSON");
			assert_eq!(body["messages"][0]["role"], "user");
			assert!(body.get("anthropic_version").is_none());
		},
		_ => {
			assert_eq!(headers[header::AUTHORIZATION], format!("Bearer {SECRET}"));
			assert!(!body.is_empty(), "{transport:?} encoded request body");
			assert!(
				path.contains(expected_path_fragment(transport)),
				"{transport:?} uses its catalog endpoint"
			);
			let encoded: serde_json::Value =
				serde_json::from_slice(body).expect("HTTP provider request JSON");
			match transport {
				TransportId::OpenAiCodex => {
					assert_eq!(encoded["model"], "matrix-model");
					let login = oauth_params::load_embedded().expect("bundled OAuth rows");
					let row = oauth_params::lookup(&login, "openai-codex").expect("Codex login row");
					let authorized = row
						.extra_auth_params
						.get("originator")
						.expect("Codex login authorizes an originator");
					assert_eq!(
						headers["originator"],
						*authorized.as_str(),
						"Codex requests must present the originator its credential was minted for"
					);
					assert_eq!(headers["originator"], CODEX_ORIGINATOR);
					assert_eq!(headers["version"], CODEX_CLIENT_VERSION);
				},
				TransportId::OpenAiChat
				| TransportId::OpenAiResponses
				| TransportId::AnthropicMessages => {
					assert_eq!(encoded["model"], "matrix-model");
				},
				TransportId::AnthropicVertex | TransportId::GoogleGenAi | TransportId::GoogleVertex => {
					assert!(encoded.is_object(), "{transport:?} request envelope");
				},
				TransportId::GoogleCca => {
					assert_eq!(encoded["project"], PROJECT);
				},
				TransportId::OllamaChat => {
					assert_eq!(encoded["model"], "matrix-model");
					assert_eq!(encoded["stream"], true);
					assert_eq!(headers[header::ACCEPT], "application/x-ndjson");
				},
				_ => unreachable!("specialized requests handled above"),
			}
		},
	}
}

const fn expected_path_fragment(transport: TransportId) -> &'static str {
	match transport {
		TransportId::OpenAiChat => "/matrix/chat/completions",
		TransportId::OpenAiResponses => "/matrix/responses",
		TransportId::OpenAiCodex => "/matrix/codex/responses",
		TransportId::AnthropicMessages => "/matrix/v1/messages",
		TransportId::AnthropicVertex => "/publishers/anthropic/models/matrix-model:streamRawPredict",
		TransportId::GoogleGenAi => "/matrix/models/matrix-model:streamGenerateContent",
		TransportId::GoogleVertex => "/publishers/google/models/matrix-model:streamGenerateContent",
		TransportId::GoogleCca => "/matrix/v1internal:streamGenerateContent",
		TransportId::OllamaChat => "/matrix/api/chat",
		_ => "",
	}
}
struct MatrixBody {
	frames: VecDeque<Frame<Bytes>>,
	hangs:  bool,
	cancel: Option<(TransportId, flume::Sender<TransportId>)>,
}

impl MatrixBody {
	fn finite(chunks: Vec<Bytes>, trailers: Option<HeaderMap>) -> Self {
		let mut frames = chunks.into_iter().map(Frame::data).collect::<VecDeque<_>>();
		if let Some(trailers) = trailers {
			frames.push_back(Frame::trailers(trailers));
		}
		Self { frames, hangs: false, cancel: None }
	}

	fn streaming(
		chunks: Vec<Bytes>,
		transport: TransportId,
		tx: flume::Sender<TransportId>,
	) -> Self {
		Self {
			frames: chunks.into_iter().map(Frame::data).collect(),
			hangs:  true,
			cancel: Some((transport, tx)),
		}
	}
}

impl Drop for MatrixBody {
	fn drop(&mut self) {
		if let Some((transport, tx)) = self.cancel.take() {
			let _ = tx.send(transport);
		}
	}
}

impl HttpBody for MatrixBody {
	type Data = Bytes;
	type Error = Infallible;

	fn poll_frame(
		mut self: Pin<&mut Self>,
		_cx: &mut Context<'_>,
	) -> Poll<Option<Result<Frame<Bytes>, Infallible>>> {
		if let Some(frame) = self.frames.pop_front() {
			return Poll::Ready(Some(Ok(frame)));
		}
		if self.hangs {
			Poll::Pending
		} else {
			Poll::Ready(None)
		}
	}

	fn is_end_stream(&self) -> bool {
		!self.hangs && self.frames.is_empty()
	}

	fn size_hint(&self) -> SizeHint {
		SizeHint::default()
	}
}

fn response_frames(
	transport: TransportId,
	cancelling: bool,
) -> (&'static str, Vec<Bytes>, Option<HeaderMap>) {
	if transport == TransportId::Cursor {
		let messages = if cancelling {
			cursor_messages(false)
		} else {
			cursor_messages(true)
		};
		return ("application/connect+proto", vec![messages], None);
	}
	if transport == TransportId::Devin {
		let response = devin_wire::GetChatMessageResponse {
			message_id: "devin-message".into(),
			delta_text: "matrix".into(),
			stop_reason: if cancelling {
				0
			} else {
				devin_wire::StopReason::StopPattern as i32
			},
			usage: (!cancelling).then_some(devin_wire::ModelUsageStats {
				input_tokens:       3,
				output_tokens:      2,
				cache_write_tokens: 0,
				cache_read_tokens:  1,
			}),
			..Default::default()
		};
		return (
			"application/grpc",
			vec![grpc_message(&response)],
			(!cancelling).then(grpc_ok_trailers),
		);
	}
	if transport == TransportId::Omp {
		let delta = pb::TurnEvent {
			event: Some(pb::turn_event::Event::PartDelta(pb::PartDelta {
				index: 0,
				chunk: Bytes::from_static(b"matrix"),
			})),
		};
		let mut chunks = vec![grpc_message(&delta)];
		if !cancelling {
			let outcome = pb::TurnEvent {
				event: Some(pb::turn_event::Event::Outcome(pb::Outcome {
					stop: pb::StopReason::StopEndTurn as i32,
					usage: Some(pb::Usage {
						input_tokens: 3,
						output_tokens: 2,
						cache_read_tokens: 1,
						accuracy: pb::usage::Accuracy::Exact as i32,
						..Default::default()
					}),
					provider: "omp-upstream".into(),
					model: "matrix-model".into(),
					..Default::default()
				})),
			};
			chunks.push(grpc_message(&outcome));
		}
		return ("application/grpc", chunks, (!cancelling).then(grpc_ok_trailers));
	}
	if transport == TransportId::AnthropicBedrock {
		let events = anthropic_events(cancelling)
			.into_iter()
			.map(|event| bedrock_event(&event))
			.collect();
		return ("application/vnd.amazon.eventstream", events, None);
	}
	if transport == TransportId::BedrockConverse {
		return ("application/vnd.amazon.eventstream", bedrock_converse_events(cancelling), None);
	}
	if transport == TransportId::OllamaChat {
		return ("application/x-ndjson", vec![ndjson(&ollama_events(cancelling))], None);
	}
	let events = match transport {
		TransportId::OpenAiChat => openai_chat_events(cancelling),
		TransportId::OpenAiResponses | TransportId::OpenAiCodex => response_api_events(cancelling),
		TransportId::AnthropicMessages | TransportId::AnthropicVertex => anthropic_events(cancelling),
		TransportId::GoogleGenAi | TransportId::GoogleVertex => google_events(cancelling, false),
		TransportId::GoogleCca => google_events(cancelling, true),
		_ => unreachable!("specialized response handled above"),
	};
	("text/event-stream", vec![sse(&events)], None)
}

fn openai_chat_events(cancelling: bool) -> Vec<Vec<u8>> {
	let mut events = vec![br#"{"id":"chatcmpl_matrix","choices":[{"index":0,"delta":{"content":"matrix"},"finish_reason":null}]}"#.to_vec()];
	if !cancelling {
		events.push(
			br#"{"id":"chatcmpl_matrix","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#
				.to_vec(),
		);
		events.push(br#"{"id":"chatcmpl_matrix","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":1}}}"#.to_vec());
		events.push(b"[DONE]".to_vec());
	}
	events
}

fn response_api_events(cancelling: bool) -> Vec<Vec<u8>> {
	let mut events = vec![br#"{"type":"response.output_item.added","output_index":0,"item":{"id":"msg_matrix","type":"message","role":"assistant","content":[]}}"#.to_vec(), br#"{"type":"response.output_text.delta","output_index":0,"delta":"matrix"}"#.to_vec()];
	if !cancelling {
		events.push(br#"{"type":"response.completed","response":{"id":"resp_matrix","model":"matrix-model","status":"completed","usage":{"input_tokens":3,"output_tokens":2,"input_tokens_details":{"cached_tokens":1}}}}"#.to_vec());
	}
	events
}

fn ollama_events(cancelling: bool) -> Vec<Vec<u8>> {
	let mut events =
		vec![br#"{"message":{"role":"assistant","content":"matrix"},"done":false}"#.to_vec()];
	if !cancelling {
		events.push(
			br#"{"message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":3,"eval_count":2}"#
				.to_vec(),
		);
	}
	events
}

fn anthropic_events(cancelling: bool) -> Vec<Vec<u8>> {
	let mut events = vec![
		br#"{"type":"message_start","message":{"model":"matrix-model","usage":{"input_tokens":3,"cache_read_input_tokens":1}}}"#.to_vec(),
		br#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#.to_vec(),
		br#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"matrix"}}"#.to_vec(),
	];
	if !cancelling {
		events.extend([
			br#"{"type":"content_block_stop","index":0}"#.to_vec(),
			br#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}"#.to_vec(),
			br#"{"type":"message_stop"}"#.to_vec(),
		]);
	}
	events
}

fn bedrock_converse_events(cancelling: bool) -> Vec<Bytes> {
	let mut events = vec![
		bedrock_raw_event("messageStart", br#"{"role":"assistant"}"#),
		bedrock_raw_event(
			"contentBlockDelta",
			br#"{"contentBlockIndex":0,"delta":{"text":"matrix"}}"#,
		),
	];
	if !cancelling {
		events.extend([
			bedrock_raw_event("contentBlockStop", br#"{"contentBlockIndex":0}"#),
			bedrock_raw_event("messageStop", br#"{"stopReason":"end_turn"}"#),
			bedrock_raw_event(
				"metadata",
				br#"{"usage":{"inputTokens":3,"outputTokens":2,"totalTokens":5,"cacheReadInputTokens":1},"metrics":{"latencyMs":7}}"#,
			),
		]);
	}
	events
}

fn google_events(cancelling: bool, cca: bool) -> Vec<Vec<u8>> {
	let response = if cancelling {
		serde_json::json!({"candidates":[{"content":{"parts":[{"text":"matrix"}]}}]})
	} else {
		serde_json::json!({"candidates":[{"content":{"parts":[{"text":"matrix"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":2,"cachedContentTokenCount":1}})
	};
	let response = if cca {
		serde_json::json!({"response": response})
	} else {
		response
	};
	vec![serde_json::to_vec(&response).expect("Google fixture JSON")]
}

fn sse(events: &[Vec<u8>]) -> Bytes {
	let mut bytes = Vec::new();
	for event in events {
		bytes.extend_from_slice(b"data: ");
		bytes.extend_from_slice(event);
		bytes.extend_from_slice(b"\n\n");
	}
	Bytes::from(bytes)
}

fn ndjson(events: &[Vec<u8>]) -> Bytes {
	let mut bytes = Vec::new();
	for event in events {
		bytes.extend_from_slice(event);
		bytes.push(b'\n');
	}
	Bytes::from(bytes)
}

fn cursor_messages(terminal: bool) -> Bytes {
	use cursor_wire::interaction_update::Message;
	let mut output = Vec::new();
	let mut push = |message| {
		let message = cursor_wire::AgentServerMessage {
			message: Some(cursor_wire::agent_server_message::Message::InteractionUpdate(Box::new(
				cursor_wire::InteractionUpdate { message: Some(message) },
			))),
		};
		output.extend_from_slice(&connect_message(&message));
	};
	push(Message::TextDelta(cursor_wire::TextDeltaUpdate { text: "matrix".into() }));
	if terminal {
		push(Message::TokenDelta(cursor_wire::TokenDeltaUpdate { tokens: 2 }));
		push(Message::TurnEnded(cursor_wire::TurnEndedUpdate {}));
	}
	Bytes::from(output)
}

fn connect_message(message: &impl prost::Message) -> Bytes {
	let payload = message.encode_to_vec();
	let mut output = Vec::with_capacity(payload.len() + 5);
	output.push(0);
	output.extend_from_slice(
		&u32::try_from(payload.len())
			.expect("Connect fixture size")
			.to_be_bytes(),
	);
	output.extend_from_slice(&payload);
	Bytes::from(output)
}

fn grpc_message(message: &impl prost::Message) -> Bytes {
	connect_message(message)
}

fn grpc_payload(frame: &[u8]) -> &[u8] {
	assert!(frame.len() >= 5 && frame[0] == 0, "uncompressed gRPC/Connect frame");
	let len = u32::from_be_bytes(frame[1..5].try_into().expect("gRPC length")) as usize;
	assert!(frame.len() >= 5 + len, "complete gRPC frame");
	&frame[5..5 + len]
}

fn grpc_ok_trailers() -> HeaderMap {
	let mut trailers = HeaderMap::new();
	trailers.insert("grpc-status", HeaderValue::from_static("0"));
	trailers
}

fn bedrock_event(event: &[u8]) -> Bytes {
	let payload = serde_json::to_vec(&serde_json::json!({ "bytes": base64_bytes(event) }))
		.expect("Bedrock payload");
	bedrock_raw_event("chunk", &payload)
}

fn bedrock_raw_event(event_type: &str, payload: &[u8]) -> Bytes {
	let mut headers = Vec::new();
	string_header(&mut headers, ":message-type", "event");
	string_header(&mut headers, ":event-type", event_type);
	string_header(&mut headers, ":content-type", "application/json");
	let total_len = 16 + headers.len() + payload.len();
	let mut message = Vec::with_capacity(total_len);
	message.extend_from_slice(
		&u32::try_from(total_len)
			.expect("eventstream size")
			.to_be_bytes(),
	);
	message.extend_from_slice(
		&u32::try_from(headers.len())
			.expect("eventstream headers")
			.to_be_bytes(),
	);
	message.extend_from_slice(&crc32fast::hash(&message).to_be_bytes());
	message.extend_from_slice(&headers);
	message.extend_from_slice(payload);
	message.extend_from_slice(&crc32fast::hash(&message).to_be_bytes());
	Bytes::from(message)
}

fn string_header(output: &mut Vec<u8>, name: &str, value: &str) {
	output.push(u8::try_from(name.len()).expect("header name"));
	output.extend_from_slice(name.as_bytes());
	output.push(7);
	output.extend_from_slice(
		&u16::try_from(value.len())
			.expect("header value")
			.to_be_bytes(),
	);
	output.extend_from_slice(value.as_bytes());
}

fn base64_bytes(bytes: &[u8]) -> String {
	const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
	let mut output = String::with_capacity(bytes.len().div_ceil(3) * 4);
	for chunk in bytes.chunks(3) {
		let value = (u32::from(chunk[0]) << 16)
			| (u32::from(*chunk.get(1).unwrap_or(&0)) << 8)
			| u32::from(*chunk.get(2).unwrap_or(&0));
		output.push(char::from(TABLE[((value >> 18) & 63) as usize]));
		output.push(char::from(TABLE[((value >> 12) & 63) as usize]));
		output.push(if chunk.len() > 1 {
			char::from(TABLE[((value >> 6) & 63) as usize])
		} else {
			'='
		});
		output.push(if chunk.len() > 2 {
			char::from(TABLE[(value & 63) as usize])
		} else {
			'='
		});
	}
	output
}

struct FixtureAuth;
#[async_trait]
impl WorkflowAuth for FixtureAuth {
	async fn apply(&self, headers: &mut HeaderMap) -> Result<(), Error> {
		headers.insert(header::AUTHORIZATION, HeaderValue::from_static("Bearer gitlab-lease"));
		Ok(())
	}
}

struct FixtureExecutor;
#[async_trait]
impl Executor for FixtureExecutor {
	async fn invoke(
		&self,
		_invocation: Invoke,
		_inputs: flume::Sender<InvokeInput>,
	) -> InvokeComplete {
		panic!("matrix transports do not invoke tools")
	}
}

async fn spawn_websocket_fixture() -> (String, tokio::task::JoinHandle<()>) {
	let listener = TcpListener::bind("127.0.0.1:0")
		.await
		.expect("bind GitLab fixture");
	let address = listener.local_addr().expect("GitLab address");
	let task = tokio::spawn(async move {
		for cancelling in [false, true] {
			let (stream, _) = listener.accept().await.expect("GitLab connection");
			#[allow(
				clippy::result_large_err,
				reason = "the error type is fixed by tungstenite's external callback signature"
			)]
			let handshake =
				|request: &tokio_tungstenite::tungstenite::handshake::server::Request,
				 response: tokio_tungstenite::tungstenite::handshake::server::Response| {
					assert_eq!(request.headers()[header::AUTHORIZATION], "Bearer gitlab-lease");
					Ok(response)
				};
			let mut socket = accept_hdr_async(stream, handshake)
				.await
				.expect("GitLab handshake");
			let start = socket
				.next()
				.await
				.expect("GitLab start")
				.expect("GitLab frame")
				.into_text()
				.expect("GitLab text");
			assert!(start.contains("startRequest"), "GitLab request encoding");
			if cancelling {
				socket
					.send(WsMessage::Text(
						serde_json::json!({"eventID":"cancel","text":"matrix"})
							.to_string()
							.into(),
					))
					.await
					.expect("GitLab cancel delta");
				assert!(
					matches!(socket.next().await, Some(Ok(WsMessage::Close(_)) | Err(_)) | None),
					"GitLab socket closes on stream drop"
				);
			} else {
				socket.send(WsMessage::Text(serde_json::json!({"eventID":"done","text":"matrix","agent_context_usage":{"Chat Agent":{"total_tokens":3,"max_tokens":4096}},"status":"FINISHED"}).to_string().into())).await.expect("GitLab terminal");
			}
		}
	});
	(format!("ws://{address}"), task)
}

fn provider(id: &'static str, transport: TransportId, base_url: String) -> ProviderEntry {
	let mut headers = BTreeMap::new();
	if HTTP_TRANSPORTS.contains(&transport) {
		headers
			.insert(Str::new_static("x-matrix-transport"), Str::new(transport_name_owned(transport)));
	}
	let auth = if matches!(transport, TransportId::AnthropicBedrock | TransportId::BedrockConverse) {
		AuthSpec::AwsSigV4
	} else if HTTP_TRANSPORTS.contains(&transport)
		|| matches!(transport, TransportId::Cursor | TransportId::Devin)
	{
		AuthSpec::Bearer { env: smallvec![] }
	} else {
		AuthSpec::None
	};
	let mut compat = Compat::default();
	if transport == TransportId::OllamaChat {
		compat.stream_protocol = StreamProtocol::Ndjson;
	}
	ProviderEntry::builder()
		.id(Str::new_static(id))
		.transport(transport)
		.base_url(Str::new(base_url))
		.auth(auth)
		.facets(smallvec![Facet::Chat])
		.headers(headers)
		.compat(compat)
		.build()
}

fn card(provider: &'static str) -> ModelCard {
	ModelCard::builder()
		.id(fmts!("{provider}/model"))
		.provider(Str::new_static(provider))
		.model(Str::new_static("matrix-model"))
		.name(Str::new_static("matrix-model"))
		.family(Str::new_static("matrix"))
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

fn request(provider: &'static str) -> ChatRequest {
	let message = Message::builder()
		.role(Role::User)
		.parts(vec![Part::Text(Str::new_static("matrix request"))])
		.build();
	let item = Item::builder()
		.seq(0)
		.kind(ItemKind::Message(message))
		.props(Props::default())
		.build();
	ChatRequest::builder()
		.model(fmts!("{provider}/model"))
		.thread(Thread::builder().items(vec![item]).build())
		.tools(Vec::new())
		.provider_options(Props::default())
		.build()
}

fn transport_name(transport: &str) -> Option<TransportId> {
	Some(match transport {
		"open-ai-chat" => TransportId::OpenAiChat,
		"open-ai-responses" => TransportId::OpenAiResponses,
		"open-ai-codex" => TransportId::OpenAiCodex,
		"anthropic-messages" => TransportId::AnthropicMessages,
		"anthropic-bedrock" => TransportId::AnthropicBedrock,
		"bedrock-converse" => TransportId::BedrockConverse,
		"anthropic-vertex" => TransportId::AnthropicVertex,
		"google-gen-ai" => TransportId::GoogleGenAi,
		"google-vertex" => TransportId::GoogleVertex,
		"google-cca" => TransportId::GoogleCca,
		"ollama-chat" => TransportId::OllamaChat,
		_ => return None,
	})
}

const fn transport_name_owned(transport: TransportId) -> &'static str {
	match transport {
		TransportId::OpenAiChat => "open-ai-chat",
		TransportId::OpenAiResponses => "open-ai-responses",
		TransportId::OpenAiCodex => "open-ai-codex",
		TransportId::AnthropicMessages => "anthropic-messages",
		TransportId::AnthropicBedrock => "anthropic-bedrock",
		TransportId::BedrockConverse => "bedrock-converse",
		TransportId::AnthropicVertex => "anthropic-vertex",
		TransportId::GoogleGenAi => "google-gen-ai",
		TransportId::GoogleVertex => "google-vertex",
		TransportId::GoogleCca => "google-cca",
		TransportId::OllamaChat => "ollama-chat",
		TransportId::Cursor => "cursor",
		TransportId::Devin => "devin",
		TransportId::GitLabDuoWorkflow => "gitlab-duo-workflow",
		TransportId::Omp => "omp",
		TransportId::Embedded => "embedded",
	}
}

#[tokio::test]
async fn generated_catalog_transports_complete_through_registered_production_routes_and_cancel_sockets()
 {
	let generated: BTreeSet<_> = load_builtin()
		.expect("generated provider catalog")
		.values()
		.filter(|provider| provider.facets.contains(&Facet::Chat))
		.map(|provider| provider.transport)
		.chain([
			TransportId::AnthropicBedrock,
			TransportId::AnthropicVertex,
			TransportId::Omp,
			TransportId::Embedded,
		])
		.collect();
	let tested: BTreeSet<_> = STREAMED_TRANSPORTS
		.into_iter()
		.chain([TransportId::Embedded])
		.collect();
	assert_eq!(generated, tested, "a generated production transport lacks a registration fixture");

	let listener = TcpListener::bind("127.0.0.1:0")
		.await
		.expect("bind production protocol fixture");
	let address = listener.local_addr().expect("fixture address");
	let (cancelled_tx, cancelled_rx) = flume::unbounded();
	let socket_state = Arc::new(SocketState {
		calls:     Mutex::new(BTreeMap::new()),
		captured:  Mutex::new(Vec::new()),
		cancelled: cancelled_tx,
	});
	let server_state = Arc::clone(&socket_state);
	let socket_server = tokio::spawn(async move {
		loop {
			let (socket, _) = listener
				.accept()
				.await
				.expect("accept production transport");
			let state = Arc::clone(&server_state);
			tokio::spawn(async move {
				let _ = http2::Builder::new(TokioExecutor::new())
					.serve_connection(
						TokioIo::new(socket),
						hyper_service_fn(move |request| Arc::clone(&state).respond(request)),
					)
					.await;
			});
		}
	});

	let (gitlab_url, gitlab_server) = spawn_websocket_fixture().await;
	let base = format!("http://{address}/matrix");
	let specs: [(&str, TransportId); 16] = [
		("openai-chat", TransportId::OpenAiChat),
		("openai-responses", TransportId::OpenAiResponses),
		("openai-codex", TransportId::OpenAiCodex),
		("anthropic", TransportId::AnthropicMessages),
		("bedrock", TransportId::AnthropicBedrock),
		("bedrock-converse", TransportId::BedrockConverse),
		("anthropic-vertex", TransportId::AnthropicVertex),
		("google", TransportId::GoogleGenAi),
		("google-vertex", TransportId::GoogleVertex),
		("google-cca", TransportId::GoogleCca),
		("ollama-cloud", TransportId::OllamaChat),
		("cursor", TransportId::Cursor),
		("devin", TransportId::Devin),
		("gitlab", TransportId::GitLabDuoWorkflow),
		("federated", TransportId::Omp),
		("embedded", TransportId::Embedded),
	];
	let providers = specs.map(|(id, transport)| {
		provider(
			id,
			transport,
			if transport == TransportId::GitLabDuoWorkflow {
				gitlab_url.clone()
			} else {
				base.clone()
			},
		)
	});
	let cards = specs.map(|(id, _)| card(id)).to_vec();
	let catalog = ModelCatalog::new(cards);
	let resolver = Arc::new(ChatResolver::new(Arc::new(RwLock::new(Registry::new(
		&catalog,
		Arc::new(Credentials),
	)))));

	let broker_directory = tempfile::tempdir().expect("broker directory");
	let broker_store = Arc::new(
		Store::open(broker_directory.path().join("broker.sqlite")).expect("production broker store"),
	);
	for provider in providers
		.iter()
		.filter(|provider| HTTP_TRANSPORTS.contains(&provider.transport))
	{
		if matches!(provider.transport, TransportId::AnthropicBedrock | TransportId::BedrockConverse)
		{
			broker_store
				.upsert_aws(
					provider.id.as_str(),
					"matrix-account",
					b"AKIDMATRIX",
					b"matrix-secret-signing-key",
					Some(b"matrix-session"),
					1,
				)
				.expect("insert Bedrock credential");
		} else if provider.transport == TransportId::GoogleCca {
			broker_store
				.upsert_minted_bearer(
					provider.id.as_str(),
					"matrix-account",
					SECRET.as_bytes(),
					4_102_444_800_000,
					&serde_json::json!({ "antigravity": { "project_id": PROJECT } }),
					1,
				)
				.expect("insert CCA credential with project identity");
		} else {
			broker_store
				.upsert_api_key(provider.id.as_str(), "matrix-account", SECRET.as_bytes(), 1)
				.expect("insert provider credential");
		}
	}
	for (provider, secret) in
		[("cursor", b"cursor-secret".as_slice()), ("devin", b"devin-key".as_slice())]
	{
		broker_store
			.upsert_api_key(provider, "matrix-account", secret, 1)
			.expect("insert specialized provider credential");
	}
	let provider_catalog: ProviderCatalog = providers
		.iter()
		.cloned()
		.map(|provider| (provider.id.clone(), provider))
		.collect();
	let broker = BrokerCredentialSource::new(
		Arc::clone(&broker_store),
		Arc::new(provider_catalog),
		Arc::new(NoBrokerRefresh),
	);
	let routed_leases: Arc<BTreeMap<_, _>> = Arc::new(
		providers
			.iter()
			.filter(|provider| HTTP_TRANSPORTS.contains(&provider.transport))
			.map(|provider| {
				let lease = broker
					.lease(provider.id.as_str())
					.expect("broker lease lookup")
					.expect("provider credential lease");
				let metadata = broker.metadata(&lease).expect("broker credential metadata");
				(provider.id.clone(), (lease, metadata))
			})
			.collect(),
	);

	let mut connector = HttpConnector::new();
	connector.enforce_http(true);
	let hyper = Client::builder(TokioExecutor::new())
		.http2_only(true)
		.build(connector);
	let limits = KeyedLimitsLayer::new(LimitConfig::default()).layer(hyper);
	let cursor: Arc<dyn Chat> = Arc::new(CursorChat::new(
		format!("http://{address}"),
		SpecializedCredentialAuth::new(broker.clone(), "cursor"),
	));
	let devin_channel = Endpoint::from_shared(format!("http://{address}"))
		.expect("Devin endpoint")
		.connect()
		.await
		.expect("Devin channel");
	let devin: Arc<dyn Chat> = Arc::new(DevinChat::new(
		devin_channel,
		SpecializedCredentialAuth::new(broker.clone(), "devin"),
	));
	let egress = AuthInjectLayer::new(broker).layer(limits);
	let gitlab = Arc::new(GitLabDuoChat::new(
		WorkflowConfig::new(gitlab_url, "workflow-matrix", "session-matrix"),
		Arc::new(FixtureAuth),
	));
	let omp = Arc::new(OmpFederation::new(
		InferenceClient::connect(format!("http://{address}"))
			.await
			.expect("OMP channel"),
	));
	let local = Arc::new(Embedded::new(Arc::new(
		Inference::builder()
			.text(TextSelection::FoundationModels)
			.build()
			.await
			.unwrap_or_else(|error| {
				panic!(
					"macOS 26 on Apple silicon with Apple Intelligence enabled is required for the \
					 production embedded turn: {error}"
				)
			}),
	)));

	let registration = register_production_routes(
		&resolver,
		providers.iter(),
		egress,
		{
			let routed_leases = Arc::clone(&routed_leases);
			move |provider| {
				let (lease, metadata) = routed_leases
					.get(&provider.id)
					.cloned()
					.expect("HTTP provider has broker lease");
				dependencies(lease, metadata)
			}
		},
		|_| RouteStackConfig::default(),
		|provider| ProviderRoute {
			project: PROJECT.into(),
			region: if matches!(
				provider.transport,
				TransportId::AnthropicBedrock | TransportId::BedrockConverse
			) {
				"us-east-1".into()
			} else {
				REGION.into()
			},
			..ProviderRoute::default()
		},
		SpecializedChats {
			by_provider:         BTreeMap::from([
				(Str::new_static("cursor"), cursor),
				(Str::new_static("devin"), devin),
			]),
			cursor:              None,
			devin:               None,
			gitlab_duo_workflow: Some(gitlab),
			embedded:            Some(local),
			omp:                 Some(omp),
		},
	)
	.expect("all production transports register");
	assert_eq!(registration.registered, tested.len());

	let chat = RoutedChat::new(Arc::clone(&resolver));
	let executor = Arc::new(FixtureExecutor) as Arc<dyn Executor>;
	for (provider, transport) in specs {
		let selected_executor =
			matches!(transport, TransportId::Cursor | TransportId::GitLabDuoWorkflow)
				.then(|| Arc::clone(&executor));
		let events: Vec<_> = chat
			.turn(request(provider), selected_executor)
			.await
			.expect("production stream starts")
			.collect()
			.await;
		assert_eq!(
			events
				.iter()
				.filter(|event| matches!(event, TurnEvent::Outcome(_) | TurnEvent::Error(_)))
				.count(),
			1,
			"{transport:?} emits one terminal"
		);
		let outcome = events
			.iter()
			.find_map(|event| {
				if let TurnEvent::Outcome(outcome) = event {
					Some(outcome)
				} else {
					None
				}
			})
			.unwrap_or_else(|| panic!("{transport:?} successful outcome; events: {events:#?}"));
		assert!(outcome.usage.is_some(), "{transport:?} decodes usage");
	}

	for (provider, transport) in specs {
		if transport == TransportId::Embedded {
			continue;
		}
		let selected_executor =
			matches!(transport, TransportId::Cursor | TransportId::GitLabDuoWorkflow)
				.then(|| Arc::clone(&executor));
		let mut stream = chat
			.turn(request(provider), selected_executor)
			.await
			.expect("cancellation stream starts");
		loop {
			match stream.next().await.expect("cancellation stream commits") {
				TurnEvent::PartStart { .. } | TurnEvent::PartDelta { .. } => break,
				TurnEvent::Outcome(_) | TurnEvent::Error(_) => {
					panic!("{transport:?} cancellation probe terminated before it was live")
				},
				_ => {},
			}
		}
		drop(stream);
	}

	let mut cancelled = BTreeSet::new();
	while cancelled.len() < STREAMED_TRANSPORTS.len() - 1 {
		let transport = tokio::time::timeout(Duration::from_secs(2), cancelled_rx.recv_async())
			.await
			.expect("socket cancellation deadline")
			.expect("cancellation channel");
		cancelled.insert(transport);
	}
	let expected_socket_cancellations: BTreeSet<_> = STREAMED_TRANSPORTS
		.into_iter()
		.filter(|transport| *transport != TransportId::GitLabDuoWorkflow)
		.collect();
	assert_eq!(cancelled, expected_socket_cancellations);

	gitlab_server.await.expect("GitLab fixture");
	let captured = socket_state.captured.lock();
	for transport in STREAMED_TRANSPORTS {
		if transport == TransportId::GitLabDuoWorkflow {
			continue;
		}
		assert_eq!(
			captured
				.iter()
				.filter(|request| request.transport == transport)
				.count(),
			2,
			"{transport:?} terminal and cancellation requests crossed real sockets"
		);
	}
	assert!(
		captured
			.iter()
			.all(|request| !request.path.is_empty() && !request.body.is_empty())
	);
	assert!(
		captured
			.iter()
			.filter(|request| HTTP_TRANSPORTS.contains(&request.transport))
			.all(|request| request.headers.contains_key(header::AUTHORIZATION))
	);
	drop(captured);
	socket_server.abort();
}
