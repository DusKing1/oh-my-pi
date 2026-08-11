//! GitLab Duo Workflow authenticated WebSocket transport.
//!
//! Authentication remains behind [`WorkflowAuth`]: production implementations
//! redeem a canonical broker lease directly into request headers. This crate
//! never accepts a raw token or returns credential material.

pub mod discovery;

use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use bytes::Bytes;
use futures::{SinkExt as _, StreamExt as _, stream::BoxStream};
use http::{HeaderMap, HeaderValue, header};
use omp_core::Str;
use omp_llm_types::{
	Accuracy, CallIdMapper, Chat, ChatOutcome, ChatRequest, Diagnostic, Error, Executor, Fallback,
	Invoke, Item, ItemKind, Message, Part, Props, Retryability, Role, StopReason, StreamPartKind,
	ToolCall, ToolChoice, ToolResult, TurnError, TurnErrorKind, TurnEvent, Unsupported,
	UnsupportedAction, UnsupportedSink, Usage,
};
use serde_json::{Map, Value, json};
use tokio::net::TcpStream;
use tokio_tungstenite::{
	MaybeTlsStream, WebSocketStream, connect_async,
	tungstenite::{Message as WsMessage, client::IntoClientRequest as _},
};

const PROVIDER: &str = "gitlab-duo-agent";
const CLIENT_VERSION: &str = "1.0";
const LANGUAGE_SERVER_VERSION: &str = "8.104.0";
const DEFAULT_IDLE_TIMEOUT: Duration = Duration::from_secs(90);
const DEFAULT_INVOKE_TIMEOUT_MS: u64 = 120_000;

/// Injects request-ready authentication into a WebSocket handshake.
///
/// Implementations redeem a canonical credential lease directly into the
/// supplied handshake headers. They must not expose a raw secret through
/// another API.
#[async_trait]
pub trait WorkflowAuth: Send + Sync {
	/// Applies authentication to one connection attempt.
	async fn apply(&self, headers: &mut HeaderMap) -> Result<(), Error>;
}

/// One connected text-frame WebSocket used by the workflow state machine.
#[async_trait]
pub trait WorkflowSocket: Send {
	/// Sends one JSON text frame.
	async fn send_text(&mut self, text: String) -> Result<(), Error>;
	/// Receives the next JSON text frame, or `None` after a clean close.
	async fn receive_text(&mut self) -> Result<Option<String>, Error>;
	/// Sends a close frame and releases the connection.
	async fn close(&mut self);
}

/// Creates authenticated workflow `WebSockets`.
#[async_trait]
pub trait WorkflowSocketConnector: Send + Sync {
	/// Opens one socket using request-ready headers.
	async fn connect(&self, url: &str, headers: HeaderMap)
	-> Result<Box<dyn WorkflowSocket>, Error>;
}

/// Production connector backed by `tokio-tungstenite` and rustls.
#[derive(Clone, Copy, Debug, Default)]
pub struct TungsteniteConnector;

#[async_trait]
impl WorkflowSocketConnector for TungsteniteConnector {
	async fn connect(
		&self,
		url: &str,
		headers: HeaderMap,
	) -> Result<Box<dyn WorkflowSocket>, Error> {
		let mut request = url.into_client_request().map_err(transport_error)?;
		for (name, value) in headers {
			if let Some(name) = name {
				request.headers_mut().insert(name, value);
			}
		}
		let (socket, _) = connect_async(request).await.map_err(transport_error)?;
		Ok(Box::new(TungsteniteSocket(socket)))
	}
}

struct TungsteniteSocket(WebSocketStream<MaybeTlsStream<TcpStream>>);

#[async_trait]
impl WorkflowSocket for TungsteniteSocket {
	async fn send_text(&mut self, text: String) -> Result<(), Error> {
		self
			.0
			.send(WsMessage::Text(text.into()))
			.await
			.map_err(transport_error)
	}

	async fn receive_text(&mut self) -> Result<Option<String>, Error> {
		loop {
			match self.0.next().await {
				Some(Ok(WsMessage::Text(text))) => return Ok(Some(text.to_string())),
				Some(Ok(WsMessage::Binary(bytes))) => {
					return String::from_utf8(bytes.to_vec())
						.map(Some)
						.map_err(|error| Error::Transport(Str::from(error.to_string())));
				},
				Some(Ok(WsMessage::Ping(payload))) => {
					self
						.0
						.send(WsMessage::Pong(payload))
						.await
						.map_err(transport_error)?;
				},
				Some(Ok(WsMessage::Pong(_) | WsMessage::Frame(_))) => {},
				Some(Ok(WsMessage::Close(_))) | None => return Ok(None),
				Some(Err(error)) => return Err(transport_error(error)),
			}
		}
	}

	async fn close(&mut self) {
		let _ = self.0.close(None).await;
	}
}

/// Transport configuration for a GitLab Duo Workflow route.
#[derive(Clone, Debug)]
pub struct WorkflowConfig {
	/// Authenticated WebSocket endpoint.
	pub websocket_url:  Str,
	/// Server-created workflow identifier.
	pub workflow_id:    Str,
	/// Stable transport session identifier used by reconnect/resume.
	pub session_id:     Str,
	/// Maximum reconnects after an unclean transport loss.
	pub max_reconnects: u32,
	/// Maximum silence between workflow frames.
	pub idle_timeout:   Duration,
}

impl WorkflowConfig {
	/// Creates a route configuration for a server-created workflow.
	#[must_use]
	pub fn new(
		websocket_url: impl Into<Str>,
		workflow_id: impl Into<Str>,
		session_id: impl Into<Str>,
	) -> Self {
		Self {
			websocket_url:  websocket_url.into(),
			workflow_id:    workflow_id.into(),
			session_id:     session_id.into(),
			max_reconnects: 1,
			idle_timeout:   DEFAULT_IDLE_TIMEOUT,
		}
	}
}

/// GitLab Duo Workflow chat facet.
#[derive(Clone)]
pub struct GitLabDuoChat {
	config:    WorkflowConfig,
	auth:      Arc<dyn WorkflowAuth>,
	connector: Arc<dyn WorkflowSocketConnector>,
}

impl GitLabDuoChat {
	/// Creates a production GitLab Duo Workflow facet.
	#[must_use]
	pub fn new(config: WorkflowConfig, auth: Arc<dyn WorkflowAuth>) -> Self {
		Self { config, auth, connector: Arc::new(TungsteniteConnector) }
	}

	/// Replaces the connector, primarily for deterministic protocol fixtures.
	#[must_use]
	pub fn with_connector(mut self, connector: Arc<dyn WorkflowSocketConnector>) -> Self {
		self.connector = connector;
		self
	}
}

#[async_trait]
impl Chat for GitLabDuoChat {
	async fn turn(
		&self,
		request: ChatRequest,
		executor: Option<Arc<dyn Executor>>,
	) -> Result<BoxStream<'static, TurnEvent>, Error> {
		let start = build_start_request(&request, &self.config)?;
		let unsupported = validate_request(&request)?;
		let model = request.model;
		let config = self.config.clone();
		let auth = Arc::clone(&self.auth);
		let connector = Arc::clone(&self.connector);
		let stream = async_stream::stream! {
			 yield TurnEvent::Accepted { replay: false };
			 let mut state = DecodeState::new(model, unsupported);
			 let mut reconnects = 0_u32;
			 let mut socket = match open_socket(&*connector, &*auth, &config).await {
				  Ok(mut socket) => {
						if let Err(error) = socket.send_text(start.to_string()).await {
							 yield terminal_error(&mut state, error);
							 return;
						}
						socket
				  },
				  Err(error) => {
						yield terminal_error(&mut state, error);
						return;
				  },
			 };

			 loop {
				  let received = tokio::time::timeout(config.idle_timeout, socket.receive_text()).await;
				  let frame = match received {
						Ok(Ok(Some(frame))) => frame,
						Ok(Ok(None) | Err(_)) if reconnects < config.max_reconnects => {
							 reconnects += 1;
							 yield TurnEvent::Attempt {
								  number: reconnects + 1,
								  reason: Str::new_static("GitLab Duo Workflow socket reconnect/resume"),
							 };
							 match reconnect_socket(&*connector, &*auth, &config, state.last_event_id.as_deref()).await {
								  Ok(next) => { socket = next; continue; },
								  Err(error) => { yield terminal_error(&mut state, error); return; },
							 }
						},
						Err(_) if reconnects < config.max_reconnects => {
							 socket.close().await;
							 reconnects += 1;
							 yield TurnEvent::Attempt {
								  number: reconnects + 1,
								  reason: Str::new_static("GitLab Duo Workflow idle reconnect/resume"),
							 };
							 match reconnect_socket(&*connector, &*auth, &config, state.last_event_id.as_deref()).await {
								  Ok(next) => { socket = next; continue; },
								  Err(error) => { yield terminal_error(&mut state, error); return; },
							 }
						},
						Ok(Ok(None)) => {
							 yield terminal_error(&mut state, Error::Transport(Str::new_static("GitLab Duo Workflow socket closed before a terminal status")));
							 return;
						},
						Ok(Err(error)) => { yield terminal_error(&mut state, error); return; },
						Err(_) => {
							 yield terminal_error(&mut state, Error::Transport(Str::new_static("GitLab Duo Workflow socket idle timeout")));
							 return;
						},
				  };

				  let decoded = match decode_frame(&frame, &mut state) {
						Ok(decoded) => decoded,
						Err(error) => {
							 socket.close().await;
							 yield terminal_error(&mut state, error);
							 return;
						},
				  };
				  for event in decoded.events { yield event; }

				  if let Some(action) = decoded.action {
						let Some(executor) = executor.as_ref() else {
							 socket.close().await;
							 yield terminal_error(&mut state, Error::Unsupported(vec![
								  Unsupported::builder()
										.what(Str::new_static("gitlab-duo-workflow/executor"))
										.detail(Str::new_static("GitLab Duo Workflow requested an in-turn MCP action"))
										.action(UnsupportedAction::Dropped)
										.build(),
							 ]));
							 return;
						};
						let invocation = action.invocation.clone();
						yield TurnEvent::Invoke(invocation.clone());
						let (inputs, _streamed_inputs) = flume::unbounded();
						let completion = executor.invoke(invocation, inputs).await;
						let result = completion.tool_result.unwrap_or_else(|| ToolResult::builder()
							 .call_id(action.call.id)
							 .name(action.call.name.clone())
							 .parts(vec![Part::Text(Str::new_static("tool completed without a transcript result"))])
							 .is_error(true)
							 .build());
						state.output.push(Item::builder()
							 .seq(0)
							 .kind(ItemKind::ToolResult(result.clone()))
							 .props(Props::default())
							 .build());
						let response = action_response(&action.request_id, &result);
						if let Err(error) = socket.send_text(response.to_string()).await {
							 yield terminal_error(&mut state, error);
							 return;
						}
				  }

				  if decoded.terminal {
						socket.close().await;
						yield terminal_outcome(&mut state);
						return;
				  }
			 }
		};
		Ok(Box::pin(stream))
	}
}

async fn open_socket(
	connector: &dyn WorkflowSocketConnector,
	auth: &dyn WorkflowAuth,
	config: &WorkflowConfig,
) -> Result<Box<dyn WorkflowSocket>, Error> {
	let mut headers = HeaderMap::new();
	auth.apply(&mut headers).await?;
	headers.insert("x-gitlab-client-type", HeaderValue::from_static("node-websocket"));
	headers.insert(
		"x-gitlab-language-server-version",
		HeaderValue::from_static(LANGUAGE_SERVER_VERSION),
	);
	headers.insert(
		header::USER_AGENT,
		HeaderValue::from_static("unknown/unknown unknown/unknown gitlab-language-server/8.104.0"),
	);
	let mut origin = url::Url::parse(&config.websocket_url).map_err(transport_error)?;
	let origin_scheme = if origin.scheme() == "ws" {
		"http"
	} else {
		"https"
	};
	origin.set_scheme(origin_scheme).map_err(|()| {
		Error::Transport(Str::new_static("GitLab Duo Workflow endpoint has no HTTP origin"))
	})?;
	headers.insert(header::ORIGIN, header_value(origin.origin().ascii_serialization().as_str())?);
	connector.connect(&config.websocket_url, headers).await
}

async fn reconnect_socket(
	connector: &dyn WorkflowSocketConnector,
	auth: &dyn WorkflowAuth,
	config: &WorkflowConfig,
	last_event_id: Option<&str>,
) -> Result<Box<dyn WorkflowSocket>, Error> {
	let mut socket = open_socket(connector, auth, config).await?;
	socket
		.send_text(
			json!({
				 "resumeRequest": {
					  "workflowID": config.workflow_id,
					  "sessionID": config.session_id,
					  "lastEventID": last_event_id.unwrap_or_default(),
				 }
			})
			.to_string(),
		)
		.await?;
	Ok(socket)
}

fn build_start_request(request: &ChatRequest, config: &WorkflowConfig) -> Result<Value, Error> {
	let tools = request
		.tools
		.iter()
		.map(|tool| {
			let schema = serde_json::from_slice::<Value>(&tool.schema_json).map_err(|error| {
				Error::Unsupported(vec![
					Unsupported::builder()
						.what(Str::from(format!("tools.{}/schema", tool.name)))
						.detail(Str::from(format!("tool schema is not valid JSON: {error}")))
						.action(UnsupportedAction::Dropped)
						.build(),
				])
			})?;
			Ok(json!({
				 "name": tool.name,
				 "originalToolName": tool.name,
				 "serverName": "omp",
				 "description": tool.description,
				 "inputSchema": schema.to_string(),
				 "isApproved": true,
			}))
		})
		.collect::<Result<Vec<_>, Error>>()?;
	let goal = render_chatml(request)?;
	let system = workflow_system_prompt(request);
	Ok(json!({
		 "startRequest": {
			  "workflowID": config.workflow_id,
			  "sessionID": config.session_id,
			  "clientVersion": CLIENT_VERSION,
			  "workflowDefinition": "ambient",
			  "goal": goal,
			  "workflowMetadata": json!({
				  "environment": "ide",
				  "client_type": "node-websocket",
				  "selectedModelIdentifier": request.model,
			  }).to_string(),
			  "additional_context": [],
			  "clientCapabilities": [
				  "incremental_streaming",
				  "read_file_chunked",
				  "shell_command",
				  "command_timeout",
				  "tool_call_approval"
			  ],
			  "mcpTools": tools,
			  "preapproved_tools": request.tools.iter().map(|tool| tool.name.as_str()).collect::<Vec<_>>(),
			  "flowConfigSchemaVersion": "v1",
			  "flowConfig": {
				  "version": "v1",
				  "environment": "ambient",
				  "flow": {"entry_point": "omp_agent"},
				  "components": [{
					  "name": "omp_agent",
					  "type": "AgentComponent",
					  "prompt_id": "omp_inline_prompt",
					  "toolset": [],
					  "inputs": [{"from": "context:goal", "as": "goal"}],
					  "ui_log_events": [
						  "on_agent_reasoning",
						  "on_agent_final_answer",
						  "on_tool_execution_success",
						  "on_tool_execution_failed"
					  ]
				  }],
				  "routers": [{"from": "omp_agent", "to": "end"}],
				  "prompts": [{
					  "name": "omp_inline_prompt",
					  "prompt_id": "omp_inline_prompt",
					  "unit_primitives": ["duo_agent_platform"],
					  "prompt_template": {
						  "system": system,
						  "user": "{{goal}}",
						  "placeholder": "history"
					  }
				  }]
			  }
		 }
	}))
}

fn validate_request(request: &ChatRequest) -> Result<Vec<Unsupported>, Error> {
	let mut unsupported = UnsupportedSink::new();
	if request.sampling.is_some() {
		unsupported.drop_feature(
			"sampling",
			"GitLab Duo Workflow selects sampling controls in the server-side flow",
		);
	}
	if let Some(choice) = &request.tool_choice
		&& !matches!(&choice.value, ToolChoice::Auto)
	{
		record_feature(
			&mut unsupported,
			"tool_choice",
			"GitLab Duo Workflow exposes preapproved MCP tools but cannot force tool selection",
			choice.on_unsupported,
		)?;
	}
	if let Some(thinking) = &request.thinking {
		record_feature(
			&mut unsupported,
			"thinking",
			"GitLab Duo Workflow controls reasoning in the server-side flow",
			thinking.on_unsupported,
		)?;
	}
	if let Some(format) = &request.response_format {
		record_feature(
			&mut unsupported,
			"response_format",
			"GitLab Duo Workflow has no structured-output channel",
			format.on_unsupported,
		)?;
	}
	for (present, what, detail) in [
		(request.cache.is_some(), "cache", "workflow/session correlation is transport-owned"),
		(
			request.meta.is_some(),
			"meta",
			"request attribution is not represented by the workflow wire",
		),
		(request.service_tier.is_some(), "service_tier", "GitLab selects the workflow service tier"),
		(
			request.service_tier_by_family.is_some(),
			"service_tier_by_family",
			"GitLab selects the workflow service tier",
		),
		(request.task_budget.is_some(), "task_budget", "GitLab does not accept a task budget"),
		(
			request.responses_include.is_some(),
			"responses_include",
			"OpenAI Responses include fields do not apply to GitLab Workflow",
		),
	] {
		if present {
			unsupported.drop_feature(what, detail);
		}
	}
	if let Some(options) = &request.provider_options {
		for key in options.0.keys() {
			unsupported.drop_feature(
				key.clone(),
				"GitLab Duo Workflow does not recognize this provider option",
			);
		}
	}
	Ok(unsupported.into_vec())
}

fn record_feature(
	unsupported: &mut UnsupportedSink,
	what: &'static str,
	detail: &'static str,
	fallback: Fallback,
) -> Result<(), Error> {
	match fallback {
		Fallback::Ignore => {
			unsupported.drop_feature(what, detail);
			Ok(())
		},
		Fallback::Error | Fallback::Emulate => Err(Error::Unsupported(vec![
			Unsupported::builder()
				.what(Str::new_static(what))
				.detail(Str::new_static(detail))
				.action(UnsupportedAction::Dropped)
				.build(),
		])),
		_ => Err(Error::Unsupported(vec![
			Unsupported::builder()
				.what(Str::new_static(what))
				.detail(Str::new_static(detail))
				.action(UnsupportedAction::Dropped)
				.build(),
		])),
	}
}

/// Renders canonical history as the flat `ChatML` goal used when a workflow is
/// created or restarted.
pub fn render_chatml(request: &ChatRequest) -> Result<String, Error> {
	let replay = request
		.thread
		.items
		.iter()
		.filter(
			|item| !matches!(&item.kind, ItemKind::Message(message) if message.role == Role::System),
		)
		.collect::<Vec<_>>();
	if let [item] = replay.as_slice()
		&& let ItemKind::Message(message) = &item.kind
	{
		return Ok(parts_text(&message.parts));
	}
	let mut turns = Vec::with_capacity(replay.len());
	for item in replay {
		match &item.kind {
			ItemKind::Message(message) => {
				turns.push(format!(
					"<|im_start|>{}\n{}\n<|im_end|>",
					role_name(message.role),
					parts_text(&message.parts)
				));
			},
			ItemKind::ToolCall(call) => {
				let args = String::from_utf8_lossy(&call.args_json);
				turns.push(format!(
					"<|im_start|>assistant\n<ran {}>{}</ran>\n<|im_end|>",
					call.name, args
				));
			},
			ItemKind::ToolResult(result) => {
				let status = if result.is_error { " status=error" } else { "" };
				turns.push(format!(
					"<|im_start|>tool\n<ran:result{status}>\n{}\n<|im_end|>",
					parts_text(&result.parts)
				));
			},
			_ => {},
		}
	}
	Ok(turns.join("\n"))
}

fn workflow_system_prompt(request: &ChatRequest) -> String {
	request
		.thread
		.items
		.iter()
		.filter_map(|item| match &item.kind {
			ItemKind::Message(message) if message.role == Role::System => {
				Some(parts_text(&message.parts))
			},
			_ => None,
		})
		.collect::<Vec<_>>()
		.join("\n\n")
}

const fn role_name(role: Role) -> &'static str {
	match role {
		Role::System => "system",
		Role::User => "user",
		Role::Assistant => "assistant",
		_ => "user",
	}
}

fn parts_text(parts: &[Part]) -> String {
	parts
		.iter()
		.map(|part| match part {
			Part::Text(text) => text.as_str().to_owned(),
			Part::Thinking(thinking) => thinking.text.as_str().to_owned(),
			Part::Blob(blob) => format!("[{} blob: {} bytes]", blob.mime, blob.size),
			_ => String::new(),
		})
		.collect::<Vec<_>>()
		.join("\n")
}

struct DecodeState {
	model:           Str,
	mapper:          CallIdMapper,
	next_part:       u32,
	open_text:       Option<u32>,
	text:            String,
	checkpoint_text: String,
	output:          Vec<Item>,
	usage:           Option<Usage>,
	last_event_id:   Option<String>,
	unsupported:     Vec<Unsupported>,
}

impl DecodeState {
	fn new(model: Str, unsupported: Vec<Unsupported>) -> Self {
		Self {
			model,
			mapper: CallIdMapper::new(),
			next_part: 0,
			open_text: None,
			text: String::new(),
			checkpoint_text: String::new(),
			output: Vec::new(),
			usage: None,
			last_event_id: None,
			unsupported,
		}
	}
}

struct DecodedFrame {
	events:   Vec<TurnEvent>,
	action:   Option<DecodedAction>,
	terminal: bool,
}

struct DecodedAction {
	request_id: Str,
	call:       ToolCall,
	invocation: Invoke,
}

fn decode_frame(frame: &str, state: &mut DecodeState) -> Result<DecodedFrame, Error> {
	let value: Value = serde_json::from_str(frame).map_err(|error| {
		Error::Provider(Str::from(format!("malformed GitLab Duo Workflow frame: {error}")))
	})?;
	let event = value.as_object().ok_or_else(|| {
		Error::Provider(Str::new_static("malformed GitLab Duo Workflow frame: expected object"))
	})?;
	if let Some(id) = string_at(event, &["eventID", "event_id", "id"]) {
		state.last_event_id = Some(id.to_owned());
	}
	update_usage(event, state);
	let mut events = Vec::new();
	if let Some(checkpoint) = checkpoint_text(event)? {
		let delta = if checkpoint.starts_with(&state.checkpoint_text) {
			&checkpoint[state.checkpoint_text.len()..]
		} else {
			checkpoint.as_str()
		};
		if !delta.is_empty() {
			let index = if let Some(index) = state.open_text {
				index
			} else {
				let index = state.next_part;
				state.next_part += 1;
				state.open_text = Some(index);
				events.push(TurnEvent::PartStart {
					index,
					kind: StreamPartKind::Text,
					tool_call_id: Str::default(),
					tool_name: Str::default(),
				});
				index
			};
			state.text.push_str(delta);
			events
				.push(TurnEvent::PartDelta { index, chunk: Bytes::copy_from_slice(delta.as_bytes()) });
		}
		state.checkpoint_text = checkpoint;
	}

	let action = extract_action(event)?.map(|action| {
		if let Some(index) = state.open_text.take() {
			events.push(TurnEvent::PartEnd { index, signature: Bytes::new() });
			push_text_output(state);
		}
		let canonical_id = state.mapper.observe(action.request_id);
		let args =
			serde_json::to_vec(&action.arguments).expect("JSON value serialization is infallible");
		let tool_name = Str::from(action.tool_name.as_str());
		let call = ToolCall::builder()
			.id(canonical_id)
			.name(tool_name.clone())
			.args_json(Bytes::from(args.clone()))
			.thought_signature(Bytes::new())
			.build();
		let index = state.next_part;
		state.next_part += 1;
		events.push(TurnEvent::PartStart {
			index,
			kind: StreamPartKind::ToolCall,
			tool_call_id: Str::from(action.request_id),
			tool_name: tool_name.clone(),
		});
		events.push(TurnEvent::PartDelta { index, chunk: Bytes::from(args) });
		events.push(TurnEvent::PartEnd { index, signature: Bytes::new() });
		state.output.push(
			Item::builder()
				.seq(0)
				.kind(ItemKind::ToolCall(call.clone()))
				.props(Props::default())
				.build(),
		);
		let mut props = Props::default();
		props.insert_ns(PROVIDER, "request_id", Value::String(action.request_id.to_owned()));
		let invocation = Invoke::builder()
			.invocation_id(Str::from(action.request_id))
			.name(tool_name)
			.tool_call(call.clone())
			.vendor(Bytes::new())
			.timeout_ms(DEFAULT_INVOKE_TIMEOUT_MS)
			.props(props)
			.build();
		DecodedAction { request_id: Str::from(action.request_id), call, invocation }
	});

	let status = status(event);
	let terminal = matches!(status, Some("FINISHED" | "INPUT_REQUIRED"));
	if terminal && let Some(index) = state.open_text.take() {
		events.push(TurnEvent::PartEnd { index, signature: Bytes::new() });
		push_text_output(state);
	}
	if matches!(status, Some("FAILED" | "STOPPED")) {
		let detail = string_at(event, &["error", "message"]).unwrap_or(status.unwrap_or("FAILED"));
		return Err(Error::Provider(Str::from(format!("GitLab Duo Workflow {status:?}: {detail}"))));
	}
	Ok(DecodedFrame { events, action, terminal })
}

struct ActionRef<'a> {
	request_id: &'a str,
	tool_name:  String,
	arguments:  Value,
}

fn extract_action(event: &Map<String, Value>) -> Result<Option<ActionRef<'_>>, Error> {
	let wrapped = event
		.get("action")
		.or_else(|| event.get("workflowAction"))
		.or_else(|| event.get("toolCall"))
		.and_then(Value::as_object);
	if let Some(action) = wrapped {
		if action.get("newCheckpoint").is_some() {
			return Ok(None);
		}
		let Some(name) = string_at(action, &["name", "action", "type"])
			.or_else(|| string_at(event, &["actionName"]))
		else {
			return Ok(None);
		};
		let request_id = string_at(action, &["requestID", "requestId", "request_id", "id"])
			.or_else(|| string_at(event, &["requestID", "requestId", "request_id"]))
			.ok_or_else(|| malformed_action(name, "missing requestID"))?;
		let args = action
			.get("args")
			.or_else(|| action.get("arguments"))
			.and_then(Value::as_object)
			.unwrap_or(action);
		return map_action(name, request_id, args).map(Some);
	}
	for name in ["runMCPTool", "run_mcp_tool"] {
		if let Some(args) = event.get(name).and_then(Value::as_object) {
			let request_id = string_at(event, &["requestID", "requestId", "request_id"])
				.ok_or_else(|| malformed_action(name, "missing requestID"))?;
			return map_action(name, request_id, args).map(Some);
		}
	}
	Ok(None)
}

fn map_action<'a>(
	action_name: &str,
	request_id: &'a str,
	args: &Map<String, Value>,
) -> Result<ActionRef<'a>, Error> {
	if action_name != "runMCPTool" && action_name != "run_mcp_tool" {
		return Ok(ActionRef {
			request_id,
			tool_name: action_name.to_owned(),
			arguments: Value::Object(args.clone()),
		});
	}
	let raw_name = string_at(args, &["toolName", "tool_name", "name"])
		.ok_or_else(|| malformed_action(action_name, "missing MCP tool name"))?;
	let tool_name = raw_name
		.strip_prefix("mcp__omp__")
		.unwrap_or(raw_name)
		.to_owned();
	let arguments = match args.get("args").or_else(|| args.get("arguments")) {
		Some(Value::String(encoded)) => serde_json::from_str::<Value>(encoded)
			.ok()
			.filter(Value::is_object)
			.unwrap_or_else(|| Value::Object(Map::new())),
		Some(Value::Object(arguments)) => Value::Object(arguments.clone()),
		_ => Value::Object(Map::new()),
	};
	Ok(ActionRef { request_id, tool_name, arguments })
}

fn malformed_action(action: &str, detail: &str) -> Error {
	Error::Provider(Str::from(format!("malformed GitLab Duo Workflow action `{action}`: {detail}")))
}

fn checkpoint_text(event: &Map<String, Value>) -> Result<Option<String>, Error> {
	if let Some(text) = string_at(event, &["text", "delta", "content"]) {
		return Ok(Some(text.to_owned()));
	}
	let checkpoint = event
		.get("action")
		.and_then(Value::as_object)
		.and_then(|action| action.get("newCheckpoint"))
		.or_else(|| event.get("newCheckpoint"))
		.or_else(|| event.get("checkpoint"));
	let Some(checkpoint) = checkpoint else {
		return Ok(None);
	};
	if let Some(object) = checkpoint.as_object() {
		if let Some(text) = string_at(object, &["message", "text", "content"]) {
			return Ok(Some(text.to_owned()));
		}
		if let Some(text) = object
			.get("checkpoint")
			.and_then(Value::as_object)
			.and_then(|nested| string_at(nested, &["message", "text"]))
		{
			return Ok(Some(text.to_owned()));
		}
	}
	let serialized = checkpoint
		.as_object()
		.and_then(|object| object.get("checkpoint"))
		.unwrap_or(checkpoint);
	let checkpoint = if let Some(text) = serialized.as_str() {
		serde_json::from_str::<Value>(text).map_err(|error| {
			Error::Provider(Str::from(format!("malformed GitLab Duo Workflow checkpoint: {error}")))
		})?
	} else {
		serialized.clone()
	};
	let logs = checkpoint
		.pointer("/channel_values/ui_chat_log")
		.or_else(|| checkpoint.pointer("/values/ui_chat_log"))
		.and_then(Value::as_array);
	let Some(logs) = logs else { return Ok(None) };
	let text = logs
		.iter()
		.filter_map(|entry| {
			let object = entry.as_object()?;
			let role = string_at(object, &["role", "message_type", "type"])?;
			if role.contains("agent") || role == "assistant" || role == "reasoning" {
				string_at(object, &["content", "message", "text"])
			} else {
				None
			}
		})
		.collect::<Vec<_>>()
		.join("");
	Ok(Some(text))
}

fn update_usage(event: &Map<String, Value>, state: &mut DecodeState) {
	let action = event.get("action").and_then(Value::as_object);
	let checkpoint = action
		.and_then(|value| value.get("newCheckpoint"))
		.or_else(|| event.get("newCheckpoint"))
		.or_else(|| event.get("checkpoint"))
		.and_then(Value::as_object);
	let usage = [Some(event), action, checkpoint]
		.into_iter()
		.flatten()
		.find_map(|source| source.get("agent_context_usage").and_then(Value::as_object));
	let Some(usage) = usage else { return };
	let selected = usage
		.get("Chat Agent")
		.or_else(|| usage.get("context_builder"))
		.or_else(|| usage.values().next())
		.and_then(Value::as_object);
	let Some(selected) = selected else { return };
	let input = selected
		.get("total_tokens")
		.and_then(Value::as_u64)
		.unwrap_or(0);
	state.usage = Some(
		Usage::builder()
			.input_tokens(input)
			.output_tokens(0)
			.cache_read_tokens(0)
			.cache_write_tokens(0)
			.accuracy(Accuracy::Estimated)
			.detail(Props::default())
			.build(),
	);
}

fn status(event: &Map<String, Value>) -> Option<&str> {
	string_at(event, &["status"])
		.or_else(|| {
			event
				.get("workflowStatus")
				.and_then(Value::as_object)
				.and_then(|v| string_at(v, &["status"]))
		})
		.or_else(|| {
			event
				.get("newCheckpoint")
				.and_then(Value::as_object)
				.and_then(|v| string_at(v, &["status"]))
		})
		.or_else(|| {
			event
				.get("action")
				.and_then(Value::as_object)
				.and_then(|action| action.get("newCheckpoint"))
				.and_then(Value::as_object)
				.and_then(|checkpoint| string_at(checkpoint, &["status"]))
		})
}

fn string_at<'a>(object: &'a Map<String, Value>, keys: &[&str]) -> Option<&'a str> {
	keys.iter().find_map(|key| {
		object
			.get(*key)
			.and_then(Value::as_str)
			.filter(|value| !value.is_empty())
	})
}

fn action_response(request_id: &str, result: &ToolResult) -> Value {
	let text = parts_text(&result.parts);
	let plain = if result.is_error {
		json!({ "error": text })
	} else {
		json!({ "response": text })
	};
	json!({
		 "actionResponse": {
			  "requestID": request_id,
			  "plainTextResponse": plain
		 }
	})
}

fn push_text_output(state: &mut DecodeState) {
	if state.text.is_empty() {
		return;
	}
	state.output.push(
		Item::builder()
			.seq(0)
			.kind(ItemKind::Message(
				Message::builder()
					.role(Role::Assistant)
					.parts(vec![Part::Text(Str::from(std::mem::take(&mut state.text)))])
					.build(),
			))
			.props(Props::default())
			.build(),
	);
}

fn terminal_outcome(state: &mut DecodeState) -> TurnEvent {
	debug_assert!(state.open_text.is_none(), "terminal decoder must close every streamed part");
	push_text_output(state);
	TurnEvent::Outcome(
		ChatOutcome::builder()
			.output(std::mem::take(&mut state.output))
			.stop(StopReason::EndTurn)
			.maybe_usage(state.usage.take())
			.unsupported(std::mem::take(&mut state.unsupported))
			.provider(Str::new_static(PROVIDER))
			.model(std::mem::take(&mut state.model))
			.props(Props::default())
			.build(),
	)
}

fn terminal_error(state: &mut DecodeState, error: Error) -> TurnEvent {
	let (kind, detail, unsupported) = match error {
		Error::Unsupported(unsupported) => (
			TurnErrorKind::Unsupported,
			Str::new_static("GitLab Duo Workflow requires an in-turn executor"),
			unsupported,
		),
		Error::Provider(detail) | Error::Transport(detail) => {
			(TurnErrorKind::Upstream, detail, Vec::new())
		},
		_ => (TurnErrorKind::Upstream, Str::new_static("GitLab Duo Workflow failed"), Vec::new()),
	};
	let code = if kind == TurnErrorKind::Unsupported {
		"unsupported"
	} else if detail.contains("malformed") {
		"malformed_frame"
	} else if detail.contains("credential") || detail.contains("authorization") {
		"auth"
	} else if detail.contains("timeout") {
		"timeout"
	} else {
		"transport"
	};
	let kind = if code == "auth" {
		TurnErrorKind::Auth
	} else {
		kind
	};
	let diagnostic = Diagnostic::builder()
		.provider(Str::new_static(PROVIDER))
		.model(state.model.clone())
		.attempt(1)
		.code(Str::new_static(code))
		.detail(detail.clone())
		.retryability(Retryability::Never)
		.build();
	TurnEvent::Error(
		TurnError::builder()
			.kind(kind)
			.detail(detail)
			.unsupported(unsupported)
			.retry_after_ms(0)
			.diagnostics(vec![diagnostic])
			.build(),
	)
}

fn header_value(value: &str) -> Result<HeaderValue, Error> {
	HeaderValue::from_str(value).map_err(|_| {
		Error::Transport(Str::new_static(
			"GitLab Duo Workflow correlation id is not a valid header value",
		))
	})
}

fn transport_error(error: impl std::fmt::Display) -> Error {
	Error::Transport(Str::from(error.to_string()))
}
