//! GitLab Duo direct-access delegation and typed Workflow WebSocket protocol.

use std::collections::BTreeMap;

use bytes::Bytes;
use omp_core::Str;
use secrecy::SecretString;
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;

use super::{
	anthropic::AnthropicCodec, openai_chat::OpenAiChatCodec, openai_responses::OpenAiResponsesCodec,
};
use crate::{
	body::BodySource,
	call::{ChatRequest, ContentPart, Message, OperationCall, Role, Setting, ToolResultContent},
	codec::{
		Codec, DecodeContext, Decoder, DecoderState, EncodeContext, EncodedRequest,
		ProviderControlEvent, ProviderControlInput, ProviderStateEvent, RawCompletion, RawEvent,
		RequestHeader, RequestMethod, SizeBounds, ToolInputKind, UnvalidatedToolCall,
	},
	error::{Error, ErrorDetail, ErrorKind, ErrorPhase, RetryAction},
	event::{BlockKind, ChatEvent, FinishReason, UsageUpdate, WorkflowResponse},
	id::ToolCallId,
	receipt::{ExecutionReceipt, ReasonId, Usage, UsageSource},
	transport::{Frame, FramingProtocol, WebSocketMessage},
};

const CLIENT_VERSION: &str = "1.0";
const DEFAULT_INVOKE_TIMEOUT_MS: u64 = 120_000;

/// Stable GitLab GraphQL query for namespace-scoped Duo model availability.
pub const GITLAB_DUO_MODELS_QUERY: &str =
	"query lsp_aiChatAvailableModels($rootNamespaceId: GroupID!) { \
	 aiChatAvailableModels(rootNamespaceId: $rootNamespaceId) { defaultModel { name ref } \
	 selectableModels { name ref } pinnedModel { name ref } } }";

/// Typed GraphQL request for GitLab Duo model discovery.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GitLabDiscoveryRequest {
	/// Static query text pinned to the verified discovery contract.
	pub query:     &'static str,
	/// Namespace variables for the query.
	pub variables: GitLabDiscoveryVariables,
}

/// Typed variables for Duo discovery.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct GitLabDiscoveryVariables {
	/// Root namespace represented as a GitLab GraphQL gid.
	#[serde(rename = "rootNamespaceId")]
	pub root_namespace_id: Str,
}

/// Non-secret raw discovery record passed to catalog normalization.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GitLabDiscoveredModel {
	/// Provider-local model reference.
	pub provider_model_id: Str,
	/// Display name from the latest precedence source.
	pub name:              Str,
	/// Whether the pinned-model field supplied the final record.
	pub pinned:            bool,
}

/// Encodes namespace-scoped GitLab Duo discovery without credential headers.
pub fn encode_discovery_request(root_namespace_id: impl Into<Str>) -> Result<Bytes, Error> {
	serde_json::to_vec(&GitLabDiscoveryRequest {
		query:     GITLAB_DUO_MODELS_QUERY,
		variables: GitLabDiscoveryVariables { root_namespace_id: root_namespace_id.into() },
	})
	.map(Bytes::from)
	.map_err(|_| protocol_error(ErrorPhase::Discovery, "gitlab.discovery.request"))
}

/// Decodes selectable/default/pinned records with documented replacement
/// precedence.
pub fn decode_discovery_response(payload: &[u8]) -> Result<Vec<GitLabDiscoveredModel>, Error> {
	let response: GitLabDiscoveryResponse = serde_json::from_slice(payload)
		.map_err(|_| protocol_error(ErrorPhase::Discovery, "gitlab.discovery.response"))?;
	let available = response
		.data
		.ai_chat_available_models
		.ok_or_else(|| protocol_error(ErrorPhase::Discovery, "gitlab.discovery.unavailable"))?;
	let mut models = BTreeMap::new();
	for (model, pinned) in available
		.selectable_models
		.into_iter()
		.map(|model| (model, false))
		.chain(
			available
				.default_model
				.into_iter()
				.map(|model| (model, false)),
		)
		.chain(
			available
				.pinned_model
				.into_iter()
				.map(|model| (model, true)),
		) {
		if model.reference.trim().is_empty() {
			continue;
		}
		let provider_model_id = Str::from(model.reference.trim());
		let name = if model.name.trim().is_empty() {
			provider_model_id.clone()
		} else {
			Str::from(model.name.trim())
		};
		models.insert(provider_model_id.clone(), GitLabDiscoveredModel {
			provider_model_id,
			name,
			pinned,
		});
	}
	Ok(models.into_values().collect())
}

#[derive(Deserialize)]
struct GitLabDiscoveryResponse {
	data: GitLabDiscoveryData,
}

#[derive(Deserialize)]
struct GitLabDiscoveryData {
	#[serde(rename = "aiChatAvailableModels")]
	ai_chat_available_models: Option<GitLabAvailableModels>,
}

#[derive(Deserialize)]
struct GitLabAvailableModels {
	#[serde(rename = "defaultModel", default)]
	default_model:     Option<GitLabModelRef>,
	#[serde(rename = "selectableModels", default)]
	selectable_models: Vec<GitLabModelRef>,
	#[serde(rename = "pinnedModel", default)]
	pinned_model:      Option<GitLabModelRef>,
}

#[derive(Deserialize)]
struct GitLabModelRef {
	name:      String,
	#[serde(rename = "ref")]
	reference: String,
}

/// Protocol family returned by GitLab direct-access route resolution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GitLabDelegationTarget {
	/// OpenAI Chat Completions and SSE.
	OpenAiChat,
	/// OpenAI Responses and SSE.
	OpenAiResponses,
	/// Anthropic Messages and typed SSE.
	AnthropicMessages,
}

/// Typed direct-access route data produced by catalog/auth planning.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GitLabDirectRoute {
	/// GitLab direct-access exchange endpoint.
	pub exchange_endpoint: Str,
	/// Exact shared wire family selected by route data.
	pub delegation:        GitLabDelegationTarget,
}

impl GitLabDirectRoute {
	/// Encodes the credential-free direct-access exchange request.
	pub fn encode_exchange(
		&self,
		request: &GitLabTokenExchangeRequest,
	) -> Result<EncodedRequest, Error> {
		let body = serde_json::to_vec(request)
			.map(Bytes::from)
			.map_err(|_| protocol_error(ErrorPhase::Encoding, "gitlab.direct_access.request"))?;
		Ok(EncodedRequest {
			operation:   omp_llm_catalog::OperationKind::Auth,
			method:      RequestMethod::Post,
			uri:         self.exchange_endpoint.clone(),
			headers:     Box::new([RequestHeader {
				name:  Str::new_static("content-type"),
				value: Str::new_static("application/json"),
			}]),
			body:        BodySource::Bytes(body),
			framing:     FramingProtocol::Raw,
			bounds:      SizeBounds {
				request_body: 1024 * 1024,
				frame:        1024 * 1024,
				response:     1024 * 1024,
			},
			sealed_body: None,
		})
	}
}

/// Request body for GitLab Duo direct-access token exchange.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GitLabTokenExchangeRequest {
	/// Workflow definition requested from GitLab.
	pub workflow_definition: Str,
	/// Root namespace represented as a GitLab GraphQL gid.
	pub root_namespace_id:   Str,
	/// Optional project scope.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub project_id:          Option<Str>,
}

/// Secret direct-access grant retained only at the auth/credential boundary.
pub struct GitLabDirectAccessGrant {
	token:              SecretString,
	/// Exact proxy base URL returned by GitLab.
	pub proxy_endpoint: Str,
	/// Grant lifetime in seconds, when supplied.
	pub expires_in:     Option<u64>,
}

impl GitLabDirectAccessGrant {
	/// Decodes a typed GitLab grant without exposing token bytes through `Debug`
	/// or serde.
	pub fn decode(payload: &[u8]) -> Result<Self, Error> {
		let wire: DirectAccessGrantWire = serde_json::from_slice(payload).map_err(|_| {
			protocol_error(ErrorPhase::Authentication, "gitlab.direct_access.response")
		})?;
		if wire.token.is_empty() || wire.proxy_endpoint.is_empty() {
			return Err(protocol_error(
				ErrorPhase::Authentication,
				"gitlab.direct_access.missing_grant_fields",
			));
		}
		Ok(Self {
			token:          SecretString::from(wire.token),
			proxy_endpoint: wire.proxy_endpoint,
			expires_in:     wire.expires_in,
		})
	}

	/// Transfers the secret into credential-lease storage without cloning it.
	pub(crate) fn into_token(self) -> SecretString {
		self.token
	}
}

#[derive(Deserialize)]
struct DirectAccessGrantWire {
	#[serde(alias = "access_token")]
	token:          String,
	#[serde(rename = "base_url", alias = "proxy_endpoint", alias = "ai_gateway_url")]
	proxy_endpoint: Str,
	#[serde(default)]
	expires_in:     Option<u64>,
}

/// Typed shared-codec delegation selected from [`GitLabDirectRoute`].
#[derive(Clone, Debug)]
pub enum GitLabDelegatingCodec {
	/// Delegate request/stream behavior to the shared OpenAI Chat codec.
	OpenAiChat(OpenAiChatCodec),
	/// Delegate request/stream behavior to the shared OpenAI Responses codec.
	OpenAiResponses(OpenAiResponsesCodec),
	/// Delegate request/stream behavior to the shared Anthropic codec.
	AnthropicMessages(AnthropicCodec),
}

impl GitLabDelegatingCodec {
	/// Constructs the selected shared codec from typed route data, never
	/// provider names.
	#[must_use]
	pub fn from_route(route: &GitLabDirectRoute) -> Self {
		match route.delegation {
			GitLabDelegationTarget::OpenAiChat => Self::OpenAiChat(OpenAiChatCodec::default()),
			GitLabDelegationTarget::OpenAiResponses => {
				Self::OpenAiResponses(OpenAiResponsesCodec::default())
			},
			GitLabDelegationTarget::AnthropicMessages => {
				Self::AnthropicMessages(AnthropicCodec::direct())
			},
		}
	}
}

impl Codec for GitLabDelegatingCodec {
	fn encode(
		&self,
		context: &EncodeContext<'_>,
		operation: &OperationCall,
	) -> Result<EncodedRequest, Error> {
		match self {
			Self::OpenAiChat(codec) => codec.encode(context, operation),
			Self::OpenAiResponses(codec) => codec.encode(context, operation),
			Self::AnthropicMessages(codec) => codec.encode(context, operation),
		}
	}

	fn decoder(&self, context: &DecodeContext<'_>) -> Result<DecoderState, Error> {
		match self {
			Self::OpenAiChat(codec) => codec.decoder(context),
			Self::OpenAiResponses(codec) => codec.decoder(context),
			Self::AnthropicMessages(codec) => codec.decoder(context),
		}
	}
}

/// Stable workflow/session/reconnect state carried outside socket ownership.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkflowSession {
	/// Workflow identity allocated by GitLab or canonical session planning.
	pub workflow_id:   Str,
	/// Stable workflow session identity.
	pub session_id:    Str,
	/// Last fully decoded provider event.
	pub last_event_id: Option<Str>,
	/// Number of resume attempts already emitted.
	pub reconnects:    u32,
}

impl WorkflowSession {
	/// Creates initial workflow state.
	pub fn new(workflow_id: impl Into<Str>, session_id: impl Into<Str>) -> Self {
		Self {
			workflow_id:   workflow_id.into(),
			session_id:    session_id.into(),
			last_event_id: None,
			reconnects:    0,
		}
	}

	/// Records an acknowledged checkpoint without changing session identity.
	pub fn checkpoint(&mut self, event_id: impl Into<Str>) {
		self.last_event_id = Some(event_id.into());
	}

	/// Encodes the exact typed resume frame and advances reconnect evidence.
	pub fn resume_frame(&mut self) -> Result<Bytes, Error> {
		let last_event_id = self
			.last_event_id
			.clone()
			.ok_or_else(|| protocol_error(ErrorPhase::Session, "gitlab.resume.missing_checkpoint"))?;
		self.reconnects = self.reconnects.saturating_add(1);
		serde_json::to_vec(&ResumeFrame {
			resume_request: ResumeRequest {
				workflow_id: self.workflow_id.clone(),
				session_id: self.session_id.clone(),
				last_event_id,
			},
		})
		.map(Bytes::from)
		.map_err(|_| protocol_error(ErrorPhase::Session, "gitlab.resume.serialization"))
	}
}

/// Typed result returned to a pending GitLab workflow action.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkflowActionResult {
	/// Provider request identity from the action frame.
	pub request_id: Str,
	/// Plain textual result.
	pub text:       Str,
	/// Whether the tool failed.
	pub is_error:   bool,
}

impl WorkflowActionResult {
	/// Encodes the exact action-response frame expected by the workflow socket.
	pub fn encode(&self) -> Result<Bytes, Error> {
		let plain = if self.is_error {
			PlainTextResponse::Error { error: self.text.clone() }
		} else {
			PlainTextResponse::Success { response: self.text.clone() }
		};
		serde_json::to_vec(&ActionResponseFrame {
			action_response: ActionResponse {
				request_id:          self.request_id.clone(),
				plain_text_response: plain,
			},
		})
		.map(Bytes::from)
		.map_err(|_| protocol_error(ErrorPhase::Encoding, "gitlab.action_response.serialization"))
	}
}

/// Stateless GitLab Duo Workflow WebSocket codec.
#[derive(Clone, Debug, Default)]
pub struct GitLabWorkflowCodec;

impl GitLabWorkflowCodec {
	/// Constructs the workflow codec.
	#[must_use]
	pub const fn new() -> Self {
		Self
	}
}

impl Codec for GitLabWorkflowCodec {
	fn encode(
		&self,
		context: &EncodeContext<'_>,
		operation: &OperationCall,
	) -> Result<EncodedRequest, Error> {
		match operation {
			OperationCall::Chat(request) => {
				let _ = wire_target(context)?;
				let workflow_id = context.session.map_or_else(
					|| context.request_id.as_str(),
					|session| session.conversation.as_str(),
				);
				let session_id = context
					.session
					.map_or_else(|| context.request_id.as_str(), |session| session.turn.as_str());
				let start = build_start_request(
					request,
					context,
					WorkflowSession::new(workflow_id, session_id),
				)?;
				let body = serde_json::to_vec(&start)
					.map(Bytes::from)
					.map_err(|_| protocol_error(ErrorPhase::Encoding, "gitlab.start.serialization"))?;
				Ok(EncodedRequest {
					operation:   omp_llm_catalog::OperationKind::Chat,
					method:      RequestMethod::Post,
					uri:         context.route.endpoint.base_url.clone(),
					headers:     Box::new([
						RequestHeader {
							name:  Str::new_static("user-agent"),
							value: Str::from(format!("omp/{}", env!("CARGO_PKG_VERSION"))),
						},
						RequestHeader {
							name:  Str::new_static("x-gitlab-client-type"),
							value: Str::new_static("node-websocket"),
						},
						RequestHeader {
							name:  Str::new_static("x-gitlab-language-server-version"),
							value: Str::new_static("8.104.0"),
						},
						RequestHeader {
							name:  Str::new_static("origin"),
							value: websocket_origin(&context.route.endpoint.base_url)?,
						},
					]),
					body:        BodySource::Bytes(body),
					framing:     FramingProtocol::WebSocket,
					bounds:      SizeBounds {
						request_body: 16 * 1024 * 1024,
						frame:        16 * 1024 * 1024,
						response:     256 * 1024 * 1024,
					},
					sealed_body: None,
				})
			},
			OperationCall::DiscoverModels(request) => {
				if request.cursor.is_some() {
					return Err(invalid_request("gitlab.discovery.pagination_unsupported"));
				}
				let namespace = gitlab_root_namespace(context)?;
				Ok(EncodedRequest {
					operation:   omp_llm_catalog::OperationKind::DiscoverModels,
					method:      RequestMethod::Post,
					uri:         endpoint(&context.route.endpoint.base_url, "/api/graphql"),
					headers:     Box::new([
						RequestHeader {
							name:  Str::new_static("accept"),
							value: Str::new_static("application/json"),
						},
						RequestHeader {
							name:  Str::new_static("content-type"),
							value: Str::new_static("application/json"),
						},
					]),
					body:        BodySource::Bytes(encode_discovery_request(namespace)?),
					framing:     FramingProtocol::Raw,
					bounds:      SizeBounds {
						request_body: 1024 * 1024,
						frame:        16 * 1024 * 1024,
						response:     16 * 1024 * 1024,
					},
					sealed_body: None,
				})
			},
			_ => Err(invalid_request("gitlab.workflow.operation_unsupported")),
		}
	}

	fn decoder(&self, context: &DecodeContext<'_>) -> Result<DecoderState, Error> {
		match context.operation {
			omp_llm_catalog::OperationKind::Chat => Ok(Box::new(WorkflowDecoder::default())),
			omp_llm_catalog::OperationKind::DiscoverModels => Ok(Box::new(GitLabDiscoveryDecoder {
				provider:  context.provider.clone(),
				route:     context.route.clone(),
				completed: false,
			})),
			_ => Err(invalid_request("gitlab.workflow.operation_unsupported")),
		}
	}
}

#[derive(Serialize)]
struct StartFrame {
	#[serde(rename = "startRequest")]
	start_request: StartRequest,
}

#[derive(Serialize)]
struct StartRequest {
	#[serde(rename = "workflowID")]
	workflow_id: Str,
	#[serde(rename = "sessionID")]
	session_id: Str,
	#[serde(rename = "clientVersion")]
	client_version: &'static str,
	#[serde(rename = "workflowDefinition")]
	workflow_definition: &'static str,
	goal: String,
	#[serde(rename = "workflowMetadata")]
	workflow_metadata: String,
	additional_context: Vec<AdditionalContext>,
	#[serde(rename = "clientCapabilities")]
	client_capabilities: [&'static str; 5],
	#[serde(rename = "mcpTools")]
	mcp_tools: Vec<McpTool>,
	preapproved_tools: Vec<Str>,
	#[serde(rename = "flowConfigSchemaVersion")]
	flow_config_schema_version: &'static str,
	#[serde(rename = "flowConfig")]
	flow_config: FlowConfig,
}

#[derive(Serialize)]
struct AdditionalContext;

#[derive(Serialize)]
struct WorkflowMetadata {
	environment:               &'static str,
	client_type:               &'static str,
	#[serde(rename = "selectedModelIdentifier")]
	selected_model_identifier: Str,
}

#[derive(Serialize)]
struct McpTool {
	name:               Str,
	#[serde(rename = "originalToolName")]
	original_tool_name: Str,
	#[serde(rename = "serverName")]
	server_name:        &'static str,
	description:        Str,
	#[serde(rename = "inputSchema")]
	input_schema:       String,
	#[serde(rename = "isApproved")]
	is_approved:        bool,
}

#[derive(Serialize)]
struct FlowConfig {
	version:     &'static str,
	environment: &'static str,
	flow:        Flow,
	components:  [FlowComponent; 1],
	routers:     [FlowRouter; 1],
	prompts:     [FlowPrompt; 1],
}

#[derive(Serialize)]
struct Flow {
	entry_point: &'static str,
}

#[derive(Serialize)]
struct FlowComponent {
	name:          &'static str,
	#[serde(rename = "type")]
	kind:          &'static str,
	prompt_id:     &'static str,
	toolset:       [&'static str; 0],
	inputs:        [FlowInput; 1],
	ui_log_events: [&'static str; 4],
}

#[derive(Serialize)]
struct FlowInput {
	from:    &'static str,
	#[serde(rename = "as")]
	as_name: &'static str,
}

#[derive(Serialize)]
struct FlowRouter {
	from: &'static str,
	to:   &'static str,
}

#[derive(Serialize)]
struct FlowPrompt {
	name:            &'static str,
	prompt_id:       &'static str,
	unit_primitives: [&'static str; 1],
	prompt_template: PromptTemplate,
}

#[derive(Serialize)]
struct PromptTemplate {
	system:      String,
	user:        &'static str,
	placeholder: &'static str,
}

fn build_start_request(
	request: &ChatRequest,
	context: &EncodeContext<'_>,
	session: WorkflowSession,
) -> Result<StartFrame, Error> {
	if !request.hosted_tools.is_empty() {
		return Err(invalid_request("gitlab.workflow.hosted_tools.unsupported"));
	}
	if !matches!(request.tool_choice, Setting::Unset)
		|| !matches!(request.output, Setting::Unset)
		|| !matches!(request.reasoning, Setting::Unset)
		|| !matches!(request.verbosity, Setting::Unset)
		|| !matches!(request.cache_retention, Setting::Unset)
		|| !matches!(request.service_tier, Setting::Unset)
		|| request.sampling.temperature.is_some()
		|| request.sampling.top_p.is_some()
		|| request.sampling.top_k.is_some()
		|| request.sampling.seed.is_some()
		|| !request.sampling.stop.is_empty()
		|| request.sampling.presence_penalty.is_some()
		|| request.sampling.frequency_penalty.is_some()
		|| request.max_output_tokens.is_some()
		|| request.top_logprobs.is_some()
		|| !request.safety.is_empty()
	{
		return Err(invalid_request("gitlab.workflow.explicit_controls.unsupported"));
	}
	let mut tools = Vec::with_capacity(request.tools.len());
	for tool in request.tools.iter() {
		let Some((parameters, _)) = tool.input.json_schema() else {
			return Err(capability_error("gitlab.workflow.tool_grammar.unsupported"));
		};
		tools.push(McpTool {
			name:               tool.name.clone(),
			original_tool_name: tool.name.clone(),
			server_name:        "omp",
			description:        tool.description.clone().unwrap_or_default(),
			input_schema:       serde_json::to_string(parameters.as_value())
				.map_err(|_| invalid_request("gitlab.workflow.tool_schema.serialization"))?,
			is_approved:        true,
		});
	}
	let system = workflow_system_prompt(request, context)?;
	let goal = render_chatml(request, context)?;
	let workflow_metadata = serde_json::to_string(&WorkflowMetadata {
		environment:               "ide",
		client_type:               "node-websocket",
		selected_model_identifier: Str::from(wire_target(context)?.wire_model.as_str()),
	})
	.map_err(|_| protocol_error(ErrorPhase::Encoding, "gitlab.metadata.serialization"))?;
	Ok(StartFrame {
		start_request: StartRequest {
			workflow_id: session.workflow_id,
			session_id: session.session_id,
			client_version: CLIENT_VERSION,
			workflow_definition: "ambient",
			goal,
			workflow_metadata,
			additional_context: Vec::new(),
			client_capabilities: [
				"incremental_streaming",
				"read_file_chunked",
				"shell_command",
				"command_timeout",
				"tool_call_approval",
			],
			preapproved_tools: request.tools.iter().map(|tool| tool.name.clone()).collect(),
			mcp_tools: tools,
			flow_config_schema_version: "v1",
			flow_config: FlowConfig {
				version:     "v1",
				environment: "ambient",
				flow:        Flow { entry_point: "omp_agent" },
				components:  [FlowComponent {
					name:          "omp_agent",
					kind:          "AgentComponent",
					prompt_id:     "omp_inline_prompt",
					toolset:       [],
					inputs:        [FlowInput { from: "context:goal", as_name: "goal" }],
					ui_log_events: [
						"on_agent_reasoning",
						"on_agent_final_answer",
						"on_tool_execution_success",
						"on_tool_execution_failed",
					],
				}],
				routers:     [FlowRouter { from: "omp_agent", to: "end" }],
				prompts:     [FlowPrompt {
					name:            "omp_inline_prompt",
					prompt_id:       "omp_inline_prompt",
					unit_primitives: ["duo_agent_platform"],
					prompt_template: PromptTemplate { system, user: "{{goal}}", placeholder: "history" },
				}],
			},
		},
	})
}

/// Renders canonical history into GitLab Workflow's flat ChatML goal.
pub fn render_chatml(request: &ChatRequest, context: &EncodeContext<'_>) -> Result<String, Error> {
	let replay = request
		.messages
		.iter()
		.filter(|message| !matches!(message.role, Role::System | Role::Developer))
		.collect::<Vec<_>>();
	if let [message] = replay.as_slice() {
		return message_text(message, context);
	}
	let mut output = String::new();
	for message in replay {
		if !output.is_empty() {
			output.push('\n');
		}
		output.push_str("<|im_start|>");
		output.push_str(role_name(message.role));
		output.push('\n');
		output.push_str(&message_text(message, context)?);
		output.push_str("\n<|im_end|>");
	}
	Ok(output)
}

fn workflow_system_prompt(
	request: &ChatRequest,
	context: &EncodeContext<'_>,
) -> Result<String, Error> {
	let mut output = String::new();
	for message in request
		.messages
		.iter()
		.filter(|message| matches!(message.role, Role::System | Role::Developer))
	{
		if !output.is_empty() {
			output.push_str("\n\n");
		}
		output.push_str(&message_text(message, context)?);
	}
	Ok(output)
}

fn message_text(message: &Message, context: &EncodeContext<'_>) -> Result<String, Error> {
	let mut output = String::new();
	for part in message.content.iter() {
		if !output.is_empty() {
			output.push('\n');
		}
		match part {
			ContentPart::Text { text, proof } => {
				validate_proof(proof.as_ref(), context)?;
				if proof.is_some() {
					return Err(invalid_request("gitlab.workflow.text_proof.unrepresentable"));
				}
				output.push_str(text);
			},
			ContentPart::Reasoning { text, proof } => {
				validate_proof(proof.as_ref(), context)?;
				if proof.is_some() {
					return Err(invalid_request("gitlab.workflow.reasoning_proof.unrepresentable"));
				}
				output.push_str(text);
			},
			ContentPart::ToolCall { call: _, name, arguments, proof } => {
				if proof.is_some() {
					validate_proof(proof.as_ref(), context)?;
					return Err(invalid_request("gitlab.workflow.tool_proof.unrepresentable"));
				}
				output.push_str("<ran ");
				output.push_str(name);
				output.push('>');
				output.push_str(
					&serde_json::to_string(arguments.as_value())
						.map_err(|_| invalid_request("gitlab.workflow.tool_arguments.serialization"))?,
				);
				output.push_str("</ran>");
			},
			ContentPart::ToolResult { name, content, is_error, .. } => {
				if name.is_some() {
					return Err(invalid_request("gitlab.workflow.tool_result.name_unrepresentable"));
				}
				output.push_str(if *is_error {
					"<ran:result status=error>"
				} else {
					"<ran:result>"
				});
				append_tool_result(&mut output, content)?;
				output.push_str("</ran:result>");
			},
			ContentPart::Image(_) | ContentPart::Audio(_) | ContentPart::Document(_) => {
				return Err(invalid_request("gitlab.workflow.media.unsupported"));
			},
			ContentPart::CachePoint(_) => {
				return Err(invalid_request("gitlab.workflow.explicit_cache_point.unsupported"));
			},
		}
	}
	Ok(output)
}

fn append_tool_result(output: &mut String, content: &[ToolResultContent]) -> Result<(), Error> {
	for (index, part) in content.iter().enumerate() {
		if index > 0 {
			output.push('\n');
		}
		match part {
			ToolResultContent::Text(text) => output.push_str(text),
			ToolResultContent::Json(json) => output.push_str(
				&serde_json::to_string(json.as_value())
					.map_err(|_| invalid_request("gitlab.workflow.tool_result.serialization"))?,
			),
			ToolResultContent::Image(_) | ToolResultContent::Document(_) => {
				return Err(invalid_request("gitlab.workflow.tool_result.media.unsupported"));
			},
		}
	}
	Ok(())
}

const fn role_name(role: Role) -> &'static str {
	match role {
		Role::Assistant => "assistant",
		Role::Tool => "tool",
		Role::System => "system",
		Role::Developer => "developer",
		Role::User => "user",
	}
}

fn validate_proof(
	proof: Option<&crate::call::ProviderProof>,
	context: &EncodeContext<'_>,
) -> Result<(), Error> {
	if let Some(proof) = proof
		&& (proof.provider != context.route.provider || proof.codec != wire_target(context)?.codec)
	{
		return Err(invalid_request("gitlab.workflow.provider_proof.scope_mismatch"));
	}
	Ok(())
}

#[derive(Serialize)]
struct ResumeFrame {
	#[serde(rename = "resumeRequest")]
	resume_request: ResumeRequest,
}

#[derive(Serialize)]
struct ResumeRequest {
	#[serde(rename = "workflowID")]
	workflow_id:   Str,
	#[serde(rename = "sessionID")]
	session_id:    Str,
	#[serde(rename = "lastEventID")]
	last_event_id: Str,
}

#[derive(Serialize)]
struct ActionResponseFrame {
	#[serde(rename = "actionResponse")]
	action_response: ActionResponse,
}

#[derive(Serialize)]
struct ActionResponse {
	#[serde(rename = "requestID")]
	request_id:          Str,
	#[serde(rename = "plainTextResponse")]
	plain_text_response: PlainTextResponse,
}

#[derive(Serialize)]
#[serde(untagged)]
enum PlainTextResponse {
	Success { response: Str },
	Error { error: Str },
}

struct GitLabDiscoveryDecoder {
	provider:  omp_llm_catalog::ProviderId,
	route:     omp_llm_catalog::RouteId,
	completed: bool,
}

impl Decoder for GitLabDiscoveryDecoder {
	fn push(&mut self, frame: Frame, emit: &mut dyn FnMut(RawEvent)) -> Result<(), Error> {
		if self.completed {
			return Err(protocol_error(ErrorPhase::Discovery, "gitlab.discovery.duplicate_response"));
		}
		let Frame::Raw(payload) = frame else {
			return Err(protocol_error(ErrorPhase::Discovery, "gitlab.discovery.frame.expected_raw"));
		};
		let rows = decode_discovery_response(&payload)?
			.into_iter()
			.map(|model| omp_llm_catalog::DiscoveredModel {
				provider:              self.provider.clone(),
				route:                 self.route.clone(),
				wire_model:            omp_llm_catalog::WireModelId::from(model.provider_model_id),
				aliases:               Box::new([]),
				display_name:          Some(model.name),
				declared_family:       None,
				declared_operations:   omp_llm_catalog::OperationBits::for_kind(
					omp_llm_catalog::OperationKind::Chat,
				),
				declared_capabilities: None,
				declared_limits:       None,
				extended_context_mode: None,
				availability:          Some(omp_llm_catalog::ModelAvailability::Available),
				source:                Str::new_static("gitlab.ai-chat-available-models"),
				observed_at_ms:        None,
				updated_at_ms:         None,
				deprecated:            None,
			})
			.collect();
		self.completed = true;
		emit(RawEvent::DiscoveredModels { rows, next_cursor: None });
		Ok(())
	}

	fn finish(&mut self, _emit: &mut dyn FnMut(RawEvent)) -> Result<(), Error> {
		if self.completed {
			Ok(())
		} else {
			Err(protocol_error(ErrorPhase::Discovery, "gitlab.discovery.response_missing"))
		}
	}
}

#[derive(Default)]
struct WorkflowDecoder {
	next_index:      u32,
	text_index:      Option<u32>,
	checkpoint_text: String,
	usage:           Usage,
	completed:       bool,
}

impl Decoder for WorkflowDecoder {
	fn push(&mut self, frame: Frame, emit: &mut dyn FnMut(RawEvent)) -> Result<(), Error> {
		let Frame::WebSocket(message) = frame else {
			return Err(protocol_error(ErrorPhase::Streaming, "gitlab.frame.expected_websocket"));
		};
		let payload = match message {
			WebSocketMessage::Text(payload) | WebSocketMessage::Binary(payload) => payload,
			WebSocketMessage::Ping(_) | WebSocketMessage::Pong(_) => return Ok(()),
			WebSocketMessage::Close { .. } => {
				return if self.completed {
					Ok(())
				} else {
					Err(protocol_error(ErrorPhase::Streaming, "gitlab.socket.closed_before_terminal"))
				};
			},
		};
		let inbound: WorkflowInbound = serde_json::from_slice(&payload)
			.map_err(|_| protocol_error(ErrorPhase::Streaming, "gitlab.malformed_frame"))?;
		self.push_inbound(inbound, emit)
	}

	fn supports_control(&self) -> bool {
		true
	}

	fn encode_control(&mut self, input: ProviderControlInput) -> Result<Option<Bytes>, Error> {
		let WorkflowResponse::WorkflowActionResponse(response) = input else {
			return Err(protocol_error(
				ErrorPhase::Streaming,
				"gitlab.workflow.control_kind_unsupported",
			));
		};
		let text = std::str::from_utf8(&response.response)
			.map_err(|_| protocol_error(ErrorPhase::Streaming, "gitlab.action_response.utf8"))?;
		WorkflowActionResult {
			request_id: response.invocation,
			text:       Str::from(text),
			is_error:   response.is_error,
		}
		.encode()
		.map(Some)
	}

	fn finish(&mut self, _emit: &mut dyn FnMut(RawEvent)) -> Result<(), Error> {
		if self.completed {
			Ok(())
		} else {
			Err(protocol_error(ErrorPhase::Streaming, "gitlab.stream.incomplete"))
		}
	}
}

impl WorkflowDecoder {
	fn push_inbound(
		&mut self,
		inbound: WorkflowInbound,
		emit: &mut dyn FnMut(RawEvent),
	) -> Result<(), Error> {
		if self.completed {
			return Err(protocol_error(ErrorPhase::Streaming, "gitlab.frame.after_terminal"));
		}
		let checkpoint = inbound.checkpoint();
		let text = inbound.text.clone().or_else(|| {
			checkpoint
				.and_then(WorkflowCheckpoint::text)
				.map(ToOwned::to_owned)
		});
		let status = inbound
			.status
			.clone()
			.or_else(|| {
				inbound
					.workflow_status
					.as_ref()
					.and_then(|value| value.status.clone())
			})
			.or_else(|| checkpoint.and_then(|value| value.status.clone()));
		let message = inbound.message.clone().or_else(|| {
			checkpoint
				.and_then(WorkflowCheckpoint::message)
				.map(Str::from)
		});
		let usage = inbound
			.agent_context_usage
			.as_ref()
			.or_else(|| checkpoint.and_then(|value| value.agent_context_usage.as_ref()))
			.and_then(select_usage)
			.map(|usage| usage.total_tokens);
		if let Some(event_id) = inbound.event_id.clone() {
			emit(RawEvent::ProviderState(ProviderStateEvent::Checkpoint {
				id:   Some(event_id),
				data: Bytes::new(),
			}));
		}
		if let Some(input_tokens) = usage {
			self.usage = Usage { input_tokens, source: UsageSource::Estimated, ..Usage::default() };
		}
		if let Some(checkpoint) = text.as_deref() {
			let delta = if checkpoint.starts_with(&self.checkpoint_text) {
				&checkpoint[self.checkpoint_text.len()..]
			} else {
				checkpoint
			};
			if !delta.is_empty() {
				let index = *self.text_index.get_or_insert_with(|| {
					let index = self.next_index;
					self.next_index = self.next_index.saturating_add(1);
					emit(RawEvent::Chat(ChatEvent::BlockStarted { index, kind: BlockKind::Text }));
					index
				});
				emit(RawEvent::Chat(ChatEvent::TextDelta { index, text: Str::from(delta) }));
			}
			self.checkpoint_text.clear();
			self.checkpoint_text.push_str(checkpoint);
		}

		if let Some(action) = inbound.action {
			if let Some(action) = action.into_mcp()? {
				self.emit_action(action, emit)?;
			}
		} else if let Some(action) = inbound.run_mcp_tool.or(inbound.run_mcp_tool_snake) {
			let request_id = inbound
				.request_id
				.ok_or_else(|| malformed_action("runMCPTool", "missing requestID"))?;
			self.emit_action(action.into_call(request_id, "runMCPTool")?, emit)?;
		}

		match status.as_deref() {
			Some("FINISHED") => {
				emit(RawEvent::Chat(ChatEvent::Usage(UsageUpdate {
					usage:        self.usage,
					final_update: true,
				})));
				emit(RawEvent::Completion(RawCompletion {
					reason: FinishReason::Stop,
					blocks: self.next_index,
					usage:  self.usage,
				}));
				self.completed = true;
			},
			Some("FAILED" | "STOPPED") => {
				let detail = message.unwrap_or_else(|| status.clone().unwrap_or_default());
				let mut failure = protocol_error_dynamic(
					ErrorPhase::Streaming,
					format!("GitLab Duo Workflow {status:?}: {detail}"),
				);
				failure.code = status.clone();
				failure.committed = self.next_index != 0;
				emit(RawEvent::Failure(failure));
				self.completed = true;
			},
			_ => {},
		}
		Ok(())
	}

	fn emit_action(
		&mut self,
		action: DecodedAction,
		emit: &mut dyn FnMut(RawEvent),
	) -> Result<(), Error> {
		self.text_index = None;
		let arguments = action.arguments_bytes()?;
		emit(RawEvent::Control(ProviderControlEvent::WorkflowAction {
			request_id: action.request_id,
			name: action.tool_name,
			arguments,
			timeout_ms: Some(DEFAULT_INVOKE_TIMEOUT_MS),
		}));
		Ok(())
	}
}

#[derive(Deserialize)]
struct WorkflowInbound {
	#[serde(rename = "eventID", alias = "event_id", alias = "id")]
	#[serde(default)]
	event_id:            Option<Str>,
	#[serde(alias = "delta", alias = "content")]
	#[serde(default)]
	text:                Option<String>,
	#[serde(default)]
	status:              Option<Str>,
	#[serde(rename = "workflowStatus", default)]
	workflow_status:     Option<WorkflowStatus>,
	#[serde(default)]
	checkpoint:          Option<WorkflowCheckpoint>,
	#[serde(rename = "newCheckpoint", default)]
	new_checkpoint:      Option<WorkflowCheckpoint>,
	#[serde(default)]
	action:              Option<ActionEnvelope>,
	#[serde(rename = "runMCPTool", default)]
	run_mcp_tool:        Option<McpArguments>,
	#[serde(rename = "run_mcp_tool", default)]
	run_mcp_tool_snake:  Option<McpArguments>,
	#[serde(rename = "requestID", alias = "requestId", alias = "request_id", default)]
	request_id:          Option<Str>,
	#[serde(default)]
	agent_context_usage: Option<BTreeMap<String, ContextUsage>>,
	#[serde(default)]
	message:             Option<Str>,
}

impl WorkflowInbound {
	fn checkpoint(&self) -> Option<&WorkflowCheckpoint> {
		self
			.new_checkpoint
			.as_ref()
			.or(self.checkpoint.as_ref())
			.or_else(|| {
				self
					.action
					.as_ref()
					.and_then(|action| action.new_checkpoint.as_ref())
			})
	}
}

#[derive(Deserialize)]
struct WorkflowStatus {
	#[serde(default)]
	status: Option<Str>,
}

#[derive(Deserialize)]
struct WorkflowCheckpoint {
	#[serde(default)]
	message:             Option<String>,
	#[serde(default)]
	text:                Option<String>,
	#[serde(default)]
	content:             Option<String>,
	#[serde(default)]
	status:              Option<Str>,
	#[serde(default)]
	agent_context_usage: Option<BTreeMap<String, ContextUsage>>,
	#[serde(default)]
	checkpoint:          Option<Box<WorkflowCheckpoint>>,
}

impl WorkflowCheckpoint {
	fn text(&self) -> Option<&str> {
		self
			.text
			.as_deref()
			.or(self.content.as_deref())
			.or(self.message.as_deref())
			.or_else(|| self.checkpoint.as_deref().and_then(Self::text))
	}

	fn message(&self) -> Option<&str> {
		self
			.message
			.as_deref()
			.or_else(|| self.checkpoint.as_deref().and_then(Self::message))
	}
}

#[derive(Deserialize)]
struct ActionEnvelope {
	#[serde(alias = "action", alias = "type", default)]
	name:           Option<Str>,
	#[serde(rename = "requestID", alias = "requestId", alias = "request_id", alias = "id")]
	#[serde(default)]
	request_id:     Option<Str>,
	#[serde(default)]
	args:           Option<McpArguments>,
	#[serde(default)]
	arguments:      Option<McpArguments>,
	#[serde(rename = "newCheckpoint", default)]
	new_checkpoint: Option<WorkflowCheckpoint>,
}

impl ActionEnvelope {
	fn into_mcp(self) -> Result<Option<DecodedAction>, Error> {
		let Some(name) = self.name else {
			return Ok(None);
		};
		let request_id = self
			.request_id
			.ok_or_else(|| malformed_action(name.as_str(), "missing requestID"))?;
		let args = self.args.or(self.arguments).unwrap_or_default();
		if matches!(name.as_str(), "runMCPTool" | "run_mcp_tool") {
			args.into_call(request_id, name.as_str()).map(Some)
		} else {
			Err(malformed_action(name.as_str(), "unsupported workflow action"))
		}
	}
}

#[derive(Default, Deserialize)]
struct McpArguments {
	#[serde(rename = "toolName", alias = "tool_name", alias = "name", default)]
	tool_name: Option<Str>,
	#[serde(default)]
	arguments: Option<RawArguments>,
	#[serde(default)]
	args:      Option<RawArguments>,
}

impl McpArguments {
	fn into_call(self, request_id: Str, action_name: &str) -> Result<DecodedAction, Error> {
		let tool_name = self
			.tool_name
			.ok_or_else(|| malformed_action(action_name, "missing MCP tool name"))?;
		let tool_name = tool_name
			.as_str()
			.strip_prefix("mcp__omp__")
			.unwrap_or(tool_name.as_str());
		Ok(DecodedAction {
			request_id,
			tool_name: Str::from(tool_name),
			arguments: self.arguments.or(self.args),
		})
	}
}

#[derive(Deserialize)]
#[serde(transparent)]
struct RawArguments(Box<RawValue>);

struct DecodedAction {
	request_id: Str,
	tool_name:  Str,
	arguments:  Option<RawArguments>,
}

impl DecodedAction {
	fn arguments_bytes(&self) -> Result<Bytes, Error> {
		match &self.arguments {
			Some(arguments) => {
				let raw = arguments.0.get();
				if raw.starts_with('"') {
					let encoded: String = serde_json::from_str(raw).map_err(|_| {
						malformed_action(self.tool_name.as_str(), "arguments string is not JSON")
					})?;
					let decoded: Box<RawValue> = serde_json::from_str(&encoded).map_err(|_| {
						malformed_action(self.tool_name.as_str(), "arguments string is not JSON")
					})?;
					Ok(Bytes::copy_from_slice(decoded.get().as_bytes()))
				} else {
					Ok(Bytes::copy_from_slice(raw.as_bytes()))
				}
			},
			None => Ok(Bytes::from_static(b"{}")),
		}
	}
}

#[derive(Deserialize)]
struct ContextUsage {
	total_tokens: u64,
	#[serde(default)]
	_max_tokens:  u64,
}
fn websocket_origin(endpoint: &str) -> Result<Str, Error> {
	let (scheme, rest) = if let Some(rest) = endpoint.strip_prefix("wss://") {
		("https://", rest)
	} else if let Some(rest) = endpoint.strip_prefix("ws://") {
		("http://", rest)
	} else {
		return Err(invalid_request("gitlab.workflow.websocket_scheme"));
	};
	let authority = rest.split('/').next().unwrap_or_default();
	if authority.is_empty() || authority.contains('@') {
		return Err(invalid_request("gitlab.workflow.websocket_authority"));
	}
	Ok(Str::from(format!("{scheme}{authority}")))
}

fn gitlab_root_namespace(context: &EncodeContext<'_>) -> Result<Str, Error> {
	let namespace = context
		.account
		.and_then(|account| account.organization.as_ref())
		.ok_or_else(|| invalid_request("gitlab.discovery.organization_required"))?
		.as_str();
	if namespace.bytes().all(|byte| byte.is_ascii_digit()) {
		Ok(Str::from(format!("gid://gitlab/Group/{namespace}")))
	} else {
		Ok(Str::from(namespace))
	}
}

fn endpoint(base: &str, path: &str) -> Str {
	Str::from(format!("{}{}", base.trim_end_matches('/'), path))
}

fn wire_target<'a>(
	context: &'a EncodeContext<'_>,
) -> Result<&'a omp_llm_catalog::WireTarget, Error> {
	context
		.target
		.ok_or_else(|| invalid_request("gitlab.model_target.required"))
}

fn select_usage(usage: &BTreeMap<String, ContextUsage>) -> Option<&ContextUsage> {
	usage
		.get("Chat Agent")
		.or_else(|| usage.get("context_builder"))
		.or_else(|| usage.values().next())
}

fn malformed_action(action: &str, detail: &str) -> Error {
	let mut error = protocol_error(ErrorPhase::Streaming, "gitlab.malformed_action");
	error.code = Some(Str::from(format!("{action}: {detail}")));
	error
}

fn invalid_request(reason: &'static str) -> Error {
	let mut error = protocol_error(ErrorPhase::Encoding, reason);
	error.kind = ErrorKind::InvalidRequest;
	error
}

fn capability_error(reason: &'static str) -> Error {
	let mut error = protocol_error(ErrorPhase::Encoding, reason);
	error.kind = ErrorKind::CapabilityMismatch;
	error
}

fn protocol_error_dynamic(phase: ErrorPhase, reason: String) -> Error {
	let mut error =
		Error::new(ErrorKind::Protocol, phase, RetryAction::Never, ExecutionReceipt::default());
	error.detail = Some(ErrorDetail::Protocol { reason: ReasonId(Str::from(reason)) });
	error
}

fn protocol_error(phase: ErrorPhase, reason: &'static str) -> Error {
	let mut error =
		Error::new(ErrorKind::Protocol, phase, RetryAction::Never, ExecutionReceipt::default());
	error.detail = Some(ErrorDetail::Protocol { reason: ReasonId(Str::new_static(reason)) });
	error
}

/// Default action invocation timeout carried by workflow orchestration.
pub const fn default_invoke_timeout_ms() -> u64 {
	DEFAULT_INVOKE_TIMEOUT_MS
}

#[cfg(test)]
mod tests {
	use std::sync::Arc;

	use omp_llm_catalog::Catalog;
	use serde::Deserialize;
	use serde_json::value::RawValue;

	use super::*;
	use crate::{
		call::{
			NegotiationPolicy, Sampling, ToolDefinition, ToolGrammar, ToolGrammarSyntax,
			ToolInputConstraint,
		},
		codec::EncodeAttempt,
		id::RequestId,
	};

	#[derive(Deserialize)]
	struct CassetteLine {
		direction: String,
		#[serde(default)]
		payload:   Option<Box<RawValue>>,
		#[serde(default)]
		raw:       Option<String>,
	}

	#[test]
	fn replays_workflow_reconnect_fixture_with_checkpoint_usage_and_tool_call() {
		let fixture = include_str!(concat!(
			env!("CARGO_MANIFEST_DIR"),
			"/../../fixtures/llm-oracle/agent-protocols/gitlab/stream.reconnect.jsonl"
		));
		let mut decoder = WorkflowDecoder::default();
		let mut events = Vec::new();
		for line in fixture.lines() {
			let row: CassetteLine = serde_json::from_str(line).expect("typed cassette row");
			if row.direction != "server_to_client" {
				continue;
			}
			let payload = row.payload.expect("server payload");
			decoder
				.push(
					Frame::WebSocket(WebSocketMessage::Text(Bytes::copy_from_slice(
						payload.get().as_bytes(),
					))),
					&mut |event| events.push(event),
				)
				.expect("workflow fixture frame");
		}
		decoder.finish(&mut |_| {}).expect("terminal fixture");

		let text = events
			.iter()
			.filter_map(|event| match event {
				RawEvent::Chat(ChatEvent::TextDelta { text, .. }) => Some(text.as_str()),
				_ => None,
			})
			.collect::<String>();
		assert_eq!(text, "working done");
		assert!(events.iter().any(|event| matches!(
			event,
			RawEvent::ProviderState(ProviderStateEvent::Checkpoint {
				id: Some(id),
				..
			}) if id.as_str() == "2"
		)));
		assert!(events.iter().any(|event| matches!(
			event,
			RawEvent::Control(ProviderControlEvent::WorkflowAction {
				name,
				arguments,
				timeout_ms: Some(DEFAULT_INVOKE_TIMEOUT_MS),
				..
			}) if name.as_str() == "edit" && arguments.as_ref() == br#"{"path":"a.rs"}"#
		)));
		assert!(events.iter().any(|event| matches!(
			event,
			RawEvent::Completion(RawCompletion {
				usage: Usage { input_tokens: 321, source: UsageSource::Estimated, .. },
				..
			})
		)));
	}

	#[test]
	fn action_and_resume_frames_match_protocol_fixture() {
		let response = WorkflowActionResult {
			request_id: Str::new_static("req-9"),
			text:       Str::new_static("tool-ok"),
			is_error:   false,
		}
		.encode()
		.expect("action response");
		assert_eq!(
			response.as_ref(),
			br#"{"actionResponse":{"requestID":"req-9","plainTextResponse":{"response":"tool-ok"}}}"#
		);

		let mut session = WorkflowSession::new("workflow-7", "session-a");
		session.checkpoint("2");
		let resume = session.resume_frame().expect("resume frame");
		assert_eq!(
			resume.as_ref(),
			br#"{"resumeRequest":{"workflowID":"workflow-7","sessionID":"session-a","lastEventID":"2"}}"#
		);
		assert_eq!(session.reconnects, 1);
	}

	#[derive(Deserialize)]
	struct DiscoveryFixture {
		request:  DiscoveryRequestFixture,
		response: Box<RawValue>,
	}

	#[derive(Deserialize)]
	struct DiscoveryRequestFixture {
		body: Box<RawValue>,
	}

	#[derive(Debug, Deserialize, Eq, PartialEq)]
	struct ExpectedDiscoveryRequest {
		query:     String,
		variables: GitLabDiscoveryVariables,
	}

	#[test]
	fn replays_graphql_discovery_precedence_without_model_name_classification() {
		let fixture: DiscoveryFixture = serde_json::from_str(include_str!(concat!(
			env!("CARGO_MANIFEST_DIR"),
			"/../../fixtures/llm-oracle/agent-protocols/gitlab/discovery.graphql.json"
		)))
		.expect("discovery fixture");
		let encoded =
			encode_discovery_request("gid://gitlab/Group/42").expect("typed discovery request");
		let actual: ExpectedDiscoveryRequest =
			serde_json::from_slice(&encoded).expect("encoded typed discovery request");
		let expected: ExpectedDiscoveryRequest =
			serde_json::from_str(fixture.request.body.get()).expect("fixture typed discovery request");
		assert_eq!(actual, expected);
		let models = decode_discovery_response(fixture.response.get().as_bytes())
			.expect("typed discovery response");
		assert_eq!(
			models
				.iter()
				.map(|model| model.provider_model_id.as_str())
				.collect::<Vec<_>>(),
			["claude_opus_4_8", "claude_sonnet_4_6_vertex"]
		);
		assert_eq!(models[1].name.as_str(), "Sonnet pinned");
		assert!(models[1].pinned);
	}

	#[test]
	fn discovery_decoder_projects_typed_rows() {
		let fixture: DiscoveryFixture = serde_json::from_str(include_str!(concat!(
			env!("CARGO_MANIFEST_DIR"),
			"/../../fixtures/llm-oracle/agent-protocols/gitlab/discovery.graphql.json"
		)))
		.expect("discovery fixture");
		let mut decoder = GitLabDiscoveryDecoder {
			provider:  omp_llm_catalog::ProviderId::from("gitlab-duo-agent"),
			route:     omp_llm_catalog::RouteId::from("gitlab-duo-agent/primary"),
			completed: false,
		};
		let mut events = Vec::new();
		decoder
			.push(
				Frame::Raw(Bytes::copy_from_slice(fixture.response.get().as_bytes())),
				&mut |event| events.push(event),
			)
			.expect("typed discovery response");
		decoder.finish(&mut |_| {}).expect("complete discovery");
		assert!(matches!(
			events.as_slice(),
			[RawEvent::DiscoveredModels { rows, next_cursor: None }]
				if rows.len() == 2
					&& rows[0].wire_model.as_str() == "claude_opus_4_8"
					&& rows[1].wire_model.as_str() == "claude_sonnet_4_6_vertex"
		));
	}

	#[test]
	fn websocket_origin_preserves_authority_and_maps_scheme() {
		assert_eq!(
			websocket_origin("ws://127.0.0.1:43123/path")
				.expect("ws origin")
				.as_str(),
			"http://127.0.0.1:43123"
		);
		assert_eq!(
			websocket_origin("wss://gitlab.example/ws")
				.expect("wss origin")
				.as_str(),
			"https://gitlab.example"
		);
	}

	#[derive(Deserialize)]
	struct ActionFixture {
		cases: Vec<ActionCase>,
	}

	#[derive(Deserialize)]
	struct ActionCase {
		input:    Box<RawValue>,
		#[serde(default)]
		expected: Option<ExpectedAction>,
	}

	#[derive(Deserialize)]
	struct ExpectedAction {
		request_id: Str,
		tool_name:  Str,
	}

	#[test]
	fn replays_every_typed_action_fixture_shape() {
		let fixture: ActionFixture = serde_json::from_str(include_str!(concat!(
			env!("CARGO_MANIFEST_DIR"),
			"/../../fixtures/llm-oracle/agent-protocols/gitlab/action_cases.json"
		)))
		.expect("action fixture");
		for case in fixture.cases {
			let mut decoder = WorkflowDecoder::default();
			let mut events = Vec::new();
			let result = decoder.push(
				Frame::WebSocket(WebSocketMessage::Text(Bytes::copy_from_slice(
					case.input.get().as_bytes(),
				))),
				&mut |event| events.push(event),
			);
			if let Some(expected) = case.expected {
				result.expect("supported action shape");
				assert!(events.iter().any(|event| matches!(
					event,
					RawEvent::Control(ProviderControlEvent::WorkflowAction {
						request_id,
						name,
						..
					}) if request_id.as_str() == expected.request_id.as_str()
						&& name.as_str() == expected.tool_name.as_str()
				)));
			} else if case.input.get().contains("\"action\"") {
				assert!(
					result.is_err()
						|| events
							.iter()
							.any(|event| matches!(event, RawEvent::Failure(_)))
				);
			}
		}
	}

	#[test]
	fn malformed_and_cancelled_fixtures_never_fabricate_completion() {
		let malformed_fixture = include_str!(concat!(
			env!("CARGO_MANIFEST_DIR"),
			"/../../fixtures/llm-oracle/agent-protocols/gitlab/stream.malformed.jsonl"
		));
		let mut malformed = WorkflowDecoder::default();
		let mut malformed_error = None;
		for line in malformed_fixture.lines() {
			let row: CassetteLine = serde_json::from_str(line).expect("malformed cassette row");
			if row.direction == "server_to_client" {
				let raw = row.raw.expect("raw malformed payload");
				malformed_error = malformed
					.push(Frame::WebSocket(WebSocketMessage::Text(Bytes::from(raw))), &mut |_| {})
					.err();
			}
		}
		assert_eq!(malformed_error.expect("malformed error").kind, ErrorKind::Protocol);

		let cancel_fixture = include_str!(concat!(
			env!("CARGO_MANIFEST_DIR"),
			"/../../fixtures/llm-oracle/agent-protocols/gitlab/stream.cancel.jsonl"
		));
		let mut cancelled = WorkflowDecoder::default();
		let mut events = Vec::new();
		for line in cancel_fixture.lines() {
			let row: CassetteLine = serde_json::from_str(line).expect("cancel cassette row");
			if row.direction != "server_to_client" {
				continue;
			}
			let payload = row.payload.expect("server checkpoint payload");
			cancelled
				.push(
					Frame::WebSocket(WebSocketMessage::Text(Bytes::copy_from_slice(
						payload.get().as_bytes(),
					))),
					&mut |event| events.push(event),
				)
				.expect("checkpoint");
		}
		assert!(cancelled.finish(&mut |_| {}).is_err());
		assert!(
			!events
				.iter()
				.any(|event| matches!(event, RawEvent::Completion(_)))
		);
	}

	#[test]
	fn direct_access_delegation_is_typed_route_data() {
		let routes = [
			(GitLabDelegationTarget::OpenAiChat, "openai-chat"),
			(GitLabDelegationTarget::OpenAiResponses, "openai-responses"),
			(GitLabDelegationTarget::AnthropicMessages, "anthropic"),
		];
		for (delegation, expected) in routes {
			let route = GitLabDirectRoute {
				exchange_endpoint: Str::new_static("https://gitlab.example/direct_access"),
				delegation,
			};
			let selected = match GitLabDelegatingCodec::from_route(&route) {
				GitLabDelegatingCodec::OpenAiChat(_) => "openai-chat",
				GitLabDelegatingCodec::OpenAiResponses(_) => "openai-responses",
				GitLabDelegatingCodec::AnthropicMessages(_) => "anthropic",
			};
			assert_eq!(selected, expected);
		}
	}

	#[derive(Deserialize)]
	struct StartFixture {
		wire: StartWire,
	}

	#[derive(Deserialize)]
	struct StartWire {
		#[serde(rename = "startRequest")]
		start_request: StartContract,
	}

	#[derive(Deserialize)]
	struct StartContract {
		#[serde(rename = "workflowID")]
		workflow_id:         Str,
		#[serde(rename = "sessionID")]
		session_id:          Str,
		#[serde(rename = "clientVersion")]
		client_version:      Str,
		#[serde(rename = "workflowDefinition")]
		workflow_definition: Str,
		#[serde(rename = "clientCapabilities")]
		client_capabilities: Vec<Str>,
		#[serde(rename = "mcpTools")]
		mcp_tools:           Vec<McpToolContract>,
	}

	#[derive(Deserialize)]
	struct McpToolContract {
		name:         Str,
		#[serde(rename = "inputSchema")]
		input_schema: String,
	}

	#[derive(Deserialize)]
	struct AuthFixture {
		websocket_url: Str,
		headers:       AuthHeaderContract,
	}

	#[derive(Deserialize)]
	struct AuthHeaderContract {
		#[serde(rename = "x-gitlab-client-type")]
		client_type:             Str,
		#[serde(rename = "x-gitlab-language-server-version")]
		language_server_version: Str,
		origin:                  Str,
	}

	#[derive(Deserialize)]
	struct TerminalContract {
		terminal_count: u32,
	}

	#[derive(Deserialize)]
	struct CancelContract {
		expected_transport: CancelTransportContract,
	}

	#[derive(Deserialize)]
	struct CancelTransportContract {
		send_close_or_drop_socket:          bool,
		fallback_transport:                 bool,
		terminal_event_after_consumer_drop: bool,
	}

	#[test]
	fn typed_contract_fixtures_pin_start_auth_and_terminal_behavior() {
		let start: StartFixture = serde_json::from_str(include_str!(concat!(
			env!("CARGO_MANIFEST_DIR"),
			"/../../fixtures/llm-oracle/agent-protocols/gitlab/start_request.json"
		)))
		.expect("start request contract");
		let start = start.wire.start_request;
		assert_eq!(start.workflow_id.as_str(), "workflow-7");
		assert_eq!(start.session_id.as_str(), "session-a");
		assert_eq!(start.client_version.as_str(), CLIENT_VERSION);
		assert_eq!(start.workflow_definition.as_str(), "ambient");
		assert_eq!(
			start
				.client_capabilities
				.iter()
				.map(Str::as_str)
				.collect::<Vec<_>>(),
			[
				"incremental_streaming",
				"read_file_chunked",
				"shell_command",
				"command_timeout",
				"tool_call_approval",
			]
		);
		assert_eq!(start.mcp_tools.len(), 1);
		assert_eq!(start.mcp_tools[0].name.as_str(), "edit");
		assert_eq!(start.mcp_tools[0].input_schema, r#"{"type":"object"}"#);

		let auth: AuthFixture = serde_json::from_str(include_str!(concat!(
			env!("CARGO_MANIFEST_DIR"),
			"/../../fixtures/llm-oracle/agent-protocols/gitlab/auth_headers.json"
		)))
		.expect("auth header contract");
		assert_eq!(auth.headers.client_type.as_str(), "node-websocket");
		assert_eq!(auth.headers.language_server_version.as_str(), "8.104.0");
		assert_eq!(
			websocket_origin(auth.websocket_url.as_str()).expect("fixture origin"),
			auth.headers.origin
		);

		for expected in [
			include_str!(concat!(
				env!("CARGO_MANIFEST_DIR"),
				"/../../fixtures/llm-oracle/agent-protocols/gitlab/expected.reconnect.json"
			)),
			include_str!(concat!(
				env!("CARGO_MANIFEST_DIR"),
				"/../../fixtures/llm-oracle/agent-protocols/gitlab/expected.malformed.json"
			)),
		] {
			let contract: TerminalContract =
				serde_json::from_str(expected).expect("terminal contract");
			assert_eq!(contract.terminal_count, 1);
		}
		let cancel: CancelContract = serde_json::from_str(include_str!(concat!(
			env!("CARGO_MANIFEST_DIR"),
			"/../../fixtures/llm-oracle/agent-protocols/gitlab/expected.cancel.json"
		)))
		.expect("cancel contract");
		assert!(cancel.expected_transport.send_close_or_drop_socket);
		assert!(!cancel.expected_transport.fallback_transport);
		assert!(!cancel.expected_transport.terminal_event_after_consumer_drop);
	}

	#[derive(Deserialize)]
	struct ErrorCases {
		cases: Vec<ErrorCase>,
	}

	#[derive(Deserialize)]
	struct ErrorCase {
		expected: ErrorExpected,
	}

	#[derive(Deserialize)]
	struct ErrorExpected {
		diagnostic_code: Str,
		terminal_count:  u32,
	}

	#[test]
	fn every_error_contract_is_terminal_and_typed() {
		let errors: ErrorCases = serde_json::from_str(include_str!(concat!(
			env!("CARGO_MANIFEST_DIR"),
			"/../../fixtures/llm-oracle/agent-protocols/gitlab/errors.json"
		)))
		.expect("error contracts");
		assert_eq!(errors.cases.len(), 5);
		for case in errors.cases {
			assert_eq!(case.expected.terminal_count, 1);
			assert!(matches!(
				case.expected.diagnostic_code.as_str(),
				"transport" | "timeout" | "unsupported" | "auth" | "malformed_frame"
			));
		}
	}

	#[test]
	fn workflow_codec_rejects_custom_tool_grammars_before_wire_encoding() {
		let catalog = Catalog::embedded();
		let model = catalog.models().first().expect("embedded model");
		let route = model
			.routes
			.iter()
			.find_map(|route| catalog.route(route))
			.expect("embedded route");
		let policy = catalog
			.wire_policy(&model.wire_policy)
			.expect("embedded wire policy");
		let request_id = RequestId::new("gitlab-grammar-rejection");
		let context = EncodeContext {
			request_id: &request_id,
			route,
			target: None,
			policy_model: None,
			policy,
			thinking_policy: None,
			thinking_selection: None,
			session: None,
			server_state: None,
			account: None,
			attempt: EncodeAttempt { index: 0, provisional: false },
		};
		let request = ChatRequest {
			messages:          Arc::from([]),
			tools:             Arc::from([ToolDefinition {
				name:        Str::new_static("match_input"),
				description: None,
				input:       ToolInputConstraint::Grammar(ToolGrammar {
					syntax:     ToolGrammarSyntax::Lark,
					definition: Str::new_static("start: WORD"),
				}),
			}]),
			hosted_tools:      Arc::from([]),
			tool_choice:       Setting::Unset,
			output:            Setting::Unset,
			reasoning:         Setting::Unset,
			verbosity:         Setting::Unset,
			cache_retention:   Setting::Unset,
			service_tier:      Setting::Unset,
			sampling:          Sampling::default(),
			max_output_tokens: None,
			top_logprobs:      None,
			safety:            Arc::from([]),
			negotiation:       NegotiationPolicy::default(),
		};
		let error =
			match build_start_request(&request, &context, WorkflowSession::new("workflow", "session"))
			{
				Ok(_) => panic!("GitLab Workflow must reject custom tool grammars"),
				Err(error) => error,
			};
		assert_eq!(error.kind, ErrorKind::CapabilityMismatch);
		assert!(matches!(
			error.detail,
			Some(ErrorDetail::Protocol { reason })
				if reason.0.as_str() == "gitlab.workflow.tool_grammar.unsupported"
		));
	}
}
