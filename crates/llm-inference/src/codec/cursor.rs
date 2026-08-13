//! Cursor Agent Connect/protobuf request lowering and incremental event
//! projection.
//!
//! Bindings are generated into `OUT_DIR` from the verified checked-in schema.
//! The live `Run` endpoint is intentionally driven as a bidirectional Connect
//! stream even though the pinned descriptor declares the method unary;
//! descriptor tests make that observed drift explicit.

use std::{collections::BTreeMap, fmt};

use bytes::{BufMut as _, Bytes, BytesMut};
use omp_core::Str;
use omp_llm_catalog::{
	Availability, ChatCapabilities, DiscoveredModel, ExtendedContextMode, ModelCapabilities,
	OperationBits, OperationKind, ReasoningCapabilities, ReasoningFeatureBits, WireModelId,
};
use prost::Message;
use prost_types::FileDescriptorSet;

use super::{
	Codec, DecodeContext, Decoder, DecoderState, EncodeContext, EncodedRequest,
	ProviderControlEvent, ProviderStateEvent, RawCompletion, RawEvent, RequestHeader, RequestMethod,
	SizeBounds, ToolInputKind, UnvalidatedToolCall,
};
use crate::{
	body::BodySource,
	call::{ChatRequest, ContentPart, OperationCall, Role, Setting},
	error::{Error, ErrorDetail, ErrorKind, ErrorPhase, RetryAction},
	event::{BlockKind, ChatEvent, FinishReason, UsageUpdate},
	id::ToolCallId,
	receipt::{ExecutionReceipt, ReasonId, Usage, UsageSource},
	transport::{ConnectEnvelopeKind, Frame, FramingProtocol},
};

/// Prost bindings generated from the verified Cursor Agent schema.
pub mod wire {
	include!(concat!(env!("OUT_DIR"), "/agent.v1.rs"));
}

/// SHA-256 of the checked-in source compiled by this crate.
pub const SCHEMA_SHA256: &str = "aa6d1715e8ba8309c9049d3d1d9acbea75454f852a82ff22292843c1010ae527";
/// Repository commit from which the checked-in schema was recovered.
pub const SCHEMA_SOURCE_COMMIT: &str = "b6e01c8a3c836032823e13a404ceca2e968b6411";
/// Cursor's bidirectional Agent Connect method.
pub const RUN_PATH: &str = "/agent.v1.AgentService/Run";
/// Cursor's reconnect event-stream method.
pub const RUN_SSE_PATH: &str = "/agent.v1.AgentService/RunSSE";
/// Cursor's unary model-discovery method.
pub const DISCOVERY_PATH: &str = "/agent.v1.AgentService/GetUsableModels";
/// Cursor's non-secret client-version header value pinned by the protocol
/// fixtures.
pub const CLIENT_VERSION: &str = "cli-2026.07.23-e383d2b";
/// Maximum accepted protobuf payload size.
pub const MAX_MESSAGE_BYTES: usize = 16 * 1024 * 1024;

const CONNECT_END_STREAM: u8 = 0x02;

/// Encoded descriptor set for the exact schema used to generate [`wire`].
pub static FILE_DESCRIPTOR_SET: &[u8] =
	include_bytes!(concat!(env!("OUT_DIR"), "/cursor-agent-descriptor.bin"));

/// Decodes the generated binding descriptor for drift inspection.
pub fn descriptor_set() -> Result<FileDescriptorSet, prost::DecodeError> {
	FileDescriptorSet::decode(FILE_DESCRIPTOR_SET)
}

/// Stable Cursor codec failure class.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CursorErrorKind {
	/// Protobuf or Connect bytes violated the schema.
	Malformed,
	/// A complete stream ended without its required terminal signal.
	Truncated,
	/// Input arrived after terminal completion or cancellation.
	AfterTerminal,
	/// The caller cancelled decoding.
	Cancelled,
	/// Cursor rejected authentication.
	Authentication,
	/// Cursor returned a non-success status.
	Upstream,
	/// Cursor reported a context-window overflow.
	ContextOverflow,
	/// A requested canonical shape has no lossless Cursor projection.
	Unsupported,
}

/// Secret-free typed Cursor codec error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CursorProtocolError {
	/// Stable error classification.
	pub kind:      CursorErrorKind,
	/// Sanitized protocol reason.
	pub reason:    Str,
	/// HTTP status when the failure came from a response handshake.
	pub status:    Option<u16>,
	/// Whether an ordinary canonical event had already been emitted.
	pub committed: bool,
}

impl CursorProtocolError {
	fn new(kind: CursorErrorKind, reason: &'static str, committed: bool) -> Self {
		Self { kind, reason: Str::new_static(reason), committed, status: None }
	}
}

impl fmt::Display for CursorProtocolError {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter.write_str(self.reason.as_str())
	}
}

impl std::error::Error for CursorProtocolError {}

/// Maps an HTTP response status into Cursor's secret-free error vocabulary.
pub fn classify_http_status(status: u16) -> Option<CursorProtocolError> {
	let mut error = match status {
		200..=299 => return None,
		401 | 403 => CursorProtocolError::new(
			CursorErrorKind::Authentication,
			"Cursor authentication failed",
			false,
		),
		_ => CursorProtocolError::new(
			CursorErrorKind::Upstream,
			"Cursor Connect returned a non-success HTTP status",
			false,
		),
	};
	error.status = Some(status);
	Some(error)
}

/// Non-secret request-header profile selected by the operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CursorHeaderProfile {
	/// Bidirectional Connect/protobuf run.
	Run,
	/// Unary raw-protobuf discovery.
	Discovery,
}

/// Visits every public header required by the selected Cursor protocol profile.
pub fn for_each_public_header(
	profile: CursorHeaderProfile,
	mut visit: impl FnMut(&'static str, &'static str),
) {
	match profile {
		CursorHeaderProfile::Run => {
			visit("content-type", "application/connect+proto");
			visit("connect-protocol-version", "1");
		},
		CursorHeaderProfile::Discovery => {
			visit("content-type", "application/proto");
			visit("accept", "application/proto");
			visit("te", "trailers");
		},
	}
	visit("x-ghost-mode", "true");
	visit("x-cursor-client-version", CLIENT_VERSION);
	visit("x-cursor-client-type", "cli");
}

/// One caller-declared tool exposed through Cursor's MCP-compatible tool list.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CursorToolDefinition {
	/// Canonical tool name.
	pub name:              Str,
	/// Optional human-readable description.
	pub description:       Option<Str>,
	/// Exact JSON Schema text.
	pub input_schema_json: Str,
}

/// Instruction role retained inside Cursor's serialized root prompt messages.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CursorPromptRole {
	/// System instruction.
	System,
	/// Developer instruction.
	Developer,
}

impl CursorPromptRole {
	const fn as_str(self) -> &'static str {
		match self {
			Self::System => "system",
			Self::Developer => "developer",
		}
	}
}

/// One typed Cursor root prompt message.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CursorRootPrompt {
	/// Semantic instruction role.
	pub role: CursorPromptRole,
	/// Instruction text.
	pub text: Str,
}

/// Action carried by one Cursor Agent run request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CursorRunAction {
	/// Submit one user message.
	UserMessage {
		/// Stable message identity supplied by the caller.
		message_id: Str,
		/// Plain-text message body.
		text:       Str,
	},
	/// Resume from the supplied provider checkpoint.
	Resume,
	/// Cancel the active provider turn.
	Cancel,
}

/// Typed inputs for a Cursor `AgentRunRequest`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CursorRunRequest {
	/// Opaque selected wire model identity.
	pub model_id:        Str,
	/// Whether the catalog-selected Cursor max mode is enabled.
	pub max_mode:        bool,
	/// Optional provider conversation identity.
	pub conversation_id: Option<Str>,
	/// Serialized `ConversationStateStructure` from an authoritative checkpoint.
	pub checkpoint:      Option<Bytes>,
	/// Ordered system/developer prompt messages for a fresh session.
	pub root_prompts:    Box<[CursorRootPrompt]>,
	/// Caller tools projected through Cursor's MCP tool schema.
	pub tools:           Box<[CursorToolDefinition]>,
	/// Current action.
	pub action:          CursorRunAction,
}

/// Opaque authoritative Cursor session checkpoint bound by session middleware.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CursorSessionCheckpoint {
	/// Optional provider conversation identity.
	pub conversation_id: Option<Str>,
	/// Serialized `ConversationStateStructure`.
	pub state:           Bytes,
}

/// Typed request for Cursor's `RunSSE` reconnect method.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CursorReconnectRequest {
	/// Stable request identity whose server stream is resumed.
	pub request_id: Str,
}

/// Encodes one typed run request as a Connect data envelope.
pub fn encode_run_request(request: &CursorRunRequest) -> Result<Bytes, CursorProtocolError> {
	let state = request
		.checkpoint
		.as_ref()
		.map(|bytes| wire::ConversationStateStructure::decode(bytes.clone()))
		.transpose()
		.map_err(|_| {
			CursorProtocolError::new(
				CursorErrorKind::Malformed,
				"invalid Cursor session checkpoint",
				false,
			)
		})?
		.unwrap_or_else(|| wire::ConversationStateStructure {
			root_prompt_messages_json: request
				.root_prompts
				.iter()
				.map(encode_root_prompt)
				.collect(),
			..Default::default()
		});

	let action = match &request.action {
		CursorRunAction::UserMessage { message_id, text } => {
			wire::conversation_action::Action::UserMessageAction(wire::UserMessageAction {
				user_message:                 Some(wire::UserMessage {
					text: text.as_str().to_owned(),
					message_id: message_id.as_str().to_owned(),
					..Default::default()
				}),
				request_context:              None,
				send_to_interaction_listener: None,
			})
		},
		CursorRunAction::Resume => {
			wire::conversation_action::Action::ResumeAction(wire::ResumeAction {
				request_context: None,
			})
		},
		CursorRunAction::Cancel => {
			wire::conversation_action::Action::CancelAction(wire::CancelAction {})
		},
	};
	let tools = request
		.tools
		.iter()
		.map(|tool| wire::McpToolDefinition {
			name:                tool.name.as_str().to_owned(),
			provider_identifier: "omp".to_owned(),
			tool_name:           tool.name.as_str().to_owned(),
			description:         tool
				.description
				.as_ref()
				.map_or_else(String::new, |value| value.as_str().to_owned()),
			input_schema:        Bytes::new(),
			input_schema_json:   Some(tool.input_schema_json.as_str().to_owned()),
		})
		.collect();
	let model = wire::ModelDetails {
		model_id: request.model_id.as_str().to_owned(),
		display_model_id: request.model_id.as_str().to_owned(),
		display_name: request.model_id.as_str().to_owned(),
		max_mode: Some(request.max_mode),
		..Default::default()
	};
	let run = wire::AgentRunRequest {
		conversation_state: Some(state),
		action: Some(wire::ConversationAction { action: Some(action) }),
		model_details: Some(model),
		requested_model: Some(wire::RequestedModel {
			model_id: request.model_id.as_str().to_owned(),
			max_mode: request.max_mode,
			..Default::default()
		}),
		mcp_tools: Some(wire::McpTools { mcp_tools: tools }),
		conversation_id: request
			.conversation_id
			.as_ref()
			.map(|id| id.as_str().to_owned()),
		..Default::default()
	};
	Ok(connect_message(&wire::AgentClientMessage {
		message: Some(wire::agent_client_message::Message::RunRequest(run)),
	}))
}

/// Encodes the request body for `RunSSE` reconnect.
pub fn encode_reconnect_request(request: &CursorReconnectRequest) -> Bytes {
	Bytes::from(
		wire::BidiRequestId { request_id: request.request_id.as_str().to_owned() }.encode_to_vec(),
	)
}

/// Encodes the unary model-discovery request body.
pub fn encode_discovery_request(custom_model_ids: &[Str]) -> Bytes {
	Bytes::from(
		wire::GetUsableModelsRequest {
			custom_model_ids: custom_model_ids
				.iter()
				.map(|id| id.as_str().to_owned())
				.collect(),
		}
		.encode_to_vec(),
	)
}

/// Adds a Connect data envelope around one protobuf message.
pub fn connect_message(message: &impl Message) -> Bytes {
	let payload_len = message.encoded_len();
	let mut bytes = BytesMut::with_capacity(payload_len + 5);
	bytes.put_u8(0);
	bytes.put_u32(u32::try_from(payload_len).expect("Cursor protobuf message exceeds u32 framing"));
	message.encode(&mut bytes).expect("BytesMut is growable");
	bytes.freeze()
}

fn encode_root_prompt(prompt: &CursorRootPrompt) -> Bytes {
	#[derive(serde::Serialize)]
	struct RootPrompt<'a> {
		role:    &'static str,
		content: &'a str,
	}

	Bytes::from(
		serde_json::to_vec(&RootPrompt {
			role:    prompt.role.as_str(),
			content: prompt.text.as_str(),
		})
		.expect("a borrowed string always serializes as JSON"),
	)
}

/// Non-secret model facts observed directly in Cursor discovery.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CursorDiscoveredModel {
	/// Cursor model identity.
	pub id:        Str,
	/// Human-readable display name.
	pub name:      Str,
	/// Cursor aliases reported on the wire.
	pub aliases:   Box<[Str]>,
	/// Whether Cursor supplied thinking metadata.
	pub reasoning: bool,
	/// Whether Cursor advertises max mode.
	pub max_mode:  bool,
}

/// Decodes a raw protobuf or one Connect data envelope from model discovery.
pub fn decode_discovery_response(
	payload: &[u8],
) -> Result<Vec<CursorDiscoveredModel>, CursorProtocolError> {
	let protobuf = first_discovery_message(payload)?;
	let response = wire::GetUsableModelsResponse::decode(protobuf).map_err(|_| {
		CursorProtocolError::new(
			CursorErrorKind::Malformed,
			"malformed Cursor discovery protobuf",
			false,
		)
	})?;
	let mut models = BTreeMap::new();
	for model in response.models {
		let id = model.model_id.trim();
		if id.is_empty() {
			continue;
		}
		let name = if model.display_name.trim().is_empty() {
			id
		} else {
			model.display_name.trim()
		};
		models.insert(id.to_owned(), CursorDiscoveredModel {
			id:        Str::from(id),
			name:      Str::from(name),
			aliases:   model.aliases.into_iter().map(Str::from).collect(),
			reasoning: model.thinking_details.is_some(),
			max_mode:  model.max_mode.unwrap_or(false),
		});
	}
	Ok(models.into_values().collect())
}

fn first_discovery_message(payload: &[u8]) -> Result<&[u8], CursorProtocolError> {
	if payload
		.first()
		.copied()
		.is_some_and(|flags| flags & !0x03 == 0)
	{
		if payload.len() < 5 {
			return Err(CursorProtocolError::new(
				CursorErrorKind::Truncated,
				"incomplete Cursor discovery frame header",
				false,
			));
		}
		let length =
			u32::from_be_bytes(payload[1..5].try_into().expect("fixed four-byte length")) as usize;
		let end = 5usize.checked_add(length).ok_or_else(|| {
			CursorProtocolError::new(
				CursorErrorKind::Malformed,
				"Cursor discovery frame length overflow",
				false,
			)
		})?;
		if end > payload.len() {
			return Err(CursorProtocolError::new(
				CursorErrorKind::Truncated,
				"incomplete Cursor discovery frame payload",
				false,
			));
		}
		if payload[0] & CONNECT_END_STREAM != 0 {
			return Err(CursorProtocolError::new(
				CursorErrorKind::Truncated,
				"Cursor discovery returned only an end-stream frame",
				false,
			));
		}
		return Ok(&payload[5..end]);
	}
	Ok(payload)
}

/// Cursor server-requested shell execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CursorShellInvocation {
	/// Numeric correlation identifier.
	pub id:                u32,
	/// Optional attachable execution identity.
	pub exec_id:           Str,
	/// Canonical tool-call identity.
	pub call_id:           ToolCallId,
	/// Command text.
	pub command:           Str,
	/// Requested working directory.
	pub working_directory: Str,
	/// Soft timeout in milliseconds.
	pub timeout_ms:        u32,
	/// Whether Cursor expects incremental `ShellStream` frames.
	pub streaming:         bool,
}

/// Completed shell execution supplied back to Cursor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CursorShellCompletion {
	/// Process exited successfully.
	Exited {
		/// Captured standard output.
		stdout:                  Str,
		/// Captured standard error.
		stderr:                  Str,
		/// Local execution duration.
		local_execution_time_ms: u32,
	},
	/// Process exited unsuccessfully.
	Failed {
		/// Exit code.
		code:                    u32,
		/// Captured standard output.
		stdout:                  Str,
		/// Captured standard error.
		stderr:                  Str,
		/// Local execution duration.
		local_execution_time_ms: u32,
	},
	/// Caller policy rejected execution.
	Rejected {
		/// Sanitized policy reason.
		reason:      Str,
		/// Whether the command was read-only.
		is_readonly: bool,
	},
	/// Caller denied execution permission.
	PermissionDenied {
		/// Sanitized denial reason.
		reason:      Str,
		/// Whether the command was read-only.
		is_readonly: bool,
	},
	/// Execution exceeded the declared deadline.
	TimedOut {
		/// Applied timeout in milliseconds.
		timeout_ms: u32,
	},
}

/// Builds Cursor's initial shell-stream frame.
pub fn shell_start(invocation: &CursorShellInvocation) -> wire::AgentClientMessage {
	exec_message(
		invocation,
		wire::exec_client_message::Message::ShellStream(wire::ShellStream {
			event: Some(wire::shell_stream::Event::Start(wire::ShellStreamStart {
				sandbox_policy: None,
			})),
		}),
		None,
	)
}

/// Builds one incremental shell stdout frame.
pub fn shell_stdout(invocation: &CursorShellInvocation, text: &str) -> wire::AgentClientMessage {
	exec_message(
		invocation,
		wire::exec_client_message::Message::ShellStream(wire::ShellStream {
			event: Some(wire::shell_stream::Event::Stdout(wire::ShellStreamStdout {
				data: text.to_owned(),
			})),
		}),
		None,
	)
}

/// Builds one incremental shell stderr frame.
pub fn shell_stderr(invocation: &CursorShellInvocation, text: &str) -> wire::AgentClientMessage {
	exec_message(
		invocation,
		wire::exec_client_message::Message::ShellStream(wire::ShellStream {
			event: Some(wire::shell_stream::Event::Stderr(wire::ShellStreamStderr {
				data: text.to_owned(),
			})),
		}),
		None,
	)
}

/// Builds the ordered terminal shell frames: optional diagnostic, exit, result,
/// close.
pub fn shell_completion_frames(
	invocation: &CursorShellInvocation,
	completion: &CursorShellCompletion,
) -> Vec<wire::AgentClientMessage> {
	let mut frames = Vec::with_capacity(4);
	if let CursorShellCompletion::TimedOut { timeout_ms } = completion {
		frames.push(shell_stderr(invocation, &format!("Command timed out after {timeout_ms}ms")));
	}
	if matches!(completion, CursorShellCompletion::Rejected { .. }) {
		frames.push(exec_message(
			invocation,
			wire::exec_client_message::Message::ShellStream(wire::ShellStream {
				event: Some(wire::shell_stream::Event::Rejected(shell_rejected(
					invocation, completion,
				))),
			}),
			None,
		));
	}
	if matches!(completion, CursorShellCompletion::PermissionDenied { .. }) {
		frames.push(exec_message(
			invocation,
			wire::exec_client_message::Message::ShellStream(wire::ShellStream {
				event: Some(wire::shell_stream::Event::PermissionDenied(shell_denied(
					invocation, completion,
				))),
			}),
			None,
		));
	}
	let (code, aborted, local_ms) = match completion {
		CursorShellCompletion::Exited { local_execution_time_ms, .. } => {
			(0, false, Some(*local_execution_time_ms as i32))
		},
		CursorShellCompletion::Failed { code, local_execution_time_ms, .. } => {
			(*code, false, Some(*local_execution_time_ms as i32))
		},
		CursorShellCompletion::Rejected { .. } | CursorShellCompletion::PermissionDenied { .. } => {
			(1, false, None)
		},
		CursorShellCompletion::TimedOut { .. } => (1, true, None),
	};
	frames.push(exec_message(
		invocation,
		wire::exec_client_message::Message::ShellStream(wire::ShellStream {
			event: Some(wire::shell_stream::Event::Exit(wire::ShellStreamExit {
				code,
				cwd: invocation.working_directory.as_str().to_owned(),
				output_location: None,
				aborted,
				abort_reason: None,
				local_execution_time_ms: local_ms,
			})),
		}),
		local_ms,
	));
	frames.push(exec_message(invocation, shell_result(invocation, completion), local_ms));
	frames.push(wire::AgentClientMessage {
		message: Some(wire::agent_client_message::Message::ExecClientControlMessage(
			wire::ExecClientControlMessage {
				message: Some(wire::exec_client_control_message::Message::StreamClose(
					wire::ExecClientStreamClose { id: invocation.id },
				)),
			},
		)),
	});
	frames
}

fn exec_message(
	invocation: &CursorShellInvocation,
	message: wire::exec_client_message::Message,
	local_execution_time_ms: Option<i32>,
) -> wire::AgentClientMessage {
	wire::AgentClientMessage {
		message: Some(wire::agent_client_message::Message::ExecClientMessage(
			wire::ExecClientMessage {
				id: invocation.id,
				exec_id: invocation.exec_id.as_str().to_owned(),
				message: Some(message),
				local_execution_time_ms,
				..Default::default()
			},
		)),
	}
}

fn shell_result(
	invocation: &CursorShellInvocation,
	completion: &CursorShellCompletion,
) -> wire::exec_client_message::Message {
	let result = match completion {
		CursorShellCompletion::Exited { stdout, stderr, local_execution_time_ms } => {
			wire::shell_result::Result::Success(wire::ShellSuccess {
				command: invocation.command.as_str().to_owned(),
				working_directory: invocation.working_directory.as_str().to_owned(),
				exit_code: 0,
				stdout: stdout.as_str().to_owned(),
				stderr: stderr.as_str().to_owned(),
				execution_time: *local_execution_time_ms as i32,
				local_execution_time_ms: Some(*local_execution_time_ms as i32),
				..Default::default()
			})
		},
		CursorShellCompletion::Failed { code, stdout, stderr, local_execution_time_ms } => {
			wire::shell_result::Result::Failure(wire::ShellFailure {
				command: invocation.command.as_str().to_owned(),
				working_directory: invocation.working_directory.as_str().to_owned(),
				exit_code: *code as i32,
				stdout: stdout.as_str().to_owned(),
				stderr: stderr.as_str().to_owned(),
				execution_time: *local_execution_time_ms as i32,
				local_execution_time_ms: Some(*local_execution_time_ms as i32),
				..Default::default()
			})
		},
		CursorShellCompletion::Rejected { .. } => {
			wire::shell_result::Result::Rejected(shell_rejected(invocation, completion))
		},
		CursorShellCompletion::PermissionDenied { .. } => {
			wire::shell_result::Result::PermissionDenied(shell_denied(invocation, completion))
		},
		CursorShellCompletion::TimedOut { timeout_ms } => {
			wire::shell_result::Result::Timeout(wire::ShellTimeout {
				command:           invocation.command.as_str().to_owned(),
				working_directory: invocation.working_directory.as_str().to_owned(),
				timeout_ms:        *timeout_ms as i32,
			})
		},
	};
	wire::exec_client_message::Message::ShellResult(wire::ShellResult {
		result: Some(result),
		..Default::default()
	})
}

fn shell_rejected(
	invocation: &CursorShellInvocation,
	completion: &CursorShellCompletion,
) -> wire::ShellRejected {
	let CursorShellCompletion::Rejected { reason, is_readonly } = completion else {
		unreachable!("shell_rejected called with another completion")
	};
	wire::ShellRejected {
		command:           invocation.command.as_str().to_owned(),
		working_directory: invocation.working_directory.as_str().to_owned(),
		reason:            reason.as_str().to_owned(),
		is_readonly:       *is_readonly,
	}
}

fn shell_denied(
	invocation: &CursorShellInvocation,
	completion: &CursorShellCompletion,
) -> wire::ShellPermissionDenied {
	let CursorShellCompletion::PermissionDenied { reason, is_readonly } = completion else {
		unreachable!("shell_denied called with another completion")
	};
	wire::ShellPermissionDenied {
		command:           invocation.command.as_str().to_owned(),
		working_directory: invocation.working_directory.as_str().to_owned(),
		error:             reason.as_str().to_owned(),
		is_readonly:       *is_readonly,
	}
}

/// Provider-specific output that cannot be represented as generative chat text.
#[derive(Debug)]
pub enum CursorEvent {
	/// Canonical generative event.
	Chat(ChatEvent),
	/// Complete but not yet schema-validated tool arguments.
	ToolCallComplete {
		/// Canonical block index.
		index:     u32,
		/// Tool-call identity.
		id:        ToolCallId,
		/// Tool name.
		name:      Str,
		/// Exact assembled argument bytes.
		arguments: Bytes,
	},
	/// Cursor requested shell execution.
	ShellInvoke(CursorShellInvocation),
	/// Cursor cancelled one outstanding execution.
	InvokeCancel {
		/// Numeric Cursor correlation identifier.
		id: u32,
	},
	/// Authoritative provider checkpoint for session resume.
	Checkpoint {
		/// Encoded `ConversationStateStructure`.
		data: Bytes,
	},
	/// Correlated Cursor interaction query requiring a typed response.
	InteractionQuery {
		/// Query correlation identifier.
		id:    u32,
		/// Generated typed query payload.
		query: wire::interaction_query::Query,
	},
	/// Terminal chat facts awaiting final receipt accounting.
	Completion {
		/// Normalized provider finish reason.
		reason: FinishReason,
		/// Number of canonical blocks emitted.
		blocks: u32,
		/// Final provider-reported usage.
		usage:  Usage,
	},
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OpenKind {
	Text,
	Thinking,
	Tool,
}

#[derive(Debug)]
struct OpenBlock {
	index:     u32,
	kind:      OpenKind,
	tool_id:   ToolCallId,
	tool_name: Str,
	arguments: BytesMut,
}

/// Stateful protobuf projector for one Cursor Agent attempt.
#[derive(Debug, Default)]
pub struct CursorDecoder {
	open:       Option<OpenBlock>,
	next_index: u32,
	blocks:     u32,
	usage:      Usage,
	saw_usage:  bool,
	saw_tool:   bool,
	committed:  bool,
	terminal:   bool,
	cancelled:  bool,
}

impl CursorDecoder {
	/// Decodes one complete `AgentServerMessage` protobuf payload.
	pub fn push_payload(&mut self, payload: Bytes) -> Result<Vec<CursorEvent>, CursorProtocolError> {
		if self.cancelled {
			return Err(CursorProtocolError::new(
				CursorErrorKind::Cancelled,
				"Cursor decoder is cancelled",
				self.committed,
			));
		}
		if self.terminal {
			return Err(CursorProtocolError::new(
				CursorErrorKind::AfterTerminal,
				"Cursor payload arrived after terminal completion",
				self.committed,
			));
		}
		if payload.len() > MAX_MESSAGE_BYTES {
			return Err(CursorProtocolError::new(
				CursorErrorKind::Malformed,
				"Cursor protobuf message exceeds codec bound",
				self.committed,
			));
		}
		let message = wire::AgentServerMessage::decode(payload).map_err(|_| {
			CursorProtocolError::new(
				CursorErrorKind::Malformed,
				"malformed Cursor AgentServerMessage",
				self.committed,
			)
		})?;
		self.project(message)
	}

	/// Applies a Connect end-stream payload without exposing it as protobuf.
	pub fn push_end_stream(
		&mut self,
		payload: &[u8],
	) -> Result<Vec<CursorEvent>, CursorProtocolError> {
		if self.cancelled {
			return Err(CursorProtocolError::new(
				CursorErrorKind::Cancelled,
				"Cursor decoder is cancelled",
				self.committed,
			));
		}
		#[derive(serde::Deserialize)]
		struct EndStream<'a> {
			#[serde(borrow)]
			error: Option<EndError<'a>>,
		}
		#[derive(serde::Deserialize)]
		struct EndError<'a> {
			#[serde(default, borrow)]
			code: Option<&'a str>,
		}
		let trailer = serde_json::from_slice::<EndStream<'_>>(payload).map_err(|_| {
			CursorProtocolError::new(
				CursorErrorKind::Malformed,
				"malformed Cursor Connect end-stream payload",
				self.committed,
			)
		})?;
		if let Some(error) = trailer.error {
			let kind = match error.code {
				Some("context_length_exceeded" | "context_overflow") => {
					CursorErrorKind::ContextOverflow
				},
				_ => CursorErrorKind::Upstream,
			};
			return Err(CursorProtocolError::new(
				kind,
				"Cursor Connect end-stream error",
				self.committed,
			));
		}
		self.terminal = true;
		Ok(Vec::new())
	}

	/// Marks local cancellation and prevents late provider frames from
	/// surfacing.
	pub fn cancel(&mut self) {
		self.cancelled = true;
		self.open = None;
	}

	/// Verifies that the attempt reached a protocol terminal.
	pub fn finish(&mut self) -> Result<(), CursorProtocolError> {
		if self.cancelled || self.terminal {
			return Ok(());
		}
		Err(CursorProtocolError::new(
			CursorErrorKind::Truncated,
			"Cursor stream ended before terminal completion",
			self.committed,
		))
	}

	fn project(
		&mut self,
		message: wire::AgentServerMessage,
	) -> Result<Vec<CursorEvent>, CursorProtocolError> {
		match message.message {
			Some(wire::agent_server_message::Message::InteractionUpdate(update)) => {
				self.project_interaction(update)
			},
			Some(wire::agent_server_message::Message::ExecServerMessage(exec)) => {
				Ok(vec![CursorEvent::ShellInvoke(shell_invocation(exec)?)])
			},
			Some(wire::agent_server_message::Message::ExecServerControlMessage(control)) => {
				let Some(wire::exec_server_control_message::Message::Abort(abort)) = control.message
				else {
					return Ok(Vec::new());
				};
				Ok(vec![CursorEvent::InvokeCancel { id: abort.id }])
			},
			Some(wire::agent_server_message::Message::ConversationCheckpointUpdate(checkpoint)) => {
				Ok(vec![CursorEvent::Checkpoint { data: Bytes::from(checkpoint.encode_to_vec()) }])
			},
			Some(wire::agent_server_message::Message::InteractionQuery(query)) => {
				let Some(payload) = query.query else {
					return Ok(Vec::new());
				};
				Ok(vec![CursorEvent::InteractionQuery { id: query.id, query: payload }])
			},
			Some(wire::agent_server_message::Message::KvServerMessage(_)) | None => Ok(Vec::new()),
		}
	}

	fn project_interaction(
		&mut self,
		update: wire::InteractionUpdate,
	) -> Result<Vec<CursorEvent>, CursorProtocolError> {
		use wire::interaction_update::Message;
		let mut events = Vec::with_capacity(3);
		match update.message {
			Some(Message::TextDelta(delta)) => self.push_text(OpenKind::Text, delta.text, &mut events),
			Some(Message::ThinkingDelta(delta)) => {
				self.push_text(OpenKind::Thinking, delta.text, &mut events)
			},
			Some(Message::ToolCallStarted(started)) => {
				self.start_tool(started.call_id, started.tool_call.as_ref(), &mut events)
			},
			Some(Message::PartialToolCall(partial)) => {
				let id = call_id(partial.call_id, partial.tool_call.as_ref());
				if !matches!(self.open.as_ref(), Some(open) if open.kind == OpenKind::Tool && open.tool_id == id)
				{
					self.start_tool(id.as_str().to_owned(), partial.tool_call.as_ref(), &mut events);
				}
				if let Some(open) = self.open.as_mut()
					&& !partial.args_text_delta.is_empty()
				{
					let bytes = Bytes::from(partial.args_text_delta);
					open.arguments.extend_from_slice(&bytes);
					events.push(CursorEvent::Chat(ChatEvent::ToolArgumentsDelta {
						index: open.index,
						bytes,
					}));
					self.committed = true;
				}
			},
			Some(Message::ToolCallCompleted(completed)) => {
				let id = call_id(completed.call_id, completed.tool_call.as_ref());
				if let Some(open) = self.open.take() {
					if open.kind != OpenKind::Tool || (!id.as_str().is_empty() && open.tool_id != id) {
						self.open = Some(open);
						return Err(CursorProtocolError::new(
							CursorErrorKind::Malformed,
							"Cursor completed a different tool call",
							self.committed,
						));
					}
					events.push(CursorEvent::ToolCallComplete {
						index:     open.index,
						id:        open.tool_id,
						name:      open.tool_name,
						arguments: open.arguments.freeze(),
					});
				}
			},
			Some(Message::TokenDelta(delta)) => {
				self.usage.output_tokens = self
					.usage
					.output_tokens
					.saturating_add(delta.tokens.max(0) as u64);
				self.usage.source = UsageSource::Provider;
				self.saw_usage = true;
			},
			Some(Message::TurnEnded(_)) => {
				self.open = None;
				if self.saw_usage {
					events.push(CursorEvent::Chat(ChatEvent::Usage(UsageUpdate {
						usage:        self.usage,
						final_update: true,
					})));
				}
				events.push(CursorEvent::Completion {
					reason: if self.saw_tool {
						FinishReason::ToolCalls
					} else {
						FinishReason::Stop
					},
					blocks: self.blocks,
					usage:  self.usage,
				});
				self.committed = true;
				self.terminal = true;
			},
			Some(
				Message::ThinkingCompleted(_)
				| Message::UserMessageAppended(_)
				| Message::Summary(_)
				| Message::SummaryStarted(_)
				| Message::SummaryCompleted(_)
				| Message::ShellOutputDelta(_)
				| Message::Heartbeat(_)
				| Message::ToolCallDelta(_)
				| Message::StepStarted(_)
				| Message::StepCompleted(_),
			)
			| None => {},
		}
		Ok(events)
	}

	fn push_text(&mut self, kind: OpenKind, text: String, events: &mut Vec<CursorEvent>) {
		let index = if let Some(open) = self.open.as_ref().filter(|open| open.kind == kind) {
			open.index
		} else {
			let index = self.start_block(kind, ToolCallId::default(), Str::default());
			events.push(CursorEvent::Chat(ChatEvent::BlockStarted {
				index,
				kind: if kind == OpenKind::Text {
					BlockKind::Text
				} else {
					BlockKind::Thinking
				},
			}));
			self.committed = true;
			index
		};
		if text.is_empty() {
			return;
		}
		let text = Str::from(text);
		events.push(CursorEvent::Chat(if kind == OpenKind::Text {
			ChatEvent::TextDelta { index, text }
		} else {
			ChatEvent::ThinkingDelta { index, text }
		}));
		self.committed = true;
	}

	fn start_tool(
		&mut self,
		call_id_text: String,
		tool: Option<&wire::ToolCall>,
		events: &mut Vec<CursorEvent>,
	) {
		let id = call_id(call_id_text, tool);
		let name = tool_name(tool);
		let index = self.start_block(OpenKind::Tool, id.clone(), name.clone());
		self.saw_tool = true;
		events.push(CursorEvent::Chat(ChatEvent::BlockStarted { index, kind: BlockKind::ToolCall }));
		events.push(CursorEvent::Chat(ChatEvent::ToolCallStarted { index, id, name }));
		self.committed = true;
	}

	fn start_block(&mut self, kind: OpenKind, tool_id: ToolCallId, tool_name: Str) -> u32 {
		let index = self.next_index;
		self.next_index = self.next_index.saturating_add(1);
		self.blocks = self.blocks.saturating_add(1);
		self.open = Some(OpenBlock { index, kind, tool_id, tool_name, arguments: BytesMut::new() });
		index
	}
}

fn shell_invocation(
	exec: wire::ExecServerMessage,
) -> Result<CursorShellInvocation, CursorProtocolError> {
	let (args, streaming) = match exec.message {
		Some(wire::exec_server_message::Message::ShellArgs(args)) => (args, false),
		Some(wire::exec_server_message::Message::ShellStreamArgs(args)) => (args, true),
		_ => {
			return Err(CursorProtocolError::new(
				CursorErrorKind::Unsupported,
				"Cursor requested an unsupported exec operation",
				false,
			));
		},
	};
	Ok(CursorShellInvocation {
		id: exec.id,
		exec_id: Str::from(exec.exec_id),
		call_id: ToolCallId::from(args.tool_call_id),
		command: Str::from(args.command),
		working_directory: Str::from(args.working_directory),
		timeout_ms: args.timeout.max(0) as u32,
		streaming,
	})
}

fn call_id(call_id: String, tool: Option<&wire::ToolCall>) -> ToolCallId {
	if call_id.is_empty() {
		ToolCallId::from(
			tool
				.and_then(|tool| tool.tool_call_id.as_deref())
				.unwrap_or_default(),
		)
	} else {
		ToolCallId::from(call_id)
	}
}

fn tool_name(tool: Option<&wire::ToolCall>) -> Str {
	use wire::tool_call::Tool;
	let Some(tool) = tool.and_then(|tool| tool.tool.as_ref()) else {
		return Str::default();
	};
	let name = match tool {
		Tool::ShellToolCall(_) | Tool::PiBashToolCall(_) => "bash",
		Tool::DeleteToolCall(_) => "delete",
		Tool::GlobToolCall(_) => "glob",
		Tool::GrepToolCall(_) | Tool::PiGrepToolCall(_) => "grep",
		Tool::ReadToolCall(_) | Tool::PiReadToolCall(_) => "read",
		Tool::UpdateTodosToolCall(_) => "update_todos",
		Tool::ReadTodosToolCall(_) => "read_todos",
		Tool::EditToolCall(_) | Tool::PiEditToolCall(_) => "edit",
		Tool::LsToolCall(_) | Tool::PiLsToolCall(_) => "ls",
		Tool::ReadLintsToolCall(_) => "read_lints",
		Tool::McpToolCall(call) => {
			return call.args.as_ref().map_or_else(Str::default, |args| {
				Str::from(if args.tool_name.is_empty() {
					args.name.as_str()
				} else {
					args.tool_name.as_str()
				})
			});
		},
		Tool::SemSearchToolCall(_) => "sem_search",
		Tool::CreatePlanToolCall(_) => "create_plan",
		Tool::WebSearchToolCall(_) => "web_search",
		Tool::TaskToolCall(_) => "task",
		Tool::ListMcpResourcesToolCall(_) => "list_mcp_resources",
		Tool::ReadMcpResourceToolCall(_) => "read_mcp_resource",
		Tool::ApplyAgentDiffToolCall(_) => "apply_agent_diff",
		Tool::AskQuestionToolCall(_) => "ask_question",
		Tool::FetchToolCall(_) => "fetch",
		Tool::SwitchModeToolCall(_) => "switch_mode",
		Tool::ExaSearchToolCall(_) => "exa_search",
		Tool::ExaFetchToolCall(_) => "exa_fetch",
		Tool::GenerateImageToolCall(_) => "generate_image",
		Tool::RecordScreenToolCall(_) => "record_screen",
		Tool::ComputerUseToolCall(_) => "computer_use",
		Tool::WriteShellStdinToolCall(_) => "write_shell_stdin",
		Tool::ReflectToolCall(_) => "reflect",
		Tool::SetupVmEnvironmentToolCall(_) => "setup_vm_environment",
		Tool::TruncatedToolCall(_) => "truncated",
		Tool::StartGrindExecutionToolCall(_) => "start_grind_execution",
		Tool::StartGrindPlanningToolCall(_) => "start_grind_planning",
		Tool::PiWriteToolCall(_) => "write",
		Tool::PiFindToolCall(_) => "find",
		Tool::ConnectScmToolCall(_) => "connect_scm",
		Tool::SearchConversationsToolCall(_) => "search_conversations",
	};
	Str::new_static(name)
}

/// Sans-I/O Cursor Agent codec registered under the catalog codec id `cursor`.
#[derive(Clone, Copy, Debug, Default)]
pub struct CursorCodec;

impl CursorCodec {
	/// Constructs the stateless Cursor codec.
	pub const fn new() -> Self {
		Self
	}
}

impl Codec for CursorCodec {
	fn encode(
		&self,
		context: &EncodeContext<'_>,
		operation: &OperationCall,
	) -> Result<EncodedRequest, Error> {
		match operation {
			OperationCall::Chat(request) => encode_chat_call(context, request),
			OperationCall::DiscoverModels(request) => {
				if request.cursor.is_some() {
					return Err(encoding_error("cursor_discovery_has_no_pagination"));
				}
				let mut headers = Vec::with_capacity(6);
				for_each_public_header(CursorHeaderProfile::Discovery, |name, value| {
					headers.push(RequestHeader {
						name:  Str::new_static(name),
						value: Str::new_static(value),
					});
				});
				Ok(EncodedRequest::new(
					OperationKind::DiscoverModels,
					RequestMethod::Post,
					endpoint_uri(context.route.endpoint.base_url.as_str(), DISCOVERY_PATH),
					headers.into_boxed_slice(),
					BodySource::bytes(encode_discovery_request(&[])),
					FramingProtocol::Raw,
					cursor_bounds(),
				))
			},
			_ => Err(encoding_error("cursor_operation_not_supported")),
		}
	}

	fn decoder(&self, context: &DecodeContext<'_>) -> Result<DecoderState, Error> {
		if !matches!(context.operation, OperationKind::Chat | OperationKind::DiscoverModels) {
			return Err(encoding_error("cursor_operation_not_supported"));
		}
		Ok(Box::new(CursorWireDecoder {
			operation:      context.operation,
			provider:       context.provider.clone(),
			route:          context.route.clone(),
			agent:          CursorDecoder::default(),
			discovery_done: false,
		}))
	}
}

fn encode_chat_call(
	context: &EncodeContext<'_>,
	request: &ChatRequest,
) -> Result<EncodedRequest, Error> {
	reject_unprojected_chat_options(request)?;
	let mut roots = Vec::new();
	let mut user = None;
	let target = context
		.target
		.ok_or_else(|| encoding_error("cursor_chat_requires_wire_target"))?;
	for message in request.messages.iter() {
		if message.name.is_some() {
			return Err(encoding_error("cursor_named_message_not_supported"));
		}
		let text = cursor_message_text(context, message.content.as_ref())?;
		match message.role {
			Role::System => roots.push(CursorRootPrompt { role: CursorPromptRole::System, text }),
			Role::Developer => {
				roots.push(CursorRootPrompt { role: CursorPromptRole::Developer, text })
			},
			Role::User if user.is_none() => user = Some(text),
			Role::User => return Err(encoding_error("cursor_requires_delta_or_checkpoint_context")),
			Role::Assistant | Role::Tool => {
				return Err(encoding_error("cursor_requires_provider_checkpoint_for_history"));
			},
		}
	}
	let user = user.ok_or_else(|| encoding_error("cursor_chat_requires_user_message"))?;
	let tools = request
		.tools
		.iter()
		.map(|tool| {
			let Some((parameters, _)) = tool.input.json_schema() else {
				return Err(encoding_error("cursor_tool_grammar_unsupported"));
			};
			let schema = serde_json::to_string(parameters.as_value())
				.map_err(|_| encoding_error("cursor_tool_schema_not_serializable"))?;
			Ok(CursorToolDefinition {
				name:              tool.name.clone(),
				description:       tool.description.clone(),
				input_schema_json: Str::from(schema),
			})
		})
		.collect::<Result<Vec<_>, Error>>()?;
	let max_mode = match context.policy.context.extended_mode {
		Some(ExtendedContextMode::Standard) => false,
		Some(ExtendedContextMode::Extended) => true,
		None => return Err(encoding_error("cursor_extended_context_mode_unknown")),
	};
	let run = CursorRunRequest {
		model_id: Str::from(target.wire_model.as_str()),
		max_mode,
		conversation_id: context
			.session
			.map(|session| Str::from(session.conversation.as_str())),
		checkpoint: None,
		root_prompts: roots.into_boxed_slice(),
		tools: tools.into_boxed_slice(),
		action: CursorRunAction::UserMessage {
			message_id: Str::from(context.request_id.as_str()),
			text:       user,
		},
	};
	let mut headers = Vec::with_capacity(5);
	for_each_public_header(CursorHeaderProfile::Run, |name, value| {
		headers.push(RequestHeader { name: Str::new_static(name), value: Str::new_static(value) });
	});
	Ok(EncodedRequest::new(
		OperationKind::Chat,
		RequestMethod::Post,
		endpoint_uri(context.route.endpoint.base_url.as_str(), RUN_PATH),
		headers.into_boxed_slice(),
		BodySource::bytes(encode_run_request(&run).map_err(inference_error)?),
		FramingProtocol::Connect,
		cursor_bounds(),
	))
}

fn cursor_message_text(context: &EncodeContext<'_>, content: &[ContentPart]) -> Result<Str, Error> {
	let mut text = String::new();
	for part in content {
		match part {
			ContentPart::Text { text: part, proof: None } => text.push_str(part.as_str()),
			ContentPart::Text { proof: Some(proof), .. }
			| ContentPart::Reasoning { proof: Some(proof), .. }
			| ContentPart::ToolCall { proof: Some(proof), .. } => {
				let target = context
					.target
					.ok_or_else(|| encoding_error("cursor_continuation_proof_requires_wire_target"))?;
				if proof.provider != context.route.provider || proof.codec != target.codec {
					return Err(encoding_error("provider_proof_scope_mismatch"));
				}
				return Err(encoding_error("cursor_continuation_proof_requires_checkpoint_reseed"));
			},
			ContentPart::Reasoning { .. }
			| ContentPart::Image(_)
			| ContentPart::Audio(_)
			| ContentPart::Document(_)
			| ContentPart::ToolCall { .. }
			| ContentPart::ToolResult { .. }
			| ContentPart::CachePoint(_) => {
				return Err(encoding_error("cursor_message_part_not_losslessly_projectable"));
			},
		}
	}
	Ok(Str::from(text))
}

fn reject_unprojected_chat_options(request: &ChatRequest) -> Result<(), Error> {
	if !request.hosted_tools.is_empty()
		|| !matches!(request.tool_choice, Setting::Unset)
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
		return Err(encoding_error("cursor_chat_option_not_losslessly_projectable"));
	}
	Ok(())
}

fn endpoint_uri(base: &str, path: &str) -> Str {
	let mut uri = String::with_capacity(base.len() + path.len());
	uri.push_str(base.trim_end_matches('/'));
	uri.push_str(path);
	Str::from(uri)
}

const fn cursor_bounds() -> SizeBounds {
	SizeBounds {
		request_body: MAX_MESSAGE_BYTES as u64,
		frame:        MAX_MESSAGE_BYTES as u64,
		response:     256 * 1024 * 1024,
	}
}

struct CursorWireDecoder {
	operation:      OperationKind,
	provider:       omp_llm_catalog::ProviderId,
	route:          omp_llm_catalog::RouteId,
	agent:          CursorDecoder,
	discovery_done: bool,
}

impl Decoder for CursorWireDecoder {
	fn push(&mut self, frame: Frame, emit: &mut dyn FnMut(RawEvent)) -> Result<(), Error> {
		match self.operation {
			OperationKind::Chat => {
				let Frame::Connect(envelope) = frame else {
					return Err(self.attach(encoding_error("cursor_chat_expected_connect_frame")));
				};
				if envelope.is_compressed() {
					return Err(self.attach(encoding_error("cursor_compressed_connect_not_supported")));
				}
				let events = match envelope.kind {
					ConnectEnvelopeKind::Message => self.agent.push_payload(envelope.payload),
					ConnectEnvelopeKind::EndStream => self.agent.push_end_stream(&envelope.payload),
				}
				.map_err(|error| self.attach(inference_error(error)))?;
				for event in events {
					emit(cursor_raw_event(event));
				}
				Ok(())
			},
			OperationKind::DiscoverModels => {
				if self.discovery_done {
					return Err(self.attach(encoding_error("cursor_discovery_response_repeated")));
				}
				let bytes = match frame {
					Frame::Raw(bytes) => bytes,
					Frame::Connect(envelope)
						if envelope.kind == ConnectEnvelopeKind::Message && !envelope.is_compressed() =>
					{
						envelope.payload
					},
					_ => return Err(self.attach(encoding_error("cursor_discovery_expected_protobuf"))),
				};
				let models = decode_discovery_response(&bytes)
					.map_err(|error| self.attach(inference_error(error)))?;
				let rows = models
					.into_iter()
					.map(|model| DiscoveredModel {
						provider:              self.provider.clone(),
						route:                 self.route.clone(),
						wire_model:            WireModelId::new(model.id),
						aliases:               model
							.aliases
							.into_vec()
							.into_iter()
							.map(WireModelId::new)
							.collect(),
						display_name:          Some(model.name),
						declared_family:       None,
						declared_operations:   OperationBits::for_kind(OperationKind::Chat),
						declared_capabilities: Some(discovered_capabilities(model.reasoning)),
						declared_limits:       None,
						extended_context_mode: Some(ExtendedContextMode::from_enabled(model.max_mode)),
						availability:          None,
						source:                Str::new_static("cursor_get_usable_models"),
						observed_at_ms:        None,
						updated_at_ms:         None,
						deprecated:            None,
					})
					.collect();
				emit(RawEvent::DiscoveredModels { rows, next_cursor: None });
				self.discovery_done = true;
				Ok(())
			},
			_ => Err(self.attach(encoding_error("cursor_operation_not_supported"))),
		}
	}

	fn finish(&mut self, _emit: &mut dyn FnMut(RawEvent)) -> Result<(), Error> {
		match self.operation {
			OperationKind::Chat => self
				.agent
				.finish()
				.map_err(|error| self.attach(inference_error(error))),
			OperationKind::DiscoverModels if self.discovery_done => Ok(()),
			OperationKind::DiscoverModels => {
				Err(self.attach(encoding_error("cursor_discovery_response_missing")))
			},
			_ => Err(self.attach(encoding_error("cursor_operation_not_supported"))),
		}
	}
}
fn discovered_capabilities(reasoning: bool) -> ModelCapabilities {
	ModelCapabilities {
		operations:    OperationBits::for_kind(OperationKind::Chat),
		chat:          Some(ChatCapabilities {
			roles:             unknown_availability(),
			mid_session_roles: unknown_availability(),
			tools:             unknown_availability(),
			structured_output: unknown_availability(),
			grammar:           unknown_availability(),
			text_verbosity:    unknown_availability(),
			reasoning:         if reasoning {
				Availability::Native(ReasoningCapabilities {
					features:              ReasoningFeatureBits::VISIBLE,
					efforts:               Box::new([]),
					minimum_budget_tokens: None,
					maximum_budget_tokens: None,
				})
			} else {
				Availability::Unsupported
			},
			input_modalities:  unknown_availability(),
			hosted_tools:      unknown_availability(),
			prompt_caching:    unknown_availability(),
			service_tiers:     unknown_availability(),
			sampling:          unknown_availability(),
			safety:            unknown_availability(),
			determinism:       unknown_availability(),
			server_state:      unknown_availability(),
			logprobs:          unknown_availability(),
		}),
		embeddings:    None,
		image:         None,
		video:         None,
		speech:        None,
		transcription: None,
		realtime:      None,
		search:        None,
		tokenization:  None,
	}
}

const fn unknown_availability<T>() -> Availability<T> {
	Availability::Unknown
}

impl CursorWireDecoder {
	fn attach(&self, mut error: Error) -> Error {
		error.provider = Some(self.provider.clone());
		error.route = Some(self.route.clone());
		error
	}
}

fn cursor_raw_event(event: CursorEvent) -> RawEvent {
	match event {
		CursorEvent::Chat(event) => RawEvent::Chat(event),
		CursorEvent::ToolCallComplete { index, id, name, arguments } => RawEvent::ToolCallComplete {
			index,
			call: UnvalidatedToolCall { id, name, input_kind: ToolInputKind::Json, arguments },
		},
		CursorEvent::ShellInvoke(invocation) => {
			RawEvent::Control(ProviderControlEvent::ShellInvoke {
				invocation: Str::from(invocation.id.to_string()),
				exec:       (!invocation.exec_id.is_empty()).then_some(invocation.exec_id),
				call:       invocation.call_id,
				command:    invocation.command,
				cwd:        (!invocation.working_directory.is_empty())
					.then_some(invocation.working_directory),
				timeout_ms: (invocation.timeout_ms != 0).then_some(invocation.timeout_ms as u64),
				streaming:  invocation.streaming,
			})
		},
		CursorEvent::InvokeCancel { id } => {
			RawEvent::Control(ProviderControlEvent::Cancel { call: ToolCallId::from(id.to_string()) })
		},
		CursorEvent::Checkpoint { data } => {
			RawEvent::ProviderState(ProviderStateEvent::Checkpoint { id: None, data })
		},
		CursorEvent::InteractionQuery { id, query } => {
			let kind = interaction_query_kind(&query);
			let payload =
				Bytes::from(wire::InteractionQuery { id, query: Some(query) }.encode_to_vec());
			RawEvent::Control(ProviderControlEvent::InteractionQuery { id, kind, payload })
		},
		CursorEvent::Completion { reason, blocks, usage } => {
			RawEvent::Completion(RawCompletion { reason, blocks, usage })
		},
	}
}

fn interaction_query_kind(query: &wire::interaction_query::Query) -> Str {
	use wire::interaction_query::Query;
	Str::new_static(match query {
		Query::WebSearchRequestQuery(_) => "web_search",
		Query::AskQuestionInteractionQuery(_) => "ask_question",
		Query::SwitchModeRequestQuery(_) => "switch_mode",
		Query::ExaSearchRequestQuery(_) => "exa_search",
		Query::ExaFetchRequestQuery(_) => "exa_fetch",
		Query::CreatePlanRequestQuery(_) => "create_plan",
		Query::SetupVmEnvironmentArgs(_) => "setup_vm_environment",
	})
}

fn inference_error(error: CursorProtocolError) -> Error {
	let kind = match error.kind {
		CursorErrorKind::Malformed | CursorErrorKind::Truncated | CursorErrorKind::AfterTerminal => {
			ErrorKind::StreamCorruption
		},
		CursorErrorKind::Cancelled => ErrorKind::Cancelled,
		CursorErrorKind::Authentication => ErrorKind::Authentication,
		CursorErrorKind::Upstream => ErrorKind::Protocol,
		CursorErrorKind::ContextOverflow => ErrorKind::ContextOverflow,
		CursorErrorKind::Unsupported => ErrorKind::CapabilityMismatch,
	};
	let mut inference =
		Error::new(kind, ErrorPhase::Streaming, RetryAction::Never, ExecutionReceipt::default());
	inference.status = error.status;
	inference.committed = error.committed;
	inference.code = Some(error.reason.clone());
	inference.detail = Some(ErrorDetail::Protocol { reason: ReasonId(error.reason) });
	inference
}

fn encoding_error(reason: &'static str) -> Error {
	let reason = Str::new_static(reason);
	let mut error = Error::new(
		ErrorKind::CapabilityMismatch,
		ErrorPhase::Encoding,
		RetryAction::Never,
		ExecutionReceipt::default(),
	);
	error.code = Some(reason.clone());
	error.detail = Some(ErrorDetail::Protocol { reason: ReasonId(reason) });
	error
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::transport::{ConnectDecoder as ConnectFramer, IncrementalFramer as _};

	const FIXTURES: &str =
		concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/llm-oracle/agent-protocols/cursor/");

	fn fixture(name: &str) -> Vec<u8> {
		std::fs::read(format!("{FIXTURES}{name}")).expect("Cursor oracle fixture")
	}

	fn update(message: wire::interaction_update::Message) -> Bytes {
		Bytes::from(
			wire::AgentServerMessage {
				message: Some(wire::agent_server_message::Message::InteractionUpdate(
					wire::InteractionUpdate { message: Some(message) },
				)),
			}
			.encode_to_vec(),
		)
	}

	#[test]
	fn descriptor_pins_service_drift_and_binding_numbers() {
		let descriptors = descriptor_set().expect("checked-in descriptor decodes");
		let file = descriptors
			.file
			.iter()
			.find(|file| file.package.as_deref() == Some("agent.v1"))
			.expect("agent.v1 descriptor");
		let service = file
			.service
			.iter()
			.find(|service| service.name.as_deref() == Some("AgentService"))
			.expect("AgentService descriptor");
		let run = service
			.method
			.iter()
			.find(|method| method.name.as_deref() == Some("Run"))
			.expect("Run method");
		assert_eq!(run.input_type.as_deref(), Some(".agent.v1.AgentClientMessage"));
		assert_eq!(run.output_type.as_deref(), Some(".agent.v1.AgentServerMessage"));
		assert!(!run.client_streaming.unwrap_or(false), "known source descriptor drift pin");
		assert!(!run.server_streaming.unwrap_or(false), "known source descriptor drift pin");

		let interaction = file
			.message_type
			.iter()
			.find(|message| message.name.as_deref() == Some("InteractionUpdate"))
			.expect("InteractionUpdate descriptor");
		let field = |name: &str| {
			interaction
				.field
				.iter()
				.find(|field| field.name.as_deref() == Some(name))
				.and_then(|field| field.number)
		};
		assert_eq!(field("text_delta"), Some(1));
		assert_eq!(field("partial_tool_call"), Some(7));
		assert_eq!(field("token_delta"), Some(8));
		assert_eq!(field("turn_ended"), Some(14));
		assert_eq!(field("tool_call_delta"), Some(15));
	}

	#[test]
	fn discovery_replays_raw_and_connect_fixtures_without_model_heuristics() {
		#[derive(serde::Deserialize)]
		struct Expected {
			models: Vec<ExpectedModel>,
		}
		#[derive(serde::Deserialize)]
		struct ExpectedModel {
			id:        String,
			name:      String,
			reasoning: bool,
			max_mode:  bool,
		}

		let expected: Expected = serde_json::from_slice(&fixture("discovery.expected.json"))
			.expect("typed discovery expectation");
		let raw =
			decode_discovery_response(&fixture("discovery.response.raw.bin")).expect("raw discovery");
		let framed = decode_discovery_response(&fixture("discovery.response.connect.bin"))
			.expect("Connect discovery");
		assert_eq!(raw, framed);
		assert_eq!(raw.len(), expected.models.len());
		for (actual, expected) in raw.iter().zip(expected.models) {
			assert_eq!(actual.id.as_str(), expected.id);
			assert_eq!(actual.name.as_str(), expected.name);
			assert_eq!(actual.reasoning, expected.reasoning);
			assert_eq!(actual.max_mode, expected.max_mode);
			assert!(actual.aliases.is_empty());
		}
		assert_eq!(encode_discovery_request(&[]), Bytes::from(fixture("discovery.request.bin")));
	}

	#[test]
	fn recorded_tool_stream_projects_incrementally_and_authorizes_nothing() {
		let tool = wire::ToolCall {
			tool_call_id: Some("call-read".to_owned()),
			tool:         Some(wire::tool_call::Tool::ReadToolCall(wire::ReadToolCall::default())),
		};
		let payloads = [
			update(wire::interaction_update::Message::ThinkingDelta(wire::ThinkingDeltaUpdate {
				text: "Inspect first.".to_owned(),
			})),
			update(wire::interaction_update::Message::ToolCallStarted(wire::ToolCallStartedUpdate {
				call_id: "call-read".to_owned(),
				tool_call: Some(tool.clone()),
				..Default::default()
			})),
			update(wire::interaction_update::Message::PartialToolCall(wire::PartialToolCallUpdate {
				call_id: "call-read".to_owned(),
				tool_call: Some(tool.clone()),
				args_text_delta: "{\"pa".to_owned(),
				..Default::default()
			})),
			update(wire::interaction_update::Message::PartialToolCall(wire::PartialToolCallUpdate {
				call_id: "call-read".to_owned(),
				tool_call: Some(tool),
				args_text_delta: "th\":\"package.json\"}".to_owned(),
				..Default::default()
			})),
			update(wire::interaction_update::Message::ToolCallCompleted(
				wire::ToolCallCompletedUpdate { call_id: "call-read".to_owned(), ..Default::default() },
			)),
			update(wire::interaction_update::Message::TextDelta(wire::TextDeltaUpdate {
				text: "Done.".to_owned(),
			})),
			update(wire::interaction_update::Message::TokenDelta(wire::TokenDeltaUpdate {
				tokens: 8,
			})),
			update(wire::interaction_update::Message::TurnEnded(wire::TurnEndedUpdate {})),
		];
		let mut decoder = CursorDecoder::default();
		let events: Vec<_> = payloads
			.into_iter()
			.flat_map(|payload| decoder.push_payload(payload).expect("recorded protobuf"))
			.collect();
		assert!(events.iter().any(|event| matches!(
			event,
			CursorEvent::Chat(ChatEvent::ThinkingDelta { text, .. }) if text == "Inspect first."
		)));
		assert!(events.iter().any(|event| matches!(
			event,
			CursorEvent::ToolCallComplete { id, name, arguments, .. }
				if id.as_str() == "call-read"
					&& name == "read"
					&& arguments.as_ref() == br#"{"path":"package.json"}"#
		)));
		assert!(
			!events
				.iter()
				.any(|event| matches!(event, CursorEvent::Chat(ChatEvent::ToolCallReady { .. }))),
			"codec completion is not schema-validation authorization"
		);
		assert!(events.iter().any(|event| matches!(
			event,
			CursorEvent::Completion {
				reason: FinishReason::ToolCalls,
				usage,
				..
			} if usage.output_tokens == 8
		)));

		#[derive(serde::Deserialize)]
		#[serde(rename_all = "camelCase")]
		struct RecordedLine {
			frame:   String,
			payload: RecordedPayload,
		}
		#[derive(serde::Deserialize)]
		#[serde(rename_all = "camelCase")]
		struct RecordedPayload {
			thinking: Option<RecordedText>,
			tool_call_started: Option<RecordedTool>,
			tool_call_args_text_delta: Option<RecordedArgs>,
			tool_call_completed: Option<RecordedTool>,
			text_delta: Option<RecordedText>,
			usage: Option<RecordedUsage>,
			done: Option<RecordedDone>,
		}
		#[derive(serde::Deserialize)]
		struct RecordedText {
			text: String,
		}
		#[derive(serde::Deserialize)]
		struct RecordedTool {
			id:   String,
			name: String,
		}
		#[derive(serde::Deserialize)]
		#[serde(rename_all = "camelCase")]
		struct RecordedArgs {
			id:        String,
			args_text: String,
		}
		#[derive(serde::Deserialize)]
		#[serde(rename_all = "camelCase")]
		struct RecordedUsage {
			input_tokens:        u64,
			output_tokens:       u64,
			cached_input_tokens: u64,
		}
		#[derive(serde::Deserialize)]
		#[serde(rename_all = "camelCase")]
		struct RecordedDone {
			stop_reason: String,
		}

		let stream = String::from_utf8(fixture("stream.tool_args.jsonl")).expect("UTF-8 JSONL");
		let records: Vec<RecordedLine> = stream
			.lines()
			.map(|line| serde_json::from_str(line).expect("typed Cursor stream record"))
			.collect();
		assert_eq!(records.len(), 8);
		assert!(
			records
				.iter()
				.all(|record| record.frame == "interaction_update")
		);
		assert_eq!(records[0].payload.thinking.as_ref().expect("thinking").text, "Inspect first.");
		assert_eq!(
			records[1]
				.payload
				.tool_call_started
				.as_ref()
				.expect("tool")
				.id,
			"call-read"
		);
		assert_eq!(
			records[1]
				.payload
				.tool_call_started
				.as_ref()
				.expect("tool")
				.name,
			"read"
		);
		assert_eq!(
			records[2]
				.payload
				.tool_call_args_text_delta
				.as_ref()
				.expect("args")
				.id,
			"call-read"
		);
		assert_eq!(
			records[2..=3]
				.iter()
				.map(|record| record
					.payload
					.tool_call_args_text_delta
					.as_ref()
					.expect("args")
					.args_text
					.as_str())
				.collect::<String>(),
			r#"{"path":"package.json"}"#
		);
		assert_eq!(
			records[4]
				.payload
				.tool_call_completed
				.as_ref()
				.expect("tool")
				.name,
			"read"
		);
		assert_eq!(records[5].payload.text_delta.as_ref().expect("text").text, "Done.");
		let usage = records[6].payload.usage.as_ref().expect("usage");
		assert_eq!((usage.input_tokens, usage.output_tokens, usage.cached_input_tokens), (21, 8, 13));
		assert_eq!(records[7].payload.done.as_ref().expect("done").stop_reason, "tool_use");
	}

	#[test]
	fn connect_terminal_malformed_overflow_and_late_frames_are_typed() {
		#[derive(serde::Deserialize)]
		struct TerminalCases {
			cases: Vec<TerminalCase>,
		}
		#[derive(serde::Deserialize)]
		struct TerminalCase {
			id: String,
			path: Option<String>,
			chunks_hex: Option<Vec<String>>,
			expected_payloads_hex: Option<Vec<String>>,
			expected_buffered_bytes: Option<usize>,
		}

		let cases: TerminalCases =
			serde_json::from_slice(&fixture("connect.terminal_expectations.json"))
				.expect("typed terminal fixture");
		for case in cases.cases {
			let chunks = case.chunks_hex.map_or_else(
				|| vec![fixture(case.path.as_deref().expect("path-backed terminal case"))],
				|hex| hex.into_iter().map(|chunk| decode_hex(&chunk)).collect(),
			);
			let mut framing = ConnectFramer::new();
			let mut envelopes = Vec::new();
			for chunk in chunks {
				envelopes.extend(framing.push(Bytes::from(chunk)).expect("framing fixture"));
			}
			if let Some(buffered) = case.expected_buffered_bytes {
				assert_eq!(framing.buffered_len(), buffered, "{}", case.id);
				continue;
			}
			let mut decoder = CursorDecoder::default();
			let mut messages = Vec::new();
			for envelope in envelopes {
				match envelope.kind {
					ConnectEnvelopeKind::Message => messages.push(envelope.payload),
					ConnectEnvelopeKind::EndStream => {
						let result = decoder.push_end_stream(&envelope.payload);
						if case.id == "error_end_stream" {
							assert_eq!(result.expect_err("error trailer").kind, CursorErrorKind::Upstream);
						} else {
							result.expect("normal trailer");
						}
					},
				}
			}
			let expected = case
				.expected_payloads_hex
				.unwrap_or_default()
				.into_iter()
				.map(|payload| Bytes::from(decode_hex(&payload)))
				.collect::<Vec<_>>();
			assert_eq!(messages, expected, "{}", case.id);
		}

		let mut malformed = CursorDecoder::default();
		assert_eq!(
			malformed
				.push_payload(Bytes::from_static(b"\xff"))
				.expect_err("malformed")
				.kind,
			CursorErrorKind::Malformed
		);
		assert_eq!(
			malformed
				.push_end_stream(br#"{"error":{"code":"context_length_exceeded"}}"#)
				.expect_err("overflow")
				.kind,
			CursorErrorKind::ContextOverflow
		);

		let mut terminal = CursorDecoder::default();
		terminal
			.push_payload(update(wire::interaction_update::Message::TurnEnded(
				wire::TurnEndedUpdate {},
			)))
			.expect("turn end");
		assert_eq!(
			terminal
				.push_payload(update(wire::interaction_update::Message::Heartbeat(
					wire::HeartbeatUpdate {},
				)))
				.expect_err("late frame")
				.kind,
			CursorErrorKind::AfterTerminal
		);
	}

	#[test]
	fn checkpoint_reconnect_cancel_and_shell_paths_remain_correlated() {
		let checkpoint = wire::ConversationStateStructure {
			pending_tool_calls: vec!["pending".to_owned()],
			self_summary_count: 3,
			..Default::default()
		};
		let server = wire::AgentServerMessage {
			message: Some(wire::agent_server_message::Message::ConversationCheckpointUpdate(
				checkpoint.clone(),
			)),
		};
		let mut decoder = CursorDecoder::default();
		let events = decoder
			.push_payload(Bytes::from(server.encode_to_vec()))
			.expect("checkpoint");
		let CursorEvent::Checkpoint { data } = &events[0] else {
			panic!("checkpoint event")
		};
		assert_eq!(
			wire::ConversationStateStructure::decode(data.clone()).expect("checkpoint protobuf"),
			checkpoint
		);
		let reconnect =
			encode_reconnect_request(&CursorReconnectRequest { request_id: Str::from("request-7") });
		assert_eq!(
			wire::BidiRequestId::decode(reconnect)
				.expect("reconnect request")
				.request_id,
			"request-7"
		);

		let abort = wire::AgentServerMessage {
			message: Some(wire::agent_server_message::Message::ExecServerControlMessage(
				wire::ExecServerControlMessage {
					message: Some(wire::exec_server_control_message::Message::Abort(
						wire::ExecServerAbort { id: 17 },
					)),
				},
			)),
		};
		assert!(matches!(
			CursorDecoder::default()
				.push_payload(Bytes::from(abort.encode_to_vec()))
				.expect("abort")
				.as_slice(),
			[CursorEvent::InvokeCancel { id: 17 }]
		));

		#[derive(serde::Deserialize)]
		struct CancelFixture {
			input:    CancelInput,
			expected: CancelExpected,
		}
		#[derive(serde::Deserialize)]
		struct CancelInput {
			server_abort_id: u32,
		}
		#[derive(serde::Deserialize)]
		struct CancelExpected {
			executor_future_dropped: bool,
			no_completion_frames:    bool,
		}
		let cancel_fixture: CancelFixture =
			serde_json::from_slice(&fixture("connect.cancel.json")).expect("typed cancel fixture");
		assert_eq!(cancel_fixture.input.server_abort_id, 17);
		assert!(
			cancel_fixture.expected.executor_future_dropped
				&& cancel_fixture.expected.no_completion_frames
		);
		let mut cancelled = CursorDecoder::default();
		cancelled.cancel();
		assert_eq!(
			cancelled
				.push_payload(update(wire::interaction_update::Message::Heartbeat(
					wire::HeartbeatUpdate {},
				)))
				.expect_err("cancel suppresses late frames")
				.kind,
			CursorErrorKind::Cancelled
		);
		cancelled.finish().expect("cancel is terminal");

		let invocation = CursorShellInvocation {
			id:                17,
			exec_id:           Str::from("exec-17"),
			call_id:           ToolCallId::from("call-shell"),
			command:           Str::from("printf colours"),
			working_directory: Str::from("/work/project"),
			timeout_ms:        750,
			streaming:         true,
		};
		assert!(matches!(
			shell_start(&invocation).message,
			Some(wire::agent_client_message::Message::ExecClientMessage(wire::ExecClientMessage {
				message: Some(wire::exec_client_message::Message::ShellStream(wire::ShellStream {
					event: Some(wire::shell_stream::Event::Start(_)),
				})),
				..
			}))
		));
		for (frame, expected) in [
			(shell_stdout(&invocation, "plain"), ("stdout", "plain")),
			(shell_stderr(&invocation, "warning\n"), ("stderr", "warning\n")),
			(
				shell_stdout(&invocation, "\u{1b}[31mred\u{1b}[0m\n"),
				("stdout", "\u{1b}[31mred\u{1b}[0m\n"),
			),
		] {
			let Some(wire::agent_client_message::Message::ExecClientMessage(exec)) = frame.message
			else {
				panic!("shell output exec frame")
			};
			let Some(wire::exec_client_message::Message::ShellStream(stream)) = exec.message else {
				panic!("shell stream frame")
			};
			let (channel, data) = match stream.event.expect("shell event") {
				wire::shell_stream::Event::Stdout(stdout) => ("stdout", stdout.data),
				wire::shell_stream::Event::Stderr(stderr) => ("stderr", stderr.data),
				_ => panic!("unexpected shell output event"),
			};
			assert_eq!((channel, data.as_str()), expected);
		}
		for completion in [
			CursorShellCompletion::Exited {
				stdout:                  Str::from("committed output"),
				stderr:                  Str::default(),
				local_execution_time_ms: 41,
			},
			CursorShellCompletion::Failed {
				code:                    23,
				stdout:                  Str::default(),
				stderr:                  Str::default(),
				local_execution_time_ms: 41,
			},
			CursorShellCompletion::Rejected {
				reason:      Str::from("policy detail"),
				is_readonly: true,
			},
			CursorShellCompletion::PermissionDenied {
				reason:      Str::from("policy detail"),
				is_readonly: true,
			},
			CursorShellCompletion::TimedOut { timeout_ms: 750 },
		] {
			let frames = shell_completion_frames(&invocation, &completion);
			assert!(matches!(
				frames.last().and_then(|frame| frame.message.as_ref()),
				Some(wire::agent_client_message::Message::ExecClientControlMessage(
					wire::ExecClientControlMessage {
						message: Some(wire::exec_client_control_message::Message::StreamClose(
							wire::ExecClientStreamClose { id: 17 }
						)),
					}
				))
			));
		}
		#[derive(serde::Deserialize)]
		struct ShellFixture {
			case:    String,
			context: ShellContextFixture,
		}
		#[derive(serde::Deserialize)]
		struct ShellContextFixture {
			id:      u32,
			exec_id: String,
			command: String,
			cwd:     String,
		}
		let shell: ShellFixture = serde_json::from_slice(&fixture("connect.shell_stream.json"))
			.expect("typed shell fixture");
		assert_eq!(shell.case, "stream_order");
		assert_eq!(
			(
				shell.context.id,
				shell.context.exec_id.as_str(),
				shell.context.command.as_str(),
				shell.context.cwd.as_str()
			),
			(17, "exec-17", "printf colours", "/work/project")
		);

		#[derive(serde::Deserialize)]
		struct StatusFixture {
			cases: Vec<StatusCase>,
		}
		#[derive(serde::Deserialize)]
		struct StatusCase {
			outcome: String,
		}
		let statuses: StatusFixture =
			serde_json::from_slice(&fixture("connect.statuses.json")).expect("typed status fixture");
		assert_eq!(
			statuses
				.cases
				.into_iter()
				.map(|case| case.outcome)
				.collect::<Vec<_>>(),
			["exited", "failed", "rejected", "denied", "timeout"]
		);

		#[derive(serde::Deserialize)]
		struct DeadlineFixture {
			input:    DeadlineInput,
			expected: DeadlineExpected,
		}
		#[derive(serde::Deserialize)]
		struct DeadlineInput {
			timeout_ms: u32,
		}
		#[derive(serde::Deserialize)]
		struct DeadlineExpected {
			executor_future_dropped: bool,
			no_completion_frames:    bool,
		}
		let deadline: DeadlineFixture =
			serde_json::from_slice(&fixture("connect.deadline.json")).expect("typed deadline fixture");
		assert_eq!(deadline.input.timeout_ms, 1);
		assert!(deadline.expected.executor_future_dropped && deadline.expected.no_completion_frames);
	}

	#[test]
	fn request_headers_error_and_expectation_fixtures_are_typed() {
		#[derive(serde::Deserialize)]
		struct HeaderFixture {
			request: HeaderRequest,
		}
		#[derive(serde::Deserialize)]
		struct HeaderRequest {
			method:  String,
			url:     String,
			headers: BTreeMap<String, String>,
		}
		let headers: HeaderFixture =
			serde_json::from_slice(&fixture("chat.headers.json")).expect("typed header fixture");
		assert_eq!(headers.request.method, "POST");
		assert!(headers.request.url.ends_with(RUN_PATH));
		assert_eq!(headers.request.headers["x-cursor-client-version"], CLIENT_VERSION);
		assert_eq!(headers.request.headers["content-type"], "application/connect+proto");

		#[derive(serde::Deserialize)]
		struct RequestFixture {
			canonical_intent: CanonicalIntent,
		}
		#[derive(serde::Deserialize)]
		struct CanonicalIntent {
			model: String,
			tools: Vec<FixtureTool>,
		}
		#[derive(serde::Deserialize)]
		struct FixtureTool {
			name: String,
		}
		let request: RequestFixture =
			serde_json::from_slice(&fixture("request.tool_call.json")).expect("typed request fixture");
		let encoded = encode_run_request(&CursorRunRequest {
			model_id:        Str::from(request.canonical_intent.model.as_str()),
			max_mode:        false,
			conversation_id: None,
			checkpoint:      None,
			root_prompts:    Box::new([]),
			tools:           request
				.canonical_intent
				.tools
				.into_iter()
				.map(|tool| CursorToolDefinition {
					name:              Str::from(tool.name),
					description:       None,
					input_schema_json: Str::from(
						r#"{"type":"object","properties":{"path":{"type":"string"}}}"#,
					),
				})
				.collect(),
			action:          CursorRunAction::UserMessage {
				message_id: Str::from("request-fixture"),
				text:       Str::from("Inspect package.json"),
			},
		})
		.expect("typed run request");
		let run = wire::AgentClientMessage::decode(encoded.slice(5..)).expect("framed run request");
		let Some(wire::agent_client_message::Message::RunRequest(run)) = run.message else {
			panic!("run request")
		};
		assert_eq!(run.model_details.as_ref().expect("model").model_id, "cursor-composer-2.5");
		assert!(
			!run
				.model_details
				.as_ref()
				.expect("model")
				.max_mode
				.expect("explicit ordinary mode")
		);
		assert!(
			!run
				.requested_model
				.as_ref()
				.expect("requested model")
				.max_mode
		);
		assert_eq!(run.mcp_tools.as_ref().expect("tools").mcp_tools[0].name, "read");

		#[derive(serde::Deserialize)]
		struct ToolExpectation {
			outcome: ToolOutcome,
		}
		#[derive(serde::Deserialize)]
		struct ToolOutcome {
			text:       String,
			thinking:   String,
			tool_calls: Vec<ToolOutcomeCall>,
		}
		#[derive(serde::Deserialize)]
		struct ToolOutcomeCall {
			id:   String,
			name: String,
		}
		let expectation: ToolExpectation =
			serde_json::from_slice(&fixture("expected.tool_args.json"))
				.expect("typed tool expectation");
		assert_eq!(
			(expectation.outcome.text.as_str(), expectation.outcome.thinking.as_str()),
			("Done.", "Inspect first.")
		);
		assert_eq!(
			(
				expectation.outcome.tool_calls[0].id.as_str(),
				expectation.outcome.tool_calls[0].name.as_str()
			),
			("call-read", "read")
		);

		#[derive(serde::Deserialize)]
		struct DecodeContract {
			framing:          FramingContract,
			state_invariants: StateInvariants,
		}
		#[derive(serde::Deserialize)]
		struct FramingContract {
			header_bytes:          usize,
			data_flags:            u8,
			end_stream_mask:       u8,
			incremental_buffering: bool,
		}
		#[derive(serde::Deserialize)]
		struct StateInvariants {
			token_delta_is_additive_and_saturating: bool,
			turn_end_stop_reason:                   String,
		}
		let contract: DecodeContract =
			serde_json::from_slice(&fixture("connect.decode_contract.json"))
				.expect("typed decode contract");
		assert_eq!(
			(
				contract.framing.header_bytes,
				contract.framing.data_flags,
				contract.framing.end_stream_mask
			),
			(5, 0, 2)
		);
		assert!(
			contract.framing.incremental_buffering
				&& contract
					.state_invariants
					.token_delta_is_additive_and_saturating
		);
		assert!(
			contract
				.state_invariants
				.turn_end_stop_reason
				.contains("tool_use")
		);

		#[derive(serde::Deserialize)]
		struct ErrorFixture {
			cases: Vec<ErrorCase>,
		}
		#[derive(serde::Deserialize)]
		struct ErrorCase {
			input:    String,
			expected: ErrorExpected,
		}
		#[derive(serde::Deserialize)]
		struct ErrorExpected {
			error_kind: Option<String>,
		}
		let errors: ErrorFixture =
			serde_json::from_slice(&fixture("errors.json")).expect("typed error fixture");
		assert!(errors.cases.iter().any(|case| {
			case.input == "authentication_failure"
				&& case.expected.error_kind.as_deref() == Some("upstream")
		}));
		assert_eq!(
			classify_http_status(401)
				.expect("authentication status")
				.kind,
			CursorErrorKind::Authentication
		);
		assert_eq!(
			classify_http_status(500).expect("upstream status").kind,
			CursorErrorKind::Upstream
		);

		#[derive(serde::Deserialize)]
		struct DiscoveryHttpFixture {
			request: DiscoveryHttpRequest,
		}
		#[derive(serde::Deserialize)]
		struct DiscoveryHttpRequest {
			method:    String,
			url:       String,
			headers:   BTreeMap<String, String>,
			body_path: String,
		}
		let discovery: DiscoveryHttpFixture = serde_json::from_slice(&fixture("discovery.http.json"))
			.expect("typed discovery HTTP fixture");
		assert_eq!(discovery.request.method, "POST");
		assert!(discovery.request.url.ends_with(DISCOVERY_PATH));
		assert_eq!(discovery.request.headers["content-type"], "application/proto");
		assert_eq!(discovery.request.body_path, "discovery.request.bin");
	}

	fn decode_hex(input: &str) -> Vec<u8> {
		omp_core::encoding::hex::decode(input)
			.into_vec()
			.expect("hex oracle fixture")
	}
}
