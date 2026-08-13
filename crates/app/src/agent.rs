//! Application-owned tool execution over the typed inference client.

use std::{future::Future, sync::Arc};

use bytes::Bytes;
use futures::StreamExt as _;
use omp_core::Str;
use omp_llm_inference::{
	answer::{Answer, ChatStream},
	call::{ChatRequest, ContentPart, Message, OpaqueJson, Role, ToolDefinition, ToolResultContent},
	client::Client,
	error::Error,
	event::{ChatEvent, Completion, ToolCall},
	plan::Planner,
};
use tower::Service;

/// Application-owned implementation of one tool exposed to a model.
pub trait OwnedToolHandler: Send + Sync {
	/// Future returned by this handler.
	type Execute<'a>: Future<Output = OwnedToolOutput> + Send + 'a
	where
		Self: 'a;

	/// Portable definition advertised to the model.
	fn definition(&self) -> &ToolDefinition;

	/// Executes one schema-validated canonical invocation.
	fn execute(&self, args_json: Bytes) -> Self::Execute<'_>;
}

/// Canonical application output produced by an owned tool.
#[derive(Clone, Debug)]
pub struct OwnedToolOutput {
	/// Opaque structured result returned to the model.
	pub result:   OpaqueJson,
	/// Whether execution failed.
	pub is_error: bool,
}

impl OwnedToolOutput {
	/// Creates a successful textual tool result.
	#[must_use]
	pub fn text(text: impl Into<Str>) -> Self {
		Self {
			result:   OpaqueJson::new(serde_json::Value::String(text.into().as_str().to_owned())),
			is_error: false,
		}
	}
}

/// Authoritative outcomes from one tool-using turn and its follow-up.
#[derive(Debug)]
pub struct OwnedToolLoopOutcome {
	/// Completion that authorized tool execution.
	pub tool_turn:      Completion,
	/// Canonical result appended between the two model turns.
	pub tool_result:    OwnedToolOutput,
	/// Completion of the clean follow-up turn.
	pub follow_up_turn: Completion,
}

/// Failure of the canonical two-turn owned-tool protocol.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum OwnedToolLoopError {
	/// The typed inference service rejected or failed the request.
	#[error(transparent)]
	Inference(#[from] Error),
	/// The canonical stream violated the one-tool loop contract.
	#[error("invalid owned-tool stream: {0}")]
	Protocol(&'static str),
	/// The model invoked a different tool from the one declared by this loop.
	#[error("model invoked undeclared tool `{0}`")]
	UndeclaredTool(Str),
	/// Tool arguments could not be serialized for the application handler.
	#[error("canonical tool arguments could not be serialized")]
	Arguments(#[source] serde_json::Error),
}

/// Runs one application-owned tool invocation and a clean follow-up turn.
///
/// Only [`ChatEvent::ToolCallReady`] authorizes execution. Partial call events
/// are display-only and are never forwarded to the handler.
pub async fn run_owned_tool_loop<S, P, H>(
	client: &mut Client<S, P>,
	mut request: ChatRequest,
	handler: &H,
) -> Result<OwnedToolLoopOutcome, OwnedToolLoopError>
where
	S: Service<omp_llm_inference::call::Call, Response = Answer, Error = Error>,
	P: Planner,
	H: OwnedToolHandler,
{
	request.tools = Arc::from([handler.definition().clone()]);
	let (tool_turn, call) = collect_tool_turn(client.execute(request.clone()).await?).await?;
	if call.name != handler.definition().name {
		return Err(OwnedToolLoopError::UndeclaredTool(call.name));
	}
	let args = Bytes::from(
		serde_json::to_vec(call.arguments.as_value()).map_err(OwnedToolLoopError::Arguments)?,
	);
	let output = handler.execute(args).await;

	let mut messages = request.messages.iter().cloned().collect::<Vec<_>>();
	messages.push(Message {
		role:    Role::Assistant,
		content: Arc::from([ContentPart::ToolCall {
			call:      call.id.clone(),
			name:      call.name.clone(),
			arguments: call.arguments.clone(),
			proof:     None,
		}]),
		name:    None,
	});
	messages.push(Message {
		role:    Role::Tool,
		content: Arc::from([ContentPart::ToolResult {
			call:     call.id,
			name:     Some(call.name),
			content:  Arc::from([ToolResultContent::Json(output.result.clone())]),
			is_error: output.is_error,
		}]),
		name:    None,
	});
	request.messages = messages.into();
	let follow_up_turn = collect_completion(client.execute(request).await?).await?;
	Ok(OwnedToolLoopOutcome { tool_turn, tool_result: output, follow_up_turn })
}

async fn collect_tool_turn(
	mut stream: ChatStream,
) -> Result<(Completion, ToolCall), OwnedToolLoopError> {
	let mut ready = None;
	let mut completion = None;
	while let Some(event) = stream.next().await {
		match event? {
			ChatEvent::ToolCallReady { call, .. } if ready.is_none() => ready = Some(call),
			ChatEvent::ToolCallReady { .. } => {
				return Err(OwnedToolLoopError::Protocol("turn authorized more than one tool call"));
			},
			ChatEvent::Completed(done) if completion.is_none() => completion = Some(done),
			ChatEvent::Completed(_) => {
				return Err(OwnedToolLoopError::Protocol("turn completed more than once"));
			},
			_ => {},
		}
	}
	Ok((
		completion.ok_or(OwnedToolLoopError::Protocol("tool turn ended without completion"))?,
		ready.ok_or(OwnedToolLoopError::Protocol("tool turn never authorized a tool call"))?,
	))
}

async fn collect_completion(mut stream: ChatStream) -> Result<Completion, OwnedToolLoopError> {
	let mut completion = None;
	while let Some(event) = stream.next().await {
		if let ChatEvent::Completed(done) = event? {
			if completion.replace(done).is_some() {
				return Err(OwnedToolLoopError::Protocol("follow-up completed more than once"));
			}
		}
	}
	completion.ok_or(OwnedToolLoopError::Protocol("follow-up ended without completion"))
}
