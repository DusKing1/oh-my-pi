//! Minimal agent-owned tool execution over the production [`Chat`] boundary.
//!
//! Provider and dialect details stay below [`Chat`]. This module consumes only
//! canonical turn events, executes one declared application tool, commits its
//! canonical result to history, and performs the follow-up turn through the
//! same production service.

use std::future::Future;

use bytes::{Bytes, BytesMut};
use futures::{Stream, StreamExt as _};
use omp_core::SmolStr;
use omp_llm_types::{
	Chat, ChatOutcome, ChatRequest, Item, ItemKind, MessageAttribution, Part, Props, StreamPartKind,
	ToolCall, ToolDef, ToolResult, TurnEvent, ids::CallId,
};

/// Application-owned implementation of one tool exposed to a model.
///
/// The definition is injected into both requests. Execution receives exactly
/// the streamed canonical JSON bytes; provider-specific call syntax never
/// crosses this boundary.
pub trait OwnedToolHandler: Send + Sync {
	/// Future returned by this handler.
	type Execute<'a>: Future<Output = OwnedToolOutput> + Send + 'a
	where
		Self: 'a;

	/// Portable definition advertised to the model.
	fn definition(&self) -> &ToolDef;

	/// Executes one canonical invocation.
	fn execute(&self, args_json: Bytes) -> Self::Execute<'_>;
}

/// Canonical application output produced by an owned tool.
#[derive(Clone, Debug, PartialEq)]
pub struct OwnedToolOutput {
	/// Ordered text or media returned by the tool.
	pub parts:    Vec<Part>,
	/// Whether execution failed.
	pub is_error: bool,
	/// Optional structured application detail.
	pub details:  Option<serde_json::Value>,
}

impl OwnedToolOutput {
	/// Creates a successful textual tool result.
	#[must_use]
	pub fn text(text: impl Into<SmolStr>) -> Self {
		Self { parts: vec![Part::Text(text.into())], is_error: false, details: None }
	}
}

/// Authoritative outcomes from one tool-using turn and its follow-up.
#[derive(Debug)]
pub struct OwnedToolLoopOutcome {
	/// Outcome that requested tool execution.
	pub tool_turn:      ChatOutcome,
	/// Canonical result appended between the two model turns.
	pub tool_result:    ToolResult,
	/// Clean follow-up model outcome.
	pub follow_up_turn: ChatOutcome,
}

/// Failure of the canonical two-turn owned-tool protocol.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum OwnedToolLoopError {
	/// The production chat service rejected the request before streaming.
	#[error(transparent)]
	Chat(#[from] omp_llm_types::Error),
	/// The turn ended with a canonical terminal error.
	#[error("turn failed: {0:?}")]
	Turn(omp_llm_types::TurnError),
	/// The canonical stream violated the one-tool loop contract.
	#[error("invalid owned-tool stream: {0}")]
	Protocol(&'static str),
	/// The streamed call identifier was not a nonzero canonical ULID.
	#[error("invalid canonical tool-call id `{0}`")]
	InvalidCallId(SmolStr),
	/// The model invoked a different tool from the one declared by this loop.
	#[error("model invoked undeclared tool `{0}`")]
	UndeclaredTool(SmolStr),
	/// Streamed argument deltas disagreed with the authoritative outcome.
	#[error("streamed tool arguments disagree with the committed outcome")]
	ArgumentMismatch,
}

#[derive(Debug)]
struct StreamedCall {
	id:        CallId,
	name:      SmolStr,
	args_json: Bytes,
}

#[derive(Debug)]
struct OpenCall {
	index: u32,
	id:    SmolStr,
	name:  SmolStr,
	args:  BytesMut,
}

/// Runs one application-owned tool invocation and a clean follow-up turn.
///
/// The original request is kept canonical. This function replaces its tool
/// inventory with the handler's single declared definition, observes argument
/// deltas without accumulated snapshots, executes exactly once, appends the
/// authoritative first outcome and matching [`ToolResult`] to the thread, then
/// submits the follow-up through the same [`Chat`] implementation.
pub async fn run_owned_tool_loop<H>(
	chat: &dyn Chat,
	mut request: ChatRequest,
	handler: &H,
) -> Result<OwnedToolLoopOutcome, OwnedToolLoopError>
where
	H: OwnedToolHandler,
{
	request.tools.clear();
	request.tools.push(handler.definition().clone());

	let mut first_stream = chat.turn(request.clone(), None).await?;
	let (tool_turn, streamed) = collect_tool_turn(&mut first_stream).await?;
	if streamed.name.as_str() != handler.definition().name.as_str() {
		return Err(OwnedToolLoopError::UndeclaredTool(streamed.name));
	}
	let committed = unique_committed_call(&tool_turn)?;
	if committed.id != streamed.id
		|| committed.name != streamed.name
		|| committed.args_json != streamed.args_json
	{
		return Err(OwnedToolLoopError::ArgumentMismatch);
	}

	let output = handler.execute(streamed.args_json).await;
	let result = ToolResult::builder()
		.call_id(committed.id)
		.name(committed.name.clone())
		.parts(output.parts)
		.is_error(output.is_error)
		.maybe_details(output.details)
		.maybe_attribution(Some(MessageAttribution::Agent))
		.maybe_pruned_at_ms(None)
		.maybe_useless(None)
		.maybe_provider_metadata(None)
		.build();

	request.thread.items.extend(tool_turn.output.clone());
	request.thread.items.push(
		Item::builder()
			.seq(0)
			.kind(ItemKind::ToolResult(result.clone()))
			.props(Props::default())
			.build(),
	);

	let mut follow_up_stream = chat.turn(request, None).await?;
	let follow_up_turn = collect_terminal_outcome(&mut follow_up_stream).await?;
	Ok(OwnedToolLoopOutcome { tool_turn, tool_result: result, follow_up_turn })
}

async fn collect_tool_turn<S>(
	stream: &mut S,
) -> Result<(ChatOutcome, StreamedCall), OwnedToolLoopError>
where
	S: Stream<Item = TurnEvent> + Unpin,
{
	let mut open: Option<OpenCall> = None;
	let mut completed: Option<StreamedCall> = None;
	let mut terminal = None;
	while let Some(event) = stream.next().await {
		if terminal.is_some() {
			return Err(OwnedToolLoopError::Protocol("event followed terminal event"));
		}
		match event {
			TurnEvent::PartStart {
				index,
				kind: StreamPartKind::ToolCall,
				tool_call_id,
				tool_name,
			} => {
				if open.is_some() || completed.is_some() {
					return Err(OwnedToolLoopError::Protocol("turn emitted more than one tool call"));
				}
				open =
					Some(OpenCall { index, id: tool_call_id, name: tool_name, args: BytesMut::new() });
			},
			TurnEvent::PartDelta { index, chunk }
				if open.as_ref().is_some_and(|call| call.index == index) =>
			{
				open
					.as_mut()
					.expect("guarded open call")
					.args
					.extend_from_slice(&chunk);
			},
			TurnEvent::PartEnd { index, .. }
				if open.as_ref().is_some_and(|call| call.index == index) =>
			{
				let call = open.take().expect("guarded open call");
				let id: CallId = call
					.id
					.parse()
					.map_err(|_| OwnedToolLoopError::InvalidCallId(call.id.clone()))?;
				if id.as_ulid().to_bytes() == [0; 16] {
					return Err(OwnedToolLoopError::InvalidCallId(call.id));
				}
				completed = Some(StreamedCall { id, name: call.name, args_json: call.args.freeze() });
			},
			TurnEvent::Outcome(outcome) => terminal = Some(Ok(outcome)),
			TurnEvent::Error(error) => terminal = Some(Err(error)),
			_ => {},
		}
	}
	let outcome = terminal
		.ok_or(OwnedToolLoopError::Protocol("turn ended without a terminal event"))?
		.map_err(OwnedToolLoopError::Turn)?;
	let call =
		completed.ok_or(OwnedToolLoopError::Protocol("tool turn ended without a complete call"))?;
	Ok((outcome, call))
}

async fn collect_terminal_outcome<S>(stream: &mut S) -> Result<ChatOutcome, OwnedToolLoopError>
where
	S: Stream<Item = TurnEvent> + Unpin,
{
	let mut terminal = None;
	while let Some(event) = stream.next().await {
		if terminal.is_some() {
			return Err(OwnedToolLoopError::Protocol("event followed terminal event"));
		}
		match event {
			TurnEvent::Outcome(outcome) => terminal = Some(Ok(outcome)),
			TurnEvent::Error(error) => terminal = Some(Err(error)),
			_ => {},
		}
	}
	terminal
		.ok_or(OwnedToolLoopError::Protocol("follow-up ended without a terminal event"))?
		.map_err(OwnedToolLoopError::Turn)
}

fn unique_committed_call(outcome: &ChatOutcome) -> Result<&ToolCall, OwnedToolLoopError> {
	let mut calls = outcome.output.iter().filter_map(|item| match &item.kind {
		ItemKind::ToolCall(call) => Some(call),
		_ => None,
	});
	let call = calls
		.next()
		.ok_or(OwnedToolLoopError::Protocol("tool outcome omitted the canonical tool call"))?;
	if calls.next().is_some() {
		return Err(OwnedToolLoopError::Protocol(
			"tool outcome committed more than one canonical tool call",
		));
	}
	Ok(call)
}
