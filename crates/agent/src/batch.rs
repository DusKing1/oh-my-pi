//! Speculative environment invocations and ordered concurrent tool batches.

use std::{fmt, sync::Arc, time::Duration};

use bytes::Bytes;
use futures::{StreamExt, stream::FuturesUnordered};
use omp_core::{IntoStr, Str};
use omp_env::{ClientError, EnvClient, Invocation};
use omp_proto::{
	env::v1::InvokeTool,
	thread::v1::{Item, Part as CanonicalPart},
};
use omp_tool::{
	Abort, ArgIssue, ArgPath, JobRef, Outcome, Part, PromptCaps, Registry, ToolIdentity, Verdict,
};
use serde_json::Value;
use tokio::sync::watch;

use crate::{
	events::{AgentEvent, EventBus},
	project::{tool_result_item, tool_result_item_canonical_parts},
	supervise::{InvocationTerminal, drain_terminal, interrupt_with_grace},
};

/// Failure to open, relay, decode, project, or lower a tool invocation.
#[derive(Debug)]
pub enum BatchError {
	/// The environment channel rejected an operation.
	Environment(ClientError),
	/// A terminal environment payload was not a supported structured outcome.
	InvalidVerdict(serde_json::Error),
	/// Canonical result construction failed.
	Projection(Str),
}

impl fmt::Display for BatchError {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::Environment(error) => write!(formatter, "environment invocation failed: {error}"),
			Self::InvalidVerdict(error) => write!(formatter, "invalid tool verdict: {error}"),
			Self::Projection(error) => write!(formatter, "canonical tool result failed: {error}"),
		}
	}
}

impl std::error::Error for BatchError {
	fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
		match self {
			Self::Environment(error) => Some(error),
			Self::InvalidVerdict(error) => Some(error),
			Self::Projection(_) => None,
		}
	}
}

impl From<ClientError> for BatchError {
	fn from(error: ClientError) -> Self {
		Self::Environment(error)
	}
}

/// An environment invocation opened before its model arguments are committed.
///
/// Relaying fragments may prepare environment-owned resources, but only
/// [`commit`](Self::commit) creates a call eligible to send `ArgsCommitted`.
/// Dropping this handle structurally cancels the uncommitted invocation.
pub struct SpeculativeCall {
	call_id:    Str,
	identity:   ToolIdentity,
	invocation: Invocation,
	events:     EventBus,
}

impl SpeculativeCall {
	/// Opens an environment invocation without authorizing effects.
	pub async fn open(
		env: &EnvClient,
		events: &EventBus,
		call_id: Str,
		identity: ToolIdentity,
		deadline: Duration,
	) -> Result<Self, BatchError> {
		let invocation = env
			.invoke(InvokeTool {
				invocation_id: call_id.to_string(),
				name:          identity.name.to_string(),
				rev:           identity.rev.to_string(),
				deadline_ms:   u64::try_from(deadline.as_millis()).unwrap_or(u64::MAX),
				props:         Default::default(),
			})
			.await?;
		events.publish(AgentEvent::ToolOpened {
			call_id: call_id.clone(),
			name:    identity.name.clone(),
			rev:     identity.rev.clone(),
		});
		Ok(Self { call_id, identity, invocation, events: events.clone() })
	}

	/// Returns the stable model-authored call identifier.
	pub fn call_id(&self) -> &Str {
		&self.call_id
	}

	/// Returns the exact live tool identity selected when speculation opened.
	pub fn identity(&self) -> &ToolIdentity {
		&self.identity
	}

	/// Relays one provider argument fragment verbatim without validating it.
	pub async fn relay_fragment(&self, fragment: Str) -> Result<(), BatchError> {
		self.invocation.arg_text(fragment.clone()).await?;
		self.events.publish(AgentEvent::ToolArgs {
			call_id:  self.call_id.clone(),
			fragment: Bytes::copy_from_slice(fragment.as_bytes()),
		});
		Ok(())
	}

	/// Binds authoritative committed argument bytes to this invocation.
	///
	/// This local transition performs no I/O. [`ToolBatch::drive`] sends every
	/// batch member's commit gate concurrently, so issued-order iteration cannot
	/// serialize otherwise independent tool effects.
	pub fn commit(self, raw_args: Bytes) -> CommittedCall {
		CommittedCall {
			call_id: self.call_id,
			identity: self.identity,
			raw_args,
			invocation: self.invocation,
			events: self.events,
		}
	}
}

/// An authoritative call waiting for the concurrent `ArgsCommitted` gate.
pub struct CommittedCall {
	call_id:    Str,
	identity:   ToolIdentity,
	raw_args:   Bytes,
	invocation: Invocation,
	events:     EventBus,
}

impl CommittedCall {
	/// Returns the stable model-authored call identifier.
	pub fn call_id(&self) -> &Str {
		&self.call_id
	}

	/// Returns the exact committed model argument bytes.
	pub fn raw_args(&self) -> &Bytes {
		&self.raw_args
	}

	/// Returns the tool identity fixed when speculation opened.
	pub fn identity(&self) -> &ToolIdentity {
		&self.identity
	}
}

/// One ordered batch completion shared with the event feed.
#[derive(Clone)]
pub struct BatchResult {
	event: Arc<AgentEvent>,
	job:   Option<JobRef>,
}

impl BatchResult {
	/// Borrows the canonical result item carried by this completion's event.
	pub fn item(&self) -> &Item {
		match self.event.as_ref() {
			AgentEvent::ToolFinished { item, .. } => item,
			_ => unreachable!("batch results only retain ToolFinished events"),
		}
	}

	/// Borrows the already-published immutable result event.
	pub fn event(&self) -> &Arc<AgentEvent> {
		&self.event
	}

	/// Returns detached job ownership when work outlives the turn.
	pub fn job(&self) -> Option<&JobRef> {
		self.job.as_ref()
	}

	/// Takes detached job ownership for registration with the job board.
	pub fn into_job(self) -> Option<JobRef> {
		self.job
	}

	/// Returns whether this completion transferred work to the job board.
	pub fn is_detached(&self) -> bool {
		self.job.is_some()
	}
}

/// A set of committed calls driven concurrently and returned in issued order.
pub struct ToolBatch {
	calls: Vec<CommittedCall>,
}

impl ToolBatch {
	/// Creates a batch in model-issued order.
	pub fn new(calls: Vec<CommittedCall>) -> Self {
		Self { calls }
	}

	/// Returns the number of calls in the batch.
	pub fn len(&self) -> usize {
		self.calls.len()
	}

	/// Returns whether the batch contains no calls.
	pub fn is_empty(&self) -> bool {
		self.calls.is_empty()
	}

	/// Sends every commit gate and drives all calls concurrently.
	pub async fn drive(
		self,
		registry: &Registry,
		caps: &PromptCaps,
	) -> Result<Vec<BatchResult>, BatchError> {
		self.drive_inner(registry, caps, None, Duration::ZERO).await
	}

	/// Drives the batch with one watch-broadcast cooperative interrupt source.
	///
	/// The first nonempty reason interrupts every still-running call. Each call
	/// gets `grace` to report a cooperative verdict before its `RunGuard` queues
	/// structural cancellation; peers remain independent throughout.
	pub async fn drive_interruptible(
		self,
		registry: &Registry,
		caps: &PromptCaps,
		interrupt: watch::Receiver<Option<Str>>,
		grace: Duration,
	) -> Result<Vec<BatchResult>, BatchError> {
		self
			.drive_inner(registry, caps, Some(interrupt), grace)
			.await
	}

	async fn drive_inner(
		self,
		registry: &Registry,
		caps: &PromptCaps,
		interrupt: Option<watch::Receiver<Option<Str>>>,
		grace: Duration,
	) -> Result<Vec<BatchResult>, BatchError> {
		let count = self.calls.len();
		let mut running = FuturesUnordered::new();
		for (index, call) in self.calls.into_iter().enumerate() {
			running.push(run_call(index, call, registry, caps, interrupt.clone(), grace));
		}

		let mut ordered = Vec::with_capacity(count);
		ordered.resize_with(count, || None);
		while let Some((index, result)) = running.next().await {
			ordered[index] = Some(result);
		}
		ordered
			.into_iter()
			.map(|result| result.expect("every batch call produced exactly one completion"))
			.collect()
	}
}

async fn run_call(
	index: usize,
	mut call: CommittedCall,
	registry: &Registry,
	caps: &PromptCaps,
	mut interrupt: Option<watch::Receiver<Option<Str>>>,
	grace: Duration,
) -> (usize, Result<BatchResult, BatchError>) {
	if let Some(reason) = interrupt
		.as_mut()
		.and_then(|receiver| receiver.borrow_and_update().clone())
	{
		let reason = format!("interrupted before execution: {reason}").to_str();
		return (index, lower_abort(&call, Abort::Skipped { reason }));
	}
	if let Err(error) = call.invocation.commit_args(call.raw_args.clone()).await {
		let reason = format!("ArgsCommitted delivery failed: {error}").to_str();
		return (index, lower_abort(&call, Abort::EffectsUnknown { reason }));
	}

	let mut publish_update = |update: omp_proto::env::v1::Update| {
		call
			.events
			.publish(AgentEvent::ToolUpdate { call_id: call.call_id.clone(), json: update.json });
	};
	let terminal = if let Some(receiver) = interrupt.as_mut() {
		tokio::select! {
			terminal = drain_terminal(&mut call.invocation, &mut publish_update) => terminal,
			reason = wait_for_interrupt(receiver) => {
				interrupt_with_grace(
					&mut call.invocation,
					reason,
					grace,
					&mut publish_update,
				).await
			},
		}
	} else {
		drain_terminal(&mut call.invocation, &mut publish_update).await
	};

	let result = match terminal {
		Ok(InvocationTerminal::Verdict(verdict)) => lower_verdict(&call, registry, caps, verdict),
		Ok(InvocationTerminal::StreamError(error)) => lower_abort(&call, Abort::EffectsUnknown {
			reason: format!("environment invocation stream lost: {}", error.message).to_str(),
		}),
		Ok(InvocationTerminal::Closed) => lower_abort(&call, Abort::MissingOutcome),
		Err(error) => lower_abort(&call, Abort::EffectsUnknown {
			reason: format!("environment invocation failed: {error}").to_str(),
		}),
	};
	(index, result)
}

async fn wait_for_interrupt(receiver: &mut watch::Receiver<Option<Str>>) -> Str {
	loop {
		if let Some(reason) = receiver.borrow_and_update().clone() {
			return reason;
		}
		if receiver.changed().await.is_err() {
			std::future::pending::<()>().await;
		}
	}
}

fn lower_verdict(
	call: &CommittedCall,
	registry: &Registry,
	caps: &PromptCaps,
	wire: omp_proto::env::v1::Verdict,
) -> Result<BatchResult, BatchError> {
	if let Ok(Outcome::Detached(job)) = serde_json::from_slice::<Outcome<Value, Value>>(&wire.json) {
		return lower_detached(call, wire.json, job);
	}

	let verdict = serde_json::from_slice::<Verdict<Value, Value>>(&wire.json)
		.map_err(BatchError::InvalidVerdict)?;
	if let Some(parts) = harness_parts(&verdict) {
		return lower_tool_parts(call, &wire.json, wire.is_error, wire.useless, &parts);
	}
	match registry.prompt(&call.identity, &wire.json, caps) {
		Ok(Some(parts)) => lower_tool_parts(call, &wire.json, wire.is_error, wire.useless, &parts),
		Ok(None) => unreachable!("harness verdict branches were handled before registry projection"),
		Err(_) => lower_canonical_parts(call, &wire.json, wire.is_error, wire.useless, wire.parts),
	}
}

fn lower_detached(
	call: &CommittedCall,
	raw: Bytes,
	job: JobRef,
) -> Result<BatchResult, BatchError> {
	let text =
		format!("job started; artifact will land at job://{} ({})", job.id, job.artifact.description)
			.to_str();
	let parts = [Part::Text { text }];
	let item = tool_result_item(0, &call.call_id, &call.identity, &raw, false, false, &parts)
		.map_err(|error| BatchError::Projection(error.to_string().to_str()))?;
	let event = finish_event(call, item);
	Ok(BatchResult { event, job: Some(job) })
}

fn lower_abort(call: &CommittedCall, abort: Abort) -> Result<BatchResult, BatchError> {
	let verdict = Verdict::<Value, Value>::Aborted(abort);
	let raw = Bytes::from(serde_json::to_vec(&verdict).map_err(BatchError::InvalidVerdict)?);
	let parts = harness_parts(&verdict).expect("aborted verdict always uses the harness renderer");
	lower_tool_parts(call, &raw, true, false, &parts)
}

fn lower_tool_parts(
	call: &CommittedCall,
	verdict: &[u8],
	is_error: bool,
	useless: bool,
	parts: &[Part],
) -> Result<BatchResult, BatchError> {
	let item = tool_result_item(0, &call.call_id, &call.identity, verdict, is_error, useless, parts)
		.map_err(|error| BatchError::Projection(error.to_string().to_str()))?;
	Ok(BatchResult { event: finish_event(call, item), job: None })
}

fn lower_canonical_parts(
	call: &CommittedCall,
	verdict: &[u8],
	is_error: bool,
	useless: bool,
	parts: Vec<CanonicalPart>,
) -> Result<BatchResult, BatchError> {
	let item = tool_result_item_canonical_parts(
		0,
		&call.call_id,
		&call.identity,
		verdict,
		is_error,
		useless,
		parts,
	)
	.map_err(|error| BatchError::Projection(error.to_string().to_str()))?;
	Ok(BatchResult { event: finish_event(call, item), job: None })
}

fn finish_event(call: &CommittedCall, item: Item) -> Arc<AgentEvent> {
	call
		.events
		.publish(AgentEvent::ToolFinished { call_id: call.call_id.clone(), item })
}

fn harness_parts(verdict: &Verdict<Value, Value>) -> Option<Vec<Part>> {
	let text = match verdict {
		Verdict::Args(issue) => render_arg_issue(issue),
		Verdict::Aborted(abort) => render_abort(abort),
		Verdict::Ok(_) | Verdict::Fault(_) => return None,
	};
	Some(vec![Part::Text { text }])
}

fn render_arg_issue(issue: &ArgIssue) -> Str {
	let mut path = String::from("$");
	for segment in &issue.path {
		match segment {
			ArgPath::Key(key) => {
				path.push('[');
				path.push_str(&serde_json::to_string(key.as_str()).unwrap_or_else(|_| "\"?\"".into()));
				path.push(']');
			},
			ArgPath::Index(index) => {
				path.push('[');
				path.push_str(&index.to_string());
				path.push(']');
			},
		}
	}
	let kind_json = serde_json::to_string(&issue.kind)
		.expect("serializing a fieldless argument issue kind cannot fail");
	let kind = kind_json.trim_matches('"');
	let mut text = format!("invalid arguments at {path}: expected {} ({kind})", issue.expected);
	if let Some(found) = &issue.found {
		text.push_str("; found ");
		text.push_str(found);
	}
	if let Some(example) = &issue.example {
		text.push_str("; example ");
		text.push_str(example);
	}
	text.to_str()
}

fn render_abort(abort: &Abort) -> Str {
	match abort {
		Abort::Skipped { reason } => format!("skipped: {reason}").to_str(),
		Abort::Interrupted { reason } => format!("interrupted: {reason}").to_str(),
		Abort::EffectsUnknown { reason } => {
			format!("aborted with effects unknown: {reason}").to_str()
		},
		Abort::InputDropped => Str::new_static("aborted: invocation input dropped before commit"),
		Abort::MissingOutcome => {
			Str::new_static("aborted: executor ended without a terminal outcome")
		},
	}
}
