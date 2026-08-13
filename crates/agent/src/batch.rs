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

/// One exact serialized tool update emitted while a batch call is live.
#[derive(Clone, Debug)]
pub(crate) struct BatchUpdate {
	pub(crate) call_id:  Str,
	pub(crate) identity: ToolIdentity,
	pub(crate) json:     Bytes,
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
	///
	/// Results remain in issued order. Once a call is authorized, environment
	/// or lowering failures become canonical `EffectsUnknown` results so every
	/// committed call remains journalable and peer truth is never discarded.
	pub async fn drive(self, registry: &Registry, caps: &PromptCaps) -> Vec<BatchResult> {
		self.drive_inner(registry, caps, None, Duration::ZERO, None).await
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
	) -> Vec<BatchResult> {
		self
			.drive_inner(registry, caps, Some(interrupt), grace, None)
			.await
	}

	/// Drives an interruptible batch while copying exact updates to `updates`.
	pub(crate) async fn drive_streaming(
		self,
		registry: &Registry,
		caps: &PromptCaps,
		interrupt: watch::Receiver<Option<Str>>,
		grace: Duration,
		updates: flume::Sender<BatchUpdate>,
	) -> Vec<BatchResult> {
		self
			.drive_inner(registry, caps, Some(interrupt), grace, Some(updates))
			.await
	}

	async fn drive_inner(
		self,
		registry: &Registry,
		caps: &PromptCaps,
		interrupt: Option<watch::Receiver<Option<Str>>>,
		grace: Duration,
		updates: Option<flume::Sender<BatchUpdate>>,
	) -> Vec<BatchResult> {
		let count = self.calls.len();
		let mut running = FuturesUnordered::new();
		for (index, call) in self.calls.into_iter().enumerate() {
			running.push(run_call(
				index,
				call,
				registry,
				caps,
				interrupt.clone(),
				grace,
				updates.clone(),
			));
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
	updates: Option<flume::Sender<BatchUpdate>>,
) -> (usize, BatchResult) {
	if let Some(reason) = interrupt
		.as_mut()
		.and_then(|receiver| receiver.borrow_and_update().clone())
	{
		let reason = format!("interrupted before execution: {reason}").to_str();
		return (index, lower_abort_total(&call, Abort::Skipped { reason }));
	}
	let commit = if let Some(receiver) = interrupt.as_mut() {
		tokio::select! {
			biased;
			reason = wait_for_interrupt(receiver) => {
				let reason = format!("interrupted before execution: {reason}").to_str();
				return (index, lower_abort_total(&call, Abort::Skipped { reason }));
			},
			result = call.invocation.commit_args(call.raw_args.clone()) => result,
		}
	} else {
		call.invocation.commit_args(call.raw_args.clone()).await
	};
	if let Err(error) = commit {
		let reason = format!("ArgsCommitted delivery failed: {error}").to_str();
		return (index, lower_abort_total(&call, Abort::EffectsUnknown { reason }));
	}

	let mut publish_update = |update: omp_proto::env::v1::Update| {
		let json = update.json;
		call.events.publish(AgentEvent::ToolUpdate {
			call_id: call.call_id.clone(),
			json:    json.clone(),
		});
		if let Some(updates) = updates.as_ref() {
			let _ = updates.send(BatchUpdate {
				call_id:  call.call_id.clone(),
				identity: call.identity.clone(),
				json,
			});
		}
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
		Ok(InvocationTerminal::Verdict(verdict)) => {
			lower_verdict(&call, registry, caps, verdict).unwrap_or_else(|error| {
				lower_abort_total(
					&call,
					Abort::EffectsUnknown {
						reason: format!("failed to lower environment verdict: {error}").to_str(),
					},
				)
			})
		},
		Ok(InvocationTerminal::StreamError(error)) => lower_abort_total(
			&call,
			Abort::EffectsUnknown {
				reason: format!("environment invocation stream lost: {}", error.message).to_str(),
			},
		),
		Ok(InvocationTerminal::Closed) => lower_abort_total(&call, Abort::MissingOutcome),
		Ok(InvocationTerminal::CancelUnobserved) => lower_abort_total(
			&call,
			Abort::EffectsUnknown {
				reason: Str::new_static(
					"environment owner did not report terminal truth after cancellation",
				),
			},
		),
		Err(error) => lower_abort_total(
			&call,
			Abort::EffectsUnknown {
				reason: format!("environment invocation failed: {error}").to_str(),
			},
		),
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
	let is_error = !matches!(verdict, Verdict::Ok(_));
	if let Some(parts) = harness_parts(&verdict) {
		return lower_tool_parts(call, &wire.json, is_error, wire.useless, &parts);
	}
	match registry.prompt(&call.identity, &wire.json, caps) {
		Ok(Some(parts)) => lower_tool_parts(call, &wire.json, is_error, wire.useless, &parts),
		Ok(None) => unreachable!("harness verdict branches were handled before registry projection"),
		Err(_) => lower_canonical_parts(call, &wire.json, is_error, wire.useless, wire.parts),
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

fn lower_abort_total(call: &CommittedCall, abort: Abort) -> BatchResult {
	lower_abort(call, abort).expect(
		"harness-owned Aborted verdict serialization and canonical lowering are infallible",
	)
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

#[cfg(test)]
mod tests {
	use std::collections::HashMap;

	use omp_env::frame::{self, client_frame, server_frame};
	use omp_proto::thread::v1::{Part as ThreadPart, part};
	use omp_tool::Rev;

	use super::*;

	fn identity(name: &'static str) -> ToolIdentity {
		ToolIdentity {
			name: Str::new_static(name),
			rev:  Rev { family: Str::new_static("test"), n: 1 },
		}
	}

	fn caps() -> PromptCaps {
		PromptCaps { maximum_parts: 8, maximum_text_bytes: 4096, media: false }
	}

	fn terminal_text(result: &BatchResult) -> &str {
		let Some(omp_proto::thread::v1::item::Kind::ToolResult(result)) =
			result.item().kind.as_ref()
		else {
			panic!("batch completion was not a ToolResult");
		};
		let Some(ThreadPart { kind: Some(part::Kind::Text(text)) }) = result.parts.first() else {
			panic!("tool result did not contain text");
		};
		text
	}

	#[tokio::test]
	async fn two_calls_preserve_order_and_malformed_terminal_becomes_effects_unknown() {
		let (client, transport) = EnvClient::in_process(0);
		let (requests, responses) = transport.into_parts();
		let server = tokio::spawn(async move {
			let mut opened = HashMap::new();
			while opened.len() < 2 {
				let frame = requests.recv_async().await.expect("invoke frame");
				let Some(client_frame::Body::InvokeTool(invoke)) = frame.body else {
					continue;
				};
				opened.insert(invoke.invocation_id, frame.request_id);
			}
			let mut committed = HashMap::new();
			while committed.len() < 2 {
				let frame = requests.recv_async().await.expect("commit frame");
				let Some(client_frame::Body::ArgsCommitted(commit)) = frame.body else {
					continue;
				};
				committed.insert(commit.invocation_id, frame.request_id);
			}
			let second = committed["second"];
			responses
				.send_async(frame::ServerFrame {
					request_id: second,
					body: Some(server_frame::Body::Verdict(frame::Verdict {
						invocation_id: "second".into(),
						json: Bytes::from_static(b"not-json"),
						..Default::default()
					})),
					..Default::default()
				})
				.await
				.expect("malformed verdict");
			let first = committed["first"];
			responses
				.send_async(frame::ServerFrame {
					request_id: first,
					body: Some(server_frame::Body::Verdict(frame::Verdict {
						invocation_id: "first".into(),
						json: Bytes::from_static(br#"{"kind":"ok","value":{"answer":1}}"#),
						parts: vec![ThreadPart {
							kind: Some(part::Kind::Text("one".into())),
						}],
						..Default::default()
					})),
					..Default::default()
				})
				.await
				.expect("valid verdict");
		});
		let events = EventBus::new();
		let observed = events.subscribe_lossless();
		let first = SpeculativeCall::open(
			&client,
			&events,
			Str::new_static("first"),
			identity("first_tool"),
			Duration::from_secs(1),
		)
		.await
		.expect("open first");
		let second = SpeculativeCall::open(
			&client,
			&events,
			Str::new_static("second"),
			identity("second_tool"),
			Duration::from_secs(1),
		)
		.await
		.expect("open second");
		let results = ToolBatch::new(vec![
			first.commit(Bytes::from_static(b"{}")),
			second.commit(Bytes::from_static(b"{}")),
		])
		.drive(&Registry::new(), &caps())
		.await;
		server.await.expect("scripted env task");

		assert_eq!(results.len(), 2);
		assert_eq!(terminal_text(&results[0]), "one");
		assert!(terminal_text(&results[1]).contains("failed to lower environment verdict"));
		let mut finished = 0;
		while let Ok(event) = observed.try_recv() {
			if matches!(event.as_ref(), AgentEvent::ToolFinished { .. }) {
				finished += 1;
			}
		}
		assert_eq!(finished, 2, "every committed call emits exactly one result");
	}

	#[tokio::test]
	async fn interrupt_before_commit_yields_skipped_without_args_committed() {
		let (client, transport) = EnvClient::in_process(0);
		let (requests, _responses) = transport.into_parts();
		let events = EventBus::new();
		let call = SpeculativeCall::open(
			&client,
			&events,
			Str::new_static("skipped"),
			identity("skipped_tool"),
			Duration::from_secs(1),
		)
		.await
		.expect("open call");
		let opened = requests.recv_async().await.expect("invoke frame");
		assert!(matches!(opened.body, Some(client_frame::Body::InvokeTool(_))));
		let (_interrupt_tx, interrupt_rx) =
			watch::channel(Some(Str::new_static("user interrupted")));
		let results = ToolBatch::new(vec![call.commit(Bytes::from_static(b"{}"))])
			.drive_interruptible(&Registry::new(), &caps(), interrupt_rx, Duration::from_millis(10))
			.await;
		assert_eq!(results.len(), 1);
		assert!(terminal_text(&results[0]).starts_with("skipped:"));
		while let Ok(frame) = requests.try_recv() {
			assert!(
				!matches!(frame.body, Some(client_frame::Body::ArgsCommitted(_))),
				"interrupted unstarted call sent ArgsCommitted"
			);
		}
	}
}
