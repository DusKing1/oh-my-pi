//! Concurrent server-initiated tool invocations for a live gateway turn.

use std::{collections::HashMap, fmt, sync::Arc, time::Duration};

use bytes::Bytes;
use omp_core::{IntoStr, Str};
use omp_env::EnvClient;
use omp_proto::{
	inference::v1::{ExecStatus, Invoke, InvokeComplete, exec_status},
	thread::v1::item,
};
use omp_tool::{PromptCaps, Registry, RegistryError, ToolIdentity};
use tokio::sync::watch;

use crate::{
	BatchError, EventBus, SpeculativeCall, ToolBatch,
	batch::BatchUpdate,
	turn::InvokeFrame,
};

/// Failure while dispatching a server-initiated invocation.
#[derive(Debug)]
pub(crate) enum DuplexError {
	/// The environment invocation or canonical lowering failed.
	Batch(BatchError),
	/// A typed tool update could not be projected to an invocation frame.
	Registry(RegistryError),
	/// A completed batch did not contain its promised canonical tool result.
	MissingToolResult,
}

impl fmt::Display for DuplexError {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::Batch(error) => write!(formatter, "duplex tool invocation failed: {error}"),
			Self::Registry(error) => write!(formatter, "duplex update projection failed: {error}"),
			Self::MissingToolResult => formatter.write_str("duplex batch omitted its tool result"),
		}
	}
}

impl std::error::Error for DuplexError {
	fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
		match self {
			Self::Batch(error) => Some(error),
			Self::Registry(error) => Some(error),
			Self::MissingToolResult => None,
		}
	}
}

impl From<BatchError> for DuplexError {
	fn from(error: BatchError) -> Self {
		Self::Batch(error)
	}
}

impl From<RegistryError> for DuplexError {
	fn from(error: RegistryError) -> Self {
		Self::Registry(error)
	}
}

struct ActiveInvocation {
	token:     u64,
	interrupt: watch::Sender<Option<Str>>,
}

struct Completion {
	invocation_id: String,
	token:         u64,
	result:        Result<InvokeFrame, DuplexError>,
}

/// Owns concurrent in-turn invocations and suppresses cancelled completions.
pub(crate) struct DuplexManager {
	env:             EnvClient,
	registry:        Arc<Registry>,
	events:          EventBus,
	caps:            PromptCaps,
	interrupt_grace: Duration,
	active:          HashMap<String, ActiveInvocation>,
	completion_tx:   flume::Sender<Completion>,
	completion_rx:   flume::Receiver<Completion>,
	next_token:      u64,
}

impl DuplexManager {
	/// Creates an empty manager sharing the loop's environment and event feed.
	pub(crate) fn new(
		env: EnvClient,
		registry: Arc<Registry>,
		events: EventBus,
		caps: PromptCaps,
		interrupt_grace: Duration,
	) -> Self {
		let (completion_tx, completion_rx) = flume::unbounded();
		Self {
			env,
			registry,
			events,
			caps,
			interrupt_grace,
			active: HashMap::new(),
			completion_tx,
			completion_rx,
			next_token: 1,
		}
	}

	/// Starts one invocation without waiting for any other invocation.
	pub(crate) fn start(&mut self, invoke: Invoke) {
		let invocation_id = invoke.invocation_id.clone();
		if let Some(previous) = self.active.remove(&invocation_id) {
			let _ = previous
				.interrupt
				.send(Some(Str::from("superseded duplicate invocation")));
		}

		let token = self.next_token;
		self.next_token = self.next_token.wrapping_add(1);
		let (interrupt, receiver) = watch::channel(None);
		self
			.active
			.insert(invocation_id.clone(), ActiveInvocation { token, interrupt });

		let env = self.env.clone();
		let registry = Arc::clone(&self.registry);
		let events = self.events.clone();
		let caps = self.caps;
		let grace = self.interrupt_grace;
		let completions = self.completion_tx.clone();
		let _task = tokio::spawn(async move {
			let result = run_invocation(
				invoke,
				env,
				registry,
				events,
				caps,
				receiver,
				grace,
				completions.clone(),
				token,
			)
			.await
			.map(InvokeFrame::Complete);
			let _ = completions.send_async(Completion { invocation_id, token, result }).await;
		});
	}

	/// Cooperatively interrupts a live invocation before structural
	/// cancellation. Unknown and already-completed identifiers are harmless.
	pub(crate) fn cancel(&mut self, invocation_id: &str) {
		if let Some(active) = self.active.remove(invocation_id) {
			let _ = active
				.interrupt
				.send(Some(Str::from("gateway cancelled invocation")));
		}
	}

	/// Returns whether no invocation can still produce a completion.
	pub(crate) fn is_empty(&self) -> bool {
		self.active.is_empty()
	}

	/// Waits for the next non-cancelled input or completion frame.
	pub(crate) async fn next(&mut self) -> Option<(String, Result<InvokeFrame, DuplexError>)> {
		while !self.active.is_empty() {
			let completion = self.completion_rx.recv_async().await.ok()?;
			let current = self.active.get(&completion.invocation_id);
			if !matches!(current, Some(active) if active.token == completion.token) {
				continue;
			}
			if !matches!(&completion.result, Ok(InvokeFrame::Input(_))) {
				self.active.remove(&completion.invocation_id);
			}
			return Some((completion.invocation_id, completion.result));
		}
		None
	}
}

impl Drop for DuplexManager {
	fn drop(&mut self) {
		for (_, active) in self.active.drain() {
			let _ = active
				.interrupt
				.send(Some(Str::from("duplex manager shutting down")));
		}
	}
}

async fn run_invocation(
	invoke: Invoke,
	env: EnvClient,
	registry: Arc<Registry>,
	events: EventBus,
	caps: PromptCaps,
	interrupt: watch::Receiver<Option<Str>>,
	grace: Duration,
	frames: flume::Sender<Completion>,
	token: u64,
) -> Result<InvokeComplete, DuplexError> {
	let invocation_id = invoke.invocation_id;
	let frame_invocation_id = invocation_id.clone();
	let Some(call) = invoke.tool_call else {
		return Ok(failed_completion(invocation_id, "control invocation has no canonical tool call"));
	};
	if invocation_id.is_empty() {
		return Ok(failed_completion(invocation_id, "invocation id is empty"));
	}
	if invoke.timeout_ms == 0 {
		return Ok(failed_completion(invocation_id, "invocation deadline is zero"));
	}
	if call.id.is_empty() || call.name.is_empty() || invoke.name.is_empty() {
		return Ok(failed_completion(invocation_id, "canonical tool call is incomplete"));
	}
	if invoke.name != call.name {
		return Ok(failed_completion(
			invocation_id,
			"dispatch name differs from canonical tool call",
		));
	}
	let Some((name, rev)) = registry.live_identity(&invoke.name) else {
		return Ok(failed_completion(invocation_id, "invocation names an unknown tool"));
	};
	let identity = ToolIdentity { name: name.clone(), rev: rev.clone() };
	let raw_args = Bytes::from(call.args_json.clone());
	let fragment = match std::str::from_utf8(&call.args_json) {
		Ok(fragment) => fragment.to_str(),
		Err(_) => return Ok(failed_completion(invocation_id, "tool arguments are not UTF-8")),
	};
	let deadline = Duration::from_millis(invoke.timeout_ms);
	let speculative =
		SpeculativeCall::open(&env, &events, Str::from(call.id.as_str()), identity, deadline).await?;
	speculative.relay_fragment(fragment).await?;
	let committed = speculative.commit(raw_args);
	let (updates_tx, updates_rx) = flume::unbounded();
	let drive = ToolBatch::new(vec![committed]).drive_streaming(
		registry.as_ref(),
		&caps,
		interrupt,
		grace,
		updates_tx,
	);
	tokio::pin!(drive);
	let mut results = loop {
		tokio::select! {
			biased;
			update = updates_rx.recv_async() => {
				let Ok(update) = update else {
					break drive.await;
				};
				send_update(
					&registry,
					&frame_invocation_id,
					token,
					&frames,
					update,
				).await?;
			},
			results = &mut drive => break results,
		}
	};
	while let Ok(update) = updates_rx.try_recv() {
		send_update(
			&registry,
			&frame_invocation_id,
			token,
			&frames,
			update,
		).await?;
	}
	let result = results.pop().ok_or(DuplexError::MissingToolResult)?;
	let tool_result = match result.item().kind.as_ref() {
		Some(item::Kind::ToolResult(result)) => result.clone(),
		_ => return Err(DuplexError::MissingToolResult),
	};
	let is_error = tool_result.is_error;
	Ok(InvokeComplete {
		invocation_id,
		tool_result: Some(tool_result),
		status: Some(ExecStatus {
			outcome: if is_error {
				exec_status::Outcome::Failed as i32
			} else {
				exec_status::Outcome::Exited as i32
			},
			..Default::default()
		}),
		..Default::default()
	})
}

async fn send_update(
	registry: &Registry,
	invocation_id: &str,
	token: u64,
	frames: &flume::Sender<Completion>,
	update: BatchUpdate,
) -> Result<(), DuplexError> {
	debug_assert!(!update.call_id.is_empty());
	let Some(input) = registry.invoke_input(&update.identity, invocation_id, &update.json)? else {
		return Ok(());
	};
	let _ = frames
		.send_async(Completion {
			invocation_id: invocation_id.to_owned(),
			token,
			result: Ok(InvokeFrame::Input(input)),
		})
		.await;
	Ok(())
}

fn failed_completion(invocation_id: String, reason: &str) -> InvokeComplete {
	InvokeComplete {
		invocation_id,
		status: Some(ExecStatus {
			outcome: exec_status::Outcome::Failed as i32,
			reason: reason.to_owned(),
			..Default::default()
		}),
		..Default::default()
	}
}

#[cfg(test)]
mod tests;
