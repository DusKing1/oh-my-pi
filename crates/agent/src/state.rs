//! Immutable, watch-published configuration for agent turns.

use std::{
	fmt,
	num::NonZeroU32,
	sync::Arc,
	time::{Duration, Instant},
};

use omp_core::Str;
use thiserror::Error;
use tokio::sync::watch;

use crate::{
	TurnOptions,
	prompt::{
		PromptError, PromptSource, RenderedPrompt, WorkspaceInput, WorkspacePromptSource,
		render_prompt,
	},
};

/// Bounded loop-level retry policy for recoverable turn failures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetryPolicy {
	max_attempts:    NonZeroU32,
	initial_backoff: Duration,
	max_backoff:     Duration,
}

impl RetryPolicy {
	/// Creates a bounded retry policy.
	///
	/// `max_attempts` includes the initial submission. The maximum backoff must
	/// not be shorter than the initial backoff.
	pub fn new(
		max_attempts: NonZeroU32,
		initial_backoff: Duration,
		max_backoff: Duration,
	) -> Result<Self, RetryPolicyError> {
		if initial_backoff > max_backoff {
			return Err(RetryPolicyError::BackoffOrder);
		}
		Ok(Self { max_attempts, initial_backoff, max_backoff })
	}

	/// Maximum submissions of one stable turn identity, including the first.
	#[inline]
	pub const fn max_attempts(self) -> NonZeroU32 {
		self.max_attempts
	}

	/// Backoff used for the first retry.
	#[inline]
	pub const fn initial_backoff(self) -> Duration {
		self.initial_backoff
	}

	/// Upper bound applied to retry backoff.
	#[inline]
	pub const fn max_backoff(self) -> Duration {
		self.max_backoff
	}
}

impl Default for RetryPolicy {
	fn default() -> Self {
		Self {
			max_attempts:    NonZeroU32::new(3).expect("three is non-zero"),
			initial_backoff: Duration::from_millis(250),
			max_backoff:     Duration::from_secs(4),
		}
	}
}

/// Invalid retry-policy configuration.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum RetryPolicyError {
	/// The initial delay exceeded its declared upper bound.
	#[error("initial retry backoff exceeds maximum backoff")]
	BackoffOrder,
}

/// Immutable authoritative configuration consumed by one agent turn.
///
/// A loop reads a fresh snapshot before every submission. Cloning a snapshot
/// shares prompt sources, tool names, workspace bytes, and context files.
#[derive(Clone)]
pub struct AgentSnapshot {
	/// Per-turn gateway options.
	pub turn:             TurnOptions,
	/// Names of tools enabled for this turn, in stable publication order.
	pub enabled_tools:    Arc<[Str]>,
	/// Immutable workspace and context-file input.
	pub workspace:        WorkspaceInput,
	/// Synchronous source used to construct the canonical prompt head.
	pub prompt_source:    Arc<dyn PromptSource>,
	/// Whether immediate interrupts are demoted to turn-boundary interrupts.
	pub defer_interrupts: bool,
	/// Absolute deadline for the active logical turn, when bounded by the host.
	pub deadline:         Option<Instant>,
	/// Bounded loop-level recovery policy.
	pub retry:            RetryPolicy,
}

impl AgentSnapshot {
	/// Creates a snapshot with the deterministic workspace prompt source.
	pub fn new(turn: TurnOptions, workspace: WorkspaceInput) -> Self {
		Self {
			turn,
			enabled_tools: Arc::from([]),
			workspace,
			prompt_source: Arc::new(WorkspacePromptSource),
			defer_interrupts: false,
			deadline: None,
			retry: RetryPolicy::default(),
		}
	}

	/// Renders the prompt twice and returns only a deterministic canonical head.
	#[inline]
	pub fn render_prompt(&self) -> Result<RenderedPrompt, PromptError> {
		render_prompt(self.prompt_source.as_ref(), &self.workspace)
	}
}

impl Default for AgentSnapshot {
	fn default() -> Self {
		Self::new(TurnOptions::default(), WorkspaceInput::default())
	}
}

impl fmt::Debug for AgentSnapshot {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter
			.debug_struct("AgentSnapshot")
			.field("turn", &self.turn)
			.field("enabled_tools", &self.enabled_tools)
			.field("workspace", &self.workspace)
			.field("prompt_source", &format_args!("<dyn PromptSource>"))
			.field("defer_interrupts", &self.defer_interrupts)
			.field("deadline", &self.deadline)
			.field("retry", &self.retry)
			.finish()
	}
}

/// Authoritative agent configuration published as immutable snapshots.
///
/// Readers clone the current [`Arc`] directly from a Tokio watch value. An
/// update clones the prior snapshot, applies one synchronous mutation, and
/// atomically replaces the published pointer while holding the watch slot.
#[derive(Clone, Debug)]
pub struct AgentState {
	sender: watch::Sender<Arc<AgentSnapshot>>,
}

impl AgentState {
	/// Creates state with one initially published snapshot.
	pub fn new(initial: AgentSnapshot) -> Self {
		let (sender, _receiver) = watch::channel(Arc::new(initial));
		Self { sender }
	}

	/// Returns the currently published immutable snapshot.
	#[inline]
	pub fn snapshot(&self) -> Arc<AgentSnapshot> {
		self.sender.borrow().clone()
	}

	/// Subscribes to future snapshot publications.
	///
	/// The receiver's current value is the snapshot published at subscription
	/// time; lagging readers observe the newest value without an update queue.
	#[inline]
	pub fn subscribe(&self) -> watch::Receiver<Arc<AgentSnapshot>> {
		self.sender.subscribe()
	}

	/// Atomically derives and publishes a new snapshot from the current value.
	///
	/// Concurrent callers are serialized by the watch slot, so each closure sees
	/// the snapshot published by the preceding update rather than losing writes.
	pub fn update(&self, update: impl FnOnce(&mut AgentSnapshot)) -> Arc<AgentSnapshot> {
		let mut update = Some(update);
		let mut published = None;
		self.sender.send_modify(|current| {
			let mut next = (**current).clone();
			update.take().expect("watch invokes update once")(&mut next);
			let next = Arc::new(next);
			published = Some(next.clone());
			*current = next;
		});
		published.expect("watch invokes update once")
	}

	/// Atomically replaces and returns the previously published snapshot.
	#[inline]
	pub fn replace(&self, next: AgentSnapshot) -> Arc<AgentSnapshot> {
		self.sender.send_replace(Arc::new(next))
	}
}

impl Default for AgentState {
	fn default() -> Self {
		Self::new(AgentSnapshot::default())
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn update_publishes_a_new_immutable_snapshot() {
		let state = AgentState::default();
		let old = state.snapshot();
		let receiver = state.subscribe();
		let published = state.update(|snapshot| snapshot.defer_interrupts = true);

		assert!(!old.defer_interrupts);
		assert!(published.defer_interrupts);
		assert!(Arc::ptr_eq(&published, &state.snapshot()));
		assert!(receiver.has_changed().expect("sender remains alive"));
	}

	#[test]
	fn sequential_updates_derive_from_latest_publication() {
		let state = AgentState::default();
		state.update(|snapshot| snapshot.enabled_tools = Arc::from([Str::from("read")]));
		let published = state.update(|snapshot| snapshot.defer_interrupts = true);

		assert_eq!(published.enabled_tools.as_ref(), &[Str::from("read")]);
		assert!(published.defer_interrupts);
	}
}
