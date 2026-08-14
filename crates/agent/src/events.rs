//! Ordered, nonblocking fan-out for agent lifecycle events.

use std::sync::{
	Arc,
	atomic::{AtomicU8, AtomicU64, Ordering},
};

use bytes::Bytes;
use omp_core::Str;
use omp_llm_inference::TurnId;
use omp_proto::{inference::v1::TurnEvent, thread::v1::Item};
use omp_tool::Rev;
use parking_lot::Mutex;

use crate::state::AgentSnapshot;

/// Observable phase of the agent loop.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AgentPhase {
	/// Waiting for work.
	#[default]
	Idle,
	/// Rebuilding canonical thread state from the journal.
	Projecting,
	/// Streaming or recovering an inference turn.
	Turning,
	/// Executing a committed batch of tool calls.
	ToolBatch,
}

impl AgentPhase {
	const fn encode(self) -> u8 {
		match self {
			Self::Idle => 0,
			Self::Projecting => 1,
			Self::Turning => 2,
			Self::ToolBatch => 3,
		}
	}

	const fn decode(encoded: u8) -> Self {
		match encoded {
			1 => Self::Projecting,
			2 => Self::Turning,
			3 => Self::ToolBatch,
			_ => Self::Idle,
		}
	}
}

/// One immutable observation emitted by the agent loop.
#[derive(Clone, Debug)]
pub enum AgentEvent {
	/// A newly published authoritative agent snapshot.
	Snapshot(Arc<AgentSnapshot>),
	/// A lifecycle transition between loop phases.
	PhaseChanged {
		/// Phase exited by the loop.
		from: AgentPhase,
		/// Phase entered by the loop.
		to:   AgentPhase,
	},
	/// An inference event, preserved without lossy adaptation.
	Turn {
		/// Logical turn that emitted the event.
		turn_id: TurnId,
		/// Canonical turn protocol event.
		event:   Box<TurnEvent>,
	},
	/// A speculative tool invocation was opened.
	ToolOpened {
		/// Stable call identifier.
		call_id: Str,
		/// Model-facing tool name.
		name:    Str,
		/// Tool argument and rendering revision.
		rev:     Rev,
	},
	/// A raw model-authored argument fragment arrived.
	ToolArgs {
		/// Stable call identifier.
		call_id:  Str,
		/// Unparsed argument bytes in arrival order.
		fragment: Bytes,
		/// Loop-owned best-effort view of all argument fragments so far.
		view:     omp_slopjson::Value,
	},
	/// A tool emitted an ephemeral update that must not enter the thread.
	ToolUpdate {
		/// Stable call identifier.
		call_id: Str,
		/// Raw structured update bytes.
		json:    Bytes,
	},
	/// A tool completed and lowered to a canonical thread item.
	ToolFinished {
		/// Stable call identifier.
		call_id: Str,
		/// Canonical result item staged for the next delta.
		item:    Item,
	},
	/// A detached job began settlement tracking.
	JobRegistered {
		/// Stable detached-job identifier.
		job_id: Str,
	},
	/// A detached job reached a terminal settlement.
	JobSettled {
		/// Stable detached-job identifier.
		job_id: Str,
	},
	/// The loop reached an error that is visible to hosts.
	Failed {
		/// Logical turn involved, when failure occurred within a turn.
		turn_id: Option<TurnId>,
		/// Stable human-readable failure description.
		message: Str,
	},
}

#[derive(Debug)]
struct LossySender {
	tx:      flume::Sender<Arc<AgentEvent>>,
	dropped: Arc<AtomicU64>,
}

#[derive(Debug, Default)]
struct Subscribers {
	lossless: Vec<flume::Sender<Arc<AgentEvent>>>,
	lossy:    Vec<LossySender>,
}

#[derive(Debug, Default)]
struct EventBusInner {
	subscribers:   Mutex<Subscribers>,
	dropped_lossy: AtomicU64,
	phase:         AtomicU8,
}

/// Cloneable ordered fan-out for immutable shared agent events.
///
/// Publication never waits for a consumer: journal subscribers use unbounded
/// channels, while bounded UI subscribers drop on saturation and account for
/// each loss. One mutex establishes the same concurrent publication order for
/// every subscriber.
#[derive(Clone, Debug, Default)]
pub struct EventBus {
	inner: Arc<EventBusInner>,
}

impl EventBus {
	/// Creates an event bus with no subscribers.
	pub fn new() -> Self {
		Self::default()
	}

	/// Adds an unbounded, lossless subscriber suitable for journaling.
	pub fn subscribe_lossless(&self) -> EventSubscription {
		let (tx, rx) = flume::unbounded();
		self.inner.subscribers.lock().lossless.push(tx);
		EventSubscription { rx }
	}

	/// Adds a bounded, lossy subscriber suitable for UI presentation.
	///
	/// A zero capacity is valid and acts as a pure best-effort rendezvous.
	pub fn subscribe_ui(&self, capacity: usize) -> LossyEventSubscription {
		let (tx, rx) = flume::bounded(capacity);
		let dropped = Arc::new(AtomicU64::new(0));
		self
			.inner
			.subscribers
			.lock()
			.lossy
			.push(LossySender { tx, dropped: dropped.clone() });
		LossyEventSubscription { rx, dropped }
	}

	/// Publishes an owned event and returns its shared representation.
	pub fn publish(&self, event: AgentEvent) -> Arc<AgentEvent> {
		self.publish_shared(Arc::new(event))
	}

	/// Publishes an already shared event without another event allocation.
	pub fn publish_shared(&self, event: Arc<AgentEvent>) -> Arc<AgentEvent> {
		let mut subscribers = self.inner.subscribers.lock();
		subscribers
			.lossless
			.retain(|tx| tx.try_send(event.clone()).is_ok());
		subscribers
			.lossy
			.retain(|subscriber| match subscriber.tx.try_send(event.clone()) {
				Ok(()) => true,
				Err(flume::TrySendError::Full(_)) => {
					subscriber.dropped.fetch_add(1, Ordering::Relaxed);
					self.inner.dropped_lossy.fetch_add(1, Ordering::Relaxed);
					true
				},
				Err(flume::TrySendError::Disconnected(_)) => false,
			});
		event
	}

	/// Publishes a phase transition after updating the allocation-free phase
	/// snapshot.
	pub fn transition(&self, from: AgentPhase, to: AgentPhase) -> Arc<AgentEvent> {
		self.inner.phase.store(to.encode(), Ordering::Release);
		self.publish(AgentEvent::PhaseChanged { from, to })
	}

	/// Returns the latest phase without subscribing or allocating.
	pub fn phase(&self) -> AgentPhase {
		AgentPhase::decode(self.inner.phase.load(Ordering::Acquire))
	}

	/// Returns the cumulative number of events dropped by all lossy subscribers.
	pub fn dropped_lossy(&self) -> u64 {
		self.inner.dropped_lossy.load(Ordering::Relaxed)
	}
}

/// Receiving half of an ordered lossless event subscription.
pub struct EventSubscription {
	rx: flume::Receiver<Arc<AgentEvent>>,
}

impl EventSubscription {
	/// Receives the next event asynchronously.
	pub async fn recv(&self) -> Result<Arc<AgentEvent>, flume::RecvError> {
		self.rx.recv_async().await
	}

	/// Attempts to receive the next event without blocking.
	pub fn try_recv(&self) -> Result<Arc<AgentEvent>, flume::TryRecvError> {
		self.rx.try_recv()
	}

	/// Returns the number of events currently buffered for this subscriber.
	pub fn len(&self) -> usize {
		self.rx.len()
	}

	/// Returns whether this subscriber currently has no buffered events.
	pub fn is_empty(&self) -> bool {
		self.rx.is_empty()
	}
}

/// Receiving half of an ordered bounded UI event subscription.
pub struct LossyEventSubscription {
	rx:      flume::Receiver<Arc<AgentEvent>>,
	dropped: Arc<AtomicU64>,
}

impl LossyEventSubscription {
	/// Receives the next retained event asynchronously.
	pub async fn recv(&self) -> Result<Arc<AgentEvent>, flume::RecvError> {
		self.rx.recv_async().await
	}

	/// Attempts to receive the next retained event without blocking.
	pub fn try_recv(&self) -> Result<Arc<AgentEvent>, flume::TryRecvError> {
		self.rx.try_recv()
	}

	/// Returns the cumulative number of events dropped for this subscriber.
	pub fn dropped(&self) -> u64 {
		self.dropped.load(Ordering::Relaxed)
	}

	/// Returns the number of retained events currently buffered.
	pub fn len(&self) -> usize {
		self.rx.len()
	}

	/// Returns whether this subscriber currently has no buffered events.
	pub fn is_empty(&self) -> bool {
		self.rx.is_empty()
	}
}

#[cfg(test)]
mod tests {
	use super::{AgentEvent, AgentPhase, EventBus};

	#[test]
	fn phase_snapshot_tracks_transitions_across_clones() {
		let bus = EventBus::new();
		let clone = bus.clone();
		let events = bus.subscribe_lossless();

		assert_eq!(bus.phase(), AgentPhase::Idle);
		clone.transition(AgentPhase::Idle, AgentPhase::Projecting);
		assert_eq!(bus.phase(), AgentPhase::Projecting);

		let event = events
			.try_recv()
			.expect("transition must remain observable");
		assert!(matches!(event.as_ref(), AgentEvent::PhaseChanged {
			from: AgentPhase::Idle,
			to:   AgentPhase::Projecting,
		}));
		assert_eq!(clone.phase(), AgentPhase::Projecting);

		bus.transition(AgentPhase::Projecting, AgentPhase::Turning);
		assert_eq!(clone.phase(), AgentPhase::Turning);
	}
}
