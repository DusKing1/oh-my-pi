//! Single-channel interrupt mailbox with point-specific draining.

use std::collections::VecDeque;

use omp_core::Str;
use omp_proto::thread::v1::Item;

/// Earliest loop point at which an interrupt may be observed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InterruptClass {
	/// Between tool completions while a batch is running.
	Immediate,
	/// After a committed turn outcome and before the next submission.
	TurnBoundary,
	/// When the loop would otherwise become idle.
	Idle,
}

impl InterruptClass {
	const fn index(self) -> usize {
		match self {
			Self::Immediate => 0,
			Self::TurnBoundary => 1,
			Self::Idle => 2,
		}
	}
}

/// A loop location at which queued interrupts are drained.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DrainPoint {
	/// The completion boundary between tools in a batch.
	Immediate,
	/// The boundary following a committed turn outcome.
	TurnBoundary,
	/// The point at which the loop would otherwise stop.
	Idle,
}

impl DrainPoint {
	const fn highest_class(self) -> usize {
		match self {
			Self::Immediate => InterruptClass::Immediate.index(),
			Self::TurnBoundary => InterruptClass::TurnBoundary.index(),
			Self::Idle => InterruptClass::Idle.index(),
		}
	}
}

/// Typed attribution for an interrupt producer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InterruptSource {
	/// Settlement notification for one detached job.
	Job {
		/// Stable detached-job identifier.
		id: Str,
	},
	/// Named producer without a more specific structured source.
	Producer(Str),
}

/// Canonical thread input delivered asynchronously to the agent loop.
#[derive(Clone, Debug, PartialEq)]
pub struct Interrupt {
	/// Earliest point at which this input may interrupt the loop.
	pub class:  InterruptClass,
	/// Canonical thread item to append on delivery.
	pub item:   Item,
	/// Typed attribution for the producer of this input.
	pub source: InterruptSource,
}

/// Cloneable nonblocking producer for the agent's sole command mailbox.
#[derive(Clone, Debug)]
pub struct MailboxSender {
	tx: flume::Sender<Interrupt>,
}

impl MailboxSender {
	/// Enqueues an interrupt without blocking the producer.
	///
	/// The mailbox is unbounded, so this fails only after its receiver closes.
	pub fn try_enqueue(&self, interrupt: Interrupt) -> Result<(), flume::TrySendError<Interrupt>> {
		self.tx.try_send(interrupt)
	}

	/// Returns whether the receiving mailbox has closed.
	pub fn is_disconnected(&self) -> bool {
		self.tx.is_disconnected()
	}
}

/// Single-consumer interrupt mailbox with an ordered backlog.
///
/// Shutdown is deliberately absent: the owner races [`Self::wait`] against a
/// `tokio::watch` receiver, so selecting shutdown never consumes an interrupt.
pub struct Mailbox {
	tx:      flume::Sender<Interrupt>,
	rx:      flume::Receiver<Interrupt>,
	backlog: VecDeque<Interrupt>,
}
impl Default for Mailbox {
	fn default() -> Self {
		Self::new()
	}
}

impl Mailbox {
	/// Creates an empty unbounded mailbox.
	pub fn new() -> Self {
		let (tx, rx) = flume::unbounded();
		Self { tx, rx, backlog: VecDeque::new() }
	}

	/// Returns a cloneable producer for this mailbox.
	pub fn sender(&self) -> MailboxSender {
		MailboxSender { tx: self.tx.clone() }
	}

	/// Waits until one interrupt is retained in the local backlog.
	///
	/// Cancelling this future leaves the channel unchanged. Once it completes,
	/// the received value remains owned by the mailbox until a matching drain.
	pub async fn wait(&mut self) -> Result<(), flume::RecvError> {
		let interrupt = self.rx.recv_async().await?;
		self.push_back(interrupt);
		Ok(())
	}

	/// Drains every interrupt eligible at `point` in class-precedence order.
	///
	/// FIFO is preserved within each class. When `defer_interrupts` is set,
	/// queued immediate interrupts are permanently demoted to the turn boundary
	/// before eligibility is evaluated, so an immediate-point drain retains
	/// them.
	pub fn drain(&mut self, point: DrainPoint, defer_interrupts: bool) -> Vec<Interrupt> {
		self.pump(defer_interrupts);
		if defer_interrupts {
			self.demote_immediate();
		}

		let mut drained = Vec::new();
		for class in 0..=point.highest_class() {
			let queued = self.backlog.len();
			for _ in 0..queued {
				let Some(interrupt) = self.backlog.pop_front() else {
					break;
				};
				if interrupt.class.index() == class {
					drained.push(interrupt);
				} else {
					self.backlog.push_back(interrupt);
				}
			}
		}
		drained
	}

	/// Restores previously drained interrupts ahead of newer inputs.
	///
	/// This is the rollback operation for a drain whose surrounding loop action
	/// aborts before the items are staged into a thread delta.
	pub fn requeue_front(&mut self, interrupts: Vec<Interrupt>) {
		for interrupt in interrupts.into_iter().rev() {
			self.backlog.push_front(interrupt);
		}
	}

	/// Returns the number of interrupts retained locally and in the channel.
	pub fn len(&self) -> usize {
		self.backlog.len() + self.rx.len()
	}

	/// Returns whether no interrupts are currently queued.
	pub fn is_empty(&self) -> bool {
		self.backlog.is_empty() && self.rx.is_empty()
	}

	fn pump(&mut self, defer_interrupts: bool) {
		while let Ok(mut interrupt) = self.rx.try_recv() {
			if defer_interrupts && interrupt.class == InterruptClass::Immediate {
				interrupt.class = InterruptClass::TurnBoundary;
			}
			self.push_back(interrupt);
		}
	}

	fn push_back(&mut self, interrupt: Interrupt) {
		self.backlog.push_back(interrupt);
	}

	fn demote_immediate(&mut self) {
		for interrupt in &mut self.backlog {
			if interrupt.class == InterruptClass::Immediate {
				interrupt.class = InterruptClass::TurnBoundary;
			}
		}
	}
}
