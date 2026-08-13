//! Detached-job registration and settlement delivery.

use std::collections::{BTreeMap, btree_map::Entry};

use omp_core::Str;
use omp_proto::thread::v1::Item;
use omp_tool::JobRef;
use parking_lot::{Mutex, MutexGuard};

use crate::mailbox::{Interrupt, InterruptClass, MailboxSender};

/// Thread-safe registry of detached jobs awaiting settlement.
///
/// The board owns descriptors only. Environment resources and artifact storage
/// remain owned by the environment that issued each [`JobRef`].
pub struct JobBoard {
	mailbox: MailboxSender,
	pending: Mutex<BTreeMap<Str, JobRef>>,
}

impl JobBoard {
	/// Creates an empty board that delivers settlements through `mailbox`.
	pub fn new(mailbox: MailboxSender) -> Self {
		Self { mailbox, pending: Mutex::new(BTreeMap::new()) }
	}

	/// Registers a detached job without replacing an existing stable identifier.
	///
	/// Returns `true` when the job was inserted and `false` for a duplicate.
	pub fn register(&self, job: JobRef) -> bool {
		match self.pending.lock().entry(job.id.clone()) {
			Entry::Vacant(entry) => {
				entry.insert(job);
				true
			},
			Entry::Occupied(_) => false,
		}
	}

	/// Settles a pending job with one canonical thread item.
	///
	/// A known identifier enqueues exactly one turn-boundary interrupt and is
	/// removed only after enqueue succeeds. Unknown and already-settled
	/// identifiers return `Ok(false)` without touching the mailbox.
	pub fn settle(&self, job_id: &str, item: Item) -> Result<bool, flume::TrySendError<Interrupt>> {
		let mut pending = self.pending.lock();
		if !pending.contains_key(job_id) {
			return Ok(false);
		}

		self.mailbox.try_enqueue(Interrupt {
			class: InterruptClass::TurnBoundary,
			item,
			source: Str::from("job-board"),
		})?;
		pending.remove(job_id);
		Ok(true)
	}

	/// Borrows pending jobs in stable identifier order without allocating.
	pub fn pending(&self) -> PendingJobs<'_> {
		PendingJobs { guard: self.pending.lock() }
	}

	/// Returns the number of jobs awaiting settlement.
	pub fn len(&self) -> usize {
		self.pending.lock().len()
	}

	/// Returns whether no jobs await settlement.
	pub fn is_empty(&self) -> bool {
		self.pending.lock().is_empty()
	}
}

/// Locked, allocation-free view of jobs awaiting settlement.
pub struct PendingJobs<'a> {
	guard: MutexGuard<'a, BTreeMap<Str, JobRef>>,
}

impl PendingJobs<'_> {
	/// Iterates descriptors in stable job-identifier order.
	pub fn iter(&self) -> impl DoubleEndedIterator<Item = &JobRef> + ExactSizeIterator + Clone + '_ {
		self.guard.values()
	}

	/// Returns the number of jobs in this view.
	pub fn len(&self) -> usize {
		self.guard.len()
	}

	/// Returns whether this view contains no jobs.
	pub fn is_empty(&self) -> bool {
		self.guard.is_empty()
	}
}

#[cfg(test)]
mod tests {
	use std::{
		sync::atomic::{AtomicUsize, Ordering},
		thread,
	};

	use omp_tool::{ArtifactLifetime, ExpectedArtifact};

	use super::*;
	use crate::mailbox::{DrainPoint, Mailbox};

	fn job(id: &str, lifetime: ArtifactLifetime) -> JobRef {
		JobRef {
			id:       Str::from(id),
			artifact: ExpectedArtifact {
				description: Str::from("detached output"),
				media_type: None,
				lifetime,
			},
		}
	}

	#[test]
	fn pending_view_is_stable_and_duplicates_preserve_the_first_descriptor() {
		let mailbox = Mailbox::new();
		let board = JobBoard::new(mailbox.sender());
		assert!(board.register(job("job-b", ArtifactLifetime::Durable)));
		assert!(board.register(job("job-a", ArtifactLifetime::Session)));
		assert!(!board.register(job("job-a", ArtifactLifetime::Ephemeral)));

		let pending = board.pending();
		assert_eq!(pending.len(), 2);
		let mut jobs = pending.iter();
		assert_eq!(jobs.next().unwrap().id, "job-a");
		assert_eq!(jobs.next().unwrap().id, "job-b");
		assert_eq!(jobs.next(), None);
		assert_eq!(pending.iter().next().unwrap().artifact.lifetime, ArtifactLifetime::Session);
	}

	#[test]
	fn concurrent_settlement_enqueues_once_and_removes_pending_state() {
		let mut mailbox = Mailbox::new();
		let board = JobBoard::new(mailbox.sender());
		assert!(board.register(job("job-1", ArtifactLifetime::Session)));
		assert!(!board.settle("unknown", Item::default()).unwrap());

		let settled = AtomicUsize::new(0);
		thread::scope(|scope| {
			for seq in 0..8 {
				let board = &board;
				let settled = &settled;
				scope.spawn(move || {
					if board
						.settle("job-1", Item { seq, ..Item::default() })
						.unwrap()
					{
						settled.fetch_add(1, Ordering::Relaxed);
					}
				});
			}
		});

		assert_eq!(settled.load(Ordering::Relaxed), 1);
		assert!(board.is_empty());
		assert_eq!(mailbox.len(), 1);
		let interrupts = mailbox.drain(DrainPoint::TurnBoundary, false);
		assert_eq!(interrupts.len(), 1);
		assert_eq!(interrupts[0].class, InterruptClass::TurnBoundary);
		assert_eq!(interrupts[0].source, "job-board");
		assert!(!board.settle("job-1", Item::default()).unwrap());
		assert!(mailbox.is_empty());
	}
}
