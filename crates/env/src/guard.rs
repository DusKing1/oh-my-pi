use std::sync::atomic::{AtomicBool, Ordering};

use flume::Sender;

/// A nonblocking cancellation guard for one invocation or command request.
///
/// Dropping an armed guard queues cancellation for exactly
/// [`Self::request_id`]. The server-owned session containing an exec request is
/// not represented by this guard and therefore cannot be cancelled by it.
/// Detached work must call [`Self::relinquish`] explicitly before the guard is
/// dropped.
#[derive(Debug)]
pub struct RunGuard {
	state: GuardState,
}

#[derive(Debug)]
struct GuardState {
	request_id: u64,
	armed:      AtomicBool,
	cancel:     Sender<u64>,
}

impl RunGuard {
	pub(crate) fn new(request_id: u64, cancel: Sender<u64>) -> Self {
		Self { state: GuardState { request_id, armed: AtomicBool::new(true), cancel } }
	}

	/// Returns the request correlation identifier scoped by this guard.
	#[must_use]
	pub fn request_id(&self) -> u64 {
		self.state.request_id
	}

	/// Returns whether dropping this guard will request cancellation.
	#[must_use]
	pub fn is_armed(&self) -> bool {
		self.state.armed.load(Ordering::Acquire)
	}

	/// Queues cancellation now.
	///
	/// This operation never blocks. Repeated calls and a later drop are
	/// idempotent: at most one cancellation is queued.
	pub fn cancel(&self) {
		self.state.cancel();
	}

	/// Explicitly transfers responsibility for detached work to the server.
	///
	/// Consuming the guard without sending cancellation makes the ownership
	/// transition visible at the call site.
	pub fn relinquish(self) {
		self.state.disarm();
	}
}

impl Drop for RunGuard {
	fn drop(&mut self) {
		self.state.cancel();
	}
}

impl GuardState {
	fn disarm(&self) {
		self.armed.store(false, Ordering::Release);
	}

	fn cancel(&self) {
		if self
			.armed
			.compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
			.is_ok()
		{
			// This sender is always the unbounded guard-control queue owned by the
			// client dispatcher, so try_send is nonblocking and cannot be full.
			let _ = self.cancel.try_send(self.request_id);
		}
	}
}
