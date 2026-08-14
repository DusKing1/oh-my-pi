//! Pauseable watchdog for eval cells.
//!
//! The timeout is an idle-work window, not a wall-clock deadline. Host bridge
//! calls pause it, and the final matching resume starts a fresh full window.

use std::{future::Future, sync::Arc, time::Duration};

use parking_lot::Mutex;
use tokio::{sync::Notify, time::Instant};

/// A cloneable handle shared with host-assisted eval operations.
#[derive(Clone, Debug)]
pub struct TimeoutHandle {
	inner: Arc<Inner>,
}

#[derive(Debug)]
struct Inner {
	state:   Mutex<State>,
	changed: Notify,
}

#[derive(Debug)]
struct State {
	window:      Option<Duration>,
	deadline:    Option<Instant>,
	pause_depth: usize,
	disposed:    bool,
	generation:  u64,
}

impl TimeoutHandle {
	/// Creates a watchdog. `None` disables timeout accounting.
	#[must_use]
	pub fn new(window: Option<Duration>) -> Self {
		let window = normalize_window(window);
		Self {
			inner: Arc::new(Inner {
				state:   Mutex::new(State {
					window,
					deadline: window.map(|window| Instant::now() + window),
					pause_depth: 0,
					disposed: false,
					generation: 0,
				}),
				changed: Notify::new(),
			}),
		}
	}

	/// Starts timeout accounting for a new cell on a session-shared handle.
	///
	/// Outstanding pause guards from the previous cell are invalidated.
	pub fn restart(&self, window: Option<Duration>) {
		let mut state = self.inner.state.lock();
		state.generation = state.generation.wrapping_add(1);
		state.window = normalize_window(window);
		state.deadline = state.window.map(|window| Instant::now() + window);
		state.pause_depth = 0;
		state.disposed = false;
		self.inner.changed.notify_waiters();
	}

	/// Pauses timeout accounting until the returned guard is dropped.
	///
	/// Pauses are reference-counted so concurrent bridge calls cannot resume the
	/// cell while another host-assisted wait is still in flight.
	#[must_use]
	pub fn pause(&self) -> TimeoutPause {
		let mut state = self.inner.state.lock();
		let generation = state.generation;
		if !state.disposed {
			state.pause_depth = state.pause_depth.saturating_add(1);
			if state.pause_depth == 1 {
				state.deadline = None;
				self.inner.changed.notify_waiters();
			}
		}
		TimeoutPause { timeout: Some(self.clone()), generation }
	}

	/// Runs one host operation outside the eval cell's compute budget.
	pub async fn host_wait<F: Future>(&self, operation: F) -> F::Output {
		let _pause = self.pause();
		operation.await
	}

	/// Stops the watchdog. Safe to call more than once.
	pub fn dispose(&self) {
		let mut state = self.inner.state.lock();
		if state.disposed {
			return;
		}
		state.disposed = true;
		state.deadline = None;
		state.pause_depth = 0;
		state.generation = state.generation.wrapping_add(1);
		self.inner.changed.notify_waiters();
	}

	/// Resolves when the active timeout window expires. A disabled or disposed
	/// watchdog remains pending so it can be used directly in `select!`.
	pub async fn expired(&self) {
		loop {
			// Register before inspecting state so a pause/resume between the
			// inspection and select cannot be lost.
			let changed = self.inner.changed.notified();
			let deadline = {
				let state = self.inner.state.lock();
				if state.disposed || state.window.is_none() {
					None
				} else {
					state.deadline
				}
			};

			let Some(deadline) = deadline else {
				changed.await;
				continue;
			};

			tokio::pin!(changed);
			let sleep = tokio::time::sleep_until(deadline);
			tokio::pin!(sleep);
			tokio::select! {
				() = &mut sleep => {
					let mut state = self.inner.state.lock();
					if !state.disposed && state.pause_depth == 0 && state.deadline.is_some_and(|current| current <= Instant::now()) {
						state.disposed = true;
						state.deadline = None;
						return;
					}
				},
				() = &mut changed => {},
			}
		}
	}

	fn resume(&self, generation: u64) {
		let mut state = self.inner.state.lock();
		if state.disposed || state.generation != generation || state.pause_depth == 0 {
			return;
		}
		state.pause_depth -= 1;
		if state.pause_depth != 0 {
			return;
		}
		state.deadline = state.window.map(|window| Instant::now() + window);
		self.inner.changed.notify_waiters();
	}
}

/// RAII pause token. Dropping the last outstanding token starts a fresh full
/// timeout window.
#[derive(Debug)]
pub struct TimeoutPause {
	timeout:    Option<TimeoutHandle>,
	generation: u64,
}

impl Drop for TimeoutPause {
	fn drop(&mut self) {
		if let Some(timeout) = self.timeout.take() {
			timeout.resume(self.generation);
		}
	}
}

fn normalize_window(window: Option<Duration>) -> Option<Duration> {
	match window {
		Some(window) => Some(window.max(Duration::from_millis(1))),
		None => None,
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[tokio::test]
	async fn host_wait_pauses_then_resumes_with_a_fresh_window() {
		let timeout = TimeoutHandle::new(Some(Duration::from_millis(40)));
		timeout
			.host_wait(tokio::time::sleep(Duration::from_millis(80)))
			.await;

		assert!(
			tokio::time::timeout(Duration::from_millis(15), timeout.expired())
				.await
				.is_err()
		);
		assert!(
			tokio::time::timeout(Duration::from_millis(80), timeout.expired())
				.await
				.is_ok()
		);
	}

	#[tokio::test]
	async fn nested_pauses_resume_only_after_the_final_guard_drops() {
		let timeout = TimeoutHandle::new(Some(Duration::from_millis(30)));
		let outer = timeout.pause();
		let inner = timeout.pause();
		tokio::time::sleep(Duration::from_millis(40)).await;
		drop(outer);
		assert!(
			tokio::time::timeout(Duration::from_millis(40), timeout.expired())
				.await
				.is_err()
		);
		drop(inner);
		assert!(
			tokio::time::timeout(Duration::from_millis(15), timeout.expired())
				.await
				.is_err()
		);
		assert!(
			tokio::time::timeout(Duration::from_millis(60), timeout.expired())
				.await
				.is_ok()
		);
	}

	#[tokio::test]
	async fn pause_wins_a_deadline_boundary_race() {
		let timeout = TimeoutHandle::new(Some(Duration::from_millis(25)));
		tokio::time::sleep(Duration::from_millis(20)).await;
		let pause = timeout.pause();
		tokio::time::sleep(Duration::from_millis(20)).await;
		assert!(
			tokio::time::timeout(Duration::from_millis(10), timeout.expired())
				.await
				.is_err()
		);
		drop(pause);
		assert!(
			tokio::time::timeout(Duration::from_millis(60), timeout.expired())
				.await
				.is_ok()
		);
	}

	#[tokio::test]
	async fn disabled_and_disposed_watchdogs_never_expire() {
		let disabled = TimeoutHandle::new(None);
		assert!(
			tokio::time::timeout(Duration::from_millis(10), disabled.expired())
				.await
				.is_err()
		);

		let disposed = TimeoutHandle::new(Some(Duration::from_millis(1)));
		disposed.dispose();
		assert!(
			tokio::time::timeout(Duration::from_millis(10), disposed.expired())
				.await
				.is_err()
		);
	}

	#[tokio::test]
	async fn restart_reuses_a_session_handle_without_accepting_stale_pause_guards() {
		let timeout = TimeoutHandle::new(Some(Duration::from_millis(20)));
		let stale = timeout.pause();
		timeout.dispose();
		timeout.restart(Some(Duration::from_millis(35)));
		drop(stale);

		assert!(
			tokio::time::timeout(Duration::from_millis(15), timeout.expired())
				.await
				.is_err()
		);
		assert!(
			tokio::time::timeout(Duration::from_millis(60), timeout.expired())
				.await
				.is_ok()
		);
	}

	#[tokio::test]
	async fn an_expiration_waiter_survives_dispose_and_observes_restart() {
		let timeout = TimeoutHandle::new(Some(Duration::from_millis(20)));
		timeout.dispose();
		let expired = timeout.expired();
		tokio::pin!(expired);
		assert!(
			tokio::time::timeout(Duration::from_millis(10), &mut expired)
				.await
				.is_err()
		);
		timeout.restart(Some(Duration::from_millis(25)));
		assert!(
			tokio::time::timeout(Duration::from_millis(60), &mut expired)
				.await
				.is_ok()
		);
	}
}
