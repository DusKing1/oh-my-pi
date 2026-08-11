//! Render-loop stall detection with an optional background probe.

use std::{
	sync::Arc,
	thread::{self, JoinHandle},
	time::{Duration, Instant},
};

use omp_core::Str;
use parking_lot::{Condvar, Mutex};

const STALL_THRESHOLD: Duration = Duration::from_millis(250);
const SYSTEM_SLEEP_THRESHOLD: Duration = Duration::from_secs(60);

/// A render-loop stall returned by [`LoopWatchdogCore::check`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StallReport {
	/// Time elapsed since the render loop's most recent tick.
	pub elapsed: Duration,
	/// Phase label active when the stall was detected.
	pub phase:   Str,
}

/// Deterministic state machine underlying [`LoopWatchdog`].
///
/// Times are monotonic durations from any caller-chosen epoch. A continuous
/// stall is reported once, and gaps over 60 seconds are treated as system
/// sleep.
#[derive(Debug)]
pub struct LoopWatchdogCore {
	last_tick: Duration,
	phase:     Str,
	reported:  bool,
}

impl LoopWatchdogCore {
	/// Create a watchdog core whose last successful render-loop tick was `now`.
	#[must_use]
	pub fn new(now: Duration) -> Self {
		Self { last_tick: now, phase: "unknown".into(), reported: false }
	}

	/// Record progress by the render loop at monotonic time `now`.
	pub const fn tick(&mut self, now: Duration) {
		self.last_tick = now;
		self.reported = false;
	}

	/// Set the label attached to a subsequently detected stall.
	pub fn set_phase(&mut self, phase: impl Into<Str>) {
		self.phase = phase.into();
	}

	/// Check for a newly detected stall at monotonic time `now`.
	///
	/// Returns one report after 250 ms without a tick. Further checks stay
	/// silent until [`Self::tick`] records progress. Gaps over 60 seconds are
	/// suppressed.
	#[must_use]
	pub fn check(&mut self, now: Duration) -> Option<StallReport> {
		let elapsed = now.saturating_sub(self.last_tick);
		if elapsed > SYSTEM_SLEEP_THRESHOLD {
			self.reported = false;
			return None;
		}
		if elapsed <= STALL_THRESHOLD {
			self.reported = false;
			return None;
		}
		if self.reported {
			return None;
		}
		self.reported = true;
		Some(StallReport { elapsed, phase: self.phase.clone() })
	}
}

struct Shared {
	core:     LoopWatchdogCore,
	stopping: bool,
}

/// Background render-loop watchdog.
///
/// Call [`Self::tick`] after each successful render-loop iteration and update
/// [`Self::set_phase`] when entering a diagnostic phase. A background probe
/// invokes the supplied callback once per continuous stall longer than 250 ms.
/// Gaps longer than 60 seconds are ignored as probable system sleep.
pub struct LoopWatchdog {
	origin: Instant,
	shared: Arc<(Mutex<Shared>, Condvar)>,
	worker: Option<JoinHandle<()>>,
}

impl LoopWatchdog {
	/// Start a watchdog and send detected stalls to `report`.
	#[must_use]
	pub fn new(report: impl Fn(Duration, &str) + Send + 'static) -> Self {
		let origin = Instant::now();
		let shared = Arc::new((
			Mutex::new(Shared { core: LoopWatchdogCore::new(Duration::ZERO), stopping: false }),
			Condvar::new(),
		));
		let worker_shared = Arc::clone(&shared);
		let worker = thread::spawn(move || {
			loop {
				let (lock, wake) = &*worker_shared;
				let mut guard = lock.lock();
				wake.wait_for(&mut guard, STALL_THRESHOLD);
				if guard.stopping {
					break;
				}
				let stall = guard.core.check(origin.elapsed());
				drop(guard);
				if let Some(stall) = stall {
					report(stall.elapsed, &stall.phase);
				}
			}
		});

		Self { origin, shared, worker: Some(worker) }
	}

	/// Record progress by the render loop.
	pub fn tick(&self) {
		let (lock, wake) = &*self.shared;
		let mut shared = lock.lock();
		shared.core.tick(self.origin.elapsed());
		drop(shared);
		wake.notify_one();
	}

	/// Set the phase label attached to a subsequently detected stall.
	pub fn set_phase(&self, phase: impl Into<Str>) {
		let (lock, _) = &*self.shared;
		let mut shared = lock.lock();
		shared.core.set_phase(phase);
	}

	/// Stop the background probe and wait for it to exit.
	pub fn stop(&mut self) {
		let Some(worker) = self.worker.take() else {
			return;
		};
		let (lock, wake) = &*self.shared;
		let mut shared = lock.lock();
		shared.stopping = true;
		drop(shared);
		wake.notify_one();
		let _ = worker.join();
	}
}

impl Drop for LoopWatchdog {
	fn drop(&mut self) {
		self.stop();
	}
}

#[cfg(test)]
mod tests {
	use std::time::Duration;

	use super::LoopWatchdogCore;

	#[test]
	fn reports_a_300ms_stall_with_phase() {
		let mut watchdog = LoopWatchdogCore::new(Duration::ZERO);
		watchdog.set_phase("render");
		let report = watchdog
			.check(Duration::from_millis(300))
			.expect("300 ms should exceed the stall threshold");
		assert_eq!(report.elapsed, Duration::from_millis(300));
		assert_eq!(report.phase, "render");
		assert_eq!(watchdog.check(Duration::from_millis(400)), None);
	}

	#[test]
	fn ignores_a_90s_system_sleep_gap() {
		let mut watchdog = LoopWatchdogCore::new(Duration::ZERO);
		watchdog.set_phase("render");
		assert_eq!(watchdog.check(Duration::from_secs(90)), None);
	}

	#[test]
	fn stays_silent_under_normal_ticks() {
		let mut watchdog = LoopWatchdogCore::new(Duration::ZERO);
		for millis in [200, 400, 600, 800] {
			let now = Duration::from_millis(millis);
			assert_eq!(watchdog.check(now), None);
			watchdog.tick(now);
		}
	}
}
