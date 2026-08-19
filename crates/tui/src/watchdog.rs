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
const CPU_BUSY_RATIO: f64 = 0.5;

#[cfg(unix)]
fn process_cpu_time() -> Option<Duration> {
	use nix::sys::{
		resource::{UsageWho, getrusage},
		time::TimeValLike,
	};

	let usage = getrusage(UsageWho::RUSAGE_SELF).ok()?;
	let micros = usage
		.user_time()
		.num_microseconds()
		.checked_add(usage.system_time().num_microseconds())?;
	u64::try_from(micros).ok().map(Duration::from_micros)
}

#[cfg(windows)]
fn process_cpu_time() -> Option<Duration> {
	use windows_sys::Win32::{
		Foundation::FILETIME,
		System::Threading::{GetCurrentProcess, GetProcessTimes},
	};

	let mut created = FILETIME::default();
	let mut exited = FILETIME::default();
	let mut kernel = FILETIME::default();
	let mut user = FILETIME::default();
	// SAFETY: GetCurrentProcess returns a valid pseudo-handle, and every output
	// pointer refers to a live FILETIME for the duration of the call.
	let success = unsafe {
		GetProcessTimes(
			GetCurrentProcess(),
			&raw mut created,
			&raw mut exited,
			&raw mut kernel,
			&raw mut user,
		)
	};
	if success == 0 {
		return None;
	}
	let ticks =
		|time: FILETIME| (u64::from(time.dwHighDateTime) << 32) | u64::from(time.dwLowDateTime);
	Some(Duration::from_nanos(
		ticks(kernel)
			.saturating_add(ticks(user))
			.saturating_mul(100),
	))
}

#[cfg(not(any(unix, windows)))]
const fn process_cpu_time() -> Option<Duration> {
	None
}

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
/// Wall and CPU times are monotonic durations from caller-chosen epochs. A
/// continuous stall is reported once. Gaps over 60 seconds are treated as
/// system sleep only when the process burned less than half of the wall gap on
/// CPU.
#[derive(Debug)]
pub struct LoopWatchdogCore {
	last_tick:     Duration,
	last_cpu_time: Duration,
	phase:         Str,
	reported:      bool,
}

impl LoopWatchdogCore {
	/// Create a watchdog core with the latest wall and process CPU times.
	#[must_use]
	pub fn new(now: Duration, cpu_time: Duration) -> Self {
		Self {
			last_tick:     now,
			last_cpu_time: cpu_time,
			phase:         "unknown".into(),
			reported:      false,
		}
	}

	/// Record render-loop progress with the current wall and process CPU times.
	pub const fn tick(&mut self, now: Duration, cpu_time: Duration) {
		self.last_tick = now;
		self.last_cpu_time = cpu_time;
		self.reported = false;
	}

	/// Set the label attached to a subsequently detected stall.
	pub fn set_phase(&mut self, phase: impl Into<Str>) {
		self.phase = phase.into();
	}

	/// Check for a newly detected stall at the current wall and process CPU
	/// times.
	///
	/// Returns one report after 250 ms without a tick. Further checks stay
	/// silent until [`Self::tick`] records progress. A gap over 60 seconds is
	/// suppressed only when the process consumed little CPU during it.
	#[must_use]
	pub fn check(&mut self, now: Duration, cpu_time: Duration) -> Option<StallReport> {
		let elapsed = now.saturating_sub(self.last_tick);
		let cpu_elapsed = cpu_time.saturating_sub(self.last_cpu_time);
		if elapsed > SYSTEM_SLEEP_THRESHOLD && cpu_elapsed < elapsed.mul_f64(CPU_BUSY_RATIO) {
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
/// Long gaps are ignored as system sleep only when process CPU time stayed low.
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
		let initial_cpu = process_cpu_time().unwrap_or(Duration::ZERO);
		let shared = Arc::new((
			Mutex::new(Shared {
				core:     LoopWatchdogCore::new(Duration::ZERO, initial_cpu),
				stopping: false,
			}),
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
				let now = origin.elapsed();
				let cpu_time = process_cpu_time().unwrap_or_else(|| initial_cpu.saturating_add(now));
				let stall = guard.core.check(now, cpu_time);
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
		let now = self.origin.elapsed();
		let cpu_time =
			process_cpu_time().unwrap_or_else(|| shared.core.last_cpu_time.saturating_add(now));
		shared.core.tick(now, cpu_time);
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
		let mut watchdog = LoopWatchdogCore::new(Duration::ZERO, Duration::ZERO);
		watchdog.set_phase("render");
		let report = watchdog
			.check(Duration::from_millis(300), Duration::from_millis(300))
			.expect("300 ms should exceed the stall threshold");
		assert_eq!(report.elapsed, Duration::from_millis(300));
		assert_eq!(report.phase, "render");
		assert_eq!(watchdog.check(Duration::from_millis(400), Duration::from_millis(400)), None,);
	}

	#[test]
	fn ignores_a_90s_system_sleep_gap() {
		let mut watchdog = LoopWatchdogCore::new(Duration::ZERO, Duration::ZERO);
		watchdog.set_phase("render");
		assert_eq!(watchdog.check(Duration::from_secs(90), Duration::from_millis(3)), None,);
	}

	#[test]
	fn classifies_long_gap_by_process_cpu_time() {
		let mut busy = LoopWatchdogCore::new(Duration::ZERO, Duration::ZERO);
		let report = busy
			.check(Duration::from_secs(90), Duration::from_secs(80))
			.expect("a long gap spent running on CPU is a stall");
		assert_eq!(report.elapsed, Duration::from_secs(90));

		let mut suspended = LoopWatchdogCore::new(Duration::ZERO, Duration::ZERO);
		assert_eq!(
			suspended.check(Duration::from_secs(90), Duration::from_millis(3)),
			None,
			"a suspend/resume gap burns no process CPU",
		);
	}

	#[test]
	fn stays_silent_under_normal_ticks() {
		let mut watchdog = LoopWatchdogCore::new(Duration::ZERO, Duration::ZERO);
		for millis in [200, 400, 600, 800] {
			let now = Duration::from_millis(millis);
			assert_eq!(watchdog.check(now, now), None);
			watchdog.tick(now, now);
		}
	}
}
