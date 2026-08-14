//! Session-owned eval kernel scheduling and lifecycle management.
//!
//! A session serializes cells, while independent sessions may execute in
//! parallel. Reset invalidates active and queued work from the previous epoch,
//! and unhealthy kernels are replaced before reuse.

use std::{
	collections::HashMap,
	fmt,
	future::Future,
	sync::{
		Arc,
		atomic::{AtomicU64, Ordering},
	},
	time::Duration,
};

use omp_core::Str;
use parking_lot::Mutex;
use tokio::sync::Mutex as AsyncMutex;
use tokio_util::sync::CancellationToken;

use super::idle_timeout::TimeoutHandle;

/// Whether a retained kernel is safe to receive another cell.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum KernelHealth {
	Ready,
	Poisoned,
	Dead,
}

/// Per-cell controls available to the runtime and authenticated host bridge.
#[derive(Clone, Debug)]
pub(crate) struct CellControl {
	pub(crate) cancel:  CancellationToken,
	pub(crate) timeout: TimeoutHandle,
}

impl CellControl {
	/// Runs host-assisted work without charging it to the cell compute timeout.
	pub(crate) async fn host_wait<F: Future>(&self, operation: F) -> F::Output {
		self.timeout.host_wait(operation).await
	}
}

/// Persistent kernel contract consumed by the session scheduler.
///
/// `interrupt` is synchronous so dropping a caller future still interrupts its
/// cell. Implementations perform graceful/asynchronous cleanup in `shutdown`.
pub(crate) trait SessionKernel: Send + Sync + 'static {
	type Output: Send + 'static;
	type Error: Send + 'static;

	fn health(&self) -> KernelHealth;
	fn interrupt(&self);
	async fn execute(&self, code: Str, control: CellControl) -> Result<Self::Output, Self::Error>;
	async fn shutdown(&self);
}

/// Starts an isolated kernel for a logical eval session.
pub(crate) trait KernelFactory: Send + Sync + 'static {
	type Kernel: SessionKernel;

	async fn start(
		&self,
		session: &Str,
		cancel: CancellationToken,
	) -> Result<Arc<Self::Kernel>, <Self::Kernel as SessionKernel>::Error>;
}

/// One scheduled eval cell.
#[derive(Debug)]
pub(crate) struct ExecutionRequest {
	pub(crate) code:    Str,
	pub(crate) timeout: Option<Duration>,
	pub(crate) reset:   bool,
	pub(crate) cancel:  CancellationToken,
}

/// Scheduler-level completion failures.
#[derive(Debug)]
pub(crate) enum LifecycleError<E> {
	Cancelled,
	TimedOut { kernel_killed: bool },
	Superseded,
	ShuttingDown,
	Kernel(E),
}

impl<E: fmt::Display> fmt::Display for LifecycleError<E> {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::Cancelled => formatter.write_str("eval cell cancelled"),
			Self::TimedOut { kernel_killed: true } => formatter.write_str(
				"eval cell timed out and the kernel was unresponsive to interrupt; the kernel has \
				 been killed and will be recreated on the next call.",
			),
			Self::TimedOut { kernel_killed: false } => formatter.write_str(
				"eval cell timed out; kernel interrupted but remains running. Reset the kernel via { \
				 reset: true } if state appears corrupted.",
			),
			Self::Superseded => formatter.write_str("eval cell superseded by kernel reset"),
			Self::ShuttingDown => formatter.write_str("eval session is shutting down"),
			Self::Kernel(error) => write!(formatter, "eval kernel failed: {error}"),
		}
	}
}

/// Lifecycle timing knobs. Production uses the pi-compatible five-second
/// interrupt grace; tests can choose a shorter deterministic boundary.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LifecycleConfig {
	pub(crate) interrupt_grace: Duration,
	pub(crate) shutdown_grace:  Duration,
}

impl Default for LifecycleConfig {
	fn default() -> Self {
		Self { interrupt_grace: Duration::from_secs(5), shutdown_grace: Duration::from_secs(5) }
	}
}

/// Persistent kernel pool owned by one application/session environment.
pub(crate) struct KernelRegistry<F: KernelFactory> {
	factory:  Arc<F>,
	config:   LifecycleConfig,
	sessions: Mutex<HashMap<Str, Arc<Session<F::Kernel>>>>,
	shutdown: CancellationToken,
}

struct Session<K: SessionKernel> {
	key:       Str,
	gate:      AsyncMutex<()>,
	kernel:    AsyncMutex<Option<Arc<K>>>,
	epoch:     Mutex<Epoch>,
	active:    Mutex<Option<ActiveCell<K>>>,
	next_cell: AtomicU64,
}

#[derive(Clone)]
struct EpochLease {
	id:     u64,
	cancel: CancellationToken,
}

struct Epoch {
	id:     u64,
	cancel: CancellationToken,
}

struct ActiveCell<K: SessionKernel> {
	id:     u64,
	cancel: CancellationToken,
	kernel: Arc<K>,
}

struct ActiveLease<K: SessionKernel> {
	session:   Arc<Session<K>>,
	id:        u64,
	completed: bool,
}

impl<F: KernelFactory> KernelRegistry<F> {
	#[must_use]
	pub(crate) fn new(factory: Arc<F>, config: LifecycleConfig) -> Self {
		Self {
			factory,
			config,
			sessions: Mutex::new(HashMap::new()),
			shutdown: CancellationToken::new(),
		}
	}

	/// Executes one cell on its persistent session kernel.
	pub(crate) async fn execute(
		&self,
		session_key: Str,
		request: ExecutionRequest,
	) -> Result<
		<F::Kernel as SessionKernel>::Output,
		LifecycleError<<F::Kernel as SessionKernel>::Error>,
	> {
		if self.shutdown.is_cancelled() {
			return Err(LifecycleError::ShuttingDown);
		}
		if request.cancel.is_cancelled() {
			return Err(LifecycleError::Cancelled);
		}

		let session = self.session(session_key);
		let epoch = session.begin_epoch(request.reset);
		let gate = tokio::select! {
			biased;
			() = self.shutdown.cancelled() => return Err(LifecycleError::ShuttingDown),
			() = request.cancel.cancelled() => return Err(LifecycleError::Cancelled),
			() = epoch.cancel.cancelled() => return Err(LifecycleError::Superseded),
			gate = session.gate.lock() => gate,
		};

		if !session.is_epoch(&epoch) {
			return Err(LifecycleError::Superseded);
		}
		if request.reset {
			self.teardown_kernel(&session).await;
			if !session.is_epoch(&epoch) {
				return Err(LifecycleError::Superseded);
			}
		}

		let mut retry_crashed_kernel = true;
		loop {
			let kernel = self.ensure_kernel(&session, &epoch, &request).await?;
			let result = self
				.run_cell(&session, &epoch, &request, Arc::clone(&kernel))
				.await;
			match result {
				Err(LifecycleError::Kernel(error))
					if retry_crashed_kernel && kernel.health() != KernelHealth::Ready =>
				{
					retry_crashed_kernel = false;
					self.discard_kernel(&session, &kernel).await;
					if !session.is_epoch(&epoch) {
						return Err(LifecycleError::Superseded);
					}
					drop(error);
				},
				other => {
					drop(gate);
					return other;
				},
			}
		}
	}

	/// Cancels work and tears down every retained kernel. The scheduler itself
	/// spawns no background tasks, so completion means all owned state is
	/// drained.
	pub(crate) async fn shutdown_all(&self) {
		self.shutdown.cancel();
		let sessions = self
			.sessions
			.lock()
			.drain()
			.map(|(_, session)| session)
			.collect::<Vec<_>>();
		for session in &sessions {
			session.invalidate();
			session.cancel_active();
		}
		for session in sessions {
			let _gate = tokio::time::timeout(self.config.shutdown_grace, session.gate.lock())
				.await
				.ok();
			self.teardown_kernel(&session).await;
		}
	}

	#[cfg(test)]
	fn session_count(&self) -> usize {
		self.sessions.lock().len()
	}

	fn session(&self, key: Str) -> Arc<Session<F::Kernel>> {
		let mut sessions = self.sessions.lock();
		Arc::clone(
			sessions
				.entry(key.clone())
				.or_insert_with(|| Arc::new(Session::new(key))),
		)
	}

	async fn ensure_kernel(
		&self,
		session: &Arc<Session<F::Kernel>>,
		epoch: &EpochLease,
		request: &ExecutionRequest,
	) -> Result<Arc<F::Kernel>, LifecycleError<<F::Kernel as SessionKernel>::Error>> {
		let mut slot = session.kernel.lock().await;
		if let Some(kernel) = slot.as_ref() {
			if kernel.health() == KernelHealth::Ready {
				return Ok(Arc::clone(kernel));
			}
		}
		if let Some(stale) = slot.take() {
			stale.interrupt();
			let _ = tokio::time::timeout(self.config.shutdown_grace, stale.shutdown()).await;
		}

		let start_cancel = CancellationToken::new();
		let start = self.factory.start(&session.key, start_cancel.clone());
		tokio::pin!(start);
		let started = tokio::select! {
			biased;
			() = self.shutdown.cancelled() => {
				start_cancel.cancel();
				return Err(LifecycleError::ShuttingDown);
			},
			() = request.cancel.cancelled() => {
				start_cancel.cancel();
				return Err(LifecycleError::Cancelled);
			},
			() = epoch.cancel.cancelled() => {
				start_cancel.cancel();
				return Err(LifecycleError::Superseded);
			},
			result = &mut start => result.map_err(LifecycleError::Kernel)?,
		};
		if !session.is_epoch(epoch) {
			started.interrupt();
			let _ = tokio::time::timeout(self.config.shutdown_grace, started.shutdown()).await;
			return Err(LifecycleError::Superseded);
		}
		*slot = Some(Arc::clone(&started));
		Ok(started)
	}

	async fn run_cell(
		&self,
		session: &Arc<Session<F::Kernel>>,
		epoch: &EpochLease,
		request: &ExecutionRequest,
		kernel: Arc<F::Kernel>,
	) -> Result<
		<F::Kernel as SessionKernel>::Output,
		LifecycleError<<F::Kernel as SessionKernel>::Error>,
	> {
		let timeout = TimeoutHandle::new(request.timeout);
		let control = CellControl { cancel: CancellationToken::new(), timeout: timeout.clone() };
		let mut active = session.activate(Arc::clone(&kernel), control.cancel.clone());
		let execution = kernel.execute(request.code.clone(), control.clone());
		tokio::pin!(execution);

		enum Stop {
			Cancelled,
			TimedOut,
			Superseded,
			Shutdown,
		}

		let stop = tokio::select! {
			biased;
			() = self.shutdown.cancelled() => Some(Stop::Shutdown),
			() = request.cancel.cancelled() => Some(Stop::Cancelled),
			() = epoch.cancel.cancelled() => Some(Stop::Superseded),
			() = control.cancel.cancelled() => Some(Stop::Superseded),
			() = timeout.expired() => Some(Stop::TimedOut),
			result = &mut execution => {
				active.complete();
				timeout.dispose();
				return result.map_err(LifecycleError::Kernel);
			},
		}
		.expect("every non-execution branch has a stop reason");

		control.cancel.cancel();
		kernel.interrupt();
		timeout.dispose();
		let settled = tokio::time::timeout(self.config.interrupt_grace, &mut execution)
			.await
			.is_ok();
		active.complete();
		let kernel_killed = if settled {
			false
		} else {
			kernel.interrupt();
			let _ = tokio::time::timeout(self.config.shutdown_grace, kernel.shutdown()).await;
			self.clear_kernel(session, &kernel).await;
			true
		};

		Err(match stop {
			Stop::Cancelled => LifecycleError::Cancelled,
			Stop::TimedOut => LifecycleError::TimedOut { kernel_killed },
			Stop::Superseded => LifecycleError::Superseded,
			Stop::Shutdown => LifecycleError::ShuttingDown,
		})
	}

	async fn discard_kernel(&self, session: &Session<F::Kernel>, kernel: &Arc<F::Kernel>) {
		self.clear_kernel(session, kernel).await;
		kernel.interrupt();
		let _ = tokio::time::timeout(self.config.shutdown_grace, kernel.shutdown()).await;
	}

	async fn clear_kernel(&self, session: &Session<F::Kernel>, kernel: &Arc<F::Kernel>) {
		let mut slot = session.kernel.lock().await;
		if slot
			.as_ref()
			.is_some_and(|current| Arc::ptr_eq(current, kernel))
		{
			slot.take();
		}
	}

	async fn teardown_kernel(&self, session: &Session<F::Kernel>) {
		let kernel = session.kernel.lock().await.take();
		if let Some(kernel) = kernel {
			kernel.interrupt();
			let _ = tokio::time::timeout(self.config.shutdown_grace, kernel.shutdown()).await;
		}
	}
}

impl<F: KernelFactory> Drop for KernelRegistry<F> {
	fn drop(&mut self) {
		self.shutdown.cancel();
		for session in self.sessions.get_mut().values() {
			session.invalidate();
			session.cancel_active();
		}
	}
}

impl<K: SessionKernel> Session<K> {
	fn new(key: Str) -> Self {
		Self {
			key,
			gate: AsyncMutex::new(()),
			kernel: AsyncMutex::new(None),
			epoch: Mutex::new(Epoch { id: 0, cancel: CancellationToken::new() }),
			active: Mutex::new(None),
			next_cell: AtomicU64::new(1),
		}
	}

	fn begin_epoch(&self, reset: bool) -> EpochLease {
		let mut epoch = self.epoch.lock();
		if reset {
			epoch.cancel.cancel();
			epoch.id = epoch.id.wrapping_add(1);
			epoch.cancel = CancellationToken::new();
			drop(epoch);
			self.cancel_active();
			let epoch = self.epoch.lock();
			return EpochLease { id: epoch.id, cancel: epoch.cancel.clone() };
		}
		EpochLease { id: epoch.id, cancel: epoch.cancel.clone() }
	}

	fn is_epoch(&self, expected: &EpochLease) -> bool {
		let epoch = self.epoch.lock();
		epoch.id == expected.id && !expected.cancel.is_cancelled()
	}

	fn invalidate(&self) {
		let mut epoch = self.epoch.lock();
		epoch.cancel.cancel();
		epoch.id = epoch.id.wrapping_add(1);
		epoch.cancel = CancellationToken::new();
	}

	fn activate(self: &Arc<Self>, kernel: Arc<K>, cancel: CancellationToken) -> ActiveLease<K> {
		let id = self.next_cell.fetch_add(1, Ordering::Relaxed);
		*self.active.lock() = Some(ActiveCell { id, cancel, kernel });
		ActiveLease { session: Arc::clone(self), id, completed: false }
	}

	fn cancel_active(&self) {
		if let Some(active) = self.active.lock().as_ref() {
			active.cancel.cancel();
			active.kernel.interrupt();
		}
	}
}

impl<K: SessionKernel> ActiveLease<K> {
	fn complete(&mut self) {
		let mut active = self.session.active.lock();
		if active.as_ref().is_some_and(|cell| cell.id == self.id) {
			active.take();
		}
		self.completed = true;
	}
}

impl<K: SessionKernel> Drop for ActiveLease<K> {
	fn drop(&mut self) {
		if self.completed {
			return;
		}
		let mut active = self.session.active.lock();
		if active.as_ref().is_some_and(|cell| cell.id == self.id) {
			if let Some(cell) = active.take() {
				cell.cancel.cancel();
				cell.kernel.interrupt();
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use std::sync::atomic::{AtomicU8, AtomicUsize};

	use tokio::sync::Notify;

	use super::*;

	#[derive(Debug, Clone, Eq, PartialEq)]
	struct FakeError(&'static str);

	impl fmt::Display for FakeError {
		fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
			formatter.write_str(self.0)
		}
	}

	struct FakeFactory {
		starts:         AtomicUsize,
		shutdowns:      Arc<AtomicUsize>,
		interrupts:     Arc<AtomicUsize>,
		concurrent:     Arc<AtomicUsize>,
		max_concurrent: Arc<AtomicUsize>,
		started:        Arc<Notify>,
		release:        Arc<Notify>,
		crashes_left:   Arc<AtomicUsize>,
	}

	struct FakeKernel {
		state:          Mutex<i64>,
		health:         AtomicU8,
		shutdowns:      Arc<AtomicUsize>,
		interrupts:     Arc<AtomicUsize>,
		concurrent:     Arc<AtomicUsize>,
		max_concurrent: Arc<AtomicUsize>,
		started:        Arc<Notify>,
		release:        Arc<Notify>,
		crashes_left:   Arc<AtomicUsize>,
	}

	impl FakeFactory {
		fn new() -> Arc<Self> {
			Arc::new(Self {
				starts:         AtomicUsize::new(0),
				shutdowns:      Arc::new(AtomicUsize::new(0)),
				interrupts:     Arc::new(AtomicUsize::new(0)),
				concurrent:     Arc::new(AtomicUsize::new(0)),
				max_concurrent: Arc::new(AtomicUsize::new(0)),
				started:        Arc::new(Notify::new()),
				release:        Arc::new(Notify::new()),
				crashes_left:   Arc::new(AtomicUsize::new(0)),
			})
		}
	}

	impl KernelFactory for FakeFactory {
		type Kernel = FakeKernel;

		async fn start(
			&self,
			_session: &Str,
			cancel: CancellationToken,
		) -> Result<Arc<FakeKernel>, FakeError> {
			if cancel.is_cancelled() {
				return Err(FakeError("start cancelled"));
			}
			self.starts.fetch_add(1, Ordering::SeqCst);
			Ok(Arc::new(FakeKernel {
				state:          Mutex::new(0),
				health:         AtomicU8::new(0),
				shutdowns:      Arc::clone(&self.shutdowns),
				interrupts:     Arc::clone(&self.interrupts),
				concurrent:     Arc::clone(&self.concurrent),
				max_concurrent: Arc::clone(&self.max_concurrent),
				started:        Arc::clone(&self.started),
				release:        Arc::clone(&self.release),
				crashes_left:   Arc::clone(&self.crashes_left),
			}))
		}
	}

	impl SessionKernel for FakeKernel {
		type Error = FakeError;
		type Output = i64;

		fn health(&self) -> KernelHealth {
			match self.health.load(Ordering::SeqCst) {
				0 => KernelHealth::Ready,
				1 => KernelHealth::Poisoned,
				_ => KernelHealth::Dead,
			}
		}

		fn interrupt(&self) {
			self.interrupts.fetch_add(1, Ordering::SeqCst);
		}

		async fn execute(&self, code: Str, control: CellControl) -> Result<i64, FakeError> {
			let concurrent = self.concurrent.fetch_add(1, Ordering::SeqCst) + 1;
			self.max_concurrent.fetch_max(concurrent, Ordering::SeqCst);
			struct Running<'a>(&'a AtomicUsize);
			impl Drop for Running<'_> {
				fn drop(&mut self) {
					self.0.fetch_sub(1, Ordering::SeqCst);
				}
			}
			let _running = Running(&self.concurrent);
			self.started.notify_one();

			match code.as_str() {
				"get" => Ok(*self.state.lock()),
				"inc" => {
					let mut state = self.state.lock();
					*state += 1;
					Ok(*state)
				},
				"host-wait" => {
					control
						.host_wait(tokio::time::sleep(Duration::from_millis(70)))
						.await;
					tokio::time::sleep(Duration::from_millis(5)).await;
					Ok(7)
				},
				"compute" => {
					tokio::select! {
						() = tokio::time::sleep(Duration::from_millis(100)) => Ok(1),
						() = control.cancel.cancelled() => Err(FakeError("interrupted")),
					}
				},
				"hold" => {
					tokio::select! {
						() = self.release.notified() => Ok(1),
						() = control.cancel.cancelled() => Err(FakeError("interrupted")),
					}
				},
				"crash"
					if self
						.crashes_left
						.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |left| left.checked_sub(1))
						.is_ok() =>
				{
					self.health.store(2, Ordering::SeqCst);
					Err(FakeError("crashed"))
				},
				"crash" => Ok(9),
				_ => Err(FakeError("unknown code")),
			}
		}

		async fn shutdown(&self) {
			self.health.store(2, Ordering::SeqCst);
			self.shutdowns.fetch_add(1, Ordering::SeqCst);
		}
	}

	fn request(code: &'static str) -> ExecutionRequest {
		ExecutionRequest {
			code:    Str::new_static(code),
			timeout: Some(Duration::from_secs(1)),
			reset:   false,
			cancel:  CancellationToken::new(),
		}
	}

	fn registry(factory: Arc<FakeFactory>) -> Arc<KernelRegistry<FakeFactory>> {
		Arc::new(KernelRegistry::new(factory, LifecycleConfig {
			interrupt_grace: Duration::from_millis(20),
			shutdown_grace:  Duration::from_millis(20),
		}))
	}

	#[tokio::test]
	async fn persistent_kernel_reset_tears_down_and_recreates_only_that_session() {
		let factory = FakeFactory::new();
		let registry = registry(Arc::clone(&factory));
		let key = Str::new_static("session-a:py");
		assert_eq!(registry.execute(key.clone(), request("inc")).await.unwrap(), 1);
		assert_eq!(registry.execute(key.clone(), request("get")).await.unwrap(), 1);

		let mut reset = request("get");
		reset.reset = true;
		assert_eq!(registry.execute(key, reset).await.unwrap(), 0);
		assert_eq!(factory.starts.load(Ordering::SeqCst), 2);
		assert_eq!(factory.shutdowns.load(Ordering::SeqCst), 1);
	}

	#[tokio::test]
	async fn host_assisted_wait_pauses_timeout_but_compute_resumes_it() {
		let factory = FakeFactory::new();
		let registry = registry(factory);
		let mut paused = request("host-wait");
		paused.timeout = Some(Duration::from_millis(25));
		assert_eq!(
			registry
				.execute(Str::new_static("paused:py"), paused)
				.await
				.unwrap(),
			7
		);

		let mut compute = request("compute");
		compute.timeout = Some(Duration::from_millis(25));
		assert!(matches!(
			registry
				.execute(Str::new_static("paused:py"), compute)
				.await,
			Err(LifecycleError::TimedOut { kernel_killed: false })
		));
	}

	#[tokio::test]
	async fn explicit_cancellation_interrupts_without_discarding_a_responsive_kernel() {
		let factory = FakeFactory::new();
		let registry = registry(Arc::clone(&factory));
		let cancel = CancellationToken::new();
		let mut held = request("hold");
		held.cancel = cancel.clone();
		let running = {
			let registry = Arc::clone(&registry);
			tokio::spawn(async move { registry.execute(Str::new_static("cancel:py"), held).await })
		};
		factory.started.notified().await;
		cancel.cancel();
		assert!(matches!(running.await.unwrap(), Err(LifecycleError::Cancelled)));
		assert_eq!(
			registry
				.execute(Str::new_static("cancel:py"), request("get"))
				.await
				.unwrap(),
			0
		);
		assert_eq!(factory.starts.load(Ordering::SeqCst), 1);
		assert!(factory.interrupts.load(Ordering::SeqCst) > 0);
	}

	#[tokio::test]
	async fn cells_are_sequenced_and_reset_invalidates_active_and_queued_work() {
		let factory = FakeFactory::new();
		let registry = registry(Arc::clone(&factory));
		let key = Str::new_static("sequence:py");
		let first = {
			let registry = Arc::clone(&registry);
			let key = key.clone();
			tokio::spawn(async move { registry.execute(key, request("hold")).await })
		};
		factory.started.notified().await;
		let queued = {
			let registry = Arc::clone(&registry);
			let key = key.clone();
			tokio::spawn(async move { registry.execute(key, request("inc")).await })
		};
		tokio::task::yield_now().await;
		let mut reset_request = request("get");
		reset_request.reset = true;
		let reset = {
			let registry = Arc::clone(&registry);
			tokio::spawn(async move { registry.execute(key, reset_request).await })
		};

		assert!(matches!(first.await.unwrap(), Err(LifecycleError::Superseded)));
		assert!(matches!(queued.await.unwrap(), Err(LifecycleError::Superseded)));
		assert_eq!(reset.await.unwrap().unwrap(), 0);
		assert_eq!(factory.max_concurrent.load(Ordering::SeqCst), 1);
	}

	#[tokio::test]
	async fn crashed_kernel_is_recreated_and_the_cell_is_retried_once() {
		let factory = FakeFactory::new();
		factory.crashes_left.store(1, Ordering::SeqCst);
		let registry = registry(Arc::clone(&factory));
		assert_eq!(
			registry
				.execute(Str::new_static("crash:py"), request("crash"))
				.await
				.unwrap(),
			9
		);
		assert_eq!(factory.starts.load(Ordering::SeqCst), 2);
		assert_eq!(factory.shutdowns.load(Ordering::SeqCst), 1);
	}

	#[tokio::test]
	async fn shutdown_cancels_active_cells_and_drains_all_session_state() {
		let factory = FakeFactory::new();
		let registry = registry(Arc::clone(&factory));
		let running = {
			let registry = Arc::clone(&registry);
			tokio::spawn(async move {
				registry
					.execute(Str::new_static("shutdown:py"), request("hold"))
					.await
			})
		};
		factory.started.notified().await;
		registry.shutdown_all().await;
		assert!(matches!(running.await.unwrap(), Err(LifecycleError::ShuttingDown)));
		assert_eq!(registry.session_count(), 0);
		assert_eq!(factory.concurrent.load(Ordering::SeqCst), 0);
		assert_eq!(factory.shutdowns.load(Ordering::SeqCst), 1);
	}
}
