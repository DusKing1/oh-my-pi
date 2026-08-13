//! Shared lifecycle, admission, cancellation, and memory accounting for local
//! engines.

use std::{
	ops::Deref,
	sync::{
		Arc,
		atomic::{AtomicU64, AtomicUsize, Ordering},
	},
	time::{Duration, Instant},
};

use omp_core::Str;
use parking_lot::Mutex;
use tokio_util::sync::CancellationToken;

/// Stable local-runtime failure category.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LocalErrorKind {
	/// The host cannot run this backend.
	Unsupported,
	/// A request option is invalid.
	InvalidInput,
	/// Admission or memory capacity is exhausted.
	Overloaded,
	/// The caller cancelled the operation.
	Cancelled,
	/// A model artifact failed verification.
	Artifact,
	/// The inference engine failed.
	Backend,
}

/// Secret-free failure returned by local backends.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[error("{message}")]
pub struct LocalError {
	/// Stable failure category.
	pub kind:    LocalErrorKind,
	/// Diagnostic safe for receipts and logs.
	pub message: Str,
}

impl LocalError {
	/// Constructs a local failure.
	pub fn new(kind: LocalErrorKind, message: impl Into<Str>) -> Self {
		Self { kind, message: message.into() }
	}

	/// Constructs a cancellation failure.
	pub fn cancelled() -> Self {
		Self::new(LocalErrorKind::Cancelled, "local inference was cancelled")
	}
}

/// Result returned by local inference components.
pub type LocalResult<T> = Result<T, LocalError>;

/// Structured host and model availability evidence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AvailabilityEvidence {
	/// Whether the backend can accept work now.
	pub available: bool,
	/// Stable evidence code such as `unsupported_operating_system`.
	pub code:      Str,
	/// Human-readable, secret-free detail.
	pub detail:    Str,
}

impl AvailabilityEvidence {
	/// Constructs positive availability evidence.
	pub fn available(detail: impl Into<Str>) -> Self {
		Self { available: true, code: "available".into(), detail: detail.into() }
	}

	/// Constructs negative availability evidence.
	pub fn unavailable(code: impl Into<Str>, detail: impl Into<Str>) -> Self {
		Self { available: false, code: code.into(), detail: detail.into() }
	}
}

/// Cancellation handle shared by every local adapter.
pub type LocalCancellation = CancellationToken;

/// Process-local memory admission pool.
#[derive(Debug)]
pub struct MemoryPool {
	limit: usize,
	used:  AtomicUsize,
}

impl MemoryPool {
	/// Creates a pool with a hard byte limit.
	pub const fn new(limit: usize) -> Self {
		Self { limit, used: AtomicUsize::new(0) }
	}

	/// Returns the configured byte limit.
	pub const fn limit(&self) -> usize {
		self.limit
	}

	/// Returns currently reserved bytes.
	pub fn used(&self) -> usize {
		self.used.load(Ordering::Acquire)
	}

	/// Atomically reserves bytes or returns typed overload evidence.
	pub fn reserve(self: &Arc<Self>, bytes: usize) -> LocalResult<MemoryReservation> {
		let mut current = self.used.load(Ordering::Acquire);
		loop {
			let Some(next) = current.checked_add(bytes) else {
				return Err(LocalError::new(LocalErrorKind::Overloaded, "memory reservation overflow"));
			};
			if next > self.limit {
				return Err(LocalError::new(
					LocalErrorKind::Overloaded,
					"local inference memory budget is exhausted",
				));
			}
			match self
				.used
				.compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Acquire)
			{
				Ok(_) => return Ok(MemoryReservation { pool: Arc::clone(self), bytes }),
				Err(observed) => current = observed,
			}
		}
	}
}

/// RAII memory reservation released on drop.
#[derive(Debug)]
pub struct MemoryReservation {
	pool:  Arc<MemoryPool>,
	bytes: usize,
}

impl MemoryReservation {
	/// Returns reserved bytes.
	pub const fn bytes(&self) -> usize {
		self.bytes
	}
}

impl Drop for MemoryReservation {
	fn drop(&mut self) {
		self.pool.used.fetch_sub(self.bytes, Ordering::AcqRel);
	}
}

/// Non-waiting admission gate whose failure is explicit backpressure.
#[derive(Debug)]
pub struct AdmissionControl {
	limit:  usize,
	active: AtomicUsize,
}

impl AdmissionControl {
	/// Creates a gate allowing `limit` concurrent operations.
	pub fn new(limit: usize) -> LocalResult<Self> {
		if limit == 0 {
			return Err(LocalError::new(
				LocalErrorKind::InvalidInput,
				"local admission limit must be non-zero",
			));
		}
		Ok(Self { limit, active: AtomicUsize::new(0) })
	}

	/// Attempts admission without hiding queueing or overload.
	pub fn try_acquire(self: &Arc<Self>) -> LocalResult<AdmissionPermit> {
		let mut current = self.active.load(Ordering::Acquire);
		loop {
			if current >= self.limit {
				return Err(LocalError::new(
					LocalErrorKind::Overloaded,
					"local inference concurrency limit is active",
				));
			}
			match self.active.compare_exchange_weak(
				current,
				current + 1,
				Ordering::AcqRel,
				Ordering::Acquire,
			) {
				Ok(_) => return Ok(AdmissionPermit(Arc::clone(self))),
				Err(observed) => current = observed,
			}
		}
	}

	/// Returns the number of active operations.
	pub fn active(&self) -> usize {
		self.active.load(Ordering::Acquire)
	}
}

/// Admission permit released on drop.
#[derive(Debug)]
pub struct AdmissionPermit(Arc<AdmissionControl>);

impl Drop for AdmissionPermit {
	fn drop(&mut self) {
		self.0.active.fetch_sub(1, Ordering::AcqRel);
	}
}

struct Loaded<E> {
	engine:    E,
	_memory:   MemoryReservation,
	instance:  u64,
	last_used: Instant,
}

struct RuntimeState<E> {
	loaded: Option<Loaded<E>>,
}

struct RuntimeInner<E> {
	state:         Mutex<RuntimeState<E>>,
	loader:        Arc<dyn Fn() -> LocalResult<E> + Send + Sync>,
	memory:        Arc<MemoryPool>,
	model_bytes:   usize,
	admission:     Arc<AdmissionControl>,
	idle_timeout:  Duration,
	next_instance: AtomicU64,
	next_request:  AtomicU64,
}

/// Shared lazy-loading runtime for one isolated local model configuration.
pub struct LocalRuntime<E> {
	inner: Arc<RuntimeInner<E>>,
}

impl<E> Clone for LocalRuntime<E> {
	fn clone(&self) -> Self {
		Self { inner: Arc::clone(&self.inner) }
	}
}

impl<E> LocalRuntime<E> {
	/// Creates a lazy runtime with explicit memory and concurrency bounds.
	pub fn new(
		loader: impl Fn() -> LocalResult<E> + Send + Sync + 'static,
		memory: Arc<MemoryPool>,
		model_bytes: usize,
		max_concurrency: usize,
		idle_timeout: Duration,
	) -> LocalResult<Self> {
		if max_concurrency != 1 {
			return Err(LocalError::new(
				LocalErrorKind::InvalidInput,
				"serialized local runtimes require a concurrency limit of one",
			));
		}
		Ok(Self {
			inner: Arc::new(RuntimeInner {
				state: Mutex::new(RuntimeState { loaded: None }),
				loader: Arc::new(loader),
				memory,
				model_bytes,
				admission: Arc::new(AdmissionControl::new(max_concurrency)?),
				idle_timeout,
				next_instance: AtomicU64::new(1),
				next_request: AtomicU64::new(1),
			}),
		})
	}

	/// Acquires admission and lazily loads the model after reserving its memory.
	pub fn acquire(&self, cancel: &LocalCancellation) -> LocalResult<RuntimeLease<E>> {
		if cancel.is_cancelled() {
			return Err(LocalError::cancelled());
		}
		let permit = self.inner.admission.try_acquire()?;
		let now = Instant::now();
		let mut state = self.inner.state.lock();
		if state.loaded.is_none() {
			let reservation = self.inner.memory.reserve(self.inner.model_bytes)?;
			let engine = (self.inner.loader)()?;
			let instance = self.inner.next_instance.fetch_add(1, Ordering::Relaxed);
			state.loaded = Some(Loaded { engine, _memory: reservation, instance, last_used: now });
		}
		let request = self.inner.next_request.fetch_add(1, Ordering::Relaxed);
		let instance = state.loaded.as_ref().expect("loaded above").instance;
		drop(state);
		Ok(RuntimeLease { runtime: self.clone(), _permit: permit, request, instance })
	}

	/// Unloads an inactive model after its configured idle interval.
	pub fn unload_if_idle(&self, now: Instant) -> bool {
		if self.inner.admission.active() != 0 {
			return false;
		}
		let mut state = self.inner.state.lock();
		// Admission is acquired before the runtime-state lock. Recheck while
		// holding that lock so a racing acquisition cannot lose its engine.
		if self.inner.admission.active() != 0 {
			return false;
		}
		if state.loaded.as_ref().is_some_and(|loaded| {
			now.saturating_duration_since(loaded.last_used) >= self.inner.idle_timeout
		}) {
			state.loaded = None;
			true
		} else {
			false
		}
	}

	/// Returns whether the model is currently loaded.
	pub fn is_loaded(&self) -> bool {
		self.inner.state.lock().loaded.is_some()
	}
}

/// One admitted operation and its model-instance isolation receipt.
pub struct RuntimeLease<E> {
	runtime:  LocalRuntime<E>,
	_permit:  AdmissionPermit,
	request:  u64,
	instance: u64,
}

impl<E> RuntimeLease<E> {
	/// Runs a synchronous engine operation while preserving model isolation.
	pub fn with_engine<T>(
		&self,
		operation: impl FnOnce(&mut E) -> LocalResult<T>,
	) -> LocalResult<T> {
		let mut state = self.runtime.inner.state.lock();
		let loaded = state.loaded.as_mut().ok_or_else(|| {
			LocalError::new(LocalErrorKind::Backend, "local model was unloaded during an operation")
		})?;
		if loaded.instance != self.instance {
			return Err(LocalError::new(
				LocalErrorKind::Backend,
				"local model instance changed during an operation",
			));
		}
		let output = operation(&mut loaded.engine);
		loaded.last_used = Instant::now();
		output
	}

	/// Returns a receipt identifying this request and loaded model instance.
	pub const fn receipt(&self) -> LocalExecutionReceipt {
		LocalExecutionReceipt { request: self.request, model_instance: self.instance }
	}
}

/// Non-secret evidence that one request ran on one isolated model instance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LocalExecutionReceipt {
	/// Process-local monotonically increasing request sequence.
	pub request:        u64,
	/// Process-local model load generation.
	pub model_instance: u64,
}

impl Deref for LocalError {
	type Target = str;

	fn deref(&self) -> &Self::Target {
		self.message.as_str()
	}
}
