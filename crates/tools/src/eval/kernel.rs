//! Persistent embedded-CPython implementation of [`super::EvalExec`].
//!
//! Each opened session owns a dedicated worker thread and Python globals
//! dictionary. Cells are serialized by that worker, so names persist in source
//! order; resetting atomically replaces the dictionary before the next cell.

use std::{
	collections::HashMap,
	ffi::{c_long, c_ulong},
	sync::{
		Arc, LazyLock,
		atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering},
	},
	thread,
	time::Duration,
};

use bytes::Bytes;
use flume::{Receiver, Sender};
use omp_core::{CowBytes, Str};
use omp_py::Engine;
use parking_lot::Mutex;
use pyo3::{
	Py, PyAny, PyResult, Python, ffi,
	ffi::c_str,
	prelude::*,
	pyclass, pymethods,
	types::{PyAnyMethods, PyDict, PyDictMethods, PyModule},
};
use serde_json::Value;
use tokio::sync::Mutex as AsyncMutex;

use super::{
	CellOutcome, CellStatus, CellValue, EvalExec, EvalRun, Fault, OutputChannel, PythonException,
	RunCompletion, RunEvent, RunRequest, Session, Update,
};

const BOOTSTRAP: &std::ffi::CStr = c_str!(
	r#"
import ast as _omp_ast
import asyncio as _omp_asyncio
import inspect as _omp_inspect
import json as _omp_json
import sys as _omp_sys
import threading as _omp_threading
import time as _omp_time
import traceback as _omp_traceback

_OMP_TLA = getattr(_omp_ast, "PyCF_ALLOW_TOP_LEVEL_AWAIT", 0x2000)
_OMP_TIMEOUT_MESSAGE = "OMP eval cell timed out"

def _omp_new_namespace():
    return {
        "__name__": "__main__",
        "__builtins__": __builtins__,
        "__omp_async_runner": _omp_asyncio.Runner(),
    }

def _omp_compile(source):
    module = _omp_ast.parse(source, "<cell>", "exec")
    if not module.body:
        return None, None
    last = module.body[-1]
    if isinstance(last, _omp_ast.Expr):
        body = _omp_ast.Module(body=module.body[:-1], type_ignores=[])
        expr = _omp_ast.Expression(body=last.value)
        _omp_ast.copy_location(expr, last)
        return (compile(body, "<cell>", "exec", flags=_OMP_TLA),
                compile(expr, "<cell>", "eval", flags=_OMP_TLA))
    return compile(module, "<cell>", "exec", flags=_OMP_TLA), None

async def _omp_run_async(code, ns, want_value):
    if code.co_flags & _omp_inspect.CO_COROUTINE:
        value = await eval(code, ns)
        return value if want_value else None
    if want_value:
        return eval(code, ns)
    exec(code, ns)
    return None

def _omp_run(code, ns, want_value):
    if code is None:
        return None
    return ns["__omp_async_runner"].run(_omp_run_async(code, ns, want_value))

def _omp_run_cell(source, ns, timeout_seconds):
    started = _omp_time.perf_counter()
    deadline = None if timeout_seconds is None else started + timeout_seconds
    pause_depth = 0
    timeout_lock = _omp_threading.RLock()
    previous_trace = _omp_sys.gettrace()

    def timeout_pause():
        nonlocal deadline, pause_depth
        with timeout_lock:
            pause_depth += 1
            deadline = None

    def timeout_resume():
        nonlocal deadline, pause_depth
        with timeout_lock:
            if pause_depth == 0:
                return
            pause_depth -= 1
            if pause_depth == 0:
                deadline = None if timeout_seconds is None else _omp_time.perf_counter() + timeout_seconds

    def timeout_trace(frame, event, arg):
        with timeout_lock:
            expired = deadline is not None and _omp_time.perf_counter() >= deadline
        if expired:
            raise TimeoutError(_OMP_TIMEOUT_MESSAGE)
        return timeout_trace

    ns["__omp_timeout_pause__"] = timeout_pause
    ns["__omp_timeout_resume__"] = timeout_resume

    outcome = "complete"
    result_text = None
    result_json = None
    error_name = None
    error_message = None
    error_traceback = []
    try:
        if deadline is not None:
            _omp_sys.settrace(timeout_trace)
        body, expr = _omp_compile(source)
        _omp_run(body, ns, False)
        if expr is not None:
            value = _omp_run(expr, ns, True)
            if value is not None:
                result_text = repr(value)
                try:
                    result_json = _omp_json.dumps(value, allow_nan=False, separators=(",", ":"))
                except (TypeError, ValueError, OverflowError):
                    result_json = None
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            outcome = "cancelled"
        elif isinstance(exc, TimeoutError) and str(exc) == _OMP_TIMEOUT_MESSAGE:
            outcome = "timeout"
        else:
            outcome = "error"
        error_name = type(exc).__name__
        error_message = str(exc)
        error_traceback = _omp_traceback.format_exception(type(exc), exc, exc.__traceback__)
    finally:
        _omp_sys.settrace(previous_trace)
        ns.pop("__omp_timeout_pause__", None)
        ns.pop("__omp_timeout_resume__", None)

    return {
        "outcome": outcome,
        "result_text": result_text,
        "result_json": result_json,
        "error_name": error_name,
        "error_message": error_message,
        "error_traceback": error_traceback,
        "duration_ms": int((_omp_time.perf_counter() - started) * 1000),
    }
"#
);

/// Cloneable embedded Python resource shared by one or more eval tool values.
///
/// The caller owns OMP's process-wide [`Engine`] and must initialize it exactly
/// once. `EmbeddedPython` only creates isolated persistent session workers.
#[derive(Clone)]
pub struct EmbeddedPython {
	inner: Arc<Inner>,
}

struct Inner {
	engine:       Arc<Engine>,
	next_session: AtomicU64,
	next_cell:    AtomicU64,
	installer:    Arc<dyn NamespaceInstaller>,
	workers:      Mutex<HashMap<Bytes, Arc<Worker>>>,
}

struct Worker {
	commands: Sender<Command>,
	state:    Arc<WorkerState>,
	enqueue:  AsyncMutex<()>,
}

struct WorkerState {
	engine:    Arc<Engine>,
	thread_id: AtomicI64,
	epoch:     AtomicU64,
	alive:     AtomicBool,
	active:    Mutex<Option<ActiveCell>>,
	installer: Arc<dyn NamespaceInstaller>,
}

struct ActiveCell {
	cell_id:   Bytes,
	cancelled: Arc<AtomicBool>,
}

struct Command {
	cell_id:   Bytes,
	request:   RunRequest,
	events:    Sender<Result<RunEvent, Fault>>,
	cancelled: Arc<AtomicBool>,
	epoch:     u64,
}
unsafe extern "C" {
	fn PyThread_get_thread_ident() -> c_ulong;
}

impl WorkerState {
	fn interrupt_if_active(&self, target: &Arc<AtomicBool>) -> Result<(), Fault> {
		let active = self.active.lock();
		let Some(active) = active.as_ref() else {
			return Ok(());
		};
		if !Arc::ptr_eq(&active.cancelled, target) {
			return Ok(());
		}
		self.installer.cancel_cell(&active.cell_id);
		self.interrupt_thread()
	}

	fn cancel_active(&self) -> Result<(), Fault> {
		let active = self.active.lock();
		let Some(active) = active.as_ref() else {
			return Ok(());
		};
		active.cancelled.store(true, Ordering::Release);
		self.installer.cancel_cell(&active.cell_id);
		self.interrupt_thread()
	}

	fn interrupt_thread(&self) -> Result<(), Fault> {
		let id = self.thread_id.load(Ordering::Acquire);
		if id == 0 {
			return Ok(());
		}
		let changed = self.engine.attach(|_| {
			// SAFETY: `id` is CPython's identifier for this live worker thread and
			// `PyExc_KeyboardInterrupt` is an immortal runtime-owned exception type.
			// The caller is attached while CPython selects the target thread state.
			let changed =
				unsafe { ffi::PyThreadState_SetAsyncExc(id as c_long, ffi::PyExc_KeyboardInterrupt) };
			if changed > 1 {
				// SAFETY: passing NULL clears the exception set by the preceding call.
				unsafe { ffi::PyThreadState_SetAsyncExc(id as c_long, std::ptr::null_mut()) };
			}
			changed
		});
		if changed == 1 {
			return Ok(());
		}
		Err(Fault::Resource {
			operation: Str::from("cancel"),
			message:   Str::from("CPython did not identify exactly one active eval thread"),
		})
	}
}

impl Drop for Worker {
	fn drop(&mut self) {
		let _ = self.state.cancel_active();
	}
}

const STDOUT_ROUTER_ATTR: &str = "__omp_stdout_router__";
const STDERR_ROUTER_ATTR: &str = "__omp_stderr_router__";
static OUTPUT_ROUTER_INIT: Mutex<()> = Mutex::new(());
static OUTPUT_ROUTERS: LazyLock<OutputRouters> = LazyLock::new(|| OutputRouters {
	stdout: Arc::new(OutputRouterState::new(OutputChannel::Stdout)),
	stderr: Arc::new(OutputRouterState::new(OutputChannel::Stderr)),
});

struct CaptureSink {
	events:   Sender<Result<RunEvent, Fault>>,
	sequence: Arc<AtomicU64>,
	buffer:   Mutex<Vec<u8>>,
}

impl CaptureSink {
	fn new(events: Sender<Result<RunEvent, Fault>>, sequence: Arc<AtomicU64>) -> Self {
		Self { events, sequence, buffer: Mutex::new(Vec::new()) }
	}

	fn write(&self, channel: OutputChannel, text: &str) -> usize {
		if text.is_empty() {
			return 0;
		}
		self.buffer.lock().extend_from_slice(text.as_bytes());
		let sequence = self.sequence.fetch_add(1, Ordering::Relaxed);
		let _ = self.events.send(Ok(RunEvent::Output(Update {
			channel,
			data: CowBytes::owned(Bytes::copy_from_slice(text.as_bytes())),
			sequence,
		})));
		text.chars().count()
	}

	fn snapshot(&self) -> Vec<u8> {
		self.buffer.lock().clone()
	}
}

struct OutputRouterState {
	channel: OutputChannel,
	active:  Mutex<HashMap<c_ulong, Arc<CaptureSink>>>,
}

impl OutputRouterState {
	fn new(channel: OutputChannel) -> Self {
		Self { channel, active: Mutex::new(HashMap::new()) }
	}

	fn bind(&self, thread_id: c_ulong, sink: Arc<CaptureSink>) {
		self.active.lock().insert(thread_id, sink);
	}

	fn unbind(&self, thread_id: c_ulong, sink: &Arc<CaptureSink>) {
		let mut active = self.active.lock();
		if active
			.get(&thread_id)
			.is_some_and(|current| Arc::ptr_eq(current, sink))
		{
			active.remove(&thread_id);
		}
	}

	fn write(&self, text: &str) -> usize {
		let thread_id = unsafe { PyThread_get_thread_ident() };
		let sink = self.active.lock().get(&thread_id).cloned();
		sink.map_or_else(|| text.chars().count(), |sink| sink.write(self.channel, text))
	}
}

#[pyclass(frozen)]
struct OutputRouter {
	state: Arc<OutputRouterState>,
}

#[pymethods]
impl OutputRouter {
	fn write(&self, text: &str) -> usize {
		self.state.write(text)
	}

	fn flush(&self) {}
}

#[derive(Clone)]
struct OutputRouters {
	stdout: Arc<OutputRouterState>,
	stderr: Arc<OutputRouterState>,
}

impl OutputRouters {
	fn bind(&self, thread_id: c_ulong, events: Sender<Result<RunEvent, Fault>>) -> OutputBinding {
		let sequence = Arc::new(AtomicU64::new(0));
		let stdout = Arc::new(CaptureSink::new(events.clone(), Arc::clone(&sequence)));
		let stderr = Arc::new(CaptureSink::new(events, sequence));
		self.stdout.bind(thread_id, Arc::clone(&stdout));
		self.stderr.bind(thread_id, Arc::clone(&stderr));
		OutputBinding {
			thread_id,
			stdout_router: Arc::clone(&self.stdout),
			stderr_router: Arc::clone(&self.stderr),
			stdout,
			stderr,
		}
	}
}

struct OutputBinding {
	thread_id:     c_ulong,
	stdout_router: Arc<OutputRouterState>,
	stderr_router: Arc<OutputRouterState>,
	stdout:        Arc<CaptureSink>,
	stderr:        Arc<CaptureSink>,
}

impl Drop for OutputBinding {
	fn drop(&mut self) {
		self.stdout_router.unbind(self.thread_id, &self.stdout);
		self.stderr_router.unbind(self.thread_id, &self.stderr);
	}
}

#[pyclass]
struct DisplayCollector {
	entries: Mutex<Vec<(Py<PyAny>, bool)>>,
}

impl DisplayCollector {
	fn new() -> Self {
		Self { entries: Mutex::new(Vec::new()) }
	}

	fn clear(&self) {
		self.entries.lock().clear();
	}

	fn drain(&self, py: Python<'_>) -> PyResult<Vec<super::DisplayOutput>> {
		let entries = std::mem::take(&mut *self.entries.lock());
		let mut outputs = Vec::with_capacity(entries.len());
		for (value, raw) in entries {
			if raw {
				let bound = value.bind(py);
				if let Ok(bundle) = bound.cast::<PyDict>() {
					if let Some(status) = bundle.get_item("application/x-omp-status")? {
						if let Some(event) = python_to_json(py, &status)? {
							outputs.push(super::DisplayOutput::Status { event });
						}
						continue;
					}
					if let Some(json) = bundle.get_item("application/json")? {
						if let Some(data) = python_to_json(py, &json)? {
							outputs.push(super::DisplayOutput::Json { data });
						}
						continue;
					}
					if let Some(markdown) = bundle.get_item("text/markdown")? {
						outputs.push(super::DisplayOutput::Markdown {
							text: Str::from(markdown.extract::<String>()?),
						});
						continue;
					}
					if let Some(text) = bundle.get_item("text/plain")? {
						outputs.push(super::DisplayOutput::Markdown {
							text: Str::from(text.extract::<String>()?),
						});
					}
				}
			} else if let Some(data) = python_to_json(py, value.bind(py))? {
				outputs.push(super::DisplayOutput::Json { data });
			} else {
				outputs.push(super::DisplayOutput::Markdown {
					text: Str::from(value.bind(py).repr()?.extract::<String>()?),
				});
			}
		}
		Ok(outputs)
	}
}

#[pymethods]
impl DisplayCollector {
	#[pyo3(signature = (value, raw=false))]
	fn __call__(&self, value: Py<PyAny>, raw: bool) {
		self.entries.lock().push((value, raw));
	}
}

/// Active embedded-Python cell with ordered events and cooperative interrupt.
pub struct EmbeddedRun {
	events:    Receiver<Result<RunEvent, Fault>>,
	state:     Arc<WorkerState>,
	cancelled: Arc<AtomicBool>,
}

/// Installs session-scoped helpers into a newly-created Python namespace.
///
/// The app adapter uses this seam to inject its authenticated host bridge and
/// normative prelude without introducing an `omp-tools` → `omp-app` cycle.
/// Installation runs once at session creation and again after every reset.
pub trait NamespaceInstaller: Send + Sync + 'static {
	/// Adds helpers to `globals`; existing user state is always absent here.
	fn install(&self, py: Python<'_>, globals: &Bound<'_, PyDict>) -> PyResult<()>;

	/// Activates per-cell bridge credentials.
	fn begin_cell(
		&self,
		_py: Python<'_>,
		_globals: &Bound<'_, PyDict>,
		_cell_id: &Bytes,
		_timeout: Option<Duration>,
	) -> PyResult<()> {
		Ok(())
	}

	/// Revokes per-cell bridge credentials and timeout accounting.
	fn end_cell(
		&self,
		_py: Python<'_>,
		_globals: &Bound<'_, PyDict>,
		_cell_id: &Bytes,
	) -> PyResult<()> {
		Ok(())
	}

	/// Cancels host work owned by an interrupted cell before Python receives its
	/// interrupt.
	fn cancel_cell(&self, _cell_id: &Bytes) {}
}

#[derive(Debug)]
struct EmptyNamespaceInstaller;

impl NamespaceInstaller for EmptyNamespaceInstaller {
	fn install(&self, _py: Python<'_>, _globals: &Bound<'_, PyDict>) -> PyResult<()> {
		Ok(())
	}
}

impl EmbeddedPython {
	/// Creates a Python eval resource over the already-booted embedded runtime.
	///
	/// This constructor installs no host helpers. Production app wiring should
	/// use [`Self::with_installer`] so the authenticated bridge and prelude are
	/// present from the first cell.
	pub fn new(engine: Arc<Engine>) -> Self {
		Self::with_installer(engine, Arc::new(EmptyNamespaceInstaller))
	}

	/// Creates a Python eval resource with a namespace bootstrap installer.
	pub fn with_installer(engine: Arc<Engine>, installer: Arc<dyn NamespaceInstaller>) -> Self {
		Self {
			inner: Arc::new(Inner {
				engine,
				installer,
				next_session: AtomicU64::new(1),
				next_cell: AtomicU64::new(1),
				workers: Mutex::new(HashMap::new()),
			}),
		}
	}

	fn worker(&self, session: &Session) -> Result<Arc<Worker>, Fault> {
		let mut workers = self.inner.workers.lock();
		let current = workers
			.get(&session.id)
			.cloned()
			.ok_or_else(|| Fault::SessionLost { message: Str::from("unknown Python session") })?;
		if current.state.alive.load(Ordering::Acquire) {
			return Ok(current);
		}
		let label = String::from_utf8_lossy(session.id.as_ref());
		let replacement = self.spawn_worker(&label)?;
		workers.insert(session.id.clone(), Arc::clone(&replacement));
		Ok(replacement)
	}

	fn spawn_worker(&self, label: &str) -> Result<Arc<Worker>, Fault> {
		let (commands, receiver) = flume::unbounded();
		let engine = Arc::clone(&self.inner.engine);
		let installer = Arc::clone(&self.inner.installer);
		let state = Arc::new(WorkerState {
			engine:    Arc::clone(&engine),
			thread_id: AtomicI64::new(0),
			epoch:     AtomicU64::new(0),
			alive:     AtomicBool::new(true),
			active:    Mutex::new(None),
			installer: Arc::clone(&installer),
		});
		let worker =
			Arc::new(Worker { commands, state: Arc::clone(&state), enqueue: AsyncMutex::new(()) });
		thread::Builder::new()
			.name(format!("omp-eval-py-{label}"))
			.spawn(move || worker_main(&engine, &state, receiver, installer.as_ref()))
			.map_err(|error| Fault::Resource {
				operation: Str::from("open_session"),
				message:   Str::from(error.to_string()),
			})?;
		Ok(worker)
	}
}

impl EvalExec for EmbeddedPython {
	type Run = EmbeddedRun;

	async fn open_session(&self) -> Result<Session, Fault> {
		let number = self.inner.next_session.fetch_add(1, Ordering::Relaxed);
		let id = Bytes::from(format!("py-{number}"));
		let worker = self.spawn_worker(&number.to_string())?;
		self.inner.workers.lock().insert(id.clone(), worker);
		Ok(Session { id })
	}

	async fn run<'a>(
		&'a self,
		session: &'a Session,
		request: RunRequest,
	) -> Result<Self::Run, Fault> {
		let worker = self.worker(session)?;
		let _enqueue = worker.enqueue.lock().await;
		let epoch = if request.reset {
			let epoch = worker
				.state
				.epoch
				.fetch_add(1, Ordering::AcqRel)
				.wrapping_add(1);
			worker.state.cancel_active()?;
			epoch
		} else {
			worker.state.epoch.load(Ordering::Acquire)
		};
		let number = self.inner.next_cell.fetch_add(1, Ordering::Relaxed);
		let cell_id =
			Bytes::from(format!("{}:cell-{number}", String::from_utf8_lossy(session.id.as_ref())));
		let (events, receiver) = flume::unbounded();
		let cancelled = Arc::new(AtomicBool::new(false));
		worker
			.commands
			.send_async(Command { cell_id, request, events, cancelled: Arc::clone(&cancelled), epoch })
			.await
			.map_err(|_| Fault::SessionLost {
				message: Str::from("Python worker stopped before accepting the cell"),
			})?;
		Ok(EmbeddedRun { events: receiver, state: Arc::clone(&worker.state), cancelled })
	}
}

impl EvalRun for EmbeddedRun {
	async fn next_event(&mut self) -> Result<Option<RunEvent>, Fault> {
		match self.events.recv_async().await {
			Ok(event) => event.map(Some),
			Err(_) => Ok(None),
		}
	}

	async fn cancel(&self) -> Result<(), Fault> {
		self.cancelled.store(true, Ordering::Release);
		self.state.interrupt_if_active(&self.cancelled)
	}
}
impl Drop for EmbeddedRun {
	fn drop(&mut self) {
		self.cancelled.store(true, Ordering::Release);
		let _ = self.state.interrupt_if_active(&self.cancelled);
	}
}

struct WorkerAlive<'a>(&'a AtomicBool);

impl Drop for WorkerAlive<'_> {
	fn drop(&mut self) {
		self.0.store(false, Ordering::Release);
	}
}

fn worker_main(
	engine: &Engine,
	state: &WorkerState,
	commands: Receiver<Command>,
	installer: &dyn NamespaceInstaller,
) {
	let _alive = WorkerAlive(&state.alive);
	engine.attach(|py| {
		let thread_id = unsafe { PyThread_get_thread_ident() };
		state
			.thread_id
			.store(i64::try_from(thread_id).unwrap_or(i64::MAX), Ordering::Release);
		let setup = match prepare_python(py) {
			Ok(setup) => setup,
			Err(error) => {
				fail_worker(&commands, Str::from(format_python_error(py, error)));
				return;
			},
		};
		let (runner, namespace_factory) = setup;
		let mut namespace = match new_namespace(py, &namespace_factory, installer) {
			Ok(namespace) => namespace,
			Err(error) => {
				fail_worker(&commands, Str::from(format_python_error(py, error)));
				return;
			},
		};

		while let Ok(command) = py.detach(|| commands.recv()) {
			let _ = command
				.events
				.send(Ok(RunEvent::Started { cell_id: command.cell_id.clone() }));
			if command.epoch != state.epoch.load(Ordering::Acquire) {
				let _ = command
					.events
					.send(Ok(RunEvent::Completed(cancelled_completion())));
				continue;
			}
			if command.request.reset {
				match new_namespace(py, &namespace_factory, installer) {
					Ok(fresh) => {
						close_namespace(py, &namespace);
						namespace = fresh;
					},
					Err(error) => {
						let _ = command.events.send(Err(Fault::Resource {
							operation: Str::from("reset"),
							message:   Str::from(format_python_error(py, error)),
						}));
						continue;
					},
				}
			}
			if command.cancelled.load(Ordering::Acquire) {
				let _ = command
					.events
					.send(Ok(RunEvent::Completed(cancelled_completion())));
				continue;
			}
			{
				let mut active = state.active.lock();
				*active = Some(ActiveCell {
					cell_id:   command.cell_id.clone(),
					cancelled: Arc::clone(&command.cancelled),
				});
			}
			if command.cancelled.load(Ordering::Acquire) {
				state.active.lock().take();
				let _ = command
					.events
					.send(Ok(RunEvent::Completed(cancelled_completion())));
				continue;
			}
			let result = execute_cell(
				py,
				&runner,
				&namespace,
				&command.cell_id,
				&command.request,
				command.events.clone(),
				installer,
			);
			{
				let mut active = state.active.lock();
				if active
					.as_ref()
					.is_some_and(|current| Arc::ptr_eq(&current.cancelled, &command.cancelled))
				{
					active.take();
				}
			}
			match result {
				Ok(completion) => {
					let _ = command.events.send(Ok(RunEvent::Completed(completion)));
				},
				Err(_) if command.cancelled.load(Ordering::Acquire) => {
					let _ = command
						.events
						.send(Ok(RunEvent::Completed(cancelled_completion())));
				},
				Err(error) => {
					state.alive.store(false, Ordering::Release);
					let _ = command.events.send(Err(Fault::Resource {
						operation: Str::from("execute"),
						message:   Str::from(format_python_error(py, error)),
					}));
					break;
				},
			}
		}
		close_namespace(py, &namespace);
		state.thread_id.store(0, Ordering::Release);
	});
}
fn cancelled_completion() -> RunCompletion {
	RunCompletion {
		status:          CellStatus {
			outcome:     CellOutcome::Cancelled,
			exit_code:   None,
			duration_ms: 0,
			exception:   None,
		},
		result:          None,
		display_outputs: Vec::new(),
		truncated:       false,
		spilled_output:  None,
		total_lines:     0,
		total_bytes:     0,
	}
}

fn prepare_python(py: Python<'_>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
	ensure_output_routers(py)?;
	let module = PyModule::from_code(py, BOOTSTRAP, c_str!("<omp-eval>"), c_str!("_omp_eval"))?;
	Ok((module.getattr("_omp_run_cell")?.unbind(), module.getattr("_omp_new_namespace")?.unbind()))
}

fn ensure_output_routers(py: Python<'_>) -> PyResult<OutputRouters> {
	let outputs = (*OUTPUT_ROUTERS).clone();
	let _initializing = OUTPUT_ROUTER_INIT.lock();
	let sys = PyModule::import(py, "sys")?;
	let current = sys
		.getattr(STDOUT_ROUTER_ATTR)
		.ok()
		.and_then(|value| value.extract::<Py<OutputRouter>>().ok())
		.zip(
			sys.getattr(STDERR_ROUTER_ATTR)
				.ok()
				.and_then(|value| value.extract::<Py<OutputRouter>>().ok()),
		)
		.filter(|(stdout, stderr)| {
			Arc::ptr_eq(&stdout.borrow(py).state, &outputs.stdout)
				&& Arc::ptr_eq(&stderr.borrow(py).state, &outputs.stderr)
		});
	let (stdout, stderr) = if let Some(current) = current {
		current
	} else {
		let stdout = Py::new(py, OutputRouter { state: Arc::clone(&outputs.stdout) })?;
		let stderr = Py::new(py, OutputRouter { state: Arc::clone(&outputs.stderr) })?;
		sys.setattr(STDOUT_ROUTER_ATTR, stdout.bind(py))?;
		sys.setattr(STDERR_ROUTER_ATTR, stderr.bind(py))?;
		(stdout, stderr)
	};
	sys.setattr("stdout", stdout.bind(py))?;
	sys.setattr("stderr", stderr.bind(py))?;
	Ok(outputs)
}

fn new_namespace(
	py: Python<'_>,
	factory: &Py<PyAny>,
	installer: &dyn NamespaceInstaller,
) -> PyResult<Py<PyDict>> {
	let value = factory.bind(py).call0()?;
	let globals = value.cast::<PyDict>()?;
	globals.set_item("__omp_display", Py::new(py, DisplayCollector::new())?)?;
	installer.install(py, globals)?;
	Ok(globals.clone().unbind())
}

fn close_namespace(py: Python<'_>, namespace: &Py<PyDict>) {
	if let Ok(Some(runner)) = namespace.bind(py).get_item("__omp_async_runner") {
		let _ = runner.call_method0("close");
	}
}

fn execute_cell(
	py: Python<'_>,
	runner: &Py<PyAny>,
	namespace: &Py<PyDict>,
	cell_id: &Bytes,
	request: &RunRequest,
	events: Sender<Result<RunEvent, Fault>>,
	installer: &dyn NamespaceInstaller,
) -> PyResult<RunCompletion> {
	let timeout = request.timeout.map(|duration| duration.as_secs_f64());
	let display = namespace
		.bind(py)
		.get_item("__omp_display")?
		.ok_or_else(|| {
			pyo3::exceptions::PyRuntimeError::new_err("eval namespace has no __omp_display collector")
		})?
		.extract::<Py<DisplayCollector>>()?;
	display.borrow(py).clear();
	let outputs = ensure_output_routers(py)?;
	let thread_id = unsafe { PyThread_get_thread_ident() };
	let capture = outputs.bind(thread_id, events);
	installer.begin_cell(py, namespace.bind(py), cell_id, request.timeout)?;
	let execution = runner
		.bind(py)
		.call1((request.code.as_str(), namespace.bind(py), timeout));
	let ended = installer.end_cell(py, namespace.bind(py), cell_id);
	let value = match (execution, ended) {
		(Err(error), _) => return Err(error),
		(Ok(_), Err(error)) => return Err(error),
		(Ok(value), Ok(())) => value,
	};
	let result = value.cast::<PyDict>()?;
	let outcome_name = get_string(result, "outcome")?;
	let outcome = match outcome_name.as_str() {
		"complete" => CellOutcome::Complete,
		"error" => CellOutcome::Error,
		"timeout" => CellOutcome::Timeout,
		"cancelled" => CellOutcome::Cancelled,
		other => {
			return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
				"eval runner returned unknown outcome {other:?}"
			)));
		},
	};
	let result_text = get_optional_string(result, "result_text")?;
	let result_json = get_optional_string(result, "result_json")?
		.map(|json| serde_json::from_str::<Value>(&json))
		.transpose()
		.map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
	let cell_value = result_text.map(|text| CellValue { text: Str::from(text), json: result_json });
	let error_name = get_optional_string(result, "error_name")?;
	let exception = if let Some(name) = error_name {
		let message = get_optional_string(result, "error_message")?.unwrap_or_default();
		let traceback = result
			.get_item("error_traceback")?
			.ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("error_traceback"))?
			.extract::<Vec<String>>()?
			.into_iter()
			.map(Str::from)
			.collect();
		Some(PythonException { name: Str::from(name), message: Str::from(message), traceback })
	} else {
		None
	};
	let duration_ms = result
		.get_item("duration_ms")?
		.ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("duration_ms"))?
		.extract::<u64>()?;

	let stdout_bytes = capture.stdout.snapshot();
	let stderr_bytes = capture.stderr.snapshot();
	let total_bytes = stdout_bytes.len() + stderr_bytes.len();
	let total_lines = count_lines(&stdout_bytes) + count_lines(&stderr_bytes);
	let display_outputs = display.borrow(py).drain(py)?;
	Ok(RunCompletion {
		status: CellStatus {
			outcome,
			exit_code: match outcome {
				CellOutcome::Complete => Some(0),
				CellOutcome::Error | CellOutcome::Timeout => Some(1),
				CellOutcome::Cancelled => None,
			},
			duration_ms,
			exception,
		},
		result: cell_value,
		display_outputs,
		truncated: false,
		spilled_output: None,
		total_lines,
		total_bytes,
	})
}

fn get_string(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<String> {
	dict
		.get_item(key)?
		.ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_owned()))?
		.extract()
}

fn get_optional_string(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<String>> {
	let value = dict
		.get_item(key)?
		.ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_owned()))?;
	if value.is_none() {
		Ok(None)
	} else {
		value.extract().map(Some)
	}
}

fn python_to_json(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Option<Value>> {
	let json = PyModule::import(py, "json")?;
	let encoded = match json.call_method1("dumps", (value,)) {
		Ok(encoded) => encoded.extract::<String>()?,
		Err(error) if error.is_instance_of::<pyo3::exceptions::PyTypeError>(py) => {
			return Ok(None);
		},
		Err(error) => return Err(error),
	};
	serde_json::from_str(&encoded)
		.map(Some)
		.map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}

fn count_lines(bytes: &[u8]) -> usize {
	if bytes.is_empty() {
		0
	} else {
		bytes.iter().filter(|byte| **byte == b'\n').count()
			+ usize::from(bytes.last() != Some(&b'\n'))
	}
}

fn fail_worker(commands: &Receiver<Command>, message: Str) {
	while let Ok(command) = commands.try_recv() {
		let _ = command.events.send(Err(Fault::Resource {
			operation: Str::from("initialize"),
			message:   message.clone(),
		}));
	}
}

fn format_python_error(py: Python<'_>, error: pyo3::PyErr) -> String {
	let formatted = PyModule::import(py, "traceback").and_then(|traceback| {
		traceback
			.call_method1(
				"format_exception",
				(error.get_type(py), error.value(py), error.traceback(py)),
			)?
			.extract::<Vec<String>>()
	});
	formatted.map_or_else(|_| error.to_string(), |lines| lines.concat())
}

#[cfg(test)]
mod tests {
	use std::sync::{Arc, LazyLock};

	use super::*;

	static ENGINE: LazyLock<Arc<Engine>> =
		LazyLock::new(|| Arc::new(Engine::builder().init().expect("embedded Python boots")));

	fn runtime() -> EmbeddedPython {
		EmbeddedPython::new(Arc::clone(&ENGINE))
	}

	async fn run_to_completion(
		runtime: &EmbeddedPython,
		session: &Session,
		code: &str,
		reset: bool,
	) -> (Vec<Update>, RunCompletion) {
		let mut run = runtime
			.run(session, RunRequest {
				code: Str::from(code),
				timeout: Some(Duration::from_secs(2)),
				reset,
			})
			.await
			.expect("cell starts");
		let mut updates = Vec::new();
		loop {
			match run.next_event().await.expect("event") {
				Some(RunEvent::Started { .. }) => {},
				Some(RunEvent::Output(update)) => updates.push(update),
				Some(RunEvent::Completed(done)) => return (updates, done),
				None => panic!("worker ended before completion"),
			}
		}
	}
	async fn completion(run: &mut EmbeddedRun) -> RunCompletion {
		loop {
			match run.next_event().await.expect("event") {
				Some(RunEvent::Started { .. } | RunEvent::Output(_)) => {},
				Some(RunEvent::Completed(done)) => return done,
				None => panic!("worker ended before completion"),
			}
		}
	}

	#[tokio::test]
	async fn state_persists_then_reset_replaces_namespace() {
		let runtime = runtime();
		let session = runtime.open_session().await.expect("session opens");
		let (_, first) = run_to_completion(&runtime, &session, "answer = 40", false).await;
		assert_eq!(first.status.outcome, CellOutcome::Complete);
		let (_, second) = run_to_completion(&runtime, &session, "answer + 2", false).await;
		assert_eq!(second.result.expect("REPL result").text, "42");
		let (_, reset) = run_to_completion(&runtime, &session, "'answer' in globals()", true).await;
		assert_eq!(reset.result.expect("REPL result").json, Some(Value::Bool(false)));
	}

	#[tokio::test]
	async fn stdout_stderr_and_result_keep_separate_boundaries() {
		let runtime = runtime();
		let session = runtime.open_session().await.expect("session opens");
		let (updates, done) = run_to_completion(
			&runtime,
			&session,
			"import sys\nprint('out')\nprint('err', file=sys.stderr)\n{'ok': True}",
			false,
		)
		.await;
		assert!(
			updates
				.windows(2)
				.all(|pair| pair[0].sequence < pair[1].sequence)
		);
		let stdout = updates
			.iter()
			.filter(|update| update.channel == OutputChannel::Stdout)
			.flat_map(|update| update.data.iter().copied())
			.collect::<Vec<_>>();
		let stderr = updates
			.iter()
			.filter(|update| update.channel == OutputChannel::Stderr)
			.flat_map(|update| update.data.iter().copied())
			.collect::<Vec<_>>();
		assert_eq!(stdout, b"out\n");
		assert_eq!(stderr, b"err\n");
		assert_eq!(done.result.expect("result").json, Some(serde_json::json!({"ok": true})));
	}

	#[tokio::test]
	async fn independent_sessions_execute_concurrently_without_output_cross_talk() {
		const BARRIER_MODULE: &str = "_omp_eval_parallel_barrier";
		ENGINE
			.attach(|py| -> PyResult<()> {
				let sys = PyModule::import(py, "sys")?;
				let modules = sys.getattr("modules")?;
				let threading = PyModule::import(py, "threading")?;
				let barrier = threading.getattr("Barrier")?.call1((2,))?;
				modules.set_item(BARRIER_MODULE, barrier)
			})
			.expect("shared barrier installs");

		let runtime = runtime();
		let left = runtime.open_session().await.expect("left session opens");
		let right = runtime.open_session().await.expect("right session opens");
		let left_code =
			format!("import sys\nsys.modules[{BARRIER_MODULE:?}].wait(timeout=1)\nprint('left')");
		let right_code =
			format!("import sys\nsys.modules[{BARRIER_MODULE:?}].wait(timeout=1)\nprint('right')");
		let (left_result, right_result) = tokio::join!(
			run_to_completion(&runtime, &left, &left_code, false),
			run_to_completion(&runtime, &right, &right_code, false),
		);
		let (left_updates, left_done) = left_result;
		let (right_updates, right_done) = right_result;
		assert_eq!(left_done.status.outcome, CellOutcome::Complete);
		assert_eq!(right_done.status.outcome, CellOutcome::Complete);

		let stdout = |updates: Vec<Update>| {
			updates
				.into_iter()
				.filter(|update| update.channel == OutputChannel::Stdout)
				.flat_map(|update| update.data.to_vec())
				.collect::<Vec<_>>()
		};
		assert_eq!(stdout(left_updates), b"left\n");
		assert_eq!(stdout(right_updates), b"right\n");
	}

	#[tokio::test]
	async fn failed_cell_keeps_prior_state_and_structured_traceback() {
		let runtime = runtime();
		let session = runtime.open_session().await.expect("session opens");
		run_to_completion(&runtime, &session, "kept = 7", false).await;
		let (_, failed) =
			run_to_completion(&runtime, &session, "raise ValueError('boom')", false).await;
		assert_eq!(failed.status.outcome, CellOutcome::Error);
		let error = failed.status.exception.expect("exception");
		assert_eq!(error.name, "ValueError");
		assert_eq!(error.message, "boom");
		let (_, after) = run_to_completion(&runtime, &session, "kept", false).await;
		assert_eq!(after.result.expect("result").text, "7");
	}

	#[tokio::test]
	async fn cancel_before_a_queued_cell_becomes_active_is_not_lost() {
		let runtime = runtime();
		let session = runtime.open_session().await.expect("session opens");
		let mut active = runtime
			.run(&session, RunRequest {
				code:    Str::new_static("while True: pass"),
				timeout: Some(Duration::from_secs(2)),
				reset:   false,
			})
			.await
			.expect("active cell starts");
		assert!(matches!(active.next_event().await.unwrap(), Some(RunEvent::Started { .. })));

		let mut queued = runtime
			.run(&session, RunRequest {
				code:    Str::new_static("queued_effect = True"),
				timeout: Some(Duration::from_secs(2)),
				reset:   false,
			})
			.await
			.expect("queued cell accepted");
		queued.cancel().await.expect("queued cell cancels");
		active.cancel().await.expect("active cell interrupts");

		assert_eq!(completion(&mut active).await.status.outcome, CellOutcome::Cancelled);
		assert_eq!(completion(&mut queued).await.status.outcome, CellOutcome::Cancelled);
		let (_, observed) =
			run_to_completion(&runtime, &session, "'queued_effect' in globals()", false).await;
		assert_eq!(observed.result.expect("boolean result").json, Some(Value::Bool(false)));
	}

	#[tokio::test]
	async fn reset_interrupts_active_work_invalidates_queued_cells_and_recreates_state() {
		let runtime = runtime();
		let session = runtime.open_session().await.expect("session opens");
		run_to_completion(&runtime, &session, "kept = 7", false).await;
		let mut active = runtime
			.run(&session, RunRequest {
				code:    Str::new_static("while True: pass"),
				timeout: Some(Duration::from_secs(2)),
				reset:   false,
			})
			.await
			.expect("active cell starts");
		assert!(matches!(active.next_event().await.unwrap(), Some(RunEvent::Started { .. })));
		let mut stale = runtime
			.run(&session, RunRequest {
				code:    Str::new_static("stale_effect = True"),
				timeout: Some(Duration::from_secs(2)),
				reset:   false,
			})
			.await
			.expect("stale cell queues");
		let mut reset = runtime
			.run(&session, RunRequest {
				code:    Str::new_static("('kept' in globals(), 'stale_effect' in globals())"),
				timeout: Some(Duration::from_secs(2)),
				reset:   true,
			})
			.await
			.expect("reset cell queues");

		assert_eq!(completion(&mut active).await.status.outcome, CellOutcome::Cancelled);
		assert_eq!(completion(&mut stale).await.status.outcome, CellOutcome::Cancelled);
		let reset = completion(&mut reset).await;
		assert_eq!(reset.result.expect("reset result").json, Some(serde_json::json!([false, false])));
	}

	#[tokio::test]
	async fn timeout_and_dropped_run_leave_the_worker_available_for_the_next_cell() {
		let runtime = runtime();
		let session = runtime.open_session().await.expect("session opens");
		let mut timed_out = runtime
			.run(&session, RunRequest {
				code:    Str::new_static("while True: pass"),
				timeout: Some(Duration::from_millis(25)),
				reset:   false,
			})
			.await
			.expect("timed cell starts");
		assert_eq!(completion(&mut timed_out).await.status.outcome, CellOutcome::Timeout);

		let mut dropped = runtime
			.run(&session, RunRequest {
				code:    Str::new_static("while True: pass"),
				timeout: Some(Duration::from_secs(2)),
				reset:   false,
			})
			.await
			.expect("dropped cell starts");
		assert!(matches!(dropped.next_event().await.unwrap(), Some(RunEvent::Started { .. })));
		drop(dropped);

		let (_, next) = run_to_completion(&runtime, &session, "6 * 7", false).await;
		assert_eq!(next.result.expect("next result").text, "42");
	}

	#[tokio::test]
	async fn top_level_await_returns_the_final_expression() {
		let runtime = runtime();
		let session = runtime.open_session().await.expect("session opens");
		let (_, done) = run_to_completion(
			&runtime,
			&session,
			"import asyncio\nawait asyncio.sleep(0, result=42)",
			false,
		)
		.await;
		assert_eq!(done.status.outcome, CellOutcome::Complete);
		assert_eq!(done.result.expect("await result").json, Some(Value::from(42)));
	}

	#[tokio::test]
	async fn display_collector_preserves_json_and_status_boundaries() {
		let runtime = runtime();
		let session = runtime.open_session().await.expect("session opens");
		let (_, done) = run_to_completion(
			&runtime,
			&session,
			concat!(
				"__omp_display({'application/json': {'answer': 42}}, raw=True)\n",
				"__omp_display({'application/x-omp-status': {'op': 'phase', 'title': 'load'}}, \
				 raw=True)",
			),
			false,
		)
		.await;
		assert_eq!(done.display_outputs, vec![
			super::super::DisplayOutput::Json { data: serde_json::json!({"answer": 42}) },
			super::super::DisplayOutput::Status {
				event: serde_json::json!({"op": "phase", "title": "load"}),
			},
		],);
	}

	#[tokio::test]
	async fn timeout_pause_excludes_host_wait_and_resume_starts_a_fresh_window() {
		let runtime = runtime();
		let session = runtime.open_session().await.expect("session opens");
		let mut run = runtime
			.run(&session, RunRequest {
				code:    Str::new_static(concat!(
					"import time\n",
					"__omp_timeout_pause__()\n",
					"time.sleep(0.075)\n",
					"__omp_timeout_resume__()\n",
					"7",
				)),
				timeout: Some(Duration::from_millis(30)),
				reset:   false,
			})
			.await
			.expect("paused cell starts");
		let done = completion(&mut run).await;
		assert_eq!(done.status.outcome, CellOutcome::Complete);
		assert_eq!(done.result.expect("result after host wait").json, Some(Value::from(7)));
	}

	#[tokio::test]
	async fn sessions_have_isolated_persistent_namespaces() {
		let runtime = runtime();
		let first = runtime.open_session().await.expect("first session opens");
		let second = runtime.open_session().await.expect("second session opens");
		run_to_completion(&runtime, &first, "private_value = 42", false).await;
		let (_, isolated) =
			run_to_completion(&runtime, &second, "'private_value' in globals()", false).await;
		assert_eq!(isolated.result.expect("isolation result").json, Some(Value::Bool(false)));
		let (_, persisted) = run_to_completion(&runtime, &first, "private_value", false).await;
		assert_eq!(persisted.result.expect("persistent result").json, Some(Value::from(42)));
	}

	#[derive(Debug)]
	struct FailFirstCell(AtomicBool);

	impl NamespaceInstaller for FailFirstCell {
		fn install(&self, _py: Python<'_>, _globals: &Bound<'_, PyDict>) -> PyResult<()> {
			Ok(())
		}

		fn begin_cell(
			&self,
			_py: Python<'_>,
			_globals: &Bound<'_, PyDict>,
			_cell_id: &Bytes,
			_timeout: Option<Duration>,
		) -> PyResult<()> {
			if self.0.swap(false, Ordering::AcqRel) {
				Err(pyo3::exceptions::PyRuntimeError::new_err("poisoned worker"))
			} else {
				Ok(())
			}
		}
	}

	#[tokio::test]
	async fn poisoned_worker_is_recreated_on_the_next_cell() {
		let runtime = EmbeddedPython::with_installer(
			Arc::clone(&ENGINE),
			Arc::new(FailFirstCell(AtomicBool::new(true))),
		);
		let session = runtime.open_session().await.expect("session opens");
		let mut failed = runtime
			.run(&session, RunRequest {
				code:    Str::new_static("1"),
				timeout: Some(Duration::from_secs(1)),
				reset:   false,
			})
			.await
			.expect("first cell accepted");
		assert!(matches!(failed.next_event().await.unwrap(), Some(RunEvent::Started { .. })));
		assert!(matches!(failed.next_event().await, Err(Fault::Resource { .. })));

		let (_, recovered) = run_to_completion(&runtime, &session, "6 * 7", false).await;
		assert_eq!(recovered.result.expect("recovered result").json, Some(Value::from(42)));
	}

	#[tokio::test]
	async fn cancelling_the_reset_cell_does_not_restore_the_old_namespace() {
		let runtime = runtime();
		let session = runtime.open_session().await.expect("session opens");
		run_to_completion(&runtime, &session, "old_state = 1", false).await;
		let mut active = runtime
			.run(&session, RunRequest {
				code:    Str::new_static("while True: pass"),
				timeout: Some(Duration::from_secs(2)),
				reset:   false,
			})
			.await
			.expect("active cell starts");
		assert!(matches!(active.next_event().await.unwrap(), Some(RunEvent::Started { .. })));
		let mut reset = runtime
			.run(&session, RunRequest {
				code:    Str::new_static("'old_state' in globals()"),
				timeout: Some(Duration::from_secs(2)),
				reset:   true,
			})
			.await
			.expect("reset accepted");
		reset.cancel().await.expect("reset execution cancels");
		assert_eq!(completion(&mut active).await.status.outcome, CellOutcome::Cancelled);
		assert_eq!(completion(&mut reset).await.status.outcome, CellOutcome::Cancelled);

		let (_, observed) =
			run_to_completion(&runtime, &session, "'old_state' in globals()", false).await;
		assert_eq!(observed.result.expect("state observation").json, Some(Value::Bool(false)));
	}
	#[tokio::test]
	async fn cancelling_one_worker_does_not_interrupt_another_session() {
		let runtime = runtime();
		let first_session = runtime.open_session().await.expect("first session opens");
		let second_session = runtime.open_session().await.expect("second session opens");
		let request = || RunRequest {
			code:    Str::new_static("while True: pass"),
			timeout: Some(Duration::from_secs(2)),
			reset:   false,
		};
		let mut first = runtime
			.run(&first_session, request())
			.await
			.expect("first starts");
		let mut second = runtime
			.run(&second_session, request())
			.await
			.expect("second starts");
		assert!(matches!(first.next_event().await.unwrap(), Some(RunEvent::Started { .. })));
		assert!(matches!(second.next_event().await.unwrap(), Some(RunEvent::Started { .. })));

		first.cancel().await.expect("first cancels");
		assert_eq!(completion(&mut first).await.status.outcome, CellOutcome::Cancelled);
		assert!(
			tokio::time::timeout(Duration::from_millis(30), second.next_event())
				.await
				.is_err(),
			"second worker must remain active",
		);
		second.cancel().await.expect("second cancels independently");
		assert_eq!(completion(&mut second).await.status.outcome, CellOutcome::Cancelled);
	}
}
