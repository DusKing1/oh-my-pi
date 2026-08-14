//! Same-binary child-process supervision for persistent Python eval sessions.

use std::{
	collections::{BTreeMap, HashMap},
	io,
	path::{Path, PathBuf},
	process::Stdio,
	sync::{
		Arc,
		atomic::{AtomicBool, AtomicU64, Ordering},
	},
	time::{Duration, Instant},
};

use async_trait::async_trait;
use bytes::Bytes;
use omp_core::Str;
use omp_tools::eval::{
	CellOutcome, CellStatus, EvalExec, EvalRun, Fault, PythonException, RunCompletion, RunEvent,
	RunRequest, Session, Update, idle_timeout::TimeoutHandle, kernel::EmbeddedPython,
};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;
use tokio::{
	io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt},
	process::{Child, ChildStdin, ChildStdout, Command},
	sync::{Mutex as AsyncMutex, oneshot},
};
use tokio_util::sync::CancellationToken;
use ulid::Ulid;

use super::bridge::{
	BridgeCapabilities, BridgeHost, BridgeHostError, BridgeNamespaceInstaller, ChildBridgeTransport,
	EvalSessionConfig, SessionBridgeHost,
};

/// Private argv selector used to re-enter `omp` as an eval kernel child.
pub const EVAL_CHILD_ARG: &str = "__omp-eval-child";

const MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;
const TERMINATE_GRACE: Duration = Duration::from_millis(100);
const CHILD_TIMEOUT_EXIT: i32 = 124;

/// Production [`EvalExec`] that owns one killable same-binary child per
/// session.
#[derive(Clone)]
pub struct ProcessEvalExec {
	inner: Arc<ProcessEvalInner>,
}

struct ProcessEvalInner {
	executable: PathBuf,
	host:       Arc<SessionBridgeHost>,
	sessions:   Mutex<HashMap<Bytes, Arc<ProcessSession>>>,
	next_cell:  AtomicU64,
}

struct ProcessSession {
	id:          Bytes,
	child:       AsyncMutex<Option<EvalChild>>,
	run_gate:    AsyncMutex<()>,
	needs_reset: AtomicBool,
}

/// Active cell in a process-backed Python session.
pub struct ProcessEvalRun {
	events:          flume::Receiver<Result<RunEvent, Fault>>,
	cancelled:       CancellationToken,
	terminal:        bool,
	effective_reset: bool,
}

impl ProcessEvalExec {
	/// Resolves the real `omp` executable and constructs the production
	/// resource.
	pub fn production(host: Arc<SessionBridgeHost>) -> Result<Self, io::Error> {
		resolve_omp_executable().map(|executable| Self::new(executable, host))
	}

	/// Constructs the resource with an explicit same-binary executable.
	#[must_use]
	pub fn new(executable: PathBuf, host: Arc<SessionBridgeHost>) -> Self {
		Self {
			inner: Arc::new(ProcessEvalInner {
				executable,
				host,
				sessions: Mutex::new(HashMap::new()),
				next_cell: AtomicU64::new(1),
			}),
		}
	}
}

impl EvalExec for ProcessEvalExec {
	type Run = ProcessEvalRun;

	async fn open_session(&self) -> Result<Session, Fault> {
		let id = Bytes::from(format!("py-process-{}", Ulid::generate()));
		self.inner.sessions.lock().insert(
			id.clone(),
			Arc::new(ProcessSession {
				id:          id.clone(),
				child:       AsyncMutex::new(None),
				run_gate:    AsyncMutex::new(()),
				needs_reset: AtomicBool::new(false),
			}),
		);
		Ok(Session { id })
	}

	async fn run<'a>(
		&'a self,
		session: &'a Session,
		mut request: RunRequest,
	) -> Result<Self::Run, Fault> {
		let owned = self
			.inner
			.sessions
			.lock()
			.get(&session.id)
			.cloned()
			.ok_or_else(|| Fault::SessionLost {
				message: Str::from("unknown Python process session"),
			})?;
		let number = self.inner.next_cell.fetch_add(1, Ordering::Relaxed);
		let cell_id =
			Bytes::from(format!("{}:cell-{number}", String::from_utf8_lossy(session.id.as_ref())));
		let forced_reset = owned.needs_reset.swap(false, Ordering::AcqRel);
		request.reset |= forced_reset;
		let effective_reset = request.reset;
		let (events_tx, events) = flume::unbounded();
		let cancelled = CancellationToken::new();
		let task_cancelled = cancelled.clone();
		let executable = self.inner.executable.clone();
		let host = Arc::clone(&self.inner.host);
		tokio::spawn(async move {
			let _gate = owned.run_gate.lock().await;
			if task_cancelled.is_cancelled() {
				owned.needs_reset.store(true, Ordering::Release);
				return;
			}
			let mut child_slot = owned.child.lock().await;
			if request.reset
				&& let Some(mut stale) = child_slot.take()
			{
				stale.terminate().await;
			}
			if child_slot.is_none() {
				match EvalChild::spawn(&executable, &owned.id, Arc::clone(&host)).await {
					Ok(child) => *child_slot = Some(child),
					Err(error) => {
						owned.needs_reset.store(true, Ordering::Release);
						let _ = events_tx.send(Err(resource_fault("open_session", error)));
						return;
					},
				}
			}
			if request.reset {
				// The whole process was replaced above; repeating a namespace-only
				// reset in the fresh child would only repeat setup.
				request.reset = false;
			}
			let child = child_slot.as_mut().expect("eval child initialized above");
			let keep = child
				.run_cell(cell_id, request, task_cancelled, &events_tx, host, &owned.needs_reset)
				.await;
			if !keep {
				child.terminate().await;
				*child_slot = None;
				owned.needs_reset.store(true, Ordering::Release);
			}
		});
		Ok(ProcessEvalRun { events, cancelled, terminal: false, effective_reset })
	}
}

impl EvalRun for ProcessEvalRun {
	async fn next_event(&mut self) -> Result<Option<RunEvent>, Fault> {
		match self.events.recv_async().await {
			Ok(Ok(event)) => {
				if matches!(event, RunEvent::Completed(_)) {
					self.terminal = true;
				}
				Ok(Some(event))
			},
			Ok(Err(error)) => {
				self.terminal = true;
				Err(error)
			},
			Err(_) => Ok(None),
		}
	}

	async fn cancel(&self) -> Result<(), Fault> {
		self.cancelled.cancel();
		Ok(())
	}

	fn reset(&self) -> bool {
		self.effective_reset
	}
}

impl Drop for ProcessEvalRun {
	fn drop(&mut self) {
		if !self.terminal {
			self.cancelled.cancel();
		}
	}
}
struct EvalChild {
	child:         Child,
	stdin:         ChildStdin,
	stdout:        ChildStdout,
	token:         Str,
	next_run:      AtomicU64,
	process_group: Option<u32>,
}

impl EvalChild {
	async fn spawn(
		executable: &Path,
		session_id: &Bytes,
		host: Arc<SessionBridgeHost>,
	) -> Result<Self, ProcessError> {
		let capabilities = host.capabilities()?.allowed_names();
		let config = host.session_config().map(WireSessionConfig::from);
		let token = Str::from(Ulid::generate().to_string());
		let mut command = Command::new(executable);
		command
			.arg(EVAL_CHILD_ARG)
			.stdin(Stdio::piped())
			.stdout(Stdio::piped())
			.stderr(Stdio::inherit())
			.kill_on_drop(true);
		#[cfg(unix)]
		{
			use std::os::unix::process::CommandExt;
			command.as_std_mut().process_group(0);
		}
		#[cfg(windows)]
		{
			use windows_sys::Win32::System::Threading::CREATE_NEW_PROCESS_GROUP;
			command.creation_flags(CREATE_NEW_PROCESS_GROUP);
		}
		let mut child = command.spawn()?;
		let process_group = child.id();
		let stdin = child
			.stdin
			.take()
			.ok_or_else(|| ProcessError::Protocol(Str::from("eval child stdin unavailable")))?;
		let stdout = child
			.stdout
			.take()
			.ok_or_else(|| ProcessError::Protocol(Str::from("eval child stdout unavailable")))?;
		let mut process = Self {
			child,
			stdin,
			stdout,
			token: token.clone(),
			next_run: AtomicU64::new(1),
			process_group,
		};
		write_frame(&mut process.stdin, &ParentFrame::Init {
			token,
			session_id: session_id.clone(),
			capabilities,
			config,
		})
		.await?;
		match tokio::time::timeout(Duration::from_secs(5), read_frame(&mut process.stdout)).await {
			Ok(Ok(Some(ChildFrame::Ready))) => Ok(process),
			Ok(Ok(Some(ChildFrame::Fatal { message }))) => Err(ProcessError::Protocol(message)),
			Ok(Ok(Some(_))) => Err(ProcessError::Protocol(Str::from(
				"eval child did not send Ready as its first frame",
			))),
			Ok(Ok(None)) => Err(ProcessError::Exited),
			Ok(Err(error)) => Err(error),
			Err(_) => Err(ProcessError::Protocol(Str::from("eval child startup timed out"))),
		}
	}

	async fn run_cell(
		&mut self,
		cell_id: Bytes,
		request: RunRequest,
		cancelled: CancellationToken,
		events: &flume::Sender<Result<RunEvent, Fault>>,
		host: Arc<SessionBridgeHost>,
		needs_reset: &AtomicBool,
	) -> bool {
		let run_id = self.next_run.fetch_add(1, Ordering::Relaxed);
		let started = Instant::now();
		let timeout = TimeoutHandle::new(request.timeout);
		if let Err(error) = write_frame(&mut self.stdin, &ParentFrame::Run {
			run_id,
			cell_id: cell_id.clone(),
			code: request.code,
			timeout_ms: request
				.timeout
				.map(|duration| u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)),
			reset: request.reset,
		})
		.await
		{
			needs_reset.store(true, Ordering::Release);
			let _ = events.send(Err(session_lost(error)));
			return false;
		}

		loop {
			let frame = tokio::select! {
				() = cancelled.cancelled() => {
					needs_reset.store(true, Ordering::Release);
					timeout.dispose();
					return false;
				},
				() = timeout.expired() => {
					needs_reset.store(true, Ordering::Release);
					let _ = events.send(Ok(RunEvent::Completed(Box::new(timeout_completion(elapsed_ms(started))))));
					return false;
				},
				frame = read_frame(&mut self.stdout) => frame,
			};
			let frame = match frame {
				Ok(Some(frame)) => frame,
				Ok(None) | Err(ProcessError::Exited) => {
					needs_reset.store(true, Ordering::Release);
					if self
						.child
						.try_wait()
						.ok()
						.flatten()
						.and_then(|status| status.code())
						== Some(CHILD_TIMEOUT_EXIT)
					{
						let _ = events.send(Ok(RunEvent::Completed(Box::new(timeout_completion(
							elapsed_ms(started),
						)))));
					} else {
						let _ = events.send(Err(Fault::SessionLost {
							message: Str::from("Python eval child exited during the active cell"),
						}));
					}
					return false;
				},
				Err(error) => {
					needs_reset.store(true, Ordering::Release);
					let _ = events.send(Err(session_lost(error)));
					return false;
				},
			};
			match frame {
				ChildFrame::Started { run_id: actual, cell_id: actual_cell }
					if actual == run_id && actual_cell == cell_id =>
				{
					let _ = events.send(Ok(RunEvent::Started { cell_id: actual_cell }));
				},
				ChildFrame::Output { run_id: actual, update } if actual == run_id => {
					let _ = events.send(Ok(RunEvent::Output(update)));
				},
				ChildFrame::Completed { run_id: actual, completion } if actual == run_id => {
					timeout.dispose();
					let _ = events.send(Ok(RunEvent::Completed(completion)));
					return true;
				},
				ChildFrame::BridgeCall { run_id: actual, request_id, token, name, args }
					if actual == run_id && token == self.token =>
				{
					let capabilities = match host.capabilities() {
						Ok(value) if value.allows(name.as_str()) => value,
						Ok(_) => {
							let _ = write_frame(&mut self.stdin, &ParentFrame::BridgeResponse {
								request_id,
								value: None,
								error: Some(Str::from(format!("bridge capability denied: {name}"))),
							})
							.await;
							continue;
						},
						Err(error) => {
							let _ = write_frame(&mut self.stdin, &ParentFrame::BridgeResponse {
								request_id,
								value: None,
								error: Some(Str::from(error.to_string())),
							})
							.await;
							continue;
						},
					};
					let _ = capabilities;
					let call = timeout.host_wait(host.call(name.as_str(), args));
					tokio::pin!(call);
					let response = tokio::select! {
						() = cancelled.cancelled() => {
							needs_reset.store(true, Ordering::Release);
							timeout.dispose();
							return false;
						},
						result = &mut call => result,
					};
					let (value, error) = match response {
						Ok(value) => (Some(value), None),
						Err(error) => (None, Some(Str::from(error.to_string()))),
					};
					if write_frame(&mut self.stdin, &ParentFrame::BridgeResponse {
						request_id,
						value,
						error,
					})
					.await
					.is_err()
					{
						needs_reset.store(true, Ordering::Release);
						let _ = events.send(Err(Fault::SessionLost {
							message: Str::from("Python eval child exited during a host bridge response"),
						}));
						return false;
					}
				},
				ChildFrame::Fatal { message } => {
					needs_reset.store(true, Ordering::Release);
					let _ = events.send(Err(Fault::SessionLost { message }));
					return false;
				},
				_ => {
					needs_reset.store(true, Ordering::Release);
					let _ = events.send(Err(Fault::SessionLost {
						message: Str::from("Python eval child sent an invalid or out-of-order frame"),
					}));
					return false;
				},
			}
		}
	}

	async fn terminate(&mut self) {
		let pid = self.process_group;
		#[cfg(unix)]
		if let Some(pid) = pid {
			let _ = nix::sys::signal::killpg(
				nix::unistd::Pid::from_raw(pid.cast_signed()),
				nix::sys::signal::Signal::SIGINT,
			);
		}
		#[cfg(windows)]
		if let Some(pid) = pid {
			unsafe {
				let _ = windows_sys::Win32::System::Console::GenerateConsoleCtrlEvent(
					windows_sys::Win32::System::Console::CTRL_BREAK_EVENT,
					pid,
				);
			}
		}

		if tokio::time::timeout(TERMINATE_GRACE, self.child.wait())
			.await
			.is_ok()
		{
			return;
		}
		#[cfg(unix)]
		if let Some(pid) = pid {
			let _ = nix::sys::signal::killpg(
				nix::unistd::Pid::from_raw(pid.cast_signed()),
				nix::sys::signal::Signal::SIGKILL,
			);
		}
		#[cfg(windows)]
		{
			let _ = self.child.start_kill();
		}
		let _ = self.child.wait().await;
	}
}
impl Drop for EvalChild {
	fn drop(&mut self) {
		let pid = self.process_group;
		#[cfg(unix)]
		if let Some(pid) = pid {
			let _ = nix::sys::signal::killpg(
				nix::unistd::Pid::from_raw(pid.cast_signed()),
				nix::sys::signal::Signal::SIGKILL,
			);
		}
		#[cfg(windows)]
		{
			let _ = self.child.start_kill();
		}
	}
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ParentFrame {
	Init {
		token:        Str,
		session_id:   Bytes,
		capabilities: Vec<Str>,
		config:       Option<WireSessionConfig>,
	},
	Run {
		run_id:     u64,
		cell_id:    Bytes,
		code:       Str,
		timeout_ms: Option<u64>,
		reset:      bool,
	},
	BridgeResponse {
		request_id: u64,
		value:      Option<Value>,
		error:      Option<Str>,
	},
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ChildFrame {
	Ready,
	Started {
		run_id:  u64,
		cell_id: Bytes,
	},
	Output {
		run_id: u64,
		update: Update,
	},
	Completed {
		run_id:     u64,
		completion: Box<RunCompletion>,
	},
	BridgeCall {
		run_id:     u64,
		request_id: u64,
		token:      Str,
		name:       Str,
		args:       Value,
	},
	Fatal {
		message: Str,
	},
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct WireSessionConfig {
	local_roots_json: Str,
	artifacts_dir:    Str,
	session_file:     Str,
}

impl From<EvalSessionConfig> for WireSessionConfig {
	fn from(config: EvalSessionConfig) -> Self {
		Self {
			local_roots_json: config.local_roots_json,
			artifacts_dir:    config.artifacts_dir,
			session_file:     config.session_file,
		}
	}
}

impl From<WireSessionConfig> for EvalSessionConfig {
	fn from(config: WireSessionConfig) -> Self {
		Self {
			local_roots_json: config.local_roots_json,
			artifacts_dir:    config.artifacts_dir,
			session_file:     config.session_file,
		}
	}
}

struct ChildBridgeHost {
	token:        Str,
	capabilities: BridgeCapabilities,
	config:       Option<EvalSessionConfig>,
	outgoing:     flume::Sender<ChildFrame>,
	pending:      Mutex<BTreeMap<u64, oneshot::Sender<Result<Value, Str>>>>,
	next_request: AtomicU64,
	active_run:   AtomicU64,
}

impl ChildBridgeHost {
	fn resolve(&self, request_id: u64, result: Result<Value, Str>) {
		if let Some(pending) = self.pending.lock().remove(&request_id) {
			let _ = pending.send(result);
		}
	}
}

#[async_trait]
impl ChildBridgeTransport for ChildBridgeHost {
	fn capabilities(&self) -> BridgeCapabilities {
		self.capabilities.clone()
	}

	fn session_config(&self) -> Option<EvalSessionConfig> {
		self.config.clone()
	}

	async fn call(&self, name: &str, args: Value) -> Result<Value, BridgeHostError> {
		if !self.capabilities.allows(name) {
			return Err(BridgeHostError::message(format!("bridge capability denied: {name}")));
		}
		let request_id = self.next_request.fetch_add(1, Ordering::Relaxed);
		let run_id = self.active_run.load(Ordering::Acquire);
		let (sender, receiver) = oneshot::channel();
		self.pending.lock().insert(request_id, sender);
		if self
			.outgoing
			.send(ChildFrame::BridgeCall {
				run_id,
				request_id,
				token: self.token.clone(),
				name: Str::from(name),
				args,
			})
			.is_err()
		{
			self.pending.lock().remove(&request_id);
			return Err(BridgeHostError::message("eval parent bridge disconnected"));
		}
		receiver
			.await
			.map_err(|_| BridgeHostError::message("eval parent bridge response was dropped"))?
			.map_err(BridgeHostError::message)
	}
}

/// Runs the hidden eval child entry before ordinary CLI or telemetry startup.
pub async fn run_eval_child_entry() -> Result<(), ProcessError> {
	let mut stdin = tokio::io::stdin();
	let (token, capabilities, config) = match read_frame::<_, ParentFrame>(&mut stdin).await? {
		Some(ParentFrame::Init { token, session_id: _, capabilities, config }) => {
			(token, capabilities, config)
		},
		Some(_) => {
			return Err(ProcessError::Protocol(Str::from("Init must be the first eval child frame")));
		},
		None => return Ok(()),
	};
	let (outgoing, outgoing_rx) = flume::unbounded();
	let child_host = Arc::new(ChildBridgeHost {
		token,
		capabilities: BridgeCapabilities::from_allowed_names(capabilities),
		config: config.map(EvalSessionConfig::from),
		outgoing,
		pending: Mutex::new(BTreeMap::new()),
		next_request: AtomicU64::new(1),
		active_run: AtomicU64::new(0),
	});
	let writer = tokio::spawn(async move {
		let mut stdout = tokio::io::stdout();
		while let Ok(frame) = outgoing_rx.recv_async().await {
			write_frame(&mut stdout, &frame).await?;
		}
		Ok::<(), ProcessError>(())
	});
	let runtime = tokio::runtime::Handle::current();
	let transport: Arc<dyn ChildBridgeTransport> = child_host.clone();
	let installer = Arc::new(BridgeNamespaceInstaller::new_child(transport, runtime));
	let engine = omp_py::Engine::builder()
		.init()
		.map(Arc::new)
		.map_err(|error| ProcessError::Python(Str::from(error.to_string())))?;
	let eval = EmbeddedPython::with_installer(engine, installer);
	let session = eval.open_session().await.map_err(ProcessError::Eval)?;
	child_host
		.outgoing
		.send(ChildFrame::Ready)
		.map_err(|_| ProcessError::Exited)?;
	let active = Arc::new(AtomicBool::new(false));
	loop {
		match read_frame::<_, ParentFrame>(&mut stdin).await? {
			Some(ParentFrame::Run { run_id, cell_id, code, timeout_ms, reset }) => {
				if active.swap(true, Ordering::AcqRel) {
					child_host
						.outgoing
						.send(ChildFrame::Fatal {
							message: Str::from("eval child received overlapping Run frames"),
						})
						.map_err(|_| ProcessError::Exited)?;
					continue;
				}
				child_host.active_run.store(run_id, Ordering::Release);
				let mut run = match eval
					.run(&session, RunRequest {
						code,
						timeout: timeout_ms.map(Duration::from_millis),
						reset,
					})
					.await
				{
					Ok(run) => run,
					Err(error) => {
						active.store(false, Ordering::Release);
						child_host
							.outgoing
							.send(ChildFrame::Fatal { message: Str::from(format!("{error:?}")) })
							.map_err(|_| ProcessError::Exited)?;
						continue;
					},
				};
				let outgoing = child_host.outgoing.clone();
				let active_run = Arc::clone(&active);
				tokio::spawn(async move {
					loop {
						match run.next_event().await {
							Ok(Some(RunEvent::Started { .. })) => {
								let _ =
									outgoing.send(ChildFrame::Started { run_id, cell_id: cell_id.clone() });
							},
							Ok(Some(RunEvent::Output(update))) => {
								let _ = outgoing.send(ChildFrame::Output { run_id, update });
							},
							Ok(Some(RunEvent::Completed(completion))) => {
								active_run.store(false, Ordering::Release);
								let _ = outgoing.send(ChildFrame::Completed { run_id, completion });
								break;
							},
							Ok(None) => {
								active_run.store(false, Ordering::Release);
								let _ = outgoing.send(ChildFrame::Fatal {
									message: Str::from("embedded eval stream ended without completion"),
								});
								break;
							},
							Err(error) => {
								active_run.store(false, Ordering::Release);
								let _ = outgoing
									.send(ChildFrame::Fatal { message: Str::from(format!("{error:?}")) });
								break;
							},
						}
					}
				});
			},
			Some(ParentFrame::BridgeResponse { request_id, value, error }) => {
				let result = match (value, error) {
					(Some(value), None) => Ok(value),
					(None, Some(error)) => Err(error),
					_ => Err(Str::from("malformed eval parent bridge response")),
				};
				child_host.resolve(request_id, result);
			},
			Some(ParentFrame::Init { .. }) => {
				return Err(ProcessError::Protocol(Str::from("duplicate eval child Init frame")));
			},
			None => break,
		}
	}
	writer.abort();
	let _ = writer.await;
	Ok(())
}

/// Eval child startup, framing, bridge, or embedded-runtime failure.
#[derive(Debug, thiserror::Error)]
pub enum ProcessError {
	/// Standard-I/O transport failed.
	#[error("eval child I/O failed: {0}")]
	Io(#[from] io::Error),
	/// A frame exceeded the fixed transport bound.
	#[error("eval child frame exceeded the {MAX_FRAME_BYTES}-byte limit")]
	FrameTooLarge,
	/// A bounded frame did not contain valid protocol JSON.
	#[error("eval child sent an invalid frame: {0}")]
	Json(#[from] serde_json::Error),
	/// Parent and child violated the expected protocol sequence.
	#[error("eval child protocol violation: {0}")]
	Protocol(Str),
	/// The child could not initialize embedded Python.
	#[error("eval child embedded Python failed: {0}")]
	Python(Str),
	/// The child's embedded eval kernel rejected an operation.
	#[error("eval child kernel failed: {0:?}")]
	Eval(Fault),
	/// The child closed its protocol stream.
	#[error("eval child exited")]
	Exited,
	/// The authenticated host bridge rejected startup or dispatch.
	#[error(transparent)]
	Bridge(#[from] BridgeHostError),
}

async fn write_frame<W: AsyncWrite + Unpin, T: Serialize>(
	writer: &mut W,
	frame: &T,
) -> Result<(), ProcessError> {
	let encoded = serde_json::to_vec(frame)?;
	if encoded.len() > MAX_FRAME_BYTES {
		return Err(ProcessError::FrameTooLarge);
	}
	let length = u32::try_from(encoded.len()).map_err(|_| ProcessError::FrameTooLarge)?;
	writer.write_all(&length.to_be_bytes()).await?;
	writer.write_all(&encoded).await?;
	writer.flush().await?;
	Ok(())
}

async fn read_frame<R: AsyncRead + Unpin, T: DeserializeOwned>(
	reader: &mut R,
) -> Result<Option<T>, ProcessError> {
	let mut prefix = [0_u8; 4];
	match reader.read_exact(&mut prefix).await {
		Ok(_) => {},
		Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
		Err(error) => return Err(ProcessError::Io(error)),
	}
	let length = usize::try_from(u32::from_be_bytes(prefix)).unwrap_or(usize::MAX);
	if length == 0 || length > MAX_FRAME_BYTES {
		return Err(ProcessError::FrameTooLarge);
	}
	let mut encoded = vec![0; length];
	reader.read_exact(&mut encoded).await?;
	serde_json::from_slice(&encoded)
		.map(Some)
		.map_err(ProcessError::from)
}

fn resolve_omp_executable() -> io::Result<PathBuf> {
	if let Some(path) = std::env::var_os("CARGO_BIN_EXE_omp") {
		let path = PathBuf::from(path);
		if path.is_file() {
			return Ok(path);
		}
	}
	let current = std::env::current_exe()?;
	if current.file_stem().is_some_and(|name| name == "omp") {
		return Ok(current);
	}
	let mut directory = current
		.parent()
		.ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "current executable has no parent"))?;
	if directory.file_name().is_some_and(|name| name == "deps") {
		directory = directory.parent().ok_or_else(|| {
			io::Error::new(io::ErrorKind::NotFound, "target deps directory has no parent")
		})?;
	}
	let sibling = directory.join(format!("omp{}", std::env::consts::EXE_SUFFIX));
	if sibling.is_file() {
		return Ok(sibling);
	}
	Err(io::Error::new(
		io::ErrorKind::NotFound,
		format!(
			"real omp executable not found (set CARGO_BIN_EXE_omp or build {})",
			sibling.display()
		),
	))
}

fn elapsed_ms(started: Instant) -> u64 {
	u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX)
}

fn timeout_completion(duration_ms: u64) -> RunCompletion {
	RunCompletion {
		status:          CellStatus {
			outcome: CellOutcome::Timeout,
			exit_code: Some(1),
			duration_ms,
			exception: Some(PythonException {
				name:      Str::new_static("TimeoutError"),
				message:   Str::new_static("OMP eval cell timed out"),
				traceback: Vec::new(),
			}),
		},
		result:          None,
		display_outputs: Vec::new(),
		truncated:       false,
		spilled_output:  None,
		total_lines:     0,
		total_bytes:     0,
	}
}

fn resource_fault(operation: &'static str, error: ProcessError) -> Fault {
	Fault::Resource {
		operation: Str::new_static(operation),
		message:   Str::from(error.to_string()),
	}
}

fn session_lost(error: ProcessError) -> Fault {
	Fault::SessionLost { message: Str::from(error.to_string()) }
}
