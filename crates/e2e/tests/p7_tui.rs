#![cfg(unix)]

use std::{
	collections::VecDeque,
	io::{BufRead as _, BufReader, Read as _, Write as _},
	os::{fd::{AsFd as _, AsRawFd as _}, unix::net::UnixStream},
	path::Path,
	process::{Child, Command, Stdio},
	sync::{
		Arc,
		atomic::{AtomicBool, Ordering},
	},
	task::{Context, Poll},
	thread,
	time::{Duration, Instant},
};

use bytes::Bytes;
use flume::{Receiver, Sender};
use futures::future::BoxFuture;
use nix::{
	fcntl::{FcntlArg, OFlag, fcntl},
	pty::{Winsize, openpty},
	sys::termios::{Termios, tcgetattr},
	unistd::ttyname,
};
use omp_app::{
	daemon::{DaemonConfig, DaemonHandle},
	endpoint::LocalEndpoint,
};
use omp_core::Str;
use omp_llm_catalog::{
	CompiledCatalog, ManagementCapabilities, OperationBits, OperationKind,
	snapshot::{Catalog, SnapshotProvenance},
};
use omp_llm_inference::{
	Answer, Error as InferenceError, Registry,
	call::{Call, OpaqueJson},
	event::{BlockKind, ChatEvent, Completion, FinishReason, ToolCall},
	id::ToolCallId,
	layer::{LayerCall, stack::RouteProviderService},
	provider::fake::{FakeProvider, FakeScript},
	receipt::{ExecutionReceipt, ReasonId, Usage},
	registry::RouteUnavailable,
	session::ConversationSessionPlanner,
};
use omp_tool::{Constraint, Ev, IncomingParams, Part, PromptCaps, Rev, Tool, ToolSpec};
use parking_lot::Mutex;
use serde_json::{Value, json};
use tower::Service;

const READY_TIMEOUT: Duration = Duration::from_secs(12);
const CHECKPOINT_TIMEOUT: Duration = Duration::from_secs(15);
const IO_TIMEOUT: Duration = Duration::from_secs(2);

struct ProofTool {
	spec: ToolSpec,
}

impl ProofTool {
	fn new(name: &'static str, family: &'static str) -> Self {
		Self {
			spec: ToolSpec {
				name: name.into(),
				rev: Rev { family: family.into(), n: 1 },
				description: "P7 gateway-side executor declaration".into(),
				schema: Bytes::from_static(
					br#"{"type":"object","additionalProperties":true}"#,
				),
				constraint: Constraint::None,
			},
		}
	}
}

impl Tool for ProofTool {
	type Fault = Value;
	type Params = Value;
	type Payload = Value;
	type Update = Value;

	fn spec(&self) -> &ToolSpec {
		&self.spec
	}

	fn call<'call>(
		&'call self,
		_params: IncomingParams<'call>,
	) -> impl futures::Stream<Item = Ev<Self::Update, Self::Payload, Self::Fault>> + Send + 'call {
		futures::stream::empty()
	}

	fn prompt(&self, _view: Result<&Self::Payload, &Self::Fault>, _caps: &PromptCaps) -> Vec<Part> {
		Vec::new()
	}
}

#[derive(Clone)]
struct GatedRoute {
	fake:  FakeProvider,
	gates: Arc<Mutex<VecDeque<Receiver<()>>>>,
}

impl Service<LayerCall<Call>> for GatedRoute {
	type Error = InferenceError;
	type Future = BoxFuture<'static, Result<Answer, InferenceError>>;
	type Response = Answer;

	fn poll_ready(&mut self, context: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		<FakeProvider as Service<Call>>::poll_ready(&mut self.fake, context)
	}

	fn call(&mut self, request: LayerCall<Call>) -> Self::Future {
		let gate = self.gates.lock().pop_front().expect("every scripted provider call has a gate");
		let response = <FakeProvider as Service<Call>>::call(&mut self.fake, request.payload);
		Box::pin(async move {
			gate.recv_async().await.expect("scripted provider gate remains open");
			response.await
		})
	}
}

struct ScriptedGateway {
	_handle: DaemonHandle,
	model:   String,
	permits: Vec<Sender<()>>,
}

impl ScriptedGateway {
	async fn start(scratch: &Path, socket: &Path, shell_release: &Path) -> Self {
		let scripts = scripts(shell_release);
		let mut senders = Vec::with_capacity(scripts.len());
		let mut receivers = VecDeque::with_capacity(scripts.len());
		for _ in 0..scripts.len() {
			let (sender, receiver) = flume::bounded(1);
			senders.push(sender);
			receivers.push_back(receiver);
		}
		let (registry, sessions, fake, model) = scripted_registry(scratch, receivers);
		fake.extend(scripts);

		let mut tools = omp_tool::Registry::new();
		for (name, family) in [
			("read", ""),
			("edit", "hl"),
			("shell", ""),
			("grep", ""),
			("glob", ""),
			("p7_unknown", ""),
		] {
			tools.register(ProofTool::new(name, family)).expect("proof tool registers");
		}
		let (responses, _ignored) = flume::bounded(32);
		let handle = tokio::time::timeout(
			READY_TIMEOUT,
			DaemonHandle::start_for_test(
				DaemonConfig::local(LocalEndpoint::from(socket.to_path_buf()))
					.with_data_dir(scratch.join("gateway-state")),
				registry,
				sessions,
				Arc::new(tools),
				responses,
			),
		)
		.await
		.expect("gateway startup timed out")
		.expect("scripted gateway starts");
		Self { _handle: handle, model, permits: senders }
	}

	fn release(&self, call: usize) {
		self.permits[call].send(()).expect("scripted call gate remains open");
	}
}

fn scripted_registry(
	scratch: &Path,
	gates: VecDeque<Receiver<()>>,
) -> (Registry, ConversationSessionPlanner, FakeProvider, String) {
	let mut compiled: CompiledCatalog = serde_json::from_str(include_str!(
		"../../llm-catalog/data/catalog.normalized.json"
	))
	.expect("normalized catalog");
	for provider in &mut compiled.providers {
		provider.management = ManagementCapabilities {
			operations: OperationBits::empty(),
			multiple_accounts: false,
			refresh: false,
			principal_quota: false,
		};
	}
	let artifacts = Catalog::encode(compiled, SnapshotProvenance { source_digest: [0; 32] })
		.expect("catalog snapshot");
	let catalog = Arc::new(Catalog::decode(&artifacts.postcard).expect("catalog decode"));
	let model = catalog
		.models()
		.iter()
		.find(|candidate| candidate.capabilities.operations.contains_kind(OperationKind::Chat))
		.expect("chat model");
	let route_id = model.routes.first().expect("chat route").clone();
	let route = catalog.route(&route_id).expect("selected route");
	let fake = FakeProvider::new(route.provider.clone(), route_id.clone());
	let route_service = RouteProviderService::new(GatedRoute {
		fake: fake.clone(),
		gates: Arc::new(Mutex::new(gates)),
	});
	let mut builder = Registry::builder(catalog.clone());
	for candidate in catalog.routes() {
		builder = if candidate.id == route_id {
			builder
				.register_route(candidate.id.clone(), route_service.clone())
				.expect("scripted route registers")
		} else {
			builder
				.register_unavailable(RouteUnavailable {
					route: candidate.id.clone(),
					reason: ReasonId(Str::from("p7-scripted-route-only")),
					operation: None,
				})
				.expect("unavailable route registers")
		};
	}
	let sessions = ConversationSessionPlanner::open(scratch.join("sessions.db"), catalog)
		.expect("conversation store opens");
	(builder.build().expect("base registry"), sessions, fake, model.key.as_str().to_owned())
}


fn tool_script(calls: &[(&str, &str, Value)]) -> FakeScript {
	let mut events = Vec::with_capacity(calls.len() * 3 + 1);
	for (index, (id, name, arguments)) in calls.iter().enumerate() {
		let index = u32::try_from(index).expect("small scripted batch");
		let id = ToolCallId::from(*id);
		events.push(Ok(ChatEvent::ToolCallStarted {
			index,
			id: id.clone(),
			name: Str::from(*name),
		}));
		events.push(Ok(ChatEvent::ToolArgumentsDelta {
			index,
			bytes: Bytes::from(serde_json::to_vec(arguments).expect("tool args encode")),
		}));
		events.push(Ok(ChatEvent::ToolCallReady {
			index,
			call: ToolCall {
				id,
				name: Str::from(*name),
				arguments: OpaqueJson::new(arguments.clone()),
			},
		}));
	}
	events.push(Ok(completed(FinishReason::ToolCalls, calls.len())));
	FakeScript::chat(events)
}

fn text_script(text: &'static str) -> FakeScript {
	FakeScript::chat(vec![
		Ok(ChatEvent::BlockStarted { index: 0, kind: BlockKind::Text }),
		Ok(ChatEvent::TextDelta { index: 0, text: Str::from(text) }),
		Ok(completed(FinishReason::Stop, 1)),
	])
}

fn completed(reason: FinishReason, blocks: usize) -> ChatEvent {
	ChatEvent::Completed(Completion {
		reason,
		blocks: u32::try_from(blocks).expect("small script"),
		usage: Usage::default(),
		receipt: ExecutionReceipt::default(),
	})
}

fn scripts(shell_release: &Path) -> Vec<FakeScript> {
	let release = shell_quote(shell_release);
	vec![
		tool_script(&[("read-1", "read", json!({ "path": "scratch.txt" }))]),
		tool_script(&[(
			"edit-1",
			"edit",
			json!({ "path": "scratch.txt", "patch": "PUT 1.=1:\n+new" }),
		)]),
		tool_script(&[(
			"shell-1",
			"shell",
			json!({
				"command": format!(
					"printf 'live-tail\\n'; while [ ! -f {release} ]; do sleep 0.05; done; printf 'live-error\\n' >&2; exit 7"
				)
			}),
		)]),
		tool_script(&[(
			"unknown-1",
			"p7_unknown",
			json!({ "path": "mystery.fixture", "opaque": true }),
		)]),
		text_script("The deterministic tool sequence is complete."),
		tool_script(&[
			(
				"batch-1",
				"shell",
				json!({ "command": "printf 'batch-one-started\\n'; sleep 30" }),
			),
			("batch-2", "shell", json!({ "command": "printf 'batch-two-ran\\n'" })),
			("batch-3", "shell", json!({ "command": "printf 'batch-three-ran\\n'" })),
		]),
		text_script("The steering item reached the next turn."),
	]
}

fn shell_quote(path: &Path) -> String {
	format!("'{}'", path.display().to_string().replace('’', "'\\''").replace('\'', "'\\''"))
}

#[derive(Clone, Debug)]
struct Snapshot {
	text:   String,
	frame:  String,
	tree:   Value,
	values: Value,
}

impl Snapshot {
	fn combined(&self) -> String {
		format!("{}\n{}\n{}", self.text, self.frame, self.tree)
	}
}

struct DebugClient {
	reader: BufReader<UnixStream>,
	writer: UnixStream,
}

impl DebugClient {
	fn connect(path: &Path, deadline: Instant) -> Self {
		let mut last = None;
		loop {
			match UnixStream::connect(path) {
				Ok(stream) => {
					stream.set_read_timeout(Some(IO_TIMEOUT)).expect("debug read timeout");
					stream.set_write_timeout(Some(IO_TIMEOUT)).expect("debug write timeout");
					let writer = stream.try_clone().expect("clone debug socket");
					return Self { reader: BufReader::new(stream), writer };
				},
				Err(error) => last = Some(error),
			}
			assert!(Instant::now() < deadline, "debug socket did not become ready: {last:?}");
			thread::sleep(Duration::from_millis(20));
		}
	}

	fn request(&mut self, request: Value) -> Result<Value, String> {
		serde_json::to_writer(&mut self.writer, &request).map_err(|error| error.to_string())?;
		self.writer.write_all(b"\n").map_err(|error| error.to_string())?;
		self.writer.flush().map_err(|error| error.to_string())?;
		let mut line = String::new();
		self.reader.read_line(&mut line).map_err(|error| error.to_string())?;
		if line.is_empty() {
			return Err("debug socket closed".to_owned());
		}
		let response: Value = serde_json::from_str(&line).map_err(|error| error.to_string())?;
		if response.get("ok").and_then(Value::as_bool) != Some(true) {
			return Err(format!("debug request {request} failed: {response}"));
		}
		Ok(response)
	}

	fn op(&mut self, op: &'static str) -> Result<Value, String> {
		self.request(json!({ "op": op }))
	}

	fn keys(&mut self, keys: &str) {
		self.request(json!({ "op": "keys", "keys": keys }))
			.unwrap_or_else(|error| panic!("key injection failed: {error}"));
	}

	fn snapshot(&mut self) -> Result<Snapshot, String> {
		let text = lines(&self.op("text")?);
		let frame = lines(&self.op("frame")?);
		let tree = self.op("tree")?.get("tree").cloned().ok_or("tree response missing tree")?;
		let values = self
			.op("values")?
			.get("values")
			.cloned()
			.ok_or("values response missing values")?;
		Ok(Snapshot { text, frame, tree, values })
	}
}

fn lines(response: &Value) -> String {
	response
		.get("lines")
		.and_then(Value::as_array)
		.into_iter()
		.flatten()
		.filter_map(Value::as_str)
		.collect::<Vec<_>>()
		.join("\n")
}

struct PtyChild {
	child:      Child,
	master:     std::os::fd::OwnedFd,
	slave:      std::os::fd::OwnedFd,
	before:     Termios,
	raw:        Arc<Mutex<Vec<u8>>>,
	reader_end: Arc<AtomicBool>,
	reader:     Option<thread::JoinHandle<()>>,
}

impl PtyChild {
	fn spawn(binary: &Path, args: &[String], project: &Path, debug: &Path) -> Self {
		let window = Winsize { ws_row: 48, ws_col: 120, ws_xpixel: 0, ws_ypixel: 0 };
		let pty = openpty(Some(&window), None).expect("open PTY");
		let device = ttyname(&pty.slave).expect("PTY slave path");
		let before = tcgetattr(&pty.slave).expect("initial PTY termios");
		fcntl(&pty.master, FcntlArg::F_SETFL(OFlag::O_NONBLOCK)).expect("nonblocking PTY master");
		let reader_fd = pty.master.try_clone().expect("clone PTY master");
		let raw = Arc::new(Mutex::new(Vec::new()));
		let reader_raw = raw.clone();
		let reader_end = Arc::new(AtomicBool::new(false));
		let reader_stop = reader_end.clone();
		let reader = thread::spawn(move || {
			let mut buffer = [0_u8; 16 * 1024];
			loop {
				match nix::unistd::read(&reader_fd, &mut buffer) {
					Ok(0) if reader_stop.load(Ordering::Acquire) => break,
					Ok(0) => thread::sleep(Duration::from_millis(5)),
					Ok(count) => reader_raw.lock().extend_from_slice(&buffer[..count]),
					Err(nix::errno::Errno::EAGAIN) if reader_stop.load(Ordering::Acquire) => break,
					Err(nix::errno::Errno::EAGAIN) => thread::sleep(Duration::from_millis(5)),
					Err(nix::errno::Errno::EIO) => break,
					Err(error) => panic!("PTY read failed: {error}"),
				}
			}
		});

		let home = project.parent().expect("project has parent").join("home");
		std::fs::create_dir_all(&home).expect("create isolated home");
		let child = Command::new(binary)
			.args(args)
			.current_dir(project)
			.env("TERM", "xterm-256color")
			.env("HOME", &home)
			.env("OMP_DATA_DIR", home.join("data"))
			.env("OMP_TTY", &device)
			.env("OMP_TUI_DEBUG", debug)
			.env("NO_COLOR", "1")
			.stdout(Stdio::piped())
			.stderr(Stdio::piped())
			.spawn()
			.expect("spawn omp chat");
		Self {
			child,
			master: pty.master,
			slave: pty.slave,
			before,
			raw,
			reader_end,
			reader: Some(reader),
		}
	}

	fn resize(&self, rows: u16, cols: u16) {
		let window = libc::winsize {
			ws_row: rows,
			ws_col: cols,
			ws_xpixel: 0,
			ws_ypixel: 0,
		};
		// SAFETY: master is a live PTY and window is a valid winsize value.
		let result = unsafe { libc::ioctl(self.master.as_fd().as_raw_fd(), libc::TIOCSWINSZ, &window) };
		assert_eq!(result, 0, "TIOCSWINSZ failed: {}", std::io::Error::last_os_error());
	}

	fn raw(&self) -> Vec<u8> {
		self.raw.lock().clone()
	}

	fn wait(mut self, timeout: Duration) -> (std::process::ExitStatus, Vec<u8>, String, String, Termios) {
		let deadline = Instant::now() + timeout;
		let status = loop {
			match self.child.try_wait().expect("poll omp chat") {
				Some(status) => break status,
				None if Instant::now() < deadline => thread::sleep(Duration::from_millis(20)),
				None => {
					let raw = visible(&self.raw());
					let _ = self.child.kill();
					panic!("omp chat did not exit in {timeout:?}; raw PTY:\n{raw}");
				},
			}
		};
		self.reader_end.store(true, Ordering::Release);
		if let Some(reader) = self.reader.take() {
			reader.join().expect("PTY reader joins");
		}
		let mut stdout = String::new();
		let mut stderr = String::new();
		if let Some(mut pipe) = self.child.stdout.take() {
			pipe.read_to_string(&mut stdout).expect("read child stdout");
		}
		if let Some(mut pipe) = self.child.stderr.take() {
			pipe.read_to_string(&mut stderr).expect("read child stderr");
		}
		let after = tcgetattr(&self.slave).expect("final PTY termios");
		(status, self.raw(), stdout, stderr, after)
	}
}

fn wait_snapshot(
	debug: &mut DebugClient,
	raw: &Arc<Mutex<Vec<u8>>>,
	label: &str,
	mut ready: impl FnMut(&Snapshot) -> bool,
) -> Snapshot {
	let deadline = Instant::now() + CHECKPOINT_TIMEOUT;
	let mut last = None;
	let mut error = None;
	loop {
		match debug.snapshot() {
			Ok(snapshot) if ready(&snapshot) => return snapshot,
			Ok(snapshot) => last = Some(snapshot),
			Err(problem) => error = Some(problem),
		}
		if Instant::now() >= deadline {
			let snapshot = last.map_or_else(|| "<none>".to_owned(), |value| format!("{value:#?}"));
			panic!(
				"checkpoint {label:?} timed out\nlast error: {error:?}\nlast snapshot:\n{snapshot}\nraw PTY:\n{}",
				visible(&raw.lock()),
			);
		}
		thread::sleep(Duration::from_millis(15));
	}
}

fn assert_surface(snapshot: &Snapshot, label: &str) {
	let tree = snapshot.tree.to_string();
	let values = snapshot.values.as_object();
	assert!(tree.contains("TranscriptView") && tree.contains("transcript"), "{label}: transcript missing: {tree}");
	assert!(tree.contains("Editor") && tree.contains("input"), "{label}: editor missing: {tree}");
	assert!(values.is_some_and(|map| map.contains_key("input")), "{label}: input value missing: {}", snapshot.values);
}

fn visible(bytes: &[u8]) -> String {
	let mut out = String::new();
	for &byte in bytes.iter().rev().take(96 * 1024).collect::<Vec<_>>().iter().rev() {
		match byte {
			b'\n' => out.push('\n'),
			b'\r' => out.push_str("\\r"),
			b'\t' => out.push_str("\\t"),
			0x20..=0x7e => out.push(char::from(byte)),
			_ => out.push_str(&format!("\\x{byte:02x}")),
		}
	}
	out
}

fn assert_restored(raw: &[u8], before: &Termios, after: &Termios, diagnostics: &str) {
	for sequence in ["\x1b[?1049h", "\x1b[?1047h", "\x1b[?47h"] {
		assert!(!raw.windows(sequence.len()).any(|window| window == sequence.as_bytes()), "alternate-buffer entry {sequence:?} observed\n{diagnostics}");
	}
	for mode in ["\x1b[?1000h", "\x1b[?1002h", "\x1b[?1003h", "\x1b[?1006h"] {
		assert!(!raw.windows(mode.len()).any(|window| window == mode.as_bytes()), "mouse tracking enable {mode:?} observed\n{diagnostics}");
	}
	let hide = raw.windows(6).rposition(|window| window == b"\x1b[?25l");
	let show = raw.windows(6).rposition(|window| window == b"\x1b[?25h");
	assert!(show.is_some() && hide.is_none_or(|hidden| show > Some(hidden)), "cursor was not restored; hide={hide:?} show={show:?}\n{diagnostics}");
	assert_eq!(after.input_flags, before.input_flags, "input flags not restored\n{diagnostics}");
	assert_eq!(after.output_flags, before.output_flags, "output flags not restored\n{diagnostics}");
	assert_eq!(after.control_flags, before.control_flags, "control flags not restored\n{diagnostics}");
	assert_eq!(after.local_flags, before.local_flags, "local flags not restored\n{diagnostics}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_tui_drives_real_pty_tools_interrupt_resize_and_clean_quit() {
	let scratch = tempfile::tempdir().expect("scratch root");
	let project = scratch.path().join("project");
	std::fs::create_dir(&project).expect("project directory");
	std::fs::write(project.join("scratch.txt"), "old\n").expect("write read/edit fixture");
	let shell_release = scratch.path().join("release-shell");
	let gateway_socket = scratch.path().join("gateway.sock");
	let debug_socket = scratch.path().join("tui-debug.sock");
	let gateway = ScriptedGateway::start(scratch.path(), &gateway_socket, &shell_release).await;
	gateway.release(0);

	let binary = omp_e2e::support::omp_binary();
	let args = vec![
		"chat".to_owned(),
		"--model".to_owned(),
		gateway.model.clone(),
		"--project".to_owned(),
		project.display().to_string(),
		"--gateway".to_owned(),
		gateway_socket.display().to_string(),
	];
	let process = PtyChild::spawn(&binary, &args, &project, &debug_socket);
	let raw_capture = process.raw.clone();
	let mut debug = DebugClient::connect(&debug_socket, Instant::now() + READY_TIMEOUT);
	let ready = wait_snapshot(&mut debug, &raw_capture, "chat shell ready", |snapshot| {
		let tree = snapshot.tree.to_string();
		tree.contains("transcript") && tree.contains("input") && tree.contains("status")
	});
	assert_surface(&ready, "ready");

	debug.keys("'exercise deterministic tools' enter");
	let read = wait_snapshot(&mut debug, &raw_capture, "read card final", |snapshot| {
		let all = snapshot.combined();
		all.contains("read scratch.txt") && snapshot.tree.to_string().contains("ToolCard")
	});
	assert_surface(&read, "read");
	assert!(read.tree.to_string().contains("read-1"), "read card id absent: {}", read.tree);

	gateway.release(1);
	let preview = wait_snapshot(&mut debug, &raw_capture, "edit preview", |snapshot| {
		snapshot.tree.to_string().contains("DiffView")
			&& snapshot.combined().contains("edit scratch.txt")
			&& std::fs::read_to_string(project.join("scratch.txt")).as_deref() == Ok("old\n")
	});
	assert_surface(&preview, "edit preview");
	let final_edit = wait_snapshot(&mut debug, &raw_capture, "edit final", |snapshot| {
		snapshot.tree.to_string().contains("DiffView")
			&& snapshot.combined().contains("edit scratch.txt")
			&& std::fs::read_to_string(project.join("scratch.txt")).as_deref() == Ok("new\n")
	});
	assert_surface(&final_edit, "edit final");

	gateway.release(2);
	let shell_live = wait_snapshot(&mut debug, &raw_capture, "shell live tail", |snapshot| {
		snapshot.combined().contains("live-tail")
			&& snapshot.tree.to_string().contains("TextLeaf")
			&& snapshot.tree.to_string().contains("shell-1")
	});
	assert_surface(&shell_live, "shell live");
	assert!(shell_live.frame.contains("read scratch.txt") && shell_live.frame.contains("edit scratch.txt"), "prior transcript vanished during shell stream: {}", shell_live.frame);
	std::fs::write(&shell_release, b"release").expect("release shell fixture");
	let shell_final = wait_snapshot(&mut debug, &raw_capture, "shell exit badge", |snapshot| {
		snapshot.combined().contains("exit 7") && snapshot.tree.to_string().contains("shell-1")
	});
	assert_surface(&shell_final, "shell final");

	gateway.release(3);
	let unknown = wait_snapshot(&mut debug, &raw_capture, "unknown generic card", |snapshot| {
		snapshot.combined().contains("p7_unknown")
			&& snapshot.combined().contains("mystery.fixture")
			&& snapshot.tree.to_string().contains("unknown-1")
	});
	assert_surface(&unknown, "unknown");
	gateway.release(4);
	let summary = wait_snapshot(&mut debug, &raw_capture, "first turn complete", |snapshot| {
		snapshot.frame.contains("deterministic tool sequence is complete")
	});
	assert_surface(&summary, "summary");

	gateway.release(5);
	debug.keys("'interrupt the batch' enter");
	let batch_live = wait_snapshot(&mut debug, &raw_capture, "batch running", |snapshot| {
		snapshot.combined().contains("batch-one-started")
	});
	assert_surface(&batch_live, "batch live");

	process.resize(32, 92);
	debug.op("resize").unwrap_or_else(|error| panic!("resize injection failed: {error}"));
	let resized = wait_snapshot(&mut debug, &raw_capture, "streaming resize", |snapshot| {
		snapshot.frame.contains("batch-one-started")
			&& snapshot.frame.contains("read scratch.txt")
			&& snapshot.frame.contains("mystery.fixture")
	});
	assert_surface(&resized, "resized");
	let info = debug.op("info").expect("resize info");
	assert_eq!(info.get("rows").and_then(Value::as_u64), Some(32), "resize rows: {info}");
	assert_eq!(info.get("cols").and_then(Value::as_u64), Some(92), "resize cols: {info}");
	assert_eq!(info.get("alt_screen").and_then(Value::as_bool), Some(false), "chat entered alt screen: {info}");

	debug.keys("esc");
	let interrupted = wait_snapshot(&mut debug, &raw_capture, "batch interrupted and skipped", |snapshot| {
		let frame = snapshot.frame.to_ascii_lowercase();
		frame.contains("interrupt") && frame.matches("skipped").count() >= 2
	});
	assert_surface(&interrupted, "interrupt");
	assert!(!interrupted.frame.contains("batch-two-ran") && !interrupted.frame.contains("batch-three-ran"), "skipped tools executed:\n{}", interrupted.frame);

	gateway.release(6);
	let steering = wait_snapshot(&mut debug, &raw_capture, "steering next turn", |snapshot| {
		let frame = snapshot.frame.to_ascii_lowercase();
		frame.contains("steering item") && frame.contains("next turn")
	});
	assert_surface(&steering, "steering");

	debug.keys("'/quit' enter");
	drop(debug);
	let before = process.before.clone();
	let (status, raw, stdout, stderr, after) = process.wait(READY_TIMEOUT);
	let diagnostics = format!(
		"status={status}\nstdout={stdout}\nstderr={stderr}\nlast frame={}\nraw={}",
		steering.frame,
		visible(&raw),
	);
	assert!(status.success(), "omp chat did not exit cleanly\n{diagnostics}");
	assert_restored(&raw, &before, &after, &diagnostics);
}
