#![cfg(unix)]

use std::{
	fs,
	future::Future,
	os::unix::ffi::OsStrExt as _,
	path::{Path, PathBuf},
	sync::{
		Arc,
		atomic::{AtomicBool, Ordering},
	},
	time::{Duration, Instant},
};

use async_stream::stream;
use bytes::Bytes;
use futures::Stream;
use nix::{
	errno::Errno,
	sys::signal,
	unistd::{Pid, getpgid},
};
use omp_app::envd::{
	server::EnvServer,
	worker::ToolWorkerConfig,
};
use omp_core::Str;
use omp_e2e::support::omp_binary;
use omp_env::{EnvClient, ExecEvent, Invocation, InvocationEvent};
use omp_proto::{
	SCHEMA_REV,
	env::v1::{
		ClientHello, ExecOutcome, ExecRequest, ExecStatusMsg, InvokeTool, OpenSessionRequest,
		Script,
	},
};
use omp_tool::{
	Abort, Constraint, Ev, IncomingParams, Outcome, ParamError, Part, PromptCaps, Registry, Rev,
	Tool, ToolSpec, Verdict,
};
use serde_json::{Value, json};
use tempfile::TempDir;

const STARTUP_DEADLINE: Duration = Duration::from_secs(10);
const EVENT_DEADLINE: Duration = Duration::from_secs(5);
const POLL_INTERVAL: Duration = Duration::from_millis(10);
const NATIVE_SLEEP: Duration = Duration::from_millis(400);

const PY_EXTENSION: &str = r#"
import ctypes
import os
import signal

# A courtesy interpreter signal must not be the cancellation mechanism.
signal.signal(signal.SIGINT, signal.SIG_IGN)
_sleep = ctypes.CDLL(None).sleep
_sleep.argtypes = [ctypes.c_uint]
_sleep.restype = ctypes.c_uint


def block(params):
    with open(params["started"], "w", encoding="utf-8") as marker:
        marker.write(str(os.getpid()))
        marker.flush()
    _sleep(3600)
    return {"parts": [], "details": {"unexpected": "ctypes sleep returned"}}


def echo(params):
    return {
        "parts": [],
        "details": {"message": params["message"], "pid": os.getpid()},
    }


OMP_TOOLS = [
    {
        "name": "matrix_block",
        "description": "blocks in a native C sleep until its worker is killed",
        "schema": {
            "type": "object",
            "properties": {"started": {"type": "string"}},
            "required": ["started"],
            "additionalProperties": False,
        },
        "rev": "py.1",
        "strict": True,
        "handler": block,
    },
    {
        "name": "matrix_echo",
        "description": "proves the replacement worker serves the next call",
        "schema": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
            "additionalProperties": False,
        },
        "rev": "py.1",
        "strict": True,
        "handler": echo,
    },
]
"#;

struct LocalEnv {
	client:      EnvClient,
	root:        TempDir,
	_state:      TempDir,
	server_task: tokio::task::JoinHandle<()>,
}

impl LocalEnv {
	async fn start(registry: Registry, worker: ToolWorkerConfig) -> Self {
		let root = tempfile::tempdir().expect("workspace scratch directory");
		let state = tempfile::tempdir().expect("environment state scratch directory");
		let server = Arc::new(
			EnvServer::open_local(root.path(), state.path(), registry, worker)
				.await
				.expect("open real local environment authority"),
		);
		let (client, transport) = EnvClient::in_process(64);
		let server_task = {
			let server = Arc::clone(&server);
			tokio::spawn(async move { server.serve_in_process(transport).await })
		};
		client
			.hello(ClientHello {
				client: "p2-cancel-matrix".into(),
				schema_rev: SCHEMA_REV,
				..ClientHello::default()
			})
			.await
			.expect("environment hello");
		Self { client, root, _state: state, server_task }
	}
}

impl Drop for LocalEnv {
	fn drop(&mut self) {
		self.server_task.abort();
	}
}

struct DropProbe(Arc<AtomicBool>);

impl Drop for DropProbe {
	fn drop(&mut self) {
		self.0.store(true, Ordering::Release);
	}
}

struct CancellableSleeper {
	spec:    ToolSpec,
	started: PathBuf,
	marker:  PathBuf,
	dropped: Arc<AtomicBool>,
}

impl CancellableSleeper {
	fn new(started: PathBuf, marker: PathBuf, dropped: Arc<AtomicBool>) -> Self {
		Self {
			spec: ToolSpec {
				name: Str::new_static("matrix_sleeper"),
				rev: Rev { family: Str::new_static("e2e"), n: 1 },
				description: Str::new_static("sleeps before attempting a marker mutation"),
				schema: Bytes::from_static(br#"{"type":"object"}"#),
				constraint: Constraint::None,
			},
			started,
			marker,
			dropped,
		}
	}
}

impl Tool for CancellableSleeper {
	type Fault = Value;
	type Params = Value;
	type Payload = Value;
	type Update = Value;

	fn spec(&self) -> &ToolSpec {
		&self.spec
	}

	fn call<'c>(
		&'c self,
		mut params: IncomingParams<'c>,
	) -> impl Stream<Item = Ev<Value, Value, Value>> + Send + 'c {
		stream! {
			if params.committed().await.is_err() {
				yield Ev::Aborted(Abort::InputDropped);
				return;
			}
			let started = self.started.clone();
			let marker = self.marker.clone();
			let dropped = Arc::clone(&self.dropped);
			let result = params.interruptable().pull(|mut doc| async move {
				let _: Value = doc.whole().await?;
				fs::write(started, b"sleeping").expect("write sleeper start marker");
				let _drop_probe = DropProbe(dropped);
				tokio::time::sleep(NATIVE_SLEEP).await;
				fs::write(marker, b"mutated").expect("write forbidden mutation marker");
				Ok(())
			}).await;
			match result {
				Err(ParamError::Interrupted(interrupt)) => {
					yield Ev::Aborted(Abort::Interrupted { reason: interrupt.reason });
				},
				Err(error) => yield Ev::Done(Outcome::Done {
					result: Err(json!({"error": error.to_string()})),
					useless: false,
				}),
				Ok(()) => yield Ev::Done(Outcome::Done {
					result: Ok(json!({"unexpected": "sleep completed"})),
					useless: false,
				}),
			}
		}
	}

	fn prompt(&self, _view: Result<&Value, &Value>, _caps: &PromptCaps) -> Vec<Part> {
		Vec::new()
	}
}

#[tokio::test]
async fn rust_drop_cancellation_is_exact_and_cannot_mutate_after_interrupt() {
	let scratch = tempfile::tempdir().expect("native cancellation scratch");
	let started = scratch.path().join("sleep-started");
	let marker = scratch.path().join("must-not-exist");
	let dropped = Arc::new(AtomicBool::new(false));
	let mut registry = Registry::new();
	registry
		.register(CancellableSleeper::new(
			started.clone(),
			marker.clone(),
			Arc::clone(&dropped),
		))
		.expect("register cancellable sleeper");
	let worker = ToolWorkerConfig::new(omp_binary().expect("resolve worker-capable omp binary"));
	let env = within(STARTUP_DEADLINE, LocalEnv::start(registry, worker)).await;

	let mut skipped = within(
		EVENT_DEADLINE,
		env.client.invoke(InvokeTool {
			invocation_id: "rust-skipped".into(),
			name: "matrix_sleeper".into(),
			rev: "e2e.1".into(),
			..InvokeTool::default()
		}),
	)
	.await
	.expect("open precommit sleeper");
	expect_accepted(&mut skipped).await;
	skipped.guard().cancel();
	let skipped_terminal = next_verdict(&mut skipped).await;
	assert_eq!(
		decode_verdict(&skipped_terminal),
		Verdict::Aborted(Abort::Skipped {
			reason: Str::new_static("invocation cancelled before argument commitment"),
		})
	);
	assert_eq!(
		skipped_terminal.json,
		Bytes::from_static(
			br#"{"kind":"aborted","value":{"kind":"skipped","reason":"invocation cancelled before argument commitment"}}"#,
		),
	);
	assert_abort_envelope(&skipped_terminal);

	let mut interrupted = within(
		EVENT_DEADLINE,
		env.client.invoke(InvokeTool {
			invocation_id: "rust-interrupted".into(),
			name: "matrix_sleeper".into(),
			rev: "e2e.1".into(),
			..InvokeTool::default()
		}),
	)
	.await
	.expect("open committed sleeper");
	expect_accepted(&mut interrupted).await;
	within(EVENT_DEADLINE, interrupted.commit_args(Bytes::from_static(b"{}")))
		.await
		.expect("commit sleeper arguments");
	wait_for_file(&started, EVENT_DEADLINE).await;
	interrupted.guard().cancel();
	let interrupted_terminal = next_verdict(&mut interrupted).await;
	assert_eq!(
		decode_verdict(&interrupted_terminal),
		Verdict::Aborted(Abort::Interrupted {
			reason: Str::new_static("invocation cancelled by client"),
		})
	);
	assert_eq!(
		interrupted_terminal.json,
		Bytes::from_static(
			br#"{"kind":"aborted","value":{"kind":"interrupted","reason":"invocation cancelled by client"}}"#,
		),
	);
	assert_abort_envelope(&interrupted_terminal);
	assert!(dropped.load(Ordering::Acquire), "the sleeping Rust future was not dropped");
	assert!(!marker.exists(), "cancelled Rust future mutated its marker");

	// Wait beyond the original sleep. This catches implementations that detach the
	// mutation instead of making the operation future genuinely drop-cancellable.
	tokio::time::sleep(NATIVE_SLEEP + Duration::from_millis(200)).await;
	assert!(!marker.exists(), "a detached mutation escaped cancellation");
}

#[tokio::test]
async fn dropping_exec_run_kills_the_whole_pgid_but_preserves_its_session() {
	let worker = ToolWorkerConfig::new(omp_binary().expect("resolve worker-capable omp binary"));
	let env = within(STARTUP_DEADLINE, LocalEnv::start(Registry::new(), worker)).await;
	let opened = within(
		EVENT_DEADLINE,
		env.client.open_session(OpenSessionRequest {
			cwd_uri: file_uri(env.root.path()),
			..OpenSessionRequest::default()
		}),
	)
	.await
	.expect("open persistent exec session");

	let retained = env.root.path().join("retained-cwd");
	let parent_marker = env.root.path().join("term-ignoring-parent.pid");
	let grandchild_marker = env.root.path().join("term-ignoring-grandchild.pid");
	let script = format!(
		"mkdir retained-cwd; cd retained-cwd; sh -c 'trap \"\" TERM; echo $$ > \"{}\"; (trap \"\" TERM; sleep 3600) & echo $! > \"{}\"; wait'",
		parent_marker.display(),
		grandchild_marker.display(),
	);
	let mut run = within(
		EVENT_DEADLINE,
		env.client.exec(exec_request(&opened.session, script)),
	)
	.await
	.expect("start TERM-ignoring process tree");
	match next_exec_event(&mut run).await {
		ExecEvent::Started(_) => {},
		other => panic!("first exec event was not Started: {other:?}"),
	}
	let parent = wait_for_pid(&parent_marker, EVENT_DEADLINE).await;
	let grandchild = wait_for_pid(&grandchild_marker, EVENT_DEADLINE).await;
	let parent_group = getpgid(Some(parent)).expect("read parent process group");
	let grandchild_group = getpgid(Some(grandchild)).expect("read grandchild process group");
	assert_eq!(parent_group, grandchild_group, "descendant escaped the command process group");
	assert!(process_alive(parent) && process_alive(grandchild));

	// ExecRun owns only the request-scoped RunGuard. The persistent session is a
	// distinct server resource and must remain alive after this drop.
	drop(run);
	wait_process_group_dead(parent_group, EVENT_DEADLINE).await;
	assert!(!process_alive(parent), "TERM-ignoring parent survived guard drop");
	assert!(!process_alive(grandchild), "TERM-ignoring grandchild survived guard drop");

	// Retain a second stream while exercising the same RunGuard cancellation
	// request so the resource-owned terminal shape remains observable. Dropping
	// the first guard above is what proved tree teardown; this companion run
	// proves that teardown is classified, not merely inferred from dead PIDs.
	let mut status_probe = within(
		EVENT_DEADLINE,
		env.client.exec(exec_request(&opened.session, "sleep 3600")),
	)
	.await
	.expect("start cancellation status probe in the surviving session");
	match next_exec_event(&mut status_probe).await {
		ExecEvent::Started(_) => {},
		other => panic!("status probe first event was not Started: {other:?}"),
	}
	status_probe.guard().cancel();
	let (_, cancelled) = collect_exec(&mut status_probe).await;
	assert_eq!(cancelled.outcome, ExecOutcome::Cancelled as i32);
	assert_eq!(cancelled.exit_code, None);
	assert!(cancelled.signal.is_empty());
	assert!(cancelled.aborted);

	let mut pwd = within(
		EVENT_DEADLINE,
		env.client.exec(exec_request(&opened.session, "pwd")),
	)
	.await
	.expect("same session accepts a command after cancellation");
	let (output, status) = collect_exec(&mut pwd).await;
	assert_eq!(status.outcome, ExecOutcome::Exited as i32);
	assert_eq!(status.exit_code, Some(0));
	assert!(!status.aborted);
	let expected = retained.as_os_str().as_bytes();
	assert!(
		output.windows(expected.len()).any(|window| window == expected),
		"persistent session lost its cwd",
	);
}

#[tokio::test]
async fn python_native_sleep_requires_sigkill_then_respawns_and_serves() {
	let site = tempfile::tempdir().expect("Python extension scratch");
	fs::write(site.path().join("cancel_matrix_tools.py"), PY_EXTENSION)
		.expect("write Python cancellation extension");
	let mut worker = ToolWorkerConfig::new(omp_binary().expect("resolve worker-capable omp binary"));
	worker.python_site = Some(site.path().to_owned());
	worker.modules = vec![Str::new_static("cancel_matrix_tools")];
	worker.health_timeout = Duration::from_secs(5);
	worker.interrupt_grace = Duration::from_millis(150);
	worker.initial_backoff = Duration::from_millis(10);
	worker.max_backoff = Duration::from_millis(50);
	let hard_kill_grace = worker.interrupt_grace;
	let env = within(STARTUP_DEADLINE, LocalEnv::start(Registry::new(), worker)).await;
	let started = site.path().join("ctypes-sleep-started");

	let mut blocked = within(
		EVENT_DEADLINE,
		env.client.invoke(InvokeTool {
			invocation_id: "python-blocked".into(),
			name: "matrix_block".into(),
			rev: "py.1".into(),
			deadline_ms: 10_000,
			..InvokeTool::default()
		}),
	)
	.await
	.expect("open Python native blocker");
	expect_accepted(&mut blocked).await;
	let args = serde_json::to_vec(&json!({
		"started": started.to_str().expect("temporary path is UTF-8"),
	}))
	.expect("serialize Python blocker arguments");
	within(EVENT_DEADLINE, blocked.commit_args(Bytes::from(args)))
		.await
		.expect("commit Python blocker arguments");
	let blocked_pid = wait_for_pid(&started, EVENT_DEADLINE).await;

	// This is deliberately weaker than cancellation. The worker cannot read the
	// protocol frame while ctypes holds its thread, and SIGINT is ignored.
	within(
		EVENT_DEADLINE,
		blocked.interrupt(Str::new_static("courtesy interpreter interrupt")),
	)
	.await
	.expect("send courtesy worker interrupt");
	tokio::time::sleep(hard_kill_grace + Duration::from_millis(75)).await;
	assert!(process_alive(blocked_pid), "courtesy interrupt killed the native worker");
	assert!(
		tokio::time::timeout(Duration::from_millis(50), blocked.next_event())
			.await
			.is_err(),
		"blocking ctypes call returned to the interpreter",
	);

	let cancelled_at = Instant::now();
	blocked.guard().cancel();
	let terminal = next_verdict(&mut blocked).await;
	let elapsed = cancelled_at.elapsed();
	let Verdict::Aborted(Abort::EffectsUnknown { reason }) = decode_verdict(&terminal) else {
		panic!("dispatched Python cancellation was not effects-unknown");
	};
	assert_eq!(reason, "environment invocation cancelled");
	assert_abort_envelope(&terminal);
	assert!(
		elapsed >= hard_kill_grace.saturating_sub(Duration::from_millis(25)),
		"worker exited before the SIGKILL grace elapsed: {elapsed:?}",
	);
	assert!(!process_alive(blocked_pid), "SIGKILL did not terminate worker {blocked_pid}");

	let mut next = within(
		EVENT_DEADLINE,
		env.client.invoke(InvokeTool {
			invocation_id: "python-after-respawn".into(),
			name: "matrix_echo".into(),
			rev: "py.1".into(),
			deadline_ms: 5_000,
			..InvokeTool::default()
		}),
	)
	.await
	.expect("open Python call after supervisor respawn");
	expect_accepted(&mut next).await;
	within(
		EVENT_DEADLINE,
		next.commit_args(Bytes::from_static(br#"{"message":"after respawn"}"#)),
	)
	.await
	.expect("commit post-respawn Python call");
	let success = next_verdict(&mut next).await;
	assert!(!success.is_error);
	assert!(!success.useless);
	assert!(success.parts.is_empty());
	let Verdict::Ok(details) = decode_verdict(&success) else {
		panic!("replacement worker did not return an ok verdict");
	};
	assert_eq!(details["message"], "after respawn");
	let replacement_pid = details["pid"].as_i64().expect("replacement worker pid") as i32;
	assert_ne!(replacement_pid, blocked_pid.as_raw(), "supervisor reused the killed worker");
	assert!(process_alive(Pid::from_raw(replacement_pid)));
}

async fn within<T>(deadline: Duration, future: impl Future<Output = T>) -> T {
	tokio::time::timeout(deadline, future)
		.await
		.expect("operation exceeded its hard deadline")
}

async fn expect_accepted(invocation: &mut Invocation) {
	match within(EVENT_DEADLINE, invocation.next_event())
		.await
		.expect("invocation event")
		.expect("invocation stream closed before acceptance")
	{
		InvocationEvent::Accepted(_) => {},
		other => panic!("first invocation event was not Accepted: {other:?}"),
	}
}

async fn next_verdict(invocation: &mut Invocation) -> omp_proto::env::v1::Verdict {
	within(EVENT_DEADLINE, async {
		loop {
			match invocation
				.next_event()
				.await
				.expect("invocation event")
				.expect("invocation stream closed before terminal truth")
			{
				InvocationEvent::Verdict(verdict) => return verdict,
				InvocationEvent::Update(_) => {},
				InvocationEvent::Accepted(_) => panic!("invocation was accepted twice"),
				InvocationEvent::StreamError(error) => {
					panic!("invocation stream lost continuity: {}", error.message)
				},
			}
		}
	})
	.await
}

fn decode_verdict(terminal: &omp_proto::env::v1::Verdict) -> Verdict<Value, Value> {
	serde_json::from_slice(&terminal.json).expect("decode structured terminal verdict")
}

fn assert_abort_envelope(terminal: &omp_proto::env::v1::Verdict) {
	assert!(terminal.is_error);
	assert!(!terminal.useless);
	assert!(terminal.parts.is_empty());
}

fn file_uri(path: &Path) -> String {
	url::Url::from_directory_path(path)
		.expect("temporary directory has a file URI")
		.to_string()
}

fn exec_request(session: &[u8], script: impl Into<String>) -> ExecRequest {
	ExecRequest {
		session: Bytes::copy_from_slice(session),
		source: Some(Script { text: script.into(), ..Script::default() }),
		..ExecRequest::default()
	}
}

async fn next_exec_event(run: &mut omp_env::ExecRun) -> ExecEvent {
	within(EVENT_DEADLINE, run.next_event())
		.await
		.expect("exec event")
		.expect("exec stream closed before terminal status")
}

async fn collect_exec(run: &mut omp_env::ExecRun) -> (Vec<u8>, ExecStatusMsg) {
	within(EVENT_DEADLINE, async {
		let mut output = Vec::new();
		loop {
			match run
				.next_event()
				.await
				.expect("exec event")
				.expect("exec stream closed before exit")
			{
				ExecEvent::Started(_) => {},
				ExecEvent::Output(frame) => output.extend_from_slice(&frame.data),
				ExecEvent::Exit(exit) => return (output, exit.status.expect("exec terminal status")),
				ExecEvent::StreamError(error) => panic!("exec stream error: {}", error.message),
			}
		}
	})
	.await
}

async fn wait_for_file(path: &Path, deadline: Duration) {
	within(deadline, async {
		while !path.exists() {
			tokio::time::sleep(POLL_INTERVAL).await;
		}
	})
	.await;
}

async fn wait_for_pid(path: &Path, deadline: Duration) -> Pid {
	within(deadline, async {
		loop {
			if let Ok(raw) = fs::read(path)
				&& let Some(pid) = parse_ascii_pid(&raw)
			{
				return Pid::from_raw(pid);
			}
			tokio::time::sleep(POLL_INTERVAL).await;
		}
	})
	.await
}

fn parse_ascii_pid(raw: &[u8]) -> Option<i32> {
	let raw = raw.strip_suffix(b"\n").unwrap_or(raw);
	(!raw.is_empty()).then_some(())?;
	raw.iter().try_fold(0_i32, |pid, byte| {
		if !byte.is_ascii_digit() {
			return None;
		}
		pid.checked_mul(10)?.checked_add(i32::from(*byte - b'0'))
	})
}

fn process_alive(pid: Pid) -> bool {
	match signal::kill(pid, None) {
		Ok(()) | Err(Errno::EPERM) => true,
		Err(Errno::ESRCH) => false,
		Err(error) => panic!("process liveness probe failed: {error}"),
	}
}

fn process_group_alive(group: Pid) -> bool {
	process_alive(Pid::from_raw(-group.as_raw()))
}

async fn wait_process_group_dead(group: Pid, deadline: Duration) {
	within(deadline, async {
		while process_group_alive(group) {
			tokio::time::sleep(POLL_INTERVAL).await;
		}
	})
	.await;
}
