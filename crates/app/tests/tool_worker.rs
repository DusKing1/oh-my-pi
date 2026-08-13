#![cfg(unix)]

use std::{
	fs,
	path::Path,
	time::{Duration, Instant},
};

use bytes::Bytes;
use nix::{errno::Errno, sys::signal, unistd::Pid};
use omp_app::envd::worker::{
	CommittedToolCall, ToolWorkerConfig, ToolWorkerSupervisor, WorkerAbortKind, WorkerEvent,
};
use omp_core::Str;
use omp_proto::{thread::v1::part, toolhost::v1::ToolComplete};
use serde_json::{Value, json};

const EXTENSION: &str = r#"
import ctypes
import os
import signal

# The supervisor's SIGINT is explicitly a courtesy. Ignore it so the native
# sleep proves cancellation reaches the grace deadline and requires SIGKILL.
signal.signal(signal.SIGINT, signal.SIG_IGN)

_libc = ctypes.CDLL(None)
_sleep = _libc.sleep
_sleep.argtypes = [ctypes.c_uint]
_sleep.restype = ctypes.c_uint


def echo_update(params):
    if set(params) != {"message", "commit_seal"} or params["commit_seal"] != "committed":
        raise RuntimeError("tool observed arguments before commitment")
    result = {
        "message": params["message"],
        "commit_seal": params["commit_seal"],
        "pid": os.getpid(),
    }
    return {
        "updates": [result],
        "parts": [params["message"]],
        "details": result,
    }


def native_block(params):
    with open(params["started"], "w", encoding="utf-8") as marker:
        marker.write(str(os.getpid()))
        marker.flush()
    _sleep(params["seconds"])
    return {"parts": ["native sleep returned"], "details": {"pid": os.getpid()}}


OMP_TOOLS = [
    {
        "name": "echo_update",
        "description": "echoes one committed invocation and emits an update",
        "schema": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "commit_seal": {"const": "committed"},
            },
            "required": ["message", "commit_seal"],
            "additionalProperties": False,
        },
        "rev": "r1",
        "strict": True,
        "handler": echo_update,
    },
    {
        "name": "native_block",
        "description": "blocks in the platform C sleep function",
        "schema": {
            "type": "object",
            "properties": {
                "started": {"type": "string"},
                "seconds": {"type": "integer"},
            },
            "required": ["started", "seconds"],
            "additionalProperties": False,
        },
        "rev": "r1",
        "strict": True,
        "handler": native_block,
    },
]
"#;

#[tokio::test]
async fn same_binary_worker_kills_native_call_and_respawns() {
	let site = tempfile::tempdir().expect("Python site scratch directory");
	fs::write(site.path().join("phase1_worker_tools.py"), EXTENSION)
		.expect("write temporary Python extension");

	let mut config = ToolWorkerConfig::new(env!("CARGO_BIN_EXE_omp").into());
	config.python_site = Some(site.path().to_owned());
	config.modules = vec![Str::new_static("phase1_worker_tools")];
	config.health_timeout = Duration::from_secs(5);
	config.interrupt_grace = Duration::from_millis(250);
	config.initial_backoff = Duration::from_millis(10);
	config.max_backoff = Duration::from_millis(50);
	let interrupt_grace = config.interrupt_grace;

	let supervisor =
		tokio::time::timeout(Duration::from_secs(10), ToolWorkerSupervisor::spawn(config))
			.await
			.expect("worker hello and registration timed out")
			.expect("spawn same-binary Python worker");

	let names = supervisor
		.registrations()
		.iter()
		.map(|decl| {
			decl
				.definition
				.as_ref()
				.expect("registered definition")
				.name
				.as_str()
		})
		.collect::<Vec<_>>();
	assert_eq!(names, ["echo_update", "native_block"]);
	assert!(
		supervisor
			.registrations()
			.iter()
			.all(|decl| decl.rev == "r1")
	);

	let (first_update, first_complete) = tokio::time::timeout(
		Duration::from_secs(5),
		echo_roundtrip(&supervisor, "echo-before", "before kill"),
	)
	.await
	.expect("initial echo invocation timed out");
	assert_eq!(first_update["message"], "before kill");
	assert_eq!(first_update["commit_seal"], "committed");
	assert_eq!(completion_text(&first_complete), "before kill");
	let first_details: Value =
		serde_json::from_slice(&first_complete.details_json).expect("echo completion details JSON");
	assert_eq!(first_details, first_update);
	let first_pid = first_details["pid"]
		.as_i64()
		.expect("worker pid in echo details") as i32;

	let started = site.path().join("native-call-started");
	let mut blocked = supervisor
		.invoke_committed(call(
			"native-block",
			"native_block",
			json!({ "started": started, "seconds": 30 }),
			Duration::from_secs(60),
		))
		.expect("dispatch committed native invocation");
	let blocked_pid = wait_for_marker(&started).await;
	assert_eq!(blocked_pid, first_pid, "native call did not run in the warm worker");

	blocked.interrupt("courtesy interrupt");
	tokio::time::sleep(Duration::from_millis(75)).await;
	assert!(
		signal::kill(Pid::from_raw(blocked_pid), None).is_ok(),
		"courtesy interrupt structurally killed worker {blocked_pid}",
	);

	let cancelled_at = Instant::now();
	blocked.cancel("integration cancellation");
	let abort = match tokio::time::timeout(Duration::from_secs(3), blocked.next())
		.await
		.expect("native cancellation exceeded grace plus kill window")
		.expect("supervisor closed before reporting cancellation")
	{
		WorkerEvent::Aborted(abort) => abort,
		WorkerEvent::Update(_) => panic!("native blocker unexpectedly emitted an update"),
		WorkerEvent::Complete(_) => panic!("native blocker completed instead of being killed"),
	};
	let cancel_elapsed = cancelled_at.elapsed();
	assert_eq!(abort.kind, WorkerAbortKind::Cancelled);
	assert!(abort.effects_unknown, "dispatched worker cancellation must report effects unknown");
	assert!(
		cancel_elapsed >= interrupt_grace.saturating_sub(Duration::from_millis(25)),
		"native call ended cooperatively before the hard-kill grace elapsed: {cancel_elapsed:?}"
	);
	assert!(
		matches!(signal::kill(Pid::from_raw(blocked_pid), None), Err(Errno::ESRCH)),
		"cancelled native worker process {blocked_pid} is still alive"
	);

	let (second_update, second_complete) = tokio::time::timeout(
		Duration::from_secs(5),
		echo_roundtrip(&supervisor, "echo-after", "after respawn"),
	)
	.await
	.expect("respawned worker did not serve the next invocation");
	assert_eq!(second_update["message"], "after respawn");
	assert_eq!(completion_text(&second_complete), "after respawn");
	let second_details: Value =
		serde_json::from_slice(&second_complete.details_json).expect("respawn echo details JSON");
	let second_pid = second_details["pid"]
		.as_i64()
		.expect("respawned worker pid") as i32;
	assert_ne!(second_pid, blocked_pid, "supervisor reused the cancelled worker process");

	supervisor.shutdown().await;
}

fn call(
	call_id: &'static str,
	name: &'static str,
	args: Value,
	deadline: Duration,
) -> CommittedToolCall {
	CommittedToolCall {
		call_id: Str::new_static(call_id),
		name: Str::new_static(name),
		rev: Str::new_static("r1"),
		args_json: Bytes::from(serde_json::to_vec(&args).expect("serialize committed arguments")),
		deadline,
	}
}

async fn echo_roundtrip(
	supervisor: &ToolWorkerSupervisor,
	call_id: &'static str,
	message: &'static str,
) -> (Value, ToolComplete) {
	let mut invocation = supervisor
		.invoke_committed(call(
			call_id,
			"echo_update",
			json!({ "message": message, "commit_seal": "committed" }),
			Duration::from_secs(5),
		))
		.expect("dispatch committed echo invocation");
	let update = match invocation.next().await.expect("echo update event") {
		WorkerEvent::Update(update) => update,
		WorkerEvent::Complete(_) => panic!("echo completed before its update"),
		WorkerEvent::Aborted(abort) => panic!("echo aborted: {}", abort.reason),
	};
	assert_eq!(update.call_id, call_id);
	let update = serde_json::from_slice(&update.json).expect("echo update JSON");
	let complete = match invocation.next().await.expect("echo completion event") {
		WorkerEvent::Complete(complete) => complete,
		WorkerEvent::Update(_) => panic!("echo emitted an unexpected second update"),
		WorkerEvent::Aborted(abort) => panic!("echo aborted after update: {}", abort.reason),
	};
	assert_eq!(complete.call_id, call_id);
	assert!(!complete.is_error, "echo completion reported an error");
	(update, complete)
}

async fn wait_for_marker(path: &Path) -> i32 {
	tokio::time::timeout(Duration::from_secs(3), async {
		loop {
			if let Ok(pid) = fs::read_to_string(path) {
				return pid.parse().expect("native marker contains worker pid");
			}
			tokio::time::sleep(Duration::from_millis(10)).await;
		}
	})
	.await
	.expect("native Python call did not enter ctypes sleep")
}

fn completion_text(complete: &ToolComplete) -> &str {
	match complete.parts.as_slice() {
		[part] => match part.kind.as_ref() {
			Some(part::Kind::Text(text)) => text,
			other => panic!("expected one text completion part, got {other:?}"),
		},
		parts => panic!("expected one completion part, got {}", parts.len()),
	}
}
