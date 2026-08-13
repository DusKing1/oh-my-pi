#![cfg(unix)]

use std::{
	path::{Path, PathBuf},
	sync::Arc,
	time::Duration,
};

use async_stream::stream;
use bytes::Bytes;
use futures::Stream;
use omp_app::envd::{
	exec::{ExecEvent as HostExecEvent, ExecHost},
	server::EnvServer,
	worker::{PY_EVAL_MODULE, ToolWorkerConfig, ToolWorkerSupervisor},
	workspace::{WorkspaceError, WorkspaceHost},
};
use omp_core::Str;
use omp_env::{BlobDownloadEvent, EnvClient, ExecEvent, InvocationEvent, ProcessAttachmentEvent};
use omp_proto::{
	SCHEMA_REV,
	blob::v1::{Chunk, GetRequest},
	env::v1::{
		ClientHello, ExecOutcome, ExecRequest, InvokeTool, ListProcesses, OpenSessionRequest,
		ProcessSpec, Script, StartProcess, StopProcess,
	},
};
use omp_tool::{
	Abort, Constraint, Ev, IncomingParams, LoweringCaps, Outcome, ParamError, Part, PromptCaps,
	Registry, Rev, Tool, ToolRoute, ToolSpec, Verdict,
};
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio_util::sync::CancellationToken;

struct EffectTool {
	spec:   ToolSpec,
	marker: PathBuf,
}

impl EffectTool {
	fn new(marker: PathBuf) -> Self {
		Self {
			spec: ToolSpec {
				name:        Str::new_static("effect_probe"),
				rev:         Rev { family: Str::new_static("test"), n: 1 },
				description: Str::new_static("records a committed invocation"),
				schema:      Bytes::from_static(br#"{"type":"object"}"#),
				constraint:  Constraint::None,
			},
			marker,
		}
	}
}

impl Tool for EffectTool {
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
			match params.whole::<Value>().await {
				Ok(value) => {
					std::fs::write(&self.marker, b"committed").expect("write effect marker");
					yield Ev::Done(Outcome::Done { result: Ok(value), useless: true });
				},
				Err(error) => yield Ev::Done(Outcome::Done {
					result: Err(json!({"error": error.to_string()})),
					useless: false,
				}),
			}
		}
	}

	fn prompt(&self, _view: Result<&Value, &Value>, _caps: &PromptCaps) -> Vec<Part> {
		Vec::new()
	}
}
struct SpeculativeLease {
	marker: PathBuf,
}

impl Drop for SpeculativeLease {
	fn drop(&mut self) {
		let _ = std::fs::remove_file(&self.marker);
	}
}

struct StreamingTool {
	spec:   ToolSpec,
	lease:  PathBuf,
	effect: PathBuf,
}

impl StreamingTool {
	fn new(lease: PathBuf, effect: PathBuf) -> Self {
		Self {
			spec: ToolSpec {
				name:        Str::new_static("streaming_probe"),
				rev:         Rev { family: Str::new_static("test"), n: 1 },
				description: Str::new_static("prepares from streamed arguments before commitment"),
				schema:      Bytes::from_static(
					br#"{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}"#,
				),
				constraint:  Constraint::None,
			},
			lease,
			effect,
		}
	}
}

impl Tool for StreamingTool {
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
			let path = match params.pull(|mut doc| async move {
				doc.json().object().key("path").string().finish().await
			}).await {
				Ok(path) => path,
				Err(_) => {
					yield Ev::Aborted(Abort::InputDropped);
					return;
				},
			};
			std::fs::write(&self.lease, path.as_bytes()).expect("open speculative lease");
			let _lease = SpeculativeLease { marker: self.lease.clone() };
			yield Ev::Update(json!({"state": "prepared", "path": path}));
			if params.committed().await.is_err() {
				yield Ev::Aborted(Abort::InputDropped);
				return;
			}
			std::fs::write(&self.effect, path.as_bytes()).expect("record committed effect");
			tokio::time::sleep(Duration::from_millis(100)).await;
			yield Ev::Done(Outcome::Done {
				result: Ok(json!({"path": path})),
				useless: false,
			});
		}
	}

	fn prompt(&self, _view: Result<&Value, &Value>, _caps: &PromptCaps) -> Vec<Part> {
		Vec::new()
	}
}


struct BlockingTool {
	spec:    ToolSpec,
	started: PathBuf,
}

impl BlockingTool {
	fn new(started: PathBuf) -> Self {
		Self {
			spec: ToolSpec {
				name:        Str::new_static("native_block"),
				rev:         Rev { family: Str::new_static("test"), n: 1 },
				description: Str::new_static("waits until the environment cancels it"),
				schema:      Bytes::from_static(br#"{"type":"object"}"#),
				constraint:  Constraint::None,
			},
			started,
		}
	}
}

impl Tool for BlockingTool {
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
			match params.committed().await {
				Ok(_) => {
					std::fs::write(&self.started, b"started").expect("write native start marker");
					yield Ev::Update(json!({"state": "started"}));
					std::future::pending::<()>().await;
				},
				Err(_) => yield Ev::Aborted(Abort::InputDropped),
			}
		}
	}

	fn prompt(&self, _view: Result<&Value, &Value>, _caps: &PromptCaps) -> Vec<Part> {
		Vec::new()
	}
}

struct CooperativeInterruptTool {
	spec: ToolSpec,
}

impl CooperativeInterruptTool {
	fn new() -> Self {
		Self {
			spec: ToolSpec {
				name:        Str::new_static("cooperative_interrupt"),
				rev:         Rev { family: Str::new_static("test"), n: 1 },
				description: Str::new_static("reports cooperative interrupt truth"),
				schema:      Bytes::from_static(br#"{"type":"object"}"#),
				constraint:  Constraint::None,
			},
		}
	}
}

impl Tool for CooperativeInterruptTool {
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
			yield Ev::Update(json!({"state": "waiting"}));
			let interrupted: Result<(), ParamError> = params
				.interruptable()
				.pull(|_| async { std::future::pending().await })
				.await;
			match interrupted {
				Err(ParamError::Interrupted(interrupt)) => {
					yield Ev::Aborted(Abort::Interrupted { reason: interrupt.reason });
				},
				_ => yield Ev::Aborted(Abort::MissingOutcome),
			}
		}
	}

	fn prompt(&self, _view: Result<&Value, &Value>, _caps: &PromptCaps) -> Vec<Part> {
		Vec::new()
	}
}

const WORKER_CANCEL_EXTENSION: &str = r#"
import ctypes
import os
import signal

signal.signal(signal.SIGINT, signal.SIG_IGN)
_sleep = ctypes.CDLL(None).sleep
_sleep.argtypes = [ctypes.c_uint]
_sleep.restype = ctypes.c_uint

def block(params):
    with open(params["started"], "w", encoding="utf-8") as marker:
        marker.write(str(os.getpid()))
        marker.flush()
    _sleep(params["seconds"])
    return {"parts": [], "details": {"unexpected": "completed"}}

def echo(params):
    return {"parts": [], "details": {"message": params["message"]}}

def fail(params):
    return {"parts": [], "details": {"code": params["code"]}, "is_error": True}

OMP_TOOLS = [
    {
        "name": "worker_block",
        "description": "blocks in native code until killed",
        "schema": {
            "type": "object",
            "properties": {
                "started": {"type": "string"},
                "seconds": {"type": "integer"},
            },
            "required": ["started", "seconds"],
            "additionalProperties": False,
        },
        "rev": "r.1",
        "strict": True,
        "handler": block,
    },
    {
        "name": "worker_echo",
        "description": "serves the request after cancellation respawn",
        "schema": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
            "additionalProperties": False,
        },
        "rev": "r.1",
        "strict": True,
        "handler": echo,
    },
    {
        "name": "worker_fail",
        "description": "returns a structured tool fault",
        "schema": {
            "type": "object",
            "properties": {"code": {"type": "integer"}},
            "required": ["code"],
            "additionalProperties": False,
        },
        "rev": "r.1",
        "strict": True,
        "handler": fail,
    },
]
"#;

struct Harness {
	_client:     EnvClient,
	server:      Arc<EnvServer>,
	root:        TempDir,
	_state:      TempDir,
	server_task: tokio::task::JoinHandle<()>,
}

impl Harness {
	async fn start(registry: Registry) -> Self {
		Self::start_with_worker(
			registry,
			ToolWorkerConfig::new(PathBuf::from(env!("CARGO_BIN_EXE_omp"))),
		)
		.await
	}

	async fn start_with_worker(registry: Registry, worker: ToolWorkerConfig) -> Self {
		let root = tempfile::tempdir().expect("workspace scratch directory");
		let state = tempfile::tempdir().expect("state scratch directory");
		let server = Arc::new(
			EnvServer::open_local(root.path(), state.path(), registry, worker)
				.await
				.expect("real local environment host"),
		);
		let (client, transport) = EnvClient::in_process(64);
		let host = Arc::clone(&server);
		let server_task = tokio::spawn(async move { host.serve_in_process(transport).await });
		client
			.hello(ClientHello {
				client: "envd-contract".into(),
				schema_rev: SCHEMA_REV,
				..ClientHello::default()
			})
			.await
			.expect("environment hello");
		Self { _client: client, server, root, _state: state, server_task }
	}

	fn client(&self) -> &EnvClient {
		&self._client
	}

	async fn connect(&self, name: &str) -> (EnvClient, tokio::task::JoinHandle<()>) {
		let (client, transport) = EnvClient::in_process(64);
		let host = Arc::clone(&self.server);
		let task = tokio::spawn(async move { host.serve_in_process(transport).await });
		client
			.hello(ClientHello {
				client: name.to_owned(),
				schema_rev: SCHEMA_REV,
				..ClientHello::default()
			})
			.await
			.expect("additional environment hello");
		(client, task)
	}
}

impl Drop for Harness {
	fn drop(&mut self) {
		self.server_task.abort();
	}
}

fn cwd_uri(path: &Path) -> String {
	url::Url::from_directory_path(path)
		.expect("directory file URI")
		.to_string()
}

fn exec_request(session: &[u8], script: impl Into<String>) -> ExecRequest {
	ExecRequest {
		session: Bytes::copy_from_slice(session),
		source: Some(Script { text: script.into(), ..Script::default() }),
		..ExecRequest::default()
	}
}

async fn collect_exec(run: &mut omp_env::ExecRun) -> (Vec<u8>, omp_proto::env::v1::ExecStatusMsg) {
	let mut output = Vec::new();
	loop {
		match tokio::time::timeout(Duration::from_secs(10), run.next_event())
			.await
			.expect("exec event timeout")
			.expect("exec event")
			.expect("exec stream closed")
		{
			ExecEvent::Started(_) => {},
			ExecEvent::Output(frame) => output.extend_from_slice(&frame.data),
			ExecEvent::Exit(exit) => return (output, exit.status.expect("terminal status")),
			ExecEvent::StreamError(error) => panic!("exec stream error: {}", error.message),
		}
	}
}

async fn invoke_builtin(
	client: &EnvClient,
	invocation_id: &str,
	name: &str,
	rev: &str,
	args: Value,
) -> omp_proto::env::v1::Verdict {
	let mut invocation = client
		.invoke(InvokeTool {
			invocation_id: invocation_id.into(),
			name: name.into(),
			rev: rev.into(),
			..InvokeTool::default()
		})
		.await
		.expect("open built-in invocation");
	assert!(matches!(
		invocation.next_event().await.expect("built-in accepted"),
		Some(InvocationEvent::Accepted(_))
	));
	invocation
		.commit_args(Bytes::from(serde_json::to_vec(&args).expect("encode built-in args")))
		.await
		.expect("commit built-in arguments");
	loop {
		match invocation
			.next_event()
			.await
			.expect("built-in event")
			.expect("built-in stream closed")
		{
			InvocationEvent::Verdict(verdict) => return verdict,
			InvocationEvent::Update(_) => {},
			InvocationEvent::Accepted(_) => panic!("built-in invocation was accepted twice"),
			InvocationEvent::StreamError(error) => panic!("built-in stream failed: {}", error.message),
		}
	}
}

#[tokio::test]
async fn production_registry_advertises_and_dispatches_all_native_adapters() {
	let harness = Harness::start(Registry::new()).await;
	std::fs::write(harness.root.path().join("note.txt"), "before\n").expect("workspace fixture");
	let registry = harness.server.registry();
	let agent_registry = harness.server.registry();
	assert!(Arc::ptr_eq(&registry, &agent_registry));
	assert_eq!(registry.live_hash(), agent_registry.live_hash());
	let advertised = registry.advertise(LoweringCaps {
		strict_schema: true,
		grammar: omp_llm_catalog::GrammarBits::empty(),
	});
	let identities = advertised
		.iter()
		.map(|tool| (tool.identity.name.as_str(), tool.identity.rev.to_string()))
		.collect::<Vec<_>>();
	assert_eq!(
		identities,
		[
			("edit", "hl.1".to_owned()),
			("glob", "1".to_owned()),
			("grep", "1".to_owned()),
			("read", "1".to_owned()),
			("shell", "1".to_owned()),
		]
	);

	let read = invoke_builtin(
		harness.client(),
		"builtin-read",
		"read",
		"1",
		json!({"path":"note.txt"}),
	)
	.await;
	assert!(!read.is_error, "read adapter returned an error");
	let read_verdict: Verdict<Value, Value> =
		serde_json::from_slice(&read.json).expect("typed read verdict");
	let Verdict::Ok(read_payload) = read_verdict else {
		panic!("read did not return an ok payload");
	};
	assert!(!read_payload["revision"].as_str().expect("read revision").is_empty());
	let patch = "PUT 1.=1:\n+after\n";
	let edit = invoke_builtin(
		harness.client(),
		"builtin-edit",
		"edit",
		"hl.1",
		json!({"path":"note.txt","patch":patch}),
	)
	.await;
	assert!(
		!edit.is_error,
		"edit adapter returned an error: {}",
		String::from_utf8_lossy(&edit.json)
	);
	assert_eq!(
		std::fs::read_to_string(harness.root.path().join("note.txt")).expect("edited fixture"),
		"after\n"
	);

	let shell = invoke_builtin(
		harness.client(),
		"builtin-shell",
		"shell",
		"1",
		json!({"command":"printf shell-ok"}),
	)
	.await;
	assert!(!shell.is_error, "shell adapter returned an error");
	let grep = invoke_builtin(
		harness.client(),
		"builtin-grep",
		"grep",
		"1",
		json!({"patterns":["after"],"limit":10}),
	)
	.await;
	assert!(!grep.is_error, "grep adapter returned an error");
	let glob = invoke_builtin(
		harness.client(),
		"builtin-glob",
		"glob",
		"1",
		json!({"patterns":["*.txt"],"limit":10}),
	)
	.await;
	assert!(!glob.is_error, "glob adapter returned an error");
}


#[tokio::test]
async fn opt_in_python_adds_one_worker_route_and_default_adds_none() {
	let mut worker = ToolWorkerConfig::new(PathBuf::from(env!("CARGO_BIN_EXE_omp")));
	worker.modules.push(Str::new_static(PY_EVAL_MODULE));
	let harness = Harness::start_with_worker(Registry::new(), worker).await;
	let registry = harness.server.registry();
	let advertised = registry.advertise(LoweringCaps {
		strict_schema: true,
		grammar: omp_llm_catalog::GrammarBits::empty(),
	});
	assert_eq!(advertised.len(), 6);
	assert_eq!(registry.route("py_eval").expect("python route"), ToolRoute::Worker);
	assert_eq!(
		registry
			.live_identity("py_eval")
			.map(|(_, revision)| revision.to_string())
			.as_deref(),
		Some("1")
	);
	let verdict = invoke_builtin(
		harness.client(),
		"builtin-python",
		"py_eval",
		"1",
		json!({"code":"40 + 2"}),
	)
	.await;
	assert!(!verdict.is_error, "python worker route returned an error");
}
#[tokio::test]
async fn native_streaming_prepares_before_commit_and_fuses_commit_cancel_terminals() {
	let scratch = tempfile::tempdir().expect("streaming native scratch");
	let lease = scratch.path().join("lease");
	let effect = scratch.path().join("effect");
	let mut registry = Registry::new();
	registry
		.register(StreamingTool::new(lease.clone(), effect.clone()))
		.expect("register streaming tool");
	let harness = Harness::start(registry).await;

	let mut cancelled = harness
		.client()
		.invoke(omp_proto::env::v1::InvokeTool {
			invocation_id: "stream-cancel".into(),
			name: "streaming_probe".into(),
			rev: "test.1".into(),
			..Default::default()
		})
		.await
		.expect("open cancellable streaming invocation");
	assert!(matches!(
		cancelled.next_event().await.expect("cancel accepted"),
		Some(InvocationEvent::Accepted(_))
	));
	cancelled
		.arg_text(Str::new_static(r#"{"pa"#))
		.await
		.expect("first cancellable argument fragment");
	cancelled
		.arg_text(Str::new_static(r#"th":"cancel"}"#))
		.await
		.expect("second cancellable argument fragment");
	let update = tokio::time::timeout(Duration::from_secs(1), cancelled.next_event())
		.await
		.expect("speculative update timeout")
		.expect("speculative update event")
		.expect("speculative stream closed");
	assert!(matches!(update, InvocationEvent::Update(_)));
	assert_eq!(std::fs::read(&lease).expect("speculative lease marker"), b"cancel");
	assert!(!effect.exists(), "streamed preparation performed an effect before commit");

	cancelled.guard().cancel();
	let terminal = cancelled
		.next_event()
		.await
		.expect("cancel terminal event")
		.expect("cancel stream closed");
	let InvocationEvent::Verdict(terminal) = terminal else {
		panic!("precommit cancel did not produce a verdict");
	};
	let verdict: Verdict<Value, Value> =
		serde_json::from_slice(&terminal.json).expect("decode precommit cancel verdict");
	assert!(matches!(&verdict, Verdict::Aborted(Abort::Skipped { .. })));
	assert!(!matches!(&verdict, Verdict::Aborted(Abort::EffectsUnknown { .. })));
	assert!(
		cancelled
			.next_event()
			.await
			.expect("closed cancelled invocation")
			.is_none(),
		"precommit cancellation emitted more than one terminal",
	);
	tokio::time::timeout(Duration::from_secs(1), async {
		while lease.exists() {
			tokio::task::yield_now().await;
		}
	})
	.await
	.expect("speculative lease was not released");
	assert!(!effect.exists(), "cancelled precommit invocation performed an effect");

	let mut committed = harness
		.client()
		.invoke(omp_proto::env::v1::InvokeTool {
			invocation_id: "stream-commit".into(),
			name: "streaming_probe".into(),
			rev: "test.1".into(),
			..Default::default()
		})
		.await
		.expect("open committed streaming invocation");
	assert!(matches!(
		committed.next_event().await.expect("commit accepted"),
		Some(InvocationEvent::Accepted(_))
	));
	committed
		.arg_text(Str::new_static(r#"{"path":"comm"#))
		.await
		.expect("first committed argument fragment");
	committed
		.arg_text(Str::new_static(r#"itted"}"#))
		.await
		.expect("second committed argument fragment");
	assert!(matches!(
		committed.next_event().await.expect("committed speculative update"),
		Some(InvocationEvent::Update(_))
	));
	assert!(!effect.exists(), "effect marker appeared before ArgsCommitted");
	committed
		.commit_args(Bytes::from_static(br#"{"path":"committed"}"#))
		.await
		.expect("commit streamed arguments");
	let terminal = committed
		.next_event()
		.await
		.expect("committed verdict")
		.expect("committed stream closed");
	assert!(matches!(terminal, InvocationEvent::Verdict(_)));
	assert_eq!(std::fs::read(&effect).expect("committed effect marker"), b"committed");
	assert!(!lease.exists(), "committed speculative lease was not released");

	let mut duplicate = harness
		.client()
		.invoke(omp_proto::env::v1::InvokeTool {
			invocation_id: "stream-duplicate".into(),
			name: "streaming_probe".into(),
			rev: "test.1".into(),
			..Default::default()
		})
		.await
		.expect("open duplicate-commit invocation");
	assert!(matches!(
		duplicate.next_event().await.expect("duplicate accepted"),
		Some(InvocationEvent::Accepted(_))
	));
	duplicate
		.arg_text(Str::new_static(r#"{"path":"duplicate"}"#))
		.await
		.expect("duplicate argument fragment");
	assert!(matches!(
		duplicate.next_event().await.expect("duplicate speculative update"),
		Some(InvocationEvent::Update(_))
	));
	duplicate
		.commit_args(Bytes::from_static(br#"{"path":"duplicate"}"#))
		.await
		.expect("first duplicate commit");
	duplicate
		.commit_args(Bytes::from_static(br#"{"path":"duplicate"}"#))
		.await
		.expect("send duplicate commit");
	let error = duplicate
		.next_event()
		.await
		.expect_err("duplicate ArgsCommitted was not rejected");
	let omp_env::ClientError::Protocol(error) = error else {
		panic!("duplicate ArgsCommitted returned a non-protocol error");
	};
	assert_eq!(
		error.code,
		omp_proto::env::v1::ProtocolErrorCode::AlreadyExists as i32
	);
	tokio::time::sleep(Duration::from_millis(200)).await;
	assert_eq!(std::fs::read(&effect).expect("duplicate committed effect"), b"duplicate");
	assert!(!lease.exists(), "duplicate-commit request leaked its speculative lease");
	let mut reopened = harness
		.client()
		.invoke(omp_proto::env::v1::InvokeTool {
			invocation_id: "stream-duplicate".into(),
			name: "streaming_probe".into(),
			rev: "test.1".into(),
			..Default::default()
		})
		.await
		.expect("reopen cleaned duplicate invocation");
	assert!(matches!(
		reopened.next_event().await.expect("reopened accepted"),
		Some(InvocationEvent::Accepted(_))
	));
	reopened.guard().cancel();
}

#[tokio::test]
async fn native_cancel_emits_one_bounded_effects_unknown_verdict_and_next_request_succeeds() {
	let scratch = tempfile::tempdir().expect("native cancellation scratch");
	let started = scratch.path().join("started");
	let completed = scratch.path().join("completed");
	let mut registry = Registry::new();
	registry
		.register(BlockingTool::new(started.clone()))
		.expect("register blocking native tool");
	registry
		.register(EffectTool::new(completed.clone()))
		.expect("register follow-up native tool");
	let harness = Harness::start(registry).await;

	let mut blocked = harness
		.client()
		.invoke(omp_proto::env::v1::InvokeTool {
			invocation_id: "native-cancel".into(),
			name: "native_block".into(),
			rev: "test.1".into(),
			..Default::default()
		})
		.await
		.expect("open blocking native invocation");
	assert!(matches!(
		blocked.next_event().await.expect("native accepted"),
		Some(InvocationEvent::Accepted(_))
	));
	blocked
		.commit_args(Bytes::from_static(b"{}"))
		.await
		.expect("commit blocking native invocation");
	assert!(matches!(
		blocked.next_event().await.expect("native started update"),
		Some(InvocationEvent::Update(_))
	));
	assert!(started.exists(), "native invocation did not enter its committed body");

	blocked.guard().cancel();
	let terminal = tokio::time::timeout(Duration::from_secs(2), blocked.next_event())
		.await
		.expect("native structural cancellation exceeded its bound")
		.expect("native cancellation event")
		.expect("native cancellation stream closed");
	let InvocationEvent::Verdict(terminal) = terminal else {
		panic!("native cancellation did not produce a verdict");
	};
	let verdict: Verdict<Value, Value> =
		serde_json::from_slice(&terminal.json).expect("decode native cancellation verdict");
	assert!(matches!(verdict, Verdict::Aborted(Abort::EffectsUnknown { .. })));
	assert!(terminal.is_error);
	assert!(!terminal.useless);
	assert!(
		blocked
			.next_event()
			.await
			.expect("closed native cancellation stream")
			.is_none(),
		"native invocation leaked an update or terminal after its verdict",
	);

	let mut next = harness
		.client()
		.invoke(omp_proto::env::v1::InvokeTool {
			invocation_id: "native-next".into(),
			name: "effect_probe".into(),
			rev: "test.1".into(),
			..Default::default()
		})
		.await
		.expect("open native request after cancellation");
	assert!(matches!(
		next.next_event().await.expect("next native accepted"),
		Some(InvocationEvent::Accepted(_))
	));
	next
		.commit_args(Bytes::from_static(b"{}"))
		.await
		.expect("commit next native request");
	assert!(matches!(
		next.next_event().await.expect("next native verdict"),
		Some(InvocationEvent::Verdict(_))
	));
	assert_eq!(std::fs::read(completed).expect("follow-up native effect"), b"committed");
}

#[tokio::test]
async fn native_interrupt_is_steering_only_and_preserves_cooperative_truth() {
	let mut registry = Registry::new();
	registry
		.register(CooperativeInterruptTool::new())
		.expect("register cooperative interrupt tool");
	let harness = Harness::start(registry).await;
	let mut invocation = harness
		.client()
		.invoke(omp_proto::env::v1::InvokeTool {
			invocation_id: "native-interrupt".into(),
			name: "cooperative_interrupt".into(),
			rev: "test.1".into(),
			..Default::default()
		})
		.await
		.expect("open cooperative invocation");
	assert!(matches!(
		invocation.next_event().await.expect("cooperative accepted"),
		Some(InvocationEvent::Accepted(_))
	));
	invocation
		.commit_args(Bytes::from_static(b"{}"))
		.await
		.expect("commit cooperative invocation");
	assert!(matches!(
		invocation
			.next_event()
			.await
			.expect("cooperative waiting update"),
		Some(InvocationEvent::Update(_))
	));
	invocation
		.interrupt(Str::new_static("steer cooperatively"))
		.await
		.expect("send cooperative interrupt");
	let terminal = tokio::time::timeout(Duration::from_secs(1), invocation.next_event())
		.await
		.expect("cooperative interrupt terminal timeout")
		.expect("cooperative interrupt event")
		.expect("cooperative interrupt stream closed");
	let InvocationEvent::Verdict(terminal) = terminal else {
		panic!("cooperative interrupt did not produce a verdict");
	};
	let verdict: Verdict<Value, Value> =
		serde_json::from_slice(&terminal.json).expect("decode cooperative interrupt verdict");
	assert!(matches!(
		verdict,
		Verdict::Aborted(Abort::Interrupted { reason })
			if reason == "steer cooperatively"
	));
}

#[tokio::test]
async fn native_deadline_interrupts_then_structurally_reports_effects_unknown() {
	let scratch = tempfile::tempdir().expect("native deadline scratch");
	let started = scratch.path().join("deadline-started");
	let mut registry = Registry::new();
	registry
		.register(BlockingTool::new(started.clone()))
		.expect("register deadline native tool");
	let harness = Harness::start(registry).await;
	let mut invocation = harness
		.client()
		.invoke(omp_proto::env::v1::InvokeTool {
			invocation_id: "native-deadline".into(),
			name: "native_block".into(),
			rev: "test.1".into(),
			deadline_ms: 50,
			..Default::default()
		})
		.await
		.expect("open deadline native invocation");
	assert!(matches!(
		invocation
			.next_event()
			.await
			.expect("deadline native accepted"),
		Some(InvocationEvent::Accepted(_))
	));
	invocation
		.commit_args(Bytes::from_static(b"{}"))
		.await
		.expect("commit deadline native invocation");
	assert!(matches!(
		invocation
			.next_event()
			.await
			.expect("deadline native update"),
		Some(InvocationEvent::Update(_))
	));
	let terminal = tokio::time::timeout(Duration::from_secs(2), invocation.next_event())
		.await
		.expect("native deadline plus grace exceeded bound")
		.expect("native deadline event")
		.expect("native deadline stream closed");
	let InvocationEvent::Verdict(terminal) = terminal else {
		panic!("native deadline did not produce a verdict");
	};
	let verdict: Verdict<Value, Value> =
		serde_json::from_slice(&terminal.json).expect("decode native deadline verdict");
	assert!(matches!(verdict, Verdict::Aborted(Abort::EffectsUnknown { .. })));
	assert!(started.exists(), "native deadline fired before committed execution began");
}

#[tokio::test]
async fn worker_cancel_forwards_effects_unknown_once_and_respawn_serves_next_request() {
	let site = tempfile::tempdir().expect("worker extension scratch");
	std::fs::write(site.path().join("envd_cancel_tools.py"), WORKER_CANCEL_EXTENSION)
		.expect("write worker cancellation extension");
	let mut worker = ToolWorkerConfig::new(PathBuf::from(env!("CARGO_BIN_EXE_omp")));
	worker.python_site = Some(site.path().to_owned());
	worker.modules = vec![Str::new_static("envd_cancel_tools")];
	worker.health_timeout = Duration::from_secs(5);
	worker.interrupt_grace = Duration::from_millis(150);
	worker.initial_backoff = Duration::from_millis(10);
	worker.max_backoff = Duration::from_millis(50);
	let harness = Harness::start_with_worker(Registry::new(), worker).await;
	let started = site.path().join("worker-started");

	let mut blocked = harness
		.client()
		.invoke(omp_proto::env::v1::InvokeTool {
			invocation_id: "worker-cancel".into(),
			name: "worker_block".into(),
			rev: "r.1".into(),
			..Default::default()
		})
		.await
		.expect("open blocking worker invocation");
	assert!(matches!(
		blocked.next_event().await.expect("worker accepted"),
		Some(InvocationEvent::Accepted(_))
	));
	blocked
		.commit_args(Bytes::from(
			serde_json::to_vec(&json!({
				"started": started.to_string_lossy(),
				"seconds": 30,
			}))
			.expect("serialize worker arguments"),
		))
		.await
		.expect("commit blocking worker invocation");
	tokio::time::timeout(Duration::from_secs(3), async {
		while !started.exists() {
			tokio::time::sleep(Duration::from_millis(10)).await;
		}
	})
	.await
	.expect("worker invocation did not enter native sleep");

	blocked.guard().cancel();
	let terminal = tokio::time::timeout(Duration::from_secs(3), blocked.next_event())
		.await
		.expect("worker cancellation terminal timeout")
		.expect("worker cancellation event")
		.expect("worker cancellation stream closed");
	let InvocationEvent::Verdict(terminal) = terminal else {
		panic!("worker cancellation did not produce a verdict");
	};
	let verdict: Verdict<Value, Value> =
		serde_json::from_slice(&terminal.json).expect("decode worker cancellation verdict");
	assert!(matches!(verdict, Verdict::Aborted(Abort::EffectsUnknown { .. })));
	assert!(terminal.is_error);
	assert!(!terminal.useless);
	assert!(
		blocked
			.next_event()
			.await
			.expect("closed worker cancellation stream")
			.is_none(),
		"worker invocation leaked an update or terminal after its verdict",
	);

	let mut next = harness
		.client()
		.invoke(omp_proto::env::v1::InvokeTool {
			invocation_id: "worker-next".into(),
			name: "worker_echo".into(),
			rev: "r.1".into(),
			..Default::default()
		})
		.await
		.expect("open worker request after cancellation");
	assert!(matches!(
		next.next_event().await.expect("next worker accepted"),
		Some(InvocationEvent::Accepted(_))
	));
	next
		.commit_args(Bytes::from_static(br#"{"message":"after cancellation"}"#))
		.await
		.expect("commit next worker request");
	let next_terminal = tokio::time::timeout(Duration::from_secs(5), async {
		loop {
			match next
				.next_event()
				.await
				.expect("next worker event")
				.expect("next worker stream closed")
			{
				InvocationEvent::Verdict(verdict) => break verdict,
				InvocationEvent::Update(_) => {},
				InvocationEvent::Accepted(_) => panic!("worker request was accepted twice"),
				InvocationEvent::StreamError(error) => {
					panic!("next worker stream failed: {}", error.message)
				},
			}
		}
	})
	.await
	.expect("respawned worker did not serve next request");
	assert_eq!(next_terminal.invocation_id, "worker-next");
	assert!(!next_terminal.is_error);
	assert!(!next_terminal.useless);
	assert_eq!(
		next_terminal.json,
		Bytes::from_static(br#"{"kind":"ok","value":{"message":"after cancellation"}}"#,),
	);
	let verdict: Verdict<Value, Value> =
		serde_json::from_slice(&next_terminal.json).expect("decode worker success verdict");
	assert_eq!(verdict, Verdict::Ok(json!({"message": "after cancellation"})));

	let mut fault = harness
		.client()
		.invoke(omp_proto::env::v1::InvokeTool {
			invocation_id: "worker-fault".into(),
			name: "worker_fail".into(),
			rev: "r.1".into(),
			..Default::default()
		})
		.await
		.expect("open worker fault request");
	assert!(matches!(
		fault.next_event().await.expect("worker fault accepted"),
		Some(InvocationEvent::Accepted(_))
	));
	fault
		.commit_args(Bytes::from_static(br#"{"code":409}"#))
		.await
		.expect("commit worker fault request");
	let fault_terminal = tokio::time::timeout(Duration::from_secs(5), async {
		loop {
			match fault
				.next_event()
				.await
				.expect("worker fault event")
				.expect("worker fault stream closed")
			{
				InvocationEvent::Verdict(verdict) => break verdict,
				InvocationEvent::Update(_) => {},
				InvocationEvent::Accepted(_) => panic!("worker fault was accepted twice"),
				InvocationEvent::StreamError(error) => {
					panic!("worker fault stream failed: {}", error.message)
				},
			}
		}
	})
	.await
	.expect("worker did not return its structured fault");
	assert!(fault_terminal.is_error);
	assert!(!fault_terminal.useless);
	assert_eq!(fault_terminal.json, Bytes::from_static(br#"{"kind":"fault","value":{"code":409}}"#),);
	let verdict: Verdict<Value, Value> =
		serde_json::from_slice(&fault_terminal.json).expect("decode worker fault verdict");
	assert_eq!(verdict, Verdict::Fault(json!({"code": 409})));
}

#[tokio::test]
async fn same_worker_invocation_id_on_two_connections_cancels_only_its_owner() {
	let site = tempfile::tempdir().expect("worker collision scratch");
	std::fs::write(site.path().join("envd_cancel_tools.py"), WORKER_CANCEL_EXTENSION)
		.expect("write worker collision extension");
	let mut worker = ToolWorkerConfig::new(PathBuf::from(env!("CARGO_BIN_EXE_omp")));
	worker.python_site = Some(site.path().to_owned());
	worker.modules = vec![Str::new_static("envd_cancel_tools")];
	worker.health_timeout = Duration::from_secs(5);
	worker.interrupt_grace = Duration::from_millis(100);
	worker.initial_backoff = Duration::from_millis(10);
	worker.max_backoff = Duration::from_millis(50);
	let harness = Harness::start_with_worker(Registry::new(), worker).await;
	let (client_b, client_b_task) = harness.connect("envd-contract-b").await;
	let started_a = site.path().join("worker-a-started");
	let started_b = site.path().join("worker-b-started");

	let mut invocation_a = harness
		.client()
		.invoke(omp_proto::env::v1::InvokeTool {
			invocation_id: "shared-id".into(),
			name: "worker_block".into(),
			rev: "r.1".into(),
			..Default::default()
		})
		.await
		.expect("open worker A");
	assert!(matches!(
		invocation_a.next_event().await.expect("worker A accepted"),
		Some(InvocationEvent::Accepted(_))
	));
	invocation_a
		.commit_args(Bytes::from(
			serde_json::to_vec(&json!({
				"started": started_a.to_string_lossy(),
				"seconds": 30,
			}))
			.expect("serialize worker A arguments"),
		))
		.await
		.expect("commit worker A");
	tokio::time::timeout(Duration::from_secs(3), async {
		while !started_a.exists() {
			tokio::time::sleep(Duration::from_millis(10)).await;
		}
	})
	.await
	.expect("worker A did not start");

	let mut invocation_b = client_b
		.invoke(omp_proto::env::v1::InvokeTool {
			invocation_id: "shared-id".into(),
			name: "worker_block".into(),
			rev: "r.1".into(),
			..Default::default()
		})
		.await
		.expect("open worker B with colliding external id");
	assert!(matches!(
		invocation_b.next_event().await.expect("worker B accepted"),
		Some(InvocationEvent::Accepted(_))
	));
	invocation_b
		.commit_args(Bytes::from(
			serde_json::to_vec(&json!({
				"started": started_b.to_string_lossy(),
				"seconds": 30,
			}))
			.expect("serialize worker B arguments"),
		))
		.await
		.expect("commit worker B");
	invocation_b.guard().cancel();

	let terminal_b = tokio::time::timeout(Duration::from_secs(2), invocation_b.next_event())
		.await
		.expect("worker B queued cancellation timeout")
		.expect("worker B cancellation event")
		.expect("worker B cancellation stream closed");
	let InvocationEvent::Verdict(terminal_b) = terminal_b else {
		panic!("worker B cancellation did not produce a verdict");
	};
	let verdict_b: Verdict<Value, Value> =
		serde_json::from_slice(&terminal_b.json).expect("decode worker B cancellation");
	assert!(matches!(verdict_b, Verdict::Aborted(Abort::Skipped { .. })));
	assert!(!started_b.exists(), "cancelled worker B was dispatched");
	assert!(
		tokio::time::timeout(Duration::from_millis(100), invocation_a.next_event())
			.await
			.is_err(),
		"worker B cancellation terminated worker A",
	);

	invocation_a.guard().cancel();
	let terminal_a = tokio::time::timeout(Duration::from_secs(3), invocation_a.next_event())
		.await
		.expect("worker A cancellation timeout")
		.expect("worker A cancellation event")
		.expect("worker A cancellation stream closed");
	let InvocationEvent::Verdict(terminal_a) = terminal_a else {
		panic!("worker A cancellation did not produce a verdict");
	};
	let verdict_a: Verdict<Value, Value> =
		serde_json::from_slice(&terminal_a.json).expect("decode worker A cancellation");
	assert!(matches!(verdict_a, Verdict::Aborted(Abort::EffectsUnknown { .. })));

	let mut next = client_b
		.invoke(omp_proto::env::v1::InvokeTool {
			invocation_id: "shared-id".into(),
			name: "worker_echo".into(),
			rev: "r.1".into(),
			..Default::default()
		})
		.await
		.expect("reuse external id after worker B terminal");
	assert!(matches!(
		next.next_event().await.expect("follow-up worker accepted"),
		Some(InvocationEvent::Accepted(_))
	));
	next
		.commit_args(Bytes::from_static(br#"{"message":"still isolated"}"#))
		.await
		.expect("commit follow-up worker");
	assert!(matches!(
		tokio::time::timeout(Duration::from_secs(5), next.next_event())
			.await
			.expect("follow-up worker timeout")
			.expect("follow-up worker event"),
		Some(InvocationEvent::Verdict(_))
	));
	client_b_task.abort();
}

#[tokio::test]
async fn cancelled_exec_preserves_session_cwd_and_kills_term_ignoring_tree() {
	let harness = Harness::start(Registry::new()).await;
	let client = harness.client();
	let opened = client
		.open_session(OpenSessionRequest {
			cwd_uri: cwd_uri(harness.root.path()),
			..Default::default()
		})
		.await
		.expect("open session");
	let child_pid = harness.root.path().join("child.pid");
	let grandchild_pid = harness.root.path().join("grandchild.pid");
	let script = format!(
		"cd sub 2>/dev/null || mkdir sub && cd sub; sh -c 'trap \"\" TERM; (trap \"\" TERM; sleep \
		 30) & echo $! > {}; echo $$ > {}; wait'",
		grandchild_pid.display(),
		child_pid.display()
	);
	let mut run = client
		.exec(exec_request(&opened.session, script))
		.await
		.expect("start cancellable run");
	assert!(matches!(run.next_event().await.expect("started"), Some(ExecEvent::Started(_))));
	for _ in 0..100 {
		if child_pid.exists() && grandchild_pid.exists() {
			break;
		}
		tokio::time::sleep(Duration::from_millis(10)).await;
	}
	assert!(child_pid.exists() && grandchild_pid.exists(), "child tree did not start");
	drop(run);
	for pid_file in [&child_pid, &grandchild_pid] {
		let pid: i32 = std::fs::read_to_string(pid_file)
			.expect("pid file")
			.trim()
			.parse()
			.expect("pid");
		let mut dead = false;
		for _ in 0..100 {
			if unsafe { libc::kill(pid, 0) } == -1 {
				dead = true;
				break;
			}
			tokio::time::sleep(Duration::from_millis(25)).await;
		}
		assert!(dead, "cancelled process {pid} is still alive");
	}
	let mut pwd = client
		.exec(exec_request(&opened.session, "pwd"))
		.await
		.expect("session survived");
	let (output, status) = collect_exec(&mut pwd).await;
	assert_eq!(status.outcome, ExecOutcome::Exited as i32);
	assert!(
		String::from_utf8_lossy(&output).contains("/sub"),
		"cwd did not persist: {}",
		String::from_utf8_lossy(&output)
	);
}

#[tokio::test]
async fn blob_and_named_process_frames_route_through_one_host() {
	let harness = Harness::start(Registry::new()).await;
	let client = harness.client();
	let payload = Bytes::from_static(b"host-routed-blob");
	let upload = client.blob_put().await.expect("begin blob upload");
	upload
		.send_chunk(Chunk { data: payload.clone(), ..Default::default() })
		.await
		.expect("blob chunk");
	let stored = upload.commit().await.expect("commit blob");
	let mut download = client
		.blob_get(GetRequest { hash: stored.hash.clone(), ..Default::default() })
		.await
		.expect("get blob");
	let mut received = Vec::new();
	loop {
		match download
			.next_event()
			.await
			.expect("blob event")
			.expect("blob event present")
		{
			BlobDownloadEvent::Chunk(chunk) => received.extend_from_slice(&chunk.data),
			BlobDownloadEvent::Complete(_) => break,
		}
	}
	assert_eq!(received, payload);

	client
		.start_process(StartProcess {
			name: "contract-process".into(),
			spec: Some(ProcessSpec {
				source: Some(Script { text: "echo ready; sleep 30".into(), ..Default::default() }),
				cwd_uri: cwd_uri(harness.root.path()),
				..Default::default()
			}),
			..Default::default()
		})
		.await
		.expect("start named process");
	let listed = client
		.list_processes(ListProcesses::default())
		.await
		.expect("list processes");
	assert_eq!(
		listed
			.processes
			.iter()
			.map(|p| p.name.as_str())
			.collect::<Vec<_>>(),
		["contract-process"]
	);
	let mut attachment = client
		.attach_output(omp_proto::env::v1::AttachOutput {
			name: "contract-process".into(),
			..Default::default()
		})
		.await
		.expect("attach output");
	assert!(matches!(
		attachment.next_event().await.expect("attached"),
		Some(ProcessAttachmentEvent::Attached(_))
	));
	client
		.stop_process(StopProcess {
			name: "contract-process".into(),
			grace_ms: 50,
			..Default::default()
		})
		.await
		.expect("stop process");
	loop {
		let event = tokio::time::timeout(Duration::from_secs(10), attachment.next_event())
			.await
			.expect("named process stop timeout")
			.expect("process state");
		if let Some(ProcessAttachmentEvent::State(state)) = event
			&& state
				.process
				.as_ref()
				.and_then(|p| p.status.as_ref())
				.is_some()
		{
			break;
		}
	}
	let mut exited_attachment = client
		.attach_output(omp_proto::env::v1::AttachOutput {
			name: "contract-process".into(),
			..Default::default()
		})
		.await
		.expect("attach already-terminal process");
	assert!(matches!(
		exited_attachment.next_event().await.expect("attached"),
		Some(ProcessAttachmentEvent::Attached(_))
	));
	loop {
		let event = tokio::time::timeout(Duration::from_secs(2), exited_attachment.next_event())
			.await
			.expect("already-terminal attachment state timeout")
			.expect("already-terminal process state");
		if let Some(ProcessAttachmentEvent::State(state)) = event
			&& state
				.process
				.as_ref()
				.and_then(|process| process.status.as_ref())
				.is_some()
		{
			break;
		}
	}
}

#[tokio::test]
async fn named_process_attach_has_no_gap_between_backlog_and_future_output() {
	let harness = Harness::start(Registry::new()).await;
	let client = harness.client();
	client
		.start_process(StartProcess {
			name: "attach-race".into(),
			spec: Some(ProcessSpec {
				source: Some(Script {
					text: "i=0; while [ $i -lt 50 ]; do echo output; sleep 0.01; i=$((i + 1)); done".into(),
					..Default::default()
				}),
				cwd_uri: cwd_uri(harness.root.path()),
				..Default::default()
			}),
			..Default::default()
		})
		.await
		.expect("start racing named process");
	let mut attachment = client
		.attach_output(omp_proto::env::v1::AttachOutput {
			name: "attach-race".into(),
			..Default::default()
		})
		.await
		.expect("attach while output is active");
	assert!(matches!(
		attachment.next_event().await.expect("attached"),
		Some(ProcessAttachmentEvent::Attached(_))
	));

	let mut sequences = Vec::new();
	loop {
		let event = tokio::time::timeout(Duration::from_secs(10), attachment.next_event())
			.await
			.expect("attach race timeout")
			.expect("attachment event")
			.expect("attachment remains open");
		match event {
			ProcessAttachmentEvent::Output(output) => sequences.push(output.sequence),
			ProcessAttachmentEvent::State(state)
				if state
					.process
					.as_ref()
					.and_then(|process| process.status.as_ref())
					.is_some() =>
			{
				break;
			},
			_ => {},
		}
	}
	assert!(!sequences.is_empty());
	assert_eq!(sequences[0], 1);
	assert!(
		sequences.windows(2).all(|pair| pair[1] == pair[0] + 1),
		"attachment must not lose output at the snapshot/subscription boundary"
	);
}

#[tokio::test]
async fn timeout_cancel_and_workspace_cancel_have_distinct_truth() {
	let root = tempfile::tempdir().expect("workspace");
	std::fs::write(root.path().join("data"), b"needle").expect("workspace file");
	let workspace = WorkspaceHost::open(root.path()).expect("workspace host");
	let cancelled = CancellationToken::new();
	cancelled.cancel();
	assert!(matches!(
		workspace.search(&workspace.request(), b"needle", None, &cancelled),
		Err(WorkspaceError::Cancelled)
	));

	let exec = ExecHost::new();
	let opened = exec
		.open_session(OpenSessionRequest { cwd_uri: cwd_uri(root.path()), ..Default::default() })
		.await
		.expect("session");
	let (_, timed) = exec
		.exec(
			exec_request(&opened.session, "trap '' TERM; sleep 30"),
			Some(Duration::from_millis(50)),
		)
		.await
		.expect("timed run");
	let timeout_status = loop {
		if let Some(HostExecEvent::Exit(exit)) = timed.next_event().await {
			break exit.status.expect("timeout status");
		}
	};
	assert_eq!(timeout_status.outcome, ExecOutcome::Timeout as i32);

	let (_, cancelled) = exec
		.exec(exec_request(&opened.session, "trap '' TERM; sleep 30"), None)
		.await
		.expect("cancelled run");
	cancelled.cancel();
	let cancelled_status = loop {
		if let Some(HostExecEvent::Exit(exit)) = cancelled.next_event().await {
			break exit.status.expect("cancel status");
		}
	};
	assert_eq!(cancelled_status.outcome, ExecOutcome::Cancelled as i32);
	assert_ne!(timeout_status.outcome, cancelled_status.outcome);
}

#[tokio::test]
async fn real_embedded_python_worker_registers_configured_extensions_when_available() {
	let (Some(site), Some(module)) =
		(std::env::var_os("OMP_TEST_PY_SITE"), std::env::var_os("OMP_TEST_PY_MODULE"))
	else {
		return;
	};
	let mut config = ToolWorkerConfig::new(PathBuf::from(env!("CARGO_BIN_EXE_omp")));
	config.python_site = Some(PathBuf::from(site));
	config.modules = vec![Str::from(module.to_string_lossy().into_owned())];
	let supervisor = ToolWorkerSupervisor::spawn(config)
		.await
		.expect("real embedded Python worker and extension");
	assert!(!supervisor.registrations().is_empty(), "configured extension registered no tools");
	supervisor.shutdown().await;
}
