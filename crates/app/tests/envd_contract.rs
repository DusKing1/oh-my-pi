#![cfg(unix)]
//! End-to-end environment daemon contract tests.

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
	const fn new(marker: PathBuf) -> Self {
		Self::named("effect_probe", marker)
	}

	const fn named(name: &'static str, marker: PathBuf) -> Self {
		Self {
			spec: ToolSpec {
				name:        Str::new_static(name),
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
	const fn new(lease: PathBuf, effect: PathBuf) -> Self {
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
			let Ok(path) = params.pull(|mut doc| async move {
				doc.json().object().key("path").string().finish().await
			}).await else {
				yield Ev::Aborted(Abort::InputDropped);
				return;
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
	const fn new(started: PathBuf) -> Self {
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
	const fn new() -> Self {
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
	client:      EnvClient,
	server:      Arc<EnvServer>,
	root:        TempDir,
	state:       TempDir,
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
		Self { client, server, root, state, server_task }
	}

	const fn client(&self) -> &EnvClient {
		&self.client
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
async fn write_name_is_reserved_before_production_registry_assembly() {
	let root = tempfile::tempdir().expect("workspace scratch directory");
	let state = tempfile::tempdir().expect("state scratch directory");
	let marker = state.path().join("reserved-write-marker");
	let mut registry = Registry::new();
	registry
		.register(EffectTool::named("write", marker))
		.expect("register colliding caller write tool");
	let result = EnvServer::open_local(
		root.path(),
		state.path(),
		registry,
		ToolWorkerConfig::new(PathBuf::from(env!("CARGO_BIN_EXE_omp"))),
	)
	.await;
	let Err(error) = result else {
		panic!("production registry accepted a caller-owned write tool");
	};
	assert_eq!(error.to_string(), "duplicate production tool name: write");
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
		grammar:       omp_llm_catalog::GrammarBits::empty(),
	});
	let identities = advertised
		.iter()
		.map(|tool| (tool.identity.name.as_str(), tool.identity.rev.to_string()))
		.collect::<Vec<_>>();
	assert_eq!(identities, [
		("edit", "hl.1".to_owned()),
		("eval", "1".to_owned()),
		("glob", "1".to_owned()),
		("grep", "1".to_owned()),
		("read", "1".to_owned()),
		("shell", "1".to_owned()),
		("write", "1".to_owned()),
	]);
	let write_definition = advertised
		.iter()
		.find(|tool| tool.identity.name == "write")
		.expect("advertised write definition");
	assert_eq!(
		write_definition.definition.description.as_deref(),
		Some(
			"Creates or overwrites file at specified path.\n\n<conditions>\n- Creating new files \
			 explicitly required by task\n- Replacing entire file contents when editing would be \
			 more complex\n- Supports `.tar`, `.tar.gz`, `.tgz`, `.zip`, and ZIP-based \
			 `.jar`/`.war`/`.ear`/`.apk` archive entries via `archive.ext:path/inside/archive`\n- \
			 Supports SQLite row operations via `db.sqlite:table` (insert), `db.sqlite:table:key` \
			 (update with JSON content, delete with empty content)\n</conditions>\n\n<critical>\n- \
			 You SHOULD use Edit tool for modifying existing files\n- You NEVER create documentation \
			 files (*.md, README) unless explicitly requested\n- You NEVER use emojis unless \
			 requested\n</critical>"
		)
	);
	let (write_schema, write_strict) = write_definition
		.definition
		.input
		.json_schema()
		.expect("write uses JSON Schema grammar");
	assert!(write_strict, "write schema must remain strict");
	assert_eq!(
		write_schema.as_value(),
		&json!({
			"type": "object",
			"additionalProperties": false,
			"required": ["path", "content"],
			"properties": {
				"path": {"type": "string", "description": "file path"},
				"content": {"type": "string", "description": "file content"}
			}
		})
	);
	let definition = |name: &str| {
		advertised
			.iter()
			.find(|tool| tool.identity.name == name)
			.unwrap_or_else(|| panic!("advertised {name} definition"))
	};
	let schema = |name: &str| {
		definition(name)
			.definition
			.input
			.json_schema()
			.unwrap_or_else(|| panic!("{name} uses JSON Schema grammar"))
			.0
			.as_value()
			.clone()
	};
	assert_eq!(
		schema("grep"),
		json!({
			"type": "object",
			"additionalProperties": false,
			"required": ["pattern"],
			"properties": {
				"pattern": {"type": "string", "description": "regex pattern"},
				"path": {"type": "string", "description": "file, directory, glob, internal URL, or \"<file>:<lines>\" selector to search; pass several as a semicolon-delimited list (\"src; tests\"). Omitted -> searches the workspace root (\".\")"},
				"case": {"type": "boolean", "description": "case-sensitive search"},
				"gitignore": {"type": "boolean", "description": "respect gitignore"},
				"skip": {"type": ["number", "null"], "description": "files to skip before collecting results — use to paginate when the prior call hit the file limit"}
			}
		})
	);
	assert_eq!(
		schema("glob"),
		json!({
			"type": "object",
			"additionalProperties": false,
			"properties": {
				"path": {"type": "string", "description": "glob, file, or directory to search — a single path or a semicolon-delimited list (\"src/**/*.ts; test/**/*.ts\"). Omitted -> searches the workspace root (\".\")"},
				"hidden": {"type": "boolean", "description": "include hidden files"},
				"gitignore": {"type": "boolean", "description": "respect gitignore"},
				"limit": {"type": "number", "description": "max results"}
			}
		})
	);
	assert_eq!(
		schema("read"),
		json!({
			"type": "object",
			"additionalProperties": false,
			"required": ["path"],
			"properties": {
				"path": {"type": "string", "description": "Local path, internal URI (e.g. skill://), or URL. Inline selectors are supported."}
			}
		})
	);
	assert_eq!(
		schema("edit"),
		json!({
			"type": "object",
			"additionalProperties": false,
			"required": ["input"],
			"properties": {"input": {"type": "string"}}
		})
	);
	let edit_description = definition("edit")
		.definition
		.description
		.as_deref()
		.expect("edit description");
	assert!(edit_description.starts_with("Line-anchored patch language:"));
	assert!(edit_description.contains("RE-GROUND AFTER EVERY EDIT"));
	assert!(edit_description.ends_with("</critical>\n"));
	let read_description = definition("read")
		.definition
		.description
		.as_deref()
		.expect("read description");
	assert!(read_description.contains("Summary footer names elided ranges?"));
	assert!(read_description.contains("NEVER guess `..`/`…` content."));
	assert_eq!(
		definition("grep").definition.description.as_deref(),
		Some(
			"Searches files/internal URLs: Rust regex, PCRE2 fallback.\n\n<instruction>\n- `path`: \
			 known files, directories, globs, internal URLs; roots `;`-separated.\n- Broad searches \
			 may time out → narrow scope or use `glob` first.\n- One-file line selector: \
			 `src/foo.ts:50-100`; never selects search root.\n- Literal `\\n` or `\\\\n` enables \
			 cross-line patterns.\n</instruction>\n\n<critical>\n- MUST use instead of shell \
			 `grep`/`rg`.\n</critical>"
		)
	);
	assert_eq!(
		definition("glob").definition.description.as_deref(),
		Some(
			"Globs files and directories with fast pattern matching.\n\n<instruction>\n- `path`: \
			 glob, file, or directory; separate targets with `;` (`src/**/*.ts; test/**/*.ts`).\n- \
			 `gitignore` defaults `true`. Set `false` for ignored files such as `.env*`, logs, or \
			 build output.\n- `hidden` defaults `true`; pair it with `gitignore: false` for ignored \
			 dotfiles.\n</instruction>\n\n<output>\nMatches are newest-first and grouped by \
			 directory; directories end in `/`.\n</output>"
		)
	);

	let read =
		invoke_builtin(harness.client(), "builtin-read", "read", "1", json!({"path":"note.txt"}))
			.await;
	assert!(!read.is_error, "read adapter returned an error");
	let read_verdict: Verdict<Value, Value> =
		serde_json::from_slice(&read.json).expect("typed read verdict");
	let Verdict::Ok(read_payload) = read_verdict else {
		panic!("read did not return an ok payload");
	};
	let read_text = read_payload["parts"][0]["text"]
		.as_str()
		.expect("read text part");
	assert!(
		read_text.starts_with("[note.txt#"),
		"read must mint the edit anchor used by the shared document adapter: {read_text}"
	);
	let tag = omp_hashline::compute_snapshot_tag(b"before\n");
	let patch = format!("[note.txt#{tag}]\nPUT 1.=1:\n+after");
	let edit =
		invoke_builtin(harness.client(), "builtin-edit", "edit", "hl.1", json!({"input":patch}))
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

	let write = invoke_builtin(
		harness.client(),
		"builtin-write",
		"write",
		"1",
		json!({"path":"nested/written.txt","content":"written through adapter\n"}),
	)
	.await;
	assert!(
		!write.is_error,
		"write adapter returned an error: {}",
		String::from_utf8_lossy(&write.json)
	);
	let write_verdict: Verdict<Value, Value> =
		serde_json::from_slice(&write.json).expect("typed write verdict");
	let Verdict::Ok(write_payload) = write_verdict else {
		panic!("write did not return an ok payload");
	};
	assert_eq!(write_payload["display_path"], "nested/written.txt");
	assert_eq!(write_payload["operation"], json!({"kind":"plain"}));
	assert_eq!(write_payload["byte_len"], 24);
	assert_eq!(write_payload["reported_len"], 24);
	let write_tag = write_payload["snapshot_tag"]
		.as_str()
		.expect("plain write records a shared hashline snapshot");
	let write_edit = invoke_builtin(
		harness.client(),
		"builtin-edit-written",
		"edit",
		"hl.1",
		json!({"input":format!(
			"[nested/written.txt#{write_tag}]\nPUT 1.=1:\n+changed through shared snapshot"
		)}),
	)
	.await;
	assert!(
		!write_edit.is_error,
		"write snapshot was not consumable by edit: {}",
		String::from_utf8_lossy(&write_edit.json)
	);
	assert_eq!(
		std::fs::read_to_string(harness.root.path().join("nested/written.txt"))
			.expect("write/edit fixture"),
		"changed through shared snapshot\n"
	);
	let written = invoke_builtin(
		harness.client(),
		"builtin-read-written",
		"read",
		"1",
		json!({"path":"nested/written.txt:raw"}),
	)
	.await;
	assert!(!written.is_error, "write/read round trip returned an error");
	let written_verdict: Verdict<Value, Value> =
		serde_json::from_slice(&written.json).expect("typed read-after-write verdict");
	assert!(matches!(written_verdict, Verdict::Ok(_)));

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
		json!({"pattern":"after","path":"note.txt"}),
	)
	.await;
	assert!(!grep.is_error, "grep adapter returned an error");
	let glob = invoke_builtin(
		harness.client(),
		"builtin-glob",
		"glob",
		"1",
		json!({"path":"*.txt","limit":10}),
	)
	.await;
	assert!(!glob.is_error, "glob adapter returned an error");
}

#[tokio::test]
async fn production_eval_covers_bridge_persistence_reset_timeout_cancellation_and_recovery() {
	let harness = Harness::start(Registry::new()).await;
	std::fs::write(harness.root.path().join("bridge-note.txt"), "bridge\n")
		.expect("eval bridge fixture");
	let changed_cwd = harness.root.path().join("eval-mutated-cwd");
	std::fs::create_dir(&changed_cwd).expect("eval cwd mutation fixture");
	let changed_cwd_literal =
		serde_json::to_string(&changed_cwd.to_string_lossy()).expect("encode eval cwd fixture");
	let expected_cwd = std::env::current_dir().expect("current test directory");
	let expected_cwd_literal =
		serde_json::to_string(&expected_cwd.to_string_lossy()).expect("encode expected cwd");

	let seed = invoke_builtin(
		harness.client(),
		"eval-seed",
		"eval",
		"1",
		json!({
			"language":"py",
			"code":format!(
				"import builtins, math, os, sys, threading\nstate = 40\nbuiltins.OMP_EVAL_LEAK = \
				 'owner-a'\nmath.OMP_EVAL_LEAK = 'owner-a'\nsys.modules['omp_eval_leak'] = \
				 object()\nos.environ['OMP_EVAL_LEAK'] = 'owner-a'\nos.chdir({changed_cwd_literal})\ndef \
				 _leaked_thread():\n    while True:\n        pass\nthreading.Thread(target=_leaked_thread, \
				 daemon=False).start()\nprint('seeded')"
			),
			"title":"seed"
		}),
	)
	.await;
	assert!(!seed.is_error, "embedded Python seed cell failed");
	let seed: Verdict<omp_tools::eval::Payload, omp_tools::eval::Fault> =
		serde_json::from_slice(&seed.json).expect("typed eval seed verdict");
	let Verdict::Ok(seed) = seed else {
		panic!("embedded Python seed cell returned a fault");
	};
	assert_eq!(seed.frames.len(), 1);
	assert_eq!(seed.frames[0].channel, omp_tools::eval::OutputChannel::Stdout);
	assert_eq!(seed.frames[0].data.as_ref(), b"seeded\n");
	assert_eq!(seed.status.outcome, omp_tools::eval::CellOutcome::Complete);

	let (unrelated, unrelated_task) = harness.connect("eval-unrelated-owner").await;
	let isolated = invoke_builtin(
		&unrelated,
		"eval-owner-isolation",
		"eval",
		"1",
		json!({
			"language":"py",
			"code":format!(
				"import builtins, math, os, sys\n(hasattr(builtins, 'OMP_EVAL_LEAK'), \
				 hasattr(math, 'OMP_EVAL_LEAK'), 'omp_eval_leak' in sys.modules, \
				 os.environ.get('OMP_EVAL_LEAK'), os.getcwd() == {expected_cwd_literal})"
			)
		}),
	)
	.await;
	assert!(!isolated.is_error, "unrelated eval owner failed");
	let isolated: Verdict<omp_tools::eval::Payload, omp_tools::eval::Fault> =
		serde_json::from_slice(&isolated.json).expect("typed owner-isolation verdict");
	let Verdict::Ok(isolated) = isolated else {
		panic!("unrelated eval owner returned a fault");
	};
	assert_eq!(
		isolated.result.and_then(|result| result.json),
		Some(json!([false, false, false, null, true])),
		"Python process globals leaked between authenticated owners"
	);
	unrelated_task.abort();

	let bridged_glob = invoke_builtin(
		harness.client(),
		"eval-tool-bridge",
		"eval",
		"1",
		json!({"language":"py","code":"tool.glob({'path': 'bridge-note.txt'})"}),
	)
	.await;
	assert!(!bridged_glob.is_error, "eval tool bridge call failed");
	let bridged_glob: Verdict<omp_tools::eval::Payload, omp_tools::eval::Fault> =
		serde_json::from_slice(&bridged_glob.json).expect("typed eval bridge verdict");
	let Verdict::Ok(bridged_glob) = bridged_glob else {
		panic!("eval tool bridge returned a fault");
	};
	assert_eq!(bridged_glob.status.outcome, omp_tools::eval::CellOutcome::Complete);
	assert!(
		bridged_glob
			.result
			.as_ref()
			.and_then(|result| result.json.as_ref())
			.and_then(Value::as_str)
			.is_some_and(|output| output.contains("bridge-note.txt")),
		"glob bridge result did not contain the fixture path"
	);

	let denied_completion = invoke_builtin(
		harness.client(),
		"eval-completion-denied",
		"eval",
		"1",
		json!({
			"language":"py",
			"code":"try:\n    completion('no parent')\nexcept RuntimeError as error:\n    print(str(error))"
		}),
	)
	.await;
	assert!(!denied_completion.is_error, "completion denial cell failed");
	let denied_completion: Verdict<omp_tools::eval::Payload, omp_tools::eval::Fault> =
		serde_json::from_slice(&denied_completion.json).expect("typed completion denial verdict");
	let Verdict::Ok(denied_completion) = denied_completion else {
		panic!("completion denial returned a resource fault");
	};
	assert_eq!(denied_completion.frames.len(), 1);
	assert_eq!(
		denied_completion.frames[0].data.as_ref(),
		b"bridge capability denied: __completion__\n"
	);

	let continued = invoke_builtin(
		harness.client(),
		"eval-continued",
		"eval",
		"1",
		json!({"language":"py","code":"state += 2\nprint(f'cell={state}')\nstate"}),
	)
	.await;
	assert!(!continued.is_error, "embedded Python continuation cell failed");
	let continued: Verdict<omp_tools::eval::Payload, omp_tools::eval::Fault> =
		serde_json::from_slice(&continued.json).expect("typed eval continuation verdict");
	let Verdict::Ok(continued) = continued else {
		panic!("embedded Python continuation cell returned a fault");
	};
	assert_eq!(continued.session_id, seed.session_id);
	assert_eq!(continued.frames.len(), 1);
	assert_eq!(continued.frames[0].data.as_ref(), b"cell=42\n");
	assert_eq!(
		continued.result,
		Some(omp_tools::eval::CellValue { text: Str::from("42"), json: Some(json!(42)) })
	);

	let reset = invoke_builtin(
		harness.client(),
		"eval-reset",
		"eval",
		"1",
		json!({
			"language":"py",
			"code":format!(
				"import builtins, math, os, sys\n('state' in globals(), \
				 hasattr(builtins, 'OMP_EVAL_LEAK'), hasattr(math, 'OMP_EVAL_LEAK'), \
				 'omp_eval_leak' in sys.modules, os.environ.get('OMP_EVAL_LEAK'), \
				 os.getcwd() == {expected_cwd_literal})"
			),
			"reset":true
		}),
	)
	.await;
	assert!(!reset.is_error, "embedded Python reset cell failed");
	let reset: Verdict<omp_tools::eval::Payload, omp_tools::eval::Fault> =
		serde_json::from_slice(&reset.json).expect("typed eval reset verdict");
	let Verdict::Ok(reset) = reset else {
		panic!("embedded Python reset cell returned a fault");
	};
	assert_eq!(reset.session_id, seed.session_id);
	assert!(reset.reset);
	assert_eq!(
		reset.result.and_then(|result| result.json),
		Some(json!([false, false, false, false, null, true])),
		"reset did not replace process-global Python state"
	);

	let timeout_started = std::time::Instant::now();
	let timed_out = invoke_builtin(
		harness.client(),
		"eval-timeout",
		"eval",
		"1",
		json!({"language":"py","code":"import time\ntime.sleep(5)","timeout":0.025}),
	)
	.await;
	assert!(!timed_out.is_error, "timed-out Python cell did not return typed cell truth");
	let timed_out: Verdict<omp_tools::eval::Payload, omp_tools::eval::Fault> =
		serde_json::from_slice(&timed_out.json).expect("typed eval timeout verdict");
	let Verdict::Ok(timed_out) = timed_out else {
		panic!("timed-out Python cell returned a resource fault");
	};
	assert_eq!(timed_out.status.outcome, omp_tools::eval::CellOutcome::Timeout);
	assert!(
		timeout_started.elapsed() < Duration::from_millis(500),
		"hard eval timeout exceeded 500ms: {:?}",
		timeout_started.elapsed()
	);
	assert_eq!(
		timed_out
			.status
			.exception
			.as_ref()
			.map(|exception| exception.name.as_str()),
		Some("TimeoutError")
	);

	let recovered = invoke_builtin(
		harness.client(),
		"eval-after-timeout",
		"eval",
		"1",
		json!({"language":"py","code":"6 * 7"}),
	)
	.await;
	assert!(!recovered.is_error, "Python kernel did not recover after timeout");
	let recovered: Verdict<omp_tools::eval::Payload, omp_tools::eval::Fault> =
		serde_json::from_slice(&recovered.json).expect("typed post-timeout eval verdict");
	let Verdict::Ok(recovered) = recovered else {
		panic!("post-timeout Python cell returned a fault");
	};
	assert_eq!(recovered.session_id, seed.session_id);
	assert!(recovered.reset, "respawn after timeout was not reported as a reset");
	assert_eq!(
		recovered.result,
		Some(omp_tools::eval::CellValue { text: Str::from("42"), json: Some(json!(42)) })
	);

	let started = harness.root.path().join("eval-cancel-started");
	let started_literal =
		serde_json::to_string(&started.to_string_lossy()).expect("encode cancellation marker path");
	let code = format!(
		"import threading\nfrom pathlib import Path\ndef spin_forever():\n    while True:\n        \
		 pass\nthreading.Thread(target=spin_forever, \
		 daemon=False).start()\nPath({started_literal}).write_text('started')\nwhile True:\n    pass"
	);
	let mut cancelled = harness
		.client()
		.invoke(InvokeTool {
			invocation_id: "eval-cancel".into(),
			name: "eval".into(),
			rev: "1".into(),
			..InvokeTool::default()
		})
		.await
		.expect("open cancellable eval invocation");
	assert!(matches!(
		cancelled
			.next_event()
			.await
			.expect("eval cancellation accepted"),
		Some(InvocationEvent::Accepted(_))
	));
	cancelled
		.commit_args(Bytes::from(
			serde_json::to_vec(&json!({"language":"py","code":code}))
				.expect("encode cancellable eval arguments"),
		))
		.await
		.expect("commit cancellable eval arguments");
	tokio::time::timeout(Duration::from_secs(2), async {
		while !started.exists() {
			tokio::task::yield_now().await;
		}
	})
	.await
	.expect("embedded Python cancellation cell never became active");
	cancelled.guard().cancel();
	let terminal = tokio::time::timeout(Duration::from_secs(2), cancelled.next_event())
		.await
		.expect("eval cancellation terminal timeout")
		.expect("eval cancellation terminal event")
		.expect("eval cancellation stream closed");
	let InvocationEvent::Verdict(terminal) = terminal else {
		panic!("eval cancellation did not produce a verdict");
	};
	let verdict: Verdict<Value, Value> =
		serde_json::from_slice(&terminal.json).expect("decode eval cancellation verdict");
	assert!(matches!(verdict, Verdict::Aborted(Abort::EffectsUnknown { .. })));

	let after_cancel = invoke_builtin(
		harness.client(),
		"eval-after-cancel",
		"eval",
		"1",
		json!({"language":"py","code":"7 * 7"}),
	)
	.await;
	assert!(!after_cancel.is_error, "Python kernel did not recover after cancellation");
	let after_cancel: Verdict<omp_tools::eval::Payload, omp_tools::eval::Fault> =
		serde_json::from_slice(&after_cancel.json).expect("typed post-cancel eval verdict");
	let Verdict::Ok(after_cancel) = after_cancel else {
		panic!("post-cancel Python cell returned a fault");
	};
	assert!(after_cancel.reset, "respawn after cancellation was not reported as a reset");
	assert_eq!(
		after_cancel.result,
		Some(omp_tools::eval::CellValue { text: Str::from("49"), json: Some(json!(49)) })
	);

	let crashed = invoke_builtin(
		harness.client(),
		"eval-child-crash",
		"eval",
		"1",
		json!({"language":"py","code":"import os\nos._exit(17)"}),
	)
	.await;
	assert!(crashed.is_error, "eval child crash was reported as a successful cell");
	let crashed: Verdict<omp_tools::eval::Payload, omp_tools::eval::Fault> =
		serde_json::from_slice(&crashed.json).expect("typed eval crash verdict");
	assert!(matches!(crashed, Verdict::Fault(omp_tools::eval::Fault::SessionLost { .. })));

	let after_crash = invoke_builtin(
		harness.client(),
		"eval-after-crash",
		"eval",
		"1",
		json!({"language":"py","code":"8 * 8"}),
	)
	.await;
	let after_crash: Verdict<omp_tools::eval::Payload, omp_tools::eval::Fault> =
		serde_json::from_slice(&after_crash.json).expect("typed post-crash eval verdict");
	let Verdict::Ok(after_crash) = after_crash else {
		panic!("post-crash Python cell returned a fault");
	};
	assert!(after_crash.reset, "respawn after crash was not reported as a reset");
	assert_eq!(after_crash.result.and_then(|result| result.json), Some(json!(64)));
}

#[tokio::test]
async fn uds_clients_cannot_invoke_session_local_eval_but_retain_ordinary_tools() {
	let harness = Harness::start(Registry::new()).await;
	std::fs::write(harness.root.path().join("uds-note.txt"), "uds read\n")
		.expect("UDS read fixture");
	let advertised = harness.server.registry().advertise(LoweringCaps {
		strict_schema: true,
		grammar:       omp_llm_catalog::GrammarBits::empty(),
	});
	assert!(
		advertised.iter().any(|tool| tool.identity.name == "eval"),
		"in-process registry did not advertise eval"
	);
	let local_eval = invoke_builtin(
		harness.client(),
		"local-eval-capability",
		"eval",
		"1",
		json!({"language":"py","code":"2 + 3"}),
	)
	.await;
	assert!(!local_eval.is_error, "session-local in-process eval was denied");

	let socket = harness.state.path().join("env-remote.sock");
	let shutdown = CancellationToken::new();
	let server = Arc::clone(&harness.server);
	let serve_shutdown = shutdown.clone();
	let socket_for_server = socket.clone();
	let server_task = tokio::spawn(async move {
		server
			.serve_uds(&socket_for_server, serve_shutdown, None)
			.await
	});
	tokio::time::timeout(Duration::from_secs(2), async {
		while !socket.exists() {
			tokio::task::yield_now().await;
		}
	})
	.await
	.expect("UDS environment socket did not become ready");
	let (remote, bridge_task) = EnvServer::connect_owner_uds(&socket)
		.await
		.expect("connect owner UDS client");
	remote
		.hello(ClientHello {
			client: "envd-contract-uds".into(),
			schema_rev: SCHEMA_REV,
			..ClientHello::default()
		})
		.await
		.expect("UDS environment hello");

	let mut denied = remote
		.invoke(InvokeTool {
			invocation_id: "remote-eval-denied".into(),
			name: "eval".into(),
			rev: "1".into(),
			..InvokeTool::default()
		})
		.await
		.expect("open denied remote eval request");
	let error = denied
		.next_event()
		.await
		.expect_err("UDS eval unexpectedly produced an event");
	let omp_env::ClientError::Protocol(error) = error else {
		panic!("UDS eval denial was not a typed protocol error");
	};
	assert_eq!(error.code, omp_proto::env::v1::ProtocolErrorCode::PermissionDenied as i32);
	assert_eq!(error.message, "eval is available only through the session-local environment");

	let read = invoke_builtin(
		&remote,
		"remote-read-allowed",
		"read",
		"1",
		json!({"path":"uds-note.txt:raw"}),
	)
	.await;
	assert!(!read.is_error, "ordinary UDS read was denied");

	shutdown.cancel();
	bridge_task.abort();
	let _ = server_task.await;
}

#[tokio::test]
async fn opt_in_python_adds_one_worker_route_and_default_adds_none() {
	let mut worker = ToolWorkerConfig::new(PathBuf::from(env!("CARGO_BIN_EXE_omp")));
	worker.modules.push(Str::new_static(PY_EVAL_MODULE));
	let harness = Harness::start_with_worker(Registry::new(), worker).await;
	let registry = harness.server.registry();
	let advertised = registry.advertise(LoweringCaps {
		strict_schema: true,
		grammar:       omp_llm_catalog::GrammarBits::empty(),
	});
	assert_eq!(advertised.len(), 8);
	assert_eq!(registry.route("py_eval").expect("python route"), ToolRoute::Worker);
	assert_eq!(
		registry
			.live_identity("py_eval")
			.map(|(_, revision)| revision.to_string())
			.as_deref(),
		Some("1")
	);
	let verdict =
		invoke_builtin(harness.client(), "builtin-python", "py_eval", "1", json!({"code":"40 + 2"}))
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
		committed
			.next_event()
			.await
			.expect("committed speculative update"),
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
		duplicate
			.next_event()
			.await
			.expect("duplicate speculative update"),
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
	assert_eq!(error.code, omp_proto::env::v1::ProtocolErrorCode::AlreadyExists as i32);
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
			// SAFETY: `pid` is a parsed child process identifier; `kill` only reads it.
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
	let upload = client.blob_put().expect("begin blob upload");
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
	while let BlobDownloadEvent::Chunk(chunk) = download
		.next_event()
		.await
		.expect("blob event")
		.expect("blob event present")
	{
		received.extend_from_slice(&chunk.data);
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
					text: "i=0; while [ $i -lt 50 ]; do echo output; sleep 0.01; i=$((i + 1)); done"
						.into(),
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
	assert_ne!(sequences, [] as [u64; 0]);
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

#[tokio::test]
async fn uds_retire_unlinks_listener_and_drains_existing_clients() {
	let harness = Harness::start(Registry::new()).await;
	let socket = harness.state.path().join("env-retire.sock");
	let shutdown = CancellationToken::new();
	let server = Arc::clone(&harness.server);
	let serve_shutdown = shutdown.clone();
	let socket_for_server = socket.clone();
	let mut server_task = tokio::spawn(async move {
		server
			.serve_uds(&socket_for_server, serve_shutdown, None)
			.await
	});
	tokio::time::timeout(Duration::from_secs(2), async {
		while !socket.exists() {
			tokio::task::yield_now().await;
		}
	})
	.await
	.expect("UDS environment socket did not become ready");

	let (retiring, retiring_bridge) = EnvServer::connect_owner_uds(&socket)
		.await
		.expect("connect retiring client");
	let retiring_hello = retiring
		.hello(ClientHello {
			client: "envd-retiring".into(),
			schema_rev: SCHEMA_REV,
			..ClientHello::default()
		})
		.await
		.expect("retiring client hello");
	let (remaining, remaining_bridge) = EnvServer::connect_owner_uds(&socket)
		.await
		.expect("connect remaining client");
	let remaining_hello = remaining
		.hello(ClientHello {
			client: "envd-remaining".into(),
			schema_rev: SCHEMA_REV,
			..ClientHello::default()
		})
		.await
		.expect("remaining client hello");
	assert!(!retiring_hello.server_build.is_empty());
	assert_eq!(retiring_hello.server_build, remaining_hello.server_build);

	retiring.retire().await.expect("retire acknowledgement");
	tokio::time::timeout(Duration::from_secs(2), async {
		loop {
			match tokio::net::UnixStream::connect(&socket).await {
				Err(error)
					if matches!(
						error.kind(),
						std::io::ErrorKind::NotFound | std::io::ErrorKind::ConnectionRefused
					) =>
				{
					break;
				},
				_ => tokio::task::yield_now().await,
			}
		}
	})
	.await
	.expect("retired UDS listener remained reachable");
	remaining
		.list_processes(ListProcesses::default())
		.await
		.expect("existing client request after retire");

	drop(retiring);
	retiring_bridge.abort();
	assert!(
		tokio::time::timeout(Duration::from_millis(50), &mut server_task)
			.await
			.is_err(),
		"server exited while an existing client remained connected"
	);
	drop(remaining);
	remaining_bridge.abort();
	tokio::time::timeout(Duration::from_secs(2), server_task)
		.await
		.expect("retired server did not finish draining")
		.expect("retired server task panicked")
		.expect("retired server failed");
}

#[tokio::test]
async fn in_process_retire_is_rejected_as_unsupported() {
	let harness = Harness::start(Registry::new()).await;
	let error = harness
		.client()
		.retire()
		.await
		.expect_err("in-process retire succeeded");
	let omp_env::ClientError::Protocol(error) = error else {
		panic!("in-process retire did not return a protocol error");
	};
	assert_eq!(error.code, omp_proto::env::v1::ProtocolErrorCode::Unsupported as i32);
	assert_eq!(error.message, "retire is not available on this transport");
}
