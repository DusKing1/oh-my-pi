//! Exact Python-only eval schema and rendering goldens.

use std::{
	future::{Future, ready},
	sync::{Arc, LazyLock},
};

use bytes::Bytes;
use futures::StreamExt as _;
use omp_core::Str;
use omp_tool::{BlobRef, Ev, IncomingParams, Outcome, Part, PromptCaps, Tool};
use omp_tools::eval::{
	self, CellOutcome, CellStatus, CellValue, DisplayOutput, EvalExec, EvalRun, Fault, Language,
	OutputChannel, OutputFrame, Payload, RunEvent, RunRequest, Session,
};
use serde_json::json;

#[derive(Clone)]
struct UnusedExec;

struct UnusedRun;

impl EvalRun for UnusedRun {
	fn next_event(&mut self) -> impl Future<Output = Result<Option<RunEvent>, Fault>> + Send + '_ {
		ready(Ok(None))
	}

	fn cancel(&self) -> impl Future<Output = Result<(), Fault>> + Send + '_ {
		ready(Ok(()))
	}
}

impl EvalExec for UnusedExec {
	type Run = UnusedRun;

	fn open_session(&self) -> impl Future<Output = Result<Session, Fault>> + Send + '_ {
		ready(Err(Fault::SessionLost { message: "unused".into() }))
	}

	fn run<'a>(
		&'a self,
		_session: &'a Session,
		_request: RunRequest,
	) -> impl Future<Output = Result<Self::Run, Fault>> + Send + 'a {
		ready(Err(Fault::SessionLost { message: "unused".into() }))
	}
}

fn tool() -> impl Tool<Payload = Payload, Fault = Fault> {
	eval::eval(UnusedExec)
}

static PYTHON: LazyLock<Arc<omp_py::Engine>> = LazyLock::new(|| {
	Arc::new(
		omp_py::Engine::builder()
			.init()
			.expect("initialize embedded Python for eval test"),
	)
});

async fn execute(tool: &eval::EvalTool<eval::kernel::EmbeddedPython>, code: &str) -> Payload {
	let raw = json!({"language":"py","code":code}).to_string();
	let (feed, params) = IncomingParams::channel();
	feed
		.args_committed(Str::from(raw))
		.expect("eval invocation remains live");
	let mut events = Box::pin(tool.call(params));
	while let Some(event) = events.next().await {
		match event {
			Ev::Done(Outcome::Done { result: Ok(payload), .. }) => return payload,
			Ev::Done(Outcome::Done { result: Err(fault), .. }) => {
				panic!("eval returned a fault: {fault:?}")
			},
			Ev::Done(Outcome::Detached(_)) => panic!("eval unexpectedly detached"),
			Ev::Args(issue) => panic!("eval rejected arguments: {issue:?}"),
			Ev::Aborted(abort) => panic!("eval aborted: {abort:?}"),
			Ev::Update(_) => {},
		}
	}
	panic!("eval stream ended without a terminal payload")
}

fn status(outcome: CellOutcome) -> CellStatus {
	CellStatus {
		outcome,
		exit_code: if outcome == CellOutcome::Complete {
			Some(0)
		} else {
			Some(1)
		},
		duration_ms: 12,
		exception: None,
	}
}

fn payload() -> Payload {
	Payload {
		session_id:      Bytes::from_static(b"session"),
		cell_id:         Bytes::from_static(b"cell"),
		language:        Language::Py,
		title:           Some("cell".into()),
		code:            "print('before')".into(),
		reset:           false,
		frames:          Vec::new(),
		result:          None,
		display_outputs: Vec::new(),
		status:          status(CellOutcome::Complete),
		truncated:       false,
		spilled_output:  None,
		total_lines:     0,
		total_bytes:     0,
	}
}

fn frame(channel: OutputChannel, text: &str, sequence: u64) -> OutputFrame {
	OutputFrame { channel, data: text.as_bytes().to_vec().into(), sequence }
}

fn project(payload: &Payload, media: bool) -> Vec<Part> {
	tool().prompt(Ok(payload), &PromptCaps {
		maximum_parts: 8,
		maximum_text_bytes: 64 * 1024,
		media,
	})
}

fn text(parts: &[Part]) -> String {
	parts
		.iter()
		.filter_map(|part| match part {
			Part::Text { text } => Some(text.as_str()),
			Part::Json { .. } | Part::Blob { .. } => None,
		})
		.collect()
}

#[test]
fn python_only_schema_is_exact() {
	let actual: serde_json::Value = serde_json::from_slice(&tool().spec().schema).unwrap();
	assert_eq!(
		actual,
		json!({
			"type": "object",
			"additionalProperties": false,
			"required": ["language", "code"],
			"properties": {
				"language": {
					"type": "string",
					"enum": ["py"],
					"description": "runtime: \"py\" for the IPython kernel"
				},
				"code": {
					"type": "string",
					"description": "code to run in this eval call, verbatim. Use top-level await freely."
				},
				"title": {
					"type": "string",
					"description": "short label shown in transcript (e.g. \"imports\", \"load config\")"
				},
				"timeout": {
					"type": "number",
					"description": "timeout for this eval call in seconds; 0 disables the cell timeout"
				},
				"reset": {
					"type": "boolean",
					"description": "wipe this language's kernel before running. Other languages are untouched."
				}
			}
		})
	);
}

#[test]
fn python_model_description_is_exact_and_has_no_javascript_branch() {
	let expected = r#"Run one step of code in a persistent kernel. State persists across calls and subagents.

Work incrementally: imports → define → test → use, each its own cell. Re-run setup ONLY after `reset`, kernel crash.
Parallelize *within* a cell with `parallel(thunks)`, not by batching.

Top-level `await` works; `asyncio.run(…)` raises error.

On error, fix and re-run only the failing step.

<prelude>
Sync; kwargs.
```
display(value) → None        print(value, ...) → None
read(path, offset?=1, limit?=None) → str
write(path, content) → str
env(key?=None, value?=None) → str | None | dict
output(*ids, format?="raw", query=None, offset=None, limit=None) → str | dict | list[dict]
tool.<name>(args) → unknown
    Invoke any session tool; `args` = its parameter object.
completion(prompt, model?="default"|"smol"|"slow", system=None, schema=None) → str | dict
    Oneshot, stateless (no history/tools). `model`: "smol" fast | "default" session | "slow" most capable. `schema` (JSON-Schema) → parsed object.
agent(prompt, agent?="task", label=None, schema=None, schema_mode?="permissive", isolated=None, apply=None, merge=None, handle=False) → str | dict
    Run a subagent → final output. `agent` selects a discovered agent; omit it to use `task`. `schema` overrides agent/session schemas; `schemaMode`/`schema_mode`: "permissive" | "strict". Effective schemas return parsed data. `isolated` requests a worktree; `apply`/`merge` control its changes. Background via `local://` files named in the prompt. `handle` → { text, output, handle: "agent://<id>", id, agent }, parsed `data` when structured.
parallel(thunks) → list     pipeline(items, ...stages) → list
log(message) → None         phase(title) → None
budget → `budget.total` (ceiling or None), `budget.spent()`, `budget.remaining()`; ceiling `+Nk` advisory, `+Nk!` hard.
```
</prelude>
<dag>
Acyclic waves via `agent(…, handle=true)` + `pipeline`/`parallel`:
- **Name nodes.** Capture agent result → `handle` (`agent://<id>`) + `output`.
- **Wire edges.** Put upstream `handle`/`output` in downstream prompt. Bulk: `write("local://<name>.md", …)`.
- **`pipeline`** = staged waves, barrier between stages. **`parallel`** = one wave.
- **Isolate failure.** Wrap risky nodes in try/except; a failure degrades only its subtree.
- **Acyclic only.** No node waits on its own descendant.
</dag>

<critical>
Prior top-level names survive into the next cell — reuse; NEVER re-import/re-declare. Re-read only if file changed since last read.
</critical>"#;
	assert_eq!(tool().spec().description, expected);
	assert!(!tool().spec().description.contains("JavaScript"));
	assert!(!tool().spec().description.contains("Bun"));
}

#[test]
fn stdout_stderr_result_and_display_json_projection_is_exact() {
	let mut value = payload();
	value.frames = vec![
		frame(OutputChannel::Stdout, "before\n", 1),
		frame(OutputChannel::Stderr, "warning\n", 2),
	];
	value.result = Some(CellValue { text: "42".into(), json: Some(json!(42)) });
	value.display_outputs =
		vec![DisplayOutput::Json { data: json!({"exit_code": 0, "stdout": "hi"}) }];
	assert_eq!(
		text(&project(&value, false)),
		"before\nwarning\n42\n\ndisplay[1]:\n{\n  \"exit_code\": 0,\n  \"stdout\": \"hi\"\n}"
	);
}

#[test]
fn no_output_and_python_error_projection_are_exact() {
	assert_eq!(text(&project(&payload(), false)), "(no output)");

	let mut value = payload();
	value.status = CellStatus {
		outcome:     CellOutcome::Error,
		exit_code:   Some(1),
		duration_ms: 4,
		exception:   Some(eval::PythonException {
			name:      "ValueError".into(),
			message:   "bad value".into(),
			traceback: vec![
				"Traceback (most recent call last):".into(),
				"ValueError: bad value".into(),
			],
		}),
	};
	assert_eq!(
		text(&project(&value, false)),
		"Traceback (most recent call last):\nValueError: bad value\n\nCommand exited with code 1"
	);
}

#[test]
fn oversized_display_json_and_spilled_output_lookup_are_exact() {
	let mut value = payload();
	value.display_outputs =
		vec![DisplayOutput::Json { data: json!({"payload": "x".repeat(9_000)}) }];
	value.truncated = true;
	value.total_lines = 40;
	value.total_bytes = 20_000;
	value.spilled_output = Some(BlobRef {
		hash:       Str::from("sha256:full-eval-output"),
		media_type: Str::from("text/plain"),
		byte_len:   20_000,
	});
	let rendered = text(&project(&value, false));
	assert!(rendered.contains("[…"));
	assert!(rendered.contains("ch elided…]"));
	assert!(
		rendered.ends_with(
			"[truncated: 4 of 40 lines shown; full output in blob sha256:full-eval-output]"
		)
	);
}

#[test]
fn image_display_projects_blob_without_base64_text() {
	let mut value = payload();
	value.display_outputs = vec![DisplayOutput::Image {
		blob:      BlobRef {
			hash:       Str::from("sha256:image"),
			media_type: Str::from("image/png"),
			byte_len:   68,
		},
		mime_type: Str::from("image/png"),
	}];
	let parts = project(&value, true);
	assert_eq!(text(&parts), "(displayed 1 image; no text output)");
	assert!(matches!(
		parts.as_slice(),
		[Part::Text { .. }, Part::Blob { blob, alt: Some(alt) }]
			if blob.hash == "sha256:image" && alt == "display image 1"
	));
}

#[test]
fn invalid_timeout_fault_projection_is_exact() {
	let parts = tool().prompt(Err(&Fault::InvalidTimeout), &PromptCaps {
		maximum_parts:      1,
		maximum_text_bytes: 1024,
		media:              false,
	});
	assert_eq!(text(&parts), "eval timeout must be a finite non-negative number");
}

#[tokio::test]
async fn external_session_reset_separates_chat_state_and_preserves_the_new_session() {
	let runtime = eval::kernel::EmbeddedPython::new(Arc::clone(&PYTHON));
	let (tool, control) = eval::eval_controlled(runtime);

	let session_a = execute(&tool, "session_value = 'A'\nsession_value").await;
	assert_eq!(session_a.result, Some(CellValue { text: Str::from("'A'"), json: Some(json!("A")) }));
	let persisted_a = execute(&tool, "session_value").await;
	assert_eq!(
		persisted_a.result,
		Some(CellValue { text: Str::from("'A'"), json: Some(json!("A")) })
	);

	control.request_reset();
	let session_b = execute(&tool, "'session_value' in globals()").await;
	assert!(session_b.reset, "session-owner reset was not consumed by the next cell");
	assert_eq!(
		session_b.result,
		Some(CellValue { text: Str::from("False"), json: Some(json!(false)) })
	);

	execute(&tool, "session_value = 'B'").await;
	let persisted_b = execute(&tool, "session_value").await;
	assert!(!persisted_b.reset, "external reset leaked into later session-B cells");
	assert_eq!(
		persisted_b.result,
		Some(CellValue { text: Str::from("'B'"), json: Some(json!("B")) })
	);
}
