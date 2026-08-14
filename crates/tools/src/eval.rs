//! Python-only, persistent-session evaluation tool.
//!
//! The tool is a protocol boundary. [`kernel::EmbeddedPython`] is the
//! production in-process resource: it gives every opened session a dedicated
//! worker and persistent Python namespace while sharing OMP's single embedded
//! CPython runtime.

use std::{
	collections::HashMap,
	fmt::Write as _,
	future::Future,
	sync::{
		Arc,
		atomic::{AtomicU64, Ordering},
	},
	time::Duration,
};

use async_stream::stream;
use bytes::Bytes;
use futures::{FutureExt, Stream, future::Either, pin_mut};
use omp_core::{CowBytes, Str};
use omp_proto::inference::v1::{InvokeInput, invoke_input};
use omp_tool::{
	Abort, ArgIssue, ArgIssueKind, BlobRef, CommitError, Constraint, Ev, IncomingParams,
	InterruptWaitError, Outcome, ParamError, Part, PromptCaps, Rev, Tool, ToolParam, ToolSpec,
};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::OnceCell;

use crate::render::TextProjection;

/// Runtime-work timeout accounting shared with host bridge scheduling.
pub mod idle_timeout;
/// Embedded CPython implementation of the eval resource boundary.
pub mod kernel;

const EVAL_DESCRIPTION: &str = r#"Run one step of code in a persistent kernel. State persists across calls and subagents.

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

const MAX_DISPLAY_TEXT_BYTES: usize = 8_000;

/// Runtime accepted by this build of `eval@1`.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, ToolParam)]
#[serde(rename_all = "lowercase")]
pub enum Language {
	/// OMP's embedded `CPython` runtime.
	Py,
}

/// Complete arguments for one Python cell.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize, ToolParam)]
#[serde(deny_unknown_fields)]
pub struct Params {
	/// runtime: "py" for the IPython kernel
	pub language: Language,
	/// code to run in this eval call, verbatim. Use top-level await freely.
	pub code:     Str,
	/// short label shown in transcript (e.g. "imports", "load config")
	pub title:    Option<Str>,
	/// timeout for this eval call in seconds; 0 disables the cell timeout
	pub timeout:  Option<f64>,
	/// wipe this language's kernel before running. Other languages are
	/// untouched.
	pub reset:    Option<bool>,
}

/// Ordered text stream emitted by Python.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputChannel {
	/// Python standard output.
	Stdout,
	/// Python standard error.
	Stderr,
}

/// A live, cell-bounded output update.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Update {
	/// Stream that owns these bytes.
	pub channel:  OutputChannel,
	/// Exact bytes captured within this cell.
	#[serde(with = "cow_bytes")]
	pub data:     CowBytes<'static>,
	/// Monotonic sequence within the cell.
	pub sequence: u64,
}

/// One retained output frame in the durable result.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct OutputFrame {
	/// Stream that owns these bytes.
	pub channel:  OutputChannel,
	/// Exact bytes captured within this cell.
	#[serde(with = "cow_bytes")]
	pub data:     CowBytes<'static>,
	/// Monotonic sequence within the cell.
	pub sequence: u64,
}

/// Rich output captured from a Python cell.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DisplayOutput {
	/// JSON-compatible display value.
	Json {
		/// Displayed value.
		data: Value,
	},
	/// Image already persisted by the host.
	Image {
		/// Durable blob containing the encoded image.
		blob:      BlobRef,
		/// Image media type.
		mime_type: Str,
	},
	/// Markdown display value.
	Markdown {
		/// Markdown source.
		text: Str,
	},
	/// Structured progress event emitted by a helper.
	Status {
		/// Helper event object.
		event: Value,
	},
}

/// REPL value of the final expression in a cell.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct CellValue {
	/// Stable plain-text representation.
	pub text: Str,
	/// JSON value when Python's JSON encoder accepts the object.
	pub json: Option<Value>,
}

/// Python exception retained as durable cell truth.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PythonException {
	/// Python exception class name.
	pub name:      Str,
	/// Exception message without the class prefix.
	pub message:   Str,
	/// Formatted traceback lines in Python order.
	pub traceback: Vec<Str>,
}

/// Terminal disposition of a cell.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CellOutcome {
	/// Cell completed normally.
	Complete,
	/// Python raised an exception.
	Error,
	/// Runtime-work timeout expired.
	Timeout,
	/// Invocation owner interrupted the cell.
	Cancelled,
}

/// Terminal cell status.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CellStatus {
	/// Stable terminal disposition.
	pub outcome:     CellOutcome,
	/// Process-style status used by transcript consumers (`0` or `1`).
	pub exit_code:   Option<i32>,
	/// Host-measured execution duration.
	pub duration_ms: u64,
	/// Python exception, if any.
	pub exception:   Option<PythonException>,
}

/// Complete terminal result supplied by an eval resource.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct RunCompletion {
	/// Terminal cell status.
	pub status:          CellStatus,
	/// Final REPL value, if the cell produced one.
	pub result:          Option<CellValue>,
	/// Rich display values emitted during execution.
	pub display_outputs: Vec<DisplayOutput>,
	/// Whether retained text was truncated.
	pub truncated:       bool,
	/// Durable full-output blob when text was truncated.
	pub spilled_output:  Option<BlobRef>,
	/// Full output line count before truncation.
	pub total_lines:     usize,
	/// Full output byte count before truncation.
	pub total_bytes:     usize,
}

/// Durable result of one eval call.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct Payload {
	/// Stable identity of the persistent Python session.
	pub session_id:      Bytes,
	/// Host identity of this cell.
	pub cell_id:         Bytes,
	/// Executed runtime.
	pub language:        Language,
	/// Optional caller-provided label.
	pub title:           Option<Str>,
	/// Exact submitted source.
	pub code:            Str,
	/// Whether the namespace was reset immediately before this cell.
	pub reset:           bool,
	/// Ordered retained stdout/stderr frames.
	pub frames:          Vec<OutputFrame>,
	/// Final expression value.
	pub result:          Option<CellValue>,
	/// Rich display values.
	pub display_outputs: Vec<DisplayOutput>,
	/// Terminal status.
	pub status:          CellStatus,
	/// Whether retained output was truncated.
	pub truncated:       bool,
	/// Durable full-output blob when truncated.
	pub spilled_output:  Option<BlobRef>,
	/// Full output line count.
	pub total_lines:     usize,
	/// Full output byte count.
	pub total_bytes:     usize,
}

/// Typed eval resource or validation failure.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Fault {
	/// The timeout was negative or not finite.
	InvalidTimeout,
	/// The environment resource rejected or lost an operation.
	Resource {
		/// Operation that failed.
		operation: Str,
		/// Resource-owned diagnostic.
		message:   Str,
	},
	/// A worker ended without a terminal cell event.
	SessionLost {
		/// Resource-owned diagnostic.
		message: Str,
	},
}

/// Opaque handle for one persistent Python session.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Session {
	/// Resource-owned stable session identifier.
	pub id: Bytes,
}

/// Request to execute one cell in a persistent session.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunRequest {
	/// Exact source text.
	pub code:    Str,
	/// Runtime-work timeout. `None` disables the timeout.
	pub timeout: Option<Duration>,
	/// Whether to replace the persistent namespace first.
	pub reset:   bool,
}

/// Ordered event from an active cell.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RunEvent {
	/// Resource assigned a cell identity.
	Started {
		/// Stable resource-owned identity.
		cell_id: Bytes,
	},
	/// Cell-bounded stdout or stderr.
	Output(Update),
	/// Terminal result.
	Completed(Box<RunCompletion>),
}

/// Request-scoped active Python cell.
pub trait EvalRun: Send {
	/// Waits for the next ordered event.
	fn next_event(&mut self) -> impl Future<Output = Result<Option<RunEvent>, Fault>> + Send + '_;

	/// Interrupts the active cell without disposing its session.
	fn cancel(&self) -> impl Future<Output = Result<(), Fault>> + Send + '_;
}

/// Zero-box resource boundary used by the native eval executor.
pub trait EvalExec: Clone + Send + Sync + 'static {
	/// Active run handle.
	type Run: EvalRun;

	/// Opens the persistent Python session owned by this tool instance.
	fn open_session(&self) -> impl Future<Output = Result<Session, Fault>> + Send + '_;

	/// Starts one cell in an existing session.
	fn run<'a>(
		&'a self,
		session: &'a Session,
		request: RunRequest,
	) -> impl Future<Output = Result<Self::Run, Fault>> + Send + 'a;
}

fn format_display_json(outputs: &[DisplayOutput]) -> String {
	let mut rendered = Vec::new();
	let mut index = 0usize;
	for output in outputs {
		let DisplayOutput::Json { data } = output else {
			continue;
		};
		index += 1;
		let mut text = serde_json::to_string_pretty(data).unwrap_or_else(|_| data.to_string());
		if text.len() > MAX_DISPLAY_TEXT_BYTES {
			let mut end = MAX_DISPLAY_TEXT_BYTES;
			while !text.is_char_boundary(end) {
				end -= 1;
			}
			let elided = text.len() - end;
			text.truncate(end);
			let _ = writeln!(text, "[…{elided}ch elided…]");
		}
		rendered.push(format!("display[{index}]:\n{text}"));
	}
	rendered.join("\n\n")
}

/// Python-only `eval@1` implementation retaining one lazy session per owner.
pub struct EvalTool<E: EvalExec> {
	exec:     E,
	sessions: Mutex<HashMap<Str, Arc<OwnerSession>>>,
	control:  EvalSessionControl,
	spec:     ToolSpec,
}

struct OwnerSession {
	session:          OnceCell<Session>,
	reset_generation: AtomicU64,
}

/// External reset trigger used when chat identity changes.
#[derive(Clone, Default)]
pub struct EvalSessionControl {
	reset_generation: Arc<AtomicU64>,
}

impl EvalSessionControl {
	/// Makes every owner's next committed cell start with a fresh namespace.
	pub fn request_reset(&self) {
		self.reset_generation.fetch_add(1, Ordering::AcqRel);
	}
}

/// Constructs `eval@1` over a persistent Python resource.
pub fn eval<E: EvalExec>(exec: E) -> EvalTool<E> {
	eval_controlled(exec).0
}

/// Constructs `eval@1` together with its owning session reset capability.
pub fn eval_controlled<E: EvalExec>(exec: E) -> (EvalTool<E>, EvalSessionControl) {
	let control = EvalSessionControl::default();
	let tool = EvalTool {
		exec,
		sessions: Mutex::new(HashMap::new()),
		control: control.clone(),
		spec: ToolSpec {
			name:        Str::from("eval"),
			rev:         Rev { family: Str::default(), n: 1 },
			description: Str::from(EVAL_DESCRIPTION),
			schema:      omp_tool::schema::<Params>(),
			constraint:  Constraint::Schema { priority: 100 },
		},
	};
	(tool, control)
}

impl<E: EvalExec> Tool for EvalTool<E> {
	type Fault = Fault;
	type Params = Params;
	type Payload = Payload;
	type Update = Update;

	fn spec(&self) -> &ToolSpec {
		&self.spec
	}

	fn call<'c>(
		&'c self,
		mut params: IncomingParams<'c>,
	) -> impl Stream<Item = Ev<Self::Update, Self::Payload, Self::Fault>> + Send + 'c {
		let owner = params
			.owner()
			.cloned()
			.unwrap_or_else(|| Str::new_static("__direct_eval_owner__"));
		stream! {
			let args = match params.whole::<Params>().await {
				Ok(args) => args,
				Err(error) => {
					yield param_event(error);
					return;
				},
			};
			let timeout = match args.timeout {
				None => Some(Duration::from_secs(30)),
				Some(0.0) => None,
				Some(value) if value.is_finite() && value > 0.0 => Some(Duration::from_secs_f64(value)),
				Some(_) => {
					yield Ev::Done(Outcome::Done { result: Err(Fault::InvalidTimeout), useless: false });
					return;
				},
			};
			if let Err(error) = params.committed().await {
				yield commit_event(error);
				return;
			}

			let reset_generation = self.control.reset_generation.load(Ordering::Acquire);
			let owned = self
				.sessions
				.lock()
				.entry(owner)
				.or_insert_with(|| Arc::new(OwnerSession {
					session: OnceCell::new(),
					reset_generation: AtomicU64::new(reset_generation),
				}))
				.clone();
			let session = match owned.session.get_or_try_init(|| self.exec.open_session()).await {
				Ok(session) => session.clone(),
				Err(fault) => {
					yield Ev::Done(Outcome::Done { result: Err(fault), useless: false });
					return;
				},
			};
			let reset = args.reset.unwrap_or(false)
				|| owned.reset_generation.swap(reset_generation, Ordering::AcqRel) != reset_generation;
			let mut run = match self.exec.run(&session, RunRequest {
				code: args.code.clone(),
				timeout,
				reset,
			}).await {
				Ok(run) => run,
				Err(fault) => {
					yield Ev::Done(Outcome::Done { result: Err(fault), useless: false });
					return;
				},
			};

			let mut cell_id = Bytes::new();
			let mut frames = Vec::new();
			let mut cancellation_reason: Option<Str> = None;
			loop {
				let event = if cancellation_reason.is_some() {
					run.next_event().await
				} else {
					let selected = {
						let next = run.next_event().fuse();
						let interrupt = params.next_interrupt().fuse();
						pin_mut!(next, interrupt);
						match futures::future::select(interrupt, next).await {
							Either::Left((interrupt, _)) => Either::Left(interrupt),
							Either::Right((event, _)) => Either::Right(event),
						}
					};
					match selected {
						Either::Left(interrupt) => {
							let reason = match interrupt {
								Ok(interrupt) => interrupt.reason,
								Err(InterruptWaitError::Closed) => Str::from("invocation owner disappeared"),
								Err(InterruptWaitError::Protocol(reason)) => reason,
							};
							if let Err(fault) = run.cancel().await {
								yield Ev::Done(Outcome::Done { result: Err(fault), useless: false });
								return;
							}
							cancellation_reason = Some(reason);
							continue;
						},
						Either::Right(event) => event,
					}
				};

				match event {
					Ok(Some(RunEvent::Started { cell_id: id })) => cell_id = id,
					Ok(Some(RunEvent::Output(update))) => {
						frames.push(OutputFrame {
							channel: update.channel,
							data: update.data.clone(),
							sequence: update.sequence,
						});
						yield Ev::Update(update);
					},
					Ok(Some(RunEvent::Completed(done))) => {
						let done = *done;
						yield Ev::Done(Outcome::Done {
							result: Ok(Payload {
								session_id: session.id,
								cell_id,
								language: args.language,
								title: args.title,
								code: args.code,
								reset,
								frames,
								result: done.result,
								display_outputs: done.display_outputs,
								status: done.status,
								truncated: done.truncated,
								spilled_output: done.spilled_output,
								total_lines: done.total_lines,
								total_bytes: done.total_bytes,
							}),
							useless: false,
						});
						return;
					},
					Ok(None) => {
						yield Ev::Aborted(Abort::EffectsUnknown {
							reason: cancellation_reason.unwrap_or_else(|| Str::from("eval event stream ended before terminal status")),
						});
						return;
					},
					Err(fault) => {
						yield Ev::Done(Outcome::Done { result: Err(fault), useless: false });
						return;
					},
				}
			}
		}
	}

	fn prompt(&self, view: Result<&Payload, &Fault>, caps: &PromptCaps) -> Vec<Part> {
		let payload = match view {
			Ok(payload) => payload,
			Err(fault) => {
				let Some(mut projection) = TextProjection::new(*caps) else {
					return Vec::new();
				};
				let message = match fault {
					Fault::InvalidTimeout => {
						"eval timeout must be a finite non-negative number".to_owned()
					},
					Fault::Resource { operation, message } => {
						format!("eval {operation} failed: {message}")
					},
					Fault::SessionLost { message } => format!("eval session lost: {message}"),
				};
				projection.push(&message);
				return projection.finish();
			},
		};

		let mut stdout = String::new();
		for frame in &payload.frames {
			stdout.push_str(&String::from_utf8_lossy(&frame.data));
		}
		if let Some(result) = &payload.result
			&& !result.text.is_empty()
		{
			stdout.push_str(&result.text);
			if !result.text.ends_with('\n') {
				stdout.push('\n');
			}
		}
		for display in &payload.display_outputs {
			if let DisplayOutput::Markdown { text } = display {
				stdout.push_str(text);
				if !text.ends_with('\n') {
					stdout.push('\n');
				}
			}
		}
		if let Some(exception) = &payload.status.exception {
			if exception.traceback.is_empty() {
				stdout.push_str(&exception.name);
				stdout.push_str(": ");
				stdout.push_str(&exception.message);
				stdout.push('\n');
			} else {
				for line in &exception.traceback {
					stdout.push_str(line);
					if !line.ends_with('\n') {
						stdout.push('\n');
					}
				}
			}
		}

		let stdout = stdout.trim();
		let display_text = format_display_json(&payload.display_outputs);
		let image_count = payload
			.display_outputs
			.iter()
			.filter(|output| matches!(output, DisplayOutput::Image { .. }))
			.count();
		let visible_display = if display_text.is_empty() && image_count != 0 && stdout.is_empty() {
			format!(
				"(displayed {image_count} image{}; no text output)",
				if image_count == 1 { "" } else { "s" }
			)
		} else {
			display_text
		};
		let stdout_empty = stdout.is_empty();
		let visible_display_empty = visible_display.is_empty();
		let mut text = match (stdout_empty, visible_display_empty) {
			(false, false) => format!("{stdout}\n\n{visible_display}"),
			(false, true) => stdout.to_owned(),
			(true, false) => visible_display,
			(true, true) => "(no output)".to_owned(),
		};

		match payload.status.outcome {
			CellOutcome::Error => {
				let code = payload.status.exit_code.unwrap_or(1);
				text = if stdout_empty && visible_display_empty {
					format!("Command exited with code {code}")
				} else {
					format!("{text}\n\nCommand exited with code {code}")
				};
			},
			CellOutcome::Timeout if stdout_empty && visible_display_empty => {
				text.clear();
				text.push_str("Command timed out");
			},
			CellOutcome::Cancelled if stdout_empty && visible_display_empty => {
				text.clear();
				text.push_str("Command aborted");
			},
			CellOutcome::Complete | CellOutcome::Timeout | CellOutcome::Cancelled => {},
		}

		if payload.truncated {
			let shown_lines = if text.is_empty() {
				0
			} else {
				text.lines().count()
			};
			if let Some(blob) = &payload.spilled_output {
				let _ = write!(
					text,
					"\n\n[truncated: {shown_lines} of {} lines shown; full output in blob {}]",
					payload.total_lines, blob.hash
				);
			} else {
				let _ = write!(
					text,
					"\n\n[truncated: {shown_lines} of {} lines shown]",
					payload.total_lines
				);
			}
		}

		let Some(mut projection) = TextProjection::new(*caps) else {
			return Vec::new();
		};
		projection.push(&text);
		let mut parts = projection.finish();
		if caps.media {
			let mut image_index = 0usize;
			for output in &payload.display_outputs {
				if parts.len() >= usize::from(caps.maximum_parts) {
					break;
				}
				if let DisplayOutput::Image { blob, .. } = output {
					image_index += 1;
					parts.push(Part::Blob {
						blob: blob.clone(),
						alt:  Some(Str::from(format!("display image {image_index}"))),
					});
				}
			}
		}
		parts
	}

	fn invoke_input(&self, update: &Update, invocation_id: &str) -> Option<InvokeInput> {
		let channel = match update.channel {
			OutputChannel::Stdout => invoke_input::chunk::Channel::Stdout,
			OutputChannel::Stderr => invoke_input::chunk::Channel::Stderr,
		};
		Some(InvokeInput {
			invocation_id: invocation_id.to_owned(),
			payload:       Some(invoke_input::Payload::Chunk(invoke_input::Chunk {
				channel: channel as i32,
				data:    update.data.clone().into_bytes(),
			})),
		})
	}
}

fn param_event<U, P>(error: ParamError) -> Ev<U, P, Fault> {
	match error {
		ParamError::Args(issue) => Ev::Args(*issue),
		ParamError::Interrupted(interrupt) => {
			Ev::Aborted(Abort::Interrupted { reason: interrupt.reason })
		},
		ParamError::Protocol(reason) => Ev::Args(protocol_issue(reason)),
	}
}

fn commit_event<U, P>(error: CommitError) -> Ev<U, P, Fault> {
	match error {
		CommitError::Aborted => Ev::Aborted(Abort::InputDropped),
		CommitError::Interrupted(interrupt) => {
			Ev::Aborted(Abort::Interrupted { reason: interrupt.reason })
		},
		CommitError::Protocol(reason) => Ev::Args(protocol_issue(reason)),
	}
}

fn protocol_issue(reason: Str) -> ArgIssue {
	ArgIssue {
		path:     Vec::new(),
		expected: Str::from("one complete eval@1 Python cell object"),
		kind:     ArgIssueKind::Protocol,
		example:  Some(Str::from(r#"{"language":"py","code":"1 + 1"}"#)),
		found:    Some(reason),
	}
}

mod cow_bytes {
	use omp_core::CowBytes;
	use serde::{Deserialize, Deserializer, Serialize, Serializer};

	pub(super) fn serialize<S: Serializer>(
		value: &CowBytes<'static>,
		serializer: S,
	) -> Result<S::Ok, S::Error> {
		value.serialize(serializer)
	}

	pub(super) fn deserialize<'de, D: Deserializer<'de>>(
		deserializer: D,
	) -> Result<CowBytes<'static>, D::Error> {
		Vec::<u8>::deserialize(deserializer).map(CowBytes::from)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn params_accept_omitted_optionals_and_reject_invalid_fields() {
		let python: Params = serde_json::from_value(serde_json::json!({
			"language": "py",
			"code": "value = 1"
		}))
		.expect("Python cell parses");
		assert_eq!(python.language, Language::Py);
		assert_eq!(python.title, None);
		assert_eq!(python.timeout, None);
		assert_eq!(python.reset, None);
		assert!(
			serde_json::from_value::<Params>(serde_json::json!({
				"language": "js",
				"code": "1 + 1"
			}))
			.is_err()
		);
		assert!(
			serde_json::from_value::<Params>(serde_json::json!({
				"language": "py",
				"code": "1 + 1",
				"extra": true
			}))
			.is_err()
		);
	}
}
