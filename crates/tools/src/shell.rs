use std::future::Future;

use async_stream::stream;
use bytes::Bytes;
use futures::{FutureExt, Stream, future::Either, pin_mut};
use omp_core::{CowBytes, Str};
use omp_tool::{
	Abort, ArgIssue, ArgIssueKind, ArtifactLifetime, BlobRef, CommitError, Constraint, Ev,
	ExpectedArtifact, IncomingParams, InterruptWaitError, JobOwner, JobRef, Outcome, ParamError,
	Part, PromptCaps, Rev, Tool, ToolSpec,
};
use omp_proto::inference::v1::{InvokeInput, invoke_input};
use serde::{Deserialize, Serialize};
use tokio::sync::OnceCell;

use crate::render::TextProjection;

const TRANSCRIPT_LIMIT: usize = 64 * 1024;
const SHELL_SCHEMA: &[u8] = br#"{
  "$schema":"https://json-schema.org/draft/2020-12/schema",
  "type":"object",
  "additionalProperties":false,
  "properties":{
    "command":{"type":"string","minLength":1,"description":"Shell script to execute."},
    "timeout_ms":{"type":"integer","minimum":1,"description":"Host-enforced execution timeout in milliseconds."},
    "detach":{"type":"boolean","default":false,"description":"Run as a persistent named process."},
    "name":{"type":"string","minLength":1,"description":"Required stable process name when detach is true."}
  },
  "required":["command"],
  "allOf":[{"if":{"properties":{"detach":{"const":true}},"required":["detach"]},"then":{"required":["name"]}}]
}"#;

/// Complete arguments for `shell@1`.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Params {
	/// Shell script to execute.
	pub command:    Str,
	/// Optional host-enforced timeout in milliseconds.
	#[serde(default)]
	pub timeout_ms: Option<u64>,
	/// Whether to transfer the script to the named-process owner.
	#[serde(default)]
	pub detach:     bool,
	/// Stable named-process name. Required when `detach` is true.
	#[serde(default)]
	pub name:       Option<Str>,
}

/// Ordered output channel from a shell command.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputChannel {
	/// Standard output.
	Stdout,
	/// Standard error.
	Stderr,
	/// Combined pseudo-terminal output.
	Pty,
}

/// One ordered live output update.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Update {
	/// Output stream carrying the bytes.
	pub channel:  OutputChannel,
	/// Exact output bytes.
	#[serde(with = "cow_bytes")]
	pub data:     CowBytes<'static>,
	/// Host-assigned ordering sequence.
	pub sequence: u64,
}

/// One retained output frame in the durable transcript.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TranscriptFrame {
	/// Output stream carrying the bytes.
	pub channel:  OutputChannel,
	/// Exact retained output bytes.
	#[serde(with = "cow_bytes")]
	pub data:     CowBytes<'static>,
	/// Host-assigned ordering sequence.
	pub sequence: u64,
}

/// Terminal process disposition reported by the environment owner.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecOutcome {
	/// The script exited normally.
	Exited,
	/// The script failed to launch or execute.
	Failed,
	/// The host-enforced deadline expired.
	Timeout,
	/// The request owner cancelled the command.
	Cancelled,
	/// Execution was denied by policy.
	Denied,
}

/// Complete terminal execution truth from the host.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ExecStatus {
	/// Stable terminal disposition.
	pub outcome:         ExecOutcome,
	/// Process exit code when one exists.
	pub exit_code:       Option<i32>,
	/// Terminating signal when one exists.
	pub signal:          Option<Str>,
	/// Host-measured elapsed wall time.
	pub wall_clock_ms:   u64,
	/// Host-provided reference to output omitted from the live transcript.
	pub spilled_output:  Option<BlobRef>,
	/// Whether cancellation happened after launch.
	pub aborted:         bool,
	/// Whether the host cannot establish the final effect state.
	pub effects_unknown: bool,
}

/// Durable foreground shell result.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Payload {
	/// Stable identity of the reused environment session.
	pub session_id:           Bytes,
	/// Host identity of this command execution.
	pub exec_id:              Bytes,
	/// Exact submitted script.
	pub command:              Str,
	/// Bounded ordered output retained in the verdict.
	pub transcript:           Vec<TranscriptFrame>,
	/// Whether output exceeded the verdict transcript cap.
	pub transcript_truncated: bool,
	/// Terminal host status, preserved without reinterpretation.
	pub status:               ExecStatus,
}

/// Typed shell resource failure.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Fault {
	/// The environment resource rejected or lost an operation.
	Resource {
		/// Operation that failed.
		operation: Str,
		/// Resource-owned diagnostic.
		message:   Str,
	},
	/// Detached execution did not provide its required stable name.
	DetachNameRequired,
}

/// Module-owned handle for one persistent environment session.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Session {
	/// Opaque environment session identifier, preserved byte-for-byte.
	pub id: Bytes,
}

/// Request to run one command in an existing persistent session.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunRequest {
	/// Exact script text.
	pub command:    Str,
	/// Optional server-enforced timeout in milliseconds.
	pub timeout_ms: Option<u64>,
}

/// Request to create one persistent named process.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DetachRequest {
	/// Stable process name.
	pub name:       Str,
	/// Exact script text.
	pub command:    Str,
	/// Optional server-enforced timeout in milliseconds.
	pub timeout_ms: Option<u64>,
}

/// Named-process result owned by the shell resource adapter.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DetachedJob {
	/// Stable environment job identifier.
	pub id: Str,
	/// Named process generation that authoritatively reports settlement.
	pub owner: JobOwner,
}

/// One event consumed from a foreground environment run.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RunEvent {
	/// The host assigned an execution identity.
	Started {
		/// Stable host execution identity.
		exec_id: Bytes,
	},
	/// Ordered process output.
	Output(Update),
	/// Terminal process status.
	Exit(ExecStatus),
}

/// Request-scoped foreground run whose cancellation leaves its session open.
pub trait ShellRun: Send {
	/// Waits for the next ordered run event.
	fn next_event(&mut self) -> impl Future<Output = Result<Option<RunEvent>, Fault>> + Send + '_;

	/// Requests process-tree cancellation without closing the containing
	/// session.
	fn cancel(&self) -> impl Future<Output = Result<(), Fault>> + Send + '_;
}

/// Zero-box environment resource boundary used by the native shell executor.
pub trait ShellExec: Clone + Send + Sync + 'static {
	/// Request-scoped run handle retaining the host cancellation guard.
	type Run: ShellRun;

	/// Lazily opens the one persistent session owned by this tool.
	fn open_session(&self) -> impl Future<Output = Result<Session, Fault>> + Send + '_;

	/// Starts a foreground script in the existing session.
	fn run<'a>(
		&'a self,
		session: &'a Session,
		request: RunRequest,
	) -> impl Future<Output = Result<Self::Run, Fault>> + Send + 'a;

	/// Transfers a script to the environment named-process owner.
	fn detach(
		&self,
		request: DetachRequest,
	) -> impl Future<Output = Result<DetachedJob, Fault>> + Send + '_;
}

/// Generic `shell@1` implementation retaining one lazy persistent session.
pub struct ShellTool<E: ShellExec> {
	exec:             E,
	session:          OnceCell<Session>,
	spec:             ToolSpec,
	transcript_limit: usize,
}

/// Constructs the native `shell@1` executor over an environment resource.
pub fn shell<E: ShellExec>(exec: E) -> ShellTool<E> {
	ShellTool {
		exec,
		session: OnceCell::new(),
		spec: ToolSpec {
			name:        Str::from("shell"),
			rev:         Rev { family: Str::default(), n: 1 },
			description: Str::from(
				"Execute a shell script in a persistent session, or start a named detached process.",
			),
			schema:      Bytes::from_static(SHELL_SCHEMA),
			constraint:  Constraint::Schema { priority: 100 },
		},
		transcript_limit: TRANSCRIPT_LIMIT,
	}
}

impl<E: ShellExec> ShellTool<E> {
	/// Overrides the durable transcript byte cap while retaining all live
	/// updates.
	#[must_use]
	pub fn with_transcript_limit(mut self, byte_limit: usize) -> Self {
		self.transcript_limit = byte_limit;
		self
	}
}

impl<E: ShellExec> Tool for ShellTool<E> {
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
		stream! {
			let args = match params.whole::<Params>().await {
				Ok(args) => args,
				Err(error) => {
					yield param_event(error);
					return;
				},
			};
			if let Err(error) = params.committed().await {
				yield commit_event(error);
				return;
			}

			if args.detach {
				let Some(name) = args.name else {
					yield Ev::Done(Outcome::Done { result: Err(Fault::DetachNameRequired), useless: false });
					return;
				};
				let detached = {
					let work = self.exec.detach(DetachRequest {
						name,
						command: args.command,
						timeout_ms: args.timeout_ms,
					}).fuse();
					let interrupt = params.next_interrupt().fuse();
					pin_mut!(work, interrupt);
					match futures::future::select(interrupt, work).await {
						Either::Left((interrupt, remaining)) => {
							drop(remaining);
							Either::Left(interrupt)
						},
						Either::Right((result, remaining)) => {
							drop(remaining);
							Either::Right(result)
						},
					}
				};
				match detached {
					Either::Left(interrupt) => {
						let reason = match interrupt {
							Ok(interrupt) => interrupt.reason,
							Err(InterruptWaitError::Closed) => Str::from("invocation owner disappeared during detach"),
							Err(InterruptWaitError::Protocol(reason)) => reason,
						};
						yield Ev::Aborted(Abort::EffectsUnknown { reason });
					},
					Either::Right(Ok(job)) => yield Ev::Done(Outcome::Detached(JobRef {
						id: job.id,
						owner: job.owner,
						artifact: ExpectedArtifact {
							description: Str::from("named process settlement"),
							media_type: Some(Str::from("application/vnd.omp.process-settlement+json")),
							lifetime: ArtifactLifetime::Session,
						},
					})),
					Either::Right(Err(fault)) => {
						yield Ev::Done(Outcome::Done { result: Err(fault), useless: false });
					},
				}
				return;
			}

			let session = match self.session.get_or_try_init(|| self.exec.open_session()).await {
				Ok(session) => session.clone(),
				Err(fault) => {
					yield Ev::Done(Outcome::Done { result: Err(fault), useless: false });
					return;
				},
			};
			let session_id = session.id.clone();
			let command = args.command;
			let mut run = match self.exec.run(&session, RunRequest {
				command: command.clone(),
				timeout_ms: args.timeout_ms,
			}).await {
				Ok(run) => run,
				Err(fault) => {
					yield Ev::Done(Outcome::Done { result: Err(fault), useless: false });
					return;
				},
			};

			let mut exec_id = Bytes::new();
			let mut transcript = Vec::new();
			let mut retained = 0usize;
			let mut transcript_truncated = false;
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
							Either::Left((interrupt, remaining)) => {
								drop(remaining);
								Either::Right(interrupt)
							},
							Either::Right((event, remaining)) => {
								drop(remaining);
								Either::Left(event)
							},
						}
					};
					match selected {
						Either::Left(event) => event,
						Either::Right(interrupt) => {
							let reason = match interrupt {
								Ok(interrupt) => interrupt.reason,
								Err(InterruptWaitError::Closed) => Str::from("invocation owner disappeared"),
								Err(InterruptWaitError::Protocol(reason)) => reason,
							};
							if run.cancel().await.is_err() {
								yield Ev::Aborted(Abort::EffectsUnknown { reason });
								return;
							}
							cancellation_reason = Some(reason);
							continue;
						},
					}
				};

				match event {
					Ok(Some(RunEvent::Started { exec_id: id })) => exec_id = id,
					Ok(Some(RunEvent::Output(update))) => {
						let next_len = retained.saturating_add(update.data.len());
						if next_len <= self.transcript_limit {
							retained = next_len;
							transcript.push(TranscriptFrame {
								channel: update.channel,
								data: update.data.clone(),
								sequence: update.sequence,
							});
						} else {
							transcript_truncated = true;
						}
						yield Ev::Update(update);
					},
					Ok(Some(RunEvent::Exit(status))) => {
						yield Ev::Done(Outcome::Done {
							result: Ok(Payload {
								session_id,
								exec_id,
								command,
								transcript,
								transcript_truncated,
								status,
							}),
							useless: false,
						});
						return;
					},
					Ok(None) => {
						yield Ev::Aborted(Abort::EffectsUnknown {
							reason: cancellation_reason.unwrap_or_else(|| Str::from("exec event stream ended before terminal status")),
						});
						return;
					},
					Err(fault) => {
						let reason = match fault {
							Fault::Resource { operation, message } => Str::from(format!("{operation}: {message}")),
							Fault::DetachNameRequired => Str::from("unexpected detach fault during foreground execution"),
						};
						yield Ev::Aborted(Abort::EffectsUnknown { reason });
						return;
					},
				}
			}
		}
	}

	fn prompt(&self, view: Result<&Payload, &Fault>, caps: &PromptCaps) -> Vec<Part> {
		let Some(mut projection) = TextProjection::new(caps) else {
			return Vec::new();
		};
		match view {
			Ok(payload) => {
				let status = format!(
					"[status={:?}; exit={:?}; signal={:?}; {}ms{}]\n",
					payload.status.outcome,
					payload.status.exit_code,
					payload.status.signal,
					payload.status.wall_clock_ms,
					if payload.status.spilled_output.is_some() {
						"; overflow spilled"
					} else {
						""
					},
				);
				if projection.push(&status) {
					for frame in &payload.transcript {
						let text = String::from_utf8_lossy(&frame.data);
						if !projection.push(&text) {
							break;
						}
					}
				}
			},
			Err(fault) => {
				let text = match fault {
					Fault::Resource { operation, message } => {
						format!("shell {operation} failed: {message}")
					},
					Fault::DetachNameRequired => "shell detach requires a non-empty name".to_owned(),
				};
				projection.push(&text);
			},
		}
		projection.finish()
	}

	fn invoke_input(&self, update: &Update, invocation_id: &str) -> Option<InvokeInput> {
		let channel = match update.channel {
			OutputChannel::Stdout | OutputChannel::Pty => {
				invoke_input::chunk::Channel::Stdout
			},
			OutputChannel::Stderr => invoke_input::chunk::Channel::Stderr,
		};
		Some(InvokeInput {
			invocation_id: invocation_id.to_owned(),
			payload: Some(invoke_input::Payload::Chunk(invoke_input::Chunk {
				channel: channel as i32,
				data: update.data.clone().into_bytes(),
			})),
		})
	}
}

fn param_event<U, P>(error: ParamError) -> Ev<U, P, Fault> {
	match error {
		ParamError::Args(issue) => Ev::Args(issue),
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
		expected: Str::from("one complete shell@1 argument object"),
		kind:     ArgIssueKind::Protocol,
		example:  Some(Str::from(r#"{"command":"printf hello"}"#)),
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
