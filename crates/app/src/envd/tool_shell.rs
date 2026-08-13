use std::{future::{self, Future}, time::Duration};

use omp_core::{CowBytes, Str, encoding::hex, fmts};
use omp_proto::env::v1::{
	ExecOutcome as EnvExecOutcome, ExecRequest, OpenSessionRequest, OutputChannel as EnvOutputChannel,
	ProcessSpec, RestartPolicy, RestartSpec, Script, StartProcess,
};
use omp_tool::{BlobRef, JobOwner};
use omp_tools::shell::{
	DetachRequest, DetachedJob, ExecOutcome, ExecStatus, Fault, OutputChannel, RunEvent, RunRequest,
	Session, ShellExec, ShellRun, Update,
};

use super::exec::{ExecError, ExecEvent, ExecHost, ExecRun};

/// Shell resource adapter backed by the app-owned execution host.
#[derive(Clone)]
pub(crate) struct ShellExecHost {
	host:    ExecHost,
	cwd_uri: Str,
}

impl ShellExecHost {
	/// Binds shell execution to the workspace root URI used for sessions and
	/// detached processes.
	pub(crate) fn new(host: ExecHost, cwd_uri: Str) -> Self {
		Self { host, cwd_uri }
	}
}

/// Foreground shell run retaining the concrete host's process-tree guard.
pub(crate) struct HostShellRun {
	started: Option<bytes::Bytes>,
	run:     ExecRun,
}

impl ShellRun for HostShellRun {
	fn next_event(
		&mut self,
	) -> impl Future<Output = Result<Option<RunEvent>, Fault>> + Send + '_ {
		async move {
			if let Some(exec_id) = self.started.take() {
				return Ok(Some(RunEvent::Started { exec_id }));
			}
			let Some(event) = self.run.next_event().await else {
				return Ok(None);
			};
			map_event(event).map(Some)
		}
	}

	fn cancel(&self) -> impl Future<Output = Result<(), Fault>> + Send + '_ {
		self.run.cancel();
		future::ready(Ok(()))
	}
}

impl ShellExec for ShellExecHost {
	type Run = HostShellRun;

	fn open_session(&self) -> impl Future<Output = Result<Session, Fault>> + Send + '_ {
		async move {
			let opened = self
				.host
				.open_session(OpenSessionRequest {
					cwd_uri: self.cwd_uri.to_string(),
					pty: None,
					..Default::default()
				})
				.await
				.map_err(|error| resource_fault("open_session", error))?;
			Ok(Session { id: opened.session })
		}
	}

	fn run<'a>(
		&'a self,
		session: &'a Session,
		request: RunRequest,
	) -> impl Future<Output = Result<Self::Run, Fault>> + Send + 'a {
		async move {
			let (started, run) = self
				.host
				.exec(
					ExecRequest {
						session: session.id.clone(),
						source: Some(Script {
							text: request.command.to_string(),
							..Default::default()
						}),
						..Default::default()
					},
					request.timeout_ms.map(Duration::from_millis),
				)
				.await
				.map_err(|error| resource_fault("run", error))?;
			Ok(HostShellRun { started: Some(started.exec), run })
		}
	}

	fn detach(
		&self,
		request: DetachRequest,
	) -> impl Future<Output = Result<DetachedJob, Fault>> + Send + '_ {
		async move {
			let started = self
				.host
				.start_process(StartProcess {
					name: request.name.to_string(),
					spec: Some(ProcessSpec {
						source: Some(Script {
							text: request.command.to_string(),
							..Default::default()
						}),
						cwd_uri: self.cwd_uri.to_string(),
						restart: Some(RestartSpec {
							policy: RestartPolicy::Never as i32,
							..Default::default()
						}),
						..Default::default()
					}),
					..Default::default()
				})
				.await
				.map_err(|error| resource_fault("detach", error))?;
			let id = fmts!("{}#{}", started.name, started.generation);
			Ok(DetachedJob {
				id,
				owner: JobOwner::NamedProcess {
					name: Str::from(started.name),
					generation: started.generation,
				},
			})
		}
	}
}

fn map_event(event: ExecEvent) -> Result<RunEvent, Fault> {
	match event {
		ExecEvent::Output(frame) => {
			let channel = match EnvOutputChannel::try_from(frame.channel) {
				Ok(EnvOutputChannel::Stdout) => OutputChannel::Stdout,
				Ok(EnvOutputChannel::Stderr) => OutputChannel::Stderr,
				Ok(EnvOutputChannel::Pty) => OutputChannel::Pty,
				Ok(EnvOutputChannel::Unspecified) | Err(_) => {
					return Err(protocol_fault(
						"next_event",
						fmts!("invalid output channel {}", frame.channel),
					));
				},
			};
			Ok(RunEvent::Output(Update {
				channel,
				data: CowBytes::owned(frame.data),
				sequence: frame.sequence,
			}))
		},
		ExecEvent::Exit(event) => {
			let status = event
				.status
				.ok_or_else(|| protocol_fault("next_event", "terminal event omitted status"))?;
			let outcome = match EnvExecOutcome::try_from(status.outcome) {
				Ok(EnvExecOutcome::Exited) => ExecOutcome::Exited,
				Ok(EnvExecOutcome::Failed) => ExecOutcome::Failed,
				Ok(EnvExecOutcome::Timeout) => ExecOutcome::Timeout,
				Ok(EnvExecOutcome::Cancelled) => ExecOutcome::Cancelled,
				Ok(EnvExecOutcome::Denied) => ExecOutcome::Denied,
				Ok(EnvExecOutcome::Unspecified) | Err(_) => {
					return Err(protocol_fault(
						"next_event",
						fmts!("invalid execution outcome {}", status.outcome),
					));
				},
			};
			let signal = (!status.signal.is_empty()).then(|| Str::from(status.signal));
			let spilled_output = status.spilled_output.map(|blob| BlobRef {
				hash: Str::from(hex::encode(&blob.hash).into_string()),
				media_type: Str::from(blob.mime),
				byte_len: blob.size,
			});
			Ok(RunEvent::Exit(ExecStatus {
				outcome,
				exit_code: status.exit_code,
				signal,
				wall_clock_ms: status.wall_clock_ms,
				spilled_output,
				aborted: status.aborted,
				effects_unknown: false,
			}))
		},
	}
}

fn resource_fault(operation: &'static str, error: ExecError) -> Fault {
	protocol_fault(operation, fmts!("{error}"))
}

fn protocol_fault(operation: &'static str, message: impl Into<Str>) -> Fault {
	Fault::Resource { operation: Str::new_static(operation), message: message.into() }
}
