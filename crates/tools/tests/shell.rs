//! Behavioral contracts for the persistent native `shell@1` executor.

use std::{collections::VecDeque, sync::Arc};

use bytes::Bytes;
use futures::{FutureExt, StreamExt, executor::block_on, pin_mut};
use omp_core::{CowBytes, Str};
use omp_proto::inference::v1::invoke_input;
use omp_tool::{
	Abort, ArtifactLifetime, ErasedEv, ErasedOutcome, IncomingParams, Interrupt, JobOwner, Part,
	PromptCaps, Registry, Tool, ToolIdentity, Verdict,
};
use omp_tools::shell::{
	self, DetachRequest, DetachedJob, ExecOutcome, ExecStatus, Fault, OutputChannel, Payload,
	RunEvent, RunRequest, Session, ShellExec, ShellRun, Update,
};
use parking_lot::Mutex;

#[derive(Default)]
struct State {
	opens:     usize,
	runs:      Vec<(Bytes, RunRequest)>,
	detaches:  Vec<DetachRequest>,
	cancels:   usize,
	cwd:       String,
	env_value: String,
}

#[derive(Clone, Default)]
struct FakeExec {
	state: Arc<Mutex<State>>,
}

struct FakeRun {
	events:    VecDeque<RunEvent>,
	cancelled: Arc<Mutex<Option<RunEvent>>>,
	state:     Arc<Mutex<State>>,
}

impl ShellRun for FakeRun {
	fn next_event(&mut self) -> impl Future<Output = Result<Option<RunEvent>, Fault>> + Send + '_ {
		async move {
			if let Some(event) = self.events.pop_front() {
				return Ok(Some(event));
			}
			if let Some(event) = self.cancelled.lock().take() {
				return Ok(Some(event));
			}
			futures::future::pending().await
		}
	}

	fn cancel(&self) -> impl Future<Output = Result<(), Fault>> + Send + '_ {
		async move {
			self.state.lock().cancels += 1;
			*self.cancelled.lock() = Some(RunEvent::Exit(status(ExecOutcome::Cancelled)));
			Ok(())
		}
	}
}

impl ShellExec for FakeExec {
	type Run = FakeRun;

	fn open_session(&self) -> impl Future<Output = Result<Session, Fault>> + Send + '_ {
		async move {
			let mut state = self.state.lock();
			state.opens += 1;
			if state.cwd.is_empty() {
				state.cwd = "/workspace".into();
			}
			Ok(Session { id: Bytes::from_static(b"session-41") })
		}
	}

	fn run<'a>(
		&'a self,
		session: &'a Session,
		request: RunRequest,
	) -> impl Future<Output = Result<Self::Run, Fault>> + Send + 'a {
		async move {
			let mut events = VecDeque::new();
			let command = request.command.clone();
			self.state.lock().runs.push((session.id.clone(), request));
			events.push_back(RunEvent::Started { exec_id: Bytes::from(format!("exec-{command}")) });
			match command.as_str() {
				"set-state" => {
					let mut state = self.state.lock();
					state.cwd = "/workspace/subdir".into();
					state.env_value = "preserved".into();
					events.push_back(RunEvent::Exit(status(ExecOutcome::Exited)));
				},
				"show-state" => {
					let state = self.state.lock();
					let text = format!("{}\n{}\n", state.cwd, state.env_value).into_bytes();
					drop(state);
					events.push_back(RunEvent::Output(Update {
						channel:  OutputChannel::Stdout,
						data:     text.into(),
						sequence: 1,
					}));
					events.push_back(RunEvent::Exit(status(ExecOutcome::Exited)));
				},
				"ordered" => {
					for (channel, data, sequence) in [
						(OutputChannel::Stdout, CowBytes::from_static(b"one"), 4),
						(OutputChannel::Stderr, CowBytes::from_static(b"two"), 5),
						(OutputChannel::Stdout, CowBytes::from_static(b"three"), 6),
					] {
						events.push_back(RunEvent::Output(Update { channel, data, sequence }));
					}
					events.push_back(RunEvent::Exit(status(ExecOutcome::Exited)));
				},
				"timeout" => events.push_back(RunEvent::Exit(status(ExecOutcome::Timeout))),
				"overflow" => {
					events.push_back(RunEvent::Output(Update {
						channel:  OutputChannel::Stdout,
						data:     CowBytes::owned(Bytes::from(vec![b'x'; 16])),
						sequence: 1,
					}));
					let mut terminal = status(ExecOutcome::Exited);
					terminal.spilled_output = Some(omp_tool::BlobRef {
						hash:       Str::from("sha256:overflow"),
						media_type: Str::from("application/octet-stream"),
						byte_len:   4096,
					});
					events.push_back(RunEvent::Exit(terminal));
				},
				"wait" => {},
				"effects-unknown" => {
					let mut terminal = status(ExecOutcome::Cancelled);
					terminal.effects_unknown = true;
					events.push_back(RunEvent::Exit(terminal));
				},
				_ => events.push_back(RunEvent::Exit(status(ExecOutcome::Exited))),
			}
			Ok(FakeRun {
				events,
				cancelled: Arc::new(Mutex::new(None)),
				state: Arc::clone(&self.state),
			})
		}
	}

	fn detach(
		&self,
		request: DetachRequest,
	) -> impl Future<Output = Result<DetachedJob, Fault>> + Send + '_ {
		async move {
			let pending = request.command == "pending-detach";
			let id = Str::from(format!("process:{}:1", request.name));
			let owner_name = request.name.clone();
			self.state.lock().detaches.push(request);
			if pending {
				futures::future::pending().await
			}
			Ok(DetachedJob {
				id,
				owner: JobOwner::NamedProcess { name: owner_name, generation: 1 },
			})
		}
	}
}

fn status(outcome: ExecOutcome) -> ExecStatus {
	ExecStatus {
		outcome,
		exit_code: (outcome == ExecOutcome::Exited).then_some(0),
		signal: None,
		wall_clock_ms: 7,
		spilled_output: None,
		aborted: matches!(outcome, ExecOutcome::Timeout | ExecOutcome::Cancelled),
		effects_unknown: false,
	}
}

fn registry(exec: FakeExec, transcript_limit: usize) -> Registry {
	let mut registry = Registry::new();
	registry
		.register(shell::shell(exec).with_transcript_limit(transcript_limit))
		.expect("shell schema and revision register");
	registry
}

fn call(registry: &Registry, raw: &str) -> Vec<ErasedEv> {
	let (feed, params) = IncomingParams::channel();
	feed.args_committed(Str::from(raw)).unwrap();
	let stream = registry.invoke("shell", params).unwrap();
	block_on(stream.map(|event| event.unwrap()).collect())
}

fn payload(events: &[ErasedEv]) -> Payload {
	let ErasedEv::Done(ErasedOutcome::Done { verdict, .. }) = events.last().unwrap() else {
		panic!("foreground call must end in a verdict")
	};
	let verdict: Verdict<Payload, Fault> = serde_json::from_slice(verdict).unwrap();
	let Verdict::Ok(payload) = verdict else {
		panic!("expected successful payload")
	};
	payload
}

#[test]
fn execution_waits_for_the_explicit_commit_gate() {
	let exec = FakeExec::default();
	let registry = registry(exec.clone(), 1024);
	let (feed, params) = IncomingParams::channel();
	feed
		.arg_text(Str::from(r#"{"command":"ordered"}"#))
		.unwrap();
	let stream = registry.invoke("shell", params).unwrap();
	pin_mut!(stream);
	assert!(stream.next().now_or_never().is_none());
	assert_eq!(exec.state.lock().opens, 0);
	assert!(exec.state.lock().runs.is_empty());

	feed
		.args_committed(Str::from(r#"{"command":"ordered"}"#))
		.unwrap();
	let events = block_on(stream.map(|event| event.unwrap()).collect::<Vec<_>>());
	assert_eq!(payload(&events).status.outcome, ExecOutcome::Exited);
	assert_eq!(exec.state.lock().runs.len(), 1);
}

#[test]
fn one_session_is_reused_with_its_cwd_and_environment_state() {
	let exec = FakeExec::default();
	let registry = registry(exec.clone(), 1024);
	assert_eq!(
		payload(&call(&registry, r#"{"command":"set-state"}"#)).session_id,
		Bytes::from_static(b"session-41"),
	);
	let shown = payload(&call(&registry, r#"{"command":"show-state"}"#));
	assert_eq!(shown.session_id, Bytes::from_static(b"session-41"));
	assert_eq!(shown.transcript[0].data, b"/workspace/subdir\npreserved\n");
	let state = exec.state.lock();
	assert_eq!(state.opens, 1);
	assert!(
		state
			.runs
			.iter()
			.all(|run| run.0 == Bytes::from_static(b"session-41"))
	);
}

#[test]
fn live_updates_and_durable_transcript_preserve_host_order() {
	let exec = FakeExec::default();
	let events = call(&registry(exec, 1024), r#"{"command":"ordered"}"#);
	let updates = events
		.iter()
		.filter_map(|event| match event {
			ErasedEv::Update(json) => Some(serde_json::from_slice::<Update>(json).unwrap()),
			ErasedEv::Done(_) => None,
		})
		.collect::<Vec<_>>();
	assert_eq!(
		updates
			.iter()
			.map(|update| update.sequence)
			.collect::<Vec<_>>(),
		[4, 5, 6]
	);
	assert_eq!(
		payload(&events)
			.transcript
			.iter()
			.map(|frame| frame.sequence)
			.collect::<Vec<_>>(),
		[4, 5, 6]
	);
}

#[test]
fn timeout_status_is_not_rewritten_as_a_generic_failure() {
	let exec = FakeExec::default();
	let events = call(&registry(exec.clone(), 1024), r#"{"command":"timeout","timeout_ms":23}"#);
	let result = payload(&events);
	assert_eq!(result.status.outcome, ExecOutcome::Timeout);
	assert!(result.status.aborted);
	assert_eq!(exec.state.lock().runs[0].1.timeout_ms, Some(23));
}

#[test]
fn transcript_overflow_preserves_the_host_blob_reference() {
	let exec = FakeExec::default();
	let events = call(&registry(exec, 4), r#"{"command":"overflow"}"#);
	let result = payload(&events);
	assert!(result.transcript_truncated);
	assert!(result.transcript.is_empty());
	assert_eq!(result.status.spilled_output.as_ref().unwrap().hash, "sha256:overflow");
	assert!(matches!(events.first(), Some(ErasedEv::Update(_))), "live output is never capped");
}

#[test]
fn detach_returns_a_named_session_lifetime_job_reference() {
	let exec = FakeExec::default();
	let events = call(
		&registry(exec.clone(), 1024),
		r#"{"command":"serve","detach":true,"name":"web","timeout_ms":50}"#,
	);
	let ErasedEv::Done(ErasedOutcome::Detached(job)) = events.last().unwrap() else {
		panic!("detach must return a detached outcome")
	};
	assert_eq!(job.id, "process:web:1");
	assert_eq!(
		job.owner,
		JobOwner::NamedProcess { name: Str::from("web"), generation: 1 }
	);
	assert_eq!(job.artifact.lifetime, ArtifactLifetime::Session);
	assert_eq!(
		job.artifact.media_type.as_deref(),
		Some("application/vnd.omp.process-settlement+json")
	);
	let state = exec.state.lock();
	assert_eq!(state.opens, 0, "detach does not open the foreground session");
	assert_eq!(state.detaches[0].timeout_ms, Some(50));
}


#[test]
fn interrupt_during_detach_reports_effect_uncertainty() {
	let exec = FakeExec::default();
	let registry = registry(exec.clone(), 1024);
	let (feed, params) = IncomingParams::channel();
	feed
		.args_committed(Str::from(
			r#"{"command":"pending-detach","detach":true,"name":"pending"}"#,
		))
		.unwrap();
	let wait_state = Arc::clone(&exec.state);
	let interrupter = std::thread::spawn(move || {
		while wait_state.lock().detaches.is_empty() {
			std::thread::yield_now();
		}
		feed
			.interrupt(Interrupt {
				class: Str::from("immediate"),
				reason: Str::from("stop detach"),
			})
			.unwrap();
	});
	let stream = registry.invoke("shell", params).unwrap();
	let events = block_on(stream.map(|event| event.unwrap()).collect::<Vec<_>>());
	interrupter.join().unwrap();
	let ErasedEv::Done(ErasedOutcome::Done { verdict, .. }) = events.last().unwrap() else {
		panic!("interrupted detach must produce a verdict")
	};
	let verdict: Verdict<Payload, Fault> = serde_json::from_slice(verdict).unwrap();
	assert!(matches!(verdict, Verdict::Aborted(Abort::EffectsUnknown { .. })));
	assert_eq!(exec.state.lock().detaches.len(), 1);
}

#[test]
fn output_update_clones_share_owned_bytes() {
	let update = Update {
		channel: OutputChannel::Stdout,
		data: CowBytes::owned(Bytes::from(vec![1, 2, 3, 4])),
		sequence: 1,
	};
	let cloned = update.clone();
	assert_eq!(update.data.as_ptr(), cloned.data.as_ptr());
}

#[test]
fn shell_updates_map_exactly_to_live_invoke_input_chunks() {
	let tool = shell::shell(FakeExec::default());
	for (source, expected) in [
		(OutputChannel::Stdout, invoke_input::chunk::Channel::Stdout),
		(OutputChannel::Stderr, invoke_input::chunk::Channel::Stderr),
		(OutputChannel::Pty, invoke_input::chunk::Channel::Stdout),
	] {
		let update = Update {
			channel: source,
			data: CowBytes::owned(Bytes::from(vec![7, 8, 9])),
			sequence: 42,
		};
		let source_ptr = update.data.as_ptr();
		let input = tool.invoke_input(&update, "invocation-17").unwrap();
		assert_eq!(input.invocation_id, "invocation-17");
		let Some(invoke_input::Payload::Chunk(chunk)) = input.payload else {
			panic!("shell update must map to an invocation chunk")
		};
		assert_eq!(chunk.channel, expected as i32);
		assert_eq!(chunk.data, Bytes::from_static(&[7, 8, 9]));
		assert_eq!(chunk.data.as_ptr(), source_ptr, "owned output must remain zero-copy");
	}
}
#[test]
fn interrupt_cancels_only_the_run_and_the_next_call_reuses_the_session() {
	let exec = FakeExec::default();
	let registry = registry(exec.clone(), 1024);
	let (feed, params) = IncomingParams::channel();
	feed
		.args_committed(Str::from(r#"{"command":"wait"}"#))
		.unwrap();
	feed
		.interrupt(Interrupt { class: Str::from("immediate"), reason: Str::from("stop now") })
		.unwrap();
	let stream = registry.invoke("shell", params).unwrap();
	let events = block_on(stream.map(|event| event.unwrap()).collect::<Vec<_>>());
	assert_eq!(payload(&events).status.outcome, ExecOutcome::Cancelled);
	assert_eq!(exec.state.lock().cancels, 1);

	let later = call(&registry, r#"{"command":"show-state"}"#);
	assert_eq!(payload(&later).status.outcome, ExecOutcome::Exited);
	let state = exec.state.lock();
	assert_eq!(state.opens, 1);
	assert!(
		state
			.runs
			.iter()
			.all(|run| run.0 == Bytes::from_static(b"session-41"))
	);
}

#[test]
fn malformed_whole_arguments_are_a_structured_args_verdict() {
	let exec = FakeExec::default();
	let events = call(&registry(exec.clone(), 1024), r#"{"command":17}"#);
	let ErasedEv::Done(ErasedOutcome::Done { verdict, .. }) = events.last().unwrap() else {
		panic!("malformed args must produce a verdict")
	};
	let verdict: Verdict<Payload, Fault> = serde_json::from_slice(verdict).unwrap();
	assert!(matches!(verdict, Verdict::Args(_)));
	let state = exec.state.lock();
	assert_eq!(state.opens, 0);
	assert!(state.runs.is_empty());
}

#[test]
fn missing_detach_name_is_a_typed_fault() {
	let exec = FakeExec::default();
	let events = call(&registry(exec, 1024), r#"{"command":"serve","detach":true}"#);
	let ErasedEv::Done(ErasedOutcome::Done { verdict, .. }) = events.last().unwrap() else {
		panic!("invalid detach policy must produce a verdict")
	};
	let verdict: Verdict<Payload, Fault> = serde_json::from_slice(verdict).unwrap();
	assert!(matches!(verdict, Verdict::Fault(Fault::DetachNameRequired)));
}

#[test]
fn effects_unknown_terminal_status_remains_structured_truth() {
	let events = call(&registry(FakeExec::default(), 1024), r#"{"command":"effects-unknown"}"#);
	let result = payload(&events);
	assert_eq!(result.status.outcome, ExecOutcome::Cancelled);
	assert!(result.status.effects_unknown);
	assert!(result.status.aborted);
}

#[test]
fn prompt_projection_retains_status_and_obeys_text_caps() {
	let registry = registry(FakeExec::default(), 1024);
	let events = call(&registry, r#"{"command":"ordered"}"#);
	let ErasedEv::Done(ErasedOutcome::Done { verdict, .. }) = events.last().unwrap() else {
		panic!("foreground call must produce a verdict")
	};
	let (name, rev) = registry.live_identity("shell").unwrap();
	let parts = registry
		.prompt(&ToolIdentity { name: name.clone(), rev: rev.clone() }, verdict, &PromptCaps {
			maximum_parts:      1,
			maximum_text_bytes: 96,
			media:              false,
		})
		.unwrap()
		.unwrap();
	let [Part::Text { text }] = parts.as_slice() else {
		panic!("shell prompt must be one capped text part")
	};
	assert!(text.len() <= 96);
	assert!(text.contains("status=Exited"));
}
