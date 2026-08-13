#![cfg(unix)]

use std::{
	path::{Path, PathBuf},
	sync::Arc,
	time::Duration,
};

use bytes::Bytes;
use nix::{sys::stat::Mode, unistd::mkfifo};
use omp_agent::{
	Agent, AgentEvent, AgentSnapshot, AgentState, EventSubscription, Journal, TurnId, TurnInput,
	TurnOptions, WorkspaceInput,
};
use omp_app::envd::{server::EnvServer, worker::ToolWorkerConfig};
use omp_core::Str;
use omp_e2e::support::{
	Gate, ScriptedStep, ScriptedTurn, ScriptedTurnClient, omp_binary, outcome_event,
	tool_call_item, user_item,
};
use omp_env::{BlobDownloadEvent, EnvClient, ProcessAttachmentEvent};
use omp_proto::{
	SCHEMA_REV,
	blob::v1::GetRequest,
	env::v1::{
		AttachOutput, ClientHello, ProcessSpec, ProcessState, RestartPolicy, RestartSpec, Script,
		StartProcess,
	},
	inference::v1::{self as inference, StopReason},
	thread::v1::{self as thread, Revision},
};
use omp_storage::transcript::{Entry, Header, Kind, SessionId};
use omp_tool::{
	ArtifactLifetime, ExpectedArtifact, JobOwner, JobRef, Outcome as ToolOutcome, PromptCaps,
	Registry, ToolIdentity,
};
use serde::Deserialize;
use serde_json::Value as JsonValue;
use tempfile::TempDir;

const LIMIT: Duration = Duration::from_secs(15);
const SETTLEMENT_MIME: &str = "application/vnd.omp.process-settlement+json";

struct RealEnv {
	client: EnvClient,
	server: Arc<EnvServer>,
	root: TempDir,
	_state: TempDir,
	tasks: Vec<tokio::task::JoinHandle<()>>,
}

impl RealEnv {
	async fn spawn() -> Self {
		let root = tempfile::tempdir().expect("workspace scratch directory");
		let state = tempfile::tempdir().expect("environment state directory");
		let server = Arc::new(
			EnvServer::open_local(
				root.path(),
				state.path(),
				Registry::new(),
				ToolWorkerConfig::new(omp_binary().expect("Cargo-built e2e host")),
			)
			.await
			.expect("real local environment"),
		);
		let (client, task) = connect_env(&server, "p3-primary").await;
		Self { client, server, root, _state: state, tasks: vec![task] }
	}

	async fn reconnect(&mut self, name: &str) -> EnvClient {
		let (client, task) = connect_env(&self.server, name).await;
		self.tasks.push(task);
		client
	}

	fn registry(&self) -> Arc<Registry> {
		self.server.registry()
	}

	fn cwd_uri(&self) -> String {
		let mut uri = String::from("file://");
		uri.push_str(self.root.path().to_str().expect("scratch path is UTF-8"));
		if !uri.ends_with('/') {
			uri.push('/');
		}
		uri
	}

	async fn read_blob(&self, hash: Bytes) -> Vec<u8> {
		let mut download = self
			.client
			.blob_get(GetRequest { hash, ..Default::default() })
			.await
			.expect("settlement artifact is addressable");
		let mut bytes = Vec::new();
		loop {
			match tokio::time::timeout(LIMIT, download.next_event())
				.await
				.expect("blob download timeout")
				.expect("blob download event")
				.expect("blob download did not close early")
			{
				BlobDownloadEvent::Chunk(chunk) => bytes.extend_from_slice(&chunk.data),
				BlobDownloadEvent::Complete(_) => return bytes,
			}
		}
	}
}

impl Drop for RealEnv {
	fn drop(&mut self) {
		for task in &self.tasks {
			task.abort();
		}
	}
}

async fn connect_env(
	server: &Arc<EnvServer>,
	name: &str,
) -> (EnvClient, tokio::task::JoinHandle<()>) {
	let (client, transport) = EnvClient::in_process(64);
	let host = Arc::clone(server);
	let task = tokio::spawn(async move { host.serve_in_process(transport).await });
	client
		.hello(ClientHello {
			client: name.to_owned(),
			schema_rev: SCHEMA_REV,
			..Default::default()
		})
		.await
		.expect("environment hello");
	(client, task)
}

fn journal(path: &Path, root: &Path) -> Journal {
	Journal::create(
		path,
		&Header {
			v: 4,
			id: SessionId(Str::new_static("p3-detached-jobs")),
			created: 1,
			cwd: root.to_owned(),
		},
	)
	.expect("create agent journal")
}

fn state(root: &Path, registry: Arc<Registry>) -> AgentState {
	let mut turn = TurnOptions::default();
	turn.context_id = Some(Str::new_static("p3-context"));
	AgentState::new(AgentSnapshot::new(
		turn,
		WorkspaceInput::new(root, Arc::from([])),
		registry,
	))
}

fn revision(head: u64) -> Option<Revision> {
	Some(Revision { head, token: Bytes::from(head.to_le_bytes().to_vec()) })
}

fn scripted(outcomes: impl IntoIterator<Item = inference::Outcome>) -> ScriptedTurnClient {
	ScriptedTurnClient::new(
		outcomes
			.into_iter()
			.map(|outcome| ScriptedTurn::events([outcome_event(outcome)])),
	)
}

fn end_outcome(head: u64) -> inference::Outcome {
	inference::Outcome {
		stop: StopReason::StopEndTurn as i32,
		revision: revision(head),
		provider: "p3-script".to_owned(),
		model: "deterministic".to_owned(),
		..Default::default()
	}
}

fn shell_call(name: &str, command: String) -> thread::Item {
	tool_call_item(
		2,
		"shell-detached",
		&ToolIdentity {
			name: Str::new_static("shell"),
			rev: omp_tool::Rev { family: Str::default(), n: 1 },
		},
		Bytes::from(
			serde_json::to_vec(&serde_json::json!({
				"command": command,
				"detach": true,
				"name": name,
			}))
			.expect("shell args serialize"),
		),
	)
}

fn tool_use_outcome(call: thread::Item, head: u64) -> inference::Outcome {
	inference::Outcome {
		output: vec![call],
		stop: StopReason::StopToolUse as i32,
		revision: revision(head),
		provider: "p3-script".to_owned(),
		model: "deterministic".to_owned(),
		..Default::default()
	}
}

async fn wait_board_empty(board: &omp_agent::JobBoard) {
	tokio::time::timeout(LIMIT, async {
		while !board.is_empty() {
			tokio::task::yield_now().await;
		}
	})
	.await
	.expect("detached settlement watcher timeout");
}

async fn release_fifo(path: PathBuf) {
	tokio::time::timeout(
		LIMIT,
		tokio::task::spawn_blocking(move || std::fs::write(path, b"go\n")),
	)
	.await
	.expect("FIFO writer timeout")
	.expect("FIFO writer task")
	.expect("release detached process");
}

async fn one_job_event(
	events: &EventSubscription,
	job_id: &str,
	registered: bool,
) -> Arc<AgentEvent> {
	loop {
		let event = tokio::time::timeout(LIMIT, events.recv())
			.await
			.expect("agent job event timeout")
			.expect("agent event bus closed");
		let matches = match event.as_ref() {
			AgentEvent::JobRegistered { job_id: actual } if registered => actual == job_id,
			AgentEvent::JobSettled { job_id: actual } if !registered => actual == job_id,
			_ => false,
		};
		if matches {
			return event;
		}
	}
}

fn delta(input: &TurnInput) -> &inference::ThreadDelta {
	match input {
		TurnInput::Delta(_, delta) => delta,
		TurnInput::Full(_) => panic!("expected incremental ThreadDelta"),
	}
}

fn tool_result(items: &[thread::Item], call_id: &str) -> &thread::ToolResult {
	let mut matching = items.iter().filter_map(|item| match item.kind.as_ref() {
		Some(thread::item::Kind::ToolResult(result)) if result.call_id == call_id => Some(result),
		_ => None,
	});
	let result = matching.next().expect("canonical detached ToolResult");
	assert!(matching.next().is_none(), "detached ToolResult duplicated");
	result
}

fn detached_ref(result: &thread::ToolResult) -> JobRef {
	let details = result.details.as_ref().expect("detached result retains exact structured truth");
	let json = proto_json(details);
	match serde_json::from_value::<ToolOutcome<JsonValue, JsonValue>>(json)
		.expect("detached result details decode")
	{
		ToolOutcome::Detached(job) => job,
		ToolOutcome::Done { .. } => panic!("detached result lowered as synchronous outcome"),
	}
}

fn proto_json(value: &inference::Value) -> JsonValue {
	match value.kind.as_ref().expect("proto JSON value kind") {
		inference::value::Kind::Null(_) => JsonValue::Null,
		inference::value::Kind::Bool(value) => JsonValue::Bool(*value),
		inference::value::Kind::Int(value) => JsonValue::from(*value),
		inference::value::Kind::Uint(value) => JsonValue::from(*value),
		inference::value::Kind::Double(value) => serde_json::Number::from_f64(*value)
			.map(JsonValue::Number)
			.expect("finite JSON number"),
		inference::value::Kind::String(value) => JsonValue::String(value.clone()),
		inference::value::Kind::List(values) => {
			JsonValue::Array(values.values.iter().map(proto_json).collect())
		},
		inference::value::Kind::Map(values) => JsonValue::Object(
			values
				.fields
				.iter()
				.map(|(key, value)| (key.clone(), proto_json(value)))
				.collect(),
		),
	}
}

fn settlement_item(items: &[thread::Item], job_id: &str) -> &thread::Item {
	let mut matching = items.iter().filter(|item| settlement_parts(item, job_id).is_some());
	let item = matching.next().expect("ThreadDelta carries detached settlement");
	assert!(matching.next().is_none(), "ThreadDelta duplicated detached settlement");
	item
}

fn settlement_parts<'a>(
	item: &'a thread::Item,
	job_id: &str,
) -> Option<(&'a str, &'a thread::Blob)> {
	let Some(thread::item::Kind::Message(message)) = item.kind.as_ref() else {
		return None;
	};
	if message.role != thread::Role::System as i32 {
		return None;
	}
	let text = message.parts.iter().find_map(|part| match part.kind.as_ref() {
		Some(thread::part::Kind::Text(text)) if text.contains(job_id) => Some(text.as_str()),
		_ => None,
	})?;
	let blob = message.parts.iter().find_map(|part| match part.kind.as_ref() {
		Some(thread::part::Kind::Blob(blob)) => Some(blob),
		_ => None,
	})?;
	Some((text, blob))
}

#[derive(Debug, Deserialize)]
struct SettlementArtifact {
	job_id: String,
	owner: ArtifactOwner,
	expected_artifact: ArtifactExpectation,
	output: Vec<ArtifactOutput>,
	state: ArtifactState,
}

#[derive(Debug, Deserialize)]
struct ArtifactOwner {
	name: String,
	generation: u64,
}

#[derive(Debug, Deserialize)]
struct ArtifactExpectation {
	description: String,
	media_type: Option<String>,
	lifetime: String,
}

#[derive(Debug, Deserialize)]
struct ArtifactOutput {
	sequence: u64,
	channel: i32,
	data: Vec<u8>,
}

#[derive(Debug, Deserialize)]
struct ArtifactState {
	state: i32,
	status: Option<ArtifactStatus>,
}

#[derive(Debug, Deserialize)]
struct ArtifactStatus {
	outcome: i32,
	exit_code: Option<i32>,
	aborted: bool,
}

async fn assert_artifact(
	env: &RealEnv,
	item: &thread::Item,
	job: &JobRef,
	expected_output: &[u8],
) {
	let (text, blob) = settlement_parts(item, job.id.as_str()).expect("canonical settlement parts");
	assert!(text.contains("settled"));
	assert_eq!(blob.mime, SETTLEMENT_MIME);
	assert!(blob.inline.is_empty(), "settlement must remain blob-authoritative");
	let raw = env.read_blob(blob.hash.clone()).await;
	assert_eq!(blob.size, u64::try_from(raw.len()).expect("artifact length fits u64"));
	let artifact: SettlementArtifact =
		serde_json::from_slice(&raw).expect("structured process-settlement artifact");
	assert_eq!(artifact.job_id, job.id.as_str());
	let JobOwner::NamedProcess { name, generation } = &job.owner;
	assert_eq!(artifact.owner.name, name.as_str());
	assert_eq!(artifact.owner.generation, *generation);
	assert_eq!(artifact.expected_artifact.description, job.artifact.description.as_str());
	assert_eq!(
		artifact.expected_artifact.media_type.as_deref(),
		job.artifact.media_type.as_deref(),
	);
	assert_eq!(artifact.expected_artifact.lifetime, "session");
	assert_eq!(artifact.state.state, ProcessState::Exited as i32);
	let status = artifact.state.status.expect("terminal process status");
	assert_eq!(status.exit_code, Some(0));
	assert!(!status.aborted);
	assert_ne!(status.outcome, 0);
	assert!(
		artifact.output.windows(2).all(|pair| pair[0].sequence < pair[1].sequence),
		"process output sequences must be strictly ordered",
	);
	assert!(artifact.output.iter().all(|frame| frame.channel != 0));
	let ordered: Vec<u8> = artifact.output.into_iter().flat_map(|frame| frame.data).collect();
	assert_eq!(ordered, expected_output, "artifact bytes differ from ordered process output");
}

fn job_event_counts(journal: &Journal, job_id: &str) -> (usize, usize) {
	let log = journal.load().expect("load durable transcript");
	let mut registered = 0;
	let mut settled = 0;
	for index in 0..u64::try_from(log.len()).expect("log length fits u64") {
		let Some(Entry::Ok(event)) = log.get(index) else {
			continue;
		};
		match &event.kind {
			Kind::JobRegistered(event) if event.job.id == job_id => registered += 1,
			Kind::JobSettled(event) if event.job_id == job_id => settled += 1,
			_ => {},
		}
	}
	(registered, settled)
}

async fn wait_terminal(client: &EnvClient, name: &str, generation: u64) {
	let mut attachment = client
		.attach_output(AttachOutput { name: name.to_owned(), after_sequence: 0, props: None })
		.await
		.expect("attach to named process");
	loop {
		let event = tokio::time::timeout(LIMIT, attachment.next_event())
			.await
			.expect("process terminal timeout")
			.expect("process attachment event")
			.expect("process attachment did not close early");
		match event {
			ProcessAttachmentEvent::Attached(attached) => {
				assert_eq!(attached.name, name);
				assert_eq!(attached.generation, generation);
			},
			ProcessAttachmentEvent::Output(output) => {
				assert_eq!(output.name, name);
				assert_eq!(output.generation, generation);
			},
			ProcessAttachmentEvent::State(state) => {
				let process = state.process.expect("process state info");
				assert_eq!(process.name, name);
				assert_eq!(process.generation, generation);
				if process.status.is_some() {
					return;
				}
			},
			ProcessAttachmentEvent::StreamError(error) => {
				panic!("process attachment failed: {error:?}")
			},
		}
	}
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn detached_shell_settles_once_after_reconnect_with_exact_artifact() {
	let mut env = RealEnv::spawn().await;
	let journal_path = env.root.path().join("agent.jsonl");
	let fifo = env.root.path().join("release.fifo");
	mkfifo(&fifo, Mode::S_IRUSR | Mode::S_IWUSR).expect("create deterministic process gate");
	let process_name = "p3-detached";
	let command = format!(
		"printf 'output-1\\noutput-2\\n'; read _ < '{}'; printf 'output-3\\n'",
		fifo.display(),
	);
	let initial_client = scripted([
		tool_use_outcome(shell_call(process_name, command), 3),
		end_outcome(4),
	]);
	let initial_capture = initial_client.clone();
	let mut agent = Agent::new(
		initial_client,
		env.client.clone(),
		state(env.root.path(), env.registry()),
		journal(&journal_path, env.root.path()),
		PromptCaps::default(),
	);
	let events = agent.events().subscribe_lossless();
	agent
		.submit([user_item("start detached shell")], TurnId::new("p3-start"))
		.await
		.expect("detached tool turn");
	let captures = initial_capture.captures();
	assert_eq!(captures.len(), 2);
	let result = tool_result(&delta(&captures[1].input).append, "shell-detached");
	assert!(!result.is_error);
	let job = detached_ref(result);
	assert_eq!(job.id, format!("{process_name}#1").as_str());
	assert_eq!(
		job.owner,
		JobOwner::NamedProcess { name: Str::from(process_name), generation: 1 },
	);
	assert_eq!(job.artifact.description, "named process settlement");
	assert_eq!(job.artifact.media_type.as_deref(), Some(SETTLEMENT_MIME));
	assert_eq!(job.artifact.lifetime, ArtifactLifetime::Session);
	let text = result.parts.iter().find_map(|part| match part.kind.as_ref() {
		Some(thread::part::Kind::Text(text)) => Some(text.as_str()),
		_ => None,
	});
	assert_eq!(
		text,
		Some(format!(
			"job started; artifact will land at job://{} ({})",
			job.id, job.artifact.description
		)
		.as_str()),
	);
	let _registered = one_job_event(&events, job.id.as_str(), true).await;
	assert_eq!(job_event_counts(agent.journal(), job.id.as_str()), (1, 0));
	assert_eq!(agent.jobs().len(), 1);

	drop(agent);
	let reconnected = env.reconnect("p3-reconnected").await;
	let settlement_gate = Gate::default();
	let next_client = ScriptedTurnClient::new([
		ScriptedTurn::steps([
			ScriptedStep::Wait(settlement_gate.clone()),
			ScriptedStep::from(outcome_event(end_outcome(5))),
		]),
		ScriptedTurn::events([outcome_event(end_outcome(6))]),
	]);
	let next_capture = next_client.clone();
	let reopened_journal = Journal::open(&journal_path).expect("reopen pending detached journal");
	let mut reopened = Agent::new(
		next_client,
		reconnected,
		state(env.root.path(), env.registry()),
		reopened_journal,
		PromptCaps::default(),
	);
	let settled_events = reopened.events().subscribe_lossless();
	let board = Arc::clone(reopened.jobs());
	let release = tokio::spawn({
		let settlement_gate = settlement_gate.clone();
		async move {
			release_fifo(fifo).await;
			wait_board_empty(&board).await;
			settlement_gate.release();
		}
	});
	reopened
		.submit([user_item("observe settlement")], TurnId::new("p3-settlement"))
		.await
		.expect("turn after detached settlement");
	release.await.expect("settlement release task");
	let _settled = one_job_event(&settled_events, job.id.as_str(), false).await;
	let next = next_capture.captures();
	assert_eq!(next.len(), 2);
	let settlement = settlement_item(&delta(&next[1].input).append, job.id.as_str());
	assert_eq!(
		delta(&next[1].input)
			.append
			.iter()
			.filter(|item| settlement_parts(item, job.id.as_str()).is_some())
			.count(),
		1,
	);
	assert_artifact(&env, settlement, &job, b"output-1\noutput-2\noutput-3\n").await;
	assert_eq!(job_event_counts(reopened.journal(), job.id.as_str()), (1, 1));
	assert!(reopened.jobs().is_empty());

	// Register a job only after the real named process has already exited. Reopening
	// must reconstruct its watcher from durable truth and consume retained output.
	let early_name = "p3-already-exited";
	let started = env
		.client
		.start_process(StartProcess {
			name: early_name.to_owned(),
			spec: Some(ProcessSpec {
				source: Some(Script {
					text: "printf 'early-1\\nearly-2\\n'".to_owned(),
					..Default::default()
				}),
				cwd_uri: env.cwd_uri(),
				restart: Some(RestartSpec {
					policy: RestartPolicy::Never as i32,
					..Default::default()
				}),
				..Default::default()
			}),
			..Default::default()
		})
		.await
		.expect("start early-exit named process");
	wait_terminal(&env.client, early_name, started.generation).await;
	let early_job = JobRef {
		id: Str::from(format!("{early_name}#{}", started.generation)),
		owner: JobOwner::NamedProcess {
			name: Str::from(early_name),
			generation: started.generation,
		},
		artifact: ExpectedArtifact {
			description: Str::new_static("expected PNG render"),
			media_type: Some(Str::new_static("image/png")),
			lifetime: ArtifactLifetime::Session,
		},
	};
	drop(reopened);
	let mut durable = Journal::open(&journal_path).expect("open journal for durable registration");
	durable
		.register_job(10, early_job.clone())
		.expect("register already-exited job");
	drop(durable);
	let early_gate = Gate::default();
	let final_client = ScriptedTurnClient::new([
		ScriptedTurn::steps([
			ScriptedStep::Wait(early_gate.clone()),
			ScriptedStep::from(outcome_event(end_outcome(7))),
		]),
		ScriptedTurn::events([outcome_event(end_outcome(8))]),
	]);
	let final_capture = final_client.clone();
	let mut final_agent = Agent::new(
		final_client,
		env.reconnect("p3-final-reopen").await,
		state(env.root.path(), env.registry()),
		Journal::open(&journal_path).expect("reopen already-exited job"),
		PromptCaps::default(),
	);
	let final_events = final_agent.events().subscribe_lossless();
	let final_board = Arc::clone(final_agent.jobs());
	let early_release = tokio::spawn({
		let early_gate = early_gate.clone();
		async move {
			wait_board_empty(&final_board).await;
			early_gate.release();
		}
	});
	final_agent
		.submit([user_item("observe retained early exit")], TurnId::new("p3-early-exit"))
		.await
		.expect("turn after already-exited attachment");
	early_release.await.expect("early-exit release task");
	let _early_settled = one_job_event(&final_events, early_job.id.as_str(), false).await;
	let final_turns = final_capture.captures();
	assert_eq!(final_turns.len(), 2);
	let early_settlement =
		settlement_item(&delta(&final_turns[1].input).append, early_job.id.as_str());
	assert_artifact(&env, early_settlement, &early_job, b"early-1\nearly-2\n").await;
	assert_eq!(job_event_counts(final_agent.journal(), early_job.id.as_str()), (1, 1));
	assert_eq!(job_event_counts(final_agent.journal(), job.id.as_str()), (1, 1));
	assert!(final_agent.jobs().is_empty());
}
