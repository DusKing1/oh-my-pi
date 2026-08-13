#![cfg(unix)]

use std::{
	collections::BTreeSet,
	fs,
	os::unix::fs::PermissionsExt as _,
	path::{Path, PathBuf},
	sync::Arc,
	time::Duration,
};

use anyhow::{Context as _, Result, anyhow, ensure};
use bytes::Bytes;
use omp_agent::{
	Agent, AgentEvent, AgentSnapshot, AgentState, EventSubscription, Journal, TurnId,
};
use omp_app::envd::docs::{DocumentHost, DocumentLease};
use omp_core::Str;
use omp_e2e::support::{
	DEFAULT_TIMEOUT, DocServerTask, EnvHarness, Gate, Scratch, ScriptedStep, ScriptedTurn,
	ScriptedTurnClient, accepted_event, outcome_event, tool_call_item, turn_event, user_item, within,
};
use omp_env::EnvClient;
use omp_proto::{
	document::v1 as document, inference::v1 as inference, thread::v1 as thread,
};
use omp_storage::transcript::{Header, SessionId};
use omp_tool::{PromptCaps, Registry, Rev, ToolIdentity, Verdict};
use omp_tools::edit::{self, FormatPolicy};
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

const TEST_TIMEOUT: Duration = Duration::from_secs(20);
const STORM_COUNT: usize = 100;
const PINNED_READERS: usize = 4;
const PINNED_READS: usize = 25;

#[derive(Debug, Deserialize)]
struct LspRecord {
	kind:    String,
	#[serde(default)]
	uri:     String,
	#[serde(default)]
	version: Option<i64>,
	#[serde(default)]
	text:    String,
}

#[derive(Debug)]
struct CommitRecord {
	sequence: u64,
	start:    usize,
	end:      usize,
	bytes:    Bytes,
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn p1_real_docserver_rebases_two_agent_loops_and_survives_the_storm() -> Result<()> {
	within("complete P1 race proof", Duration::from_secs(90), async {
		let scratch = Scratch::new().context("create scratch project")?;
		let race_initial = b"fn main() {\n    let value = 1;\n}\n";
		scratch.write("f.rs", race_initial)?;

		let lsp_log = scratch.state().join("lsp.jsonl");
		let lsp_config = install_lsp_fixture(&scratch, &lsp_log)?;
		let docserver = DocServerTask::spawn(
			scratch.project(),
			scratch.socket("docserver.sock"),
			vec![lsp_config],
		)
		.await?;
		let direct_a = docserver.connect().await?;
		let uri = file_uri(&scratch, "f.rs")?;
		let env = EnvHarness::spawn_attached(&scratch, docserver.socket()).await?;
		let env_a_connection = env.connect_client("p1-agent-a").await?;
		let env_b_connection = env.connect_client("p1-agent-b").await?;
		let env_a = env_a_connection.client_clone();
		let env_b = env_b_connection.client_clone();

		let identity = ToolIdentity {
			name: Str::new_static("edit"),
			rev:  Rev { family: Str::new_static("hl"), n: 1 },
		};
		let mut registry = Registry::new();
		registry.register(edit::tool(direct_a.clone(), FormatPolicy::Configured))?;
		let registry = Arc::new(registry);

		let a1_args = edit_args("f.rs", "PUT 2.=2:\n+    let value = 2;")?;
		let a2_args = edit_args("f.rs", "PUT 2.=2:\n+    let value = 3;")?;
		let b_args = edit_args("f.rs", "PUT 1.=1:\n+fn main() { // agent B")?;
		let stale_gate = Gate::default();
		let a_client = ScriptedTurnClient::new([
			tool_turn(&identity, "a-rev2", a1_args, None),
			end_turn(),
			tool_turn(&identity, "a-stale-rev2", a2_args, Some(stale_gate.clone())),
			end_turn(),
		]);
		let b_client = ScriptedTurnClient::new([
			tool_turn(&identity, "b-rev3", b_args, None),
			end_turn(),
		]);
		let (mut agent_a, events_a) = agent(
			a_client.clone(),
			env_a,
			Arc::clone(&registry),
			&scratch,
			"agent-a",
		)?;
		let (mut agent_b, events_b) = agent(
			b_client.clone(),
			env_b,
			Arc::clone(&registry),
			&scratch,
			"agent-b",
		)?;

		agent_a
			.submit([user_item("A: publish revision two")], TurnId::new("p1-a-1"))
			.await?;
		let a1 = next_edit_payload(&events_a, "a-rev2").await?;
		let a1_bytes = b"fn main() {\n    let value = 2;\n}\n";
		ensure!(scratch.read("f.rs")? == a1_bytes, "A revision two was not durable");
		wait_lsp_kind(&lsp_log, &uri, "close", 1).await?;

		let a_task = tokio::spawn(async move {
			let result = agent_a
				.submit(
					[user_item("A: edit again from my revision-two view")],
					TurnId::new("p1-a-2"),
				)
				.await;
			(result, agent_a)
		});
		stale_gate.wait_arrived(TEST_TIMEOUT).await?;
		wait_lsp_kind(&lsp_log, &uri, "open", 2).await?;

		agent_b
			.submit([user_item("B: publish revision three")], TurnId::new("p1-b-1"))
			.await?;
		let b = next_edit_payload(&events_b, "b-rev3").await?;
		let b_bytes = b"fn main() { // agent B\n    let value = 2;\n}\n";
		ensure!(scratch.read("f.rs")? == b_bytes, "B revision three was not durable");

		stale_gate.release();
		let (a_result, agent_a_done) = within("stale A completion", TEST_TIMEOUT, a_task).await??;
		a_result?;
		let a2 = next_edit_payload(&events_a, "a-stale-rev2").await?;
		let race_final = b"fn main() { // agent B\n    let value = 3;\n}\n";
		ensure!(scratch.read("f.rs")? == race_final, "stale A overwrote B or lost its own edit");

		ensure!(a2.rebased, "A's revision-two proposal was not daemon-rebased");
		ensure!(!a1.rebased && !b.rebased, "fresh edits were incorrectly reported as rebased");
		let a1_old = revision_sequence(&a1.old_revision)?;
		let a1_new = revision_sequence(&a1.new_revision)?;
		let b_old = revision_sequence(&b.old_revision)?;
		let b_new = revision_sequence(&b.new_revision)?;
		let a2_old = revision_sequence(&a2.old_revision)?;
		let a2_new = revision_sequence(&a2.new_revision)?;
		ensure!(a1_old < a1_new && a1_new < b_new && b_new < a2_new, "race revisions regressed");
		ensure!(b_old == a1_new && a2_old == a1_new, "A did not cite rev2 while B advanced from rev2");
		ensure!(a_client.remaining() == 0 && b_client.remaining() == 0, "agent loop left scripted turns unconsumed");

		let race_records = lsp_records(&lsp_log)?;
		let published_race: BTreeSet<_> = [a1_bytes.as_slice(), b_bytes.as_slice(), race_final.as_slice()]
			.into_iter()
			.map(|bytes| String::from_utf8(bytes.to_vec()).expect("UTF-8 race fixture"))
			.collect();
		assert_lsp_publication(&race_records, &uri, &published_race, race_final)?;
		ensure!(
			race_records.iter().filter(|record| record.kind == "format" && record.uri == uri).count() >= 3,
			"configured LSP formatter did not run for every race edit"
		);

		storm(&scratch, &docserver, &lsp_log).await?;

		drop(agent_a_done);
		drop(agent_b);
		drop(env_b_connection);
		drop(env_a_connection);
		env.shutdown().await?;
		drop(direct_a);
		docserver.shutdown().await?;
		Ok(())
	})
	.await?
}

fn agent(
	client: ScriptedTurnClient,
	env: EnvClient,
	registry: Arc<Registry>,
	scratch: &Scratch,
	name: &str,
) -> Result<(Agent<ScriptedTurnClient>, EventSubscription)> {
	let journal = Journal::create(
		&scratch.state().join(format!("{name}.jsonl")),
		&Header {
			v: 4,
			id: SessionId(Str::from(name)),
			created: 1,
			cwd: scratch.project().to_owned(),
		},
	)?;
	let mut snapshot = AgentSnapshot::new(Default::default(), Default::default(), registry);
	snapshot.enabled_tools = Arc::from([Str::new_static("edit")]);
	let agent = Agent::new(
		client,
		env,
		AgentState::new(snapshot),
		journal,
		PromptCaps { maximum_parts: 16, maximum_text_bytes: 128 * 1024, media: false },
	);
	let events = agent.events().subscribe_lossless();
	Ok((agent, events))
}

fn edit_args(path: &str, patch: &str) -> Result<Bytes> {
	Ok(Bytes::from(serde_json::to_vec(&serde_json::json!({ "path": path, "patch": patch }))?))
}

fn tool_turn(
	identity: &ToolIdentity,
	call_id: &str,
	args: Bytes,
	gate: Option<Gate>,
) -> ScriptedTurn {
	let start = turn_event(inference::turn_event::Event::PartStart(inference::PartStart {
		index: 0,
		kind: inference::part_start::Kind::ToolCall as i32,
		tool_call_id: call_id.to_owned(),
		tool_name: identity.name.to_string(),
	}));
	let delta = turn_event(inference::turn_event::Event::PartDelta(inference::PartDelta {
		index: 0,
		chunk: args.clone(),
	}));
	let end = turn_event(inference::turn_event::Event::PartEnd(inference::PartEnd {
		index: 0,
		signature: Bytes::new(),
	}));
	let outcome = outcome_event(inference::Outcome {
		output: vec![tool_call_item(1, call_id, identity, args)],
		stop: inference::StopReason::StopToolUse as i32,
		..Default::default()
	});
	let mut steps = vec![
		ScriptedStep::from(accepted_event(false)),
		ScriptedStep::from(start),
		ScriptedStep::from(delta),
	];
	if let Some(gate) = gate {
		steps.push(ScriptedStep::Wait(gate));
	}
	steps.extend([ScriptedStep::from(end), ScriptedStep::from(outcome)]);
	ScriptedTurn::steps(steps)
}

fn end_turn() -> ScriptedTurn {
	ScriptedTurn::events([
		accepted_event(false),
		outcome_event(inference::Outcome {
			stop: inference::StopReason::StopEndTurn as i32,
			..Default::default()
		}),
	])
}

async fn next_edit_payload(events: &EventSubscription, call_id: &str) -> Result<edit::Payload> {
	within("successful edit tool result", TEST_TIMEOUT, async {
		loop {
			let event = events.recv().await?;
			let AgentEvent::ToolFinished { call_id: completed, item } = event.as_ref() else {
				continue;
			};
			if completed.as_str() != call_id {
				continue;
			}
			let Some(thread::item::Kind::ToolResult(result)) = item.kind.as_ref() else {
				return Err(anyhow!("ToolFinished did not carry ToolResult"));
			};
			ensure!(!result.is_error, "{call_id} returned a typed failure instead of committing");
			let details = result.details.as_ref().ok_or_else(|| anyhow!("missing edit verdict"))?;
			let verdict: Verdict<edit::Payload, edit::Fault> =
				serde_json::from_value(proto_json(details).ok_or_else(|| anyhow!("invalid edit verdict"))?)?;
			match verdict {
				Verdict::Ok(payload) => return Ok(payload),
				other => return Err(anyhow!("edit did not commit: {other:?}")),
			}
		}
	})
	.await?
}

fn proto_json(value: &inference::Value) -> Option<serde_json::Value> {
	Some(match value.kind.as_ref()? {
		inference::value::Kind::Null(_) => serde_json::Value::Null,
		inference::value::Kind::Bool(value) => (*value).into(),
		inference::value::Kind::Int(value) => (*value).into(),
		inference::value::Kind::Uint(value) => (*value).into(),
		inference::value::Kind::Double(value) => serde_json::Number::from_f64(*value)?.into(),
		inference::value::Kind::String(value) => value.clone().into(),
		inference::value::Kind::List(values) => serde_json::Value::Array(
			values.values.iter().map(proto_json).collect::<Option<Vec<_>>>()?,
		),
		inference::value::Kind::Map(values) => serde_json::Value::Object(
			values
				.fields
				.iter()
				.map(|(key, value)| Some((key.clone(), proto_json(value)?)))
				.collect::<Option<serde_json::Map<_, _>>>()?,
		),
	})
}

fn revision_sequence(revision: &Str) -> Result<u64> {
	revision
		.as_str()
		.split_once(':')
		.ok_or_else(|| anyhow!("revision identity omitted sequence"))?
		.0
		.parse()
		.context("parse revision sequence")
}

async fn storm(scratch: &Scratch, docserver: &DocServerTask, lsp_log: &Path) -> Result<()> {
	let initial = (0..STORM_COUNT)
		.map(|index| format!("old-{index:03}\n"))
		.collect::<String>()
		.into_bytes();
	scratch.write("storm.rs", &initial)?;
	let uri = file_uri(scratch, "storm.rs")?;
	let host_a = docserver.connect().await?;
	let host_b = docserver.connect().await?;
	let readers = docserver.connect().await?;
	let cancel = CancellationToken::new();

	let mut leases = Vec::with_capacity(STORM_COUNT);
	for index in 0..STORM_COUNT {
		let host = if index % 2 == 0 { &host_a } else { &host_b };
		leases.push(open(host, &uri, &cancel).await?);
	}
	let base_sequence = leases[0]
		.head()
		.revision
		.as_ref()
		.ok_or_else(|| anyhow!("storm base omitted revision"))?
		.sequence;
	ensure!(leases.iter().all(|lease| lease.head().revision.as_ref().is_some_and(|revision| revision.sequence == base_sequence)), "storm writers were not pinned to one base");

	let mut pinned = Vec::with_capacity(PINNED_READERS);
	for _ in 0..PINNED_READERS {
		pinned.push(open(&readers, &uri, &cancel).await?);
	}
	let barrier = Arc::new(tokio::sync::Barrier::new(STORM_COUNT + PINNED_READERS + 1));
	let mut reader_tasks = Vec::new();
	for lease in pinned {
		let host = readers.clone();
		let expected = initial.clone();
		let barrier = Arc::clone(&barrier);
		reader_tasks.push(tokio::spawn(async move {
			barrier.wait().await;
			for _ in 0..PINNED_READS {
				let bytes = read_whole(&host, &lease).await?;
				ensure!(bytes.as_ref() == expected.as_slice(), "pinned reader observed a torn or newer head");
				tokio::task::yield_now().await;
			}
			Ok::<_, anyhow::Error>(lease)
		}));
	}

	let mut commits = Vec::with_capacity(STORM_COUNT);
	for (index, lease) in leases.into_iter().enumerate() {
		let host = if index % 2 == 0 { host_a.clone() } else { host_b.clone() };
		let barrier = Arc::clone(&barrier);
		let start = index * 8;
		let end = start + 7;
		let replacement = Bytes::from(format!("new-{index:03}"));
		commits.push(tokio::spawn(async move {
			barrier.wait().await;
			let mut lease = lease;
			let response = host
				.commit(
					&mut lease,
					Bytes::copy_from_slice(&(10_000_u128 + index as u128).to_be_bytes()),
					document::TextMutation {
						base_revision: None,
						change: Some(document::text_mutation::Change::Edits(document::ByteEdits {
							edits: vec![document::ByteEdit {
								start: start as u64,
								end: end as u64,
								replacement: replacement.clone(),
							}],
						})),
						stale_policy: document::StalePolicy::RebaseNonOverlapping as i32,
						format_policy: document::FormatPolicy::Disabled as i32,
					},
					&CancellationToken::new(),
				)
				.await?;
			match response.outcome {
				Some(document::commit_transaction_response::Outcome::Committed(committed)) => {
					let operation = committed.operations.into_iter().next().ok_or_else(|| anyhow!("committed storm op omitted result"))?;
					let sequence = operation.head.and_then(|head| head.revision).ok_or_else(|| anyhow!("committed storm op omitted revision"))?.sequence;
					Ok::<_, anyhow::Error>(Some(CommitRecord { sequence, start, end, bytes: replacement }))
				},
				Some(document::commit_transaction_response::Outcome::Rejected(rejected)) => {
					ensure!(matches!(
						document::TransactionRejectReason::try_from(rejected.reason),
						Ok(document::TransactionRejectReason::StaleBase | document::TransactionRejectReason::OverlappingChange)
					), "storm rejection was not a typed conflict: {rejected:?}");
					Ok(None)
				},
				Some(document::commit_transaction_response::Outcome::PartiallyCommitted(partial)) => {
					Err(anyhow!("single-operation storm partially committed: {partial:?}"))
				},
				None => Err(anyhow!("storm transaction omitted outcome")),
			}
		}));
	}
	barrier.wait().await;

	let mut committed = Vec::new();
	for task in commits {
		if let Some(record) = within("storm commit", TEST_TIMEOUT, task).await??? {
			committed.push(record);
		}
	}
	for task in reader_tasks {
		let lease = within("pinned reader", TEST_TIMEOUT, task).await???;
		readers.close(lease, &CancellationToken::new()).await?;
	}
	ensure!(!committed.is_empty(), "storm did not commit any operation");
	committed.sort_by_key(|record| record.sequence);
	ensure!(committed.windows(2).all(|pair| pair[0].sequence < pair[1].sequence), "storm revisions were not strictly monotone");
	let mut folded = initial.clone();
	let mut published = BTreeSet::new();
	for record in &committed {
		ensure!(record.end - record.start == record.bytes.len(), "storm replacement changed line width");
		folded.splice(record.start..record.end, record.bytes.iter().copied());
		published.insert(String::from_utf8(folded.clone())?);
	}
	let final_bytes = scratch.read("storm.rs")?;
	ensure!(final_bytes == folded, "final storm bytes differ from the revision-ordered fold of commits");
	let final_lease = open(&host_a, &uri, &CancellationToken::new()).await?;
	let final_head = final_lease.head().revision.as_ref().ok_or_else(|| anyhow!("final storm head omitted revision"))?.sequence;
	ensure!(final_head == committed.last().expect("nonempty commits").sequence, "final head regressed behind last commit");
	ensure!(read_whole(&host_a, &final_lease).await?.as_ref() == folded.as_slice(), "final pinned read disagreed with disk");
	host_a.close(final_lease, &CancellationToken::new()).await?;

	let records = lsp_records(lsp_log)?;
	assert_lsp_publication(&records, &uri, &published, &folded)?;
	let changed = records.iter().filter(|record| record.kind == "change" && record.uri == uri).count();
	ensure!(changed >= committed.len(), "LSP missed published storm heads");
	Ok(())
}

async fn open(host: &DocumentHost, uri: &str, cancel: &CancellationToken) -> Result<DocumentLease> {
	within(
		"open pinned document",
		TEST_TIMEOUT,
		host.open(Str::new(uri), Some(Str::new_static("rust")), cancel),
	)
	.await?
	.context("open pinned document")
}

async fn read_whole(host: &DocumentHost, lease: &DocumentLease) -> Result<Bytes> {
	let response = host
		.read(
			lease,
			document::ReadSelection {
				selection: Some(document::read_selection::Selection::Whole(document::WholeDocument {})),
			},
			&CancellationToken::new(),
		)
		.await?;
	match response.body {
		Some(document::read_document_response::Body::Content(bytes)) => Ok(bytes),
		_ => Err(anyhow!("whole read returned slices or no body")),
	}
}

fn file_uri(scratch: &Scratch, relative: &str) -> Result<String> {
	url::Url::from_file_path(scratch.project().join(relative))
		.map(String::from)
		.map_err(|()| anyhow!("fixture path is not an absolute file URI"))
}

fn install_lsp_fixture(scratch: &Scratch, log: &Path) -> Result<PathBuf> {
	let executable = scratch.state().join("lsp_fixture.py");
	fs::write(&executable, LSP_FIXTURE)?;
	fs::set_permissions(&executable, fs::Permissions::from_mode(0o700))?;
	let config = scratch.state().join("lsp.json");
	fs::write(
		&config,
		serde_json::to_vec(&serde_json::json!({
			"name": "p1-fixture",
			"priority": 100,
			"selector": { "languages": ["rust"], "schemes": ["file"], "path_patterns": ["**/*.rs"] },
			"executable": executable,
			"env": { "OMP_LSP_LOG": log },
			"transport": { "initialize_timeout_ms": 5000, "shutdown_timeout_ms": 1000 }
		}))?,
	)?;
	Ok(config)
}

fn lsp_records(path: &Path) -> Result<Vec<LspRecord>> {
	match fs::read_to_string(path) {
		Ok(text) => text
			.lines()
			.filter(|line| !line.is_empty())
			.map(|line| serde_json::from_str(line).context("decode LSP fixture record"))
			.collect(),
		Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
		Err(error) => Err(error.into()),
	}
}

async fn wait_lsp_kind(path: &Path, uri: &str, kind: &str, minimum: usize) -> Result<()> {
	within("LSP lifecycle attribution", TEST_TIMEOUT, async {
		loop {
			if lsp_records(path)?.iter().filter(|record| record.kind == kind && record.uri == uri).count() >= minimum {
				return Ok::<_, anyhow::Error>(());
			}
			tokio::time::sleep(Duration::from_millis(10)).await;
		}
	})
	.await?
}

fn assert_lsp_publication(
	records: &[LspRecord],
	uri: &str,
	published: &BTreeSet<String>,
	final_bytes: &[u8],
) -> Result<()> {
	let changes: Vec<_> = records
		.iter()
		.filter(|record| record.kind == "change" && record.uri == uri)
		.collect();
	ensure!(!changes.is_empty(), "LSP received no didChange for {uri}");
	ensure!(
		changes.iter().all(|record| published.contains(&record.text)),
		"LSP didChange was attributed to bytes that never became a published head"
	);
	ensure!(
		changes
			.windows(2)
			.all(|pair| pair[0].version.zip(pair[1].version).is_some_and(|(left, right)| left < right)),
		"LSP versions regressed or were omitted"
	);
	ensure!(changes.last().is_some_and(|record| record.text.as_bytes() == final_bytes), "LSP final text desynchronized from published head");
	Ok(())
}

const LSP_FIXTURE: &[u8] = br#"#!/usr/bin/env python3
import json, os, sys

log_path = os.environ["OMP_LSP_LOG"]
documents = {}

def record(kind, uri="", version=None, text=""):
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"kind":kind,"uri":uri,"version":version,"text":text}, separators=(",", ":")) + "\n")

def send(identifier, result):
    payload = json.dumps({"jsonrpc":"2.0","id":identifier,"result":result}, separators=(",", ":")).encode()
    sys.stdout.buffer.write(b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload)
    sys.stdout.buffer.flush()

def read_message():
    length = None
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        name, value = line.decode("ascii").split(":", 1)
        if name.lower() == "content-length":
            length = int(value.strip())
    if length is None:
        return None
    body = sys.stdin.buffer.read(length)
    return json.loads(body)

while True:
    message = read_message()
    if message is None:
        break
    method = message.get("method")
    params = message.get("params") or {}
    if method == "initialize":
        send(message["id"], {"capabilities":{"positionEncoding":"utf-8","textDocumentSync":{"openClose":True,"change":1},"documentFormattingProvider":True}})
    elif method == "textDocument/didOpen":
        document = params["textDocument"]
        documents[document["uri"]] = document["text"]
        record("open", document["uri"], document.get("version"), document["text"])
    elif method == "textDocument/didChange":
        document = params["textDocument"]
        changes = params.get("contentChanges") or []
        if changes:
            documents[document["uri"]] = changes[-1]["text"]
        record("change", document["uri"], document.get("version"), documents.get(document["uri"], ""))
    elif method == "textDocument/formatting":
        uri = params["textDocument"]["uri"]
        record("format", uri, None, documents.get(uri, ""))
        send(message["id"], [])
    elif method == "textDocument/didClose":
        document = params["textDocument"]
        record("close", document["uri"])
        documents.pop(document["uri"], None)
    elif method == "shutdown":
        send(message["id"], None)
    elif method == "exit":
        break
    elif "id" in message:
        send(message["id"], None)
"#;
