//! Bounded recovery for provider turns that complete without actionable output.

use std::{
	collections::VecDeque,
	future::{Future, Ready, pending, ready},
	sync::{
		Arc,
		atomic::{AtomicUsize, Ordering},
	},
};

use futures::stream;
use omp_agent::{
	Agent, AgentError, AgentPhase, AgentSnapshot, AgentState, Error, InvokeFrame, Journal,
	TurnClient, TurnId, TurnInput, TurnInputRecord, TurnOptions, TurnOptionsRecord, TurnSession,
	TurnStart,
};
use omp_core::Str;
use omp_env::EnvClient;
use omp_proto::{
	inference::v1 as pb,
	thread::v1::{self as thread, Item},
};
use omp_storage::transcript::{Header, SessionId};
use omp_tool::PromptCaps;
use parking_lot::Mutex;

const RETRY_TEXT: &str = "<system-injection>\nStopped without actionable output; task incomplete. \
                          Continue with a user-visible final answer or the next required tool \
                          call.\nAttempt #1/3\n</system-injection>";
const CAP_DETAIL: &str = "Assistant returned no final output after retry cap; try switching models";

#[derive(Clone)]
struct ScriptedClient {
	script: Arc<Mutex<VecDeque<Result<pb::Outcome, Box<pb::TurnError>>>>>,
	opened: Arc<Mutex<Vec<TurnInput>>>,
}

struct ScriptedSession {
	events: Vec<Result<pb::TurnEvent, Error>>,
}

impl TurnSession for ScriptedSession {
	fn events(
		&mut self,
	) -> impl futures::Stream<Item = Result<pb::TurnEvent, Error>> + Send + Unpin + '_ {
		stream::iter(std::mem::take(&mut self.events))
	}

	fn submit(
		&mut self,
		_frame: InvokeFrame,
	) -> impl Future<Output = Result<(), Error>> + Send + '_ {
		ready(Ok(()))
	}
}

impl TurnClient for ScriptedClient {
	type Session<'client> = ScriptedSession;

	fn turn<'client>(
		&'client self,
		_turn_id: TurnId,
		input: TurnInput,
		_options: &'client TurnOptions,
	) -> impl Future<Output = Result<Self::Session<'client>, Error>> + Send + 'client {
		self.opened.lock().push(input);
		let outcome = self
			.script
			.lock()
			.pop_front()
			.expect("one script entry per turn");
		let event = match outcome {
			Ok(outcome) => Ok(pb::TurnEvent { event: Some(pb::turn_event::Event::Outcome(outcome)) }),
			Err(error) => Err(Error::Terminal(error)),
		};
		ready(Ok(ScriptedSession { events: vec![event] }))
	}
}
#[derive(Clone)]
struct CrashClient {
	opened: flume::Sender<TurnInput>,
	calls:  Arc<AtomicUsize>,
}

impl TurnClient for CrashClient {
	type Session<'client> = ScriptedSession;

	fn turn<'client>(
		&'client self,
		_turn_id: TurnId,
		input: TurnInput,
		_options: &'client TurnOptions,
	) -> impl Future<Output = Result<Self::Session<'client>, Error>> + Send + 'client {
		let opened = self.opened.clone();
		let call = self.calls.fetch_add(1, Ordering::Relaxed);
		async move {
			opened.send_async(input).await.map_err(|_| Error::Closed)?;
			if call == 0 {
				return Ok(ScriptedSession {
					events: vec![Err(Error::Terminal(Box::new(pb::TurnError {
						kind: pb::turn_error::Kind::EmptyOutput as i32,
						detail: "provider detail".to_owned(),
						..pb::TurnError::default()
					})))],
				});
			}
			pending().await
		}
	}
}

fn user_text(text: &str) -> Item {
	Item {
		seq:           0,
		created_at_ms: 1,
		kind:          Some(thread::item::Kind::Message(thread::Message {
			role:  thread::Role::User as i32,
			parts: vec![thread::Part { kind: Some(thread::part::Kind::Text(text.to_owned())) }],
		})),
		props:         None,
	}
}

fn terminal(kind: pb::turn_error::Kind) -> Result<pb::Outcome, Box<pb::TurnError>> {
	Err(Box::new(pb::TurnError {
		kind: kind as i32,
		detail: "provider detail".to_owned(),
		..pb::TurnError::default()
	}))
}

fn success() -> pb::Outcome {
	pb::Outcome { stop: pb::StopReason::StopEndTurn as i32, ..pb::Outcome::default() }
}

fn input_texts(input: &TurnInput) -> Vec<&str> {
	let items = match input {
		TurnInput::Full(thread) => thread.items.as_slice(),
		TurnInput::Delta(_, delta) => delta.append.as_slice(),
	};
	items
		.iter()
		.filter_map(|item| match item.kind.as_ref() {
			Some(thread::item::Kind::Message(message)) => {
				message
					.parts
					.iter()
					.find_map(|part| match part.kind.as_ref() {
						Some(thread::part::Kind::Text(text)) => Some(text.as_str()),
						_ => None,
					})
			},
			_ => None,
		})
		.collect()
}
fn build_agent(
	journal: Journal,
	script: Vec<Result<pb::Outcome, Box<pb::TurnError>>>,
) -> (Agent<ScriptedClient>, Arc<Mutex<Vec<TurnInput>>>) {
	let opened = Arc::new(Mutex::new(Vec::new()));
	let client =
		ScriptedClient { script: Arc::new(Mutex::new(script.into())), opened: Arc::clone(&opened) };
	let (env, _transport) = EnvClient::in_process(1);
	let agent =
		Agent::new(client, env, AgentState::new(AgentSnapshot::default()), journal, PromptCaps {
			maximum_parts:      16,
			maximum_text_bytes: 16_384,
			media:              false,
		});
	(agent, opened)
}

fn turn_start(id: &str) -> TurnStart {
	TurnStart {
		turn_id:            Str::from(id),
		item_events:        Vec::new(),
		prompt_hash:        [0; 32],
		prompt_head_events: Vec::new(),
		toolset_hash:       [0; 32],
		enabled_tools:      Vec::new(),
		sequence_targets:   Vec::new(),
		input:              TurnInputRecord::Full { thread: thread::Thread::default() },
		options:            TurnOptionsRecord {
			context_id: None,
			params:     pb::ChatParams::default(),
			executor:   None,
			props:      None,
		},
	}
}
fn exhausted_journal(path: &std::path::Path) -> Journal {
	let mut journal = Journal::create(path, &Header {
		v:       4,
		id:      SessionId(Str::from("empty-output-exhausted-test")),
		created: 1,
		cwd:     std::env::temp_dir(),
	})
	.expect("create journal");
	journal
		.start_turn(2, turn_start("prior-success"))
		.expect("start prior success");
	journal
		.append_gateway_outcome(3, "prior-success", success())
		.expect("commit prior success");
	let texts = [
		"capped original".to_owned(),
		RETRY_TEXT.to_owned(),
		RETRY_TEXT.replace("Attempt #1/3", "Attempt #2/3"),
		RETRY_TEXT.replace("Attempt #1/3", "Attempt #3/3"),
	];
	for (attempt, text) in texts.into_iter().enumerate() {
		let id = format!("failed-{attempt}");
		let event = journal
			.append_turn_input(4 + attempt as u64 * 2, &id, user_text(&text), None)
			.expect("stage capped-chain input");
		let mut start = turn_start(&id);
		start.item_events = vec![event];
		start.input =
			TurnInputRecord::Full { thread: thread::Thread { items: vec![user_text(&text)] } };
		journal
			.start_turn(5 + attempt as u64 * 2, start)
			.expect("start capped-chain turn");
		journal
			.abort_turn(6 + attempt as u64 * 2, &id, attempt < 3)
			.expect("abort capped-chain turn");
	}
	journal
}

fn agent(
	script: Vec<Result<pb::Outcome, Box<pb::TurnError>>>,
) -> (Agent<ScriptedClient>, Arc<Mutex<Vec<TurnInput>>>, std::path::PathBuf) {
	let path = std::env::temp_dir().join(format!(
		"omp-agent-empty-output-{}-{}.jsonl",
		std::process::id(),
		ulid::Ulid::generate()
	));
	let journal = Journal::create(&path, &Header {
		v:       4,
		id:      SessionId(Str::from("empty-output-test")),
		created: 1,
		cwd:     std::env::temp_dir(),
	})
	.expect("create journal");
	let (agent, opened) = build_agent(journal, script);
	(agent, opened, path)
}

#[tokio::test]
async fn empty_output_continues_with_numbered_user_reminder() {
	let (mut agent, opened, path) =
		agent(vec![terminal(pb::turn_error::Kind::EmptyOutput), Ok(success())]);
	let result = agent
		.submit([user_text("original")], TurnId::new("root"))
		.await
		.expect("recovered submission");
	assert_eq!(result.committed_turns, 1);
	let opened = opened.lock();
	assert_eq!(opened.len(), 2);
	assert!(input_texts(&opened[1]).contains(&RETRY_TEXT));
	assert!(matches!(&opened[1], TurnInput::Full(_)));
	assert!(input_texts(&opened[1]).contains(&"original"));
	drop(opened);
	drop(agent);
	std::fs::remove_file(path).expect("remove journal");
}

#[tokio::test]
async fn fourth_empty_output_hits_cap_after_exactly_three_reminders() {
	let (mut agent, opened, path) = agent(vec![
		terminal(pb::turn_error::Kind::EmptyOutput),
		terminal(pb::turn_error::Kind::EmptyOutput),
		terminal(pb::turn_error::Kind::EmptyOutput),
		terminal(pb::turn_error::Kind::EmptyOutput),
		Ok(success()),
	]);
	let error = agent
		.submit([user_text("original")], TurnId::new("root"))
		.await
		.expect_err("retry cap must fail");
	let AgentError::Turn(Error::Terminal(error)) = error else {
		panic!("expected terminal turn error")
	};
	assert_eq!(error.detail, CAP_DETAIL);
	let inputs = opened.lock();
	assert_eq!(inputs.len(), 4);
	let reminders: Vec<_> = inputs
		.iter()
		.skip(1)
		.map(|input| {
			input_texts(input)
				.into_iter()
				.rfind(|text| text.starts_with("<system-injection>\nStopped without actionable output"))
				.expect("follow-up turn contains its reminder")
		})
		.collect();
	assert_eq!(reminders.len(), 3);
	assert!(reminders[0].contains("Attempt #1/3"));
	assert!(reminders[1].contains("Attempt #2/3"));
	assert!(reminders[2].contains("Attempt #3/3"));
	drop(inputs);
	assert_eq!(agent.events().phase(), AgentPhase::Idle);
	agent
		.submit([user_text("fresh after cap")], TurnId::new("fresh"))
		.await
		.expect("fresh prompt succeeds after terminal cap");
	assert_eq!(agent.events().phase(), AgentPhase::Idle);
	assert_eq!(opened.lock().len(), 5);
	drop(agent);
	std::fs::remove_file(path).expect("remove journal");
}

#[tokio::test]
async fn retry_count_survives_journal_reopen() {
	let path = std::env::temp_dir().join(format!(
		"omp-agent-empty-output-reopen-{}-{}.jsonl",
		std::process::id(),
		ulid::Ulid::generate()
	));
	let mut journal = Journal::create(&path, &Header {
		v:       4,
		id:      SessionId(Str::from("empty-output-reopen-test")),
		created: 1,
		cwd:     std::env::temp_dir(),
	})
	.expect("create journal");
	let original = user_text("original");
	let first_event = journal
		.append_turn_input(2, "failed-1", original.clone(), None)
		.expect("stage original input");
	let mut first = turn_start("failed-1");
	first.item_events = vec![first_event];
	first.input = TurnInputRecord::Full { thread: thread::Thread { items: vec![original.clone()] } };
	journal
		.start_turn(3, first)
		.expect("start first failed turn");
	journal
		.abort_turn(4, "failed-1", true)
		.expect("abort first failed turn");
	let first_reminder = user_text(RETRY_TEXT);
	let second_event = journal
		.append_turn_input(5, "failed-2", first_reminder.clone(), None)
		.expect("stage first reminder");
	let mut second = turn_start("failed-2");
	second.item_events = vec![second_event];
	second.input =
		TurnInputRecord::Full { thread: thread::Thread { items: vec![original, first_reminder] } };
	journal
		.start_turn(6, second)
		.expect("start second failed turn");
	journal
		.abort_turn(7, "failed-2", true)
		.expect("abort second failed turn");
	drop(journal);

	let reopened = Journal::open(&path).expect("reopen journal");
	let (mut agent, opened) = build_agent(reopened, vec![
		terminal(pb::turn_error::Kind::EmptyOutput),
		terminal(pb::turn_error::Kind::EmptyOutput),
	]);
	let error = agent
		.submit([], TurnId::new("root"))
		.await
		.expect_err("persisted retry count must reach cap");
	let AgentError::Turn(Error::Terminal(error)) = error else {
		panic!("expected terminal turn error")
	};
	assert_eq!(error.detail, CAP_DETAIL);
	let opened = opened.lock();
	assert_eq!(opened.len(), 2);
	assert!(
		input_texts(&opened[1])
			.iter()
			.any(|text| text.contains("Attempt #3/3"))
	);
	// The full-reseed projection retains prior reminders; each attempt number
	// must appear exactly once (no duplicated reminder on reopen).
	for attempt in ["Attempt #1/3", "Attempt #2/3", "Attempt #3/3"] {
		let occurrences = input_texts(&opened[1])
			.iter()
			.filter(|text| text.contains(attempt))
			.count();
		assert_eq!(occurrences, 1, "{attempt} duplicated in reopened continuation");
	}
	drop(opened);
	drop(agent);
	std::fs::remove_file(path).expect("remove journal");
}

#[tokio::test]
async fn crash_after_abort_reclaims_input_under_fresh_full_reseed() {
	let path = std::env::temp_dir().join(format!(
		"omp-agent-empty-output-abort-gap-{}-{}.jsonl",
		std::process::id(),
		ulid::Ulid::generate()
	));
	let mut journal = Journal::create(&path, &Header {
		v:       4,
		id:      SessionId(Str::from("empty-output-abort-gap-test")),
		created: 1,
		cwd:     std::env::temp_dir(),
	})
	.expect("create journal");
	let prior_revision = thread::Revision { head: 0, token: vec![1].into() };
	let mut prior = turn_start("prior-success");
	prior.options.context_id = Some(Str::from("context"));
	journal
		.start_turn(2, prior)
		.expect("start prior successful turn");
	journal
		.append_gateway_outcome(3, "prior-success", pb::Outcome {
			stop: pb::StopReason::StopEndTurn as i32,
			revision: Some(prior_revision.clone()),
			..pb::Outcome::default()
		})
		.expect("commit prior successful turn");
	let original = user_text("original");
	let input_event = journal
		.append_turn_input(4, "failed", original.clone(), None)
		.expect("stage original input");
	let mut failed = turn_start("failed");
	failed.item_events = vec![input_event];
	failed.sequence_targets = vec![input_event];
	failed.input = TurnInputRecord::Delta {
		context: pb::ContextRef {
			context_id: "context".to_owned(),
			expected:   Some(prior_revision),
		},
		delta:   pb::ThreadDelta { truncate_to: None, append: vec![original] },
	};
	failed.options.context_id = Some(Str::from("context"));
	journal.start_turn(5, failed).expect("start failed turn");
	journal
		.abort_turn(6, "failed", true)
		.expect("abort failed turn");
	drop(journal);

	let reopened = Journal::open(&path).expect("reopen after abort gap");
	let (recovery_id, recovery_events) = reopened
		.pending_input_submission()
		.expect("released input remains startup-visible");
	assert_ne!(recovery_id.as_str(), "failed");
	assert_eq!(recovery_events, &[input_event]);
	let (mut agent, opened) =
		build_agent(reopened, vec![terminal(pb::turn_error::Kind::EmptyOutput), Ok(success())]);
	agent
		.submit([], TurnId::new("restart"))
		.await
		.expect("fresh reclaimed submission succeeds");
	let opened = opened.lock();
	assert_eq!(opened.len(), 2);
	assert!(matches!(&opened[0], TurnInput::Full(_)));

	let first_texts = input_texts(&opened[0]);
	assert!(first_texts.contains(&"original"));
	assert!(first_texts.contains(&RETRY_TEXT));
	assert!(
		input_texts(&opened[1])
			.iter()
			.any(|text| text.contains("Attempt #2/3"))
	);
	drop(opened);
	drop(agent);
	std::fs::remove_file(path).expect("remove journal");
}
#[tokio::test]
async fn exhausted_chain_is_not_released_and_fresh_user_prompt_resets_cap() {
	let path = std::env::temp_dir().join(format!(
		"omp-agent-empty-output-exhausted-{}-{}.jsonl",
		std::process::id(),
		ulid::Ulid::generate()
	));
	let journal = exhausted_journal(&path);
	drop(journal);

	let reopened = Journal::open(&path).expect("reopen exhausted chain");
	assert_eq!(reopened.trailing_aborts(), 0);
	assert!(reopened.pending_turn().is_none());
	assert!(reopened.pending_input_submission().is_none());
	let (mut agent, opened) =
		build_agent(reopened, vec![terminal(pb::turn_error::Kind::EmptyOutput), Ok(success())]);
	agent
		.submit([user_text("fresh task")], TurnId::new("fresh"))
		.await
		.expect("fresh user task recovers after exhausted chain");
	let opened = opened.lock();
	assert_eq!(opened.len(), 2);
	let first_texts = input_texts(&opened[0]);
	assert!(first_texts.contains(&"fresh task"));
	assert_eq!(
		first_texts
			.iter()
			.filter(|text| text.contains("Attempt #3/3"))
			.count(),
		1,
		"reopen must not duplicate the final reminder"
	);
	assert_eq!(
		input_texts(&opened[1])
			.iter()
			.filter(|text| **text == RETRY_TEXT)
			.count(),
		2,
		"the fresh epoch must append its own Attempt #1/3 reminder"
	);
	drop(opened);
	drop(agent);
	std::fs::remove_file(path).expect("remove journal");
}

#[tokio::test]
async fn fresh_epoch_abort_releases_only_fresh_inputs_after_reopen() {
	let path = std::env::temp_dir().join(format!(
		"omp-agent-empty-output-new-epoch-{}-{}.jsonl",
		std::process::id(),
		ulid::Ulid::generate()
	));
	let mut journal = exhausted_journal(&path);
	let fresh = user_text("fresh crash task");
	let fresh_event = journal
		.append_turn_input(20, "fresh-failed", fresh.clone(), None)
		.expect("stage fresh epoch input");
	let mut start = turn_start("fresh-failed");
	start.item_events = vec![fresh_event];
	start.input = TurnInputRecord::Full { thread: thread::Thread { items: vec![fresh] } };
	journal
		.start_turn(21, start)
		.expect("start fresh epoch turn");
	journal
		.abort_turn(22, "fresh-failed", true)
		.expect("abort fresh epoch turn");
	drop(journal);

	let reopened = Journal::open(&path).expect("reopen fresh recovery epoch");
	assert_eq!(reopened.trailing_aborts(), 1);
	let (recovery_id, events) = reopened
		.pending_input_submission()
		.expect("fresh epoch remains startup-visible");
	assert_ne!(recovery_id.as_str(), "fresh-failed");
	assert_eq!(events, &[fresh_event], "old exhausted inputs stay fenced");
	let (mut agent, opened) =
		build_agent(reopened, vec![terminal(pb::turn_error::Kind::EmptyOutput), Ok(success())]);
	agent
		.submit([], TurnId::new("restart"))
		.await
		.expect("fresh epoch resumes through second reminder");
	let opened = opened.lock();
	assert_eq!(opened.len(), 2);
	assert!(matches!(&opened[0], TurnInput::Full(_)));
	assert!(input_texts(&opened[0]).contains(&"fresh crash task"));
	assert!(
		input_texts(&opened[1])
			.iter()
			.any(|text| text.contains("Attempt #2/3"))
	);
	drop(opened);
	drop(agent);
	std::fs::remove_file(path).expect("remove journal");
}

#[tokio::test]
async fn crash_replay_reseeds_original_input_and_preserves_retry_count() {
	let path = std::env::temp_dir().join(format!(
		"omp-agent-empty-output-crash-{}-{}.jsonl",
		std::process::id(),
		ulid::Ulid::generate()
	));
	let mut journal = Journal::create(&path, &Header {
		v:       4,
		id:      SessionId(Str::from("empty-output-crash-test")),
		created: 1,
		cwd:     std::env::temp_dir(),
	})
	.expect("create journal");
	journal
		.start_turn(2, turn_start("prior-success"))
		.expect("start prior successful turn");
	journal
		.append_gateway_outcome(3, "prior-success", success())
		.expect("commit prior successful turn");

	let (opened_tx, opened_rx) = flume::unbounded();
	let client = CrashClient { opened: opened_tx, calls: Arc::new(AtomicUsize::new(0)) };
	let (env, _transport) = EnvClient::in_process(1);
	let mut first_agent =
		Agent::new(client, env, AgentState::new(AgentSnapshot::default()), journal, PromptCaps {
			maximum_parts:      16,
			maximum_text_bytes: 16_384,
			media:              false,
		});
	let running = tokio::spawn(async move {
		first_agent
			.submit([user_text("original")], TurnId::new("root"))
			.await
	});
	opened_rx
		.recv_async()
		.await
		.expect("observe failed request");
	opened_rx
		.recv_async()
		.await
		.expect("observe live continuation");
	running.abort();
	assert!(
		running
			.await
			.expect_err("crashed task must be cancelled")
			.is_cancelled()
	);

	let reopened = Journal::open(&path).expect("reopen after interrupted continuation");
	assert_eq!(reopened.trailing_aborts(), 1);
	let (mut replayed, opened) = build_agent(reopened, vec![
		terminal(pb::turn_error::Kind::EmptyOutput),
		terminal(pb::turn_error::Kind::EmptyOutput),
		terminal(pb::turn_error::Kind::EmptyOutput),
	]);
	let error = replayed
		.submit([], TurnId::new("restart"))
		.await
		.expect_err("persisted abort must count toward cap");
	let AgentError::Turn(Error::Terminal(error)) = error else {
		panic!("expected terminal turn error")
	};
	assert_eq!(error.detail, CAP_DETAIL);
	let opened = opened.lock();
	assert_eq!(opened.len(), 3);
	assert!(matches!(&opened[0], TurnInput::Full(_)));
	let first_texts = input_texts(&opened[0]);
	assert!(first_texts.contains(&"original"));
	assert!(first_texts.contains(&RETRY_TEXT));
	assert!(
		input_texts(&opened[1])
			.iter()
			.any(|text| text.contains("Attempt #2/3"))
	);
	assert!(
		input_texts(&opened[2])
			.iter()
			.any(|text| text.contains("Attempt #3/3"))
	);
	drop(opened);
	drop(replayed);
	std::fs::remove_file(path).expect("remove journal");
}

#[tokio::test]
async fn other_terminal_error_fails_without_reminder() {
	let (mut agent, opened, path) = agent(vec![terminal(pb::turn_error::Kind::Upstream)]);
	let error = agent
		.submit([user_text("original")], TurnId::new("root"))
		.await
		.expect_err("upstream error must fail immediately");
	assert!(matches!(error, AgentError::Turn(Error::Terminal(_))));
	let opened = opened.lock();
	assert_eq!(opened.len(), 1);
	assert!(
		!input_texts(&opened[0])
			.iter()
			.any(|text| text.contains("Stopped without actionable output"))
	);
	drop(opened);
	drop(agent);
	std::fs::remove_file(path).expect("remove journal");
}
