//! Bounded recovery for provider turns that complete without actionable output.

use std::{
	collections::VecDeque,
	future::{Ready, ready},
	sync::{Arc, Mutex},
};

use futures::stream;
use omp_agent::{
	Agent, AgentError, AgentSnapshot, AgentState, Error, InvokeFrame, Journal, TurnClient, TurnId,
	TurnInput, TurnOptions, TurnSession,
};
use omp_core::Str;
use omp_env::EnvClient;
use omp_proto::{
	inference::v1 as pb,
	thread::v1::{self as thread, Item},
};
use omp_storage::transcript::{Header, SessionId};
use omp_tool::PromptCaps;

const RETRY_TEXT: &str = "<system-injection>\nStopped without actionable output; task incomplete. \
                          Continue with a user-visible final answer or the next required tool \
                          call.\nAttempt #1/3\n</system-injection>";
const CAP_DETAIL: &str = "Assistant returned no final output after retry cap; try switching models";

#[derive(Clone)]
struct ScriptedClient {
	script: Arc<Mutex<VecDeque<Result<pb::Outcome, pb::TurnError>>>>,
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

	fn submit(&mut self, _frame: InvokeFrame) -> Ready<Result<(), Error>> {
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
	) -> Ready<Result<Self::Session<'client>, Error>> {
		self.opened.lock().expect("opened lock").push(input);
		let event = match self
			.script
			.lock()
			.expect("script lock")
			.pop_front()
			.expect("one script entry per turn")
		{
			Ok(outcome) => Ok(pb::TurnEvent { event: Some(pb::turn_event::Event::Outcome(outcome)) }),
			Err(error) => Err(Error::Terminal(error)),
		};
		ready(Ok(ScriptedSession { events: vec![event] }))
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

fn terminal(kind: pb::turn_error::Kind) -> Result<pb::Outcome, pb::TurnError> {
	Err(pb::TurnError {
		kind: kind as i32,
		detail: "provider detail".to_owned(),
		..pb::TurnError::default()
	})
}

fn success() -> Result<pb::Outcome, pb::TurnError> {
	Ok(pb::Outcome { stop: pb::StopReason::StopEndTurn as i32, ..pb::Outcome::default() })
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

fn agent(
	script: Vec<Result<pb::Outcome, pb::TurnError>>,
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
	(agent, opened, path)
}

#[tokio::test]
async fn empty_output_continues_with_numbered_user_reminder() {
	let (mut agent, opened, path) =
		agent(vec![terminal(pb::turn_error::Kind::EmptyOutput), success()]);
	let result = agent
		.submit([user_text("original")], TurnId::new("root"))
		.await
		.expect("recovered submission");
	assert_eq!(result.committed_turns, 1);
	let opened = opened.lock().expect("opened lock");
	assert_eq!(opened.len(), 2);
	assert!(input_texts(&opened[1]).contains(&RETRY_TEXT));
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
	]);
	let error = agent
		.submit([user_text("original")], TurnId::new("root"))
		.await
		.expect_err("retry cap must fail");
	let AgentError::Turn(Error::Terminal(error)) = error else {
		panic!("expected terminal turn error")
	};
	assert_eq!(error.detail, CAP_DETAIL);
	let opened = opened.lock().expect("opened lock");
	assert_eq!(opened.len(), 4);
	let reminders: Vec<_> = opened
		.iter()
		.skip(1)
		.map(|input| {
			input_texts(input)
				.into_iter()
				.filter(|text| {
					text.starts_with("<system-injection>\nStopped without actionable output")
				})
				.next_back()
				.expect("follow-up turn contains its reminder")
		})
		.collect();
	assert_eq!(reminders.len(), 3);
	assert!(reminders[0].contains("Attempt #1/3"));
	assert!(reminders[1].contains("Attempt #2/3"));
	assert!(reminders[2].contains("Attempt #3/3"));
	drop(opened);
	drop(agent);
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
	let opened = opened.lock().expect("opened lock");
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
