use std::{
	collections::VecDeque,
	future::Future,
	pin::Pin,
	sync::Arc,
	task::{Context, Poll},
};

use futures::Stream;
use omp_agent::{Error, InvokeFrame, TurnClient, TurnId, TurnInput, TurnOptions, TurnSession};
use omp_proto::inference::v1::TurnEvent;
use parking_lot::Mutex;

use super::Gate;

/// One ordered action in a deterministic turn script.
#[derive(Debug)]
pub enum ScriptedStep {
	/// Emits one canonical event or typed turn failure.
	Event(Result<TurnEvent, Error>),
	/// Marks arrival, then pauses the stream until the test releases the gate.
	Wait(Gate),
}

impl From<TurnEvent> for ScriptedStep {
	fn from(event: TurnEvent) -> Self {
		Self::Event(Ok(event))
	}
}

impl From<Result<TurnEvent, Error>> for ScriptedStep {
	fn from(event: Result<TurnEvent, Error>) -> Self {
		Self::Event(event)
	}
}

/// One deterministic turn event stream consumed by [`ScriptedTurnClient`].
#[derive(Debug)]
pub struct ScriptedTurn {
	steps: VecDeque<ScriptedStep>,
}

impl ScriptedTurn {
	/// Scripts an ordered successful event stream.
	#[must_use]
	pub fn events(events: impl IntoIterator<Item = TurnEvent>) -> Self {
		Self { steps: events.into_iter().map(ScriptedStep::from).collect() }
	}

	/// Scripts an ordered stream that may terminate with a typed turn error.
	#[must_use]
	pub fn results(events: impl IntoIterator<Item = Result<TurnEvent, Error>>) -> Self {
		Self { steps: events.into_iter().map(ScriptedStep::from).collect() }
	}

	/// Scripts events interleaved with externally released deterministic gates.
	#[must_use]
	pub fn steps(steps: impl IntoIterator<Item = ScriptedStep>) -> Self {
		Self { steps: steps.into_iter().collect() }
	}
}

/// Exact request observed at the scripted turn seam.
#[derive(Clone, Debug)]
pub struct CapturedTurn {
	/// Stable logical turn identity.
	pub turn_id: TurnId,
	/// Full or incremental canonical input.
	pub input: TurnInput,
	/// Per-turn options seen by the transport.
	pub options: TurnOptions,
	/// Duplex frames submitted in response to server invocations.
	pub submitted: Arc<Mutex<Vec<InvokeFrame>>>,
}

/// Queue-backed deterministic [`TurnClient`] that records every request and duplex response.
#[derive(Clone, Debug)]
pub struct ScriptedTurnClient {
	scripts: Arc<Mutex<VecDeque<ScriptedTurn>>>,
	captured: Arc<Mutex<Vec<CapturedTurn>>>,
}

impl ScriptedTurnClient {
	/// Creates a client that consumes exactly one script per opened turn.
	#[must_use]
	pub fn new(scripts: impl IntoIterator<Item = ScriptedTurn>) -> Self {
		Self {
			scripts: Arc::new(Mutex::new(scripts.into_iter().collect())),
			captured: Arc::new(Mutex::new(Vec::new())),
		}
	}

	/// Returns a stable snapshot of all opened turns and submitted invocation frames.
	#[must_use]
	pub fn captures(&self) -> Vec<CapturedTurn> {
		self.captured.lock().clone()
	}

	/// Returns the number of scripts not yet consumed.
	#[must_use]
	pub fn remaining(&self) -> usize {
		self.scripts.lock().len()
	}
}

impl TurnClient for ScriptedTurnClient {
	type Session<'client> = ScriptedTurnSession;

	fn turn<'client>(
		&'client self,
		turn_id: TurnId,
		input: TurnInput,
		options: &'client TurnOptions,
	) -> impl Future<Output = Result<Self::Session<'client>, Error>> + Send + 'client {
		let script = self.scripts.lock().pop_front();
		let captured = Arc::clone(&self.captured);
		let options = options.clone();
		async move {
			let script = script.ok_or(Error::Invalid("scripted turn queue exhausted"))?;
			let submitted = Arc::new(Mutex::new(Vec::new()));
			captured.lock().push(CapturedTurn {
				turn_id,
				input,
				options,
				submitted: Arc::clone(&submitted),
			});
			Ok(ScriptedTurnSession { steps: script.steps, submitted })
		}
	}
}

/// One live scripted turn session.
#[derive(Debug)]
pub struct ScriptedTurnSession {
	steps: VecDeque<ScriptedStep>,
	submitted: Arc<Mutex<Vec<InvokeFrame>>>,
}

impl TurnSession for ScriptedTurnSession {
	fn events(&mut self) -> impl Stream<Item = Result<TurnEvent, Error>> + Send + Unpin + '_ {
		ScriptedEventStream { steps: &mut self.steps, waiting: None }
	}

	fn submit(&mut self, frame: InvokeFrame) -> impl Future<Output = Result<(), Error>> + Send + '_ {
		self.submitted.lock().push(frame);
		std::future::ready(Ok(()))
	}
}

struct ScriptedEventStream<'session> {
	steps: &'session mut VecDeque<ScriptedStep>,
	waiting: Option<Pin<Box<dyn Future<Output = ()> + Send + 'static>>>,
}

impl Unpin for ScriptedEventStream<'_> {}

impl Stream for ScriptedEventStream<'_> {
	type Item = Result<TurnEvent, Error>;

	fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
		loop {
			if let Some(waiting) = &mut self.waiting {
				if waiting.as_mut().poll(context).is_pending() {
					return Poll::Pending;
				}
				self.waiting = None;
			}
			match self.steps.pop_front() {
				Some(ScriptedStep::Event(event)) => return Poll::Ready(Some(event)),
				Some(ScriptedStep::Wait(gate)) => {
					gate.arrive();
					self.waiting = Some(Box::pin(async move { gate.released().await }));
				},
				None => return Poll::Ready(None),
			}
		}
	}
}
