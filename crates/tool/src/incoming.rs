//! Linear invocation framing layered over `omp-slopjson`.

use std::{collections::VecDeque, future::Future, marker::PhantomData};

use flume::{Receiver, Sender};
use futures::{FutureExt, pin_mut, select_biased};
use omp_core::Str;
use omp_slopjson::{
	IncomingDoc, IncomingError, IncomingFeed, PullIssue, PullIssueKind, PullPathSegment,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use thiserror::Error;

use crate::{ArgIssue, ArgIssueKind, ArgPath};

/// One event in the client-to-executor invocation stream.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InvocationEvent {
	/// Exact provider-emitted argument text fragment.
	ArgText {
		/// Raw text chunk emitted by the provider.
		fragment: Str,
	},
	/// Explicit durable commitment point and authoritative complete arguments.
	ArgsCommitted {
		/// Final complete argument payload string.
		raw: Str,
	},
	/// Steering observed by an interruptible pull or commitment wait.
	Interrupt(Interrupt),
}

/// Structured invocation interrupt.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Interrupt {
	/// Stable interrupt class supplied by the loop.
	pub class:  Str,
	/// Human-readable reason or steering item.
	pub reason: Str,
}

/// Sending failed because the invocation consumer is gone.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[error("invocation event stream is closed")]
pub struct InvocationSendError;

/// Producer side of one linear invocation event stream.
///
/// Dropping the last producer before `args_committed` disconnects the stream;
/// [`IncomingParams`] then reports an abort rather than inventing commitment.
#[derive(Clone)]
pub struct InvocationFeed {
	tx: Sender<InvocationEvent>,
}

impl InvocationFeed {
	/// Relays one raw argument text fragment verbatim.
	pub fn arg_text(&self, fragment: Str) -> Result<(), InvocationSendError> {
		self.send(InvocationEvent::ArgText { fragment })
	}

	/// Marks the invocation durable with its exact complete raw arguments.
	pub fn args_committed(&self, raw: Str) -> Result<(), InvocationSendError> {
		self.send(InvocationEvent::ArgsCommitted { raw })
	}

	/// Relays one steering interrupt.
	pub fn interrupt(&self, interrupt: Interrupt) -> Result<(), InvocationSendError> {
		self.send(InvocationEvent::Interrupt(interrupt))
	}

	fn send(&self, event: InvocationEvent) -> Result<(), InvocationSendError> {
		self.tx.send(event).map_err(|_| InvocationSendError)
	}
}

/// Failure while pulling a requested argument value.
#[derive(Debug, Error)]
pub enum ParamError {
	/// A pulled value was absent, malformed, mistyped, or abandoned.
	#[error("argument pull failed")]
	Args(ArgIssue),
	/// An interrupt was observed through [`IncomingParams::interruptable`].
	#[error("argument pull interrupted")]
	Interrupted(Interrupt),
	/// Framing violated the one-stream commitment protocol.
	#[error("invalid invocation framing: {0}")]
	Protocol(Str),
}

/// Failure while waiting for the explicit effect gate.
#[derive(Debug, Error)]
pub enum CommitError {
	/// Feed disappeared before commitment.
	#[error("invocation feed dropped before argument commitment")]
	Aborted,
	/// An interrupt was observed by an interruptible wait.
	#[error("argument commitment interrupted")]
	Interrupted(Interrupt),
	/// Framing violated the one-stream commitment protocol.
	#[error("invalid invocation framing: {0}")]
	Protocol(Str),
}
/// Failure while waiting for the next invocation interrupt.
#[derive(Debug, Error)]
pub enum InterruptWaitError {
	/// The invocation owner disappeared before sending another interrupt.
	#[error("invocation event stream closed")]
	Closed,
	/// Framing violated the one-stream invocation protocol.
	#[error("invalid invocation framing: {0}")]
	Protocol(Str),
}

/// Tool-facing cursor over one invocation's single inbound event stream.
///
/// Raw fragments are fed to one `omp-slopjson` document. [`pull`](Self::pull)
/// transfers that document into exactly one linear cursor session; callers can
/// pull selected keys inside the session without touching unknown siblings.
/// Effects must await [`committed`](Self::committed), which succeeds only after
/// an explicit [`InvocationEvent::ArgsCommitted`].
pub struct IncomingParams<'c> {
	events:     Receiver<InvocationEvent>,
	feed:       Option<IncomingFeed>,
	doc:        Option<IncomingDoc>,
	assembled:  String,
	committed:  Option<Str>,
	interrupts: VecDeque<Interrupt>,
	protocol:   Option<Str>,
	_scope:     PhantomData<&'c ()>,
}

impl IncomingParams<'static> {
	/// Creates the producer and consumer sides of one invocation.
	pub fn channel() -> (InvocationFeed, Self) {
		let (tx, events) = flume::unbounded();
		let (feed, doc) = IncomingDoc::channel();
		(InvocationFeed { tx }, Self {
			events,
			feed: Some(feed),
			doc: Some(doc),
			assembled: String::new(),
			committed: None,
			interrupts: VecDeque::new(),
			protocol: None,
			_scope: PhantomData,
		})
	}
}

impl<'c> IncomingParams<'c> {
	/// Constructs a cursor from an existing linear event receiver.
	pub fn from_receiver(events: Receiver<InvocationEvent>) -> Self {
		let (feed, doc) = IncomingDoc::channel();
		Self {
			events,
			feed: Some(feed),
			doc: Some(doc),
			assembled: String::new(),
			committed: None,
			interrupts: VecDeque::new(),
			protocol: None,
			_scope: PhantomData,
		}
	}

	/// Runs the invocation's sole JSON cursor session while continuing to feed
	/// it.
	///
	/// The closure owns the document, making fan-out impossible. It may pull
	/// only selected object keys; malformed or unknown siblings are never
	/// parsed merely because they exist.
	pub async fn pull<R, F, Fut>(&mut self, operation: F) -> Result<R, ParamError>
	where
		F: FnOnce(IncomingDoc) -> Fut,
		Fut: Future<Output = Result<R, IncomingError>>,
	{
		self.drive_pull(operation, false).await
	}

	/// Explicitly opts into decoding and validating the complete argument shape.
	pub async fn whole<T: DeserializeOwned>(&mut self) -> Result<T, ParamError> {
		self
			.pull(|mut doc| async move { doc.whole::<T>().await })
			.await
	}

	/// Waits for the explicit durable argument commitment frame.
	///
	/// Disconnect/drop before that frame is [`CommitError::Aborted`].
	pub async fn committed(&mut self) -> Result<Str, CommitError> {
		self.drive_commit(false).await
	}

	/// Returns a view whose pulls and commitment wait observe interrupts.
	pub fn interruptable(&mut self) -> InterruptibleParams<'_, 'c> {
		InterruptibleParams { inner: self }
	}

	/// Removes the oldest interrupt observed by a non-interruptible operation.
	pub fn take_interrupt(&mut self) -> Option<Interrupt> {
		self.interrupts.pop_front()
	}

	/// Waits for and consumes the next structured interrupt.
	///
	/// This is the cooperative-cancellation arm for resource operations that
	/// begin after argument commitment. A closed feed is reported separately
	/// because the resource owner must then establish terminal effect truth.
	pub async fn next_interrupt(&mut self) -> Result<Interrupt, InterruptWaitError> {
		if let Some(interrupt) = self.interrupts.pop_front() {
			return Ok(interrupt);
		}
		loop {
			match self.events.recv_async().await {
				Ok(event) => {
					if let Some(interrupt) = self.apply(event).map_err(InterruptWaitError::Protocol)? {
						self.interrupts.pop_back();
						return Ok(interrupt);
					}
				},
				Err(_) => {
					self.disconnect();
					return Err(InterruptWaitError::Closed);
				},
			}
		}
	}

	async fn drive_pull<R, F, Fut>(&mut self, operation: F, observe: bool) -> Result<R, ParamError>
	where
		F: FnOnce(IncomingDoc) -> Fut,
		Fut: Future<Output = Result<R, IncomingError>>,
	{
		if let Some(problem) = self.protocol.clone() {
			return Err(ParamError::Protocol(problem));
		}
		if observe && let Some(interrupt) = self.interrupts.pop_front() {
			return Err(ParamError::Interrupted(interrupt));
		}
		let doc = self.doc.take().ok_or_else(|| {
			ParamError::Protocol(Str::from("JSON cursor session was already consumed"))
		})?;
		let events = self.events.clone();
		let pull = operation(doc).fuse();
		pin_mut!(pull);
		loop {
			let receive = events.recv_async().fuse();
			pin_mut!(receive);
			select_biased! {
				result = pull => return result.map_err(param_error),
				event = receive => match event {
					Ok(event) => {
						if let Some(interrupt) = self.apply(event).map_err(ParamError::Protocol)?
							&& observe
						{
							self.interrupts.pop_back();
							return Err(ParamError::Interrupted(interrupt));
						}
					},
					Err(_) => self.disconnect(),
				},
			}
		}
	}

	async fn drive_commit(&mut self, observe: bool) -> Result<Str, CommitError> {
		if let Some(problem) = self.protocol.clone() {
			return Err(CommitError::Protocol(problem));
		}
		if let Some(raw) = self.committed.clone() {
			return Ok(raw);
		}
		if observe && let Some(interrupt) = self.interrupts.pop_front() {
			return Err(CommitError::Interrupted(interrupt));
		}
		loop {
			match self.events.recv_async().await {
				Ok(event) => {
					if let Some(interrupt) = self.apply(event).map_err(CommitError::Protocol)?
						&& observe
					{
						self.interrupts.pop_back();
						return Err(CommitError::Interrupted(interrupt));
					}
					if let Some(raw) = self.committed.clone() {
						return Ok(raw);
					}
				},
				Err(_) => {
					self.disconnect();
					return Err(CommitError::Aborted);
				},
			}
		}
	}

	fn apply(&mut self, event: InvocationEvent) -> Result<Option<Interrupt>, Str> {
		match event {
			InvocationEvent::ArgText { fragment } => {
				if self.committed.is_some() {
					return self.protocol("argument text arrived after commitment");
				}
				self.assembled.push_str(&fragment);
				if let Some(feed) = &mut self.feed {
					feed
						.push(&fragment)
						.map_err(|_| Str::from("JSON feed closed before commitment"))?;
				}
				Ok(None)
			},
			InvocationEvent::ArgsCommitted { raw } => {
				if self.committed.is_some() {
					return self.protocol("duplicate argument commitment");
				}
				if self.assembled.is_empty() && !raw.is_empty() {
					if let Some(feed) = &mut self.feed {
						feed
							.push(&raw)
							.map_err(|_| Str::from("JSON feed closed before commitment"))?;
					}
					self.assembled.push_str(&raw);
				}
				if self.assembled != raw.as_str() {
					return self.protocol("committed arguments differ from streamed fragments");
				}
				self.committed = Some(raw);
				if let Some(feed) = self.feed.take() {
					feed.finish();
				}
				Ok(None)
			},
			InvocationEvent::Interrupt(interrupt) => {
				self.interrupts.push_back(interrupt.clone());
				Ok(Some(interrupt))
			},
		}
	}

	fn protocol<T>(&mut self, message: &'static str) -> Result<T, Str> {
		let problem = Str::from(message);
		self.protocol = Some(problem.clone());
		if let Some(feed) = self.feed.take() {
			feed.abort();
		}
		Err(problem)
	}

	fn disconnect(&mut self) {
		if self.committed.is_none() {
			if let Some(feed) = self.feed.take() {
				feed.abort();
			}
		}
	}
}

/// Interrupt-observing view over an [`IncomingParams`] cursor.
pub struct InterruptibleParams<'p, 'c> {
	inner: &'p mut IncomingParams<'c>,
}

impl InterruptibleParams<'_, '_> {
	/// Runs the sole JSON cursor and returns immediately on observed interrupt.
	pub async fn pull<R, F, Fut>(&mut self, operation: F) -> Result<R, ParamError>
	where
		F: FnOnce(IncomingDoc) -> Fut,
		Fut: Future<Output = Result<R, IncomingError>>,
	{
		self.inner.drive_pull(operation, true).await
	}

	/// Whole-document decode with interrupt observation.
	pub async fn whole<T: DeserializeOwned>(&mut self) -> Result<T, ParamError> {
		self
			.pull(|mut doc| async move { doc.whole::<T>().await })
			.await
	}

	/// Explicit commitment wait with interrupt observation.
	pub async fn committed(&mut self) -> Result<Str, CommitError> {
		self.inner.drive_commit(true).await
	}
}

fn param_error(error: IncomingError) -> ParamError {
	match error {
		IncomingError::Pull(issue) => ParamError::Args(arg_issue(issue)),
		IncomingError::Aborted => ParamError::Args(ArgIssue {
			path:     Vec::new(),
			expected: Str::from("complete pulled value"),
			kind:     ArgIssueKind::Aborted,
			example:  None,
			found:    None,
		}),
		IncomingError::Parse(_) => ParamError::Args(ArgIssue {
			path:     Vec::new(),
			expected: Str::from("declared whole argument shape"),
			kind:     ArgIssueKind::Malformed,
			example:  None,
			found:    None,
		}),
	}
}

fn arg_issue(issue: PullIssue) -> ArgIssue {
	let (kind, found) = match issue.kind {
		PullIssueKind::Missing => (ArgIssueKind::Missing, None),
		PullIssueKind::Incomplete => (ArgIssueKind::Incomplete, None),
		PullIssueKind::Aborted => (ArgIssueKind::Aborted, None),
		PullIssueKind::Malformed => (ArgIssueKind::Malformed, None),
		PullIssueKind::TypeMismatch { found } => (ArgIssueKind::TypeMismatch, Some(Str::from(found))),
	};
	ArgIssue {
		path: issue
			.path
			.into_iter()
			.map(|part| match part {
				PullPathSegment::Key(key) => ArgPath::Key(key),
				PullPathSegment::Index(index) => ArgPath::Index(index as u64),
			})
			.collect(),
		expected: Str::from(issue.expected),
		kind,
		example: None,
		found,
	}
}
