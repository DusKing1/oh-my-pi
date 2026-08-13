//! Transport-neutral contracts for one inference turn.

use std::{
	fmt,
	future::Future,
	pin::Pin,
	task::{Context, Poll},
};

use futures::Stream;
use omp_core::Str;
use omp_llm_inference::TurnId;
use omp_proto::{
	inference::v1::{
		self as pb, ChatParams, ContextRef, Executor, InvokeComplete, InvokeInput, ThreadDelta,
		TurnError, TurnEvent, TurnFrame, ValueMap,
	},
	thread::v1::Thread,
};

/// The canonical conversation input for one logical turn.
#[derive(Clone, Debug)]
pub enum TurnInput {
	/// Supplies a complete thread for a stateless turn or context reseed.
	Full(Thread),
	/// Applies an atomic delta against a held context revision.
	Delta(ContextRef, ThreadDelta),
}

/// Per-turn inference options passed through to the live protocol.
#[derive(Clone, Debug, Default)]
pub struct TurnOptions {
	/// Context to seed when [`TurnInput::Full`] is used.
	///
	/// Leaving this absent makes the full turn stateless. Incremental turns take
	/// their context identity from [`TurnInput::Delta`].
	pub context_id: Option<Str>,
	/// Canonical chat parameters, including model, tools, and sampling controls.
	pub params:     ChatParams,
	/// In-turn invocation capability advertised to the gateway.
	pub executor:   Option<Executor>,
	/// Namespaced turn-level extension properties.
	pub props:      Option<ValueMap>,
}

/// A client response frame for a live server-initiated invocation.
#[derive(Clone, Debug)]
pub enum InvokeFrame {
	/// Streams one canonical input chunk or opaque vendor frame.
	Input(InvokeInput),
	/// Completes an invocation with its canonical result and typed status.
	Complete(InvokeComplete),
}

impl From<InvokeInput> for InvokeFrame {
	#[inline]
	fn from(frame: InvokeInput) -> Self {
		Self::Input(frame)
	}
}

impl From<InvokeComplete> for InvokeFrame {
	#[inline]
	fn from(frame: InvokeComplete) -> Self {
		Self::Complete(frame)
	}
}

impl From<InvokeFrame> for TurnFrame {
	#[inline]
	fn from(frame: InvokeFrame) -> Self {
		let frame = match frame {
			InvokeFrame::Input(frame) => pb::turn_frame::Frame::Input(frame),
			InvokeFrame::Complete(frame) => pb::turn_frame::Frame::Complete(frame),
		};
		Self { frame: Some(frame) }
	}
}

/// A turn-layer failure.
///
/// Protocol terminal errors retain their generated [`TurnError`] verbatim so
/// higher layers can inspect diagnostics and stable error identities. Conflict
/// and need-full are separate variants because they are recoveries owned by the
/// agent loop rather than transport policy.
pub enum Error {
	/// The submitted revision was stale; the embedded error carries the actual
	/// revision.
	Conflict(TurnError),
	/// The referenced context is absent and must be reseeded with a full thread.
	NeedFull(TurnError),
	/// A non-recoverable terminal protocol error.
	Terminal(TurnError),
	/// The tonic request or response stream failed.
	Rpc(tonic::Status),
	/// An RPC channel could not be established.
	Connect(tonic::transport::Error),
	/// The peer violated the turn framing contract.
	Protocol(&'static str),
	/// The local request is invalid before it reaches the service.
	Invalid(&'static str),
	/// The invocation send side is no longer available.
	Closed,
}

impl Error {
	/// Returns the retained terminal protocol error, when this came from one.
	#[inline]
	pub fn turn_error(&self) -> Option<&TurnError> {
		match self {
			Self::Conflict(error) | Self::NeedFull(error) | Self::Terminal(error) => Some(error),
			Self::Rpc(_) | Self::Connect(_) | Self::Protocol(_) | Self::Invalid(_) | Self::Closed => {
				None
			},
		}
	}

	/// Reports whether policy above this seam may recover by rebasing or
	/// reseeding.
	#[inline]
	pub fn is_recovery(&self) -> bool {
		matches!(self, Self::Conflict(_) | Self::NeedFull(_))
	}

	pub(crate) fn from_turn(error: TurnError) -> Self {
		match pb::turn_error::Kind::try_from(error.kind).unwrap_or(pb::turn_error::Kind::Unspecified)
		{
			pb::turn_error::Kind::Conflict => Self::Conflict(error),
			pb::turn_error::Kind::NeedFull => Self::NeedFull(error),
			_ => Self::Terminal(error),
		}
	}
}

impl fmt::Debug for Error {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		fmt::Display::fmt(self, formatter)
	}
}

impl fmt::Display for Error {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::Conflict(_) => formatter.write_str("turn context conflict"),
			Self::NeedFull(_) => formatter.write_str("turn context requires a full reseed"),
			Self::Terminal(error) => write!(
				formatter,
				"terminal turn error ({:?})",
				pb::turn_error::Kind::try_from(error.kind).unwrap_or(pb::turn_error::Kind::Unspecified)
			),
			Self::Rpc(status) => write!(formatter, "turn RPC failed ({:?})", status.code()),
			Self::Connect(_) => formatter.write_str("turn RPC connection failed"),
			Self::Protocol(message) => write!(formatter, "turn protocol error: {message}"),
			Self::Invalid(message) => write!(formatter, "invalid turn: {message}"),
			Self::Closed => formatter.write_str("turn invocation stream is closed"),
		}
	}
}

impl std::error::Error for Error {}

impl From<tonic::Status> for Error {
	#[inline]
	fn from(status: tonic::Status) -> Self {
		Self::Rpc(status)
	}
}

impl From<tonic::transport::Error> for Error {
	#[inline]
	fn from(error: tonic::transport::Error) -> Self {
		Self::Connect(error)
	}
}

/// Starts logical turns against an inference gateway.
///
/// Implementations provide transport only. They never retry, rebase, reseed,
/// journal, or deduplicate beyond forwarding the caller's stable [`TurnId`].
pub trait TurnClient: Send + Sync {
	/// Duplex session returned for one admitted turn.
	type Session<'client>: TurnSession + 'client
	where
		Self: 'client;

	/// Opens one logical turn, moving its canonical input into the request.
	fn turn<'client>(
		&'client self,
		turn_id: TurnId,
		input: TurnInput,
		options: &'client TurnOptions,
	) -> impl Future<Output = Result<Self::Session<'client>, Error>> + Send + 'client;
}

/// A live duplex turn.
///
/// Poll one event at a time, then release the returned event stream before
/// calling [`TurnSession::submit`] in response to an invocation. Dropping the
/// session closes both halves and structurally cancels unfinished upstream
/// work.
pub trait TurnSession: Send {
	/// Borrows the ordered event stream.
	///
	/// `Accepted { replay: true }` is passed through unchanged. An in-band
	/// terminal error becomes a typed [`Error`]; a successful [`pb::Outcome`]
	/// remains a regular canonical [`TurnEvent`].
	fn events(&mut self) -> impl Stream<Item = Result<TurnEvent, Error>> + Send + Unpin + '_;

	/// Submits invocation input or completion on the still-live turn stream.
	fn submit(&mut self, frame: InvokeFrame) -> impl Future<Output = Result<(), Error>> + Send + '_;
}

pub(crate) fn open_frame(
	turn_id: &TurnId,
	input: TurnInput,
	options: &TurnOptions,
) -> Result<TurnFrame, Error> {
	if turn_id.as_str().is_empty() {
		return Err(Error::Invalid("turn_id must not be empty"));
	}
	if options.context_id.as_ref().is_some_and(Str::is_empty) {
		return Err(Error::Invalid("context_id must not be empty when present"));
	}

	let input = match input {
		TurnInput::Full(thread) => pb::turn_request::Input::Seed(pb::Seed {
			context_id: options
				.context_id
				.as_ref()
				.map_or_else(String::new, |id| id.as_str().to_owned()),
			thread:     Some(thread),
		}),
		TurnInput::Delta(context, delta) => pb::turn_request::Input::Incremental(pb::Incremental {
			context: Some(context),
			delta:   Some(delta),
		}),
	};
	Ok(TurnFrame {
		frame: Some(pb::turn_frame::Frame::Open(pb::TurnRequest {
			turn_id:  turn_id.as_str().to_owned(),
			input:    Some(input),
			params:   Some(options.params.clone()),
			executor: options.executor.clone(),
			props:    options.props.clone(),
		})),
	})
}

pub(crate) fn invocation_stream(
	receiver: flume::Receiver<TurnFrame>,
) -> impl Stream<Item = TurnFrame> + Send + 'static {
	futures::stream::unfold(receiver, |receiver| async move {
		let frame = receiver.recv_async().await.ok()?;
		Some((frame, receiver))
	})
}

pub(crate) struct EventStream<'session> {
	stream:   &'session mut tonic::Streaming<TurnEvent>,
	terminal: &'session mut bool,
}

impl<'session> EventStream<'session> {
	pub(crate) fn new(
		stream: &'session mut tonic::Streaming<TurnEvent>,
		terminal: &'session mut bool,
	) -> Self {
		Self { stream, terminal }
	}
}

impl Stream for EventStream<'_> {
	type Item = Result<TurnEvent, Error>;

	fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
		if *self.terminal {
			return Poll::Ready(None);
		}
		match Pin::new(&mut *self.stream).poll_next(context) {
			Poll::Pending => Poll::Pending,
			Poll::Ready(Some(Err(status))) => {
				*self.terminal = true;
				Poll::Ready(Some(Err(Error::Rpc(status))))
			},
			Poll::Ready(Some(Ok(event))) => match event.event {
				Some(pb::turn_event::Event::Error(error)) => {
					*self.terminal = true;
					Poll::Ready(Some(Err(Error::from_turn(error))))
				},
				event @ Some(pb::turn_event::Event::Outcome(_)) => {
					*self.terminal = true;
					Poll::Ready(Some(Ok(TurnEvent { event })))
				},
				event @ Some(_) => Poll::Ready(Some(Ok(TurnEvent { event }))),
				None => {
					*self.terminal = true;
					Poll::Ready(Some(Err(Error::Protocol("TurnEvent.event is required"))))
				},
			},
			Poll::Ready(None) => {
				*self.terminal = true;
				Poll::Ready(Some(Err(Error::Protocol("turn stream ended without a terminal event"))))
			},
		}
	}
}
