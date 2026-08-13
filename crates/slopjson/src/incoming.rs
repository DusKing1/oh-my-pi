//! Typed cursors over a JSON document while its text is still arriving.
//!
//! [`IncomingDoc::channel`] returns a push-side [`IncomingFeed`] and a
//! read-side root cursor. The producer appends UTF-8 fragments, then explicitly
//! calls [`IncomingFeed::finish`] or [`IncomingFeed::abort`]. Dropping the feed
//! aborts it. There is one shared append-only buffer and one exclusive linear
//! cursor: child cursors retain a mutable borrow of their parent, and pulls are
//! ordinary futures whose cancellation releases that borrow. There are no
//! snapshots, per-field events, or broadcast/fan-out channels.
//!
//! A scalar completes at its closing quote/delimiter, and a container completes
//! only when its closing delimiter arrives. Finished-but-truncated input yields
//! [`PullIssueKind::Incomplete`]; abandoned input yields
//! [`PullIssueKind::Aborted`]. String chunks contain only decoded bytes whose
//! meaning is stable, so an escape or Unicode escape may span any number of
//! fragments.
//!
//! Pulling an [`IncomingObject::key`] makes that key required: a missing or
//! mistyped value is a structured [`PullIssue`]. Object members never pulled
//! are skipped without validation. [`IncomingDoc::whole`] is the explicit
//! whole-document pull and runs only after successful input completion.
//!
//! Object cursors bind the first occurrence of a duplicate key. In contrast,
//! [`IncomingJson::value`] and whole-container collection use the crate's final
//! [`crate::parse`] path, whose [`crate::Object`] has normal last-write-wins
//! behavior. Consumers for which duplicates are significant must detect that
//! divergence themselves.
//!
//! Mid-stream cursors tolerate incomplete tokens but read double-quoted
//! strings with the final parser's strict closing rule ([`Mode::Incoming`]):
//! an unescaped inner `"` can never swallow a sibling key or value. A pulled
//! scalar completes only once a value terminator follows it, like numbers,
//! so structural garbage after a value surfaces as
//! [`PullIssueKind::Incomplete`] rather than a silently misparsed pull.
//! Single-quote recovery (`'it's'`) is shared with the final parser and
//! passes both.

use std::{
	fmt,
	future::poll_fn,
	marker::PhantomData,
	sync::Arc,
	task::{Poll, Waker},
};

use omp_core::Str;
use parking_lot::Mutex;
use serde::de::DeserializeOwned;
use smallvec::SmallVec;
use thiserror::Error;

use crate::{
	Number, Object, ParseError, Value, parse,
	parser::{MAX_DEPTH, Mode, Parser},
};

/// Failure while awaiting an incoming JSON value.
#[derive(Debug, Error)]
pub enum IncomingError {
	/// A pulled JSON value was missing, mistyped, malformed, incomplete, or
	/// aborted.
	#[error(transparent)]
	Pull(#[from] PullIssue),
	/// The producer abandoned the input before marking it complete.
	#[error("incoming JSON input was aborted")]
	Aborted,
	/// An explicitly requested whole-document decode failed.
	#[error(transparent)]
	Parse(#[from] ParseError),
}

/// Structured reason a pulled JSON value could not be supplied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PullIssue {
	/// Full key/index path pulled by the consumer.
	pub path:     Vec<PullPathSegment>,
	/// Shape requested by the typed cursor.
	pub expected: &'static str,
	/// Why the pull could not produce that shape.
	pub kind:     PullIssueKind,
}

impl fmt::Display for PullIssue {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "invalid JSON pull {:?}: expected {} ({})", self.path, self.expected, self.kind)
	}
}

impl std::error::Error for PullIssue {}

/// Location component in a pulled JSON path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PullPathSegment {
	/// Object member name.
	Key(Str),
	/// Array element index.
	Index(usize),
}

/// Kind of pull failure represented by [`PullIssue`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PullIssueKind {
	/// The requested member was absent when its container completed.
	Missing,
	/// The producer finished before the pulled value's closing token.
	Incomplete,
	/// The producer abandoned the input before the pull completed.
	Aborted,
	/// A complete pulled value could not be parsed.
	Malformed,
	/// A value was present with a different JSON shape.
	TypeMismatch {
		/// Shape observed in the input.
		found: &'static str,
	},
}

impl fmt::Display for PullIssueKind {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::Missing => f.write_str("missing"),
			Self::Incomplete => f.write_str("incomplete"),
			Self::Aborted => f.write_str("aborted"),
			Self::Malformed => f.write_str("malformed"),
			Self::TypeMismatch { found } => write!(f, "found {found}"),
		}
	}
}

/// Error returned when text is pushed after the feed has closed.
#[derive(Debug, Clone, Copy, Error, PartialEq, Eq)]
#[error("incoming JSON feed is already closed")]
pub struct FeedClosed;

type WakerSet = SmallVec<Waker, 4>;

#[derive(Default)]
struct InputState {
	text:   String,
	end:    End,
	wakers: WakerSet,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum End {
	#[default]
	Open,
	Finished,
	Aborted,
}

#[derive(Default)]
struct Shared {
	state: Mutex<InputState>,
}

/// Push side of an [`IncomingDoc`] channel.
///
/// Dropping this handle abandons the input, like [`abort`](Self::abort).
/// Call [`finish`](Self::finish) explicitly to mark the document complete.
pub struct IncomingFeed {
	shared: Arc<Shared>,
	closed: bool,
}

impl IncomingFeed {
	/// Append one UTF-8 fragment and wake every pending cursor.
	pub fn push(&mut self, fragment: &str) -> Result<(), FeedClosed> {
		let mut state = self.shared.state.lock();
		if state.end != End::Open {
			return Err(FeedClosed);
		}
		state.text.push_str(fragment);
		let wakers = std::mem::take(&mut state.wakers);
		drop(state);
		wake_all(wakers);
		Ok(())
	}

	/// Mark the input complete and wake the pending cursor.
	pub fn finish(mut self) {
		self.close(End::Finished);
	}

	/// Abandon the input and wake the pending cursor.
	pub fn abort(mut self) {
		self.close(End::Aborted);
	}

	fn close(&mut self, end: End) {
		if self.closed {
			return;
		}
		self.closed = true;
		let mut state = self.shared.state.lock();
		state.end = end;
		let wakers = std::mem::take(&mut state.wakers);
		drop(state);
		wake_all(wakers);
	}
}

impl Drop for IncomingFeed {
	fn drop(&mut self) {
		self.close(End::Aborted);
	}
}

/// Root cursor over one growing JSON document.
pub struct IncomingDoc {
	shared: Arc<Shared>,
}

impl IncomingDoc {
	/// Create a push feed and its read-side document cursor.
	pub fn channel() -> (IncomingFeed, Self) {
		let shared = Arc::new(Shared::default());
		(IncomingFeed { shared: Arc::clone(&shared), closed: false }, Self { shared })
	}

	/// Await explicit input completion.
	///
	/// Returns [`IncomingError::Aborted`] if the feed is aborted or dropped
	/// without an explicit [`IncomingFeed::finish`].
	pub async fn finished(&self) -> Result<(), IncomingError> {
		poll_fn(|cx| {
			let mut state = self.shared.state.lock();
			match state.end {
				End::Finished => Poll::Ready(Ok(())),
				End::Aborted => Poll::Ready(Err(IncomingError::Aborted)),
				End::Open => {
					register_waker(&mut state.wakers, cx.waker());
					Poll::Pending
				},
			}
		})
		.await
	}

	/// Deserialize the entire finished document into `T`.
	///
	/// This is an explicit whole-document pull and waits for
	/// [`IncomingFeed::finish`]. The mutable borrow makes it one ordinary,
	/// cancellation-composable pull: dropping the future releases the cursor
	/// rather than leaving a subscription behind. Aborted input is not decoded.
	pub async fn whole<T: DeserializeOwned>(&mut self) -> Result<T, IncomingError> {
		self.finished().await?;
		let state = self.shared.state.lock();
		crate::from_str(&state.text).map_err(IncomingError::from)
	}

	/// Borrow the single linear cursor for the root JSON value.
	///
	/// A cursor and every child derived from it retain this mutable borrow.
	/// Consequently a document cannot be snapshotted or fanned out into
	/// concurrent pulls; cancelling or completing the pull releases it.
	pub fn json(&mut self) -> IncomingJson<'_> {
		IncomingJson { shared: Arc::clone(&self.shared), path: Vec::new(), _linear: PhantomData }
	}
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum PathPart {
	Key(Str),
	Index(usize),
}

/// Cursor for one JSON value in the incoming document.
pub struct IncomingJson<'doc> {
	shared:  Arc<Shared>,
	path:    Vec<PathPart>,
	_linear: PhantomData<&'doc mut IncomingDoc>,
}

impl<'doc> IncomingJson<'doc> {
	/// Await and parse the complete value.
	pub async fn value(&mut self) -> Result<Value, IncomingError> {
		self.value_with("value").await
	}

	/// Await and deserialize this complete value into `T`.
	///
	/// Choosing this method explicitly opts the pulled subtree into complete
	/// typed validation. A malformed or mistyped subtree is reported at this
	/// cursor's structured pull path.
	pub async fn whole<T: DeserializeOwned>(&mut self) -> Result<T, IncomingError> {
		let expected = std::any::type_name::<T>();
		let located = wait_for(&self.shared, &self.path, WaitMode::Complete, expected).await?;
		let state = self.shared.state.lock();
		crate::from_str(&state.text[located.start..located.end.expect("complete wait has an end")])
			.map_err(|_| pull_issue(&self.path, expected, PullIssueKind::Malformed))
	}

	/// Convert this cursor into a decoded incremental string cursor.
	pub fn string(self) -> IncomingString<'doc> {
		IncomingString { json: self, emitted: 0, done: false }
	}

	/// Convert this cursor into an array element cursor.
	pub fn array(self) -> IncomingArray<'doc> {
		IncomingArray { json: self, next: 0 }
	}

	/// Convert this cursor into an object cursor.
	pub fn object(self) -> IncomingObject<'doc> {
		IncomingObject { json: self }
	}

	/// Await a complete number.
	pub async fn number(&mut self) -> Result<Number, IncomingError> {
		match self.value_with("number").await? {
			Value::Number(value) => Ok(value),
			other => Err(type_mismatch(&self.path, "number", value_name(&other))),
		}
	}

	/// Await a complete boolean.
	pub async fn boolean(&mut self) -> Result<bool, IncomingError> {
		match self.value_with("boolean").await? {
			Value::Bool(value) => Ok(value),
			other => Err(type_mismatch(&self.path, "boolean", value_name(&other))),
		}
	}

	/// Await a complete null value.
	pub async fn null(&mut self) -> Result<(), IncomingError> {
		match self.value_with("null").await? {
			Value::Null => Ok(()),
			other => Err(type_mismatch(&self.path, "null", value_name(&other))),
		}
	}

	async fn value_with(&self, expected: &'static str) -> Result<Value, IncomingError> {
		let located = wait_for(&self.shared, &self.path, WaitMode::Complete, expected).await?;
		match located.kind {
			Kind::Null => Ok(Value::Null),
			Kind::Bool(value) => Ok(Value::Bool(value)),
			Kind::Number(value) => Ok(Value::Number(value)),
			Kind::String { value, .. } => Ok(Value::String(value)),
			Kind::Array | Kind::Object => {
				let state = self.shared.state.lock();
				parse(&state.text[located.start..located.end.expect("complete wait has an end")])
					.map_err(|_| pull_issue(&self.path, expected, PullIssueKind::Malformed))
			},
		}
	}
}

/// Incremental decoded string consumer.
///
/// Chunks are owned [`Str`] slices because the append buffer may grow while an
/// async caller holds a chunk. They are emitted in order without overlap;
/// [`finish`](Self::finish) returns the complete decoded string independently
/// of whether chunks were consumed.
pub struct IncomingString<'doc> {
	json:    IncomingJson<'doc>,
	emitted: usize,
	done:    bool,
}

impl IncomingString<'_> {
	/// Await the next stable decoded chunk, or `None` after the closing quote.
	pub async fn next_chunk(&mut self) -> Result<Option<Str>, IncomingError> {
		if self.done {
			return Ok(None);
		}
		let located =
			wait_for(&self.json.shared, &self.json.path, WaitMode::Chunk(self.emitted), "string")
				.await?;
		let Kind::String { value, stable_len } = located.kind else {
			return Err(type_mismatch(&self.json.path, "string", located.kind.name()));
		};
		if stable_len > self.emitted {
			let chunk = Str::from(&value[self.emitted..stable_len]);
			self.emitted = stable_len;
			return Ok(Some(chunk));
		}
		self.done = true;
		Ok(None)
	}

	/// Await the closing quote and return the complete decoded string.
	pub async fn finish(self) -> Result<Str, IncomingError> {
		let located =
			wait_for(&self.json.shared, &self.json.path, WaitMode::Complete, "string").await?;
		match located.kind {
			Kind::String { value, .. } => Ok(value),
			other => Err(type_mismatch(&self.json.path, "string", other.name())),
		}
	}
}

/// Linear cursor over elements of an incoming array.
pub struct IncomingArray<'doc> {
	json: IncomingJson<'doc>,
	next: usize,
}

impl IncomingArray<'_> {
	/// Await the start of the next element.
	///
	/// The returned element cursor mutably reborrows this array, so the caller
	/// must consume or cancel it before advancing again. `None` is returned
	/// only after the array's closing bracket.
	pub async fn next(&mut self) -> Result<Option<IncomingJson<'_>>, IncomingError> {
		let root = wait_for(&self.json.shared, &self.json.path, WaitMode::Started, "array").await?;
		if !matches!(root.kind, Kind::Array) {
			return Err(type_mismatch(&self.json.path, "array", root.kind.name()));
		}
		let mut path = self.json.path.clone();
		path.push(PathPart::Index(self.next));
		match wait_for_raw(&self.json.shared, &path, WaitMode::Started, "value").await? {
			Some(_) => {
				self.next += 1;
				Ok(Some(IncomingJson {
					shared: Arc::clone(&self.json.shared),
					path,
					_linear: PhantomData,
				}))
			},
			None => Ok(None),
		}
	}

	/// Await the closing bracket and collect fully parsed elements.
	pub async fn collect(self) -> Result<Vec<Value>, IncomingError> {
		match self.json.value_with("array").await? {
			Value::Array(values) => Ok(values),
			other => Err(type_mismatch(&self.json.path, "array", value_name(&other))),
		}
	}
}

/// Linear cursor for keyed pulls and final collection of an incoming object.
pub struct IncomingObject<'doc> {
	json: IncomingJson<'doc>,
}

impl IncomingObject<'_> {
	/// Return a cursor bound to the first occurrence of `name`.
	///
	/// The returned cursor mutably reborrows this object. Awaiting it resolves
	/// as soon as the key's value starts; consuming or cancelling it permits
	/// the next keyed pull.
	pub fn key(&mut self, name: impl Into<Str>) -> IncomingJson<'_> {
		let mut path = self.json.path.clone();
		path.push(PathPart::Key(name.into()));
		IncomingJson { shared: Arc::clone(&self.json.shared), path, _linear: PhantomData }
	}

	/// Await the closing brace and collect the object.
	///
	/// Final collection uses [`crate::Object`]'s last-write-wins duplicate-key
	/// semantics, unlike [`key`](Self::key), which binds the first occurrence.
	pub async fn collect(self) -> Result<Object, IncomingError> {
		match self.json.value_with("object").await? {
			Value::Object(value) => Ok(value),
			other => Err(type_mismatch(&self.json.path, "object", value_name(&other))),
		}
	}
}

#[derive(Clone, Copy)]
enum WaitMode {
	/// Ready as soon as the value's first token has arrived.
	Started,
	/// Ready only once the value's closing token has arrived.
	Complete,
	/// Ready when a string has stable decoded bytes past this offset, its end
	/// is decided, or it turns out not to be a string at all.
	Chunk(usize),
}

impl WaitMode {
	const fn is_ready(self, located: &Located) -> bool {
		match self {
			Self::Started => true,
			Self::Complete => located.end.is_some(),
			Self::Chunk(emitted) => match located.kind {
				Kind::String { stable_len, .. } => stable_len > emitted || located.end.is_some(),
				_ => true,
			},
		}
	}
}

async fn wait_for(
	shared: &Shared,
	path: &[PathPart],
	mode: WaitMode,
	expected: &'static str,
) -> Result<Located, IncomingError> {
	wait_for_raw(shared, path, mode, expected)
		.await?
		.ok_or_else(|| pull_issue(path, expected, PullIssueKind::Missing))
}

async fn wait_for_raw(
	shared: &Shared,
	path: &[PathPart],
	mode: WaitMode,
	expected: &'static str,
) -> Result<Option<Located>, IncomingError> {
	poll_fn(|cx| {
		let mut state = shared.state.lock();
		let end = state.end;
		match locate(&state.text, path, end == End::Finished) {
			Probe::Located(value) if mode.is_ready(&value) => Poll::Ready(Ok(Some(value))),
			Probe::Located(_) | Probe::Pending => match end {
				End::Finished => {
					Poll::Ready(Err(pull_issue(path, expected, PullIssueKind::Incomplete)))
				},
				End::Aborted => Poll::Ready(Err(pull_issue(path, expected, PullIssueKind::Aborted))),
				End::Open => {
					register_waker(&mut state.wakers, cx.waker());
					Poll::Pending
				},
			},
			Probe::Missing => Poll::Ready(Ok(None)),
			Probe::Type { expected: structural, found } => {
				let expected = if structural == "value" {
					expected
				} else {
					structural
				};
				Poll::Ready(Err(type_mismatch(path, expected, found)))
			},
		}
	})
	.await
}

fn register_waker(wakers: &mut WakerSet, waker: &Waker) {
	if !wakers.iter().any(|registered| registered.will_wake(waker)) {
		wakers.push(waker.clone());
	}
}

fn wake_all(wakers: WakerSet) {
	for waker in wakers {
		waker.wake();
	}
}

fn type_mismatch(path: &[PathPart], expected: &'static str, found: &'static str) -> IncomingError {
	pull_issue(path, expected, PullIssueKind::TypeMismatch { found })
}

fn pull_issue(path: &[PathPart], expected: &'static str, kind: PullIssueKind) -> IncomingError {
	IncomingError::Pull(PullIssue {
		path: path
			.iter()
			.map(|part| match part {
				PathPart::Key(key) => PullPathSegment::Key(key.clone()),
				PathPart::Index(index) => PullPathSegment::Index(*index),
			})
			.collect(),
		expected,
		kind,
	})
}

fn value_name(value: &Value) -> &'static str {
	match value {
		Value::Null => "null",
		Value::Bool(_) => "boolean",
		Value::Number(_) => "number",
		Value::String(_) => "string",
		Value::Array(_) => "array",
		Value::Object(_) => "object",
	}
}

#[derive(Debug)]
struct Located {
	start: usize,
	end:   Option<usize>,
	kind:  Kind,
}

#[derive(Debug)]
enum Kind {
	Null,
	Bool(bool),
	Number(Number),
	String { value: Str, stable_len: usize },
	Array,
	Object,
}

impl Kind {
	const fn name(&self) -> &'static str {
		match self {
			Self::Null => "null",
			Self::Bool(_) => "boolean",
			Self::Number(_) => "number",
			Self::String { .. } => "string",
			Self::Array => "array",
			Self::Object => "object",
		}
	}
}

enum Probe {
	Located(Located),
	Pending,
	Missing,
	Type { expected: &'static str, found: &'static str },
}

fn locate(src: &str, path: &[PathPart], ended: bool) -> Probe {
	let mut parser = Parser::new(src, Mode::Incoming);
	parser.ws();
	select_value(&mut parser, path, ended, 0)
}

fn select_value(parser: &mut Parser<'_>, path: &[PathPart], ended: bool, depth: u32) -> Probe {
	let Some(byte) = parser.peek() else {
		return Probe::Pending;
	};
	if path.is_empty() {
		return scan_value(parser, ended, depth);
	}
	match (&path[0], byte) {
		(PathPart::Key(key), b'{') => select_key(parser, key, &path[1..], ended, depth),
		(PathPart::Index(index), b'[') => select_index(parser, *index, &path[1..], ended, depth),
		(PathPart::Key(_), _) => Probe::Type { expected: "object", found: byte_name(byte) },
		(PathPart::Index(_), _) => Probe::Type { expected: "array", found: byte_name(byte) },
	}
}

fn select_key(
	parser: &mut Parser<'_>,
	wanted: &str,
	rest: &[PathPart],
	ended: bool,
	depth: u32,
) -> Probe {
	if depth >= MAX_DEPTH {
		return Probe::Pending;
	}
	parser.bump();
	loop {
		parser.ws();
		match parser.peek() {
			None => return Probe::Pending,
			Some(b'}') => {
				parser.bump();
				return Probe::Missing;
			},
			Some(b',') => {
				parser.bump();
				continue;
			},
			_ => {},
		}
		let key_matches = match parser.peek() {
			Some(quote @ (b'"' | b'\'')) => {
				let progress = parser
					.string_progress(quote)
					.expect("lenient string never fails");
				if !progress.complete {
					return Probe::Pending;
				}
				<&str>::from(&progress.value) == wanted
			},
			Some(_) => {
				let key = parser.unquoted_key();
				if key.is_empty() {
					return Probe::Pending;
				}
				key == wanted
			},
			None => return Probe::Pending,
		};
		parser.ws();
		if parser.peek() != Some(b':') {
			return Probe::Pending;
		}
		parser.bump();
		parser.ws();
		if parser.at_end() {
			return Probe::Pending;
		}
		if key_matches {
			return select_value(parser, rest, ended, depth + 1);
		}
		match scan_value(parser, ended, depth + 1) {
			Probe::Located(Located { end: Some(_), .. }) => {},
			Probe::Located(_) | Probe::Pending => return Probe::Pending,
			Probe::Missing | Probe::Type { .. } => return Probe::Pending,
		}
		parser.ws();
		match parser.peek() {
			Some(b',') => parser.bump(),
			Some(b'}') => {
				parser.bump();
				return Probe::Missing;
			},
			_ => return Probe::Pending,
		}
	}
}

fn select_index(
	parser: &mut Parser<'_>,
	wanted: usize,
	rest: &[PathPart],
	ended: bool,
	depth: u32,
) -> Probe {
	if depth >= MAX_DEPTH {
		return Probe::Pending;
	}
	parser.bump();
	let mut index = 0;
	loop {
		parser.ws();
		match parser.peek() {
			None => return Probe::Pending,
			Some(b']') => {
				parser.bump();
				return Probe::Missing;
			},
			Some(b',') => {
				parser.bump();
				continue;
			},
			_ => {},
		}
		if index == wanted {
			return select_value(parser, rest, ended, depth + 1);
		}
		match scan_value(parser, ended, depth + 1) {
			Probe::Located(Located { end: Some(_), .. }) => {},
			Probe::Located(_) | Probe::Pending => return Probe::Pending,
			Probe::Missing | Probe::Type { .. } => return Probe::Pending,
		}
		index += 1;
		parser.ws();
		match parser.peek() {
			Some(b',') => parser.bump(),
			Some(b']') => {
				parser.bump();
				return Probe::Missing;
			},
			_ => return Probe::Pending,
		}
	}
}

fn scan_value(parser: &mut Parser<'_>, ended: bool, depth: u32) -> Probe {
	let start = parser.pos();
	let Some(byte) = parser.peek() else {
		return Probe::Pending;
	};
	match byte {
		b'{' => scan_object(parser, ended, depth, start),
		b'[' => scan_array(parser, ended, depth, start),
		quote @ (b'"' | b'\'') => {
			let progress = parser
				.string_progress(quote)
				.expect("lenient string never fails");
			let stable_len = progress.stable_len;
			let value = Str::from(progress.value);
			let end = parser.pos();
			// Like numbers and keywords, a string is complete only once a value
			// terminator follows (or the input ended): an edge close may still
			// be reopened by later fragments via single-quote recovery.
			let complete = progress.complete && scalar_complete(parser, ended);
			Probe::Located(Located {
				start,
				end: complete.then_some(end),
				kind: Kind::String { value, stable_len },
			})
		},
		b'-' | b'+' | b'.' | b'0'..=b'9' => {
			let Ok(Some(number)) = parser.number() else {
				return Probe::Pending;
			};
			let end = parser.pos();
			let complete = scalar_complete(parser, ended);
			Probe::Located(Located { start, end: complete.then_some(end), kind: Kind::Number(number) })
		},
		_ => {
			if let Some(atom) = parser.match_keyword() {
				let end = parser.pos();
				let complete = scalar_complete(parser, ended);
				let kind = match atom {
					crate::parser::Atom::Bool(value) => Kind::Bool(value),
					crate::parser::Atom::Null => Kind::Null,
				};
				Probe::Located(Located { start, end: complete.then_some(end), kind })
			} else if let Ok(word) = parser.bareword() {
				let end = parser.pos();
				let complete = scalar_complete(parser, ended);
				Probe::Located(Located {
					start,
					end: complete.then_some(end),
					kind: Kind::String { value: Str::from(word), stable_len: word.len() },
				})
			} else {
				Probe::Pending
			}
		},
	}
}

fn scalar_complete(parser: &mut Parser<'_>, ended: bool) -> bool {
	parser.ws();
	matches!(parser.peek(), Some(b',' | b'}' | b']')) || (ended && parser.at_end())
}

fn scan_object(parser: &mut Parser<'_>, ended: bool, depth: u32, start: usize) -> Probe {
	if depth >= MAX_DEPTH {
		return Probe::Pending;
	}
	parser.bump();
	loop {
		parser.ws();
		match parser.peek() {
			None => return incomplete_container(start, Kind::Object),
			Some(b'}') => {
				parser.bump();
				return Probe::Located(Located { start, end: Some(parser.pos()), kind: Kind::Object });
			},
			Some(b',') => {
				parser.bump();
				continue;
			},
			_ => {},
		}
		match parser.peek() {
			Some(quote @ (b'"' | b'\'')) => {
				let progress = parser
					.string_progress(quote)
					.expect("lenient string never fails");
				if !progress.complete {
					return incomplete_container(start, Kind::Object);
				}
			},
			Some(_) => {
				if parser.unquoted_key().is_empty() {
					return incomplete_container(start, Kind::Object);
				}
			},
			None => return incomplete_container(start, Kind::Object),
		}
		parser.ws();
		if parser.peek() != Some(b':') {
			return incomplete_container(start, Kind::Object);
		}
		parser.bump();
		parser.ws();
		match scan_value(parser, ended, depth + 1) {
			Probe::Located(Located { end: Some(_), .. }) => {},
			_ => return incomplete_container(start, Kind::Object),
		}
		parser.ws();
		match parser.peek() {
			Some(b',') => parser.bump(),
			Some(b'}') => {
				parser.bump();
				return Probe::Located(Located { start, end: Some(parser.pos()), kind: Kind::Object });
			},
			_ => return incomplete_container(start, Kind::Object),
		}
	}
}

fn scan_array(parser: &mut Parser<'_>, ended: bool, depth: u32, start: usize) -> Probe {
	if depth >= MAX_DEPTH {
		return Probe::Pending;
	}
	parser.bump();
	loop {
		parser.ws();
		match parser.peek() {
			None => return incomplete_container(start, Kind::Array),
			Some(b']') => {
				parser.bump();
				return Probe::Located(Located { start, end: Some(parser.pos()), kind: Kind::Array });
			},
			Some(b',') => {
				parser.bump();
				continue;
			},
			_ => {},
		}
		match scan_value(parser, ended, depth + 1) {
			Probe::Located(Located { end: Some(_), .. }) => {},
			_ => return incomplete_container(start, Kind::Array),
		}
		parser.ws();
		match parser.peek() {
			Some(b',') => parser.bump(),
			Some(b']') => {
				parser.bump();
				return Probe::Located(Located { start, end: Some(parser.pos()), kind: Kind::Array });
			},
			_ => return incomplete_container(start, Kind::Array),
		}
	}
}

fn incomplete_container(start: usize, kind: Kind) -> Probe {
	Probe::Located(Located { start, end: None, kind })
}

fn byte_name(byte: u8) -> &'static str {
	match byte {
		b'{' => "object",
		b'[' => "array",
		b'"' | b'\'' => "string",
		b'-' | b'+' | b'.' | b'0'..=b'9' => "number",
		b't' | b'f' | b'T' | b'F' => "boolean",
		b'n' | b'N' => "null",
		_ => "value",
	}
}

impl fmt::Debug for IncomingJson<'_> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_struct("IncomingJson")
			.field("path_len", &self.path.len())
			.finish_non_exhaustive()
	}
}
