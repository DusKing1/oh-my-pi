//! Transcript loading and live-chain reconstruction.

use std::{fs, path::Path};

use serde_json::value::{RawValue, to_raw_value};

use super::{
	codec::{Error, Header, read_header, read_line},
	event::{Event, Kind},
	raweq::raw_eq,
};

/// One physical event line in a loaded transcript.
#[derive(Debug, Clone)]
pub enum Entry {
	/// A decoded event, including verbatim unknown events.
	Ok(Box<Event>),
	/// A malformed line retained at its physical event index.
	Tombstone(Box<RawValue>),
}
/// Equality is byte equality of stored JSON text, preserving verbatim round
/// trips.
impl PartialEq for Entry {
	fn eq(&self, other: &Self) -> bool {
		match (self, other) {
			(Self::Ok(a), Self::Ok(b)) => a == b,
			(Self::Tombstone(a), Self::Tombstone(b)) => raw_eq(a, b),
			_ => false,
		}
	}
}

impl Eq for Entry {}

/// A loaded transcript with physical event indexes preserved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Log {
	header: Header,
	events: Vec<Entry>,
}

impl Log {
	/// Returns the line-zero identity header.
	#[must_use]
	pub const fn header(&self) -> &Header {
		&self.header
	}

	/// Returns the number of physical event lines, including tombstones.
	#[must_use]
	pub const fn len(&self) -> usize {
		self.events.len()
	}

	/// Returns whether the transcript contains no event lines.
	#[must_use]
	pub const fn is_empty(&self) -> bool {
		self.events.is_empty()
	}

	/// Returns the entry at a physical event index.
	#[must_use]
	pub fn get(&self, index: u64) -> Option<&Entry> {
		usize::try_from(index)
			.ok()
			.and_then(|index| self.events.get(index))
	}

	/// Reconstructs the current live chain with one forward fold.
	///
	/// Ordinary events chain implicitly from the previous event. A rewind
	/// truncates the working chain to its target (or to the root), replacing
	/// the 6.1 million explicit parent pointers that 5,257 rewinds represented
	/// in the measured corpus. Reset begins a new chain boundary. Compact
	/// places its summary before the suffix beginning at `first_kept`, so the
	/// summary stands in for the discarded prefix. Amend and label events
	/// annotate a target but remain on the current chain; they do not navigate
	/// to that target. Tombstones behave as opaque ordinary events so their
	/// indexes remain addressable. No by-id or parent map is built.
	#[must_use]
	pub fn live(&self) -> Vec<u64> {
		let mut live = Vec::new();
		for (index, entry) in self.events.iter().enumerate() {
			let index = u64::try_from(index).expect("event indexes fit in u64");
			match entry {
				Entry::Ok(event) => match event.as_ref() {
					Event { kind: Kind::Item(record), .. } if record.turn_id.is_some() => {},
					Event { kind: Kind::TurnReceipt(receipt), .. } => {
						let complete = receipt.item_events.len() == receipt.outcome.output.len()
							&& receipt.item_events.iter().zip(&receipt.outcome.output).all(
								|(item_index, expected)| {
									matches!(
										self.get(*item_index),
										Some(Entry::Ok(item_event))
											if matches!(
												&item_event.kind,
												Kind::Item(record)
													if record.turn_id.as_ref() == Some(&receipt.turn_id)
														&& &record.item == expected
											)
									)
								},
							);
						if complete {
							live.extend(receipt.item_events.iter().copied());
						}
					},
					Event { kind: Kind::Rewind { to }, .. } => match to {
						None => live.clear(),
						Some(target) => {
							if let Some(position) = live.iter().position(|candidate| candidate == target) {
								live.truncate(position + 1);
							} else {
								live.clear();
								live.push(*target);
							}
						},
					},
					Event { kind: Kind::Reset, .. } => {
						live.clear();
						live.push(index);
					},
					Event { kind: Kind::Compact { first_kept, .. }, .. } => {
						if let Some(position) = live.iter().position(|candidate| candidate == first_kept)
						{
							live.rotate_left(position);
							live.truncate(live.len() - position);
							live.insert(0, index);
						} else {
							live.clear();
							live.push(index);
						}
					},
					Event { kind: Kind::PromptRewriteIntent(_) | Kind::PromptRewriteStage(_), .. } => {},
					Event { kind: Kind::PromptRewriteCommit(commit), .. } => {
						let Some(Entry::Ok(intent_event)) = self.get(commit.intent) else {
							continue;
						};
						let Kind::PromptRewriteIntent(intent) = &intent_event.kind else {
							continue;
						};
						if commit.head_events.len() != intent.head.len() {
							continue;
						}
						let complete = commit.head_events.iter().enumerate().all(
							|(ordinal, stage_index)| {
								matches!(
									self.get(*stage_index),
									Some(Entry::Ok(stage_event))
										if matches!(
											&stage_event.kind,
											Kind::PromptRewriteStage(stage)
												if stage.intent == commit.intent
													&& stage.ordinal == ordinal as u64
													&& stage.item == intent.head[ordinal]
										)
								)
							},
						);
						if complete {
							live.clear();
							live.extend(commit.head_events.iter().copied());
							live.extend(intent.preserved_tail.iter().copied());
						}
					},
					_ => live.push(index),
				},
				Entry::Tombstone(_) => live.push(index),
			}
		}
		live
	}
}

/// Loads a transcript while preserving every physical event index.
pub fn load(path: &Path) -> Result<Log, Error> {
	let bytes = fs::read(path)?;
	if bytes.is_empty() {
		return Err(Error::MissingHeader);
	}
	let (header_line, event_bytes) = match bytes.iter().position(|byte| *byte == b'\n') {
		Some(end) => (&bytes[..end], &bytes[end + 1..]),
		None => (&bytes[..], &[][..]),
	};
	let header = read_header(header_line)?;
	let mut events = Vec::new();
	let mut start = 0;
	for end in event_bytes
		.iter()
		.enumerate()
		.filter_map(|(index, byte)| (*byte == b'\n').then_some(index))
	{
		push_entry(&mut events, &event_bytes[start..end]);
		start = end + 1;
	}
	if start < event_bytes.len() {
		push_entry(&mut events, &event_bytes[start..]);
	}
	Ok(Log { header, events })
}

fn push_entry(events: &mut Vec<Entry>, line: &[u8]) {
	if let Ok(event) = read_line(line) {
		events.push(Entry::Ok(Box::new(event)));
	} else {
		let source = String::from_utf8_lossy(line);
		let raw = to_raw_value(source.as_ref()).expect("a JSON string is always serializable");
		events.push(Entry::Tombstone(raw));
	}
}
