//! Durable append-only journal operations for canonical agent turns.

use std::{
	collections::BTreeMap,
	path::{Path, PathBuf},
};

use omp_core::Str;
use omp_proto::thread::v1::Item;
pub use omp_storage::transcript::TurnReceipt;
use omp_storage::transcript::{self, AmendPatch, Event, Header, ItemRecord, Kind, Log, Writer};
use thiserror::Error;

use crate::prompt::PromptHash;
/// Journal append or validation failure.
#[derive(Debug, Error)]
pub enum JournalError {
	/// Transcript storage failed.
	#[error(transparent)]
	Storage(#[from] transcript::Error),
	/// A sequence amendment targeted a non-item event.
	#[error("sequence amendment target {0} is not a canonical item")]
	InvalidItemTarget(u64),
	/// Gateway-assigned item sequences begin at one.
	#[error("gateway sequence must be nonzero")]
	ZeroSequence,
	/// A resumed turn replay did not match its already journaled prefix.
	#[error("replayed outcome for turn {0} differs from its durable prefix")]
	TurnReplayMismatch(Str),
}

/// Append-only transcript owner with an in-memory terminal-turn index.
pub struct Journal {
	path:       PathBuf,
	writer:     Writer,
	receipts:   BTreeMap<Str, TurnReceipt>,
	pending:    BTreeMap<Str, Vec<(u64, Item, Option<[u8; 32]>)>>,
	item_count: u64,
}

impl Journal {
	/// Creates an empty transcript-v4 journal.
	pub fn create(path: &Path, header: &Header) -> Result<Self, JournalError> {
		let writer = Writer::create(path, header)?;
		Ok(Self {
			path: path.to_owned(),
			writer,
			receipts: BTreeMap::new(),
			pending: BTreeMap::new(),
			item_count: 0,
		})
	}

	/// Opens an existing transcript and restores terminal turn receipts.
	pub fn open(path: &Path) -> Result<Self, JournalError> {
		let log = transcript::load(path)?;
		let mut receipts = BTreeMap::new();
		let mut pending: BTreeMap<Str, Vec<(u64, Item, Option<[u8; 32]>)>> = BTreeMap::new();
		let mut item_count = 0_u64;
		for index in 0..u64::try_from(log.len()).expect("transcript length fits in u64") {
			let Some(transcript::Entry::Ok(event)) = log.get(index) else {
				continue;
			};
			match &event.kind {
				Kind::Item(record) => {
					item_count = item_count.saturating_add(1);
					if let Some(turn_id) = &record.turn_id {
						pending.entry(turn_id.clone()).or_default().push((
							index,
							record.item.clone(),
							record.prompt_hash,
						));
					}
				},
				Kind::TurnReceipt(receipt) => {
					receipts.insert(receipt.turn_id.clone(), receipt.clone());
				},
				_ => {},
			}
		}
		pending.retain(|turn_id, _| !receipts.contains_key(turn_id));
		let writer = Writer::open_append(path)?;
		Ok(Self { path: path.to_owned(), writer, receipts, pending, item_count })
	}

	/// Reloads the durable journal for pure projection.
	pub fn load(&self) -> Result<Log, JournalError> {
		Ok(transcript::load(&self.path)?)
	}

	/// Appends one local item optimistically with sequence zero.
	pub fn append_optimistic(
		&mut self,
		ts: u64,
		mut item: Item,
		prompt_hash: Option<PromptHash>,
	) -> Result<u64, JournalError> {
		item.seq = 0;
		let index = self.writer.append(&Event {
			ts,
			kind: Kind::Item(ItemRecord {
				item,
				turn_id: None,
				prompt_hash: prompt_hash.map(PromptHash::into_bytes),
			}),
		})?;
		self.item_count = self.item_count.saturating_add(1);
		Ok(index)
	}

	/// Appends every item from a completed turn and then its terminal receipt.
	///
	/// A turn identifier with an existing terminal receipt is an exact replay:
	/// no item or receipt line is appended and the existing receipt is returned.
	///
	/// A durable unterminated prefix is compared item-for-item and resumed
	/// without duplicating its already appended lines.
	pub fn append_outcome<I>(
		&mut self,
		ts: u64,
		turn_id: &str,
		prompt_hash: Option<PromptHash>,
		items: I,
	) -> Result<(TurnReceipt, bool), JournalError>
	where
		I: IntoIterator<Item = Item>,
	{
		if let Some(receipt) = self.receipts.get(turn_id) {
			return Ok((receipt.clone(), true));
		}
		let existing = self.pending.remove(turn_id).unwrap_or_default();
		let prompt_hash = prompt_hash.map(PromptHash::into_bytes);

		let turn_id = Str::from(turn_id);
		let mut item_events = Vec::with_capacity(existing.len());
		item_events.extend(existing.iter().map(|(index, ..)| *index));
		let mut replayed = 0_usize;
		let mut mismatch = false;
		for (position, item) in items.into_iter().enumerate() {
			replayed = position.saturating_add(1);
			if let Some((_, durable, durable_hash)) = existing.get(position) {
				if durable != &item || durable_hash != &prompt_hash {
					mismatch = true;
					break;
				}
				continue;
			}
			let index = self.writer.append(&Event {
				ts,
				kind: Kind::Item(ItemRecord { item, turn_id: Some(turn_id.clone()), prompt_hash }),
			})?;
			item_events.push(index);
			self.item_count = self.item_count.saturating_add(1);
		}
		if mismatch || replayed < existing.len() {
			self.pending.insert(turn_id.clone(), existing);
			return Err(JournalError::TurnReplayMismatch(turn_id));
		}
		let receipt = TurnReceipt { turn_id: turn_id.clone(), item_events };
		self
			.writer
			.append(&Event { ts, kind: Kind::TurnReceipt(receipt.clone()) })?;
		self.receipts.insert(turn_id, receipt.clone());
		Ok((receipt, false))
	}

	/// Appends a later event assigning a gateway sequence to an item event.
	pub fn amend_seq(&mut self, ts: u64, target: u64, seq: u64) -> Result<u64, JournalError> {
		if seq == 0 {
			return Err(JournalError::ZeroSequence);
		}
		let log = self.load()?;
		if !matches!(log.get(target), Some(transcript::Entry::Ok(event)) if matches!(&event.kind, Kind::Item(_)))
		{
			return Err(JournalError::InvalidItemTarget(target));
		}
		Ok(self
			.writer
			.append(&Event { ts, kind: Kind::Amend { target, patch: AmendPatch::Seq { seq } } })?)
	}

	/// Returns whether a turn has a terminal durable receipt.
	#[must_use]
	pub fn contains_turn(&self, turn_id: &str) -> bool {
		self.receipts.contains_key(turn_id)
	}

	/// Returns the number of canonical item events observed by this writer.
	#[must_use]
	pub const fn item_count(&self) -> u64 {
		self.item_count
	}
}
