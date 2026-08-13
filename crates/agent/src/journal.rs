//! Durable append-only journal operations for canonical agent turns.

use std::{
	collections::{BTreeMap, BTreeSet, VecDeque},
	path::{Path, PathBuf},
};

use omp_core::Str;
use omp_proto::{inference::v1::Outcome, thread::v1::Item};
use omp_tool::{Abort, JobRef};
pub use omp_storage::transcript::{TurnInputRecord, TurnOptionsRecord, TurnReceipt, TurnStart};
use omp_storage::transcript::{
	self, AmendPatch, Event, Header, ItemRecord, JobRegistered, JobSettled, Kind, Log,
	PromptRewriteCommit, PromptRewriteIntent, PromptRewriteStage, ToolBatchAuthorized,
	TurnInputItem, Writer,
};
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
	/// A logical turn was started again with a different prompt identity.
	#[error("turn start for {0} changed its durable prompt identity")]
	TurnStartMismatch(Str),
	/// One optimistic item was claimed by two live logical turns.
	#[error("journal item {target} is already claimed by live turn {turn_id}")]
	ItemAlreadyClaimed {
		/// Physical item event index.
		target:  u64,
		/// Existing logical turn.
		turn_id: Str,
	},
	/// A turn start referenced an absent or non-item event.
	#[error("turn start references non-item event {0}")]
	InvalidTurnInput(u64),
	/// A terminal gateway outcome arrived without a durable turn start.
	#[error("gateway outcome for turn {0} has no durable turn start")]
	MissingTurnStart(Str),
	/// A durable receipt revision cannot assign its recorded input sequences.
	#[error("turn receipt for {0} has an invalid sequence range")]
	InvalidSequenceRange(Str),
	/// A recorded sequence amendment disagrees with its authoritative receipt.
	#[error("sequence amendment for event {target} is {actual}, expected {expected}")]
	SequenceReplayMismatch {
		/// Patched item event.
		target:   u64,
		/// Durable amendment value.
		actual:   u64,
		/// Receipt-derived value.
		expected: u64,
	},
	/// A prompt rewrite contained missing, duplicate, or mismatched stages.
	#[error("prompt rewrite intent {0} is corrupt")]
	CorruptPromptRewrite(u64),
	/// A repeated detached-job event disagreed with its durable record.
	#[error("detached job {0} replay differs from durable truth")]
	JobReplayMismatch(Str),
	/// A repeated tool-batch authorization disagreed with durable truth.
	#[error("tool batch authorization for turn {0} changed")]
	ToolBatchReplayMismatch(Str),
	/// A tool batch references a turn without its durable terminal receipt.
	#[error("tool batch authorization for turn {0} has no receipt")]
	ToolBatchWithoutReceipt(Str),
	/// An authorization named a call absent from its turn receipt.
	#[error("tool call {call_id} is absent from receipt {turn_id}")]
	UnknownReceiptCall {
		/// Gateway turn identifier.
		turn_id: Str,
		/// Missing call identifier.
		call_id: Str,
	},
	/// Canonical recovery-result construction failed.
	#[error(transparent)]
	Projection(#[from] crate::ProjectionError),
}

/// Append-only transcript owner with an in-memory terminal-turn index.
pub struct Journal {
	path:         PathBuf,
	writer:       Writer,
	receipts:     BTreeMap<Str, TurnReceipt>,
	starts:       BTreeMap<Str, (u64, TurnStart)>,
	claims:       BTreeMap<u64, Str>,
	last_start:   Option<TurnStart>,
	last_receipt: Option<TurnReceipt>,
	active_prompt: Option<([u8; 32], Vec<u64>)>,
	pending:      BTreeMap<Str, Vec<(u64, Item, Option<[u8; 32]>)>>,
	pending_jobs: BTreeMap<Str, (u64, JobRef)>,
	settled_jobs: BTreeMap<Str, (u64, Item)>,
	authorized_batches: BTreeMap<Str, (u64, Vec<Str>)>,
	recoverable_settlements: Vec<u64>,
	pending_inputs: VecDeque<(Str, Vec<u64>)>,
	item_count:   u64,
}

impl Journal {
	/// Creates an empty transcript-v4 journal.
	pub fn create(path: &Path, header: &Header) -> Result<Self, JournalError> {
		let writer = Writer::create(path, header)?;
		Ok(Self {
			path: path.to_owned(),
			writer,
			receipts: BTreeMap::new(),
			starts: BTreeMap::new(),
			claims: BTreeMap::new(),
			last_start: None,
			last_receipt: None,
			active_prompt: None,
			pending: BTreeMap::new(),
			pending_jobs: BTreeMap::new(),
			authorized_batches: BTreeMap::new(),
			recoverable_settlements: Vec::new(),
			pending_inputs: VecDeque::new(),
			settled_jobs: BTreeMap::new(),
			item_count: 0,
		})
	}

	/// Opens an existing transcript and restores terminal turn receipts.
	pub fn open(path: &Path) -> Result<Self, JournalError> {
		let log = transcript::load(path)?;
		let mut receipts = BTreeMap::new();
		let mut starts: BTreeMap<Str, (u64, TurnStart)> = BTreeMap::new();
		let mut last_start = None;
		let mut pending: BTreeMap<Str, Vec<(u64, Item, Option<[u8; 32]>)>> = BTreeMap::new();
		let mut pending_jobs = BTreeMap::new();
		let mut settled_jobs = BTreeMap::new();
		let mut item_count = 0_u64;
		let mut last_receipt = None;
		let mut authorized_batches = BTreeMap::new();
		let mut turn_inputs = BTreeMap::<Str, Vec<u64>>::new();
		let mut turn_input_order = Vec::new();
		let mut started_turns = BTreeSet::new();
		let mut claimed_ever = BTreeSet::new();
		let mut settled_input_events = Vec::new();
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
				Kind::TurnInput(input) => {
					item_count = item_count.saturating_add(1);
					if !turn_inputs.contains_key(input.turn_id.as_str()) {
						turn_input_order.push(input.turn_id.clone());
					}
					turn_inputs.entry(input.turn_id.clone()).or_default().push(index);
				},
				Kind::PromptRewriteStage(_) => {
					item_count = item_count.saturating_add(1);
				},
				Kind::TurnStart(start) => {
					starts.insert(start.turn_id.clone(), (index, start.clone()));
					last_start = Some(start.clone());
					started_turns.insert(start.turn_id.clone());
					claimed_ever.extend(start.item_events.iter().copied());
				},
				Kind::TurnReceipt(receipt) => {
					receipts.insert(receipt.turn_id.clone(), receipt.clone());
					last_receipt = Some(receipt.clone());
				},
				Kind::JobRegistered(registered) => {
					if !settled_jobs.contains_key(registered.job.id.as_str()) {
						pending_jobs.insert(
							registered.job.id.clone(),
							(index, registered.job.clone()),
						);
					}
				},
				Kind::JobSettled(settled) => {
					pending_jobs.remove(settled.job_id.as_str());
					item_count = item_count.saturating_add(1);
					settled_jobs
						.insert(settled.job_id.clone(), (index, settled.settlement.clone()));
					settled_input_events.push(index);
				},
				Kind::ToolBatchAuthorized(batch) => {
					authorized_batches.insert(batch.turn_id.clone(), (index, batch.call_ids.clone()));
				},
				_ => {},
			}
		}
		starts.retain(|turn_id, _| !receipts.contains_key(turn_id));
		let mut claims = BTreeMap::new();
		for (turn_id, (_, start)) in &starts {
			for target in &start.item_events {
				if !matches!(log.get(*target), Some(transcript::Entry::Ok(event)) if event_item(&event.kind).is_some())
				{
					return Err(JournalError::InvalidTurnInput(*target));
				}
				if let Some(existing) = claims.insert(*target, turn_id.clone())
					&& existing != *turn_id
				{
					return Err(JournalError::ItemAlreadyClaimed {
						target:  *target,
						turn_id: existing,
					});
				}
			}
			for target in start.prompt_head_events.iter().chain(&start.sequence_targets) {
				if !matches!(log.get(*target), Some(transcript::Entry::Ok(event)) if event_item(&event.kind).is_some())
				{
					return Err(JournalError::InvalidTurnInput(*target));
				}
			}
		}
		pending.retain(|turn_id, _| !receipts.contains_key(turn_id));
		for turn_id in started_turns.iter().chain(receipts.keys()) {
			turn_inputs.remove(turn_id.as_str());
		}
		let mut writer = Writer::open_append(path)?;
		let (recovered_items, active_prompt) = recover_prompt_rewrites(&log, &mut writer)?;
		item_count = item_count.saturating_add(recovered_items);
		recover_sequence_amendments(&log, &mut writer)?;
		let recovered_batches = recover_tool_batches(&log, &mut writer)?;
		item_count = item_count.saturating_add(
			u64::try_from(recovered_batches.len()).expect("recovered batch length fits in u64"),
		);
		for (turn_id, index) in recovered_batches {
			if !turn_inputs.contains_key(turn_id.as_str()) {
				turn_input_order.push(turn_id.clone());
			}
			turn_inputs.entry(turn_id).or_default().push(index);
		}
		let pending_inputs = turn_input_order
			.into_iter()
			.filter_map(|turn_id| turn_inputs.remove(turn_id.as_str()).map(|events| (turn_id, events)))
			.collect();
		let recoverable_settlements = settled_input_events
			.into_iter()
			.filter(|index| !claimed_ever.contains(index))
			.collect();
		Ok(Self {
			path: path.to_owned(),
			writer,
			receipts,
			starts,
			claims,
			last_start,
			last_receipt,
			active_prompt,
			pending,
			pending_jobs,
			settled_jobs,
			authorized_batches,
			recoverable_settlements,
			pending_inputs,
			item_count,
		})
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

	/// Stages one canonical input under the logical turn that must submit it.
	pub fn append_turn_input(
		&mut self,
		ts: u64,
		turn_id: &str,
		mut item: Item,
		prompt_hash: Option<PromptHash>,
	) -> Result<u64, JournalError> {
		item.seq = 0;
		let turn_id = Str::from(turn_id);
		let index = self.writer.append(&Event {
			ts,
			kind: Kind::TurnInput(TurnInputItem {
				turn_id: turn_id.clone(),
				item,
				prompt_hash: prompt_hash.map(PromptHash::into_bytes),
			}),
		})?;
		self.item_count = self.item_count.saturating_add(1);
		if let Some((_, events)) = self
			.pending_inputs
			.iter_mut()
			.find(|(durable_turn_id, _)| durable_turn_id == &turn_id)
		{
			events.push(index);
		} else {
			self.pending_inputs.push_back((turn_id, vec![index]));
		}
		Ok(index)
	}

	/// Atomically replaces the system-prompt head while retaining an ordered tail.
	///
	/// The intent and hidden stages do not change the live chain. Only the final
	/// commit publishes `[new head, preserved tail]`; reopening the journal
	/// idempotently completes an interrupted materialization.
	pub fn rewrite_prompt_head(
		&mut self,
		ts: u64,
		prompt_hash: PromptHash,
		head: &[Item],
		preserved_tail: &[u64],
	) -> Result<Vec<u64>, JournalError> {
		let live = self.live_item_events()?;
		for target in preserved_tail {
			if !live.contains(target) {
				return Err(JournalError::InvalidTurnInput(*target));
			}
		}
		let intent = PromptRewriteIntent {
			prompt_hash: prompt_hash.into_bytes(),
			head: head.to_vec(),
			preserved_tail: preserved_tail.to_vec(),
		};
		let intent_event =
			self.writer.append(&Event { ts, kind: Kind::PromptRewriteIntent(intent) })?;
		let mut head_events = Vec::with_capacity(head.len());
		for (ordinal, item) in head.iter().enumerate() {
			let stage = PromptRewriteStage {
				intent: intent_event,
				ordinal: u64::try_from(ordinal).expect("prompt head length fits in u64"),
				item: item.clone(),
			};
			head_events
				.push(self.writer.append(&Event { ts, kind: Kind::PromptRewriteStage(stage) })?);
			self.item_count = self.item_count.saturating_add(1);
		}
		self.writer.append(&Event {
			ts,
			kind: Kind::PromptRewriteCommit(PromptRewriteCommit {
				intent: intent_event,
				head_events: head_events.clone(),
			}),
		})?;
		self.active_prompt = Some((prompt_hash.into_bytes(), head_events.clone()));
		Ok(head_events)
	}



	/// Durably fixes a logical turn before its transport is opened.
	///
	/// Re-recording identical metadata is idempotent. Conflict and NeedFull
	/// recovery may supersede only the input envelope and claimed item set; the
	/// logical turn identity and prompt identity remain fixed.
	pub fn start_turn(&mut self, ts: u64, start: TurnStart) -> Result<u64, JournalError> {
		if let Some(receipt) = self.receipts.get(start.turn_id.as_str()) {
			return Ok(receipt.item_events.last().copied().unwrap_or_default());
		}
		if let Some((index, durable)) = self.starts.get(start.turn_id.as_str()) {
			if durable == &start {
				return Ok(*index);
			}
			if durable.prompt_hash != start.prompt_hash
				|| durable.prompt_head_events != start.prompt_head_events
				|| durable.toolset_hash != start.toolset_hash
				|| durable.enabled_tools != start.enabled_tools
			{
				return Err(JournalError::TurnStartMismatch(start.turn_id));
			}
		}

		let log = self.load()?;
		for target in start
			.item_events
			.iter()
			.chain(&start.prompt_head_events)
			.chain(&start.sequence_targets)
		{
			if !matches!(log.get(*target), Some(transcript::Entry::Ok(event)) if event_item(&event.kind).is_some())
			{
				return Err(JournalError::InvalidTurnInput(*target));
			}
		}
		for target in &start.item_events {
			if let Some(turn_id) = self.claims.get(target)
				&& turn_id != &start.turn_id
			{
				return Err(JournalError::ItemAlreadyClaimed {
					target:  *target,
					turn_id: turn_id.clone(),
				});
			}
		}
		if let Some((_, durable)) = self.starts.get(start.turn_id.as_str()) {
			for target in &durable.item_events {
				self.claims.remove(target);
			}
		}
		let index = self
			.writer
			.append(&Event { ts, kind: Kind::TurnStart(start.clone()) })?;
		for target in &start.item_events {
			self.claims.insert(*target, start.turn_id.clone());
		}
		self
			.recoverable_settlements
			.retain(|target| !start.item_events.contains(target));
		if let Some(position) = self
			.pending_inputs
			.iter()
			.position(|(turn_id, _)| turn_id == &start.turn_id)
		{
			self.pending_inputs.remove(position);
		}
		self.last_start = Some(start.clone());
		self.starts.insert(start.turn_id.clone(), (index, start));
		Ok(index)
	}

	/// Returns the earliest live turn start that lacks a terminal receipt.
	pub fn pending_turn(&self) -> Option<&TurnStart> {
		self
			.starts
			.values()
			.min_by_key(|(index, _)| *index)
			.map(|(_, start)| start)
	}

	/// Returns durable start metadata for one unmatched logical turn.
	pub fn turn_start(&self, turn_id: &str) -> Option<&TurnStart> {
		self.starts.get(turn_id).map(|(_, start)| start)
	}

	/// Clones the canonical items at ordered physical item-event indexes.
	pub fn items_at(&self, targets: &[u64]) -> Result<Vec<Item>, JournalError> {
		let log = self.load()?;
		targets
			.iter()
			.map(|target| match log.get(*target) {
				Some(transcript::Entry::Ok(event)) => event_item(&event.kind)
					.cloned()
					.ok_or(JournalError::InvalidTurnInput(*target)),
				_ => Err(JournalError::InvalidTurnInput(*target)),
			})
			.collect()
	}

	/// Returns metadata from the most recently opened logical turn.
	pub fn latest_turn_start(&self) -> Option<&TurnStart> {
		self.last_start.as_ref()
	}

	/// Returns every live canonical item event in projection order.
	pub fn live_item_events(&self) -> Result<Vec<u64>, JournalError> {
		let log = self.load()?;
		Ok(log
			.live()
			.into_iter()
			.filter(|index| {
				matches!(log.get(*index), Some(transcript::Entry::Ok(event)) if event_item(&event.kind).is_some())
			})
			.collect())
	}

	/// Appends an authoritative gateway outcome and its terminal receipt.
	///
	/// Prompt identity and head boundaries come from the durable [`TurnStart`],
	/// never from mutable caller state. Replaying an existing receipt succeeds
	/// only when the complete canonical outcome is field-exact; it appends no
	/// duplicate items or receipt.
	pub fn append_gateway_outcome(
		&mut self,
		ts: u64,
		turn_id: &str,
		outcome: Outcome,
	) -> Result<(TurnReceipt, bool), JournalError> {
		if let Some(receipt) = self.receipts.get(turn_id) {
			if receipt.outcome != outcome {
				return Err(JournalError::TurnReplayMismatch(Str::from(turn_id)));
			}
			return Ok((receipt.clone(), true));
		}

		let turn_id = Str::from(turn_id);
		let Some((_, start)) = self.starts.get(turn_id.as_str()).cloned() else {
			return Err(JournalError::MissingTurnStart(turn_id));
		};
		let existing = self.pending.get(turn_id.as_str()).cloned().unwrap_or_default();
		let prompt_hash = Some(start.prompt_hash);
		let mut item_events = Vec::with_capacity(outcome.output.len());
		item_events.extend(existing.iter().map(|(index, ..)| *index));
		let mut replayed = 0_usize;
		let mut mismatch = false;
		for (position, item) in outcome.output.iter().enumerate() {
			replayed = position.saturating_add(1);
			if let Some((_, durable, durable_hash)) = existing.get(position) {
				if durable != item || durable_hash != &prompt_hash {
					mismatch = true;
					break;
				}
				continue;
			}
			let index = self.writer.append(&Event {
				ts,
				kind: Kind::Item(ItemRecord {
					item: item.clone(),
					turn_id: Some(turn_id.clone()),
					prompt_hash,
				}),
			})?;
			item_events.push(index);
			self.item_count = self.item_count.saturating_add(1);
			self
				.pending
				.entry(turn_id.clone())
				.or_default()
				.push((index, item.clone(), prompt_hash));
		}
		if mismatch || replayed < existing.len() {

			return Err(JournalError::TurnReplayMismatch(turn_id));
		}
		let receipt = TurnReceipt {
			turn_id: turn_id.clone(),
			prompt_hash: start.prompt_hash,
			prompt_head_events: start.prompt_head_events,
			item_events,
			outcome,
		};

		self
			.writer
			.append(&Event { ts, kind: Kind::TurnReceipt(receipt.clone()) })?;
		self.pending.remove(turn_id.as_str());
		self.starts.remove(turn_id.as_str());
		self.claims.retain(|_, claimed| claimed != &turn_id);
		self.last_receipt = Some(receipt.clone());
		self.receipts.insert(turn_id, receipt.clone());
		Ok((receipt, false))
	}
	/// Durably authorizes one committed tool batch before any effect may start.
	pub fn authorize_tool_batch(
		&mut self,
		ts: u64,
		turn_id: &str,
		call_ids: &[Str],
	) -> Result<u64, JournalError> {
		if let Some((index, durable)) = self.authorized_batches.get(turn_id) {
			if durable == call_ids {
				return Ok(*index);
			}
			return Err(JournalError::ToolBatchReplayMismatch(Str::from(turn_id)));
		}
		let Some(receipt) = self.receipts.get(turn_id) else {
			return Err(JournalError::ToolBatchWithoutReceipt(Str::from(turn_id)));
		};
		for call_id in call_ids {
			let present = receipt.outcome.output.iter().any(|item| {
				matches!(
					item.kind.as_ref(),
					Some(omp_proto::thread::v1::item::Kind::ToolCall(call))
						if call.id == call_id.as_str()
				)
			});
			if !present {
				return Err(JournalError::UnknownReceiptCall {
					turn_id: Str::from(turn_id),
					call_id: call_id.clone(),
				});
			}
		}
		let batch = ToolBatchAuthorized {
			turn_id: Str::from(turn_id),
			call_ids: call_ids.to_vec(),
		};
		let index =
			self.writer.append(&Event { ts, kind: Kind::ToolBatchAuthorized(batch.clone()) })?;
		self
			.authorized_batches
			.insert(batch.turn_id, (index, batch.call_ids));
		Ok(index)
	}
	/// Durably registers detached work for restart-safe settlement watching.
	///
	/// Re-registering the exact same job is idempotent. A job already settled
	/// remains terminal and is not made pending again.
	pub fn register_job(&mut self, ts: u64, job: JobRef) -> Result<u64, JournalError> {
		if let Some((index, _)) = self.settled_jobs.get(job.id.as_str()) {
			return Ok(*index);
		}
		if let Some((index, durable)) = self.pending_jobs.get(job.id.as_str()) {
			if durable != &job {
				return Err(JournalError::JobReplayMismatch(job.id));
			}
			return Ok(*index);
		}
		let index = self.writer.append(&Event {
			ts,
			kind: Kind::JobRegistered(JobRegistered { job: job.clone() }),
		})?;
		self.pending_jobs.insert(job.id.clone(), (index, job));
		Ok(index)
	}

	/// Durably records one canonical detached-job settlement.
	///
	/// Duplicate identical settlements are idempotent; differing duplicates are
	/// rejected without appending another line.
	pub fn settle_job(
		&mut self,
		ts: u64,
		job_id: &str,
		settlement: Item,
	) -> Result<u64, JournalError> {
		if let Some((index, durable)) = self.settled_jobs.get(job_id) {
			if durable != &settlement {
				return Err(JournalError::JobReplayMismatch(Str::from(job_id)));
			}
			return Ok(*index);
		}
		let job_id = Str::from(job_id);
		let index = self.writer.append(&Event {
			ts,
			kind: Kind::JobSettled(JobSettled {
				job_id: job_id.clone(),
				settlement: settlement.clone(),
			}),

		})?;
		self.item_count = self.item_count.saturating_add(1);
		self.pending_jobs.remove(job_id.as_str());
		self.settled_jobs.insert(job_id, (index, settlement));
		self.recoverable_settlements.push(index);
		Ok(index)
	}
	/// Returns unclaimed durable settlement or recovered-result item events.
	#[must_use]
	pub fn recoverable_input_events(&self) -> &[u64] {
		self.pending_inputs.front().map_or(&[], |(_, events)| events.as_slice())
	}

	/// Returns unclaimed durable detached-job settlement event IDs.
	#[must_use]
	pub fn recoverable_settlement_events(&self) -> &[u64] {
		&self.recoverable_settlements
	}

	/// Returns the earliest staged input whose turn transport never opened.
	#[must_use]
	pub fn pending_input_submission(&self) -> Option<(&Str, &[u64])> {
		self
			.pending_inputs
			.front()
			.map(|(turn_id, events)| (turn_id, events.as_slice()))
	}

	/// Iterates detached jobs still awaiting settlement without allocating.
	pub fn pending_jobs(&self) -> impl Iterator<Item = &JobRef> {
		self.pending_jobs.values().map(|(_, job)| job)
	}


	/// Appends a later event assigning a gateway sequence to an item event.
	pub fn amend_seq(&mut self, ts: u64, target: u64, seq: u64) -> Result<u64, JournalError> {
		if seq == 0 {
			return Err(JournalError::ZeroSequence);
		}
		let log = self.load()?;
		if !matches!(log.get(target), Some(transcript::Entry::Ok(event)) if event_item(&event.kind).is_some())
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


	/// Returns the authoritative committed prompt identity and head event IDs.
	#[must_use]
	pub fn active_prompt(&self) -> Option<([u8; 32], &[u64])> {
		self
			.active_prompt
			.as_ref()
			.map(|(hash, events)| (*hash, events.as_slice()))
	}

	/// Returns the most recently appended terminal receipt in physical order.
	#[must_use]
	pub fn latest_receipt(&self) -> Option<&TurnReceipt> {
		self.last_receipt.as_ref()
	}

	/// Returns the durable terminal receipt for one logical turn.
	#[must_use]
	pub fn receipt(&self, turn_id: &str) -> Option<&TurnReceipt> {
		self.receipts.get(turn_id)
	}

	/// Returns the number of canonical item events observed by this writer.
	#[must_use]
	pub const fn item_count(&self) -> u64 {
		self.item_count
	}
}

fn recover_tool_batches(
	log: &Log,
	writer: &mut Writer,
) -> Result<Vec<(Str, u64)>, JournalError> {
	let mut results = BTreeMap::<Str, Option<Str>>::new();
	let mut authorized = BTreeMap::<Str, BTreeSet<Str>>::new();
	let mut receipts = Vec::new();
	for index in 0..u64::try_from(log.len()).expect("transcript length fits in u64") {
		let Some(transcript::Entry::Ok(event)) = log.get(index) else {
			continue;
		};
		if let Some(item) = event_item(&event.kind)
			&& let Some(omp_proto::thread::v1::item::Kind::ToolResult(result)) = item.kind.as_ref()
		{
			let recovery_turn = match &event.kind {
				Kind::TurnInput(input) => Some(input.turn_id.clone()),
				_ => None,
			};
			results.entry(Str::from(result.call_id.as_str())).or_insert(recovery_turn);
		}
		match &event.kind {
			Kind::ToolBatchAuthorized(batch) => {
				authorized.insert(batch.turn_id.clone(), batch.call_ids.iter().cloned().collect());
			},
			Kind::TurnReceipt(receipt)
				if receipt.outcome.stop
					== omp_proto::inference::v1::StopReason::StopToolUse as i32 =>
			{
				receipts.push((event.ts, receipt));
			},
			_ => {},
		}
	}

	let mut recovered = Vec::new();
	for (ts, receipt) in receipts {
		let calls: Vec<_> = receipt
			.outcome
			.output
			.iter()
			.filter_map(|item| match item.kind.as_ref() {
				Some(omp_proto::thread::v1::item::Kind::ToolCall(call)) => Some((item, call)),
				_ => None,
			})
			.collect();
		let mut recovery_turn = calls
			.iter()
			.find_map(|(_, call)| results.get(call.id.as_str()).and_then(Clone::clone));
		let authorized_calls = authorized.get(receipt.turn_id.as_str());
		for (item, call) in calls {
			let call_id = Str::from(call.id.as_str());
			if results.contains_key(&call_id) {
				continue;
			}
			let abort = if authorized_calls.is_some_and(|calls| calls.contains(&call_id)) {
				Abort::EffectsUnknown {
					reason: Str::new_static("agent restarted after invocation authorization"),
				}
			} else {
				Abort::Skipped {
					reason: Str::new_static("agent restarted before invocation authorization"),
				}
			};
			let result = crate::project::recovery_tool_result_item(ts, item, abort)?;
			let recovery_turn = recovery_turn
				.get_or_insert_with(|| Str::from(ulid::Ulid::generate().to_string()));
			let index = writer.append(&Event {
				ts,
				kind: Kind::TurnInput(TurnInputItem {
					turn_id: recovery_turn.clone(),
					item: result,
					prompt_hash: Some(receipt.prompt_hash),
				}),
			})?;
			results.insert(call_id, Some(recovery_turn.clone()));
			recovered.push((recovery_turn.clone(), index));
		}
	}
	Ok(recovered)
}
struct SequenceRecovery {
	ts:         u64,
	receipt:    TurnReceipt,
	start:      TurnStart,
	amendments: BTreeMap<u64, u64>,
}

fn recover_sequence_amendments(log: &Log, writer: &mut Writer) -> Result<(), JournalError> {
	let mut starts = BTreeMap::new();
	let mut recoveries = Vec::<SequenceRecovery>::new();
	let mut active = None;
	for index in 0..u64::try_from(log.len()).expect("transcript length fits in u64") {
		let Some(transcript::Entry::Ok(event)) = log.get(index) else {
			continue;
		};
		match &event.kind {
			Kind::TurnStart(start) => {
				starts.insert(start.turn_id.clone(), start.clone());
				active = None;
			},
			Kind::TurnReceipt(receipt) => {
				let Some(start) = starts.get(receipt.turn_id.as_str()).cloned() else {
					return Err(JournalError::MissingTurnStart(receipt.turn_id.clone()));
				};
				recoveries.push(SequenceRecovery {
					ts: event.ts,
					receipt: receipt.clone(),
					start,
					amendments: BTreeMap::new(),
				});
				active = Some(recoveries.len() - 1);
			},
			Kind::Amend { target, patch: AmendPatch::Seq { seq } } => {
				if let Some(recovery) = active.and_then(|position| recoveries.get_mut(position))
					&& recovery.start.sequence_targets.contains(target)
				{
					recovery.amendments.insert(*target, *seq);
				}
			},
			_ => {},
		}
	}
	for recovery in recoveries {
		let Some(revision) = recovery.receipt.outcome.revision.as_ref() else {
			continue;
		};
		let input_len = u64::try_from(recovery.start.sequence_targets.len())
			.map_err(|_| JournalError::InvalidSequenceRange(recovery.receipt.turn_id.clone()))?;
		let output_len = u64::try_from(recovery.receipt.outcome.output.len())
			.map_err(|_| JournalError::InvalidSequenceRange(recovery.receipt.turn_id.clone()))?;
		let first_input = revision
			.head
			.checked_sub(output_len)
			.and_then(|head| head.checked_add(1))
			.and_then(|first_output| first_output.checked_sub(input_len))
			.ok_or_else(|| JournalError::InvalidSequenceRange(recovery.receipt.turn_id.clone()))?;
		for (offset, target) in recovery.start.sequence_targets.iter().enumerate() {
			if !matches!(
				log.get(*target),
				Some(transcript::Entry::Ok(event)) if event_item(&event.kind).is_some()
			) {
				return Err(JournalError::InvalidTurnInput(*target));
			}
			let expected = first_input
				.checked_add(u64::try_from(offset).expect("sequence target length fits in u64"))
				.ok_or_else(|| JournalError::InvalidSequenceRange(recovery.receipt.turn_id.clone()))?;
			if let Some(actual) = recovery.amendments.get(target) {
				if *actual != expected {
					return Err(JournalError::SequenceReplayMismatch {
						target: *target,
						actual: *actual,
						expected,
					});
				}
				continue;
			}
			writer.append(&Event {
				ts: recovery.ts,
				kind: Kind::Amend { target: *target, patch: AmendPatch::Seq { seq: expected } },
			})?;
		}
	}
	Ok(())
}
struct RewriteRecovery {
	ts:        u64,
	intent:    PromptRewriteIntent,
	stages:    Vec<Option<u64>>,
	committed: bool,
}

fn recover_prompt_rewrites(
	log: &Log,
	writer: &mut Writer,
) -> Result<(u64, Option<([u8; 32], Vec<u64>)>), JournalError> {
	let mut rewrites = BTreeMap::<u64, RewriteRecovery>::new();
	for index in 0..u64::try_from(log.len()).expect("transcript length fits in u64") {
		let Some(transcript::Entry::Ok(event)) = log.get(index) else {
			continue;
		};
		match &event.kind {
			Kind::PromptRewriteIntent(intent) => {
				rewrites.insert(index, RewriteRecovery {
					ts: event.ts,
					intent: intent.clone(),
					stages: vec![None; intent.head.len()],
					committed: false,
				});
			},
			Kind::PromptRewriteStage(stage) => {
				let Some(rewrite) = rewrites.get_mut(&stage.intent) else {
					return Err(JournalError::CorruptPromptRewrite(stage.intent));
				};
				let ordinal = usize::try_from(stage.ordinal)
					.map_err(|_| JournalError::CorruptPromptRewrite(stage.intent))?;
				let Some(expected) = rewrite.intent.head.get(ordinal) else {
					return Err(JournalError::CorruptPromptRewrite(stage.intent));
				};
				if expected != &stage.item || rewrite.stages[ordinal].replace(index).is_some() {
					return Err(JournalError::CorruptPromptRewrite(stage.intent));
				}
			},
			Kind::PromptRewriteCommit(commit) => {
				let Some(rewrite) = rewrites.get_mut(&commit.intent) else {
					return Err(JournalError::CorruptPromptRewrite(commit.intent));
				};
				let complete = rewrite
					.stages
					.iter()
					.copied()
					.collect::<Option<Vec<_>>>()
					.is_some_and(|stages| stages == commit.head_events);
				if !complete || rewrite.committed {
					return Err(JournalError::CorruptPromptRewrite(commit.intent));
				}
				rewrite.committed = true;
			},
			_ => {},
		}
	}

	let mut recovered_items = 0_u64;
	let mut active_prompt = None;
	for (intent_event, rewrite) in &mut rewrites {
		if !rewrite.committed {
			for (ordinal, stage_event) in rewrite.stages.iter_mut().enumerate() {
				if stage_event.is_some() {
					continue;
				}
				let index = writer.append(&Event {
					ts: rewrite.ts,
					kind: Kind::PromptRewriteStage(PromptRewriteStage {
						intent: *intent_event,
						ordinal: u64::try_from(ordinal).expect("prompt head length fits in u64"),
						item: rewrite.intent.head[ordinal].clone(),
					}),
				})?;
				*stage_event = Some(index);
				recovered_items = recovered_items.saturating_add(1);
			}
		}
		let head_events = rewrite
			.stages
			.iter()
			.copied()
			.collect::<Option<Vec<_>>>()
			.expect("committed or recovered prompt stages are complete");
		if !rewrite.committed {
			writer.append(&Event {
				ts: rewrite.ts,
				kind: Kind::PromptRewriteCommit(PromptRewriteCommit {
					intent: *intent_event,
					head_events: head_events.clone(),
				}),
			})?;
		}
		active_prompt = Some((rewrite.intent.prompt_hash, head_events));
	}
	Ok((recovered_items, active_prompt))
}

fn event_item(kind: &Kind) -> Option<&Item> {
	match kind {
		Kind::Item(record) => Some(&record.item),
		Kind::TurnInput(input) => Some(&input.item),
		Kind::PromptRewriteStage(stage) => Some(&stage.item),
		Kind::JobSettled(settled) => Some(&settled.settlement),
		_ => None,
	}
}

#[cfg(test)]
mod tests {
	use std::sync::atomic::{AtomicU64, Ordering};

	use omp_proto::{inference::v1 as pb, thread::v1 as thread_pb};
	use omp_proto::prost::Message as _;
	use omp_tool::{ArtifactLifetime, ExpectedArtifact, JobOwner};
	use omp_storage::transcript::{Entry, Header, SessionId};

	use super::*;
	use crate::{PromptHash, project_journal};

	static NEXT_PATH: AtomicU64 = AtomicU64::new(0);

	fn path(name: &str) -> PathBuf {
		std::env::temp_dir().join(format!(
			"omp-agent-journal-{name}-{}-{}.jsonl",
			std::process::id(),
			NEXT_PATH.fetch_add(1, Ordering::Relaxed)
		))
	}

	fn header() -> Header {
		Header {
			v:       4,
			id:      SessionId(Str::from("journal-test")),
			created: 1,
			cwd:     std::env::temp_dir(),
		}
	}

	fn message(text: &str) -> Item {
		Item {
			kind: Some(thread_pb::item::Kind::Message(thread_pb::Message {
				role:  thread_pb::Role::User as i32,
				parts: vec![thread_pb::Part {
					kind: Some(thread_pb::part::Kind::Text(text.to_owned())),
				}],
			})),
			..Default::default()
		}
	}

	fn outcome() -> Outcome {
		Outcome {
			output: vec![Item {
				seq:           3,
				created_at_ms: 9,
				kind:          Some(thread_pb::item::Kind::Message(thread_pb::Message {
					role:  thread_pb::Role::Assistant as i32,
					parts: vec![thread_pb::Part {
						kind: Some(thread_pb::part::Kind::Text("answer".to_owned())),
					}],
				})),
				props:         None,
			}],
			stop: pb::StopReason::StopEndTurn as i32,
			revision: Some(thread_pb::Revision { head: 3, token: vec![0xa5; 32].into() }),
			provider: "provider".to_owned(),
			model: "model".to_owned(),
			duration_ms: Some(42),
			..Default::default()
		}
	}

	fn caps() -> omp_tool::PromptCaps {
		omp_tool::PromptCaps {
			maximum_parts: 1,
			maximum_text_bytes: 1024,
			media: false,
		}
	}

	fn tool_outcome() -> Outcome {
		Outcome {
			output: vec![Item {
				seq: 3,
				created_at_ms: 4,
				kind: Some(thread_pb::item::Kind::ToolCall(thread_pb::ToolCall {
					id: "call-1".to_owned(),
					name: "read".to_owned(),
					args_json: br#"{"path":"x"}"#.to_vec().into(),
					..Default::default()
				})),
				props: Some(pb::ValueMap {
					fields: BTreeMap::from([(omp_tool::TOOL_REV_PROP.to_owned(), pb::Value {
						kind: Some(pb::value::Kind::String("1".to_owned())),
					})]),
				}),
			}],
			stop: pb::StopReason::StopToolUse as i32,
			revision: Some(thread_pb::Revision { head: 3, token: vec![4; 32].into() }),
			provider: "provider".to_owned(),
			model: "model".to_owned(),
			..Default::default()
		}
	}

	fn assert_tool_crash_recovery(authorized: bool, expected_text: &str) {
		let path = path(if authorized { "authorized-tool" } else { "unmarked-tool" });
		let hash = PromptHash::from([2; 32]);
		let mut journal = Journal::create(&path, &header()).expect("create journal");
		let input = journal
			.append_turn_input(2, "turn", message("input"), Some(hash))
			.expect("append turn input");
		journal
			.start_turn(3, TurnStart {
				turn_id: Str::from("turn"),
				item_events: vec![input],
				prompt_hash: hash.into_bytes(),
				prompt_head_events: Vec::new(),
				toolset_hash: [3; 32],
				enabled_tools: vec![Str::from("read")],
				sequence_targets: vec![input],
				input: TurnInputRecord::Full {
					thread: thread_pb::Thread { items: vec![message("input")] },
				},
				options: TurnOptionsRecord {
					context_id: None,
					params: pb::ChatParams::default(),
					executor: None,
					props: None,
				},
			})
			.expect("start turn");
		journal
			.append_gateway_outcome(4, "turn", tool_outcome())
			.expect("append tool outcome");
		if authorized {
			journal
				.authorize_tool_batch(5, "turn", &[Str::from("call-1")])
				.expect("authorize tool batch");
		}
		drop(journal);
		let reopened = Journal::open(&path).expect("recover unresolved tool");
		let (recovery_turn, indexes) =
			reopened.pending_input_submission().expect("recovery submission");
		ulid::Ulid::from_string(recovery_turn.as_str()).expect("recovery turn id is a ULID");
		assert_eq!(indexes.len(), 1);
		let log = reopened.load().expect("load recovery");
		let Some(Entry::Ok(event)) = log.get(indexes[0]) else {
			panic!("recovered input missing");
		};
		let Kind::TurnInput(input) = &event.kind else {
			panic!("recovery must be typed turn input");
		};
		let Some(thread_pb::item::Kind::ToolResult(result)) = input.item.kind.as_ref() else {
			panic!("recovery input must be tool result");
		};
		assert_eq!(result.call_id, "call-1");
		let Some(thread_pb::part::Kind::Text(text)) =
			result.parts.first().and_then(|part| part.kind.as_ref())
		else {
			panic!("recovery result text missing");
		};
		assert!(text.contains(expected_text));
		let bytes = std::fs::read(&path).expect("read once-recovered journal");
		drop(reopened);

		let reopened = Journal::open(&path).expect("reopen recovered tool");
		assert_eq!(std::fs::read(&path).expect("read twice-recovered journal"), bytes);
		assert_eq!(reopened.pending_input_submission().expect("same recovery").1.len(), 1);
		std::fs::remove_file(path).expect("remove journal");
	}
	#[test]
	fn staged_turn_input_reopens_with_exact_turn_id() {
		let path = path("staged-input");
		let turn_id = ulid::Ulid::generate().to_string();
		let mut journal = Journal::create(&path, &header()).expect("create journal");
		let index = journal
			.append_turn_input(2, &turn_id, message("input"), Some(PromptHash::from([1; 32])))
			.expect("append staged input");
		drop(journal);
		let reopened = Journal::open(&path).expect("reopen staged input");
		let (durable_turn_id, indexes) =
			reopened.pending_input_submission().expect("pending staged input");
		assert_eq!(durable_turn_id.as_str(), turn_id);
		assert_eq!(indexes, &[index]);
		assert_eq!(reopened.recoverable_input_events(), &[index]);
		std::fs::remove_file(path).expect("remove journal");
	}

	#[test]
	fn partial_tool_batch_recovery_coalesces_missing_results_into_existing_follow_up() {
		let path = path("partial-tool-batch");
		let hash = PromptHash::from([3; 32]);
		let mut journal = Journal::create(&path, &header()).expect("create journal");

		let input = journal
			.append_turn_input(2, "turn", message("input"), Some(hash))
			.expect("append turn input");
		journal
			.start_turn(3, TurnStart {
				turn_id: Str::from("turn"),
				item_events: vec![input],
				prompt_hash: hash.into_bytes(),
				prompt_head_events: Vec::new(),
				toolset_hash: [3; 32],
				enabled_tools: vec![Str::from("read")],
				sequence_targets: vec![input],
				input: TurnInputRecord::Full {
					thread: thread_pb::Thread { items: vec![message("input")] },
				},
				options: TurnOptionsRecord {
					context_id: None,
					params: pb::ChatParams::default(),
					executor: None,
					props: None,
				},
			})
			.expect("start turn");
		let mut outcome = tool_outcome();
		let mut second = outcome.output[0].clone();
		second.seq = 4;
		let Some(thread_pb::item::Kind::ToolCall(call)) = second.kind.as_mut() else {
			panic!("fixture call missing");
		};
		call.id = "call-2".to_owned();
		outcome.output.push(second);
		outcome.revision.as_mut().expect("revision").head = 4;
		journal
			.append_gateway_outcome(4, "turn", outcome.clone())
			.expect("append tool outcome");
		journal
			.authorize_tool_batch(5, "turn", &[Str::from("call-1"), Str::from("call-2")])
			.expect("authorize tool batch");
		let follow_up = ulid::Ulid::generate().to_string();
		let first_result = crate::project::recovery_tool_result_item(
			6,
			&outcome.output[0],
			Abort::Interrupted { reason: Str::new_static("fixture terminal result") },
		)
		.expect("build first terminal result");
		journal
			.append_turn_input(6, &follow_up, first_result, Some(hash))
			.expect("append first result");
		drop(journal);

		let reopened = Journal::open(&path).expect("recover missing batch result");
		let (turn_id, indexes) = reopened.pending_input_submission().expect("recovery submission");
		assert_eq!(turn_id.as_str(), follow_up);
		assert_eq!(indexes.len(), 2);
		let bytes = std::fs::read(&path).expect("read recovered batch");
		drop(reopened);
		let reopened = Journal::open(&path).expect("reopen recovered batch");
		assert_eq!(std::fs::read(&path).expect("read idempotent batch"), bytes);
		assert_eq!(reopened.pending_input_submission().expect("same group").1.len(), 2);
		std::fs::remove_file(path).expect("remove journal");
	}

	#[test]
	fn ordered_staged_turn_groups_and_settlement_survive_sequential_reopens() {
		let path = path("staged-queue");
		let first_turn = ulid::Ulid::generate().to_string();
		let second_turn = ulid::Ulid::generate().to_string();
		let mut journal = Journal::create(&path, &header()).expect("create journal");
		let first = journal
			.append_turn_input(2, &first_turn, message("first"), None)
			.expect("append first group");
		let second = journal
			.append_turn_input(3, &second_turn, message("second"), None)
			.expect("append second group");
		let settlement = journal
			.settle_job(4, "job", message("settled"))
			.expect("append settlement");
		drop(journal);

		let mut journal = Journal::open(&path).expect("first reopen");
		assert_eq!(
			journal.pending_input_submission(),
			Some((&Str::from(first_turn.as_str()), [first].as_slice()))
		);
		assert_eq!(journal.recoverable_settlement_events(), &[settlement]);
		journal
			.start_turn(5, TurnStart {
				turn_id: Str::from(first_turn.as_str()),
				item_events: vec![first, settlement],
				prompt_hash: [0; 32],
				prompt_head_events: Vec::new(),
				toolset_hash: [0; 32],
				enabled_tools: Vec::new(),
				sequence_targets: vec![first, settlement],
				input: TurnInputRecord::Full {
					thread: thread_pb::Thread {
						items: vec![message("first"), message("settled")],
					},
				},
				options: TurnOptionsRecord {
					context_id: None,
					params: pb::ChatParams::default(),
					executor: None,
					props: None,
				},
			})
			.expect("start first group");
		journal
			.append_gateway_outcome(6, &first_turn, outcome())
			.expect("complete first group");
		drop(journal);

		let mut journal = Journal::open(&path).expect("second reopen");
		let (turn_id, indexes) = journal.pending_input_submission().expect("second group");
		assert_eq!(turn_id.as_str(), second_turn);
		assert_eq!(indexes, &[second]);
		assert!(journal.recoverable_settlement_events().is_empty());
		journal
			.start_turn(7, TurnStart {
				turn_id: Str::from(second_turn.as_str()),
				item_events: vec![second],
				prompt_hash: [0; 32],
				prompt_head_events: Vec::new(),
				toolset_hash: [0; 32],
				enabled_tools: Vec::new(),
				sequence_targets: vec![second],
				input: TurnInputRecord::Full {
					thread: thread_pb::Thread { items: vec![message("second")] },
				},
				options: TurnOptionsRecord {
					context_id: None,
					params: pb::ChatParams::default(),
					executor: None,
					props: None,
				},
			})
			.expect("start second group");
		journal
			.append_gateway_outcome(8, &second_turn, outcome())
			.expect("complete second group");
		drop(journal);

		let journal = Journal::open(&path).expect("final reopen");
		assert!(journal.pending_input_submission().is_none());
		assert!(journal.recoverable_settlement_events().is_empty());
		std::fs::remove_file(path).expect("remove journal");
	}
	#[test]
	fn tool_crash_recovery_distinguishes_authorized_effect_uncertainty() {
		assert_tool_crash_recovery(true, "effects unknown");
		assert_tool_crash_recovery(false, "skipped");
	}

	#[test]
	fn gateway_outcome_receipt_round_trips_and_replays_exactly_once() {
		let path = path("receipt");
		let prompt_hash = PromptHash::from([7; 32]);
		let mut journal = Journal::create(&path, &header()).expect("create journal");
		let prompt_event = journal
			.append_optimistic(2, message("system"), Some(prompt_hash))
			.expect("append prompt");
		let input_event = journal
			.append_optimistic(3, message("input"), Some(prompt_hash))
			.expect("append input");
		journal
			.start_turn(4, TurnStart {
				turn_id:            Str::from("turn"),
				item_events:        vec![input_event],
				prompt_hash:        prompt_hash.into_bytes(),
				prompt_head_events: vec![prompt_event],
				toolset_hash:       [8; 32],
				enabled_tools:      Vec::new(),
				sequence_targets:   vec![input_event],
				input:              TurnInputRecord::Delta {
					context: pb::ContextRef {
						context_id: "context".to_owned(),
						expected: Some(thread_pb::Revision { head: 2, token: vec![3; 32].into() }),
					},
					delta: pb::ThreadDelta { truncate_to: None, append: vec![message("input")] },
				},
				options:            TurnOptionsRecord {
					context_id: None,
					params: pb::ChatParams::default(),
					executor: None,
					props: None,
				},
			})
			.expect("start turn");

		let expected = outcome();
		let (receipt, replay) = journal
			.append_gateway_outcome(5, "turn", expected.clone())
			.expect("append outcome");
		assert!(!replay);
		assert_eq!(receipt.prompt_hash, prompt_hash.into_bytes());
		assert_eq!(receipt.prompt_head_events, vec![prompt_event]);
		assert_eq!(receipt.outcome, expected);
		let bytes = std::fs::read(&path).expect("read committed journal");

		let (replayed, replay) = journal
			.append_gateway_outcome(6, "turn", expected.clone())
			.expect("replay exact outcome");
		assert!(replay);
		assert_eq!(replayed, receipt);
		assert_eq!(std::fs::read(&path).expect("read replayed journal"), bytes);

		let mut different = expected.clone();
		different.provider = "other".to_owned();
		assert!(matches!(
			journal.append_gateway_outcome(7, "turn", different),
			Err(JournalError::TurnReplayMismatch(_))
		));
		assert_eq!(std::fs::read(&path).expect("read rejected replay journal"), bytes);
		drop(journal);

		let reopened = Journal::open(&path).expect("reopen journal");
		assert_eq!(reopened.receipt("turn"), Some(&receipt));
		assert!(reopened.pending_turn().is_none());
		let projected = project_journal(&reopened.load().expect("load recovered"), &omp_tool::Registry::new(), &caps())
			.expect("project recovered");
		assert_eq!(projected.items[1].seq, 2, "reopen must recover the missing sequence patch");
		std::fs::remove_file(path).expect("remove journal");
	}

	#[test]
	fn partial_turn_output_stays_hidden_and_exact_start_reopens() {
		let path = path("partial-output");
		let prompt_hash = PromptHash::from([9; 32]);
		let mut journal = Journal::create(&path, &header()).expect("create journal");
		let prompt_event = journal
			.append_optimistic(2, message("system"), Some(prompt_hash))
			.expect("append prompt");
		let input = message("input");
		let input_event = journal
			.append_optimistic(3, input.clone(), Some(prompt_hash))
			.expect("append input");
		let start = TurnStart {
			turn_id: Str::from("turn"),
			item_events: vec![input_event],
			prompt_hash: prompt_hash.into_bytes(),
			prompt_head_events: vec![prompt_event],
			toolset_hash: [6; 32],
			enabled_tools: Vec::new(),
			sequence_targets: vec![input_event],
			input: TurnInputRecord::Full {
				thread: thread_pb::Thread { items: vec![message("system"), input] },
			},
			options: TurnOptionsRecord {
				context_id: Some(Str::from("seed")),
				params: pb::ChatParams {
					model: "provider/model".to_owned(),
					..Default::default()
				},
				executor: None,
				props: None,
			},
		};
		journal.start_turn(4, start.clone()).expect("start turn");
		journal
			.writer
			.append(&Event {
				ts: 5,
				kind: Kind::Item(ItemRecord {
					item: outcome().output[0].clone(),
					turn_id: Some(start.turn_id.clone()),
					prompt_hash: Some(start.prompt_hash),
				}),
			})
			.expect("append interrupted output prefix");
		drop(journal);

		let reopened = Journal::open(&path).expect("reopen partial turn");
		assert_eq!(reopened.pending_turn(), Some(&start));
		let projected = project_journal(&reopened.load().expect("load partial"), &omp_tool::Registry::new(), &caps())
			.expect("project partial");
		assert_eq!(projected.items, vec![message("system"), message("input")]);
		std::fs::remove_file(path).expect("remove journal");
	}

	#[test]
	fn sequence_amendment_projects_without_mutating_item_event() {
		let path = path("sequence");
		let mut journal = Journal::create(&path, &header()).expect("create journal");
		let target = journal
			.append_optimistic(2, message("input"), None)
			.expect("append optimistic item");
		journal
			.amend_seq(3, target, 9)
			.expect("append sequence amendment");
		let log = journal.load().expect("load journal");
		let Some(Entry::Ok(event)) = log.get(target) else {
			panic!("item event missing")
		};
		let Kind::Item(record) = &event.kind else {
			panic!("target is not an item")
		};
		assert_eq!(record.item.seq, 0);
		let projected = project_journal(
			&log,
			&omp_tool::Registry::new(),
			&caps(),
		)
		.expect("project journal");
		assert_eq!(projected.items[0].seq, 9);
		drop(journal);
		std::fs::remove_file(path).expect("remove journal");
	}

	#[test]
	fn prompt_rewrite_recovers_every_partial_materialization_once() {
		for staged in 0..=2 {
			let path = path("prompt-rewrite");
			let mut journal = Journal::create(&path, &header()).expect("create journal");
			let old_head = journal
				.append_optimistic(2, message("old-head"), Some(PromptHash::from([1; 32])))
				.expect("append old head");
			let tail = journal
				.append_optimistic(3, message("tail"), Some(PromptHash::from([1; 32])))
				.expect("append tail");
			drop(journal);

			let head = vec![message("new-head-a"), message("new-head-b")];
			let mut writer = Writer::open_append(&path).expect("open raw writer");
			let intent = writer
				.append(&Event {
					ts: 4,
					kind: Kind::PromptRewriteIntent(PromptRewriteIntent {
						prompt_hash: [2; 32],
						head: head.clone(),
						preserved_tail: vec![tail],
					}),
				})
				.expect("append rewrite intent");
			for (ordinal, item) in head.iter().take(staged).enumerate() {
				writer
					.append(&Event {
						ts: 4,
						kind: Kind::PromptRewriteStage(PromptRewriteStage {
							intent,
							ordinal: ordinal as u64,
							item: item.clone(),
						}),
					})
					.expect("append partial stage");
			}
			drop(writer);

			let pending = transcript::load(&path).expect("load incomplete rewrite");
			assert_eq!(pending.live(), vec![old_head, tail]);
			let pending_thread =
				project_journal(&pending, &omp_tool::Registry::new(), &caps())
					.expect("project old live chain");
			assert_eq!(pending_thread.items, vec![message("old-head"), message("tail")]);

			let recovered = Journal::open(&path).expect("recover rewrite");
			let live = recovered.live_item_events().expect("read recovered live indexes");
			assert_eq!(live.len(), 3);
			assert_eq!(live[2], tail);
			let (active_hash, active_head) = recovered.active_prompt().expect("active recovered prompt");
			assert_eq!(active_hash, [2; 32]);
			assert_eq!(active_head, &live[..2]);
			assert!(!live.contains(&old_head));
			assert_eq!(
				recovered.items_at(&live).expect("read rewritten items"),
				vec![head[0].clone(), head[1].clone(), message("tail")]
			);
			let projected =
				project_journal(&recovered.load().expect("load recovered journal"), &omp_tool::Registry::new(), &caps())
					.expect("project recovered rewrite");
			let projected_bytes = projected.encode_to_vec();
			drop(recovered);
			let recovered_bytes = std::fs::read(&path).expect("read recovered bytes");

			let reopened = Journal::open(&path).expect("reopen completed rewrite");
			assert_eq!(reopened.live_item_events().expect("read stable live indexes"), live);
			let reprojection =
				project_journal(&reopened.load().expect("reload journal"), &omp_tool::Registry::new(), &caps())
					.expect("reproject completed rewrite");
			assert_eq!(reprojection.encode_to_vec(), projected_bytes);
			drop(reopened);
			assert_eq!(
				std::fs::read(&path).expect("read idempotently reopened bytes"),
				recovered_bytes,
				"reopening must not duplicate stages or commit"
			);
			std::fs::remove_file(path).expect("remove journal");
		}
	}

	#[test]
	fn detached_jobs_reconstruct_pending_minus_settled_without_duplicates() {
		let path = path("jobs");
		let mut journal = Journal::create(&path, &header()).expect("create journal");
		let job = |id: &'static str| JobRef {
			id: Str::new_static(id),
			owner: JobOwner::NamedProcess {
				name: Str::new_static(id),
				generation: 1,
			},
			artifact: ExpectedArtifact {
				description: Str::new_static("artifact"),
				media_type: Some(Str::new_static("text/plain")),
				lifetime: ArtifactLifetime::Session,
			},
		};
		let first = job("job-a");
		let second = job("job-b");
		let first_index = journal.register_job(2, first.clone()).expect("register first job");
		assert_eq!(
			journal.register_job(3, first.clone()).expect("repeat first registration"),
			first_index
		);
		journal.register_job(4, second.clone()).expect("register second job");
		let settlement = message("job-a settled");
		let settlement_index = journal
			.settle_job(5, first.id.as_str(), settlement.clone())
			.expect("settle first job");
		assert_eq!(
			journal
				.settle_job(6, first.id.as_str(), settlement.clone())
				.expect("repeat settlement"),
			settlement_index
		);
		assert_eq!(journal.pending_jobs().cloned().collect::<Vec<_>>(), vec![second.clone()]);
		drop(journal);

		let mut reopened = Journal::open(&path).expect("reopen jobs");
		assert_eq!(reopened.pending_jobs().cloned().collect::<Vec<_>>(), vec![second.clone()]);
		reopened
			.settle_job(7, second.id.as_str(), message("job-b settled"))
			.expect("settle resumed job");
		assert_eq!(reopened.pending_jobs().count(), 0);
		assert!(matches!(
			reopened.settle_job(8, first.id.as_str(), message("different")),
			Err(JournalError::JobReplayMismatch(_))
		));
		drop(reopened);
		std::fs::remove_file(path).expect("remove journal");
	}
}

