//! Account quota windows kept separate from request-rate throttling.

use std::{collections::BTreeMap, time::SystemTime};

use omp_core::Str;
use strum::{Display, EnumString, IntoStaticStr};

use super::rate::Sample;

/// Identifies one independently reset quota window.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct QuotaWindowId(pub Str);

impl QuotaWindowId {
	/// Creates a quota-window identifier.
	pub fn new(value: impl Into<Str>) -> Self {
		Self(value.into())
	}

	/// Borrows the stable window identifier.
	pub fn as_str(&self) -> &str {
		self.0.as_str()
	}
}

/// Provenance of a quota measurement.
#[derive(Clone, Copy, Debug, Display, EnumString, Eq, IntoStaticStr, PartialEq)]
#[strum(serialize_all = "snake_case", const_into_str)]
pub enum QuotaProvenance {
	/// A usage or quota endpoint reported the value.
	Provider,
	/// Response headers reported the value.
	Header,
	/// A structured provider error reported exhaustion.
	Error,
	/// The runtime derived the value from accepted usage.
	Measured,
}

/// A partial receipt for one account quota window.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QuotaObservation {
	/// Window being updated.
	pub window:      QuotaWindowId,
	/// Amount consumed, when reported.
	pub consumed:    Option<u64>,
	/// Amount remaining, when reported.
	pub remaining:   Option<u64>,
	/// Total allowance, when reported.
	pub limit:       Option<u64>,
	/// Absolute reset time, when reported.
	pub reset_at:    Option<SystemTime>,
	/// Whether structured evidence explicitly says the quota is exhausted.
	pub exhausted:   Option<bool>,
	/// Evidence provenance.
	pub provenance:  QuotaProvenance,
	/// Time at which the receipt was observed.
	pub observed_at: SystemTime,
}

/// Merged state for one quota window.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QuotaWindow {
	/// Most recent consumed-value sample.
	pub consumed:  Option<Sample<u64>>,
	/// Most recent remaining-value sample.
	pub remaining: Option<Sample<u64>>,
	/// Most recent limit sample.
	pub limit:     Option<Sample<u64>>,
	/// Most recent reset sample.
	pub reset_at:  Option<Sample<SystemTime>>,
	/// Most recent explicit exhaustion sample.
	pub exhausted: Option<Sample<bool>>,
	/// Every partial receipt, retained in arrival order.
	pub receipts:  Vec<QuotaObservation>,
}

impl QuotaWindow {
	fn new() -> Self {
		Self {
			consumed:  None,
			remaining: None,
			limit:     None,
			reset_at:  None,
			exhausted: None,
			receipts:  Vec::new(),
		}
	}

	fn apply(&mut self, observation: QuotaObservation) {
		merge_sample(&mut self.consumed, observation.consumed, observation.observed_at);
		merge_sample(&mut self.remaining, observation.remaining, observation.observed_at);
		merge_sample(&mut self.limit, observation.limit, observation.observed_at);
		merge_sample(&mut self.reset_at, observation.reset_at, observation.observed_at);
		merge_sample(&mut self.exhausted, observation.exhausted, observation.observed_at);
		self.receipts.push(observation);
	}

	/// Computes availability at a supplied deterministic clock instant.
	pub fn availability(&self, now: SystemTime) -> QuotaAvailability {
		let exhausted = match (self.exhausted, self.remaining) {
			(Some(explicit), Some(remaining)) if explicit.observed_at >= remaining.observed_at => {
				explicit.value
			},
			(_, Some(remaining)) => remaining.value == 0,
			(Some(explicit), None) => explicit.value,
			(None, None) => false,
		};
		if !exhausted {
			return QuotaAvailability::Available;
		}
		match self.reset_at.map(|sample| sample.value) {
			Some(reset_at) if reset_at > now => QuotaAvailability::Exhausted { reset_at },
			Some(_) => QuotaAvailability::Available,
			None => QuotaAvailability::ExhaustedUnknownReset,
		}
	}
}

fn merge_sample<T: Copy>(slot: &mut Option<Sample<T>>, value: Option<T>, observed_at: SystemTime) {
	let Some(value) = value else { return };
	if slot
		.as_ref()
		.is_none_or(|current| observed_at >= current.observed_at)
	{
		*slot = Some(Sample { value, observed_at });
	}
}

/// Current quota eligibility across all account windows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuotaAvailability {
	/// No current window is exhausted.
	Available,
	/// At least one window is exhausted until this deterministic latest reset.
	Exhausted { reset_at: SystemTime },
	/// A window is exhausted without a reported reset.
	ExhaustedUnknownReset,
}

/// Quota state for one account, independent of request-rate state.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct QuotaState {
	windows: BTreeMap<QuotaWindowId, QuotaWindow>,
}

impl QuotaState {
	/// Applies a partial receipt without clearing omitted measurements.
	pub fn apply(&mut self, observation: QuotaObservation) {
		self
			.windows
			.entry(observation.window.clone())
			.or_insert_with(QuotaWindow::new)
			.apply(observation);
	}

	/// Records a quota-classified 429 without modifying rate state.
	pub fn record_429(
		&mut self,
		window: QuotaWindowId,
		reset_at: Option<SystemTime>,
		observed_at: SystemTime,
	) {
		self.apply(QuotaObservation {
			window,
			consumed: None,
			remaining: Some(0),
			limit: None,
			reset_at,
			exhausted: Some(true),
			provenance: QuotaProvenance::Error,
			observed_at,
		});
	}

	/// Returns a window by identifier.
	pub fn window(&self, id: &QuotaWindowId) -> Option<&QuotaWindow> {
		self.windows.get(id)
	}

	/// Iterates over windows in stable identifier order.
	pub fn windows(&self) -> impl ExactSizeIterator<Item = (&QuotaWindowId, &QuotaWindow)> {
		self.windows.iter()
	}

	/// Computes aggregate availability; the latest active reset wins
	/// deterministically.
	pub fn availability(&self, now: SystemTime) -> QuotaAvailability {
		let mut reset_at = None;
		for window in self.windows.values() {
			match window.availability(now) {
				QuotaAvailability::Available => {},
				QuotaAvailability::Exhausted { reset_at: candidate } => {
					reset_at =
						Some(reset_at.map_or(candidate, |current: SystemTime| current.max(candidate)));
				},
				QuotaAvailability::ExhaustedUnknownReset => {
					return QuotaAvailability::ExhaustedUnknownReset;
				},
			}
		}
		reset_at
			.map_or(QuotaAvailability::Available, |reset_at| QuotaAvailability::Exhausted { reset_at })
	}

	/// Returns the smallest current known remaining amount for deterministic
	/// ranking.
	///
	/// A sample observed before an elapsed reset belongs to the previous quota
	/// window and is unknown.
	pub fn minimum_remaining(&self, now: SystemTime) -> Option<u64> {
		self
			.windows
			.values()
			.filter_map(|window| {
				let remaining = window.remaining?;
				if window
					.reset_at
					.is_some_and(|reset| reset.value <= now && remaining.observed_at < reset.value)
				{
					None
				} else {
					Some(remaining.value)
				}
			})
			.min()
	}
}
