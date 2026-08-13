//! Dimensioned integer pricing, long-context tiers, and checked cost
//! arithmetic.

use std::fmt;

use serde::{Deserialize, Serialize};
use strum::{Display, EnumString, IntoStaticStr};

/// Number of nano-US dollars, where one US dollar is one billion units.
#[derive(
	Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
)]
#[repr(transparent)]
#[serde(transparent)]
pub struct NanoUsd(u64);

impl NanoUsd {
	/// Zero cost.
	pub const ZERO: Self = Self(0);

	/// Creates an amount from nano-US dollars.
	#[must_use]
	pub const fn from_nanos(nanos: u64) -> Self {
		Self(nanos)
	}

	/// Returns the amount in nano-US dollars.
	#[must_use]
	pub const fn as_nanos(self) -> u64 {
		self.0
	}

	/// Adds two monetary amounts while detecting overflow.
	pub const fn checked_add(self, other: Self) -> Option<Self> {
		match self.0.checked_add(other.0) {
			Some(value) => Some(Self(value)),
			None => None,
		}
	}
}

impl fmt::Display for NanoUsd {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(formatter, "{} nano-USD", self.0)
	}
}

/// Billing dimension for one price component.
#[derive(
	Clone,
	Copy,
	Debug,
	Display,
	EnumString,
	Eq,
	Hash,
	IntoStaticStr,
	Ord,
	PartialEq,
	PartialOrd,
	Serialize,
	Deserialize,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case", ascii_case_insensitive, const_into_str)]
pub enum PriceUnit {
	/// One million uncached input tokens.
	MtokInput,
	/// One million output tokens.
	MtokOutput,
	/// One million prompt-cache read tokens.
	MtokCacheRead,
	/// One million prompt-cache write tokens.
	MtokCacheWrite,
	/// One generated image.
	Image,
	/// One generated video second.
	VideoSecond,
	/// One generated or transcribed audio second.
	AudioSecond,
	/// One million input characters.
	McharInput,
	/// One request.
	Request,
}

/// Exact integer price for one billing unit.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Price {
	/// Billing dimension.
	pub unit:      PriceUnit,
	/// Price in billionths of one US dollar.
	pub nanos_usd: u64,
}

/// Replacement pricing activated above a prompt-token threshold.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PriceTier {
	/// Exclusive lower prompt-token threshold.
	pub prompt_tokens_above: u64,
	/// Replacement component prices in deterministic unit order.
	pub components:          Box<[Price]>,
}

/// Integer-only model pricing schedule.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct Pricing {
	/// Base component prices in deterministic unit order.
	pub components: Box<[Price]>,
	/// Threshold tiers in ascending threshold order.
	pub tiers:      Box<[PriceTier]>,
}

impl Pricing {
	/// Canonically sorts and validates a complete price schedule.
	pub fn new(mut components: Vec<Price>, mut tiers: Vec<PriceTier>) -> Result<Self, PricingError> {
		components.sort_unstable_by_key(|price| price.unit);
		for tier in &mut tiers {
			let mut prices = tier.components.to_vec();
			prices.sort_unstable_by_key(|price| price.unit);
			tier.components = prices.into_boxed_slice();
		}
		tiers.sort_unstable_by_key(|tier| tier.prompt_tokens_above);
		let pricing =
			Self { components: components.into_boxed_slice(), tiers: tiers.into_boxed_slice() };
		pricing.validate()?;
		Ok(pricing)
	}

	/// Validates deterministic ordering and uniqueness.
	pub fn validate(&self) -> Result<(), PricingError> {
		validate_components(&self.components)?;
		let mut previous = None;
		for tier in &self.tiers {
			if previous.is_some_and(|threshold| threshold >= tier.prompt_tokens_above) {
				return Err(PricingError::TiersNotStrictlyOrdered);
			}
			validate_components(&tier.components)?;
			previous = Some(tier.prompt_tokens_above);
		}
		Ok(())
	}

	/// Selects the highest replacement tier whose threshold is strictly
	/// exceeded.
	pub fn components_for_prompt(&self, prompt_tokens: u64) -> &[Price] {
		self
			.tiers
			.iter()
			.take_while(|tier| prompt_tokens > tier.prompt_tokens_above)
			.last()
			.map_or(&self.components, |tier| tier.components.as_ref())
	}

	/// Computes cost without applying a quota or premium multiplier.
	pub fn cost(&self, usage: UsageDimensions) -> Result<NanoUsd, CostError> {
		self.cost_with_multiplier(usage, None)
	}

	/// Computes cost and applies an explicit fixed-point multiplier.
	pub fn cost_with_multiplier(
		&self,
		usage: UsageDimensions,
		multiplier: Option<PremiumMultiplier>,
	) -> Result<NanoUsd, CostError> {
		self.validate().map_err(CostError::InvalidSchedule)?;
		let prompt_tokens = usage.prompt_tokens().ok_or(CostError::Overflow)?;
		let mut total = 0_u128;
		for price in self.components_for_prompt(prompt_tokens) {
			let (quantity, divisor) = usage.quantity(price.unit);
			let product = u128::from(price.nanos_usd) * u128::from(quantity);
			let charge = ceil_div(product, divisor);
			total = total.checked_add(charge).ok_or(CostError::Overflow)?;
		}
		let total = u64::try_from(total).map_err(|_| CostError::Overflow)?;
		let cost = NanoUsd::from_nanos(total);
		match multiplier {
			Some(multiplier) => multiplier.apply(cost),
			None => Ok(cost),
		}
	}
}

fn validate_components(components: &[Price]) -> Result<(), PricingError> {
	let mut previous = None;
	for component in components {
		if previous.is_some_and(|unit| unit >= component.unit) {
			return Err(PricingError::ComponentsNotStrictlyOrdered);
		}
		previous = Some(component.unit);
	}
	Ok(())
}

const fn ceil_div(numerator: u128, divisor: u64) -> u128 {
	if numerator == 0 {
		return 0;
	}
	(numerator - 1) / divisor as u128 + 1
}

/// Exact usage quantities for every price dimension.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct UsageDimensions {
	/// Uncached prompt tokens.
	pub input_tokens:       u64,
	/// Generated tokens.
	pub output_tokens:      u64,
	/// Prompt-cache read tokens.
	pub cache_read_tokens:  u64,
	/// Prompt-cache write tokens.
	pub cache_write_tokens: u64,
	/// Generated images.
	pub images:             u64,
	/// Generated video seconds.
	pub video_seconds:      u64,
	/// Generated or transcribed audio seconds.
	pub audio_seconds:      u64,
	/// Input characters.
	pub input_characters:   u64,
	/// Billable requests.
	pub requests:           u64,
}

impl UsageDimensions {
	/// Returns total prompt tokens used for tier selection.
	#[must_use]
	pub const fn prompt_tokens(self) -> Option<u64> {
		let Some(tokens) = self.input_tokens.checked_add(self.cache_read_tokens) else {
			return None;
		};
		tokens.checked_add(self.cache_write_tokens)
	}

	const fn quantity(self, unit: PriceUnit) -> (u64, u64) {
		const MILLION: u64 = 1_000_000;
		match unit {
			PriceUnit::MtokInput => (self.input_tokens, MILLION),
			PriceUnit::MtokOutput => (self.output_tokens, MILLION),
			PriceUnit::MtokCacheRead => (self.cache_read_tokens, MILLION),
			PriceUnit::MtokCacheWrite => (self.cache_write_tokens, MILLION),
			PriceUnit::Image => (self.images, 1),
			PriceUnit::VideoSecond => (self.video_seconds, 1),
			PriceUnit::AudioSecond => (self.audio_seconds, 1),
			PriceUnit::McharInput => (self.input_characters, MILLION),
			PriceUnit::Request => (self.requests, 1),
		}
	}
}

/// Exact non-negative multiplier scaled by one million.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[repr(transparent)]
#[serde(transparent)]
pub struct PremiumMultiplier(u64);

impl PremiumMultiplier {
	/// Identity multiplier.
	pub const ONE: Self = Self(Self::SCALE);
	/// Fixed-point scale representing `1.0`.
	pub const SCALE: u64 = 1_000_000;

	/// Constructs a multiplier from its millionth-scale integer.
	#[must_use]
	pub const fn from_millionths(millionths: u64) -> Self {
		Self(millionths)
	}

	/// Returns the millionth-scale integer.
	#[must_use]
	pub const fn as_millionths(self) -> u64 {
		self.0
	}

	/// Applies the multiplier with upward nano-dollar rounding.
	pub fn apply(self, amount: NanoUsd) -> Result<NanoUsd, CostError> {
		let product = u128::from(amount.as_nanos()) * u128::from(self.0);
		let scaled = ceil_div(product, Self::SCALE);
		let nanos = u64::try_from(scaled).map_err(|_| CostError::Overflow)?;
		Ok(NanoUsd::from_nanos(nanos))
	}
}

/// Invalid price schedule.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PricingError {
	/// Price units were duplicated or not in canonical order.
	ComponentsNotStrictlyOrdered,
	/// Tier thresholds were duplicated or not in ascending order.
	TiersNotStrictlyOrdered,
}

impl fmt::Display for PricingError {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::ComponentsNotStrictlyOrdered => {
				formatter.write_str("price components must have unique units in canonical order")
			},
			Self::TiersNotStrictlyOrdered => {
				formatter.write_str("price tiers must have unique ascending thresholds")
			},
		}
	}
}

impl std::error::Error for PricingError {}

/// Checked cost calculation failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CostError {
	/// The input schedule was not canonical and valid.
	InvalidSchedule(PricingError),
	/// Prompt aggregation, cost accumulation, or multiplier application
	/// overflowed.
	Overflow,
}

impl fmt::Display for CostError {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::InvalidSchedule(error) => write!(formatter, "invalid price schedule: {error}"),
			Self::Overflow => formatter.write_str("nano-USD cost overflow"),
		}
	}
}

impl std::error::Error for CostError {}

#[cfg(test)]
mod tests {
	use omp_core::Str;
	use serde::Deserialize;

	use super::*;

	#[derive(Deserialize)]
	struct PriceFixture {
		daybreak: Vec<FixtureModel>,
	}

	#[derive(Deserialize)]
	struct FixtureModel {
		model:         Str,
		pricing:       Vec<Price>,
		pricing_tiers: Vec<FixtureTier>,
	}

	#[derive(Deserialize)]
	struct FixtureTier {
		prompt_tokens_above: u64,
		pricing:             Vec<Price>,
	}

	#[test]
	fn fixture_long_context_price_boundary_is_strict_and_exact() {
		let fixture: PriceFixture = serde_json::from_str(include_str!(
			"../../../fixtures/llm-oracle/catalog-policy/aliases-tiers-and-deepseek.json"
		))
		.expect("pricing fixture parses");
		let model = fixture
			.daybreak
			.into_iter()
			.find(|model| model.model == "daybreak-blue-latest")
			.expect("tiered fixture model");
		let tiers = model
			.pricing_tiers
			.into_iter()
			.map(|tier| PriceTier {
				prompt_tokens_above: tier.prompt_tokens_above,
				components:          tier.pricing.into_boxed_slice(),
			})
			.collect();
		let pricing =
			Pricing::new(model.pricing, tiers).expect("fixture schedule is canonicalizable");

		let boundary = pricing
			.cost(UsageDimensions { input_tokens: 272_000, ..UsageDimensions::default() })
			.expect("boundary cost");
		assert_eq!(boundary, NanoUsd::from_nanos(1_360_000_000));

		let above = pricing
			.cost(UsageDimensions { input_tokens: 272_001, ..UsageDimensions::default() })
			.expect("long-context cost");
		assert_eq!(above, NanoUsd::from_nanos(2_720_010_000));
	}

	#[test]
	fn every_dimension_uses_integer_arithmetic_and_rounds_up() {
		let components = vec![
			Price { unit: PriceUnit::MtokInput, nanos_usd: 1 },
			Price { unit: PriceUnit::MtokOutput, nanos_usd: 1 },
			Price { unit: PriceUnit::MtokCacheRead, nanos_usd: 1 },
			Price { unit: PriceUnit::MtokCacheWrite, nanos_usd: 1 },
			Price { unit: PriceUnit::Image, nanos_usd: 2 },
			Price { unit: PriceUnit::VideoSecond, nanos_usd: 3 },
			Price { unit: PriceUnit::AudioSecond, nanos_usd: 5 },
			Price { unit: PriceUnit::McharInput, nanos_usd: 1 },
			Price { unit: PriceUnit::Request, nanos_usd: 7 },
		];
		let pricing = Pricing::new(components, Vec::new()).expect("ordered dimensions");
		let cost = pricing
			.cost(UsageDimensions {
				input_tokens:       1,
				output_tokens:      1,
				cache_read_tokens:  1,
				cache_write_tokens: 1,
				images:             1,
				video_seconds:      1,
				audio_seconds:      1,
				input_characters:   1,
				requests:           1,
			})
			.expect("small exact cost");
		assert_eq!(cost, NanoUsd::from_nanos(22));
	}

	#[test]
	fn cost_and_premium_overflow_are_reported() {
		let pricing =
			Pricing::new(vec![Price { unit: PriceUnit::Request, nanos_usd: u64::MAX }], Vec::new())
				.expect("single component");
		assert_eq!(
			pricing.cost(UsageDimensions { requests: 2, ..UsageDimensions::default() }),
			Err(CostError::Overflow),
		);
		assert_eq!(
			pricing.cost(UsageDimensions {
				input_tokens: u64::MAX,
				cache_read_tokens: 1,
				..UsageDimensions::default()
			}),
			Err(CostError::Overflow),
		);
		assert_eq!(
			PremiumMultiplier::from_millionths(u64::MAX).apply(NanoUsd::from_nanos(u64::MAX)),
			Err(CostError::Overflow),
		);
	}

	#[test]
	fn premium_multiplier_is_exact_fixed_point() {
		let multiplier = PremiumMultiplier::from_millionths(330_000);
		assert_eq!(multiplier.as_millionths(), 330_000);
		assert_eq!(
			multiplier
				.apply(NanoUsd::from_nanos(10))
				.expect("scaled cost"),
			NanoUsd::from_nanos(4),
		);
		assert_eq!(serde_json::to_string(&multiplier).expect("serialize"), "330000");
	}
}
