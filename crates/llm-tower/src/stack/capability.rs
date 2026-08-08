//! Provider capability admission and portable feature fallback.
//!
//! This is a layer rather than caller-side policy because otherwise every
//! consumer needing forced tool calls would have to reproduce provider
//! compatibility checks, the bounded retry loop, and the cache-cost tradeoff:
//! forcing a tool on Anthropic changes the request shape and costs a cache
//! miss.

use omp_core::SmolStr;
use omp_llm_catalog::compat::Compat;
use omp_llm_types::{
	Fallback, Feature, TurnError, TurnErrorKind, TurnEvent, Unsupported, UnsupportedAction,
};

/// Facet-layer capability resolver bound to one provider compatibility row.
#[derive(Clone, Copy, Debug)]
pub struct CapabilityResolver {
	compat:               Compat,
	forced_tool_attempts: u32,
}

impl CapabilityResolver {
	/// Creates a resolver with a bounded forced-tool retry policy.
	#[must_use]
	pub fn new(compat: Compat, forced_tool_attempts: u32) -> Self {
		Self { compat, forced_tool_attempts: forced_tool_attempts.max(1) }
	}

	/// Resolves a feature using the relevant compatibility-axis selector.
	pub fn resolve<T>(
		&self,
		feature: Feature<T>,
		is_supported: impl FnOnce(&Compat) -> bool,
		what: impl Into<SmolStr>,
		detail: impl Into<SmolStr>,
	) -> Result<(FeatureResolution<T>, Option<Unsupported>), TurnError> {
		resolve_feature(feature, &self.compat, is_supported, what, detail)
	}

	/// Creates the shared forced-tool emulation ladder for this provider.
	#[must_use]
	pub fn forced_tool_escalation(&self) -> ForcedToolEscalation {
		ForcedToolEscalation::new(&self.compat, self.forced_tool_attempts)
	}
}

/// The admitted representation of a requested feature.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FeatureResolution<T> {
	/// The provider supports the feature without alteration.
	Native(T),
	/// The feature was omitted under [`Fallback::Ignore`].
	Dropped,
	/// The value is retained for a portable soft implementation.
	Emulated(T),
}

/// Resolves one portable feature against provider compatibility data.
///
/// `is_supported` names the relevant [`Compat`] axis. Keeping the axis selector
/// at the call site makes this usable for every `Feature<T>` without inventing
/// a second capability table beside the catalog.
pub fn resolve_feature<T>(
	feature: Feature<T>,
	compat: &Compat,
	is_supported: impl FnOnce(&Compat) -> bool,
	what: impl Into<SmolStr>,
	detail: impl Into<SmolStr>,
) -> Result<(FeatureResolution<T>, Option<Unsupported>), TurnError> {
	if is_supported(compat) {
		return Ok((FeatureResolution::Native(feature.value), None));
	}
	let unsupported = Unsupported::builder()
		.what(what.into())
		.detail(detail.into())
		.action(match feature.on_unsupported {
			Fallback::Emulate => UnsupportedAction::Emulated,
			Fallback::Error | Fallback::Ignore => UnsupportedAction::Dropped,
			_ => UnsupportedAction::Dropped,
		})
		.build();
	match feature.on_unsupported {
		Fallback::Error => Err(unsupported_error(unsupported)),
		Fallback::Ignore => Ok((FeatureResolution::Dropped, Some(unsupported))),
		Fallback::Emulate => Ok((FeatureResolution::Emulated(feature.value), Some(unsupported))),
		_ => Err(unsupported_error(unsupported)),
	}
}

fn unsupported_error(unsupported: Unsupported) -> TurnError {
	TurnError::builder()
		.kind(TurnErrorKind::Unsupported)
		.detail(unsupported.detail.clone())
		.unsupported(vec![unsupported])
		.retry_after_ms(0)
		.build()
}

/// Provider request strategy for one forced-tool attempt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ForcedToolStrategy {
	/// Add model-facing guidance while leaving provider tool choice unforced.
	InjectSoftPrompt,
	/// Use the provider's native forced-tool control.
	ForceNative,
}

/// One observable outbound attempt in forced-tool emulation.
#[derive(Clone, Debug, PartialEq)]
pub struct ForcedToolAttempt {
	/// Event which must be forwarded to the caller before this attempt starts.
	pub event:    TurnEvent,
	/// Request strategy for the attempt.
	pub strategy: ForcedToolStrategy,
}

/// Stateful, bounded forced-tool emulation ladder.
///
/// The first attempt is always soft to preserve prompt-cache affinity. After a
/// non-compliant result, native forcing is used when available; otherwise soft
/// attempts continue only to the configured bound.
#[derive(Clone, Debug)]
pub struct ForcedToolEscalation {
	forced_supported: bool,
	max_attempts:     u32,
	attempt:          u32,
	finished:         bool,
}

impl ForcedToolEscalation {
	/// Creates a ladder with at least one outbound attempt.
	#[must_use]
	pub fn new(compat: &Compat, max_attempts: u32) -> Self {
		Self {
			forced_supported: compat.forced_tool_choice,
			max_attempts:     max_attempts.max(1),
			attempt:          0,
			finished:         false,
		}
	}

	/// Starts the ladder with cache-friendly soft prompt injection.
	///
	/// Calling this more than once returns the same terminal bounded-retry error
	/// rather than accidentally issuing an unreported extra request.
	pub fn start(&mut self) -> Result<ForcedToolAttempt, TurnError> {
		if self.attempt != 0 || self.finished {
			return Err(Self::failure("forced-tool escalation already started"));
		}
		self.attempt = 1;
		Ok(self.plan(ForcedToolStrategy::InjectSoftPrompt, "soft forced-tool prompt"))
	}

	/// Verifies the preceding result and, when needed, returns the next attempt.
	///
	/// `None` means the model complied and no further provider request is
	/// needed.
	pub fn verify(&mut self, complied: bool) -> Result<Option<ForcedToolAttempt>, TurnError> {
		if self.attempt == 0 {
			return Err(Self::failure("forced-tool escalation was not started"));
		}
		if self.finished {
			return Ok(None);
		}
		if complied {
			self.finished = true;
			return Ok(None);
		}
		if self.attempt >= self.max_attempts {
			self.finished = true;
			return Err(Self::failure(
				"model did not produce the required tool call within the retry bound",
			));
		}
		self.attempt += 1;
		let (strategy, reason) = if self.forced_supported {
			(
				ForcedToolStrategy::ForceNative,
				"soft tool forcing was not obeyed; escalating to native forcing",
			)
		} else {
			(
				ForcedToolStrategy::InjectSoftPrompt,
				"soft tool forcing was not obeyed; retrying without native support",
			)
		};
		Ok(Some(self.plan(strategy, reason)))
	}

	fn plan(&self, strategy: ForcedToolStrategy, reason: &'static str) -> ForcedToolAttempt {
		ForcedToolAttempt {
			event: TurnEvent::Attempt { number: self.attempt, reason: SmolStr::new(reason) },
			strategy,
		}
	}

	fn failure(detail: &'static str) -> TurnError {
		TurnError::builder()
			.kind(TurnErrorKind::Unsupported)
			.detail(SmolStr::new(detail))
			.unsupported(Vec::new())
			.retry_after_ms(0)
			.build()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn feature(policy: Fallback) -> Feature<u8> {
		Feature::builder().value(7).on_unsupported(policy).build()
	}

	#[test]
	fn every_fallback_takes_its_documented_path() {
		let compat = Compat::default();
		let error =
			resolve_feature(feature(Fallback::Error), &compat, |_| false, "format", "unsupported")
				.expect_err("error policy must reject admission");
		assert_eq!(error.kind, TurnErrorKind::Unsupported);
		assert_eq!(error.unsupported[0].action, UnsupportedAction::Dropped);

		let ignored =
			resolve_feature(feature(Fallback::Ignore), &compat, |_| false, "format", "unsupported")
				.expect("ignore policy must admit");
		assert_eq!(ignored.0, FeatureResolution::Dropped);
		assert_eq!(ignored.1.expect("omission is reported").action, UnsupportedAction::Dropped);

		let emulated =
			resolve_feature(feature(Fallback::Emulate), &compat, |_| false, "format", "unsupported")
				.expect("emulate policy must admit");
		assert_eq!(emulated.0, FeatureResolution::Emulated(7));
		assert_eq!(emulated.1.expect("emulation is reported").action, UnsupportedAction::Emulated);
	}

	#[test]
	fn forced_tool_ladder_is_visible_and_bounded() {
		let mut compat = Compat::default();
		compat.forced_tool_choice = false;
		let compat = compat;
		let mut ladder = ForcedToolEscalation::new(&compat, 3);
		let first = ladder.start().expect("first attempt");
		assert_eq!(first.strategy, ForcedToolStrategy::InjectSoftPrompt);
		assert!(matches!(first.event, TurnEvent::Attempt { number: 1, .. }));
		let second = ladder
			.verify(false)
			.expect("second attempt")
			.expect("retry");
		assert!(matches!(second.event, TurnEvent::Attempt { number: 2, .. }));
		let third = ladder.verify(false).expect("third attempt").expect("retry");
		assert!(matches!(third.event, TurnEvent::Attempt { number: 3, .. }));
		assert_eq!(
			ladder
				.verify(false)
				.expect_err("bound must stop retries")
				.kind,
			TurnErrorKind::Unsupported
		);
	}

	#[test]
	fn forced_tool_ladder_escalates_only_after_verification() {
		let mut ladder = ForcedToolEscalation::new(&Compat::default(), 2);
		assert_eq!(
			ladder.start().expect("first attempt").strategy,
			ForcedToolStrategy::InjectSoftPrompt
		);
		assert_eq!(
			ladder
				.verify(false)
				.expect("second attempt")
				.expect("retry")
				.strategy,
			ForcedToolStrategy::ForceNative
		);
		assert!(ladder.verify(true).expect("compliance completes").is_none());
	}
}
