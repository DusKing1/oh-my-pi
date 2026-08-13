//! Credential-free, immutable execution plans and capability negotiation
//! evidence.
use std::{
	sync::Arc,
	time::{Duration, Instant},
};

use omp_core::Str;

use crate::{
	call::Call,
	catalog::{
		CatalogRevision, CodecId, Emulation, ModelKey, OperationKind, PolicyModel, ProviderId,
		RouteId, ThinkingPolicy, ThinkingSelection, WirePolicy, WireTarget,
	},
	error::{Error, ErrorDetail, ErrorKind},
	receipt::{Adjustment, ExecutionBudget, ExecutionReceipt, FeatureId, ReasonId, Replayability},
};

/// Whether a caller expressed a hard requirement or an adjustable preference.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequirementStrength {
	/// Planning fails unless the capability is satisfied.
	Required,
	/// Planning may continue only with explicit adjustment evidence.
	Preferred,
}

/// One feature requested from a selected model and route.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapabilityRequirement {
	/// Stable feature identity.
	pub feature:  FeatureId,
	/// Whether absence is fatal or adjustable.
	pub strength: RequirementStrength,
}

/// Route-scoped evidence for a requested capability.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapabilityEvidence {
	/// Stable feature identity.
	pub feature:      FeatureId,
	/// Native, emulated, unsupported, or unknown route behavior.
	pub availability: CapabilityAvailability,
	/// Evidence provenance or constraint result.
	pub reason:       ReasonId,
}

/// Constraint-checked capability availability used by the planner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CapabilityAvailability {
	/// The selected codec and route implement the capability directly.
	Native,
	/// The runtime can reproduce the requested behavior by an explicit method.
	Emulated(Emulation),
	/// Available evidence proves the behavior cannot be provided.
	Unsupported,
	/// Available evidence does not establish support or non-support.
	Unknown,
}

/// The decision made for one requested capability.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NegotiationDecision {
	/// The selected route provides the feature natively.
	Native { feature: FeatureId },
	/// The selected route provides the feature through an allowed emulation.
	Emulated { feature: FeatureId, method: Emulation },
	/// An unknown preferred feature was accepted under explicit best-effort
	/// policy.
	UnknownAccepted { feature: FeatureId, reason: ReasonId },
	/// A preferred feature was dropped with receipt evidence.
	Dropped { feature: FeatureId, reason: ReasonId },
}

/// Caller policy governing native, emulated, unknown, and dropped behavior.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlanningPolicy {
	/// Whether native support is mandatory for required features.
	pub allow_emulation:           bool,
	/// Whether lossy emulation is allowed when the catalog labels it explicitly.
	pub allow_lossy_emulation:     bool,
	/// Whether unknown support may satisfy a preference.
	pub allow_unknown_preferences: bool,
	/// Whether unsupported preferences may be dropped with an adjustment.
	pub allow_dropped_preferences: bool,
}

impl Default for PlanningPolicy {
	fn default() -> Self {
		Self {
			allow_emulation:           false,
			allow_lossy_emulation:     false,
			allow_unknown_preferences: false,
			allow_dropped_preferences: false,
		}
	}
}

/// A typed codec-specific option requested by an operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeOptionRequirement {
	/// Codec that alone may serialize the option.
	pub codec:    CodecId,
	/// Whether a mismatch is fatal or may be dropped with evidence.
	pub strength: RequirementStrength,
	/// Stable feature identity used in errors and receipts.
	pub feature:  FeatureId,
}

/// Explicit input-body facts used to reject unsafe multi-attempt plans.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplayRequirements {
	/// Aggregate replayability of every operation body component.
	pub replayability:           Replayability,
	/// Whether planning may require a semantic retry behind an output gate.
	pub semantic_retry_possible: bool,
	/// Whether secure staging was explicitly requested by the caller.
	pub staging_explicit:        bool,
	/// Maximum body bytes the caller permits staging.
	pub staging_limit:           Option<u64>,
}

impl Default for ReplayRequirements {
	fn default() -> Self {
		Self {
			replayability:           Replayability::Replayable,
			semantic_retry_possible: false,
			staging_explicit:        false,
			staging_limit:           None,
		}
	}
}

/// How the plan will make body data available to attempts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayPlan {
	/// Every attempt opens an independent replayable source.
	Replayable,
	/// Exactly one attempt is permitted and automatic fallback is suppressed.
	OneShotSingleAttempt,
	/// Explicit secure staging must complete before the first attempt.
	SecureStaging { maximum_bytes: u64 },
}

/// Health evidence for one concrete route service.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum RouteHealth {
	/// Recent probes or attempts establish that the route is healthy.
	Healthy,
	/// No runtime observation is available.
	Unknown,
	/// The route remains usable but has degraded observations.
	Degraded,
	/// The route is not currently eligible for execution.
	Unavailable,
}

/// Route-scoped, credential-free runtime capability and ranking evidence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeRouteEvidence {
	/// Route whose runtime state was observed.
	pub route:            RouteId,
	/// Monotonic registry state generation.
	pub generation:       u64,
	/// Current route health classification.
	pub health:           RouteHealth,
	/// Remaining quota score in millionths, where larger is preferred.
	pub quota_millionths: u32,
	/// Smoothed end-to-end latency used for deterministic ranking.
	pub latency:          Duration,
	/// Whether an existing session or account binding prefers this route.
	pub affinity:         bool,
	/// Route-specific operation support observed at runtime.
	pub operation:        CapabilityAvailability,
	/// Route-specific requested-feature evidence.
	pub capabilities:     Arc<[CapabilityEvidence]>,
}

/// Exact caller-authorized model sequence retained by a plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FallbackScope {
	/// Primary normalized model selected by the exact selector, absent for
	/// management operations.
	pub primary:  Option<ModelKey>,
	/// Ordered normalized models explicitly named as fallbacks.
	pub explicit: Arc<[ModelKey]>,
}

/// Clone-cheap side-effect-free planner used by typed clients.
pub trait Planner: Clone + Send + Sync + 'static {
	/// Produces an immutable credential-free execution plan for a call.
	fn plan(&self, call: &Call, now: Instant) -> Result<ExecutionPlan, Error>;

	/// Revalidates expiry, catalog revision, and volatile registry generation.
	fn validate(&self, plan: &ExecutionPlan, now: Instant) -> Result<(), Error>;
}

/// One exact, already-negotiated route authorized for pre-commit fallback.
#[derive(Clone, Debug)]
pub struct PlannedFallback {
	/// Normalized model, absent only for model-less management operations.
	pub model:              Option<ModelKey>,
	/// Provider domain.
	pub provider:           ProviderId,
	/// Concrete route.
	pub route:              RouteId,
	/// Codec used by the route.
	pub codec:              CodecId,
	/// Exact interned wire-lowering policy.
	pub wire_policy:        Arc<WirePolicy>,
	/// Exact optional thinking policy.
	pub thinking_policy:    Option<Arc<ThinkingPolicy>>,
	/// Fully resolved thinking selection for this route.
	pub thinking_selection: Option<ThinkingSelection>,
	/// Candidate-specific capability negotiation outcomes.
	pub decisions:          Arc<[NegotiationDecision]>,
	/// Router-facing model facts.
	pub policy_model:       Option<Arc<PolicyModel>>,
	/// Exact codec-facing target.
	pub wire_target:        Option<WireTarget>,
	/// Route-scoped runtime evidence.
	pub runtime_evidence:   RuntimeRouteEvidence,
}

/// Immutable, credential-free execution plan.
#[derive(Clone, Debug)]
pub struct ExecutionPlan {
	/// Wall-clock instant captured during side-effect-free planning for
	/// time-sensitive policy.
	pub planned_at:          std::time::SystemTime,
	/// Catalog revision against which selection and negotiation ran.
	pub catalog_revision:    CatalogRevision,
	/// Registry generation against which route services were inspected.
	pub registry_generation: u64,
	/// Absolute time after which volatile evidence must be replanned.
	pub expires_at:          Instant,
	/// Planned operation kind.
	pub operation:           OperationKind,
	/// Selected normalized model for model-scoped operations.
	pub model:               Option<ModelKey>,
	/// Selected provider domain.
	pub provider:            ProviderId,
	/// Selected concrete route.
	pub route:               RouteId,
	/// Selected codec.
	pub codec:               CodecId,
	/// Router-facing catalog facts; absent for model-less management operations.
	pub policy_model:        Option<Arc<PolicyModel>>,
	/// Exact interned wire policy selected for encoding.
	pub wire_policy:         Arc<WirePolicy>,
	/// Exact optional thinking policy selected for model-scoped encoding.
	pub thinking_policy:     Option<Arc<ThinkingPolicy>>,
	/// Fully resolved effort, budget, reasoning mode, and opaque wire model.
	pub thinking_selection:  Option<ThinkingSelection>,
	/// Capability negotiation outcomes.
	pub decisions:           Arc<[NegotiationDecision]>,
	/// Exact caller-authorized model fallback scope.
	pub fallback_scope:      FallbackScope,
	/// Ordered exact fallback routes authorized during planning; no runtime
	/// candidate invention.
	pub fallbacks:           Arc<[PlannedFallback]>,
	/// Input replay or staging behavior required before execution.
	pub replay:              ReplayPlan,
	/// Cross-attempt budget copied from the request.
	pub budget:              ExecutionBudget,
	/// Route-scoped runtime facts used during planning.
	pub runtime_evidence:    RuntimeRouteEvidence,
	pub(crate) wire_target:  Option<WireTarget>,
}

impl ExecutionPlan {
	/// Rejects an expired plan or a plan produced for different catalog/registry
	/// state.
	pub fn validate(
		&self,
		now: Instant,
		catalog_revision: &CatalogRevision,
		registry_generation: u64,
	) -> Result<(), Error> {
		if plan_is_current(
			now,
			self.expires_at,
			&self.catalog_revision,
			catalog_revision,
			self.registry_generation,
			registry_generation,
		) {
			return Ok(());
		}

		Err(Error::planning(
			ErrorKind::StalePlan,
			ErrorDetail::StalePlan {
				planned_revision: Str::from(self.catalog_revision.as_str()),
				current_revision: if now > self.expires_at {
					Str::from("expired")
				} else if &self.catalog_revision != catalog_revision {
					Str::from(catalog_revision.as_str())
				} else {
					Str::from("registry-state-changed")
				},
			},
			ExecutionReceipt::default(),
		))
	}

	/// Borrows the codec-only wire target at the encoding boundary.
	pub(crate) fn wire_target(&self) -> Option<&WireTarget> {
		self.wire_target.as_ref()
	}
}

fn plan_is_current(
	now: Instant,
	expires_at: Instant,
	planned_revision: &CatalogRevision,
	current_revision: &CatalogRevision,
	planned_generation: u64,
	current_generation: u64,
) -> bool {
	now <= expires_at
		&& planned_revision == current_revision
		&& planned_generation == current_generation
}

/// Negotiates requested capabilities without acquiring credentials or touching
/// a network.
pub fn negotiate(
	requirements: &[CapabilityRequirement],
	evidence: &[CapabilityEvidence],
	policy: PlanningPolicy,
) -> Result<(Vec<NegotiationDecision>, Vec<Adjustment>), Error> {
	let mut decisions = Vec::with_capacity(requirements.len());
	let mut adjustments = Vec::new();
	for requirement in requirements {
		let observed = evidence
			.iter()
			.find(|item| item.feature == requirement.feature);
		let availability = observed.map_or(CapabilityAvailability::Unknown, |item| item.availability);
		let reason = observed
			.map(|item| item.reason.clone())
			.unwrap_or_else(|| ReasonId(Str::from("no-route-evidence")));

		match availability {
			CapabilityAvailability::Native => {
				decisions.push(NegotiationDecision::Native { feature: requirement.feature.clone() })
			},
			CapabilityAvailability::Emulated(method)
				if policy.allow_emulation
					&& (policy.allow_lossy_emulation || emulation_is_lossless(method)) =>
			{
				decisions.push(NegotiationDecision::Emulated {
					feature: requirement.feature.clone(),
					method,
				});
			},
			CapabilityAvailability::Unknown
				if requirement.strength == RequirementStrength::Preferred
					&& policy.allow_unknown_preferences =>
			{
				decisions.push(NegotiationDecision::UnknownAccepted {
					feature: requirement.feature.clone(),
					reason,
				});
			},
			CapabilityAvailability::Unsupported
				if requirement.strength == RequirementStrength::Preferred
					&& policy.allow_dropped_preferences =>
			{
				decisions.push(NegotiationDecision::Dropped {
					feature: requirement.feature.clone(),
					reason:  reason.clone(),
				});
				adjustments.push(Adjustment::Dropped { feature: requirement.feature.clone(), reason });
			},
			CapabilityAvailability::Unknown => {
				return Err(capability_error(ErrorKind::CapabilityUnknown, requirement, reason));
			},
			CapabilityAvailability::Unsupported | CapabilityAvailability::Emulated(_) => {
				return Err(capability_error(ErrorKind::CapabilityMismatch, requirement, reason));
			},
		}
	}
	Ok((decisions, adjustments))
}

/// Validates a codec-specific option before any authentication or encoding
/// occurs.
pub fn negotiate_native_option(
	requirement: Option<&NativeOptionRequirement>,
	selected_codec: &CodecId,
	allow_drop_preferred: bool,
) -> Result<Option<NegotiationDecision>, Error> {
	let Some(requirement) = requirement else {
		return Ok(None);
	};
	if &requirement.codec == selected_codec {
		return Ok(Some(NegotiationDecision::Native { feature: requirement.feature.clone() }));
	}
	if requirement.strength == RequirementStrength::Preferred && allow_drop_preferred {
		return Ok(Some(NegotiationDecision::Dropped {
			feature: requirement.feature.clone(),
			reason:  ReasonId(Str::from("native-option-codec-mismatch")),
		}));
	}
	Err(Error::planning(
		ErrorKind::CodecMismatch,
		ErrorDetail::Capability {
			feature: Str::from(requirement.feature.0.as_str()),
			reason:  ReasonId(Str::from("native-option-codec-mismatch")),
		},
		ExecutionReceipt::default(),
	))
}

/// Derives explicit retry/staging behavior from aggregate body evidence.
pub fn plan_replay(
	requirements: &ReplayRequirements,
	budget: &ExecutionBudget,
) -> Result<ReplayPlan, Error> {
	match requirements.replayability {
		Replayability::Replayable | Replayability::Staged => Ok(ReplayPlan::Replayable),
		Replayability::OneShot if requirements.semantic_retry_possible => {
			if !requirements.staging_explicit {
				return Err(replay_error(
					ErrorKind::StagingRequired,
					"semantic-retry-requires-explicit-staging",
				));
			}
			let maximum_bytes = requirements
				.staging_limit
				.unwrap_or(0)
				.min(budget.max_staging_bytes);
			if maximum_bytes == 0 {
				return Err(replay_error(ErrorKind::StagingRequired, "staging-budget-is-zero"));
			}
			Ok(ReplayPlan::SecureStaging { maximum_bytes })
		},
		Replayability::OneShot if budget.max_attempts > 1 => {
			Err(replay_error(ErrorKind::ReplayRequired, "one-shot-body-forbids-multiple-attempts"))
		},
		Replayability::OneShot => Ok(ReplayPlan::OneShotSingleAttempt),
	}
}

fn capability_error(
	kind: ErrorKind,
	requirement: &CapabilityRequirement,
	reason: ReasonId,
) -> Error {
	Error::planning(
		kind,
		ErrorDetail::Capability { feature: Str::from(requirement.feature.0.as_str()), reason },
		ExecutionReceipt::default(),
	)
}

fn replay_error(kind: ErrorKind, reason: &'static str) -> Error {
	Error::planning(
		kind,
		ErrorDetail::Replay { reason: ReasonId(Str::from(reason)) },
		ExecutionReceipt::default(),
	)
}

const fn emulation_is_lossless(method: Emulation) -> bool {
	!matches!(method, Emulation::PromptInstruction)
}

#[cfg(test)]
mod tests {
	use std::time::{Duration, Instant};

	use omp_core::Str;

	use super::*;
	use crate::{
		catalog::{CatalogRevision, CodecId, Emulation},
		receipt::{Cost, ExecutionBudget, FeatureId, ReasonId},
	};

	fn requirement(strength: RequirementStrength) -> CapabilityRequirement {
		CapabilityRequirement { feature: FeatureId(Str::from("structured-output")), strength }
	}

	fn budget(attempts: u32, staging: u64) -> ExecutionBudget {
		ExecutionBudget {
			max_elapsed:           None,
			max_attempts:          attempts,
			max_input_tokens:      None,
			max_output_tokens:     None,
			max_cost:              None::<Cost>,
			max_provisional_bytes: 0,
			max_staging_bytes:     staging,
		}
	}

	#[test]
	fn unknown_and_unsupported_have_distinct_typed_failures() {
		let unknown = negotiate(
			&[requirement(RequirementStrength::Required)],
			&[CapabilityEvidence {
				feature:      FeatureId(Str::from("structured-output")),
				availability: CapabilityAvailability::Unknown,
				reason:       ReasonId(Str::from("not-observed")),
			}],
			PlanningPolicy::default(),
		)
		.expect_err("unknown requirement must fail");
		assert_eq!(unknown.kind, ErrorKind::CapabilityUnknown);

		let unsupported = negotiate(
			&[requirement(RequirementStrength::Required)],
			&[CapabilityEvidence {
				feature:      FeatureId(Str::from("structured-output")),
				availability: CapabilityAvailability::Unsupported,
				reason:       ReasonId(Str::from("proven-absent")),
			}],
			PlanningPolicy::default(),
		)
		.expect_err("unsupported requirement must fail");
		assert_eq!(unsupported.kind, ErrorKind::CapabilityMismatch);
	}

	#[test]
	fn native_and_emulated_features_require_explicit_policy() {
		let native = CapabilityEvidence {
			feature:      FeatureId(Str::from("structured-output")),
			availability: CapabilityAvailability::Native,
			reason:       ReasonId(Str::from("route-native")),
		};
		let (decisions, _) = negotiate(
			&[requirement(RequirementStrength::Required)],
			&[native],
			PlanningPolicy::default(),
		)
		.unwrap();
		assert!(matches!(decisions.as_slice(), [NegotiationDecision::Native { .. }]));

		let emulated = CapabilityEvidence {
			feature:      FeatureId(Str::from("structured-output")),
			availability: CapabilityAvailability::Emulated(Emulation::ResponseTransform),
			reason:       ReasonId(Str::from("bounded-validator")),
		};
		assert_eq!(
			negotiate(
				&[requirement(RequirementStrength::Required)],
				&[emulated.clone()],
				PlanningPolicy::default()
			)
			.expect_err("emulation defaults forbidden")
			.kind,
			ErrorKind::CapabilityMismatch,
		);
		let policy = PlanningPolicy { allow_emulation: true, ..PlanningPolicy::default() };
		assert!(matches!(
			negotiate(&[requirement(RequirementStrength::Required)], &[emulated], policy)
				.unwrap()
				.0
				.as_slice(),
			[NegotiationDecision::Emulated { .. }]
		));
	}

	#[test]
	fn wrong_codec_native_options_fail_unless_preferred_drop_is_explicit() {
		let option = NativeOptionRequirement {
			codec:    CodecId::from("openai"),
			strength: RequirementStrength::Required,
			feature:  FeatureId(Str::from("openai-prediction")),
		};
		assert_eq!(
			negotiate_native_option(Some(&option), &CodecId::from("anthropic"), true)
				.expect_err("required mismatch")
				.kind,
			ErrorKind::CodecMismatch,
		);
		let preferred =
			NativeOptionRequirement { strength: RequirementStrength::Preferred, ..option };
		assert!(matches!(
			negotiate_native_option(Some(&preferred), &CodecId::from("anthropic"), true).unwrap(),
			Some(NegotiationDecision::Dropped { .. })
		));
	}

	#[test]
	fn one_shot_replay_and_staging_are_explicit() {
		let one_shot = ReplayRequirements {
			replayability:           Replayability::OneShot,
			semantic_retry_possible: false,
			staging_explicit:        false,
			staging_limit:           None,
		};
		assert_eq!(
			plan_replay(&one_shot, &budget(2, 0))
				.expect_err("multiple attempts")
				.kind,
			ErrorKind::ReplayRequired
		);

		let semantic = ReplayRequirements { semantic_retry_possible: true, ..one_shot };
		assert_eq!(
			plan_replay(&semantic, &budget(1, 64))
				.expect_err("implicit staging")
				.kind,
			ErrorKind::StagingRequired
		);
		let staged =
			ReplayRequirements { staging_explicit: true, staging_limit: Some(128), ..semantic };
		assert_eq!(plan_replay(&staged, &budget(2, 64)).unwrap(), ReplayPlan::SecureStaging {
			maximum_bytes: 64,
		});
	}

	#[test]
	fn expiry_revision_and_registry_generation_make_plans_stale() {
		let now = Instant::now();
		let expiry = now + Duration::from_secs(1);
		let revision = CatalogRevision::from("r1");
		assert!(plan_is_current(now, expiry, &revision, &revision, 7, 7));
		assert!(!plan_is_current(
			expiry + Duration::from_nanos(1),
			expiry,
			&revision,
			&revision,
			7,
			7
		));
		assert!(!plan_is_current(now, expiry, &revision, &CatalogRevision::from("r2"), 7, 7));
		assert!(!plan_is_current(now, expiry, &revision, &revision, 7, 8));
	}
}
