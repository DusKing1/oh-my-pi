//! Exact catalog selection, immutable overlays, aliases, fallback chains, and
//! constraints.

use std::{collections::BTreeMap, error::Error, fmt};

use omp_core::Str;
use serde::{Deserialize, Serialize};

use crate::{
	AuthSpecId, Availability, CatalogAlias, CatalogRevision, CodecId, ContextStrategy,
	EmbeddingFormatBits, EvidenceConfidence, FamilyId, GrammarBits, HostedToolBits, ModalityBits,
	ModelAvailability, ModelCapabilities, ModelKey, ModelLimits, ModelRemoteCompaction, ModelSpec,
	OperationKind, PolicyModel, PremiumMultiplier, Pricing, ProvenanceKind, ProvenanceSource,
	ProviderDef, ProviderId, ReasoningFeatureBits, RoleBits, RouteDef, RouteId, RouteRestrictions,
	SamplingControlBits, StructuredOutputBits, TextVerbosityBits, ThinkingPolicyId, ThinkingRouting,
	ToolFeatureBits, TransportKind, TrustDomain, WireModelId, WirePolicyId,
};

/// An exact provider and normalized-model selector.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct ExactSelector {
	/// Commercial or local provider domain.
	pub provider: ProviderId,
	/// Stable normalized model key.
	pub model:    ModelKey,
}

impl ExactSelector {
	/// Creates an exact selector without parsing or normalizing either
	/// identifier.
	pub fn new(provider: impl Into<ProviderId>, model: impl Into<ModelKey>) -> Self {
		Self { provider: provider.into(), model: model.into() }
	}
}

/// A provider-scoped exact alias selector.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct AliasSelector {
	/// Provider in whose model namespace the alias is resolved.
	pub provider: ProviderId,
	/// Alias spelling, matched byte-for-byte.
	pub alias:    Str,
}

impl AliasSelector {
	/// Creates a provider-scoped alias selector without normalizing its
	/// spelling.
	pub fn new(provider: impl Into<ProviderId>, alias: impl Into<Str>) -> Self {
		Self { provider: provider.into(), alias: alias.into() }
	}
}

/// An exact selector or an exact, declaratively registered alias.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum ModelSelector {
	/// Selects one provider/model pair exactly.
	Exact(ExactSelector),
	/// Selects one provider-scoped alias exactly.
	Alias(AliasSelector),
}

/// An explicitly ordered cross-model fallback chain.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FallbackChain {
	/// First selector attempted.
	pub primary:   ModelSelector,
	/// Additional selectors in caller-declared order.
	pub fallbacks: Box<[ModelSelector]>,
}

impl FallbackChain {
	/// Creates a chain containing only an exact primary selector.
	pub fn exact(primary: ExactSelector) -> Self {
		Self { primary: ModelSelector::Exact(primary), fallbacks: Box::new([]) }
	}

	pub fn iter(&self) -> impl Iterator<Item = &ModelSelector> + DoubleEndedIterator + '_ {
		std::iter::once(&self.primary).chain(self.fallbacks.iter())
	}
}

/// A provider-scoped alias supplied by a discovery or user overlay.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ScopedAlias {
	/// Provider namespace in which the alias is visible.
	pub provider:   ProviderId,
	/// Canonical compiler alias record.
	pub definition: CatalogAlias,
}

/// Model field names tracked by field-level provenance.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelField {
	/// Normalized family.
	Family,
	/// Display name.
	DisplayName,
	/// Route-specific wire identifiers.
	WireIds,
	/// Eligible route order.
	Routes,
	/// Typed capability evidence.
	Capabilities,
	/// Token and batch limits.
	Limits,
	/// Reasoning policy.
	Thinking,
	/// Model-specific effort spelling and wire-model routing.
	ThinkingRouting,
	/// Wire policy.
	WirePolicy,
	/// Context strategy.
	Context,
	/// Price schedule.
	Pricing,
	/// Availability state.
	Availability,
	/// Context-promotion target.
	ContextPromotionTarget,
	/// Remote compaction contract.
	RemoteCompaction,
	/// Premium quota multiplier.
	PremiumMultiplier,
	/// Latest provider update time.
	UpdatedAt,
	/// Temporary block expiry.
	BlockedUntil,
	/// Deprecation state.
	Deprecated,
}

/// Route field names tracked by field-level provenance.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouteField {
	/// Wire codec.
	Codec,
	/// Network or local transport.
	Transport,
	/// Endpoint URL and region.
	Endpoint,
	/// Authentication specification.
	Auth,
	/// Static header profile.
	Headers,
	/// Discovery specification.
	Discovery,
	/// Route capability restrictions.
	CapabilityLimits,
	/// Endpoint trust boundary.
	TrustDomain,
	/// Route priority.
	Priority,
}

/// Field-granular provenance for a resolved model and its eligible routes.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FieldProvenance {
	/// Winning evidence source for each model field.
	pub model:  BTreeMap<ModelField, ProvenanceSource>,
	/// Winning evidence source for each route field, keyed first by route.
	pub routes: BTreeMap<RouteId, BTreeMap<RouteField, ProvenanceSource>>,
}

/// A partial model replacement; omitted fields retain lower-precedence values.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelPatch {
	/// Replacement family.
	pub family: Option<FamilyId>,
	/// Replacement display name.
	pub display_name: Option<Str>,
	/// Replacement route/wire-id pairs.
	pub wire_ids: Option<Box<[(RouteId, WireModelId)]>>,
	/// Replacement route order.
	pub routes: Option<Box<[RouteId]>>,
	/// Replacement capability evidence.
	pub capabilities: Option<ModelCapabilities>,
	/// Replacement limits.
	pub limits: Option<ModelLimits>,
	/// `Some(None)` explicitly clears the reasoning policy.
	pub thinking: Option<Option<ThinkingPolicyId>>,
	/// Replacement model-specific effort spelling and wire-model routing.
	pub thinking_routing: Option<ThinkingRouting>,
	/// Replacement wire policy.
	pub wire_policy: Option<WirePolicyId>,
	/// Replacement context strategy.
	pub context: Option<ContextStrategy>,
	/// Replacement pricing.
	pub pricing: Option<Pricing>,
	/// Replacement model availability.
	pub availability: Option<ModelAvailability>,
	/// `Some(None)` explicitly clears context promotion.
	pub context_promotion_target: Option<Option<ModelKey>>,
	/// `Some(None)` explicitly clears remote compaction.
	pub remote_compaction: Option<Option<ModelRemoteCompaction>>,
	/// `Some(None)` explicitly clears the premium multiplier.
	pub premium_multiplier_millionths: Option<Option<PremiumMultiplier>>,
	/// `Some(None)` explicitly clears the latest provider update time.
	pub updated_at_ms: Option<Option<u64>>,
	/// `Some(None)` explicitly clears the block expiry.
	pub blocked_until_ms: Option<Option<u64>>,
	/// Replacement deprecation state.
	pub deprecated: Option<bool>,
}

/// A model addition or partial replacement in one overlay layer.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelOverlay {
	/// Exact provider/model pair affected by this entry.
	pub selector: ExactSelector,
	/// Complete record used when the base catalog has no matching model.
	pub added:    Option<ModelSpec>,
	/// Field-granular changes applied after an optional addition.
	pub patch:    ModelPatch,
}

/// A partial route replacement; omitted fields retain lower-precedence values.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct RoutePatch {
	/// Replacement codec.
	pub codec:             Option<CodecId>,
	/// Replacement transport.
	pub transport:         Option<TransportKind>,
	/// Replacement endpoint.
	pub endpoint:          Option<crate::EndpointSpec>,
	/// Replacement authentication specification.
	pub auth:              Option<AuthSpecId>,
	/// Replacement header profile.
	pub headers:           Option<crate::HeaderProfileId>,
	/// `Some(None)` explicitly disables discovery.
	pub discovery:         Option<Option<crate::DiscoverySpecId>>,
	/// Replacement route restrictions.
	pub capability_limits: Option<RouteRestrictions>,
	/// Replacement trust boundary.
	pub trust_domain:      Option<TrustDomain>,
	/// `Some(None)` explicitly clears priority.
	pub priority:          Option<Option<u32>>,
}

/// A route addition or partial replacement in one overlay layer.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RouteOverlay {
	/// Route affected by this entry.
	pub route: RouteId,
	/// Complete route used when the base catalog has no matching route.
	pub added: Option<RouteDef>,
	/// Field-granular changes applied after an optional addition.
	pub patch: RoutePatch,
}

/// One immutable overlay layer.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CatalogOverlay {
	/// Evidence source assigned to every field changed by this layer.
	pub source:  ProvenanceSource,
	/// Model additions and patches.
	pub models:  Box<[ModelOverlay]>,
	/// Route additions and patches.
	pub routes:  Box<[RouteOverlay]>,
	/// Exact alias additions or replacements.
	pub aliases: Box<[ScopedAlias]>,
}

/// Explicit authority for security-sensitive configured route changes.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct UnsafeTrustScope {
	endpoint_trust: bool,
	auth_trust:     bool,
}

impl UnsafeTrustScope {
	/// Grants both endpoint and authentication trust authority.
	pub const ALL: Self = Self { endpoint_trust: true, auth_trust: true };
	/// Grants authority to change authentication requirements.
	pub const AUTH: Self = Self { endpoint_trust: false, auth_trust: true };
	/// Grants authority to change endpoint and redirect trust boundaries.
	pub const ENDPOINT: Self = Self { endpoint_trust: true, auth_trust: false };
	/// Grants no security-sensitive override authority.
	pub const NONE: Self = Self { endpoint_trust: false, auth_trust: false };
}

/// A typed capability requirement used during deterministic resolution.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "capability", content = "required")]
pub enum CapabilityConstraint {
	/// Chat roles.
	ChatRoles(RoleBits),
	/// Tool-call behaviors.
	ToolFeatures(ToolFeatureBits),
	/// Structured output forms.
	StructuredOutput(StructuredOutputBits),
	/// Grammar languages.
	Grammar(GrammarBits),
	/// Text verbosity controls.
	TextVerbosity(TextVerbosityBits),
	/// Reasoning behaviors.
	Reasoning(ReasoningFeatureBits),
	/// Chat input modalities.
	InputModalities(ModalityBits),
	/// Hosted chat tools.
	HostedTools(HostedToolBits),
	/// Sampling controls.
	Sampling(SamplingControlBits),
	/// Embedding output formats.
	EmbeddingFormats(EmbeddingFormatBits),
}

/// Typed model and route constraints.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResolutionConstraints {
	/// Operation that must have positive support evidence.
	pub operation:              OperationKind,
	/// Minimum model context size.
	pub minimum_context_tokens: Option<u64>,
	/// Minimum model output size.
	pub minimum_output_tokens:  Option<u64>,
	/// Additional typed positive-evidence requirements.
	pub capabilities:           Box<[CapabilityConstraint]>,
}

/// Why one exact selector did not satisfy its constraints.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ConstraintFailure {
	/// The operation lacks positive support evidence.
	OperationUnknown(OperationKind),
	/// The model or route has an insufficient declared limit.
	Limit { field: Str, required: u64, available: Option<u64> },
	/// A required capability is explicitly unsupported.
	Unsupported(CapabilityConstraint),
	/// A required capability lacks positive evidence.
	Unknown(CapabilityConstraint),
	/// Every provider-matching route rejects the operation or its requested
	/// limits.
	NoEligibleRoute(OperationKind),
}

/// One eligible route after provider filtering and deterministic ranking.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResolvedRoute {
	/// Route identifier.
	pub id:       RouteId,
	/// Route priority, where larger values sort first.
	pub priority: Option<u32>,
}

/// Router-safe result of exact model resolution.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResolvedModel {
	/// Canonical exact selector after alias expansion.
	pub selector:   ExactSelector,
	/// Router-facing facts without raw wire model identifiers.
	pub policy:     PolicyModel,
	/// Eligible routes ordered by descending priority then ascending route id.
	pub routes:     Box<[ResolvedRoute]>,
	/// Winning source for every model and route field.
	pub provenance: FieldProvenance,
}

/// Catalog overlay or resolution failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResolveError {
	/// An overlay source was supplied to the wrong precedence tier.
	WrongOverlayKind { expected: ProvenanceKind, actual: ProvenanceKind },
	/// A configured endpoint or trust-domain change lacked explicit authority.
	UnsafeEndpointChange(RouteId),
	/// A configured authentication change lacked explicit authority.
	UnsafeAuthChange(RouteId),
	/// A model addition did not match its selector key.
	MismatchedModelAddition(ExactSelector),
	/// A route addition did not match its declared route id.
	MismatchedRouteAddition(RouteId),
	/// An exact provider/model pair was not found.
	ModelNotFound(ExactSelector),
	/// A selected provider was not declared.
	ProviderNotFound(ProviderId),
	/// No route connects the selected provider and model.
	NoEligibleRoute(ExactSelector),
	/// A provider-scoped alias was not declared exactly.
	AliasNotFound(AliasSelector),
	/// The exact selector failed typed constraints.
	Constraints { selector: ExactSelector, failures: Box<[ConstraintFailure]> },
	/// Every explicitly named selector failed.
	FallbacksExhausted(Box<[ResolveError]>),
}

impl fmt::Display for ResolveError {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(formatter, "catalog resolution failed: {self:?}")
	}
}

impl Error for ResolveError {}

/// Borrowed immutable bundled catalog input.
pub struct BundledCatalog<'a> {
	/// Bundled providers.
	pub providers: &'a [ProviderDef],
	/// Bundled routes.
	pub routes:    &'a [RouteDef],
	/// Bundled models.
	pub models:    &'a [ModelSpec],
	/// Bundled aliases.
	pub aliases:   &'a [CatalogAlias],
	/// Bundled field provenance source.
	pub source:    ProvenanceSource,
}

/// Resolver over an immutable bundled catalog plus ordered immutable overlays.
pub struct CatalogResolver<'a> {
	base:      BundledCatalog<'a>,
	discovery: Vec<CatalogOverlay>,
	user:      Vec<CatalogOverlay>,
}

impl<'a> CatalogResolver<'a> {
	/// Creates a resolver borrowing, but never mutating, the bundled catalog.
	pub fn new(base: BundledCatalog<'a>) -> Self {
		Self { base, discovery: Vec::new(), user: Vec::new() }
	}

	/// Adds a runtime-discovery overlay after validating its precedence class.
	pub fn add_discovery(&mut self, overlay: CatalogOverlay) -> Result<(), ResolveError> {
		if overlay.source.kind != ProvenanceKind::Discovered {
			return Err(ResolveError::WrongOverlayKind {
				expected: ProvenanceKind::Discovered,
				actual:   overlay.source.kind,
			});
		}
		validate_overlay(&overlay, UnsafeTrustScope::NONE)?;
		self.discovery.push(overlay);
		Ok(())
	}

	/// Adds a user overlay after validating security-sensitive changes against
	/// `scope`.
	pub fn add_user(
		&mut self,
		overlay: CatalogOverlay,
		scope: UnsafeTrustScope,
	) -> Result<(), ResolveError> {
		if overlay.source.kind != ProvenanceKind::Configured {
			return Err(ResolveError::WrongOverlayKind {
				expected: ProvenanceKind::Configured,
				actual:   overlay.source.kind,
			});
		}
		validate_overlay(&overlay, scope)?;
		self.user.push(overlay);
		Ok(())
	}

	/// Resolves the first satisfiable selector in an explicit fallback chain.
	pub fn resolve(
		&self,
		chain: &FallbackChain,
		constraints: &ResolutionConstraints,
	) -> Result<ResolvedModel, ResolveError> {
		let mut failures = Vec::new();
		for selector in chain.iter() {
			match self.resolve_one(selector, constraints) {
				Ok(resolved) => return Ok(resolved),
				Err(error) => failures.push(error),
			}
		}
		Err(ResolveError::FallbacksExhausted(failures.into_boxed_slice()))
	}

	/// Resolves every satisfiable explicitly named selector without inventing
	/// candidates.
	pub fn resolve_candidates(
		&self,
		chain: &FallbackChain,
		constraints: &ResolutionConstraints,
	) -> Vec<Result<ResolvedModel, ResolveError>> {
		chain
			.iter()
			.map(|selector| self.resolve_one(selector, constraints))
			.collect()
	}

	fn resolve_one(
		&self,
		selector: &ModelSelector,
		constraints: &ResolutionConstraints,
	) -> Result<ResolvedModel, ResolveError> {
		let exact = self.expand_selector(selector)?;
		if !self
			.base
			.providers
			.iter()
			.any(|provider| provider.id == exact.provider)
		{
			return Err(ResolveError::ProviderNotFound(exact.provider));
		}
		let mut model = self
			.base
			.models
			.iter()
			.find(|model| {
				model.key == exact.model && model_has_provider(model, &exact.provider, self.base.routes)
			})
			.cloned();
		let mut model_sources = all_model_sources(self.base.source.clone());
		for overlay in self.discovery.iter().chain(self.user.iter()) {
			for entry in overlay
				.models
				.iter()
				.filter(|entry| entry.selector == exact)
			{
				if model.is_none() {
					let added = entry
						.added
						.clone()
						.ok_or_else(|| ResolveError::ModelNotFound(exact.clone()))?;
					if added.key != exact.model {
						return Err(ResolveError::MismatchedModelAddition(exact.clone()));
					}
					model = Some(added);
					model_sources = all_model_sources(overlay.source.clone());
				}
				if let Some(added) = &entry.added {
					let model = model.as_mut().expect("model initialized above");
					let mut evidence = model.provenance.sources.to_vec();
					for source in &added.provenance.sources {
						if !evidence.contains(source) {
							evidence.push(source.clone());
						}
					}
					model.provenance.sources = evidence.into_boxed_slice();
				}
				apply_model_patch(
					model.as_mut().expect("model initialized above"),
					&entry.patch,
					&overlay.source,
					&mut model_sources,
				);
			}
		}
		let model = model.ok_or_else(|| ResolveError::ModelNotFound(exact.clone()))?;
		let mut routes = Vec::new();
		let mut route_sources = BTreeMap::new();
		for route_id in model.routes.iter() {
			let mut route = self
				.base
				.routes
				.iter()
				.find(|route| route.id == *route_id)
				.cloned();
			let mut sources = all_route_sources(self.base.source.clone());
			for overlay in self.discovery.iter().chain(self.user.iter()) {
				for entry in overlay
					.routes
					.iter()
					.filter(|entry| entry.route == *route_id)
				{
					if route.is_none() {
						let added = entry
							.added
							.clone()
							.ok_or_else(|| ResolveError::MismatchedRouteAddition(route_id.clone()))?;
						if added.id != *route_id {
							return Err(ResolveError::MismatchedRouteAddition(route_id.clone()));
						}
						route = Some(added);
						sources = all_route_sources(overlay.source.clone());
					}
					apply_route_patch(
						route.as_mut().expect("route initialized above"),
						&entry.patch,
						&overlay.source,
						&mut sources,
					);
				}
			}
			if let Some(route) = route.filter(|route| route.provider == exact.provider) {
				route_sources.insert(route.id.clone(), sources);
				routes.push(route);
			}
		}
		if routes.is_empty() {
			return Err(ResolveError::NoEligibleRoute(exact));
		}
		let failures = constraint_failures(&model, constraints);
		if !failures.is_empty() {
			return Err(ResolveError::Constraints {
				selector: exact,
				failures: failures.into_boxed_slice(),
			});
		}
		routes.retain(|route| route_satisfies(route, constraints));
		route_sources.retain(|route, _| routes.iter().any(|candidate| candidate.id == *route));
		if routes.is_empty() {
			return Err(ResolveError::Constraints {
				selector: exact,
				failures: Box::new([ConstraintFailure::NoEligibleRoute(constraints.operation)]),
			});
		}
		routes.sort_by(|left, right| {
			right
				.priority
				.unwrap_or(0)
				.cmp(&left.priority.unwrap_or(0))
				.then_with(|| left.id.cmp(&right.id))
		});
		let resolved_routes = routes
			.into_iter()
			.map(|route| ResolvedRoute { id: route.id, priority: route.priority })
			.collect::<Vec<_>>()
			.into_boxed_slice();
		Ok(ResolvedModel {
			selector:   exact,
			policy:     PolicyModel::from(&model),
			routes:     resolved_routes,
			provenance: FieldProvenance { model: model_sources, routes: route_sources },
		})
	}

	fn expand_selector(&self, selector: &ModelSelector) -> Result<ExactSelector, ResolveError> {
		match selector {
			ModelSelector::Exact(exact) => Ok(exact.clone()),
			ModelSelector::Alias(alias) => self
				.alias_target(alias)
				.map(|model| ExactSelector { provider: alias.provider.clone(), model })
				.ok_or_else(|| ResolveError::AliasNotFound(alias.clone())),
		}
	}

	fn alias_target(&self, selector: &AliasSelector) -> Option<ModelKey> {
		let mut target = self
			.base
			.aliases
			.iter()
			.find(|entry| entry.alias == selector.alias)
			.map(|entry| entry.target.clone());
		for overlay in self.discovery.iter().chain(self.user.iter()) {
			if let Some(entry) = overlay.aliases.iter().find(|entry| {
				entry.provider == selector.provider && entry.definition.alias == selector.alias
			}) {
				target = Some(entry.definition.target.clone());
			}
		}
		target
	}
}

fn validate_overlay(overlay: &CatalogOverlay, scope: UnsafeTrustScope) -> Result<(), ResolveError> {
	for route in &overlay.routes {
		if (route.added.is_some()
			|| route.patch.endpoint.is_some()
			|| route.patch.trust_domain.is_some())
			&& !scope.endpoint_trust
		{
			return Err(ResolveError::UnsafeEndpointChange(route.route.clone()));
		}
		if (route.added.is_some() || route.patch.auth.is_some()) && !scope.auth_trust {
			return Err(ResolveError::UnsafeAuthChange(route.route.clone()));
		}
		if let Some(added) = &route.added {
			if added.id != route.route {
				return Err(ResolveError::MismatchedRouteAddition(route.route.clone()));
			}
		}
	}
	for model in &overlay.models {
		if let Some(added) = &model.added {
			if added.key != model.selector.model {
				return Err(ResolveError::MismatchedModelAddition(model.selector.clone()));
			}
		}
	}
	Ok(())
}

fn model_has_provider(model: &ModelSpec, provider: &ProviderId, routes: &[RouteDef]) -> bool {
	model.routes.iter().any(|route_id| {
		routes
			.iter()
			.any(|route| route.id == *route_id && route.provider == *provider)
	})
}

fn all_model_sources(source: ProvenanceSource) -> BTreeMap<ModelField, ProvenanceSource> {
	[
		ModelField::Family,
		ModelField::DisplayName,
		ModelField::WireIds,
		ModelField::Routes,
		ModelField::Capabilities,
		ModelField::Limits,
		ModelField::Thinking,
		ModelField::ThinkingRouting,
		ModelField::WirePolicy,
		ModelField::Context,
		ModelField::Pricing,
		ModelField::Availability,
		ModelField::ContextPromotionTarget,
		ModelField::RemoteCompaction,
		ModelField::PremiumMultiplier,
		ModelField::UpdatedAt,
		ModelField::BlockedUntil,
		ModelField::Deprecated,
	]
	.into_iter()
	.map(|field| (field, source.clone()))
	.collect()
}

fn all_route_sources(source: ProvenanceSource) -> BTreeMap<RouteField, ProvenanceSource> {
	[
		RouteField::Codec,
		RouteField::Transport,
		RouteField::Endpoint,
		RouteField::Auth,
		RouteField::Headers,
		RouteField::Discovery,
		RouteField::CapabilityLimits,
		RouteField::TrustDomain,
		RouteField::Priority,
	]
	.into_iter()
	.map(|field| (field, source.clone()))
	.collect()
}

macro_rules! patch_field {
	($patch:expr, $target:expr, $member:ident, $field:expr, $source:expr, $sources:expr) => {
		if let Some(value) = &$patch.$member {
			$target.$member = value.clone();
			$sources.insert($field, $source.clone());
		}
	};
}

fn apply_model_patch(
	model: &mut ModelSpec,
	patch: &ModelPatch,
	source: &ProvenanceSource,
	sources: &mut BTreeMap<ModelField, ProvenanceSource>,
) {
	patch_field!(patch, model, family, ModelField::Family, source, sources);
	patch_field!(patch, model, display_name, ModelField::DisplayName, source, sources);
	patch_field!(patch, model, wire_ids, ModelField::WireIds, source, sources);
	patch_field!(patch, model, routes, ModelField::Routes, source, sources);
	patch_field!(patch, model, capabilities, ModelField::Capabilities, source, sources);
	patch_field!(patch, model, limits, ModelField::Limits, source, sources);
	patch_field!(patch, model, thinking, ModelField::Thinking, source, sources);
	patch_field!(patch, model, thinking_routing, ModelField::ThinkingRouting, source, sources);
	patch_field!(patch, model, wire_policy, ModelField::WirePolicy, source, sources);
	patch_field!(patch, model, context, ModelField::Context, source, sources);
	patch_field!(patch, model, pricing, ModelField::Pricing, source, sources);
	patch_field!(patch, model, availability, ModelField::Availability, source, sources);
	patch_field!(
		patch,
		model,
		context_promotion_target,
		ModelField::ContextPromotionTarget,
		source,
		sources
	);
	patch_field!(patch, model, remote_compaction, ModelField::RemoteCompaction, source, sources);
	patch_field!(
		patch,
		model,
		premium_multiplier_millionths,
		ModelField::PremiumMultiplier,
		source,
		sources
	);
	if let Some(value) = patch.updated_at_ms {
		model.provenance.updated_at_ms = value;
		sources.insert(ModelField::UpdatedAt, source.clone());
	}
	if let Some(value) = patch.blocked_until_ms {
		model.provenance.blocked_until_ms = value;
		sources.insert(ModelField::BlockedUntil, source.clone());
	}
	if let Some(value) = patch.deprecated {
		model.provenance.deprecated = value;
		sources.insert(ModelField::Deprecated, source.clone());
	}
	if !model
		.provenance
		.sources
		.iter()
		.any(|existing| existing == source)
	{
		let mut evidence = model.provenance.sources.to_vec();
		evidence.push(source.clone());
		model.provenance.sources = evidence.into_boxed_slice();
	}
}

fn apply_route_patch(
	route: &mut RouteDef,
	patch: &RoutePatch,
	source: &ProvenanceSource,
	sources: &mut BTreeMap<RouteField, ProvenanceSource>,
) {
	patch_field!(patch, route, codec, RouteField::Codec, source, sources);
	patch_field!(patch, route, transport, RouteField::Transport, source, sources);
	patch_field!(patch, route, endpoint, RouteField::Endpoint, source, sources);
	patch_field!(patch, route, auth, RouteField::Auth, source, sources);
	patch_field!(patch, route, headers, RouteField::Headers, source, sources);
	patch_field!(patch, route, discovery, RouteField::Discovery, source, sources);
	patch_field!(patch, route, capability_limits, RouteField::CapabilityLimits, source, sources);
	patch_field!(patch, route, trust_domain, RouteField::TrustDomain, source, sources);
	patch_field!(patch, route, priority, RouteField::Priority, source, sources);
}

fn constraint_failures(
	model: &ModelSpec,
	constraints: &ResolutionConstraints,
) -> Vec<ConstraintFailure> {
	let mut failures = Vec::new();
	if !model
		.capabilities
		.operations
		.contains_kind(constraints.operation)
	{
		failures.push(ConstraintFailure::OperationUnknown(constraints.operation));
	}
	check_limit(
		"context_tokens",
		constraints.minimum_context_tokens,
		model.limits.context_window,
		&mut failures,
	);
	check_limit(
		"output_tokens",
		constraints.minimum_output_tokens,
		model.limits.maximum_output_tokens,
		&mut failures,
	);
	for requirement in &constraints.capabilities {
		match capability_support(&model.capabilities, requirement) {
			CapabilitySupport::Supported => {},
			CapabilitySupport::Unsupported => {
				failures.push(ConstraintFailure::Unsupported(requirement.clone()))
			},
			CapabilitySupport::Unknown => {
				failures.push(ConstraintFailure::Unknown(requirement.clone()))
			},
		}
	}
	failures
}

fn route_satisfies(route: &RouteDef, constraints: &ResolutionConstraints) -> bool {
	if route
		.capability_limits
		.operations
		.is_some_and(|allowed| !allowed.contains_kind(constraints.operation))
	{
		return false;
	}
	if route
		.capability_limits
		.maximum_context_tokens
		.zip(constraints.minimum_context_tokens)
		.is_some_and(|(available, required)| available < required)
	{
		return false;
	}
	!route
		.capability_limits
		.maximum_output_tokens
		.zip(constraints.minimum_output_tokens)
		.is_some_and(|(available, required)| available < required)
}

fn check_limit(
	field: &str,
	required: Option<u64>,
	available: Option<u64>,
	failures: &mut Vec<ConstraintFailure>,
) {
	if let Some(required) = required {
		if available.is_none_or(|available| available < required) {
			failures.push(ConstraintFailure::Limit { field: field.into(), required, available });
		}
	}
}

#[derive(Clone, Copy)]
enum CapabilitySupport {
	Supported,
	Unsupported,
	Unknown,
}

fn availability_bits<C>(
	availability: Option<&Availability<C>>,
	contains: impl FnOnce(&C) -> bool,
) -> CapabilitySupport {
	match availability {
		Some(Availability::Native(value) | Availability::Emulated { constraints: value, .. }) => {
			if contains(value) {
				CapabilitySupport::Supported
			} else {
				CapabilitySupport::Unsupported
			}
		},
		Some(Availability::Unsupported) => CapabilitySupport::Unsupported,
		Some(Availability::Unknown) | None => CapabilitySupport::Unknown,
	}
}

fn capability_support(
	capabilities: &ModelCapabilities,
	requirement: &CapabilityConstraint,
) -> CapabilitySupport {
	let chat = capabilities.chat.as_ref();
	match requirement {
		CapabilityConstraint::ChatRoles(required) => {
			availability_bits(chat.map(|value| &value.roles), |value| value.contains(*required))
		},
		CapabilityConstraint::ToolFeatures(required) => match chat.map(|value| &value.tools) {
			Some(Availability::Native(value) | Availability::Emulated { constraints: value, .. }) => {
				if value.features.contains(*required) {
					CapabilitySupport::Supported
				} else {
					CapabilitySupport::Unsupported
				}
			},
			Some(Availability::Unsupported) => CapabilitySupport::Unsupported,
			Some(Availability::Unknown) | None => CapabilitySupport::Unknown,
		},
		CapabilityConstraint::StructuredOutput(required) => {
			availability_bits(chat.map(|value| &value.structured_output), |value| {
				value.contains(*required)
			})
		},
		CapabilityConstraint::Grammar(required) => {
			availability_bits(chat.map(|value| &value.grammar), |value| value.contains(*required))
		},
		CapabilityConstraint::TextVerbosity(required) => {
			availability_bits(chat.map(|value| &value.text_verbosity), |value| {
				value.contains(*required)
			})
		},
		CapabilityConstraint::Reasoning(required) => {
			availability_bits(chat.map(|value| &value.reasoning), |value| {
				value.features.contains(*required)
			})
		},
		CapabilityConstraint::InputModalities(required) => {
			availability_bits(chat.map(|value| &value.input_modalities), |value| {
				value.contains(*required)
			})
		},
		CapabilityConstraint::HostedTools(required) => {
			availability_bits(chat.map(|value| &value.hosted_tools), |value| value.contains(*required))
		},
		CapabilityConstraint::Sampling(required) => {
			availability_bits(chat.map(|value| &value.sampling), |value| value.contains(*required))
		},
		CapabilityConstraint::EmbeddingFormats(required) => match capabilities.embeddings.as_ref() {
			Some(value) if value.formats.contains(*required) => CapabilitySupport::Supported,
			Some(_) => CapabilitySupport::Unsupported,
			None => CapabilitySupport::Unknown,
		},
	}
}

/// Creates a provenance source suitable for a synthetic bundled catalog in
/// tests or builders.
pub fn bundled_source(
	origin: impl Into<Str>,
	revision: Option<CatalogRevision>,
) -> ProvenanceSource {
	ProvenanceSource {
		kind: ProvenanceKind::Bundled,
		origin: origin.into(),
		revision,
		confidence: EvidenceConfidence::Verified,
		observed_at_ms: None,
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{
		CodexTransportPreference, EndpointSpec, HeaderProfileId, ManagementCapabilities,
		ModelProvenance, RedirectTrust, RegistryMapping, StructuredOutputBits, TransportKind,
	};

	fn source(kind: ProvenanceKind, origin: &str) -> ProvenanceSource {
		ProvenanceSource {
			kind,
			origin: origin.into(),
			revision: None,
			confidence: EvidenceConfidence::Declared,
			observed_at_ms: None,
		}
	}

	fn provider(id: &str, routes: &[&str]) -> ProviderDef {
		ProviderDef {
			id:                 id.into(),
			name:               id.into(),
			auth:               Box::new([AuthSpecId::from("auth")]),
			management:         ManagementCapabilities {
				operations:        crate::OperationBits::empty(),
				multiple_accounts: false,
				refresh:           false,
				principal_quota:   false,
			},
			routes:             routes
				.iter()
				.map(|route| RouteId::from(*route))
				.collect::<Vec<_>>()
				.into_boxed_slice(),
			wire_policy:        WirePolicyId::from("wire"),
			discovery_defaults: None,
			mapping:            RegistryMapping::Concrete,
		}
	}

	fn route(id: &str, provider: &str, priority: u32) -> RouteDef {
		RouteDef {
			id:                 id.into(),
			provider:           provider.into(),
			codec:              CodecId::from("codec"),
			codec_profile:      crate::CodecProfile::default(),
			transport:          TransportKind::Http,
			endpoint:           EndpointSpec {
				base_url: format!("https://{id}.test").into(),
				region:   None,
			},
			auth:               AuthSpecId::from("auth"),
			headers:            HeaderProfileId::from("headers"),
			discovery:          None,
			capability_limits:  RouteRestrictions::default(),
			trust_domain:       TrustDomain {
				origin:          format!("https://{id}.test").into(),
				redirects:       RedirectTrust::SameOrigin,
				allow_plaintext: false,
			},
			codex_transport:    CodexTransportPreference::HttpOnly,
			use_responses_lite: None,
			priority:           Some(priority),
		}
	}

	fn model(key: &str, route_ids: &[&str], chat: bool) -> ModelSpec {
		let mut operations = crate::OperationBits::empty();
		let chat_capabilities = chat.then(|| {
			operations.insert_kind(OperationKind::Chat);
			crate::ChatCapabilities {
				roles:             Availability::Native(RoleBits::SYSTEM | RoleBits::DEVELOPER),
				mid_session_roles: Availability::Unknown,
				tools:             Availability::Unknown,
				structured_output: Availability::Native(StructuredOutputBits::JSON_OBJECT),
				grammar:           Availability::Unknown,
				text_verbosity:    Availability::Unknown,
				reasoning:         Availability::Unknown,
				input_modalities:  Availability::Unknown,
				hosted_tools:      Availability::Unknown,
				prompt_caching:    Availability::Unknown,
				service_tiers:     Availability::Unknown,
				sampling:          Availability::Unknown,
				safety:            Availability::Unknown,
				determinism:       Availability::Unknown,
				server_state:      Availability::Unknown,
				logprobs:          Availability::Unknown,
			}
		});
		ModelSpec {
			key: key.into(),
			family: FamilyId::from("family"),
			display_name: key.into(),
			wire_ids: route_ids
				.iter()
				.map(|route| (RouteId::from(*route), WireModelId::from(key)))
				.collect::<Vec<_>>()
				.into_boxed_slice(),
			routes: route_ids
				.iter()
				.map(|route| RouteId::from(*route))
				.collect::<Vec<_>>()
				.into_boxed_slice(),
			capabilities: ModelCapabilities {
				operations,
				chat: chat_capabilities,
				embeddings: None,
				image: None,
				video: None,
				speech: None,
				transcription: None,
				realtime: None,
				search: None,
				tokenization: None,
			},
			limits: ModelLimits {
				context_window:        Some(16_000),
				maximum_input_tokens:  Some(14_000),
				maximum_output_tokens: Some(2_000),
				maximum_batch:         None,
			},
			thinking: None,
			thinking_routing: ThinkingRouting::default(),
			wire_policy: WirePolicyId::from("wire"),
			context: ContextStrategy::Replay,
			pricing: Pricing::default(),
			availability: ModelAvailability::Available,
			provenance: ModelProvenance {
				sources:          Box::new([source(ProvenanceKind::Bundled, "base")]),
				updated_at_ms:    None,
				blocked_until_ms: None,
				deprecated:       false,
			},
			context_promotion_target: None,
			remote_compaction: None,
			premium_multiplier_millionths: None,
		}
	}

	fn constraints() -> ResolutionConstraints {
		ResolutionConstraints {
			operation:              OperationKind::Chat,
			minimum_context_tokens: None,
			minimum_output_tokens:  None,
			capabilities:           Box::new([]),
		}
	}

	#[test]
	fn precedence_is_field_granular_and_does_not_mutate_bundled_records() {
		let providers = [provider("p", &["r"])];
		let routes = [route("r", "p", 1)];
		let models = [model("m", &["r"], true)];
		let base_name = models[0].display_name.clone();
		let mut resolver = CatalogResolver::new(BundledCatalog {
			providers: &providers,
			routes:    &routes,
			models:    &models,
			aliases:   &[],
			source:    source(ProvenanceKind::Bundled, "base"),
		});
		resolver
			.add_discovery(CatalogOverlay {
				source:  source(ProvenanceKind::Discovered, "discovery"),
				models:  Box::new([ModelOverlay {
					selector: ExactSelector::new("p", "m"),
					added:    None,
					patch:    ModelPatch {
						display_name: Some("discovered".into()),
						limits: Some(ModelLimits { context_window: Some(32_000), ..models[0].limits }),
						wire_policy: Some(WirePolicyId::from("discovery-wire")),
						..ModelPatch::default()
					},
				}]),
				routes:  Box::new([RouteOverlay {
					route: RouteId::from("r"),
					added: None,
					patch: RoutePatch { priority: Some(Some(2)), ..RoutePatch::default() },
				}]),
				aliases: Box::new([]),
			})
			.expect("discovery overlay accepted");
		resolver
			.add_user(
				CatalogOverlay {
					source:  source(ProvenanceKind::Configured, "user"),
					models:  Box::new([ModelOverlay {
						selector: ExactSelector::new("p", "m"),
						added:    None,
						patch:    ModelPatch {
							display_name: Some("configured".into()),
							limits: Some(ModelLimits { context_window: Some(64_000), ..models[0].limits }),
							..ModelPatch::default()
						},
					}]),
					routes:  Box::new([RouteOverlay {
						route: RouteId::from("r"),
						added: None,
						patch: RoutePatch { priority: Some(Some(3)), ..RoutePatch::default() },
					}]),
					aliases: Box::new([]),
				},
				UnsafeTrustScope::NONE,
			)
			.expect("safe user overlay accepted");
		let resolved = resolver
			.resolve(&FallbackChain::exact(ExactSelector::new("p", "m")), &constraints())
			.expect("model resolves");
		assert_eq!(models[0].display_name, base_name);
		assert_eq!(resolved.policy.limits.context_window, Some(64_000));
		assert_eq!(resolved.provenance.model[&ModelField::DisplayName].origin, "user");
		assert_eq!(resolved.provenance.model[&ModelField::Limits].origin, "user");
		assert_eq!(resolved.policy.wire_policy, WirePolicyId::from("discovery-wire"));
		assert_eq!(resolved.provenance.model[&ModelField::WirePolicy].origin, "discovery");
		assert_eq!(routes[0].priority, Some(1));
		assert_eq!(resolved.routes[0].priority, Some(3));
		assert_eq!(
			resolved.provenance.routes[&RouteId::from("r")][&RouteField::Priority].origin,
			"user"
		);
	}

	#[test]
	fn endpoint_and_auth_changes_require_their_explicit_unsafe_scope() {
		let providers = [provider("p", &["r"])];
		let routes = [route("r", "p", 1)];
		let models = [model("m", &["r"], true)];
		let make = || CatalogOverlay {
			source:  source(ProvenanceKind::Configured, "user"),
			models:  Box::new([]),
			routes:  Box::new([RouteOverlay {
				route: RouteId::from("r"),
				added: None,
				patch: RoutePatch {
					endpoint: Some(EndpointSpec {
						base_url: "https://changed.test".into(),
						region:   None,
					}),
					auth: Some(AuthSpecId::from("other-auth")),
					..RoutePatch::default()
				},
			}]),
			aliases: Box::new([]),
		};
		let mut denied = CatalogResolver::new(BundledCatalog {
			providers: &providers,
			routes:    &routes,
			models:    &models,
			aliases:   &[],
			source:    source(ProvenanceKind::Bundled, "base"),
		});
		assert_eq!(
			denied.add_user(make(), UnsafeTrustScope::NONE),
			Err(ResolveError::UnsafeEndpointChange(RouteId::from("r")))
		);
		let mut endpoint_only = CatalogResolver::new(BundledCatalog {
			providers: &providers,
			routes:    &routes,
			models:    &models,
			aliases:   &[],
			source:    source(ProvenanceKind::Bundled, "base"),
		});
		assert_eq!(
			endpoint_only.add_user(make(), UnsafeTrustScope::ENDPOINT),
			Err(ResolveError::UnsafeAuthChange(RouteId::from("r")))
		);
		let mut allowed = CatalogResolver::new(BundledCatalog {
			providers: &providers,
			routes:    &routes,
			models:    &models,
			aliases:   &[],
			source:    source(ProvenanceKind::Bundled, "base"),
		});
		allowed
			.add_user(make(), UnsafeTrustScope::ALL)
			.expect("both scopes explicitly granted");
	}

	#[test]
	fn aliases_and_selectors_are_exact_even_for_adversarial_model_names() {
		let providers = [provider("p", &["r"])];
		let routes = [route("r", "p", 1)];
		let models = [model("gpt", &["r"], true), model("gpt-malicious-thinking", &["r"], true)];
		let aliases = [CatalogAlias {
			alias:      "safe".into(),
			target:     ModelKey::from("gpt"),
			rationale:  "test".into(),
			provenance: "test".into(),
		}];
		let resolver = CatalogResolver::new(BundledCatalog {
			providers: &providers,
			routes:    &routes,
			models:    &models,
			aliases:   &aliases,
			source:    source(ProvenanceKind::Bundled, "base"),
		});
		let alias = FallbackChain {
			primary:   ModelSelector::Alias(AliasSelector::new("p", "safe")),
			fallbacks: Box::new([]),
		};
		assert_eq!(
			resolver
				.resolve(&alias, &constraints())
				.expect("exact alias")
				.selector
				.model,
			"gpt"
		);
		let prefix = FallbackChain {
			primary:   ModelSelector::Alias(AliasSelector::new("p", "saf")),
			fallbacks: Box::new([]),
		};
		assert!(matches!(
			resolver.resolve(&prefix, &constraints()),
			Err(ResolveError::FallbacksExhausted(errors))
				if matches!(&errors[0], ResolveError::AliasNotFound(alias) if alias.alias == "saf")
		));
		assert_eq!(
			resolver
				.resolve(
					&FallbackChain::exact(ExactSelector::new("p", "gpt-malicious-thinking")),
					&constraints()
				)
				.expect("adversarial exact model")
				.selector
				.model,
			"gpt-malicious-thinking"
		);
	}

	#[test]
	fn alias_precedence_is_bundled_then_discovery_then_user() {
		let providers = [provider("p", &["r"])];
		let routes = [route("r", "p", 1)];
		let models = [
			model("bundled", &["r"], true),
			model("discovered", &["r"], true),
			model("user", &["r"], true),
		];
		let aliases = [CatalogAlias {
			alias:      "current".into(),
			target:     ModelKey::from("bundled"),
			rationale:  "test".into(),
			provenance: "test".into(),
		}];
		let mut resolver = CatalogResolver::new(BundledCatalog {
			providers: &providers,
			routes:    &routes,
			models:    &models,
			aliases:   &aliases,
			source:    source(ProvenanceKind::Bundled, "base"),
		});
		resolver
			.add_discovery(CatalogOverlay {
				source:  source(ProvenanceKind::Discovered, "discovery"),
				models:  Box::new([]),
				routes:  Box::new([]),
				aliases: Box::new([ScopedAlias {
					provider:   ProviderId::from("p"),
					definition: CatalogAlias {
						alias:      "current".into(),
						target:     ModelKey::from("discovered"),
						rationale:  "test".into(),
						provenance: "discovery".into(),
					},
				}]),
			})
			.expect("discovery alias accepted");
		resolver
			.add_user(
				CatalogOverlay {
					source:  source(ProvenanceKind::Configured, "user"),
					models:  Box::new([]),
					routes:  Box::new([]),
					aliases: Box::new([ScopedAlias {
						provider:   ProviderId::from("p"),
						definition: CatalogAlias {
							alias:      "current".into(),
							target:     ModelKey::from("user"),
							rationale:  "test".into(),
							provenance: "user".into(),
						},
					}]),
				},
				UnsafeTrustScope::NONE,
			)
			.expect("user alias accepted");
		let chain = FallbackChain {
			primary:   ModelSelector::Alias(AliasSelector::new("p", "current")),
			fallbacks: Box::new([]),
		};
		assert_eq!(
			resolver
				.resolve(&chain, &constraints())
				.expect("highest alias wins")
				.selector
				.model,
			"user"
		);
	}

	#[test]
	fn fallback_order_and_route_ties_are_deterministic_and_never_implicit() {
		let providers = [provider("p", &["z", "a"])];
		let routes = [route("z", "p", 7), route("a", "p", 7)];
		let models = [
			model("unknown", &["z"], false),
			model("good", &["z", "a"], true),
			model("also-good", &["a"], true),
		];
		let resolver = CatalogResolver::new(BundledCatalog {
			providers: &providers,
			routes:    &routes,
			models:    &models,
			aliases:   &[],
			source:    source(ProvenanceKind::Bundled, "base"),
		});
		let explicit = FallbackChain {
			primary:   ModelSelector::Exact(ExactSelector::new("p", "unknown")),
			fallbacks: Box::new([
				ModelSelector::Exact(ExactSelector::new("p", "good")),
				ModelSelector::Exact(ExactSelector::new("p", "also-good")),
			]),
		};
		let resolved = resolver
			.resolve(&explicit, &constraints())
			.expect("explicit fallback succeeds");
		assert_eq!(resolved.selector.model, "good");
		assert_eq!(
			resolved
				.routes
				.iter()
				.map(|route| route.id.as_str())
				.collect::<Vec<_>>(),
			["a", "z"]
		);
		let no_fallback = FallbackChain::exact(ExactSelector::new("p", "unknown"));
		assert!(
			matches!(resolver.resolve(&no_fallback, &constraints()), Err(ResolveError::FallbacksExhausted(errors)) if errors.len() == 1)
		);
	}

	#[test]
	fn typed_constraints_require_positive_evidence() {
		let providers = [provider("p", &["r"])];
		let routes = [route("r", "p", 1)];
		let models = [model("m", &["r"], true)];
		let resolver = CatalogResolver::new(BundledCatalog {
			providers: &providers,
			routes:    &routes,
			models:    &models,
			aliases:   &[],
			source:    source(ProvenanceKind::Bundled, "base"),
		});
		let required = ResolutionConstraints {
			operation:              OperationKind::Chat,
			minimum_context_tokens: Some(8_000),
			minimum_output_tokens:  Some(1_000),
			capabilities:           Box::new([
				CapabilityConstraint::StructuredOutput(StructuredOutputBits::JSON_OBJECT),
				CapabilityConstraint::Grammar(GrammarBits::EBNF),
			]),
		};
		assert!(matches!(
			resolver.resolve(&FallbackChain::exact(ExactSelector::new("p", "m")), &required),
			Err(ResolveError::FallbacksExhausted(errors))
				if matches!(&errors[0], ResolveError::Constraints { failures, .. }
					if failures.contains(&ConstraintFailure::Unknown(CapabilityConstraint::Grammar(GrammarBits::EBNF))))
		));
	}
}
