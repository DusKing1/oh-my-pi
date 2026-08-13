//! Runtime model-discovery normalization for the production registry.

use omp_llm_catalog::{
	ContextStrategy, Pricing, RouteId, ThinkingPolicyId, WirePolicyId,
	discover::{DiscoveredModel, DiscoveryDefaults, DiscoveryNormalizer, NormalizedDiscovery},
};

/// Normalizes provider-returned model rows conservatively before applying them
/// as runtime catalog overlays.
///
/// Missing evidence remains unknown; this module never infers capabilities from
/// provider or model names.
pub fn normalize(
	rows: &[DiscoveredModel],
	wire_policy: WirePolicyId,
	extended_wire_policy: Option<WirePolicyId>,
	thinking: Option<ThinkingPolicyId>,
) -> Result<Vec<NormalizedDiscovery>, omp_llm_catalog::discover::DiscoveryError> {
	DiscoveryNormalizer::new(DiscoveryDefaults {
		wire_policy,
		extended_wire_policy,
		context: ContextStrategy::Replay,
		thinking,
		pricing: Pricing::default(),
	})
	.normalize_batch(rows)
}

/// Returns the route restriction carried by an authenticated discovery request.
#[must_use]
pub fn route_scope(route: RouteId) -> RouteId {
	route
}
