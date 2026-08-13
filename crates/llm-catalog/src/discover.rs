//! Conservative normalization of runtime provider model discovery.

use std::{collections::BTreeMap, error::Error, fmt, sync::Arc};

use omp_core::Str;
use serde::{Deserialize, Serialize};

use crate::{
	Availability, CatalogAlias, CatalogOverlay, ChatCapabilities, ClassificationEvidence,
	ClassificationInput, ClassificationPhase, ContextStrategy, DiscoveryPagination, DiscoverySpec,
	DiscoverySpecId, EvidenceConfidence, ExactSelector, ExtendedContextMode, FamilyId,
	ModelAvailability, ModelCapabilities, ModelKey, ModelLimits, ModelOverlay, ModelPatch,
	ModelProvenance, ModelSpec, OperationBits, OperationKind, Pricing, ProvenanceKind,
	ProvenanceSource, ProviderDef, ProviderId, RouteDef, RouteId, ScopedAlias, ThinkingPolicyId,
	ThinkingRouting, WireModelId, WirePolicyId, classify,
};

/// Provider-declared facts for one remotely discovered wire model.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DiscoveredModel {
	/// Provider domain that returned the row.
	pub provider:              ProviderId,
	/// Route whose discovery endpoint returned the row.
	pub route:                 RouteId,
	/// Opaque identifier sent back to that route's codec.
	pub wire_model:            WireModelId,
	/// Provider-declared alternate wire identifiers for the same deployment.
	#[serde(default)]
	pub aliases:               Box<[WireModelId]>,
	/// Provider-declared display name, if present.
	pub display_name:          Option<Str>,
	/// Provider-declared family, if present.
	pub declared_family:       Option<FamilyId>,
	/// Positive operation evidence reported by the provider.
	pub declared_operations:   OperationBits,
	/// Detailed provider-declared capability evidence, if present.
	pub declared_capabilities: Option<ModelCapabilities>,
	/// Provider-declared limits, if present.
	pub declared_limits:       Option<ModelLimits>,
	/// Provider-declared standard or extended context serving mode.
	pub extended_context_mode: Option<ExtendedContextMode>,
	/// Provider-declared availability, if present.
	pub availability:          Option<ModelAvailability>,
	/// Stable discovery source name or content address.
	pub source:                Str,
	/// Observation time in Unix milliseconds, if available.
	pub observed_at_ms:        Option<u64>,
	/// Provider update time in Unix milliseconds, if available.
	pub updated_at_ms:         Option<u64>,
	/// Provider deprecation declaration, if present.
	pub deprecated:            Option<bool>,
}

/// Catalog policy defaults required to materialize a newly discovered model.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DiscoveryDefaults {
	/// Default wire-lowering policy for the discovery route.
	pub wire_policy:          WirePolicyId,
	/// Interned lowering policy whose context policy selects extended mode.
	pub extended_wire_policy: Option<WirePolicyId>,
	/// Default context strategy for the discovery route.
	pub context:              ContextStrategy,
	/// Optional default reasoning policy.
	pub thinking:             Option<ThinkingPolicyId>,
	/// Default price schedule; discovery never fabricates prices.
	pub pricing:              Pricing,
}

/// Runtime discovery normalization failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DiscoveryError {
	/// Extended context evidence lacked a pre-interned extended lowering policy.
	MissingExtendedContextPolicy {
		/// Provider whose discovery row declared extended context.
		provider: ProviderId,
		/// Opaque wire model whose extended lowering policy was missing.
		model:    WireModelId,
	},
	/// A discovered row was not emitted by the projector's bound provider and
	/// route.
	RowScopeMismatch {
		/// Expected provider.
		expected_provider: ProviderId,
		/// Expected route.
		expected_route:    RouteId,
		/// Provider carried by the row.
		actual_provider:   ProviderId,
		/// Route carried by the row.
		actual_route:      RouteId,
	},
	/// The route is not configured for discovery.
	RouteDiscoveryMissing(RouteId),
	/// The supplied discovery specification is not the one bound to the route.
	RouteDiscoveryMismatch {
		/// Route being projected.
		route:    RouteId,
		/// Discovery specification bound to the route.
		expected: DiscoverySpecId,
		/// Supplied discovery specification.
		actual:   DiscoverySpecId,
	},
	/// The route belongs to a different provider.
	RouteProviderMismatch {
		/// Route being projected.
		route:    RouteId,
		/// Provider expected by the projector.
		expected: ProviderId,
		/// Provider declared by the route.
		actual:   ProviderId,
	},
	/// The supplied provider defaults use a different wire policy.
	ProviderPolicyMismatch {
		/// Provider being projected.
		provider: ProviderId,
		/// Provider-default wire policy.
		expected: WirePolicyId,
		/// Supplied default wire policy.
		actual:   WirePolicyId,
	},
	/// A single-page discovery endpoint returned a continuation cursor.
	UnexpectedContinuation(DiscoverySpecId),
	/// A page-number continuation was not canonical decimal text.
	InvalidPageNumber {
		/// Discovery specification that rejected the value.
		spec:  DiscoverySpecId,
		/// Rejected continuation value.
		value: Str,
	},
	/// Two discovered models declared the same alias for different targets.
	AliasConflict {
		/// Conflicting alias.
		alias:  Str,
		/// First canonical target.
		first:  ModelKey,
		/// Second canonical target.
		second: ModelKey,
	},
}

impl fmt::Display for DiscoveryError {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(formatter, "discovery normalization failed: {self:?}")
	}
}

impl Error for DiscoveryError {}

/// Typed continuation for the next route-bound discovery request.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum DiscoveryContinuation {
	/// The remote listing is complete.
	Complete,
	/// Send an opaque cursor in the declared query parameter.
	Cursor {
		/// Query parameter declared by the route's discovery specification.
		query_parameter: Str,
		/// Opaque provider cursor, preserved byte-for-byte.
		value:           Str,
	},
	/// Send a canonical numeric page in the declared query parameter.
	PageNumber {
		/// Query parameter declared by the route's discovery specification.
		query_parameter: Str,
		/// Next page number.
		page:            u32,
	},
}

/// Conservative route-bound result for one remote discovery page.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DiscoveryPage {
	/// Exact discovery specification used to interpret the response.
	pub spec:          DiscoverySpecId,
	/// Normalized models in deterministic provider/model order.
	pub models:        Box<[ModelSpec]>,
	/// Exact provider-scoped aliases in deterministic alias order.
	pub aliases:       Box<[ScopedAlias]>,
	/// Typed next-page request or completion marker.
	pub continuation:  DiscoveryContinuation,
	/// Whether successful absence proves model unavailability.
	pub authoritative: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RouteDiscoveryConfig {
	provider:   ProviderId,
	route:      RouteId,
	spec:       DiscoverySpec,
	normalizer: DiscoveryNormalizer,
}

/// Clone-cheap projector bound to one exact provider route and discovery
/// specification.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RouteDiscoveryProjector {
	config: Arc<RouteDiscoveryConfig>,
}

impl RouteDiscoveryProjector {
	/// Validates and binds provider defaults to an exact route discovery
	/// specification.
	pub fn new(
		provider: &ProviderDef,
		route: &RouteDef,
		spec: &DiscoverySpec,
		defaults: DiscoveryDefaults,
	) -> Result<Self, DiscoveryError> {
		if route.provider != provider.id {
			return Err(DiscoveryError::RouteProviderMismatch {
				route:    route.id.clone(),
				expected: provider.id.clone(),
				actual:   route.provider.clone(),
			});
		}
		let expected = route
			.discovery
			.clone()
			.ok_or_else(|| DiscoveryError::RouteDiscoveryMissing(route.id.clone()))?;
		if expected != spec.id {
			return Err(DiscoveryError::RouteDiscoveryMismatch {
				route: route.id.clone(),
				expected,
				actual: spec.id.clone(),
			});
		}
		if defaults.wire_policy != provider.wire_policy {
			return Err(DiscoveryError::ProviderPolicyMismatch {
				provider: provider.id.clone(),
				expected: provider.wire_policy.clone(),
				actual:   defaults.wire_policy,
			});
		}
		Ok(Self {
			config: Arc::new(RouteDiscoveryConfig {
				provider:   provider.id.clone(),
				route:      route.id.clone(),
				spec:       spec.clone(),
				normalizer: DiscoveryNormalizer::new(defaults),
			}),
		})
	}

	/// Returns the exact provider bound to this projector.
	pub fn provider(&self) -> &ProviderId {
		&self.config.provider
	}

	/// Returns the exact route bound to this projector.
	pub fn route(&self) -> &RouteId {
		&self.config.route
	}

	/// Returns the exact discovery specification bound to this projector.
	pub fn spec(&self) -> &DiscoverySpec {
		&self.config.spec
	}

	/// Normalizes one page and interprets its continuation using route-bound
	/// configuration.
	pub fn project(
		&self,
		rows: &[DiscoveredModel],
		next_cursor: Option<Str>,
	) -> Result<DiscoveryPage, DiscoveryError> {
		for row in rows {
			if row.provider != self.config.provider || row.route != self.config.route {
				return Err(DiscoveryError::RowScopeMismatch {
					expected_provider: self.config.provider.clone(),
					expected_route:    self.config.route.clone(),
					actual_provider:   row.provider.clone(),
					actual_route:      row.route.clone(),
				});
			}
		}
		let normalized = self.config.normalizer.normalize_batch(rows)?;
		let mut aliases = BTreeMap::<Str, ScopedAlias>::new();
		let mut models = Vec::with_capacity(normalized.len());
		for record in normalized {
			for definition in record.aliases {
				if let Some(existing) = aliases.get(definition.alias.as_str()) {
					if existing.definition.target != definition.target {
						return Err(DiscoveryError::AliasConflict {
							alias:  definition.alias,
							first:  existing.definition.target.clone(),
							second: definition.target,
						});
					}
					continue;
				}
				aliases.insert(definition.alias.clone(), ScopedAlias {
					provider: record.provider.clone(),
					definition,
				});
			}
			models.push(record.model);
		}
		Ok(DiscoveryPage {
			spec:          self.config.spec.id.clone(),
			models:        models.into_boxed_slice(),
			aliases:       aliases.into_values().collect::<Vec<_>>().into_boxed_slice(),
			continuation:  continuation(&self.config.spec, next_cursor)?,
			authoritative: self.config.spec.authoritative,
		})
	}
}

/// One normalized discovered model plus auditable classifier evidence.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NormalizedDiscovery {
	/// Provider paired with the classifier-normalized logical model key.
	pub provider:       ProviderId,
	/// Discovery-layer provenance assigned to every overlaid field.
	pub source:         ProvenanceSource,
	/// Conservative model record suitable for a discovery overlay.
	pub model:          ModelSpec,
	/// Evidence returned by the public catalog classifier for every collapsed
	/// row.
	pub classification: Box<[ClassificationEvidence]>,
	/// Provider-declared aliases retained as canonical catalog alias records.
	pub aliases:        Box<[CatalogAlias]>,
}

impl NormalizedDiscovery {
	/// Converts this normalized row into a complete discovery overlay without
	/// dropping aliases.
	pub fn into_catalog_overlay(self) -> CatalogOverlay {
		let Self { provider, source, model, aliases, .. } = self;
		let scoped_aliases = aliases
			.into_vec()
			.into_iter()
			.map(|definition| ScopedAlias { provider: provider.clone(), definition })
			.collect::<Vec<_>>()
			.into_boxed_slice();
		let model_overlay = ModelOverlay {
			selector: ExactSelector { provider, model: model.key.clone() },
			added:    Some(model.clone()),
			patch:    ModelPatch {
				family: Some(model.family),
				display_name: Some(model.display_name),
				wire_ids: Some(model.wire_ids),
				routes: Some(model.routes),
				capabilities: Some(model.capabilities),
				limits: Some(model.limits),
				thinking: Some(model.thinking),
				thinking_routing: Some(model.thinking_routing),
				wire_policy: Some(model.wire_policy),
				context: Some(model.context),
				pricing: Some(model.pricing),
				availability: Some(model.availability),
				context_promotion_target: Some(model.context_promotion_target),
				remote_compaction: Some(model.remote_compaction),
				premium_multiplier_millionths: Some(model.premium_multiplier_millionths),
				updated_at_ms: Some(model.provenance.updated_at_ms),
				blocked_until_ms: Some(model.provenance.blocked_until_ms),
				deprecated: Some(model.provenance.deprecated),
			},
		};
		CatalogOverlay {
			source,
			models: Box::new([model_overlay]),
			routes: Box::new([]),
			aliases: scoped_aliases,
		}
	}
}

/// Stateless conservative runtime-discovery normalizer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiscoveryNormalizer {
	defaults: Arc<DiscoveryDefaults>,
}

impl DiscoveryNormalizer {
	/// Creates a normalizer using route-owned catalog policy defaults.
	pub fn new(defaults: DiscoveryDefaults) -> Self {
		Self { defaults: Arc::new(defaults) }
	}

	/// Normalizes one row through the public classifier API.
	pub fn normalize(&self, row: &DiscoveredModel) -> Result<NormalizedDiscovery, DiscoveryError> {
		let classification = classify(ClassificationInput {
			phase:          ClassificationPhase::DiscoveryNormalizer,
			provider:       row.provider.as_str(),
			model:          row.wire_model.as_str(),
			observed_at_ms: row.observed_at_ms,
		});
		let family = row
			.declared_family
			.clone()
			.unwrap_or_else(|| classification.family.clone());
		let capabilities = declared_capabilities(row);
		let wire_policy = match row.extended_context_mode {
			Some(ExtendedContextMode::Extended) => {
				self.defaults.extended_wire_policy.clone().ok_or_else(|| {
					DiscoveryError::MissingExtendedContextPolicy {
						provider: row.provider.clone(),
						model:    row.wire_model.clone(),
					}
				})?
			},
			Some(ExtendedContextMode::Standard) | None => self.defaults.wire_policy.clone(),
		};
		let mut limits = row.declared_limits.unwrap_or_default();
		match row.extended_context_mode {
			Some(ExtendedContextMode::Standard) => limits.context_window = Some(200_000),
			Some(ExtendedContextMode::Extended) => limits.context_window = Some(1_000_000),
			None => {},
		}
		let declared = ProvenanceSource {
			kind:           ProvenanceKind::Discovered,
			origin:         row.source.clone(),
			revision:       None,
			confidence:     EvidenceConfidence::Declared,
			observed_at_ms: row.observed_at_ms,
		};
		let classified = ProvenanceSource {
			kind:           ProvenanceKind::Discovered,
			origin:         classification.evidence.provenance.clone(),
			revision:       None,
			confidence:     EvidenceConfidence::Inferred,
			observed_at_ms: row.observed_at_ms,
		};
		let aliases = row
			.aliases
			.iter()
			.filter(|alias| *alias != &row.wire_model)
			.map(|alias| CatalogAlias {
				alias:      alias.as_str().into(),
				target:     ModelKey::new(classification.logical_model.clone()),
				rationale:  "provider discovery declared an alternate wire model identifier".into(),
				provenance: row.source.clone(),
			})
			.collect::<Vec<_>>()
			.into_boxed_slice();
		Ok(NormalizedDiscovery {
			provider: row.provider.clone(),
			source: declared.clone(),
			model: ModelSpec {
				key: ModelKey::new(classification.logical_model.clone()),
				family,
				display_name: row
					.display_name
					.clone()
					.unwrap_or_else(|| classification.logical_model.clone()),
				wire_ids: Box::new([(row.route.clone(), row.wire_model.clone())]),
				routes: Box::new([row.route.clone()]),
				capabilities,
				limits,
				thinking: self.defaults.thinking.clone(),
				thinking_routing: ThinkingRouting::default(),
				wire_policy,
				context: self.defaults.context,
				pricing: self.defaults.pricing.clone(),
				availability: row.availability.unwrap_or(ModelAvailability::Unspecified),
				provenance: ModelProvenance {
					sources:          Box::new([declared, classified]),
					updated_at_ms:    row.updated_at_ms,
					blocked_until_ms: None,
					deprecated:       row.deprecated.unwrap_or(false),
				},
				context_promotion_target: None,
				remote_compaction: None,
				premium_multiplier_millionths: None,
			},
			classification: Box::new([classification.evidence]),
			aliases,
		})
	}

	/// Normalizes and conservatively collapses rows sharing one provider/logical
	/// model.
	pub fn normalize_batch(
		&self,
		rows: &[DiscoveredModel],
	) -> Result<Vec<NormalizedDiscovery>, DiscoveryError> {
		let mut normalized = rows
			.iter()
			.map(|row| self.normalize(row))
			.collect::<Result<Vec<_>, _>>()?;
		normalized.sort_by(|left, right| {
			left
				.provider
				.cmp(&right.provider)
				.then_with(|| left.model.key.cmp(&right.model.key))
				.then_with(|| left.model.routes[0].cmp(&right.model.routes[0]))
				.then_with(|| left.model.wire_ids[0].1.cmp(&right.model.wire_ids[0].1))
		});
		let mut grouped = BTreeMap::<(ProviderId, ModelKey), NormalizedDiscovery>::new();
		for item in normalized {
			let key = (item.provider.clone(), item.model.key.clone());
			if let Some(existing) = grouped.get_mut(&key) {
				merge_discovery(existing, item);
			} else {
				grouped.insert(key, item);
			}
		}
		Ok(grouped.into_values().collect())
	}
}

fn continuation(
	spec: &DiscoverySpec,
	next_cursor: Option<Str>,
) -> Result<DiscoveryContinuation, DiscoveryError> {
	let Some(value) = next_cursor else {
		return Ok(DiscoveryContinuation::Complete);
	};
	match &spec.pagination {
		DiscoveryPagination::SinglePage => {
			Err(DiscoveryError::UnexpectedContinuation(spec.id.clone()))
		},
		DiscoveryPagination::Cursor { query_parameter } => {
			Ok(DiscoveryContinuation::Cursor { query_parameter: query_parameter.clone(), value })
		},
		DiscoveryPagination::PageNumber { query_parameter, first_page } => {
			let page = value
				.parse::<u32>()
				.ok()
				.filter(|page| page.to_string() == value.as_str() && page >= first_page);
			page
				.map(|page| DiscoveryContinuation::PageNumber {
					query_parameter: query_parameter.clone(),
					page,
				})
				.ok_or(DiscoveryError::InvalidPageNumber { spec: spec.id.clone(), value })
		},
	}
}

/// Returns a model-capability record containing no positive evidence.
pub fn unknown_capabilities() -> ModelCapabilities {
	ModelCapabilities {
		operations:    OperationBits::empty(),
		chat:          None,
		embeddings:    None,
		image:         None,
		video:         None,
		speech:        None,
		transcription: None,
		realtime:      None,
		search:        None,
		tokenization:  None,
	}
}

fn declared_capabilities(row: &DiscoveredModel) -> ModelCapabilities {
	let mut capabilities = row
		.declared_capabilities
		.clone()
		.unwrap_or_else(unknown_capabilities);
	capabilities.operations.insert(row.declared_operations);
	if capabilities.operations.contains_kind(OperationKind::Chat) && capabilities.chat.is_none() {
		capabilities.chat = Some(unknown_chat_capabilities());
	}
	capabilities
}

fn unknown_chat_capabilities() -> ChatCapabilities {
	ChatCapabilities {
		roles:             Availability::Unknown,
		mid_session_roles: Availability::Unknown,
		tools:             Availability::Unknown,
		structured_output: Availability::Unknown,
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
}

fn merge_discovery(existing: &mut NormalizedDiscovery, incoming: NormalizedDiscovery) {
	existing.model.wire_ids =
		merge_sorted_unique(&existing.model.wire_ids, &incoming.model.wire_ids);
	existing.model.routes = merge_sorted_unique(&existing.model.routes, &incoming.model.routes);
	existing
		.model
		.capabilities
		.operations
		.insert(incoming.model.capabilities.operations);
	merge_detail(&mut existing.model.capabilities.chat, incoming.model.capabilities.chat);
	merge_detail(
		&mut existing.model.capabilities.embeddings,
		incoming.model.capabilities.embeddings,
	);
	merge_detail(&mut existing.model.capabilities.image, incoming.model.capabilities.image);
	merge_detail(&mut existing.model.capabilities.video, incoming.model.capabilities.video);
	merge_detail(&mut existing.model.capabilities.speech, incoming.model.capabilities.speech);
	merge_detail(
		&mut existing.model.capabilities.transcription,
		incoming.model.capabilities.transcription,
	);
	merge_detail(&mut existing.model.capabilities.realtime, incoming.model.capabilities.realtime);
	merge_detail(&mut existing.model.capabilities.search, incoming.model.capabilities.search);
	merge_detail(
		&mut existing.model.capabilities.tokenization,
		incoming.model.capabilities.tokenization,
	);
	existing.model.limits.context_window =
		conservative_min(existing.model.limits.context_window, incoming.model.limits.context_window);
	existing.model.limits.maximum_input_tokens = conservative_min(
		existing.model.limits.maximum_input_tokens,
		incoming.model.limits.maximum_input_tokens,
	);
	existing.model.limits.maximum_output_tokens = conservative_min(
		existing.model.limits.maximum_output_tokens,
		incoming.model.limits.maximum_output_tokens,
	);
	existing.model.limits.maximum_batch =
		conservative_min(existing.model.limits.maximum_batch, incoming.model.limits.maximum_batch);
	if existing.model.family != incoming.model.family {
		existing.model.family = FamilyId::new("unknown");
	}
	let mut sources = existing.model.provenance.sources.to_vec();
	for source in incoming.model.provenance.sources {
		if !sources.contains(&source) {
			sources.push(source);
		}
	}
	existing.model.provenance.sources = sources.into_boxed_slice();
	existing.model.provenance.updated_at_ms =
		match (existing.model.provenance.updated_at_ms, incoming.model.provenance.updated_at_ms) {
			(Some(left), Some(right)) => Some(left.max(right)),
			(left, right) => left.or(right),
		};
	existing.model.provenance.deprecated |= incoming.model.provenance.deprecated;
	let mut evidence = existing.classification.to_vec();
	evidence.extend(incoming.classification);
	existing.classification = evidence.into_boxed_slice();
	let mut aliases = existing.aliases.to_vec();
	aliases.extend(incoming.aliases);
	aliases.sort_by(|left, right| {
		left
			.alias
			.cmp(&right.alias)
			.then_with(|| left.target.cmp(&right.target))
	});
	aliases.dedup();
	existing.aliases = aliases.into_boxed_slice();
}

fn merge_detail<T: Eq>(existing: &mut Option<T>, incoming: Option<T>) {
	if existing.as_ref() != incoming.as_ref() {
		*existing = None;
	}
}

fn merge_sorted_unique<T: Clone + Ord>(left: &[T], right: &[T]) -> Box<[T]> {
	let mut values = Vec::with_capacity(left.len() + right.len());
	values.extend_from_slice(left);
	values.extend_from_slice(right);
	values.sort();
	values.dedup();
	values.into_boxed_slice()
}

fn conservative_min<T: Ord>(left: Option<T>, right: Option<T>) -> Option<T> {
	match (left, right) {
		(Some(left), Some(right)) => Some(left.min(right)),
		(None, _) | (_, None) => None,
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{
		AuthSpecId, CodecId, CodecProfile, CodexTransportPreference, DiscoveryKind, EndpointSpec,
		HeaderProfileId, ManagementCapabilities, RedirectTrust, RegistryMapping, RoleBits,
		RouteRestrictions, StructuredOutputBits, TransportKind, TrustDomain,
	};

	fn defaults() -> DiscoveryDefaults {
		DiscoveryDefaults {
			wire_policy:          WirePolicyId::from("wire"),
			extended_wire_policy: None,
			context:              ContextStrategy::Replay,
			thinking:             None,
			pricing:              Pricing::default(),
		}
	}

	fn row(model: &str) -> DiscoveredModel {
		DiscoveredModel {
			provider:              ProviderId::from("provider"),
			route:                 RouteId::from("route"),
			wire_model:            WireModelId::from(model),
			display_name:          None,
			aliases:               Box::new([]),
			declared_family:       None,
			declared_operations:   OperationBits::empty(),
			declared_capabilities: None,
			declared_limits:       None,
			extended_context_mode: None,
			availability:          None,
			source:                "provider-list".into(),
			observed_at_ms:        Some(7),
			updated_at_ms:         None,
			deprecated:            None,
		}
	}

	fn projector(pagination: DiscoveryPagination) -> RouteDiscoveryProjector {
		let provider = ProviderDef {
			id:                 ProviderId::from("provider"),
			name:               "Provider".into(),
			auth:               Box::new([AuthSpecId::from("auth")]),
			management:         ManagementCapabilities {
				operations:        OperationBits::empty(),
				multiple_accounts: false,
				refresh:           false,
				principal_quota:   false,
			},
			routes:             Box::new([RouteId::from("route")]),
			wire_policy:        WirePolicyId::from("wire"),
			discovery_defaults: None,
			mapping:            RegistryMapping::Concrete,
		};
		let route = RouteDef {
			id:                 RouteId::from("route"),
			provider:           ProviderId::from("provider"),
			codec_profile:      CodecProfile::Standard,
			codec:              CodecId::from("codec"),
			transport:          TransportKind::Http,
			endpoint:           EndpointSpec {
				base_url: "https://provider.test".into(),
				region:   None,
			},
			auth:               AuthSpecId::from("auth"),
			headers:            HeaderProfileId::from("headers"),
			discovery:          Some(DiscoverySpecId::from("models")),
			capability_limits:  RouteRestrictions::default(),
			trust_domain:       TrustDomain {
				origin:          "https://provider.test".into(),
				redirects:       RedirectTrust::SameOrigin,
				allow_plaintext: false,
			},
			codex_transport:    CodexTransportPreference::HttpOnly,
			use_responses_lite: None,
			priority:           None,
		};
		let spec = DiscoverySpec {
			id: DiscoverySpecId::from("models"),
			kind: DiscoveryKind::Specialized,
			label: "models".into(),
			path: "/models".into(),
			pagination,
			authoritative: true,
		};
		RouteDiscoveryProjector::new(&provider, &route, &spec, defaults())
			.expect("route-bound projector configuration is valid")
	}

	#[test]
	fn route_projector_preserves_cursor_provenance_unknowns_and_alias_order() {
		let projector = projector(DiscoveryPagination::Cursor { query_parameter: "after".into() });
		let mut z = row("z-model");
		z.aliases = Box::new([WireModelId::from("z-alias")]);
		let mut a = row("a-model");
		a.aliases = Box::new([WireModelId::from("a-alias")]);
		let page = projector
			.project(&[z, a], Some(Str::from("opaque +/% cursor")))
			.expect("route-bound page projects");
		assert_eq!(
			page
				.models
				.iter()
				.map(|model| model.key.as_str())
				.collect::<Vec<_>>(),
			["a-model", "z-model"]
		);
		assert_eq!(
			page
				.aliases
				.iter()
				.map(|alias| alias.definition.alias.as_str())
				.collect::<Vec<_>>(),
			["a-alias", "z-alias"]
		);
		assert_eq!(page.continuation, DiscoveryContinuation::Cursor {
			query_parameter: "after".into(),
			value:           "opaque +/% cursor".into(),
		});
		assert!(page.authoritative);
		assert!(page.models.iter().all(|model| {
			model.capabilities.operations == OperationBits::empty()
				&& model.capabilities.chat.is_none()
				&& model.provenance.sources[0].kind == ProvenanceKind::Discovered
		}));
	}

	#[test]
	fn route_projector_rejects_scope_alias_and_pagination_ambiguity() {
		let cursor = projector(DiscoveryPagination::Cursor { query_parameter: "after".into() });
		let mut wrong = row("wrong");
		wrong.route = RouteId::from("other-route");
		assert!(matches!(
			cursor.project(&[wrong], None),
			Err(DiscoveryError::RowScopeMismatch { actual_route, .. }) if actual_route == "other-route"
		));

		let mut first = row("first");
		first.aliases = Box::new([WireModelId::from("shared")]);
		let mut second = row("second");
		second.aliases = Box::new([WireModelId::from("shared")]);
		assert!(matches!(
			cursor.project(&[first, second], None),
			Err(DiscoveryError::AliasConflict { alias, .. }) if alias == "shared"
		));

		let single = projector(DiscoveryPagination::SinglePage);
		assert_eq!(
			single.project(&[], Some("unexpected".into())),
			Err(DiscoveryError::UnexpectedContinuation(DiscoverySpecId::from("models")))
		);

		let numbered = projector(DiscoveryPagination::PageNumber {
			query_parameter: "page".into(),
			first_page:      1,
		});
		assert!(matches!(
			numbered.project(&[], Some("01".into())),
			Err(DiscoveryError::InvalidPageNumber { value, .. }) if value == "01"
		));
		assert_eq!(
			numbered
				.project(&[], Some("2".into()))
				.expect("canonical page number")
				.continuation,
			DiscoveryContinuation::PageNumber { query_parameter: "page".into(), page: 2 }
		);
	}

	#[test]
	fn absent_discovery_evidence_never_becomes_native() {
		let normalized = DiscoveryNormalizer::new(defaults())
			.normalize(&row("adversarial-native-tools-gpt"))
			.expect("ordinary discovery normalizes");
		assert_eq!(normalized.model.capabilities.operations, OperationBits::empty());
		assert!(normalized.model.capabilities.chat.is_none());
		assert_eq!(normalized.model.limits, ModelLimits::default());
		assert_eq!(normalized.model.provenance.sources[0].confidence, EvidenceConfidence::Declared);
	}

	#[test]
	fn extended_context_evidence_requires_and_selects_an_explicit_policy() {
		let mut discovered = row("extended");
		discovered.extended_context_mode = Some(ExtendedContextMode::Extended);
		assert_eq!(
			DiscoveryNormalizer::new(defaults()).normalize(&discovered),
			Err(DiscoveryError::MissingExtendedContextPolicy {
				provider: ProviderId::from("provider"),
				model:    WireModelId::from("extended"),
			})
		);
		let mut configured = defaults();
		configured.extended_wire_policy = Some(WirePolicyId::from("extended-wire"));
		let normalized = DiscoveryNormalizer::new(configured)
			.normalize(&discovered)
			.expect("extended mode has an explicit lowering policy");
		assert_eq!(normalized.model.limits.context_window, Some(1_000_000));
		assert_eq!(normalized.model.wire_policy, WirePolicyId::from("extended-wire"));
	}

	#[test]
	fn positive_operation_evidence_keeps_unspecified_chat_axes_unknown() {
		let mut discovered = row("chat-model");
		discovered
			.declared_operations
			.insert_kind(OperationKind::Chat);
		let normalized = DiscoveryNormalizer::new(defaults())
			.normalize(&discovered)
			.expect("declared chat normalizes");
		let chat = normalized
			.model
			.capabilities
			.chat
			.expect("positive chat operation gets an evidence shell");
		assert!(matches!(chat.roles, Availability::Unknown));
		assert!(matches!(chat.tools, Availability::Unknown));
		assert!(matches!(chat.structured_output, Availability::Unknown));
	}

	#[test]
	fn provider_declared_capabilities_are_preserved_with_provenance() {
		let mut discovered = row("declared");
		discovered.declared_family = Some(FamilyId::from("provider-family"));
		discovered
			.declared_operations
			.insert_kind(OperationKind::Chat);
		let mut capabilities = unknown_capabilities();
		capabilities.chat = Some(ChatCapabilities {
			roles:             Availability::Native(RoleBits::SYSTEM),
			mid_session_roles: Availability::Unknown,
			tools:             Availability::Unknown,
			structured_output: Availability::Native(StructuredOutputBits::JSON_SCHEMA),
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
		});
		discovered.declared_capabilities = Some(capabilities);
		let normalized = DiscoveryNormalizer::new(defaults())
			.normalize(&discovered)
			.expect("declared capabilities normalize");
		assert_eq!(normalized.model.family, "provider-family");
		assert!(matches!(
			normalized.model.capabilities.chat.as_ref().map(|chat| &chat.structured_output),
			Some(Availability::Native(bits)) if bits.contains(StructuredOutputBits::JSON_SCHEMA)
		));
		assert_eq!(normalized.model.provenance.sources[0].origin, "provider-list");
	}

	#[test]
	fn batch_collapse_is_deterministic_and_conflicts_become_unknown() {
		let mut high = row("novel-high");
		high.route = RouteId::from("z-route");
		high.declared_operations.insert_kind(OperationKind::Chat);
		let mut high_capabilities = unknown_capabilities();
		high_capabilities.chat = Some(ChatCapabilities {
			roles:             Availability::Unsupported,
			mid_session_roles: Availability::Unknown,
			tools:             Availability::Unknown,
			structured_output: Availability::Unknown,
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
		});
		high.declared_capabilities = Some(high_capabilities);
		let mut low = row("novel-low");
		low.route = RouteId::from("a-route");
		low.declared_operations.insert_kind(OperationKind::Chat);
		let normalized = DiscoveryNormalizer::new(defaults())
			.normalize_batch(&[high, low])
			.expect("effort siblings normalize");
		assert_eq!(normalized.len(), 1);
		assert_eq!(normalized[0].model.key, "novel");
		assert_eq!(
			normalized[0]
				.model
				.routes
				.iter()
				.map(RouteId::as_str)
				.collect::<Vec<_>>(),
			["a-route", "z-route"]
		);
		assert!(normalized[0].model.capabilities.chat.is_none());
	}
}
