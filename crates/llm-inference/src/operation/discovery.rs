//! Runtime model discovery service and conservative catalog normalization.

use std::{
	collections::{BTreeMap, BTreeSet},
	future::Future,
	num::NonZeroU32,
	sync::Arc,
	task::{Context, Poll},
};

use omp_core::Str;
use tower::Service;

use crate::{
	answer::{Answer, AnswerBody, ModelDiscoveryPage},
	call::{DiscoveryRequest, OperationCall},
	catalog::{
		DiscoveredModel, DiscoveryNormalizer, ModelSpec, OperationKind, Pricing, ProviderId,
		RouteDef, RouteId, WireModelId, snapshot::Catalog,
	},
	error::{Error, ErrorDetail, ErrorKind, ErrorPhase, RetryAction},
	operation::{OperationRequest, OperationResponse},
	receipt::{ExecutionReceipt, ReasonId},
};

/// Provider wire rows and continuation state returned by a discovery codec.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RawDiscoveryPage {
	/// Typed provider-declared rows; no inferred defaults have been applied.
	pub models:      Vec<DiscoveredModel>,
	/// Opaque continuation cursor.
	pub next_cursor: Option<Str>,
}

/// Route-scoped projector applying canonical discovery normalization during
/// response recovery.
#[derive(Clone, Debug)]
pub struct CatalogDiscoveryProjector {
	normalizer: DiscoveryNormalizer,
	allowlist:  Option<Arc<BTreeMap<WireModelId, ModelSpec>>>,
	provider:   ProviderId,
	route:      RouteId,
}

impl CatalogDiscoveryProjector {
	/// Constructs a projector from exact route identity and compiler-owned
	/// normalization defaults.
	pub fn new(normalizer: DiscoveryNormalizer, provider: ProviderId, route: RouteId) -> Self {
		Self { normalizer, allowlist: None, provider, route }
	}

	/// Constructs a mixed bundled/unknown projector from authored provider
	/// discovery defaults.
	pub fn for_route(catalog: &Catalog, route: &RouteDef) -> Result<Self, Error> {
		let discovery = route
			.discovery
			.as_ref()
			.ok_or_else(|| capability_error("discovery.spec", "route_has_no_discovery_spec"))?;
		catalog
			.discovery_spec(discovery)
			.ok_or_else(|| capability_error("discovery.spec", "catalog_discovery_spec_missing"))?;
		let provider = catalog.provider(&route.provider).ok_or_else(|| {
			capability_error("discovery.provider", "catalog_discovery_provider_missing")
		})?;
		let defaults = catalog
			.discovery_defaults(&route.provider)
			.cloned()
			.ok_or_else(|| {
				capability_error("discovery.defaults", "provider_discovery_defaults_missing")
			})?;
		if defaults.thinking.is_some() || defaults.pricing != Pricing::default() {
			return Err(capability_error(
				"discovery.defaults",
				"discovery_defaults_must_not_inherit_thinking_or_pricing",
			));
		}
		if defaults.wire_policy != provider.wire_policy {
			return Err(capability_error(
				"discovery.defaults",
				"discovery_wire_policy_does_not_match_provider",
			));
		}
		let mut allowlist = BTreeMap::new();
		for model in catalog.models() {
			for (candidate, wire_model) in model.wire_ids.iter() {
				if candidate != &route.id {
					continue;
				}
				if allowlist
					.insert(wire_model.clone(), model.clone())
					.is_some()
				{
					return Err(capability_error(
						"discovery.allowlist",
						"duplicate_route_wire_model_identifier",
					));
				}
			}
		}
		Ok(Self {
			normalizer: DiscoveryNormalizer::new(defaults),
			allowlist:  Some(Arc::new(allowlist)),
			provider:   route.provider.clone(),
			route:      route.id.clone(),
		})
	}
}

impl crate::layer::recover::DiscoveryProjector for CatalogDiscoveryProjector {
	fn project(
		&self,
		request: &DiscoveryRequest,
		rows: Vec<DiscoveredModel>,
		next_cursor: Option<Str>,
	) -> Result<ModelDiscoveryPage, Error> {
		match &self.allowlist {
			None => normalize_page(
				&self.normalizer,
				&self.provider,
				&self.route,
				request,
				RawDiscoveryPage { models: rows, next_cursor },
			),
			Some(allowlist) => project_mixed_page(
				allowlist,
				&self.normalizer,
				&self.provider,
				&self.route,
				request,
				rows,
				next_cursor,
			),
		}
	}
}

fn project_mixed_page(
	allowlist: &BTreeMap<WireModelId, ModelSpec>,
	normalizer: &DiscoveryNormalizer,
	provider: &ProviderId,
	route: &RouteId,
	request: &DiscoveryRequest,
	rows: Vec<DiscoveredModel>,
	next_cursor: Option<Str>,
) -> Result<ModelDiscoveryPage, Error> {
	if rows.len() > request.page_size as usize {
		return Err(protocol_error("discovery_backend_exceeded_page_size"));
	}
	if next_cursor.as_ref().is_some_and(|cursor| cursor.is_empty()) {
		return Err(protocol_error("discovery_backend_returned_empty_cursor"));
	}
	let mut seen_wire: BTreeSet<WireModelId> = BTreeSet::new();
	let mut seen_models = BTreeSet::new();
	let mut models = Vec::new();
	for row in rows {
		if &row.provider != provider || &row.route != route {
			return Err(protocol_error("discovery_row_route_mismatch"));
		}
		if !seen_wire.insert(row.wire_model.clone()) {
			continue;
		}
		let model = match allowlist.get(&row.wire_model) {
			Some(model) => model.clone(),
			None => {
				normalizer
					.normalize(&row)
					.map_err(|_| protocol_error("discovery_normalization_failed"))?
					.model
			},
		};
		if !seen_models.insert(model.key.clone()) {
			continue;
		}
		if request
			.operation
			.is_some_and(|operation| !model.capabilities.operations.contains_kind(operation))
		{
			continue;
		}
		models.push(model);
	}
	Ok(ModelDiscoveryPage { models, next_cursor })
}

/// Concrete discovery service over a provider-specific typed backend.
#[derive(Clone, Debug)]
pub struct DiscoveryService<S> {
	inner:             S,
	normalizer:        DiscoveryNormalizer,
	maximum_page_size: NonZeroU32,
}

impl<S> DiscoveryService<S> {
	/// Constructs a discovery service with route-owned policy defaults.
	pub const fn new(
		inner: S,
		normalizer: DiscoveryNormalizer,
		maximum_page_size: NonZeroU32,
	) -> Self {
		Self { inner, normalizer, maximum_page_size }
	}

	/// Validates a discovery request without silently changing an explicit page
	/// size.
	pub fn prepare(&self, request: &DiscoveryRequest) -> Result<DiscoveryRequest, Error> {
		if request.page_size == 0 {
			return Err(request_error("discovery.page_size", "zero_discovery_page_size"));
		}
		if request.page_size > self.maximum_page_size.get() {
			return Err(capability_error(
				"discovery.page_size",
				"discovery_page_size_exceeds_route_limit",
			));
		}
		if request
			.cursor
			.as_ref()
			.is_some_and(|cursor| cursor.is_empty())
		{
			return Err(request_error("discovery.cursor", "empty_discovery_cursor"));
		}
		Ok(DiscoveryRequest {
			provider:  request.provider.clone(),
			route:     request.route.clone(),
			cursor:    request.cursor.clone(),
			page_size: request.page_size,
			operation: request.operation,
		})
	}
}

impl<S> Service<crate::call::Call> for DiscoveryService<S>
where
	S: Service<
			OperationRequest<DiscoveryRequest>,
			Response = OperationResponse<RawDiscoveryPage>,
			Error = Error,
		>,
	S::Future: Send + 'static,
{
	type Error = Error;
	type Response = Answer;

	type Future = impl Future<Output = Result<Answer, Error>> + Send;

	fn poll_ready(&mut self, context: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		self.inner.poll_ready(context)
	}

	fn call(&mut self, call: crate::call::Call) -> Self::Future {
		let prepared = match &call.operation {
			OperationCall::DiscoverModels(request) => self.prepare(request).map(Arc::new),
			_ => Err(wrong_operation(&call)),
		};
		let pending = prepared.as_ref().ok().map(|request| {
			self
				.inner
				.call(OperationRequest::from_call(&call, Arc::clone(request)))
		});
		let normalizer = self.normalizer.clone();
		async move {
			let request = prepared?;
			let Some(pending) = pending else {
				return Err(protocol_error("discovery_backend_not_called"));
			};
			let response = pending.await?;
			if response.meta.model.is_some() {
				return Err(protocol_error("discovery_response_must_not_select_model"));
			}
			if request
				.provider
				.as_ref()
				.is_some_and(|provider| provider != &response.meta.provider)
				|| request
					.route
					.as_ref()
					.is_some_and(|route| route != &response.meta.route)
			{
				return Err(protocol_error("discovery_response_selector_mismatch"));
			}
			let page = normalize_page(
				&normalizer,
				&response.meta.provider,
				&response.meta.route,
				&request,
				response.output,
			)?;
			Ok(OperationResponse { meta: response.meta, receipt: response.receipt, output: page }
				.into_answer(AnswerBody::Models))
		}
	}
}

/// Validates provider rows and applies the canonical conservative discovery
/// normalizer.
pub fn normalize_page(
	normalizer: &DiscoveryNormalizer,
	provider: &ProviderId,
	route: &RouteId,
	request: &DiscoveryRequest,
	page: RawDiscoveryPage,
) -> Result<ModelDiscoveryPage, Error> {
	if page.models.len() > request.page_size as usize {
		return Err(protocol_error("discovery_backend_exceeded_page_size"));
	}
	if page
		.next_cursor
		.as_ref()
		.is_some_and(|cursor| cursor.is_empty())
	{
		return Err(protocol_error("discovery_backend_returned_empty_cursor"));
	}
	for row in &page.models {
		if &row.provider != provider || &row.route != route {
			return Err(protocol_error("discovery_row_route_mismatch"));
		}
	}
	let mut models = normalizer
		.normalize_batch(&page.models)
		.map_err(|_| protocol_error("discovery_normalization_failed"))?
		.into_iter()
		.map(|normalized| normalized.model)
		.collect::<Vec<_>>();
	if let Some(operation) = request.operation {
		models.retain(|model| model.capabilities.operations.contains_kind(operation));
	}
	Ok(ModelDiscoveryPage { models, next_cursor: page.next_cursor })
}

fn wrong_operation(call: &crate::call::Call) -> Error {
	let mut error = Error::new(
		ErrorKind::InternalInvariant,
		ErrorPhase::Internal,
		RetryAction::Never,
		ExecutionReceipt::default(),
	);
	error.request_id = Some(call.id.clone());
	error.detail = Some(ErrorDetail::Capability {
		feature: Str::from(OperationKind::DiscoverModels.to_string()),
		reason:  ReasonId(Str::from("operation_service_mismatch")),
	});
	error
}

fn request_error(feature: &'static str, reason: &'static str) -> Error {
	Error::planning(
		ErrorKind::InvalidRequest,
		ErrorDetail::Capability { feature: Str::from(feature), reason: ReasonId(Str::from(reason)) },
		ExecutionReceipt::default(),
	)
}

fn capability_error(feature: &'static str, reason: &'static str) -> Error {
	Error::planning(
		ErrorKind::CapabilityMismatch,
		ErrorDetail::Capability { feature: Str::from(feature), reason: ReasonId(Str::from(reason)) },
		ExecutionReceipt::default(),
	)
}

fn protocol_error(reason: &'static str) -> Error {
	let mut error = Error::new(
		ErrorKind::ProviderContractMismatch,
		ErrorPhase::Discovery,
		RetryAction::Never,
		ExecutionReceipt::default(),
	);
	error.detail = Some(ErrorDetail::Protocol { reason: ReasonId(Str::from(reason)) });
	error
}

#[cfg(test)]
mod tests {
	use std::{collections::BTreeMap, num::NonZeroU32};

	use super::{
		CatalogDiscoveryProjector, DiscoveryService, RawDiscoveryPage, normalize_page,
		project_mixed_page,
	};
	use crate::{
		call::DiscoveryRequest,
		catalog::{
			ContextStrategy, DiscoveredModel, DiscoveryDefaults, DiscoveryNormalizer, OperationBits,
			OperationKind, Pricing, ProviderId, RouteId, WireModelId, WirePolicyId,
		},
		layer::recover::DiscoveryProjector,
	};

	fn discovered(provider: &ProviderId, route: &RouteId, wire_model: &str) -> DiscoveredModel {
		let mut operations = OperationBits::empty();
		operations.insert_kind(OperationKind::Embed);
		DiscoveredModel {
			provider:              provider.clone(),
			route:                 route.clone(),
			wire_model:            WireModelId::from(wire_model),
			aliases:               Box::new([]),
			display_name:          None,
			declared_family:       None,
			declared_operations:   operations,
			declared_capabilities: None,
			declared_limits:       None,
			extended_context_mode: None,
			availability:          None,
			source:                "oracle".into(),
			observed_at_ms:        Some(1),
			updated_at_ms:         None,
			deprecated:            None,
		}
	}

	fn projector(provider: &ProviderId, route: &RouteId) -> CatalogDiscoveryProjector {
		CatalogDiscoveryProjector::new(
			DiscoveryNormalizer::new(DiscoveryDefaults {
				wire_policy:          WirePolicyId::from("wire"),
				extended_wire_policy: None,
				context:              ContextStrategy::Replay,
				thinking:             None,
				pricing:              Pricing::default(),
			}),
			provider.clone(),
			route.clone(),
		)
	}
	#[test]
	fn discovery_request_page_bound_is_enforced_before_backend_execution() {
		let normalizer = DiscoveryNormalizer::new(DiscoveryDefaults {
			wire_policy: WirePolicyId::from("wire"),

			extended_wire_policy: None,
			context:              ContextStrategy::Replay,
			thinking:             None,
			pricing:              Pricing::default(),
		});
		let service = DiscoveryService::new((), normalizer, NonZeroU32::new(100).expect("non-zero"));
		assert!(
			service
				.prepare(&DiscoveryRequest {
					provider:  None,
					route:     None,
					cursor:    Some("next".into()),
					page_size: 1_000,
					operation: None,
				})
				.is_err()
		);
		let prepared = service
			.prepare(&DiscoveryRequest {
				provider:  None,
				route:     None,
				cursor:    Some("next".into()),
				page_size: 100,
				operation: None,
			})
			.expect("bounded request");
		assert_eq!(prepared.page_size, 100);
		assert!(
			service
				.prepare(&DiscoveryRequest {
					provider:  None,
					route:     None,
					cursor:    None,
					page_size: 0,
					operation: None,
				})
				.is_err()
		);
	}

	#[test]
	fn provider_rows_are_normalized_and_capability_filtered() {
		let normalizer = DiscoveryNormalizer::new(DiscoveryDefaults {
			wire_policy:          WirePolicyId::from("wire"),
			extended_wire_policy: None,
			context:              ContextStrategy::Replay,
			thinking:             None,
			pricing:              Pricing::default(),
		});
		let provider = ProviderId::from("provider");
		let route = RouteId::from("route");
		let mut operations = OperationBits::empty();
		operations.insert_kind(OperationKind::Embed);
		let page = normalize_page(
			&normalizer,
			&provider,
			&route,
			&DiscoveryRequest {
				provider:  Some(provider.clone()),
				route:     Some(route.clone()),
				cursor:    None,
				page_size: 10,
				operation: Some(OperationKind::Embed),
			},
			RawDiscoveryPage {
				models:      vec![DiscoveredModel {
					provider:              provider.clone(),
					route:                 route.clone(),
					wire_model:            WireModelId::from("embedding-model"),
					aliases:               Box::new([]),
					display_name:          None,
					declared_family:       None,
					declared_operations:   operations,
					declared_capabilities: None,
					declared_limits:       None,
					extended_context_mode: None,
					availability:          None,
					source:                "oracle".into(),
					observed_at_ms:        Some(1),
					updated_at_ms:         None,
					deprecated:            None,
				}],
				next_cursor: Some("next".into()),
			},
		)
		.expect("normalized page");
		assert_eq!(page.models.len(), 1);
		assert!(
			page.models[0]
				.capabilities
				.operations
				.contains_kind(OperationKind::Embed)
		);
		assert_eq!(page.next_cursor.as_deref(), Some("next"));
	}

	#[test]
	fn catalog_projector_deduplicates_and_preserves_pagination_deterministically() {
		let provider = ProviderId::from("provider");
		let route = RouteId::from("route");
		let row = discovered(&provider, &route, "embedding-model");
		let request = DiscoveryRequest {
			provider:  Some(provider.clone()),
			route:     Some(route.clone()),
			cursor:    Some("page-1".into()),
			page_size: 2,
			operation: Some(OperationKind::Embed),
		};
		let projector = projector(&provider, &route);
		let first = projector
			.project(&request, vec![row.clone(), row.clone()], Some("page-2".into()))
			.expect("projected page");
		let replay = projector
			.project(&request, vec![row.clone(), row], Some("page-2".into()))
			.expect("deterministic replay");
		assert_eq!(first.models.len(), 1);
		assert_eq!(replay.models, first.models);
		assert_eq!(replay.next_cursor, first.next_cursor);
	}

	#[test]
	fn catalog_projector_rejects_scope_size_and_empty_cursor() {
		let provider = ProviderId::from("provider");
		let route = RouteId::from("route");
		let projector = projector(&provider, &route);
		let request = DiscoveryRequest {
			provider:  Some(provider.clone()),
			route:     Some(route.clone()),
			cursor:    None,
			page_size: 1,
			operation: None,
		};
		let valid = discovered(&provider, &route, "one");
		let wrong = discovered(&ProviderId::from("other"), &route, "wrong");
		assert!(projector.project(&request, vec![wrong], None).is_err());
		assert!(
			projector
				.project(&request, vec![valid.clone(), discovered(&provider, &route, "two")], None)
				.is_err()
		);
		assert!(
			projector
				.project(&request, vec![valid], Some("".into()))
				.is_err()
		);
	}

	#[test]
	fn mixed_projection_preserves_known_models_and_conservatively_normalizes_unknown_rows() {
		let provider = ProviderId::from("provider");
		let route = RouteId::from("route");
		let normalizer = DiscoveryNormalizer::new(DiscoveryDefaults {
			wire_policy:          WirePolicyId::from("wire"),
			extended_wire_policy: None,
			context:              ContextStrategy::Replay,
			thinking:             None,
			pricing:              Pricing::default(),
		});
		let known_row = discovered(&provider, &route, "known");
		let known = normalizer
			.normalize(&known_row)
			.expect("known fixture")
			.model;
		let mut allowlist = BTreeMap::new();
		allowlist.insert(WireModelId::from("known"), known.clone());
		let request = DiscoveryRequest {
			provider:  Some(provider.clone()),
			route:     Some(route.clone()),
			cursor:    None,
			page_size: 4,
			operation: None,
		};
		let page = project_mixed_page(
			&allowlist,
			&normalizer,
			&provider,
			&route,
			&request,
			vec![known_row.clone(), discovered(&provider, &route, "unknown"), known_row],
			Some("next".into()),
		)
		.expect("mixed page");
		assert_eq!(page.models.len(), 2);
		assert_eq!(page.models[0], known);
		assert_eq!(page.models[1].thinking, None);
		assert_eq!(page.models[1].pricing, Pricing::default());
		assert_eq!(page.next_cursor.as_deref(), Some("next"));
	}
}
