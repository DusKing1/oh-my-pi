//! Capability-first, exact-selector routing and deterministic route ranking.

use std::{
	cmp::Ordering,
	collections::HashMap,
	sync::Arc,
	time::{Duration, Instant},
};

use omp_core::Str;

use crate::{
	call::{Call, OperationCall, Target},
	catalog::{ModelKey, OperationKind, PolicyModel, PriceUnit, ProviderId, RouteId, WireTarget},
	error::{Error, ErrorDetail, ErrorKind},
	plan::{
		CapabilityAvailability, CapabilityRequirement, ExecutionPlan, FallbackScope,
		NativeOptionRequirement, Planner, PlanningPolicy, ReplayRequirements, RuntimeRouteEvidence,
		negotiate, negotiate_native_option, plan_replay,
	},
	receipt::{ExecutionBudget, ExecutionReceipt, ReasonId},
	registry::Registry,
};

/// One exact model/route candidate produced by catalog resolution.
#[derive(Clone, Debug)]
pub struct RouteCandidate {
	/// Normalized model key used only for exact selection and receipts.
	pub model:        ModelKey,
	/// Owning provider domain.
	pub provider:     ProviderId,
	/// Router-facing model facts with no raw model or wire identifier.
	pub policy_model: Arc<PolicyModel>,
	/// Codec-facing target retained opaquely until encoding.
	pub wire_target:  WireTarget,
}

/// Exact primary selector and caller-authorized ordered model fallback chain.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RouteSelection {
	/// Target restriction for the primary normalized model.
	pub target:          Target,
	/// Ordered normalized models the caller explicitly permits after the primary
	/// fails planning.
	pub fallback_models: Arc<[ModelKey]>,
}

impl RouteSelection {
	/// Returns the exact primary model without exposing a wire identifier.
	pub fn primary_model(&self) -> Option<&ModelKey> {
		match &self.target {
			Target::Model(model) | Target::Provider { model, .. } | Target::Route { model, .. } => {
				Some(model)
			},
			Target::ProviderService(_) | Target::RouteService(_) => None,
		}
	}
}

/// Side-effect-free input to routing and capability negotiation.
#[derive(Clone, Debug)]
pub struct PlanRequest {
	/// Exact selector and explicit ordered fallbacks.
	pub selection:      RouteSelection,
	/// Closed operation kind.
	pub operation:      OperationKind,
	/// Exact clone-cheap operation payload used for setting negotiation.
	pub operation_call: Option<OperationCall>,
	/// All resolved catalog candidates; the router never creates candidates
	/// itself.
	pub candidates:     Arc<[RouteCandidate]>,
	/// Required and preferred capability axes.
	pub requirements:   Arc<[CapabilityRequirement]>,
	/// Typed codec-specific option, when present.
	pub native_option:  Option<NativeOptionRequirement>,
	/// Caller negotiation policy.
	pub policy:         PlanningPolicy,
	/// Aggregate replay and staging facts.
	pub replay:         ReplayRequirements,
	/// Cross-attempt execution budget.
	pub budget:         ExecutionBudget,
	/// Existing account/session route affinity, when any.
	pub affinity_route: Option<RouteId>,
}

/// Immutable capability-first router over one immutable registry.
#[derive(Clone)]
pub struct Router {
	registry: Registry,
	plan_ttl: Duration,
	runtime:  Arc<HashMap<RouteId, RuntimeRouteEvidence>>,
}

impl Router {
	/// Creates a router whose plans expire after the supplied short validity
	/// window.
	pub fn new(registry: Registry, plan_ttl: Duration) -> Self {
		Self { registry, plan_ttl, runtime: Arc::new(HashMap::new()) }
	}

	/// Replaces credential-free route observations used by subsequent
	/// side-effect-free plans.
	pub fn with_runtime_evidence(mut self, runtime: HashMap<RouteId, RuntimeRouteEvidence>) -> Self {
		self.runtime = Arc::new(runtime);
		self
	}

	/// Resolves catalog candidates and operation intent directly from a
	/// canonical call.
	pub fn plan_call(&self, call: &Call, now: Instant) -> Result<ExecutionPlan, Error> {
		match &call.target {
			Target::ProviderService(provider) => self.plan_management(call, provider, None, now),
			Target::RouteService(route) => {
				let definition = self
					.registry
					.catalog()
					.route(route)
					.ok_or_else(|| target_route_not_found(route))?;
				self.plan_management(call, &definition.provider, Some(route), now)
			},
			Target::Model(model) | Target::Provider { model, .. } | Target::Route { model, .. } => {
				let spec = self
					.registry
					.catalog()
					.model(model)
					.ok_or_else(|| target_not_found(&call.target))?;
				let policy_model = Arc::new(PolicyModel::from(spec));
				let mut candidates = Vec::new();
				for route_id in spec.routes.iter() {
					let Some(route) = self.registry.catalog().route(route_id) else {
						continue;
					};
					if !target_route_allowed(&call.target, route_id, &route.provider) {
						continue;
					}
					let Some((_, wire_model)) = spec
						.wire_ids
						.iter()
						.find(|(candidate, _)| candidate == route_id)
					else {
						continue;
					};
					candidates.push(RouteCandidate {
						model:        spec.key.clone(),
						provider:     route.provider.clone(),
						policy_model: policy_model.clone(),
						wire_target:  WireTarget {
							route:      route.id.clone(),
							codec:      route.codec.clone(),
							endpoint:   route.endpoint.clone(),
							wire_model: wire_model.clone(),
						},
					});
				}
				let request = PlanRequest {
					selection:      RouteSelection {
						target:          call.target.clone(),
						fallback_models: Arc::from([]),
					},
					operation:      call.operation.kind(),
					candidates:     candidates.into(),
					requirements:   extract_requirements(&call.operation).into(),
					operation_call: Some(call.operation.clone()),
					native_option:  None,
					policy:         operation_policy(&call.operation),
					replay:         operation_replay(&call.operation),
					budget:         call.budget.clone(),
					affinity_route: None,
				};
				self.plan(&request, &self.runtime, now)
			},
		}
	}

	fn plan_management(
		&self,
		call: &Call,
		provider: &ProviderId,
		pinned_route: Option<&RouteId>,
		now: Instant,
	) -> Result<ExecutionPlan, Error> {
		let definition = self
			.registry
			.catalog()
			.provider(provider)
			.ok_or_else(|| target_not_found(&call.target))?;
		let operation = call.operation.kind();
		let provider_support = definition.management.supports(operation);
		let requirements = extract_requirements(&call.operation);
		let policy = operation_policy(&call.operation);
		let evidence_route = pinned_route
			.or_else(|| definition.routes.first())
			.ok_or_else(|| target_not_found(&call.target))?;
		let wire_policy = self
			.registry
			.catalog()
			.wire_policy(&definition.wire_policy)
			.cloned()
			.ok_or_else(|| {
				route_contract_error(evidence_route, "catalog-provider-wire-policy-missing")
			})?;
		let mut eligible = Vec::new();
		let mut last_error = None;
		for route_id in definition.routes.iter() {
			let route = self
				.registry
				.catalog()
				.route(route_id)
				.ok_or_else(|| target_route_not_found(route_id))?;
			if !model_less_route_is_candidate(
				route_id,
				pinned_route,
				provider_support,
				route.capability_limits.operations,
				operation,
			) {
				continue;
			}
			if let Err(error) = self.registry.route_service(route_id, operation) {
				last_error = Some(prefer_error(last_error, error));
				continue;
			}
			let runtime =
				self
					.runtime
					.get(route_id)
					.cloned()
					.unwrap_or_else(|| RuntimeRouteEvidence {
						route:            route_id.clone(),
						generation:       self.registry.generation(),
						health:           crate::plan::RouteHealth::Unknown,
						quota_millionths: 0,
						latency:          Duration::MAX,
						affinity:         false,
						operation:        CapabilityAvailability::Native,
						capabilities:     Arc::from([]),
					});
			if runtime.generation != self.registry.generation() {
				last_error = Some(route_contract_error(route_id, "runtime-evidence-generation-stale"));
				continue;
			}
			if runtime.health == crate::plan::RouteHealth::Unavailable {
				last_error = Some(route_contract_error(route_id, "runtime-route-unavailable"));
				continue;
			}
			match runtime.operation {
				CapabilityAvailability::Unsupported => {
					last_error = Some(capability_error(
						ErrorKind::CapabilityMismatch,
						operation,
						"runtime-operation-unsupported",
					));
					continue;
				},
				CapabilityAvailability::Unknown => {
					last_error = Some(capability_error(
						ErrorKind::CapabilityUnknown,
						operation,
						"runtime-operation-unknown",
					));
					continue;
				},
				CapabilityAvailability::Native | CapabilityAvailability::Emulated(_) => {},
			}
			let (decisions, _) = match negotiate(&requirements, &runtime.capabilities, policy) {
				Ok(value) => value,
				Err(error) => {
					last_error = Some(prefer_error(last_error, error));
					continue;
				},
			};
			eligible.push((route, runtime, decisions));
		}
		eligible.sort_by(|left, right| {
			right
				.1
				.affinity
				.cmp(&left.1.affinity)
				.then_with(|| health_rank(right.1.health).cmp(&health_rank(left.1.health)))
				.then_with(|| right.1.quota_millionths.cmp(&left.1.quota_millionths))
				.then_with(|| left.1.latency.cmp(&right.1.latency))
				.then_with(|| {
					right
						.0
						.priority
						.unwrap_or_default()
						.cmp(&left.0.priority.unwrap_or_default())
				})
				.then_with(|| left.0.id.cmp(&right.0.id))
		});
		if eligible.is_empty() {
			if let Some(error) = last_error {
				return Err(error);
			}
			return Err(capability_error(
				ErrorKind::CapabilityMismatch,
				operation,
				"model-less-operation-not-advertised",
			));
		}
		let (route, runtime, decisions) = eligible.remove(0);
		let fallbacks = eligible
			.into_iter()
			.map(|(candidate, runtime, decisions)| crate::plan::PlannedFallback {
				model:              None,
				provider:           provider.clone(),
				route:              candidate.id.clone(),
				codec:              candidate.codec.clone(),
				wire_policy:        Arc::new(wire_policy.clone()),
				thinking_policy:    None,
				thinking_selection: None,
				decisions:          decisions.into(),
				policy_model:       None,
				wire_target:        None,
				runtime_evidence:   runtime,
			})
			.collect::<Vec<_>>();
		Ok(ExecutionPlan {
			planned_at: std::time::SystemTime::now(),
			catalog_revision: self.registry.catalog_revision().clone(),
			registry_generation: self.registry.generation(),
			expires_at: now.checked_add(self.plan_ttl).unwrap_or(now),
			operation,
			model: None,
			provider: provider.clone(),
			route: route.id.clone(),
			codec: route.codec.clone(),
			policy_model: None,
			wire_policy: Arc::new(wire_policy),
			fallbacks: fallbacks.into(),
			thinking_policy: None,
			thinking_selection: None,
			decisions: decisions.into(),
			fallback_scope: FallbackScope { primary: None, explicit: Arc::from([]) },
			replay: plan_replay(&operation_replay(&call.operation), &call.budget)?,
			budget: call.budget.clone(),
			runtime_evidence: runtime,
			wire_target: None,
		})
	}

	/// Borrows the registry used to construct and later validate plans.
	pub fn registry(&self) -> &Registry {
		&self.registry
	}

	/// Produces a credential-free plan without authentication or network access.
	pub fn plan(
		&self,
		request: &PlanRequest,
		runtime: &HashMap<RouteId, RuntimeRouteEvidence>,
		now: Instant,
	) -> Result<ExecutionPlan, Error> {
		let Some(primary) = request.selection.primary_model() else {
			return Err(Error::planning(
				ErrorKind::InvalidRequest,
				ErrorDetail::Target {
					selector: Str::from("model-less-target-requires-service-planning"),
				},
				ExecutionReceipt::default(),
			));
		};
		let mut authorized = Vec::with_capacity(request.selection.fallback_models.len() + 1);
		authorized.push(primary.clone());
		authorized.extend(request.selection.fallback_models.iter().cloned());
		let authorized_scope = FallbackScope {
			primary:  Some(primary.clone()),
			explicit: request.selection.fallback_models.clone(),
		};
		let mut last_error = None;
		let mut eligible = Vec::new();
		for model in &authorized {
			let mut model_candidates = Vec::new();
			for candidate in request
				.candidates
				.iter()
				.filter(|candidate| &candidate.model == model)
			{
				if !selector_accepts(
					&request.selection.target,
					primary,
					&authorized_scope,
					candidate,
					model == primary,
				) {
					continue;
				}
				match self.evaluate_candidate(request, candidate, runtime) {
					Ok(evaluated) => model_candidates.push(evaluated),
					Err(error) => last_error = Some(prefer_error(last_error, error)),
				}
			}
			model_candidates.sort_by(compare_evaluated);
			eligible.extend(model_candidates);
		}
		if eligible.is_empty() {
			return Err(last_error.unwrap_or_else(|| target_not_found(&request.selection.target)));
		}
		let selected = self.resolved_candidate(request, &eligible[0])?;
		let fallbacks = eligible[1..]
			.iter()
			.map(|candidate| self.resolved_candidate(request, candidate))
			.collect::<Result<Vec<_>, _>>()?;
		let replay = plan_replay(&request.replay, &request.budget)?;
		Ok(ExecutionPlan {
			planned_at: std::time::SystemTime::now(),
			catalog_revision: self.registry.catalog_revision().clone(),
			registry_generation: self.registry.generation(),
			expires_at: now.checked_add(self.plan_ttl).unwrap_or(now),
			operation: request.operation,
			model: selected.model,
			provider: selected.provider,
			route: selected.route,
			codec: selected.codec,
			policy_model: selected.policy_model,
			wire_policy: selected.wire_policy,
			thinking_policy: selected.thinking_policy,
			thinking_selection: selected.thinking_selection,
			decisions: selected.decisions,
			fallback_scope: authorized_scope,
			fallbacks: fallbacks.into(),
			replay,
			budget: request.budget.clone(),
			runtime_evidence: selected.runtime_evidence,
			wire_target: selected.wire_target,
		})
	}

	fn resolved_candidate(
		&self,
		request: &PlanRequest,
		candidate: &EvaluatedCandidate,
	) -> Result<crate::plan::PlannedFallback, Error> {
		let wire_policy = self
			.registry
			.catalog()
			.wire_policy(&candidate.candidate.policy_model.wire_policy)
			.cloned()
			.ok_or_else(|| {
				route_contract_error(
					&candidate.candidate.wire_target.route,
					"catalog-wire-policy-missing",
				)
			})?;
		let mut wire_target = candidate.candidate.wire_target.clone();
		let thinking_policy = candidate
			.candidate
			.policy_model
			.thinking
			.as_ref()
			.map(|id| {
				self
					.registry
					.catalog()
					.thinking_policy(id)
					.cloned()
					.ok_or_else(|| {
						route_contract_error(&wire_target.route, "catalog-thinking-policy-missing")
					})
			})
			.transpose()?;
		let requested_effort =
			chat_thinking_effort(request.operation_call.as_ref(), thinking_policy.as_ref())?;
		let thinking_selection = thinking_policy
			.as_ref()
			.map(|policy| {
				candidate
					.candidate
					.policy_model
					.thinking_routing
					.resolve(policy, requested_effort, &wire_target.wire_model)
					.map_err(|_| {
						capability_error(
							ErrorKind::CapabilityMismatch,
							request.operation,
							"thinking-selection-unsupported",
						)
					})
			})
			.transpose()?;
		if let Some(selection) = &thinking_selection {
			wire_target.wire_model = selection.wire_model.clone();
		}
		Ok(crate::plan::PlannedFallback {
			model: Some(candidate.candidate.model.clone()),
			provider: candidate.candidate.provider.clone(),
			route: candidate.candidate.wire_target.route.clone(),
			codec: candidate.candidate.wire_target.codec.clone(),
			wire_policy: Arc::new(wire_policy),
			thinking_policy: thinking_policy.map(Arc::new),
			thinking_selection,
			decisions: candidate.decisions.clone().into(),
			policy_model: Some(candidate.candidate.policy_model.clone()),
			wire_target: Some(wire_target),
			runtime_evidence: candidate.runtime.clone(),
		})
	}

	fn evaluate_candidate(
		&self,
		request: &PlanRequest,
		candidate: &RouteCandidate,
		runtime: &HashMap<RouteId, RuntimeRouteEvidence>,
	) -> Result<EvaluatedCandidate, Error> {
		let route = &candidate.wire_target.route;
		let definition = self
			.registry
			.catalog()
			.route(route)
			.ok_or_else(|| target_route_not_found(route))?;
		if definition.provider != candidate.provider
			|| definition.codec != candidate.wire_target.codec
		{
			return Err(route_contract_error(route, "candidate-route-definition-mismatch"));
		}
		if !candidate
			.policy_model
			.capabilities
			.operations
			.contains_kind(request.operation)
		{
			return Err(capability_error(
				ErrorKind::CapabilityMismatch,
				request.operation,
				"catalog-operation-unsupported",
			));
		}
		if definition
			.capability_limits
			.operations
			.is_some_and(|operations| !operations.contains_kind(request.operation))
		{
			return Err(capability_error(
				ErrorKind::CapabilityMismatch,
				request.operation,
				"route-operation-unsupported",
			));
		}
		if !self.registry.contains_service(route) {
			return Err(
				self
					.registry
					.route_service(route, request.operation)
					.expect_err("missing service returns typed evidence"),
			);
		}
		let runtime = runtime
			.get(route)
			.cloned()
			.unwrap_or_else(|| RuntimeRouteEvidence {
				route:            route.clone(),
				operation:        CapabilityAvailability::Native,
				generation:       self.registry.generation(),
				health:           crate::plan::RouteHealth::Unknown,
				quota_millionths: 0,
				latency:          Duration::MAX,
				affinity:         request.affinity_route.as_ref() == Some(route),
				capabilities:     Arc::from([]),
			});
		if runtime.generation != self.registry.generation() {
			return Err(route_contract_error(route, "runtime-evidence-generation-stale"));
		}
		match runtime.operation {
			CapabilityAvailability::Unsupported => {
				return Err(capability_error(
					ErrorKind::CapabilityMismatch,
					request.operation,
					"runtime-operation-unsupported",
				));
			},
			CapabilityAvailability::Unknown => {
				return Err(capability_error(
					ErrorKind::CapabilityUnknown,
					request.operation,
					"runtime-operation-unknown",
				));
			},
			CapabilityAvailability::Native | CapabilityAvailability::Emulated(_) => {},
		}
		if runtime.health == crate::plan::RouteHealth::Unavailable {
			return Err(route_contract_error(route, "runtime-route-unavailable"));
		}
		let evidence = request
			.requirements
			.iter()
			.map(|requirement| {
				runtime
					.capabilities
					.iter()
					.find(|item| item.feature == requirement.feature)
					.cloned()
					.unwrap_or_else(|| catalog_capability_evidence(&candidate.policy_model, requirement))
			})
			.collect::<Vec<_>>();
		let (mut decisions, _) = negotiate(&request.requirements, &evidence, request.policy)?;
		if let Some(decision) = negotiate_native_option(
			request.native_option.as_ref(),
			&candidate.wire_target.codec,
			request.policy.allow_dropped_preferences,
		)? {
			decisions.push(decision);
		}
		Ok(EvaluatedCandidate {
			candidate: candidate.clone(),
			runtime,
			decisions,
			price: price_score(&candidate.policy_model),
		})
	}
}

impl Planner for Router {
	fn plan(&self, call: &Call, now: Instant) -> Result<ExecutionPlan, Error> {
		self.plan_call(call, now)
	}

	fn validate(&self, plan: &ExecutionPlan, now: Instant) -> Result<(), Error> {
		plan.validate(now, self.registry.catalog_revision(), self.registry.generation())
	}
}

fn target_route_allowed(target: &Target, route: &RouteId, provider: &ProviderId) -> bool {
	match target {
		Target::Model(_) => true,
		Target::Provider { provider: expected, .. } => expected == provider,
		Target::Route { route: expected, .. } => expected == route,
		Target::ProviderService(_) | Target::RouteService(_) => false,
	}
}

fn extract_requirements(operation: &OperationCall) -> Vec<CapabilityRequirement> {
	let mut requirements = Vec::new();
	match operation {
		OperationCall::Chat(request) => {
			push_setting(&mut requirements, &request.tool_choice, "chat.tools.choice");
			push_setting(&mut requirements, &request.output, "chat.structured_output");
			push_setting(&mut requirements, &request.reasoning, "chat.reasoning");
			push_setting(&mut requirements, &request.verbosity, "chat.verbosity");
			push_setting(&mut requirements, &request.cache_retention, "chat.prompt_cache");
			push_setting(&mut requirements, &request.service_tier, "chat.service_tier");
			if request.top_logprobs.is_some() {
				push_required(&mut requirements, "chat.logprobs");
			}
			if !request.safety.is_empty() {
				push_required(&mut requirements, "chat.safety");
			}
			if request.sampling.seed.is_some() {
				push_required(&mut requirements, "chat.seed");
			}
		},
		OperationCall::Embed(request) => {
			push_setting(&mut requirements, &request.dimensions, "embed.dimensions");
			push_setting(&mut requirements, &request.normalize, "embed.normalize");
		},
		OperationCall::GenerateImage(request) => {
			push_setting(&mut requirements, &request.dimensions, "image.dimensions");
			push_setting(&mut requirements, &request.quality, "image.quality");
			push_setting(&mut requirements, &request.background, "image.background");
			push_setting(&mut requirements, &request.format, "image.format");
			push_setting(&mut requirements, &request.style, "image.style");
			if !request.references.is_empty() {
				push_required(&mut requirements, "image.references");
			}
			if request.mask.is_some() {
				push_required(&mut requirements, "image.mask");
			}
		},
		OperationCall::GenerateVideo(request) => {
			push_setting(&mut requirements, &request.duration_ms, "video.duration");
			push_setting(&mut requirements, &request.dimensions, "video.dimensions");
			push_setting(&mut requirements, &request.frames_per_second, "video.fps");
			push_setting(&mut requirements, &request.audio, "video.audio");
			if request.reference.is_some() {
				push_required(&mut requirements, "video.reference");
			}
		},
		OperationCall::Speak(request) => {
			push_setting(&mut requirements, &request.format, "speech.format");
			push_setting(&mut requirements, &request.sample_rate_hz, "speech.sample_rate");
			push_setting(&mut requirements, &request.speed, "speech.speed");
			push_setting(&mut requirements, &request.timestamps, "speech.timestamps");
		},
		OperationCall::Transcribe(request) => {
			push_setting(&mut requirements, &request.diarization, "transcription.diarization");
			push_setting(&mut requirements, &request.timestamps, "transcription.timestamps");
			if request.translate_to_english {
				push_required(&mut requirements, "transcription.translation");
			}
		},
		OperationCall::Realtime(request) => {
			push_setting(&mut requirements, &request.input_audio, "realtime.input_audio");
			push_setting(&mut requirements, &request.output_audio, "realtime.output_audio");
			push_setting(&mut requirements, &request.turn_detection, "realtime.turn_detection");
			if !request.tools.is_empty() {
				push_required(&mut requirements, "realtime.tools");
			}
		},
		OperationCall::Search(request) => {
			if request.recency.is_some() {
				push_required(&mut requirements, "search.recency");
			}
			if !request.include_domains.is_empty() || !request.exclude_domains.is_empty() {
				push_required(&mut requirements, "search.domains");
			}
			if request.locale.is_some() {
				push_required(&mut requirements, "search.locale");
			}
			push_setting(&mut requirements, &request.synthesize_answer, "search.answer_synthesis");
		},
		_ => {},
	}
	requirements
}

fn push_setting<T>(
	output: &mut Vec<CapabilityRequirement>,
	setting: &crate::call::Setting<T>,
	feature: &'static str,
) {
	let strength = match setting {
		crate::call::Setting::Unset => return,
		crate::call::Setting::Require(_) => crate::plan::RequirementStrength::Required,
		crate::call::Setting::Prefer(_) => crate::plan::RequirementStrength::Preferred,
	};
	output.push(CapabilityRequirement {
		feature: crate::receipt::FeatureId(Str::from(feature)),
		strength,
	});
}

fn push_required(output: &mut Vec<CapabilityRequirement>, feature: &'static str) {
	output.push(CapabilityRequirement {
		feature:  crate::receipt::FeatureId(Str::from(feature)),
		strength: crate::plan::RequirementStrength::Required,
	});
}

fn catalog_capability_evidence(
	model: &PolicyModel,
	requirement: &CapabilityRequirement,
) -> crate::plan::CapabilityEvidence {
	let availability = match requirement.feature.0.as_str() {
		feature if feature.starts_with("chat.") => {
			model.capabilities.chat.as_ref().map(|chat| match feature {
				"chat.tools.choice" => availability_class(&chat.tools),
				"chat.structured_output" => availability_class(&chat.structured_output),
				"chat.reasoning" => availability_class(&chat.reasoning),
				"chat.verbosity" => availability_class(&chat.text_verbosity),
				"chat.prompt_cache" => availability_class(&chat.prompt_caching),
				"chat.service_tier" => availability_class(&chat.service_tiers),
				"chat.logprobs" => availability_class(&chat.logprobs),
				"chat.safety" => availability_class(&chat.safety),
				"chat.seed" => availability_class(&chat.determinism),
				_ => CapabilityAvailability::Unknown,
			})
		},
		"embed.dimensions" => model
			.capabilities
			.embeddings
			.as_ref()
			.map(|value| availability_class(&value.dimensions)),
		"search.recency" | "search.domains" | "search.locale" | "search.answer_synthesis" => model
			.capabilities
			.search
			.as_ref()
			.map(|_| CapabilityAvailability::Native),
		"image.references" | "image.mask" | "image.dimensions" | "image.quality"
		| "image.background" | "image.format" | "image.style" => model
			.capabilities
			.image
			.as_ref()
			.map(|_| CapabilityAvailability::Native),
		"video.reference" | "video.duration" | "video.dimensions" | "video.fps" | "video.audio" => {
			model
				.capabilities
				.video
				.as_ref()
				.map(|_| CapabilityAvailability::Native)
		},
		"speech.format" | "speech.sample_rate" | "speech.speed" | "speech.timestamps" => model
			.capabilities
			.speech
			.as_ref()
			.map(|_| CapabilityAvailability::Native),
		"transcription.diarization" | "transcription.timestamps" | "transcription.translation" => {
			model
				.capabilities
				.transcription
				.as_ref()
				.map(|_| CapabilityAvailability::Native)
		},
		"realtime.input_audio"
		| "realtime.output_audio"
		| "realtime.turn_detection"
		| "realtime.tools" => model
			.capabilities
			.realtime
			.as_ref()
			.map(|_| CapabilityAvailability::Native),
		"embed.normalize" => model
			.capabilities
			.embeddings
			.as_ref()
			.map(|_| CapabilityAvailability::Unknown),
		_ => None,
	}
	.unwrap_or(CapabilityAvailability::Unknown);
	crate::plan::CapabilityEvidence {
		feature: requirement.feature.clone(),
		availability,
		reason: ReasonId(Str::from("catalog-capability-evidence")),
	}
}

fn availability_class<C>(availability: &crate::catalog::Availability<C>) -> CapabilityAvailability {
	match availability {
		crate::catalog::Availability::Unsupported => CapabilityAvailability::Unsupported,
		crate::catalog::Availability::Unknown => CapabilityAvailability::Unknown,
		crate::catalog::Availability::Native(_) => CapabilityAvailability::Native,
		crate::catalog::Availability::Emulated { method, .. } => {
			CapabilityAvailability::Emulated(*method)
		},
	}
}

fn operation_policy(operation: &OperationCall) -> PlanningPolicy {
	let negotiation = match operation {
		OperationCall::Chat(request) => &request.negotiation,
		OperationCall::Embed(request) => &request.negotiation,
		OperationCall::GenerateImage(request) => &request.negotiation,
		OperationCall::GenerateVideo(request) => &request.negotiation,
		OperationCall::Speak(request) => &request.negotiation,
		OperationCall::Transcribe(request) => &request.negotiation,
		OperationCall::Realtime(request) => &request.negotiation,
		OperationCall::Search(request) => &request.negotiation,
		OperationCall::CountTokens(_)
		| OperationCall::Tokenize(_)
		| OperationCall::Detokenize(_)
		| OperationCall::Usage(_)
		| OperationCall::DiscoverModels(_)
		| OperationCall::Auth(_)
		| OperationCall::Native(_) => return PlanningPolicy::default(),
	};
	PlanningPolicy {
		allow_emulation:           !matches!(
			negotiation.emulation,
			crate::call::EmulationPolicy::Forbid
		),
		allow_lossy_emulation:     matches!(
			negotiation.emulation,
			crate::call::EmulationPolicy::AllowDeclaredLossy
		),
		allow_unknown_preferences: matches!(
			negotiation.unknown,
			crate::call::UnknownCapabilityPolicy::AllowPreferences
		),
		allow_dropped_preferences: matches!(
			negotiation.vendor_option_mismatch,
			crate::call::MismatchPolicy::DropPreferred
		),
	}
}
fn operation_replay(operation: &OperationCall) -> ReplayRequirements {
	let mut parts = Vec::new();
	let mut semantic_retry_possible = false;
	match operation {
		OperationCall::Chat(request) => {
			for message in request.messages.iter() {
				for content in message.content.iter() {
					collect_content_replay(content, &mut parts);
				}
			}
			semantic_retry_possible = !matches!(request.output, crate::call::Setting::Unset)
				|| matches!(
					request.tool_choice,
					crate::call::Setting::Require(crate::call::ToolChoice::Named(_))
						| crate::call::Setting::Require(crate::call::ToolChoice::Required)
				);
		},
		OperationCall::GenerateImage(request) => {
			for media in request.references.iter() {
				collect_media_replay(media, &mut parts);
			}
			if let Some(mask) = &request.mask {
				collect_media_replay(mask, &mut parts);
			}
		},
		OperationCall::GenerateVideo(request) => {
			if let Some(media) = &request.reference {
				collect_media_replay(media, &mut parts);
			}
		},
		OperationCall::Transcribe(request) => collect_media_replay(&request.audio, &mut parts),
		OperationCall::Realtime(_) => parts.push(crate::body::Replayability::OneShot),
		OperationCall::Native(request) => {
			if let Some(crate::call::NativePayload::Body(body)) = request.payload.as_ref() {
				parts.push(body.replay_evidence().replayability);
			}
		},
		_ => {},
	}
	let replayability = crate::body::Replayability::aggregate(parts);
	ReplayRequirements {
		replayability,
		semantic_retry_possible,
		staging_explicit: false,
		staging_limit: None,
	}
}

fn collect_media_replay(
	media: &crate::call::MediaInput,
	parts: &mut Vec<crate::body::Replayability>,
) {
	if let crate::call::MediaInput::Body { body, .. } = media {
		parts.push(body.replay_evidence().replayability);
	}
}

fn collect_content_replay(
	content: &crate::call::ContentPart,
	parts: &mut Vec<crate::body::Replayability>,
) {
	match content {
		crate::call::ContentPart::Image(media)
		| crate::call::ContentPart::Audio(media)
		| crate::call::ContentPart::Document(media) => collect_media_replay(media, parts),
		crate::call::ContentPart::ToolResult { content, .. } => {
			for value in content.iter() {
				if let crate::call::ToolResultContent::Image(media)
				| crate::call::ToolResultContent::Document(media) = value
				{
					collect_media_replay(media, parts);
				}
			}
		},
		_ => {},
	}
}

fn chat_thinking_effort(
	operation: Option<&OperationCall>,
	policy: Option<&crate::catalog::ThinkingPolicy>,
) -> Result<Option<crate::catalog::ThinkingEffort>, Error> {
	let Some(OperationCall::Chat(request)) = operation else {
		return Ok(None);
	};
	let requested = match &request.reasoning {
		crate::call::Setting::Unset => None,
		crate::call::Setting::Require(reasoning) | crate::call::Setting::Prefer(reasoning) => {
			reasoning.effort
		},
	};
	if policy.is_none() && requested.is_some() {
		return Err(capability_error(
			ErrorKind::CapabilityMismatch,
			OperationKind::Chat,
			"thinking-policy-unavailable",
		));
	}
	Ok(requested.map(reasoning_to_thinking))
}

const fn reasoning_to_thinking(
	effort: crate::catalog::ReasoningEffort,
) -> crate::catalog::ThinkingEffort {
	match effort {
		crate::catalog::ReasoningEffort::Off => crate::catalog::ThinkingEffort::Off,
		crate::catalog::ReasoningEffort::Minimal => crate::catalog::ThinkingEffort::Minimal,
		crate::catalog::ReasoningEffort::Low => crate::catalog::ThinkingEffort::Low,
		crate::catalog::ReasoningEffort::Medium => crate::catalog::ThinkingEffort::Medium,
		crate::catalog::ReasoningEffort::High => crate::catalog::ThinkingEffort::High,
		crate::catalog::ReasoningEffort::Xhigh => crate::catalog::ThinkingEffort::XHigh,
		crate::catalog::ReasoningEffort::Max => crate::catalog::ThinkingEffort::Max,
	}
}

fn model_less_route_is_candidate(
	route: &RouteId,
	pinned_route: Option<&RouteId>,
	provider_support: bool,
	route_operations: Option<crate::catalog::OperationBits>,
	operation: OperationKind,
) -> bool {
	!pinned_route.is_some_and(|pinned| pinned != route)
		&& model_less_operation_advertised(provider_support, route_operations, operation)
}

const fn model_less_operation_advertised(
	provider_support: bool,
	route_operations: Option<crate::catalog::OperationBits>,
	operation: OperationKind,
) -> bool {
	match route_operations {
		Some(operations) => operations.contains_kind(operation),
		None => provider_support,
	}
}

struct EvaluatedCandidate {
	candidate: RouteCandidate,
	runtime:   RuntimeRouteEvidence,
	decisions: Vec<crate::plan::NegotiationDecision>,
	price:     u128,
}

fn selector_accepts(
	target: &Target,
	primary: &ModelKey,
	scope: &FallbackScope,
	candidate: &RouteCandidate,
	is_primary: bool,
) -> bool {
	if !is_primary {
		return candidate.model != *primary
			&& scope.explicit.iter().any(|model| model == &candidate.model);
	}
	match target {
		Target::Model(model) => &candidate.model == model,
		Target::Provider { provider, model } => {
			&candidate.model == model && &candidate.provider == provider
		},
		Target::Route { route, model } => {
			&candidate.model == model && &candidate.wire_target.route == route
		},
		Target::ProviderService(_) | Target::RouteService(_) => false,
	}
}

fn compare_evaluated(left: &EvaluatedCandidate, right: &EvaluatedCandidate) -> Ordering {
	let left_affinity = left.runtime.affinity;
	let right_affinity = right.runtime.affinity;
	right_affinity
		.cmp(&left_affinity)
		.then_with(|| health_rank(left.runtime.health).cmp(&health_rank(right.runtime.health)))
		.then_with(|| {
			right
				.runtime
				.quota_millionths
				.cmp(&left.runtime.quota_millionths)
		})
		.then_with(|| left.runtime.latency.cmp(&right.runtime.latency))
		.then_with(|| left.price.cmp(&right.price))
		.then_with(|| left.candidate.provider.cmp(&right.candidate.provider))
		.then_with(|| {
			left
				.candidate
				.wire_target
				.route
				.cmp(&right.candidate.wire_target.route)
		})
		.then_with(|| {
			left
				.candidate
				.wire_target
				.codec
				.cmp(&right.candidate.wire_target.codec)
		})
}

const fn health_rank(health: crate::plan::RouteHealth) -> u8 {
	match health {
		crate::plan::RouteHealth::Healthy => 0,
		crate::plan::RouteHealth::Unknown => 1,
		crate::plan::RouteHealth::Degraded => 2,
		crate::plan::RouteHealth::Unavailable => 3,
	}
}

fn price_score(model: &PolicyModel) -> u128 {
	model.pricing.components.iter().fold(0_u128, |sum, price| {
		let weight = match price.unit {
			PriceUnit::MtokInput
			| PriceUnit::MtokOutput
			| PriceUnit::MtokCacheRead
			| PriceUnit::MtokCacheWrite => 1_u128,
			PriceUnit::Image
			| PriceUnit::VideoSecond
			| PriceUnit::AudioSecond
			| PriceUnit::McharInput
			| PriceUnit::Request => 1_u128,
		};
		sum.saturating_add(u128::from(price.nanos_usd).saturating_mul(weight))
	})
}

fn prefer_error(current: Option<Error>, next: Error) -> Error {
	match current {
		None => next,
		Some(current) if error_rank(next.kind) < error_rank(current.kind) => next,
		Some(current) => current,
	}
}

const fn error_rank(kind: ErrorKind) -> u8 {
	match kind {
		ErrorKind::CodecMismatch => 0,
		ErrorKind::StagingRequired | ErrorKind::ReplayRequired => 1,
		ErrorKind::CapabilityMismatch => 2,
		ErrorKind::CapabilityUnknown => 3,
		ErrorKind::RouteUnavailable => 4,
		ErrorKind::TargetNotFound => 5,
		_ => 6,
	}
}

fn target_not_found(target: &Target) -> Error {
	let selector = match target {
		Target::Model(model) => model.as_str(),
		Target::Provider { model, .. } | Target::Route { model, .. } => model.as_str(),
		Target::ProviderService(provider) => provider.as_str(),
		Target::RouteService(route) => route.as_str(),
	};
	Error::planning(
		ErrorKind::TargetNotFound,
		ErrorDetail::Target { selector: Str::from(selector) },
		ExecutionReceipt::default(),
	)
}

fn target_route_not_found(route: &RouteId) -> Error {
	Error::planning(
		ErrorKind::TargetNotFound,
		ErrorDetail::Target { selector: Str::from(route.as_str()) },
		ExecutionReceipt::default(),
	)
}

fn capability_error(kind: ErrorKind, operation: OperationKind, reason: &'static str) -> Error {
	Error::planning(
		kind,
		ErrorDetail::Capability {
			feature: Str::from(operation.to_string()),
			reason:  ReasonId(Str::from(reason)),
		},
		ExecutionReceipt::default(),
	)
}

fn route_contract_error(route: &RouteId, reason: &'static str) -> Error {
	let mut error = Error::planning(
		ErrorKind::RouteUnavailable,
		ErrorDetail::Capability {
			feature: Str::from(route.as_str()),
			reason:  ReasonId(Str::from(reason)),
		},
		ExecutionReceipt::default(),
	);
	error.route = Some(route.clone());
	error
}

#[cfg(test)]
mod tests {
	use super::*;

	const OPERATIONS: [OperationKind; 15] = [
		OperationKind::Chat,
		OperationKind::CountTokens,
		OperationKind::Tokenize,
		OperationKind::Detokenize,
		OperationKind::Embed,
		OperationKind::GenerateImage,
		OperationKind::GenerateVideo,
		OperationKind::Speak,
		OperationKind::Transcribe,
		OperationKind::Realtime,
		OperationKind::Search,
		OperationKind::Usage,
		OperationKind::DiscoverModels,
		OperationKind::Auth,
		OperationKind::Native,
	];

	#[test]
	fn provider_and_route_service_targets_obey_all_fifteen_catalog_operation_bits() {
		let route = RouteId::from("selected");
		let other = RouteId::from("other");
		for operation in OPERATIONS {
			let exact = crate::catalog::OperationBits::for_kind(operation);
			let different = if operation == OperationKind::Chat {
				crate::catalog::OperationBits::for_kind(OperationKind::Usage)
			} else {
				crate::catalog::OperationBits::for_kind(OperationKind::Chat)
			};

			assert!(model_less_route_is_candidate(&route, None, true, None, operation));
			assert!(!model_less_route_is_candidate(&route, None, false, None, operation));
			assert!(model_less_route_is_candidate(&route, None, false, Some(exact), operation));
			assert!(!model_less_route_is_candidate(&route, None, true, Some(different), operation));

			assert!(model_less_route_is_candidate(
				&route,
				Some(&route),
				false,
				Some(exact),
				operation
			));
			assert!(!model_less_route_is_candidate(
				&route,
				Some(&other),
				false,
				Some(exact),
				operation
			));
			assert!(!model_less_route_is_candidate(
				&route,
				Some(&route),
				true,
				Some(different),
				operation
			));
		}
	}
}
