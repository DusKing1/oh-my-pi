//! Canonical operation policy around one constructed route stack.

use std::{
	future::{Future, poll_fn},
	num::NonZeroU32,
	task::{Context, Poll},
};

use tower::{Layer, Service};

use crate::{
	answer::{Answer, AnswerBody},
	call::{CountAccuracy, OperationCall},
	error::Error,
	layer::LayerCall,
	operation::{
		embedding::{
			EmbeddingServiceConfig, normalize_vector, plan_embedding, validate_embedding_batch,
		},
		native::{NativePolicy, validate_native_response},
		search::{HostedSearchIntent, SearchPlan, finalize_search, plan_search},
		tokens::validate_provenance,
		usage::{UsageServiceConfig, normalize_report},
	},
};

/// Route-executable embedding behavior not represented by per-model catalog
/// capabilities.
#[derive(Clone, Copy, Debug)]
pub struct EmbeddingRoutePolicy {
	/// Backend normalization behavior.
	pub normalization:          crate::operation::embedding::NormalizationSupport,
	/// Maximum pre-tokenized input length accepted by the constructed backend.
	pub maximum_input_tokens:   Option<u32>,
	/// Whether the backend implements requested text truncation.
	pub native_text_truncation: bool,
}

/// Route-owned executable facts required by operation policy.
#[derive(Clone, Debug)]
pub struct OperationPolicyConfig {
	/// Route behavior supplementing the exact selected model's embedding
	/// capabilities.
	pub embedding:              Option<EmbeddingRoutePolicy>,
	/// Native method/path/framing allowlist when native execution is
	/// constructed.
	pub native:                 Option<NativePolicy>,
	/// Usage observation freshness policy.
	pub usage:                  UsageServiceConfig,
	/// Largest accepted discovery page.
	pub discovery_maximum_page: Option<NonZeroU32>,
	/// Whether the constructed token-count endpoint is exact.
	pub exact_token_count:      bool,
}

/// Construction-time layer applying canonical operation behavior exactly once.
#[derive(Clone, Debug)]
pub struct OperationPolicyLayer {
	config: OperationPolicyConfig,
}

impl OperationPolicyLayer {
	/// Constructs route operation policy from catalog and executable service
	/// facts.
	pub const fn new(config: OperationPolicyConfig) -> Self {
		Self { config }
	}
}

impl<S> Layer<S> for OperationPolicyLayer {
	type Service = OperationPolicyService<S>;

	fn layer(&self, inner: S) -> Self::Service {
		OperationPolicyService { inner, config: self.config.clone() }
	}
}

/// Service applying preflight partitioning and post-response canonical
/// validation.
#[derive(Clone, Debug)]
pub struct OperationPolicyService<S> {
	inner:  S,
	config: OperationPolicyConfig,
}

impl<S> Service<LayerCall<crate::call::Call>> for OperationPolicyService<S>
where
	S: Service<LayerCall<crate::call::Call>, Response = Answer, Error = Error>
		+ Clone
		+ Send
		+ 'static,
	S::Future: Send + 'static,
{
	type Error = Error;
	type Response = Answer;

	type Future = impl Future<Output = Result<Answer, Error>> + Send;

	fn poll_ready(&mut self, context: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		self.inner.poll_ready(context)
	}

	fn call(&mut self, request: LayerCall<crate::call::Call>) -> Self::Future {
		let original = request.payload.clone();
		let config = self.config.clone();
		let prepared = prepare(&request, &config);
		let first = prepared.as_ref().ok().and_then(|prepared| {
			prepared.calls.first().map(|call| {
				self
					.inner
					.call(LayerCall { payload: call.clone(), context: request.context.clone() })
			})
		});
		let mut later = self.inner.clone();
		async move {
			let prepared = prepared?;
			let Some(first) = first else {
				return Err(crate::operation::wrong_operation(&original, original.operation.kind()));
			};
			let mut answer = match first.await {
				Ok(answer) => answer,
				Err(mut error) => {
					error.receipt = request.context.receipt();
					return Err(error);
				},
			};
			for call in prepared.calls.iter().skip(1) {
				poll_fn(|context| later.poll_ready(context)).await?;
				let next = match later
					.call(LayerCall { payload: call.clone(), context: request.context.clone() })
					.await
				{
					Ok(next) => next,
					Err(mut error) => {
						error.receipt = request.context.receipt();
						return Err(error);
					},
				};
				if let Err(mut error) = merge_embedding_answer(&mut answer, next) {
					error.receipt = request.context.receipt();
					return Err(error);
				}
			}
			answer.receipt = request.context.receipt();
			finish(answer, &original.operation, prepared, &config)
		}
	}
}

#[derive(Clone, Debug)]
struct PreparedOperation {
	calls:               Vec<crate::call::Call>,
	embedding_normalize: bool,
	search:              Option<SearchPlan>,
}

fn prepare(
	request: &LayerCall<crate::call::Call>,
	config: &OperationPolicyConfig,
) -> Result<PreparedOperation, Error> {
	let mut calls = Vec::new();
	let mut embedding_normalize = false;
	let mut search = None;
	let execution = request.payload.execution.as_ref().ok_or_else(|| {
		crate::operation::media_validation_error(
			request.payload.operation.kind(),
			"operation_policy_requires_execution_plan",
		)
	})?;
	if execution.operation != request.payload.operation.kind() {
		return Err(crate::operation::wrong_operation(&request.payload, execution.operation));
	}
	match &request.payload.operation {
		OperationCall::Chat(chat) => {
			for tool in chat.hosted_tools.iter() {
				HostedSearchIntent::from_tool(tool)?;
			}
		},
		OperationCall::Embed(embed) => {
			let Some(route_policy) = config.embedding else {
				return Err(crate::operation::media_validation_error(
					request.payload.operation.kind(),
					"embedding_policy_not_constructed",
				));
			};
			let capabilities = execution
				.policy_model
				.as_ref()
				.and_then(|model| model.capabilities.embeddings.clone())
				.ok_or_else(|| {
					crate::operation::media_validation_error(
						request.payload.operation.kind(),
						"selected_model_has_no_embedding_capabilities",
					)
				})?;
			let embedding = EmbeddingServiceConfig {
				capabilities,
				normalization: route_policy.normalization,
				maximum_input_tokens: route_policy.maximum_input_tokens,
				native_text_truncation: route_policy.native_text_truncation,
			};
			let maximum = NonZeroU32::new(embedding.capabilities.maximum_batch.unwrap_or(u32::MAX))
				.ok_or_else(|| {
					crate::operation::media_validation_error(
						request.payload.operation.kind(),
						"zero_embedding_batch_capacity",
					)
				})?;
			let plan = plan_embedding(embed, &embedding, maximum)?;
			if plan.pages.len() > 1
				&& execution.replay == crate::plan::ReplayPlan::OneShotSingleAttempt
			{
				return Err(crate::operation::media_validation_error(
					request.payload.operation.kind(),
					"one_shot_embedding_cannot_open_multiple_batches",
				));
			}
			embedding_normalize = plan.normalize_locally;
			for page in plan.pages {
				let mut call = request.payload.clone();
				call.operation = OperationCall::Embed(page);
				calls.push(call);
			}
		},
		OperationCall::Search(search_request) => {
			let capabilities = execution
				.policy_model
				.as_ref()
				.and_then(|model| model.capabilities.search.clone())
				.ok_or_else(|| {
					crate::operation::media_validation_error(
						request.payload.operation.kind(),
						"selected_model_has_no_search_capabilities",
					)
				})?;
			let plan = plan_search(search_request, capabilities, execution.planned_at)?;
			let mut call = request.payload.clone();
			call.operation = OperationCall::Search(plan.backend_request());
			calls.push(call);
			search = Some(plan);
		},
		OperationCall::CountTokens(count)
			if count.accuracy == CountAccuracy::Exact && !config.exact_token_count =>
		{
			return Err(crate::operation::media_validation_error(
				request.payload.operation.kind(),
				"exact_token_count_not_constructed",
			));
		},
		OperationCall::DiscoverModels(discovery) => {
			let Some(maximum) = config.discovery_maximum_page else {
				return Err(crate::operation::media_validation_error(
					request.payload.operation.kind(),
					"discovery_policy_not_constructed",
				));
			};
			if discovery.page_size == 0 || discovery.page_size > maximum.get() {
				return Err(crate::operation::media_validation_error(
					request.payload.operation.kind(),
					"invalid_discovery_page_size",
				));
			}
		},
		OperationCall::Native(native) => {
			let Some(policy) = &config.native else {
				return Err(crate::operation::media_validation_error(
					request.payload.operation.kind(),
					"native_policy_not_constructed",
				));
			};
			policy.authorize(native)?;
		},
		_ => {},
	}
	if calls.is_empty() {
		calls.push(request.payload.clone());
	}
	Ok(PreparedOperation { calls, embedding_normalize, search })
}

fn merge_embedding_answer(target: &mut Answer, next: Answer) -> Result<(), Error> {
	if target.meta.provider != next.meta.provider
		|| target.meta.route != next.meta.route
		|| target.meta.model != next.meta.model
	{
		return Err(crate::operation::media_protocol_error(
			crate::catalog::OperationKind::Embed,
			"embedding_batch_route_changed",
		));
	}
	let target_kind = target.body.kind();
	let AnswerBody::Embeddings(target_batch) = &mut target.body else {
		return Err(Error::body_variant_mismatch(
			crate::catalog::OperationKind::Embed,
			target_kind,
			target.receipt.clone(),
		));
	};
	let Answer { receipt, body, .. } = next;
	let AnswerBody::Embeddings(mut next_batch) = body else {
		return Err(Error::body_variant_mismatch(
			crate::catalog::OperationKind::Embed,
			body.kind(),
			receipt,
		));
	};
	if target_batch.dimensions != next_batch.dimensions {
		return Err(crate::operation::media_protocol_error(
			crate::catalog::OperationKind::Embed,
			"embedding_page_dimensions_changed",
		));
	}
	let offset = target_batch.embeddings.len() as u32;
	for embedding in &mut next_batch.embeddings {
		embedding.index = embedding.index.saturating_add(offset);
	}
	target_batch.embeddings.append(&mut next_batch.embeddings);
	target_batch.usage += next_batch.usage;
	Ok(())
}

fn finish(
	mut answer: Answer,
	operation: &OperationCall,
	prepared: PreparedOperation,
	config: &OperationPolicyConfig,
) -> Result<Answer, Error> {
	match (operation, &mut answer.body) {
		(OperationCall::CountTokens(request), AnswerBody::Tokens(count)) => {
			validate_provenance(&count.provenance)?;
			if request.accuracy == CountAccuracy::Exact && !count.provenance.exact {
				return Err(crate::operation::media_protocol_error(
					operation.kind(),
					"exact_token_count_returned_estimate",
				));
			}
		},
		(OperationCall::Tokenize(_), AnswerBody::TokenIds(sequence)) => {
			validate_provenance(&sequence.provenance)?;
			if !sequence.provenance.exact {
				return Err(crate::operation::media_protocol_error(
					operation.kind(),
					"tokenize_returned_estimate",
				));
			}
		},
		(OperationCall::Detokenize(_), AnswerBody::Text(text)) => {
			validate_provenance(&text.provenance)?;
			if !text.provenance.exact {
				return Err(crate::operation::media_protocol_error(
					operation.kind(),
					"detokenize_returned_estimate",
				));
			}
		},
		(OperationCall::Embed(request), AnswerBody::Embeddings(batch)) => {
			validate_embedding_batch(batch, request.inputs.len())?;
			if let crate::call::Setting::Require(dimensions) = &request.dimensions
				&& batch.dimensions != *dimensions
			{
				return Err(crate::operation::media_protocol_error(
					operation.kind(),
					"required_embedding_dimensions_not_returned",
				));
			}
			batch.embeddings.sort_by_key(|embedding| embedding.index);
			if prepared.embedding_normalize {
				for embedding in &mut batch.embeddings {
					normalize_vector(&mut embedding.values)?;
				}
			}
		},
		(OperationCall::Search(_), AnswerBody::Search(results)) => {
			let page = crate::operation::search::SearchPage {
				documents: results
					.results
					.drain(..)
					.map(|result| crate::operation::search::SearchDocument {
						url:          result.url,
						title:        result.title,
						snippet:      result.snippet,
						score:        result.score,
						published_at: result.published_at,
						locale:       None,
					})
					.collect(),
				answer:    results.answer.take(),
				usage:     results.usage,
			};
			*results = finalize_search(prepared.search.as_ref().expect("search plan exists"), page)?;
		},
		(OperationCall::Usage(request), AnswerBody::Usage(report)) => {
			normalize_report(report, request, config.usage)?
		},
		(OperationCall::DiscoverModels(request), AnswerBody::Models(page)) => {
			validate_discovery_page(page, request, &answer.meta)?;
		},
		(OperationCall::Native(request), AnswerBody::Native(response)) => {
			let rule = config
				.native
				.as_ref()
				.expect("native policy exists")
				.authorize(request)?;
			validate_native_response(request, *rule, response)?;
		},
		_ => {},
	}
	Ok(answer)
}

fn validate_discovery_page(
	page: &crate::answer::ModelDiscoveryPage,
	request: &crate::call::DiscoveryRequest,
	meta: &crate::answer::ResponseMeta,
) -> Result<(), Error> {
	if meta.model.is_some()
		|| request
			.provider
			.as_ref()
			.is_some_and(|provider| provider != &meta.provider)
		|| request
			.route
			.as_ref()
			.is_some_and(|route| route != &meta.route)
		|| page.models.len() > request.page_size as usize
		|| page
			.next_cursor
			.as_ref()
			.is_some_and(|cursor| cursor.is_empty())
	{
		return Err(crate::operation::media_protocol_error(
			crate::catalog::OperationKind::DiscoverModels,
			"invalid_normalized_discovery_page",
		));
	}
	if let Some(operation) = request.operation
		&& page
			.models
			.iter()
			.any(|model| !model.capabilities.operations.contains_kind(operation))
	{
		return Err(crate::operation::media_protocol_error(
			crate::catalog::OperationKind::DiscoverModels,
			"discovery_page_contains_unrequested_operation",
		));
	}
	Ok(())
}

#[cfg(test)]
mod tests {
	use std::{
		sync::{
			Arc,
			atomic::{AtomicUsize, Ordering},
		},
		time::{Duration, Instant, SystemTime},
	};

	use tower::{ServiceExt, service_fn};

	use super::{OperationPolicyConfig, OperationPolicyLayer, merge_embedding_answer};
	use crate::{
		answer::{
			Answer, AnswerBody, Embedding, EmbeddingBatch, ResponseMeta, TokenCount,
			TokenizerProvenance,
		},
		body::{AttemptBodyEvidence, Replayability, RetryDecision, RetryDecisionReason},
		call::{Call, CallMeta, CountAccuracy, CountTokensRequest, OperationCall, Target},
		catalog::{CatalogRevision, CodecId, OperationKind, ProviderId, RouteId, WirePolicy},
		layer::{ExecutionContext, LayerCall},
		plan::{
			CapabilityAvailability, ExecutionPlan, FallbackScope, ReplayPlan, RouteHealth,
			RuntimeRouteEvidence,
		},
		receipt::{
			AttemptOutcome, AttemptReceipt, Cost, ExecutionBudget, ExecutionReceipt, ProviderEvidence,
			Usage,
		},
	};

	fn exact_count_call() -> Call {
		let route = RouteId::from("route");
		let operation = OperationKind::CountTokens;
		let plan = ExecutionPlan {
			planned_at: SystemTime::UNIX_EPOCH,
			catalog_revision: CatalogRevision::from("catalog"),
			registry_generation: 1,
			expires_at: Instant::now() + Duration::from_secs(60),
			operation,
			model: None,
			provider: ProviderId::from("provider"),
			route: route.clone(),
			codec: CodecId::from("codec"),
			policy_model: None,
			wire_policy: Arc::new(WirePolicy::default()),
			thinking_policy: None,
			thinking_selection: None,
			decisions: Arc::from([]),
			fallback_scope: FallbackScope { primary: None, explicit: Arc::from([]) },
			fallbacks: Arc::from([]),
			replay: ReplayPlan::Replayable,
			budget: ExecutionBudget::default(),
			runtime_evidence: RuntimeRouteEvidence {
				route,
				generation: 1,
				health: RouteHealth::Healthy,
				quota_millionths: 1_000_000,
				latency: Duration::ZERO,
				affinity: false,
				operation: CapabilityAvailability::Native,
				capabilities: Arc::from([]),
			},
			wire_target: None,
		};
		let mut call = Call::new(
			CallMeta {
				id:       crate::id::RequestId::from("request"),
				target:   Target::Route {
					route: plan.route.clone(),
					model: omp_llm_catalog::ModelKey::from("model"),
				},
				deadline: None,
				budget:   ExecutionBudget::default(),
				session:  None,
			},
			OperationCall::CountTokens(Arc::new(CountTokensRequest {
				messages: Arc::new([]),
				tools:    Arc::new([]),
				accuracy: CountAccuracy::Exact,
			})),
		);
		call.execution = Some(Arc::new(plan));
		call
	}

	#[tokio::test]
	async fn policy_rejects_exact_count_that_raw_route_would_execute() {
		let calls = Arc::new(AtomicUsize::new(0));
		let raw_calls = calls.clone();
		let raw = service_fn(move |call: LayerCall<Call>| {
			raw_calls.fetch_add(1, Ordering::Relaxed);
			async move {
				Ok::<_, crate::Error>(Answer {
					meta:    ResponseMeta {
						request_id:          call.payload.id,
						provider:            ProviderId::from("provider"),
						route:               RouteId::from("route"),
						model:               None,
						provider_request_id: None,
						created_at:          SystemTime::UNIX_EPOCH,
					},
					receipt: ExecutionReceipt::default(),
					body:    AnswerBody::Tokens(TokenCount {
						tokens:     1,
						provenance: TokenizerProvenance {
							tokenizer: "estimate".into(),
							revision:  "1".into(),
							exact:     false,
						},
					}),
				})
			}
		});
		let raw_answer = raw
			.clone()
			.oneshot(LayerCall {
				payload: exact_count_call(),
				context: ExecutionContext::new(ExecutionBudget::default()),
			})
			.await
			.expect("raw codec accepts the request without canonical operation policy");
		assert!(matches!(raw_answer.body, AnswerBody::Tokens(_)));
		assert_eq!(calls.load(Ordering::Relaxed), 1);
		let layer = OperationPolicyLayer::new(OperationPolicyConfig {
			embedding:              None,
			native:                 None,
			usage:                  crate::operation::usage::UsageServiceConfig::new(Duration::MAX),
			discovery_maximum_page: None,
			exact_token_count:      false,
		});
		let context = ExecutionContext::new(ExecutionBudget::default());
		let error = ServiceExt::oneshot(tower::Layer::layer(&layer, raw), LayerCall {
			payload: exact_count_call(),
			context,
		})
		.await
		.expect_err("policy rejects unsupported exactness before raw codec");
		assert_eq!(error.kind, crate::error::ErrorKind::InvalidRequest);
		assert_eq!(calls.load(Ordering::Relaxed), 1);
	}

	#[test]
	fn two_embedding_pages_keep_body_usage_but_not_cumulative_receipts() {
		fn attempt(index: u32) -> AttemptReceipt {
			let mut usage = Usage::default();
			usage.input_tokens = 1;
			AttemptReceipt {
				index,
				hidden: false,
				provider: Some(ProviderId::from("provider")),
				route: Some(RouteId::from("route")),
				account: None,
				principal: None,
				body: AttemptBodyEvidence {
					opened:         true,
					consumed:       true,
					replayability:  Replayability::Replayable,
					retry_decision: RetryDecision::Allow,
					reason:         RetryDecisionReason::ReplayableSource,
				},
				outcome: AttemptOutcome::Succeeded,
				usage,
				cost: Cost::from_micro_usd(1),
				provider_evidence: ProviderEvidence::default(),
				elapsed: Duration::ZERO,
			}
		}
		fn answer(index: u32, attempts: Vec<AttemptReceipt>) -> Answer {
			let mut usage = Usage::default();
			usage.input_tokens = 1;
			let mut receipt = ExecutionReceipt::default();
			receipt.attempts = attempts;
			Answer {
				meta: ResponseMeta {
					request_id:          crate::id::RequestId::from("request"),
					provider:            ProviderId::from("provider"),
					route:               RouteId::from("route"),
					model:               None,
					provider_request_id: None,
					created_at:          SystemTime::UNIX_EPOCH,
				},
				receipt,
				body: AnswerBody::Embeddings(EmbeddingBatch {
					dimensions: 1,
					embeddings: vec![Embedding { index, values: vec![1.0] }],
					usage,
				}),
			}
		}
		let first_attempt = attempt(0);
		let second_attempt = attempt(1);
		let mut combined = answer(0, vec![first_attempt.clone()]);
		merge_embedding_answer(
			&mut combined,
			answer(0, vec![first_attempt.clone(), second_attempt.clone()]),
		)
		.expect("compatible second page");
		assert_eq!(
			combined.receipt.attempts.len(),
			1,
			"cumulative second-page receipt is not merged"
		);
		let AnswerBody::Embeddings(batch) = &combined.body else {
			panic!("embedding body")
		};
		assert_eq!(batch.embeddings.len(), 2);
		assert_eq!(batch.usage.input_tokens, 2);
		let context = ExecutionContext::new(ExecutionBudget::default());
		context.with_receipt(|receipt| {
			receipt.attempts = vec![first_attempt, second_attempt];
			receipt.usage.input_tokens = 2;
			receipt.cost = Cost::from_micro_usd(2);
		});
		combined.receipt = context.receipt();
		assert_eq!(
			combined.receipt.attempts.len(),
			2,
			"shared execution receipt charges each page once"
		);
		assert_eq!(combined.receipt.usage.input_tokens, 2);
		assert_eq!(combined.receipt.cost.micro_usd, 2);
	}
}
