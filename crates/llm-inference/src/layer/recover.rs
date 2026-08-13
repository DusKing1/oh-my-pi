//! Canonical response recovery before semantic output gating.

use std::{
	fmt, mem,
	sync::Arc,
	task::{Context, Poll},
	time::SystemTime,
};

use futures::StreamExt;
use omp_llm_catalog::{PriceUnit, pricing::UsageDimensions};
use tower::{Layer, Service};

use crate::{
	answer::{AnswerBody, ModelDiscoveryPage},
	call::{Call, OperationCall, Setting, StructuredOutput, ToolDefinition},
	codec::{HandshakenResponse, RawCompletion, RawEvent, ToolInputKind, UnvalidatedToolCall},
	error::{Error, ErrorDetail, ErrorKind, ErrorPhase, RetryAction},
	event::{ChatEvent, Completion},
	layer::LayerCall,
	receipt::{AttemptOutcome, AttemptReceipt, Cost, ProviderEvidence, ReasonId},
	recovery::{
		Stage,
		empty::{EmptyCompletionStage, EmptyEvent, EmptyInput},
		json::{JsonEnforcement, JsonRepairLimits, JsonRepairStage},
		reasoning::{ReasoningLimits, ReasoningObservation, ReasoningStallGuard},
		repetition::{OutputVisibility, recovery_record},
		tools::{
			ToolAssembler, ToolAssemblyEvent, ToolAssemblyLimits, ToolFragment, validate_schema,
		},
	},
};

/// Route-scoped conservative normalization for provider discovery rows.
pub trait DiscoveryProjector: Send + Sync + 'static {
	/// Normalizes one provider page without mutating the bundled catalog.
	fn project(
		&self,
		request: &crate::call::DiscoveryRequest,
		rows: Vec<omp_llm_catalog::DiscoveredModel>,
		next_cursor: Option<omp_core::Str>,
	) -> Result<ModelDiscoveryPage, Error>;
}

/// Applies catalog-selected deterministic recovery before semantic validation.
#[derive(Clone, Default)]
pub struct RecoveryLayer {
	discovery: Option<Arc<dyn DiscoveryProjector>>,
}
impl RecoveryLayer {
	/// Creates recovery with the exact route-scoped discovery projector.
	pub fn new(discovery: Arc<dyn DiscoveryProjector>) -> Self {
		Self { discovery: Some(discovery) }
	}

	/// Creates recovery for a route that does not advertise runtime discovery.
	pub const fn without_discovery() -> Self {
		Self { discovery: None }
	}
}
impl fmt::Debug for RecoveryLayer {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter
			.debug_struct("RecoveryLayer")
			.field("discovery", &self.discovery.is_some())
			.finish()
	}
}

/// Response-recovery service retaining route-scoped immutable projectors.
#[derive(Clone)]
pub struct RecoveryService<S> {
	inner:     S,
	discovery: Option<Arc<dyn DiscoveryProjector>>,
}

impl<S> Layer<S> for RecoveryLayer {
	type Service = RecoveryService<S>;

	fn layer(&self, inner: S) -> Self::Service {
		RecoveryService { inner, discovery: self.discovery.clone() }
	}
}

impl<S> Service<LayerCall<Call>> for RecoveryService<S>
where
	S: Service<LayerCall<Call>, Response = HandshakenResponse, Error = Error> + Clone,
{
	type Error = Error;
	type Response = HandshakenResponse;

	type Future = impl Future<Output = Result<HandshakenResponse, Error>>;

	fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Error>> {
		self.inner.poll_ready(cx)
	}

	fn call(&mut self, request: LayerCall<Call>) -> Self::Future {
		let replacement = self.inner.clone();
		let mut inner = mem::replace(&mut self.inner, replacement);
		let context = request.context.clone();
		let tools = match &request.payload.operation {
			OperationCall::Chat(chat) => chat.tools.clone(),
			OperationCall::Realtime(realtime) => realtime.tools.clone(),
			_ => Default::default(),
		};
		let structured = match &request.payload.operation {
			OperationCall::Chat(chat) => match &chat.output {
				Setting::Require(output) | Setting::Prefer(output) => Some(output.clone()),
				Setting::Unset => None,
			},
			_ => None,
		};
		let discovery_request = match &request.payload.operation {
			OperationCall::DiscoverModels(request) => Some(request.clone()),
			_ => None,
		};
		let discovery = self.discovery.clone();
		let plan = request.payload.execution.clone();
		async move {
			context.checkpoint(ErrorPhase::Recovery)?;
			let mut response = inner.call(request).await?;
			let evidence = response.body.clone();
			let handshake = response.meta.clone();
			if response.events.is_some() && response.realtime.is_some() {
				return Err(recovery_error("response.events-and-realtime-conflict", &context));
			}
			let Some(mut input) = response.events.take() else {
				if response.realtime.is_some() {
					return Ok(response);
				}
				return Err(recovery_error("response.missing-events-and-realtime", &context));
			};
			let output_context = context.clone();
			let empty_policy = plan
				.as_ref()
				.and_then(|plan| plan.policy_model.as_ref())
				.map(|model| model.wire_policy.clone());
			let mut structured_index = None;
			let mut json = structured.as_ref().and_then(|output| {
				empty_policy.clone().map(|policy| {
					let enforcement = match output {
						StructuredOutput::JsonSchema { strict: true, .. } => JsonEnforcement::Strict,
						_ => JsonEnforcement::NativeOrRepair,
					};
					JsonRepairStage::new(
						enforcement,
						JsonRepairLimits::default(),
						policy,
						output_context.attempts().saturating_sub(1),
					)
				})
			});
			let guard_reasoning = plan
				.as_ref()
				.is_some_and(|plan| plan.wire_policy.reasoning.loop_guard == Some(true));
			response.events = Some(Box::pin(async_stream::stream! {
				let mut completion: Option<RawCompletion> = None;
				let mut empty = empty_policy.map(|policy| EmptyCompletionStage::new(policy, output_context.attempts().saturating_sub(1)));
				let mut reasoning_guard = guard_reasoning.then(|| ReasoningStallGuard::new(ReasoningLimits::default()));
				while let Some(item) = input.next().await {
					if let Err(error) = output_context.checkpoint(ErrorPhase::Recovery) {
						output_context.cancel();
						yield Err(error);
						return;
					}
					match item {
						Err(mut error) => {
							output_context.finalize_error(&mut error);
							error.committed = output_context.is_committed();
							yield Err(error);
							return;
						}
						Ok(RawEvent::Completion(terminal)) => {
							if completion.replace(terminal).is_some() {
								yield Err(recovery_error("response.duplicate-completion", &output_context));
								return;
							}
						}
						Ok(RawEvent::ToolCallComplete { index, call }) => {
							let event = match recover_tool(index, call, &tools, &output_context) {
								Ok(event) => event,
								Err(error) => { yield Err(error); return; },
							};
							if let Err(error) = observe_reasoning(&mut reasoning_guard, &event, &output_context) {
								yield Err(error);
								return;
							}
							match observe_empty(&mut empty, event, &output_context) {
								Ok(event) => yield Ok(RawEvent::Chat(event)),
								Err(error) => { yield Err(error); return; },
							}
						}
						Ok(RawEvent::ProviderState(state)) => output_context.stage_provider_state(state),
						Ok(RawEvent::Metadata(metadata)) => output_context.observe_provider_metadata(metadata),
						Ok(RawEvent::Telemetry(telemetry)) => output_context.observe_provider_telemetry(telemetry),
						Ok(RawEvent::Failure(mut error)) => {
							output_context.finalize_error(&mut error);
							error.committed = output_context.is_committed();
							yield Err(error);
							return;
						}
						Ok(RawEvent::Chat(ChatEvent::Completed(_))) => {
							yield Err(recovery_error("response.public-completion-before-finalization", &output_context));
							return;
						}
						Ok(RawEvent::Chat(ChatEvent::TextDelta { index, text })) => {
							let event = ChatEvent::TextDelta { index, text: text.clone() };
							if let Err(error) = observe_reasoning(&mut reasoning_guard, &event, &output_context) {
								yield Err(error);
								return;
							}
							let event = match observe_empty(&mut empty, event, &output_context) {
								Ok(event) => event,
								Err(error) => { yield Err(error); return; },
							};
							if let Some(stage) = json.as_mut() {
								structured_index.get_or_insert(index);
								if let Err(_) = stage.push(bytes::Bytes::copy_from_slice(text.as_bytes()), &mut |_| {}) {
									yield Err(structured_error("structured-output.repair-input", &output_context));
									return;
								}
							} else {
								yield Ok(RawEvent::Chat(event));
							}
						}
						Ok(RawEvent::Chat(event)) => {
							if let Err(error) = observe_reasoning(&mut reasoning_guard, &event, &output_context) {
								yield Err(error);
								return;
							}
							match observe_empty(&mut empty, event, &output_context) {
								Ok(event) => yield Ok(RawEvent::Chat(event)),
								Err(error) => { yield Err(error); return; },
							}
						}
						Ok(RawEvent::ImageGeneration(event)) => yield Ok(RawEvent::ImageGeneration(event)),
						Ok(RawEvent::VideoGeneration(event)) => yield Ok(RawEvent::VideoGeneration(event)),
						Ok(RawEvent::Audio(chunk)) => yield Ok(RawEvent::Audio(chunk)),
						Ok(RawEvent::Transcript(event)) => yield Ok(RawEvent::Transcript(event)),
						Ok(RawEvent::Answer(body)) => yield Ok(RawEvent::Answer(body)),
						Ok(RawEvent::Control(control)) => yield Ok(RawEvent::Control(control)),
						Ok(RawEvent::NativeChunk(bytes)) => yield Ok(RawEvent::NativeChunk(bytes)),
						Ok(RawEvent::DiscoveredModels { rows, next_cursor }) => {
							match project_discovery(
								discovery.as_ref(),
								discovery_request.as_deref(),
								rows,
								next_cursor,
								&output_context,
							) {
								Ok(event) => yield Ok(event),
								Err(error) => { yield Err(error); return; },
							}
						},
					}
				}
				if let Some(terminal) = completion {
					let finalized = match finalize_completion(terminal, plan.as_deref(), &handshake, evidence.evidence(), &output_context) {
						Ok(event) => event,
						Err(error) => { yield Err(error); return; },
					};
					if let Err(error) = finish_empty(&mut empty, &output_context) {
						yield Err(error);
						return;
					}
					if let Some(output) = structured.as_ref() {
						let repaired = match finish_json(&mut json, &output_context) {
							Ok(text) => text,
							Err(error) => { yield Err(error); return; },
						};
						if let Err(error) = validate_structured_output(output, &repaired, &output_context) {
							yield Err(error);
							return;
						}
						output_context.mark_structured_output_valid();
						yield Ok(RawEvent::Chat(ChatEvent::TextDelta {
							index: structured_index.unwrap_or(0),

							text: repaired.into(),
						}));
					}
					yield Ok(RawEvent::Chat(finalized));
				}
			}));
			Ok(response)
		}
	}
}
fn project_discovery(
	projector: Option<&Arc<dyn DiscoveryProjector>>,
	request: Option<&crate::call::DiscoveryRequest>,
	rows: Vec<omp_llm_catalog::DiscoveredModel>,
	next_cursor: Option<omp_core::Str>,
	context: &crate::layer::ExecutionContext,
) -> Result<RawEvent, Error> {
	let result = match (projector, request) {
		(Some(projector), Some(request)) => projector
			.project(request, rows, next_cursor)
			.map(|page| RawEvent::Answer(AnswerBody::Models(page))),
		(None, _) => Err(recovery_error("discovery.projector-missing", context)),
		(Some(_), None) => Err(recovery_error("discovery.request-missing", context)),
	};
	result.map_err(|mut error| {
		context.finalize_error(&mut error);
		error
	})
}
fn observe_reasoning(
	guard: &mut Option<ReasoningStallGuard>,
	event: &ChatEvent,
	context: &crate::layer::ExecutionContext,
) -> Result<(), Error> {
	let Some(guard) = guard.as_mut() else {
		return Ok(());
	};
	let observation = match event {
		ChatEvent::ThinkingDelta { text, .. } => ReasoningObservation {
			delta:             text,
			semantic_progress: false,
			visibility:        OutputVisibility::Gated,
		},
		ChatEvent::TextDelta { .. }
		| ChatEvent::ToolCallReady { .. }
		| ChatEvent::Artifact { .. } => ReasoningObservation {
			delta:             "",
			semantic_progress: true,
			visibility:        OutputVisibility::Gated,
		},
		_ => return Ok(()),
	};
	let Some(signal) = guard.observe(observation) else {
		return Ok(());
	};
	context.with_receipt(|receipt| {
		receipt
			.recoveries
			.push(recovery_record(context.attempts().saturating_sub(1), &signal))
	});
	let mut error = Error::new(
		ErrorKind::RepeatedReasoning,
		ErrorPhase::Recovery,
		RetryAction::SemanticRetry,
		context.receipt(),
	);
	error.committed = context.is_committed();
	error.detail =
		Some(ErrorDetail::Protocol { reason: ReasonId("reasoning.loop-detected".into()) });
	Err(error)
}

fn observe_empty(
	stage: &mut Option<EmptyCompletionStage>,
	event: ChatEvent,
	context: &crate::layer::ExecutionContext,
) -> Result<ChatEvent, Error> {
	let Some(stage) = stage.as_mut() else {
		return Ok(event);
	};
	let mut output = None;
	stage
		.push(EmptyInput::Event(event), &mut |event| output = Some(event))
		.map_err(|_| recovery_error("empty-completion.observer", context))?;
	match output {
		Some(EmptyEvent::Event(event)) => Ok(event),
		Some(EmptyEvent::Empty(_)) | None => {
			Err(recovery_error("empty-completion.invalid-observer-output", context))
		},
	}
}

fn finish_empty(
	stage: &mut Option<EmptyCompletionStage>,
	context: &crate::layer::ExecutionContext,
) -> Result<(), Error> {
	let Some(stage) = stage.as_mut() else {
		return Ok(());
	};
	let mut empty = None;
	stage
		.push(EmptyInput::Completed, &mut |event| {
			if let EmptyEvent::Empty(classification) = event {
				empty = Some(classification);
			}
		})
		.map_err(|_| recovery_error("empty-completion.classification", context))?;
	stage
		.finish(&mut |_| {})
		.map_err(|_| recovery_error("empty-completion.finish", context))?;
	let Some(classification) = empty else {
		return Ok(());
	};
	context.with_receipt(|receipt| receipt.recoveries.push(classification.recovery));
	let mut error = Error::new(
		ErrorKind::EmptyCompletion,
		ErrorPhase::Recovery,
		RetryAction::SemanticRetry,
		context.receipt(),
	);
	error.committed = context.is_committed();
	error.detail =
		Some(ErrorDetail::Protocol { reason: ReasonId("empty-completion.classified".into()) });
	Err(error)
}

fn finish_json(
	stage: &mut Option<JsonRepairStage>,
	context: &crate::layer::ExecutionContext,
) -> Result<String, Error> {
	let Some(stage) = stage.as_mut() else {
		return Err(structured_error("structured-output.repair-policy-missing", context));
	};
	let mut document = None;
	stage
		.finish(&mut |value| document = Some(value))
		.map_err(|_| structured_error("structured-output.invalid-json", context))?;
	let document =
		document.ok_or_else(|| structured_error("structured-output.missing-document", context))?;
	if let Some(recovery) = document.recovery {
		context.with_receipt(|receipt| receipt.recoveries.push(recovery));
	}
	String::from_utf8(document.bytes.to_vec())
		.map_err(|_| structured_error("structured-output.invalid-utf8", context))
}

fn validate_structured_output(
	output: &StructuredOutput,
	text: &str,
	context: &crate::layer::ExecutionContext,
) -> Result<(), Error> {
	let value = serde_json::from_str::<serde_json::Value>(text)
		.map_err(|_| structured_error("structured-output.invalid-json", context))?;
	match output {
		StructuredOutput::JsonObject if value.is_object() => Ok(()),
		StructuredOutput::JsonObject => {
			Err(structured_error("structured-output.not-object", context))
		},
		StructuredOutput::JsonSchema { schema, strict, .. } => {
			validate_schema(schema.as_value(), &value, *strict, ToolAssemblyLimits::default())
				.map_err(|_| structured_error("structured-output.schema-violation", context))
		},
		StructuredOutput::Regex(_) | StructuredOutput::Lark(_) | StructuredOutput::Ebnf(_) => {
			Err(structured_error("structured-output.validator-unavailable", context))
		},
	}
}

fn structured_error(reason: &'static str, context: &crate::layer::ExecutionContext) -> Error {
	let mut error = Error::new(
		ErrorKind::StructuredOutputFailure,
		ErrorPhase::Recovery,
		RetryAction::SemanticRetry,
		context.receipt(),
	);
	error.committed = context.is_committed();
	error.detail = Some(ErrorDetail::Protocol { reason: ReasonId(reason.into()) });
	error
}
fn finalize_completion(
	terminal: RawCompletion,
	plan: Option<&crate::plan::ExecutionPlan>,
	handshake: &crate::codec::HandshakeMeta,
	body: crate::body::AttemptBodyEvidence,
	context: &crate::layer::ExecutionContext,
) -> Result<ChatEvent, Error> {
	let plan = plan.ok_or_else(|| recovery_error("completion.missing-execution-plan", context))?;
	let model = plan
		.policy_model
		.as_ref()
		.ok_or_else(|| recovery_error("completion.missing-pricing-model", context))?;
	if model
		.pricing
		.components
		.iter()
		.any(|component| component.unit == PriceUnit::McharInput)
		|| model.pricing.tiers.iter().any(|tier| {
			tier
				.components
				.iter()
				.any(|component| component.unit == PriceUnit::McharInput)
		}) {
		return Err(recovery_error("completion.character-usage-unavailable", context));
	}
	let usage = terminal.usage;
	let dimensions = UsageDimensions {
		input_tokens:       usage.input_tokens,
		output_tokens:      usage.output_tokens.saturating_add(usage.reasoning_tokens),
		cache_read_tokens:  usage.cache_read_tokens,
		cache_write_tokens: usage.cache_write_tokens,
		images:             u64::from(usage.images),
		video_seconds:      usage.video_ms.div_ceil(1_000),
		audio_seconds:      usage
			.audio_input_ms
			.saturating_add(usage.audio_output_ms)
			.div_ceil(1_000),
		input_characters:   0,
		requests:           1,
	};
	let nanos = model
		.pricing
		.cost(dimensions)
		.map_err(|_| recovery_error("completion.pricing-overflow", context))?
		.as_nanos();
	let micro_usd = i128::from(nanos.div_ceil(1_000));
	let cost = Cost::from_micro_usd(micro_usd);
	context.charge_tokens(
		usage.input_tokens,
		usage.output_tokens.saturating_add(usage.reasoning_tokens),
	)?;
	context.charge_cost(cost)?;
	let index = context.attempts().saturating_sub(1);
	let routing = context.account_routing().unwrap_or_default();
	context.with_receipt(|receipt| {
		if !receipt
			.attempts
			.iter()
			.any(|attempt| attempt.index == index)
		{
			receipt.record_attempt(AttemptReceipt {
				index,
				hidden: false,
				provider: Some(plan.provider.clone()),
				route: Some(plan.route.clone()),
				account: routing.account,
				principal: routing.principal,
				body,
				outcome: AttemptOutcome::Succeeded,
				usage,
				cost,
				provider_evidence: ProviderEvidence {
					request_id: handshake
						.provider_request_id
						.clone()
						.or_else(|| context.provider_response_id()),
					status:     handshake.status,
					code:       None,
					summary:    None,
				},
				elapsed: context.attempt_elapsed(index),
			});
		}
		receipt.timings.total = context.elapsed();
		receipt.timings.completed_at = Some(SystemTime::now());
	});
	let receipt = context.receipt();
	Ok(ChatEvent::Completed(Completion {
		reason: terminal.reason,
		blocks: terminal.blocks,
		usage,
		receipt,
	}))
}

fn recover_tool(
	index: u32,
	call: UnvalidatedToolCall,
	definitions: &[ToolDefinition],
	context: &crate::layer::ExecutionContext,
) -> Result<ChatEvent, Error> {
	if call.input_kind != ToolInputKind::Json {
		return Err(recovery_error("tool.freeform-not-declared", context));
	}
	let mut assembler = ToolAssembler::new(
		definitions,
		ToolAssemblyLimits::default(),
		context.attempts().saturating_sub(1),
	);
	let mut ready = None;
	for fragment in [
		ToolFragment::Start {
			source_index: index,
			id:           Some(call.id),
			name:         bytes::Bytes::copy_from_slice(call.name.as_bytes()),
		},
		ToolFragment::ArgumentsDelta { source_index: index, bytes: call.arguments },
		ToolFragment::End { source_index: index },
	] {
		for event in assembler.push(fragment) {
			match event {
				ToolAssemblyEvent::Ready { call, .. } => ready = Some(call),
				ToolAssemblyEvent::Rejected { .. } => {
					context.with_receipt(|receipt| receipt.recoveries.extend(assembler.take_evidence()));
					return Err(recovery_error("tool.assembly-rejected", context));
				},
				ToolAssemblyEvent::Started { .. } | ToolAssemblyEvent::ArgumentsDelta { .. } => {},
			}
		}
	}
	context.with_receipt(|receipt| receipt.recoveries.extend(assembler.take_evidence()));
	let call = ready.ok_or_else(|| recovery_error("tool.assembly-incomplete", context))?;
	Ok(ChatEvent::ToolCallReady { index, call })
}

fn recovery_error(reason: &'static str, context: &crate::layer::ExecutionContext) -> Error {
	let mut error = Error::new(
		ErrorKind::MalformedModelOutput,
		ErrorPhase::Recovery,
		RetryAction::SemanticRetry,
		context.receipt(),
	);
	error.committed = context.is_committed();
	error.detail = Some(ErrorDetail::Protocol { reason: ReasonId(reason.into()) });
	error
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{
		call::DiscoveryRequest,
		layer::{ExecutionBudget, ExecutionContext},
	};

	struct TestProjector {
		fail: bool,
	}

	impl DiscoveryProjector for TestProjector {
		fn project(
			&self,
			_: &DiscoveryRequest,
			_: Vec<omp_llm_catalog::DiscoveredModel>,
			next_cursor: Option<omp_core::Str>,
		) -> Result<ModelDiscoveryPage, Error> {
			if self.fail {
				return Err(Error::new(
					ErrorKind::MalformedModelOutput,
					ErrorPhase::Recovery,
					RetryAction::Never,
					Default::default(),
				));
			}
			Ok(ModelDiscoveryPage { models: Vec::new(), next_cursor })
		}
	}

	fn request() -> DiscoveryRequest {
		DiscoveryRequest {
			provider:  None,
			route:     None,
			cursor:    None,
			page_size: 10,
			operation: None,
		}
	}

	#[test]
	fn discovery_page_is_emitted_only_after_projection_succeeds() {
		let context = ExecutionContext::new(ExecutionBudget::default());
		let projector: Arc<dyn DiscoveryProjector> = Arc::new(TestProjector { fail: false });
		let projected = project_discovery(
			Some(&projector),
			Some(&request()),
			Vec::new(),
			Some("next".into()),
			&context,
		);
		assert!(matches!(
			projected,
			Ok(RawEvent::Answer(AnswerBody::Models(ModelDiscoveryPage { next_cursor: Some(cursor), .. })))
				if cursor.as_str() == "next"
		));
	}

	#[test]
	fn corrupt_or_unconfigured_discovery_page_is_terminal_without_output() {
		let context = ExecutionContext::new(ExecutionBudget::default());
		let corrupt: Arc<dyn DiscoveryProjector> = Arc::new(TestProjector { fail: true });
		assert!(
			project_discovery(Some(&corrupt), Some(&request()), Vec::new(), None, &context).is_err()
		);
		assert!(project_discovery(None, Some(&request()), Vec::new(), None, &context).is_err());
		assert!(project_discovery(Some(&corrupt), None, Vec::new(), None, &context).is_err());
	}
}
