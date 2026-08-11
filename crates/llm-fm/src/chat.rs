use std::sync::Arc;

use bytes::Bytes;
use futures::{StreamExt as _, stream::BoxStream};
use omp_core::Str;
use omp_llm_types::{
	Accuracy, Chat, ChatOutcome, ChatRequest, Diagnostic, Error, Executor, Fallback, Item, ItemKind,
	Message, Part, Props, Retryability, Role, StopReason, StreamPartKind, TurnError, TurnErrorKind,
	TurnEvent, Unsupported, UnsupportedAction, Usage,
};

use crate::{AppleFm, AppleFmError, AppleFmErrorCode, AppleFmEvent, AppleFmOptions, Result};

/// Pi's fixed instruction for the small on-device model.
pub const APPLE_INTELLIGENCE_SYSTEM_PROMPT: &str =
	"You are a helpful, concise assistant running on-device via Apple Intelligence. Keep responses \
	 focused and answer directly without attempting to invoke tools or access external systems.";
const PROVIDER_ID: &str = "apple-intelligence";

/// Injectable streaming boundary used by the canonical adapter and its tests.
pub trait AppleFmEngine: Send + Sync + 'static {
	/// Starts one native request. Dropping the returned stream must cancel its
	/// work.
	fn stream(&self, options: AppleFmOptions) -> Result<BoxStream<'static, Result<AppleFmEvent>>>;
}

impl AppleFmEngine for AppleFm {
	fn stream(&self, options: AppleFmOptions) -> Result<BoxStream<'static, Result<AppleFmEvent>>> {
		Self::stream(self, options).map(|stream| Box::pin(stream) as BoxStream<'static, _>)
	}
}

/// Canonical chat facet backed by Apple's on-device Foundation Models runtime.
#[derive(Clone)]
pub struct AppleFmChat<E = AppleFm> {
	engine: E,
}

impl AppleFmChat<AppleFm> {
	/// Probes the runtime and constructs an adapter only when generation is
	/// usable.
	pub async fn load() -> Result<Self> {
		AppleFm::load().await.map(Self::new)
	}
}

impl<E> AppleFmChat<E> {
	/// Wraps an engine. Production callers normally use [`Self::load`].
	pub const fn new(engine: E) -> Self {
		Self { engine }
	}
}

#[async_trait::async_trait]
impl<E: AppleFmEngine> Chat for AppleFmChat<E> {
	async fn turn(
		&self,
		request: ChatRequest,
		_executor: Option<Arc<dyn Executor>>,
	) -> std::result::Result<BoxStream<'static, TurnEvent>, Error> {
		let projected = project(request)?;
		let native = self
			.engine
			.stream(projected.options)
			.map_err(|error| Error::Provider(actionable_message(&error)))?;
		Ok(drive(native, projected.model, projected.unsupported))
	}
}

struct Projected {
	options:     AppleFmOptions,
	model:       Str,
	unsupported: Vec<Unsupported>,
}

fn project(request: ChatRequest) -> std::result::Result<Projected, Error> {
	let mut selected = None;
	let mut selected_had_non_text = false;
	for item in request.thread.items.iter().rev() {
		let ItemKind::Message(message) = &item.kind else {
			continue;
		};
		if message.role != Role::User {
			continue;
		}
		let mut prompt = String::new();
		let mut had_non_text = false;
		for part in &message.parts {
			if let Part::Text(text) = part {
				if !prompt.is_empty() {
					prompt.push('\n');
				}
				prompt.push_str(text);
			} else {
				had_non_text = true;
			}
		}
		if !prompt.trim().is_empty() {
			selected = Some(prompt);
			selected_had_non_text = had_non_text;
			break;
		}
	}
	let prompt = selected.ok_or_else(|| {
		Error::Provider(Str::new_static("Apple Intelligence requires a non-empty text prompt."))
	})?;
	let model = request.model.clone();
	let mut unsupported = Vec::new();
	if request.thread.items.len() > 1 {
		unsupported.push(dropped(
			"thread.history",
			"Apple Intelligence uses only the latest non-empty user text message",
		));
	}
	if selected_had_non_text {
		unsupported.push(dropped(
			"thread.parts",
			"Apple Intelligence omits non-text parts from the selected user message",
		));
	}
	if !request.tools.is_empty() {
		unsupported.push(dropped("tools", "Apple Intelligence does not invoke caller tools"));
	}
	if let Some(feature) = request.tool_choice {
		admit_drop(
			feature.on_unsupported,
			"tool_choice",
			"Apple Intelligence does not select tools",
			&mut unsupported,
		)?;
	}
	if let Some(feature) = request.thinking {
		admit_drop(
			feature.on_unsupported,
			"thinking",
			"Apple Intelligence exposes no selectable thinking control",
			&mut unsupported,
		)?;
	}
	if let Some(feature) = request.response_format {
		admit_drop(
			feature.on_unsupported,
			"response_format",
			"Apple Intelligence exposes no structured-output constraint",
			&mut unsupported,
		)?;
	}
	if request.cache.is_some() {
		unsupported.push(dropped("cache", "Apple Intelligence exposes no prompt-cache controls"));
	}
	if request
		.provider_options
		.as_ref()
		.is_some_and(|options| !options.is_empty())
	{
		unsupported.push(dropped(
			"provider_options",
			"Apple Intelligence defines no provider-specific controls",
		));
	}
	for (what, set) in [
		("service_tier", request.service_tier.is_some()),
		("service_tier_by_family", request.service_tier_by_family.is_some()),
		("task_budget", request.task_budget.is_some()),
		("responses_include", request.responses_include.is_some()),
	] {
		if set {
			unsupported.push(dropped(
				what,
				"Apple Intelligence does not consume this remote-provider control",
			));
		}
	}

	let mut options = AppleFmOptions::new(prompt).system_prompt(APPLE_INTELLIGENCE_SYSTEM_PROMPT);
	if let Some(sampling) = request.sampling {
		options.temperature = sampling.temperature;
		if let Some(max) = sampling.max_output_tokens {
			options.max_tokens = Some(u32::try_from(max).unwrap_or_else(|_| {
				unsupported.push(
					Unsupported::builder()
						.what(Str::new_static("sampling.max_output_tokens"))
						.detail(Str::new_static("clamped to the Foundation Models u32 limit"))
						.action(UnsupportedAction::Clamped)
						.build(),
				);
				u32::MAX
			}));
		}
		for (what, set) in [
			("sampling.top_p", sampling.top_p.is_some()),
			("sampling.top_k", sampling.top_k.is_some()),
			("sampling.min_p", sampling.min_p.is_some()),
			("sampling.frequency_penalty", sampling.frequency_penalty.is_some()),
			("sampling.presence_penalty", sampling.presence_penalty.is_some()),
			("sampling.repetition_penalty", sampling.repetition_penalty.is_some()),
			(
				"sampling.stop",
				sampling
					.stop
					.as_ref()
					.is_some_and(|values| !values.is_empty()),
			),
		] {
			if set {
				unsupported.push(dropped(what, "Apple Intelligence exposes no such sampling control"));
			}
		}
	}
	Ok(Projected { options, model, unsupported })
}

fn drive(
	mut native: BoxStream<'static, Result<AppleFmEvent>>,
	model: Str,
	unsupported: Vec<Unsupported>,
) -> BoxStream<'static, TurnEvent> {
	Box::pin(async_stream::stream! {
		yield TurnEvent::Accepted { replay: false };
		let mut part_started = false;
		while let Some(event) = native.next().await {
			match event {
				Ok(AppleFmEvent::Delta(chunk)) => {
					if !part_started {
						yield part_start();
						part_started = true;
					}
					yield TurnEvent::PartDelta { index: 0, chunk: Bytes::copy_from_slice(chunk.as_bytes()) };
				},
				Ok(AppleFmEvent::Finished(generation)) => {
					if !part_started {
						yield part_start();
					}
					yield TurnEvent::PartEnd { index: 0, signature: Bytes::new() };
					yield outcome(generation, model, unsupported);
					return;
				},
				Err(error) => {
					if part_started {
						yield TurnEvent::PartEnd { index: 0, signature: Bytes::new() };
					}
					yield terminal_error(error, model);
					return;
				},
			}
		}
		if part_started {
			yield TurnEvent::PartEnd { index: 0, signature: Bytes::new() };
		}
		yield terminal_error(AppleFmError::runtime("Apple Foundation Models stream ended without a completion result"), model);
	})
}

const fn part_start() -> TurnEvent {
	TurnEvent::PartStart {
		index:        0,
		kind:         StreamPartKind::Text,
		tool_call_id: Str::new_static(""),
		tool_name:    Str::new_static(""),
	}
}

fn outcome(
	generation: crate::AppleFmGeneration,
	model: Str,
	unsupported: Vec<Unsupported>,
) -> TurnEvent {
	let total = u64::from(generation.prompt_tokens_estimated)
		.saturating_add(u64::from(generation.completion_tokens_estimated));
	let usage = Usage::builder()
		.input_tokens(u64::from(generation.prompt_tokens_estimated))
		.output_tokens(u64::from(generation.completion_tokens_estimated))
		.cache_read_tokens(0)
		.cache_write_tokens(0)
		.total_tokens(total)
		.accuracy(Accuracy::Estimated)
		.detail(Props::default())
		.build();
	let output = Item::builder()
		.seq(0)
		.kind(ItemKind::Message(
			Message::builder()
				.role(Role::Assistant)
				.parts(vec![Part::Text(generation.content)])
				.build(),
		))
		.props(Props::default())
		.build();
	TurnEvent::Outcome(
		ChatOutcome::builder()
			.output(vec![output])
			.stop(StopReason::EndTurn)
			.usage(usage)
			.unsupported(unsupported)
			.provider(Str::new_static(PROVIDER_ID))
			.model(model)
			.props(Props::default())
			.build(),
	)
}

fn terminal_error(error: AppleFmError, model: Str) -> TurnEvent {
	let code = error.code();
	let detail = actionable_message(&error);
	let (kind, retryability) = match code {
		AppleFmErrorCode::RateLimited => (TurnErrorKind::RateLimited, Retryability::AfterDelay),
		AppleFmErrorCode::ConcurrentRequests => (TurnErrorKind::Overloaded, Retryability::SameRoute),
		AppleFmErrorCode::UnsupportedGuide | AppleFmErrorCode::UnsupportedLocale => {
			(TurnErrorKind::Unsupported, Retryability::AfterRepair)
		},
		AppleFmErrorCode::Cancelled
		| AppleFmErrorCode::InvalidInput
		| AppleFmErrorCode::ContextOverflow
		| AppleFmErrorCode::GuardrailBlocked
		| AppleFmErrorCode::DeviceNotEligible
		| AppleFmErrorCode::AppleIntelligenceNotEnabled => (TurnErrorKind::Upstream, Retryability::Never),
		AppleFmErrorCode::ModelUnavailable | AppleFmErrorCode::ModelNotReady => {
			(TurnErrorKind::Upstream, Retryability::SameRoute)
		},
		AppleFmErrorCode::TimedOut
		| AppleFmErrorCode::DecodingFailure
		| AppleFmErrorCode::Runtime => (TurnErrorKind::Upstream, Retryability::SameRoute),
	};
	let unsupported = matches!(kind, TurnErrorKind::Unsupported)
		.then(|| vec![dropped("request", detail.as_str())])
		.unwrap_or_default();
	let diagnostic = Diagnostic::builder()
		.provider(Str::new_static(PROVIDER_ID))
		.model(model)
		.attempt(1)
		.code(Str::from(code.to_string()))
		.detail(detail.clone())
		.retryability(retryability)
		.build();
	TurnEvent::Error(
		TurnError::builder()
			.kind(kind)
			.detail(detail)
			.unsupported(unsupported)
			.retry_after_ms(0)
			.diagnostics(vec![diagnostic])
			.build(),
	)
}

fn actionable_message(error: &AppleFmError) -> Str {
	Str::new_static(match error.code() {
		AppleFmErrorCode::Cancelled => "Apple Intelligence request aborted",
		AppleFmErrorCode::GuardrailBlocked => {
			"Apple's safety guardrails blocked this request. Try rephrasing it."
		},
		AppleFmErrorCode::ContextOverflow => {
			"The request exceeds Apple Intelligence's context window. Shorten the prompt."
		},
		AppleFmErrorCode::ModelUnavailable => {
			"The on-device model is unavailable. Enable Apple Intelligence in System Settings → Apple \
			 Intelligence & Siri."
		},
		AppleFmErrorCode::DeviceNotEligible => "This Mac does not support Apple Intelligence.",
		AppleFmErrorCode::AppleIntelligenceNotEnabled => {
			"Enable Apple Intelligence in System Settings → Apple Intelligence & Siri."
		},
		AppleFmErrorCode::ModelNotReady => {
			"The Apple Intelligence model is not ready yet. Wait for its download to finish and retry."
		},
		AppleFmErrorCode::UnsupportedLocale => {
			"The current language or locale is not supported by the on-device model."
		},
		AppleFmErrorCode::RateLimited => {
			"The on-device model is rate limited. Wait a moment and retry."
		},
		AppleFmErrorCode::ConcurrentRequests => {
			"Another on-device generation is already in progress. Wait and retry."
		},
		AppleFmErrorCode::InvalidInput
		| AppleFmErrorCode::TimedOut
		| AppleFmErrorCode::UnsupportedGuide
		| AppleFmErrorCode::DecodingFailure
		| AppleFmErrorCode::Runtime => return Str::from(error.message()),
	})
}

fn admit_drop(
	fallback: Fallback,
	what: &'static str,
	detail: &'static str,
	unsupported: &mut Vec<Unsupported>,
) -> std::result::Result<(), Error> {
	if fallback != Fallback::Ignore {
		return Err(Error::Unsupported(vec![dropped(what, detail)]));
	}
	unsupported.push(dropped(what, detail));
	Ok(())
}

fn dropped(what: &'static str, detail: &str) -> Unsupported {
	Unsupported::builder()
		.what(Str::new_static(what))
		.detail(Str::from(detail))
		.action(UnsupportedAction::Dropped)
		.build()
}

#[cfg(test)]
mod tests {
	use std::{
		collections::VecDeque,
		pin::Pin,
		sync::{
			Arc,
			atomic::{AtomicBool, Ordering},
		},
		task::{Context, Poll},
	};

	use futures::{Stream, stream::BoxStream};
	use omp_llm_types::{
		ChatRequest, Item, ItemKind, Message, Part, Props, Role, Sampling, Thread, TurnErrorKind,
		TurnEvent,
	};
	use parking_lot::Mutex;

	use super::*;

	#[derive(Clone)]
	struct FakeEngine {
		requests: Arc<Mutex<Vec<AppleFmOptions>>>,
		streams:  Arc<Mutex<VecDeque<Vec<Result<AppleFmEvent>>>>>,
		dropped:  Arc<AtomicBool>,
	}

	impl FakeEngine {
		fn new(events: Vec<Result<AppleFmEvent>>) -> Self {
			Self {
				requests: Arc::new(Mutex::new(Vec::new())),
				streams:  Arc::new(Mutex::new(VecDeque::from([events]))),
				dropped:  Arc::new(AtomicBool::new(false)),
			}
		}
	}

	impl AppleFmEngine for FakeEngine {
		fn stream(
			&self,
			options: AppleFmOptions,
		) -> Result<BoxStream<'static, Result<AppleFmEvent>>> {
			self.requests.lock().push(options);
			let events = self.streams.lock().pop_front().unwrap();
			Ok(Box::pin(DropStream {
				inner:   futures::stream::iter(events),
				dropped: Arc::clone(&self.dropped),
			}))
		}
	}

	struct DropStream {
		inner:   futures::stream::Iter<std::vec::IntoIter<Result<AppleFmEvent>>>,
		dropped: Arc<AtomicBool>,
	}

	impl Stream for DropStream {
		type Item = Result<AppleFmEvent>;

		fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
			Pin::new(&mut self.inner).poll_next(cx)
		}
	}

	impl Drop for DropStream {
		fn drop(&mut self) {
			self.dropped.store(true, Ordering::SeqCst);
		}
	}

	fn message(role: Role, parts: Vec<Part>) -> Item {
		Item::builder()
			.seq(0)
			.kind(ItemKind::Message(Message::builder().role(role).parts(parts).build()))
			.props(Props::default())
			.build()
	}

	fn request(items: Vec<Item>) -> ChatRequest {
		ChatRequest::builder()
			.model(Str::new_static("apple-on-device"))
			.thread(Thread::builder().items(items).build())
			.tools(Vec::new())
			.build()
	}

	#[tokio::test]
	async fn projects_latest_non_empty_user_text_and_sampling() {
		let engine = FakeEngine::new(vec![Ok(AppleFmEvent::Finished(crate::AppleFmGeneration {
			content:                     "ok".into(),
			prompt_tokens_estimated:     2,
			completion_tokens_estimated: 1,
			context_size_documented:     4096,
		}))]);
		let chat = AppleFmChat::new(engine.clone());
		let mut input = request(vec![
			message(Role::System, vec![Part::Text("unsafe override".into())]),
			message(Role::User, vec![Part::Text("older".into())]),
			message(Role::User, vec![Part::Text("new".into()), Part::Text("question".into())]),
			message(Role::User, vec![Part::Text("   ".into())]),
		]);
		input.sampling = Some(
			Sampling::builder()
				.temperature(0.25)
				.max_output_tokens(77)
				.build(),
		);
		let events: Vec<_> = chat.turn(input, None).await.unwrap().collect().await;
		let sent = engine.requests.lock();
		assert_eq!(sent[0].prompt.as_str(), "new\nquestion");
		assert_eq!(sent[0].system_prompt.as_deref(), Some(APPLE_INTELLIGENCE_SYSTEM_PROMPT));
		assert_eq!(sent[0].temperature, Some(0.25));
		assert_eq!(sent[0].max_tokens, Some(77));
		assert!(matches!(events.last(), Some(TurnEvent::Outcome(_))));
	}

	#[tokio::test]
	async fn streams_parts_then_one_terminal_outcome_with_estimated_usage() {
		let engine = FakeEngine::new(vec![
			Ok(AppleFmEvent::Delta("hel".into())),
			Ok(AppleFmEvent::Delta("lo".into())),
			Ok(AppleFmEvent::Finished(crate::AppleFmGeneration {
				content:                     "hello".into(),
				prompt_tokens_estimated:     3,
				completion_tokens_estimated: 2,
				context_size_documented:     4096,
			})),
		]);
		let events: Vec<_> = AppleFmChat::new(engine)
			.turn(request(vec![message(Role::User, vec![Part::Text("hi".into())])]), None)
			.await
			.unwrap()
			.collect()
			.await;
		assert!(matches!(events[1], TurnEvent::PartStart { index: 0, .. }));
		assert!(matches!(events[2], TurnEvent::PartDelta { index: 0, .. }));
		assert!(matches!(events[4], TurnEvent::PartEnd { index: 0, .. }));
		let terminals: Vec<_> = events
			.iter()
			.filter(|event| matches!(event, TurnEvent::Outcome(_) | TurnEvent::Error(_)))
			.collect();
		assert_eq!(terminals.len(), 1);
		let TurnEvent::Outcome(outcome) = terminals[0] else {
			panic!("expected outcome")
		};
		let usage = outcome.usage.as_ref().unwrap();
		assert_eq!((usage.input_tokens, usage.output_tokens, usage.total_tokens), (3, 2, Some(5)));
		assert_eq!(usage.accuracy, Accuracy::Estimated);
	}

	#[tokio::test]
	async fn dropping_canonical_stream_cancels_native_stream() {
		let engine = FakeEngine::new(vec![Ok(AppleFmEvent::Delta("partial".into()))]);
		let dropped = Arc::clone(&engine.dropped);
		let stream = AppleFmChat::new(engine)
			.turn(request(vec![message(Role::User, vec![Part::Text("hi".into())])]), None)
			.await
			.unwrap();
		drop(stream);
		assert!(dropped.load(Ordering::SeqCst));
	}

	#[tokio::test]
	async fn maps_native_error_taxonomy_to_terminal_diagnostics() {
		for (code, kind, expected_message) in [
			(
				AppleFmErrorCode::GuardrailBlocked,
				TurnErrorKind::Upstream,
				"Apple's safety guardrails blocked this request. Try rephrasing it.",
			),
			(
				AppleFmErrorCode::RateLimited,
				TurnErrorKind::RateLimited,
				"The on-device model is rate limited. Wait a moment and retry.",
			),
			(
				AppleFmErrorCode::ConcurrentRequests,
				TurnErrorKind::Overloaded,
				"Another on-device generation is already in progress. Wait and retry.",
			),
			(
				AppleFmErrorCode::Cancelled,
				TurnErrorKind::Upstream,
				"Apple Intelligence request aborted",
			),
		] {
			let engine = FakeEngine::new(vec![Err(AppleFmError::new(code, "native detail"))]);
			let events: Vec<_> = AppleFmChat::new(engine)
				.turn(request(vec![message(Role::User, vec![Part::Text("hi".into())])]), None)
				.await
				.unwrap()
				.collect()
				.await;
			let TurnEvent::Error(error) = events.last().unwrap() else {
				panic!("expected terminal error")
			};
			assert_eq!(error.kind, kind);
			assert_eq!(error.detail.as_str(), expected_message);
			assert_eq!(error.diagnostics[0].code.as_str(), code.to_string());
		}
	}
}
