//! Expiring provider/model/account capability fallback.
//!
//! A learned rejection is isolated to the selected provider account and model.
//! Facts expire explicitly so a provider deployment or entitlement change is
//! eventually probed again instead of becoming a process-lifetime downgrade.

use std::{
	collections::HashMap,
	fmt,
	future::{Ready, ready},
	sync::Arc,
	task::{Context, Poll},
	time::{Duration, Instant},
};

use futures::{Stream, StreamExt};
use omp_core::{Str, SparseSet};
use omp_llm_error::{Classification, Feature, Kind};
use omp_proto::{
	inference::v1::{
		Attempt, Effort, ServiceTier, TurnError, TurnEvent, TurnRequest, Value, response_format,
		tool_choice, turn_error, turn_event, turn_request, value,
	},
	thread::v1::{Item, Message, Part, Role, item, part},
};
use parking_lot::Mutex;
use tower::{Layer, Service, ServiceExt};

use crate::{envelope::TurnRequestEnvelope, recovery::classify_turn_error};

const DEFAULT_LEARN_EXPIRY: Duration = Duration::from_secs(6 * 60 * 60);

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct LearningScope {
	provider:  Str,
	model:     Str,
	account:   Option<u64>,
	namespace: Option<Str>,
}

#[derive(Clone)]
struct LearnedFailure {
	classification: Classification,
	expires_at:     Instant,
}

type LearnedSet = Arc<Mutex<HashMap<(LearningScope, u8), LearnedFailure>>>;

/// Derives an optional route namespace for a request.
///
/// Provider, model, and account are always included by the middleware. This
/// hook adds endpoint or region identity when one provider id fronts several
/// independently deployed routes.
pub type ScopeFn = Arc<dyn Fn(&TurnRequest) -> Option<Str> + Send + Sync>;

/// Coordinator hook that removes one rejected feature from a request.
///
/// Implementations may override canonical feature mutation. The middleware
/// persists the classified failure at provider/model/account scope.
pub trait RequestRepair: Send + Sync + 'static {
	/// Returns a request with `feature` removed or adjusted.
	///
	/// `None` means that the feature is not present or cannot be stripped from
	/// this request. `cls` carries details such as allowed reasoning efforts.
	fn strip(
		&self,
		req: &TurnRequest,
		feature: Feature,
		cls: &Classification,
	) -> Option<TurnRequest>;
}

/// [`Layer`] producing capability-learning [`Learn`] services.
#[derive(Clone)]
pub struct LearnLayer {
	repair:  Arc<dyn RequestRepair>,
	learned: LearnedSet,
	scope:   Option<ScopeFn>,
	expiry:  Duration,
}

impl LearnLayer {
	/// Creates a layer whose learned facts use provider/model/account scope and
	/// expire after six hours.
	pub fn new(repair: Arc<dyn RequestRepair>) -> Self {
		Self {
			repair,
			learned: Arc::new(Mutex::new(HashMap::new())),
			scope: None,
			expiry: DEFAULT_LEARN_EXPIRY,
		}
	}

	/// Replaces the learning scope; see [`ScopeFn`] for when this is
	/// REQUIRED rather than optional.
	#[must_use]
	pub fn with_scope(mut self, scope: ScopeFn) -> Self {
		self.scope = Some(scope);
		self
	}

	/// Sets the lifetime of learned failures. Zero disables persistence while
	/// retaining same-turn bounded repair.
	#[must_use]
	pub fn with_expiry(mut self, expiry: Duration) -> Self {
		self.expiry = expiry;
		self
	}
}

impl<S> Layer<S> for LearnLayer {
	type Service = Learn<S>;

	fn layer(&self, inner: S) -> Self::Service {
		Learn {
			inner,
			repair: Arc::clone(&self.repair),
			learned: Arc::clone(&self.learned),
			scope: self.scope.clone(),
			expiry: self.expiry,
		}
	}
}

/// Capability-learning wrapper around an inference turn service.
#[derive(Clone)]
pub struct Learn<S> {
	inner:   S,
	repair:  Arc<dyn RequestRepair>,
	learned: LearnedSet,
	scope:   Option<ScopeFn>,
	expiry:  Duration,
}

impl<S> Learn<S> {
	/// Wraps `inner` with a fresh expiring provider/model/account capability
	/// store.
	pub fn new(inner: S, repair: Arc<dyn RequestRepair>) -> Self {
		Self {
			inner,
			repair,
			learned: Arc::new(Mutex::new(HashMap::new())),
			scope: None,
			expiry: DEFAULT_LEARN_EXPIRY,
		}
	}
}

impl<S, St, R> Service<R> for Learn<S>
where
	R: TurnRequestEnvelope,
	S: Service<R, Response = St> + Clone + Send + 'static,
	S::Future: Send,
	S::Error: fmt::Display + Send + 'static,
	St: Stream<Item = TurnEvent> + Send + 'static,
{
	type Error = S::Error;
	type Future = Ready<Result<Self::Response, S::Error>>;
	type Response = LearnStream<S, St, R>;

	fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), S::Error>> {
		Poll::Ready(Ok(()))
	}

	fn call(&mut self, req: R) -> Self::Future {
		let clone = self.inner.clone();
		let inner = std::mem::replace(&mut self.inner, clone);
		let repair = Arc::clone(&self.repair);
		let learned = Arc::clone(&self.learned);
		let scope_fn = self.scope.clone();
		let expiry = self.expiry;
		ready(Ok(learn_stream(inner, req, scope_fn, repair, learned, expiry)))
	}
}

/// Concrete capability-learning stream.
///
/// One heap-pinned generator per call: the single allocation keeps this
/// layer's state behind a pointer, so composed stacks stay flat. Fully
/// inline generator nesting embeds every inner layer's state in the
/// parent's and was measured to overflow the thread stack at this
/// composition depth; a hand-written pin-projected state machine is the
/// box-free replacement if this layer ever gets hot. Erase to a boxed-dyn
/// stream only
/// at the outer boundary.
pub type LearnStream<
	S: Service<R, Response = St> + Send + 'static,
	St: Stream<Item = TurnEvent> + Send + 'static,
	R: TurnRequestEnvelope,
>
	= impl Stream<Item = TurnEvent> + Send + Unpin
where
	S::Future: Send,
	S::Error: fmt::Display + Send + 'static;

#[define_opaque(LearnStream)]
fn learn_stream<S, St, R>(
	svc: S,
	req: R,
	scope_fn: Option<ScopeFn>,
	repair: Arc<dyn RequestRepair>,
	learned: LearnedSet,
	expiry: Duration,
) -> LearnStream<S, St, R>
where
	R: TurnRequestEnvelope,
	S: Service<R, Response = St> + Send + 'static,
	S::Future: Send,
	S::Error: fmt::Display + Send + 'static,
	St: Stream<Item = TurnEvent> + Send + 'static,
{
	Box::pin(async_stream::stream! {
		let mut svc = svc;
		let scope = learning_scope(&req, scope_fn.as_ref());
		let mut req = req;
		let mut repaired = apply_learned(req.request_mut(), scope.as_ref(), &repair, &learned);
		let first = match svc.ready().await {
			Ok(svc) => match svc.call(req.clone()).await {
				Ok(stream) => stream,
				Err(error) => {
					yield service_error(&error);
					return;
				},
			},
			Err(error) => {
				yield service_error(&error);
				return;
			},
		};
		let mut current = std::pin::pin!(first);
		let mut dispatch: u32 = 1;
		let mut saw_output = false;
		let mut invoked = false;
		loop {
			let Some(event) = current.next().await else {
				return;
			};
			let err = match event.event {
				Some(turn_event::Event::Error(err)) => {
					err
				},
				Some(turn_event::Event::Outcome(_)) => {
					yield event;
					return;
				},
				Some(
					turn_event::Event::PartStart(_)
					| turn_event::Event::PartDelta(_)
					| turn_event::Event::PartEnd(_),
				) => {
					saw_output = true;
					yield event;
					continue;
				},
				Some(turn_event::Event::Invoke(_) | turn_event::Event::InvokeCancel(_)) => {
					invoked = true;
					yield event;
					continue;
				},
				_ => {
					yield event;
					continue;
				},
			};

			let cls = classify_turn_error(&err);
			let Some(key) = repair_key(&cls) else {
				yield TurnEvent { event: Some(turn_event::Event::Error(err)) };
				return;
			};
			if let Some(scope) = &scope
				&& !expiry.is_zero()
				&& let Some(expires_at) = Instant::now().checked_add(expiry)
			{
				learned.lock().insert(
					(scope.clone(), key),
					LearnedFailure { classification: cls.clone(), expires_at },
				);
			}


			let replay_safe = !saw_output && !invoked;
			if !replay_safe || !repaired.insert(key) {
				yield TurnEvent { event: Some(turn_event::Event::Error(err)) };
				return;
			}
			let Some(next_request) = repair_classified(req.request(), &cls, repair.as_ref()) else {
				yield TurnEvent { event: Some(turn_event::Event::Error(err)) };
				return;
			};


			*req.request_mut() = next_request;
			let redispatch = match svc.ready().await {
				Ok(svc) => svc.call(req.clone()).await,
				Err(error) => Err(error),
			};
			let Ok(next) = redispatch else {
				yield TurnEvent { event: Some(turn_event::Event::Error(err)) };
				return;
			};
			current.set(next);
			dispatch += 1;
			saw_output = false;
			yield TurnEvent {
				event: Some(turn_event::Event::Attempt(Attempt {
					number: dispatch,
					reason: repair_reason(&cls).to_owned(),
				})),
			};
		}
	})
}

fn service_error(error: &impl fmt::Display) -> TurnEvent {
	TurnEvent {
		event: Some(turn_event::Event::Error(TurnError {
			kind: turn_error::Kind::Upstream as i32,
			detail: error.to_string(),
			..TurnError::default()
		})),
	}
}

fn learning_scope<R: TurnRequestEnvelope>(
	req: &R,
	namespace: Option<&ScopeFn>,
) -> Option<LearningScope> {
	let namespace = namespace.and_then(|scope| scope(req.request()));
	let model = model_of(req.request())?;
	let (provider, account) = req.learning_identity().map_or_else(
		|| (Str::new("unknown"), None),
		|(provider, account)| (Str::new(provider), Some(account)),
	);
	Some(LearningScope { provider, model, account, namespace })
}

fn model_of(req: &TurnRequest) -> Option<Str> {
	req.params
		.as_ref()
		.map(|params| Str::from(params.model.as_str()))
}

fn apply_learned(
	req: &mut TurnRequest,
	scope: Option<&LearningScope>,
	repair: &Arc<dyn RequestRepair>,
	learned: &LearnedSet,
) -> SparseSet<u8> {
	let Some(scope) = scope else {
		return SparseSet::new();
	};
	let now = Instant::now();
	let mut entries = learned.lock();
	entries.retain(|_, failure| failure.expires_at > now);
	let mut repairs: Vec<_> = entries
		.iter()
		.filter_map(|((learned_scope, key), failure)| {
			(learned_scope == scope).then(|| (*key, failure.classification.clone()))
		})
		.collect();
	drop(entries);
	repairs.sort_unstable_by_key(|(key, _)| *key);

	let mut repaired = SparseSet::new();
	for (key, classification) in repairs {
		if let Some(next) = repair_classified(req, &classification, repair.as_ref()) {
			*req = next;
			repaired.insert(key);
		}
	}
	repaired
}

fn repair_classified(
	req: &TurnRequest,
	classification: &Classification,
	repair: &dyn RequestRepair,
) -> Option<TurnRequest> {
	if let Some(feature) = classification
		.feature
		.filter(|_| classification.kinds.has(Kind::FeatureUnsupported))
	{
		return repair
			.strip(req, feature, classification)
			.or_else(|| canonical_feature_repair(req, feature, classification));
	}
	(classification.kinds.has(Kind::MalformedToolCall)
		&& classification.kinds.has(Kind::Deterministic))
	.then(|| llama_tool_parse_rewrite(req))
	.flatten()
}

fn canonical_feature_repair(
	req: &TurnRequest,
	feature: Feature,
	classification: &Classification,
) -> Option<TurnRequest> {
	let mut repaired = req.clone();
	let params = repaired.params.as_mut()?;
	let changed = match feature {
		Feature::StrictTools => {
			let mut changed = false;
			for tool in &mut params.tools {
				if tool.strict != Some(false) {
					tool.strict = Some(false);
					changed = true;
				}
			}
			let grammar = params
				.response_format
				.as_ref()
				.and_then(|format| format.kind.as_ref())
				.is_some_and(|kind| matches!(kind, response_format::Kind::Grammar(_)));
			if grammar {
				params.response_format = None;
				changed = true;
			} else if let Some(response_format::Kind::JsonSchema(schema)) = params
				.response_format
				.as_mut()
				.and_then(|format| format.kind.as_mut())
				&& schema.strict != Some(false)
			{
				schema.strict = Some(false);
				changed = true;
			}
			changed
		},
		Feature::SamplingParameters => {
			let Some(sampling) = &mut params.sampling else {
				return None;
			};
			let mut changed = sampling.temperature.take().is_some();
			changed |= sampling.top_p.take().is_some();
			changed |= sampling.top_k.take().is_some();
			changed |= sampling.min_p.take().is_some();
			changed |= sampling.frequency_penalty.take().is_some();
			changed |= sampling.presence_penalty.take().is_some();
			changed |= sampling.repetition_penalty.take().is_some();
			changed |= sampling.stop_present.take().is_some();
			changed |= !sampling.stop.is_empty();
			sampling.stop.clear();
			if changed && sampling.max_output_tokens.is_none() {
				params.sampling = None;
			}
			changed
		},
		Feature::StructuredOutputs => params.response_format.take().is_some(),
		Feature::ReasoningEffort => {
			let Some(thinking) = &mut params.thinking else {
				return None;
			};
			let effort = classification
				.allowed_efforts
				.iter()
				.find_map(|value| effort_named(value))
				.unwrap_or(Effort::Unspecified);
			let changed = thinking.effort != effort as i32 || thinking.budget_tokens.is_some();
			thinking.effort = effort as i32;
			thinking.budget_tokens = None;
			changed
		},
		Feature::ToolChoice => {
			let conflict = params.thinking.is_some()
				&& params.tool_choice.as_ref().is_some_and(|choice| {
					matches!(choice.mode(), tool_choice::Mode::Required | tool_choice::Mode::Named)
				});
			if conflict {
				params.thinking = None;
				true
			} else {
				let Some(choice) = &mut params.tool_choice else {
					return None;
				};
				let changed = choice.mode() != tool_choice::Mode::Auto || !choice.name.is_empty();
				choice.set_mode(tool_choice::Mode::Auto);
				choice.name.clear();
				changed
			}
		},
		Feature::FastMode => {
			let mut changed = false;
			if params.service_tier == ServiceTier::Priority as i32 {
				params.service_tier = ServiceTier::Unspecified as i32;
				changed = true;
			}
			if let Some(tiers) = &mut params.service_tier_by_family
				&& tiers.anthropic == ServiceTier::Priority as i32
			{
				tiers.anthropic = ServiceTier::Unspecified as i32;
				changed = true;
			}
			let remove_options = if let Some(options) = &mut params.provider_options {
				let priority = options
					.fields
					.get("anthropic/service_tier")
					.and_then(|value| value.kind.as_ref())
					.is_some_and(
						|kind| matches!(kind, value::Kind::String(value) if value == "priority"),
					);
				if priority {
					options.fields.remove("anthropic/service_tier");
					changed = true;
				}
				priority && options.fields.is_empty()
			} else {
				false
			};
			if remove_options {
				params.provider_options = None;
			}
			changed
		},
		Feature::ThinkingSignature => {
			let disabled = Value { kind: Some(value::Kind::Bool(false)) };
			let options = params.provider_options.get_or_insert_default();
			let changed = options.fields.get("anthropic/replay_unsigned_thinking") != Some(&disabled);
			if changed {
				options
					.fields
					.insert("anthropic/replay_unsigned_thinking".to_owned(), disabled);
			}
			changed
		},
	};
	changed.then_some(repaired)
}

fn llama_tool_parse_rewrite(req: &TurnRequest) -> Option<TurnRequest> {
	let mut repaired = req.clone();
	let instruction = Item {
		kind: Some(item::Kind::Message(Message {
			role:  Role::System as i32,
			parts: vec![Part {
				kind: Some(part::Kind::Text(
					"Tool arguments must be exactly one valid JSON object. Do not wrap them in \
					 markdown or commentary."
						.to_owned(),
				)),
			}],
		})),
		..Item::default()
	};
	match repaired.input.as_mut()? {
		turn_request::Input::Seed(seed) => {
			seed.thread.get_or_insert_default().items.push(instruction)
		},
		turn_request::Input::Incremental(incremental) => incremental
			.delta
			.get_or_insert_default()
			.append
			.push(instruction),
	}
	Some(repaired)
}

fn effort_named(value: &str) -> Option<Effort> {
	match value.to_ascii_lowercase().as_str() {
		"off" | "none" => Some(Effort::Off),
		"minimal" => Some(Effort::Minimal),
		"low" => Some(Effort::Low),
		"medium" => Some(Effort::Medium),
		"high" => Some(Effort::High),
		"xhigh" => Some(Effort::Xhigh),
		"max" => Some(Effort::Max),
		_ => None,
	}
}

fn repair_key(classification: &Classification) -> Option<u8> {
	classification
		.feature
		.filter(|_| classification.kinds.has(Kind::FeatureUnsupported))
		.map(feature_key)
		.or_else(|| {
			(classification.kinds.has(Kind::MalformedToolCall)
				&& classification.kinds.has(Kind::Deterministic))
			.then_some(7)
		})
}

fn repair_reason(classification: &Classification) -> &'static str {
	match classification.feature {
		Some(Feature::StrictTools) => "disabled unsupported StrictTools (grammar downgrade)",
		Some(Feature::StructuredOutputs) => "disabled unsupported StructuredOutputs",
		Some(Feature::ReasoningEffort) => "repaired unsupported ReasoningEffort",
		Some(Feature::ToolChoice) => "resolved unsupported ToolChoice/thinking conflict",
		Some(Feature::SamplingParameters) => "removed unsupported SamplingParameters",
		Some(Feature::FastMode) => "disabled unsupported FastMode",
		Some(Feature::ThinkingSignature) => "removed rejected ThinkingSignature",
		None => "rewrote llama.cpp tool JSON instructions",
	}
}

const fn feature_key(feature: Feature) -> u8 {
	match feature {
		Feature::StrictTools => 0,
		Feature::StructuredOutputs => 1,
		Feature::ReasoningEffort => 2,
		Feature::ToolChoice => 3,
		Feature::SamplingParameters => 6,
		Feature::FastMode => 4,
		Feature::ThinkingSignature => 5,
	}
}
