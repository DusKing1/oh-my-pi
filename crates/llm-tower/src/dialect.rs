//! Owned model-prompt dialect middleware for production provider attempts.
//!
//! The layer owns the boundary between canonical requests/events and
//! model-authored in-band syntax. Tool schemas are borrowed only while the
//! prompt and concrete scanner are constructed; streamed payloads remain
//! `Bytes` deltas and scanner output is kept in inline batches.

use std::{
	fmt,
	future::Future,
	pin::Pin,
	sync::Arc,
	task::{Context, Poll},
};

use futures::{Stream, StreamExt};
use omp_core::{Str, StrMut};
use omp_llm_catalog::compat::{Compat, ThinkingToolChoiceConflict};
use omp_llm_dialect::{
	DialectRenderOptions, DialectSelection, InbandTool, ScannerOptions,
	history::project_inband_history,
	projector::{Projection, ProjectionBatch, StreamProjector},
	prompt::write_inband_tool_prompt,
};
use omp_llm_types::{
	Chat, ChatOutcome, ChatRequest, Error as ChatError, Executor, Item, ItemKind, Message, Part,
	Props, Role, StopReason, StreamAccumulator, StreamPartKind, ToolDef, TurnError, TurnErrorKind,
	TurnEvent as NativeTurnEvent,
};
use omp_proto::inference::v1::{Effort, TurnEvent as ProtoTurnEvent, tool_choice, turn_request};
use pin_project_lite::pin_project;
use smallvec::SmallVec;
use tower::{Layer, Service};

use crate::{SingleTurn, envelope::TurnRequestEnvelope, single_turn};

/// Selection and environment inputs captured once for an owned-dialect route.
#[derive(Clone, Debug)]
pub struct OwnedDialectConfig {
	/// Route-selected dialect behavior. `Native` leaves requests and streams
	/// byte-for-byte on their provider-native path.
	pub selection:   DialectSelection,
	/// Captured `OMP_DIALECT` value. When present it overrides `selection` using
	/// the canonical dialect parser.
	pub omp_dialect: Option<Str>,
	/// Catalog compatibility axes used for reasoning/tool-choice conflict
	/// policy.
	pub compat:      Compat,
}

impl OwnedDialectConfig {
	/// Creates deterministic configuration without consulting process state.
	#[must_use]
	pub const fn new(selection: DialectSelection, compat: Compat) -> Self {
		Self { selection, omp_dialect: None, compat }
	}

	/// Adds a captured `OMP_DIALECT` override.
	#[must_use]
	pub fn with_override(mut self, value: Option<Str>) -> Self {
		self.omp_dialect = value;
		self
	}
}

/// Once-built owned-dialect layer.
#[derive(Clone, Debug)]
pub struct OwnedDialectLayer {
	config: OwnedDialectConfig,
}

impl OwnedDialectLayer {
	/// Creates a layer from route-owned configuration.
	#[must_use]
	pub const fn new(config: OwnedDialectConfig) -> Self {
		Self { config }
	}
}

impl<S> Layer<S> for OwnedDialectLayer {
	type Service = OwnedDialect<S>;

	fn layer(&self, inner: S) -> Self::Service {
		OwnedDialect { inner, config: self.config.clone() }
	}
}

/// Service adapting one canonical provider attempt to an owned model dialect.
#[derive(Clone, Debug)]
pub struct OwnedDialect<S> {
	inner:  S,
	config: OwnedDialectConfig,
}

/// Canonical [`Chat`] wrapper for specialized local routes that cannot traverse
/// the protobuf Tower stack.
///
/// Only requests whose trusted model policy declares native tools unavailable
/// enter the owned prompt/history/stream projection. Tool-capable local models
/// remain byte-for-byte on their native path.
#[derive(Clone)]
pub struct OwnedDialectChat {
	inner:            Arc<dyn Chat>,
	config:           OwnedDialectConfig,
	latest_user_only: bool,
}

impl OwnedDialectChat {
	/// Wraps one specialized native chat implementation.
	#[must_use]
	pub fn new(inner: Arc<dyn Chat>, config: OwnedDialectConfig) -> Self {
		Self { inner, config, latest_user_only: false }
	}

	/// Wraps a local implementation that accepts only one final user prompt.
	///
	/// The owned system prompt and projected result history are flattened into
	/// that prompt after projection, as required by Apple Foundation Models.
	#[must_use]
	pub fn latest_user(inner: Arc<dyn Chat>, config: OwnedDialectConfig) -> Self {
		Self { inner, config, latest_user_only: true }
	}
}

#[async_trait::async_trait]
impl Chat for OwnedDialectChat {
	async fn turn(
		&self,
		request: ChatRequest,
		executor: Option<Arc<dyn Executor>>,
	) -> Result<futures::stream::BoxStream<'static, NativeTurnEvent>, ChatError> {
		let native_tools = request
			.model_policy
			.as_deref()
			.and_then(|policy| policy.capabilities.tools);
		let selected = resolve_dialect(request.model.as_str(), &self.config, native_tools)
			.map_err(|error| ChatError::Provider(error.0))?;
		let Some(dialect) = selected else {
			return self.inner.turn(request, executor).await;
		};

		let model_policy = request.model_policy.clone();
		let mut proto = omp_proto::inference::v1::TurnRequest::from(request);
		let prepared = prepare_owned_request(&mut proto, &self.config, dialect)
			.map_err(|error| ChatError::Provider(error.0))?;
		let mut projected = ChatRequest::try_from(proto)
			.map_err(|error| ChatError::Provider(Str::from(error.to_string())))?;
		if self.latest_user_only {
			flatten_to_latest_user(&mut projected);
		}
		projected.model_policy = model_policy;
		let inner = self.inner.turn(projected, executor).await?;
		let stream =
			OwnedDialectStream::new(inner.map(ProtoTurnEvent::from), prepared).map(|event| {
				NativeTurnEvent::try_from(event).unwrap_or_else(|error| {
					NativeTurnEvent::Error(
						TurnError::builder()
							.kind(TurnErrorKind::Upstream)
							.detail(Str::from(error.to_string()))
							.unsupported(Vec::new())
							.retry_after_ms(0)
							.build(),
					)
				})
			});
		Ok(stream.boxed())
	}
}

impl<S, St, R> Service<R> for OwnedDialect<S>
where
	R: TurnRequestEnvelope,
	S: Service<R, Response = St>,
	St: Stream<Item = ProtoTurnEvent>,
{
	type Error = S::Error;
	type Response = DialectStream<St>;

	type Future = impl Future<Output = Result<Self::Response, Self::Error>>;

	fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		self.inner.poll_ready(cx)
	}

	fn call(&mut self, mut request: R) -> Self::Future {
		let native_tools = request
			.model_policy()
			.and_then(|policy| policy.capabilities.tools);
		let call = match prepare_request(request.request_mut(), &self.config, native_tools) {
			Ok(None) => DialectCall::Native(self.inner.call(request)),
			Ok(Some(prepared)) => DialectCall::Owned { future: self.inner.call(request), prepared },
			Err(error) => DialectCall::Rejected(prepare_error(error)),
		};
		async move {
			match call {
				DialectCall::Native(future) => {
					future.await.map(|inner| DialectStream::Native { inner })
				},
				DialectCall::Owned { future, prepared } => future.await.map(|inner| {
					DialectStream::Owned { inner: OwnedDialectStream::new(inner, prepared) }
				}),
				DialectCall::Rejected(event) => {
					Ok(DialectStream::Rejected { inner: single_turn(event) })
				},
			}
		}
	}
}

/// Helper which keeps native, owned, and rejected calls in one concrete future.
enum DialectCall<F> {
	Native(F),
	Owned { future: F, prepared: PreparedDialect },
	Rejected(ProtoTurnEvent),
}

pin_project! {
	/// Concrete response stream for native, owned, and pre-wire rejection paths.
	#[allow(missing_docs, reason = "pin-project-lite rejects documentation attributes on projected fields")]
	#[project = DialectStreamProj]
	pub enum DialectStream<S> {
		/// Provider-native request and response stream.
		Native {
			#[pin]
			inner: S,
		},
		/// Owned in-band projection around the provider stream.
		Owned {
			#[pin]
			inner: OwnedDialectStream<S>,
		},
		/// One terminal failure produced before provider dispatch.
		Rejected {
			#[pin]
			inner: SingleTurn,
		},
	}
}

impl<S> Stream for DialectStream<S>
where
	S: Stream<Item = ProtoTurnEvent>,
{
	type Item = ProtoTurnEvent;

	fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
		match self.project() {
			DialectStreamProj::Native { inner } => inner.poll_next(cx),
			DialectStreamProj::Owned { inner } => inner.poll_next(cx),
			DialectStreamProj::Rejected { inner } => inner.poll_next(cx),
		}
	}
}

struct PreparedDialect {
	projector: StreamProjector,
	model:     Str,
}
fn flatten_to_latest_user(request: &mut ChatRequest) {
	let mut prompt = String::new();
	let mut non_text = Vec::new();
	for item in &request.thread.items {
		let ItemKind::Message(message) = &item.kind else {
			continue;
		};
		for part in &message.parts {
			if let Part::Text(text) = part {
				if !prompt.is_empty() {
					prompt.push_str("\n\n");
				}
				prompt.push_str(text);
			} else {
				non_text.push(part.clone());
			}
		}
	}
	let mut parts = Vec::with_capacity(1 + non_text.len());
	parts.push(Part::Text(Str::from(prompt)));
	parts.extend(non_text);
	request.thread.items.clear();
	request.thread.items.push(
		Item::builder()
			.seq(0)
			.kind(ItemKind::Message(Message::builder().role(Role::User).parts(parts).build()))
			.props(Props::default())
			.build(),
	);
}

#[derive(Clone, Copy, Debug)]
enum SourcePart {
	Text,
	Thinking,
	Tool,
}

pin_project! {
	/// Allocation-disciplined owned projection stream.
	pub struct OwnedDialectStream<S> {
		#[pin]
		inner: Option<S>,
		projector: StreamProjector,
		accumulator: StreamAccumulator,
		source_parts: SmallVec<(u32, SourcePart), 8>,
		pending: SmallVec<ProtoTurnEvent, 8>,
		model: Str,
		done: bool,
	}
}

impl<S> OwnedDialectStream<S> {
	fn new(inner: S, prepared: PreparedDialect) -> Self {
		Self {
			inner:        Some(inner),
			projector:    prepared.projector,
			accumulator:  StreamAccumulator::new(),
			source_parts: SmallVec::new(),
			pending:      SmallVec::new(),
			model:        prepared.model,
			done:         false,
		}
	}
}

impl<S> Stream for OwnedDialectStream<S>
where
	S: Stream<Item = ProtoTurnEvent>,
{
	type Item = ProtoTurnEvent;

	fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
		let mut this = self.project();
		loop {
			if let Some(event) = this.pending.pop() {
				return Poll::Ready(Some(event));
			}
			if *this.done {
				return Poll::Ready(None);
			}
			let Some(inner) = this.inner.as_mut().as_pin_mut() else {
				*this.done = true;
				return Poll::Ready(None);
			};
			match inner.poll_next(cx) {
				Poll::Pending => return Poll::Pending,
				Poll::Ready(None) => {
					let batch = this.projector.finish();
					if enqueue_projection(batch, this.accumulator, this.pending) {
						finish_fabricated(this.accumulator, this.model, this.pending);
					} else {
						enqueue_terminal_error(
							"owned-dialect upstream ended without a terminal event",
							this.pending,
						);
					}
					this.inner.set(None);
					*this.done = true;
				},
				Poll::Ready(Some(event)) => {
					let native = match NativeTurnEvent::try_from(event) {
						Ok(event) => event,
						Err(error) => {
							enqueue_terminal_error(
								format!("invalid canonical dialect stream event: {error}"),
								this.pending,
							);
							this.inner.set(None);
							*this.done = true;
							continue;
						},
					};
					if process_event(
						native,
						this.projector,
						this.accumulator,
						this.source_parts,
						this.model,
						this.pending,
					) {
						this.inner.set(None);
						*this.done = true;
					}
				},
			}
		}
	}
}

fn process_event(
	event: NativeTurnEvent,
	projector: &mut StreamProjector,
	accumulator: &mut StreamAccumulator,
	source_parts: &mut SmallVec<(u32, SourcePart), 8>,
	model: &Str,
	pending: &mut SmallVec<ProtoTurnEvent, 8>,
) -> bool {
	match event {
		NativeTurnEvent::PartStart { index, kind, tool_call_id, tool_name } => {
			forget_source(source_parts, index);
			let source = match kind {
				StreamPartKind::Text => SourcePart::Text,
				StreamPartKind::Thinking => SourcePart::Thinking,
				StreamPartKind::ToolCall => SourcePart::Tool,
				_ => return false,
			};
			source_parts.push((index, source));
			if matches!(source, SourcePart::Tool) {
				let batch = projector.native_tool_start(index, tool_call_id, tool_name);
				return abort_if_fabricated(batch, accumulator, model, pending);
			}
		},
		NativeTurnEvent::PartDelta { index, chunk } => match source_kind(source_parts, index) {
			Some(SourcePart::Text) => {
				let batch = projector.feed_text(chunk);
				return abort_if_fabricated(batch, accumulator, model, pending);
			},
			Some(SourcePart::Tool) => {
				let batch = projector.native_tool_delta(index, chunk);
				return abort_if_fabricated(batch, accumulator, model, pending);
			},
			Some(SourcePart::Thinking) | None => {},
		},
		NativeTurnEvent::PartEnd { index, .. } => {
			if matches!(forget_source(source_parts, index), Some(SourcePart::Tool)) {
				let batch = projector.native_tool_end(index);
				return abort_if_fabricated(batch, accumulator, model, pending);
			}
		},
		NativeTurnEvent::Outcome(mut outcome) => {
			let batch = projector.finish();
			if abort_if_fabricated(batch, accumulator, model, pending) {
				return true;
			}
			match canonical_output(std::mem::take(accumulator)) {
				Ok((output, has_tool)) => {
					outcome.output = output;
					if has_tool {
						outcome.stop = StopReason::ToolUse;
					}
					enqueue_native(NativeTurnEvent::Outcome(outcome), pending);
				},
				Err(error) => enqueue_terminal_error(error.to_string(), pending),
			}
			return true;
		},
		NativeTurnEvent::Error(error) => {
			let batch = projector.finish();
			if abort_if_fabricated(batch, accumulator, model, pending) {
				return true;
			}
			enqueue_native(NativeTurnEvent::Error(error), pending);
			return true;
		},
		other => enqueue_native(other, pending),
	}
	false
}

fn abort_if_fabricated(
	batch: ProjectionBatch,
	accumulator: &mut StreamAccumulator,
	model: &Str,
	pending: &mut SmallVec<ProtoTurnEvent, 8>,
) -> bool {
	if enqueue_projection(batch, accumulator, pending) {
		finish_fabricated(accumulator, model, pending);
		true
	} else {
		false
	}
}

fn enqueue_projection(
	batch: ProjectionBatch,
	accumulator: &mut StreamAccumulator,
	pending: &mut SmallVec<ProtoTurnEvent, 8>,
) -> bool {
	let mut fabricated = false;
	let mut events = SmallVec::<NativeTurnEvent, 8>::new();
	for projection in batch {
		match projection {
			Projection::Event(event) => {
				if accumulator.push(&event).is_err() {
					continue;
				}
				events.push(event);
			},
			Projection::AbortFabricatedToolResult => fabricated = true,
			_ => {},
		}
	}
	for event in events.into_iter().rev() {
		pending.push(ProtoTurnEvent::from(event));
	}
	fabricated
}

fn finish_fabricated(
	accumulator: &mut StreamAccumulator,
	model: &Str,
	pending: &mut SmallVec<ProtoTurnEvent, 8>,
) {
	match canonical_output(std::mem::take(accumulator)) {
		Ok((output, has_tool)) if has_tool => {
			let outcome = ChatOutcome::builder()
				.output(output)
				.stop(StopReason::ToolUse)
				.provider(Str::default())
				.model(model.clone())
				.unsupported(Vec::new())
				.props(Props::default())
				.build();
			pending.insert(0, ProtoTurnEvent::from(NativeTurnEvent::Outcome(outcome)));
		},
		Ok(_) => enqueue_terminal_error("model fabricated a tool result before a tool call", pending),
		Err(error) => enqueue_terminal_error(error.to_string(), pending),
	}
}

fn canonical_output(
	accumulator: StreamAccumulator,
) -> Result<(Vec<Item>, bool), omp_llm_types::AccumulatorError> {
	let message = accumulator.message()?;
	let calls = accumulator.completed_tool_calls();
	let mut output = Vec::with_capacity(usize::from(!message.parts.is_empty()) + calls.len());
	if !message.parts.is_empty() {
		output.push(
			Item::builder()
				.seq(0)
				.kind(ItemKind::Message(message))
				.props(Props::default())
				.build(),
		);
	}
	let has_tool = !calls.is_empty();
	output.extend(calls.into_iter().map(|call| {
		Item::builder()
			.seq(0)
			.kind(ItemKind::ToolCall(call))
			.props(Props::default())
			.build()
	}));
	Ok((output, has_tool))
}

fn enqueue_native(event: NativeTurnEvent, pending: &mut SmallVec<ProtoTurnEvent, 8>) {
	pending.insert(0, ProtoTurnEvent::from(event));
}

fn enqueue_terminal_error(detail: impl AsRef<str>, pending: &mut SmallVec<ProtoTurnEvent, 8>) {
	enqueue_native(
		NativeTurnEvent::Error(
			TurnError::builder()
				.kind(TurnErrorKind::Upstream)
				.detail(Str::new(detail.as_ref()))
				.unsupported(Vec::new())
				.retry_after_ms(0)
				.build(),
		),
		pending,
	);
}

fn source_kind(parts: &[(u32, SourcePart)], index: u32) -> Option<SourcePart> {
	parts
		.iter()
		.find_map(|(source, kind)| (*source == index).then_some(*kind))
}

fn forget_source(parts: &mut SmallVec<(u32, SourcePart), 8>, index: u32) -> Option<SourcePart> {
	let at = parts.iter().position(|(source, _)| *source == index)?;
	Some(parts.swap_remove(at).1)
}

fn resolve_dialect(
	model: &str,
	config: &OwnedDialectConfig,
	native_tools: Option<bool>,
) -> Result<Option<omp_llm_dialect::Dialect>, PrepareError> {
	let selection = if config.omp_dialect.is_none()
		&& config.selection == DialectSelection::Auto
		&& native_tools != Some(false)
	{
		DialectSelection::Native
	} else {
		config.selection
	};
	selection
		.resolve(model, config.omp_dialect.as_deref())
		.map_err(|error| PrepareError::new(error.to_string()))
}

fn prepare_request(
	request: &mut omp_proto::inference::v1::TurnRequest,
	config: &OwnedDialectConfig,
	native_tools: Option<bool>,
) -> Result<Option<PreparedDialect>, PrepareError> {
	let model = request
		.params
		.as_ref()
		.map_or("", |params| params.model.as_str());
	let dialect = resolve_dialect(model, config, native_tools)?;
	let Some(dialect) = dialect else {
		return Ok(None);
	};
	prepare_owned_request(request, config, dialect).map(Some)
}

fn prepare_owned_request(
	request: &mut omp_proto::inference::v1::TurnRequest,
	config: &OwnedDialectConfig,
	dialect: omp_llm_dialect::Dialect,
) -> Result<PreparedDialect, PrepareError> {
	let params = request
		.params
		.as_mut()
		.ok_or_else(|| PrepareError::new("missing TurnRequest.params"))?;
	let model = Str::new(&params.model);
	apply_reasoning_policy(params, config.compat);
	params.tool_choice = None;
	let tools: Vec<ToolDef> = std::mem::take(&mut params.tools)
		.into_iter()
		.map(Into::into)
		.collect();
	let schemas: Vec<serde_json::Value> = tools
		.iter()
		.map(|tool| {
			serde_json::from_slice(&tool.schema_json).map_err(|error| {
				PrepareError::new(format!("invalid schema for dialect tool `{}`: {error}", tool.name))
			})
		})
		.collect::<Result<_, _>>()?;
	let views: SmallVec<InbandTool<'_>, 16> = tools
		.iter()
		.zip(&schemas)
		.map(|(tool, schema)| {
			InbandTool::new(
				tool.name.as_str(),
				(!tool.description.is_empty()).then_some(tool.description.as_str()),
				schema,
				&[],
			)
		})
		.collect();
	let input = request
		.input
		.as_mut()
		.ok_or_else(|| PrepareError::new("missing TurnRequest.input"))?;
	let turn_request::Input::Seed(seed) = input else {
		return Err(PrepareError::new("owned dialect requires a full seed thread"));
	};
	let thread = seed
		.thread
		.take()
		.ok_or_else(|| PrepareError::new("missing Seed.thread"))?
		.try_into()
		.map_err(|error: omp_llm_types::ConvertError| PrepareError::new(error.to_string()))?;
	let mut projected = project_inband_history(&thread, dialect, DialectRenderOptions::new(&views))
		.map_err(|error| PrepareError::new(error.to_string()))?;
	if !views.is_empty() {
		let mut prompt = StrMut::default();
		write_inband_tool_prompt(&mut prompt, &views, dialect)
			.map_err(|_| PrepareError::new("dialect prompt rendering failed"))?;
		let at = projected
			.items
			.iter()
			.take_while(
				|item| matches!(&item.kind, ItemKind::Message(message) if message.role == Role::System),
			)
			.count();
		projected.items.insert(
			at,
			Item::builder()
				.seq(0)
				.kind(ItemKind::Message(
					Message::builder()
						.role(Role::System)
						.parts(vec![Part::Text(prompt.freeze())])
						.build(),
				))
				.props(Props::default())
				.build(),
		);
	}
	seed.thread = Some(projected.into());
	let mut options = ScannerOptions::new(&views);
	options.parse_thinking = true;
	let projector = StreamProjector::new(dialect, options);
	Ok(PreparedDialect { projector, model })
}

fn apply_reasoning_policy(params: &mut omp_proto::inference::v1::ChatParams, compat: Compat) {
	let Some(reasoning) = params.thinking.as_mut() else {
		return;
	};
	let any_choice = params.tool_choice.is_some();
	let forced = params.tool_choice.as_ref().is_some_and(|choice| {
		matches!(choice.mode(), tool_choice::Mode::Required | tool_choice::Mode::Named)
	});
	let conflict = match compat.thinking_tool_choice_conflict {
		ThinkingToolChoiceConflict::None => false,
		ThinkingToolChoiceConflict::DropThinkingWhenForced => forced,
		ThinkingToolChoiceConflict::DropThinkingWhenAny => any_choice,
		ThinkingToolChoiceConflict::DropThinkingWhenEffort => {
			!matches!(reasoning.effort(), Effort::Unspecified | Effort::Off)
		},
	};
	if conflict {
		params.thinking = None;
	} else {
		reasoning.set_effort(Effort::Off);
		reasoning.budget_tokens = None;
		reasoning.hide_summary = Some(false);
	}
}

#[derive(Debug)]
struct PrepareError(Str);

impl PrepareError {
	fn new(detail: impl AsRef<str>) -> Self {
		Self(Str::new(detail.as_ref()))
	}
}

impl fmt::Display for PrepareError {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter.write_str(self.0.as_str())
	}
}

fn prepare_error(error: PrepareError) -> ProtoTurnEvent {
	NativeTurnEvent::Error(
		TurnError::builder()
			.kind(TurnErrorKind::Unsupported)
			.detail(error.0)
			.unsupported(Vec::new())
			.retry_after_ms(0)
			.build(),
	)
	.into()
}
