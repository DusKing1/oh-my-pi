//! Final projection from recovered semantic output into the closed [`Answer`]
//! contract.

use std::{
	mem,
	task::{Context, Poll},
	time::SystemTime,
};

use bytes::BytesMut;
use futures::StreamExt;
use omp_core::Str;
use tower::{Layer, Service};

use crate::{
	answer::{Answer, AnswerBody, AnswerKind, NativeResponse, NativeResponseBody, ResponseMeta},
	call::{Call, NativeResponseFraming, OperationCall, RawJson},
	catalog::OperationKind,
	codec::{HandshakenResponse, RawEvent},
	error::{Error, ErrorDetail, ErrorKind, ErrorPhase, RetryAction},
	event::ChatEvent,
	layer::LayerCall,
	receipt::ReasonId,
};

/// Projects one post-semantic response into the public typed answer.
#[derive(Clone, Copy, Debug, Default)]
pub struct AnswerLayer;

/// Typed answer projection service.
#[derive(Clone, Debug)]
pub struct AnswerService<S> {
	inner: S,
}

impl<S> Layer<S> for AnswerLayer {
	type Service = AnswerService<S>;

	fn layer(&self, inner: S) -> Self::Service {
		AnswerService { inner }
	}
}
struct AbortOnDrop(crate::layer::ExecutionContext, bool);
impl AbortOnDrop {
	fn disarm(&mut self) {
		assert!(mem::replace(&mut self.1, false), "session abort guard disarmed once");
	}
}
impl Drop for AbortOnDrop {
	fn drop(&mut self) {
		if self.1 {
			self.0.abort_session();
		}
	}
}

impl<S> Service<LayerCall<Call>> for AnswerService<S>
where
	S: Service<LayerCall<Call>, Response = HandshakenResponse, Error = Error> + Clone,
{
	type Error = Error;
	type Response = Answer;

	type Future = impl Future<Output = Result<Answer, Error>>;

	fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Error>> {
		self.inner.poll_ready(cx)
	}

	fn call(&mut self, request: LayerCall<Call>) -> Self::Future {
		let replacement = self.inner.clone();
		let mut inner = mem::replace(&mut self.inner, replacement);
		async move {
			let mut abort = AbortOnDrop(request.context.clone(), true);
			request.context.checkpoint(ErrorPhase::Streaming)?;
			let operation = request.payload.operation.kind();
			let native = match &request.payload.operation {
				OperationCall::Native(value) => Some(value.clone()),
				_ => None,
			};
			let plan = request
				.payload
				.execution
				.clone()
				.ok_or_else(|| invariant("answer.missing-execution-plan", &request.context))?;
			if plan.operation != operation {
				return Err(invariant("answer.operation-plan-mismatch", &request.context));
			}
			let mut response = match inner.call(request.clone()).await {
				Ok(response) => response,
				Err(error) if matches!(error.action, crate::error::RetryAction::ReseedSession) => {
					request.context.abort_session_for_reseed();
					abort.disarm();
					return Err(error);
				},
				Err(error) => return Err(error),
			};
			let meta = ResponseMeta {
				request_id:          request.payload.id.clone(),
				provider:            plan.provider.clone(),
				route:               plan.route.clone(),
				model:               plan.model.clone(),
				provider_request_id: response.meta.provider_request_id.clone(),
				created_at:          SystemTime::now(),
			};
			if operation == OperationKind::Realtime {
				let session = response
					.realtime
					.take()
					.ok_or_else(|| invariant("realtime.missing-session", &request.context))?;
				if response.events.is_some() {
					return Err(invariant("realtime.unexpected-events", &request.context));
				}
				abort.disarm();
				return Ok(Answer {
					meta,
					receipt: request.context.receipt(),
					body: AnswerBody::Realtime(session),
				});
			}
			if response.realtime.is_some() {
				return Err(invariant("answer.unexpected-realtime-session", &request.context));
			}
			let events = response
				.events
				.take()
				.ok_or_else(|| invariant("answer.missing-events", &request.context))?;
			if operation == OperationKind::Chat {
				let output = chat_stream(events, meta.clone(), request.context.clone());
				abort.disarm();
				return Ok(Answer {
					meta,
					receipt: request.context.receipt(),
					body: AnswerBody::Chat(output),
				});
			}
			if operation == OperationKind::GenerateImage {
				let output = image_stream(events, request.context.clone());
				abort.disarm();
				return Ok(Answer {
					meta,
					receipt: request.context.receipt(),
					body: AnswerBody::Images(output),
				});
			}
			if operation == OperationKind::Speak {
				let output = audio_stream(events, request.context.clone());
				abort.disarm();
				return Ok(Answer {
					meta,
					receipt: request.context.receipt(),
					body: AnswerBody::Speech(output),
				});
			}
			if operation == OperationKind::Transcribe {
				let output = transcript_stream(events, request.context.clone());
				abort.disarm();
				return Ok(Answer {
					meta,
					receipt: request.context.receipt(),
					body: AnswerBody::Transcript(output),
				});
			}
			if let Some(native) = native.as_ref()
				&& native.response_framing == NativeResponseFraming::Sse
			{
				let status = response
					.meta
					.status
					.ok_or_else(|| invariant("native.missing-status", &request.context))?;
				let provider_request_id = meta.provider_request_id.clone();
				let bytes = native_stream(events, native.max_response_bytes, request.context.clone());
				abort.disarm();
				return Ok(Answer {
					meta,
					receipt: request.context.receipt(),
					body: AnswerBody::Native(NativeResponse {
						status,
						media_type: content_type(&response.meta.headers),
						body: NativeResponseBody::Stream(bytes),
						provider_request_id,
					}),
				});
			}
			let body = match unary_body(
				operation,
				native.as_deref(),
				&response.meta,
				events,
				&request.context,
			)
			.await
			{
				Ok(body) => body,
				Err(error) if matches!(error.action, crate::error::RetryAction::ReseedSession) => {
					request.context.abort_session_for_reseed();
					abort.disarm();
					return Err(error);
				},
				Err(error) => return Err(error),
			};
			request.context.commit_session()?;
			abort.disarm();
			request.context.with_receipt(|receipt| {
				receipt.timings.total = request.context.elapsed();
				receipt.timings.completed_at = Some(SystemTime::now());
			});
			Ok(Answer { meta, receipt: request.context.receipt(), body })
		}
	}
}

fn chat_stream(
	mut input: crate::codec::RawEventStream,
	meta: ResponseMeta,
	context: crate::layer::ExecutionContext,
) -> crate::answer::ChatStream {
	Box::pin(async_stream::stream! {
		let mut abort = AbortOnDrop(context.clone(), true);
		if let Err(mut error) = context.checkpoint(ErrorPhase::Streaming) {
			context.finalize_error(&mut error); error.committed = false; context.abort_session(); abort.disarm(); yield Err(error); return;
		}
		yield Ok(ChatEvent::Started(meta));
		loop {
			let item = match next_with_deadline(&mut input, &context).await {
				Ok(item) => item,
				Err(mut error) => {
					context.cancel(); context.abort_session(); abort.disarm(); error.committed = context.is_committed(); error.receipt = context.receipt();
					yield Err(error);
					break;
				},
			};
			match item {
				None => {
					let mut error = Error::new(ErrorKind::StreamCorruption, ErrorPhase::Streaming, RetryAction::Never, context.receipt());
					error.committed = context.is_committed();
					error.detail = Some(ErrorDetail::Protocol { reason: ReasonId("chat.missing-terminal-completion".into()) });
					context.abort_session(); abort.disarm();
					yield Err(error);
					break;
				},
				Some(Ok(RawEvent::Chat(ChatEvent::Started(_)))) => {},
				Some(Ok(RawEvent::Chat(event))) => {
					let terminal = matches!(event, ChatEvent::Completed(_));
					if let Err(mut error) = context.record_session_event(&event) {
						context.finalize_error(&mut error); error.committed = context.is_committed(); context.abort_session(); abort.disarm(); yield Err(error); break;
					}
					if event.commits_output() { context.commit(); }
					if let ChatEvent::Completed(completion) = &event {
						context.merge_receipt(&completion.receipt);
						if let Err(mut error) = context.commit_session() {
							context.finalize_error(&mut error); error.committed = context.is_committed(); abort.disarm(); yield Err(error); break;
						}
					}
					yield Ok(event);
					if terminal { abort.disarm(); break; }
				},
				Some(Err(mut error)) => { context.finalize_error(&mut error); error.committed = context.is_committed(); context.abort_session(); abort.disarm(); yield Err(error); break; },
				Some(Ok(other)) => { let error = mismatch(OperationKind::Chat, raw_kind(&other), &context); context.abort_session(); abort.disarm(); yield Err(error); break; },
			}
		}
	})
}

fn image_stream(
	mut input: crate::codec::RawEventStream,
	context: crate::layer::ExecutionContext,
) -> crate::answer::GenerationStream<crate::answer::ImageArtifact> {
	Box::pin(async_stream::stream! {
		let mut abort = AbortOnDrop(context.clone(), true);
		loop {
			let item = match next_with_deadline(&mut input, &context).await {
				Ok(Some(item)) => item,
				Ok(None) => { yield Err(finalize_stream_error(invariant("image.missing-terminal", &context), &context)); return; },
				Err(error) => { yield Err(finalize_stream_error(error, &context)); return; },
			};
			match item {
				Ok(RawEvent::ImageGeneration(event)) => {
					let terminal = matches!(event, crate::answer::GenerationEvent::Completed(_));
					if !terminal { context.commit(); }
					yield Ok(event);
					if terminal {
						if let Err(mut error) = context.commit_session() { context.finalize_error(&mut error); yield Err(error); return; }
						abort.disarm();
						return;
					}
				}
				Ok(RawEvent::Metadata(metadata)) => context.observe_provider_metadata(metadata),
				Ok(RawEvent::Telemetry(telemetry)) => context.observe_provider_telemetry(telemetry),
				Ok(RawEvent::Failure(error)) | Err(error) => { yield Err(finalize_stream_error(error, &context)); return; },
				Ok(other) => { yield Err(finalize_stream_error(mismatch(OperationKind::GenerateImage, raw_kind(&other), &context), &context)); return; },
			}
		}
	})
}

fn audio_stream(
	mut input: crate::codec::RawEventStream,
	context: crate::layer::ExecutionContext,
) -> crate::answer::AudioStream {
	Box::pin(async_stream::stream! {
		let mut abort = AbortOnDrop(context.clone(), true);
		loop {
			let item = match next_with_deadline(&mut input, &context).await {
				Ok(Some(item)) => item,
				Ok(None) => { yield Err(finalize_stream_error(invariant("speech.missing-terminal", &context), &context)); return; },
				Err(error) => { yield Err(finalize_stream_error(error, &context)); return; },
			};
			match item {
				Ok(RawEvent::Audio(chunk)) => {
					let terminal = chunk.final_chunk;
					context.commit();
					yield Ok(chunk);
					if terminal {
						if let Err(error) = context.commit_session() { yield Err(error); return; }
						abort.disarm();
						return;
					}
				}
				Ok(RawEvent::Metadata(metadata)) => context.observe_provider_metadata(metadata),
				Ok(RawEvent::Telemetry(telemetry)) => context.observe_provider_telemetry(telemetry),
				Ok(RawEvent::Failure(error)) | Err(error) => { yield Err(finalize_stream_error(error, &context)); return; },
				Ok(other) => { yield Err(finalize_stream_error(mismatch(OperationKind::Speak, raw_kind(&other), &context), &context)); return; },
			}
		}
	})
}

fn transcript_stream(
	mut input: crate::codec::RawEventStream,
	context: crate::layer::ExecutionContext,
) -> crate::answer::TranscriptStream {
	Box::pin(async_stream::stream! {
		let mut abort = AbortOnDrop(context.clone(), true);
		loop {
			let item = match next_with_deadline(&mut input, &context).await {
				Ok(Some(item)) => item,
				Ok(None) => { yield Err(finalize_stream_error(invariant("transcript.missing-terminal", &context), &context)); return; },
				Err(error) => { yield Err(finalize_stream_error(error, &context)); return; },
			};
			match item {
				Ok(RawEvent::Transcript(event)) => {
					let terminal = matches!(event, crate::answer::TranscriptEvent::Completed { .. });
					context.commit();
					yield Ok(event);
					if terminal {
						if let Err(error) = context.commit_session() { yield Err(error); return; }
						abort.disarm();
						return;
					}
				}
				Ok(RawEvent::Metadata(metadata)) => context.observe_provider_metadata(metadata),
				Ok(RawEvent::Telemetry(telemetry)) => context.observe_provider_telemetry(telemetry),
				Ok(RawEvent::Failure(error)) | Err(error) => { yield Err(finalize_stream_error(error, &context)); return; },
				Ok(other) => { yield Err(finalize_stream_error(mismatch(OperationKind::Transcribe, raw_kind(&other), &context), &context)); return; },
			}
		}
	})
}

fn native_stream(
	mut input: crate::codec::RawEventStream,
	limit: u64,
	context: crate::layer::ExecutionContext,
) -> crate::body::ByteStream {
	Box::pin(async_stream::stream! {
		let mut abort = AbortOnDrop(context.clone(), true);
		let mut observed = 0_u64;
		loop {
			let item = match next_with_deadline(&mut input, &context).await { Ok(Some(item)) => item, Ok(None) => break, Err(error) => { context.cancel(); context.abort_session(); abort.disarm(); yield Err(finalize_stream_error(error, &context)); return; } };
			match item {
				Ok(RawEvent::NativeChunk(bytes)) => {
					observed = observed.saturating_add(bytes.len() as u64);
					if observed > limit { let error = limit_error(limit, observed, &context); context.abort_session(); abort.disarm(); yield Err(error); return; }
					context.commit(); yield Ok(bytes);
				},
				Ok(RawEvent::Metadata(metadata)) => context.observe_provider_metadata(metadata),
				Ok(RawEvent::Telemetry(telemetry)) => context.observe_provider_telemetry(telemetry),
				Ok(RawEvent::ProviderState(state)) => context.stage_provider_state(state),
				Ok(RawEvent::Failure(mut error)) | Err(mut error) => { context.finalize_error(&mut error); error.committed = context.is_committed(); context.abort_session(); abort.disarm(); yield Err(error); return; },
				Ok(other) => { let error = finalize_stream_error(mismatch(OperationKind::Native, raw_kind(&other), &context), &context); context.abort_session(); abort.disarm(); yield Err(error); return; },
			}
		}
		if let Err(mut error) = context.commit_session() { context.finalize_error(&mut error); yield Err(error); return; }
		abort.disarm();
	})
}

async fn next_with_deadline(
	input: &mut crate::codec::RawEventStream,
	context: &crate::layer::ExecutionContext,
) -> Result<Option<Result<RawEvent, Error>>, Error> {
	loop {
		context.checkpoint(ErrorPhase::Streaming)?;
		let Some(limit) = context.budget().max_elapsed else {
			return Ok(input.next().await);
		};
		let remaining = limit.saturating_sub(context.elapsed());
		tokio::select! {
			biased;
			item = input.next() => return Ok(item),
			_ = tokio::time::sleep(remaining) => {
				if let Err(error) = context.checkpoint(ErrorPhase::Streaming) { return Err(error); }
			},
		}
	}
}

async fn unary_body(
	operation: OperationKind,
	native: Option<&crate::call::NativeRequest>,
	handshake: &crate::codec::HandshakeMeta,
	mut events: crate::codec::RawEventStream,
	context: &crate::layer::ExecutionContext,
) -> Result<AnswerBody, Error> {
	let mut answer = None;
	let mut native_bytes = BytesMut::new();
	loop {
		let Some(item) = next_with_deadline(&mut events, context).await? else {
			break;
		};
		match item {
			Err(mut error) | Ok(RawEvent::Failure(mut error)) => {
				context.finalize_error(&mut error);
				return Err(error);
			},
			Ok(RawEvent::Answer(body)) if answer.is_none() => answer = Some(body),
			Ok(RawEvent::Answer(body)) => return Err(mismatch(operation, body.kind(), context)),
			Ok(RawEvent::NativeChunk(bytes)) if operation == OperationKind::Native => {
				let limit = native.map_or(0, |request| request.max_response_bytes);
				let observed = native_bytes.len() as u64 + bytes.len() as u64;
				if observed > limit {
					return Err(limit_error(limit, observed, context));
				}
				native_bytes.extend_from_slice(&bytes);
			},
			Ok(RawEvent::ProviderState(state)) => context.stage_provider_state(state),
			Ok(RawEvent::Metadata(metadata)) => context.observe_provider_metadata(metadata),
			Ok(RawEvent::Telemetry(telemetry)) => context.observe_provider_telemetry(telemetry),
			Ok(other) => return Err(mismatch(operation, raw_kind(&other), context)),
		}
	}
	let body = if operation == OperationKind::Native && answer.is_none() {
		let request = native.ok_or_else(|| invariant("native.request-missing", context))?;
		let status = handshake
			.status
			.ok_or_else(|| invariant("native.missing-status", context))?;
		let bytes = native_bytes.freeze();
		let body = match request.response_framing {
			NativeResponseFraming::Json => NativeResponseBody::Json(
				RawJson::new(bytes, request.max_response_bytes)
					.map_err(|_| invariant("native.invalid-json-response", context))?,
			),
			NativeResponseFraming::Bytes => NativeResponseBody::Bytes(bytes),
			NativeResponseFraming::Sse => {
				return Err(invariant("native.streaming-projection-reached-unary", context));
			},
		};
		AnswerBody::Native(NativeResponse {
			status,
			media_type: content_type(&handshake.headers),
			body,
			provider_request_id: handshake.provider_request_id.clone(),
		})
	} else {
		answer.ok_or_else(|| invariant("answer.missing-body", context))?
	};
	if expected_kind(operation) != body.kind() {
		return Err(mismatch(operation, body.kind(), context));
	}
	Ok(body)
}

fn expected_kind(operation: OperationKind) -> AnswerKind {
	match operation {
		OperationKind::Chat => AnswerKind::Chat,
		OperationKind::CountTokens => AnswerKind::Tokens,
		OperationKind::Tokenize => AnswerKind::TokenIds,
		OperationKind::Detokenize => AnswerKind::Text,
		OperationKind::Embed => AnswerKind::Embeddings,
		OperationKind::GenerateImage => AnswerKind::Images,
		OperationKind::GenerateVideo => AnswerKind::Video,
		OperationKind::Speak => AnswerKind::Speech,
		OperationKind::Transcribe => AnswerKind::Transcript,
		OperationKind::Realtime => AnswerKind::Realtime,
		OperationKind::Search => AnswerKind::Search,
		OperationKind::Usage => AnswerKind::Usage,
		OperationKind::DiscoverModels => AnswerKind::Models,
		OperationKind::Auth => AnswerKind::Auth,
		OperationKind::Native => AnswerKind::Native,
	}
}

fn raw_kind(event: &RawEvent) -> AnswerKind {
	match event {
		RawEvent::Chat(_) | RawEvent::Completion(_) | RawEvent::ToolCallComplete { .. } => {
			AnswerKind::Chat
		},
		RawEvent::Answer(body) => body.kind(),
		RawEvent::NativeChunk(_) => AnswerKind::Native,
		RawEvent::ImageGeneration(_) => AnswerKind::Images,
		RawEvent::VideoGeneration(_) => AnswerKind::Video,
		RawEvent::Audio(_) => AnswerKind::Speech,
		RawEvent::Transcript(_) => AnswerKind::Transcript,
		RawEvent::DiscoveredModels { .. } => AnswerKind::Models,
		RawEvent::Control(_) => AnswerKind::Realtime,
		RawEvent::ProviderState(_)
		| RawEvent::Metadata(_)
		| RawEvent::Telemetry(_)
		| RawEvent::Failure(_) => AnswerKind::Native,
	}
}
fn finalize_stream_error(mut error: Error, context: &crate::layer::ExecutionContext) -> Error {
	context.finalize_error(&mut error);
	error.committed = context.is_committed();
	error.receipt = context.receipt();
	error
}
fn mismatch(
	expected: OperationKind,
	actual: AnswerKind,
	context: &crate::layer::ExecutionContext,
) -> Error {
	Error::body_variant_mismatch(expected, actual, context.receipt())
}
fn invariant(reason: &'static str, context: &crate::layer::ExecutionContext) -> Error {
	let mut error = Error::new(
		ErrorKind::ProviderContractMismatch,
		ErrorPhase::Internal,
		RetryAction::Never,
		context.receipt(),
	);
	error.detail = Some(ErrorDetail::Protocol { reason: ReasonId(reason.into()) });
	error
}
fn limit_error(limit: u64, observed: u64, context: &crate::layer::ExecutionContext) -> Error {
	let mut error = Error::new(
		ErrorKind::PolicyBufferExceeded,
		ErrorPhase::Streaming,
		RetryAction::Never,
		context.receipt(),
	);
	error.committed = context.is_committed();
	error.detail = Some(ErrorDetail::Budget {
		dimension: "native_response_bytes".into(),
		limit:     limit as u128,
		observed:  observed as u128,
	});
	error
}
fn content_type(headers: &[crate::codec::RequestHeader]) -> Option<Str> {
	headers
		.iter()
		.find(|header| header.name.as_str().eq_ignore_ascii_case("content-type"))
		.map(|header| header.value.clone())
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{answer::AudioChunk, receipt::ExecutionBudget};

	#[tokio::test]
	async fn audio_terminal_commits_only_after_final_chunk() {
		let context = crate::layer::ExecutionContext::new(ExecutionBudget::default());
		let input: crate::codec::RawEventStream =
			Box::pin(futures::stream::iter([Ok(RawEvent::Audio(AudioChunk {
				bytes:       bytes::Bytes::from_static(b"audio"),
				start_ms:    Some(0),
				end_ms:      Some(1),
				final_chunk: true,
			}))]));
		assert!(!context.is_committed());
		let mut output = audio_stream(input, context.clone());
		assert!(output.next().await.unwrap().is_ok());
		assert!(output.next().await.is_none());
		assert!(context.is_committed());
	}

	#[tokio::test]
	async fn audio_failure_after_visible_chunk_is_committed() {
		let context = crate::layer::ExecutionContext::new(ExecutionBudget::default());
		let failure = Error::new(
			ErrorKind::StreamCorruption,
			ErrorPhase::Streaming,
			RetryAction::Never,
			context.receipt(),
		);
		let input: crate::codec::RawEventStream = Box::pin(futures::stream::iter([
			Ok(RawEvent::Audio(AudioChunk {
				bytes:       bytes::Bytes::from_static(b"partial"),
				start_ms:    Some(0),
				end_ms:      Some(1),
				final_chunk: false,
			})),
			Err(failure),
		]));
		let mut output = audio_stream(input, context);
		assert!(output.next().await.unwrap().is_ok());
		let error = output.next().await.unwrap().unwrap_err();
		assert!(error.committed);
	}
}
