use std::{
	collections::VecDeque,
	pin::Pin,
	sync::{
		Arc, Mutex,
		atomic::{AtomicUsize, Ordering},
	},
	time::Duration,
};

use bytes::Bytes;
use futures::{Stream, StreamExt};
use omp_agent::{
	Error, InProcTurnClient, InvokeFrame, RpcTurnSession, TurnClient, TurnId, TurnInput,
	TurnOptions, TurnSession,
};
use omp_proto::{
	inference::v1::{self as pb, inference_server::Inference},
	thread::v1::{Item, Revision, Thread},
};
use tonic::{Request, Response, Status};

type RpcStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;

fn flume_stream<T: Send + 'static>(receiver: flume::Receiver<T>) -> impl Stream<Item = T> + Send {
	futures::stream::unfold(receiver, |receiver| async move {
		let item = receiver.recv_async().await.ok()?;
		Some((item, receiver))
	})
}

struct Exchange {
	before_input: Vec<Result<pb::TurnEvent, Status>>,
	input_count:  usize,
	after_input:  Vec<Result<pb::TurnEvent, Status>>,
}

impl Exchange {
	fn events(events: Vec<pb::TurnEvent>) -> Self {
		Self {
			before_input: events.into_iter().map(Ok).collect(),
			input_count:  0,
			after_input:  Vec::new(),
		}
	}

	fn duplex(before_input: Vec<pb::TurnEvent>, after_input: Vec<pb::TurnEvent>) -> Self {
		Self {
			before_input: before_input.into_iter().map(Ok).collect(),
			input_count:  2,
			after_input:  after_input.into_iter().map(Ok).collect(),
		}
	}
}

#[derive(Clone)]
struct ScriptedInference {
	state: Arc<ScriptState>,
}

struct ScriptState {
	exchanges: Mutex<VecDeque<Exchange>>,
	opens:     flume::Sender<pb::TurnRequest>,
	inputs:    flume::Sender<Vec<pb::TurnFrame>>,
	calls:     AtomicUsize,
}

struct Observed {
	opens:  flume::Receiver<pb::TurnRequest>,
	inputs: flume::Receiver<Vec<pb::TurnFrame>>,
	state:  Arc<ScriptState>,
}

impl ScriptedInference {
	fn new(exchanges: Vec<Exchange>) -> (Self, Observed) {
		let (open_sender, opens) = flume::bounded(exchanges.len().max(1));
		let (input_sender, inputs) = flume::bounded(exchanges.len().max(1));
		let state = Arc::new(ScriptState {
			exchanges: Mutex::new(exchanges.into()),
			opens:     open_sender,
			inputs:    input_sender,
			calls:     AtomicUsize::new(0),
		});
		(Self { state: Arc::clone(&state) }, Observed { opens, inputs, state })
	}
}

macro_rules! impl_scripted_inference {
	(
		unary { $($unary:ident: $unary_request:ty => $unary_response:ty),* $(,)? }
		stream { $($stream:ident: $stream_request:ty => $stream_response:ty),* $(,)? }
	) => {
		#[tonic::async_trait]
		impl Inference for ScriptedInference {
			type AttachGenerationStream = RpcStream<pb::GenerationStatus>;
			type GenerateImageStream = RpcStream<pb::ImageEvent>;
			type NativeStream = RpcStream<pb::NativeChunk>;
			type RealtimeStream = RpcStream<pb::RealtimeEvent>;
			type SpeakStream = RpcStream<pb::SpeakEvent>;
			type TurnStream = RpcStream<pb::TurnEvent>;
			type WatchModelsStream = RpcStream<pb::ModelEvent>;

			async fn turn(
				&self,
				request: Request<tonic::Streaming<pb::TurnFrame>>,
			) -> Result<Response<Self::TurnStream>, Status> {
				self.state.calls.fetch_add(1, Ordering::SeqCst);
				let mut incoming = request.into_inner();
				let first = incoming
					.message()
					.await?
					.ok_or_else(|| Status::invalid_argument("missing opening frame"))?;
				let open = match first.frame {
					Some(pb::turn_frame::Frame::Open(open)) => open,
					_ => return Err(Status::invalid_argument("first frame was not open")),
				};
				self.state
					.opens
					.send_async(open)
					.await
					.map_err(|_| Status::internal("test observer closed"))?;
				let exchange = self
					.state
					.exchanges
					.lock()
					.map_err(|_| Status::internal("script lock poisoned"))?
					.pop_front()
					.ok_or_else(|| Status::failed_precondition("no scripted exchange"))?;

				let inputs = self.state.inputs.clone();
				let (sender, receiver) = flume::bounded(1);
				tokio::spawn(async move {
					for event in exchange.before_input {
						if sender.send_async(event).await.is_err() {
							return;
						}
					}

					if exchange.input_count != 0 {
						let mut observed = Vec::with_capacity(exchange.input_count);
						for _ in 0..exchange.input_count {
							match incoming.message().await {
								Ok(Some(frame)) => observed.push(frame),
								Ok(None) => {
									let _ = sender
										.send_async(Err(Status::failed_precondition(
											"invocation stream closed early",
										)))
										.await;
									return;
								},
								Err(status) => {
									let _ = sender.send_async(Err(status)).await;
									return;
								},
							}
						}
						if inputs.send_async(observed).await.is_err() {
							return;
						}
					}

					for event in exchange.after_input {
						if sender.send_async(event).await.is_err() {
							return;
						}
					}
				});
				Ok(Response::new(Box::pin(flume_stream(receiver))))
			}

			$(
				async fn $unary(
					&self,
					_request: Request<$unary_request>,
				) -> Result<Response<$unary_response>, Status> {
					Err(Status::unimplemented(stringify!($unary)))
				}
			)*

			$(
				async fn $stream(
					&self,
					_request: Request<$stream_request>,
				) -> Result<Response<$stream_response>, Status> {
					Err(Status::unimplemented(stringify!($stream)))
				}
			)*
		}
	};
}

impl_scripted_inference! {
	unary {
		fork: pb::ForkRequest => pb::ForkResponse,
		drop: pb::DropRequest => pb::DropResponse,
		count_tokens: pb::CountTokensRequest => pb::CountTokensResponse,
		tokenize: pb::TokenizeRequest => pb::TokenizeResponse,
		detokenize: pb::DetokenizeRequest => pb::DetokenizeResponse,
		embed: pb::EmbedRequest => pb::EmbedResponse,
		transcribe: pb::TranscribeRequest => pb::TranscribeResponse,
		generate_video: pb::GenerateVideoRequest => pb::GenerationStatus,
		get_generation: pb::GetGenerationRequest => pb::GenerationStatus,
		cancel_generation: pb::CancelGenerationRequest => pb::GenerationStatus,
		search: pb::SearchRequest => pb::SearchResponse,
		usage: pb::UsageRequest => pb::UsageResponse,
		list_providers: pb::ListProvidersRequest => pb::ListProvidersResponse,
		list_models: pb::ListModelsRequest => pb::ListModelsResponse,
		refresh_models: pb::RefreshModelsRequest => pb::ListModelsResponse,
	}
	stream {
		realtime: tonic::Streaming<pb::RealtimeFrame> => Self::RealtimeStream,
		generate_image: pb::GenerateImageRequest => Self::GenerateImageStream,
		speak: pb::SpeakRequest => Self::SpeakStream,
		attach_generation: pb::AttachGenerationRequest => Self::AttachGenerationStream,
		native: pb::NativeRequest => Self::NativeStream,
		watch_models: pb::WatchModelsRequest => Self::WatchModelsStream,
	}
}

fn event(event: pb::turn_event::Event) -> pb::TurnEvent {
	pb::TurnEvent { event: Some(event) }
}

fn accepted(replay: bool) -> pb::TurnEvent {
	event(pb::turn_event::Event::Accepted(pb::Accepted { replay }))
}

fn outcome(provider: &str, revision: Option<Revision>) -> pb::TurnEvent {
	event(pb::turn_event::Event::Outcome(pb::Outcome {
		provider: provider.to_owned(),
		revision,
		..Default::default()
	}))
}

fn tool_call_item() -> Item {
	Item {
		seq:           17,
		created_at_ms: 23,
		kind:          Some(omp_proto::thread::v1::item::Kind::ToolCall(
			omp_proto::thread::v1::ToolCall {
				id: "call-1".to_owned(),
				name: "edit".to_owned(),
				args_json: Bytes::from_static(br#"{"path":"src/lib.rs"}"#),
				..Default::default()
			},
		)),
		props:         Some(pb::ValueMap {
			fields: std::collections::BTreeMap::from([("omp/tool-rev".to_owned(), pb::Value {
				kind: Some(pb::value::Kind::String("hl.2".to_owned())),
			})]),
		}),
	}
}

fn outcome_with_output(provider: &str, output: Vec<Item>) -> pb::TurnEvent {
	event(pb::turn_event::Event::Outcome(pb::Outcome {
		provider: provider.to_owned(),
		output,
		..Default::default()
	}))
}

async fn observe<T>(receiver: &flume::Receiver<T>) -> T {
	tokio::time::timeout(Duration::from_secs(2), receiver.recv_async())
		.await
		.expect("scripted service did not observe the request")
		.expect("scripted service observation channel closed")
}

async fn next_event(session: &mut RpcTurnSession) -> Option<Result<pb::TurnEvent, Error>> {
	let mut events = session.events();
	tokio::time::timeout(Duration::from_secs(2), events.next())
		.await
		.expect("scripted service did not produce the next event")
}

#[tokio::test]
async fn full_then_delta_preserve_the_injected_services_context_revision() {
	let revision = Revision { head: 9, token: Bytes::from_static(b"service-revision") };
	let (service, observed) = ScriptedInference::new(vec![
		Exchange::events(vec![accepted(false), outcome("injected/full", Some(revision.clone()))]),
		Exchange::events(vec![accepted(false), outcome("injected/delta", None)]),
	]);
	let client = InProcTurnClient::new(service)
		.await
		.expect("in-process channel");
	let thread = Thread { items: vec![Item { seq: 4, ..Default::default() }] };
	let options =
		TurnOptions { context_id: Some("context-from-caller".into()), ..Default::default() };

	let mut full = client
		.turn(TurnId::new("full-turn"), TurnInput::Full(thread.clone()), &options)
		.await
		.expect("full turn opens");
	assert!(matches!(
		next_event(&mut full).await,
		Some(Ok(pb::TurnEvent {
			event: Some(pb::turn_event::Event::Accepted(pb::Accepted { replay: false })),
		}))
	));
	let returned_revision = match next_event(&mut full).await {
		Some(Ok(pb::TurnEvent { event: Some(pb::turn_event::Event::Outcome(outcome)) })) => {
			assert_eq!(outcome.provider, "injected/full");
			outcome.revision.expect("stateful outcome revision")
		},
		other => panic!("expected full outcome, got {other:?}"),
	};
	assert_eq!(returned_revision, revision);

	let full_open = observe(&observed.opens).await;
	assert_eq!(full_open.turn_id, "full-turn");
	match full_open.input {
		Some(pb::turn_request::Input::Seed(seed)) => {
			assert_eq!(seed.context_id, "context-from-caller");
			assert_eq!(seed.thread, Some(thread));
		},
		other => panic!("expected seed input, got {other:?}"),
	}

	let context = pb::ContextRef {
		context_id: "context-from-caller".to_owned(),
		expected:   Some(returned_revision.clone()),
	};
	let delta = pb::ThreadDelta {
		truncate_to: Some(7),
		append:      vec![Item { seq: 0, ..Default::default() }],
	};
	let mut incremental = client
		.turn(TurnId::new("delta-turn"), TurnInput::Delta(context.clone(), delta.clone()), &options)
		.await
		.expect("delta turn opens");
	let _ = next_event(&mut incremental)
		.await
		.expect("accepted")
		.expect("accepted event");
	match next_event(&mut incremental).await {
		Some(Ok(pb::TurnEvent { event: Some(pb::turn_event::Event::Outcome(outcome)) })) => {
			assert_eq!(outcome.provider, "injected/delta")
		},
		other => panic!("expected delta outcome, got {other:?}"),
	}

	let delta_open = observe(&observed.opens).await;
	assert_eq!(delta_open.turn_id, "delta-turn");
	match delta_open.input {
		Some(pb::turn_request::Input::Incremental(incremental)) => {
			assert_eq!(incremental.context, Some(context));
			assert_eq!(incremental.delta, Some(delta));
		},
		other => panic!("expected incremental input, got {other:?}"),
	}
	assert_eq!(observed.state.calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn conflict_and_need_full_are_typed_terminal_recoveries_without_seam_policy() {
	let conflict = pb::TurnError {
		kind: pb::turn_error::Kind::Conflict as i32,
		detail: "stale revision".to_owned(),
		actual: Some(Revision { head: 12, token: Bytes::from_static(&[12]) }),
		error_id: Some(41),
		..Default::default()
	};
	let need_full = pb::TurnError {
		kind: pb::turn_error::Kind::NeedFull as i32,
		detail: "context evicted".to_owned(),
		error_id: Some(42),
		..Default::default()
	};
	let (service, observed) = ScriptedInference::new(vec![
		Exchange::events(vec![event(pb::turn_event::Event::Error(conflict.clone()))]),
		Exchange::events(vec![event(pb::turn_event::Event::Error(need_full.clone()))]),
	]);
	let client = InProcTurnClient::new(service)
		.await
		.expect("in-process channel");

	for (turn_id, expected) in [("conflict", &conflict), ("need-full", &need_full)] {
		let mut session = client
			.turn(TurnId::new(turn_id), TurnInput::Full(Thread::default()), &TurnOptions::default())
			.await
			.expect("turn opens");
		let error = next_event(&mut session)
			.await
			.expect("terminal error event")
			.expect_err("error is not exposed as a regular event");
		match (&error, pb::turn_error::Kind::try_from(expected.kind).expect("known kind")) {
			(Error::Conflict(actual), pb::turn_error::Kind::Conflict)
			| (Error::NeedFull(actual), pb::turn_error::Kind::NeedFull) => {
				assert_eq!(actual, expected);
				assert!(error.is_recovery());
				assert_eq!(error.turn_error(), Some(expected));
			},
			other => panic!("wrong recovery classification: {other:?}"),
		}
		assert!(next_event(&mut session).await.is_none());
	}
	assert_eq!(observed.state.calls.load(Ordering::SeqCst), 2);
	assert_eq!(observe(&observed.opens).await.turn_id, "conflict");
	assert_eq!(observe(&observed.opens).await.turn_id, "need-full");
}

#[tokio::test]
async fn replay_acceptance_and_unknown_terminal_errors_pass_through_verbatim() {
	let unknown = pb::TurnError {
		kind: 777,
		detail: "future gateway error".to_owned(),
		retry_after_ms: 55,
		error_id: Some(99),
		..Default::default()
	};
	let committed_call = tool_call_item();
	let (service, _observed) = ScriptedInference::new(vec![
		Exchange::events(vec![
			accepted(true),
			outcome_with_output("replayed", vec![committed_call.clone()]),
		]),
		Exchange::events(vec![event(pb::turn_event::Event::Error(unknown.clone()))]),
	]);
	let client = InProcTurnClient::new(service)
		.await
		.expect("in-process channel");

	let mut replay = client
		.turn(TurnId::new("same-turn"), TurnInput::Full(Thread::default()), &TurnOptions::default())
		.await
		.expect("replay opens");
	match next_event(&mut replay).await {
		Some(Ok(pb::TurnEvent { event: Some(pb::turn_event::Event::Accepted(accepted)) })) => {
			assert!(accepted.replay)
		},
		other => panic!("expected replay acceptance, got {other:?}"),
	}
	match next_event(&mut replay).await {
		Some(Ok(pb::TurnEvent { event: Some(pb::turn_event::Event::Outcome(outcome)) })) => {
			assert_eq!(outcome.output, vec![committed_call]);
		},
		other => panic!("expected replay outcome, got {other:?}"),
	}

	let mut future_error = client
		.turn(
			TurnId::new("future-error"),
			TurnInput::Full(Thread::default()),
			&TurnOptions::default(),
		)
		.await
		.expect("future-error turn opens");
	match next_event(&mut future_error).await {
		Some(Err(Error::Terminal(actual))) => assert_eq!(actual, unknown),
		other => panic!("expected retained unknown terminal error, got {other:?}"),
	}
	assert!(next_event(&mut future_error).await.is_none());
}

#[tokio::test]
async fn invocation_frames_flow_while_the_response_stream_is_live() {
	let invoke = pb::Invoke {
		invocation_id: "invoke-7".to_owned(),
		name: "exec.shell".to_owned(),
		..Default::default()
	};
	let (service, observed) = ScriptedInference::new(vec![Exchange::duplex(
		vec![accepted(false), event(pb::turn_event::Event::Invoke(invoke.clone()))],
		vec![outcome("after-invocation", None)],
	)]);
	let client = InProcTurnClient::new(service)
		.await
		.expect("in-process channel");
	let options = TurnOptions {
		executor: Some(pb::Executor { tools: vec!["exec.shell".to_owned()] }),
		..Default::default()
	};
	let mut session = client
		.turn(TurnId::new("duplex"), TurnInput::Full(Thread::default()), &options)
		.await
		.expect("duplex turn opens");

	let _ = next_event(&mut session)
		.await
		.expect("accepted")
		.expect("accepted event");
	match next_event(&mut session).await {
		Some(Ok(pb::TurnEvent { event: Some(pb::turn_event::Event::Invoke(actual)) })) => {
			assert_eq!(actual, invoke)
		},
		other => panic!("expected invocation, got {other:?}"),
	}

	let input = pb::InvokeInput {
		invocation_id: "invoke-7".to_owned(),
		payload:       Some(pb::invoke_input::Payload::Chunk(pb::invoke_input::Chunk {
			channel: pb::invoke_input::chunk::Channel::Stdout as i32,
			data:    Bytes::from_static(b"streamed output"),
		})),
	};
	let complete = pb::InvokeComplete {
		invocation_id: "invoke-7".to_owned(),
		status: Some(pb::ExecStatus {
			outcome: pb::exec_status::Outcome::Exited as i32,
			exit_code: 0,
			..Default::default()
		}),
		..Default::default()
	};
	session
		.submit(InvokeFrame::Input(input.clone()))
		.await
		.expect("input accepted");
	session
		.submit(InvokeFrame::Complete(complete.clone()))
		.await
		.expect("completion accepted");

	let frames = observe(&observed.inputs).await;
	assert_eq!(frames.len(), 2);
	assert_eq!(frames[0].frame, Some(pb::turn_frame::Frame::Input(input)));
	assert_eq!(frames[1].frame, Some(pb::turn_frame::Frame::Complete(complete)));
	match next_event(&mut session).await {
		Some(Ok(pb::TurnEvent { event: Some(pb::turn_event::Event::Outcome(outcome)) })) => {
			assert_eq!(outcome.provider, "after-invocation")
		},
		other => panic!("expected post-invocation outcome, got {other:?}"),
	}
}

#[tokio::test]
async fn sessions_keep_the_injected_server_alive_and_report_response_channel_shutdown() {
	let (service, _observed) = ScriptedInference::new(vec![Exchange::events(vec![
		accepted(false),
		outcome("server-owned-by-session", None),
	])]);
	let client = InProcTurnClient::new(service)
		.await
		.expect("in-process channel");
	let mut session = client
		.turn(
			TurnId::new("survives-client"),
			TurnInput::Full(Thread::default()),
			&TurnOptions::default(),
		)
		.await
		.expect("turn opens");
	drop(client);
	let _ = next_event(&mut session)
		.await
		.expect("accepted after client drop")
		.expect("accepted");
	match next_event(&mut session).await {
		Some(Ok(pb::TurnEvent { event: Some(pb::turn_event::Event::Outcome(outcome)) })) => {
			assert_eq!(outcome.provider, "server-owned-by-session")
		},
		other => panic!("expected outcome after dropping client, got {other:?}"),
	}

	let (service, _observed) = ScriptedInference::new(vec![Exchange::events(vec![accepted(false)])]);
	let client = InProcTurnClient::new(service)
		.await
		.expect("in-process channel");
	let options = TurnOptions {
		executor: Some(pb::Executor { tools: vec!["exec.shell".to_owned()] }),
		..Default::default()
	};
	let mut shutdown = client
		.turn(TurnId::new("shutdown"), TurnInput::Full(Thread::default()), &options)
		.await
		.expect("turn opens");
	let _ = next_event(&mut shutdown)
		.await
		.expect("accepted before shutdown")
		.expect("accepted");
	assert!(matches!(
		next_event(&mut shutdown).await,
		Some(Err(Error::Protocol("turn stream ended without a terminal event")))
	));
	assert!(next_event(&mut shutdown).await.is_none());
	assert!(matches!(
		shutdown
			.submit(InvokeFrame::Complete(pb::InvokeComplete::default()))
			.await,
		Err(Error::Closed)
	));
}
