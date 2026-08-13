#![cfg(unix)]

use std::{
	collections::VecDeque,
	fs::{self, OpenOptions},
	future::{Future, ready},
	io::Write as _,
	path::{Path, PathBuf},
	pin::Pin,
	process::{Child, Command, ExitStatus, Stdio},
	sync::{
		Arc,
		atomic::{AtomicUsize, Ordering},
	},
	task::{Context, Poll},
	time::{Duration, Instant},
};
use std::os::unix::process::CommandExt as _;

use async_stream::stream;
use bytes::Bytes;
use futures::{Stream, StreamExt as _};
use omp_agent::{
	Agent, AgentError, AgentSnapshot, AgentState, Error as TurnError, InvokeFrame, Journal,
	PromptHash, RpcTurnClient, RpcTurnSession, TurnClient, TurnId, TurnInput, TurnInputRecord,
	TurnOptions, TurnOptionsRecord, TurnSession, TurnStart, project_journal,
};
use omp_app::{
	daemon::{DaemonConfig, DaemonHandle},
	endpoint::LocalEndpoint,
	envd::{server::EnvServer, worker::ToolWorkerConfig},
};
use omp_core::Str;
use omp_e2e::support::omp_binary;
use omp_env::EnvClient;
use omp_llm_catalog::{
	CompiledCatalog, ManagementCapabilities, OperationBits, OperationKind,
	snapshot::{Catalog, SnapshotProvenance},
};
use omp_llm_inference::{
	Answer, Error as InferenceError, Registry as InferenceRegistry,
	call::Call,
	event::{BlockKind, ChatEvent, Completion, FinishReason, WorkflowResponse},
	layer::{LayerCall, stack::RouteProviderService},
	provider::fake::{FakeProvider, FakeScript},
	receipt::{ExecutionReceipt, ReasonId, Usage},
	registry::RouteUnavailable,
	session::ConversationSessionPlanner,
};
use omp_proto::{
	SCHEMA_REV,
	env::v1::ClientHello,
	inference::v1 as pb,
	prost::Message as _,
	thread::v1 as thread,
};
use omp_storage::transcript::{self, AmendPatch, Entry, Header, Kind, SessionId};
use omp_tool::{
	Abort, Constraint, Ev, IncomingParams, Part as ToolPart, PromptCaps, Registry, Rev,
	TOOL_REV_PROP, Tool, ToolSpec,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tower::Service;

const TEST_NAME: &str = "crash_resume_replays_exact_durable_truth";
const CHILD_ENV: &str = "OMP_P6_CHILD";
const ROOT_ENV: &str = "OMP_P6_ROOT";
const ROOT_TURN: &str = "p6-root-turn";
const BATCH_TURN: &str = "p6-batch-turn";
const TOOL_NAME: &str = "p6_hang";


#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
struct OpenRecord {
	turn_id: Str,
	input: TurnInputRecord,
	options: TurnOptionsRecord,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
struct GatewayTurn {
	open: OpenRecord,
	outcome: pb::Outcome,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
struct GatewayState {
	turns: Vec<GatewayTurn>,
	accepted: Vec<bool>,
}

#[derive(Clone)]
struct NeverTurnClient {
	opens: Arc<AtomicUsize>,
}

impl TurnClient for NeverTurnClient {
	type Session<'client> = DiskTurnSession;

	fn turn<'client>(
		&'client self,
		_turn_id: TurnId,
		_input: TurnInput,
		_options: &'client TurnOptions,
	) -> impl Future<Output = Result<Self::Session<'client>, TurnError>> + Send + 'client {
		self.opens.fetch_add(1, Ordering::SeqCst);
		ready(Err(TurnError::Protocol(
			"toolset mismatch must fail before opening the turn client",
		)))
	}
}

#[derive(Clone)]
struct DiskTurnClient {
	path: PathBuf,
}

impl DiskTurnClient {
	fn new(path: PathBuf) -> Self {
		Self { path }
	}

	fn open(
		&self,
		turn_id: TurnId,
		input: TurnInput,
		options: &TurnOptions,
	) -> Result<DiskTurnSession, TurnError> {
		let open = OpenRecord {
			turn_id: turn_id.as_str().into(),
			input: input_record(&input),
			options: options_record(options),
		};
		let mut state = load_gateway(&self.path);
		assert!(
			state.turns.iter().all(|record| record.open.turn_id != open.turn_id),
			"receipt recovery reopened an already terminal scripted provider turn"
		);
		let (outcome, events) = if state.turns.is_empty() {
			assert_eq!(turn_id.as_str(), BATCH_TURN, "initial batch turn id was reminted");
			let outcome = batch_outcome(&input);
			(outcome.clone(), batch_events(outcome))
		} else {
			assert_ne!(turn_id.as_str(), BATCH_TURN, "receipt recovery reopened the gateway turn");
			assert_interrupted_follow_up(&input);
			let outcome = end_outcome(&input, "after interrupted batch");
			(outcome.clone(), vec![accepted(false), outcome_event(outcome)])
		};
		state.turns.push(GatewayTurn { open, outcome });
		state.accepted.push(false);
		store_gateway(&self.path, &state);
		Ok(DiskTurnSession { events: events.into() })
	}
}

impl TurnClient for DiskTurnClient {
	type Session<'client> = DiskTurnSession;

	fn turn<'client>(
		&'client self,
		turn_id: TurnId,
		input: TurnInput,
		options: &'client TurnOptions,
	) -> impl Future<Output = Result<Self::Session<'client>, TurnError>> + Send + 'client {
		ready(self.open(turn_id, input, options))
	}
}

struct DiskTurnSession {
	events: VecDeque<pb::TurnEvent>,
}

struct DiskEvents<'a> {
	session: &'a mut DiskTurnSession,
}

impl Stream for DiskEvents<'_> {
	type Item = Result<pb::TurnEvent, TurnError>;

	fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
		Poll::Ready(self.session.events.pop_front().map(Ok))
	}
}

impl TurnSession for DiskTurnSession {
	fn events(&mut self) -> impl Stream<Item = Result<pb::TurnEvent, TurnError>> + Send + Unpin + '_ {
		DiskEvents { session: self }
	}

	fn submit(&mut self, _frame: InvokeFrame) -> impl Future<Output = Result<(), TurnError>> + Send + '_ {
		ready(Ok(()))
	}
}

struct HangingTool {
	spec: ToolSpec,
	effects: PathBuf,
}

impl HangingTool {
	fn new(effects: PathBuf) -> Self {
		Self {
			spec: ToolSpec {
				name: TOOL_NAME.into(),
				rev: Rev { family: "p6".into(), n: 1 },
				description: "waits forever after its durable effect gate".into(),
				schema: Bytes::from_static(
					br#"{"type":"object","properties":{"call":{"type":"string"}},"required":["call"]}"#,
				),
				constraint: Constraint::None,
			},
			effects,
		}
	}
}

impl Tool for HangingTool {
	type Fault = Value;
	type Params = Value;
	type Payload = Value;
	type Update = Value;

	fn spec(&self) -> &ToolSpec {
		&self.spec
	}

	fn call<'c>(
		&'c self,
		mut params: IncomingParams<'c>,
	) -> impl Stream<Item = Ev<Value, Value, Value>> + Send + 'c {
		stream! {
			match params.committed().await {
				Ok(raw) => {
					let value: Value = serde_json::from_str(raw.as_str()).expect("committed test args");
					let call = value["call"].as_str().expect("call name");
					let mut file = OpenOptions::new()
						.create(true)
						.append(true)
						.open(&self.effects)
						.expect("open effects marker");
					writeln!(file, "{call}").expect("record committed effect");
					file.sync_data().expect("sync committed effect");
					futures::future::pending::<()>().await;
				},
				Err(_) => yield Ev::Aborted(Abort::InputDropped),
			}
		}
	}

	fn prompt(&self, _view: Result<&Value, &Value>, _caps: &PromptCaps) -> Vec<ToolPart> {
		Vec::new()
	}
}

#[derive(Clone)]
struct FakeRoute(FakeProvider);

impl Service<LayerCall<Call>> for FakeRoute {
	type Error = InferenceError;
	type Future = <FakeProvider as Service<Call>>::Future;
	type Response = Answer;

	fn poll_ready(&mut self, context: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		<FakeProvider as Service<Call>>::poll_ready(&mut self.0, context)
	}

	fn call(&mut self, request: LayerCall<Call>) -> Self::Future {
		<FakeProvider as Service<Call>>::call(&mut self.0, request.payload)
	}
}

async fn rpc_host(
	root: &Path,
	first: bool,
) -> (DaemonHandle, RpcTurnClient, String, flume::Receiver<WorkflowResponse>) {
	let mut compiled: CompiledCatalog =
		serde_json::from_str(include_str!("../../llm-catalog/data/catalog.normalized.json"))
			.expect("normalized catalog");
	for provider in &mut compiled.providers {
		provider.management = ManagementCapabilities {
			operations: OperationBits::empty(),
			multiple_accounts: false,
			refresh: false,
			principal_quota: false,
		};
	}
	let artifacts = Catalog::encode(compiled, SnapshotProvenance { source_digest: [0; 32] })
		.expect("P6 catalog snapshot");
	let catalog = Arc::new(Catalog::decode(&artifacts.postcard).expect("P6 catalog"));
	let model = catalog
		.models()
		.iter()
		.find(|candidate| candidate.capabilities.operations.contains_kind(OperationKind::Chat))
		.expect("chat model");
	let route_id = model.routes.first().expect("chat route").clone();
	let route = catalog.route(&route_id).expect("catalog route");
	let fake = FakeProvider::new(route.provider.clone(), route_id.clone());
	if first {
		fake.extend([FakeScript::chat(vec![
			Ok(ChatEvent::BlockStarted { index: 0, kind: BlockKind::Text }),
			Ok(ChatEvent::TextDelta {
				index: 0,
				text: Str::from("the durable RPC outcome"),
			}),
			Ok(ChatEvent::Completed(Completion {
				reason: FinishReason::Stop,
				blocks: 1,
				usage: Usage::default(),
				receipt: ExecutionReceipt::default(),
			})),
		])]);
	}
	let route_service = RouteProviderService::new(FakeRoute(fake));
	let mut builder = InferenceRegistry::builder(Arc::clone(&catalog));
	for candidate in catalog.routes() {
		builder = if candidate.id == route_id {
			builder
				.register_route(candidate.id.clone(), route_service.clone())
				.expect("register P6 fake route")
		} else {
			builder
				.register_unavailable(RouteUnavailable {
					route: candidate.id.clone(),
					reason: ReasonId(Str::from("p6-route-unavailable")),
					operation: None,
				})
				.expect("register unavailable route")
		};
	}
	let registry = builder.build().expect("P6 inference registry");
	let sessions = ConversationSessionPlanner::open(root.join("sessions.db"), Arc::clone(&catalog))
		.expect("open durable conversation store");
	let socket = root.join(if first { "gateway-first.sock" } else { "gateway-resume.sock" });
	let (responses_tx, responses_rx) = flume::bounded(8);
	let daemon = DaemonHandle::start_for_test(
		DaemonConfig::local(LocalEndpoint::from(socket.clone())).with_data_dir(root.join("gateway-data")),
		registry,
		sessions,
		Arc::new(Registry::new()),
		responses_tx,
	)
	.await
	.expect("start real RPC gateway");
	let channel = omp_rpc::uds::connect(&socket).await.expect("connect real RPC gateway");
	(daemon, RpcTurnClient::new(channel), model.key.as_str().to_owned(), responses_rx)
}

async fn next_rpc(session: &mut RpcTurnSession) -> pb::TurnEvent {
	tokio::time::timeout(Duration::from_secs(3), async {
		session
			.events()
			.next()
			.await
			.expect("RPC turn stream ended")
			.expect("RPC turn event failed")
	})
	.await
	.expect("RPC turn event timed out")
}

async fn rpc_outcome(session: &mut RpcTurnSession) -> pb::Outcome {
	loop {
		let event = next_rpc(session).await;
		if let Some(pb::turn_event::Event::Outcome(outcome)) = event.event {
			return outcome;
		}
	}
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn crash_resume_replays_exact_durable_truth() {
	if let (Ok(stage), Ok(root)) = (std::env::var(CHILD_ENV), std::env::var(ROOT_ENV)) {
		run_child(&stage, Path::new(&root)).await;
		return;
	}

	let scratch = tempfile::tempdir().expect("P6 scratch directory");

	let replay_root = scratch.path().join("replay");
	fs::create_dir(&replay_root).expect("replay scenario directory");
	let accepted_marker = replay_root.join("accepted");
	let mut first = spawn_child("replay-crash", &replay_root);
	wait_for_file(&accepted_marker).await;
	kill_at_boundary(&mut first);
	let resumed = run_child_process("replay-resume", &replay_root).await;
	assert!(resumed.success(), "resume child failed: {resumed}");
	assert_single_receipt(&replay_root.join("journal.jsonl"), ROOT_TURN);

	let patch_root = scratch.path().join("receipt-patch");
	fs::create_dir(&patch_root).expect("receipt-patch scenario directory");
	let receipt_marker = patch_root.join("receipt");
	let mut receipt = spawn_child("receipt-crash", &patch_root);
	wait_for_file(&receipt_marker).await;
	kill_at_boundary(&mut receipt);
	let patched = run_child_process("receipt-resume", &patch_root).await;
	assert!(patched.success(), "receipt recovery child failed: {patched}");
	let patch_state = load_gateway(&patch_root.join("gateway.json"));
	assert_eq!(patch_state.turns.len(), 1, "sequence recovery opened another gateway turn");
	assert_eq!(patch_state.accepted, vec![false]);
	assert_recovered_sequences(&patch_root.join("journal.jsonl"));

	let batch_root = scratch.path().join("batch");
	fs::create_dir(&batch_root).expect("batch scenario directory");
	let effects = batch_root.join("effects");
	let mut batch = spawn_child("batch-crash", &batch_root);
	wait_for_lines(&effects, 2).await;
	kill_at_boundary(&mut batch);
	assert_effects(&effects);
	let completed = run_child_process("batch-resume", &batch_root).await;
	assert!(completed.success(), "batch recovery child failed: {completed}");
	assert_effects(&effects);
	assert_batch_recovery(&batch_root.join("journal.jsonl"), &batch_root.join("gateway.json"));
}

#[tokio::test]
async fn resume_rejects_changed_toolset_before_opening_any_authority() {
	let scratch = tempfile::tempdir().expect("toolset mismatch scratch directory");
	let journal_path = scratch.path().join("journal.jsonl");
	let mut journal = Journal::create(&journal_path, &header(scratch.path(), "toolset-mismatch"))
		.expect("create toolset mismatch journal");
	let hash = PromptHash::from([19; 32]);
	let user = message(thread::Role::User, "resume only with the frozen toolset");
	let input_event = journal
		.append_optimistic(1, user.clone(), Some(hash))
		.expect("append pending turn input");
	let durable_registry = Registry::new();
	let input = TurnInput::Full(thread::Thread { items: vec![user] });
	let options = TurnOptions::default();
	journal
		.start_turn(2, TurnStart {
			turn_id: "toolset-mismatch-turn".into(),
			item_events: vec![input_event],
			prompt_hash: hash.into_bytes(),
			prompt_head_events: Vec::new(),
			toolset_hash: durable_registry.live_hash(),
			enabled_tools: Vec::new(),
			sequence_targets: vec![input_event],
			input: input_record(&input),
			options: options_record(&options),
		})
		.expect("record pending turn under original toolset");

	let mut changed_registry = Registry::new();
	changed_registry
		.register(HangingTool::new(scratch.path().join("must-not-run")))
		.expect("register changed live tool");
	let mut snapshot =
		AgentSnapshot::new(TurnOptions::default(), Default::default(), Arc::new(changed_registry));
	snapshot.enabled_tools = Arc::from([Str::from(TOOL_NAME)]);
	let opens = Arc::new(AtomicUsize::new(0));
	let client = NeverTurnClient { opens: Arc::clone(&opens) };
	let (env, transport) = EnvClient::in_process(8);
	let (environment_requests, _environment_responses) = transport.into_parts();
	let mut agent = Agent::new(client, env, AgentState::new(snapshot), journal, caps());

	let error = agent
		.submit(Vec::<thread::Item>::new(), TurnId::new("ignored-resume-root"))
		.await
		.expect_err("changed toolset must reject resume");
	assert!(matches!(error, AgentError::ToolsetMismatch { .. }));
	assert_eq!(opens.load(Ordering::SeqCst), 0, "turn client opened before mismatch rejection");
	assert!(environment_requests.is_empty(), "environment opened before mismatch rejection");
	assert!(!scratch.path().join("must-not-run").exists(), "changed tool effect was launched");
}

async fn run_child(stage: &str, root: &Path) {
	match stage {
		"replay-crash" => replay_child(root, true, false).await,
		"replay-resume" => replay_child(root, false, true).await,
		"receipt-crash" => receipt_child(root, true),
		"receipt-resume" => receipt_child(root, false),
		"batch-crash" => batch_child(root, true).await,
		"batch-resume" => batch_child(root, false).await,
		other => panic!("unknown P6 child stage {other}"),
	}
}

async fn replay_child(root: &Path, create: bool, _mutated: bool) {
	let journal_path = root.join("journal.jsonl");
	let (_daemon, client, model, _responses) = rpc_host(root, create).await;
	if create {
		let mut journal =
			Journal::create(&journal_path, &header(root, "replay")).expect("create replay journal");
		let hash = PromptHash::from([11; 32]);
		let prompt = message(thread::Role::System, "durable RPC prompt");
		let user = message(thread::Role::User, "survive this RPC host crash");
		let prompt_event = journal
			.append_optimistic(1, prompt.clone(), Some(hash))
			.expect("append durable prompt");
		let input_event = journal
			.append_optimistic(2, user.clone(), Some(hash))
			.expect("append durable input");
		let input = TurnInput::Full(thread::Thread { items: vec![prompt, user] });
		let options = TurnOptions {
			context_id: Some("durable-rpc-context".into()),
			params: pb::ChatParams { model, ..pb::ChatParams::default() },
			..TurnOptions::default()
		};
		journal
			.start_turn(3, TurnStart {
				turn_id: ROOT_TURN.into(),
				item_events: vec![input_event],
				prompt_hash: hash.into_bytes(),
				prompt_head_events: vec![prompt_event],
				toolset_hash: Registry::new().live_hash(),
				enabled_tools: Vec::new(),
				sequence_targets: vec![prompt_event, input_event],
				input: input_record(&input),
				options: options_record(&options),
			})
			.expect("durable TurnStart before RPC");
		let mut session = client
			.turn(TurnId::new(ROOT_TURN), input, &options)
			.await
			.expect("open real RPC turn");
		assert!(matches!(
			next_rpc(&mut session).await.event,
			Some(pb::turn_event::Event::Accepted(pb::Accepted { replay: false }))
		));
		let outcome = rpc_outcome(&mut session).await;
		fs::write(root.join("rpc-outcome.bin"), outcome.encode_to_vec())
			.expect("persist expected RPC outcome");
		write_marker(&root.join("accepted"));
		loop {
			std::thread::park();
		}
	} else {
		let mut journal = Journal::open(&journal_path).expect("reopen pending RPC journal");
		let start = journal.pending_turn().cloned().expect("pending durable TurnStart");
		let poison = TurnOptions {
			context_id: Some("poison-context".into()),
			params: pb::ChatParams { model: "poison/model".to_owned(), ..pb::ChatParams::default() },
			..TurnOptions::default()
		};
		assert_ne!(start.options, options_record(&poison), "fixture must distinguish mutable state");
		let input = restore_input(&start.input);
		let options = restore_options(&start.options);
		let mut session = client
			.turn(TurnId::new(start.turn_id.clone()), input, &options)
			.await
			.expect("resubmit exact durable RPC turn");
		assert!(matches!(
			next_rpc(&mut session).await.event,
			Some(pb::turn_event::Event::Accepted(pb::Accepted { replay: true }))
		));
		let outcome = rpc_outcome(&mut session).await;
		let expected = pb::Outcome::decode(
			fs::read(root.join("rpc-outcome.bin"))
				.expect("read first host outcome")
				.as_slice(),
		)
		.expect("decode first host outcome");
		assert_eq!(outcome, expected, "RPC replay outcome changed bytes");
		journal
			.append_gateway_outcome(4, start.turn_id.as_str(), outcome.clone())
			.expect("journal replayed RPC outcome once");
		patch_sequences(&mut journal, &start, &outcome);
	}
}

fn receipt_child(root: &Path, crash: bool) {
	let journal_path = root.join("journal.jsonl");
	if crash {
		let mut journal = Journal::create(&journal_path, &header(root, "receipt"))
			.expect("create receipt journal");
		let hash = PromptHash::from([7; 32]);
		let prompt = journal
			.append_optimistic(1, message(thread::Role::System, "fixed prompt"), Some(hash))
			.expect("append prompt");
		let input = journal
			.append_optimistic(2, message(thread::Role::User, "fixed input"), Some(hash))
			.expect("append input");
		let full = thread::Thread {
			items: vec![
				message(thread::Role::System, "fixed prompt"),
				message(thread::Role::User, "fixed input"),
			],
		};
		let options = TurnOptions { context_id: Some("receipt-context".into()), ..Default::default() };
		let open = OpenRecord {
			turn_id: "receipt-turn".into(),
			input: TurnInputRecord::Full { thread: full.clone() },
			options: options_record(&options),
		};
		let outcome = end_outcome(&TurnInput::Full(full), "receipt answer");
		journal
			.start_turn(3, TurnStart {
				turn_id: "receipt-turn".into(),
				item_events: vec![input],
				prompt_hash: hash.into_bytes(),
				prompt_head_events: vec![prompt],
				toolset_hash: Registry::new().live_hash(),
				enabled_tools: Vec::new(),
				sequence_targets: vec![prompt, input],
				input: open.input.clone(),
				options: open.options.clone(),
			})
			.expect("durable turn start");
		store_gateway(&root.join("gateway.json"), &GatewayState {
			turns: vec![GatewayTurn { open, outcome: outcome.clone() }],
			accepted: vec![false],
		});
		journal
			.append_gateway_outcome(4, "receipt-turn", outcome)
			.expect("durable terminal receipt");
		write_marker(&root.join("receipt"));
		loop {
			std::thread::park();
		}
	} else {
		let journal = Journal::open(&journal_path).expect("recover missing sequence amendments");
		let first = fs::read(&journal_path).expect("read recovered journal");
		drop(journal);
		let reopened = Journal::open(&journal_path).expect("reopen recovered journal");
		drop(reopened);
		assert_eq!(fs::read(&journal_path).expect("read stable journal"), first);
	}
}

async fn batch_child(root: &Path, create: bool) {
	let journal_path = root.join("journal.jsonl");
	let journal = if create {
		Journal::create(&journal_path, &header(root, "batch")).expect("create batch journal")
	} else {
		Journal::open(&journal_path).expect("open batch journal")
	};
	let mut agent_registry = Registry::new();
	agent_registry
		.register(HangingTool::new(root.join("effects")))
		.expect("register agent hanging tool");
	let agent_registry = Arc::new(agent_registry);
	let mut environment_registry = Registry::new();
	environment_registry
		.register(HangingTool::new(root.join("effects")))
		.expect("register environment hanging tool");
	let state_dir = root.join("env-state");
	let workspace = root.join("workspace");
	fs::create_dir_all(&state_dir).expect("environment state directory");
	fs::create_dir_all(&workspace).expect("environment workspace directory");
	let server = Arc::new(
		EnvServer::open_local(
			&workspace,
			&state_dir,
			environment_registry,
			ToolWorkerConfig::new(omp_binary().expect("worker-capable host binary")),
		)
		.await
		.expect("real local environment host"),
	);
	let (env, transport) = EnvClient::in_process(64);
	let host = Arc::clone(&server);
	let server_task = tokio::spawn(async move { host.serve_in_process(transport).await });
	env.hello(ClientHello {
		client: "p6-crash-resume".to_owned(),
		schema_rev: SCHEMA_REV,
		..ClientHello::default()
	})
	.await
	.expect("environment handshake");
	let options = TurnOptions { context_id: Some("batch-context".into()), ..Default::default() };
	let mut snapshot = AgentSnapshot::new(options, Default::default(), agent_registry);
	snapshot.enabled_tools = Arc::from([Str::from(TOOL_NAME)]);
	let client = DiskTurnClient::new(root.join("gateway.json"));
	let mut agent = Agent::new(client, env, AgentState::new(snapshot), journal, caps());
	let items = if create { vec![message(thread::Role::User, "run the durable batch")] } else { Vec::new() };
	let result = agent.submit(items, TurnId::new(if create { BATCH_TURN } else { "unused-resume-root" })).await;
	if create {
		let _ = result.expect("batch remains live until parent kills this host");
		panic!("hanging tool batch unexpectedly completed");
	} else {
		let summary = result.expect("recover interrupted batch and proceed");
		assert_eq!(summary.outcome.provider, "p6-gateway");
		assert_eq!(summary.outcome.stop(), pb::StopReason::StopEndTurn);
	}
	server_task.abort();
}

fn spawn_child(stage: &str, root: &Path) -> Child {
	let mut command = Command::new(std::env::current_exe().expect("current P6 test executable"));
	command.process_group(0);
	command
		.arg(TEST_NAME)
		.arg("--exact")
		.arg("--nocapture")
		.arg("--test-threads=1")
		.env(CHILD_ENV, stage)
		.env(ROOT_ENV, root)
		.stdin(Stdio::null())
		.stdout(Stdio::inherit())
		.stderr(Stdio::inherit());
	command.spawn().expect("spawn child host process")
}

async fn run_child_process(stage: &str, root: &Path) -> ExitStatus {
	let mut child = spawn_child(stage, root);
	let deadline = Instant::now() + Duration::from_secs(15);
	loop {
		if let Some(status) = child.try_wait().expect("query child host") {
			return status;
		}
		if Instant::now() >= deadline {
			kill_process_group(&mut child);
			let _ = child.wait();
			panic!("child stage {stage} exceeded its bounded deadline");
		}
		tokio::time::sleep(Duration::from_millis(20)).await;
	}
}

fn kill_at_boundary(child: &mut Child) {
	kill_process_group(child);
	let status = child.wait().expect("reap killed child");
	assert!(!status.success(), "crash boundary child exited cleanly before kill");
}

fn kill_process_group(child: &mut Child) {
	if let Ok(group) = i32::try_from(child.id()) {
		let _ = nix::sys::signal::killpg(
			nix::unistd::Pid::from_raw(group),
			Some(nix::sys::signal::Signal::SIGKILL),
		);
		return;
	}
	let _ = child.kill();
}

async fn wait_for_file(path: &Path) {
	let deadline = Instant::now() + Duration::from_secs(10);
	while !path.exists() {
		assert!(Instant::now() < deadline, "timed out waiting for {}", path.display());
		tokio::time::sleep(Duration::from_millis(10)).await;
	}
}

async fn wait_for_lines(path: &Path, count: usize) {
	let deadline = Instant::now() + Duration::from_secs(10);
	loop {
		let observed = fs::read_to_string(path)
			.map(|text| text.lines().count())
			.unwrap_or_default();
		if observed >= count {
			return;
		}
		assert!(Instant::now() < deadline, "timed out waiting for {count} durable effects");
		tokio::time::sleep(Duration::from_millis(10)).await;
	}
}

fn assert_single_receipt(path: &Path, turn_id: &str) {
	let journal = Journal::open(path).expect("open completed replay journal");
	let log = journal.load().expect("load completed replay journal");
	let receipts = event_count(&log, |kind| matches!(kind, Kind::TurnReceipt(receipt) if receipt.turn_id == turn_id));
	assert_eq!(receipts, 1, "terminal receipt duplicated");
	let receipt = journal.receipt(turn_id).expect("root turn receipt");
	assert_eq!(receipt.outcome.output.len(), 1);
	let projected = project_journal(&log, &Registry::new(), &caps()).expect("project replay journal");
	assert_eq!(projected.items.iter().filter(|item| item.seq == receipt.outcome.output[0].seq).count(), 1);
}

fn assert_recovered_sequences(path: &Path) {
	let journal = Journal::open(path).expect("open sequence-recovered journal");
	let log = journal.load().expect("load sequence-recovered journal");
	let amendments = event_count(&log, |kind| matches!(kind, Kind::Amend { patch: AmendPatch::Seq { .. }, .. }));
	assert_eq!(amendments, 2, "sequence recovery duplicated or omitted amendments");
	let projected = project_journal(&log, &Registry::new(), &caps()).expect("project recovered journal");
	let seqs: Vec<_> = projected.items.iter().map(|item| item.seq).collect();
	assert_eq!(seqs, vec![1, 2, 3], "recovered sequence assignment drifted");
}

fn assert_effects(path: &Path) {
	let mut effects: Vec<_> = fs::read_to_string(path)
		.expect("read durable effects")
		.lines()
		.map(str::to_owned)
		.collect();
	effects.sort();
	assert_eq!(effects, vec!["durable-a", "durable-b"]);
}

fn assert_batch_recovery(journal_path: &Path, gateway_path: &Path) {
	let state = load_gateway(gateway_path);
	assert_eq!(state.turns.len(), 2, "recovery performed an extra gateway turn");
	assert_eq!(state.accepted, vec![false, false], "receipt recovery replayed the terminal gateway turn");
	assert_ne!(state.turns[0].open.turn_id, state.turns[1].open.turn_id);
	let journal = Journal::open(journal_path).expect("open recovered batch journal");
	let log = journal.load().expect("load recovered batch journal");
	let mut registry = Registry::new();
	registry
		.register(HangingTool::new(journal_path.with_extension("unused-effects")))
		.expect("register projection tool");
	let projected = project_journal(&log, &registry, &caps()).expect("project interrupted batch");
	let mut result_ids = Vec::new();
	for item in &projected.items {
		if let Some(thread::item::Kind::ToolResult(result)) = &item.kind {
			result_ids.push(result.call_id.as_str());
			assert!(result.is_error, "synthesized interrupted result must be an error");
		}
	}
	result_ids.sort();
	assert_eq!(result_ids, vec!["durable-a", "durable-b"]);
	let mut nonzero: Vec<_> = projected.items.iter().map(|item| item.seq).filter(|seq| *seq != 0).collect();
	let expected: Vec<_> = (1..=u64::try_from(nonzero.len()).expect("item count")).collect();
	assert_eq!(nonzero, expected, "recovery introduced duplicate or drifting sequences");
	nonzero.clear();
}

fn assert_interrupted_follow_up(input: &TurnInput) {
	let mut ids = Vec::new();
	for item in input_items(input) {
		if let Some(thread::item::Kind::ToolResult(result)) = &item.kind {
			assert!(result.is_error, "unfinished call did not synthesize an interrupted error");
			ids.push(result.call_id.as_str());
		}
	}
	ids.sort();
	assert_eq!(ids, vec!["durable-a", "durable-b"], "recovery duplicated, omitted, or invented tool results");
}

fn event_count(log: &transcript::Log, predicate: impl Fn(&Kind) -> bool) -> usize {
	(0..u64::try_from(log.len()).expect("log length"))
		.filter(|index| matches!(log.get(*index), Some(Entry::Ok(event)) if predicate(&event.kind)))
		.count()
}

fn input_record(input: &TurnInput) -> TurnInputRecord {
	match input {
		TurnInput::Full(thread) => TurnInputRecord::Full { thread: thread.clone() },
		TurnInput::Delta(context, delta) => TurnInputRecord::Delta {
			context: context.clone(),
			delta: delta.clone(),
		},
	}
}

fn options_record(options: &TurnOptions) -> TurnOptionsRecord {
	TurnOptionsRecord {
		context_id: options.context_id.clone(),
		params: options.params.clone(),
		executor: options.executor.clone(),
		props: options.props.clone(),
	}
}

fn restore_input(record: &TurnInputRecord) -> TurnInput {
	match record {
		TurnInputRecord::Full { thread } => TurnInput::Full(thread.clone()),
		TurnInputRecord::Delta { context, delta } => TurnInput::Delta(context.clone(), delta.clone()),
	}
}

fn restore_options(record: &TurnOptionsRecord) -> TurnOptions {
	TurnOptions {
		context_id: record.context_id.clone(),
		params: record.params.clone(),
		executor: record.executor.clone(),
		props: record.props.clone(),
	}
}

fn patch_sequences(journal: &mut Journal, start: &TurnStart, outcome: &pb::Outcome) {
	let Some(revision) = outcome.revision.as_ref() else {
		return;
	};
	let first = revision.head
		.checked_sub(u64::try_from(outcome.output.len()).expect("output length"))
		.and_then(|head| head.checked_add(1))
		.and_then(|output| {
			output.checked_sub(u64::try_from(start.sequence_targets.len()).expect("input length"))
		})
		.expect("valid receipt sequence range");
	for (offset, target) in start.sequence_targets.iter().enumerate() {
		journal
			.amend_seq(
				5,
				*target,
				first + u64::try_from(offset).expect("sequence offset"),
			)
			.expect("patch replay input sequence");
	}
}

fn input_items(input: &TurnInput) -> &[thread::Item] {
	match input {
		TurnInput::Full(thread) => &thread.items,
		TurnInput::Delta(_, delta) => &delta.append,
	}
}

fn input_head(input: &TurnInput) -> u64 {
	match input {
		TurnInput::Full(thread) => u64::try_from(thread.items.len()).expect("thread length"),
		TurnInput::Delta(context, delta) => {
			let expected = context.expected.as_ref().expect("delta revision");
			delta.truncate_to.unwrap_or(expected.head)
				+ u64::try_from(delta.append.len()).expect("delta length")
		},
	}
}

fn end_outcome(input: &TurnInput, text: &str) -> pb::Outcome {
	let head = input_head(input);
	pb::Outcome {
		output: vec![thread::Item {
			seq: head + 1,
			created_at_ms: 9,
			kind: Some(thread::item::Kind::Message(thread::Message {
				role: thread::Role::Assistant as i32,
				parts: vec![thread::Part { kind: Some(thread::part::Kind::Text(text.to_owned())) }],
			})),
			props: None,
		}],
		stop: pb::StopReason::StopEndTurn as i32,
		revision: Some(revision(head + 1)),
		provider: "p6-gateway".to_owned(),
		model: "p6-model".to_owned(),
		..pb::Outcome::default()
	}
}

fn batch_outcome(input: &TurnInput) -> pb::Outcome {
	let head = input_head(input);
	pb::Outcome {
		output: vec![tool_call(head + 1, "durable-a"), tool_call(head + 2, "durable-b")],
		stop: pb::StopReason::StopToolUse as i32,
		revision: Some(revision(head + 2)),
		provider: "p6-gateway".to_owned(),
		model: "p6-model".to_owned(),
		..pb::Outcome::default()
	}
}

fn batch_events(outcome: pb::Outcome) -> Vec<pb::TurnEvent> {
	let mut events = vec![accepted(false)];
	for (index, id) in [(0, "durable-a"), (1, "durable-b"), (2, "ghost-absent")] {
		events.push(event(pb::turn_event::Event::PartStart(pb::PartStart {
			index,
			kind: pb::part_start::Kind::ToolCall as i32,
			tool_call_id: id.to_owned(),
			tool_name: TOOL_NAME.to_owned(),
		})));
		events.push(event(pb::turn_event::Event::PartDelta(pb::PartDelta {
			index,
			chunk: Bytes::from(format!(r#"{{"call":"{id}"}}"#)),
		})));
	}
	events.push(outcome_event(outcome));
	events
}

fn tool_call(seq: u64, id: &str) -> thread::Item {
	thread::Item {
		seq,
		created_at_ms: 8,
		kind: Some(thread::item::Kind::ToolCall(thread::ToolCall {
			id: id.to_owned(),
			name: TOOL_NAME.to_owned(),
			args_json: Bytes::from(format!(r#"{{"call":"{id}"}}"#)),
			..thread::ToolCall::default()
		})),
		props: Some(pb::ValueMap {
			fields: [(TOOL_REV_PROP.to_owned(), pb::Value {
				kind: Some(pb::value::Kind::String("p6.1".to_owned())),
			})]
			.into_iter()
			.collect(),
		}),
	}
}

fn message(role: thread::Role, text: &str) -> thread::Item {
	thread::Item {
		kind: Some(thread::item::Kind::Message(thread::Message {
			role: role as i32,
			parts: vec![thread::Part { kind: Some(thread::part::Kind::Text(text.to_owned())) }],
		})),
		..thread::Item::default()
	}
}

fn accepted(replay: bool) -> pb::TurnEvent {
	event(pb::turn_event::Event::Accepted(pb::Accepted { replay }))
}

fn outcome_event(outcome: pb::Outcome) -> pb::TurnEvent {
	event(pb::turn_event::Event::Outcome(outcome))
}

fn event(event: pb::turn_event::Event) -> pb::TurnEvent {
	pb::TurnEvent { event: Some(event) }
}

fn revision(head: u64) -> thread::Revision {
	thread::Revision { head, token: Bytes::from(vec![u8::try_from(head % 251).expect("token byte"); 32]) }
}

fn caps() -> PromptCaps {
	PromptCaps { maximum_parts: 8, maximum_text_bytes: 4096, media: false }
}

fn header(root: &Path, id: &str) -> Header {
	Header { v: 4, id: SessionId(Str::from(id)), created: 1, cwd: root.to_owned() }
}

fn load_gateway(path: &Path) -> GatewayState {
	match fs::read(path) {
		Ok(bytes) => serde_json::from_slice(&bytes).expect("decode durable gateway state"),
		Err(error) if error.kind() == std::io::ErrorKind::NotFound => GatewayState::default(),
		Err(error) => panic!("read durable gateway state: {error}"),
	}
}

fn store_gateway(path: &Path, state: &GatewayState) {
	let bytes = serde_json::to_vec(state).expect("encode durable gateway state");
	let temporary = path.with_extension("tmp");
	fs::write(&temporary, bytes).expect("write temporary gateway state");
	OpenOptions::new()
		.read(true)
		.open(&temporary)
		.expect("open temporary gateway state")
		.sync_all()
		.expect("sync temporary gateway state");
	fs::rename(&temporary, path).expect("publish durable gateway state");
}

fn write_marker(path: &Path) {
	fs::write(path, b"ready").expect("write crash-boundary marker");
	OpenOptions::new()
		.read(true)
		.open(path)
		.expect("open crash-boundary marker")
		.sync_all()
		.expect("sync crash-boundary marker");
}
