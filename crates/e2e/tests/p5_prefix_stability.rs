#![feature(impl_trait_in_assoc_type)]

use std::{
	collections::BTreeMap,
	future::Future,
	num::NonZeroUsize,
	sync::Arc,
	time::Duration,
};

use bytes::Bytes;
use futures::{FutureExt as _, Stream};
use omp_agent::{
	Agent, AgentSnapshot, AgentState, ContextFile, InProcTurnClient, Journal, TurnClient, TurnId,
	TurnInput, TurnOptions, WorkspaceInput,
};
use omp_app::rpc_adapter::InferenceRpc;
use omp_core::Str;
use omp_e2e::support::{Scratch, user_item, within};
use omp_llm_catalog::{CompiledCatalog, OperationKind, snapshot::{Catalog, SnapshotProvenance}};
use omp_llm_inference::{
	AccountPool, Error, ErrorKind, ErrorPhase, ExecutionReceipt, Registry, RetryAction,
	account::AccountSummary,
	auth::{
		AuthLoginEngine, AuthManager, AuthRefreshEngine, CredentialBroker, CredentialBrokerEngines,
		CredentialStore, HeadlessKeySource, KeyId,
	},
	call::{AuthMethod, LoginRequest},
	answer::AuthSession,
	codec::{google_cca::{AntigravityFingerprint, AntigravityPolicy, CcaHeaders}, openai_chat::OpenAiChatCodec},
	layer::{admission::AdmissionController, stack::BuiltinConfig},
	provider::builtin::{AuthApplicationConfig, GoogleCcaConfig, LocalRouteBackend, ProductionDependencies},
	session::{ConversationSessionPlanner, InMemoryConversationStore},
	transport::{
		Frame, SseEvent,
		cassette::{CassetteAttempt, CassetteBodyAction, CassetteTerminal, CassetteTransport},
		http::HttpTransport,
		websocket_transport::WebSocketTransport,
	},
};
use omp_proto::{inference::v1 as pb, prost::Message as _, thread::v1 as thread};
use omp_storage::transcript::{Header, SessionId};
use omp_tool::{Constraint, Ev, IncomingParams, Part, PromptCaps, Rev, Tool, ToolSpec};
use parking_lot::Mutex;

const MODEL: &str = "apple-intelligence/apple-intelligence";
const ROUTE: &str = "route-15d4d866935964367e95fddfe4b98065053b172594f9334bdbe6e6cca7123886";
const BODY_LIMIT: usize = 1024 * 1024;

#[derive(Clone)]
struct Instrumented<C> {
	inner: C,
	turns: Arc<Mutex<Vec<CapturedInput>>>,
}

#[derive(Clone, Debug)]
struct CapturedInput {
	input: TurnInput,
	options: TurnOptions,
}

impl<C> Instrumented<C> {
	fn new(inner: C) -> Self {
		Self { inner, turns: Arc::new(Mutex::new(Vec::new())) }
	}

	fn captures(&self) -> Vec<CapturedInput> {
		self.turns.lock().clone()
	}
}

impl<C: TurnClient> TurnClient for Instrumented<C> {
	type Session<'client> = C::Session<'client>;

	fn turn<'client>(
		&'client self,
		turn_id: TurnId,
		input: TurnInput,
		options: &'client TurnOptions,
	) -> impl Future<Output = Result<Self::Session<'client>, omp_agent::Error>> + Send + 'client {
		self.turns.lock().push(CapturedInput { input: input.clone(), options: options.clone() });
		self.inner.turn(turn_id, input, options)
	}
}

struct RevisionTool {
	spec: ToolSpec,
}

impl Tool for RevisionTool {
	type Params = serde_json::Value;
	type Update = serde_json::Value;
	type Payload = serde_json::Value;
	type Fault = serde_json::Value;

	fn spec(&self) -> &ToolSpec {
		&self.spec
	}

	fn call<'c>(
		&'c self,
		_params: IncomingParams<'c>,
	) -> impl Stream<Item = Ev<Self::Update, Self::Payload, Self::Fault>> + Send + 'c {
		futures::stream::empty()
	}

	fn prompt(&self, _view: Result<&Self::Payload, &Self::Fault>, _caps: &PromptCaps) -> Vec<Part> {
		Vec::new()
	}
}

#[derive(Clone, Copy)]
struct UnusedLogin(AuthMethod);

impl AuthLoginEngine for UnusedLogin {
	fn method(&self) -> AuthMethod {
		self.0
	}

	fn begin(
		&self,
		_request: LoginRequest,
		_spec: omp_llm_catalog::AuthSpecId,
	) -> futures::future::BoxFuture<'_, Result<AuthSession, Error>> {
		async { Err(unused_auth_error()) }.boxed()
	}
}

struct UnusedRefresh;

impl AuthRefreshEngine for UnusedRefresh {
	fn refresh(
		&self,
		_account: omp_llm_inference::AccountId,
	) -> futures::future::BoxFuture<'_, Result<AccountSummary, Error>> {
		async { Err(unused_auth_error()) }.boxed()
	}
}

fn unused_auth_error() -> Error {
	Error::new(
		ErrorKind::InternalInvariant,
		ErrorPhase::Authentication,
		RetryAction::Never,
		ExecutionReceipt::default(),
	)
}

fn tool_registry(revision: u16) -> Arc<omp_tool::Registry> {
	let mut registry = omp_tool::Registry::new();
	registry
		.register(RevisionTool {
			spec: ToolSpec {
				name: "probe".into(),
				rev: Rev { family: "json".into(), n: revision },
				description: format!("prefix probe revision {revision}").into(),
				schema: Bytes::from(format!(
					r#"{{"type":"object","properties":{{"revision":{{"const":{revision}}}}}}}}"#
				)),
				constraint: Constraint::None,
			},
		})
		.expect("register revision tool");
	Arc::new(registry)
}

fn tool_def(revision: u16) -> pb::ToolDef {
	pb::ToolDef {
		name: "probe".to_owned(),
		description: format!("prefix probe revision {revision}"),
		schema_json: Bytes::from(format!(
			r#"{{"type":"object","properties":{{"revision":{{"const":{revision}}}}}}}}"#
		)),
		strict: None,
	}
}

fn catalog() -> Arc<Catalog> {
	let mut value: serde_json::Value = serde_json::from_str(include_str!(
		"../../llm-catalog/data/catalog.normalized.json"
	))
	.expect("normalized catalog JSON");
	let model = value["models"]
		.as_array_mut()
		.expect("models array")
		.iter_mut()
		.find(|model| model["key"] == MODEL)
		.expect("offline local model");
	model["capabilities"]["chat"]["tools"] = serde_json::json!({
		"native": { "features": 0, "maximum_tools": null }
	});
	let compiled: CompiledCatalog = serde_json::from_value(value).expect("modified test catalog");
	let artifacts = Catalog::encode(compiled, SnapshotProvenance { source_digest: [0; 32] })
		.expect("encode test catalog");
	Arc::new(Catalog::decode(&artifacts.postcard).expect("decode test catalog"))
}

fn cassette_attempt() -> CassetteAttempt {
	CassetteAttempt {
		status: Some(200),
		headers: Box::new([]),
		provider_request_id: Some(Str::from("p5-cassette")),
		body: CassetteBodyAction::Drain,
		frames: vec![
			Frame::Sse(SseEvent {
				name: None,
				data: Bytes::from_static(
					br#"{"id":"p5","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
				),
			}),
			Frame::Sse(SseEvent { name: None, data: Bytes::from_static(b"[DONE]") }),
		]
		.into_boxed_slice(),
		terminal: CassetteTerminal::Complete,
	}
}

fn auth_manager(
	catalog: Arc<Catalog>,
	path: &std::path::Path,
	broker: CredentialBroker,
	accounts: AccountPool,
) -> AuthManager {
	let store = Arc::new(
		CredentialStore::open(
			path,
			Arc::new(HeadlessKeySource::new(KeyId::new("p5-e2e"), [7; 32])),
		)
		.expect("credential store"),
	);
	let login = [
		AuthMethod::ApiKey,
		AuthMethod::OAuthPkce,
		AuthMethod::OAuthDevice,
		AuthMethod::ApplicationDefault,
		AuthMethod::AwsCredentialChain,
		AuthMethod::SessionToken,
	]
	.into_iter()
	.map(|method| Arc::new(UnusedLogin(method)) as Arc<dyn AuthLoginEngine>)
	.collect();
	AuthManager::new(catalog, store, broker, accounts, login, Arc::new(UnusedRefresh))
		.expect("test auth manager")
}

async fn gateway(
	scratch: &Scratch,
	cassette: CassetteTransport,
	tools: Arc<omp_tool::Registry>,
) -> InProcTurnClient {
	let catalog = catalog();
	let broker = CredentialBroker::system(&catalog, CredentialBrokerEngines::default())
		.expect("credential broker");
	let accounts = AccountPool::new();
	let auth = auth_manager(
		Arc::clone(&catalog),
		&scratch.state().join("credentials.sqlite"),
		broker.clone(),
		accounts.clone(),
	);
	let sessions = ConversationSessionPlanner::with_in_memory(
		Arc::new(InMemoryConversationStore::new()),
		Arc::clone(&catalog),
	);
	let dependencies = ProductionDependencies::new(
		broker,
		auth,
		accounts,
		sessions.clone(),
		WebSocketTransport::new(),
		GoogleCcaConfig {
			gemini_cli_platform: "test".into(),
			gemini_cli_arch: "test".into(),
			antigravity_headers: CcaHeaders::antigravity(
				&AntigravityFingerprint::default(),
				false,
				None,
			),
			antigravity_policy: AntigravityPolicy::default(),
		},
		HttpTransport::new(),
		AuthApplicationConfig { signing_regions: Arc::new(BTreeMap::new()) },
		AdmissionController::new(8, 8),
		Duration::from_secs(2),
		Arc::new(BTreeMap::new()),
	)
	.with_local_routes([(
		ROUTE.into(),
		LocalRouteBackend::new(
			Arc::new(OpenAiChatCodec::default()),
			cassette,
			Duration::from_secs(2),
		),
	)]);
	let registry = Registry::builder(catalog)
		.with_builtins(BuiltinConfig::production(dependencies))
		.expect("compose production route stack")
		.build()
		.expect("build inference registry");
	let service = InferenceRpc::new(registry, sessions, tools);
	InProcTurnClient::new(service).await.expect("start in-process gateway")
}

fn journal(scratch: &Scratch) -> Journal {
	Journal::create(
		&scratch.state().join("p5.jsonl"),
		&Header {
			v: 4,
			id: SessionId(Str::from("p5-prefix-stability")),
			created: 0,
			cwd: scratch.project().to_owned(),
		},
	)
	.expect("create agent journal")
}

fn context_file(path: &std::path::Path) -> ContextFile {
	ContextFile::new("AGENTS.md", std::fs::read(path).expect("read context file"))
}

fn array_contents<'a>(body: &'a [u8], field: &[u8]) -> &'a [u8] {
	let mut needle = Vec::with_capacity(field.len() + 4);
	needle.push(b'"');
	needle.extend_from_slice(field);
	needle.extend_from_slice(b"\":[");
	let start = body
		.windows(needle.len())
		.position(|window| window == needle)
		.map(|index| index + needle.len())
		.expect("captured request contains expected array");
	let mut depth = 1_u32;
	let mut quoted = false;
	let mut escaped = false;
	for (offset, byte) in body[start..].iter().copied().enumerate() {
		if quoted {
			if escaped {
				escaped = false;
			} else if byte == b'\\' {
				escaped = true;
			} else if byte == b'"' {
				quoted = false;
			}
			continue;
		}
		match byte {
			b'"' => quoted = true,
			b'[' => depth += 1,
			b']' => {
				depth -= 1;
				if depth == 0 {
					return &body[start..start + offset];
				}
			},
			_ => {},
		}
	}
	panic!("captured request array is unterminated")
}

fn assert_prefix(left: &[u8], right: &[u8], label: &str) {
	assert!(right.starts_with(left), "{label} was not byte-prefix stable");
}

#[tokio::test]
async fn delta_context_prompt_rewind_and_tool_revision_preserve_exact_provider_prefixes() {
	let scratch = Scratch::new().expect("scratch workspace");
	let prompt_path = scratch.write("AGENTS.md", b"stable prompt v1\n").expect("initial prompt");
	let tools_v1 = tool_registry(1);
	let tools_v2 = tool_registry(2);
	assert_ne!(tools_v1.live_hash(), tools_v2.live_hash(), "revision swap changes live hash");

	let cassette = CassetteTransport::new(Arc::<[CassetteAttempt]>::from(
		(0..7).map(|_| cassette_attempt()).collect::<Vec<_>>(),
	))
	.with_request_body_capture(NonZeroUsize::new(BODY_LIMIT).expect("nonzero body limit"));
	let cassette_probe = cassette.clone();
	let client = Instrumented::new(gateway(&scratch, cassette, Arc::clone(&tools_v1)).await);
	let probe = client.clone();
	let options = TurnOptions {
		context_id: Some(Str::from("p5-context")),
		params: pb::ChatParams { model: MODEL.to_owned(), tools: vec![tool_def(1)], ..Default::default() },
		executor: None,
		props: None,
	};
	let workspace = WorkspaceInput::new(
		scratch.project(),
		Arc::<[ContextFile]>::from([context_file(&prompt_path)]),
	);
	let state = AgentState::new(AgentSnapshot::new(options, workspace, Arc::clone(&tools_v1)));
	let (env, _env_transport) = omp_env::EnvClient::in_process(4);
	let mut agent = Agent::new(
		client,
		env,
		state.clone(),
		journal(&scratch),
		PromptCaps { maximum_parts: 4, maximum_text_bytes: 4096, media: false },
	);

	let mut revisions = Vec::new();
	for (turn, text) in [(1, "steady one"), (2, "steady two"), (3, "steady tri")] {
		if turn == 3 {
			state.update(|snapshot| snapshot.registry = Arc::clone(&tools_v1));
		}
		let summary = within(
			"p5 steady turn",
			Duration::from_secs(5),
			agent.submit([user_item(text)], TurnId::new(format!("p5-{turn}"))),
		)
		.await
		.expect("steady turn stays within deadline")
		.expect("steady turn succeeds");
		revisions.push(summary.outcome.revision.expect("stateful revision"));
	}

	scratch.write("AGENTS.md", b"stable prompt v2\n").expect("mutate real context file");
	state.update(|snapshot| {
		snapshot.workspace.context_files = Arc::from([context_file(&prompt_path)]);
	});
	let fourth = within(
		"p5 prompt rewind",
		Duration::from_secs(5),
		agent.submit([user_item("after prompt")], TurnId::new("p5-4")),
	)
	.await
	.expect("prompt rewind stays within deadline")
	.expect("prompt rewind succeeds without conflict");
	revisions.push(fourth.outcome.revision.expect("prompt rewind revision"));

	let fifth = within(
		"p5 unchanged registry",
		Duration::from_secs(5),
		agent.submit([user_item("same tools")], TurnId::new("p5-5")),
	)
	.await
	.expect("unchanged registry stays within deadline")
	.expect("unchanged registry turn succeeds");
	revisions.push(fifth.outcome.revision.expect("unchanged registry revision"));

	state.update(|snapshot| {
		snapshot.registry = Arc::clone(&tools_v2);
		snapshot.turn.params.tools = vec![tool_def(2)];
	});
	let sixth = within(
		"p5 registry revision swap",
		Duration::from_secs(5),
		agent.submit([user_item("new tools")], TurnId::new("p5-6")),
	)
	.await
	.expect("revision swap stays within deadline")
	.expect("revision swap succeeds");
	revisions.push(sixth.outcome.revision.expect("revision swap revision"));
	let seventh = within(
		"p5 stable swapped registry",
		Duration::from_secs(5),
		agent.submit([user_item("tools stay")], TurnId::new("p5-7")),
	)
	.await
	.expect("post-swap turn stays within deadline")
	.expect("post-swap steady turn succeeds");
	revisions.push(seventh.outcome.revision.expect("post-swap revision"));

	for pair in revisions.windows(2) {
		assert!(pair[0].head < pair[1].head, "gateway revisions must be strictly monotone");
	}

	let turns = probe.captures();
	assert_eq!(turns.len(), 7, "no implicit reseed submission or retry");
	assert!(matches!(turns[0].input, TurnInput::Full(_)), "only turn one seeds");
	let (second_context, second_delta) = match &turns[1].input {
		TurnInput::Delta(context, delta) => (context, delta),
		TurnInput::Full(_) => panic!("turn two must be Delta-only"),
	};
	let (third_context, third_delta) = match &turns[2].input {
		TurnInput::Delta(context, delta) => (context, delta),
		TurnInput::Full(_) => panic!("turn three must be Delta-only"),
	};
	assert_eq!(second_context.context_id, "p5-context");
	assert_eq!(third_context.context_id, second_context.context_id);
	assert_eq!(second_delta.truncate_to, None);
	assert_eq!(third_delta.truncate_to, None);
	assert_eq!(second_delta.encoded_len(), third_delta.encoded_len(), "equal-size new items have history-independent delta bytes");
	assert_eq!(second_delta.append.len(), 1);
	assert_eq!(third_delta.append.len(), 1);

	let (fourth_context, fourth_delta) = match &turns[3].input {
		TurnInput::Delta(context, delta) => (context, delta),
		TurnInput::Full(_) => panic!("prompt replacement must not reseed the gateway context"),
	};
	assert_eq!(fourth_context.context_id, second_context.context_id);
	assert_eq!(fourth_delta.truncate_to, Some(0));
	let fourth_json = serde_json::to_vec(&fourth_delta.append).expect("serialize rewind append");
	assert!(fourth_json.windows(b"stable prompt v2".len()).any(|w| w == b"stable prompt v2"));
	for tail in [b"steady one".as_slice(), b"steady two", b"steady tri", b"after prompt"] {
		assert!(fourth_json.windows(tail.len()).any(|window| window == tail), "rewind preserves tail");
	}
	assert!(matches!(&turns[4].input, TurnInput::Delta(_, thread::ThreadDelta { truncate_to: None, .. })));
	assert!(matches!(&turns[5].input, TurnInput::Delta(_, thread::ThreadDelta { truncate_to: Some(0), .. })));
	assert!(matches!(&turns[6].input, TurnInput::Delta(_, thread::ThreadDelta { truncate_to: None, .. })));
	assert_eq!(turns[4].options.params.tools, vec![tool_def(1)]);
	assert_eq!(turns[5].options.params.tools, vec![tool_def(2)]);
	assert_eq!(turns[6].options.params.tools, vec![tool_def(2)]);

	let captures = cassette_probe.captures();
	assert_eq!(captures.len(), 7, "one provider attempt per logical turn, including exactly one prompt reseed");
	let bodies: Vec<Bytes> = captures
		.into_iter()
		.map(|capture| {
			let body = capture.request_body.expect("sanctioned cassette body capture enabled");
			assert!(!body.truncated, "request capture bound must retain exact bytes");
			assert_eq!(body.observed_bytes, body.bytes.len() as u64);
			body.bytes
		})
		.collect();

	let messages: Vec<&[u8]> = bodies.iter().map(|body| array_contents(body, b"messages")).collect();
	assert_prefix(messages[0], messages[1], "turn 1→2 dialect messages");
	assert_prefix(messages[1], messages[2], "turn 2→3 dialect messages");
	assert!(messages[3].windows(b"stable prompt v2".len()).any(|w| w == b"stable prompt v2"));
	assert!(!messages[3].windows(b"stable prompt v1".len()).any(|w| w == b"stable prompt v1"));
	for tail in [b"steady one".as_slice(), b"steady two", b"steady tri"] {
		assert!(messages[3].windows(tail.len()).any(|window| window == tail), "provider replay preserves canonical tail");
	}

	let tool_bytes: Vec<&[u8]> = bodies.iter().map(|body| array_contents(body, b"tools")).collect();
	assert_eq!(tool_bytes[0], tool_bytes[1]);
	assert_eq!(tool_bytes[1], tool_bytes[2]);
	assert_eq!(tool_bytes[3], tool_bytes[4], "prompt edit alone leaves tools byte-stable");
	assert_ne!(tool_bytes[4], tool_bytes[5], "live revision swap changes provider tools once");
	assert_eq!(tool_bytes[5], tool_bytes[6], "unchanged swapped registry is byte-stable");
}
