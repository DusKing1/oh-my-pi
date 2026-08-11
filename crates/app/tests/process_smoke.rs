#![cfg(unix)]

//! End-to-end daemon, auth, catalog, and inference process smoke coverage.

use std::{
	collections::BTreeSet,
	fs,
	path::{Path, PathBuf},
	process::{Output, Stdio},
	sync::{
		Arc,
		atomic::{AtomicUsize, Ordering},
	},
	time::{Duration, SystemTime, UNIX_EPOCH},
};

use omp_proto::{
	auth::v1 as auth_pb, blob::v1 as blob_pb, gateway::v1 as gateway_pb, inference::v1 as pb,
	thread::v1 as thread_pb,
};
use tokio::{
	io::{AsyncReadExt as _, AsyncWriteExt as _},
	net::{TcpListener, TcpStream, UnixStream},
	process::{Child, Command},
	sync::{mpsc, oneshot},
	time::{sleep, timeout},
};
use tonic::{Code, Response, Status, transport::Channel};
use tonic_health::pb::{
	HealthCheckRequest, health_check_response::ServingStatus, health_client::HealthClient,
};

const PROVIDER: &str = "cerebras";
const MODEL: &str = "cerebras/zai-glm-4.7";
const PROVIDER_SECRET: &str = "process-smoke-provider-secret-never-log";
const GATEWAY_SECRET: &str = "process-smoke-gateway-secret-never-log";
const FIXTURE_ANSWER: &str = "process smoke answer";
const PROCESS_TIMEOUT: Duration = Duration::from_secs(15);
const TURN_TIMEOUT: Duration = Duration::from_secs(10);

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn daemon_composition_mounts_catalog_facets_and_drains_cleanly() {
	let scratch = Scratch::new();
	let project = scratch.path().join("project");
	let data_dir = scratch.path().join("data");
	let home = scratch.path().join("home");
	fs::create_dir_all(project.join(".omp")).expect("create project fixture");
	fs::create_dir_all(&data_dir).expect("create broker fixture directory");
	fs::create_dir_all(&home).expect("create isolated home");

	let missing_auth_socket = scratch.path().join("missing-auth.sock");
	let mut missing_auth = omp_command(&project, &home, &data_dir);
	missing_auth
		.env_remove("OMP_GATEWAY_TOKEN")
		.args(["serve", "--endpoint"])
		.arg(&missing_auth_socket)
		.args(["--data-dir"])
		.arg(&data_dir);
	let missing_auth = timeout(PROCESS_TIMEOUT, missing_auth.output())
		.await
		.expect("missing-auth startup stayed bounded")
		.expect("run missing-auth startup");
	assert!(!missing_auth.status.success(), "daemon accepted an unauthenticated facade");
	assert!(
		String::from_utf8_lossy(&missing_auth.stderr).contains("OMP_GATEWAY_TOKEN"),
		"missing-auth failure was not explicit: {}",
		String::from_utf8_lossy(&missing_auth.stderr)
	);
	assert!(!missing_auth_socket.exists(), "failed startup left a local socket");

	let missing_state_socket = scratch.path().join("missing-state.sock");
	let mut missing_state = Command::new(env!("CARGO_BIN_EXE_omp"));
	missing_state
		.current_dir(&project)
		.env_clear()
		.env("OMP_GATEWAY_TOKEN", GATEWAY_SECRET)
		.args(["serve", "--endpoint"])
		.arg(&missing_state_socket);
	let missing_state = timeout(PROCESS_TIMEOUT, missing_state.output())
		.await
		.expect("missing-state startup stayed bounded")
		.expect("run missing-state startup");
	assert!(!missing_state.status.success(), "daemon accepted a missing broker/data directory");
	assert!(
		String::from_utf8_lossy(&missing_state.stderr).contains("data directory is unavailable"),
		"missing-state failure was not explicit: {}",
		String::from_utf8_lossy(&missing_state.stderr)
	);
	assert!(!missing_state_socket.exists(), "failed startup left a local socket");
	for boundary in [&missing_state.stdout, &missing_state.stderr] {
		assert!(
			!String::from_utf8_lossy(boundary).contains(GATEWAY_SECRET),
			"gateway token appeared in failed child process output"
		);
	}

	let bad_tls = omp_rpc::TlsConfig {
		cert:      scratch.path().join("missing-cert.pem"),
		key:       scratch.path().join("missing-key.pem"),
		client_ca: None,
	};
	assert!(
		omp_llm_gateway::listener::RemoteTls::mutual_tls(bad_tls.clone()).is_err(),
		"mTLS accepted a missing client CA policy"
	);
	assert!(
		omp_llm_gateway::listener::RemoteTls::bearer(bad_tls.clone(), b"").is_err(),
		"remote bearer policy accepted an empty token"
	);
	let bad_tls = omp_llm_gateway::listener::RemoteTls::bearer(bad_tls, b"gateway-test")
		.expect("non-empty bearer policy");
	let bad_tls_error = omp_llm_gateway::listener::RemoteListener::bind(
		"127.0.0.1:0".parse().expect("loopback address"),
		bad_tls,
	)
	.await
	.err()
	.expect("missing TLS identity must fail before binding");
	assert!(
		bad_tls_error.to_string().contains("I/O error"),
		"bad TLS failure was not explicit: {bad_tls_error}"
	);

	let listener = TcpListener::bind(("127.0.0.1", 0))
		.await
		.expect("bind local provider fixture");
	let provider_addr = listener.local_addr().expect("provider fixture address");
	fs::write(
		project.join(".omp/providers.toml"),
		format!("[providers.{PROVIDER}]\nbase_url = \"http://{provider_addr}/v1\"\n"),
	)
	.expect("write provider catalog overlay");
	let migration = scratch.path().join("credential.json");
	fs::write(
		&migration,
		format!(
			"{{\"{PROVIDER}\":{{\"type\":\"api_key\",\"key\":\"{PROVIDER_SECRET}\",\"source\":\"\
			 process-smoke\"}}}}"
		),
	)
	.expect("write omp auth migration fixture");

	let (requests_tx, mut requests_rx) = mpsc::unbounded_channel();
	let chat_requests = Arc::new(AtomicUsize::new(0));
	let (fixture_shutdown_tx, fixture_shutdown_rx) = oneshot::channel();
	let fixture = tokio::spawn(run_provider_fixture(
		listener,
		requests_tx,
		Arc::clone(&chat_requests),
		fixture_shutdown_rx,
	));

	let mut captured = Vec::new();
	let empty = run_omp(&project, &home, &data_dir, [
		"auth",
		"--data-dir",
		path_arg(&data_dir),
		"list",
		"--json",
	])
	.await;
	assert_success("initial auth list", &empty);
	assert_eq!(String::from_utf8_lossy(&empty.stdout).trim(), "[]");
	captured.push(empty);

	let migrated = run_omp(&project, &home, &data_dir, [
		"auth",
		"--data-dir",
		path_arg(&data_dir),
		"migrate",
		"--json-file",
		path_arg(&migration),
	])
	.await;
	assert_success("auth migration", &migrated);
	assert_eq!(String::from_utf8_lossy(&migrated.stdout).trim(), "imported 1 credential(s)");
	captured.push(migrated);

	let socket = scratch.path().join("gateway.sock");
	let mut command = omp_command(&project, &home, &data_dir);
	command
		.args(["serve", "--uds"])
		.arg(&socket)
		.args(["--data-dir"])
		.arg(&data_dir)
		.stdout(Stdio::piped())
		.stderr(Stdio::piped())
		.kill_on_drop(true);
	let mut server = command.spawn().expect("spawn omp serve process");
	let channel = wait_for_gateway(&mut server, &socket).await;

	let hello = gateway_pb::gateway_client::GatewayClient::new(channel.clone())
		.hello(gateway_pb::HelloRequest {
			client:       "daemon-composition-test".into(),
			schema_rev:   omp_proto::SCHEMA_REV,
			capabilities: Vec::new(),
		})
		.await
		.expect("Hello is mounted and ready")
		.into_inner();
	assert_eq!(hello.schema_rev, omp_proto::SCHEMA_REV);
	let capabilities = hello.capabilities;
	let unique_capabilities: BTreeSet<_> = capabilities.iter().collect();
	assert_eq!(
		unique_capabilities.len(),
		capabilities.len(),
		"Hello advertised a duplicate mounted capability: {capabilities:?}"
	);
	for required in [
		"inference.turn",
		"inference.invoke",
		"inference.contexts",
		"inference.models",
		"auth",
		"blob.v1",
		"foreign-facades",
	] {
		assert!(
			capabilities.iter().any(|capability| capability == required),
			"Hello omitted mounted capability {required}: {capabilities:?}"
		);
	}

	assert_eq!(
		facade_status(&socket, "GET", "/v1/models", b"").await,
		200,
		"catalog facade was not mounted"
	);
	for (method, path) in [
		("POST", "/v1/chat/completions"),
		("POST", "/v1/responses"),
		("POST", "/v1/messages"),
		("POST", "/v1/messages/count_tokens"),
		("POST", "/v1/embeddings"),
		("POST", "/v1/images/generations"),
		("POST", "/v1/images/edits"),
		("POST", "/v1/audio/speech"),
		("POST", "/v1/audio/transcriptions"),
		("POST", "/v1/audio/translations"),
		("POST", "/v1/videos"),
		("GET", "/v1/videos/missing"),
		("DELETE", "/v1/videos/missing"),
	] {
		let status = facade_status(&socket, method, path, b"{}").await;
		assert_ne!(status, 404, "advertised foreign facade route {method} {path} was absent");
		assert_ne!(status, 401, "gateway bearer was not accepted for {method} {path}");
	}

	let mut health = HealthClient::new(channel.clone());
	for service in [
		"",
		"omp.inference.v1.Inference",
		"omp.auth.v1.Auth",
		"omp.blob.v1.Blob",
		"omp.gateway.v1.Gateway",
	] {
		let status = health
			.check(HealthCheckRequest { service: service.into() })
			.await
			.unwrap_or_else(|error| panic!("health check for {service:?} failed: {error}"))
			.into_inner()
			.status;
		assert_eq!(status, ServingStatus::Serving as i32, "{service:?} was not serving at readiness");
	}

	let providers = pb::inference_client::InferenceClient::new(channel.clone())
		.list_providers(pb::ListProvidersRequest::default())
		.await
		.expect("ListProviders is mounted")
		.into_inner()
		.providers;
	for provider in providers.iter().filter(|provider| provider.credentialed) {
		for facet in &provider.facets {
			let capability = match pb::Facet::try_from(*facet).expect("catalog emitted known facet") {
				pb::Facet::Unspecified => continue,
				pb::Facet::Chat | pb::Facet::Realtime => "inference.turn",
				pb::Facet::Embed => "inference.embed",
				pb::Facet::ImageGen => "inference.image",
				pb::Facet::VideoGen => "inference.video",
				pb::Facet::Speak => "inference.speak",
				pb::Facet::Transcribe => "inference.transcribe",
				pb::Facet::Search => "search",
			};
			assert!(
				capabilities.iter().any(|actual| actual == capability),
				"credentialed catalog provider {} advertises facet {facet} without {capability}",
				provider.id
			);
		}
	}

	let mut mounted = pb::inference_client::InferenceClient::new(channel.clone());
	probe_mounted("Fork", mounted.fork(pb::ForkRequest::default())).await;
	probe_mounted("Drop", mounted.drop(pb::DropRequest::default())).await;
	probe_mounted("WatchModels", mounted.watch_models(pb::WatchModelsRequest::default())).await;
	if capabilities
		.iter()
		.any(|capability| capability == "inference.count-tokens")
	{
		probe_mounted("CountTokens", mounted.count_tokens(pb::CountTokensRequest::default())).await;
	}
	if capabilities
		.iter()
		.any(|capability| capability == "inference.embed")
	{
		probe_mounted("Embed", mounted.embed(pb::EmbedRequest::default())).await;
	}
	if capabilities
		.iter()
		.any(|capability| capability == "inference.image")
	{
		probe_mounted("GenerateImage", mounted.generate_image(pb::GenerateImageRequest::default()))
			.await;
	}
	if capabilities
		.iter()
		.any(|capability| capability == "inference.speak")
	{
		probe_mounted("Speak", mounted.speak(pb::SpeakRequest::default())).await;
	}
	if capabilities
		.iter()
		.any(|capability| capability == "inference.transcribe")
	{
		probe_mounted("Transcribe", mounted.transcribe(pb::TranscribeRequest::default())).await;
	}
	if capabilities
		.iter()
		.any(|capability| capability == "inference.video")
	{
		probe_mounted("GenerateVideo", mounted.generate_video(pb::GenerateVideoRequest::default()))
			.await;
		probe_mounted("GetGeneration", mounted.get_generation(pb::GetGenerationRequest::default()))
			.await;
	}
	if capabilities.iter().any(|capability| capability == "search") {
		probe_mounted(
			"Search",
			mounted.search(pb::SearchRequest {
				engine: "composition-test-unknown-engine".into(),
				..Default::default()
			}),
		)
		.await;
	}
	let credentials = auth_pb::auth_client::AuthClient::new(channel.clone())
		.list_credentials(auth_pb::ListCredentialsRequest::default())
		.await
		.expect("Auth.ListCredentials is mounted")
		.into_inner();
	assert_eq!(
		credentials.credentials.len(),
		1,
		"migration must import exactly one broker credential"
	);
	assert_eq!(credentials.credentials[0].provider, PROVIDER);
	let missing_blob = blob_pb::blob_client::BlobClient::new(channel.clone())
		.stat(blob_pb::StatRequest { hash: vec![0; 32].into() })
		.await
		.expect("Blob.Stat is mounted")
		.into_inner();
	assert!(!missing_blob.present);

	let listed = run_omp(&project, &home, &data_dir, [
		"auth",
		"--data-dir",
		path_arg(&data_dir),
		"list",
		"--provider",
		PROVIDER,
		"--json",
	])
	.await;
	assert_success("populated auth list", &listed);
	let listed_text = String::from_utf8_lossy(&listed.stdout);
	assert!(listed_text.contains(&format!("\"provider\": \"{PROVIDER}\"")));
	assert!(listed_text.contains("\"state\": \"active\""));
	assert!(listed_text.contains("\"identity\": \"imported\""));
	captured.push(listed);

	let mut client = pb::inference_client::InferenceClient::new(channel);
	let models = timeout(
		TURN_TIMEOUT,
		client.list_models(pb::ListModelsRequest {
			provider:       PROVIDER.into(),
			facet:          pb::Facet::Chat as i32,
			available_only: true,
		}),
	)
	.await
	.expect("ListModels stayed bounded")
	.expect("ListModels succeeds")
	.into_inner()
	.models;
	let card = models
		.iter()
		.find(|card| card.id == MODEL)
		.expect("fixture model is listed as available");
	assert_eq!(card.provider, PROVIDER);
	assert_eq!(card.availability, pb::Availability::Available as i32);

	let opened = timeout(
		TURN_TIMEOUT,
		client.turn(futures::stream::iter([turn_open("01JPROCESS0SMOKE0TURN000000")])),
	)
	.await
	.expect("opening native Turn stayed bounded");
	let mut events = match opened {
		Ok(response) => response.into_inner(),
		Err(error) => daemon_rpc_failure(server, "native Turn opens", error).await,
	};
	let outcome = timeout(TURN_TIMEOUT, async {
		let mut terminal = None;
		loop {
			let event = match events.message().await {
				Ok(event) => event,
				Err(error) => return Err(format!("receive native Turn event: {error}")),
			};
			match event {
				Some(_) if terminal.is_some() => {
					panic!("native Turn emitted an event after its terminal outcome")
				},
				Some(pb::TurnEvent { event: Some(pb::turn_event::Event::Outcome(outcome)) }) => {
					terminal = Some(outcome);
				},
				Some(pb::TurnEvent { event: Some(pb::turn_event::Event::Error(error)) }) => {
					return Err(error.detail);
				},
				Some(_) => {},
				None => {
					return Ok(terminal.expect("native Turn has one authoritative outcome"));
				},
			}
		}
	})
	.await
	.expect("native Turn stream stayed bounded");
	let outcome = match outcome {
		Ok(outcome) => outcome,
		Err(detail) => {
			let detail = format!(
				"{detail}; provider fixture chat request count={}",
				chat_requests.load(Ordering::SeqCst),
			);
			daemon_rpc_failure(server, "native Turn failed", detail).await
		},
	};
	assert_eq!(outcome.provider, PROVIDER);
	assert_eq!(outcome.model, "zai-glm-4.7");
	assert_eq!(outcome_text(&outcome), FIXTURE_ANSWER);

	let request = timeout(TURN_TIMEOUT, async {
		loop {
			let request = requests_rx
				.recv()
				.await
				.expect("provider fixture remains live");
			if request.method == "POST" && request.path.ends_with("/chat/completions") {
				return request;
			}
		}
	})
	.await
	.expect("provider request observation stayed bounded");
	let expected_authorization = format!("Bearer {PROVIDER_SECRET}");
	assert_eq!(request.header("authorization"), Some(expected_authorization.as_str()));
	let request_body = String::from_utf8(request.body).expect("provider request body is UTF-8 JSON");
	assert!(request_body.contains("\"model\":\"zai-glm-4.7\""));
	assert!(request_body.contains("process smoke prompt"));

	let mut failed_events = timeout(
		TURN_TIMEOUT,
		client.turn(futures::stream::iter([turn_open("01JPROCESS0SMOKE0FAIL000000")])),
	)
	.await
	.expect("opening forced-failure Turn stayed bounded")
	.expect("forced-failure Turn opens")
	.into_inner();
	let failure = timeout(TURN_TIMEOUT, async {
		loop {
			let event = failed_events
				.message()
				.await
				.expect("receive forced-failure Turn event")
				.expect("forced-failure Turn has a terminal event");
			if let Some(pb::turn_event::Event::Error(error)) = event.event {
				return error;
			}
		}
	})
	.await
	.expect("forced-failure Turn stayed bounded");
	for secret in [PROVIDER_SECRET, GATEWAY_SECRET] {
		assert!(!failure.detail.contains(secret), "secret appeared in RPC failure");
		assert!(!format!("{failure:?}").contains(secret), "secret appeared in RPC Debug");
	}

	let failed_request = timeout(TURN_TIMEOUT, async {
		loop {
			let request = requests_rx
				.recv()
				.await
				.expect("provider fixture remains live");
			if request.method == "POST" && request.path.ends_with("/chat/completions") {
				return request;
			}
		}
	})
	.await
	.expect("forced-failure provider request observation stayed bounded");
	assert_eq!(failed_request.header("authorization"), Some(expected_authorization.as_str()));

	send_sigterm(server.id().expect("serve process id"));
	let server_output = timeout(PROCESS_TIMEOUT, server.wait_with_output())
		.await
		.expect("serve process exits after SIGTERM")
		.expect("collect serve process output");
	assert_success("omp serve", &server_output);
	captured.push(server_output);
	assert!(!socket.exists(), "graceful shutdown removes the UDS");

	let _ = fixture_shutdown_tx.send(());
	timeout(PROCESS_TIMEOUT, fixture)
		.await
		.expect("provider fixture stops")
		.expect("provider fixture task succeeds")
		.expect("provider fixture serves without I/O errors");
	assert!(
		chat_requests.load(Ordering::SeqCst) >= 2,
		"success and forced-failure provider turns were attempted"
	);

	for output in captured {
		let stdout = String::from_utf8_lossy(&output.stdout);
		let stderr = String::from_utf8_lossy(&output.stderr);
		for secret in [PROVIDER_SECRET, GATEWAY_SECRET] {
			assert!(!stdout.contains(secret), "secret appeared in process stdout");
			assert!(!stderr.contains(secret), "secret appeared in process stderr");
		}
	}
}

async fn facade_status(socket: &Path, method: &str, path: &str, body: &[u8]) -> u16 {
	let mut stream = UnixStream::connect(socket)
		.await
		.unwrap_or_else(|error| panic!("connect facade socket for {method} {path}: {error}"));
	let head = format!(
		"{method} {path} HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer \
		 {GATEWAY_SECRET}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: \
		 close\r\n\r\n",
		body.len()
	);
	stream
		.write_all(head.as_bytes())
		.await
		.unwrap_or_else(|error| panic!("write facade headers for {method} {path}: {error}"));
	stream
		.write_all(body)
		.await
		.unwrap_or_else(|error| panic!("write facade body for {method} {path}: {error}"));
	let mut response = Vec::new();
	timeout(TURN_TIMEOUT, stream.read_to_end(&mut response))
		.await
		.unwrap_or_else(|_| panic!("facade route {method} {path} stayed open"))
		.unwrap_or_else(|error| panic!("read facade response for {method} {path}: {error}"));
	let status_line = String::from_utf8_lossy(&response);
	status_line
		.lines()
		.next()
		.and_then(|line| line.split_ascii_whitespace().nth(1))
		.and_then(|status| status.parse().ok())
		.unwrap_or_else(|| panic!("invalid facade response for {method} {path}: {status_line}"))
}

async fn probe_mounted<T>(name: &str, call: impl Future<Output = Result<Response<T>, Status>>) {
	match timeout(TURN_TIMEOUT, call).await {
		Ok(Ok(_)) => {},
		Ok(Err(status)) => assert_ne!(
			status.code(),
			Code::Unimplemented,
			"{name} was advertised but not mounted: {status}"
		),
		Err(_) => panic!("{name} did not complete before the listener deadline"),
	}
}

fn turn_open(turn_id: &str) -> pb::TurnFrame {
	let prompt = thread_pb::Part {
		kind: Some(thread_pb::part::Kind::Text("process smoke prompt".into())),
		..Default::default()
	};
	let item = thread_pb::Item {
		seq: 0,
		kind: Some(thread_pb::item::Kind::Message(thread_pb::Message {
			role: thread_pb::Role::User as i32,
			parts: vec![prompt],
			..Default::default()
		})),
		..Default::default()
	};
	pb::TurnFrame {
		frame: Some(pb::turn_frame::Frame::Open(pb::TurnRequest {
			turn_id: turn_id.into(),
			input: Some(pb::turn_request::Input::Seed(pb::Seed {
				context_id: String::new(),
				thread: Some(thread_pb::Thread { items: vec![item], ..Default::default() }),
				..Default::default()
			})),
			params: Some(pb::ChatParams { model: MODEL.into(), ..Default::default() }),
			..Default::default()
		})),
		..Default::default()
	}
}

fn outcome_text(outcome: &pb::Outcome) -> String {
	let mut text = String::new();
	for item in &outcome.output {
		let Some(thread_pb::item::Kind::Message(message)) = &item.kind else {
			continue;
		};
		for part in &message.parts {
			if let Some(thread_pb::part::Kind::Text(chunk)) = &part.kind {
				text.push_str(chunk);
			}
		}
	}
	text
}

async fn wait_for_gateway(child: &mut Child, socket: &Path) -> Channel {
	timeout(PROCESS_TIMEOUT, async {
		loop {
			if let Some(status) = child.try_wait().expect("inspect serve process") {
				let mut stderr = String::new();
				if let Some(pipe) = child.stderr.as_mut() {
					pipe
						.read_to_string(&mut stderr)
						.await
						.expect("read failed daemon diagnostics");
				}
				panic!("omp serve exited before readiness: {status}: {stderr}");
			}
			match omp_rpc::uds::connect(socket).await {
				Ok(channel) => return channel,
				Err(_) => sleep(Duration::from_millis(20)).await,
			}
		}
	})
	.await
	.expect("omp serve binds its UDS before the readiness deadline")
}

async fn daemon_rpc_failure(mut child: Child, context: &str, error: impl std::fmt::Display) -> ! {
	let status = child.try_wait().expect("inspect failed serve process");
	if status.is_none() {
		child
			.kill()
			.await
			.expect("stop serve process after broken RPC");
	}
	let output = timeout(PROCESS_TIMEOUT, child.wait_with_output())
		.await
		.expect("failed serve process exits")
		.expect("collect failed serve process diagnostics");
	panic!(
		"{context}: {error}; omp serve status: {}; stderr:\n{}",
		output.status,
		String::from_utf8_lossy(&output.stderr),
	);
}

async fn run_omp<const N: usize>(
	project: &Path,
	home: &Path,
	data_dir: &Path,
	args: [&str; N],
) -> Output {
	timeout(PROCESS_TIMEOUT, omp_command(project, home, data_dir).args(args).output())
		.await
		.expect("omp CLI process stayed bounded")
		.expect("run omp CLI process")
}

fn omp_command(project: &Path, home: &Path, data_dir: &Path) -> Command {
	let mut command = Command::new(env!("CARGO_BIN_EXE_omp"));
	command
		.current_dir(project)
		.env_clear()
		.env("HOME", home)
		.env("OMP_DATA_DIR", data_dir)
		.env("OMP_GATEWAY_TOKEN", GATEWAY_SECRET);
	command
}

fn assert_success(name: &str, output: &Output) {
	assert!(
		output.status.success(),
		"{name} failed with {}: {}",
		output.status,
		String::from_utf8_lossy(&output.stderr)
	);
}

fn path_arg(path: &Path) -> &str {
	path.to_str().expect("fixture paths are UTF-8")
}

struct Scratch(PathBuf);

impl Scratch {
	fn new() -> Self {
		let nonce = SystemTime::now()
			.duration_since(UNIX_EPOCH)
			.expect("system clock after epoch")
			.as_nanos();
		// macOS's per-user temporary directory can exceed sockaddr_un once the
		// fixture suffix is added; /tmp keeps the real UDS path below that limit.
		let path =
			PathBuf::from("/tmp").join(format!("omp-process-smoke-{}-{nonce}", std::process::id()));
		fs::create_dir(&path).expect("create process smoke scratch directory");
		Self(path)
	}

	fn path(&self) -> &Path {
		&self.0
	}
}

impl Drop for Scratch {
	fn drop(&mut self) {
		let _ = fs::remove_dir_all(&self.0);
	}
}

#[derive(Debug)]
struct FixtureRequest {
	method:  String,
	path:    String,
	headers: Vec<(String, String)>,
	body:    Vec<u8>,
}

impl FixtureRequest {
	fn header(&self, name: &str) -> Option<&str> {
		self
			.headers
			.iter()
			.find_map(|(key, value)| (key == name).then_some(value.as_str()))
	}
}

async fn run_provider_fixture(
	listener: TcpListener,
	requests: mpsc::UnboundedSender<FixtureRequest>,
	chat_requests: Arc<AtomicUsize>,
	mut shutdown: oneshot::Receiver<()>,
) -> std::io::Result<()> {
	loop {
		let (mut stream, _) = tokio::select! {
			biased;
			_ = &mut shutdown => return Ok(()),
			accepted = listener.accept() => accepted?,
		};
		let request = read_request(&mut stream).await?;
		let is_models = request.method == "GET" && request.path.ends_with("/models");
		let is_chat = request.method == "POST" && request.path.ends_with("/chat/completions");
		let chat_attempt = is_chat.then(|| {
			chat_requests
				.fetch_add(1, Ordering::SeqCst)
				.saturating_add(1)
		});
		let _ = requests.send(request);
		if is_models {
			write_response(
				&mut stream,
				"application/json",
				br#"{"object":"list","data":[{"id":"zai-glm-4.7","object":"model","created":0,"owned_by":"cerebras"}]}"#,
			)
			.await?;
		} else if chat_attempt == Some(1) {
			write_response(
				&mut stream,
				"text/event-stream",
				concat!(
					r#"data: {"id":"chatcmpl-process-smoke","object":"chat.completion.chunk","created":0,"model":"zai-glm-4.7","choices":[{"index":0,"delta":{"role":"assistant","content":"process smoke answer"},"finish_reason":null}]}"#,
					"\n\n",
					r#"data: {"id":"chatcmpl-process-smoke","object":"chat.completion.chunk","created":0,"model":"zai-glm-4.7","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
					"\n\n",
					"data: [DONE]\n\n"
				)
				.as_bytes(),
			)
			.await?;
		} else if is_chat {
			let echoed = format!(
				"forced failure echoed Authorization: Bearer {PROVIDER_SECRET}; \
				 Cookie={GATEWAY_SECRET}"
			);
			write_status(&mut stream, "400 Bad Request", echoed.as_bytes()).await?;
		} else {
			write_status(&mut stream, "404 Not Found", b"unexpected fixture route").await?;
		}
	}
}

async fn read_request(stream: &mut TcpStream) -> std::io::Result<FixtureRequest> {
	let mut bytes = Vec::with_capacity(4096);
	let (header_end, content_length) = loop {
		if bytes.len() > 1_048_576 {
			return Err(std::io::Error::other("provider request exceeded 1 MiB"));
		}
		let mut chunk = [0_u8; 4096];
		let read = stream.read(&mut chunk).await?;
		if read == 0 {
			return Err(std::io::Error::new(
				std::io::ErrorKind::UnexpectedEof,
				"provider request ended before headers",
			));
		}
		bytes.extend_from_slice(&chunk[..read]);
		if let Some(header_end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
			let header_end = header_end + 4;
			let headers = std::str::from_utf8(&bytes[..header_end]).map_err(std::io::Error::other)?;
			let content_length = headers
				.lines()
				.find_map(|line| {
					let (name, value) = line.split_once(':')?;
					name
						.eq_ignore_ascii_case("content-length")
						.then(|| value.trim().parse::<usize>().ok())
						.flatten()
				})
				.unwrap_or(0);
			break (header_end, content_length);
		}
	};
	while bytes.len() < header_end + content_length {
		let mut chunk = [0_u8; 4096];
		let read = stream.read(&mut chunk).await?;
		if read == 0 {
			return Err(std::io::Error::new(
				std::io::ErrorKind::UnexpectedEof,
				"provider request ended before its body",
			));
		}
		bytes.extend_from_slice(&chunk[..read]);
	}
	let head = std::str::from_utf8(&bytes[..header_end]).map_err(std::io::Error::other)?;
	let mut lines = head.lines();
	let request_line = lines
		.next()
		.ok_or_else(|| std::io::Error::other("missing request line"))?;
	let mut request_parts = request_line.split_whitespace();
	let method = request_parts.next().unwrap_or_default().to_owned();
	let path = request_parts.next().unwrap_or_default().to_owned();
	let headers = lines
		.filter_map(|line| {
			let (name, value) = line.split_once(':')?;
			Some((name.to_ascii_lowercase(), value.trim().to_owned()))
		})
		.collect();
	Ok(FixtureRequest {
		method,
		path,
		headers,
		body: bytes[header_end..header_end + content_length].to_vec(),
	})
}

async fn write_response(
	stream: &mut TcpStream,
	content_type: &str,
	body: &[u8],
) -> std::io::Result<()> {
	let head = format!(
		"HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: \
		 close\r\n\r\n",
		body.len()
	);
	stream.write_all(head.as_bytes()).await?;
	stream.write_all(body).await?;
	stream.shutdown().await
}

async fn write_status(stream: &mut TcpStream, status: &str, body: &[u8]) -> std::io::Result<()> {
	let head = format!(
		"HTTP/1.1 {status}\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: \
		 close\r\n\r\n",
		body.len()
	);
	stream.write_all(head.as_bytes()).await?;
	stream.write_all(body).await?;
	stream.shutdown().await
}

fn send_sigterm(pid: u32) {
	unsafe extern "C" {
		fn kill(pid: i32, signal: i32) -> i32;
	}
	const SIGTERM: i32 = 15;
	// SAFETY: `pid` came from the live child process and SIGTERM is a valid POSIX
	// signal.
	let result = unsafe { kill(pid as i32, SIGTERM) };
	assert_eq!(result, 0, "send SIGTERM to omp serve");
}
