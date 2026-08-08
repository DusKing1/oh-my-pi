//! Windows named-pipe inference transport integration coverage.
#![cfg(windows)]

use std::{
	fs,
	os::windows::ffi::OsStrExt as _,
	path::Path,
	sync::{
		Arc,
		atomic::{AtomicUsize, Ordering},
	},
	time::{Duration, SystemTime, UNIX_EPOCH},
};

use futures::stream;
use omp_app::daemon::{DaemonConfig, DaemonHandle};
use omp_llm_broker::store::Store;
use omp_llm_gateway::{
	listener::LocalListener,
	local::{LocalEndpoint, connect},
};
use omp_proto::{inference::v1 as pb, thread::v1 as thread_pb};
use tokio::{
	io::{AsyncReadExt as _, AsyncWriteExt as _},
	net::{TcpListener, TcpStream},
	sync::oneshot,
	time::{sleep, timeout},
};
use windows_sys::Win32::System::Pipes::WaitNamedPipeW;

const PROVIDER: &str = "cerebras";
const MODEL: &str = "cerebras/zai-glm-4.7";
const TIMEOUT: Duration = Duration::from_secs(15);

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn named_pipe_turn_cancel_shutdown_and_restart() {
	let scratch = tempfile::tempdir().expect("create scratch directory");
	let project = scratch.path().join("project");
	let data = scratch.path().join("data");
	fs::create_dir_all(project.join(".omp")).expect("create project overlay directory");
	fs::create_dir_all(&data).expect("create data directory");

	let provider = TcpListener::bind(("127.0.0.1", 0))
		.await
		.expect("bind provider fixture");
	let provider_addr = provider.local_addr().expect("provider address");
	fs::write(
		project.join(".omp/providers.toml"),
		format!("[providers.{PROVIDER}]\nbase_url = \"http://{provider_addr}/v1\"\n"),
	)
	.expect("write provider overlay");
	let store = Store::open(data.join("broker.db")).expect("open broker store");
	store
		.upsert_api_key(PROVIDER, "pipe-test", b"provider-secret", now_ms())
		.expect("insert provider credential");
	drop(store);

	let (cancelled_tx, cancelled_rx) = oneshot::channel();
	let calls = Arc::new(AtomicUsize::new(0));
	let fixture_calls = Arc::clone(&calls);
	let fixture =
		tokio::spawn(async move { run_provider(provider, fixture_calls, cancelled_tx).await });

	let endpoint = LocalEndpoint::native(format!(
		r"\\.\pipe\omp-gateway-test-{}-{}",
		std::process::id(),
		now_ms()
	));
	let first = start(&endpoint, &project, &data).await;
	assert_eq!(first.readiness().local_endpoints, vec![endpoint.clone()]);
	assert!(
		LocalListener::bind(endpoint.as_path()).await.is_err(),
		"an active named pipe cannot be replaced as stale"
	);

	// The next pipe instance is listening before the first connection is handed
	// to tonic, so independent clients can become ready concurrently.
	let (one, two) = tokio::join!(connect_ready(&endpoint), connect_ready(&endpoint));
	let mut one = pb::inference_client::InferenceClient::new(one);
	let mut two = pb::inference_client::InferenceClient::new(two);
	let listed = one
		.list_models(pb::ListModelsRequest {
			provider:       PROVIDER.into(),
			facet:          pb::Facet::Chat as i32,
			available_only: true,
		})
		.await
		.expect("list models over named pipe")
		.into_inner();
	assert!(listed.models.iter().any(|model| model.id == MODEL));

	let mut completed = one
		.turn(stream::iter([turn_open("pipe-complete")]))
		.await
		.expect("open completed turn")
		.into_inner();
	let mut saw_outcome = false;
	while let Some(event) = timeout(TIMEOUT, completed.message())
		.await
		.expect("bounded turn")
		.expect("read completed turn")
	{
		if matches!(event.event, Some(pb::turn_event::Event::Outcome(_))) {
			saw_outcome = true;
		}
	}
	assert!(saw_outcome, "completed named-pipe turn has one outcome");

	let mut cancelled = two
		.turn(stream::iter([turn_open("pipe-cancel")]))
		.await
		.expect("open cancellable turn")
		.into_inner();
	loop {
		let event = timeout(TIMEOUT, cancelled.message())
			.await
			.expect("bounded first delta")
			.expect("read cancellable turn")
			.expect("cancellable turn remains open");
		if matches!(event.event, Some(pb::turn_event::Event::PartDelta(_))) {
			break;
		}
	}
	drop(cancelled);
	timeout(TIMEOUT, cancelled_rx)
		.await
		.expect("upstream cancellation stayed bounded")
		.expect("fixture observed upstream cancellation");

	first
		.shutdown()
		.await
		.expect("graceful named-pipe shutdown");
	assert!(!pipe_exists(&endpoint), "shutdown removes the named-pipe object");
	assert!(connect(&endpoint).await.is_err(), "removed pipe cannot accept a client");

	let restarted = start(&endpoint, &project, &data).await;
	let channel = connect_ready(&endpoint).await;
	omp_rpc::handshake(channel, "windows-restart-proof", &["inference"])
		.await
		.expect("restarted pipe serves the same tonic services");
	restarted
		.shutdown()
		.await
		.expect("shutdown restarted daemon");

	fixture.abort();
	let _ = fixture.await;
	assert_eq!(calls.load(Ordering::SeqCst), 2);
}

async fn start(endpoint: &LocalEndpoint, project: &Path, data: &Path) -> DaemonHandle {
	timeout(
		TIMEOUT,
		DaemonHandle::start(
			DaemonConfig::local(endpoint.clone())
				.with_project_dir(project.to_owned())
				.with_data_dir(data.to_owned())
				.with_gateway_token("local-facade-token"),
		),
	)
	.await
	.expect("daemon startup stayed bounded")
	.expect("start daemon")
}

async fn connect_ready(endpoint: &LocalEndpoint) -> tonic::transport::Channel {
	timeout(TIMEOUT, async {
		loop {
			match connect(endpoint).await {
				Ok(channel) => {
					omp_rpc::handshake(channel.clone(), "windows-pipe-proof", &["inference"])
						.await
						.expect("gateway handshake");
					return channel;
				},
				Err(_) => sleep(Duration::from_millis(10)).await,
			}
		}
	})
	.await
	.expect("named pipe became ready")
}

fn turn_open(turn_id: &str) -> pb::TurnFrame {
	let item = thread_pb::Item {
		seq: 0,
		kind: Some(thread_pb::item::Kind::Message(thread_pb::Message {
			role: thread_pb::Role::User as i32,
			parts: vec![thread_pb::Part {
				kind: Some(thread_pb::part::Kind::Text("named pipe prompt".into())),
				..Default::default()
			}],
			..Default::default()
		})),
		..Default::default()
	};
	pb::TurnFrame {
		frame: Some(pb::turn_frame::Frame::Open(pb::TurnRequest {
			turn_id: turn_id.into(),
			input: Some(pb::turn_request::Input::Seed(pb::Seed {
				thread: Some(thread_pb::Thread { items: vec![item], ..Default::default() }),
				..Default::default()
			})),
			params: Some(pb::ChatParams { model: MODEL.into(), ..Default::default() }),
			..Default::default()
		})),
		..Default::default()
	}
}

async fn run_provider(
	listener: TcpListener,
	calls: Arc<AtomicUsize>,
	cancelled: oneshot::Sender<()>,
) -> std::io::Result<()> {
	let mut cancelled = Some(cancelled);
	loop {
		let (mut stream, _) = listener.accept().await?;
		let path = read_request_path(&mut stream).await?;
		if !path.ends_with("/chat/completions") {
			write_response(&mut stream, b"not found", "text/plain").await?;
			continue;
		}
		let call = calls.fetch_add(1, Ordering::SeqCst);
		if call == 0 {
			let body = concat!(
				r#"data: {"id":"pipe","object":"chat.completion.chunk","created":0,"model":"zai-glm-4.7","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]}"#,
				"\n\n",
				r#"data: {"id":"pipe","object":"chat.completion.chunk","created":0,"model":"zai-glm-4.7","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
				"\n\n",
				"data: [DONE]\n\n"
			);
			write_response(&mut stream, body.as_bytes(), "text/event-stream").await?;
		} else {
			stream
				.write_all(
					b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n",
				)
				.await?;
			stream.write_all(b"data: {\"id\":\"pipe-cancel\",\"object\":\"chat.completion.chunk\",\"created\":0,\"model\":\"zai-glm-4.7\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"started\"},\"finish_reason\":null}]}\n\n").await?;
			stream.flush().await?;
			let mut byte = [0_u8; 1];
			let _ = stream.read(&mut byte).await;
			if let Some(sender) = cancelled.take() {
				let _ = sender.send(());
			}
		}
	}
}

async fn read_request_path(stream: &mut TcpStream) -> std::io::Result<String> {
	let mut bytes = Vec::with_capacity(4096);
	let (header_end, content_length, path) = loop {
		let mut chunk = [0_u8; 4096];
		let read = stream.read(&mut chunk).await?;
		if read == 0 {
			return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof));
		}
		bytes.extend_from_slice(&chunk[..read]);
		if let Some(end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
			let header_end = end + 4;
			let head = std::str::from_utf8(&bytes[..header_end]).map_err(std::io::Error::other)?;
			let path = head
				.split_whitespace()
				.nth(1)
				.unwrap_or_default()
				.to_owned();
			let content_length = head
				.lines()
				.find_map(|line| {
					let (name, value) = line.split_once(':')?;
					name
						.eq_ignore_ascii_case("content-length")
						.then(|| value.trim().parse::<usize>().ok())
						.flatten()
				})
				.unwrap_or(0);
			break (header_end, content_length, path);
		}
	};
	while bytes.len() < header_end + content_length {
		let mut chunk = [0_u8; 4096];
		let read = stream.read(&mut chunk).await?;
		if read == 0 {
			return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof));
		}
		bytes.extend_from_slice(&chunk[..read]);
	}
	Ok(path)
}

async fn write_response(
	stream: &mut TcpStream,
	body: &[u8],
	content_type: &str,
) -> std::io::Result<()> {
	stream
		.write_all(
			format!(
				"HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: \
				 close\r\n\r\n",
				body.len()
			)
			.as_bytes(),
		)
		.await?;
	stream.write_all(body).await?;
	stream.shutdown().await
}

fn pipe_exists(endpoint: &LocalEndpoint) -> bool {
	let mut name: Vec<u16> = endpoint.as_path().as_os_str().encode_wide().collect();
	name.push(0);
	// SAFETY: `name` is a live, NUL-terminated UTF-16 named-pipe path.
	unsafe { WaitNamedPipeW(name.as_ptr(), 0) != 0 }
}

fn now_ms() -> u64 {
	SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.unwrap_or_default()
		.as_millis() as u64
}
