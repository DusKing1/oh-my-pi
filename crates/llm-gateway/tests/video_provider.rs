//! End-to-end proofs for the durable OpenAI video generation backend.

use std::{
	future::{Future, ready},
	sync::{
		Arc,
		atomic::{AtomicUsize, Ordering},
	},
	time::Duration,
};

use bytes::Bytes;
use futures::StreamExt;
use http::{Request, header};
use omp_core::Str;
use omp_llm_egress::{
	auth_inject::{AuthInjectLayer, CredentialLease, CredentialSource},
	client::{Body, EgressClient},
};
use omp_llm_gateway::videos::{OpenAiVideoBackend, VideoCredentialLeases, VideoError};
use omp_llm_types::{
	AspectRatio, BlobPart, GenerateVideoRequest, GenerationState, Props, VideoResolution,
	facet::VideoGen,
};
use omp_storage::blob::{BlobRef, BlobStore};
use tower::Layer;
use wiremock::{
	Mock, MockServer, ResponseTemplate,
	matchers::{header as header_match, method, path},
};

#[derive(Clone)]
struct TestCredentials {
	state: Arc<TestCredentialState>,
}

struct TestCredentialState {
	next: AtomicUsize,
}

impl TestCredentials {
	fn new() -> Self {
		Self { state: Arc::new(TestCredentialState { next: AtomicUsize::new(0) }) }
	}

	fn lease(id: u64) -> CredentialLease {
		CredentialLease::new("openai", id, 1)
	}
}

#[derive(Debug, thiserror::Error)]
#[error("test credential error")]
struct TestCredentialError;

impl VideoCredentialLeases for TestCredentials {
	fn select(&self) -> Result<CredentialLease, VideoError> {
		let id = if self.state.next.fetch_add(1, Ordering::SeqCst) == 0 {
			11
		} else {
			22
		};
		Ok(Self::lease(id))
	}

	fn by_id(&self, credential_id: u64) -> Result<CredentialLease, VideoError> {
		match credential_id {
			11 | 22 => Ok(Self::lease(credential_id)),
			_ => Err(VideoError::Credential("unknown test credential".into())),
		}
	}
}

impl CredentialSource for TestCredentials {
	type Error = TestCredentialError;

	fn lease(&self, _provider: &str) -> Result<Option<CredentialLease>, Self::Error> {
		Ok(None)
	}

	fn apply(
		&self,
		lease: &CredentialLease,
		request: &mut Request<Body>,
	) -> Result<(), Self::Error> {
		let secret = match lease.credential_id() {
			11 => "Bearer account-a",
			22 => "Bearer account-b",
			_ => return Err(TestCredentialError),
		};
		request
			.headers_mut()
			.insert(header::AUTHORIZATION, secret.parse().map_err(|_| TestCredentialError)?);
		Ok(())
	}

	fn refresh(
		&self,
		lease: CredentialLease,
	) -> impl Future<Output = Result<CredentialLease, Self::Error>> + Send + 'static {
		ready(Ok(lease))
	}
}

fn request(duration: u32) -> GenerateVideoRequest {
	let frame = Bytes::from_static(b"reference-frame");
	GenerateVideoRequest::builder()
		.model(Str::new_static("sora-2-pro"))
		.prompt(Str::new_static("a paper boat crossing a moonlit lake"))
		.duration_seconds(duration)
		.aspect_ratio(AspectRatio::Wide16x9)
		.resolution(VideoResolution::P1080)
		.start_frame(
			BlobPart::builder()
				.hash(*blake3::hash(&frame).as_bytes())
				.mime(Str::new_static("image/png"))
				.size(frame.len() as u64)
				.inline(frame)
				.build(),
		)
		.references(Vec::new())
		.props(Props::default())
		.build()
}

#[tokio::test]
async fn sora_jobs_are_durable_account_bound_terminal_once_and_cleanable() {
	let server = MockServer::start().await;
	Mock::given(method("POST"))
		.and(path("/videos"))
		.and(header_match("authorization", "Bearer account-a"))
		.respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
			"id":"video_a", "status":"queued", "created_at":100
		})))
		.expect(1)
		.mount(&server)
		.await;
	let polls = Arc::new(AtomicUsize::new(0));
	let poll_responses = Arc::clone(&polls);
	Mock::given(method("GET"))
		.and(path("/videos/video_a"))
		.and(header_match("authorization", "Bearer account-a"))
		.respond_with(move |_request: &wiremock::Request| {
			if poll_responses.fetch_add(1, Ordering::SeqCst) == 0 {
				ResponseTemplate::new(200).set_body_json(serde_json::json!({
					"id":"video_a", "status":"in_progress", "progress":37
				}))
			} else {
				ResponseTemplate::new(200).set_body_json(serde_json::json!({
					"id":"video_a",
					"status":"completed",
					"progress":100,
					"completed_at":102,
					"usage":{"video_seconds":8,"cost_nanos_usd":123456}
				}))
			}
		})
		.expect(2)
		.mount(&server)
		.await;
	Mock::given(method("GET"))
		.and(path("/videos/video_a/content"))
		.and(header_match("authorization", "Bearer account-a"))
		.respond_with(
			ResponseTemplate::new(200)
				.insert_header("content-type", "video/mp4")
				.set_body_bytes(b"durable-video"),
		)
		.expect(1)
		.mount(&server)
		.await;
	Mock::given(method("POST"))
		.and(path("/videos"))
		.and(header_match("authorization", "Bearer account-b"))
		.respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
			"id":"video_b", "status":"queued", "created_at":101
		})))
		.expect(1)
		.mount(&server)
		.await;
	Mock::given(method("DELETE"))
		.and(path("/videos/video_b"))
		.and(header_match("authorization", "Bearer account-b"))
		.respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"deleted":true})))
		.expect(1)
		.mount(&server)
		.await;

	let temporary = tempfile::tempdir().expect("temporary state");
	let store = Arc::new(BlobStore::open(temporary.path().join("blobs")).expect("blob store"));
	let credentials = TestCredentials::new();
	let service =
		AuthInjectLayer::new(credentials.clone()).layer(EgressClient::new(Duration::from_secs(5)));
	let backend = OpenAiVideoBackend::new(
		service.clone(),
		Arc::new(credentials.clone()),
		Arc::clone(&store),
		temporary.path().join("video-jobs"),
		Some(&server.uri()),
	)
	.expect("video backend")
	.with_poll_interval(Duration::from_millis(1));

	let first = backend
		.submit(request(8))
		.await
		.expect("submit first account");
	assert_ne!(first.id, "video_a", "upstream ids must remain gateway-private");
	let running = backend.get(first.clone()).await.expect("poll running job");
	assert_eq!(running.state, GenerationState::Running);
	assert_eq!(running.progress_percent, 37.0);
	assert_eq!(running.generation_id, first.id);

	let completed_payload = serde_json::to_vec(&serde_json::json!({
		"id":"evt_video_a",
		"object":"event",
		"created_at":102,
		"type":"video.completed",
		"data":{"id":"video_a"}
	}))
	.expect("webhook json");
	let completed = backend
		.receive_webhook(&completed_payload)
		.await
		.expect("completion webhook");
	assert_eq!(completed.state, GenerationState::Completed);
	assert_eq!(completed.generation_id, first.id);
	assert_eq!(
		completed
			.usage
			.as_ref()
			.expect("usage")
			.detail
			.get_ns("openai", "video_seconds")
			.and_then(serde_json::Value::as_u64),
		Some(8)
	);
	assert_eq!(completed.cost.expect("cost").nanos_usd, 123456);
	let blob = completed.artifacts[0]
		.blob
		.as_ref()
		.expect("durable artifact");
	assert_eq!(
		store
			.get(&BlobRef { hash: blob.hash, size: blob.size })
			.expect("blob bytes"),
		Bytes::from_static(b"durable-video")
	);

	let restarted = OpenAiVideoBackend::new(
		service,
		Arc::new(credentials.clone()),
		Arc::clone(&store),
		temporary.path().join("video-jobs"),
		Some(&server.uri()),
	)
	.expect("restarted backend")
	.with_poll_interval(Duration::from_millis(1));
	let attached: Vec<_> = restarted
		.attach(first.clone())
		.await
		.expect("reattach")
		.collect()
		.await;
	assert_eq!(attached.len(), 1, "a durable terminal job emits exactly one terminal snapshot");
	assert_eq!(attached[0].state, GenerationState::Completed);
	let duplicate = restarted
		.receive_webhook(&completed_payload)
		.await
		.expect("duplicate webhook");
	assert_eq!(duplicate.state, GenerationState::Completed);

	let second = restarted
		.submit(request(4))
		.await
		.expect("submit second account");
	assert_ne!(second.id, "video_b", "upstream ids must remain gateway-private");
	let cancelled = restarted
		.cancel(second.clone())
		.await
		.expect("cancel second job");
	assert_eq!(cancelled.state, GenerationState::Cancelled);
	assert_eq!(
		restarted
			.cancel(second)
			.await
			.expect("idempotent cancellation")
			.state,
		GenerationState::Cancelled
	);

	let invalid = restarted
		.submit(request(5))
		.await
		.expect_err("invalid duration must fail before egress");
	assert!(matches!(invalid, omp_llm_types::facet::Error::Unsupported(_)));
	let mut invalid_audio = request(4);
	invalid_audio.audio = Some(true);
	let invalid = restarted
		.submit(invalid_audio)
		.await
		.expect_err("unsupported audio control must fail before egress");
	assert!(matches!(invalid, omp_llm_types::facet::Error::Unsupported(_)));

	let requests = server
		.received_requests()
		.await
		.expect("recorded provider traffic");
	let submission = requests
		.iter()
		.find(|request| {
			request.method.as_str() == "POST"
				&& request
					.headers
					.get("authorization")
					.is_some_and(|value| value.as_bytes() == b"Bearer account-a")
		})
		.expect("account A submission");
	let multipart = String::from_utf8_lossy(&submission.body);
	assert!(multipart.contains("name=\"model\"\r\n\r\nsora-2-pro"));
	assert!(multipart.contains("name=\"prompt\"\r\n\r\na paper boat crossing a moonlit lake"));
	assert!(multipart.contains("name=\"seconds\"\r\n\r\n8"));
	assert!(multipart.contains("name=\"size\"\r\n\r\n1792x1024"));
	assert!(multipart.contains("name=\"input_reference\""));
	assert!(
		submission
			.body
			.windows(b"reference-frame".len())
			.any(|window| window == b"reference-frame")
	);

	restarted
		.cleanup(&first)
		.await
		.expect("cleanup completed job");
	let missing = restarted.get(first.clone()).await;
	assert!(missing.is_err());
}
