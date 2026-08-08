//! SDK-shaped, byte-level facade probes transcribed from official client
//! traffic.
//!
//! Request framing follows the generated resources in
//! <https://github.com/openai/openai-python/tree/main/src/openai/resources> and
//! <https://github.com/anthropics/anthropic-sdk-python/tree/main/src/anthropic/resources>.
//! The multipart forms follow the clients' `files=` request construction. These
//! tests intentionally drive the in-process router: paths, auth/version
//! headers, JSON bodies, multipart boundaries, SSE, and binary framing match
//! the clients' wire traffic.

use std::{path::Path, sync::Arc};

use async_trait::async_trait;
use bytes::Bytes;
use futures::{
	StreamExt,
	stream::{self, BoxStream},
};
use http::{Request, StatusCode, header};
use http_body_util::{BodyExt, Full};
use omp_llm_catalog::{
	models::Availability,
	registry::{CredentialView, Registry},
};
use omp_llm_gateway::facade::{
	FacadeAuth, FacadeConfig, FacadeState, ModelsRepresentation, Router,
};
use omp_llm_types::{
	Accuracy, AspectRatio, AudioEncoding, BlobPart, ChatOutcome, ChatRequest, CountRequest,
	CountResponse, EmbedRequest, EmbedResponse, EmbeddingVector, GenerateImageRequest,
	GenerateVideoRequest, GenerationArtifact, GenerationState, GenerationStatus, ImageDone,
	ImageEvent, ItemKind, Props, SpeakChunk, SpeakEvent, SpeakRequest, StopReason, StreamPartKind,
	TranscribeRequest, TranscribeResponse, TranscriptSegment, TurnError, TurnErrorKind, TurnEvent,
	Usage, VideoResolution,
	facet::{
		Chat, CountTokens, Embed, Error, Facets, GenerationHandle, ImageGen, Speak, Transcribe,
		VideoGen,
	},
};
use omp_storage::blob::BlobStore;
use parking_lot::Mutex;
use serde_json::{Value, json};

const TOKEN: &str = "gateway-test-token";
const TOOL_ARGUMENTS: &str = r#"{ "city": "Zürich", "units": ["c", "f"] }"#;

struct Credentials;

impl CredentialView for Credentials {
	fn availability(&self, _provider: &str) -> Availability {
		Availability::Available
	}
}

#[derive(Default)]
struct FakeChat {
	seen_tool_arguments: Mutex<Option<Bytes>>,
	seen_requests:       Mutex<Vec<ChatRequest>>,
}

#[async_trait]
impl Chat for FakeChat {
	async fn turn(
		&self,
		request: ChatRequest,
		_executor: Option<Arc<dyn omp_llm_types::facet::Executor>>,
	) -> Result<BoxStream<'static, TurnEvent>, Error> {
		self.seen_requests.lock().push(request.clone());
		for item in &request.thread.items {
			if let ItemKind::ToolCall(call) = &item.kind {
				*self.seen_tool_arguments.lock() = Some(call.args_json.clone());
			}
		}
		let model = request.model;
		let error = match model.as_str() {
			"error-rate" => Some(turn_error(TurnErrorKind::RateLimited, 1_501)),
			"error-conflict" => Some(turn_error(TurnErrorKind::Conflict, 0)),
			"error-auth" => Some(turn_error(TurnErrorKind::Auth, 0)),
			_ => None,
		};
		if let Some(error) = error {
			return Ok(stream::iter([TurnEvent::Error(error)]).boxed());
		}
		let outcome = outcome(model);
		Ok(stream::iter([
			TurnEvent::PartStart {
				index:        0,
				kind:         StreamPartKind::Text,
				tool_call_id: "".into(),
				tool_name:    "".into(),
			},
			TurnEvent::PartDelta { index: 0, chunk: Bytes::from_static(b"hello") },
			TurnEvent::PartEnd { index: 0, signature: Bytes::new() },
			TurnEvent::Outcome(outcome),
		])
		.boxed())
	}
}

struct FakeCount;
#[async_trait]
impl CountTokens for FakeCount {
	async fn count(&self, _request: CountRequest) -> Result<CountResponse, Error> {
		Ok(CountResponse::builder()
			.tokens(17)
			.accuracy(Accuracy::Exact)
			.build())
	}
}

struct FakeEmbed;
#[async_trait]
impl Embed for FakeEmbed {
	async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse, Error> {
		let vectors = request
			.texts
			.iter()
			.enumerate()
			.map(|(index, _)| {
				EmbeddingVector::builder()
					.values(vec![index as f32, 0.5])
					.build()
			})
			.collect();
		Ok(EmbedResponse::builder()
			.vectors(vectors)
			.usage(usage())
			.build())
	}
}

struct FakeImages;
#[async_trait]
impl ImageGen for FakeImages {
	async fn generate(
		&self,
		request: GenerateImageRequest,
	) -> Result<BoxStream<'static, ImageEvent>, Error> {
		if request.prompt == "edit the logo" {
			assert_eq!(request.input_images[0].inline, Bytes::from_static(b"PNG-input"));
		}
		let image = BlobPart::builder()
			.hash([7; 32])
			.mime("image/png".into())
			.size(9)
			.inline(Bytes::from_static(b"PNG-output"))
			.build();
		let done = ImageDone::builder()
			.images(vec![image])
			.revised_prompt("revised".into())
			.text("".into())
			.unsupported(Vec::new())
			.props(Props::default())
			.build();
		Ok(stream::iter([ImageEvent::Done(done)]).boxed())
	}
}

struct FakeSpeak;
#[async_trait]
impl Speak for FakeSpeak {
	async fn speak(&self, request: SpeakRequest) -> Result<BoxStream<'static, SpeakEvent>, Error> {
		assert_eq!(request.encoding, AudioEncoding::Opus);
		let first = SpeakChunk::builder()
			.audio(Bytes::from_static(b"OggS"))
			.transcript_delta("".into())
			.build();
		let second = SpeakChunk::builder()
			.audio(Bytes::from_static(b"audio"))
			.transcript_delta("".into())
			.build();
		Ok(stream::iter([SpeakEvent::Chunk(first), SpeakEvent::Chunk(second)]).boxed())
	}
}

struct FakeTranscribe;
#[async_trait]
impl Transcribe for FakeTranscribe {
	async fn transcribe(&self, request: TranscribeRequest) -> Result<TranscribeResponse, Error> {
		assert_eq!(request.audio.inline, Bytes::from_static(b"RIFF-audio"));
		let text = if request.translate {
			"translated"
		} else {
			"transcribed"
		};
		Ok(TranscribeResponse::builder()
			.text(text.into())
			.language("en".into())
			.duration_ms(750)
			.segments(vec![
				TranscriptSegment::builder()
					.start_ms(0)
					.end_ms(750)
					.text(text.into())
					.build(),
			])
			.words(Vec::new())
			.unsupported(Vec::new())
			.props(Props::default())
			.build())
	}
}

struct FakeVideo {
	status: GenerationStatus,
}
#[async_trait]
impl VideoGen for FakeVideo {
	async fn submit(&self, request: GenerateVideoRequest) -> Result<GenerationHandle, Error> {
		assert_eq!(request.prompt, "waves");
		assert_eq!(request.duration_seconds, Some(4));
		assert_eq!(request.resolution, Some(VideoResolution::P720));
		assert_eq!(request.aspect_ratio, Some(AspectRatio::Wide16x9));
		Ok(GenerationHandle::builder().id("video-sdk-1".into()).build())
	}

	async fn get(&self, handle: GenerationHandle) -> Result<GenerationStatus, Error> {
		assert_eq!(handle.id, "video-sdk-1");
		Ok(self.status.clone())
	}

	async fn attach(
		&self,
		_handle: GenerationHandle,
	) -> Result<BoxStream<'static, GenerationStatus>, Error> {
		Ok(stream::empty().boxed())
	}

	async fn cancel(&self, _handle: GenerationHandle) -> Result<GenerationStatus, Error> {
		Ok(self.status.clone())
	}
}

fn usage() -> Usage {
	Usage::builder()
		.input_tokens(3)
		.output_tokens(2)
		.cache_read_tokens(0)
		.cache_write_tokens(0)
		.accuracy(Accuracy::Exact)
		.detail(Props::default())
		.build()
}

fn outcome(model: omp_core::SmolStr) -> ChatOutcome {
	ChatOutcome::builder()
		.output(Vec::new())
		.stop(StopReason::EndTurn)
		.usage(usage())
		.unsupported(Vec::new())
		.provider("fake".into())
		.model(model)
		.props(Props::default())
		.build()
}

fn turn_error(kind: TurnErrorKind, retry_after_ms: u64) -> TurnError {
	TurnError::builder()
		.kind(kind)
		.detail("canonical failure".into())
		.unsupported(Vec::new())
		.retry_after_ms(retry_after_ms)
		.build()
}

fn state(
	path: &Path,
	chat: Arc<FakeChat>,
	representation: ModelsRepresentation,
) -> Arc<FacadeState> {
	let blobs = BlobStore::open(path).expect("blob store");
	let reference = blobs.put(b"video artifact bytes").expect("video artifact");
	let blob = BlobPart::builder()
		.hash(reference.hash)
		.mime("video/mp4".into())
		.size(reference.size)
		.inline(Bytes::new())
		.build();
	let artifact = GenerationArtifact::builder()
		.blob(blob)
		.variant("video".into())
		.url("".into())
		.url_expires_at_ms(0)
		.build();
	let status = GenerationStatus::builder()
		.generation_id("video-sdk-1".into())
		.state(GenerationState::Completed)
		.progress_percent(100.0)
		.detail("".into())
		.artifacts(vec![artifact])
		.unsupported(Vec::new())
		.created_at_ms(1_000)
		.updated_at_ms(2_000)
		.props(Props::default())
		.build();
	Arc::new(FacadeState {
		facets:   Arc::new(Facets {
			chat: Some(chat),
			count_tokens: Some(Arc::new(FakeCount)),
			embed: Some(Arc::new(FakeEmbed)),
			image_gen: Some(Arc::new(FakeImages)),
			speak: Some(Arc::new(FakeSpeak)),
			transcribe: Some(Arc::new(FakeTranscribe)),
			video_gen: Some(Arc::new(FakeVideo { status })),
			..Facets::default()
		}),
		registry: Arc::new(parking_lot::RwLock::new(Registry::from_cards(
			&[],
			Arc::new(Credentials),
		))),
		blobs:    Arc::new(blobs),
		auth:     FacadeAuth::new(TOKEN),
		config:   FacadeConfig { models_representation: representation },
	})
}

fn openai(method: &str, uri: &str, body: impl Into<Bytes>) -> Request<Full<Bytes>> {
	Request::builder()
		.method(method)
		.uri(uri)
		.header(header::AUTHORIZATION, format!("Bearer {TOKEN}"))
		.header(header::CONTENT_TYPE, "application/json")
		.body(Full::new(body.into()))
		.expect("OpenAI SDK request")
}

fn anthropic(uri: &str, body: impl Into<Bytes>) -> Request<Full<Bytes>> {
	Request::post(uri)
		.header("x-api-key", TOKEN)
		.header("anthropic-version", "2023-06-01")
		.header(header::CONTENT_TYPE, "application/json")
		.body(Full::new(body.into()))
		.expect("Anthropic SDK request")
}

async fn bytes(response: http::Response<omp_llm_gateway::facade::FacadeBody>) -> Bytes {
	response
		.into_body()
		.collect()
		.await
		.expect("infallible body")
		.to_bytes()
}

async fn json_body(response: http::Response<omp_llm_gateway::facade::FacadeBody>) -> Value {
	serde_json::from_slice(&bytes(response).await).expect("JSON response")
}

#[tokio::test]
async fn foreign_request_fields_are_forwarded_without_silent_loss() {
	let directory = tempfile::tempdir().expect("temporary directory");
	let chat = Arc::new(FakeChat::default());
	let router = Router::new(state(directory.path(), Arc::clone(&chat), ModelsRepresentation::Auto));

	let response = router
		.route(openai(
			"POST",
			"/v1/chat/completions",
			br#"{"model":"sdk-model","messages":[{"role":"user","content":"hello"}],"temperature":0.25}"#
				.as_slice(),
		))
		.await;
	assert_eq!(response.status(), StatusCode::OK);

	let response = router
		.route(openai(
			"POST",
			"/v1/responses",
			br#"{"model":"sdk-model","input":"hello","store":true}"#.as_slice(),
		))
		.await;
	assert_eq!(response.status(), StatusCode::OK);

	let response = router
		.route(anthropic(
			"/v1/messages",
			br#"{"model":"claude-sdk","max_tokens":37,"messages":[{"role":"user","content":"hello"}]}"#
				.as_slice(),
		))
		.await;
	assert_eq!(response.status(), StatusCode::OK);

	let requests = chat.seen_requests.lock();
	assert_eq!(
		requests[0]
			.sampling
			.as_ref()
			.and_then(|sampling| sampling.temperature),
		Some(0.25),
	);
	assert_eq!(
		requests[1]
			.provider_options
			.as_ref()
			.and_then(|options| options.get_ns("openai", "store")),
		Some(&Value::Bool(true)),
	);
	assert_eq!(
		requests[2]
			.sampling
			.as_ref()
			.and_then(|sampling| sampling.max_output_tokens),
		Some(37),
	);
}

#[tokio::test]
async fn openai_and_anthropic_sdk_routes() {
	let directory = tempfile::tempdir().expect("temporary directory");
	let chat = Arc::new(FakeChat::default());
	let router = Router::new(state(directory.path(), Arc::clone(&chat), ModelsRepresentation::Auto));

	let openai_list = json_body(
		router
			.route(openai("GET", "/v1/models", Bytes::new()))
			.await,
	)
	.await;
	assert_eq!(openai_list["object"], "list");
	let anthropic_list_request = Request::get("/v1/models")
		.header("x-api-key", TOKEN)
		.header("anthropic-version", "2023-06-01")
		.body(Full::new(Bytes::new()))
		.expect("Anthropic models.list request");
	let anthropic_list = json_body(router.route(anthropic_list_request).await).await;
	assert_eq!(anthropic_list["has_more"], false);

	// The basic non-streaming Chat Completions slice is provider-backed in
	// `provider_http2`; this fixture remains for foreign-wire translation cases.

	let response = router.route(openai("POST", "/v1/chat/completions", format!(r#"{{"model":"sdk-model","stream":true,"stream_options":{{"include_usage":true}},"messages":[{{"role":"assistant","tool_calls":[{{"id":"call_sdk","type":"function","function":{{"name":"weather","arguments":{}}}}}]}},{{"role":"tool","tool_call_id":"call_sdk","content":"sunny"}}]}}"#, serde_json::to_string(TOOL_ARGUMENTS).expect("JSON string")))).await;
	assert_eq!(response.headers()[header::CONTENT_TYPE], "text/event-stream");
	let stream_bytes = bytes(response).await;
	assert!(stream_bytes.ends_with(b"data: [DONE]\n\n"));
	assert_eq!(chat.seen_tool_arguments.lock().as_deref(), Some(TOOL_ARGUMENTS.as_bytes()));

	let response = router
		.route(openai(
			"POST",
			"/v1/responses",
			br#"{"model":"sdk-model","input":"hello","stream":true}"#.as_slice(),
		))
		.await;
	let response_sse = bytes(response).await;
	assert!(
		response_sse
			.windows(b"event: response.completed".len())
			.any(|w| w == b"event: response.completed")
	);

	for input in [json!("one"), json!(["one", "two"])] {
		let response = router
			.route(openai(
				"POST",
				"/v1/embeddings",
				serde_json::to_vec(&json!({"model":"embed","input":input})).expect("JSON"),
			))
			.await;
		assert_eq!(json_body(response).await["object"], "list");
	}

	let response = router
		.route(openai(
			"POST",
			"/v1/images/generations",
			br#"{"model":"gpt-image-1","prompt":"draw","response_format":"b64_json"}"#.as_slice(),
		))
		.await;
	assert_eq!(json_body(response).await["data"][0]["b64_json"], "UE5HLW91dHB1dA==");

	let image_boundary = "openai-python-boundary-7MA4YWxkTrZu0gW";
	let image_body = format!(
		"--{image_boundary}\r\nContent-Disposition: form-data; \
		 name=\"model\"\r\n\r\ngpt-image-1\r\n--{image_boundary}\r\nContent-Disposition: form-data; \
		 name=\"prompt\"\r\n\r\nedit the logo\r\n--{image_boundary}\r\nContent-Disposition: \
		 form-data; \
		 name=\"response_format\"\r\n\r\nb64_json\r\n--{image_boundary}\r\nContent-Disposition: \
		 form-data; name=\"image\"; filename=\"logo.png\"\r\nContent-Type: \
		 image/png\r\n\r\nPNG-input\r\n--{image_boundary}--\r\n"
	);
	let edit = Request::post("/v1/images/edits")
		.header(header::AUTHORIZATION, format!("Bearer {TOKEN}"))
		.header(header::CONTENT_TYPE, format!("multipart/form-data; boundary={image_boundary}"))
		.body(Full::new(Bytes::from(image_body)))
		.expect("image edit request");
	assert_eq!(router.route(edit).await.status(), StatusCode::OK);

	let speech = router
		.route(openai(
			"POST",
			"/v1/audio/speech",
			br#"{"model":"tts-1","input":"hello","voice":"alloy","response_format":"opus"}"#
				.as_slice(),
		))
		.await;
	assert_eq!(speech.headers()[header::CONTENT_TYPE], "audio/ogg");
	assert_eq!(speech.headers()[header::TRANSFER_ENCODING], "chunked");
	assert_eq!(bytes(speech).await, Bytes::from_static(b"OggSaudio"));

	let audio_boundary = "openai-python-boundary-XyZ";
	let audio_body = Bytes::from(format!(
		"--{audio_boundary}\r\nContent-Disposition: form-data; \
		 name=\"model\"\r\n\r\nwhisper-1\r\n--{audio_boundary}\r\nContent-Disposition: form-data; \
		 name=\"response_format\"\r\n\r\nverbose_json\r\n--{audio_boundary}\r\nContent-Disposition: \
		 form-data; name=\"file\"; filename=\"speech.wav\"\r\nContent-Type: \
		 audio/wav\r\n\r\nRIFF-audio\r\n--{audio_boundary}--\r\n"
	));
	let transcription = Request::post("/v1/audio/transcriptions")
		.header(header::AUTHORIZATION, format!("Bearer {TOKEN}"))
		.header(header::CONTENT_TYPE, format!("multipart/form-data; boundary={audio_boundary}"))
		.body(Full::new(audio_body.clone()))
		.expect("transcription request");
	let transcript = json_body(router.route(transcription).await).await;
	assert_eq!(transcript["segments"][0]["text"], "transcribed");

	let json_audio = br#"{"model":"whisper-1","file":"UklGRi1hdWRpbw==","mime_type":"audio/wav"}"#;
	assert_eq!(
		json_body(
			router
				.route(openai("POST", "/v1/audio/transcriptions", json_audio.as_slice()))
				.await
		)
		.await["text"],
		"transcribed"
	);
	let translation = Request::post("/v1/audio/translations")
		.header(header::AUTHORIZATION, format!("Bearer {TOKEN}"))
		.header(header::CONTENT_TYPE, format!("multipart/form-data; boundary={audio_boundary}"))
		.body(Full::new(audio_body))
		.expect("translation request");
	assert_eq!(json_body(router.route(translation).await).await["text"], "translated");

	let submitted = router
		.route(openai(
			"POST",
			"/v1/videos",
			br#"{"model":"sora","prompt":"waves","seconds":"4","size":"1280x720"}"#.as_slice(),
		))
		.await;
	assert_eq!(submitted.status(), StatusCode::ACCEPTED);
	assert_eq!(json_body(submitted).await["id"], "video-sdk-1");
	assert_eq!(
		json_body(
			router
				.route(openai("GET", "/v1/videos/video-sdk-1", Bytes::new()))
				.await
		)
		.await["status"],
		"completed"
	);
	let content = router
		.route(openai("GET", "/v1/videos/video-sdk-1/content", Bytes::new()))
		.await;
	assert_eq!(content.headers()[header::CONTENT_TYPE], "video/mp4");
	assert_eq!(bytes(content).await, Bytes::from_static(b"video artifact bytes"));

	let message = router.route(anthropic("/v1/messages", br#"{"model":"claude-sdk","max_tokens":32,"stream":true,"messages":[{"role":"user","content":"hello"}]}"#.as_slice())).await;
	let message_sse = bytes(message).await;
	assert!(
		message_sse
			.windows(b"event: message_stop".len())
			.any(|w| w == b"event: message_stop")
	);
	let count = router
		.route(anthropic(
			"/v1/messages/count_tokens",
			br#"{"model":"claude-sdk","messages":[{"role":"user","content":"hello"}]}"#.as_slice(),
		))
		.await;
	assert_eq!(json_body(count).await["input_tokens"], 17);
}

#[tokio::test]
async fn models_representation_selection_is_header_then_listener_override() {
	let directory = tempfile::tempdir().expect("temporary directory");
	let chat = Arc::new(FakeChat::default());
	let auto = Router::new(state(directory.path(), Arc::clone(&chat), ModelsRepresentation::Auto));
	let absent = json_body(auto.route(openai("GET", "/v1/models", Bytes::new())).await).await;
	assert_eq!(absent["object"], "list", "Auto without anthropic-version is OpenAI");
	assert!(absent.get("has_more").is_none());

	let present_request = Request::get("/v1/models")
		.header(header::AUTHORIZATION, format!("Bearer {TOKEN}"))
		.header("anthropic-version", "2023-06-01")
		.body(Full::new(Bytes::new()))
		.expect("models request");
	let present = json_body(auto.route(present_request).await).await;
	assert_eq!(present["has_more"], false, "Auto with anthropic-version is Anthropic");
	assert!(present.get("object").is_none());

	let forced_openai =
		Router::new(state(directory.path(), Arc::clone(&chat), ModelsRepresentation::OpenAi));
	let override_request = Request::get("/v1/models")
		.header(header::AUTHORIZATION, format!("Bearer {TOKEN}"))
		.header("anthropic-version", "2023-06-01")
		.body(Full::new(Bytes::new()))
		.expect("models request");
	let forced = json_body(forced_openai.route(override_request).await).await;
	assert_eq!(forced["object"], "list", "OpenAI listener override wins over header");
	assert!(forced.get("has_more").is_none());

	let forced_anthropic =
		Router::new(state(directory.path(), chat, ModelsRepresentation::Anthropic));
	let forced = json_body(
		forced_anthropic
			.route(openai("GET", "/v1/models", Bytes::new()))
			.await,
	)
	.await;
	assert_eq!(forced["has_more"], false, "Anthropic listener override wins without header");
	assert!(forced.get("object").is_none());
}

#[tokio::test]
async fn facade_auth_accepts_gateway_credentials_and_never_provider_credentials() {
	let directory = tempfile::tempdir().expect("temporary directory");
	let router = Router::new(state(
		directory.path(),
		Arc::new(FakeChat::default()),
		ModelsRepresentation::Auto,
	));
	assert_eq!(
		router
			.route(openai("GET", "/v1/models", Bytes::new()))
			.await
			.status(),
		StatusCode::OK
	);
	assert_eq!(
		router
			.route(
				Request::get("/v1/models")
					.header("x-api-key", TOKEN)
					.body(Full::new(Bytes::new()))
					.expect("x-api-key request")
			)
			.await
			.status(),
		StatusCode::OK
	);
	for request in [
		Request::get("/v1/models")
			.body(Full::new(Bytes::new()))
			.expect("missing auth"),
		Request::get("/v1/models")
			.header(header::AUTHORIZATION, "Bearer provider-secret")
			.body(Full::new(Bytes::new()))
			.expect("provider bearer"),
		Request::get("/v1/models")
			.header("x-api-key", "provider-secret")
			.body(Full::new(Bytes::new()))
			.expect("provider key"),
	] {
		assert_eq!(router.route(request).await.status(), StatusCode::UNAUTHORIZED);
	}
	let with_provider_header = Request::post("/v1/chat/completions")
		.header(header::AUTHORIZATION, format!("Bearer {TOKEN}"))
		.header("OpenAI-Api-Key", "provider-secret")
		.header(header::CONTENT_TYPE, "application/json")
		.body(Full::new(Bytes::from_static(
			br#"{"model":"sdk-model","messages":[{"role":"user","content":"hello"}]}"#,
		)))
		.expect("request with attempted provider override");
	assert_eq!(router.route(with_provider_header).await.status(), StatusCode::OK);
}

#[tokio::test]
async fn canonical_turn_errors_use_each_vendor_envelope_and_status() {
	let directory = tempfile::tempdir().expect("temporary directory");
	let router = Router::new(state(
		directory.path(),
		Arc::new(FakeChat::default()),
		ModelsRepresentation::Auto,
	));
	let rate = router
		.route(openai(
			"POST",
			"/v1/chat/completions",
			br#"{"model":"error-rate","messages":[]}"#.as_slice(),
		))
		.await;
	assert_eq!(rate.status(), StatusCode::TOO_MANY_REQUESTS);
	assert_eq!(rate.headers()[header::RETRY_AFTER], "2");
	assert_eq!(json_body(rate).await["error"]["type"], "rate_limit_error");
	let conflict = router
		.route(openai(
			"POST",
			"/v1/responses",
			br#"{"model":"error-conflict","input":"x"}"#.as_slice(),
		))
		.await;
	assert_eq!(conflict.status(), StatusCode::BAD_REQUEST);
	assert_eq!(json_body(conflict).await["error"]["type"], "invalid_request_error");
	let auth = router
		.route(anthropic(
			"/v1/messages",
			br#"{"model":"error-auth","max_tokens":1,"messages":[]}"#.as_slice(),
		))
		.await;
	assert_eq!(auth.status(), StatusCode::UNAUTHORIZED);
	let envelope = json_body(auth).await;
	assert_eq!(envelope["type"], "error");
	assert_eq!(envelope["error"]["type"], "authentication_error");
}
