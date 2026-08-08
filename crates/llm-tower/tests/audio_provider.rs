//! Byte-level remote audio provider fixtures.

use std::{
	collections::VecDeque,
	convert::Infallible,
	pin::Pin,
	sync::{
		Arc, Mutex,
		atomic::{AtomicBool, Ordering},
	},
	task::{Context, Poll},
};

use bytes::Bytes;
use futures::StreamExt;
use http::{Request, Response, header};
use http_body_util::{BodyExt, Full};
use hyper::body::{Body, Frame, SizeHint};
use omp_core::SmolStr;
use omp_llm_catalog::provider::load_builtin;
use omp_llm_egress::{auth_inject::AuthContext, client};
use omp_llm_openai::OpenAiAudioError;
use omp_llm_tower::{
	audio::{AudioAttemptError, AudioProviderAttempt},
	provider::ProviderRoute,
};
use omp_llm_types::{
	AudioEncoding, BlobPart, Props, SpeakEvent, SpeakRequest, TranscribeRequest,
	TranscriptionGranularity,
};
use tower::service_fn;

struct FixtureBody {
	frames:  VecDeque<Result<Frame<Bytes>, Infallible>>,
	dropped: Arc<AtomicBool>,
}

impl FixtureBody {
	fn chunks(chunks: impl IntoIterator<Item = &'static [u8]>, dropped: Arc<AtomicBool>) -> Self {
		Self {
			frames: chunks
				.into_iter()
				.map(|chunk| Ok(Frame::data(Bytes::from_static(chunk))))
				.collect(),
			dropped,
		}
	}
}

impl Drop for FixtureBody {
	fn drop(&mut self) {
		self.dropped.store(true, Ordering::SeqCst);
	}
}

impl Body for FixtureBody {
	type Data = Bytes;
	type Error = Infallible;

	fn poll_frame(
		mut self: Pin<&mut Self>,
		_cx: &mut Context<'_>,
	) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
		Poll::Ready(self.frames.pop_front())
	}

	fn is_end_stream(&self) -> bool {
		self.frames.is_empty()
	}

	fn size_hint(&self) -> SizeHint {
		SizeHint::default()
	}
}

fn openai(base_url: &str) -> omp_llm_catalog::provider::ProviderEntry {
	let mut provider = load_builtin().expect("built-in providers")["openai"].clone();
	provider.base_url = base_url.into();
	provider
}

fn speech() -> SpeakRequest {
	SpeakRequest::builder()
		.model("gpt-4o-mini-tts".into())
		.text("hello".into())
		.voice("alloy".into())
		.encoding(AudioEncoding::Mp3)
		.maybe_speed(Some(1.25))
		.instructions("warm".into())
		.props(Props::default())
		.build()
}

#[tokio::test]
async fn streamed_speech_preserves_chunks_auth_isolation_and_one_terminal() {
	let captured = Arc::new(Mutex::new(None::<Request<client::Body>>));
	let dropped = Arc::new(AtomicBool::new(false));
	let service = service_fn({
		let captured = Arc::clone(&captured);
		let dropped = Arc::clone(&dropped);
		move |request| {
			*captured.lock().expect("capture lock") = Some(request);
			let body =
				FixtureBody::chunks([b"ID3-one".as_slice(), b"-two".as_slice()], Arc::clone(&dropped));
			async move {
				Ok::<_, Infallible>(
					Response::builder()
						.header(header::CONTENT_TYPE, "audio/mpeg")
						.body(body)
						.expect("fixture response"),
				)
			}
		}
	});
	let attempt = AudioProviderAttempt::new(
		openai("https://fixture.test/v1"),
		ProviderRoute::default(),
		service,
	)
	.expect("OpenAI audio route");
	let events: Vec<_> = attempt
		.speak(speech())
		.await
		.expect("speech starts")
		.collect()
		.await;
	assert_eq!(
		events
			.iter()
			.filter(|event| matches!(event, SpeakEvent::Done(_)))
			.count(),
		1
	);
	let chunks: Vec<_> = events
		.iter()
		.filter_map(|event| match event {
			SpeakEvent::Chunk(chunk) => Some(chunk.audio.clone()),
			_ => None,
		})
		.collect();
	assert_eq!(chunks, [Bytes::from_static(b"ID3-one"), Bytes::from_static(b"-two")]);
	let done = events
		.iter()
		.find_map(|event| match event {
			SpeakEvent::Done(done) => Some(done),
			_ => None,
		})
		.expect("one terminal result");
	assert_eq!(done.audio.inline, Bytes::from_static(b"ID3-one-two"));
	let request = captured
		.lock()
		.expect("capture lock")
		.take()
		.expect("captured request");
	assert_eq!(request.uri(), "https://fixture.test/v1/audio/speech");
	assert_eq!(
		request
			.extensions()
			.get::<AuthContext>()
			.map(AuthContext::provider),
		Some("openai")
	);
	assert!(request.headers().get(header::AUTHORIZATION).is_none());
	let body: serde_json::Value = serde_json::from_slice(
		&request
			.into_body()
			.collect()
			.await
			.expect("request body")
			.to_bytes(),
	)
	.expect("speech JSON");
	assert_eq!(body["voice"], "alloy");
	assert_eq!(body["speed"], 1.25);
	assert!(dropped.load(Ordering::SeqCst));
}

#[tokio::test]
async fn dropping_speech_stream_closes_the_upstream_body() {
	let dropped = Arc::new(AtomicBool::new(false));
	let service = service_fn({
		let dropped = Arc::clone(&dropped);
		move |_request| {
			let body = FixtureBody::chunks(
				[b"ID3-first".as_slice(), b"-second".as_slice(), b"-third".as_slice()],
				Arc::clone(&dropped),
			);
			async move {
				Ok::<_, Infallible>(
					Response::builder()
						.header(header::CONTENT_TYPE, "audio/mpeg")
						.body(body)
						.expect("fixture response"),
				)
			}
		}
	});
	let attempt = AudioProviderAttempt::new(
		openai("https://fixture.test/v1"),
		ProviderRoute::default(),
		service,
	)
	.expect("OpenAI audio route");
	let mut stream = attempt.speak(speech()).await.expect("speech starts");
	assert!(matches!(stream.next().await, Some(SpeakEvent::Chunk(_))));
	assert!(!dropped.load(Ordering::SeqCst));
	drop(stream);
	assert!(dropped.load(Ordering::SeqCst));
}

#[tokio::test]
async fn multipart_transcription_decodes_timestamps_diarization_and_usage() {
	let captured = Arc::new(Mutex::new(None::<Request<client::Body>>));
	let service = service_fn({
		let captured = Arc::clone(&captured);
		move |request| {
			*captured.lock().expect("capture lock") = Some(request);
			async move {
				Ok::<_, Infallible>(
					Response::builder()
						.header(header::CONTENT_TYPE, "application/json")
						.body(Full::new(Bytes::from_static(
							br#"{
						"text":"hello world","language":"en","duration":1.25,
						"segments":[{"start":0.0,"end":1.25,"text":"hello world","speaker":"A"}],
						"words":[],
						"usage":{"input_tokens":12,"output_tokens":3}
					}"#,
						)))
						.expect("fixture response"),
				)
			}
		}
	});
	let audio = Bytes::from_static(b"RIFF\x04\x00\x00\x00WAVE");
	let request = TranscribeRequest::builder()
		.model("gpt-4o-transcribe-diarize".into())
		.audio(
			BlobPart::builder()
				.hash(*blake3::hash(&audio).as_bytes())
				.mime("audio/wav".into())
				.size(audio.len() as u64)
				.inline(audio.clone())
				.build(),
		)
		.language("en".into())
		.prompt("OMP".into())
		.translate(false)
		.granularities(vec![TranscriptionGranularity::Segment])
		.diarize(true)
		.maybe_temperature(Some(0.2))
		.props(Props::default())
		.build();
	let attempt = AudioProviderAttempt::new(
		openai("https://fixture.test/v1"),
		ProviderRoute::default(),
		service,
	)
	.expect("OpenAI audio route");
	let response = attempt.transcribe(request).await.expect("transcription");
	assert_eq!(response.duration_ms, 1250);
	assert_eq!(response.segments[0].speaker, Some(0));
	assert!(response.words.is_empty());
	assert_eq!(response.usage.as_ref().map(|usage| usage.input_tokens), Some(12));
	let request = captured
		.lock()
		.expect("capture lock")
		.take()
		.expect("captured request");
	let content_type = request.headers()[header::CONTENT_TYPE]
		.to_str()
		.expect("content type");
	assert!(content_type.starts_with("multipart/form-data; boundary=omp-audio-"));
	assert_eq!(
		request
			.extensions()
			.get::<AuthContext>()
			.map(AuthContext::provider),
		Some("openai")
	);
	let multipart = request
		.into_body()
		.collect()
		.await
		.expect("multipart body")
		.to_bytes();
	let multipart_text = String::from_utf8_lossy(&multipart);
	assert!(multipart_text.contains("name=\"file\"; filename=\"audio.wav\""));
	assert!(multipart_text.contains("name=\"response_format\"\r\n\r\ndiarized_json"));
	assert!(multipart_text.contains("name=\"chunking_strategy\"\r\n\r\nauto"));
	assert!(!multipart_text.contains("name=\"timestamp_granularities[]\""));
	assert!(multipart_text.contains("name=\"language\"\r\n\r\nen"));
	assert!(multipart_text.contains("name=\"prompt\"\r\n\r\nOMP"));
	assert!(
		multipart
			.windows(audio.len())
			.any(|window| window == audio.as_ref())
	);
}

#[tokio::test]
async fn invalid_recording_media_is_rejected_before_egress() {
	let called = Arc::new(AtomicBool::new(false));
	let service = service_fn({
		let called = Arc::clone(&called);
		move |_request: Request<client::Body>| {
			called.store(true, Ordering::SeqCst);
			async move {
				Ok::<_, Infallible>(Response::new(Full::new(Bytes::from_static(
					br#"{"text":"should not run"}"#,
				))))
			}
		}
	});
	let audio = Bytes::from_static(b"not audio");
	let request = TranscribeRequest::builder()
		.model("whisper-1".into())
		.audio(
			BlobPart::builder()
				.hash(*blake3::hash(&audio).as_bytes())
				.mime("text/plain".into())
				.size(audio.len() as u64)
				.inline(audio)
				.build(),
		)
		.language(SmolStr::default())
		.prompt(SmolStr::default())
		.translate(false)
		.granularities(Vec::new())
		.diarize(false)
		.props(Props::default())
		.build();
	let attempt = AudioProviderAttempt::new(
		openai("https://fixture.test/v1"),
		ProviderRoute::default(),
		service,
	)
	.expect("OpenAI audio route");
	assert!(matches!(
		attempt.transcribe(request).await,
		Err(AudioAttemptError::Encode(OpenAiAudioError::Invalid(_)))
	));
	assert!(!called.load(Ordering::SeqCst));
}

#[tokio::test]
async fn azure_audio_version_is_selected_by_catalog_data() {
	let captured = Arc::new(Mutex::new(None::<Request<client::Body>>));
	let service = service_fn({
		let captured = Arc::clone(&captured);
		move |request: Request<client::Body>| {
			*captured.lock().expect("capture lock") = Some(request);
			async move {
				Ok::<_, Infallible>(
					Response::builder()
						.header(header::CONTENT_TYPE, "audio/mpeg")
						.body(FixtureBody::chunks(
							[b"ID3-audio".as_slice()],
							Arc::new(AtomicBool::new(false)),
						))
						.expect("fixture response"),
				)
			}
		}
	});
	let provider = load_builtin().expect("built-in providers")["azure"].clone();
	let mut route = ProviderRoute::default();
	route.region = "eastus".into();
	let attempt = AudioProviderAttempt::new(provider, route, service).expect("Azure audio route");
	let mut request = speech();
	request.instructions = SmolStr::default();
	let _: Vec<_> = attempt
		.speak(request)
		.await
		.expect("speech starts")
		.collect()
		.await;
	let request = captured
		.lock()
		.expect("capture lock")
		.take()
		.expect("captured request");
	assert_eq!(
		request.uri(),
		"https://eastus.openai.azure.com/openai/deployments/gpt-4o-mini-tts/audio/speech?api-version=2025-04-01-preview",
	);
	let body: serde_json::Value = serde_json::from_slice(
		&request
			.into_body()
			.collect()
			.await
			.expect("request body")
			.to_bytes(),
	)
	.expect("Azure speech JSON");
	assert!(body.get("model").is_none(), "deployment path owns the model");
	assert!(body.get("instructions").is_none());
}
