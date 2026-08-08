//! OpenAI-compatible speech synthesis, transcription, and translation facades.

use std::{convert::Infallible, fmt::Display, sync::Arc};

use bytes::Bytes;
use futures::{StreamExt, stream};
use http::{Request, Response, StatusCode, header};
use http_body_util::{BodyExt, StreamBody};
use hyper::body::{Body, Frame};
use omp_core::{SmolStr, base64};
use omp_llm_types::{
	AudioEncoding, BlobPart, Props, SpeakEvent, SpeakRequest, TranscribeRequest, TranscribeResponse,
	TranscriptionGranularity,
};
use omp_storage::blob::BlobRef;
use serde::{Deserialize, Serialize};

use super::{
	FacadeError, FacadeResponse, FacadeState, Vendor, error_response, json_response, read_json,
};

#[derive(Deserialize)]
struct SpeechRequest {
	model:           SmolStr,
	input:           SmolStr,
	voice:           SmolStr,
	#[serde(default = "default_mp3", alias = "format")]
	response_format: SmolStr,
	#[serde(default)]
	speed:           Option<f64>,
	#[serde(default)]
	instructions:    SmolStr,
}

#[derive(Deserialize)]
struct JsonTranscriptionRequest {
	model:                   SmolStr,
	#[serde(alias = "audio")]
	file:                    JsonAudio,
	#[serde(default = "default_audio_mime")]
	mime_type:               SmolStr,
	#[serde(default)]
	language:                SmolStr,
	#[serde(default)]
	prompt:                  SmolStr,
	#[serde(default = "default_json")]
	response_format:         SmolStr,
	#[serde(default)]
	temperature:             Option<f64>,
	#[serde(default)]
	timestamp_granularities: Vec<SmolStr>,
	#[serde(default)]
	diarize:                 bool,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum JsonAudio {
	Base64(SmolStr),
	Object {
		data:      SmolStr,
		#[serde(default)]
		mime_type: Option<SmolStr>,
	},
}

struct ParsedTranscription {
	model:           SmolStr,
	audio:           BlobPart,
	language:        SmolStr,
	prompt:          SmolStr,
	response_format: SmolStr,
	temperature:     Option<f64>,
	granularities:   Vec<TranscriptionGranularity>,
	diarize:         bool,
}

#[derive(Serialize)]
struct SimpleTranscript {
	text: SmolStr,
}

#[derive(Serialize)]
struct VerboseTranscript {
	task:     &'static str,
	language: SmolStr,
	duration: f64,
	text:     SmolStr,
	segments: Vec<VerboseSegment>,
	words:    Vec<VerboseWord>,
}

#[derive(Serialize)]
struct VerboseSegment {
	id:      usize,
	start:   f64,
	end:     f64,
	text:    SmolStr,
	#[serde(skip_serializing_if = "Option::is_none")]
	speaker: Option<u32>,
}

#[derive(Serialize)]
struct VerboseWord {
	word:    SmolStr,
	start:   f64,
	end:     f64,
	#[serde(skip_serializing_if = "Option::is_none")]
	speaker: Option<u32>,
}

fn default_mp3() -> SmolStr {
	SmolStr::new("mp3")
}
fn default_json() -> SmolStr {
	SmolStr::new("json")
}
fn default_audio_mime() -> SmolStr {
	SmolStr::new("application/octet-stream")
}

pub(crate) async fn handle<B>(request: Request<B>, state: Arc<FacadeState>) -> FacadeResponse
where
	B: Body<Data = Bytes> + Send + 'static,
	B::Error: Display,
{
	match request.uri().path() {
		"/v1/audio/speech" => speech(request, Arc::clone(&state)).await,
		"/v1/audio/transcriptions" => transcription(request, &state, false).await,
		"/v1/audio/translations" => transcription(request, &state, true).await,
		_ => invalid("audio route not found"),
	}
}

async fn speech<B>(request: Request<B>, state: Arc<FacadeState>) -> FacadeResponse
where
	B: Body<Data = Bytes> + Send + 'static,
	B::Error: Display,
{
	let wire: SpeechRequest = match read_json(request, Vendor::OpenAi).await {
		Ok(wire) => wire,
		Err(response) => return *response,
	};
	let (encoding, content_type) = match speech_encoding(&wire.response_format) {
		Ok(value) => value,
		Err(detail) => return invalid(detail),
	};
	let Some(speak) = &state.facets.speak else {
		return invalid("speech synthesis is not available");
	};
	let canonical = SpeakRequest::builder()
		.model(wire.model)
		.text(wire.input)
		.voice(wire.voice)
		.encoding(encoding)
		.maybe_speed(wire.speed)
		.instructions(wire.instructions)
		.props(Props::default())
		.build();
	let events = match speak.speak(canonical).await {
		Ok(events) => events,
		Err(error) => return error_response(Vendor::OpenAi, FacadeError::Facet(error)),
	};
	let blobs = Arc::clone(&state.blobs);
	let frames = async_stream::stream! {
		let mut events = events;
		let mut streamed = false;
		while let Some(event) = events.next().await {
			match event {
				SpeakEvent::Chunk(chunk) => {
					streamed = true;
					yield Ok::<_, Infallible>(Frame::data(chunk.audio));
				},
				SpeakEvent::Done(done) => {
					if done.audio.inline.is_empty() {
						if !streamed
							&& let Ok(audio) = blobs.get(&BlobRef {
								hash: done.audio.hash,
								size: done.audio.size,
							})
						{
							yield Ok::<_, Infallible>(Frame::data(audio));
						}
					} else {
						if blobs.put(&done.audio.inline).is_err() {
							break;
						}
						if !streamed {
							yield Ok::<_, Infallible>(Frame::data(done.audio.inline));
						}
					}
					break;
				},
				_ => {},
			}
		}
	};
	Response::builder()
		.status(StatusCode::OK)
		.header(header::CONTENT_TYPE, content_type)
		.header(header::TRANSFER_ENCODING, "chunked")
		.body(BodyExt::boxed_unsync(StreamBody::new(frames)))
		.expect("static speech response is valid")
}

fn speech_encoding(value: &str) -> Result<(AudioEncoding, &'static str), &'static str> {
	match value {
		"mp3" => Ok((AudioEncoding::Mp3, "audio/mpeg")),
		"opus" => Ok((AudioEncoding::Opus, "audio/ogg")),
		"wav" => Ok((AudioEncoding::Wav, "audio/wav")),
		"pcm" | "pcm16" => Ok((AudioEncoding::Pcm16, "audio/pcm")),
		"aac" => Ok((AudioEncoding::Aac, "audio/aac")),
		"flac" => Ok((AudioEncoding::Flac, "audio/flac")),
		_ => Err("response_format must be mp3, opus, wav, pcm, aac, or flac"),
	}
}

async fn transcription<B>(
	request: Request<B>,
	state: &FacadeState,
	translate: bool,
) -> FacadeResponse
where
	B: Body<Data = Bytes> + Send + 'static,
	B::Error: Display,
{
	let is_multipart = request
		.headers()
		.get(header::CONTENT_TYPE)
		.and_then(|value| value.to_str().ok())
		.is_some_and(|value| value.starts_with("multipart/form-data"));
	let mut parsed = if is_multipart {
		match parse_multipart(request).await {
			Ok(parsed) => parsed,
			Err(error) => return invalid(error),
		}
	} else {
		let wire: JsonTranscriptionRequest = match read_json(request, Vendor::OpenAi).await {
			Ok(wire) => wire,
			Err(response) => return *response,
		};
		match parse_json_transcription(wire) {
			Ok(parsed) => parsed,
			Err(error) => return invalid(error),
		}
	};
	if !matches!(parsed.response_format.as_str(), "json" | "verbose_json" | "diarized_json") {
		return invalid("response_format must be json, verbose_json, or diarized_json");
	}
	if parsed.response_format == "diarized_json" {
		parsed.diarize = true;
	}
	if !translate && parsed.granularities.is_empty() {
		if parsed.diarize {
			parsed.granularities = vec![TranscriptionGranularity::Segment];
		} else if parsed.response_format == "verbose_json" {
			parsed.granularities =
				vec![TranscriptionGranularity::Segment, TranscriptionGranularity::Word];
		}
	}
	if parsed.audio.inline.is_empty() {
		return invalid("file must not be empty");
	}
	if let Err(error) = state.blobs.put(&parsed.audio.inline) {
		return error_response(
			Vendor::OpenAi,
			FacadeError::Facet(omp_llm_types::facet::Error::Transport(error.to_string().into())),
		);
	}
	let Some(transcribe) = &state.facets.transcribe else {
		return invalid("transcription is not available");
	};
	let mut props = Props::default();
	props.insert_ns("openai", "response_format", parsed.response_format.as_str().into());
	let request = TranscribeRequest::builder()
		.model(parsed.model)
		.audio(parsed.audio)
		.language(parsed.language)
		.prompt(parsed.prompt)
		.translate(translate)
		.granularities(parsed.granularities)
		.diarize(parsed.diarize)
		.maybe_temperature(parsed.temperature)
		.props(props)
		.build();
	let response = match transcribe.transcribe(request).await {
		Ok(response) => response,
		Err(error) => return error_response(Vendor::OpenAi, FacadeError::Facet(error)),
	};
	transcription_response(response, parsed.response_format.as_str(), translate)
}

fn parse_json_transcription(
	wire: JsonTranscriptionRequest,
) -> Result<ParsedTranscription, SmolStr> {
	let (encoded, mime) = match wire.file {
		JsonAudio::Base64(data) => (data, wire.mime_type),
		JsonAudio::Object { data, mime_type } => (data, mime_type.unwrap_or(wire.mime_type)),
	};
	let audio = base64::decode(encoded.as_bytes())
		.into_vec()
		.map_err(|_| SmolStr::new("file must be valid base64"))?;
	let audio = Bytes::from(audio);
	Ok(ParsedTranscription {
		model:           wire.model,
		audio:           blob_part(audio, mime),
		language:        wire.language,
		prompt:          wire.prompt,
		response_format: wire.response_format,
		temperature:     wire.temperature,
		granularities:   parse_granularities(&wire.timestamp_granularities)?,
		diarize:         wire.diarize,
	})
}

async fn parse_multipart<B>(request: Request<B>) -> Result<ParsedTranscription, SmolStr>
where
	B: Body<Data = Bytes> + Send + 'static,
	B::Error: Display,
{
	let content_type = request
		.headers()
		.get(header::CONTENT_TYPE)
		.and_then(|value| value.to_str().ok())
		.ok_or_else(|| SmolStr::new("multipart Content-Type is required"))?;
	let boundary = multer::parse_boundary(content_type)
		.map_err(|error| SmolStr::new(format!("invalid multipart boundary: {error}")))?;
	let bytes = request
		.into_body()
		.collect()
		.await
		.map_err(|error| SmolStr::new(format!("failed to read multipart body: {error}")))?
		.to_bytes();
	let source = stream::once(async move { Ok::<Bytes, Infallible>(bytes) });
	let mut multipart = multer::Multipart::new(source, boundary);
	let mut model = None;
	let mut audio = None;
	let mut language = SmolStr::default();
	let mut prompt = SmolStr::default();
	let mut response_format = default_json();
	let mut temperature = None;
	let mut timestamp_granularities = Vec::new();
	let mut diarize = false;
	while let Some(field) = multipart
		.next_field()
		.await
		.map_err(|error| SmolStr::new(format!("invalid multipart body: {error}")))?
	{
		let name = field.name().unwrap_or_default().to_owned();
		let mime = field
			.content_type()
			.map_or_else(default_audio_mime, |value| SmolStr::from(value.to_string()));
		let data = field
			.bytes()
			.await
			.map_err(|error| SmolStr::new(format!("invalid multipart field: {error}")))?;
		match name.as_str() {
			"file" => audio = Some(blob_part(data, mime)),
			"model" => model = Some(text_field(data, "model")?),
			"language" => language = text_field(data, "language")?,
			"prompt" => prompt = text_field(data, "prompt")?,
			"response_format" => response_format = text_field(data, "response_format")?,
			"temperature" => {
				temperature = Some(
					text_field(data, "temperature")?
						.parse()
						.map_err(|_| SmolStr::new("invalid temperature"))?,
				);
			},
			"timestamp_granularities[]" | "timestamp_granularities" => {
				timestamp_granularities.push(text_field(data, "timestamp_granularities")?);
			},
			"diarize" => diarize = bool_field(data, "diarize")?,
			_ => {},
		}
	}
	Ok(ParsedTranscription {
		model: model.ok_or_else(|| SmolStr::new("model is required"))?,
		audio: audio.ok_or_else(|| SmolStr::new("file is required"))?,
		language,
		prompt,
		response_format,
		temperature,
		granularities: parse_granularities(&timestamp_granularities)?,
		diarize,
	})
}

fn parse_granularities(values: &[SmolStr]) -> Result<Vec<TranscriptionGranularity>, SmolStr> {
	values
		.iter()
		.map(|value| match value.as_str() {
			"segment" => Ok(TranscriptionGranularity::Segment),
			"word" => Ok(TranscriptionGranularity::Word),
			_ => Err(SmolStr::new("timestamp_granularities values must be segment or word")),
		})
		.collect()
}

fn text_field(bytes: Bytes, name: &str) -> Result<SmolStr, SmolStr> {
	let text =
		std::str::from_utf8(&bytes).map_err(|_| SmolStr::new(format!("{name} must be UTF-8")))?;
	Ok(SmolStr::from(text))
}

fn bool_field(bytes: Bytes, name: &str) -> Result<bool, SmolStr> {
	match text_field(bytes, name)?.as_str() {
		"true" | "1" => Ok(true),
		"false" | "0" => Ok(false),
		_ => Err(SmolStr::new(format!("{name} must be true or false"))),
	}
}

fn blob_part(bytes: Bytes, mime: SmolStr) -> BlobPart {
	BlobPart::builder()
		.hash(*blake3::hash(&bytes).as_bytes())
		.mime(mime)
		.size(bytes.len() as u64)
		.inline(bytes)
		.build()
}

fn transcription_response(
	response: TranscribeResponse,
	format: &str,
	translate: bool,
) -> FacadeResponse {
	if format == "json" {
		return json_response(StatusCode::OK, &SimpleTranscript { text: response.text });
	}
	let segments = response
		.segments
		.into_iter()
		.enumerate()
		.map(|(id, segment)| VerboseSegment {
			id,
			start: segment.start_ms as f64 / 1000.0,
			end: segment.end_ms as f64 / 1000.0,
			text: segment.text,
			speaker: segment.speaker,
		})
		.collect();
	let words = response
		.words
		.into_iter()
		.map(|word| VerboseWord {
			word:    word.word,
			start:   word.start_ms as f64 / 1000.0,
			end:     word.end_ms as f64 / 1000.0,
			speaker: word.speaker,
		})
		.collect();
	json_response(StatusCode::OK, &VerboseTranscript {
		task: if translate { "translate" } else { "transcribe" },
		language: response.language,
		duration: response.duration_ms as f64 / 1000.0,
		text: response.text,
		segments,
		words,
	})
}

fn invalid(detail: impl Into<SmolStr>) -> FacadeResponse {
	error_response(Vendor::OpenAi, FacadeError::Invalid(detail.into()))
}

#[cfg(test)]
mod tests {
	use async_trait::async_trait;
	use http_body_util::Full;
	use omp_llm_catalog::{
		models::Availability,
		registry::{CredentialView, Registry},
	};
	use omp_llm_types::{
		SpeakChunk, SpeakDone, TranscriptSegment, TranscriptWord,
		facet::{Error, Facets, Speak, Transcribe},
	};
	use omp_storage::blob::BlobStore;

	use super::*;

	struct Credentials;

	impl CredentialView for Credentials {
		fn availability(&self, _provider: &str) -> Availability {
			Availability::Available
		}
	}

	struct FakeSpeak;

	#[async_trait]
	impl Speak for FakeSpeak {
		async fn speak(
			&self,
			request: SpeakRequest,
		) -> Result<futures::stream::BoxStream<'static, SpeakEvent>, Error> {
			assert_eq!(request.encoding, AudioEncoding::Mp3);
			let audio = Bytes::from_static(b"\x01\x02audio");
			Ok(stream::iter([
				SpeakEvent::Chunk(
					SpeakChunk::builder()
						.audio(audio.clone())
						.transcript_delta(SmolStr::new(""))
						.build(),
				),
				SpeakEvent::Done(
					SpeakDone::builder()
						.audio(blob_part(audio, "audio/mpeg".into()))
						.duration_ms(10)
						.unsupported(Vec::new())
						.props(Props::default())
						.build(),
				),
			])
			.boxed())
		}
	}

	struct FakeTranscribe;

	#[async_trait]
	impl Transcribe for FakeTranscribe {
		async fn transcribe(&self, request: TranscribeRequest) -> Result<TranscribeResponse, Error> {
			let task = if request.translate {
				"translated"
			} else {
				"transcribed"
			};
			Ok(TranscribeResponse::builder()
				.text(SmolStr::new(task))
				.language(SmolStr::new("en"))
				.duration_ms(500)
				.segments(vec![
					TranscriptSegment::builder()
						.start_ms(0)
						.end_ms(500)
						.text(SmolStr::new(task))
						.build(),
				])
				.words(Vec::new())
				.unsupported(Vec::new())
				.props(Props::default())
				.build())
		}
	}
	fn speech_state(directory: &std::path::Path) -> Arc<FacadeState> {
		Arc::new(FacadeState {
			facets:   Arc::new(Facets {
				speak: Some(Arc::new(FakeSpeak)),
				transcribe: Some(Arc::new(FakeTranscribe)),
				..Facets::default()
			}),
			registry: Arc::new(parking_lot::RwLock::new(Registry::from_cards(
				&[],
				Arc::new(Credentials),
			))),
			blobs:    Arc::new(BlobStore::open(directory).expect("blob store")),
			auth:     super::super::FacadeAuth::new("token"),
			config:   super::super::FacadeConfig::default(),
		})
	}

	#[tokio::test]
	async fn verbose_response_includes_segments_and_words() {
		let response = TranscribeResponse::builder()
			.text(SmolStr::new("hello"))
			.language(SmolStr::new("en"))
			.duration_ms(1250)
			.segments(vec![
				TranscriptSegment::builder()
					.start_ms(0)
					.end_ms(1000)
					.text(SmolStr::new("hello"))
					.build(),
			])
			.words(vec![
				TranscriptWord::builder()
					.start_ms(0)
					.end_ms(1000)
					.word(SmolStr::new("hello"))
					.build(),
			])
			.unsupported(Vec::new())
			.props(Props::default())
			.build();
		let facade = transcription_response(response, "verbose_json", false);
		assert_eq!(facade.status(), StatusCode::OK);
		let bytes = facade
			.into_body()
			.collect()
			.await
			.expect("infallible")
			.to_bytes();
		let value: serde_json::Value = serde_json::from_slice(&bytes).expect("JSON");
		assert_eq!(value["duration"], 1.25);
		assert_eq!(value["segments"][0]["text"], "hello");
		assert_eq!(value["words"][0]["word"], "hello");
	}

	#[test]
	fn speech_content_types_match_wire_encoding() {
		assert_eq!(speech_encoding("mp3"), Ok((AudioEncoding::Mp3, "audio/mpeg")));
		assert_eq!(speech_encoding("opus"), Ok((AudioEncoding::Opus, "audio/ogg")));
		assert_eq!(speech_encoding("wav"), Ok((AudioEncoding::Wav, "audio/wav")));
		assert_eq!(speech_encoding("pcm"), Ok((AudioEncoding::Pcm16, "audio/pcm")));
		assert_eq!(speech_encoding("aac"), Ok((AudioEncoding::Aac, "audio/aac")));
		assert_eq!(speech_encoding("flac"), Ok((AudioEncoding::Flac, "audio/flac")));
	}

	#[tokio::test]
	async fn speech_is_chunked_binary_with_requested_content_type() {
		let directory = tempfile::tempdir().expect("temporary directory");
		let request = Request::post("/v1/audio/speech")
			.body(Full::new(Bytes::from_static(
				br#"{"model":"tts","input":"hello","voice":"alloy","response_format":"mp3"}"#,
			)))
			.expect("request");
		let state = speech_state(directory.path());
		let response = handle(request, Arc::clone(&state)).await;
		assert_eq!(response.status(), StatusCode::OK);
		assert_eq!(response.headers()[header::CONTENT_TYPE], "audio/mpeg");
		assert_eq!(response.headers()[header::TRANSFER_ENCODING], "chunked");
		let body = response
			.into_body()
			.collect()
			.await
			.expect("infallible")
			.to_bytes();
		assert_eq!(body, Bytes::from_static(b"\x01\x02audio"));
		assert!(serde_json::from_slice::<serde_json::Value>(&body).is_err());
		assert!(
			state
				.blobs
				.has(&BlobRef { hash: *blake3::hash(&body).as_bytes(), size: body.len() as u64 })
		);
	}

	#[tokio::test]
	async fn multipart_transcription_collects_uploaded_file() {
		let boundary = "omp-audio-boundary";
		let body = format!(
			"--{boundary}\r\nContent-Disposition: form-data; \
			 name=\"model\"\r\n\r\nwhisper\r\n--{boundary}\r\nContent-Disposition: form-data; \
			 name=\"response_format\"\r\n\r\nverbose_json\r\n--{boundary}\r\nContent-Disposition: \
			 form-data; name=\"diarize\"\r\n\r\ntrue\r\n--{boundary}\r\nContent-Disposition: \
			 form-data; name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: \
			 audio/wav\r\n\r\nsamples\r\n--{boundary}--\r\n"
		);
		let request = Request::post("/v1/audio/transcriptions")
			.header(header::CONTENT_TYPE, format!("multipart/form-data; boundary={boundary}"))
			.body(http_body_util::Full::new(Bytes::from(body)))
			.expect("multipart request");
		let parsed = parse_multipart(request)
			.await
			.expect("valid multipart transcription");
		assert_eq!(parsed.model, "whisper");
		assert_eq!(parsed.response_format, "verbose_json");
		assert_eq!(parsed.audio.mime, "audio/wav");
		assert_eq!(parsed.audio.inline, Bytes::from_static(b"samples"));
		assert!(parsed.diarize);
	}

	#[tokio::test]
	async fn base64_json_transcription_and_translation_use_the_transcribe_facet() {
		let directory = tempfile::tempdir().expect("temporary directory");
		let transcription = Request::post("/v1/audio/transcriptions")
			.body(Full::new(Bytes::from_static(
				br#"{"model":"whisper","file":"c2FtcGxlcw==","mime_type":"audio/wav"}"#,
			)))
			.expect("transcription request");
		let state = speech_state(directory.path());
		let response = handle(transcription, Arc::clone(&state)).await;
		let uploaded = Bytes::from_static(b"samples");
		assert!(
			state.blobs.has(&BlobRef {
				hash: *blake3::hash(&uploaded).as_bytes(),
				size: uploaded.len() as u64,
			})
		);
		let bytes = response
			.into_body()
			.collect()
			.await
			.expect("infallible")
			.to_bytes();
		let value: serde_json::Value = serde_json::from_slice(&bytes).expect("JSON");
		assert_eq!(value["text"], "transcribed");

		let translation = Request::post("/v1/audio/translations")
			.body(Full::new(Bytes::from_static(
				br#"{"model":"whisper","file":"c2FtcGxlcw==","mime_type":"audio/wav"}"#,
			)))
			.expect("translation request");
		let response = handle(translation, speech_state(directory.path())).await;
		let bytes = response
			.into_body()
			.collect()
			.await
			.expect("infallible")
			.to_bytes();
		let value: serde_json::Value = serde_json::from_slice(&bytes).expect("JSON");
		assert_eq!(value["text"], "translated");
	}
}
