//! OpenAI-compatible speech synthesis and recorded-audio transcription codec.
//!
//! The codec owns only wire encoding and response decoding. Authentication,
//! endpoint selection, retries, and response-body streaming remain transport
//! concerns so credential bytes never enter codec state.

use std::collections::BTreeMap;

use bytes::{Bytes, BytesMut};
use omp_core::SmolStr;
use omp_llm_types::{
	Accuracy, AudioEncoding, BlobPart, Props, SpeakRequest, TranscribeRequest, TranscribeResponse,
	TranscriptSegment, TranscriptWord, TranscriptionGranularity, Usage,
};
use serde_json::{Value, json};

/// One buffered provider request produced by the audio codec.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodedAudioRequest {
	/// Provider endpoint suffix relative to the catalog base URL.
	pub path:         &'static str,
	/// Request `Content-Type`, including the multipart boundary when applicable.
	pub content_type: SmolStr,
	/// Requested response media type.
	pub accept:       &'static str,
	/// Fully encoded request body.
	pub body:         Bytes,
}

/// Protocol-level OpenAI audio failure.
#[derive(Clone, Debug, thiserror::Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum OpenAiAudioError {
	/// A required request value was absent or outside provider limits.
	#[error("invalid audio request: {0}")]
	Invalid(SmolStr),
	/// A canonical control has no faithful OpenAI wire representation.
	#[error("unsupported audio control: {0}")]
	Unsupported(SmolStr),
	/// The provider response was not valid for the selected operation.
	#[error("invalid audio response: {0}")]
	Decode(SmolStr),
}

/// Data-selected request differences within OpenAI-compatible audio APIs.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum OpenAiAudioProfile {
	/// Standard OpenAI `/v1/audio` request bodies.
	#[default]
	Standard,
	/// A deployment-scoped endpoint where the model lives only in the URL.
	DeploymentPath,
}

/// Stateless codec for OpenAI-compatible `/audio` endpoints.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenAiAudioCodec;

impl OpenAiAudioCodec {
	/// Encodes a standard OpenAI speech request.
	pub fn encode_speech(
		&self,
		request: &SpeakRequest,
	) -> Result<EncodedAudioRequest, OpenAiAudioError> {
		self.encode_speech_for(request, OpenAiAudioProfile::Standard)
	}

	/// Encodes speech using the catalog-selected endpoint profile.
	pub fn encode_speech_for(
		&self,
		request: &SpeakRequest,
		profile: OpenAiAudioProfile,
	) -> Result<EncodedAudioRequest, OpenAiAudioError> {
		if request.model.trim().is_empty() {
			return Err(invalid("model is required"));
		}
		if request.text.is_empty() {
			return Err(invalid("speech input is required"));
		}
		if request.text.chars().count() > 4096 {
			return Err(invalid("speech input exceeds 4096 characters"));
		}
		if request.voice.trim().is_empty() {
			return Err(invalid("voice is required"));
		}
		if request.sample_rate_hz.is_some() {
			return Err(unsupported("sample_rate_hz is not accepted by OpenAI speech"));
		}
		if request.clone.is_some() {
			return Err(unsupported("voice cloning is not accepted by OpenAI speech"));
		}
		if profile == OpenAiAudioProfile::DeploymentPath && !request.instructions.is_empty() {
			return Err(unsupported(
				"instructions are not accepted by the deployment-scoped speech API",
			));
		}
		if let Some(speed) = request.speed
			&& (!speed.is_finite() || !(0.25..=4.0).contains(&speed))
		{
			return Err(invalid("speed must be between 0.25 and 4.0"));
		}
		let (encoding, accept) = encoding_wire(request.encoding)?;

		let mut body = json!({
			"input": request.text,
			"voice": request.voice,
			"response_format": encoding,
		});
		if profile == OpenAiAudioProfile::Standard {
			body["model"] = request.model.as_str().into();
		}
		if let Some(speed) = request.speed {
			body["speed"] = speed.into();
		}
		if !request.instructions.is_empty() {
			body["instructions"] = request.instructions.as_str().into();
		}
		let body = serde_json::to_vec(&body)
			.map(Bytes::from)
			.map_err(|error| OpenAiAudioError::Invalid(error.to_string().into()))?;
		Ok(EncodedAudioRequest {
			path: "/audio/speech",
			content_type: "application/json".into(),
			accept,
			body,
		})
	}

	/// Encodes an inline recording as OpenAI multipart form data.
	pub fn encode_transcription(
		&self,
		request: &TranscribeRequest,
	) -> Result<EncodedAudioRequest, OpenAiAudioError> {
		validate_transcription(request)?;
		let extension = media_extension(&request.audio)?;
		let boundary = multipart_boundary(&request.audio.hash);
		let mut body = BytesMut::with_capacity(request.audio.inline.len().saturating_add(768));
		field(&mut body, &boundary, "model", request.model.as_bytes());
		file_field(
			&mut body,
			&boundary,
			"file",
			&format!("audio.{extension}"),
			media_mime(extension),
			&request.audio.inline,
		);
		if !request.prompt.is_empty() {
			field(&mut body, &boundary, "prompt", request.prompt.as_bytes());
		}
		if !request.language.is_empty() {
			field(&mut body, &boundary, "language", request.language.as_bytes());
		}
		if let Some(temperature) = request.temperature {
			field(&mut body, &boundary, "temperature", temperature.to_string().as_bytes());
		}
		let requested_format = request
			.props
			.get_ns("openai", "response_format")
			.and_then(Value::as_str);
		let response_format = if request.diarize {
			"diarized_json"
		} else if let Some(format @ ("json" | "verbose_json")) = requested_format {
			format
		} else if requested_format.is_some() {
			return Err(unsupported("unknown transcription response format"));
		} else if request.granularities.is_empty() {
			"json"
		} else {
			"verbose_json"
		};
		field(&mut body, &boundary, "response_format", response_format.as_bytes());
		if !request.diarize {
			for granularity in &request.granularities {
				field(&mut body, &boundary, "timestamp_granularities[]", match granularity {
					TranscriptionGranularity::Segment => b"segment",
					TranscriptionGranularity::Word => b"word",
					_ => return Err(unsupported("unknown timestamp granularity")),
				});
			}
		}
		if request.diarize {
			field(&mut body, &boundary, "chunking_strategy", b"auto");
		}
		body.extend_from_slice(b"--");
		body.extend_from_slice(boundary.as_bytes());
		body.extend_from_slice(b"--\r\n");
		Ok(EncodedAudioRequest {
			path:         if request.translate {
				"/audio/translations"
			} else {
				"/audio/transcriptions"
			},
			content_type: format!("multipart/form-data; boundary={boundary}").into(),
			accept:       "application/json",
			body:         body.freeze(),
		})
	}

	/// Decodes JSON, verbose JSON, or diarized JSON transcription output.
	pub fn decode_transcription(&self, body: &[u8]) -> Result<TranscribeResponse, OpenAiAudioError> {
		let value: Value = serde_json::from_slice(body)
			.map_err(|error| OpenAiAudioError::Decode(error.to_string().into()))?;
		let text = required_str(&value, "text")?;
		let language = value
			.get("language")
			.and_then(Value::as_str)
			.or_else(|| value.pointer("/languages/0/code").and_then(Value::as_str))
			.unwrap_or_default();
		let duration_ms = seconds_ms(value.get("duration"))
			.or_else(|| seconds_ms(value.pointer("/usage/seconds")))
			.unwrap_or(0);
		let mut speakers = BTreeMap::<String, u32>::new();
		let segments = value
			.get("segments")
			.and_then(Value::as_array)
			.into_iter()
			.flatten()
			.map(|segment| decode_segment(segment, &mut speakers))
			.collect::<Result<Vec<_>, _>>()?;
		let words = value
			.get("words")
			.and_then(Value::as_array)
			.into_iter()
			.flatten()
			.map(|word| decode_word(word, &mut speakers))
			.collect::<Result<Vec<_>, _>>()?;
		Ok(TranscribeResponse::builder()
			.text(text.into())
			.language(language.into())
			.duration_ms(duration_ms)
			.segments(segments)
			.words(words)
			.maybe_usage(decode_usage(value.get("usage")))
			.maybe_cost(None)
			.unsupported(Vec::new())
			.props(Props::default())
			.build())
	}
}

fn validate_transcription(request: &TranscribeRequest) -> Result<(), OpenAiAudioError> {
	if request.model.trim().is_empty() {
		return Err(invalid("model is required"));
	}
	if request.audio.inline.is_empty() {
		return Err(invalid("recording bytes are required"));
	}
	if request.audio.size != request.audio.inline.len() as u64 {
		return Err(invalid("recording size does not match inline bytes"));
	}
	if *blake3::hash(&request.audio.inline).as_bytes() != request.audio.hash {
		return Err(invalid("recording hash does not match inline bytes"));
	}
	if request.audio.size > 25 * 1024 * 1024 {
		return Err(invalid("recording exceeds the 25 MiB provider limit"));
	}
	if let Some(temperature) = request.temperature
		&& (!temperature.is_finite() || !(0.0..=1.0).contains(&temperature))
	{
		return Err(invalid("temperature must be between 0 and 1"));
	}
	if request.diarize
		&& request
			.granularities
			.contains(&TranscriptionGranularity::Word)
	{
		return Err(unsupported("diarized transcription does not expose word timestamps"));
	}
	if request.translate
		&& (!request.language.is_empty() || !request.granularities.is_empty() || request.diarize)
	{
		return Err(unsupported("translations do not accept language, timestamps, or diarization"));
	}
	Ok(())
}

fn media_extension(audio: &BlobPart) -> Result<&'static str, OpenAiAudioError> {
	let extension = match audio
		.mime
		.as_str()
		.split(';')
		.next()
		.unwrap_or_default()
		.trim()
	{
		"audio/mpeg" | "audio/mp3" => "mp3",
		"audio/mp4" | "video/mp4" | "audio/x-m4a" => "m4a",
		"audio/wav" | "audio/x-wav" => "wav",
		"audio/webm" | "video/webm" => "webm",
		"audio/ogg" | "application/ogg" => "ogg",
		"audio/flac" | "audio/x-flac" => "flac",
		"application/octet-stream" => {
			return sniff_extension(&audio.inline)
				.ok_or_else(|| invalid("recording media type could not be determined"));
		},
		_ => return Err(invalid("unsupported recording media type")),
	};
	if sniff_extension(&audio.inline) != Some(extension) {
		return Err(invalid("recording bytes do not match the declared media type"));
	}
	Ok(extension)
}

fn media_mime(extension: &str) -> &'static str {
	match extension {
		"mp3" => "audio/mpeg",
		"m4a" => "audio/mp4",
		"wav" => "audio/wav",
		"webm" => "audio/webm",
		"ogg" => "audio/ogg",
		"flac" => "audio/flac",
		_ => "application/octet-stream",
	}
}

fn sniff_extension(bytes: &[u8]) -> Option<&'static str> {
	if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WAVE") {
		Some("wav")
	} else if bytes.starts_with(b"ID3") || bytes.first().is_some_and(|byte| byte & 0xe0 == 0xe0) {
		Some("mp3")
	} else if bytes.starts_with(b"OggS") {
		Some("ogg")
	} else if bytes.starts_with(b"fLaC") {
		Some("flac")
	} else if bytes.starts_with(&[0x1a, 0x45, 0xdf, 0xa3]) {
		Some("webm")
	} else if bytes.get(4..8) == Some(b"ftyp") {
		Some("m4a")
	} else {
		None
	}
}

fn decode_segment(
	value: &Value,
	speakers: &mut BTreeMap<String, u32>,
) -> Result<TranscriptSegment, OpenAiAudioError> {
	Ok(TranscriptSegment::builder()
		.start_ms(required_seconds_ms(value, "start")?)
		.end_ms(required_seconds_ms(value, "end")?)
		.text(required_str(value, "text")?.into())
		.maybe_speaker(decode_speaker(value.get("speaker"), speakers))
		.maybe_confidence(value.get("confidence").and_then(Value::as_f64))
		.build())
}

fn decode_word(
	value: &Value,
	speakers: &mut BTreeMap<String, u32>,
) -> Result<TranscriptWord, OpenAiAudioError> {
	Ok(TranscriptWord::builder()
		.start_ms(required_seconds_ms(value, "start")?)
		.end_ms(required_seconds_ms(value, "end")?)
		.word(required_str(value, "word")?.into())
		.maybe_speaker(decode_speaker(value.get("speaker"), speakers))
		.build())
}

fn decode_speaker(value: Option<&Value>, speakers: &mut BTreeMap<String, u32>) -> Option<u32> {
	let value = value?;
	if let Some(number) = value.as_u64().and_then(|value| u32::try_from(value).ok()) {
		return Some(number);
	}
	let label = value.as_str()?;
	if let Ok(number) = label.parse::<u32>() {
		return Some(number);
	}
	if let Some(number) = speakers.get(label) {
		return Some(*number);
	}
	let number = u32::try_from(speakers.len()).ok()?;
	speakers.insert(label.to_owned(), number);
	Some(number)
}

fn decode_usage(value: Option<&Value>) -> Option<Usage> {
	let value = value?;
	let input = value
		.get("input_tokens")
		.and_then(Value::as_u64)
		.unwrap_or(0);
	let output = value
		.get("output_tokens")
		.and_then(Value::as_u64)
		.unwrap_or(0);
	let seconds = value.get("seconds").and_then(Value::as_f64);
	if input == 0 && output == 0 && seconds.is_none() {
		return None;
	}
	let mut detail = Props::default();
	if let Some(seconds) = seconds {
		detail.insert_ns("openai", "audio_seconds", seconds.into());
	}
	if let Some(details) = value.get("input_token_details") {
		if let Some(tokens) = details.get("audio_tokens").and_then(Value::as_u64) {
			detail.insert_ns("openai", "input_audio_tokens", tokens.into());
		}
		if let Some(tokens) = details.get("text_tokens").and_then(Value::as_u64) {
			detail.insert_ns("openai", "input_text_tokens", tokens.into());
		}
	}
	Some(
		Usage::builder()
			.input_tokens(input)
			.output_tokens(output)
			.cache_read_tokens(0)
			.cache_write_tokens(0)
			.accuracy(Accuracy::Exact)
			.detail(detail)
			.build(),
	)
}

fn required_str<'a>(value: &'a Value, name: &str) -> Result<&'a str, OpenAiAudioError> {
	value
		.get(name)
		.and_then(Value::as_str)
		.ok_or_else(|| OpenAiAudioError::Decode(format!("missing {name}").into()))
}

fn required_seconds_ms(value: &Value, name: &str) -> Result<u64, OpenAiAudioError> {
	seconds_ms(value.get(name))
		.ok_or_else(|| OpenAiAudioError::Decode(format!("invalid {name} timestamp").into()))
}

fn seconds_ms(value: Option<&Value>) -> Option<u64> {
	let seconds = value?.as_f64()?;
	if !seconds.is_finite() || seconds < 0.0 || seconds > u64::MAX as f64 / 1000.0 {
		return None;
	}
	Some((seconds * 1000.0).round() as u64)
}

fn encoding_wire(
	encoding: AudioEncoding,
) -> Result<(&'static str, &'static str), OpenAiAudioError> {
	match encoding {
		AudioEncoding::Mp3 => Ok(("mp3", "audio/mpeg")),
		AudioEncoding::Pcm16 => Ok(("pcm", "audio/pcm")),
		AudioEncoding::Wav => Ok(("wav", "audio/wav")),
		AudioEncoding::Opus => Ok(("opus", "audio/ogg")),
		AudioEncoding::Aac => Ok(("aac", "audio/aac")),
		AudioEncoding::Flac => Ok(("flac", "audio/flac")),
		_ => Err(unsupported("unknown speech encoding")),
	}
}

fn multipart_boundary(hash: &[u8; 32]) -> String {
	const HEX: &[u8; 16] = b"0123456789abcdef";
	let mut boundary = String::with_capacity(36);
	boundary.push_str("omp-audio-");
	for byte in &hash[..12] {
		boundary.push(HEX[(byte >> 4) as usize] as char);
		boundary.push(HEX[(byte & 0x0f) as usize] as char);
	}
	boundary
}

fn field(body: &mut BytesMut, boundary: &str, name: &str, value: &[u8]) {
	body.extend_from_slice(b"--");
	body.extend_from_slice(boundary.as_bytes());
	body.extend_from_slice(b"\r\nContent-Disposition: form-data; name=\"");
	body.extend_from_slice(name.as_bytes());
	body.extend_from_slice(b"\"\r\n\r\n");
	body.extend_from_slice(value);
	body.extend_from_slice(b"\r\n");
}

fn file_field(
	body: &mut BytesMut,
	boundary: &str,
	name: &str,
	filename: &str,
	mime: &str,
	value: &[u8],
) {
	body.extend_from_slice(b"--");
	body.extend_from_slice(boundary.as_bytes());
	body.extend_from_slice(b"\r\nContent-Disposition: form-data; name=\"");
	body.extend_from_slice(name.as_bytes());
	body.extend_from_slice(b"\"; filename=\"");
	body.extend_from_slice(filename.as_bytes());
	body.extend_from_slice(b"\"\r\nContent-Type: ");
	body.extend_from_slice(mime.as_bytes());
	body.extend_from_slice(b"\r\n\r\n");
	body.extend_from_slice(value);
	body.extend_from_slice(b"\r\n");
}

fn invalid(message: &'static str) -> OpenAiAudioError {
	OpenAiAudioError::Invalid(message.into())
}

fn unsupported(message: &'static str) -> OpenAiAudioError {
	OpenAiAudioError::Unsupported(message.into())
}
