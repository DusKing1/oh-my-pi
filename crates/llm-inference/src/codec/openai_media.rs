//! Typed sans-I/O OpenAI-compatible image and transcription wire codec.

use std::{sync::Arc, time::Duration};

use bytes::Bytes;
use omp_core::{Str, encoding::base64};
use serde::{Deserialize, Serialize};

use crate::{
	answer::{
		Artifact, ArtifactBody, AudioChunk, GenerationEvent, GenerationSummary, ImageArtifact,
		Speaker, TranscriptEvent,
	},
	body::BodySource,
	call::{
		AudioFormat, Background, Dimensions, ImageFormat, ImageQuality, ImageRequest, MediaInput,
		OperationCall, Setting, SpeechRequest, TimestampGranularity, TranscriptionRequest,
	},
	catalog::OperationKind,
	codec::{
		Codec, DecodeContext, Decoder, DecoderState, EncodeContext, EncodedRequest, RawEvent,
		RequestHeader, RequestMethod, SizeBounds,
	},
	error::{Error, ErrorKind, ErrorPhase, RetryAction},
	receipt::{Cost, ExecutionReceipt, Usage},
	transport::{Frame, FramingProtocol},
};

/// Bounded paths and payload limits for OpenAI-compatible media endpoints.
#[derive(Clone, Debug)]
pub struct OpenAiMediaProfile {
	/// Image generation and edit path.
	pub images_path:              Str,
	/// Speech transcription path.
	pub transcription_path:       Str,
	/// Speech translation path.
	pub translation_path:         Str,
	/// Streamed speech synthesis path.
	pub speech_path:              Str,
	/// Maximum encoded request body bytes.
	pub max_request_bytes:        u64,
	/// Maximum framed response bytes.
	pub max_frame_bytes:          u64,
	/// Maximum aggregate response bytes.
	pub max_response_bytes:       u64,
	/// Provider default image dimensions used only when caller leaves dimensions
	/// unset.
	pub default_image_dimensions: Dimensions,
}

impl Default for OpenAiMediaProfile {
	fn default() -> Self {
		Self {
			images_path:              Str::from("/v1/images/generations"),
			speech_path:              Str::from("/v1/audio/speech"),
			transcription_path:       Str::from("/v1/audio/transcriptions"),
			translation_path:         Str::from("/v1/audio/translations"),
			max_request_bytes:        128 * 1024 * 1024,
			max_frame_bytes:          32 * 1024 * 1024,
			max_response_bytes:       128 * 1024 * 1024,
			default_image_dimensions: Dimensions { width: 1024, height: 1024 },
		}
	}
}

/// OpenAI-compatible media codec; networking, retry, polling, and storage
/// remain outside it.
#[derive(Clone, Debug, Default)]
pub struct OpenAiMediaCodec {
	/// Route-specific endpoint and bound profile.
	pub profile: OpenAiMediaProfile,
}

impl Codec for OpenAiMediaCodec {
	fn encode(
		&self,
		context: &EncodeContext<'_>,
		operation: &OperationCall,
	) -> Result<EncodedRequest, Error> {
		let target = context
			.target
			.ok_or_else(|| encoding_error(ErrorKind::InvalidRequest))?;
		match operation {
			OperationCall::GenerateImage(request) => {
				let (body, content_type, path) = encode_image(
					context.request_id.as_str(),
					target.wire_model.as_str(),
					request,
					&self.profile,
				)?;
				Ok(encoded(
					OperationKind::GenerateImage,
					target.endpoint.base_url.as_str(),
					path.as_str(),
					content_type,
					body,
					&self.profile,
				))
			},
			OperationCall::Speak(request) => {
				let body = encode_speech(target.wire_model.as_str(), request)?;
				Ok(encoded_with_framing(
					OperationKind::Speak,
					target.endpoint.base_url.as_str(),
					self.profile.speech_path.as_str(),
					Str::from("application/json"),
					BodySource::Bytes(body),
					FramingProtocol::RawChunks,
					&self.profile,
				))
			},
			OperationCall::Transcribe(request) => {
				let path = if request.translate_to_english {
					self.profile.translation_path.as_str()
				} else {
					self.profile.transcription_path.as_str()
				};
				let (body, content_type) = encode_transcription(
					context.request_id.as_str(),
					target.wire_model.as_str(),
					request,
				)?;
				Ok(encoded(
					OperationKind::Transcribe,
					target.endpoint.base_url.as_str(),
					path,
					content_type,
					body,
					&self.profile,
				))
			},
			_ => Err(capability_error()),
		}
	}

	fn decoder(&self, context: &DecodeContext<'_>) -> Result<DecoderState, Error> {
		if context.operation != context.operation_call.kind() {
			return Err(encoding_error(ErrorKind::InternalInvariant));
		}
		match context.operation_call {
			OperationCall::GenerateImage(request) => {
				let dimensions =
					setting(&request.dimensions).unwrap_or(self.profile.default_image_dimensions);
				let format = setting(&request.format).unwrap_or(ImageFormat::Png);
				Ok(Box::new(ImageDecoder { dimensions, format, count: request.count, finished: false }))
			},
			OperationCall::Speak(_) => Ok(Box::new(SpeechDecoder { pending: None, finished: false })),
			OperationCall::Transcribe(request) => Ok(Box::new(TranscriptionDecoder {
				timestamps: setting(&request.timestamps).unwrap_or(TimestampGranularity::None),
				finished:   false,
			})),
			_ => Err(capability_error()),
		}
	}
}

fn encoded(
	operation: OperationKind,
	base: &str,
	path: &str,
	content_type: Str,
	body: BodySource,
	profile: &OpenAiMediaProfile,
) -> EncodedRequest {
	EncodedRequest::new(
		operation,
		RequestMethod::Post,
		Str::from(join_uri(base, path)),
		vec![RequestHeader { name: Str::from("content-type"), value: content_type }]
			.into_boxed_slice(),
		body,
		FramingProtocol::Raw,
		SizeBounds {
			request_body: profile.max_request_bytes,
			frame:        profile.max_frame_bytes,
			response:     profile.max_response_bytes,
		},
	)
}

fn encoded_with_framing(
	operation: OperationKind,
	base: &str,
	path: &str,
	content_type: Str,
	body: BodySource,
	framing: FramingProtocol,
	profile: &OpenAiMediaProfile,
) -> EncodedRequest {
	let mut request = encoded(operation, base, path, content_type, body, profile);
	request.framing = framing;
	request
}

fn encode_image(
	request_id: &str,
	model: &str,
	request: &ImageRequest,
	profile: &OpenAiMediaProfile,
) -> Result<(BodySource, Str, Str), Error> {
	if !request.safety.is_empty() || request.seed.is_some() {
		return Err(capability_error());
	}
	if request.references.is_empty() && request.mask.is_none() {
		let dimensions = setting(&request.dimensions);
		let wire = ImageJsonRequest {
			model,
			prompt: request.prompt.as_str(),
			n: request.count,
			size: dimensions.map(size_string),
			quality: setting(&request.quality).map(quality_string),
			background: setting(&request.background).map(background_string),
			output_format: setting(&request.format).map(format_string),
			style: setting_ref(&request.style).map(Str::as_str),
		};
		let body = serde_json::to_vec(&wire)
			.map(Bytes::from)
			.map_err(|_| encoding_error(ErrorKind::InvalidRequest))?;
		return Ok((
			BodySource::Bytes(body),
			Str::from("application/json"),
			profile.images_path.clone(),
		));
	}
	let boundary = boundary(request_id);
	let mut parts = Vec::new();
	push_text(&mut parts, &boundary, "model", model);
	push_text(&mut parts, &boundary, "prompt", request.prompt.as_str());
	push_text(&mut parts, &boundary, "n", &request.count.to_string());
	if let Some(value) = setting(&request.dimensions) {
		push_text(&mut parts, &boundary, "size", &size_string(value));
	}
	if let Some(value) = setting(&request.quality) {
		push_text(&mut parts, &boundary, "quality", quality_string(value));
	}
	if let Some(value) = setting(&request.background) {
		push_text(&mut parts, &boundary, "background", background_string(value));
	}
	if let Some(value) = setting(&request.format) {
		push_text(&mut parts, &boundary, "output_format", format_string(value));
	}
	if let Some(value) = setting_ref(&request.style) {
		push_text(&mut parts, &boundary, "style", value);
	}
	for (index, image) in request.references.iter().enumerate() {
		push_media(&mut parts, &boundary, "image[]", &format!("image-{index}"), image)?;
	}
	if let Some(mask) = &request.mask {
		push_media(&mut parts, &boundary, "mask", "mask", mask)?;
	}
	parts.push(BodySource::Bytes(Bytes::from(format!("--{boundary}--\r\n"))));
	Ok((
		BodySource::multipart(Arc::<[BodySource]>::from(parts)),
		Str::from(format!("multipart/form-data; boundary={boundary}")),
		Str::from("/v1/images/edits"),
	))
}

fn encode_speech(model: &str, request: &SpeechRequest) -> Result<Bytes, Error> {
	if !matches!(&request.sample_rate_hz, Setting::Unset) {
		return Err(capability_error());
	}
	if setting(&request.timestamps).is_some_and(|value| value != TimestampGranularity::None) {
		return Err(capability_error());
	}
	let format = match setting(&request.format).unwrap_or(AudioFormat::Mp3) {
		AudioFormat::Pcm16 => "pcm",
		AudioFormat::Mp3 => "mp3",
		AudioFormat::Aac => "aac",
		AudioFormat::Opus => "opus",
		AudioFormat::Flac => "flac",
		AudioFormat::Wav => "wav",
		AudioFormat::Pcm24 | AudioFormat::F32 => return Err(capability_error()),
	};
	let wire = SpeechWireRequest {
		model,
		input: request.text.as_str(),
		voice: request.voice.as_str(),
		response_format: format,
		speed: setting(&request.speed),
	};
	serde_json::to_vec(&wire)
		.map(Bytes::from)
		.map_err(|_| encoding_error(ErrorKind::InvalidRequest))
}

#[derive(Serialize)]
struct SpeechWireRequest<'a> {
	model:           &'a str,
	input:           &'a str,
	voice:           &'a str,
	response_format: &'static str,
	#[serde(skip_serializing_if = "Option::is_none")]
	speed:           Option<f32>,
}

fn encode_transcription(
	request_id: &str,
	model: &str,
	request: &TranscriptionRequest,
) -> Result<(BodySource, Str), Error> {
	let boundary = boundary(request_id);
	let mut parts = Vec::new();
	push_text(&mut parts, &boundary, "model", model);
	if let Some(language) = &request.language {
		push_text(&mut parts, &boundary, "language", language);
	}
	if let Some(prompt) = &request.prompt {
		push_text(&mut parts, &boundary, "prompt", prompt);
	}
	let diarization = matches!(&request.diarization, Setting::Require(true) | Setting::Prefer(true));
	if request.translate_to_english && diarization {
		return Err(capability_error());
	}
	let timestamps = setting(&request.timestamps).unwrap_or(TimestampGranularity::None);
	let response_format = if diarization {
		"diarized_json"
	} else if timestamps == TimestampGranularity::None {
		"json"
	} else {
		"verbose_json"
	};
	push_text(&mut parts, &boundary, "response_format", response_format);
	if timestamps == TimestampGranularity::Word {
		push_text(&mut parts, &boundary, "timestamp_granularities[]", "word");
	}
	if timestamps != TimestampGranularity::None {
		push_text(&mut parts, &boundary, "timestamp_granularities[]", "segment");
	}
	push_media(&mut parts, &boundary, "file", "audio", &request.audio)?;
	parts.push(BodySource::Bytes(Bytes::from(format!("--{boundary}--\r\n"))));
	Ok((
		BodySource::multipart(Arc::<[BodySource]>::from(parts)),
		Str::from(format!("multipart/form-data; boundary={boundary}")),
	))
}

#[derive(Serialize)]
struct ImageJsonRequest<'a> {
	model:         &'a str,
	prompt:        &'a str,
	n:             u32,
	#[serde(skip_serializing_if = "Option::is_none")]
	size:          Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	quality:       Option<&'static str>,
	#[serde(skip_serializing_if = "Option::is_none")]
	background:    Option<&'static str>,
	#[serde(skip_serializing_if = "Option::is_none")]
	output_format: Option<&'static str>,
	#[serde(skip_serializing_if = "Option::is_none")]
	style:         Option<&'a str>,
}

fn push_text(parts: &mut Vec<BodySource>, boundary: &str, name: &str, value: &str) {
	parts.push(BodySource::Bytes(Bytes::from(format!(
		"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
	))));
}

fn push_media(
	parts: &mut Vec<BodySource>,
	boundary: &str,
	field: &str,
	filename: &str,
	media: &MediaInput,
) -> Result<(), Error> {
	let (media_type, body) = match media {
		MediaInput::Bytes { media_type, data } => (media_type, BodySource::Bytes(data.clone())),
		MediaInput::Body { media_type, body, .. } => (media_type, body.clone()),
		MediaInput::Stored(_) | MediaInput::Remote { .. } => {
			return Err(encoding_error(ErrorKind::StagingRequired));
		},
	};
	parts.push(BodySource::Bytes(Bytes::from(format!(
		"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; \
		 filename=\"{filename}\"\r\nContent-Type: {media_type}\r\n\r\n"
	))));
	parts.push(body);
	parts.push(BodySource::Bytes(Bytes::from_static(b"\r\n")));
	Ok(())
}

struct ImageDecoder {
	dimensions: Dimensions,
	format:     ImageFormat,
	count:      u32,
	finished:   bool,
}

impl Decoder for ImageDecoder {
	fn push(&mut self, frame: Frame, emit: &mut dyn FnMut(RawEvent)) -> Result<(), Error> {
		if self.finished {
			return Err(protocol_error(true));
		}
		let Frame::Raw(bytes) = frame else {
			return Err(protocol_error(false));
		};
		let response: ImageResponse =
			serde_json::from_slice(&bytes).map_err(|_| protocol_error(false))?;
		if response.data.len() as u32 != self.count {
			return Err(protocol_error(false));
		}
		for item in response.data {
			let encoded = item.b64_json.ok_or_else(|| protocol_error(false))?;
			let bytes = base64::decode(encoded.as_bytes())
				.into_vec()
				.map(Bytes::from)
				.map_err(|_| protocol_error(false))?;
			let media_type = match self.format {
				ImageFormat::Png => "image/png",
				ImageFormat::Jpeg => "image/jpeg",
				ImageFormat::Webp => "image/webp",
			};
			emit(RawEvent::ImageGeneration(GenerationEvent::Artifact(ImageArtifact {
				artifact:       Artifact {
					media_type: Str::from(media_type),
					size:       Some(bytes.len() as u64),
					digest:     None,
					body:       ArtifactBody::Bytes(bytes),
				},
				width:          self.dimensions.width,
				height:         self.dimensions.height,
				revised_prompt: item.revised_prompt,
			})));
		}
		emit(RawEvent::ImageGeneration(GenerationEvent::Completed(GenerationSummary {
			artifacts: self.count,
			elapsed:   Duration::ZERO,
			usage:     Usage::default(),
			cost:      Cost::default(),
		})));
		self.finished = true;
		Ok(())
	}

	fn finish(&mut self, _emit: &mut dyn FnMut(RawEvent)) -> Result<(), Error> {
		if self.finished {
			Ok(())
		} else {
			Err(protocol_error(false))
		}
	}
}

#[derive(Deserialize)]
struct ImageResponse {
	data: Vec<ImageResponseItem>,
}
#[derive(Deserialize)]
struct ImageResponseItem {
	#[serde(default)]
	b64_json:       Option<Str>,
	#[serde(default)]
	revised_prompt: Option<Str>,
}

struct SpeechDecoder {
	pending:  Option<Bytes>,
	finished: bool,
}

impl Decoder for SpeechDecoder {
	fn push(&mut self, frame: Frame, emit: &mut dyn FnMut(RawEvent)) -> Result<(), Error> {
		if self.finished {
			return Err(protocol_error(true));
		}
		let Frame::Raw(bytes) = frame else {
			return Err(protocol_error(false));
		};
		if bytes.is_empty() {
			return Ok(());
		}
		if let Some(previous) = self.pending.replace(bytes) {
			emit(RawEvent::Audio(AudioChunk {
				bytes:       previous,
				start_ms:    None,
				end_ms:      None,
				final_chunk: false,
			}));
		}
		Ok(())
	}

	fn finish(&mut self, emit: &mut dyn FnMut(RawEvent)) -> Result<(), Error> {
		if self.finished {
			return Err(protocol_error(true));
		}
		let bytes = self.pending.take().ok_or_else(|| protocol_error(false))?;
		emit(RawEvent::Audio(AudioChunk { bytes, start_ms: None, end_ms: None, final_chunk: true }));
		self.finished = true;
		Ok(())
	}
}

struct TranscriptionDecoder {
	timestamps: TimestampGranularity,
	finished:   bool,
}

impl Decoder for TranscriptionDecoder {
	fn push(&mut self, frame: Frame, emit: &mut dyn FnMut(RawEvent)) -> Result<(), Error> {
		if self.finished {
			return Err(protocol_error(true));
		}
		let Frame::Raw(bytes) = frame else {
			return Err(protocol_error(false));
		};
		let response: TranscriptionResponse =
			serde_json::from_slice(&bytes).map_err(|_| protocol_error(false))?;
		emit(RawEvent::Transcript(TranscriptEvent::Started { language: response.language.clone() }));
		if self.timestamps != TimestampGranularity::None {
			for segment in response.segments {
				emit(RawEvent::Transcript(TranscriptEvent::Segment {
					index:    segment.id,
					text:     segment.text,
					start_ms: seconds_ms(segment.start)?,
					end_ms:   seconds_ms(segment.end)?,
					speaker:  segment
						.speaker
						.map(|label| Speaker { index: 0, label: Some(label) }),
				}));
			}
		}
		if self.timestamps == TimestampGranularity::Word {
			for word in response.words {
				emit(RawEvent::Transcript(TranscriptEvent::Word {
					text:       word.word,
					start_ms:   seconds_ms(word.start)?,
					end_ms:     seconds_ms(word.end)?,
					confidence: word.confidence,
					speaker:    word
						.speaker
						.map(|label| Speaker { index: 0, label: Some(label) }),
				}));
			}
		}
		emit(RawEvent::Transcript(TranscriptEvent::Completed {
			text:  response.text,
			usage: Usage::default(),
		}));
		self.finished = true;
		Ok(())
	}

	fn finish(&mut self, _emit: &mut dyn FnMut(RawEvent)) -> Result<(), Error> {
		if self.finished {
			Ok(())
		} else {
			Err(protocol_error(false))
		}
	}
}

#[derive(Deserialize)]
struct TranscriptionResponse {
	text:     Str,
	#[serde(default)]
	language: Option<Str>,
	#[serde(default)]
	segments: Vec<TranscriptionSegment>,
	#[serde(default)]
	words:    Vec<TranscriptionWord>,
}
#[derive(Deserialize)]
struct TranscriptionSegment {
	id:      u32,
	start:   f64,
	end:     f64,
	text:    Str,
	#[serde(default)]
	speaker: Option<Str>,
}
#[derive(Deserialize)]
struct TranscriptionWord {
	word:       Str,
	start:      f64,
	end:        f64,
	#[serde(default)]
	confidence: Option<f32>,
	#[serde(default)]
	speaker:    Option<Str>,
}

fn seconds_ms(seconds: f64) -> Result<u64, Error> {
	if !seconds.is_finite() || seconds < 0.0 || seconds > u64::MAX as f64 / 1000.0 {
		return Err(protocol_error(false));
	}
	Ok((seconds * 1000.0).round() as u64)
}

fn setting<T: Copy>(setting: &Setting<T>) -> Option<T> {
	match setting {
		Setting::Unset => None,
		Setting::Require(value) | Setting::Prefer(value) => Some(*value),
	}
}
fn setting_ref<T>(setting: &Setting<T>) -> Option<&T> {
	match setting {
		Setting::Unset => None,
		Setting::Require(value) | Setting::Prefer(value) => Some(value),
	}
}
fn size_string(value: Dimensions) -> String {
	format!("{}x{}", value.width, value.height)
}
const fn quality_string(value: ImageQuality) -> &'static str {
	match value {
		ImageQuality::Draft => "low",
		ImageQuality::Standard => "medium",
		ImageQuality::High => "high",
	}
}
const fn background_string(value: Background) -> &'static str {
	match value {
		Background::Opaque => "opaque",
		Background::Transparent => "transparent",
		Background::Auto => "auto",
	}
}
const fn format_string(value: ImageFormat) -> &'static str {
	match value {
		ImageFormat::Png => "png",
		ImageFormat::Jpeg => "jpeg",
		ImageFormat::Webp => "webp",
	}
}

fn boundary(request_id: &str) -> String {
	let mut value = String::from("omp-");
	for character in request_id
		.chars()
		.filter(|character| character.is_ascii_alphanumeric())
		.take(48)
	{
		value.push(character);
	}
	if value.len() == 4 {
		value.push_str("media");
	}
	value
}
fn join_uri(base: &str, path: &str) -> String {
	let mut uri = String::with_capacity(base.len() + path.len() + 1);
	uri.push_str(base.trim_end_matches('/'));
	if !path.starts_with('/') {
		uri.push('/');
	}
	uri.push_str(path);
	uri
}
fn capability_error() -> Error {
	Error::new(
		ErrorKind::CapabilityMismatch,
		ErrorPhase::Encoding,
		RetryAction::Never,
		ExecutionReceipt::default(),
	)
}
fn encoding_error(kind: ErrorKind) -> Error {
	Error::new(kind, ErrorPhase::Encoding, RetryAction::Never, ExecutionReceipt::default())
}
fn protocol_error(committed: bool) -> Error {
	let mut error = Error::new(
		ErrorKind::Protocol,
		if committed {
			ErrorPhase::Streaming
		} else {
			ErrorPhase::Handshake
		},
		RetryAction::Never,
		ExecutionReceipt::default(),
	);
	error.committed = committed;
	error
}

#[cfg(test)]
mod tests {
	use std::sync::Arc;

	use super::*;
	use crate::{
		body::{OneShotBody, Replayability, byte_stream},
		call::{ImageRequest, NegotiationPolicy},
	};

	#[test]
	fn image_edit_multipart_preserves_consumable_one_shot_evidence() {
		let body =
			BodySource::OneShot(Arc::new(OneShotBody::new(byte_stream(Bytes::from_static(b"image")))));
		let request = ImageRequest {
			prompt:      Str::from("edit"),
			references:  Arc::from([MediaInput::Body {
				media_type: Str::from("image/png"),
				body,
				name: Some(Str::from("input.png")),
			}]),
			mask:        None,
			count:       1,
			dimensions:  Setting::Unset,
			quality:     Setting::Unset,
			background:  Setting::Unset,
			format:      Setting::Require(ImageFormat::Png),
			style:       Setting::Unset,
			safety:      Arc::from([]),
			seed:        None,
			negotiation: NegotiationPolicy::default(),
		};
		let (body, content_type, path) =
			encode_image("request", "image-model", &request, &OpenAiMediaProfile::default())
				.expect("valid multipart edit");
		assert_eq!(body.replayability(), Replayability::OneShot);
		assert!(content_type.starts_with("multipart/form-data; boundary="));
		assert_eq!(path.as_str(), "/v1/images/edits");
	}
}
