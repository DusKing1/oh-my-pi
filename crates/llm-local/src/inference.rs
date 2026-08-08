//! Canonical in-process inference over local model runtimes.
//!
//! [`Inference`] owns independently configured text, embedding, speech, and
//! transcription engines. It implements
//! [`LocalEngine`](omp_llm_transport::embedded::LocalEngine), adapting only at
//! the engine-native boundaries (`ChatMessage`, `GenerationOptions`, `Audio`,
//! and the corresponding embedding, synthesis, and transcription options).
//! Canonical requests and events otherwise flow unchanged through the embedded
//! transport and catalog router.

use bytes::Bytes;
use futures::{StreamExt, stream::BoxStream};
use omp_core::{SmolStr, format_smol};
use omp_llm_transport::embedded::LocalEngine;
use omp_llm_types as native;
use tokio_util::sync::CancellationToken;

use crate::{
	AppleFm, AppleFmError, AppleFmErrorCode, AppleFmEvent, AppleFmOptions, AppleFmStream, Audio,
	ChatMessage, ChatRole, DevicePreference, Embedder, EmbedderBuilder, EmbeddingModel,
	EmbeddingOptions, Error, GenerationOptions, GenerationStop, GenerationStream, Hub, Kokoro,
	KokoroVoice, Parakeet, Result, SmallModel, SttModel, SynthesisOptions, TextGenerator,
	TextGeneratorBuilder, TextModel, TranscriptionOptions, Whisper, WhisperBuilder,
};

/// Byte size of one [`native::SpeakEvent::Chunk`] payload.
const SPEAK_CHUNK_BYTES: usize = 16 * 1024;

/// Text backend selection for [`InferenceBuilder::text`].
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum TextSelection {
	/// Apple Foundation Models when available, otherwise the default curated
	/// small GGUF model.
	#[default]
	Auto,
	/// Require Apple's on-device Foundation Models runtime.
	FoundationModels,
	/// A curated, remote, or local GGUF model served by llama.cpp.
	Gguf(TextModel),
}

/// Speech-recognition backend selection for [`InferenceBuilder::stt`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SttSelection {
	/// A Whisper checkpoint served by whisper.cpp.
	Whisper(SttModel),
	/// The multilingual Parakeet TDT recognizer.
	Parakeet,
}

impl Default for SttSelection {
	fn default() -> Self {
		Self::Whisper(SttModel::default())
	}
}

enum TextEngine {
	Fm(AppleFm),
	Gguf { generator: TextGenerator, model_id: SmolStr },
}

impl TextEngine {
	async fn load(selection: TextSelection, hub: &Hub, device: DevicePreference) -> Result<Self> {
		match selection {
			TextSelection::FoundationModels => Self::load_fm().await,
			TextSelection::Auto => {
				let availability = AppleFm::availability().await.map_err(fm_error)?;
				if availability.available {
					Self::load_fm().await
				} else {
					Self::load_gguf(SmallModel::default().into(), hub, device).await
				}
			},
			TextSelection::Gguf(model) => Self::load_gguf(model, hub, device).await,
		}
	}

	async fn load_fm() -> Result<Self> {
		AppleFm::load().await.map(Self::Fm).map_err(fm_error)
	}

	async fn load_gguf(model: TextModel, hub: &Hub, device: DevicePreference) -> Result<Self> {
		let model_id = text_model_id(&model);
		let generator = TextGeneratorBuilder::new(model)
			.hub(hub.clone())
			.device(device)
			.build()
			.await?;
		Ok(Self::Gguf { generator, model_id })
	}
}

enum SttEngine {
	Whisper(Whisper),
	Parakeet(Parakeet),
}

/// Configures an [`Inference`] facade facet by facet.
///
/// Facets are loaded concurrently during [`InferenceBuilder::build`]; model
/// downloads resolve through the shared [`Hub`].
#[derive(Clone, Default)]
pub struct InferenceBuilder {
	hub:        Option<Hub>,
	device:     DevicePreference,
	text:       Option<TextSelection>,
	stt:        Option<SttSelection>,
	tts:        bool,
	embeddings: Option<EmbeddingModel>,
}

impl InferenceBuilder {
	/// Shares one Hugging Face cache, credentials, and offline policy across
	/// every configured facet.
	pub fn hub(mut self, hub: Hub) -> Self {
		self.hub = Some(hub);
		self
	}

	/// Selects CPU, Metal, or CUDA execution for every configured facet.
	pub const fn device(mut self, device: DevicePreference) -> Self {
		self.device = device;
		self
	}

	/// Enables chat turns and token counting on the selected text backend.
	pub fn text(mut self, selection: TextSelection) -> Self {
		self.text = Some(selection);
		self
	}

	/// Enables transcription on the selected recognizer.
	pub fn stt(mut self, selection: SttSelection) -> Self {
		self.stt = Some(selection);
		self
	}

	/// Enables speech synthesis on the default Kokoro-82M model.
	pub const fn tts(mut self) -> Self {
		self.tts = true;
		self
	}

	/// Enables embeddings on a curated or fastembed model.
	pub const fn embeddings(mut self, model: EmbeddingModel) -> Self {
		self.embeddings = Some(model);
		self
	}

	/// Downloads and loads every configured facet concurrently.
	pub async fn build(self) -> Result<Inference> {
		let hub = match self.hub {
			Some(hub) => hub,
			None => Hub::new()?,
		};
		let device = self.device;
		let text = async {
			match self.text {
				None => Ok(None),
				Some(selection) => TextEngine::load(selection, &hub, device).await.map(Some),
			}
		};
		let stt = async {
			match self.stt {
				None => Ok(None),
				Some(SttSelection::Whisper(model)) => WhisperBuilder::new(model)
					.hub(hub.clone())
					.device(device)
					.build()
					.await
					.map(|whisper| Some(SttEngine::Whisper(whisper))),
				Some(SttSelection::Parakeet) => Parakeet::builder()
					.hub(hub.clone())
					.device(device)
					.build()
					.await
					.map(|parakeet| Some(SttEngine::Parakeet(parakeet))),
			}
		};
		let tts = async {
			if self.tts {
				Kokoro::builder()
					.hub(hub.clone())
					.device(device)
					.build()
					.await
					.map(Some)
			} else {
				Ok(None)
			}
		};
		let embeddings = async {
			match self.embeddings {
				None => Ok(None),
				Some(model) => EmbedderBuilder::new(model)
					.hub(hub.clone())
					.device(device)
					.build()
					.await
					.map(Some),
			}
		};
		let (text, stt, tts, embedder) = tokio::try_join!(text, stt, tts, embeddings)?;
		Ok(Inference { text, stt, tts, embedder })
	}
}

/// Local runtime implementing canonical embedded inference facets.
///
/// Mount this engine with [`crate::Embedded`], then register that canonical
/// transport in the application provider router.
pub struct Inference {
	text:     Option<TextEngine>,
	stt:      Option<SttEngine>,
	tts:      Option<Kokoro>,
	embedder: Option<Embedder>,
}

impl Inference {
	/// Starts configuring a facade with no facets enabled.
	pub fn builder() -> InferenceBuilder {
		InferenceBuilder::default()
	}

	fn start_chat(
		&self,
		request: native::ChatRequest,
	) -> std::result::Result<BoxStream<'static, native::TurnEvent>, native::Error> {
		let engine = self
			.text
			.as_ref()
			.ok_or_else(|| unsupported_error("chat", "the local text facet was not configured"))?;
		let (messages, sampling, mut unsupported_features) = prepare_chat(request)?;
		match engine {
			TextEngine::Gguf { generator, model_id } => {
				let stream = generator
					.stream(messages, gguf_options(&sampling)?)
					.map_err(facet_error)?;
				Ok(drive_gguf_turn(stream, model_id.clone(), unsupported_features))
			},
			TextEngine::Fm(fm) => {
				let (system_prompt, prompt) = fm_prompt(&messages).map_err(facet_error)?;
				let mut options = AppleFmOptions::new(prompt);
				if let Some(system) = system_prompt {
					options = options.system_prompt(system);
				}
				fm_sampling(&sampling, &mut options, &mut unsupported_features);
				let stream = fm.stream(options).map_err(fm_error).map_err(facet_error)?;
				Ok(drive_fm_turn(stream, unsupported_features))
			},
		}
	}

	async fn embed_native(
		&self,
		request: native::EmbedRequest,
	) -> std::result::Result<native::EmbedResponse, native::Error> {
		let embedder = self.embedder.as_ref().ok_or_else(|| {
			unsupported_error("embed", "the local embeddings facet was not configured")
		})?;
		if request.texts.is_empty() {
			return Err(native::Error::Provider(SmolStr::new(
				"at least one embedding text is required",
			)));
		}
		if request.dimensions == Some(0) {
			return Err(native::Error::Provider(SmolStr::new(
				"embedding dimensions must be non-zero",
			)));
		}
		let input_tokens = request
			.texts
			.iter()
			.map(|text| estimate_tokens(text.as_str()))
			.sum();
		let cancel = CancellationToken::new();
		let _cancel_on_drop = cancel.clone().drop_guard();
		let vectors = embedder
			.embed(
				request
					.texts
					.into_iter()
					.map(|text| text.to_string())
					.collect(),
				EmbeddingOptions::default(),
				cancel,
			)
			.await
			.map_err(facet_error)?;
		if let Some(dimensions) = request.dimensions {
			let native = vectors.first().map_or(0, Vec::len);
			if dimensions as usize != native {
				return Err(unsupported_error(
					"dimensions",
					"local embedding models do not support dimensionality reduction",
				));
			}
		}
		let usage = native::Usage::builder()
			.input_tokens(input_tokens)
			.output_tokens(0)
			.cache_read_tokens(0)
			.cache_write_tokens(0)
			.accuracy(native::Accuracy::Estimated)
			.detail(native::Props::default())
			.build();
		Ok(native::EmbedResponse::builder()
			.vectors(
				vectors
					.into_iter()
					.map(|values| native::EmbeddingVector::builder().values(values).build())
					.collect(),
			)
			.usage(usage)
			.build())
	}

	async fn speak_native(
		&self,
		request: native::SpeakRequest,
	) -> std::result::Result<BoxStream<'static, native::SpeakEvent>, native::Error> {
		let kokoro = self.tts.as_ref().ok_or_else(|| {
			unsupported_error("speak", "the local speech-synthesis facet was not configured")
		})?;
		if request.text.trim().is_empty() {
			return Err(native::Error::Provider(SmolStr::new("cannot synthesize empty text")));
		}
		if request.clone.is_some() {
			return Err(unsupported_error(
				"clone",
				"the local Kokoro engine does not support voice cloning",
			));
		}
		let encoding = match request.encoding {
			native::AudioEncoding::Pcm16 => native::AudioEncoding::Pcm16,
			native::AudioEncoding::Wav => native::AudioEncoding::Wav,
			_ => {
				return Err(unsupported_error("encoding", "local synthesis emits PCM16 or WAV"));
			},
		};
		if request.sample_rate_hz == Some(0) {
			return Err(native::Error::Provider(SmolStr::new("sample rate must be non-zero")));
		}
		let speed = request.speed.map(f64_to_f32).transpose()?.unwrap_or(1.0);
		if speed <= 0.0 {
			return Err(native::Error::Provider(SmolStr::new(
				"synthesis speed must be greater than zero",
			)));
		}
		let voice = if request.voice.is_empty() {
			KokoroVoice::default()
		} else {
			KokoroVoice::new(request.voice).map_err(facet_error)?
		};
		let mut unsupported_features = Vec::new();
		if !request.instructions.is_empty() {
			unsupported_features
				.push(unsupported("instructions", "Kokoro has no style-instruction channel"));
		}
		let cancel = CancellationToken::new();
		let _cancel_on_drop = cancel.clone().drop_guard();
		let audio = kokoro
			.synthesize(
				request.text.to_string(),
				&voice,
				SynthesisOptions { speed, ..SynthesisOptions::default() },
				cancel,
			)
			.await
			.map_err(facet_error)?;
		let duration_ms = audio.duration().as_millis() as u64;
		let local_encoding = match encoding {
			native::AudioEncoding::Pcm16 => LocalAudioEncoding::Pcm16,
			native::AudioEncoding::Wav => LocalAudioEncoding::Wav,
			_ => unreachable!("validated local encoding"),
		};
		let (encoded, sample_rate_hz) =
			encode_speech(audio, local_encoding, request.sample_rate_hz).map_err(facet_error)?;
		let encoded = Bytes::from(encoded);
		let mut hash = [0_u8; 32];
		hash.copy_from_slice(blake3::hash(&encoded).as_bytes());
		let blob = native::BlobPart::builder()
			.hash(hash)
			.mime(SmolStr::new(match encoding {
				native::AudioEncoding::Pcm16 => "audio/L16",
				native::AudioEncoding::Wav => "audio/wav",
				_ => unreachable!("validated local encoding"),
			}))
			.size(encoded.len() as u64)
			.inline(encoded.clone())
			.build();
		let mut props = native::Props::default();
		props.insert_ns("omp.local", "sample_rate_hz", serde_json::Value::from(sample_rate_hz));
		let mut events =
			Vec::with_capacity(encoded.len().div_ceil(SPEAK_CHUNK_BYTES).saturating_add(1));
		for start in (0..encoded.len()).step_by(SPEAK_CHUNK_BYTES) {
			let end = (start + SPEAK_CHUNK_BYTES).min(encoded.len());
			events.push(native::SpeakEvent::Chunk(
				native::SpeakChunk::builder()
					.audio(encoded.slice(start..end))
					.transcript_delta(SmolStr::new(""))
					.build(),
			));
		}
		events.push(native::SpeakEvent::Done(
			native::SpeakDone::builder()
				.audio(blob)
				.duration_ms(duration_ms)
				.unsupported(unsupported_features)
				.props(props)
				.build(),
		));
		Ok(futures::stream::iter(events).boxed())
	}

	async fn transcribe_native(
		&self,
		request: native::TranscribeRequest,
	) -> std::result::Result<native::TranscribeResponse, native::Error> {
		let engine = self.stt.as_ref().ok_or_else(|| {
			unsupported_error("transcribe", "the local transcription facet was not configured")
		})?;
		let audio = decode_inline_audio(&request.audio)?;
		let duration_ms = audio.duration().as_millis() as u64;
		let mut unsupported_features = Vec::new();
		if request
			.granularities
			.contains(&native::TranscriptionGranularity::Word)
		{
			unsupported_features.push(unsupported(
				"granularities.word",
				"local recognizers do not expose word-level timestamps",
			));
		}
		if request.diarize {
			unsupported_features
				.push(unsupported("diarize", "local recognizers do not separate speakers"));
		}
		let mut temperature = request.temperature.map(f64_to_f32).transpose()?;
		if matches!(engine, SttEngine::Parakeet(_)) && temperature.is_some() {
			unsupported_features.push(unsupported("temperature", "Parakeet always decodes greedily"));
			temperature = None;
		}
		let options = TranscriptionOptions {
			language: (!request.language.is_empty()).then_some(request.language),
			translate: request.translate,
			timestamps: request.granularities.is_empty()
				|| request
					.granularities
					.contains(&native::TranscriptionGranularity::Segment),
			initial_prompt: (!request.prompt.is_empty()).then_some(request.prompt),
			temperature,
		};
		let cancel = CancellationToken::new();
		let _cancel_on_drop = cancel.clone().drop_guard();
		let transcription = match engine {
			SttEngine::Whisper(whisper) => whisper
				.transcribe(audio, options, cancel)
				.await
				.map_err(facet_error)?,
			SttEngine::Parakeet(parakeet) => parakeet
				.transcribe(audio, options, cancel)
				.await
				.map_err(facet_error)?,
		};
		Ok(native::TranscribeResponse::builder()
			.text(transcription.text)
			.language(transcription.language.unwrap_or_default())
			.duration_ms(duration_ms)
			.segments(
				transcription
					.segments
					.into_iter()
					.map(|segment| {
						native::TranscriptSegment::builder()
							.start_ms(segment.start.as_millis() as u64)
							.end_ms(segment.end.as_millis() as u64)
							.text(segment.text)
							.build()
					})
					.collect(),
			)
			.words(Vec::new())
			.unsupported(unsupported_features)
			.props(native::Props::default())
			.build())
	}

	/// Stops every configured backend after queued work drains.
	pub async fn shutdown(&self) -> Result<()> {
		if let Some(engine) = &self.stt {
			match engine {
				SttEngine::Whisper(whisper) => whisper.shutdown().await?,
				SttEngine::Parakeet(parakeet) => parakeet.shutdown().await?,
			}
		}
		if let Some(kokoro) = &self.tts {
			kokoro.shutdown().await?;
		}
		if let Some(embedder) = &self.embedder {
			embedder.shutdown().await?;
		}
		if let Some(TextEngine::Gguf { generator, .. }) = &self.text {
			generator.shutdown().await?;
		}
		Ok(())
	}
}

impl LocalEngine for Inference {
	fn chat(
		&self,
		request: native::ChatRequest,
		_executor: Option<std::sync::Arc<dyn native::Executor>>,
	) -> impl Future<Output = std::result::Result<BoxStream<'static, native::TurnEvent>, native::Error>>
	+ Send
	+ '_ {
		std::future::ready(self.start_chat(request))
	}

	fn embed(
		&self,
		request: native::EmbedRequest,
	) -> impl Future<Output = std::result::Result<native::EmbedResponse, native::Error>> + Send + '_
	{
		self.embed_native(request)
	}

	fn speak(
		&self,
		request: native::SpeakRequest,
	) -> impl Future<Output = std::result::Result<BoxStream<'static, native::SpeakEvent>, native::Error>>
	+ Send
	+ '_ {
		self.speak_native(request)
	}

	fn transcribe(
		&self,
		request: native::TranscribeRequest,
	) -> impl Future<Output = std::result::Result<native::TranscribeResponse, native::Error>> + Send + '_
	{
		self.transcribe_native(request)
	}
}

fn prepare_chat(
	request: native::ChatRequest,
) -> std::result::Result<
	(Vec<ChatMessage>, native::Sampling, Vec<native::Unsupported>),
	native::Error,
> {
	let messages = canonical_messages(request.thread)?;
	if messages.is_empty() {
		return Err(native::Error::Provider(SmolStr::new("at least one chat message is required")));
	}
	let mut unsupported_features = Vec::new();
	if !request.tools.is_empty() {
		unsupported_features.push(unsupported("tools", "local text engines do not invoke tools"));
	}
	if let Some(feature) = request.tool_choice
		&& matches!(feature.value, native::ToolChoice::Required | native::ToolChoice::Named(_))
	{
		admit_drop(
			feature.on_unsupported,
			"tool_choice",
			"local text engines do not invoke tools",
			&mut unsupported_features,
		)?;
	}
	if let Some(feature) = request.thinking {
		admit_drop(
			feature.on_unsupported,
			"thinking",
			"local text engines expose no selectable thinking controls",
			&mut unsupported_features,
		)?;
	}
	if let Some(feature) = request.response_format {
		admit_drop(
			feature.on_unsupported,
			"response_format",
			"local text engines apply no structured-output constraint",
			&mut unsupported_features,
		)?;
	}
	if request.cache.is_some() {
		unsupported_features
			.push(unsupported("cache", "embedded local engines retain no prompt-cache affinity"));
	}
	if request
		.provider_options
		.as_ref()
		.is_some_and(|options| !options.is_empty())
	{
		unsupported_features.push(unsupported(
			"provider_options",
			"embedded local engines define no provider-specific options",
		));
	}
	Ok((messages, request.sampling.unwrap_or_default(), unsupported_features))
}

fn canonical_messages(
	thread: native::Thread,
) -> std::result::Result<Vec<ChatMessage>, native::Error> {
	thread
		.items
		.into_iter()
		.map(|item| {
			let native::ItemKind::Message(message) = item.kind else {
				return Err(unsupported_error(
					"thread.items",
					"local text engines cannot replay tool calls or tool results",
				));
			};
			let role = match message.role {
				native::Role::System => ChatRole::System,
				native::Role::User => ChatRole::User,
				native::Role::Assistant => ChatRole::Assistant,
				_ => return Err(unsupported_error("thread.role", "unknown local message role")),
			};
			let mut content = String::new();
			for part in message.parts {
				match part {
					native::Part::Text(text) => content.push_str(text.as_str()),
					_ => {
						return Err(unsupported_error(
							"thread.parts",
							"local text engines accept text message parts only",
						));
					},
				}
			}
			Ok(ChatMessage { role, content: content.into() })
		})
		.collect()
}

fn decode_inline_audio(blob: &native::BlobPart) -> std::result::Result<Audio, native::Error> {
	if blob.inline.is_empty() {
		return Err(unsupported_error(
			"audio.inline",
			"embedded transcription requires inline PCM16 WAV audio",
		));
	}
	let bytes = blob.inline.as_ref();
	if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
		return Err(unsupported_error(
			"audio.mime",
			"embedded transcription accepts PCM16 WAV audio",
		));
	}
	let mut format = None;
	let mut data = None;
	let mut offset = 12usize;
	while offset.saturating_add(8) <= bytes.len() {
		let id = &bytes[offset..offset + 4];
		let length = u32::from_le_bytes(
			bytes[offset + 4..offset + 8]
				.try_into()
				.expect("four-byte WAV chunk length"),
		) as usize;
		let start = offset + 8;
		let end = start
			.checked_add(length)
			.ok_or_else(|| native::Error::Provider(SmolStr::new("WAV chunk length overflowed")))?;
		let chunk = bytes
			.get(start..end)
			.ok_or_else(|| native::Error::Provider(SmolStr::new("truncated inline WAV chunk")))?;
		if id == b"fmt " {
			if chunk.len() < 16 {
				return Err(native::Error::Provider(SmolStr::new("truncated WAV format chunk")));
			}
			format = Some((
				u16::from_le_bytes([chunk[0], chunk[1]]),
				u16::from_le_bytes([chunk[2], chunk[3]]),
				u32::from_le_bytes(chunk[4..8].try_into().expect("four-byte WAV rate")),
				u16::from_le_bytes([chunk[14], chunk[15]]),
			));
		} else if id == b"data" {
			data = Some(chunk);
		}
		offset = end.saturating_add(length & 1);
	}
	let Some((encoding, channels, sample_rate, bits)) = format else {
		return Err(native::Error::Provider(SmolStr::new("inline WAV has no format chunk")));
	};
	if encoding != 1 || bits != 16 {
		return Err(unsupported_error(
			"audio.mime",
			"embedded transcription accepts integer PCM16 WAV audio",
		));
	}
	let data =
		data.ok_or_else(|| native::Error::Provider(SmolStr::new("inline WAV has no data chunk")))?;
	if !data.len().is_multiple_of(2) {
		return Err(native::Error::Provider(SmolStr::new(
			"inline PCM16 WAV contains a partial sample",
		)));
	}
	let samples = data
		.as_chunks::<2>()
		.0
		.iter()
		.map(|sample| f32::from(i16::from_le_bytes(*sample)) / 32_768.0)
		.collect();
	Audio::new(samples, sample_rate, channels).map_err(facet_error)
}

// ── Turn drivers ────────────────────────────────────────────────────────

fn drive_gguf_turn(
	mut stream: GenerationStream,
	model_id: SmolStr,
	unsupported_features: Vec<native::Unsupported>,
) -> BoxStream<'static, native::TurnEvent> {
	Box::pin(async_stream::stream! {
		yield native::TurnEvent::Accepted { replay: false };
		yield native::TurnEvent::PartStart {
			index: 0,
			kind: native::StreamPartKind::Text,
			tool_call_id: SmolStr::new(""),
			tool_name: SmolStr::new(""),
		};
		let mut text = String::new();
		while let Some(item) = stream.next().await {
			match item {
				Ok(chunk) => {
					text.push_str(chunk.as_str());
					yield native::TurnEvent::PartDelta {
						index: 0,
						chunk: Bytes::copy_from_slice(chunk.as_bytes()),
					};
				},
				Err(error) => {
					yield canonical_turn_error(error);
					return;
				},
			}
		}
		yield native::TurnEvent::PartEnd { index: 0, signature: Bytes::new() };
		let (usage, stop) = match stream.summary() {
			Some(summary) => (
				Some(
					native::Usage::builder()
						.input_tokens(summary.prompt_tokens)
						.output_tokens(summary.output_tokens)
						.cache_read_tokens(0)
						.cache_write_tokens(0)
						.accuracy(native::Accuracy::Exact)
						.detail(native::Props::default())
						.build(),
				),
				stop_reason(summary.stop),
			),
			None => (None, native::StopReason::EndTurn),
		};
		yield native_outcome(
			text.into(),
			stop,
			usage,
			unsupported_features,
			SmolStr::new("localai"),
			model_id,
		);
	})
}

fn drive_fm_turn(
	mut stream: AppleFmStream,
	unsupported_features: Vec<native::Unsupported>,
) -> BoxStream<'static, native::TurnEvent> {
	Box::pin(async_stream::stream! {
		yield native::TurnEvent::Accepted { replay: false };
		yield native::TurnEvent::PartStart {
			index: 0,
			kind: native::StreamPartKind::Text,
			tool_call_id: SmolStr::new(""),
			tool_name: SmolStr::new(""),
		};
		let mut text = String::new();
		while let Some(item) = stream.next().await {
			match item {
				Ok(AppleFmEvent::Delta(chunk)) => {
					text.push_str(chunk.as_str());
					yield native::TurnEvent::PartDelta {
						index: 0,
						chunk: Bytes::copy_from_slice(chunk.as_bytes()),
					};
				},
				Ok(AppleFmEvent::Finished(generation)) => {
					yield native::TurnEvent::PartEnd { index: 0, signature: Bytes::new() };
					let usage = native::Usage::builder()
						.input_tokens(u64::from(generation.prompt_tokens_estimated))
						.output_tokens(u64::from(generation.completion_tokens_estimated))
						.cache_read_tokens(0)
						.cache_write_tokens(0)
						.accuracy(native::Accuracy::Estimated)
						.detail(native::Props::default())
						.build();
					yield native_outcome(
						generation.content,
						native::StopReason::EndTurn,
						Some(usage),
						unsupported_features,
						SmolStr::new("apple"),
						SmolStr::new("foundation-models"),
					);
					return;
				},
				Err(error) => {
					yield canonical_turn_error(fm_error(error));
					return;
				},
			}
		}
		yield native::TurnEvent::PartEnd { index: 0, signature: Bytes::new() };
		yield native_outcome(
			text.into(),
			native::StopReason::EndTurn,
			None,
			unsupported_features,
			SmolStr::new("apple"),
			SmolStr::new("foundation-models"),
		);
	})
}

fn native_outcome(
	text: SmolStr,
	stop: native::StopReason,
	usage: Option<native::Usage>,
	unsupported_features: Vec<native::Unsupported>,
	provider: SmolStr,
	model: SmolStr,
) -> native::TurnEvent {
	let output = native::Item::builder()
		.seq(0)
		.kind(native::ItemKind::Message(
			native::Message::builder()
				.role(native::Role::Assistant)
				.parts(vec![native::Part::Text(text)])
				.build(),
		))
		.props(native::Props::default())
		.build();
	native::TurnEvent::Outcome(
		native::ChatOutcome::builder()
			.output(vec![output])
			.stop(stop)
			.maybe_usage(usage)
			.unsupported(unsupported_features)
			.provider(provider)
			.model(model)
			.props(native::Props::default())
			.build(),
	)
}

fn canonical_turn_error(error: Error) -> native::TurnEvent {
	native::TurnEvent::Error(
		native::TurnError::builder()
			.kind(match error.kind() {
				crate::ErrorKind::Unsupported => native::TurnErrorKind::Unsupported,
				_ => native::TurnErrorKind::Upstream,
			})
			.detail(SmolStr::from(error.to_string()))
			.unsupported(Vec::new())
			.retry_after_ms(0)
			.build(),
	)
}

fn gguf_options(
	sampling: &native::Sampling,
) -> std::result::Result<GenerationOptions, native::Error> {
	let max_tokens = match sampling.max_output_tokens {
		None => GenerationOptions::default().max_tokens,
		Some(0) => {
			return Err(native::Error::Provider(SmolStr::new(
				"maximum output tokens must be non-zero",
			)));
		},
		Some(max) => usize::try_from(max).map_err(|_| {
			native::Error::Provider(SmolStr::new("maximum output tokens exceed this platform"))
		})?,
	};
	Ok(GenerationOptions {
		max_tokens,
		temperature: sampling.temperature.map(f64_to_f32).transpose()?,
		top_p: sampling.top_p.map(f64_to_f32).transpose()?,
		top_k: sampling.top_k,
		min_p: sampling.min_p.map(f64_to_f32).transpose()?,
		repetition_penalty: None,
		frequency_penalty: sampling.frequency_penalty.map(f64_to_f32).transpose()?,
		presence_penalty: sampling.presence_penalty.map(f64_to_f32).transpose()?,
		seed: 0,
		stop: sampling.stop.clone().unwrap_or_default(),
	})
}

fn fm_sampling(
	sampling: &native::Sampling,
	options: &mut AppleFmOptions,
	unsupported_features: &mut Vec<native::Unsupported>,
) {
	if let Some(temperature) = sampling.temperature {
		options.temperature = Some(temperature);
	}
	if let Some(max_tokens) = sampling.max_output_tokens {
		if let Ok(max_tokens) = u32::try_from(max_tokens) {
			options.max_tokens = Some(max_tokens);
		} else {
			unsupported_features.push(
				native::Unsupported::builder()
					.what(SmolStr::new("sampling.max_output_tokens"))
					.detail(format_smol!("clamped to {}, the framework's u32 limit", u32::MAX))
					.action(native::UnsupportedAction::Clamped)
					.build(),
			);
			options.max_tokens = Some(u32::MAX);
		}
	}
	for (what, set) in [
		("sampling.top_p", sampling.top_p.is_some()),
		("sampling.top_k", sampling.top_k.is_some()),
		("sampling.min_p", sampling.min_p.is_some()),
		("sampling.frequency_penalty", sampling.frequency_penalty.is_some()),
		("sampling.presence_penalty", sampling.presence_penalty.is_some()),
		("sampling.stop", sampling.stop.as_ref().is_some_and(|stop| !stop.is_empty())),
	] {
		if set {
			unsupported_features
				.push(unsupported(what, "Apple Foundation Models exposes no such sampling control"));
		}
	}
}

fn fm_prompt(messages: &[ChatMessage]) -> Result<(Option<SmolStr>, SmolStr)> {
	let mut system = Vec::new();
	let mut turns = Vec::new();
	for message in messages {
		match message.role {
			ChatRole::System => system.push(message.content.as_str()),
			ChatRole::User => turns.push(("User", message.content.as_str())),
			ChatRole::Assistant => turns.push(("Assistant", message.content.as_str())),
		}
	}
	if turns.is_empty() {
		return Err(Error::invalid("at least one non-system chat message is required"));
	}
	let system_prompt = (!system.is_empty()).then(|| system.join("\n\n").into());
	let prompt = if let [("User", only)] = turns.as_slice() {
		(*only).into()
	} else {
		turns
			.iter()
			.map(|(role, content)| format!("{role}: {content}"))
			.collect::<Vec<_>>()
			.join("\n\n")
			.into()
	};
	Ok((system_prompt, prompt))
}

const fn stop_reason(stop: GenerationStop) -> native::StopReason {
	if matches!(stop, GenerationStop::MaxTokens) {
		native::StopReason::MaxTokens
	} else {
		native::StopReason::EndTurn
	}
}

fn admit_drop(
	fallback: native::Fallback,
	what: &'static str,
	detail: &'static str,
	unsupported_features: &mut Vec<native::Unsupported>,
) -> std::result::Result<(), native::Error> {
	if !matches!(fallback, native::Fallback::Ignore) {
		return Err(unsupported_error(what, detail));
	}
	unsupported_features.push(unsupported(what, detail));
	Ok(())
}

fn unsupported(what: &'static str, detail: &'static str) -> native::Unsupported {
	native::Unsupported::builder()
		.what(SmolStr::new(what))
		.detail(SmolStr::new(detail))
		.action(native::UnsupportedAction::Dropped)
		.build()
}

fn unsupported_error(what: &'static str, detail: &'static str) -> native::Error {
	native::Error::Unsupported(vec![unsupported(what, detail)])
}

fn facet_error(error: Error) -> native::Error {
	match error {
		Error::Unavailable(detail) => native::Error::Unsupported(vec![
			native::Unsupported::builder()
				.what(SmolStr::new("local_engine"))
				.detail(detail)
				.action(native::UnsupportedAction::Dropped)
				.build(),
		]),
		other => native::Error::Provider(SmolStr::from(other.to_string())),
	}
}

fn f64_to_f32(value: f64) -> std::result::Result<f32, native::Error> {
	if !value.is_finite() || value < f64::from(f32::MIN) || value > f64::from(f32::MAX) {
		return Err(native::Error::Provider(SmolStr::new(
			"local floating-point control is outside the supported f32 range",
		)));
	}
	Ok(value as f32)
}

fn fm_error(error: AppleFmError) -> Error {
	match error.code() {
		AppleFmErrorCode::Cancelled => Error::Cancelled,
		AppleFmErrorCode::InvalidInput | AppleFmErrorCode::ContextOverflow => {
			Error::invalid(error.message())
		},
		AppleFmErrorCode::ModelUnavailable
		| AppleFmErrorCode::DeviceNotEligible
		| AppleFmErrorCode::AppleIntelligenceNotEnabled
		| AppleFmErrorCode::ModelNotReady => Error::unavailable(error.message()),
		_ => Error::backend("apple-fm", error.message()),
	}
}

fn text_model_id(model: &TextModel) -> SmolStr {
	match model {
		TextModel::Small(small) => match small {
			SmallModel::Lfm2_350M => "lfm2-350m".into(),
			SmallModel::Lfm2_700M => "lfm2-700m".into(),
			SmallModel::Lfm2_1_2B => "lfm2-1.2b".into(),
			SmallModel::Qwen3_600M => "qwen3-600m".into(),
			SmallModel::Qwen2_5_500M => "qwen2.5-500m".into(),
		},
		TextModel::HuggingFace { repo, .. } => repo.id().into(),
		TextModel::Local(path) => path
			.file_name()
			.map_or_else(|| "local.gguf".into(), |name| name.to_string_lossy().into_owned().into()),
	}
}

fn estimate_tokens(text: &str) -> u64 {
	text.len().div_ceil(4).max(1) as u64
}

#[derive(Clone, Copy)]
enum LocalAudioEncoding {
	Pcm16,
	Wav,
}
// ── Speech encoding ─────────────────────────────────────────────────────

fn encode_speech(
	audio: Audio,
	encoding: LocalAudioEncoding,
	sample_rate_hz: Option<u32>,
) -> Result<(Vec<u8>, u32)> {
	let rate = sample_rate_hz.unwrap_or_else(|| audio.sample_rate());
	let samples = match sample_rate_hz {
		Some(rate) if rate != audio.sample_rate() => audio.into_mono_at(rate)?,
		_ => audio.into_samples(),
	};
	let pcm = pcm16_bytes(&samples);
	match encoding {
		LocalAudioEncoding::Pcm16 => Ok((pcm, rate)),
		LocalAudioEncoding::Wav => Ok((wav_bytes(&pcm, rate)?, rate)),
	}
}

fn pcm16_bytes(samples: &[f32]) -> Vec<u8> {
	let mut bytes = Vec::with_capacity(samples.len() * 2);
	for &sample in samples {
		let value = (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16;
		bytes.extend_from_slice(&value.to_le_bytes());
	}
	bytes
}

fn wav_bytes(pcm: &[u8], sample_rate: u32) -> Result<Vec<u8>> {
	let data_size = u32::try_from(pcm.len())
		.map_err(|_| Error::invalid("audio is too large for a RIFF WAV file"))?;
	let riff_size = data_size
		.checked_add(36)
		.ok_or_else(|| Error::invalid("audio is too large for a RIFF WAV file"))?;
	let byte_rate = sample_rate
		.checked_mul(2)
		.ok_or_else(|| Error::invalid("WAV byte rate overflowed"))?;
	let mut wav = Vec::with_capacity(pcm.len() + 44);
	wav.extend_from_slice(b"RIFF");
	wav.extend_from_slice(&riff_size.to_le_bytes());
	wav.extend_from_slice(b"WAVE");
	wav.extend_from_slice(b"fmt ");
	wav.extend_from_slice(&16_u32.to_le_bytes());
	wav.extend_from_slice(&1_u16.to_le_bytes()); // PCM
	wav.extend_from_slice(&1_u16.to_le_bytes()); // mono
	wav.extend_from_slice(&sample_rate.to_le_bytes());
	wav.extend_from_slice(&byte_rate.to_le_bytes());
	wav.extend_from_slice(&2_u16.to_le_bytes()); // block align
	wav.extend_from_slice(&16_u16.to_le_bytes()); // bits per sample
	wav.extend_from_slice(b"data");
	wav.extend_from_slice(&data_size.to_le_bytes());
	wav.extend_from_slice(pcm);
	Ok(wav)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn pcm16_quantizes_and_clamps() {
		let bytes = pcm16_bytes(&[0.0, 1.0, -1.0, 2.0, -2.0]);
		let samples: Vec<i16> = bytes
			.as_chunks::<2>()
			.0
			.iter()
			.map(|chunk| i16::from_le_bytes(*chunk))
			.collect();
		assert_eq!(samples, [0, i16::MAX, -i16::MAX, i16::MAX, -i16::MAX]);
	}

	#[test]
	fn wav_header_describes_mono_pcm16() {
		let pcm = pcm16_bytes(&[0.0; 2400]);
		let wav = wav_bytes(&pcm, 24_000).unwrap();
		assert_eq!(&wav[0..4], b"RIFF");
		assert_eq!(&wav[8..12], b"WAVE");
		assert_eq!(u32::from_le_bytes(wav[4..8].try_into().unwrap()), wav.len() as u32 - 8);
		assert_eq!(u16::from_le_bytes(wav[20..22].try_into().unwrap()), 1);
		assert_eq!(u16::from_le_bytes(wav[22..24].try_into().unwrap()), 1);
		assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 24_000);
		assert_eq!(u16::from_le_bytes(wav[34..36].try_into().unwrap()), 16);
		assert_eq!(u32::from_le_bytes(wav[40..44].try_into().unwrap()), pcm.len() as u32);
		assert_eq!(wav.len(), 44 + pcm.len());
	}
}
