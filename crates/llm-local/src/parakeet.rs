use std::{path::PathBuf, time::Duration};

use omp_core::Str;
use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig, OfflineTransducerModelConfig};
use tokio_util::sync::CancellationToken;

use crate::{
	Accelerator, Audio, DevicePreference, Error, Hub, ModelRepo, Result, Transcription,
	TranscriptionOptions, TranscriptionSegment, worker::Worker,
};

const PARAKEET_REPO: &str = "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8";
const PARAKEET_SAMPLE_RATE: u32 = 16_000;

/// Repository-relative files forming a sherpa-onnx `NeMo` transducer model.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParakeetFiles {
	/// Acoustic encoder ONNX graph.
	pub encoder: Str,
	/// Autoregressive decoder ONNX graph.
	pub decoder: Str,
	/// Encoder-decoder joiner ONNX graph.
	pub joiner:  Str,
	/// Token table used to reconstruct recognized text.
	pub tokens:  Str,
}

impl Default for ParakeetFiles {
	fn default() -> Self {
		Self {
			encoder: "encoder.int8.onnx".into(),
			decoder: "decoder.int8.onnx".into(),
			joiner:  "joiner.int8.onnx".into(),
			tokens:  "tokens.txt".into(),
		}
	}
}

/// Configures the multilingual Parakeet TDT recognizer used by the prior local
/// STT stack.
#[derive(Clone)]
pub struct ParakeetBuilder {
	hub:     Option<Hub>,
	repo:    ModelRepo,
	files:   ParakeetFiles,
	device:  DevicePreference,
	threads: Option<usize>,
	debug:   bool,
}

impl ParakeetBuilder {
	/// Creates a builder for the 0.6B Parakeet TDT v3 int8 model.
	pub fn new() -> Self {
		Self {
			hub:     None,
			repo:    ModelRepo::new(PARAKEET_REPO),
			files:   ParakeetFiles::default(),
			device:  DevicePreference::Auto,
			threads: None,
			debug:   false,
		}
	}

	/// Shares Hugging Face cache, authentication, endpoint, and offline policy.
	pub fn hub(mut self, hub: Hub) -> Self {
		self.hub = Some(hub);
		self
	}

	/// Overrides the model repository and four `NeMo` transducer files.
	pub fn source(mut self, repo: ModelRepo, files: ParakeetFiles) -> Self {
		self.repo = repo;
		self.files = files;
		self
	}

	/// Selects execution; current sherpa-onnx release archives support CPU on
	/// every target.
	pub const fn device(mut self, device: DevicePreference) -> Self {
		self.device = device;
		self
	}

	/// Sets the ONNX Runtime intra-op thread count.
	pub const fn threads(mut self, threads: usize) -> Self {
		self.threads = Some(threads);
		self
	}

	/// Enables sherpa-onnx diagnostic logging.
	pub const fn debug(mut self, enabled: bool) -> Self {
		self.debug = enabled;
		self
	}

	/// Fetches all four model files concurrently and initializes sherpa-onnx off
	/// Tokio.
	pub async fn build(self) -> Result<Parakeet> {
		if self.threads == Some(0) {
			return Err(Error::invalid("Parakeet thread count must be non-zero"));
		}
		if !matches!(self.device, DevicePreference::Auto | DevicePreference::Cpu) {
			return Err(Error::unavailable(
				"the bundled sherpa-onnx runtime supports Parakeet on CPU",
			));
		}
		let hub = match self.hub {
			Some(hub) => hub,
			None => Hub::new()?,
		};
		let names = [self.files.encoder, self.files.decoder, self.files.joiner, self.files.tokens];
		let paths = hub.files(&self.repo, names).await?;
		let [encoder, decoder, joiner, tokens]: [PathBuf; 4] = paths
			.try_into()
			.map_err(|_| Error::backend("parakeet", "model download returned the wrong file count"))?;
		let threads = self.threads.unwrap_or_else(|| {
			std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
		});
		let threads =
			i32::try_from(threads).map_err(|_| Error::invalid("Parakeet thread count exceeds i32"))?;
		let debug = self.debug;
		let worker = Worker::spawn("omp-llm-local-parakeet", move || {
			let mut config = OfflineRecognizerConfig::default();
			config.model_config.transducer = OfflineTransducerModelConfig {
				encoder: Some(path_text(encoder)?),
				decoder: Some(path_text(decoder)?),
				joiner:  Some(path_text(joiner)?),
			};
			config.model_config.tokens = Some(path_text(tokens)?);
			config.model_config.model_type = Some("nemo_transducer".into());
			config.model_config.provider = Some("cpu".into());
			config.model_config.num_threads = threads;
			config.model_config.debug = debug;
			OfflineRecognizer::create(&config)
				.ok_or_else(|| Error::backend("parakeet", "failed to create offline recognizer"))
		})
		.await?;
		Ok(Parakeet { worker })
	}
}

impl Default for ParakeetBuilder {
	fn default() -> Self {
		Self::new()
	}
}

/// Fully asynchronous multilingual Parakeet TDT speech recognizer.
#[derive(Clone)]
pub struct Parakeet {
	worker: Worker<OfflineRecognizer>,
}

impl Parakeet {
	/// Starts a builder for Parakeet TDT 0.6B v3 int8.
	pub fn builder() -> ParakeetBuilder {
		ParakeetBuilder::new()
	}

	/// Execution backend selected for this model.
	pub const fn accelerator(&self) -> Accelerator {
		Accelerator::Cpu
	}

	/// Consumes PCM audio and runs one multilingual offline decode.
	pub async fn transcribe(
		&self,
		audio: Audio,
		options: TranscriptionOptions,
		cancel: CancellationToken,
	) -> Result<Transcription> {
		validate_options(&options)?;
		let samples = tokio::task::spawn_blocking(move || audio.into_mono_at(PARAKEET_SAMPLE_RATE))
			.await
			.map_err(|error| Error::backend("audio resampler", error))??;
		transcribe(&self.worker, samples, options, cancel).await
	}

	/// Opens an incremental input session whose buffered audio is moved into one
	/// final decode.
	pub fn session(&self, options: TranscriptionOptions) -> ParakeetSession {
		ParakeetSession { worker: self.worker.clone(), options, samples: Vec::new() }
	}

	/// Stops the recognizer after queued requests drain.
	pub async fn shutdown(&self) -> Result<()> {
		self.worker.shutdown().await
	}
}

/// Incremental Parakeet audio-input session for live capture pipelines.
pub struct ParakeetSession {
	worker:  Worker<OfflineRecognizer>,
	options: TranscriptionOptions,
	samples: Vec<f32>,
}

impl ParakeetSession {
	/// Appends one captured PCM chunk after downmixing and resampling it to 16
	/// kHz.
	pub fn push(&mut self, audio: Audio) -> Result<()> {
		self
			.samples
			.extend(audio.into_mono_at(PARAKEET_SAMPLE_RATE)?);
		Ok(())
	}

	/// Moves all buffered audio into sherpa-onnx and closes the session.
	pub async fn finish(self, cancel: CancellationToken) -> Result<Transcription> {
		validate_options(&self.options)?;
		transcribe(&self.worker, self.samples, self.options, cancel).await
	}
}

async fn transcribe(
	worker: &Worker<OfflineRecognizer>,
	samples: Vec<f32>,
	options: TranscriptionOptions,
	cancel: CancellationToken,
) -> Result<Transcription> {
	worker
		.run(cancel, move |recognizer, cancel| {
			let stream = match options.initial_prompt.as_deref() {
				Some(hotword) => recognizer.create_stream_with_hotwords(hotword),
				None => recognizer.create_stream(),
			};
			stream.accept_waveform(PARAKEET_SAMPLE_RATE as i32, &samples);
			recognizer.decode(&stream);
			if cancel.is_cancelled() {
				return Err(Error::Cancelled);
			}
			let result = stream
				.get_result()
				.ok_or_else(|| Error::backend("parakeet", "recognizer returned no result"))?;
			let audio_end =
				Duration::from_secs_f64(samples.len() as f64 / f64::from(PARAKEET_SAMPLE_RATE));
			let segments = if options.timestamps {
				timed_segments(&result, audio_end)
			} else {
				Vec::new()
			};
			Ok(Transcription { text: result.text.trim().into(), segments, language: None })
		})
		.await
}

fn timed_segments(
	result: &sherpa_onnx::OfflineRecognizerResult,
	audio_end: Duration,
) -> Vec<TranscriptionSegment> {
	let Some(timestamps) = result.timestamps.as_deref() else {
		return whole_segment(&result.text, audio_end);
	};
	if timestamps.len() != result.tokens.len() || timestamps.is_empty() {
		return whole_segment(&result.text, audio_end);
	}
	let durations = result.durations.as_deref();
	result
		.tokens
		.iter()
		.zip(timestamps)
		.enumerate()
		.map(|(index, (token, start))| {
			let start = seconds(*start);
			let end = durations.and_then(|values| values.get(index)).map_or_else(
				|| {
					timestamps
						.get(index + 1)
						.map_or(audio_end, |value| seconds(*value))
				},
				|duration| start.saturating_add(seconds(*duration)),
			);
			TranscriptionSegment {
				text: token.as_str().into(),
				start,
				end: end.min(audio_end),
				no_speech_probability: None,
			}
		})
		.collect()
}

fn whole_segment(text: &str, end: Duration) -> Vec<TranscriptionSegment> {
	if text.trim().is_empty() {
		Vec::new()
	} else {
		vec![TranscriptionSegment {
			text: text.trim().into(),
			start: Duration::ZERO,
			end,
			no_speech_probability: None,
		}]
	}
}

fn seconds(value: f32) -> Duration {
	if value.is_finite() && value > 0.0 {
		Duration::from_secs_f32(value)
	} else {
		Duration::ZERO
	}
}

fn validate_options(options: &TranscriptionOptions) -> Result<()> {
	if options.translate {
		return Err(Error::invalid("Parakeet does not support translation"));
	}
	if options.language.is_some() {
		return Err(Error::invalid("Parakeet detects its supported languages automatically"));
	}
	if options.temperature.is_some() {
		return Err(Error::invalid("Parakeet always decodes greedily"));
	}
	Ok(())
}

fn path_text(path: PathBuf) -> Result<String> {
	path
		.into_os_string()
		.into_string()
		.map_err(|_| Error::invalid("model path is not valid UTF-8"))
}

#[cfg(test)]
mod tests {
	use std::time::Duration;

	use sherpa_onnx::OfflineRecognizerResult;

	use super::timed_segments;

	#[test]
	fn token_timestamps_are_bounded_by_submitted_audio() {
		let result = OfflineRecognizerResult {
			text:       "hello world".into(),
			tokens:     vec!["hello".into(), "world".into()],
			timestamps: Some(vec![0.25, 0.75]),
			durations:  None,
		};
		let segments = timed_segments(&result, Duration::from_secs(1));
		assert_eq!(segments.len(), 2);
		assert_eq!(segments[0].start, Duration::from_millis(250));
		assert_eq!(segments[0].end, Duration::from_millis(750));
		assert_eq!(segments[1].end, Duration::from_secs(1));
	}
}
