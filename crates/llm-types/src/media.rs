use bon::Builder;
use bytes::Bytes;
use omp_core::SmolStr;

use crate::{BlobPart, Cost, Props, Unsupported, Usage};

/// Portable image or video frame aspect ratios.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AspectRatio {
	/// Square output.
	Square,
	/// Widescreen landscape output.
	Wide16x9,
	/// Widescreen portrait output.
	Tall9x16,
	/// Standard landscape output.
	Landscape4x3,
	/// Standard portrait output.
	Portrait3x4,
	/// Photographic landscape output.
	Landscape3x2,
	/// Photographic portrait output.
	Portrait2x3,
	/// Ultrawide cinematic output.
	Ultrawide21x9,
}

/// Explicit image pixel dimensions.
#[non_exhaustive]
#[derive(Builder, Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImageSize {
	/// Pixel width.
	pub width:  u32,
	/// Pixel height.
	pub height: u32,
}

/// Provider-independent image quality tiers.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ImageQuality {
	/// Cost- and latency-oriented quality.
	Low,
	/// Balanced quality.
	Medium,
	/// Highest portable quality tier.
	High,
}

/// Encodings for generated images.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ImageFormat {
	/// PNG image.
	Png,
	/// WebP image.
	Webp,
	/// JPEG image.
	Jpeg,
	/// SVG vector image.
	Svg,
}

/// Requested image background treatment.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ImageBackground {
	/// Fully rendered opaque background.
	Opaque,
	/// Transparent background where the provider supports alpha.
	Transparent,
}

/// Final-prompt image generation or editing request.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, PartialEq)]
pub struct GenerateImageRequest {
	/// Catalog model used for generation.
	pub model:        SmolStr,
	/// Fully assembled prompt; prompt engineering remains tool-level policy.
	pub prompt:       SmolStr,
	/// Images requested; zero preserves the protocol's default of one.
	pub n:            u32,
	/// Desired ratio when explicit dimensions are absent.
	pub aspect_ratio: Option<AspectRatio>,
	/// Explicit dimensions, taking precedence over the ratio where supported.
	pub size:         Option<ImageSize>,
	/// Optional quality tier.
	pub quality:      Option<ImageQuality>,
	/// Optional output encoding.
	pub format:       Option<ImageFormat>,
	/// Optional background treatment.
	pub background:   Option<ImageBackground>,
	/// Lossy compression quality from 0 through 100.
	pub compression:  Option<u32>,
	/// Deterministic provider seed where supported.
	pub seed:         Option<u64>,
	/// Edit, image-to-image, and style references that select provider edit
	/// endpoints.
	pub input_images: Vec<BlobPart>,
	/// Namespaced provider-specific controls.
	pub props:        Props,
}

/// Streamed image-generation progress or terminal output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum ImageEvent {
	/// A low-resolution progressive preview.
	Partial(ImagePartial),
	/// Terminal durable image outputs and accounting.
	Done(ImageDone),
}

/// One progressive image preview.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, PartialEq)]
pub struct ImagePartial {
	/// Index of the final image this preview refines.
	pub index:   u32,
	/// Low-resolution content-addressed preview.
	pub preview: BlobPart,
}

/// Terminal image-generation result.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, PartialEq)]
pub struct ImageDone {
	/// Durable generated images ingested into the blob store.
	pub images:         Vec<BlobPart>,
	/// Provider-rewritten prompt when reported.
	pub revised_prompt: SmolStr,
	/// Commentary interleaved by chat-shaped image backends.
	pub text:           SmolStr,
	/// Provider usage when available.
	pub usage:          Option<Usage>,
	/// Metered cost when available.
	pub cost:           Option<Cost>,
	/// Controls changed or omitted by the provider path.
	pub unsupported:    Vec<Unsupported>,
	/// Namespaced result metadata.
	pub props:          Props,
}

/// Encodings for synthesized speech.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AudioEncoding {
	/// MPEG Layer III audio.
	Mp3,
	/// Headerless signed 16-bit PCM samples.
	Pcm16,
	/// WAV container.
	Wav,
	/// Opus audio.
	Opus,
	/// AAC audio.
	Aac,
	/// FLAC audio.
	Flac,
}

/// Stateless voice-cloning reference.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, PartialEq)]
pub struct VoiceClone {
	/// Reference voice sample.
	pub reference:  BlobPart,
	/// Known transcript of the sample, improving clone alignment where
	/// supported.
	pub transcript: SmolStr,
}

/// Speech-synthesis request.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, PartialEq)]
pub struct SpeakRequest {
	/// Catalog model used for synthesis.
	pub model:          SmolStr,
	/// Text to speak.
	pub text:           SmolStr,
	/// Provider voice id.
	pub voice:          SmolStr,
	/// Required output codec.
	pub encoding:       AudioEncoding,
	/// Explicit sample rate; absence preserves the provider or codec default.
	pub sample_rate_hz: Option<u32>,
	/// Playback speed multiplier; absence means normal speed.
	pub speed:          Option<f64>,
	/// Expressive tonal or style direction.
	pub instructions:   SmolStr,
	/// Optional stateless voice-cloning input.
	pub clone:          Option<VoiceClone>,
	/// Namespaced synthesis controls.
	pub props:          Props,
}

/// Streamed speech audio or terminal durable result.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum SpeakEvent {
	/// Encoded bytes in playback order.
	Chunk(SpeakChunk),
	/// Terminal durable utterance and accounting.
	Done(SpeakDone),
}

/// One playback-ordered encoded speech fragment.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, Eq, PartialEq)]
pub struct SpeakChunk {
	/// Raw codec bytes; framing follows the requested encoding and sample rate.
	pub audio:            Bytes,
	/// Transcript alignment delta when reported by the provider.
	pub transcript_delta: SmolStr,
}

/// Terminal speech-synthesis result.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, PartialEq)]
pub struct SpeakDone {
	/// Full utterance ingested into the blob store.
	pub audio:       BlobPart,
	/// Playback duration.
	pub duration_ms: u64,
	/// Provider usage when available.
	pub usage:       Option<Usage>,
	/// Metered cost when available.
	pub cost:        Option<Cost>,
	/// Controls changed or omitted by the provider path.
	pub unsupported: Vec<Unsupported>,
	/// Namespaced result metadata.
	pub props:       Props,
}

/// Requested transcription timestamp granularity.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TranscriptionGranularity {
	/// Phrase- or utterance-level segments.
	Segment,
	/// Individual word timing.
	Word,
}

/// Recorded-audio transcription request.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, PartialEq)]
pub struct TranscribeRequest {
	/// Catalog model used for transcription.
	pub model:         SmolStr,
	/// Recorded audio payload.
	pub audio:         BlobPart,
	/// ISO-639-1 language hint; empty requests auto-detection.
	pub language:      SmolStr,
	/// Vocabulary or domain hint used to steer spelling.
	pub prompt:        SmolStr,
	/// Translate the transcript to English.
	pub translate:     bool,
	/// Timestamp detail requested from the provider.
	pub granularities: Vec<TranscriptionGranularity>,
	/// Request speaker separation and labels.
	pub diarize:       bool,
	/// Optional decoding temperature.
	pub temperature:   Option<f64>,
	/// Namespaced transcription controls.
	pub props:         Props,
}

/// One timestamped transcription segment.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, PartialEq)]
pub struct TranscriptSegment {
	/// Inclusive start offset from the recording beginning.
	pub start_ms:   u64,
	/// Exclusive end offset from the recording beginning.
	pub end_ms:     u64,
	/// Segment text.
	pub text:       SmolStr,
	/// Diarization label when requested and supported.
	pub speaker:    Option<u32>,
	/// Provider confidence from zero through one when reported.
	pub confidence: Option<f64>,
}

/// One timestamped transcript word.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, Eq, PartialEq)]
pub struct TranscriptWord {
	/// Inclusive start offset from the recording beginning.
	pub start_ms: u64,
	/// Exclusive end offset from the recording beginning.
	pub end_ms:   u64,
	/// Recognized word.
	pub word:     SmolStr,
	/// Diarization label when requested and supported.
	pub speaker:  Option<u32>,
}

/// Complete transcription result and accounting.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, PartialEq)]
pub struct TranscribeResponse {
	/// Full transcript text.
	pub text:        SmolStr,
	/// Detected or confirmed ISO-639-1 language.
	pub language:    SmolStr,
	/// Recording duration.
	pub duration_ms: u64,
	/// Requested segment-level timings.
	pub segments:    Vec<TranscriptSegment>,
	/// Requested word-level timings.
	pub words:       Vec<TranscriptWord>,
	/// Provider usage when available.
	pub usage:       Option<Usage>,
	/// Metered cost when available.
	pub cost:        Option<Cost>,
	/// Controls changed or omitted by the provider path.
	pub unsupported: Vec<Unsupported>,
	/// Namespaced result metadata.
	pub props:       Props,
}

/// Portable output tiers for generated video.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum VideoResolution {
	/// 480p output.
	P480,
	/// 720p output.
	P720,
	/// 1080p output.
	P1080,
	/// 4K output.
	K4,
}

/// Asynchronous video-generation submission.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, PartialEq)]
pub struct GenerateVideoRequest {
	/// Catalog model used for generation.
	pub model:            SmolStr,
	/// Fully assembled video prompt.
	pub prompt:           SmolStr,
	/// Requested duration; provider clamping is reported as unsupported detail.
	pub duration_seconds: Option<u32>,
	/// Desired frame aspect ratio.
	pub aspect_ratio:     Option<AspectRatio>,
	/// Desired output resolution.
	pub resolution:       Option<VideoResolution>,
	/// Deterministic provider seed where supported.
	pub seed:             Option<u64>,
	/// Whether to synthesize audio; absence preserves the provider default.
	pub audio:            Option<bool>,
	/// Optional first frame for image-to-video generation.
	pub start_frame:      Option<BlobPart>,
	/// Optional terminal frame constraint.
	pub end_frame:        Option<BlobPart>,
	/// Subject and style guidance references.
	pub references:       Vec<BlobPart>,
	/// Namespaced video controls.
	pub props:            Props,
}

/// Lifecycle state of an asynchronous generation.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GenerationState {
	/// Accepted and waiting for provider capacity.
	Queued,
	/// Provider render is in progress.
	Running,
	/// Durable artifacts are available.
	Completed,
	/// Rendering ended in a classified failure.
	Failed,
	/// Explicit cancellation completed.
	Cancelled,
}

/// One durable or provider-linked generation output.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, PartialEq)]
pub struct GenerationArtifact {
	/// Durable blob-store output when ingestion has completed.
	pub blob:              Option<BlobPart>,
	/// Output role such as video, thumbnail, or spritesheet.
	pub variant:           SmolStr,
	/// Provider passthrough URL for clients that need the expiring original.
	pub url:               SmolStr,
	/// Expiration of the provider URL in Unix milliseconds.
	pub url_expires_at_ms: u64,
}

/// Reconnectable lifecycle snapshot for a long-running generation.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, PartialEq)]
pub struct GenerationStatus {
	/// Gateway-scoped identity stable across client reconnects.
	pub generation_id:    SmolStr,
	/// Current lifecycle state.
	pub state:            GenerationState,
	/// Provider progress from zero through one hundred where available.
	pub progress_percent: f64,
	/// Classified failure detail in the failed state.
	pub detail:           SmolStr,
	/// Outputs already ingested or linked by the gateway.
	pub artifacts:        Vec<GenerationArtifact>,
	/// Provider usage when available.
	pub usage:            Option<Usage>,
	/// Metered cost when available.
	pub cost:             Option<Cost>,
	/// Controls changed or omitted by the provider path.
	pub unsupported:      Vec<Unsupported>,
	/// Submission timestamp in Unix milliseconds.
	pub created_at_ms:    u64,
	/// Last lifecycle update in Unix milliseconds.
	pub updated_at_ms:    u64,
	/// Namespaced lifecycle metadata.
	pub props:            Props,
}
