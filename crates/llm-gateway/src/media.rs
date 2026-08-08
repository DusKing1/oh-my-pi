//! Durable gateway orchestration for image, speech, transcription, and video
//! facets.
//!
//! Every inline media value is ingested into the content-addressed blob store
//! and replaced with a hash reference. Hash-only inputs are resolved before
//! provider dispatch. This is what makes media reusable across turns without
//! re-uploading it.

use std::{
	sync::Arc,
	time::{Duration, SystemTime, UNIX_EPOCH},
};

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use futures::{StreamExt, stream::BoxStream};
use omp_core::SmolStr;
use omp_llm_types::{
	BlobPart, GenerateImageRequest, GenerateVideoRequest, GenerationArtifact, GenerationState,
	GenerationStatus, ImageEvent, SpeakEvent, SpeakRequest, TranscribeRequest, TranscribeResponse,
	facet::{Error as FacetError, Facets, GenerationHandle, ImageGen, Speak, Transcribe, VideoGen},
};
use omp_storage::blob::{BlobRef, BlobStore};
use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use tokio::sync::watch;

const POLL_INTERVAL: Duration = Duration::from_secs(2);
const URL_PASSTHROUGH_NAMESPACE: &str = "omp";
const URL_PASSTHROUGH_NAME: &str = "url_passthrough";

/// Downloads an expiring provider artifact before it is announced as complete.
#[async_trait]
pub trait ArtifactDownloader: Send + Sync {
	/// Downloads the complete body at `url`.
	async fn download(&self, url: &str) -> Result<Bytes, FacetError>;
}

/// Downloader used when URL-backed video is not configured at the gateway.
#[derive(Debug, Default)]
pub struct RejectingDownloader;

#[async_trait]
impl ArtifactDownloader for RejectingDownloader {
	async fn download(&self, url: &str) -> Result<Bytes, FacetError> {
		Err(FacetError::Transport(format!("no artifact downloader configured for {url}").into()))
	}
}

struct Job {
	provider_handle:  GenerationHandle,
	passthrough_urls: bool,
	status:           watch::Sender<GenerationStatus>,
}

/// Gateway media facets backed by a shared content-addressed store.
///
/// Video jobs are owned by this value rather than an attach stream.
/// Consequently, dropping every observer never stops a render; only
/// [`Self::cancel_generation`] requests cancellation.
#[derive(Clone)]
pub struct MediaFacets {
	store:         Arc<BlobStore>,
	image:         Option<Arc<dyn ImageGen>>,
	speak:         Option<Arc<dyn Speak>>,
	transcribe:    Option<Arc<dyn Transcribe>>,
	video:         Option<Arc<dyn VideoGen>>,
	downloader:    Arc<dyn ArtifactDownloader>,
	jobs:          Arc<RwLock<FxHashMap<SmolStr, Arc<Job>>>>,
	poll_interval: Duration,
}

impl MediaFacets {
	/// Constructs media orchestration around provider facet implementations.
	#[must_use]
	pub fn new(
		store: Arc<BlobStore>,
		image: Arc<dyn ImageGen>,
		speak: Arc<dyn Speak>,
		transcribe: Arc<dyn Transcribe>,
		video: Arc<dyn VideoGen>,
		downloader: Arc<dyn ArtifactDownloader>,
	) -> Self {
		Self {
			store,
			image: Some(image),
			speak: Some(speak),
			transcribe: Some(transcribe),
			video: Some(video),
			downloader,
			jobs: Arc::new(RwLock::new(FxHashMap::default())),
			poll_interval: POLL_INTERVAL,
		}
	}

	/// Constructs orchestration for exactly the media facets enabled in a shared
	/// production facet registry.
	#[must_use]
	pub fn from_facets(
		store: Arc<BlobStore>,
		facets: &Facets,
		downloader: Arc<dyn ArtifactDownloader>,
	) -> Self {
		Self {
			store,
			image: facets.image_gen.clone(),
			speak: facets.speak.clone(),
			transcribe: facets.transcribe.clone(),
			video: facets.video_gen.clone(),
			downloader,
			jobs: Arc::new(RwLock::new(FxHashMap::default())),
			poll_interval: POLL_INTERVAL,
		}
	}

	/// Resolves a hash input or ingests inline bytes, returning a hash-only
	/// part.
	///
	/// This canonicalization makes the part reusable in later turns without
	/// uploading its payload again.
	pub fn ingest_blob(&self, part: BlobPart) -> Result<BlobPart, FacetError> {
		ingest_blob(&self.store, part)
	}

	/// Generates images, preserving provider partial order and making every
	/// preview and terminal image durable before forwarding it.
	pub async fn generate_image(
		&self,
		mut request: GenerateImageRequest,
	) -> Result<BoxStream<'static, ImageEvent>, FacetError> {
		request.input_images = request
			.input_images
			.into_iter()
			.map(|part| resolve_blob(&self.store, part))
			.collect::<Result<_, _>>()?;
		let image = self.image.as_ref().ok_or_else(unsupported_media)?;
		let mut upstream = image.generate(request).await?;
		let store = Arc::clone(&self.store);
		Ok(Box::pin(async_stream::stream! {
			while let Some(event) = upstream.next().await {
				let normalized = match event {
					ImageEvent::Partial(mut partial) => {
						match ingest_blob(&store, partial.preview) {
							Ok(preview) => partial.preview = preview,
							Err(_) => break,
						}
						ImageEvent::Partial(partial)
					},
					ImageEvent::Done(mut done) => {
						let images = done.images.into_iter()
							.map(|part| ingest_blob(&store, part))
							.collect::<Result<Vec<_>, _>>();
						match images {
							Ok(images) => done.images = images,
							Err(_) => break,
						}
						ImageEvent::Done(done)
					},
					_ => continue,
				};
				yield normalized;
			}
		}))
	}

	/// Synthesizes ordered chunks and emits a terminal blob containing the full
	/// concatenated utterance.
	pub async fn speak(
		&self,
		mut request: SpeakRequest,
	) -> Result<BoxStream<'static, SpeakEvent>, FacetError> {
		if let Some(clone) = request.clone.as_mut() {
			clone.reference = self.ingest_blob(clone.reference.clone())?;
		}
		let speak = self.speak.as_ref().ok_or_else(unsupported_media)?;
		let mut upstream = speak.speak(request).await?;
		let store = Arc::clone(&self.store);
		Ok(Box::pin(async_stream::stream! {
			let mut utterance = BytesMut::new();
			while let Some(event) = upstream.next().await {
				match event {
					SpeakEvent::Chunk(chunk) => {
						utterance.extend_from_slice(&chunk.audio);
						yield SpeakEvent::Chunk(chunk);
					},
					SpeakEvent::Done(mut done) => {
						let audio = if utterance.is_empty() {
							ingest_blob(&store, done.audio)
						} else {
							ingest_blob(&store, BlobPart::builder()
								.hash([0; 32])
								.mime(done.audio.mime.clone())
								.size(utterance.len() as u64)
								.inline(utterance.split().freeze())
								.build())
						};
						match audio {
							Ok(audio) => done.audio = audio,
							Err(_) => break,
						}
						yield SpeakEvent::Done(done);
						break;
					},
					_ => {},
				}
			}
		}))
	}

	/// Transcribes a complete recording after resolving it from the blob store.
	pub async fn transcribe(
		&self,
		mut request: TranscribeRequest,
	) -> Result<TranscribeResponse, FacetError> {
		request.audio = self.ingest_blob(request.audio)?;
		let transcribe = self.transcribe.as_ref().ok_or_else(unsupported_media)?;
		transcribe.transcribe(request).await
	}

	/// Submits a video render and immediately returns a gateway-scoped queued
	/// snapshot. Provider identifiers remain private to the job registry.
	///
	/// Setting the `omp/url_passthrough` boolean property preserves expiring
	/// provider URLs instead of downloading them; otherwise every output variant
	/// crosses the blob-store durability barrier before completion is published.
	pub async fn generate_video(
		&self,
		mut request: GenerateVideoRequest,
	) -> Result<GenerationStatus, FacetError> {
		if let Some(part) = request.start_frame.take() {
			request.start_frame = Some(self.ingest_blob(part)?);
		}
		if let Some(part) = request.end_frame.take() {
			request.end_frame = Some(self.ingest_blob(part)?);
		}
		request.references = request
			.references
			.into_iter()
			.map(|part| self.ingest_blob(part))
			.collect::<Result<_, _>>()?;
		let passthrough_urls = request
			.props
			.get_ns(URL_PASSTHROUGH_NAMESPACE, URL_PASSTHROUGH_NAME)
			.and_then(serde_json::Value::as_bool)
			.unwrap_or(false);
		let video = self.video.as_ref().ok_or_else(unsupported_media)?;
		let provider_handle = video.submit(request).await?;
		let generation_id: SmolStr = ulid::Ulid::generate().to_string().into();
		let now = now_ms();
		let queued = GenerationStatus::builder()
			.generation_id(generation_id.clone())
			.state(GenerationState::Queued)
			.progress_percent(0.0)
			.detail(SmolStr::default())
			.artifacts(Vec::new())
			.unsupported(Vec::new())
			.created_at_ms(now)
			.updated_at_ms(now)
			.props(Default::default())
			.build();
		let (status, _) = watch::channel(queued.clone());
		let job = Arc::new(Job { provider_handle, passthrough_urls, status });
		self.jobs.write().insert(generation_id, Arc::clone(&job));
		let this = self.clone();
		tokio::spawn(async move { this.drive_job(job).await });
		Ok(queued)
	}

	/// Returns the latest snapshot for a gateway generation id.
	pub fn get_generation(&self, generation_id: &str) -> Result<GenerationStatus, FacetError> {
		let job = self.job(generation_id)?;
		let status = job.status.borrow().clone();
		Ok(status)
	}

	/// Streams the current snapshot and every later change. Dropping this stream
	/// only detaches the observer and never cancels the provider render.
	pub fn attach_generation(
		&self,
		generation_id: &str,
	) -> Result<BoxStream<'static, GenerationStatus>, FacetError> {
		let job = self.job(generation_id)?;
		let mut receiver = job.status.subscribe();
		Ok(Box::pin(async_stream::stream! {
			let initial = receiver.borrow().clone();
			let terminal = is_terminal(initial.state);
			yield initial;
			if !terminal {
				loop {
					if receiver.changed().await.is_err() {
						break;
					}
					let status = receiver.borrow().clone();
					let terminal = is_terminal(status.state);
					yield status;
					if terminal {
						break;
					}
				}
			}
		}))
	}

	/// Explicitly cancels a render and publishes the provider's cancelled state.
	pub async fn cancel_generation(
		&self,
		generation_id: &str,
	) -> Result<GenerationStatus, FacetError> {
		let job = self.job(generation_id)?;
		let current = job.status.borrow().clone();
		if is_terminal(current.state) {
			return Ok(current);
		}
		let video = self.video.as_ref().ok_or_else(unsupported_media)?;
		let mut status = video.cancel(job.provider_handle.clone()).await?;
		status.generation_id = generation_id.into();
		status.state = GenerationState::Cancelled;
		status.updated_at_ms = now_ms();
		job.status.send_replace(status.clone());
		Ok(status)
	}

	/// Applies a provider webhook snapshot to a live job. The same durable
	/// artifact barrier used by polling runs before a completed state is
	/// visible.
	pub async fn receive_webhook(
		&self,
		generation_id: &str,
		status: GenerationStatus,
	) -> Result<GenerationStatus, FacetError> {
		let job = self.job(generation_id)?;
		self.publish_status(&job, status).await
	}

	fn job(&self, generation_id: &str) -> Result<Arc<Job>, FacetError> {
		self
			.jobs
			.read()
			.get(generation_id)
			.cloned()
			.ok_or_else(|| FacetError::Provider(format!("unknown generation {generation_id}").into()))
	}

	async fn drive_job(&self, job: Arc<Job>) {
		let Some(video) = self.video.as_ref() else {
			fail_job(&job, unsupported_media());
			return;
		};
		let attached = video.attach(job.provider_handle.clone()).await;
		if let Ok(mut statuses) = attached {
			while let Some(status) = statuses.next().await {
				match self.publish_status(&job, status).await {
					Ok(status) if is_terminal(status.state) => return,
					Ok(_) => {},
					Err(error) => {
						fail_job(&job, error);
						return;
					},
				}
			}
		}
		loop {
			if is_terminal(job.status.borrow().state) {
				return;
			}
			tokio::time::sleep(self.poll_interval).await;
			match video.get(job.provider_handle.clone()).await {
				Ok(status) => match self.publish_status(&job, status).await {
					Ok(status) if is_terminal(status.state) => return,
					Ok(_) => {},
					Err(error) => {
						fail_job(&job, error);
						return;
					},
				},
				Err(error) => {
					fail_job(&job, error);
					return;
				},
			}
		}
	}

	async fn publish_status(
		&self,
		job: &Job,
		mut status: GenerationStatus,
	) -> Result<GenerationStatus, FacetError> {
		let current = job.status.borrow().clone();
		if is_terminal(current.state) {
			return Ok(current);
		}
		status.generation_id = current.generation_id.clone();
		status.created_at_ms = current.created_at_ms;
		status.updated_at_ms = now_ms();
		if status.state == GenerationState::Completed {
			for artifact in &mut status.artifacts {
				self.ingest_artifact(artifact, job.passthrough_urls).await?;
			}
		}
		let latest = job.status.borrow().clone();
		if is_terminal(latest.state) {
			return Ok(latest);
		}
		job.status.send_replace(status.clone());
		Ok(status)
	}

	async fn ingest_artifact(
		&self,
		artifact: &mut GenerationArtifact,
		passthrough_url: bool,
	) -> Result<(), FacetError> {
		if let Some(blob) = artifact.blob.take() {
			artifact.blob = Some(self.ingest_blob(blob)?);
			return Ok(());
		}
		if passthrough_url {
			return Ok(());
		}
		if artifact.url.is_empty() {
			return Err(FacetError::Transport("completed video artifact has no payload".into()));
		}
		let bytes = self.downloader.download(artifact.url.as_str()).await?;
		artifact.blob = Some(
			self.ingest_blob(
				BlobPart::builder()
					.hash([0; 32])
					.mime(SmolStr::new_static("application/octet-stream"))
					.size(bytes.len() as u64)
					.inline(bytes)
					.build(),
			)?,
		);
		Ok(())
	}
}

fn unsupported_media() -> FacetError {
	FacetError::Unsupported(Vec::new())
}

fn ingest_blob(store: &BlobStore, part: BlobPart) -> Result<BlobPart, FacetError> {
	let mime = part.mime;
	let reference = if part.inline.is_empty() {
		let reference = BlobRef { hash: part.hash, size: part.size };
		store.get(&reference).map_err(storage_error)?;
		reference
	} else {
		store.put(&part.inline).map_err(storage_error)?
	};
	Ok(BlobPart::builder()
		.hash(reference.hash)
		.mime(mime)
		.size(reference.size)
		.inline(Bytes::new())
		.build())
}

fn resolve_blob(store: &BlobStore, part: BlobPart) -> Result<BlobPart, FacetError> {
	let mime = part.mime;
	let (reference, inline) = if part.inline.is_empty() {
		let reference = BlobRef { hash: part.hash, size: part.size };
		let inline = store.get(&reference).map_err(storage_error)?;
		(reference, inline)
	} else {
		let reference = store.put(&part.inline).map_err(storage_error)?;
		(reference, part.inline)
	};
	Ok(BlobPart::builder()
		.hash(reference.hash)
		.mime(mime)
		.size(reference.size)
		.inline(inline)
		.build())
}

fn fail_job(job: &Job, error: FacetError) {
	let mut failed = job.status.borrow().clone();
	failed.state = GenerationState::Failed;
	failed.detail = public_failure_detail(&error);
	failed.updated_at_ms = now_ms();
	job.status.send_replace(failed);
}

fn public_failure_detail(_error: &FacetError) -> SmolStr {
	SmolStr::new_static("media generation failed")
}

fn storage_error(error: omp_storage::blob::Error) -> FacetError {
	FacetError::Transport(format!("blob store: {error}").into())
}

const fn is_terminal(state: GenerationState) -> bool {
	matches!(
		state,
		GenerationState::Completed | GenerationState::Failed | GenerationState::Cancelled
	)
}

fn now_ms() -> u64 {
	SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.unwrap_or_default()
		.as_millis()
		.try_into()
		.unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
	use std::sync::{
		Arc,
		atomic::{AtomicUsize, Ordering},
	};

	use async_trait::async_trait;
	use bytes::Bytes;
	use futures::{
		StreamExt,
		stream::{self, BoxStream},
	};
	use omp_llm_types::{
		AudioEncoding, GenerateImageRequest, GenerateVideoRequest, GenerationArtifact,
		GenerationState, GenerationStatus, ImageDone, ImageEvent, ImagePartial, SpeakChunk,
		SpeakDone, SpeakEvent, SpeakRequest, TranscribeRequest, TranscribeResponse,
		facet::{GenerationHandle, ImageGen, Speak, Transcribe, VideoGen},
	};
	use parking_lot::Mutex;

	use super::*;

	#[derive(Default)]
	struct FakeProvider {
		image_events: Mutex<Option<Vec<ImageEvent>>>,
		speak_events: Mutex<Option<Vec<SpeakEvent>>>,
		cancelled:    AtomicUsize,
	}

	#[async_trait]
	impl ImageGen for FakeProvider {
		async fn generate(
			&self,
			_request: GenerateImageRequest,
		) -> Result<BoxStream<'static, ImageEvent>, FacetError> {
			Ok(Box::pin(stream::iter(self.image_events.lock().take().unwrap_or_default())))
		}
	}

	#[async_trait]
	impl Speak for FakeProvider {
		async fn speak(
			&self,
			_request: SpeakRequest,
		) -> Result<BoxStream<'static, SpeakEvent>, FacetError> {
			Ok(Box::pin(stream::iter(self.speak_events.lock().take().unwrap_or_default())))
		}
	}

	#[async_trait]
	impl Transcribe for FakeProvider {
		async fn transcribe(
			&self,
			_request: TranscribeRequest,
		) -> Result<TranscribeResponse, FacetError> {
			Ok(TranscribeResponse::builder()
				.text("transcript".into())
				.language("en".into())
				.duration_ms(10)
				.segments(Vec::new())
				.words(Vec::new())
				.unsupported(Vec::new())
				.props(Default::default())
				.build())
		}
	}

	#[async_trait]
	impl VideoGen for FakeProvider {
		async fn submit(
			&self,
			_request: GenerateVideoRequest,
		) -> Result<GenerationHandle, FacetError> {
			Ok(GenerationHandle::builder()
				.id("private-upstream-id".into())
				.build())
		}

		async fn get(&self, _handle: GenerationHandle) -> Result<GenerationStatus, FacetError> {
			Ok(status("private-upstream-id", GenerationState::Running, Vec::new()))
		}

		async fn attach(
			&self,
			_handle: GenerationHandle,
		) -> Result<BoxStream<'static, GenerationStatus>, FacetError> {
			Ok(Box::pin(stream::pending()))
		}

		async fn cancel(&self, _handle: GenerationHandle) -> Result<GenerationStatus, FacetError> {
			self.cancelled.fetch_add(1, Ordering::Relaxed);
			Ok(status("private-upstream-id", GenerationState::Cancelled, Vec::new()))
		}
	}

	struct FakeDownloader(Bytes);

	#[async_trait]
	impl ArtifactDownloader for FakeDownloader {
		async fn download(&self, _url: &str) -> Result<Bytes, FacetError> {
			Ok(self.0.clone())
		}
	}

	fn blob(bytes: &'static [u8], mime: &str) -> BlobPart {
		BlobPart::builder()
			.hash([0; 32])
			.mime(mime.into())
			.size(bytes.len() as u64)
			.inline(Bytes::from_static(bytes))
			.build()
	}

	fn status(
		id: &str,
		state: GenerationState,
		artifacts: Vec<GenerationArtifact>,
	) -> GenerationStatus {
		GenerationStatus::builder()
			.generation_id(id.into())
			.state(state)
			.progress_percent(if state == GenerationState::Completed {
				100.0
			} else {
				0.0
			})
			.detail("".into())
			.artifacts(artifacts)
			.unsupported(Vec::new())
			.created_at_ms(1)
			.updated_at_ms(1)
			.props(Default::default())
			.build()
	}

	fn image_request() -> GenerateImageRequest {
		GenerateImageRequest::builder()
			.model("image-model".into())
			.prompt("draw".into())
			.n(1)
			.input_images(Vec::new())
			.props(Default::default())
			.build()
	}

	fn speak_request() -> SpeakRequest {
		SpeakRequest::builder()
			.model("speech-model".into())
			.text("hello".into())
			.voice("voice".into())
			.encoding(AudioEncoding::Mp3)
			.instructions("".into())
			.props(Default::default())
			.build()
	}

	fn video_request() -> GenerateVideoRequest {
		GenerateVideoRequest::builder()
			.model("video-model".into())
			.prompt("move".into())
			.references(Vec::new())
			.props(Default::default())
			.build()
	}

	fn gateway(
		provider: Arc<FakeProvider>,
		downloader: Arc<dyn ArtifactDownloader>,
	) -> (MediaFacets, tempfile::TempDir, Arc<BlobStore>) {
		let temporary = tempfile::tempdir().expect("temporary directory");
		let store = Arc::new(BlobStore::open(temporary.path()).expect("blob store"));
		let facets = MediaFacets::new(
			Arc::clone(&store),
			provider.clone(),
			provider.clone(),
			provider.clone(),
			provider,
			downloader,
		);
		(facets, temporary, store)
	}

	#[test]
	fn inline_blob_is_ingested_and_replaced_by_hash_reference() {
		let temporary = tempfile::tempdir().expect("temporary directory");
		let store = BlobStore::open(temporary.path()).expect("blob store");
		let normalized = ingest_blob(&store, blob(b"payload", "image/png")).expect("ingest");
		assert!(normalized.inline.is_empty());
		assert_ne!(normalized.hash, [0; 32]);
		assert!(store.has(&BlobRef { hash: normalized.hash, size: normalized.size }));
	}

	#[tokio::test]
	async fn image_partials_are_forwarded_before_blob_backed_done() {
		let provider = Arc::new(FakeProvider::default());
		let _ = provider.image_events.lock().replace(vec![
			ImageEvent::Partial(
				ImagePartial::builder()
					.index(0)
					.preview(blob(b"p", "image/png"))
					.build(),
			),
			ImageEvent::Done(
				ImageDone::builder()
					.images(vec![blob(b"final", "image/png")])
					.revised_prompt("".into())
					.text("".into())
					.unsupported(Vec::new())
					.props(Default::default())
					.build(),
			),
		]);
		let (gateway, _temporary, _) = gateway(provider, Arc::new(RejectingDownloader));
		let events: Vec<_> = gateway
			.generate_image(image_request())
			.await
			.expect("generate")
			.collect()
			.await;
		assert!(matches!(events.as_slice(), [ImageEvent::Partial(_), ImageEvent::Done(_)]));
		match &events[1] {
			ImageEvent::Done(done) => assert!(done.images[0].inline.is_empty()),
			_ => unreachable!(),
		}
	}

	#[tokio::test]
	async fn speak_chunks_remain_in_playback_order() {
		let provider = Arc::new(FakeProvider::default());
		let _ = provider.speak_events.lock().replace(vec![
			SpeakEvent::Chunk(
				SpeakChunk::builder()
					.audio(Bytes::from_static(b"a"))
					.transcript_delta("".into())
					.build(),
			),
			SpeakEvent::Chunk(
				SpeakChunk::builder()
					.audio(Bytes::from_static(b"b"))
					.transcript_delta("".into())
					.build(),
			),
			SpeakEvent::Done(
				SpeakDone::builder()
					.audio(blob(b"ignored", "audio/mpeg"))
					.duration_ms(2)
					.unsupported(Vec::new())
					.props(Default::default())
					.build(),
			),
		]);
		let (gateway, _temporary, store) = gateway(provider, Arc::new(RejectingDownloader));
		let events: Vec<_> = gateway
			.speak(speak_request())
			.await
			.expect("speak")
			.collect()
			.await;
		assert_eq!(events.len(), 3);
		let SpeakEvent::Done(done) = &events[2] else {
			panic!("missing done")
		};
		let stored = store
			.get(&BlobRef { hash: done.audio.hash, size: done.audio.size })
			.expect("audio");
		assert_eq!(stored, Bytes::from_static(b"ab"));
	}

	#[tokio::test]
	async fn video_submit_returns_queued_gateway_id() {
		let provider = Arc::new(FakeProvider::default());
		let (gateway, _temporary, _) = gateway(provider, Arc::new(RejectingDownloader));
		let queued = gateway
			.generate_video(video_request())
			.await
			.expect("submit");
		assert_eq!(queued.state, GenerationState::Queued);
		assert!(!queued.generation_id.is_empty());
		assert_ne!(queued.generation_id.as_str(), "private-upstream-id");
	}

	#[tokio::test]
	async fn dropped_attach_does_not_cancel_and_reattach_observes_completion() {
		let provider = Arc::new(FakeProvider::default());
		let (gateway, _temporary, _) = gateway(provider.clone(), Arc::new(RejectingDownloader));
		let queued = gateway
			.generate_video(video_request())
			.await
			.expect("submit");
		drop(
			gateway
				.attach_generation(&queued.generation_id)
				.expect("attach"),
		);
		gateway
			.receive_webhook(
				&queued.generation_id,
				status("upstream", GenerationState::Completed, vec![
					GenerationArtifact::builder()
						.blob(blob(b"video", "video/mp4"))
						.variant("video".into())
						.url("".into())
						.url_expires_at_ms(0)
						.build(),
				]),
			)
			.await
			.expect("webhook");
		let observed = gateway
			.attach_generation(&queued.generation_id)
			.expect("reattach")
			.next()
			.await
			.expect("status");
		assert_eq!(observed.state, GenerationState::Completed);
		assert_eq!(provider.cancelled.load(Ordering::Relaxed), 0);
	}

	#[tokio::test]
	async fn url_artifact_is_stored_before_completed_is_reported() {
		let provider = Arc::new(FakeProvider::default());
		let (gateway, _temporary, store) =
			gateway(provider, Arc::new(FakeDownloader(Bytes::from_static(b"downloaded"))));
		let queued = gateway
			.generate_video(video_request())
			.await
			.expect("submit");
		let completed = gateway
			.receive_webhook(
				&queued.generation_id,
				status("upstream", GenerationState::Completed, vec![
					GenerationArtifact::builder()
						.variant("video".into())
						.url("https://provider.invalid/expiring".into())
						.url_expires_at_ms(10)
						.build(),
				]),
			)
			.await
			.expect("webhook");
		let blob = completed.artifacts[0]
			.blob
			.as_ref()
			.expect("durable artifact");
		assert!(store.has(&BlobRef { hash: blob.hash, size: blob.size }));
		assert_eq!(
			gateway
				.get_generation(&queued.generation_id)
				.expect("snapshot")
				.state,
			GenerationState::Completed
		);
	}

	#[tokio::test]
	async fn cancel_transitions_job_to_cancelled() {
		let provider = Arc::new(FakeProvider::default());
		let (gateway, _temporary, _) = gateway(provider.clone(), Arc::new(RejectingDownloader));
		let queued = gateway
			.generate_video(video_request())
			.await
			.expect("submit");
		let cancelled = gateway
			.cancel_generation(&queued.generation_id)
			.await
			.expect("cancel");
		assert_eq!(cancelled.state, GenerationState::Cancelled);
		assert_eq!(
			gateway
				.get_generation(&queued.generation_id)
				.expect("snapshot")
				.state,
			GenerationState::Cancelled
		);
		assert_eq!(provider.cancelled.load(Ordering::Relaxed), 1);
	}
	#[test]
	fn failed_generation_snapshot_redacts_provider_diagnostics() {
		const CANARY: &str = "canary-video-api-key-in-provider-error";
		let error = FacetError::Provider(CANARY.into());
		let detail = public_failure_detail(&error);
		assert_eq!(detail, "media generation failed");
		assert!(!detail.contains(CANARY));
	}
}
