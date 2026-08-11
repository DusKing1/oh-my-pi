//! Durable production adapter for `OpenAI`'s asynchronous Sora video API.
//!
//! Provider job ids and non-secret credential identities are persisted beside
//! the blob store. Completed bytes cross the blob-store durability barrier
//! before a terminal status is returned, so polling and reattachment continue
//! to work after daemon restart without retaining expiring provider URLs.

use std::{
	fmt,
	marker::PhantomData,
	path::{Path, PathBuf},
	sync::Arc,
	time::{Duration, SystemTime, UNIX_EPOCH},
};

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::BoxStream;
use http::{Method, Request, Response, StatusCode, header};
use http_body_util::{BodyExt, Full};
use omp_core::Str;
use omp_llm_egress::{
	auth_inject::{AuthContext, CredentialLease},
	client::Body,
};
use omp_llm_types::{
	Accuracy, AspectRatio, BlobPart, Cost, GenerateVideoRequest, GenerationArtifact,
	GenerationState, GenerationStatus, Props, Unsupported, UnsupportedAction, Usage,
	VideoResolution,
	facet::{self, GenerationHandle, VideoGen},
};
use omp_storage::blob::{BlobRef, BlobStore};
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::{fs, sync::Mutex as AsyncMutex};
use tower::{Service, ServiceExt};

const PROVIDER: &str = "openai";
const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_POLL_INTERVAL: Duration = Duration::from_secs(2);

/// Non-secret credential selection used to bind every job to the account that
/// submitted it. Implementations redeem no secrets here; the egress auth layer
/// receives the returned canonical lease on each request.
pub trait VideoCredentialLeases: Send + Sync {
	/// Selects the credential for a new `OpenAI` video job.
	fn select(&self) -> Result<CredentialLease, VideoError>;
	/// Refreshes the current generation of the credential that owns a job.
	fn by_id(&self, credential_id: u64) -> Result<CredentialLease, VideoError>;
}

/// Initialization failure for the durable video state directory.
#[derive(Debug, thiserror::Error)]
pub enum VideoInitError {
	/// The durable job directory could not be prepared.
	#[error("failed to prepare video job directory: {0}")]
	Io(#[from] std::io::Error),
}

/// Typed `OpenAI` video adapter failure.
#[derive(Debug, thiserror::Error)]
pub enum VideoError {
	/// A typed request control is not supported by the selected Sora path.
	#[error("invalid video control {control}: {detail}")]
	InvalidControl {
		/// Stable canonical control name.
		control: &'static str,
		/// Caller-safe reason.
		detail:  Str,
	},
	/// No usable credential exists, or a job's owning credential disappeared.
	#[error("video credential unavailable: {0}")]
	Credential(Str),
	/// Provider authentication was rejected.
	#[error("OpenAI video authentication failed")]
	Authentication,
	/// The provider does not know this job.
	#[error("OpenAI video job not found: {0}")]
	NotFound(Str),
	/// Provider returned a classified HTTP failure.
	#[error("OpenAI video request failed with HTTP {status}: {detail}")]
	Provider {
		/// Upstream HTTP status.
		status: u16,
		/// Caller-safe upstream detail.
		detail: Str,
	},
	/// Egress failed before a usable provider response was obtained.
	#[error("OpenAI video transport failed: {0}")]
	Transport(Str),
	/// Provider response violated the videos protocol.
	#[error("invalid OpenAI video response: {0}")]
	Protocol(Str),
	/// Durable job or blob persistence failed.
	#[error("video persistence failed: {0}")]
	Persistence(Str),
}

/// `OpenAI` Sora job adapter over the daemon-owned egress stack.
///
/// `S` is normally `AuthInject<BrokerCredentialSource, ...>`. Every request is
/// tagged with both `AuthContext("openai")` and the persisted credential lease,
/// preventing polling one account's opaque job id with another account's key.
pub struct OpenAiVideoBackend<S, B = hyper::body::Incoming> {
	service:       S,
	credentials:   Arc<dyn VideoCredentialLeases>,
	store:         Arc<BlobStore>,
	jobs_dir:      Arc<PathBuf>,
	base_url:      Arc<str>,
	poll_interval: Duration,
	locks:         Arc<Mutex<FxHashMap<Str, Arc<AsyncMutex<()>>>>>,
	response_body: PhantomData<fn() -> B>,
}

impl<S: Clone, B> Clone for OpenAiVideoBackend<S, B> {
	fn clone(&self) -> Self {
		Self {
			service:       self.service.clone(),
			credentials:   Arc::clone(&self.credentials),
			store:         Arc::clone(&self.store),
			jobs_dir:      Arc::clone(&self.jobs_dir),
			base_url:      Arc::clone(&self.base_url),
			poll_interval: self.poll_interval,
			locks:         Arc::clone(&self.locks),
			response_body: PhantomData,
		}
	}
}

impl<S, B> OpenAiVideoBackend<S, B> {
	/// Creates a durable Sora adapter.
	pub fn new(
		service: S,
		credentials: Arc<dyn VideoCredentialLeases>,
		store: Arc<BlobStore>,
		jobs_dir: impl AsRef<Path>,
		base_url: Option<&str>,
	) -> Result<Self, VideoInitError> {
		std::fs::create_dir_all(jobs_dir.as_ref())?;
		Ok(Self {
			service,
			credentials,
			store,
			jobs_dir: Arc::new(jobs_dir.as_ref().to_owned()),
			base_url: Arc::from(base_url.unwrap_or(DEFAULT_BASE_URL).trim_end_matches('/')),
			poll_interval: DEFAULT_POLL_INTERVAL,
			locks: Arc::new(Mutex::new(FxHashMap::default())),
			response_body: PhantomData,
		})
	}

	/// Overrides the attach polling interval. Intended for deployment tuning and
	/// deterministic local-provider tests.
	#[must_use]
	pub const fn with_poll_interval(mut self, interval: Duration) -> Self {
		self.poll_interval = interval;
		self
	}
}

impl<S, B> OpenAiVideoBackend<S, B>
where
	S: Service<Request<Body>, Response = Response<B>> + Clone + Send + Sync + 'static,
	S::Future: Send + 'static,
	S::Error: fmt::Display + Send + Sync + 'static,
	B: hyper::body::Body<Data = Bytes> + Send + Unpin + 'static,
	B::Error: fmt::Display + Send + Sync,
{
	/// Applies a trusted provider webhook payload to an existing durable job.
	/// Terminal jobs are immutable, so a late or duplicate webhook cannot create
	/// a second terminal transition.
	pub async fn receive_webhook(&self, payload: &[u8]) -> Result<GenerationStatus, VideoError> {
		let payload: WireWebhook = serde_json::from_slice(payload)
			.map_err(|error| VideoError::Protocol(error.to_string().into()))?;
		match payload {
			WireWebhook::Event(event) => {
				if !matches!(event.kind.as_str(), "video.completed" | "video.failed") {
					return Err(VideoError::Protocol("unsupported OpenAI video webhook type".into()));
				}
				validate_id(&event.data.id)?;
				let current = self.load_upstream(&event.data.id).await?;
				self
					.get_inner(GenerationHandle::builder().id(current.id).build())
					.await
			},
			WireWebhook::Status(wire) => {
				validate_id(&wire.id)?;
				let current = self.load_upstream(&wire.id).await?;
				let lock = self.job_lock(&current.id);
				let _guard = lock.lock().await;
				let current = self.load(&current.id).await?;
				if current.is_terminal() {
					return Ok(current.status());
				}
				let lease = self.credentials.by_id(current.credential_id)?;
				self.materialize(wire, lease, current).await
			},
		}
	}

	/// Removes durable local metadata for a terminal job. Blob bytes remain
	/// content-addressed and may still be referenced elsewhere.
	pub async fn cleanup(&self, handle: &GenerationHandle) -> Result<(), VideoError> {
		validate_id(&handle.id)?;
		let initial = self.load(&handle.id).await?;
		let lock = self.job_lock(&initial.id);
		let _guard = lock.lock().await;
		let job = self.load(&initial.id).await?;
		if !job.is_terminal() {
			return Err(VideoError::InvalidControl {
				control: "cleanup",
				detail:  "a running video job cannot be cleaned up".into(),
			});
		}
		remove_if_exists(self.job_path(&handle.id)).await?;
		remove_if_exists(self.upstream_path(&job.upstream_id)).await?;
		self.locks.lock().remove(&job.id);
		Ok(())
	}

	async fn submit_inner(
		&self,
		request: GenerateVideoRequest,
	) -> Result<GenerationHandle, VideoError> {
		let controls = Controls::try_from(&request)?;
		let lease = self.credentials.select()?;
		let boundary = format!("omp-video-{}", ulid::Ulid::generate());
		let body = multipart(&self.store, &boundary, &request, &controls)?;
		let mut outbound = Request::builder()
			.method(Method::POST)
			.uri(format!("{}/videos", self.base_url))
			.header(header::CONTENT_TYPE, format!("multipart/form-data; boundary={boundary}"))
			.body(Full::new(body))
			.map_err(|error| VideoError::Transport(error.to_string().into()))?;
		bind_credential(&mut outbound, lease.clone());
		let response = self.send(outbound).await?;
		let wire: WireVideo = parse_json(&response.body)?;
		validate_id(&wire.id)?;
		validate_wire(&wire)?;
		let gateway_id: Str = ulid::Ulid::generate().to_string().into();
		let created =
			StoredJob::from_wire(&wire, gateway_id.clone(), lease.credential_id(), &controls);
		let lock = self.job_lock(&gateway_id);
		let _guard = lock.lock().await;
		self.save(&created).await?;
		self.materialize(wire, lease, created).await?;
		Ok(GenerationHandle::builder().id(gateway_id).build())
	}

	async fn get_inner(&self, handle: GenerationHandle) -> Result<GenerationStatus, VideoError> {
		validate_id(&handle.id)?;
		let initial = self.load(&handle.id).await?;
		let lock = self.job_lock(&initial.id);
		let _guard = lock.lock().await;
		let current = self.load(&initial.id).await?;
		if current.is_terminal() {
			return Ok(current.status());
		}
		let lease = self.credentials.by_id(current.credential_id)?;
		let mut outbound = Request::builder()
			.method(Method::GET)
			.uri(format!("{}/videos/{}", self.base_url, current.upstream_id))
			.body(Full::new(Bytes::new()))
			.map_err(|error| VideoError::Transport(error.to_string().into()))?;
		bind_credential(&mut outbound, lease.clone());
		let response = self.send(outbound).await?;
		let wire: WireVideo = parse_json(&response.body)?;
		if wire.id != current.upstream_id {
			return Err(VideoError::Protocol("status id did not match the requested job".into()));
		}
		self.materialize(wire, lease, current).await
	}

	async fn cancel_inner(&self, handle: GenerationHandle) -> Result<GenerationStatus, VideoError> {
		validate_id(&handle.id)?;
		let initial = self.load(&handle.id).await?;
		let lock = self.job_lock(&initial.id);
		let _guard = lock.lock().await;
		let mut current = self.load(&initial.id).await?;
		if current.is_terminal() {
			return Ok(current.status());
		}
		let lease = self.credentials.by_id(current.credential_id)?;
		let mut outbound = Request::builder()
			.method(Method::DELETE)
			.uri(format!("{}/videos/{}", self.base_url, current.upstream_id))
			.body(Full::new(Bytes::new()))
			.map_err(|error| VideoError::Transport(error.to_string().into()))?;
		bind_credential(&mut outbound, lease);
		let _ = self.send(outbound).await?;
		current.state = StoredState::Cancelled;
		current.progress = current.progress.min(100.0);
		current.updated_at_ms = now_ms();
		self.save(&current).await?;
		self.locks.lock().remove(&current.id);
		Ok(current.status())
	}

	async fn materialize(
		&self,
		wire: WireVideo,
		lease: CredentialLease,
		previous: StoredJob,
	) -> Result<GenerationStatus, VideoError> {
		validate_wire(&wire)?;
		if wire.id != previous.upstream_id {
			return Err(VideoError::Protocol(
				"status id did not match the durable provider job".into(),
			));
		}
		let controls = previous.controls();
		let mut job =
			StoredJob::from_wire(&wire, previous.id.clone(), lease.credential_id(), &controls);
		job.created_at_ms = previous.created_at_ms;
		job.artifact = previous.artifact;
		if job.state == StoredState::Completed && job.artifact.is_none() {
			let mut outbound = Request::builder()
				.method(Method::GET)
				.uri(format!("{}/videos/{}/content", self.base_url, job.upstream_id))
				.body(Full::new(Bytes::new()))
				.map_err(|error| VideoError::Transport(error.to_string().into()))?;
			bind_credential(&mut outbound, lease);
			let response = self.send(outbound).await?;
			let reference = self
				.store
				.put(&response.body)
				.map_err(|error| VideoError::Persistence(error.to_string().into()))?;
			job.artifact = Some(StoredArtifact {
				hash: reference.hash,
				size: reference.size,
				mime: response.content_type.unwrap_or_else(|| "video/mp4".into()),
			});
		}
		self.save(&job).await?;
		if job.is_terminal() {
			self.locks.lock().remove(&job.id);
		}
		Ok(job.status())
	}

	async fn send(&self, request: Request<Body>) -> Result<HttpResponse, VideoError> {
		let response = self
			.service
			.clone()
			.oneshot(request)
			.await
			.map_err(|error| VideoError::Transport(error.to_string().into()))?;
		let status = response.status();
		let content_type = response
			.headers()
			.get(header::CONTENT_TYPE)
			.and_then(|value| value.to_str().ok())
			.map(Str::from);
		let body = response
			.into_body()
			.collect()
			.await
			.map_err(|error| VideoError::Transport(error.to_string().into()))?
			.to_bytes();
		if status.is_success() {
			return Ok(HttpResponse { body, content_type });
		}
		if status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN {
			return Err(VideoError::Authentication);
		}
		if status == StatusCode::NOT_FOUND {
			return Err(VideoError::NotFound("requested video job".into()));
		}
		let detail = provider_detail(&body);
		Err(VideoError::Provider { status: status.as_u16(), detail })
	}

	async fn load(&self, id: &str) -> Result<StoredJob, VideoError> {
		let bytes = fs::read(self.job_path(id)).await.map_err(|error| {
			if error.kind() == std::io::ErrorKind::NotFound {
				VideoError::NotFound(id.into())
			} else {
				VideoError::Persistence(error.to_string().into())
			}
		})?;
		serde_json::from_slice(&bytes)
			.map_err(|error| VideoError::Persistence(error.to_string().into()))
	}

	async fn load_upstream(&self, upstream_id: &str) -> Result<StoredJob, VideoError> {
		let gateway_id = fs::read_to_string(self.upstream_path(upstream_id))
			.await
			.map_err(|error| {
				if error.kind() == std::io::ErrorKind::NotFound {
					VideoError::NotFound(upstream_id.into())
				} else {
					VideoError::Persistence(error.to_string().into())
				}
			})?;
		self.load(&gateway_id).await
	}

	async fn save(&self, job: &StoredJob) -> Result<(), VideoError> {
		let body = serde_json::to_vec(job)
			.map_err(|error| VideoError::Persistence(error.to_string().into()))?;
		let path = self.job_path(&job.id);
		let temporary = path.with_extension(format!("{}.tmp", ulid::Ulid::generate()));
		fs::write(&temporary, body)
			.await
			.map_err(|error| VideoError::Persistence(error.to_string().into()))?;
		fs::rename(&temporary, &path)
			.await
			.map_err(|error| VideoError::Persistence(error.to_string().into()))?;
		fs::write(self.upstream_path(&job.upstream_id), job.id.as_bytes())
			.await
			.map_err(|error| VideoError::Persistence(error.to_string().into()))
	}

	fn job_path(&self, id: &str) -> PathBuf {
		let name = blake3::hash(id.as_bytes()).to_hex();
		self.jobs_dir.join(format!("{name}.json"))
	}

	fn upstream_path(&self, upstream_id: &str) -> PathBuf {
		let name = blake3::hash(upstream_id.as_bytes()).to_hex();
		self.jobs_dir.join(format!("{name}.upstream"))
	}

	fn job_lock(&self, id: &str) -> Arc<AsyncMutex<()>> {
		let mut locks = self.locks.lock();
		Arc::clone(
			locks
				.entry(id.into())
				.or_insert_with(|| Arc::new(AsyncMutex::new(()))),
		)
	}
}

#[async_trait]
impl<S, B> VideoGen for OpenAiVideoBackend<S, B>
where
	S: Service<Request<Body>, Response = Response<B>> + Clone + Send + Sync + 'static,
	S::Future: Send + 'static,
	S::Error: fmt::Display + Send + Sync + 'static,
	B: hyper::body::Body<Data = Bytes> + Send + Unpin + 'static,
	B::Error: fmt::Display + Send + Sync,
{
	async fn submit(&self, request: GenerateVideoRequest) -> Result<GenerationHandle, facet::Error> {
		self.submit_inner(request).await.map_err(facet_error)
	}

	async fn get(&self, handle: GenerationHandle) -> Result<GenerationStatus, facet::Error> {
		self.get_inner(handle).await.map_err(facet_error)
	}

	async fn attach(
		&self,
		handle: GenerationHandle,
	) -> Result<BoxStream<'static, GenerationStatus>, facet::Error> {
		validate_id(&handle.id).map_err(facet_error)?;
		let backend: Self = (*self).clone();
		let interval = self.poll_interval;
		Ok(Box::pin(async_stream::stream! {
			loop {
				match backend.get_inner(handle.clone()).await {
					Ok(status) => {
						let terminal = is_terminal(status.state);
						yield status;
						if terminal { return; }
					},
					Err(error) => {
						yield failed_status(handle.id.clone(), error);
						return;
					},
				}
				tokio::time::sleep(interval).await;
			}
		}))
	}

	async fn cancel(&self, handle: GenerationHandle) -> Result<GenerationStatus, facet::Error> {
		self.cancel_inner(handle).await.map_err(facet_error)
	}
}

fn facet_error(error: VideoError) -> facet::Error {
	match error {
		VideoError::InvalidControl { control, detail } => facet::Error::Unsupported(vec![
			Unsupported::builder()
				.what(control.into())
				.detail(detail)
				.action(UnsupportedAction::Dropped)
				.build(),
		]),
		error @ VideoError::Transport(_) => facet::Error::Transport(error.to_string().into()),
		error => facet::Error::Provider(error.to_string().into()),
	}
}

#[derive(Clone, Default)]
struct Controls {
	model:   Str,
	seconds: u32,
	size:    Str,
}

impl TryFrom<&GenerateVideoRequest> for Controls {
	type Error = VideoError;

	fn try_from(request: &GenerateVideoRequest) -> Result<Self, Self::Error> {
		if request.prompt.trim().is_empty() {
			return Err(invalid("prompt", "prompt must not be empty"));
		}
		if !matches!(
			request.model.as_str(),
			"sora-2"
				| "sora-2-pro"
				| "sora-2-2025-10-06"
				| "sora-2-pro-2025-10-06"
				| "sora-2-2025-12-08"
		) {
			return Err(invalid("model", "model is not a catalog-supported Sora 2 release"));
		}
		let seconds = request.duration_seconds.unwrap_or(4);
		if !matches!(seconds, 4 | 8 | 12) {
			return Err(invalid("duration_seconds", "Sora accepts exactly 4, 8, or 12 seconds"));
		}
		if request.seed.is_some() {
			return Err(invalid("seed", "Sora does not accept a deterministic seed"));
		}
		if request.audio.is_some() {
			return Err(invalid("audio", "Sora does not expose an audio control"));
		}
		if request.end_frame.is_some() {
			return Err(invalid("end_frame", "Sora accepts only an input reference frame"));
		}
		if !request.references.is_empty() {
			return Err(invalid("references", "Sora accepts one input reference through start_frame"));
		}
		if let Some(frame) = &request.start_frame {
			if !frame.mime.starts_with("image/")
				|| frame.mime.contains('\r')
				|| frame.mime.contains('\n')
				|| frame.mime.len() > 255
			{
				return Err(invalid(
					"start_frame.mime",
					"input reference must have a valid image MIME type",
				));
			}
			if !frame.inline.is_empty() && frame.size != frame.inline.len() as u64 {
				return Err(invalid(
					"start_frame.size",
					"input reference size does not match its bytes",
				));
			}
		}
		if !request.props.is_empty() {
			return Err(invalid("props", "Sora has no namespaced video controls"));
		}
		let aspect = request.aspect_ratio.unwrap_or(AspectRatio::Tall9x16);
		let resolution = request.resolution.unwrap_or(VideoResolution::P720);
		let size = match (aspect, resolution) {
			(AspectRatio::Wide16x9, VideoResolution::P720) => "1280x720",
			(AspectRatio::Tall9x16, VideoResolution::P720) => "720x1280",
			(AspectRatio::Wide16x9, VideoResolution::P1080) => "1792x1024",
			(AspectRatio::Tall9x16, VideoResolution::P1080) => "1024x1792",
			_ => return Err(invalid("size", "Sora accepts 16:9 or 9:16 at 720p or 1080p")),
		};
		Ok(Self { model: request.model.clone(), seconds, size: size.into() })
	}
}

fn invalid(control: &'static str, detail: &'static str) -> VideoError {
	VideoError::InvalidControl { control, detail: detail.into() }
}

fn multipart(
	store: &BlobStore,
	boundary: &str,
	request: &GenerateVideoRequest,
	controls: &Controls,
) -> Result<Bytes, VideoError> {
	let frame = request
		.start_frame
		.as_ref()
		.map(|part| {
			if part.inline.is_empty() {
				store
					.get(&BlobRef { hash: part.hash, size: part.size })
					.map_err(|error| VideoError::Persistence(error.to_string().into()))
			} else {
				Ok(part.inline.clone())
			}
		})
		.transpose()?;
	let mut body =
		Vec::with_capacity(request.prompt.len() + frame.as_ref().map_or(0, Bytes::len) + 512);
	field(&mut body, boundary, "model", &controls.model);
	field(&mut body, boundary, "prompt", &request.prompt);
	field(&mut body, boundary, "seconds", &controls.seconds.to_string());
	field(&mut body, boundary, "size", &controls.size);
	if let (Some(part), Some(frame)) = (&request.start_frame, frame) {
		body.extend_from_slice(
			format!(
				"--{boundary}\r\nContent-Disposition: form-data; name=\"input_reference\"; \
				 filename=\"reference\"\r\nContent-Type: {}\r\n\r\n",
				part.mime
			)
			.as_bytes(),
		);
		body.extend_from_slice(&frame);
		body.extend_from_slice(b"\r\n");
	}
	body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
	Ok(Bytes::from(body))
}

fn field(body: &mut Vec<u8>, boundary: &str, name: &str, value: &str) {
	body.extend_from_slice(
		format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n")
			.as_bytes(),
	);
}

fn bind_credential(request: &mut Request<Body>, lease: CredentialLease) {
	request.extensions_mut().insert(AuthContext::new(PROVIDER));
	request.extensions_mut().insert(lease);
}

fn validate_id(id: &str) -> Result<(), VideoError> {
	if id.is_empty()
		|| !id
			.bytes()
			.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
	{
		return Err(VideoError::Protocol("unsafe or empty video id".into()));
	}
	Ok(())
}

fn validate_wire(wire: &WireVideo) -> Result<(), VideoError> {
	if !matches!(
		wire.status.as_str(),
		"queued" | "in_progress" | "running" | "completed" | "failed" | "cancelled"
	) {
		return Err(VideoError::Protocol(format!("unknown video status {}", wire.status).into()));
	}
	if wire
		.progress
		.is_some_and(|progress| !(0.0..=100.0).contains(&progress))
	{
		return Err(VideoError::Protocol(
			"video progress was outside zero through one hundred".into(),
		));
	}
	Ok(())
}

struct HttpResponse {
	body:         Bytes,
	content_type: Option<Str>,
}

fn parse_json<T: for<'de> Deserialize<'de>>(body: &[u8]) -> Result<T, VideoError> {
	serde_json::from_slice(body).map_err(|error| VideoError::Protocol(error.to_string().into()))
}

fn provider_detail(body: &[u8]) -> Str {
	let code = serde_json::from_slice::<Value>(body)
		.ok()
		.and_then(|value| {
			value
				.pointer("/error/code")
				.and_then(Value::as_str)
				.map(str::to_owned)
		})
		.filter(|code| {
			!code.is_empty()
				&& code.len() <= 64
				&& code
					.bytes()
					.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
		});
	code.map_or_else(
		|| Str::new_static("provider rejected the video request"),
		|code| format!("provider error code {code}").into(),
	)
}

#[derive(Deserialize)]
#[serde(untagged)]
enum WireWebhook {
	Event(WireWebhookEvent),
	Status(WireVideo),
}

#[derive(Deserialize)]
struct WireWebhookEvent {
	#[serde(rename = "type")]
	kind: Str,
	data: WireWebhookData,
}

#[derive(Deserialize)]
struct WireWebhookData {
	id: Str,
}

#[derive(Deserialize)]
struct WireVideo {
	id:           Str,
	status:       Str,
	#[serde(default)]
	progress:     Option<f64>,
	#[serde(default)]
	created_at:   Option<u64>,
	#[serde(default)]
	completed_at: Option<u64>,
	#[serde(default)]
	error:        Option<Value>,
	#[serde(default)]
	usage:        Option<WireUsage>,
}

#[derive(Deserialize)]
struct WireUsage {
	#[serde(default)]
	video_seconds:  Option<u64>,
	#[serde(default)]
	cost_nanos_usd: Option<u64>,
}

#[derive(Clone, Copy, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum StoredState {
	Queued,
	Running,
	Completed,
	Failed,
	Cancelled,
}

#[derive(Clone, Deserialize, Serialize)]
struct StoredArtifact {
	hash: [u8; 32],
	size: u64,
	mime: Str,
}

#[derive(Clone, Deserialize, Serialize)]
struct StoredJob {
	id:             Str,
	upstream_id:    Str,
	credential_id:  u64,
	state:          StoredState,
	progress:       f64,
	detail:         Str,
	created_at_ms:  u64,
	updated_at_ms:  u64,
	model:          Str,
	seconds:        u32,
	size:           Str,
	artifact:       Option<StoredArtifact>,
	usage_seconds:  Option<u64>,
	cost_nanos_usd: Option<u64>,
}

impl StoredJob {
	fn from_wire(wire: &WireVideo, id: Str, credential_id: u64, controls: &Controls) -> Self {
		let state = match wire.status.as_str() {
			"queued" => StoredState::Queued,
			"in_progress" | "running" => StoredState::Running,
			"completed" => StoredState::Completed,
			"failed" => StoredState::Failed,
			"cancelled" => StoredState::Cancelled,
			_ => StoredState::Failed,
		};
		let now = now_ms();
		let created_at_ms = wire
			.created_at
			.map_or(now, |seconds| seconds.saturating_mul(1000));
		let updated_at_ms = wire
			.completed_at
			.map_or(now, |seconds| seconds.saturating_mul(1000));
		Self {
			id,
			upstream_id: wire.id.clone(),
			credential_id,
			state,
			progress: wire.progress.unwrap_or_else(|| {
				if state == StoredState::Completed {
					100.0
				} else {
					0.0
				}
			}),
			detail: if wire.error.is_some() {
				"video generation failed".into()
			} else {
				Str::default()
			},
			created_at_ms,
			updated_at_ms,
			model: controls.model.clone(),
			seconds: controls.seconds,
			size: controls.size.clone(),
			artifact: None,
			usage_seconds: wire
				.usage
				.as_ref()
				.and_then(|usage| usage.video_seconds)
				.or_else(|| (state == StoredState::Completed).then_some(controls.seconds.into())),
			cost_nanos_usd: wire.usage.as_ref().and_then(|usage| usage.cost_nanos_usd),
		}
	}

	fn controls(&self) -> Controls {
		Controls { model: self.model.clone(), seconds: self.seconds, size: self.size.clone() }
	}

	const fn is_terminal(&self) -> bool {
		matches!(self.state, StoredState::Completed | StoredState::Failed | StoredState::Cancelled)
	}

	fn status(&self) -> GenerationStatus {
		let mut props = Props::default();
		props.insert_ns("openai", "model", Value::String(self.model.to_string()));
		props.insert_ns("openai", "size", Value::String(self.size.to_string()));
		let artifacts = self.artifact.as_ref().map_or_else(Vec::new, |artifact| {
			vec![
				GenerationArtifact::builder()
					.blob(
						BlobPart::builder()
							.hash(artifact.hash)
							.mime(artifact.mime.clone())
							.size(artifact.size)
							.inline(Bytes::new())
							.build(),
					)
					.variant("video".into())
					.url(Str::default())
					.url_expires_at_ms(0)
					.build(),
			]
		});
		let usage = self.usage_seconds.map(|seconds| {
			let mut detail = Props::default();
			detail.insert_ns("openai", "video_seconds", Value::from(seconds));
			Usage::builder()
				.input_tokens(0)
				.output_tokens(0)
				.cache_read_tokens(0)
				.cache_write_tokens(0)
				.accuracy(Accuracy::Exact)
				.detail(detail)
				.build()
		});
		GenerationStatus::builder()
			.generation_id(self.id.clone())
			.state(match self.state {
				StoredState::Queued => GenerationState::Queued,
				StoredState::Running => GenerationState::Running,
				StoredState::Completed => GenerationState::Completed,
				StoredState::Failed => GenerationState::Failed,
				StoredState::Cancelled => GenerationState::Cancelled,
			})
			.progress_percent(self.progress)
			.detail(self.detail.clone())
			.artifacts(artifacts)
			.maybe_usage(usage)
			.maybe_cost(self.cost_nanos_usd.map(|nanos_usd| {
				Cost::builder()
					.nanos_usd(nanos_usd)
					.estimated(false)
					.build()
			}))
			.unsupported(Vec::new())
			.created_at_ms(self.created_at_ms)
			.updated_at_ms(self.updated_at_ms)
			.props(props)
			.build()
	}
}

const fn is_terminal(state: GenerationState) -> bool {
	matches!(
		state,
		GenerationState::Completed | GenerationState::Failed | GenerationState::Cancelled
	)
}

fn failed_status(id: Str, _error: VideoError) -> GenerationStatus {
	let now = now_ms();
	GenerationStatus::builder()
		.generation_id(id)
		.state(GenerationState::Failed)
		.progress_percent(0.0)
		.detail("video generation failed".into())
		.artifacts(Vec::new())
		.unsupported(Vec::new())
		.created_at_ms(now)
		.updated_at_ms(now)
		.props(Props::default())
		.build()
}

async fn remove_if_exists(path: PathBuf) -> Result<(), VideoError> {
	match fs::remove_file(path).await {
		Ok(()) => Ok(()),
		Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
		Err(error) => Err(VideoError::Persistence(error.to_string().into())),
	}
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
	use super::*;

	#[test]
	fn provider_diagnostics_never_echo_secret_canaries() {
		let canary = "VIDEO_SECRET_CANARY";
		let detail = provider_detail(
			format!(r#"{{"error":{{"message":"{canary}","code":"bad key {canary}"}}}}"#).as_bytes(),
		);
		assert!(!detail.contains(canary));

		let status = failed_status("gateway-job".into(), VideoError::Provider {
			status: 500,
			detail: canary.into(),
		});
		assert!(!status.detail.contains(canary));
		assert_eq!(status.detail, "video generation failed");

		let wire: WireVideo = serde_json::from_value(serde_json::json!({
			"id":"provider-job",
			"status":"failed",
			"error":{"message":canary}
		}))
		.expect("wire video");
		let job = StoredJob::from_wire(&wire, "gateway-job".into(), 1, &Controls {
			model:   "sora-2".into(),
			seconds: 4,
			size:    "720x1280".into(),
		});
		assert!(!job.detail.contains(canary));
	}
}
