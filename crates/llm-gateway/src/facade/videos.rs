//! OpenAI-compatible asynchronous video generation facade.
//!
//! Only the official `/v1/videos` route family is mounted. We deliberately do
//! not duplicate it under `OpenRouter`'s `/api/v1/videos`: one canonical
//! foreign path avoids divergent auth, cache, and artifact-download behavior.

use std::{convert::Infallible, fmt::Display, sync::Arc};

use bytes::{Bytes, BytesMut};
use http::{Method, Request, Response, StatusCode, header};
use http_body_util::{BodyExt, StreamBody};
use hyper::body::{Body, Frame};
use omp_core::{SmolStr, base64};
use omp_llm_types::{
	AspectRatio, BlobPart, GenerateVideoRequest, GenerationState, GenerationStatus, Props,
	VideoResolution, facet::GenerationHandle,
};
use omp_storage::blob::BlobRef;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncReadExt;

use super::{
	FacadeError, FacadeResponse, FacadeState, Vendor, error_response, json_response, read_json,
};

#[derive(Deserialize)]
struct VideoRequest {
	model:        SmolStr,
	prompt:       SmolStr,
	#[serde(default, alias = "seconds")]
	duration:     Option<VideoDuration>,
	#[serde(default)]
	aspect_ratio: Option<SmolStr>,
	#[serde(default)]
	resolution:   Option<SmolStr>,
	#[serde(default)]
	size:         Option<SmolStr>,
	#[serde(default)]
	seed:         Option<u64>,
	#[serde(default)]
	audio:        Option<bool>,
	#[serde(default)]
	start_frame:  Option<EncodedBlob>,
	#[serde(default)]
	end_frame:    Option<EncodedBlob>,
	#[serde(default)]
	references:   Vec<EncodedBlob>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum VideoDuration {
	Number(u32),
	Text(SmolStr),
}

impl VideoDuration {
	fn into_seconds(self) -> Result<u32, FacadeError> {
		let seconds = match self {
			Self::Number(seconds) => seconds,
			Self::Text(seconds) => seconds
				.parse()
				.map_err(|_| invalid_error("seconds must be a positive integer"))?,
		};
		if seconds == 0 {
			Err(invalid_error("seconds must be a positive integer"))
		} else {
			Ok(seconds)
		}
	}
}

#[derive(Deserialize)]
struct EncodedBlob {
	data:      SmolStr,
	#[serde(default = "default_image_mime")]
	mime_type: SmolStr,
}

#[derive(Serialize)]
struct SubmittedVideo {
	id:     SmolStr,
	object: &'static str,
	status: &'static str,
}

#[derive(Serialize)]
struct VideoStatus {
	id:         SmolStr,
	object:     &'static str,
	status:     &'static str,
	progress:   f64,
	created_at: u64,
	updated_at: u64,
	#[serde(skip_serializing_if = "Option::is_none")]
	error:      Option<VideoError>,
}

#[derive(Serialize)]
struct VideoError {
	message: SmolStr,
}

fn default_image_mime() -> SmolStr {
	SmolStr::new("image/png")
}

pub(crate) async fn handle<B>(request: Request<B>, state: Arc<FacadeState>) -> FacadeResponse
where
	B: Body<Data = Bytes> + Send + 'static,
	B::Error: Display,
{
	let method = request.method().clone();
	let path = request.uri().path().to_owned();
	if path == "/v1/videos" {
		return if method == Method::POST {
			submit(request, &state).await
		} else {
			invalid("video submission requires POST")
		};
	}
	let Some(suffix) = path.strip_prefix("/v1/videos/") else {
		return invalid("video route not found");
	};
	let (id, content) = suffix
		.strip_suffix("/content")
		.map_or((suffix, false), |id| (id, true));
	if id.is_empty() || id.contains('/') {
		return invalid("invalid video id");
	}
	match (method, content) {
		(Method::GET, true) => download(&state, id).await,
		(Method::GET, false) => poll(&state, id).await,
		(Method::DELETE, false) => cancel(&state, id).await,
		_ => invalid("unsupported video operation"),
	}
}

async fn submit<B>(request: Request<B>, state: &FacadeState) -> FacadeResponse
where
	B: Body<Data = Bytes> + Send + 'static,
	B::Error: Display,
{
	let wire: VideoRequest = match read_json(request, Vendor::OpenAi).await {
		Ok(wire) => wire,
		Err(response) => return *response,
	};
	let canonical = match canonical_request(wire) {
		Ok(request) => request,
		Err(error) => return error_response(Vendor::OpenAi, error),
	};
	let Some(video_gen) = &state.facets.video_gen else {
		return invalid("video generation is not available");
	};
	let handle = match video_gen.submit(canonical).await {
		Ok(handle) => handle,
		Err(error) => return error_response(Vendor::OpenAi, FacadeError::Facet(error)),
	};
	json_response(StatusCode::ACCEPTED, &SubmittedVideo {
		id:     handle.id,
		object: "video",
		status: "queued",
	})
}

fn canonical_request(wire: VideoRequest) -> Result<GenerateVideoRequest, FacadeError> {
	let duration = wire.duration.map(VideoDuration::into_seconds).transpose()?;
	let size = wire.size.as_deref().map(parse_video_size).transpose()?;
	let aspect_ratio = wire
		.aspect_ratio
		.as_deref()
		.map(parse_aspect_ratio)
		.transpose()?
		.or_else(|| size.map(|(aspect_ratio, _)| aspect_ratio));
	let resolution = wire
		.resolution
		.as_deref()
		.map(parse_resolution)
		.transpose()?
		.or_else(|| size.map(|(_, resolution)| resolution));
	let start_frame = wire.start_frame.map(decode_blob).transpose()?;
	let end_frame = wire.end_frame.map(decode_blob).transpose()?;
	let references = wire
		.references
		.into_iter()
		.map(decode_blob)
		.collect::<Result<Vec<_>, _>>()?;
	Ok(GenerateVideoRequest::builder()
		.model(wire.model)
		.prompt(wire.prompt)
		.maybe_duration_seconds(duration)
		.maybe_aspect_ratio(aspect_ratio)
		.maybe_resolution(resolution)
		.maybe_seed(wire.seed)
		.maybe_audio(wire.audio)
		.maybe_start_frame(start_frame)
		.maybe_end_frame(end_frame)
		.references(references)
		.props(Props::default())
		.build())
}

fn parse_aspect_ratio(value: &str) -> Result<AspectRatio, FacadeError> {
	match value {
		"1:1" => Ok(AspectRatio::Square),
		"16:9" => Ok(AspectRatio::Wide16x9),
		"9:16" => Ok(AspectRatio::Tall9x16),
		"4:3" => Ok(AspectRatio::Landscape4x3),
		"3:4" => Ok(AspectRatio::Portrait3x4),
		"3:2" => Ok(AspectRatio::Landscape3x2),
		"2:3" => Ok(AspectRatio::Portrait2x3),
		"21:9" => Ok(AspectRatio::Ultrawide21x9),
		_ => Err(invalid_error("unsupported aspect_ratio")),
	}
}

fn parse_resolution(value: &str) -> Result<VideoResolution, FacadeError> {
	match value {
		"480p" => Ok(VideoResolution::P480),
		"720p" => Ok(VideoResolution::P720),
		"1080p" => Ok(VideoResolution::P1080),
		"4k" | "2160p" => Ok(VideoResolution::K4),
		_ => Err(invalid_error("resolution must be 480p, 720p, 1080p, or 4k")),
	}
}

fn parse_video_size(value: &str) -> Result<(AspectRatio, VideoResolution), FacadeError> {
	match value {
		"1280x720" => Ok((AspectRatio::Wide16x9, VideoResolution::P720)),
		"720x1280" => Ok((AspectRatio::Tall9x16, VideoResolution::P720)),
		"1792x1024" => Ok((AspectRatio::Wide16x9, VideoResolution::P1080)),
		"1024x1792" => Ok((AspectRatio::Tall9x16, VideoResolution::P1080)),
		_ => Err(invalid_error("unsupported OpenAI video size")),
	}
}
fn decode_blob(encoded: EncodedBlob) -> Result<BlobPart, FacadeError> {
	let bytes = base64::decode(encoded.data.as_bytes())
		.into_vec()
		.map_err(|_| invalid_error("video frame data must be valid base64"))?;
	let bytes = Bytes::from(bytes);
	Ok(BlobPart::builder()
		.hash(*blake3::hash(&bytes).as_bytes())
		.mime(encoded.mime_type)
		.size(bytes.len() as u64)
		.inline(bytes)
		.build())
}

async fn poll(state: &FacadeState, id: &str) -> FacadeResponse {
	match get_status(state, id).await {
		Ok(status) => json_response(StatusCode::OK, &wire_status(status)),
		Err(error) => error_response(Vendor::OpenAi, error),
	}
}

async fn cancel(state: &FacadeState, id: &str) -> FacadeResponse {
	let Some(video_gen) = &state.facets.video_gen else {
		return invalid("video generation is not available");
	};
	let handle = GenerationHandle::builder().id(SmolStr::new(id)).build();
	match video_gen.cancel(handle).await {
		Ok(status) => json_response(StatusCode::OK, &wire_status(status)),
		Err(error) => error_response(Vendor::OpenAi, FacadeError::Facet(error)),
	}
}

async fn download(state: &FacadeState, id: &str) -> FacadeResponse {
	let status = match get_status(state, id).await {
		Ok(status) => status,
		Err(error) => return error_response(Vendor::OpenAi, error),
	};
	if status.state != GenerationState::Completed {
		return json_response(
			StatusCode::CONFLICT,
			&serde_json::json!({"error":{"message":"video is not complete","type":"invalid_request_error"}}),
		);
	}
	let artifact = status
		.artifacts
		.iter()
		.find(|artifact| artifact.variant == "video" && artifact.blob.is_some())
		.or_else(|| {
			status
				.artifacts
				.iter()
				.find(|artifact| artifact.blob.is_some())
		});
	let Some(blob) = artifact.and_then(|artifact| artifact.blob.as_ref()) else {
		return json_response(
			StatusCode::NOT_FOUND,
			&serde_json::json!({"error":{"message":"video artifact is unavailable","type":"api_error"}}),
		);
	};
	let reference = BlobRef { hash: blob.hash, size: blob.size };
	let path = state.blobs.path(&reference);
	let mut file = match tokio::fs::File::open(path).await {
		Ok(file) => file,
		Err(error) => {
			return json_response(
				StatusCode::NOT_FOUND,
				&serde_json::json!({"error":{"message":error.to_string(),"type":"api_error"}}),
			);
		},
	};
	let length = match file.metadata().await {
		Ok(metadata) => metadata.len(),
		Err(error) => {
			return json_response(
				StatusCode::INTERNAL_SERVER_ERROR,
				&serde_json::json!({"error":{"message":error.to_string(),"type":"api_error"}}),
			);
		},
	};
	if length != reference.size {
		return json_response(
			StatusCode::INTERNAL_SERVER_ERROR,
			&serde_json::json!({"error":{"message":"video artifact length does not match its blob reference","type":"api_error"}}),
		);
	}
	let mime = if blob.mime.is_empty() {
		"video/mp4"
	} else {
		blob.mime.as_str()
	};
	let frames = async_stream::stream! {
		let mut buffer = BytesMut::zeroed(64 * 1024);
		loop {
			match file.read(buffer.as_mut()).await {
				Ok(0) | Err(_) => break,
				Ok(read) => yield Ok::<_, Infallible>(Frame::data(Bytes::copy_from_slice(&buffer[..read]))),
			}
		}
	};
	Response::builder()
		.status(StatusCode::OK)
		.header(header::CONTENT_TYPE, mime)
		.body(BodyExt::boxed_unsync(StreamBody::new(frames)))
		.expect("static video response is valid")
}

async fn get_status(state: &FacadeState, id: &str) -> Result<GenerationStatus, FacadeError> {
	let Some(video_gen) = &state.facets.video_gen else {
		return Err(invalid_error("video generation is not available"));
	};
	video_gen
		.get(GenerationHandle::builder().id(SmolStr::from(id)).build())
		.await
		.map_err(FacadeError::Facet)
}

fn wire_status(status: GenerationStatus) -> VideoStatus {
	let (wire, error) = match status.state {
		GenerationState::Queued => ("queued", None),
		GenerationState::Running => ("in_progress", None),
		GenerationState::Completed => ("completed", None),
		GenerationState::Cancelled => ("cancelled", None),
		GenerationState::Failed => ("failed", Some(VideoError { message: status.detail.clone() })),
		_ => ("failed", Some(VideoError { message: SmolStr::new("unknown generation state") })),
	};
	VideoStatus {
		id: status.generation_id,
		object: "video",
		status: wire,
		progress: status.progress_percent,
		created_at: status.created_at_ms / 1000,
		updated_at: status.updated_at_ms / 1000,
		error,
	}
}
fn invalid_error(detail: impl Into<SmolStr>) -> FacadeError {
	FacadeError::Invalid(detail.into())
}

fn invalid(detail: impl Into<SmolStr>) -> FacadeResponse {
	error_response(Vendor::OpenAi, FacadeError::Invalid(detail.into()))
}

#[cfg(test)]
mod tests {
	use async_trait::async_trait;
	use futures::{StreamExt, stream::BoxStream};
	use http_body_util::{BodyExt, Full};
	use omp_llm_catalog::{
		models::Availability,
		registry::{CredentialView, Registry},
	};
	use omp_llm_types::{
		GenerationArtifact,
		facet::{Error, Facets, VideoGen},
	};
	use omp_storage::blob::BlobStore;

	use super::*;

	struct Credentials;

	impl CredentialView for Credentials {
		fn availability(&self, _provider: &str) -> Availability {
			Availability::Available
		}
	}

	struct FakeVideo {
		status: GenerationStatus,
	}

	#[async_trait]
	impl VideoGen for FakeVideo {
		async fn submit(&self, request: GenerateVideoRequest) -> Result<GenerationHandle, Error> {
			assert_eq!(request.prompt, "ocean");
			Ok(GenerationHandle::builder()
				.id(SmolStr::new("video-1"))
				.build())
		}

		async fn get(&self, handle: GenerationHandle) -> Result<GenerationStatus, Error> {
			assert_eq!(handle.id, "video-1");
			Ok(self.status.clone())
		}

		async fn attach(
			&self,
			_handle: GenerationHandle,
		) -> Result<BoxStream<'static, GenerationStatus>, Error> {
			Ok(futures::stream::empty().boxed())
		}

		async fn cancel(&self, _handle: GenerationHandle) -> Result<GenerationStatus, Error> {
			Ok(self.status.clone())
		}
	}

	fn video_state(directory: &std::path::Path) -> Arc<FacadeState> {
		let blobs = BlobStore::open(directory).expect("blob store");
		let reference = blobs.put(b"video bytes").expect("store video");
		let blob = BlobPart::builder()
			.hash(reference.hash)
			.mime(SmolStr::new("video/mp4"))
			.size(reference.size)
			.inline(Bytes::new())
			.build();
		let artifact = GenerationArtifact::builder()
			.blob(blob)
			.variant(SmolStr::new("video"))
			.url(SmolStr::new(""))
			.url_expires_at_ms(0)
			.build();
		let status = GenerationStatus::builder()
			.generation_id(SmolStr::new("video-1"))
			.state(GenerationState::Completed)
			.progress_percent(100.0)
			.detail(SmolStr::new(""))
			.artifacts(vec![artifact])
			.unsupported(Vec::new())
			.created_at_ms(1000)
			.updated_at_ms(2000)
			.props(Props::default())
			.build();
		Arc::new(FacadeState {
			facets:   Arc::new(Facets {
				video_gen: Some(Arc::new(FakeVideo { status })),
				..Facets::default()
			}),
			registry: Arc::new(parking_lot::RwLock::new(Registry::from_cards(
				&[],
				Arc::new(Credentials),
			))),
			blobs:    Arc::new(blobs),
			auth:     super::super::FacadeAuth::new("token"),
			config:   super::super::FacadeConfig::default(),
		})
	}

	#[test]
	fn maps_completed_job_status() {
		let status = GenerationStatus::builder()
			.generation_id(SmolStr::new("video-1"))
			.state(GenerationState::Completed)
			.progress_percent(100.0)
			.detail(SmolStr::new(""))
			.artifacts(Vec::new())
			.unsupported(Vec::new())
			.created_at_ms(1000)
			.updated_at_ms(2000)
			.props(Props::default())
			.build();
		let wire = wire_status(status);
		assert_eq!(wire.id, "video-1");
		assert_eq!(wire.status, "completed");
		assert_eq!(wire.updated_at, 2);
	}

	#[tokio::test]
	async fn submit_poll_cancel_and_download_share_one_video_job() {
		let directory = tempfile::tempdir().expect("temporary directory");
		let state = video_state(directory.path());
		let submit_request = Request::post("/v1/videos")
			.body(Full::new(Bytes::from_static(br#"{"model":"video","prompt":"ocean"}"#)))
			.expect("submit request");
		let submitted = handle(submit_request, Arc::clone(&state)).await;
		assert_eq!(submitted.status(), StatusCode::ACCEPTED);
		let submitted_body = submitted
			.into_body()
			.collect()
			.await
			.expect("infallible")
			.to_bytes();
		let submitted_json: serde_json::Value =
			serde_json::from_slice(&submitted_body).expect("JSON");
		assert_eq!(submitted_json["id"], "video-1");

		let poll_request = Request::get("/v1/videos/video-1")
			.body(Full::new(Bytes::new()))
			.expect("poll request");
		let polled = handle(poll_request, Arc::clone(&state)).await;
		let polled_body = polled
			.into_body()
			.collect()
			.await
			.expect("infallible")
			.to_bytes();
		let polled_json: serde_json::Value = serde_json::from_slice(&polled_body).expect("JSON");
		assert_eq!(polled_json["status"], "completed");

		let cancel_request = Request::delete("/v1/videos/video-1")
			.body(Full::new(Bytes::new()))
			.expect("cancel request");
		let cancelled = handle(cancel_request, Arc::clone(&state)).await;
		assert_eq!(cancelled.status(), StatusCode::OK);

		let content_request = Request::get("/v1/videos/video-1/content")
			.body(Full::new(Bytes::new()))
			.expect("content request");
		let content = handle(content_request, state).await;
		assert_eq!(content.headers()[header::CONTENT_TYPE], "video/mp4");
		let bytes = content
			.into_body()
			.collect()
			.await
			.expect("infallible")
			.to_bytes();
		assert_eq!(bytes, Bytes::from_static(b"video bytes"));
	}
}
