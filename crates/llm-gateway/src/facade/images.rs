//! OpenAI-compatible image generation and multipart edit facades.

use std::{
	convert::Infallible,
	fmt::Display,
	sync::Arc,
	time::{SystemTime, UNIX_EPOCH},
};

use bytes::Bytes;
use futures::{StreamExt, stream};
use http::{Request, StatusCode, header};
use http_body_util::BodyExt;
use hyper::body::Body;
use omp_core::{Str, base64};
use omp_llm_types::{
	BlobPart, GenerateImageRequest, ImageBackground, ImageDone, ImageEvent, ImageFormat,
	ImageQuality, ImageSize, Props,
};
use omp_storage::blob::BlobRef;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::{
	FacadeError, FacadeResponse, FacadeState, Vendor, error_response, json_response, read_json,
};

#[derive(Deserialize)]
struct ImageRequest {
	model:           Str,
	prompt:          Str,
	#[serde(default = "one")]
	n:               u32,
	#[serde(default)]
	size:            Option<Str>,
	#[serde(default)]
	quality:         Option<Str>,
	#[serde(default, alias = "output_format")]
	format:          Option<Str>,
	#[serde(default)]
	background:      Option<Str>,
	#[serde(default = "default_response_format")]
	response_format: Str,
}

struct ParsedEdit {
	request: ImageRequest,
	images:  Vec<BlobPart>,
}

#[derive(Serialize)]
struct ImageResponse {
	created: u64,
	data:    Vec<ImageData>,
}

#[derive(Serialize)]
struct ImageData {
	#[serde(skip_serializing_if = "Option::is_none")]
	b64_json:       Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	url:            Option<String>,
	#[serde(skip_serializing_if = "Str::is_empty")]
	revised_prompt: Str,
}

const fn one() -> u32 {
	1
}

fn default_response_format() -> Str {
	Str::new("url")
}

pub(crate) async fn handle<B>(request: Request<B>, state: Arc<FacadeState>) -> FacadeResponse
where
	B: Body<Data = Bytes> + Send + 'static,
	B::Error: Display,
{
	let path = request.uri().path().to_owned();
	let parsed = if path == "/v1/images/edits" {
		match parse_edit(request).await {
			Ok(parsed) => parsed,
			Err(error) => return error_response(Vendor::OpenAi, error),
		}
	} else {
		let wire: ImageRequest = match read_json(request, Vendor::OpenAi).await {
			Ok(wire) => wire,
			Err(response) => return *response,
		};
		ParsedEdit { request: wire, images: Vec::new() }
	};
	let Some(image_gen) = &state.facets.image_gen else {
		return unavailable("image generation is not available");
	};
	let canonical = match canonical_request(&parsed.request, parsed.images) {
		Ok(request) => request,
		Err(detail) => return unavailable(detail),
	};
	let mut events = match image_gen.generate(canonical).await {
		Ok(events) => events,
		Err(error) => return error_response(Vendor::OpenAi, FacadeError::Facet(error)),
	};
	let mut done = None;
	while let Some(event) = events.next().await {
		if let ImageEvent::Done(result) = event {
			done = Some(result);
		}
	}
	let Some(done) = done else {
		return unavailable("image stream ended without a result");
	};
	image_response(&state, done, parsed.request.response_format.as_str())
}

fn canonical_request(
	wire: &ImageRequest,
	images: Vec<BlobPart>,
) -> Result<GenerateImageRequest, &'static str> {
	if wire.n == 0 {
		return Err("n must be greater than zero");
	}
	let size = match wire.size.as_deref() {
		None | Some("auto") => None,
		Some(value) => Some(parse_size(value)?),
	};
	let quality = match wire.quality.as_deref() {
		None | Some("auto") => None,
		Some(value) => Some(parse_quality(value)?),
	};
	let format = wire.format.as_deref().map(parse_format).transpose()?;
	let background = match wire.background.as_deref() {
		None | Some("auto") => None,
		Some(value) => Some(parse_background(value)?),
	};
	if !matches!(wire.response_format.as_str(), "b64_json" | "url") {
		return Err("response_format must be b64_json or url");
	}
	Ok(GenerateImageRequest::builder()
		.model(wire.model.clone())
		.prompt(wire.prompt.clone())
		.n(wire.n)
		.maybe_size(size)
		.maybe_quality(quality)
		.maybe_format(format)
		.maybe_background(background)
		.input_images(images)
		.props(Props::default())
		.build())
}

fn parse_size(value: &str) -> Result<ImageSize, &'static str> {
	let Some((width, height)) = value.split_once('x') else {
		return Err("size must be WIDTHxHEIGHT");
	};
	let width = width.parse().map_err(|_| "invalid image width")?;
	let height = height.parse().map_err(|_| "invalid image height")?;
	Ok(ImageSize::builder().width(width).height(height).build())
}

fn parse_quality(value: &str) -> Result<ImageQuality, &'static str> {
	match value {
		"low" => Ok(ImageQuality::Low),
		"medium" | "standard" => Ok(ImageQuality::Medium),
		"high" | "hd" => Ok(ImageQuality::High),
		_ => Err("quality must be low, medium, high, standard, hd, or auto"),
	}
}

fn parse_format(value: &str) -> Result<ImageFormat, &'static str> {
	match value {
		"png" => Ok(ImageFormat::Png),
		"webp" => Ok(ImageFormat::Webp),
		"jpeg" | "jpg" => Ok(ImageFormat::Jpeg),
		"svg" => Ok(ImageFormat::Svg),
		_ => Err("format must be png, webp, jpeg, or svg"),
	}
}

fn parse_background(value: &str) -> Result<ImageBackground, &'static str> {
	match value {
		"opaque" => Ok(ImageBackground::Opaque),
		"transparent" => Ok(ImageBackground::Transparent),
		_ => Err("background must be opaque, transparent, or auto"),
	}
}

async fn parse_edit<B>(request: Request<B>) -> Result<ParsedEdit, FacadeError>
where
	B: Body<Data = Bytes> + Send + 'static,
	B::Error: Display,
{
	let content_type = request
		.headers()
		.get(header::CONTENT_TYPE)
		.and_then(|value| value.to_str().ok())
		.ok_or_else(|| invalid_error("multipart Content-Type is required"))?;
	let boundary = multer::parse_boundary(content_type)
		.map_err(|error| invalid_error(format!("invalid multipart boundary: {error}")))?;
	let bytes = request
		.into_body()
		.collect()
		.await
		.map_err(|error| invalid_error(format!("failed to read multipart body: {error}")))?
		.to_bytes();
	let source = stream::once(async move { Ok::<Bytes, Infallible>(bytes) });
	let mut multipart = multer::Multipart::new(source, boundary);
	let mut model = None;
	let mut prompt = None;
	let mut n = 1;
	let mut size = None;
	let mut quality = None;
	let mut format = None;
	let mut background = None;
	let mut response_format = default_response_format();
	let mut images = Vec::new();
	while let Some(field) = multipart
		.next_field()
		.await
		.map_err(|error| invalid_error(format!("invalid multipart body: {error}")))?
	{
		let name = field.name().unwrap_or_default().to_owned();
		let mime = field
			.content_type()
			.map_or_else(|| "application/octet-stream".to_owned(), ToString::to_string);
		let data = field
			.bytes()
			.await
			.map_err(|error| invalid_error(format!("invalid multipart field: {error}")))?;
		match name.as_str() {
			"image" => images.push(blob_part(data, &mime)),
			"model" => model = Some(text_field(data, "model")?),
			"prompt" => prompt = Some(text_field(data, "prompt")?),
			"n" => {
				n = text_field(data, "n")?
					.parse()
					.map_err(|_| invalid_error("invalid n"))?;
			},
			"size" => size = Some(text_field(data, "size")?),
			"quality" => quality = Some(text_field(data, "quality")?),
			"output_format" | "format" => format = Some(text_field(data, "format")?),
			"background" => background = Some(text_field(data, "background")?),
			"response_format" => response_format = text_field(data, "response_format")?,
			_ => {},
		}
	}
	if images.is_empty() {
		return Err(invalid_error("at least one image field is required"));
	}
	Ok(ParsedEdit {
		request: ImageRequest {
			model: model.ok_or_else(|| invalid_error("model is required"))?,
			prompt: prompt.ok_or_else(|| invalid_error("prompt is required"))?,
			n,
			size,
			quality,
			format,
			background,
			response_format,
		},
		images,
	})
}

fn text_field(bytes: Bytes, name: &str) -> Result<Str, FacadeError> {
	let text =
		std::str::from_utf8(&bytes).map_err(|_| invalid_error(format!("{name} must be UTF-8")))?;
	Ok(Str::from(text))
}

fn blob_part(bytes: Bytes, mime: &str) -> BlobPart {
	BlobPart::builder()
		.hash(*blake3::hash(&bytes).as_bytes())
		.mime(Str::from(mime))
		.size(bytes.len() as u64)
		.inline(bytes)
		.build()
}

fn image_response(state: &FacadeState, done: ImageDone, response_format: &str) -> FacadeResponse {
	let mut data = Vec::with_capacity(done.images.len());
	for image in done.images {
		let bytes = if image.inline.is_empty() {
			match state
				.blobs
				.get(&BlobRef { hash: image.hash, size: image.size })
			{
				Ok(bytes) => bytes,
				Err(error) => return server_error(format!("generated image is unavailable: {error}")),
			}
		} else {
			image.inline.clone()
		};
		let encoded = base64::encode(&bytes).into_string();
		let (b64_json, url) = if response_format == "b64_json" {
			(Some(encoded), None)
		} else {
			(None, Some(format!("data:{};base64,{encoded}", image.mime)))
		};
		data.push(ImageData { b64_json, url, revised_prompt: done.revised_prompt.clone() });
	}
	let created = done
		.props
		.get_ns("openai", "created")
		.and_then(serde_json::Value::as_u64)
		.unwrap_or_else(|| {
			SystemTime::now()
				.duration_since(UNIX_EPOCH)
				.map_or(0, |duration| duration.as_secs())
		});
	json_response(StatusCode::OK, &ImageResponse { created, data })
}
fn invalid_error(detail: impl Into<Str>) -> FacadeError {
	FacadeError::Invalid(detail.into())
}

fn unavailable(detail: impl Into<Str>) -> FacadeResponse {
	error_response(Vendor::OpenAi, FacadeError::Invalid(detail.into()))
}

fn server_error(detail: impl Display) -> FacadeResponse {
	json_response(
		StatusCode::INTERNAL_SERVER_ERROR,
		&json!({"error":{"message":detail.to_string(),"type":"api_error"}}),
	)
}

#[cfg(test)]
mod tests {
	use async_trait::async_trait;
	use http_body_util::{BodyExt, Full};
	use omp_llm_catalog::{
		models::Availability,
		registry::{CredentialView, Registry},
	};
	use omp_llm_types::facet::{Error, Facets, ImageGen};
	use omp_storage::blob::BlobStore;

	use super::*;

	struct Credentials;

	impl CredentialView for Credentials {
		fn availability(&self, _provider: &str) -> Availability {
			Availability::Available
		}
	}

	struct FakeImages;

	#[async_trait]
	impl ImageGen for FakeImages {
		async fn generate(
			&self,
			request: GenerateImageRequest,
		) -> Result<futures::stream::BoxStream<'static, ImageEvent>, Error> {
			if request.prompt == "edit this" {
				assert_eq!(request.input_images.len(), 1);
				assert_eq!(request.input_images[0].inline, Bytes::from_static(b"pixels"));
			} else {
				assert_eq!(request.prompt, "paint");
				assert_eq!(request.input_images, [] as [omp_llm_types::BlobPart; 0]);
			}
			let bytes = Bytes::from_static(b"png");
			let image = blob_part(bytes, "image/png");
			let done = ImageDone::builder()
				.images(vec![image])
				.revised_prompt(Str::new("paint vividly"))
				.text(Str::new(""))
				.unsupported(Vec::new())
				.props(Props::default())
				.build();
			Ok(futures::stream::iter([ImageEvent::Done(done)]).boxed())
		}
	}

	fn image_state(directory: &std::path::Path) -> Arc<FacadeState> {
		Arc::new(FacadeState {
			facets:   Arc::new(Facets { image_gen: Some(Arc::new(FakeImages)), ..Facets::default() }),
			registry: Arc::new(parking_lot::RwLock::new(Registry::from_cards(
				&[],
				Arc::new(Credentials),
			))),
			blobs:    Arc::new(BlobStore::open(directory).expect("blob store")),
			auth:     super::super::FacadeAuth::new("token"),
			config:   super::super::FacadeConfig::default(),
		})
	}

	#[test]
	fn image_options_map_to_canonical_values() {
		let wire = ImageRequest {
			model:           "image".into(),
			prompt:          "paint".into(),
			n:               2,
			size:            Some("1024x768".into()),
			quality:         Some("hd".into()),
			format:          Some("webp".into()),
			background:      Some("transparent".into()),
			response_format: "b64_json".into(),
		};
		let request = canonical_request(&wire, Vec::new()).expect("valid request");
		assert_eq!(request.size.expect("size").width, 1024);
		assert_eq!(request.quality, Some(ImageQuality::High));
		assert_eq!(request.format, Some(ImageFormat::Webp));
		assert_eq!(request.background, Some(ImageBackground::Transparent));
	}

	#[tokio::test]
	async fn generation_returns_requested_base64_representation() {
		let directory = tempfile::tempdir().expect("temporary directory");
		let request = Request::post("/v1/images/generations")
			.body(Full::new(Bytes::from_static(
				br#"{"model":"image","prompt":"paint","response_format":"b64_json"}"#,
			)))
			.expect("request");
		let response = handle(request, image_state(directory.path())).await;
		assert_eq!(response.status(), StatusCode::OK);
		let body = response
			.into_body()
			.collect()
			.await
			.expect("infallible")
			.to_bytes();
		let value: serde_json::Value = serde_json::from_slice(&body).expect("JSON");
		assert_eq!(value["data"][0]["b64_json"], "cG5n");
		assert!(value["data"][0].get("url").is_none());
	}

	#[tokio::test]
	async fn multipart_edit_collects_image_input() {
		let boundary = "omp-image-boundary";
		let body = format!(
			"--{boundary}\r\nContent-Disposition: form-data; \
			 name=\"model\"\r\n\r\nimage-model\r\n--{boundary}\r\nContent-Disposition: form-data; \
			 name=\"prompt\"\r\n\r\nedit this\r\n--{boundary}\r\nContent-Disposition: form-data; \
			 name=\"response_format\"\r\n\r\nb64_json\r\n--{boundary}\r\nContent-Disposition: \
			 form-data; name=\"image\"; filename=\"input.png\"\r\nContent-Type: \
			 image/png\r\n\r\npixels\r\n--{boundary}--\r\n"
		);
		let request = Request::post("/v1/images/edits")
			.header(header::CONTENT_TYPE, format!("multipart/form-data; boundary={boundary}"))
			.body(Full::new(Bytes::from(body)))
			.expect("multipart request");
		let directory = tempfile::tempdir().expect("temporary directory");
		let response = handle(request, image_state(directory.path())).await;
		assert_eq!(response.status(), StatusCode::OK);
		let body = response
			.into_body()
			.collect()
			.await
			.expect("infallible")
			.to_bytes();
		let value: serde_json::Value = serde_json::from_slice(&body).expect("JSON");
		assert_eq!(value["data"][0]["b64_json"], "cG5n");
	}
}
