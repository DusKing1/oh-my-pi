//! Production HTTP adapters for the six supported image providers.
//!
//! A namespaced per-call `<image-provider>/base_url` override has highest
//! precedence, followed by the overlay-materialized [`ProviderCatalog`] route
//! and finally the provider's compiled safety default. Credentials never carry
//! endpoints or models.

use std::{fmt::Display, sync::Arc, time::Duration};

use async_trait::async_trait;
use bytes::Bytes;
use http::{Method, Request, Response, StatusCode, header};
use http_body_util::{BodyExt, Full};
use hyper::body::Body as HyperBody;
use omp_core::{Str, base64};
use omp_llm_catalog::{
	codex::{CODEX_CLIENT_VERSION, CODEX_ORIGINATOR},
	provider::ProviderCatalog,
};
use omp_llm_egress::{
	auth_inject::{AuthContext, AuthInjectLayer, CredentialMetadataSource},
	client::{Body, EgressClient},
};
use omp_llm_types::{
	AspectRatio, BlobPart, GenerateImageRequest, ImageDone, ImageFormat, ImageSize, Props,
};
use serde_json::{Value, json};
use tower::{Layer, Service, ServiceExt};

use crate::images::{
	ImageAttemptError, ImageBackend, ImageCredential, ImageProvider, ImageProviderError,
	ImageProviderErrorKind,
};

const IMAGE_TIMEOUT: Duration = Duration::from_secs(180);
const IMAGE_SYSTEM_INSTRUCTION: &str = "You are an AI image generator. Generate images based on \
                                        user descriptions. Focus on creating high-quality, \
                                        visually appealing images that match the user's request.";

#[async_trait]
trait ImageHttpClient: Send + Sync {
	async fn execute(&self, request: Request<Body>) -> Result<Response<Bytes>, Str>;
}

#[async_trait]
impl<S, R> ImageHttpClient for S
where
	S: Service<Request<Body>, Response = Response<R>> + Clone + Send + Sync + 'static,
	S::Future: Send,
	S::Error: Display + Send,
	R: HyperBody<Data = Bytes> + Send + 'static,
	R::Error: Display + Send,
{
	async fn execute(&self, request: Request<Body>) -> Result<Response<Bytes>, Str> {
		let response = (*self)
			.clone()
			.oneshot(request)
			.await
			.map_err(|error| error.to_string())?;
		let (parts, body) = response.into_parts();
		let body = body
			.collect()
			.await
			.map_err(|error| error.to_string())?
			.to_bytes();
		Ok(Response::from_parts(parts, body))
	}
}

/// Production provider adapters sharing the gateway's authenticated egress
/// service.
#[derive(Clone)]
pub struct EgressImageBackend {
	client:    Arc<dyn ImageHttpClient>,
	providers: Arc<ProviderCatalog>,
}

impl EgressImageBackend {
	/// Creates an image backend using `client` and the overlay-materialized
	/// provider catalog for every route and static header.
	#[must_use]
	pub fn new<S, R>(client: S, providers: Arc<ProviderCatalog>) -> Self
	where
		S: Service<Request<Body>, Response = Response<R>> + Clone + Send + Sync + 'static,
		S::Future: Send,
		S::Error: Display + Send,
		R: HyperBody<Data = Bytes> + Send + 'static,
		R::Error: Display + Send,
	{
		Self { client: Arc::new(client), providers }
	}

	/// Creates an image backend whose requests redeem canonical credential
	/// leases and validate their non-secret metadata in the egress layer.
	#[must_use]
	pub fn authenticated<C>(client: EgressClient, source: C, providers: Arc<ProviderCatalog>) -> Self
	where
		C: CredentialMetadataSource,
		C::Error: Display,
	{
		Self::new(AuthInjectLayer::new(source).layer(client), providers)
	}

	/// Creates the authenticated production backend with the image deadline.
	#[must_use]
	pub fn production<C>(source: C, providers: Arc<ProviderCatalog>) -> Self
	where
		C: CredentialMetadataSource,
		C::Error: Display,
	{
		Self::authenticated(EgressClient::new(IMAGE_TIMEOUT), source, providers)
	}

	async fn post_json(
		&self,
		provider: ImageProvider,
		credential: &ImageCredential,
		url: &str,
		headers: &[(&str, String)],
		body: Value,
	) -> Result<HttpPayload, ImageAttemptError> {
		let mut builder = Request::builder()
			.method(Method::POST)
			.uri(url)
			.header(header::CONTENT_TYPE, "application/json");
		if let Some(route) = self.providers.get(provider.catalog_id()) {
			for (name, value) in &route.headers {
				builder = builder.header(name.as_str(), value.as_str());
			}
		}
		for (name, value) in headers {
			builder = builder.header(*name, value);
		}
		let bytes = serde_json::to_vec(&body).map_err(|error| parse_error(provider, error))?;
		let mut request = builder
			.body(Full::new(Bytes::from(bytes)))
			.map_err(|error| transport_error(provider, error))?;
		let auth_provider = if credential.auth_provider.is_empty() {
			provider.id()
		} else {
			credential.auth_provider.as_str()
		};
		request
			.extensions_mut()
			.insert(AuthContext::new(auth_provider));
		if let Some(lease) = &credential.lease {
			request.extensions_mut().insert(lease.clone());
		}
		self.send(provider, request).await
	}

	async fn send(
		&self,
		provider: ImageProvider,
		request: Request<Body>,
	) -> Result<HttpPayload, ImageAttemptError> {
		let response = self
			.client
			.execute(request)
			.await
			.map_err(|error| transport_error(provider, error))?;
		let status = response.status();
		let content_type = response
			.headers()
			.get(header::CONTENT_TYPE)
			.and_then(|value| value.to_str().ok())
			.unwrap_or_default()
			.to_owned();
		let body = response.into_body();
		if !status.is_success() {
			return Err(status_error(provider, status, provider_message(&body)));
		}
		Ok(HttpPayload { body, content_type })
	}

	async fn download(
		&self,
		provider: ImageProvider,
		url: &str,
	) -> Result<BlobPart, ImageAttemptError> {
		if let Some((mime, encoded)) = parse_data_url(url) {
			return decode_blob(provider, encoded, mime);
		}
		let request = Request::builder()
			.method(Method::GET)
			.uri(url)
			.body(Full::new(Bytes::new()))
			.map_err(|error| transport_error(provider, error))?;
		let payload = self.send(provider, request).await?;
		let mime = payload
			.content_type
			.split(';')
			.next()
			.filter(|value| value.starts_with("image/"))
			.unwrap_or("application/octet-stream");
		Ok(blob(payload.body, mime))
	}

	async fn openai(
		&self,
		provider: ImageProvider,
		credential: &ImageCredential,
		request: &GenerateImageRequest,
	) -> Result<ImageDone, ImageAttemptError> {
		let codex = provider == ImageProvider::OpenAiCodex;
		let base = endpoint(
			&self.providers,
			request,
			provider,
			if codex {
				"https://chatgpt.com/backend-api"
			} else {
				"https://api.openai.com/v1"
			},
		);
		let url =
			format!("{}/{}responses", base.trim_end_matches('/'), if codex { "codex/" } else { "" });
		let model = model(request, provider, if codex { "gpt-5.5" } else { "gpt-image-2" });
		let mut content = vec![json!({"type":"input_text", "text":request.prompt})];
		for image in &request.input_images {
			content.push(json!({"type":"input_image", "detail":"auto", "image_url":data_url(image)}));
		}
		let mut tool = json!({
			"type":"image_generation",
			"action": if request.input_images.is_empty() { "generate" } else { "edit" },
			"output_format": format_name(request.format.unwrap_or(ImageFormat::Webp)),
		});
		if let Some(size) = openai_size(request) {
			tool["size"] = size.into();
		}
		let mut body = json!({
			"model":model,
			"input":[{"role":"user", "content":content}],
			"tools":[tool],
			"tool_choice":{"type":"image_generation"},
			"store":false,
		});
		if codex {
			body["stream"] = true.into();
			body["instructions"] = IMAGE_SYSTEM_INSTRUCTION.into();
		}
		let mut headers = Vec::new();
		if codex {
			headers.push(("openai-beta", "responses=experimental".to_owned()));
			headers.push(("originator", CODEX_ORIGINATOR.to_owned()));
			headers.push(("version", CODEX_CLIENT_VERSION.to_owned()));
			if let Some(account) = &credential.account_id {
				headers.push(("chatgpt-account-id", account.to_string()));
			}
			if let Some(session) = request
				.props
				.get_ns("openai-codex", "session_id")
				.and_then(Value::as_str)
			{
				headers.push(("conversation_id", session.to_owned()));
				headers.push(("session_id", session.to_owned()));
			}
		}
		let payload = self
			.post_json(provider, credential, &url, &headers, body)
			.await?;
		let value = if codex || payload.content_type.contains("text/event-stream") {
			last_sse_response(provider, &payload.body)?
		} else {
			serde_json::from_slice(&payload.body).map_err(|error| parse_error(provider, error))?
		};
		let mut images = Vec::new();
		let mut revised = Str::default();
		let mut text = Vec::new();
		for output in value
			.get("output")
			.and_then(Value::as_array)
			.into_iter()
			.flatten()
		{
			if output.get("type").and_then(Value::as_str) == Some("image_generation_call") {
				if let Some(encoded) = output.get("result").and_then(Value::as_str) {
					images.push(decode_blob(
						provider,
						encoded,
						format_mime(request.format.unwrap_or(ImageFormat::Webp)),
					)?);
				}
				if let Some(prompt) = output.get("revised_prompt").and_then(Value::as_str) {
					revised = prompt.into();
				}
			}
			for part in output
				.get("content")
				.and_then(Value::as_array)
				.into_iter()
				.flatten()
			{
				if let Some(value) = part
					.get("text")
					.or_else(|| part.get("refusal"))
					.and_then(Value::as_str)
				{
					text.push(value);
				}
			}
		}
		finish(provider, request, images, revised, text.join("\n").into(), model)
	}

	async fn antigravity(
		&self,
		credential: &ImageCredential,
		request: &GenerateImageRequest,
	) -> Result<ImageDone, ImageAttemptError> {
		let provider = ImageProvider::Antigravity;
		let project = request
			.props
			.get_ns("antigravity", "project_id")
			.and_then(Value::as_str)
			.or(credential.project_id.as_deref())
			.ok_or_else(|| ImageProviderError {
				provider: provider.id().into(),
				kind:     ImageProviderErrorKind::Parse,
				message:  "Antigravity requires project_id in the per-call override or credential \
				           metadata"
					.into(),
			})?;
		let base =
			endpoint(&self.providers, request, provider, "https://daily-cloudcode-pa.googleapis.com");
		let url = format!("{}/v1internal:streamGenerateContent?alt=sse", base.trim_end_matches('/'));
		let model = model(request, provider, "gemini-3-pro-image");
		let parts = gemini_parts(request);
		let body = json!({
			"project":project,
			"model":model,
			"request":{
				"contents":[{"role":"user", "parts":parts}],
				"systemInstruction":{"parts":[{"text":IMAGE_SYSTEM_INSTRUCTION}]},
				"generationConfig":{"responseModalities":["IMAGE"], "imageConfig":image_config(request), "candidateCount":1},
				"safetySettings":[
					{"category":"HARM_CATEGORY_HARASSMENT","threshold":"BLOCK_ONLY_HIGH"},
					{"category":"HARM_CATEGORY_HATE_SPEECH","threshold":"BLOCK_ONLY_HIGH"},
					{"category":"HARM_CATEGORY_SEXUALLY_EXPLICIT","threshold":"BLOCK_ONLY_HIGH"},
					{"category":"HARM_CATEGORY_DANGEROUS_CONTENT","threshold":"BLOCK_ONLY_HIGH"},
					{"category":"HARM_CATEGORY_CIVIC_INTEGRITY","threshold":"BLOCK_ONLY_HIGH"}
				]
			},
			"requestType":"agent",
			"requestId":format!("agent-{}", ulid::Ulid::generate()),
			"userAgent":"antigravity"
		});
		let payload = self
			.post_json(provider, credential, &url, &[("accept", "text/event-stream".to_owned())], body)
			.await?;
		let chunks = sse_values(provider, &payload.body)?;
		let mut images = Vec::new();
		let mut texts = Vec::new();
		for chunk in chunks {
			if let Some(response) = chunk.get("response") {
				collect_gemini_parts(provider, response, &mut images, &mut texts)?;
			}
		}
		finish(provider, request, images, Str::default(), texts.join(" ").into(), model)
	}

	async fn xai(
		&self,
		credential: &ImageCredential,
		request: &GenerateImageRequest,
	) -> Result<ImageDone, ImageAttemptError> {
		let provider = ImageProvider::Xai;
		if request.input_images.len() > 3 {
			return Err(parse_message(provider, "xAI image edits accept at most three references"));
		}
		let base = endpoint(&self.providers, request, provider, "https://api.x.ai/v1");
		let edit = !request.input_images.is_empty();
		let url = format!(
			"{}/images/{}",
			base.trim_end_matches('/'),
			if edit { "edits" } else { "generations" }
		);
		let model = model(request, provider, "grok-imagine-image");
		let mut body = json!({
			"model":model,
			"prompt":request.prompt,
			"aspect_ratio":aspect_ratio(request.aspect_ratio).unwrap_or("1:1"),
			"resolution":if request.size.is_some_and(|size| size.width > 1024 || size.height > 1024) { "2k" } else { "1k" },
			"n":if request.n == 0 { 1 } else { request.n },
			"response_format":"b64_json"
		});
		if edit && request.input_images.len() == 1 {
			body["image"] = json!({"type":"image_url", "url":data_url(&request.input_images[0])});
		} else if edit {
			body["images"] = Value::Array(
				request
					.input_images
					.iter()
					.map(|image| json!({"type":"image_url", "url":data_url(image)}))
					.collect(),
			);
		}
		let payload = self
			.post_json(provider, credential, &url, &[], body)
			.await?;
		let value: Value =
			serde_json::from_slice(&payload.body).map_err(|error| parse_error(provider, error))?;
		let (images, revised) = self.openai_data(provider, &value).await?;
		finish(provider, request, images, revised, Str::default(), model)
	}

	async fn openrouter(
		&self,
		credential: &ImageCredential,
		request: &GenerateImageRequest,
	) -> Result<ImageDone, ImageAttemptError> {
		let provider = ImageProvider::OpenRouter;
		let base = endpoint(&self.providers, request, provider, "https://openrouter.ai/api/v1");
		let url = format!("{}/chat/completions", base.trim_end_matches('/'));
		let mut model = model(request, provider, "google/gemini-3-pro-image-preview");
		if !model.contains('/') {
			model = format!("google/{model}").into();
		}
		let mut content = vec![json!({"type":"text", "text":request.prompt})];
		for image in &request.input_images {
			content.push(json!({"type":"image_url", "image_url":{"url":data_url(image)}}));
		}
		let payload = self
			.post_json(
				provider,
				credential,
				&url,
				&[
					("http-referer", "https://omp.sh/".to_owned()),
					("x-openrouter-title", "omp".to_owned()),
					("x-openrouter-categories", "cli-agent".to_owned()),
				],
				json!({"model":model, "messages":[{"role":"user", "content":content}]}),
			)
			.await?;
		let value: Value =
			serde_json::from_slice(&payload.body).map_err(|error| parse_error(provider, error))?;
		let message = value.pointer("/choices/0/message").unwrap_or(&Value::Null);
		let mut texts = Vec::new();
		let mut urls = Vec::new();
		match message.get("content") {
			Some(Value::String(text)) if !text.trim().is_empty() => texts.push(text.as_str()),
			Some(Value::Array(parts)) => {
				for part in parts {
					if part.get("type").and_then(Value::as_str) == Some("text") {
						if let Some(text) = part.get("text").and_then(Value::as_str) {
							texts.push(text);
						}
					} else if part.get("type").and_then(Value::as_str) == Some("image_url")
						&& let Some(url) = part.pointer("/image_url/url").and_then(Value::as_str)
					{
						urls.push(url);
					}
				}
			},
			_ => {},
		}
		for entry in message
			.get("images")
			.and_then(Value::as_array)
			.into_iter()
			.flatten()
		{
			if let Some(url) = entry
				.as_str()
				.or_else(|| entry.pointer("/image_url/url").and_then(Value::as_str))
				.or_else(|| entry.get("url").and_then(Value::as_str))
			{
				urls.push(url);
			}
		}
		let mut images = Vec::with_capacity(urls.len());
		for url in urls {
			images.push(self.download(provider, url).await?);
		}
		let text = texts.join("\n");
		finish(provider, request, images, Str::default(), text.into(), model)
	}

	async fn gemini(
		&self,
		credential: &ImageCredential,
		request: &GenerateImageRequest,
	) -> Result<ImageDone, ImageAttemptError> {
		let provider = ImageProvider::Gemini;
		let base = endpoint(
			&self.providers,
			request,
			provider,
			"https://generativelanguage.googleapis.com/v1beta",
		);
		let model = model(request, provider, "gemini-3-pro-image-preview");
		let url = format!("{}/models/{}:generateContent", base.trim_end_matches('/'), model);
		let body = json!({
			"contents":[{"role":"user", "parts":gemini_parts(request)}],
			"generationConfig":{"responseModalities":["IMAGE"], "imageConfig":image_config(request)}
		});
		let payload = self
			.post_json(provider, credential, &url, &[], body)
			.await?;
		let value: Value =
			serde_json::from_slice(&payload.body).map_err(|error| parse_error(provider, error))?;
		let mut images = Vec::new();
		let mut texts = Vec::new();
		collect_gemini_parts(provider, &value, &mut images, &mut texts)?;
		finish(provider, request, images, Str::default(), texts.join(" ").into(), model)
	}

	async fn openai_data(
		&self,
		provider: ImageProvider,
		value: &Value,
	) -> Result<(Vec<BlobPart>, Str), ImageAttemptError> {
		let mut images = Vec::new();
		let mut revised = Str::default();
		for entry in value
			.get("data")
			.and_then(Value::as_array)
			.into_iter()
			.flatten()
		{
			if let Some(encoded) = entry.get("b64_json").and_then(Value::as_str) {
				images.push(decode_blob(provider, encoded, "image/png")?);
			} else if let Some(url) = entry.get("url").and_then(Value::as_str) {
				images.push(self.download(provider, url).await?);
			}
			if let Some(prompt) = entry.get("revised_prompt").and_then(Value::as_str) {
				revised = prompt.into();
			}
		}
		Ok((images, revised))
	}
}

#[async_trait]
impl ImageBackend for EgressImageBackend {
	async fn generate(
		&self,
		provider: ImageProvider,
		credential: &ImageCredential,
		request: &GenerateImageRequest,
	) -> Result<ImageDone, ImageAttemptError> {
		match provider {
			ImageProvider::OpenAi | ImageProvider::OpenAiCodex => {
				self.openai(provider, credential, request).await
			},
			ImageProvider::Antigravity => self.antigravity(credential, request).await,
			ImageProvider::Xai => self.xai(credential, request).await,
			ImageProvider::OpenRouter => self.openrouter(credential, request).await,
			ImageProvider::Gemini => self.gemini(credential, request).await,
		}
	}
}

struct HttpPayload {
	body:         Bytes,
	content_type: String,
}

/// Resolves an endpoint with explicit per-call override taking precedence over
/// the overlay-materialized provider catalog, then the built-in safety default.
fn endpoint(
	providers: &ProviderCatalog,
	request: &GenerateImageRequest,
	provider: ImageProvider,
	default: &str,
) -> String {
	request
		.props
		.get_ns(provider.id(), "base_url")
		.and_then(Value::as_str)
		.map(ToOwned::to_owned)
		.or_else(|| {
			providers
				.get(provider.catalog_id())
				.map(|entry| entry.base_url.to_string())
				.filter(|value| !value.is_empty())
		})
		.unwrap_or_else(|| default.to_owned())
}

fn model(request: &GenerateImageRequest, provider: ImageProvider, default: &str) -> Str {
	request
		.props
		.get_ns(provider.id(), "model")
		.and_then(Value::as_str)
		.map(Str::from)
		.or_else(|| (!request.model.is_empty()).then(|| request.model.clone()))
		.unwrap_or_else(|| default.into())
}

fn gemini_parts(request: &GenerateImageRequest) -> Vec<Value> {
	let mut parts = request.input_images.iter().map(|image| json!({"inlineData":{"data":base64::encode(&image.inline).into_string(), "mimeType":image.mime}})).collect::<Vec<_>>();
	parts.push(json!({"text":request.prompt}));
	parts
}

fn image_config(request: &GenerateImageRequest) -> Value {
	let mut config = serde_json::Map::new();
	if let Some(ratio) = aspect_ratio(request.aspect_ratio) {
		config.insert("aspectRatio".to_owned(), ratio.into());
	}
	if let Some(size) = request.size {
		config.insert("imageSize".to_owned(), size_name(size).into());
	}
	Value::Object(config)
}

fn aspect_ratio(ratio: Option<AspectRatio>) -> Option<&'static str> {
	Some(match ratio? {
		AspectRatio::Square => "1:1",
		AspectRatio::Wide16x9 => "16:9",
		AspectRatio::Tall9x16 => "9:16",
		AspectRatio::Landscape4x3 => "4:3",
		AspectRatio::Portrait3x4 => "3:4",
		AspectRatio::Landscape3x2 => "3:2",
		AspectRatio::Portrait2x3 => "2:3",
		AspectRatio::Ultrawide21x9 => "21:9",
		_ => return None,
	})
}

fn openai_size(request: &GenerateImageRequest) -> Option<String> {
	if let Some(size) = request.size {
		return Some(size_name(size));
	}
	Some(
		match request.aspect_ratio? {
			AspectRatio::Square => "1024x1024",
			AspectRatio::Portrait3x4 | AspectRatio::Tall9x16 => "1024x1536",
			AspectRatio::Landscape4x3 | AspectRatio::Wide16x9 => "1536x1024",
			_ => return None,
		}
		.to_owned(),
	)
}

fn size_name(size: ImageSize) -> String {
	format!("{}x{}", size.width, size.height)
}

const fn format_name(format: ImageFormat) -> &'static str {
	match format {
		ImageFormat::Png => "png",
		ImageFormat::Webp => "webp",
		ImageFormat::Jpeg => "jpeg",
		ImageFormat::Svg => "svg",
		_ => "webp",
	}
}

const fn format_mime(format: ImageFormat) -> &'static str {
	match format {
		ImageFormat::Png => "image/png",
		ImageFormat::Webp => "image/webp",
		ImageFormat::Jpeg => "image/jpeg",
		ImageFormat::Svg => "image/svg+xml",
		_ => "image/webp",
	}
}
fn data_url(image: &BlobPart) -> String {
	format!("data:{};base64,{}", image.mime, base64::encode(&image.inline))
}

fn parse_data_url(url: &str) -> Option<(&str, &str)> {
	let value = url.strip_prefix("data:")?;
	let (meta, data) = value.split_once(',')?;
	Some((meta.strip_suffix(";base64")?, data))
}

fn decode_blob(
	provider: ImageProvider,
	encoded: &str,
	mime: &str,
) -> Result<BlobPart, ImageAttemptError> {
	let bytes = base64::decode(encoded.as_bytes())
		.into_vec()
		.map_err(|error| parse_error(provider, error))?;
	Ok(blob(Bytes::from(bytes), mime))
}

fn blob(bytes: Bytes, mime: &str) -> BlobPart {
	BlobPart::builder()
		.hash(*blake3::hash(&bytes).as_bytes())
		.mime(Str::from(mime))
		.size(bytes.len() as u64)
		.inline(bytes)
		.build()
}

fn finish(
	provider: ImageProvider,
	request: &GenerateImageRequest,
	images: Vec<BlobPart>,
	revised_prompt: Str,
	text: Str,
	model: Str,
) -> Result<ImageDone, ImageAttemptError> {
	if images.is_empty() {
		return Err(parse_message(provider, "provider returned no image data"));
	}
	let mut props: Props = request.props.clone();
	props.insert_ns(provider.id(), "model", Value::String(model.to_string()));
	Ok(ImageDone::builder()
		.images(images)
		.revised_prompt(revised_prompt)
		.text(text)
		.unsupported(Vec::new())
		.props(props)
		.build())
}

fn collect_gemini_parts(
	provider: ImageProvider,
	value: &Value,
	images: &mut Vec<BlobPart>,
	texts: &mut Vec<String>,
) -> Result<(), ImageAttemptError> {
	for candidate in value
		.get("candidates")
		.and_then(Value::as_array)
		.into_iter()
		.flatten()
	{
		for part in candidate
			.pointer("/content/parts")
			.and_then(Value::as_array)
			.into_iter()
			.flatten()
		{
			if let Some(text) = part.get("text").and_then(Value::as_str) {
				texts.push(text.to_owned());
			}
			let inline = part.get("inlineData").or_else(|| part.get("inline_data"));
			if let Some(inline) = inline
				&& let (Some(data), Some(mime)) = (
					inline.get("data").and_then(Value::as_str),
					inline
						.get("mimeType")
						.or_else(|| inline.get("mime_type"))
						.and_then(Value::as_str),
				) {
				images.push(decode_blob(provider, data, mime)?);
			}
		}
	}
	Ok(())
}

fn sse_values(provider: ImageProvider, body: &[u8]) -> Result<Vec<Value>, ImageAttemptError> {
	let text = std::str::from_utf8(body).map_err(|error| parse_error(provider, error))?;
	text
		.lines()
		.filter_map(|line| line.strip_prefix("data:").map(str::trim))
		.filter(|data| !data.is_empty() && *data != "[DONE]")
		.map(|data| serde_json::from_str(data).map_err(|error| parse_error(provider, error)))
		.collect()
}

fn last_sse_response(provider: ImageProvider, body: &[u8]) -> Result<Value, ImageAttemptError> {
	let values = sse_values(provider, body)?;
	for event in values.iter().rev() {
		if matches!(
			event.get("type").and_then(Value::as_str),
			Some("response.completed" | "response.done")
		) && let Some(response) = event.get("response")
		{
			return Ok(response.clone());
		}
	}
	let output = values
		.iter()
		.filter(|event| {
			event.get("type").and_then(Value::as_str) == Some("response.output_item.done")
		})
		.filter_map(|event| event.get("item").cloned())
		.collect();
	Ok(json!({"output":Value::Array(output)}))
}

fn provider_message(body: &[u8]) -> Str {
	serde_json::from_slice::<Value>(body)
		.ok()
		.and_then(|value| {
			value
				.pointer("/error/message")
				.and_then(Value::as_str)
				.map(Str::from)
		})
		.unwrap_or_else(|| String::from_utf8_lossy(body).into_owned().into())
}

fn status_error(provider: ImageProvider, status: StatusCode, message: Str) -> ImageAttemptError {
	ImageProviderError {
		provider: provider.id().into(),
		kind: ImageProviderErrorKind::Status(status.as_u16()),
		message,
	}
	.into()
}

fn parse_error(provider: ImageProvider, error: impl std::fmt::Display) -> ImageAttemptError {
	parse_message(provider, error.to_string())
}
fn parse_message(provider: ImageProvider, message: impl Into<Str>) -> ImageAttemptError {
	ImageProviderError {
		provider: provider.id().into(),
		kind:     ImageProviderErrorKind::Parse,
		message:  message.into(),
	}
	.into()
}
fn transport_error(provider: ImageProvider, error: impl std::fmt::Display) -> ImageAttemptError {
	ImageProviderError {
		provider: provider.id().into(),
		kind:     ImageProviderErrorKind::Transport,
		message:  error.to_string().into(),
	}
	.into()
}
