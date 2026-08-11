//! Integration coverage for production image provider adapters.

use std::{collections::VecDeque, future, sync::Arc};

use async_trait::async_trait;
use bytes::Bytes;
use http::Request;
use omp_core::Str;
use omp_llm_catalog::provider::{ProviderCatalog, load_providers};
use omp_llm_egress::{
	auth_inject::{
		CredentialAuthKind, CredentialLease, CredentialMetadata, CredentialMetadataSource,
		CredentialSource,
	},
	client::{Body, EgressClient},
};
use omp_llm_gateway::{
	image_backends::EgressImageBackend,
	images::{
		ImageAttemptError, ImageBackend, ImageCredential, ImageCredentials, ImageProvider,
		ImageProviderError, ImageProviderErrorKind, ImageRegistry, ImageRegistryError,
		LeasedImageCredentials,
	},
};
use omp_llm_types::{BlobPart, GenerateImageRequest, ImageDone, Props};
use parking_lot::Mutex;
use serde_json::{Value, json};
use wiremock::{
	Mock, MockServer, ResponseTemplate,
	matchers::{method, path},
};

#[derive(Default)]
struct Credentials(Vec<(ImageProvider, ImageCredential)>);

impl ImageCredentials for Credentials {
	fn credential(&self, provider: ImageProvider) -> Option<ImageCredential> {
		self
			.0
			.iter()
			.find(|(candidate, _)| *candidate == provider)
			.map(|(_, value)| value.clone())
	}
}

fn credential() -> ImageCredential {
	ImageCredential {
		lease:         None,
		auth_provider: "test".into(),
		project_id:    Some("project".into()),
		account_id:    None,
	}
}

fn catalog_with_base(provider: &str, base_url: &str) -> Arc<ProviderCatalog> {
	let source = format!(
		r#"[providers."{provider}"]
transport = "open-ai-responses"
base_url = "{base_url}"
auth = {{ type = "none" }}
facets = ["chat"]
"#,
	);
	Arc::new(load_providers(&source).expect("provider overlay"))
}

fn catalog(provider: &str, server: &MockServer) -> Arc<ProviderCatalog> {
	catalog_with_base(provider, &server.uri())
}

fn request(provider: &str, inputs: Vec<BlobPart>) -> GenerateImageRequest {
	let mut props = Props::default();
	props.insert_ns("image", "provider", provider.into());
	GenerateImageRequest::builder()
		.model("test-image-model".into())
		.prompt("preserve this prompt exactly".into())
		.n(1)
		.input_images(inputs)
		.props(props)
		.build()
}

fn reference() -> BlobPart {
	let bytes = Bytes::from_static(b"reference-image");
	BlobPart::builder()
		.hash(*blake3::hash(&bytes).as_bytes())
		.mime(Str::new_static("image/png"))
		.size(bytes.len() as u64)
		.inline(bytes)
		.build()
}

#[tokio::test]
async fn gemini_decodes_inline_image_and_forwards_reference() {
	let server = MockServer::start().await;
	Mock::given(method("POST"))
		.and(path("/models/test-image-model:generateContent"))
		.respond_with(ResponseTemplate::new(200).set_body_json(json!({
			"candidates":[{"content":{"parts":[
				{"text":"provider commentary"},
				{"inlineData":{"data":"Z2VuZXJhdGVkLWltYWdl", "mimeType":"image/png"}}
			]}}]
		})))
		.mount(&server)
		.await;
	let backend = EgressImageBackend::new(
		EgressClient::new(std::time::Duration::from_secs(5)),
		catalog("google", &server),
	);
	let done = backend
		.generate(ImageProvider::Gemini, &credential(), &request("gemini", vec![reference()]))
		.await
		.expect("Gemini result");
	assert_eq!(done.images[0].inline, Bytes::from_static(b"generated-image"));
	assert_eq!(done.text, "provider commentary");
	let requests = server.received_requests().await.expect("recorded requests");
	let body: Value = serde_json::from_slice(&requests[0].body).expect("JSON request");
	assert_eq!(
		body
			.pointer("/contents/0/parts/0/inlineData/data")
			.and_then(Value::as_str),
		Some("cmVmZXJlbmNlLWltYWdl")
	);
	assert_eq!(
		body
			.pointer("/contents/0/parts/1/text")
			.and_then(Value::as_str),
		Some("preserve this prompt exactly")
	);
}

#[tokio::test]
async fn openai_uses_overlay_materialized_base_url() {
	let server = MockServer::start().await;
	Mock::given(method("POST"))
		.and(path("/openai/v1/responses"))
		.respond_with(ResponseTemplate::new(200).set_body_json(json!({
			"output":[{
				"type":"image_generation_call",
				"result":"b3ZlcmxheS1pbWFnZQ==",
				"revised_prompt":"overlay revised prompt"
			}]
		})))
		.mount(&server)
		.await;
	let backend = EgressImageBackend::new(
		EgressClient::new(std::time::Duration::from_secs(5)),
		catalog_with_base("openai", &format!("{}/openai/v1", server.uri())),
	);
	let done = backend
		.generate(ImageProvider::OpenAi, &credential(), &request("openai", Vec::new()))
		.await
		.expect("OpenAI overlay result");
	assert_eq!(done.images[0].inline, Bytes::from_static(b"overlay-image"));
	assert_eq!(done.revised_prompt, "overlay revised prompt");
}

#[tokio::test]
async fn xai_ingests_remote_url_and_preserves_revised_prompt() {
	let server = MockServer::start().await;
	let image_url = format!("{}/result.png", server.uri());
	Mock::given(method("POST"))
		.and(path("/images/generations"))
		.respond_with(ResponseTemplate::new(200).set_body_json(json!({
			"data":[{"url":image_url, "revised_prompt":"revised by provider"}]
		})))
		.mount(&server)
		.await;
	Mock::given(method("GET"))
		.and(path("/result.png"))
		.respond_with(
			ResponseTemplate::new(200)
				.insert_header("content-type", "image/png")
				.set_body_bytes(b"downloaded-image"),
		)
		.mount(&server)
		.await;
	let backend = EgressImageBackend::new(
		EgressClient::new(std::time::Duration::from_secs(5)),
		catalog("xai", &server),
	);
	let done = backend
		.generate(ImageProvider::Xai, &credential(), &request("xai", Vec::new()))
		.await
		.expect("xAI result");
	assert_eq!(done.images[0].inline, Bytes::from_static(b"downloaded-image"));
	assert_eq!(done.revised_prompt, "revised by provider");
}

#[tokio::test]
async fn antigravity_per_call_project_overrides_credential_metadata() {
	let server = MockServer::start().await;
	Mock::given(method("POST"))
		.and(path("/v1internal:streamGenerateContent"))
		.respond_with(ResponseTemplate::new(200).set_body_string(concat!(
			r#"data: {"response":{"candidates":[{"content":{"parts":[{"inlineData":{"data":"YW50aQ==","mimeType":"image/png"}}]}}]}}"#,
			"\n\n",
		)))
		.mount(&server)
		.await;
	let mut request = request("antigravity", Vec::new());
	request
		.props
		.insert_ns("antigravity", "project_id", "per-call-project".into());
	let backend = EgressImageBackend::new(
		EgressClient::new(std::time::Duration::from_secs(5)),
		catalog("google-antigravity", &server),
	);
	backend
		.generate(ImageProvider::Antigravity, &credential(), &request)
		.await
		.expect("Antigravity result");
	let requests = server.received_requests().await.expect("recorded requests");
	let body: Value = serde_json::from_slice(&requests[0].body).expect("JSON request");
	assert_eq!(body.get("project").and_then(Value::as_str), Some("per-call-project"));
}

struct ScriptedBackend {
	attempts: Mutex<Vec<ImageProvider>>,
	results:  Mutex<VecDeque<Result<ImageDone, ImageAttemptError>>>,
}

#[async_trait]
impl ImageBackend for ScriptedBackend {
	async fn generate(
		&self,
		provider: ImageProvider,
		_credential: &ImageCredential,
		_request: &GenerateImageRequest,
	) -> Result<ImageDone, ImageAttemptError> {
		self.attempts.lock().push(provider);
		self.results.lock().pop_front().expect("scripted result")
	}
}

fn success() -> ImageDone {
	ImageDone::builder()
		.images(vec![reference()])
		.revised_prompt(Str::default())
		.text(Str::default())
		.unsupported(Vec::new())
		.props(Props::default())
		.build()
}

fn retryable(provider: ImageProvider) -> ImageAttemptError {
	ImageProviderError {
		provider: provider.id().into(),
		kind:     ImageProviderErrorKind::Status(429),
		message:  "rate limited".into(),
	}
	.into()
}

#[tokio::test]
async fn ordered_fallback_advances_but_cancellation_is_a_hard_stop() {
	let credentials = Arc::new(Credentials(vec![
		(ImageProvider::OpenAi, ImageCredential {
			auth_provider: "openai".into(),
			..Default::default()
		}),
		(ImageProvider::Gemini, ImageCredential {
			auth_provider: "gemini".into(),
			..Default::default()
		}),
	]));
	let fallback_backend = Arc::new(ScriptedBackend {
		attempts: Mutex::new(Vec::new()),
		results:  Mutex::new(VecDeque::from([Err(retryable(ImageProvider::OpenAi)), Ok(success())])),
	});
	let registry = ImageRegistry::new(credentials.clone(), fallback_backend.clone())
		.with_configured_order(["openai", "gemini"]);
	let mut unpinned = request("openai", Vec::new());
	unpinned.props = Props::default();
	registry
		.execute(unpinned.clone())
		.await
		.expect("fallback result");
	assert_eq!(*fallback_backend.attempts.lock(), vec![
		ImageProvider::OpenAi,
		ImageProvider::Gemini
	]);

	let cancelled_backend = Arc::new(ScriptedBackend {
		attempts: Mutex::new(Vec::new()),
		results:  Mutex::new(VecDeque::from([Err(ImageAttemptError::Cancelled), Ok(success())])),
	});
	let registry = ImageRegistry::new(credentials, cancelled_backend.clone())
		.with_configured_order(["openai", "gemini"]);
	assert!(matches!(registry.execute(unpinned).await, Err(ImageRegistryError::Cancelled)));
	assert_eq!(*cancelled_backend.attempts.lock(), vec![ImageProvider::OpenAi]);
}

#[derive(Clone)]
struct MetadataSource;

impl CredentialSource for MetadataSource {
	type Error = std::io::Error;

	fn lease(&self, provider: &str) -> Result<Option<CredentialLease>, Self::Error> {
		Ok(Some(CredentialLease::new(provider, 7, 3)))
	}

	fn apply(
		&self,
		_lease: &CredentialLease,
		_request: &mut Request<Body>,
	) -> Result<(), Self::Error> {
		Ok(())
	}

	// The trait requires a `'static` future; `ready` preserves that contract
	// without allocation.
	#[allow(clippy::manual_async_fn, reason = "retain an allocation-free `static` ready future")]
	fn refresh(
		&self,
		lease: CredentialLease,
	) -> impl Future<Output = Result<CredentialLease, Self::Error>> + Send + 'static {
		future::ready(Ok(lease))
	}
}

impl CredentialMetadataSource for MetadataSource {
	fn metadata(&self, lease: &CredentialLease) -> Result<CredentialMetadata, Self::Error> {
		assert_eq!(lease.generation(), 3);
		Ok(CredentialMetadata {
			auth_kind:       CredentialAuthKind::OAuth,
			identity:        "user@example.com".into(),
			account_id:      Some("account".into()),
			project_id:      Some("credential-project".into()),
			organization_id: None,
		})
	}
}

#[test]
fn leased_admission_resolves_generation_validated_project_metadata() {
	let credential = LeasedImageCredentials::new(MetadataSource)
		.credential(ImageProvider::Antigravity)
		.expect("credential metadata");
	assert_eq!(credential.project_id.as_deref(), Some("credential-project"));
	assert_eq!(credential.account_id.as_deref(), Some("account"));
}
