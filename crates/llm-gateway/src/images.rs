//! Provider selection and ordered fallback for production image generation.
//!
//! The explicit per-call `antigravity/project_id` property takes precedence
//! over the selected lease's generation-validated non-secret project metadata.

use std::sync::Arc;

use async_trait::async_trait;
use futures::{stream, stream::BoxStream};
use omp_core::Str;
use omp_llm_egress::auth_inject::{CredentialLease, CredentialMetadataSource};
use omp_llm_types::{GenerateImageRequest, ImageDone, ImageEvent, facet};
use smallvec::SmallVec;

/// Image providers in the default Pi-compatible attempt order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageProvider {
	/// `OpenAI` Responses image-generation tool.
	OpenAi,
	/// ChatGPT/Codex Responses image-generation tool.
	OpenAiCodex,
	/// Google Cloud Code Antigravity image generation.
	Antigravity,
	/// xAI Grok Imagine images API.
	Xai,
	/// `OpenRouter` chat-completions image output.
	OpenRouter,
	/// Google Gemini generate-content image output.
	Gemini,
}

impl ImageProvider {
	/// Stable configuration identifier.
	#[must_use]
	pub const fn id(self) -> &'static str {
		match self {
			Self::OpenAi => "openai",
			Self::OpenAiCodex => "openai-codex",
			Self::Antigravity => "antigravity",
			Self::Xai => "xai",
			Self::OpenRouter => "openrouter",
			Self::Gemini => "gemini",
		}
	}

	/// Canonical provider catalog and broker credential identifier.
	#[must_use]
	pub const fn catalog_id(self) -> &'static str {
		match self {
			Self::OpenAi => "openai",
			Self::OpenAiCodex => "openai-codex",
			Self::Antigravity => "google-antigravity",
			Self::Xai => "xai",
			Self::OpenRouter => "openrouter",
			Self::Gemini => "google",
		}
	}

	/// Parses a stable provider identifier.
	#[must_use]
	pub fn from_id(id: &str) -> Option<Self> {
		Some(match id {
			"openai" => Self::OpenAi,
			"openai-codex" => Self::OpenAiCodex,
			"antigravity" => Self::Antigravity,
			"xai" => Self::Xai,
			"openrouter" => Self::OpenRouter,
			"gemini" => Self::Gemini,
			_ => return None,
		})
	}
}

/// Pi-compatible automatic provider order.
pub const IMAGE_PROVIDER_ORDER: [ImageProvider; 6] = [
	ImageProvider::OpenAi,
	ImageProvider::OpenAiCodex,
	ImageProvider::Antigravity,
	ImageProvider::Xai,
	ImageProvider::OpenRouter,
	ImageProvider::Gemini,
];

/// Non-secret credential lease and public provider metadata for one image path.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ImageCredential {
	/// Opaque credential generation selected for this attempt.
	pub lease:         Option<CredentialLease>,
	/// Provider id supplied to the authentication injection layer.
	pub auth_provider: Str,
	/// Google Cloud project for Antigravity.
	pub project_id:    Option<Str>,
	/// `ChatGPT` account id used by the Codex endpoint.
	pub account_id:    Option<Str>,
}

/// Read-only non-secret credential lease source used to admit candidates.
pub trait ImageCredentials: Send + Sync {
	/// Returns an opaque lease and public metadata when a provider is
	/// configured.
	fn credential(&self, provider: ImageProvider) -> Option<ImageCredential>;
}

/// Adapts canonical credential leases and validated metadata to image
/// admission.
#[derive(Clone)]
pub struct LeasedImageCredentials<C> {
	source: C,
}

impl<C> LeasedImageCredentials<C> {
	/// Creates image admission backed by canonical credential leases.
	#[must_use]
	pub const fn new(source: C) -> Self {
		Self { source }
	}
}

impl<C> ImageCredentials for LeasedImageCredentials<C>
where
	C: CredentialMetadataSource,
{
	fn credential(&self, provider: ImageProvider) -> Option<ImageCredential> {
		let lease = self.source.lease(provider.catalog_id()).ok()??;
		let metadata = self.source.metadata(&lease).ok()?;
		Some(ImageCredential {
			auth_provider: lease.provider().into(),
			lease: Some(lease),
			project_id: metadata.project_id,
			account_id: metadata.account_id,
			..ImageCredential::default()
		})
	}
}

/// Provider failure category controlling fallback safety.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageProviderErrorKind {
	/// Upstream HTTP status.
	Status(u16),
	/// Provider response could not be decoded or contained no image.
	Parse,
	/// Transport failure with unknown request commit state.
	Transport,
}

/// One provider's image-generation failure.
#[derive(Clone, Debug, thiserror::Error)]
#[error("{provider}: {message}")]
pub struct ImageProviderError {
	/// Provider that failed.
	pub provider: Str,
	/// Failure classification.
	pub kind:     ImageProviderErrorKind,
	/// Caller-safe diagnostic.
	pub message:  Str,
}

impl ImageProviderError {
	/// Whether policy may advance to the next configured provider.
	#[must_use]
	pub const fn permits_fallback(&self) -> bool {
		matches!(self.kind, ImageProviderErrorKind::Status(_) | ImageProviderErrorKind::Parse)
	}
}

/// Result of one provider adapter attempt.
#[derive(Clone, Debug, thiserror::Error)]
pub enum ImageAttemptError {
	/// Caller cancellation is a hard stop and is never converted into fallback.
	#[error("image generation cancelled")]
	Cancelled,
	/// Provider failure.
	#[error(transparent)]
	Provider(#[from] ImageProviderError),
}

/// Adapter boundary for the six distinct image wire protocols.
#[async_trait]
pub trait ImageBackend: Send + Sync {
	/// Executes one provider attempt.
	async fn generate(
		&self,
		provider: ImageProvider,
		credential: &ImageCredential,
		request: &GenerateImageRequest,
	) -> Result<ImageDone, ImageAttemptError>;
}

/// Image registry failure.
#[derive(Clone, Debug, thiserror::Error)]
pub enum ImageRegistryError {
	/// A request pinned an unknown provider.
	#[error("unknown image provider: {0}")]
	UnknownProvider(Str),
	/// No provider in the candidate chain has credentials.
	#[error("no configured image provider")]
	NoAvailableProvider,
	/// Cancellation stopped the chain.
	#[error("image generation cancelled")]
	Cancelled,
	/// A non-fallback-safe error stopped the chain.
	#[error(transparent)]
	Provider(ImageProviderError),
	/// Every credentialed provider failed safely.
	#[error("all image providers failed")]
	AllProvidersFailed {
		/// Failures in attempt order.
		failures: Box<SmallVec<ImageProviderError, 6>>,
	},
}

/// Credential-aware image provider selection and fallback.
pub struct ImageRegistry {
	credentials:      Arc<dyn ImageCredentials>,
	backend:          Arc<dyn ImageBackend>,
	configured_order: SmallVec<ImageProvider, 6>,
}

impl ImageRegistry {
	/// Creates an image registry with Pi's default provider order.
	#[must_use]
	pub fn new(credentials: Arc<dyn ImageCredentials>, backend: Arc<dyn ImageBackend>) -> Self {
		Self { credentials, backend, configured_order: SmallVec::new() }
	}

	/// Adds a configured priority prefix, ignoring unknown and duplicate ids.
	#[must_use]
	pub fn with_configured_order<I, S>(mut self, order: I) -> Self
	where
		I: IntoIterator<Item = S>,
		S: AsRef<str>,
	{
		for id in order {
			if let Some(provider) = ImageProvider::from_id(id.as_ref())
				&& !self.configured_order.contains(&provider)
			{
				self.configured_order.push(provider);
			}
		}
		self
	}

	/// Resolves the credentialed candidate order, including a per-call override.
	pub fn candidates(
		&self,
		request: &GenerateImageRequest,
	) -> Result<SmallVec<ImageProvider, 6>, ImageRegistryError> {
		let pinned = request
			.props
			.get_ns("image", "provider")
			.or_else(|| request.props.get_ns("omp", "image_provider"))
			.and_then(serde_json::Value::as_str);
		let mut ordered = SmallVec::new();
		if let Some(id) = pinned {
			ordered.push(
				ImageProvider::from_id(id)
					.ok_or_else(|| ImageRegistryError::UnknownProvider(id.into()))?,
			);
		}
		for provider in self
			.configured_order
			.iter()
			.chain(IMAGE_PROVIDER_ORDER.iter())
		{
			if !ordered.contains(provider) {
				ordered.push(*provider);
			}
		}
		ordered.retain(|provider| self.credentials.credential(*provider).is_some());
		Ok(ordered)
	}

	/// Executes a request with ordered fallback and cancellation hard-stop.
	pub async fn execute(
		&self,
		request: GenerateImageRequest,
	) -> Result<ImageDone, ImageRegistryError> {
		let candidates = self.candidates(&request)?;
		if candidates.is_empty() {
			return Err(ImageRegistryError::NoAvailableProvider);
		}
		let mut failures = SmallVec::new();
		for provider in candidates {
			let Some(credential) = self.credentials.credential(provider) else {
				continue;
			};
			match self.backend.generate(provider, &credential, &request).await {
				Ok(mut done) => {
					done
						.props
						.insert_ns("image", "provider", provider.id().into());
					return Ok(done);
				},
				Err(ImageAttemptError::Cancelled) => return Err(ImageRegistryError::Cancelled),
				Err(ImageAttemptError::Provider(error)) if error.permits_fallback() => {
					failures.push(error);
				},
				Err(ImageAttemptError::Provider(error)) => {
					return Err(ImageRegistryError::Provider(error));
				},
			}
		}
		Err(ImageRegistryError::AllProvidersFailed { failures: Box::new(failures) })
	}
}

#[async_trait]
impl facet::ImageGen for ImageRegistry {
	async fn generate(
		&self,
		request: GenerateImageRequest,
	) -> Result<BoxStream<'static, ImageEvent>, facet::Error> {
		let done = self
			.execute(request)
			.await
			.map_err(|error| facet::Error::Provider(error.to_string().into()))?;
		Ok(Box::pin(stream::once(async move { ImageEvent::Done(done) })))
	}
}
