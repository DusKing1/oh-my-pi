//! Vertex AI deployment routing and provider-status classification.

use omp_core::Str;
use omp_llm_catalog::provider::{BaseUrlVars, ProviderEntry, expand_base_url};
use omp_llm_types::{Error, TurnErrorKind};

use crate::adc::AdcRoute;

const STREAM_ACTION: &str = "streamGenerateContent";

/// Resolved, non-secret Vertex deployment coordinates.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VertexDeployment {
	project:  String,
	location: String,
}

impl VertexDeployment {
	/// Validates deployment coordinates discovered by the ADC engine.
	pub fn new(project: impl Into<String>, location: impl Into<String>) -> Result<Self, Error> {
		let project = project.into();
		let location = location.into();
		validate_segment("project", &project)?;
		validate_segment("location", &location)?;
		Ok(Self { project, location })
	}

	/// Creates a deployment from ADC route discovery.
	pub fn from_adc(route: AdcRoute) -> Result<Self, Error> {
		Self::new(route.project, route.location)
	}

	/// Returns the selected project id.
	#[must_use]
	pub fn project(&self) -> &str {
		&self.project
	}

	/// Returns the selected location.
	#[must_use]
	pub fn location(&self) -> &str {
		&self.location
	}

	/// Builds the canonical Vertex publisher/model streaming endpoint.
	pub fn stream_endpoint(&self, model: &str) -> Result<Str, Error> {
		validate_segment("model", model)?;
		let host = if self.location == "global" {
			"aiplatform.googleapis.com".to_owned()
		} else {
			format!("{}-aiplatform.googleapis.com", self.location)
		};
		Ok(Str::from(format!(
			"https://{host}/v1/projects/{}/locations/{}/publishers/google/models/{model}:{STREAM_ACTION}?alt=sse",
			self.project, self.location
		)))
	}
}

/// Expands a catalog Vertex base URL and appends its ADC publisher/model path.
///
/// This variant preserves an operator-supplied base URL while using the same
/// validated path shape as [`VertexDeployment::stream_endpoint`].
pub fn vertex_stream_url(
	provider: &ProviderEntry,
	project: &str,
	region: &str,
	model: &str,
) -> Result<Str, Error> {
	validate_segment("project", project)?;
	validate_segment("location", region)?;
	validate_segment("model", model)?;
	let base_template = if region == "global" {
		Str::from(
			provider
				.base_url
				.replace(
					concat!("{", "location}-aiplatform.googleapis.com"),
					"aiplatform.googleapis.com",
				)
				.replace(
					concat!("{", "region}-aiplatform.googleapis.com"),
					"aiplatform.googleapis.com",
				),
		)
	} else {
		provider.base_url.clone()
	};
	let base = expand_base_url(
		&base_template,
		BaseUrlVars::builder()
			.maybe_region(Some(region))
			.maybe_location(Some(region))
			.build(),
	)
	.map_err(provider_error)?;
	Ok(Str::from(format!(
		"{}/projects/{project}/locations/{region}/publishers/google/models/{model}:{STREAM_ACTION}?\
		 alt=sse",
		base.trim_end_matches('/')
	)))
}

/// Classifies a Vertex HTTP status before any provider error detail is logged.
///
/// Both authentication failure and permission denial are credential-route
/// failures: retries must not rotate into anonymous or API-key behavior.
#[must_use]
pub const fn classify_status(status: u16) -> TurnErrorKind {
	match status {
		401 | 403 => TurnErrorKind::Auth,
		429 => TurnErrorKind::RateLimited,
		_ => TurnErrorKind::Upstream,
	}
}

fn validate_segment(name: &'static str, value: &str) -> Result<(), Error> {
	if value.is_empty()
		|| !value
			.bytes()
			.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
	{
		return Err(Error::Provider(Str::from(format!("invalid Vertex {name}"))));
	}
	Ok(())
}

#[cold]
fn provider_error(error: impl std::fmt::Display) -> Error {
	Error::Provider(Str::from(error.to_string()))
}

#[cfg(test)]
mod tests {
	use omp_llm_catalog::provider::load_builtin;

	use super::*;

	#[test]
	fn regional_and_global_endpoints_use_exact_publisher_paths() {
		let regional = VertexDeployment::new("my-project", "us-central1").unwrap();
		assert_eq!(
			regional.stream_endpoint("gemini-2.5-pro").unwrap(),
			"https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-2.5-pro:streamGenerateContent?alt=sse"
		);
		let global = VertexDeployment::new("my-project", "global").unwrap();
		assert_eq!(
			global.stream_endpoint("gemini-2.5-pro").unwrap(),
			"https://aiplatform.googleapis.com/v1/projects/my-project/locations/global/publishers/google/models/gemini-2.5-pro:streamGenerateContent?alt=sse"
		);
	}

	#[test]
	fn catalog_global_endpoint_removes_the_regional_host_prefix() {
		let providers = load_builtin().unwrap();
		let provider = &providers["google-vertex"];
		assert_eq!(
			vertex_stream_url(provider, "my-project", "global", "gemini-2.5-pro").unwrap(),
			"https://aiplatform.googleapis.com/v1/projects/my-project/locations/global/publishers/google/models/gemini-2.5-pro:streamGenerateContent?alt=sse"
		);
	}

	#[test]
	fn endpoint_segments_cannot_escape_the_publisher_path() {
		assert!(VertexDeployment::new("project/other", "global").is_err());
		assert!(VertexDeployment::new("project", "us-central1?key=secret").is_err());
		assert!(
			VertexDeployment::new("project", "global")
				.unwrap()
				.stream_endpoint("publishers/attacker/model")
				.is_err()
		);
	}

	#[test]
	fn unauthorized_and_forbidden_are_classified_as_auth() {
		assert_eq!(classify_status(401), TurnErrorKind::Auth);
		assert_eq!(classify_status(403), TurnErrorKind::Auth);
		assert_eq!(classify_status(429), TurnErrorKind::RateLimited);
		assert_eq!(classify_status(500), TurnErrorKind::Upstream);
	}
}
