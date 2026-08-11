//! Model listing projections for `OpenAI` and Anthropic SDKs.
//!
//! This is deliberately one route with two representations. An explicit
//! per-listener [`ModelsRepresentation`](super::ModelsRepresentation) override
//! wins. In `Auto`, the presence of `anthropic-version` selects Anthropic (the
//! Anthropic SDK always sends it; `x-api-key` is only corroborating evidence),
//! and its absence selects `OpenAI`.

use std::{fmt::Display, sync::Arc};

use bytes::Bytes;
use http::{HeaderMap, Request, StatusCode};
use hyper::body::Body;
use omp_core::Str;
use omp_llm_catalog::{models::ModelCard, provider::Facet as CatalogFacet, registry::ListFilter};
use serde::Serialize;

use super::{FacadeResponse, FacadeState, ModelsRepresentation, json_response};

#[derive(Serialize)]
struct OpenAiList {
	object: &'static str,
	data:   Vec<OpenAiModel>,
}

#[derive(Serialize)]
struct OpenAiModel {
	id:       Str,
	object:   &'static str,
	created:  u64,
	owned_by: Str,
}

#[derive(Serialize)]
struct AnthropicList {
	data:     Vec<AnthropicModel>,
	has_more: bool,
	first_id: Option<Str>,
	last_id:  Option<Str>,
}

#[derive(Serialize)]
struct AnthropicModel {
	id:           Str,
	display_name: Str,
	created_at:   String,
	#[serde(rename = "type")]
	kind:         &'static str,
}

pub(crate) fn handle<B>(request: Request<B>, state: Arc<FacadeState>) -> FacadeResponse
where
	B: Body<Data = Bytes> + Send + 'static,
	B::Error: Display,
{
	let representation =
		select_representation(state.config.models_representation, request.headers());
	let filter = ListFilter::builder()
		.facet(CatalogFacet::Chat)
		.available_only(true)
		.build();
	let (models, _) = state.registry.read().list(&filter);
	match representation {
		ModelsRepresentation::Anthropic => anthropic_response(models),
		ModelsRepresentation::OpenAi | ModelsRepresentation::Auto => openai_response(models),
	}
}

fn select_representation(
	override_: ModelsRepresentation,
	headers: &HeaderMap,
) -> ModelsRepresentation {
	match override_ {
		ModelsRepresentation::Auto if headers.contains_key("anthropic-version") => {
			ModelsRepresentation::Anthropic
		},
		ModelsRepresentation::Auto => ModelsRepresentation::OpenAi,
		explicit => explicit,
	}
}

fn openai_response(models: Vec<ModelCard>) -> FacadeResponse {
	let data = models
		.into_iter()
		.map(|model| OpenAiModel {
			id:       model.id,
			object:   "model",
			created:  model.updated_at_ms / 1000,
			owned_by: model.provider,
		})
		.collect();
	json_response(StatusCode::OK, &OpenAiList { object: "list", data })
}

fn anthropic_response(models: Vec<ModelCard>) -> FacadeResponse {
	let data: Vec<_> = models
		.into_iter()
		.map(|model| AnthropicModel {
			id:           model.id,
			display_name: if model.name.is_empty() {
				model.model
			} else {
				model.name
			},
			created_at:   rfc3339(model.updated_at_ms),
			kind:         "model",
		})
		.collect();
	let first_id = data.first().map(|model| model.id.clone());
	let last_id = data.last().map(|model| model.id.clone());
	json_response(StatusCode::OK, &AnthropicList { data, has_more: false, first_id, last_id })
}

fn rfc3339(unix_ms: u64) -> String {
	let seconds = unix_ms / 1000;
	let days = seconds / 86_400;
	let day_seconds = seconds % 86_400;
	let (year, month, day) = civil_from_days(days as i64);
	let hour = day_seconds / 3600;
	let minute = day_seconds % 3600 / 60;
	let second = day_seconds % 60;
	format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

fn civil_from_days(days_since_epoch: i64) -> (i64, u32, u32) {
	let z = days_since_epoch + 719_468;
	let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
	let day_of_era = z - era * 146_097;
	let year_of_era =
		(day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
	let mut year = year_of_era + era * 400;
	let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
	let month_prime = (5 * day_of_year + 2) / 153;
	let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
	let month = month_prime + if month_prime < 10 { 3 } else { -9 };
	year += i64::from(month <= 2);
	(year, month as u32, day as u32)
}

#[cfg(test)]
mod tests {
	use std::sync::Arc;

	use http::HeaderValue;
	use http_body_util::{BodyExt, Full};
	use omp_llm_catalog::{
		models::Availability,
		registry::{CredentialView, Registry},
	};
	use omp_llm_types::facet::Facets;
	use omp_storage::blob::BlobStore;

	use super::*;
	use crate::facade::{FacadeAuth, FacadeConfig};

	struct Credentials;

	impl CredentialView for Credentials {
		fn availability(&self, _provider: &str) -> Availability {
			Availability::Available
		}
	}

	fn state(
		directory: &std::path::Path,
		models_representation: ModelsRepresentation,
	) -> Arc<FacadeState> {
		Arc::new(FacadeState {
			facets:   Arc::new(Facets::default()),
			registry: Arc::new(parking_lot::RwLock::new(Registry::from_cards(
				&[],
				Arc::new(Credentials),
			))),
			blobs:    Arc::new(BlobStore::open(directory).expect("blob store")),
			auth:     FacadeAuth::new("token"),
			config:   FacadeConfig { models_representation },
		})
	}

	#[test]
	fn auto_selects_anthropic_when_version_header_is_present() {
		let mut headers = HeaderMap::new();
		headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
		assert_eq!(
			select_representation(ModelsRepresentation::Auto, &headers),
			ModelsRepresentation::Anthropic
		);
	}

	#[test]
	fn auto_selects_openai_without_anthropic_version() {
		let mut headers = HeaderMap::new();
		headers.insert("x-api-key", HeaderValue::from_static("gateway-token"));
		assert_eq!(
			select_representation(ModelsRepresentation::Auto, &headers),
			ModelsRepresentation::OpenAi
		);
	}

	#[tokio::test]
	async fn models_route_uses_header_sniff_and_listener_override() {
		let directory = tempfile::tempdir().expect("temporary directory");

		let anthropic_request = Request::get("/v1/models")
			.header("anthropic-version", "2023-06-01")
			.body(Full::new(Bytes::new()))
			.expect("request");
		let response = handle(anthropic_request, state(directory.path(), ModelsRepresentation::Auto));
		let body = response
			.into_body()
			.collect()
			.await
			.expect("infallible")
			.to_bytes();
		let value: serde_json::Value = serde_json::from_slice(&body).expect("JSON");
		assert!(value.get("has_more").is_some());
		assert!(value.get("object").is_none());

		let openai_request = Request::get("/v1/models")
			.body(Full::new(Bytes::new()))
			.expect("request");
		let response = handle(openai_request, state(directory.path(), ModelsRepresentation::Auto));
		let body = response
			.into_body()
			.collect()
			.await
			.expect("infallible")
			.to_bytes();
		let value: serde_json::Value = serde_json::from_slice(&body).expect("JSON");
		assert_eq!(value["object"], "list");
		assert!(value.get("has_more").is_none());

		let overridden_request = Request::get("/v1/models")
			.header("anthropic-version", "2023-06-01")
			.body(Full::new(Bytes::new()))
			.expect("request");
		let response =
			handle(overridden_request, state(directory.path(), ModelsRepresentation::OpenAi));
		let body = response
			.into_body()
			.collect()
			.await
			.expect("infallible")
			.to_bytes();
		let value: serde_json::Value = serde_json::from_slice(&body).expect("JSON");
		assert_eq!(value["object"], "list");
	}
	#[test]
	fn explicit_override_wins_over_header_sniff() {
		let mut headers = HeaderMap::new();
		headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
		assert_eq!(
			select_representation(ModelsRepresentation::OpenAi, &headers),
			ModelsRepresentation::OpenAi
		);
		assert_eq!(
			select_representation(ModelsRepresentation::Anthropic, &HeaderMap::new()),
			ModelsRepresentation::Anthropic
		);
	}

	#[test]
	fn anthropic_dates_are_rfc3339_utc() {
		assert_eq!(rfc3339(0), "1970-01-01T00:00:00Z");
		assert_eq!(rfc3339(1_704_067_200_000), "2024-01-01T00:00:00Z");
	}
}
