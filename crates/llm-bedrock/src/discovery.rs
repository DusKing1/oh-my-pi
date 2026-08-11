//! Amazon Bedrock foundation-model discovery.
//!
//! Listing is served by the `bedrock` control plane, a different AWS service
//! from the `bedrock-runtime` data plane the provider row points at, so the
//! runtime host is rewritten rather than reused.
//!
//! Signing is not performed here. The protocol attaches the non-secret
//! [`AwsSigV4Context`]; the broker observes it during credential redemption and
//! signs the final URI, headers, and body immediately before dispatch, so this
//! module never sees key material.

use std::{collections::BTreeMap, time::SystemTime};

use async_trait::async_trait;
use bytes::Bytes;
use http::{Method, Request, header::ACCEPT};
use omp_llm_catalog::{
	discovery::{Account, DiscoveryHttp, DiscoveryProtocol, Error, HttpResponse, discovered_card},
	models::{Modality, ModelCard},
	provider::{BaseUrlVars, ProviderEntry, TransportId, expand_base_url},
};
use omp_llm_egress::auth_inject::AwsSigV4Context;

/// AWS service name used in the `SigV4` credential scope for model listing.
const SIGV4_SERVICE: &str = "bedrock";

/// Region used when neither the runtime nor the endpoint names one, matching
/// the AWS SDK default.
const DEFAULT_REGION: &str = "us-east-1";

/// Bedrock's `ListFoundationModels` control-plane protocol.
pub struct BedrockDiscovery;

#[async_trait]
impl DiscoveryProtocol for BedrockDiscovery {
	fn transports(&self) -> &'static [TransportId] {
		&[TransportId::BedrockConverse]
	}

	async fn discover(
		&self,
		provider: &ProviderEntry,
		account: &Account,
		http: &dyn DiscoveryHttp,
	) -> Result<Vec<ModelCard>, Error> {
		let region = region(provider, account);
		let mut request = Request::builder()
			.method(Method::GET)
			.uri(list_endpoint(provider, &region)?)
			.header(ACCEPT, "application/json")
			.body(Bytes::new())
			.map_err(Error::transport)?;
		request.extensions_mut().insert(AwsSigV4Context {
			service:   SIGV4_SERVICE.into(),
			region:    region.into(),
			signed_at: SystemTime::now(),
		});
		let response: HttpResponse = http.execute(provider, account, request).await?;
		parse_foundation_models(provider, response.ensure_success(provider)?)
	}
}

/// Registered by the application at daemon start-up.
pub static DISCOVERY: BedrockDiscovery = BedrockDiscovery;

/// Resolves the signing region for a listing call.
///
/// The runtime supplies the region it read from the environment; a provider row
/// whose `base_url` was overridden to a concrete regional host is the next
/// authority, and the SDK-compatible default is last.
fn region(provider: &ProviderEntry, account: &Account) -> String {
	if let Some(region) = account.region.as_deref().filter(|value| !value.is_empty()) {
		return region.to_owned();
	}
	endpoint_region(provider.base_url.as_str())
		.map_or_else(|| DEFAULT_REGION.to_owned(), str::to_owned)
}

/// Extracts the region from a concrete regional Bedrock host.
///
/// A row still carrying the `{region}` placeholder yields `None`.
fn endpoint_region(base_url: &str) -> Option<&str> {
	let host = base_url
		.split_once("://")
		.map_or(base_url, |(_, rest)| rest)
		.split(['/', ':'])
		.next()?;
	let region = host
		.split('.')
		.nth(1)
		.filter(|region| !region.is_empty() && !region.contains('{'))?;
	host.starts_with("bedrock").then_some(region)
}

/// Builds the control-plane listing URL for `region`.
fn list_endpoint(provider: &ProviderEntry, region: &str) -> Result<String, Error> {
	let base = expand_base_url(
		provider.base_url.as_str(),
		BaseUrlVars::builder()
			.region(region)
			.location(region)
			.build(),
	)
	.map_err(Error::transport)?;
	let (scheme, rest) = base
		.as_str()
		.split_once("://")
		.ok_or_else(|| Error::payload(provider, "Bedrock base URL has no scheme"))?;
	let host = rest.split(['/', '?']).next().unwrap_or(rest);
	// `bedrock-runtime[-fips].<region>.<suffix>` ->
	// `bedrock[-fips].<region>.<suffix>`.
	let host = host
		.strip_prefix("bedrock-runtime")
		.map_or_else(|| host.to_owned(), |tail| format!("bedrock{tail}"));
	Ok(format!("{scheme}://{host}/foundation-models"))
}

/// Parses a `ListFoundationModels` response into discovered cards.
///
/// Models are dropped only on explicit disqualifying evidence: a non-`ACTIVE`
/// lifecycle, provisioned-only inference, no text output, or a declared
/// absence of response streaming, which `ConverseStream` requires. Absent and
/// empty fields are permissive.
///
/// `inferenceTypesSupported` is `ON_DEMAND | PROVISIONED` only. A model
/// reachable exclusively through a cross-region inference profile reports
/// neither, so it is listed under its foundation id; resolving the invocable
/// `<geo>.<model>` profile id would take a separate `ListInferenceProfiles`
/// call, which this protocol does not make.
///
/// # Errors
///
/// Returns [`Error::InvalidPayload`] when the response has no `modelSummaries`.
pub fn parse_foundation_models(
	provider: &ProviderEntry,
	body: &[u8],
) -> Result<Vec<ModelCard>, Error> {
	let payload: serde_json::Value =
		serde_json::from_slice(body).map_err(|error| Error::payload(provider, error))?;
	let summaries = payload
		.get("modelSummaries")
		.and_then(serde_json::Value::as_array)
		.ok_or_else(|| Error::payload(provider, "missing modelSummaries array"))?;
	let mut cards = BTreeMap::new();
	for summary in summaries {
		let Some(model) = summary
			.get("modelId")
			.and_then(serde_json::Value::as_str)
			.filter(|value| !value.is_empty())
		else {
			continue;
		};
		if !is_usable(summary) {
			continue;
		}
		let name = summary
			.get("modelName")
			.and_then(serde_json::Value::as_str)
			.filter(|value| !value.is_empty())
			.unwrap_or(model);
		let family = summary
			.get("providerName")
			.and_then(serde_json::Value::as_str)
			.filter(|value| !value.is_empty())
			.unwrap_or_else(|| model.split('.').next().unwrap_or(model));
		let mut card = discovered_card(provider, model, name, family);
		if has_modality(summary, "inputModalities", "IMAGE") {
			card.inputs.push(Modality::Image);
		}
		cards.insert(card.id.clone(), card);
	}
	Ok(cards.into_values().collect())
}

/// Returns whether a summary describes a model this transport can invoke.
fn is_usable(summary: &serde_json::Value) -> bool {
	let lifecycle_active = summary
		.pointer("/modelLifecycle/status")
		.and_then(serde_json::Value::as_str)
		.is_none_or(|status| status == "ACTIVE");
	let streams = summary
		.get("responseStreamingSupported")
		.and_then(serde_json::Value::as_bool)
		.unwrap_or(true);
	let invocable = string_list(summary, "inferenceTypesSupported")
		.is_none_or(|types| types.iter().any(|kind| kind.as_str() == Some("ON_DEMAND")));
	let text_output = string_list(summary, "outputModalities")
		.is_none_or(|_| has_modality(summary, "outputModalities", "TEXT"));
	lifecycle_active && streams && invocable && text_output
}

fn string_list<'a>(
	summary: &'a serde_json::Value,
	field: &str,
) -> Option<&'a Vec<serde_json::Value>> {
	summary
		.get(field)
		.and_then(serde_json::Value::as_array)
		.filter(|values| !values.is_empty())
}

fn has_modality(summary: &serde_json::Value, field: &str, modality: &str) -> bool {
	string_list(summary, field).is_some_and(|values| {
		values
			.iter()
			.filter_map(serde_json::Value::as_str)
			.any(|value| value == modality)
	})
}

#[cfg(test)]
mod tests {
	use omp_llm_catalog::provider::load_builtin;

	use super::*;

	fn bedrock() -> ProviderEntry {
		load_builtin()
			.expect("built-in providers")
			.get("amazon-bedrock")
			.expect("amazon-bedrock row")
			.clone()
	}

	#[test]
	fn listing_targets_the_control_plane_not_the_runtime_host() {
		let url = list_endpoint(&bedrock(), "eu-west-1").expect("regional endpoint");
		assert_eq!(url, "https://bedrock.eu-west-1.amazonaws.com/foundation-models");
	}

	#[test]
	fn region_prefers_the_runtime_over_the_sdk_default() {
		let provider = bedrock();
		let scoped = Account::new("7", "aws").with_region(Some("ap-south-1".into()));
		assert_eq!(region(&provider, &scoped), "ap-south-1");
		// The shipped row is still a `{region}` template, so nothing can be
		// parsed out of it and the SDK-compatible default applies.
		assert_eq!(region(&provider, &Account::new("7", "aws")), "us-east-1");
	}

	#[test]
	fn parser_keeps_invocable_chat_models_and_drops_the_rest() {
		let cards = parse_foundation_models(
			&bedrock(),
			br#"{"modelSummaries":[
				{"modelId":"anthropic.claude-3-5-sonnet-20241022-v2:0",
				 "modelName":"Claude 3.5 Sonnet v2","providerName":"Anthropic",
				 "inputModalities":["TEXT","IMAGE"],"outputModalities":["TEXT"],
				 "responseStreamingSupported":true,
				 "inferenceTypesSupported":["ON_DEMAND"],
				 "modelLifecycle":{"status":"ACTIVE"}},
				{"modelId":"amazon.titan-embed-text-v2:0","providerName":"Amazon",
				 "outputModalities":["EMBEDDING"],"responseStreamingSupported":false,
				 "modelLifecycle":{"status":"ACTIVE"}},
				{"modelId":"anthropic.claude-v2","providerName":"Anthropic",
				 "outputModalities":["TEXT"],"modelLifecycle":{"status":"LEGACY"}},
				{"modelId":"meta.llama-provisioned","providerName":"Meta",
				 "outputModalities":["TEXT"],"inferenceTypesSupported":["PROVISIONED"]}
			]}"#,
		)
		.expect("ListFoundationModels fixture");
		let ids: Vec<&str> = cards.iter().map(|card| card.model.as_str()).collect();
		assert_eq!(ids, ["anthropic.claude-3-5-sonnet-20241022-v2:0"]);
		assert_eq!(cards[0].family.as_str(), "Anthropic");
		assert_eq!(cards[0].name.as_str(), "Claude 3.5 Sonnet v2");
		assert!(cards[0].inputs.contains(&Modality::Image));
	}

	#[test]
	fn parser_keeps_inference_profile_only_models() {
		// A model reachable only through a cross-region profile reports neither
		// ON_DEMAND nor PROVISIONED. Dropping it would hide most current
		// Anthropic and Nova models from the catalog.
		let cards = parse_foundation_models(
			&bedrock(),
			br#"{"modelSummaries":[
				{"modelId":"anthropic.claude-sonnet-4-5-20250929-v1:0",
				 "modelName":"Claude Sonnet 4.5","providerName":"Anthropic",
				 "outputModalities":["TEXT"],"inferenceTypesSupported":[],
				 "responseStreamingSupported":true,
				 "modelLifecycle":{"status":"ACTIVE"}}
			]}"#,
		)
		.expect("inference-profile-only fixture");
		assert_eq!(cards.len(), 1);
		assert_eq!(cards[0].model.as_str(), "anthropic.claude-sonnet-4-5-20250929-v1:0");
	}

	#[test]
	fn parser_is_permissive_when_aws_omits_optional_fields() {
		let cards = parse_foundation_models(
			&bedrock(),
			br#"{"modelSummaries":[{"modelId":"amazon.nova-pro-v1:0"}]}"#,
		)
		.expect("sparse fixture");
		assert_eq!(cards.len(), 1);
		// No providerName: the id's vendor prefix is the family.
		assert_eq!(cards[0].family.as_str(), "amazon");
	}

	#[test]
	fn parser_rejects_a_payload_without_model_summaries() {
		let error = parse_foundation_models(&bedrock(), br#"{"models":[]}"#)
			.expect_err("a missing modelSummaries array is a payload error");
		assert!(matches!(error, Error::InvalidPayload { .. }), "{error:?}");
	}
}
