//! GitLab Duo Agent model discovery through namespace-scoped GraphQL
//! availability.

use std::collections::BTreeMap;

use async_trait::async_trait;
use bytes::Bytes;
use http::{Method, Request, header::CONTENT_TYPE};
use omp_llm_catalog::{
	discovery::{Account, DiscoveryHttp, DiscoveryProtocol, Error, discovered_card, infer_family},
	models::ModelCard,
	provider::{ProviderEntry, TransportId},
};
use serde_json::Value;

/// Discovers the GitLab Duo models available to an account's namespace.
pub struct DuoDiscovery;

#[async_trait]
impl DiscoveryProtocol for DuoDiscovery {
	fn transports(&self) -> &'static [TransportId] {
		&[TransportId::GitLabDuoWorkflow]
	}

	async fn discover(
		&self,
		provider: &ProviderEntry,
		account: &Account,
		http: &dyn DiscoveryHttp,
	) -> Result<Vec<ModelCard>, Error> {
		if let Some(namespace) = account.organization_id.as_deref() {
			let cards = namespace_models(provider, account, namespace, http).await?;
			if !cards.is_empty() {
				return Ok(cards);
			}
		}
		if let Some(project) = account.project_id.as_deref()
			&& let Some(namespace) = project_namespace(provider, account, project, http).await?
		{
			let cards = namespace_models(provider, account, &namespace, http).await?;
			if !cards.is_empty() {
				return Ok(cards);
			}
		}
		for page in 1..=50 {
			let url = format!(
				"{}/api/v4/groups?top_level_only=true&per_page=100&page={page}",
				provider.base_url.trim_end_matches('/')
			);
			let response = http.get(provider, account, &url).await?;
			let groups: Vec<Value> =
				serde_json::from_slice(response.ensure_success(provider)?).map_err(Error::transport)?;
			let count = groups.len();
			for group in groups {
				let Some(namespace) = group.get("id").and_then(Value::as_u64) else {
					continue;
				};
				let cards = namespace_models(provider, account, &namespace.to_string(), http).await?;
				if !cards.is_empty() {
					return Ok(cards);
				}
			}
			if count < 100 {
				break;
			}
		}
		Err(Error::InvalidPayload {
			provider: provider.id.clone(),
			detail:   "no GitLab namespace exposes Duo models".into(),
		})
	}
}

async fn project_namespace(
	provider: &ProviderEntry,
	account: &Account,
	project: &str,
	http: &dyn DiscoveryHttp,
) -> Result<Option<String>, Error> {
	let encoded = url::form_urlencoded::byte_serialize(project.as_bytes()).collect::<String>();
	let url = format!("{}/api/v4/projects/{encoded}", provider.base_url.trim_end_matches('/'));
	let response = http.get(provider, account, &url).await?;
	let payload: Value =
		serde_json::from_slice(response.ensure_success(provider)?).map_err(Error::transport)?;
	Ok(payload
		.pointer("/namespace/root_ancestor/id")
		.or_else(|| payload.pointer("/namespace/id"))
		.and_then(|value| {
			value
				.as_str()
				.map(ToOwned::to_owned)
				.or_else(|| value.as_u64().map(|value| value.to_string()))
		}))
}

async fn namespace_models(
	provider: &ProviderEntry,
	account: &Account,
	namespace: &str,
	http: &dyn DiscoveryHttp,
) -> Result<Vec<ModelCard>, Error> {
	const QUERY: &str = "query lsp_aiChatAvailableModels($rootNamespaceId: GroupID!) { \
	                     aiChatAvailableModels(rootNamespaceId: $rootNamespaceId) { defaultModel { \
	                     name ref } selectableModels { name ref } pinnedModel { name ref } } }";
	let root_namespace_id = if namespace.bytes().all(|byte| byte.is_ascii_digit()) {
		format!("gid://gitlab/Group/{namespace}")
	} else {
		namespace.to_owned()
	};
	let body = serde_json::to_vec(&serde_json::json!({
		"query": QUERY,
		"variables": { "rootNamespaceId": root_namespace_id },
	}))
	.map_err(Error::transport)?;
	let request = Request::builder()
		.method(Method::POST)
		.uri(format!("{}/api/graphql", provider.base_url.trim_end_matches('/')))
		.header(CONTENT_TYPE, "application/json")
		.body(Bytes::from(body))
		.map_err(Error::transport)?;
	let response = http.execute(provider, account, request).await?;
	parse_gitlab_duo_models(provider, response.ensure_success(provider)?)
}

/// Parses GitLab Duo's `aiChatAvailableModels` GraphQL payload.
///
/// # Errors
/// Returns [`Error::InvalidPayload`] for malformed JSON. A successful GraphQL
/// response with no availability yields an empty source.
pub fn parse_gitlab_duo_models(
	provider: &ProviderEntry,
	body: &[u8],
) -> Result<Vec<ModelCard>, Error> {
	let payload: Value =
		serde_json::from_slice(body).map_err(|error| Error::payload(provider, error))?;
	let Some(availability) = payload
		.pointer("/data/aiChatAvailableModels")
		.and_then(Value::as_object)
	else {
		return Ok(Vec::new());
	};
	let entries = availability
		.get("selectableModels")
		.and_then(Value::as_array)
		.into_iter()
		.flatten()
		.chain(availability.get("defaultModel"))
		.chain(availability.get("pinnedModel"));
	let mut models = BTreeMap::new();
	for entry in entries {
		let Some(model) = entry
			.get("ref")
			.and_then(Value::as_str)
			.filter(|value| !value.is_empty())
		else {
			continue;
		};
		let name = entry
			.get("name")
			.and_then(Value::as_str)
			.filter(|value| !value.is_empty())
			.unwrap_or(model);
		let mut card = discovered_card(provider, model, name, infer_family(model));
		card.context_window =
			if model.contains("opus") || model.contains("sonnet") || model.contains("gemini") {
				1_000_000
			} else if model.contains("gpt-5") {
				400_000
			} else {
				200_000
			};
		card.max_output_tokens = 0;
		models.insert(card.id.clone(), card);
	}
	Ok(models.into_values().collect())
}

/// GitLab Duo discovery protocol registered by the application at daemon
/// start-up.
pub static DISCOVERY: DuoDiscovery = DuoDiscovery;

#[cfg(test)]
mod tests {
	use std::collections::BTreeMap;

	use omp_core::Str;
	use omp_llm_catalog::{
		compat::Compat,
		provider::{AuthSpec, Facet as ProviderFacet},
	};

	use super::*;

	fn provider(id: &str, base_url: &str) -> ProviderEntry {
		ProviderEntry::builder()
			.id(Str::from(id))
			.transport(TransportId::GitLabDuoWorkflow)
			.base_url(Str::from(base_url))
			.auth(AuthSpec::None)
			.facets(std::iter::once(ProviderFacet::Chat).collect())
			.headers(BTreeMap::new())
			.compat(Compat::default())
			.build()
	}

	#[test]
	fn gitlab_parser_merges_default_selectable_and_pinned_refs() {
		let provider = provider("gitlab-duo-agent", "https://gitlab.com");
		let cards = parse_gitlab_duo_models(
			&provider,
			br#"{"data":{"aiChatAvailableModels":{
				"defaultModel":{"name":"Sonnet","ref":"claude_sonnet_4_6_vertex"},
				"selectableModels":[{"name":"Opus","ref":"claude_opus_4_8"}],
				"pinnedModel":{"name":"Sonnet pinned","ref":"claude_sonnet_4_6_vertex"}
			}}}"#,
		)
		.expect("GitLab GraphQL fixture");
		assert_eq!(cards.len(), 2);
		assert!(cards.iter().all(|card| card.context_window == 1_000_000));
	}
}
