//! Validated, indexed access to the checked-in binary catalog snapshot.

use std::sync::LazyLock;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
	compile::{CatalogAlias, CompileError, CompiledCatalog},
	id::{
		AuthSpecId, DiscoverySpecId, HeaderProfileId, ModelKey, OAuthSpecId, ProviderId, RouteId,
		ThinkingPolicyId, WirePolicyId,
	},
	model::ModelSpec,
	policy::WirePolicy,
	provider::{AuthSpec, DiscoverySpec, HeaderProfile, OAuthSpec, ProviderDef, RouteDef},
	thinking::ThinkingPolicy,
};

const MAGIC: &[u8; 8] = b"OMPLLCAT";
const SCHEMA_VERSION: u32 = 1;
const HEADER_LEN: usize = 8 + 4 + 32 + 32 + 32;
const EMBEDDED_BYTES: &[u8] = include_bytes!("../data/catalog.postcard");

static EMBEDDED: LazyLock<Result<Catalog, SnapshotError>> = LazyLock::new(load_embedded);

/// Provenance hashes bound into a generated snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SnapshotProvenance {
	/// Digest of the ordered source lock entries.
	pub source_digest: [u8; 32],
}

/// Deterministic checked-in outputs produced from one compiled catalog.
#[derive(Debug, Eq, PartialEq)]
pub struct SnapshotArtifacts {
	/// Canonical normalized JSON retained for review.
	pub normalized_json: Vec<u8>,
	/// Private indexed postcard representation loaded at runtime.
	pub postcard:        Vec<u8>,
}

/// Validated catalog with compact deterministic lookup indexes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Catalog {
	catalog:                CompiledCatalog,
	provider_models:        Box<[(u32, u32)]>,
	model_index:            Box<[u32]>,
	wire_policy_ids:        Box<[WirePolicyId]>,
	thinking_policy_ids:    Box<[ThinkingPolicyId]>,
	source_digest:          [u8; 32],
	normalized_json_sha256: [u8; 32],
}

#[derive(Serialize, Deserialize)]
struct SnapshotPayload {
	catalog:             CompiledCatalog,
	provider_models:     Box<[(u32, u32)]>,
	model_index:         Box<[u32]>,
	wire_policy_ids:     Box<[WirePolicyId]>,
	thinking_policy_ids: Box<[ThinkingPolicyId]>,
}

/// Failure to generate deterministic snapshot artifacts.
#[derive(Debug, thiserror::Error)]
pub enum SnapshotBuildError {
	/// The normalized review artifact could not be encoded.
	#[error(transparent)]
	Compile(#[from] CompileError),
	/// The private postcard payload could not be encoded.
	#[error("catalog postcard encoding failed: {0}")]
	Postcard(#[from] postcard::Error),
	/// Compiled records violate an index invariant.
	#[error(transparent)]
	Invalid(#[from] SnapshotError),
}

/// Failure to validate or decode a binary catalog snapshot.
#[derive(Debug, thiserror::Error)]
pub enum SnapshotError {
	/// The snapshot ends before its complete header.
	#[error("catalog snapshot is truncated")]
	Truncated,
	/// The file does not carry the catalog snapshot magic.
	#[error("catalog snapshot magic is invalid")]
	InvalidMagic,
	/// The snapshot schema is not supported by this runtime.
	#[error("unsupported catalog snapshot schema {0}")]
	UnsupportedSchema(u32),
	/// The snapshot was generated from a different source lock.
	#[error("catalog snapshot source digest does not match the checked source lock")]
	SourceDigestMismatch,
	/// The postcard payload was changed after generation.
	#[error("catalog snapshot payload hash mismatch")]
	PayloadHashMismatch,
	/// The private postcard payload is malformed.
	#[error("catalog postcard decoding failed: {0}")]
	Postcard(#[from] postcard::Error),
	/// A compiled record or lookup index violates a catalog invariant.
	#[error("catalog snapshot invariant failed: {0}")]
	Invariant(&'static str),
}

impl Catalog {
	/// Returns the process-wide embedded catalog, panicking with validation
	/// evidence on corruption.
	#[must_use]
	pub fn embedded() -> &'static Self {
		match Self::try_embedded() {
			Ok(catalog) => catalog,
			Err(error) => panic!("embedded catalog is invalid: {error}"),
		}
	}

	/// Tries to open the process-wide embedded catalog without parsing JSON.
	pub fn try_embedded() -> Result<&'static Self, &'static SnapshotError> {
		EMBEDDED.as_ref()
	}

	/// Produces canonical JSON and the private postcard snapshot from compiled
	/// records.
	pub fn encode(
		catalog: CompiledCatalog,
		provenance: SnapshotProvenance,
	) -> Result<SnapshotArtifacts, SnapshotBuildError> {
		validate_catalog(&catalog)?;
		let normalized_json = catalog.normalized_json()?;
		let normalized_json_sha256 = Sha256::digest(&normalized_json);
		let provider_models = provider_model_index(&catalog)?;
		let model_index = model_index(&catalog)?;
		let wire_policy_ids = catalog
			.wire_policies
			.iter()
			.map(WirePolicy::content_id)
			.collect::<Vec<_>>()
			.into_boxed_slice();
		let thinking_policy_ids = catalog
			.thinking_policies
			.iter()
			.map(ThinkingPolicy::content_id)
			.collect::<Vec<_>>()
			.into_boxed_slice();
		ensure_strictly_sorted(&wire_policy_ids, "wire policy ids are not unique and sorted")?;
		ensure_strictly_sorted(
			&thinking_policy_ids,
			"thinking policy ids are not unique and sorted",
		)?;
		let payload = postcard::to_allocvec(&SnapshotPayload {
			catalog,
			provider_models,
			model_index,
			wire_policy_ids,
			thinking_policy_ids,
		})?;
		let payload_sha256 = Sha256::digest(&payload);
		let mut postcard = Vec::with_capacity(HEADER_LEN + payload.len());
		postcard.extend_from_slice(MAGIC);
		postcard.extend_from_slice(&SCHEMA_VERSION.to_le_bytes());
		postcard.extend_from_slice(&provenance.source_digest);
		postcard.extend_from_slice(&normalized_json_sha256);
		postcard.extend_from_slice(&payload_sha256);
		postcard.extend_from_slice(&payload);
		Ok(SnapshotArtifacts { normalized_json, postcard })
	}

	/// Decodes and validates arbitrary snapshot bytes against their
	/// self-contained hashes.
	pub fn decode(bytes: &[u8]) -> Result<Self, SnapshotError> {
		Self::decode_inner(bytes, None)
	}

	/// Decodes snapshot bytes while requiring a particular source-lock digest.
	pub fn decode_for_source(bytes: &[u8], source_digest: [u8; 32]) -> Result<Self, SnapshotError> {
		Self::decode_inner(bytes, Some(source_digest))
	}

	fn decode_inner(bytes: &[u8], expected_source: Option<[u8; 32]>) -> Result<Self, SnapshotError> {
		if bytes.len() < HEADER_LEN {
			return Err(SnapshotError::Truncated);
		}
		if &bytes[..8] != MAGIC {
			return Err(SnapshotError::InvalidMagic);
		}
		let schema = u32::from_le_bytes(bytes[8..12].try_into().expect("fixed schema field"));
		if schema != SCHEMA_VERSION {
			return Err(SnapshotError::UnsupportedSchema(schema));
		}
		let source_digest: [u8; 32] = bytes[12..44].try_into().expect("fixed digest field");
		if expected_source.is_some_and(|expected| expected != source_digest) {
			return Err(SnapshotError::SourceDigestMismatch);
		}
		let normalized_json_sha256 = bytes[44..76].try_into().expect("fixed digest field");
		let expected_payload_hash: [u8; 32] = bytes[76..108].try_into().expect("fixed digest field");
		let actual_payload_hash: [u8; 32] = Sha256::digest(&bytes[HEADER_LEN..]).into();
		if actual_payload_hash != expected_payload_hash {
			return Err(SnapshotError::PayloadHashMismatch);
		}
		let payload: SnapshotPayload = postcard::from_bytes(&bytes[HEADER_LEN..])?;
		validate_catalog(&payload.catalog)?;
		let expected_provider_models = provider_model_index(&payload.catalog)?;
		let expected_model_index = model_index(&payload.catalog)?;
		if payload.model_index != expected_model_index {
			return Err(SnapshotError::Invariant("model key index does not match catalog records"));
		}
		if payload.provider_models != expected_provider_models {
			return Err(SnapshotError::Invariant(
				"provider/model index does not match catalog records",
			));
		}
		validate_policy_ids(&payload)?;
		Ok(Self {
			catalog: payload.catalog,
			provider_models: payload.provider_models,
			model_index: payload.model_index,
			wire_policy_ids: payload.wire_policy_ids,
			thinking_policy_ids: payload.thinking_policy_ids,
			source_digest,
			normalized_json_sha256,
		})
	}

	/// Returns the immutable catalog revision.
	#[must_use]
	pub fn revision(&self) -> &crate::CatalogRevision {
		&self.catalog.revision
	}

	/// Returns the verified compiler census.
	#[must_use]
	pub const fn census(&self) -> crate::compile::CompilerCensus {
		self.catalog.census
	}

	/// Returns providers in stable identifier order.
	#[must_use]
	pub fn providers(&self) -> &[ProviderDef] {
		&self.catalog.providers
	}

	/// Returns routes in stable identifier order.
	#[must_use]
	pub fn routes(&self) -> &[RouteDef] {
		&self.catalog.routes
	}

	/// Returns models in stable key order.
	#[must_use]
	pub fn models(&self) -> &[ModelSpec] {
		&self.catalog.models
	}

	/// Returns interned authentication specifications in stable identifier
	/// order.
	#[must_use]
	pub fn auth_specs(&self) -> &[AuthSpec] {
		&self.catalog.auth_specs
	}

	/// Returns interned public OAuth flow specifications in stable identifier
	/// order.
	#[must_use]
	pub fn oauth_specs(&self) -> &[OAuthSpec] {
		&self.catalog.oauth_specs
	}

	/// Returns interned safe header profiles in stable identifier order.
	#[must_use]
	pub fn header_profiles(&self) -> &[HeaderProfile] {
		&self.catalog.header_profiles
	}

	/// Returns interned discovery specifications in stable identifier order.
	#[must_use]
	pub fn discovery_specs(&self) -> &[DiscoverySpec] {
		&self.catalog.discovery_specs
	}

	/// Returns aliases in stable selector order.
	#[must_use]
	pub fn aliases(&self) -> &[CatalogAlias] {
		&self.catalog.aliases
	}

	/// Returns the source-lock digest bound into this snapshot.
	#[must_use]
	pub const fn source_digest(&self) -> &[u8; 32] {
		&self.source_digest
	}

	/// Returns the hash of the normalized JSON reviewed with this snapshot.
	#[must_use]
	pub const fn normalized_json_sha256(&self) -> &[u8; 32] {
		&self.normalized_json_sha256
	}

	/// Looks up one provider by exact stable identifier.
	#[must_use]
	pub fn provider(&self, id: &ProviderId) -> Option<&ProviderDef> {
		self
			.catalog
			.providers
			.binary_search_by(|record| record.id.cmp(id))
			.ok()
			.map(|index| &self.catalog.providers[index])
	}

	/// Returns authored conservative discovery defaults for one exact provider.
	#[must_use]
	pub fn discovery_defaults(
		&self,
		id: &ProviderId,
	) -> Option<&crate::discover::DiscoveryDefaults> {
		self.provider(id)?.discovery_defaults.as_ref()
	}

	/// Looks up one route by exact stable identifier.
	#[must_use]
	pub fn route(&self, id: &RouteId) -> Option<&RouteDef> {
		self
			.catalog
			.routes
			.binary_search_by(|record| record.id.cmp(id))
			.ok()
			.map(|index| &self.catalog.routes[index])
	}

	/// Looks up one model by exact normalized key.
	#[must_use]
	pub fn model(&self, key: &ModelKey) -> Option<&ModelSpec> {
		let index = self.model_position(key)?;
		Some(&self.catalog.models[index])
	}

	/// Looks up a model only when it is exposed by the requested provider.
	#[must_use]
	pub fn model_for_provider(&self, provider: &ProviderId, key: &ModelKey) -> Option<&ModelSpec> {
		let provider_index = self
			.catalog
			.providers
			.binary_search_by(|record| record.id.cmp(provider))
			.ok()?;
		let model_index = self.model_position(key)?;
		let pair = (u32::try_from(provider_index).ok()?, u32::try_from(model_index).ok()?);
		self
			.provider_models
			.binary_search(&pair)
			.ok()
			.map(|_| &self.catalog.models[model_index])
	}

	fn model_position(&self, key: &ModelKey) -> Option<usize> {
		let position = self
			.model_index
			.binary_search_by(|index| self.catalog.models[*index as usize].key.cmp(key))
			.ok()?;
		usize::try_from(self.model_index[position]).ok()
	}

	/// Resolves an exact alias to its canonical model record.
	#[must_use]
	pub fn resolve_alias(&self, alias: &str) -> Option<&ModelSpec> {
		let index = self
			.catalog
			.aliases
			.binary_search_by(|record| record.alias.as_str().cmp(alias))
			.ok()?;
		self.model(&self.catalog.aliases[index].target)
	}

	/// Looks up an interned authentication specification.
	#[must_use]
	pub fn auth_spec(&self, id: &AuthSpecId) -> Option<&AuthSpec> {
		self
			.catalog
			.auth_specs
			.binary_search_by(|record| record.id.cmp(id))
			.ok()
			.map(|index| &self.catalog.auth_specs[index])
	}

	/// Looks up an interned public OAuth flow specification.
	#[must_use]
	pub fn oauth_spec(&self, id: &OAuthSpecId) -> Option<&OAuthSpec> {
		self
			.catalog
			.oauth_specs
			.binary_search_by(|record| record.id.cmp(id))
			.ok()
			.map(|index| &self.catalog.oauth_specs[index])
	}

	/// Looks up an interned safe header profile.
	#[must_use]
	pub fn header_profile(&self, id: &HeaderProfileId) -> Option<&HeaderProfile> {
		self
			.catalog
			.header_profiles
			.binary_search_by(|record| record.id.cmp(id))
			.ok()
			.map(|index| &self.catalog.header_profiles[index])
	}

	/// Looks up an interned discovery specification.
	#[must_use]
	pub fn discovery_spec(&self, id: &DiscoverySpecId) -> Option<&DiscoverySpec> {
		self
			.catalog
			.discovery_specs
			.binary_search_by(|record| record.id.cmp(id))
			.ok()
			.map(|index| &self.catalog.discovery_specs[index])
	}

	/// Looks up an interned wire policy without re-hashing it.
	#[must_use]
	pub fn wire_policy(&self, id: &WirePolicyId) -> Option<&WirePolicy> {
		let index = self.wire_policy_ids.binary_search(id).ok()?;
		Some(&self.catalog.wire_policies[index])
	}

	/// Looks up an interned thinking policy without re-hashing it.
	#[must_use]
	pub fn thinking_policy(&self, id: &ThinkingPolicyId) -> Option<&ThinkingPolicy> {
		let index = self.thinking_policy_ids.binary_search(id).ok()?;
		Some(&self.catalog.thinking_policies[index])
	}
}

fn validate_catalog(catalog: &CompiledCatalog) -> Result<(), SnapshotError> {
	if catalog.schema_version != SCHEMA_VERSION {
		return Err(SnapshotError::UnsupportedSchema(catalog.schema_version));
	}
	if catalog.revision.as_str().is_empty() {
		return Err(SnapshotError::Invariant("catalog revision is empty"));
	}
	ensure_sorted_by(&catalog.providers, |record| &record.id, "providers are not uniquely sorted")?;
	ensure_sorted_by(&catalog.routes, |record| &record.id, "routes are not uniquely sorted")?;
	model_index(catalog)?;
	ensure_sorted_by(
		&catalog.auth_specs,
		|record| &record.id,
		"auth specs are not uniquely sorted",
	)?;
	ensure_sorted_by(
		&catalog.oauth_specs,
		|record| &record.id,
		"OAuth specs are not uniquely sorted",
	)?;
	ensure_sorted_by(
		&catalog.header_profiles,
		|record| &record.id,
		"header profiles are not uniquely sorted",
	)?;
	ensure_sorted_by(
		&catalog.discovery_specs,
		|record| &record.id,
		"discovery specs are not uniquely sorted",
	)?;
	for auth in &catalog.auth_specs {
		if let Some(oauth) = &auth.oauth {
			if catalog
				.oauth_specs
				.binary_search_by(|record| record.id.cmp(oauth))
				.is_err()
			{
				return Err(SnapshotError::Invariant("auth spec references an unknown OAuth flow"));
			}
		}
	}
	for route in &catalog.routes {
		if catalog
			.providers
			.binary_search_by(|record| record.id.cmp(&route.provider))
			.is_err()
		{
			return Err(SnapshotError::Invariant("route references an unknown provider"));
		}
	}
	for model in &catalog.models {
		for route in &model.routes {
			if catalog
				.routes
				.binary_search_by(|record| record.id.cmp(route))
				.is_err()
			{
				return Err(SnapshotError::Invariant("model references an unknown route"));
			}
		}
		for (route, _) in &model.wire_ids {
			if catalog
				.routes
				.binary_search_by(|record| record.id.cmp(route))
				.is_err()
			{
				return Err(SnapshotError::Invariant("wire model id references an unknown route"));
			}
		}
	}
	for pair in catalog.aliases.windows(2) {
		if pair[0].alias >= pair[1].alias {
			return Err(SnapshotError::Invariant("aliases are not uniquely sorted"));
		}
	}
	for alias in &catalog.aliases {
		if !catalog.models.iter().any(|model| model.key == alias.target) {
			return Err(SnapshotError::Invariant("alias references an unknown model"));
		}
	}
	Ok(())
}

fn model_index(catalog: &CompiledCatalog) -> Result<Box<[u32]>, SnapshotError> {
	let mut index = (0..catalog.models.len())
		.map(|index| {
			u32::try_from(index).map_err(|_| SnapshotError::Invariant("model index exceeds u32"))
		})
		.collect::<Result<Vec<_>, _>>()?;
	index.sort_unstable_by(|left, right| {
		catalog.models[*left as usize]
			.key
			.cmp(&catalog.models[*right as usize].key)
	});
	for pair in index.windows(2) {
		if catalog.models[pair[0] as usize].key == catalog.models[pair[1] as usize].key {
			return Err(SnapshotError::Invariant("model keys are not unique"));
		}
	}
	Ok(index.into_boxed_slice())
}

fn provider_model_index(catalog: &CompiledCatalog) -> Result<Box<[(u32, u32)]>, SnapshotError> {
	let mut pairs = Vec::new();
	for (model_index, model) in catalog.models.iter().enumerate() {
		for route_id in &model.routes {
			let route_index = catalog
				.routes
				.binary_search_by(|route| route.id.cmp(route_id))
				.map_err(|_| SnapshotError::Invariant("model references an unknown route"))?;
			let provider_index = catalog
				.providers
				.binary_search_by(|provider| provider.id.cmp(&catalog.routes[route_index].provider))
				.map_err(|_| SnapshotError::Invariant("route references an unknown provider"))?;
			pairs.push((
				u32::try_from(provider_index)
					.map_err(|_| SnapshotError::Invariant("provider index exceeds u32"))?,
				u32::try_from(model_index)
					.map_err(|_| SnapshotError::Invariant("model index exceeds u32"))?,
			));
		}
	}
	pairs.sort_unstable();
	pairs.dedup();
	Ok(pairs.into_boxed_slice())
}

fn validate_policy_ids(payload: &SnapshotPayload) -> Result<(), SnapshotError> {
	if payload.wire_policy_ids.len() != payload.catalog.wire_policies.len()
		|| payload.thinking_policy_ids.len() != payload.catalog.thinking_policies.len()
	{
		return Err(SnapshotError::Invariant("policy index length does not match policy table"));
	}
	ensure_strictly_sorted(&payload.wire_policy_ids, "wire policy ids are not uniquely sorted")?;
	ensure_strictly_sorted(
		&payload.thinking_policy_ids,
		"thinking policy ids are not uniquely sorted",
	)
}

fn ensure_sorted_by<T, K: Ord>(
	values: &[T],
	key: impl Fn(&T) -> &K,
	message: &'static str,
) -> Result<(), SnapshotError> {
	if values.windows(2).any(|pair| key(&pair[0]) >= key(&pair[1])) {
		return Err(SnapshotError::Invariant(message));
	}
	Ok(())
}

fn ensure_strictly_sorted<T: Ord>(
	values: &[T],
	message: &'static str,
) -> Result<(), SnapshotError> {
	if values.windows(2).any(|pair| pair[0] >= pair[1]) {
		return Err(SnapshotError::Invariant(message));
	}
	Ok(())
}

fn load_embedded() -> Result<Catalog, SnapshotError> {
	Catalog::decode_for_source(EMBEDDED_BYTES, embedded_source_digest())
}

fn embedded_source_digest() -> [u8; 32] {
	decode_hex_digest(env!("OMP_LLM_CATALOG_SOURCE_DIGEST"))
		.expect("build.rs emits a validated source digest")
}

fn decode_hex_digest(value: &str) -> Option<[u8; 32]> {
	if value.len() != 64 {
		return None;
	}
	let mut digest = [0_u8; 32];
	for (index, byte) in digest.iter_mut().enumerate() {
		*byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16).ok()?;
	}
	Some(digest)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn embedded_snapshot_opens_and_indexes_deterministically() {
		let catalog = Catalog::embedded();
		assert!(!catalog.providers().is_empty());
		assert!(!catalog.routes().is_empty());
		assert!(!catalog.models().is_empty());
		assert!(!catalog.oauth_specs().is_empty());
		for oauth in catalog.oauth_specs() {
			assert_eq!(catalog.oauth_spec(&oauth.id), Some(oauth));
		}
		for auth in catalog.auth_specs() {
			if let Some(oauth) = &auth.oauth {
				assert!(catalog.oauth_spec(oauth).is_some(), "OAuth reference must resolve");
			}
		}
		for provider in catalog.providers() {
			assert_eq!(catalog.provider(&provider.id), Some(provider));
		}
		for route in catalog.routes() {
			assert_eq!(catalog.route(&route.id), Some(route));
		}
		for model in catalog.models() {
			assert_eq!(catalog.model(&model.key), Some(model));
		}
	}

	#[test]
	fn discovery_defaults_are_borrowed_from_exact_provider_records() {
		let catalog = Catalog::embedded();
		for provider in catalog.providers() {
			assert_eq!(catalog.discovery_defaults(&provider.id), provider.discovery_defaults.as_ref(),);
		}
		assert!(
			catalog
				.discovery_defaults(&ProviderId::from("missing-provider"))
				.is_none()
		);
	}

	#[test]
	fn corruption_and_provenance_mismatch_fail_loudly() {
		let mut corrupt = EMBEDDED_BYTES.to_vec();
		let last = corrupt.last_mut().expect("embedded snapshot is nonempty");
		*last ^= 0x80;
		assert!(matches!(Catalog::decode(&corrupt), Err(SnapshotError::PayloadHashMismatch)));
		let mut wrong_source = embedded_source_digest();
		wrong_source[0] ^= 0x80;
		assert!(matches!(
			Catalog::decode_for_source(EMBEDDED_BYTES, wrong_source),
			Err(SnapshotError::SourceDigestMismatch)
		));
	}

	#[test]
	fn alias_and_provider_model_indexes_match_catalog_relationships() {
		let catalog = Catalog::embedded();
		for alias in &catalog.catalog.aliases {
			assert_eq!(
				catalog
					.resolve_alias(alias.alias.as_str())
					.map(|model| &model.key),
				Some(&alias.target)
			);
		}
		for model in catalog.models() {
			for route_id in &model.routes {
				let route = catalog.route(route_id).expect("validated model route");
				assert_eq!(catalog.model_for_provider(&route.provider, &model.key), Some(model));
			}
		}
	}
}
