//! Standalone typed model-discovery codecs selected by catalog discovery kind.

use bytes::Bytes;
use omp_core::Str;
use omp_llm_catalog::{
	DiscoveredModel, DiscoveryKind, DiscoveryPagination, DiscoverySpec, FamilyId, ModelAvailability,
	ModelLimits, OperationBits, OperationKind, ProviderId, RouteId, WireModelId,
};
use serde::Deserialize;

use crate::{
	body::BodySource,
	call::{DiscoveryRequest, OperationCall},
	codec::{
		Codec, DecodeContext, Decoder, DecoderState, EncodeContext, EncodedRequest, RawEvent,
		RequestMethod, SizeBounds,
	},
	error::{Error, ErrorKind, ErrorPhase, RetryAction},
	receipt::ExecutionReceipt,
	transport::{Frame, FramingProtocol},
};

const MAX_FRAME_BYTES: u64 = 16 * 1024 * 1024;
const MAX_RESPONSE_BYTES: u64 = 256 * 1024 * 1024;

/// Standalone OpenAI-compatible `{ "data": [...] }` model discovery codec.
#[derive(Clone, Debug)]
pub struct OpenAiModelsDiscoveryCodec {
	core: DiscoveryCore,
}

impl OpenAiModelsDiscoveryCodec {
	/// Constructs a codec from an exact OpenAI-models discovery specification.
	pub fn from_spec(spec: &DiscoverySpec) -> Result<Self, Error> {
		DiscoveryCore::new(spec, DiscoveryKind::OpenAiModels).map(|core| Self { core })
	}
}

/// Standalone Google Generative Language `{ "models": [...] }` discovery codec.
#[derive(Clone, Debug)]
pub struct GoogleModelsDiscoveryCodec {
	core: DiscoveryCore,
}

impl GoogleModelsDiscoveryCodec {
	/// Constructs a codec from an exact Google-models discovery specification.
	pub fn from_spec(spec: &DiscoverySpec) -> Result<Self, Error> {
		DiscoveryCore::new(spec, DiscoveryKind::GoogleModels).map(|core| Self { core })
	}
}

/// Standalone account-scoped `{ "models": [...] }` discovery codec.
#[derive(Clone, Debug)]
pub struct AccountModelsDiscoveryCodec {
	core: DiscoveryCore,
}

impl AccountModelsDiscoveryCodec {
	/// Constructs a codec from an exact account-models discovery specification.
	pub fn from_spec(spec: &DiscoverySpec) -> Result<Self, Error> {
		DiscoveryCore::new(spec, DiscoveryKind::AccountModels).map(|core| Self { core })
	}
}

/// Standalone Ollama `/api/tags` discovery codec.
#[derive(Clone, Debug)]
pub struct OllamaTagsDiscoveryCodec {
	core: DiscoveryCore,
}

impl OllamaTagsDiscoveryCodec {
	/// Constructs a codec from an exact Ollama-tags discovery specification.
	pub fn from_spec(spec: &DiscoverySpec) -> Result<Self, Error> {
		DiscoveryCore::new(spec, DiscoveryKind::OllamaTags).map(|core| Self { core })
	}
}

macro_rules! impl_discovery_codec {
	($codec:ty, $flavor:expr) => {
		impl Codec for $codec {
			fn encode(
				&self,
				context: &EncodeContext<'_>,
				operation: &OperationCall,
			) -> Result<EncodedRequest, Error> {
				self.core.encode(context, operation)
			}

			fn decoder(&self, context: &DecodeContext<'_>) -> Result<DecoderState, Error> {
				self.core.decoder(context, $flavor)
			}
		}
	};
}

impl_discovery_codec!(OpenAiModelsDiscoveryCodec, DiscoveryFlavor::OpenAi);
impl_discovery_codec!(GoogleModelsDiscoveryCodec, DiscoveryFlavor::Google);
impl_discovery_codec!(AccountModelsDiscoveryCodec, DiscoveryFlavor::Account);
impl_discovery_codec!(OllamaTagsDiscoveryCodec, DiscoveryFlavor::Ollama);

#[derive(Clone, Debug)]
struct DiscoveryCore {
	spec: DiscoverySpec,
}

impl DiscoveryCore {
	fn new(spec: &DiscoverySpec, expected: DiscoveryKind) -> Result<Self, Error> {
		if spec.kind != expected {
			return Err(invalid_request());
		}
		Ok(Self { spec: spec.clone() })
	}

	fn encode(
		&self,
		context: &EncodeContext<'_>,
		operation: &OperationCall,
	) -> Result<EncodedRequest, Error> {
		let OperationCall::DiscoverModels(request) = operation else {
			return Err(invalid_request());
		};
		let uri = discovery_uri(
			context.route.endpoint.base_url.as_str(),
			self.spec.path.as_str(),
			&self.spec.pagination,
			request,
		)?;
		Ok(EncodedRequest::new(
			OperationKind::DiscoverModels,
			RequestMethod::Get,
			uri,
			Box::new([]),
			BodySource::Bytes(Bytes::new()),
			FramingProtocol::Raw,
			SizeBounds {
				request_body: 0,
				frame:        MAX_FRAME_BYTES,
				response:     MAX_RESPONSE_BYTES,
			},
		))
	}

	fn decoder(
		&self,
		context: &DecodeContext<'_>,
		flavor: DiscoveryFlavor,
	) -> Result<DecoderState, Error> {
		if context.operation != OperationKind::DiscoverModels
			|| context.operation_call.kind() != OperationKind::DiscoverModels
		{
			return Err(invalid_request());
		}
		let OperationCall::DiscoverModels(request) = context.operation_call else {
			return Err(invalid_request());
		};
		Ok(Box::new(DiscoveryDecoder {
			provider: context.provider.clone(),
			route: context.route.clone(),
			source: self.spec.label.clone(),
			pagination: self.spec.pagination.clone(),
			request_cursor: request.cursor.clone(),
			flavor,
			done: false,
		}))
	}
}

fn discovery_uri(
	base: &str,
	path: &str,
	pagination: &DiscoveryPagination,
	request: &DiscoveryRequest,
) -> Result<Str, Error> {
	let mut joined = String::with_capacity(base.len() + path.len() + 1);
	joined.push_str(base.trim_end_matches('/'));
	if !path.starts_with('/') {
		joined.push('/');
	}
	joined.push_str(path);
	let mut uri = url::Url::parse(&joined).map_err(|_| invalid_request())?;
	match pagination {
		DiscoveryPagination::SinglePage => {
			if request.cursor.is_some() {
				return Err(invalid_request());
			}
		},
		DiscoveryPagination::Cursor { query_parameter } => {
			if let Some(cursor) = &request.cursor {
				uri.query_pairs_mut()
					.append_pair(query_parameter.as_str(), cursor.as_str());
			}
		},
		DiscoveryPagination::PageNumber { query_parameter, first_page } => {
			let page = request
				.cursor
				.as_ref()
				.map(|cursor| {
					cursor
						.as_str()
						.parse::<u32>()
						.map_err(|_| invalid_request())
				})
				.transpose()?
				.unwrap_or(*first_page);
			uri.query_pairs_mut()
				.append_pair(query_parameter.as_str(), &page.to_string());
		},
	}
	Ok(Str::from(uri.to_string()))
}

#[derive(Clone, Copy, Debug)]
enum DiscoveryFlavor {
	OpenAi,
	Google,
	Account,
	Ollama,
}

struct DiscoveryDecoder {
	provider:       ProviderId,
	route:          RouteId,
	source:         Str,
	pagination:     DiscoveryPagination,
	request_cursor: Option<Str>,
	flavor:         DiscoveryFlavor,
	done:           bool,
}

impl Decoder for DiscoveryDecoder {
	fn push(&mut self, frame: Frame, emit: &mut dyn FnMut(RawEvent)) -> Result<(), Error> {
		if self.done {
			return Err(protocol_error());
		}
		let Frame::Raw(payload) = frame else {
			return Err(protocol_error());
		};
		let (rows, provider_cursor) = match self.flavor {
			DiscoveryFlavor::OpenAi => self.decode_openai(&payload)?,
			DiscoveryFlavor::Google => self.decode_google(&payload)?,
			DiscoveryFlavor::Account => self.decode_account(&payload)?,
			DiscoveryFlavor::Ollama => self.decode_ollama(&payload)?,
		};
		let next_cursor = self.next_cursor(provider_cursor)?;
		emit(RawEvent::DiscoveredModels { rows, next_cursor });
		self.done = true;
		Ok(())
	}

	fn finish(&mut self, _emit: &mut dyn FnMut(RawEvent)) -> Result<(), Error> {
		if self.done {
			Ok(())
		} else {
			Err(protocol_error())
		}
	}
}

impl DiscoveryDecoder {
	fn decode_openai(
		&self,
		payload: &[u8],
	) -> Result<(Vec<DiscoveredModel>, ProviderCursor), Error> {
		let envelope: OpenAiEnvelope =
			serde_json::from_slice(payload).map_err(|_| protocol_error())?;
		let cursor = envelope.cursor();
		let rows = envelope
			.data
			.into_iter()
			.map(|model| self.row(model.id, None, None, None))
			.collect();
		Ok((rows, cursor))
	}

	fn decode_google(
		&self,
		payload: &[u8],
	) -> Result<(Vec<DiscoveredModel>, ProviderCursor), Error> {
		let envelope: GoogleModelsEnvelope =
			serde_json::from_slice(payload).map_err(|_| protocol_error())?;
		let mut rows = Vec::with_capacity(envelope.models.len());
		for model in envelope.models {
			let name: &str = model.name.as_str();
			let wire_model = name.strip_prefix("models/").unwrap_or(name);
			if wire_model.is_empty() {
				return Err(protocol_error());
			}
			let mut operations = OperationBits::empty();
			for method in model.supported_generation_methods {
				match method.as_str() {
					"generateContent" | "streamGenerateContent" => {
						operations.insert_kind(OperationKind::Chat);
					},
					"countTokens" => operations.insert_kind(OperationKind::CountTokens),
					"embedContent" | "batchEmbedContents" => {
						operations.insert_kind(OperationKind::Embed);
					},
					_ => {},
				}
			}
			let limits = (model.input_token_limit.is_some() || model.output_token_limit.is_some())
				.then_some(ModelLimits {
					context_window:        None,
					maximum_input_tokens:  model.input_token_limit,
					maximum_output_tokens: model.output_token_limit,
					maximum_batch:         None,
				});
			let mut row = self.row(wire_model.into(), model.display_name, None, None);
			if let Some(base_model) = model.base_model_id {
				let alias = base_model
					.as_str()
					.strip_prefix("models/")
					.unwrap_or(base_model.as_str());
				if !alias.is_empty() && alias != wire_model {
					row.aliases = vec![WireModelId::from(alias)].into_boxed_slice();
				}
			}
			row.declared_operations = operations;
			row.declared_limits = limits;
			rows.push(row);
		}
		Ok((rows, ProviderCursor { next: envelope.next_page_token, ..ProviderCursor::default() }))
	}

	fn decode_account(
		&self,
		payload: &[u8],
	) -> Result<(Vec<DiscoveredModel>, ProviderCursor), Error> {
		let envelope: AccountEnvelope =
			serde_json::from_slice(payload).map_err(|_| protocol_error())?;
		let cursor = envelope.cursor();
		let mut rows = Vec::with_capacity(envelope.models.len());
		for model in envelope.models {
			let wire_model = model.slug.or(model.id).ok_or_else(protocol_error)?;
			let availability = match model.visibility.as_ref().map(Str::as_str) {
				Some("visible" | "available") => Some(ModelAvailability::Available),
				Some("hide" | "hidden" | "disabled") => Some(ModelAvailability::Disabled),
				_ => None,
			};
			rows.push(self.row(
				wire_model,
				model.display_name,
				model.family.map(FamilyId::from),
				availability,
			));
		}
		Ok((rows, cursor))
	}

	fn decode_ollama(
		&self,
		payload: &[u8],
	) -> Result<(Vec<DiscoveredModel>, ProviderCursor), Error> {
		let envelope = crate::codec::ollama::decode_tags(payload)?;
		let rows = envelope
			.models
			.into_iter()
			.map(|model| {
				let wire_model = model.model.unwrap_or_else(|| model.name.clone());
				let family = model
					.details
					.and_then(|details| details.family)
					.map(FamilyId::from);
				self.row(wire_model, Some(model.name), family, None)
			})
			.collect();
		Ok((rows, ProviderCursor::default()))
	}

	fn row(
		&self,
		wire_model: Str,
		display_name: Option<Str>,
		declared_family: Option<FamilyId>,
		availability: Option<ModelAvailability>,
	) -> DiscoveredModel {
		DiscoveredModel {
			provider: self.provider.clone(),
			route: self.route.clone(),
			wire_model: WireModelId::from(wire_model),
			aliases: Box::new([]),
			display_name,
			declared_family,
			declared_operations: OperationBits::empty(),
			declared_capabilities: None,
			declared_limits: None,
			extended_context_mode: None,
			availability,
			source: self.source.clone(),
			observed_at_ms: None,
			updated_at_ms: None,
			deprecated: None,
		}
	}

	fn next_cursor(&self, provider: ProviderCursor) -> Result<Option<Str>, Error> {
		match &self.pagination {
			DiscoveryPagination::SinglePage => Ok(None),
			DiscoveryPagination::Cursor { .. } => Ok(provider.next.or_else(|| {
				provider
					.has_more
					.and_then(|more| more.then_some(provider.last_id).flatten())
			})),
			DiscoveryPagination::PageNumber { first_page, .. } => {
				if provider.has_more != Some(true) {
					return Ok(None);
				}
				let current = self
					.request_cursor
					.as_ref()
					.map(|value| value.as_str().parse::<u32>().map_err(|_| protocol_error()))
					.transpose()?
					.unwrap_or(*first_page);
				current
					.checked_add(1)
					.map(|page| Some(Str::from(page.to_string())))
					.ok_or_else(protocol_error)
			},
		}
	}
}

#[derive(Default)]
struct ProviderCursor {
	next:     Option<Str>,
	has_more: Option<bool>,
	last_id:  Option<Str>,
}

#[derive(Deserialize)]
struct OpenAiEnvelope {
	data:        Vec<OpenAiModel>,
	#[serde(default)]
	next:        Option<Str>,
	#[serde(default)]
	next_cursor: Option<Str>,
	#[serde(default)]
	has_more:    Option<bool>,
	#[serde(default)]
	last_id:     Option<Str>,
}

impl OpenAiEnvelope {
	fn cursor(&self) -> ProviderCursor {
		ProviderCursor {
			next:     self.next_cursor.clone().or_else(|| self.next.clone()),
			has_more: self.has_more,
			last_id:  self.last_id.clone(),
		}
	}
}

#[derive(Deserialize)]
struct OpenAiModel {
	id:        Str,
	#[serde(default, rename = "created")]
	_created:  Option<u64>,
	#[serde(default, rename = "owned_by")]
	_owned_by: Option<Str>,
	#[serde(default, rename = "object")]
	_object:   Option<Str>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoogleModelsEnvelope {
	models:          Vec<GoogleModel>,
	#[serde(default)]
	next_page_token: Option<Str>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoogleModel {
	name: Str,
	#[serde(default)]
	display_name: Option<Str>,
	#[serde(default)]
	base_model_id: Option<Str>,
	#[serde(default)]
	_version: Option<Str>,
	#[serde(default)]
	_description: Option<Str>,
	#[serde(default)]
	supported_generation_methods: Vec<Str>,
	#[serde(default)]
	input_token_limit: Option<u64>,
	#[serde(default)]
	output_token_limit: Option<u64>,
}

#[derive(Deserialize)]
struct AccountEnvelope {
	models:      Vec<AccountModel>,
	#[serde(default)]
	next:        Option<Str>,
	#[serde(default)]
	next_cursor: Option<Str>,
	#[serde(default)]
	has_more:    Option<bool>,
	#[serde(default)]
	last_id:     Option<Str>,
}

impl AccountEnvelope {
	fn cursor(&self) -> ProviderCursor {
		ProviderCursor {
			next:     self.next_cursor.clone().or_else(|| self.next.clone()),
			has_more: self.has_more,
			last_id:  self.last_id.clone(),
		}
	}
}

#[derive(Deserialize)]
struct AccountModel {
	#[serde(default)]
	slug:         Option<Str>,
	#[serde(default)]
	id:           Option<Str>,
	#[serde(default)]
	display_name: Option<Str>,
	#[serde(default)]
	family:       Option<Str>,
	#[serde(default)]
	visibility:   Option<Str>,
}

fn invalid_request() -> Error {
	Error::new(
		ErrorKind::InvalidRequest,
		ErrorPhase::Encoding,
		RetryAction::Never,
		ExecutionReceipt::default(),
	)
}

fn protocol_error() -> Error {
	Error::new(
		ErrorKind::Protocol,
		ErrorPhase::Discovery,
		RetryAction::Never,
		ExecutionReceipt::default(),
	)
}

#[cfg(test)]
mod tests {
	use super::*;

	fn request(cursor: Option<&str>) -> DiscoveryRequest {
		DiscoveryRequest {
			provider:  None,
			route:     None,
			cursor:    cursor.map(Str::from),
			page_size: 100,
			operation: None,
		}
	}

	fn decoder(flavor: DiscoveryFlavor, pagination: DiscoveryPagination) -> DiscoveryDecoder {
		DiscoveryDecoder {
			provider: ProviderId::from("provider"),
			route: RouteId::from("route"),
			source: Str::from("fixture"),
			pagination,
			request_cursor: None,
			flavor,
			done: false,
		}
	}

	fn discovered(
		flavor: DiscoveryFlavor,
		pagination: DiscoveryPagination,
		fixture: &'static [u8],
	) -> (Vec<DiscoveredModel>, Option<Str>) {
		let mut decoder = decoder(flavor, pagination);
		let mut output = None;
		decoder
			.push(Frame::Raw(Bytes::from_static(fixture)), &mut |event| {
				if let RawEvent::DiscoveredModels { rows, next_cursor } = event {
					output = Some((rows, next_cursor));
				}
			})
			.expect("fixture decodes");
		decoder.finish(&mut |_| {}).expect("decoder completes");
		output.expect("discovery event")
	}
	#[test]
	fn standalone_constructors_require_their_exact_discovery_kind() {
		let mut spec = DiscoverySpec {
			id:            omp_llm_catalog::DiscoverySpecId::from("fixture"),
			kind:          DiscoveryKind::OpenAiModels,
			label:         Str::from("fixture"),
			path:          Str::from("/models"),
			pagination:    DiscoveryPagination::SinglePage,
			authoritative: false,
		};
		OpenAiModelsDiscoveryCodec::from_spec(&spec).expect("OpenAI models kind");
		assert_eq!(
			AccountModelsDiscoveryCodec::from_spec(&spec)
				.expect_err("account codec rejects OpenAI kind")
				.kind,
			ErrorKind::InvalidRequest,
		);
		spec.kind = DiscoveryKind::AccountModels;
		AccountModelsDiscoveryCodec::from_spec(&spec).expect("account models kind");
		spec.kind = DiscoveryKind::OllamaTags;
		OllamaTagsDiscoveryCodec::from_spec(&spec).expect("Ollama tags kind");
		spec.kind = DiscoveryKind::Specialized;
		assert_eq!(
			OllamaTagsDiscoveryCodec::from_spec(&spec)
				.expect_err("specialized Ollama discovery remains codec-owned")
				.kind,
			ErrorKind::InvalidRequest,
		);
	}

	#[test]
	fn openai_models_fixture_preserves_only_declared_raw_identity() {
		let (rows, next) = discovered(
			DiscoveryFlavor::OpenAi,
			DiscoveryPagination::SinglePage,
			include_bytes!("../../../../fixtures/llm-oracle/openai/chat/response.models.json"),
		);
		assert_eq!(next, None);
		assert_eq!(rows.len(), 2);
		assert_eq!(rows[0].wire_model.as_str(), "gpt-4.1");
		assert_eq!(rows[1].wire_model.as_str(), "gpt-4.1-mini");
		assert_eq!(rows[0].provider.as_str(), "provider");
		assert_eq!(rows[0].route.as_str(), "route");
		assert_eq!(rows[0].source.as_str(), "fixture");
		assert!(rows[0].declared_operations.is_empty());
		assert_eq!(rows[0].declared_family, None);
		assert_eq!(rows[0].declared_capabilities, None);
	}

	#[test]
	fn google_constructor_requires_exact_kind_and_page_token_query_is_opaque() {
		let mut spec = DiscoverySpec {
			id:            omp_llm_catalog::DiscoverySpecId::from("google-models"),
			kind:          DiscoveryKind::GoogleModels,
			label:         "google-list-models".into(),
			path:          "/v1beta/models".into(),
			pagination:    DiscoveryPagination::Cursor { query_parameter: "pageToken".into() },
			authoritative: false,
		};
		GoogleModelsDiscoveryCodec::from_spec(&spec).expect("Google discovery kind constructs");
		spec.kind = DiscoveryKind::OpenAiModels;
		assert_eq!(
			GoogleModelsDiscoveryCodec::from_spec(&spec)
				.expect_err("wrong discovery kind is rejected")
				.kind,
			ErrorKind::InvalidRequest,
		);
		let uri = discovery_uri(
			"https://generativelanguage.googleapis.com",
			"/v1beta/models",
			&DiscoveryPagination::Cursor { query_parameter: "pageToken".into() },
			&request(Some("opaque +/% token")),
		)
		.expect("Google cursor URI");
		assert_eq!(
			uri.as_str(),
			"https://generativelanguage.googleapis.com/v1beta/models?pageToken=opaque+%2B%2F%25+token",
		);
	}

	#[test]
	fn google_list_models_fixture_preserves_declared_operations_limits_and_cursor() {
		let (rows, next) = discovered(
			DiscoveryFlavor::Google,
			DiscoveryPagination::Cursor { query_parameter: Str::from("pageToken") },
			include_bytes!("fixtures/google_list_models.json"),
		);
		assert_eq!(next.as_ref().map(Str::as_str), Some("page-token-REDACTED"));
		assert_eq!(rows.len(), 2);
		assert_eq!(rows[0].wire_model.as_str(), "gemini-1.5-flash-001");
		assert_eq!(rows[0].display_name.as_ref().map(Str::as_str), Some("Gemini 1.5 Flash"),);
		assert_eq!(
			rows[0]
				.aliases
				.iter()
				.map(WireModelId::as_str)
				.collect::<Vec<_>>(),
			vec!["gemini-1.5-flash"],
		);
		assert!(
			rows[0]
				.declared_operations
				.contains_kind(OperationKind::Chat)
		);
		assert!(
			rows[0]
				.declared_operations
				.contains_kind(OperationKind::CountTokens)
		);
		assert!(
			!rows[0]
				.declared_operations
				.contains_kind(OperationKind::Embed)
		);
		assert_eq!(
			rows[0].declared_limits,
			Some(ModelLimits {
				context_window:        None,
				maximum_input_tokens:  Some(1_048_576),
				maximum_output_tokens: Some(8_192),
				maximum_batch:         None,
			}),
		);
		assert_eq!(rows[1].wire_model.as_str(), "text-embedding-004");
		assert!(
			rows[1]
				.declared_operations
				.contains_kind(OperationKind::Embed)
		);
		assert!(
			!rows[1]
				.declared_operations
				.contains_kind(OperationKind::Chat)
		);
		assert_eq!(rows[0].declared_family, None);
		assert_eq!(rows[0].declared_capabilities, None);
		assert_eq!(rows[0].availability, None);
	}

	#[test]
	fn account_models_fixture_preserves_provider_fields_and_cursor() {
		let (rows, next) = discovered(
			DiscoveryFlavor::Account,
			DiscoveryPagination::Cursor { query_parameter: Str::from("after") },
			include_bytes!("../../../../fixtures/llm-oracle/openai/chat/response.account_models.json"),
		);
		assert_eq!(next.as_ref().map(Str::as_str), Some("retired-account-model"));
		assert_eq!(rows[0].wire_model.as_str(), "gpt-5.2-codex");
		assert_eq!(rows[0].display_name.as_ref().map(Str::as_str), Some("GPT-5.2 Codex"));
		assert_eq!(rows[0].declared_family.as_ref().map(FamilyId::as_str), Some("gpt-5-codex"));
		assert_eq!(rows[0].availability, Some(ModelAvailability::Available));
		assert_eq!(rows[1].availability, Some(ModelAvailability::Disabled));
	}

	#[test]
	fn ollama_tags_fixture_uses_exact_wire_model_and_declared_family() {
		let (rows, next) = discovered(
			DiscoveryFlavor::Ollama,
			DiscoveryPagination::SinglePage,
			include_bytes!("../../../../fixtures/llm-oracle/openai/chat/response.ollama_tags.json"),
		);
		assert_eq!(next, None);
		assert_eq!(rows.len(), 1);
		assert_eq!(rows[0].wire_model.as_str(), "qwen3:8b");
		assert_eq!(rows[0].display_name.as_ref().map(Str::as_str), Some("qwen3:8b"));
		assert_eq!(rows[0].declared_family.as_ref().map(FamilyId::as_str), Some("qwen3"));
	}

	#[test]
	fn pagination_lowers_exact_query_and_single_page_rejects_cursor() {
		let cursor_uri = discovery_uri(
			"https://example.test/v1",
			"models",
			&DiscoveryPagination::Cursor { query_parameter: Str::from("after") },
			&request(Some("opaque +/% cursor")),
		)
		.expect("cursor URI");
		assert_eq!(
			cursor_uri.as_str(),
			"https://example.test/v1/models?after=opaque+%2B%2F%25+cursor"
		);
		let page_uri = discovery_uri(
			"https://example.test",
			"/models",
			&DiscoveryPagination::PageNumber {
				query_parameter: Str::from("page"),
				first_page:      1,
			},
			&request(Some("7")),
		)
		.expect("page URI");
		assert_eq!(page_uri.as_str(), "https://example.test/models?page=7");
		let error = discovery_uri(
			"https://example.test",
			"/models",
			&DiscoveryPagination::SinglePage,
			&request(Some("unexpected")),
		)
		.expect_err("single-page discovery rejects a cursor");
		assert_eq!(error.kind, ErrorKind::InvalidRequest);
	}

	#[test]
	fn known_model_field_with_wrong_type_is_rejected() {
		let mut decoder = decoder(DiscoveryFlavor::OpenAi, DiscoveryPagination::SinglePage);
		let error = decoder
			.push(
				Frame::Raw(Bytes::from_static(br#"{"data":[{"id":7,"created":"yesterday"}]}"#)),
				&mut |_| {},
			)
			.expect_err("typed model fields reject wrong types");
		assert_eq!(error.kind, ErrorKind::Protocol);
		assert_eq!(error.phase, ErrorPhase::Discovery);
	}
}
