//! Typed Cloud Code Assist envelopes, discovery shapes, and stream adapter.

use std::collections::BTreeMap;

use bytes::Bytes;
use omp_core::Str;
use omp_llm_catalog::{
	Availability, ChatCapabilities, DiscoveredModel, ModalityBits, ModelAvailability,
	ModelCapabilities, ModelLimits, OperationBits, OperationKind, ProviderId, ReasoningCapabilities,
	ReasoningFeatureBits, RouteId, WireModelId,
};
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;

use super::gemini::{
	CanonicalGeminiDecoder, GeminiCodec, GeminiDecoder, GenerateContentRequest,
	GenerateContentResponse, GoogleCodecError, GoogleCodecErrorKind, GoogleDecodedEvent,
	GoogleProofScope, GoogleRequestOptions, GoogleThinkingPolicy,
};
use crate::{
	body::BodySource,
	call::{DiscoveryRequest, OperationCall},
	codec::{
		Codec, DecodeContext, Decoder, DecoderState, EncodeContext, EncodedRequest, RawEvent,
		RequestHeader, RequestMethod, SizeBounds,
	},
	error::{Error, ErrorKind, ErrorPhase, RetryAction},
	receipt::ExecutionReceipt,
	transport::{Frame, FramingProtocol},
};
/// CCA streaming generation path.
pub const STREAM_GENERATE_PATH: &str = "/v1internal:streamGenerateContent?alt=sse";
/// CCA model-discovery path.
pub const FETCH_AVAILABLE_MODELS_PATH: &str = "/v1internal:fetchAvailableModels";
/// Gemini CLI's default CCA origin.
pub const GEMINI_CLI_DEFAULT_BASE: &str = "https://cloudcode-pa.googleapis.com";
/// Antigravity's production CCA origin.
pub const ANTIGRAVITY_PRODUCTION_BASE: &str = "https://daily-cloudcode-pa.googleapis.com";
/// Antigravity's sandbox CCA origin.
pub const ANTIGRAVITY_SANDBOX_BASE: &str = "https://daily-cloudcode-pa.sandbox.googleapis.com";
/// Gemini CLI's public client fingerprint metadata.
pub const GEMINI_CLI_CLIENT_METADATA: &str =
	"ideType=IDE_UNSPECIFIED,platform=PLATFORM_UNSPECIFIED,pluginType=GEMINI";
/// Identity instruction emitted by the real Antigravity agent client.
pub const ANTIGRAVITY_SYSTEM_INSTRUCTION: &str =
	"You are Antigravity, a powerful agentic AI coding assistant designed by the Google Deepmind \
	 team working on Advanced Agentic Coding.You are pair programming with a USER to solve their \
	 coding task. The task may require creating a new codebase, modifying or debugging an existing \
	 codebase, or simply answering a question.**Absolute paths only****Proactiveness**";
/// Beta header required for interleaved Claude reasoning in the Antigravity
/// client.
pub const CLAUDE_THINKING_BETA: &str = "interleaved-thinking-2025-05-14";
/// Explicit directive used by the Antigravity forced-tool policy.
pub const ANTIGRAVITY_FORCED_TOOL_DIRECTIVE: &str =
	"TOOL-ONLY TURN. This turn accepts a tool call and nothing else; a text reply here is \
	 discarded unread and you will be re-prompted. Emit the tool call now.";

/// Data-selected CCA client fingerprint.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CcaFlavor {
	/// Gemini CLI request envelope.
	#[default]
	GeminiCli,
	/// Antigravity agent request envelope.
	Antigravity,
}

/// Gemini CLI Cloud Code Assist envelope.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CcaRequestEnvelope {
	/// Opaque wire model name supplied by the selected catalog target.
	pub model:   Str,
	/// Credential project identity supplied by auth/account middleware.
	pub project: Str,
	/// Typed GenerateContent request.
	pub request: GenerateContentRequest,
}

/// Immutable Antigravity attempt metadata owned by conversation/session state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AntigravityRequestMetadata {
	/// Stable trajectory identity.
	pub trajectory_id:                Str,
	/// Stable request identity reused for replay of the same attempt.
	pub request_id:                   Str,
	/// Optional session identity.
	pub session_id:                   Option<Str>,
	/// Previous execution identity.
	pub last_execution_id:            Option<Str>,
	/// Previous trajectory step.
	pub last_step_index:              Option<u64>,
	/// Explicit model enum supplied by catalog policy.
	pub model_enum:                   Option<Str>,
	/// Whether the conversation has used a Claude-routed model.
	pub used_claude:                  bool,
	/// Explicit Antigravity identity instruction supplied by route policy.
	pub system_identity:              Option<Str>,
	/// Whether Antigravity's `VALIDATED` function-calling mode is required.
	pub validated_tool_config:        bool,
	/// Whether planning intentionally appends the forced-tool directive.
	pub append_forced_tool_directive: bool,
	/// Explicit output-token policy supplied by the catalog.
	pub max_output_tokens:            Option<u64>,
}

/// Antigravity-specific request root.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AntigravityRequestEnvelope {
	/// Opaque wire model name.
	pub model:        Str,
	/// Credential project identity.
	pub project:      Str,
	/// Request type expected by Antigravity.
	pub request_type: Str,
	/// Public client user-agent fingerprint copied into the JSON envelope.
	pub user_agent:   Str,
	/// Stable request identity.
	pub request_id:   Str,
	/// Typed request body.
	pub request:      AntigravityGenerateContentRequest,
}

/// Antigravity's GenerateContent request extensions.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AntigravityGenerateContentRequest {
	/// Standard GenerateContent fields flattened into this request.
	#[serde(flatten)]
	pub generate:   GenerateContentRequest,
	/// Optional provider session identity.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub session_id: Option<Str>,
	/// Conversation and model labels.
	pub labels:     AntigravityLabels,
}

/// Antigravity trajectory labels.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct AntigravityLabels {
	/// Trajectory identity.
	pub trajectory_id:            Str,
	/// Previous execution identity.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub last_execution_id:        Option<Str>,
	/// Previous step index encoded as a decimal string by the real client.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub last_step_index:          Option<Str>,
	/// Optional model enum from catalog policy.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub model_enum:               Option<Str>,
	/// Conservative cross-turn Claude usage marker encoded as a string by the
	/// real client.
	pub used_claude:              Str,
	/// Duplicate conservative marker required by the Antigravity wire contract.
	pub used_claude_conservative: Str,
}

/// Wraps a typed GenerateContent request for Gemini CLI CCA.
#[must_use]
pub fn wrap_request(
	request: GenerateContentRequest,
	model: Str,
	project: Str,
) -> CcaRequestEnvelope {
	CcaRequestEnvelope { model, project, request }
}

/// Shapes a typed GenerateContent request as Antigravity without model-name
/// heuristics.
#[must_use]
pub fn wrap_antigravity_request(
	mut request: GenerateContentRequest,
	model: Str,
	project: Str,
	metadata: &AntigravityRequestMetadata,
) -> AntigravityRequestEnvelope {
	if let Some(limit) = metadata.max_output_tokens {
		let config = request
			.generation_config
			.get_or_insert_with(Default::default);
		config.max_output_tokens = Some(limit);
	}
	if let Some(identity) = &metadata.system_identity {
		let system =
			request
				.system_instruction
				.get_or_insert_with(|| super::gemini::GoogleSystemInstruction {
					role:  Some("user".into()),
					parts: Vec::new(),
				});
		system.role = Some("user".into());
		if !system
			.parts
			.first()
			.is_some_and(|part| part.text.as_ref() == Some(identity))
		{
			system.parts.insert(0, super::gemini::GooglePart {
				text: Some(identity.clone()),
				..Default::default()
			});
		}
	}
	if metadata.validated_tool_config {
		request.tool_config = Some(super::gemini::GoogleToolConfig {
			function_calling_config: super::gemini::GoogleFunctionCallingConfig {
				mode:                   "VALIDATED".into(),
				allowed_function_names: Vec::new(),
			},
		});
	}
	if metadata.append_forced_tool_directive {
		request.contents.push(super::gemini::GoogleContent {
			role:  "user".into(),
			parts: vec![super::gemini::GooglePart {
				text: Some(ANTIGRAVITY_FORCED_TOOL_DIRECTIVE.into()),
				..Default::default()
			}],
		});
	}
	let used_claude: Str = metadata.used_claude.to_string().into();
	AntigravityRequestEnvelope {
		model,
		project,
		request_type: "agent".into(),
		user_agent: "antigravity".into(),
		request_id: metadata.request_id.clone(),
		request: AntigravityGenerateContentRequest {
			generate:   request,
			session_id: metadata.session_id.clone(),
			labels:     AntigravityLabels {
				trajectory_id:            metadata.trajectory_id.clone(),
				last_execution_id:        metadata.last_execution_id.clone(),
				last_step_index:          metadata
					.last_step_index
					.map(|value| value.to_string().into()),
				model_enum:               metadata.model_enum.clone(),
				used_claude:              used_claude.clone(),
				used_claude_conservative: used_claude,
			},
		},
	}
}

/// Public, non-secret headers required by a CCA client fingerprint.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CcaHeaders {
	/// User-Agent value.
	pub user_agent:      Str,
	/// Gemini CLI client metadata, when selected.
	pub client_metadata: Option<Str>,
	/// Interleaved-thinking beta, when explicitly selected by catalog policy.
	pub anthropic_beta:  Option<Str>,
	/// Quota project, applied only when explicitly supplied by account policy.
	pub quota_project:   Option<Str>,
}

impl CcaHeaders {
	/// Builds Gemini CLI's public fingerprint from explicit platform
	/// coordinates.
	#[must_use]
	pub fn gemini_cli(model: &str, platform: &str, arch: &str) -> Self {
		Self {
			user_agent:      format!("GeminiCLI/0.46.0/{model} ({platform}; {arch}; terminal)").into(),
			client_metadata: Some(GEMINI_CLI_CLIENT_METADATA.into()),
			anthropic_beta:  None,
			quota_project:   None,
		}
	}

	/// Builds Antigravity's public fingerprint from explicit policy inputs.
	#[must_use]
	pub fn antigravity(
		os: &str,
		arch: &str,
		interleaved_thinking: bool,
		quota_project: Option<Str>,
	) -> Self {
		Self {
			user_agent: format!("antigravity/hub/2.1.4 {os}/{arch}").into(),
			client_metadata: None,
			anthropic_beta: interleaved_thinking.then(|| CLAUDE_THINKING_BETA.into()),
			quota_project,
		}
	}
}

/// Explicit Antigravity lowering policy supplied by catalog/route composition.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AntigravityPolicy {
	/// Identity instruction to prepend, if any.
	pub system_identity:              Option<Str>,
	/// Whether function calling uses `VALIDATED`.
	pub validated_tool_config:        bool,
	/// Whether a forced-tool directive is appended.
	pub append_forced_tool_directive: bool,
	/// Explicit output bound.
	pub max_output_tokens:            Option<u64>,
	/// Explicit model enum label.
	pub model_enum:                   Option<Str>,
	/// Conservative cross-turn Claude evidence.
	pub used_claude:                  bool,
}

/// Pure Cloud Code Assist codec with account project supplied through encode
/// context.
#[derive(Clone, Debug)]
pub struct GoogleCcaCodec {
	gemini:                 GeminiCodec,
	flavor:                 CcaFlavor,
	headers:                CcaHeaders,
	gemini_cli_coordinates: Option<(Str, Str)>,
	antigravity:            Option<AntigravityPolicy>,
}

impl GoogleCcaCodec {
	/// Creates a Gemini CLI CCA codec.
	pub const fn gemini_cli(thinking: Option<GoogleThinkingPolicy>, headers: CcaHeaders) -> Self {
		Self {
			gemini: GeminiCodec::cloud_code_assist(thinking),
			flavor: CcaFlavor::GeminiCli,
			headers,
			antigravity: None,
			gemini_cli_coordinates: None,
		}
	}

	/// Creates an Antigravity CCA codec from explicit non-heuristic policy.
	pub const fn antigravity(
		thinking: Option<GoogleThinkingPolicy>,
		headers: CcaHeaders,
		policy: AntigravityPolicy,
	) -> Self {
		Self {
			gemini: GeminiCodec::cloud_code_assist(thinking),
			flavor: CcaFlavor::Antigravity,
			headers,
			antigravity: Some(policy),
			gemini_cli_coordinates: None,
		}
	}

	/// Creates a route-safe Gemini CLI codec whose model-bearing user agent is
	/// finalized at encode.
	pub fn gemini_cli_for_route(
		thinking: Option<GoogleThinkingPolicy>,
		platform: impl Into<Str>,
		arch: impl Into<Str>,
	) -> Self {
		Self {
			gemini:                 GeminiCodec::cloud_code_assist(thinking),
			flavor:                 CcaFlavor::GeminiCli,
			headers:                CcaHeaders {
				user_agent:      Str::default(),
				client_metadata: Some(GEMINI_CLI_CLIENT_METADATA.into()),
				anthropic_beta:  None,
				quota_project:   None,
			},
			gemini_cli_coordinates: Some((platform.into(), arch.into())),
			antigravity:            None,
		}
	}

	fn request_headers(&self, model: &str) -> Box<[RequestHeader]> {
		let user_agent = self.gemini_cli_coordinates.as_ref().map_or_else(
			|| self.headers.user_agent.clone(),
			|(platform, arch)| {
				format!("GeminiCLI/0.46.0/{model} ({platform}; {arch}; terminal)").into()
			},
		);
		let mut headers = vec![
			RequestHeader { name: "content-type".into(), value: "application/json".into() },
			RequestHeader { name: "accept".into(), value: "text/event-stream".into() },
			RequestHeader { name: "user-agent".into(), value: user_agent },
		];
		if let Some(value) = &self.headers.client_metadata {
			headers.push(RequestHeader { name: "client-metadata".into(), value: value.clone() });
		}
		if let Some(value) = &self.headers.anthropic_beta {
			headers.push(RequestHeader { name: "anthropic-beta".into(), value: value.clone() });
		}
		if let Some(value) = &self.headers.quota_project {
			headers.push(RequestHeader { name: "x-goog-user-project".into(), value: value.clone() });
		}
		headers.into_boxed_slice()
	}

	fn discovery_headers(&self) -> Box<[RequestHeader]> {
		let mut headers = vec![
			RequestHeader { name: "content-type".into(), value: "application/json".into() },
			RequestHeader { name: "accept".into(), value: "application/json".into() },
		];
		if !self.headers.user_agent.is_empty() {
			headers.push(RequestHeader {
				name:  "user-agent".into(),
				value: self.headers.user_agent.clone(),
			});
		}
		if let Some(value) = &self.headers.client_metadata {
			headers.push(RequestHeader { name: "client-metadata".into(), value: value.clone() });
		}
		if let Some(value) = &self.headers.quota_project {
			headers.push(RequestHeader { name: "x-goog-user-project".into(), value: value.clone() });
		}
		headers.into_boxed_slice()
	}

	fn encode_discovery(
		&self,
		base_url: &str,
		request: &DiscoveryRequest,
	) -> Result<EncodedRequest, Error> {
		if request.cursor.is_some() {
			return Err(cca_discovery_error(
				ErrorKind::InvalidRequest,
				ErrorPhase::Encoding,
				"cca_discovery_cursor_unsupported",
			));
		}
		if request
			.operation
			.is_some_and(|operation| operation != OperationKind::Chat)
		{
			return Err(cca_discovery_error(
				ErrorKind::CapabilityMismatch,
				ErrorPhase::Encoding,
				"cca_discovery_operation_unsupported",
			));
		}
		Ok(EncodedRequest {
			operation:   OperationKind::DiscoverModels,
			method:      RequestMethod::Post,
			uri:         format!("{}{}", base_url.trim_end_matches('/'), FETCH_AVAILABLE_MODELS_PATH,)
				.into(),
			headers:     self.discovery_headers(),
			body:        BodySource::Bytes(Bytes::from_static(b"{}")),
			framing:     FramingProtocol::Raw,
			bounds:      SizeBounds {
				request_body: 2,
				frame:        16 * 1024 * 1024,
				response:     256 * 1024 * 1024,
			},
			sealed_body: None,
		})
	}
}

impl Codec for GoogleCcaCodec {
	fn encode(
		&self,
		context: &EncodeContext<'_>,
		operation: &OperationCall,
	) -> Result<EncodedRequest, Error> {
		if let OperationCall::DiscoverModels(request) = operation {
			return self.encode_discovery(context.route.endpoint.base_url.as_str(), request);
		}
		let OperationCall::Chat(request) = operation else {
			return Err(
				cca_provider_error(
					"Cloud Code Assist codec only accepts chat or model-discovery operations",
				)
				.into_inference(false),
			);
		};
		let target = context.target.ok_or_else(|| {
			cca_provider_error("Cloud Code Assist requires a selected wire model target")
				.into_inference(false)
		})?;
		if self.gemini_cli_coordinates.is_some()
			&& target
				.wire_model
				.as_str()
				.bytes()
				.any(|byte| byte.is_ascii_control())
		{
			return Err(
				cca_provider_error(
					"CCA wire model contains control bytes unsafe for the Gemini CLI user agent",
				)
				.into_inference(false),
			);
		}
		if let Some(selection) = context.thinking_selection
			&& selection.wire_model.as_str() != target.wire_model.as_str()
		{
			return Err(
				cca_provider_error(
					"CCA thinking selection wire model does not match the encoded target",
				)
				.into_inference(false),
			);
		}
		let project = context
			.account
			.and_then(|account| account.project.as_ref())
			.ok_or_else(|| {
				cca_provider_error("Cloud Code Assist requires an account project before encoding")
					.into_inference(false)
			})?;
		let options = GoogleRequestOptions {
			proof_scope: Some(GoogleProofScope {
				provider: context.route.provider.clone(),
				codec:    target.codec.clone(),
			}),
			..GoogleRequestOptions::default()
		};
		let projection = self
			.gemini
			.project_for_encode(request, &options, context.thinking_policy, context.thinking_selection)
			.map_err(|error| error.into_inference(false))?;
		if let Some(adjustment) = projection.adjustments.first() {
			return Err(
				cca_provider_error(format!(
					"planning did not account for unsupported CCA feature `{}`: {}",
					adjustment.what, adjustment.detail,
				))
				.into_inference(false),
			);
		}
		let model: Str = target.wire_model.as_str().into();
		let project: Str = project.as_str().into();
		let body = match self.flavor {
			CcaFlavor::GeminiCli => {
				serde_json::to_vec(&wrap_request(projection.request, model, project))
			},
			CcaFlavor::Antigravity => {
				let policy = self.antigravity.as_ref().ok_or_else(|| {
					cca_provider_error("Antigravity codec is missing explicit lowering policy")
						.into_inference(false)
				})?;
				let trajectory_id = context.session.map_or_else(
					|| context.request_id.as_str().into(),
					|session| session.conversation.as_str().into(),
				);
				let metadata = AntigravityRequestMetadata {
					trajectory_id,
					request_id: context.request_id.as_str().into(),
					session_id: context
						.session
						.map(|session| session.conversation.as_str().into()),
					last_execution_id: None,
					last_step_index: None,
					model_enum: policy.model_enum.clone(),
					used_claude: policy.used_claude,
					system_identity: policy.system_identity.clone(),
					validated_tool_config: policy.validated_tool_config,
					append_forced_tool_directive: policy.append_forced_tool_directive,
					max_output_tokens: policy.max_output_tokens,
				};
				serde_json::to_vec(&wrap_antigravity_request(
					projection.request,
					model,
					project,
					&metadata,
				))
			},
		}
		.map_err(|error| {
			cca_decode_error(format!("invalid CCA request: {error}")).into_inference(false)
		})?;
		Ok(EncodedRequest {
			operation:   OperationKind::Chat,
			method:      RequestMethod::Post,
			uri:         format!(
				"{}{}",
				target.endpoint.base_url.as_str().trim_end_matches('/'),
				STREAM_GENERATE_PATH,
			)
			.into(),
			headers:     self.request_headers(target.wire_model.as_str()),
			body:        BodySource::Bytes(Bytes::from(body)),
			framing:     FramingProtocol::Sse,
			bounds:      SizeBounds {
				request_body: 32 * 1024 * 1024,
				frame:        16 * 1024 * 1024,
				response:     512 * 1024 * 1024,
			},
			sealed_body: None,
		})
	}

	fn decoder(&self, context: &DecodeContext<'_>) -> Result<DecoderState, Error> {
		if context.operation == OperationKind::DiscoverModels
			&& matches!(context.operation_call, OperationCall::DiscoverModels(_))
		{
			return Ok(Box::new(CcaDiscoveryDecoder {
				provider: context.provider.clone(),
				route:    context.route.clone(),
				done:     false,
			}));
		}
		if context.operation == OperationKind::Chat
			&& matches!(context.operation_call, OperationCall::Chat(_))
		{
			return Ok(Box::new(CanonicalCcaDecoder::default()));
		}
		Err(
			cca_decode_error(
				"Cloud Code Assist decoder requires matching chat or model-discovery intent",
			)
			.into_inference(false),
		)
	}
}

#[derive(Debug, Default)]
struct CanonicalCcaDecoder {
	inner:     CcaDecoder,
	canonical: CanonicalGeminiDecoder,
}

impl Decoder for CanonicalCcaDecoder {
	fn push(&mut self, frame: Frame, emit: &mut dyn FnMut(RawEvent)) -> Result<(), Error> {
		let data = match frame {
			Frame::Sse(event) => event.data,
			Frame::Raw(data) => data,
			_ => {
				return Err(
					cca_decode_error("Cloud Code Assist decoder requires SSE or unary raw frames")
						.into_inference(self.canonical.committed()),
				);
			},
		};
		let events = self
			.inner
			.push_json(&data)
			.map_err(|error| error.into_inference(self.canonical.committed()))?;
		self.canonical.emit_events(events, emit);
		Ok(())
	}

	fn finish(&mut self, emit: &mut dyn FnMut(RawEvent)) -> Result<(), Error> {
		let events = self
			.inner
			.finish()
			.map_err(|error| error.into_inference(self.canonical.committed()))?;
		self.canonical.emit_events(events, emit);
		Ok(())
	}
}

/// Typed CCA stream envelope.
#[derive(Debug, Deserialize)]
pub struct CcaResponseEnvelope {
	/// Embedded GenerateContent response.
	pub response: Option<GenerateContentResponse>,
	/// In-band CCA error.
	pub error:    Option<CcaWireError>,
}

/// Typed CCA in-band error.
#[derive(Debug, Deserialize, Serialize)]
pub struct CcaWireError {
	/// Numeric status.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub code:    Option<u16>,
	/// Human-readable detail.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub message: Option<Str>,
	/// Symbolic status.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub status:  Option<Str>,
}

/// Decodes and unwraps one CCA response envelope without generic JSON
/// traversal.
pub fn unwrap_response(data: &[u8]) -> Result<GenerateContentResponse, GoogleCodecError> {
	let raw: Box<RawValue> = serde_json::from_slice(data)
		.map_err(|error| cca_decode_error(format!("invalid CCA response JSON: {error}")))?;
	if !raw.get().trim_start().starts_with('{') {
		return Err(cca_provider_error("CCA stream event is not an object"));
	}
	let envelope: CcaResponseEnvelope = serde_json::from_str(raw.get())
		.map_err(|error| cca_decode_error(format!("invalid CCA response JSON: {error}")))?;
	if let Some(error) = envelope.error {
		let encoded = serde_json::to_string(&error)
			.map_err(|failure| cca_decode_error(format!("invalid CCA error envelope: {failure}")))?;
		return Err(cca_provider_error(format!("CCA in-band error: {encoded}")));
	}
	envelope
		.response
		.ok_or_else(|| cca_provider_error("CCA stream event has no response"))
}

/// Converts opaque continuation-proof bytes to CCA's UTF-8 JSON string
/// representation.
pub fn thought_signature_to_wire(signature: &Bytes) -> Result<Str, GoogleCodecError> {
	std::str::from_utf8(signature)
		.map(Str::from)
		.map_err(|error| cca_provider_error(format!("CCA thought signature is not UTF-8: {error}")))
}

/// Converts CCA's continuation-proof string to opaque canonical bytes.
#[must_use]
pub fn thought_signature_from_wire(signature: &str) -> Bytes {
	Bytes::copy_from_slice(signature.as_bytes())
}

/// CCA discovery response root.
#[derive(Debug, Deserialize)]
pub struct CcaDiscoveryResponse {
	/// Models keyed by their opaque wire IDs.
	pub models: BTreeMap<Str, CcaDiscoveredModel>,
}

/// One typed CCA model-discovery record.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CcaDiscoveredModel {
	/// Human-readable name.
	pub display_name:      Option<Str>,
	/// Explicit internal-only marker.
	#[serde(default)]
	pub is_internal:       bool,
	/// Explicit reasoning support evidence.
	pub supports_thinking: Option<bool>,
	/// Explicit image-input support evidence.
	pub supports_images:   Option<bool>,
	/// Explicit context bound.
	pub max_tokens:        Option<u64>,
	/// Explicit output bound.
	pub max_output_tokens: Option<u64>,
}

/// Conservative discovery record passed to catalog normalization.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CcaModelRecord {
	/// Opaque wire model ID.
	pub wire_id:           Str,
	/// Display name if the server supplied one.
	pub display_name:      Option<Str>,
	/// Whether the provider declares this row internal-only.
	pub is_internal:       bool,
	/// Reasoning support evidence; absence remains unknown.
	pub supports_thinking: Option<bool>,
	/// Image support evidence; absence remains unknown.
	pub supports_images:   Option<bool>,
	/// Context bound if supplied.
	pub max_tokens:        Option<u64>,
	/// Output bound if supplied.
	pub max_output_tokens: Option<u64>,
}

/// Parses model discovery without family inference, defaults, or model-name
/// filtering.
pub fn parse_cca_models(body: &[u8]) -> Result<Vec<CcaModelRecord>, GoogleCodecError> {
	let payload: CcaDiscoveryResponse = serde_json::from_slice(body)
		.map_err(|error| cca_decode_error(format!("invalid CCA model discovery JSON: {error}")))?;
	Ok(payload
		.models
		.into_iter()
		.filter_map(|(wire_id, model)| {
			if wire_id.is_empty() {
				return None;
			}
			Some(CcaModelRecord {
				wire_id,
				display_name: model.display_name.filter(|name| !name.is_empty()),
				is_internal: model.is_internal,
				supports_thinking: model.supports_thinking,
				supports_images: model.supports_images,
				max_tokens: model.max_tokens,
				max_output_tokens: model.max_output_tokens,
			})
		})
		.collect())
}

fn cca_discovered_rows(
	provider: &ProviderId,
	route: &RouteId,
	records: Vec<CcaModelRecord>,
) -> Vec<DiscoveredModel> {
	records
		.into_iter()
		.map(|record| {
			let detailed = record.supports_thinking.is_some() || record.supports_images.is_some();
			let declared_capabilities = detailed.then(|| ModelCapabilities {
				operations:    OperationBits::for_kind(OperationKind::Chat),
				chat:          Some(ChatCapabilities {
					roles:             Availability::Unknown,
					mid_session_roles: Availability::Unknown,
					tools:             Availability::Unknown,
					structured_output: Availability::Unknown,
					grammar:           Availability::Unknown,
					text_verbosity:    Availability::Unknown,
					reasoning:         match record.supports_thinking {
						Some(true) => Availability::Native(ReasoningCapabilities {
							features:              ReasoningFeatureBits::empty(),
							efforts:               Box::new([]),
							minimum_budget_tokens: None,
							maximum_budget_tokens: None,
						}),
						Some(false) => Availability::Unsupported,
						None => Availability::Unknown,
					},
					input_modalities:  match record.supports_images {
						Some(true) => Availability::Native(ModalityBits::IMAGE),
						Some(false) => Availability::Unsupported,
						None => Availability::Unknown,
					},
					hosted_tools:      Availability::Unknown,
					prompt_caching:    Availability::Unknown,
					service_tiers:     Availability::Unknown,
					sampling:          Availability::Unknown,
					safety:            Availability::Unknown,
					determinism:       Availability::Unknown,
					server_state:      Availability::Unknown,
					logprobs:          Availability::Unknown,
				}),
				embeddings:    None,
				image:         None,
				video:         None,
				speech:        None,
				transcription: None,
				realtime:      None,
				search:        None,
				tokenization:  None,
			});
			let declared_limits = (record.max_tokens.is_some() || record.max_output_tokens.is_some())
				.then_some(ModelLimits {
					context_window:        record.max_tokens,
					maximum_input_tokens:  None,
					maximum_output_tokens: record.max_output_tokens,
					maximum_batch:         None,
				});
			DiscoveredModel {
				provider: provider.clone(),
				route: route.clone(),
				wire_model: WireModelId::from(record.wire_id),
				aliases: Box::new([]),
				display_name: record.display_name,
				declared_family: None,
				declared_operations: OperationBits::for_kind(OperationKind::Chat),
				declared_capabilities,
				declared_limits,
				extended_context_mode: None,
				availability: Some(if record.is_internal {
					ModelAvailability::Disabled
				} else {
					ModelAvailability::Available
				}),
				source: "cca-fetch-available-models".into(),
				observed_at_ms: None,
				updated_at_ms: None,
				deprecated: None,
			}
		})
		.collect()
}

#[derive(Debug)]
struct CcaDiscoveryDecoder {
	provider: ProviderId,
	route:    RouteId,
	done:     bool,
}

impl CcaDiscoveryDecoder {
	fn error(&self, code: &'static str) -> Error {
		let mut error = cca_discovery_error(ErrorKind::Protocol, ErrorPhase::Discovery, code);
		error.provider = Some(self.provider.clone());
		error.route = Some(self.route.clone());
		error
	}
}

impl Decoder for CcaDiscoveryDecoder {
	fn push(&mut self, frame: Frame, emit: &mut dyn FnMut(RawEvent)) -> Result<(), Error> {
		if self.done {
			return Err(self.error("cca_discovery_multiple_bodies"));
		}
		let Frame::Raw(body) = frame else {
			return Err(self.error("cca_discovery_expected_raw_body"));
		};
		let records =
			parse_cca_models(&body).map_err(|_| self.error("cca_discovery_invalid_payload"))?;
		self.done = true;
		emit(RawEvent::DiscoveredModels {
			rows:        cca_discovered_rows(&self.provider, &self.route, records),
			next_cursor: None,
		});
		Ok(())
	}

	fn finish(&mut self, _emit: &mut dyn FnMut(RawEvent)) -> Result<(), Error> {
		if self.done {
			Ok(())
		} else {
			Err(self.error("cca_discovery_empty_response"))
		}
	}
}

/// Incremental CCA decoder that unwraps envelopes then delegates
/// GenerateContent semantics.
#[derive(Debug, Default)]
pub struct CcaDecoder {
	gemini:    GeminiDecoder,
	visible:   CcaVisibleTextFilter,
	completed: bool,
}

impl CcaDecoder {
	/// Decodes one complete CCA SSE data field.
	pub fn push_json(&mut self, data: &[u8]) -> Result<Vec<GoogleDecodedEvent>, GoogleCodecError> {
		if self.completed || data.is_empty() {
			return Ok(Vec::new());
		}
		if data == b"[DONE]" {
			return self.finish();
		}
		let response = unwrap_response(data)?;
		let events = self.gemini.push_response(response)?;
		Ok(self.filter(events))
	}

	/// Completes the embedded GenerateContent stream and flushes visible
	/// buffered text.
	pub fn finish(&mut self) -> Result<Vec<GoogleDecodedEvent>, GoogleCodecError> {
		if self.completed {
			return Ok(Vec::new());
		}
		self.completed = true;
		let terminal = self.gemini.finish()?;
		let mut events = self.filter(terminal);
		self.visible.finish(&mut events);
		Ok(events)
	}

	fn filter(&mut self, events: Vec<GoogleDecodedEvent>) -> Vec<GoogleDecodedEvent> {
		let mut output = Vec::new();
		for event in events {
			match event {
				GoogleDecodedEvent::Text { index, text, signature } => {
					self.visible.push(index, text, signature, &mut output);
				},
				GoogleDecodedEvent::Thinking { index, text, signature } => {
					self.visible.flush_pending(&mut output);
					output.push(GoogleDecodedEvent::Thinking { index, text, signature });
				},
				GoogleDecodedEvent::Completed(_) | GoogleDecodedEvent::Error(_) => {
					self.visible.finish(&mut output);
					output.push(event);
				},
				_ => output.push(event),
			}
		}
		output
	}
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum VisibleMode {
	#[default]
	PlanningPrefix,
	Text,
	ThinkingTag,
}

#[derive(Debug, Default)]
struct CcaVisibleTextFilter {
	mode:              VisibleMode,
	pending:           String,
	pending_index:     u32,
	pending_signature: Option<Str>,
}

impl CcaVisibleTextFilter {
	fn push(
		&mut self,
		index: u32,
		text: Str,
		signature: Option<Str>,
		output: &mut Vec<GoogleDecodedEvent>,
	) {
		if self.pending.is_empty() {
			self.pending_index = index;
		}
		self.pending.push_str(&text);
		if signature.is_some() {
			self.pending_signature = signature;
		}
		loop {
			let progressed = match self.mode {
				VisibleMode::PlanningPrefix => self.consume_planning_prefix(output),
				VisibleMode::Text => self.consume_text(output),
				VisibleMode::ThinkingTag => self.consume_thinking(output),
			};
			if !progressed {
				break;
			}
		}
	}

	fn consume_planning_prefix(&mut self, output: &mut Vec<GoogleDecodedEvent>) -> bool {
		if !self.pending.starts_with('{') {
			self.mode = VisibleMode::Text;
			return true;
		}
		let Some(end) = json_object_end(self.pending.as_bytes()) else {
			return false;
		};
		let candidate = &self.pending[..end];
		#[derive(Deserialize)]
		struct PlanningProbe {
			thought: Option<Box<RawValue>>,
			call:    Option<Box<RawValue>>,
		}
		let planning = serde_json::from_str::<PlanningProbe>(candidate)
			.is_ok_and(|probe| probe.thought.is_some() || probe.call.is_some());
		if planning {
			self.pending.drain(..end);
			if self.pending.is_empty() {
				self.pending_signature = None;
			}
		} else {
			let visible: Str = self.pending.drain(..end).collect::<String>().into();
			output.push(GoogleDecodedEvent::Text {
				index:     self.pending_index,
				text:      visible,
				signature: self.pending_signature.take(),
			});
		}
		self.mode = VisibleMode::Text;
		true
	}

	fn consume_text(&mut self, output: &mut Vec<GoogleDecodedEvent>) -> bool {
		const OPEN: &str = "<thinking>";
		if let Some(position) = self.pending.find(OPEN) {
			if position > 0 {
				let visible: Str = self.pending.drain(..position).collect::<String>().into();
				output.push(GoogleDecodedEvent::Text {
					index:     self.pending_index,
					text:      visible,
					signature: self.pending_signature.take(),
				});
			}
			self.pending.drain(..OPEN.len());
			self.mode = VisibleMode::ThinkingTag;
			return true;
		}
		let retain = longest_suffix_prefix(&self.pending, OPEN);
		if self.pending.len() > retain {
			let emit_len = self.pending.len() - retain;
			let visible: Str = self.pending.drain(..emit_len).collect::<String>().into();
			output.push(GoogleDecodedEvent::Text {
				index:     self.pending_index,
				text:      visible,
				signature: self.pending_signature.take(),
			});
		}
		false
	}

	fn consume_thinking(&mut self, output: &mut Vec<GoogleDecodedEvent>) -> bool {
		const CLOSE: &str = "</thinking>";
		if let Some(position) = self.pending.find(CLOSE) {
			if position > 0 {
				let thought: Str = self.pending.drain(..position).collect::<String>().into();
				output.push(GoogleDecodedEvent::Thinking {
					index:     self.pending_index,
					text:      thought,
					signature: self.pending_signature.take(),
				});
			}
			self.pending.drain(..CLOSE.len());
			self.mode = VisibleMode::Text;
			return true;
		}
		let retain = longest_suffix_prefix(&self.pending, CLOSE);
		if self.pending.len() > retain {
			let emit_len = self.pending.len() - retain;
			let thought: Str = self.pending.drain(..emit_len).collect::<String>().into();
			output.push(GoogleDecodedEvent::Thinking {
				index:     self.pending_index,
				text:      thought,
				signature: self.pending_signature.take(),
			});
		}
		false
	}

	fn flush_pending(&mut self, output: &mut Vec<GoogleDecodedEvent>) {
		if self.pending.is_empty() {
			return;
		}
		let text: Str = std::mem::take(&mut self.pending).into();
		if matches!(self.mode, VisibleMode::ThinkingTag) {
			output.push(GoogleDecodedEvent::Thinking {
				index: self.pending_index,
				text,
				signature: self.pending_signature.take(),
			});
		} else {
			output.push(GoogleDecodedEvent::Text {
				index: self.pending_index,
				text,
				signature: self.pending_signature.take(),
			});
		}
	}

	fn finish(&mut self, output: &mut Vec<GoogleDecodedEvent>) {
		if matches!(self.mode, VisibleMode::PlanningPrefix) && self.pending.starts_with('{') {
			self.pending.clear();
			self.pending_signature = None;
			return;
		}
		self.flush_pending(output);
	}
}

fn cca_discovery_error(kind: ErrorKind, phase: ErrorPhase, code: &'static str) -> Error {
	let mut error = Error::new(kind, phase, RetryAction::Never, ExecutionReceipt::default());
	error.code = Some(code.into());
	error
}

fn longest_suffix_prefix(value: &str, marker: &str) -> usize {
	let limit = value.len().min(marker.len().saturating_sub(1));
	(1..=limit)
		.rev()
		.find(|length| value.ends_with(&marker[..*length]))
		.unwrap_or(0)
}

fn json_object_end(input: &[u8]) -> Option<usize> {
	let mut depth = 0_u32;
	let mut string = false;
	let mut escaped = false;
	for (index, byte) in input.iter().copied().enumerate() {
		if string {
			if escaped {
				escaped = false;
			} else if byte == b'\\' {
				escaped = true;
			} else if byte == b'\"' {
				string = false;
			}
			continue;
		}
		match byte {
			b'\"' => string = true,
			b'{' => depth = depth.saturating_add(1),
			b'}' => {
				depth = depth.saturating_sub(1);
				if depth == 0 {
					return Some(index + 1);
				}
			},
			_ => {},
		}
	}
	None
}

fn cca_provider_error(detail: impl Into<Str>) -> GoogleCodecError {
	GoogleCodecError {
		kind:           GoogleCodecErrorKind::Upstream,
		detail:         detail.into(),
		status:         None,
		code:           None,
		retry_after_ms: 0,
	}
}

fn cca_decode_error(detail: impl Into<Str>) -> GoogleCodecError {
	GoogleCodecError {
		kind:           GoogleCodecErrorKind::Decode,
		detail:         detail.into(),
		status:         None,
		code:           None,
		retry_after_ms: 0,
	}
}

#[cfg(test)]
mod tests {
	use serde_json::Value;

	use super::*;

	#[test]
	fn request_and_response_envelopes_match_oracle() {
		let request = GenerateContentRequest {
			contents: vec![super::super::gemini::GoogleContent {
				role:  "user".into(),
				parts: vec![super::super::gemini::GooglePart {
					text: Some("hello".into()),
					..Default::default()
				}],
			}],
			..Default::default()
		};
		let wrapped = wrap_request(request, "gemini-3.5-flash".into(), "project-REDACTED".into());
		let actual = serde_json::to_value(wrapped).expect("envelope serializes");
		let expected: Value = serde_json::from_str(r#"{"model":"gemini-3.5-flash","project":"project-REDACTED","request":{"contents":[{"role":"user","parts":[{"text":"hello"}]}]}}"#).expect("expected JSON");
		assert_eq!(actual, expected);
		assert!(unwrap_response(br#"{"response":{"candidates":[]}}"#).is_ok());
	}

	#[test]
	fn parallel_function_responses_match_recorded_envelope() {
		let request = crate::call::ChatRequest {
			messages:          vec![
				crate::call::Message {
					role:    crate::call::Role::Tool,
					content: vec![crate::call::ContentPart::ToolResult {
						call:     crate::id::ToolCallId::new("call_a"),
						name:     Some("read".into()),
						content:  vec![crate::call::ToolResultContent::Text("A".into())].into(),
						is_error: false,
					}]
					.into(),
					name:    None,
				},
				crate::call::Message {
					role:    crate::call::Role::Tool,
					content: vec![crate::call::ContentPart::ToolResult {
						call:     crate::id::ToolCallId::new("call_b"),
						name:     Some("grep".into()),
						content:  vec![crate::call::ToolResultContent::Text("B".into())].into(),
						is_error: false,
					}]
					.into(),
					name:    None,
				},
			]
			.into(),
			tools:             std::sync::Arc::from([]),
			hosted_tools:      std::sync::Arc::from([]),
			tool_choice:       crate::call::Setting::Unset,
			output:            crate::call::Setting::Unset,
			reasoning:         crate::call::Setting::Unset,
			verbosity:         crate::call::Setting::Unset,
			cache_retention:   crate::call::Setting::Unset,
			service_tier:      crate::call::Setting::Unset,
			sampling:          crate::call::Sampling::default(),
			max_output_tokens: None,
			top_logprobs:      None,
			safety:            std::sync::Arc::from([]),
			negotiation:       Default::default(),
		};
		let projected = GeminiCodec::cloud_code_assist(None)
			.project(&request, &GoogleRequestOptions::default())
			.expect("parallel function responses project");
		assert_eq!(projected.request.contents.len(), 1);
		assert_eq!(projected.request.contents[0].parts.len(), 2);
		let actual =
			wrap_request(projected.request, "gemini-3.5-flash".into(), "project-REDACTED".into());

		#[derive(Deserialize)]
		struct Oracle {
			wire_body: CcaRequestEnvelope,
		}
		let oracle: Oracle = serde_json::from_str(include_str!(
			"../../../../fixtures/llm-oracle/google/recorded/google_cca/request.\
			 parallel_function_responses.json"
		))
		.expect("recorded CCA parallel response fixture parses");
		assert_eq!(
			serde_json::to_value(actual).expect("actual envelope serializes"),
			serde_json::to_value(oracle.wire_body).expect("oracle envelope serializes"),
		);
	}

	#[test]
	fn antigravity_adapter_matches_recorded_request() {
		let schema: super::super::gemini::GoogleSchema =
			serde_json::from_str(r#"{"type":"object"}"#).expect("typed schema");
		let request = GenerateContentRequest {
			contents: vec![super::super::gemini::GoogleContent {
				role:  "user".into(),
				parts: vec![super::super::gemini::GooglePart {
					text: Some("inspect the repository".into()),
					..Default::default()
				}],
			}],
			system_instruction: Some(super::super::gemini::GoogleSystemInstruction {
				role:  None,
				parts: vec![super::super::gemini::GooglePart {
					text: Some("Use the repository tools.".into()),
					..Default::default()
				}],
			}),
			tools: vec![super::super::gemini::GoogleTool {
				function_declarations: vec![super::super::gemini::GoogleFunctionDeclaration {
					name:                   "read".into(),
					description:            None,
					parameters_json_schema: None,
					parameters:             Some(schema),
				}],
				..Default::default()
			}],
			..Default::default()
		};
		let metadata = AntigravityRequestMetadata {
			trajectory_id:                "22222222-2222-2222-2222-222222222222".into(),
			request_id:                   "agent/11111111-1111-1111-1111-111111111111/1700000000000/\
			                               22222222-2222-2222-2222-222222222222/2"
				.into(),
			session_id:                   Some("-8392019482710394817".into()),
			last_execution_id:            Some("execution-before".into()),
			last_step_index:              Some(1),
			model_enum:                   Some("MODEL_PLACEHOLDER_M20".into()),
			used_claude:                  false,
			system_identity:              Some(ANTIGRAVITY_SYSTEM_INSTRUCTION.into()),
			validated_tool_config:        true,
			append_forced_tool_directive: false,
			max_output_tokens:            Some(65_536),
		};
		let actual = serde_json::to_value(wrap_antigravity_request(
			request,
			"gemini-3.5-flash-low".into(),
			"project-REDACTED".into(),
			&metadata,
		))
		.expect("request serializes");
		let expected: Value = serde_json::from_str(include_str!(
			"../../../../fixtures/llm-oracle/google/recorded/google_cca/request.antigravity.json"
		))
		.expect("recorded request parses");
		assert_eq!(actual, expected);
	}

	#[test]
	fn malformed_envelopes_report_exact_shapes() {
		assert_eq!(
			unwrap_response(b"[]")
				.expect_err("array rejected")
				.detail
				.as_str(),
			"CCA stream event is not an object"
		);
		assert_eq!(
			unwrap_response(b"{}")
				.expect_err("missing response")
				.detail
				.as_str(),
			"CCA stream event has no response"
		);
		assert_eq!(
			unwrap_response(br#"{"error":{"code":500,"message":"upstream failed"}}"#)
				.expect_err("error envelope")
				.detail
				.as_str(),
			"CCA in-band error: {\"code\":500,\"message\":\"upstream failed\"}",
		);
		let signature_error = thought_signature_to_wire(&Bytes::from_static(&[0xff]))
			.expect_err("non-UTF-8 CCA signature is rejected");
		assert!(
			signature_error
				.detail
				.starts_with("CCA thought signature is not UTF-8:")
		);
		let stream_error = CcaDecoder::default()
			.push_json(b"not-json")
			.expect_err("malformed CCA stream JSON is rejected");
		assert!(
			stream_error
				.detail
				.starts_with("invalid CCA response JSON:")
		);
	}

	#[test]
	fn discovery_preserves_unknown_evidence() {
		let records = parse_cca_models(br#"{"models":{"model-a":{"displayName":"A","supportsThinking":true,"maxTokens":1000},"internal":{"isInternal":true}}}"#)
			.expect("discovery parses");
		assert_eq!(records.len(), 2);
		let public = records
			.iter()
			.find(|record| record.wire_id.as_str() == "model-a")
			.expect("public record");
		assert!(!public.is_internal);
		assert_eq!(public.supports_thinking, Some(true));
		assert_eq!(public.supports_images, None);
		assert_eq!(public.max_output_tokens, None);
		let internal = records
			.iter()
			.find(|record| record.wire_id.as_str() == "internal")
			.expect("internal record remains explicit raw evidence");
		assert!(internal.is_internal);
		let rows = cca_discovered_rows(
			&ProviderId::from("google-cca"),
			&RouteId::from("google-cca-primary"),
			records,
		);
		let internal = rows
			.iter()
			.find(|row| row.wire_model.as_str() == "internal")
			.expect("internal discovered row");
		assert_eq!(internal.availability, Some(ModelAvailability::Disabled));
	}

	#[test]
	fn discovery_decoder_projects_explicit_evidence_without_name_inference() {
		let mut decoder = CcaDiscoveryDecoder {
			provider: ProviderId::from("google-cca"),
			route:    RouteId::from("google-cca-primary"),
			done:     false,
		};
		let mut events = Vec::new();
		decoder.push(
			Frame::Raw(Bytes::from_static(
				br#"{"models":{"opaque-model":{"displayName":"Opaque","supportsThinking":true,"supportsImages":false,"maxTokens":1000000,"maxOutputTokens":65536}}}"#,
			)),
			&mut |event| events.push(event),
		).expect("typed CCA discovery body decodes");
		decoder
			.finish(&mut |_| {})
			.expect("completed discovery finishes");
		let [RawEvent::DiscoveredModels { rows, next_cursor }] = events.as_slice() else {
			panic!("expected exactly one discovered-model page");
		};
		assert!(next_cursor.is_none());

		assert_eq!(rows.len(), 1);
		assert_eq!(rows[0].wire_model.as_str(), "opaque-model");
		assert_eq!(rows[0].declared_family, None);
		assert!(
			rows[0]
				.declared_operations
				.contains_kind(OperationKind::Chat)
		);
		assert_eq!(
			rows[0].declared_limits,
			Some(ModelLimits {
				context_window:        Some(1_000_000),
				maximum_input_tokens:  None,
				maximum_output_tokens: Some(65_536),
				maximum_batch:         None,
			}),
		);
		let chat = rows[0]
			.declared_capabilities
			.as_ref()
			.and_then(|capabilities| capabilities.chat.as_ref())
			.expect("explicit discovery flags produce detailed chat evidence");
		assert!(matches!(chat.reasoning, Availability::Native(_)));
		assert!(matches!(chat.input_modalities, Availability::Unsupported));
	}
	#[test]
	fn discovery_request_matches_fetch_available_models_wire_contract() {
		let codec = GoogleCcaCodec::antigravity(
			None,
			CcaHeaders::antigravity("darwin", "arm64", false, Some("quota-project".into())),
			AntigravityPolicy::default(),
		);
		let request = DiscoveryRequest {
			provider:  None,
			route:     None,
			cursor:    None,
			page_size: 100,
			operation: Some(OperationKind::Chat),
		};
		let encoded = codec
			.encode_discovery("https://daily-cloudcode-pa.googleapis.com/", &request)
			.expect("CCA discovery request encodes");
		assert_eq!(encoded.operation, OperationKind::DiscoverModels);
		assert_eq!(encoded.method, RequestMethod::Post);
		assert_eq!(
			encoded.uri.as_str(),
			"https://daily-cloudcode-pa.googleapis.com/v1internal:fetchAvailableModels",
		);
		assert_eq!(encoded.framing, FramingProtocol::Raw);
		let BodySource::Bytes(body) = encoded.body else {
			panic!("inline discovery body")
		};
		assert_eq!(body.as_ref(), b"{}");
		assert!(encoded.headers.iter().any(|header| {
			header.name.as_str() == "accept" && header.value.as_str() == "application/json"
		}));
		assert!(encoded.headers.iter().any(|header| {
			header.name.as_str() == "user-agent"
				&& header.value.as_str() == "antigravity/hub/2.1.4 darwin/arm64"
		}));
		assert!(encoded.headers.iter().any(|header| {
			header.name.as_str() == "x-goog-user-project" && header.value.as_str() == "quota-project"
		}));
		let mut unsupported = request.clone();
		unsupported.operation = Some(OperationKind::Embed);
		let error =
			match codec.encode_discovery("https://daily-cloudcode-pa.googleapis.com", &unsupported) {
				Ok(_) => panic!("CCA discovery accepted a non-chat capability filter"),
				Err(error) => error,
			};
		assert_eq!(error.kind, ErrorKind::CapabilityMismatch);
		assert_eq!(error.phase, ErrorPhase::Encoding);
		let mut continued = request;
		continued.cursor = Some("not-supported".into());
		let error =
			match codec.encode_discovery("https://daily-cloudcode-pa.googleapis.com", &continued) {
				Ok(_) => panic!("CCA discovery accepted an unsupported cursor"),
				Err(error) => error,
			};
		assert_eq!(error.kind, ErrorKind::InvalidRequest);
		assert_eq!(error.phase, ErrorPhase::Encoding);
	}

	#[test]
	fn discovery_decoder_rejects_missing_map_wrong_types_and_extra_frames() {
		for malformed in [
			br#"{}"#.as_slice(),
			br#"{"models":[]}"#.as_slice(),
			br#"{"models":{"model":{"supportsThinking":"yes"}}}"#.as_slice(),
		] {
			let mut decoder = CcaDiscoveryDecoder {
				provider: ProviderId::from("google-cca"),
				route:    RouteId::from("google-cca-primary"),
				done:     false,
			};
			let error = decoder
				.push(Frame::Raw(Bytes::copy_from_slice(malformed)), &mut |_| {})
				.expect_err("malformed CCA discovery body is rejected");
			assert_eq!(error.kind, ErrorKind::Protocol);
			assert_eq!(error.phase, ErrorPhase::Discovery);
			assert_eq!(error.provider.as_ref().map(ProviderId::as_str), Some("google-cca"));
			assert_eq!(error.route.as_ref().map(RouteId::as_str), Some("google-cca-primary"));
			assert_eq!(error.code.as_ref().map(Str::as_str), Some("cca_discovery_invalid_payload"));
		}
		let mut decoder = CcaDiscoveryDecoder {
			provider: ProviderId::from("google-cca"),
			route:    RouteId::from("google-cca-primary"),
			done:     false,
		};
		decoder
			.push(Frame::Raw(Bytes::from_static(br#"{"models":{}}"#)), &mut |_| {})
			.expect("first page");
		assert_eq!(
			decoder
				.push(Frame::Raw(Bytes::from_static(br#"{"models":{}}"#)), &mut |_| {},)
				.expect_err("second body rejected")
				.kind,
			ErrorKind::Protocol,
		);
	}

	#[test]
	fn antigravity_leaks_are_removed_and_thinking_tags_healed() {
		let mut decoder = CcaDecoder::default();
		let frames: [&[u8]; 5] = [
			br#"{"response":{"candidates":[{"content":{"parts":[{"text":"{\"tho"}]}}]}}"#,
			br#"{"response":{"candidates":[{"content":{"parts":[{"text":"ught\":\"secret\",\"call\":\"read\"}visible"}]}}]}}"#,
			br#"{"response":{"candidates":[{"content":{"parts":[{"text":" before <thi"}]}}]}}"#,
			br#"{"response":{"candidates":[{"content":{"parts":[{"text":"nking>healed"}]}}]}}"#,
			br#"{"response":{"candidates":[{"content":{"parts":[{"text":" secret</thinking> after"}]},"finishReason":"STOP"}]}}"#,
		];
		let mut events = Vec::new();
		for frame in frames {
			events.extend(decoder.push_json(frame).expect("frame decodes"));
		}
		let text = events
			.iter()
			.filter_map(|event| match event {
				GoogleDecodedEvent::Text { text, .. } => Some(text.as_str()),
				_ => None,
			})
			.collect::<String>();
		let thinking = events
			.iter()
			.filter_map(|event| match event {
				GoogleDecodedEvent::Thinking { text, .. } => Some(text.as_str()),
				_ => None,
			})
			.collect::<String>();
		assert_eq!(text, "visible before  after");
		assert_eq!(thinking, "healed secret");
	}

	#[test]
	fn recorded_signature_stream_projects_canonical_tool_completion() {
		let mut decoder = CanonicalCcaDecoder::default();
		let mut events = Vec::new();
		for data in include_str!(
			"../../../../fixtures/llm-oracle/google/recorded/google_cca/stream.signature_tool.sse"
		)
		.lines()
		.filter_map(|line| line.strip_prefix("data: "))
		{
			decoder
				.push(Frame::Raw(Bytes::copy_from_slice(data.as_bytes())), &mut |event| {
					events.push(event)
				})
				.expect("recorded CCA frame decodes");
		}
		decoder
			.finish(&mut |event| events.push(event))
			.expect("recorded CCA stream finishes");

		assert!(events.iter().any(|event| matches!(
			event,
			RawEvent::Chat(crate::event::ChatEvent::ThinkingDelta { text, .. })
				if text.as_str() == "Plan lookup"
		)));
		assert!(events.iter().any(|event| matches!(
			event,
			RawEvent::Chat(crate::event::ChatEvent::ToolCallStarted { id, name, .. })
				if id.as_str() == "call_lookup" && name.as_str() == "lookup"
		)));
		assert!(events.iter().any(|event| matches!(
			event,
			RawEvent::Chat(crate::event::ChatEvent::ToolArgumentsDelta { bytes, .. })
				if bytes.as_ref() == br#"{"q":"x"}"#
		)));
		assert!(events.iter().any(|event| matches!(
			event,
			RawEvent::Chat(crate::event::ChatEvent::Usage(crate::event::UsageUpdate {
				usage: crate::receipt::Usage {
					input_tokens: 10,
					output_tokens: 5,
					cache_read_tokens: 6,
					..
				},
				..
			}))
		)));
		assert!(matches!(
			events.last(),
			Some(RawEvent::Completion(crate::codec::RawCompletion {
				reason: crate::event::FinishReason::ToolCalls,
				blocks: 2,
				..
			}))
		));
	}

	#[test]
	fn malformed_tool_arguments_stay_private_through_cca_envelopes() {
		#[derive(Deserialize)]
		struct MalformedToolArguments {
			name:     Str,
			response: Str,
			expected: Str,
		}

		let fixtures: Vec<MalformedToolArguments> =
			serde_json::from_str(include_str!("fixtures/google_malformed_tool_arguments.json"))
				.expect("malformed tool argument fixtures parse");
		for fixture in fixtures {
			let envelope = format!(r#"{{"response":{}}}"#, fixture.response);
			let mut decoder = CanonicalCcaDecoder::default();
			let mut events = Vec::new();
			decoder
				.push(Frame::Raw(Bytes::from(envelope)), &mut |event| events.push(event))
				.unwrap_or_else(|error| panic!("{} CCA envelope decodes: {error}", fixture.name));
			assert!(
				!events.iter().any(|event| matches!(
					event,
					RawEvent::Chat(crate::event::ChatEvent::ToolArgumentsDelta { .. })
				)),
				"{} leaked malformed arguments into an ordinary CCA delta",
				fixture.name
			);
			assert!(
				events.iter().any(|event| matches!(
					event,
					RawEvent::ToolCallComplete {
						call: crate::codec::UnvalidatedToolCall {
							arguments,
							input_kind: crate::codec::ToolInputKind::Json,
							..
						},
						..
					} if arguments.as_ref() == fixture.expected.as_bytes()
				)),
				"{} did not preserve CCA private repair evidence",
				fixture.name
			);
			assert!(
				!events.iter().any(|event| matches!(
					event,
					RawEvent::Chat(crate::event::ChatEvent::ToolCallReady { .. })
				)),
				"{} authorized a CCA tool before repair and validation",
				fixture.name
			);
		}
	}

	#[test]
	fn antigravity_headers_and_forced_tool_directive_use_explicit_policy() {
		let headers =
			CcaHeaders::antigravity("darwin", "arm64", true, Some("project-REDACTED".into()));
		assert_eq!(headers.user_agent.as_str(), "antigravity/hub/2.1.4 darwin/arm64");
		assert_eq!(headers.anthropic_beta.as_deref(), Some(CLAUDE_THINKING_BETA));
		assert_eq!(headers.quota_project.as_deref(), Some("project-REDACTED"));

		let envelope = wrap_antigravity_request(
			GenerateContentRequest::default(),
			"wire-model".into(),
			"project-REDACTED".into(),
			&AntigravityRequestMetadata {
				trajectory_id:                "trajectory".into(),
				request_id:                   "request".into(),
				session_id:                   None,
				last_execution_id:            None,
				last_step_index:              None,
				model_enum:                   None,
				used_claude:                  false,
				system_identity:              None,
				validated_tool_config:        false,
				append_forced_tool_directive: true,
				max_output_tokens:            None,
			},
		);
		let directive = envelope
			.request
			.generate
			.contents
			.last()
			.and_then(|content| content.parts.last())
			.and_then(|part| part.text.as_deref());
		assert_eq!(directive, Some(ANTIGRAVITY_FORCED_TOOL_DIRECTIVE));
	}

	#[test]
	fn route_safe_gemini_cli_header_uses_selected_wire_model() {
		let codec = GoogleCcaCodec::gemini_cli_for_route(None, "darwin", "arm64");
		let headers = codec.request_headers("gemini-selected");
		let user_agent = headers
			.iter()
			.find(|header| header.name.as_str() == "user-agent")
			.expect("user-agent header");
		assert_eq!(
			user_agent.value.as_str(),
			"GeminiCLI/0.46.0/gemini-selected (darwin; arm64; terminal)",
		);
	}
}
