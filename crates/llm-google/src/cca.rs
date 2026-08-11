//! Google Cloud Code Assist request envelopes and streamed response projection.

use std::iter::FusedIterator;

use bytes::Bytes;
use omp_core::{Str, fmts};
use omp_llm_catalog::{
	compat::{Compat, ToolSchemaFlavor},
	provider::{ProviderEntry, TransportId},
};
use omp_llm_transport::{DecodeState, Frame, Transport, normalize};
use omp_llm_types::{ChatRequest, Error, TurnEvent, Unsupported};
use serde_json::{Map, Value, json};
use smallvec::SmallVec;

use crate::leak_filter::{HealedFragment, PlanningLeakFilter, ThinkingLeakFilter};

/// Path appended to a Cloud Code Assist endpoint for streaming generation.
pub const STREAM_GENERATE_PATH: &str = "/v1internal:streamGenerateContent?alt=sse";

/// Gemini CLI's Cloud Code Assist client metadata header.
pub const GEMINI_CLI_CLIENT_METADATA: &str =
	"ideType=IDE_UNSPECIFIED,platform=PLATFORM_UNSPECIFIED,pluginType=GEMINI";

/// Beta required by Antigravity for interleaved Claude thinking.
pub const CLAUDE_THINKING_BETA: &str = "interleaved-thinking-2025-05-14";

/// Identity instruction emitted by the real Antigravity agent client.
pub const ANTIGRAVITY_SYSTEM_INSTRUCTION: &str =
	"You are Antigravity, a powerful agentic AI coding assistant designed by the Google Deepmind \
	 team working on Advanced Agentic Coding.You are pair programming with a USER to solve their \
	 coding task. The task may require creating a new codebase, modifying or debugging an existing \
	 codebase, or simply answering a question.**Absolute paths only****Proactiveness**";

/// Data-selected Cloud Code Assist client behavior.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CcaFlavor {
	/// Official Gemini CLI request shape and fingerprint.
	#[default]
	GeminiCli,
	/// Antigravity agent envelope, fingerprint, and endpoint policy.
	Antigravity,
}

/// Conversation-stable Antigravity envelope metadata minted by the broker.
///
/// The codec is intentionally pure: the broker/session owner advances
/// trajectory state and supplies this immutable snapshot for one attempt.
/// Retries therefore serialize the exact same request identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AntigravityRequestMetadata {
	session_id:        Str,
	request_id:        Str,
	trajectory_id:     Str,
	step_index:        u64,
	last_execution_id: Option<Str>,
	model_enum:        Option<Str>,
}

impl AntigravityRequestMetadata {
	/// Creates required metadata for one Antigravity trajectory step.
	#[must_use]
	pub const fn new(session_id: Str, request_id: Str, trajectory_id: Str, step_index: u64) -> Self {
		Self {
			session_id,
			request_id,
			trajectory_id,
			step_index,
			last_execution_id: None,
			model_enum: None,
		}
	}

	/// Echoes the preceding successful CCA `responseId`.
	#[must_use]
	pub fn with_last_execution_id(mut self, execution_id: Str) -> Self {
		self.last_execution_id = Some(execution_id);
		self
	}

	/// Adds the captured Antigravity model telemetry enum.
	#[must_use]
	pub fn with_model_enum(mut self, model_enum: Str) -> Self {
		self.model_enum = Some(model_enum);
		self
	}

	/// Returns the signed-decimal conversation session id.
	#[must_use]
	pub fn session_id(&self) -> &str {
		&self.session_id
	}

	/// Returns the structured `agent/...` request id.
	#[must_use]
	pub fn request_id(&self) -> &str {
		&self.request_id
	}
}

/// Header values mandated by the selected CCA client fingerprint.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CcaHeaders {
	/// Exact user-agent string.
	pub user_agent:      Str,
	/// Gemini CLI client metadata; absent for Antigravity.
	pub client_metadata: Option<&'static str>,
	/// Claude interleaved-thinking beta; absent otherwise.
	pub anthropic_beta:  Option<&'static str>,
	/// Billing project forwarded as `X-Goog-User-Project`.
	pub quota_project:   Option<Str>,
}

impl CcaHeaders {
	/// Returns concrete HTTP header pairs for the production egress request.
	#[must_use]
	pub fn entries(&self) -> SmallVec<(&'static str, Str), 4> {
		let mut entries = SmallVec::new();
		entries.push(("User-Agent", self.user_agent.clone()));
		if let Some(metadata) = self.client_metadata {
			entries.push(("Client-Metadata", Str::from(metadata)));
		}
		if let Some(beta) = self.anthropic_beta {
			entries.push(("anthropic-beta", Str::from(beta)));
		}
		if let Some(project) = &self.quota_project {
			entries.push(("X-Goog-User-Project", project.clone()));
		}
		entries
	}
}

/// The Cloud Code Assist codec for one credential project and endpoint attempt.
///
/// Project/account data is resolved by the credential broker before codec
/// construction. This layer performs no credential I/O or direct HTTP dispatch.
#[derive(Clone)]
pub struct CcaCodec {
	project:             Str,
	flavor:              CcaFlavor,
	antigravity:         Option<AntigravityRequestMetadata>,
	identity:            Option<Str>,
	account_id:          Option<Str>,
	organization_id:     Option<Str>,
	quota_project:       Option<Str>,
	served_endpoint:     Option<Str>,
	leak_filter_enabled: bool,
	tool_names:          SmallVec<Str, 8>,
}

impl std::fmt::Debug for CcaCodec {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		formatter
			.debug_struct("CcaCodec")
			.field("project", &"[redacted]")
			.field("flavor", &self.flavor)
			.field("antigravity", &self.antigravity)
			.field("identity", &self.identity.as_ref().map(|_| "[present]"))
			.field("account_id", &self.account_id.as_ref().map(|_| "[present]"))
			.field("organization_id", &self.organization_id.as_ref().map(|_| "[present]"))
			.field("quota_project", &self.quota_project.as_ref().map(|_| "[present]"))
			.field("served_endpoint", &self.served_endpoint)
			.field("leak_filter_enabled", &self.leak_filter_enabled)
			.field("tool_count", &self.tool_names.len())
			.finish()
	}
}

impl CcaCodec {
	/// Creates a Gemini CLI codec using the credential's resolved project id.
	#[must_use]
	pub const fn new(project: Str) -> Self {
		Self {
			project,
			flavor: CcaFlavor::GeminiCli,
			antigravity: None,
			identity: None,
			account_id: None,
			organization_id: None,
			quota_project: None,
			served_endpoint: None,
			leak_filter_enabled: false,
			tool_names: SmallVec::new(),
		}
	}

	/// Creates an Antigravity codec for one immutable trajectory step.
	#[must_use]
	pub const fn antigravity(project: Str, metadata: AntigravityRequestMetadata) -> Self {
		Self {
			project,
			flavor: CcaFlavor::Antigravity,
			antigravity: Some(metadata),
			identity: None,
			account_id: None,
			organization_id: None,
			quota_project: None,
			served_endpoint: None,
			leak_filter_enabled: false,
			tool_names: SmallVec::new(),
		}
	}

	/// Records the broker-selected non-secret account identity on the outcome.
	#[must_use]
	pub fn with_identity(mut self, identity: Str) -> Self {
		self.identity = Some(identity);
		self
	}

	/// Records the broker-selected non-secret account id on the outcome.
	#[must_use]
	pub fn with_account_id(mut self, account_id: Str) -> Self {
		self.account_id = Some(account_id);
		self
	}

	/// Records the broker-selected non-secret organization id on the outcome.
	#[must_use]
	pub fn with_organization_id(mut self, organization_id: Str) -> Self {
		self.organization_id = Some(organization_id);
		self
	}

	/// Overlays the validated credential lease's resolved CCA project id.
	///
	/// This preserves Antigravity trajectory and endpoint configuration while
	/// replacing any catalog placeholder immediately before request encoding.
	/// The same validated project becomes the default Google quota project; a
	/// later [`Self::with_quota_project`] call may select a distinct billing id.
	#[must_use]
	pub fn with_project_id(mut self, project_id: Str) -> Self {
		self.project = project_id.clone();
		self.quota_project = Some(project_id);
		self
	}

	/// Selects the project billed by Google and emitted in egress headers.
	#[must_use]
	pub fn with_quota_project(mut self, project: Str) -> Self {
		self.quota_project = Some(project);
		self
	}

	/// Associates this attempt with the provider endpoint that will serve it.
	#[must_use]
	pub fn with_served_endpoint(mut self, endpoint: Str) -> Self {
		self.served_endpoint = Some(endpoint);
		self
	}

	/// Enables Flash planning-leak suppression with the active tool names.
	#[must_use]
	pub fn with_planning_leak_filter(mut self, tool_names: impl IntoIterator<Item = Str>) -> Self {
		self.leak_filter_enabled = true;
		self.tool_names = tool_names.into_iter().collect();
		self
	}

	/// Returns the credential project carried by every encoded envelope.
	#[must_use]
	pub fn project(&self) -> &str {
		self.project.as_str()
	}

	/// Builds the exact client fingerprint headers for a wire model.
	#[must_use]
	pub fn request_headers(&self, model: &str, reasoning: bool) -> CcaHeaders {
		let claude_thinking = self.flavor == CcaFlavor::Antigravity
			&& model.to_ascii_lowercase().contains("claude")
			&& reasoning;
		CcaHeaders {
			user_agent:      match self.flavor {
				CcaFlavor::GeminiCli => gemini_cli_user_agent(model),
				CcaFlavor::Antigravity => antigravity_user_agent(),
			},
			client_metadata: (self.flavor == CcaFlavor::GeminiCli)
				.then_some(GEMINI_CLI_CLIENT_METADATA),
			anthropic_beta:  claude_thinking.then_some(CLAUDE_THINKING_BETA),
			quota_project:   self.quota_project.clone(),
		}
	}
}

impl Transport for CcaCodec {
	fn id(&self) -> TransportId {
		TransportId::GoogleCca
	}

	fn encode(
		&self,
		req: &ChatRequest,
		compat: &Compat,
	) -> Result<(Bytes, Vec<Unsupported>), Error> {
		let (normalized_tools, mut unsupported) =
			normalize::normalize_tools(ToolSchemaFlavor::Cca, &req.tools);
		let (request, mut projection_report) = crate::encode_request_with_tools(
			req,
			&normalized_tools,
			compat,
			crate::GoogleVariant::CCA,
		)?;
		unsupported.append(&mut projection_report);
		let envelope = match (&self.flavor, &self.antigravity) {
			(CcaFlavor::Antigravity, Some(metadata)) => {
				wrap_antigravity_request(request, req.model.as_str(), self.project.as_str(), metadata)
			},
			_ => wrap_request(request, req.model.as_str(), self.project.as_str()),
		};
		let body = serde_json::to_vec(&envelope)
			.map_err(|error| Error::Provider(fmts!("cannot encode CCA request: {error}")))?;
		Ok((Bytes::from(body), unsupported))
	}

	fn decode(
		&self,
		frame: Frame<'_>,
		state: &mut DecodeState,
	) -> Result<SmallVec<TurnEvent, 2>, Error> {
		let state = state.get_or_insert_with(|| CcaDecodeState {
			google:            DecodeState::default(),
			leak_filter:       self
				.leak_filter_enabled
				.then(|| PlanningLeakFilter::new(self.tool_names.iter().cloned())),
			thinking_filter:   ThinkingLeakFilter::default(),
			pending_signature: None,
		});
		let mut events = match frame {
			Frame::Done => finish_cca_stream(state)?,
			Frame::Data(data) | Frame::Event { data, .. } if data == b"[DONE]" => {
				finish_cca_stream(state)?
			},
			Frame::Data(data) | Frame::Event { data, .. } => {
				let value: Value = serde_json::from_slice(data)
					.map_err(|error| Error::Provider(fmts!("invalid CCA response JSON: {error}")))?;
				let mut response = response_or_error(value)?;
				filter_visible_parts(
					&mut response,
					state.leak_filter.as_mut(),
					&mut state.pending_signature,
				)?;
				heal_thinking_parts(
					&mut response,
					&mut state.thinking_filter,
					&mut state.pending_signature,
				);
				crate::decode_value(response, &mut state.google)?
			},
			_ => SmallVec::new(),
		};
		record_cca_properties(&mut events, self);
		Ok(events)
	}
}

#[derive(Default)]
struct CcaDecodeState {
	google:            DecodeState,
	leak_filter:       Option<PlanningLeakFilter>,
	thinking_filter:   ThinkingLeakFilter,
	pending_signature: Option<Str>,
}

/// Wraps a public `GenAI` request body in the Gemini CLI CCA envelope.
#[must_use]
pub fn wrap_request(request: Value, model: &str, project: &str) -> Value {
	json!({ "model": model, "project": project, "request": request })
}

/// Shapes a request as the Antigravity agent client does.
#[must_use]
pub fn wrap_antigravity_request(
	mut request: Value,
	model: &str,
	project: &str,
	metadata: &AntigravityRequestMetadata,
) -> Value {
	let request_object = request
		.as_object_mut()
		.expect("Google request is an object");
	shape_antigravity_request(request_object, model, metadata);
	json!({
		"project": project,
		"model": model,
		"request": request,
		"requestType": "agent",
		"userAgent": "antigravity",
		"requestId": metadata.request_id.as_str(),
	})
}

/// Removes the Cloud Code Assist response envelope around one `GenAI` chunk.
pub fn unwrap_response(envelope: Value) -> Result<Value, Error> {
	let mut object = envelope
		.as_object()
		.cloned()
		.ok_or_else(|| Error::Provider("CCA stream event is not an object".into()))?;
	if let Some(error) = object.get("error") {
		return Err(Error::Provider(fmts!("CCA in-band error: {error}")));
	}
	object
		.remove("response")
		.ok_or_else(|| Error::Provider("CCA stream event has no response".into()))
}
/// Converts a canonical opaque thought signature to CCA's JSON string form.
pub fn thought_signature_to_wire(signature: &Bytes) -> Result<Str, Error> {
	std::str::from_utf8(signature)
		.map(Str::from)
		.map_err(|error| Error::Provider(fmts!("CCA thought signature is not UTF-8: {error}")))
}

/// Converts a CCA thought-signature string to canonical opaque bytes.
#[must_use]
pub fn thought_signature_from_wire(signature: &str) -> Bytes {
	Bytes::copy_from_slice(signature.as_bytes())
}
fn shape_antigravity_request(
	request: &mut Map<String, Value>,
	model: &str,
	metadata: &AntigravityRequestMetadata,
) {
	let is_claude = model.to_ascii_lowercase().contains("claude");
	let needs_identity = is_claude || model.to_ascii_lowercase().contains("gemini-3");
	let system = request
		.entry("systemInstruction")
		.or_insert_with(|| json!({ "parts": [] }));
	if let Some(system) = system.as_object_mut() {
		system.insert("role".into(), Value::String("user".into()));
		let parts = system
			.entry("parts")
			.or_insert_with(|| Value::Array(Vec::new()));
		if needs_identity
			&& let Some(parts) = parts.as_array_mut()
			&& !parts.first().is_some_and(|part| {
				part.get("text").and_then(Value::as_str) == Some(ANTIGRAVITY_SYSTEM_INSTRUCTION)
			}) {
			parts.insert(0, json!({ "text": ANTIGRAVITY_SYSTEM_INSTRUCTION }));
		}
	}

	let has_tools = request
		.get("tools")
		.is_some_and(|tools| tools.as_array().is_some_and(|tools| !tools.is_empty()));
	if is_claude || has_tools && !request.contains_key("toolConfig") {
		request
			.insert("toolConfig".into(), json!({ "functionCallingConfig": { "mode": "VALIDATED" } }));
	}
	if let Some(max_tokens) = antigravity_max_output_tokens(model) {
		let generation = request
			.entry("generationConfig")
			.or_insert_with(|| Value::Object(Map::new()));
		if let Some(generation) = generation.as_object_mut() {
			generation.insert("maxOutputTokens".into(), Value::from(max_tokens));
		}
	}

	let mut labels = Map::new();
	if let Some(execution_id) = &metadata.last_execution_id {
		labels.insert("last_execution_id".into(), Value::String(execution_id.to_string()));
	}
	labels.insert(
		"last_step_index".into(),
		Value::String(metadata.step_index.saturating_sub(1).to_string()),
	);
	if let Some(model_enum) = metadata
		.model_enum
		.as_deref()
		.or_else(|| antigravity_model_enum(model))
	{
		labels.insert("model_enum".into(), Value::String(model_enum.into()));
	}
	labels.insert("trajectory_id".into(), Value::String(metadata.trajectory_id.to_string()));
	labels.insert("used_claude".into(), Value::String(is_claude.to_string()));
	labels.insert("used_claude_conservative".into(), Value::String(is_claude.to_string()));
	request.insert("labels".into(), Value::Object(labels));
	request.insert("sessionId".into(), Value::String(metadata.session_id.to_string()));
}

fn antigravity_max_output_tokens(model: &str) -> Option<u64> {
	match model {
		"gemini-3.5-flash-extra-low" | "gemini-3.5-flash-low" | "gemini-3-flash-agent" => {
			Some(65_536)
		},
		"gemini-3.1-pro-low" | "gemini-pro-agent" => Some(65_535),
		"claude-sonnet-4-6" | "claude-opus-4-6-thinking" => Some(64_000),
		_ => None,
	}
}

fn antigravity_model_enum(model: &str) -> Option<&'static str> {
	match model {
		"gemini-3.5-flash-extra-low" => Some("MODEL_PLACEHOLDER_M187"),
		"gemini-3.5-flash-low" => Some("MODEL_PLACEHOLDER_M20"),
		"gemini-3-flash-agent" => Some("MODEL_PLACEHOLDER_M132"),
		"gemini-3.1-pro-low" => Some("MODEL_PLACEHOLDER_M36"),
		"gemini-pro-agent" => Some("MODEL_PLACEHOLDER_M16"),
		_ => None,
	}
}

fn response_or_error(envelope: Value) -> Result<Value, Error> {
	let object = envelope
		.as_object()
		.ok_or_else(|| Error::Provider("CCA stream event is not an object".into()))?;
	if let Some(error) = object.get("error") {
		return Ok(json!({ "error": error }));
	}
	object
		.get("response")
		.cloned()
		.ok_or_else(|| Error::Provider("CCA stream event has no response".into()))
}

fn filter_visible_parts(
	response: &mut Value,
	filter: Option<&mut PlanningLeakFilter>,
	pending_signature: &mut Option<Str>,
) -> Result<(), Error> {
	let Some(filter) = filter else {
		return Ok(());
	};
	let Some(candidates) = response.get_mut("candidates").and_then(Value::as_array_mut) else {
		return Ok(());
	};
	for candidate in candidates {
		let finished = candidate
			.get("finishReason")
			.and_then(Value::as_str)
			.is_some();
		let Some(parts) = candidate_parts(candidate, finished) else {
			continue;
		};
		for part in parts.iter_mut() {
			if part.get("functionCall").is_some() {
				filter.discard_probe();
				*pending_signature = None;
				continue;
			}
			if part.get("thought").and_then(Value::as_bool) == Some(true) {
				continue;
			}
			let Some(text) = part.get("text").and_then(Value::as_str) else {
				continue;
			};
			if text.is_empty() {
				continue;
			}
			if let Some(signature) = part.get("thoughtSignature").and_then(Value::as_str) {
				*pending_signature = Some(Str::from(signature));
			}
			let chunks = filter.feed(text.as_bytes()).map_err(|error| {
				Error::Provider(fmts!("invalid UTF-8 in CCA visible text: {error}"))
			})?;
			let visible = join_chunks(chunks);
			if let Some(object) = part.as_object_mut() {
				if visible.is_empty() {
					object.remove("text");
					object.remove("thoughtSignature");
				} else {
					object.insert(
						"text".into(),
						Value::String(String::from_utf8(visible.to_vec()).expect("filter returns UTF-8")),
					);
					if let Some(signature) = pending_signature.take() {
						object.insert("thoughtSignature".into(), Value::String(signature.to_string()));
					}
				}
			}
		}
		if finished {
			let chunks = filter.finish().map_err(|error| {
				Error::Provider(fmts!("truncated UTF-8 in CCA visible text: {error}"))
			})?;
			let visible = join_chunks(chunks);
			if visible.is_empty() {
   				*pending_signature = None;
   			} else {
   				let mut part = Map::new();
   				part.insert(
   					"text".into(),
   					Value::String(String::from_utf8(visible.to_vec()).expect("filter returns UTF-8")),
   				);
   				if let Some(signature) = pending_signature.take() {
   					part.insert("thoughtSignature".into(), Value::String(signature.to_string()));
   				}
   				parts.push(Value::Object(part));
   			}
		}
	}
	Ok(())
}
fn candidate_parts(candidate: &mut Value, create: bool) -> Option<&mut Vec<Value>> {
	let candidate = candidate.as_object_mut()?;
	if create {
		let content = candidate
			.entry("content")
			.or_insert_with(|| Value::Object(Map::new()))
			.as_object_mut()?;
		return content
			.entry("parts")
			.or_insert_with(|| Value::Array(Vec::new()))
			.as_array_mut();
	}
	candidate
		.get_mut("content")?
		.get_mut("parts")?
		.as_array_mut()
}

fn heal_thinking_parts(
	response: &mut Value,
	filter: &mut ThinkingLeakFilter,
	pending_signature: &mut Option<Str>,
) {
	let Some(candidates) = response.get_mut("candidates").and_then(Value::as_array_mut) else {
		return;
	};
	for candidate in candidates {
		let finished = candidate
			.get("finishReason")
			.and_then(Value::as_str)
			.is_some();
		let Some(parts) = candidate_parts(candidate, finished) else {
			continue;
		};
		let original = std::mem::take(parts);
		for part in original {
			if part.get("functionCall").is_some()
				|| part.get("thought").and_then(Value::as_bool) == Some(true)
			{
				parts.push(part);
				continue;
			}
			let Some(text) = part.get("text").and_then(Value::as_str) else {
				parts.push(part);
				continue;
			};
			if text.is_empty() {
				parts.push(part);
				continue;
			}
			if let Some(signature) = part.get("thoughtSignature").and_then(Value::as_str) {
				*pending_signature = Some(Str::from(signature));
			}
			let mut base = part.as_object().cloned().unwrap_or_default();
			base.remove("text");
			base.remove("thought");
			base.remove("thoughtSignature");
			append_healed_parts(parts, base, filter.feed(text), pending_signature);
		}
		if finished {
			append_healed_parts(parts, Map::new(), filter.finish(), pending_signature);
		}
	}
}

fn append_healed_parts(
	parts: &mut Vec<Value>,
	base: Map<String, Value>,
	fragments: SmallVec<HealedFragment, 2>,
	pending_signature: &mut Option<Str>,
) {
	let mut fragments = fragments.into_iter().peekable();
	while let Some(fragment) = fragments.next() {
		let mut part = base.clone();
		let (thinking, bytes) = match fragment {
			HealedFragment::Text(bytes) => (false, bytes),
			HealedFragment::Thinking(bytes) => (true, bytes),
		};
		part.insert(
			"text".into(),
			Value::String(String::from_utf8(bytes.to_vec()).expect("healer returns UTF-8")),
		);
		if thinking {
			part.insert("thought".into(), Value::Bool(true));
		}
		if fragments.peek().is_none()
			&& let Some(signature) = pending_signature.take()
		{
			part.insert("thoughtSignature".into(), Value::String(signature.to_string()));
		}
		parts.push(Value::Object(part));
	}
}

fn finish_cca_stream(state: &mut CcaDecodeState) -> Result<SmallVec<TurnEvent, 2>, Error> {
	let mut events = SmallVec::new();
	let mut parts = Vec::new();
	if let Some(filter) = &mut state.leak_filter {
		let visible = join_chunks(filter.finish().map_err(|error| {
			Error::Provider(fmts!("truncated UTF-8 in CCA visible text: {error}"))
		})?);
		if !visible.is_empty() {
			let text = std::str::from_utf8(&visible).expect("planning filter returns UTF-8");
			append_healed_parts(
				&mut parts,
				Map::new(),
				state.thinking_filter.feed(text),
				&mut state.pending_signature,
			);
		}
	}
	append_healed_parts(
		&mut parts,
		Map::new(),
		state.thinking_filter.finish(),
		&mut state.pending_signature,
	);
	if !parts.is_empty() {
		events.extend(crate::decode_value(
			json!({ "candidates": [{ "content": { "parts": parts } }] }),
			&mut state.google,
		)?);
	}
	events.extend(crate::finish_stream(&mut state.google));
	Ok(events)
}

fn join_chunks(chunks: SmallVec<Bytes, 2>) -> Bytes {
	match chunks.len() {
		0 => Bytes::new(),
		1 => chunks.into_iter().next().expect("one chunk"),
		_ => {
			let total = chunks.iter().map(Bytes::len).sum();
			let mut joined = Vec::with_capacity(total);
			for chunk in chunks {
				joined.extend_from_slice(&chunk);
			}
			Bytes::from(joined)
		},
	}
}

fn gemini_cli_user_agent(model: &str) -> Str {
	let platform = match std::env::consts::OS {
		"macos" => "darwin",
		"windows" => "win32",
		other => other,
	};
	let arch = match std::env::consts::ARCH {
		"x86_64" => "x64",
		"x86" => "ia32",
		"aarch64" => "arm64",
		other => other,
	};
	fmts!("GeminiCLI/0.46.0/{model} ({platform}; {arch}; terminal)")
}

fn antigravity_user_agent() -> Str {
	let os = match std::env::consts::OS {
		"macos" => "darwin",
		"windows" => "windows",
		other => other,
	};
	let arch = match std::env::consts::ARCH {
		"x86_64" => "amd64",
		"x86" => "386",
		"aarch64" => "arm64",
		other => other,
	};
	fmts!("antigravity/hub/2.1.4 {os}/{arch}")
}

/// Antigravity host selection policy applied to catalog endpoint data.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CcaEndpointMode {
	/// Prefer the last successful catalog host, then try the remaining hosts.
	#[default]
	Auto,
	/// Use only the catalog's primary production host.
	Production,
	/// Use only the first catalog fallback host.
	Sandbox,
}

/// Ordered Cloud Code Assist endpoints selected from catalog provider rows.
///
/// Rows are retained in caller order. This keeps production-before-sandbox
/// policy in catalog data rather than embedding host names in retry logic.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CcaEndpointPlan {
	endpoints: SmallVec<Str, 2>,
}

impl CcaEndpointPlan {
	/// Builds an ordered plan from Google CCA provider entries.
	#[must_use]
	pub fn from_provider_entries<'a>(entries: impl IntoIterator<Item = &'a ProviderEntry>) -> Self {
		let mut endpoints = SmallVec::new();
		for entry in entries {
			if entry.transport != TransportId::GoogleCca {
				continue;
			}
			endpoints.push(entry.base_url.clone());
			endpoints.extend(entry.fallback_base_urls.iter().cloned());
		}
		Self { endpoints }
	}

	/// Applies explicit Antigravity production/sandbox selection or auto
	/// affinity.
	#[must_use]
	pub fn with_mode(mut self, mode: CcaEndpointMode, last_good: Option<&str>) -> Self {
		match mode {
			CcaEndpointMode::Auto => {
				if let Some(index) = last_good.and_then(|last_good| {
					self
						.endpoints
						.iter()
						.position(|endpoint| endpoint == last_good)
				}) && index != 0
				{
					let endpoint = self.endpoints.remove(index);
					self.endpoints.insert(0, endpoint);
				}
			},
			CcaEndpointMode::Production => self.endpoints.truncate(1),
			CcaEndpointMode::Sandbox => {
				if self.endpoints.len() > 1 {
					let endpoint = self.endpoints.remove(1);
					self.endpoints.clear();
					self.endpoints.push(endpoint);
				} else {
					self.endpoints.clear();
				}
			},
		}
		self
	}

	/// Returns endpoints in their catalog-defined attempt order.
	pub fn endpoints(
		&self,
	) -> impl Clone + DoubleEndedIterator<Item = &str> + ExactSizeIterator + FusedIterator + '_ {
		self.endpoints.iter().map(Str::as_str)
	}

	/// Returns the next endpoint when the completed attempt is eligible for host
	/// fallback.
	#[must_use]
	pub fn next_after_failure(
		&self,
		attempted: &str,
		failure: CcaAttemptFailure,
		response_started: bool,
	) -> Option<&str> {
		if !should_fallback(failure, response_started) {
			return None;
		}
		let index = self
			.endpoints
			.iter()
			.position(|endpoint| endpoint == attempted)?;
		self.endpoints.get(index + 1).map(Str::as_str)
	}

	/// Builds the streaming URL for an endpoint without changing its host.
	#[must_use]
	pub fn stream_url(endpoint: &str) -> Str {
		fmts!("{}{}", endpoint.trim_end_matches('/'), STREAM_GENERATE_PATH)
	}
}

/// Failure evidence used to decide whether another CCA endpoint may be tried.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CcaAttemptFailure {
	/// A non-successful HTTP response status.
	HttpStatus(u16),
	/// A connection, DNS, TLS, or other pre-response transport failure.
	Transport,
	/// A response stream completed without meaningful content.
	EmptyBody,
	/// A response began but ended without a finish reason.
	IncompleteStream,
	/// The provider emitted an explicit output error.
	Output,
}

/// Returns whether the next catalog endpoint should be attempted.
///
/// CCA fallback is limited to failures before response content begins. HTTP
/// 408, 429, and 5xx statuses are transient, as are retryable transport and
/// empty-body failures. Explicit model output failures never change hosts.
#[must_use]
pub const fn should_fallback(failure: CcaAttemptFailure, response_started: bool) -> bool {
	if response_started {
		return false;
	}
	match failure {
		CcaAttemptFailure::HttpStatus(status) => status == 408 || status == 429 || status >= 500,
		CcaAttemptFailure::Transport | CcaAttemptFailure::EmptyBody => true,
		CcaAttemptFailure::IncompleteStream | CcaAttemptFailure::Output => false,
	}
}

fn record_cca_properties(events: &mut SmallVec<TurnEvent, 2>, codec: &CcaCodec) {
	for event in events {
		if let TurnEvent::Outcome(outcome) = event {
			if let Some(endpoint) = &codec.served_endpoint {
				outcome.props.insert_ns(
					"google-cca",
					"served_endpoint",
					Value::String(endpoint.to_string()),
				);
			}
			if let Some(identity) = &codec.identity {
				outcome
					.props
					.insert_ns("google-cca", "identity", Value::String(identity.to_string()));
			}
			if let Some(account_id) = &codec.account_id {
				outcome.props.insert_ns(
					"google-cca",
					"account_id",
					Value::String(account_id.to_string()),
				);
			}
			if let Some(organization_id) = &codec.organization_id {
				outcome.props.insert_ns(
					"google-cca",
					"organization_id",
					Value::String(organization_id.to_string()),
				);
			}
			if let Some(project) = &codec.quota_project {
				outcome.props.insert_ns(
					"google-cca",
					"quota_project",
					Value::String(project.to_string()),
				);
			}
		}
	}
}

#[cfg(test)]
fn record_served_endpoint(events: &mut SmallVec<TurnEvent, 2>, endpoint: &str) {
	for event in events {
		if let TurnEvent::Outcome(outcome) = event {
			outcome
				.props
				.insert_ns("google-cca", "served_endpoint", Value::String(endpoint.into()));
		}
	}
}

#[cfg(test)]
mod tests {
	use std::{collections::BTreeMap, sync::Arc};

	use bytes::Bytes;
	use omp_core::Str;
	use omp_llm_catalog::provider::{AuthSpec, Facet};
	use omp_llm_types::{
		ChatOutcome, Effort, Fallback, Feature, Reasoning, ResolvedModelPolicy, ResolvedThinkingMode,
		ResolvedThinkingPolicy, StopReason, Thread, ToolDef,
	};
	use serde_json::json;
	use smallvec::smallvec;

	use super::*;

	fn entry(id: &str, endpoint: &str) -> ProviderEntry {
		ProviderEntry::builder()
			.id(Str::from(id))
			.transport(TransportId::GoogleCca)
			.base_url(Str::from(endpoint))
			.fallback_base_urls(SmallVec::new())
			.auth(AuthSpec::None)
			.facets([Facet::Chat].into())
			.headers(Default::default())
			.compat(Compat::default())
			.build()
	}

	#[test]
	fn wraps_request_and_unwraps_response_without_changing_genai_body() {
		let inner = json!({"contents": [{"role": "user", "parts": [{"text": "hello"}]}]});
		let envelope = wrap_request(inner.clone(), "gemini-3.5-flash", "project-a");
		assert_eq!(envelope["model"], "gemini-3.5-flash");
		assert_eq!(envelope["project"], "project-a");
		assert_eq!(envelope["request"], inner);
		assert_eq!(unwrap_response(json!({"response": inner.clone()})).unwrap(), inner);
	}
	fn antigravity_metadata() -> AntigravityRequestMetadata {
		AntigravityRequestMetadata::new(
			Str::from("-8392019482710394817"),
			Str::from(
				"agent/11111111-1111-1111-1111-111111111111/1700000000000/\
				 22222222-2222-2222-2222-222222222222/2",
			),
			Str::from("22222222-2222-2222-2222-222222222222"),
			2,
		)
		.with_last_execution_id(Str::from("execution-before"))
		.with_model_enum(Str::from("MODEL_PLACEHOLDER_M20"))
	}

	#[test]
	fn antigravity_envelope_matches_recorded_shape_and_headers() {
		let request = json!({
			"contents": [{"role": "user", "parts": [{"text": "inspect the repository"}]}],
			"systemInstruction": {"parts": [{"text": "Use the repository tools."}]},
			"tools": [{"functionDeclarations": [{"name": "read", "parameters": {"type": "object"}}]}]
		});
		let envelope = wrap_antigravity_request(
			request,
			"gemini-3.5-flash-low",
			"project-a",
			&antigravity_metadata(),
		);
		let expected: Value = serde_json::from_str(include_str!(
			"../tests/fixtures/google_cca/request.antigravity.json"
		))
		.unwrap();
		assert_eq!(envelope, expected);

		let leased = CcaCodec::antigravity(Str::from("catalog-placeholder"), antigravity_metadata())
			.with_project_id(Str::from("project-a"));
		assert!(
			leased
				.request_headers("gemini-3.5-flash", false)
				.entries()
				.contains(&("X-Goog-User-Project", Str::from("project-a")))
		);
		let codec = leased.with_quota_project(Str::from("billing-project"));
		assert_eq!(codec.project(), "project-a");
		let headers = codec.request_headers("claude-sonnet-4-6", true);
		assert!(headers.user_agent.starts_with("antigravity/hub/2.1.4 "));
		assert_eq!(headers.client_metadata, None);
		assert_eq!(headers.anthropic_beta, Some(CLAUDE_THINKING_BETA));
		assert!(
			headers
				.entries()
				.contains(&("X-Goog-User-Project", Str::from("billing-project")))
		);

		let cli = CcaCodec::new(Str::from("project-a")).request_headers("gemini-3.5-flash", false);
		assert!(
			cli.user_agent
				.starts_with("GeminiCLI/0.46.0/gemini-3.5-flash (")
		);
		assert_eq!(cli.client_metadata, Some(GEMINI_CLI_CLIENT_METADATA));
		assert_eq!(cli.anthropic_beta, None);
	}

	#[test]
	fn split_planning_leak_is_suppressed_while_content_signatures_usage_and_route_survive() {
		let fixture = include_str!("../tests/fixtures/google_cca/stream.antigravity_leak.sse");
		let codec = CcaCodec::antigravity(Str::from("project-a"), antigravity_metadata())
			.with_planning_leak_filter([Str::from("read")])
			.with_identity(Str::from("developer@example.com"))
			.with_account_id(Str::from("account-123"))
			.with_organization_id(Str::from("organization-456"))
			.with_quota_project(Str::from("billing-project"))
			.with_served_endpoint(Str::from("https://daily-cloudcode-pa.googleapis.com"));
		let debug = format!("{codec:?}");
		assert!(!debug.contains("project-a"));
		assert!(!debug.contains("developer@example.com"));
		assert!(!debug.contains("account-123"));
		assert!(!debug.contains("organization-456"));
		let mut state = DecodeState::default();
		let mut events = Vec::new();
		let signature = Bytes::from_static(b"opaque-base64-like-signature");
		assert_eq!(
			thought_signature_from_wire(thought_signature_to_wire(&signature).unwrap().as_str()),
			signature
		);
		for line in fixture.lines().filter(|line| !line.is_empty()) {
			let data = line.strip_prefix("data: ").expect("fixture SSE data line");
			events.extend(
				codec
					.decode(Frame::Event { name: None, data: data.as_bytes() }, &mut state)
					.unwrap(),
			);
		}
		let kinds = events
			.iter()
			.filter_map(|event| match event {
				TurnEvent::PartStart { index, kind, .. } => Some((*index, *kind)),
				_ => None,
			})
			.collect::<std::collections::BTreeMap<_, _>>();
		let channel = |wanted| {
			events
				.iter()
				.filter_map(|event| match event {
					TurnEvent::PartDelta { index, chunk } if kinds.get(index) == Some(&wanted) => {
						Some(String::from_utf8_lossy(chunk))
					},
					_ => None,
				})
				.collect::<String>()
		};
		let visible = channel(omp_llm_types::StreamPartKind::Text);
		let thinking = channel(omp_llm_types::StreamPartKind::Thinking);
		assert!(!visible.contains("internal plan"));
		assert!(!visible.contains("\"paths\""));
		assert!(!visible.contains("healed secret"));
		assert!(!visible.contains("<thinking>"));
		assert!(visible.contains("正文 ✓"));
		assert!(visible.contains(" before  after"));
		assert!(thinking.contains("healed secret"));
		assert!(thinking.contains("legitimate reasoning"));
		let signatures = events
			.iter()
			.filter_map(|event| match event {
				TurnEvent::PartEnd { signature, .. } if !signature.is_empty() => {
					Some(signature.as_ref())
				},
				_ => None,
			})
			.collect::<Vec<_>>();
		assert!(signatures.contains(&b"visible-signature".as_slice()));
		assert!(signatures.contains(&b"thinking-signature".as_slice()));
		assert_eq!(
			events
				.iter()
				.filter(|event| matches!(event, TurnEvent::Outcome(_)))
				.count(),
			1
		);
		let outcome = events
			.iter()
			.find_map(|event| match event {
				TurnEvent::Outcome(outcome) => Some(outcome),
				_ => None,
			})
			.unwrap();
		assert_eq!(outcome.props.get_ns("google", "response_id"), Some(&json!("execution-after")));
		assert_eq!(
			outcome.props.get_ns("google-cca", "served_endpoint"),
			Some(&json!("https://daily-cloudcode-pa.googleapis.com"))
		);
		assert_eq!(
			outcome.props.get_ns("google-cca", "identity"),
			Some(&json!("developer@example.com"))
		);
		assert_eq!(outcome.props.get_ns("google-cca", "account_id"), Some(&json!("account-123")));
		assert_eq!(
			outcome.props.get_ns("google-cca", "organization_id"),
			Some(&json!("organization-456"))
		);
		assert_eq!(
			outcome.props.get_ns("google-cca", "quota_project"),
			Some(&json!("billing-project"))
		);
		let usage = outcome.usage.as_ref().expect("fixture supplies usage");
		assert_eq!(usage.input_tokens, 13);
		assert_eq!(usage.output_tokens, 7);
		assert_eq!(usage.cache_read_tokens, 3);
		assert_eq!(usage.detail.get_ns("google", "thoughts_tokens"), Some(&json!(2)));
		assert!(codec.decode(Frame::Done, &mut state).unwrap().is_empty());
	}

	#[test]
	fn in_band_error_is_the_only_terminal_and_later_deltas_are_ignored() {
		let fixture = include_str!("../tests/fixtures/google_cca/stream.error.sse");
		let codec = CcaCodec::new(Str::from("project-a"));
		let mut state = DecodeState::default();
		let mut events = Vec::new();
		for line in fixture.lines().filter(|line| !line.is_empty()) {
			let data = line.strip_prefix("data: ").unwrap();
			events.extend(
				codec
					.decode(Frame::Event { name: None, data: data.as_bytes() }, &mut state)
					.unwrap(),
			);
		}
		assert_eq!(
			events
				.iter()
				.filter(|event| matches!(event, TurnEvent::Outcome(_) | TurnEvent::Error(_)))
				.count(),
			1
		);
		assert!(events.iter().any(|event| matches!(
			event,
			TurnEvent::Error(error)
				if error.detail.contains("Individual quota reached")
					&& error.kind == omp_llm_types::TurnErrorKind::RateLimited
					&& error.retry_after_ms > 0
		)));
		assert!(
			!events
				.iter()
				.any(|event| matches!(event, TurnEvent::PartDelta { .. }))
		);
		assert!(codec.decode(Frame::Done, &mut state).unwrap().is_empty());
	}

	#[test]
	fn cancelled_body_end_emits_one_terminal_and_accepts_no_later_delta() {
		let codec = CcaCodec::new(Str::from("project-a"));
		let mut state = DecodeState::default();
		let first = br#"{"response":{"candidates":[{"content":{"parts":[{"text":"partial"}]}}]}}"#;
		let mut events = codec
			.decode(Frame::Data(first), &mut state)
			.unwrap()
			.into_vec();
		events.extend(codec.decode(Frame::Done, &mut state).unwrap());
		assert_eq!(
			events
				.iter()
				.filter(|event| matches!(event, TurnEvent::Outcome(_) | TurnEvent::Error(_)))
				.count(),
			1
		);
		let later = br#"{"response":{"candidates":[{"content":{"parts":[{"text":"late"}]},"finishReason":"STOP"}]}}"#;
		assert!(
			codec
				.decode(Frame::Data(later), &mut state)
				.unwrap()
				.is_empty()
		);
	}

	#[test]
	fn unwraps_recorded_cca_sse_chunks_before_google_projection() {
		let fixture = include_str!("../tests/fixtures/google_cca/stream.signature_tool.sse");
		let codec = CcaCodec::new(Str::from("project-a"));
		let mut state = DecodeState::default();
		let mut events = Vec::new();
		for line in fixture.lines().filter(|line| !line.is_empty()) {
			let data = line.strip_prefix("data: ").expect("fixture SSE data line");
			events.extend(
				codec
					.decode(Frame::Event { name: None, data: data.as_bytes() }, &mut state)
					.expect("fixture chunk decodes"),
			);
		}
		assert_eq!(
			events
				.iter()
				.filter(|event| matches!(event, TurnEvent::Outcome(_)))
				.count(),
			1,
			"fixture has exactly one terminal outcome"
		);
	}

	#[test]
	fn preserves_catalog_endpoint_order_and_falls_back_only_for_transient_failure() {
		let mut production = entry("production", "https://production.example");
		production
			.fallback_base_urls
			.push(Str::from("https://sandbox.example"));
		let plan = CcaEndpointPlan::from_provider_entries([&production]);
		assert_eq!(plan.endpoints().collect::<Vec<_>>(), [
			"https://production.example",
			"https://sandbox.example"
		]);
		assert_eq!(
			plan
				.clone()
				.with_mode(CcaEndpointMode::Sandbox, None)
				.endpoints()
				.collect::<Vec<_>>(),
			["https://sandbox.example"]
		);
		assert_eq!(
			plan
				.clone()
				.with_mode(CcaEndpointMode::Production, None)
				.endpoints()
				.collect::<Vec<_>>(),
			["https://production.example"]
		);
		assert_eq!(
			plan
				
				.with_mode(CcaEndpointMode::Auto, Some("https://sandbox.example"))
				.endpoints()
				.collect::<Vec<_>>(),
			["https://sandbox.example", "https://production.example"]
		);
		assert!(should_fallback(CcaAttemptFailure::HttpStatus(503), false));
		assert!(!should_fallback(CcaAttemptFailure::HttpStatus(400), false));
		assert!(!should_fallback(CcaAttemptFailure::Transport, true));
	}

	#[test]
	fn successful_fallback_records_the_serving_endpoint_on_the_outcome() {
		let mut production = entry("production", "https://production.example");
		production
			.fallback_base_urls
			.push(Str::from("https://sandbox.example"));
		let plan = CcaEndpointPlan::from_provider_entries([&production]);
		let served = plan
			.next_after_failure(
				"https://production.example",
				CcaAttemptFailure::HttpStatus(503),
				false,
			)
			.expect("transient production failure selects sandbox");
		let mut events = SmallVec::new();
		events.push(TurnEvent::Outcome(
			ChatOutcome::builder()
				.output(Vec::new())
				.stop(StopReason::EndTurn)
				.unsupported(Vec::new())
				.provider(Str::from("google-antigravity"))
				.model(Str::from("gemini-3.5-flash"))
				.props(Default::default())
				.build(),
		));
		record_served_endpoint(&mut events, served);
		let TurnEvent::Outcome(outcome) = &events[0] else {
			panic!("expected outcome");
		};
		assert_eq!(
			outcome.props.get_ns("google-cca", "served_endpoint"),
			Some(&json!("https://sandbox.example"))
		);
	}

	#[test]
	fn cca_schema_normalization_removes_only_rejected_keywords() {
		let schema = json!({
			"$schema": "https://json-schema.org/draft/2020-12/schema",
			"title": "Arguments",
			"type": "object",
			"additionalProperties": false,
			"patternProperties": {"^x": {"type": "string"}},
			"propertyNames": {"maxLength": 12},
			"properties": {
				"city": {"type": "string", "title": "City", "description": "kept"}
			},
			"required": ["city"],
			"x-extra": true
		});
		let tool = ToolDef::builder()
			.name(Str::from("weather"))
			.description(Str::from("Weather"))
			.schema_json(Bytes::from(serde_json::to_vec(&schema).unwrap()))
			.strict(true)
			.build();
		let (normalized, _) = normalize::normalize_tool(ToolSchemaFlavor::Cca, &tool);
		let value: Value = serde_json::from_slice(&normalized.schema_json).unwrap();
		assert_eq!(
			value,
			json!({
				"type": "object",
				"properties": {"city": {"type": "string", "description": "kept"}},
				"required": ["city"],
				"x-extra": true
			})
		);
	}
	#[test]
	fn cca_off_suppression_emits_the_policy_mode_floor_without_enabling_thoughts() {
		let codec = CcaCodec::new(Str::from("project-a"));
		let mut request = ChatRequest::builder()
			.model(Str::from("resolved-wire-model"))
			.thread(Thread::default())
			.tools(Vec::new())
			.thinking(
				Feature::builder()
					.value(Reasoning::builder().effort(Effort::Off).build())
					.on_unsupported(Fallback::Error)
					.build(),
			)
			.build();

		request.model_policy = Some(Arc::new(ResolvedModelPolicy {
			thinking: Some(ResolvedThinkingPolicy {
				mode:              ResolvedThinkingMode::GoogleLevel,
				efforts:           smallvec![Effort::Low, Effort::High],
				default_effort:    None,
				effort_map:        BTreeMap::new(),
				effort_routing:    BTreeMap::new(),
				effort_budgets:    BTreeMap::new(),
				supports_display:  None,
				suppress_when_off: Some(true),
				requires_effort:   None,
			}),
			..ResolvedModelPolicy::default()
		}));
		let (level_body, level_unsupported) = codec.encode(&request, &Compat::default()).unwrap();
		let level_body: Value = serde_json::from_slice(&level_body).unwrap();
		assert_eq!(level_unsupported, [] as [omp_llm_types::Unsupported; 0]);
		assert_eq!(
			level_body["request"]["generationConfig"]["thinkingConfig"],
			json!({"includeThoughts": false, "thinkingLevel": "LOW"})
		);

		request.model_policy = Some(Arc::new(ResolvedModelPolicy {
			thinking: Some(ResolvedThinkingPolicy {
				mode:              ResolvedThinkingMode::Budget,
				efforts:           smallvec![
					Effort::Minimal,
					Effort::Low,
					Effort::Medium,
					Effort::High
				],
				default_effort:    None,
				effort_map:        BTreeMap::new(),
				effort_routing:    BTreeMap::new(),
				effort_budgets:    BTreeMap::new(),
				supports_display:  None,
				suppress_when_off: Some(true),
				requires_effort:   None,
			}),
			..ResolvedModelPolicy::default()
		}));
		let (budget_body, budget_unsupported) = codec.encode(&request, &Compat::default()).unwrap();
		let budget_body: Value = serde_json::from_slice(&budget_body).unwrap();
		assert_eq!(budget_unsupported, [] as [omp_llm_types::Unsupported; 0]);
		assert_eq!(
			budget_body["request"]["generationConfig"]["thinkingConfig"],
			json!({"includeThoughts": false, "thinkingBudget": 0})
		);
	}
}
