//! Discovery-driven wire-corpus conformance tests.

use std::{
	collections::{BTreeMap, BTreeSet},
	fs,
	path::{Path, PathBuf},
	time::UNIX_EPOCH,
};

use bytes::Bytes;
use futures::{StreamExt, executor::block_on, stream};
use omp_llm_anthropic::AnthropicCodec;
use omp_llm_catalog::compat::{
	Compat, ImageEncodingFormat, LeakedThinkingHealer, ReasoningWireFormat, ToolSchemaFlavor,
};
use omp_llm_egress::retry::parse_retry_after;
use omp_llm_google::{
	GoogleCodec,
	cca::{AntigravityRequestMetadata, CcaCodec},
	stream::{RetryDecision, SemanticRetryBudget},
};
use omp_llm_openai::{OpenAiChatCodec, OpenAiResponsesCodec};
use omp_llm_tower::stack::combinators::heal;
use omp_llm_transport::{DecodeState, Frame, Transport, sse::SseDecoder};
use omp_llm_types::{
	BlobPart, CacheHint, CacheRetention, ChatOutcome, ChatRequest, Fallback, Feature, Item,
	ItemKind, JsonSchema, Message, Part, Props, ResponseFormat, ResponseFormatKind, Role, Sampling,
	StopReason, StreamPartKind, Thinking, Thread, ToolCall, ToolChoice, ToolDef, ToolResult,
	TurnErrorKind, TurnEvent,
	ids::{CallId, CallIdMapper, ToolCallIdProfile},
};
use serde_json::{Value, json};

const FIXED_IDS: [&str; 8] = [
	"01ARZ3NDEKTSV4RRFFQ69G5FAV",
	"01ARZ3NDEKTSV4RRFFQ69G5FAW",
	"01ARZ3NDEKTSV4RRFFQ69G5FAX",
	"01ARZ3NDEKTSV4RRFFQ69G5FAY",
	"01ARZ3NDEKTSV4RRFFQ69G5FAZ",
	"01ARZ3NDEKTSV4RRFFQ69G5FB0",
	"01ARZ3NDEKTSV4RRFFQ69G5FB1",
	"01ARZ3NDEKTSV4RRFFQ69G5FB2",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CodecKind {
	Anthropic,
	OpenAiChat,
	OpenAiResponses,
	Google,
	Cca,
}

#[derive(Clone, Copy)]
struct TransportCase {
	directory: &'static str,
	kind:      CodecKind,
}

const TRANSPORTS: [TransportCase; 5] = [
	TransportCase {
		directory: "../llm-anthropic/tests/fixtures/anthropic",
		kind:      CodecKind::Anthropic,
	},
	TransportCase {
		directory: "../llm-openai/tests/fixtures/openai_chat",
		kind:      CodecKind::OpenAiChat,
	},
	TransportCase {
		directory: "../llm-openai/tests/fixtures/openai_responses",
		kind:      CodecKind::OpenAiResponses,
	},
	TransportCase {
		directory: "../llm-google/tests/fixtures/google_genai",
		kind:      CodecKind::Google,
	},
	TransportCase {
		directory: "../llm-google/tests/fixtures/google_cca",
		kind:      CodecKind::Cca,
	},
];

/// Discovers cases from the corpus at runtime: adding a correctly named fixture
/// must extend coverage without changing this test.
#[test]
fn wire_fixture_corpus() {
	let root = Path::new(env!("CARGO_MANIFEST_DIR"));
	let mut failures = Vec::new();
	let mut exercised = BTreeSet::new();

	for transport in TRANSPORTS {
		let directory = root.join(transport.directory);
		let files = fixture_files(&directory, &mut failures);
		if files.is_empty() {
			failures.push(format!("{}: transport has no fixtures", directory.display()));
			continue;
		}
		let cases = discover_cases(&files, &mut failures);
		for (case_name, case_files) in cases {
			run_case(transport, &case_name, &case_files, &mut exercised, &mut failures);
		}
		for file in files {
			if !exercised.contains(&file) {
				failures.push(format!("{}: fixture was never exercised", file.display()));
			}
		}
	}

	assert!(failures.is_empty(), "wire fixture failures:\n{}", failures.join("\n\n"));
}

fn fixture_files(directory: &Path, failures: &mut Vec<String>) -> Vec<PathBuf> {
	let mut files = match fs::read_dir(directory) {
		Ok(entries) => entries
			.filter_map(Result::ok)
			.map(|entry| entry.path())
			// Multi-attempt fixtures are owned by `responses_recovery_fixtures`.
			.filter(|path| {
				path.is_file()
					&& !path
						.file_name()
						.and_then(|name| name.to_str())
						.is_some_and(|name| name.starts_with("recovery."))
			})
			.collect::<Vec<_>>(),
		Err(error) => {
			failures.push(format!("{}: cannot discover fixtures: {error}", directory.display()));
			Vec::new()
		},
	};
	files.sort();
	files
}

fn discover_cases(files: &[PathBuf], failures: &mut Vec<String>) -> BTreeMap<String, Vec<PathBuf>> {
	let mut cases = BTreeMap::<String, Vec<PathBuf>>::new();
	for file in files {
		let Some(name) = file.file_name().and_then(|name| name.to_str()) else {
			failures.push(format!("{}: non-UTF-8 fixture name", file.display()));
			continue;
		};
		let Some((kind, suffix)) = name.split_once('.') else {
			failures.push(format!("{}: fixture name has no kind", file.display()));
			continue;
		};
		if !matches!(kind, "request" | "stream" | "expect" | "response" | "error") {
			failures.push(format!("{}: unknown fixture kind `{kind}`", file.display()));
			continue;
		}
		let Some((case, _extension)) = suffix.rsplit_once('.') else {
			failures.push(format!("{}: fixture name has no extension", file.display()));
			continue;
		};
		cases.entry(case.to_owned()).or_default().push(file.clone());
	}
	cases
}

fn run_case(
	transport: TransportCase,
	case_name: &str,
	files: &[PathBuf],
	exercised: &mut BTreeSet<PathBuf>,
	failures: &mut Vec<String>,
) {
	let request = find_kind(files, "request");
	let stream_file = find_kind(files, "stream");
	let expect = find_kind(files, "expect");
	let response = find_kind(files, "response");
	let error = find_kind(files, "error");

	if let Some(path) = request {
		exercised.insert(path.to_owned());
		run_encode(transport, path, failures);
	}
	match (stream_file, expect) {
		(Some(stream_path), Some(expect_path)) => {
			exercised.insert(stream_path.to_owned());
			exercised.insert(expect_path.to_owned());
			run_decode(transport, case_name, stream_path, expect_path, failures);
		},
		(Some(path), None) => {
			failures.push(format!("{}: stream has no matching expect fixture", path.display()));
		},
		(None, Some(path)) => {
			failures.push(format!("{}: expect has no matching stream fixture", path.display()));
		},
		(None, None) => {},
	}
	if let Some(path) = response {
		exercised.insert(path.to_owned());
		run_response(transport, path, failures);
	}
	if let Some(path) = error {
		exercised.insert(path.to_owned());
		if transport.kind == CodecKind::OpenAiChat {
			run_openai_error(transport, path, failures);
		} else {
			failures.push(format!(
				"{}: `{}` transport has no error-fixture handler",
				path.display(),
				transport.directory
			));
		}
	}
}

fn find_kind<'a>(files: &'a [PathBuf], kind: &str) -> Option<&'a Path> {
	files.iter().find_map(|path| {
		path
			.file_name()
			.and_then(|name| name.to_str())
			.filter(|name| name.starts_with(&format!("{kind}.")))
			.map(|_| path.as_path())
	})
}

fn read_json(path: &Path) -> Result<Value, String> {
	let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
	serde_json::from_slice(&bytes)
		.map_err(|error| format!("{}: invalid JSON: {error}", path.display()))
}

fn run_encode(transport: TransportCase, path: &Path, failures: &mut Vec<String>) {
	let fixture = match read_json(path) {
		Ok(value) => value,
		Err(error) => {
			failures.push(error);
			return;
		},
	};
	let source = match fs::read(path) {
		Ok(source) => source,
		Err(error) => {
			failures.push(format!("{}: cannot read raw encode fixture: {error}", path.display()));
			return;
		},
	};
	let mut ids = LogicalIds::default();
	let raw_intent = raw_object_field(&source, "canonical_intent").unwrap_or_default();
	let request =
		if transport.kind == CodecKind::OpenAiResponses && fixture["canonical_intent"].is_null() {
			openai_responses_request(&fixture["wire_body"])
		} else if transport.kind == CodecKind::Cca && fixture["requestType"] == "agent" {
			canonical_request(&antigravity_intent(&fixture), &[], &mut ids)
		} else {
			canonical_request(&fixture["canonical_intent"], raw_intent, &mut ids)
		};
	let request = match request {
		Ok(request) => request,
		Err(error) => {
			failures.push(format!("{}: {error}", path.display()));
			return;
		},
	};
	let codec = make_codec(transport.kind, Some(&fixture));
	let fixture_compat =
		if transport.kind == CodecKind::OpenAiResponses && fixture["canonical_intent"].is_null() {
			Compat::default()
		} else {
			compat(transport.kind)
		};
	let (body, _unsupported) = match codec.encode(&request, &fixture_compat) {
		Ok(encoded) => encoded,
		Err(error) => {
			failures.push(format!("{}: encode failed: {error}", path.display()));
			return;
		},
	};
	let actual: Value = match serde_json::from_slice(&body) {
		Ok(value) => value,
		Err(error) => {
			failures.push(format!("{}: codec emitted invalid JSON: {error}", path.display()));
			return;
		},
	};
	let mut expected = if transport.kind == CodecKind::Cca && fixture["requestType"] == "agent" {
		fixture.clone()
	} else {
		fixture["wire_body"].clone()
	};
	replace_logical_ids(&mut expected, &ids, transport.kind);
	if actual != expected {
		failures.push(json_diff(path, "encoded body", &expected, &actual));
	}
	let expected_arguments = verbatim_argument_strings(&expected);
	let actual_arguments = verbatim_argument_strings(&actual);
	if actual_arguments != expected_arguments {
		failures.push(format!(
			"{}: verbatim tool-call argument strings differ\nexpected: \
			 {expected_arguments:?}\nactual:   {actual_arguments:?}",
			path.display()
		));
	}
	let expected_lexemes = raw_object_field(&source, "wire_body")
		.map(raw_argument_lexemes)
		.unwrap_or_default();
	let actual_lexemes = raw_argument_lexemes(&body);
	if actual_lexemes != expected_lexemes {
		failures.push(format!(
			"{}: tool-call argument JSON string lexemes were reserialized\nexpected raw lexemes: \
			 {}\nactual raw lexemes:   {}",
			path.display(),
			debug_bytes(&expected_lexemes),
			debug_bytes(&actual_lexemes),
		));
	}
}

fn make_codec(kind: CodecKind, fixture: Option<&Value>) -> Box<dyn Transport> {
	match kind {
		CodecKind::Anthropic => Box::new(AnthropicCodec::new()),
		CodecKind::OpenAiChat => Box::new(OpenAiChatCodec),
		CodecKind::OpenAiResponses => fixture
			.and_then(|value| value["canonical_intent"]["continuation"]["response_id"].as_str())
			.map_or_else(
				|| Box::new(OpenAiResponsesCodec::new()) as Box<dyn Transport>,
				|id| Box::new(OpenAiResponsesCodec::with_previous_response_id(id)),
			),
		CodecKind::Google => Box::new(GoogleCodec::gen_ai()),
		CodecKind::Cca => {
			let body = fixture.map(|value| {
				if value["requestType"] == "agent" {
					value
				} else {
					&value["wire_body"]
				}
			});
			let project = body
				.and_then(|value| value["project"].as_str())
				.unwrap_or("project-REDACTED");
			if let Some(body) = body.filter(|value| value["requestType"] == "agent") {
				let labels = &body["request"]["labels"];
				let metadata = AntigravityRequestMetadata::new(
					body["request"]["sessionId"]
						.as_str()
						.unwrap_or_default()
						.into(),
					body["requestId"].as_str().unwrap_or_default().into(),
					labels["trajectory_id"].as_str().unwrap_or_default().into(),
					labels["last_step_index"]
						.as_str()
						.and_then(|value| value.parse().ok())
						.unwrap_or(0_u64)
						.saturating_add(1),
				)
				.with_last_execution_id(
					labels["last_execution_id"]
						.as_str()
						.unwrap_or_default()
						.into(),
				)
				.with_model_enum(labels["model_enum"].as_str().unwrap_or_default().into());
				Box::new(CcaCodec::antigravity(project.into(), metadata))
			} else {
				Box::new(CcaCodec::new(project.into()))
			}
		},
	}
}

fn antigravity_intent(fixture: &Value) -> Value {
	let request = &fixture["request"];
	let identity = omp_llm_google::cca::ANTIGRAVITY_SYSTEM_INSTRUCTION;
	let system = request["systemInstruction"]["parts"]
		.as_array()
		.into_iter()
		.flatten()
		.filter_map(|part| part["text"].as_str())
		.filter(|text| *text != identity)
		.map(|text| json!({"type":"text","text":text}))
		.collect::<Vec<_>>();
	let mut messages = Vec::new();
	if !system.is_empty() {
		messages.push(json!({"role":"system","content":system}));
	}
	for content in request["contents"].as_array().into_iter().flatten() {
		let role = if content["role"] == "model" {
			"assistant"
		} else {
			"user"
		};
		let parts = content["parts"]
			.as_array()
			.into_iter()
			.flatten()
			.filter_map(|part| part["text"].as_str())
			.map(|text| json!({"type":"text","text":text}))
			.collect::<Vec<_>>();
		messages.push(json!({"role":role,"content":parts}));
	}
	let tools = request["tools"]
		.as_array()
		.into_iter()
		.flatten()
		.flat_map(|group| {
			group["functionDeclarations"]
				.as_array()
				.into_iter()
				.flatten()
		})
		.map(|tool| {
			json!({
				"name": tool["name"],
				"description": tool["description"].as_str().unwrap_or_default(),
				"schema": tool.get("parameters").unwrap_or(&Value::Null)
			})
		})
		.collect::<Vec<_>>();
	json!({"model":fixture["model"],"messages":messages,"tools":tools})
}

fn compat(kind: CodecKind) -> Compat {
	let mut compat = Compat::default();
	match kind {
		CodecKind::Anthropic => {
			compat.reasoning_wire_format = ReasoningWireFormat::Anthropic;
			compat.image_encoding_format = ImageEncodingFormat::AnthropicSource;
			compat.tool_schema_flavor = ToolSchemaFlavor::Anthropic;
		},
		CodecKind::OpenAiResponses => {
			compat.reasoning_wire_format = ReasoningWireFormat::OpenAiResponses;
			compat.stateful_response_chaining = true;
		},
		CodecKind::Google => {
			compat.reasoning_wire_format = ReasoningWireFormat::Google;
			compat.tool_schema_flavor = ToolSchemaFlavor::Google;
		},
		CodecKind::Cca => {
			compat.reasoning_wire_format = ReasoningWireFormat::Google;
			compat.tool_schema_flavor = ToolSchemaFlavor::Cca;
		},
		CodecKind::OpenAiChat => {},
	}
	compat
}

#[derive(Default)]
struct LogicalIds {
	ids: BTreeMap<String, CallId>,
}

impl LogicalIds {
	fn get(&mut self, logical: &str) -> Result<CallId, String> {
		if let Some(id) = self.ids.get(logical) {
			return Ok(*id);
		}
		let encoded = FIXED_IDS
			.get(self.ids.len())
			.ok_or_else(|| "fixture has more tool ids than the harness id pool".to_owned())?;
		let id = encoded
			.parse()
			.map_err(|error| format!("invalid harness ULID: {error}"))?;
		self.ids.insert(logical.to_owned(), id);
		Ok(id)
	}
}

fn openai_responses_request(wire: &Value) -> Result<ChatRequest, String> {
	let mut items = Vec::new();
	for input in wire["input"].as_array().into_iter().flatten() {
		let role = match input["role"].as_str().unwrap_or("user") {
			"user" => Role::User,
			"assistant" => Role::Assistant,
			"system" => Role::System,
			other => return Err(format!("unsupported Responses fixture role `{other}`")),
		};
		let mut parts = Vec::new();
		let mut props = Props::default();
		for key in ["cache_control", "metadata", "prompt_cache_breakpoint"] {
			if let Some(value) = input.get(key) {
				props.insert_ns("openai", key, value.clone());
			}
		}
		for content in input["content"].as_array().into_iter().flatten() {
			match content["type"].as_str() {
				Some("input_text") => {
					parts.push(Part::Text(content["text"].as_str().unwrap_or_default().into()));
				},
				Some("input_image" | "input_file") => {
					let (source, filename) = if content["type"] == "input_image" {
						(
							content["image_url"]
								.as_str()
								.ok_or("input image has no image_url")?,
							None,
						)
					} else {
						(
							content["file_data"]
								.as_str()
								.ok_or("input file has no file_data")?,
							content["filename"].as_str(),
						)
					};
					let encoded = source
						.strip_prefix("data:")
						.ok_or("Responses fixture media is not an inline data URL")?;
					let (mime, data) = encoded
						.split_once(";base64,")
						.ok_or("Responses fixture data URL is not base64")?;
					let inline = decode_base64(data)?;
					if let Some(detail) = content.get("detail") {
						props.insert_ns("openai", "image_detail", detail.clone());
					}
					if let Some(filename) = filename {
						props.insert_ns("openai", "filename", Value::String(filename.into()));
					}
					parts.push(Part::Blob(
						BlobPart::builder()
							.hash([0; 32])
							.mime(mime.into())
							.size(u64::try_from(inline.len()).map_err(|error| error.to_string())?)
							.inline(Bytes::from(inline))
							.build(),
					));
				},
				other => return Err(format!("unsupported Responses input content {other:?}")),
			}
		}
		items.push(
			Item::builder()
				.seq(0)
				.kind(ItemKind::Message(Message::builder().role(role).parts(parts).build()))
				.props(props)
				.build(),
		);
	}
	let mut options = Props::default();
	for (name, value) in [
		("service_tier", wire.get("service_tier")),
		("metadata", wire.get("metadata")),
		("parallel_tool_calls", wire.get("parallel_tool_calls")),
		("include", wire.get("include")),
		("store", wire.get("store")),
	] {
		if let Some(value) = value {
			options.insert_ns("openai", name, value.clone());
		}
	}
	if let Some(verbosity) = wire.pointer("/text/verbosity") {
		options.insert_ns("openai", "verbosity", verbosity.clone());
	}
	if let Some(tools) = wire.get("tools") {
		options.insert_ns("openai", "hosted_tools", tools.clone());
	}
	let mut request = ChatRequest::builder()
		.model(wire["model"].as_str().unwrap_or_default().into())
		.thread(Thread::builder().items(items).build())
		.tools(Vec::new())
		.provider_options(options)
		.build();
	if let Some(session_key) = wire["prompt_cache_key"].as_str() {
		let cache = CacheHint::builder().session_key(session_key.into());
		request.cache = Some(if wire["prompt_cache_retention"].as_str() == Some("24h") {
			cache.retention(CacheRetention::Long).build()
		} else {
			cache.build()
		});
	}
	Ok(request)
}

fn canonical_request(
	intent: &Value,
	raw_intent: &[u8],
	ids: &mut LogicalIds,
) -> Result<ChatRequest, String> {
	let mut items = Vec::new();
	let mut tool_names = BTreeMap::new();
	let raw_arguments = raw_values_for_field(raw_intent, "arguments");
	let mut raw_arguments = raw_arguments.iter();
	for message in intent["messages"].as_array().into_iter().flatten() {
		let role = message["role"].as_str().ok_or("message has no role")?;
		if role == "tool" {
			let logical = message["tool_call_id"]
				.as_str()
				.ok_or("tool result has no tool_call_id")?;
			let parts = parse_content(&message["content"])?;
			let name = message["tool_name"]
				.as_str()
				.map(Into::into)
				.or_else(|| tool_names.get(logical).cloned())
				.unwrap_or_default();
			items.push(item(ItemKind::ToolResult(
				ToolResult::builder()
					.call_id(ids.get(logical)?)
					.name(name)
					.parts(parts)
					.is_error(message["is_error"].as_bool().unwrap_or(false))
					.build(),
			)));
			continue;
		}
		let role = match role {
			"system" => Role::System,
			"user" => Role::User,
			"assistant" => Role::Assistant,
			other => return Err(format!("unknown message role `{other}`")),
		};
		let content = message["content"]
			.as_array()
			.ok_or("message content is not an array")?;
		let mut buffered = Vec::new();
		for part in content {
			if part["type"] == "tool_call" {
				if !buffered.is_empty() {
					items.push(item(ItemKind::Message(
						Message::builder()
							.role(role)
							.parts(std::mem::take(&mut buffered))
							.build(),
					)));
				}
				let logical = part["id"].as_str().ok_or("tool call has no id")?;
				let name = part["name"].as_str().unwrap_or_default();
				tool_names.insert(logical, name.into());
				let args = raw_arguments
					.next()
					.map_or_else(|| serde_json::to_vec(&part["arguments"]), |raw| Ok((*raw).to_vec()))
					.map_err(|error| format!("cannot serialize tool arguments: {error}"))?;
				items.push(item(ItemKind::ToolCall(
					ToolCall::builder()
						.id(ids.get(logical)?)
						.name(name.into())
						.args_json(Bytes::from(args))
						.thought_signature(Bytes::new())
						.build(),
				)));
			} else {
				buffered.push(parse_part(part)?);
			}
		}
		if !buffered.is_empty() {
			items.push(item(ItemKind::Message(Message::builder().role(role).parts(buffered).build())));
		}
	}
	let tools = intent["tools"]
		.as_array()
		.into_iter()
		.flatten()
		.map(|tool| {
			let schema = serde_json::to_vec(&tool["schema"])
				.map_err(|error| format!("cannot serialize tool schema: {error}"))?;
			Ok(ToolDef::builder()
				.name(tool["name"].as_str().unwrap_or_default().into())
				.description(tool["description"].as_str().unwrap_or_default().into())
				.schema_json(Bytes::from(schema))
				.maybe_strict(tool["strict"].as_bool())
				.build())
		})
		.collect::<Result<Vec<_>, String>>()?;
	let tool_choice = parse_tool_choice(&intent["tool_choice"])?;
	let sampling = parse_sampling(&intent["sampling"])?;
	let response_format = parse_response_format(&intent["response_format"])?;
	let provider_options = parse_provider_options(&intent["provider_options"])?;
	Ok(ChatRequest::builder()
		.model(intent["model"].as_str().unwrap_or_default().into())
		.thread(Thread::builder().items(items).build())
		.tools(tools)
		.maybe_tool_choice(tool_choice)
		.maybe_sampling(sampling)
		.maybe_response_format(response_format)
		.maybe_provider_options(provider_options)
		.build())
}

fn parse_tool_choice(value: &Value) -> Result<Option<Feature<ToolChoice>>, String> {
	if value.is_null() {
		return Ok(None);
	}
	let kind = value
		.as_str()
		.or_else(|| value["type"].as_str())
		.ok_or("tool_choice must be a string or object")?;
	let choice = match kind {
		"auto" => ToolChoice::Auto,
		"none" => ToolChoice::None,
		"required" | "any" => ToolChoice::Required,
		"named" | "tool" => ToolChoice::Named(
			value["name"]
				.as_str()
				.ok_or("named tool_choice has no name")?
				.into(),
		),
		other => return Err(format!("unknown tool_choice `{other}`")),
	};
	Ok(Some(
		Feature::builder()
			.value(choice)
			.on_unsupported(Fallback::Error)
			.build(),
	))
}

fn parse_sampling(value: &Value) -> Result<Option<Sampling>, String> {
	if value.is_null() {
		return Ok(None);
	}
	let object = value.as_object().ok_or("sampling must be an object")?;
	let stop = object
		.get("stop")
		.or_else(|| object.get("stop_sequences"))
		.map(|value| {
			value
				.as_array()
				.ok_or("sampling stop must be an array")?
				.iter()
				.map(|value| {
					value
						.as_str()
						.map(Into::into)
						.ok_or("sampling stop must contain strings")
				})
				.collect::<Result<Vec<_>, _>>()
		})
		.transpose()?;
	Ok(Some(
		Sampling::builder()
			.maybe_temperature(value["temperature"].as_f64())
			.maybe_top_p(value["top_p"].as_f64())
			.maybe_top_k(value["top_k"].as_u64().map(|value| value as u32))
			.maybe_stop(stop)
			.maybe_max_output_tokens(value["max_output_tokens"].as_u64())
			.build(),
	))
}

fn parse_response_format(value: &Value) -> Result<Option<Feature<ResponseFormat>>, String> {
	if value.is_null() {
		return Ok(None);
	}
	let kind = value["type"].as_str().or_else(|| value["kind"].as_str());
	if kind != Some("json_schema") {
		return Err("only json_schema response_format is supported by fixtures".into());
	}
	let schema = serde_json::to_vec(&value["schema"])
		.map_err(|error| format!("cannot serialize response schema: {error}"))?;
	Ok(Some(
		Feature::builder()
			.value(
				ResponseFormat::builder()
					.kind(ResponseFormatKind::JsonSchema(
						JsonSchema::builder()
							.name(value["name"].as_str().unwrap_or_default().into())
							.schema_json(Bytes::from(schema))
							.maybe_strict(value["strict"].as_bool())
							.build(),
					))
					.build(),
			)
			.on_unsupported(Fallback::Error)
			.build(),
	))
}

fn parse_provider_options(value: &Value) -> Result<Option<Props>, String> {
	if value.is_null() {
		return Ok(None);
	}
	let object = value
		.as_object()
		.ok_or("provider_options must be an object")?;
	let mut props = Props::default();
	for (key, value) in object {
		if let Some((namespace, name)) = key.split_once('/') {
			props.insert_ns(namespace, name, value.clone());
		} else if let Some(namespaced) = value.as_object() {
			for (name, value) in namespaced {
				props.insert_ns(key, name, value.clone());
			}
		} else {
			return Err(format!("provider option `{key}` is not namespaced"));
		}
	}
	Ok(Some(props))
}

fn item(kind: ItemKind) -> Item {
	Item::builder()
		.seq(0)
		.kind(kind)
		.props(Props::default())
		.build()
}

fn parse_content(value: &Value) -> Result<Vec<Part>, String> {
	if let Some(text) = value.as_str() {
		return Ok(vec![Part::Text(text.into())]);
	}
	value
		.as_array()
		.ok_or_else(|| "content is neither text nor an array".to_owned())?
		.iter()
		.map(parse_part)
		.collect()
}

fn parse_part(value: &Value) -> Result<Part, String> {
	match value["type"].as_str() {
		Some("text") => Ok(Part::Text(value["text"].as_str().unwrap_or_default().into())),
		Some("thinking") => Ok(Part::Thinking(
			Thinking::builder()
				.text(value["text"].as_str().unwrap_or_default().into())
				.signature(Bytes::copy_from_slice(
					value["signature"].as_str().unwrap_or_default().as_bytes(),
				))
				.redacted(value["redacted"].as_bool().unwrap_or(false))
				.build(),
		)),
		Some("image" | "document") => {
			let kind = value["type"].as_str().unwrap_or_default();
			let inline = decode_base64(
				value["data"]
					.as_str()
					.ok_or_else(|| format!("{kind} has no base64 data"))?,
			)?;
			Ok(Part::Blob(
				BlobPart::builder()
					.hash([0; 32])
					.mime(
						value["media_type"]
							.as_str()
							.unwrap_or("application/octet-stream")
							.into(),
					)
					.size(u64::try_from(inline.len()).map_err(|error| error.to_string())?)
					.inline(Bytes::from(inline))
					.build(),
			))
		},
		other => Err(format!("unsupported canonical content part {other:?}")),
	}
}

fn decode_base64(value: &str) -> Result<Vec<u8>, String> {
	const fn digit(byte: u8) -> Option<u8> {
		match byte {
			b'A'..=b'Z' => Some(byte - b'A'),
			b'a'..=b'z' => Some(byte - b'a' + 26),
			b'0'..=b'9' => Some(byte - b'0' + 52),
			b'+' => Some(62),
			b'/' => Some(63),
			_ => None,
		}
	}
	let mut out = Vec::with_capacity(value.len() / 4 * 3);
	for block in value.as_bytes().chunks(4) {
		if block.len() != 4 {
			return Err("base64 length is not divisible by four".into());
		}
		let a = digit(block[0]).ok_or("invalid base64 digit")?;
		let b = digit(block[1]).ok_or("invalid base64 digit")?;
		let c = (block[2] != b'=').then(|| digit(block[2])).flatten();
		let d = (block[3] != b'=').then(|| digit(block[3])).flatten();
		out.push((a << 2) | (b >> 4));
		if let Some(c) = c {
			out.push((b << 4) | (c >> 2));
			if let Some(d) = d {
				out.push((c << 6) | d);
			}
		}
	}
	Ok(out)
}

fn replace_logical_ids(value: &mut Value, ids: &LogicalIds, kind: CodecKind) {
	let mapper = CallIdMapper::new();
	replace_logical_ids_inner(value, ids, kind, &mapper);
}

fn replace_logical_ids_inner(
	value: &mut Value,
	ids: &LogicalIds,
	kind: CodecKind,
	mapper: &CallIdMapper,
) {
	match value {
		Value::Array(values) => values
			.iter_mut()
			.for_each(|value| replace_logical_ids_inner(value, ids, kind, mapper)),
		Value::Object(values) => values
			.values_mut()
			.for_each(|value| replace_logical_ids_inner(value, ids, kind, mapper)),
		Value::String(text) => {
			if let Some(id) = ids.ids.get(text.as_str()) {
				*text = if kind == CodecKind::Anthropic {
					mapper.to_wire(id, ToolCallIdProfile::Anthropic).to_string()
				} else {
					id.to_string()
				};
			}
		},
		Value::Null | Value::Bool(_) | Value::Number(_) => {},
	}
}

fn verbatim_argument_strings(value: &Value) -> Vec<String> {
	fn visit(value: &Value, parent: Option<&str>, out: &mut Vec<String>) {
		match value {
			Value::Object(map) => {
				for (key, value) in map {
					visit(value, Some(key), out);
				}
			},
			Value::Array(values) => values.iter().for_each(|value| visit(value, parent, out)),
			Value::String(text) if matches!(parent, Some("arguments" | "input" | "args")) => {
				out.push(text.clone());
			},
			Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {},
		}
	}
	let mut out = Vec::new();
	visit(value, None, &mut out);
	out
}

fn raw_object_field<'a>(bytes: &'a [u8], wanted: &str) -> Option<&'a [u8]> {
	let mut cursor = 0;
	while cursor < bytes.len() {
		if bytes[cursor] != b'"' {
			cursor += 1;
			continue;
		}
		let end = json_string_end(bytes, cursor)?;
		let key = serde_json::from_slice::<String>(&bytes[cursor..end]).ok()?;
		let mut value_start = end;
		while bytes.get(value_start).is_some_and(u8::is_ascii_whitespace) {
			value_start += 1;
		}
		if bytes.get(value_start) != Some(&b':') {
			cursor = end;
			continue;
		}
		value_start += 1;
		while bytes.get(value_start).is_some_and(u8::is_ascii_whitespace) {
			value_start += 1;
		}
		if key == wanted {
			let value_end = json_value_end(bytes, value_start)?;
			return Some(&bytes[value_start..value_end]);
		}
		cursor = end;
	}
	None
}

fn raw_values_for_field<'a>(bytes: &'a [u8], field: &str) -> Vec<&'a [u8]> {
	let mut out = Vec::new();
	let mut rest = bytes;
	while let Some(value) = raw_object_field(rest, field) {
		out.push(value);
		let consumed = value.as_ptr() as usize - rest.as_ptr() as usize + value.len();
		rest = &rest[consumed..];
	}
	out
}

fn raw_argument_lexemes(bytes: &[u8]) -> Vec<Vec<u8>> {
	let mut out = Vec::new();
	for field in ["arguments", "input", "args"] {
		let mut rest = bytes;
		while let Some(value) = raw_object_field(rest, field) {
			if value.first() == Some(&b'"')
				&& let Some(end) = json_string_end(value, 0)
			{
				out.push(value[..end].to_vec());
			}
			let consumed = value.as_ptr() as usize - rest.as_ptr() as usize + value.len();
			rest = &rest[consumed..];
		}
	}
	out
}

fn json_string_end(bytes: &[u8], start: usize) -> Option<usize> {
	if bytes.get(start) != Some(&b'"') {
		return None;
	}
	let mut escaped = false;
	for (offset, &byte) in bytes[start + 1..].iter().enumerate() {
		if escaped {
			escaped = false;
		} else if byte == b'\\' {
			escaped = true;
		} else if byte == b'"' {
			return Some(start + offset + 2);
		}
	}
	None
}

fn json_value_end(bytes: &[u8], start: usize) -> Option<usize> {
	match *bytes.get(start)? {
		b'"' => json_string_end(bytes, start),
		b'{' | b'[' => {
			let opening = bytes[start];
			let closing = if opening == b'{' { b'}' } else { b']' };
			let mut depth = 0_u32;
			let mut cursor = start;
			while cursor < bytes.len() {
				if bytes[cursor] == b'"' {
					cursor = json_string_end(bytes, cursor)?;
					continue;
				}
				if bytes[cursor] == opening {
					depth += 1;
				} else if bytes[cursor] == closing {
					depth -= 1;
					if depth == 0 {
						return Some(cursor + 1);
					}
				}
				cursor += 1;
			}
			None
		},
		_ => bytes[start..]
			.iter()
			.position(|byte| matches!(byte, b',' | b'}' | b']') || byte.is_ascii_whitespace())
			.map(|offset| start + offset)
			.or(Some(bytes.len())),
	}
}

fn debug_bytes(values: &[Vec<u8>]) -> String {
	values
		.iter()
		.map(|value| String::from_utf8_lossy(value).into_owned())
		.collect::<Vec<_>>()
		.join(", ")
}

fn run_decode(
	transport: TransportCase,
	case_name: &str,
	stream_path: &Path,
	expect_path: &Path,
	failures: &mut Vec<String>,
) {
	let fixture = match read_json(expect_path) {
		Ok(value) => value,
		Err(error) => {
			failures.push(error);
			return;
		},
	};
	if transport.kind == CodecKind::Google && case_name == "retry_cases" {
		run_google_retry_cases(stream_path, &fixture, failures);
		return;
	}
	let (raw, physical) = match stream_bytes(stream_path) {
		Ok(value) => value,
		Err(error) => {
			failures.push(error);
			return;
		},
	};
	let mut strategies = vec![
		("whole", vec![raw.as_slice()]),
		("seven-byte", raw.chunks(7).collect()),
		("utf8-hostile", utf8_hostile_chunks(&raw)),
	];
	if let Some(boundaries) = physical.as_ref() {
		strategies.push(("recorded-physical", chunks_at(&raw, boundaries)));
	}

	let mut baseline: Option<(&str, Vec<Value>)> = None;
	for (strategy, chunks) in strategies {
		let mut events = match decode_chunks(transport.kind, case_name, &chunks) {
			Ok(events) => events,
			Err(error) => {
				failures.push(format!("{} [{strategy}]: {error}", stream_path.display()));
				continue;
			},
		};
		if transport.kind == CodecKind::OpenAiChat && case_name == "leaked_tags" {
			assert_leaked_tag_modes(stream_path, &events, failures);
			events = block_on(heal(stream::iter(events), healer_compat()).collect::<Vec<_>>());
		}
		let actual = normalized_actual(&events, transport.kind);
		if let Some((baseline_name, baseline_events)) = &baseline {
			if &actual != baseline_events {
				failures.push(event_diff(
					stream_path,
					&format!("chunking `{baseline_name}`"),
					baseline_events,
					&format!("chunking `{strategy}`"),
					&actual,
				));
			}
		} else {
			baseline = Some((strategy, actual.clone()));
		}
		let expected = normalized_expected(&fixture["events"]);
		if actual != expected {
			failures.push(event_diff(stream_path, "fixture", &expected, strategy, &actual));
		}
	}
}

fn run_google_retry_cases(path: &Path, expected: &Value, failures: &mut Vec<String>) {
	let source = match read_json(path) {
		Ok(value) => value,
		Err(error) => {
			failures.push(error);
			return;
		},
	};
	let cases = &expected["cases"];
	for (name, values) in [
		("empty_stream", source["empty_stream"]["responses"].as_array()),
		("empty_response", source["empty_response"]["responses"].as_array()),
	] {
		let actual = values
			.into_iter()
			.flatten()
			.map(|value| decode_google_retry_value(path, name, value, failures))
			.collect::<Vec<_>>();
		if cases[name]["decoded"].as_array() != Some(&actual) {
			failures.push(json_diff(
				path,
				&format!("{name} decoded terminals"),
				&cases[name]["decoded"],
				&Value::Array(actual),
			));
		}
	}
	let overload_values =
		[&source["overload_recovery"]["first"], &source["overload_recovery"]["second"]];
	let overload = overload_values
		.into_iter()
		.map(|value| decode_google_retry_value(path, "overload_recovery", value, failures))
		.collect::<Vec<_>>();
	if cases["overload_recovery"]["decoded"].as_array() != Some(&overload) {
		failures.push(json_diff(
			path,
			"overload recovery decoded terminals",
			&cases["overload_recovery"]["decoded"],
			&Value::Array(overload),
		));
	}
	let context_wire = json!({"error": source["context_overflow"]["error"]});
	let context = vec![decode_google_retry_value(path, "context_overflow", &context_wire, failures)];
	if cases["context_overflow"]["decoded"].as_array() != Some(&context) {
		failures.push(json_diff(
			path,
			"context overflow decoded terminal",
			&cases["context_overflow"]["decoded"],
			&Value::Array(context),
		));
	}

	let mut empty_stream = SemanticRetryBudget::default();
	let empty_stream = (0..3)
		.map(|_| retry_decision(empty_stream.empty_stream()))
		.collect::<Vec<_>>();
	let mut empty_response = SemanticRetryBudget::default();
	let empty_response = (0..3)
		.map(|_| retry_decision(empty_response.empty_response()))
		.collect::<Vec<_>>();
	let mut overload_budget = SemanticRetryBudget::default();
	let overload_retry = vec![retry_decision(overload_budget.overload())];
	let mut cancelled = SemanticRetryBudget::default();
	cancelled.cancel();
	let cancellation = vec![retry_decision(cancelled.empty_stream())];
	for (name, actual) in [
		("empty_stream", empty_stream),
		("empty_response", empty_response),
		("overload_recovery", overload_retry),
		("cancellation", cancellation),
	] {
		if cases[name]["retry"].as_array() != Some(&actual) {
			failures.push(json_diff(
				path,
				&format!("{name} retry decisions"),
				&cases[name]["retry"],
				&Value::Array(actual),
			));
		}
	}
}

fn decode_google_retry_value(
	path: &Path,
	case_name: &str,
	value: &Value,
	failures: &mut Vec<String>,
) -> Value {
	let raw =
		format!("data: {}\n\n", serde_json::to_string(value).expect("fixture value serializes"))
			.into_bytes();
	let strategies = [
		("whole", vec![raw.as_slice()]),
		("seven-byte", raw.chunks(7).collect()),
		("utf8-hostile", utf8_hostile_chunks(&raw)),
	];
	let mut baseline = None;
	for (strategy, chunks) in strategies {
		let terminal = match decode_chunks(CodecKind::Google, case_name, &chunks) {
			Ok(events) => normalized_actual(&events, CodecKind::Google)
				.into_iter()
				.rev()
				.find_map(|event| match event["type"].as_str() {
					Some("done") => event["reason"].as_str().map(|value| json!(value)),
					Some("transport_error") => event["kind"].as_str().map(|value| json!(value)),
					_ => None,
				})
				.unwrap_or_else(|| json!("missing_terminal")),
			Err(error) => json!(format!("decode_error:{error}")),
		};
		if let Some(expected) = &baseline {
			if &terminal != expected {
				failures.push(format!(
					"{} [{case_name}/{strategy}]: terminal differs across chunking",
					path.display()
				));
			}
		} else {
			baseline = Some(terminal);
		}
	}
	baseline.unwrap_or_else(|| json!("missing_terminal"))
}

fn retry_decision(decision: RetryDecision) -> Value {
	match decision {
		RetryDecision::RetryAfter(delay) => json!(delay),
		RetryDecision::Terminal => json!("terminal"),
	}
}

fn healer_compat() -> Compat {
	let mut compat = Compat::default();
	compat.leaked_thinking_healer = LeakedThinkingHealer::Thinking;
	compat
}

fn stream_bytes(path: &Path) -> Result<(Vec<u8>, Option<Vec<usize>>), String> {
	if path.extension().and_then(|extension| extension.to_str()) == Some("json") {
		let fixture = read_json(path)?;
		let raw = fixture["concatenated_utf8"]
			.as_str()
			.ok_or_else(|| format!("{}: physical stream has no concatenated_utf8", path.display()))?
			.as_bytes()
			.to_vec();
		let boundaries = fixture["boundaries"]
			.as_array()
			.ok_or_else(|| format!("{}: physical stream has no boundaries", path.display()))?
			.iter()
			.map(|value| value.as_u64().and_then(|value| usize::try_from(value).ok()))
			.collect::<Option<Vec<_>>>()
			.ok_or_else(|| format!("{}: invalid physical boundaries", path.display()))?;
		Ok((raw, Some(boundaries)))
	} else {
		fs::read(path)
			.map(|bytes| (bytes, None))
			.map_err(|error| format!("{}: {error}", path.display()))
	}
}

fn utf8_hostile_chunks(bytes: &[u8]) -> Vec<&[u8]> {
	let mut boundaries = bytes
		.iter()
		.enumerate()
		.filter_map(|(index, byte)| (index > 0 && byte & 0xc0 == 0x80).then_some(index))
		.collect::<Vec<_>>();
	if boundaries.is_empty() && bytes.len() > 2 {
		boundaries.extend([bytes.len() / 3, bytes.len() * 2 / 3]);
	}
	chunks_at(bytes, &boundaries)
}

fn chunks_at<'a>(bytes: &'a [u8], boundaries: &[usize]) -> Vec<&'a [u8]> {
	let mut start = 0;
	let mut chunks = Vec::new();
	for &end in boundaries {
		if end > start && end < bytes.len() {
			chunks.push(&bytes[start..end]);
			start = end;
		}
	}
	chunks.push(&bytes[start..]);
	chunks
}

fn decode_chunks(
	kind: CodecKind,
	case_name: &str,
	chunks: &[&[u8]],
) -> Result<Vec<TurnEvent>, String> {
	let codec: Box<dyn Transport> = if kind == CodecKind::Cca && case_name == "antigravity_leak" {
		Box::new(CcaCodec::new("project-REDACTED".into()).with_planning_leak_filter(["read".into()]))
	} else {
		make_codec(kind, None)
	};
	let mut transport = SseDecoder::new();
	let mut state = DecodeState::default();
	let mut events = Vec::new();
	for chunk in chunks {
		let frames = transport
			.push(Bytes::copy_from_slice(chunk))
			.collect::<Vec<_>>();
		for frame in frames {
			events.extend(
				codec
					.decode(Frame::Event { name: frame.name.as_deref(), data: &frame.data }, &mut state)
					.map_err(|error| error.to_string())?,
			);
		}
	}
	events.extend(
		codec
			.decode(Frame::Done, &mut state)
			.map_err(|error| error.to_string())?,
	);
	Ok(events)
}

fn assert_leaked_tag_modes(path: &Path, events: &[TurnEvent], failures: &mut Vec<String>) {
	let pass_through =
		block_on(heal(stream::iter(events.to_vec()), Compat::default()).collect::<Vec<_>>());
	if pass_through != events {
		failures.push(format!(
			"{}: healer-off mode did not pass events through byte-for-byte",
			path.display()
		));
	}
	let raw = part_text(events, StreamPartKind::Text);
	if !raw
		.windows(b"<think>".len())
		.any(|window| window == b"<think>")
	{
		failures.push(format!(
			"{}: healer-off stream no longer contains leaked <think> markup",
			path.display()
		));
	}
}

fn part_text(events: &[TurnEvent], wanted: StreamPartKind) -> Vec<u8> {
	let mut kinds = BTreeMap::new();
	let mut out = Vec::new();
	for event in events {
		match event {
			TurnEvent::PartStart { index, kind, .. } => {
				kinds.insert(*index, *kind);
			},
			TurnEvent::PartDelta { index, chunk } if kinds.get(index) == Some(&wanted) => {
				out.extend_from_slice(chunk);
			},
			_ => {},
		}
	}
	out
}

fn normalized_actual(events: &[TurnEvent], kind: CodecKind) -> Vec<Value> {
	let outcome = events.iter().find_map(|event| match event {
		TurnEvent::Outcome(outcome) => Some(outcome),
		_ => None,
	});
	let metadata = outcome.map(outcome_metadata).unwrap_or_default();
	let mut kinds = BTreeMap::new();
	let mut tool_names = BTreeMap::new();
	let mut thinking_number = 0;
	let mut tool_number = 0;
	let mut text_number = 0;
	let mut out = Vec::new();
	for event in events {
		match event {
			TurnEvent::PartStart { index, kind: part_kind, tool_name, .. } => {
				kinds.insert(*index, *part_kind);
				if *part_kind == StreamPartKind::ToolCall {
					tool_names.insert(*index, tool_name.clone());
					out.push(json!({"type":"tool_start","name":tool_name}));
				} else if *part_kind == StreamPartKind::Thinking {
					if let Some(Some(encrypted)) = metadata.encrypted.get(thinking_number) {
						out.push(json!({"type":"thinking_encrypted","data":encrypted}));
					}
					thinking_number += 1;
				}
			},
			TurnEvent::PartDelta { index, chunk } => match kinds.get(index) {
				Some(StreamPartKind::Text) => out.push(json!({"type":"text_delta","text":String::from_utf8_lossy(chunk)})),
				Some(StreamPartKind::Thinking) => out.push(json!({"type":"thinking_delta","text":String::from_utf8_lossy(chunk)})),
				Some(StreamPartKind::ToolCall) => out.push(json!({"type":"tool_args","fragment":String::from_utf8_lossy(chunk)})),
				None => out.push(json!({"type":"orphan_delta","index":index,"bytes":String::from_utf8_lossy(chunk)})),
				Some(_) => out.push(json!({"type":"unknown_delta_kind","index":index})),
			},
			TurnEvent::PartEnd { index, .. } => match kinds.get(index) {
				Some(StreamPartKind::Text) => {
					if let Some(Some(signature)) = metadata.text_signatures.get(text_number) {
						out.push(json!({"type":"signature","kind":"text","value":signature}));
					}
					text_number += 1;
				},
				Some(StreamPartKind::Thinking) => {
					let signature_index = thinking_number.saturating_sub(1);
					if let Some(Some(signature)) = metadata.thinking_signatures.get(signature_index) {
						out.push(json!({"type":"signature","kind":"thinking","value":signature}));
					}
				},
				Some(StreamPartKind::ToolCall) => {
					if let Some(Some(signature)) = metadata.tool_signatures.get(tool_number) {
						out.push(json!({"type":"signature","kind":"tool","value":signature}));
					}
					tool_number += 1;
				},
				_ => {},
			},
			TurnEvent::Outcome(outcome) => {
				if let Some(usage) = &outcome.usage {
					out.push(json!({
						"type":"usage", "input":usage.input_tokens, "output":usage.output_tokens,
						"cache_read":usage.cache_read_tokens, "cache_write":usage.cache_write_tokens
					}));
				}
				out.push(json!({"type":"done","reason":stop_name(outcome.stop)}));
			},
			TurnEvent::Error(error) => out.push(json!({
				"type":"transport_error",
				"kind": if error.kind == TurnErrorKind::Upstream && error.detail.contains("ended before") { "truncated_stream" } else { error_kind_name(error.kind) }
			})),
			TurnEvent::Accepted { .. }
			| TurnEvent::Attempt { .. }
			| TurnEvent::Invoke(_)
			| TurnEvent::InvokeCancel { .. } => {},
			_ => {},
		}
	}
	let _ = (kind, tool_names);
	out
}

#[derive(Default)]
struct OutcomeMetadata {
	thinking_signatures: Vec<Option<String>>,
	tool_signatures:     Vec<Option<String>>,
	text_signatures:     Vec<Option<String>>,
	encrypted:           Vec<Option<String>>,
}

fn outcome_metadata(outcome: &ChatOutcome) -> OutcomeMetadata {
	let mut metadata = OutcomeMetadata::default();
	for item in &outcome.output {
		match &item.kind {
			ItemKind::Message(message) => {
				let signatures = item
					.props
					.get_ns("google", "text_thought_signatures")
					.and_then(Value::as_array);
				for (index, part) in message.parts.iter().enumerate() {
					match part {
						Part::Thinking(thinking) => {
							let signature = String::from_utf8_lossy(&thinking.signature).into_owned();
							metadata
								.encrypted
								.push(thinking.redacted.then(|| signature.clone()));
							metadata
								.thinking_signatures
								.push((!thinking.redacted && !signature.is_empty()).then_some(signature));
						},
						Part::Text(_) => metadata.text_signatures.push(
							signatures
								.and_then(|values| values.get(index))
								.and_then(Value::as_str)
								.map(str::to_owned),
						),
						Part::Blob(_) => {},
						_ => {},
					}
				}
			},
			ItemKind::ToolCall(call) => {
				metadata.tool_signatures.push(
					(!call.thought_signature.is_empty())
						.then(|| String::from_utf8_lossy(&call.thought_signature).into_owned()),
				);
			},
			ItemKind::ToolResult(_) => {},
			_ => {},
		}
	}
	metadata
}

fn normalized_expected(events: &Value) -> Vec<Value> {
	let mut out = Vec::new();
	for event in events.as_array().into_iter().flatten() {
		match event["type"].as_str() {
			Some("thinking_delta") => {
				out.push(json!({"type":"thinking_delta","text":event["text"]}));
				if let Some(signature) = event["signature"].as_str() {
					out.push(json!({"type":"signature","kind":"thinking","value":signature}));
				}
			},
			Some("thinking_signature") => {
				out.push(json!({"type":"signature","kind":"thinking","value":event["signature"]}));
			},
			Some("thinking_encrypted") => {
				out.push(json!({"type":"thinking_encrypted","data":event["data"]}));
			},
			Some("text_delta") => {
				out.push(json!({"type":"text_delta","text":event["text"]}));
				if let Some(signature) = event["signature"].as_str() {
					out.push(json!({"type":"signature","kind":"text","value":signature}));
				}
			},
			Some("tool_call_start") => out.push(json!({"type":"tool_start","name":event["name"]})),
			Some("tool_call_arguments_delta") => {
				out.push(json!({"type":"tool_args","fragment":event["fragment"]}));
			},
			Some("tool_call") => {
				out.push(json!({"type":"tool_start","name":event["name"]}));
				out.push(
					json!({"type":"tool_args","fragment":serde_json::to_string(&event["arguments"]).unwrap()}),
				);
				if let Some(signature) = event["signature"].as_str() {
					out.push(json!({"type":"signature","kind":"tool","value":signature}));
				}
			},
			Some("usage") => out.push(json!({
				"type":"usage", "input":event["input"], "output":event["output"],
				"cache_read":event["cache_read"], "cache_write":event["cache_write"]
			})),
			Some("done") => out.push(json!({"type":"done","reason":event["reason"]})),
			Some("transport_error") => {
				out.push(json!({"type":"transport_error","kind":event["kind"]}));
			},
			other => out.push(json!({"type":"unknown_fixture_event","original_type":other})),
		}
	}
	out
}

const fn stop_name(stop: StopReason) -> &'static str {
	match stop {
		StopReason::EndTurn => "stop",
		StopReason::ToolUse => "tool_use",
		StopReason::MaxTokens => "max_tokens",
		StopReason::ContentFilter => "content_filter",
		_ => "unknown",
	}
}

const fn error_kind_name(kind: TurnErrorKind) -> &'static str {
	match kind {
		TurnErrorKind::Conflict => "conflict",
		TurnErrorKind::NeedFull => "need_full",
		TurnErrorKind::Unsupported => "unsupported",
		TurnErrorKind::Auth => "auth",
		TurnErrorKind::RateLimited => "rate_limit",
		TurnErrorKind::Upstream => "upstream",
		TurnErrorKind::Overloaded => "overloaded",
		TurnErrorKind::InvokeTimeout => "invoke_timeout",
		_ => "unknown",
	}
}

fn run_openai_error(transport: TransportCase, path: &Path, failures: &mut Vec<String>) {
	let body = match fs::read(path) {
		Ok(body) => body,
		Err(error) => {
			failures.push(format!("{}: cannot read error fixture: {error}", path.display()));
			return;
		},
	};
	let fixture = match serde_json::from_slice::<Value>(&body) {
		Ok(fixture) => fixture,
		Err(error) => {
			failures.push(format!("{}: invalid error fixture JSON: {error}", path.display()));
			return;
		},
	};
	let expected_detail = fixture
		.pointer("/error/metadata/raw")
		.or_else(|| fixture.pointer("/error/message"))
		.and_then(Value::as_str)
		.unwrap_or_default();
	if expected_detail.is_empty() {
		failures.push(format!(
			"{}: error fixture has neither error.message nor error.metadata.raw",
			path.display()
		));
		return;
	}
	for (phase, seed) in [
		("pre-commit", None),
		(
			"post-commit",
			Some(
				br#"{"id":"fixture_commit","choices":[{"index":0,"delta":{"content":"committed"},"finish_reason":null}]}"#
					.as_slice(),
			),
		),
	] {
		let codec = make_codec(transport.kind, None);
		let mut state = DecodeState::default();
		if let Some(seed) = seed {
			match codec.decode(Frame::Data(seed), &mut state) {
				Ok(events) if events.iter().any(|event| matches!(event, TurnEvent::PartDelta { .. })) => {},
				Ok(_) => {
					failures.push(format!(
						"{} [{phase}]: seed did not commit a visible event",
						path.display()
					));
					continue;
				},
				Err(error) => {
					failures.push(format!(
						"{} [{phase}]: seed decode failed: {error}",
						path.display()
					));
					continue;
				},
			}
		}
		let events = match codec.decode(Frame::Data(&body), &mut state) {
			Ok(events) => events,
			Err(error) => {
				failures.push(format!("{} [{phase}]: error decode failed: {error}", path.display()));
				continue;
			},
		};
		let terminal = events
			.iter()
			.filter_map(|event| match event {
				TurnEvent::Error(error) => Some(error),
				TurnEvent::Outcome(_) => None,
				_ => None,
			})
			.collect::<Vec<_>>();
		if terminal.len() != 1
			|| terminal[0].kind != TurnErrorKind::Upstream
			|| !terminal[0].detail.contains(expected_detail)
			|| events.iter().any(|event| matches!(event, TurnEvent::Outcome(_)))
		{
			failures.push(format!(
				"{} [{phase}]: expected one upstream error retaining `{expected_detail}`, got {}",
				path.display(),
				serde_json::to_string(&normalized_actual(&events, transport.kind))
					.unwrap_or_else(|_| "<unserializable>".into())
			));
		}
		match codec.decode(Frame::Done, &mut state) {
			Ok(events) if events.is_empty() => {},
			Ok(events) => failures.push(format!(
				"{} [{phase}]: terminal error was followed by {} events",
				path.display(),
				events.len()
			)),
			Err(error) => failures.push(format!(
				"{} [{phase}]: terminal Done decode failed: {error}",
				path.display()
			)),
		}
	}
}

fn run_response(transport: TransportCase, path: &Path, failures: &mut Vec<String>) {
	let fixture = match read_json(path) {
		Ok(value) => value,
		Err(error) => {
			failures.push(error);
			return;
		},
	};
	let body = match serde_json::to_vec(&fixture["body"]) {
		Ok(body) => body,
		Err(error) => {
			failures.push(format!("{}: cannot serialize response body: {error}", path.display()));
			return;
		},
	};
	let codec = make_codec(transport.kind, None);
	let mut state = DecodeState::default();
	let events = match codec.decode(Frame::Data(&body), &mut state) {
		Ok(events) => events,
		Err(error) => {
			failures.push(format!("{}: error response decode failed: {error}", path.display()));
			return;
		},
	};
	let error = events.iter().find_map(|event| match event {
		TurnEvent::Error(error) => Some(error),
		_ => None,
	});
	let Some(error) = error else {
		failures.push(format!("{}: response did not produce TurnEvent::Error", path.display()));
		return;
	};
	let expected_class = fixture["expect"]["error_kind"].as_str().unwrap_or_default();
	let expected_kind = match expected_class {
		"rate_limit" => TurnErrorKind::RateLimited,
		"invalid_parameter" | "context_overflow" => TurnErrorKind::Upstream,
		_ => TurnErrorKind::Upstream,
	};
	if error.kind != expected_kind {
		failures.push(format!(
			"{}: error kind mismatch: expected {expected_kind:?} for `{expected_class}`, got {:?}",
			path.display(),
			error.kind
		));
	}
	if expected_class == "invalid_parameter" {
		let parameter = fixture["expect"]["parameter"].as_str().unwrap_or_default();
		if !error.detail.contains(parameter) {
			failures.push(format!(
				"{}: invalid-parameter detail does not retain `{parameter}`",
				path.display()
			));
		}
	}
	if let Some(header) = fixture["http"]["headers"]["retry-after"].as_str() {
		let actual = parse_retry_after(header, UNIX_EPOCH)
			.and_then(|duration| u64::try_from(duration.as_millis()).ok());
		let expected = fixture["expect"]["retry_after_ms"].as_u64();
		if actual != expected {
			failures.push(format!(
				"{}: Retry-After mismatch: expected {expected:?}, got {actual:?}",
				path.display()
			));
		}
	}
}

fn json_diff(path: &Path, label: &str, expected: &Value, actual: &Value) -> String {
	format!(
		"{}: {label} mismatch\nexpected: {}\nactual:   {}",
		path.display(),
		serde_json::to_string(expected).unwrap_or_else(|_| "<unserializable>".into()),
		serde_json::to_string(actual).unwrap_or_else(|_| "<unserializable>".into())
	)
}

fn event_diff(
	path: &Path,
	expected_label: &str,
	expected: &[Value],
	actual_label: &str,
	actual: &[Value],
) -> String {
	let first = expected
		.iter()
		.zip(actual)
		.position(|(left, right)| left != right)
		.unwrap_or_else(|| expected.len().min(actual.len()));
	let expected_event = expected
		.get(first)
		.map_or_else(|| "<end>".into(), Value::to_string);
	let actual_event = actual
		.get(first)
		.map_or_else(|| "<end>".into(), Value::to_string);
	format!(
		"{}: event mismatch at index {first}\n{expected_label} ({}/{}): \
		 {expected_event}\n{actual_label} ({}/{}): {actual_event}",
		path.display(),
		expected.len(),
		first + usize::from(first < expected.len()),
		actual.len(),
		first + usize::from(first < actual.len())
	)
}
