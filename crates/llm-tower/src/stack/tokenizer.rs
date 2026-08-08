//! Catalog-selected prompt token counting for OpenAI model families.
//!
//! Rank selection follows OpenAI's published `tiktoken` model table. Catalog
//! metadata may pin an encoding with `openai/tokenizer`; otherwise the
//! provider-local model id selects the ranks. Selection is deliberately gated
//! to catalog cards classified as OpenAI or GPT-OSS so an OpenAI-compatible
//! transport cannot accidentally apply these ranks to Anthropic or Gemini.

use omp_core::SmolStr;
use omp_llm_catalog::models::ModelCard;
use omp_llm_types::{
	CountInput, CountRequest, ImageDetail, ItemKind, Part, Role, Thread, ToolDef, facet::Error,
};
use serde_json::Value;
use tiktoken_rs::{
	CoreBPE, cl100k_base_singleton, o200k_base_singleton, o200k_harmony_singleton,
	p50k_base_singleton, r50k_base_singleton,
	tokenizer::{Tokenizer as RankTokenizer, get_tokenizer},
};

const MESSAGE_OVERHEAD: u64 = 3;
const NAME_OVERHEAD: u64 = 1;
const FUNCTION_CALL_OVERHEAD: u64 = 1;
const REPLY_PRIMING: u64 = 3;
const PROPERTY_INITIALIZATION: u64 = 3;
const PROPERTY_KEY_OVERHEAD: u64 = 3;
const ENUM_INITIALIZATION_DISCOUNT: u64 = 3;
const ENUM_ITEM_OVERHEAD: u64 = 3;
const FUNCTION_END: u64 = 12;
const IMAGE_BASE_TOKENS: u64 = 85;
const IMAGE_TILE_TOKENS: u64 = 170;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PromptFraming {
	Modern,
	Gpt35Turbo0301,
}

/// An exact local counter for one projected prompt representation.
pub trait Tokenizer: Send + Sync + 'static {
	/// Counts the fully projected request.
	fn count(&self, request: &CountRequest) -> Result<u64, Error>;
}

/// OpenAI tokenizer ranks and prompt-framing rules selected for one model card.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OpenAiTokenizer {
	ranks:   RankTokenizer,
	framing: PromptFraming,
}

impl OpenAiTokenizer {
	/// Selects published ranks from a catalog card.
	///
	/// `openai/tokenizer` accepts `o200k_base`, `o200k_harmony`, `cl100k_base`,
	/// `p50k_base`, or `r50k_base`. An unrecognized explicit value disables an
	/// exact local count rather than silently choosing different ranks.
	#[must_use]
	pub fn for_model(model: &ModelCard) -> Option<Self> {
		let family = model.family.as_str();
		if !matches!(family, "openai" | "gpt-oss")
			&& !(family.is_empty() && matches!(model.provider.as_str(), "openai" | "openai-codex"))
		{
			return None;
		}
		let model_id = model
			.model
			.rsplit('/')
			.next()
			.unwrap_or(model.model.as_str());
		let ranks = if let Some(name) = model
			.props
			.get_ns("openai", "tokenizer")
			.and_then(Value::as_str)
		{
			ranks_from_name(name)?
		} else {
			get_tokenizer(model_id)?
		};
		Self::from_ranks(ranks, model_id)
	}

	fn from_ranks(ranks: RankTokenizer, model_id: &str) -> Option<Self> {
		matches!(
			ranks,
			RankTokenizer::O200kHarmony
				| RankTokenizer::O200kBase
				| RankTokenizer::Cl100kBase
				| RankTokenizer::P50kBase
				| RankTokenizer::R50kBase
		)
		.then_some(Self {
			ranks,
			framing: if model_id == "gpt-3.5-turbo-0301" {
				PromptFraming::Gpt35Turbo0301
			} else {
				PromptFraming::Modern
			},
		})
	}

	fn bpe(self) -> &'static CoreBPE {
		match self.ranks {
			RankTokenizer::O200kHarmony => o200k_harmony_singleton(),
			RankTokenizer::O200kBase => o200k_base_singleton(),
			RankTokenizer::Cl100kBase => cl100k_base_singleton(),
			RankTokenizer::P50kBase | RankTokenizer::P50kEdit => p50k_base_singleton(),
			RankTokenizer::R50kBase | RankTokenizer::Gpt2 => r50k_base_singleton(),
		}
	}

	fn token_count(self, text: &str) -> u64 {
		u64::try_from(self.bpe().encode_ordinary(text).len()).unwrap_or(u64::MAX)
	}

	fn function_initialization(self) -> u64 {
		if self.ranks == RankTokenizer::Cl100kBase {
			10
		} else {
			7
		}
	}

	fn message_overhead(self) -> u64 {
		match self.framing {
			PromptFraming::Modern => MESSAGE_OVERHEAD,
			PromptFraming::Gpt35Turbo0301 => 4,
		}
	}

	fn add_name_overhead(self, tokens: u64) -> u64 {
		match self.framing {
			PromptFraming::Modern => tokens.saturating_add(NAME_OVERHEAD),
			PromptFraming::Gpt35Turbo0301 => tokens.saturating_sub(1),
		}
	}

	fn count_thread(self, thread: &Thread) -> Result<u64, Error> {
		let mut tokens = REPLY_PRIMING;
		let mut assistant_message_open = false;
		for item in &thread.items {
			match &item.kind {
				ItemKind::Message(message) => {
					tokens = tokens
						.saturating_add(self.message_overhead())
						.saturating_add(self.token_count(role_name(message.role)?));
					for part in &message.parts {
						tokens = tokens.saturating_add(self.count_part(part)?);
					}
					assistant_message_open = message.role == Role::Assistant;
				},
				ItemKind::ToolCall(call) => {
					if !assistant_message_open {
						tokens = tokens
							.saturating_add(self.message_overhead())
							.saturating_add(self.token_count("assistant"));
					}
					let arguments = std::str::from_utf8(&call.args_json)
						.map_err(|_| Error::Provider(SmolStr::from("tool arguments are not UTF-8")))?;
					tokens = tokens
						.saturating_add(self.token_count(call.name.as_str()))
						.saturating_add(self.token_count(arguments))
						.saturating_add(FUNCTION_CALL_OVERHEAD);
					assistant_message_open = true;
				},
				ItemKind::ToolResult(result) => {
					tokens = self.add_name_overhead(
						tokens
							.saturating_add(self.message_overhead())
							.saturating_add(self.token_count("tool"))
							.saturating_add(self.token_count(result.name.as_str())),
					);
					for part in &result.parts {
						tokens = tokens.saturating_add(self.count_part(part)?);
					}
					assistant_message_open = false;
				},
				_ => {},
			}
		}
		Ok(tokens)
	}

	fn count_part(self, part: &Part) -> Result<u64, Error> {
		match part {
			Part::Text(text) => Ok(self.token_count(text.as_str())),
			Part::Thinking(thinking) => Ok(self.token_count(thinking.text.as_str())),
			Part::Blob(blob) => Ok(match blob.detail.unwrap_or(ImageDetail::Auto) {
				ImageDetail::Auto | ImageDetail::Low => IMAGE_BASE_TOKENS,
				ImageDetail::High | ImageDetail::Original => {
					IMAGE_BASE_TOKENS.saturating_add(IMAGE_TILE_TOKENS)
				},
				_ => IMAGE_BASE_TOKENS,
			}),
			Part::Fallback(fallback) => Ok(self
				.token_count(fallback.from_model.as_str())
				.saturating_add(self.token_count(fallback.to_model.as_str()))),
			Part::ServerTool(tool) => {
				let payload = std::str::from_utf8(&tool.payload_json)
					.map_err(|_| Error::Provider(SmolStr::from("server tool payload is not UTF-8")))?;
				Ok(self
					.token_count(tool.name.as_str())
					.saturating_add(self.token_count(payload))
					.saturating_add(FUNCTION_CALL_OVERHEAD))
			},
			_ => Ok(0),
		}
	}

	fn count_tools(self, tools: &[ToolDef]) -> Result<u64, Error> {
		if tools.is_empty() {
			return Ok(0);
		}
		let mut tokens = 0u64;
		let mut line = String::new();
		for tool in tools {
			let schema: Value = serde_json::from_slice(&tool.schema_json).map_err(|error| {
				Error::Provider(SmolStr::new(format!("invalid tool JSON Schema: {error}")))
			})?;
			tokens = tokens.saturating_add(self.function_initialization());
			line.clear();
			line.push_str(tool.name.as_str());
			line.push_str(":");
			line.push_str(tool.description.trim_end_matches('.'));
			tokens = tokens.saturating_add(self.token_count(line.as_str()));

			if let Some(properties) = schema.get("properties").and_then(Value::as_object)
				&& !properties.is_empty()
			{
				tokens = tokens.saturating_add(PROPERTY_INITIALIZATION);
				for (name, property) in properties {
					tokens = tokens.saturating_add(PROPERTY_KEY_OVERHEAD);
					if let Some(values) = property.get("enum").and_then(Value::as_array) {
						tokens = tokens.saturating_sub(ENUM_INITIALIZATION_DISCOUNT);
						for value in values.iter().filter_map(Value::as_str) {
							tokens = tokens
								.saturating_add(ENUM_ITEM_OVERHEAD)
								.saturating_add(self.token_count(value));
						}
					}
					line.clear();
					line.push_str(name);
					line.push_str(":");
					if let Some(kind) = property.get("type").and_then(Value::as_str) {
						line.push_str(kind);
					}
					line.push_str(":");
					if let Some(description) = property.get("description").and_then(Value::as_str) {
						line.push_str(description.trim_end_matches('.'));
					}
					tokens = tokens.saturating_add(self.token_count(line.as_str()));
				}
			}
		}
		Ok(tokens.saturating_add(FUNCTION_END))
	}
}

impl Tokenizer for OpenAiTokenizer {
	fn count(&self, request: &CountRequest) -> Result<u64, Error> {
		let CountInput::Thread(thread) = &request.input else {
			return Err(Error::Provider(SmolStr::from(
				"local token counting requires a resolved inline thread",
			)));
		};
		self.count_thread(thread).and_then(|thread_tokens| {
			self
				.count_tools(&request.tools)
				.map(|tools| thread_tokens.saturating_add(tools))
		})
	}
}

fn ranks_from_name(name: &str) -> Option<RankTokenizer> {
	match name {
		"o200k_base" => Some(RankTokenizer::O200kBase),
		"o200k_harmony" => Some(RankTokenizer::O200kHarmony),
		"cl100k_base" => Some(RankTokenizer::Cl100kBase),
		"p50k_base" => Some(RankTokenizer::P50kBase),
		"r50k_base" | "gpt2" => Some(RankTokenizer::R50kBase),
		_ => None,
	}
}

fn role_name(role: Role) -> Result<&'static str, Error> {
	match role {
		Role::System => Ok("system"),
		Role::User => Ok("user"),
		Role::Assistant => Ok("assistant"),
		_ => Err(Error::Provider(SmolStr::from("unsupported message role"))),
	}
}

#[cfg(test)]
mod tests {
	use bytes::Bytes;
	use omp_llm_types::{BlobPart, Item, Message, Props};

	use super::*;

	fn fixture_request(items: Vec<Item>, tools: Vec<ToolDef>) -> CountRequest {
		CountRequest::builder()
			.model(SmolStr::from("fixture/model"))
			.input(CountInput::Thread(Thread::builder().items(items).build()))
			.tools(tools)
			.build()
	}

	fn message(role: Role, text: &str) -> Item {
		Item::builder()
			.seq(0)
			.kind(ItemKind::Message(
				Message::builder()
					.role(role)
					.parts(vec![Part::Text(SmolStr::new(text))])
					.build(),
			))
			.props(Props::default())
			.build()
	}

	fn tokenizer(ranks: RankTokenizer) -> OpenAiTokenizer {
		OpenAiTokenizer::from_ranks(ranks, "fixture").expect("supported fixture ranks")
	}

	#[test]
	fn published_rank_fixtures_cover_openai_and_codex_encodings() {
		let fixtures = [
			("antidisestablishmentarianism", 5, 6, 6),
			("2 + 2 = 4", 5, 7, 7),
			("お誕生日おめでとう", 14, 9, 8),
		];
		let codex = tokenizer(RankTokenizer::P50kBase);
		let cl100k = tokenizer(RankTokenizer::Cl100kBase);
		let o200k = tokenizer(RankTokenizer::O200kBase);
		for (text, codex_tokens, cl100k_tokens, o200k_tokens) in fixtures {
			assert_eq!(codex.token_count(text), codex_tokens, "Codex fixture: {text}");
			assert_eq!(cl100k.token_count(text), cl100k_tokens, "cl100k fixture: {text}");
			assert_eq!(o200k.token_count(text), o200k_tokens, "o200k fixture: {text}");
		}
	}

	#[test]
	fn published_tool_schema_fixture_includes_prompt_framing() {
		let items = vec![
			message(
				Role::System,
				"You are a helpful assistant that can answer to questions about the weather.",
			),
			message(Role::User, "What's the weather like in San Francisco?"),
		];
		let tool = ToolDef::builder()
			.name(SmolStr::from("get_current_weather"))
			.description(SmolStr::from(
				"Get the current weather in a given location",
			))
			.schema_json(Bytes::from_static(
				br#"{"type":"object","properties":{"location":{"type":"string","description":"The city and state, e.g. San Francisco, CA"},"unit":{"type":"string","description":"The unit of temperature to return","enum":["celsius","fahrenheit"]}},"required":["location"]}"#,
			))
			.build();
		let request = fixture_request(items, vec![tool]);
		assert_eq!(
			tokenizer(RankTokenizer::Cl100kBase)
				.count(&request)
				.unwrap(),
			105,
		);
		assert_eq!(tokenizer(RankTokenizer::O200kBase).count(&request).unwrap(), 101,);
	}

	#[test]
	fn image_detail_changes_the_published_tile_overhead() {
		let image_item = |detail| {
			Item::builder()
				.seq(0)
				.kind(ItemKind::Message(
					Message::builder()
						.role(Role::User)
						.parts(vec![Part::Blob(
							BlobPart::builder()
								.hash([0; 32])
								.mime(SmolStr::from("image/png"))
								.size(0)
								.inline(Bytes::new())
								.detail(detail)
								.build(),
						)])
						.build(),
				))
				.props(Props::default())
				.build()
		};
		let tokenizer = tokenizer(RankTokenizer::O200kBase);
		let low = tokenizer
			.count(&fixture_request(vec![image_item(ImageDetail::Low)], Vec::new()))
			.unwrap();
		let high = tokenizer
			.count(&fixture_request(vec![image_item(ImageDetail::High)], Vec::new()))
			.unwrap();
		assert_eq!(low, 92);
		assert_eq!(high, 262);
	}
}
