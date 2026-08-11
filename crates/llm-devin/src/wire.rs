//! Hand-written `prost` wire types for Devin Cascade (Exa / Codeium API).
//!
//! Pinned source provenance:
//! `packages/ai/src/providers/devin/proto/`
//!
//! Excludes unused `buf.validate` and `cel.expr` schema types.
#![allow(
	missing_docs,
	clippy::pedantic,
	clippy::nursery,
	reason = "handwritten mirror of a frozen external wire schema"
)]

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum ChatMessageRequestType {
	Unspecified        = 0,
	General            = 1,
	ContextCheck       = 2,
	Plan               = 3,
	Command            = 4,
	Cascade            = 5,
	Eval               = 6,
	WindsurfReview     = 7,
	VibeAndReplace     = 8,
	Deepwiki           = 9,
	Devstral           = 10,
	CodemapGeneration  = 11,
	CodemapSuggestions = 12,
	SmartFriend        = 13,
	Lifeguard          = 14,
	Checkpoint         = 15,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum CacheControlType {
	Unspecified = 0,
	Ephemeral   = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum ChatMessageSource {
	Unspecified  = 0,
	User         = 1,
	System       = 2,
	Unknown      = 3,
	Tool         = 4,
	SystemPrompt = 5,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum ConversationalPlannerMode {
	Unspecified = 0,
	Default     = 1,
	ReadOnly    = 2,
	NoTool      = 3,
	Explore     = 4,
	Planning    = 5,
	Auto        = 6,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum StopReason {
	Unspecified          = 0,
	Incomplete           = 1,
	StopPattern          = 2,
	MaxTokens            = 3,
	MinLogProb           = 4,
	MaxNewlines          = 5,
	ExitScope            = 6,
	NonfiniteLogitOrProb = 7,
	FirstNonWhitespaceLine = 8,
	Partial              = 9,
	FunctionCall         = 10,
	ContentFilter        = 11,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Metadata {
	#[prost(string, tag = "1")]
	pub ide_name:          ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub extension_version: ::prost::alloc::string::String,
	#[prost(string, tag = "3")]
	pub api_key:           ::prost::alloc::string::String,
	#[prost(string, tag = "4")]
	pub locale:            ::prost::alloc::string::String,
	#[prost(string, tag = "7")]
	pub ide_version:       ::prost::alloc::string::String,
	#[prost(string, tag = "10")]
	pub session_id:        ::prost::alloc::string::String,
	#[prost(string, tag = "12")]
	pub extension_name:    ::prost::alloc::string::String,
	#[prost(string, tag = "21")]
	pub user_jwt:          ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetUserJwtRequest {
	#[prost(message, optional, tag = "1")]
	pub metadata: ::core::option::Option<Metadata>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetUserJwtResponse {
	#[prost(string, tag = "1")]
	pub user_jwt:              ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub custom_api_server_url: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CompletionConfiguration {
	#[prost(uint64, tag = "1")]
	pub num_completions:        u64,
	#[prost(uint64, tag = "2")]
	pub max_tokens:             u64,
	#[prost(uint64, tag = "3")]
	pub max_newlines:           u64,
	#[prost(double, tag = "5")]
	pub temperature:            f64,
	#[prost(double, tag = "6")]
	pub first_temperature:      f64,
	#[prost(uint64, tag = "7")]
	pub top_k:                  u64,
	#[prost(double, tag = "8")]
	pub top_p:                  f64,
	#[prost(string, repeated, tag = "9")]
	pub stop_patterns:          ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
	#[prost(double, tag = "11")]
	pub fim_eot_prob_threshold: f64,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ChatToolCall {
	#[prost(string, tag = "1")]
	pub id:             ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub name:           ::prost::alloc::string::String,
	#[prost(string, tag = "3")]
	pub arguments_json: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ImageData {
	#[prost(string, tag = "1")]
	pub base64_data: ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub mime_type:   ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ModelUsageStats {
	#[prost(uint64, tag = "2")]
	pub input_tokens:       u64,
	#[prost(uint64, tag = "3")]
	pub output_tokens:      u64,
	#[prost(uint64, tag = "4")]
	pub cache_write_tokens: u64,
	#[prost(uint64, tag = "5")]
	pub cache_read_tokens:  u64,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ChatMessagePrompt {
	#[prost(string, tag = "1")]
	pub message_id:           ::prost::alloc::string::String,
	#[prost(enumeration = "ChatMessageSource", tag = "2")]
	pub source:               i32,
	#[prost(string, tag = "3")]
	pub prompt:               ::prost::alloc::string::String,
	#[prost(message, repeated, tag = "6")]
	pub tool_calls:           ::prost::alloc::vec::Vec<ChatToolCall>,
	#[prost(string, tag = "7")]
	pub tool_call_id:         ::prost::alloc::string::String,
	#[prost(bool, tag = "9")]
	pub tool_result_is_error: bool,
	#[prost(message, repeated, tag = "10")]
	pub images:               ::prost::alloc::vec::Vec<ImageData>,
	#[prost(string, tag = "11")]
	pub thinking:             ::prost::alloc::string::String,
	#[prost(string, tag = "12")]
	pub signature:            ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PromptCacheOptions {
	#[prost(enumeration = "CacheControlType", tag = "1")]
	pub r#type: i32,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ChatToolDefinition {
	#[prost(string, tag = "1")]
	pub name:               ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub description:        ::prost::alloc::string::String,
	#[prost(string, tag = "3")]
	pub json_schema_string: ::prost::alloc::string::String,
	#[prost(bool, tag = "12")]
	pub strict:             bool,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ChatToolChoice {
	#[prost(oneof = "chat_tool_choice::Choice", tags = "1, 2")]
	pub choice: ::core::option::Option<chat_tool_choice::Choice>,
}

pub mod chat_tool_choice {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Choice {
		#[prost(string, tag = "1")]
		OptionName(::prost::alloc::string::String),
		#[prost(string, tag = "2")]
		ToolName(::prost::alloc::string::String),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetChatMessageRequest {
	#[prost(message, optional, tag = "1")]
	pub metadata: ::core::option::Option<Metadata>,
	#[prost(string, tag = "2")]
	pub prompt: ::prost::alloc::string::String,
	#[prost(message, repeated, tag = "3")]
	pub chat_message_prompts: ::prost::alloc::vec::Vec<ChatMessagePrompt>,
	#[prost(enumeration = "ChatMessageRequestType", tag = "7")]
	pub request_type: i32,
	#[prost(message, optional, tag = "8")]
	pub configuration: ::core::option::Option<CompletionConfiguration>,
	#[prost(message, repeated, tag = "10")]
	pub tools: ::prost::alloc::vec::Vec<ChatToolDefinition>,
	#[prost(bool, tag = "11")]
	pub disable_parallel_tool_calls: bool,
	#[prost(message, optional, tag = "12")]
	pub tool_choice: ::core::option::Option<ChatToolChoice>,
	#[prost(message, optional, tag = "13")]
	pub system_prompt_cache_options: ::core::option::Option<PromptCacheOptions>,
	#[prost(string, tag = "16")]
	pub cascade_id: ::prost::alloc::string::String,
	#[prost(enumeration = "ConversationalPlannerMode", tag = "20")]
	pub planner_mode: i32,
	#[prost(string, tag = "21")]
	pub chat_model_uid: ::prost::alloc::string::String,
	#[prost(string, tag = "22")]
	pub execution_id: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetChatMessageResponse {
	#[prost(string, tag = "1")]
	pub message_id:       ::prost::alloc::string::String,
	#[prost(string, tag = "3")]
	pub delta_text:       ::prost::alloc::string::String,
	#[prost(enumeration = "StopReason", tag = "5")]
	pub stop_reason:      i32,
	#[prost(message, repeated, tag = "6")]
	pub delta_tool_calls: ::prost::alloc::vec::Vec<ChatToolCall>,
	#[prost(message, optional, tag = "7")]
	pub usage:            ::core::option::Option<ModelUsageStats>,
	#[prost(string, tag = "9")]
	pub delta_thinking:   ::prost::alloc::string::String,
	#[prost(string, tag = "10")]
	pub delta_signature:  ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetCliModelConfigsRequest {
	#[prost(message, optional, tag = "1")]
	pub metadata: Option<Metadata>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetCliModelConfigsResponse {
	#[prost(message, repeated, tag = "1")]
	pub client_model_configs: Vec<ClientModelConfig>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ClientModelConfig {
	#[prost(string, tag = "1")]
	pub label:           String,
	#[prost(bool, tag = "4")]
	pub disabled:        bool,
	#[prost(bool, tag = "5")]
	pub supports_images: bool,
	#[prost(int32, tag = "18")]
	pub max_tokens:      i32,
	#[prost(string, tag = "22")]
	pub model_uid:       String,
	#[prost(message, optional, tag = "23")]
	pub model_info:      Option<ModelInfo>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ModelInfo {
	#[prost(message, optional, tag = "6")]
	pub model_features:    Option<ModelFeatures>,
	#[prost(int32, tag = "13")]
	pub max_output_tokens: i32,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ModelFeatures {
	#[prost(bool, tag = "11")]
	pub supports_images:     bool,
	#[prost(bool, tag = "12")]
	pub supports_tool_calls: bool,
	#[prost(bool, tag = "15")]
	pub supports_thinking:   bool,
}

pub mod exa {
	pub mod api_server_pb {
		pub use crate::wire::{
			ChatMessageRequestType, GetChatMessageRequest, GetChatMessageResponse,
			GetCliModelConfigsRequest, GetCliModelConfigsResponse,
		};
	}
	pub mod chat_pb {
		pub use crate::wire::{
			CacheControlType, ChatMessagePrompt, ChatToolChoice, ChatToolDefinition,
			PromptCacheOptions, chat_tool_choice,
		};
	}
	pub mod codeium_common_pb {
		pub use crate::wire::{
			ChatMessageSource, ChatToolCall, ClientModelConfig, CompletionConfiguration,
			ConversationalPlannerMode, ImageData, Metadata, ModelFeatures, ModelInfo, ModelUsageStats,
			StopReason,
		};
	}
}
