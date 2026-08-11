//! `OpenAI` chat, Responses, embeddings, and audio transport codecs.

pub mod audio;
pub mod discovery;
pub mod embeddings;
mod model_policy;
pub mod openai_chat;
pub mod openai_codex;
pub mod openai_codex_attestation;
pub mod openai_codex_responses_lite;
pub mod openai_codex_websocket;
pub mod openai_responses;
mod responses_tool_repair;

pub use audio::{EncodedAudioRequest, OpenAiAudioCodec, OpenAiAudioError, OpenAiAudioProfile};
pub use openai_chat::OpenAiChatCodec;
pub use openai_codex::{
	CodexAttestation, CodexCredentialMetadata, CodexHeaderContext, CodexHeaderPlan,
	CodexHeaderValue, CodexRequestIdentity, CodexWireTransport, OpenAiCodexCodec,
	apply_codex_client_metadata, build_codex_header_plan, resolve_codex_responses_url,
};
pub use openai_codex_attestation::{
	CodexAttestationError, CodexAttestationSignals, CodexAttestor, CodexDeviceCheckResult,
	CodexDeviceToken, build_codex_attestation,
};
pub use openai_codex_responses_lite::{
	CODEX_PROVIDER_NAMESPACE, RESPONSES_LITE_OPTION, transform_codex_request,
};
pub use openai_codex_websocket::{
	CodexContinuationState, CodexFallbackAction, CodexFrameDisposition, CodexFrameRouter,
	CodexReplaySafety, CodexWebSocketFailure, CodexWebSocketProtocolError, RESPONSE_CREATE,
	classify_codex_fallback, classify_codex_websocket_failure, codex_websocket_url,
};
pub use openai_responses::OpenAiResponsesCodec;
