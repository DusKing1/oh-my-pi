//! Typed sans-I/O OpenAI Realtime WebSocket protocol.

use std::collections::BTreeMap;

use bytes::Bytes;
use omp_core::{Str, encoding::base64};
use serde::{Deserialize, Serialize};

use crate::{
	answer::{AudioChunk, RealtimeEvent, RealtimeInput},
	call::{
		AudioFormat, OpaqueJson, RealtimeEagerness, RealtimeModality, RealtimeRequest, Setting,
		ToolDefinition, ToolResultContent, TurnDetection,
	},
	catalog::OperationKind,
	codec::{
		EncodedRequest, RealtimeEvents, RealtimeWireCodec, RealtimeWireFrames, RequestHeader,
		RequestMethod, SizeBounds,
	},
	error::{Error, ErrorKind, ErrorPhase, RetryAction},
	receipt::ExecutionReceipt,
	transport::FramingProtocol,
};

/// Encodes the credential-free WebSocket upgrade request for a realtime
/// session.
pub fn encode_handshake(
	base_url: &str,
	wire_model: &str,
	maximum_frame_bytes: u64,
) -> Result<EncodedRequest, Error> {
	let mut uri = url::Url::parse(base_url).map_err(|_| capability_error())?;
	uri.set_path("/v1/realtime");
	uri.set_query(None);
	uri.query_pairs_mut().append_pair("model", wire_model);
	Ok(EncodedRequest::new(
		OperationKind::Realtime,
		RequestMethod::Get,
		Str::from(uri.to_string()),
		vec![RequestHeader { name: Str::from("openai-beta"), value: Str::from("realtime=v1") }]
			.into_boxed_slice(),
		crate::body::BodySource::Bytes(Bytes::new()),
		FramingProtocol::WebSocket,
		SizeBounds { request_body: 0, frame: maximum_frame_bytes, response: u64::MAX },
	))
}

/// OpenAI Realtime codec state initialized from exact canonical request intent.
pub struct OpenAiRealtimeWireCodec {
	request:         RealtimeRequest,
	calls:           BTreeMap<Str, PendingTool>,
	next_call_index: u32,
}

struct PendingTool {
	index:     u32,
	name:      Str,
	arguments: String,
}

impl OpenAiRealtimeWireCodec {
	/// Constructs wire state without opening a connection.
	pub fn new(request: RealtimeRequest) -> Self {
		Self { request, calls: BTreeMap::new(), next_call_index: 0 }
	}
}

impl RealtimeWireCodec for OpenAiRealtimeWireCodec {
	fn initial_frames(&mut self) -> Result<RealtimeWireFrames, Error> {
		let frame = SessionUpdateFrame {
			kind:    SessionUpdateKind::SessionUpdate,
			session: lower_session(&self.request)?,
		};
		one_json(&frame)
	}

	fn encode(&mut self, input: RealtimeInput) -> Result<RealtimeWireFrames, Error> {
		match input {
			RealtimeInput::Audio(audio) => one_json(&AudioAppendFrame {
				kind:  AudioAppendKind::InputAudioBufferAppend,
				audio: base64::encode(&audio).into_string(),
			}),
			RealtimeInput::Text(text) => one_json(&ConversationItemCreateFrame {
				kind: ConversationItemCreateKind::ConversationItemCreate,
				item: RealtimeItem::Message {
					role:    "user",
					content: vec![RealtimeContent::InputText { text }],
				},
			}),
			RealtimeInput::ToolResult { call, content, is_error, .. } => {
				let output = encode_tool_content(&content, is_error)?;
				one_json(&ConversationItemCreateFrame {
					kind: ConversationItemCreateKind::ConversationItemCreate,
					item: RealtimeItem::FunctionCallOutput { call_id: call.as_str(), output },
				})
			},
			RealtimeInput::Commit => {
				let mut frames = RealtimeWireFrames::new();
				frames.push(json_bytes(&TypeOnly { kind: TypeOnlyKind::InputAudioBufferCommit })?);
				frames.push(json_bytes(&TypeOnly { kind: TypeOnlyKind::ResponseCreate })?);
				Ok(frames)
			},
			RealtimeInput::CancelResponse => {
				one_json(&TypeOnly { kind: TypeOnlyKind::ResponseCancel })
			},
			RealtimeInput::Close => Ok(RealtimeWireFrames::new()),
		}
	}

	fn decode(&mut self, payload: Bytes) -> Result<RealtimeEvents, Error> {
		let event: ServerEvent = serde_json::from_slice(&payload).map_err(|_| protocol_error())?;
		let mut events = RealtimeEvents::new();
		match event {
			ServerEvent::SessionCreated
			| ServerEvent::SessionUpdated
			| ServerEvent::ResponseCreated
			| ServerEvent::ContentPartDone
			| ServerEvent::OutputItemDone
			| ServerEvent::RateLimitsUpdated
			| ServerEvent::ConversationItemCreated
			| ServerEvent::SpeechStarted
			| ServerEvent::SpeechStopped
			| ServerEvent::TextDone
			| ServerEvent::AudioTranscriptDone => {},
			ServerEvent::InputAudioCommitted => events.push(RealtimeEvent::InputCommitted),
			ServerEvent::AudioDelta { delta } => {
				let bytes = base64::decode(delta.as_bytes())
					.into_vec()
					.map(Bytes::from)
					.map_err(|_| protocol_error())?;
				events.push(RealtimeEvent::Audio(AudioChunk {
					bytes,
					start_ms: None,
					end_ms: None,
					final_chunk: false,
				}));
			},
			ServerEvent::AudioDone => events.push(RealtimeEvent::Audio(AudioChunk {
				bytes:       Bytes::new(),
				start_ms:    None,
				end_ms:      None,
				final_chunk: true,
			})),
			ServerEvent::TextDelta { item_id, output_index, delta }
			| ServerEvent::AudioTranscriptDelta { item_id, output_index, delta } => {
				let index = output_index.unwrap_or_else(|| stable_index(item_id.as_str()));
				self.next_call_index = self.next_call_index.max(index.saturating_add(1));
				events.push(RealtimeEvent::Chat(crate::event::ChatEvent::TextDelta {
					index,
					text: delta,
				}));
			},
			ServerEvent::ContentPartAdded {
				item_id,
				output_index,
				part: RealtimeContentPart::Text,
			} => {
				let index = output_index.unwrap_or_else(|| stable_index(item_id.as_str()));
				self.next_call_index = self.next_call_index.max(index.saturating_add(1));
				events.push(RealtimeEvent::Chat(crate::event::ChatEvent::BlockStarted {
					index,
					kind: crate::event::BlockKind::Text,
				}));
			},
			ServerEvent::ContentPartAdded { part: RealtimeContentPart::Audio, .. } => {},
			ServerEvent::OutputItemAdded {
				output_index,
				item: RealtimeOutputItem::FunctionCall { call_id, name },
			} => {
				let index = output_index.unwrap_or(self.next_call_index);
				self.calls.insert(call_id.clone(), PendingTool {
					index,
					name: name.clone(),
					arguments: String::new(),
				});
				self.next_call_index = self.next_call_index.max(index.saturating_add(1));
				events.push(RealtimeEvent::Chat(crate::event::ChatEvent::BlockStarted {
					index,
					kind: crate::event::BlockKind::ToolCall,
				}));
				events.push(RealtimeEvent::Chat(crate::event::ChatEvent::ToolCallStarted {
					index,
					id: crate::id::ToolCallId::from(call_id),
					name,
				}));
			},
			ServerEvent::OutputItemAdded { .. } => {},
			ServerEvent::FunctionCallDelta { call_id, delta } => {
				let call = self.calls.get_mut(&call_id).ok_or_else(protocol_error)?;
				call.arguments.push_str(delta.as_str());
				events.push(RealtimeEvent::Chat(crate::event::ChatEvent::ToolArgumentsDelta {
					index: call.index,
					bytes: Bytes::copy_from_slice(delta.as_bytes()),
				}));
			},
			ServerEvent::FunctionCallDone { call_id, name, arguments } => {
				let mut call = match self.calls.remove(&call_id) {
					Some(call) => call,
					None => {
						let name = name.ok_or_else(protocol_error)?;
						let index = self.next_call_index;
						self.next_call_index = self.next_call_index.saturating_add(1);
						PendingTool { index, name, arguments: String::new() }
					},
				};
				if !arguments.is_empty() {
					call.arguments = arguments.to_string();
				}
				let value = serde_json::from_str(&call.arguments).map_err(|_| protocol_error())?;
				events.push(RealtimeEvent::Chat(crate::event::ChatEvent::ToolCallReady {
					index: call.index,
					call:  crate::event::ToolCall {
						id:        crate::id::ToolCallId::from(call_id),
						name:      call.name,
						arguments: OpaqueJson::new(value),
					},
				}));
			},
			ServerEvent::ResponseDone => events.push(RealtimeEvent::Chat(
				crate::event::ChatEvent::Completed(crate::event::Completion {
					reason:  crate::event::FinishReason::Stop,
					blocks:  self.next_call_index,
					usage:   crate::receipt::Usage::default(),
					receipt: ExecutionReceipt::default(),
				}),
			)),
			ServerEvent::Error { error } => return Err(provider_error(error)),
		}
		Ok(events)
	}
}

#[derive(Serialize)]
struct SessionUpdateFrame<'a> {
	#[serde(rename = "type")]
	kind:    SessionUpdateKind,
	session: RealtimeSessionConfig<'a>,
}
#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum SessionUpdateKind {
	#[serde(rename = "session.update")]
	SessionUpdate,
}

#[derive(Serialize)]
struct RealtimeSessionConfig<'a> {
	#[serde(skip_serializing_if = "Option::is_none")]
	instructions:        Option<&'a str>,
	modalities:          Vec<&'static str>,
	#[serde(skip_serializing_if = "Option::is_none")]
	voice:               Option<&'a str>,
	#[serde(skip_serializing_if = "Option::is_none")]
	input_audio_format:  Option<&'static str>,
	#[serde(skip_serializing_if = "Option::is_none")]
	output_audio_format: Option<&'static str>,
	turn_detection:      RealtimeTurnDetection,
	#[serde(skip_serializing_if = "Vec::is_empty")]
	tools:               Vec<RealtimeTool<'a>>,
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum RealtimeTurnDetection {
	#[serde(rename = "none")]
	None,
	#[serde(rename = "server_vad")]
	ServerVad { threshold: f32, silence_duration_ms: u32, prefix_padding_ms: u32 },
	#[serde(rename = "semantic_vad")]
	SemanticVad { eagerness: &'static str },
}
#[derive(Serialize)]
struct RealtimeTool<'a> {
	#[serde(rename = "type")]
	kind:        &'static str,
	name:        &'a str,
	#[serde(skip_serializing_if = "Option::is_none")]
	description: Option<&'a str>,
	parameters:  &'a serde_json::Value,
}

fn lower_session(request: &RealtimeRequest) -> Result<RealtimeSessionConfig<'_>, Error> {
	let input_audio_format = setting(&request.input_audio)
		.map(audio_format)
		.transpose()?;
	let output_audio_format = setting(&request.output_audio)
		.map(audio_format)
		.transpose()?;
	let turn_detection = match setting_ref(&request.turn_detection) {
		None | Some(TurnDetection::Manual) => RealtimeTurnDetection::None,
		Some(TurnDetection::ServerVad { threshold, silence_ms, prefix_padding_ms }) => {
			RealtimeTurnDetection::ServerVad {
				threshold:           *threshold,
				silence_duration_ms: *silence_ms,
				prefix_padding_ms:   *prefix_padding_ms,
			}
		},
		Some(TurnDetection::SemanticVad { eagerness }) => RealtimeTurnDetection::SemanticVad {
			eagerness: match eagerness {
				RealtimeEagerness::Low => "low",
				RealtimeEagerness::Medium => "medium",
				RealtimeEagerness::High => "high",
				RealtimeEagerness::Auto => "auto",
			},
		},
	};
	let tools = request.tools.iter().map(lower_tool).collect();
	Ok(RealtimeSessionConfig {
		instructions: request.instructions.as_ref().map(Str::as_str),
		modalities: request
			.modalities
			.iter()
			.map(|modality| match modality {
				RealtimeModality::Text => "text",
				RealtimeModality::Audio => "audio",
			})
			.collect(),
		voice: request.voice.as_ref().map(Str::as_str),
		input_audio_format,
		output_audio_format,
		turn_detection,
		tools,
	})
}
fn lower_tool(tool: &ToolDefinition) -> RealtimeTool<'_> {
	RealtimeTool {
		kind:        "function",
		name:        tool.name.as_str(),
		description: tool.description.as_ref().map(Str::as_str),
		parameters:  tool.parameters.as_value(),
	}
}
fn audio_format(format: AudioFormat) -> Result<&'static str, Error> {
	match format {
		AudioFormat::Pcm16 => Ok("pcm16"),
		AudioFormat::Pcm24
		| AudioFormat::F32
		| AudioFormat::Mp3
		| AudioFormat::Aac
		| AudioFormat::Opus
		| AudioFormat::Flac
		| AudioFormat::Wav => Err(capability_error()),
	}
}

#[derive(Serialize)]
struct AudioAppendFrame {
	#[serde(rename = "type")]
	kind:  AudioAppendKind,
	audio: String,
}
#[derive(Serialize)]
enum AudioAppendKind {
	#[serde(rename = "input_audio_buffer.append")]
	InputAudioBufferAppend,
}
#[derive(Serialize)]
struct ConversationItemCreateFrame<'a> {
	#[serde(rename = "type")]
	kind: ConversationItemCreateKind,
	item: RealtimeItem<'a>,
}
#[derive(Serialize)]
enum ConversationItemCreateKind {
	#[serde(rename = "conversation.item.create")]
	ConversationItemCreate,
}
#[derive(Serialize)]
#[serde(tag = "type")]
enum RealtimeItem<'a> {
	#[serde(rename = "message")]
	Message { role: &'static str, content: Vec<RealtimeContent> },
	#[serde(rename = "function_call_output")]
	FunctionCallOutput { call_id: &'a str, output: Str },
}
#[derive(Serialize)]
#[serde(tag = "type")]
enum RealtimeContent {
	#[serde(rename = "input_text")]
	InputText { text: Str },
}
#[derive(Serialize)]
struct TypeOnly {
	#[serde(rename = "type")]
	kind: TypeOnlyKind,
}
#[derive(Serialize)]
enum TypeOnlyKind {
	#[serde(rename = "input_audio_buffer.commit")]
	InputAudioBufferCommit,
	#[serde(rename = "response.create")]
	ResponseCreate,
	#[serde(rename = "response.cancel")]
	ResponseCancel,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum ServerEvent {
	#[serde(rename = "session.created")]
	SessionCreated,
	#[serde(rename = "session.updated")]
	SessionUpdated,
	#[serde(rename = "input_audio_buffer.committed")]
	InputAudioCommitted,
	#[serde(rename = "input_audio_buffer.speech_started")]
	SpeechStarted,
	#[serde(rename = "input_audio_buffer.speech_stopped")]
	SpeechStopped,
	#[serde(rename = "conversation.item.created")]
	ConversationItemCreated,
	#[serde(rename = "response.created")]
	ResponseCreated,
	#[serde(rename = "response.output_item.added")]
	OutputItemAdded {
		#[serde(default)]
		output_index: Option<u32>,
		item:         RealtimeOutputItem,
	},
	#[serde(rename = "response.output_item.done")]
	OutputItemDone,
	#[serde(rename = "response.content_part.added")]
	ContentPartAdded {
		item_id:      Str,
		#[serde(default)]
		output_index: Option<u32>,
		part:         RealtimeContentPart,
	},
	#[serde(rename = "response.content_part.done")]
	ContentPartDone,
	#[serde(rename = "response.text.delta", alias = "response.output_text.delta")]
	TextDelta {
		item_id:      Str,
		#[serde(default)]
		output_index: Option<u32>,
		delta:        Str,
	},
	#[serde(rename = "response.text.done", alias = "response.output_text.done")]
	TextDone,
	#[serde(rename = "response.audio.delta", alias = "response.output_audio.delta")]
	AudioDelta { delta: Str },
	#[serde(rename = "response.audio.done", alias = "response.output_audio.done")]
	AudioDone,
	#[serde(
		rename = "response.audio_transcript.delta",
		alias = "response.output_audio_transcript.delta"
	)]
	AudioTranscriptDelta {
		item_id:      Str,
		#[serde(default)]
		output_index: Option<u32>,
		delta:        Str,
	},
	#[serde(
		rename = "response.audio_transcript.done",
		alias = "response.output_audio_transcript.done"
	)]
	AudioTranscriptDone,
	#[serde(rename = "response.function_call_arguments.delta")]
	FunctionCallDelta { call_id: Str, delta: Str },
	#[serde(rename = "response.function_call_arguments.done")]
	FunctionCallDone {
		call_id:   Str,
		#[serde(default)]
		name:      Option<Str>,
		arguments: Str,
	},
	#[serde(rename = "rate_limits.updated")]
	RateLimitsUpdated,
	#[serde(rename = "response.done")]
	ResponseDone,
	#[serde(rename = "error")]
	Error { error: RealtimeWireError },
}
#[derive(Deserialize)]
#[serde(tag = "type")]
enum RealtimeOutputItem {
	#[serde(rename = "function_call")]
	FunctionCall { call_id: Str, name: Str },
	#[serde(rename = "message")]
	Message,
}
#[derive(Deserialize)]
#[serde(tag = "type")]
enum RealtimeContentPart {
	#[serde(rename = "text", alias = "output_text")]
	Text,
	#[serde(rename = "audio", alias = "output_audio")]
	Audio,
}
#[derive(Deserialize)]
struct RealtimeWireError {
	#[serde(default)]
	code:    Option<Str>,
	#[serde(default)]
	message: Option<Str>,
}

fn encode_tool_content(content: &[ToolResultContent], is_error: bool) -> Result<Str, Error> {
	#[derive(Serialize)]
	struct ToolOutput<'a> {
		is_error: bool,
		content:  &'a [ToolResultContentWire],
	}
	#[derive(Serialize)]
	#[serde(tag = "type")]
	enum ToolResultContentWire {
		#[serde(rename = "text")]
		Text { text: Str },
		#[serde(rename = "json")]
		Json { value: serde_json::Value },
	}
	let mut wire = Vec::with_capacity(content.len());
	for item in content {
		wire.push(match item {
			ToolResultContent::Text(text) => ToolResultContentWire::Text { text: text.clone() },
			ToolResultContent::Json(value) => {
				ToolResultContentWire::Json { value: value.as_value().clone() }
			},
			ToolResultContent::Image(_) | ToolResultContent::Document(_) => {
				return Err(capability_error());
			},
		});
	}
	serde_json::to_string(&ToolOutput { is_error, content: &wire })
		.map(Str::from)
		.map_err(|_| protocol_error())
}
fn one_json(value: &impl Serialize) -> Result<RealtimeWireFrames, Error> {
	let mut frames = RealtimeWireFrames::new();
	frames.push(json_bytes(value)?);
	Ok(frames)
}
fn json_bytes(value: &impl Serialize) -> Result<Bytes, Error> {
	serde_json::to_vec(value)
		.map(Bytes::from)
		.map_err(|_| protocol_error())
}
fn stable_index(value: &str) -> u32 {
	value
		.bytes()
		.fold(2_166_136_261_u32, |hash, byte| (hash ^ u32::from(byte)).wrapping_mul(16_777_619))
}
fn setting<T: Copy>(setting: &Setting<T>) -> Option<T> {
	match setting {
		Setting::Unset => None,
		Setting::Require(value) | Setting::Prefer(value) => Some(*value),
	}
}
fn setting_ref<T>(setting: &Setting<T>) -> Option<&T> {
	match setting {
		Setting::Unset => None,
		Setting::Require(value) | Setting::Prefer(value) => Some(value),
	}
}
fn provider_error(error: RealtimeWireError) -> Error {
	let mut typed = Error::new(
		ErrorKind::Protocol,
		ErrorPhase::Streaming,
		RetryAction::Never,
		ExecutionReceipt::default(),
	);
	typed.code = error.code.or(error.message);
	typed
}
fn capability_error() -> Error {
	Error::new(
		ErrorKind::CapabilityMismatch,
		ErrorPhase::Encoding,
		RetryAction::Never,
		ExecutionReceipt::default(),
	)
}
fn protocol_error() -> Error {
	Error::new(
		ErrorKind::Protocol,
		ErrorPhase::Streaming,
		RetryAction::Never,
		ExecutionReceipt::default(),
	)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::event::{BlockKind, ChatEvent};

	fn codec() -> OpenAiRealtimeWireCodec {
		OpenAiRealtimeWireCodec::new(RealtimeRequest {
			instructions:   None,
			modalities:     std::sync::Arc::from([RealtimeModality::Text]),
			voice:          None,
			input_audio:    Setting::Unset,
			output_audio:   Setting::Unset,
			turn_detection: Setting::Unset,
			tools:          std::sync::Arc::from([]),
			negotiation:    crate::call::NegotiationPolicy::default(),
		})
	}

	#[test]
	fn normal_response_sequence_preserves_explicit_noops() {
		let mut codec = codec();
		assert!(
			codec
				.decode(Bytes::from_static(br#"{"type":"session.created"}"#))
				.unwrap()
				.is_empty()
		);
		assert!(
			codec
				.decode(Bytes::from_static(br#"{"type":"response.created"}"#))
				.unwrap()
				.is_empty()
		);
		assert!(codec.decode(Bytes::from_static(br#"{"type":"response.output_item.added","output_index":0,"item":{"type":"message"}}"#)).unwrap().is_empty());
		let events = codec.decode(Bytes::from_static(br#"{"type":"response.content_part.added","item_id":"item_1","output_index":0,"part":{"type":"text"}}"#)).unwrap();
		assert!(matches!(events.as_slice(), [RealtimeEvent::Chat(ChatEvent::BlockStarted {
			index: 0,
			kind:  BlockKind::Text,
		})]));
		let events = codec
			.decode(Bytes::from_static(
				br#"{"type":"response.text.delta","item_id":"item_1","output_index":0,"delta":"hi"}"#,
			))
			.unwrap();
		assert!(
			matches!(events.as_slice(), [RealtimeEvent::Chat(ChatEvent::TextDelta { index: 0, text })] if text.as_str() == "hi")
		);
		assert!(
			codec
				.decode(Bytes::from_static(br#"{"type":"response.content_part.done"}"#))
				.unwrap()
				.is_empty()
		);
		assert!(
			codec
				.decode(Bytes::from_static(br#"{"type":"rate_limits.updated","rate_limits":[]}"#))
				.unwrap()
				.is_empty()
		);
		assert!(matches!(
			codec
				.decode(Bytes::from_static(br#"{"type":"response.done"}"#))
				.unwrap()
				.as_slice(),
			[RealtimeEvent::Chat(ChatEvent::Completed(_))]
		));
	}

	#[test]
	fn function_item_emits_ordered_block_and_call_start() {
		let mut codec = codec();
		let events = codec.decode(Bytes::from_static(br#"{"type":"response.output_item.added","output_index":2,"item":{"type":"function_call","call_id":"call_1","name":"lookup"}}"#)).unwrap();
		assert!(matches!(events.as_slice(),
			[RealtimeEvent::Chat(ChatEvent::BlockStarted { index: 2, kind: BlockKind::ToolCall }),
			 RealtimeEvent::Chat(ChatEvent::ToolCallStarted { index: 2, id, name })]
			if id.as_str() == "call_1" && name.as_str() == "lookup"));
	}

	#[test]
	fn unknown_or_malformed_server_event_is_terminal_error() {
		let mut codec = codec();
		assert!(
			codec
				.decode(Bytes::from_static(br#"{"type":"response.future_event"}"#))
				.is_err()
		);
		assert!(
			codec
				.decode(Bytes::from_static(br#"{"type":"response.text.delta","delta":7}"#))
				.is_err()
		);
	}
}
