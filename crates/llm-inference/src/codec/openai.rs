//! Comprehensive OpenAI-compatible codec composition without provider-local
//! operation loops.

use crate::{
	call::OperationCall,
	catalog::OperationKind,
	codec::{
		Codec, DecodeContext, DecoderState, EncodeContext, EncodedRequest, RealtimeWireCodecState,
		openai_chat::OpenAiChatCodec, openai_media::OpenAiMediaCodec,
		openai_realtime::OpenAiRealtimeWireCodec,
	},
	error::Error,
};

/// One operation-dispatching codec for routes using the OpenAI-compatible
/// protocol family.
#[derive(Clone, Debug, Default)]
pub struct OpenAiCodec {
	chat:  OpenAiChatCodec,
	media: OpenAiMediaCodec,
}

impl Codec for OpenAiCodec {
	fn encode(
		&self,
		context: &EncodeContext<'_>,
		operation: &OperationCall,
	) -> Result<EncodedRequest, Error> {
		match operation {
			OperationCall::Chat(_) => self.chat.encode(context, operation),
			OperationCall::GenerateImage(_)
			| OperationCall::Speak(_)
			| OperationCall::Transcribe(_) => self.media.encode(context, operation),
			OperationCall::Realtime(_) => Err(crate::error::Error::new(
				crate::error::ErrorKind::InternalInvariant,
				crate::error::ErrorPhase::Encoding,
				crate::error::RetryAction::Never,
				crate::receipt::ExecutionReceipt::default(),
			)),
			_ => self.chat.encode(context, operation),
		}
	}

	fn encode_realtime_handshake(
		&self,
		context: &EncodeContext<'_>,
		operation: &OperationCall,
	) -> Result<Option<EncodedRequest>, Error> {
		let OperationCall::Realtime(_) = operation else {
			return Ok(None);
		};
		let target = context
			.target
			.filter(|_| context.policy_model.is_some())
			.ok_or_else(|| {
				crate::error::Error::new(
					crate::error::ErrorKind::InvalidRequest,
					crate::error::ErrorPhase::Encoding,
					crate::error::RetryAction::Never,
					crate::receipt::ExecutionReceipt::default(),
				)
			})?;
		let maximum_frame_bytes = self.chat.maximum_frame_bytes(context.policy);
		crate::codec::openai_realtime::encode_handshake(
			target.endpoint.base_url.as_str(),
			target.wire_model.as_str(),
			maximum_frame_bytes,
		)
		.map(Some)
	}

	fn decoder(&self, context: &DecodeContext<'_>) -> Result<DecoderState, Error> {
		match context.operation {
			OperationKind::GenerateImage | OperationKind::Speak | OperationKind::Transcribe => {
				self.media.decoder(context)
			},
			OperationKind::Realtime => Err(crate::error::Error::new(
				crate::error::ErrorKind::InternalInvariant,
				crate::error::ErrorPhase::Encoding,
				crate::error::RetryAction::Never,
				crate::receipt::ExecutionReceipt::default(),
			)),
			_ => self.chat.decoder(context),
		}
	}

	fn realtime(
		&self,
		context: &DecodeContext<'_>,
	) -> Result<Option<RealtimeWireCodecState>, Error> {
		let OperationCall::Realtime(request) = context.operation_call else {
			return Ok(None);
		};
		context.debug_assert_valid();
		Ok(Some(Box::new(OpenAiRealtimeWireCodec::new((**request).clone()))))
	}
}
