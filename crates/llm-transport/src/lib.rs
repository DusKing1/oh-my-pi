//! Provider wire transports and streaming decoders.

use std::any::Any;

use bytes::{Buf as _, Bytes};
use omp_llm_catalog::{compat::Compat, provider::TransportId};
use omp_llm_types::{ChatRequest, Error, TurnEvent, Unsupported};
use smallvec::SmallVec;

pub mod embedded;
pub mod ndjson;
pub mod normalize;
pub mod omp;
pub mod sse;

pub use normalize::with_tool_use_precedence;

/// Narrows an owned byte buffer without promoting exact-capacity vector storage
/// to shared storage.
///
/// Moving the retained range to the initialized tail lets `advance` reduce both
/// length and capacity before freezing. Other representations fall back to
/// ordinary slice semantics.
fn narrow_owned(bytes: Bytes, start: usize, end: usize) -> Bytes {
	debug_assert!(start <= end);
	debug_assert!(end <= bytes.len());
	match bytes.try_into_mut() {
		Ok(mut bytes) if bytes.capacity() == bytes.len() => {
			let retained = end - start;
			let tail = bytes.len() - retained;
			bytes.copy_within(start..end, tail);
			bytes.advance(tail);
			bytes.freeze()
		},
		Ok(bytes) => bytes.freeze().slice(start..end),
		Err(bytes) => bytes.slice(start..end),
	}
}

/// A decoded transport frame presented to a provider codec.
#[non_exhaustive]
pub enum Frame<'a> {
	/// An unnamed payload, such as one line from an NDJSON stream.
	Data(&'a [u8]),
	/// A named event, such as an SSE event and its assembled data payload.
	Event {
		/// The optional SSE event name.
		name: Option<&'a str>,
		/// The event data bytes.
		data: &'a [u8],
	},
	/// The transport's explicit end-of-stream sentinel.
	Done,
}

/// Codec-owned state retained between decoded transport frames.
///
/// A decode state belongs to one codec for one turn. Codecs can install their
/// concrete state once with [`DecodeState::get_or_insert_with`] without putting
/// provider-specific fields in this shared transport layer.
#[derive(Default)]
pub struct DecodeState {
	inner: Option<Box<dyn Any + Send>>,
}

impl DecodeState {
	/// Returns the codec state, initializing it with `make` when necessary.
	///
	/// # Panics
	///
	/// Panics if this decode state was previously initialized with another
	/// concrete type. Reusing one state across transports is a caller error.
	pub fn get_or_insert_with<T: Send + 'static>(&mut self, make: impl FnOnce() -> T) -> &mut T {
		if self.inner.is_none() {
			self.inner = Some(Box::new(make()));
		}
		self
			.inner
			.as_deref_mut()
			.and_then(|state| state.downcast_mut::<T>())
			.expect("decode state reused with a different transport")
	}
}

/// A provider-specific request encoder and response decoder.
///
/// # Encoding law
///
/// [`Transport::encode`] never drops a requested feature silently. Every
/// feature the provider cannot honor is returned in the `Vec<Unsupported>`
/// alongside the wire body; the unsupported report is part of the return type
/// so callers cannot accidentally forget it.
pub trait Transport: Send + Sync {
	/// Returns the catalog identifier for this wire transport.
	fn id(&self) -> TransportId;

	/// Encodes a request and reports everything the provider cannot honor.
	fn encode(&self, req: &ChatRequest, compat: &Compat)
	-> Result<(Bytes, Vec<Unsupported>), Error>;

	/// Converts a decoded transport frame into canonical turn events.
	fn decode(
		&self,
		frame: Frame<'_>,
		state: &mut DecodeState,
	) -> Result<SmallVec<TurnEvent, 2>, Error>;
}

#[cfg(test)]
mod tests {
	use omp_llm_types::StopReason;

	use super::with_tool_use_precedence;

	#[test]
	fn tool_use_only_precedes_benign_success_terminals() {
		assert_eq!(with_tool_use_precedence(StopReason::EndTurn, true), StopReason::ToolUse);
		assert_eq!(with_tool_use_precedence(StopReason::MaxTokens, true), StopReason::ToolUse);
		assert_eq!(
			with_tool_use_precedence(StopReason::ContentFilter, true),
			StopReason::ContentFilter
		);
		assert_eq!(with_tool_use_precedence(StopReason::EndTurn, false), StopReason::EndTurn);
	}
}
