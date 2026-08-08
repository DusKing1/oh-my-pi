//! Incremental Server-Sent Events decoding over byte chunks.

use bytes::{Bytes, BytesMut};
use omp_core::SmolStr;
use smallvec::SmallVec;

/// One assembled Server-Sent Event.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SseEvent {
	/// The optional value of the event's `event:` field.
	pub name: Option<SmolStr>,
	/// The assembled `data:` payload.
	///
	/// Multiple data fields are joined with a single line feed as required by
	/// the `EventSource` specification.
	pub data: Bytes,
}

/// An incremental, allocation-conscious Server-Sent Events decoder.
///
/// Complete single-line payloads are slices of the decoder's frozen input
/// buffer. Only multi-line payloads require a new allocation in order to
/// insert the specification-mandated line-feed separators.
#[derive(Default)]
pub struct SseDecoder {
	buffer:        BytesMut,
	last_event_id: Option<SmolStr>,
	retry_ms:      Option<u64>,
	done:          bool,
}

impl SseDecoder {
	/// Creates an empty decoder.
	#[must_use]
	pub fn new() -> Self {
		Self {
			buffer:        BytesMut::new(),
			last_event_id: None,
			retry_ms:      None,
			done:          false,
		}
	}

	/// Feeds a transport chunk and yields every complete event it contains.
	///
	/// An incomplete final event remains buffered for the next call. The
	/// terminal `data: [DONE]` sentinel is consumed, emits no event, and causes
	/// all subsequent input to be ignored.
	pub fn push(
		&mut self,
		chunk: Bytes,
	) -> impl DoubleEndedIterator<Item = SseEvent>
	+ Clone
	+ ExactSizeIterator
	+ std::iter::FusedIterator
	+ '_ {
		let mut events: SmallVec<SseEvent, 4> = SmallVec::new();
		if self.done {
			return events.into_iter();
		}
		if self.buffer.is_empty() && complete_event_len(&chunk) == Some(chunk.len()) {
			if let Some(event) = self.parse_event(chunk) {
				if event.data.as_ref() == b"[DONE]" {
					self.done = true;
				} else {
					events.push(event);
				}
			}
			return events.into_iter();
		}
		self.append(chunk);

		while let Some(frame_len) = complete_event_len(&self.buffer) {
			let frame = self.buffer.split_to(frame_len).freeze();
			if let Some(event) = self.parse_event(frame) {
				if event.data.as_ref() == b"[DONE]" {
					self.done = true;
					self.buffer.clear();
					break;
				}
				events.push(event);
			}
		}
		events.into_iter()
	}

	/// Returns the most recently accepted `id:` field.
	#[must_use]
	pub fn last_event_id(&self) -> Option<&str> {
		self.last_event_id.as_ref().map(SmolStr::as_str)
	}

	/// Returns the most recently accepted non-negative `retry:` value.
	#[must_use]
	pub const fn retry_ms(&self) -> Option<u64> {
		self.retry_ms
	}

	/// Returns whether the terminal `[DONE]` sentinel has been consumed.
	#[must_use]
	pub const fn is_done(&self) -> bool {
		self.done
	}

	fn append(&mut self, chunk: Bytes) {
		if chunk.is_empty() {
			return;
		}
		if self.buffer.is_empty() {
			self.buffer = match chunk.try_into_mut() {
				Ok(chunk) => chunk,
				Err(chunk) => BytesMut::from(chunk.as_ref()),
			};
		} else {
			self.buffer.extend_from_slice(&chunk);
		}
	}

	fn parse_event(&mut self, frame: Bytes) -> Option<SseEvent> {
		let mut data: SmallVec<(usize, usize), 4> = SmallVec::new();
		let mut name = None;
		let mut cursor = 0;

		while cursor < frame.len() {
			let newline = frame[cursor..]
				.iter()
				.position(|&byte| byte == b'\n')
				.map_or(frame.len(), |offset| cursor + offset);
			let mut end = newline;
			if end > cursor && frame[end - 1] == b'\r' {
				end -= 1;
			}
			if end == cursor {
				break;
			}

			let line = &frame[cursor..end];
			if line.first() != Some(&b':') {
				let colon = line.iter().position(|&byte| byte == b':');
				let (field, mut value_start) = match colon {
					Some(offset) => (&line[..offset], cursor + offset + 1),
					None => (line, end),
				};
				if value_start < end && frame[value_start] == b' ' {
					value_start += 1;
				}

				match field {
					b"data" => data.push((value_start, end)),
					b"event" => match SmolStr::from_utf8(&frame[value_start..end]) {
						Ok(value) if !value.is_empty() => name = Some(value),
						Ok(_) | Err(_) => {},
					},
					b"id" if !frame[value_start..end].contains(&0) => {
						if let Ok(value) = SmolStr::from_utf8(&frame[value_start..end]) {
							self.last_event_id = Some(value);
						}
					},
					b"id" => {},
					b"retry" => {
						if let Some(value) = parse_decimal(&frame[value_start..end]) {
							self.retry_ms = Some(value);
						}
					},
					_ => {},
				}
			}
			cursor = newline.saturating_add(1);
		}

		match data.as_slice() {
			[] => None,
			&[(start, end)] => Some(SseEvent { name, data: crate::narrow_owned(frame, start, end) }),
			ranges => {
				let payload_len = ranges
					.iter()
					.map(|(start, end)| end - start)
					.sum::<usize>()
					.saturating_add(ranges.len() - 1);
				let mut payload = BytesMut::with_capacity(payload_len);
				for (index, &(start, end)) in ranges.iter().enumerate() {
					if index != 0 {
						payload.extend_from_slice(b"\n");
					}
					payload.extend_from_slice(&frame[start..end]);
				}
				Some(SseEvent { name, data: payload.freeze() })
			},
		}
	}
}

fn complete_event_len(buffer: &[u8]) -> Option<usize> {
	let mut line_start = 0;
	for (index, &byte) in buffer.iter().enumerate() {
		if byte != b'\n' {
			continue;
		}
		let line_end = if index > line_start && buffer[index - 1] == b'\r' {
			index - 1
		} else {
			index
		};
		if line_end == line_start {
			return Some(index + 1);
		}
		line_start = index + 1;
	}
	None
}

fn parse_decimal(bytes: &[u8]) -> Option<u64> {
	if bytes.is_empty() {
		return None;
	}
	bytes.iter().try_fold(0_u64, |value, &byte| {
		byte
			.is_ascii_digit()
			.then_some(byte - b'0')
			.and_then(|digit| value.checked_mul(10)?.checked_add(u64::from(digit)))
	})
}

#[cfg(test)]
mod tests {
	use bytes::Bytes;

	use super::SseDecoder;

	#[test]
	fn event_split_mid_field_across_three_chunks() {
		let mut decoder = SseDecoder::new();
		assert_eq!(decoder.push(Bytes::from_static(b"eve")).count(), 0);
		assert_eq!(
			decoder
				.push(Bytes::from_static(b"nt: mes\ndata: hel"))
				.count(),
			0
		);
		let events = decoder
			.push(Bytes::from_static(b"lo\n\n"))
			.collect::<Vec<_>>();
		assert_eq!(events.len(), 1);
		assert_eq!(events[0].name.as_ref().map(|name| name.as_str()), Some("mes"));
		assert_eq!(events[0].data.as_ref(), b"hello");
	}

	#[test]
	fn multiline_data_is_joined_with_line_feeds() {
		let mut decoder = SseDecoder::new();
		let events = decoder
			.push(Bytes::from_static(b"data: first\ndata: second\ndata:\n\n"))
			.collect::<Vec<_>>();
		assert_eq!(events[0].data.as_ref(), b"first\nsecond\n");
	}

	#[test]
	fn split_utf8_sequence_is_preserved() {
		let mut decoder = SseDecoder::new();
		assert_eq!(decoder.push(Bytes::from_static(b"data: caf\xc3")).count(), 0);
		let events = decoder
			.push(Bytes::from_static(b"\xa9\n\n"))
			.collect::<Vec<_>>();
		assert_eq!(std::str::from_utf8(&events[0].data), Ok("café"));
	}

	#[test]
	fn crlf_and_lf_framing_both_parse() {
		let mut decoder = SseDecoder::new();
		let events = decoder
			.push(Bytes::from_static(b"data: one\r\n\r\ndata: two\n\n"))
			.collect::<Vec<_>>();
		assert_eq!(events.len(), 2);
		assert_eq!(events[0].data.as_ref(), b"one");
		assert_eq!(events[1].data.as_ref(), b"two");
	}

	#[test]
	fn comments_and_unknown_fields_are_ignored() {
		let mut decoder = SseDecoder::new();
		let events = decoder
			.push(Bytes::from_static(
				b": keepalive\nunknown: value\nid: cursor-1\nretry: 250\ndata: ok\n\n",
			))
			.collect::<Vec<_>>();
		assert_eq!(events.len(), 1);
		assert_eq!(events[0].data.as_ref(), b"ok");
		assert_eq!(decoder.last_event_id(), Some("cursor-1"));
		assert_eq!(decoder.retry_ms(), Some(250));
	}

	#[test]
	fn coalesced_done_sentinel_terminates_after_emitting_preceding_event_once() {
		let mut decoder = SseDecoder::new();
		let events = decoder
			.push(Bytes::from_static(b"data: before\n\ndata: [DONE]\n\n"))
			.collect::<Vec<_>>();

		assert_eq!(events.len(), 1);
		assert_eq!(events[0].data.as_ref(), b"before");
		assert!(events.iter().all(|event| event.data.as_ref() != b"[DONE]"));
		assert!(decoder.is_done());
		assert_eq!(decoder.push(Bytes::from_static(b"data: later\n\n")).count(), 0);
	}
}
