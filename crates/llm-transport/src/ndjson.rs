//! Incremental newline-delimited JSON framing over byte chunks.

use bytes::{Bytes, BytesMut};
use smallvec::SmallVec;

/// An incremental newline-delimited JSON decoder.
///
/// The decoder does not parse JSON itself. It returns each non-empty record as
/// a zero-copy [`Bytes`] slice and retains an incomplete trailing record until
/// a later call supplies its newline.
#[derive(Default)]
pub struct NdjsonDecoder {
	buffer: BytesMut,
}

impl NdjsonDecoder {
	/// Creates an empty decoder.
	#[must_use]
	pub fn new() -> Self {
		Self { buffer: BytesMut::new() }
	}

	/// Feeds a transport chunk and yields every complete non-empty JSON record.
	///
	/// Both LF and CRLF delimiters are accepted and are not included in the
	/// returned record bytes.
	pub fn push(
		&mut self,
		chunk: Bytes,
	) -> impl DoubleEndedIterator<Item = Bytes> + Clone + ExactSizeIterator + std::iter::FusedIterator + '_
	{
		let mut records: SmallVec<Bytes, 4> = SmallVec::new();
		if self.buffer.is_empty()
			&& chunk.last() == Some(&b'\n')
			&& !chunk[..chunk.len() - 1].contains(&b'\n')
		{
			let mut end = chunk.len() - 1;
			if end != 0 && chunk[end - 1] == b'\r' {
				end -= 1;
			}
			if end != 0 {
				records.push(crate::narrow_owned(chunk, 0, end));
			}
			return records.into_iter();
		}
		self.append(chunk);

		while let Some(newline) = self.buffer.iter().position(|&byte| byte == b'\n') {
			let mut line = self.buffer.split_to(newline + 1).freeze();
			let mut end = newline;
			if end != 0 && line[end - 1] == b'\r' {
				end -= 1;
			}
			if end != 0 {
				line.truncate(end);
				records.push(line);
			}
		}
		records.into_iter()
	}

	/// Returns the number of bytes retained in the incomplete trailing record.
	#[must_use]
	pub fn buffered_len(&self) -> usize {
		self.buffer.len()
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
}

#[cfg(test)]
mod tests {
	use bytes::Bytes;

	use super::NdjsonDecoder;

	#[test]
	fn partial_trailing_line_is_retained_until_completed() {
		let mut decoder = NdjsonDecoder::new();
		let records = decoder
			.push(Bytes::from_static(b"{\"first\":1}\n{\"second\":"))
			.collect::<Vec<_>>();
		assert_eq!(records.as_slice(), &[Bytes::from_static(b"{\"first\":1}")]);
		assert_eq!(decoder.buffered_len(), b"{\"second\":".len());

		let records = decoder
			.push(Bytes::from_static(b"2}\r\n"))
			.collect::<Vec<_>>();
		assert_eq!(records.as_slice(), &[Bytes::from_static(b"{\"second\":2}")]);
		assert_eq!(decoder.buffered_len(), 0);
	}
}
