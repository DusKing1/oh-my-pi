//! Power-of-2 base encodings using bit manipulation.
//!
//! This module implements encodings where the base is a power of 2 (e.g.,
//! Base64, Base32, Hex). These encodings use efficient bit manipulation
//! algorithms instead of division.

use core::{fmt, mem::MaybeUninit, slice, str};
use std::{cmp::Ordering, intrinsics, io, iter::FusedIterator};

use bytes::{BufMut, Bytes};

use super::{error::*, fixed_arr::*, opt};

/// Encoding lookup table mapping 8-bit values to output characters.
pub type ETable = [u8; 256];
/// Decoding lookup table mapping ASCII characters to base-N values.
pub type DTable = [u8; 256];

/// A compile-time dictionary for base-N encoding/decoding.
///
/// # Type Parameters
/// - `N`: The base (must be a power of 2 and <= 256)
/// - `PAD`: Optional padding character (0 means no padding)
#[derive(Clone, Copy, Debug)]
pub struct Encoding<const N: usize> {
	dtable: DTable,
	etable: ETable,
	pad:    u8,
}

impl<const N: usize> Encoding<N> {
	/// The number of bits encoded per character.
	pub const BITS_PER_CHAR: u32 = N.trailing_zeros();
	/// The input group size (number of bytes per encoding group).
	pub const GROUP_IN: usize = (Self::GROUP_OUT * Self::BITS_PER_CHAR as usize) / 8;
	/// The output group size (number of characters per encoding group).
	pub const GROUP_OUT: usize = 8 / Self::const_gcd(8, Self::BITS_PER_CHAR as usize);
	/// The bit mask for extracting character values.
	pub const MASK: u8 = (N - 1) as u8;

	/// Const GCD for computing group sizes.
	const fn const_gcd(mut a: usize, mut b: usize) -> usize {
		while b != 0 {
			let t = a % b;
			a = b;
			b = t;
		}
		a
	}

	/// Creates a new encoding from an alphabet.
	pub const fn new(alphabet: &[u8; N], pad: Option<u8>) -> Self {
		assert!(N.is_power_of_two(), "base must be power of 2");
		assert!(N <= 256, "base must be <= 256");

		let mut dtable: DTable = [0xff; _];
		let mut etable: ETable = [0; _];
		let mut i = 0;
		while i < N {
			let ch = alphabet[i] as usize;
			assert!(dtable[ch] == 0xff, "duplicate character in alphabet");
			dtable[ch] = i as u8;
			etable[i] = ch as u8;
			i += 1;
		}
		while i < 256 {
			etable[i] = alphabet[i % N];
			i += 1;
		}

		let pad = if let Some(pad) = pad {
			assert!(pad != 0, "padding char must not be 0");
			assert!(dtable[pad as usize] == 0xff, "padding char must not be in alphabet");
			// Use 0xfe to mark padding (0xff is for invalid chars)
			dtable[pad as usize] = 0xfe;
			pad
		} else {
			0
		};

		Self { etable, dtable, pad }
	}

	/// Add lowercase aliases for [A-Z] if present in alphabet.
	pub const fn with_lowercase(mut self) -> Self {
		let mut c = b'A';
		while c <= b'Z' {
			let idx = self.dtable[c as usize];
			if idx != 0xff {
				self.dtable[(c | 0x20) as usize] = idx;
			}
			c += 1;
		}
		self
	}

	/// Returns the bit mask for extracting character values.
	#[inline(always)]
	pub const fn mask(&self) -> u8 {
		Self::MASK
	}

	/// Returns the output group size (number of characters per encoding group).
	#[inline(always)]
	pub const fn group_size_out(&self) -> usize {
		Self::GROUP_OUT
	}

	/// Returns the input group size (number of bytes per encoding group).
	#[inline(always)]
	pub const fn group_size_in(&self) -> usize {
		Self::GROUP_IN
	}

	/// Returns the padding character if padding is enabled.
	#[inline(always)]
	pub const fn padding(&self) -> Option<u8> {
		if self.pad == 0 { None } else { Some(self.pad) }
	}

	/// Returns the bits per character.
	#[inline(always)]
	pub const fn bits_per_char(&self) -> u32 {
		Self::BITS_PER_CHAR
	}

	/// Returns the exact encoded length for a given source byte length.
	#[inline]
	pub const fn encode_len(&self, src_len: usize) -> usize {
		let bits = Self::BITS_PER_CHAR as usize;
		let chars = (src_len * 8).div_ceil(bits);
		if self.pad == 0 {
			chars
		} else {
			let group = Self::GROUP_OUT;
			chars.div_ceil(group) * group
		}
	}

	/// Returns the decoded length for a given number of characters (excluding
	/// padding).
	#[inline]
	pub const fn decode_len(&self, chars_len_upto_pad: usize) -> usize {
		let bits = Self::BITS_PER_CHAR as usize;
		(chars_len_upto_pad * bits) / 8
	}

	/// Encodes a single value (masked to valid range) to its character.
	#[inline(always)]
	pub const fn encode(&self, value: u8) -> u8 {
		self.etable[value as usize]
	}

	/// Decodes a single character to its value, returning `None` for invalid
	/// characters.
	#[inline(always)]
	pub const fn decode(&self, ch: u8) -> Option<u8> {
		let val = self.dtable[ch as usize];
		if val >= 0xfe {
			std::hint::cold_path();
			None
		} else {
			Some(val)
		}
	}

	/// Encodes bytes into a mutable buffer, returning the number of characters
	/// written. Routes to optimized implementation at runtime or const
	/// fallback at compile time.
	#[inline]
	pub const fn encode_mut(&self, src: &[u8], dst: &mut [u8]) -> usize {
		intrinsics::const_eval_select((self, src, dst), Self::encode_const, Self::encode_opt)
	}

	/// Decodes characters into a mutable buffer, returning the number of bytes
	/// written. Routes to optimized implementation at runtime or const
	/// fallback at compile time.
	#[inline]
	pub const fn decode_mut(&self, src: &[u8], dst: &mut [u8]) -> Result<usize> {
		intrinsics::const_eval_select((self, src, dst), Self::decode_const, Self::decode_opt)
	}

	/// Encodes a fixed-size byte array, returning an `ArrayStr` wrapper.
	#[inline]
	pub const fn encode_n<const L: usize>(&self, src: &[u8; L]) -> ArrayStr<L> {
		let mut out = [[0u8; 2]; L];
		// Safety guard: ArrayStr<L> has 2*L bytes of capacity.
		// Ensure the encoding actually fits to avoid silent truncation (e.g. Base64
		// padded, L==1).
		assert!(
			self.encode_len(L) <= (L * 2),
			"encode_n: capacity (2*L) insufficient for this encoding"
		);
		let len = self.encode_mut(src, out.as_flattened_mut());
		ArrayStr::new(out, len)
	}

	/// Decodes a fixed-size character array, returning an `Array` wrapper.
	#[inline]
	pub const fn decode_n<const L: usize>(&self, src: &[u8; L]) -> Option<Array<L>> {
		let mut out = [0u8; L];
		let Ok(len) = self.decode_mut(src, &mut out) else {
			return None;
		};
		Some(Array::new(out, len))
	}

	/// Creates an encoding writer that wraps an `io::Write` and encodes bytes
	/// before writing them.
	///
	/// The returned writer buffers input, encodes it using this encoding, and
	/// writes the encoded output to the inner writer.
	///
	/// # Examples
	/// ```
	/// use std::io::Write;
	///
	/// use omp_core::base64;
	///
	/// let mut output = Vec::new();
	/// let mut writer = base64::STD.encode_writer(&mut output);
	/// writer.write_all(b"Hello").unwrap();
	/// writer.flush().unwrap();
	/// ```
	#[inline]
	pub const fn encode_writer<W: io::Write>(&self, inner: W) -> EncodeWriter<W, N> {
		EncodeWriter::new(inner, *self)
	}

	/// Creates a decoding writer that wraps an `io::Write` and decodes bytes
	/// before writing them.
	///
	/// The returned writer buffers encoded input, decodes it using this
	/// encoding, and writes the decoded output to the inner writer.
	///
	/// # Examples
	/// ```
	/// use std::io::Write;
	///
	/// use omp_core::base64;
	///
	/// let mut output = Vec::new();
	/// {
	/// 	let mut writer = base64::STD.decode_writer(&mut output);
	/// 	writer.write_all(b"SGVsbG8=").unwrap();
	/// 	writer.flush().unwrap();
	/// }
	/// assert_eq!(output, b"Hello");
	/// ```
	#[inline]
	pub const fn decode_writer<W: io::Write>(&self, inner: W) -> DecodeWriter<W, N> {
		DecodeWriter::new(inner, *self)
	}

	/// Fast runtime encoder (non-const) - routes to specialized paths.
	#[doc(hidden)]
	#[inline(always)]
	pub fn encode_opt(this: &Self, src: &[u8], dst: &mut [u8]) -> usize {
		if const { N == 64 } {
			opt::enc64(src, &this.etable, dst, this.pad)
		} else if const { N == 32 } {
			opt::enc32(src, &this.etable, dst, this.pad)
		} else {
			Self::encode_const(this, src, dst)
		}
	}

	/// Fast runtime decoder (non-const) - routes to specialized paths.
	#[doc(hidden)]
	#[inline(always)]
	pub fn decode_opt(this: &Self, src: &[u8], dst: &mut [u8]) -> Result<usize> {
		if const { N == 64 } {
			opt::dec64(src, &this.dtable, dst, this.pad)
		} else if const { N == 32 } {
			opt::dec32(src, &this.dtable, dst, this.pad)
		} else {
			Self::decode_const(this, src, dst)
		}
	}

	/// Regular compile-time encoder.
	#[doc(hidden)]
	#[inline(always)]
	pub const fn encode_const(this: &Self, src: &[u8], dst: &mut [u8]) -> usize {
		let bits = Self::BITS_PER_CHAR;
		let mask = Self::MASK;
		let group_in = Self::GROUP_IN;
		let group_out = Self::GROUP_OUT;
		let pad = this.pad;

		let mut src_idx = 0;
		let mut dst_idx = 0;

		// Process full groups
		while src_idx + group_in <= src.len() && dst_idx + group_out <= dst.len() {
			let mut buffer = 0u64;
			let mut buffer_bits = 0u32;

			let mut i = 0;
			while i < group_in {
				buffer = (buffer << 8) | (src[src_idx] as u64);
				buffer_bits += 8;
				src_idx += 1;
				i += 1;
			}

			let mut j = 0;
			while j < group_out {
				buffer_bits -= bits;
				let val = ((buffer >> buffer_bits) & (mask as u64)) as u8;
				dst[dst_idx] = this.encode(val);
				dst_idx += 1;
				j += 1;
			}
		}

		// Handle remaining bytes
		if src_idx < src.len() {
			let mut buffer = 0u64;
			let mut buffer_bits = 0u32;

			while src_idx < src.len() {
				buffer = (buffer << 8) | (src[src_idx] as u64);
				buffer_bits += 8;
				src_idx += 1;
			}

			while buffer_bits >= bits && dst_idx < dst.len() {
				buffer_bits -= bits;
				let val = ((buffer >> buffer_bits) & (mask as u64)) as u8;
				dst[dst_idx] = this.encode(val);
				dst_idx += 1;
			}

			if buffer_bits > 0 && dst_idx < dst.len() {
				let val = ((buffer << (bits - buffer_bits)) & (mask as u64)) as u8;
				dst[dst_idx] = this.encode(val);
				dst_idx += 1;
			}

			// Add padding
			if pad != 0 {
				let expected = (src.len() * 8).div_ceil(bits as usize).div_ceil(group_out) * group_out;
				while dst_idx < expected && dst_idx < dst.len() {
					dst[dst_idx] = pad;
					dst_idx += 1;
				}
			}
		}

		dst_idx
	}

	/// Regular compile-time decoder.
	#[doc(hidden)]
	#[inline(always)]
	pub const fn decode_const(this: &Self, src: &[u8], dst: &mut [u8]) -> Result<usize> {
		let bits = Self::BITS_PER_CHAR;
		let pad = this.pad;

		let mut src_idx = 0;
		let mut dst_idx = 0;
		let mut buffer = 0u64;
		let mut buffer_bits = 0u32;

		while src_idx < src.len() {
			let ch = src[src_idx];
			src_idx += 1;

			if pad != 0 && ch == pad {
				break;
			}

			let Some(val) = this.decode(ch) else {
				return Err(DecodeError::InvalidCharacter(ch));
			};

			buffer = (buffer << bits) | (val as u64);
			buffer_bits += bits;

			while buffer_bits >= 8 && dst_idx < dst.len() {
				buffer_bits -= 8;
				dst[dst_idx] = ((buffer >> buffer_bits) & 0xff) as u8;
				dst_idx += 1;
			}
		}

		Ok(dst_idx)
	}
}

// ============================================================================
// DECODER
// ============================================================================

/// A streaming base-N decoder that converts encoded characters to bytes.
///
/// # Examples
/// ```
/// use omp_core::base64;
/// let encoded = b"SGVsbG8=";
/// let decoded = base64::decode(encoded).into_vec().unwrap();
/// assert_eq!(decoded, b"Hello");
/// ```
#[derive(Debug, Clone)]
pub struct Decoder<'a, const N: usize> {
	/// Remaining source bytes
	src:              &'a [u8],
	/// Dictionary for decoding
	enc:              &'a Encoding<N>,
	/// Bit accumulator
	buffer:           u64,
	/// Number of bits in buffer
	buffer_bits:      u8,
	/// Cached bits per character
	bits:             u8,
	/// Remaining non-padding chars to decode (decremented as we consume)
	remaining_nonpad: usize,
}

impl<'a, const N: usize> Decoder<'a, N> {
	/// Creates a new decoder from a byte slice.
	pub const fn new(src: &'a [u8], enc: &'a Encoding<N>) -> Self {
		let bits = Encoding::<N>::BITS_PER_CHAR as u8;

		// Find padding position once for O(1) len()
		let remaining_nonpad = if enc.pad == 0 {
			src.len()
		} else {
			let mut i = 0;
			while i < src.len() {
				if src[i] == enc.pad {
					break;
				}
				i += 1;
			}
			i
		};

		Self { src, enc, buffer: 0, buffer_bits: 0, bits, remaining_nonpad }
	}

	/// Collects the decoded bytes into a `Vec<u8>`.
	pub fn into_vec(self) -> Result<Vec<u8>> {
		let cap = self.enc.decode_len(self.remaining_nonpad);
		let mut buf = vec![0u8; cap];
		let written = self.enc.decode_mut(self.src, &mut buf)?;
		buf.truncate(written);
		Ok(buf)
	}

	/// Collects the decoded bytes into a `Bytes`.
	pub fn into_bytes(self) -> Result<Bytes> {
		self.into_vec().map(Bytes::from)
	}

	/// Collects the decoded bytes into a slice.
	#[inline]
	pub const fn into_slice(self, buf: &mut [u8]) -> Result<usize> {
		self.enc.decode_mut(self.src, buf)
	}

	/// Collects the decoded bytes into a `[u8; N]`.
	pub fn into_array<const K: usize>(self) -> Result<[u8; K]> {
		let mut buf = [0u8; K];
		let n = self.into_slice(&mut buf)?;
		if n != K {
			return Err(DecodeError::InputTooShort);
		}
		Ok(buf)
	}

	/// Collects the decoded bytes into a `BufMut`.
	pub fn into_buf<B: BufMut>(mut self, mut buf: B) -> Result<B> {
		loop {
			let mut n = 0;
			// SAFETY: We do not read the uninitialized bytes, only write to them.
			let chunk = unsafe { buf.chunk_mut().as_uninit_slice_mut() };

			// If no space available, we're done (caller must provide enough capacity)
			if chunk.is_empty() {
				break Ok(buf);
			}

			for (b, dst) in self.by_ref().zip(&mut *chunk) {
				dst.write(b?);
				n += 1;
			}
			let exhausted = n < chunk.len();

			// SAFETY: We've written `n` bytes to the buffer.
			unsafe { buf.advance_mut(n) };

			if exhausted {
				break Ok(buf);
			}
		}
	}

	/// Extends an existing collection with the decoded bytes.
	pub fn extend_into<E: Extend<u8> + ?Sized>(self, buf: &mut E) -> Result<usize> {
		let mut n = 0;
		for byte in self {
			buf.extend_one(byte?);
			n += 1;
		}
		Ok(n)
	}

	/// Writes the decoded bytes to an `io::Write`.
	pub fn write_into<W: io::Write + ?Sized>(mut self, writer: &mut W) -> io::Result<usize> {
		let mut tmp = [MaybeUninit::<u8>::uninit(); 512];
		let mut total = 0;
		loop {
			let mut i = 0;
			for dst in &mut tmp {
				if let Some(b) = self.next() {
					dst.write(b.map_err(io::Error::other)?);
					i += 1;
				} else {
					break;
				}
			}
			if i == 0 {
				break;
			}
			// SAFETY: We've written exactly `i` bytes to `tmp` via MaybeUninit::write,
			// fully initializing tmp[0..i]. slice::from_raw_parts creates a view of
			// these initialized bytes for writing.
			unsafe {
				writer.write_all(slice::from_raw_parts(tmp.as_ptr().cast(), i))?;
			}
			total += i;
		}
		Ok(total)
	}
}

impl<const N: usize> Iterator for Decoder<'_, N> {
	type Item = Result<u8>;

	fn next(&mut self) -> Option<Self::Item> {
		let bits = self.bits;

		loop {
			// If we have enough bits, emit a byte
			if self.buffer_bits >= 8 {
				let shift = self.buffer_bits - 8;
				let byte = ((self.buffer >> shift) & 0xff) as u8;
				self.buffer_bits -= 8;
				// No need to mask - just track buffer_bits
				return Some(Ok(byte));
			}

			// Try to get next character
			let ch = *self.src.split_off_first()?;

			// Break on first padding (consistent with const decode_mut)
			if self.enc.pad != 0 && ch == self.enc.pad {
				return None;
			}

			// Decode character
			let Some(val) = self.enc.decode(ch) else {
				return Some(Err(DecodeError::InvalidCharacter(ch)));
			};

			// Decrement remaining non-padding count
			self.remaining_nonpad = self.remaining_nonpad.saturating_sub(1);

			// Add to buffer (pack into high end)
			self.buffer = (self.buffer << bits) | (val as u64);
			self.buffer_bits += bits;
		}
	}

	fn size_hint(&self) -> (usize, Option<usize>) {
		let n = self.len();
		(n, Some(n))
	}
}

impl<const N: usize> ExactSizeIterator for Decoder<'_, N> {
	#[inline]
	fn len(&self) -> usize {
		// O(1) using cached remaining_nonpad
		let bits = self.bits as usize;
		((self.remaining_nonpad * bits) + self.buffer_bits as usize) / 8
	}
}

impl<const N: usize> FusedIterator for Decoder<'_, N> {}

impl<const N: usize, const K: usize> TryFrom<Decoder<'_, N>> for [u8; K] {
	type Error = DecodeError;

	fn try_from(decoder: Decoder<'_, N>) -> Result<Self> {
		decoder.into_array()
	}
}

impl<const N: usize> fmt::Display for Decoder<'_, N> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		for byte in self.clone() {
			if let Ok(byte) = byte {
				write!(f, "{byte:02x}")?;
			} else {
				write!(f, "??")?;
			}
		}
		Ok(())
	}
}

impl<const N: usize> PartialEq<[u8]> for Decoder<'_, N> {
	fn eq(&self, other: &[u8]) -> bool {
		self.clone().eq(other.iter().map(|&b| Ok(b)))
	}
}

impl<const N: usize> PartialOrd<[u8]> for Decoder<'_, N> {
	fn partial_cmp(&self, other: &[u8]) -> Option<Ordering> {
		Some(self.clone().cmp(other.iter().map(|&b| Ok(b))))
	}
}

// ============================================================================
// ENCODER
// ============================================================================

/// Iterator that encodes bytes as individual base-N ASCII characters.
///
/// # Examples
/// ```
/// use omp_core::base64;
/// let data = b"Hello";
/// let encoded = base64::encode(data).into_string();
/// assert_eq!(encoded, "SGVsbG8=");
/// ```
#[derive(Debug, Clone)]
pub struct Encoder<'a, const N: usize> {
	/// Source bytes
	src:           &'a [u8],
	/// Dictionary for encoding
	enc:           &'a Encoding<N>,
	/// Bit accumulator
	buffer:        u64,
	/// Number of bits in buffer
	buffer_bits:   usize,
	/// Characters emitted so far
	chars_emitted: usize,
	/// Pending padding characters
	padding_count: usize,
	/// Whether we're done with source
	done:          bool,
}

impl<'a, const N: usize> Encoder<'a, N> {
	/// Creates a new encoder.
	pub const fn new(src: &'a [u8], enc: &'a Encoding<N>) -> Self {
		Self { src, enc, buffer: 0, buffer_bits: 0, chars_emitted: 0, padding_count: 0, done: false }
	}

	/// Collects into a `Vec<u8>`.
	pub fn into_vec(self) -> Vec<u8> {
		let out_len = self.enc.encode_len(self.src.len());
		let mut buf = vec![0u8; out_len];
		let written = self.enc.encode_mut(self.src, &mut buf);
		buf.truncate(written);
		buf
	}

	/// Collects into a `Bytes`.
	pub fn into_bytes(self) -> Bytes {
		Bytes::from(self.into_vec())
	}

	/// Collects into a String.
	pub fn into_string(self) -> String {
		ascii_to_str_owned(self.into_vec())
	}

	/// Extends into an existing buffer.
	pub fn extend_into<E: Extend<u8> + ?Sized>(self, buf: &mut E) {
		buf.extend(self);
	}

	/// Collects into a `BufMut`.
	pub fn into_buf<B: BufMut>(mut self, mut buf: B) -> B {
		loop {
			let mut n = 0;
			// SAFETY: We do not read the uninitialized bytes, only write to them.
			let chunk = unsafe { buf.chunk_mut().as_uninit_slice_mut() };

			// If no space available, we're done (caller must provide enough capacity)
			if chunk.is_empty() {
				break buf;
			}

			for (b, dst) in self.by_ref().zip(&mut *chunk) {
				dst.write(b);
				n += 1;
			}
			let exhausted = n < chunk.len();
			// SAFETY: We've written `n` bytes to the buffer.
			unsafe { buf.advance_mut(n) };
			if exhausted {
				break buf;
			}
		}
	}

	/// Writes to an `io::Write`.
	pub fn write_into<W: io::Write + ?Sized>(mut self, writer: &mut W) -> io::Result<usize> {
		let mut tmp = [MaybeUninit::<u8>::uninit(); 512];
		let mut total = 0;
		loop {
			let mut i = 0;
			for dst in &mut tmp {
				if let Some(b) = self.next() {
					dst.write(b);
					i += 1;
				} else {
					break;
				}
			}
			if i == 0 {
				break;
			}
			// SAFETY: We've written exactly `i` bytes to `tmp` via MaybeUninit::write,
			// fully initializing tmp[0..i]. slice::from_raw_parts creates a view of
			// these initialized bytes for writing.
			unsafe {
				writer.write_all(slice::from_raw_parts(tmp.as_ptr().cast(), i))?;
			}
			total += i;
		}
		Ok(total)
	}

	/// Writes to a `fmt::Write`.
	pub fn format_into<W: fmt::Write + ?Sized>(self, writer: &mut W) -> fmt::Result {
		for byte in self {
			writer.write_char(byte as char)?;
		}
		Ok(())
	}
}

impl<const N: usize> From<Encoder<'_, N>> for String {
	fn from(encoder: Encoder<'_, N>) -> Self {
		encoder.into_string()
	}
}

impl<const N: usize> From<Encoder<'_, N>> for Bytes {
	fn from(encoder: Encoder<'_, N>) -> Self {
		encoder.into_bytes()
	}
}

impl<const N: usize> From<Encoder<'_, N>> for Vec<u8> {
	fn from(encoder: Encoder<'_, N>) -> Self {
		encoder.into_vec()
	}
}

impl<const N: usize> Iterator for Encoder<'_, N> {
	type Item = u8;

	fn next(&mut self) -> Option<Self::Item> {
		let bits = Encoding::<N>::BITS_PER_CHAR as usize;
		let mask = Encoding::<N>::MASK;

		loop {
			// Emit padding if any
			if self.padding_count > 0 {
				self.padding_count -= 1;
				return self.enc.padding();
			}

			// Emit character if we have enough bits
			if self.buffer_bits >= bits {
				let shift = self.buffer_bits - bits;
				let val = ((self.buffer >> shift) as u8) & mask;
				self.buffer_bits -= bits;
				self.chars_emitted += 1;
				return Some(self.enc.encode(val));
			}

			// Refill from source
			if let Some(byte) = self.src.split_off_first() {
				self.buffer = (self.buffer << 8) | (*byte as u64);
				self.buffer_bits += 8;
				continue;
			}

			// Source exhausted
			if !self.done {
				self.done = true;

				// Emit final partial character
				if self.buffer_bits > 0 {
					let val = ((self.buffer << (bits - self.buffer_bits)) as u8) & mask;
					self.buffer_bits = 0;
					let total_chars = self.chars_emitted + 1;

					if self.enc.pad != 0 {
						let group_out = Encoding::<N>::GROUP_OUT;
						let rem = total_chars % group_out;
						if rem > 0 {
							self.padding_count = group_out - rem;
						}
					}

					self.chars_emitted += 1;
					return Some(self.enc.encode(val));
				}

				// Calculate padding for exact multiples
				if self.enc.pad != 0 && self.chars_emitted > 0 {
					let group_out = Encoding::<N>::GROUP_OUT;
					let rem = self.chars_emitted % group_out;
					if rem > 0 {
						self.padding_count = group_out - rem;
						continue;
					}
				}
			}

			return None;
		}
	}

	fn size_hint(&self) -> (usize, Option<usize>) {
		let n = self.len();
		(n, Some(n))
	}
}

impl<const N: usize> ExactSizeIterator for Encoder<'_, N> {
	#[inline]
	fn len(&self) -> usize {
		let bits = Encoding::<N>::BITS_PER_CHAR as usize;
		let total_bits = (self.src.len() * 8) + self.buffer_bits;
		let chars = total_bits.div_ceil(bits) + self.padding_count;
		if self.enc.pad != 0 {
			let group = Encoding::<N>::GROUP_OUT;
			chars.div_ceil(group) * group
		} else {
			chars
		}
	}
}

impl<const N: usize> FusedIterator for Encoder<'_, N> {}

impl<const N: usize> serde::Serialize for Encoder<'_, N> {
	fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
		let len = self.enc.encode_len(self.src.len());
		serialize(serializer, len, |buffer| self.enc.encode_mut(self.src, buffer))
	}
}

impl<const N: usize> fmt::Display for Encoder<'_, N> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		format_with_precision(self.clone(), f)
	}
}

impl<const N: usize> Eq for Encoder<'_, N> {}

impl<'b, const N: usize, const M: usize> PartialEq<Encoder<'b, M>> for Encoder<'_, N> {
	fn eq(&self, other: &Encoder<'b, M>) -> bool {
		self.clone().eq(other.clone())
	}
}
impl<const N: usize> Ord for Encoder<'_, N> {
	fn cmp(&self, other: &Self) -> Ordering {
		self.clone().cmp(other.clone())
	}
}
impl<'b, const N: usize, const M: usize> PartialOrd<Encoder<'b, M>> for Encoder<'_, N> {
	fn partial_cmp(&self, other: &Encoder<'b, M>) -> Option<Ordering> {
		Some(self.clone().cmp(other.clone()))
	}
}

impl<const N: usize> PartialEq<[u8]> for Encoder<'_, N> {
	fn eq(&self, other: &[u8]) -> bool {
		self.clone().eq(other.iter().copied())
	}
}

impl<const N: usize> PartialEq<str> for Encoder<'_, N> {
	fn eq(&self, other: &str) -> bool {
		self.clone().eq(other.as_bytes().iter().copied())
	}
}

impl<const N: usize> PartialOrd<[u8]> for Encoder<'_, N> {
	fn partial_cmp(&self, other: &[u8]) -> Option<Ordering> {
		Some(self.clone().cmp(other.iter().copied()))
	}
}

impl<const N: usize> PartialOrd<str> for Encoder<'_, N> {
	fn partial_cmp(&self, other: &str) -> Option<Ordering> {
		Some(self.clone().cmp(other.as_bytes().iter().copied()))
	}
}

// ============================================================================
// IO WRITERS
// ============================================================================

/// Buffer size for input data (raw bytes for encoding, encoded chars for
/// decoding).
const INPUT_BUFFER_SIZE: usize = 768;
/// Buffer size for output data (encoded chars for encoding, raw bytes for
/// decoding).
const OUTPUT_BUFFER_SIZE: usize = 2048;

/// A buffered writer that encodes raw bytes before writing them to an inner
/// writer.
///
/// This writer buffers input bytes, encodes them using the specified
/// `Encoding<N>`, and writes the encoded output to the wrapped writer. It
/// implements `io::Write`, allowing transparent encoding of data streams.
///
/// # Examples
/// ```
/// use std::io::Write;
///
/// use omp_core::encoding::{EncodeWriter, base64};
///
/// let mut output = Vec::new();
/// let mut writer = base64::encode_writer(&mut output);
/// writer.write_all(b"Hello, World!").unwrap();
/// writer.flush().unwrap();
/// // output now contains base64-encoded data
/// ```
pub struct EncodeWriter<W: io::Write, const N: usize> {
	inner:      Option<W>,
	enc:        Encoding<N>,
	input_buf:  [MaybeUninit<u8>; INPUT_BUFFER_SIZE],
	input_len:  usize,
	output_buf: [u8; OUTPUT_BUFFER_SIZE],
}

impl<W: io::Write, const N: usize> EncodeWriter<W, N> {
	/// Creates a new encoding writer that wraps the given writer.
	pub const fn new(inner: W, enc: Encoding<N>) -> Self {
		Self {
			inner: Some(inner),
			enc,
			input_buf: [MaybeUninit::uninit(); INPUT_BUFFER_SIZE],
			input_len: 0,
			output_buf: [0u8; OUTPUT_BUFFER_SIZE],
		}
	}

	/// Consumes this writer, flushing any buffered data and returning the inner
	/// writer.
	///
	/// # Errors
	/// Returns an error if flushing fails.
	pub fn into_inner(mut self) -> io::Result<W> {
		io::Write::flush(&mut self)?;
		let inner = self.inner.take().expect("EncodeWriter already consumed");
		// Prevent Drop from running since we're consuming the writer
		core::mem::forget(self);
		Ok(inner)
	}

	/// Encodes and flushes the input buffer to the inner writer.
	fn flush_input(&mut self) -> io::Result<()> {
		if self.input_len == 0 {
			return Ok(());
		}

		// SAFETY: We have initialized input_buf[0..input_len] via writes in the write()
		// method. The pointer cast from MaybeUninit<u8> to u8 is valid because
		// MaybeUninit<u8> has the same layout as u8, and we've initialized these
		// bytes.
		let input_slice =
			unsafe { slice::from_raw_parts(self.input_buf.as_ptr().cast::<u8>(), self.input_len) };

		let encoded_len = self.enc.encode_mut(input_slice, &mut self.output_buf);
		let encoded_data = &self.output_buf[..encoded_len];

		// Write the encoded data to the inner writer
		let inner = self.inner.as_mut().expect("EncodeWriter inner is None");
		inner.write_all(encoded_data)?;

		// Reset input buffer
		self.input_len = 0;
		Ok(())
	}
}

impl<W: io::Write, const N: usize> io::Write for EncodeWriter<W, N> {
	fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
		let mut bytes_written = 0;

		while bytes_written < buf.len() {
			let available = INPUT_BUFFER_SIZE - self.input_len;
			if available == 0 {
				self.flush_input()?;
				continue;
			}

			let to_copy = (buf.len() - bytes_written).min(available);

			// SAFETY: We're writing to input_buf[input_len..input_len+to_copy], which is
			// within bounds since input_len + to_copy <= INPUT_BUFFER_SIZE. The
			// source pointer is valid for `to_copy` bytes. We cast from *const u8
			// to *mut u8 for the destination, which is safe because
			// MaybeUninit<u8> has the same layout as u8.
			unsafe {
				let dst = self.input_buf.as_mut_ptr().add(self.input_len).cast::<u8>();
				let src = buf.as_ptr().add(bytes_written);
				core::ptr::copy_nonoverlapping(src, dst, to_copy);
			}

			self.input_len += to_copy;
			bytes_written += to_copy;
		}

		Ok(bytes_written)
	}

	fn flush(&mut self) -> io::Result<()> {
		self.flush_input()?;
		self
			.inner
			.as_mut()
			.expect("EncodeWriter inner is None")
			.flush()
	}
}

impl<W: io::Write, const N: usize> Drop for EncodeWriter<W, N> {
	fn drop(&mut self) {
		let _ = io::Write::flush(self);
	}
}

/// A buffered writer that decodes encoded bytes before writing them to an inner
/// writer.
///
/// This writer buffers encoded characters, decodes them using the specified
/// `Encoding<N>`, and writes the decoded output to the wrapped writer. It
/// implements `io::Write`, allowing transparent decoding of data streams.
///
/// Only complete encoding groups are decoded during normal writes. Incomplete
/// groups are kept in the buffer until flush is called, at which point they are
/// decoded (potentially with padding handling for base64/base32).
///
/// # Examples
/// ```
/// use std::io::Write;
///
/// use omp_core::encoding::{DecodeWriter, base64};
///
/// let mut output = Vec::new();
/// let mut writer = base64::decode_writer(&mut output);
/// writer.write_all(b"SGVsbG8sIFdvcmxkIQ==").unwrap();
/// writer.flush().unwrap();
/// drop(writer);
/// assert_eq!(output, b"Hello, World!");
/// ```
pub struct DecodeWriter<W: io::Write, const N: usize> {
	inner:      Option<W>,
	enc:        Encoding<N>,
	input_buf:  [MaybeUninit<u8>; INPUT_BUFFER_SIZE],
	input_len:  usize,
	output_buf: [u8; OUTPUT_BUFFER_SIZE],
}

impl<W: io::Write, const N: usize> DecodeWriter<W, N> {
	/// Creates a new decoding writer that wraps the given writer.
	pub const fn new(inner: W, enc: Encoding<N>) -> Self {
		Self {
			inner: Some(inner),
			enc,
			input_buf: [MaybeUninit::uninit(); INPUT_BUFFER_SIZE],
			input_len: 0,
			output_buf: [0u8; OUTPUT_BUFFER_SIZE],
		}
	}

	/// Consumes this writer, flushing any buffered data and returning the inner
	/// writer.
	///
	/// # Errors
	/// Returns an error if flushing or decoding fails.
	pub fn into_inner(mut self) -> io::Result<W> {
		io::Write::flush(&mut self)?;
		let inner = self.inner.take().expect("DecodeWriter already consumed");
		// Prevent Drop from running since we're consuming the writer
		core::mem::forget(self);
		Ok(inner)
	}

	/// Decodes and flushes the input buffer to the inner writer.
	///
	/// If `final_flush` is true, all remaining input is decoded, including
	/// incomplete groups. Otherwise, only complete groups are decoded and
	/// incomplete groups are kept in the buffer.
	fn flush_input(&mut self, final_flush: bool) -> io::Result<()> {
		if self.input_len == 0 {
			return Ok(());
		}

		let group_out = self.enc.group_size_out();

		// Determine how many characters to decode
		let to_decode = if final_flush {
			// Decode everything, including incomplete groups
			self.input_len
		} else {
			// Only decode complete groups
			(self.input_len / group_out) * group_out
		};

		if to_decode == 0 {
			return Ok(());
		}

		// SAFETY: We have initialized input_buf[0..to_decode] via writes in the write()
		// method. The pointer cast from MaybeUninit<u8> to u8 is valid because
		// MaybeUninit<u8> has the same layout as u8, and we've initialized these
		// bytes.
		let input_slice =
			unsafe { slice::from_raw_parts(self.input_buf.as_ptr().cast::<u8>(), to_decode) };

		let decoded_len = self
			.enc
			.decode_mut(input_slice, &mut self.output_buf)
			.map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
		let decoded_data = &self.output_buf[..decoded_len];

		// Write the decoded data to the inner writer
		let inner = self.inner.as_mut().expect("DecodeWriter inner is None");
		inner.write_all(decoded_data)?;

		// Move remaining bytes to the start of the buffer
		let remaining = self.input_len - to_decode;
		if remaining > 0 {
			// SAFETY: We're moving initialized bytes within the buffer. The source range
			// [to_decode..input_len] and destination range [0..remaining] don't overlap
			// because remaining < to_decode (when remaining > 0). Both ranges are within
			// INPUT_BUFFER_SIZE.
			unsafe {
				core::ptr::copy(
					self.input_buf.as_ptr().add(to_decode),
					self.input_buf.as_mut_ptr(),
					remaining,
				);
			}
		}
		self.input_len = remaining;

		Ok(())
	}
}

impl<W: io::Write, const N: usize> io::Write for DecodeWriter<W, N> {
	fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
		let mut bytes_written = 0;

		while bytes_written < buf.len() {
			let available = INPUT_BUFFER_SIZE - self.input_len;
			if available == 0 {
				self.flush_input(false)?;
				continue;
			}

			let to_copy = (buf.len() - bytes_written).min(available);

			// SAFETY: We're writing to input_buf[input_len..input_len+to_copy], which is
			// within bounds since input_len + to_copy <= INPUT_BUFFER_SIZE. The
			// source pointer is valid for `to_copy` bytes. We cast from *const u8
			// to *mut u8 for the destination, which is safe because
			// MaybeUninit<u8> has the same layout as u8.
			unsafe {
				let dst = self.input_buf.as_mut_ptr().add(self.input_len).cast::<u8>();
				let src = buf.as_ptr().add(bytes_written);
				core::ptr::copy_nonoverlapping(src, dst, to_copy);
			}

			self.input_len += to_copy;
			bytes_written += to_copy;
		}

		Ok(bytes_written)
	}

	fn flush(&mut self) -> io::Result<()> {
		self.flush_input(true)?;
		self
			.inner
			.as_mut()
			.expect("DecodeWriter inner is None")
			.flush()
	}
}

impl<W: io::Write, const N: usize> Drop for DecodeWriter<W, N> {
	fn drop(&mut self) {
		let _ = io::Write::flush(self);
	}
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

/// Defines a base-N encoding constant and its encode/decode functions.
macro_rules! define_encoding {
    ($mod_name:ident, $n:literal, $pad:literal, $alphabet:expr, $lowercase:literal) => {
         #[doc = concat!(
            stringify!($mod_name), " encoding and decoding.\n\n",
            "This module provides ", stringify!($mod_name), " encoding/decoding using a ",
            stringify!($n), "-character alphabet. Both padded and unpadded variants are available.\n\n",
        )]
        pub mod $mod_name {
            use super::*;

            #[doc = concat!(
                "Encoding dictionary for ", stringify!($mod_name), ".\n\n",
                "Defines the ", stringify!($n), "-character alphabet, padding character, and ",
                "encoding/decoding behavior. Use [`STD`] or [`RAW`] for standard configurations."
            )]
            pub type Encoding = super::Encoding<$n>;

            #[doc = concat!(
                "Streaming decoder for ", stringify!($mod_name), "-encoded data.\n\n",
                "Decodes ", stringify!($mod_name), " characters into bytes. ",
                "Create with [`decode()`] or use [`STD`]/[`RAW`] decoder methods."
            )]
            pub type Decoder<'a> = super::Decoder<'a, $n>;

            #[doc = concat!(
                "Streaming encoder for ", stringify!($mod_name), " encoding.\n\n",
                "Encodes bytes into ", stringify!($mod_name), " characters. ",
                "Create with [`encode()`] or use [`STD`]/[`RAW`] encoder methods."
            )]
            pub type Encoder<'a> = super::Encoder<'a, $n>;

            #[doc = concat!(
                "Standard ", stringify!($mod_name), " encoding with padding.\n\n",
                "Uses a ", stringify!($n), "-character alphabet with `=` padding."
            )]
            pub const STD: Encoding = {
                let enc = Encoding::new($alphabet, Some($pad));
                #[allow(unused_mut, reason = "conditional mutation in const context for lowercase flag")]
                let mut enc = enc;
                if $lowercase {
                    enc.with_lowercase()
                } else {
                    enc
                }
            };

            #[doc = concat!("Padded ", stringify!($mod_name), " encoding (alias of STD).")]
            pub const PADDED: Encoding = STD;

            #[doc = concat!(
                "Raw ", stringify!($mod_name), " encoding without padding.\n\n",
                "Uses a ", stringify!($n), "-character alphabet with no padding characters."
            )]
            pub const RAW: Encoding = {
                let enc = Encoding::new($alphabet, None);
                #[allow(unused_mut, reason = "conditional mutation in const context for lowercase flag")]
                let mut enc = enc;
                if $lowercase {
                    enc.with_lowercase()
                } else {
                    enc
                }
            };

            #[doc = concat!(
                "Encodes a byte slice to ", stringify!($mod_name), " with padding.\n\n",
                "# Arguments\n\n",
                "* `src` - The byte slice to encode\n\n",
                "# Returns\n\n",
                "An iterator that yields encoded bytes.\n\n",
                "# Example\n\n",
                "```\n",
                "use omp_core::", stringify!($mod_name), ";\n",
                "let encoded = ", stringify!($mod_name), "::encode(b\"hello\").into_string();\n",
                "```"
            )]
            pub fn encode<S: AsRef<[u8]> + ?Sized>(src: &S) -> Encoder {
                Encoder::new(src.as_ref(), &STD)
            }

            #[doc = concat!(
                "Decodes ", stringify!($mod_name), " encoded bytes with padding.\n\n",
                "# Arguments\n\n",
                "* `src` - The encoded byte slice to decode\n\n",
                "# Returns\n\n",
                "An iterator that yields decoded bytes.\n\n",
                "# Example\n\n",
                "```\n",
                "use omp_core::", stringify!($mod_name), ";\n",
                "let encoded = ", stringify!($mod_name), "::encode(b\"hello\").into_string();\n",
                "let decoded = ", stringify!($mod_name), "::decode(&encoded).into_vec().unwrap();\n",
                "assert_eq!(decoded, b\"hello\");\n",
                "```"
            )]
            pub fn decode<S: AsRef<[u8]> + ?Sized>(src: &S) -> Decoder {
                Decoder::new(src.as_ref(), &STD)
            }

            #[doc = concat!(
                "Encodes a fixed-size byte array to ", stringify!($mod_name), " with padding.\n\n",
                "# Arguments\n\n",
                "* `src` - The byte array to encode\n\n",
                "# Returns\n\n",
                "An `ArrayStr` containing the encoded output.\n\n",
                "# Example\n\n",
                "```\n",
                "use omp_core::", stringify!($mod_name), ";\n",
                "let data = [0x48u8, 0x65, 0x6c, 0x6c, 0x6f];\n",
                "let encoded = ", stringify!($mod_name), "::encode_n(&data);\n",
                "```"
            )]
            pub const fn encode_n<const N: usize>(src: &[u8; N]) -> ArrayStr<N> {
                STD.encode_n(src)
            }

            #[doc = concat!(
                "Decodes a fixed-size ", stringify!($mod_name), " array with padding.\n\n",
                "# Arguments\n\n",
                "* `src` - The encoded byte array to decode\n\n",
                "# Returns\n\n",
                "A `Result` containing the decoded byte array or an error.\n\n",
                "# Example\n\n",
                "```\n",
                "use omp_core::", stringify!($mod_name), ";\n",
                "let data = [0x48u8, 0x65, 0x6c, 0x6c, 0x6f];\n",
                "let encoded = ", stringify!($mod_name), "::encode(&data).into_vec();\n",
                "let decoded = ", stringify!($mod_name), "::decode(&encoded).into_vec().unwrap();\n",
                "assert_eq!(&decoded, &data);\n",
                "```"
            )]
            pub const fn decode_n<const N: usize>(src: &[u8; N]) -> Option<Array<N>> {
                STD.decode_n(src)
            }

            #[doc = concat!(
                "Encodes a byte slice to ", stringify!($mod_name), " without padding.\n\n",
                "# Arguments\n\n",
                "* `src` - The byte slice to encode\n\n",
                "# Returns\n\n",
                "An iterator that yields encoded bytes without padding characters.\n\n",
                "# Example\n\n",
                "```\n",
                "use omp_core::", stringify!($mod_name), ";\n",
                "let encoded = ", stringify!($mod_name), "::encode_raw(b\"hello\").into_string();\n",
                "```"
            )]
            pub fn encode_raw<S: AsRef<[u8]> + ?Sized>(src: &S) -> Encoder {
                Encoder::new(src.as_ref(), &RAW)
            }

            #[doc = concat!(
                "Decodes ", stringify!($mod_name), " encoded bytes without padding.\n\n",
                "# Arguments\n\n",
                "* `src` - The encoded byte slice to decode\n\n",
                "# Returns\n\n",
                "An iterator that yields decoded bytes.\n\n",
                "# Example\n\n",
                "```\n",
                "use omp_core::", stringify!($mod_name), ";\n",
                "let encoded = ", stringify!($mod_name), "::encode_raw(b\"hello\").into_string();\n",
                "let decoded = ", stringify!($mod_name), "::decode_raw(&encoded).into_vec().unwrap();\n",
                "assert_eq!(decoded, b\"hello\");\n",
                "```"
            )]
            pub fn decode_raw<S: AsRef<[u8]> + ?Sized>(src: &S) -> Decoder {
                Decoder::new(src.as_ref(), &RAW)
            }

            #[doc = concat!(
                "Encodes a fixed-size byte array to ", stringify!($mod_name), " without padding.\n\n",
                "# Arguments\n\n",
                "* `src` - The byte array to encode\n\n",
                "# Returns\n\n",
                "An `ArrayStr` containing the encoded output without padding.\n\n",
                "# Example\n\n",
                "```\n",
                "use omp_core::", stringify!($mod_name), ";\n",
                "let data = [0x48u8, 0x65, 0x6c, 0x6c, 0x6f];\n",
                "let encoded = ", stringify!($mod_name), "::encode_raw_n(&data);\n",
                "```"
            )]
            pub const fn encode_raw_n<const N: usize>(src: &[u8; N]) -> ArrayStr<N> {
                RAW.encode_n(src)
            }

            #[doc = concat!(
                "Decodes a fixed-size ", stringify!($mod_name), " array without padding.\n\n",
                "# Arguments\n\n",
                "* `src` - The encoded byte array to decode\n\n",
                "# Returns\n\n",
                "A `Result` containing the decoded byte array or an error.\n\n",
                "# Example\n\n",
                "```\n",
                "use omp_core::", stringify!($mod_name), ";\n",
                "let data = [0x48u8, 0x65, 0x6c, 0x6c, 0x6f];\n",
                "let encoded = ", stringify!($mod_name), "::encode_raw(&data).into_vec();\n",
                "let decoded = ", stringify!($mod_name), "::decode_raw(&encoded).into_vec().unwrap();\n",
                "assert_eq!(&decoded, &data);\n",
                "```"
            )]
            pub const fn decode_raw_n<const N: usize>(src: &[u8; N]) -> Option<Array<N>> {
                RAW.decode_n(src)
            }

            #[doc = concat!(
                "Encodes bytes into a mutable buffer with padding.\n\n",
                "# Arguments\n\n",
                "* `src` - The byte slice to encode\n",
                "* `dst` - The mutable buffer to write encoded characters to\n\n",
                "# Returns\n\n",
                "The number of characters written to `dst`.\n\n",
                "# Example\n\n",
                "```\n",
                "use omp_core::", stringify!($mod_name), ";\n",
                "let mut buf = [0u8; 16];\n",
                "let n = ", stringify!($mod_name), "::encode_mut(b\"hello\", &mut buf);\n",
                "```"
            )]
            pub const fn encode_mut(src: &[u8], dst: &mut [u8]) -> usize {
                STD.encode_mut(src, dst)
            }

            #[doc = concat!(
                "Decodes ", stringify!($mod_name), " encoded bytes into a mutable buffer with padding.\n\n",
                "# Arguments\n\n",
                "* `src` - The encoded byte slice to decode\n",
                "* `dst` - The mutable buffer to write decoded bytes to\n\n",
                "# Returns\n\n",
                "The number of bytes written to `dst`, or an error if decoding failed.\n\n",
                "# Example\n\n",
                "```\n",
                "use omp_core::", stringify!($mod_name), ";\n",
                "let mut buf = [0u8; 10];\n",
                "let encoded = ", stringify!($mod_name), "::encode(b\"hello\").into_vec();\n",
                "let n = ", stringify!($mod_name), "::decode_mut(&encoded, &mut buf).unwrap();\n",
                "```"
            )]
            pub const fn decode_mut(src: &[u8], dst: &mut [u8]) -> Result<usize> {
                STD.decode_mut(src, dst)
            }

            #[doc = concat!(
                "Encodes bytes into a mutable buffer without padding.\n\n",
                "# Arguments\n\n",
                "* `src` - The byte slice to encode\n",
                "* `dst` - The mutable buffer to write encoded characters to\n\n",
                "# Returns\n\n",
                "The number of characters written to `dst`.\n\n",
                "# Example\n\n",
                "```\n",
                "use omp_core::", stringify!($mod_name), ";\n",
                "let mut buf = [0u8; 16];\n",
                "let n = ", stringify!($mod_name), "::encode_raw_mut(b\"hello\", &mut buf);\n",
                "```"
            )]
            pub const fn encode_raw_mut(src: &[u8], dst: &mut [u8]) -> usize {
                RAW.encode_mut(src, dst)
            }

            #[doc = concat!(
                "Decodes ", stringify!($mod_name), " encoded bytes into a mutable buffer without padding.\n\n",
                "# Arguments\n\n",
                "* `src` - The encoded byte slice to decode\n",
                "* `dst` - The mutable buffer to write decoded bytes to\n\n",
                "# Returns\n\n",
                "The number of bytes written to `dst`, or an error if decoding failed.\n\n",
                "# Example\n\n",
                "```\n",
                "use omp_core::", stringify!($mod_name), ";\n",
                "let mut buf = [0u8; 10];\n",
                "let encoded = ", stringify!($mod_name), "::encode_raw(b\"hello\").into_vec();\n",
                "let n = ", stringify!($mod_name), "::decode_raw_mut(&encoded, &mut buf).unwrap();\n",
                "```"
            )]
            pub const fn decode_raw_mut(src: &[u8], dst: &mut [u8]) -> Result<usize> {
                RAW.decode_mut(src, dst)
            }

            #[doc = concat!(
                "Returns the encoded length including padding.\n\n",
                "Calculates how many ", stringify!($mod_name), " characters (including padding) ",
                "are needed to encode `src_len` bytes.\n\n",
                "# Arguments\n\n",
                "* `src_len` - Number of source bytes\n\n",
                "# Returns\n\n",
                "Number of ", stringify!($mod_name), " characters in padded output\n\n",
                "# Example\n\n",
                "```\n",
                "use omp_core::", stringify!($mod_name), ";\n",
                "// 5 bytes encodes to a specific character count\n",
                "let output_len = ", stringify!($mod_name), "::encode_len(5);\n",
                "let encoded = ", stringify!($mod_name), "::encode(&[0u8; 5]).into_vec();\n",
                "assert_eq!(encoded.len(), output_len);\n",
                "```"
            )]
            #[inline]
            pub const fn encode_len(src_len: usize) -> usize {
                STD.encode_len(src_len)
            }

            #[doc = concat!(
                "Returns the decoded output length for padded input.\n\n",
                "Given the number of non-padding ", stringify!($mod_name), " characters, ",
                "calculates how many bytes will result from decoding.\n\n",
                "# Arguments\n\n",
                "* `src_len` - Number of encoded characters (padding characters not counted)\n\n",
                "# Returns\n\n",
                "Number of decoded bytes\n\n",
                "# Example\n\n",
                "```\n",
                "use omp_core::", stringify!($mod_name), ";\n",
                "// Encoding 5 bytes produces some chars; decoding those chars yields 5 bytes\n",
                "let data = [0u8; 5];\n",
                "let encoded = ", stringify!($mod_name), "::encode(&data).into_vec();\n",
                "let non_padding_len = encoded.iter().position(|&b| b == b'=').unwrap_or(encoded.len());\n",
                "assert_eq!(", stringify!($mod_name), "::decode_len(non_padding_len), 5);\n",
                "```"
            )]
            #[inline]
            pub const fn decode_len(src_len: usize) -> usize {
                STD.decode_len(src_len)
            }

            #[doc = concat!(
                "Returns the encoded length without padding.\n\n",
                "Calculates how many ", stringify!($mod_name), " characters (no padding) ",
                "are needed to encode `src_len` bytes.\n\n",
                "# Arguments\n\n",
                "* `src_len` - Number of source bytes\n\n",
                "# Returns\n\n",
                "Number of ", stringify!($mod_name), " characters in unpadded output\n\n",
                "# Example\n\n",
                "```\n",
                "use omp_core::", stringify!($mod_name), ";\n",
                "// 5 bytes encodes to a specific character count (no padding)\n",
                "let output_len = ", stringify!($mod_name), "::encode_raw_len(5);\n",
                "let encoded = ", stringify!($mod_name), "::encode_raw(&[0u8; 5]).into_vec();\n",
                "assert_eq!(encoded.len(), output_len);\n",
                "```"
            )]
            #[inline]
            pub const fn encode_raw_len(src_len: usize) -> usize {
                RAW.encode_len(src_len)
            }

            #[doc = concat!(
                "Returns the decoded output length for unpadded input.\n\n",
                "Given the number of ", stringify!($mod_name), " characters (no padding), ",
                "calculates how many bytes will result from decoding.\n\n",
                "# Arguments\n\n",
                "* `src_len` - Number of unpadded encoded characters\n\n",
                "# Returns\n\n",
                "Number of decoded bytes\n\n",
                "# Example\n\n",
                "```\n",
                "use omp_core::", stringify!($mod_name), ";\n",
                "// Encoding 5 bytes without padding produces chars; decoding yields 5 bytes\n",
                "let data = [0u8; 5];\n",
                "let encoded = ", stringify!($mod_name), "::encode_raw(&data).into_vec();\n",
                "assert_eq!(", stringify!($mod_name), "::decode_raw_len(encoded.len()), 5);\n",
                "```"
            )]
            #[inline]
            pub const fn decode_raw_len(src_len: usize) -> usize {
                RAW.decode_len(src_len)
            }

            #[doc = concat!(
                "Creates an encoding writer that wraps an `io::Write`.\\n\\n",
                "The writer buffers raw bytes, encodes them to ", stringify!($mod_name), " with padding, ",
                "and writes the encoded output to the inner writer.\\n\\n",
                "# Examples\\n\\n",
                "```\\n",
                "use omp_core::", stringify!($mod_name), ";\\n",
                "use std::io::Write;\\n",
                "\\n",
                "let mut output = Vec::new();\\n",
                "let mut writer = ", stringify!($mod_name), "::encode_writer(&mut output);\\n",
                "writer.write_all(b\\\"Hello\\\").unwrap();\\n",
                "writer.flush().unwrap();\\n",
                "```\\n"
            )]
            pub const fn encode_writer<W: std::io::Write>(inner: W) -> super::EncodeWriter<W, $n> {
                super::EncodeWriter::new(inner, STD)
            }

            #[doc = concat!(
                "Creates a decoding writer that wraps an `io::Write`.\\n\\n",
                "The writer buffers ", stringify!($mod_name), " encoded bytes with padding, ",
                "decodes them, and writes the raw output to the inner writer.\\n\\n",
                "# Examples\\n\\n",
                "```\\n",
                "use omp_core::", stringify!($mod_name), ";\\n",
                "use std::io::Write;\\n",
                "\\n",
                "let mut output = Vec::new();\\n",
                "let encoded = ", stringify!($mod_name), "::encode(b\\\"Hello\\\").into_vec();\\n",
                "let mut writer = ", stringify!($mod_name), "::decode_writer(&mut output);\\n",
                "writer.write_all(&encoded).unwrap();\\n",
                "writer.flush().unwrap();\\n",
                "assert_eq!(output, b\\\"Hello\\\");\\n",
                "```\\n"
            )]
            pub const fn decode_writer<W: std::io::Write>(inner: W) -> super::DecodeWriter<W, $n> {
                super::DecodeWriter::new(inner, STD)
            }

            #[doc = concat!(
                "Creates an encoding writer for unpadded ", stringify!($mod_name), ".\\n\\n",
                "The writer buffers raw bytes, encodes them to ", stringify!($mod_name), " without padding, ",
                "and writes the encoded output to the inner writer.\\n\\n",
                "# Examples\\n\\n",
                "```\\n",
                "use omp_core::", stringify!($mod_name), ";\\n",
                "use std::io::Write;\\n",
                "\\n",
                "let mut output = Vec::new();\\n",
                "let mut writer = ", stringify!($mod_name), "::encode_writer_raw(&mut output);\\n",
                "writer.write_all(b\\\"Hello\\\").unwrap();\\n",
                "writer.flush().unwrap();\\n",
                "```\\n"
            )]
            pub const fn encode_writer_raw<W: std::io::Write>(inner: W) -> super::EncodeWriter<W, $n> {
                super::EncodeWriter::new(inner, RAW)
            }

            #[doc = concat!(
                "Creates a decoding writer for unpadded ", stringify!($mod_name), ".\\n\\n",
                "The writer buffers ", stringify!($mod_name), " encoded bytes without padding, ",
                "decodes them, and writes the raw output to the inner writer.\\n\\n",
                "# Examples\\n\\n",
                "```\\n",
                "use omp_core::", stringify!($mod_name), ";\\n",
                "use std::io::Write;\\n",
                "\\n",
                "let mut output = Vec::new();\\n",
                "let encoded = ", stringify!($mod_name), "::encode_raw(b\\\"Hello\\\").into_vec();\\n",
                "let mut writer = ", stringify!($mod_name), "::decode_writer_raw(&mut output);\\n",
                "writer.write_all(&encoded).unwrap();\\n",
                "writer.flush().unwrap();\\n",
                "assert_eq!(output, b\\\"Hello\\\");\\n",
                "```\\n"
            )]
            pub const fn decode_writer_raw<W: std::io::Write>(inner: W) -> super::DecodeWriter<W, $n> {
                super::DecodeWriter::new(inner, RAW)
            }
        }
    };
}

// Base64 variants
define_encoding!(
	base64,
	64,
	b'=',
	b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/",
	false
);
define_encoding!(
	base64_url,
	64,
	b'=',
	b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_",
	false
);

// Base32 variants
define_encoding!(base32, 32, b'=', b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567", false);
define_encoding!(base32_hex, 32, b'=', b"0123456789ABCDEFGHIJKLMNOPQRSTUV", false);
define_encoding!(base32_dns, 32, b'=', b"0123456789abcdefghijklmnopqrstuv", false);
