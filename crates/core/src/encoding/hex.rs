//! Hexadecimal encoding and decoding with streaming support.
//!
//! Provides upper/lowercase hex codecs with chunked encoding, streaming
//! decoders, and utilities for formatting and parsing hex strings.

use core::{mem::MaybeUninit, slice, str};
use std::{cmp::Ordering, fmt, io, iter::FusedIterator};

use bytes::{BufMut, Bytes};

use super::fixed_arr::serialize;
pub use super::{Array, ArrayStr, DecodeError, Result};

// ============================================================================
// HEX DECODER
// ============================================================================

/// A streaming hex decoder that converts hex characters to bytes.
///
/// Supports both even and odd-length hex strings. For odd-length strings,
/// the first character is treated as a single nibble (0x0-0xF).
///
/// # Examples
/// ```
/// use omp_core::hex::Decoder;
/// let hex = b"48656c6c6f"; // "Hello"
/// let decoded = Decoder::new(hex).into_vec().unwrap();
/// assert_eq!(decoded, b"Hello");
/// ```
#[derive(Debug, Clone)]
pub struct Decoder<'a> {
	/// Remaining single nibble for odd-length inputs
	rem: Option<u8>,
	/// Source iterator of hex characters
	src: &'a [[u8; 2]],
}

impl<'a> Decoder<'a> {
	/// Creates a new hex decoder from an iterator of bytes (ASCII hex
	/// characters).
	///
	/// If the input has odd length, the first character is consumed and stored
	/// as a single nibble to be emitted first.
	pub const fn new(src: &'a [u8]) -> Self {
		let (rem, src) = src.as_rchunks();
		Self { rem: rem.first().copied(), src }
	}

	/// Skips the `0x` or `0X` prefix if present at the current position.
	///
	/// Consumes the decoder, skips the prefix if found, and returns a new
	/// decoder. This method clones the iterator to peek ahead for prefix
	/// detection.
	///
	/// # Examples
	/// ```
	/// use omp_core::hex::Decoder;
	/// let hex = b"0x48656c6c6f";
	/// let result = Decoder::new(hex).skip_0x().into_vec().unwrap();
	/// assert_eq!(result, b"Hello");
	/// ```
	pub const fn skip_0x(self) -> Self {
		match (self.rem, self.src) {
			(Some(b'0'), [[b'x' | b'X', x], rest @ ..]) => Self { rem: Some(*x), src: rest },
			(None, [[b'0', b'x' | b'X'], rest @ ..]) => Self { rem: None, src: rest },
			_ => self,
		}
	}

	/// Skips leading zero characters from the current position.
	///
	/// Consumes the decoder, skips all leading '0' characters, and returns a
	/// new decoder.
	///
	/// # Examples
	/// ```
	/// use omp_core::hex::Decoder;
	/// let hex = b"000048656c6c6f";
	/// let result = Decoder::new(hex).skip_leading_zeros().into_vec().unwrap();
	/// assert_eq!(result, b"Hello");
	/// ```
	pub const fn skip_leading_zeros(mut self) -> Self {
		if let Some(v) = self.rem {
			if v == b'0' {
				self.rem = None;
			} else {
				return self;
			}
		}

		loop {
			match self.src {
				[[b'0', b'0'], rest @ ..] => {
					self.src = rest;
					continue;
				},
				[[b'0', x], rest @ ..] => {
					self.rem = Some(*x);
					self.src = rest;
				},
				_ => {},
			}
			break self;
		}
	}

	/// Collects the decoded bytes into a `Vec<u8>`.
	///
	/// Pre-allocates capacity based on the known decoded length.
	pub fn into_vec(self) -> Result<Vec<u8>> {
		let len = self.len();
		let mut buf = Vec::<u8>::with_capacity(len);
		let base = buf.spare_capacity_mut();
		let mut di = 0;

		// Handle odd-length prefix
		if let Some(rem) = self.rem {
			let n = parse_nibble(rem).ok_or(DecodeError::InvalidCharacter(rem))?;
			// SAFETY: di=0 is always in bounds since len = rem.is_some() + src.len() >= 1
			unsafe { base.get_unchecked_mut(di) }.write(n);
			di += 1;
		}

		// HOT LOOP: Process 8 pairs at a time with minimal branching
		let mut si = 0;
		while si + 8 <= self.src.len() {
			for _ in 0..8 {
				let b = parse_byte(self.src[si])?;
				// SAFETY: di < len is guaranteed by allocation and loop invariant
				unsafe { base.get_unchecked_mut(di) }.write(b);
				di += 1;
				si += 1;
			}
		}

		// Process remaining pairs
		for &[h, l] in &self.src[si..] {
			let b = parse_byte([h, l])?;
			// SAFETY: di < len is guaranteed by allocation
			unsafe { base.get_unchecked_mut(di) }.write(b);
			di += 1;
		}

		debug_assert_eq!(di, len);
		// SAFETY: We've initialized exactly `len` bytes via MaybeUninit::write:
		// one nibble if rem was Some, then one byte per element in src.
		// All writes succeeded (no error returns), so all bytes are valid.
		unsafe { buf.set_len(len) };
		Ok(buf)
	}

	/// Collects the decoded bytes into a `Bytes`.
	pub fn into_bytes(self) -> Result<Bytes> {
		self.into_vec().map(Bytes::from)
	}

	/// Collects the decoded bytes into a slice.
	pub fn into_slice(mut self, mut buf: &mut [u8]) -> Result<usize> {
		// Handle odd-length prefix
		let mut n = if let Some(rem) = self.rem {
			let Some(d) = buf.split_off_first_mut() else {
				return Ok(0);
			};
			*d = parse_nibble(rem).ok_or(DecodeError::InvalidCharacter(rem))?;
			1
		} else {
			0
		};

		// Process pairs
		while let Some(d) = buf.split_off_first_mut()
			&& let Some(&hl) = self.src.split_off_first()
		{
			*d = parse_byte(hl)?;
			n += 1;
		}
		Ok(n)
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
	///
	/// Efficiently writes decoded bytes directly into the buffer's
	/// uninitialized memory.
	#[inline]
	pub fn into_buf<B: BufMut>(mut self, mut buf: B) -> Result<B> {
		loop {
			let mut n = 0;
			// SAFETY: BufMut::chunk_mut() returns a valid uninit slice within the buffer's
			// capacity. We only write to these bytes via MaybeUninit::write, never read
			// them.
			let chunk = unsafe { buf.chunk_mut().as_uninit_slice_mut() };
			for (b, dst) in self.by_ref().zip(&mut *chunk) {
				dst.write(b?);
				n += 1;
			}
			let exhausted = n < chunk.len();

			// SAFETY: We have initialized exactly `n` bytes via MaybeUninit::write in the
			// loop above. BufMut's contract requires we advance by the number of bytes
			// written.
			unsafe { buf.advance_mut(n) };

			// If we didn't fill the buffer, we've exhausted the source
			if exhausted {
				break Ok(buf);
			}
		}
	}

	/// Extends an existing collection with the decoded bytes.
	///
	/// Returns the number of bytes decoded and added to the collection.
	pub fn extend_into<E: Extend<u8> + ?Sized>(self, buf: &mut E) -> Result<usize> {
		buf.extend_reserve(self.len());
		let mut n = 0;
		for byte in self {
			let byte = byte?;
			buf.extend_one(byte);
			n += 1;
		}
		Ok(n)
	}

	/// Writes the decoded bytes to an `io::Write`.
	///
	/// Returns the number of bytes written.
	pub fn write_into<W: io::Write + ?Sized>(mut self, writer: &mut W) -> io::Result<usize> {
		let mut buf = [MaybeUninit::<u8>::uninit(); 512];
		let mut n = 0;
		let mut done = false;
		while !done {
			let mut i = 0;
			for dst in &mut buf {
				if let Some(b) = self.next() {
					dst.write(b.map_err(io::Error::other)?);
					i += 1;
				} else {
					done = true;
					break;
				}
			}
			n += i;

			if i != 0 {
				// SAFETY: We've initialized exactly buf[0..i] via MaybeUninit::write in the
				// loop above. Casting the pointer from MaybeUninit<u8> to u8 is valid for the
				// first `i` elements, and slice::from_raw_parts creates a valid &[u8] view.
				unsafe { writer.write_all(slice::from_raw_parts(buf.as_ptr().cast(), i))? };
			}
		}
		Ok(n)
	}
}

impl Iterator for Decoder<'_> {
	type Item = Result<u8>;

	#[inline]
	fn next(&mut self) -> Option<Self::Item> {
		// Handle odd-length source: emit single nibble first
		if let Some(s0) = self.rem.take() {
			return Some(parse_nibble(s0).ok_or(DecodeError::InvalidCharacter(s0)));
		}

		// Process two hex characters into one byte
		match parse_byte(*self.src.split_off_first()?) {
			Ok(b) => Some(Ok(b)),
			Err(e) => Some(Err(e)),
		}
	}

	#[inline]
	fn size_hint(&self) -> (usize, Option<usize>) {
		let n = self.len();
		(n, Some(n))
	}
}

impl DoubleEndedIterator for Decoder<'_> {
	#[inline]
	fn next_back(&mut self) -> Option<Self::Item> {
		// Try to get two characters from the back
		if let Some(x) = self.src.split_off_last() {
			return match parse_byte(*x) {
				Ok(b) => Some(Ok(b)),
				Err(e) => Some(Err(e)),
			};
		}

		// If no pairs left, return the single nibble if present
		if let Some(s0) = self.rem.take() {
			return Some(parse_nibble(s0).ok_or(DecodeError::InvalidCharacter(s0)));
		}

		None
	}
}

impl ExactSizeIterator for Decoder<'_> {
	#[inline]
	fn len(&self) -> usize {
		self.src.len() + usize::from(self.rem.is_some())
	}
}

impl FusedIterator for Decoder<'_> {}

impl<const N: usize> TryFrom<Decoder<'_>> for [u8; N] {
	type Error = DecodeError;

	fn try_from(decoder: Decoder<'_>) -> Result<Self> {
		decoder.into_array()
	}
}

impl fmt::Display for Decoder<'_> {
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

impl PartialEq<[u8]> for Decoder<'_> {
	fn eq(&self, other: &[u8]) -> bool {
		self.clone().eq(other.iter().map(|&b| Ok(b)))
	}
}

impl PartialOrd<[u8]> for Decoder<'_> {
	fn partial_cmp(&self, other: &[u8]) -> Option<Ordering> {
		Some(self.clone().cmp(other.iter().map(|&b| Ok(b))))
	}
}

/// Decodes a hex string in-place, writing to a destination buffer.
///
/// Returns the number of bytes written.
/// This is a const function suitable for compile-time evaluation.
///
/// # Errors
/// Returns `Error::InvalidCharacter` if any invalid hex character is
/// encountered.
pub const fn decode_mut(src: &[u8], dst: &mut [u8]) -> Result<usize> {
	let mut i = 0;
	let mut si = src;

	// Handle odd-length source: decode single nibble first
	if let [s0, sn @ ..] = si
		&& sn.len() & 1 == 0
		&& dst.len() > i
	{
		let Some(s0) = parse_nibble(*s0) else {
			return Err(DecodeError::InvalidCharacter(*s0));
		};
		dst[i] = s0;
		si = sn;
		i += 1;
	}

	// Process pairs of hex characters
	while let [c0, c1, sn @ ..] = si
		&& dst.len() > i
	{
		let c0v = *c0;
		let c1v = *c1;
		dst[i] = match parse_byte([c0v, c1v]) {
			Ok(b) => b,
			Err(e) => return Err(e),
		};
		si = sn;
		i += 1;
	}

	Ok(i)
}

/// Decodes a hex string into a vector of bytes.
///
/// # Examples
/// ```
/// use omp_core::hex::decode;
/// let result = decode(b"48656c6c6f").into_vec().unwrap();
/// assert_eq!(result, b"Hello");
/// ```
#[inline]
pub fn decode<T: AsRef<[u8]> + ?Sized>(src: &T) -> Decoder<'_> {
	Decoder::new(src.as_ref())
}

// ============================================================================
// HEX PREFIX UTILITIES
// ============================================================================

/// Skips the `0x` or `0X` prefix from a byte slice if present.
///
/// # Examples
/// ```
/// use omp_core::hex;
/// let result = hex::Decoder::new(hex::skip_0x(b"0x48656c6c6f"))
/// 	.into_vec()
/// 	.unwrap();
/// assert_eq!(result, b"Hello");
///
/// // No prefix - returns original slice
/// assert_eq!(hex::skip_0x(b"48656c6c6f"), b"48656c6c6f");
/// ```
#[inline]
pub const fn skip_0x(src: &[u8]) -> &[u8] {
	match src {
		[b'0', b'x' | b'X', rest @ ..] => rest,
		_ => src,
	}
}

/// Skips leading zero characters from a byte slice.
///
/// # Examples
/// ```
/// use omp_core::hex;
/// let result = hex::Decoder::new(hex::skip_leading_zeros(b"000048656c6c6f"))
/// 	.into_vec()
/// 	.unwrap();
/// assert_eq!(result, b"Hello");
///
/// // All zeros - returns empty slice
/// assert_eq!(hex::skip_leading_zeros(b"0000"), b"");
///
/// // No leading zeros - returns original slice
/// assert_eq!(hex::skip_leading_zeros(b"48656c6c6f"), b"48656c6c6f");
/// ```
#[inline]
pub const fn skip_leading_zeros(mut src: &[u8]) -> &[u8] {
	while let [b'0', rest @ ..] = src {
		src = rest;
	}
	src
}

// ============================================================================
// HEX PARSING UTILITIES
// ============================================================================

/// Lookup table for fast hex character to nibble conversion.
/// Maps ASCII characters to nibble values (0-15), with invalid chars mapped to
/// 0x80.
const DEC_TABLE: [u8; 0x100] = {
	let mut table = [0x80; 0x100];
	let mut i = 0;

	// Map '0'-'9' to 1-10 (actual values 0-9)
	while i <= 0xf {
		let c = char::from_digit(i as u32, 0x10).unwrap();
		table[c.to_ascii_lowercase() as usize] = i;
		table[c.to_ascii_uppercase() as usize] = i;
		i += 1;
	}

	table
};

/// Safely parses a hex character into a nibble (0-15).
///
/// Returns `None` for invalid hex characters.
///
/// # Examples
/// ```
/// use omp_core::hex::parse_nibble;
/// assert_eq!(parse_nibble(b'A'), Some(10));
/// assert_eq!(parse_nibble(b'f'), Some(15));
/// assert_eq!(parse_nibble(b'0'), Some(0));
/// assert_eq!(parse_nibble(b'g'), None);
/// ```
#[inline]
pub const fn parse_nibble(b: u8) -> Option<u8> {
	let v = DEC_TABLE[b as usize];
	if v.cast_signed() >= 0 {
		Some(v)
	} else {
		std::hint::cold_path();
		None
	}
}

/// Parses two hex characters into a byte.
///
/// # Examples
/// ```
/// use omp_core::hex::parse_byte;
/// assert_eq!(parse_byte([b'4', b'8']).unwrap(), 0x48);
/// assert_eq!(parse_byte([b'f', b'f']).unwrap(), 0xff);
/// ```
#[inline]
pub const fn parse_byte([h, l]: [u8; 2]) -> Result<u8> {
	let hv = DEC_TABLE[h as usize];
	let lv = DEC_TABLE[l as usize];

	if (hv | lv).cast_signed() >= 0 {
		Ok((hv << 4) | lv)
	} else {
		std::hint::cold_path();
		let inv = if hv.cast_signed() >= 0 { l } else { h };
		Err(DecodeError::InvalidCharacter(inv))
	}
}

// ============================================================================
// HEX ENCODER
// ============================================================================

const ALPHABET: [u8; 32] = *b"0123456789abcdef0123456789ABCDEF";

/// Byte->pair LUT for lowercase hex (256 bytes -> 256 u16 pairs)
const LUT: [[u16; 256]; 2] = {
	let mut t = [[0u16; 256]; 2];
	{
		let t = t[0].as_mut_slice();
		let mut i = 0u16;
		while i < 256 {
			let b = (i & 0xff) as u8;
			let h = LOWER.encode_nibble(b >> 4);
			let l = LOWER.encode_nibble(b & 0x0f);
			t[i as usize] = u16::from_ne_bytes([h, l]);
			i += 1;
		}
	}
	{
		let t = t[1].as_mut_slice();
		let mut i = 0u16;
		while i < 256 {
			let b = (i & 0xff) as u8;
			let h = UPPER.encode_nibble(b >> 4);
			let l = UPPER.encode_nibble(b & 0x0f);
			t[i as usize] = u16::from_ne_bytes([h, l]);
			i += 1;
		}
	}
	t
};

/// Character set for hex encoding (uppercase or lowercase).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Encoding {
	/// Lowercase hex digits (a-f)
	Lowercase = 0,
	/// Uppercase hex digits (A-F)
	Uppercase = 1,
}

/// Standard hex encoding.
pub const STD: Encoding = Encoding::Lowercase;

/// Lowercase hex digits (a-f)
pub const LOWER: Encoding = Encoding::Lowercase;

/// Uppercase hex digits (A-F)
pub const UPPER: Encoding = Encoding::Uppercase;

impl Encoding {
	/// Converts a nibble (0-15) to a hex character using branchless arithmetic.
	///
	/// # Examples
	/// ```
	/// use omp_core::hex::{LOWER, UPPER};
	/// assert_eq!(LOWER.encode_nibble(10), b'a');
	/// assert_eq!(UPPER.encode_nibble(15), b'F');
	/// ```
	#[inline]
	pub const fn encode_nibble(self, nibble: u8) -> u8 {
		let idx = (((self as usize) << 4) + (nibble as usize)) & 31;
		ALPHABET[idx]
	}

	/// Encodes bytes to hex into a flat byte buffer.
	///
	/// Writes ASCII hex characters to `dst` and returns the number of bytes
	/// written. The output length is `min(dst.len(), 2 * src.len())`.
	///
	/// # Examples
	/// ```
	/// use omp_core::hex;
	/// let mut buf = [0u8; 10];
	/// let n = hex::LOWER.encode_mut(b"Hello", &mut buf);
	/// assert_eq!(n, 10);
	/// assert_eq!(&buf, b"48656c6c6f");
	/// ```
	#[inline]
	pub const fn encode_mut(self, src: &[u8], dst: &mut [u8]) -> usize {
		let lut = self.lut();
		let mut n = dst.len() >> 1;
		if n > src.len() {
			n = src.len();
		}

		let src = src.as_ptr();
		let dst = dst.as_mut_ptr();
		let mut i = 0;
		while i < n {
			// SAFETY: we only write within `n * 2` bytes, which is <= dst.len()/2 as well
			// as <= src.len()
			unsafe {
				dst.add(i << 1)
					.cast::<u16>()
					.write_unaligned(lut[src.add(i).read() as usize]);
			}
			i += 1;
		}
		n << 1
	}

	/// Encodes a single byte to a hex pair.
	///
	/// # Examples
	/// ```
	/// use omp_core::hex::{LOWER, UPPER};
	/// assert_eq!(LOWER.encode_byte(0x48), *b"48");
	/// assert_eq!(UPPER.encode_byte(0xff), *b"FF");
	/// ```
	#[inline]
	pub const fn encode_byte(self, byte: u8) -> [u8; 2] {
		LUT[self as usize][byte as usize].to_ne_bytes()
	}

	/// Encodes a byte array to a hex string at compile time.
	#[inline]
	pub const fn encode_n<const N: usize>(self, src: &[u8; N]) -> ArrayStr<N> {
		let mut out = [[0u8; 2]; N];
		self.encode_mut(src.as_slice(), out.as_flattened_mut());
		ArrayStr::new(out, N * 2)
	}

	/// Returns the lookup table for this charset.
	#[inline]
	pub const fn lut(self) -> &'static [u16; 256] {
		&LUT[self as usize]
	}
}

/// Iterator that encodes bytes as individual hex ASCII characters.
///
/// Each input byte produces two output bytes (hex characters).
/// Maintains state for bidirectional iteration.
///
/// # Examples
/// ```
/// use omp_core::hex::Encoder;
/// let data = b"Hi";
/// let chars: Vec<u8> = Encoder::new(data).collect();
/// assert_eq!(chars, b"4869");
/// ```
#[derive(Debug, Clone)]
pub struct Encoder<'a> {
	/// Source iterator of bytes to encode
	src:     &'a [u8],
	/// Output charset (uppercase or lowercase)
	charset: Encoding,
	/// Pending low nibble from forward iteration
	low:     Option<u8>,
	/// Pending high nibble from backward iteration
	high:    Option<u8>,
}

impl<'a> From<&'a [u8]> for Encoder<'a> {
	fn from(src: &'a [u8]) -> Self {
		Self { src, charset: LOWER, low: None, high: None }
	}
}

impl<'a> Encoder<'a> {
	/// Creates a new ASCII encoder from an iterator of bytes.
	pub fn new(src: &'a [u8]) -> Self {
		src.into()
	}

	/// Sets the encoder to lowercase mode.
	pub const fn lower(mut self) -> Self {
		self.charset = LOWER;
		self
	}

	/// Sets the encoder to uppercase mode.
	pub const fn upper(mut self) -> Self {
		self.charset = UPPER;
		self
	}

	/// Sets the encoder charset.
	pub const fn with_charset(mut self, charset: Encoding) -> Self {
		self.charset = charset;
		self
	}

	/// Converts to a `CharEncoder` that yields `char` instead of `u8`.
	#[define_opaque(CharEncoder)]
	pub fn into_chars(self) -> CharEncoder<'a> {
		self.map(|x| x as char)
	}

	/// Collects into a `Vec<u8>`.
	pub fn into_vec(self) -> Vec<u8> {
		let lut = self.charset.lut();
		let out_len = self.len();
		let mut buf = Vec::<u8>::with_capacity(out_len);
		if let Some(low) = self.low {
			buf.push(self.charset.encode_nibble(low));
		}
		let pairs_end = buf.len() + 2 * self.src.len();
		let base = buf.spare_capacity_mut();
		for (i, &byte) in self.src.iter().enumerate() {
			// SAFETY: We allocated `out_len >= pairs_end` capacity. Each src byte
			// at index `i` writes the u16 at offset `i` in spare capacity, staying
			// within bounds. Writes don't overlap since i is unique.
			unsafe {
				base
					.as_mut_ptr()
					.cast::<u16>()
					.add(i)
					.write_unaligned(lut[byte as usize]);
			};
		}
		// SAFETY: bytes [len, pairs_end) were fully initialized by the loop
		// above; bytes below len were initialized by the optional push.
		unsafe { buf.set_len(pairs_end) };
		// Publish the pairs BEFORE appending the pending high nibble: push
		// writes at the current length, so it must come after set_len.
		if let Some(high) = self.high {
			buf.push(self.charset.encode_nibble(high));
		}
		debug_assert_eq!(buf.len(), out_len);
		buf
	}

	/// Collects into a `Bytes`.
	pub fn into_bytes(self) -> Bytes {
		Bytes::from(self.into_vec())
	}

	/// Collects into a String.
	pub fn into_string(self) -> String {
		super::ascii_to_str_owned(self.into_vec())
	}

	/// Extends into an existing buffer.
	pub fn extend_into<E: Extend<u8> + ?Sized>(self, buf: &mut E) {
		buf.extend(self);
	}

	/// Collects the encoded output into a `BufMut`.
	pub fn into_buf<B: BufMut>(mut self, mut buf: B) -> B {
		loop {
			let mut n = 0;
			// SAFETY: BufMut::chunk_mut() returns a valid uninit slice within the buffer's
			// capacity. We only write to these bytes via MaybeUninit::write, never read
			// them.
			let chunk = unsafe { buf.chunk_mut().as_uninit_slice_mut() };
			for (b, d) in self.by_ref().zip(&mut *chunk) {
				d.write(b);
				n += 1;
			}
			let exhausted = n < chunk.len();

			// SAFETY: We have initialized exactly `n` bytes via MaybeUninit::write in the
			// loop above. BufMut's contract requires we advance by the number of bytes
			// written.
			unsafe { buf.advance_mut(n) };

			// If we didn't fill the buffer, we've exhausted the source
			if exhausted {
				break buf;
			}
		}
	}

	/// Writes to an `io::Write`.
	pub fn write_into<W: io::Write + ?Sized>(self, writer: &mut W) -> io::Result<usize> {
		let Self { src: mut it, charset, low, mut high } = self;

		let mut n = 0;

		let mut buf = MaybeUninit::<[[u8; 2]; 64]>::uninit().transpose();
		if let Some(low) = low {
			buf[0].write([0, charset.encode_nibble(low)]);
			n += 1;

			for d in &mut buf[1..] {
				let Some(&b) = it.split_off_first() else {
					break;
				};
				d.write(charset.encode_byte(b));
				n += 2;
			}

			// SAFETY: The loop initialized the flattened byte range 1..=n. The
			// range is in bounds because `buf` holds 128 bytes and `n <= 127`.
			let data = unsafe { slice::from_raw_parts(buf.as_ptr().cast::<u8>().add(1), n) };
			writer.write_all(data)?;
		}

		while !it.is_empty() {
			let mut local = 0;
			for d in &mut buf {
				let Some(&b) = it.split_off_first() else {
					if let Some(hi) = high.take() {
						// Counted via `local` so the flattened write below includes it.
						d.write([charset.encode_nibble(hi), 0]);
						local += 1;
					}
					break;
				};
				d.write(charset.encode_byte(b));
				local += 2;
			}
			// SAFETY: The loop initialized the first `local` flattened bytes, and
			// `local` cannot exceed the 128-byte capacity of `buf`.
			let data = unsafe { slice::from_raw_parts(buf.as_ptr().cast::<u8>(), local) };
			writer.write_all(data)?;
			n += local;
		}

		if let Some(high) = high {
			writer.write_all(&[charset.encode_nibble(high)])?;
			n += 1;
		}
		Ok(n)
	}

	/// Writes to a `fmt::Write`.
	pub fn format_into<W: fmt::Write + ?Sized>(self, writer: &mut W) -> fmt::Result {
		for bytes in self {
			writer.write_str(super::ascii_to_str(&[bytes]))?;
		}
		Ok(())
	}
}

impl From<Encoder<'_>> for String {
	fn from(encoder: Encoder<'_>) -> Self {
		encoder.into_string()
	}
}

impl From<Encoder<'_>> for Bytes {
	fn from(encoder: Encoder<'_>) -> Self {
		encoder.into_bytes()
	}
}

impl From<Encoder<'_>> for Vec<u8> {
	fn from(encoder: Encoder<'_>) -> Self {
		encoder.into_vec()
	}
}

impl Iterator for Encoder<'_> {
	type Item = u8;

	fn next(&mut self) -> Option<Self::Item> {
		// If we have a pending low nibble, emit it
		if let Some(low) = self.low.take() {
			return Some(self.charset.encode_nibble(low));
		}

		// If we can read from source, get next byte
		if let Some(byte) = self.src.split_off_first() {
			let byte = *byte;
			let high = byte >> 4;
			let low = byte & 0x0f;
			self.low = Some(low);
			return Some(self.charset.encode_nibble(high));
		}

		// If we have a pending high nibble, emit it
		if let Some(high) = self.high.take() {
			return Some(self.charset.encode_nibble(high));
		}

		None
	}

	fn size_hint(&self) -> (usize, Option<usize>) {
		let n = self.len();
		(n, Some(n))
	}
}

impl DoubleEndedIterator for Encoder<'_> {
	fn next_back(&mut self) -> Option<Self::Item> {
		// If we have a pending high nibble, emit it
		if let Some(high) = self.high.take() {
			return Some(self.charset.encode_nibble(high));
		}

		// If we can read from source, get next byte
		if let Some(byte) = self.src.split_off_last() {
			let byte = *byte;
			let high = byte >> 4;
			let low = byte & 0x0f;
			self.high = Some(high);
			return Some(self.charset.encode_nibble(low));
		}

		// If we have a pending low nibble, emit it
		if let Some(low) = self.low.take() {
			return Some(self.charset.encode_nibble(low));
		}

		None
	}
}

impl ExactSizeIterator for Encoder<'_> {
	fn len(&self) -> usize {
		let rem = self.low.is_some() as usize + self.high.is_some() as usize;
		(self.src.len() << 1) + rem
	}
}

impl FusedIterator for Encoder<'_> {}

impl fmt::Display for Encoder<'_> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		super::format_with_precision(self.clone(), f)
	}
}

impl fmt::LowerHex for Encoder<'_> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		fmt::Display::fmt(&self.clone().lower(), f)
	}
}

impl fmt::UpperHex for Encoder<'_> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		fmt::Display::fmt(&self.clone().upper(), f)
	}
}

impl<'b> PartialEq<Encoder<'b>> for Encoder<'_> {
	fn eq(&self, other: &Encoder<'b>) -> bool {
		self.clone().eq(other.clone())
	}
}

impl Ord for Encoder<'_> {
	fn cmp(&self, other: &Self) -> Ordering {
		self.clone().cmp(other.clone())
	}
}

impl<'b> PartialOrd<Encoder<'b>> for Encoder<'_> {
	fn partial_cmp(&self, other: &Encoder<'b>) -> Option<Ordering> {
		self.clone().partial_cmp(other.clone())
	}
}

impl Eq for Encoder<'_> {}

impl PartialEq<[u8]> for Encoder<'_> {
	fn eq(&self, other: &[u8]) -> bool {
		self.clone().eq(other.iter().copied())
	}
}

impl PartialEq<str> for Encoder<'_> {
	fn eq(&self, other: &str) -> bool {
		self.clone().eq(other.as_bytes().iter().copied())
	}
}

impl PartialOrd<[u8]> for Encoder<'_> {
	fn partial_cmp(&self, other: &[u8]) -> Option<Ordering> {
		Some(self.clone().cmp(other.iter().copied()))
	}
}

impl PartialOrd<str> for Encoder<'_> {
	fn partial_cmp(&self, other: &str) -> Option<Ordering> {
		Some(self.clone().cmp(other.as_bytes().iter().copied()))
	}
}

impl serde::Serialize for Encoder<'_> {
	fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
		let len = encode_len(self.src.len());
		serialize(serializer, len, |buffer| self.charset.encode_mut(self.src, buffer))
	}
}

/// Iterator that encodes bytes as hex characters (`char` type).
///
/// This is an opaque type alias that yields `char` values instead of `u8`.
/// Use [`Encoder::into_chars`] to create one.
pub type CharEncoder<'a> =
	impl ExactSizeIterator<Item = char> + DoubleEndedIterator + FusedIterator + 'a;

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

/// Encodes bytes to lowercase hex, returning an `Encoder`.
///
/// # Examples
/// ```
/// use omp_core::hex::encode;
/// let hex_string = encode(b"Hello").into_string();
/// assert_eq!(hex_string, "48656c6c6f");
/// ```
pub fn encode<I: AsRef<[u8]> + ?Sized>(src: &I) -> Encoder<'_> {
	Encoder::new(src.as_ref())
}

/// Encodes bytes to lowercase hex into a flat byte buffer.
///
/// Writes ASCII hex characters to `dst` and returns the number of bytes
/// written. The output length is `min(dst.len(), 2 * src.len())`.
///
/// # Examples
/// ```
/// use omp_core::hex;
/// let mut buf = [0u8; 10];
/// let n = hex::encode_mut(b"Hello", &mut buf);
/// assert_eq!(n, 10);
/// assert_eq!(&buf, b"48656c6c6f");
/// ```
#[inline]
pub const fn encode_mut(src: &[u8], dst: &mut [u8]) -> usize {
	LOWER.encode_mut(src, dst)
}

// ============================================================================
// CONST ENCODING/DECODING HELPERS
// ============================================================================

/// Decodes hex at compile time with validation.
///
/// # Examples
/// ```
/// use omp_core::hex::{Array, decode_n};
/// let decoded = decode_n(b"48656c6c6f").unwrap();
/// assert_eq!(&*decoded, b"Hello");
/// ```
pub const fn decode_n<const N: usize>(src: &[u8; N]) -> Option<Array<N>> {
	let mut out = [0; _];
	let Ok(written) = decode_mut(src.as_slice(), out.as_mut_slice()) else {
		return None;
	};
	Some(Array::new(out, written))
}

/// Encodes bytes to lowercase hex at compile time.
///
/// # Examples
/// ```
/// use omp_core::hex::{ArrayStr, encode_n};
/// const ENCODED: ArrayStr<5> = encode_n(b"Hello");
/// assert_eq!(&*ENCODED, "48656c6c6f");
/// ```
#[inline]
pub const fn encode_n<const N: usize>(src: &[u8; N]) -> ArrayStr<N> {
	LOWER.encode_n(src)
}

/// Returns the exact encoded length for a given source byte length.
///
/// Each byte encodes to exactly 2 hex characters.
///
/// # Examples
/// ```
/// use omp_core::hex::encode_len;
/// assert_eq!(encode_len(5), 10);
/// assert_eq!(encode_len(0), 0);
/// ```
#[inline]
pub const fn encode_len(src_len: usize) -> usize {
	src_len << 1
}

/// Returns the decoded length for a given hex string length.
///
/// For even-length inputs, returns `src_len / 2`.
/// For odd-length inputs, returns `(src_len + 1) / 2`.
///
/// # Examples
/// ```
/// use omp_core::hex::decode_len;
/// assert_eq!(decode_len(10), 5);
/// assert_eq!(decode_len(9), 5);
/// assert_eq!(decode_len(0), 0);
/// ```
#[inline]
pub const fn decode_len(src_len: usize) -> usize {
	src_len.div_ceil(2)
}
