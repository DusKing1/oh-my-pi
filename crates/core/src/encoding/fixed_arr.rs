//! Fixed-size array wrappers and formatting utilities for encoded strings.
//!
//! Provides [`Array`] and [`ArrayStr`] wrappers that efficiently store
//! encoded/decoded data with dynamic length tracking while maintaining
//! stack-allocated storage. These types support zero-copy access via `Deref`
//! and include specialized formatting implementations for hex and base-N
//! encodings.

use std::{
	borrow::{Borrow, BorrowMut},
	fmt::{self, Write},
	ops::{Deref, DerefMut},
};

/// A byte array wrapper that derefs to the decoded portion.
#[derive(Debug, Clone, Copy)]
pub struct Array<const N: usize>([u8; N], u16);

impl<const N: usize> Default for Array<N> {
	fn default() -> Self {
		Self([0u8; N], 0)
	}
}

impl<const N: usize> Array<N> {
	/// Creates a new `Array` wrapping the given data with a specified valid
	/// length.
	pub(crate) const fn new(data: [u8; N], len: usize) -> Self {
		debug_assert!(len <= N);
		debug_assert!(len <= u16::MAX as usize);
		Self(data, len as u16)
	}

	/// Returns the decoded bytes as a slice.
	#[inline(always)]
	pub const fn as_bytes(&self) -> &[u8] {
		// SAFETY: self.1 <= N is guaranteed by the constructor assert.
		unsafe { self.0.split_at_unchecked(self.1 as usize).0 }
	}

	/// Returns the decoded bytes as a mutable slice.
	#[inline(always)]
	pub const fn as_bytes_mut(&mut self) -> &mut [u8] {
		// SAFETY: self.1 <= N is guaranteed by the constructor assert.
		unsafe { self.0.split_at_mut_unchecked(self.1 as usize).0 }
	}
}

impl<const N: usize> Deref for Array<N> {
	type Target = [u8];

	#[inline]
	fn deref(&self) -> &Self::Target {
		self.as_bytes()
	}
}

impl<const N: usize> DerefMut for Array<N> {
	#[inline]
	fn deref_mut(&mut self) -> &mut Self::Target {
		self.as_bytes_mut()
	}
}

impl<const N: usize> AsRef<[u8]> for Array<N> {
	#[inline]
	fn as_ref(&self) -> &[u8] {
		self.as_bytes()
	}
}

impl<const N: usize> AsMut<[u8]> for Array<N> {
	#[inline]
	fn as_mut(&mut self) -> &mut [u8] {
		self.as_bytes_mut()
	}
}

/// An encoded string wrapper that stores the result.
#[derive(Clone, Copy)]
pub struct ArrayStr<const N: usize>([[u8; 2]; N], u16);

impl<const N: usize> Default for ArrayStr<N> {
	fn default() -> Self {
		Self([[0u8; 2]; N], 0)
	}
}

impl<const N: usize> ArrayStr<N> {
	/// Creates a new `ArrayStr` wrapping the given data with a specified valid
	/// length.
	pub(crate) const fn new(data: [[u8; 2]; N], len: usize) -> Self {
		debug_assert!(len <= N * 2, "Invalid length");
		debug_assert!(len <= u16::MAX as usize, "Invalid length");
		debug_assert!(data.as_flattened().split_at(len).0.is_ascii(), "Invalid ASCII");
		Self(data, len as u16)
	}

	/// Truncates the `ArrayStr` to the specified length.
	/// Does nothing if `new_len` is greater than the current length of the
	/// `ArrayStr`.
	pub const fn truncate(&mut self, new_len: usize) {
		if new_len <= self.1 as usize {
			self.1 = new_len as u16;
		}
	}

	/// Returns the encoded string as bytes.
	#[inline(always)]
	pub const fn as_bytes(&self) -> &[u8] {
		// SAFETY: self.1 <= N * 2 is guaranteed by the constructor assert.
		unsafe { self.0.as_flattened().split_at_unchecked(self.1 as usize).0 }
	}

	/// Returns the encoded string as bytes mut.
	///
	/// # Safety
	/// Caller must ensure that any modifications maintain valid base-N ASCII
	/// characters.
	#[inline(always)]
	pub const unsafe fn as_bytes_mut(&mut self) -> &mut [u8] {
		// SAFETY: self.1 <= N * 2 is guaranteed by the constructor assert.
		unsafe {
			self
				.0
				.as_flattened_mut()
				.split_at_mut_unchecked(self.1 as usize)
				.0
		}
	}

	/// Returns the encoded string as a str.
	#[inline(always)]
	pub const fn as_str(&self) -> &str {
		// SAFETY: Encoder produces only valid base-N ASCII characters, which are all
		// valid UTF-8.
		ascii_to_str(self.as_bytes())
	}

	/// Returns the encoded string as a mutable str.
	#[inline]
	pub const fn as_str_mut(&mut self) -> &mut str {
		// SAFETY: We will only return a mutable str instance so cannot become invalid
		// UTF-8.
		ascii_to_str_mut(unsafe { self.as_bytes_mut() })
	}
}

impl<const N: usize> Deref for ArrayStr<N> {
	type Target = str;

	#[inline]
	fn deref(&self) -> &Self::Target {
		self.as_str()
	}
}

impl<const N: usize> DerefMut for ArrayStr<N> {
	#[inline]
	fn deref_mut(&mut self) -> &mut Self::Target {
		self.as_str_mut()
	}
}

impl<const N: usize> AsRef<[u8]> for ArrayStr<N> {
	#[inline]
	fn as_ref(&self) -> &[u8] {
		self.as_bytes()
	}
}

impl<const N: usize> AsRef<str> for ArrayStr<N> {
	#[inline]
	fn as_ref(&self) -> &str {
		self.as_str()
	}
}

impl<const N: usize> AsMut<str> for ArrayStr<N> {
	#[inline]
	fn as_mut(&mut self) -> &mut str {
		self.as_str_mut()
	}
}

impl<const N: usize> Borrow<str> for ArrayStr<N> {
	#[inline]
	fn borrow(&self) -> &str {
		self.as_str()
	}
}

impl<const N: usize> BorrowMut<str> for ArrayStr<N> {
	#[inline]
	fn borrow_mut(&mut self) -> &mut str {
		self.as_str_mut()
	}
}

impl<const N: usize> fmt::Debug for ArrayStr<N> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str("\"")?;
		fmt::Display::fmt(self, f)?;
		f.write_str("\"")?;
		Ok(())
	}
}

impl<const N: usize> fmt::Display for ArrayStr<N> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		if self.1 as usize == N * 2 && f.alternate() {
			f.write_str("0x")?;
		}
		format_with_precision(self.as_bytes().iter().copied(), f)
	}
}

impl<const N: usize> fmt::LowerHex for ArrayStr<N> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		// If this is a hex array, it will be entirely fiilled
		if self.1 as usize == N * 2 {
			if f.alternate() {
				f.write_str("0x")?;
			}
			format_with_precision(
				self
					.as_bytes()
					.iter()
					.copied()
					.map(|b| b.to_ascii_lowercase()),
				f,
			)
		} else {
			format_with_precision(self.as_bytes().iter().copied(), f)
		}
	}
}

impl<const N: usize> fmt::UpperHex for ArrayStr<N> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		// If this is a hex array, it will be entirely fiilled
		if self.1 as usize == N * 2 {
			if f.alternate() {
				f.write_str("0x")?;
			}
			format_with_precision(
				self
					.as_bytes()
					.iter()
					.copied()
					.map(|b| b.to_ascii_uppercase()),
				f,
			)
		} else {
			format_with_precision(self.as_bytes().iter().copied(), f)
		}
	}
}

// ============================================================================
// SHARED FORMATTING UTILITY
// ============================================================================

/// Converts ASCII bytes to a string slice without validation.
///
/// # Safety
/// Must only be called on ASCII hex character bytes.
#[inline(always)]
pub const fn ascii_to_str(b: &[u8]) -> &str {
	debug_assert!(b.is_ascii(), "Invalid ASCII");
	// SAFETY: This function is only called on byte slices produced by the hex
	// encoders, which only generate ASCII hex characters ('0'-'9', 'a'-'f',
	// 'A'-'F'). All ASCII is valid UTF-8.
	unsafe { str::from_utf8_unchecked(b) }
}

/// Converts ASCII bytes to a mutable string slice without validation.
///
/// # Safety
/// Must only be called on ASCII hex character bytes.
#[inline(always)]
pub const fn ascii_to_str_mut(b: &mut [u8]) -> &mut str {
	debug_assert!(b.is_ascii(), "Invalid ASCII");
	// SAFETY: This function is only called on byte slices produced by the hex
	// encoders, which only generate ASCII hex characters ('0'-'9', 'a'-'f',
	// 'A'-'F'). All ASCII is valid UTF-8.
	unsafe { str::from_utf8_unchecked_mut(b) }
}

/// Converts ASCII bytes to an owned string without validation.
///
/// # Safety
/// Must only be called on ASCII hex character bytes.
#[inline(always)]
pub fn ascii_to_str_owned(b: Vec<u8>) -> String {
	debug_assert!(b.as_slice().is_ascii(), "Invalid ASCII");
	// SAFETY: This function is only called on byte slices produced by the hex
	// encoders, which only generate ASCII hex characters ('0'-'9', 'a'-'f',
	// 'A'-'F'). All ASCII is valid UTF-8.
	unsafe { String::from_utf8_unchecked(b) }
}

/// Formats an iterator with optional precision and alignment.
///
/// Used internally by `Display`, `LowerHex`, and `UpperHex` implementations.
pub fn format_with_precision(
	mut it: impl ExactSizeIterator<Item = u8>,
	f: &mut fmt::Formatter<'_>,
) -> fmt::Result {
	let Some(mut prec) = f.precision() else {
		for item in it {
			f.write_str(ascii_to_str(&[item]))?;
		}
		return Ok(());
	};

	let (align, fill) = match (f.align(), f.fill()) {
		(Some(align), ' ') => (align, '…'),
		(Some(align), fill) => (align, fill),
		(None, _) => (fmt::Alignment::Center, '…'),
	};

	let len = it.len();
	match align {
		fmt::Alignment::Left => {
			for item in it.by_ref() {
				let Some(pnew) = prec.checked_sub(1) else {
					f.write_char(fill)?;
					break;
				};
				prec = pnew;
				f.write_str(ascii_to_str(&[item]))?;
			}
		},
		fmt::Alignment::Right => {
			// If exact size iter:
			let skip_count = len.saturating_sub(prec);
			if skip_count > 0 {
				f.write_char(fill)?;
			}
			for i in it.skip(skip_count) {
				f.write_str(ascii_to_str(&[i]))?;
			}
		},
		fmt::Alignment::Center => {
			// If exact size iter:
			let (mut l, mut r) = (len, 0);
			if prec < l {
				r = prec >> 1;
				l = prec - r;
			}

			for i in (&mut it).take(l) {
				f.write_str(ascii_to_str(&[i]))?;
			}
			if r > 0 {
				f.write_char(fill)?;
				for i in it.skip(len - r - l) {
					f.write_str(ascii_to_str(&[i]))?;
				}
			}
		},
	}
	Ok(())
}

pub fn serialize<S, F>(serializer: S, n: usize, wr: F) -> Result<S::Ok, S::Error>
where
	S: serde::Serializer,
	F: FnOnce(&mut [u8]) -> usize,
{
	let mut stack = [0u8; 1024];
	let mut heap = Vec::new();
	let buffer = if let Some(slice) = stack.get_mut(..n) {
		slice
	} else {
		heap.resize(n, 0);
		&mut heap
	};

	let written = wr(buffer);
	serializer.serialize_str(ascii_to_str(&buffer[..written]))
}
