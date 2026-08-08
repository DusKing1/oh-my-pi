//! Small string optimization with stack allocation for short strings.
//!
//! `SmolStr` stores strings up to 23 bytes inline, avoiding heap allocation for
//! typical short strings. Longer strings use reference-counted heap storage
//! with O(1) cloning.

use std::{
	borrow::{Borrow, BorrowMut, Cow},
	boxed::Box,
	cmp::Ordering,
	convert::Infallible,
	fmt,
	hash::{self, Hash, Hasher},
	iter::FromIterator,
	mem,
	ops::{Add, Deref, DerefMut, Index},
	ptr, str,
	string::String,
	sync::Arc,
};

use bytes::{Bytes, BytesMut};
use bytes_utils::{Str, StrMut, string::StorageMut};

/// A `SmolStr` is a string type that has the following properties:
///
/// * `size_of::<SmolStr>() == 32`
/// * `Clone` is `O(1)`
/// * Strings are stack-allocated if they are:
///     * Up to 23 bytes long
/// * Additionally, a `SmolStr` can be explicitly created from a `&'static str`
///   without allocation
///
/// Unlike `String`, however, `SmolStr` is immutable. The primary use case for
#[derive(Default, Clone)]
#[repr(transparent)]
pub struct SmolStr(Repr<Str>);

/// An error type for UTF-8 validation.
pub type Utf8Error = bytes_utils::string::Utf8Error<Bytes>;

/// An error type for UTF-8 validation.
pub type Utf8ErrorMut = bytes_utils::string::Utf8Error<BytesMut>;

impl SmolStr {
	/// Constructs a `SmolStr` from a `Bytes` object without checking for UTF-8
	/// validity.
	///
	/// # Safety
	///
	/// The caller must ensure that the bytes are valid UTF-8. If this condition
	/// is not met, the behavior is undefined.
	#[inline]
	pub unsafe fn from_utf8_unchecked_owned(u: impl Into<Bytes>) -> Self {
		// SAFETY: The caller guarantees that the bytes are valid UTF-8.
		Self(Repr::Heap(unsafe { Str::from_inner_unchecked(u.into()) }))
	}

	/// Promotes an inline representation to a heap representation in place.
	///
	/// This function converts the internal representation of the `SmolStr` from
	/// inline to heap and returns a reference to the [`Str`].
	#[inline]
	pub fn promote(&mut self) -> &mut Str {
		if let Repr::Inline(buf) = &mut self.0 {
			self.0 = Repr::Heap(buf.as_str().into());
		}
		let Repr::Heap(data) = &mut self.0 else {
			unreachable!();
		};
		data
	}

	/// Constructs a `SmolStr` from a byte slice without checking for UTF-8
	/// validity.
	///
	/// # Safety
	///
	/// The caller must ensure that the bytes are valid UTF-8. If this condition
	/// is not met, the behavior is undefined.
	#[inline]
	pub unsafe fn from_utf8_unchecked(u: &[u8]) -> Self {
		// SAFETY: The caller guarantees that the bytes are valid UTF-8.
		Self::new(unsafe { str::from_utf8_unchecked(u) })
	}

	/// Constructs a `SmolStr` from a `Bytes` object, checking for UTF-8
	/// validity.
	///
	/// Returns an error if the bytes are not valid UTF-8.
	#[inline]
	pub fn from_utf8_owned(u: impl Into<Bytes>) -> Result<Self, Utf8Error> {
		Ok(Self(Repr::Heap(Str::from_inner(u.into())?)))
	}

	/// Constructs a `SmolStr` from a byte slice, checking for UTF-8 validity.
	///
	/// Returns an error if the bytes are not valid UTF-8.
	#[inline]
	pub fn from_utf8(u: &[u8]) -> Result<Self, str::Utf8Error> {
		Ok(Self::new(str::from_utf8(u)?))
	}

	/// Constructs an inline variant of `SmolStr`.
	///
	/// This never allocates.
	///
	/// # Panics
	///
	/// Panics if `text.len() > 23`.
	#[inline]
	pub fn new_inline(text: &str) -> Self {
		Self(Repr::new_inline(text).expect("len <= INLINE_CAP"))
	}

	/// Constructs a `SmolStr` from a statically allocated string.
	///
	/// This never allocates.
	#[inline(always)]
	pub const fn new_static(text: &'static str) -> Self {
		// NOTE: this never uses the inline storage; if a canonical
		// representation is needed, we could check for `len() < INLINE_CAP`
		// and call `new_inline`, but this would mean an extra branch.
		Self(Repr::Heap(Str::from_static(text)))
	}

	/// Constructs a `SmolStr` from a `str`, heap-allocating if necessary.
	#[inline(always)]
	pub fn new(text: impl AsRef<str>) -> Self {
		Self(Repr::copy_from_str(text.as_ref()))
	}

	/// Returns a `&str` slice of this `SmolStr`.
	#[inline(always)]
	pub fn as_str(&self) -> &str {
		self.0.as_str()
	}

	/// Returns the length of `self` in bytes.
	#[inline(always)]
	pub fn len(&self) -> usize {
		self.0.len()
	}

	/// Returns `true` if `self` has a length of zero bytes.
	#[inline(always)]
	pub fn is_empty(&self) -> bool {
		self.0.is_empty()
	}

	/// Returns `true` if `self` is heap-allocated.
	#[inline(always)]
	pub const fn is_spilled(&self) -> bool {
		matches!(self.0, Repr::Heap(..))
	}

	/// Returns `true` if the string is unique.
	#[inline(always)]
	pub fn is_unique(&self) -> bool {
		match &self.0 {
			Repr::Heap(data) => data.inner().is_unique(),
			Repr::Inline(_) => true,
		}
	}

	/// Strips a prefix from the string, returning the remainder as a new
	/// `SmolStr`. Returns `None` if the string doesn't start with the prefix.
	#[inline]
	pub fn strip_prefix(&self, prefix: &str) -> Option<Self> {
		let s = self.as_str();
		s.strip_prefix(prefix).map(|r| self.slice_ref(r))
	}

	/// Strips a suffix from the string, returning the remainder as a new
	/// `SmolStr`. Returns `None` if the string doesn't end with the suffix.
	#[inline]
	pub fn strip_suffix(&self, suffix: &str) -> Option<Self> {
		let s = self.as_str();
		s.strip_suffix(suffix).map(|r| self.slice_ref(r))
	}

	/// Returns a substring as a new `SmolStr`.
	/// For heap-allocated strings, this is a zero-copy operation.
	///
	/// # Panics
	/// Panics if the range is not on valid UTF-8 boundaries.
	#[inline]
	pub fn slice<R>(&self, range: R) -> Self
	where
		str: Index<R, Output = str>,
	{
		match &self.0 {
			Repr::Heap(data) => Self(Repr::Heap(data.slice(range))),
			_ => Self::new_inline(&self[range]),
		}
	}

	/// Extracts owned representation of the slice passed.
	/// For heap-allocated strings, this is a zero-copy operation.
	#[inline]
	pub fn slice_ref(&self, subset: &str) -> Self {
		match &self.0 {
			Repr::Heap(data) => Self(Repr::Heap(data.slice_ref(subset))),
			_ => Self::new_inline(subset),
		}
	}

	/// Splits the string at the given byte index and returns two `SmolStr`s.
	/// For heap-allocated strings, this creates two zero-copy references.
	///
	/// # Panics
	/// Panics if `at` is not on a UTF-8 character boundary.
	#[inline]
	pub fn split_at(&self, at: usize) -> (Self, Self) {
		match &self.0 {
			Repr::Heap(data) => {
				let (left, right) = data.clone().split_at_bytes(at);
				(Self(Repr::Heap(left)), Self(Repr::Heap(right)))
			},
			Repr::Inline(buf) => {
				let (left, right) = buf.split_at(at);
				(Self::new_inline(left), Self::new_inline(right))
			},
		}
	}

	/// Returns a string with leading whitespace removed.
	/// For heap strings, this is zero-copy when possible.
	#[inline]
	pub fn trim_start(&self) -> Self {
		let trimmed = self.as_str().trim_start();
		self.slice_ref(trimmed)
	}

	/// Returns a string with trailing whitespace removed.
	/// For heap strings, this is zero-copy when possible.
	#[inline]
	pub fn trim_end(&self) -> Self {
		let trimmed = self.as_str().trim_end();
		self.slice_ref(trimmed)
	}

	/// Returns a string with leading and trailing whitespace removed.
	/// For heap strings, this is zero-copy when possible.
	#[inline]
	pub fn trim(&self) -> Self {
		let trimmed = self.as_str().trim();
		self.slice_ref(trimmed)
	}

	/// Truncates the `SmolStr` to the specified length.
	///
	/// If `len` is greater than the current length, this has no effect.
	///
	/// # Panics
	///
	/// Panics if `len` is greater than the current length of the `SmolStr` or if
	/// `len` is not on a valid UTF-8 character boundary.
	#[inline]
	pub fn truncate(&mut self, len: usize) {
		match &mut self.0 {
			Repr::Inline(buf) => {
				buf.truncate(len);
			},
			Repr::Heap(heap) => {
				assert!(heap.is_char_boundary(len), "Index is not on a char boundary");
				// SAFETY: The bytes are valid UTF-8 because they originated from a
				// heap-allocated string that was previously validated. Truncating at a
				// char boundary (verified by the assert above) preserves UTF-8 validity.
				unsafe {
					let mut bytes = mem::take(heap).into_inner();
					bytes.truncate(len);
					*heap = Str::from_inner_unchecked(bytes);
				}
			},
		}
	}

	/// Splits on the given separator and returns an iterator of `SmolStr`s.
	/// For heap strings, the splits are zero-copy references.
	pub fn split<'s>(
		&'s self,
		separator: &'s str,
	) -> impl Clone + std::iter::FusedIterator<Item = Self> + 's {
		self
			.as_str()
			.split(separator)
			.map(move |s| self.slice_ref(s))
	}

	/// Converts the string to ASCII lowercase.
	///
	/// Reuses the allocation when uniquely owned; otherwise copies.
	pub fn into_ascii_lowercase(self) -> Self {
		let mut buf = SmolStrMut::from(self);
		buf.make_ascii_lowercase();
		buf.freeze()
	}

	/// Converts the string to ASCII uppercase.
	///
	/// Reuses the allocation when uniquely owned; otherwise copies.
	pub fn into_ascii_uppercase(self) -> Self {
		let mut buf = SmolStrMut::from(self);
		buf.make_ascii_uppercase();
		buf.freeze()
	}

	/// Returns a byte slice of this `SmolStr`.
	#[inline(always)]
	pub fn as_bytes(&self) -> &[u8] {
		self.as_str().as_bytes()
	}

	/// Tries to convert this `SmolStr` into a `SmolStrMut`.
	#[inline]
	pub fn try_into_mut(self) -> Result<SmolStrMut, Self> {
		match self.0 {
			Repr::Heap(data) => match data.into_inner().try_into_mut() {
				// SAFETY: The data is valid UTF-8 because it came from a Str,
				// and BytesMut preserves the UTF-8 bytes when converted.
				Ok(data) => Ok(SmolStrMut(Repr::Heap(unsafe { StrMut::from_inner_unchecked(data) }))),
				// SAFETY: If try_into_mut fails, we reconstruct the original Str from
				// the returned Bytes. The bytes are still valid UTF-8.
				Err(e) => Err(Self(Repr::Heap(unsafe { Str::from_inner_unchecked(e) }))),
			},
			Repr::Inline(buf) => Ok(SmolStrMut(Repr::Inline(buf))),
		}
	}
}

/// Extension trait for `str` to provide `_smol` versions of methods that
/// return an allocated `String`.
pub trait SmolStrExt {
	/// Converts the string to lowercase using ASCII rules and returns a
	/// `SmolStr`.
	///
	/// This is a `_smol` version of [`str::to_ascii_lowercase`].
	///
	/// # Examples
	///
	/// ```
	/// use omp_core::smolstr::SmolStrExt;
	///
	/// let s = "HELLO";
	/// let smol_str = s.to_ascii_lowercase_smol();
	/// assert_eq!(smol_str, "hello");
	/// ```
	fn to_ascii_lowercase_smol(&self) -> SmolStr;

	/// Converts the string to uppercase using ASCII rules and returns a
	/// `SmolStr`.
	///
	/// This is a `_smol` version of [`str::to_ascii_uppercase`].
	///
	/// # Examples
	///
	/// ```
	/// use omp_core::smolstr::SmolStrExt;
	///
	/// let s = "hello";
	/// let smol_str = s.to_ascii_uppercase_smol();
	/// assert_eq!(smol_str, "HELLO");
	/// ```
	fn to_ascii_uppercase_smol(&self) -> SmolStr;
}

impl SmolStrExt for str {
	fn to_ascii_lowercase_smol(&self) -> SmolStr {
		let mut s = SmolStrMut::new(self);
		s.make_ascii_lowercase();
		s.freeze()
	}

	fn to_ascii_uppercase_smol(&self) -> SmolStr {
		let mut s = SmolStrMut::new(self);
		s.make_ascii_uppercase();
		s.freeze()
	}
}

// ============================
// Comparison
// ============================

impl Eq for SmolStr {}
impl PartialEq<Self> for SmolStr {
	fn eq(&self, other: &Self) -> bool {
		self.0.ptr_eq(&other.0) || self.as_str() == other.as_str()
	}
}

impl PartialEq<str> for SmolStr {
	#[inline(always)]
	fn eq(&self, other: &str) -> bool {
		self.as_str() == other
	}
}

impl PartialEq<SmolStr> for str {
	#[inline(always)]
	fn eq(&self, other: &SmolStr) -> bool {
		other == self
	}
}

impl<'a> PartialEq<&'a str> for SmolStr {
	#[inline(always)]
	fn eq(&self, other: &&'a str) -> bool {
		self == *other
	}
}

impl PartialEq<SmolStr> for &str {
	#[inline(always)]
	fn eq(&self, other: &SmolStr) -> bool {
		*self == other
	}
}

impl PartialEq<String> for SmolStr {
	#[inline(always)]
	fn eq(&self, other: &String) -> bool {
		self.as_str() == other
	}
}

impl PartialEq<SmolStr> for String {
	#[inline(always)]
	fn eq(&self, other: &SmolStr) -> bool {
		other == self
	}
}

impl<'a> PartialEq<&'a String> for SmolStr {
	#[inline(always)]
	fn eq(&self, other: &&'a String) -> bool {
		self == *other
	}
}

impl PartialEq<SmolStr> for &String {
	#[inline(always)]
	fn eq(&self, other: &SmolStr) -> bool {
		*self == other
	}
}

impl Ord for SmolStr {
	fn cmp(&self, other: &Self) -> Ordering {
		self.as_str().cmp(other.as_str())
	}
}

impl PartialOrd for SmolStr {
	fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
		Some(self.cmp(other))
	}
}

impl PartialOrd<str> for SmolStr {
	#[inline(always)]
	fn partial_cmp(&self, other: &str) -> Option<Ordering> {
		self.as_str().partial_cmp(other)
	}
}

impl PartialOrd<SmolStr> for str {
	#[inline(always)]
	fn partial_cmp(&self, other: &SmolStr) -> Option<Ordering> {
		self.partial_cmp(other.as_str())
	}
}

impl<'a> PartialOrd<&'a str> for SmolStr {
	#[inline(always)]
	fn partial_cmp(&self, other: &&'a str) -> Option<Ordering> {
		self.partial_cmp(*other)
	}
}

impl PartialOrd<SmolStr> for &str {
	#[inline(always)]
	fn partial_cmp(&self, other: &SmolStr) -> Option<Ordering> {
		(*self).partial_cmp(other)
	}
}

impl hash::Hash for SmolStr {
	fn hash<H: hash::Hasher>(&self, hasher: &mut H) {
		self.as_str().hash(hasher);
	}
}

// ============================
// Formatting
// ============================

impl fmt::Debug for SmolStr {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		fmt::Debug::fmt(self.as_str(), f)
	}
}

impl fmt::Display for SmolStr {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		fmt::Display::fmt(self.as_str(), f)
	}
}

impl fmt::Debug for SmolStrMut {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		fmt::Debug::fmt(self.as_str(), f)
	}
}

impl fmt::Display for SmolStrMut {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		fmt::Display::fmt(self.as_str(), f)
	}
}

// ============================
// SmolStrMut Comparison
// ============================

impl Eq for SmolStrMut {}
impl PartialEq<Self> for SmolStrMut {
	fn eq(&self, other: &Self) -> bool {
		self.0.ptr_eq(&other.0) || self.as_str() == other.as_str()
	}
}

impl PartialEq<str> for SmolStrMut {
	#[inline(always)]
	fn eq(&self, other: &str) -> bool {
		self.as_str() == other
	}
}

impl PartialEq<SmolStrMut> for str {
	#[inline(always)]
	fn eq(&self, other: &SmolStrMut) -> bool {
		other == self
	}
}

impl<'a> PartialEq<&'a str> for SmolStrMut {
	#[inline(always)]
	fn eq(&self, other: &&'a str) -> bool {
		self == *other
	}
}

impl PartialEq<SmolStrMut> for &str {
	#[inline(always)]
	fn eq(&self, other: &SmolStrMut) -> bool {
		*self == other
	}
}

impl PartialEq<String> for SmolStrMut {
	#[inline(always)]
	fn eq(&self, other: &String) -> bool {
		self.as_str() == other
	}
}

impl PartialEq<SmolStrMut> for String {
	#[inline(always)]
	fn eq(&self, other: &SmolStrMut) -> bool {
		other == self
	}
}

impl<'a> PartialEq<&'a String> for SmolStrMut {
	#[inline(always)]
	fn eq(&self, other: &&'a String) -> bool {
		self == *other
	}
}

impl PartialEq<SmolStrMut> for &String {
	#[inline(always)]
	fn eq(&self, other: &SmolStrMut) -> bool {
		*self == other
	}
}

impl Ord for SmolStrMut {
	fn cmp(&self, other: &Self) -> Ordering {
		self.as_str().cmp(other.as_str())
	}
}

impl PartialOrd for SmolStrMut {
	fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
		Some(self.cmp(other))
	}
}

impl PartialOrd<str> for SmolStrMut {
	#[inline(always)]
	fn partial_cmp(&self, other: &str) -> Option<Ordering> {
		self.as_str().partial_cmp(other)
	}
}

impl PartialOrd<SmolStrMut> for str {
	#[inline(always)]
	fn partial_cmp(&self, other: &SmolStrMut) -> Option<Ordering> {
		self.partial_cmp(other.as_str())
	}
}

impl<'a> PartialOrd<&'a str> for SmolStrMut {
	#[inline(always)]
	fn partial_cmp(&self, other: &&'a str) -> Option<Ordering> {
		self.partial_cmp(*other)
	}
}

impl PartialOrd<SmolStrMut> for &str {
	#[inline(always)]
	fn partial_cmp(&self, other: &SmolStrMut) -> Option<Ordering> {
		(*self).partial_cmp(other)
	}
}

impl hash::Hash for SmolStrMut {
	fn hash<H: hash::Hasher>(&self, hasher: &mut H) {
		self.as_str().hash(hasher);
	}
}

// ============================
// Borrows
// ============================

impl AsRef<str> for SmolStr {
	#[inline(always)]
	fn as_ref(&self) -> &str {
		self.as_str()
	}
}

impl AsRef<[u8]> for SmolStr {
	#[inline(always)]
	fn as_ref(&self) -> &[u8] {
		self.as_str().as_bytes()
	}
}

impl AsRef<std::ffi::OsStr> for SmolStr {
	#[inline(always)]
	fn as_ref(&self) -> &std::ffi::OsStr {
		AsRef::<std::ffi::OsStr>::as_ref(self.as_str())
	}
}

impl AsRef<std::path::Path> for SmolStr {
	#[inline(always)]
	fn as_ref(&self) -> &std::path::Path {
		AsRef::<std::path::Path>::as_ref(self.as_str())
	}
}

impl Borrow<str> for SmolStr {
	#[inline(always)]
	fn borrow(&self) -> &str {
		self.as_str()
	}
}

impl Deref for SmolStr {
	type Target = str;

	#[inline(always)]
	fn deref(&self) -> &str {
		self.as_str()
	}
}

impl AsRef<str> for SmolStrMut {
	#[inline(always)]
	fn as_ref(&self) -> &str {
		self.as_str()
	}
}

impl AsMut<str> for SmolStrMut {
	#[inline(always)]
	fn as_mut(&mut self) -> &mut str {
		self.as_str_mut()
	}
}

impl AsRef<[u8]> for SmolStrMut {
	#[inline(always)]
	fn as_ref(&self) -> &[u8] {
		self.as_str().as_bytes()
	}
}

impl AsRef<std::ffi::OsStr> for SmolStrMut {
	#[inline(always)]
	fn as_ref(&self) -> &std::ffi::OsStr {
		AsRef::<std::ffi::OsStr>::as_ref(self.as_str())
	}
}

impl AsRef<std::path::Path> for SmolStrMut {
	#[inline(always)]
	fn as_ref(&self) -> &std::path::Path {
		AsRef::<std::path::Path>::as_ref(self.as_str())
	}
}

impl Borrow<str> for SmolStrMut {
	#[inline(always)]
	fn borrow(&self) -> &str {
		self.as_str()
	}
}

impl BorrowMut<str> for SmolStrMut {
	#[inline(always)]
	fn borrow_mut(&mut self) -> &mut str {
		self.as_str_mut()
	}
}

impl Deref for SmolStrMut {
	type Target = str;

	#[inline(always)]
	fn deref(&self) -> &str {
		self.as_str()
	}
}

impl DerefMut for SmolStrMut {
	#[inline(always)]
	fn deref_mut(&mut self) -> &mut str {
		self.as_str_mut()
	}
}

// ============================
// Add implementations
// ============================

impl Add<&str> for SmolStr {
	type Output = Self;

	#[inline]
	fn add(self, rhs: &str) -> Self::Output {
		match self.try_into_mut() {
			Ok(mut lhs) => {
				lhs.push_str(rhs);
				lhs.freeze()
			},
			Err(this) => {
				let mut lhs = SmolStrMut::with_capacity(this.len() + rhs.len());
				lhs.push_str(&this);
				lhs.push_str(rhs);
				lhs.freeze()
			},
		}
	}
}

impl Add<SmolStr> for &str {
	type Output = SmolStr;

	#[inline]
	fn add(self, rhs: SmolStr) -> Self::Output {
		match rhs.try_into_mut() {
			Ok(mut rhs) => {
				rhs.insert(0, self);
				rhs.freeze()
			},
			Err(rhs) => {
				let mut result = SmolStrMut::with_capacity(self.len() + rhs.len());
				result.push_str(self);
				result.push_str(rhs.as_str());
				result.freeze()
			},
		}
	}
}

impl Add<&str> for &SmolStr {
	type Output = SmolStr;

	#[inline]
	fn add(self, rhs: &str) -> Self::Output {
		let mut result = SmolStrMut::with_capacity(self.len() + rhs.len());
		result.push_str(self);
		result.push_str(rhs);
		result.freeze()
	}
}

impl Add<&SmolStr> for &str {
	type Output = SmolStr;

	#[inline]
	fn add(self, rhs: &SmolStr) -> Self::Output {
		let mut result = SmolStrMut::with_capacity(self.len() + rhs.len());
		result.push_str(self);
		result.push_str(rhs.as_str());
		result.freeze()
	}
}

impl Add<&str> for SmolStrMut {
	type Output = Self;

	#[inline]
	fn add(mut self, rhs: &str) -> Self::Output {
		self.push_str(rhs);
		self
	}
}

impl Add<SmolStrMut> for &str {
	type Output = SmolStrMut;

	#[inline]
	fn add(self, mut rhs: SmolStrMut) -> Self::Output {
		// Optimize by inserting at the beginning if we own the rhs
		rhs.insert(0, self);
		rhs
	}
}

impl Add<&str> for &SmolStrMut {
	type Output = SmolStrMut;

	#[inline]
	fn add(self, rhs: &str) -> Self::Output {
		let mut result = SmolStrMut::with_capacity(self.len() + rhs.len());
		result.push_str(self.as_str());
		result.push_str(rhs);
		result
	}
}

impl Add<&SmolStrMut> for &str {
	type Output = SmolStrMut;

	#[inline]
	fn add(self, rhs: &SmolStrMut) -> Self::Output {
		let mut result = SmolStrMut::with_capacity(self.len() + rhs.len());
		result.push_str(self);
		result.push_str(rhs.as_str());
		result
	}
}

// SmolStr + SmolStr combinations
impl Add<Self> for SmolStr {
	type Output = Self;

	#[inline]
	fn add(self, rhs: Self) -> Self::Output {
		match self.try_into_mut() {
			Ok(mut lhs) => {
				lhs.push_str(&rhs);
				lhs.freeze()
			},
			Err(lhs) => match rhs.try_into_mut() {
				Ok(mut rhs) => {
					rhs.insert(0, &lhs);
					rhs.freeze()
				},
				Err(rhs) => {
					let mut result = SmolStrMut::with_capacity(lhs.len() + rhs.len());
					result.push_str(&lhs);
					result.push_str(&rhs);
					result.freeze()
				},
			},
		}
	}
}

impl Add<&Self> for SmolStr {
	type Output = Self;

	#[inline]
	fn add(self, rhs: &Self) -> Self::Output {
		match self.try_into_mut() {
			Ok(mut lhs) => {
				lhs.push_str(rhs);
				lhs.freeze()
			},
			Err(lhs) => {
				let mut result = SmolStrMut::with_capacity(lhs.len() + rhs.len());
				result.push_str(&lhs);
				result.push_str(rhs);
				result.freeze()
			},
		}
	}
}

impl Add<SmolStr> for &SmolStr {
	type Output = SmolStr;

	#[inline]
	fn add(self, rhs: SmolStr) -> Self::Output {
		match rhs.try_into_mut() {
			Ok(mut rhs) => {
				rhs.insert(0, self);
				rhs.freeze()
			},
			Err(rhs) => {
				let mut result = SmolStrMut::with_capacity(self.len() + rhs.len());
				result.push_str(self);
				result.push_str(&rhs);
				result.freeze()
			},
		}
	}
}

impl Add<&SmolStr> for &SmolStr {
	type Output = SmolStr;

	#[inline]
	fn add(self, rhs: &SmolStr) -> Self::Output {
		let mut result = SmolStrMut::with_capacity(self.len() + rhs.len());
		result.push_str(self);
		result.push_str(rhs);
		result.freeze()
	}
}

// SmolStrMut + SmolStrMut combinations
impl Add<Self> for SmolStrMut {
	type Output = Self;

	#[inline]
	fn add(mut self, rhs: Self) -> Self::Output {
		self.push_str(&rhs);
		self
	}
}

impl Add<&Self> for SmolStrMut {
	type Output = Self;

	#[inline]
	fn add(mut self, rhs: &Self) -> Self::Output {
		self.push_str(rhs.as_str());
		self
	}
}

impl Add<SmolStrMut> for &SmolStrMut {
	type Output = SmolStrMut;

	#[inline]
	fn add(self, mut rhs: SmolStrMut) -> Self::Output {
		rhs.insert(0, self.as_str());
		rhs
	}
}

impl Add<&SmolStrMut> for &SmolStrMut {
	type Output = SmolStrMut;

	#[inline]
	fn add(self, rhs: &SmolStrMut) -> Self::Output {
		let mut result = SmolStrMut::with_capacity(self.len() + rhs.len());
		result.push_str(self.as_str());
		result.push_str(rhs.as_str());
		result
	}
}

// ============================
// Repr
// ============================

const INLINE_CAP: usize = 23;

#[derive(Debug)]
enum Repr<H> {
	Inline(heapless::String<INLINE_CAP, u8>),
	Heap(H),
}

impl<H: Clone> Clone for Repr<H> {
	#[inline]
	fn clone(&self) -> Self {
		match self {
			Self::Heap(data) => Self::Heap(data.clone()),
			// SAFETY: For Inline variant, we perform a bitwise copy using ptr::read.
			// This is safe because the Inline variant contains only Copy types
			// (heapless::String which is a wrapper around a fixed-size array).
			_ => unsafe { ptr::read(self as *const Self) },
		}
	}
}

impl<H> Default for Repr<H> {
	#[inline]
	fn default() -> Self {
		Self::new()
	}
}

impl<H> Repr<H> {
	#[inline(always)]
	const fn new() -> Self {
		Self::Inline(heapless::String::new())
	}
}

impl<H> Repr<H>
where
	H: Deref<Target = str> + for<'a> From<&'a str>,
{
	/// This function tries to create a new `Repr::Inline` or `Repr::Static`
	/// If it isn't possible, this function returns None
	#[inline(always)]
	fn new_inline(text: &str) -> Option<Self> {
		heapless::String::try_from(text).ok().map(Self::Inline)
	}

	#[inline(always)]
	fn copy_from_str(text: &str) -> Self {
		match heapless::String::try_from(text) {
			Ok(buf) => Self::Inline(buf),
			Err(_) => Self::Heap(text.into()),
		}
	}

	#[inline(always)]
	fn len(&self) -> usize {
		match self {
			Self::Heap(data) => data.len(),
			Self::Inline(buf) => buf.len(),
		}
	}

	#[inline(always)]
	fn is_empty(&self) -> bool {
		match self {
			Self::Heap(data) => data.is_empty(),
			Self::Inline(buf) => buf.is_empty(),
		}
	}

	#[inline]
	fn as_str(&self) -> &str {
		match self {
			Self::Heap(data) => data,
			Self::Inline(buf) => buf.as_str(),
		}
	}

	#[inline]
	fn ptr_eq(&self, other: &Self) -> bool {
		let (this, that) = (self.as_str(), other.as_str());
		// Pointer identity alone is not equality: zero-copy prefix slices
		// share their parent's start pointer with a different length.
		ptr::eq(this.as_ptr(), that.as_ptr()) && this.len() == that.len()
	}
}

// ============================
// Extend / Format
// ============================

/// Formats arguments to a [`SmolStr`], potentially without allocating.
///
/// See [`std::format!`] or [`format_args!`] for syntax documentation.
#[macro_export]
macro_rules! format_smol {
    ($($tt:tt)*) => {{
        let mut w = $crate::smolstr::SmolStrMut::default();
        ::std::fmt::Write::write_fmt(&mut w, format_args!($($tt)*))
         .expect("a formatting trait implementation returned an error");
        w.freeze()
    }};
}

/// Formats arguments to a [`SmolStrMut`], potentially without allocating.
///
/// See [`std::format!`] or [`format_args!`] for syntax documentation.
#[macro_export]
macro_rules! format_smolmut {
    ($($tt:tt)*) => {{
        let mut w = $crate::smolstr::SmolStrMut::default();
        ::std::fmt::Write::write_fmt(&mut w, format_args!($($tt)*))
         .expect("a formatting trait implementation returned an error");
        w
    }};
}

macro_rules! impl_extend {
    // Case with explicit lifetime
    (for<$lt:lifetime> $type:ty, ($this:ident, $item:ident) => $($body:tt)*) => {
        impl<$lt> Extend<$type> for SmolStrMut {
            fn extend<T: IntoIterator<Item = $type>>(&mut self, iter: T) {
                let $this: &mut SmolStrMut = self;
                for $item in iter {
                    $($body)*
                }
            }
        }
        impl<$lt> FromIterator<$type> for SmolStrMut {
            fn from_iter<T: IntoIterator<Item = $type>>(iter: T) -> Self {
                let mut $this = SmolStrMut::default();
                $this.extend(iter);
                $this
            }
        }
        impl<$lt> FromIterator<$type> for SmolStr {
            fn from_iter<T: IntoIterator<Item = $type>>(iter: T) -> Self {
                let mut $this = SmolStrMut::default();
                $this.extend(iter);
                $this.freeze()
            }
        }
    };
    // Case without lifetimes
    ($type:ty, ($this:ident, $item:ident) => $($body:tt)*) => {
        impl Extend<$type> for SmolStrMut {
            fn extend<T: IntoIterator<Item = $type>>(&mut self, iter: T) {
                let $this: &mut SmolStrMut = self;
                for $item in iter {
                    $($body)*
                }
            }
        }
        impl FromIterator<$type> for SmolStrMut {
            fn from_iter<T: IntoIterator<Item = $type>>(iter: T) -> Self {
                let mut $this = SmolStrMut::default();
                $this.extend(iter);
                $this
            }
        }
        impl FromIterator<$type> for SmolStr {
            fn from_iter<T: IntoIterator<Item = $type>>(iter: T) -> Self {
                let mut $this = SmolStrMut::default();
                $this.extend(iter);
                $this.freeze()
            }
        }
    };
}

impl_extend!(char, (s, rhs) => s.push(rhs));
impl_extend!(String, (s, rhs) => s.push_str(rhs.as_str()));
impl_extend!(for<'a> &'a String, (s, rhs) => s.push_str(rhs.as_str()));
impl_extend!(for<'a> &'a str, (s, rhs) => s.push_str(rhs));

// ============================
// SmolStrMut
// ============================

/// Mutable, growable counterpart of [`SmolStr`]: same inline layout for
/// strings up to 23 bytes, heap-backed above.
///
/// Build with `push`/`push_str` (or
/// [`format_smolmut!`](crate::format_smolmut)), then [`freeze`](Self::freeze)
/// into an immutable [`SmolStr`] without copying.
#[derive(Default, Clone)]
#[repr(transparent)]
pub struct SmolStrMut(Repr<StrMut>);

impl SmolStrMut {
	/// Constructs a `SmolStrMut` from a `BytesMut` object without checking for
	/// UTF-8 validity.
	///
	/// # Safety
	///
	/// The caller must ensure that the bytes are valid UTF-8. If this condition
	/// is not met, the behavior is undefined.
	#[inline]
	pub unsafe fn from_utf8_unchecked_owned(u: impl Into<BytesMut>) -> Self {
		// SAFETY: The caller guarantees that the bytes are valid UTF-8.
		Self(Repr::Heap(unsafe { StrMut::from_inner_unchecked(u.into()) }))
	}

	/// Constructs a `SmolStrMut` from a byte slice without checking for UTF-8
	/// validity.
	///
	/// # Safety
	///
	/// The caller must ensure that the bytes are valid UTF-8. If this condition
	/// is not met, the behavior is undefined.
	#[inline]
	pub unsafe fn from_utf8_unchecked(u: &[u8]) -> Self {
		// SAFETY: The caller guarantees that the bytes are valid UTF-8.
		Self::new(unsafe { str::from_utf8_unchecked(u) })
	}

	/// Constructs a `SmolStrMut` from a `BytesMut` object, checking for UTF-8
	/// validity.
	///
	/// Returns an error if the bytes are not valid UTF-8.
	#[inline]
	pub fn from_utf8_owned(u: impl Into<BytesMut>) -> Result<Self, Utf8ErrorMut> {
		let u: BytesMut = u.into();
		Ok(Self(Repr::Heap(StrMut::from_inner(u)?)))
	}

	/// Constructs a `SmolStrMut` from a byte slice, checking for UTF-8 validity.
	///
	/// Returns an error if the bytes are not valid UTF-8.
	#[inline]
	pub fn from_utf8(u: &[u8]) -> Result<Self, str::Utf8Error> {
		Ok(Self::new(str::from_utf8(u)?))
	}

	/// Constructs an inline variant of `SmolStrMut`.
	///
	/// This never allocates.
	///
	/// # Panics
	///
	/// Panics if `text.len() > 23`.
	#[inline]
	pub fn new_inline(text: &str) -> Self {
		Self(Repr::new_inline(text).expect("len <= INLINE_CAP"))
	}

	/// Constructs a `SmolStr` from a `str`, heap-allocating if necessary.
	#[inline(always)]
	pub fn new(text: impl AsRef<str>) -> Self {
		Self(Repr::copy_from_str(text.as_ref()))
	}

	/// Constructs a `SmolStrMut` with the given capacity.
	#[inline]
	pub fn with_capacity(capacity: usize) -> Self {
		if capacity > INLINE_CAP {
			// SAFETY: A newly allocated BytesMut with capacity is empty, and an
			// empty byte buffer is trivially valid UTF-8.
			Self(Repr::Heap(unsafe {
				StrMut::from_inner_unchecked(BytesMut::with_capacity(capacity))
			}))
		} else {
			Self(Repr::new())
		}
	}

	/// Returns a `&str` slice of this `SmolStr`.
	#[inline(always)]
	pub fn as_str(&self) -> &str {
		self.0.as_str()
	}

	/// Returns a mutable `&str` slice of this `SmolStr`.
	#[inline(always)]
	pub fn as_str_mut(&mut self) -> &mut str {
		match &mut self.0 {
			// SAFETY: StrMut guarantees that its inner BytesMut contains valid UTF-8.
			Repr::Heap(data) => unsafe { str::from_utf8_unchecked_mut(data.as_bytes_mut()) },
			Repr::Inline(buf) => buf.as_mut_str(),
		}
	}

	/// Returns the length of `self` in bytes.
	#[inline(always)]
	pub fn len(&self) -> usize {
		self.0.len()
	}

	/// Returns `true` if `self` has a length of zero bytes.
	#[inline(always)]
	pub fn is_empty(&self) -> bool {
		self.0.is_empty()
	}

	/// Truncates the `SmolStr` to the specified length.
	///
	/// If `len` is greater than the current length, this has no effect.
	///
	/// # Panics
	///
	/// Panics if `len` is greater than the current length of the `SmolStr` or if
	/// `len` is not on a valid UTF-8 character boundary.
	#[inline]
	pub fn truncate(&mut self, len: usize) {
		match &mut self.0 {
			Repr::Inline(buf) => {
				buf.truncate(len);
			},
			Repr::Heap(heap) => {
				assert!(heap.is_char_boundary(len), "Index is not on a char boundary");
				// SAFETY: Truncating at a char boundary (verified by the assert above)
				// preserves UTF-8 validity of the underlying byte buffer.
				unsafe {
					heap.inner_mut().truncate(len);
				}
			},
		}
	}

	/// Returns `true` if `self` is heap-allocated.
	#[inline(always)]
	pub const fn is_spilled(&self) -> bool {
		matches!(self.0, Repr::Heap(..))
	}

	/// Reserves capacity for at least `additional` more bytes to be inserted in
	/// the given `SmolStrMut`.
	#[inline]
	pub fn reserve(&mut self, additional: usize) {
		match &mut self.0 {
			Repr::Inline(buf) => {
				let cap = buf.len() + additional;
				if cap > INLINE_CAP {
					let cap = cap.next_power_of_two();
					// SAFETY: A newly allocated BytesMut with capacity is empty, and an
					// empty byte buffer is trivially valid UTF-8.
					let mut heap = unsafe { StrMut::from_inner_unchecked(BytesMut::with_capacity(cap)) };
					heap.push_str(buf.as_str());
					*self = Self(Repr::Heap(heap));
				}
			},
			Repr::Heap(heap) => {
				// SAFETY: Reserving capacity does not modify the existing UTF-8 bytes,
				// only extends the available capacity.
				unsafe {
					heap.inner_mut().reserve(additional);
				}
			},
		}
	}

	/// Builds a [`SmolStr`] from `self`.
	#[must_use]
	#[inline]
	pub fn freeze(self) -> SmolStr {
		SmolStr(match self.0 {
			Repr::Inline(buf) => Repr::Inline(buf),
			Repr::Heap(heap) => Repr::Heap(heap.freeze()),
		})
	}

	/// Appends the given [`char`] to the end of `self`'s buffer.
	#[inline]
	pub fn push(&mut self, c: char) {
		let mut buf = [0; 4];
		self.push_str(c.encode_utf8(&mut buf));
	}

	/// Appends a given string slice onto the end of `self`'s buffer.
	#[inline]
	pub fn push_str(&mut self, s: &str) {
		match &mut self.0 {
			Repr::Inline(buf) => {
				let len = buf.len();
				if buf.push_str(s).is_err() {
					let mut heap = BytesMut::with_capacity((len + s.len()).next_power_of_two());
					heap.extend_from_slice(buf.as_bytes());
					heap.extend_from_slice(s.as_bytes());
					// SAFETY: We copy valid UTF-8 bytes from buf and s into heap.
					// Both sources are valid UTF-8, so the result is valid UTF-8.
					*self = Self(Repr::Heap(unsafe { StrMut::from_inner_unchecked(heap) }));
				}
			},
			Repr::Heap(heap) => heap.push_str(s),
		}
	}

	/// Pushes raw bytes onto the end of `self`'s buffer.
	///
	/// # Safety
	///
	/// The caller must ensure that the bytes are valid UTF-8. If this condition
	/// is not met, the behavior is undefined.
	#[inline]
	pub unsafe fn extend_from_bytes_unchecked(&mut self, s: &[u8]) {
		match &mut self.0 {
			Repr::Inline(buf) => {
				let len = buf.len();
				// SAFETY: The caller guarantees that s contains valid UTF-8 bytes.
				// We extend the inline buffer's internal vector directly.
				if unsafe { buf.as_mut_vec().extend_from_slice(s).is_err() } {
					let mut heap = BytesMut::with_capacity((len + s.len()).next_power_of_two());
					heap.extend_from_slice(buf.as_bytes());
					heap.extend_from_slice(s);
					// SAFETY: buf contains valid UTF-8 and the caller guarantees s
					// contains valid UTF-8, so heap contains valid UTF-8.
					*self = Self(Repr::Heap(unsafe { StrMut::from_inner_unchecked(heap) }));
				}
			},
			// SAFETY: The caller guarantees that s contains valid UTF-8 bytes.
			Repr::Heap(heap) => unsafe { heap.inner_mut().push_slice(s) },
		}
	}

	/// Inserts a given string slice at the specified position in `self`'s
	/// buffer.
	///
	/// # Panics
	///
	/// Panics if `index` is greater than the current length of the string or if
	/// it is not on a valid UTF-8 character boundary.
	#[inline]
	pub fn insert(&mut self, index: usize, s: &str) {
		match &mut self.0 {
			Repr::Inline(buf) => {
				// First check if index is on a valid char boundary
				assert!(
					buf.is_char_boundary(index),
					"index is not on a valid UTF-8 character boundary"
				);

				if buf.insert_str(index, s).is_err() {
					// Inline buffer doesn't have enough capacity, promote to heap
					let old_len = buf.len();
					let new_len = old_len + s.len();
					let mut heap = BytesMut::with_capacity(new_len.next_power_of_two());

					// Copy the part before the insertion point
					heap.extend_from_slice(&buf.as_bytes()[..index]);
					// Insert the new string
					heap.extend_from_slice(s.as_bytes());
					// Copy the part after the insertion point
					heap.extend_from_slice(&buf.as_bytes()[index..]);

					// SAFETY: We copy valid UTF-8 bytes from buf (before and after index)
					// and valid UTF-8 bytes from s. The result is valid UTF-8 because we
					// insert at a valid char boundary (verified by the assert above).
					*self = Self(Repr::Heap(unsafe { StrMut::from_inner_unchecked(heap) }));
				}
			},
			Repr::Heap(heap) => {
				assert!(
					heap.is_char_boundary(index),
					"index is not on a valid UTF-8 character boundary"
				);
				// SAFETY: We have mutable access to the SmolStrMut, so we can get
				// mutable access to the inner BytesMut.
				let inner = unsafe { heap.inner_mut() };

				let len = inner.len();
				let string_len = s.len();
				inner.reserve(string_len);

				// SAFETY: Move the bytes starting from `index` to their new location
				// `string_len` bytes ahead. This is safe because we checked there is
				// sufficient capacity, and `index` is a char boundary.
				unsafe {
					let ptr = inner.as_mut_ptr();
					core::ptr::copy(ptr.add(index), ptr.add(index + string_len), len - index);
				}

				// SAFETY: Copy the new string slice into the vacated region if
				// `index != len`, or into the uninitialized spare capacity otherwise.
				// The source (s) and destination do not overlap because s is an
				// independent string slice.
				unsafe {
					core::ptr::copy_nonoverlapping(
						s.as_ptr(),
						inner.as_mut_ptr().add(index),
						string_len,
					);
				}

				// SAFETY: We've just initialized `string_len` bytes at position
				// `index`, and moved the existing bytes to make room. The total
				// length is now len + string_len. The resulting bytes are valid
				// UTF-8 because we inserted at a char boundary and s is valid UTF-8.
				unsafe {
					inner.set_len(len + string_len);
				}
			},
		}
	}
}

impl fmt::Write for SmolStrMut {
	#[inline]
	fn write_str(&mut self, s: &str) -> fmt::Result {
		self.push_str(s);
		Ok(())
	}
}

// ============================
// IntoSmolStr
// ============================

/// Convert value to [`SmolStr`]/[`SmolStrMut`] using [`fmt::Display`],
/// potentially without allocating.
///
/// Almost identical to [`ToString`], but converts to [`SmolStr`]/[`SmolStrMut`]
/// instead.
pub trait IntoSmolStr: fmt::Display {
	/// Convert value to [`SmolStr`].
	fn into_smolstr(self) -> SmolStr
	where
		Self: Sized,
	{
		self.to_smolstr()
	}

	/// Convert value to [`SmolStrMut`].
	fn into_smolmut(self) -> SmolStrMut
	where
		Self: Sized,
	{
		self.into_smolstr().into()
	}

	/// Convert value to [`SmolStr`].
	fn to_smolstr(&self) -> SmolStr {
		format_smol!("{self}")
	}

	/// Convert value to [`SmolStrMut`].
	fn to_smolmut(&self) -> SmolStrMut {
		self.to_smolstr().into()
	}
}

impl IntoSmolStr for &str {
	#[inline]
	fn into_smolstr(self) -> SmolStr {
		SmolStr::new(self)
	}

	#[inline]
	fn into_smolmut(self) -> SmolStrMut {
		SmolStrMut::new(self)
	}

	#[inline]
	fn to_smolstr(&self) -> SmolStr {
		SmolStr::new(self)
	}

	#[inline]
	fn to_smolmut(&self) -> SmolStrMut {
		SmolStrMut::new(self)
	}
}

impl IntoSmolStr for &mut str {
	#[inline]
	fn into_smolstr(self) -> SmolStr {
		SmolStr::new(self)
	}

	#[inline]
	fn into_smolmut(self) -> SmolStrMut {
		SmolStrMut::new(self)
	}

	#[inline]
	fn to_smolstr(&self) -> SmolStr {
		SmolStr::new(self)
	}

	#[inline]
	fn to_smolmut(&self) -> SmolStrMut {
		SmolStrMut::new(self)
	}
}

impl IntoSmolStr for SmolStr {
	#[inline]
	fn into_smolstr(self) -> SmolStr {
		self
	}

	#[inline]
	fn into_smolmut(self) -> SmolStrMut {
		self.into()
	}

	#[inline]
	fn to_smolstr(&self) -> SmolStr {
		self.clone()
	}

	#[inline]
	fn to_smolmut(&self) -> SmolStrMut {
		SmolStrMut::new(self.as_str())
	}
}

impl IntoSmolStr for SmolStrMut {
	#[inline]
	fn into_smolstr(self) -> SmolStr {
		self.freeze()
	}

	#[inline]
	fn into_smolmut(self) -> SmolStrMut {
		self
	}

	#[inline]
	fn to_smolstr(&self) -> SmolStr {
		SmolStr::new(self.as_str())
	}

	#[inline]
	fn to_smolmut(&self) -> SmolStrMut {
		self.clone()
	}
}

impl IntoSmolStr for SmolCow<'_> {
	#[inline]
	fn into_smolstr(self) -> SmolStr {
		match self {
			SmolCow::Borrowed(s) => SmolStr::new(s),
			SmolCow::Owned(s) => s.freeze(),
		}
	}

	#[inline]
	fn into_smolmut(self) -> SmolStrMut {
		match self {
			SmolCow::Borrowed(s) => SmolStrMut::new(s),
			SmolCow::Owned(s) => s,
		}
	}

	#[inline]
	fn to_smolstr(&self) -> SmolStr {
		SmolStr::new(self.as_str())
	}

	#[inline]
	fn to_smolmut(&self) -> SmolStrMut {
		SmolStrMut::new(self.as_str())
	}
}

impl IntoSmolStr for String {
	#[inline]
	fn into_smolstr(self) -> SmolStr {
		SmolStr(Repr::Heap(self.into()))
	}

	#[inline]
	fn into_smolmut(self) -> SmolStrMut {
		// SAFETY: String guarantees its contents are valid UTF-8. We convert
		// into bytes and then wrap in StrMut, preserving UTF-8 validity.
		SmolStrMut(Repr::Heap(unsafe {
			StrMut::from_inner_unchecked(Bytes::from(self.into_bytes()).into())
		}))
	}

	#[inline]
	fn to_smolstr(&self) -> SmolStr {
		self.as_str().into()
	}

	#[inline]
	fn to_smolmut(&self) -> SmolStrMut {
		self.as_str().into()
	}
}

impl IntoSmolStr for Str {
	#[inline]
	fn into_smolstr(self) -> SmolStr {
		SmolStr(Repr::Heap(self))
	}

	#[inline]
	fn into_smolmut(self) -> SmolStrMut {
		// SAFETY: Str guarantees its contents are valid UTF-8. Converting
		// to BytesMut preserves the UTF-8 bytes.
		SmolStrMut(Repr::Heap(unsafe { StrMut::from_inner_unchecked(self.into_inner().into()) }))
	}

	#[inline]
	fn to_smolstr(&self) -> SmolStr {
		SmolStr(Repr::Heap(self.clone()))
	}

	#[inline]
	fn to_smolmut(&self) -> SmolStrMut {
		self.deref().into()
	}
}

impl IntoSmolStr for Cow<'_, str> {
	#[inline]
	fn into_smolstr(self) -> SmolStr {
		match self {
			Cow::Borrowed(s) => SmolStr::new(s),
			Cow::Owned(s) => SmolStr(Repr::Heap(s.into())),
		}
	}

	#[inline]
	fn into_smolmut(self) -> SmolStrMut {
		match self {
			Cow::Borrowed(s) => SmolStrMut::new(s),
			Cow::Owned(s) => s.into_smolmut(),
		}
	}

	#[inline]
	fn to_smolstr(&self) -> SmolStr {
		match self {
			Cow::Borrowed(s) => SmolStr::new(s),
			Cow::Owned(s) => s.into_smolstr(),
		}
	}

	#[inline]
	fn to_smolmut(&self) -> SmolStrMut {
		match self {
			Cow::Borrowed(s) => SmolStrMut::new(s),
			Cow::Owned(s) => s.into_smolmut(),
		}
	}
}

impl IntoSmolStr for Box<str> {
	#[inline]
	fn into_smolstr(self) -> SmolStr {
		SmolStr(Repr::Heap(self.into()))
	}

	#[inline]
	fn into_smolmut(self) -> SmolStrMut {
		// SAFETY: Box<str> guarantees its contents are valid UTF-8. Converting
		// to boxed bytes and then to BytesMut preserves the UTF-8 bytes.
		SmolStrMut(Repr::Heap(unsafe {
			StrMut::from_inner_unchecked(Bytes::from(self.into_boxed_bytes()).into())
		}))
	}

	#[inline]
	fn to_smolstr(&self) -> SmolStr {
		SmolStr::new(self.as_ref())
	}

	#[inline]
	fn to_smolmut(&self) -> SmolStrMut {
		SmolStrMut::new(self.as_ref())
	}
}

impl IntoSmolStr for Arc<str> {
	#[inline]
	fn into_smolstr(self) -> SmolStr {
		let bytes: Arc<[u8]> = self.into();
		// SAFETY: Arc<str> guarantees its contents are valid UTF-8. Converting
		// to Arc<[u8]> preserves the bytes without modification.
		SmolStr(Repr::Heap(unsafe { Str::from_inner_unchecked(Bytes::from_owner(bytes)) }))
	}

	#[inline]
	fn into_smolmut(self) -> SmolStrMut {
		SmolStrMut::new(self.as_ref())
	}

	#[inline]
	fn to_smolstr(&self) -> SmolStr {
		SmolStr::new(self.as_ref())
	}

	#[inline]
	fn to_smolmut(&self) -> SmolStrMut {
		SmolStrMut::new(self.as_ref())
	}
}

impl<T> IntoSmolStr for &T
where
	T: fmt::Display + ?Sized,
{
	default fn into_smolstr(self) -> SmolStr {
		format_smol!("{}", self)
	}

	#[inline]
	default fn into_smolmut(self) -> SmolStrMut {
		format_smolmut!("{}", self)
	}

	#[inline]
	default fn to_smolstr(&self) -> SmolStr {
		format_smol!("{}", *self)
	}

	#[inline]
	default fn to_smolmut(&self) -> SmolStrMut {
		format_smolmut!("{}", *self)
	}
}

// ============================
// From
// ============================

impl str::FromStr for SmolStr {
	type Err = Infallible;

	#[inline]
	fn from_str(s: &str) -> Result<Self, Self::Err> {
		Ok(Self::from(s))
	}
}

impl From<fmt::Arguments<'_>> for SmolStr {
	#[inline]
	fn from(args: fmt::Arguments<'_>) -> Self {
		args.into_smolstr()
	}
}

impl From<&str> for SmolStr {
	#[inline]
	fn from(s: &str) -> Self {
		Self::new(s)
	}
}

impl From<&mut str> for SmolStr {
	#[inline]
	fn from(s: &mut str) -> Self {
		Self::new(s)
	}
}

impl From<&str> for SmolStrMut {
	#[inline]
	fn from(s: &str) -> Self {
		Self::new(s)
	}
}

impl From<&mut str> for SmolStrMut {
	#[inline]
	fn from(s: &mut str) -> Self {
		Self::new(s)
	}
}

impl From<&String> for SmolStr {
	#[inline]
	fn from(s: &String) -> Self {
		Self::new(s)
	}
}

impl From<&String> for SmolStrMut {
	#[inline]
	fn from(s: &String) -> Self {
		Self::new(s)
	}
}

impl From<String> for SmolStr {
	#[inline(always)]
	fn from(text: String) -> Self {
		Self(Repr::Heap(text.into()))
	}
}

impl From<String> for SmolStrMut {
	#[inline(always)]
	fn from(text: String) -> Self {
		// SAFETY: String guarantees its contents are valid UTF-8. Converting
		// into bytes preserves those UTF-8 bytes.
		Self(Repr::Heap(unsafe {
			StrMut::from_inner_unchecked(Bytes::from(text.into_bytes()).into())
		}))
	}
}

impl From<&Str> for SmolStr {
	#[inline]
	fn from(s: &Str) -> Self {
		Self(Repr::Heap(s.clone()))
	}
}

impl From<&Str> for SmolStrMut {
	#[inline]
	fn from(s: &Str) -> Self {
		Self::new(&**s)
	}
}

impl From<Str> for SmolStr {
	#[inline(always)]
	fn from(text: Str) -> Self {
		Self(Repr::Heap(text))
	}
}

impl From<Str> for SmolStrMut {
	#[inline(always)]
	fn from(text: Str) -> Self {
		// SAFETY: Str guarantees its contents are valid UTF-8. Converting
		// to BytesMut preserves the UTF-8 bytes.
		Self(Repr::Heap(unsafe { StrMut::from_inner_unchecked(BytesMut::from(text.into_inner())) }))
	}
}

impl From<StrMut> for SmolStr {
	#[inline]
	fn from(value: StrMut) -> Self {
		Self(Repr::Heap(value.freeze()))
	}
}

impl From<StrMut> for SmolStrMut {
	#[inline]
	fn from(value: StrMut) -> Self {
		Self(Repr::Heap(value))
	}
}

impl<'a> From<Cow<'a, str>> for SmolStr {
	#[inline]
	fn from(s: Cow<'a, str>) -> Self {
		match s {
			Cow::Borrowed(borrowed) => Self::new(borrowed),
			Cow::Owned(owned) => Self(Repr::Heap(owned.into())),
		}
	}
}

impl<'a> From<Cow<'a, str>> for SmolStrMut {
	#[inline]
	fn from(s: Cow<'a, str>) -> Self {
		match s {
			Cow::Borrowed(borrowed) => borrowed.into(),
			Cow::Owned(owned) => owned.into(),
		}
	}
}

impl From<SmolStr> for Str {
	#[inline(always)]
	fn from(text: SmolStr) -> Self {
		match text.0 {
			Repr::Heap(data) => data,
			_ => text.as_str().into(),
		}
	}
}

impl From<SmolStrMut> for Str {
	#[inline(always)]
	fn from(text: SmolStrMut) -> Self {
		match text.0 {
			Repr::Heap(data) => data.freeze(),
			_ => text.as_str().into(),
		}
	}
}

impl From<SmolStr> for String {
	#[inline(always)]
	fn from(text: SmolStr) -> Self {
		text.as_str().into()
	}
}

impl From<SmolStrMut> for String {
	#[inline(always)]
	fn from(text: SmolStrMut) -> Self {
		text.as_str().into()
	}
}

impl From<SmolStr> for Bytes {
	#[inline(always)]
	fn from(text: SmolStr) -> Self {
		match text.0 {
			Repr::Heap(data) => data.into(),
			Repr::Inline(buf) => Self::copy_from_slice(buf.as_bytes()),
		}
	}
}

impl From<SmolStrMut> for Bytes {
	#[inline(always)]
	fn from(text: SmolStrMut) -> Self {
		match text.0 {
			Repr::Heap(data) => data.into_inner().into(),
			Repr::Inline(buf) => Self::copy_from_slice(buf.as_bytes()),
		}
	}
}

impl From<SmolStr> for BytesMut {
	#[inline(always)]
	fn from(value: SmolStr) -> Self {
		match value.0 {
			Repr::Heap(data) => data.into_inner().into(),
			Repr::Inline(buf) => Self::from(buf.as_bytes()),
		}
	}
}

impl From<SmolStrMut> for BytesMut {
	#[inline(always)]
	fn from(text: SmolStrMut) -> Self {
		match text.0 {
			Repr::Heap(data) => data.into_inner(),
			Repr::Inline(buf) => Self::from(buf.as_bytes()),
		}
	}
}

impl From<SmolStr> for StrMut {
	#[inline(always)]
	fn from(value: SmolStr) -> Self {
		// SAFETY: SmolStr is guaranteed to contain valid UTF-8, so converting it to
		// BytesMut and then to StrMut preserves UTF-8 validity.
		unsafe { Self::from_inner_unchecked(BytesMut::from(value)) }
	}
}

impl From<SmolStrMut> for StrMut {
	#[inline(always)]
	fn from(value: SmolStrMut) -> Self {
		match value.0 {
			Repr::Heap(data) => data,
			// SAFETY: buf contains valid UTF-8 (guaranteed by SmolStrMut invariants).
			// Converting to BytesMut preserves the UTF-8 bytes.
			Repr::Inline(buf) => unsafe { Self::from_inner_unchecked(BytesMut::from(buf.as_bytes())) },
		}
	}
}

impl From<SmolStrMut> for SmolStr {
	#[inline]
	fn from(value: SmolStrMut) -> Self {
		value.freeze()
	}
}

impl From<SmolStr> for SmolStrMut {
	#[inline]
	fn from(value: SmolStr) -> Self {
		match value.0 {
			Repr::Inline(buf) => Self(Repr::Inline(buf)),
			// SAFETY: heap contains valid UTF-8 (guaranteed by SmolStr invariants).
			Repr::Heap(heap) => unsafe { Self::from_utf8_unchecked_owned(heap.into_inner()) },
		}
	}
}

impl From<Arc<str>> for SmolStr {
	#[inline]
	fn from(value: Arc<str>) -> Self {
		let bytes: Arc<[u8]> = value.into();
		// SAFETY: Arc<str> guarantees its contents are valid UTF-8. Converting
		// to Arc<[u8]> preserves the bytes without modification.
		Self(Repr::Heap(unsafe { Str::from_inner_unchecked(Bytes::from_owner(bytes)) }))
	}
}

impl From<Box<str>> for SmolStr {
	#[inline]
	fn from(value: Box<str>) -> Self {
		Self(Repr::Heap(value.into()))
	}
}

impl From<Box<str>> for SmolStrMut {
	#[inline]
	fn from(value: Box<str>) -> Self {
		// SAFETY: Box<str> guarantees its contents are valid UTF-8. Converting
		// to boxed bytes and then to BytesMut preserves the UTF-8 bytes.
		Self(Repr::Heap(unsafe {
			StrMut::from_inner_unchecked(Bytes::from(value.into_boxed_bytes()).into())
		}))
	}
}

// ============================
// Serde
// ============================

impl serde::Serialize for SmolStr {
	fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
	where
		S: serde::Serializer,
	{
		self.as_str().serialize(serializer)
	}
}

impl<'de> serde::Deserialize<'de> for SmolStr {
	fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
	where
		D: serde::Deserializer<'de>,
	{
		use serde::de::{Error, Unexpected};
		struct SmolStrVisitor;

		impl serde::de::Visitor<'_> for SmolStrVisitor {
			type Value = SmolStr;

			fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
				formatter.write_str("a string")
			}

			fn visit_str<E: Error>(self, v: &str) -> Result<Self::Value, E> {
				Ok(SmolStr::from(v))
			}

			fn visit_string<E: Error>(self, v: String) -> Result<Self::Value, E> {
				Ok(SmolStr::from(v))
			}

			fn visit_bytes<E: Error>(self, v: &[u8]) -> Result<Self::Value, E> {
				match str::from_utf8(v) {
					Ok(s) => Ok(SmolStr::from(s)),
					Err(_) => Err(Error::invalid_value(Unexpected::Bytes(v), &self)),
				}
			}

			fn visit_byte_buf<E: Error>(self, v: Vec<u8>) -> Result<Self::Value, E> {
				match String::from_utf8(v) {
					Ok(s) => Ok(SmolStr::from(s)),
					Err(e) => Err(Error::invalid_value(Unexpected::Bytes(&e.into_bytes()), &self)),
				}
			}
		}

		deserializer.deserialize_str(SmolStrVisitor)
	}
}

impl serde::Serialize for SmolCow<'_> {
	fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
	where
		S: serde::Serializer,
	{
		self.as_str().serialize(serializer)
	}
}

impl<'de: 'a, 'a> serde::Deserialize<'de> for SmolCow<'a> {
	fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
	where
		D: serde::Deserializer<'de>,
	{
		use serde::de::{Error, Unexpected};

		struct SmolCowVisitor;

		impl<'de> serde::de::Visitor<'de> for SmolCowVisitor {
			type Value = SmolCow<'de>;

			fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
				formatter.write_str("a string")
			}

			fn visit_str<E: Error>(self, v: &str) -> Result<Self::Value, E> {
				Ok(SmolCow::Owned(SmolStrMut::from(v)))
			}

			fn visit_borrowed_str<E: Error>(self, v: &'de str) -> Result<Self::Value, E> {
				Ok(SmolCow::Borrowed(v))
			}

			fn visit_string<E: Error>(self, v: String) -> Result<Self::Value, E> {
				Ok(SmolCow::Owned(SmolStrMut::from(v)))
			}

			fn visit_bytes<E: Error>(self, v: &[u8]) -> Result<Self::Value, E> {
				match str::from_utf8(v) {
					Ok(s) => Ok(SmolCow::Owned(SmolStrMut::from(s))),
					Err(_) => Err(Error::invalid_value(Unexpected::Bytes(v), &self)),
				}
			}

			fn visit_borrowed_bytes<E>(self, v: &'de [u8]) -> Result<Self::Value, E>
			where
				E: serde::de::Error,
			{
				match str::from_utf8(v) {
					Ok(s) => Ok(SmolCow::Borrowed(s)),
					Err(_) => Err(Error::invalid_value(Unexpected::Bytes(v), &self)),
				}
			}

			fn visit_byte_buf<E: Error>(self, v: Vec<u8>) -> Result<Self::Value, E> {
				match String::from_utf8(v) {
					Ok(s) => Ok(SmolCow::Owned(SmolStrMut::from(s))),
					Err(e) => Err(Error::invalid_value(Unexpected::Bytes(&e.into_bytes()), &self)),
				}
			}
		}

		deserializer.deserialize_str(SmolCowVisitor)
	}
}

/// A clone-on-write smart pointer for strings with small string optimization.
///
/// The type `SmolCow` is a smart pointer providing clone-on-write
/// functionality: it can enclose and provide immutable access to borrowed data,
/// and clone the data lazily when mutation or ownership is required.
///
/// `SmolCow` implements `Deref`, which means that you can call non-mutating
/// methods directly on the data it encloses.
#[derive(Clone)]
pub enum SmolCow<'a> {
	/// Borrowed data.
	Borrowed(&'a str),
	/// Owned data.
	Owned(SmolStrMut),
}

impl<'a> SmolCow<'a> {
	/// Creates a new `SmolCow` from a string slice.
	#[inline]
	pub const fn from_str(s: &'a str) -> Self {
		SmolCow::Borrowed(s)
	}

	/// Creates a new `SmolCow` from an owned `SmolStrMut`.
	#[inline]
	pub const fn from_owned(s: SmolStrMut) -> Self {
		SmolCow::Owned(s)
	}

	/// Returns a `&str` slice of this `SmolCow`.
	#[inline]
	pub fn as_str(&self) -> &str {
		match self {
			SmolCow::Borrowed(s) => s,
			SmolCow::Owned(s) => s.as_str(),
		}
	}

	/// Returns the length of the string in bytes.
	#[inline]
	pub fn len(&self) -> usize {
		self.as_str().len()
	}

	/// Returns `true` if the string has a length of zero bytes.
	#[inline]
	pub fn is_empty(&self) -> bool {
		self.as_str().is_empty()
	}

	/// Returns true if the data is borrowed.
	#[inline]
	pub const fn is_borrowed(&self) -> bool {
		matches!(self, SmolCow::Borrowed(_))
	}

	/// Returns true if the data is owned.
	#[inline]
	pub const fn is_owned(&self) -> bool {
		matches!(self, SmolCow::Owned(_))
	}

	/// Assigns the slice passed to the `SmolCow`.
	#[inline]
	pub fn assign_slice_ref(&mut self, subset: &'a str) {
		match self {
			SmolCow::Owned(owned) => {
				// SAFETY: We copy valid UTF-8 bytes from `subset` to the beginning of
				// `owned`, then truncate to the copied length. The source is valid UTF-8,
				// so the result is valid UTF-8. The memory regions do not overlap
				// because subset is an independent string slice.
				unsafe {
					std::ptr::copy(subset.as_ptr(), owned.as_mut_ptr(), subset.len());
					owned.truncate(subset.len());
				}
			},
			SmolCow::Borrowed(_) => *self = SmolCow::Borrowed(subset),
		}
	}

	/// Assigns self to be equal to the range given within the `SmolCow` itself.
	#[inline]
	pub fn assign_range(&mut self, range: impl Into<std::ops::Range<usize>>) {
		let range = range.into();
		match self {
			SmolCow::Owned(owned) => {
				let len = range.end - range.start;
				// SAFETY: We copy a substring of valid UTF-8 bytes within `owned` to the
				// beginning, then truncate. The range is validated by the caller (or will
				// panic on invalid access). Both pointers derive from one mutable borrow,
				// and ptr::copy handles the overlap. The resulting bytes are valid UTF-8
				// because they're a substring of valid UTF-8.
				unsafe {
					if range.start > 0 {
						let base = owned.as_str_mut().as_mut_ptr();
						std::ptr::copy(base.add(range.start).cast_const(), base, len);
					}
					owned.truncate(len);
				}
			},
			SmolCow::Borrowed(s) => {
				*self = SmolCow::Borrowed(&s[range]);
			},
		}
	}

	/// Converts the `SmolCow` into a `SmolCow` that represents the specified
	/// range.
	///
	/// Clones the data if it is not already owned.
	///
	/// # Panics
	///
	/// Panics if the range is out of bounds.
	#[inline]
	pub fn into_range(self, range: impl Into<std::ops::Range<usize>>) -> Self {
		let range = range.into();
		match self {
			SmolCow::Borrowed(s) => SmolCow::Borrowed(&s[range]),
			SmolCow::Owned(mut owned) => {
				let len = range.end - range.start;
				// SAFETY: We copy a substring of valid UTF-8 bytes within `owned` to the
				// beginning, then truncate. The range is validated by the caller (or will
				// panic on invalid access). Both pointers derive from one mutable borrow,
				// and ptr::copy handles the overlap. The resulting bytes are valid UTF-8
				// because they're a substring of valid UTF-8.
				unsafe {
					if range.start > 0 {
						let base = owned.as_str_mut().as_mut_ptr();
						std::ptr::copy(base.add(range.start).cast_const(), base, len);
					}
					owned.truncate(len);
				}
				SmolCow::Owned(owned)
			},
		}
	}

	/// Converts the `SmolCow` into a `SmolCow` that represents the specified
	/// slice reference.
	///
	/// Clones the data if it is not already owned.
	///
	/// # Panics
	///
	/// Panics if the slice reference is out of bounds.
	#[inline]
	pub fn into_slice_ref(self, subset: &'a str) -> Self {
		match self {
			SmolCow::Borrowed(_) => SmolCow::Borrowed(subset),
			SmolCow::Owned(mut owned) => {
				// SAFETY: We copy valid UTF-8 bytes from `subset` to the beginning of
				// `owned`, then truncate to the copied length. The source is valid UTF-8,
				// so the result is valid UTF-8. The memory regions do not overlap
				// because subset is an independent string slice.
				unsafe {
					std::ptr::copy(subset.as_ptr(), owned.as_mut_ptr(), subset.len());
					owned.truncate(subset.len());
				}
				SmolCow::Owned(owned)
			},
		}
	}

	/// Converts the `SmolCow` into an owned `SmolCow`.
	#[inline]
	pub fn into_owned(self) -> SmolCow<'static> {
		match self {
			SmolCow::Borrowed(s) => SmolCow::Owned(s.into()),
			SmolCow::Owned(owned) => SmolCow::Owned(owned),
		}
	}

	/// Borrows the `SmolCow` with a new lifetime.
	#[inline]
	pub fn borrow(&self) -> SmolCow<'_> {
		match self {
			SmolCow::Borrowed(s) => SmolCow::Borrowed(s),
			SmolCow::Owned(o) => SmolCow::Borrowed(o.as_str()),
		}
	}

	/// Returns a new `SmolCow` with the given string appended.
	#[inline]
	pub fn push_str(&mut self, s: &str) {
		self.as_mut().push_str(s);
	}

	/// Returns a new `SmolCow` with the given character appended.
	#[inline]
	pub fn push(&mut self, c: char) {
		self.as_mut().push(c);
	}

	/// Truncates the string to the specified length.
	#[inline]
	pub fn truncate(&mut self, len: usize) {
		if self.len() > len {
			match self {
				SmolCow::Borrowed(s) => {
					*self = SmolCow::Borrowed(s.split_at(len).0);
				},
				SmolCow::Owned(s) => {
					s.truncate(len);
				},
			}
		}
	}

	/// Trims the string in place, removing leading and trailing whitespace.
	///
	/// This method does not allocate if the `SmolCow` is a borrowed `&str`.
	#[inline]
	pub fn trim(&mut self) {
		match self {
			SmolCow::Borrowed(s) => {
				*self = SmolCow::Borrowed(s.trim());
			},
			SmolCow::Owned(s) => {
				let range = s
					.substr_range(s.trim())
					.expect("substr_range should not fail");
				self.assign_range(range);
			},
		}
	}

	/// Trims the end of the string in place, removing trailing whitespace.
	///
	/// This method does not allocate if the `SmolCow` is a borrowed `&str`.
	#[inline]
	pub fn trim_end(&mut self) {
		match self {
			SmolCow::Borrowed(s) => {
				*self = SmolCow::Borrowed(s.trim_end());
			},
			SmolCow::Owned(s) => {
				let range = s
					.substr_range(s.trim_end())
					.expect("substr_range should not fail");
				self.assign_range(range);
			},
		}
	}

	/// Trims the start of the string in place, removing leading whitespace.
	///
	/// This method does not allocate if the `SmolCow` is a borrowed `&str`.
	#[inline]
	pub fn trim_start(&mut self) {
		match self {
			SmolCow::Borrowed(s) => {
				*self = SmolCow::Borrowed(s.trim_start());
			},
			SmolCow::Owned(s) => {
				let range = s
					.substr_range(s.trim_start())
					.expect("substr_range should not fail");
				self.assign_range(range);
			},
		}
	}

	/// Makes the string ASCII lowercase.
	#[inline]
	pub fn make_ascii_lowercase(&mut self) {
		// Only convert to owned if we actually need to modify
		if self.as_str().bytes().any(|b| b.is_ascii_uppercase()) {
			self.as_mut().make_ascii_lowercase();
		}
	}

	/// Makes the string ASCII uppercase.
	#[inline]
	pub fn make_ascii_uppercase(&mut self) {
		// Only convert to owned if we actually need to modify
		if self.as_str().bytes().any(|b| b.is_ascii_lowercase()) {
			self.as_mut().make_ascii_uppercase();
		}
	}
}

impl AsMut<SmolStrMut> for SmolCow<'_> {
	fn as_mut(&mut self) -> &mut SmolStrMut {
		match self {
			SmolCow::Borrowed(s) => {
				*self = SmolCow::Owned(SmolStrMut::new(s));
				match self {
					SmolCow::Owned(owned) => owned,
					_ => unreachable!(),
				}
			},
			SmolCow::Owned(owned) => owned,
		}
	}
}

// ============================
// Conversions
// ============================

impl<'a> From<&'a str> for SmolCow<'a> {
	#[inline]
	fn from(s: &'a str) -> Self {
		SmolCow::Borrowed(s)
	}
}

impl From<SmolStrMut> for SmolCow<'_> {
	#[inline]
	fn from(s: SmolStrMut) -> Self {
		SmolCow::Owned(s)
	}
}

impl From<SmolStr> for SmolCow<'_> {
	#[inline]
	fn from(s: SmolStr) -> Self {
		SmolCow::Owned(SmolStrMut::from(s))
	}
}

impl From<String> for SmolCow<'_> {
	#[inline]
	fn from(s: String) -> Self {
		SmolCow::Owned(SmolStrMut::from(s))
	}
}

impl<'a> From<Cow<'a, str>> for SmolCow<'a> {
	#[inline]
	fn from(cow: Cow<'a, str>) -> Self {
		match cow {
			Cow::Borrowed(s) => SmolCow::Borrowed(s),
			Cow::Owned(s) => SmolCow::Owned(SmolStrMut::from(s)),
		}
	}
}

impl<'a> From<SmolCow<'a>> for Cow<'a, str> {
	#[inline]
	fn from(smol: SmolCow<'a>) -> Self {
		match smol {
			SmolCow::Borrowed(s) => Cow::Borrowed(s),
			SmolCow::Owned(s) => Cow::Owned(s.as_str().to_string()),
		}
	}
}

impl<'a> From<SmolCow<'a>> for SmolStr {
	#[inline]
	fn from(smol: SmolCow<'a>) -> Self {
		match smol {
			SmolCow::Borrowed(s) => Self::from(s),
			SmolCow::Owned(s) => Self::from(s),
		}
	}
}

impl<'a> From<SmolCow<'a>> for SmolStrMut {
	#[inline]
	fn from(smol: SmolCow<'a>) -> Self {
		match smol {
			SmolCow::Borrowed(s) => Self::from(s),
			SmolCow::Owned(s) => s,
		}
	}
}

// ============================
// Deref and AsRef
// ============================

impl Deref for SmolCow<'_> {
	type Target = str;

	#[inline]
	fn deref(&self) -> &str {
		self.as_str()
	}
}

impl AsRef<str> for SmolCow<'_> {
	#[inline]
	fn as_ref(&self) -> &str {
		self.as_str()
	}
}

impl AsRef<[u8]> for SmolCow<'_> {
	#[inline]
	fn as_ref(&self) -> &[u8] {
		self.as_str().as_bytes()
	}
}

impl Borrow<str> for SmolCow<'_> {
	#[inline]
	fn borrow(&self) -> &str {
		self.as_str()
	}
}

// ============================
// Display and Debug
// ============================

impl fmt::Display for SmolCow<'_> {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		fmt::Display::fmt(self.as_str(), f)
	}
}

impl fmt::Debug for SmolCow<'_> {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		fmt::Debug::fmt(self.as_str(), f)
	}
}

// ============================
// Equality
// ============================

impl PartialEq for SmolCow<'_> {
	#[inline]
	fn eq(&self, other: &Self) -> bool {
		self.as_str() == other.as_str()
	}
}

impl Eq for SmolCow<'_> {}

impl PartialEq<str> for SmolCow<'_> {
	#[inline]
	fn eq(&self, other: &str) -> bool {
		self.as_str() == other
	}
}

impl PartialEq<SmolCow<'_>> for str {
	#[inline]
	fn eq(&self, other: &SmolCow<'_>) -> bool {
		self == other.as_str()
	}
}

impl PartialEq<&str> for SmolCow<'_> {
	#[inline]
	fn eq(&self, other: &&str) -> bool {
		self.as_str() == *other
	}
}

impl PartialEq<SmolCow<'_>> for &str {
	#[inline]
	fn eq(&self, other: &SmolCow<'_>) -> bool {
		*self == other.as_str()
	}
}

impl PartialEq<String> for SmolCow<'_> {
	#[inline]
	fn eq(&self, other: &String) -> bool {
		self.as_str() == other.as_str()
	}
}

impl PartialEq<SmolCow<'_>> for String {
	#[inline]
	fn eq(&self, other: &SmolCow<'_>) -> bool {
		self.as_str() == other.as_str()
	}
}

impl PartialEq<SmolStr> for SmolCow<'_> {
	#[inline]
	fn eq(&self, other: &SmolStr) -> bool {
		self.as_str() == other.as_str()
	}
}

impl PartialEq<SmolCow<'_>> for SmolStr {
	#[inline]
	fn eq(&self, other: &SmolCow<'_>) -> bool {
		self.as_str() == other.as_str()
	}
}

impl PartialEq<SmolStrMut> for SmolCow<'_> {
	#[inline]
	fn eq(&self, other: &SmolStrMut) -> bool {
		self.as_str() == other.as_str()
	}
}

impl PartialEq<SmolCow<'_>> for SmolStrMut {
	#[inline]
	fn eq(&self, other: &SmolCow<'_>) -> bool {
		self.as_str() == other.as_str()
	}
}

// ============================
// Ordering
// ============================

impl PartialOrd for SmolCow<'_> {
	#[inline]
	fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
		Some(self.cmp(other))
	}
}

impl Ord for SmolCow<'_> {
	#[inline]
	fn cmp(&self, other: &Self) -> Ordering {
		self.as_str().cmp(other.as_str())
	}
}

impl PartialOrd<str> for SmolCow<'_> {
	#[inline(always)]
	fn partial_cmp(&self, other: &str) -> Option<Ordering> {
		self.as_str().partial_cmp(other)
	}
}

impl PartialOrd<SmolCow<'_>> for str {
	#[inline(always)]
	fn partial_cmp(&self, other: &SmolCow<'_>) -> Option<Ordering> {
		self.partial_cmp(other.as_str())
	}
}

impl PartialOrd<&str> for SmolCow<'_> {
	#[inline(always)]
	fn partial_cmp(&self, other: &&str) -> Option<Ordering> {
		self.partial_cmp(*other)
	}
}

impl PartialOrd<SmolCow<'_>> for &str {
	#[inline(always)]
	fn partial_cmp(&self, other: &SmolCow<'_>) -> Option<Ordering> {
		(*self).partial_cmp(other)
	}
}

// ============================
// Hash
// ============================

impl Hash for SmolCow<'_> {
	#[inline]
	fn hash<H: Hasher>(&self, state: &mut H) {
		self.as_str().hash(state);
	}
}

// ============================
// Default
// ============================

impl Default for SmolCow<'_> {
	#[inline]
	fn default() -> Self {
		SmolCow::Owned(SmolStrMut::default())
	}
}

// ============================
// Tests
// ============================

#[cfg(test)]
mod tests {
	use serde_json as json;

	use super::*;

	const PREFIX: &str = "prefix__";
	const REMAINDER: &str = "abcdefghijklmnopjklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
	const LONG_TEXT: &str = "prefix__abcdefghijklmnopjklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
	const SPACED: &str = "   prefix__abcdefghijklmnopjklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

	fn heap_ptr(s: &SmolStr) -> (*const u8, usize) {
		match &s.0 {
			Repr::Heap(data) => (data.inner().as_ptr(), data.len()),
			_ => panic!("expected heap representation"),
		}
	}

	#[test]
	fn shared_buffer_prefix_slice_is_not_equal_to_parent() {
		let parent = SmolStr::new(LONG_TEXT);
		let prefix = parent.slice(..PREFIX.len());
		// Both share the parent's start pointer; equality must still compare
		// lengths, not just pointer identity.
		assert_eq!(heap_ptr(&prefix).0, heap_ptr(&parent).0);
		assert_ne!(prefix, parent);
		assert_ne!(parent, prefix);
		assert_eq!(prefix, SmolStr::new(PREFIX));
		assert_eq!(parent, parent.slice(..));
	}

	#[test]
	fn strip_prefix_reuses_heap_storage() {
		let value = SmolStr::new(LONG_TEXT);
		assert!(value.is_spilled());

		let (orig_ptr, orig_len) = heap_ptr(&value);
		assert_eq!(orig_len, LONG_TEXT.len());

		let remainder = value.strip_prefix(PREFIX).expect("prefix matches");
		assert_eq!(remainder.as_str(), REMAINDER);

		let (rem_ptr, rem_len) = heap_ptr(&remainder);
		assert_eq!(rem_len, REMAINDER.len());

		// SAFETY: both pointers originate from the same allocation provided by
		// bytes::Bytes, so offset_from is valid.
		unsafe {
			assert_eq!(rem_ptr.offset_from(orig_ptr) as usize, PREFIX.len());
		}
	}

	#[test]
	fn split_at_reuses_heap_storage() {
		let value = SmolStr::new(LONG_TEXT);
		assert!(value.is_spilled());

		let split_at = PREFIX.len();
		let (left, right) = value.split_at(split_at);

		assert_eq!(left.as_str(), &LONG_TEXT[..split_at]);
		assert_eq!(right.as_str(), &LONG_TEXT[split_at..]);

		let (orig_ptr, _) = heap_ptr(&value);
		let (left_ptr, left_len) = heap_ptr(&left);
		let (right_ptr, right_len) = heap_ptr(&right);

		assert_eq!(left_len, split_at);
		assert_eq!(right_len, LONG_TEXT.len() - split_at);

		// SAFETY: All pointers originate from the same allocation provided by
		// bytes::Bytes, so offset_from is valid.
		unsafe {
			assert_eq!(left_ptr.offset_from(orig_ptr) as usize, 0);
			assert_eq!(right_ptr.offset_from(orig_ptr) as usize, split_at);
		}
	}

	#[test]
	fn trim_start_reuses_heap_storage() {
		let value = SmolStr::new(SPACED);
		assert!(value.is_spilled());

		let trimmed = value.trim_start();
		assert_eq!(trimmed.as_str(), LONG_TEXT);

		let (orig_ptr, _) = heap_ptr(&value);
		let (trim_ptr, _) = heap_ptr(&trimmed);

		// SAFETY: Both pointers originate from the same allocation provided by
		// bytes::Bytes, so offset_from is valid.
		unsafe {
			assert_eq!(trim_ptr.offset_from(orig_ptr) as usize, 3);
		}
	}

	#[test]
	fn test_add_operations() {
		// Test SmolStr + &str
		let s1 = SmolStr::new("hello");
		let result = s1 + " world";
		assert_eq!(result.as_str(), "hello world");

		// Test &str + SmolStr
		let s2 = SmolStr::new("world");
		let result = "hello " + s2;
		assert_eq!(result.as_str(), "hello world");

		// Test &SmolStr + &str
		let s3 = SmolStr::new("hello");
		let result = &s3 + " world";
		assert_eq!(result.as_str(), "hello world");

		// Test &str + &SmolStr
		let s4 = SmolStr::new("world");
		let result = "hello " + &s4;
		assert_eq!(result.as_str(), "hello world");

		// Test SmolStrMut + &str
		let m1 = SmolStrMut::new("hello");
		let result = m1 + " world";
		assert_eq!(result.as_str(), "hello world");

		// Test &str + SmolStrMut
		let m2 = SmolStrMut::new("world");
		let result = "hello " + m2;
		assert_eq!(result.as_str(), "hello world");

		// Test &SmolStrMut + &str
		let m3 = SmolStrMut::new("hello");
		let result = &m3 + " world";
		assert_eq!(result.as_str(), "hello world");

		// Test &str + &SmolStrMut
		let m4 = SmolStrMut::new("world");
		let result = "hello " + &m4;
		assert_eq!(result.as_str(), "hello world");

		// Test with heap-allocated strings
		let long = SmolStr::new("this is a very long string that will be heap allocated");
		let result = &long + " and more";
		assert_eq!(
			result.as_str(),
			"this is a very long string that will be heap allocated and more"
		);
	}

	#[test]
	fn test_insert_inline_to_heap_promotion() {
		// Test that insert properly promotes from inline to heap when needed
		let mut s = SmolStrMut::new("hello world");
		assert!(!s.is_spilled()); // Should be inline

		// Insert something that will exceed inline capacity
		s.insert(6, "beautiful and wonderful ");
		assert_eq!(s.as_str(), "hello beautiful and wonderful world");
		assert!(s.is_spilled()); // Should now be on heap

		// Test insert at the beginning
		let mut s = SmolStrMut::new("world");
		s.insert(0, "hello ");
		assert_eq!(s.as_str(), "hello world");

		// Test insert at the end
		let mut s = SmolStrMut::new("hello");
		s.insert(5, " world");
		assert_eq!(s.as_str(), "hello world");

		// Test insert that triggers promotion with more text
		let mut s = SmolStrMut::new("12345678901234567890"); // 20 chars, near limit
		s.insert(10, "ABCDE"); // This will exceed inline capacity
		assert_eq!(s.as_str(), "1234567890ABCDE1234567890");
		assert!(s.is_spilled());
	}

	#[test]
	fn test_cow_basic_operations() {
		// Test borrowed variant
		let s = "hello world";
		let cow = SmolCow::from(s);
		assert!(cow.is_borrowed());
		assert!(!cow.is_owned());
		assert_eq!(cow.as_str(), "hello world");
		assert_eq!(cow.len(), 11);
		assert!(!cow.is_empty());

		// Test owned variant
		let mut_str = SmolStrMut::new("hello");
		let cow = SmolCow::from(mut_str);
		assert!(!cow.is_borrowed());
		assert!(cow.is_owned());
		assert_eq!(cow.as_str(), "hello");
	}

	#[test]
	fn test_cow_to_mut() {
		// Start with borrowed
		let s = "hello";
		let mut cow = SmolCow::from(s);
		assert!(cow.is_borrowed());

		// Convert to owned via as_mut
		let mutable = cow.as_mut();
		mutable.push_str(" world");
		assert!(cow.is_owned());
		assert_eq!(cow.as_str(), "hello world");

		// Second call to as_mut doesn't clone again
		let mutable2 = cow.as_mut();
		mutable2.push_str("!");
		assert_eq!(cow.as_str(), "hello world!");
	}

	#[test]
	fn test_cow_assign_slice_ref() {
		let original = "hello world";

		// Test with borrowed
		let mut cow = SmolCow::from(original);
		cow.assign_slice_ref("goodbye");
		assert!(cow.is_borrowed());
		assert_eq!(cow.as_str(), "goodbye");

		// Test with owned
		let mut cow = SmolCow::from(SmolStrMut::new("hello world"));
		cow.assign_slice_ref("bye");
		assert!(cow.is_owned());
		assert_eq!(cow.as_str(), "bye");
	}

	#[test]
	fn test_cow_assign_range() {
		// Test with borrowed
		let mut cow = SmolCow::from("hello world");
		cow.assign_range(0..5);
		assert!(cow.is_borrowed());
		assert_eq!(cow.as_str(), "hello");

		// Test with owned
		let mut cow = SmolCow::from(SmolStrMut::new("hello world"));
		cow.assign_range(6..11);
		assert!(cow.is_owned());
		assert_eq!(cow.as_str(), "world");

		// Test range in middle
		let mut cow = SmolCow::from(SmolStrMut::new("hello world"));
		cow.assign_range(3..8);
		assert_eq!(cow.as_str(), "lo wo");
	}

	#[test]
	fn test_cow_into_range() {
		// Test with borrowed
		let cow = SmolCow::from("hello world");
		let new_cow = cow.into_range(0..5);
		assert!(new_cow.is_borrowed());
		assert_eq!(new_cow.as_str(), "hello");

		// Test with owned
		let cow = SmolCow::from(SmolStrMut::new("hello world"));
		let new_cow = cow.into_range(6..11);
		assert!(new_cow.is_owned());
		assert_eq!(new_cow.as_str(), "world");
	}

	#[test]
	fn test_cow_into_slice_ref() {
		let original = "hello world";

		// Test with borrowed
		let cow = SmolCow::from(original);
		let subset = &original[6..11];
		let new_cow = cow.into_slice_ref(subset);
		assert!(new_cow.is_borrowed());
		assert_eq!(new_cow.as_str(), "world");

		// Test with owned
		let cow = SmolCow::from(SmolStrMut::new("hello world"));
		let new_cow = cow.into_slice_ref("new");
		assert!(new_cow.is_owned());
		assert_eq!(new_cow.as_str(), "new");
	}

	#[test]
	fn test_cow_mutating_operations() {
		// Test push_str
		let mut cow = SmolCow::from("hello");
		cow.push_str(" world");
		assert!(cow.is_owned());
		assert_eq!(cow.as_str(), "hello world");

		// Test push
		let mut cow = SmolCow::from("hello");
		cow.push('!');
		assert_eq!(cow.as_str(), "hello!");

		// Test truncate with borrowed
		let mut cow = SmolCow::from("hello world");
		cow.truncate(5);
		assert!(cow.is_borrowed());
		assert_eq!(cow.as_str(), "hello");

		// Test truncate with owned
		let mut cow = SmolCow::from(SmolStrMut::new("hello world"));
		cow.truncate(5);
		assert!(cow.is_owned());
		assert_eq!(cow.as_str(), "hello");
	}

	#[test]
	fn test_cow_trim_operations() {
		// Test trim with borrowed
		let mut cow = SmolCow::from("  hello world  ");
		cow.trim();
		assert!(cow.is_borrowed());
		assert_eq!(cow.as_str(), "hello world");

		// Test trim with owned
		let mut cow = SmolCow::from(SmolStrMut::new("  hello world  "));
		cow.trim();
		assert!(cow.is_owned());
		assert_eq!(cow.as_str(), "hello world");

		// Test trim_start with borrowed
		let mut cow = SmolCow::from("  hello");
		cow.trim_start();
		assert!(cow.is_borrowed());
		assert_eq!(cow.as_str(), "hello");

		// Test trim_start with owned
		let mut cow = SmolCow::from(SmolStrMut::new("  hello"));
		cow.trim_start();
		assert!(cow.is_owned());
		assert_eq!(cow.as_str(), "hello");

		// Test trim_end with borrowed
		let mut cow = SmolCow::from("hello  ");
		cow.trim_end();
		assert!(cow.is_borrowed());
		assert_eq!(cow.as_str(), "hello");

		// Test trim_end with owned
		let mut cow = SmolCow::from(SmolStrMut::new("hello  "));
		cow.trim_end();
		assert!(cow.is_owned());
		assert_eq!(cow.as_str(), "hello");
	}

	#[test]
	fn test_cow_ascii_case_conversion() {
		// Test lowercase - should not convert to owned if no changes needed
		let mut cow = SmolCow::from("hello");
		cow.make_ascii_lowercase();
		assert!(cow.is_borrowed()); // Still borrowed since no uppercase letters

		let mut cow = SmolCow::from("HELLO");
		cow.make_ascii_lowercase();
		assert!(cow.is_owned());
		assert_eq!(cow.as_str(), "hello");

		// Test uppercase
		let mut cow = SmolCow::from("HELLO");
		cow.make_ascii_uppercase();
		assert!(cow.is_borrowed()); // Still borrowed since no lowercase letters

		let mut cow = SmolCow::from("hello");
		cow.make_ascii_uppercase();
		assert!(cow.is_owned());
		assert_eq!(cow.as_str(), "HELLO");
	}

	#[test]
	fn test_cow_equality() {
		let cow1 = SmolCow::from("hello");
		let cow2 = SmolCow::from(SmolStrMut::new("hello"));

		assert_eq!(cow1, cow2);
		assert_eq!(cow1, "hello");
		assert_eq!("hello", cow1);
		assert_eq!(cow1, String::from("hello"));
		assert_eq!(cow1, SmolStr::new("hello"));
		assert_eq!(cow1, SmolStrMut::new("hello"));
	}

	#[test]
	fn test_cow_ordering() {
		let cow1 = SmolCow::from("apple");
		let cow2 = SmolCow::from("banana");
		let cow3 = SmolCow::from(SmolStrMut::new("apple"));

		assert!(cow1 < cow2);
		assert!(cow2 > cow1);
		assert_eq!(cow1.cmp(&cow3), std::cmp::Ordering::Equal);
	}

	#[test]
	fn test_cow_hash() {
		use std::{
			collections::hash_map::DefaultHasher,
			hash::{Hash, Hasher},
		};

		let cow1 = SmolCow::from("hello");
		let cow2 = SmolCow::from(SmolStrMut::new("hello"));

		let mut hasher1 = DefaultHasher::new();
		cow1.hash(&mut hasher1);
		let hash1 = hasher1.finish();

		let mut hasher2 = DefaultHasher::new();
		cow2.hash(&mut hasher2);
		let hash2 = hasher2.finish();

		assert_eq!(hash1, hash2);
	}

	#[test]
	fn test_cow_into_owned() {
		// From borrowed
		let cow = SmolCow::from("hello");
		let owned = cow.into_smolmut();
		assert_eq!(owned.as_str(), "hello");

		// From already owned
		let cow = SmolCow::from(SmolStrMut::new("world"));
		let owned = cow.into_smolmut();
		assert_eq!(owned.as_str(), "world");
	}

	#[test]
	fn test_cow_into_smolstr() {
		let cow = SmolCow::from("hello");
		let smol_str = cow.into_smolstr();
		assert_eq!(smol_str.as_str(), "hello");
	}

	#[test]
	fn test_cow_cow_conversion() {
		// From Cow to SmolCow
		let std_cow = Cow::Borrowed("hello");
		let smol_cow: SmolCow = std_cow.into();
		assert!(smol_cow.is_borrowed());

		let std_cow = Cow::<str>::Owned(String::from("world"));
		let smol_cow: SmolCow = std_cow.into();
		assert!(smol_cow.is_owned());

		// From SmolCow to Cow
		let smol_cow = SmolCow::from("hello");
		let std_cow: Cow<str> = smol_cow.into();
		assert!(matches!(std_cow, Cow::Borrowed(_)));
	}

	#[test]
	fn test_cow_from_conversions() {
		// From SmolStr
		let smol_str = SmolStr::new("hello");
		let cow: SmolCow = smol_str.into();
		assert!(cow.is_owned());
		assert_eq!(cow.as_str(), "hello");

		// From String
		let string = String::from("world");
		let cow: SmolCow = string.into();
		assert!(cow.is_owned());
		assert_eq!(cow.as_str(), "world");

		// Into SmolStr
		let cow = SmolCow::from("test");
		let smol_str: SmolStr = cow.into();
		assert_eq!(smol_str.as_str(), "test");

		// Into SmolStrMut
		let cow = SmolCow::from("test");
		let smol_str_mut: SmolStrMut = cow.into();
		assert_eq!(smol_str_mut.as_str(), "test");
	}

	#[test]
	fn test_cow_default() {
		let cow: SmolCow = Default::default();
		assert!(cow.is_owned());
		assert_eq!(cow.as_str(), "");
		assert!(cow.is_empty());
	}

	#[test]
	fn test_cow_deref_and_borrow() {
		let cow = SmolCow::from("hello world");

		// Test Deref
		assert_eq!(&*cow, "hello world");

		// Test AsRef<str>
		let s: &str = cow.as_ref();
		assert_eq!(s, "hello world");

		// Test AsRef<[u8]>
		let bytes: &[u8] = cow.as_ref();
		assert_eq!(bytes, b"hello world");
	}

	#[test]
	fn test_cow_serde() {
		// Test serialization
		let cow = SmolCow::from("hello world");
		let json = json::to_string(&cow).unwrap();
		assert_eq!(json, r#""hello world""#);

		// Test deserialization - will succesfully borrow!
		let deserialized: SmolCow = json::from_str(&json).unwrap();
		assert!(!deserialized.is_owned());
		assert_eq!(deserialized.as_str(), "hello world");

		// Test with owned variant
		let cow = SmolCow::from(SmolStrMut::new("test"));
		let json = json::to_string(&cow).unwrap();
		assert_eq!(json, r#""test""#);
	}

	// ============================
	// Empty and Boundary Tests (from lib/core/tests/smol_str.rs)
	// ============================

	#[test]
	fn test_empty_strings() {
		// Empty SmolStr
		let s = SmolStr::new("");
		assert!(s.is_empty());
		assert_eq!(s.len(), 0);
		assert!(!s.is_spilled());
		assert_eq!(s.as_str(), "");

		// Empty SmolStrMut
		let mut m = SmolStrMut::new("");
		assert!(m.is_empty());
		assert_eq!(m.len(), 0);
		assert!(!m.is_spilled());
		m.push_str("");
		assert!(m.is_empty());

		// Operations on empty
		assert_eq!(s.trim(), "");
		assert_eq!(s.trim_start(), "");
		assert_eq!(s.trim_end(), "");
		assert_eq!(s.strip_prefix("x"), None);
		assert_eq!(s.strip_suffix("x"), None);

		let (l, r) = s.split_at(0);
		assert!(l.is_empty());
		assert!(r.is_empty());

		// Empty split
		let parts: Vec<_> = s.split("x").collect();
		assert_eq!(parts.len(), 1);
		assert_eq!(parts[0], "");
	}

	#[test]
	fn test_exact_inline_capacity() {
		// 23 bytes - maximum inline capacity
		let text = "12345678901234567890123"; // exactly 23 bytes
		assert_eq!(text.len(), 23);

		let s = SmolStr::new(text);
		assert!(!s.is_spilled());
		assert_eq!(s.len(), 23);
		assert_eq!(s.as_str(), text);

		let m = SmolStrMut::new(text);
		assert!(!m.is_spilled());
		assert_eq!(m.len(), 23);
	}

	#[test]
	fn test_boundary_24_bytes() {
		// 24 bytes - first to require heap
		let text = "123456789012345678901234"; // 24 bytes
		assert_eq!(text.len(), 24);

		let s = SmolStr::new(text);
		assert!(s.is_spilled());
		assert_eq!(s.len(), 24);
		assert_eq!(s.as_str(), text);

		let m = SmolStrMut::new(text);
		assert!(m.is_spilled());
		assert_eq!(m.len(), 24);
	}

	#[test]
	fn test_slice_boundaries() {
		let text = "0123456789";
		let s = SmolStr::new(text);

		// Full range
		assert_eq!(s.slice(..), text);
		assert_eq!(s.slice(0..10), text);

		// Empty slices
		assert_eq!(s.slice(0..0), "");
		assert_eq!(s.slice(5..5), "");
		assert_eq!(s.slice(10..10), "");

		// Single char
		assert_eq!(s.slice(0..1), "0");
		assert_eq!(s.slice(9..10), "9");

		// Prefix/suffix
		assert_eq!(s.slice(..5), "01234");
		assert_eq!(s.slice(5..), "56789");
	}

	#[test]
	#[should_panic(expected = "self.is_char_boundary(new_len)")]
	fn test_truncate_non_char_boundary_inline() {
		let mut s = SmolStr::new("hello world 世界");
		s.truncate(13); // Middle of '世' (3 bytes)
	}

	#[test]
	#[should_panic(expected = "Index is not on a char boundary")]
	fn test_truncate_non_char_boundary_heap() {
		let mut s = SmolStr::new("hello world hello world 世界");
		s.truncate(26); // Middle of '世'
	}

	// ============================
	// UTF-8 Validation Tests
	// ============================

	#[test]
	fn test_from_utf8_valid() {
		let valid = b"hello world";
		let s = SmolStr::from_utf8(valid).unwrap();
		assert_eq!(s.as_str(), "hello world");

		let valid_utf8 = "hello 世界 🌍".as_bytes();
		let s = SmolStr::from_utf8(valid_utf8).unwrap();
		assert_eq!(s.as_str(), "hello 世界 🌍");
	}

	#[test]
	fn test_from_utf8_invalid() {
		let invalid = &[0xff, 0xfe, 0xfd];
		assert!(SmolStr::from_utf8(invalid).is_err());

		let invalid = &[0xc0, 0x80]; // Overlong encoding
		assert!(SmolStr::from_utf8(invalid).is_err());
	}

	#[test]
	fn test_from_utf8_owned_valid() {
		let valid = bytes::Bytes::from("hello world");
		let s = SmolStr::from_utf8_owned(valid).unwrap();
		assert_eq!(s.as_str(), "hello world");
		assert!(s.is_spilled());
	}

	#[test]
	fn test_from_utf8_owned_invalid() {
		let invalid = bytes::Bytes::from_static(&[0xff, 0xfe, 0xfd]);
		assert!(SmolStr::from_utf8_owned(invalid).is_err());
	}

	#[test]
	fn test_from_utf8_mut_valid() {
		let valid = b"hello";
		let m = SmolStrMut::from_utf8(valid).unwrap();
		assert_eq!(m.as_str(), "hello");
	}

	#[test]
	fn test_from_utf8_mut_invalid() {
		let invalid = &[0xff, 0xfe];
		assert!(SmolStrMut::from_utf8(invalid).is_err());
	}

	#[test]
	fn test_multibyte_char_boundaries() {
		// Emoji (4 bytes) + Chinese (3 bytes each)
		let text = "🌍世界";
		let s = SmolStr::new(text);
		assert_eq!(s.len(), 10); // 4 + 3 + 3

		// Split at valid boundaries
		let (l, r) = s.split_at(4);
		assert_eq!(l.as_str(), "🌍");
		assert_eq!(r.as_str(), "世界");

		// Truncate at valid boundary
		let mut s2 = SmolStr::new(text);
		s2.truncate(4);
		assert_eq!(s2.as_str(), "🌍");
	}

	#[test]
	#[should_panic(expected = "byte index 1 is not a char boundary")]
	fn test_split_at_invalid_boundary() {
		let s = SmolStr::new("世界");
		let _ = s.split_at(1); // Middle of '世' (3 bytes)
	}

	// ============================
	// Error Path Tests
	// ============================

	#[test]
	#[should_panic(expected = "len <= INLINE_CAP")]
	fn test_new_inline_panic() {
		let too_long = "123456789012345678901234"; // 24 bytes
		let _ = SmolStr::new_inline(too_long);
	}

	#[test]
	#[should_panic(expected = "len <= INLINE_CAP")]
	fn test_new_inline_mut_panic() {
		let too_long = "123456789012345678901234";
		let _ = SmolStrMut::new_inline(too_long);
	}

	#[test]
	#[should_panic(expected = "index is not on a valid UTF-8 character boundary")]
	fn test_insert_non_char_boundary_inline() {
		let mut m = SmolStrMut::new("世界");
		m.insert(1, "x"); // Middle of '世'
	}

	#[test]
	#[should_panic(expected = "index is not on a valid UTF-8 character boundary")]
	fn test_insert_non_char_boundary_heap() {
		let mut m = SmolStrMut::new("hello world hello world 世界");
		m.insert(25, "x"); // Middle of '世'
	}

	#[test]
	fn test_try_into_mut_success_inline() {
		let s = SmolStr::new("hello");
		let m = s.try_into_mut().unwrap();
		assert_eq!(m.as_str(), "hello");
		assert!(!m.is_spilled());
	}

	#[test]
	fn test_try_into_mut_success_heap_unique() {
		let s = SmolStr::new("hello world hello world!!");
		let m = s.try_into_mut().unwrap();
		assert_eq!(m.as_str(), "hello world hello world!!");
		assert!(m.is_spilled());
	}

	#[test]
	fn test_try_into_mut_failure_shared() {
		let s = SmolStr::new("hello world hello world!!");
		let s2 = s.clone();

		// Should fail because Bytes is shared
		let result = s.try_into_mut();
		assert!(result.is_err());

		// Original value should be recoverable
		let original = result.unwrap_err();
		assert_eq!(original.as_str(), "hello world hello world!!");
		assert_eq!(s2.as_str(), "hello world hello world!!");
	}

	// ============================
	// Inline→Heap Transition Tests
	// ============================

	#[test]
	fn test_push_str_inline_to_heap() {
		let mut m = SmolStrMut::new("12345678901234567890"); // 20 bytes
		assert!(!m.is_spilled());

		m.push_str("1234"); // Total 24 bytes
		assert!(m.is_spilled());
		assert_eq!(m.as_str(), "123456789012345678901234");
	}

	#[test]
	fn test_push_char_inline_to_heap() {
		let mut m = SmolStrMut::new("1234567890123456789012"); // 22 bytes
		assert!(!m.is_spilled());

		m.push('x');
		assert!(!m.is_spilled()); // 23 bytes, still inline

		m.push('y');
		assert!(m.is_spilled()); // 24 bytes, now heap
		assert_eq!(m.as_str(), "1234567890123456789012xy");
	}

	#[test]
	fn test_push_multibyte_char_promotion() {
		let mut m = SmolStrMut::new("12345678901234567890"); // 20 bytes
		assert!(!m.is_spilled());

		m.push('🌍'); // 4 bytes emoji
		assert!(m.is_spilled()); // 24 bytes total
		assert_eq!(m.as_str(), "12345678901234567890🌍");
	}

	#[test]
	fn test_reserve_triggers_promotion() {
		let mut m = SmolStrMut::new("hello"); // 5 bytes inline
		assert!(!m.is_spilled());

		m.reserve(20); // Reserve 20 more, total capacity 25
		assert!(m.is_spilled());
		assert_eq!(m.as_str(), "hello");

		// Should still have capacity
		m.push_str("12345678901234567890"); // 25 bytes total
		assert_eq!(m.as_str(), "hello12345678901234567890");
	}

	#[test]
	fn test_sequential_operations_promotion() {
		let mut m = SmolStrMut::new("abc");
		assert!(!m.is_spilled());

		for _ in 0..5 {
			m.push_str("1234"); // 4 bytes each
		}
		// Total: 3 + 20 = 23 bytes (still inline)
		assert!(!m.is_spilled());

		m.push('x');
		assert!(m.is_spilled()); // 24 bytes
	}

	// ============================
	// Slicing & Zero-Copy Tests
	// ============================

	#[test]
	fn test_strip_prefix_none() {
		let s = SmolStr::new("hello world");
		assert_eq!(s.strip_prefix("goodbye"), None);
		assert_eq!(s.strip_prefix("hello world!"), None);
	}

	#[test]
	fn test_strip_suffix_none() {
		let s = SmolStr::new("hello world");
		assert_eq!(s.strip_suffix("goodbye"), None);
		assert_eq!(s.strip_suffix("!hello world"), None);
	}

	#[test]
	fn test_slice_ref_inline() {
		let text = "hello world";
		let s = SmolStr::new(text);
		assert!(!s.is_spilled());

		let subset = &text[6..11]; // "world"
		let sliced = s.slice_ref(subset);
		assert_eq!(sliced.as_str(), "world");
		assert!(!sliced.is_spilled()); // Should still be inline
	}

	#[test]
	fn test_trim_end_zero_copy_heap() {
		let text = "hello world hello world hello   ";
		let s = SmolStr::new(text);
		assert!(s.is_spilled());

		let trimmed = s.trim_end();
		assert_eq!(trimmed.as_str(), "hello world hello world hello");
		assert!(trimmed.is_spilled());
	}

	#[test]
	fn test_split_iterator_complete() {
		let s = SmolStr::new("a,b,c,d,e");
		let parts: Vec<_> = s.split(",").collect();
		assert_eq!(parts.len(), 5);
		assert_eq!(parts[0], "a");
		assert_eq!(parts[1], "b");
		assert_eq!(parts[4], "e");

		// Multiple separators
		let s2 = SmolStr::new("a,,b");
		let parts2: Vec<_> = s2.split(",").collect();
		assert_eq!(parts2.len(), 3);
		assert_eq!(parts2[1], ""); // Empty between ,,
	}

	#[test]
	fn test_split_no_separator() {
		let s = SmolStr::new("hello");
		let parts: Vec<_> = s.split(",").collect();
		assert_eq!(parts.len(), 1);
		assert_eq!(parts[0], "hello");
	}

	// ============================
	// Uniqueness & Sharing Tests
	// ============================

	#[test]
	fn test_is_unique_inline() {
		let s = SmolStr::new("hello");
		assert!(s.is_unique()); // Inline always unique
	}

	#[test]
	fn test_is_unique_heap_unshared() {
		let s = SmolStr::new("hello world hello world!!");
		assert!(s.is_unique()); // Newly created heap string is unique
	}

	#[test]
	fn test_is_unique_heap_shared() {
		let s1 = SmolStr::new("hello world hello world!!");
		let s2 = s1.clone();

		assert!(!s1.is_unique()); // Now shared
		assert!(!s2.is_unique());
	}

	#[test]
	fn test_into_ascii_uppercase_leaves_shared_clone_untouched() {
		let s1 = SmolStr::new("hello world hello world!!");
		let s2 = s1.clone();

		// Shared storage: the conversion must copy, not mutate in place.
		let upper = s1.into_ascii_uppercase();

		assert_eq!(upper.as_str(), "HELLO WORLD HELLO WORLD!!");
		assert_eq!(s2.as_str(), "hello world hello world!!"); // Unchanged
	}

	#[test]
	fn test_promote_inline_to_heap() {
		let mut s = SmolStr::new("hello");
		assert!(!s.is_spilled());

		let heap_str = s.promote();
		assert_eq!(&**heap_str, "hello");
		assert!(s.is_spilled());
	}

	// ============================
	// Conversion Tests
	// ============================

	#[test]
	fn test_from_arc_str() {
		let arc: std::sync::Arc<str> = "hello world hello world".into();
		let s = SmolStr::from(arc);
		assert!(s.is_spilled());
		assert_eq!(s.as_str(), "hello world hello world");
	}

	#[test]
	fn test_from_box_str() {
		let boxed: Box<str> = "hello world".into();
		let s = SmolStr::from(boxed);
		assert_eq!(s.as_str(), "hello world");
	}

	#[test]
	fn test_from_cow_borrowed() {
		let cow = std::borrow::Cow::Borrowed("hello");
		let s = SmolStr::from(cow);
		assert_eq!(s.as_str(), "hello");
	}

	#[test]
	fn test_from_cow_owned() {
		let cow = std::borrow::Cow::<str>::Owned(String::from("hello world hello world"));
		let s = SmolStr::from(cow);
		assert_eq!(s.as_str(), "hello world hello world");
	}

	#[test]
	fn test_into_string() {
		let s = SmolStr::new("hello world");
		let string: String = s.into();
		assert_eq!(string, "hello world");
	}

	#[test]
	fn test_into_bytes() {
		let s = SmolStr::new("hello world hello world");
		let bytes: bytes::Bytes = s.into();
		assert_eq!(&bytes[..], b"hello world hello world");
	}

	#[test]
	fn test_into_bytes_inline() {
		let s = SmolStr::new("hello");
		let bytes: bytes::Bytes = s.into();
		assert_eq!(&bytes[..], b"hello");
	}

	#[test]
	fn test_from_iterator_char() {
		let chars = vec!['h', 'e', 'l', 'l', 'o'];
		let s: SmolStr = chars.into_iter().collect();
		assert_eq!(s.as_str(), "hello");
	}

	#[test]
	fn test_from_iterator_str_ref() {
		let strs = vec!["hello", " ", "world"];
		let s: SmolStr = strs.into_iter().collect();
		assert_eq!(s.as_str(), "hello world");
	}

	#[test]
	fn test_from_iterator_string() {
		let strings = vec![String::from("hello"), String::from(" "), String::from("world")];
		let s: SmolStr = strings.into_iter().collect();
		assert_eq!(s.as_str(), "hello world");
	}

	#[test]
	fn test_from_iterator_triggers_heap() {
		let s: SmolStr = "123456789012345678901234".chars().collect();
		assert!(s.is_spilled());
		assert_eq!(s.as_str(), "123456789012345678901234");
	}

	// ============================
	// SmolStrExt Tests
	// ============================

	#[test]
	fn test_to_ascii_lowercase_smol() {
		let s = "HELLO WORLD";
		let lower = s.to_ascii_lowercase_smol();
		assert_eq!(lower.as_str(), "hello world");
		assert!(!lower.is_spilled());

		let long = "HELLO WORLD HELLO WORLD!!";
		let lower_long = long.to_ascii_lowercase_smol();
		assert_eq!(lower_long.as_str(), "hello world hello world!!");
		assert!(lower_long.is_spilled());
	}

	#[test]
	fn test_to_ascii_uppercase_smol() {
		let s = "hello world";
		let upper = s.to_ascii_uppercase_smol();
		assert_eq!(upper.as_str(), "HELLO WORLD");

		let already_upper = "HELLO";
		let upper2 = already_upper.to_ascii_uppercase_smol();
		assert_eq!(upper2.as_str(), "HELLO");
	}

	#[test]
	fn test_into_ascii_lowercase() {
		let s = SmolStr::new("HELLO");
		let lower = s.into_ascii_lowercase();
		assert_eq!(lower.as_str(), "hello");
	}

	#[test]
	fn test_into_ascii_uppercase() {
		let s = SmolStr::new("hello");
		let upper = s.into_ascii_uppercase();
		assert_eq!(upper.as_str(), "HELLO");
	}

	// ============================
	// Macro Tests
	// ============================

	#[test]
	fn test_format_smol_basic() {
		let s = format_smol!("hello {}", "world");
		assert_eq!(s.as_str(), "hello world");
	}

	#[test]
	fn test_format_smol_numbers() {
		let s = format_smol!("count: {}, pi: {:.2}", 42, std::f64::consts::PI);
		assert_eq!(s.as_str(), "count: 42, pi: 3.14");
	}

	#[test]
	fn test_format_smol_no_args() {
		let s = format_smol!("static text");
		assert_eq!(s.as_str(), "static text");
	}

	#[test]
	fn test_format_smolmut_basic() {
		let mut s = format_smolmut!("hello {}", "world");
		assert_eq!(s.as_str(), "hello world");
		s.push('!');
		assert_eq!(s.as_str(), "hello world!");
	}

	#[test]
	fn test_format_smol_heap_allocation() {
		let s = format_smol!("{}", "123456789012345678901234");
		assert!(s.is_spilled());
		assert_eq!(s.as_str(), "123456789012345678901234");
	}

	// ============================
	// Static String Tests
	// ============================

	#[test]
	fn test_new_static() {
		const STATIC: &str = "hello world hello world";
		let s = SmolStr::new_static(STATIC);

		// Static strings always use heap representation (never inline)
		assert!(s.is_spilled());
		assert_eq!(s.as_str(), STATIC);

		// Even short strings
		const SHORT: &str = "hi";
		let s2 = SmolStr::new_static(SHORT);
		assert!(s2.is_spilled()); // Still heap due to static
	}

	// ============================
	// Serde Tests
	// ============================

	#[test]
	fn test_smolstr_serde_roundtrip() {
		let s = SmolStr::new("hello world");
		let json = json::to_string(&s).unwrap();
		assert_eq!(json, r#""hello world""#);

		let deserialized: SmolStr = json::from_str(&json).unwrap();
		assert_eq!(deserialized.as_str(), "hello world");
	}

	#[test]
	fn test_smolstr_serde_empty() {
		let s = SmolStr::new("");
		let json = json::to_string(&s).unwrap();
		assert_eq!(json, r#""""#);

		let deserialized: SmolStr = json::from_str(&json).unwrap();
		assert!(deserialized.is_empty());
	}

	#[test]
	fn test_smolstr_serde_multibyte() {
		let s = SmolStr::new("世界 🌍");
		let json = json::to_string(&s).unwrap();

		let deserialized: SmolStr = json::from_str(&json).unwrap();
		assert_eq!(deserialized.as_str(), "世界 🌍");
	}

	// ============================
	// Edge Cases
	// ============================

	#[test]
	fn test_clone_inline() {
		let s1 = SmolStr::new("hello");
		let s2 = s1.clone();
		assert_eq!(s1, s2);
		assert!(!s1.is_spilled());
		assert!(!s2.is_spilled());
	}

	#[test]
	fn test_clone_heap() {
		let s1 = SmolStr::new("hello world hello world!!");
		let s2 = s1.clone();
		assert_eq!(s1, s2);

		// Should share same backing
		assert!(!s1.is_unique());
		assert!(!s2.is_unique());
	}

	#[test]
	fn test_default_smolstr() {
		let s = SmolStr::default();
		assert!(s.is_empty());
		assert!(!s.is_spilled());
	}

	#[test]
	fn test_default_smolstrmut() {
		let m = SmolStrMut::default();
		assert!(m.is_empty());
		assert!(!m.is_spilled());
	}

	#[test]
	fn test_with_capacity_inline() {
		let m = SmolStrMut::with_capacity(10);
		assert!(!m.is_spilled());
		assert!(m.is_empty());
	}

	#[test]
	fn test_with_capacity_heap() {
		let m = SmolStrMut::with_capacity(100);
		assert!(m.is_spilled());
		assert!(m.is_empty());
	}

	#[test]
	fn test_freeze_inline() {
		let m = SmolStrMut::new("hello");
		let s = m.freeze();
		assert_eq!(s.as_str(), "hello");
		assert!(!s.is_spilled());
	}

	#[test]
	fn test_freeze_heap() {
		let m = SmolStrMut::new("hello world hello world!!");
		let s = m.freeze();
		assert_eq!(s.as_str(), "hello world hello world!!");
		assert!(s.is_spilled());
	}

	#[test]
	fn test_extend_empty_iterator() {
		let mut m = SmolStrMut::new("hello");
		let empty: Vec<&str> = vec![];
		m.extend(empty);
		assert_eq!(m.as_str(), "hello");
	}

	#[test]
	fn test_equality_different_repr() {
		// Same content, different representations
		let inline = SmolStr::new("hello");
		let heap = SmolStr::new("hello world hello world!!").slice(0..5);

		assert_eq!(inline, heap);
		assert!(!inline.is_spilled());
		assert!(heap.is_spilled());
	}

	#[test]
	fn test_ordering() {
		let a = SmolStr::new("apple");
		let b = SmolStr::new("banana");
		let c = SmolStr::new("cherry");

		assert!(a < b);
		assert!(b < c);
		assert!(a < c);
		assert!((b >= a));
	}

	#[test]
	fn test_hash_consistency() {
		use std::{
			collections::hash_map::DefaultHasher,
			hash::{Hash, Hasher},
		};

		let s1 = SmolStr::new("hello world");
		let s2 = SmolStr::new("hello world hello world").slice(0..11);

		let mut h1 = DefaultHasher::new();
		s1.hash(&mut h1);

		let mut h2 = DefaultHasher::new();
		s2.hash(&mut h2);

		assert_eq!(h1.finish(), h2.finish());
	}

	#[test]
	fn test_as_bytes() {
		let s = SmolStr::new("hello 🌍");
		let bytes = s.as_bytes();
		assert_eq!(bytes, "hello 🌍".as_bytes());
	}

	#[test]
	fn test_insert_at_end() {
		let mut m = SmolStrMut::new("hello");
		m.insert(5, " world");
		assert_eq!(m.as_str(), "hello world");
	}

	#[test]
	fn test_insert_at_start() {
		let mut m = SmolStrMut::new("world");
		m.insert(0, "hello ");
		assert_eq!(m.as_str(), "hello world");
	}

	#[test]
	fn test_truncate_noop() {
		let mut s = SmolStr::new("hello");
		s.truncate(100); // Greater than length
		assert_eq!(s.as_str(), "hello");

		s.truncate(5); // Exact length
		assert_eq!(s.as_str(), "hello");
	}

	#[test]
	fn test_truncate_mut_noop() {
		let mut m = SmolStrMut::new("hello");
		m.truncate(100);
		assert_eq!(m.as_str(), "hello");
	}

	#[test]
	fn test_deref_coercion() {
		let s = SmolStr::new("hello");
		let len = s.len(); // Should work via Deref
		assert_eq!(len, 5);

		// Can pass to function expecting &str
		fn takes_str(s: &str) -> usize {
			s.len()
		}
		assert_eq!(takes_str(&s), 5);
	}

	#[test]
	fn test_borrow_trait() {
		use std::borrow::Borrow;

		let s = SmolStr::new("hello");
		let borrowed: &str = s.borrow();
		assert_eq!(borrowed, "hello");
	}

	#[test]
	fn test_as_ref_os_str() {
		let s = SmolStr::new("hello");
		let os_str: &std::ffi::OsStr = s.as_ref();
		assert_eq!(os_str, "hello");
	}

	#[test]
	fn test_as_ref_path() {
		let s = SmolStr::new("/tmp/file.txt");
		let path: &std::path::Path = s.as_ref();
		assert_eq!(path.to_str().unwrap(), "/tmp/file.txt");
	}

	#[test]
	fn test_partial_eq_str() {
		let s = SmolStr::new("hello");
		assert_eq!(s, "hello");
		assert_eq!("hello", s);
		assert_ne!(s, "world");
	}

	#[test]
	fn test_partial_eq_string() {
		let s = SmolStr::new("hello");
		let string = String::from("hello");
		assert_eq!(s, string);
		assert_eq!(string, s);
	}

	#[test]
	fn test_smolstrmut_deref_mut() {
		let mut m = SmolStrMut::new("hello");
		let str_mut: &mut str = &mut m;
		str_mut.make_ascii_uppercase();
		assert_eq!(m.as_str(), "HELLO");
	}

	#[test]
	fn test_from_str_parse() {
		let s: SmolStr = "hello world".parse().unwrap();
		assert_eq!(s.as_str(), "hello world");
	}

	#[test]
	fn test_into_smolstr_trait() {
		use super::IntoSmolStr;

		let s: SmolStr = "hello".into_smolstr();
		assert_eq!(s.as_str(), "hello");

		let num: i32 = 42;
		let s2 = (&num).to_smolstr();
		assert_eq!(s2.as_str(), "42");
	}

	#[test]
	fn test_write_trait() {
		use std::fmt::Write;

		let mut m = SmolStrMut::new("hello");
		write!(&mut m, " {}", 42).unwrap();
		assert_eq!(m.as_str(), "hello 42");
	}

	#[test]
	fn test_display_format() {
		let s = SmolStr::new("hello");
		let formatted = format!("{s}");
		assert_eq!(formatted, "hello");
	}

	#[test]
	fn test_debug_format() {
		let s = SmolStr::new("hello");
		let formatted = format!("{s:?}");
		assert_eq!(formatted, "\"hello\"");
	}

	#[test]
	fn test_slice_heap_zero_copy() {
		let s = SmolStr::new("hello world hello world!!");
		let sliced = s.slice(6..11);
		assert_eq!(sliced.as_str(), "world");
		assert!(sliced.is_spilled());
	}

	#[test]
	fn test_slice_inline_creates_inline() {
		let s = SmolStr::new("hello world");
		let sliced = s.slice(0..5);
		assert_eq!(sliced.as_str(), "hello");
		assert!(!sliced.is_spilled());
	}

	#[test]
	fn test_reserve_heap_noop() {
		let mut m = SmolStrMut::new("hello world hello world!!");
		assert!(m.is_spilled());

		let initial_len = m.len();
		m.reserve(10);

		assert_eq!(m.len(), initial_len);
		assert_eq!(m.as_str(), "hello world hello world!!");
	}

	#[test]
	fn test_push_empty_string() {
		let mut m = SmolStrMut::new("hello");
		m.push_str("");
		assert_eq!(m.as_str(), "hello");
	}

	#[test]
	fn test_split_empty_separator() {
		let s = SmolStr::new("hello");
		let parts = s.split("").filter(|s| !s.is_empty()).count();
		assert_eq!(parts, 5);
	}

	#[test]
	fn test_from_bytes_mut() {
		let bytes_mut = bytes::BytesMut::from("hello world hello world");
		let result = SmolStrMut::from_utf8_owned(bytes_mut);
		assert!(result.is_ok());
		let m = result.unwrap();
		assert_eq!(m.as_str(), "hello world hello world");
		assert!(m.is_spilled());
	}

	#[test]
	fn test_into_bytes_mut() {
		let m = SmolStrMut::new("hello world hello world");
		let bytes_mut: bytes::BytesMut = m.into();
		assert_eq!(&bytes_mut[..], b"hello world hello world");
	}

	#[test]
	fn test_smolstr_to_str() {
		let s = SmolStr::new("hello");
		let str_ref: bytes_utils::Str = s.into();
		assert_eq!(&*str_ref, "hello");
	}

	#[test]
	fn test_smolstrmut_to_strmut() {
		let m = SmolStrMut::new("hello world hello world");
		let str_mut: bytes_utils::StrMut = m.into();
		assert_eq!(&*str_mut, "hello world hello world");
	}
}
