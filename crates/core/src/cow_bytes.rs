//! Clone-on-write byte slice with `Bytes` as the owned variant.
//!
//! Provides a `Cow`-like type where the borrowed variant is `&[u8]` and the
//! owned variant is `Bytes`, enabling cheap clones and zero-copy slicing when
//! owned.

use core::{
	borrow::Borrow,
	cmp::Ordering,
	fmt,
	hash::{Hash, Hasher},
	ops::{Bound, Deref, RangeBounds},
};

use bytes::{Bytes, BytesMut};

/// Clone-on-write byte slice with `Bytes` as the owned variant.
///
/// Similar to `Cow<'a, [u8]>`, but uses `Bytes` for the owned variant instead
/// of `Vec<u8>`. This provides:
///
/// - Cheap cloning when owned (via reference counting)
/// - Zero-copy slicing operations
/// - Efficient sharing of byte buffers
///
/// # Examples
///
/// ```
/// use omp_core::CowBytes;
///
/// // Borrowed variant
/// let data = b"hello world";
/// let cow = CowBytes::borrowed(data);
/// assert_eq!(cow.len(), 11);
///
/// // Owned variant from static
/// let cow = CowBytes::from_static(b"hello");
/// assert_eq!(&*cow, b"hello");
///
/// // Owned variant from Vec
/// let cow = CowBytes::owned(vec![1, 2, 3].into());
/// assert_eq!(&*cow, &[1, 2, 3]);
/// ```
#[derive(Clone)]
pub enum CowBytes<'a> {
	/// Borrowed byte slice.
	Borrowed(&'a [u8]),
	/// Owned `Bytes` buffer.
	Owned(Bytes),
	/// Owned `BytesMut` buffer.
	OwnedMut(BytesMut),
}

impl<'a> CowBytes<'a> {
	/// Creates a borrowed variant from a byte slice.
	///
	/// # Examples
	///
	/// ```
	/// use omp_core::CowBytes;
	///
	/// let data = b"hello";
	/// let cow = CowBytes::borrowed(data);
	/// assert!(matches!(cow, CowBytes::Borrowed(_)));
	/// ```
	#[inline]
	pub const fn borrowed(bytes: &'a [u8]) -> Self {
		Self::Borrowed(bytes)
	}

	/// Creates an owned variant from `Bytes`.
	///
	/// # Examples
	///
	/// ```
	/// use bytes::Bytes;
	/// use omp_core::CowBytes;
	///
	/// let bytes = Bytes::from(vec![1, 2, 3]);
	/// let cow = CowBytes::owned(bytes);
	/// assert!(matches!(cow, CowBytes::Owned(_)));
	/// ```
	#[inline]
	pub const fn owned(bytes: Bytes) -> Self {
		Self::Owned(bytes)
	}

	/// Creates an owned variant from `BytesMut`.
	///
	/// # Examples
	///
	/// ```
	/// use bytes::{Bytes, BytesMut};
	/// use omp_core::CowBytes;
	///
	/// let bytes = BytesMut::from(&[1, 2, 3][..]);
	/// let cow = CowBytes::owned_mut(bytes);
	/// assert!(matches!(cow, CowBytes::OwnedMut(_)));
	/// ```
	#[inline]
	pub const fn owned_mut(bytes: BytesMut) -> Self {
		Self::OwnedMut(bytes)
	}

	/// Creates an empty owned variant.
	///
	/// # Examples
	///
	/// ```
	/// use omp_core::CowBytes;
	///
	/// let cow = CowBytes::new();
	/// assert!(cow.is_empty());
	/// assert!(matches!(cow, CowBytes::Owned(_)));
	/// ```
	#[inline]
	pub const fn new() -> Self {
		Self::Owned(Bytes::new())
	}

	/// Creates an owned variant from a static slice.
	///
	/// This does not allocate; the returned `Bytes` will point directly to the
	/// static slice.
	///
	/// # Examples
	///
	/// ```
	/// use omp_core::CowBytes;
	///
	/// let cow = CowBytes::from_static(b"hello");
	/// assert_eq!(&*cow, b"hello");
	/// ```
	#[inline]
	pub const fn from_static(bytes: &'static [u8]) -> Self {
		Self::Owned(Bytes::from_static(bytes))
	}

	/// Creates an owned variant by copying a slice.
	///
	/// # Examples
	///
	/// ```
	/// use omp_core::CowBytes;
	///
	/// let cow = CowBytes::copy_from_slice(b"hello");
	/// assert_eq!(&*cow, b"hello");
	/// ```
	#[inline]
	pub fn copy_from_slice(data: &[u8]) -> Self {
		Self::Owned(Bytes::copy_from_slice(data))
	}

	/// Returns the length in bytes.
	///
	/// # Examples
	///
	/// ```
	/// use omp_core::CowBytes;
	///
	/// let cow = CowBytes::borrowed(b"hello");
	/// assert_eq!(cow.len(), 5);
	/// ```
	#[inline]
	pub fn len(&self) -> usize {
		match self {
			Self::Borrowed(bytes) => bytes.len(),
			Self::Owned(bytes) => bytes.len(),
			Self::OwnedMut(bytes) => bytes.len(),
		}
	}

	/// Returns `true` if the length is 0.
	///
	/// # Examples
	///
	/// ```
	/// use omp_core::CowBytes;
	///
	/// let cow = CowBytes::new();
	/// assert!(cow.is_empty());
	/// ```
	#[inline]
	pub fn is_empty(&self) -> bool {
		self.len() == 0
	}

	/// Returns `true` if this is the borrowed variant.
	///
	/// # Examples
	///
	/// ```
	/// use omp_core::CowBytes;
	///
	/// let cow = CowBytes::borrowed(b"hello");
	/// assert!(cow.is_borrowed());
	///
	/// let cow = CowBytes::new();
	/// assert!(!cow.is_borrowed());
	/// ```
	#[inline]
	pub const fn is_borrowed(&self) -> bool {
		matches!(self, Self::Borrowed(_))
	}

	/// Returns `true` if this is the `Owned` variant.
	///
	/// # Examples
	///
	/// ```
	/// use omp_core::CowBytes;
	///
	/// let cow = CowBytes::new();
	/// assert!(cow.is_owned());
	///
	/// let cow = CowBytes::borrowed(b"hello");
	/// assert!(!cow.is_owned());
	/// ```
	#[inline]
	pub const fn is_owned(&self) -> bool {
		matches!(self, Self::Owned(_))
	}

	/// Returns `true` if this is the `OwnedMut` variant.
	///
	/// # Examples
	///
	/// ```
	/// use bytes::BytesMut;
	/// use omp_core::CowBytes;
	///
	/// let cow = CowBytes::owned_mut(BytesMut::from(&[1, 2, 3][..]));
	/// assert!(cow.is_owned_mut());
	///
	/// let cow = CowBytes::new();
	/// assert!(!cow.is_owned_mut());
	/// ```
	#[inline]
	pub const fn is_owned_mut(&self) -> bool {
		matches!(self, Self::OwnedMut(_))
	}

	/// Clones this into an owned variant with `'static` lifetime.
	///
	/// All borrowed data will be cloned into an owned `Bytes` buffer.
	///
	/// # Examples
	///
	/// ```
	/// use omp_core::CowBytes;
	///
	/// let cow = CowBytes::borrowed(b"hello");
	/// let owned = cow.to_owned();
	/// assert!(matches!(owned, CowBytes::Owned(_)));
	/// ```
	#[inline]
	pub fn to_owned(&self) -> CowBytes<'static> {
		match self {
			Self::Borrowed(bytes) => CowBytes::Owned(Bytes::copy_from_slice(bytes)),
			Self::Owned(bytes) => CowBytes::Owned(bytes.clone()),
			Self::OwnedMut(bytes) => CowBytes::OwnedMut(bytes.clone()),
		}
	}

	/// Converts to an owned variant with `'static` lifetime, consuming self.
	///
	/// This method consumes the `CowBytes` instance and converts borrowed data
	/// into owned data, resulting in `CowBytes<'static>`. When already owned,
	/// this is a zero-cost operation.
	///
	/// # Examples
	///
	/// ```
	/// use omp_core::CowBytes;
	///
	/// let cow = CowBytes::borrowed(b"hello");
	/// let owned = cow.into_owned();
	/// assert!(matches!(owned, CowBytes::Owned(_)));
	/// ```
	#[inline]
	pub fn into_owned(self) -> CowBytes<'static> {
		match self {
			Self::Borrowed(bytes) => CowBytes::Owned(Bytes::copy_from_slice(bytes)),
			Self::Owned(bytes) => CowBytes::Owned(bytes),
			Self::OwnedMut(bytes) => CowBytes::OwnedMut(bytes),
		}
	}

	/// Creates a borrowed view of this `CowBytes`.
	///
	/// This is useful for passing by reference without allocating. If this is
	/// already borrowed, returns a copy of the borrow. If owned, returns a
	/// borrowed reference to the owned data.
	///
	/// # Examples
	///
	/// ```
	/// use omp_core::CowBytes;
	///
	/// let owned = CowBytes::owned(vec![1, 2, 3].into());
	/// let borrowed = owned.borrow();
	/// assert!(borrowed.is_borrowed());
	/// assert_eq!(&*borrowed, &[1, 2, 3]);
	/// ```
	#[inline]
	pub fn borrow(&self) -> CowBytes<'_> {
		CowBytes::Borrowed(self.as_slice())
	}

	/// Converts to a `Bytes` instance, cloning the data if borrowed.
	///
	/// # Examples
	///
	/// ```
	/// use omp_core::CowBytes;
	///
	/// let cow = CowBytes::borrowed(b"hello");
	/// let bytes = cow.into_bytes();
	/// assert_eq!(bytes, b"hello"[..]);
	/// ```
	#[inline]
	pub fn into_bytes(self) -> Bytes {
		match self {
			Self::Borrowed(bytes) => Bytes::copy_from_slice(bytes),
			Self::Owned(bytes) => bytes,
			Self::OwnedMut(bytes) => bytes.freeze(),
		}
	}

	/// Converts to a `BytesMut` instance, cloning the data if borrowed.
	///
	/// # Examples
	///
	/// ```
	/// use omp_core::CowBytes;
	///
	/// let cow = CowBytes::borrowed(b"hello");
	/// let bytes = cow.into_bytes_mut();
	/// assert_eq!(&bytes[..], b"hello");
	/// ```
	#[inline]
	pub fn into_bytes_mut(self) -> BytesMut {
		match self {
			Self::Borrowed(bytes) => BytesMut::from(bytes),
			Self::Owned(bytes) => BytesMut::from(bytes),
			Self::OwnedMut(bytes) => bytes,
		}
	}

	/// Returns a slice of self for the provided range.
	///
	/// If owned, this will increment the reference count for the underlying
	/// memory and return a new owned variant set to the slice. If borrowed,
	/// returns a new borrowed variant.
	///
	/// For `OwnedMut` variant, this freezes to `Bytes` and slices, returning
	/// an `Owned` variant.
	///
	/// # Examples
	///
	/// ```
	/// use omp_core::CowBytes;
	///
	/// let cow = CowBytes::from_static(b"hello world");
	/// let slice = cow.slice(0..5);
	/// assert_eq!(&*slice, b"hello");
	/// ```
	///
	/// # Panics
	///
	/// Panics if the range is out of bounds.
	#[inline]
	pub fn slice(&self, range: impl RangeBounds<usize>) -> Self {
		match self {
			Self::Borrowed(bytes) => {
				let start = match range.start_bound() {
					Bound::Included(&n) => n,
					Bound::Excluded(&n) => n + 1,
					Bound::Unbounded => 0,
				};
				let end = match range.end_bound() {
					Bound::Included(&n) => n + 1,
					Bound::Excluded(&n) => n,
					Bound::Unbounded => bytes.len(),
				};
				Self::Borrowed(&bytes[start..end])
			},
			Self::Owned(bytes) => Self::Owned(bytes.slice(range)),
			Self::OwnedMut(bytes) => {
				// BytesMut doesn't support cheap slicing like Bytes does,
				// so we freeze to Bytes first
				Self::Owned(bytes.clone().freeze().slice(range))
			},
		}
	}

	/// Clears the contents, converting to an empty owned variant if borrowed.
	///
	/// # Examples
	///
	/// ```
	/// use omp_core::CowBytes;
	///
	/// let mut cow = CowBytes::from_static(b"hello");
	/// cow.clear();
	/// assert!(cow.is_empty());
	/// ```
	#[inline]
	pub fn clear(&mut self) {
		match self {
			Self::Borrowed(_) => *self = Self::Owned(Bytes::new()),
			Self::Owned(bytes) => {
				*bytes = Bytes::new();
			},
			Self::OwnedMut(bytes) => bytes.clear(),
		}
	}

	/// Shortens the buffer, keeping the first `len` bytes.
	///
	/// If `len` is greater than the current length, this has no effect.
	///
	/// # Examples
	///
	/// ```
	/// use omp_core::CowBytes;
	///
	/// let mut cow = CowBytes::from_static(b"hello world");
	/// cow.truncate(5);
	/// assert_eq!(&*cow, b"hello");
	/// ```
	#[inline]
	pub fn truncate(&mut self, len: usize) {
		if len >= self.len() {
			return;
		}
		match self {
			Self::Borrowed(bytes) => *bytes = &bytes[..len],
			Self::Owned(bytes) => bytes.truncate(len),
			Self::OwnedMut(bytes) => bytes.truncate(len),
		}
	}

	/// Splits the buffer into two at the given index.
	///
	/// Afterwards `self` contains elements `[0, at)`, and the returned variant
	/// contains elements `[at, len)`.
	///
	/// # Examples
	///
	/// ```
	/// use omp_core::CowBytes;
	///
	/// let mut cow = CowBytes::from_static(b"hello world");
	/// let rest = cow.split_off(6);
	/// assert_eq!(&*cow, b"hello ");
	/// assert_eq!(&*rest, b"world");
	/// ```
	///
	/// # Panics
	///
	/// Panics if `at > len`.
	#[inline]
	pub fn split_off(&mut self, at: usize) -> Self {
		match self {
			Self::Borrowed(bytes) => {
				let (left, right) = bytes.split_at(at);
				*bytes = left;
				Self::Borrowed(right)
			},
			Self::Owned(bytes) => Self::Owned(bytes.split_off(at)),
			Self::OwnedMut(bytes) => Self::OwnedMut(bytes.split_off(at)),
		}
	}

	/// Splits the buffer into two at the given index.
	///
	/// Afterwards `self` contains elements `[at, len)`, and the returned variant
	/// contains elements `[0, at)`.
	///
	/// # Examples
	///
	/// ```
	/// use omp_core::CowBytes;
	///
	/// let mut cow = CowBytes::from_static(b"hello world");
	/// let prefix = cow.split_to(6);
	/// assert_eq!(&*prefix, b"hello ");
	/// assert_eq!(&*cow, b"world");
	/// ```
	///
	/// # Panics
	///
	/// Panics if `at > len`.
	#[inline]
	pub fn split_to(&mut self, at: usize) -> Self {
		match self {
			Self::Borrowed(bytes) => {
				let (left, right) = bytes.split_at(at);
				*bytes = right;
				Self::Borrowed(left)
			},
			Self::Owned(bytes) => Self::Owned(bytes.split_to(at)),
			Self::OwnedMut(bytes) => Self::OwnedMut(bytes.split_to(at)),
		}
	}

	/// Returns `true` if this is an owned variant with a unique reference to
	/// the data.
	///
	/// Always returns `false` for borrowed variants.
	///
	/// # Examples
	///
	/// ```
	/// use omp_core::CowBytes;
	///
	/// let cow = CowBytes::owned(vec![1, 2, 3].into());
	/// assert!(cow.is_unique());
	///
	/// let cow = CowBytes::borrowed(b"hello");
	/// assert!(!cow.is_unique());
	/// ```
	#[inline]
	pub fn is_unique(&self) -> bool {
		match self {
			Self::Borrowed(_) => false,
			Self::Owned(bytes) => bytes.is_unique(),
			Self::OwnedMut(_) => true,
		}
	}

	/// Extracts a slice containing the entire buffer.
	///
	/// # Examples
	///
	/// ```
	/// use omp_core::CowBytes;
	///
	/// let cow = CowBytes::borrowed(b"hello");
	/// assert_eq!(cow.as_slice(), b"hello");
	/// ```
	#[inline]
	pub fn as_slice(&self) -> &[u8] {
		match self {
			Self::Borrowed(bytes) => bytes,
			Self::Owned(bytes) => bytes,
			Self::OwnedMut(bytes) => bytes,
		}
	}
}

impl Default for CowBytes<'_> {
	#[inline]
	fn default() -> Self {
		Self::new()
	}
}

impl Deref for CowBytes<'_> {
	type Target = [u8];

	#[inline]
	fn deref(&self) -> &[u8] {
		self.as_slice()
	}
}

impl AsRef<[u8]> for CowBytes<'_> {
	#[inline]
	fn as_ref(&self) -> &[u8] {
		self.as_slice()
	}
}

impl Borrow<[u8]> for CowBytes<'_> {
	#[inline]
	fn borrow(&self) -> &[u8] {
		self.as_slice()
	}
}

impl<'a> From<&'a [u8]> for CowBytes<'a> {
	#[inline]
	fn from(slice: &'a [u8]) -> Self {
		Self::Borrowed(slice)
	}
}

impl<'a, const N: usize> From<&'a [u8; N]> for CowBytes<'a> {
	#[inline]
	fn from(slice: &'a [u8; N]) -> Self {
		Self::Borrowed(slice)
	}
}

impl<const N: usize> From<[u8; N]> for CowBytes<'_> {
	#[inline]
	fn from(slice: [u8; N]) -> Self {
		Self::Owned(Bytes::from_owner(slice))
	}
}

impl<'a> From<&'a str> for CowBytes<'a> {
	#[inline]
	fn from(s: &'a str) -> Self {
		Self::Borrowed(s.as_bytes())
	}
}

impl From<Bytes> for CowBytes<'_> {
	#[inline]
	fn from(bytes: Bytes) -> Self {
		Self::Owned(bytes)
	}
}

impl From<BytesMut> for CowBytes<'_> {
	#[inline]
	fn from(bytes: BytesMut) -> Self {
		Self::OwnedMut(bytes)
	}
}

impl From<Vec<u8>> for CowBytes<'_> {
	#[inline]
	fn from(vec: Vec<u8>) -> Self {
		Self::Owned(Bytes::from(vec))
	}
}

impl From<Box<[u8]>> for CowBytes<'_> {
	#[inline]
	fn from(vec: Box<[u8]>) -> Self {
		Self::Owned(Bytes::from_owner(vec))
	}
}

impl From<String> for CowBytes<'_> {
	#[inline]
	fn from(s: String) -> Self {
		Self::Owned(Bytes::from(s))
	}
}

impl<'a> From<CowBytes<'a>> for Bytes {
	#[inline]
	fn from(cow: CowBytes<'a>) -> Self {
		cow.into_bytes()
	}
}

impl<'a> From<CowBytes<'a>> for BytesMut {
	#[inline]
	fn from(cow: CowBytes<'a>) -> Self {
		match cow {
			CowBytes::Borrowed(bytes) => Self::from(bytes),
			CowBytes::Owned(bytes) => Self::from(bytes),
			CowBytes::OwnedMut(bytes) => bytes,
		}
	}
}

/// Serializes using the serializer's native byte representation.
///
/// Human-readable serializers decide how bytes are represented; in
/// `serde_json`, they are emitted as an array of numbers rather than base64.
impl serde::Serialize for CowBytes<'_> {
	fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
		serializer.serialize_bytes(self.as_ref())
	}
}

struct CowBytesVisitor<'a>(core::marker::PhantomData<&'a [u8]>);

impl<'a, 'de: 'a> serde::de::Visitor<'de> for CowBytesVisitor<'a> {
	type Value = CowBytes<'a>;

	fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter.write_str("bytes, a byte sequence, or a UTF-8 string")
	}

	fn visit_borrowed_bytes<E>(self, value: &'de [u8]) -> Result<Self::Value, E>
	where
		E: serde::de::Error,
	{
		Ok(CowBytes::Borrowed(value))
	}

	fn visit_bytes<E>(self, value: &[u8]) -> Result<Self::Value, E>
	where
		E: serde::de::Error,
	{
		Ok(CowBytes::copy_from_slice(value))
	}

	fn visit_byte_buf<E>(self, value: Vec<u8>) -> Result<Self::Value, E>
	where
		E: serde::de::Error,
	{
		Ok(CowBytes::Owned(Bytes::from(value)))
	}

	fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
	where
		A: serde::de::SeqAccess<'de>,
	{
		let mut bytes = Vec::with_capacity(sequence.size_hint().unwrap_or(0));
		while let Some(byte) = sequence.next_element()? {
			bytes.push(byte);
		}
		Ok(CowBytes::Owned(Bytes::from(bytes)))
	}

	fn visit_borrowed_str<E>(self, value: &'de str) -> Result<Self::Value, E>
	where
		E: serde::de::Error,
	{
		Ok(CowBytes::Borrowed(value.as_bytes()))
	}

	fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
	where
		E: serde::de::Error,
	{
		Ok(CowBytes::copy_from_slice(value.as_bytes()))
	}

	fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
	where
		E: serde::de::Error,
	{
		Ok(CowBytes::Owned(Bytes::from(value.into_bytes())))
	}
}

/// Deserializes borrowed bytes and strings without copying, and owns all
/// transient byte buffers, strings, and byte sequences.
///
/// JSON byte arrays deserialize into the owned variant.
impl<'a, 'de: 'a> serde::Deserialize<'de> for CowBytes<'a> {
	fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
		deserializer.deserialize_bytes(CowBytesVisitor(core::marker::PhantomData))
	}
}

impl fmt::Debug for CowBytes<'_> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		fmt::Debug::fmt(self.as_slice(), f)
	}
}

impl fmt::Display for CowBytes<'_> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "{:?}", self.as_slice())
	}
}

impl PartialEq for CowBytes<'_> {
	#[inline]
	fn eq(&self, other: &Self) -> bool {
		self.as_slice() == other.as_slice()
	}
}

impl Eq for CowBytes<'_> {}

impl PartialOrd for CowBytes<'_> {
	#[inline]
	fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
		Some(self.cmp(other))
	}
}

impl Ord for CowBytes<'_> {
	#[inline]
	fn cmp(&self, other: &Self) -> Ordering {
		self.as_slice().cmp(other.as_slice())
	}
}

impl Hash for CowBytes<'_> {
	#[inline]
	fn hash<H: Hasher>(&self, state: &mut H) {
		self.as_slice().hash(state);
	}
}

impl<'a> PartialEq<&'a [u8]> for CowBytes<'_> {
	#[inline]
	fn eq(&self, other: &&'a [u8]) -> bool {
		self.as_slice() == *other
	}
}

impl<'a> PartialEq<CowBytes<'a>> for &'a [u8] {
	#[inline]
	fn eq(&self, other: &CowBytes<'a>) -> bool {
		*self == other.as_slice()
	}
}

impl PartialEq<[u8]> for CowBytes<'_> {
	#[inline]
	fn eq(&self, other: &[u8]) -> bool {
		self.as_slice() == other
	}
}

impl PartialEq<CowBytes<'_>> for [u8] {
	#[inline]
	fn eq(&self, other: &CowBytes<'_>) -> bool {
		self == other.as_slice()
	}
}

impl<'a> PartialEq<&'a str> for CowBytes<'_> {
	#[inline]
	fn eq(&self, other: &&'a str) -> bool {
		self.as_slice() == other.as_bytes()
	}
}

impl<'a> PartialEq<CowBytes<'a>> for &'a str {
	#[inline]
	fn eq(&self, other: &CowBytes<'a>) -> bool {
		self.as_bytes() == other.as_slice()
	}
}

impl PartialEq<str> for CowBytes<'_> {
	#[inline]
	fn eq(&self, other: &str) -> bool {
		self.as_slice() == other.as_bytes()
	}
}

impl PartialEq<CowBytes<'_>> for str {
	#[inline]
	fn eq(&self, other: &CowBytes<'_>) -> bool {
		self.as_bytes() == other.as_slice()
	}
}

impl PartialEq<Bytes> for CowBytes<'_> {
	#[inline]
	fn eq(&self, other: &Bytes) -> bool {
		self.as_slice() == &**other
	}
}

impl PartialEq<CowBytes<'_>> for Bytes {
	#[inline]
	fn eq(&self, other: &CowBytes<'_>) -> bool {
		&**self == other.as_slice()
	}
}

impl PartialEq<String> for CowBytes<'_> {
	#[inline]
	fn eq(&self, other: &String) -> bool {
		self.as_slice() == other.as_bytes()
	}
}

impl PartialEq<CowBytes<'_>> for String {
	#[inline]
	fn eq(&self, other: &CowBytes<'_>) -> bool {
		self.as_bytes() == other.as_slice()
	}
}

impl<const N: usize> PartialEq<&[u8; N]> for CowBytes<'_> {
	#[inline]
	fn eq(&self, other: &&[u8; N]) -> bool {
		self.as_slice() == &other[..]
	}
}

impl<const N: usize> PartialEq<CowBytes<'_>> for &[u8; N] {
	#[inline]
	fn eq(&self, other: &CowBytes<'_>) -> bool {
		&self[..] == other.as_slice()
	}
}

impl<const N: usize> PartialEq<[u8; N]> for CowBytes<'_> {
	#[inline]
	fn eq(&self, other: &[u8; N]) -> bool {
		self.as_slice() == &other[..]
	}
}

impl<const N: usize> PartialEq<CowBytes<'_>> for [u8; N] {
	#[inline]
	fn eq(&self, other: &CowBytes<'_>) -> bool {
		&self[..] == other.as_slice()
	}
}

#[cfg(test)]
mod tests {
	use core::hash::{Hash, Hasher};
	use std::collections::hash_map::DefaultHasher;

	use proptest::prelude::*;

	use super::*;

	// ============================================================================
	// CONSTRUCTION & VARIANT CHECKS
	// ============================================================================

	#[test]
	fn test_borrowed_construction() {
		let data = b"hello";
		let cow = CowBytes::borrowed(data);
		assert!(cow.is_borrowed());
		assert!(!cow.is_owned());
		assert!(!cow.is_owned_mut());
		assert_eq!(&*cow, data);
	}

	#[test]
	fn test_owned_construction() {
		let bytes = Bytes::from(vec![1, 2, 3]);
		let cow = CowBytes::owned(bytes);
		assert!(!cow.is_borrowed());
		assert!(cow.is_owned());
		assert!(!cow.is_owned_mut());
		assert_eq!(&*cow, &[1, 2, 3]);
	}

	#[test]
	fn test_owned_mut_construction() {
		let bytes = BytesMut::from(&[1, 2, 3][..]);
		let cow = CowBytes::owned_mut(bytes);
		assert!(!cow.is_borrowed());
		assert!(!cow.is_owned());
		assert!(cow.is_owned_mut());
		assert_eq!(&*cow, &[1, 2, 3]);
	}

	#[test]
	fn test_new_is_empty_owned() {
		let cow = CowBytes::new();
		assert!(cow.is_empty());
		assert!(cow.is_owned());
		assert_eq!(cow.len(), 0);
	}

	#[test]
	fn test_from_static() {
		let cow = CowBytes::from_static(b"static data");
		assert!(cow.is_owned());
		assert_eq!(&*cow, b"static data");
	}

	#[test]
	fn test_copy_from_slice() {
		let cow = CowBytes::copy_from_slice(b"copied");
		assert!(cow.is_owned());
		assert_eq!(&*cow, b"copied");
	}

	// ============================================================================
	// SPLIT OPERATIONS - BORROWED
	// ============================================================================

	#[test]
	fn test_split_off_borrowed() {
		let data = b"hello world";
		let mut cow = CowBytes::borrowed(data);
		let rest = cow.split_off(6);

		assert!(cow.is_borrowed());
		assert!(rest.is_borrowed());
		assert_eq!(&*cow, b"hello ");
		assert_eq!(&*rest, b"world");
	}

	#[test]
	fn test_split_off_borrowed_at_zero() {
		let data = b"hello";
		let mut cow = CowBytes::borrowed(data);
		let rest = cow.split_off(0);

		assert_eq!(&*cow, b"");
		assert_eq!(&*rest, b"hello");
	}

	#[test]
	fn test_split_off_borrowed_at_end() {
		let data = b"hello";
		let mut cow = CowBytes::borrowed(data);
		let rest = cow.split_off(5);

		assert_eq!(&*cow, b"hello");
		assert_eq!(&*rest, b"");
	}

	#[test]
	fn test_split_to_borrowed() {
		let data = b"hello world";
		let mut cow = CowBytes::borrowed(data);
		let prefix = cow.split_to(6);

		assert!(cow.is_borrowed());
		assert!(prefix.is_borrowed());
		assert_eq!(&*prefix, b"hello ");
		assert_eq!(&*cow, b"world");
	}

	#[test]
	fn test_split_to_borrowed_at_zero() {
		let data = b"hello";
		let mut cow = CowBytes::borrowed(data);
		let prefix = cow.split_to(0);

		assert_eq!(&*prefix, b"");
		assert_eq!(&*cow, b"hello");
	}

	#[test]
	fn test_split_to_borrowed_at_end() {
		let data = b"hello";
		let mut cow = CowBytes::borrowed(data);
		let prefix = cow.split_to(5);

		assert_eq!(&*prefix, b"hello");
		assert_eq!(&*cow, b"");
	}

	#[test]
	#[should_panic(expected = "mid > len")]
	fn test_split_off_borrowed_out_of_bounds() {
		let data = b"hello";
		let mut cow = CowBytes::borrowed(data);
		cow.split_off(10);
	}

	#[test]
	#[should_panic(expected = "mid > len")]
	fn test_split_to_borrowed_out_of_bounds() {
		let data = b"hello";
		let mut cow = CowBytes::borrowed(data);
		cow.split_to(10);
	}

	// ============================================================================
	// SPLIT OPERATIONS - OWNED
	// ============================================================================

	#[test]
	fn test_split_off_owned() {
		let mut cow = CowBytes::owned(Bytes::from(b"hello world".to_vec()));
		let rest = cow.split_off(6);

		assert!(cow.is_owned());
		assert!(rest.is_owned());
		assert_eq!(&*cow, b"hello ");
		assert_eq!(&*rest, b"world");
	}

	#[test]
	fn test_split_to_owned() {
		let mut cow = CowBytes::owned(Bytes::from(b"hello world".to_vec()));
		let prefix = cow.split_to(6);

		assert!(cow.is_owned());
		assert!(prefix.is_owned());
		assert_eq!(&*prefix, b"hello ");
		assert_eq!(&*cow, b"world");
	}

	// ============================================================================
	// SPLIT OPERATIONS - OWNEDMUT
	// ============================================================================

	#[test]
	fn test_split_off_owned_mut() {
		let mut cow = CowBytes::owned_mut(BytesMut::from(&b"hello world"[..]));
		let rest = cow.split_off(6);

		assert!(cow.is_owned_mut());
		assert!(rest.is_owned_mut());
		assert_eq!(&*cow, b"hello ");
		assert_eq!(&*rest, b"world");
	}

	#[test]
	fn test_split_to_owned_mut() {
		let mut cow = CowBytes::owned_mut(BytesMut::from(&b"hello world"[..]));
		let prefix = cow.split_to(6);

		assert!(cow.is_owned_mut());
		assert!(prefix.is_owned_mut());
		assert_eq!(&*prefix, b"hello ");
		assert_eq!(&*cow, b"world");
	}

	// ============================================================================
	// SLICE OPERATIONS
	// ============================================================================

	#[test]
	fn test_slice_borrowed() {
		let cow = CowBytes::borrowed(b"hello world");
		let slice = cow.slice(0..5);

		assert!(slice.is_borrowed());
		assert_eq!(&*slice, b"hello");
	}

	#[test]
	fn test_slice_owned() {
		let cow = CowBytes::owned(Bytes::from(b"hello world".to_vec()));
		let slice = cow.slice(0..5);

		assert!(slice.is_owned());
		assert_eq!(&*slice, b"hello");
	}

	#[test]
	fn test_slice_owned_mut() {
		let cow = CowBytes::owned_mut(BytesMut::from(&b"hello world"[..]));
		let slice = cow.slice(0..5);

		// OwnedMut freezes to Owned when slicing
		assert!(slice.is_owned());
		assert_eq!(&*slice, b"hello");
	}

	#[test]
	fn test_slice_full_range() {
		let cow = CowBytes::borrowed(b"hello");
		let slice = cow.slice(..);
		assert_eq!(&*slice, b"hello");
	}

	#[test]
	fn test_slice_from_range() {
		let cow = CowBytes::borrowed(b"hello");
		let slice = cow.slice(2..);
		assert_eq!(&*slice, b"llo");
	}

	#[test]
	fn test_slice_to_range() {
		let cow = CowBytes::borrowed(b"hello");
		let slice = cow.slice(..3);
		assert_eq!(&*slice, b"hel");
	}

	#[test]
	fn test_slice_inclusive_range() {
		let cow = CowBytes::borrowed(b"hello");
		let slice = cow.slice(1..=3);
		assert_eq!(&*slice, b"ell");
	}

	#[test]
	#[should_panic(expected = "range end index 10 out of range for slice of length 5")]
	fn test_slice_out_of_bounds() {
		let cow = CowBytes::borrowed(b"hello");
		cow.slice(0..10);
	}

	// ============================================================================
	// TRUNCATE & CLEAR
	// ============================================================================

	#[test]
	fn test_truncate_borrowed() {
		let mut cow = CowBytes::borrowed(b"hello world");
		cow.truncate(5);
		assert!(cow.is_borrowed());
		assert_eq!(&*cow, b"hello");
	}

	#[test]
	fn test_truncate_owned() {
		let mut cow = CowBytes::owned(Bytes::from(b"hello world".to_vec()));
		cow.truncate(5);
		assert!(cow.is_owned());
		assert_eq!(&*cow, b"hello");
	}

	#[test]
	fn test_truncate_owned_mut() {
		let mut cow = CowBytes::owned_mut(BytesMut::from(&b"hello world"[..]));
		cow.truncate(5);
		assert!(cow.is_owned_mut());
		assert_eq!(&*cow, b"hello");
	}

	#[test]
	fn test_truncate_no_op_if_len_larger() {
		let mut cow = CowBytes::borrowed(b"hello");
		cow.truncate(10);
		assert_eq!(&*cow, b"hello");
	}

	#[test]
	fn test_clear_borrowed() {
		let mut cow = CowBytes::borrowed(b"hello");
		cow.clear();
		assert!(cow.is_owned());
		assert!(cow.is_empty());
	}

	#[test]
	fn test_clear_owned() {
		let mut cow = CowBytes::owned(Bytes::from(b"hello".to_vec()));
		cow.clear();
		assert!(cow.is_owned());
		assert!(cow.is_empty());
	}

	#[test]
	fn test_clear_owned_mut() {
		let mut cow = CowBytes::owned_mut(BytesMut::from(&b"hello"[..]));
		cow.clear();
		assert!(cow.is_owned_mut());
		assert!(cow.is_empty());
	}

	// ============================================================================
	// CONVERSIONS
	// ============================================================================

	#[test]
	fn test_to_owned_borrowed() {
		let cow = CowBytes::borrowed(b"hello");
		let owned = cow.to_owned();
		assert!(owned.is_owned());
		assert_eq!(&*owned, b"hello");
	}

	#[test]
	fn test_to_owned_owned() {
		let cow = CowBytes::owned(Bytes::from(b"hello".to_vec()));
		let owned = cow.to_owned();
		assert!(owned.is_owned());
		assert_eq!(&*owned, b"hello");
	}

	#[test]
	fn test_into_owned_borrowed() {
		let cow = CowBytes::borrowed(b"hello");
		let owned = cow.into_owned();
		assert!(owned.is_owned());
		assert_eq!(&*owned, b"hello");
	}

	#[test]
	fn test_into_owned_zero_cost_when_owned() {
		let bytes = Bytes::from(vec![1, 2, 3]);
		let ptr = bytes.as_ptr();

		let cow = CowBytes::owned(bytes);
		let owned = cow.into_owned();

		assert_eq!(owned.as_ptr(), ptr);
		assert!(owned.is_owned());
	}

	#[test]
	fn test_borrow() {
		let cow = CowBytes::owned(Bytes::from(b"hello".to_vec()));
		let borrowed = cow.borrow();
		assert!(borrowed.is_borrowed());
		assert_eq!(&*borrowed, b"hello");
	}

	#[test]
	fn test_into_bytes_borrowed() {
		let cow = CowBytes::borrowed(b"hello");
		let bytes = cow.into_bytes();
		assert_eq!(&bytes[..], b"hello");
	}

	#[test]
	fn test_into_bytes_owned() {
		let cow = CowBytes::owned(Bytes::from(b"hello".to_vec()));
		let bytes = cow.into_bytes();
		assert_eq!(&bytes[..], b"hello");
	}

	#[test]
	fn test_into_bytes_mut_borrowed() {
		let cow = CowBytes::borrowed(b"hello");
		let bytes = cow.into_bytes_mut();
		assert_eq!(&bytes[..], b"hello");
	}

	#[test]
	fn test_into_bytes_mut_owned() {
		let cow = CowBytes::owned(Bytes::from(b"hello".to_vec()));
		let bytes = cow.into_bytes_mut();
		assert_eq!(&bytes[..], b"hello");
	}

	#[test]
	fn test_into_bytes_mut_owned_mut() {
		let cow = CowBytes::owned_mut(BytesMut::from(&b"hello"[..]));
		let bytes = cow.into_bytes_mut();
		assert_eq!(&bytes[..], b"hello");
	}

	// ============================================================================
	// FROM IMPLS
	// ============================================================================

	#[test]
	fn test_from_slice() {
		let data: &[u8] = b"hello";
		let cow: CowBytes = data.into();
		assert!(cow.is_borrowed());
		assert_eq!(&*cow, b"hello");
	}

	#[test]
	fn test_from_array_ref() {
		let data: &[u8; 5] = b"hello";
		let cow: CowBytes = data.into();
		assert!(cow.is_borrowed());
		assert_eq!(&*cow, b"hello");
	}

	#[test]
	fn test_from_array() {
		let data: [u8; 5] = *b"hello";
		let cow: CowBytes = data.into();
		assert!(cow.is_owned());
		assert_eq!(&*cow, b"hello");
	}

	#[test]
	fn test_from_str_ref() {
		let s: &str = "hello";
		let cow: CowBytes = s.into();
		assert!(cow.is_borrowed());
		assert_eq!(&*cow, b"hello");
	}

	#[test]
	fn test_from_bytes() {
		let bytes = Bytes::from(b"hello".to_vec());
		let cow: CowBytes = bytes.into();
		assert!(cow.is_owned());
		assert_eq!(&*cow, b"hello");
	}

	#[test]
	fn test_from_bytes_mut() {
		let bytes = BytesMut::from(&b"hello"[..]);
		let cow: CowBytes = bytes.into();
		assert!(cow.is_owned_mut());
		assert_eq!(&*cow, b"hello");
	}

	#[test]
	fn test_from_vec() {
		let vec = vec![1, 2, 3];
		let cow: CowBytes = vec.into();
		assert!(cow.is_owned());
		assert_eq!(&*cow, &[1, 2, 3]);
	}

	#[test]
	fn test_from_box_slice() {
		let boxed: Box<[u8]> = vec![1, 2, 3].into_boxed_slice();
		let cow: CowBytes = boxed.into();
		assert!(cow.is_owned());
		assert_eq!(&*cow, &[1, 2, 3]);
	}

	#[test]
	fn test_from_string() {
		let s = String::from("hello");
		let cow: CowBytes = s.into();
		assert!(cow.is_owned());
		assert_eq!(&*cow, b"hello");
	}

	#[test]
	fn test_into_bytes_from_cow() {
		let cow = CowBytes::borrowed(b"hello");
		let bytes: Bytes = cow.into();
		assert_eq!(&bytes[..], b"hello");
	}

	#[test]
	fn test_into_bytes_mut_from_cow() {
		let cow = CowBytes::borrowed(b"hello");
		let bytes: BytesMut = cow.into();
		assert_eq!(&bytes[..], b"hello");
	}

	// ============================================================================
	// IS_UNIQUE
	// ============================================================================

	#[test]
	fn test_is_unique_borrowed() {
		let cow = CowBytes::borrowed(b"hello");
		assert!(!cow.is_unique());
	}

	#[test]
	fn test_is_unique_owned_unique() {
		let cow = CowBytes::owned(Bytes::from(b"hello".to_vec()));
		assert!(cow.is_unique());
	}

	#[test]
	fn test_is_unique_owned_shared() {
		let bytes = Bytes::from(b"hello".to_vec());
		let cow1 = CowBytes::owned(bytes.clone());
		let _cow2 = CowBytes::owned(bytes);
		assert!(!cow1.is_unique());
	}

	#[test]
	fn test_is_unique_owned_mut() {
		let cow = CowBytes::owned_mut(BytesMut::from(&b"hello"[..]));
		assert!(cow.is_unique());
	}

	// ============================================================================
	// PARTIALEQ IMPLEMENTATIONS
	// ============================================================================

	#[test]
	fn test_eq_cow_cow() {
		let cow1 = CowBytes::borrowed(b"hello");
		let cow2 = CowBytes::owned(Bytes::from(b"hello".to_vec()));
		let cow3 = CowBytes::owned_mut(BytesMut::from(&b"hello"[..]));

		assert_eq!(cow1, cow2);
		assert_eq!(cow1, cow3);
		assert_eq!(cow2, cow3);
	}

	#[test]
	fn test_eq_cow_slice() {
		let cow = CowBytes::borrowed(b"hello");
		assert_eq!(cow, b"hello"[..]);
		assert_eq!(b"hello"[..], cow);
	}

	#[test]
	fn test_eq_cow_slice_ref() {
		let cow = CowBytes::borrowed(b"hello");
		assert_eq!(cow, &b"hello"[..]);
		assert_eq!(&b"hello"[..], cow);
	}

	#[test]
	fn test_eq_cow_str() {
		let cow = CowBytes::borrowed(b"hello");
		assert_eq!(cow, "hello");
		assert_eq!("hello", cow);
	}

	#[test]
	fn test_eq_cow_str_ref() {
		let cow = CowBytes::borrowed(b"hello");
		let s: &str = "hello";
		assert_eq!(cow, s);
		assert_eq!(s, cow);
	}

	#[test]
	fn test_eq_cow_bytes() {
		let cow = CowBytes::borrowed(b"hello");
		let bytes = Bytes::from(b"hello".to_vec());
		assert_eq!(cow, bytes);
		assert_eq!(bytes, cow);
	}

	#[test]
	fn test_eq_cow_string() {
		let cow = CowBytes::borrowed(b"hello");
		let s = String::from("hello");
		assert_eq!(cow, s);
		assert_eq!(s, cow);
	}

	#[test]
	fn test_eq_cow_array() {
		let cow = CowBytes::borrowed(b"hello");
		assert_eq!(cow, b"hello");
		assert_eq!(b"hello", cow);
	}

	#[test]
	fn test_eq_cow_array_ref() {
		let cow = CowBytes::borrowed(b"hello");
		let arr: &[u8; 5] = b"hello";
		assert_eq!(cow, arr);
		assert_eq!(arr, cow);
	}

	// ============================================================================
	// ORD
	// ============================================================================

	#[test]
	fn test_ord_cow() {
		let a = CowBytes::borrowed(b"apple");
		let b = CowBytes::owned(Bytes::from(b"banana".to_vec()));
		let c = CowBytes::owned_mut(BytesMut::from(&b"cherry"[..]));

		assert!(a < b);
		assert!(b < c);
		assert!(a < c);
	}

	#[test]
	fn test_partial_ord_reflexive() {
		let cow = CowBytes::borrowed(b"hello");
		assert_eq!(cow.partial_cmp(&cow), Some(core::cmp::Ordering::Equal));
	}

	// ============================================================================
	// HASH
	// ============================================================================

	#[test]
	fn test_hash_consistency() {
		let cow1 = CowBytes::borrowed(b"hello");
		let cow2 = CowBytes::owned(Bytes::from(b"hello".to_vec()));
		let cow3 = CowBytes::owned_mut(BytesMut::from(&b"hello"[..]));

		let hash1 = hash(&cow1);
		let hash2 = hash(&cow2);
		let hash3 = hash(&cow3);

		assert_eq!(hash1, hash2);
		assert_eq!(hash1, hash3);
	}

	#[test]
	fn test_hash_different_values() {
		let cow1 = CowBytes::borrowed(b"hello");
		let cow2 = CowBytes::borrowed(b"world");

		let hash1 = hash(&cow1);
		let hash2 = hash(&cow2);

		assert_ne!(hash1, hash2);
	}

	fn hash<T: Hash>(val: &T) -> u64 {
		let mut hasher = DefaultHasher::new();
		val.hash(&mut hasher);
		hasher.finish()
	}

	// ============================================================================
	// DISPLAY & DEBUG
	// ============================================================================

	#[test]
	fn test_debug_format() {
		let cow = CowBytes::borrowed(b"hello");
		let debug = format!("{cow:?}");
		assert!(debug.contains("104"));
	}

	#[test]
	fn test_display_format() {
		let cow = CowBytes::borrowed(b"hello");
		let display = format!("{cow}");
		assert!(display.contains("104"));
	}

	// ============================================================================
	// SERDE
	// ============================================================================

	#[test]
	fn test_serde_json_roundtrip() {
		let original = CowBytes::owned(Bytes::from(vec![1, 2, 3, 4, 5]));
		let json = serde_json::to_string(&original).unwrap();
		assert_eq!(json, "[1,2,3,4,5]");

		let deserialized: CowBytes = serde_json::from_str(&json).unwrap();
		assert_eq!(original, deserialized);
		assert_eq!(&*deserialized, &[1, 2, 3, 4, 5]);
	}

	#[test]
	fn test_serde_json_borrowed() {
		let cow = CowBytes::borrowed(b"hello");
		let json = serde_json::to_string(&cow).unwrap();
		let decoded: CowBytes = serde_json::from_str(&json).unwrap();
		assert_eq!(decoded, b"hello"[..]);
	}

	#[test]
	fn test_serde_json_empty() {
		let cow = CowBytes::new();
		let json = serde_json::to_string(&cow).unwrap();
		assert_eq!(json, "[]");

		let decoded: CowBytes = serde_json::from_str(&json).unwrap();
		assert!(decoded.is_empty());
	}

	#[test]
	fn test_serde_borrowed_bytes_zero_copy() {
		use serde::de::value::{BorrowedBytesDeserializer, Error};

		let original = b"borrowed";
		let deserializer = BorrowedBytesDeserializer::<Error>::new(original);
		let decoded = <CowBytes<'_> as serde::Deserialize>::deserialize(deserializer).unwrap();
		let CowBytes::Borrowed(bytes) = decoded else {
			panic!("borrowed bytes should stay borrowed");
		};
		assert_eq!(bytes, original);
		assert!(core::ptr::eq(bytes.as_ptr(), original.as_ptr()));
	}

	// ============================================================================
	// PROPTEST
	// ============================================================================

	proptest! {
		#[test]
		fn proptest_borrowed_roundtrip(data in prop::collection::vec(any::<u8>(), 0..1000)) {
			let cow = CowBytes::borrowed(&data);
			prop_assert_eq!(&*cow, data.as_slice());
		}

		#[test]
		fn proptest_owned_roundtrip(data in prop::collection::vec(any::<u8>(), 0..1000)) {
			let cow = CowBytes::owned(Bytes::from(data.clone()));
			prop_assert_eq!(&*cow, data.as_slice());
		}

		#[test]
		fn proptest_split_off_preserves_data(
			data in prop::collection::vec(any::<u8>(), 1..1000),
			at in 0usize..1000
		) {
			let at = at % data.len();
			let mut cow = CowBytes::borrowed(&data);
			let rest = cow.split_off(at);

			prop_assert_eq!(&*cow, &data[..at]);
			prop_assert_eq!(&*rest, &data[at..]);
		}

		#[test]
		fn proptest_split_to_preserves_data(
			data in prop::collection::vec(any::<u8>(), 1..1000),
			at in 0usize..1000
		) {
			let at = at % data.len();
			let mut cow = CowBytes::borrowed(&data);
			let prefix = cow.split_to(at);

			prop_assert_eq!(&*prefix, &data[..at]);
			prop_assert_eq!(&*cow, &data[at..]);
		}

		#[test]
		fn proptest_truncate_preserves_prefix(
			data in prop::collection::vec(any::<u8>(), 1..1000),
			len in 0usize..1000
		) {
			let len = len % data.len();
			let mut cow = CowBytes::borrowed(&data);
			cow.truncate(len);
			prop_assert_eq!(&*cow, &data[..len]);
		}

		#[test]
		fn proptest_hash_eq_consistency(data in prop::collection::vec(any::<u8>(), 0..100)) {
			let cow1 = CowBytes::borrowed(&data);
			let cow2 = CowBytes::owned(Bytes::from(data.clone()));

			if cow1 == cow2 {
				prop_assert_eq!(hash(&cow1), hash(&cow2));
			}
		}

		#[test]
		fn proptest_ord_consistency(
			data1 in prop::collection::vec(any::<u8>(), 0..100),
			data2 in prop::collection::vec(any::<u8>(), 0..100)
		) {
			let cow1 = CowBytes::borrowed(&data1);
			let cow2 = CowBytes::borrowed(&data2);

			let cmp1 = cow1.cmp(&cow2);
			let cmp2 = data1.cmp(&data2);
			prop_assert_eq!(cmp1, cmp2);
		}
	}
}
