//! Sparse set backed by bitmap storage.
//!
//! `SparseSet<K>` stores sets where keys map to indices, using a bitmap for
//! presence tracking. Provides fast membership tests and iteration over present
//! keys.

use std::{fmt, iter::FusedIterator, marker::PhantomData};

use serde::{Deserialize, Serialize};
use smol_bitmap::SmolBitmap;

use crate::sparse_index::TrySparseIndex;

/// A sparse set of keys convertible to indices.
///
/// [`SparseSet`] provides an efficient storage mechanism for sets
/// where keys can be converted to `usize` indices. It uses a bitmap to track
/// which indices are present, achieving both memory efficiency and fast lookup
/// times.
///
/// The key type `K` must implement [`Into<usize>`] and [`From<usize>`] to
/// convert between keys and indices. This makes it ideal for enum keys,
/// small integers, or other types with a natural index representation.
///
/// # Examples
///
/// ```
/// use omp_core::{sparse_index::TrySparseIndex, sparse_set::SparseSet};
///
/// #[repr(usize)]
/// #[derive(Copy, Clone, Debug, PartialEq)]
/// enum Status {
/// 	Active  = 0,
/// 	Pending = 1,
/// 	Closed  = 2,
/// }
///
/// #[derive(Debug)]
/// struct StatusError(&'static str);
///
/// impl std::fmt::Display for StatusError {
/// 	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
/// 		write!(f, "{}", self.0)
/// 	}
/// }
///
/// impl std::error::Error for StatusError {}
///
/// impl TrySparseIndex for Status {
/// 	type Error = StatusError;
///
/// 	fn index(&self) -> usize {
/// 		*self as usize
/// 	}
///
/// 	fn try_from_index(index: usize) -> Result<Self, Self::Error> {
/// 		match index {
/// 			0 => Ok(Status::Active),
/// 			1 => Ok(Status::Pending),
/// 			2 => Ok(Status::Closed),
/// 			_ => Err(StatusError("Invalid status value")),
/// 		}
/// 	}
/// }
///
/// let mut set = SparseSet::new();
/// set.insert(Status::Active);
/// set.insert(Status::Closed);
///
/// assert!(set.contains(Status::Active));
/// assert!(!set.contains(Status::Pending));
/// ```
#[repr(transparent)]
pub struct SparseSet<K> {
	/// Bitmap tracking which indices are present
	bits:     SmolBitmap,
	/// Phantom data to maintain type information for keys
	_phantom: PhantomData<K>,
}

impl<K> Clone for SparseSet<K> {
	fn clone(&self) -> Self {
		Self { bits: self.bits.clone(), _phantom: PhantomData }
	}
}

impl<K: TrySparseIndex + fmt::Debug> fmt::Debug for SparseSet<K> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		let mut s = f.debug_set();
		for key in self {
			s.entry(&key);
		}
		s.finish()
	}
}

impl<K: TrySparseIndex + Serialize> Serialize for SparseSet<K> {
	fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
	where
		S: serde::Serializer,
	{
		use serde::ser::SerializeSeq;

		if serializer.is_human_readable() {
			let mut seq = serializer.serialize_seq(Some(self.len()))?;
			for key in self {
				seq.serialize_element(&key)?;
			}
			seq.end()
		} else {
			self.bits.serialize(serializer)
		}
	}
}
impl<'de, K: TrySparseIndex + Deserialize<'de>> Deserialize<'de> for SparseSet<K> {
	fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
	where
		D: serde::Deserializer<'de>,
	{
		if deserializer.is_human_readable() {
			use std::fmt;

			use serde::de::{SeqAccess, Visitor};

			struct SparseSetVisitor<K> {
				_phantom: PhantomData<K>,
			}

			impl<'de, K: TrySparseIndex + Deserialize<'de>> Visitor<'de> for SparseSetVisitor<K> {
				type Value = SparseSet<K>;

				fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
					formatter.write_str("a sequence of indices")
				}

				fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
				where
					A: SeqAccess<'de>,
				{
					let mut set = SparseSet::new();
					if let Some(len) = seq.size_hint() {
						set.reserve(len);
					}
					let mut prev_index = None;
					while let Some(key) = seq.next_element::<K>()? {
						let index = key.index();
						if prev_index.is_some_and(|prev| prev >= index) {
							return Err(serde::de::Error::invalid_value(
								serde::de::Unexpected::Unsigned(index as u64),
								&"indices must be in ascending order",
							));
						}
						prev_index = Some(index);
						set.bits.insert(index);
					}
					Ok(set)
				}
			}

			deserializer.deserialize_seq(SparseSetVisitor { _phantom: PhantomData })
		} else {
			let bits = Deserialize::deserialize(deserializer)?;
			let set = Self { bits, _phantom: PhantomData };
			K::validate_sorted(set.bits.iter()).map_err(serde::de::Error::custom)?;
			Ok(set)
		}
	}
}

impl<K> Default for SparseSet<K> {
	fn default() -> Self {
		Self::new()
	}
}

impl<K> PartialEq for SparseSet<K> {
	fn eq(&self, other: &Self) -> bool {
		self.bits == other.bits
	}
}

impl<K> Eq for SparseSet<K> {}

impl<K> SparseSet<K> {
	/// Creates a new empty sparse index set.
	pub const fn new() -> Self {
		Self { bits: SmolBitmap::new(), _phantom: PhantomData }
	}

	/// Creates a new sparse index set with the specified capacity.
	///
	/// # Arguments
	///
	/// * `capacity` - The maximum index that might be stored
	pub fn with_capacity(capacity: usize) -> Self {
		Self { bits: SmolBitmap::with_capacity(capacity), _phantom: PhantomData }
	}

	/// Returns the number of elements in the set.
	#[inline]
	pub fn len(&self) -> usize {
		self.bits.len()
	}

	/// Returns `true` if the set contains no elements.
	#[inline]
	pub fn is_empty(&self) -> bool {
		self.bits.is_empty()
	}

	/// Returns the capacity of the underlying bitmap.
	#[inline]
	pub const fn capacity(&self) -> usize {
		self.bits.capacity()
	}

	/// Clears the set, removing all elements.
	pub fn clear(&mut self) {
		self.bits.clear();
	}

	/// Shrinks the capacity of the set as much as possible.
	pub fn shrink_to_fit(&mut self) {
		self.bits.shrink_to_fit();
	}

	/// Reserves capacity for at least `additional` more elements to be
	/// inserted in the set.
	pub fn reserve(&mut self, additional: usize) {
		self.bits.reserve(additional);
	}

	/// Decomposes the set into its raw bitmap.
	///
	/// # Returns
	///
	/// The underlying `SmolBitmap`
	#[inline]
	pub fn into_parts(self) -> SmolBitmap {
		self.bits
	}

	/// Constructs a sparse set from its raw bitmap.
	///
	/// # Arguments
	///
	/// * `bits` - The bitmap tracking which indices are present
	#[inline]
	pub const fn from_parts(bits: SmolBitmap) -> Self {
		Self { bits, _phantom: PhantomData }
	}

	/// Returns `true` if the set is a subset of another.
	pub fn is_subset(&self, other: &Self) -> bool {
		self.bits.is_subset(&other.bits)
	}

	/// Returns `true` if the set is a superset of another.
	pub fn is_superset(&self, other: &Self) -> bool {
		self.bits.is_superset(&other.bits)
	}

	/// Returns `true` if the set has no elements in common with another.
	pub fn is_disjoint(&self, other: &Self) -> bool {
		self.bits.is_disjoint(&other.bits)
	}

	/// Computes the union with another set.
	pub fn union(&self, other: &Self) -> Self {
		Self { bits: self.bits.union(&other.bits), _phantom: PhantomData }
	}

	/// Computes the intersection with another set.
	pub fn intersection(&self, other: &Self) -> Self {
		Self { bits: self.bits.intersection(&other.bits), _phantom: PhantomData }
	}

	/// Computes the difference with another set.
	pub fn difference(&self, other: &Self) -> Self {
		Self { bits: self.bits.difference(&other.bits), _phantom: PhantomData }
	}

	/// Computes the symmetric difference with another set.
	pub fn symmetric_difference(&self, other: &Self) -> Self {
		Self { bits: self.bits.symmetric_difference(&other.bits), _phantom: PhantomData }
	}
}

impl<K: TrySparseIndex> SparseSet<K> {
	/// Returns `true` if the set contains the specified key.
	///
	/// # Arguments
	///
	/// * `key` - The key to check for
	#[inline]
	pub fn contains(&self, key: K) -> bool {
		self.bits.get(key.index())
	}

	/// Adds a key to the set.
	///
	/// If the set did not have this key present, `true` is returned.
	/// If the set did have this key present, `false` is returned.
	///
	/// # Arguments
	///
	/// * `key` - The key to insert
	///
	/// # Returns
	///
	/// `true` if the key was newly inserted, `false` if it was already present
	pub fn insert(&mut self, key: K) -> bool {
		self.bits.insert(key.index())
	}

	/// Removes a key from the set.
	///
	/// # Arguments
	///
	/// * `key` - The key to remove
	///
	/// # Returns
	///
	/// `true` if the key was present, `false` otherwise
	pub fn remove(&mut self, key: K) -> bool {
		self.bits.remove(key.index())
	}

	/// Retains only the elements specified by the predicate.
	///
	/// In other words, remove all keys `k` such that `f(&k)` returns `false`.
	pub fn retain<F>(&mut self, mut f: F)
	where
		F: FnMut(K) -> bool,
	{
		self.bits.retain(|idx| f(K::from_index(idx)));
	}

	/// Returns an iterator over the keys of the set.
	///
	/// The iterator yields keys in the order of their index values.
	#[define_opaque(Iter)]
	pub fn iter(&self) -> Iter<'_, K> {
		self.bits.iter().map(K::from_index)
	}

	/// Returns the minimum (first) element in the set, or [`None`] if the set is
	/// empty.
	pub fn first(&self) -> Option<K> {
		self.bits.first().map(K::from_index)
	}

	/// Returns the maximum (last) element in the set, or [`None`] if the set is
	/// empty.
	pub fn last(&self) -> Option<K> {
		self.bits.last().map(K::from_index)
	}

	/// Returns `true` if the set has holes (gaps) in its indices.
	///
	/// A set is considered sparse if there are missing indices between the
	/// first and last elements. An empty set or a set with a single element
	/// is considered non-sparse.
	pub fn is_sparse(&self) -> bool {
		match (self.bits.first(), self.bits.last()) {
			(Some(first), Some(last)) => {
				// If we have all consecutive indices from first to last,
				// the count should equal (last - first + 1)
				self.len() < (last - first + 1)
			},
			_ => false, // Empty or single element sets are not sparse
		}
	}
}

/// Iterator over members in key order; see [`SparseSet::iter`].
pub type Iter<'a, K: TrySparseIndex> =
	impl DoubleEndedIterator<Item = K> + ExactSizeIterator + FusedIterator + Clone;
/// Owning iterator over members in key order.
pub type IntoIter<K: TrySparseIndex> =
	impl DoubleEndedIterator<Item = K> + ExactSizeIterator + FusedIterator + Clone;

impl<'a, K: TrySparseIndex> IntoIterator for &'a SparseSet<K> {
	type IntoIter = Iter<'a, K>;
	type Item = K;

	fn into_iter(self) -> Self::IntoIter {
		self.iter()
	}
}

impl<K: TrySparseIndex> IntoIterator for SparseSet<K> {
	type IntoIter = IntoIter<K>;
	type Item = K;

	#[define_opaque(IntoIter)]
	fn into_iter(self) -> Self::IntoIter {
		self.bits.into_iter().map(K::from_index)
	}
}

impl<K: TrySparseIndex> FromIterator<K> for SparseSet<K> {
	fn from_iter<T: IntoIterator<Item = K>>(iter: T) -> Self {
		let iter = iter.into_iter();
		let mut set = Self::with_capacity(iter.size_hint().0);
		for key in iter {
			set.insert(key);
		}
		set
	}
}

impl<K: TrySparseIndex> Extend<K> for SparseSet<K> {
	fn extend<T: IntoIterator<Item = K>>(&mut self, iter: T) {
		let iter = iter.into_iter();
		self.reserve(iter.size_hint().0);
		for key in iter {
			self.insert(key);
		}
	}
}
