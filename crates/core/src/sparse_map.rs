//! Sparse map backed by bitmap occupancy tracking.
//!
//! `SparseMap<K, V>` stores key-value pairs where keys map to indices. Uses a
//! bitmap for presence and a packed vector for values, minimizing memory
//! overhead for sparse mappings.

use std::{
	fmt,
	iter::{self, FusedIterator},
	marker::PhantomData,
	ops::{Index, IndexMut},
	slice,
};

use serde::{Deserialize, Serialize};
use smol_bitmap::SmolBitmap;

use crate::{sparse_index::TrySparseIndex, sparse_set::SparseSet};

/// A sparse map from keys convertible to indices to values.
///
/// [`SparseMap`] provides an efficient storage mechanism for mappings
/// where keys can be converted to `usize` indices. It uses a bitmap to track
/// which indices are occupied and a packed vector to store only the present
/// values, achieving both memory efficiency and fast lookup times.
///
/// The key type `K` must implement [`Into<usize>`] and [`From<usize>`] to
/// convert between keys and indices. This makes it ideal for enum keys,
/// small integers, or other types with a natural index representation.
///
/// # Examples
///
/// ```
/// use omp_core::{sparse_index::TrySparseIndex, sparse_map::SparseMap};
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
/// struct StatusError(String);
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
/// 			_ => Err(StatusError("Invalid status value".to_string())),
/// 		}
/// 	}
/// }
///
/// let mut map = SparseMap::new();
/// map.insert(Status::Active, "running");
/// map.insert(Status::Closed, "finished");
///
/// assert_eq!(map.get(Status::Active), Some(&"running"));
/// assert_eq!(map.get(Status::Pending), None);
/// assert_eq!(map[Status::Active], "running");
/// ```
pub struct SparseMap<K, V> {
	/// Bitmap tracking which indices have values stored
	bits:     SmolBitmap,
	/// Packed storage for values at occupied indices
	values:   Vec<V>,
	/// Phantom data to maintain type information for keys
	_phantom: PhantomData<K>,
}

impl<K, V: Clone> Clone for SparseMap<K, V> {
	fn clone(&self) -> Self {
		Self { bits: self.bits.clone(), values: self.values.clone(), _phantom: PhantomData }
	}
}

impl<K: TrySparseIndex + fmt::Debug, V: fmt::Debug> fmt::Debug for SparseMap<K, V> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		let mut s = f.debug_map();
		for (index, value) in self {
			s.entry(&index, &value);
		}
		s.finish()
	}
}
impl<K: TrySparseIndex + Serialize, V: Serialize> Serialize for SparseMap<K, V> {
	fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
	where
		S: serde::Serializer,
	{
		if serializer.is_human_readable() {
			use serde::ser::SerializeSeq;
			let mut seq = serializer.serialize_seq(Some(self.len()))?;
			for (index, value) in self {
				seq.serialize_element(&(index, value))?;
			}
			seq.end()
		} else {
			use serde::ser::SerializeTuple;
			let mut tuple = serializer.serialize_tuple(2)?;
			tuple.serialize_element(&self.bits)?;
			tuple.serialize_element(&self.values)?;
			tuple.end()
		}
	}
}

impl<'de, K: TrySparseIndex + Deserialize<'de>, V: Deserialize<'de>> Deserialize<'de>
	for SparseMap<K, V>
{
	fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
	where
		D: serde::Deserializer<'de>,
	{
		if deserializer.is_human_readable() {
			use std::fmt;

			use serde::de::{SeqAccess, Visitor};

			struct SparseMapVisitor<K, V> {
				_phantom: PhantomData<(K, V)>,
			}

			impl<'de, K: TrySparseIndex + Deserialize<'de>, V: Deserialize<'de>> Visitor<'de>
				for SparseMapVisitor<K, V>
			{
				type Value = SparseMap<K, V>;

				fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
					formatter.write_str("a sequence of (index, value) pairs")
				}

				fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
				where
					A: SeqAccess<'de>,
				{
					let mut map = SparseMap::new();
					if let Some(len) = seq.size_hint() {
						map.reserve(len);
					}
					let mut prev_index = None;
					while let Some((index, value)) = seq.next_element::<(K, V)>()? {
						let index = index.index();
						if prev_index.is_some_and(|prev| prev >= index) {
							return Err(serde::de::Error::invalid_value(
								serde::de::Unexpected::Unsigned(index as u64),
								&"indices must be in ascending order",
							));
						}
						prev_index = Some(index);
						map.bits.insert(index);
						map.values.push(value);
					}
					Ok(map)
				}
			}

			deserializer.deserialize_seq(SparseMapVisitor { _phantom: PhantomData })
		} else {
			let (bits, values): (SmolBitmap, Vec<V>) = Deserialize::deserialize(deserializer)?;
			// Reject occupancy/value mismatches: rank-based lookups misassociate
			// values (or panic in insert/remove) if the packed vector length does
			// not equal the number of set bits.
			if bits.count_ones() != values.len() {
				return Err(serde::de::Error::invalid_length(
					values.len(),
					&"as many values as set bits in the occupancy bitmap",
				));
			}
			K::validate_sorted(bits.iter()).map_err(serde::de::Error::custom)?;
			Ok(Self { bits, values, _phantom: PhantomData })
		}
	}
}

impl<K, V> Default for SparseMap<K, V> {
	fn default() -> Self {
		Self::new()
	}
}

impl<K, V: PartialEq> PartialEq for SparseMap<K, V> {
	fn eq(&self, other: &Self) -> bool {
		self.bits == other.bits && self.values == other.values
	}
}

impl<K, V: Eq> Eq for SparseMap<K, V> {}

impl<K, V> SparseMap<K, V> {
	/// Creates a sparse index map from a sequence of values.
	///
	/// The values are assigned indices starting from 0. This is equivalent to
	/// inserting each value with its position as the key.
	///
	/// # Arguments
	///
	/// * `values` - A sequence of values to insert
	pub fn from_sequence(values: Vec<V>) -> Self {
		let len = values.len();
		let mut bits = SmolBitmap::with_capacity(len);

		// Set all bits from 0 to len-1
		for i in 0..len {
			bits.insert(i);
		}

		Self { bits, values, _phantom: PhantomData }
	}

	/// Creates a new empty sparse index map.
	pub const fn new() -> Self {
		Self { bits: SmolBitmap::new(), values: Vec::new(), _phantom: PhantomData }
	}

	/// Creates a new sparse index map with the specified capacity.
	///
	/// # Arguments
	///
	/// * `capacity` - The maximum index that might be stored
	pub fn with_capacity(capacity: usize) -> Self {
		Self {
			bits:     SmolBitmap::with_capacity(capacity),
			values:   Vec::with_capacity(capacity),
			_phantom: PhantomData,
		}
	}

	/// Returns the number of key-value pairs in the map.
	#[inline]
	pub const fn len(&self) -> usize {
		self.values.len()
	}

	/// Returns `true` if the map contains no elements.
	#[inline]
	pub const fn is_empty(&self) -> bool {
		self.values.is_empty()
	}

	/// Returns the capacity of the underlying bitmap.
	#[inline]
	pub const fn capacity(&self) -> usize {
		self.bits.capacity()
	}

	/// Clears the map, removing all key-value pairs.
	pub fn clear(&mut self) {
		self.bits.clear();
		self.values.clear();
	}

	/// Shrinks the capacity of the map as much as possible.
	pub fn shrink_to_fit(&mut self) {
		self.bits.shrink_to_fit();
		self.values.shrink_to_fit();
	}

	/// Reserves capacity for at least `additional` more elements to be
	/// inserted in the map.
	pub fn reserve(&mut self, additional: usize) {
		self.bits.reserve(additional);
		self.values.reserve(additional);
	}

	/// Returns a set of all keys in the map.
	pub const fn key_set(&self) -> &SparseSet<K> {
		// SAFETY: This is safe because SparseSet is repr(transparent)
		// over SmolBitmap.
		unsafe { &*(&raw const self.bits).cast::<SparseSet<K>>() }
	}

	/// Decomposes the map into its raw parts: bitmap and values vector.
	///
	/// # Returns
	///
	/// A tuple of `(SmolBitmap, Vec<V>)` representing the bitmap and values
	#[inline]
	pub fn into_parts(self) -> (SmolBitmap, Vec<V>) {
		(self.bits, self.values)
	}

	/// Constructs a sparse map from its raw parts: bitmap and values vector.
	///
	/// # Arguments
	///
	/// * `bits` - The bitmap tracking which indices have values
	/// * `values` - The packed vector of values corresponding to set bits
	///
	/// # Safety
	///
	/// The caller must ensure that the bitmap and values vector are consistent:
	/// - The number of set bits in the bitmap must equal the length of the
	///   values vector
	/// - The values must be in the order corresponding to the set bits in the
	///   bitmap
	#[inline]
	pub fn from_parts(bits: SmolBitmap, values: Vec<V>) -> Self {
		assert_eq!(bits.count_ones(), values.len(), "bitmap and values length mismatch");
		Self { bits, values, _phantom: PhantomData }
	}
}

impl<K: TrySparseIndex, V> SparseMap<K, V> {
	/// Gets a reference to the value corresponding to the key.
	///
	/// # Arguments
	///
	/// * `key` - The key to look up
	///
	/// # Returns
	///
	/// A reference to the value, or [`None`] if the key is not present
	#[inline]
	pub fn get(&self, key: K) -> Option<&V> {
		let index = key.index();
		if self.bits.get(index) {
			self.values.get(self.bits.rank(index))
		} else {
			None
		}
	}

	/// Gets a mutable reference to the value corresponding to the key.
	///
	/// # Arguments
	///
	/// * `key` - The key to look up
	///
	/// # Returns
	///
	/// A mutable reference to the value, or [`None`] if the key is not
	/// present
	#[inline]
	pub fn get_mut(&mut self, key: K) -> Option<&mut V> {
		let index = key.index();
		if self.bits.get(index) {
			let pos = self.bits.rank(index);
			self.values.get_mut(pos)
		} else {
			None
		}
	}

	/// Gets the value corresponding to the key, or inserts a default value if
	/// not present.
	///
	/// If the key exists in the map, returns a mutable reference to the existing
	/// value. If the key does not exist, inserts the provided value and
	/// returns a mutable reference to it.
	///
	/// # Arguments
	///
	/// * `key` - The key to look up or insert
	/// * `value` - The value to insert if the key is not present
	///
	/// # Returns
	///
	/// A mutable reference to the value (either existing or newly inserted)
	#[inline]
	pub fn get_or_insert(&mut self, key: K, value: V) -> &mut V {
		let index = key.index();
		let pos = if self.bits.insert(index) {
			// New entry
			let pos = self.bits.rank(index);
			self.values.insert(pos, value);
			pos
		} else {
			// Existing entry
			self.bits.rank(index)
		};
		&mut self.values[pos]
	}

	/// Gets the value corresponding to the key, or inserts a computed default
	/// value if not present.
	///
	/// If the key exists in the map, returns a mutable reference to the existing
	/// value. If the key does not exist, calls the provided closure to
	/// compute a value, inserts it, and returns a mutable reference to it.
	///
	/// # Arguments
	///
	/// * `key` - The key to look up or insert
	/// * `f` - A closure that computes the value to insert if the key is not
	///   present
	///
	/// # Returns
	///
	/// A mutable reference to the value (either existing or newly inserted)
	#[inline]
	pub fn get_or_insert_with<F>(&mut self, key: K, f: F) -> &mut V
	where
		F: FnOnce() -> V,
	{
		let index = key.index();
		let pos = if self.bits.insert(index) {
			// New entry
			let pos = self.bits.rank(index);
			self.values.insert(pos, f());
			pos
		} else {
			// Existing entry
			self.bits.rank(index)
		};
		&mut self.values[pos]
	}

	/// Returns `true` if the map contains a value for the specified key.
	///
	/// # Arguments
	///
	/// * `key` - The key to check for
	#[inline]
	pub fn contains_key(&self, key: K) -> bool {
		self.bits.get(key.index())
	}

	/// Inserts a key-value pair into the map.
	///
	/// If the map did not have this key present, [`None`] is returned.
	/// If the map did have this key present, the value is updated and the old
	/// value is returned.
	///
	/// # Arguments
	///
	/// * `key` - The key to insert
	/// * `value` - The value to associate with the key
	///
	/// # Returns
	///
	/// The previous value associated with the key, if any
	pub fn insert(&mut self, key: K, value: V) -> Option<V> {
		let index = key.index();
		let pos = self.bits.rank(index);
		if self.bits.insert(index) {
			// New entry
			self.values.insert(pos, value);
			None
		} else {
			// Existing entry - replace value
			Some(std::mem::replace(&mut self.values[pos], value))
		}
	}

	/// Removes a key from the map, returning the value at the key if the key
	/// was previously in the map.
	///
	/// # Arguments
	///
	/// * `key` - The key to remove
	///
	/// # Returns
	///
	/// The removed value, or [`None`] if the key was not present
	pub fn remove(&mut self, key: K) -> Option<V> {
		let index = key.index();
		if self.bits.remove(index) {
			let pos = self.bits.rank(index);
			Some(self.values.remove(pos))
		} else {
			None
		}
	}

	/// Retains only the elements specified by the predicate.
	///
	/// In other words, remove all pairs `(k, v)` such that `f(&k, &mut v)`
	/// returns `false`.
	pub fn retain<F>(&mut self, mut f: F)
	where
		F: FnMut(K, &mut V) -> bool,
	{
		let mut write_idx = 0;
		let mut read_idx = 0;
		self.bits.retain(|idx| {
			let key = K::from_index(idx);
			let should_retain = f(key, &mut self.values[read_idx]);
			if should_retain {
				if write_idx != read_idx {
					self.values.swap(write_idx, read_idx);
				}
				write_idx += 1;
			}
			read_idx += 1;
			should_retain
		});
		self.values.truncate(write_idx);
	}

	/// Returns an iterator over the key-value pairs of the map.
	///
	/// The iterator yields pairs in the order of their index values.
	#[define_opaque(Iter)]
	pub fn iter(&self) -> Iter<'_, K, V> {
		iter::zip(self.bits.iter().map(K::from_index), self.values.iter())
	}

	/// Returns a mutable iterator over the key-value pairs of the map.
	#[define_opaque(IterMut)]
	pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
		iter::zip(self.bits.iter().map(K::from_index), self.values.iter_mut())
	}

	/// Returns an iterator over the keys of the map.
	#[define_opaque(KeyIter)]
	pub fn keys(&self) -> KeyIter<'_, K> {
		self.bits.iter().map(K::from_index)
	}

	/// Returns an iterator over the values of the map.
	#[inline]
	pub fn values(&self) -> slice::Iter<'_, V> {
		self.values.iter()
	}

	/// Returns a mutable iterator over the values of the map.
	#[inline]
	pub fn values_mut(&mut self) -> slice::IterMut<'_, V> {
		self.values.iter_mut()
	}

	/// Finds the position in the values vector for the given index.
	///
	/// This counts the number of set bits before the given index to determine
	/// where the corresponding value is stored in the packed values vector.
	#[inline]
	pub fn rank(&self, key: K) -> usize {
		self.bits.rank(key.index())
	}

	/// Returns the first (minimum) key-value pair in the map, or [`None`] if the
	/// map is empty.
	pub fn first(&self) -> Option<(K, &V)> {
		self
			.bits
			.first()
			.and_then(|idx| self.values.first().map(|v| (K::from_index(idx), v)))
	}

	/// Returns the last (maximum) key-value pair in the map, or [`None`] if the
	/// map is empty.
	pub fn last(&self) -> Option<(K, &V)> {
		self
			.bits
			.last()
			.and_then(|idx| self.values.last().map(|v| (K::from_index(idx), v)))
	}

	/// Returns `true` if the map has holes (gaps) in its indices.
	///
	/// A map is considered sparse if there are missing indices between the
	/// first and last elements. An empty map or a map with a single element
	/// is considered non-sparse.
	pub fn is_sparse(&self) -> bool {
		match (self.bits.first(), self.bits.last()) {
			(Some(first), Some(last)) => {
				// If we have all consecutive indices from first to last,
				// the count should equal (last - first + 1)
				self.len() < (last - first + 1)
			},
			_ => false, // Empty or single element maps are not sparse
		}
	}
}

/// Iterator over key/value pairs in key order; see [`SparseMap::iter`].
pub type Iter<'a, K: TrySparseIndex, V: 'a> =
	impl DoubleEndedIterator<Item = (K, &'a V)> + ExactSizeIterator + FusedIterator + Clone;
/// Iterator over key/mutable-value pairs in key order; see
/// [`SparseMap::iter_mut`].
pub type IterMut<'a, K: TrySparseIndex, V: 'a> =
	impl DoubleEndedIterator<Item = (K, &'a mut V)> + ExactSizeIterator + FusedIterator;
/// Iterator over keys in key order; see [`SparseMap::keys`].
pub type KeyIter<'a, K: TrySparseIndex> =
	impl DoubleEndedIterator<Item = K> + ExactSizeIterator + FusedIterator + Clone;
/// Owning iterator over key/value pairs in key order.
pub type IntoIter<K: TrySparseIndex, V> =
	impl DoubleEndedIterator<Item = (K, V)> + ExactSizeIterator + FusedIterator;

// Index trait for convenient access
impl<K: TrySparseIndex, V> Index<K> for SparseMap<K, V> {
	type Output = V;

	fn index(&self, key: K) -> &Self::Output {
		self.get(key).expect("key not found in SparseMap")
	}
}

impl<K: TrySparseIndex, V> IndexMut<K> for SparseMap<K, V> {
	fn index_mut(&mut self, key: K) -> &mut Self::Output {
		self.get_mut(key).expect("key not found in SparseMap")
	}
}

impl<'a, K: TrySparseIndex, V> IntoIterator for &'a SparseMap<K, V> {
	type IntoIter = Iter<'a, K, V>;
	type Item = (K, &'a V);

	fn into_iter(self) -> Self::IntoIter {
		self.iter()
	}
}

impl<'a, K: TrySparseIndex, V> IntoIterator for &'a mut SparseMap<K, V> {
	type IntoIter = IterMut<'a, K, V>;
	type Item = (K, &'a mut V);

	fn into_iter(self) -> Self::IntoIter {
		self.iter_mut()
	}
}

impl<K: TrySparseIndex, V> IntoIterator for SparseMap<K, V> {
	type IntoIter = IntoIter<K, V>;
	type Item = (K, V);

	#[define_opaque(IntoIter)]
	fn into_iter(self) -> Self::IntoIter {
		let indices = self.bits.into_iter().map(K::from_index);
		let values = self.values.into_iter();
		indices.zip(values)
	}
}

impl<K: TrySparseIndex, V> FromIterator<(K, V)> for SparseMap<K, V> {
	fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
		let iter = iter.into_iter();
		let mut map = Self::with_capacity(iter.size_hint().0);
		let mut hi = None;

		for (key, value) in iter {
			let ix = key.index();
			match hi {
				Some(hi) if ix == hi => {
					*map.values.last_mut().expect("can't have key without value") = value;
				},
				Some(hi) if ix < hi => {
					map.insert(K::from_index(ix), value);
				},
				_ => {
					map.bits.insert(ix);
					map.values.push(value);
					hi = Some(ix);
				},
			}
		}

		map
	}
}

impl<K: TrySparseIndex, V> Extend<(K, V)> for SparseMap<K, V> {
	fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
		let mut hi = self.bits.last();
		for (key, value) in iter {
			let ix = key.index();
			match hi {
				Some(hi) if ix == hi => {
					*self
						.values
						.last_mut()
						.expect("can't have key without value") = value;
				},
				Some(hi) if ix < hi => {
					self.insert(K::from_index(ix), value);
				},
				_ => {
					self.bits.insert(ix);
					self.values.push(value);
					hi = Some(ix);
				},
			}
		}
	}
}
