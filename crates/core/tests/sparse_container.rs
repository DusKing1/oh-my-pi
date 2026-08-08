//! Comprehensive tests for `SparseSet` and `SparseMap` covering edge cases and
//! untested paths.

use std::num::NonZeroU8;

use omp_core::{
	sparse_index::{NumericIndexError, TrySparseIndex},
	sparse_map::SparseMap,
	sparse_set::SparseSet,
};

// ===== Test Keys =====

#[derive(Debug, thiserror::Error)]
#[error("invalid test key: {0}")]
struct TestKeyError(usize);

#[repr(usize)]
#[derive(
	Debug, Copy, Clone, PartialEq, Eq, Ord, PartialOrd, serde::Serialize, serde::Deserialize,
)]
enum TestKey {
	A = 0,
	B = 1,
	C = 2,
	Z = 100,
}

impl TrySparseIndex for TestKey {
	type Error = TestKeyError;

	fn index(&self) -> usize {
		*self as usize
	}

	fn try_from_index(index: usize) -> Result<Self, Self::Error> {
		match index {
			0 => Ok(Self::A),
			1 => Ok(Self::B),
			2 => Ok(Self::C),
			100 => Ok(Self::Z),
			_ => Err(TestKeyError(index)),
		}
	}
}

// ===== SparseSet Tests =====

#[test]
fn sparse_set_with_capacity() {
	let set: SparseSet<u8> = SparseSet::with_capacity(100);
	assert_eq!(set.capacity(), 127); // SmolBitmap rounds up to block boundary
	assert!(set.is_empty());
}

#[test]
fn sparse_set_reserve() {
	let mut set: SparseSet<u8> = SparseSet::new();
	set.reserve(100);
	assert!(set.capacity() >= 100);
}

#[test]
fn sparse_set_shrink_to_fit() {
	let mut set: SparseSet<u8> = SparseSet::with_capacity(1000);
	set.insert(1);
	set.insert(2);
	set.shrink_to_fit();

	assert!(set.capacity() < 1000);
}

#[test]
fn sparse_set_first_last_empty() {
	let set: SparseSet<u8> = SparseSet::new();
	assert_eq!(set.first(), None);
	assert_eq!(set.last(), None);
}

#[test]
fn sparse_set_first_last() {
	let mut set: SparseSet<u8> = SparseSet::new();
	set.insert(10);
	set.insert(5);
	set.insert(20);

	assert_eq!(set.first(), Some(5));
	assert_eq!(set.last(), Some(20));
}

#[test]
fn sparse_set_is_sparse_empty() {
	let set: SparseSet<u8> = SparseSet::new();
	assert!(!set.is_sparse());
}

#[test]
fn sparse_set_is_sparse_single() {
	let mut set: SparseSet<u8> = SparseSet::new();
	set.insert(42);
	assert!(!set.is_sparse());
}

#[test]
fn sparse_set_is_sparse_consecutive() {
	let mut set: SparseSet<u8> = SparseSet::new();
	set.insert(10);
	set.insert(11);
	set.insert(12);
	assert!(!set.is_sparse());
}

#[test]
fn sparse_set_is_sparse_with_gaps() {
	let mut set: SparseSet<u8> = SparseSet::new();
	set.insert(10);
	set.insert(15);
	assert!(set.is_sparse());
}

#[test]
fn sparse_set_union() {
	let mut set1: SparseSet<u8> = SparseSet::new();
	set1.insert(1);
	set1.insert(2);

	let mut set2: SparseSet<u8> = SparseSet::new();
	set2.insert(2);
	set2.insert(3);

	let union = set1.union(&set2);
	assert!(union.contains(1));
	assert!(union.contains(2));
	assert!(union.contains(3));
	assert_eq!(union.len(), 3);
}

#[test]
fn sparse_set_intersection() {
	let mut set1: SparseSet<u8> = SparseSet::new();
	set1.insert(1);
	set1.insert(2);
	set1.insert(3);

	let mut set2: SparseSet<u8> = SparseSet::new();
	set2.insert(2);
	set2.insert(3);
	set2.insert(4);

	let intersection = set1.intersection(&set2);
	assert!(!intersection.contains(1));
	assert!(intersection.contains(2));
	assert!(intersection.contains(3));
	assert!(!intersection.contains(4));
	assert_eq!(intersection.len(), 2);
}

#[test]
fn sparse_set_difference() {
	let mut set1: SparseSet<u8> = SparseSet::new();
	set1.insert(1);
	set1.insert(2);
	set1.insert(3);

	let mut set2: SparseSet<u8> = SparseSet::new();
	set2.insert(2);
	set2.insert(4);

	let diff = set1.difference(&set2);
	assert!(diff.contains(1));
	assert!(!diff.contains(2));
	assert!(diff.contains(3));
	assert!(!diff.contains(4));
}

#[test]
fn sparse_set_symmetric_difference() {
	let mut set1: SparseSet<u8> = SparseSet::new();
	set1.insert(1);
	set1.insert(2);

	let mut set2: SparseSet<u8> = SparseSet::new();
	set2.insert(2);
	set2.insert(3);

	let sym_diff = set1.symmetric_difference(&set2);
	assert!(sym_diff.contains(1));
	assert!(!sym_diff.contains(2));
	assert!(sym_diff.contains(3));
}

#[test]
fn sparse_set_is_subset() {
	let mut set1: SparseSet<u8> = SparseSet::new();
	set1.insert(1);
	set1.insert(2);

	let mut set2: SparseSet<u8> = SparseSet::new();
	set2.insert(1);
	set2.insert(2);
	set2.insert(3);

	assert!(set1.is_subset(&set2));
	assert!(!set2.is_subset(&set1));
}

#[test]
fn sparse_set_is_superset() {
	let mut set1: SparseSet<u8> = SparseSet::new();
	set1.insert(1);
	set1.insert(2);
	set1.insert(3);

	let mut set2: SparseSet<u8> = SparseSet::new();
	set2.insert(1);
	set2.insert(2);

	assert!(set1.is_superset(&set2));
	assert!(!set2.is_superset(&set1));
}

#[test]
fn sparse_set_is_disjoint() {
	let mut set1: SparseSet<u8> = SparseSet::new();
	set1.insert(1);
	set1.insert(2);

	let mut set2: SparseSet<u8> = SparseSet::new();
	set2.insert(3);
	set2.insert(4);

	assert!(set1.is_disjoint(&set2));

	set2.insert(2);
	assert!(!set1.is_disjoint(&set2));
}

#[test]
fn sparse_set_retain() {
	let mut set: SparseSet<u8> = SparseSet::new();
	set.insert(1);
	set.insert(2);
	set.insert(3);
	set.insert(4);

	set.retain(|k| k % 2 == 0);

	assert!(!set.contains(1));
	assert!(set.contains(2));
	assert!(!set.contains(3));
	assert!(set.contains(4));
}

#[test]
fn sparse_set_into_parts_from_parts() {
	let mut set: SparseSet<u8> = SparseSet::new();
	set.insert(1);
	set.insert(5);
	set.insert(10);

	let bits = set.into_parts();
	let restored: SparseSet<u8> = SparseSet::from_parts(bits);

	assert!(restored.contains(1));
	assert!(restored.contains(5));
	assert!(restored.contains(10));
	assert_eq!(restored.len(), 3);
}

#[test]
fn sparse_set_iter_double_ended() {
	let mut set: SparseSet<u8> = SparseSet::new();
	set.insert(1);
	set.insert(5);
	set.insert(10);

	let mut iter = set.iter();
	assert_eq!(iter.next(), Some(1));
	assert_eq!(iter.next_back(), Some(10));
	assert_eq!(iter.next(), Some(5));
	assert_eq!(iter.next(), None);
}

#[test]
fn sparse_set_iter_exact_size() {
	let mut set: SparseSet<u8> = SparseSet::new();
	set.insert(1);
	set.insert(5);

	let iter = set.iter();
	assert_eq!(iter.len(), 2);
}

#[test]
fn sparse_set_serde_binary() {
	let mut set: SparseSet<TestKey> = SparseSet::new();
	set.insert(TestKey::A);
	set.insert(TestKey::Z);

	let bytes = postcard::to_allocvec(&set).unwrap();
	let restored: SparseSet<TestKey> = postcard::from_bytes(&bytes).unwrap();

	assert_eq!(set, restored);
}

#[test]
fn sparse_set_serde_json() {
	let mut set: SparseSet<TestKey> = SparseSet::new();
	set.insert(TestKey::A);
	set.insert(TestKey::C);

	let json = serde_json::to_string(&set).unwrap();
	let restored: SparseSet<TestKey> = serde_json::from_str(&json).unwrap();

	assert_eq!(set, restored);
}

#[test]
fn sparse_set_serde_json_validates_ordering() {
	// Manually crafted out-of-order JSON should fail
	let bad_json = r#"[{"B":1},{"A":0}]"#;
	let result: Result<SparseSet<TestKey>, _> = serde_json::from_str(bad_json);
	assert!(result.is_err());
}

// ===== SparseMap Tests =====

#[test]
fn sparse_map_from_sequence() {
	let values = vec![10, 20, 30];
	let map: SparseMap<usize, i32> = SparseMap::from_sequence(values);

	assert_eq!(map.len(), 3);
	assert_eq!(map.get(0), Some(&10));
	assert_eq!(map.get(1), Some(&20));
	assert_eq!(map.get(2), Some(&30));
}

#[test]
fn sparse_map_with_capacity() {
	let map: SparseMap<u8, i32> = SparseMap::with_capacity(100);
	assert_eq!(map.capacity(), 127); // SmolBitmap rounds up to block boundary
	assert!(map.is_empty());
}

#[test]
fn sparse_map_reserve() {
	let mut map: SparseMap<u8, i32> = SparseMap::new();
	map.reserve(100);
	assert!(map.capacity() >= 100);
}

#[test]
fn sparse_map_shrink_to_fit() {
	let mut map: SparseMap<u8, i32> = SparseMap::with_capacity(1000);
	map.insert(1, 10);
	map.shrink_to_fit();

	assert!(map.capacity() < 1000);
}

#[test]
fn sparse_map_get_or_insert_new() {
	let mut map: SparseMap<u8, i32> = SparseMap::new();

	let val = map.get_or_insert(5, 100);
	assert_eq!(*val, 100);

	*val = 200;
	assert_eq!(map.get(5), Some(&200));
}

#[test]
fn sparse_map_get_or_insert_existing() {
	let mut map: SparseMap<u8, i32> = SparseMap::new();
	map.insert(5, 100);

	let val = map.get_or_insert(5, 999);
	assert_eq!(*val, 100); // Original value
}

#[test]
fn sparse_map_get_or_insert_with_new() {
	let mut map: SparseMap<u8, i32> = SparseMap::new();

	let val = map.get_or_insert_with(5, || 100);
	assert_eq!(*val, 100);
}

#[test]
fn sparse_map_get_or_insert_with_existing() {
	let mut map: SparseMap<u8, i32> = SparseMap::new();
	map.insert(5, 100);

	let mut called = false;
	let val = map.get_or_insert_with(5, || {
		called = true;
		999
	});

	assert_eq!(*val, 100);
	assert!(!called); // Closure not called
}

#[test]
fn sparse_map_key_set() {
	let mut map: SparseMap<TestKey, i32> = SparseMap::new();
	map.insert(TestKey::A, 10);
	map.insert(TestKey::C, 30);

	let key_set = map.key_set();
	assert!(key_set.contains(TestKey::A));
	assert!(!key_set.contains(TestKey::B));
	assert!(key_set.contains(TestKey::C));
}

#[test]
fn sparse_map_rank() {
	let mut map: SparseMap<u8, i32> = SparseMap::new();
	map.insert(5, 50);
	map.insert(10, 100);
	map.insert(15, 150);

	assert_eq!(map.rank(5), 0);
	assert_eq!(map.rank(10), 1);
	assert_eq!(map.rank(15), 2);
}

#[test]
fn sparse_map_first_last_empty() {
	let map: SparseMap<u8, i32> = SparseMap::new();
	assert_eq!(map.first(), None);
	assert_eq!(map.last(), None);
}

#[test]
fn sparse_map_first_last() {
	let mut map: SparseMap<u8, i32> = SparseMap::new();
	map.insert(10, 100);
	map.insert(5, 50);
	map.insert(20, 200);

	assert_eq!(map.first(), Some((5, &50)));
	assert_eq!(map.last(), Some((20, &200)));
}

#[test]
fn sparse_map_is_sparse_empty() {
	let map: SparseMap<u8, i32> = SparseMap::new();
	assert!(!map.is_sparse());
}

#[test]
fn sparse_map_is_sparse_single() {
	let mut map: SparseMap<u8, i32> = SparseMap::new();
	map.insert(42, 100);
	assert!(!map.is_sparse());
}

#[test]
fn sparse_map_is_sparse_consecutive() {
	let mut map: SparseMap<u8, i32> = SparseMap::new();
	map.insert(10, 100);
	map.insert(11, 110);
	map.insert(12, 120);
	assert!(!map.is_sparse());
}

#[test]
fn sparse_map_is_sparse_with_gaps() {
	let mut map: SparseMap<u8, i32> = SparseMap::new();
	map.insert(10, 100);
	map.insert(15, 150);
	assert!(map.is_sparse());
}

#[test]
fn sparse_map_into_parts_from_parts() {
	let mut map: SparseMap<u8, i32> = SparseMap::new();
	map.insert(1, 10);
	map.insert(5, 50);

	let (bits, values) = map.into_parts();
	let restored: SparseMap<u8, i32> = SparseMap::from_parts(bits, values);

	assert_eq!(restored.get(1), Some(&10));
	assert_eq!(restored.get(5), Some(&50));
}

#[test]
#[should_panic(expected = "bitmap and values length mismatch")]
fn sparse_map_from_parts_mismatched() {
	use smol_bitmap::SmolBitmap;

	let mut bits = SmolBitmap::new();
	bits.insert(1);
	bits.insert(2);

	let values = vec![10]; // Wrong length

	let _map: SparseMap<u8, i32> = SparseMap::from_parts(bits, values);
}

#[test]
fn sparse_map_values_values_mut() {
	let mut map: SparseMap<u8, i32> = SparseMap::new();
	map.insert(1, 10);
	map.insert(2, 20);

	let values: Vec<_> = map.values().copied().collect();
	assert_eq!(values, vec![10, 20]);

	for val in map.values_mut() {
		*val *= 2;
	}

	assert_eq!(map.get(1), Some(&20));
	assert_eq!(map.get(2), Some(&40));
}

#[test]
fn sparse_map_keys() {
	let mut map: SparseMap<TestKey, i32> = SparseMap::new();
	map.insert(TestKey::A, 10);
	map.insert(TestKey::C, 30);

	let keys: Vec<_> = map.keys().collect();
	assert_eq!(keys, vec![TestKey::A, TestKey::C]);
}

#[test]
fn sparse_map_iter_double_ended() {
	let mut map: SparseMap<u8, i32> = SparseMap::new();
	map.insert(1, 10);
	map.insert(5, 50);
	map.insert(10, 100);

	let mut iter = map.iter();
	assert_eq!(iter.next(), Some((1, &10)));
	assert_eq!(iter.next_back(), Some((10, &100)));
	assert_eq!(iter.next(), Some((5, &50)));
	assert_eq!(iter.next(), None);
}

#[test]
fn sparse_map_index_trait() {
	let mut map: SparseMap<u8, i32> = SparseMap::new();
	map.insert(5, 50);

	assert_eq!(map[5], 50);
}

#[test]
#[should_panic(expected = "key not found")]
fn sparse_map_index_trait_panic() {
	let map: SparseMap<u8, i32> = SparseMap::new();
	let _ = map[5];
}

#[test]
fn sparse_map_index_mut_trait() {
	let mut map: SparseMap<u8, i32> = SparseMap::new();
	map.insert(5, 50);

	map[5] = 100;
	assert_eq!(map.get(5), Some(&100));
}

#[test]
fn sparse_map_serde_binary() {
	let mut map: SparseMap<TestKey, i32> = SparseMap::new();
	map.insert(TestKey::A, 10);
	map.insert(TestKey::Z, 100);

	let bytes = postcard::to_allocvec(&map).unwrap();
	let restored: SparseMap<TestKey, i32> = postcard::from_bytes(&bytes).unwrap();

	assert_eq!(map, restored);
}

#[test]
fn sparse_map_serde_json() {
	let mut map: SparseMap<TestKey, i32> = SparseMap::new();
	map.insert(TestKey::A, 10);
	map.insert(TestKey::C, 30);

	let json = serde_json::to_string(&map).unwrap();
	let restored: SparseMap<TestKey, i32> = serde_json::from_str(&json).unwrap();

	assert_eq!(map, restored);
}

#[test]
fn sparse_map_from_iter_ordered() {
	let pairs = vec![(TestKey::A, 10), (TestKey::B, 20), (TestKey::C, 30)];

	let map: SparseMap<_, _> = pairs.into_iter().collect();

	assert_eq!(map.len(), 3);
	assert_eq!(map.get(TestKey::A), Some(&10));
}

#[test]
fn sparse_map_from_iter_unordered() {
	let pairs = vec![(TestKey::C, 30), (TestKey::A, 10), (TestKey::B, 20)];

	let map: SparseMap<_, _> = pairs.into_iter().collect();

	assert_eq!(map.len(), 3);
	assert_eq!(map.get(TestKey::A), Some(&10));
}

#[test]
fn sparse_map_from_iter_duplicate_keys() {
	let pairs = vec![(TestKey::A, 10), (TestKey::A, 20)];

	let map: SparseMap<_, _> = pairs.into_iter().collect();

	// Last value wins
	assert_eq!(map.len(), 1);
	assert_eq!(map.get(TestKey::A), Some(&20));
}

#[test]
fn sparse_map_extend() {
	let mut map: SparseMap<TestKey, i32> = SparseMap::new();
	map.insert(TestKey::A, 10);

	map.extend(vec![(TestKey::B, 20), (TestKey::C, 30)]);

	assert_eq!(map.len(), 3);
}

#[test]
fn sparse_map_extend_with_duplicates() {
	let mut map: SparseMap<TestKey, i32> = SparseMap::new();
	map.insert(TestKey::A, 10);

	map.extend(vec![(TestKey::A, 100)]);

	assert_eq!(map.get(TestKey::A), Some(&100));
}

// ===== TrySparseIndex Tests =====

#[test]
fn try_sparse_index_u8_bounds() {
	assert!(u8::try_from_index(255).is_ok());
	assert!(u8::try_from_index(256).is_err());
}

#[test]
fn try_sparse_index_u16_bounds() {
	assert!(u16::try_from_index(65535).is_ok());
	assert!(u16::try_from_index(65536).is_err());
}

#[test]
fn try_sparse_index_i8_bounds() {
	assert!(i8::try_from_index(127).is_ok());
	assert!(i8::try_from_index(128).is_err());
}

#[test]
fn try_sparse_index_nonzero_u8() {
	// NonZeroU8 index 0 maps to value 1
	let nz = NonZeroU8::try_from_index(0).unwrap();
	assert_eq!(nz.get(), 1);
	assert_eq!(nz.index(), 0);

	// Index 254 maps to value 255
	let nz = NonZeroU8::try_from_index(254).unwrap();
	assert_eq!(nz.get(), 255);

	// Index 255 is out of bounds
	assert!(NonZeroU8::try_from_index(255).is_err());
}

#[test]
fn try_sparse_index_validate_sorted() {
	let indices = vec![1, 5, 10, 20];
	assert!(u8::validate_sorted(indices.into_iter()).is_ok());

	let bad_indices = vec![1, 5, 300];
	assert!(u8::validate_sorted(bad_indices.into_iter()).is_err());
}

#[test]
fn try_sparse_index_validate_sorted_empty() {
	let indices: Vec<usize> = vec![];
	assert!(u8::validate_sorted(indices.into_iter()).is_ok());
}

#[test]
fn numeric_index_error_display() {
	let err = NumericIndexError::OutOfBounds { max: 255, received: 300 };
	let msg = format!("{err}");
	assert!(msg.contains("255"));
	assert!(msg.contains("300"));
}

// ===== Edge Case Combinations =====

#[test]
fn sparse_set_retain_all_removed() {
	let mut set: SparseSet<u8> = SparseSet::new();
	set.insert(1);
	set.insert(2);
	set.insert(3);

	set.retain(|_| false);

	assert!(set.is_empty());
}

#[test]
fn sparse_map_retain_all_removed() {
	let mut map: SparseMap<u8, i32> = SparseMap::new();
	map.insert(1, 10);
	map.insert(2, 20);
	map.insert(3, 30);

	map.retain(|_, _| false);

	assert!(map.is_empty());
}

#[test]
fn sparse_map_retain_with_mutation() {
	let mut map: SparseMap<u8, i32> = SparseMap::new();
	map.insert(1, 10);
	map.insert(2, 20);
	map.insert(3, 30);

	map.retain(|k, v| {
		*v *= 2;
		k % 2 == 1
	});

	assert_eq!(map.get(1), Some(&20));
	assert_eq!(map.get(2), None);
	assert_eq!(map.get(3), Some(&60));
}

#[test]
fn sparse_set_clone() {
	let mut set1: SparseSet<u8> = SparseSet::new();
	set1.insert(1);
	set1.insert(5);

	let set2 = set1.clone();

	assert_eq!(set1, set2);
}

#[test]
fn sparse_map_clone() {
	let mut map1: SparseMap<u8, String> = SparseMap::new();
	map1.insert(1, "hello".to_string());

	let map2 = map1.clone();

	assert_eq!(map1, map2);
}

#[test]
fn sparse_set_debug_format() {
	let mut set: SparseSet<TestKey> = SparseSet::new();
	set.insert(TestKey::A);
	set.insert(TestKey::C);

	let debug = format!("{set:?}");
	assert!(debug.contains('A'));
	assert!(debug.contains('C'));
}

#[test]
fn sparse_map_debug_format() {
	let mut map: SparseMap<TestKey, i32> = SparseMap::new();
	map.insert(TestKey::A, 10);

	let debug = format!("{map:?}");
	assert!(debug.contains('A'));
	assert!(debug.contains("10"));
}

#[test]
fn sparse_set_default() {
	let set: SparseSet<u8> = Default::default();
	assert!(set.is_empty());
}

#[test]
fn sparse_map_default() {
	let map: SparseMap<u8, i32> = Default::default();
	assert!(map.is_empty());
}

// ===== NonZero from_index boundary regressions =====
// `from_index` must never construct NonZero(0) through wrap/truncation; out of
// range indices panic instead.

#[test]
fn test_nonzero_u8_from_index_upper_boundary() {
	let v = NonZeroU8::from_index(254);
	assert_eq!(v.get(), 255);
	assert_eq!(v.index(), 254);
}

#[test]
#[should_panic(expected = "index out of range")]
fn test_nonzero_u8_from_index_wrap_panics() {
	// 255 + 1 truncates to 0 as u8 — previously constructed NonZeroU8(0).
	let _ = NonZeroU8::from_index(255);
}

#[test]
#[should_panic(expected = "index out of range")]
fn test_nonzero_i8_from_index_truncation_panics() {
	// 255 + 1 = 256 truncates to 0 as i8 despite the checked usize add.
	let _ = std::num::NonZeroI8::from_index(255);
}

#[test]
fn test_nonzero_i8_from_index_upper_boundary() {
	let v = std::num::NonZeroI8::from_index(126);
	assert_eq!(v.get(), 127);
	assert_eq!(v.index(), 126);
}

#[test]
#[should_panic(expected = "index out of range")]
fn test_nonzero_usize_from_index_overflow_panics() {
	// usize::MAX + 1 wraps to 0 in release builds.
	let _ = std::num::NonZeroUsize::from_index(usize::MAX);
}

#[test]
fn test_nonzero_usize_from_index_upper_boundary() {
	assert_eq!(std::num::NonZeroUsize::from_index(usize::MAX - 1).get(), usize::MAX);
}

// ===== Malformed binary input regressions =====
// The non-human-readable format is `(bits, values)`; a forged payload with
// mismatched occupancy/value counts must be rejected at deserialization, not
// panic later inside safe insert/remove.

#[test]
fn test_sparse_map_binary_rejects_bitmap_value_count_mismatch() {
	let mut bits = smol_bitmap::SmolBitmap::new();
	bits.insert(TestKey::A.index());
	bits.insert(TestKey::C.index());

	// Two occupied slots, one value: reject.
	let forged = postcard::to_allocvec(&(&bits, vec![7i32])).unwrap();
	postcard::from_bytes::<SparseMap<TestKey, i32>>(&forged)
		.expect_err("missing values must be rejected");

	// Two occupied slots, three values: reject.
	let forged = postcard::to_allocvec(&(&bits, vec![7i32, 8, 9])).unwrap();
	postcard::from_bytes::<SparseMap<TestKey, i32>>(&forged)
		.expect_err("excess values must be rejected");

	// Matching counts parse and associate correctly.
	let valid = postcard::to_allocvec(&(&bits, vec![7i32, 9])).unwrap();
	let map = postcard::from_bytes::<SparseMap<TestKey, i32>>(&valid).unwrap();
	assert_eq!(map.get(TestKey::A), Some(&7));
	assert_eq!(map.get(TestKey::C), Some(&9));
	assert_eq!(map.get(TestKey::B), None);
}

#[test]
fn test_sparse_map_binary_rejects_gapped_key_payload() {
	// TestKey is valid only at {0, 1, 2, 100}: index 50 sits in a gap. The
	// extremes (0, 100) are valid, so a min/max-only validation would accept
	// this payload and iteration would panic on from_index(50).
	let mut bits = smol_bitmap::SmolBitmap::new();
	bits.insert(TestKey::A.index());
	bits.insert(50);
	bits.insert(TestKey::Z.index());

	let forged = postcard::to_allocvec(&(&bits, vec![1i32, 2, 3])).unwrap();
	postcard::from_bytes::<SparseMap<TestKey, i32>>(&forged)
		.expect_err("gap index must be rejected");
}

#[test]
fn test_sparse_set_binary_rejects_gapped_key_payload() {
	let mut bits = smol_bitmap::SmolBitmap::new();
	bits.insert(TestKey::A.index());
	bits.insert(50);
	bits.insert(TestKey::Z.index());

	let forged = postcard::to_allocvec(&bits).unwrap();
	postcard::from_bytes::<SparseSet<TestKey>>(&forged).expect_err("gap index must be rejected");
}
