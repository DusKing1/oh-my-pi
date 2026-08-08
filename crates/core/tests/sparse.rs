//! `SparseMap` / `SparseSet` contract tests.
use omp_core::{sparse_index::SparseIndex, sparse_map::SparseMap};

#[repr(usize)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum TestKey {
	First  = 0,
	Second = 1,
	Third  = 2,
	Far    = 100,
}

impl SparseIndex for TestKey {
	fn index(&self) -> usize {
		*self as usize
	}

	fn from_index(index: usize) -> Self {
		Self::from(index)
	}
}

impl From<TestKey> for usize {
	fn from(key: TestKey) -> Self {
		key as Self
	}
}

impl From<usize> for TestKey {
	fn from(value: usize) -> Self {
		match value {
			0 => Self::First,
			1 => Self::Second,
			2 => Self::Third,
			100 => Self::Far,
			_ => unreachable!("Invalid TestKey value: {value}"),
		}
	}
}

#[test]
fn test_new_and_empty() {
	let map: SparseMap<TestKey, i32> = SparseMap::new();
	assert!(map.is_empty());
	assert_eq!(map.len(), 0);
}

#[test]
fn test_insert_and_get() {
	let mut map = SparseMap::new();

	assert_eq!(map.insert(TestKey::First, 10), None);
	assert_eq!(map.insert(TestKey::Third, 30), None);

	assert_eq!(map.get(TestKey::First), Some(&10));
	assert_eq!(map.get(TestKey::Second), None);
	assert_eq!(map.get(TestKey::Third), Some(&30));
	assert_eq!(map.len(), 2);
}

#[test]
fn test_insert_replace() {
	let mut map = SparseMap::new();

	map.insert(TestKey::First, 10);
	assert_eq!(map.insert(TestKey::First, 20), Some(10));
	assert_eq!(map.get(TestKey::First), Some(&20));
	assert_eq!(map.len(), 1);
}

#[test]
fn test_remove() {
	let mut map = SparseMap::new();

	map.insert(TestKey::First, 10);
	map.insert(TestKey::Second, 20);
	map.insert(TestKey::Third, 30);

	assert_eq!(map.remove(TestKey::Second), Some(20));
	assert_eq!(map.len(), 2);
	assert_eq!(map.get(TestKey::Second), None);
	assert_eq!(map.get(TestKey::First), Some(&10));
	assert_eq!(map.get(TestKey::Third), Some(&30));

	// Remove non-existent key
	assert_eq!(map.remove(TestKey::Second), None);
}

#[test]
fn test_get_mut() {
	let mut map = SparseMap::new();

	map.insert(TestKey::First, 10);

	if let Some(value) = map.get_mut(TestKey::First) {
		*value = 20;
	}

	assert_eq!(map.get(TestKey::First), Some(&20));
}

#[test]
fn test_contains_key() {
	let mut map = SparseMap::new();

	map.insert(TestKey::First, 10);

	assert!(map.contains_key(TestKey::First));
	assert!(!map.contains_key(TestKey::Second));
}

#[test]
fn test_clear() {
	let mut map = SparseMap::new();

	map.insert(TestKey::First, 10);
	map.insert(TestKey::Second, 20);

	map.clear();

	assert!(map.is_empty());
	assert_eq!(map.len(), 0);
	assert_eq!(map.get(TestKey::First), None);
}

#[test]
fn test_sparse_indices() {
	let mut map = SparseMap::new();

	// Insert with a large gap
	map.insert(TestKey::First, "first");
	map.insert(TestKey::Far, "far");

	assert_eq!(map.len(), 2);
	assert_eq!(map.get(TestKey::First), Some(&"first"));
	assert_eq!(map.get(TestKey::Far), Some(&"far"));
	assert_eq!(map.get(TestKey::Second), None);
	assert_eq!(map.get(TestKey::Third), None);
}

#[test]
fn test_iter() {
	let mut map = SparseMap::new();

	map.insert(TestKey::Third, 30);
	map.insert(TestKey::First, 10);
	map.insert(TestKey::Second, 20);

	let mut items: Vec<_> = map.iter().collect();
	items.sort_by_key(|(k, _)| *k as usize);

	assert_eq!(items, vec![(TestKey::First, &10), (TestKey::Second, &20), (TestKey::Third, &30),]);
}

#[test]
fn test_iter_mut() {
	let mut map = SparseMap::new();

	map.insert(TestKey::First, 10);
	map.insert(TestKey::Second, 20);

	for (_, value) in &mut map {
		*value *= 2;
	}

	assert_eq!(map.get(TestKey::First), Some(&20));
	assert_eq!(map.get(TestKey::Second), Some(&40));
}

#[test]
fn test_keys_values() {
	let mut map = SparseMap::new();

	map.insert(TestKey::First, 10);
	map.insert(TestKey::Third, 30);

	let keys: Vec<_> = map.keys().collect();
	let values: Vec<_> = map.values().copied().collect();

	assert!(keys.contains(&TestKey::First));
	assert!(keys.contains(&TestKey::Third));
	assert_eq!(keys.len(), 2);

	assert!(values.contains(&10));
	assert!(values.contains(&30));
	assert_eq!(values.len(), 2);
}

#[test]
fn test_from_iter() {
	let pairs = vec![(TestKey::First, 10), (TestKey::Second, 20), (TestKey::Third, 30)];

	let map: SparseMap<_, _> = pairs.into_iter().collect();

	assert_eq!(map.len(), 3);
	assert_eq!(map.get(TestKey::First), Some(&10));
	assert_eq!(map.get(TestKey::Second), Some(&20));
	assert_eq!(map.get(TestKey::Third), Some(&30));
}

#[test]
fn test_extend() {
	let mut map = SparseMap::new();
	map.insert(TestKey::First, 10);

	let more = vec![(TestKey::Second, 20), (TestKey::Third, 30)];

	map.extend(more);

	assert_eq!(map.len(), 3);
	assert_eq!(map.get(TestKey::Second), Some(&20));
}

#[test]
fn test_retain() {
	let mut map = SparseMap::new();

	map.insert(TestKey::First, 10);
	map.insert(TestKey::Second, 20);
	map.insert(TestKey::Third, 30);
	map.insert(TestKey::Far, 100);

	// Keep only even values
	map.retain(|_, v| *v % 20 == 0);

	assert_eq!(map.len(), 2);
	assert_eq!(map.get(TestKey::First), None);
	assert_eq!(map.get(TestKey::Second), Some(&20));
	assert_eq!(map.get(TestKey::Third), None);
	assert_eq!(map.get(TestKey::Far), Some(&100));
}

#[test]
fn test_index_trait() {
	let mut map = SparseMap::new();
	map.insert(TestKey::First, 10);

	assert_eq!(map[TestKey::First], 10);
}

#[test]
#[should_panic(expected = "key not found")]
fn test_index_trait_panic() {
	let map: SparseMap<TestKey, i32> = SparseMap::new();
	let _ = map[TestKey::First];
}

#[test]
fn test_eq() {
	let mut map1 = SparseMap::new();
	let mut map2 = SparseMap::new();

	map1.insert(TestKey::First, 10);
	map1.insert(TestKey::Second, 20);

	map2.insert(TestKey::First, 10);
	map2.insert(TestKey::Second, 20);

	assert_eq!(map1, map2);

	map2.insert(TestKey::Third, 30);
	assert_ne!(map1, map2);
}

#[test]
fn test_with_usize_keys() {
	let mut map: SparseMap<usize, &str> = SparseMap::new();

	map.insert(0, "zero");
	map.insert(5, "five");
	map.insert(1000, "thousand");

	assert_eq!(map.get(0), Some(&"zero"));
	assert_eq!(map.get(5), Some(&"five"));
	assert_eq!(map.get(1000), Some(&"thousand"));
	assert_eq!(map.get(500), None);
}

#[test]
fn test_large_map() {
	let mut map = SparseMap::new();

	// Insert many items
	for i in 0..1000usize {
		if i.is_multiple_of(3) {
			map.insert(i, i * 2);
		}
	}

	assert_eq!(map.len(), 334); // 0, 3, 6, ..., 999

	// Verify some values
	assert_eq!(map.get(0), Some(&0));
	assert_eq!(map.get(3), Some(&6));
	assert_eq!(map.get(999), Some(&1998));
	assert_eq!(map.get(1), None);
	assert_eq!(map.get(998), None);
}
