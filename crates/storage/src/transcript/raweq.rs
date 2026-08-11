//! Byte-verbatim equality helpers for raw JSON values.

use std::collections::BTreeMap;

use omp_core::Str;
use serde_json::value::RawValue;

pub fn raw_eq(a: &RawValue, b: &RawValue) -> bool {
	a.get() == b.get()
}

pub fn opt_raw_eq(a: Option<&RawValue>, b: Option<&RawValue>) -> bool {
	match (a, b) {
		(Some(a), Some(b)) => raw_eq(a, b),
		(None, None) => true,
		_ => false,
	}
}

pub fn map_raw_eq(a: &BTreeMap<Str, Box<RawValue>>, b: &BTreeMap<Str, Box<RawValue>>) -> bool {
	a.len() == b.len()
		&& a
			.iter()
			.all(|(key, value)| b.get(key).is_some_and(|other| raw_eq(value, other)))
}

#[cfg(test)]
mod tests {
	use super::*;

	fn raw(value: &str) -> Box<RawValue> {
		RawValue::from_string(value.to_owned()).expect("test value is valid JSON")
	}

	#[test]
	fn equality_observes_raw_json_bytes() {
		let compact = raw(r#"{"a":1}"#);
		let same = raw(r#"{"a":1}"#);
		let spaced = raw(r#"{ "a": 1 }"#);
		assert!(raw_eq(&compact, &same));
		assert!(!raw_eq(&compact, &spaced));

		assert!(opt_raw_eq(Some(&*compact), Some(&*same)));
		assert!(!opt_raw_eq(Some(&*compact), Some(&*spaced)));
		assert!(!opt_raw_eq(Some(&*compact), None));

		let mut compact_map = BTreeMap::new();
		compact_map.insert(Str::new("field"), compact);
		let mut same_map = BTreeMap::new();
		same_map.insert(Str::new("field"), same);
		let mut spaced_map = BTreeMap::new();
		spaced_map.insert(Str::new("field"), spaced);
		assert!(map_raw_eq(&compact_map, &same_map));
		assert!(!map_raw_eq(&compact_map, &spaced_map));
	}
}
