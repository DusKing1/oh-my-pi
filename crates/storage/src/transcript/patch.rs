//! Tri-state updates for inference configuration.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A field update that distinguishes omission from explicit clearing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Patch<T> {
	/// Leave the previous value unchanged.
	#[default]
	Unchanged,
	/// Set the field to a new value.
	Set(T),
	/// Explicitly clear the previous value.
	Clear,
}

impl<T> Patch<T> {
	/// Returns whether this patch leaves the field unchanged.
	#[must_use]
	pub const fn is_unchanged(&self) -> bool {
		matches!(self, Self::Unchanged)
	}

	/// Borrows the value held by a set patch.
	#[must_use]
	pub const fn as_ref(&self) -> Patch<&T> {
		match self {
			Self::Unchanged => Patch::Unchanged,
			Self::Set(value) => Patch::Set(value),
			Self::Clear => Patch::Clear,
		}
	}

	/// Maps the value held by a set patch without changing its state.
	pub fn map<U>(self, map: impl FnOnce(T) -> U) -> Patch<U> {
		match self {
			Self::Unchanged => Patch::Unchanged,
			Self::Set(value) => Patch::Set(map(value)),
			Self::Clear => Patch::Clear,
		}
	}

	/// Converts to the nested-option representation used by serde fields.
	///
	/// `None` is unchanged, `Some(None)` is clear, and `Some(Some(value))` is
	/// set.
	pub fn into_nested_option(self) -> Option<Option<T>> {
		match self {
			Self::Unchanged => None,
			Self::Set(value) => Some(Some(value)),
			Self::Clear => Some(None),
		}
	}

	/// Converts from the nested-option representation used by serde fields.
	#[must_use]
	pub fn from_nested_option(value: Option<Option<T>>) -> Self {
		match value {
			None => Self::Unchanged,
			Some(Some(value)) => Self::Set(value),
			Some(None) => Self::Clear,
		}
	}
}

impl<T> Serialize for Patch<T>
where
	T: Serialize,
{
	fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
	where
		S: Serializer,
	{
		match self {
			Self::Set(value) => serializer.serialize_some(value),
			Self::Unchanged | Self::Clear => serializer.serialize_none(),
		}
	}
}

impl<'de, T> Deserialize<'de> for Patch<T>
where
	T: Deserialize<'de>,
{
	fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
	where
		D: Deserializer<'de>,
	{
		Option::<T>::deserialize(deserializer).map(|value| match value {
			Some(value) => Self::Set(value),
			None => Self::Clear,
		})
	}
}

#[cfg(test)]
mod tests {
	use serde::{Deserialize, Serialize};

	use super::Patch;

	#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
	struct Holder {
		#[serde(default, skip_serializing_if = "Patch::is_unchanged")]
		value: Patch<u64>,
	}

	#[test]
	fn field_states_round_trip() {
		let cases = [
			(r"{}", Patch::Unchanged),
			(r#"{"value":null}"#, Patch::Clear),
			(r#"{"value":7}"#, Patch::Set(7)),
		];
		for (json, expected) in cases {
			let decoded: Holder = serde_json::from_str(json).expect("valid patch field");
			assert_eq!(decoded.value, expected);
			let encoded = serde_json::to_string(&decoded).expect("serializable patch field");
			let round_trip: Holder = serde_json::from_str(&encoded).expect("encoded patch field");
			assert_eq!(round_trip, decoded);
		}
	}

	#[test]
	fn unchanged_is_an_absent_key() {
		let encoded = serde_json::to_string(&Holder { value: Patch::Unchanged })
			.expect("serializable patch field");
		assert_eq!(encoded, "{}");
	}
}
