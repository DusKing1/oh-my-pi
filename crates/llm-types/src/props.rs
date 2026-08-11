use std::collections::BTreeMap;

use omp_core::{Str, StrMut};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Deterministic, namespaced extension properties shared by requests and
/// outcomes.
///
/// Keys use `namespace/name` form, such as `openai/verbosity` or `x/my-ext`.
/// Unknown keys are ignored by providers that cannot honor them, but never
/// silently: they produce an [`Unsupported`](crate::Unsupported) report. JSON
/// integers remain exact through protobuf conversion, including unsigned values
/// above `i64::MAX`; this prevents seeds, token budgets, and ids above 2^53
/// from being silently rounded through an IEEE-754 double. Once a property
/// proves portable across providers, it graduates to a typed field in a later
/// schema revision.
#[non_exhaustive]
#[repr(transparent)]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Props(
	/// Deterministically ordered namespaced keys and their JSON values.
	pub BTreeMap<Str, Value>,
);

impl Props {
	/// Returns a property addressed by its namespace and local name.
	#[must_use]
	pub fn get_ns(&self, namespace: &str, name: &str) -> Option<&Value> {
		let mut key = StrMut::with_capacity(namespace.len() + name.len() + 1);
		key.push_str(namespace);
		key.push_str("/");
		key.push_str(name);
		self.0.get(key.as_str())
	}

	/// Inserts a property under a namespaced key, returning the value previously
	/// stored there.
	pub fn insert_ns(&mut self, namespace: &str, name: &str, value: Value) -> Option<Value> {
		let mut key = StrMut::with_capacity(namespace.len() + name.len() + 1);
		key.push_str(namespace);
		key.push_str("/");
		key.push_str(name);
		self.0.insert(key.freeze(), value)
	}

	/// Returns whether no extension properties are present.
	#[must_use]
	pub fn is_empty(&self) -> bool {
		self.0.is_empty()
	}
}
