//! Canonical identifiers and provider-wire tool-call ID projection.

use std::{collections::BTreeMap, fmt, str::FromStr};

use omp_core::Str;
use parking_lot::Mutex;
use ulid::{DecodeError, Ulid};

macro_rules! ulid_id {
	($name:ident, $description:literal) => {
		#[doc = $description]
		#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
		pub struct $name(Ulid);

		impl $name {
			/// Mints a fresh, time-sortable identifier.
			#[must_use]
			pub fn new() -> Self {
				Self(Ulid::generate())
			}

			/// Wraps an already validated ULID.
			#[must_use]
			pub const fn from_ulid(value: Ulid) -> Self {
				Self(value)
			}

			/// Returns the underlying ULID.
			#[must_use]
			pub const fn as_ulid(self) -> Ulid {
				self.0
			}
		}

		impl Default for $name {
			fn default() -> Self {
				Self::new()
			}
		}

		impl fmt::Display for $name {
			fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
				self.0.fmt(f)
			}
		}

		impl FromStr for $name {
			type Err = DecodeError;

			fn from_str(value: &str) -> Result<Self, Self::Err> {
				value.parse().map(Self)
			}
		}
	};
}

ulid_id!(TurnId, "A client-minted idempotency key for one logical model turn.");
ulid_id!(ContextId, "A client-minted identifier for a retained conversation context.");
ulid_id!(CallId, "The canonical identifier of a transcript tool call.");
ulid_id!(InvocationId, "The correlation identifier of an in-turn invocation.");

/// Transport/dialect rules for projecting canonical tool-call identifiers.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum ToolCallIdProfile {
	/// Preserve provider-originated identifiers and canonical ULIDs verbatim.
	#[default]
	Preserve,
	/// Emit OpenAI-style identifiers beginning with `call_`.
	OpenAi,
	/// Emit Anthropic-style identifiers beginning with `toolu_`.
	Anthropic,
	/// Emit exactly nine ASCII alphanumeric characters for Mistral.
	Mistral9,
}

/// A turn-local, bidirectional projection between canonical and wire call IDs.
///
/// Codecs keep one mapper for a turn. The reverse table is essential:
/// truncating an identifier is not reversible, and merely taking the first nine
/// ULID characters would make every call minted in the same time window
/// collide. Collision probing makes every live mapping unique within the
/// provider's finite namespace.
#[derive(Debug, Default)]
pub struct CallIdMapper {
	state: Mutex<MappingState>,
}

#[derive(Debug, Default)]
struct MappingState {
	forward:          BTreeMap<(ToolCallIdProfile, CallId), Str>,
	observed_forward: BTreeMap<CallId, Str>,
	reverse:          BTreeMap<Str, CallId>,
}

impl CallIdMapper {
	/// Creates an empty turn-local mapping.
	#[must_use]
	pub fn new() -> Self {
		Self::default()
	}

	/// Projects a canonical ID to a provider-safe wire ID and remembers it.
	///
	/// Repeated calls for the same canonical ID and profile return the same
	/// value. Distinct IDs are collision-resolved rather than silently aliased.
	/// Provider-originated IDs remain byte-for-byte identical only under
	/// [`ToolCallIdProfile::Preserve`].
	#[must_use]
	pub fn to_wire(&self, id: &CallId, profile: ToolCallIdProfile) -> Str {
		let mut state = self.state.lock();
		if profile == ToolCallIdProfile::Preserve
			&& let Some(wire) = state.observed_forward.get(id)
		{
			return wire.clone();
		}
		if let Some(wire) = state.forward.get(&(profile, *id)) {
			return wire.clone();
		}

		let canonical = id.to_string();
		let base = match profile {
			ToolCallIdProfile::Preserve => canonical,
			ToolCallIdProfile::OpenAi => format!("call_{canonical}"),
			ToolCallIdProfile::Anthropic => format!("toolu_{canonical}"),
			ToolCallIdProfile::Mistral9 => canonical[..9].to_owned(),
		};
		let mut wire = Str::from(base.as_str());
		if let Some(existing) = state.reverse.get(&wire)
			&& existing != id
		{
			wire = unique_wire(&state, profile, &base);
		}

		state.forward.insert((profile, *id), wire.clone());
		state.reverse.insert(wire.clone(), *id);
		wire
	}

	/// Registers a provider-originated wire ID and returns its canonical ID.
	///
	/// Repeated sightings of one wire ID return the same canonical ID. A newly
	/// observed ID is assigned once and recorded in both directions for the
	/// lifetime of this turn-local mapper.
	#[must_use]
	pub fn observe(&self, wire: &str) -> CallId {
		let mut state = self.state.lock();
		let wire = Str::from(wire);
		if let Some(id) = state.reverse.get(&wire) {
			return *id;
		}

		let id = CallId::new();
		state.observed_forward.insert(id, wire.clone());
		state.reverse.insert(wire, id);
		id
	}

	/// Restores the canonical ID previously assigned to a provider wire ID.
	///
	/// Returns `None` for an ID not emitted by this mapper. Codecs must not
	/// invent a canonical call identity when the provider violates pairing.
	#[must_use]
	pub fn from_wire(&self, wire: &str, _profile: ToolCallIdProfile) -> Option<CallId> {
		let state = self.state.lock();
		state.reverse.get(wire).copied()
	}
}

fn unique_wire(state: &MappingState, profile: ToolCallIdProfile, base: &str) -> Str {
	let width = match profile {
		ToolCallIdProfile::Mistral9 => 9,
		ToolCallIdProfile::OpenAi => 40,
		ToolCallIdProfile::Anthropic => 64,
		ToolCallIdProfile::Preserve => base.len(),
	};
	for nonce in 1_u64.. {
		let suffix = base62(nonce);
		let keep = width.saturating_sub(suffix.len());
		let prefix = &base[..keep.min(base.len())];
		let candidate = Str::from(format!("{prefix}{suffix}"));
		if !state.reverse.contains_key(&candidate) {
			return candidate;
		}
	}
	unreachable!("the finite wire namespace cannot be exhausted by a Rust map")
}

fn base62(mut value: u64) -> String {
	const DIGITS: &[u8; 62] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
	let mut encoded = [0_u8; 11];
	let mut cursor = encoded.len();
	loop {
		cursor -= 1;
		encoded[cursor] = DIGITS[(value % 62) as usize];
		value /= 62;
		if value == 0 {
			break;
		}
	}
	String::from_utf8(encoded[cursor..].to_vec()).expect("base62 alphabet is UTF-8")
}

#[cfg(test)]
mod tests {
	use super::{CallId, CallIdMapper, ToolCallIdProfile};

	const CANONICAL: &str = "01ARZ3NDEKTSV4RRFFQ69G5FAV";

	#[test]
	fn every_profile_is_deterministic_and_round_trips() {
		let mapper = CallIdMapper::new();
		let id: CallId = CANONICAL.parse().unwrap();
		let cases = [
			(ToolCallIdProfile::Preserve, CANONICAL),
			(ToolCallIdProfile::OpenAi, "call_01ARZ3NDEKTSV4RRFFQ69G5FAV"),
			(ToolCallIdProfile::Anthropic, "toolu_01ARZ3NDEKTSV4RRFFQ69G5FAV"),
			(ToolCallIdProfile::Mistral9, "01ARZ3NDE"),
		];
		for (profile, expected) in cases {
			let wire = mapper.to_wire(&id, profile);
			assert_eq!(wire, expected);
			assert_eq!(mapper.to_wire(&id, profile), wire);
			assert_eq!(mapper.from_wire(&wire, profile), Some(id));
		}
	}

	#[test]
	fn colliding_truncations_are_disambiguated() {
		let mapper = CallIdMapper::new();
		let first: CallId = CANONICAL.parse().unwrap();
		let second: CallId = "01ARZ3NDEKTSV4RRFFQ69G5FAW".parse().unwrap();
		let first_wire = mapper.to_wire(&first, ToolCallIdProfile::Mistral9);
		let second_wire = mapper.to_wire(&second, ToolCallIdProfile::Mistral9);
		assert_ne!(first_wire, second_wire);
		assert_eq!(mapper.from_wire(&first_wire, ToolCallIdProfile::Mistral9), Some(first));
		assert_eq!(mapper.from_wire(&second_wire, ToolCallIdProfile::Mistral9), Some(second));
	}

	#[test]
	fn preserve_profile_leaves_foreign_ids_verbatim() {
		let mapper = CallIdMapper::new();
		let foreign = "vendor|call+/=with punctuation";
		let id = mapper.observe(foreign);
		assert_eq!(mapper.to_wire(&id, ToolCallIdProfile::Preserve), foreign);
		assert_eq!(mapper.observe(foreign), id);
		assert_ne!(mapper.observe("another-foreign-id"), id);
	}

	#[test]
	fn profiled_foreign_ids_are_stable_and_prefixed() {
		let mapper = CallIdMapper::new();
		let id = mapper.observe("foreign");
		let openai = mapper.to_wire(&id, ToolCallIdProfile::OpenAi);
		let anthropic = mapper.to_wire(&id, ToolCallIdProfile::Anthropic);
		assert!(openai.starts_with("call_"));
		assert!(anthropic.starts_with("toolu_"));
		assert_eq!(mapper.to_wire(&id, ToolCallIdProfile::OpenAi), openai);
		assert_eq!(mapper.to_wire(&id, ToolCallIdProfile::Anthropic), anthropic);
	}
}
