//! Strongly typed catalog identifiers.

use std::{borrow::Borrow, fmt, ops::Deref};

use omp_core::Str;
use serde::{Deserialize, Serialize};

macro_rules! string_id {
	($(#[$meta:meta])* $name:ident) => {
		$(#[$meta])*
		#[derive(Clone, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
		#[repr(transparent)]
		#[serde(transparent)]
		pub struct $name(Str);

		impl $name {
			/// Creates an identifier from stored catalog text.
			#[inline]
			pub fn new(value: impl Into<Str>) -> Self {
				Self(value.into())
			}

			/// Borrows the identifier as text.
			#[inline]
			pub fn as_str(&self) -> &str {
				self.0.as_str()
			}

			/// Returns the underlying allocation-conscious string.
			#[inline]
			pub fn into_inner(self) -> Str {
				self.0
			}
		}

		impl AsRef<str> for $name {
			#[inline]
			fn as_ref(&self) -> &str {
				self.as_str()
			}
		}

		impl Borrow<str> for $name {
			#[inline]
			fn borrow(&self) -> &str {
				self.as_str()
			}
		}

		impl Deref for $name {
			type Target = str;

			#[inline]
			fn deref(&self) -> &Self::Target {
				self.as_str()
			}
		}

		impl fmt::Display for $name {
			fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
				formatter.write_str(self.as_str())
			}
		}

		impl fmt::Debug for $name {
			fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
				fmt::Debug::fmt(&self.0, formatter)
			}
		}

		impl From<Str> for $name {
			#[inline]
			fn from(value: Str) -> Self {
				Self(value)
			}
		}

		impl From<&str> for $name {
			#[inline]
			fn from(value: &str) -> Self {
				Self(Str::from(value))
			}
		}

		impl From<String> for $name {
			#[inline]
			fn from(value: String) -> Self {
				Self(Str::from(value))
			}
		}

		impl From<$name> for Str {
			#[inline]
			fn from(value: $name) -> Self {
				value.0
			}
		}

		impl PartialEq<str> for $name {
			#[inline]
			fn eq(&self, other: &str) -> bool {
				self.as_str() == other
			}
		}

		impl PartialEq<&str> for $name {
			#[inline]
			fn eq(&self, other: &&str) -> bool {
				self.as_str() == *other
			}
		}
	};
}

string_id!(/// Identifies a commercial, hosted, or local provider domain.
	ProviderId);
string_id!(/// Identifies one concrete provider route.
	RouteId);
string_id!(/// Identifies one wire codec implementation.
	CodecId);
string_id!(/// Identifies one normalized selectable model deployment.
	ModelKey);
string_id!(/// Identifies a normalized model family.
	FamilyId);
string_id!(/// Identifies an interned authentication specification.
	AuthSpecId);
string_id!(/// Identifies an interned public OAuth flow specification.
	OAuthSpecId);
string_id!(/// Identifies an interned static header profile.
	HeaderProfileId);
string_id!(/// Identifies an interned model-discovery specification.
	DiscoverySpecId);
string_id!(/// Identifies an interned wire-lowering policy.
	WirePolicyId);
string_id!(/// Identifies an interned reasoning policy.
	ThinkingPolicyId);
string_id!(/// Carries the opaque model identifier expected by a wire endpoint.
	WireModelId);
string_id!(/// Identifies an immutable catalog revision.
	CatalogRevision);
