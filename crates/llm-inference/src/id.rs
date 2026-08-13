//! Strongly typed identifiers for runtime inference state.

use std::{borrow::Borrow, fmt, ops::Deref};

use omp_core::Str;
use serde::{Deserialize, Serialize};

macro_rules! runtime_id {
	($(#[$meta:meta])* $name:ident) => {
		$(#[$meta])*
		#[derive(Clone, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
		#[repr(transparent)]
		#[serde(transparent)]
		pub struct $name(Str);

		impl $name {
			/// Creates an identifier from stored text.
			#[inline]
			pub fn new(value: impl Into<Str>) -> Self {
				Self(value.into())
			}

			/// Borrows the identifier as text.
			#[inline]
			pub fn as_str(&self) -> &str {
				self.0.as_str()
			}

			/// Returns the allocation-conscious stored string.
			#[inline]
			pub fn into_inner(self) -> Str {
				self.0
			}
		}

		impl AsRef<str> for $name {
			fn as_ref(&self) -> &str { self.as_str() }
		}

		impl Borrow<str> for $name {
			fn borrow(&self) -> &str { self.as_str() }
		}

		impl Deref for $name {
			type Target = str;
			fn deref(&self) -> &Self::Target { self.as_str() }
		}

		impl fmt::Display for $name {
			fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
				formatter.write_str(self.as_str())
			}
		}

		impl fmt::Debug for $name {
			fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
				self.0.fmt(formatter)
			}
		}

		impl From<Str> for $name {
			fn from(value: Str) -> Self { Self(value) }
		}

		impl From<&str> for $name {
			fn from(value: &str) -> Self { Self(Str::from(value)) }
		}

		impl From<String> for $name {
			fn from(value: String) -> Self { Self(Str::from(value)) }
		}
	};
}

runtime_id!(/// Identifies one logical inference request across all attempts.
	RequestId);
runtime_id!(/// Identifies a credential-bearing account without exposing its secret.
	AccountId);
runtime_id!(/// Identifies the authenticated principal that owns account affinity.
	PrincipalId);
runtime_id!(/// Identifies a cloud or account-scoped project.
	ProjectId);
runtime_id!(/// Identifies an account tenant.
	TenantId);
runtime_id!(/// Identifies an account organization.
	OrganizationId);
runtime_id!(/// Identifies a routing or billing region.
	RegionId);
runtime_id!(/// Identifies an append-only conversation.
	ConversationId);
runtime_id!(/// Identifies an immutable committed conversation revision.
	Revision);
runtime_id!(/// Identifies an idempotent conversation turn.
	TurnId);
runtime_id!(/// Identifies a canonical tool call.
	ToolCallId);
runtime_id!(/// Identifies a resumable media-generation job.
	GenerationHandle);
runtime_id!(/// Identifies an interactive authentication session.
	LoginSessionId);
