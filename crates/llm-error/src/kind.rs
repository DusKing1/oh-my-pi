//! Failure-kind bitset.
//!
//! Kinds are deliberately NOT mutually exclusive: a single provider error
//! legitimately carries several truths at once (a 429 that is also an
//! account-scoped usage limit, a timeout that is also transient). Every
//! attempt to collapse this into one enum in prior systems recreated
//! misclassification bugs, so the set is the primitive and predicates are
//! derived views.

/// A single classified failure kind. See [`Kinds`] for the set type.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum Kind {
	/// Worth retrying the same request against the same route.
	Transient          = 1 << 0,
	/// Request or stream deadline elapsed.
	Timeout            = 1 << 1,
	/// Socket/DNS/TLS/HTTP2 transport failure, not a provider verdict.
	Transport          = 1 << 2,
	/// Provider-side 5xx / internal error.
	ServerError        = 1 << 3,
	/// Short-window throttle (per-minute / burst). Back off, keep credential.
	RateThrottle       = 1 << 4,
	/// Concurrency cap. Shed briefly; rotating credentials would burn healthy
	/// siblings on a cap that clears in seconds.
	ConcurrencyCap     = 1 << 5,
	/// Model/fleet capacity exhausted (overloaded, `no_capacity`, 529).
	ModelCapacity      = 1 << 6,
	/// Account/plan usage or quota exhausted. Rotation-worthy.
	UsageLimit         = 1 << 7,
	/// Categorical billing signal (402, insufficient balance/credits).
	Billing            = 1 << 8,
	/// Credential rejected (401/403/invalid key).
	AuthFailed         = 1 << 9,
	/// OAuth grant is definitively dead (`invalid_grant`, revoked, refresh
	/// token expired) — re-login required, refresh will not help.
	OAuthExpired       = 1 << 10,
	/// Account-scoped policy denial (e.g. Codex `cyber_policy`). Terminal for
	/// this credential, but a sibling account may be entitled.
	AccountPolicy      = 1 << 11,
	/// Content filter / refusal verdict on the request or output.
	ContentBlocked     = 1 << 12,
	/// Prompt exceeds the model context window.
	ContextOverflow    = 1 << 13,
	/// Request body too large at the transport/gateway layer (413 family).
	RequestTooLarge    = 1 << 14,
	/// Provider rejected the request shape (`invalid_request_error` family).
	InvalidRequest     = 1 << 15,
	/// A specific request feature is unsupported; see
	/// [`Classification::feature`](crate::Classification::feature).
	FeatureUnsupported = 1 << 16,
	/// Model id unknown, decommissioned, or not entitled for this account.
	InvalidModel       = 1 << 17,
	/// Server-held conversation state is stale (`OpenAI` Responses
	/// `previous_response_id` not found / expired). Replay without the
	/// stale reference.
	StaleSessionItem   = 1 << 18,
	/// Model emitted an unparseable/malformed tool call.
	MalformedToolCall  = 1 << 19,
	/// Detected reasoning/output repetition loop; discard and re-sample.
	ThinkingLoop       = 1 << 20,
	/// Stream ended mid-payload (truncated JSON, disconnect before terminal
	/// event).
	StreamCorruption   = 1 << 21,
	/// Stream events arrived out of protocol order (safe to retry, unlike
	/// arbitrary corruption).
	StreamOrder        = 1 << 22,
	/// Nominal success carrying no usable content.
	EmptyResponse      = 1 << 23,
	/// Provider reported an error finish reason without an error envelope.
	ProviderFinish     = 1 << 24,
	/// Deliberate cancellation (caller abort).
	Aborted            = 1 << 25,
	/// Guard bit: the failure is deterministic despite a retryable-looking
	/// surface (e.g. llama.cpp malformed tool JSON behind HTTP 500).
	/// Strips same-request retryability.
	Deterministic      = 1 << 26,
}

/// Set of [`Kind`]s packed into a `u32`.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct Kinds(u32);

impl Kinds {
	/// The empty set (unclassified).
	pub const EMPTY: Self = Self(0);
	/// Kinds whose recovery requires mutating the request, the session, or
	/// the credential - re-sending the exact same bytes fails identically
	/// (or worse, spins on a dead credential). An identical-replay layer
	/// MUST refuse these; repair-first lanes own them.
	pub const EXACT_REPLAY_BARS: Self = Self(
		Kind::Deterministic as u32
			| Kind::ContentBlocked as u32
			| Kind::Aborted as u32
			| Kind::StaleSessionItem as u32
			| Kind::FeatureUnsupported as u32
			| Kind::ContextOverflow as u32
			| Kind::RequestTooLarge as u32
			| Kind::InvalidRequest as u32
			| Kind::InvalidModel as u32
			| Kind::AuthFailed as u32
			| Kind::OAuthExpired as u32
			| Kind::AccountPolicy as u32
			| Kind::UsageLimit as u32
			| Kind::Billing as u32,
	);
	/// Kinds for which SOME retry lane exists (same route, after repair, or
	/// after credential work). Gate exact replays with
	/// [`EXACT_REPLAY_BARS`](Self::EXACT_REPLAY_BARS) on top.
	pub const RETRYABLE: Self = Self(
		Kind::Transient as u32
			| Kind::Timeout as u32
			| Kind::Transport as u32
			| Kind::ServerError as u32
			| Kind::RateThrottle as u32
			| Kind::ConcurrencyCap as u32
			| Kind::ModelCapacity as u32
			| Kind::StaleSessionItem as u32
			| Kind::MalformedToolCall as u32
			| Kind::ThinkingLoop as u32
			| Kind::StreamOrder as u32
			| Kind::StreamCorruption as u32
			| Kind::EmptyResponse as u32
			| Kind::ProviderFinish as u32,
	);

	/// Set containing exactly `kind`.
	#[inline]
	pub const fn only(kind: Kind) -> Self {
		Self(kind as u32)
	}

	/// Whether `kind` is present.
	#[inline]
	pub const fn has(self, kind: Kind) -> bool {
		self.0 & kind as u32 != 0
	}

	/// Whether any kind in `other` is present.
	#[inline]
	pub const fn intersects(self, other: Self) -> bool {
		self.0 & other.0 != 0
	}

	/// Whether the set is empty (nothing matched).
	#[inline]
	pub const fn is_empty(self) -> bool {
		self.0 == 0
	}

	/// Inserts `kind` in place.
	#[inline]
	pub const fn insert(&mut self, kind: Kind) {
		self.0 |= kind as u32;
	}

	/// Removes `kind` in place.
	#[inline]
	pub const fn remove(&mut self, kind: Kind) {
		self.0 &= !(kind as u32);
	}

	/// Union of two sets.
	#[inline]
	pub const fn union(self, other: Self) -> Self {
		Self(self.0 | other.0)
	}

	/// Raw bit representation (stable across a process, not a wire format).
	#[inline]
	pub const fn bits(self) -> u32 {
		self.0
	}
}

impl core::ops::BitOr<Kind> for Kinds {
	type Output = Self;

	#[inline]
	fn bitor(self, rhs: Kind) -> Self {
		Self(self.0 | rhs as u32)
	}
}

impl core::ops::BitOr for Kind {
	type Output = Kinds;

	#[inline]
	fn bitor(self, rhs: Self) -> Kinds {
		Kinds(self as u32 | rhs as u32)
	}
}

impl core::ops::BitOrAssign<Kind> for Kinds {
	#[inline]
	fn bitor_assign(&mut self, rhs: Kind) {
		self.insert(rhs);
	}
}

impl core::ops::BitOrAssign for Kinds {
	#[inline]
	fn bitor_assign(&mut self, rhs: Self) {
		self.0 |= rhs.0;
	}
}

impl From<Kind> for Kinds {
	#[inline]
	fn from(kind: Kind) -> Self {
		Self::only(kind)
	}
}

impl core::fmt::Debug for Kinds {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		const LABELS: &[(Kind, &str)] = &[
			(Kind::Transient, "transient"),
			(Kind::Timeout, "timeout"),
			(Kind::Transport, "transport"),
			(Kind::ServerError, "server-error"),
			(Kind::RateThrottle, "rate-throttle"),
			(Kind::ConcurrencyCap, "concurrency-cap"),
			(Kind::ModelCapacity, "model-capacity"),
			(Kind::UsageLimit, "usage-limit"),
			(Kind::Billing, "billing"),
			(Kind::AuthFailed, "auth-failed"),
			(Kind::OAuthExpired, "oauth-expired"),
			(Kind::AccountPolicy, "account-policy"),
			(Kind::ContentBlocked, "content-blocked"),
			(Kind::ContextOverflow, "context-overflow"),
			(Kind::RequestTooLarge, "request-too-large"),
			(Kind::InvalidRequest, "invalid-request"),
			(Kind::FeatureUnsupported, "feature-unsupported"),
			(Kind::InvalidModel, "invalid-model"),
			(Kind::StaleSessionItem, "stale-session-item"),
			(Kind::MalformedToolCall, "malformed-tool-call"),
			(Kind::ThinkingLoop, "thinking-loop"),
			(Kind::StreamCorruption, "stream-corruption"),
			(Kind::StreamOrder, "stream-order"),
			(Kind::EmptyResponse, "empty-response"),
			(Kind::ProviderFinish, "provider-finish"),
			(Kind::Aborted, "aborted"),
			(Kind::Deterministic, "deterministic"),
		];
		let mut first = true;
		f.write_str("Kinds(")?;
		for &(kind, label) in LABELS {
			if self.has(kind) {
				if !first {
					f.write_str("|")?;
				}
				f.write_str(label)?;
				first = false;
			}
		}
		if first {
			f.write_str("none")?;
		}
		f.write_str(")")
	}
}
