use bon::Builder;
use bytes::Bytes;
use omp_core::Str;

use super::{Item, Props, Revision, ToolCall, ToolResult, Unsupported};

/// Handle on server-held context coupled to the caller's exact precondition.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, Eq, PartialEq)]
pub struct ContextRef {
	/// Client-minted ULID namespaced by the authenticated caller.
	pub context_id: Str,
	/// Revision that must exactly match before any mutation is considered.
	pub expected:   Revision,
}

/// Explicit, atomic conversation mutation: truncate first when requested, then
/// append.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, Default, PartialEq)]
pub struct ThreadDelta {
	/// Intentional rewind point; absence means pure append and stale revisions
	/// never imply it.
	pub truncate_to: Option<u64>,
	/// Entries appended after any explicit rewind.
	pub append:      Vec<Item>,
}

/// Token accounting for a completed inference operation.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, PartialEq)]
pub struct Usage {
	/// Prompt-side tokens billed or estimated.
	pub input_tokens:       u64,
	/// Generated tokens billed or estimated.
	pub output_tokens:      u64,
	/// Prompt tokens served from a provider cache.
	pub cache_read_tokens:  u64,
	/// Prompt tokens written into a provider cache.
	pub cache_write_tokens: u64,
	/// Provider-reported aggregate, which may include orchestration tokens.
	pub total_tokens:       Option<u64>,
	/// Provider-reported occupied context tokens.
	pub context_tokens:     Option<u64>,
	/// Provider-side orchestration token buckets.
	pub orchestration:      Option<OrchestrationUsage>,
	/// Provider premium-request accounting.
	pub premium_requests:   Option<u64>,
	/// Reasoning tokens included in output tokens.
	pub reasoning_tokens:   Option<u64>,
	/// Cache-write token buckets split by retention lifetime.
	pub cache_ttl:          Option<CacheTtlUsage>,
	/// Provider-hosted tool request counts.
	pub server_tools:       Option<ServerToolUsage>,
	/// Whether these counts come from an exact source or an estimator.
	pub accuracy:           Accuracy,
	/// Vendor-namespaced breakdown such as service tier or media billing units.
	pub detail:             Props,
}

/// Provider-side orchestration token buckets.
#[non_exhaustive]
#[derive(Builder, Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct OrchestrationUsage {
	/// Non-cached orchestration input.
	pub input_tokens:      Option<u64>,
	/// Cached orchestration input.
	pub cache_read_tokens: Option<u64>,
	/// Orchestration output.
	pub output_tokens:     Option<u64>,
}

/// Cache-write token buckets by lifetime.
#[non_exhaustive]
#[derive(Builder, Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct CacheTtlUsage {
	/// Tokens written with five-minute retention.
	pub ephemeral_5m_tokens: Option<u64>,
	/// Tokens written with one-hour retention.
	pub ephemeral_1h_tokens: Option<u64>,
}

/// Counts of provider-hosted tool requests.
#[non_exhaustive]
#[derive(Builder, Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct ServerToolUsage {
	/// Provider-hosted web searches.
	pub web_search_requests: Option<u64>,
	/// Provider-hosted web fetches.
	pub web_fetch_requests:  Option<u64>,
}

/// Provenance of a usage count.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Accuracy {
	/// Reported by an exact tokenizer or provider endpoint.
	Exact,
	/// Produced by a heuristic or approximate tokenizer.
	Estimated,
	/// Aggregate containing both exact and estimated counts.
	Mixed,
}

/// Monetary cost for a completed operation.
#[non_exhaustive]
#[derive(Builder, Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Cost {
	/// Cost in nano-US dollars, avoiding floating-point accounting drift.
	pub nanos_usd:             u64,
	/// Input-token cost when the provider or catalog exposes the split.
	pub input_nanos_usd:       Option<u64>,
	/// Output-token cost when the provider or catalog exposes the split.
	pub output_nanos_usd:      Option<u64>,
	/// Cache-read cost when the provider or catalog exposes the split.
	pub cache_read_nanos_usd:  Option<u64>,
	/// Cache-write cost when the provider or catalog exposes the split.
	pub cache_write_nanos_usd: Option<u64>,
	/// Whether catalog rates produced the cost rather than an exact in-band
	/// provider bill.
	pub estimated:             bool,
}

/// Why model generation stopped.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StopReason {
	/// The model naturally completed the turn.
	EndTurn,
	/// The model requested tool execution.
	ToolUse,
	/// The configured output limit was reached.
	MaxTokens,
	/// Provider safety policy filtered the output.
	ContentFilter,
}

/// Classified evidence from one attempted provider route.
///
/// Provider-specific response payloads remain in namespaced props; diagnostics
/// retain only portable fields needed to make deterministic recovery choices.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, Eq, PartialEq)]
pub struct Diagnostic {
	/// Provider selected for this attempt.
	pub provider:     Str,
	/// Model selected for this attempt.
	pub model:        Str,
	/// One-based attempt number, or zero when unavailable.
	pub attempt:      u32,
	/// Stable portable classification code.
	pub code:         Str,
	/// Human-readable classified detail safe to surface to callers.
	pub detail:       Str,
	/// Safe recovery lane for this attempt.
	pub retryability: Retryability,
}

/// The safe recovery lane for a classified provider attempt.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum Retryability {
	/// The source did not classify retryability.
	#[default]
	Unspecified,
	/// Repeating or adapting the attempt is not safe.
	Never,
	/// The same request may be repeated against the same route.
	SameRoute,
	/// The request may be retried after deterministic request or session repair.
	AfterRepair,
	/// The request may be retried after refreshing or rotating credentials.
	AfterCredential,
	/// The request may be retried after a provider-directed delay.
	AfterDelay,
}

/// Authoritative successful turn record committed atomically by the gateway.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, PartialEq)]
pub struct ChatOutcome {
	/// Canonical items committed this turn, including gateway-assigned sequence
	/// numbers.
	pub output:            Vec<Item>,
	/// Terminal generation reason.
	pub stop:              StopReason,
	/// Token accounting when supplied by the provider or metering layer.
	pub usage:             Option<Usage>,
	/// Monetary accounting when rates or in-band billing are available.
	pub cost:              Option<Cost>,
	/// Features changed or omitted by the resolved provider path.
	pub unsupported:       Vec<Unsupported>,
	/// Post-commit context revision, absent for stateless turns.
	pub revision:          Option<Revision>,
	/// Provider that actually served the turn after routing and fallback
	/// resolution.
	pub provider:          Str,
	/// Model that actually served the turn after alias and role resolution.
	pub model:             Str,
	/// Upstream selected by an aggregator, distinct from the configured
	/// provider route.
	pub upstream_provider: Option<Str>,
	/// Total request duration in milliseconds.
	pub duration_ms:       Option<u64>,
	/// Time to first output token in milliseconds.
	pub ttft_ms:           Option<u64>,
	/// Provider context accounting snapshot captured at send time.
	pub context_snapshot:  Option<ContextSnapshot>,
	/// Classified attempts retained in execution order.
	#[builder(default)]
	pub diagnostics:       Vec<Diagnostic>,
	/// Namespaced outcome detail retained without polluting portable fields.
	pub props:             Props,
}

/// Context accounting captured with an outcome.
#[non_exhaustive]
#[derive(Builder, Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ContextSnapshot {
	/// Authoritative provider prompt/input tokens.
	pub prompt_tokens:                  u64,
	/// Estimated non-message tokens present at send time.
	pub non_message_tokens:             u64,
	/// Estimated tokens removed by later local history rewrites.
	pub history_rewrite_tokens_removed: Option<u64>,
	/// Timestamp of the last message included in the snapshot.
	pub last_message_timestamp_ms:      Option<u64>,
}

/// A terminal protocol failure; context remains at its pre-turn revision.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, PartialEq)]
pub struct TurnError {
	/// Stable failure class shared by native and transport callers.
	pub kind:           TurnErrorKind,
	/// Classified diagnostic detail.
	pub detail:         Str,
	/// Actual server revision for a conflict.
	pub actual:         Option<Revision>,
	/// Unsupported features when a fail-closed fallback policy tripped.
	pub unsupported:    Vec<Unsupported>,
	/// Provider-directed delay for a rate limit, in milliseconds.
	pub retry_after_ms: u64,
	/// Classified attempted-route evidence retained in execution order.
	#[builder(default)]
	pub diagnostics:    Vec<Diagnostic>,
	/// Stable machine-readable error identity.
	pub error_id:       Option<u64>,
}

/// In-band terminal turn failure classes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TurnErrorKind {
	/// The optimistic-concurrency precondition did not match server state.
	Conflict,
	/// The held context is unknown or evicted and must be reseeded in full.
	NeedFull,
	/// A requested fail-closed feature was unavailable.
	Unsupported,
	/// No usable credential remained after resolution.
	Auth,
	/// The upstream asked the caller to retry after a delay.
	RateLimited,
	/// The provider failed after bounded retries.
	Upstream,
	/// The gateway shed the request under load.
	Overloaded,
	/// A live in-turn invocation missed its server-enforced answer deadline.
	InvokeTimeout,
}

/// One server-initiated operation addressed to the client's declared in-turn
/// executor.
///
/// Canonical call/result projection is presence-based. Vendor bytes are
/// reserved for controls that genuinely cannot be represented by the
/// transcript; executors never construct vendor protobuf.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, PartialEq)]
pub struct Invoke {
	/// Correlation id allowing multiple invocations to remain live concurrently.
	pub invocation_id: Str,
	/// Executor dispatch key matched against the capabilities declared by the
	/// client.
	pub name:          Str,
	/// Canonical transcript projection, absent only for pure control
	/// invocations.
	pub tool_call:     Option<ToolCall>,
	/// Pinned-transport control payload for an otherwise unprojectable variant.
	pub vendor:        Bytes,
	/// Server-enforced deadline for receiving [`InvokeComplete`].
	pub timeout_ms:    u64,
	/// Namespaced executor hints such as a transport execution id.
	pub props:         Props,
}

/// Incremental data sent by an executor while an invocation is live.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, Eq, PartialEq)]
pub struct InvokeInput {
	/// Correlation id of the live invocation.
	pub invocation_id: Str,
	/// Canonical output chunk or unprojectable pinned control payload.
	pub payload:       InvokePayload,
}

/// Payload carried by one incremental invocation frame.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InvokePayload {
	/// Portable streamed execution output.
	Chunk(InvokeChunk),
	/// Pinned-transport data with no canonical projection.
	Vendor(Bytes),
}

/// One channel-tagged execution output fragment.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, Eq, PartialEq)]
pub struct InvokeChunk {
	/// Logical output channel projected onto the pinned provider framing.
	pub channel: InvokeChannel,
	/// Fragment bytes moved directly through the transport.
	pub data:    Bytes,
}

/// Portable channels for streamed execution output.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum InvokeChannel {
	/// Normal process output.
	Stdout,
	/// Diagnostic process output.
	Stderr,
	/// Progress data that is neither stdout nor stderr.
	Progress,
}

/// Terminal client response to an in-turn invocation.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, PartialEq)]
pub struct InvokeComplete {
	/// Correlation id of the completed invocation.
	pub invocation_id: Str,
	/// Transcript projection committed with the originating tool call when both
	/// are present.
	pub tool_result:   Option<ToolResult>,
	/// Portable execution outcome used to synthesize transport completion
	/// framing.
	pub status:        Option<ExecStatus>,
	/// Unprojectable control completion payload.
	pub vendor:        Bytes,
	/// Namespaced details outside the portable execution status.
	pub props:         Props,
}

/// Portable description of how an execution-shaped invocation ended.
#[non_exhaustive]
#[derive(Builder, Clone, Debug, Eq, PartialEq)]
pub struct ExecStatus {
	/// Terminal execution class.
	pub outcome:                 ExecOutcome,
	/// Process exit code when the invocation ran to an exit.
	pub exit_code:               i32,
	/// Terminating signal name when the process was killed.
	pub signal:                  Str,
	/// Failure, rejection, or abort detail.
	pub reason:                  Str,
	/// Working directory echoed by pinned exit frames.
	pub cwd:                     Str,
	/// Whether pinned framing killed the process but still represented it as an
	/// exit.
	pub aborted:                 bool,
	/// Persistent location of full output when the executor spilled it.
	pub output_location:         Str,
	/// Executor-measured wall-clock duration.
	pub local_execution_time_ms: u64,
	/// Whether a rejection or denial came from a read-only environment.
	pub is_readonly:             bool,
	/// Command's own timeout, distinct from the invocation-answer deadline.
	pub command_timeout_ms:      u64,
}

/// All portable execution terminal outcomes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ExecOutcome {
	/// The process ran to completion; exit code and optional signal apply.
	Exited,
	/// The executor failed before or despite running the process.
	Failed,
	/// User or policy declined execution.
	Rejected,
	/// The environment denied permission.
	Denied,
	/// The command's own timeout elapsed.
	Timeout,
	/// The executor answered a server cancellation.
	Cancelled,
}

/// Kind of delta-only assistant part entering the stream.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StreamPartKind {
	/// Visible model text.
	Text,
	/// Model reasoning text.
	Thinking,
	/// Incremental tool-call arguments.
	ToolCall,
}

/// One event in a chat turn stream.
///
/// Part events carry deltas only. Callers that need running snapshots use the
/// stream accumulator; this avoids cloning the entire partial response for
/// every token.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum TurnEvent {
	/// The context precondition passed and the turn was admitted.
	Accepted {
		/// Whether this is replay of an already committed idempotent turn.
		replay: bool,
	},
	/// A visible retry after capability resolution or credential rotation.
	Attempt {
		/// One-based outbound attempt number.
		number: u32,
		/// Why the preceding attempt was abandoned.
		reason: Str,
	},
	/// Announces the kind and metadata of a new streamed part.
	PartStart {
		/// Stable part index used by subsequent deltas and the end marker.
		index:        u32,
		/// Content kind determining how delta bytes are interpreted.
		kind:         StreamPartKind,
		/// Canonical tool-call id for tool-call parts, empty otherwise.
		tool_call_id: Str,
		/// Tool dispatch name for tool-call parts, empty otherwise.
		tool_name:    Str,
	},
	/// Adds bytes to a previously announced part without an accumulated
	/// snapshot.
	PartDelta {
		/// Index of the part receiving the fragment.
		index: u32,
		/// UTF-8 text, thinking text, or JSON argument fragment according to part
		/// kind.
		chunk: Bytes,
	},
	/// Closes a streamed part and carries any provider-supplied signature.
	PartEnd {
		/// Index of the part being closed.
		index:     u32,
		/// Opaque signature for a thinking part, empty when the part is unsigned.
		signature: Bytes,
	},
	/// Requests in-turn work from the client's declared executor.
	Invoke(Invoke),
	/// Abandons a live invocation; the client must stop work and send no more
	/// frames for it.
	InvokeCancel {
		/// Correlation id of the abandoned invocation.
		invocation_id: Str,
	},
	/// Terminal success containing the authoritative commit record.
	Outcome(ChatOutcome),
	/// Terminal failure that leaves context unchanged.
	Error(TurnError),
}
