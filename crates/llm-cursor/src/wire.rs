//! Generated prost bindings for Cursor's `agent.v1` wire schema.
//!
//! Pinned from `packages/ai/src/providers/cursor/proto/agent.proto`.

#![allow(
	missing_docs,
	non_camel_case_types,
	clippy::pedantic,
	clippy::nursery,
	reason = "handwritten full mirror preserves pinned external schema names"
)]

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum AppliedAgentChange_ChangeType {
	ChangeTypeUnspecified = 0,
	ChangeTypeCreated  = 1,
	ChangeTypeModified = 2,
	ChangeTypeDeleted  = 3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum MouseButton {
	Unspecified = 0,
	Left        = 1,
	Right       = 2,
	Middle      = 3,
	Back        = 4,
	Forward     = 5,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum ScrollDirection {
	Unspecified = 0,
	Up          = 1,
	Down        = 2,
	Left        = 3,
	Right       = 4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum CursorRuleSource {
	Unspecified = 0,
	Team        = 1,
	User        = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum DiagnosticSeverity {
	Unspecified = 0,
	Error       = 1,
	Warning     = 2,
	Information = 3,
	Hint        = 4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum RecordingMode {
	Unspecified      = 0,
	StartRecording   = 1,
	SaveRecording    = 2,
	DiscardRecording = 3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum RequestedFilePathRejectedReason {
	Unspecified       = 0,
	SlashesNotAllowed = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum PackageType {
	Unspecified    = 0,
	CursorProject  = 1,
	CursorPersonal = 2,
	ClaudeSkill    = 3,
	ClaudePlugin   = 4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum SandboxPolicy_Type {
	TypeUnspecified  = 0,
	TypeInsecureNone = 1,
	TypeWorkspaceReadwrite = 2,
	TypeWorkspaceReadonly = 3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum TimeoutBehavior {
	Unspecified = 0,
	Cancel      = 1,
	Background  = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum ShellAbortReason {
	Unspecified = 0,
	UserAbort   = 1,
	Timeout     = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum CustomSubagentPermissionMode {
	Unspecified = 0,
	Default     = 1,
	Readonly    = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum TodoStatus {
	Unspecified = 0,
	Pending     = 1,
	InProgress  = 2,
	Completed   = 3,
	Cancelled   = 4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum ClientOS {
	Unspecified = 0,
	Windows     = 1,
	Macos       = 2,
	Linux       = 3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum ArtifactUploadDispatchStatus {
	Unspecified = 0,
	Accepted    = 1,
	Rejected    = 2,
	SkippedAlreadyInProgress = 3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum Frame_Kind {
	KindUnspecified = 0,
	KindRequest     = 1,
	KindResponse    = 2,
	KindError       = 3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum BugbotDeeplinkEventKind {
	Unspecified        = 0,
	Clicked            = 1,
	HandledDialogShown = 2,
	HandledChatCreated = 3,
	Error              = 4,
	HandledFixInWeb    = 5,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum CommandClassifierResult_SuggestedSandboxMode {
	SuggestedSandboxModeUnspecified = 0,
	SuggestedSandboxModeSandbox = 1,
	SuggestedSandboxModeNoSandbox = 2,
	SuggestedSandboxModeUndetermined = 3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum ShellHookApprovalRequirement_Kind {
	ShellHookApprovalRequirementKindUnspecified = 0,
	ShellHookApprovalRequirementKindForcePrompt = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum ShellBackgroundReason {
	Unspecified = 0,
	Timeout     = 1,
	UserRequest = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum ForceBackgroundShellStatus {
	Unspecified = 0,
	Accepted    = 1,
	NotFound    = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum SubagentBackgroundReason {
	Unspecified    = 0,
	AgentRequest   = 1,
	UserRequest    = 2,
	QueuedFollowUp = 3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum ForceBackgroundSubagentStatus {
	Unspecified = 0,
	Accepted    = 1,
	NotFound    = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum SmartModeClassifierDecision {
	Unspecified = 0,
	Allow       = 1,
	Block       = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum ConversationSearchSource {
	Unspecified = 0,
	Local       = 1,
	CloudCache  = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum GetDiffRequest_OutputFormat {
	OutputFormatUnspecified = 0,
	OutputFormatNameStatus = 1,
	OutputFormatNameStatusAndNumstat = 2,
	OutputFormatFileDiffs = 3,
	OutputFormatDiffsWithBeforeAndAfter = 4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
#[repr(i32)]
pub enum GitDiff_DiffType {
	DiffTypeUnspecified = 0,
	DiffTypeDiffToHead  = 1,
	DiffTypeDiffFromBranchToMain = 2,
}

pub mod glob_tool_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::GlobToolSuccess),
		#[prost(message, tag = "2")]
		Error(super::GlobToolError),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GlobToolResult {
	#[prost(oneof = "glob_tool_result::Result", tags = "1, 2")]
	pub result: ::core::option::Option<glob_tool_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GlobToolError {
	#[prost(string, tag = "1")]
	pub error: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GlobToolSuccess {
	#[prost(string, tag = "1")]
	pub pattern:           ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub path:              ::prost::alloc::string::String,
	#[prost(string, repeated, tag = "3")]
	pub files:             ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
	#[prost(int32, tag = "4")]
	pub total_files:       i32,
	#[prost(bool, tag = "5")]
	pub client_truncated:  bool,
	#[prost(bool, tag = "6")]
	pub ripgrep_truncated: bool,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GlobToolCall {
	#[prost(bytes = "vec", tag = "1")]
	pub args:   ::prost::alloc::vec::Vec<u8>,
	#[prost(message, optional, tag = "2")]
	pub result: ::core::option::Option<GlobToolResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReadLintsToolCall {
	#[prost(message, optional, tag = "1")]
	pub args:   ::core::option::Option<ReadLintsToolArgs>,
	#[prost(message, optional, tag = "2")]
	pub result: ::core::option::Option<ReadLintsToolResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReadLintsToolArgs {
	#[prost(string, repeated, tag = "1")]
	pub paths: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
}

pub mod read_lints_tool_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::ReadLintsToolSuccess),
		#[prost(message, tag = "2")]
		Error(super::ReadLintsToolError),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReadLintsToolResult {
	#[prost(oneof = "read_lints_tool_result::Result", tags = "1, 2")]
	pub result: ::core::option::Option<read_lints_tool_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReadLintsToolSuccess {
	#[prost(message, repeated, tag = "1")]
	pub file_diagnostics:  ::prost::alloc::vec::Vec<FileDiagnostics>,
	#[prost(int32, tag = "2")]
	pub total_files:       i32,
	#[prost(int32, tag = "3")]
	pub total_diagnostics: i32,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FileDiagnostics {
	#[prost(string, tag = "1")]
	pub path:              ::prost::alloc::string::String,
	#[prost(message, repeated, tag = "2")]
	pub diagnostics:       ::prost::alloc::vec::Vec<DiagnosticItem>,
	#[prost(int32, tag = "3")]
	pub diagnostics_count: i32,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DiagnosticItem {
	#[prost(enumeration = "DiagnosticSeverity", tag = "1")]
	pub severity: i32,
	#[prost(message, optional, tag = "2")]
	pub range:    ::core::option::Option<DiagnosticRange>,
	#[prost(string, tag = "3")]
	pub message:  ::prost::alloc::string::String,
	#[prost(string, tag = "4")]
	pub source:   ::prost::alloc::string::String,
	#[prost(string, tag = "5")]
	pub code:     ::prost::alloc::string::String,
	#[prost(bool, tag = "6")]
	pub is_stale: bool,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DiagnosticRange {
	#[prost(message, optional, tag = "1")]
	pub start: ::core::option::Option<Position>,
	#[prost(message, optional, tag = "2")]
	pub end:   ::core::option::Option<Position>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReadLintsToolError {
	#[prost(string, tag = "1")]
	pub error_message: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct McpToolError {
	#[prost(string, tag = "1")]
	pub error:                  ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub read_tool_def_reminder: ::prost::alloc::string::String,
}

pub mod mcp_tool_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::McpSuccess),
		#[prost(message, tag = "2")]
		Error(super::McpToolError),
		#[prost(message, tag = "3")]
		Rejected(super::McpRejected),
		#[prost(message, tag = "4")]
		PermissionDenied(super::McpPermissionDenied),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct McpToolResult {
	#[prost(oneof = "mcp_tool_result::Result", tags = "1, 2, 3, 4")]
	pub result: ::core::option::Option<mcp_tool_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct McpToolCall {
	#[prost(message, optional, tag = "1")]
	pub args:        ::core::option::Option<McpArgs>,
	#[prost(message, optional, tag = "2")]
	pub result:      ::core::option::Option<McpToolResult>,
	#[prost(string, optional, tag = "3")]
	pub description: ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SemSearchToolCall {
	#[prost(message, optional, tag = "1")]
	pub args:   ::core::option::Option<SemSearchToolArgs>,
	#[prost(message, optional, tag = "2")]
	pub result: ::core::option::Option<SemSearchToolResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SemSearchToolArgs {
	#[prost(string, tag = "1")]
	pub query:              ::prost::alloc::string::String,
	#[prost(string, repeated, tag = "2")]
	pub target_directories: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
	#[prost(string, tag = "3")]
	pub explanation:        ::prost::alloc::string::String,
}

pub mod sem_search_tool_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::SemSearchToolSuccess),
		#[prost(message, tag = "2")]
		Error(super::SemSearchToolError),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SemSearchToolResult {
	#[prost(oneof = "sem_search_tool_result::Result", tags = "1, 2")]
	pub result: ::core::option::Option<sem_search_tool_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SemSearchToolSuccess {
	#[prost(string, tag = "1")]
	pub results:      ::prost::alloc::string::String,
	#[prost(bytes = "vec", repeated, tag = "2")]
	pub code_results: ::prost::alloc::vec::Vec<::prost::alloc::vec::Vec<u8>>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SemSearchToolError {
	#[prost(string, tag = "1")]
	pub error_message: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ListMcpResourcesToolCall {
	#[prost(message, optional, tag = "1")]
	pub args:   ::core::option::Option<ListMcpResourcesExecArgs>,
	#[prost(message, optional, tag = "2")]
	pub result: ::core::option::Option<ListMcpResourcesExecResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReadMcpResourceToolCall {
	#[prost(message, optional, tag = "1")]
	pub args:   ::core::option::Option<ReadMcpResourceExecArgs>,
	#[prost(message, optional, tag = "2")]
	pub result: ::core::option::Option<ReadMcpResourceExecResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FetchToolCall {
	#[prost(message, optional, tag = "1")]
	pub args:   ::core::option::Option<FetchArgs>,
	#[prost(message, optional, tag = "2")]
	pub result: ::core::option::Option<FetchResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RecordScreenToolCall {
	#[prost(message, optional, tag = "1")]
	pub args:   ::core::option::Option<RecordScreenArgs>,
	#[prost(message, optional, tag = "2")]
	pub result: ::core::option::Option<RecordScreenResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WriteShellStdinToolCall {
	#[prost(message, optional, tag = "1")]
	pub args:   ::core::option::Option<WriteShellStdinArgs>,
	#[prost(message, optional, tag = "2")]
	pub result: ::core::option::Option<WriteShellStdinResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReflectArgs {
	#[prost(string, tag = "1")]
	pub unexpected_action_outcomes: ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub relevant_instructions:      ::prost::alloc::string::String,
	#[prost(string, tag = "3")]
	pub scenario_analysis:          ::prost::alloc::string::String,
	#[prost(string, tag = "4")]
	pub critical_synthesis:         ::prost::alloc::string::String,
	#[prost(string, tag = "5")]
	pub next_steps:                 ::prost::alloc::string::String,
	#[prost(string, tag = "6")]
	pub tool_call_id:               ::prost::alloc::string::String,
}

pub mod reflect_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::ReflectSuccess),
		#[prost(message, tag = "2")]
		Error(super::ReflectError),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReflectResult {
	#[prost(oneof = "reflect_result::Result", tags = "1, 2")]
	pub result: ::core::option::Option<reflect_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReflectSuccess {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReflectError {
	#[prost(string, tag = "1")]
	pub error: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReflectToolCall {
	#[prost(message, optional, tag = "1")]
	pub args:   ::core::option::Option<ReflectArgs>,
	#[prost(message, optional, tag = "2")]
	pub result: ::core::option::Option<ReflectResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StartGrindExecutionArgs {
	#[prost(string, optional, tag = "1")]
	pub explanation:  ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, tag = "2")]
	pub tool_call_id: ::prost::alloc::string::String,
}

pub mod start_grind_execution_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::StartGrindExecutionSuccess),
		#[prost(message, tag = "2")]
		Error(super::StartGrindExecutionError),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StartGrindExecutionResult {
	#[prost(oneof = "start_grind_execution_result::Result", tags = "1, 2")]
	pub result: ::core::option::Option<start_grind_execution_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StartGrindExecutionSuccess {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StartGrindExecutionError {
	#[prost(string, tag = "1")]
	pub error: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StartGrindExecutionToolCall {
	#[prost(message, optional, tag = "1")]
	pub args:   ::core::option::Option<StartGrindExecutionArgs>,
	#[prost(message, optional, tag = "2")]
	pub result: ::core::option::Option<StartGrindExecutionResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StartGrindPlanningArgs {
	#[prost(string, optional, tag = "1")]
	pub explanation:  ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, tag = "2")]
	pub tool_call_id: ::prost::alloc::string::String,
}

pub mod start_grind_planning_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::StartGrindPlanningSuccess),
		#[prost(message, tag = "2")]
		Error(super::StartGrindPlanningError),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StartGrindPlanningResult {
	#[prost(oneof = "start_grind_planning_result::Result", tags = "1, 2")]
	pub result: ::core::option::Option<start_grind_planning_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StartGrindPlanningSuccess {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StartGrindPlanningError {
	#[prost(string, tag = "1")]
	pub error: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StartGrindPlanningToolCall {
	#[prost(message, optional, tag = "1")]
	pub args:   ::core::option::Option<StartGrindPlanningArgs>,
	#[prost(message, optional, tag = "2")]
	pub result: ::core::option::Option<StartGrindPlanningResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TaskArgs {
	#[prost(string, tag = "1")]
	pub description:   ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub prompt:        ::prost::alloc::string::String,
	#[prost(message, optional, tag = "3")]
	pub subagent_type: ::core::option::Option<SubagentType>,
	#[prost(string, optional, tag = "4")]
	pub model:         ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "5")]
	pub resume:        ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TaskSuccess {
	#[prost(message, repeated, tag = "1")]
	pub conversation_steps: ::prost::alloc::vec::Vec<ConversationStep>,
	#[prost(string, optional, tag = "2")]
	pub agent_id:           ::core::option::Option<::prost::alloc::string::String>,
	#[prost(bool, tag = "3")]
	pub is_background:      bool,
	#[prost(uint64, optional, tag = "4")]
	pub duration_ms:        ::core::option::Option<u64>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TaskError {
	#[prost(string, tag = "1")]
	pub error: ::prost::alloc::string::String,
}

pub mod task_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::TaskSuccess),
		#[prost(message, tag = "2")]
		Error(super::TaskError),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TaskResult {
	#[prost(oneof = "task_result::Result", tags = "1, 2")]
	pub result: ::core::option::Option<task_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TaskToolCall {
	#[prost(message, optional, tag = "1")]
	pub args:   ::core::option::Option<TaskArgs>,
	#[prost(message, optional, tag = "2")]
	pub result: ::core::option::Option<TaskResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TaskToolCallDelta {
	#[prost(message, optional, tag = "1")]
	pub interaction_update: ::core::option::Option<InteractionUpdate>,
}

pub mod tool_call {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Tool {
		#[prost(message, boxed, tag = "1")]
		ShellToolCall(::prost::alloc::boxed::Box<super::ShellToolCall>),
		#[prost(message, tag = "3")]
		DeleteToolCall(super::DeleteToolCall),
		#[prost(message, tag = "4")]
		GlobToolCall(super::GlobToolCall),
		#[prost(message, tag = "5")]
		GrepToolCall(super::GrepToolCall),
		#[prost(message, tag = "8")]
		ReadToolCall(super::ReadToolCall),
		#[prost(message, tag = "9")]
		UpdateTodosToolCall(super::UpdateTodosToolCall),
		#[prost(message, tag = "10")]
		ReadTodosToolCall(super::ReadTodosToolCall),
		#[prost(message, tag = "12")]
		EditToolCall(super::EditToolCall),
		#[prost(message, tag = "13")]
		LsToolCall(super::LsToolCall),
		#[prost(message, tag = "14")]
		ReadLintsToolCall(super::ReadLintsToolCall),
		#[prost(message, tag = "15")]
		McpToolCall(super::McpToolCall),
		#[prost(message, tag = "16")]
		SemSearchToolCall(super::SemSearchToolCall),
		#[prost(message, tag = "17")]
		CreatePlanToolCall(super::CreatePlanToolCall),
		#[prost(message, tag = "18")]
		WebSearchToolCall(super::WebSearchToolCall),
		#[prost(message, tag = "19")]
		TaskToolCall(super::TaskToolCall),
		#[prost(message, tag = "20")]
		ListMcpResourcesToolCall(super::ListMcpResourcesToolCall),
		#[prost(message, tag = "21")]
		ReadMcpResourceToolCall(super::ReadMcpResourceToolCall),
		#[prost(message, tag = "22")]
		ApplyAgentDiffToolCall(super::ApplyAgentDiffToolCall),
		#[prost(message, tag = "23")]
		AskQuestionToolCall(super::AskQuestionToolCall),
		#[prost(message, tag = "24")]
		FetchToolCall(super::FetchToolCall),
		#[prost(message, tag = "25")]
		SwitchModeToolCall(super::SwitchModeToolCall),
		#[prost(message, tag = "26")]
		ExaSearchToolCall(super::ExaSearchToolCall),
		#[prost(message, tag = "27")]
		ExaFetchToolCall(super::ExaFetchToolCall),
		#[prost(message, tag = "28")]
		GenerateImageToolCall(super::GenerateImageToolCall),
		#[prost(message, tag = "29")]
		RecordScreenToolCall(super::RecordScreenToolCall),
		#[prost(message, tag = "30")]
		ComputerUseToolCall(super::ComputerUseToolCall),
		#[prost(message, tag = "31")]
		WriteShellStdinToolCall(super::WriteShellStdinToolCall),
		#[prost(message, tag = "32")]
		ReflectToolCall(super::ReflectToolCall),
		#[prost(message, tag = "33")]
		SetupVmEnvironmentToolCall(super::SetupVmEnvironmentToolCall),
		#[prost(message, tag = "34")]
		TruncatedToolCall(super::TruncatedToolCall),
		#[prost(message, tag = "35")]
		StartGrindExecutionToolCall(super::StartGrindExecutionToolCall),
		#[prost(message, tag = "36")]
		StartGrindPlanningToolCall(super::StartGrindPlanningToolCall),
		#[prost(message, tag = "61")]
		PiReadToolCall(super::PiReadToolCall),
		#[prost(message, tag = "62")]
		PiBashToolCall(super::PiBashToolCall),
		#[prost(message, tag = "63")]
		PiEditToolCall(super::PiEditToolCall),
		#[prost(message, tag = "64")]
		PiWriteToolCall(super::PiWriteToolCall),
		#[prost(message, tag = "65")]
		PiGrepToolCall(super::PiGrepToolCall),
		#[prost(message, tag = "66")]
		PiFindToolCall(super::PiFindToolCall),
		#[prost(message, tag = "67")]
		PiLsToolCall(super::PiLsToolCall),
		#[prost(message, tag = "68")]
		ConnectScmToolCall(super::ConnectScmToolCall),
		#[prost(message, tag = "69")]
		SearchConversationsToolCall(super::SearchConversationsToolCall),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ToolCall {
	#[prost(
		oneof = "tool_call::Tool",
		tags = "1, 3, 4, 5, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, \
		        27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 61, 62, 63, 64, 65, 66, 67, 68, 69"
	)]
	pub tool:         ::core::option::Option<tool_call::Tool>,
	#[prost(string, optional, tag = "57")]
	pub tool_call_id: ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TruncatedToolCallArgs {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TruncatedToolCallSuccess {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TruncatedToolCallError {
	#[prost(string, tag = "1")]
	pub error: ::prost::alloc::string::String,
}

pub mod truncated_tool_call_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::TruncatedToolCallSuccess),
		#[prost(message, tag = "2")]
		Error(super::TruncatedToolCallError),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TruncatedToolCallResult {
	#[prost(oneof = "truncated_tool_call_result::Result", tags = "1, 2")]
	pub result: ::core::option::Option<truncated_tool_call_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TruncatedToolCall {
	#[prost(bytes = "vec", tag = "1")]
	pub original_step_blob_id: ::prost::alloc::vec::Vec<u8>,
	#[prost(message, optional, tag = "2")]
	pub args:                  ::core::option::Option<TruncatedToolCallArgs>,
	#[prost(message, optional, tag = "3")]
	pub result:                ::core::option::Option<TruncatedToolCallResult>,
}

pub mod tool_call_delta {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Delta {
		#[prost(message, boxed, tag = "1")]
		ShellToolCallDelta(::prost::alloc::boxed::Box<super::ShellToolCallDelta>),
		#[prost(message, boxed, tag = "2")]
		TaskToolCallDelta(::prost::alloc::boxed::Box<super::TaskToolCallDelta>),
		#[prost(message, tag = "3")]
		EditToolCallDelta(super::EditToolCallDelta),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ToolCallDelta {
	#[prost(oneof = "tool_call_delta::Delta", tags = "1, 2, 3")]
	pub delta: ::core::option::Option<tool_call_delta::Delta>,
}

pub mod conversation_step {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Message {
		#[prost(message, boxed, tag = "1")]
		AssistantMessage(::prost::alloc::boxed::Box<super::AssistantMessage>),
		#[prost(message, boxed, tag = "2")]
		ToolCall(::prost::alloc::boxed::Box<super::ToolCall>),
		#[prost(message, tag = "3")]
		ThinkingMessage(super::ThinkingMessage),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ConversationStep {
	#[prost(oneof = "conversation_step::Message", tags = "1, 2, 3")]
	pub message: ::core::option::Option<conversation_step::Message>,
}

pub mod conversation_action {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Action {
		#[prost(message, tag = "1")]
		UserMessageAction(super::UserMessageAction),
		#[prost(message, tag = "2")]
		ResumeAction(super::ResumeAction),
		#[prost(message, tag = "3")]
		CancelAction(super::CancelAction),
		#[prost(message, tag = "4")]
		SummarizeAction(super::SummarizeAction),
		#[prost(message, tag = "5")]
		ShellCommandAction(super::ShellCommandAction),
		#[prost(message, tag = "6")]
		StartPlanAction(super::StartPlanAction),
		#[prost(message, tag = "7")]
		ExecutePlanAction(super::ExecutePlanAction),
		#[prost(message, tag = "8")]
		AsyncAskQuestionCompletionAction(super::AsyncAskQuestionCompletionAction),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ConversationAction {
	#[prost(oneof = "conversation_action::Action", tags = "1, 2, 3, 4, 5, 6, 7, 8")]
	pub action: ::core::option::Option<conversation_action::Action>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct UserMessageAction {
	#[prost(message, optional, tag = "1")]
	pub user_message:                 ::core::option::Option<UserMessage>,
	#[prost(message, optional, tag = "2")]
	pub request_context:              ::core::option::Option<RequestContext>,
	#[prost(bool, optional, tag = "3")]
	pub send_to_interaction_listener: ::core::option::Option<bool>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CancelAction {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ResumeAction {
	#[prost(message, optional, tag = "2")]
	pub request_context: ::core::option::Option<RequestContext>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AsyncAskQuestionCompletionAction {
	#[prost(string, tag = "1")]
	pub original_tool_call_id: ::prost::alloc::string::String,
	#[prost(message, optional, tag = "2")]
	pub original_args:         ::core::option::Option<AskQuestionArgs>,
	#[prost(message, optional, tag = "3")]
	pub result:                ::core::option::Option<AskQuestionResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SummarizeAction {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ShellCommandAction {
	#[prost(message, optional, tag = "1")]
	pub shell_command: ::core::option::Option<ShellCommand>,
	#[prost(string, tag = "2")]
	pub exec_id:       ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StartPlanAction {
	#[prost(message, optional, tag = "1")]
	pub user_message:    ::core::option::Option<UserMessage>,
	#[prost(message, optional, tag = "2")]
	pub request_context: ::core::option::Option<RequestContext>,
	#[prost(bool, tag = "3")]
	pub is_spec:         bool,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExecutePlanAction {
	#[prost(message, optional, tag = "1")]
	pub request_context:   ::core::option::Option<RequestContext>,
	#[prost(message, optional, tag = "2")]
	pub plan:              ::core::option::Option<ConversationPlan>,
	#[prost(string, optional, tag = "3")]
	pub plan_file_uri:     ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "4")]
	pub plan_file_content: ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct UserMessage {
	#[prost(string, tag = "1")]
	pub text: ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub message_id: ::prost::alloc::string::String,
	#[prost(message, optional, tag = "3")]
	pub selected_context: ::core::option::Option<SelectedContext>,
	#[prost(int32, tag = "4")]
	pub mode: i32,
	#[prost(bool, optional, tag = "5")]
	pub is_simulated_msg: ::core::option::Option<bool>,
	#[prost(string, optional, tag = "6")]
	pub best_of_n_group_id: ::core::option::Option<::prost::alloc::string::String>,
	#[prost(bool, optional, tag = "7")]
	pub try_use_best_of_n_promotion: ::core::option::Option<bool>,
	#[prost(string, optional, tag = "8")]
	pub rich_text: ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AssistantMessage {
	#[prost(string, tag = "1")]
	pub text: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ThinkingMessage {
	#[prost(string, tag = "1")]
	pub text:        ::prost::alloc::string::String,
	#[prost(uint32, tag = "2")]
	pub duration_ms: u32,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ShellCommand {
	#[prost(string, tag = "1")]
	pub command: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ShellOutput {
	#[prost(string, tag = "1")]
	pub stdout:    ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub stderr:    ::prost::alloc::string::String,
	#[prost(int32, tag = "3")]
	pub exit_code: i32,
}

pub mod conversation_turn {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Turn {
		#[prost(message, boxed, tag = "1")]
		AgentConversationTurn(::prost::alloc::boxed::Box<super::AgentConversationTurn>),
		#[prost(message, tag = "2")]
		ShellConversationTurn(super::ShellConversationTurn),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ConversationTurn {
	#[prost(oneof = "conversation_turn::Turn", tags = "1, 2")]
	pub turn: ::core::option::Option<conversation_turn::Turn>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ConversationPlan {
	#[prost(string, tag = "1")]
	pub plan: ::prost::alloc::string::String,
}

pub mod conversation_turn_structure {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Turn {
		#[prost(message, tag = "1")]
		AgentConversationTurn(super::AgentConversationTurnStructure),
		#[prost(message, tag = "2")]
		ShellConversationTurn(super::ShellConversationTurnStructure),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ConversationTurnStructure {
	#[prost(oneof = "conversation_turn_structure::Turn", tags = "1, 2")]
	pub turn: ::core::option::Option<conversation_turn_structure::Turn>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AgentConversationTurn {
	#[prost(message, optional, tag = "1")]
	pub user_message: ::core::option::Option<UserMessage>,
	#[prost(message, repeated, tag = "2")]
	pub steps:        ::prost::alloc::vec::Vec<ConversationStep>,
	#[prost(string, optional, tag = "3")]
	pub request_id:   ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AgentConversationTurnStructure {
	#[prost(bytes = "vec", tag = "1")]
	pub user_message: ::prost::alloc::vec::Vec<u8>,
	#[prost(bytes = "vec", repeated, tag = "2")]
	pub steps:        ::prost::alloc::vec::Vec<::prost::alloc::vec::Vec<u8>>,
	#[prost(string, optional, tag = "3")]
	pub request_id:   ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ShellConversationTurn {
	#[prost(message, optional, tag = "1")]
	pub shell_command: ::core::option::Option<ShellCommand>,
	#[prost(message, optional, tag = "2")]
	pub shell_output:  ::core::option::Option<ShellOutput>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ShellConversationTurnStructure {
	#[prost(bytes = "vec", tag = "1")]
	pub shell_command: ::prost::alloc::vec::Vec<u8>,
	#[prost(bytes = "vec", tag = "2")]
	pub shell_output:  ::prost::alloc::vec::Vec<u8>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ConversationSummary {
	#[prost(string, tag = "1")]
	pub summary: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ConversationSummaryArchive {
	#[prost(bytes = "vec", repeated, tag = "1")]
	pub summarized_messages: ::prost::alloc::vec::Vec<::prost::alloc::vec::Vec<u8>>,
	#[prost(string, tag = "2")]
	pub summary:             ::prost::alloc::string::String,
	#[prost(uint32, tag = "3")]
	pub window_tail:         u32,
	#[prost(bytes = "vec", tag = "4")]
	pub summary_message:     ::prost::alloc::vec::Vec<u8>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ConversationTokenDetails {
	#[prost(uint32, tag = "1")]
	pub used_tokens: u32,
	#[prost(uint32, tag = "2")]
	pub max_tokens:  u32,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FileState {
	#[prost(string, optional, tag = "1")]
	pub content:         ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "2")]
	pub initial_content: ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FileStateStructure {
	#[prost(bytes = "vec", optional, tag = "1")]
	pub content:         ::core::option::Option<::prost::alloc::vec::Vec<u8>>,
	#[prost(bytes = "vec", optional, tag = "2")]
	pub initial_content: ::core::option::Option<::prost::alloc::vec::Vec<u8>>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StepTiming {
	#[prost(uint64, tag = "1")]
	pub duration_ms:  u64,
	#[prost(uint64, tag = "2")]
	pub timestamp_ms: u64,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ConversationState {
	#[prost(string, repeated, tag = "1")]
	pub root_prompt_messages_json: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
	#[prost(message, repeated, tag = "8")]
	pub turns: ::prost::alloc::vec::Vec<ConversationTurn>,
	#[prost(message, repeated, tag = "3")]
	pub todos: ::prost::alloc::vec::Vec<TodoItem>,
	#[prost(string, repeated, tag = "4")]
	pub pending_tool_calls: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
	#[prost(message, optional, tag = "5")]
	pub token_details: ::core::option::Option<ConversationTokenDetails>,
	#[prost(message, optional, tag = "6")]
	pub summary: ::core::option::Option<ConversationSummary>,
	#[prost(message, optional, tag = "7")]
	pub plan: ::core::option::Option<ConversationPlan>,
	#[prost(message, optional, tag = "9")]
	pub summary_archive: ::core::option::Option<ConversationSummaryArchive>,
	#[prost(map = "string, message", tag = "10")]
	pub file_states: ::std::collections::HashMap<::prost::alloc::string::String, FileState>,
	#[prost(message, repeated, tag = "11")]
	pub summary_archives: ::prost::alloc::vec::Vec<ConversationSummaryArchive>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SubagentPersistedState {
	#[prost(message, optional, tag = "1")]
	pub conversation_state:     ::core::option::Option<ConversationStateStructure>,
	#[prost(uint64, tag = "2")]
	pub created_timestamp_ms:   u64,
	#[prost(uint64, tag = "3")]
	pub last_used_timestamp_ms: u64,
	#[prost(message, optional, tag = "4")]
	pub subagent_type:          ::core::option::Option<SubagentType>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ConversationStateStructure {
	#[prost(bytes = "vec", repeated, tag = "2")]
	pub turns_old: ::prost::alloc::vec::Vec<::prost::alloc::vec::Vec<u8>>,
	#[prost(bytes = "vec", repeated, tag = "1")]
	pub root_prompt_messages_json: ::prost::alloc::vec::Vec<::prost::alloc::vec::Vec<u8>>,
	#[prost(bytes = "vec", repeated, tag = "8")]
	pub turns: ::prost::alloc::vec::Vec<::prost::alloc::vec::Vec<u8>>,
	#[prost(bytes = "vec", repeated, tag = "3")]
	pub todos: ::prost::alloc::vec::Vec<::prost::alloc::vec::Vec<u8>>,
	#[prost(string, repeated, tag = "4")]
	pub pending_tool_calls: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
	#[prost(message, optional, tag = "5")]
	pub token_details: ::core::option::Option<ConversationTokenDetails>,
	#[prost(bytes = "vec", optional, tag = "6")]
	pub summary: ::core::option::Option<::prost::alloc::vec::Vec<u8>>,
	#[prost(bytes = "vec", optional, tag = "7")]
	pub plan: ::core::option::Option<::prost::alloc::vec::Vec<u8>>,
	#[prost(string, repeated, tag = "9")]
	pub previous_workspace_uris: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
	#[prost(int32, optional, tag = "10")]
	pub mode: ::core::option::Option<i32>,
	#[prost(bytes = "vec", optional, tag = "11")]
	pub summary_archive: ::core::option::Option<::prost::alloc::vec::Vec<u8>>,
	#[prost(map = "string, bytes", tag = "12")]
	pub file_states:
		::std::collections::HashMap<::prost::alloc::string::String, ::prost::alloc::vec::Vec<u8>>,
	#[prost(map = "string, message", tag = "15")]
	pub file_states_v2:
		::std::collections::HashMap<::prost::alloc::string::String, FileStateStructure>,
	#[prost(bytes = "vec", repeated, tag = "13")]
	pub summary_archives: ::prost::alloc::vec::Vec<::prost::alloc::vec::Vec<u8>>,
	#[prost(message, repeated, tag = "14")]
	pub turn_timings: ::prost::alloc::vec::Vec<StepTiming>,
	#[prost(map = "string, message", tag = "16")]
	pub subagent_states:
		::std::collections::HashMap<::prost::alloc::string::String, SubagentPersistedState>,
	#[prost(uint32, tag = "17")]
	pub self_summary_count: u32,
	#[prost(string, repeated, tag = "18")]
	pub read_paths: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ThinkingDetails {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ApiKeyCredentials {
	#[prost(string, tag = "1")]
	pub api_key:  ::prost::alloc::string::String,
	#[prost(string, optional, tag = "2")]
	pub base_url: ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AzureCredentials {
	#[prost(string, tag = "1")]
	pub api_key:    ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub base_url:   ::prost::alloc::string::String,
	#[prost(string, tag = "3")]
	pub deployment: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BedrockCredentials {
	#[prost(string, tag = "1")]
	pub access_key:    ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub secret_key:    ::prost::alloc::string::String,
	#[prost(string, tag = "3")]
	pub region:        ::prost::alloc::string::String,
	#[prost(string, optional, tag = "4")]
	pub session_token: ::core::option::Option<::prost::alloc::string::String>,
}

pub mod model_details {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Credentials {
		#[prost(message, tag = "8")]
		ApiKeyCredentials(super::ApiKeyCredentials),
		#[prost(message, tag = "9")]
		AzureCredentials(super::AzureCredentials),
		#[prost(message, tag = "10")]
		BedrockCredentials(super::BedrockCredentials),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ModelDetails {
	#[prost(string, tag = "1")]
	pub model_id:           ::prost::alloc::string::String,
	#[prost(string, tag = "3")]
	pub display_model_id:   ::prost::alloc::string::String,
	#[prost(string, tag = "4")]
	pub display_name:       ::prost::alloc::string::String,
	#[prost(string, tag = "5")]
	pub display_name_short: ::prost::alloc::string::String,
	#[prost(string, repeated, tag = "6")]
	pub aliases:            ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
	#[prost(message, optional, tag = "2")]
	pub thinking_details:   ::core::option::Option<ThinkingDetails>,
	#[prost(bool, optional, tag = "7")]
	pub max_mode:           ::core::option::Option<bool>,
	#[prost(oneof = "model_details::Credentials", tags = "8, 9, 10")]
	pub credentials:        ::core::option::Option<model_details::Credentials>,
}

pub mod requested_model {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Credentials {
		#[prost(message, tag = "4")]
		ApiKeyCredentials(super::ApiKeyCredentials),
		#[prost(message, tag = "5")]
		AzureCredentials(super::AzureCredentials),
		#[prost(message, tag = "6")]
		BedrockCredentials(super::BedrockCredentials),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RequestedModel {
	#[prost(string, tag = "1")]
	pub model_id:    ::prost::alloc::string::String,
	#[prost(bool, tag = "2")]
	pub max_mode:    bool,
	#[prost(message, repeated, tag = "3")]
	pub parameters:  ::prost::alloc::vec::Vec<RequestedModel_ModelParameterbytes>,
	#[prost(oneof = "requested_model::Credentials", tags = "4, 5, 6")]
	pub credentials: ::core::option::Option<requested_model::Credentials>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RequestedModel_ModelParameterbytes {
	#[prost(string, tag = "1")]
	pub id:    ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub value: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AgentRunRequest {
	#[prost(message, optional, tag = "1")]
	pub conversation_state:      ::core::option::Option<ConversationStateStructure>,
	#[prost(message, optional, tag = "2")]
	pub action:                  ::core::option::Option<ConversationAction>,
	#[prost(message, optional, tag = "3")]
	pub model_details:           ::core::option::Option<ModelDetails>,
	#[prost(message, optional, tag = "9")]
	pub requested_model:         ::core::option::Option<RequestedModel>,
	#[prost(message, optional, tag = "4")]
	pub mcp_tools:               ::core::option::Option<McpTools>,
	#[prost(string, optional, tag = "5")]
	pub conversation_id:         ::core::option::Option<::prost::alloc::string::String>,
	#[prost(message, optional, tag = "6")]
	pub mcp_file_system_options: ::core::option::Option<McpFileSystemOptions>,
	#[prost(message, optional, tag = "7")]
	pub skill_options:           ::core::option::Option<SkillOptions>,
	#[prost(string, optional, tag = "8")]
	pub custom_system_prompt:    ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TextDeltaUpdate {
	#[prost(string, tag = "1")]
	pub text: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ToolCallStartedUpdate {
	#[prost(string, tag = "1")]
	pub call_id:       ::prost::alloc::string::String,
	#[prost(message, optional, tag = "2")]
	pub tool_call:     ::core::option::Option<ToolCall>,
	#[prost(string, tag = "3")]
	pub model_call_id: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ToolCallCompletedUpdate {
	#[prost(string, tag = "1")]
	pub call_id:       ::prost::alloc::string::String,
	#[prost(message, optional, tag = "2")]
	pub tool_call:     ::core::option::Option<ToolCall>,
	#[prost(string, tag = "3")]
	pub model_call_id: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ToolCallDeltaUpdate {
	#[prost(string, tag = "1")]
	pub call_id:         ::prost::alloc::string::String,
	#[prost(message, boxed, optional, tag = "2")]
	pub tool_call_delta: ::core::option::Option<Box<ToolCallDelta>>,
	#[prost(string, tag = "3")]
	pub model_call_id:   ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PartialToolCallUpdate {
	#[prost(string, tag = "1")]
	pub call_id:         ::prost::alloc::string::String,
	#[prost(message, optional, tag = "2")]
	pub tool_call:       ::core::option::Option<ToolCall>,
	#[prost(string, tag = "3")]
	pub args_text_delta: ::prost::alloc::string::String,
	#[prost(string, tag = "4")]
	pub model_call_id:   ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ThinkingDeltaUpdate {
	#[prost(string, tag = "1")]
	pub text: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ThinkingCompletedUpdate {
	#[prost(int32, tag = "1")]
	pub thinking_duration_ms: i32,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TokenDeltaUpdate {
	#[prost(int32, tag = "1")]
	pub tokens: i32,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SummaryUpdate {
	#[prost(string, tag = "1")]
	pub summary: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SummaryStartedUpdate {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct HeartbeatUpdate {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SummaryCompletedUpdate {}

pub mod shell_output_delta_update {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Event {
		#[prost(message, tag = "1")]
		Stdout(super::ShellStreamStdout),
		#[prost(message, tag = "2")]
		Stderr(super::ShellStreamStderr),
		#[prost(message, tag = "3")]
		Exit(super::ShellStreamExit),
		#[prost(message, tag = "4")]
		Start(super::ShellStreamStart),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ShellOutputDeltaUpdate {
	#[prost(oneof = "shell_output_delta_update::Event", tags = "1, 2, 3, 4")]
	pub event: ::core::option::Option<shell_output_delta_update::Event>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TurnEndedUpdate {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct UserMessageAppendedUpdate {
	#[prost(message, optional, tag = "1")]
	pub user_message: ::core::option::Option<UserMessage>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StepStartedUpdate {
	#[prost(uint64, tag = "1")]
	pub step_id: u64,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StepCompletedUpdate {
	#[prost(uint64, tag = "1")]
	pub step_id:          u64,
	#[prost(int64, tag = "2")]
	pub step_duration_ms: i64,
}

pub mod interaction_update {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Message {
		#[prost(message, tag = "1")]
		TextDelta(super::TextDeltaUpdate),
		#[prost(message, tag = "7")]
		PartialToolCall(super::PartialToolCallUpdate),
		#[prost(message, tag = "15")]
		ToolCallDelta(super::ToolCallDeltaUpdate),
		#[prost(message, tag = "2")]
		ToolCallStarted(super::ToolCallStartedUpdate),
		#[prost(message, tag = "3")]
		ToolCallCompleted(super::ToolCallCompletedUpdate),
		#[prost(message, tag = "4")]
		ThinkingDelta(super::ThinkingDeltaUpdate),
		#[prost(message, tag = "5")]
		ThinkingCompleted(super::ThinkingCompletedUpdate),
		#[prost(message, tag = "6")]
		UserMessageAppended(super::UserMessageAppendedUpdate),
		#[prost(message, tag = "8")]
		TokenDelta(super::TokenDeltaUpdate),
		#[prost(message, tag = "9")]
		Summary(super::SummaryUpdate),
		#[prost(message, tag = "10")]
		SummaryStarted(super::SummaryStartedUpdate),
		#[prost(message, tag = "11")]
		SummaryCompleted(super::SummaryCompletedUpdate),
		#[prost(message, tag = "12")]
		ShellOutputDelta(super::ShellOutputDeltaUpdate),
		#[prost(message, tag = "13")]
		Heartbeat(super::HeartbeatUpdate),
		#[prost(message, tag = "14")]
		TurnEnded(super::TurnEndedUpdate),
		#[prost(message, tag = "16")]
		StepStarted(super::StepStartedUpdate),
		#[prost(message, tag = "17")]
		StepCompleted(super::StepCompletedUpdate),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct InteractionUpdate {
	#[prost(
		oneof = "interaction_update::Message",
		tags = "1, 7, 15, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17"
	)]
	pub message: ::core::option::Option<interaction_update::Message>,
}

pub mod interaction_query {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Query {
		#[prost(message, tag = "2")]
		WebSearchRequestQuery(super::WebSearchRequestQuery),
		#[prost(message, tag = "3")]
		AskQuestionInteractionQuery(super::AskQuestionInteractionQuery),
		#[prost(message, tag = "4")]
		SwitchModeRequestQuery(super::SwitchModeRequestQuery),
		#[prost(message, tag = "5")]
		ExaSearchRequestQuery(super::ExaSearchRequestQuery),
		#[prost(message, tag = "6")]
		ExaFetchRequestQuery(super::ExaFetchRequestQuery),
		#[prost(message, tag = "7")]
		CreatePlanRequestQuery(super::CreatePlanRequestQuery),
		#[prost(message, tag = "8")]
		SetupVmEnvironmentArgs(super::SetupVmEnvironmentArgs),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct InteractionQuery {
	#[prost(uint32, tag = "1")]
	pub id:    u32,
	#[prost(oneof = "interaction_query::Query", tags = "2, 3, 4, 5, 6, 7, 8")]
	pub query: ::core::option::Option<interaction_query::Query>,
}

pub mod interaction_response {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "2")]
		WebSearchRequestResponse(super::WebSearchRequestResponse),
		#[prost(message, tag = "3")]
		AskQuestionInteractionResponse(super::AskQuestionInteractionResponse),
		#[prost(message, tag = "4")]
		SwitchModeRequestResponse(super::SwitchModeRequestResponse),
		#[prost(message, tag = "5")]
		ExaSearchRequestResponse(super::ExaSearchRequestResponse),
		#[prost(message, tag = "6")]
		ExaFetchRequestResponse(super::ExaFetchRequestResponse),
		#[prost(message, tag = "7")]
		CreatePlanRequestResponse(super::CreatePlanRequestResponse),
		#[prost(message, tag = "8")]
		SetupVmEnvironmentResult(super::SetupVmEnvironmentResult),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct InteractionResponse {
	#[prost(uint32, tag = "1")]
	pub id:     u32,
	#[prost(oneof = "interaction_response::Result", tags = "2, 3, 4, 5, 6, 7, 8")]
	pub result: ::core::option::Option<interaction_response::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AskQuestionInteractionQuery {
	#[prost(message, optional, tag = "1")]
	pub args:         ::core::option::Option<AskQuestionArgs>,
	#[prost(string, tag = "2")]
	pub tool_call_id: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AskQuestionInteractionResponse {
	#[prost(message, optional, tag = "1")]
	pub result: ::core::option::Option<AskQuestionResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ClientHeartbeat {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PrewarmRequest {
	#[prost(message, optional, tag = "1")]
	pub model_details:               ::core::option::Option<ModelDetails>,
	#[prost(message, optional, tag = "9")]
	pub requested_model:             ::core::option::Option<RequestedModel>,
	#[prost(string, optional, tag = "2")]
	pub conversation_id:             ::core::option::Option<::prost::alloc::string::String>,
	#[prost(message, optional, tag = "3")]
	pub conversation_state:          ::core::option::Option<ConversationStateStructure>,
	#[prost(message, optional, tag = "4")]
	pub mcp_tools:                   ::core::option::Option<McpTools>,
	#[prost(message, optional, tag = "5")]
	pub mcp_file_system_options:     ::core::option::Option<McpFileSystemOptions>,
	#[prost(string, optional, tag = "6")]
	pub best_of_n_group_id:          ::core::option::Option<::prost::alloc::string::String>,
	#[prost(bool, optional, tag = "7")]
	pub try_use_best_of_n_promotion: ::core::option::Option<bool>,
	#[prost(string, optional, tag = "8")]
	pub custom_system_prompt:        ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExecServerAbort {
	#[prost(uint32, tag = "1")]
	pub id: u32,
}

pub mod exec_server_control_message {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Message {
		#[prost(message, tag = "1")]
		Abort(super::ExecServerAbort),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExecServerControlMessage {
	#[prost(oneof = "exec_server_control_message::Message", tags = "1")]
	pub message: ::core::option::Option<exec_server_control_message::Message>,
}

pub mod agent_client_message {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Message {
		#[prost(message, boxed, tag = "1")]
		RunRequest(::prost::alloc::boxed::Box<super::AgentRunRequest>),
		#[prost(message, tag = "2")]
		ExecClientMessage(super::ExecClientMessage),
		#[prost(message, tag = "5")]
		ExecClientControlMessage(super::ExecClientControlMessage),
		#[prost(message, tag = "3")]
		KvClientMessage(super::KvClientMessage),
		#[prost(message, boxed, tag = "4")]
		ConversationAction(::prost::alloc::boxed::Box<super::ConversationAction>),
		#[prost(message, tag = "6")]
		InteractionResponse(super::InteractionResponse),
		#[prost(message, tag = "7")]
		ClientHeartbeat(super::ClientHeartbeat),
		#[prost(message, boxed, tag = "8")]
		PrewarmRequest(::prost::alloc::boxed::Box<super::PrewarmRequest>),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AgentClientMessage {
	#[prost(oneof = "agent_client_message::Message", tags = "1, 2, 5, 3, 4, 6, 7, 8")]
	pub message: ::core::option::Option<agent_client_message::Message>,
}

pub mod agent_server_message {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Message {
		#[prost(message, boxed, tag = "1")]
		InteractionUpdate(::prost::alloc::boxed::Box<super::InteractionUpdate>),
		#[prost(message, tag = "2")]
		ExecServerMessage(super::ExecServerMessage),
		#[prost(message, tag = "5")]
		ExecServerControlMessage(super::ExecServerControlMessage),
		#[prost(message, tag = "3")]
		ConversationCheckpointUpdate(super::ConversationStateStructure),
		#[prost(message, tag = "4")]
		KvServerMessage(super::KvServerMessage),
		#[prost(message, tag = "7")]
		InteractionQuery(super::InteractionQuery),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AgentServerMessage {
	#[prost(oneof = "agent_server_message::Message", tags = "1, 2, 5, 3, 4, 7")]
	pub message: ::core::option::Option<agent_server_message::Message>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NameAgentRequest {
	#[prost(string, tag = "1")]
	pub user_message: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct NameAgentResponse {
	#[prost(string, tag = "1")]
	pub name: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetUsableModelsRequest {
	#[prost(string, repeated, tag = "1")]
	pub custom_model_ids: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetUsableModelsResponse {
	#[prost(message, repeated, tag = "1")]
	pub models: ::prost::alloc::vec::Vec<ModelDetails>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetDefaultModelForCliRequest {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetDefaultModelForCliResponse {
	#[prost(message, optional, tag = "1")]
	pub model: ::core::option::Option<ModelDetails>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetAllowedModelIntentsRequest {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetAllowedModelIntentsResponse {
	#[prost(string, repeated, tag = "1")]
	pub model_intents: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct IdeEditorsStateFile {
	#[prost(string, tag = "1")]
	pub relative_path:        ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub absolute_path:        ::prost::alloc::string::String,
	#[prost(bool, optional, tag = "3")]
	pub is_currently_focused: ::core::option::Option<bool>,
	#[prost(int32, optional, tag = "4")]
	pub current_line_number:  ::core::option::Option<i32>,
	#[prost(string, optional, tag = "5")]
	pub current_line_text:    ::core::option::Option<::prost::alloc::string::String>,
	#[prost(int32, optional, tag = "6")]
	pub line_count:           ::core::option::Option<i32>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct IdeEditorsStateLite {
	#[prost(message, repeated, tag = "1")]
	pub recently_viewed_files: ::prost::alloc::vec::Vec<IdeEditorsStateFile>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ApplyAgentDiffToolCall {
	#[prost(message, optional, tag = "1")]
	pub args:   ::core::option::Option<ApplyAgentDiffArgs>,
	#[prost(message, optional, tag = "2")]
	pub result: ::core::option::Option<ApplyAgentDiffResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ApplyAgentDiffArgs {
	#[prost(string, tag = "1")]
	pub agent_id: ::prost::alloc::string::String,
}

pub mod apply_agent_diff_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::ApplyAgentDiffSuccess),
		#[prost(message, tag = "2")]
		Error(super::ApplyAgentDiffError),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ApplyAgentDiffResult {
	#[prost(oneof = "apply_agent_diff_result::Result", tags = "1, 2")]
	pub result: ::core::option::Option<apply_agent_diff_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ApplyAgentDiffSuccess {
	#[prost(message, repeated, tag = "1")]
	pub applied_changes: ::prost::alloc::vec::Vec<AppliedAgentChange>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AppliedAgentChange {
	#[prost(string, tag = "1")]
	pub path:              ::prost::alloc::string::String,
	#[prost(int32, tag = "2")]
	pub change_type:       i32,
	#[prost(string, optional, tag = "3")]
	pub before_content:    ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "4")]
	pub after_content:     ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "5")]
	pub error:             ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "6")]
	pub message_for_model: ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ApplyAgentDiffError {
	#[prost(string, tag = "1")]
	pub error:           ::prost::alloc::string::String,
	#[prost(message, repeated, tag = "2")]
	pub applied_changes: ::prost::alloc::vec::Vec<AppliedAgentChange>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AskQuestionToolCall {
	#[prost(message, optional, tag = "1")]
	pub args:   ::core::option::Option<AskQuestionArgs>,
	#[prost(message, optional, tag = "2")]
	pub result: ::core::option::Option<AskQuestionResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AskQuestionArgs {
	#[prost(string, tag = "1")]
	pub title: ::prost::alloc::string::String,
	#[prost(message, repeated, tag = "2")]
	pub questions: ::prost::alloc::vec::Vec<AskQuestionArgs_Question>,
	#[prost(bool, tag = "5")]
	pub run_async: bool,
	#[prost(string, tag = "6")]
	pub async_original_tool_call_id: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AskQuestionArgs_Question {
	#[prost(string, tag = "1")]
	pub id:             ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub prompt:         ::prost::alloc::string::String,
	#[prost(message, repeated, tag = "3")]
	pub options:        ::prost::alloc::vec::Vec<AskQuestionArgs_Option>,
	#[prost(bool, tag = "4")]
	pub allow_multiple: bool,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AskQuestionArgs_Option {
	#[prost(string, tag = "1")]
	pub id:    ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub label: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AskQuestionAsync {}

pub mod ask_question_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::AskQuestionSuccess),
		#[prost(message, tag = "2")]
		Error(super::AskQuestionError),
		#[prost(message, tag = "3")]
		Rejected(super::AskQuestionRejected),
		#[prost(message, tag = "4")]
		Async(super::AskQuestionAsync),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AskQuestionResult {
	#[prost(oneof = "ask_question_result::Result", tags = "1, 2, 3, 4")]
	pub result: ::core::option::Option<ask_question_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AskQuestionSuccess {
	#[prost(message, repeated, tag = "1")]
	pub answers: ::prost::alloc::vec::Vec<AskQuestionSuccess_Answer>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AskQuestionSuccess_Answer {
	#[prost(string, tag = "1")]
	pub question_id:         ::prost::alloc::string::String,
	#[prost(string, repeated, tag = "2")]
	pub selected_option_ids: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AskQuestionError {
	#[prost(string, tag = "1")]
	pub error_message: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AskQuestionRejected {
	#[prost(string, tag = "1")]
	pub reason: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BackgroundShellSpawnArgs {
	#[prost(string, tag = "1")]
	pub command: ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub working_directory: ::prost::alloc::string::String,
	#[prost(string, tag = "3")]
	pub tool_call_id: ::prost::alloc::string::String,
	#[prost(message, optional, tag = "4")]
	pub parsing_result: ::core::option::Option<ShellCommandParsingResult>,
	#[prost(message, optional, tag = "5")]
	pub sandbox_policy: ::core::option::Option<SandboxPolicy>,
	#[prost(bool, tag = "6")]
	pub enable_write_shell_stdin_tool: bool,
	#[prost(string, optional, tag = "7")]
	pub description: ::core::option::Option<::prost::alloc::string::String>,
	#[prost(message, optional, tag = "8")]
	pub classifier_result: ::core::option::Option<CommandClassifierResult>,
	#[prost(message, optional, tag = "9")]
	pub output_notification: ::core::option::Option<ShellOutputNotificationConfig>,
	#[prost(message, optional, tag = "10")]
	pub smart_mode_approval: ::core::option::Option<SmartModeApproval>,
	#[prost(message, optional, tag = "11")]
	pub hook_approval_requirement: ::core::option::Option<ShellHookApprovalRequirement>,
	#[prost(bool, tag = "12")]
	pub skip_approval: bool,
	#[prost(string, optional, tag = "13")]
	pub conversation_id: ::core::option::Option<::prost::alloc::string::String>,
}

pub mod background_shell_spawn_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::BackgroundShellSpawnSuccess),
		#[prost(message, tag = "2")]
		Error(super::BackgroundShellSpawnError),
		#[prost(message, tag = "3")]
		Rejected(super::ShellRejected),
		#[prost(message, tag = "4")]
		PermissionDenied(super::ShellPermissionDenied),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BackgroundShellSpawnResult {
	#[prost(oneof = "background_shell_spawn_result::Result", tags = "1, 2, 3, 4")]
	pub result: ::core::option::Option<background_shell_spawn_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BackgroundShellSpawnSuccess {
	#[prost(uint32, tag = "1")]
	pub shell_id:          u32,
	#[prost(string, tag = "2")]
	pub command:           ::prost::alloc::string::String,
	#[prost(string, tag = "3")]
	pub working_directory: ::prost::alloc::string::String,
	#[prost(uint32, optional, tag = "4")]
	pub pid:               ::core::option::Option<u32>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BackgroundShellSpawnError {
	#[prost(string, tag = "1")]
	pub command:           ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub working_directory: ::prost::alloc::string::String,
	#[prost(string, tag = "3")]
	pub error:             ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WriteShellStdinArgs {
	#[prost(uint32, tag = "1")]
	pub shell_id: u32,
	#[prost(string, tag = "2")]
	pub chars:    ::prost::alloc::string::String,
}

pub mod write_shell_stdin_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::WriteShellStdinSuccess),
		#[prost(message, tag = "2")]
		Error(super::WriteShellStdinError),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WriteShellStdinResult {
	#[prost(oneof = "write_shell_stdin_result::Result", tags = "1, 2")]
	pub result: ::core::option::Option<write_shell_stdin_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WriteShellStdinSuccess {
	#[prost(uint32, tag = "1")]
	pub shell_id: u32,
	#[prost(uint32, tag = "2")]
	pub terminal_file_length_before_input_written: u32,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WriteShellStdinError {
	#[prost(string, tag = "1")]
	pub error: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Coordinate {
	#[prost(int32, tag = "1")]
	pub x: i32,
	#[prost(int32, tag = "2")]
	pub y: i32,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ComputerUseArgs {
	#[prost(string, tag = "1")]
	pub tool_call_id: ::prost::alloc::string::String,
	#[prost(message, repeated, tag = "2")]
	pub actions:      ::prost::alloc::vec::Vec<ComputerUseAction>,
}

pub mod computer_use_action {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Action {
		#[prost(message, tag = "1")]
		MouseMove(super::MouseMoveAction),
		#[prost(message, tag = "2")]
		Click(super::ClickAction),
		#[prost(message, tag = "3")]
		MouseDown(super::MouseDownAction),
		#[prost(message, tag = "4")]
		MouseUp(super::MouseUpAction),
		#[prost(message, tag = "5")]
		Drag(super::DragAction),
		#[prost(message, tag = "6")]
		Scroll(super::ScrollAction),
		#[prost(message, tag = "7")]
		Type(super::TypeAction),
		#[prost(message, tag = "8")]
		Key(super::KeyAction),
		#[prost(message, tag = "9")]
		Wait(super::WaitAction),
		#[prost(message, tag = "10")]
		Screenshot(super::ScreenshotAction),
		#[prost(message, tag = "11")]
		CursorPosition(super::CursorPositionAction),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ComputerUseAction {
	#[prost(oneof = "computer_use_action::Action", tags = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11")]
	pub action: ::core::option::Option<computer_use_action::Action>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct MouseMoveAction {
	#[prost(message, optional, tag = "1")]
	pub coordinate: ::core::option::Option<Coordinate>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ClickAction {
	#[prost(message, optional, tag = "1")]
	pub coordinate:    ::core::option::Option<Coordinate>,
	#[prost(int32, tag = "2")]
	pub button:        i32,
	#[prost(int32, tag = "3")]
	pub count:         i32,
	#[prost(string, optional, tag = "4")]
	pub modifier_keys: ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct MouseDownAction {
	#[prost(int32, tag = "1")]
	pub button: i32,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct MouseUpAction {
	#[prost(int32, tag = "1")]
	pub button: i32,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DragAction {
	#[prost(message, repeated, tag = "1")]
	pub path:   ::prost::alloc::vec::Vec<Coordinate>,
	#[prost(int32, tag = "2")]
	pub button: i32,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ScrollAction {
	#[prost(message, optional, tag = "1")]
	pub coordinate:    ::core::option::Option<Coordinate>,
	#[prost(int32, tag = "2")]
	pub direction:     i32,
	#[prost(int32, tag = "3")]
	pub amount:        i32,
	#[prost(string, optional, tag = "4")]
	pub modifier_keys: ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TypeAction {
	#[prost(string, tag = "1")]
	pub text: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct KeyAction {
	#[prost(string, tag = "1")]
	pub key:              ::prost::alloc::string::String,
	#[prost(int32, optional, tag = "2")]
	pub hold_duration_ms: ::core::option::Option<i32>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WaitAction {
	#[prost(int32, tag = "1")]
	pub duration_ms: i32,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ScreenshotAction {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CursorPositionAction {}

pub mod computer_use_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::ComputerUseSuccess),
		#[prost(message, tag = "2")]
		Error(super::ComputerUseError),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ComputerUseResult {
	#[prost(oneof = "computer_use_result::Result", tags = "1, 2")]
	pub result: ::core::option::Option<computer_use_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ComputerUseSuccess {
	#[prost(int32, tag = "1")]
	pub action_count:    i32,
	#[prost(int32, tag = "2")]
	pub duration_ms:     i32,
	#[prost(string, optional, tag = "3")]
	pub screenshot:      ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "4")]
	pub log:             ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "5")]
	pub screenshot_path: ::core::option::Option<::prost::alloc::string::String>,
	#[prost(message, optional, tag = "6")]
	pub cursor_position: ::core::option::Option<Coordinate>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ComputerUseError {
	#[prost(string, tag = "1")]
	pub error:           ::prost::alloc::string::String,
	#[prost(int32, tag = "2")]
	pub action_count:    i32,
	#[prost(int32, tag = "3")]
	pub duration_ms:     i32,
	#[prost(string, optional, tag = "4")]
	pub log:             ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "5")]
	pub screenshot:      ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "6")]
	pub screenshot_path: ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ComputerUseToolCall {
	#[prost(message, optional, tag = "1")]
	pub args:   ::core::option::Option<ComputerUseArgs>,
	#[prost(message, optional, tag = "2")]
	pub result: ::core::option::Option<ComputerUseResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CreatePlanToolCall {
	#[prost(message, optional, tag = "1")]
	pub args:   ::core::option::Option<CreatePlanArgs>,
	#[prost(message, optional, tag = "2")]
	pub result: ::core::option::Option<CreatePlanResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Phase {
	#[prost(string, tag = "1")]
	pub name:  ::prost::alloc::string::String,
	#[prost(message, repeated, tag = "2")]
	pub todos: ::prost::alloc::vec::Vec<TodoItem>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CreatePlanArgs {
	#[prost(string, tag = "1")]
	pub plan:       ::prost::alloc::string::String,
	#[prost(message, repeated, tag = "2")]
	pub todos:      ::prost::alloc::vec::Vec<TodoItem>,
	#[prost(string, tag = "3")]
	pub overview:   ::prost::alloc::string::String,
	#[prost(string, tag = "4")]
	pub name:       ::prost::alloc::string::String,
	#[prost(bool, tag = "5")]
	pub is_project: bool,
	#[prost(message, repeated, tag = "6")]
	pub phases:     ::prost::alloc::vec::Vec<Phase>,
}

pub mod create_plan_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::CreatePlanSuccess),
		#[prost(message, tag = "2")]
		Error(super::CreatePlanError),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CreatePlanResult {
	#[prost(string, tag = "3")]
	pub plan_uri: ::prost::alloc::string::String,
	#[prost(oneof = "create_plan_result::Result", tags = "1, 2")]
	pub result:   ::core::option::Option<create_plan_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CreatePlanSuccess {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CreatePlanError {
	#[prost(string, tag = "1")]
	pub error: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CreatePlanRequestQuery {
	#[prost(message, optional, tag = "1")]
	pub args:         ::core::option::Option<CreatePlanArgs>,
	#[prost(string, tag = "2")]
	pub tool_call_id: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CreatePlanRequestResponse {
	#[prost(message, optional, tag = "1")]
	pub result: ::core::option::Option<CreatePlanResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CursorRuleTypeGlobal {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CursorRuleTypeFileGlobs {
	#[prost(string, repeated, tag = "1")]
	pub globs: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CursorRuleTypeAgentFetched {
	#[prost(string, tag = "1")]
	pub description: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CursorRuleTypeManuallyAttached {}

pub mod cursor_rule_type {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Type {
		#[prost(message, tag = "1")]
		Global(super::CursorRuleTypeGlobal),
		#[prost(message, tag = "2")]
		FileGlobbed(super::CursorRuleTypeFileGlobs),
		#[prost(message, tag = "3")]
		AgentFetched(super::CursorRuleTypeAgentFetched),
		#[prost(message, tag = "4")]
		ManuallyAttached(super::CursorRuleTypeManuallyAttached),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CursorRuleType {
	#[prost(oneof = "cursor_rule_type::Type", tags = "1, 2, 3, 4")]
	pub r#type: ::core::option::Option<cursor_rule_type::Type>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CursorRule {
	#[prost(string, tag = "1")]
	pub full_path:         ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub content:           ::prost::alloc::string::String,
	#[prost(message, optional, tag = "3")]
	pub r#type:            ::core::option::Option<CursorRuleType>,
	#[prost(int32, tag = "4")]
	pub source:            i32,
	#[prost(string, optional, tag = "5")]
	pub git_remote_origin: ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "6")]
	pub parse_error:       ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DeleteArgs {
	#[prost(string, tag = "1")]
	pub path:         ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub tool_call_id: ::prost::alloc::string::String,
}

pub mod delete_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::DeleteSuccess),
		#[prost(message, tag = "2")]
		FileNotFound(super::DeleteFileNotFound),
		#[prost(message, tag = "3")]
		NotFile(super::DeleteNotFile),
		#[prost(message, tag = "4")]
		PermissionDenied(super::DeletePermissionDenied),
		#[prost(message, tag = "5")]
		FileBusy(super::DeleteFileBusy),
		#[prost(message, tag = "6")]
		Rejected(super::DeleteRejected),
		#[prost(message, tag = "7")]
		Error(super::DeleteError),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DeleteResult {
	#[prost(oneof = "delete_result::Result", tags = "1, 2, 3, 4, 5, 6, 7")]
	pub result: ::core::option::Option<delete_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DeleteSuccess {
	#[prost(string, tag = "1")]
	pub path:         ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub deleted_file: ::prost::alloc::string::String,
	#[prost(int64, tag = "3")]
	pub file_size:    i64,
	#[prost(string, tag = "4")]
	pub prev_content: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DeleteFileNotFound {
	#[prost(string, tag = "1")]
	pub path: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DeleteNotFile {
	#[prost(string, tag = "1")]
	pub path:        ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub actual_type: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DeletePermissionDenied {
	#[prost(string, tag = "1")]
	pub path:                 ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub client_visible_error: ::prost::alloc::string::String,
	#[prost(bool, tag = "3")]
	pub is_readonly:          bool,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DeleteFileBusy {
	#[prost(string, tag = "1")]
	pub path: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DeleteRejected {
	#[prost(string, tag = "1")]
	pub path:   ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub reason: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DeleteError {
	#[prost(string, tag = "1")]
	pub path:  ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub error: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DeleteToolCall {
	#[prost(message, optional, tag = "1")]
	pub args:   ::core::option::Option<DeleteArgs>,
	#[prost(message, optional, tag = "2")]
	pub result: ::core::option::Option<DeleteResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DiagnosticsArgs {
	#[prost(string, tag = "1")]
	pub path:         ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub tool_call_id: ::prost::alloc::string::String,
}

pub mod diagnostics_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::DiagnosticsSuccess),
		#[prost(message, tag = "2")]
		Error(super::DiagnosticsError),
		#[prost(message, tag = "3")]
		Rejected(super::DiagnosticsRejected),
		#[prost(message, tag = "4")]
		FileNotFound(super::DiagnosticsFileNotFound),
		#[prost(message, tag = "5")]
		PermissionDenied(super::DiagnosticsPermissionDenied),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DiagnosticsResult {
	#[prost(oneof = "diagnostics_result::Result", tags = "1, 2, 3, 4, 5")]
	pub result: ::core::option::Option<diagnostics_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DiagnosticsSuccess {
	#[prost(string, tag = "1")]
	pub path:              ::prost::alloc::string::String,
	#[prost(message, repeated, tag = "2")]
	pub diagnostics:       ::prost::alloc::vec::Vec<Diagnostic>,
	#[prost(int32, tag = "3")]
	pub total_diagnostics: i32,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Diagnostic {
	#[prost(int32, tag = "1")]
	pub severity: i32,
	#[prost(message, optional, tag = "2")]
	pub range:    ::core::option::Option<Range>,
	#[prost(string, tag = "3")]
	pub message:  ::prost::alloc::string::String,
	#[prost(string, tag = "4")]
	pub source:   ::prost::alloc::string::String,
	#[prost(string, tag = "5")]
	pub code:     ::prost::alloc::string::String,
	#[prost(bool, tag = "6")]
	pub is_stale: bool,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DiagnosticsError {
	#[prost(string, tag = "1")]
	pub path:  ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub error: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DiagnosticsRejected {
	#[prost(string, tag = "1")]
	pub path:   ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub reason: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DiagnosticsFileNotFound {
	#[prost(string, tag = "1")]
	pub path: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DiagnosticsPermissionDenied {
	#[prost(string, tag = "1")]
	pub path: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct EditArgs {
	#[prost(string, tag = "1")]
	pub path:           ::prost::alloc::string::String,
	#[prost(string, optional, tag = "6")]
	pub stream_content: ::core::option::Option<::prost::alloc::string::String>,
}

pub mod edit_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::EditSuccess),
		#[prost(message, tag = "2")]
		FileNotFound(super::EditFileNotFound),
		#[prost(message, tag = "3")]
		ReadPermissionDenied(super::EditReadPermissionDenied),
		#[prost(message, tag = "4")]
		WritePermissionDenied(super::EditWritePermissionDenied),
		#[prost(message, tag = "6")]
		Rejected(super::EditRejected),
		#[prost(message, tag = "7")]
		Error(super::EditError),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct EditResult {
	#[prost(oneof = "edit_result::Result", tags = "1, 2, 3, 4, 6, 7")]
	pub result: ::core::option::Option<edit_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct EditSuccess {
	#[prost(string, tag = "1")]
	pub path:                     ::prost::alloc::string::String,
	#[prost(int32, optional, tag = "3")]
	pub lines_added:              ::core::option::Option<i32>,
	#[prost(int32, optional, tag = "4")]
	pub lines_removed:            ::core::option::Option<i32>,
	#[prost(string, optional, tag = "5")]
	pub diff_string:              ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "6")]
	pub before_full_file_content: ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, tag = "7")]
	pub after_full_file_content:  ::prost::alloc::string::String,
	#[prost(string, optional, tag = "8")]
	pub message:                  ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct EditFileNotFound {
	#[prost(string, tag = "1")]
	pub path: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct EditReadPermissionDenied {
	#[prost(string, tag = "1")]
	pub path: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct EditWritePermissionDenied {
	#[prost(string, tag = "1")]
	pub path:        ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub error:       ::prost::alloc::string::String,
	#[prost(bool, tag = "3")]
	pub is_readonly: bool,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct EditRejected {
	#[prost(string, tag = "1")]
	pub path:   ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub reason: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct EditError {
	#[prost(string, tag = "1")]
	pub path:                ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub error:               ::prost::alloc::string::String,
	#[prost(string, optional, tag = "5")]
	pub model_visible_error: ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct EditToolCall {
	#[prost(message, optional, tag = "1")]
	pub args:   ::core::option::Option<EditArgs>,
	#[prost(message, optional, tag = "2")]
	pub result: ::core::option::Option<EditResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct EditToolCallDelta {
	#[prost(string, tag = "1")]
	pub stream_content_delta: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExaFetchArgs {
	#[prost(string, repeated, tag = "1")]
	pub ids:          ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
	#[prost(string, tag = "2")]
	pub tool_call_id: ::prost::alloc::string::String,
}

pub mod exa_fetch_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::ExaFetchSuccess),
		#[prost(message, tag = "2")]
		Error(super::ExaFetchError),
		#[prost(message, tag = "3")]
		Rejected(super::ExaFetchRejected),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExaFetchResult {
	#[prost(oneof = "exa_fetch_result::Result", tags = "1, 2, 3")]
	pub result: ::core::option::Option<exa_fetch_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExaFetchSuccess {
	#[prost(message, repeated, tag = "1")]
	pub contents: ::prost::alloc::vec::Vec<ExaFetchContent>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExaFetchError {
	#[prost(string, tag = "1")]
	pub error: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExaFetchRejected {
	#[prost(string, tag = "1")]
	pub reason: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExaFetchContent {
	#[prost(string, tag = "1")]
	pub title:          ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub url:            ::prost::alloc::string::String,
	#[prost(string, tag = "3")]
	pub text:           ::prost::alloc::string::String,
	#[prost(string, tag = "4")]
	pub published_date: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExaFetchToolCall {
	#[prost(message, optional, tag = "1")]
	pub args:   ::core::option::Option<ExaFetchArgs>,
	#[prost(message, optional, tag = "2")]
	pub result: ::core::option::Option<ExaFetchResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExaFetchRequestQuery {
	#[prost(message, optional, tag = "1")]
	pub args: ::core::option::Option<ExaFetchArgs>,
}

pub mod exa_fetch_request_response {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Approved(super::ExaFetchRequestResponse_Approved),
		#[prost(message, tag = "2")]
		Rejected(super::ExaFetchRequestResponse_Rejected),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExaFetchRequestResponse {
	#[prost(oneof = "exa_fetch_request_response::Result", tags = "1, 2")]
	pub result: ::core::option::Option<exa_fetch_request_response::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExaFetchRequestResponse_Approved {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExaFetchRequestResponse_Rejected {
	#[prost(string, tag = "1")]
	pub reason: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExaSearchArgs {
	#[prost(string, tag = "1")]
	pub query:        ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub r#type:       ::prost::alloc::string::String,
	#[prost(int32, tag = "3")]
	pub num_results:  i32,
	#[prost(string, tag = "4")]
	pub tool_call_id: ::prost::alloc::string::String,
}

pub mod exa_search_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::ExaSearchSuccess),
		#[prost(message, tag = "2")]
		Error(super::ExaSearchError),
		#[prost(message, tag = "3")]
		Rejected(super::ExaSearchRejected),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExaSearchResult {
	#[prost(oneof = "exa_search_result::Result", tags = "1, 2, 3")]
	pub result: ::core::option::Option<exa_search_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExaSearchSuccess {
	#[prost(message, repeated, tag = "1")]
	pub references: ::prost::alloc::vec::Vec<ExaSearchReference>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExaSearchError {
	#[prost(string, tag = "1")]
	pub error: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExaSearchRejected {
	#[prost(string, tag = "1")]
	pub reason: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExaSearchReference {
	#[prost(string, tag = "1")]
	pub title:          ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub url:            ::prost::alloc::string::String,
	#[prost(string, tag = "3")]
	pub text:           ::prost::alloc::string::String,
	#[prost(string, tag = "4")]
	pub published_date: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExaSearchToolCall {
	#[prost(message, optional, tag = "1")]
	pub args:   ::core::option::Option<ExaSearchArgs>,
	#[prost(message, optional, tag = "2")]
	pub result: ::core::option::Option<ExaSearchResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExaSearchRequestQuery {
	#[prost(message, optional, tag = "1")]
	pub args: ::core::option::Option<ExaSearchArgs>,
}

pub mod exa_search_request_response {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Approved(super::ExaSearchRequestResponse_Approved),
		#[prost(message, tag = "2")]
		Rejected(super::ExaSearchRequestResponse_Rejected),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExaSearchRequestResponse {
	#[prost(oneof = "exa_search_request_response::Result", tags = "1, 2")]
	pub result: ::core::option::Option<exa_search_request_response::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExaSearchRequestResponse_Approved {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExaSearchRequestResponse_Rejected {
	#[prost(string, tag = "1")]
	pub reason: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExecClientStreamClose {
	#[prost(uint32, tag = "1")]
	pub id: u32,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExecClientThrow {
	#[prost(uint32, tag = "1")]
	pub id:          u32,
	#[prost(string, tag = "2")]
	pub error:       ::prost::alloc::string::String,
	#[prost(string, optional, tag = "3")]
	pub stack_trace: ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "4")]
	pub error_code:  ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExecClientHeartbeat {
	#[prost(uint32, tag = "1")]
	pub id: u32,
}

pub mod exec_client_control_message {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Message {
		#[prost(message, tag = "1")]
		StreamClose(super::ExecClientStreamClose),
		#[prost(message, tag = "2")]
		Throw(super::ExecClientThrow),
		#[prost(message, tag = "3")]
		Heartbeat(super::ExecClientHeartbeat),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExecClientControlMessage {
	#[prost(oneof = "exec_client_control_message::Message", tags = "1, 2, 3")]
	pub message: ::core::option::Option<exec_client_control_message::Message>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SpanContext {
	#[prost(string, tag = "1")]
	pub trace_id:    ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub span_id:     ::prost::alloc::string::String,
	#[prost(uint32, optional, tag = "3")]
	pub trace_flags: ::core::option::Option<u32>,
	#[prost(string, optional, tag = "4")]
	pub trace_state: ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AbortArgs {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AbortResult {}

pub mod exec_server_message {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Message {
		#[prost(message, tag = "2")]
		ShellArgs(super::ShellArgs),
		#[prost(message, tag = "3")]
		WriteArgs(super::WriteArgs),
		#[prost(message, tag = "4")]
		DeleteArgs(super::DeleteArgs),
		#[prost(message, tag = "5")]
		GrepArgs(super::GrepArgs),
		#[prost(message, tag = "7")]
		ReadArgs(super::ReadArgs),
		#[prost(message, tag = "8")]
		LsArgs(super::LsArgs),
		#[prost(message, tag = "9")]
		DiagnosticsArgs(super::DiagnosticsArgs),
		#[prost(message, tag = "10")]
		RequestContextArgs(super::RequestContextArgs),
		#[prost(message, tag = "11")]
		McpArgs(super::McpArgs),
		#[prost(message, tag = "14")]
		ShellStreamArgs(super::ShellArgs),
		#[prost(message, tag = "16")]
		BackgroundShellSpawnArgs(super::BackgroundShellSpawnArgs),
		#[prost(message, tag = "17")]
		ListMcpResourcesExecArgs(super::ListMcpResourcesExecArgs),
		#[prost(message, tag = "18")]
		ReadMcpResourceExecArgs(super::ReadMcpResourceExecArgs),
		#[prost(message, tag = "20")]
		FetchArgs(super::FetchArgs),
		#[prost(message, tag = "21")]
		RecordScreenArgs(super::RecordScreenArgs),
		#[prost(message, tag = "22")]
		ComputerUseArgs(super::ComputerUseArgs),
		#[prost(message, tag = "23")]
		WriteShellStdinArgs(super::WriteShellStdinArgs),
		#[prost(message, tag = "29")]
		RedactedReadArgs(super::ReadArgs),
		#[prost(message, tag = "36")]
		McpStateExecArgs(super::McpStateExecArgs),
		#[prost(message, tag = "27")]
		ExecuteHookArgs(super::ExecuteHookArgs),
		#[prost(message, tag = "28")]
		SubagentArgs(super::SubagentArgs),
		#[prost(message, tag = "30")]
		ForceBackgroundShellArgs(super::ForceBackgroundShellArgs),
		#[prost(message, tag = "31")]
		ForceBackgroundSubagentArgs(super::ForceBackgroundSubagentArgs),
		#[prost(message, tag = "37")]
		SubagentAwaitArgs(super::SubagentAwaitArgs),
		#[prost(message, tag = "38")]
		SmartModeClassifierArgs(super::SmartModeClassifierArgs),
		#[prost(message, tag = "40")]
		CanvasDiagnosticsArgs(super::CanvasDiagnosticsArgs),
		#[prost(message, tag = "41")]
		ShellAllowlistPrecheckArgs(super::ShellAllowlistPrecheckArgs),
		#[prost(message, tag = "42")]
		McpAllowlistPrecheckArgs(super::McpAllowlistPrecheckArgs),
		#[prost(message, tag = "43")]
		WebFetchAllowlistPrecheckArgs(super::WebFetchAllowlistPrecheckArgs),
		#[prost(message, tag = "44")]
		GitDiffRequest(super::GetDiffRequest),
		#[prost(message, tag = "45")]
		PiReadArgs(super::PiReadExecArgs),
		#[prost(message, tag = "46")]
		PiBashArgs(super::PiBashExecArgs),
		#[prost(message, tag = "47")]
		PiEditArgs(super::PiEditExecArgs),
		#[prost(message, tag = "48")]
		PiWriteArgs(super::PiWriteExecArgs),
		#[prost(message, tag = "49")]
		PiGrepArgs(super::PiGrepExecArgs),
		#[prost(message, tag = "50")]
		PiFindArgs(super::PiFindExecArgs),
		#[prost(message, tag = "51")]
		PiLsArgs(super::PiLsExecArgs),
		#[prost(message, tag = "52")]
		MiniSweAgentBashArgs(super::ShellArgs),
		#[prost(message, tag = "53")]
		ConversationSearchArgs(super::ConversationSearchArgs),
		#[prost(message, tag = "54")]
		AgentStoreConflictArgs(super::AgentStoreConflictArgs),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExecServerMessage {
	#[prost(uint32, tag = "1")]
	pub id: u32,
	#[prost(string, tag = "15")]
	pub exec_id: ::prost::alloc::string::String,
	#[prost(message, optional, tag = "19")]
	pub span_context: ::core::option::Option<SpanContext>,
	#[prost(
		oneof = "exec_server_message::Message",
		tags = "2, 3, 4, 5, 7, 8, 9, 10, 11, 14, 16, 17, 18, 20, 21, 22, 23, 29, 36, 27, 28, 30, \
		        31, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54"
	)]
	pub message: ::core::option::Option<exec_server_message::Message>,
	#[prost(bool, optional, tag = "55")]
	pub accept_hook_additional_contexts: ::core::option::Option<bool>,
}

pub mod exec_client_message {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Message {
		#[prost(message, boxed, tag = "2")]
		ShellResult(::prost::alloc::boxed::Box<super::ShellResult>),
		#[prost(message, tag = "3")]
		WriteResult(super::WriteResult),
		#[prost(message, tag = "4")]
		DeleteResult(super::DeleteResult),
		#[prost(message, tag = "5")]
		GrepResult(super::GrepResult),
		#[prost(message, tag = "7")]
		ReadResult(super::ReadResult),
		#[prost(message, tag = "8")]
		LsResult(super::LsResult),
		#[prost(message, tag = "9")]
		DiagnosticsResult(super::DiagnosticsResult),
		#[prost(message, tag = "10")]
		RequestContextResult(super::RequestContextResult),
		#[prost(message, tag = "11")]
		McpResult(super::McpResult),
		#[prost(message, tag = "14")]
		ShellStream(super::ShellStream),
		#[prost(message, tag = "16")]
		BackgroundShellSpawnResult(super::BackgroundShellSpawnResult),
		#[prost(message, tag = "17")]
		ListMcpResourcesExecResult(super::ListMcpResourcesExecResult),
		#[prost(message, tag = "18")]
		ReadMcpResourceExecResult(super::ReadMcpResourceExecResult),
		#[prost(message, tag = "20")]
		FetchResult(super::FetchResult),
		#[prost(message, tag = "21")]
		RecordScreenResult(super::RecordScreenResult),
		#[prost(message, tag = "22")]
		ComputerUseResult(super::ComputerUseResult),
		#[prost(message, tag = "23")]
		WriteShellStdinResult(super::WriteShellStdinResult),
		#[prost(message, tag = "29")]
		RedactedReadResult(super::ReadResult),
		#[prost(message, tag = "36")]
		McpStateExecResult(super::McpStateExecResult),
		#[prost(message, tag = "27")]
		ExecuteHookResult(super::ExecuteHookResult),
		#[prost(message, tag = "28")]
		SubagentResult(super::SubagentResult),
		#[prost(message, tag = "30")]
		ForceBackgroundShellResult(super::ForceBackgroundShellResult),
		#[prost(message, tag = "31")]
		ForceBackgroundSubagentResult(super::ForceBackgroundSubagentResult),
		#[prost(message, tag = "37")]
		SubagentAwaitResult(super::SubagentAwaitResult),
		#[prost(message, tag = "38")]
		SmartModeClassifierResult(super::SmartModeClassifierResult),
		#[prost(message, tag = "40")]
		CanvasDiagnosticsResult(super::CanvasDiagnosticsResult),
		#[prost(message, tag = "41")]
		ShellAllowlistPrecheckResult(super::ShellAllowlistPrecheckResult),
		#[prost(message, tag = "42")]
		McpAllowlistPrecheckResult(super::McpAllowlistPrecheckResult),
		#[prost(message, tag = "43")]
		WebFetchAllowlistPrecheckResult(super::WebFetchAllowlistPrecheckResult),
		#[prost(message, tag = "44")]
		GitDiffResponse(super::GetDiffResponse),
		#[prost(message, tag = "46")]
		PiReadResult(super::PiReadExecResult),
		#[prost(message, tag = "47")]
		PiBashResult(super::PiBashExecResult),
		#[prost(message, tag = "48")]
		PiEditResult(super::PiEditExecResult),
		#[prost(message, tag = "49")]
		PiWriteResult(super::PiWriteExecResult),
		#[prost(message, tag = "50")]
		PiGrepResult(super::PiGrepExecResult),
		#[prost(message, tag = "51")]
		PiFindResult(super::PiFindExecResult),
		#[prost(message, tag = "52")]
		PiLsResult(super::PiLsExecResult),
		#[prost(message, tag = "53")]
		ConversationSearchResult(super::ConversationSearchResult),
		#[prost(message, tag = "54")]
		AgentStoreConflictResult(super::AgentStoreConflictResult),
		#[prost(message, tag = "55")]
		MiniSweAgentBashResult(super::ShellResult),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExecClientMessage {
	#[prost(uint32, tag = "1")]
	pub id: u32,
	#[prost(string, tag = "15")]
	pub exec_id: ::prost::alloc::string::String,
	#[prost(
		oneof = "exec_client_message::Message",
		tags = "2, 3, 4, 5, 7, 8, 9, 10, 11, 14, 16, 17, 18, 20, 21, 22, 23, 29, 36, 27, 28, 30, \
		        31, 37, 38, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55"
	)]
	pub message: ::core::option::Option<exec_client_message::Message>,
	#[prost(int32, optional, tag = "39")]
	pub local_execution_time_ms: ::core::option::Option<i32>,
	#[prost(message, repeated, tag = "45")]
	pub hook_additional_contexts: ::prost::alloc::vec::Vec<HookAdditionalContext>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FetchArgs {
	#[prost(string, tag = "1")]
	pub url:          ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub tool_call_id: ::prost::alloc::string::String,
}

pub mod fetch_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::FetchSuccess),
		#[prost(message, tag = "2")]
		Error(super::FetchError),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FetchResult {
	#[prost(oneof = "fetch_result::Result", tags = "1, 2")]
	pub result: ::core::option::Option<fetch_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FetchSuccess {
	#[prost(string, tag = "1")]
	pub url:          ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub content:      ::prost::alloc::string::String,
	#[prost(int32, tag = "3")]
	pub status_code:  i32,
	#[prost(string, tag = "4")]
	pub content_type: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FetchError {
	#[prost(string, tag = "1")]
	pub url:   ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub error: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GenerateImageArgs {
	#[prost(string, tag = "1")]
	pub description:           ::prost::alloc::string::String,
	#[prost(string, optional, tag = "2")]
	pub file_path:             ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, repeated, tag = "5")]
	pub reference_image_paths: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
}

pub mod generate_image_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::GenerateImageSuccess),
		#[prost(message, tag = "2")]
		Error(super::GenerateImageError),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GenerateImageResult {
	#[prost(oneof = "generate_image_result::Result", tags = "1, 2")]
	pub result: ::core::option::Option<generate_image_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GenerateImageSuccess {
	#[prost(string, tag = "1")]
	pub file_path:  ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub image_data: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GenerateImageError {
	#[prost(string, tag = "1")]
	pub error: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GenerateImageToolCall {
	#[prost(message, optional, tag = "1")]
	pub args:   ::core::option::Option<GenerateImageArgs>,
	#[prost(message, optional, tag = "2")]
	pub result: ::core::option::Option<GenerateImageResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GrepArgs {
	#[prost(string, tag = "1")]
	pub pattern:          ::prost::alloc::string::String,
	#[prost(string, optional, tag = "2")]
	pub path:             ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "3")]
	pub glob:             ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "4")]
	pub output_mode:      ::core::option::Option<::prost::alloc::string::String>,
	#[prost(int32, optional, tag = "5")]
	pub context_before:   ::core::option::Option<i32>,
	#[prost(int32, optional, tag = "6")]
	pub context_after:    ::core::option::Option<i32>,
	#[prost(int32, optional, tag = "7")]
	pub context:          ::core::option::Option<i32>,
	#[prost(bool, optional, tag = "8")]
	pub case_insensitive: ::core::option::Option<bool>,
	#[prost(string, optional, tag = "9")]
	pub r#type:           ::core::option::Option<::prost::alloc::string::String>,
	#[prost(int32, optional, tag = "10")]
	pub head_limit:       ::core::option::Option<i32>,
	#[prost(bool, optional, tag = "11")]
	pub multiline:        ::core::option::Option<bool>,
	#[prost(string, optional, tag = "12")]
	pub sort:             ::core::option::Option<::prost::alloc::string::String>,
	#[prost(bool, optional, tag = "13")]
	pub sort_ascending:   ::core::option::Option<bool>,
	#[prost(string, tag = "14")]
	pub tool_call_id:     ::prost::alloc::string::String,
	#[prost(message, optional, tag = "15")]
	pub sandbox_policy:   ::core::option::Option<SandboxPolicy>,
	#[prost(int32, optional, tag = "16")]
	pub offset:           ::core::option::Option<i32>,
}

pub mod grep_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::GrepSuccess),
		#[prost(message, tag = "2")]
		Error(super::GrepError),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GrepResult {
	#[prost(oneof = "grep_result::Result", tags = "1, 2")]
	pub result: ::core::option::Option<grep_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GrepError {
	#[prost(string, tag = "1")]
	pub error: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GrepSuccess {
	#[prost(string, tag = "1")]
	pub pattern:              ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub path:                 ::prost::alloc::string::String,
	#[prost(string, tag = "3")]
	pub output_mode:          ::prost::alloc::string::String,
	#[prost(map = "string, message", tag = "4")]
	pub workspace_results:
		::std::collections::HashMap<::prost::alloc::string::String, GrepUnionResult>,
	#[prost(message, optional, tag = "5")]
	pub active_editor_result: ::core::option::Option<GrepUnionResult>,
}

pub mod grep_union_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Count(super::GrepCountResult),
		#[prost(message, tag = "2")]
		Files(super::GrepFilesResult),
		#[prost(message, tag = "3")]
		Content(super::GrepContentResult),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GrepUnionResult {
	#[prost(oneof = "grep_union_result::Result", tags = "1, 2, 3")]
	pub result: ::core::option::Option<grep_union_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GrepCountResult {
	#[prost(message, repeated, tag = "1")]
	pub counts:             ::prost::alloc::vec::Vec<GrepFileCount>,
	#[prost(int32, tag = "2")]
	pub total_files:        i32,
	#[prost(int32, tag = "3")]
	pub total_matches:      i32,
	#[prost(bool, tag = "4")]
	pub client_truncated:   bool,
	#[prost(bool, tag = "5")]
	pub ripgrep_truncated:  bool,
	#[prost(int32, optional, tag = "6")]
	pub head_limit_applied: ::core::option::Option<i32>,
	#[prost(int32, optional, tag = "7")]
	pub offset_applied:     ::core::option::Option<i32>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GrepFileCount {
	#[prost(string, tag = "1")]
	pub file:  ::prost::alloc::string::String,
	#[prost(int32, tag = "2")]
	pub count: i32,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GrepFilesResult {
	#[prost(string, repeated, tag = "1")]
	pub files:              ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
	#[prost(int32, tag = "2")]
	pub total_files:        i32,
	#[prost(bool, tag = "3")]
	pub client_truncated:   bool,
	#[prost(bool, tag = "4")]
	pub ripgrep_truncated:  bool,
	#[prost(int32, optional, tag = "5")]
	pub head_limit_applied: ::core::option::Option<i32>,
	#[prost(int32, optional, tag = "6")]
	pub offset_applied:     ::core::option::Option<i32>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GrepContentResult {
	#[prost(message, repeated, tag = "1")]
	pub matches:             ::prost::alloc::vec::Vec<GrepFileMatch>,
	#[prost(int32, tag = "2")]
	pub total_lines:         i32,
	#[prost(int32, tag = "3")]
	pub total_matched_lines: i32,
	#[prost(bool, tag = "4")]
	pub client_truncated:    bool,
	#[prost(bool, tag = "5")]
	pub ripgrep_truncated:   bool,
	#[prost(int32, optional, tag = "6")]
	pub head_limit_applied:  ::core::option::Option<i32>,
	#[prost(int32, optional, tag = "7")]
	pub offset_applied:      ::core::option::Option<i32>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GrepFileMatch {
	#[prost(string, tag = "1")]
	pub file:    ::prost::alloc::string::String,
	#[prost(message, repeated, tag = "2")]
	pub matches: ::prost::alloc::vec::Vec<GrepContentMatch>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GrepContentMatch {
	#[prost(int32, tag = "1")]
	pub line_number:       i32,
	#[prost(string, tag = "2")]
	pub content:           ::prost::alloc::string::String,
	#[prost(bool, tag = "3")]
	pub content_truncated: bool,
	#[prost(bool, tag = "4")]
	pub is_context_line:   bool,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GrepStream {
	#[prost(string, tag = "1")]
	pub pattern: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GrepToolCall {
	#[prost(message, optional, tag = "1")]
	pub args:   ::core::option::Option<GrepArgs>,
	#[prost(message, optional, tag = "2")]
	pub result: ::core::option::Option<GrepResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetBlobArgs {
	#[prost(bytes = "vec", tag = "1")]
	pub blob_id: ::prost::alloc::vec::Vec<u8>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetBlobResult {
	#[prost(bytes = "vec", optional, tag = "1")]
	pub blob_data: ::core::option::Option<::prost::alloc::vec::Vec<u8>>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SetBlobArgs {
	#[prost(bytes = "vec", tag = "1")]
	pub blob_id:   ::prost::alloc::vec::Vec<u8>,
	#[prost(bytes = "vec", tag = "2")]
	pub blob_data: ::prost::alloc::vec::Vec<u8>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SetBlobResult {
	#[prost(message, optional, tag = "1")]
	pub error: ::core::option::Option<Error>,
}

pub mod kv_server_message {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Message {
		#[prost(message, tag = "2")]
		GetBlobArgs(super::GetBlobArgs),
		#[prost(message, tag = "3")]
		SetBlobArgs(super::SetBlobArgs),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct KvServerMessage {
	#[prost(uint32, tag = "1")]
	pub id:           u32,
	#[prost(message, optional, tag = "4")]
	pub span_context: ::core::option::Option<SpanContext>,
	#[prost(oneof = "kv_server_message::Message", tags = "2, 3")]
	pub message:      ::core::option::Option<kv_server_message::Message>,
}

pub mod kv_client_message {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Message {
		#[prost(message, tag = "2")]
		GetBlobResult(super::GetBlobResult),
		#[prost(message, tag = "3")]
		SetBlobResult(super::SetBlobResult),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct KvClientMessage {
	#[prost(uint32, tag = "1")]
	pub id:      u32,
	#[prost(oneof = "kv_client_message::Message", tags = "2, 3")]
	pub message: ::core::option::Option<kv_client_message::Message>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct LsArgs {
	#[prost(string, tag = "1")]
	pub path:           ::prost::alloc::string::String,
	#[prost(string, repeated, tag = "2")]
	pub ignore:         ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
	#[prost(string, tag = "3")]
	pub tool_call_id:   ::prost::alloc::string::String,
	#[prost(message, optional, tag = "4")]
	pub sandbox_policy: ::core::option::Option<SandboxPolicy>,
	#[prost(uint32, optional, tag = "5")]
	pub timeout_ms:     ::core::option::Option<u32>,
}

pub mod ls_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::LsSuccess),
		#[prost(message, tag = "2")]
		Error(super::LsError),
		#[prost(message, tag = "3")]
		Rejected(super::LsRejected),
		#[prost(message, tag = "4")]
		Timeout(super::LsTimeout),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct LsResult {
	#[prost(oneof = "ls_result::Result", tags = "1, 2, 3, 4")]
	pub result: ::core::option::Option<ls_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct LsSuccess {
	#[prost(message, optional, tag = "1")]
	pub directory_tree_root: ::core::option::Option<LsDirectoryTreeNode>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct LsDirectoryTreeNode {
	#[prost(string, tag = "1")]
	pub abs_path: ::prost::alloc::string::String,
	#[prost(message, repeated, tag = "2")]
	pub children_dirs: ::prost::alloc::vec::Vec<LsDirectoryTreeNode>,
	#[prost(message, repeated, tag = "3")]
	pub children_files: ::prost::alloc::vec::Vec<LsDirectoryTreeNode_File>,
	#[prost(bool, tag = "4")]
	pub children_were_processed: bool,
	#[prost(map = "string, int32", tag = "5")]
	pub full_subtree_extension_counts:
		::std::collections::HashMap<::prost::alloc::string::String, i32>,
	#[prost(int32, tag = "6")]
	pub num_files: i32,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct LsDirectoryTreeNode_File {
	#[prost(string, tag = "1")]
	pub name:              ::prost::alloc::string::String,
	#[prost(message, optional, tag = "2")]
	pub terminal_metadata: ::core::option::Option<TerminalMetadata>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct LsError {
	#[prost(string, tag = "1")]
	pub path:  ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub error: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct LsRejected {
	#[prost(string, tag = "1")]
	pub path:   ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub reason: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct LsTimeout {
	#[prost(message, optional, tag = "1")]
	pub directory_tree_root: ::core::option::Option<LsDirectoryTreeNode>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TerminalMetadata {
	#[prost(string, optional, tag = "1")]
	pub cwd:              ::core::option::Option<::prost::alloc::string::String>,
	#[prost(message, repeated, tag = "2")]
	pub last_commands:    ::prost::alloc::vec::Vec<TerminalMetadata_Command>,
	#[prost(int64, optional, tag = "3")]
	pub last_modified_ms: ::core::option::Option<i64>,
	#[prost(message, optional, tag = "4")]
	pub current_command:  ::core::option::Option<TerminalMetadata_Command>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TerminalMetadata_Command {
	#[prost(string, tag = "1")]
	pub command:      ::prost::alloc::string::String,
	#[prost(int32, optional, tag = "2")]
	pub exit_code:    ::core::option::Option<i32>,
	#[prost(int64, optional, tag = "3")]
	pub timestamp_ms: ::core::option::Option<i64>,
	#[prost(int64, optional, tag = "4")]
	pub duration_ms:  ::core::option::Option<i64>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct LsToolCall {
	#[prost(message, optional, tag = "1")]
	pub args:   ::core::option::Option<LsArgs>,
	#[prost(message, optional, tag = "2")]
	pub result: ::core::option::Option<LsResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct McpArgs {
	#[prost(string, tag = "1")]
	pub name:                     ::prost::alloc::string::String,
	#[prost(map = "string, bytes", tag = "2")]
	pub args:
		::std::collections::HashMap<::prost::alloc::string::String, ::prost::alloc::vec::Vec<u8>>,
	#[prost(string, tag = "3")]
	pub tool_call_id:             ::prost::alloc::string::String,
	#[prost(string, tag = "4")]
	pub provider_identifier:      ::prost::alloc::string::String,
	#[prost(string, tag = "5")]
	pub tool_name:                ::prost::alloc::string::String,
	#[prost(message, optional, tag = "6")]
	pub smart_mode_approval:      ::core::option::Option<SmartModeApproval>,
	#[prost(bool, tag = "7")]
	pub smart_mode_approval_only: bool,
	#[prost(bool, tag = "8")]
	pub skip_approval:            bool,
	#[prost(string, tag = "9")]
	pub server_identifier:        ::prost::alloc::string::String,
}

pub mod mcp_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::McpSuccess),
		#[prost(message, tag = "2")]
		Error(super::McpError),
		#[prost(message, tag = "3")]
		Rejected(super::McpRejected),
		#[prost(message, tag = "4")]
		PermissionDenied(super::McpPermissionDenied),
		#[prost(message, tag = "5")]
		ToolNotFound(super::McpToolNotFound),
		#[prost(message, tag = "6")]
		ServerNotFound(super::McpServerNotFound),
		#[prost(message, tag = "7")]
		Approved(super::McpApproved),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct McpResult {
	#[prost(oneof = "mcp_result::Result", tags = "1, 2, 3, 4, 5, 6, 7")]
	pub result: ::core::option::Option<mcp_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct McpToolNotFound {
	#[prost(string, tag = "1")]
	pub name:            ::prost::alloc::string::String,
	#[prost(string, repeated, tag = "2")]
	pub available_tools: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct McpTextContent {
	#[prost(string, tag = "1")]
	pub text:            ::prost::alloc::string::String,
	#[prost(message, optional, tag = "2")]
	pub output_location: ::core::option::Option<OutputLocation>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct McpImageContent {
	#[prost(bytes = "vec", tag = "1")]
	pub data:      ::prost::alloc::vec::Vec<u8>,
	#[prost(string, tag = "2")]
	pub mime_type: ::prost::alloc::string::String,
}

pub mod mcp_tool_result_content_item {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Content {
		#[prost(message, tag = "1")]
		Text(super::McpTextContent),
		#[prost(message, tag = "2")]
		Image(super::McpImageContent),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct McpToolResultContentItem {
	#[prost(oneof = "mcp_tool_result_content_item::Content", tags = "1, 2")]
	pub content: ::core::option::Option<mcp_tool_result_content_item::Content>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct McpSuccess {
	#[prost(message, repeated, tag = "1")]
	pub content:  ::prost::alloc::vec::Vec<McpToolResultContentItem>,
	#[prost(bool, tag = "2")]
	pub is_error: bool,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct McpError {
	#[prost(string, tag = "1")]
	pub error: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct McpRejected {
	#[prost(string, tag = "1")]
	pub reason:      ::prost::alloc::string::String,
	#[prost(bool, tag = "2")]
	pub is_readonly: bool,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct McpPermissionDenied {
	#[prost(string, tag = "1")]
	pub error:       ::prost::alloc::string::String,
	#[prost(bool, tag = "2")]
	pub is_readonly: bool,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ListMcpResourcesExecArgs {
	#[prost(string, optional, tag = "1")]
	pub server: ::core::option::Option<::prost::alloc::string::String>,
}

pub mod list_mcp_resources_exec_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::ListMcpResourcesSuccess),
		#[prost(message, tag = "2")]
		Error(super::ListMcpResourcesError),
		#[prost(message, tag = "3")]
		Rejected(super::ListMcpResourcesRejected),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ListMcpResourcesExecResult {
	#[prost(oneof = "list_mcp_resources_exec_result::Result", tags = "1, 2, 3")]
	pub result: ::core::option::Option<list_mcp_resources_exec_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ListMcpResourcesExecResult_McpResource {
	#[prost(string, tag = "1")]
	pub uri:         ::prost::alloc::string::String,
	#[prost(string, optional, tag = "2")]
	pub name:        ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "3")]
	pub description: ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "4")]
	pub mime_type:   ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, tag = "5")]
	pub server:      ::prost::alloc::string::String,
	#[prost(map = "string, string", tag = "6")]
	pub annotations:
		::std::collections::HashMap<::prost::alloc::string::String, ::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ListMcpResourcesSuccess {
	#[prost(message, repeated, tag = "1")]
	pub resources: ::prost::alloc::vec::Vec<ListMcpResourcesExecResult_McpResource>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ListMcpResourcesError {
	#[prost(string, tag = "1")]
	pub error: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ListMcpResourcesRejected {
	#[prost(string, tag = "1")]
	pub reason: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReadMcpResourceExecArgs {
	#[prost(string, tag = "1")]
	pub server:              ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub uri:                 ::prost::alloc::string::String,
	#[prost(string, optional, tag = "3")]
	pub download_path:       ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, tag = "4")]
	pub tool_call_id:        ::prost::alloc::string::String,
	#[prost(message, optional, tag = "5")]
	pub smart_mode_approval: ::core::option::Option<SmartModeApproval>,
}

pub mod read_mcp_resource_exec_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::ReadMcpResourceSuccess),
		#[prost(message, tag = "2")]
		Error(super::ReadMcpResourceError),
		#[prost(message, tag = "3")]
		Rejected(super::ReadMcpResourceRejected),
		#[prost(message, tag = "4")]
		NotFound(super::ReadMcpResourceNotFound),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReadMcpResourceExecResult {
	#[prost(oneof = "read_mcp_resource_exec_result::Result", tags = "1, 2, 3, 4")]
	pub result: ::core::option::Option<read_mcp_resource_exec_result::Result>,
}

pub mod read_mcp_resource_success {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Content {
		#[prost(string, tag = "5")]
		Text(::prost::alloc::string::String),
		#[prost(bytes = "vec", tag = "6")]
		Blob(::prost::alloc::vec::Vec<u8>),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReadMcpResourceSuccess {
	#[prost(string, tag = "1")]
	pub uri:             ::prost::alloc::string::String,
	#[prost(string, optional, tag = "2")]
	pub name:            ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "3")]
	pub description:     ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "4")]
	pub mime_type:       ::core::option::Option<::prost::alloc::string::String>,
	#[prost(map = "string, string", tag = "7")]
	pub annotations:
		::std::collections::HashMap<::prost::alloc::string::String, ::prost::alloc::string::String>,
	#[prost(string, optional, tag = "8")]
	pub download_path:   ::core::option::Option<::prost::alloc::string::String>,
	#[prost(oneof = "read_mcp_resource_success::Content", tags = "5, 6")]
	pub content:         ::core::option::Option<read_mcp_resource_success::Content>,
	#[prost(message, optional, tag = "9")]
	pub output_location: ::core::option::Option<OutputLocation>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReadMcpResourceError {
	#[prost(string, tag = "1")]
	pub uri:   ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub error: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReadMcpResourceRejected {
	#[prost(string, tag = "1")]
	pub uri:    ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub reason: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReadMcpResourceNotFound {
	#[prost(string, tag = "1")]
	pub uri: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct McpToolDefinition {
	#[prost(string, tag = "1")]
	pub name:                ::prost::alloc::string::String,
	#[prost(string, tag = "4")]
	pub provider_identifier: ::prost::alloc::string::String,
	#[prost(string, tag = "5")]
	pub tool_name:           ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub description:         ::prost::alloc::string::String,
	#[prost(bytes = "vec", tag = "3")]
	pub input_schema:        ::prost::alloc::vec::Vec<u8>,
	#[prost(string, optional, tag = "6")]
	pub input_schema_json:   ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct McpTools {
	#[prost(message, repeated, tag = "1")]
	pub mcp_tools: ::prost::alloc::vec::Vec<McpToolDefinition>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct McpInstructions {
	#[prost(string, tag = "1")]
	pub server_name:       ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub instructions:      ::prost::alloc::string::String,
	#[prost(string, tag = "3")]
	pub server_identifier: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct McpDescriptor {
	#[prost(string, tag = "1")]
	pub server_name:             ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub server_identifier:       ::prost::alloc::string::String,
	#[prost(string, optional, tag = "3")]
	pub folder_path:             ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "4")]
	pub server_use_instructions: ::core::option::Option<::prost::alloc::string::String>,
	#[prost(message, repeated, tag = "5")]
	pub tools:                   ::prost::alloc::vec::Vec<McpToolDescriptor>,
	#[prost(string, optional, tag = "7")]
	pub plugin:                  ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "8")]
	pub marketplace:             ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "9")]
	pub plugin_db_id:            ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "10")]
	pub marketplace_id:          ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct McpToolDescriptor {
	#[prost(string, tag = "1")]
	pub tool_name:         ::prost::alloc::string::String,
	#[prost(string, optional, tag = "2")]
	pub definition_path:   ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "3")]
	pub description:       ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "5")]
	pub input_schema_json: ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct McpFileSystemOptions {
	#[prost(bool, tag = "1")]
	pub enabled:               bool,
	#[prost(string, tag = "2")]
	pub workspace_project_dir: ::prost::alloc::string::String,
	#[prost(message, repeated, tag = "3")]
	pub mcp_descriptors:       ::prost::alloc::vec::Vec<McpDescriptor>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReadArgs {
	#[prost(string, tag = "1")]
	pub path:          ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub tool_call_id:  ::prost::alloc::string::String,
	#[prost(int32, optional, tag = "4")]
	pub offset:        ::core::option::Option<i32>,
	#[prost(uint32, optional, tag = "5")]
	pub limit:         ::core::option::Option<u32>,
	#[prost(string, optional, tag = "6")]
	pub encoding_hint: ::core::option::Option<::prost::alloc::string::String>,
}

pub mod read_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::ReadSuccess),
		#[prost(message, tag = "2")]
		Error(super::ReadError),
		#[prost(message, tag = "3")]
		Rejected(super::ReadRejected),
		#[prost(message, tag = "4")]
		FileNotFound(super::ReadFileNotFound),
		#[prost(message, tag = "5")]
		PermissionDenied(super::ReadPermissionDenied),
		#[prost(message, tag = "6")]
		InvalidFile(super::ReadInvalidFile),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReadResult {
	#[prost(oneof = "read_result::Result", tags = "1, 2, 3, 4, 5, 6")]
	pub result: ::core::option::Option<read_result::Result>,
}

pub mod read_success {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Output {
		#[prost(string, tag = "2")]
		Content(::prost::alloc::string::String),
		#[prost(bytes = "vec", tag = "5")]
		Data(::prost::alloc::vec::Vec<u8>),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReadSuccess {
	#[prost(string, tag = "1")]
	pub path:           ::prost::alloc::string::String,
	#[prost(int32, tag = "3")]
	pub total_lines:    i32,
	#[prost(int64, tag = "4")]
	pub file_size:      i64,
	#[prost(bool, tag = "6")]
	pub truncated:      bool,
	#[prost(bytes = "vec", optional, tag = "7")]
	pub output_blob_id: ::core::option::Option<::prost::alloc::vec::Vec<u8>>,
	#[prost(oneof = "read_success::Output", tags = "2, 5")]
	pub output:         ::core::option::Option<read_success::Output>,
	#[prost(bool, tag = "8")]
	pub range_applied:  bool,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReadError {
	#[prost(string, tag = "1")]
	pub path:  ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub error: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReadRejected {
	#[prost(string, tag = "1")]
	pub path:   ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub reason: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReadFileNotFound {
	#[prost(string, tag = "1")]
	pub path: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReadPermissionDenied {
	#[prost(string, tag = "1")]
	pub path: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReadInvalidFile {
	#[prost(string, tag = "1")]
	pub path:   ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub reason: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReadToolCall {
	#[prost(message, optional, tag = "1")]
	pub args:   ::core::option::Option<ReadToolArgs>,
	#[prost(message, optional, tag = "2")]
	pub result: ::core::option::Option<ReadToolResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReadToolArgs {
	#[prost(string, tag = "1")]
	pub path:   ::prost::alloc::string::String,
	#[prost(int32, optional, tag = "2")]
	pub offset: ::core::option::Option<i32>,
	#[prost(int32, optional, tag = "3")]
	pub limit:  ::core::option::Option<i32>,
}

pub mod read_tool_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::ReadToolSuccess),
		#[prost(message, tag = "2")]
		Error(super::ReadToolError),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReadToolResult {
	#[prost(oneof = "read_tool_result::Result", tags = "1, 2")]
	pub result: ::core::option::Option<read_tool_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReadRange {
	#[prost(uint32, tag = "1")]
	pub start_line: u32,
	#[prost(uint32, tag = "2")]
	pub end_line:   u32,
}

pub mod read_tool_success {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Output {
		#[prost(string, tag = "1")]
		Content(::prost::alloc::string::String),
		#[prost(bytes = "vec", tag = "6")]
		Data(::prost::alloc::vec::Vec<u8>),
		#[prost(bytes = "vec", tag = "9")]
		DataBlobId(::prost::alloc::vec::Vec<u8>),
		#[prost(bytes = "vec", tag = "10")]
		ContentBlobId(::prost::alloc::vec::Vec<u8>),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReadToolSuccess {
	#[prost(bool, tag = "2")]
	pub is_empty:       bool,
	#[prost(bool, tag = "3")]
	pub exceeded_limit: bool,
	#[prost(uint32, tag = "4")]
	pub total_lines:    u32,
	#[prost(uint32, tag = "5")]
	pub file_size:      u32,
	#[prost(string, tag = "7")]
	pub path:           ::prost::alloc::string::String,
	#[prost(message, optional, tag = "8")]
	pub read_range:     ::core::option::Option<ReadRange>,
	#[prost(oneof = "read_tool_success::Output", tags = "1, 6, 9, 10")]
	pub output:         ::core::option::Option<read_tool_success::Output>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReadToolError {
	#[prost(string, tag = "1")]
	pub error_message: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RecordScreenArgs {
	#[prost(int32, tag = "1")]
	pub mode:             i32,
	#[prost(string, tag = "2")]
	pub tool_call_id:     ::prost::alloc::string::String,
	#[prost(string, optional, tag = "3")]
	pub save_as_filename: ::core::option::Option<::prost::alloc::string::String>,
}

pub mod record_screen_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		StartSuccess(super::RecordScreenStartSuccess),
		#[prost(message, tag = "2")]
		SaveSuccess(super::RecordScreenSaveSuccess),
		#[prost(message, tag = "3")]
		DiscardSuccess(super::RecordScreenDiscardSuccess),
		#[prost(message, tag = "4")]
		Failure(super::RecordScreenFailure),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RecordScreenResult {
	#[prost(oneof = "record_screen_result::Result", tags = "1, 2, 3, 4")]
	pub result: ::core::option::Option<record_screen_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RecordScreenStartSuccess {
	#[prost(bool, tag = "1")]
	pub was_prior_recording_cancelled: bool,
	#[prost(bool, tag = "2")]
	pub was_save_as_filename_ignored:  bool,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RecordScreenSaveSuccess {
	#[prost(string, tag = "1")]
	pub path: ::prost::alloc::string::String,
	#[prost(int64, tag = "2")]
	pub recording_duration_ms: i64,
	#[prost(int32, optional, tag = "3")]
	pub requested_file_path_rejected_reason: ::core::option::Option<i32>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RecordScreenDiscardSuccess {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RecordScreenFailure {
	#[prost(string, tag = "1")]
	pub error: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CursorPackagePrompt {
	#[prost(string, tag = "1")]
	pub name:      ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub file_path: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CursorPackage {
	#[prost(string, tag = "1")]
	pub name:             ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub description:      ::prost::alloc::string::String,
	#[prost(string, tag = "3")]
	pub folder_path:      ::prost::alloc::string::String,
	#[prost(bool, tag = "4")]
	pub enabled:          bool,
	#[prost(string, optional, tag = "5")]
	pub parse_error:      ::core::option::Option<::prost::alloc::string::String>,
	#[prost(message, repeated, tag = "6")]
	pub prompts:          ::prost::alloc::vec::Vec<CursorPackagePrompt>,
	#[prost(string, tag = "7")]
	pub readme_file_path: ::prost::alloc::string::String,
	#[prost(int32, tag = "8")]
	pub package_type:     i32,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RepositoryIndexingInfo {
	#[prost(string, tag = "1")]
	pub relative_workspace_path:   ::prost::alloc::string::String,
	#[prost(string, repeated, tag = "2")]
	pub remote_urls:               ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
	#[prost(string, repeated, tag = "3")]
	pub remote_names:              ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
	#[prost(string, tag = "4")]
	pub repo_name:                 ::prost::alloc::string::String,
	#[prost(string, tag = "5")]
	pub repo_owner:                ::prost::alloc::string::String,
	#[prost(bool, tag = "6")]
	pub is_tracked:                bool,
	#[prost(bool, tag = "7")]
	pub is_local:                  bool,
	#[prost(double, optional, tag = "8")]
	pub orthogonal_transform_seed: ::core::option::Option<f64>,
	#[prost(string, tag = "9")]
	pub workspace_uri:             ::prost::alloc::string::String,
	#[prost(string, tag = "10")]
	pub path_encryption_key:       ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RequestContextArgs {
	#[prost(string, optional, tag = "2")]
	pub notes_session_id:            ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "3")]
	pub workspace_id:                ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "4")]
	pub read_only_pinned_tree_sha:   ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "5")]
	pub read_only_plugin_cache_root: ::core::option::Option<::prost::alloc::string::String>,
	#[prost(bool, optional, tag = "7")]
	pub use_cached:                  ::core::option::Option<bool>,
}

pub mod request_context_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, boxed, tag = "1")]
		Success(::prost::alloc::boxed::Box<super::RequestContextSuccess>),
		#[prost(message, tag = "2")]
		Error(super::RequestContextError),
		#[prost(message, tag = "3")]
		Rejected(super::RequestContextRejected),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RequestContextResult {
	#[prost(oneof = "request_context_result::Result", tags = "1, 2, 3")]
	pub result: ::core::option::Option<request_context_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RequestContextSuccess {
	#[prost(message, optional, tag = "1")]
	pub request_context:        ::core::option::Option<RequestContext>,
	#[prost(bool, optional, tag = "2")]
	pub served_from_disk_cache: ::core::option::Option<bool>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RequestContextError {
	#[prost(string, tag = "1")]
	pub error: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RequestContextRejected {
	#[prost(string, tag = "1")]
	pub reason: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ImageProto {
	#[prost(bytes = "vec", tag = "1")]
	pub data: ::prost::alloc::vec::Vec<u8>,
	#[prost(string, tag = "2")]
	pub uuid: ::prost::alloc::string::String,
	#[prost(string, tag = "3")]
	pub path: ::prost::alloc::string::String,
	#[prost(message, optional, tag = "4")]
	pub dimension: ::core::option::Option<ImageProto_Dimension>,
	#[prost(string, optional, tag = "6")]
	pub task_specific_description: ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, tag = "7")]
	pub mime_type: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ImageProto_Dimension {
	#[prost(int32, tag = "1")]
	pub width:  i32,
	#[prost(int32, tag = "2")]
	pub height: i32,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GitRepoInfo {
	#[prost(string, tag = "1")]
	pub path:        ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub status:      ::prost::alloc::string::String,
	#[prost(string, tag = "3")]
	pub branch_name: ::prost::alloc::string::String,
	#[prost(string, optional, tag = "4")]
	pub remote_url:  ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RequestContextEnv {
	#[prost(string, tag = "1")]
	pub os_version: ::prost::alloc::string::String,
	#[prost(string, repeated, tag = "2")]
	pub workspace_paths: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
	#[prost(string, tag = "3")]
	pub shell: ::prost::alloc::string::String,
	#[prost(bool, tag = "5")]
	pub sandbox_enabled: bool,
	#[prost(string, tag = "7")]
	pub terminals_folder: ::prost::alloc::string::String,
	#[prost(string, tag = "8")]
	pub agent_shared_notes_folder: ::prost::alloc::string::String,
	#[prost(string, tag = "9")]
	pub agent_conversation_notes_folder: ::prost::alloc::string::String,
	#[prost(string, tag = "10")]
	pub time_zone: ::prost::alloc::string::String,
	#[prost(string, tag = "11")]
	pub project_folder: ::prost::alloc::string::String,
	#[prost(string, tag = "12")]
	pub agent_transcripts_folder: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DebugModeConfig {
	#[prost(string, tag = "1")]
	pub log_path:        ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub server_endpoint: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SkillDescriptor {
	#[prost(string, tag = "1")]
	pub name:             ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub description:      ::prost::alloc::string::String,
	#[prost(string, tag = "3")]
	pub folder_path:      ::prost::alloc::string::String,
	#[prost(bool, tag = "4")]
	pub enabled:          bool,
	#[prost(string, optional, tag = "5")]
	pub parse_error:      ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, tag = "6")]
	pub readme_file_path: ::prost::alloc::string::String,
	#[prost(int32, tag = "7")]
	pub package_type:     i32,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SkillOptions {
	#[prost(message, repeated, tag = "1")]
	pub skill_descriptors: ::prost::alloc::vec::Vec<SkillDescriptor>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RequestContext {
	#[prost(message, repeated, tag = "2")]
	pub rules: ::prost::alloc::vec::Vec<CursorRule>,
	#[prost(message, optional, tag = "4")]
	pub env: ::core::option::Option<RequestContextEnv>,
	#[prost(message, repeated, tag = "6")]
	pub repository_info: ::prost::alloc::vec::Vec<RepositoryIndexingInfo>,
	#[prost(message, repeated, tag = "7")]
	pub tools: ::prost::alloc::vec::Vec<McpToolDefinition>,
	#[prost(string, optional, tag = "8")]
	pub conversation_notes_listing: ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "9")]
	pub shared_notes_listing: ::core::option::Option<::prost::alloc::string::String>,
	#[prost(message, repeated, tag = "11")]
	pub git_repos: ::prost::alloc::vec::Vec<GitRepoInfo>,
	#[prost(message, repeated, tag = "13")]
	pub project_layouts: ::prost::alloc::vec::Vec<LsDirectoryTreeNode>,
	#[prost(message, repeated, tag = "14")]
	pub mcp_instructions: ::prost::alloc::vec::Vec<McpInstructions>,
	#[prost(message, optional, tag = "15")]
	pub debug_mode_config: ::core::option::Option<DebugModeConfig>,
	#[prost(string, optional, tag = "16")]
	pub cloud_rule: ::core::option::Option<::prost::alloc::string::String>,
	#[prost(bool, optional, tag = "17")]
	pub web_search_enabled: ::core::option::Option<bool>,
	#[prost(message, optional, tag = "18")]
	pub skill_options: ::core::option::Option<SkillOptions>,
	#[prost(bool, optional, tag = "19")]
	pub repository_info_should_query_prod: ::core::option::Option<bool>,
	#[prost(map = "string, string", tag = "20")]
	pub file_contents:
		::std::collections::HashMap<::prost::alloc::string::String, ::prost::alloc::string::String>,
	#[prost(string, optional, tag = "21")]
	pub user_intent_summary: ::core::option::Option<::prost::alloc::string::String>,
	#[prost(message, repeated, tag = "22")]
	pub custom_subagents: ::prost::alloc::vec::Vec<CustomSubagent>,
	#[prost(message, optional, tag = "23")]
	pub mcp_file_system_options: ::core::option::Option<McpFileSystemOptions>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SandboxPolicy {
	#[prost(int32, tag = "1")]
	pub r#type:                     i32,
	#[prost(bool, optional, tag = "2")]
	pub network_access:             ::core::option::Option<bool>,
	#[prost(string, repeated, tag = "3")]
	pub additional_readwrite_paths: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
	#[prost(string, repeated, tag = "4")]
	pub additional_readonly_paths:  ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "5")]
	pub debug_output_dir:           ::core::option::Option<::prost::alloc::string::String>,
	#[prost(bool, optional, tag = "6")]
	pub block_git_writes:           ::core::option::Option<bool>,
	#[prost(bool, optional, tag = "7")]
	pub disable_tmp_write:          ::core::option::Option<bool>,
}

pub mod selected_image {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum DataOrBlobId {
		#[prost(bytes = "vec", tag = "1")]
		BlobId(::prost::alloc::vec::Vec<u8>),
		#[prost(bytes = "vec", tag = "8")]
		Data(::prost::alloc::vec::Vec<u8>),
		#[prost(message, tag = "9")]
		BlobIdWithData(super::SelectedImage_BlobIdWithData),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SelectedImage {
	#[prost(string, tag = "2")]
	pub uuid:            ::prost::alloc::string::String,
	#[prost(string, tag = "3")]
	pub path:            ::prost::alloc::string::String,
	#[prost(message, optional, tag = "4")]
	pub dimension:       ::core::option::Option<SelectedImage_Dimension>,
	#[prost(string, tag = "7")]
	pub mime_type:       ::prost::alloc::string::String,
	#[prost(oneof = "selected_image::DataOrBlobId", tags = "1, 8, 9")]
	pub data_or_blob_id: ::core::option::Option<selected_image::DataOrBlobId>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SelectedImage_BlobIdWithData {
	#[prost(bytes = "vec", tag = "1")]
	pub blob_id: ::prost::alloc::vec::Vec<u8>,
	#[prost(bytes = "vec", tag = "2")]
	pub data:    ::prost::alloc::vec::Vec<u8>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SelectedImage_Dimension {
	#[prost(int32, tag = "1")]
	pub width:  i32,
	#[prost(int32, tag = "2")]
	pub height: i32,
}

pub mod extra_context_entry {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum DataOrBlobId {
		#[prost(string, tag = "1")]
		Data(::prost::alloc::string::String),
		#[prost(bytes = "vec", tag = "2")]
		BlobId(::prost::alloc::vec::Vec<u8>),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExtraContextEntry {
	#[prost(oneof = "extra_context_entry::DataOrBlobId", tags = "1, 2")]
	pub data_or_blob_id: ::core::option::Option<extra_context_entry::DataOrBlobId>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SelectedFile {
	#[prost(string, tag = "1")]
	pub content:       ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub path:          ::prost::alloc::string::String,
	#[prost(string, optional, tag = "3")]
	pub relative_path: ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SelectedCodeSelection {
	#[prost(string, tag = "1")]
	pub content:       ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub path:          ::prost::alloc::string::String,
	#[prost(string, optional, tag = "3")]
	pub relative_path: ::core::option::Option<::prost::alloc::string::String>,
	#[prost(message, optional, tag = "4")]
	pub range:         ::core::option::Option<Range>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SelectedTerminal {
	#[prost(string, tag = "1")]
	pub content: ::prost::alloc::string::String,
	#[prost(string, optional, tag = "2")]
	pub title:   ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "3")]
	pub path:    ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SelectedTerminalSelection {
	#[prost(string, tag = "1")]
	pub content: ::prost::alloc::string::String,
	#[prost(string, optional, tag = "2")]
	pub title:   ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "3")]
	pub path:    ::core::option::Option<::prost::alloc::string::String>,
	#[prost(message, optional, tag = "4")]
	pub range:   ::core::option::Option<Range>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SelectedFolder {
	#[prost(string, tag = "1")]
	pub path:           ::prost::alloc::string::String,
	#[prost(string, optional, tag = "2")]
	pub relative_path:  ::core::option::Option<::prost::alloc::string::String>,
	#[prost(message, optional, tag = "3")]
	pub directory_tree: ::core::option::Option<LsDirectoryTreeNode>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SelectedExternalLink {
	#[prost(string, tag = "1")]
	pub url:         ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub uuid:        ::prost::alloc::string::String,
	#[prost(string, optional, tag = "3")]
	pub pdf_content: ::core::option::Option<::prost::alloc::string::String>,
	#[prost(bool, optional, tag = "4")]
	pub is_pdf:      ::core::option::Option<bool>,
	#[prost(string, optional, tag = "5")]
	pub filename:    ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SelectedCursorRule {
	#[prost(message, optional, tag = "1")]
	pub rule: ::core::option::Option<CursorRule>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SelectedGitDiff {
	#[prost(string, tag = "1")]
	pub content: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SelectedGitDiffFromBranchToMain {
	#[prost(string, tag = "1")]
	pub content: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SelectedGitCommit {
	#[prost(string, tag = "1")]
	pub sha:         ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub message:     ::prost::alloc::string::String,
	#[prost(string, optional, tag = "3")]
	pub description: ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, tag = "4")]
	pub diff:        ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SelectedPullRequest {
	#[prost(int32, tag = "1")]
	pub number:       i32,
	#[prost(string, tag = "2")]
	pub url:          ::prost::alloc::string::String,
	#[prost(string, optional, tag = "3")]
	pub title:        ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, tag = "4")]
	pub folder_path:  ::prost::alloc::string::String,
	#[prost(string, optional, tag = "5")]
	pub summary_json: ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "6")]
	pub description:  ::core::option::Option<::prost::alloc::string::String>,
	#[prost(bytes = "vec", optional, tag = "7")]
	pub blob_id:      ::core::option::Option<::prost::alloc::vec::Vec<u8>>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SelectedGitPRDiffSelection {
	#[prost(string, tag = "1")]
	pub pr_url:       ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub file_path:    ::prost::alloc::string::String,
	#[prost(int32, tag = "3")]
	pub start_line:   i32,
	#[prost(int32, tag = "4")]
	pub end_line:     i32,
	#[prost(string, optional, tag = "5")]
	pub diff_content: ::core::option::Option<::prost::alloc::string::String>,
	#[prost(bytes = "vec", optional, tag = "6")]
	pub blob_id:      ::core::option::Option<::prost::alloc::vec::Vec<u8>>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SelectedCursorCommand {
	#[prost(string, tag = "1")]
	pub name:    ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub content: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SelectedDocumentation {
	#[prost(string, tag = "1")]
	pub doc_id: ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub name:   ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SelectedPastChat {
	#[prost(string, tag = "1")]
	pub agent_id: ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub name:     ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CallFrame {
	#[prost(string, optional, tag = "1")]
	pub function_name: ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "2")]
	pub url:           ::core::option::Option<::prost::alloc::string::String>,
	#[prost(int32, optional, tag = "3")]
	pub line_number:   ::core::option::Option<i32>,
	#[prost(int32, optional, tag = "4")]
	pub column_number: ::core::option::Option<i32>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StackTrace {
	#[prost(message, repeated, tag = "1")]
	pub call_frames:     ::prost::alloc::vec::Vec<CallFrame>,
	#[prost(string, optional, tag = "2")]
	pub raw_stack_trace: ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SelectedConsoleLog {
	#[prost(string, tag = "1")]
	pub message:          ::prost::alloc::string::String,
	#[prost(double, tag = "2")]
	pub timestamp:        f64,
	#[prost(string, tag = "3")]
	pub level:            ::prost::alloc::string::String,
	#[prost(string, tag = "4")]
	pub client_name:      ::prost::alloc::string::String,
	#[prost(string, tag = "5")]
	pub session_id:       ::prost::alloc::string::String,
	#[prost(message, optional, tag = "6")]
	pub stack_trace:      ::core::option::Option<StackTrace>,
	#[prost(string, optional, tag = "7")]
	pub object_data_json: ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SelectedUIElement {
	#[prost(string, tag = "1")]
	pub element:              ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub xpath:                ::prost::alloc::string::String,
	#[prost(string, tag = "3")]
	pub text_content:         ::prost::alloc::string::String,
	#[prost(string, tag = "4")]
	pub extra:                ::prost::alloc::string::String,
	#[prost(string, optional, tag = "5")]
	pub component:            ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "6")]
	pub component_props_json: ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SelectedSubagent {
	#[prost(string, tag = "1")]
	pub name: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SelectedContext {
	#[prost(message, repeated, tag = "1")]
	pub selected_images: ::prost::alloc::vec::Vec<SelectedImage>,
	#[prost(message, optional, tag = "2")]
	pub invocation_context: ::core::option::Option<InvocationContext>,
	#[prost(string, repeated, tag = "3")]
	pub extra_context: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
	#[prost(message, repeated, tag = "16")]
	pub extra_context_entries: ::prost::alloc::vec::Vec<ExtraContextEntry>,
	#[prost(message, repeated, tag = "4")]
	pub files: ::prost::alloc::vec::Vec<SelectedFile>,
	#[prost(message, repeated, tag = "5")]
	pub code_selections: ::prost::alloc::vec::Vec<SelectedCodeSelection>,
	#[prost(message, repeated, tag = "6")]
	pub terminals: ::prost::alloc::vec::Vec<SelectedTerminal>,
	#[prost(message, repeated, tag = "7")]
	pub terminal_selections: ::prost::alloc::vec::Vec<SelectedTerminalSelection>,
	#[prost(message, repeated, tag = "8")]
	pub folders: ::prost::alloc::vec::Vec<SelectedFolder>,
	#[prost(message, repeated, tag = "9")]
	pub external_links: ::prost::alloc::vec::Vec<SelectedExternalLink>,
	#[prost(message, repeated, tag = "10")]
	pub cursor_rules: ::prost::alloc::vec::Vec<SelectedCursorRule>,
	#[prost(message, optional, tag = "18")]
	pub git_diff: ::core::option::Option<SelectedGitDiff>,
	#[prost(message, optional, tag = "11")]
	pub git_diff_from_branch_to_main: ::core::option::Option<SelectedGitDiffFromBranchToMain>,
	#[prost(message, repeated, tag = "12")]
	pub cursor_commands: ::prost::alloc::vec::Vec<SelectedCursorCommand>,
	#[prost(message, repeated, tag = "13")]
	pub documentations: ::prost::alloc::vec::Vec<SelectedDocumentation>,
	#[prost(message, repeated, tag = "14")]
	pub ui_elements: ::prost::alloc::vec::Vec<SelectedUIElement>,
	#[prost(message, repeated, tag = "15")]
	pub console_logs: ::prost::alloc::vec::Vec<SelectedConsoleLog>,
	#[prost(message, repeated, tag = "17")]
	pub git_commits: ::prost::alloc::vec::Vec<SelectedGitCommit>,
	#[prost(message, repeated, tag = "19")]
	pub past_chats: ::prost::alloc::vec::Vec<SelectedPastChat>,
	#[prost(message, repeated, tag = "20")]
	pub git_pr_diff_selections: ::prost::alloc::vec::Vec<SelectedGitPRDiffSelection>,
	#[prost(message, repeated, tag = "21")]
	pub selected_pull_requests: ::prost::alloc::vec::Vec<SelectedPullRequest>,
	#[prost(message, repeated, tag = "22")]
	pub selected_subagents: ::prost::alloc::vec::Vec<SelectedSubagent>,
}

pub mod invocation_context {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Data {
		#[prost(message, tag = "1")]
		SlackThread(super::InvocationContext_SlackThread),
		#[prost(message, tag = "2")]
		GithubPr(super::InvocationContext_GithubPR),
		#[prost(message, tag = "3")]
		IdeState(super::InvocationContext_IdeState),
		#[prost(bytes = "vec", tag = "10")]
		BlobId(::prost::alloc::vec::Vec<u8>),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct InvocationContext {
	#[prost(oneof = "invocation_context::Data", tags = "1, 2, 3, 10")]
	pub data: ::core::option::Option<invocation_context::Data>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct InvocationContext_SlackThread {
	#[prost(string, tag = "1")]
	pub thread:          ::prost::alloc::string::String,
	#[prost(string, optional, tag = "2")]
	pub channel_name:    ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "3")]
	pub channel_purpose: ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "4")]
	pub channel_topic:   ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct InvocationContext_GithubPR {
	#[prost(string, tag = "1")]
	pub title:       ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub description: ::prost::alloc::string::String,
	#[prost(string, tag = "3")]
	pub comments:    ::prost::alloc::string::String,
	#[prost(string, optional, tag = "4")]
	pub ci_failures: ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct InvocationContext_IdeState {
	#[prost(message, repeated, tag = "1")]
	pub visible_files:         ::prost::alloc::vec::Vec<InvocationContext_IdeState_File>,
	#[prost(message, repeated, tag = "2")]
	pub recently_viewed_files: ::prost::alloc::vec::Vec<InvocationContext_IdeState_File>,
	#[prost(message, repeated, tag = "3")]
	pub currently_viewed_prs: ::prost::alloc::vec::Vec<InvocationContext_IdeState_ViewedPullRequest>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct InvocationContext_IdeState_File {
	#[prost(string, tag = "1")]
	pub path:            ::prost::alloc::string::String,
	#[prost(string, optional, tag = "2")]
	pub relative_path:   ::core::option::Option<::prost::alloc::string::String>,
	#[prost(message, optional, tag = "3")]
	pub cursor_position: ::core::option::Option<InvocationContext_IdeState_File_CursorPosition>,
	#[prost(int32, tag = "4")]
	pub total_lines:     i32,
	#[prost(string, optional, tag = "5")]
	pub active_command:  ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct InvocationContext_IdeState_File_CursorPosition {
	#[prost(int32, tag = "1")]
	pub line: i32,
	#[prost(string, tag = "2")]
	pub text: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct InvocationContext_IdeState_ViewedPullRequest {
	#[prost(int32, tag = "1")]
	pub number:       i32,
	#[prost(string, tag = "2")]
	pub url:          ::prost::alloc::string::String,
	#[prost(string, optional, tag = "3")]
	pub title:        ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "4")]
	pub folder_path:  ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "5")]
	pub summary_json: ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "6")]
	pub description:  ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SetupVmEnvironmentArgs {
	#[prost(string, tag = "2")]
	pub install_command: ::prost::alloc::string::String,
	#[prost(string, tag = "3")]
	pub start_command:   ::prost::alloc::string::String,
}

pub mod setup_vm_environment_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::SetupVmEnvironmentSuccess),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SetupVmEnvironmentResult {
	#[prost(oneof = "setup_vm_environment_result::Result", tags = "1")]
	pub result: ::core::option::Option<setup_vm_environment_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SetupVmEnvironmentSuccess {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SetupVmEnvironmentToolCall {
	#[prost(message, optional, tag = "1")]
	pub args:   ::core::option::Option<SetupVmEnvironmentArgs>,
	#[prost(message, optional, tag = "2")]
	pub result: ::core::option::Option<SetupVmEnvironmentResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ShellCommandParsingResult {
	#[prost(bool, tag = "1")]
	pub parsing_failed:             bool,
	#[prost(message, repeated, tag = "2")]
	pub executable_commands: ::prost::alloc::vec::Vec<ShellCommandParsingResult_ExecutableCommand>,
	#[prost(bool, tag = "3")]
	pub has_redirects:              bool,
	#[prost(bool, tag = "4")]
	pub has_command_substitution:   bool,
	#[prost(bool, optional, tag = "5")]
	pub all_redirects_are_dev_null: ::core::option::Option<bool>,
	#[prost(message, repeated, tag = "6")]
	pub redirects:                  ::prost::alloc::vec::Vec<ShellCommandParsingResult_Redirect>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ShellCommandParsingResult_ExecutableCommandArg {
	#[prost(string, tag = "1")]
	pub r#type: ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub value:  ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ShellCommandParsingResult_ExecutableCommand {
	#[prost(string, tag = "1")]
	pub name:      ::prost::alloc::string::String,
	#[prost(message, repeated, tag = "2")]
	pub args:      ::prost::alloc::vec::Vec<ShellCommandParsingResult_ExecutableCommandArg>,
	#[prost(string, tag = "3")]
	pub full_text: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ShellArgs {
	#[prost(string, tag = "1")]
	pub command:                     ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub working_directory:           ::prost::alloc::string::String,
	#[prost(int32, tag = "3")]
	pub timeout:                     i32,
	#[prost(string, tag = "4")]
	pub tool_call_id:                ::prost::alloc::string::String,
	#[prost(string, repeated, tag = "5")]
	pub simple_commands:             ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
	#[prost(bool, tag = "6")]
	pub has_input_redirect:          bool,
	#[prost(bool, tag = "7")]
	pub has_output_redirect:         bool,
	#[prost(message, optional, tag = "8")]
	pub parsing_result:              ::core::option::Option<ShellCommandParsingResult>,
	#[prost(message, optional, tag = "9")]
	pub requested_sandbox_policy:    ::core::option::Option<SandboxPolicy>,
	#[prost(uint64, optional, tag = "10")]
	pub file_output_threshold_bytes: ::core::option::Option<u64>,
	#[prost(bool, tag = "11")]
	pub is_background:               bool,
	#[prost(bool, tag = "12")]
	pub skip_approval:               bool,
	#[prost(int32, tag = "13")]
	pub timeout_behavior:            i32,
	#[prost(int32, optional, tag = "14")]
	pub hard_timeout:                ::core::option::Option<i32>,
	#[prost(string, optional, tag = "15")]
	pub description:                 ::core::option::Option<::prost::alloc::string::String>,
	#[prost(message, optional, tag = "16")]
	pub classifier_result:           ::core::option::Option<CommandClassifierResult>,
	#[prost(bool, tag = "17")]
	pub close_stdin:                 bool,
	#[prost(message, optional, tag = "18")]
	pub output_notification:         ::core::option::Option<ShellOutputNotificationConfig>,
	#[prost(message, optional, tag = "19")]
	pub smart_mode_approval:         ::core::option::Option<SmartModeApproval>,
	#[prost(message, optional, tag = "20")]
	pub hook_approval_requirement:   ::core::option::Option<ShellHookApprovalRequirement>,
	#[prost(string, optional, tag = "21")]
	pub conversation_id:             ::core::option::Option<::prost::alloc::string::String>,
}

pub mod shell_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::ShellSuccess),
		#[prost(message, tag = "2")]
		Failure(super::ShellFailure),
		#[prost(message, tag = "3")]
		Timeout(super::ShellTimeout),
		#[prost(message, tag = "4")]
		Rejected(super::ShellRejected),
		#[prost(message, tag = "5")]
		SpawnError(super::ShellSpawnError),
		#[prost(message, tag = "7")]
		PermissionDenied(super::ShellPermissionDenied),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ShellResult {
	#[prost(message, optional, tag = "101")]
	pub sandbox_policy:   ::core::option::Option<SandboxPolicy>,
	#[prost(bool, optional, tag = "102")]
	pub is_background:    ::core::option::Option<bool>,
	#[prost(string, optional, tag = "103")]
	pub terminals_folder: ::core::option::Option<::prost::alloc::string::String>,
	#[prost(uint32, optional, tag = "104")]
	pub pid:              ::core::option::Option<u32>,
	#[prost(oneof = "shell_result::Result", tags = "1, 2, 3, 4, 5, 7")]
	pub result:           ::core::option::Option<shell_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ShellStreamStdout {
	#[prost(string, tag = "1")]
	pub data: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ShellStreamStderr {
	#[prost(string, tag = "1")]
	pub data: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ShellStreamExit {
	#[prost(uint32, tag = "1")]
	pub code:                    u32,
	#[prost(string, tag = "2")]
	pub cwd:                     ::prost::alloc::string::String,
	#[prost(message, optional, tag = "3")]
	pub output_location:         ::core::option::Option<OutputLocation>,
	#[prost(bool, tag = "4")]
	pub aborted:                 bool,
	#[prost(int32, optional, tag = "5")]
	pub abort_reason:            ::core::option::Option<i32>,
	#[prost(int32, optional, tag = "6")]
	pub local_execution_time_ms: ::core::option::Option<i32>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ShellStreamStart {
	#[prost(message, optional, tag = "1")]
	pub sandbox_policy: ::core::option::Option<SandboxPolicy>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ShellStreamBackgrounded {
	#[prost(uint32, tag = "1")]
	pub shell_id:          u32,
	#[prost(string, tag = "2")]
	pub command:           ::prost::alloc::string::String,
	#[prost(string, tag = "3")]
	pub working_directory: ::prost::alloc::string::String,
	#[prost(uint32, optional, tag = "4")]
	pub pid:               ::core::option::Option<u32>,
	#[prost(int32, optional, tag = "5")]
	pub ms_to_wait:        ::core::option::Option<i32>,
	#[prost(enumeration = "ShellBackgroundReason", optional, tag = "6")]
	pub reason:            ::core::option::Option<i32>,
}

pub mod shell_stream {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Event {
		#[prost(message, tag = "1")]
		Stdout(super::ShellStreamStdout),
		#[prost(message, tag = "2")]
		Stderr(super::ShellStreamStderr),
		#[prost(message, tag = "3")]
		Exit(super::ShellStreamExit),
		#[prost(message, tag = "4")]
		Start(super::ShellStreamStart),
		#[prost(message, tag = "5")]
		Rejected(super::ShellRejected),
		#[prost(message, tag = "6")]
		PermissionDenied(super::ShellPermissionDenied),
		#[prost(message, tag = "7")]
		Backgrounded(super::ShellStreamBackgrounded),
		#[prost(message, tag = "8")]
		HookContext(super::ShellStreamHookContext),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ShellStream {
	#[prost(oneof = "shell_stream::Event", tags = "1, 2, 3, 4, 5, 6, 7, 8")]
	pub event: ::core::option::Option<shell_stream::Event>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct OutputLocation {
	#[prost(string, tag = "1")]
	pub file_path:  ::prost::alloc::string::String,
	#[prost(int64, tag = "2")]
	pub size_bytes: i64,
	#[prost(int64, tag = "3")]
	pub line_count: i64,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ShellSuccess {
	#[prost(string, tag = "1")]
	pub command:                 ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub working_directory:       ::prost::alloc::string::String,
	#[prost(int32, tag = "3")]
	pub exit_code:               i32,
	#[prost(string, tag = "4")]
	pub signal:                  ::prost::alloc::string::String,
	#[prost(string, tag = "5")]
	pub stdout:                  ::prost::alloc::string::String,
	#[prost(string, tag = "6")]
	pub stderr:                  ::prost::alloc::string::String,
	#[prost(int32, tag = "7")]
	pub execution_time:          i32,
	#[prost(message, optional, tag = "8")]
	pub output_location:         ::core::option::Option<OutputLocation>,
	#[prost(uint32, optional, tag = "9")]
	pub shell_id:                ::core::option::Option<u32>,
	#[prost(string, optional, tag = "10")]
	pub interleaved_output:      ::core::option::Option<::prost::alloc::string::String>,
	#[prost(uint32, optional, tag = "11")]
	pub pid:                     ::core::option::Option<u32>,
	#[prost(int32, optional, tag = "12")]
	pub ms_to_wait:              ::core::option::Option<i32>,
	#[prost(int32, optional, tag = "13")]
	pub local_execution_time_ms: ::core::option::Option<i32>,
	#[prost(enumeration = "ShellBackgroundReason", optional, tag = "14")]
	pub background_reason:       ::core::option::Option<i32>,
	#[prost(string, optional, tag = "15")]
	pub output_head:             ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "16")]
	pub output_tail:             ::core::option::Option<::prost::alloc::string::String>,
	#[prost(uint32, optional, tag = "17")]
	pub elided_chars:            ::core::option::Option<u32>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ShellFailure {
	#[prost(string, tag = "1")]
	pub command:                 ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub working_directory:       ::prost::alloc::string::String,
	#[prost(int32, tag = "3")]
	pub exit_code:               i32,
	#[prost(string, tag = "4")]
	pub signal:                  ::prost::alloc::string::String,
	#[prost(string, tag = "5")]
	pub stdout:                  ::prost::alloc::string::String,
	#[prost(string, tag = "6")]
	pub stderr:                  ::prost::alloc::string::String,
	#[prost(int32, tag = "7")]
	pub execution_time:          i32,
	#[prost(message, optional, tag = "8")]
	pub output_location:         ::core::option::Option<OutputLocation>,
	#[prost(string, optional, tag = "9")]
	pub interleaved_output:      ::core::option::Option<::prost::alloc::string::String>,
	#[prost(int32, optional, tag = "10")]
	pub abort_reason:            ::core::option::Option<i32>,
	#[prost(bool, tag = "11")]
	pub aborted:                 bool,
	#[prost(int32, optional, tag = "12")]
	pub local_execution_time_ms: ::core::option::Option<i32>,
	#[prost(string, optional, tag = "13")]
	pub output_head:             ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "14")]
	pub output_tail:             ::core::option::Option<::prost::alloc::string::String>,
	#[prost(uint32, optional, tag = "15")]
	pub elided_chars:            ::core::option::Option<u32>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ShellTimeout {
	#[prost(string, tag = "1")]
	pub command:           ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub working_directory: ::prost::alloc::string::String,
	#[prost(int32, tag = "3")]
	pub timeout_ms:        i32,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ShellRejected {
	#[prost(string, tag = "1")]
	pub command:           ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub working_directory: ::prost::alloc::string::String,
	#[prost(string, tag = "3")]
	pub reason:            ::prost::alloc::string::String,
	#[prost(bool, tag = "4")]
	pub is_readonly:       bool,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ShellPermissionDenied {
	#[prost(string, tag = "1")]
	pub command:           ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub working_directory: ::prost::alloc::string::String,
	#[prost(string, tag = "3")]
	pub error:             ::prost::alloc::string::String,
	#[prost(bool, tag = "4")]
	pub is_readonly:       bool,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ShellSpawnError {
	#[prost(string, tag = "1")]
	pub command:           ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub working_directory: ::prost::alloc::string::String,
	#[prost(string, tag = "3")]
	pub error:             ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ShellPartialResult {
	#[prost(string, tag = "1")]
	pub stdout_delta: ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub stderr_delta: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ShellToolCall {
	#[prost(message, optional, tag = "1")]
	pub args:   ::core::option::Option<ShellArgs>,
	#[prost(message, optional, tag = "2")]
	pub result: ::core::option::Option<ShellResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ShellToolCallStdoutDelta {
	#[prost(string, tag = "1")]
	pub content: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ShellToolCallStderrDelta {
	#[prost(string, tag = "1")]
	pub content: ::prost::alloc::string::String,
}

pub mod shell_tool_call_delta {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Delta {
		#[prost(message, tag = "1")]
		Stdout(super::ShellToolCallStdoutDelta),
		#[prost(message, tag = "2")]
		Stderr(super::ShellToolCallStderrDelta),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ShellToolCallDelta {
	#[prost(oneof = "shell_tool_call_delta::Delta", tags = "1, 2")]
	pub delta: ::core::option::Option<shell_tool_call_delta::Delta>,
}

pub mod subagent_type {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Type {
		#[prost(message, tag = "1")]
		Unspecified(super::SubagentTypeUnspecified),
		#[prost(message, tag = "2")]
		ComputerUse(super::SubagentTypeComputerUse),
		#[prost(message, tag = "3")]
		Custom(super::SubagentTypeCustom),
		#[prost(message, tag = "4")]
		Explore(super::SubagentTypeExplore),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SubagentType {
	#[prost(oneof = "subagent_type::Type", tags = "1, 2, 3, 4")]
	pub r#type: ::core::option::Option<subagent_type::Type>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SubagentTypeUnspecified {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SubagentTypeComputerUse {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SubagentTypeExplore {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SubagentTypeCustom {
	#[prost(string, tag = "1")]
	pub name: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CustomSubagent {
	#[prost(string, tag = "1")]
	pub full_path:       ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub name:            ::prost::alloc::string::String,
	#[prost(string, tag = "3")]
	pub description:     ::prost::alloc::string::String,
	#[prost(string, repeated, tag = "4")]
	pub tools:           ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
	#[prost(string, tag = "5")]
	pub model:           ::prost::alloc::string::String,
	#[prost(string, tag = "6")]
	pub prompt:          ::prost::alloc::string::String,
	#[prost(int32, tag = "7")]
	pub permission_mode: i32,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SwitchModeArgs {
	#[prost(string, tag = "1")]
	pub target_mode_id: ::prost::alloc::string::String,
	#[prost(string, optional, tag = "2")]
	pub explanation:    ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, tag = "3")]
	pub tool_call_id:   ::prost::alloc::string::String,
}

pub mod switch_mode_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::SwitchModeSuccess),
		#[prost(message, tag = "2")]
		Error(super::SwitchModeError),
		#[prost(message, tag = "3")]
		Rejected(super::SwitchModeRejected),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SwitchModeResult {
	#[prost(oneof = "switch_mode_result::Result", tags = "1, 2, 3")]
	pub result: ::core::option::Option<switch_mode_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SwitchModeSuccess {
	#[prost(string, tag = "1")]
	pub from_mode_id: ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub to_mode_id:   ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SwitchModeError {
	#[prost(string, tag = "1")]
	pub error: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SwitchModeRejected {
	#[prost(string, tag = "1")]
	pub reason: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SwitchModeToolCall {
	#[prost(message, optional, tag = "1")]
	pub args:   ::core::option::Option<SwitchModeArgs>,
	#[prost(message, optional, tag = "2")]
	pub result: ::core::option::Option<SwitchModeResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SwitchModeRequestQuery {
	#[prost(message, optional, tag = "1")]
	pub args: ::core::option::Option<SwitchModeArgs>,
}

pub mod switch_mode_request_response {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Approved(super::SwitchModeRequestResponse_Approved),
		#[prost(message, tag = "2")]
		Rejected(super::SwitchModeRequestResponse_Rejected),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SwitchModeRequestResponse {
	#[prost(oneof = "switch_mode_request_response::Result", tags = "1, 2")]
	pub result: ::core::option::Option<switch_mode_request_response::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SwitchModeRequestResponse_Approved {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SwitchModeRequestResponse_Rejected {
	#[prost(string, tag = "1")]
	pub reason: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TodoItem {
	#[prost(string, tag = "1")]
	pub id:           ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub content:      ::prost::alloc::string::String,
	#[prost(int32, tag = "3")]
	pub status:       i32,
	#[prost(int64, tag = "4")]
	pub created_at:   i64,
	#[prost(int64, tag = "5")]
	pub updated_at:   i64,
	#[prost(string, repeated, tag = "6")]
	pub dependencies: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct UpdateTodosToolCall {
	#[prost(message, optional, tag = "1")]
	pub args:   ::core::option::Option<UpdateTodosArgs>,
	#[prost(message, optional, tag = "2")]
	pub result: ::core::option::Option<UpdateTodosResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct UpdateTodosArgs {
	#[prost(message, repeated, tag = "1")]
	pub todos: ::prost::alloc::vec::Vec<TodoItem>,
	#[prost(bool, tag = "2")]
	pub merge: bool,
}

pub mod update_todos_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::UpdateTodosSuccess),
		#[prost(message, tag = "2")]
		Error(super::UpdateTodosError),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct UpdateTodosResult {
	#[prost(oneof = "update_todos_result::Result", tags = "1, 2")]
	pub result: ::core::option::Option<update_todos_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct UpdateTodosSuccess {
	#[prost(message, repeated, tag = "1")]
	pub todos:       ::prost::alloc::vec::Vec<TodoItem>,
	#[prost(int32, tag = "2")]
	pub total_count: i32,
	#[prost(bool, tag = "3")]
	pub was_merge:   bool,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct UpdateTodosError {
	#[prost(string, tag = "1")]
	pub error: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReadTodosToolCall {
	#[prost(message, optional, tag = "1")]
	pub args:   ::core::option::Option<ReadTodosArgs>,
	#[prost(message, optional, tag = "2")]
	pub result: ::core::option::Option<ReadTodosResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReadTodosArgs {
	#[prost(int32, repeated, tag = "1")]
	pub status_filter: ::prost::alloc::vec::Vec<i32>,
	#[prost(string, repeated, tag = "2")]
	pub id_filter:     ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
}

pub mod read_todos_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::ReadTodosSuccess),
		#[prost(message, tag = "2")]
		Error(super::ReadTodosError),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReadTodosResult {
	#[prost(oneof = "read_todos_result::Result", tags = "1, 2")]
	pub result: ::core::option::Option<read_todos_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReadTodosSuccess {
	#[prost(message, repeated, tag = "1")]
	pub todos:       ::prost::alloc::vec::Vec<TodoItem>,
	#[prost(int32, tag = "2")]
	pub total_count: i32,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReadTodosError {
	#[prost(string, tag = "1")]
	pub error: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Range {
	#[prost(message, optional, tag = "1")]
	pub start: ::core::option::Option<Position>,
	#[prost(message, optional, tag = "2")]
	pub end:   ::core::option::Option<Position>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Position {
	#[prost(uint32, tag = "1")]
	pub line:   u32,
	#[prost(uint32, tag = "2")]
	pub column: u32,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Error {
	#[prost(string, tag = "1")]
	pub message: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WebSearchArgs {
	#[prost(string, tag = "1")]
	pub search_term:  ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub tool_call_id: ::prost::alloc::string::String,
}

pub mod web_search_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::WebSearchSuccess),
		#[prost(message, tag = "2")]
		Error(super::WebSearchError),
		#[prost(message, tag = "3")]
		Rejected(super::WebSearchRejected),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WebSearchResult {
	#[prost(oneof = "web_search_result::Result", tags = "1, 2, 3")]
	pub result: ::core::option::Option<web_search_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WebSearchSuccess {
	#[prost(message, repeated, tag = "1")]
	pub references: ::prost::alloc::vec::Vec<WebSearchReference>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WebSearchError {
	#[prost(string, tag = "1")]
	pub error: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WebSearchRejected {
	#[prost(string, tag = "1")]
	pub reason: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WebSearchReference {
	#[prost(string, tag = "1")]
	pub title: ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub url:   ::prost::alloc::string::String,
	#[prost(string, tag = "3")]
	pub chunk: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WebSearchToolCall {
	#[prost(message, optional, tag = "1")]
	pub args:   ::core::option::Option<WebSearchArgs>,
	#[prost(message, optional, tag = "2")]
	pub result: ::core::option::Option<WebSearchResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WebSearchRequestQuery {
	#[prost(message, optional, tag = "1")]
	pub args: ::core::option::Option<WebSearchArgs>,
}

pub mod web_search_request_response {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Approved(super::WebSearchRequestResponse_Approved),
		#[prost(message, tag = "2")]
		Rejected(super::WebSearchRequestResponse_Rejected),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WebSearchRequestResponse {
	#[prost(oneof = "web_search_request_response::Result", tags = "1, 2")]
	pub result: ::core::option::Option<web_search_request_response::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WebSearchRequestResponse_Approved {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WebSearchRequestResponse_Rejected {
	#[prost(string, tag = "1")]
	pub reason: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WriteArgs {
	#[prost(string, tag = "1")]
	pub path: ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub file_text: ::prost::alloc::string::String,
	#[prost(string, tag = "3")]
	pub tool_call_id: ::prost::alloc::string::String,
	#[prost(bool, tag = "4")]
	pub return_file_content_after_write: bool,
	#[prost(bytes = "vec", tag = "5")]
	pub file_bytes: ::prost::alloc::vec::Vec<u8>,
	#[prost(string, optional, tag = "6")]
	pub encoding_hint: ::core::option::Option<::prost::alloc::string::String>,
}

pub mod write_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::WriteSuccess),
		#[prost(message, tag = "3")]
		PermissionDenied(super::WritePermissionDenied),
		#[prost(message, tag = "4")]
		NoSpace(super::WriteNoSpace),
		#[prost(message, tag = "5")]
		Error(super::WriteError),
		#[prost(message, tag = "6")]
		Rejected(super::WriteRejected),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WriteResult {
	#[prost(oneof = "write_result::Result", tags = "1, 3, 4, 5, 6")]
	pub result: ::core::option::Option<write_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WriteSuccess {
	#[prost(string, tag = "1")]
	pub path:                     ::prost::alloc::string::String,
	#[prost(int32, tag = "2")]
	pub lines_created:            i32,
	#[prost(int32, tag = "3")]
	pub file_size:                i32,
	#[prost(string, optional, tag = "4")]
	pub file_content_after_write: ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WritePermissionDenied {
	#[prost(string, tag = "1")]
	pub path:        ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub directory:   ::prost::alloc::string::String,
	#[prost(string, tag = "3")]
	pub operation:   ::prost::alloc::string::String,
	#[prost(string, tag = "4")]
	pub error:       ::prost::alloc::string::String,
	#[prost(bool, tag = "5")]
	pub is_readonly: bool,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WriteNoSpace {
	#[prost(string, tag = "1")]
	pub path: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WriteError {
	#[prost(string, tag = "1")]
	pub path:  ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub error: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WriteRejected {
	#[prost(string, tag = "1")]
	pub path:   ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub reason: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BootstrapStatsigRequest {
	#[prost(bool, optional, tag = "1")]
	pub ignore_dev_status: ::core::option::Option<bool>,
	#[prost(int32, optional, tag = "2")]
	pub operating_system:  ::core::option::Option<i32>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PingResponse {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExecRequest {
	#[prost(string, tag = "1")]
	pub command:     ::prost::alloc::string::String,
	#[prost(string, optional, tag = "2")]
	pub cwd:         ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, repeated, tag = "3")]
	pub args:        ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
	#[prost(map = "string, string", tag = "4")]
	pub environment:
		::std::collections::HashMap<::prost::alloc::string::String, ::prost::alloc::string::String>,
}

pub mod exec_response {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Event {
		#[prost(message, tag = "1")]
		StdoutEvent(super::StdoutEvent),
		#[prost(message, tag = "2")]
		StderrEvent(super::StderrEvent),
		#[prost(message, tag = "3")]
		ExitEvent(super::ExitEvent),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExecResponse {
	#[prost(oneof = "exec_response::Event", tags = "1, 2, 3")]
	pub event: ::core::option::Option<exec_response::Event>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StdoutEvent {
	#[prost(string, tag = "1")]
	pub data: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StderrEvent {
	#[prost(string, tag = "1")]
	pub data: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExitEvent {
	#[prost(int32, tag = "1")]
	pub exit_code: i32,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReadTextFileRequest {
	#[prost(string, tag = "1")]
	pub path: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReadTextFileResponse {
	#[prost(string, tag = "1")]
	pub content: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WriteTextFileRequest {
	#[prost(string, tag = "1")]
	pub path:    ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub content: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WriteTextFileResponse {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReadBinaryFileRequest {
	#[prost(string, tag = "1")]
	pub path: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ReadBinaryFileResponse {
	#[prost(bytes = "vec", tag = "1")]
	pub content: ::prost::alloc::vec::Vec<u8>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WriteBinaryFileRequest {
	#[prost(string, tag = "1")]
	pub path:    ::prost::alloc::string::String,
	#[prost(bytes = "vec", tag = "2")]
	pub content: ::prost::alloc::vec::Vec<u8>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WriteBinaryFileResponse {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetWorkspaceChangesHashRequest {
	#[prost(string, tag = "1")]
	pub root_path: ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub base_ref:  ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetWorkspaceChangesHashResponse {
	#[prost(string, tag = "1")]
	pub hash: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RefreshGithubAccessTokenRequest {
	#[prost(string, tag = "1")]
	pub github_access_token: ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub hostname:            ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RefreshGithubAccessTokenResponse {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WarmRemoteAccessServerRequest {
	#[prost(string, tag = "1")]
	pub commit:           ::prost::alloc::string::String,
	#[prost(int32, tag = "2")]
	pub port:             i32,
	#[prost(string, tag = "3")]
	pub connection_token: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WarmRemoteAccessServerResponse {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ListArtifactsRequest {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ArtifactUploadMetadata {
	#[prost(string, tag = "1")]
	pub absolute_path:            ::prost::alloc::string::String,
	#[prost(uint64, tag = "2")]
	pub size_bytes:               u64,
	#[prost(int64, tag = "3")]
	pub updated_at_unix_ms:       i64,
	#[prost(int32, tag = "4")]
	pub status:                   i32,
	#[prost(uint64, tag = "5")]
	pub bytes_uploaded:           u64,
	#[prost(string, tag = "6")]
	pub last_error:               ::prost::alloc::string::String,
	#[prost(uint32, tag = "7")]
	pub upload_attempts:          u32,
	#[prost(int64, tag = "8")]
	pub last_started_at_unix_ms:  i64,
	#[prost(int64, tag = "9")]
	pub last_finished_at_unix_ms: i64,
	#[prost(string, tag = "10")]
	pub upload_id:                ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ListArtifactsResponse {
	#[prost(message, repeated, tag = "1")]
	pub artifacts: ::prost::alloc::vec::Vec<ArtifactUploadMetadata>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct UploadArtifactsRequest {
	#[prost(message, repeated, tag = "1")]
	pub uploads: ::prost::alloc::vec::Vec<ArtifactUploadInstruction>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ArtifactUploadInstruction {
	#[prost(string, tag = "1")]
	pub absolute_path:    ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub upload_url:       ::prost::alloc::string::String,
	#[prost(string, tag = "3")]
	pub method:           ::prost::alloc::string::String,
	#[prost(map = "string, string", tag = "4")]
	pub headers:
		::std::collections::HashMap<::prost::alloc::string::String, ::prost::alloc::string::String>,
	#[prost(string, optional, tag = "5")]
	pub content_type:     ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "6")]
	pub slack_upload_url: ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "7")]
	pub slack_file_id:    ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ArtifactUploadDispatchResult {
	#[prost(string, tag = "1")]
	pub absolute_path: ::prost::alloc::string::String,
	#[prost(int32, tag = "2")]
	pub status:        i32,
	#[prost(string, tag = "3")]
	pub message:       ::prost::alloc::string::String,
	#[prost(string, optional, tag = "4")]
	pub slack_file_id: ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct UploadArtifactsResponse {
	#[prost(message, repeated, tag = "1")]
	pub results: ::prost::alloc::vec::Vec<ArtifactUploadDispatchResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetMcpRefreshTokensRequest {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetMcpRefreshTokensResponse {
	#[prost(map = "string, string", tag = "1")]
	pub refresh_tokens:
		::std::collections::HashMap<::prost::alloc::string::String, ::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct UpdateEnvironmentVariablesRequest {
	#[prost(map = "string, string", tag = "1")]
	pub env:
		::std::collections::HashMap<::prost::alloc::string::String, ::prost::alloc::string::String>,
	#[prost(bool, tag = "2")]
	pub replace: bool,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct UpdateEnvironmentVariablesResponse {
	#[prost(uint32, tag = "1")]
	pub applied: u32,
	#[prost(uint32, tag = "2")]
	pub removed: u32,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct McpOAuthStoredData {
	#[prost(string, tag = "1")]
	pub refresh_token: ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub client_id:     ::prost::alloc::string::String,
	#[prost(string, optional, tag = "3")]
	pub client_secret: ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, repeated, tag = "4")]
	pub redirect_uris: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Frame {
	#[prost(string, tag = "1")]
	pub id:     ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub method: ::prost::alloc::string::String,
	#[prost(bytes = "vec", tag = "3")]
	pub data:   ::prost::alloc::vec::Vec<u8>,
	#[prost(int32, tag = "4")]
	pub r#kind: i32,
	#[prost(string, tag = "5")]
	pub error:  ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Empty {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BidiRequestId {
	#[prost(string, tag = "1")]
	pub request_id: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PiTruncation {
	#[prost(bool, tag = "1")]
	pub truncated:                bool,
	#[prost(string, tag = "2")]
	pub truncated_by:             ::prost::alloc::string::String,
	#[prost(uint32, tag = "3")]
	pub total_lines:              u32,
	#[prost(uint32, tag = "4")]
	pub output_lines:             u32,
	#[prost(uint32, tag = "5")]
	pub output_bytes:             u32,
	#[prost(uint32, optional, tag = "6")]
	pub max_lines:                ::core::option::Option<u32>,
	#[prost(uint32, optional, tag = "7")]
	pub max_bytes:                ::core::option::Option<u32>,
	#[prost(bool, tag = "8")]
	pub first_line_exceeds_limit: bool,
	#[prost(bool, tag = "9")]
	pub last_line_partial:        bool,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PiEditReplacement {
	#[prost(string, tag = "1")]
	pub old_text: ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub new_text: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PiReadExecArgs {
	#[prost(string, tag = "1")]
	pub path:   ::prost::alloc::string::String,
	#[prost(int32, optional, tag = "2")]
	pub offset: ::core::option::Option<i32>,
	#[prost(int32, optional, tag = "3")]
	pub limit:  ::core::option::Option<i32>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PiReadExecSuccess {
	#[prost(string, tag = "1")]
	pub output:     ::prost::alloc::string::String,
	#[prost(message, optional, tag = "2")]
	pub truncation: ::core::option::Option<PiTruncation>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PiReadExecError {
	#[prost(string, tag = "1")]
	pub error: ::prost::alloc::string::String,
}

pub mod pi_read_exec_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::PiReadExecSuccess),
		#[prost(message, tag = "2")]
		Error(super::PiReadExecError),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PiReadExecResult {
	#[prost(oneof = "pi_read_exec_result::Result", tags = "1, 2")]
	pub result: ::core::option::Option<pi_read_exec_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PiBashExecArgs {
	#[prost(string, tag = "1")]
	pub command: ::prost::alloc::string::String,
	#[prost(double, optional, tag = "2")]
	pub timeout: ::core::option::Option<f64>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PiBashExecSuccess {
	#[prost(string, tag = "1")]
	pub output:           ::prost::alloc::string::String,
	#[prost(message, optional, tag = "2")]
	pub truncation:       ::core::option::Option<PiTruncation>,
	#[prost(string, optional, tag = "3")]
	pub full_output_path: ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PiBashExecError {
	#[prost(string, tag = "1")]
	pub error:            ::prost::alloc::string::String,
	#[prost(message, optional, tag = "2")]
	pub truncation:       ::core::option::Option<PiTruncation>,
	#[prost(string, optional, tag = "3")]
	pub full_output_path: ::core::option::Option<::prost::alloc::string::String>,
}

pub mod pi_bash_exec_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::PiBashExecSuccess),
		#[prost(message, tag = "2")]
		Error(super::PiBashExecError),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PiBashExecResult {
	#[prost(oneof = "pi_bash_exec_result::Result", tags = "1, 2")]
	pub result: ::core::option::Option<pi_bash_exec_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PiEditExecArgs {
	#[prost(string, tag = "1")]
	pub path:  ::prost::alloc::string::String,
	#[prost(message, repeated, tag = "2")]
	pub edits: ::prost::alloc::vec::Vec<PiEditReplacement>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PiEditExecSuccess {
	#[prost(string, tag = "1")]
	pub output:             ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub diff:               ::prost::alloc::string::String,
	#[prost(string, tag = "3")]
	pub patch:              ::prost::alloc::string::String,
	#[prost(uint32, optional, tag = "4")]
	pub first_changed_line: ::core::option::Option<u32>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PiEditExecError {
	#[prost(string, tag = "1")]
	pub error: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PiEditExecRejected {
	#[prost(string, tag = "1")]
	pub reason: ::prost::alloc::string::String,
}

pub mod pi_edit_exec_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::PiEditExecSuccess),
		#[prost(message, tag = "2")]
		Error(super::PiEditExecError),
		#[prost(message, tag = "3")]
		Rejected(super::PiEditExecRejected),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PiEditExecResult {
	#[prost(oneof = "pi_edit_exec_result::Result", tags = "1, 2, 3")]
	pub result: ::core::option::Option<pi_edit_exec_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PiWriteExecArgs {
	#[prost(string, tag = "1")]
	pub path:    ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub content: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PiWriteExecSuccess {
	#[prost(string, tag = "1")]
	pub output: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PiWriteExecError {
	#[prost(string, tag = "1")]
	pub error: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PiWriteExecRejected {
	#[prost(string, tag = "1")]
	pub reason: ::prost::alloc::string::String,
}

pub mod pi_write_exec_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::PiWriteExecSuccess),
		#[prost(message, tag = "2")]
		Error(super::PiWriteExecError),
		#[prost(message, tag = "3")]
		Rejected(super::PiWriteExecRejected),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PiWriteExecResult {
	#[prost(oneof = "pi_write_exec_result::Result", tags = "1, 2, 3")]
	pub result: ::core::option::Option<pi_write_exec_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PiGrepExecArgs {
	#[prost(string, tag = "1")]
	pub pattern:     ::prost::alloc::string::String,
	#[prost(string, optional, tag = "2")]
	pub path:        ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "3")]
	pub glob:        ::core::option::Option<::prost::alloc::string::String>,
	#[prost(bool, optional, tag = "4")]
	pub ignore_case: ::core::option::Option<bool>,
	#[prost(bool, optional, tag = "5")]
	pub literal:     ::core::option::Option<bool>,
	#[prost(int32, optional, tag = "6")]
	pub context:     ::core::option::Option<i32>,
	#[prost(int32, optional, tag = "7")]
	pub limit:       ::core::option::Option<i32>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PiGrepExecSuccess {
	#[prost(string, tag = "1")]
	pub output:              ::prost::alloc::string::String,
	#[prost(message, optional, tag = "2")]
	pub truncation:          ::core::option::Option<PiTruncation>,
	#[prost(uint32, optional, tag = "3")]
	pub match_limit_reached: ::core::option::Option<u32>,
	#[prost(bool, tag = "4")]
	pub lines_truncated:     bool,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PiGrepExecError {
	#[prost(string, tag = "1")]
	pub error: ::prost::alloc::string::String,
}

pub mod pi_grep_exec_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::PiGrepExecSuccess),
		#[prost(message, tag = "2")]
		Error(super::PiGrepExecError),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PiGrepExecResult {
	#[prost(oneof = "pi_grep_exec_result::Result", tags = "1, 2")]
	pub result: ::core::option::Option<pi_grep_exec_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PiFindExecArgs {
	#[prost(string, tag = "1")]
	pub pattern: ::prost::alloc::string::String,
	#[prost(string, optional, tag = "2")]
	pub path:    ::core::option::Option<::prost::alloc::string::String>,
	#[prost(int32, optional, tag = "3")]
	pub limit:   ::core::option::Option<i32>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PiFindExecSuccess {
	#[prost(string, tag = "1")]
	pub output:               ::prost::alloc::string::String,
	#[prost(message, optional, tag = "2")]
	pub truncation:           ::core::option::Option<PiTruncation>,
	#[prost(uint32, optional, tag = "3")]
	pub result_limit_reached: ::core::option::Option<u32>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PiFindExecError {
	#[prost(string, tag = "1")]
	pub error: ::prost::alloc::string::String,
}

pub mod pi_find_exec_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::PiFindExecSuccess),
		#[prost(message, tag = "2")]
		Error(super::PiFindExecError),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PiFindExecResult {
	#[prost(oneof = "pi_find_exec_result::Result", tags = "1, 2")]
	pub result: ::core::option::Option<pi_find_exec_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PiLsExecArgs {
	#[prost(string, optional, tag = "1")]
	pub path:  ::core::option::Option<::prost::alloc::string::String>,
	#[prost(int32, optional, tag = "2")]
	pub limit: ::core::option::Option<i32>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PiLsExecSuccess {
	#[prost(string, tag = "1")]
	pub output:              ::prost::alloc::string::String,
	#[prost(message, optional, tag = "2")]
	pub truncation:          ::core::option::Option<PiTruncation>,
	#[prost(uint32, optional, tag = "3")]
	pub entry_limit_reached: ::core::option::Option<u32>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PiLsExecError {
	#[prost(string, tag = "1")]
	pub error: ::prost::alloc::string::String,
}

pub mod pi_ls_exec_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::PiLsExecSuccess),
		#[prost(message, tag = "2")]
		Error(super::PiLsExecError),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PiLsExecResult {
	#[prost(oneof = "pi_ls_exec_result::Result", tags = "1, 2")]
	pub result: ::core::option::Option<pi_ls_exec_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct McpServerNotFound {
	#[prost(string, tag = "1")]
	pub name:              ::prost::alloc::string::String,
	#[prost(string, repeated, tag = "2")]
	pub available_servers: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct McpApproved {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct McpStateExecArgs {
	#[prost(string, repeated, tag = "1")]
	pub server_identifiers: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
	#[prost(bool, tag = "2")]
	pub kick_only:          bool,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct McpStateServer {
	#[prost(string, tag = "1")]
	pub server_name:       ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub server_identifier: ::prost::alloc::string::String,
	#[prost(string, optional, tag = "3")]
	pub plugin:            ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "4")]
	pub marketplace:       ::core::option::Option<::prost::alloc::string::String>,
	#[prost(message, repeated, tag = "5")]
	pub tools:             ::prost::alloc::vec::Vec<McpToolDefinition>,
	#[prost(message, repeated, tag = "6")]
	pub instructions:      ::prost::alloc::vec::Vec<McpInstructions>,
	#[prost(string, optional, tag = "7")]
	pub status:            ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct McpStateSuccess {
	#[prost(message, repeated, tag = "1")]
	pub servers: ::prost::alloc::vec::Vec<McpStateServer>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct McpStateError {
	#[prost(string, tag = "1")]
	pub error: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct McpStateRejected {
	#[prost(string, tag = "1")]
	pub reason: ::prost::alloc::string::String,
}

pub mod mcp_state_exec_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::McpStateSuccess),
		#[prost(message, tag = "2")]
		Error(super::McpStateError),
		#[prost(message, tag = "3")]
		Rejected(super::McpStateRejected),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct McpStateExecResult {
	#[prost(oneof = "mcp_state_exec_result::Result", tags = "1, 2, 3")]
	pub result: ::core::option::Option<mcp_state_exec_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CommandClassifierResult_ClassifiedCommand {
	#[prost(string, tag = "1")]
	pub name: ::prost::alloc::string::String,
	#[prost(string, repeated, tag = "2")]
	pub arguments: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "3")]
	pub suggested_allowlist_entry: ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, repeated, tag = "4")]
	pub subcommand_tokens: ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CommandClassifierResult {
	#[prost(message, repeated, tag = "1")]
	pub commands:               ::prost::alloc::vec::Vec<CommandClassifierResult_ClassifiedCommand>,
	#[prost(enumeration = "CommandClassifierResult_SuggestedSandboxMode", tag = "2")]
	pub suggested_sandbox_mode: i32,
	#[prost(bool, tag = "3")]
	pub classification_failed:  bool,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ShellCommandParsingResult_Redirect {
	#[prost(string, tag = "1")]
	pub operator:         ::prost::alloc::string::String,
	#[prost(uint32, repeated, tag = "2")]
	pub destination_fds:  ::prost::alloc::vec::Vec<u32>,
	#[prost(string, tag = "3")]
	pub target_node_type: ::prost::alloc::string::String,
	#[prost(string, optional, tag = "4")]
	pub target_text:      ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ShellAllowlistPrecheckArgs {
	#[prost(string, tag = "1")]
	pub command:           ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub working_directory: ::prost::alloc::string::String,
	#[prost(message, optional, tag = "3")]
	pub parsing_result:    ::core::option::Option<ShellCommandParsingResult>,
	#[prost(message, optional, tag = "4")]
	pub classifier_result: ::core::option::Option<CommandClassifierResult>,
	#[prost(string, optional, tag = "5")]
	pub tool_call_id:      ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ShellAllowlistPrecheckResult {
	#[prost(bool, tag = "1")]
	pub allowlisted: bool,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct McpAllowlistPrecheckArgs {
	#[prost(string, tag = "1")]
	pub provider_identifier: ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub tool_name:           ::prost::alloc::string::String,
	#[prost(string, optional, tag = "3")]
	pub tool_call_id:        ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct McpAllowlistPrecheckResult {
	#[prost(bool, tag = "1")]
	pub allowlisted: bool,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WebFetchAllowlistPrecheckArgs {
	#[prost(string, tag = "1")]
	pub url:          ::prost::alloc::string::String,
	#[prost(string, optional, tag = "2")]
	pub tool_call_id: ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WebFetchAllowlistPrecheckResult {
	#[prost(bool, tag = "1")]
	pub allowlisted: bool,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SmartModeApproval {
	#[prost(string, tag = "1")]
	pub request_id: ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub reason:     ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ShellOutputNotificationConfig {
	#[prost(string, tag = "1")]
	pub pattern:            ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub reason:             ::prost::alloc::string::String,
	#[prost(double, optional, tag = "3")]
	pub debounce:           ::core::option::Option<f64>,
	#[prost(int32, optional, tag = "4")]
	pub notification_limit: ::core::option::Option<i32>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ShellHookApprovalRequirement {
	#[prost(enumeration = "ShellHookApprovalRequirement_Kind", tag = "1")]
	pub r#kind: i32,
	#[prost(string, optional, tag = "2")]
	pub reason: ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ForceBackgroundShellArgs {
	#[prost(string, tag = "1")]
	pub tool_call_id: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ForceBackgroundShellResult {
	#[prost(enumeration = "ForceBackgroundShellStatus", tag = "1")]
	pub status:       i32,
	#[prost(message, optional, tag = "2")]
	pub shell_result: ::core::option::Option<ShellResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct HookAdditionalContext {
	#[prost(string, tag = "1")]
	pub hook_event_name: ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub content:         ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ShellStreamHookContext {
	#[prost(message, repeated, tag = "1")]
	pub hook_additional_contexts: ::prost::alloc::vec::Vec<HookAdditionalContext>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SubagentArgs {
	#[prost(string, tag = "1")]
	pub tool_call_id:  ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub subagent_type: ::prost::alloc::string::String,
	#[prost(string, tag = "4")]
	pub prompt:        ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SubagentSuccess {
	#[prost(string, tag = "1")]
	pub agent_id:          ::prost::alloc::string::String,
	#[prost(string, optional, tag = "2")]
	pub final_message:     ::core::option::Option<::prost::alloc::string::String>,
	#[prost(int32, tag = "3")]
	pub tool_call_count:   i32,
	#[prost(enumeration = "SubagentBackgroundReason", tag = "4")]
	pub background_reason: i32,
	#[prost(string, optional, tag = "5")]
	pub transcript_path:   ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SubagentError {
	#[prost(string, optional, tag = "1")]
	pub agent_id: ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, tag = "2")]
	pub error:    ::prost::alloc::string::String,
}

pub mod subagent_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::SubagentSuccess),
		#[prost(message, tag = "2")]
		Error(super::SubagentError),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SubagentResult {
	#[prost(oneof = "subagent_result::Result", tags = "1, 2")]
	pub result: ::core::option::Option<subagent_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SubagentAwaitArgs {
	#[prost(string, tag = "1")]
	pub agent_id:   ::prost::alloc::string::String,
	#[prost(uint32, tag = "2")]
	pub timeout_ms: u32,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SubagentAwaitComplete {
	#[prost(string, tag = "1")]
	pub agent_id:        ::prost::alloc::string::String,
	#[prost(string, optional, tag = "2")]
	pub transcript_path: ::core::option::Option<::prost::alloc::string::String>,
	#[prost(int32, tag = "3")]
	pub tool_call_count: i32,
	#[prost(string, optional, tag = "4")]
	pub final_message:   ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SubagentAwaitStillRunning {
	#[prost(string, tag = "1")]
	pub agent_id:        ::prost::alloc::string::String,
	#[prost(string, optional, tag = "2")]
	pub transcript_path: ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SubagentAwaitNotFound {
	#[prost(string, tag = "1")]
	pub agent_id: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SubagentAwaitError {
	#[prost(string, optional, tag = "1")]
	pub agent_id: ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, tag = "2")]
	pub error:    ::prost::alloc::string::String,
}

pub mod subagent_await_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Complete(super::SubagentAwaitComplete),
		#[prost(message, tag = "2")]
		StillRunning(super::SubagentAwaitStillRunning),
		#[prost(message, tag = "3")]
		NotFound(super::SubagentAwaitNotFound),
		#[prost(message, tag = "4")]
		Error(super::SubagentAwaitError),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SubagentAwaitResult {
	#[prost(oneof = "subagent_await_result::Result", tags = "1, 2, 3, 4")]
	pub result: ::core::option::Option<subagent_await_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ForceBackgroundSubagentArgs {
	#[prost(string, tag = "1")]
	pub tool_call_id: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ForceBackgroundSubagentResult {
	#[prost(enumeration = "ForceBackgroundSubagentStatus", tag = "1")]
	pub status: i32,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PreCompactRequestQuery {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SubagentStartRequestQuery {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SubagentStopRequestQuery {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PreToolUseRequestQuery {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PostToolUseRequestQuery {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PostToolUseFailureRequestQuery {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BeforeSubmitPromptRequestQuery {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AfterAgentResponseRequestQuery {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AfterAgentThoughtRequestQuery {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StopRequestQuery {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PreCompactRequestResponse {
	#[prost(string, optional, tag = "1")]
	pub user_message: ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SubagentStartRequestResponse {
	#[prost(string, optional, tag = "1")]
	pub permission:         ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "2")]
	pub user_message:       ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "3")]
	pub additional_context: ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SubagentStopRequestResponse {
	#[prost(string, optional, tag = "1")]
	pub followup_message:   ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "2")]
	pub additional_context: ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PreToolUseRequestResponse {
	#[prost(string, optional, tag = "1")]
	pub permission:         ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "2")]
	pub user_message:       ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "3")]
	pub agent_message:      ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "4")]
	pub updated_input:      ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "5")]
	pub additional_context: ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PostToolUseRequestResponse {
	#[prost(string, optional, tag = "1")]
	pub additional_context: ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PostToolUseFailureRequestResponse {
	#[prost(string, optional, tag = "1")]
	pub additional_context: ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct BeforeSubmitPromptRequestResponse {
	#[prost(bool, optional, tag = "1")]
	pub r#continue:         ::core::option::Option<bool>,
	#[prost(string, optional, tag = "2")]
	pub user_message:       ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "3")]
	pub additional_context: ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AfterAgentResponseRequestResponse {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AfterAgentThoughtRequestResponse {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StopRequestResponse {
	#[prost(string, optional, tag = "1")]
	pub followup_message: ::core::option::Option<::prost::alloc::string::String>,
}

pub mod execute_hook_request {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Request {
		#[prost(message, tag = "1")]
		PreCompact(super::PreCompactRequestQuery),
		#[prost(message, tag = "2")]
		SubagentStart(super::SubagentStartRequestQuery),
		#[prost(message, tag = "3")]
		SubagentStop(super::SubagentStopRequestQuery),
		#[prost(message, tag = "4")]
		PreToolUse(super::PreToolUseRequestQuery),
		#[prost(message, tag = "5")]
		PostToolUse(super::PostToolUseRequestQuery),
		#[prost(message, tag = "6")]
		PostToolUseFailure(super::PostToolUseFailureRequestQuery),
		#[prost(message, tag = "7")]
		BeforeSubmitPrompt(super::BeforeSubmitPromptRequestQuery),
		#[prost(message, tag = "8")]
		AfterAgentResponse(super::AfterAgentResponseRequestQuery),
		#[prost(message, tag = "9")]
		AfterAgentThought(super::AfterAgentThoughtRequestQuery),
		#[prost(message, tag = "11")]
		Stop(super::StopRequestQuery),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExecuteHookRequest {
	#[prost(oneof = "execute_hook_request::Request", tags = "1, 2, 3, 4, 5, 6, 7, 8, 9, 11")]
	pub request: ::core::option::Option<execute_hook_request::Request>,
}

pub mod execute_hook_response {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Response {
		#[prost(message, tag = "1")]
		PreCompact(super::PreCompactRequestResponse),
		#[prost(message, tag = "2")]
		SubagentStart(super::SubagentStartRequestResponse),
		#[prost(message, tag = "3")]
		SubagentStop(super::SubagentStopRequestResponse),
		#[prost(message, tag = "4")]
		PreToolUse(super::PreToolUseRequestResponse),
		#[prost(message, tag = "5")]
		PostToolUse(super::PostToolUseRequestResponse),
		#[prost(message, tag = "6")]
		PostToolUseFailure(super::PostToolUseFailureRequestResponse),
		#[prost(message, tag = "7")]
		BeforeSubmitPrompt(super::BeforeSubmitPromptRequestResponse),
		#[prost(message, tag = "8")]
		AfterAgentResponse(super::AfterAgentResponseRequestResponse),
		#[prost(message, tag = "9")]
		AfterAgentThought(super::AfterAgentThoughtRequestResponse),
		#[prost(message, tag = "11")]
		Stop(super::StopRequestResponse),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExecuteHookResponse {
	#[prost(oneof = "execute_hook_response::Response", tags = "1, 2, 3, 4, 5, 6, 7, 8, 9, 11")]
	pub response: ::core::option::Option<execute_hook_response::Response>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExecuteHookArgs {
	#[prost(message, optional, tag = "1")]
	pub request: ::core::option::Option<ExecuteHookRequest>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ExecuteHookResult {
	#[prost(message, optional, tag = "1")]
	pub response: ::core::option::Option<ExecuteHookResponse>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SmartModeRiskTarget {
	#[prost(string, tag = "1")]
	pub action: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SmartModeClassifierConversationMessage {
	#[prost(string, tag = "1")]
	pub role:    ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub content: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SmartModeClassifierArgs {
	#[prost(string, tag = "1")]
	pub tool_call_id:           ::prost::alloc::string::String,
	#[prost(string, optional, tag = "2")]
	pub parent_conversation_id: ::core::option::Option<::prost::alloc::string::String>,
	#[prost(message, optional, tag = "3")]
	pub target:                 ::core::option::Option<SmartModeRiskTarget>,
	#[prost(message, repeated, tag = "4")]
	pub conversation_context:   ::prost::alloc::vec::Vec<SmartModeClassifierConversationMessage>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SmartModeClassifierSuccess {
	#[prost(enumeration = "SmartModeClassifierDecision", tag = "1")]
	pub decision:     i32,
	#[prost(string, optional, tag = "2")]
	pub block_reason: ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SmartModeClassifierError {
	#[prost(string, tag = "1")]
	pub error: ::prost::alloc::string::String,
}

pub mod smart_mode_classifier_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::SmartModeClassifierSuccess),
		#[prost(message, tag = "2")]
		Error(super::SmartModeClassifierError),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SmartModeClassifierResult {
	#[prost(oneof = "smart_mode_classifier_result::Result", tags = "1, 2")]
	pub result: ::core::option::Option<smart_mode_classifier_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CanvasDiagnosticsArgs {
	#[prost(string, tag = "1")]
	pub path:         ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub tool_call_id: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CanvasDiagnosticsSuccess {
	#[prost(string, tag = "1")]
	pub path:        ::prost::alloc::string::String,
	#[prost(message, repeated, tag = "2")]
	pub diagnostics: ::prost::alloc::vec::Vec<Diagnostic>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CanvasDiagnosticsError {
	#[prost(string, tag = "1")]
	pub path:  ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub error: ::prost::alloc::string::String,
}

pub mod canvas_diagnostics_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::CanvasDiagnosticsSuccess),
		#[prost(message, tag = "2")]
		Error(super::CanvasDiagnosticsError),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CanvasDiagnosticsResult {
	#[prost(oneof = "canvas_diagnostics_result::Result", tags = "1, 2")]
	pub result: ::core::option::Option<canvas_diagnostics_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ConversationSearchHit {
	#[prost(string, tag = "1")]
	pub conversation_id: ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub title:           ::prost::alloc::string::String,
	#[prost(enumeration = "ConversationSearchSource", tag = "3")]
	pub source:          i32,
	#[prost(int64, tag = "4")]
	pub updated_at_ms:   i64,
	#[prost(string, optional, tag = "5")]
	pub snippet:         ::core::option::Option<::prost::alloc::string::String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ConversationSearchSuccess {
	#[prost(message, repeated, tag = "1")]
	pub hits:       ::prost::alloc::vec::Vec<ConversationSearchHit>,
	#[prost(bool, tag = "2")]
	pub truncated:  bool,
	#[prost(bool, tag = "3")]
	pub partial:    bool,
	#[prost(bool, tag = "4")]
	pub rebuilding: bool,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ConversationSearchError {
	#[prost(string, tag = "1")]
	pub error: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ConversationSearchArgs {
	#[prost(string, tag = "1")]
	pub query:        ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub tool_call_id: ::prost::alloc::string::String,
	#[prost(int32, optional, tag = "3")]
	pub limit:        ::core::option::Option<i32>,
}

pub mod conversation_search_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::ConversationSearchSuccess),
		#[prost(message, tag = "2")]
		Error(super::ConversationSearchError),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ConversationSearchResult {
	#[prost(oneof = "conversation_search_result::Result", tags = "1, 2")]
	pub result: ::core::option::Option<conversation_search_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AgentStoreConflictCursor {
	#[prost(string, tag = "1")]
	pub journal_epoch: ::prost::alloc::string::String,
	#[prost(uint64, tag = "2")]
	pub seq:           u64,
	#[prost(string, tag = "3")]
	pub last_event_id: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AgentStoreConflictEvent {
	#[prost(uint32, tag = "1")]
	pub v:                 u32,
	#[prost(string, tag = "2")]
	pub event_id:          ::prost::alloc::string::String,
	#[prost(string, tag = "3")]
	pub journal_epoch:     ::prost::alloc::string::String,
	#[prost(uint64, tag = "4")]
	pub seq:               u64,
	#[prost(uint64, tag = "5")]
	pub ts_ms:             u64,
	#[prost(string, tag = "6")]
	pub r#kind:            ::prost::alloc::string::String,
	#[prost(string, optional, tag = "7")]
	pub store_id:          ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "8")]
	pub original_rel_path: ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "9")]
	pub conflict_rel_path: ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "10")]
	pub original_abs_path: ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "11")]
	pub conflict_abs_path: ::core::option::Option<::prost::alloc::string::String>,
	#[prost(uint64, optional, tag = "12")]
	pub preserved_bytes:   ::core::option::Option<u64>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AgentStoreConflictSuccess {
	#[prost(message, repeated, tag = "1")]
	pub events:      ::prost::alloc::vec::Vec<AgentStoreConflictEvent>,
	#[prost(message, optional, tag = "2")]
	pub next_cursor: ::core::option::Option<AgentStoreConflictCursor>,
	#[prost(bool, tag = "3")]
	pub gap:         bool,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AgentStoreConflictError {
	#[prost(string, tag = "1")]
	pub error: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AgentStoreConflictArgs {
	#[prost(message, optional, tag = "1")]
	pub cursor:  ::core::option::Option<AgentStoreConflictCursor>,
	#[prost(bool, optional, tag = "2")]
	pub advance: ::core::option::Option<bool>,
}

pub mod agent_store_conflict_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::AgentStoreConflictSuccess),
		#[prost(message, tag = "2")]
		Error(super::AgentStoreConflictError),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct AgentStoreConflictResult {
	#[prost(oneof = "agent_store_conflict_result::Result", tags = "1, 2")]
	pub result: ::core::option::Option<agent_store_conflict_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FileDiff_Chunk {
	#[prost(string, tag = "1")]
	pub content:   ::prost::alloc::string::String,
	#[prost(string, repeated, tag = "2")]
	pub lines:     ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
	#[prost(int32, tag = "3")]
	pub old_start: i32,
	#[prost(int32, tag = "4")]
	pub old_lines: i32,
	#[prost(int32, tag = "5")]
	pub new_start: i32,
	#[prost(int32, tag = "6")]
	pub new_lines: i32,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FileDiff {
	#[prost(int32, tag = "4")]
	pub added:                i32,
	#[prost(int32, tag = "5")]
	pub removed:              i32,
	#[prost(string, tag = "1")]
	pub from:                 ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub to:                   ::prost::alloc::string::String,
	#[prost(message, repeated, tag = "3")]
	pub chunks:               ::prost::alloc::vec::Vec<FileDiff_Chunk>,
	#[prost(string, optional, tag = "6")]
	pub before_file_contents: ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "7")]
	pub after_file_contents:  ::core::option::Option<::prost::alloc::string::String>,
	#[prost(bool, optional, tag = "8")]
	pub is_generated:         ::core::option::Option<bool>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GitDiff {
	#[prost(message, repeated, tag = "1")]
	pub diffs:     ::prost::alloc::vec::Vec<FileDiff>,
	#[prost(enumeration = "GitDiff_DiffType", tag = "2")]
	pub diff_type: i32,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetDiffResponse_SubmoduleDiff {
	#[prost(string, tag = "1")]
	pub relative_path: ::prost::alloc::string::String,
	#[prost(message, optional, tag = "2")]
	pub diff:          ::core::option::Option<GitDiff>,
	#[prost(bool, tag = "3")]
	pub errored:       bool,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetDiffRequest {
	#[prost(string, tag = "1")]
	pub cwd:                     ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub r#ref:                   ::prost::alloc::string::String,
	#[prost(string, tag = "3")]
	pub base_ref:                ::prost::alloc::string::String,
	#[prost(bool, tag = "4")]
	pub merge_base:              bool,
	#[prost(string, repeated, tag = "5")]
	pub target_paths:            ::prost::alloc::vec::Vec<::prost::alloc::string::String>,
	#[prost(int32, optional, tag = "6")]
	pub unified_context_lines:   ::core::option::Option<i32>,
	#[prost(int32, tag = "7")]
	pub max_untracked_files:     i32,
	#[prost(int32, tag = "9")]
	pub submodule_recurse_depth: i32,
	#[prost(bool, tag = "10")]
	pub include_space_changes:   bool,
	#[prost(bool, tag = "11")]
	pub committed_only:          bool,
	#[prost(bool, tag = "12")]
	pub compute_patch_id:        bool,
	#[prost(bool, optional, tag = "13")]
	pub return_head_sha:         ::core::option::Option<bool>,
	#[prost(int32, optional, tag = "14")]
	pub max_response_bytes:      ::core::option::Option<i32>,
	#[prost(enumeration = "GetDiffRequest_OutputFormat", optional, tag = "8")]
	pub output_format:           ::core::option::Option<i32>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetDiffResponse {
	#[prost(message, optional, tag = "1")]
	pub diff:                    ::core::option::Option<GitDiff>,
	#[prost(message, repeated, tag = "2")]
	pub submodule_diffs:         ::prost::alloc::vec::Vec<GetDiffResponse_SubmoduleDiff>,
	#[prost(string, optional, tag = "3")]
	pub patch_id:                ::core::option::Option<::prost::alloc::string::String>,
	#[prost(string, optional, tag = "4")]
	pub head_sha:                ::core::option::Option<::prost::alloc::string::String>,
	#[prost(bool, optional, tag = "5")]
	pub has_uncommitted_changes: ::core::option::Option<bool>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PiReadToolCall {
	#[prost(message, optional, tag = "1")]
	pub args:   ::core::option::Option<PiReadExecArgs>,
	#[prost(message, optional, tag = "2")]
	pub result: ::core::option::Option<PiReadExecResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PiBashToolCall {
	#[prost(message, optional, tag = "1")]
	pub args:   ::core::option::Option<PiBashExecArgs>,
	#[prost(message, optional, tag = "2")]
	pub result: ::core::option::Option<PiBashExecResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PiEditToolCall {
	#[prost(message, optional, tag = "1")]
	pub args:   ::core::option::Option<PiEditExecArgs>,
	#[prost(message, optional, tag = "2")]
	pub result: ::core::option::Option<PiEditExecResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PiWriteToolCall {
	#[prost(message, optional, tag = "1")]
	pub args:   ::core::option::Option<PiWriteExecArgs>,
	#[prost(message, optional, tag = "2")]
	pub result: ::core::option::Option<PiWriteExecResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PiGrepToolCall {
	#[prost(message, optional, tag = "1")]
	pub args:   ::core::option::Option<PiGrepExecArgs>,
	#[prost(message, optional, tag = "2")]
	pub result: ::core::option::Option<PiGrepExecResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PiFindToolCall {
	#[prost(message, optional, tag = "1")]
	pub args:   ::core::option::Option<PiFindExecArgs>,
	#[prost(message, optional, tag = "2")]
	pub result: ::core::option::Option<PiFindExecResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct PiLsToolCall {
	#[prost(message, optional, tag = "1")]
	pub args:   ::core::option::Option<PiLsExecArgs>,
	#[prost(message, optional, tag = "2")]
	pub result: ::core::option::Option<PiLsExecResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SearchConversationsToolCall {
	#[prost(message, optional, tag = "1")]
	pub args:   ::core::option::Option<ConversationSearchArgs>,
	#[prost(message, optional, tag = "2")]
	pub result: ::core::option::Option<ConversationSearchResult>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ConnectScmGithubRepository {
	#[prost(string, tag = "1")]
	pub owner: ::prost::alloc::string::String,
	#[prost(string, tag = "2")]
	pub repo:  ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ConnectScmGithub {
	#[prost(message, optional, tag = "1")]
	pub repository:      ::core::option::Option<ConnectScmGithubRepository>,
	#[prost(string, optional, tag = "2")]
	pub ghe_application: ::core::option::Option<::prost::alloc::string::String>,
}

pub mod connect_scm_args {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Target {
		#[prost(message, tag = "2")]
		Github(super::ConnectScmGithub),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ConnectScmArgs {
	#[prost(string, tag = "1")]
	pub tool_call_id: ::prost::alloc::string::String,
	#[prost(oneof = "connect_scm_args::Target", tags = "2")]
	pub target:       ::core::option::Option<connect_scm_args::Target>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ConnectScmSuccess {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ConnectScmError {
	#[prost(string, tag = "1")]
	pub error: ::prost::alloc::string::String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ConnectScmRejected {
	#[prost(string, tag = "1")]
	pub reason: ::prost::alloc::string::String,
}

pub mod connect_scm_result {
	#[derive(Clone, PartialEq, ::prost::Oneof)]
	pub enum Result {
		#[prost(message, tag = "1")]
		Success(super::ConnectScmSuccess),
		#[prost(message, tag = "2")]
		Error(super::ConnectScmError),
		#[prost(message, tag = "3")]
		Rejected(super::ConnectScmRejected),
	}
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ConnectScmResult {
	#[prost(oneof = "connect_scm_result::Result", tags = "1, 2, 3")]
	pub result: ::core::option::Option<connect_scm_result::Result>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ConnectScmToolCall {
	#[prost(message, optional, tag = "1")]
	pub args:   ::core::option::Option<ConnectScmArgs>,
	#[prost(message, optional, tag = "2")]
	pub result: ::core::option::Option<ConnectScmResult>,
}
