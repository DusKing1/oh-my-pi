//! Transport-neutral foundations for durable, interruptible OMP agent loops.
//!
//! The crate composes immutable configuration snapshots, deterministic system
//! prompts, ordered interrupts, event fan-out, journal projection, tool-batch
//! supervision, detached jobs, and the live turn transport. Durable history is
//! canonical [`Item`] data; provider, application, and UI types stay outside
//! this boundary. [`Agent`] is the durable policy loop tying these foundations
//! into complete N-turn conversations.

mod batch;
pub(crate) mod duplex;
mod events;
mod inproc;
mod jobs;
mod journal;
mod r#loop;
mod mailbox;
mod project;
mod prompt;
mod state;
mod supervise;
mod turn;

pub use batch::{BatchError, BatchResult, CommittedCall, SpeculativeCall, ToolBatch};
pub use events::{AgentEvent, AgentPhase, EventBus, EventSubscription, LossyEventSubscription};
pub use inproc::{InProcTurnClient, RpcTurnClient, RpcTurnSession};
pub use jobs::{JobBoard, PendingJobs};
pub use journal::{Journal, JournalError, TurnInputRecord, TurnOptionsRecord, TurnReceipt, TurnStart};
pub use r#loop::{Agent, AgentError, AgentRunSummary};
pub use mailbox::{DrainPoint, Interrupt, InterruptClass, InterruptSource, Mailbox, MailboxSender};
pub use omp_llm_inference::TurnId;
pub use omp_proto::{
	inference::v1::{
		Accepted, ChatParams, ContextRef, ExecStatus, Executor, Invoke, InvokeCancel, InvokeComplete,
		InvokeInput, Outcome, ThreadDelta, TurnError, TurnEvent,
	},
	thread::v1::{Item, Thread},
};
pub use project::{
	ProjectionError, project_journal, project_thread_history, tool_result_item,
	tool_result_item_canonical_parts,
};
pub use prompt::{
	ContextFile, PromptError, PromptHash, PromptSource, RenderedPrompt, VcsIdentity, WorkspaceInput,
	WorkspacePromptSource, render_prompt,
};
pub use state::{AgentSnapshot, AgentState, RetryPolicy, RetryPolicyError};
pub use turn::{Error, InvokeFrame, TurnClient, TurnInput, TurnOptions, TurnSession};
