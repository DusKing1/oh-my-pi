//! Host-agnostic immediate-mode chat scene and matching overlays.
//!
//! The crate owns presentation state only. A host forwards [`Intent`] values
//! to its backend and applies [`BackendEvent`] values to [`Chat`].

#![forbid(unsafe_code)]

pub mod host;
mod overlays;
pub mod palette;
pub mod picker;
pub mod provider_picker;
pub mod scene;
pub mod sidebar;
pub mod welcome;

use std::time::Instant;

use omp_core::Str;
pub use omp_tui::components::Attachment;
pub use overlays::{ListPicker, ListRow, PromptEvent, PromptOverlay};
pub use palette::{CommandPalette, PaletteAction, PaletteEntry, PaletteEvent};
pub use picker::{ModelPicker, PickerEvent};
pub use provider_picker::ProviderPicker;
pub use scene::{Chat, ChatKey, RenderedFrame, ToolKind};
pub use sidebar::Sidebar;
pub use welcome::{Welcome, WelcomeEvent};

/// One model shown by the model picker.
#[derive(Clone, Debug)]
pub struct ModelRow {
	/// Stable backend model key.
	pub key:         Str,
	/// Human-readable model name.
	pub name:        Str,
	/// Stable provider identifier used to resolve its packaged logo.
	pub provider_id: Str,
	/// Human-readable provider name.
	pub provider:    Str,
	/// Context-window size in tokens, when known.
	pub context:     Option<u64>,
	/// Input price in dollars per million tokens, when known.
	pub input_mtok:  Option<f64>,
	/// Output price in dollars per million tokens, when known.
	pub output_mtok: Option<f64>,
}

/// One resumable session shown by a list picker.
#[derive(Clone, Debug)]
pub struct SessionRow {
	/// Stable session identifier.
	pub id:     Str,
	/// Primary display label.
	pub label:  Str,
	/// Secondary display detail.
	pub detail: Str,
}

/// Optional repository facts for the status line.
#[derive(Clone, Debug)]
pub struct GitFacts {
	/// Current branch name.
	pub branch: Str,
	/// Number of dirty paths.
	pub dirty:  u32,
	/// Number of staged paths.
	pub staged: u32,
}

/// Complete host-supplied status snapshot.
#[derive(Clone, Debug)]
pub struct StatusFacts {
	/// Model label shown in the status line.
	pub model:          Str,
	/// Whether a backend turn is active.
	pub working:        bool,
	/// Wall-clock start of the active turn, when available.
	pub turn_started:   Option<Instant>,
	/// Context tokens currently in use.
	pub context_tokens: u64,
	/// Model context window, when known.
	pub context_window: Option<u64>,
	/// Accumulated cost in billionths of a dollar.
	pub cost_nanos:     u64,
	/// Number of queued user submissions.
	pub queued:         usize,
	/// Number of active background jobs.
	pub jobs:           usize,
	/// Current retry attempt.
	pub attempt:        u32,
	/// Number of dropped backend events.
	pub dropped:        u64,
	/// Repository facts, omitted when unavailable.
	pub git:            Option<GitFacts>,
}

impl Default for StatusFacts {
	fn default() -> Self {
		Self {
			model:          Str::default(),
			working:        false,
			turn_started:   None,
			context_tokens: 0,
			context_window: None,
			cost_nanos:     0,
			queued:         0,
			jobs:           0,
			attempt:        0,
			dropped:        0,
			git:            None,
		}
	}
}

/// How a composer submission interacts with an active turn.
///
/// Idle backends treat both modes as a plain submission; the distinction
/// only matters while a turn is running.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubmitMode {
	/// Enter: steer the active turn by delivering the message immediately.
	Steer,
	/// Alt+Enter: queue the message as a follow-up after the active turn.
	FollowUp,
}

/// Outbound intent for the host to forward to its backend.
#[derive(Clone)]
pub enum Intent {
	/// Submit composer text and staged attachments.
	Submit {
		/// User-authored composer text.
		text:        String,
		/// Attachments staged with the submission.
		attachments: Vec<Attachment>,
		/// Active-turn delivery discipline for this submission.
		mode:        SubmitMode,
	},
	/// Abort the active turn.
	Abort,
	/// Ask the backend for rewind targets.
	RewindRequest,
	/// Rewind the durable transcript to an event.
	Rewind {
		/// Event to keep as the new live-chain tail.
		event: u64,
	},
	/// Switch the active model.
	SwitchModel(Str),
	/// Start login, optionally for a specific provider.
	Login(Option<Str>),
	/// Answer the active authentication prompt.
	AuthAnswer {
		/// Unmasked value entered by the user.
		value: String,
	},
	/// Cancel the active authentication prompt.
	AuthCancel,
	/// Resume a session, or request the session picker when absent.
	Resume(Option<Str>),
	/// Start a fresh session.
	NewSession,
	/// Show help.
	Help,
	/// Quit the host.
	Quit,
}

/// One user-message target offered by history rewind.
#[derive(Clone, Debug)]
pub struct RewindTargetRow {
	/// Durable event index to keep.
	pub event: u64,
	/// Full user message text.
	pub text:  Str,
}

/// Inbound mutation emitted by a backend.
#[derive(Clone)]
pub enum BackendEvent {
	/// Replay a user message from durable history.
	UserReplayed {
		/// Message text.
		text:  Str,
		/// Display labels for replayed attachments.
		chips: Vec<Str>,
	},
	/// Begin a streamed assistant message.
	AssistantBegin {
		/// Stable message identifier.
		id: Str,
	},
	/// Append text to a streamed assistant message.
	AssistantDelta {
		/// Stable message identifier.
		id:   Str,
		/// Delta text.
		text: Str,
	},
	/// Finish a streamed assistant message.
	AssistantEnd {
		/// Stable message identifier.
		id: Str,
	},
	/// Begin a streamed tool invocation.
	ToolStarted {
		/// Stable tool-call identifier.
		id:    Str,
		/// Backend tool name.
		name:  Str,
		/// Human-readable tool title.
		title: Str,
	},
	/// Append output to a live tool invocation.
	ToolOutput {
		/// Stable tool-call identifier.
		id:    Str,
		/// Output chunk.
		chunk: Str,
	},
	/// Finish a tool invocation.
	ToolFinished {
		/// Stable tool-call identifier.
		id:      Str,
		/// Whether the invocation succeeded.
		ok:      bool,
		/// Summary lines shown in the committed card.
		summary: Vec<Str>,
	},
	/// Append an informational notice.
	Notice(Str),
	/// Append an error notice.
	Error(Str),
	/// Replace status facts.
	Status(StatusFacts),
	/// Replace the session title.
	SessionTitle(Str),
	/// Open the model picker with these rows and current selection.
	OpenModelPicker {
		/// Available models.
		rows:    Vec<ModelRow>,
		/// Current model index.
		current: usize,
	},
	/// Silently refresh cached model rows and the current selection.
	ModelsUpdated {
		/// Available models.
		rows:    Vec<ModelRow>,
		/// Current model index.
		current: usize,
	},
	/// Replace resumable sessions.
	Sessions(Vec<SessionRow>),
	/// Replace provider-login choices; each row's `id` is the provider key.
	LoginProviders(Vec<SessionRow>),
	/// Replace rewind choices.
	RewindTargets(Vec<RewindTargetRow>),
	/// Open a backend authentication prompt.
	AuthPrompt {
		/// Prompt title or message.
		message: Str,
		/// Whether input must be masked.
		masked:  bool,
	},
	/// Close the active authentication prompt.
	AuthPromptClose,
	/// Remove all transcript history.
	HistoryCleared,
	/// Acknowledge the active submission.
	Ack {
		/// Whether the submission ended by interruption.
		interrupted: bool,
	},
}
