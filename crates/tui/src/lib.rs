#![doc = include_str!("../README.md")]
#![warn(missing_docs)]
//!
//! The crate covers terminal lifecycle and capability detection, including
//! graphics protocol support and desktop notifications.

extern crate self as omp_tui;

/// Time-driven animation primitives: easing, tweens, and frame cycles.
pub mod anim;
mod color;
mod component;
/// Built-in layout, text, navigation, data, and input components.
pub mod components;
mod context;
mod debug;
mod editcore;
mod escape;
mod frame;
mod graphics;
mod icons;
/// Image format dimension probing without full decodes.
pub mod imagefmt;
mod imagereg;
mod input;
mod iterm2;
mod kitty;
pub mod latex;
pub mod markdown;
mod markup;
mod notify;
mod overlay;
/// Terminal protocol, dropped-path, and native clipboard paste handling.
pub mod paste;
mod props;
mod pump;
mod renderer;
mod rich;
mod runtime;
/// Raytraced braille scenes.
///
/// Provides vector math, an orbit camera, and a rasterizer.
pub mod scene;
/// CPU fragment-shader effects packed into half-block cells.
pub mod shader;
mod sixel;
pub mod syntax;
mod terminal;
#[doc(hidden)]
pub mod test_support;
mod tty;
/// Stable controlling-terminal identity helpers.
pub mod ttyid;
mod ui;
/// Parent-process watchdogs for terminal-owning applications.
pub mod watchdog;

pub use color::{CssColor, SystemColor};
pub use component::{
	Cached, Component, ElementFactory, Elements, EventCtx, Flow, Hit, HitTag, IntoChildren,
	IntoComponent, PaintCtx, Slot, next_slot,
};
pub use context::{Appearance, Charset, Graphics, Grid, JamoWidth, Theme, UiContext};
pub use debug::respond_debug_query;
pub use editcore::{
	BufferOutcome, Command, CompletionEdit, EditBuffer, EditOutcome, Editor, EditorCompletion,
	EditorOptions, Picker, SlashCommands, Suggestion, SuggestionDisplay, SuggestionList,
	Suggestions, TabAction, VisualRow,
};
pub use frame::{
	Cell, CellContent, Color, Decor, DecorFill, DecorKind, Frame, Gradient, LinkId, Rect, Size, Style,
	StyleSpec, with_link_url,
};
pub use graphics::{
	NotifyProtocol, ProbeParser, ProbeResults, TerminalCaps, TerminalId, TerminalPlatform, detect,
	detect_from, negotiate, negotiate_async, probe_terminal,
};
pub use icons::Icon;
pub use imagefmt::ImageFormat;
/// Returns registered PNG bytes for renderer-side image upload.
pub use imagereg::bytes as image_bytes;
pub use input::{
	Chord, InputDecoder, InputEvent, Key, Keymap, Mods, Mouse, MouseButton, MouseReport,
	TerminalResponse, UiEvent, decode_keys,
};
pub use markup::{Border, Dim, ParseError};
pub use notify::{
	Notification, NotificationAction, NotificationBuilder, NotificationSound, Urgency, notify,
};
/// Builds a component tree from declarative markup.
pub use omp_tui_macros::dom;
pub use overlay::{Layer, OverlayAnchor, OverlayBand, OverlayId, OverlayMargin, OverlayOptions};
pub use paste::{Pasted, PastedImage};
pub use props::{Prop, PropValue, Props};
pub use pump::{DebugOp, DebugQuery, TerminalEvent};
pub use renderer::{OutputState, PaintStats, Renderer};
pub use rich::{
	Clip, Measure, Pipeline, Prefix, Prefixed, Restyle, RichSink, RichText, Rows, Tee, Wrap,
	decompose,
};
pub use runtime::{App, AppEnv, AppEvent, AppOptions, ImageLoader, UiHandle};
pub use terminal::{AltScreenUse, CursorStyle, Progress, Terminal, TerminalOptions};
pub use tty::TtyOut;
pub use ui::Ui;
