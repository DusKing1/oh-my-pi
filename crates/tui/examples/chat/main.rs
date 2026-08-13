//! Interactive chat demo for the TUI component gallery.
mod commands;
mod demo;
mod picker;
mod sidebar;
mod welcome;

use std::{io, time::Duration};

use omp_tui::{
	AltScreenUse, Charset, CursorStyle, InputEvent, Key, Layer, Mouse, Pasted, Renderer, Size,
	Terminal, TerminalEvent, TerminalOptions, TtyOut, UiContext, detect,
	paste::{self, Clipboard, ClipboardRead},
};
use smallvec::SmallVec;
use tokio::{sync::oneshot, time::Instant};

use crate::{
	commands::{CommandPalette, PaletteAction, PaletteEvent},
	demo::{Demo, DemoKey, RenderedFrame},
	picker::{MODELS, ModelPicker, PickerEvent},
	sidebar::Sidebar,
	welcome::Welcome,
};

const FRAME_INTERVAL: Duration = Duration::from_millis(33);
const RESIZE_SETTLE: Duration = Duration::from_millis(120);
/// Ceiling for one background clipboard read; backend subprocesses cap
/// themselves at 5–8 s, so this only fires on a hung native handle.
const PASTE_READ_TIMEOUT: Duration = Duration::from_secs(10);

/// One in-flight background clipboard read (Ctrl+V / Ctrl+Shift+V): the
/// pending one-shot receiver, the requested scope, and the absolute
/// instant at which a hung reader is abandoned.
struct PasteRead {
	clipboard:  oneshot::Receiver<Option<Clipboard>>,
	scope:      ClipboardRead,
	abandon_at: Instant,
}

impl PasteRead {
	/// Kicks the blocking clipboard read onto its background thread.
	fn start(scope: ClipboardRead) -> Self {
		Self {
			clipboard: paste::spawn_clipboard_read(scope),
			scope,
			abandon_at: Instant::now() + PASTE_READ_TIMEOUT,
		}
	}
}

#[tokio::main]
async fn main() -> io::Result<()> {
	let caps = detect();
	// One detected presentation context — charset, graphics, appearance —
	// threads through every scene instead of per-module hardcodes.
	let ctx = UiContext::default().with_terminal_caps(&caps);
	let mut terminal =
		Terminal::enter(TerminalOptions::new(caps).cursor_style(CursorStyle::BlinkingBar))?;
	let mut renderer = Renderer::new(TtyOut::new()?);
	renderer.apply_caps(&caps)?;
	run(&mut terminal, &mut renderer, &ctx).await
}

/// Drives the chat scene and guarantees the teardown scrub: the final
/// inline screen persists into native scrollback once the shell resumes,
/// so the rail must not be left composited even when the loop fails.
/// Any overlay hold is released and the rail bands repaint from the raw
/// transcript on every exit — clean quit or error alike, a loop error
/// outranking a scrub failure. Only a renderer already poisoned by a
/// writer failure skips the repaint.
#[expect(
	clippy::future_not_send,
	reason = "chat components are deliberately confined to their terminal event-loop thread"
)]
async fn run<'a>(
	terminal: &'a mut Terminal,
	renderer: &'a mut Renderer<TtyOut>,
	ctx: &'a UiContext,
) -> io::Result<()> {
	let result = chat(terminal, renderer, ctx).await;
	let scrub = terminal.leave_alt().and_then(|()| renderer.clear_layers());
	result.and(scrub)
}

#[expect(
	clippy::future_not_send,
	reason = "chat components are deliberately confined to their terminal event-loop thread"
)]
async fn chat<'a>(
	terminal: &'a mut Terminal,
	renderer: &'a mut Renderer<TtyOut>,
	ctx: &'a UiContext,
) -> io::Result<()> {
	let mut viewport = terminal.size()?;
	if !run_welcome(terminal, renderer, ctx.charset, &mut viewport).await? {
		return Ok(());
	}
	// The welcome scene held the alternate screen; releasing it restores the
	// untouched shell and the chat pushes inline from a clean slate.
	terminal.leave_alt()?;

	let mut demo = Demo::new(ctx);
	let mut overlay: Option<Overlay> = None;
	let mut current_model = 0_usize;
	let started = Instant::now();
	let mut sidebar = Sidebar::new(MODELS[current_model].name, ctx);
	demo.set_right_inset(sidebar.reserved(viewport));
	{
		let rendered = demo.render(viewport);
		let layers = rail_layers(&mut sidebar, viewport, started.elapsed());
		present(renderer, rendered, viewport, &layers)?;
	}

	// Alternate-screen ownership for the chat scene: a resize gesture borrows
	// it for throwaway drag frames, an open overlay holds it for its lifetime.
	let mut drag_alt = false;
	let mut overlay_stale = false;
	let mut resize = None;
	// At most one in-flight background clipboard read (Ctrl+V/Ctrl+Shift+V).
	let mut paste_read: Option<PasteRead> = None;
	let mut next_frame = Instant::now() + FRAME_INTERVAL;
	loop {
		let paste_deadline = paste_read.as_ref().map(|read| read.abandon_at);
		tokio::select! {
			// The terminal branch pauses while a clipboard read is in flight:
			// the event mailbox buffers input in order, so an Enter typed
			// right after Ctrl+V lands *after* the paste instead of
			// submitting an empty prompt. The read below is bounded, so the
			// pause is too; retained App hosts get the finer-grained
			// per-event queue instead.
			event = terminal.next(), if paste_read.is_none() => match event? {
				TerminalEvent::Resize => {
					let now = Instant::now();
					let resized = observe_resize(terminal, &mut viewport, &mut resize, now)?;
					demo.set_right_inset(sidebar.reserved(viewport));
					if overlay.is_some() && resized {
						overlay_stale = true;
					}
				},
				TerminalEvent::Debug(_) => {},
				TerminalEvent::Closed => return Ok(()),
				TerminalEvent::Input(event) => {
					let Some(event) = user_event(terminal, renderer, event)? else {
						continue;
					};
					match event {
					InputEvent::Key(key) => {
						if overlay.is_some() {
							if key == Key::Ctrl('c') {
								break;
							}
							let event = overlay
								.as_mut()
								.expect("overlay checked above")
								.handle_key(key);
							if apply_overlay_event(
								event,
								&mut overlay,
								&mut current_model,
								terminal,
								renderer,
								&mut demo,
								&mut sidebar,
								viewport,
								started.elapsed(),
								&mut overlay_stale,
								&mut resize,
								ctx,
							)? {
								break;
							}
						} else if key == Key::Ctrl('b') {
							sidebar.toggle();
							demo.set_right_inset(sidebar.reserved(viewport));
						} else if key == Key::Ctrl('k') {
							overlay = Some(Overlay::Palette(CommandPalette::open(ctx)));
							open_overlay(
								terminal,
								renderer,
								&mut demo,
								overlay.as_mut().expect("palette just opened"),
								&mut sidebar,
								viewport,
								started.elapsed(),
								&mut drag_alt,
								&mut overlay_stale,
								&mut resize,
							)?;
						} else if sidebar.focused() {
							if key == Key::Ctrl('c') {
								break;
							}
							sidebar.handle_key(key);
						} else if key == Key::Ctrl('p') || key == Key::Alt('p') {
							overlay = Some(Overlay::Picker(ModelPicker::open(current_model, ctx)));
							open_overlay(
								terminal,
								renderer,
								&mut demo,
								overlay.as_mut().expect("picker just opened"),
								&mut sidebar,
								viewport,
								started.elapsed(),
								&mut drag_alt,
								&mut overlay_stale,
								&mut resize,
							)?;
						} else if let Some(scope) = ClipboardRead::for_key(key) {
							// The terminal did not claim the chord; read the
							// system clipboard off-thread, preferring images
							// unless the raw spelling asked for text only. A
							// failed spawn closes the channel, so the receive
							// branch below recovers input immediately.
							paste_read = Some(PasteRead::start(scope));
						} else {
							let key_result = demo.handle_key(key);
							if let Some(text) = demo.take_copied() {
								// OSC 52 first; remote/SSH-safe, with the
								// terminal's detached native fallback.
								terminal.copy_to_clipboard(&text)?;
							}
							if demo.take_switch_request() {
								overlay = Some(Overlay::Picker(ModelPicker::open(current_model, ctx)));
								open_overlay(
									terminal,
									renderer,
									&mut demo,
									overlay.as_mut().expect("picker just opened"),
									&mut sidebar,
									viewport,
									started.elapsed(),
									&mut drag_alt,
									&mut overlay_stale,
									&mut resize,
								)?;
							}
							if key_result == DemoKey::Quit {
								break;
							}
						}
						next_frame = Instant::now();
					},
					InputEvent::Paste(text) => {
						let event = overlay.as_mut().map(|active| active.handle_paste(&text));
						match event {
							Some(event) => {
								if apply_overlay_event(
									event,
									&mut overlay,
									&mut current_model,
									terminal,
									renderer,
									&mut demo,
									&mut sidebar,
									viewport,
									started.elapsed(),
									&mut overlay_stale,
									&mut resize,
									ctx,
								)? {
									break;
								}
							},
							None if sidebar.focused() => {},
							None => demo.handle_paste(&text),
						}
						next_frame = Instant::now();
					},
					InputEvent::Mouse(report) => {
						// An open overlay owns pointer input — hover, wheel,
						// and clicks route through the compositor's band,
						// never the occluded editor beneath.
						let event = overlay
							.as_mut()
							.map(|active| active.handle_mouse(report.col, report.row, report.kind, viewport));
						match event {
							Some(event) => {
								if apply_overlay_event(
									event,
									&mut overlay,
									&mut current_model,
									terminal,
									renderer,
									&mut demo,
									&mut sidebar,
									viewport,
									started.elapsed(),
									&mut overlay_stale,
									&mut resize,
									ctx,
								)? {
									break;
								}
							},
							None => {
								if !sidebar.handle_mouse(report.col, report.row, report.kind, viewport)
								{
									demo.handle_mouse(&report);
								}
							},
						}
						next_frame = Instant::now();
					},
					InputEvent::Focus(_) | InputEvent::Response(_) => {},
				}
				},
			},
			clipboard = async { (&mut paste_read.as_mut().expect("branch gated on Some").clipboard).await },
				if paste_read.is_some() =>
			{
				let read = paste_read.take().expect("branch gated on Some");
				// A closed channel (the reader thread never spawned) reads
				// as an empty clipboard.
				if let Ok(Some(clipboard)) = clipboard
					&& let Some(text) = clipboard_paste_text(clipboard)
					&& overlay.is_none()
					&& !sidebar.focused()
				{
					// Ctrl+Shift+V inserts verbatim: no attachment staging,
					// no large-paste collapse.
					match read.scope {
						ClipboardRead::Text => demo.handle_paste_raw(&text),
						ClipboardRead::Smart => demo.handle_paste(&text),
					}
					next_frame = Instant::now();
				}
			},
			// The deadline is absolute, so the frame tick recreating this
			// branch's future cannot reset it: a hung reader is abandoned and
			// terminal input re-enables. Dropping the receiver makes the
			// reader's eventual send fail; the detached thread dies with the
			// process instead of stalling shutdown.
			() = deadline(paste_deadline) => {
				paste_read = None;
			},
			() = deadline(Some(next_frame)) => {
				let now = Instant::now();
				let resized = observe_resize(terminal, &mut viewport, &mut resize, now)?;
				demo.set_right_inset(sidebar.reserved(viewport));
				if overlay.is_some() && resized {
					overlay_stale = true;
				}
				if resize.is_some() {
					// Drag frames compose exactly one viewport tail at the
					// new geometry — O(viewport) per frame; the O(history)
					// transcript reflow waits for the settle rebuild (or the
					// overlay close). Without an overlay, a width change
					// borrows the alternate screen (the inline transcript
					// rewraps underneath) while height-only churn repaints
					// in place — alt toggling on a height echo can
					// self-sustain.
					let preview = demo.render_resize_preview(viewport);
					if let Some(active) = overlay.as_mut() {
								  let mut layers = rail_layers(&mut sidebar, viewport, started.elapsed());
								  layers.push(active.layer(viewport));
								  renderer.preview_overlaid(&preview, &layers, viewport.height, "")?;
							  } else {
								  let width_changed =
									  resize.is_some_and(|state| state.width_changed);
								  let alt_enter = if drag_alt || !width_changed {
									  None
								  } else {
									  let staged = terminal.stage_alt_enter(AltScreenUse::Resize);
									  drag_alt = staged.is_some();
									  staged
								  };
								  let layers = rail_layers(&mut sidebar, viewport, started.elapsed());
								  renderer.preview_overlaid(
									  &preview,
									  &layers,
									  viewport.height,
									  alt_enter.as_deref().unwrap_or(""),
								  )?;
							  }
				} else if let Some(active) = overlay.as_mut() {
					let rendered = demo.render(viewport);
					let mut layers = rail_layers(&mut sidebar, viewport, started.elapsed());
					layers.push(active.layer(viewport));
					renderer.preview_overlaid(rendered.frame, &layers, viewport.height, "")?;
				} else {
					let rendered = demo.render(viewport);
					let layers = rail_layers(&mut sidebar, viewport, started.elapsed());
					present(renderer, rendered, viewport, &layers)?;
				}
				next_frame = now + FRAME_INTERVAL;
			},
			() = deadline(resize.map(ResizeState::deadline)) => {
				let now = Instant::now();
				if !resize.is_some_and(|state| state.settled(now)) {
					continue;
				}
				if overlay.is_some() {
					// The overlay keeps holding the alternate screen; the
					// transcript reflows once at close.
					overlay_stale = true;
					resize = None;
					continue;
				}
				demo.set_right_inset(sidebar.reserved(viewport));
				let rendered = demo.render(viewport);
				let alt_exit = if drag_alt {
					drag_alt = false;
					terminal.stage_alt_leave().unwrap_or("")
				} else {
					""
				};
				renderer.rebuild(
					rendered.frame.clone(),
					viewport.height,
					rendered.stable_rows,
					alt_exit,
				)?;
				// The rebuild repainted the raw document; recomposite the
				// rail on top without touching the fresh history.
				let layers = rail_layers(&mut sidebar, viewport, started.elapsed());
				if !layers.is_empty() {
					renderer.present_overlaid(
						rendered.frame,
						&[],
						viewport.height,
						rendered.stable_rows,
						&layers,
					)?;
				}
				resize = None;
				next_frame = now + FRAME_INTERVAL;
			},
		}
	}
	Ok(())
}

/// Animates the welcome card until the user resumes into the chat demo
/// (`Ok(true)`) or quits (`Ok(false)`), keeping `viewport` current across
/// resizes.
///
/// The scene owns the alternate screen for its whole lifetime: entry rides
/// the first card paint, mouse tracking is active throughout, and every
/// geometry change repaints in place immediately. The main screen stays
/// untouched underneath — the caller releases the hold on scene exit.
async fn run_welcome<'a>(
	terminal: &'a mut Terminal,
	renderer: &'a mut Renderer<TtyOut>,
	charset: Charset,
	viewport: &'a mut Size,
) -> io::Result<bool> {
	let mut alt_enter = terminal.stage_alt_enter(AltScreenUse::Interactive);
	let mut welcome = Welcome::new(charset);
	let started = Instant::now();
	let mut next_frame = Instant::now();
	loop {
		tokio::select! {
			event = terminal.next() => match event? {
				TerminalEvent::Resize => {
					if let Some(size) = terminal.take_resize()? {
						*viewport = size;
					}
				},
				TerminalEvent::Debug(_) => {},
				TerminalEvent::Closed => return Ok(false),
				TerminalEvent::Input(event) => {
					let Some(event) = user_event(terminal, renderer, event)? else {
						continue;
					};
					match event {
					InputEvent::Key(Key::Enter) => return Ok(true),
					InputEvent::Key(Key::Esc | Key::Ctrl('c')) => return Ok(false),
					InputEvent::Mouse(report) if matches!(report.kind, Mouse::Move | Mouse::Drag) => {
						welcome.point_at(report.col, report.row);
					},
					InputEvent::Key(_)
					| InputEvent::Mouse(_)
					| InputEvent::Paste(_)
					| InputEvent::Focus(_)
					| InputEvent::Response(_) => {},
				}
				},
			},
			() = deadline(Some(next_frame)) => {
				let now = Instant::now();
				if let Some(size) = terminal.take_resize()? {
					*viewport = size;
				}
				let frame = welcome.render(*viewport, started.elapsed());
				renderer.preview(
					frame,
					viewport.height,
					alt_enter.take().as_deref().unwrap_or(""),
				)?;
				next_frame = now + FRAME_INTERVAL;
			},
		}
	}
}

#[derive(Clone, Copy)]
struct ResizeState {
	last_event:    Instant,
	/// Whether any report in this gesture changed the width; only then does
	/// the drag borrow the alternate screen.
	width_changed: bool,
}

impl ResizeState {
	const fn new(last_event: Instant, width_changed: bool) -> Self {
		Self { last_event, width_changed }
	}

	const fn observe(&mut self, observed_at: Instant, width_changed: bool) {
		self.last_event = observed_at;
		self.width_changed |= width_changed;
	}

	fn deadline(self) -> Instant {
		self.last_event + RESIZE_SETTLE
	}

	fn settled(self, now: Instant) -> bool {
		now >= self.deadline()
	}
}

/// Consumes the latest resize — SIGWINCH or DEC 2048 in-band — and (re)arms
/// the settle window. Same-size reports outside a gesture are echoes
/// (terminals re-reporting geometry across alternate-screen toggles) and are
/// swallowed without arming a rebuild.
fn observe_resize(
	terminal: &mut Terminal,
	viewport: &mut Size,
	resize: &mut Option<ResizeState>,
	observed_at: Instant,
) -> io::Result<bool> {
	let Some(size) = terminal.take_resize()? else {
		return Ok(false);
	};
	if size == *viewport && resize.is_none() {
		return Ok(false);
	}
	let width_changed = size.width != viewport.width;
	*viewport = size;
	match resize {
		Some(state) => state.observe(observed_at, width_changed),
		None => *resize = Some(ResizeState::new(observed_at, width_changed)),
	}
	Ok(true)
}

fn user_event(
	terminal: &mut Terminal,
	renderer: &mut Renderer<TtyOut>,
	event: InputEvent,
) -> io::Result<Option<InputEvent>> {
	if terminal.handle_input_event(&event, renderer)? {
		// A consumed response may have completed an enhanced-paste (OSC
		// 5522) conversation; re-inject its payload as ordinary paste input
		// so the normal routing below stages images and text alike.
		return Ok(terminal.take_paste().and_then(|pasted| {
			let text = match pasted {
				Pasted::Text(text) => text,
				Pasted::Image(image) => image.persist().ok()?.display().to_string().into(),
			};
			Some(InputEvent::Paste(text))
		}));
	}
	Ok(Some(event))
}

/// Flattens a background clipboard read into paste text: images persist to
/// a temp file whose path routes like a file drop, and copied file paths
/// are quoted so spaces survive drop classification.
fn clipboard_paste_text(clipboard: Clipboard) -> Option<String> {
	match clipboard {
		Clipboard::Text(text) => Some(text),
		Clipboard::Image(image) => Some(image.persist().ok()?.display().to_string()),
		Clipboard::Paths(paths) => {
			let mut joined = String::new();
			for path in &paths {
				if !joined.is_empty() {
					joined.push(' ');
				}
				joined.push('"');
				joined.push_str(path);
				joined.push('"');
			}
			Some(joined)
		},
	}
}

fn present(
	renderer: &mut Renderer<TtyOut>,
	rendered: RenderedFrame<'_>,
	viewport: Size,
	layers: &[Layer<'_>],
) -> io::Result<()> {
	renderer
		.present_overlaid(
			rendered.frame,
			rendered.damage.as_slice(),
			viewport.height,
			rendered.stable_rows,
			layers,
		)
		.map(|_| ())
}

/// The session rail as a layer slice for this frame: empty when toggled
/// off or gated out by a small viewport, so callers composite it
/// unconditionally.
fn rail_layers(sidebar: &mut Sidebar, viewport: Size, elapsed: Duration) -> SmallVec<Layer<'_>, 2> {
	sidebar.layer(viewport, elapsed).into_iter().collect()
}

/// The modal scene overlay holding the alternate screen: at most one is
/// open at a time, and a palette action can swap it for the picker in
/// place — the hold transfers without leaving the alternate screen.
enum Overlay {
	Picker(ModelPicker),
	Palette(CommandPalette),
}

/// One routed overlay outcome, unified across overlay kinds.
enum OverlayEvent {
	/// Input handled; the overlay stays open.
	Consumed,
	/// Dismissed without effect.
	Close,
	/// The picker chose a model.
	Pick(usize),
	/// The palette activated an entry.
	Run(PaletteAction),
}

impl From<PickerEvent> for OverlayEvent {
	fn from(event: PickerEvent) -> Self {
		match event {
			PickerEvent::Consumed => Self::Consumed,
			PickerEvent::Close => Self::Close,
			PickerEvent::Pick(index) => Self::Pick(index),
		}
	}
}

impl From<PaletteEvent> for OverlayEvent {
	fn from(event: PaletteEvent) -> Self {
		match event {
			PaletteEvent::Consumed => Self::Consumed,
			PaletteEvent::Close => Self::Close,
			PaletteEvent::Run(action) => Self::Run(action),
		}
	}
}

impl Overlay {
	fn handle_key(&mut self, key: Key) -> OverlayEvent {
		match self {
			Self::Picker(picker) => picker.handle_key(key).into(),
			Self::Palette(palette) => palette.handle_key(key).into(),
		}
	}

	fn handle_paste(&mut self, text: &str) -> OverlayEvent {
		match self {
			Self::Picker(picker) => picker.handle_paste(text).into(),
			Self::Palette(palette) => palette.handle_paste(text).into(),
		}
	}

	fn handle_mouse(&mut self, col: u16, row: u16, kind: Mouse, viewport: Size) -> OverlayEvent {
		match self {
			Self::Picker(picker) => picker.handle_mouse(col, row, kind, viewport).into(),
			Self::Palette(palette) => palette.handle_mouse(col, row, kind, viewport).into(),
		}
	}

	fn layer(&mut self, viewport: Size) -> Layer<'_> {
		match self {
			Self::Picker(picker) => picker.layer(viewport),
			Self::Palette(palette) => palette.layer(viewport),
		}
	}
}

/// Takes the alternate screen for the overlay's lifetime: entry rides the
/// first composited paint, and a drag borrow already in flight simply
/// transfers ownership (its settled rebuild then waits for close).
#[expect(clippy::too_many_arguments, reason = "immediate-mode example threads its scene state")]
fn open_overlay(
	terminal: &mut Terminal,
	renderer: &mut Renderer<TtyOut>,
	demo: &mut Demo,
	overlay: &mut Overlay,
	sidebar: &mut Sidebar,
	viewport: Size,
	elapsed: Duration,
	drag_alt: &mut bool,
	overlay_stale: &mut bool,
	resize: &mut Option<ResizeState>,
) -> io::Result<()> {
	if *drag_alt || resize.take().is_some() {
		*drag_alt = false;
		*overlay_stale = true;
	}
	let alt_enter = terminal.stage_alt_enter(AltScreenUse::Interactive);
	let rendered = demo.render(viewport);
	let mut layers = rail_layers(sidebar, viewport, elapsed);
	layers.push(overlay.layer(viewport));
	renderer
		.preview_overlaid(
			rendered.frame,
			&layers,
			viewport.height,
			alt_enter.as_deref().unwrap_or(""),
		)
		.map(|_| ())
}

/// Releases the overlay's alternate-screen hold. Geometry churn while held
/// rebuilds native history inside the same synchronized update as the buffer
/// switch; otherwise the untouched main screen restores byte-exactly and one
/// full-viewport present revalidates changes that only ever painted the
/// alternate screen (streamed demo rows, the picked model).
#[expect(clippy::too_many_arguments, reason = "immediate-mode example threads its scene state")]
fn close_overlay(
	terminal: &mut Terminal,
	renderer: &mut Renderer<TtyOut>,
	demo: &mut Demo,
	sidebar: &mut Sidebar,
	viewport: Size,
	elapsed: Duration,
	overlay_stale: &mut bool,
	resize: &mut Option<ResizeState>,
) -> io::Result<()> {
	*resize = None;
	let rendered = demo.render(viewport);
	let layers = rail_layers(sidebar, viewport, elapsed);
	if *overlay_stale {
		*overlay_stale = false;
		let alt_exit = terminal.stage_alt_leave().unwrap_or("");
		renderer.rebuild(rendered.frame.clone(), viewport.height, rendered.stable_rows, alt_exit)?;
		if !layers.is_empty() {
			// The rebuild repainted the raw document; recomposite the rail
			// without touching the fresh history.
			renderer.present_overlaid(
				rendered.frame,
				&[],
				viewport.height,
				rendered.stable_rows,
				&layers,
			)?;
		}
	} else {
		terminal.leave_alt()?;
		renderer.present_overlaid(
			rendered.frame,
			&[(0, rendered.frame.size().height)],
			viewport.height,
			rendered.stable_rows,
			&layers,
		)?;
	}
	Ok(())
}

/// Applies one routed [`OverlayEvent`]: a model pick updates the session
/// model, a palette action executes (`Switch Model` swaps the palette for
/// the picker in place, keeping the alternate-screen hold), and every
/// terminal event releases the hold. Returns `true` when the host should
/// quit.
#[expect(clippy::too_many_arguments, reason = "immediate-mode example threads its scene state")]
fn apply_overlay_event(
	event: OverlayEvent,
	overlay: &mut Option<Overlay>,
	current_model: &mut usize,
	terminal: &mut Terminal,
	renderer: &mut Renderer<TtyOut>,
	demo: &mut Demo,
	sidebar: &mut Sidebar,
	viewport: Size,
	elapsed: Duration,
	overlay_stale: &mut bool,
	resize: &mut Option<ResizeState>,
	ctx: &UiContext,
) -> io::Result<bool> {
	let close = match event {
		OverlayEvent::Close => true,
		OverlayEvent::Pick(index) => {
			*current_model = index;
			demo.set_model(MODELS[index].name);
			sidebar.set_model(MODELS[index].name);
			true
		},
		OverlayEvent::Run(action) => match action {
			PaletteAction::SwitchModel => {
				let picker = overlay.insert(Overlay::Picker(ModelPicker::open(*current_model, ctx)));
				// The palette already holds an interactive alternate-screen
				// grant, so no drag borrow can be in flight: the staged alt
				// entry inside `open_overlay` is a no-op and the hold
				// transfers to the picker.
				let mut drag_alt = false;
				open_overlay(
					terminal,
					renderer,
					demo,
					picker,
					sidebar,
					viewport,
					elapsed,
					&mut drag_alt,
					overlay_stale,
					resize,
				)?;
				return Ok(false);
			},
			PaletteAction::ToggleSidebar => {
				sidebar.toggle();
				true
			},
			PaletteAction::Quit => return Ok(true),
			PaletteAction::Insert(text) => {
				// Stage the slash command in the composer; the close
				// repaint below surfaces it immediately.
				demo.handle_paste(&text);
				true
			},
		},
		OverlayEvent::Consumed => false,
	};
	if close {
		*overlay = None;
		close_overlay(terminal, renderer, demo, sidebar, viewport, elapsed, overlay_stale, resize)?;
	}
	Ok(false)
}

/// Sleeps until `at`; `None` disables the select branch.
async fn deadline(at: Option<Instant>) {
	match at {
		Some(at) => tokio::time::sleep_until(at).await,
		None => std::future::pending().await,
	}
}

#[cfg(test)]
mod tests {
	use super::{Duration, Instant, RESIZE_SETTLE, ResizeState};

	#[test]
	fn resize_settle_window_restarts_at_each_event() {
		let started_at = Instant::now();
		let mut state = ResizeState::new(started_at, false);
		state.observe(started_at + Duration::from_millis(100), true);

		assert!(!state.settled(started_at + Duration::from_millis(219)));
		assert!(state.settled(started_at + Duration::from_millis(220)));
		assert_eq!(state.deadline(), started_at + Duration::from_millis(100) + RESIZE_SETTLE);
		assert!(state.width_changed, "a width report anywhere in the gesture latches the borrow");
	}
}
