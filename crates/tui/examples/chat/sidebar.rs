//! Persistent session rail: a non-modal sidebar beside the chat transcript.
//!
//! Unlike the modal picker, the rail never holds the alternate screen: it
//! rides every inline present as a raw [`Layer`], so the transcript keeps
//! committing to native scrollback beneath it and history receives the
//! full-width document — never a sidebar cell. `Ctrl+B` toggles the rail
//! and hands it the keyboard (arrow keys drive the file list); `Esc`
//! returns typing to the composer ([`Ui::focus_first`] / [`Ui::blur`], the
//! raw-frame halves of the keyboard hand-off). Clicks move the keyboard
//! the same way wherever the session reports the pointer — this demo
//! leaves the inline mouse to the terminal for native text selection, so
//! that path is live during alternate-screen scenes.

use std::time::Duration;

use omp_core::{SmolStr, format_smol};
use omp_tui::{
	Color, Dim, Key, Layer, Mouse, OverlayAnchor, OverlayOptions, Prop, Size, Ui, UiContext,
	UiEvent, dom,
};

const CYAN: Color = Color::Rgb(62, 190, 203);
const GREEN: Color = Color::Rgb(81, 196, 112);
const DIM: Color = Color::Rgb(110, 116, 124);

/// Rail width in cells, vertical rule included.
const WIDTH: u16 = 30;
/// Smallest viewport the rail composites in; below it the band gates out
/// and the transcript gets the whole width back.
const MIN_VIEWPORT: Size = Size::new(96, 20);

/// Files the demo session pretends to touch while "implementing immutable
/// seam commits".
const FILES: [(&str, &str); 5] = [
	("renderer.rs", "+142"),
	("frame.rs", "+38"),
	("seam.rs", "+210"),
	("commit_tests.rs", "+96"),
	("seams.md", "+17"),
];

/// Retained session rail composited as a right-anchored viewport layer.
pub struct Sidebar {
	ui:              Ui,
	options:         OverlayOptions,
	/// Whether the rail is composited at all (`Ctrl+B`).
	open:            bool,
	/// Whether the rail holds the keyboard (arrow keys drive the file list).
	focused:         bool,
	elapsed_seconds: u64,
	height:          u16,
}

impl Sidebar {
	/// Builds the rail, presenting through the host's detected context.
	pub fn new(model: &str, ctx: &UiContext) -> Self {
		let options = OverlayOptions::default()
			.anchor(OverlayAnchor::Right)
			.width(Dim::Cells(WIDTH))
			.non_modal()
			.min_viewport(MIN_VIEWPORT);
		let mut ui = build(model, ctx);
		// The rail starts without the keyboard: no focus chrome or frame
		// cursor until `toggle` or a click hands it over.
		ui.blur();
		Self { ui, options, open: true, focused: false, elapsed_seconds: 0, height: 0 }
	}

	/// Whether the rail composites for `viewport`.
	const fn visible(&self, viewport: Size) -> bool {
		self.open && viewport.width >= MIN_VIEWPORT.width && viewport.height >= MIN_VIEWPORT.height
	}

	/// Columns the rail reserves at `viewport`: its full width while
	/// composited, zero when toggled off or gated out. The composer docks
	/// its right-aligned chrome against the remaining width.
	pub const fn reserved(&self, viewport: Size) -> u16 {
		if self.visible(viewport) { WIDTH } else { 0 }
	}

	/// Whether the rail currently holds the keyboard.
	pub const fn focused(&self) -> bool {
		self.focused
	}

	/// `Ctrl+B`: opening hands the rail the keyboard, closing returns it.
	pub fn toggle(&mut self) {
		self.open = !self.open;
		if self.open {
			self.focused = true;
			self.ui.focus_first();
		} else {
			self.blur();
		}
	}

	/// Routes a key while the rail holds the keyboard; `Esc` hands it back.
	pub fn handle_key(&mut self, key: Key) {
		if self.ui.handle_key(key) == UiEvent::Cancel {
			self.blur();
		}
	}

	/// Routes a mouse report through the rail's band. A click inside takes
	/// the keyboard, a click outside returns it; `false` means the gesture
	/// was not consumed and belongs to the transcript.
	pub fn handle_mouse(&mut self, col: u16, row: u16, kind: Mouse, viewport: Size) -> bool {
		if !self.open {
			return false;
		}
		if self
			.ui
			.handle_mouse_as_layer(&self.options, viewport, col, row, kind)
			.is_some()
		{
			if kind == Mouse::Click && !self.focused {
				self.focused = true;
				self.ui.focus_first();
			}
			true
		} else {
			if kind == Mouse::Click {
				self.blur();
			}
			false
		}
	}

	/// Reflects a session model switch in the rail's model row.
	pub fn set_model(&mut self, name: &str) {
		self.ui.set_text("model", name);
	}

	/// The composited rail for this frame, laid out to the full viewport
	/// height; `None` when toggled off or gated out by a small viewport.
	/// Shrinking below the minimum blurs the rail so keys never route into
	/// an invisible layer.
	pub fn layer(&mut self, viewport: Size, elapsed: Duration) -> Option<Layer<'_>> {
		if !self.visible(viewport) {
			if self.focused {
				self.blur();
			}
			return None;
		}
		if self.height != viewport.height {
			self.height = viewport.height;
			self.ui.set_prop("rail", Prop::H, viewport.height);
			self.ui.set_prop("body", Prop::H, viewport.height);
		}
		let seconds = elapsed.as_secs();
		if seconds != self.elapsed_seconds {
			self.elapsed_seconds = seconds;
			self.ui.set_text("elapsed", elapsed_label(seconds));
		}
		Some(Layer { frame: self.ui.frame(), options: &self.options, active: self.focused })
	}

	fn blur(&mut self) {
		self.focused = false;
		self.ui.blur();
	}
}

fn elapsed_label(seconds: u64) -> SmolStr {
	format_smol!("{}:{:02}", seconds / 60, seconds % 60)
}

/// Builds the retained rail tree: session facts, the touched-file list,
/// and the key hints pinned to the bottom of the band.
fn build(model: &str, ctx: &UiContext) -> Ui {
	let files = FILES;
	Ui::from_root(
		dom! {
			<row id="rail" h=24>
				<hr/>
				<col id="body" h=24 grow pad="0 1" gap=1>
					<text bold fg={CYAN}>{"session"}</text>
					<col>
						<row gap=1>
							<text fg={DIM} w=8>{"model"}</text>
							<text id="model" truncate>{model}</text>
						</row>
						<row gap=1>
							<text fg={DIM} w=8>{"elapsed"}</text>
							<text id="elapsed">{"0:00"}</text>
						</row>
						<row gap=1>
							<text fg={DIM} w=8>{"branch"}</text>
							<text truncate>{"tui/seam-commits"}</text>
						</row>
					</col>
					<hr/>
					<text bold fg={CYAN}>{"files"}</text>
					<select id="files" h={files.len() as u16}>
						for (name, delta) in files {
							<option value={name} label={name}>
								<td grow truncate><pre>{name}</pre></td>
								<td align=end><pre fg={GREEN}>{delta}</pre></td>
							</option>
						}
					</select>
					<spacer grow/>
					<text dim truncate>{"ctrl+b rail · esc back"}</text>
				</col>
			</row>
		},
		WIDTH,
		ctx.clone(),
	)
}

#[cfg(test)]
mod tests {
	use std::time::Duration;

	use omp_tui::{Size, UiContext, test_support::frame_row_text};

	use super::Sidebar;

	#[test]
	fn rail_starts_passive_without_focus_chrome_or_caret() {
		let ctx = UiContext::default();
		let viewport = Size::new(120, 30);
		let mut sidebar = Sidebar::new("Claude Fable 5", &ctx);

		let passive: Vec<String> = {
			let layer = sidebar
				.layer(viewport, Duration::ZERO)
				.expect("the rail opens by default");
			assert!(!layer.active, "a passive rail never owns the caret");
			(0..30)
				.map(|row| frame_row_text(layer.frame, row))
				.collect()
		};

		sidebar.toggle(); // hide
		sidebar.toggle(); // show again, taking the keyboard
		let layer = sidebar
			.layer(viewport, Duration::ZERO)
			.expect("the rail reopened");
		assert!(layer.active, "the toggled-open rail owns the keyboard");
		let focused: Vec<String> = (0..30)
			.map(|row| frame_row_text(layer.frame, row))
			.collect();
		assert_ne!(passive, focused, "taking the keyboard adds the focus chrome");
	}
}
