//! The host contract: a scene produces frames and routes input.

use std::time::Duration;

use omp_core::Str;
use omp_tui::{Frame, Key, Layer, MouseReport, Size, paste::ClipboardRead};
use smallvec::SmallVec;

/// What one paint of a scene looks like to the host: the retained document
/// frame (taller than the viewport when scrollback exists) plus declarative
/// viewport-anchored layers, exactly as a terminal renderer would composite.
pub struct SceneFrame<'a> {
	/// The document grid; the tail `viewport.height` rows are live.
	pub frame:       &'a Frame,
	/// Viewport dimensions in cells.
	pub viewport:    Size,
	/// Document-tail rows owned by an editing widget (the composer); the
	/// host routes plain pointer gestures there to the scene instead of
	/// starting host text selection. A fully interactive scene (welcome
	/// card) claims every row.
	pub editor_rows: u16,
	/// Z-ordered layers resolved against the viewport at paint time.
	pub layers:      SmallVec<Layer<'a>, 4>,
}

/// Post-event effect a scene requests from the host.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Effect {
	/// Not handled; the host may apply its own fallback.
	Ignored,
	/// Handled; repaint at the next tick.
	Consumed,
	/// The application asked to exit.
	Quit,
	/// Read the system clipboard (images preferred, or text only) and
	/// deliver the result to [`Scene::paste`].
	Clipboard(ClipboardRead),
	/// Write text to the system clipboard; the host performs the write
	/// detached so a slow CLI fallback never blocks the event loop.
	SetClipboard(Str),
}

/// One host-driven application: retained state, a render pass, input routing.
pub trait Scene {
	/// Updates the cell viewport. `settled` distinguishes the final geometry
	/// of a resize gesture from its intermediate drag steps, letting the
	/// scene trade full relayout for cheap previews mid-gesture.
	fn resize(&mut self, viewport: Size, settled: bool);

	/// Produces the current frame and layers.
	fn render(&mut self) -> SceneFrame<'_>;

	/// Routes one key press.
	fn key(&mut self, key: Key) -> Effect;

	/// Routes one mouse report in viewport cell coordinates.
	fn mouse(&mut self, report: MouseReport) -> Effect;

	/// Routes clipboard text; `raw` inserts verbatim without attachment
	/// staging or drop classification.
	fn paste(&mut self, text: &str, raw: bool) -> Effect;

	/// Repaint cadence while the scene animates.
	fn tick(&self) -> Duration;
}
