//! Overlay tab: a transcript with a model-switcher modal and a help layer.
//!
//! `Ctrl+K` opens the switcher, `Ctrl+G` toggles help, `Esc` closes the top
//! layer. The document keeps native scrollback while layers come and go.

use omp_tui::{
	Component, Dim, IntoComponent as _, OverlayAnchor, OverlayId, OverlayMargin, OverlayOptions,
	Size, Ui, dom,
};

pub(crate) const MODELS: [(&str, &str, &str); 4] = [
	("fable", "anthropic/claude-fable-5", "4.5s · 64t/s · $10/50"),
	("flash", "google/gemini-3.6-flash", "2.6s · 342t/s · $1.5/7.5"),
	("sol", "openai/gpt-5.6-sol", "1.7s · 41t/s · $5/30"),
	("opus", "anthropic/claude-opus-5", "6.1s · 44t/s · $5/25"),
];

/// The overlay-demo pane hosted by the gallery's `Overlay` tab.
pub(crate) fn pane() -> Box<dyn Component> {
	dom! {
		<col gap=1 pad="1 2">
			<md>{"The **overlay demo** transcript. Document content stays on the normal screen and keeps native scrollback while layers composite above it."}</md>
			<md>{"Press `Ctrl+K` to switch models, `Ctrl+G` for help, `Esc` to close the top layer."}</md>
			<text id=status fg=muted>{"model: anthropic/claude-fable-5"}</text>
			<box border=round title="Composer">
				<input id=composer placeholder="Type while overlays come and go"/>
			</box>
		</col>
	}
	.into_component()
}

/// Opens the model-switcher modal; commit surfaces as `Changed { id: "model"
/// }`.
pub(crate) fn show_picker(ui: &mut Ui) -> OverlayId {
	ui.show_overlay(
		dom! {
			<box border=round title="Switch Model">
				<col gap=1>
					<text dim>{"Session-only switch — role models stay unchanged"}</text>
					<select id=model>
						for (value, label, stats) in MODELS {
							<option value={value} desc={stats}>{label}</option>
						}
					</select>
				</col>
			</box>
		},
		OverlayOptions::default()
			.anchor(OverlayAnchor::Center)
			.width(Dim::Pct(70))
			.min_width(48)
			.max_height(Dim::Pct(60))
			.min_viewport(Size::new(40, 8)),
	)
}

/// Opens the keybinding help layer.
pub(crate) fn show_help(ui: &mut Ui) -> OverlayId {
	ui.show_overlay(
		dom! {
			<box border=round title="Help">
				<col>
					<text>{"Ctrl+K  switch model"}</text>
					<text>{"Ctrl+G  toggle this help"}</text>
					<text>{"Esc     close top layer"}</text>
					<text>{"Ctrl+C  quit"}</text>
				</col>
			</box>
		},
		OverlayOptions::default()
			.anchor(OverlayAnchor::BottomRight)
			.width(Dim::Cells(30))
			.margin(OverlayMargin::uniform(1)),
	)
}
