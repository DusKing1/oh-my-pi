//! Eclipse tab: Stencil's stippled-eclipse landing shader.
//!
//! Mounts [`omp_tui::shader::Eclipse`] — the reference
//! [`omp_tui::shader::Program`] — as a pane-filling component. The shader
//! viewport is sized at build time; the pane clips it on shrink.

use omp_tui::{Component, IntoComponent as _, Size, components::Shader, dom, shader::Eclipse};

/// The eclipse-shader pane hosted by the gallery's `Eclipse` tab, sized to
/// `viewport` minus the tab chrome.
pub(crate) fn pane(viewport: Size, rows: u16) -> Box<dyn Component> {
	dom! {
		<col>
			{Shader::new(Eclipse::default()).size(viewport.width, rows)}
			<text dim>{"stencil.so eclipse"}</text>
		</col>
	}
	.into_component()
}
