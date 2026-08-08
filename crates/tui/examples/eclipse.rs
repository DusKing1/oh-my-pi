//! Stencil's stippled-eclipse landing shader as a fullscreen effect.
//!
//! ```sh
//! cargo run -p omp-tui --example eclipse
//! ```
//!
//! Mounts [`omp_tui::shader::Eclipse`] — the reference
//! [`omp_tui::shader::Program`] — as a viewport-filling component.
//! Ctrl-C or `q` quits.

use std::io;

use omp_tui::{
	AppEvent, AppOptions, Key, Size, Ui, UiContext, components::Shader, dom, shader::Eclipse,
};

fn build_ui(viewport: Size, context: UiContext) -> Ui {
	let rows = viewport.height.saturating_sub(1).max(1);
	Ui::from_root(
		dom! {
			<col>
				{Shader::new(Eclipse::default()).size(viewport.width, rows)}
				<text dim>{"stencil.so eclipse · q quit"}</text>
			</col>
		},
		viewport.width,
		context,
	)
}

#[tokio::main]
async fn main() -> io::Result<()> {
	let mut context = None;
	let mut app = AppOptions::new()
		.quit([Key::Ctrl('c'), Key::Ctrl('q'), Key::Char('q')])
		.start(|env| {
			context = Some(env.ctx.clone());
			build_ui(env.viewport, env.ctx)
		})
		.await?;
	let context = context.expect("start ran the UI builder");
	while let Some(event) = app.next().await? {
		// The viewport is baked into the shader's size; rebuild on resize.
		if let AppEvent::Resized(viewport) = event {
			*app.ui_mut() = build_ui(viewport, context.clone());
		}
	}
	Ok(())
}
