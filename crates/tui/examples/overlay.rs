//! Overlay demo: a transcript with a model-switcher modal and a help layer.
//!
//! `Ctrl+K` opens the switcher, `Ctrl+G` toggles help, `Esc` closes the top
//! layer, `Ctrl+C` exits. The document keeps native scrollback while layers
//! come and go.

use std::{io, time::Duration};

use omp_tui::{
	Dim, InputEvent, Key, OverlayAnchor, OverlayId, OverlayMargin, OverlayOptions, Renderer, Size,
	Terminal, TerminalCaps, TerminalEvent, TerminalOptions, TtyOut, Ui, UiContext, UiEvent, dom,
};

const MODELS: [(&str, &str, &str); 4] = [
	("fable", "anthropic/claude-fable-5", "4.5s · 64t/s · $10/50"),
	("flash", "google/gemini-3.6-flash", "2.6s · 342t/s · $1.5/7.5"),
	("sol", "openai/gpt-5.6-sol", "1.7s · 41t/s · $5/30"),
	("opus", "anthropic/claude-opus-5", "6.1s · 44t/s · $5/25"),
];

fn build_base(width: u16, ctx: UiContext) -> Ui {
	Ui::from_root(
		dom! {
			<col gap=1 pad="1 2">
				<md>{"The **overlay demo** transcript. Document content stays on the normal screen and keeps native scrollback while layers composite above it."}</md>
				<md>{"Press `Ctrl+K` to switch models, `Ctrl+G` for help, `Esc` to close the top layer."}</md>
				<text id=status fg=muted>{"model: anthropic/claude-fable-5"}</text>
				<box border=round title="Composer">
					<input id=composer placeholder="Type while overlays come and go"/>
				</box>
			</col>
		},
		width,
		ctx,
	)
}

fn show_picker(ui: &mut Ui) -> OverlayId {
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

fn show_help(ui: &mut Ui) -> OverlayId {
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

#[tokio::main]
async fn main() -> io::Result<()> {
	let mut terminal = Terminal::enter(TerminalOptions::default().mouse(true))?;
	let caps = terminal.caps();
	let mut renderer = Renderer::new(TtyOut::new()?);
	renderer.set_sync_output(caps.sync_output);
	renderer.set_hyperlinks(caps.hyperlinks);
	renderer.set_tmux_passthrough(caps.inside_tmux);
	run(&mut terminal, &mut renderer, caps).await
}

#[expect(
	clippy::future_not_send,
	reason = "the retained UI is deliberately confined to its terminal event-loop thread"
)]
async fn run(
	terminal: &mut Terminal,
	renderer: &mut Renderer<TtyOut>,
	caps: TerminalCaps,
) -> io::Result<()> {
	let mut viewport = terminal.size()?;
	let ctx = UiContext::default().with_terminal_caps(&caps);
	let mut ui = build_base(viewport.width, ctx);
	let mut picker: Option<OverlayId> = None;
	let mut help: Option<OverlayId> = None;
	renderer.rebuild(ui.frame().clone(), viewport.height, 0, "")?;

	loop {
		tokio::select! {
			event = terminal.next() => match event? {
				TerminalEvent::Input(event) => {
					match event {
						InputEvent::Key(Key::Ctrl('c')) => return Ok(()),
						InputEvent::Key(Key::Ctrl('k')) if picker.is_none() => {
							picker = Some(show_picker(&mut ui));
						},
						InputEvent::Key(Key::Ctrl('g')) => match help.take() {
							Some(id) => {
								ui.close_overlay(id);
							},
							None => help = Some(show_help(&mut ui)),
						},
						// The select consumes Enter to choose; Enter then confirms
						// the picker at the application level.
						InputEvent::Key(Key::Enter) if picker.is_some() => {
							ui.handle_key(Key::Enter);
							let id = picker.take().expect("guarded by picker.is_some()");
							let choice = ui.overlay(id).map(Ui::values).and_then(|values| {
								values
									.get("model")
									.and_then(|value| value.as_str().map(str::to_owned))
							});
							if let Some(choice) = choice {
								let label = MODELS
									.iter()
									.find(|(value, ..)| *value == choice)
									.map_or(choice.as_str(), |(_, label, _)| label);
								ui.set_text("status", format!("model: {label}"));
							}
							ui.close_overlay(id);
						},
						InputEvent::Key(key) => {
							if ui.handle_key(key) == UiEvent::Cancel {
								match ui.close_active_overlay() {
									Some(id) => {
										if picker == Some(id) {
											picker = None;
										}
										if help == Some(id) {
											help = None;
										}
									},
									None => return Ok(()),
								}
							}
						},
						InputEvent::Mouse(mouse) => {
							ui.handle_mouse(mouse.col, mouse.row, mouse.kind);
						},
						InputEvent::Paste(text) => {
							ui.handle_paste(&text);
						},
						InputEvent::Focus(_) | InputEvent::Response(_) => {},
					}
					terminal.sync_renderer(renderer)?;
				},
				TerminalEvent::Resize => {
					if let Some(size) = terminal.take_resize()? {
						viewport = size;
						ui.resize(size.width);
						renderer.rebuild(ui.frame().clone(), viewport.height, 0, "")?;
					}
				},
				TerminalEvent::Debug(_) => {},
				TerminalEvent::Closed => return Ok(()),
			},
			() = tokio::time::sleep(Duration::from_millis(250)) => {},
		}
		if let Some(size) = terminal.take_resize()? {
			viewport = size;
			ui.resize(size.width);
			renderer.rebuild(ui.frame().clone(), viewport.height, 0, "")?;
		}
		if ui.has_damage() {
			ui.present(renderer, viewport.height, 0)?;
		}
	}
}
