//! Animated UI component gallery.
//!
//! Four scenes use ordinary prop writes, and the runtime tweens each change.
//! Runs hands-free; keys retarget individual scenes.

use std::{
	future::Future,
	io,
	time::{Duration, Instant},
};

use omp_tui::{
	InputEvent, Key, Prop, Renderer, Terminal, TerminalCaps, TerminalEvent, TerminalOptions, TtyOut,
	Ui, UiContext, UiEvent, components::Spinner, dom,
};

/// One breath of the autoplay loop: long enough to watch a transition land.
const AUTOPLAY_STEP: Duration = Duration::from_millis(1600);
/// Poll ceiling while fully quiescent, so autoplay still advances.
const IDLE_POLL: Duration = Duration::from_millis(250);

const BARS: &[(&str, &str)] =
	&[("bar-linear", "linear"), ("bar-in", "in"), ("bar-out", "out"), ("bar-in-out", "in-out")];

/// `(border/text token, panel background, status line)` per mood.
const MOODS: &[(&str, &str, &str)] = &[
	("ok", "#10231a", "all systems nominal"),
	("warn", "#2b2312", "latency rising on shard 7"),
	("err", "#2b1414", "shard 7 dropped out"),
	("info", "#121f2e", "rebalancing replicas…"),
];

const PALETTES: &[&str] =
	&["#0f0c29..#f5af19", "#12c2e9..#f64f59", "#134e5e..#71b280", "#41295a..#f4e2d8"];

fn build_ui(width: u16, caps: &TerminalCaps) -> Ui {
	let (_, mood_bg, mood_text) = MOODS[0];
	let root = dom! {
		<col gap=1 pad="1 2">
			<row gap=2>
				<text bold fg="#f953c6..#43e97b" spin="4s">{"ANIMATION LAB"}</text>
				{Spinner::new()}
				<text dim>{"anim · ease · spin as plain props"}</text>
			</row>
			<box title="Easing race" border=round bc=muted pad="0 1">
				<col>
					for (id, ease) in BARS {
						<row gap=1>
							<text w=7 dim>{*ease}</text>
							<col id={*id} w=12% h=1 bg=accent anim="900ms" ease={*ease}/>
						</row>
					}
				</col>
			</box>
			<row gap=1>
				<box id=mood grow title="Mood" border=round bleed anim="450ms" bc=ok bg={mood_bg}>
					<text id="mood-text" fg=ok anim="450ms">{mood_text}</text>
				</box>
				<box id=hero grow title="Gradient morph" border=round bleed anim="900ms"
					ease="in-out" bg={PALETTES[0]} angle=25 spin="6s">
					<text bold>{"endpoints tween"}</text>
					<text dim>{"angle spins forever"}</text>
				</box>
			</row>
			<row gap=1>
				<col id=sidebar w=14 anim="500ms" ease="in-out" bg="#1b2735" pad="0 1">
					<text bold>{"sidebar"}</text>
					<text dim>{"w tweens"}</text>
				</col>
				<box id=drawer grow h=4 anim="600ms" ease="in-out" border=round bc=muted
					title="Drawer">
					<md>{"Height tweens through **layout wakes**: every row below shifts \
							smoothly.\n\n- retarget mid-flight: it resumes from the screen\n- \
							first paint never animates\n- settled components request no frames"}</md>
				</box>
			</row>
			<text dim>
				{"space race · m mood · g gradient · s sidebar · d drawer · a autoplay · q quit"}
			</text>
		</col>
	};
	Ui::from_root(root, width, UiContext::default().with_terminal_caps(caps))
}

/// Scene state; every transition is just a prop write on retained ids.
struct Lab {
	race_wide:    bool,
	mood:         usize,
	palette:      usize,
	sidebar_wide: bool,
	drawer_open:  bool,
	autoplay:     bool,
	step:         usize,
}

impl Lab {
	const fn new() -> Self {
		Self {
			race_wide:    false,
			mood:         0,
			palette:      0,
			sidebar_wide: false,
			drawer_open:  false,
			autoplay:     true,
			step:         0,
		}
	}

	fn race(&mut self, ui: &mut Ui) {
		self.race_wide = !self.race_wide;
		let target = if self.race_wide { "88%" } else { "12%" };
		for (id, _) in BARS {
			ui.set_prop(id, Prop::W, target);
		}
	}

	fn mood(&mut self, ui: &mut Ui) {
		self.mood = (self.mood + 1) % MOODS.len();
		let (token, bg, text) = MOODS[self.mood];
		ui.set_prop("mood", Prop::Bc, token);
		ui.set_prop("mood", Prop::Bg, bg);
		ui.set_prop("mood-text", Prop::Fg, token);
		ui.set_text("mood-text", text);
	}

	fn palette(&mut self, ui: &mut Ui) {
		self.palette = (self.palette + 1) % PALETTES.len();
		ui.set_prop("hero", Prop::Bg, PALETTES[self.palette]);
	}

	fn sidebar(&mut self, ui: &mut Ui) {
		self.sidebar_wide = !self.sidebar_wide;
		ui.set_prop("sidebar", Prop::W, if self.sidebar_wide { 30_u16 } else { 14 });
	}

	fn drawer(&mut self, ui: &mut Ui) {
		self.drawer_open = !self.drawer_open;
		ui.set_height("drawer", if self.drawer_open { 10 } else { 4 });
	}

	fn advance(&mut self, ui: &mut Ui) {
		match self.step % 5 {
			0 => self.race(ui),
			1 => self.mood(ui),
			2 => self.palette(ui),
			3 => self.sidebar(ui),
			_ => self.drawer(ui),
		}
		self.step += 1;
	}
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> io::Result<()> {
	let mut terminal = Terminal::enter(TerminalOptions::default().mouse(true))?;
	let caps = terminal.caps();
	let mut renderer = Renderer::new(TtyOut::new()?);
	renderer.set_sync_output(caps.sync_output);
	run(&mut terminal, &mut renderer, caps).await
}

async fn run<'a>(
	terminal: &'a mut Terminal,
	renderer: &'a mut Renderer<TtyOut>,
	caps: TerminalCaps,
) -> io::Result<()> {
	let mut viewport = terminal.size()?;
	let mut ui = build_ui(viewport.width, &caps);
	let mut lab = Lab::new();
	let started = Instant::now();
	let mut next_step = started + AUTOPLAY_STEP;

	renderer.rebuild(ui.frame().clone(), viewport.height, 0, "")?;

	loop {
		// Sleep until the earliest animation deadline or the autoplay step;
		// a quiescent scene (autoplay off, everything settled) idles.
		let now = started.elapsed();
		let wake = ui.next_wake().map(|at| at.saturating_sub(now));
		let step = lab
			.autoplay
			.then(|| next_step.saturating_duration_since(Instant::now()));
		let timeout = wake.unwrap_or(IDLE_POLL).min(step.unwrap_or(IDLE_POLL));

		let deadline = tokio::time::Instant::now() + timeout;
		tokio::select! {
			event = terminal.next() => match event? {
				TerminalEvent::Input(event) => {
					match event {
						InputEvent::Key(key) => {
							if handle_key(key, &mut lab, &mut ui) {
								return Ok(());
							}
						},
						InputEvent::Mouse(mouse) => {
							ui.handle_mouse(mouse.col, mouse.row, mouse.kind);
						},
						InputEvent::Paste(_) | InputEvent::Focus(_) | InputEvent::Response(_) => {},
					}
					terminal.sync_renderer(renderer)?;
				},
				TerminalEvent::Resize => {
					if let Some(size) = terminal.take_resize()? {
						viewport = size;
						ui.resize(viewport.width);
						renderer.rebuild(ui.frame().clone(), viewport.height, 0, "")?;
					}
					continue;
				},
				TerminalEvent::Debug(_) => {},
				TerminalEvent::Closed => return Ok(()),
			},
			() = tokio::time::sleep_until(deadline) => {},
		}
		if let Some(size) = terminal.take_resize()? {
			viewport = size;
			ui.resize(viewport.width);
			renderer.rebuild(ui.frame().clone(), viewport.height, 0, "")?;
			continue;
		}

		if lab.autoplay && Instant::now() >= next_step {
			lab.advance(&mut ui);
			next_step += AUTOPLAY_STEP;
		}
		ui.tick(started.elapsed());
		ui.present(renderer, viewport.height, 0)?;
	}
}

/// Routes one key; returns `true` to quit.
fn handle_key(key: Key, lab: &mut Lab, ui: &mut Ui) -> bool {
	match key {
		Key::Char('q') | Key::Esc | Key::Ctrl('c') => return true,
		Key::Space => lab.race(ui),
		Key::Char('m') => lab.mood(ui),
		Key::Char('g') => lab.palette(ui),
		Key::Char('s') => lab.sidebar(ui),
		Key::Char('d') => lab.drawer(ui),
		Key::Char('a') => lab.autoplay = !lab.autoplay,
		_ => {
			if ui.handle_key(key) == UiEvent::Cancel {
				return true;
			}
		},
	}
	false
}
