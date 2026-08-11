//! Animation lab tab: prop-tween scenes with autoplay.
//!
//! Four scenes use ordinary prop writes, and the runtime tweens each change.
//! Runs hands-free while the tab is active; keys retarget individual scenes.

use std::time::Duration;

use omp_tui::{Component, IntoComponent as _, Key, Prop, Ui, components::Spinner, dom};

/// One breath of the autoplay loop: long enough to watch a transition land.
pub const AUTOPLAY_STEP: Duration = Duration::from_millis(1600);

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

/// The animation-lab pane hosted by the gallery's `Anim` tab.
pub fn pane() -> Box<dyn Component> {
	let (_, mood_bg, mood_text) = MOODS[0];
	dom! {
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
				{"space race · m mood · g gradient · s sidebar · d drawer · a autoplay"}
			</text>
		</col>
	}
	.into_component()
}

/// Scene state; every transition is just a prop write on retained ids.
pub struct Lab {
	race_wide:           bool,
	mood:                usize,
	palette:             usize,
	sidebar_wide:        bool,
	drawer_open:         bool,
	pub(crate) autoplay: bool,
	step:                usize,
}

impl Lab {
	pub(crate) const fn new() -> Self {
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

	/// One autoplay step: cycles through the five scene toggles.
	pub(crate) fn advance(&mut self, ui: &mut Ui) {
		match self.step % 5 {
			0 => self.race(ui),
			1 => self.mood(ui),
			2 => self.palette(ui),
			3 => self.sidebar(ui),
			_ => self.drawer(ui),
		}
		self.step += 1;
	}

	/// Routes one unclaimed key while the Anim tab is active.
	pub(crate) fn handle_key(&mut self, key: Key, ui: &mut Ui) {
		match key {
			Key::Space => self.race(ui),
			Key::Char('m') => self.mood(ui),
			Key::Char('g') => self.palette(ui),
			Key::Char('s') => self.sidebar(ui),
			Key::Char('d') => self.drawer(ui),
			Key::Char('a') => self.autoplay = !self.autoplay,
			_ => {},
		}
	}
}
