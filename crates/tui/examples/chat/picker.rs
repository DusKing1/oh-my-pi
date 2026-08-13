//! Session model picker: pi's `Switch Model` overlay (`model-picker.ts` /
//! `model-browser.ts`) rebuilt on core primitives.
//!
//! `Ctrl+P` opens it over the chat transcript. The list is a `<select
//! filter>` whose options carry `<td>` cells, so the core widget owns the
//! query editor (always-on typing, paste, `Ctrl+U`/`Ctrl+W`, the Esc cancel
//! ladder, the hardware caret), fuzzy ranking, cursor movement (`↑`/`↓`
//! wrap, `PageUp`/`PageDown`/`Home`/`End` clamp), windowed scrolling,
//! hover, wheel, and click activation — the picker only routes the
//! surfaced [`UiEvent`]s. Columns align across rows through the shared
//! table solver; the name cell start-truncates (`truncate=start`) so the
//! distinctive id tail survives narrow widths, and every glyph — the
//! context-window icon, dots, cursor — resolves through the host's
//! detected [`UiContext`] charset. A leading `@` swaps the catalog to
//! quick roles; `Enter` (or a click) applies the selected model for this
//! session only.

use std::{path::Path, sync::LazyLock};

use omp_core::{Str, StrMut, fmts};
use omp_tui::{
	Charset, Color, Component, Dim, Icon, IntoComponent as _, Key, Layer, Mouse, OverlayAnchor,
	OverlayOptions, Prop, Size, Ui, UiContext, UiEvent, dom,
};

const GREEN: Color = Color::Rgb(81, 196, 112);
const CYAN: Color = Color::Rgb(62, 190, 203);
const PURPLE: Color = Color::Rgb(171, 119, 230);
const GOLD: Color = Color::Rgb(210, 167, 86);
const TEXT: Color = Color::Rgb(194, 198, 204);
const DIM: Color = Color::Rgb(110, 116, 124);

/// One catalog entry with the display stats pi renders per row.
pub struct ModelSpec {
	pub provider:  &'static str,
	pub id:        &'static str,
	pub name:      &'static str,
	/// Time to first token, preformatted (`"4.5s"`).
	pub ttft:      &'static str,
	/// Throughput, preformatted (`"64t/s"`).
	pub tps:       &'static str,
	/// Context window (`"1m"`, `"272k"`).
	pub ctx:       &'static str,
	/// Output limit (`"128k"`).
	pub out:       &'static str,
	/// `$input/output` per M tokens, or `"free"`.
	pub cost:      &'static str,
	pub reasoning: bool,
	pub vision:    bool,
}

/// The demo catalog, mirroring pi's switcher screenshot.
pub const MODELS: [ModelSpec; 6] = [
	ModelSpec {
		provider:  "anthropic",
		id:        "claude-fable-5",
		name:      "Claude Fable 5",
		ttft:      "4.5s",
		tps:       "64t/s",
		ctx:       "1m",
		out:       "128k",
		cost:      "$10/50",
		reasoning: true,
		vision:    true,
	},
	ModelSpec {
		provider:  "google-antigravity",
		id:        "gemini-3.6-flash",
		name:      "Gemini 3.6 Flash",
		ttft:      "2.6s",
		tps:       "342t/s",
		ctx:       "1m",
		out:       "65k",
		cost:      "$1.5/7.5",
		reasoning: true,
		vision:    true,
	},
	ModelSpec {
		provider:  "openai-codex",
		id:        "gpt-5.6-sol",
		name:      "GPT 5.6 Sol",
		ttft:      "1.7s",
		tps:       "41t/s",
		ctx:       "272k",
		out:       "128k",
		cost:      "$5/30",
		reasoning: true,
		vision:    false,
	},
	ModelSpec {
		provider:  "google-antigravity",
		id:        "gemini-3.1-pro",
		name:      "Gemini 3.1 Pro",
		ttft:      "3.3s",
		tps:       "89t/s",
		ctx:       "1m",
		out:       "65k",
		cost:      "$2/12",
		reasoning: true,
		vision:    true,
	},
	ModelSpec {
		provider:  "ollama",
		id:        "lfm2:2.6b",
		name:      "LFM2 2.6B",
		ttft:      "4.8s",
		tps:       "0.4t/s",
		ctx:       "128k",
		out:       "32k",
		cost:      "free",
		reasoning: false,
		vision:    false,
	},
	ModelSpec {
		provider:  "anthropic",
		id:        "claude-opus-5",
		name:      "Claude Opus 5",
		ttft:      "6.1s",
		tps:       "44t/s",
		ctx:       "1m",
		out:       "64k",
		cost:      "$5/25",
		reasoning: true,
		vision:    true,
	},
];

/// A configured or auto-selected role, surfaced as `@role` quick items and as
/// chips under the detail block (pi's `ResolvedRoleModel`).
struct Role {
	name:       &'static str,
	model:      usize,
	color:      Color,
	/// Thinking-level glyph suffix shown after the chip and quick item.
	thinking:   Option<&'static str>,
	/// Configured roles render a solid `●` chip; auto-selected ones `○`.
	configured: bool,
}

const ROLES: [Role; 5] = [
	Role {
		name:       "default",
		model:      0,
		color:      GREEN,
		thinking:   None,
		configured: true,
	},
	Role {
		name:       "smol",
		model:      1,
		color:      CYAN,
		thinking:   Some("◔"),
		configured: true,
	},
	Role {
		name:       "slow",
		model:      5,
		color:      PURPLE,
		thinking:   Some("◕"),
		configured: true,
	},
	Role {
		name:       "plan",
		model:      0,
		color:      GOLD,
		thinking:   Some("◑"),
		configured: false,
	},
	Role {
		name:       "fable",
		model:      0,
		color:      PURPLE,
		thinking:   Some("◑"),
		configured: false,
	},
];

const STATUS_MODELS: &str = "Session-only switch — role models stay unchanged";
const STATUS_ROLES: &str = "Quick role switch — applies its model and thinking for this session";
const HINT_MODELS: &str =
	"↑/↓ models · Enter use for this session · type to search · @ quick roles · Esc close";
const HINT_ROLES: &str = "↑/↓ roles · Enter apply role model · type to search · Esc close";

/// Rows the picker occupies beyond the list: box borders, status, the
/// select's query row, a blank, facts, chips, and the hint bar.
const FRAME_ROWS: u16 = 8;

/// Vendored logo directory; resolved at runtime because this module is
/// compiled into several crates (`CARGO_MANIFEST_DIR` differs per host).
static LOGO_DIR: LazyLock<String> = LazyLock::new(|| {
	let local = Path::new(file!())
		.parent()
		.map(|dir| dir.join("../assets/login"));
	match local.filter(|dir| dir.exists()) {
		Some(dir) => dir.to_string_lossy().into_owned(),
		None => concat!(env!("CARGO_MANIFEST_DIR"), "/examples/assets/login").to_string(),
	}
});

/// What a routed input event did to the picker.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PickerEvent {
	/// The picker consumed the event (repaint and continue).
	Consumed,
	/// Close the overlay without switching.
	Close,
	/// Apply this catalog model for the session and close.
	Pick(usize),
}

/// Which catalog the select currently lists.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
	Models,
	Roles,
}

/// pi's responsive perf column: full `ttft tps` at 96 cells, `tps` at 76,
/// gone below (model-browser.ts perf tiers).
#[derive(Clone, Copy, PartialEq, Eq)]
enum PerfTier {
	Full,
	Tps,
	None,
}

impl PerfTier {
	const fn of(width: u16) -> Self {
		if width >= 96 {
			Self::Full
		} else if width >= 76 {
			Self::Tps
		} else {
			Self::None
		}
	}

	fn cell(self, model: &ModelSpec) -> Str {
		match self {
			Self::Full => fmts!("{} {}", model.ttft, model.tps),
			Self::Tps => Str::new_static(model.tps),
			Self::None => Str::new_static(""),
		}
	}
}

/// Retained picker overlay: one `Ui` per catalog mode, rebuilt only when
/// the mode or the perf tier changes; everything else is core select state.
pub struct ModelPicker {
	ui:      Ui,
	mode:    Mode,
	tier:    PerfTier,
	current: usize,
	ctx:     UiContext,
	options: OverlayOptions,
	/// Query carried across catalog swaps (the `@` mode toggle).
	query:   Str,
	/// List rows granted by the last viewport.
	rows:    u16,
}

impl ModelPicker {
	/// Opens the picker with the session's current model preselected,
	/// presenting through the host's detected context (charset, graphics,
	/// theme).
	pub fn open(current: usize, ctx: &UiContext) -> Self {
		let options = OverlayOptions::default()
			.anchor(OverlayAnchor::Bottom)
			.width(Dim::Pct(100))
			.z(10);
		let mode = Mode::Models;
		let tier = PerfTier::Full;
		let ui = build(mode, tier, current, "", 5, 100, ctx);
		let mut picker = Self {
			ui,
			mode,
			tier,
			current,
			ctx: ctx.clone(),
			options,
			query: Str::default(),
			rows: 5,
		};
		picker.show_detail(Some(current));
		picker
	}

	/// Routes a key through the retained tree and maps the surfaced event.
	pub fn handle_key(&mut self, key: Key) -> PickerEvent {
		let event = self.ui.handle_key(key);
		self.route(event)
	}

	/// Routes pasted text into the select's query editor.
	pub fn handle_paste(&mut self, text: &str) -> PickerEvent {
		let event = self.ui.handle_paste(text);
		self.route(event)
	}

	/// Routes a mouse report through the compositor's own band: hover,
	/// wheel, and clicks land in the select; a click outside the layer
	/// closes the picker (pi's dismiss gesture).
	pub fn handle_mouse(&mut self, col: u16, row: u16, kind: Mouse, viewport: Size) -> PickerEvent {
		match self
			.ui
			.handle_mouse_as_layer(&self.options, viewport, col, row, kind)
		{
			Some(event) => self.route(event),
			None if kind == Mouse::Click => PickerEvent::Close,
			None => PickerEvent::Consumed,
		}
	}

	/// The composited layer for this frame: bottom-anchored, full width,
	/// at most 40% of the viewport tall.
	pub fn layer(&mut self, viewport: Size) -> Layer<'_> {
		let rows = (viewport.height * 2 / 5).saturating_sub(FRAME_ROWS).max(5);
		let tier = PerfTier::of(viewport.width);
		if tier != self.tier {
			self.tier = tier;
			self.rebuild();
		}
		if rows != self.rows {
			self.rows = rows;
			// One query row plus the windowed list.
			self.ui.set_prop("models", Prop::H, rows.saturating_add(1));
		}
		if self.ui.frame().size().width != viewport.width {
			self.ui.resize(viewport.width);
		}
		Layer { frame: self.ui.frame(), options: &self.options, active: true }
	}

	/// Applies one surfaced [`UiEvent`] to picker state.
	fn route(&mut self, event: UiEvent) -> PickerEvent {
		match event {
			UiEvent::Cancel => PickerEvent::Close,
			UiEvent::Changed { value, .. } => value
				.as_str()
				.parse()
				.map_or(PickerEvent::Consumed, PickerEvent::Pick),
			UiEvent::Highlighted { value, .. } => {
				self.show_detail(value.as_str().parse().ok());
				PickerEvent::Consumed
			},
			UiEvent::Filtered { query, value, .. } => {
				let wants_roles = query.starts_with('@');
				self.query = query;
				if wants_roles == (self.mode == Mode::Roles) {
					self.show_detail(value.and_then(|value| value.as_str().parse().ok()));
				} else {
					self.mode = if wants_roles {
						Mode::Roles
					} else {
						Mode::Models
					};
					self.rebuild();
				}
				PickerEvent::Consumed
			},
			UiEvent::None | UiEvent::Submit | UiEvent::Pressed(_) | UiEvent::Copied(_) => {
				PickerEvent::Consumed
			},
		}
	}

	/// Rebuilds the retained tree for the current mode/tier, reseeding the
	/// select's query so typing continuity survives the swap.
	fn rebuild(&mut self) {
		let width = self.ui.frame().size().width;
		self.ui = build(self.mode, self.tier, self.current, &self.query, self.rows, width, &self.ctx);
		let initial = match self.mode {
			Mode::Models => Some(self.current),
			// The role list keeps its configured order; detail follows the
			// first row until the next cursor event.
			Mode::Roles => ROLES.first().map(|role| role.model),
		};
		self.show_detail(initial);
	}

	/// Points the facts line and role chips at `model`; `None` (no match)
	/// blanks the details while keeping the frame height stable.
	fn show_detail(&mut self, model: Option<usize>) {
		show_detail_on(&mut self.ui, model);
	}
}

/// Points the facts line and role chips of any Ui hosting the picker tree
/// at `model`; `None` blanks the details while keeping the height stable.
pub fn show_detail_on(ui: &mut Ui, model: Option<usize>) {
	let facts = model.map_or_else(|| Str::new_static(" "), |index| facts(&MODELS[index]));
	ui.set_text("facts", facts);
	// Hide before show: the document must never transiently exceed its
	// steady height — a raw-frame layer's retained frame keeps the
	// high-water mark, which would leave a stale extra row.
	for index in (0..MODELS.len()).filter(|&index| model != Some(index)) {
		ui.set_visible(&fmts!("chips-{index}"), false);
	}
	if let Some(index) = model {
		ui.set_visible(&fmts!("chips-{index}"), true);
	}
}

/// The facts line under the list (name, limits, price, capabilities).
fn facts(model: &ModelSpec) -> Str {
	fn part(facts: &mut StrMut, text: &str) {
		if !facts.is_empty() {
			facts.push_str(" · ");
		}
		facts.push_str(text);
	}
	let mut line = StrMut::with_capacity(96);
	part(&mut line, model.name);
	part(&mut line, &fmts!("{} ctx", model.ctx));
	part(&mut line, &fmts!("{} out", model.out));
	part(&mut line, &fmts!("{} per M", model.cost));
	if model.reasoning {
		part(&mut line, "reasoning");
	}
	if model.vision {
		part(&mut line, "vision");
	}
	part(&mut line, &fmts!("~{}", model.tps));
	part(&mut line, &fmts!("{} ttft", model.ttft));
	line.freeze()
}

/// One option row's static content, shared by both catalogs.
struct RowSpec {
	value:       Str,
	label:       Str,
	logo:        Str,
	prefix:      Str,
	prefix_fg:   Color,
	name:        Str,
	name_fg:     Color,
	current:     bool,
	recommended: bool,
	perf:        Str,
	ctx:         Str,
	cost:        Str,
}

fn model_rows(tier: PerfTier, current: usize, charset: Charset) -> Vec<RowSpec> {
	MODELS
		.iter()
		.enumerate()
		.map(|(index, model)| RowSpec {
			value:       fmts!("{index}"),
			label:       fmts!("{}/{}", model.provider, model.id),
			logo:        fmts!("{}/{}.png", &*LOGO_DIR, model.provider),
			prefix:      fmts!("{}/", model.provider),
			prefix_fg:   DIM,
			name:        Str::new_static(model.id),
			name_fg:     TEXT,
			current:     index == current,
			recommended: index == current,
			perf:        tier.cell(model),
			ctx:         fmts!("{} {}", model.ctx, charset.icon(Icon::Context)),
			cost:        Str::new_static(model.cost),
		})
		.collect()
}

fn role_rows(tier: PerfTier, current: usize, charset: Charset) -> Vec<RowSpec> {
	ROLES
		.iter()
		.enumerate()
		.map(|(index, role)| {
			let model = &MODELS[role.model];
			let name = match role.thinking {
				Some(glyph) => fmts!("{} {glyph}", role.name),
				None => Str::new_static(role.name),
			};
			RowSpec {
				value: fmts!("{}", role.model),
				// The `@` stays in the haystack so `@arc` matches and the
				// bare `@` keeps every role visible.
				label: fmts!("@{}", role.name),
				logo: fmts!("{}/{}.png", &*LOGO_DIR, model.provider),
				prefix: Str::default(),
				prefix_fg: DIM,
				name,
				name_fg: role.color,
				current: role.model == current,
				recommended: index == 0,
				perf: tier.cell(model),
				ctx: fmts!("{} {}", model.ctx, charset.icon(Icon::Context)),
				cost: Str::new_static(model.cost),
			}
		})
		.collect()
}

/// One role chip (`● default`, `○ plan ◑`) under the facts line.
struct Chip {
	text:  Str,
	color: Color,
}

/// The chip row for `model`: its current marker plus every role resolving
/// to it. Dots resolve through the charset — solid (`enabled`) for
/// configured roles, hollow (`shadowed`) for auto-selected ones.
fn chips(model: usize, current: usize, charset: Charset) -> Vec<Chip> {
	let mut chips = Vec::new();
	if model == current {
		chips.push(Chip { text: fmts!("{} current", charset.icon(Icon::Enabled)), color: GREEN });
	}
	for role in ROLES.iter().filter(|role| role.model == model) {
		let dot = if role.configured {
			charset.icon(Icon::Enabled)
		} else {
			charset.icon(Icon::Shadowed)
		};
		let color = if role.configured { role.color } else { DIM };
		let mut text = StrMut::with_capacity(16);
		text.push_str(dot);
		text.push(' ');
		text.push_str(role.name);
		if let Some(glyph) = role.thinking {
			text.push(' ');
			text.push_str(glyph);
		}
		chips.push(Chip { text: text.freeze(), color });
	}
	if chips.is_empty() {
		chips.push(Chip { text: Str::new_static(" "), color: DIM });
	}
	chips
}

/// The models-catalog picker pane at full perf tier for `width`: the tree
/// behind [`ModelPicker`], reusable as inline content by other examples.
#[allow(dead_code, reason = "consumed by the gallery example's #[path] include of this module")]
pub fn models_pane(current: usize, rows: u16, width: u16, charset: Charset) -> Box<dyn Component> {
	tree(Mode::Models, PerfTier::of(width), current, "", rows, charset)
}

/// Builds the retained overlay tree for one catalog mode.
fn build(
	mode: Mode,
	tier: PerfTier,
	current: usize,
	query: &str,
	rows: u16,
	width: u16,
	ctx: &UiContext,
) -> Ui {
	Ui::from_root(tree(mode, tier, current, query, rows, ctx.charset), width, ctx.clone())
}

/// The picker component tree for one catalog mode.
fn tree(
	mode: Mode,
	tier: PerfTier,
	current: usize,
	query: &str,
	rows: u16,
	charset: Charset,
) -> Box<dyn Component> {
	let list = match mode {
		Mode::Models => model_rows(tier, current, charset),
		Mode::Roles => role_rows(tier, current, charset),
	};
	let status = match mode {
		Mode::Models => STATUS_MODELS,
		Mode::Roles => STATUS_ROLES,
	};
	let hint = match mode {
		Mode::Models => HINT_MODELS,
		Mode::Roles => HINT_ROLES,
	};
	let current_dot = fmts!(" {}", charset.icon(Icon::Enabled));
	let seed = Str::from(query);
	let height = rows.saturating_add(1);
	dom! {
			<box border=round title="Switch Model" pad-x=1>
				<col>
					<text fg=muted truncate>{status}</text>
					<select id="models" filter={seed} h={height}>
						for row in list {
							<option value={row.value} label={row.label} recommended={row.recommended}>
								<td><img src={row.logo} w=2 h=1 trim/></td>
								<td truncate=start grow>
									if !row.prefix.is_empty() {
										<pre fg={row.prefix_fg}>{row.prefix}</pre>
									}
									<pre fg={row.name_fg}>{row.name}</pre>
									if row.current {
										<pre fg={GREEN}>{current_dot.clone()}</pre>
									}
								</td>
								if tier != PerfTier::None {
									<td align=end><pre fg={DIM}>{row.perf}</pre></td>
								}
								<td align=end><pre fg={DIM}>{row.ctx}</pre></td>
								<td align=end><pre fg={DIM}>{row.cost}</pre></td>
							</option>
						}
					</select>
					<spacer h=1/>
					<text id="facts" fg=muted truncate>{" "}</text>
					for model in 0..MODELS.len() {
						<row id={fmts!("chips-{model}")}>
							for (index, chip) in chips(model, current, charset).into_iter().enumerate() {
								if index > 0 {
									<pre fg={DIM}>{" · "}</pre>
								}
								<pre fg={chip.color}>{chip.text}</pre>
							}
						</row>
					}
					<text dim truncate>{hint}</text>
				</col>
			</box>
	}
	.into_component()
}

#[cfg(test)]
mod tests {
	use super::{FRAME_ROWS, Key, MODELS, ModelPicker, Mouse, PickerEvent, Size, UiContext};

	fn ctx() -> UiContext {
		UiContext::default()
	}

	fn opened(current: usize) -> ModelPicker {
		let mut picker = ModelPicker::open(current, &ctx());
		// Establish geometry the way a presenting host does.
		let _ = picker.layer(Size::new(120, 50));
		picker
	}

	#[test]
	fn typing_filters_and_enter_picks_the_ranked_model() {
		let mut picker = opened(0);
		for ch in "flash".chars() {
			assert_eq!(picker.handle_key(Key::Char(ch)), PickerEvent::Consumed);
		}
		assert_eq!(picker.handle_key(Key::Enter), PickerEvent::Pick(1));
	}

	#[test]
	fn escape_clears_the_query_before_closing() {
		let mut picker = opened(0);
		picker.handle_key(Key::Char('x'));
		assert_eq!(picker.handle_key(Key::Esc), PickerEvent::Consumed);
		assert_eq!(picker.handle_key(Key::Esc), PickerEvent::Close);
	}

	#[test]
	fn at_prefix_switches_to_quick_roles() {
		let mut picker = opened(0);
		picker.handle_key(Key::Char('@'));
		for ch in "slow".chars() {
			picker.handle_key(Key::Char(ch));
		}
		assert_eq!(picker.handle_key(Key::Enter), PickerEvent::Pick(5));
	}

	#[test]
	fn cursor_opens_on_the_current_model() {
		let mut picker = opened(5);
		assert_eq!(picker.handle_key(Key::Enter), PickerEvent::Pick(5));
	}

	#[test]
	fn single_steps_wrap_while_jumps_clamp() {
		let mut picker = opened(0);
		picker.handle_key(Key::Up);
		assert_eq!(picker.handle_key(Key::Enter), PickerEvent::Pick(MODELS.len() - 1));
		picker.handle_key(Key::Down);
		assert_eq!(picker.handle_key(Key::Enter), PickerEvent::Pick(0));

		picker.handle_key(Key::PageDown);
		picker.handle_key(Key::PageDown);
		picker.handle_key(Key::PageDown);
		assert_eq!(
			picker.handle_key(Key::Enter),
			PickerEvent::Pick(MODELS.len() - 1),
			"page jumps clamp at the end instead of wrapping"
		);
		picker.handle_key(Key::Home);
		assert_eq!(picker.handle_key(Key::Enter), PickerEvent::Pick(0));
		picker.handle_key(Key::End);
		assert_eq!(picker.handle_key(Key::Enter), PickerEvent::Pick(MODELS.len() - 1));
	}

	#[test]
	fn paste_lands_in_the_query() {
		let mut picker = opened(0);
		picker.handle_paste("gpt-5.6");
		assert_eq!(picker.handle_key(Key::Enter), PickerEvent::Pick(2));
	}

	#[test]
	fn selection_survives_query_edits_min_clamped() {
		let mut picker = opened(5);
		picker.handle_key(Key::Char('a'));
		// "a" matches fewer than six models; the index min-clamps instead of
		// resetting to the top (pi #applyQuery).
		let picked = picker.handle_key(Key::Enter);
		assert!(matches!(picked, PickerEvent::Pick(_)));
	}

	#[test]
	fn mouse_click_activates_the_hit_row_and_outside_closes() {
		let viewport = Size::new(120, 50);
		let mut picker = ModelPicker::open(0, &ctx());
		let band = picker.layer(viewport).band(viewport);
		// Rows begin under the border, status, and query lines.
		let first_row = band.y + 3;
		assert_eq!(
			picker.handle_mouse(10, first_row + 2, Mouse::Click, viewport),
			PickerEvent::Pick(2),
			"click commits the row under the pointer"
		);
		assert_eq!(
			picker.handle_mouse(10, 0, Mouse::Click, viewport),
			PickerEvent::Close,
			"click outside the layer dismisses"
		);
		assert_eq!(
			picker.handle_mouse(10, 0, Mouse::Move, viewport),
			PickerEvent::Consumed,
			"motion outside only clears hover"
		);
	}

	#[test]
	fn wheel_moves_the_selection() {
		let viewport = Size::new(120, 50);
		let mut picker = ModelPicker::open(0, &ctx());
		let band = picker.layer(viewport).band(viewport);
		let inside = band.y + 4;
		picker.handle_mouse(10, inside, Mouse::WheelDown, viewport);
		assert_eq!(picker.handle_key(Key::Enter), PickerEvent::Pick(1));
	}

	#[test]
	fn overlay_height_is_stable_at_forty_percent_regardless_of_filtering() {
		let viewport = Size::new(120, 50);
		// max(5, floor(50 * 0.4) - FRAME_ROWS) list rows plus the frame.
		let expected = (50 * 2 / 5 - FRAME_ROWS).max(5) + FRAME_ROWS;

		let mut picker = ModelPicker::open(0, &ctx());
		let band = picker.layer(viewport).band(viewport);
		assert_eq!(band.rows, expected, "short catalogs pad the list to max visible");
		assert_eq!(band.y, 50 - band.rows, "anchored to the viewport bottom");

		for ch in "flash".chars() {
			picker.handle_key(Key::Char(ch));
		}
		let band = picker.layer(viewport).band(viewport);
		assert_eq!(band.rows, expected, "filtering never shrinks the box");

		for ch in "zzz".chars() {
			picker.handle_key(Key::Char(ch));
		}
		let band = picker.layer(viewport).band(viewport);
		assert_eq!(band.rows, expected, "empty results keep the frame");
	}
}
