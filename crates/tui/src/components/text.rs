use std::time::Duration;

use omp_core::{SmolStr, SmolStrMut};
use smallvec::SmallVec;
use xutf::Text;

use crate::{
	anim,
	component::{Component, MemoKey, PaintCtx, Slot, next_slot},
	frame::{Rect, Style},
	markup::Truncate,
	props::{Prop, PropValue, Props},
	rich::{Pipeline, RichSink, RichText, cell_width},
};

/// Wrapped or truncated text backing the `<text>` markup tag.
pub struct TextLeaf {
	props:        Props,
	slot:         Slot,
	text:         SmolStr,
	rich:         RichText,
	version:      u64,
	cached_width: u16,
	cached:       Option<MemoKey>,
	/// Resolved style baked into `rich` — part of the memo key so prop or
	/// theme changes (including animated swaps) re-render the runs.
	cached_style: Style,
	/// Byte end of the rendered slice of `text` — the whole text without
	/// `reveal`, the shown prefix under it. Part of the memo key so a
	/// moving reveal cursor re-renders the runs.
	cached_end:   usize,
	/// Reveal cursor and grapheme memos; allocated on first paced render.
	reveal:       Option<Box<RevealState>>,
}

impl TextLeaf {
	/// Creates an empty text leaf.
	pub fn new() -> Self {
		Self {
			props:        Props::new(),
			slot:         next_slot(),
			text:         SmolStr::default(),
			rich:         RichText::default(),
			version:      1,
			cached_width: 0,
			cached:       None,
			cached_style: Style::new(),
			cached_end:   0,
			reveal:       None,
		}
	}

	/// Sets one text property.
	pub fn with(mut self, prop: Prop, value: impl Into<PropValue>) -> Self {
		self.props.set(prop, value);
		self.version = self.version.wrapping_add(1);
		self
	}

	/// Sets one text property from a string.
	pub fn with_str(self, prop: Prop, value: &str) -> Self {
		self.with(prop, value)
	}

	/// Appends plain text content.
	pub fn text(mut self, text: impl Into<SmolStr>) -> Self {
		append(&mut self.text, text.into());
		self.version = self.version.wrapping_add(1);
		self
	}

	fn render(&mut self, ctx: &crate::UiContext, width: u16) {
		let width = width.max(1);
		let style = self.props.style(&ctx.theme);
		let key = MemoKey::new(self.version, ctx);
		let end = if let Some(horizon) = self.props.reveal() {
			let reveal = self.reveal.get_or_insert_default();
			reveal.sync(&self.text);
			reveal.advance(ctx.now, horizon)
		} else {
			// Dropping the prop drops the cursor, so re-enabling
			// reveals from the start again.
			self.reveal = None;
			self.text.len()
		};
		if self.cached_width == width
			&& self.cached == Some(key)
			&& self.cached_style == style
			&& self.cached_end == end
		{
			return;
		}
		let visible = &self.text[..end];
		self.rich.clear();
		match self.props.truncate() {
			Some(Truncate::End) => {
				let mut clip = (&mut self.rich).clip(width, Some('…'));
				for (index, line) in visible.split('\n').enumerate() {
					if index > 0 {
						clip.run(style, " ");
					}
					clip.run(style, line);
				}
			},
			Some(Truncate::Start) => {
				let mut runs: SmallVec<(Style, SmolStr), 8> = SmallVec::new();
				for (index, line) in visible.split('\n').enumerate() {
					if index > 0 {
						runs.push((style, SmolStr::new_static(" ")));
					}
					if !line.is_empty() {
						runs.push((style, self.text.slice_ref(line)));
					}
				}
				clip_start_runs(&mut self.rich, width, &runs);
			},
			None => {
				let mut wrap = (&mut self.rich).wrap(width);
				// text is escape-free by contract: ANSI is parsed only at the
				// external ingress (rich::decompose), never inside components
				for (index, line) in visible.split('\n').enumerate() {
					if index > 0 {
						wrap.newline();
					}
					if !line.is_empty() {
						wrap.run(style, line);
					}
				}
				wrap.finish();
			},
		}
		self.cached_width = width;
		self.cached = Some(key);
		self.cached_style = style;
		self.cached_end = end;
	}
}

impl Default for TextLeaf {
	fn default() -> Self {
		Self::new()
	}
}

impl Component for TextLeaf {
	fn props(&self) -> &Props {
		&self.props
	}

	fn props_mut(&mut self) -> &mut Props {
		&mut self.props
	}

	fn slot(&self) -> Slot {
		self.slot
	}

	fn measure(&mut self, _ctx: &crate::UiContext) -> (u16, u16) {
		let mut widest_word = 0;
		let mut total = 0u16;
		for word in self.text.split_whitespace() {
			let width = cell_width(word);
			widest_word = widest_word.max(width);
			total = total.saturating_add(width).saturating_add(1);
		}
		let natural = total.saturating_sub(1);
		// Truncation can always collapse to a lone ellipsis, so a
		// truncated leaf never blocks a column from shrinking.
		if self.props.truncate().is_some() {
			return (natural.min(1), natural);
		}
		(widest_word, natural)
	}

	fn height(&mut self, ctx: &crate::UiContext, width: u16) -> u16 {
		self.render(ctx, width);
		RichText::rows(&self.rich)
	}

	fn paint(&mut self, pc: &mut PaintCtx<'_>, rect: Rect) {
		self.render(pc.ctx, rect.width);
		match self.props.shimmer() {
			Some(period) => {
				paint_rich_shimmer(pc, rect, &self.rich, self.props.align(), period);
				pc.wake(self.slot, pc.now.saturating_add(anim::FRAME));
			},
			None => paint_rich(pc, rect, &self.rich, self.props.align()),
		}
		// An unsettled reveal moves geometry as rows fill, so its frame
		// cadence must relayout, not just repaint.
		if let Some(reveal) = self.reveal.as_deref()
			&& !reveal.is_settled()
		{
			pc.wake_layout(self.slot, pc.now.saturating_add(anim::FRAME));
		}
	}

	fn set_text(&mut self, _ctx: &crate::UiContext, text: SmolStr) -> bool {
		if self.text == text {
			return false;
		}
		self.text = text;
		self.version = self.version.wrapping_add(1);
		true
	}
}

impl TextLeaf {
	/// Verbatim content for single-line flattening by grid cells.
	pub(crate) const fn content(&self) -> &SmolStr {
		&self.text
	}
}

/// Reveal bookkeeping for one leaf: the pacing cursor plus grapheme-cluster
/// memos over its append-only text. Counting resumes from the final counted
/// cluster and slicing from the last shown cluster — an append can extend
/// those clusters but never earlier ones — so each streamed chunk and each
/// cursor step re-segments only the suffix it touched.
#[derive(Default)]
struct RevealState {
	pace:        anim::Reveal,
	/// The text the memos below describe (O(1) clone of the leaf's text).
	seen:        SmolStr,
	/// Grapheme clusters in `seen`.
	total:       usize,
	/// Byte start of the final cluster of `seen`.
	tail:        usize,
	/// Clusters currently shown.
	shown_units: usize,
	/// Byte end of the shown prefix.
	shown_end:   usize,
	/// Byte start of the final shown cluster.
	shown_from:  usize,
}

impl RevealState {
	/// Reconciles the memos with the leaf's current text: an extension
	/// recounts from the final counted cluster, anything else recounts in
	/// full and restarts the cursor.
	fn sync(&mut self, text: &SmolStr) {
		if self.seen == *text {
			return;
		}
		if text.len() > self.seen.len() && text.starts_with(self.seen.as_str()) {
			let (count, tail) = count_clusters(text, self.tail);
			self.total = if self.total == 0 {
				count
			} else {
				self.total - 1 + count
			};
			self.tail = tail;
		} else {
			let (count, tail) = count_clusters(text, 0);
			self.total = count;
			self.tail = tail;
			self.pace.reset();
			self.shown_units = 0;
			self.shown_end = 0;
			self.shown_from = 0;
		}
		self.seen = text.clone();
	}

	/// Advances the cursor at `now` and returns the byte end of the shown
	/// prefix. Always re-walks from the final shown cluster, so an append
	/// that extended it is picked up even when the cursor held still.
	fn advance(&mut self, now: Duration, horizon: Duration) -> usize {
		let units = self.pace.advance(now, self.total, horizon);
		if units >= self.total {
			self.shown_units = self.total;
			self.shown_end = self.seen.len();
			self.shown_from = self.tail;
			return self.shown_end;
		}
		if units == 0 {
			self.shown_units = 0;
			self.shown_end = 0;
			self.shown_from = 0;
			return 0;
		}
		let (start, need, base) = if self.shown_units > 0 && units >= self.shown_units {
			(self.shown_from, units - self.shown_units + 1, self.shown_units - 1)
		} else {
			(0, units, 0)
		};
		let mut offset = start;
		let mut last = start;
		let mut walked = 0;
		for cluster in xutf::graphemes_str(&self.seen[start..]) {
			last = offset;
			offset += cluster.len();
			walked += 1;
			if walked == need {
				break;
			}
		}
		self.shown_units = base + walked;
		self.shown_from = last;
		self.shown_end = offset;
		self.shown_end
	}

	/// Whether the shown prefix covers the whole text.
	const fn is_settled(&self) -> bool {
		self.shown_units >= self.total
	}
}

/// Counts grapheme clusters of `text` from byte offset `start`, also
/// reporting the byte start of the final cluster (where an append could
/// extend it). Empty input reports `(0, start)`.
fn count_clusters(text: &str, start: usize) -> (usize, usize) {
	let mut count = 0;
	let mut tail = start;
	let mut offset = start;
	for cluster in xutf::graphemes_str(&text[start..]) {
		count += 1;
		tail = offset;
		offset += cluster.len();
	}
	(count, tail)
}

/// Preformatted text backing the `<pre>` markup tag.
pub struct Pre {
	props: Props,
	slot:  Slot,
	text:  SmolStr,
}

impl Pre {
	/// Creates an empty preformatted block.
	pub fn new() -> Self {
		Self { props: Props::new(), slot: next_slot(), text: SmolStr::default() }
	}

	/// Sets one property.
	pub fn with(mut self, prop: Prop, value: impl Into<PropValue>) -> Self {
		self.props.set(prop, value);
		self
	}

	/// Sets one property from a string.
	pub fn with_str(self, prop: Prop, value: &str) -> Self {
		self.with(prop, value)
	}

	/// Appends preformatted text content.
	pub fn text(mut self, text: impl Into<SmolStr>) -> Self {
		append(&mut self.text, text.into());
		self
	}

	/// Verbatim content for single-line flattening by grid cells.
	pub(crate) const fn content(&self) -> &SmolStr {
		&self.text
	}
}

impl Default for Pre {
	fn default() -> Self {
		Self::new()
	}
}

impl Component for Pre {
	fn props(&self) -> &Props {
		&self.props
	}

	fn props_mut(&mut self) -> &mut Props {
		&mut self.props
	}

	fn slot(&self) -> Slot {
		self.slot
	}

	fn measure(&mut self, _ctx: &crate::UiContext) -> (u16, u16) {
		let width = self.text.lines().map(cell_width).max().unwrap_or(0);
		(width, width)
	}

	fn height(&mut self, _ctx: &crate::UiContext, _width: u16) -> u16 {
		u16::try_from(self.text.lines().count()).unwrap_or(u16::MAX)
	}

	fn paint(&mut self, pc: &mut PaintCtx<'_>, rect: Rect) {
		let width = self.text.lines().map(cell_width).max().unwrap_or(0);
		let slack = rect.width.saturating_sub(width.min(rect.width));
		let x = rect
			.x
			.saturating_add(alignment_slack(self.props.align(), slack));
		let right = rect.x.saturating_add(rect.width);
		let style = self.props.style(&pc.ctx.theme);
		let clip = pc.clip.min(rect.y.saturating_add(rect.height));
		for (row, line) in self.text.lines().enumerate() {
			let y = rect
				.y
				.saturating_add(u16::try_from(row).unwrap_or(u16::MAX));
			if y >= clip {
				break;
			}
			put_clipped(pc.frame, x, y, right, line, style);
		}
	}

	fn gradient_bounds(&self, content: Rect) -> Option<Rect> {
		let width = self
			.text
			.lines()
			.map(cell_width)
			.max()
			.unwrap_or(0)
			.min(content.width);
		let slack = content.width.saturating_sub(width);
		let x = content
			.x
			.saturating_add(alignment_slack(self.props.align(), slack));
		let height = u16::try_from(self.text.lines().count())
			.unwrap_or(u16::MAX)
			.min(content.height);
		Some(Rect::new(x, content.y, width, height))
	}

	fn set_text(&mut self, _ctx: &crate::UiContext, text: SmolStr) -> bool {
		if self.text == text {
			return false;
		}
		self.text = text;
		true
	}
}

pub(super) fn append(target: &mut SmolStr, suffix: SmolStr) {
	if target.is_empty() {
		*target = suffix;
		return;
	}
	let mut joined = SmolStrMut::with_capacity(target.len().saturating_add(suffix.len()));
	joined.push_str(target);
	joined.push_str(&suffix);
	*target = joined.freeze();
}

/// Emits `runs` as one line clipped to `width` cells keeping the tail: a
/// leading ellipsis replaces however much of the head does not fit.
pub(super) fn clip_start_runs(rich: &mut RichText, width: u16, runs: &[(Style, SmolStr)]) {
	let width = width.max(1);
	let total = runs
		.iter()
		.fold(0_u16, |sum, (_, text)| sum.saturating_add(cell_width(text)));
	if total <= width {
		for (style, text) in runs {
			rich.run(*style, text);
		}
		return;
	}
	// Reserve one cell for the ellipsis, then walk graphemes forward until
	// the dropped prefix frees enough room for the remaining tail.
	let budget = width - 1;
	let mut drop = total.saturating_add(1).saturating_sub(width);
	let marker = runs.first().map_or(Style::new(), |(style, _)| *style);
	rich.run(marker, "…");
	// The clip pipeline guards the exact edge (a wide grapheme straddling
	// the boundary), so the tail can never overrun the cell budget.
	let mut clip = (&mut *rich).clip(budget.saturating_add(1), None);
	for (style, text) in runs {
		if drop == 0 {
			clip.run(*style, text);
			continue;
		}
		let run_width = cell_width(text);
		if run_width <= drop {
			drop -= run_width;
			continue;
		}
		let mut cut = text.len();
		let mut walked = 0_u16;
		for (offset, grapheme) in text.as_str().grapheme_indices() {
			if walked >= drop {
				cut = offset;
				break;
			}
			walked = walked.saturating_add(cell_width(grapheme));
		}
		drop = 0;
		clip.run(*style, &text.as_str()[cut..]);
	}
}

pub(super) fn truncate_rich(
	rich: &mut RichText,
	width: u16,
	fallback: Style,
	truncate: Option<Truncate>,
) {
	let Some(mode) = truncate else { return };
	if RichText::rows(rich) <= 1 {
		return;
	}
	match mode {
		Truncate::End => {
			let row: SmallVec<(Style, SmolStr), 4> = rich
				.row_runs(0)
				.map(|(style, text)| (style, SmolStr::new(text)))
				.collect();
			rich.clear();
			{
				let mut clip = (&mut *rich).clip(width.saturating_sub(1), None);
				for (style, text) in &row {
					clip.run(*style, text);
				}
			}
			let style = rich.row_runs(0).last().map_or(fallback, |(style, _)| style);
			rich.run(style, "…");
		},
		Truncate::Start => {
			// Rejoin the wrapped rows with single spaces and keep the tail.
			let mut joined: SmallVec<(Style, SmolStr), 8> = SmallVec::new();
			for row in 0..RichText::rows(rich) {
				if row > 0 {
					let style = joined.last().map_or(fallback, |(style, _)| *style);
					joined.push((style, SmolStr::new_static(" ")));
				}
				for (style, text) in rich.row_runs(row) {
					joined.push((style, SmolStr::new(text)));
				}
			}
			rich.clear();
			clip_start_runs(rich, width, &joined);
		},
	}
}

pub(super) const fn alignment_slack(align: crate::markup::Align, slack: u16) -> u16 {
	match align {
		crate::markup::Align::Start => 0,
		crate::markup::Align::Center => slack / 2,
		crate::markup::Align::End => slack,
	}
}

pub(super) fn put_clipped(
	frame: &mut crate::Frame,
	x: u16,
	y: u16,
	right: u16,
	text: &str,
	style: Style,
) -> u16 {
	let room = right.saturating_sub(x);
	if room == 0 {
		return x;
	}
	let visible = text.truncate_width(usize::from(room));
	frame.put(x, y, visible, style)
}

pub(super) fn paint_rich(
	pc: &mut PaintCtx<'_>,
	rect: Rect,
	rich: &RichText,
	align: crate::markup::Align,
) {
	let right = rect.x.saturating_add(rect.width);
	let clip = pc.clip.min(rect.y.saturating_add(rect.height));
	for row in 0..RichText::rows(rich) {
		let y = rect.y.saturating_add(row);
		if y >= clip {
			break;
		}
		let slack = rect.width.saturating_sub(rich.row_width(row));
		let mut x = rect.x.saturating_add(alignment_slack(align, slack));
		for (style, text) in rich.row_runs(row) {
			x = put_clipped(pc.frame, x, y, right, text, style);
			if x >= right {
				break;
			}
		}
	}
}

/// [`paint_rich`] under a `shimmer` crest: every cell restyles by its
/// distance from the sweep, and each row rides the same phase.
fn paint_rich_shimmer(
	pc: &mut PaintCtx<'_>,
	rect: Rect,
	rich: &RichText,
	align: crate::markup::Align,
	period: std::time::Duration,
) {
	let right = rect.x.saturating_add(rect.width);
	let clip = pc.clip.min(rect.y.saturating_add(rect.height));
	for row in 0..RichText::rows(rich) {
		let y = rect.y.saturating_add(row);
		if y >= clip {
			break;
		}
		let slack = rect.width.saturating_sub(rich.row_width(row));
		let start = rect.x.saturating_add(alignment_slack(align, slack));
		let shimmer = anim::Shimmer::new(pc.now, period, rich.row_width(row));
		let mut x = start;
		'runs: for (style, text) in rich.row_runs(row) {
			for grapheme in xutf::graphemes_str(text) {
				if x >= right {
					break 'runs;
				}
				let next = pc
					.frame
					.put(x, y, grapheme, shimmer.style_at(x - start, style));
				if next == x {
					break 'runs;
				}
				x = next;
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{
		UiContext,
		component::{Component, PaintCtx},
		components::{Callout, Icon, Latex, Markdown},
		frame::{Frame, Rect, Size},
		test_support::frame_row_text,
		ui::Ui,
	};

	fn paint(component: &mut dyn Component, width: u16, height: u16) -> Frame {
		let ctx = UiContext::default();
		let mut frame = Frame::new(Size::new(width, height));
		let mut hits = Vec::new();
		let mut wakes = Vec::new();
		let mut pc = PaintCtx::new(&mut frame, &ctx, &mut hits, &mut wakes);
		component.paint(&mut pc, Rect::new(0, 0, width, height));
		frame
	}

	#[test]
	fn text_wraps_and_aligns_rows() {
		let mut text = TextLeaf::new()
			.with(Prop::Align, "center")
			.text("one two three");
		let frame = paint(&mut text, 7, 2);
		assert_eq!(frame_row_text(&frame, 0), "one two");
		assert_eq!(frame_row_text(&frame, 1), " three");
	}

	#[test]
	fn pre_paints_verbatim_rows() {
		let mut pre = Pre::new().text("A\n B");
		let frame = paint(&mut pre, 4, 2);
		assert_eq!(frame_row_text(&frame, 0), "A");
		assert_eq!(frame_row_text(&frame, 1), " B");
	}

	#[test]
	fn markdown_paints_paragraph_and_fenced_code() {
		let mut markdown = Markdown::new().text("paragraph\n\n```rust\nlet x = 1;\n```");
		let frame = paint(&mut markdown, 24, 8);
		let rows = (0..8)
			.map(|row| frame_row_text(&frame, row))
			.collect::<Vec<_>>();
		assert!(rows.iter().any(|row| row.contains("paragraph")));
		assert!(rows.iter().any(|row| row.contains("let x = 1;")));
	}

	#[test]
	fn latex_paints_inline_when_block_layout_is_unavailable() {
		let mut latex = Latex::new().text(r"\unknown{x}");
		let frame = paint(&mut latex, 20, 3);
		assert!((0..3).any(|row| !frame_row_text(&frame, row).is_empty()));
	}

	#[test]
	fn callout_paints_header_and_body_rail() {
		let mut callout = Callout::new()
			.with(Prop::Title, "Advisor")
			.with(Prop::Badge, "1")
			.text("body");
		let frame = paint(&mut callout, 20, 3);
		assert!(frame_row_text(&frame, 0).contains("Advisor"));
		assert!(frame_row_text(&frame, 1).contains("body"));
		assert!(frame_row_text(&frame, 1).starts_with('▎'));
	}

	#[test]
	fn icon_measure_matches_painted_glyph_width() {
		let ctx = UiContext::default();
		let mut icon = Icon::named("folder");
		let (min, natural) = icon.measure(&ctx);
		assert_eq!(min, natural);
		let frame = paint(&mut icon, min.max(1), 1);
		assert_eq!(cell_width(&frame_row_text(&frame, 0)), min);
	}

	#[test]
	fn reveal_types_out_streamed_appends_and_settles() {
		let mut ui = Ui::from_root(
			TextLeaf::new()
				.with(Prop::Reveal, true)
				.with(Prop::Id, "stream")
				.text("abcdef"),
			20,
			UiContext::default(),
		);
		// The construction paint arms the cursor without revealing anything
		// and schedules the first frame.
		assert_eq!(frame_row_text(ui.frame(), 0), "");
		assert_eq!(ui.next_wake(), Some(Duration::from_millis(33)));

		// Each tick earns one 33ms frame at the 90 clusters/s floor: 2.97.
		assert!(ui.tick(Duration::from_millis(34)));
		assert_eq!(frame_row_text(ui.frame(), 0), "ab");
		ui.tick(Duration::from_millis(68));
		assert_eq!(frame_row_text(ui.frame(), 0), "abcde");
		ui.tick(Duration::from_millis(102));
		assert_eq!(frame_row_text(ui.frame(), 0), "abcdef");
		assert_eq!(ui.next_wake(), None, "a settled reveal stops waking");

		// An append resumes from the cursor instead of jumping or resetting.
		assert!(ui.set_text("stream", "abcdefghijkl"));
		assert_eq!(frame_row_text(ui.frame(), 0), "abcdef");
		assert!(ui.next_wake().is_some(), "new backlog re-arms the frame cadence");
		ui.tick(Duration::from_millis(136));
		assert_eq!(frame_row_text(ui.frame(), 0), "abcdefgh");
		ui.tick(Duration::from_millis(170));
		assert_eq!(frame_row_text(ui.frame(), 0), "abcdefghijk");
		ui.tick(Duration::from_millis(204));
		assert_eq!(frame_row_text(ui.frame(), 0), "abcdefghijkl");

		// A replacement restarts the typewriter from nothing.
		assert!(ui.set_text("stream", "xyz"));
		assert_eq!(frame_row_text(ui.frame(), 0), "");
		ui.tick(Duration::from_millis(238));
		assert_eq!(frame_row_text(ui.frame(), 0), "xy");
		ui.tick(Duration::from_millis(272));
		assert_eq!(frame_row_text(ui.frame(), 0), "xyz");
		assert_eq!(ui.next_wake(), None);
	}

	#[test]
	fn reveal_grows_height_as_rows_fill() {
		let mut ui = Ui::from_root(
			TextLeaf::new().with(Prop::Reveal, true).text("aaa bbb"),
			3,
			UiContext::default(),
		);
		assert_eq!(ui.height(), 1, "an empty reveal holds the blank row a bare leaf has");
		ui.tick(Duration::from_millis(34));
		assert_eq!(ui.height(), 1, "two clusters still fit the first row");
		ui.tick(Duration::from_millis(67));
		assert_eq!(ui.height(), 2, "the fifth cluster wraps onto a second row");
		ui.tick(Duration::from_millis(100));
		assert_eq!(frame_row_text(ui.frame(), 0), "aaa");
		assert_eq!(frame_row_text(ui.frame(), 1), "bbb");
	}

	#[test]
	fn reveal_state_extends_counts_across_cluster_boundaries() {
		let mut state = RevealState::default();
		state.sync(&SmolStr::new("e"));
		assert_eq!(state.total, 1);
		// The appended combining mark extends the final cluster in place.
		state.sync(&SmolStr::new("e\u{301}"));
		assert_eq!(state.total, 1);
		state.sync(&SmolStr::new("e\u{301}f"));
		assert_eq!(state.total, 2);
		// Replacement restarts the cursor from nothing.
		state.sync(&SmolStr::new("zz"));
		assert_eq!(state.total, 2);
		assert_eq!(state.advance(Duration::ZERO, Duration::from_millis(250)), 0);
	}

	#[test]
	fn reveal_state_reslices_the_boundary_cluster_after_an_append() {
		let mut state = RevealState::default();
		state.sync(&SmolStr::new("ab"));
		// A zero horizon snaps the cursor to everything counted so far.
		assert_eq!(state.advance(Duration::ZERO, Duration::ZERO), 2);
		state.sync(&SmolStr::new("ab\u{301}c"));
		// The shown boundary cluster grew; the slice re-walks it before
		// advancing rather than splitting the combining mark off.
		let end = state.advance(Duration::from_millis(1), Duration::from_millis(250));
		assert_eq!(&state.seen[..end], "ab\u{301}");
		assert!(!state.is_settled());
	}
}
