use std::{
	cell::RefCell,
	rc::Rc,
	time::{Duration, Instant},
};

use omp_core::{Str, fmts};
use serde_json::Value;
use smallvec::SmallVec;
use xutf::Text;

use super::Img;
use crate::{
	BufferOutcome, EditBuffer,
	component::{
		Cached, Component, EventCtx, Flow, Hit, HitTag, IntoComponent, PaintCtx, Slot, next_slot,
	},
	context::{Charset, UiContext},
	frame::{Color, Frame, Rect, Style},
	input::{Key, Mouse, UiEvent, byte_at_column, sanitize_paste},
	markup::Border,
	props::{Prop, PropValue, Props},
	syntax::{SyntaxRun, highlight_xml, xml_comment_state},
};

/// Focusable editable leaf used by [`EditorPane`].
pub struct EditInput {
	props:       Props,
	slot:        Slot,
	buffer:      EditBuffer,
	attachments: Option<Attachments>,
	dragging:    bool,
	last_click:  Option<((u16, u16), Instant)>,
}

impl EditInput {
	/// Creates an empty editor.
	pub fn new() -> Self {
		Self {
			props:       Props::new(),
			slot:        next_slot(),
			buffer:      EditBuffer::default(),
			attachments: None,
			dragging:    false,
			last_click:  None,
		}
	}

	/// Binds the composer's shared attachment queue: image path drops stage
	/// automatically, large pastes collapse into atomic `<icon> #N` chips,
	/// and deleting a chip hides its card until an undo restores it.
	pub fn attachments(mut self, attachments: Attachments) -> Self {
		self.attachments = Some(attachments);
		self
	}

	/// Hides staged attachments whose chip left the buffer (an undo that
	/// restores the chip re-shows them), returning whether anything
	/// changed.
	fn reconcile(&self, ctx: &UiContext) -> bool {
		let Some(attachments) = &self.attachments else {
			return false;
		};
		let text = self.buffer.text();
		let ranges = self.buffer.atom_ranges();
		attachments.set_visible(|attachment| {
			let chip = chip_label(attachment, ctx.charset);
			ranges
				.iter()
				.any(|&(start, end)| text.get(start..end) == Some(chip.as_str()))
		})
	}

	#[allow(dead_code, reason = "acceptance-suite probe")]
	pub(crate) const fn buffer(&self) -> &EditBuffer {
		&self.buffer
	}

	#[allow(dead_code, reason = "acceptance-suite probe")]
	pub(crate) const fn buffer_mut(&mut self) -> &mut EditBuffer {
		&mut self.buffer
	}

	/// Sets one editor property, updating its buffer for `value`.
	pub fn with(mut self, prop: Prop, value: impl Into<PropValue>) -> Self {
		let value = value.into();
		if prop == Prop::Value
			&& let PropValue::Str(text) = &value
		{
			self.buffer = EditBuffer::new(text);
		}
		self.props.set(prop, value);
		self
	}

	/// Sets one editor property from a string.
	pub fn with_str(self, prop: Prop, value: &str) -> Self {
		self.with(prop, value)
	}

	fn text_width(width: u16) -> u16 {
		width.saturating_sub(2).max(1)
	}

	fn page_rows(ec: &EventCtx<'_>) -> usize {
		usize::from(ec.view_rows.max(1))
	}
}

impl Default for EditInput {
	fn default() -> Self {
		Self::new()
	}
}

impl Component for EditInput {
	fn props(&self) -> &Props {
		&self.props
	}

	fn props_mut(&mut self) -> &mut Props {
		&mut self.props
	}

	fn slot(&self) -> Slot {
		self.slot
	}

	fn measure(&mut self, _ctx: &UiContext) -> (u16, u16) {
		(20, 40)
	}

	fn height(&mut self, _ctx: &UiContext, _width: u16) -> u16 {
		4
	}

	fn paint(&mut self, pc: &mut PaintCtx<'_>, rect: Rect) {
		pc.hits
			.push(Hit { rect, slot: self.slot, tag: HitTag::Press });
		let focused = pc.focus == Some(self.slot);
		let text = self.buffer.text();
		let atoms = self.buffer.atom_ranges();
		let rows = self
			.buffer
			.rows(Self::text_width(rect.width), usize::from(rect.height));
		let rail = if focused {
			Style::new().fg(pc.ctx.theme.accent)
		} else {
			Style::new().fg(pc.ctx.theme.muted)
		};
		let cursor_style = Style::new()
			.fg(pc.ctx.theme.contrast)
			.bg(pc.ctx.theme.accent);
		let buffer_start = text.as_ptr() as usize;
		let mut scanned = 0;
		let mut in_comment = false;
		for (row, content) in rows.iter().enumerate() {
			let y = rect
				.y
				.saturating_add(u16::try_from(row).unwrap_or(u16::MAX));
			if y >= pc.clip {
				break;
			}
			let start = (content.text.as_ptr() as usize)
				.saturating_sub(buffer_start)
				.min(text.len());
			in_comment = xml_comment_state(&text[scanned..start], in_comment);
			let (runs, next_comment) = highlight_xml(content.text, &pc.ctx.theme, in_comment);
			in_comment = next_comment;
			scanned = start.saturating_add(content.text.len()).min(text.len());

			// Chip styling wins over the XML runs it overlaps; the style
			// derives from the FULL atom text, so a chip wrapped across
			// rows keeps its color on every row.
			let mut chips: SmallVec<(usize, usize, Style), 4> = SmallVec::new();
			for &(atom_start, atom_end) in &atoms {
				let from = atom_start.max(start);
				let to = atom_end.min(scanned);
				if from < to
					&& let Some(style) = chip_style(&text[atom_start..atom_end])
				{
					chips.push((from - start, to - start, style));
				}
			}
			let runs = overlay_chip_runs(&runs, &chips, content.text.len());

			let x = pc.frame.put(rect.x, y, pc.ctx.charset.rail(), rail);
			let selection = self.buffer.selection_span(content);
			let selection_bytes = selection.map(|(start, end)| {
				(byte_at_column(content.text, start), byte_at_column(content.text, end))
			});
			let cursor = (focused && selection.is_none())
				.then_some(content.cursor_column)
				.flatten()
				.map(|column| byte_at_column(content.text, column));
			paint_xml_runs(
				pc.frame,
				x,
				y,
				content.text,
				&runs,
				selection_bytes,
				pc.ctx.theme.selection,
				cursor,
				cursor_style,
			);
		}
	}

	fn focusable(&self) -> bool {
		true
	}

	fn key(&mut self, ec: &mut EventCtx<'_>, key: Key) -> Flow {
		if matches!(key, Key::Up) && self.buffer.at_visual_start()
			|| matches!(key, Key::Down) && self.buffer.at_visual_end()
		{
			return Flow::Skip;
		}
		if matches!(
			self
				.buffer
				.handle(key, Self::text_width(ec.width), Self::page_rows(ec)),
			BufferOutcome::Changed
		) {
			if self.reconcile(ec.ctx) {
				// The pane's attachment band changed height outside this
				// leaf's own box.
				ec.request_layout();
			}
			match self.buffer.take_copied() {
				// The host owns the clipboard write (OSC 52 / native).
				Some(text) => Flow::Event(UiEvent::Copied(text)),
				None => Flow::Consumed,
			}
		} else {
			Flow::Skip
		}
	}

	fn mouse(
		&mut self,
		ec: &mut EventCtx<'_>,
		_tag: HitTag,
		at: (u16, u16),
		rect: Rect,
		mouse: Mouse,
	) -> Flow {
		match mouse {
			Mouse::Click => {
				let now = Instant::now();
				let same_cell = self.last_click.is_some_and(|(cell, then)| {
					cell == at && now.duration_since(then) <= Duration::from_millis(400)
				});
				let row = usize::from(at.1.saturating_sub(rect.y));
				let column = at.0.saturating_sub(rect.x + 2);
				let width = Self::text_width(rect.width);
				if same_cell {
					self.buffer.select_word_visual_row(row, column, width);
					self.last_click = None;
				} else {
					self.buffer.set_cursor_visual_row(row, column, width);
					self.last_click = Some((at, now));
				}
				self.dragging = true;
				Flow::Consumed
			},
			Mouse::Drag if self.dragging => {
				self.buffer.extend_selection_visual_row(
					usize::from(at.1.saturating_sub(rect.y)),
					at.0.saturating_sub(rect.x + 2),
					Self::text_width(rect.width),
				);
				Flow::Consumed
			},
			Mouse::Release if self.dragging => {
				self.dragging = false;
				Flow::Consumed
			},
			Mouse::WheelUp | Mouse::WheelDown => {
				let delta = if mouse == Mouse::WheelUp { -1 } else { 1 };
				if self
					.buffer
					.scroll_rows(delta, Self::text_width(ec.width), Self::page_rows(ec))
				{
					Flow::Consumed
				} else {
					Flow::Skip
				}
			},
			Mouse::RightClick
			| Mouse::MiddleClick
			| Mouse::Move
			| Mouse::Drag
			| Mouse::Release
			| Mouse::WheelLeft
			| Mouse::WheelRight => Flow::Skip,
		}
	}

	fn paste(&mut self, ec: &mut EventCtx<'_>, text: &str) -> Flow {
		if let Some(attachments) = &self.attachments {
			let paths = crate::paste::dropped_paths(text);
			// Requiring real files keeps prose that merely resembles a path out of the
			// band.
			if !paths.is_empty()
				&& paths.iter().all(|path| {
					crate::paste::is_image_path(path) && std::path::Path::new(path.as_str()).exists()
				}) {
				for path in paths {
					let attachment = attachments.push_image(path.clone());
					let chip = chip_label(&attachment, ec.ctx.charset);
					let _ = self.buffer.insert_reference(&chip, path.as_str());
					let _ = self.buffer.insert_text(" ");
				}
				ec.request_layout();
				return Flow::Consumed;
			}
		}
		if let Some(attachments) = &self.attachments
			&& collapses_to_chip(text)
		{
			let attachment = attachments.push_text(text);
			let chip = chip_label(&attachment, ec.ctx.charset);
			let payload = sanitize_paste(text);
			let _ = self.buffer.insert_reference(&chip, &payload);
			let _ = self.buffer.insert_text(" ");
			ec.request_layout();
			return Flow::Consumed;
		}
		let sanitized = sanitize_paste(text);
		let path_prefix = matches!(sanitized.as_bytes().first(), Some(b'/' | b'~' | b'.'));
		let before_is_word = self.buffer.text()[..self.buffer.cursor()]
			.chars()
			.next_back()
			.is_some_and(|ch| ch.is_alphanumeric() || ch == '_');
		if path_prefix && before_is_word {
			let _ = self.buffer.insert_text(" ");
		}
		if matches!(self.buffer.insert_text(&sanitized), BufferOutcome::Changed) {
			Flow::Consumed
		} else {
			Flow::Skip
		}
	}

	fn paste_raw(&mut self, _ec: &mut EventCtx<'_>, text: &str) -> Flow {
		// Verbatim insertion (pi's raw-paste binding): the text stays inline
		// and editable — no attachment staging, no large-paste chip, no
		// auto-spacing. Sanitization still applies inside `insert_text`.
		if matches!(self.buffer.insert_text(text), BufferOutcome::Changed) {
			Flow::Consumed
		} else {
			Flow::Skip
		}
	}

	fn value(&self, out: &mut serde_json::Map<String, Value>) {
		if let Some(id) = self.props.id() {
			out.insert(id.to_string(), Value::String(self.buffer.expanded_text()));
		}
	}

	fn set_text(&mut self, _ctx: &UiContext, text: Str) -> bool {
		if self.buffer.text() == text {
			return false;
		}
		self.buffer = EditBuffer::new(&text);
		true
	}
}

/// Attachment preview thumbnail content, in cells.
const PREVIEW_COLS: u16 = 12;
const PREVIEW_ROWS: u16 = 4;
/// Blank columns between adjacent preview frames.
const PREVIEW_GAP: u16 = 2;
/// One preview frame: thumbnail content plus its colored border.
const PREVIEW_BOX_COLS: u16 = PREVIEW_COLS + 2;
const PREVIEW_BOX_ROWS: u16 = PREVIEW_ROWS + 2;
/// Identity palette cycled by marker number; see [`attachment_color`].
const ATTACHMENT_COLORS: [Color; 6] = [
	Color::Rgb(255, 179, 102),
	Color::Rgb(125, 207, 255),
	Color::Rgb(189, 147, 249),
	Color::Rgb(105, 220, 158),
	Color::Rgb(255, 141, 188),
	Color::Rgb(240, 223, 120),
];

/// Identity color for attachment marker `N` (1-based).
///
/// The preview frame and any host-rendered reference chip share it, so an
/// attachment stays recognizable from composer to transcript.
pub const fn attachment_color(marker: usize) -> Color {
	ATTACHMENT_COLORS[marker.saturating_sub(1) % ATTACHMENT_COLORS.len()]
}

/// The composer chip text for one attachment: `<icon> #N` with the tier's
/// image or text-file glyph.
///
/// Hosts insert it as an atomic reference (see
/// [`crate::EditBuffer::insert_reference`]); [`EditInput`] does so
/// automatically for staged image path drops and large paste cards.
pub fn chip_label(attachment: &Attachment, charset: Charset) -> Str {
	let icon = charset.icon(match attachment.content {
		AttachmentContent::Image { .. } => crate::Icon::Image,
		AttachmentContent::Text { .. } => crate::Icon::TextFile,
	});
	fmts!("{icon} #{}", attachment.marker)
}

/// Chip style for an atomic marker: a trailing `#N` selects the marker's
/// identity color. `None` leaves other atoms on their base styling.
fn chip_style(marker: &str) -> Option<Style> {
	let digits = &marker[marker.rfind('#')? + 1..];
	if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
		return None;
	}
	let marker: usize = digits.parse().ok()?;
	(marker > 0).then(|| Style::new().fg(attachment_color(marker)).bold())
}

/// Whether a paste is large enough to collapse into an attachment chip.
fn collapses_to_chip(text: &str) -> bool {
	text.len() > 1000 || text.bytes().filter(|byte| *byte == b'\n').count() >= 10
}

/// Splices chip-styled runs over one row's syntax runs; chips win where
/// they overlap. `len` is the row's byte length.
fn overlay_chip_runs(
	runs: &[SyntaxRun],
	chips: &[(usize, usize, Style)],
	len: usize,
) -> SmallVec<SyntaxRun, 16> {
	let mut merged: SmallVec<SyntaxRun, 16> = SmallVec::new();
	if chips.is_empty() {
		merged.extend_from_slice(runs);
		return merged;
	}
	let base = |at: usize| {
		runs
			.iter()
			.find(|run| run.start <= at && at < run.end)
			.map_or_else(
				|| {
					let next = runs
						.iter()
						.map(|run| run.start)
						.filter(|start| *start > at)
						.min()
						.unwrap_or(len);
					(next, Style::new())
				},
				|run| (run.end, run.style),
			)
	};
	fn emit(
		base: &impl Fn(usize) -> (usize, Style),
		from: usize,
		to: usize,
		merged: &mut SmallVec<SyntaxRun, 16>,
	) {
		let mut at = from;
		while at < to {
			let (run_end, style) = base(at);
			let end = run_end.min(to);
			merged.push(SyntaxRun { start: at, end, style });
			at = end;
		}
	}
	let mut at = 0;
	for &(start, end, style) in chips {
		emit(&base, at, start, &mut merged);
		merged.push(SyntaxRun { start, end, style });
		at = end;
	}
	emit(&base, at, len, &mut merged);
	merged
}

/// One staged composer attachment.
#[derive(Clone)]
pub struct Attachment {
	/// What the attachment holds and how its preview card renders.
	pub content: AttachmentContent,
	/// 1-based marker number (`#N`), stable until the queue is drained.
	pub marker:  usize,
	/// Identity color shared by the preview frame and chip highlights.
	pub color:   Color,
}

/// Content behind one [`Attachment`].
#[derive(Clone)]
pub enum AttachmentContent {
	/// An image staged from a file source.
	Image {
		/// Image source path.
		source:     Str,
		/// Pixel dimensions probed from the source header, when recognized.
		dimensions: Option<(u32, u32)>,
	},
	/// Pasted text collapsed out of the composer.
	Text {
		/// Leading rows previewed inside the card, pre-clipped to the frame.
		snippet: Str,
		/// Logical line count of the paste.
		lines:   usize,
		/// Character count of the paste.
		chars:   usize,
	},
}

/// Shared handle to the attachments staged on an [`EditorPane`] composer.
///
/// The composer's owner keeps a clone (see [`EditorPane::attachments`]),
/// stages images with [`Attachments::push_image`] and collapsed pastes with
/// [`Attachments::push_text`], and drains the queue on submit with
/// [`Attachments::take`]. The pane renders one framed card per visible
/// attachment above its status band, tinted with the attachment's identity
/// color and captioned with its `#N` marker plus pixel resolution or paste
/// size.
///
/// [`Attachments::set_visible`] reconciles the band with the composer text:
/// an attachment whose inline reference was deleted hides — and returns on
/// undo — without losing its marker number.
///
/// Mutations change the pane's height out of band, so the owner triggers a
/// relayout afterwards (e.g. [`crate::Ui::resize`] at the current width).
#[derive(Clone, Default)]
pub struct Attachments {
	state: Rc<RefCell<AttachmentState>>,
}

#[derive(Default)]
struct AttachmentState {
	staged:  Vec<Staged>,
	/// Monotonic marker source; survives hides so numbers stay stable.
	counter: usize,
	version: u64,
}

struct Staged {
	attachment: Attachment,
	hidden:     bool,
}

impl Attachments {
	/// Creates an empty attachment queue.
	pub fn new() -> Self {
		Self::default()
	}

	/// Stages an image source, probing its pixel dimensions from the file
	/// header, and returns the staged descriptor.
	pub fn push_image(&self, source: impl Into<Str>) -> Attachment {
		let source = source.into();
		let dimensions = probe_dimensions(source.as_str());
		self.stage(AttachmentContent::Image { source, dimensions })
	}

	/// Stages pasted text collapsed out of the composer and returns the
	/// staged descriptor.
	pub fn push_text(&self, text: &str) -> Attachment {
		let lines = text.bytes().filter(|byte| *byte == b'\n').count() + 1;
		let chars = text.chars().count();
		let mut snippet = String::new();
		for (index, line) in text.split('\n').take(usize::from(PREVIEW_ROWS)).enumerate() {
			if index > 0 {
				snippet.push('\n');
			}
			snippet.push_str(&line[..byte_at_column(line, PREVIEW_COLS)]);
		}
		self.stage(AttachmentContent::Text { snippet: Str::from(snippet), lines, chars })
	}

	fn stage(&self, content: AttachmentContent) -> Attachment {
		let mut state = self.state.borrow_mut();
		state.counter += 1;
		let attachment =
			Attachment { content, marker: state.counter, color: attachment_color(state.counter) };
		state
			.staged
			.push(Staged { attachment: attachment.clone(), hidden: false });
		state.version += 1;
		attachment
	}

	/// Drains the whole queue, restarting marker numbering, and returns
	/// the visible attachments in marker order. Hidden descriptors — whose
	/// inline references the user deleted — are discarded, never handed to
	/// the host.
	pub fn take(&self) -> Vec<Attachment> {
		let mut state = self.state.borrow_mut();
		if !state.staged.is_empty() {
			state.version += 1;
		}
		state.counter = 0;
		std::mem::take(&mut state.staged)
			.into_iter()
			.filter(|staged| !staged.hidden)
			.map(|staged| staged.attachment)
			.collect()
	}

	/// Shows exactly the attachments `visible` accepts and hides the rest
	/// (ones whose inline reference was deleted), returning whether
	/// anything changed. Hidden attachments stay staged, so an undo can
	/// bring them back.
	pub fn set_visible(&self, mut visible: impl FnMut(&Attachment) -> bool) -> bool {
		let mut state = self.state.borrow_mut();
		let mut changed = false;
		for staged in &mut state.staged {
			let hide = !visible(&staged.attachment);
			changed |= staged.hidden != hide;
			staged.hidden = hide;
		}
		if changed {
			state.version += 1;
		}
		changed
	}

	/// Number of visible attachments.
	pub fn len(&self) -> usize {
		self
			.state
			.borrow()
			.staged
			.iter()
			.filter(|staged| !staged.hidden)
			.count()
	}

	/// Whether no attachment is visible.
	pub fn is_empty(&self) -> bool {
		self.len() == 0
	}
}

/// Probes `source`'s pixel dimensions from its header bytes.
fn probe_dimensions(source: &str) -> Option<(u32, u32)> {
	let bytes = std::fs::read(source).ok()?;
	let probed = crate::imagefmt::dimensions(&bytes)?;
	Some((probed.width, probed.height))
}

/// Editor shell with replaceable editable content, status chrome, and an
/// attachment preview band.
pub struct EditorPane {
	props:       Props,
	slot:        Slot,
	/// `[input, status?, previews..]`; previews start at
	/// [`EditorPane::preview_start`].
	children:    SmallVec<Cached, 2>,
	has_status:  bool,
	attachments: Attachments,
	/// Attachment-state version the preview children were built from.
	synced:      u64,
	/// Preview band rectangle captured at place time; zero when empty.
	band:        Rect,
}

impl EditorPane {
	/// Creates an editor shell with a default [`EditInput`].
	pub fn new() -> Self {
		let attachments = Attachments::new();
		let mut children = SmallVec::new();
		children.push(Cached::new(Box::new(EditInput::new().attachments(attachments.clone()))));
		Self {
			props: Props::new(),
			slot: next_slot(),
			children,
			has_status: false,
			attachments,
			synced: 0,
			band: Rect::new(0, 0, 0, 0),
		}
	}

	/// Replaces the editable leaf.
	pub fn input(mut self, input: impl IntoComponent) -> Self {
		self.children[0] = Cached::new(input.into_component());
		self
	}

	/// Adds or replaces the status band above the editable content.
	pub fn status(mut self, status: impl IntoComponent) -> Self {
		let status = Cached::new(status.into_component());
		if self.has_status {
			self.children[1] = status;
		} else {
			self.children.insert(1, status);
			self.has_status = true;
		}
		self
	}

	/// Shared handle to this composer's staged attachments.
	pub fn attachments(&self) -> Attachments {
		self.attachments.clone()
	}

	/// Index of the first image-preview child: input, then the optional
	/// status.
	fn preview_start(&self) -> usize {
		1 + usize::from(self.has_status)
	}

	/// Rows of the preview band: framed cards plus a blank spacer row
	/// above the status line.
	fn band_rows(&self) -> u16 {
		if self.attachments.is_empty() {
			0
		} else {
			PREVIEW_BOX_ROWS + 1
		}
	}

	/// Rebuilds the image-preview children when the shared attachment
	/// queue changed.
	fn sync_attachments(&mut self) {
		let state = self.attachments.state.borrow();
		if state.version == self.synced {
			return;
		}
		self.synced = state.version;
		let keep = 1 + usize::from(self.has_status);
		self.children.truncate(keep);
		for staged in state.staged.iter().filter(|staged| !staged.hidden) {
			if let AttachmentContent::Image { source, .. } = &staged.attachment.content {
				self.children.push(Cached::new(Box::new(
					Img::new()
						.with(Prop::Src, source.clone())
						.with(Prop::W, PREVIEW_COLS)
						.with(Prop::H, PREVIEW_ROWS)
						.with(Prop::Trim, true),
				)));
			}
		}
	}

	/// Paints each attachment card: a rounded frame tinted with its
	/// identity color, captioned `<icon> #N` on the top edge and the pixel
	/// resolution or paste size on the bottom edge. Image cards hold a
	/// thumbnail; paste cards preview the leading text.
	fn paint_previews(&mut self, pc: &mut PaintCtx<'_>) {
		if self.band.height == 0 {
			return;
		}
		let (tl, tr, bl, br, horizontal, vertical) = pc.ctx.charset.border(Border::Round);
		let handle = self.attachments.clone();
		let state = handle.state.borrow();
		let right_limit = self.band.x.saturating_add(self.band.width);
		let top = self.band.y;
		let bottom = top.saturating_add(PREVIEW_BOX_ROWS.saturating_sub(1));
		let snippet_style = Style::new().fg(pc.ctx.theme.muted);
		let mut glyph = [0_u8; 4];
		let mut x = self.band.x;
		let mut image_child = self.preview_start();
		for staged in state.staged.iter().filter(|staged| !staged.hidden) {
			let attachment = &staged.attachment;
			if x.saturating_add(PREVIEW_BOX_COLS) > right_limit {
				break;
			}
			let line = Style::new().fg(attachment.color);
			let label = line.bold();
			let (icon, size) = match &attachment.content {
				AttachmentContent::Image { dimensions, .. } => (
					pc.ctx.charset.icon(crate::Icon::Image),
					dimensions.map(|(width, height)| fmts!("{width}x{height}")),
				),
				AttachmentContent::Text { lines, chars, .. } => (
					pc.ctx.charset.icon(crate::Icon::TextFile),
					Some(if *lines > 1 {
						fmts!("+{lines} lines")
					} else {
						fmts!("{chars} chars")
					}),
				),
			};
			let name = fmts!("{icon} #{}", attachment.marker);
			frame_caption_row(pc, x, top, PREVIEW_BOX_COLS, (tl, tr, horizontal), &name, line, label);
			frame_caption_row(
				pc,
				x,
				bottom,
				PREVIEW_BOX_COLS,
				(bl, br, horizontal),
				size.as_deref().unwrap_or(""),
				line,
				label,
			);
			let rail = vertical.encode_utf8(&mut glyph);
			let frame_right = x.saturating_add(PREVIEW_BOX_COLS.saturating_sub(1));
			for row in top.saturating_add(1)..bottom {
				if row >= pc.clip {
					break;
				}
				pc.frame.put(x, row, rail, line);
				pc.frame.put(frame_right, row, rail, line);
			}
			match &attachment.content {
				AttachmentContent::Image { .. } => {
					if let Some(child) = self.children.get_mut(image_child) {
						if child.visible {
							child.paint(pc);
						}
						image_child += 1;
					}
				},
				AttachmentContent::Text { snippet, .. } => {
					for (offset, text) in snippet.as_str().split('\n').enumerate() {
						let y = top
							.saturating_add(1)
							.saturating_add(u16::try_from(offset).unwrap_or(u16::MAX));
						if y >= bottom || y >= pc.clip {
							break;
						}
						pc.frame.put(x.saturating_add(1), y, text, snippet_style);
					}
				},
			}
			x = x
				.saturating_add(PREVIEW_BOX_COLS)
				.saturating_add(PREVIEW_GAP);
		}
	}

	/// Sets one editor-shell property.
	pub fn with(mut self, prop: Prop, value: impl Into<PropValue>) -> Self {
		let value = value.into();
		if matches!(prop, Prop::Id | Prop::Value) {
			self.children[0]
				.comp_mut()
				.props_mut()
				.set(prop, value.clone());
			if prop == Prop::Value
				&& let PropValue::Str(text) = &value
			{
				self.children[0]
					.comp_mut()
					.set_text(&UiContext::default(), text.clone());
			}
		} else {
			self.props.set(prop, value);
		}
		self
	}

	/// Sets one editor-shell property from a string.
	pub fn with_str(self, prop: Prop, value: &str) -> Self {
		self.with(prop, value)
	}

	#[cfg(test)]
	pub(crate) fn buffer(&self) -> &EditBuffer {
		self.children[0]
			.comp()
			.downcast_ref::<EditInput>()
			.expect("default editor input was replaced")
			.buffer()
	}

	#[cfg(test)]
	pub(crate) fn buffer_mut(&mut self) -> &mut EditBuffer {
		self.children[0]
			.comp_mut()
			.downcast_mut::<EditInput>()
			.expect("default editor input was replaced")
			.buffer_mut()
	}
}

impl Default for EditorPane {
	fn default() -> Self {
		Self::new()
	}
}

impl Component for EditorPane {
	fn props(&self) -> &Props {
		&self.props
	}

	fn props_mut(&mut self) -> &mut Props {
		&mut self.props
	}

	fn slot(&self) -> Slot {
		self.slot
	}

	fn children(&self) -> &[Cached] {
		&self.children
	}

	fn children_mut(&mut self) -> &mut [Cached] {
		&mut self.children
	}

	fn ring(&self, out: &mut Vec<Slot>) {
		self.children[0].comp().ring(out);
	}

	fn measure(&mut self, ctx: &UiContext) -> (u16, u16) {
		self.sync_attachments();
		self.children[0].measure(ctx)
	}

	fn height(&mut self, ctx: &UiContext, width: u16) -> u16 {
		self.sync_attachments();
		let input = self.children[0].height(ctx, width);
		let status = u16::from(self.has_status && self.props.border().is_none());
		input
			.saturating_add(status)
			.saturating_add(self.band_rows())
	}

	fn place(&mut self, ctx: &UiContext, rect: Rect) {
		self.sync_attachments();
		let bordered = self.props.border().is_some();
		let band = self.band_rows();
		let status_height = u16::from(self.has_status && !bordered);
		let top = band.saturating_add(status_height);
		self.children[0].place(
			ctx,
			Rect::new(rect.x, rect.y.saturating_add(top), rect.width, rect.height.saturating_sub(top)),
		);
		if self.has_status {
			let (x, y, width) = if bordered {
				(rect.x.saturating_sub(1), rect.y.saturating_sub(1), rect.width.saturating_add(2))
			} else {
				(rect.x, rect.y.saturating_add(band), rect.width)
			};
			let status = &mut self.children[1];
			let _ = status.measure(ctx);
			let _ = status.height(ctx, width);
			status.place(ctx, Rect::new(x, y, width, 1));
		}
		self.band =
			Rect::new(rect.x, rect.y, rect.width, if band > 0 { PREVIEW_BOX_ROWS } else { 0 });
		let right = rect.x.saturating_add(rect.width);
		let handle = self.attachments.clone();
		let state = handle.state.borrow();
		let mut x = rect.x;
		let mut image_child = self.preview_start();
		for staged in state.staged.iter().filter(|staged| !staged.hidden) {
			let fits = x.saturating_add(PREVIEW_BOX_COLS) <= right;
			if matches!(staged.attachment.content, AttachmentContent::Image { .. })
				&& let Some(child) = self.children.get_mut(image_child)
			{
				image_child += 1;
				child.visible = fits;
				if fits {
					let _ = child.measure(ctx);
					let _ = child.height(ctx, PREVIEW_COLS);
					child.place(
						ctx,
						Rect::new(
							x.saturating_add(1),
							rect.y.saturating_add(1),
							PREVIEW_COLS,
							PREVIEW_ROWS,
						),
					);
				}
			}
			if fits {
				x = x
					.saturating_add(PREVIEW_BOX_COLS)
					.saturating_add(PREVIEW_GAP);
			}
		}
	}

	fn paint(&mut self, pc: &mut PaintCtx<'_>, rect: Rect) {
		if self.children[0].rect.width == 0 {
			self.place(pc.ctx, rect);
		}
		self.children[0].paint(pc);
		if self.has_status {
			let status = &mut self.children[1];
			if self.props.border().is_some() {
				// The status owns the top row, so remove the border before
				// painting its shaped band.
				pc.frame.fill(status.rect, Style::new());
			}
			status.paint(pc);
		}
		self.paint_previews(pc);
	}

	fn value(&self, out: &mut serde_json::Map<String, Value>) {
		self.children[0].comp().value(out);
	}

	fn set_text(&mut self, ctx: &UiContext, text: Str) -> bool {
		self.children[0].comp_mut().set_text(ctx, text)
	}
}

/// Paints one preview-frame edge: corners, rule fill, and an optional
/// centered caption set off by single spaces.
fn frame_caption_row(
	pc: &mut PaintCtx<'_>,
	x: u16,
	y: u16,
	width: u16,
	(left, right, horizontal): (char, char, char),
	caption: &str,
	line: Style,
	label: Style,
) {
	if y >= pc.clip || width < 2 {
		return;
	}
	let mut glyph = [0_u8; 4];
	let right_x = x.saturating_add(width.saturating_sub(1));
	let mut at = pc.frame.put(x, y, left.encode_utf8(&mut glyph), line);
	let caption_width = u16::try_from(xutf::width_str(caption)).unwrap_or(u16::MAX);
	if !caption.is_empty() && caption_width.saturating_add(2) <= width.saturating_sub(2) {
		let lead = (width.saturating_sub(2) - caption_width.saturating_add(2)) / 2;
		let caption_x = at.saturating_add(lead);
		for column in at..caption_x {
			pc.frame
				.put(column, y, horizontal.encode_utf8(&mut glyph), line);
		}
		at = pc.frame.put(caption_x, y, " ", line);
		at = pc.frame.put(at, y, caption, label);
		at = pc.frame.put(at, y, " ", line);
	}
	for column in at..right_x {
		pc.frame
			.put(column, y, horizontal.encode_utf8(&mut glyph), line);
	}
	pc.frame
		.put(right_x, y, right.encode_utf8(&mut glyph), line);
}

fn paint_xml_range(
	frame: &mut Frame,
	mut x: u16,
	y: u16,
	text: &str,
	runs: &[SyntaxRun],
	start: usize,
	end: usize,
	selection: Option<(usize, usize)>,
	selection_color: Color,
) -> u16 {
	for run in runs {
		let from = run.start.max(start);
		let to = run.end.min(end);
		if from >= to {
			continue;
		}
		let Some((selection_start, selection_end)) = selection else {
			x = frame.put(x, y, &text[from..to], run.style);
			continue;
		};
		let selected_from = from.max(selection_start);
		let selected_to = to.min(selection_end);
		if selected_from >= selected_to {
			x = frame.put(x, y, &text[from..to], run.style);
			continue;
		}
		if from < selected_from {
			x = frame.put(x, y, &text[from..selected_from], run.style);
		}
		x = frame.put(x, y, &text[selected_from..selected_to], run.style.bg(selection_color));
		if selected_to < to {
			x = frame.put(x, y, &text[selected_to..to], run.style);
		}
	}
	x
}

fn paint_xml_runs(
	frame: &mut Frame,
	x: u16,
	y: u16,
	text: &str,
	runs: &[SyntaxRun],
	selection: Option<(usize, usize)>,
	selection_color: Color,
	cursor: Option<usize>,
	cursor_style: Style,
) {
	let Some(cursor) = cursor else {
		paint_xml_range(frame, x, y, text, runs, 0, text.len(), selection, selection_color);
		return;
	};
	let mut x = paint_xml_range(frame, x, y, text, runs, 0, cursor, selection, selection_color);
	if cursor == text.len() {
		frame.put(x, y, " ", cursor_style);
		return;
	}
	let under = text[cursor..].graphemes().next().unwrap_or(" ");
	x = frame.put(x, y, under, cursor_style);
	paint_xml_range(
		frame,
		x,
		y,
		text,
		runs,
		cursor + under.len(),
		text.len(),
		selection,
		selection_color,
	);
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{
		Color, Ui,
		components::{Input, Segment, Status},
		context::{Charset, UiContext},
		frame::{Frame, Size},
		markup::Border,
		test_support::frame_row_text,
	};
	fn temp_drop_file(test: &str, name: &str, bytes: &[u8]) -> std::path::PathBuf {
		let dir = std::env::temp_dir().join(format!("omp-editor-drop-{test}-{}", std::process::id()));
		std::fs::create_dir_all(&dir).unwrap();
		let path = dir.join(name);
		std::fs::write(&path, bytes).unwrap();
		path
	}

	fn editor_pane(ui: &Ui) -> &EditorPane {
		ui.root()
			.comp()
			.downcast_ref::<EditorPane>()
			.expect("UI root is an editor pane")
	}

	#[test]
	fn selection_paint_replaces_only_the_glyph_background() {
		let mut frame = Frame::new(Size::new(3, 1));
		let foreground = Color::Rgb(0x11, 0x22, 0x33);
		let selection = Color::Rgb(0x44, 0x55, 0x66);
		let style = Style::new().fg(foreground).bold();
		let runs = [SyntaxRun { start: 0, end: 3, style }];
		paint_xml_runs(&mut frame, 0, 0, "abc", &runs, Some((1, 2)), selection, None, Style::new());

		assert_eq!(frame.cell(0, 0).style(), style);
		assert_eq!(frame.cell(1, 0).style(), style.bg(selection));
		assert_eq!(frame.cell(2, 0).style(), style);
	}

	#[test]
	fn editor_mouse_drag_and_double_click_select_text() {
		let mut ui = Ui::from_root(
			EditInput::new()
				.with(Prop::Id, "composer")
				.with(Prop::Value, "hello world"),
			40,
			UiContext::default(),
		);
		ui.focus_first();
		let hit = ui
			.hits()
			.iter()
			.find(|hit| hit.tag == HitTag::Press)
			.copied()
			.expect("editor press target");

		ui.handle_mouse(hit.rect.x + 2, hit.rect.y, Mouse::Click);
		ui.handle_mouse(hit.rect.x + 7, hit.rect.y, Mouse::Drag);
		ui.handle_mouse(hit.rect.x + 7, hit.rect.y, Mouse::Release);
		let selected = ui
			.root()
			.comp()
			.downcast_ref::<EditInput>()
			.expect("editor input")
			.buffer()
			.selected_text();
		assert_eq!(selected, Some("hello"));

		ui.handle_mouse(hit.rect.x + 8, hit.rect.y, Mouse::Click);
		ui.handle_mouse(hit.rect.x + 8, hit.rect.y, Mouse::Click);
		let selected = ui
			.root()
			.comp()
			.downcast_ref::<EditInput>()
			.expect("editor input")
			.buffer()
			.selected_text();
		assert_eq!(selected, Some("world"));
	}

	struct GrowingInput {
		props: Props,
		slot:  Slot,
		rows:  u16,
	}

	impl GrowingInput {
		fn new() -> Self {
			Self { props: Props::new(), slot: next_slot(), rows: 1 }
		}
	}

	impl Component for GrowingInput {
		fn props(&self) -> &Props {
			&self.props
		}

		fn props_mut(&mut self) -> &mut Props {
			&mut self.props
		}

		fn slot(&self) -> Slot {
			self.slot
		}

		fn measure(&mut self, _ctx: &UiContext) -> (u16, u16) {
			(1, 8)
		}

		fn height(&mut self, _ctx: &UiContext, _width: u16) -> u16 {
			self.rows
		}

		fn paint(&mut self, pc: &mut PaintCtx<'_>, rect: Rect) {
			pc.frame
				.set_cursor(rect.x, rect.y.saturating_add(self.rows.saturating_sub(1)));
		}

		fn focusable(&self) -> bool {
			true
		}

		fn key(&mut self, _ec: &mut EventCtx<'_>, key: Key) -> Flow {
			if key == Key::ShiftEnter {
				self.rows = self.rows.saturating_add(1);
				Flow::Consumed
			} else {
				Flow::Skip
			}
		}
	}

	#[test]
	fn ui_routes_multiline_growth_to_the_editor_input_cache() {
		let mut ui =
			Ui::from_root(EditorPane::new().input(GrowingInput::new()), 14, UiContext::default());
		let initial_height = ui.height();
		ui.handle_key(Key::ShiftEnter);
		ui.handle_key(Key::ShiftEnter);

		assert_eq!(ui.height(), initial_height.saturating_add(2));
		assert_eq!(ui.frame().size().height, ui.height());
		let (cursor_x, cursor_y) = ui.frame().cursor().expect("focused editor cursor");
		assert!(cursor_x < ui.frame().size().width);
		assert!(cursor_y < ui.frame().size().height);
	}

	#[test]
	fn editor_status_replaces_top_border_with_rounded_band() {
		let ctx = UiContext { charset: Charset::NerdFont, ..UiContext::default() };
		let mut editor = Cached::new(Box::new(
			EditorPane::new().with(Prop::Border, Border::Round).status(
				Status::new()
					.with(Prop::Bg, "yellow")
					.segment(Segment::new().label("ready")),
			),
		));
		let height = editor.height(&ctx, 20);
		editor.place(&ctx, Rect::new(0, 0, 20, height));
		let mut frame = Frame::new(Size::new(20, height));
		let mut hits = Vec::new();
		editor.paint(&mut PaintCtx::new(&mut frame, &ctx, &mut hits, &mut Vec::new()));

		assert_eq!(frame_row_text(&frame, 0), "\u{e0b6} ready \u{e0b0}");
		assert_eq!(frame.cell(0, 0).style.foreground_color(), Color::Rgb(255, 255, 0));
		assert_eq!(frame.cell(0, 0).style.background_color(), Color::Default);
		assert_eq!(frame.cell(9, 0).style.background_color(), Color::Default);
	}

	#[test]
	fn unbordered_editor_status_reserves_a_borderless_header_row() {
		let ctx = UiContext { charset: Charset::NerdFont, ..UiContext::default() };
		let mut editor = Cached::new(Box::new(
			EditorPane::new().with(Prop::Value, "body").status(
				Status::new()
					.with(Prop::Bg, "yellow")
					.segment(Segment::new().label("ready")),
			),
		));
		let height = editor.height(&ctx, 20);
		editor.place(&ctx, Rect::new(0, 0, 20, height));
		let mut frame = Frame::new(Size::new(20, height));
		let mut hits = Vec::new();
		editor.paint(&mut PaintCtx::new(&mut frame, &ctx, &mut hits, &mut Vec::new()));

		assert_eq!(frame_row_text(&frame, 0), "\u{e0b6} ready \u{e0b0}");
		assert!(frame_row_text(&frame, 1).contains("body"));
		for row in 0..height {
			let text = frame_row_text(&frame, row);
			assert!(
				!text
					.chars()
					.any(|glyph| matches!(glyph, '╭' | '╮' | '╰' | '╯' | '│' | '─')),
				"unexpected editor border on row {row}: {text}",
			);
		}
	}

	#[test]
	fn editor_status_is_excluded_from_the_focus_ring() {
		let editor = EditorPane::new().status(Input::new());
		let mut ring = Vec::new();
		editor.ring(&mut ring);
		assert_eq!(ring, vec![editor.children[0].comp().slot()]);
	}

	#[test]
	fn attachments_render_framed_previews_with_markers_and_resolution() {
		let dir = std::env::temp_dir().join(format!("omp-editor-attach-{}", std::process::id()));
		std::fs::create_dir_all(&dir).unwrap();
		let probed = dir.join("shot.png");
		let mut png = b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR".to_vec();
		png.extend(528_u32.to_be_bytes());
		png.extend(200_u32.to_be_bytes());
		std::fs::write(&probed, png).unwrap();

		let ctx = UiContext::default();
		let pane = EditorPane::new()
			.with(Prop::Value, "body")
			.status(Status::new().segment(Segment::new().label("ready")));
		let attachments = pane.attachments();
		let mut editor = Cached::new(Box::new(pane));
		let base = editor.height(&ctx, 40);

		let first = attachments.push_image(probed.to_str().expect("temp path is UTF-8"));
		assert_eq!(first.marker, 1);
		assert!(
			matches!(first.content, AttachmentContent::Image { dimensions: Some((528, 200)), .. }),
			"PNG header probes its resolution"
		);
		assert_eq!(first.color, attachment_color(1));
		assert_eq!(attachments.push_image("/nope/b.png").marker, 2);
		editor.invalidate();
		let height = editor.height(&ctx, 40);
		assert_eq!(
			height,
			base + PREVIEW_BOX_ROWS + 1,
			"band adds the framed previews plus the spacer row"
		);
		editor.place(&ctx, Rect::new(0, 0, 40, height));
		let mut frame = Frame::new(Size::new(40, height));
		let mut hits = Vec::new();
		editor.paint(&mut PaintCtx::new(&mut frame, &ctx, &mut hits, &mut Vec::new()));

		let top = frame_row_text(&frame, 0);
		assert!(top.contains("#1"), "first frame caption missing: {top}");
		assert!(top.contains("#2"), "second frame caption missing: {top}");
		let bottom = frame_row_text(&frame, PREVIEW_BOX_ROWS - 1);
		assert!(bottom.contains("528x200"), "resolution caption missing: {bottom}");
		assert_eq!(frame.cell(0, 0).style.foreground_color(), attachment_color(1));
		assert_eq!(
			frame
				.cell(PREVIEW_BOX_COLS + PREVIEW_GAP, 0)
				.style
				.foreground_color(),
			attachment_color(2),
			"each frame is tinted with its own identity color"
		);
		assert_eq!(
			frame_row_text(&frame, PREVIEW_BOX_ROWS).trim(),
			"",
			"a spacer row separates the band from the status line"
		);
		assert!(frame_row_text(&frame, PREVIEW_BOX_ROWS + 1).contains("ready"));
		assert!(frame_row_text(&frame, PREVIEW_BOX_ROWS + 2).contains("body"));

		assert_eq!(attachments.take().len(), 2);
		editor.invalidate();
		assert_eq!(editor.height(&ctx, 40), base, "taking attachments collapses the band");
		std::fs::remove_dir_all(&dir).ok();
	}

	#[test]
	fn attachment_previews_hide_when_the_composer_is_too_narrow() {
		let ctx = UiContext::default();
		let pane = EditorPane::new();
		let attachments = pane.attachments();
		attachments.push_image("/nope/a.png");
		attachments.push_image("/nope/b.png");
		let mut editor = Cached::new(Box::new(pane));
		let height = editor.height(&ctx, 20);
		editor.place(&ctx, Rect::new(0, 0, 20, height));
		let mut frame = Frame::new(Size::new(20, height));
		let mut hits = Vec::new();
		editor.paint(&mut PaintCtx::new(&mut frame, &ctx, &mut hits, &mut Vec::new()));

		let captions = frame_row_text(&frame, 0);
		assert!(captions.contains("#1"), "caption row: {captions}");
		assert!(!captions.contains("#2"), "overflowing preview must stay hidden: {captions}");
	}

	#[test]
	fn paste_cards_preview_leading_text_with_size_caption() {
		let ctx = UiContext::default();
		let pane = EditorPane::new();
		let attachments = pane.attachments();
		let paste = (0..12)
			.map(|n| format!("line{n}"))
			.collect::<Vec<_>>()
			.join("\n");
		let card = attachments.push_text(&paste);
		assert_eq!(card.marker, 1);
		assert!(matches!(card.content, AttachmentContent::Text { lines: 12, .. }));

		let mut editor = Cached::new(Box::new(pane));
		let height = editor.height(&ctx, 40);
		editor.place(&ctx, Rect::new(0, 0, 40, height));
		let mut frame = Frame::new(Size::new(40, height));
		let mut hits = Vec::new();
		editor.paint(&mut PaintCtx::new(&mut frame, &ctx, &mut hits, &mut Vec::new()));

		assert!(frame_row_text(&frame, 0).contains("#1"));
		assert!(frame_row_text(&frame, 1).contains("line0"), "card previews the paste text");
		assert!(
			frame_row_text(&frame, PREVIEW_BOX_ROWS - 1).contains("+12 lines"),
			"bottom edge captions the paste size"
		);
	}

	#[test]
	fn quoted_image_path_drop_stages_a_reference_chip() {
		let path = temp_drop_file("quoted", "drop test.png", b"\x89PNG\r\n\x1a\n");
		let normalized = path.to_str().expect("temp path is UTF-8");
		let pasted = format!("'{normalized}'");
		let pane = EditorPane::new().with(Prop::Id, "composer");
		let attachments = pane.attachments();
		let mut ui = Ui::from_root(pane, 60, UiContext::default());
		ui.focus_first();

		ui.handle_paste(&pasted);

		assert_eq!(attachments.len(), 1);
		let visible = editor_pane(&ui).buffer().text();
		assert!(visible.contains("#1"));
		assert!(!visible.contains(&pasted));
		assert!(visible.ends_with(' '));
		assert_eq!(editor_pane(&ui).buffer().expanded_text(), format!("{normalized} "));
		std::fs::remove_dir_all(path.parent().unwrap()).ok();
	}

	#[test]
	fn file_url_image_drop_stages_its_normalized_path() {
		let path = temp_drop_file("file-url", "url drop.png", b"\x89PNG\r\n\x1a\n");
		let normalized = path.to_str().expect("temp path is UTF-8");
		let pasted = format!("file://{}", normalized.replace(' ', "%20"));
		let pane = EditorPane::new().with(Prop::Id, "composer");
		let attachments = pane.attachments();
		let mut ui = Ui::from_root(pane, 60, UiContext::default());
		ui.focus_first();

		ui.handle_paste(&pasted);

		assert_eq!(attachments.len(), 1);
		assert_eq!(editor_pane(&ui).buffer().expanded_text(), format!("{normalized} "));
		std::fs::remove_dir_all(path.parent().unwrap()).ok();
	}

	#[test]
	fn escaped_image_path_drop_stages_chips_in_order() {
		let first = temp_drop_file("escaped", "drop one.png", b"\x89PNG\r\n\x1a\n");
		let second = temp_drop_file("escaped", "drop two.gif", b"GIF89a");
		let first_text = first.to_str().expect("temp path is UTF-8");
		let second_text = second.to_str().expect("temp path is UTF-8");
		let pasted =
			format!("{} {}", first_text.replace(' ', "\\ "), second_text.replace(' ', "\\ "));
		let pane = EditorPane::new().with(Prop::Id, "composer");
		let attachments = pane.attachments();
		let mut ui = Ui::from_root(pane, 60, UiContext::default());
		ui.focus_first();

		ui.handle_paste(&pasted);

		assert_eq!(attachments.len(), 2);
		let visible = editor_pane(&ui).buffer().text();
		assert!(visible.find("#1").unwrap() < visible.find("#2").unwrap());
		assert_eq!(editor_pane(&ui).buffer().expanded_text(), format!("{first_text} {second_text} "));
		std::fs::remove_dir_all(first.parent().unwrap()).ok();
	}

	#[test]
	fn missing_image_path_drop_remains_plain_text() {
		let path = std::env::temp_dir()
			.join(format!("omp-editor-drop-missing-{}", std::process::id()))
			.join("missing image.png");
		std::fs::remove_file(&path).ok();
		let pasted = format!("'{}'", path.to_str().expect("temp path is UTF-8"));
		let pane = EditorPane::new().with(Prop::Id, "composer");
		let attachments = pane.attachments();
		let mut ui = Ui::from_root(pane, 60, UiContext::default());
		ui.focus_first();

		ui.handle_paste(&pasted);

		assert!(attachments.is_empty());
		assert_eq!(editor_pane(&ui).buffer().text(), pasted);
	}

	#[test]
	fn existing_non_image_path_drop_remains_plain_text() {
		let path = temp_drop_file("non-image", "notes.txt", b"not an image");
		let pasted = path.to_str().expect("temp path is UTF-8");
		let pane = EditorPane::new().with(Prop::Id, "composer");
		let attachments = pane.attachments();
		let mut ui = Ui::from_root(pane, 60, UiContext::default());
		ui.focus_first();

		ui.handle_paste(pasted);

		assert!(attachments.is_empty());
		assert_eq!(editor_pane(&ui).buffer().text(), pasted);
		std::fs::remove_dir_all(path.parent().unwrap()).ok();
	}

	#[test]
	fn image_path_drop_without_attachment_binding_remains_plain_text() {
		let path = temp_drop_file("unbound", "drop test.png", b"\x89PNG\r\n\x1a\n");
		let pasted = format!("'{}'", path.to_str().expect("temp path is UTF-8"));
		let mut ui =
			Ui::from_root(EditInput::new().with(Prop::Id, "composer"), 60, UiContext::default());
		ui.focus_first();

		ui.handle_paste(&pasted);

		let input = ui
			.root()
			.comp()
			.downcast_ref::<EditInput>()
			.expect("UI root is an editor input");
		assert_eq!(input.buffer().text(), pasted);
		std::fs::remove_dir_all(path.parent().unwrap()).ok();
	}

	#[test]
	fn plain_path_paste_separates_from_a_preceding_word_only() {
		let mut after_word = Ui::from_root(
			EditInput::new()
				.with(Prop::Id, "composer")
				.with(Prop::Value, "word"),
			40,
			UiContext::default(),
		);
		after_word.focus_first();
		after_word.handle_paste("/tmp");
		assert_eq!(after_word.values()["composer"], Value::String("word /tmp".to_owned()));

		let mut after_space = Ui::from_root(
			EditInput::new()
				.with(Prop::Id, "composer")
				.with(Prop::Value, "word "),
			40,
			UiContext::default(),
		);
		after_space.focus_first();
		after_space.handle_paste("/tmp");
		assert_eq!(after_space.values()["composer"], Value::String("word /tmp".to_owned()));
	}

	#[test]
	fn hidden_attachments_keep_markers_but_never_reach_take() {
		let attachments = Attachments::new();
		attachments.push_image("/nope/a.png");
		attachments.push_image("/nope/b.png");
		assert!(attachments.set_visible(|attachment| attachment.marker != 1));
		assert_eq!(attachments.len(), 1, "hiding drops the visible count");
		assert_eq!(attachments.push_image("/nope/c.png").marker, 3, "markers stay stable");

		// Undo made the first reference reappear.
		assert!(attachments.set_visible(|_| true));
		assert_eq!(attachments.len(), 3);

		// Deleted again; the drain must never hand it to the host.
		assert!(attachments.set_visible(|attachment| attachment.marker != 1));
		let taken = attachments.take();
		assert_eq!(
			taken.iter().map(|a| a.marker).collect::<Vec<_>>(),
			vec![2, 3],
			"take returns only visible attachments"
		);
		assert!(attachments.is_empty());
		assert_eq!(attachments.push_image("/nope/d.png").marker, 1, "numbering restarts");
	}

	#[test]
	fn default_editor_collapses_large_pastes_into_atomic_chip_cards() {
		let mut ui = Ui::from_root(
			EditorPane::new()
				.with(Prop::Id, "composer")
				.status(Status::new().segment(Segment::new().label("ready"))),
			40,
			UiContext::default(),
		);
		ui.focus_first();
		let base = ui.height();
		let paste = (0..12)
			.map(|n| format!("line{n}"))
			.collect::<Vec<_>>()
			.join("\n");
		ui.handle_paste(&paste);
		assert_eq!(
			ui.height(),
			base + PREVIEW_BOX_ROWS + 1,
			"a routed paste grows the pane's band without a manual relayout"
		);
		assert!(frame_row_text(ui.frame(), 0).contains("#1"));
		assert!(frame_row_text(ui.frame(), PREVIEW_BOX_ROWS - 1).contains("+12 lines"));

		// The chip paints in its identity color inside the input row.
		let input_row = PREVIEW_BOX_ROWS + 2;
		let text = frame_row_text(ui.frame(), input_row);
		let hash = text.find('#').expect("chip in the input row");
		let column = u16::try_from(xutf::width_str(&text[..hash])).expect("narrow row");
		assert_eq!(ui.frame().cell(column, input_row).style.foreground_color(), attachment_color(1));

		// Backspace over the trailing space, then the chip: one atomic unit
		// whose removal collapses the band through the same event path.
		ui.handle_key(Key::Backspace);
		ui.handle_key(Key::Backspace);
		assert_eq!(ui.height(), base, "deleting the chip collapses the band");
		assert_eq!(
			ui.values().get("composer").and_then(Value::as_str),
			Some(""),
			"a deleted paste never reaches the submitted value"
		);

		// Undo restores the chip, its card, and the expanded payload.
		ui.handle_key(Key::Ctrl('_'));
		assert_eq!(ui.height(), base + PREVIEW_BOX_ROWS + 1, "undo restores the band");
		let values = ui.values();
		assert_eq!(
			values
				.get("composer")
				.and_then(Value::as_str)
				.map(|value| value.trim_end().to_owned()),
			Some(paste),
			"the restored chip expands back to the pasted text"
		);
	}

	#[test]
	fn raw_paste_bypasses_chips_and_drop_classification() {
		let dir = std::env::temp_dir().join(format!("omp-tui-raw-paste-{}", std::process::id()));
		std::fs::create_dir_all(&dir).unwrap();
		let path = dir.join("raw.png");
		std::fs::write(&path, b"\x89PNG\r\n\x1a\n\0\0\0\0").unwrap();

		let mut ui = Ui::from_root(
			EditorPane::new()
				.with(Prop::Id, "composer")
				.status(Status::new().segment(Segment::new().label("ready"))),
			40,
			UiContext::default(),
		);
		ui.focus_first();
		let base = ui.height();

		// An existing image path inserts as text instead of staging an
		// attachment (Ctrl+Shift+V contract: verbatim insertion).
		let path_text = path.to_str().expect("temp path is UTF-8").to_owned();
		ui.handle_paste_raw(&path_text);
		assert_eq!(ui.height(), base, "no attachment band appears");
		assert_eq!(
			ui.values().get("composer").and_then(Value::as_str),
			Some(path_text.as_str()),
			"the path stays inline text"
		);

		// Large text stays inline and editable instead of collapsing.
		let mut ui = Ui::from_root(
			EditorPane::new()
				.with(Prop::Id, "composer")
				.status(Status::new().segment(Segment::new().label("ready"))),
			40,
			UiContext::default(),
		);
		ui.focus_first();
		let base = ui.height();
		let big = (0..12)
			.map(|n| format!("line{n}"))
			.collect::<Vec<_>>()
			.join("\n");
		ui.handle_paste_raw(&big);
		assert_eq!(ui.height(), base, "no chip card band appears");
		assert_eq!(
			ui.values().get("composer").and_then(Value::as_str),
			Some(big.as_str()),
			"the full text stays inline"
		);
		std::fs::remove_dir_all(&dir).ok();
	}
}
