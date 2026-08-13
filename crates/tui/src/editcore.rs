//! Pi-compatible editing core: the flat-text [`EditBuffer`] (grapheme-safe
//! word wrapping and navigation, undo, kill-ring yank/yank-pop, atomic
//! references, character jumps, sticky page motion) and the
//! [`Editor`] built on top of it (pluggable completion, inline ghost
//! hints, emoji expansion, prompt history).

use std::{cell::Cell, cmp::Reverse, collections::HashMap, ops::Range, sync::LazyLock};

use omp_core::{Str, fmts, str::IntoStr};
use smallvec::SmallVec;
use xutf::Text;

use crate::{
	input::{Key, sanitize_paste},
	rich::cell_width,
};

const KILL_CAP: usize = 60;
const UNDO_CAP: usize = 100;
const PICKER_ROWS: usize = 5;
const MAX_INPUT_ROWS: usize = 8;
const MAX_EMOJI_SUGGESTIONS: usize = 12;
const HISTORY_CAPACITY: usize = 100;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Whether an editing command changed the buffer.
pub enum BufferOutcome {
	/// Text, cursor, or transient editing state changed.
	Changed,
	/// The key had no applicable effect.
	Ignored,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Action {
	Kill,
	Yank,
	YankPop,
	TypeWord,
	Other,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Jump {
	Forward,
	Backward,
}

/// One atomic unit in the visible text: the `start..end` marker range is
/// displayed, navigated, and deleted as a whole, and `payload` replaces it
/// in the submitted text. Ranges are maintained through every edit, so
/// text that merely looks like a marker is never treated as one.
#[derive(Clone, Debug)]
struct Atom {
	start:   usize,
	end:     usize,
	payload: Str,
}

#[derive(Clone, Copy, Debug)]
/// One visual word-wrapped row borrowed from an [`EditBuffer`].
pub struct VisualRow<'a> {
	/// UTF-8 byte start in the complete buffer.
	pub start:         usize,
	/// UTF-8 byte end in the complete buffer.
	pub end:           usize,
	/// Grapheme-aligned text belonging to the row.
	pub text:          &'a str,
	/// Cursor cell column when this row owns the cursor.
	pub cursor_column: Option<u16>,
}

#[derive(Clone, Copy, Debug)]
struct Segment {
	start: usize,
	end:   usize,
	last:  bool,
}

/// Shared Pi-style flat text editing model used by the widget and chat editors.
#[derive(Clone, Debug)]
pub struct EditBuffer {
	text:          String,
	cursor:        usize,
	anchor:        Option<usize>,
	copied:        Option<Str>,
	desired:       Option<u16>,
	kill_ring:     Vec<String>,
	kill_index:    usize,
	last_yank:     Option<(usize, usize)>,
	last_action:   Action,
	undo:          Vec<(String, usize, Vec<Atom>)>,
	atoms:         Vec<Atom>,
	jump:          Option<Jump>,
	layout_width:  u16,
	xml:           bool,
	view_offset:   Cell<usize>,
	manual_scroll: Cell<bool>,
}

impl Default for EditBuffer {
	fn default() -> Self {
		Self::new("")
	}
}

impl EditBuffer {
	#[must_use]
	/// Creates a buffer with the cursor at the end of sanitized `text`.
	pub fn new(text: &str) -> Self {
		let text = sanitize_paste(text);
		let cursor = text.len();
		Self {
			text,
			cursor,
			anchor: None,
			copied: None,
			desired: None,
			kill_ring: Vec::new(),
			kill_index: 0,
			last_yank: None,
			last_action: Action::Other,
			undo: Vec::new(),
			atoms: Vec::new(),
			jump: None,
			layout_width: 80,
			view_offset: Cell::new(0),
			manual_scroll: Cell::new(false),
			xml: true,
		}
	}

	/// Enables `</` close-tag completion (on by default).
	pub const fn set_xml(&mut self, xml: bool) {
		self.xml = xml;
	}

	#[must_use]
	/// Returns the visible marker text.
	pub fn text(&self) -> &str {
		&self.text
	}

	#[must_use]
	/// Returns the UTF-8 byte cursor.
	pub const fn cursor(&self) -> usize {
		self.cursor
	}

	#[must_use]
	/// Returns the normalized selected UTF-8 byte range, or `None` when
	/// collapsed.
	pub fn selection(&self) -> Option<Range<usize>> {
		let anchor = self.anchor?;
		if anchor == self.cursor {
			return None;
		}
		let (start, end) = if anchor < self.cursor {
			(anchor, self.cursor)
		} else {
			(self.cursor, anchor)
		};
		let (start, end) = self.expand_to_atoms(start, end);
		Some(start..end)
	}

	#[must_use]
	/// Returns the selected visible text.
	pub fn selected_text(&self) -> Option<&str> {
		self.selection().map(|range| &self.text[range])
	}

	/// Selects the complete buffer.
	pub const fn select_all(&mut self) {
		self.anchor = Some(0);
		self.cursor = self.text.len();
		self.desired = None;
		self.break_sequence();
	}

	/// Collapses the active selection at its cursor edge.
	pub const fn clear_selection(&mut self) {
		self.anchor = None;
	}

	/// Takes the text captured by the last `Copy`/`Cut`, handing the
	/// clipboard write to the host (OSC 52 on terminals, a detached
	/// native write on the GPU host).
	pub const fn take_copied(&mut self) -> Option<Str> {
		self.copied.take()
	}

	#[must_use]
	/// Returns the selected display-column span intersecting `row`.
	pub fn selection_span(&self, row: &VisualRow<'_>) -> Option<(u16, u16)> {
		let selection = self.selection()?;
		let start = selection.start.max(row.start);
		let end = selection.end.min(row.end);
		if start >= end {
			return None;
		}
		Some((cell_width(&row.text[..start - row.start]), cell_width(&row.text[..end - row.start])))
	}

	#[must_use]
	/// Returns the number of logical newline-delimited lines.
	pub fn line_count(&self) -> usize {
		self.text.bytes().filter(|byte| *byte == b'\n').count() + 1
	}

	#[must_use]
	/// Returns the zero-based logical cursor line.
	pub fn cursor_line(&self) -> usize {
		self.text[..self.cursor]
			.bytes()
			.filter(|byte| *byte == b'\n')
			.count()
	}

	#[must_use]
	/// Returns the cursor's cell column within its logical line.
	pub fn cursor_column(&self) -> u16 {
		let (start, _) = self.line_bounds();
		cell_width(&self.text[start..self.cursor])
	}

	/// Iterates logical lines without allocating.
	pub fn logical_lines(
		&self,
	) -> impl DoubleEndedIterator<Item = &str> + Clone + std::iter::FusedIterator + '_ {
		self.text.split('\n')
	}

	/// Replaces text without creating an undo entry, for history browsing.
	pub fn replace_external(&mut self, text: &str, cursor_at_start: bool) {
		self.text = sanitize_paste(text);
		self.atoms.clear();
		self.cursor = if cursor_at_start { 0 } else { self.text.len() };
		self.undo.clear();
		self.anchor = None;
		self.desired = None;
		self.break_sequence();
	}

	/// Places the cursor on a logical line and cell column.
	pub fn set_cursor_line_column(&mut self, line: usize, column: u16) {
		let mut start = 0;
		for _ in 0..line {
			let Some(offset) = self.text[start..].find('\n') else {
				break;
			};
			start += offset + 1;
		}
		let end = self.text[start..]
			.find('\n')
			.map_or(self.text.len(), |offset| start + offset);
		let at = start + byte_at_column(&self.text[start..end], column);
		self.cursor = self.snap_position(at, at >= self.cursor);
		self.anchor = None;
		self.desired = None;
		self.break_sequence();
	}

	/// Places the cursor on a visible visual row and cell column.
	pub fn set_cursor_visual_row(&mut self, row: usize, column: u16, width_limit: u16) {
		let at = self.visual_position(row, column, width_limit);
		self.cursor = self.snap_position(at, at >= self.cursor);
		self.anchor = None;
		self.desired = None;
		self.break_sequence();
	}

	/// Extends the selection to a visible visual row and cell column.
	pub fn extend_selection_visual_row(&mut self, row: usize, column: u16, width_limit: u16) {
		let anchor = *self.anchor.get_or_insert(self.cursor);
		let at = self.visual_position(row, column, width_limit);
		self.cursor = self.snap_position(at, at >= anchor);
		self.desired = None;
		self.break_sequence();
	}

	/// Selects the coarse word around a position on a visible visual row.
	pub fn select_word_visual_row(&mut self, row: usize, column: u16, width_limit: u16) {
		let at = self.visual_position(row, column, width_limit);
		let (seed_start, seed_end) = if let Some(grapheme) = self.text[at..].graphemes().next() {
			(at, at + grapheme.len())
		} else if let Some((start, grapheme)) = self.text[..at].grapheme_indices().next_back() {
			(start, start + grapheme.len())
		} else {
			self.cursor = at;
			self.anchor = None;
			return;
		};
		let (start, end) =
			self.expand_to_atoms(word_left(&self.text, seed_end), word_right(&self.text, seed_start));
		self.anchor = Some(start);
		self.cursor = end;
		self.desired = None;
		self.break_sequence();
	}

	/// Replaces a byte range as one undoable edit. A non-empty range that
	/// touches an atomic marker widens to the whole marker, so partial
	/// replacements can never tear a unit apart.
	pub fn replace_range(&mut self, range: std::ops::Range<usize>, replacement: &str) {
		self.snapshot();
		let (start, end) = if range.is_empty() {
			(range.start, range.end)
		} else {
			self.expand_to_atoms(range.start, range.end)
		};
		self.cursor = start + replacement.len();
		self.splice(start..end, replacement);
		self.anchor = None;
		self.desired = None;
		self.break_sequence();
	}

	/// Widens `start..end` to whole-atom bounds for every atom it touches.
	fn expand_to_atoms(&self, mut start: usize, mut end: usize) -> (usize, usize) {
		for atom in &self.atoms {
			if start < atom.end && end > atom.start {
				start = start.min(atom.start);
				end = end.max(atom.end);
			}
		}
		(start, end)
	}

	/// Replaces `range` with `replacement`, shifting the atoms behind the
	/// edit and dropping any atom the edit tears through.
	fn splice(&mut self, range: std::ops::Range<usize>, replacement: &str) {
		let inserted = replacement.len();
		self.atoms.retain_mut(|atom| {
			if atom.end <= range.start {
				return true;
			}
			if atom.start >= range.end {
				atom.start = atom.start - range.len() + inserted;
				atom.end = atom.end - range.len() + inserted;
				return true;
			}
			false
		});
		self.text.replace_range(range, replacement);
	}

	/// Inserts sanitized text at the cursor.
	///
	/// Text is inserted verbatim; hosts that collapse large pastes into
	/// compact chips stage them through [`EditBuffer::insert_reference`].
	pub fn insert_text(&mut self, text: &str) -> BufferOutcome {
		let sanitized = sanitize_paste(text);
		if sanitized.is_empty() {
			return BufferOutcome::Ignored;
		}
		self.snapshot();
		self.break_sequence();
		let range = self.selection().unwrap_or(self.cursor..self.cursor);
		let start = range.start;
		self.splice(range, &sanitized);
		self.cursor = start + sanitized.len();
		self.anchor = None;
		self.desired = None;
		BufferOutcome::Changed
	}

	#[must_use]
	/// Returns text with every atomic reference expanded to its payload.
	pub fn expanded_text(&self) -> String {
		let mut atoms: SmallVec<&Atom, 4> = self.atoms.iter().collect();
		atoms.sort_unstable_by_key(|atom| atom.start);
		let mut result = String::with_capacity(self.text.len());
		let mut at = 0;
		for atom in atoms {
			result.push_str(&self.text[at..atom.start]);
			result.push_str(&atom.payload);
			at = atom.end;
		}
		result.push_str(&self.text[at..]);
		result
	}

	/// Inserts an atomic reference at the cursor: `marker` is displayed,
	/// navigated, and deleted as one unit, and expands to `payload` in the
	/// submitted text. `marker` must be a single line.
	///
	/// The unit is tracked by position, not by content: typed text that
	/// happens to equal `marker` stays ordinary text.
	pub fn insert_reference(&mut self, marker: &str, payload: &str) -> BufferOutcome {
		if marker.is_empty() || marker.contains('\n') {
			return BufferOutcome::Ignored;
		}
		self.snapshot();
		self.break_sequence();
		let range = self.selection().unwrap_or(self.cursor..self.cursor);
		let start = range.start;
		self.splice(range, marker);
		self.cursor = start + marker.len();
		self.anchor = None;
		self
			.atoms
			.push(Atom { start, end: start + marker.len(), payload: Str::new(payload) });
		self.desired = None;
		BufferOutcome::Changed
	}

	/// Byte ranges of atomic markers present in the text, ascending. Slice
	/// [`EditBuffer::text`] with a range to recover the marker.
	pub fn atom_ranges(&self) -> SmallVec<(usize, usize), 4> {
		let mut ranges: SmallVec<(usize, usize), 4> = self
			.atoms
			.iter()
			.map(|atom| (atom.start, atom.end))
			.collect();
		ranges.sort_unstable();
		ranges
	}

	/// Returns expanded text and resets the buffer after submission.
	pub fn clear_after_submit(&mut self) -> String {
		let result = self.expanded_text();
		self.text.clear();
		self.cursor = 0;
		self.anchor = None;
		self.desired = None;
		self.undo.clear();
		self.atoms.clear();
		self.break_sequence();
		result
	}

	/// Applies a decoded editor key at the given layout width.
	pub fn handle(&mut self, key: Key, width: u16, page_rows: usize) -> BufferOutcome {
		self.layout_width = width.max(1);
		// The copy stash lives exactly one key: hosts drain it right after
		// the `Copy`/`Cut` that filled it, and any other key voids it so a
		// later drain can never emit stale clipboard contents.
		self.copied = None;
		self.manual_scroll.set(false);
		if let Some(jump) = self.jump.take() {
			return match key {
				Key::Char(ch) => self.jump_to(ch, jump),
				Key::Space => self.jump_to(' ', jump),
				_ => BufferOutcome::Ignored,
			};
		}
		match key {
			Key::Ctrl(']') => {
				self.anchor = None;
				self.jump = Some(Jump::Forward);
				self.break_sequence();
				BufferOutcome::Changed
			},
			Key::CtrlAlt(']') => {
				self.anchor = None;
				self.jump = Some(Jump::Backward);
				self.break_sequence();
				BufferOutcome::Changed
			},
			Key::Ctrl('-' | '_') => self.undo(),
			Key::Ctrl('y') => self.yank(),
			Key::Alt('y') => self.yank_pop(),
			Key::Ctrl('k') => self.kill_line_end(),
			Key::Ctrl('u') => self.kill_line_start(),
			Key::Ctrl('w') => self.kill_word_backward(),
			Key::WordDelete => self.kill_word_forward(),
			Key::Backspace => self.backspace(),
			Key::Delete | Key::Ctrl('d') => self.delete(),
			Key::Left | Key::Ctrl('b') => self.collapse_or(false, Self::move_left),
			Key::Right | Key::Ctrl('f') => self.collapse_or(true, Self::move_right),
			Key::WordLeft => self.collapse_or(false, |buffer| {
				let at = buffer.word_left();
				buffer.move_to(at)
			}),
			Key::WordRight => self.collapse_or(true, |buffer| {
				let at = buffer.word_right();
				buffer.move_to(at)
			}),
			Key::Home | Key::Ctrl('a') => self.collapse_or(false, |buffer| {
				let at = buffer.line_bounds().0;
				buffer.move_to(at)
			}),
			Key::End | Key::Ctrl('e') => self.collapse_or(true, |buffer| {
				let at = buffer.line_bounds().1;
				buffer.move_to(at)
			}),
			Key::Up => self.collapse_or(false, |buffer| buffer.move_visual(-1)),
			Key::Down => self.collapse_or(true, |buffer| buffer.move_visual(1)),
			Key::PageUp => {
				self.collapse_or(false, |buffer| buffer.move_visual(-(page_rows.max(1) as isize)))
			},
			Key::PageDown => {
				self.collapse_or(true, |buffer| buffer.move_visual(page_rows.max(1) as isize))
			},
			Key::SelectLeft => self.extend(Self::move_left),
			Key::SelectRight => self.extend(Self::move_right),
			Key::SelectWordLeft => self.extend(|buffer| {
				let at = buffer.word_left();
				buffer.move_to(at)
			}),
			Key::SelectWordRight => self.extend(|buffer| {
				let at = buffer.word_right();
				buffer.move_to(at)
			}),
			Key::SelectHome => self.extend(|buffer| {
				let at = buffer.line_bounds().0;
				buffer.move_to(at)
			}),
			Key::SelectEnd => self.extend(|buffer| {
				let at = buffer.line_bounds().1;
				buffer.move_to(at)
			}),
			Key::SelectUp => self.extend(|buffer| buffer.move_visual(-1)),
			Key::SelectDown => self.extend(|buffer| buffer.move_visual(1)),
			Key::SelectAll => {
				self.select_all();
				BufferOutcome::Changed
			},
			Key::Copy => self.copy_selection(),
			Key::Cut => self.cut_selection(),
			Key::Esc => {
				let changed = self.selection().is_some();
				self.anchor = None;
				self.break_sequence();
				if changed {
					BufferOutcome::Changed
				} else {
					BufferOutcome::Ignored
				}
			},
			Key::Enter | Key::ShiftEnter => self.insert_char('\n'),
			Key::Space => self.insert_char(' '),
			Key::Char(ch) => self.insert_char(ch),
			_ => {
				self.break_sequence();
				BufferOutcome::Ignored
			},
		}
	}

	/// Returns the visible visual rows.
	///
	/// Keyboard editing keeps the cursor in view. A manual viewport scroll
	/// remains detached until the next editing command.
	#[must_use]
	pub fn rows(&self, width_limit: u16, max_rows: usize) -> SmallVec<VisualRow<'_>, 8> {
		let segments = self.segments(width_limit.max(1));
		let cursor_row = self.segment_at_cursor(&segments);
		let visible = segments.len().min(max_rows);
		let max_offset = segments.len() - visible;
		let first = if self.manual_scroll.get() {
			self.view_offset.get().min(max_offset)
		} else {
			cursor_row
				.saturating_sub(max_rows.saturating_sub(1))
				.min(max_offset)
		};
		self.view_offset.set(first);
		segments[first..first + visible]
			.iter()
			.map(|segment| VisualRow {
				start:         segment.start,
				end:           segment.end,
				text:          &self.text[segment.start..segment.end],
				cursor_column: (self.cursor >= segment.start
					&& self.cursor <= segment.end
					&& (segment.last || self.cursor < segment.end))
					.then(|| cell_width(&self.text[segment.start..self.cursor])),
			})
			.collect()
	}

	/// Moves the visible row window without moving the cursor.
	///
	/// Returns whether the clamped viewport offset changed.
	pub fn scroll_rows(&self, delta: i32, width_limit: u16, max_rows: usize) -> bool {
		let segments = self.segments(width_limit.max(1));
		let visible = segments.len().min(max_rows);
		let max_offset = segments.len().saturating_sub(visible);
		let current = if self.manual_scroll.get() {
			self.view_offset.get().min(max_offset)
		} else {
			self
				.segment_at_cursor(&segments)
				.saturating_sub(max_rows.saturating_sub(1))
				.min(max_offset)
		};
		let next = (current as i64 + i64::from(delta)).clamp(0, max_offset as i64) as usize;
		self.view_offset.set(next);
		self.manual_scroll.set(true);
		next != current
	}

	#[must_use]
	/// Returns the clipped visual row count.
	pub fn visual_height(&self, width: u16, max_rows: usize) -> usize {
		self.segments(width.max(1)).len().min(max_rows)
	}

	#[must_use]
	/// Reports whether the cursor is at the document's visual start.
	pub fn at_visual_start(&self) -> bool {
		self.segment_at_cursor(&self.segments(self.layout_width)) == 0 && self.cursor == 0
	}

	#[must_use]
	/// Reports whether the cursor is at the document's visual end.
	pub fn at_visual_end(&self) -> bool {
		let segments = self.segments(self.layout_width);
		self.segment_at_cursor(&segments) + 1 == segments.len() && self.cursor == self.text.len()
	}

	fn snapshot(&mut self) {
		if self
			.undo
			.last()
			.is_some_and(|state| state.0 == self.text && state.1 == self.cursor)
		{
			return;
		}
		if self.undo.len() == UNDO_CAP {
			self.undo.remove(0);
		}
		self
			.undo
			.push((self.text.clone(), self.cursor, self.atoms.clone()));
	}

	fn undo(&mut self) -> BufferOutcome {
		let Some((text, cursor, atoms)) = self.undo.pop() else {
			self.break_sequence();
			return BufferOutcome::Ignored;
		};
		self.text = text;
		self.cursor = cursor;
		self.atoms = atoms;
		self.anchor = None;
		self.desired = None;
		self.break_sequence();
		BufferOutcome::Changed
	}

	const fn break_sequence(&mut self) {
		self.last_action = Action::Other;
		self.last_yank = None;
	}

	fn collapse_or(
		&mut self,
		forward: bool,
		motion: impl FnOnce(&mut Self) -> BufferOutcome,
	) -> BufferOutcome {
		if let Some(selection) = self.selection() {
			self.cursor = if forward {
				selection.end
			} else {
				selection.start
			};
			self.anchor = None;
			self.desired = None;
			self.break_sequence();
			BufferOutcome::Changed
		} else {
			self.anchor = None;
			motion(self)
		}
	}

	fn extend(&mut self, motion: impl FnOnce(&mut Self) -> BufferOutcome) -> BufferOutcome {
		self.anchor.get_or_insert(self.cursor);
		motion(self)
	}

	fn insert_char(&mut self, ch: char) -> BufferOutcome {
		let word = ch.is_alphanumeric() || ch == '_';
		let selection = self.selection();
		if selection.is_some() || !word || self.last_action != Action::TypeWord {
			self.snapshot();
		}
		if let Some(range) = selection {
			self.cursor = range.start;
			self.splice(range, "");
			self.anchor = None;
			self.last_action = Action::Other;
		}
		if ch == '/'
			&& self.xml
			&& self.text[..self.cursor].ends_with('<')
			&& let Some(name) = nearest_open_tag(&self.text[..self.cursor - 1])
		{
			let name = Str::new(name);
			let mut expansion = String::with_capacity(name.len() + 2);
			expansion.push('/');
			expansion.push_str(&name);
			expansion.push('>');
			self.splice(self.cursor..self.cursor, &expansion);
			self.cursor += expansion.len();
		} else {
			let mut encoded = [0_u8; 4];
			self.splice(self.cursor..self.cursor, ch.encode_utf8(&mut encoded));
			self.cursor += ch.len_utf8();
		}
		self.anchor = None;
		self.desired = None;
		self.last_action = if word {
			Action::TypeWord
		} else {
			Action::Other
		};
		self.last_yank = None;
		BufferOutcome::Changed
	}

	fn move_to(&mut self, cursor: usize) -> BufferOutcome {
		self.break_sequence();
		self.desired = None;
		let forward = cursor >= self.cursor;
		let cursor = self.snap_position(cursor, forward);
		if cursor == self.cursor {
			BufferOutcome::Ignored
		} else {
			self.cursor = cursor;
			BufferOutcome::Changed
		}
	}

	fn move_left(&mut self) -> BufferOutcome {
		let Some((mut at, _)) = self.text[..self.cursor].grapheme_indices().next_back() else {
			self.break_sequence();
			return BufferOutcome::Ignored;
		};
		if let Some((start, _)) = self.atomic_at(at) {
			at = start;
		}
		self.move_to(at)
	}

	fn move_right(&mut self) -> BufferOutcome {
		let Some(grapheme) = self.text[self.cursor..].graphemes().next() else {
			self.break_sequence();
			return BufferOutcome::Ignored;
		};
		let at = self
			.atomic_at(self.cursor)
			.map_or(self.cursor + grapheme.len(), |(_, end)| end);
		self.move_to(at)
	}

	fn move_visual(&mut self, delta: isize) -> BufferOutcome {
		self.break_sequence();
		let segments = self.segments(self.layout_width);
		let current = self.segment_at_cursor(&segments);
		let target = current.saturating_add_signed(delta).min(segments.len() - 1);
		if target == current {
			let edge = self.snap_position(
				if delta < 0 {
					segments[current].start
				} else {
					segments[current].end
				},
				delta > 0,
			);
			if edge == self.cursor {
				return BufferOutcome::Ignored;
			}
			self.cursor = edge;
			return BufferOutcome::Changed;
		}
		let source = segments[current];
		let destination = segments[target];
		let column = self
			.desired
			.unwrap_or_else(|| cell_width(&self.text[source.start..self.cursor]));
		let max = if destination.last {
			cell_width(&self.text[destination.start..destination.end])
		} else {
			let text = &self.text[destination.start..destination.end];
			text
				.graphemes()
				.next_back()
				.map_or(0, |g| cell_width(text).saturating_sub(cell_width(g)))
		};
		let at = destination.start
			+ byte_at_column(&self.text[destination.start..destination.end], column.min(max));
		self.cursor = self.snap_position(at, delta > 0);
		self.desired = Some(column);
		BufferOutcome::Changed
	}

	fn backspace(&mut self) -> BufferOutcome {
		if let Some(range) = self.selection() {
			return self.delete_range(range.start, range.end, false);
		}
		let Some((mut start, _)) = self.text[..self.cursor].grapheme_indices().next_back() else {
			self.break_sequence();
			return BufferOutcome::Ignored;
		};
		if let Some((token_start, _)) = self.atomic_at(start) {
			start = token_start;
		}
		self.delete_range(start, self.cursor, false)
	}

	fn delete(&mut self) -> BufferOutcome {
		if let Some(range) = self.selection() {
			return self.delete_range(range.start, range.end, false);
		}
		let Some(grapheme) = self.text[self.cursor..].graphemes().next() else {
			self.break_sequence();
			return BufferOutcome::Ignored;
		};
		let end = self
			.atomic_at(self.cursor)
			.map_or(self.cursor + grapheme.len(), |(_, end)| end);
		self.delete_range(self.cursor, end, false)
	}

	fn kill_line_start(&mut self) -> BufferOutcome {
		let (start, _) = self.line_bounds();
		let start = if start == self.cursor && start > 0 {
			start - 1
		} else {
			start
		};
		self.delete_range(start, self.cursor, true)
	}

	fn kill_line_end(&mut self) -> BufferOutcome {
		let (_, end) = self.line_bounds();
		let end = if self.cursor < end {
			end
		} else if end < self.text.len() {
			end + 1
		} else {
			end
		};
		self.delete_range(self.cursor, end, true)
	}

	fn kill_word_backward(&mut self) -> BufferOutcome {
		let start = self.word_left();
		self.delete_range(start, self.cursor, true)
	}

	fn kill_word_forward(&mut self) -> BufferOutcome {
		let end = self.word_right();
		self.delete_range(self.cursor, end, true)
	}

	fn delete_range(&mut self, start: usize, end: usize, kill: bool) -> BufferOutcome {
		if start == end {
			if !kill {
				self.break_sequence();
			}
			return BufferOutcome::Ignored;
		}
		let (start, end) = self.expand_to_atoms(start, end);
		self.snapshot();
		let removed = self.text[start..end].to_owned();
		let backward = end == self.cursor;
		self.splice(start..end, "");
		self.cursor = start;
		self.anchor = None;
		self.desired = None;
		self.last_yank = None;
		if kill {
			self.record_kill(removed, backward);
		} else {
			self.last_action = Action::Other;
		}
		BufferOutcome::Changed
	}

	fn record_kill(&mut self, killed: String, backward: bool) {
		if self.last_action == Action::Kill && !self.kill_ring.is_empty() {
			if backward {
				self.kill_ring[0].insert_str(0, &killed);
			} else {
				self.kill_ring[0].push_str(&killed);
			}
		} else {
			self.kill_ring.insert(0, killed);
			if self.kill_ring.len() > KILL_CAP {
				self.kill_ring.pop();
			}
		}
		self.kill_index = 0;
		self.last_action = Action::Kill;
	}

	fn yank(&mut self) -> BufferOutcome {
		let Some(value) = self.kill_ring.first().cloned() else {
			self.break_sequence();
			return BufferOutcome::Ignored;
		};
		self.snapshot();
		let range = self.selection().unwrap_or(self.cursor..self.cursor);
		let start = range.start;
		self.splice(range, &value);
		self.cursor = start + value.len();
		self.anchor = None;
		self.kill_index = 0;
		self.last_yank = Some((start, self.cursor));
		self.last_action = Action::Yank;
		self.desired = None;
		BufferOutcome::Changed
	}

	fn copy_selection(&mut self) -> BufferOutcome {
		let Some(range) = self.selection() else {
			return BufferOutcome::Ignored;
		};
		self.copied = Some(Str::from(&self.text[range]));
		BufferOutcome::Changed
	}

	fn cut_selection(&mut self) -> BufferOutcome {
		let Some(range) = self.selection() else {
			return BufferOutcome::Ignored;
		};
		self.copied = Some(Str::from(&self.text[range.clone()]));
		self.delete_range(range.start, range.end, true)
	}

	fn yank_pop(&mut self) -> BufferOutcome {
		if !matches!(self.last_action, Action::Yank | Action::YankPop) || self.kill_ring.len() < 2 {
			self.break_sequence();
			return BufferOutcome::Ignored;
		}
		let Some((start, end)) = self.last_yank else {
			return BufferOutcome::Ignored;
		};
		self.snapshot();
		self.kill_index = (self.kill_index + 1) % self.kill_ring.len();
		let value = self.kill_ring[self.kill_index].clone();
		self.splice(start..end, &value);
		self.cursor = start + value.len();
		self.last_yank = Some((start, self.cursor));
		self.last_action = Action::YankPop;
		BufferOutcome::Changed
	}

	fn jump_to(&mut self, ch: char, jump: Jump) -> BufferOutcome {
		self.break_sequence();
		let found = match jump {
			Jump::Forward => self.text[self.cursor..]
				.char_indices()
				.find(|(offset, candidate)| *offset > 0 && *candidate == ch)
				.map(|(offset, _)| self.cursor + offset),
			Jump::Backward => self.text[..self.cursor]
				.char_indices()
				.rev()
				.find(|(_, candidate)| *candidate == ch)
				.map(|(offset, _)| offset),
		};
		found.map_or(BufferOutcome::Ignored, |at| {
			self.cursor = self.snap_position(at, matches!(jump, Jump::Forward));
			BufferOutcome::Changed
		})
	}

	fn snap_position(&self, at: usize, forward: bool) -> usize {
		self
			.atomic_at(at)
			.map_or(at, |(start, end)| if forward { end } else { start })
	}

	fn line_bounds(&self) -> (usize, usize) {
		let start = self.text[..self.cursor].rfind('\n').map_or(0, |at| at + 1);
		let end = self.text[self.cursor..]
			.find('\n')
			.map_or(self.text.len(), |at| self.cursor + at);
		(start, end)
	}

	fn word_left(&self) -> usize {
		if self.cursor > 0 && self.text.as_bytes()[self.cursor - 1] == b'\n' {
			self.cursor - 1
		} else {
			word_left(&self.text, self.cursor)
		}
	}

	fn word_right(&self) -> usize {
		if self.text.as_bytes().get(self.cursor) == Some(&b'\n') {
			self.cursor + 1
		} else {
			word_right(&self.text, self.cursor)
		}
	}

	fn atomic_at(&self, index: usize) -> Option<(usize, usize)> {
		self
			.atoms
			.iter()
			.find(|atom| index >= atom.start && index < atom.end)
			.map(|atom| (atom.start, atom.end))
	}

	fn visual_position(&self, row: usize, column: u16, width_limit: u16) -> usize {
		let segments = self.segments(width_limit.max(1));
		let index = self
			.view_offset
			.get()
			.saturating_add(row)
			.min(segments.len() - 1);
		let segment = segments[index];
		segment.start + byte_at_column(&self.text[segment.start..segment.end], column)
	}

	fn segments(&self, width_limit: u16) -> SmallVec<Segment, 16> {
		let mut result = SmallVec::new();
		let mut logical_start = 0;
		loop {
			let logical_end = self.text[logical_start..]
				.find('\n')
				.map_or(self.text.len(), |at| logical_start + at);
			if logical_start == logical_end {
				result.push(Segment { start: logical_start, end: logical_end, last: true });
			} else if self.text[logical_start..logical_end]
				.bytes()
				.all(|byte| matches!(byte, b' '..=b'~'))
			{
				let limit = usize::from(width_limit.max(1));
				let mut start = logical_start;
				while start < logical_end {
					let hard_end = start.saturating_add(limit).min(logical_end);
					let end = if hard_end < logical_end {
						self.text.as_bytes()[start..hard_end]
							.iter()
							.rposition(|byte| *byte == b' ')
							.map_or(hard_end, |offset| start + offset + 1)
					} else {
						hard_end
					};
					result.push(Segment { start, end, last: end == logical_end });
					start = end;
				}
			} else {
				let mut start = logical_start;
				while start < logical_end {
					let mut cells = 0u16;
					let mut end = start;
					let mut whitespace_end = None;
					for (offset, grapheme) in self.text[start..logical_end].grapheme_indices() {
						let next = cells.saturating_add(cell_width(grapheme));
						if next > width_limit && end > start {
							break;
						}
						cells = next;
						end = start + offset + grapheme.len();
						if grapheme.chars().all(char::is_whitespace) {
							whitespace_end = Some(end);
						}
						if cells >= width_limit {
							break;
						}
					}
					if end < logical_end
						&& let Some(boundary) = whitespace_end.filter(|at| *at > start)
					{
						end = boundary;
					}
					if end == start {
						end = start + self.text[start..].graphemes().next().map_or(0, str::len);
					}
					result.push(Segment { start, end, last: end == logical_end });
					start = end;
				}
			}
			if logical_end == self.text.len() {
				break;
			}
			logical_start = logical_end + 1;
		}
		result
	}

	fn segment_at_cursor(&self, segments: &[Segment]) -> usize {
		segments
			.iter()
			.position(|segment| {
				self.cursor >= segment.start
					&& (self.cursor < segment.end || segment.last && self.cursor == segment.end)
			})
			.unwrap_or(segments.len() - 1)
	}
}

fn nearest_open_tag(text: &str) -> Option<&str> {
	let mut stack: SmallVec<&str, 16> = SmallVec::new();
	let mut offset = 0;
	while let Some(relative) = text[offset..].find('<') {
		let start = offset + relative;
		let rest = &text[start..];
		if let Some(body) = rest.strip_prefix("<!--") {
			let Some(end) = body.find("-->") else {
				break;
			};
			offset = start + 4 + end + 3;
			continue;
		}
		let processing = rest.starts_with("<?");
		let Some(end) = tag_end(text, start + 1, processing) else {
			break;
		};
		offset = end + 1;
		if processing || rest.starts_with("<!") {
			continue;
		}

		let mut name_start = start + 1;
		let closing = text.as_bytes().get(name_start) == Some(&b'/');
		if closing {
			name_start += 1;
		}
		let name_end = text[name_start..end]
			.find(|ch: char| {
				ch.is_whitespace() || matches!(ch, '/' | '>' | '<' | '=' | '?' | '!' | '"' | '\'')
			})
			.map_or(end, |relative| name_start + relative);
		if name_end == name_start {
			continue;
		}
		if closing {
			stack.pop();
		} else if !text[name_end..end].trim_ascii_end().ends_with('/') {
			stack.push(&text[name_start..name_end]);
		}
	}
	stack.pop()
}

fn tag_end(text: &str, start: usize, processing: bool) -> Option<usize> {
	let mut quote = None;
	let mut previous = None;
	for (relative, ch) in text[start..].char_indices() {
		if let Some(delimiter) = quote {
			if ch == delimiter {
				quote = None;
			}
		} else if matches!(ch, '"' | '\'') {
			quote = Some(ch);
		} else if ch == '>' && (!processing || previous == Some('?')) {
			return Some(start + relative);
		}
		previous = Some(ch);
	}
	None
}

fn byte_at_column(text: &str, column: u16) -> usize {
	text.truncate_width(usize::from(column)).len()
}
#[derive(Clone, Copy, Eq, PartialEq)]
enum WordClass {
	Word,
	Whitespace,
	Cjk,
	Delimiter,
}

const fn is_cjk(character: char) -> bool {
	matches!(
		character as u32,
		0x2E80..=0x2FFF
			| 0x3040..=0x30FF
			| 0x3100..=0x312F
			| 0x3130..=0x318F
			| 0x31A0..=0x31BF
			| 0x31F0..=0x31FF
			| 0x3400..=0x4DBF
			| 0x4E00..=0x9FFF
			| 0xA960..=0xA97F
			| 0xAC00..=0xD7AF
			| 0xF900..=0xFAFF
			| 0x20000..=0x2FA1F
	)
}

fn word_class(grapheme: &str) -> WordClass {
	let Some(character) = grapheme.chars().next() else {
		return WordClass::Delimiter;
	};
	if character.is_whitespace() {
		WordClass::Whitespace
	} else if is_cjk(character) {
		WordClass::Cjk
	} else if character.is_alphanumeric() || character == '_' {
		WordClass::Word
	} else {
		WordClass::Delimiter
	}
}

fn is_word_joiner(grapheme: &str) -> bool {
	matches!(grapheme, "'" | "’" | "-" | "‐" | "‑")
}

fn word_left(text: &str, at: usize) -> usize {
	let mut graphemes = text[..at].grapheme_indices().rev().peekable();
	while graphemes
		.peek()
		.is_some_and(|(_, grapheme)| word_class(grapheme) == WordClass::Whitespace)
	{
		graphemes.next();
	}
	let Some((offset, grapheme)) = graphemes.next() else {
		return 0;
	};
	let class = word_class(grapheme);
	if class == WordClass::Cjk {
		return offset;
	}
	if class != WordClass::Word {
		let mut target = offset;
		while let Some((offset, grapheme)) = graphemes.peek() {
			if word_class(grapheme) != class {
				break;
			}
			target = *offset;
			graphemes.next();
		}
		return target;
	}
	let mut target = offset;
	while let Some((offset, grapheme)) = graphemes.next() {
		if word_class(grapheme) == WordClass::Word {
			target = offset;
		} else if is_word_joiner(grapheme)
			&& graphemes
				.peek()
				.is_some_and(|(_, left)| word_class(left) == WordClass::Word)
		{
			let (left, _) = graphemes.next().expect("peeked left word");
			target = left;
		} else {
			break;
		}
	}
	target
}

fn word_right(text: &str, at: usize) -> usize {
	let mut graphemes = text[at..].grapheme_indices().peekable();
	while graphemes
		.peek()
		.is_some_and(|(_, grapheme)| word_class(grapheme) == WordClass::Whitespace)
	{
		graphemes.next();
	}
	let Some((first_at, first)) = graphemes.next() else {
		return text.len();
	};
	let class = word_class(first);
	let mut end = at + first_at + first.len();
	if class == WordClass::Cjk {
		return end;
	}
	if class != WordClass::Word {
		while let Some((_, grapheme)) = graphemes.peek() {
			if word_class(grapheme) != class {
				break;
			}
			let (offset, grapheme) = graphemes.next().expect("peeked delimiter");
			end = at + offset + grapheme.len();
		}
		return end;
	}
	while let Some((offset, grapheme)) = graphemes.next() {
		if word_class(grapheme) == WordClass::Word
			|| (is_word_joiner(grapheme)
				&& graphemes
					.peek()
					.is_some_and(|(_, right)| word_class(right) == WordClass::Word))
		{
			end = at + offset + grapheme.len();
		} else {
			break;
		}
	}
	end
}

type EmojiBuckets = HashMap<&'static str, Vec<[&'static str; 2]>>;

static EMOJI_BUCKETS: LazyLock<EmojiBuckets> = LazyLock::new(|| {
	serde_json::from_str(include_str!("emojis.json")).expect("embedded emoji data must be valid")
});

/// Feature switches for [`Editor::new`]; everything defaults on.
/// Completion is not a switch: register one with [`Editor::set_completion`].
#[derive(Clone, Copy, Debug)]
pub struct EditorOptions {
	/// `:emoji` shortcode dropdown plus inline `:shortcode:` and
	/// emoticon (`:-)`) expansion while typing.
	pub emoji:   bool,
	/// Up/Down prompt history with draft restore below the newest entry.
	pub history: bool,
	/// XML affordances: `</` completes the innermost open tag, and
	/// renderers should apply structural markup highlighting.
	pub xml:     bool,
}

impl Default for EditorOptions {
	fn default() -> Self {
		Self { emoji: true, history: true, xml: true }
	}
}

const EMOTICONS: &[(&str, &str)] = &[
	(":'-(", "😢"),
	(">:-(", "😠"),
	(":-)", "🙂"),
	(":-(", "🙁"),
	(":-D", "😃"),
	(":-P", "😛"),
	(":-p", "😛"),
	(":-O", "😮"),
	(":-o", "😮"),
	(":-|", "😐"),
	(":-/", "😕"),
	(":-\\", "😕"),
	(":-*", "😘"),
	(";-)", "😉"),
	(";-P", "😜"),
	(":')", "🥲"),
	(":'D", "😂"),
	(":'(", "😢"),
	("</3", "💔"),
	(">:(", "😠"),
	("B-)", "😎"),
	("8-)", "😎"),
	("o.O", "😳"),
	("O.o", "😳"),
	(":)", "🙂"),
	(":(", "🙁"),
	(":D", "😃"),
	(":P", "😛"),
	(":p", "😛"),
	(":O", "😮"),
	(":o", "😮"),
	(":|", "😐"),
	(":/", "😕"),
	(":\\", "😕"),
	(":*", "😘"),
	(";)", "😉"),
	(":3", "😺"),
	("<3", "❤️"),
	("xD", "😆"),
	("XD", "😆"),
	("B)", "😎"),
	("8)", "😎"),
];

/// Display content for one completion row.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SuggestionDisplay {
	/// A plain text label (command name, file path, mention, …).
	Text(Str),
	/// An emoji paired with its shortcode or emoticon.
	Emoji {
		/// The emoji inserted on acceptance.
		emoji:     &'static str,
		/// The `:shortcode:` name or emoticon spelling that matched.
		shortcode: &'static str,
	},
}

/// One selectable completion row.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Suggestion {
	value:       Str,
	display:     SuggestionDisplay,
	description: Option<Str>,
	hint:        Option<Str>,
}

impl Suggestion {
	/// Builds a row: on acceptance `insert` replaces the completion's
	/// prefix range verbatim; `label` is shown in the dropdown.
	pub fn new(insert: impl Into<Str>, label: impl Into<Str>) -> Self {
		Self {
			value:       insert.into(),
			display:     SuggestionDisplay::Text(label.into()),
			description: None,
			hint:        None,
		}
	}

	/// Explanatory text shown beside the label.
	#[must_use]
	pub fn with_description(mut self, description: impl Into<Str>) -> Self {
		self.description = Some(description.into());
		self
	}

	/// Ghost text shown after the cursor while this row is selected.
	#[must_use]
	pub fn with_hint(mut self, hint: impl Into<Str>) -> Self {
		self.hint = Some(hint.into());
		self
	}

	/// Returns the row's dropdown label.
	#[must_use]
	pub const fn display(&self) -> &SuggestionDisplay {
		&self.display
	}

	/// Returns optional explanatory text shown beside the label.
	#[must_use]
	pub fn description(&self) -> Option<&str> {
		self.description.as_deref()
	}
}

/// Ranked dropdown rows; inline up to eight before spilling.
pub type SuggestionList = SmallVec<Suggestion, 8>;

/// Ranked dropdown suggestions returned by [`EditorCompletion::suggest`].
pub struct Suggestions {
	/// Byte offset where the completed prefix starts; acceptance replaces
	/// `prefix_start..cursor` with the chosen suggestion's insert text.
	pub prefix_start: usize,
	/// Rows in display order; empty closes the dropdown.
	pub items:        SuggestionList,
}

/// Buffer edit returned by [`EditorCompletion::tab`]: replaces `range`
/// with `insert` and leaves the cursor after it.
pub struct CompletionEdit {
	/// Byte range to replace.
	pub range:  std::ops::Range<usize>,
	/// Replacement text.
	pub insert: Str,
}

/// Provider verdict for a Tab press, from [`EditorCompletion::tab`].
pub enum TabAction {
	/// Accept the selected dropdown row (no-op when none is open).
	Accept,
	/// Apply a buffer edit, e.g. materializing the current ghost hint.
	Edit(CompletionEdit),
	/// Pass Tab through to the embedding app.
	Pass,
}

/// Pluggable completion engine registered with [`Editor::set_completion`].
///
/// The editor consults it after every edit, so an implementation chooses
/// its own trigger convention (`/`, `@`, `#`, or none at all) by
/// inspecting the text before the cursor. [`SlashCommands`] is the
/// built-in pi-style implementation.
pub trait EditorCompletion {
	/// Dropdown suggestions for the current text and byte cursor, or
	/// `None` to close the dropdown.
	fn suggest(&mut self, text: &str, cursor: usize) -> Option<Suggestions>;

	/// Dim ghost text rendered after the cursor (usage hints, AI
	/// completion). Re-queried after every edit.
	fn hint(&mut self, text: &str, cursor: usize) -> Option<Str> {
		let _ = (text, cursor);
		None
	}

	/// Tab pressed. `selected` is the highlighted row while this engine's
	/// dropdown is open. Defaults to pi's behavior: accept the open row,
	/// otherwise pass Tab through to the embedding app. The built-in
	/// emoji dropdown always accepts without consulting the engine.
	fn tab(&mut self, text: &str, cursor: usize, selected: Option<&Suggestion>) -> TabAction {
		let _ = (text, cursor);
		if selected.is_some() {
			TabAction::Accept
		} else {
			TabAction::Pass
		}
	}
}

/// Active completion dropdown state.
pub struct Picker {
	prefix_start: usize,
	suggestions:  SuggestionList,
	selected:     usize,
	/// Produced by the registered engine (vs the built-in emoji dropdown).
	provided:     bool,
}

impl Picker {
	/// Returns the centered five-row suggestion window and its first index.
	#[must_use]
	pub fn visible_suggestions(&self) -> (usize, &[Suggestion]) {
		let visible = self.suggestions.len().min(PICKER_ROWS);
		let max_start = self.suggestions.len().saturating_sub(visible);
		let start = self.selected.saturating_sub(PICKER_ROWS / 2).min(max_start);
		(start, &self.suggestions[start..start + visible])
	}

	/// Returns the selected suggestion's absolute index.
	#[must_use]
	pub const fn selected(&self) -> usize {
		self.selected
	}

	/// Returns the total number of matching suggestions.
	#[must_use]
	pub const fn len(&self) -> usize {
		self.suggestions.len()
	}

	/// Reports whether no suggestions matched (never true for a live picker).
	#[must_use]
	pub const fn is_empty(&self) -> bool {
		self.suggestions.is_empty()
	}
}

/// Result of handling one terminal key event.
#[derive(Debug, Eq, PartialEq)]
pub enum EditOutcome {
	/// Editor contents or selection changed.
	Changed,
	/// Complete input was submitted, with paste markers expanded.
	Submitted(String),
	/// The key had no editor meaning; the embedding app may act on it.
	Ignored,
}

/// Editable multiline input with Pi-compatible completion and editing.
///
/// Wraps an [`EditBuffer`] with a pluggable [`EditorCompletion`] dropdown,
/// inline ghost hints, built-in emoji expansion, and prompt history —
/// each governed by [`EditorOptions`].
pub struct Editor {
	buffer:            EditBuffer,
	picker:            Option<Picker>,
	completion:        Option<Box<dyn EditorCompletion>>,
	options:           EditorOptions,
	hint:              Option<Str>,
	history:           Vec<Str>,
	history_index:     Option<usize>,
	history_draft:     Str,
	last_layout_width: Cell<u16>,
}

impl Editor {
	/// Creates an empty editor with the given feature switches.
	#[must_use]
	pub fn new(options: EditorOptions) -> Self {
		let mut buffer = EditBuffer::default();
		buffer.set_xml(options.xml);
		Self {
			buffer,
			picker: None,
			completion: None,
			options,
			hint: None,
			history: Vec::new(),
			history_index: None,
			history_draft: Str::new_static(""),
			last_layout_width: Cell::new(80),
		}
	}

	/// Registers the completion engine driving the dropdown, ghost text,
	/// and Tab behavior; replaces any previous one.
	pub fn set_completion(&mut self, completion: Box<dyn EditorCompletion>) {
		self.completion = Some(completion);
		self.refresh();
	}

	/// Returns the feature switches the editor was built with, so
	/// renderers can honor them (e.g. XML highlighting).
	#[must_use]
	pub const fn options(&self) -> EditorOptions {
		self.options
	}

	/// Returns the visible text, with paste markers unexpanded.
	#[must_use]
	pub fn text(&self) -> &str {
		self.buffer.text()
	}

	/// Returns the open completion dropdown, if any.
	#[must_use]
	pub const fn picker(&self) -> Option<&Picker> {
		self.picker.as_ref()
	}

	/// Returns the rows the open completion dropdown occupies (0 when closed).
	#[must_use]
	pub fn picker_height(&self) -> u16 {
		u16::try_from(
			self
				.picker
				.as_ref()
				.map_or(0, |picker| picker.len().min(PICKER_ROWS)),
		)
		.unwrap_or(u16::MAX)
	}

	#[cfg(test)]
	fn input_height(&self) -> u16 {
		u16::try_from(
			self
				.buffer
				.visual_height(self.last_layout_width.get(), MAX_INPUT_ROWS),
		)
		.unwrap_or(u16::MAX)
	}

	/// Returns the clipped input row count at `width`, remembering the
	/// width for subsequent key handling.
	pub fn input_height_for(&self, width: u16) -> u16 {
		self.last_layout_width.set(width.max(1));
		u16::try_from(self.buffer.visual_height(width.max(1), MAX_INPUT_ROWS)).unwrap_or(u16::MAX)
	}

	/// Returns the cursor-centered visible input rows at `width`.
	pub fn view(&self, width: u16) -> SmallVec<VisualRow<'_>, 8> {
		self.last_layout_width.set(width.max(1));
		self.buffer.rows(width, MAX_INPUT_ROWS)
	}

	/// Places the cursor on a visual input row and refreshes derived editor
	/// state.
	pub fn set_cursor_visual_row(&mut self, row: usize, column: u16, width: u16) {
		self.buffer.set_cursor_visual_row(row, column, width);
		self.refresh();
	}

	#[must_use]
	/// Returns the selected display-column span intersecting `row`.
	pub fn selection_span(&self, row: &VisualRow<'_>) -> Option<(u16, u16)> {
		self.buffer.selection_span(row)
	}

	/// Takes the text captured by the last `Copy`/`Cut`; see
	/// [`EditBuffer::take_copied`].
	pub fn take_copied(&mut self) -> Option<Str> {
		self.buffer.take_copied()
	}

	/// Extends the selection to a visual input row and refreshes derived state.
	pub fn extend_selection_visual_row(&mut self, row: usize, column: u16, width: u16) {
		self.buffer.extend_selection_visual_row(row, column, width);
		self.refresh();
	}

	/// Selects the word around a visual input position and refreshes derived
	/// state.
	pub fn select_word_visual_row(&mut self, row: usize, column: u16, width: u16) {
		self.buffer.select_word_visual_row(row, column, width);
		self.refresh();
	}

	/// Scrolls the input viewport by `delta` visual rows.
	///
	/// Returns whether the clamped viewport offset changed.
	pub fn scroll_rows(&self, delta: i32, width: u16, max_rows: usize) -> bool {
		self.buffer.scroll_rows(delta, width, max_rows)
	}

	/// Applies one decoded terminal key.
	pub fn handle_key(&mut self, key: Key) -> EditOutcome {
		self.handle(key)
	}

	/// Applies one decoded editor key.
	///
	/// While the dropdown is open, navigation and acceptance keys drive
	/// it and `Esc` closes it; every other key edits the buffer as usual.
	pub fn handle(&mut self, key: Key) -> EditOutcome {
		if self.picker.is_some() {
			return match key {
				Key::Esc => {
					self.picker = None;
					EditOutcome::Changed
				},
				Key::Up => self.select_previous(),
				Key::Down => self.select_next(),
				Key::PageUp => self.select_page(false),
				Key::PageDown => self.select_page(true),
				Key::Enter => self.accept_picker(),
				Key::Tab => self.tab_complete(),
				_ => self.handle_without_picker(key),
			};
		}
		self.handle_without_picker(key)
	}

	fn handle_without_picker(&mut self, key: Key) -> EditOutcome {
		match key {
			Key::Enter => self.submit(),
			Key::Tab => self.tab_complete(),
			Key::Up if self.options.history && self.history_gate_up() => self.history_older(),
			Key::Down
				if self.options.history
					&& self.history_index.is_some()
					&& self.buffer.at_visual_end() =>
			{
				self.history_newer()
			},
			_ => {
				if matches!(key, Key::Char(_) | Key::Space | Key::Backspace | Key::Delete)
					&& self.history_index.is_some()
				{
					self.history_index = None;
				}
				let outcome = self
					.buffer
					.handle(key, self.last_layout_width.get(), MAX_INPUT_ROWS);
				if matches!(outcome, BufferOutcome::Changed) {
					if self.options.emoji {
						match key {
							Key::Char(':') => self.replace_shortcode(),
							Key::Char(character) if character.is_whitespace() => self.replace_emoticon(),
							Key::Space => self.replace_emoticon(),
							_ => {},
						}
					}
					self.refresh();
					EditOutcome::Changed
				} else {
					EditOutcome::Ignored
				}
			},
		}
	}

	fn tab_complete(&mut self) -> EditOutcome {
		// the built-in emoji dropdown accepts without consulting the engine
		if self.picker.as_ref().is_some_and(|picker| !picker.provided) {
			return self.accept_picker();
		}
		let action = match self.completion.as_mut() {
			Some(completion) => {
				let selected = self
					.picker
					.as_ref()
					.map(|picker| &picker.suggestions[picker.selected]);
				completion.tab(self.buffer.text(), self.buffer.cursor(), selected)
			},
			None if self.picker.is_some() => TabAction::Accept,
			None => TabAction::Pass,
		};
		match action {
			TabAction::Accept if self.picker.is_some() => self.accept_picker(),
			TabAction::Edit(edit) => {
				self.buffer.replace_range(edit.range, &edit.insert);
				self.refresh();
				EditOutcome::Changed
			},
			TabAction::Accept | TabAction::Pass => EditOutcome::Ignored,
		}
	}

	fn history_gate_up(&self) -> bool {
		if !self.buffer.at_visual_start() {
			return false;
		}
		self.history_index.is_some() || self.buffer.text().is_empty()
	}

	fn history_older(&mut self) -> EditOutcome {
		if self.history.is_empty() {
			return EditOutcome::Ignored;
		}
		let next = self.history_index.map_or(0, |index| index + 1);
		if next >= self.history.len() {
			return EditOutcome::Ignored;
		}
		if self.history_index.is_none() {
			self.history_draft = Str::new(self.buffer.text());
		}
		self.history_index = Some(next);
		self.buffer.replace_external(&self.history[next], true);
		self.refresh();
		EditOutcome::Changed
	}

	fn history_newer(&mut self) -> EditOutcome {
		let Some(index) = self.history_index else {
			return EditOutcome::Ignored;
		};
		if index == 0 {
			self.history_index = None;
			self.buffer.replace_external(&self.history_draft, false);
		} else {
			self.history_index = Some(index - 1);
			self
				.buffer
				.replace_external(&self.history[index - 1], false);
		}
		self.refresh();
		EditOutcome::Changed
	}

	/// Inserts sanitized text at the cursor (pastes, programmatic prefill).
	pub fn insert_text(&mut self, text: &str) -> EditOutcome {
		self.history_index = None;
		if matches!(self.buffer.insert_text(text), BufferOutcome::Changed) {
			self.refresh();
			EditOutcome::Changed
		} else {
			EditOutcome::Ignored
		}
	}

	/// Inserts an atomic reference at the cursor; see
	/// [`EditBuffer::insert_reference`].
	pub fn insert_reference(&mut self, marker: &str, payload: &str) -> EditOutcome {
		self.history_index = None;
		if matches!(self.buffer.insert_reference(marker, payload), BufferOutcome::Changed) {
			self.refresh();
			EditOutcome::Changed
		} else {
			EditOutcome::Ignored
		}
	}

	/// Byte ranges of atomic markers in the visible text; see
	/// [`EditBuffer::atom_ranges`].
	#[must_use]
	pub fn atom_ranges(&self) -> SmallVec<(usize, usize), 4> {
		self.buffer.atom_ranges()
	}

	fn submit(&mut self) -> EditOutcome {
		if self.buffer.text().trim().is_empty() {
			return EditOutcome::Ignored;
		}
		if self.options.history {
			let submitted = self.buffer.expanded_text();
			self
				.history
				.retain(|entry| entry.as_str() != submitted.as_str());
			self.history.insert(0, submitted.into_str());
			self.history.truncate(HISTORY_CAPACITY);
		}
		self.history_index = None;
		self.picker = None;
		self.hint = None;
		EditOutcome::Submitted(self.buffer.clear_after_submit())
	}

	const fn select_previous(&mut self) -> EditOutcome {
		let picker = self.picker.as_mut().expect("picker presence was checked");
		picker.selected = if picker.selected == 0 {
			picker.len() - 1
		} else {
			picker.selected - 1
		};
		EditOutcome::Changed
	}

	const fn select_next(&mut self) -> EditOutcome {
		let picker = self.picker.as_mut().expect("picker presence was checked");
		picker.selected = (picker.selected + 1) % picker.len();
		EditOutcome::Changed
	}

	fn select_page(&mut self, down: bool) -> EditOutcome {
		let picker = self.picker.as_mut().expect("picker presence was checked");
		picker.selected = if down {
			picker
				.selected
				.saturating_add(PICKER_ROWS)
				.min(picker.len() - 1)
		} else {
			picker.selected.saturating_sub(PICKER_ROWS)
		};
		EditOutcome::Changed
	}

	fn accept_picker(&mut self) -> EditOutcome {
		let picker = self.picker.take().expect("picker presence was checked");
		let suggestion = &picker.suggestions[picker.selected];
		self
			.buffer
			.replace_range(picker.prefix_start..self.buffer.cursor(), &suggestion.value);
		self.refresh();
		EditOutcome::Changed
	}

	/// Re-queries the completion engine (dropdown and ghost hint), falling
	/// back to the built-in emoji dropdown when the engine declines.
	fn refresh(&mut self) {
		let cursor = self.buffer.cursor();
		let text = self.buffer.text();
		let mut picker = self.completion.as_mut().and_then(|completion| {
			let suggestions = completion.suggest(text, cursor)?;
			(!suggestions.items.is_empty()).then_some(Picker {
				prefix_start: suggestions.prefix_start,
				suggestions:  suggestions.items,
				selected:     0,
				provided:     true,
			})
		});
		if picker.is_none() && self.options.emoji {
			picker = emoji_picker(&text[..cursor]);
		}
		self.hint = self
			.completion
			.as_mut()
			.and_then(|completion| completion.hint(text, cursor));
		self.picker = picker;
	}

	/// Dim ghost text rendered after the cursor: the selected suggestion's
	/// hint while the dropdown is open, otherwise the completion engine's
	/// latest [`EditorCompletion::hint`].
	#[must_use]
	pub fn inline_hint(&self) -> Option<Str> {
		if let Some(picker) = &self.picker
			&& let Some(hint) = &picker.suggestions[picker.selected].hint
		{
			return Some(hint.clone());
		}
		self.hint.clone()
	}

	fn replace_shortcode(&mut self) {
		let cursor = self.buffer.cursor();
		let before = &self.buffer.text()[..cursor];
		let bytes = before.as_bytes();
		if bytes.last() != Some(&b':') {
			return;
		}
		let close = bytes.len() - 1;
		let mut name_start = close;
		while name_start > 0 && is_name_byte(bytes[name_start - 1]) {
			name_start -= 1;
		}
		if name_start == close || name_start == 0 || bytes[name_start - 1] != b':' {
			return;
		}
		let open = name_start - 1;
		if !has_left_boundary(bytes, open) {
			return;
		}
		let name = before[name_start..close].to_ascii_lowercase();
		if let Some(emoji) = lookup_emoji(&name) {
			self.buffer.replace_range(open..cursor, emoji);
		}
	}

	fn replace_emoticon(&mut self) {
		let cursor = self.buffer.cursor();
		let before = &self.buffer.text()[..cursor];
		let Some(terminator) = before.chars().next_back() else {
			return;
		};
		let tail = before.len() - terminator.len_utf8();
		for &(pattern, emoji) in EMOTICONS {
			let Some(start) = tail.checked_sub(pattern.len()) else {
				continue;
			};
			if before.get(start..tail) != Some(pattern) || !has_left_boundary(before.as_bytes(), start)
			{
				continue;
			}
			let mut replacement = String::with_capacity(emoji.len() + terminator.len_utf8());
			replacement.push_str(emoji);
			replacement.push(terminator);
			self.buffer.replace_range(start..cursor, &replacement);
			break;
		}
	}
}

/// One slash-command palette entry completed by [`SlashCommands`].
#[derive(Clone)]
pub struct Command {
	name:        Str,
	description: Str,
	aliases:     SmallVec<Str, 1>,
	args:        Box<[CommandArg]>,
	hint:        Option<Str>,
}

/// One argument candidate completed after a command name (`/mcp add …`).
#[derive(Clone)]
struct CommandArg {
	name:        Str,
	description: Str,
	usage:       Option<Str>,
}

impl Command {
	/// Builds a palette entry from its name, blurb, and alias spellings.
	pub fn new(name: &str, description: &str, aliases: &[&str]) -> Self {
		Self {
			name:        Str::new(name),
			description: Str::new(description),
			aliases:     aliases.iter().map(Str::new).collect(),
			args:        Box::default(),
			hint:        None,
		}
	}

	/// Argument candidates offered once the command name is complete:
	/// `(name, description, usage)`, with `""` usage meaning none. Usage
	/// text ghosts after the argument pi-style (`<path>`, `<a> <b>`).
	#[must_use]
	pub fn with_args(mut self, args: &[(&str, &str, &str)]) -> Self {
		self.args = args
			.iter()
			.map(|&(name, description, usage)| CommandArg {
				name:        Str::new(name),
				description: Str::new(description),
				usage:       (!usage.is_empty()).then(|| Str::new(usage)),
			})
			.collect();
		self
	}

	/// Usage hint shown as dim ghost text after the cursor, pi-style
	/// (e.g. `<name> [--scope project|user]`).
	#[must_use]
	pub fn with_hint(mut self, hint: &str) -> Self {
		self.hint = Some(Str::new(hint));
		self
	}

	/// The command's primary spelling, without the leading `/`.
	pub fn name(&self) -> &str {
		&self.name
	}

	/// The one-line blurb shown beside the command name.
	pub fn description(&self) -> &str {
		&self.description
	}
}

/// Pi-compatible slash-command completion over a fixed [`Command`] palette.
///
/// `/` at a line start opens ranked name completion, the first argument
/// completes against candidates, and usage text ghosts after the cursor
/// (pi `buildSubcommandInlineHint`).
pub struct SlashCommands {
	commands: Box<[Command]>,
}

impl SlashCommands {
	/// Wraps a command palette for [`Editor::set_completion`].
	pub fn new(commands: impl Into<Box<[Command]>>) -> Self {
		Self { commands: commands.into() }
	}

	fn find(&self, name: &str) -> Option<&Command> {
		self
			.commands
			.iter()
			.find(|command| command.name == name || command.aliases.iter().any(|a| a == name))
	}

	fn name_suggestions(&self, line_start: usize, line: &str) -> Option<Suggestions> {
		let trimmed = line.trim_start_matches([' ', '\t']);
		let body = trimmed.strip_prefix('/')?;
		// a second slash means a path, not a command
		if body.contains('/') {
			return None;
		}
		let prefix_start = line_start + line.len() - trimmed.len();
		let query = body.to_ascii_lowercase();
		let mut ranked: SmallVec<(u16, Suggestion), 8> = SmallVec::new();
		for command in &self.commands {
			let mut selected_name = &command.name;
			let mut score = command_score(&query, &command.name);
			for alias in &command.aliases {
				let alias_score = command_score(&query, alias);
				if alias_score > score {
					selected_name = alias;
					score = alias_score;
				}
			}
			let description_score = fuzzy_score(&query, &command.description.to_ascii_lowercase()) / 2;
			score = score.max(description_score);
			if score > 0 {
				ranked.push((score, Suggestion {
					value:       fmts!("/{selected_name} "),
					display:     SuggestionDisplay::Text(selected_name.clone()),
					description: Some(command.description.clone()),
					hint:        command.hint.clone(),
				}));
			}
		}
		ranked.sort_by_key(|(score, _)| Reverse(*score));
		let items = ranked
			.into_iter()
			.map(|(_, suggestion)| suggestion)
			.collect::<SuggestionList>();
		(!items.is_empty()).then_some(Suggestions { prefix_start, items })
	}

	/// Completion for the argument position of a recognized command: the
	/// text after `/name ` matches against the command's argument
	/// candidates. Only the first argument completes; later words are
	/// free-form.
	fn argument_suggestions(&self, cursor: usize, body: &str, space: usize) -> Option<Suggestions> {
		let (name, rest) = body.split_at(space);
		let partial = rest.trim_start_matches([' ', '\t']);
		if partial.contains(char::is_whitespace) {
			return None;
		}
		let command = self.find(name)?;
		if command.args.is_empty() {
			return None;
		}
		let prefix_start = cursor - partial.len();
		let query = partial.to_ascii_lowercase();
		let mut ranked: SmallVec<(u16, Suggestion), 8> = SmallVec::new();
		for arg in &command.args {
			let score = command_score(&query, &arg.name);
			if score > 0 {
				ranked.push((score, Suggestion {
					value:       fmts!("{} ", arg.name),
					display:     SuggestionDisplay::Text(arg.name.clone()),
					description: Some(arg.description.clone()),
					hint:        None,
				}));
			}
		}
		ranked.sort_by_key(|(score, _)| Reverse(*score));
		let items = ranked
			.into_iter()
			.map(|(_, suggestion)| suggestion)
			.collect::<SuggestionList>();
		(!items.is_empty()).then_some(Suggestions { prefix_start, items })
	}
}

impl EditorCompletion for SlashCommands {
	fn suggest(&mut self, text: &str, cursor: usize) -> Option<Suggestions> {
		let before = &text[..cursor];
		let line_start = before.rfind('\n').map_or(0, |index| index + 1);
		let line = &before[line_start..];
		let body = line.trim_start_matches([' ', '\t']).strip_prefix('/')?;
		match body.find(char::is_whitespace) {
			Some(space) => self.argument_suggestions(cursor, body, space),
			None => self.name_suggestions(line_start, line),
		}
	}

	/// Pi-style usage ghosting: bare `/name ` shows the command's own
	/// usage; a partial argument shows its remaining characters plus
	/// usage; a chosen argument ghosts the usage words not yet typed.
	fn hint(&mut self, text: &str, cursor: usize) -> Option<Str> {
		let line_start = text[..cursor].rfind('\n').map_or(0, |at| at + 1);
		let line = &text[line_start..cursor];
		let body = line.trim_start_matches([' ', '\t']).strip_prefix('/')?;
		let space = body.find(char::is_whitespace)?;
		let (name, rest) = body.split_at(space);
		let command = self.find(name)?;
		let argument = rest.trim_start_matches([' ', '\t']);
		if argument.is_empty() {
			return command.hint.clone();
		}
		match argument.find(char::is_whitespace) {
			None => {
				// still typing the argument name: remaining chars + usage
				let prefix = argument.to_ascii_lowercase();
				let matched = command
					.args
					.iter()
					.find(|arg| arg.name.starts_with(&prefix))?;
				let remaining = &matched.name.as_str()[prefix.len()..];
				match &matched.usage {
					Some(usage) => Some(fmts!("{remaining} {usage}")),
					None if remaining.is_empty() => None,
					None => Some(Str::new(remaining)),
				}
			},
			Some(argument_end) => {
				// argument chosen: ghost the usage words not yet typed
				let (chosen, after) = argument.split_at(argument_end);
				let arg = command.args.iter().find(|arg| arg.name == chosen)?;
				let usage = arg.usage.as_deref()?;
				let typed = after.split_whitespace().count();
				if typed == 0 {
					return Some(Str::new(usage));
				}
				let mut words = usage.split(' ');
				for _ in 0..typed {
					words.next()?;
				}
				let remaining = words.collect::<Vec<_>>().join(" ");
				(!remaining.is_empty()).then(|| Str::new(&remaining))
			},
		}
	}
}

fn emoji_picker(text_before_cursor: &str) -> Option<Picker> {
	let (prefix_start, query) = emoji_trigger(text_before_cursor)?;
	let mut suggestions = SuggestionList::new();
	let wanted = format!(":{query}");
	for &(pattern, emoji) in EMOTICONS {
		if suggestions.len() >= MAX_EMOJI_SUGGESTIONS {
			break;
		}
		if pattern.len() >= wanted.len() && pattern[..wanted.len()].eq_ignore_ascii_case(&wanted) {
			suggestions.push(Suggestion {
				value:       Str::new_static(emoji),
				display:     SuggestionDisplay::Emoji { emoji, shortcode: pattern },
				description: None,
				hint:        None,
			});
		}
	}
	let first = query.get(..1)?;
	if let Some(bucket) = EMOJI_BUCKETS.get(first) {
		let start = bucket.partition_point(|entry| entry[0] < query.as_str());
		for entry in &bucket[start..] {
			if suggestions.len() >= MAX_EMOJI_SUGGESTIONS || !entry[0].starts_with(&query) {
				break;
			}
			suggestions.push(Suggestion {
				value:       Str::new_static(entry[1]),
				display:     SuggestionDisplay::Emoji { emoji: entry[1], shortcode: entry[0] },
				description: None,
				hint:        None,
			});
		}
	}
	if suggestions.is_empty() {
		None
	} else {
		Some(Picker { prefix_start, suggestions, selected: 0, provided: false })
	}
}

fn emoji_trigger(text: &str) -> Option<(usize, String)> {
	let bytes = text.as_bytes();
	let mut index = bytes.len();
	while index > 0 && is_name_byte(bytes[index - 1]) {
		index -= 1;
	}
	if index == 0 || bytes[index - 1] != b':' {
		return None;
	}
	let colon = index - 1;
	if !has_left_boundary(bytes, colon) || index == bytes.len() {
		return None;
	}
	Some((colon, text[index..].to_ascii_lowercase()))
}

const fn is_name_byte(byte: u8) -> bool {
	byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'+' | b'-')
}

const fn has_left_boundary(bytes: &[u8], index: usize) -> bool {
	index == 0
		|| matches!(bytes[index - 1], b' ' | b'\t' | b'\n' | b'\r' | b'(' | b'[' | b'{' | b'>')
}

fn lookup_emoji(name: &str) -> Option<&'static str> {
	let bucket = EMOJI_BUCKETS.get(name.get(..1)?)?;
	let index = bucket.partition_point(|entry| entry[0] < name);
	bucket
		.get(index)
		.filter(|entry| entry[0] == name)
		.map(|entry| entry[1])
}

fn command_score(query: &str, target: &str) -> u16 {
	if query.is_empty() {
		1
	} else if query == target {
		1_000
	} else if target.starts_with(query) {
		900
	} else {
		fuzzy_score(query, target)
	}
}

fn fuzzy_score(query: &str, target: &str) -> u16 {
	if query.is_empty() {
		return 1;
	}
	let mut query_bytes = query.bytes();
	let Some(mut wanted) = query_bytes.next() else {
		return 1;
	};
	let mut matched = 0_u16;
	let mut gaps = 0_u16;
	for byte in target.bytes() {
		if byte == wanted {
			matched = matched.saturating_add(1);
			if let Some(next) = query_bytes.next() {
				wanted = next;
			} else {
				return 500_u16
					.saturating_add(matched.saturating_mul(8))
					.saturating_sub(gaps);
			}
		} else if matched > 0 {
			gaps = gaps.saturating_add(1);
		}
	}
	0
}

#[cfg(test)]
mod tests {
	use super::*;

	fn type_slash(text: &str) -> EditBuffer {
		let mut buffer = EditBuffer::new(text);
		assert_eq!(buffer.handle(Key::Char('/'), 80, 10), BufferOutcome::Changed);
		buffer
	}

	#[test]
	fn close_tag_completes_innermost_open_element() {
		let buffer = type_slash("<box><row gap=1><Foo.Bar>hi<");
		assert_eq!(buffer.text(), "<box><row gap=1><Foo.Bar>hi</Foo.Bar>");
		assert_eq!(buffer.cursor(), buffer.text().len());
	}

	#[test]
	fn close_tag_ignores_self_closing_elements() {
		let buffer = type_slash("<a><hr/><");
		assert_eq!(buffer.text(), "<a><hr/></a>");
		assert_eq!(buffer.cursor(), buffer.text().len());
	}

	#[test]
	fn close_tag_pops_already_closed_pair() {
		let buffer = type_slash("<a><b></b><");
		assert_eq!(buffer.text(), "<a><b></b></a>");
		assert_eq!(buffer.cursor(), buffer.text().len());
	}

	#[test]
	fn close_tag_respects_quoted_attribute_delimiters() {
		let buffer = type_slash("<a t=\"x>y\"><");
		assert_eq!(buffer.text(), "<a t=\"x>y\"></a>");
		assert_eq!(buffer.cursor(), buffer.text().len());
	}

	#[test]
	fn close_tag_ignores_comment_contents() {
		let buffer = type_slash("<a><!-- <b> --><");
		assert_eq!(buffer.text(), "<a><!-- <b> --></a>");
		assert_eq!(buffer.cursor(), buffer.text().len());
	}

	#[test]
	fn close_tag_types_literal_slash_when_stack_is_empty() {
		let buffer = type_slash("<");
		assert_eq!(buffer.text(), "</");
		assert_eq!(buffer.cursor(), buffer.text().len());
	}
	#[test]
	fn pasting_a_document_does_not_complete_its_closing_tags() {
		// completion is a typing affordance; a pasted document already
		// carries its closers, so duplicating them would corrupt the paste
		let document = "<box bg=\"black\">\n  <row gap=\"1\">\n    <col>hi</col>\n  </row>\n</box>";
		let mut buffer = EditBuffer::new("");
		assert_eq!(buffer.insert_text(document), BufferOutcome::Changed);
		assert_eq!(buffer.text(), document);
	}

	fn key(key: Key) -> Key {
		key
	}

	/// Small palette with enough shape for ranking-sensitive expectations.
	fn palette() -> Vec<Command> {
		vec![
			Command::new("security", "Plan, run, inspect, and compare security scans", &[])
				.with_args(&[
					("plan", "Draft a scan plan", ""),
					("import", "Import an external report", "<path>"),
					("compare", "Diff two runs", "<run-a> <run-b>"),
				])
				.with_hint("plan|import|compare"),
			Command::new("settings", "Open settings menu", &[]),
			Command::new("setup", "Open provider setup", &["providers"]),
		]
	}

	fn editor() -> Editor {
		let mut editor = Editor::new(EditorOptions::default());
		editor.set_completion(Box::new(SlashCommands::new(palette())));
		editor
	}

	fn type_text(editor: &mut Editor, text: &str) {
		for character in text.chars() {
			assert_eq!(editor.handle_key(key(Key::Char(character))), EditOutcome::Changed);
		}
	}

	#[test]
	fn command_picker_navigates_and_inserts_a_trailing_space() {
		let mut editor = editor();
		type_text(&mut editor, "/");
		assert_eq!(editor.handle_key(key(Key::Down)), EditOutcome::Changed);
		assert_eq!(editor.handle_key(key(Key::Enter)), EditOutcome::Changed);
		assert_eq!(editor.text(), "/settings ");
	}

	#[test]
	fn options_gate_emoji_history_and_xml() {
		// no completion registered: `/` never opens a dropdown
		let mut editor = Editor::new(EditorOptions::default());
		type_text(&mut editor, "/se");
		assert!(editor.picker().is_none(), "no completion registered");
		assert!(editor.inline_hint().is_none());

		let mut editor = Editor::new(EditorOptions { emoji: false, ..EditorOptions::default() });
		type_text(&mut editor, ":joy");
		assert!(editor.picker().is_none(), "emoji dropdown disabled");
		type_text(&mut editor, ": ");
		assert_eq!(editor.text(), ":joy: ", "shortcode expansion disabled");

		let mut editor = Editor::new(EditorOptions { history: false, ..EditorOptions::default() });
		type_text(&mut editor, "one");
		assert_eq!(editor.handle(Key::Enter), EditOutcome::Submitted("one".into()));
		assert_eq!(editor.handle(Key::Up), EditOutcome::Ignored, "history disabled");

		let mut editor = Editor::new(EditorOptions { xml: false, ..EditorOptions::default() });
		type_text(&mut editor, "<a></");
		assert_eq!(editor.text(), "<a></", "close-tag completion disabled");
	}

	#[test]
	fn argument_completion_inside_a_slash_command() {
		let mut editor = editor();
		type_text(&mut editor, "/security i");
		let picker = editor.picker().expect("argument candidates open");
		assert_eq!(
			*picker.suggestions[picker.selected].display(),
			SuggestionDisplay::Text("import".into())
		);
		assert_eq!(editor.handle_key(key(Key::Enter)), EditOutcome::Changed);
		assert_eq!(editor.text(), "/security import ");
		// second word is free-form: no picker re-opens
		assert!(editor.picker().is_none());
	}

	#[test]
	fn inline_hint_follows_selection_arguments_and_usage() {
		let mut editor = editor();
		type_text(&mut editor, "/sec");
		assert_eq!(editor.inline_hint().as_deref(), Some("plan|import|compare"));
		assert_eq!(editor.handle_key(key(Key::Enter)), EditOutcome::Changed);
		assert_eq!(editor.text(), "/security ");
		// bare `/name ` ghosts the command usage, picker open or not
		assert_eq!(editor.inline_hint().as_deref(), Some("plan|import|compare"));
		// typing an argument prefix ghosts the remaining name + its usage
		type_text(&mut editor, "im");
		assert_eq!(editor.inline_hint().as_deref(), Some("port <path>"));
		// accepting the argument ghosts its remaining usage
		assert_eq!(editor.handle_key(key(Key::Enter)), EditOutcome::Changed);
		assert_eq!(editor.text(), "/security import ");
		assert_eq!(editor.inline_hint().as_deref(), Some("<path>"));
		// usage words already typed stop ghosting
		type_text(&mut editor, "report.json");
		assert_eq!(editor.inline_hint(), None);
		// multi-word usages ghost only the remainder (pi counts whole and
		// in-progress words alike)
		let mut compare = self::editor();
		type_text(&mut compare, "/security compare one");
		assert_eq!(compare.inline_hint().as_deref(), Some("<run-b>"));
		type_text(&mut compare, " two");
		assert_eq!(compare.inline_hint(), None, "usage fully consumed");
	}

	#[test]
	fn emoji_picker_and_shortcode_use_the_same_dataset() {
		let mut picker_editor = editor();
		type_text(&mut picker_editor, ":joy");
		let picker = picker_editor.picker().expect("joy opens the picker");
		assert_eq!(*picker.suggestions[picker.selected].display(), SuggestionDisplay::Emoji {
			emoji:     "😂",
			shortcode: "joy",
		});
		assert_eq!(picker_editor.handle_key(key(Key::Enter)), EditOutcome::Changed);
		assert_eq!(picker_editor.text(), "😂");

		let mut shortcode_editor = editor();
		type_text(&mut shortcode_editor, ":joy:");
		assert_eq!(shortcode_editor.text(), "😂");
	}

	#[test]
	fn emoticon_replacement_is_unicode_boundary_safe() {
		let mut editor = editor();
		type_text(&mut editor, "é:) ");
		assert_eq!(editor.text(), "é:) ");
	}

	#[test]
	fn cursor_navigation_and_backspace_preserve_graphemes() {
		let mut editor = editor();
		type_text(&mut editor, "a👩‍💻b");
		assert_eq!(editor.handle_key(key(Key::Left)), EditOutcome::Changed);
		assert_eq!(editor.handle_key(key(Key::Backspace)), EditOutcome::Changed);
		assert_eq!(editor.text(), "ab");
	}

	#[test]
	fn shift_enter_adds_lines_and_vertical_navigation_preserves_column() {
		let mut editor = editor();
		type_text(&mut editor, "first");
		assert_eq!(editor.handle_key(Key::ShiftEnter), EditOutcome::Changed);
		type_text(&mut editor, "second");

		assert_eq!(editor.text(), "first\nsecond");
		assert_eq!(editor.input_height(), 2);
		{
			let rows = editor.view(20);
			assert_eq!(rows.iter().map(|row| row.text).collect::<Vec<_>>(), ["first", "second"]);
			assert_eq!(rows[0].cursor_column, None);
			assert_eq!(rows[1].cursor_column, Some(6));
		}

		assert_eq!(editor.handle_key(key(Key::Up)), EditOutcome::Changed);
		assert_eq!(editor.view(20)[0].cursor_column, Some(5));
		assert_eq!(editor.handle_key(key(Key::Down)), EditOutcome::Changed);
		assert_eq!(editor.view(20)[1].cursor_column, Some(6));

		assert_eq!(
			editor.handle_key(key(Key::Enter)),
			EditOutcome::Submitted("first\nsecond".to_owned())
		);
	}

	#[test]
	fn slash_commands_complete_at_the_start_of_later_lines() {
		let mut editor = editor();
		type_text(&mut editor, "context");
		editor.handle_key(Key::ShiftEnter);
		type_text(&mut editor, "/set");

		assert!(editor.picker().is_some());
		assert_eq!(editor.handle_key(key(Key::Enter)), EditOutcome::Changed);
		assert_eq!(editor.text(), "context\n/settings ");
	}

	#[test]
	fn control_a_e_and_u_are_scoped_to_logical_lines() {
		let mut editor = editor();
		assert_eq!(editor.insert_text("one\ntwo"), EditOutcome::Changed);
		assert_eq!(editor.handle_key(Key::Ctrl('a')), EditOutcome::Changed);
		assert_eq!(editor.buffer.cursor(), 4);
		assert_eq!(editor.handle_key(Key::Ctrl('e')), EditOutcome::Changed);
		assert_eq!(editor.buffer.cursor(), 7);
		// Ctrl-U kills only to this line's start, rather than clearing the document.
		assert_eq!(editor.handle_key(Key::Ctrl('u')), EditOutcome::Changed);
		assert_eq!(editor.text(), "one\n");
		assert_eq!(editor.handle_key(Key::Ctrl('u')), EditOutcome::Changed);
		assert_eq!(editor.text(), "one");
	}

	#[test]
	fn word_motion_keeps_apostrophes_and_hyphens_inside_words() {
		let mut editor = editor();
		editor.insert_text("don't foo-bar");
		assert_eq!(editor.handle(Key::WordLeft), EditOutcome::Changed);
		assert_eq!(editor.buffer.cursor(), 6);
		assert_eq!(editor.handle(Key::WordLeft), EditOutcome::Changed);
		assert_eq!(editor.buffer.cursor(), 0);
		assert_eq!(editor.handle(Key::WordRight), EditOutcome::Changed);
		assert_eq!(editor.buffer.cursor(), 5);
	}

	#[test]
	fn word_deletes_merge_logical_lines() {
		let mut editor = editor();
		editor.insert_text("first\nsecond");
		editor.handle(Key::Ctrl('a'));
		assert_eq!(editor.handle(Key::Ctrl('w')), EditOutcome::Changed);
		assert_eq!(editor.text(), "firstsecond");

		let mut forward = self::editor();
		forward.insert_text("first\nsecond");
		forward.buffer.set_cursor_line_column(0, 5);
		assert_eq!(forward.handle(Key::WordDelete), EditOutcome::Changed);
		assert_eq!(forward.text(), "firstsecond");
	}

	#[test]
	fn paste_normalizes_newlines_controls_and_unicode_composition() {
		let mut editor = editor();
		assert_eq!(editor.insert_text("a\r\nb\u{0007}e\u{301}"), EditOutcome::Changed);
		assert_eq!(editor.text(), "a\nbé");
	}

	#[test]
	fn vertical_motion_snaps_at_document_boundaries_before_ignoring() {
		let mut editor = editor();
		editor.insert_text("abc");
		assert_eq!(editor.handle(Key::Up), EditOutcome::Changed);
		assert_eq!(editor.buffer.cursor(), 0);
		assert_eq!(editor.handle(Key::Up), EditOutcome::Ignored);
		assert_eq!(editor.handle(Key::Down), EditOutcome::Changed);
		assert_eq!(editor.buffer.cursor(), 3);
		assert_eq!(editor.handle(Key::Down), EditOutcome::Ignored);
	}

	#[test]
	fn kill_ring_accumulates_yanks_and_cycles_older_entries() {
		let mut editor = editor();
		type_text(&mut editor, "alpha beta gamma");
		editor.handle(Key::Ctrl('w'));
		editor.handle(Key::Ctrl('w'));
		assert_eq!(editor.handle(Key::Ctrl('y')), EditOutcome::Changed);
		assert_eq!(editor.text(), "alpha beta gamma");
		editor.handle(Key::Space);
		type_text(&mut editor, "older");
		editor.handle(Key::Ctrl('w'));
		assert_eq!(editor.handle(Key::Ctrl('y')), EditOutcome::Changed);
		assert_eq!(editor.handle(Key::Alt('y')), EditOutcome::Changed);
		assert!(editor.text().ends_with("beta gamma"));
	}

	#[test]
	fn undo_coalesces_word_typing_and_splits_at_punctuation() {
		let mut editor = editor();
		type_text(&mut editor, "abc def");
		assert_eq!(editor.handle(Key::Ctrl('-')), EditOutcome::Changed);
		assert_eq!(editor.text(), "abc ");
		assert_eq!(editor.handle(Key::Ctrl('_')), EditOutcome::Changed);
		assert_eq!(editor.text(), "abc");
		assert_eq!(editor.handle(Key::Ctrl('-')), EditOutcome::Changed);
		assert_eq!(editor.text(), "");
	}

	#[test]
	fn history_deduplicates_navigates_and_restores_the_draft() {
		let mut editor = editor();
		type_text(&mut editor, "one");
		assert_eq!(editor.handle(Key::Enter), EditOutcome::Submitted("one".into()));
		type_text(&mut editor, "two");
		assert_eq!(editor.handle(Key::Enter), EditOutcome::Submitted("two".into()));
		type_text(&mut editor, "one");
		editor.handle(Key::Enter);
		assert_eq!(editor.history.len(), 2);
		assert_eq!(editor.handle(Key::Up), EditOutcome::Changed);
		assert_eq!(editor.text(), "one");
		assert_eq!(editor.handle(Key::Up), EditOutcome::Changed);
		assert_eq!(editor.text(), "two");
		editor.history_draft = "draft".into();
		editor.handle(Key::End);
		assert_eq!(editor.handle(Key::Down), EditOutcome::Changed);
		assert_eq!(editor.handle(Key::Down), EditOutcome::Changed);
		assert_eq!(editor.text(), "draft");
	}

	#[test]
	fn reference_markers_are_atomic_for_every_delete_and_expand_on_submit() {
		let payload = (0..12)
			.map(|n| format!("line{n}"))
			.collect::<Vec<_>>()
			.join("\n");
		for key in [
			Key::Backspace,
			Key::Delete,
			Key::Ctrl('w'),
			Key::WordDelete,
			Key::Ctrl('k'),
			Key::Ctrl('u'),
		] {
			let mut editor = editor();
			editor.insert_reference("txt #1", &payload);
			assert_eq!(editor.text(), "txt #1");
			if matches!(key, Key::Delete | Key::WordDelete | Key::Ctrl('k')) {
				editor.buffer.set_cursor_line_column(0, 0);
			}
			assert_eq!(editor.handle(key), EditOutcome::Changed);
			assert_eq!(editor.text(), "", "{key:?}");
		}
		let mut editor = editor();
		editor.insert_reference("txt #1", &payload);
		assert_eq!(editor.handle(Key::Enter), EditOutcome::Submitted(payload));
	}

	#[test]
	fn references_are_positional_atoms_immune_to_lookalike_text() {
		let mut editor = editor();
		assert_eq!(editor.insert_reference("* #1", "<ref image=1/>"), EditOutcome::Changed);
		type_text(&mut editor, " hi ");
		editor.insert_text("* #1");
		assert_eq!(editor.text(), "* #1 hi * #1");
		assert_eq!(
			editor.atom_ranges().as_slice(),
			&[(0, 4)],
			"typed lookalike text never becomes an atom"
		);
		assert_eq!(
			editor.handle(Key::Enter),
			EditOutcome::Submitted("<ref image=1/> hi * #1".into()),
			"only the real reference expands"
		);
	}

	#[test]
	fn reference_markers_delete_atomically_and_undo_restores_them() {
		let mut editor = editor();
		editor.insert_reference("* #1", "<ref image=1/>");
		assert_eq!(editor.handle(Key::Backspace), EditOutcome::Changed);
		assert_eq!(editor.text(), "");
		assert!(editor.atom_ranges().is_empty());
		assert_eq!(editor.handle(Key::Ctrl('_')), EditOutcome::Changed);
		assert_eq!(editor.text(), "* #1");
		assert_eq!(
			editor.atom_ranges().as_slice(),
			&[(0, 4)],
			"undo restores the atom, not just its text"
		);
	}

	#[test]
	fn partial_replacements_widen_to_whole_reference_markers() {
		let mut torn = editor();
		type_text(&mut torn, "ab");
		torn.insert_reference("* #1", "<ref image=1/>");
		type_text(&mut torn, "cd");
		// Overlap the marker's first byte only: the whole unit must go.
		torn.buffer.replace_range(1..3, "X");
		assert_eq!(torn.text(), "aXcd");
		assert!(torn.atom_ranges().is_empty(), "torn atom is dropped whole");
		// Insertions at the marker boundary leave the unit intact.
		let mut fresh = editor();
		fresh.insert_reference("* #1", "<ref image=1/>");
		fresh.buffer.replace_range(0..0, ">");
		assert_eq!(fresh.text(), ">* #1");
		assert_eq!(fresh.atom_ranges().as_slice(), &[(1, 5)]);
	}

	#[test]
	fn character_jump_moves_forward_and_backward() {
		let mut editor = editor();
		editor.insert_text("abacad");
		editor.buffer.set_cursor_line_column(0, 0);
		editor.handle(Key::Ctrl(']'));
		editor.handle(Key::Char('a'));
		assert_eq!(editor.buffer.cursor(), 2);
		editor.handle(Key::CtrlAlt(']'));
		editor.handle(Key::Char('a'));
		assert_eq!(editor.buffer.cursor(), 0);
	}

	#[test]
	fn page_motion_uses_visible_rows_and_keeps_sticky_column() {
		let mut editor = editor();
		editor.insert_text("abcd\nx\nabcd\nx\nabcd\nx\nabcd\nx\nabcd");
		editor.buffer.set_cursor_line_column(0, 3);
		editor.handle(Key::PageDown);
		assert_eq!((editor.buffer.cursor_line(), editor.buffer.cursor_column()), (8, 3));
		editor.handle(Key::PageUp);
		assert_eq!((editor.buffer.cursor_line(), editor.buffer.cursor_column()), (0, 3));
	}

	#[test]
	fn view_word_wraps_and_vertical_motion_uses_visual_rows() {
		let mut editor = editor();
		editor.insert_text("hello world");
		let rows = editor
			.view(7)
			.iter()
			.map(|row| row.text)
			.collect::<Vec<_>>();
		assert_eq!(rows, ["hello ", "world"]);
		editor.buffer.set_cursor_line_column(0, 4);
		editor.handle(Key::Down);
		assert_eq!(editor.buffer.cursor(), 10);
		editor.handle(Key::Up);
		assert_eq!(editor.buffer.cursor(), 4);
	}

	#[test]
	fn shift_motion_extends_and_plain_motion_collapses_selection() {
		let mut buffer = EditBuffer::new("abc");
		assert_eq!(buffer.handle(Key::SelectLeft, 80, 8), BufferOutcome::Changed);
		assert_eq!(buffer.handle(Key::SelectLeft, 80, 8), BufferOutcome::Changed);
		assert_eq!(buffer.selection(), Some(1..3));
		assert_eq!(buffer.selected_text(), Some("bc"));

		assert_eq!(buffer.handle(Key::Left, 80, 8), BufferOutcome::Changed);
		assert_eq!(buffer.cursor(), 1);
		assert_eq!(buffer.selection(), None);
	}

	#[test]
	fn typing_replaces_selection_in_one_undo_step() {
		let mut buffer = EditBuffer::new("abcd");
		buffer.handle(Key::SelectLeft, 80, 8);
		buffer.handle(Key::SelectLeft, 80, 8);
		assert_eq!(buffer.handle(Key::Char('X'), 80, 8), BufferOutcome::Changed);
		assert_eq!(buffer.text(), "abX");

		assert_eq!(buffer.handle(Key::Ctrl('_'), 80, 8), BufferOutcome::Changed);
		assert_eq!(buffer.text(), "abcd");
		assert_eq!(buffer.cursor(), 2);
		assert_eq!(buffer.handle(Key::Ctrl('_'), 80, 8), BufferOutcome::Ignored);
	}

	#[test]
	fn cut_deletes_and_yank_reinserts_selection() {
		let mut buffer = EditBuffer::new("one two");
		assert_eq!(buffer.handle(Key::SelectWordLeft, 80, 8), BufferOutcome::Changed);
		assert_eq!(buffer.selected_text(), Some("two"));
		assert_eq!(buffer.handle(Key::Cut, 80, 8), BufferOutcome::Changed);
		assert_eq!(buffer.text(), "one ");
		assert_eq!(buffer.take_copied().as_deref(), Some("two"), "the host drains the cut text");
		assert_eq!(buffer.take_copied(), None, "drained once");
		assert_eq!(buffer.handle(Key::Ctrl('y'), 80, 8), BufferOutcome::Changed);
		assert_eq!(buffer.text(), "one two");
	}

	#[test]
	fn copy_stashes_text_for_the_host_without_editing() {
		let mut buffer = EditBuffer::new("one two");
		assert_eq!(buffer.handle(Key::Copy, 80, 8), BufferOutcome::Ignored, "no selection");
		assert_eq!(buffer.take_copied(), None);
		buffer.handle(Key::SelectWordLeft, 80, 8);
		assert_eq!(buffer.handle(Key::Copy, 80, 8), BufferOutcome::Changed);
		assert_eq!(buffer.text(), "one two", "copy never edits");
		assert_eq!(buffer.take_copied().as_deref(), Some("two"));
	}

	#[test]
	fn undrained_copy_is_voided_by_the_next_key() {
		let mut buffer = EditBuffer::new("one two");
		buffer.handle(Key::SelectWordLeft, 80, 8);
		assert_eq!(buffer.handle(Key::Copy, 80, 8), BufferOutcome::Changed);
		// A host that skipped the drain must not surface the old text
		// alongside a later, unrelated edit.
		assert_eq!(buffer.handle(Key::Char('x'), 80, 8), BufferOutcome::Changed);
		assert_eq!(buffer.take_copied(), None, "stash lives exactly one key");
	}

	#[test]
	fn select_all_exposes_selected_text() {
		let mut buffer = EditBuffer::new("hello\nworld");
		assert_eq!(buffer.handle(Key::SelectAll, 80, 8), BufferOutcome::Changed);
		assert_eq!(buffer.selection(), Some(0..11));
		assert_eq!(buffer.selected_text(), Some("hello\nworld"));
	}

	#[test]
	fn selection_span_maps_columns_within_a_wrapped_row() {
		let mut buffer = EditBuffer::new("abcdefghi");
		buffer.set_cursor_visual_row(1, 1, 4);
		buffer.handle(Key::SelectRight, 4, 8);
		buffer.handle(Key::SelectRight, 4, 8);
		let rows = buffer.rows(4, 8);
		assert_eq!(rows[1].text, "efgh");
		assert_eq!(buffer.selection_span(&rows[1]), Some((1, 3)));
	}

	#[test]
	fn selection_edge_inside_atomic_marker_snaps_to_the_whole_atom() {
		let mut buffer = EditBuffer::new("a");
		buffer.insert_reference("[chip]", "<ref/>");
		buffer.insert_text("z");
		buffer.set_cursor_visual_row(0, 1, 80);
		buffer.extend_selection_visual_row(0, 3, 80);
		assert_eq!(buffer.atom_ranges().as_slice(), &[(1, 7)]);
		assert_eq!(buffer.selection(), Some(1..7));
		assert_eq!(buffer.selected_text(), Some("[chip]"));
	}

	/// Toy engine: `@` mentions with any-key trigger, a fixed ghost
	/// completion after `hel`, and Tab materializing that ghost text.
	struct AtNames;

	impl EditorCompletion for AtNames {
		fn suggest(&mut self, text: &str, cursor: usize) -> Option<Suggestions> {
			let before = &text[..cursor];
			let at = before.rfind('@')?;
			let query = &before[at + 1..];
			let items = ["alice", "bob"]
				.iter()
				.filter(|name| !query.is_empty() && name.starts_with(query))
				.map(|name| Suggestion::new(fmts!("@{name} "), *name))
				.collect::<SuggestionList>();
			(!items.is_empty()).then_some(Suggestions { prefix_start: at, items })
		}

		fn hint(&mut self, text: &str, cursor: usize) -> Option<Str> {
			text[..cursor]
				.ends_with("hel")
				.then(|| Str::new("lo world"))
		}

		fn tab(&mut self, text: &str, cursor: usize, selected: Option<&Suggestion>) -> TabAction {
			// with our dropdown open, Tab belongs to the app (focus switch)
			if selected.is_some() {
				return TabAction::Pass;
			}
			match self.hint(text, cursor) {
				Some(insert) => TabAction::Edit(CompletionEdit { range: cursor..cursor, insert }),
				None => TabAction::Pass,
			}
		}
	}

	#[test]
	fn custom_completion_controls_trigger_ghost_text_and_tab() {
		let mut editor = Editor::new(EditorOptions::default());
		editor.set_completion(Box::new(AtNames));
		type_text(&mut editor, "hi @al");
		let picker = editor.picker().expect("@ trigger opens the dropdown");
		assert_eq!(picker.len(), 1);
		// the engine overrides Tab even while its own dropdown is open
		assert_eq!(editor.handle(Key::Tab), EditOutcome::Ignored);
		assert!(editor.picker().is_some(), "passthrough leaves the dropdown open");
		assert_eq!(editor.handle(Key::Enter), EditOutcome::Changed);
		assert_eq!(editor.text(), "hi @alice ");

		type_text(&mut editor, "hel");
		assert_eq!(editor.inline_hint().as_deref(), Some("lo world"));
		assert_eq!(editor.handle(Key::Tab), EditOutcome::Changed);
		assert_eq!(editor.text(), "hi @alice hello world");
		// nothing to complete: Tab passes through to the embedding app
		assert_eq!(editor.handle(Key::Tab), EditOutcome::Ignored);

		// the emoji dropdown accepts on Tab without consulting the engine
		type_text(&mut editor, " :joy");
		assert!(editor.picker().is_some(), "emoji dropdown open");
		assert_eq!(editor.handle(Key::Tab), EditOutcome::Changed);
		assert!(editor.text().ends_with("😂"), "{}", editor.text());
	}
}
