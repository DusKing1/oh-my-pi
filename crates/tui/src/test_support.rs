//! Deterministic terminal emulation for renderer verification.
//!
//! Hidden from docs and semver: this module exists so the library tests and
//! the demo binary's tests can replay emitted ANSI against one cell-accurate
//! model instead of trusting the renderer's own bookkeeping.

use xutf::Text;

use crate::frame::{CellContent, Frame, Style};

/// Returns a frame row as visible text, exactly as [`TerminalModel`] renders
/// rows: grapheme heads concatenated, continuations skipped, right-trimmed.
pub fn frame_row_text(frame: &Frame, row: u16) -> String {
	let mut text = String::new();
	if row >= frame.size().height {
		return text;
	}
	for x in 0..frame.size().width {
		match &frame.cell(x, row).content {
			CellContent::Blank => text.push(' '),
			CellContent::Grapheme { text: glyph, .. } => text.push_str(glyph),
			CellContent::Image { id, row, col, rows, cols } => {
				let (placeholder, _) = crate::kitty::placeholder_cell(*id, *row, *col, *rows, *cols);
				text.push_str(&placeholder);
			},
			CellContent::Continuation => {},
		}
	}
	text.trim_end().to_owned()
}

/// Returns the style painted at one cell, for tests asserting colors.
pub fn frame_cell_style(frame: &Frame, x: u16, y: u16) -> Style {
	if x >= frame.size().width || y >= frame.size().height {
		return Style::default();
	}
	frame.cell(x, y).style
}

/// Minimal VT model with wide-glyph cell semantics.
///
/// Printable output is segmented into grapheme clusters and measured with the
/// same width function the [`Frame`] uses, so the model verifies the
/// renderer's byte stream under the renderer's own width policy. Implements
/// exactly the sequences the renderer emits: CR, LF (scrolls at the bottom
/// margin; a top-anchored region pushes the scrolled-out row into
/// `history`, matching the emulators behind `margin_scrollback`),
/// CUU/CUD/CUF, CUP, DECSTBM, `2J`/`3J`, `2K`, and DECAWM with
/// pending-wrap semantics: a soft-wrapped row is flagged as continuing its
/// predecessor and joins it when scrolled into `history`, matching native
/// copy; erasing a line clears that flag, as mainstream emulators do.
/// Writing over either half of a wide glyph blanks the partner cell, as
/// hardware terminals do.
pub struct TerminalModel {
	width:         usize,
	height:        usize,
	cursor_row:    usize,
	cursor_col:    usize,
	/// Top scroll margin, inclusive, zero-based.
	margin_top:    usize,
	/// Bottom scroll margin, inclusive, zero-based.
	margin_bottom: usize,
	/// `None` marks the continuation cell of the preceding wide glyph.
	screen:        Vec<Vec<Option<String>>>,
	/// Whether each screen row was created by autowrap and continues the
	/// line above it, as native selection would join it.
	wrapped:       Vec<bool>,
	autowrap:      bool,
	pending_wrap:  bool,
	/// Rows scrolled into native scrollback, oldest first; a soft-wrapped
	/// row extends the previous entry instead of starting a new one.
	pub history:   Vec<String>,
}

impl TerminalModel {
	/// Creates a blank terminal of `width` x `height` cells.
	pub fn new(width: usize, height: usize) -> Self {
		Self {
			width,
			height,
			cursor_row: 0,
			cursor_col: 0,
			margin_top: 0,
			margin_bottom: height - 1,
			screen: vec![Self::blank_row(width); height],
			wrapped: vec![false; height],
			autowrap: false,
			pending_wrap: false,
			history: Vec::new(),
		}
	}

	fn blank_row(width: usize) -> Vec<Option<String>> {
		vec![Some(" ".to_owned()); width]
	}

	/// Applies a chunk of renderer output.
	pub fn apply(&mut self, output: &str) {
		let chars: Vec<char> = output.chars().collect();
		let mut index = 0;
		while index < chars.len() {
			match chars[index] {
				'\x1b' if chars.get(index + 1) == Some(&'[') => {
					index += 2;
					let start = index;
					while index < chars.len() && !('@'..='~').contains(&chars[index]) {
						index += 1;
					}
					assert!(index < chars.len(), "unterminated CSI sequence");
					let parameters: String = chars[start..index].iter().collect();
					self.apply_csi(&parameters, chars[index]);
					index += 1;
				},
				'\r' => {
					self.cursor_col = 0;
					self.pending_wrap = false;
					index += 1;
				},
				'\n' => {
					self.pending_wrap = false;
					self.line_feed();
					index += 1;
				},
				character if !character.is_control() => {
					let start = index;
					while index < chars.len() && !chars[index].is_control() {
						index += 1;
					}
					let run: String = chars[start..index].iter().collect();
					for grapheme in run.graphemes() {
						self.print(grapheme);
					}
				},
				_ => index += 1,
			}
		}
	}

	fn print(&mut self, grapheme: &str) {
		let width = grapheme.visible_width();
		if width == 0 {
			return;
		}
		if self.pending_wrap {
			self.pending_wrap = false;
			if self.autowrap {
				self.line_feed();
				self.cursor_col = 0;
				self.wrapped[self.cursor_row] = true;
			}
		}
		if self.cursor_col + width > self.width {
			return;
		}
		for offset in 0..width {
			self.clear_glyph_at(self.cursor_col + offset);
		}
		self.screen[self.cursor_row][self.cursor_col] = Some(grapheme.to_owned());
		if width >= 2 {
			for offset in 1..width {
				self.screen[self.cursor_row][self.cursor_col + offset] = None;
			}
		}
		let end = self.cursor_col + width;
		if end >= self.width {
			self.pending_wrap = true;
		}
		self.cursor_col = end.min(self.width - 1);
	}

	/// Blanks both halves of any wide glyph occupying `col`, as hardware
	/// terminals do when either half is overwritten.
	fn clear_glyph_at(&mut self, col: usize) {
		let row = &mut self.screen[self.cursor_row];
		if row[col].is_none() {
			row[col - 1] = Some(" ".to_owned());
			row[col] = Some(" ".to_owned());
			return;
		}
		if col + 1 < self.width && row[col + 1].is_none() {
			row[col + 1] = Some(" ".to_owned());
		}
	}

	fn apply_csi(&mut self, parameters: &str, command: char) {
		match command {
			'H' | 'f' => {
				let mut values = parameters.split(';');
				let row = values
					.next()
					.and_then(|value| value.parse::<usize>().ok())
					.unwrap_or(1);
				let column = values
					.next()
					.and_then(|value| value.parse::<usize>().ok())
					.unwrap_or(1);
				self.cursor_row = row.saturating_sub(1).min(self.height - 1);
				self.cursor_col = column.saturating_sub(1).min(self.width - 1);
				self.pending_wrap = false;
			},
			'h' if parameters == "?7" => {
				self.autowrap = true;
			},
			'l' if parameters == "?7" => {
				self.autowrap = false;
			},
			'A' => {
				let distance = parameters.parse::<usize>().unwrap_or(1);
				self.cursor_row = self.cursor_row.saturating_sub(distance);
				self.pending_wrap = false;
			},
			'B' => {
				let distance = parameters.parse::<usize>().unwrap_or(1);
				let limit = if self.cursor_row <= self.margin_bottom {
					self.margin_bottom
				} else {
					self.height - 1
				};
				self.cursor_row = self.cursor_row.saturating_add(distance).min(limit);
				self.pending_wrap = false;
			},
			'C' => {
				let distance = parameters.parse::<usize>().unwrap_or(1);
				self.cursor_col = self.cursor_col.saturating_add(distance).min(self.width - 1);
				self.pending_wrap = false;
			},
			'J' if parameters == "3" => {
				self.history.clear();
			},
			'K' if parameters.is_empty() || parameters == "2" => {
				self.screen[self.cursor_row] = Self::blank_row(self.width);
				self.wrapped[self.cursor_row] = false;
			},
			'J' if parameters == "2" => {
				for row in &mut self.screen {
					*row = Self::blank_row(self.width);
				}
				self.wrapped.fill(false);
			},
			'r' => {
				let mut values = parameters.split(';');
				let top = values
					.next()
					.and_then(|value| value.parse::<usize>().ok())
					.unwrap_or(1);
				let bottom = values
					.next()
					.and_then(|value| value.parse::<usize>().ok())
					.unwrap_or(self.height);
				let top = top.saturating_sub(1).min(self.height - 1);
				let bottom = bottom.saturating_sub(1).min(self.height - 1);
				assert!(top < bottom, "DECSTBM region must span at least two rows");
				self.margin_top = top;
				self.margin_bottom = bottom;
				self.cursor_row = 0;
				self.cursor_col = 0;
				self.pending_wrap = false;
			},
			_ => {},
		}
	}

	fn line_feed(&mut self) {
		if self.cursor_row == self.margin_bottom {
			let row = self.screen.remove(self.margin_top);
			let joined = self.wrapped.remove(self.margin_top);
			if self.margin_top == 0 {
				// A row continuing onto the (new) top row keeps its
				// trailing cells: they are real source content, which is
				// exactly what native copy preserves across a soft wrap.
				let continues = self.wrapped.first().copied().unwrap_or(false);
				let text = if continues {
					row.iter().flatten().map(String::as_str).collect()
				} else {
					Self::row_text(&row)
				};
				match self.history.last_mut() {
					Some(previous) if joined => previous.push_str(&text),
					_ => self.history.push(text),
				}
			}
			self
				.screen
				.insert(self.margin_bottom, Self::blank_row(self.width));
			self.wrapped.insert(self.margin_bottom, false);
			return;
		}
		if self.cursor_row + 1 < self.height {
			self.cursor_row += 1;
		}
	}

	/// Whether `row` was soft-wrapped from the line above by autowrap.
	pub fn row_wrapped(&self, row: usize) -> bool {
		self.wrapped.get(row).copied().unwrap_or(false)
	}

	/// Resets the model to a blank screen at new geometry.
	pub fn resize(&mut self, width: usize, height: usize) {
		self.width = width;
		self.height = height;
		self.cursor_row = 0;
		self.cursor_col = 0;
		self.margin_top = 0;
		self.margin_bottom = height - 1;
		self.screen = vec![Self::blank_row(width); height];
		self.wrapped = vec![false; height];
		self.pending_wrap = false;
	}

	fn row_text(row: &[Option<String>]) -> String {
		let mut text = String::new();
		for glyph in row.iter().flatten() {
			text.push_str(glyph);
		}
		text.trim_end().to_owned()
	}

	/// Returns the visible screen as right-trimmed row text, top to bottom.
	pub fn visible_rows(&self) -> Vec<String> {
		self.screen.iter().map(|row| Self::row_text(row)).collect()
	}
}
