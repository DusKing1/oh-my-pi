use omp_core::Str;

use crate::{
	component::{Component, PaintCtx, Slot, next_slot},
	context::UiContext,
	frame::{Rect, Style},
	props::Props,
	rich::{Pipeline, Prefix, RichSink, RichText, width_config_epoch},
};

/// The type of a line in a diff.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffKind {
	/// A file header or metadata line.
	Header,
	/// An unchanged context line.
	Context,
	/// An added line.
	Add,
	/// A removed line.
	Remove,
}

/// A single line in a diff.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffLine {
	/// The type of the diff line.
	pub kind: DiffKind,
	/// The text content of the diff line.
	pub text: Str,
}

/// A component that renders a diff with semantic styles.
pub struct DiffView {
	props:              Props,
	slot:               Slot,
	lines:              Vec<DiffLine>,
	rich:               RichText,
	rendered_lines:     usize,
	cached_width:       u16,
	cached_width_epoch: u64,
	cached_revision:    u64,
}

impl DiffView {
	/// Creates a new empty diff view.
	pub fn new() -> Self {
		Self {
			props:              Props::new(),
			slot:               next_slot(),
			lines:              Vec::new(),
			rich:               RichText::default(),
			rendered_lines:     0,
			cached_width:       0,
			cached_width_epoch: 0,
			cached_revision:    0,
		}
	}

	/// Appends a new line to the diff view.
	pub fn push(&mut self, kind: DiffKind, text: impl Into<Str>) {
		self.lines.push(DiffLine { kind, text: text.into() });
	}

	/// Clears all lines from the diff view.
	///
	/// Returns whether the view contained any lines before clearing.
	pub fn clear(&mut self) -> bool {
		if self.lines.is_empty() {
			return false;
		}
		self.lines.clear();
		self.rendered_lines = 0;
		true
	}

	/// Appends multiple lines to the diff view.
	///
	/// Returns whether any lines were added.
	pub fn extend(&mut self, lines: impl IntoIterator<Item = DiffLine>) -> bool {
		let start = self.lines.len();
		self.lines.extend(lines);
		self.lines.len() > start
	}

	/// Replaces all lines in the diff view.
	pub fn replace(&mut self, lines: Vec<DiffLine>) {
		self.lines = lines;
		self.rendered_lines = 0; // force full render
	}

	fn render(&mut self, ctx: &UiContext, width: u16) {
		let width = width.max(1);
		let width_epoch = width_config_epoch();
		let revision = ctx.revision;

		if self.cached_width == width
			&& self.cached_width_epoch == width_epoch
			&& self.cached_revision == revision
		{
			if self.rendered_lines == self.lines.len() {
				return;
			}
		} else {
			self.rich.clear();
			self.rendered_lines = 0;
			self.cached_width = width;
			self.cached_width_epoch = width_epoch;
			self.cached_revision = revision;
		}

		let (info, muted, ok, err) = (ctx.theme.info, ctx.theme.muted, ctx.theme.ok, ctx.theme.err);

		let mut p_header = Prefix::default();
		p_header.push(Style::new().fg(info).bold(), "  ");
		let mut p_context = Prefix::default();
		p_context.push(Style::new().fg(muted), "  ");
		let mut p_add = Prefix::default();
		p_add.push(Style::new().fg(ok), "+ ");
		let mut p_remove = Prefix::default();
		p_remove.push(Style::new().fg(err), "- ");

		let mut c_header = Prefix::default();
		c_header.push(Style::new().fg(info).bold(), "  ");
		let mut c_context = Prefix::default();
		c_context.push(Style::new().fg(muted), "  ");
		let mut c_add = Prefix::default();
		c_add.push(Style::new().fg(ok), "  ");
		let mut c_remove = Prefix::default();
		c_remove.push(Style::new().fg(err), "  ");

		let mut wrap = (&mut self.rich).wrap(width);

		for line in &self.lines[self.rendered_lines..] {
			let (prefix, cont, text_style) = match line.kind {
				DiffKind::Header => (&p_header, &c_header, Style::new().fg(info).bold()),
				DiffKind::Context => (&p_context, &c_context, Style::new().fg(ctx.theme.fg)),
				DiffKind::Add => (&p_add, &c_add, Style::new().fg(ok)),
				DiffKind::Remove => (&p_remove, &c_remove, Style::new().fg(err)),
			};

			let mut prefixed = (&mut wrap).prefixed(prefix, cont);
			for (index, physical_line) in line.text.split("\n").enumerate() {
				if index > 0 {
					prefixed.newline();
				}
				if !physical_line.is_empty() {
					prefixed.run(text_style, physical_line.as_str());
				}
			}
			prefixed.newline();
		}
		wrap.finish();
		self.rendered_lines = self.lines.len();
	}
}

impl Default for DiffView {
	fn default() -> Self {
		Self::new()
	}
}

impl Component for DiffView {
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
		(1, u16::MAX) // DiffView flows to any width
	}

	fn height(&mut self, ctx: &UiContext, width: u16) -> u16 {
		self.render(ctx, width);
		RichText::rows(&self.rich)
	}

	fn paint(&mut self, pc: &mut PaintCtx<'_>, rect: Rect) {
		self.render(pc.ctx, rect.width);
		crate::components::text::paint_rich(pc, rect, &self.rich, self.props.align());
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{
		UiContext,
		component::{Component, PaintCtx},
		frame::{Frame, Rect, Size},
		test_support::{frame_cell_style, frame_row_text},
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
	fn renders_mixed_hunks_with_semantic_styles() {
		let mut diff = DiffView::new();
		diff.push(DiffKind::Header, "src/main.rs");
		diff.push(DiffKind::Context, "fn main() {");
		diff.push(DiffKind::Remove, "    println!(\"Hello\");");
		diff.push(DiffKind::Add, "    println!(\"World\");");
		diff.push(DiffKind::Context, "}");

		let frame = paint(&mut diff, 40, 5);
		assert_eq!(frame_row_text(&frame, 0).trim_end(), "  src/main.rs");
		assert_eq!(frame_row_text(&frame, 1).trim_end(), "  fn main() {");
		assert_eq!(frame_row_text(&frame, 2).trim_end(), "-     println!(\"Hello\");");
		assert_eq!(frame_row_text(&frame, 3).trim_end(), "+     println!(\"World\");");
		assert_eq!(frame_row_text(&frame, 4).trim_end(), "  }");

		let ctx = UiContext::default();
		assert_eq!(frame_cell_style(&frame, 0, 0).foreground, ctx.theme.info);
		assert_eq!(frame_cell_style(&frame, 0, 1).foreground, ctx.theme.muted);
		assert_eq!(frame_cell_style(&frame, 0, 2).foreground, ctx.theme.err);
		assert_eq!(frame_cell_style(&frame, 0, 3).foreground, ctx.theme.ok);
	}

	#[test]
	fn incremental_replacement() {
		let mut diff = DiffView::new();
		diff.push(DiffKind::Add, "a");
		let frame1 = paint(&mut diff, 10, 2);
		assert_eq!(frame_row_text(&frame1, 0).trim_end(), "+ a");

		diff.push(DiffKind::Add, "b");
		let frame2 = paint(&mut diff, 10, 2);
		assert_eq!(frame_row_text(&frame2, 1).trim_end(), "+ b");

		diff.replace(vec![DiffLine { kind: DiffKind::Remove, text: Str::from("c") }]);
		let frame3 = paint(&mut diff, 10, 2);
		assert_eq!(frame_row_text(&frame3, 0).trim_end(), "- c");
		assert_eq!(frame_row_text(&frame3, 1).trim_end(), "");
	}

	#[test]
	fn unicode_clipping() {
		let mut diff = DiffView::new();
		diff.push(DiffKind::Add, "한글");
		let frame = paint(&mut diff, 5, 2);
		assert_eq!(frame_row_text(&frame, 0).trim_end(), "+ 한");
		assert_eq!(frame_row_text(&frame, 1).trim_end(), "  글");
	}

	#[test]
	
	fn paint_with_ctx(component: &mut dyn Component, mut ctx: UiContext, width: u16, height: u16) -> Frame {
		let mut frame = Frame::new(Size::new(width, height));
		let mut hits = Vec::new();
		let mut wakes = Vec::new();
		let mut pc = PaintCtx::new(&mut frame, &ctx, &mut hits, &mut wakes);
		component.paint(&mut pc, Rect::new(0, 0, width, height));
		frame
	}

	#[test]
	fn verifies_append_cache_matches_fresh_build() {
		let mut incremental = DiffView::new();
		let mut fresh = DiffView::new();
		let ctx = UiContext::default();

		// Incremental build
		incremental.push(DiffKind::Header, "file.txt");
		let _ = paint_with_ctx(&mut incremental, ctx.clone(), 20, 10);
		
		incremental.extend(vec![
			DiffLine { kind: DiffKind::Context, text: Str::from("line 1") },
			DiffLine { kind: DiffKind::Remove, text: Str::from("line 2") },
		]);
		let _ = paint_with_ctx(&mut incremental, ctx.clone(), 20, 10);

		incremental.push(DiffKind::Add, "line 3");
		let frame_incremental = paint_with_ctx(&mut incremental, ctx.clone(), 20, 10);

		// Fresh build
		fresh.extend(vec![
			DiffLine { kind: DiffKind::Header, text: Str::from("file.txt") },
			DiffLine { kind: DiffKind::Context, text: Str::from("line 1") },
			DiffLine { kind: DiffKind::Remove, text: Str::from("line 2") },
			DiffLine { kind: DiffKind::Add, text: Str::from("line 3") },
		]);
		let frame_fresh = paint_with_ctx(&mut fresh, ctx.clone(), 20, 10);

		// Verify identity
		assert_eq!(frame_row_text(&frame_incremental, 0), frame_row_text(&frame_fresh, 0));
		assert_eq!(frame_row_text(&frame_incremental, 1), frame_row_text(&frame_fresh, 1));
		assert_eq!(frame_row_text(&frame_incremental, 2), frame_row_text(&frame_fresh, 2));
		assert_eq!(frame_row_text(&frame_incremental, 3), frame_row_text(&frame_fresh, 3));
	}

	#[test]
	fn clear_and_extend_return_semantic_changes() {
		let mut diff = DiffView::new();
		assert!(!diff.clear()); // Empty to start
		assert!(diff.extend(vec![DiffLine { kind: DiffKind::Add, text: Str::from("x") }])); // Added something
		assert!(!diff.extend(vec![])); // Added nothing
		assert!(diff.clear()); // Cleared something
	}

	#[test]
	fn charset_routing() {
		use crate::context::Charset;
		
		let mut diff = DiffView::new();
		diff.push(DiffKind::Add, "test");
		
		let mut ctx_ascii = UiContext::default();
		ctx_ascii.charset = Charset::Ascii;
		
		let mut ctx_unicode = UiContext::default();
		ctx_unicode.charset = Charset::Unicode;

		let frame_ascii = paint_with_ctx(&mut diff, ctx_ascii, 10, 2);
		let frame_unicode = paint_with_ctx(&mut diff, ctx_unicode, 10, 2);

		assert_eq!(frame_row_text(&frame_ascii, 0).trim_end(), "+ test");
		assert_eq!(frame_row_text(&frame_unicode, 0).trim_end(), "+ test");
	}

	#[test]
	fn empty_diff() {
		let mut diff = DiffView::new();
		let frame = paint(&mut diff, 10, 2);
		assert_eq!(frame_row_text(&frame, 0).trim_end(), "");
	}
}
