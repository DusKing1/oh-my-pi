//! Width-aware Markdown rendering into styled terminal lines.

use omp_core::{Str, StrMut, fmts};
use smallvec::SmallVec;

use crate::{
	context::{Charset, Theme, UiContext},
	frame::{Color, Style},
	rich::{Pipeline, Prefix, RichSink, RichText, cell_width},
};

mod highlight;
mod inline;
mod mermaid;
mod table;

use inline::code_span_len;
pub(crate) use inline::{math_span, parse_inline};

/// Styles used by the Markdown renderer.
#[derive(Clone, Copy)]
pub struct MdTheme {
	/// Ordinary prose.
	pub base:        Style,
	/// Heading text.
	pub heading:     Style,
	/// Strong inline text.
	pub strong:      Style,
	/// Emphasized inline text.
	pub emphasis:    Style,
	/// Inline code spans.
	pub code:        Style,
	/// Code block text.
	pub code_block:  Style,
	/// Code block fences.
	pub code_border: Style,
	/// Blockquote rails and text.
	pub quote:       Style,
	/// Ordered and unordered list markers.
	pub bullet:      Style,
	/// Link labels.
	pub link:        Style,
	/// Horizontal rules.
	pub rule:        Style,
	highlight:       highlight::HighlightStyles,
	charset:         Charset,
	semantic:        Theme,
}

impl Default for MdTheme {
	fn default() -> Self {
		Self::from_theme(&crate::context::Theme::default())
	}
}

impl MdTheme {
	/// Derives Markdown styles from the shared semantic theme.
	pub const fn from_theme(theme: &Theme) -> Self {
		let code_block = Style::new().fg(theme.fg);
		Self {
			base: Style::new().fg(theme.fg),
			heading: Style::new().fg(theme.accent).bold(),
			strong: Style::new().fg(theme.fg).bold(),
			emphasis: Style::new().fg(theme.fg).italic(),
			code: Style::new().fg(theme.warn).bg(theme.hover),
			code_block,
			code_border: Style::new().fg(theme.border),
			quote: Style::new().fg(theme.muted).dim().italic(),
			bullet: Style::new().fg(theme.info),
			link: Style::new().fg(theme.accent).underline(),
			rule: Style::new().fg(theme.border),
			highlight: highlight::HighlightStyles::from_theme(theme),
			charset: Charset::Unicode,
			semantic: *theme,
		}
	}

	/// Derives Markdown styles and diagram glyphs from the presentation context.
	pub const fn from_context(context: &UiContext) -> Self {
		let mut theme = Self::from_theme(&context.theme);
		theme.charset = context.charset;
		theme
	}

	fn semantic_color(&self, name: &str) -> Option<Color> {
		self.semantic.token(name)
	}

	/// Folds a node's cascaded style into the palette: prose-family
	/// entries (`base`, `strong`, `emphasis`, `code_block`) adopt its
	/// foreground, every entry picks up its attribute flags, and semantic
	/// hues (headings, links, inline code, bullets) stay their own.
	/// Syntax-highlight token colors are a deliberate boundary and keep
	/// their palette untouched.
	pub fn cascade(mut self, style: Style) -> Self {
		if style == Style::default() {
			return self;
		}
		let flags = style.fg(Color::Default);
		self.base = style.inherit(self.base);
		self.strong = style.inherit(self.strong);
		self.emphasis = style.inherit(self.emphasis);
		self.code_block = style.inherit(self.code_block);
		self.heading = flags.inherit(self.heading);
		self.code = flags.inherit(self.code);
		self.code_border = flags.inherit(self.code_border);
		self.quote = flags.inherit(self.quote);
		self.bullet = flags.inherit(self.bullet);
		self.link = flags.inherit(self.link);
		self.rule = flags.inherit(self.rule);
		self
	}
}

/// GFM table cell alignment.
#[derive(Clone, Copy)]
pub(crate) enum Alignment {
	Left,
	Center,
	Right,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BlockKind {
	Paragraph,
	List,
	Other,
}
struct DocumentSink<'a> {
	inner:           &'a mut dyn RichSink,
	emitted:         bool,
	row_has_content: bool,
	pending_blank:   bool,
	final_blank:     bool,
}

impl RichSink for DocumentSink<'_> {
	fn run(&mut self, style: Style, text: &str) {
		if text.is_empty() {
			return;
		}
		if self.pending_blank {
			self.inner.newline();
			self.pending_blank = false;
		}
		self.emitted = true;
		self.row_has_content = true;
		self.final_blank = false;
		self.inner.run(style, text);
	}

	fn newline(&mut self) {
		self.emitted = true;
		if self.row_has_content {
			self.inner.newline();
			self.row_has_content = false;
			self.final_blank = false;
		} else {
			// Delay empty rows until later content proves they are not
			// trailing; repeated requests normalize to one blank row.
			self.pending_blank = true;
			self.final_blank = true;
		}
	}
}

/// Renders Markdown into rows no wider than `width` terminal cells.
pub fn render(src: &Str, width: u16, theme: &MdTheme, sink: &mut dyn RichSink) {
	let normalized = normalize_source(src);
	// degenerate viewports still make progress: every block renders at
	// one cell and the paint layer clips
	render_document(&normalized, width.max(1), theme, sink);
}

fn render_document(source: &Str, width: u16, theme: &MdTheme, sink: &mut dyn RichSink) {
	let mut lines: SmallVec<&str, 64> = source.as_str().split('\n').collect();
	while lines.last().is_some_and(|line| line.is_empty()) {
		lines.pop();
	}
	let mut tracked = DocumentSink {
		inner:           sink,
		emitted:         false,
		row_has_content: false,
		pending_blank:   false,
		final_blank:     false,
	};
	let mut index = 0;
	let mut previous = None;
	while index < lines.len() {
		let blank_start = index;
		while index < lines.len() && lines[index].trim().is_empty() {
			index += 1;
		}
		if index == lines.len() {
			break;
		}
		let had_blank = index > blank_start;
		let kind = block_kind(&lines, index);
		if tracked.emitted && should_separate(previous, kind, had_blank) && !tracked.final_blank {
			if tracked.row_has_content {
				tracked.newline();
			}
			tracked.newline();
		}

		if let Some((depth, text)) = atx_heading(lines[index]) {
			render_heading(text, depth, width, theme, &mut tracked);
			index += 1;
		} else if index + 1 < lines.len() {
			if let Some(depth) = setext_depth(lines[index + 1]) {
				render_heading(lines[index].trim(), depth, width, theme, &mut tracked);
				index += 2;
			} else if let Some((alignments, header)) = table_header(&lines, index) {
				let mut rows = vec![header];
				index += 2;
				while index < lines.len()
					&& lines[index].contains('|')
					&& !lines[index].trim().is_empty()
				{
					let cells = table_cells(lines[index]);
					if cells.len() != alignments.len() {
						break;
					}
					rows.push(cells);
					index += 1;
				}
				table::render_table(&rows, &alignments, width, theme, &mut tracked);
			} else {
				render_non_heading_block(&lines, &mut index, width, theme, &mut tracked);
			}
		} else {
			render_non_heading_block(&lines, &mut index, width, theme, &mut tracked);
		}
		previous = Some(kind);
	}
}

const fn should_separate(previous: Option<BlockKind>, current: BlockKind, had_blank: bool) -> bool {
	match (previous, current) {
		(None, _) => false,
		(Some(BlockKind::Paragraph), BlockKind::List) => false,
		(Some(BlockKind::List), _) => had_blank,
		_ => true,
	}
}

fn render_non_heading_block(
	lines: &[&str],
	index: &mut usize,
	width: u16,
	theme: &MdTheme,
	sink: &mut dyn RichSink,
) {
	if let Some(end) = bare_math_with_lhs_end(lines, *index) {
		let body = join_lines(&lines[*index..end]);
		render_math(body.as_str(), width, theme, sink);
		*index = end;
		return;
	}
	if let Some((fence, language)) = fence_start(lines[*index]) {
		render_fenced_code(lines, index, width, theme, fence, language, sink);
		return;
	}
	if is_indented_code(lines[*index]) {
		render_indented_code(lines, index, width, theme, sink);
		return;
	}
	if let Some(fill) = horizontal_rule(lines[*index]) {
		render_rule(fill, width, theme, sink);
		*index += 1;
		return;
	}
	if let Some((end, body)) = display_math(lines, *index) {
		render_math(body.as_str(), width, theme, sink);
		*index = end;
		return;
	}
	if let Some(end) = bare_math_end(lines, *index) {
		let body = join_lines(&lines[*index..end]);
		render_math(body.as_str(), width, theme, sink);
		*index = end;
		return;
	}
	if quote_line(lines[*index]).is_some() {
		render_blockquote(lines, index, width, theme, sink);
		return;
	}
	if let Some(marker) = list_marker(lines[*index]) {
		render_list(lines, index, width, theme, marker.indent, 0, marker.ordered, sink);
		return;
	}
	render_paragraph(lines, index, width, theme, sink);
}

fn block_kind(lines: &[&str], index: usize) -> BlockKind {
	if list_marker(lines[index]).is_some() {
		BlockKind::List
	} else if atx_heading(lines[index]).is_some()
		|| (index + 1 < lines.len() && setext_depth(lines[index + 1]).is_some())
		|| bare_math_with_lhs_end(lines, index).is_some()
		|| horizontal_rule(lines[index]).is_some()
		|| fence_start(lines[index]).is_some()
		|| is_indented_code(lines[index])
		|| quote_line(lines[index]).is_some()
		|| display_math(lines, index).is_some()
		|| bare_math_end(lines, index).is_some()
		|| table_header(lines, index).is_some()
	{
		BlockKind::Other
	} else {
		BlockKind::Paragraph
	}
}

fn render_heading(text: &str, depth: usize, width: u16, theme: &MdTheme, sink: &mut dyn RichSink) {
	let style = match depth {
		1 => theme.heading.bold().underline(),
		_ => theme.heading.bold(),
	};
	let mut wrap = (&mut *sink).wrap(width);
	if depth >= 3 {
		let mut prefix = StrMut::with_capacity(depth + 1);
		for _ in 0..depth {
			prefix.push('#');
		}
		prefix.push(' ');
		wrap.run(style, prefix.as_str());
	}
	parse_inline(text, theme, style, &mut wrap);
	wrap.finish();
}

fn atx_heading(line: &str) -> Option<(usize, &str)> {
	let trimmed = line.trim_start_matches(' ');
	if line.len().saturating_sub(trimmed.len()) > 3 {
		return None;
	}
	let depth = trimmed.bytes().take_while(|byte| *byte == b'#').count();
	if !(1..=6).contains(&depth) {
		return None;
	}
	let rest = &trimmed[depth..];
	if !rest.is_empty() && !rest.starts_with(' ') && !rest.starts_with('\t') {
		return None;
	}
	let mut text = rest.trim();
	if text.ends_with('#') {
		text = text.trim_end_matches('#').trim_end();
	}
	Some((depth, text))
}

fn setext_depth(line: &str) -> Option<usize> {
	let trimmed = line.trim();
	if trimmed.is_empty()
		|| line
			.len()
			.saturating_sub(line.trim_start_matches(' ').len())
			> 3
	{
		return None;
	}
	if trimmed.bytes().all(|byte| byte == b'=') {
		Some(1)
	} else if trimmed.bytes().all(|byte| byte == b'-') {
		Some(2)
	} else {
		None
	}
}

fn horizontal_rule(line: &str) -> Option<char> {
	let trimmed = line.trim_start_matches(' ');
	if line.len().saturating_sub(trimmed.len()) > 3 {
		return None;
	}
	let mut compact = trimmed
		.chars()
		.filter(|character| !matches!(character, ' ' | '\t'));
	let first = compact.next()?;
	if !matches!(first, '-' | '*' | '_' | '=' | '─' | '━' | '═' | '–' | '—') {
		return None;
	}
	let mut count = 1;
	for character in compact {
		if character != first {
			return None;
		}
		count += 1;
	}
	(count >= 3).then_some(match first {
		'=' => '=',
		'═' => '═',
		'━' => '━',
		'─' => '─',
		'–' => '–',
		'—' => '—',
		_ => '─',
	})
}

fn render_rule(fill: char, width: u16, theme: &MdTheme, sink: &mut dyn RichSink) {
	let fill = theme.charset.rule_fill(fill);
	let count = usize::from(width.min(80));
	if count > 0 {
		let text = repeated_char(fill, count);
		sink.run(theme.rule, text.as_str());
	}
	sink.newline();
}

fn clipped_row(sink: &mut dyn RichSink, width: u16, style: Style, text: &str) {
	let mut clipped = (&mut *sink).clip(width, None);
	clipped.run(style, text);
	clipped.newline();
}

fn fence_start(line: &str) -> Option<(char, &str)> {
	let trimmed = line.trim_start_matches(' ');
	if line.len().saturating_sub(trimmed.len()) > 3 {
		return None;
	}
	let fence = trimmed.chars().next()?;
	if !matches!(fence, '`' | '~') {
		return None;
	}
	let count = trimmed
		.chars()
		.take_while(|character| *character == fence)
		.count();
	(count >= 3).then(|| (fence, trimmed[count..].trim()))
}

fn fence_language(info: &str) -> &str {
	info.split_ascii_whitespace().next().unwrap_or("")
}

fn is_mermaid_language(info: &str) -> bool {
	fence_language(info).eq_ignore_ascii_case("mermaid")
}

fn is_closing_fence(line: &str, fence: char, opening_count: usize) -> bool {
	let candidate = line.trim_start_matches(' ');
	let close_count = candidate
		.chars()
		.take_while(|character| *character == fence)
		.count();
	close_count >= opening_count && candidate[close_count..].trim().is_empty()
}

fn render_fenced_code(
	lines: &[&str],
	index: &mut usize,
	width: u16,
	theme: &MdTheme,
	fence: char,
	language: &str,
	sink: &mut dyn RichSink,
) {
	let opening_count = lines[*index]
		.trim_start()
		.chars()
		.take_while(|character| *character == fence)
		.count();
	let syntax = fence_language(language);
	let highlighted = if is_mermaid_language(language) || highlight::supports_language(syntax) {
		let body_start = *index + 1;
		let body_end = (body_start..lines.len())
			.find(|candidate| is_closing_fence(lines[*candidate], fence, opening_count))
			.unwrap_or(lines.len());
		let body = join_lines(&lines[body_start..body_end]);
		let after = after_fence(body_end, lines.len());
		if is_mermaid_language(language) {
			let styles = mermaid::MermaidStyles {
				text:   theme.base,
				line:   Style::new().fg(theme.semantic.muted),
				accent: theme.bullet,
			};
			if mermaid::render(body.as_str(), width, theme.charset, styles, sink) {
				*index = after;
				return;
			}
		}
		let mut rows = RichText::default();
		highlight::render(body.as_str(), syntax, body_end - body_start, &theme.highlight, &mut rows)
			.then_some((rows, after))
	} else {
		None
	};

	let top = if language.is_empty() {
		Str::new_static("```")
	} else {
		fmts!("```{language}")
	};
	clipped_row(sink, width, theme.code_border, top.as_str());

	if let Some((rows, after)) = highlighted {
		push_highlighted_code_rows(&rows, width, theme, Prefix::empty_ref(), sink);
		clipped_row(sink, width, theme.code_border, "```");
		*index = after;
		return;
	}

	*index += 1;
	while *index < lines.len() {
		if is_closing_fence(lines[*index], fence, opening_count) {
			*index += 1;
			clipped_row(sink, width, theme.code_border, "```");
			return;
		}
		push_code_line(lines[*index], width, theme, sink);
		*index += 1;
	}
	clipped_row(sink, width, theme.code_border, "```");
}

fn is_indented_code(line: &str) -> bool {
	line.starts_with("    ") && !line.trim().is_empty()
}

fn render_indented_code(
	lines: &[&str],
	index: &mut usize,
	width: u16,
	theme: &MdTheme,
	sink: &mut dyn RichSink,
) {
	clipped_row(sink, width, theme.code_border, "```");
	while *index < lines.len() {
		if let Some(body) = lines[*index].strip_prefix("    ") {
			push_code_line(body, width, theme, sink);
			*index += 1;
		} else if lines[*index].trim().is_empty()
			&& *index + 1 < lines.len()
			&& lines[*index + 1].starts_with("    ")
		{
			push_code_line("", width, theme, sink);
			*index += 1;
		} else {
			break;
		}
	}
	clipped_row(sink, width, theme.code_border, "```");
}

fn push_code_line(body: &str, width: u16, theme: &MdTheme, sink: &mut dyn RichSink) {
	let mut clipped = (&mut *sink).clip(width, None);
	clipped.run(theme.code_block, "  ");
	clipped.run(theme.code_block, body);
	clipped.newline();
}

fn push_highlighted_code_rows(
	rows: &RichText,
	width: u16,
	theme: &MdTheme,
	prefix: &Prefix,
	sink: &mut dyn RichSink,
) {
	let mut gutter = Prefix::default();
	gutter.push(theme.code_block, "  ");
	let clipped = (&mut *sink).clip(width, None);
	let listed = clipped.prefixed(prefix, prefix);
	let mut guttered = listed.prefixed(&gutter, &gutter);
	rows.replay(&mut guttered);
}

const fn after_fence(end: usize, line_count: usize) -> usize {
	if end < line_count { end + 1 } else { end }
}

fn display_math(lines: &[&str], index: usize) -> Option<(usize, Str)> {
	let opening = lines[index].trim();
	let closing = match opening {
		"$$" => "$$",
		"\\[" => "\\]",
		_ => return None,
	};
	let mut end = index + 1;
	while end < lines.len() && lines[end].trim() != closing {
		end += 1;
	}
	if end >= lines.len() || end == index + 1 {
		return None;
	}
	let body = join_lines(&lines[index + 1..end]);
	(!body.trim().is_empty()).then_some((end + 1, body))
}

fn bare_math_end(lines: &[&str], index: usize) -> Option<usize> {
	let line = lines.get(index)?;
	let line = line.trim_start_matches([' ', '\t']);
	if lines[index].len().saturating_sub(line.len()) > 3 {
		return None;
	}
	let rest = line.strip_prefix("\\begin{")?;
	let close = rest.find('}')?;
	let environment = &rest[..close];
	if !crate::latex::is_bare_math_environment(environment) {
		return None;
	}
	let end_token = fmts!("\\end{{{environment}}}");
	for (offset, candidate) in lines[index..].iter().enumerate() {
		if offset > 0 && candidate.trim().is_empty() {
			return None;
		}
		if candidate.contains(end_token.as_str()) {
			return Some(index + offset + 1);
		}
	}
	None
}

fn bare_math_with_lhs_end(lines: &[&str], index: usize) -> Option<usize> {
	let lhs = lines.get(index)?.trim_end();
	let last = lhs.chars().last()?;
	if !matches!(last, '=' | '(' | '[' | '{') {
		return None;
	}
	if index + 1 >= lines.len() {
		return None;
	}
	bare_math_end(lines, index + 1)
}

fn render_math(body: &str, width: u16, theme: &MdTheme, sink: &mut dyn RichSink) {
	{
		let mut clipped = (&mut *sink).clip(width, None);
		if crate::latex::latex_block(body, theme.base, &mut clipped) {
			return;
		}
	}
	let mut wrap = (&mut *sink).wrap(width);
	crate::latex::latex_inline(body, theme.base, &mut wrap);
	wrap.finish();
}

fn sole_display_math(text: &str) -> Option<&str> {
	let trimmed = text.trim();
	if let Some(body) = trimmed
		.strip_prefix("$$")
		.and_then(|body| body.strip_suffix("$$"))
	{
		return (!body.trim().is_empty()).then_some(body);
	}
	trimmed
		.strip_prefix("\\[")
		.and_then(|body| body.strip_suffix("\\]"))
}

fn quote_line(line: &str) -> Option<&str> {
	let trimmed = line.trim_start_matches(' ');
	if line.len().saturating_sub(trimmed.len()) > 3 {
		return None;
	}
	trimmed
		.strip_prefix('>')
		.map(|text| text.strip_prefix(' ').unwrap_or(text))
}

fn render_blockquote(
	lines: &[&str],
	index: &mut usize,
	width: u16,
	theme: &MdTheme,
	sink: &mut dyn RichSink,
) {
	let mut inner = StrMut::new("");
	let mut saw_quote = false;
	while *index < lines.len() {
		if let Some(text) = quote_line(lines[*index]) {
			if saw_quote {
				inner.push('\n');
			}
			inner.push_str(text);
			saw_quote = true;
			*index += 1;
			continue;
		}
		if saw_quote
			&& !lines[*index].trim().is_empty()
			&& block_kind(lines, *index) == BlockKind::Paragraph
		{
			inner.push('\n');
			inner.push_str(lines[*index]);
			*index += 1;
			continue;
		}
		break;
	}
	let mut quote_theme = *theme;
	quote_theme.base = theme.quote.italic();
	quote_theme.strong = theme.strong.italic();
	quote_theme.emphasis = theme.emphasis.italic();
	quote_theme.code = theme.code.italic();
	quote_theme.link = theme.link.italic();
	let inner = inner.freeze();
	let mut rendered = RichText::default();
	render_document(&inner, width.saturating_sub(2).max(1), &quote_theme, &mut rendered);
	let mut rail = Prefix::default();
	rail.push(theme.quote, theme.charset.quote_rail());
	let clipped = (&mut *sink).clip(width, None);
	let mut bordered = clipped.prefixed(&rail, &rail);
	rendered.replay(&mut bordered);
}

#[derive(Clone, Copy)]
struct ListMarker<'a> {
	indent:  usize,
	ordered: bool,
	start:   usize,
	text:    &'a str,
}

fn list_marker(line: &str) -> Option<ListMarker<'_>> {
	let spaces = line.bytes().take_while(|byte| *byte == b' ').count();
	let rest = &line[spaces..];
	if let Some(text) = rest
		.strip_prefix("- ")
		.or_else(|| rest.strip_prefix("* "))
		.or_else(|| rest.strip_prefix("+ "))
	{
		return Some(ListMarker { indent: spaces, ordered: false, start: 1, text });
	}
	if matches!(rest, "-" | "*" | "+") {
		return Some(ListMarker { indent: spaces, ordered: false, start: 1, text: "" });
	}
	let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
	if digits == 0 || digits > 9 {
		return None;
	}
	let delimiter = rest.as_bytes().get(digits).copied()?;
	if !matches!(delimiter, b'.' | b')') {
		return None;
	}
	let after = &rest[digits + 1..];
	let text = if after.is_empty() {
		""
	} else {
		after
			.strip_prefix(' ')
			.or_else(|| after.strip_prefix('\t'))?
	};
	Some(ListMarker { indent: spaces, ordered: true, start: rest[..digits].parse().ok()?, text })
}

fn push_prefix_spaces(prefix: &mut Prefix, style: Style, mut count: usize) {
	while count > 0 {
		let take = count.min(crate::rich::SPACES.len());
		prefix.push(style, &crate::rich::SPACES[..take]);
		count -= take;
	}
}

fn render_list(
	lines: &[&str],
	index: &mut usize,
	width: u16,
	theme: &MdTheme,
	root_indent: usize,
	depth: usize,
	ordered: bool,
	sink: &mut dyn RichSink,
) {
	let start = list_marker(lines[*index]).map_or(1, |marker| marker.start);
	let mut ordinal = start;
	while *index < lines.len() {
		let Some(marker) = list_marker(lines[*index]) else {
			break;
		};
		if marker.indent != root_indent || marker.ordered != ordered {
			break;
		}
		let marker_text = if ordered {
			let value = ordinal;
			ordinal = ordinal.saturating_add(1);
			fmts!("{value}. ")
		} else {
			Str::new_static("- ")
		};
		let indent = depth.saturating_mul(2);
		let mut first_prefix = Prefix::default();
		push_prefix_spaces(&mut first_prefix, theme.base, indent);
		first_prefix.push(theme.bullet, marker_text.as_str());
		let continuation_width = usize::from(cell_width(marker_text.as_str()));
		let mut continuation = Prefix::default();
		push_prefix_spaces(&mut continuation, theme.base, indent.saturating_add(continuation_width));
		if let Some((fence, language)) = fence_start(marker.text) {
			render_list_fenced_code(
				lines,
				index,
				width,
				theme,
				fence,
				language,
				&first_prefix,
				&continuation,
				root_indent,
				sink,
			);
		} else {
			render_list_text(marker.text, width, theme, &first_prefix, &continuation, sink);
			*index += 1;
		}
		loop {
			if *index >= lines.len() {
				break;
			}
			if lines[*index].trim().is_empty() {
				let blank_at = *index;
				while *index < lines.len() && lines[*index].trim().is_empty() {
					*index += 1;
				}
				if *index < lines.len() {
					let next_indent = lines[*index]
						.bytes()
						.take_while(|byte| *byte == b' ')
						.count();
					if list_marker(lines[*index]).is_some_and(|next| next.indent >= root_indent)
						|| next_indent > root_indent
					{
						// pi renders loose lists tight: blank lines between
						// items or item paragraphs never survive
						continue;
					}
				}
				*index = blank_at;
				break;
			}
			let Some(next_marker) = list_marker(lines[*index]) else {
				let leading = lines[*index]
					.bytes()
					.take_while(|byte| *byte == b' ')
					.count();
				if leading <= root_indent && starts_block(lines, *index) {
					break;
				}
				let text = trim_list_continuation(lines[*index], root_indent);
				if let Some((fence, language)) = fence_start(text) {
					render_list_fenced_code(
						lines,
						index,
						width,
						theme,
						fence,
						language,
						&continuation,
						&continuation,
						root_indent,
						sink,
					);
				} else {
					render_list_text(text, width, theme, &continuation, &continuation, sink);
					*index += 1;
				}
				continue;
			};
			if next_marker.indent == root_indent && next_marker.ordered == ordered {
				break;
			}
			if next_marker.indent <= root_indent {
				break;
			}
			render_list(
				lines,
				index,
				width,
				theme,
				next_marker.indent,
				depth + 1,
				next_marker.ordered,
				sink,
			);
		}
	}
}

fn render_list_fenced_code(
	lines: &[&str],
	index: &mut usize,
	width: u16,
	theme: &MdTheme,
	fence: char,
	language: &str,
	first_prefix: &Prefix,
	continuation: &Prefix,
	root_indent: usize,
	sink: &mut dyn RichSink,
) {
	let opening_count = 3;
	let syntax = fence_language(language);
	let highlighted = if is_mermaid_language(language) || highlight::supports_language(syntax) {
		let body_start = *index + 1;
		let body_end = (body_start..lines.len())
			.find(|candidate| {
				is_closing_fence(
					trim_list_continuation(lines[*candidate], root_indent),
					fence,
					opening_count,
				)
			})
			.unwrap_or(lines.len());
		let body = join_list_fence_lines(lines, body_start, body_end, root_indent);
		let after = after_fence(body_end, lines.len());
		if is_mermaid_language(language) {
			let styles = mermaid::MermaidStyles {
				text:   theme.base,
				line:   Style::new().fg(theme.semantic.muted),
				accent: theme.bullet,
			};
			let content_width = width.saturating_sub(continuation.width());
			let clipped = (&mut *sink).clip(width, None);
			let mut prefixed = clipped.prefixed(first_prefix, continuation);
			if mermaid::render(body.as_str(), content_width, theme.charset, styles, &mut prefixed) {
				*index = after;
				return;
			}
		}
		let mut rows = RichText::default();
		highlight::render(body.as_str(), syntax, body_end - body_start, &theme.highlight, &mut rows)
			.then_some((rows, after))
	} else {
		None
	};

	let top = if language.is_empty() {
		Str::new_static("```")
	} else {
		fmts!("```{language}")
	};
	{
		let clipped = (&mut *sink).clip(width, None);
		let mut prefixed = clipped.prefixed(first_prefix, continuation);
		prefixed.run(theme.code_border, top.as_str());
		prefixed.newline();
	}

	if let Some((rows, after)) = highlighted {
		push_highlighted_code_rows(&rows, width, theme, continuation, sink);
		let clipped = (&mut *sink).clip(width, None);
		let mut prefixed = clipped.prefixed(continuation, continuation);
		prefixed.run(theme.code_border, "```");
		prefixed.newline();
		*index = after;
		return;
	}

	*index += 1;
	while *index < lines.len() {
		let body = trim_list_continuation(lines[*index], root_indent);
		if is_closing_fence(body, fence, opening_count) {
			*index += 1;
			let clipped = (&mut *sink).clip(width, None);
			let mut prefixed = clipped.prefixed(continuation, continuation);
			prefixed.run(theme.code_border, "```");
			prefixed.newline();
			return;
		}
		let clipped = (&mut *sink).clip(width, None);
		let mut prefixed = clipped.prefixed(continuation, continuation);
		prefixed.run(theme.code_block, "  ");
		prefixed.run(theme.code_block, body);
		prefixed.newline();
		*index += 1;
	}
	let clipped = (&mut *sink).clip(width, None);
	let mut prefixed = clipped.prefixed(continuation, continuation);
	prefixed.run(theme.code_border, "```");
	prefixed.newline();
}

fn trim_list_continuation(line: &str, root_indent: usize) -> &str {
	let wanted = root_indent.saturating_add(2);
	let spaces = line.bytes().take_while(|byte| *byte == b' ').count();
	&line[spaces.min(wanted)..]
}

fn join_list_fence_lines(lines: &[&str], start: usize, end: usize, root_indent: usize) -> Str {
	let slice = &lines[start..end];
	let capacity = slice
		.iter()
		.map(|line| trim_list_continuation(line, root_indent).len() + 1)
		.sum();
	let mut joined = StrMut::with_capacity(capacity);
	for (index, line) in slice.iter().enumerate() {
		if index > 0 {
			joined.push('\n');
		}
		joined.push_str(trim_list_continuation(line, root_indent));
	}
	joined.freeze()
}

fn render_list_text(
	text: &str,
	width: u16,
	theme: &MdTheme,
	first: &Prefix,
	continuation: &Prefix,
	sink: &mut dyn RichSink,
) {
	if text.is_empty() {
		let clipped = (&mut *sink).clip(width, None);
		let mut prefixed = clipped.prefixed(first, continuation);
		prefixed.newline();
		return;
	}
	if let Some(body) = sole_display_math(text) {
		{
			let clipped = (&mut *sink).clip(width, None);
			let mut prefixed = clipped.prefixed(first, continuation);
			if crate::latex::latex_block(body, theme.base, &mut prefixed) {
				return;
			}
		}
		let mut wrapped = (&mut *sink).wrap_prefixed(width, first, continuation);
		crate::latex::latex_inline(body, theme.base, &mut wrapped);
		wrapped.finish();
		return;
	}
	let mut wrapped = (&mut *sink).wrap_prefixed(width, first, continuation);
	parse_inline(text.trim_end(), theme, theme.base, &mut wrapped);
	wrapped.finish();
}

fn starts_block(lines: &[&str], index: usize) -> bool {
	atx_heading(lines[index]).is_some()
		|| horizontal_rule(lines[index]).is_some()
		|| fence_start(lines[index]).is_some()
		|| is_indented_code(lines[index])
		|| quote_line(lines[index]).is_some()
		|| list_marker(lines[index]).is_some()
		|| display_math(lines, index).is_some()
		|| bare_math_end(lines, index).is_some()
		|| bare_math_with_lhs_end(lines, index).is_some()
		|| table_header(lines, index).is_some()
}

fn render_paragraph(
	lines: &[&str],
	index: &mut usize,
	width: u16,
	theme: &MdTheme,
	sink: &mut dyn RichSink,
) {
	let start = *index;
	while *index < lines.len() && !lines[*index].trim().is_empty() {
		if *index > start && starts_block(lines, *index) {
			break;
		}
		if *index == start && *index + 1 < lines.len() && setext_depth(lines[*index + 1]).is_some() {
			break;
		}
		*index += 1;
	}
	if *index == start {
		*index += 1;
	}
	let paragraph = &lines[start..*index];
	if paragraph.len() == 1
		&& let Some(body) = sole_display_math(paragraph[0])
	{
		render_math(body, width, theme, sink);
		return;
	}
	for text in paragraph {
		let mut wrapped = (&mut *sink).wrap(width);
		parse_inline(text.trim_end(), theme, theme.base, &mut wrapped);
		wrapped.finish();
	}
}

fn table_header<'a>(lines: &[&'a str], index: usize) -> Option<(Vec<Alignment>, Vec<&'a str>)> {
	if index + 1 >= lines.len() || !lines[index].contains('|') {
		return None;
	}
	let alignments = table_separator(lines[index + 1])?;
	let header = table_cells(lines[index]);
	(header.len() == alignments.len() && !header.is_empty()).then_some((alignments, header))
}

fn table_cells(line: &str) -> Vec<&str> {
	let trimmed = line.trim();
	let body = trimmed.strip_prefix('|').unwrap_or(trimmed);
	let body = body.strip_suffix('|').unwrap_or(body);
	body.split('|').map(str::trim).collect()
}

fn table_separator(line: &str) -> Option<Vec<Alignment>> {
	if !line.contains('|') {
		return None;
	}
	let cells = table_cells(line);
	if cells.is_empty() {
		return None;
	}
	cells
		.into_iter()
		.map(|cell| {
			let left = cell.starts_with(':');
			let right = cell.ends_with(':');
			let dashes = cell.trim_matches(':');
			if dashes.len() < 3 || !dashes.bytes().all(|byte| byte == b'-') {
				return None;
			}
			Some(match (left, right) {
				(true, true) => Alignment::Center,
				(false, true) => Alignment::Right,
				_ => Alignment::Left,
			})
		})
		.collect()
}

fn repeated_char(character: char, count: usize) -> Str {
	let mut output = StrMut::with_capacity(count.saturating_mul(character.len_utf8()));
	for _ in 0..count {
		output.push(character);
	}
	output.freeze()
}

fn join_lines(lines: &[&str]) -> Str {
	let capacity = lines.iter().map(|line| line.len() + 1).sum();
	let mut joined = StrMut::with_capacity(capacity);
	for (index, line) in lines.iter().enumerate() {
		if index > 0 {
			joined.push('\n');
		}
		joined.push_str(line);
	}
	joined.freeze()
}

fn normalize_source(source: &Str) -> Str {
	if !source.as_str().contains('\t')
		&& !source.as_str().contains('<')
		&& !source.as_str().contains('&')
	{
		return source.clone();
	}
	let tabs = replace_tabs(source);
	if !tabs.as_str().contains('<') && !tabs.as_str().contains('&') {
		return tabs;
	}
	let text = tabs.as_str();
	let mut output = StrMut::with_capacity(text.len());
	let mut outside_start = 0;
	let mut in_fence: Option<(char, usize)> = None;
	for line in text.split_inclusive('\n') {
		let offset = line.as_ptr() as usize - text.as_ptr() as usize;
		let fence = fence_start(line.trim_end_matches('\n'));
		if let Some((active, opening_count)) = in_fence {
			output.push_str(line);
			let closes = fence.is_some_and(|(candidate, language)| {
				candidate == active
					&& language.is_empty()
					&& line
						.trim_start()
						.chars()
						.take_while(|character| *character == candidate)
						.count() >= opening_count
			});
			if closes {
				in_fence = None;
				outside_start = offset + line.len();
			}
			continue;
		}
		if let Some((opening, _)) = fence {
			if outside_start < offset {
				normalize_html_chunk(&text[outside_start..offset], &mut output);
			}
			output.push_str(line);
			let opening_count = line
				.trim_start()
				.chars()
				.take_while(|character| *character == opening)
				.count();
			in_fence = Some((opening, opening_count));
		}
	}
	if in_fence.is_none() && outside_start < text.len() {
		normalize_html_chunk(&text[outside_start..], &mut output);
	}
	output.freeze()
}

fn replace_tabs(source: &Str) -> Str {
	if !source.as_str().contains('\t') {
		return source.clone();
	}
	let tab_count = source
		.as_str()
		.bytes()
		.filter(|byte| *byte == b'\t')
		.count();
	let mut output = StrMut::with_capacity(source.len().saturating_add(tab_count.saturating_mul(2)));
	let mut start = 0;
	for (offset, character) in source.as_str().char_indices() {
		if character == '\t' {
			output.push_str(&source.as_str()[start..offset]);
			output.push_str("   ");
			start = offset + 1;
		}
	}
	output.push_str(&source.as_str()[start..]);
	output.freeze()
}

#[derive(Clone, Copy)]
struct HtmlList {
	ordered: bool,
	next:    usize,
}

fn normalize_html_chunk(raw: &str, output: &mut StrMut) {
	let mut lists: SmallVec<HtmlList, 4> = SmallVec::new();
	let mut quote_depth = 0_usize;
	let mut cursor = 0;
	while cursor < raw.len() {
		// An inline code span keeps its contents verbatim: tags between
		// matching backtick runs are code, not HTML for this pass.
		if raw.as_bytes()[cursor] == b'`' {
			let run = raw[cursor..]
				.bytes()
				.take_while(|byte| *byte == b'`')
				.count();
			let len = code_span_len(&raw[cursor..]).unwrap_or(run);
			decode_html_text(&raw[cursor..cursor + len], output, quote_depth);
			cursor += len;
			continue;
		}
		if raw[cursor..].starts_with("<!--") {
			if let Some(end) = raw[cursor + 4..].find("-->") {
				cursor += end + 7;
				continue;
			}
			decode_html_text(&raw[cursor..], output, quote_depth);
			break;
		}
		// An odd backslash run escapes the `<`: the tag is markdown text.
		let escaped = raw[..cursor]
			.bytes()
			.rev()
			.take_while(|byte| *byte == b'\\')
			.count()
			% 2 == 1;
		if raw.as_bytes()[cursor] == b'<'
			&& !escaped
			&& let Some(close) = raw[cursor..].find('>')
		{
			let tag = &raw[cursor..=(cursor + close)];
			if let Some((name, closing, self_closing)) = html_tag(tag) {
				let line_start = raw[..cursor].rfind('\n').map_or(0, |at| at + 1);
				let line_end = raw[cursor..].find('\n').map_or(raw.len(), |at| cursor + at);
				let in_table = raw[line_start..line_end].contains('|');
				// A four-space-indented line is an indented code block:
				// its tags stay literal.
				let indented = raw[line_start..]
					.bytes()
					.take_while(|byte| *byte == b' ')
					.count() >= 4;
				if indented {
					output.push_str(tag);
					cursor += tag.len();
					continue;
				}
				if in_table && matches!(name, "br" | "hr" | "p" | "ol" | "ul" | "li" | "code") {
					output.push_str(tag);
					cursor += tag.len();
					continue;
				}
				match name {
					// spans carry style attributes: kept verbatim for the
					// inline layer (`html_span`); bare `<text>` is a no-op
					"span" => output.push_str(tag),
					"text" => {},
					"code" => output.push_str(tag),
					"br" => append_html_break(output, true, quote_depth),
					"hr" => {
						// spaces/tabs only around the tag on its line: a
						// standalone `<hr>` becomes a thematic break
						let before = raw[..cursor].trim_end_matches([' ', '\t']);
						let after = raw[cursor + tag.len()..].trim_start_matches([' ', '\t']);
						let standalone = (before.is_empty() || before.ends_with('\n'))
							&& (after.is_empty() || after.starts_with('\n'));
						if standalone {
							append_html_break(output, false, quote_depth);
							output.push_str("---");
						}
						append_html_break(output, true, quote_depth);
					},
					"p" => append_html_break(output, closing, quote_depth),
					"blockquote" => {
						if closing {
							append_html_break(output, false, quote_depth);
							quote_depth = quote_depth.saturating_sub(1);
						} else if !self_closing {
							append_html_break(output, false, quote_depth);
							quote_depth += 1;
							append_quote_prefix(output, quote_depth);
						}
					},
					"ol" | "ul" => {
						if closing {
							lists.pop();
						} else if !self_closing {
							lists.push(HtmlList {
								ordered: name == "ol",
								next:    if name == "ol" { html_ol_start(tag) } else { 1 },
							});
						}
					},
					"li" => {
						append_html_break(output, false, quote_depth);
						if !closing {
							let depth = lists.len().saturating_sub(1);
							for _ in 0..depth {
								output.push_str("  ");
							}
							if let Some(list) = lists.last_mut() {
								if list.ordered {
									output.push_str(fmts!("{}. ", list.next).as_str());
									list.next = list.next.saturating_add(1);
								} else {
									output.push_str("- ");
								}
							} else {
								output.push_str("- ");
							}
						}
					},
					_ => output.push_str(tag),
				}
				cursor += tag.len();
				continue;
			}
		}
		let next_tag = raw[cursor..].find('<').map_or(raw.len(), |at| cursor + at);
		let next_span = raw[cursor..].find('`').map_or(raw.len(), |at| cursor + at);
		let boundary = next_tag.min(next_span);
		let end = if boundary == cursor {
			raw[cursor..]
				.chars()
				.next()
				.map_or(raw.len(), |character| cursor + character.len_utf8())
		} else {
			boundary
		};
		let text = &raw[cursor..end];
		if !(text.trim().is_empty() && text.contains('\n') && !lists.is_empty()) {
			decode_html_text(text, output, quote_depth);
		}
		cursor = end;
	}
}

fn decode_html_text(text: &str, output: &mut StrMut, quote_depth: usize) {
	if quote_depth == 0 || !text.contains('\n') {
		decode_entities(text, output);
		return;
	}
	let mut decoded = StrMut::with_capacity(text.len());
	decode_entities(text, &mut decoded);
	for chunk in decoded.as_str().split_inclusive('\n') {
		output.push_str(chunk);
		if chunk.ends_with('\n') {
			append_quote_prefix(output, quote_depth);
		}
	}
}

/// The HTML tag names Markdown normalizes itself.
const HTML_TAGS: &[&str] =
	&["br", "p", "ol", "ul", "li", "span", "text", "code", "hr", "blockquote"];

fn html_tag(tag: &str) -> Option<(&str, bool, bool)> {
	let inside = tag.strip_prefix('<')?.strip_suffix('>')?.trim();
	let closing = inside.starts_with('/');
	let body = inside.strip_prefix('/').unwrap_or(inside).trim_start();
	let name_end = body
		.find(|character: char| character.is_whitespace() || character == '/')
		.unwrap_or(body.len());
	let name = &body[..name_end];
	let canonical = HTML_TAGS
		.iter()
		.find(|candidate| name.eq_ignore_ascii_case(candidate))?;
	Some((canonical, closing, body.trim_end().ends_with('/')))
}

fn html_ol_start(tag: &str) -> usize {
	let lower = tag.to_ascii_lowercase();
	let Some(at) = lower.find("start") else {
		return 1;
	};
	let rest = lower[at + 5..].trim_start();
	let Some(rest) = rest.strip_prefix('=') else {
		return 1;
	};
	let rest = rest.trim_start();
	let rest = rest
		.strip_prefix('\'')
		.or_else(|| rest.strip_prefix('"'))
		.unwrap_or(rest);
	let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
	rest[..digits].parse().unwrap_or(1)
}

fn append_html_break(output: &mut StrMut, force: bool, quote_depth: usize) {
	let trimmed = output.as_str().trim_end_matches([' ', '\t']).len();
	output.truncate(trimmed);
	if force || !output.as_str().ends_with('\n') {
		output.push('\n');
	}
	if quote_depth > 0 {
		append_quote_prefix(output, quote_depth);
	}
}

fn append_quote_prefix(output: &mut StrMut, depth: usize) {
	for _ in 0..depth {
		output.push_str("> ");
	}
}

fn decode_entities(text: &str, output: &mut StrMut) {
	let mut cursor = 0;
	while let Some(relative) = text[cursor..].find('&') {
		let at = cursor + relative;
		output.push_str(&text[cursor..at]);
		let Some(end_relative) = text[at..].find(';') else {
			output.push_str(&text[at..]);
			return;
		};
		let end = at + end_relative;
		let entity = &text[at + 1..end];
		let decoded = if entity.eq_ignore_ascii_case("amp") {
			Some('&')
		} else if entity.eq_ignore_ascii_case("lt") {
			Some('<')
		} else if entity.eq_ignore_ascii_case("gt") {
			Some('>')
		} else if entity.eq_ignore_ascii_case("quot") {
			Some('"')
		} else if entity.eq_ignore_ascii_case("apos") {
			Some('\'')
		} else if entity.eq_ignore_ascii_case("nbsp") {
			// pi decodes to a plain space; the run survives because prose
			// whitespace is never collapsed
			Some(' ')
		} else if let Some(hex) = entity
			.strip_prefix("#x")
			.or_else(|| entity.strip_prefix("#X"))
		{
			u32::from_str_radix(hex, 16).ok().and_then(char::from_u32)
		} else if let Some(decimal) = entity.strip_prefix('#') {
			decimal.parse().ok().and_then(char::from_u32)
		} else {
			None
		};
		if let Some(character) = decoded {
			output.push(character);
		} else {
			output.push_str(&text[at..=end]);
		}
		cursor = end + 1;
	}
	output.push_str(&text[cursor..]);
}

#[cfg(test)]
mod tests {
	use super::*;

	fn rendered(source: &str, width: u16, theme: &MdTheme) -> RichText {
		let source = Str::new(source);
		let mut rendered = RichText::default();
		render(&source, width, theme, &mut rendered);
		rendered
	}

	fn plain(source: &str, width: u16) -> Vec<String> {
		let rendered = rendered(source, width, &MdTheme::default());
		(0..RichText::rows(&rendered))
			.map(|row| rendered.row_text(row).to_owned())
			.collect()
	}

	#[test]
	fn ascii_tier_renders_quote_table_rule_and_bullets_in_pure_ascii() {
		let theme = MdTheme { charset: Charset::Ascii, ..MdTheme::default() };
		let source = "> quoted\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n---\n\nitems:<li>one</li>";
		let rendered = rendered(source, 24, &theme);
		let rows: Vec<String> = (0..RichText::rows(&rendered))
			.map(|row| rendered.row_text(row).to_owned())
			.collect();
		for row in &rows {
			assert!(row.is_ascii(), "non-ASCII chrome leaked: {row:?}");
		}
		assert!(rows.iter().any(|row| row.starts_with("| quoted")), "quote rail: {rows:?}");
		assert!(rows.iter().any(|row| row.starts_with("+--")), "table border: {rows:?}");
		assert!(rows.iter().any(|row| row.contains("| 1")), "table cells: {rows:?}");
		assert!(rows.iter().any(|row| row.contains("----")), "rule fill: {rows:?}");
		// Html list items normalize to native `- ` markers before render.
		assert!(rows.iter().any(|row| row.contains("- one")), "list marker: {rows:?}");
	}

	fn style_containing(rendered: &RichText, needle: &str) -> Style {
		for row in 0..rendered.rows() {
			if let Some(style) = rendered
				.row_runs(row)
				.find_map(|(style, text)| text.contains(needle).then_some(style))
			{
				return style;
			}
		}
		panic!("missing highlighted segment {needle:?}");
	}

	#[test]
	fn heading_levels_and_setext_are_preserved() {
		assert_eq!(plain("# One\n## Two\n### Three\n#### Four\n##### Five\n###### Six", 80), [
			"One",
			"",
			"Two",
			"",
			"### Three",
			"",
			"#### Four",
			"",
			"##### Five",
			"",
			"###### Six"
		]);
		assert_eq!(plain("Title\n===\n\nSubtitle\n---", 80), ["Title", "", "Subtitle"]);
	}

	#[test]
	fn custom_rule_characters_are_retained() {
		for (input, fill) in
			[("---", '─'), ("* * *", '─'), ("===", '='), ("━━━", '━'), ("═══", '═'), ("———", '—')]
		{
			assert_eq!(plain(input, 9), [fill.to_string().repeat(9)]);
		}
	}

	#[test]
	fn source_soft_breaks_are_visible_lines() {
		assert_eq!(plain("first line\nsecond line", 80), ["first line", "second line"]);
	}

	#[test]
	fn block_spacing_is_exactly_one_and_never_trailing() {
		let rendered = plain("# H\ntext\n\n---\n```rs\ncode\n```\n> quote\n\nafter", 80);
		for pair in rendered.windows(2) {
			assert!(!(pair[0].is_empty() && pair[1].is_empty()));
		}
		assert_ne!(rendered.last().map(String::as_str), Some(""));
		assert!(
			rendered
				.windows(3)
				.any(|rows| rows == ["```", "", "│ quote"])
		);
	}

	#[test]
	fn every_block_pair_has_normalized_spacing() {
		let blocks = [
			("paragraph", "paragraph"),
			("heading", "# heading"),
			("code", "```\ncode\n```"),
			("quote", "> quote"),
			("rule", "***"),
			("list", "- item"),
			("table", "| H |\n| --- |\n| C |"),
			("math", "$$\nx\n$$"),
		];
		for (left_name, left) in blocks {
			for (right_name, right) in blocks {
				let rendered = plain(&format!("{left}\n\n{right}"), 40);
				assert!(
					!rendered
						.windows(2)
						.any(|rows| rows[0].is_empty() && rows[1].is_empty()),
					"{left_name} then {right_name}",
				);
				// blank-separated sibling lists merge tight (pi parity), and a
				// paragraph flows straight into a following list
				let merges = (left_name == "paragraph" || left_name == "list") && right_name == "list";
				if !merges {
					assert!(rendered.iter().any(String::is_empty), "{left_name} then {right_name}");
				}
				assert_ne!(rendered.last().map(String::as_str), Some(""));
			}
		}
	}

	#[test]
	fn nested_ordered_and_loose_lists() {
		assert_eq!(plain("10. alpha beta gamma\n   - child\n   - second\n11. next", 16), [
			"10. alpha beta",
			"    gamma",
			"  - child",
			"  - second",
			"11. next"
		]);
		// pi renders loose lists tight: the separating blank never survives
		assert_eq!(plain("- first\n\n- second", 80), ["- first", "- second"]);
	}

	#[test]
	fn blockquote_uses_unicode_rail_on_every_row() {
		assert_eq!(plain("> A long quoted sentence that wraps", 14), [
			"│ A long",
			"│ quoted",
			"│ sentence",
			"│ that wraps"
		]);
		assert_eq!(plain(">Foo\nbar", 80), ["│ Foo", "│ bar"]);
	}

	#[test]
	fn fenced_and_indented_code_show_fences() {
		assert_eq!(plain("```rust\n  a  b\n```", 20), ["```rust", "    a  b", "```"]);
		assert_eq!(plain("    x", 20), ["```", "  x", "```"]);
	}

	#[test]
	fn fenced_code_uses_semantic_highlighting_across_lines_and_lists() {
		let theme = MdTheme::default();
		let palette = Theme::default();
		let source = Str::new(
			"```rust\npub fn main() {\n  let value = \"hi\";\n  /* first\n     second */\n}\n```",
		);
		let fenced = rendered(source.as_str(), 80, &theme);
		assert_eq!(style_containing(&fenced, "pub").foreground_color(), palette.accent);
		assert_eq!(style_containing(&fenced, "hi").foreground_color(), palette.ok);
		assert_eq!(style_containing(&fenced, "second").foreground_color(), palette.muted);

		let listed = Str::new("- ```rust\n  let value = \"ok\";\n  ```");
		let nested = rendered(listed.as_str(), 80, &theme);
		assert_eq!(style_containing(&nested, "let").foreground_color(), palette.accent);
		assert_eq!(style_containing(&nested, "ok").foreground_color(), palette.ok);

		let unknown = Str::new("```not-a-language\nanswer = 42\n```");
		let fallback = rendered(unknown.as_str(), 80, &theme);
		assert_eq!(style_containing(&fallback, "answer"), theme.code_block);
	}

	#[test]
	fn mermaid_fences_render_and_invalid_source_falls_back() {
		let rendered = plain("```mermaid\nflowchart LR\n  A[Start] --> B[Stop]\n```", 80).join("\n");
		assert!(rendered.contains("Start"));
		assert!(rendered.contains("Stop"));
		assert!(!rendered.contains("flowchart"));
		assert!(!rendered.contains("```mermaid"));

		let invalid = plain("```mermaid\nthis is not mermaid\n```", 80).join("\n");
		assert!(invalid.contains("```mermaid"));
		assert!(invalid.contains("this is not mermaid"));
	}

	#[test]
	fn mermaid_diagrams_fit_lists_width_and_ascii_contexts() {
		let listed =
			plain("- ```mermaid\n  flowchart TD\n    A[One] --> B[Two]\n  ```", 40).join("\n");
		assert!(listed.starts_with("- "));
		assert!(listed.contains("One"));
		assert!(listed.contains("Two"));
		assert!(!listed.contains("flowchart"));

		let source =
			"```mermaid\nflowchart LR\n  A[Start] --> B[Build] --> C[Test] --> D[Deploy]\n```";
		let narrow = plain(source, 16);
		assert!(
			narrow
				.iter()
				.all(|line| crate::rich::cell_width(line) <= 16)
		);
		assert!(
			["Start", "Build", "Test", "Deploy"]
				.into_iter()
				.all(|label| narrow.iter().any(|line| line.contains(label)))
		);

		let source = Str::new(source);
		let context = UiContext { charset: Charset::Ascii, ..UiContext::default() };
		let theme = MdTheme::from_context(&context);
		let ascii = rendered(source.as_str(), 40, &theme);
		let ascii = (0..RichText::rows(&ascii))
			.map(|row| ascii.row_text(row))
			.collect::<String>();
		assert!(ascii.is_ascii());
	}

	#[test]
	fn display_math_is_promoted() {
		let block = plain("$$\nx^2\n$$", 40);
		let paragraph = plain("$$x^2$$", 40);
		assert_eq!(block, paragraph);
		assert!(block.iter().any(|line| line.contains('²')));
	}

	#[test]
	fn html_and_entities_are_normalized() {
		assert_eq!(plain("<span>&lt;x&gt;</span> &amp; &quot;q&quot; &#128512;", 80), [
			"<x> & \"q\" 😀"
		]);
		assert_eq!(plain("<ol start=3><li>Third</li><li>Fourth</li></ol>", 80), [
			"3. Third",
			"4. Fourth"
		]);
		assert_eq!(plain("<blockquote>warning<br>now</blockquote>", 80), ["│ warning", "│ now"]);
		let rule = plain("before\n\n<hr>\n\nafter", 10);
		assert!(rule.contains(&"─".repeat(10)));
		assert!(!rule.iter().any(|line| line.contains("<hr>")));
	}

	#[test]
	fn repeated_render_reuses_rich_text_capacity() {
		let source = Str::new("# Heading\n\nA paragraph with **styled** text.\n\n- one\n- two");
		let theme = MdTheme::default();
		let mut output = RichText::default();
		render(&source, 32, &theme, &mut output);
		let first = output.capacities();
		output.clear();
		render(&source, 32, &theme, &mut output);
		assert_eq!(output.capacities(), first);
	}

	#[test]
	fn degenerate_widths_make_progress() {
		for width in [0, 1] {
			let rendered = plain("```\nunclosed\n|||\n- item", width);
			assert_ne!(rendered, [] as [std::string::String; 0]);
			assert!(rendered.len() < 20);
		}
	}
}
