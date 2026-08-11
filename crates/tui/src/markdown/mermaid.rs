//! Mermaid-to-terminal rendering for fenced Markdown blocks.

use mermaid_text::RenderOptions;
use omp_core::{Str, StrMut};
use xutf::Text;

use crate::{
	context::Charset,
	frame::Style,
	rich::{Pipeline, RichSink},
};

/// Semantic styles applied to rendered diagram cells.
#[derive(Clone, Copy)]
pub(super) struct MermaidStyles {
	/// Node labels and other prose.
	pub text:   Style,
	/// Borders, connectors, and corners.
	pub line:   Style,
	/// Arrowheads, markers, and chart fills.
	pub accent: Style,
}

/// Renders Mermaid source. Returns `false` without emitting so Markdown can
/// preserve invalid source.
pub(super) fn render(
	source: &str,
	width: u16,
	charset: Charset,
	styles: MermaidStyles,
	sink: &mut dyn RichSink,
) -> bool {
	let source = source.trim();
	if source.is_empty() || width == 0 {
		return false;
	}

	let Some(rendered) = render_best(source, usize::from(width), charset) else {
		return false;
	};
	if rendered.trim().is_empty() {
		return false;
	}
	for chunk in rendered.split_inclusive('\n') {
		let text = chunk.strip_suffix('\n').unwrap_or(chunk);
		let mut clip = (&mut *sink).clip(width, None);
		style_row(text, charset, styles, &mut clip);
		clip.newline();
	}
	true
}

fn render_best(source: &str, width: usize, charset: Charset) -> Option<String> {
	let options = RenderOptions {
		max_width: Some(width),
		ascii: matches!(charset, Charset::Ascii),
		..RenderOptions::default()
	};
	let mut best = mermaid_text::render_with_options(source, &options).ok()?;
	let mut best_width = display_width(&best);
	if best_width <= width {
		return Some(best);
	}

	// Horizontal chains commonly overflow terminal panes. Match the coding
	// agent renderer by trying both primary orientations and keeping the narrowest.
	for direction in ["TD", "LR"] {
		let Some(variant_source) = force_flow_direction(source, direction) else {
			break;
		};
		if variant_source.as_str() == source {
			continue;
		}
		let Ok(candidate) = mermaid_text::render_with_options(variant_source.as_str(), &options)
		else {
			continue;
		};
		let candidate_width = display_width(&candidate);
		if candidate_width < best_width {
			best = candidate;
			best_width = candidate_width;
		}
	}
	Some(best)
}

fn display_width(rendered: &str) -> usize {
	rendered.lines().map(Text::visible_width).max().unwrap_or(0)
}

fn force_flow_direction(source: &str, direction: &str) -> Option<Str> {
	let (start, end, needs_space) = flow_direction_span(source)?;
	let mut forced = StrMut::with_capacity(source.len().saturating_add(3));
	forced.push_str(&source[..start]);
	if needs_space {
		forced.push(' ');
	}
	forced.push_str(direction);
	forced.push_str(&source[end..]);
	Some(forced.freeze())
}

fn flow_direction_span(source: &str) -> Option<(usize, usize, bool)> {
	let mut base = 0;
	for chunk in source.split_inclusive(['\n', ';']) {
		let statement = chunk
			.strip_suffix('\n')
			.or_else(|| chunk.strip_suffix(';'))
			.unwrap_or(chunk);
		let trimmed = statement.trim_start();
		let leading = statement.len().saturating_sub(trimmed.len());
		if trimmed.is_empty() || trimmed.starts_with("%%") {
			base += chunk.len();
			continue;
		}

		let keyword_end = trimmed.find(char::is_whitespace).unwrap_or(trimmed.len());
		let keyword = &trimmed[..keyword_end];
		if !keyword.eq_ignore_ascii_case("graph") && !keyword.eq_ignore_ascii_case("flowchart") {
			return None;
		}

		let rest = &trimmed[keyword_end..];
		let direction_text = rest.trim_start();
		if direction_text.is_empty() || direction_text.starts_with("%%") {
			let insertion = base + leading + keyword_end;
			return Some((insertion, insertion, true));
		}
		let whitespace = rest.len().saturating_sub(direction_text.len());
		let direction_end = direction_text
			.find(char::is_whitespace)
			.unwrap_or(direction_text.len());
		let current = &direction_text[..direction_end];
		if !["TD", "TB", "BT", "LR", "RL"]
			.iter()
			.any(|candidate| current.eq_ignore_ascii_case(candidate))
		{
			return None;
		}
		let start = base + leading + keyword_end + whitespace;
		return Some((start, start + direction_end, false));
	}
	None
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CellRole {
	Text,
	Border,
	Accent,
}

fn style_row(text: &str, charset: Charset, styles: MermaidStyles, sink: &mut dyn RichSink) {
	if text.is_empty() {
		return;
	}
	if matches!(charset, Charset::Ascii) {
		sink.run(styles.text, text);
		return;
	}

	let mut start = 0;
	let mut current = None;
	for (offset, character) in text.char_indices() {
		let role = cell_role(character);
		if current.is_some_and(|previous| previous != role) {
			sink.run(style_for(current.unwrap_or(CellRole::Text), styles), &text[start..offset]);
			start = offset;
		}
		current = Some(role);
	}
	sink.run(style_for(current.unwrap_or(CellRole::Text), styles), &text[start..]);
}

const fn style_for(role: CellRole, styles: MermaidStyles) -> Style {
	match role {
		CellRole::Text => styles.text,
		CellRole::Border => styles.line,
		CellRole::Accent => styles.accent,
	}
}

fn cell_role(character: char) -> CellRole {
	if ('\u{2500}'..='\u{257f}').contains(&character) {
		CellRole::Border
	} else if ('\u{2190}'..='\u{21ff}').contains(&character)
		|| ('\u{2580}'..='\u{259f}').contains(&character)
		|| matches!(
			character,
			'▸' | '◂'
				| '▴' | '▾'
				| '▶' | '◀'
				| '▲' | '▼'
				| '►' | '◄'
				| '●' | '○'
				| '◆' | '◇'
		) {
		CellRole::Accent
	} else {
		CellRole::Text
	}
}

#[cfg(test)]
mod tests {
	use super::{CellRole, cell_role};

	#[test]
	fn punctuation_inside_labels_keeps_the_text_role() {
		for character in ['=', '#', ':', ',', '.'] {
			assert_eq!(cell_role(character), CellRole::Text);
		}
		assert_eq!(cell_role('═'), CellRole::Border);
		assert_eq!(cell_role('●'), CellRole::Accent);
	}
}
