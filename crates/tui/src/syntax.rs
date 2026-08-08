//! XML-ish syntax highlighting for markup shown in an editor.
//!
//! One line at a time, so a caller can highlight only the rows it paints.
//! Comments are the sole construct that spans lines, so the scan threads an
//! `in_comment` flag: pass [`xml_comment_state`] over everything above the
//! first visible row, then chain the flag returned by each
//! [`highlight_xml`] call into the next.
//!
//! ```
//! # use omp_tui::{Theme, syntax::highlight_xml};
//! let (runs, in_comment) = highlight_xml("<col bg=\"red\">hi</col>", &Theme::default(), false);
//! assert!(!in_comment);
//! assert!(runs.len() > 1, "the line is split into styled runs");
//! ```

use smallvec::SmallVec;

use crate::{context::Theme, frame::Style};

/// One styled byte range of a highlighted line.
///
/// Ranges are non-empty, ordered, and cover the line without gaps, so a
/// painter can walk them and slice `text` directly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SyntaxRun {
	/// Byte offset where the run starts.
	pub start: usize,
	/// Byte offset one past the run's last byte.
	pub end:   usize,
	/// Style the run paints with.
	pub style: Style,
}

fn push_syntax_run(runs: &mut SmallVec<SyntaxRun, 16>, start: usize, end: usize, style: Style) {
	if start == end {
		return;
	}
	if let Some(last) = runs.last_mut()
		&& last.end == start
		&& last.style == style
	{
		last.end = end;
	} else {
		runs.push(SyntaxRun { start, end, style });
	}
}

fn take_while(text: &str, mut at: usize, keep: impl Fn(char) -> bool) -> usize {
	while let Some(ch) = text[at..].chars().next() {
		if !keep(ch) {
			break;
		}
		at += ch.len_utf8();
	}
	at
}

/// Threads comment state across text the caller is not highlighting,
/// such as the rows scrolled above an editor's viewport.
pub fn xml_comment_state(text: &str, mut in_comment: bool) -> bool {
	let mut at = 0;
	while at < text.len() {
		let marker = if in_comment { "-->" } else { "<!--" };
		let Some(found) = text[at..].find(marker) else {
			break;
		};
		at += found + marker.len();
		in_comment = !in_comment;
	}
	in_comment
}

/// Splits one line into styled runs, returning whether the line ends
/// inside an unterminated `<!-- ... -->` comment.
///
/// Malformed markup is expected: an unterminated tag or a bare `<` styles
/// what it can and leaves the rest as plain text.
pub fn highlight_xml(
	text: &str,
	theme: &Theme,
	mut in_comment: bool,
) -> (SmallVec<SyntaxRun, 16>, bool) {
	let plain = Style::new().fg(theme.fg);
	let punctuation = Style::new().fg(theme.muted).dim();
	let element = Style::new().fg(theme.accent);
	let attribute = Style::new().fg(theme.info);
	let value = Style::new().fg(theme.warn);
	let mut runs = SmallVec::new();
	let mut at = 0;

	while at < text.len() {
		if in_comment {
			if let Some(close) = text[at..].find("-->") {
				let end = at + close + 3;
				push_syntax_run(&mut runs, at, end, punctuation);
				at = end;
				in_comment = false;
				continue;
			}
			push_syntax_run(&mut runs, at, text.len(), punctuation);
			return (runs, true);
		}

		if text[at..].starts_with("<!--") {
			in_comment = true;
			continue;
		}
		if !text[at..].starts_with('<') {
			let end = text[at..].find('<').map_or(text.len(), |next| at + next);
			push_syntax_run(&mut runs, at, end, plain);
			at = end;
			continue;
		}

		let punctuation_end = at + if text[at..].starts_with("</") { 2 } else { 1 };
		push_syntax_run(&mut runs, at, punctuation_end, punctuation);
		at = punctuation_end;

		let whitespace_end = take_while(text, at, char::is_whitespace);
		push_syntax_run(&mut runs, at, whitespace_end, plain);
		at = whitespace_end;
		let name_end =
			take_while(text, at, |ch| !ch.is_whitespace() && !matches!(ch, '/' | '>' | '=' | '<'));
		push_syntax_run(&mut runs, at, name_end, element);
		if name_end == at {
			continue;
		}
		at = name_end;

		while at < text.len() {
			let whitespace_end = take_while(text, at, char::is_whitespace);
			push_syntax_run(&mut runs, at, whitespace_end, plain);
			at = whitespace_end;
			// a tag left open mid-typing (`<box `) ends the line here
			if at == text.len() {
				break;
			}
			if text[at..].starts_with("/>") {
				push_syntax_run(&mut runs, at, at + 2, punctuation);
				at += 2;
				break;
			}
			if text[at..].starts_with('>') {
				push_syntax_run(&mut runs, at, at + 1, punctuation);
				at += 1;
				break;
			}
			if text[at..].starts_with('=') {
				push_syntax_run(&mut runs, at, at + 1, punctuation);
				at += 1;
				continue;
			}
			if text[at..].starts_with('<') {
				break;
			}

			let attribute_end =
				take_while(text, at, |ch| !ch.is_whitespace() && !matches!(ch, '=' | '/' | '>' | '<'));
			push_syntax_run(&mut runs, at, attribute_end, attribute);
			if attribute_end == at {
				// a lone delimiter (`"`, `'`, …) with no name in front of it
				let Some(ch) = text[at..].chars().next() else {
					break;
				};
				push_syntax_run(&mut runs, at, at + ch.len_utf8(), plain);
				at += ch.len_utf8();
				continue;
			}
			at = attribute_end;

			let whitespace_end = take_while(text, at, char::is_whitespace);
			push_syntax_run(&mut runs, at, whitespace_end, plain);
			at = whitespace_end;
			if !text[at..].starts_with('=') {
				continue;
			}
			push_syntax_run(&mut runs, at, at + 1, punctuation);
			at += 1;
			let whitespace_end = take_while(text, at, char::is_whitespace);
			push_syntax_run(&mut runs, at, whitespace_end, plain);
			at = whitespace_end;

			let Some(first) = text[at..].chars().next() else {
				break;
			};
			if matches!(first, '"' | '\'') {
				let after_quote = at + first.len_utf8();
				let end = text[after_quote..]
					.find(first)
					.map_or(text.len(), |close| after_quote + close + first.len_utf8());
				push_syntax_run(&mut runs, at, end, value);
				at = end;
			} else {
				let end =
					take_while(text, at, |ch| !ch.is_whitespace() && !matches!(ch, '/' | '>' | '<'));
				push_syntax_run(&mut runs, at, end, value);
				at = end;
			}
		}
	}

	(runs, in_comment)
}

#[cfg(test)]
mod syntax_tests {
	use super::*;
	use crate::context::Theme;

	fn style_for<'a>(line: &'a str, needle: &str, runs: &'a [SyntaxRun]) -> Style {
		let start = line.find(needle).expect("token present");
		let end = start + needle.len();
		runs
			.iter()
			.find(|run| run.start <= start && run.end >= end)
			.expect("token has a style")
			.style
	}

	#[test]
	fn xml_editor_highlight_assigns_semantic_style_runs() {
		let theme = Theme::default();
		let line = r#"<col bg="red" grow>hi</col>"#;
		let (runs, in_comment) = highlight_xml(line, &theme, false);

		assert!(!in_comment);
		assert_eq!(style_for(line, "<", &runs), Style::new().fg(theme.muted).dim());
		assert_eq!(style_for(line, "col", &runs), Style::new().fg(theme.accent));
		assert_eq!(style_for(line, "bg", &runs), Style::new().fg(theme.info));
		assert_eq!(style_for(line, r#""red""#, &runs), Style::new().fg(theme.warn));
		assert_eq!(style_for(line, "hi", &runs), Style::new().fg(theme.fg));
	}

	#[test]
	fn xml_editor_highlight_tolerates_partial_tags() {
		let theme = Theme::default();
		for line in ["<foo attr=", "<"] {
			let (runs, in_comment) = highlight_xml(line, &theme, false);
			assert!(!in_comment);
			assert_eq!(runs.first().map(|run| run.start), Some(0));
			assert_eq!(runs.last().map(|run| run.end), Some(line.len()));
			assert!(runs.windows(2).all(|pair| pair[0].end == pair[1].start));
		}

		let line = "λ <foo attr=";
		let (runs, _) = highlight_xml(line, &theme, false);
		assert_eq!(style_for(line, "λ ", &runs), Style::new().fg(theme.fg));
		assert!(
			runs
				.iter()
				.all(|run| line.is_char_boundary(run.start) && line.is_char_boundary(run.end))
		);
	}

	#[test]
	fn xml_editor_comments_continue_across_lines() {
		let theme = Theme::default();
		let (first, in_comment) = highlight_xml("a <!-- open", &theme, false);
		assert!(in_comment);
		assert_eq!(style_for("a <!-- open", "<!-- open", &first), Style::new().fg(theme.muted).dim());
		let (second, in_comment) = highlight_xml("close --> tail", &theme, in_comment);
		assert!(!in_comment);
		assert_eq!(
			style_for("close --> tail", "close -->", &second),
			Style::new().fg(theme.muted).dim()
		);
		assert_eq!(style_for("close --> tail", " tail", &second), Style::new().fg(theme.fg));
	}
	/// Every keystroke is a partial line, so the scanner has to survive
	/// every prefix of a realistic one — including `<box ` and `<a t="`.
	#[test]
	fn xml_highlighting_survives_every_prefix_of_a_line() {
		let theme = Theme::default();
		let line = "<box bg=black><row gap='1'><col bg=\"red\" grow>hi<!-- c --></col></row>";
		for end in 0..=line.len() {
			if !line.is_char_boundary(end) {
				continue;
			}
			let head = &line[..end];
			let (runs, _) = highlight_xml(head, &theme, false);
			let covered: usize = runs.iter().map(|run| run.end - run.start).sum();
			assert_eq!(covered, head.len(), "runs must tile the line for {head:?}");
			assert!(
				runs.windows(2).all(|pair| pair[0].end == pair[1].start),
				"runs must be gapless and ordered for {head:?}"
			);
		}
	}
}
