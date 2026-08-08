//! Cross-model reasoning demotion for structured native history.

use std::fmt;

use crate::{
	Dialect,
	rendering::{render_thinking, write_escaped_harmony_text},
};

/// Writes prior-turn reasoning in a form safe for native structured history.
///
/// Anthropic receives bare assistant prose because tagged replay triggers its
/// reasoning-extraction classifier. Harmony and Gemma avoid their native
/// chat-template control channels and use an ordinary `<think>` block. Other
/// dialects use their inline-safe reasoning representation.
pub fn render_demoted_thinking<W: fmt::Write + ?Sized>(
	out: &mut W,
	dialect: Dialect,
	text: &str,
) -> fmt::Result {
	if text.is_empty() {
		return Ok(());
	}
	match dialect {
		Dialect::Anthropic => out.write_str(text),
		Dialect::Harmony => {
			out.write_str("<think>\n")?;
			write_escaped_harmony_text(out, text)?;
			out.write_str("\n</think>")
		},
		Dialect::Gemma => write!(out, "<think>\n{text}\n</think>"),
		_ => render_thinking(out, dialect, text),
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn every_dialect_has_nonempty_cross_model_demotion() {
		for dialect in Dialect::ALL {
			let mut rendered = String::new();
			render_demoted_thinking(&mut rendered, dialect, "private").unwrap();
			assert!(rendered.contains("private"), "{dialect}: {rendered}");
			match dialect {
				Dialect::Anthropic => assert_eq!(rendered, "private"),
				Dialect::Harmony | Dialect::Gemma => {
					assert_eq!(rendered, "<think>\nprivate\n</think>");
				},
				_ => assert_ne!(rendered, "private"),
			}
		}
	}

	#[test]
	fn harmony_demotion_escapes_untrusted_controls() {
		let mut rendered = String::new();
		render_demoted_thinking(&mut rendered, Dialect::Harmony, "<\u{7c}channel\u{7c}>analysis")
			.unwrap();
		assert_eq!(rendered, "<think>\n<\\|channel\\|>analysis\n</think>");
	}
}
