//! Parsing for shell instances.

use std::io::Read;

use crate::{Shell, extensions, trace_categories};

impl<SE: extensions::ShellExtensions> Shell<SE> {
	/// Parses the given reader as a shell program, returning the resulting
	/// Abstract Syntax Tree for the program.
	pub fn parse<R: Read>(
		&self,
		reader: R,
	) -> Result<crate::parser::ast::Program, crate::parser::ParseError> {
		let mut parser = create_parser(reader, &self.parser_options());

		tracing::debug!(target: trace_categories::PARSE, "Parsing reader as program...");
		parser.parse_program()
	}

	/// Parses the given string as a shell program, returning the resulting
	/// Abstract Syntax Tree for the program.
	///
	/// # Arguments
	///
	/// * `s` - The string to parse as a program.
	pub fn parse_string<S: Into<String>>(
		&self,
		s: S,
	) -> Result<crate::parser::ast::Program, crate::parser::ParseError> {
		parse_string_impl(s.into(), self.parser_options())
	}

	/// Returns the options that should be used for parsing shell programs;
	/// reflects the current configuration state of the shell and may change
	/// over time.
	pub const fn parser_options(&self) -> crate::parser::ParserOptions {
		crate::parser::ParserOptions {
			enable_extended_globbing: self.options.extended_globbing,
			posix_mode: self.options.posix_mode,
			sh_mode: self.options.sh_mode,
			tilde_expansion_at_word_start: true,
			tilde_expansion_after_colon: false,
			parser_impl: self.parser_impl,
		}
	}
}

#[cached::proc_macro::cached(size = 64, result = true)]
fn parse_string_impl(
	s: String,
	parser_options: crate::parser::ParserOptions,
) -> Result<crate::parser::ast::Program, crate::parser::ParseError> {
	let mut parser = create_parser(s.as_bytes(), &parser_options);

	tracing::debug!(target: trace_categories::PARSE, "Parsing string as program...");
	parser.parse_program()
}

pub(super) fn create_parser<R: Read>(
	r: R,
	parser_options: &crate::parser::ParserOptions,
) -> crate::parser::Parser<std::io::BufReader<R>> {
	let reader = std::io::BufReader::new(r);
	crate::parser::Parser::new(reader, parser_options)
}
