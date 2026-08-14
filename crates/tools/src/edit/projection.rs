//! Pure pi-compatible model-facing edit response projection.

use omp_core::{Str, StrMut};

use super::ResolvedBlock;

/// File-level outcome rendered for one hashline section.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SectionOp {
	/// The file was removed by `REM`.
	Delete,
	/// The patch applied cleanly without changing bytes.
	Noop,
	/// Content was updated and/or moved.
	Update,
}

/// Borrowed facts needed to render one section exactly as pi does.
#[derive(Clone, Copy, Debug)]
pub struct SectionView<'a> {
	/// Durable section outcome.
	pub op:                SectionOp,
	/// Authored/resolved source path.
	pub path:              &'a str,
	/// Post-edit hashline header for updates.
	pub header:            &'a str,
	/// Escalation-aware diagnostic for a no-op.
	pub noop_diagnostic:   &'a str,
	/// Destination path for `MV`.
	pub move_dest:         Option<&'a str>,
	/// Compact numbered current-file preview.
	pub preview:           &'a str,
	/// Concrete spans selected by block locators.
	pub block_resolutions: &'a [ResolvedBlock],
	/// Non-fatal application/recovery diagnostics.
	pub warnings:          &'a [Str],
}

/// Renders one section's exact model-facing success/diagnostic text.
#[must_use]
pub fn render_section(view: SectionView<'_>) -> Str {
	match view.op {
		SectionOp::Delete => return format!("Deleted {}", view.path).into(),
		SectionOp::Noop => return view.noop_diagnostic.into(),
		SectionOp::Update => {},
	}

	let estimated = view.header.len()
		+ view.preview.len()
		+ view.warnings.iter().map(Str::len).sum::<usize>()
		+ view.block_resolutions.len() * 80;
	let mut output = StrMut::with_capacity(estimated);
	output.push_str(view.header);
	for resolution in view.block_resolutions {
		output.push('\n');
		output.push_str(&format_block_resolution(resolution));
	}
	if let Some(destination) = view.move_dest {
		output.push_str("\nMoved to ");
		output.push_str(destination);
	}
	if !view.preview.is_empty() {
		output.push('\n');
		output.push_str(view.preview);
	}
	if !view.warnings.is_empty() {
		output.push_str("\n\nWarnings:\n");
		for (index, warning) in view.warnings.iter().enumerate() {
			if index > 0 {
				output.push('\n');
			}
			output.push_str(warning);
		}
	}
	output.freeze()
}

/// Joins independently rendered section responses with pi's single blank row.
#[must_use]
pub fn render_sections(sections: &[Str]) -> Str {
	let capacity =
		sections.iter().map(Str::len).sum::<usize>() + sections.len().saturating_sub(1) * 2;
	let mut output = StrMut::with_capacity(capacity);
	for (index, section) in sections.iter().enumerate() {
		if index > 0 {
			output.push_str("\n\n");
		}
		output.push_str(section);
	}
	output.freeze()
}

/// Formats one syntax-aware block resolution using authored locator
/// coordinates.
#[must_use]
pub fn format_block_resolution(resolution: &ResolvedBlock) -> Str {
	let label = match resolution.operation.as_str() {
		"replace" => format!("PUT {}*:", resolution.anchor_line),
		"insert_after" => format!("PUT >{}*:", resolution.anchor_line),
		"cut" => format!("CUT {}*", resolution.anchor_line),
		"paste_after" => format!("PUT >{}*", resolution.anchor_line),
		operation => format!("{operation} {}", resolution.anchor_line),
	};
	let lines = resolution.end - resolution.start + 1;
	let span = if resolution.start == resolution.end {
		format!("line {}", resolution.start)
	} else {
		format!("lines {}-{}", resolution.start, resolution.end)
	};
	let suffix = match resolution.operation.as_str() {
		"insert_after" => format!("; body lands after line {}", resolution.end),
		"paste_after" => format!("; clipboard lands after line {}", resolution.end),
		_ => String::new(),
	};
	format!("{label} → resolved {span} ({lines} line{}){suffix}", if lines == 1 { "" } else { "s" })
		.into()
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn renders_update_delete_move_warnings_and_blocks_exactly() {
		let resolution = ResolvedBlock {
			anchor_line: 4,
			start:       4,
			end:         7,
			operation:   "insert_after".into(),
		};
		let warnings = vec!["Recovered by remapping stale line anchors.".into()];
		assert_eq!(
			render_section(SectionView {
				op:                SectionOp::Update,
				path:              "src/old.rs",
				header:            "[src/new.rs#1A2B]",
				noop_diagnostic:   "",
				move_dest:         Some("src/new.rs"),
				preview:           "4:fn f() {\n8:after();",
				block_resolutions: &[resolution],
				warnings:          &warnings,
			}),
			"[src/new.rs#1A2B]\nPUT >4*: → resolved lines 4-7 (4 lines); body lands after line \
			 7\nMoved to src/new.rs\n4:fn f() {\n8:after();\n\nWarnings:\nRecovered by remapping \
			 stale line anchors."
		);
		assert_eq!(
			render_section(SectionView {
				op:                SectionOp::Delete,
				path:              "src/old.rs",
				header:            "",
				noop_diagnostic:   "",
				move_dest:         None,
				preview:           "",
				block_resolutions: &[],
				warnings:          &[],
			}),
			"Deleted src/old.rs"
		);
	}

	#[test]
	fn formats_every_block_operation_label_exactly() {
		for (operation, expected) in [
			("replace", "PUT 9*: → resolved line 9 (1 line)"),
			("insert_after", "PUT >9*: → resolved line 9 (1 line); body lands after line 9"),
			("cut", "CUT 9* → resolved line 9 (1 line)"),
			("paste_after", "PUT >9* → resolved line 9 (1 line); clipboard lands after line 9"),
		] {
			assert_eq!(
				format_block_resolution(&ResolvedBlock {
					anchor_line: 9,
					start:       9,
					end:         9,
					operation:   operation.into(),
				}),
				expected
			);
		}
	}

	#[test]
	fn joins_section_results_with_one_blank_row() {
		assert_eq!(
			render_sections(&["[a.rs#1A2B]\n1:A".into(), "Deleted b.rs".into()]),
			"[a.rs#1A2B]\n1:A\n\nDeleted b.rs"
		);
	}
}
