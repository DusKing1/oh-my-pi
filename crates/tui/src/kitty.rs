//! Kitty graphics protocol image transmission and Unicode placeholders.

use std::fmt::Write as _;

use omp_core::{Str, fmts};

use crate::{Color, Style, escape::esc};

pub const PLACEHOLDER_LIMIT: u16 = 297;

// Canonical kitty row/column diacritics, generated from Unicode 6.0 Mn/230
// characters. Keep this order synchronized with kitty's
// gen/rowcolumn-diacritics.txt.
const DIACRITICS: &[char] = &[
	'\u{0305}',
	'\u{030d}',
	'\u{030e}',
	'\u{0310}',
	'\u{0312}',
	'\u{033d}',
	'\u{033e}',
	'\u{033f}',
	'\u{0346}',
	'\u{034a}',
	'\u{034b}',
	'\u{034c}',
	'\u{0350}',
	'\u{0351}',
	'\u{0352}',
	'\u{0357}',
	'\u{035b}',
	'\u{0363}',
	'\u{0364}',
	'\u{0365}',
	'\u{0366}',
	'\u{0367}',
	'\u{0368}',
	'\u{0369}',
	'\u{036a}',
	'\u{036b}',
	'\u{036c}',
	'\u{036d}',
	'\u{036e}',
	'\u{036f}',
	'\u{0483}',
	'\u{0484}',
	'\u{0485}',
	'\u{0486}',
	'\u{0487}',
	'\u{0592}',
	'\u{0593}',
	'\u{0594}',
	'\u{0595}',
	'\u{0597}',
	'\u{0598}',
	'\u{0599}',
	'\u{059c}',
	'\u{059d}',
	'\u{059e}',
	'\u{059f}',
	'\u{05a0}',
	'\u{05a1}',
	'\u{05a8}',
	'\u{05a9}',
	'\u{05ab}',
	'\u{05ac}',
	'\u{05af}',
	'\u{05c4}',
	'\u{0610}',
	'\u{0611}',
	'\u{0612}',
	'\u{0613}',
	'\u{0614}',
	'\u{0615}',
	'\u{0616}',
	'\u{0617}',
	'\u{0657}',
	'\u{0658}',
	'\u{0659}',
	'\u{065a}',
	'\u{065b}',
	'\u{065d}',
	'\u{065e}',
	'\u{06d6}',
	'\u{06d7}',
	'\u{06d8}',
	'\u{06d9}',
	'\u{06da}',
	'\u{06db}',
	'\u{06dc}',
	'\u{06df}',
	'\u{06e0}',
	'\u{06e1}',
	'\u{06e2}',
	'\u{06e4}',
	'\u{06e7}',
	'\u{06e8}',
	'\u{06eb}',
	'\u{06ec}',
	'\u{0730}',
	'\u{0732}',
	'\u{0733}',
	'\u{0735}',
	'\u{0736}',
	'\u{073a}',
	'\u{073d}',
	'\u{073f}',
	'\u{0740}',
	'\u{0741}',
	'\u{0743}',
	'\u{0745}',
	'\u{0747}',
	'\u{0749}',
	'\u{074a}',
	'\u{07eb}',
	'\u{07ec}',
	'\u{07ed}',
	'\u{07ee}',
	'\u{07ef}',
	'\u{07f0}',
	'\u{07f1}',
	'\u{07f3}',
	'\u{0816}',
	'\u{0817}',
	'\u{0818}',
	'\u{0819}',
	'\u{081b}',
	'\u{081c}',
	'\u{081d}',
	'\u{081e}',
	'\u{081f}',
	'\u{0820}',
	'\u{0821}',
	'\u{0822}',
	'\u{0823}',
	'\u{0825}',
	'\u{0826}',
	'\u{0827}',
	'\u{0829}',
	'\u{082a}',
	'\u{082b}',
	'\u{082c}',
	'\u{082d}',
	'\u{0951}',
	'\u{0953}',
	'\u{0954}',
	'\u{0f82}',
	'\u{0f83}',
	'\u{0f86}',
	'\u{0f87}',
	'\u{135d}',
	'\u{135e}',
	'\u{135f}',
	'\u{17dd}',
	'\u{193a}',
	'\u{1a17}',
	'\u{1a75}',
	'\u{1a76}',
	'\u{1a77}',
	'\u{1a78}',
	'\u{1a79}',
	'\u{1a7a}',
	'\u{1a7b}',
	'\u{1a7c}',
	'\u{1b6b}',
	'\u{1b6d}',
	'\u{1b6e}',
	'\u{1b6f}',
	'\u{1b70}',
	'\u{1b71}',
	'\u{1b72}',
	'\u{1b73}',
	'\u{1cd0}',
	'\u{1cd1}',
	'\u{1cd2}',
	'\u{1cda}',
	'\u{1cdb}',
	'\u{1ce0}',
	'\u{1dc0}',
	'\u{1dc1}',
	'\u{1dc3}',
	'\u{1dc4}',
	'\u{1dc5}',
	'\u{1dc6}',
	'\u{1dc7}',
	'\u{1dc8}',
	'\u{1dc9}',
	'\u{1dcb}',
	'\u{1dcc}',
	'\u{1dd1}',
	'\u{1dd2}',
	'\u{1dd3}',
	'\u{1dd4}',
	'\u{1dd5}',
	'\u{1dd6}',
	'\u{1dd7}',
	'\u{1dd8}',
	'\u{1dd9}',
	'\u{1dda}',
	'\u{1ddb}',
	'\u{1ddc}',
	'\u{1ddd}',
	'\u{1dde}',
	'\u{1ddf}',
	'\u{1de0}',
	'\u{1de1}',
	'\u{1de2}',
	'\u{1de3}',
	'\u{1de4}',
	'\u{1de5}',
	'\u{1de6}',
	'\u{1dfe}',
	'\u{20d0}',
	'\u{20d1}',
	'\u{20d4}',
	'\u{20d5}',
	'\u{20d6}',
	'\u{20d7}',
	'\u{20db}',
	'\u{20dc}',
	'\u{20e1}',
	'\u{20e7}',
	'\u{20e9}',
	'\u{20f0}',
	'\u{2cef}',
	'\u{2cf0}',
	'\u{2cf1}',
	'\u{2de0}',
	'\u{2de1}',
	'\u{2de2}',
	'\u{2de3}',
	'\u{2de4}',
	'\u{2de5}',
	'\u{2de6}',
	'\u{2de7}',
	'\u{2de8}',
	'\u{2de9}',
	'\u{2dea}',
	'\u{2deb}',
	'\u{2dec}',
	'\u{2ded}',
	'\u{2dee}',
	'\u{2def}',
	'\u{2df0}',
	'\u{2df1}',
	'\u{2df2}',
	'\u{2df3}',
	'\u{2df4}',
	'\u{2df5}',
	'\u{2df6}',
	'\u{2df7}',
	'\u{2df8}',
	'\u{2df9}',
	'\u{2dfa}',
	'\u{2dfb}',
	'\u{2dfc}',
	'\u{2dfd}',
	'\u{2dfe}',
	'\u{2dff}',
	'\u{a66f}',
	'\u{a67c}',
	'\u{a67d}',
	'\u{a6f0}',
	'\u{a6f1}',
	'\u{a8e0}',
	'\u{a8e1}',
	'\u{a8e2}',
	'\u{a8e3}',
	'\u{a8e4}',
	'\u{a8e5}',
	'\u{a8e6}',
	'\u{a8e7}',
	'\u{a8e8}',
	'\u{a8e9}',
	'\u{a8ea}',
	'\u{a8eb}',
	'\u{a8ec}',
	'\u{a8ed}',
	'\u{a8ee}',
	'\u{a8ef}',
	'\u{a8f0}',
	'\u{a8f1}',
	'\u{aab0}',
	'\u{aab2}',
	'\u{aab3}',
	'\u{aab7}',
	'\u{aab8}',
	'\u{aabe}',
	'\u{aabf}',
	'\u{aac1}',
	'\u{fe20}',
	'\u{fe21}',
	'\u{fe22}',
	'\u{fe23}',
	'\u{fe24}',
	'\u{fe25}',
	'\u{fe26}',
	'\u{10a0f}',
	'\u{10a38}',
	'\u{1d185}',
	'\u{1d186}',
	'\u{1d187}',
	'\u{1d188}',
	'\u{1d189}',
	'\u{1d1aa}',
	'\u{1d1ab}',
	'\u{1d1ac}',
	'\u{1d1ad}',
	'\u{1d242}',
	'\u{1d243}',
	'\u{1d244}',
];

/// Deterministic nonzero placement ID for one cell box.
///
/// Virtual placements without an explicit ID accumulate in the terminal
/// (kitty and ghostty allocate a fresh internal placement per command) and
/// ID-less placeholders then resolve against an arbitrary one — stale grids
/// from earlier runs or other sizes crop the image to its top-left cells.
/// Deriving the ID from the box keeps re-placements idempotent and lets each
/// box size of one image coexist. Bounded by [`PLACEHOLDER_LIMIT`] (< 2⁹),
/// so the ID stays within kitty's 24-bit placement range.
pub const fn placement_id(rows: u16, cols: u16) -> u32 {
	(rows as u32) << 9 | cols as u32
}

/// Builds one Kitty Unicode-placeholder grapheme and its style: the image ID
/// rides the foreground color, the placement ID the underline color.
pub fn placeholder_cell(id: u32, row: u16, col: u16, rows: u16, cols: u16) -> (Str, Style) {
	let row_mark = DIACRITICS
		.get(usize::from(row))
		.copied()
		.unwrap_or(DIACRITICS[0]);
	let col_mark = DIACRITICS
		.get(usize::from(col))
		.copied()
		.unwrap_or(DIACRITICS[0]);
	let text = fmts!("\u{10eeee}{row_mark}{col_mark}");
	let placement = placement_id(rows, cols);
	let style = Style::new()
		.fg(Color::Rgb((id >> 16) as u8, (id >> 8) as u8, id as u8))
		.underline_color(Color::Rgb(
			(placement >> 16) as u8,
			(placement >> 8) as u8,
			placement as u8,
		));
	(text, style)
}

pub fn encode_base64(input: &[u8], output: &mut [u8; 4096]) -> usize {
	const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
	let mut source = 0;
	let mut target = 0;
	while source + 3 <= input.len() {
		let bits = (u32::from(input[source]) << 16)
			| (u32::from(input[source + 1]) << 8)
			| u32::from(input[source + 2]);
		output[target] = ALPHABET[((bits >> 18) & 63) as usize];
		output[target + 1] = ALPHABET[((bits >> 12) & 63) as usize];
		output[target + 2] = ALPHABET[((bits >> 6) & 63) as usize];
		output[target + 3] = ALPHABET[(bits & 63) as usize];
		source += 3;
		target += 4;
	}
	match input.len() - source {
		1 => {
			let bits = u32::from(input[source]) << 16;
			output[target] = ALPHABET[((bits >> 18) & 63) as usize];
			output[target + 1] = ALPHABET[((bits >> 12) & 63) as usize];
			output[target + 2] = b'=';
			output[target + 3] = b'=';
			target += 4;
		},
		2 => {
			let bits = (u32::from(input[source]) << 16) | (u32::from(input[source + 1]) << 8);
			output[target] = ALPHABET[((bits >> 18) & 63) as usize];
			output[target + 1] = ALPHABET[((bits >> 12) & 63) as usize];
			output[target + 2] = ALPHABET[((bits >> 6) & 63) as usize];
			output[target + 3] = b'=';
			target += 4;
		},
		_ => {},
	}
	target
}

fn append_apc_start(output: &mut String, tmux_passthrough: bool) {
	if tmux_passthrough {
		output.push_str(esc!(dcs, "tmux;", escape, apc, "G"));
	} else {
		output.push_str(esc!(apc, "G"));
	}
}

fn append_apc_end(output: &mut String, tmux_passthrough: bool) {
	if tmux_passthrough {
		output.push_str(esc!(escape, st, st));
	} else {
		output.push_str(esc!(st));
	}
}

pub fn append_transmission(output: &mut String, id: u32, png: &[u8], tmux_passthrough: bool) {
	let chunks = png.chunks(3072);
	let count = chunks.len();
	let mut encoded = [0_u8; 4096];
	for (index, chunk) in chunks.enumerate() {
		let length = encode_base64(chunk, &mut encoded);
		append_apc_start(output, tmux_passthrough);
		if index == 0 {
			let _ = write!(output, "f=100,t=d,a=t,i={id},q=2,m={};", u8::from(index + 1 < count));
		} else {
			let _ = write!(output, "m={};", u8::from(index + 1 < count));
		}
		output.push_str(std::str::from_utf8(&encoded[..length]).expect("base64 is ASCII"));
		append_apc_end(output, tmux_passthrough);
	}
}

pub fn append_placement(
	output: &mut String,
	id: u32,
	rows: u16,
	cols: u16,
	tmux_passthrough: bool,
) {
	append_apc_start(output, tmux_passthrough);
	let placement = placement_id(rows, cols);
	let _ = write!(output, "a=p,U=1,i={id},p={placement},r={rows},c={cols},q=2");
	append_apc_end(output, tmux_passthrough);
}

#[derive(Clone, Copy)]
pub struct DirectPlacement {
	pub(crate) source_x:      u32,
	pub(crate) source_y:      u32,
	pub(crate) source_width:  u32,
	pub(crate) source_height: u32,
	pub(crate) rows:          u16,
	pub(crate) cols:          u16,
}

pub fn append_direct_placement(
	output: &mut String,
	id: u32,
	placement: DirectPlacement,
	tmux_passthrough: bool,
) {
	append_apc_start(output, tmux_passthrough);
	let DirectPlacement { source_x, source_y, source_width, source_height, rows, cols } = placement;
	let _ = write!(
		output,
		"a=p,q=2,C=1,i={id},p={id},x={source_x},y={source_y},w={source_width},h={source_height},\
		 c={cols},r={rows}"
	);
	append_apc_end(output, tmux_passthrough);
}

pub fn append_delete_image(output: &mut String, id: u32, tmux_passthrough: bool) {
	append_apc_start(output, tmux_passthrough);
	let _ = write!(output, "a=d,d=I,i={id},q=2");
	append_apc_end(output, tmux_passthrough);
}

pub fn append_tmux_passthrough(output: &mut String, payload: &str) {
	output.push_str(esc!(dcs, "tmux;"));
	for part in payload.split_inclusive('\x1b') {
		output.push_str(part);
		if part.ends_with('\x1b') {
			output.push('\x1b');
		}
	}
	output.push_str(esc!(st));
}

#[cfg(test)]
mod tests {
	use super::{DIACRITICS, placeholder_cell};
	use crate::Color;

	#[test]
	fn placeholder_uses_spec_base_coordinates_and_rgb_id() {
		assert_eq!(DIACRITICS.len(), 297);
		let (text, style) = placeholder_cell(0x12_34_56, 1, 2, 2, 4);
		assert_eq!(text.as_str(), "\u{10eeee}\u{030d}\u{030e}");
		assert_eq!(style.foreground_color(), Color::Rgb(0x12, 0x34, 0x56));
	}
}
