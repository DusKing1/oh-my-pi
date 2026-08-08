use std::fmt::Write as _;

use omp_core::encoding::base64;

use crate::{
	Graphics,
	escape::esc,
	frame::Frame,
	kitty::append_tmux_passthrough,
	renderer::{image_placement, move_cursor_row},
};

const IMAGE_FILENAME: &str = "image.png";

/// A registered PNG made available to the iTerm2 image post-pass.
#[derive(Clone, Copy)]
pub struct Iterm2Image<'a> {
	/// The typed image ID referenced by frame cells.
	pub(crate) id:  u32,
	/// The original registered PNG bytes.
	pub(crate) png: &'a [u8],
}

/// The document rows currently visible in the terminal viewport.
#[derive(Clone, Copy)]
pub struct Iterm2Viewport {
	/// First visible document row.
	pub(crate) top:    u16,
	/// Number of visible rows.
	pub(crate) height: u16,
}

/// Emits fully visible iTerm2 inline-image placements after the text pass.
///
/// iTerm2's protocol cannot select a source crop. Unlike Kitty Direct and
/// sixel, a placement intersecting a viewport edge is therefore omitted until
/// its complete cell box is visible.
#[allow(clippy::too_many_arguments, reason = "mirrors the renderer image post-pass inputs")]
pub fn iterm2_output<'a>(
	graphics: Graphics,
	images: impl IntoIterator<Item = Iterm2Image<'a>>,
	frame: &Frame,
	window: Iterm2Viewport,
	previous: Option<(&Frame, Iterm2Viewport)>,
	damaged: Option<&[(u16, u16)]>,
	force: bool,
	tmux_passthrough: bool,
) -> String {
	if graphics != Graphics::Iterm2 || window.height == 0 {
		return String::new();
	}

	let mut output = String::new();
	let mut cursor_row = window.height - 1;
	for image in images {
		let placement = image_placement(frame, image.id);
		let Some((top, left, rows, cols)) = placement else {
			continue;
		};
		let bottom = top.saturating_add(rows);
		let right = left.saturating_add(cols);
		let window_bottom = window.top.saturating_add(window.height);
		if top < window.top
			|| bottom > window_bottom
			|| bottom > frame.size().height
			|| right > frame.size().width
		{
			continue;
		}

		let moved = previous.is_none_or(|(previous_frame, previous_window)| {
			image_placement(previous_frame, image.id) != placement
				|| previous_window.top != window.top
				|| previous_window.height != window.height
		});
		let intersects_damage = damaged.is_some_and(|ranges| {
			ranges
				.iter()
				.any(|&(start, end)| start < bottom && end > top)
		});
		let changed = damaged.is_none()
			&& previous.is_some_and(|(previous_frame, _)| {
				(top..bottom).any(|row| !previous_frame.row_equals(row, frame, row))
			});
		if !(force || moved || intersects_damage || changed) {
			continue;
		}

		move_cursor_row(&mut output, &mut cursor_row, top - window.top);
		output.push('\r');
		if left > 0 {
			let _ = write!(output, esc!(cursor_forward), left);
		}
		append_inline_image(&mut output, image.png, cols, rows, tmux_passthrough);
	}
	if !output.is_empty() {
		move_cursor_row(&mut output, &mut cursor_row, window.height - 1);
		output.push('\r');
	}
	output
}

fn append_inline_image(
	output: &mut String,
	png: &[u8],
	cols: u16,
	rows: u16,
	tmux_passthrough: bool,
) {
	let payload_len = base64::encode_len(png.len());
	let mut sequence = String::with_capacity(payload_len.saturating_add(128));
	let _ = write!(
		sequence,
		esc!(osc, "1337;File=inline=1;size={};width={};height={};preserveAspectRatio=1;name=",),
		png.len(),
		cols,
		rows,
	);
	sequence.extend(base64::encode(IMAGE_FILENAME.as_bytes()).map(char::from));
	sequence.push(':');
	sequence.extend(base64::encode(png).map(char::from));
	sequence.push_str(esc!(bel));
	if tmux_passthrough {
		append_tmux_passthrough(output, &sequence);
	} else {
		output.push_str(&sequence);
	}
}

#[cfg(test)]
mod tests {
	use super::{Iterm2Image, Iterm2Viewport, iterm2_output};
	use crate::{Graphics, Size, frame::Frame};

	const PNG: &[u8] = b"\x89PNG\r\n\x1a\nsmall";
	const INLINE: &str = "\x1b]1337;File=inline=1;size=13;width=8;height=4;preserveAspectRatio=1;\
	                      name=aW1hZ2UucG5n:iVBORw0KGgpzbWFsbA==\x07";

	fn image_frame(first_row: u16) -> Frame {
		let mut frame = Frame::new(Size::new(8, 4));
		for row in first_row..4 {
			for col in 0..8 {
				frame.put_image_cell(col, row, 7, row, col, 4, 8);
			}
		}
		frame
	}

	fn output(frame: &Frame, window: Iterm2Viewport, tmux: bool) -> String {
		iterm2_output(
			Graphics::Iterm2,
			[Iterm2Image { id: 7, png: PNG }],
			frame,
			window,
			None,
			None,
			true,
			tmux,
		)
	}

	#[test]
	fn emits_byte_exact_inline_image_for_cell_box() {
		let actual = output(&image_frame(0), Iterm2Viewport { top: 0, height: 4 }, false);
		assert_eq!(actual, format!("\x1b[3A\r{INLINE}\x1b[3B\r"));
	}

	#[test]
	fn wraps_inline_image_for_tmux() {
		let actual = output(&image_frame(0), Iterm2Viewport { top: 0, height: 4 }, true);
		let escaped = INLINE.replace('\x1b', "\x1b\x1b");
		assert_eq!(actual, format!("\x1b[3A\r\x1bPtmux;{escaped}\x1b\\\x1b[3B\r"));
	}

	#[test]
	fn omits_partially_clipped_placement() {
		let actual = output(&image_frame(2), Iterm2Viewport { top: 2, height: 2 }, false);
		assert_eq!(actual, "");
	}
}
