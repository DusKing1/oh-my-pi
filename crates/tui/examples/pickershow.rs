//! Throwaway: render the chat picker overlay once to the real terminal so
//! the kitty-placeholder logos can be screenshotted. Exits after 25s.

use std::io::{self, Write};

use omp_tui::{Frame, Renderer, Size, UiContext, detect};

#[path = "chat/picker.rs"]
mod picker;

fn main() -> io::Result<()> {
	let caps = detect();
	eprintln!("graphics tier: {:?}", caps.graphics);
	let mut renderer = Renderer::new(io::stdout());
	renderer.apply_caps(&caps)?;
	let (cols, rows) = term_size();
	let viewport = Size::new(cols, rows);
	let base = Frame::new(viewport);
	renderer.present(base.clone(), viewport.height, 0)?;

	let mut overlay = picker::ModelPicker::open(0, &UiContext::default().with_terminal_caps(&caps));
	let layer = overlay.layer(viewport);
	renderer.present_overlaid(&base, &[], viewport.height, 0, std::slice::from_ref(&layer))?;
	io::stdout().flush()?;
	std::thread::sleep(std::time::Duration::from_secs(25));
	Ok(())
}

fn term_size() -> (u16, u16) {
	use nix::libc;
	let mut size = libc::winsize { ws_row: 0, ws_col: 0, ws_xpixel: 0, ws_ypixel: 0 };
	// SAFETY: TIOCGWINSZ on stdout with a valid winsize out-pointer.
	if unsafe { libc::ioctl(1, libc::TIOCGWINSZ, &raw mut size) } == 0 && size.ws_col > 0 {
		(size.ws_col, size.ws_row)
	} else {
		(100, 40)
	}
}
