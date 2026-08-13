//! macOS window polish for the borderless, transparent shell: a compositor
//! shadow shaped by the window's alpha, and WindowServer blur behind the
//! translucent regions.
//!
//! Blur uses `CGSSetWindowBackgroundBlurRadius` — the classic terminal
//! approach (Alacritty ships the same call). An `NSVisualEffectView` is
//! deliberately NOT used: it attaches as a subview of the content view and
//! subviews composite *above* the view's backing `CAMetalLayer`, occluding
//! everything wgpu renders.

use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::window::Window;

// WindowServer private API, resolved through AppKit's transitive SkyLight
// link: blurs whatever sits behind the window's translucent pixels. The
// signatures mirror winit's own declarations (`ffi.rs`): the connection is
// pointer-sized — a narrower return would truncate it on arm64.
unsafe extern "C" {
	fn CGSMainConnectionID() -> *mut std::ffi::c_void;
	fn CGSSetWindowBackgroundBlurRadius(
		connection: *mut std::ffi::c_void,
		window: isize,
		radius: i64,
	) -> i32;
}

/// Enables the compositor shadow and blur-behind for `window`.
pub fn polish(window: &Window) {
	let Ok(handle) = window.window_handle() else {
		return;
	};
	let RawWindowHandle::AppKit(appkit) = handle.as_raw() else {
		return;
	};
	// SAFETY: the handle is this window's content NSView, valid for its
	// lifetime; all calls are main-thread safe and `polish` runs during
	// window creation on the main thread.
	unsafe {
		let ns_view = &*(appkit.ns_view.as_ptr() as *const objc2_app_kit::NSView);
		let Some(ns_window) = ns_view.window() else {
			return;
		};
		ns_window.setHasShadow(true);
		let number = ns_window.windowNumber() as isize;
		CGSSetWindowBackgroundBlurRadius(CGSMainConnectionID(), number, 24);
	}
}
