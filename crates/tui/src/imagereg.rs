//! Process-global interning of `<img src>` sources for typed image cells.
//!
//! [`crate::components::Img`] interns a PNG source once and paints typed
//! image cells carrying the returned ID; the renderer resolves IDs it has
//! never seen through [`bytes`] and uploads them on first reference, so
//! applications never touch terminal image IDs. Mirrors the hyperlink
//! interner in [`crate::frame`].
//!
//! IDs are allocated downward from the top of Kitty's 24-bit range so they
//! cannot collide with the low IDs applications typically pass to
//! [`crate::Renderer::register_image`].

use std::{collections::HashMap, sync::LazyLock};

use omp_core::{CowBytes, Str};
use parking_lot::Mutex;

use crate::imagefmt::{self, ImageDimensions};

/// One interned source: terminal image ID, PNG bytes, and probed dimensions.
#[derive(Clone)]
pub struct InternedImage {
	pub(crate) id:         u32,
	pub(crate) png:        CowBytes<'static>,
	pub(crate) dimensions: ImageDimensions,
}

#[derive(Default)]
struct Registry {
	/// Source path → interned entry; failures cache as `None` so missing
	/// files are probed once, not every rebuild.
	by_source: HashMap<Str, Option<InternedImage>>,
	by_id:     HashMap<u32, CowBytes<'static>>,
	allocated: u32,
}

static IMAGES: LazyLock<Mutex<Registry>> = LazyLock::new(|| Mutex::new(Registry::default()));

/// Interns a PNG file source, returning its stable terminal image ID and
/// pixel dimensions. Non-PNG or unreadable sources return `None` (the
/// half-block decoder handles those tiers separately).
pub fn intern(source: &str) -> Option<InternedImage> {
	let mut registry = IMAGES.lock();
	if let Some(cached) = registry.by_source.get(source) {
		return cached.clone();
	}
	let interned = load(source, registry.allocated);
	if interned.is_some() {
		registry.allocated += 1;
	}
	registry
		.by_source
		.insert(Str::from(source), interned.clone());
	if let Some(entry) = &interned {
		registry.by_id.insert(entry.id, entry.png.clone());
	}
	interned
}

/// PNG bytes for a registry-allocated ID, for renderer-side upload.
pub fn bytes(id: u32) -> Option<CowBytes<'static>> {
	let registry = IMAGES.lock();
	registry.by_id.get(&id).cloned()
}

fn load(source: &str, allocated: u32) -> Option<InternedImage> {
	let id = 0x00ff_ffff_u32.checked_sub(allocated)?;
	let png = std::fs::read(source).ok()?;
	// Kitty transmissions are sent as `f=100`: PNG only.
	if !png.starts_with(b"\x89PNG\r\n\x1a\n") {
		return None;
	}
	let dimensions = imagefmt::dimensions(&png)?;
	Some(InternedImage { id, png: CowBytes::from(png), dimensions })
}
