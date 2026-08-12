//! Canonical archive-relative path handling.

use omp_core::{Str, StrMut};

use crate::{Error, Limits, Result};

/// Normalizes separators and dot components while rejecting root escapes.
pub fn normalize(raw: &str, allow_empty: bool) -> Option<Str> {
	if raw.starts_with(['/', '\\']) || raw.contains('\0') {
		return None;
	}

	let mut normalized = StrMut::with_capacity(raw.len());
	let mut first = true;
	for part in raw.split(['/', '\\']) {
		if part.is_empty() || part == "." {
			continue;
		}
		if part == ".." || first && is_windows_drive(part) {
			return None;
		}
		if !first {
			normalized.push('/');
		}
		normalized.push_str(part);
		first = false;
	}

	if normalized.is_empty() && !allow_empty {
		None
	} else {
		Some(normalized.freeze())
	}
}
/// Normalizes a writer path and enforces the configured archive path bounds.
pub fn normalize_bounded(raw: &str, limits: Limits) -> Result<Str> {
	let path = normalize(raw, false).ok_or_else(|| Error::UnsafePath(Str::new(raw)))?;
	validate(&path, limits)?;
	Ok(path)
}

/// Enforces byte-length and component-depth bounds on a normalized path.
pub fn validate(path: &str, limits: Limits) -> Result<()> {
	if path.len() as u64 > limits.path_size {
		return Err(Error::PathTooLong { actual: path.len() as u64, limit: limits.path_size });
	}
	let depth =
		path.bytes().filter(|byte| *byte == b'/').count() as u64 + u64::from(!path.is_empty());
	if depth > limits.path_depth {
		return Err(Error::PathTooDeep { actual: depth, limit: limits.path_depth });
	}
	Ok(())
}

#[inline]
const fn is_windows_drive(component: &str) -> bool {
	let bytes = component.as_bytes();
	bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':'
}

#[inline]
pub fn is_directory_name(raw: &str) -> bool {
	raw.ends_with(['/', '\\'])
}

#[inline]
pub fn parent(path: &str) -> &str {
	path.rsplit_once('/').map_or("", |(parent, _)| parent)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn normalizes_portable_relative_paths() {
		assert_eq!(normalize("./a\\b//c", false).as_deref(), Some("a/b/c"));
		assert_eq!(normalize("", true).as_deref(), Some(""));
	}

	#[test]
	fn rejects_paths_that_can_escape_a_destination() {
		for path in ["../x", "a/../x", "/x", "\\x", "C:/x", "C:relative", "./C:\\x", "a\0b"] {
			assert_eq!(normalize(path, true), None, "{path}");
		}
	}
}
