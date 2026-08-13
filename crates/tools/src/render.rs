use omp_core::Str;
use omp_tool::{Part, PromptCaps};

/// Accumulates whole UTF-8 fragments without splitting a caller-owned unit.
pub(crate) struct TextProjection {
	text:      String,
	max_bytes: usize,
	truncated: bool,
}

impl TextProjection {
	pub(crate) fn new(caps: &PromptCaps) -> Option<Self> {
		(caps.maximum_parts != 0 && caps.maximum_text_bytes != 0).then(|| Self {
			text:      String::new(),
			max_bytes: usize::try_from(caps.maximum_text_bytes).unwrap_or(usize::MAX),
			truncated: false,
		})
	}

	pub(crate) fn push(&mut self, fragment: &str) -> bool {
		if self.text.len().saturating_add(fragment.len()) > self.max_bytes {
			self.truncated = true;
			return false;
		}
		self.text.push_str(fragment);
		true
	}

	pub(crate) fn finish(mut self) -> Vec<Part> {
		const MARKER: &str = "\n[truncated]";
		if self.truncated && self.text.len().saturating_add(MARKER.len()) <= self.max_bytes {
			self.text.push_str(MARKER);
		}
		if self.text.is_empty() {
			Vec::new()
		} else {
			vec![Part::Text { text: Str::from(self.text) }]
		}
	}
}
