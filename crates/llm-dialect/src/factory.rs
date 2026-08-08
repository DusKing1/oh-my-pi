//! Concrete scanner construction for catalog-owned dialect identities.

use crate::{
	Dialect,
	scanner::{
		DeepSeekScanner, GeminiScanner, GemmaScanner, GlmScanner, HarmonyScanner, JsonTagScanner,
		KimiScanner, Scanner, XmlScanner,
	},
	types::{ScannerOptions, XmlTagset},
};

/// Constructs the concrete scanner for a catalog dialect.
///
/// The returned enum uses direct dispatch and never allocates a trait object.
#[must_use]
pub fn create_scanner(dialect: Dialect, options: ScannerOptions<'_>) -> Scanner {
	match dialect {
		Dialect::Glm => Scanner::Glm(GlmScanner::new(options)),
		Dialect::Hermes => Scanner::Hermes(JsonTagScanner::new(options)),
		Dialect::Kimi => Scanner::Kimi(KimiScanner::new(options)),
		Dialect::Xml => Scanner::Xml(XmlScanner::new(options, false)),
		Dialect::Anthropic => Scanner::Anthropic(XmlScanner::new(options, false)),
		Dialect::DeepSeek => Scanner::DeepSeek(DeepSeekScanner::new(options)),
		Dialect::Harmony => Scanner::Harmony(HarmonyScanner::new(options)),
		Dialect::Qwen3 => Scanner::Qwen3(JsonTagScanner::new(options)),
		Dialect::Gemini => Scanner::Gemini(GeminiScanner::new(options)),
		Dialect::Gemma => Scanner::Gemma(GemmaScanner::new(options)),
		Dialect::MiniMax => Scanner::MiniMax(XmlScanner::new(options, true)),
	}
}

/// Constructs the generic XML scanner with an explicit tag vocabulary.
#[must_use]
pub fn create_xml_scanner(mut options: ScannerOptions<'_>, tagset: XmlTagset) -> Scanner {
	options.xml_tagset = tagset;
	Scanner::Xml(XmlScanner::new(options, false))
}
