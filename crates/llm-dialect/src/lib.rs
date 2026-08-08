//! Owned model-prompt dialect rendering and incremental in-band event scanning.
//!
//! Provider transports encode HTTP or RPC wire protocols. This crate instead
//! owns model-authored prompt syntax: tool inventories, transcript projection,
//! and scanners that recover canonical text, thinking, and tool events.

pub mod coercion;
pub mod demotion;
pub mod factory;
pub mod history;
pub mod inventory;
pub mod projector;
pub mod prompt;
pub mod rendering;
pub mod scanner;
pub mod thinking;
mod tool;
pub mod types;

pub use omp_llm_catalog::identity::{
	DIALECT_ENV, Dialect, DialectSelection, FALLBACK_DIALECT, ParseDialectError, preferred_dialect,
};
pub use types::{
	DialectError, DialectRenderOptions, DialectResult, DialectToolResult, InbandTool, ScanBatch,
	ScanEvent, ScannerOptions, ToolExample, XmlTagset,
};
