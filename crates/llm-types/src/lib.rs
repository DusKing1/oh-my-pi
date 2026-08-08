//! Canonical, provider-independent inference values and facet traits.
//!
//! These values are the in-process interface. Protobuf messages are only their
//! transport binding at process boundaries.

/// Reconstructs canonical turn events from streamed parts.
pub mod accumulator;
/// Bridges canonical values to process-boundary protobuf messages.
pub mod convert;
/// Defines canonical streamed turn events and invocation frames.
pub mod event;
/// Declares provider-neutral capability facets.
pub mod facet;
/// Supplies stable identifiers used by canonical values.
pub mod ids;
/// Models media generation and transcription values.
pub mod media;
/// Stores extensible typed provider properties.
pub mod props;
/// Defines canonical requests, responses, and tool constraints.
pub mod request;
/// Defines canonical web search requests and results.
pub mod search;
/// Models text extraction and completion values.
pub mod text;
/// Represents portable conversation threads and messages.
pub mod thread;

pub use accumulator::{AccumulatorError, StreamAccumulator};
pub use convert::ConvertError;
pub use event::*;
pub use facet::*;
pub use ids::*;
pub use media::*;
pub use props::Props;
pub use request::*;
pub use search::*;
pub use text::*;
pub use thread::*;
