//! Typed facet middleware.
//!
//! Provider attempt routes use [`builder::RouteStackBuilder`] to compose the
//! replay-capable service stack once at registration. Catalog-selected
//! watchdog, healing, loop detection, forced-tool escalation, and resampling
//! are part of that concrete stack rather than caller-side helpers.

pub mod builder;
pub mod capability;
pub mod combinators;
pub mod meter;
pub mod rotation;
pub mod routing;
pub mod tokenizer;
