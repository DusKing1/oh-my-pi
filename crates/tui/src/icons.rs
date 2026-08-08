//! Typed semantic icons with ASCII, Unicode, and Nerd Font fallbacks.
//!
//! [`Icon`] is generated from `icons.tsv`, the reviewable source of truth.
//! Callers choose meaning once and defer presentation to [`crate::Charset`].

include!(concat!(env!("OUT_DIR"), "/icons.rs"));
