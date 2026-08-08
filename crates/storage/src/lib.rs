//! Session storage: the content-addressed blob store and the transcript v4
//! append-only event log (see `TRANSCRIPT-V4.md` at the repo root).
//!
//! Two invariants rule everything here:
//! - **Append-only**: nothing written is ever edited; after-the-fact state is
//!   later events referencing earlier indexes.
//! - **Every byte exists in exactly one place**: neutral projections live in
//!   blocks, provider-native residue lives in replay capsules, large payloads
//!   live in the blob store behind typed [`blob::BlobRef`]s.

pub mod blob;
pub mod transcript;
