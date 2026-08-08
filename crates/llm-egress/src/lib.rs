#![feature(impl_trait_in_assoc_type)]

//! HTTP egress is split into two Tower stacks.
//!
//! The byte stack owns transport concerns (proxy selection, authentication,
//! limits, retry, TLS, and HTTP pooling). The typed stack owns transport
//! encoding and normalization into inference events.
//!
//! The boundary between them is the retry **commit point**. An egress service
//! future does not resolve merely because response headers arrived: it resolves
//! only after the first meaningful provider event has also decoded and
//! validated. Consequently `tower::retry` sees precisely the replayable
//! window. Before that boundary every request body is buffered and can be
//! replayed; after it, transport and provider failures are emitted as stream
//! items rather than service errors. This type boundary replaces the previous
//! implementation's mutable “replay-unsafe content pushed” flag.

pub mod auth_inject;
pub mod client;
pub mod limits;
pub mod proxy;
pub mod retry;
