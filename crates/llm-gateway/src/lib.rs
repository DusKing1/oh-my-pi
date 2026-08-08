//! The gateway: the only egress path to model providers.
//!
//! It owns the server-side half of the incremental turn protocol — the
//! context store ([`context`]) and the turn engine ([`turn`]) — plus the
//! facet services and the foreign-wire facades ([`facade`]) that let stock
//! OpenAI/Anthropic SDKs drive the same machinery.
//!
//! Design rules that outrank convenience:
//! - **One registry, one gateway.** No tool, facet, or facade hand-rolls HTTP
//!   to a vendor; everything flows through the inference egress stack so auth,
//!   proxy, retry, limits, and metering apply uniformly.
//! - **Turns are atomic.** Context mutates only at commit; cancellation or
//!   error leaves the context at its pre-turn revision.
//! - **Every facet gets a foreign route.** Capabilities reachable over
//!   `omp.inference.v1` are also reachable over the official vendor paths.

pub mod audio_backends;
pub mod blob;
pub mod context;
pub mod discovery;
pub mod facade;
pub mod federation;
pub mod image_backends;
pub mod images;
pub mod inference;
pub mod listener;
pub mod local;
pub mod media;
pub mod routes;
pub mod search;
pub mod search_backends;
pub mod turn;
pub mod videos;
