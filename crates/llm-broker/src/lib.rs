//! The auth broker: sole owner of provider credentials.
//!
//! The security boundary is the crate boundary. Provider token bytes are
//! constructible only inside [`sealed`], are `Debug`-redacted, zeroize on
//! drop, and **never serialize into any RPC**. Clients receive credential
//! *metadata* ([`service`], `omp.auth.v1`) and drive login flows; the only
//! sanctioned token egress is a short-TTL scoped token for facets that
//! require a client-direct connection.
//!
//! Because the daemon is the sole owner, refresh is a single in-process
//! single-flight — none of the cross-process lease machinery the previous
//! implementation needed (SQLite leases, CAS write-back, busy-timeout races)
//! exists here.

pub mod cli;
pub mod oauth;
pub mod sealed;
pub mod service;
pub mod source;
pub mod store;
pub mod usage;

pub use cli::BrokerCliBackend;
pub use source::{BrokerCredentialSource, BrokerCredentialSourceError, CredentialRefresher};
pub use usage::BrokerObserver;
