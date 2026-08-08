//! Provider and model catalogs.
//!
//! Curated TOML owns the deliberately small provider surface: endpoint,
//! authentication, transport, and compatibility policy. Generated JSON owns the
//! high-churn model surface: model identities, pricing, and token/output
//! limits. Keeping those sources separate makes provider quirks reviewable
//! while model metadata can be regenerated without hand-editing runtime policy.

pub mod compat;
pub mod discovery;
pub mod identity;
pub mod models;
pub mod oauth_params;
pub mod overlay;
pub mod provider;
pub mod registry;

pub use omp_llm_types::Effort;
pub use provider::{CodexTransportPreference, TransportId};
