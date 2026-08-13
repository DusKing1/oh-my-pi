#![feature(impl_trait_in_assoc_type)]
//! Typed, capability-complete inference contracts over one Tower service spine.
//!
//! Public callers retain operation-specific request and output types. Provider
//! registries erase that surface once, at construction, into
//! [`ProviderService`].

pub(crate) use omp_llm_catalog as catalog;

pub mod account;
pub mod answer;
pub mod auth;
pub mod body;
pub mod call;
pub mod client;
pub mod codec;
pub mod error;
pub mod event;
pub mod gate;
pub mod id;
pub mod layer;
#[cfg(feature = "local")]
pub mod local;
pub mod operation;
pub mod plan;
pub mod provider;
pub mod receipt;
pub mod recovery;
pub mod registry;
pub mod router;
pub mod session;
pub mod staging;
pub mod transport;

pub use answer::*;
pub use call::*;
pub use client::*;
pub use error::*;
pub use event::*;
pub use id::*;
pub use layer::{
	answer::AnswerLayer,
	recover::{DiscoveryProjector, RecoveryLayer},
};
pub use omp_llm_catalog::{
	capability::*,
	id::*,
	model::{ModelSpec, PolicyModel, WireTarget},
};
pub use plan::{ExecutionPlan, Planner};
pub use provider::ProviderService;
pub use receipt::*;
pub use registry::{Registry, RegistryBuilder, RouteUnavailable};
