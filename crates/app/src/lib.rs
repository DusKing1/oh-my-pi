#![recursion_limit = "256"]

//! The production Codex route intentionally keeps its concrete Tower type;
//! this recursion limit covers compiler trait normalization without boxing the
//! runtime transport path.
//! Production application composition and command dispatch.

use clap::Parser as _;

pub mod auth_backend;
pub mod auth_rpc;
pub mod blob_rpc;
pub mod chat;
mod chat_ui;
pub mod cli;
pub mod daemon;
pub mod discovery;
pub mod endpoint;
pub mod envd;
pub mod project_state;
pub mod rpc_adapter;

pub use miette::{IntoDiagnostic, Report, Result};

/// Parses process arguments and runs the selected production operation.
#[expect(
	clippy::future_not_send,
	reason = "the chat command runs a thread-confined terminal UI future"
)]
pub async fn run() -> Result<()> {
	cli::dispatch(cli::OmpCli::parse()).await
}
