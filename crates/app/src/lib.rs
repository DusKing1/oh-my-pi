#![recursion_limit = "256"]

//! The production Codex route intentionally keeps its concrete Tower type;
//! this recursion limit covers compiler trait normalization without boxing the
//! runtime transport path.
//! Production application composition and command dispatch.

use clap::Parser as _;

pub mod agent;
pub mod auth_backend;
pub mod cli;
pub mod daemon;
pub mod discovery;

/// Parses process arguments and runs the selected production operation.
pub async fn run() -> anyhow::Result<()> {
	cli::dispatch(cli::OmpCli::parse()).await
}
