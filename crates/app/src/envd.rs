//! Project environment-daemon assembly and production serving.

pub mod blobs;
pub mod docs;
pub mod exec;
pub mod server;
pub mod worker;
pub mod workspace;
pub use server::EnvdError;

use crate::cli::EnvdArgs;

/// Starts the project environment daemon and serves until process shutdown.
pub async fn run(args: EnvdArgs) -> crate::Result<()> {
	Ok(server::run(args).await?)
}
