//! Typed client boundary for the `omp.env.v1` environment protocol.
//!
//! The crate correlates requests and streams server events over decoded frame
//! channels. It intentionally contains no filesystem, process, document,
//! workspace, blob-store, or tool-host implementation: those resources live
//! behind the environment service in both in-process and remote deployments.

mod client;
mod guard;

pub use client::{
	BlobDownload, BlobDownloadEvent, BlobUpload, ClientError, EnvClient, ExecEvent, ExecRun,
	InProcessEnvTransport, Invocation, InvocationEvent, ProcessAttachment, ProcessAttachmentEvent,
	RequestStream,
};
pub use guard::RunGuard;
/// Generated `omp.env.v1` wire frames used at transport boundaries.
pub use omp_proto::env::v1 as frame;
