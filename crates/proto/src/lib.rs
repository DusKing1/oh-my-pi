//! Generated protobuf types for the workspace.
//!
//! `.proto` sources live in `proto/`; `build.rs` compiles them at build time
//! with protox + tonic-prost-build (no system `protoc` needed). Each protobuf
//! package maps to one module here — add an `include!` module when you add a
//! new package.
//!
//! These are transport bindings: in-process consumers use the native types
//! in their owning crates and only touch `omp-proto` at process boundaries.
//! Message types are always available. Enable the `tonic` feature to also
//! generate gRPC clients and servers; pure-type consumers keep those runtime
//! dependencies out of their graph.
//!
//! Every generated type also derives `serde::{Serialize, Deserialize}` —
//! Rust-native serde (enums as ints, `snake_case` fields), not the proto3
//! JSON mapping.
//!
//! # Example
//!
//! ```
//! use omp_proto::{prost::Message, thread::v1::Revision};
//!
//! let rev = Revision { head: 42, token: b"chain".as_ref().into() };
//!
//! // Protobuf round-trip.
//! let bytes = rev.encode_to_vec();
//! assert_eq!(Revision::decode(&bytes[..]).unwrap(), rev);
//!
//! // Serde round-trip.
//! let json = serde_json::to_string(&rev).unwrap();
//! assert_eq!(serde_json::from_str::<Revision>(&json).unwrap(), rev);
//! ```

// Re-exported so consumers use the same `prost` the codegen targeted
// (the `Message` trait is needed for encode/decode).
pub use prost;

/// Current wire-visible protobuf schema revision.
///
/// This is bumped for every wire-visible schema change and is the revision
/// compared by the `omp.gateway.v1.Hello` handshake.
pub const SCHEMA_REV: u32 = 3;

/// Generated packages under the protobuf `omp` namespace.
pub mod omp {
	/// Types generated from `omp.thread.v1`: the canonical conversation AST.
	pub mod thread {
		/// Version 1.
		pub mod v1 {
			#![allow(
				missing_docs,
				clippy::pedantic,
				clippy::nursery,
				reason = "prost/tonic output is machine-generated and cannot follow handwritten \
				          documentation and style conventions"
			)]
			#![allow(
				clippy::allow_attributes_without_reason,
				reason = "prost/tonic emits compatibility allow attributes without Rust reason \
				          metadata"
			)]
			#![allow(
				clippy::large_enum_variant,
				reason = "prost maps protobuf oneofs directly to enums; boxing would change the \
				          generated Rust API"
			)]
			include!(concat!(env!("OUT_DIR"), "/omp.thread.v1.rs"));
		}
	}

	/// Types generated from `omp.inference.v1`: inference turns and facets.
	pub mod inference {
		/// Version 1.
		pub mod v1 {
			#![allow(
				missing_docs,
				clippy::pedantic,
				clippy::nursery,
				reason = "prost/tonic output is machine-generated and cannot follow handwritten \
				          documentation and style conventions"
			)]
			#![allow(
				clippy::allow_attributes_without_reason,
				reason = "prost/tonic emits compatibility allow attributes without Rust reason \
				          metadata"
			)]
			#![allow(
				clippy::large_enum_variant,
				reason = "prost maps protobuf oneofs directly to enums; boxing would change the \
				          generated Rust API"
			)]
			include!(concat!(env!("OUT_DIR"), "/omp.inference.v1.rs"));
		}
	}

	/// Types generated from `omp.auth.v1`: authentication and credential flow.
	pub mod auth {
		/// Version 1.
		pub mod v1 {
			#![allow(
				missing_docs,
				clippy::pedantic,
				clippy::nursery,
				reason = "prost/tonic output is machine-generated and cannot follow handwritten \
				          documentation and style conventions"
			)]
			#![allow(
				clippy::allow_attributes_without_reason,
				reason = "prost/tonic emits compatibility allow attributes without Rust reason \
				          metadata"
			)]
			#![allow(
				clippy::large_enum_variant,
				reason = "prost maps protobuf oneofs directly to enums; boxing would change the \
				          generated Rust API"
			)]
			include!(concat!(env!("OUT_DIR"), "/omp.auth.v1.rs"));
		}
	}

	/// Types generated from `omp.gateway.v1`: connection pre-flight negotiation.
	pub mod gateway {
		/// Version 1.
		pub mod v1 {
			#![allow(
				missing_docs,
				clippy::pedantic,
				clippy::nursery,
				reason = "prost/tonic output is machine-generated and cannot follow handwritten \
				          documentation and style conventions"
			)]
			#![allow(
				clippy::allow_attributes_without_reason,
				reason = "prost/tonic emits compatibility allow attributes without Rust reason \
				          metadata"
			)]
			#![allow(
				clippy::large_enum_variant,
				reason = "prost maps protobuf oneofs directly to enums; boxing would change the \
				          generated Rust API"
			)]
			include!(concat!(env!("OUT_DIR"), "/omp.gateway.v1.rs"));
		}
	}

	/// Types generated from `omp.blob.v1`: content-addressed blob transfer.
	pub mod blob {
		/// Version 1.
		pub mod v1 {
			#![allow(
				missing_docs,
				clippy::pedantic,
				clippy::nursery,
				reason = "prost/tonic output is machine-generated and cannot follow handwritten \
				          documentation and style conventions"
			)]
			#![allow(
				clippy::allow_attributes_without_reason,
				reason = "prost/tonic emits compatibility allow attributes without Rust reason \
				          metadata"
			)]
			#![allow(
				clippy::large_enum_variant,
				reason = "prost maps protobuf oneofs directly to enums; boxing would change the \
				          generated Rust API"
			)]
			include!(concat!(env!("OUT_DIR"), "/omp.blob.v1.rs"));
		}
	}

	/// Types generated from `omp.document.v1`: document transactions, native
	/// watch invalidation, and synchronized LSP passthrough.
	pub mod document {
		/// Version 1.
		pub mod v1 {
			#![allow(
				missing_docs,
				clippy::pedantic,
				clippy::nursery,
				reason = "prost/tonic output is machine-generated and cannot follow handwritten \
				          documentation and style conventions"
			)]
			#![allow(
				clippy::allow_attributes_without_reason,
				reason = "prost/tonic emits compatibility allow attributes without Rust reason \
				          metadata"
			)]
			#![allow(
				clippy::large_enum_variant,
				reason = "prost maps protobuf oneofs directly to enums; boxing would change the \
				          generated Rust API"
			)]
			include!(concat!(env!("OUT_DIR"), "/omp.document.v1.rs"));
		}
	}
}

pub use omp::{auth, blob, document, gateway, inference, thread};
