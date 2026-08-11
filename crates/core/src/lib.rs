#![feature(min_specialization)]
#![feature(core_intrinsics)]
#![feature(const_eval_select)]
#![feature(extend_one)]
#![feature(maybe_uninit_uninit_array_transpose)]
#![feature(type_alias_impl_trait)]
#![allow(
	internal_features,
	reason = "core_intrinsics is required for const_eval_select in encoding"
)]

//! Core data structures and utilities for `omp`.

pub mod append_vec;
pub mod cow_bytes;
pub mod encoding;
pub mod sparse_index;
pub mod sparse_map;
pub mod sparse_set;
pub mod str;

pub use append_vec::{AppendSlice, AppendVec};
pub use cow_bytes::CowBytes;
pub use encoding::{base32, base32_dns, base32_hex, base64, base64_url, hex};
pub use sparse_map::SparseMap;
pub use sparse_set::SparseSet;
pub use str::{CowStr, IntoStr, Str, StrMut};
