//! Batteries-included shell composition.
//!
//! This crate exposes the parser and execution API from [`omp_shell_engine`]
//! together with the utility and process builtin registries from
//! [`omp_shell_builtins`]. The `omp-sh` binary is the complete composition.

pub use omp_shell_builtins::{process_builtins, utility_builtins};
pub use omp_shell_engine::*;
