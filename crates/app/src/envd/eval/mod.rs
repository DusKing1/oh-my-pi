//! Eval helper prelude and authenticated host bridge.

mod bridge;
mod process;
#[cfg(test)]
mod tests;

#[cfg(test)]
pub use bridge::{
	BridgeCapabilities, BridgeDispatcher, BridgeHost, install_python_bridge, install_python_prelude,
};
pub use bridge::{
	BridgeHostError, BridgeNamespaceInstaller, EvalSessionConfig, ParentSessionHost,
	SessionBridgeHost,
};
pub use process::{EVAL_CHILD_ARG, ProcessError, ProcessEvalExec, run_eval_child_entry};

/// Python helpers installed once in every persistent eval namespace.
pub const PYTHON_PRELUDE: &str = include_str!("python_prelude.py");
