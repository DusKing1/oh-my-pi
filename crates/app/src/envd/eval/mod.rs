//! Eval helper prelude and authenticated host bridge.

mod bridge;
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

/// Python helpers installed once in every persistent eval namespace.
pub const PYTHON_PRELUDE: &str = include_str!("python_prelude.py");
