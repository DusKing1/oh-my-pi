//! Eval helper prelude and authenticated host bridge.

mod bridge;
#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) use bridge::{
	BridgeCallError, BridgeCapabilities, BridgeClient, BridgeDispatcher, BridgeGrant, BridgeHost,
	BridgeRegistration, RegistryBridgeHost, install_python_bridge, install_python_prelude,
};
pub(crate) use bridge::{
	BridgeHostError, BridgeNamespaceInstaller, EvalSessionConfig, ParentSessionHost,
	SessionBridgeHost,
};

/// Python helpers installed once in every persistent eval namespace.
pub(crate) const PYTHON_PRELUDE: &str = include_str!("python_prelude.py");
