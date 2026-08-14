//! Workspace procedural macros.
//!
//! Entry points live here; each macro's implementation owns a module:
//!
//! - [`dom!`](macro@dom) — declarative markup lowering to `omp_tui`
//!   component-builder calls.
//! - [`ToolParam`](macro@ToolParam) — model-facing JSON Schema derive for
//!   `omp_tool` argument structs.

use proc_macro::TokenStream;

mod dom;
mod tool_param;

/// Builds one component tree from markup with child-level `for`, `if`, and
/// `match` control flow.
#[proc_macro]
pub fn dom(input: TokenStream) -> TokenStream {
	match dom::expand(input.into()) {
		Ok(tokens) => tokens.into(),
		Err(error) => error.into_compile_error().into(),
	}
}

/// Derives `omp_tool::ToolParam` for tool parameter structs and enums.
#[proc_macro_derive(ToolParam, attributes(param))]
pub fn derive_tool_param(input: TokenStream) -> TokenStream {
	tool_param::derive(input.into()).into()
}
