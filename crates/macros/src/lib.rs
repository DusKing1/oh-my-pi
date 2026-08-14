//! Workspace procedural macros.
//!
//! Entry points live here; each macro's implementation owns a module. The
//! [`dom!`] declarative markup macro lowers component-builder calls for
//! `omp-tui`.

use proc_macro::TokenStream;

mod dom;

/// Builds one component tree from markup with child-level `for`, `if`, and
/// `match` control flow.
#[proc_macro]
pub fn dom(input: TokenStream) -> TokenStream {
	match dom::expand(input.into()) {
		Ok(tokens) => tokens.into(),
		Err(error) => error.into_compile_error().into(),
	}
}
