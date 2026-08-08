# omp-tui-macros

`omp-tui-macros` provides the `dom!` procedural macro used by `omp-tui`. It parses markup with a single root element and lowers elements, attributes, expressions, string children, and child-level `for`, `if`, and `match` control flow into `omp_tui` component-builder calls.

## Structure

The crate is implemented in `src/lib.rs`. Its token parser builds a small internal tree of elements, attributes, children, and control-flow branches. Validation handles expression syntax, matching tags, data records owned by components, and editor child constraints. The lowering layer maps known tags and properties to `omp_tui::components` and `omp_tui::Prop`, emits specialized records such as table rows and tree nodes, and uses `CustomElement` or custom properties for names outside those built-in mappings.

## Philosophy

Keep the markup declarative while producing ordinary component construction code at compile time. Reject malformed or context-invalid markup with errors at the relevant token span, preserve embedded Rust expressions and control flow, and keep component-specific rules in the macro so callers receive diagnostics before runtime.
