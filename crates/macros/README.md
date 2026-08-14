# omp-macros

Workspace procedural macros. One proc-macro crate hosts every derive and
function-like macro so consumers pay for a single macro dependency; each
macro's implementation owns a module and only the entry points live at the
crate root.

## Macros

- `dom!` — parses declarative markup with a single root element and lowers
  elements, attributes, expressions, string children, and child-level `for`,
  `if`, and `match` control flow into `omp_tui` component-builder calls.
  Re-exported as `omp_tui::dom`.

