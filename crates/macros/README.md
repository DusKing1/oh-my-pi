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
- `#[derive(ToolParam)]` — the workspace's in-house schemars replacement for
  the tool-argument boundary. Reads the same metadata serde uses to
  deserialize model arguments (doc comments, `#[serde(rename/rename_all/
  default/skip/deny_unknown_fields)]`) and generates a deterministic, inline,
  model-facing JSON Schema: doc comments become descriptions, `Option<T>`
  follows the absent-property convention, and numbers never grow `format`
  annotations. Schema-only knobs (`description` overrides, numeric and length
  bounds, `nullable`, raw `extend({ ... })` JSON) ride a single
  `#[param(...)]` attribute. Runtime schema assembly lives in `omp-tool`
  (`ToolParam` trait + `__private` helpers); the derive only emits calls into
  it. Re-exported as `omp_tool::ToolParam`.
