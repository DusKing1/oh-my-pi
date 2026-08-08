# omp-slopjson

`omp-slopjson` parses the imperfect JSON commonly produced in LLM tool-call arguments. Its tolerant grammar accepts forms such as single-quoted strings, unquoted keys, Python literals, comments, trailing commas, invalid escapes, and bareword values while retaining explicit failure behavior for inputs that should not be executed. The crate also provides a JSON value model, Serde deserialization, strict prefix classification, streaming recovery, and text-level repair.

## Structure

- `parser` implements the shared tolerant lexer and token readers, including nesting limits and relaxed number handling.
- `de` drives one-pass Serde deserialization and exposes `from_str` and `parse`; `error` defines parse failures.
- `streaming` builds best-effort values from incomplete buffers, auto-closing containers and rolling incomplete atoms back to the last valid prefix.
- `classify` performs strict RFC 8259 prefix classification, while `repair` repairs escapes and control characters when callers need strict JSON text.
- `value` defines `Value`, `Number`, and insertion-ordered `Object` types with compact JSON serialization; `raw` and `macros` provide raw-value support and the `json!` construction macro.

## Philosophy

Tolerance is confined to known model-output malformations rather than treating every damaged input as usable. Complete parsing remains fallible and rejects trailing garbage, truncation, excessive nesting, malformed numbers, and non-finite values so callers do not silently act on a partial tool call. Streaming parsing is deliberately separate and total: it materializes only the recoverable prefix for display or incremental processing. Strict prefix classification likewise avoids repair because its purpose is to preserve corruption signals. Shared tokenization keeps these modes consistent without conflating their different safety contracts.
