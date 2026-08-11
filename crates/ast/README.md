# omp-ast

`omp-ast` provides Tree-sitter-backed source understanding and structural editing for omp. It centralizes supported-language selection and inference, AST-aware block resolution, structural search and rewrite operations, and compact source summaries.

## Structure

- `language` defines supported languages, parser integration, and the exported `SupportLang` type.
- `block` resolves syntax-aware source blocks.
- `ops` implements structural search and rewrite operations.
- `summary` produces structural summaries of source files.

## Philosophy

The crate keeps parsing and language support behind one shared interface while separating block resolution, transformations, and summarization into focused modules. Operations work from syntax trees rather than treating source code as undifferentiated text, so callers can reason about language structure explicitly.
