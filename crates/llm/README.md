# omp-llm

`omp-llm` is the unified facade for Oh My Pi's language-model integration crates. It gives callers one import surface for provider codecs, shared types, transport and egress infrastructure, gateway services, credential handling, middleware, and local runtimes.

## Structure

The crate re-exports its constituent crates as named modules:

- `anthropic`, `google`, `openai`, `cursor`, and `devin` expose provider-specific transport codecs and bridges.
- `types`, `error`, `transport`, and `egress` provide provider-independent values, error policy, wire transport, streaming, and HTTP egress.
- `broker`, `catalog`, `gateway`, and `tower` cover credentials, provider and model catalogs, service facades, and provider-attempt middleware.
- `local` and `fm` expose local inference runtimes and Apple Foundation Models support.

## Philosophy

This is a deliberately thin facade: implementations remain in focused crates, while this crate re-exports them under stable, discoverable module names. Provider-specific wire behavior stays separate from canonical types and shared transport policy, so consumers can use the unified surface without collapsing those boundaries.
