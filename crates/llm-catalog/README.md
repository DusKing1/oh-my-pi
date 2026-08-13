# omp-llm-catalog

`omp-llm-catalog` defines the allocation-conscious, serializable vocabulary shared by OMP's inference catalog compiler and runtime. It keeps providers, routes, codecs, models, accounts, and opaque wire model identifiers structurally separate.

The crate contains facts rather than executable provider behavior. Router-facing `PolicyModel` values expose capabilities and policy identifiers but never raw wire model identifiers; only codec-facing `WireTarget` values carry those identifiers. Unknown capability evidence remains distinct from explicit lack of support.
