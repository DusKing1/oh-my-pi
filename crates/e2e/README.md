# omp-e2e

`omp-e2e` is the non-publishable executable acceptance suite for the joined OMP agent mesh. It proves cross-crate behavior that cannot be established by a unit test: durable turn replay, real document and environment authority, cancellation ownership, detached settlement, schema isolation, incremental context, terminal UI behavior, and recorded performance baselines.

The crate is structurally split between scenario bodies and one shared authority harness. Integration tests own only P1–P8 sequencing and assertions. `src/support` owns scratch project/state roots, bounded synchronization, daemon and process-tree lifecycle, framed clients, scripted provider and turn seams, canonical wire builders, blob access, and transcript reopening. Scripts replace only nondeterministic model output; document, environment, process, blob, journal, context, and RPC authority remain production implementations.

Every wait is bounded. Every process, task, socket, and temporary root has an RAII owner so panic unwinding cannot leak test authority into a later scenario. Tests invoke Cargo-built binaries directly and never shell out to Cargo or duplicate daemon setup.
