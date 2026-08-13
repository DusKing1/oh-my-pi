# omp-agent

`omp-agent` provides the transport-neutral foundations for OMP's durable,
interruptible agent loop. It keeps the canonical `omp.thread.v1.Item` as the
only conversation shape and composes the live turn protocol with immutable
configuration, deterministic prompt heads, ordered interrupts, event fan-out,
journal projection, supervised tool batches, and detached-job settlement.

`AgentState` publishes immutable snapshots through a watch value. Each turn can
re-read the latest options, enabled tools, workspace bytes, prompt source,
interrupt policy, deadline, and bounded retry policy without sharing mutable
configuration. Prompt sources are synchronous and receive an immutable
workspace capture; every render is repeated and compared before canonical
system items and their stable BLAKE3 hash are accepted.

The transcript journal is durable truth. Gateway context is only a working
copy: projection rebuilds canonical threads, applies amendments and rewinds,
and lets the live tool registry lift historical results. One flume mailbox
orders immediate, turn-boundary, and idle inputs. Tool calls execute only
through `omp-env`; event subscribers observe shared payloads without feeding
back into the loop, and detached jobs settle through the same mailbox.

The crate contains no provider, application, UI, docserver, or shell-engine
dependencies. It also does not yet own the top-level run loop: hosts compose
these foundations with `TurnClient`, while RPC and in-process sessions retain
the same generated protocol contract and stable `TurnId` replay behavior.
