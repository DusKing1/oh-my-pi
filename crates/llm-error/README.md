# omp-llm-error

`omp-llm-error` turns inconsistent LLM provider failures into structured classifications and actionable recovery advice. It combines error envelopes, HTTP and provider evidence, extracted retry metadata, and message patterns so callers can make consistent decisions about retries, credential rotation, blocking, and terminal failures.

## Structure

- `envelope` walks known provider and proxy error-body shapes.
- `classify`, `evidence`, `kind`, and `patterns` combine structural and textual evidence into error kinds, fidelity, and classification metadata.
- `extract`, `oauth`, and `rate_limit` extract retry hints and represent OAuth and rate-limit details.
- `policy` converts classifications into ordered actions and provides retry-budget and credential-blocking primitives.
- The crate root re-exports the main evidence, classification, error-kind, rate-limit, OAuth, and policy types.

## Philosophy

Provider-specific wire formats should be handled at the classification boundary rather than spread through callers. Classification preserves the strength and details of the available evidence; policy remains a separate step that turns those facts into explicit recovery actions. Stateful retry and blocking decisions are represented by focused primitives, allowing integration layers to apply recovery consistently without hiding why an action was chosen.
