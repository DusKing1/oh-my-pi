# omp-llm-catalog

`omp-llm-catalog` defines OMP's provider and model catalogs. It loads curated provider policy, translates bundled model metadata into client-facing model cards, discovers runtime model sets, and joins those inputs with credential availability in a resumable registry.

## Structure

- `provider` loads provider endpoints, authentication, transports, facets, and compatibility policy from curated TOML.
- `models` decodes the compressed generated model catalog, exposes normalized model cards, and calculates token costs.
- `discovery` obtains live model listings through an injected HTTP client, including local and authenticated OpenAI-compatible providers.
- `registry` combines bundled and configured cards, discovery results, and an injected credential view; it exposes filtered snapshots and epoch-bearing deltas.
- `compat` and `identity` represent provider compatibility rules and model lineage, reseller references, and effort variants.
- `overlay` merges built-in, user, and project provider configuration at field level, while `oauth_params` loads data-driven login-flow parameters.

## Philosophy

Provider behavior and model metadata change at different rates, so the crate keeps them separate: reviewable TOML owns endpoints, authentication, transport, and compatibility, while generated JSON owns model identities, pricing, and limits. Runtime discovery handles providers whose models cannot be known statically. Broker concerns remain outside the crate through injected HTTP and credential interfaces, keeping catalog data and registry behavior independent of credential ownership.
