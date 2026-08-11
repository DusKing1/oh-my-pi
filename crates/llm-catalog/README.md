# omp-llm-catalog

`omp-llm-catalog` defines OMP's provider and model catalogs. It loads curated provider policy, translates bundled model metadata into client-facing model cards, discovers runtime model sets, and joins those inputs with credential availability in a resumable registry.

## Structure

- `provider` loads provider endpoints, authentication, transports, facets, and compatibility policy from curated TOML.
- `models` decodes the compressed generated model catalog, exposes normalized model cards, and calculates token costs.
- `discovery` splits live model listing along the credential boundary: `Transport` is implemented once by the runtime and injects credentials, while `ProviderDiscovery` is implemented by each provider's own `omp-llm-*` crate. `Discovery` joins them, serving listing conventions shared by many providers (OpenAI `GET /models`, Ollama tags, Google pagination) itself and dispatching `DiscoveryKind::Specialized` rows to the protocol registered for their `TransportId`.
- `registry` combines bundled and configured cards, discovery results, and an injected credential view; it exposes filtered snapshots and epoch-bearing deltas.
- `compat` and `identity` represent provider compatibility rules and model lineage, reseller references, and effort variants.
- `overlay` merges built-in, user, and project provider configuration at field level, while `oauth_params` loads data-driven login-flow parameters.

## Philosophy

Provider behavior and model metadata change at different rates, so the crate keeps them separate: reviewable TOML owns endpoints, authentication, transport, and compatibility, while generated JSON owns model identities, pricing, and limits. Runtime discovery handles providers whose models cannot be known statically. Broker concerns remain outside the crate through injected transport and credential interfaces, keeping catalog data and registry behavior independent of credential ownership.

Specialized discovery is keyed by `TransportId`, never by provider id. Providers are data rows and transports are code, so a new `providers.toml` entry naming an existing transport gains discovery without a code change, and a genuinely new wire protocol lands in its own `omp-llm-*` crate rather than in this one or in the application.
