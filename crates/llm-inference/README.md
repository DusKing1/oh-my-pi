# omp-llm-inference

`omp-llm-inference` is OMP's typed public contract for model inference. It gives every supported operation—from chat and embeddings through media, realtime, discovery, usage, authentication, and allowlisted native wire access—a concrete request and output type over one closed Tower service envelope.

The crate keeps the public edge statically typed while the provider center is erased once at service construction. Calls are cheap to clone because operation payloads are shared, streams and sessions retain explicit ownership, errors are structured and secret-free, and receipts account for every attempt, recovery, usage dimension, and integer monetary unit. Provider identity and capability vocabulary come from `omp-llm-catalog`; this crate does not infer policy from provider or model strings.
