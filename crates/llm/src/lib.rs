//! Unified access to Oh My Pi's LLM types, providers, runtimes, and services.

/// Anthropic Messages transport codec.
pub use omp_llm_anthropic as anthropic;
/// Amazon Bedrock Converse Stream transport codec.
pub use omp_llm_bedrock as bedrock;
/// Provider credential broker.
pub use omp_llm_broker as broker;
/// Provider and model catalogs.
pub use omp_llm_catalog as catalog;
/// Cursor Connect transport and execution bridge.
pub use omp_llm_cursor as cursor;
/// Devin Cascade transport codec.
pub use omp_llm_devin as devin;
/// HTTP egress stacks and policy.
pub use omp_llm_egress as egress;
/// Provider error classification and recovery policy.
pub use omp_llm_error as error;
/// Apple Foundation Models runtime.
pub use omp_llm_fm as fm;
/// Provider gateway and foreign-wire facades.
pub use omp_llm_gateway as gateway;
/// Google Generative Language and Vertex AI transport codecs.
pub use omp_llm_google as google;
/// Hardware-accelerated local inference runtimes.
pub use omp_llm_local as local;
/// Ollama native `/api/chat` transport codec.
pub use omp_llm_ollama as ollama;
/// `OpenAI` Chat and Responses transport codecs.
pub use omp_llm_openai as openai;
/// Tower middleware for provider attempts.
pub use omp_llm_tower as tower;
/// Provider wire transports and streaming decoders.
pub use omp_llm_transport as transport;
/// Canonical provider-independent values and facet traits.
pub use omp_llm_types as types;
