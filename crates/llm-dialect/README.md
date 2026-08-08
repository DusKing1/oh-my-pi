# omp-llm-dialect

`omp-llm-dialect` owns model-prompt dialects independently of provider transports. It renders borrowed tool definitions and canonical conversation history into model-facing prompts, then incrementally scans model-authored text into canonical visible text, thinking, and tool-call events.

The crate supports GLM, Hermes, Kimi, XML, Anthropic, DeepSeek, Harmony, Qwen 3, Gemini, Gemma, and MiniMax. Catalog model identity selects the default dialect; unknown families fall back to XML, while explicit native selection leaves provider-native channels intact.

Hot streaming paths use concrete scanner enum dispatch, inline event batches, and `Bytes` slices. Tool definitions and renderer inputs remain borrowed, and scanners retain only bounded delimiter and argument scratch state rather than cloning accumulated output per delta.
