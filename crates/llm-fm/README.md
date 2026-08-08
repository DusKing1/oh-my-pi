# omp-llm-fm

Async Rust bindings for Apple's on-device Foundation Models runtime. The crate exposes availability checks, complete and streaming generation, request options, cancellation, and stable error categories. It builds on every supported workspace platform, while generation is available only on eligible Apple Silicon Macs with Apple Intelligence enabled.

## Structure

- `lib.rs` defines the public `AppleFm` API, request and response types, streaming events, option validation, cancellation, and the generation deadline.
- `abi` contains the macOS runtime interface used to reach the system framework.
- `macos` implements availability checks and generation on macOS.
- `unsupported` provides the non-macOS platform implementation and reports that the model is unavailable.

## Philosophy

Keep the public API asynchronous and platform-neutral while isolating native runtime details behind platform modules. Blocking framework work runs outside the async executor, streams are cancelled when dropped, and failures use stable machine-readable categories while retaining native diagnostics. Unsupported systems remain buildable and report unavailability explicitly rather than failing at compile time.
