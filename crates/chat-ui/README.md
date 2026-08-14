# omp-chat-ui

`omp-chat-ui` is the host-agnostic designed chat scene shared by omp frontends. It owns immediate-mode presentation state: append-only transcript rows, a mutable live band, composer attachments and completion, status chrome, damage ranges, resize previews, and the matching model/list/prompt/command overlays. It does not own an agent, persistence, credentials, catalog, terminal, or synthetic demo data.

A host creates `Chat` with its `UiContext`, forwards input to the scene, sends resulting `Intent` values to its backend, and applies `BackendEvent` values through `Chat::apply_backend_event` (or the corresponding typed mutation methods). The terminal host uses `RenderedFrame::stable_rows` and `RenderedFrame::damage` to commit native scrollback without moving the live band.

The production terminal host is `omp-app`. The `omp-tui` terminal example and `omp-gui` GPU example supply mock data outside this crate to exercise the same scene. Keeping the scene immediate-mode and backend-neutral lets those hosts share visual behavior without coupling presentation to a retained application runtime or an agent implementation.
