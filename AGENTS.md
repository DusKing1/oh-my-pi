# Workspace Conventions

## Dependencies
All dependencies MUST be declared in `[workspace.dependencies]` in the root
`Cargo.toml`. Member crates reference them with `{ workspace = true }` and
never pin their own versions:

```toml
# root Cargo.toml
[workspace.dependencies]
serde = { version = "1", features = ["derive"] }

# crates/<name>/Cargo.toml
[dependencies]
serde = { workspace = true }
```

Feature additions on top of the workspace entry are fine:
`serde = { workspace = true, features = ["rc"] }`.

`serde_json` is always consumed with the `preserve_order` + `raw_value` features
(preserve order, defer parsing of passthrough payloads); `crates/slopjson`
(broken/partial/streaming JSON) mirrors the same surface.

### Naming & environment
- Environment variables are `OMP_*` prefixed, never `PI_*`. Ported code MUST
  have its upstream (`pi`, `uu`, …) env vars, context objects, and branding
  stripped, not aliased.
- The repo is pre-release: rename completely and move (don't copy) when
  restructuring. Compat shims, old names, and deprecated aliases are
  PROHIBITED; update every callsite in the same change.

### Unicode and terminal-text utilities
Use `xutf` for every workspace-owned Unicode/UTF-8/UTF-16/UTF-32, grapheme,
display-width, normalization, and ANSI/VT operation. `unicode-normalization` is
expressly banned. MUST NOT add or import a separate utility crate for any of
these concerns—including `unicode-*`, `utf8-*`, `unicode-segmentation`,
`unicode-width`, `ansi_*`, or `strip-ansi-escapes`. Use `xutf` directly; remove
redundant direct dependencies rather than wrapping or duplicating them.

## Crates
- Members live under `crates/*` (virtual workspace, resolver 3); directory
  names are unprefixed (`crates/demo`).
- Package names MUST be `omp-` prefixed (`name = "omp-demo"`).
- Every member inherits shared metadata and lints:

```toml
[package]
name = "<name>"
version.workspace = true
edition.workspace = true

[lints]
workspace = true
```

Every member also fills `[package]` metadata (`license`/`authors`/`homepage`/
`repository` from workspace, plus a real `description`) and ships a README
saying what it is and its structural philosophy.

### Taxonomy
- Related crates share a domain prefix after `omp-` (e.g. the inference
  family: `llm` facade re-exporting `llm-types`, `llm-catalog`, `llm-egress`,
  `llm-tower`, `llm-error`, `llm-broker`, `llm-gateway`, `llm-local`, `llm-fm`,
  and one crate per genuinely unique transport: `llm-openai`, `llm-anthropic`,
  `llm-google`, `llm-cursor`, `llm-devin`). `llm-openai` holds ALL of OpenAI's
  wire shapes (regular, responses, responses-lite, websocket, codex).
- Vocabulary is load-bearing: a **transport** is a provider's wire protocol; a
  **dialect** is how the thread is rendered to the LLM. NEVER call a transport
  a dialect.
- Providers are data entries, not code; only a transport with real wire
  differences earns a crate.
- Daemons are subcommands of the app crate, never standalone `*-d` crates.
## Toolchain & Style
- Pinned nightly via `rust-toolchain.toml`; edition 2024.
- Lint policy lives in `[workspace.lints.*]` in the root `Cargo.toml`;
  `#[allow]` requires a `reason`.
- Formatting: `cargo fmt` (hard tabs, 3-column, max width 100 — see
  `rustfmt.toml`). Never hand-format.

## Allocation Discipline (CRITICAL)
Prefer references (`&T`, `&str`, `&[T]`) and borrows over owned types whenever data lifetime permits — agents should not be afraid of using refs and borrows.
Think twice before reaching for `String` or `Vec` — both are growable,
heap-backed general-purpose defaults. `omp-core` ships replacements; each
is MANDATORY *in the situation it targets*, and explicitly NOT a violation
to skip outside it. The test is always the same: does the replacement
remove allocations, copies, or locking on a real code path? If not, the
default type is the right call — don't churn code to swap one for the
other.
- `Vec<T>` alternatives, by growth pattern:
  - Usually small (≲ a dozen items), hot, or short-lived →
    `smallvec::SmallVec` (inline until spill). Not worth it for cold,
    long-lived, or usually-large collections — spilled SmallVec is just a
    worse Vec.
  - Hard upper bound known at compile time → fixed array `[T; N]`
    (`[Option<T>; N]` if slots may be empty).
  - Concurrent append-only log read while written → `omp_core::AppendVec`
    (lock-free appends, stable indices). Single-threaded or fully built
    before reading → plain Vec is fine.
  - Unbounded, built once, moved once (scratch buffers, collect-and-return,
    channel payloads) → plain `Vec` is correct; none of the above apply.
- Strings: default to `omp_core::Str` — the in-repo type
  (`crates/core/src/str.rs`), NOT the smol_str crate. Inline ≤23
  bytes; heap side is `Bytes`-backed: O(1) clone, zero-copy
  slice/split/trim. Build with `StrMut` + `freeze()` or
  `fmts!`; convert via `IntoStr` (`.to_str()`). It pays off
  for stored, cloned, or sliced strings (ids, names, tokens, messages).
  `String` remains fine as a transient build buffer that is consumed
  immediately, and for APIs that require it (`fmt::Write`, FFI, serde
  sinks). Large/edited text → a rope (`ropey`).
- Byte buffers: `omp_core::CowBytes` when a buffer is shared, sliced, or
  cloned — replaces `Cow<'_, [u8]>` (borrowed or `Bytes`-owned, O(1)
  clone, zero-copy slicing). A buffer produced once and moved to a single
  consumer is fine as `Vec<u8>`; converting buys nothing.
- Maps/sets keyed by enums or small dense ints → `omp_core::SparseMap` /
  `SparseSet` (bitmap presence + packed values). `HashMap` stays correct
  for sparse/unbounded keys, strings, and anything without a small dense
  index.
- Binary↔text: `omp_core::encoding` (`hex`, `base64`, `base32`) with
  stack-allocated `ArrayStr<N>` outputs; never add an external
  hex/base64 crate. This one has no exception: external encoding crates
  are banned outright.

## Async, Iterator & Codegen Discipline (CRITICAL)
House rules, proven in a sibling codebase (tetra). Not suggestions.

- **Nightly features are the point of the pinned toolchain.** A crate MUST
  gate exactly the features it uses at the top of `lib.rs` — and again in
  every integration test or example that needs them (test targets are
  separate crates). The canonical set for trait plumbing:
  `impl_trait_in_assoc_type` + `type_alias_impl_trait` (impls infer their
  future/iterator types in associated-type position),
  `min_specialization` (`default fn` fallbacks), and
  `const_eval_select` / `core_intrinsics` for codegen hints
  (`core_intrinsics` additionally requires
  `#![allow(internal_features, reason = "…")]`). NEVER redesign an API
  around a missing stable feature when a nightly gate gives the zero-cost
  shape directly.
- **Async traits are unboxed by default.** A trait with async behavior MUST
  NOT allocate per call. Two sanctioned shapes:
  1. Callers never name the future → plain RPITIT:
     `fn run(&mut self) -> impl Future<Output = T> + Send + '_;`
  2. The future must be nameable (stored, composed, or required by a
     downstream trait like `tower::Service`) → a (generic) associated type,
     inferred on the impl side:
     ```rust
     pub trait Deliverable<A: ?Sized>: Send + 'static {
        type Result: Send + 'static;
        type Future<'c>: Future<Output = Self::Result> + Send + 'c;
        fn deliver<'c>(self, target: &'c mut A) -> Self::Future<'c>;
     }
     // impl side — the concrete type is inferred from the async block:
     type Future<'c> = impl Future<Output = Self::Result> + Send + 'c;
     fn deliver<'c>(self, target: &'c mut A) -> Self::Future<'c> {
        async move { /* … */ }
     }
     ```
  `tower::Service` / hyper service impls follow the same rule:
  `type Future = impl Future<Output = …>;` — never `BoxFuture`. An impl
  that answers synchronously returns `future::Ready<T>` / `future::ready(v)`,
  not an async block and not a box.
- **`#[async_trait]`, `BoxFuture`, and per-call `Box::pin` are quarantined.**
  Permitted ONLY at cold `dyn`-dispatch boundaries whose latency is dominated
  by real I/O (DNS lookup, remote storage, connection establishment) — one
  allocation per network round trip is noise. On anything invoked per
  message, per frame, per token, or per byte they are PROHIBITED. Where a
  hot-ish boundary genuinely needs `dyn`, box ONCE at construction behind an
  alias (`type BoxFut<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;`),
  never per poll or per request.
- **Erase types with enums before reaching for `dyn`.** When one slot must
  hold several concrete types, define an enum with a variant per common
  concrete type and a single `Boxed(Pin<Box<dyn Trait>>)` fallback; the
  constructor fast-paths known types and boxes only in the `else` arm. The
  common cases then dispatch by `match`, allocation-free.
- **Inline hints.** Small cross-crate functions on hot paths get `#[inline]`;
  `#[inline(always)]` is available and lint-sanctioned when measured.
- **Specialize fallbacks instead of dispatching at runtime.** A blanket impl
  (e.g. `Display`-based conversion) is written as `default fn`; concrete fast
  paths (`&str`, integers, …) override it via `min_specialization`. The
  generic path stays correct, the common path stays allocation- and
  format-machinery-free — with zero runtime branching.
- **Iterators are lazy, borrowed, and unboxed.** Accessors return
  `-> impl Iterator<Item = …> + '_`, declaring every capability the chain
  actually has (`+ Clone`, `+ DoubleEndedIterator`, `+ FusedIterator`,
  `+ ExactSizeIterator`) so callers keep it. Yield borrows (`&T`) or O(1)-clone
  items (`Str`, `Bytes` slices), never freshly allocated ones. NEVER
  `.collect()` into an intermediate `Vec` just to iterate again — chain
  adaptors end to end and collect only at the final owner, if at all. When the
  iterator type must be nameable (an `IntoIterator::IntoIter`, a stored
  field), alias it with TAIT instead of writing the adaptor tower out or
  boxing:
  ```rust
  pub type Iter<'s, T: 's> = impl DoubleEndedIterator<Item = &'s T> + FusedIterator + 's;
  impl<'a, T> IntoIterator for &'a Container<T> {
     type Item = &'a T;
     type IntoIter = Iter<'a, T>;
     fn into_iter(self) -> Self::IntoIter { /* plain adaptor chain */ }
  }
  ```
  Containers implement `IntoIterator` for `&T`, `&mut T`, and `T` with
  concrete (or TAIT-aliased) iterator types. `Box<dyn Iterator>` falls under
  the same quarantine as `BoxFuture`: cold, I/O-dominated boundaries only.
- **Service stacks (`tower`-style) allocate at construction, not per call.**
  Layers compose ONCE when the stack is built; a request path never
  assembles middleware. The `poll_ready` → `call` contract MUST be honored
  on the SAME service instance — readiness observed on one clone says
  nothing about the clone you then call, and skipping it hides
  backpressure. Driving that contract over a borrowed service means a
  hand-rolled pin-projected state-machine future, not a box:
  `NotReady { svc: &'c mut S, msg } → Pending(#[pin] S::Future) → Done`,
  where `poll` runs `ready!(svc.poll_ready(cx))?` then `svc.call(msg)` on
  that same `&mut S`. When an adapter purely delegates, forward the inner
  future type verbatim (`type Future = <S as Service<Req>>::Future;`) — no
  wrapper future at all. Narrow, documented exception: a type-erasure
  handle whose real readiness gate lives INSIDE the erased call may
  dispatch as `self.clone().oneshot(req)` in an inferred future and report
  always-`Ready` from its own `poll_ready` — this demands a cheap-clone
  handle (`Arc`-backed state) and a doc comment on `poll_ready` stating
  where readiness is actually enforced. Never generalize that shortcut.
  Second exception, a measured compromise for `async_stream`-based
  middleware — stream-transforming layers (retry/rotate/repair) returning a
  wrapped response stream MAY heap-pin one generator per call behind a TAIT
  alias (`Box::pin(async_stream::stream! { … })` hidden inside
  `impl Stream + Send + Unpin`). Rationale: composing such generators fully
  inline embeds every inner layer's state (and its poll frames) in the
  parent's, and a 7-layer stack was MEASURED to overflow the thread stack
  at construction in debug builds; the per-layer box keeps each layer's
  state behind one pointer. This is a property of the current generator
  implementation, not a law — a hand-written pin-projected state machine
  avoids the box entirely and is the preferred replacement when a layer is
  hot enough to justify writing one. Never cite this exception outside
  stream-returning middleware; dyn erasure still happens at most once, at
  the stack's outer boundary. Thin wrappers (permit holders, taps) and
  short-circuit responses (`Either`, one-shot `stream::once`) stay fully
  unboxed via pin-projection.
- **Scratch buffers are owned once and recycled — in one of two modes,
  never conflated.** A hot encode/frame path owns one pre-sized `BytesMut`
  (`with_capacity` at a measured watermark):
  1. *True scratch reuse* — the contents are consumed in place (written to
     I/O, copied into a frame) before the next round: `clear()` between
     rounds. Capacity genuinely survives; steady state is allocation-free.
  2. *Zero-copy ownership transfer* — the result must escape:
     `split().freeze()` hands the filled prefix (and its share of the
     backing allocation) to the receiver as `Bytes`; only the unfilled tail
     remains in the `BytesMut`, so later rounds `reserve` (amortized
     reallocation) as needed. That is the price of not copying — accept it
     knowingly; don't claim the capacity survived.
  Derived views (headers, sub-ranges) come from `slice(..)` on the frozen
  `Bytes`, never a copy. This composes with the Allocation Discipline list
  above: `CowBytes`/`Str` for storage, `BytesMut` for assembly.
- **Locks: `parking_lot::{Mutex, RwLock}`, never `std::sync`.** Reach for
  `tokio::sync::Mutex` ONLY when the guard is genuinely held across an
  `.await`; if it isn't, a `parking_lot` lock is smaller and faster.
- **Channels: `flume`, never `tokio::sync::mpsc` or `std::sync::mpsc`.**
  Actor/messagebox loops take a single flume mailbox; priority signals
  (resize, shutdown) ride `tokio::watch` + `select!`, not a second queue.

## TUI Rendering Doctrine (crates/tui, CRITICAL)
The port exists because pi's `string[]` + ANSI + `render()` contract was a
per-frame heap-grooming machine. These rules are the fix — they are not
negotiable:

- **Text is parsed ONCE, at the boundary.** ANSI/VT escapes are decomposed
  (via `xutf`) where external text enters (process output, pastes, files).
  Every component downstream assumes its text contains ZERO escapes and never
  stores any. Sinks receive `render(style, text)`; ANSI is re-emitted exactly
  once, at final materialization into the stdout buffer.
- **Caches own memory.** A cache holds one pooled text buffer plus
  `(Style, Range)` spans; re-presenting is re-slicing, not re-parsing.
  Per-frame allocation of line buffers (`Vec<Line>` built fresh each paint) is
  a bug, not a style choice.
- **Markup (TML) degrades like HTML.** An unknown tag becomes a
  `CustomElement` — forwarded to a registered renderer if one exists, else its
  children render and it layers like a `div`. A bad tag MUST NOT fail the
  whole document into raw-text fallback.
- **Props inherit like CSS.** `<col fg=blue>hi</col>` colors its text without
  an explicit `<text>`. Any prop applies to any element where it can
  meaningfully apply. Well-known props are stored typed and non-allocating;
  arbitrary KV sits beside them. Color fields accept `#xxx`, `#xxxxxx`,
  `rgb(a)`, `hsl(a)`, `lab`/`oklch`, full HTML names, and gradients as plain
  `bg`/`fg` values (with angle) — gradients are values, not special elements.
- **Context threads everywhere.** `UiContext` (charset: ascii | unicode |
  nerdfont, plus theme) reaches every component; hardcoded colors and
  hand-emitted glyphs are banned. Icons come from `icons.tsv` (generic name +
  optional specific alias, mapped across charsets, degrading inline). Border
  defaults are themed and dim, not `#fff`.
- **`dom!`/`layout!` is the canonical construction path** (typed props, loops,
  `if`/`match`, `IntoComponent` for `&str`/`String`/`Str`/`()`/Vec).
  Building markup by `write!`/`format!` into a `String` and parsing it back is
  the discouraged path.
- **Effects are props, not one-offs.** Shimmer, hover gradient + eased lift,
  streaming reveal (`<text reveal>`), truncate-from-start, tree/checklist,
  clickable scrollbars, non-committed sidebars: if an example needs it, it
  lands as a reusable prop or component in core FIRST. Never ship a visual
  feature as example-local code.
- **Examples stay near-zero boilerplate** (the `App` host, a `start`, done).
  An example touching kitty image ids, raw escape dispatch, terminal probing,
  focus routing, or clipboard internals means the ENGINE is missing the
  primitive — fix the engine, not the example. The editor itself is built
  from components so users can recompose it.
- **Alt buffer only where required** (overlays, the welcome scene). Chat and
  transcripts stay inline and mouse-selectable; quitting restores the
  terminal cleanly (no stray mouse-tracking spam).
- **Input is one mailbox.** Decoded `TerminalEvent`s (real input, debug
  injections, resize) flow through a single async flume mailbox; resize wins
  via watch + `select!`. No polling `read()` loops, no per-example key
  dispatch tables. Keyboard input instantly clears mouse-hover state; there is
  only ever one visible cursor/focus.

## Porting Subsystems
omp is a Rust rewrite of pi. When porting any subsystem:

1. **Read pi's implementation in extreme detail first** — including
   `crates/natives`, compat shims, support detection, and its tests. Every
   missed behavior (editor keys, paste/drag-drop, resize-settling, truncation)
   resurfaces as a user-reported bug within hours.
2. **Copy pi's tests; drop TS-shaped compensations.** Throttles, GC
   workarounds, and UTF-16 defenses exist because "ts is slow, rust isn't" —
   do not port them. Port behavior, not shape: reimplement where the shape is
   wrong (mermaid, slopjson, the brush parser), never wrap what should be
   native.
3. **Generalize while porting.** The port lands as themed, charset-aware,
   prop-driven engine primitives — not as a feature checkmark that only works
   in one example.
4. **Match pi exactly where it's good** (editor UX, telemetry, statusline
   semantics, alt-buffer resize handling); **exceed it where it's weak**
   (renderer contract, error taxonomy, providers-as-data).
5. **Close the gaps pi left** (missing builtins, slash-arg completion, …)
   while you're in the area.

## Working Style
- **Orchestrate in parallel.** One agent per crate/util/provider/category;
  `sonic` agents for mechanical moves and renames (use `sd`/bash for bulk
  renames, never hand edits); scouts only when the affected files are
  genuinely unknown. Sequential one-agent-at-a-time is a failure mode.
- **Finish the whole ask.** No scaffolds, no "the rest is trivial", no
  half-ports handed back. "Done" means compiles, wired, exercised.
- **Verify by running.** TUI changes are driven on a real PTY via the `tui`
  tool (below) before claiming done; every input path, resize, and
  quit-cleanup gets exercised.
- **Never revert or `git checkout` user edits.** The user edits and renames
  in flight while agents work — adapt to the tree as it is.

## Embedded Python (omp-py)
- `omp-py` (`crates/py`) is a library that statically links CPython 3.14t
  (free-threaded) and boots it in-process: `Engine::builder().init()`, then
  `engine.attach(|py| ...)`. Native modules register with
  `pyo3::append_to_inittab!` before `init`. The `omp-demo` bin ships from the
  same crate. Building requires `crates/py/scripts/fetch-python.sh` once (populates
  gitignored `vendor/python` with the python-build-standalone archive and
  derived build inputs).
- Pure-Python packages frozen into the binary (e.g. cloudpickle) are pinned
  in `crates/py/requirements.txt`; `crates/py/scripts/fetch-python.sh` resolves them
  with `uv` into gitignored `vendor/python/bundled/` (skipped while the
  stamp matches the manifest) and regenerates the tracked
  `crates/py/THIRD-PARTY-NOTICES.txt` (also available as
  `omp_py::THIRD_PARTY_LICENSES`) — rerun it after editing the manifest and
  commit the notices. omp-py's build script only validates the stamp and
  packs; native wheels are rejected at fetch time — those go into
  site-packages.
- pyo3 is configured via `PYO3_CONFIG_FILE` in `.cargo/config.toml`. The
  pgo+lto pbs variant is LLVM-22 LTO bitcode: it links through Homebrew
  LLD 22 (`brew install lld`, routed via `crates/py/scripts/ld64.lld`) — Xcode's ld64
  is too old for it and rustc's rust-lld (LLVM 23) mis-resolves symbols.
  Two things never propagate to consumers and are their explicit contract,
  enforced loudly by omp-py's build script:
  1. `PYO3_CONFIG_FILE` must point at `vendor/python/pyo3-config.txt` before
     cargo runs (this repo's `.cargo/config.toml` covers workspace members;
     external crates set it in their own `[env]` or environment) — otherwise
     pyo3 silently links a host Python.
  2. Final-link flags: consumer bin crates replicate `--ld-path=<shim>` and
     `-Wl,-export_dynamic` in their own build script — `crates/py-smoke`
     (`omp-py-smoke`) is the working example and doubles as the smoke test.
- The stdlib is embedded as marshalled bytecode and served from memory; the
  only real search path is `$OMP_PY_SITE` (default
  `~/.local/share/omp-py/site-packages`). End users install wheels with any
  free-threaded 3.14 interpreter — no repo checkout needed:
  ```sh
  uv python install 3.14t
  uv pip install --python "$(uv python find 3.14t)" \
      --target "${OMP_PY_SITE:-$HOME/.local/share/omp-py/site-packages}" numpy
  ```

## Debugging the TUI (`tui` tool, `OMP_TUI_DEBUG`, `OMP_TTY`)
Prefer the project-local `tui` agent tool (`.omp/tools/tui.ts`): it runs an
example or bin on a Bun-native PTY — a real controlling terminal, so
SIGWINCH resizes and immediate-mode hosts behave as in production — and
exposes screenshots (`text`), component trees (`tree`), widget values,
key/mouse/paste injection, resizes, and raw byte-stream stats as one
session-based tool. Its structured ops ride the first hook below; the
second exists for external harnesses without a PTY of their own:

- `OMP_TUI_DEBUG=<unix-socket-path>` — `Terminal::enter` starts a server
  thread on the socket with line-delimited JSON ops (`text`, `tree`,
  `values`, `keys`, `event`, `mouse`, `resize`, `quit`, ...); see "Debug a
  running app" in `crates/tui/README.md`. The wire speaks `TerminalEvent`:
  injected input rides the same mailbox as decoded terminal bytes,
  `text`/`info` answer from the last paint on every host, and
  `frame`/`tree`/`values` are mailbox queries only `App` hosts answer (the
  server times them out elsewhere); `quit` injects `C-c`.
- `OMP_TTY=<pty-slave-path>` reroutes ALL terminal I/O — input, rendered
  frames, capability probes, terminal identity — to that device; hold the
  master side to script the UI and capture the exact byte stream a terminal
  would see. stdout stays untouched.

```python
import fcntl, os, pty, struct, subprocess, termios
master, slave = pty.openpty()
fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 100, 0, 0))
proc = subprocess.Popen(
    ["target/debug/examples/gallery"],
    env=dict(os.environ, OMP_TTY=os.ttyname(slave), TERM="xterm-256color"),
)
os.read(master, 65536)          # frames + control sequences
os.write(master, b"\x1b[C")     # keys (write escape sequences)
os.write(master, b"\x03")       # Ctrl-C quits the examples
```

Caveats: set the winsize with `TIOCSWINSZ` before spawning (`SIGWINCH` only
reaches the controlling terminal, so live resizes don't propagate); the
capability probe waits for replies — answer DA1 (`\x1b[?62c`) or let it time
out. Feed the master stream to a VT emulator (e.g. `pyte`) for screen
assertions.
