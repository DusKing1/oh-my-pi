# `omp-gui`

`omp-gui` hosts `omp-tui` applications in a native, GPU-accelerated window. The
same retained `Frame` cell grids a terminal `Renderer` would emit as ANSI are
composited and rasterized directly by wgpu — no escape bytes, no intermediate
terminal emulator — inside a semi-transparent, decoration-less, vibrancy-backed
shell.

## Structure

- `fonts` discovers system faces (fontdb), shapes cell clusters (rustybuzz),
  rasterizes glyphs (swash), and packs them into two atlases: an R8 coverage
  atlas for outlined text and an RGBA atlas for color bitmap emoji.
- `gpu` owns the wgpu device and the two-pipeline painter: instanced SDF
  rounded rects (fills, shadows, carets, scrollbars) and instanced atlas
  glyphs, alpha-premultiplied end to end so the window can go translucent.
- `cells` is the compositor: it walks the document `Frame`'s visible window
  plus declarative `Layer` bands and emits rect/glyph instances, resolving
  `Style` attributes (reverse, dim, underline, strikethrough, wide graphemes).
- `scene` is the host contract: a `Scene` produces `SceneFrame`s and routes
  input; `mux` is the pure split-tree layout; `host` is the winit shell that
  drives windows → tabs → split panes (one scene per pane) — window
  lifecycle, cell geometry, animation ticks, smooth transcript scrolling,
  clipboard pastes, mux hotkeys, and window chrome (tab strip, pane
  dividers, drag zones, vibrancy, rounded corners).

## Philosophy

Text is parsed once, at the component boundary, exactly like the terminal
host: the GUI consumes the retained cell grid, never escape sequences. The
window is chrome the application never thinks about — it renders the same
document and layers the terminal would, with pixel freedom the terminal does
not have: smooth scrolling, soft shadows, translucency, and real emoji.

## Run the chat demo

```sh
cargo run -p omp-gui --example chat            # native window
cargo run -p omp-gui --example chat -- --shot welcome /tmp/welcome.png
```

The example reuses the terminal chat's scene modules verbatim (`#[path]`
includes, the same pattern the gallery example uses); only the host changes.
