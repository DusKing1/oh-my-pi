//! Interactive rendering gallery: markdown, LaTeX, mermaid, and live editing.
//!
//! ```sh
//! cargo run -p omp-tui --example gallery
//! ```
//!
//! Tab/Shift-Tab moves focus, ←/→ switches tabs on the tab bar, ↑/↓ and
//! PageUp/PageDown scroll the active pane, and the Live tab re-renders the
//! preview as you type. Ctrl-C or Ctrl-Q quits.

use std::io;

use omp_tui::{AppEvent, AppOptions, Key, Size, Ui, UiContext};

const MARKDOWN_TAB: &str = r#"# Rendering Gallery

Everything below is one `<md>` node: **bold**, *italic*, ~~strike~~,
`code`, [a link](https://example.com/docs), a bare https://example.com
autolink, and color chips for #C5FFD6, #4A90D9, and `#fff`.

> Quotes wrap with a rail — *and inline styles survive inside them.*

| Feature | State | Notes |
| --- | --- | --- |
| tables | done | box borders, pi column widths |
| math | done | inline and display |
| mermaid | done | fenced blocks |

```rust
fn main() {
    println!("fenced code keeps its fences");
}
```

├── markdown
│   ├── inline.rs
│   └── table.rs
└── latex

---

1. ordered
2. lists
   - with nesting
"#;

const MATH_TAB: &str = r"# Math

Inline: $e^{i\pi} + 1 = 0$, fonts $\mathbb{R}^n \to \mathcal{H}$,
scripts $x_i^2$, and currency stays put: $5 and $10.

$$
x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
$$

$$
f(x) = \begin{cases} x^2 & x > 0 \\ 0 & \text{otherwise} \end{cases}
$$

$$
\sum_{i=0}^{n} x_i \qquad \int_a^b f(x)\,dx \qquad \prod_{k=1}^{n} (1 + x_k)
$$

$$
\underbrace{a + b}_{\text{sum}} \qquad \left( \frac{1}{2} \right)^2
$$

\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}
";

const MERMAID_TAB: &str = r"# Mermaid

```mermaid
flowchart LR
  A[Lex] --> B[Parse] --> C[Layout]
  C --> D[Paint]
  C --> E[Cache]
```

```mermaid
flowchart TD
  start[Request] --> auth{Authorized?}
  auth -->|yes| serve[Serve page]
  auth -->|no| deny[401]
```
";

const LIVE_PREFILL: &str =
	"# Live *markdown* — edit me! Math: $x^2 + y^2 = r^2$, chip: #C5FFD6, **bold**, `code`";

const PANE_IDS: [&str; 3] = ["pane-md", "pane-math", "pane-mermaid"];

/// Fixed pane height for a viewport: tab bar + gaps + footer + scroll
/// chrome ≈ 8 rows.
fn pane_height(viewport: Size) -> u16 {
	viewport.height.saturating_sub(8).max(4)
}

fn build_ui(viewport: Size, context: UiContext) -> Ui {
	// keep panes inside the viewport so switching tabs never strands
	// stale rows in scrollback
	let pane_height = pane_height(viewport);

	Ui::from_root(
		omp_tui::dom! {
			<col gap=1>
				<tabs id=view>
					<tab title="Markdown">
						<scroll id="pane-md" h={pane_height}>
							<md>{MARKDOWN_TAB}</md>
						</scroll>
					</tab>
					<tab title="Math">
						<scroll id="pane-math" h={pane_height}>
							<md>{MATH_TAB}</md>
						</scroll>
					</tab>
					<tab title="Mermaid">
						<scroll id="pane-mermaid" h={pane_height}>
							<md>{MERMAID_TAB}</md>
						</scroll>
					</tab>
					<tab title="Macro">
						<box border=round title="Built with dom!">
							<col gap=1>
								<row gap=1>
									<i:info/>
									<text bold>{"Macro-built pane"}</text>
								</row>
								<gallery-note>
									<text dim>{format!("Interpolated at runtime: {} nested layout levels", 3)}</text>
								</gallery-note>
							</col>
						</box>
					</tab>
					<tab title="Live">
						<col gap=1>
							<editor id=src value={LIVE_PREFILL}/>
							<box border=round title="Preview">
								<md id=preview>{"..."}</md>
							</box>
						</col>
					</tab>
				</tabs>
				<text dim>{"Tab focus · ←/→ switch tabs · ↑/↓ PgUp/PgDn scroll · Ctrl-C quit"}</text>
			</col>
		},
		viewport.width,
		context,
	)
}

/// Mirrors the editor's text into the preview `<md>` node when it changed.
fn sync_preview(ui: &mut Ui, synced: &mut String) {
	let text = ui.values()["src"].as_str().unwrap_or_default().to_owned();
	if text != *synced {
		ui.set_text("preview", text.clone());
		*synced = text;
	}
}

#[tokio::main]
async fn main() -> io::Result<()> {
	let mut app = AppOptions::new()
		.mouse()
		.quit([Key::Ctrl('c'), Key::Ctrl('q')])
		.start(|env| build_ui(env.viewport, env.ctx))
		.await?;
	let mut synced = String::new();
	while let Some(event) = app.next().await? {
		if let AppEvent::Resized(viewport) = event {
			for pane in PANE_IDS {
				app.ui_mut().set_height(pane, pane_height(viewport));
			}
		}
		sync_preview(app.ui_mut(), &mut synced);
	}
	Ok(())
}
