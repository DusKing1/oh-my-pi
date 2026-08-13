#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

mod cells;
mod fonts;
mod gpu;
mod host;
mod input;
#[cfg(target_os = "macos")]
mod macos;
mod mux;
mod scene;
mod theme;

pub use cells::{CellMetrics, Compositor, Instances, Selection, View};
pub use fonts::{FontError, Fonts};
pub use gpu::{Gpu, GpuError, Painter, WindowGpu};
pub use host::{HostConfig, run};
pub use scene::{Effect, Scene, SceneFrame};
pub use theme::GuiTheme;
