//! The chat demo hosted in a GPU window: the terminal example's scene
//! modules verbatim (shared via `#[path]`, the gallery's pattern), driven
//! by `omp-gui` instead of the terminal host.
//!
//! Run: `cargo run -p omp-gui --example chat`
//! Shots: `cargo run -p omp-gui --example chat -- --shot welcome|chat|picker
//! OUT.png`

#[path = "../../tui/examples/chat/commands.rs"]
mod commands;
#[allow(
	dead_code,
	reason = "stable_rows/damage serve the terminal host's scrollback commits; the GUI scrolls the \
	          retained document"
)]
#[path = "../../tui/examples/chat/demo.rs"]
mod demo;
#[path = "../../tui/examples/chat/picker.rs"]
mod picker;
#[path = "../../tui/examples/chat/sidebar.rs"]
mod sidebar;
#[path = "../../tui/examples/chat/welcome.rs"]
mod welcome;

use std::time::{Duration, Instant};

use omp_gui::{Effect, HostConfig, Scene, SceneFrame};
use omp_tui::{
	Frame, Graphics, Key, Layer, Mouse, MouseReport, Size, UiContext, paste::ClipboardRead,
};
use smallvec::SmallVec;

use crate::{
	commands::{CommandPalette, PaletteAction, PaletteEvent},
	demo::{Demo, DemoKey},
	picker::{MODELS, ModelPicker, PickerEvent},
	sidebar::Sidebar,
	welcome::Welcome,
};

const FRAME_INTERVAL: Duration = Duration::from_millis(33);

fn main() {
	let mut args = std::env::args().skip(1);
	if args.next().as_deref() == Some("--shot") {
		let scene = args.next().unwrap_or_else(|| "chat".to_string());
		let out = args
			.next()
			.unwrap_or_else(|| "/tmp/omp-gui.png".to_string());
		shot(&scene, &out);
		return;
	}
	omp_gui::run(
		HostConfig { title: "omp — chat".to_string(), ..HostConfig::default() },
		ChatScene::new,
	);
}

/// The chat application as one host-agnostic scene: welcome card, then the
/// chat transcript with its rail and overlays — the terminal host's routing,
/// minus terminal plumbing (alt-screen staging, scrollback commits).
struct ChatScene {
	ctx:      UiContext,
	started:  Instant,
	phase:    Phase,
	viewport: Size,
}

enum Phase {
	Welcome(Welcome),
	Chat(ChatState),
}

struct ChatState {
	demo:          Demo,
	sidebar:       Sidebar,
	overlay:       Option<Overlay>,
	current_model: usize,
	/// Last painted document height, for document-space mouse translation.
	doc_rows:      u16,
	/// Mid-gesture resize: paint cheap previews until the settle lands.
	preview_size:  Option<Size>,
	preview:       Frame,
}

impl ChatState {
	fn new(ctx: &UiContext, viewport: Size) -> Self {
		let sidebar = Sidebar::new(MODELS[0].name, ctx);
		let mut demo = Demo::new(ctx);
		demo.set_right_inset(sidebar.reserved(viewport));
		Self {
			demo,
			sidebar,
			overlay: None,
			current_model: 0,
			doc_rows: 0,
			preview_size: None,
			preview: Frame::new(Size::new(0, 0)),
		}
	}
}

impl ChatScene {
	fn new(ctx: &UiContext) -> Self {
		Self {
			ctx:      ctx.clone(),
			started:  Instant::now(),
			phase:    Phase::Welcome(Welcome::new(ctx.charset)),
			viewport: Size::new(0, 0),
		}
	}
}

/// The modal overlay holding pointer and keyboard ownership.
enum Overlay {
	Picker(ModelPicker),
	Palette(CommandPalette),
}

/// One routed overlay outcome, unified across overlay kinds.
enum OverlayEvent {
	/// Input handled; the overlay stays open.
	Consumed,
	/// Dismissed without effect.
	Close,
	/// The picker chose a model.
	Pick(usize),
	/// The palette activated an entry.
	Run(PaletteAction),
}

impl From<PickerEvent> for OverlayEvent {
	fn from(event: PickerEvent) -> Self {
		match event {
			PickerEvent::Consumed => Self::Consumed,
			PickerEvent::Close => Self::Close,
			PickerEvent::Pick(index) => Self::Pick(index),
		}
	}
}

impl From<PaletteEvent> for OverlayEvent {
	fn from(event: PaletteEvent) -> Self {
		match event {
			PaletteEvent::Consumed => Self::Consumed,
			PaletteEvent::Close => Self::Close,
			PaletteEvent::Run(action) => Self::Run(action),
		}
	}
}

impl Overlay {
	fn handle_key(&mut self, key: Key) -> OverlayEvent {
		match self {
			Self::Picker(picker) => picker.handle_key(key).into(),
			Self::Palette(palette) => palette.handle_key(key).into(),
		}
	}

	fn handle_paste(&mut self, text: &str) -> OverlayEvent {
		match self {
			Self::Picker(picker) => picker.handle_paste(text).into(),
			Self::Palette(palette) => palette.handle_paste(text).into(),
		}
	}

	fn handle_mouse(&mut self, col: u16, row: u16, kind: Mouse, viewport: Size) -> OverlayEvent {
		match self {
			Self::Picker(picker) => picker.handle_mouse(col, row, kind, viewport).into(),
			Self::Palette(palette) => palette.handle_mouse(col, row, kind, viewport).into(),
		}
	}

	fn layer(&mut self, viewport: Size) -> Layer<'_> {
		match self {
			Self::Picker(picker) => picker.layer(viewport),
			Self::Palette(palette) => palette.layer(viewport),
		}
	}
}

/// Applies one routed [`OverlayEvent`]; returns the host effect.
fn apply_overlay_event(
	state: &mut ChatState,
	event: OverlayEvent,
	ctx: &UiContext,
	viewport: Size,
) -> Effect {
	match event {
		OverlayEvent::Consumed => Effect::Consumed,
		OverlayEvent::Close => {
			state.overlay = None;
			Effect::Consumed
		},
		OverlayEvent::Pick(index) => {
			state.current_model = index;
			state.demo.set_model(MODELS[index].name);
			state.sidebar.set_model(MODELS[index].name);
			state.overlay = None;
			Effect::Consumed
		},
		OverlayEvent::Run(action) => match action {
			PaletteAction::SwitchModel => {
				state.overlay = Some(Overlay::Picker(ModelPicker::open(state.current_model, ctx)));
				Effect::Consumed
			},
			PaletteAction::ToggleSidebar => {
				state.sidebar.toggle();
				state.demo.set_right_inset(state.sidebar.reserved(viewport));
				state.overlay = None;
				Effect::Consumed
			},
			PaletteAction::Quit => Effect::Quit,
			PaletteAction::Insert(text) => {
				state.demo.handle_paste(&text);
				state.overlay = None;
				Effect::Consumed
			},
		},
	}
}

impl Scene for ChatScene {
	fn resize(&mut self, viewport: Size, settled: bool) {
		self.viewport = viewport;
		if let Phase::Chat(state) = &mut self.phase {
			state.demo.set_right_inset(state.sidebar.reserved(viewport));
			state.preview_size = if settled { None } else { Some(viewport) };
		}
	}

	fn render(&mut self) -> SceneFrame<'_> {
		let viewport = self.viewport;
		let elapsed = self.started.elapsed();
		match &mut self.phase {
			Phase::Welcome(welcome) => SceneFrame {
				frame: welcome.render(viewport, elapsed),
				viewport,
				// The card is fully interactive; no host text selection.
				editor_rows: viewport.height,
				layers: SmallVec::new(),
			},
			Phase::Chat(state) => {
				let mut layers = SmallVec::new();
				if let Some(layer) = state.sidebar.layer(viewport, elapsed) {
					layers.push(layer);
				}
				if let Some(overlay) = state.overlay.as_mut() {
					layers.push(overlay.layer(viewport));
				}
				let editor_rows = state.demo.composer_rows();
				if state.preview_size.is_some() {
					state.preview = state.demo.render_resize_preview(viewport);
					SceneFrame { frame: &state.preview, viewport, editor_rows, layers }
				} else {
					let rendered = state.demo.render(viewport);
					state.doc_rows = rendered.frame.size().height;
					SceneFrame { frame: rendered.frame, viewport, editor_rows, layers }
				}
			},
		}
	}

	fn key(&mut self, key: Key) -> Effect {
		if matches!(self.phase, Phase::Welcome(_)) {
			return match key {
				Key::Enter => {
					self.phase = Phase::Chat(ChatState::new(&self.ctx, self.viewport));
					Effect::Consumed
				},
				Key::Esc | Key::Ctrl('c') => Effect::Quit,
				_ => Effect::Ignored,
			};
		}
		let Phase::Chat(state) = &mut self.phase else {
			return Effect::Ignored;
		};
		if state.overlay.is_some() {
			if key == Key::Ctrl('c') {
				return Effect::Quit;
			}
			let event = state
				.overlay
				.as_mut()
				.expect("overlay checked above")
				.handle_key(key);
			return apply_overlay_event(state, event, &self.ctx, self.viewport);
		}
		if key == Key::Ctrl('b') {
			state.sidebar.toggle();
			state
				.demo
				.set_right_inset(state.sidebar.reserved(self.viewport));
			return Effect::Consumed;
		}
		if key == Key::Ctrl('k') {
			state.overlay = Some(Overlay::Palette(CommandPalette::open(&self.ctx)));
			return Effect::Consumed;
		}
		if state.sidebar.focused() {
			if key == Key::Ctrl('c') {
				return Effect::Quit;
			}
			state.sidebar.handle_key(key);
			return Effect::Consumed;
		}
		if key == Key::Ctrl('p') || key == Key::Alt('p') {
			state.overlay = Some(Overlay::Picker(ModelPicker::open(state.current_model, &self.ctx)));
			return Effect::Consumed;
		}
		if let Some(scope) = ClipboardRead::for_key(key) {
			return Effect::Clipboard(scope);
		}
		let effect = match state.demo.handle_key(key) {
			DemoKey::Consumed => Effect::Consumed,
			DemoKey::Ignored => Effect::Ignored,
			DemoKey::Quit => Effect::Quit,
		};
		if state.demo.take_switch_request() {
			state.overlay = Some(Overlay::Picker(ModelPicker::open(state.current_model, &self.ctx)));
		}
		if let Some(text) = state.demo.take_copied() {
			// The host writes the clipboard detached off the event loop.
			return Effect::SetClipboard(text);
		}
		effect
	}

	fn mouse(&mut self, report: MouseReport) -> Effect {
		if let Phase::Welcome(welcome) = &mut self.phase {
			if matches!(report.kind, Mouse::Move | Mouse::Drag) {
				welcome.point_at(report.col, report.row);
			}
			return Effect::Consumed;
		}
		let Phase::Chat(state) = &mut self.phase else {
			return Effect::Consumed;
		};
		if let Some(overlay) = state.overlay.as_mut() {
			let event = overlay.handle_mouse(report.col, report.row, report.kind, self.viewport);
			return apply_overlay_event(state, event, &self.ctx, self.viewport);
		}
		if !state
			.sidebar
			.handle_mouse(report.col, report.row, report.kind, self.viewport)
		{
			// The editor UI expects document-space rows: the live window is
			// the document tail, offset by the scrollback height.
			let window_top = state.doc_rows.saturating_sub(self.viewport.height);
			let doc = MouseReport { row: report.row.saturating_add(window_top), ..report };
			state.demo.handle_mouse(&doc);
		}
		Effect::Consumed
	}

	fn paste(&mut self, text: &str, raw: bool) -> Effect {
		let Phase::Chat(state) = &mut self.phase else {
			return Effect::Consumed;
		};
		if let Some(overlay) = state.overlay.as_mut() {
			let event = overlay.handle_paste(text);
			return apply_overlay_event(state, event, &self.ctx, self.viewport);
		}
		if state.sidebar.focused() {
			return Effect::Consumed;
		}
		if raw {
			state.demo.handle_paste_raw(text);
		} else {
			state.demo.handle_paste(text);
		}
		Effect::Consumed
	}

	fn tick(&self) -> Duration {
		FRAME_INTERVAL
	}
}

/// Renders one scripted scene to a PNG without a window, for pixel-level
/// verification (`--shot welcome|chat|picker OUT.png`).
fn shot(name: &str, out: &str) {
	use omp_gui::{Compositor, Fonts, Gpu, GuiTheme, Painter, View};
	use omp_tui::Charset;

	let gpu = Gpu::new(None).expect("gpu");
	let format = wgpu::TextureFormat::Rgba8Unorm;
	let mut painter = Painter::new(&gpu, format);
	let mut fonts = Fonts::new().expect("fonts");
	println!(
		"family: {} nerd: {} italic: {}",
		fonts.family(),
		fonts.has_nerd_font(),
		fonts.has_italic()
	);
	let scale = 2.0_f32;
	let px = 14.0 * scale;
	let ctx = UiContext {
		charset: if fonts.has_nerd_font() {
			Charset::NerdFont
		} else {
			Charset::Unicode
		},
		graphics: Graphics::KittyPlaceholders,
		native_decor: true,
		..UiContext::default()
	};
	// Opaque backdrop so shots read on any viewer.
	let theme = GuiTheme::from_ctx(&ctx, 1.0);
	let mut scene = ChatScene::new(&ctx);
	let viewport = Size::new(104, 30);
	scene.resize(viewport, true);

	let metrics = fonts.cell_metrics(px);
	let margin = 10.0 * scale;
	let width = (f32::from(viewport.width) * metrics.advance + margin * 2.0).ceil() as u32;
	let height = (f32::from(viewport.height) * metrics.line_height + margin * 2.0).ceil() as u32;

	let wait = match name {
		"welcome" => Duration::from_millis(1600),
		_ => Duration::from_millis(5200),
	};
	if name != "welcome" {
		scene.key(Key::Enter);
	}
	std::thread::sleep(wait);
	if name == "picker" {
		scene.key(Key::Ctrl('p'));
	}

	// Wall-clock seed so successive shots capture different shimmer phases;
	// the scene's own animations still run off `scene.started`.
	let now = std::time::SystemTime::now()
		.duration_since(std::time::UNIX_EPOCH)
		.unwrap_or_else(|_| scene.started.elapsed());
	let frame = scene.render();
	let view = View {
		window: [width as f32, height as f32],
		origin: [margin, margin],
		scroll: 0.0,
		cursor_on: true,
		selection: None,
		now,
	};
	let mut compositor = Compositor::default();
	let instances = compositor.build(&frame, &mut fonts, &theme, &view, px);
	let (mask, color) = fonts.take_uploads();
	painter.upload_atlas(&gpu, &mask, &color);

	let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
		label: Some("shot-target"),
		size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
		mip_level_count: 1,
		sample_count: 1,
		dimension: wgpu::TextureDimension::D2,
		format,
		usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
		view_formats: &[],
	});
	let target_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
	painter.draw(
		&gpu,
		&target_view,
		width,
		height,
		&instances.batches,
		&instances.rects,
		&instances.glyphs,
	);

	let row_bytes = width * 4;
	let aligned = row_bytes.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
	let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
		label:              Some("shot-readback"),
		size:               u64::from(aligned) * u64::from(height),
		usage:              wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
		mapped_at_creation: false,
	});
	let mut encoder = gpu
		.device
		.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("shot-copy") });
	encoder.copy_texture_to_buffer(
		wgpu::TexelCopyTextureInfo {
			texture:   &texture,
			mip_level: 0,
			origin:    wgpu::Origin3d::ZERO,
			aspect:    wgpu::TextureAspect::All,
		},
		wgpu::TexelCopyBufferInfo {
			buffer: &buffer,
			layout: wgpu::TexelCopyBufferLayout {
				offset:         0,
				bytes_per_row:  Some(aligned),
				rows_per_image: Some(height),
			},
		},
		wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
	);
	gpu.queue.submit([encoder.finish()]);

	let slice = buffer.slice(..);
	let (sender, receiver) = flume::bounded(1);
	slice.map_async(wgpu::MapMode::Read, move |result| {
		sender.send(result).expect("map channel open");
	});
	gpu.device
		.poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
		.expect("device poll");
	receiver
		.recv()
		.expect("map channel result")
		.expect("buffer map");

	let mut rgba = vec![0_u8; (row_bytes * height) as usize];
	{
		let data = slice.get_mapped_range().expect("mapped range");
		for row in 0..height as usize {
			let src = &data[row * aligned as usize..][..row_bytes as usize];
			rgba[row * row_bytes as usize..][..row_bytes as usize].copy_from_slice(src);
		}
	}
	let file = std::fs::File::create(out).expect("create output png");
	let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
	encoder.set_color(png::ColorType::Rgba);
	encoder.set_depth(png::BitDepth::Eight);
	encoder
		.write_header()
		.expect("png header")
		.write_image_data(&rgba)
		.expect("png pixels");
	println!("wrote {out} ({width}x{height})");
}
