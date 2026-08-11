//! Graphviz DOT-to-terminal rendering for fenced Markdown blocks.


use layout::{
	core::{
		format::{ClipHandle, RenderBackend},
		geometry::Point,
		style::StyleAttr,
	},
	gv::{
		GraphBuilder,
		parser::{
			DotParser,
			ast::{self, AttrStmt, AttrStmtTarget, AttributeList, Stmt},
		},
	},
};
use omp_core::Str;
use xutf::Text;

use super::DiagramStyles;
use crate::{
	context::Charset,
	rich::{Pipeline, RichSink, cell_width},
};

const PIXELS_PER_COLUMN: f64 = 12.0;
const PIXELS_PER_ROW: f64 = 14.0;
const MAX_RENDER_CELLS: usize = 2_000_000;

/// Renders Graphviz DOT source without invoking the Graphviz executable.
/// Returns `false` without emitting so Markdown can preserve invalid source.
pub(super) fn render(
	source: &str,
	width: u16,
	charset: Charset,
	styles: DiagramStyles,
	sink: &mut dyn RichSink,
) -> bool {
	let source = source.trim();
	if source.is_empty() || width == 0 {
		return false;
	}

	let Some(scene) = render_best(source, usize::from(width)) else {
		return false;
	};
	let Some(raster) = Raster::new(&scene, usize::from(width), charset) else {
		return false;
	};
	raster.emit(width, styles, sink)
}

fn render_best(source: &str, width: usize) -> Option<Scene> {
	let mut parser = DotParser::new(source);
	let graph = parser.process().ok()?;
	let mut best = layout_scene(&graph)?;
	let mut best_width = best.natural_width();
	if best_width <= width {
		return Some(best);
	}

	// DOT commonly arrives with a wide LR rank direction chosen for a browser.
	// Try both terminal-friendly primary orientations and keep the narrowest.
	for direction in ["TB", "LR"] {
		let candidate_graph = with_rank_direction(&graph, direction);
		let Some(candidate) = layout_scene(&candidate_graph) else {
			continue;
		};
		let candidate_width = candidate.natural_width();
		if candidate_width < best_width {
			best = candidate;
			best_width = candidate_width;
		}
	}
	Some(best)
}

fn with_rank_direction(graph: &ast::Graph, direction: &str) -> ast::Graph {
	let mut graph = graph.clone();
	let mut attributes = AttributeList::new();
	attributes.add_attr("rankdir", direction);
	graph.list.list.push(Stmt::Attribute(AttrStmt::new(
		AttrStmtTarget::Graph,
		attributes,
	)));
	graph
}

fn layout_scene(graph: &ast::Graph) -> Option<Scene> {
	let mut builder = GraphBuilder::new();
	builder.visit_graph(graph);
	let mut graph = builder.get();
	if graph.num_nodes() == 0 {
		return None;
	}

	let mut scene = Scene::default();
	graph.do_it(false, false, false, &mut scene);
	(!scene.primitives.is_empty() && scene.bounds.is_some()).then_some(scene)
}

#[derive(Clone, Copy, Debug)]
struct Bounds {
	min: Point,
	max: Point,
}

impl Bounds {
	fn point(point: Point) -> Option<Self> {
		(point.x.is_finite() && point.y.is_finite()).then_some(Self { min: point, max: point })
	}

	fn include(&mut self, point: Point) {
		if !point.x.is_finite() || !point.y.is_finite() {
			return;
		}
		self.min.x = self.min.x.min(point.x);
		self.min.y = self.min.y.min(point.y);
		self.max.x = self.max.x.max(point.x);
		self.max.y = self.max.y.max(point.y);
	}

	fn width(self) -> f64 {
		(self.max.x - self.min.x).max(0.0)
	}

	fn height(self) -> f64 {
		(self.max.y - self.min.y).max(0.0)
	}
}

#[derive(Debug)]
enum Primitive {
	Rect {
		origin: Point,
		size:   Point,
	},
	Line {
		start: Point,
		stop:  Point,
	},
	Ellipse {
		center: Point,
		size:   Point,
	},
	Text {
		center: Point,
		text:   Str,
	},
	Arrow {
		path:   Vec<(Point, Point)>,
		dashed: bool,
		heads:  (bool, bool),
		text:   Str,
	},
}

#[derive(Default, Debug)]
struct Scene {
	primitives: Vec<Primitive>,
	bounds:     Option<Bounds>,
	clips:      usize,
}

impl Scene {
	fn include(&mut self, point: Point) {
		if let Some(bounds) = &mut self.bounds {
			bounds.include(point);
		} else {
			self.bounds = Bounds::point(point);
		}
	}

	fn include_rect(&mut self, origin: Point, size: Point) {
		self.include(origin);
		self.include(Point::new(origin.x + size.x, origin.y + size.y));
	}

	fn include_text(&mut self, center: Point, text: &str, font_size: usize) {
		let columns = text.lines().map(Text::visible_width).max().unwrap_or(0) as f64;
		let rows = text.lines().count().max(1) as f64;
		let half = Point::new(
			columns * font_size as f64 / 2.0,
			rows * font_size as f64 / 2.0,
		);
		self.include(Point::new(center.x - half.x, center.y - half.y));
		self.include(Point::new(center.x + half.x, center.y + half.y));
	}

	fn natural_width(&self) -> usize {
		self.bounds.map_or(0, |bounds| {
			((bounds.width() / PIXELS_PER_COLUMN).ceil() as usize).saturating_add(1)
		})
	}
}

impl RenderBackend for Scene {
	fn draw_rect(
		&mut self,
		origin: Point,
		size: Point,
		_look: &StyleAttr,
		_properties: Option<String>,
		_clip: Option<ClipHandle>,
	) {
		self.include_rect(origin, size);
		self.primitives.push(Primitive::Rect { origin, size });
	}

	fn draw_line(
		&mut self,
		start: Point,
		stop: Point,
		_look: &StyleAttr,
		_properties: Option<String>,
	) {
		self.include(start);
		self.include(stop);
		self.primitives.push(Primitive::Line { start, stop });
	}

	fn draw_circle(
		&mut self,
		center: Point,
		size: Point,
		_look: &StyleAttr,
		_properties: Option<String>,
	) {
		let half = Point::new(size.x / 2.0, size.y / 2.0);
		self.include(Point::new(center.x - half.x, center.y - half.y));
		self.include(Point::new(center.x + half.x, center.y + half.y));
		self.primitives.push(Primitive::Ellipse { center, size });
	}

	fn draw_text(&mut self, center: Point, text: &str, look: &StyleAttr) {
		if text.is_empty() {
			return;
		}
		self.include_text(center, text, look.font_size);
		self.primitives.push(Primitive::Text { center, text: Str::new(text) });
	}

	fn draw_arrow(
		&mut self,
		path: &[(Point, Point)],
		dashed: bool,
		heads: (bool, bool),
		look: &StyleAttr,
		_properties: Option<String>,
		text: &str,
	) {
		if path.len() < 2 {
			return;
		}
		for &(point, control) in path {
			self.include(point);
			self.include(control);
		}
		if !text.is_empty() {
			self.include_text(path[path.len() / 2].1, text, look.font_size);
		}
		self.primitives.push(Primitive::Arrow {
			path: path.to_vec(),
			dashed,
			heads,
			text: Str::new(text),
		});
	}

	fn create_clip(&mut self, _origin: Point, _size: Point, _rounded_px: usize) -> ClipHandle {
		let handle = self.clips;
		self.clips += 1;
		handle
	}
}

#[derive(Clone, Copy, Debug)]
struct Projection {
	bounds:  Bounds,
	x_scale: f64,
	y_scale: f64,
	columns: usize,
	rows:    usize,
}

impl Projection {
	fn new(bounds: Bounds, width: usize) -> Option<Self> {
		if width == 0 {
			return None;
		}
		let x_scale = if width == 1 {
			bounds.width().max(PIXELS_PER_COLUMN)
		} else {
			PIXELS_PER_COLUMN.max(bounds.width() / (width - 1) as f64)
		};
		let columns = if width == 1 {
			1
		} else {
			((bounds.width() / x_scale).ceil() as usize + 1).min(width)
		};

		let mut y_scale = PIXELS_PER_ROW;
		let mut rows = (bounds.height() / y_scale).ceil() as usize + 1;
		let max_rows = (MAX_RENDER_CELLS / columns.max(1)).max(1);
		if rows > max_rows {
			y_scale = if max_rows == 1 {
				bounds.height().max(PIXELS_PER_ROW)
			} else {
				(bounds.height() / (max_rows - 1) as f64).max(PIXELS_PER_ROW)
			};
			rows = max_rows;
		}

		Some(Self { bounds, x_scale, y_scale, columns, rows })
	}

	fn point(self, point: Point) -> GridPoint {
		let x = if self.columns == 1 {
			0
		} else {
			((point.x - self.bounds.min.x) / self.x_scale)
				.round()
				.clamp(0.0, (self.columns - 1) as f64) as i32
		};
		let y = if self.rows == 1 {
			0
		} else {
			((point.y - self.bounds.min.y) / self.y_scale)
				.round()
				.clamp(0.0, (self.rows - 1) as f64) as i32
		};
		GridPoint { x, y }
	}

}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GridPoint {
	x: i32,
	y: i32,
}

const NORTH: u8 = 1 << 0;
const EAST: u8 = 1 << 1;
const SOUTH: u8 = 1 << 2;
const WEST: u8 = 1 << 3;
const NORTH_EAST: u8 = 1 << 4;
const SOUTH_EAST: u8 = 1 << 5;
const SOUTH_WEST: u8 = 1 << 6;
const NORTH_WEST: u8 = 1 << 7;
const VERTICAL: u8 = NORTH | SOUTH;
const HORIZONTAL: u8 = EAST | WEST;
const TOP_LEFT: u8 = EAST | SOUTH;
const TOP_RIGHT: u8 = SOUTH | WEST;
const BOTTOM_LEFT: u8 = NORTH | EAST;
const BOTTOM_RIGHT: u8 = NORTH | WEST;
const TEE_RIGHT: u8 = NORTH | EAST | SOUTH;
const TEE_LEFT: u8 = NORTH | SOUTH | WEST;
const TEE_DOWN: u8 = EAST | SOUTH | WEST;
const TEE_UP: u8 = NORTH | EAST | WEST;
const CROSS: u8 = NORTH | EAST | SOUTH | WEST;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CellRole {
	Text,
	Line,
	Accent,
}

#[derive(Clone, Debug)]
struct Cell {
	line:             u8,
	line_priority:    u8,
	glyph:            Option<Str>,
	glyph_role:       CellRole,
	glyph_priority:   u8,
	wide_continuation: bool,
}

impl Default for Cell {
	fn default() -> Self {
		Self {
			line: 0,
			line_priority: 0,
			glyph: None,
			glyph_role: CellRole::Text,
			glyph_priority: 0,
			wide_continuation: false,
		}
	}
}

struct Raster {
	projection: Projection,
	cells:      Vec<Cell>,
	ascii:      bool,
}

impl Raster {
	fn new(scene: &Scene, width: usize, charset: Charset) -> Option<Self> {
		let projection = Projection::new(scene.bounds?, width)?;
		let cell_count = projection.columns.checked_mul(projection.rows)?;
		let mut raster = Self {
			projection,
			cells: vec![Cell::default(); cell_count],
			ascii: matches!(charset, Charset::Ascii),
		};
		let mut index = 0;
		while let Some(primitive) = scene.primitives.get(index) {
			if let Primitive::Ellipse { center, size } = primitive
				&& let Some(Primitive::Ellipse { center: outer_center, .. }) =
					scene.primitives.get(index + 1)
				&& concentric(*center, *outer_center)
			{
				raster.draw_ellipse(*center, *size, true);
				index += 2;
				continue;
			}
			raster.draw(primitive);
			index += 1;
		}
		Some(raster)
	}

	fn draw(&mut self, primitive: &Primitive) {
		match primitive {
			Primitive::Rect { origin, size } => self.draw_rect(*origin, *size),
			Primitive::Line { start, stop } => {
				let mut phase = 0;
				self.stroke(self.projection.point(*start), self.projection.point(*stop), 1, false, &mut phase);
			},
			Primitive::Ellipse { center, size } => self.draw_ellipse(*center, *size, false),
			Primitive::Text { center, text } => self.draw_text(*center, text.as_str(), 5),
			Primitive::Arrow { path, dashed, heads, text } => {
				self.draw_arrow(path, *dashed, *heads, text.as_str());
			},
		}
	}

	fn draw_rect(&mut self, origin: Point, size: Point) {
		let top_left = self.projection.point(origin);
		let bottom_right = self
			.projection
			.point(Point::new(origin.x + size.x, origin.y + size.y));
		let center = self
			.projection
			.point(Point::new(origin.x + size.x / 2.0, origin.y + size.y / 2.0));
		let top = top_left.y.min((center.y - 1).max(0));
		let bottom = bottom_right
			.y
			.max(center.y + 1)
			.min(self.projection.rows.saturating_sub(1) as i32);
		let top_left = GridPoint { x: top_left.x, y: top };
		let bottom_right = GridPoint { x: bottom_right.x, y: bottom };
		let top_right = GridPoint { x: bottom_right.x, y: top_left.y };
		let bottom_left = GridPoint { x: top_left.x, y: bottom_right.y };
		let mut phase = 0;
		self.stroke(top_left, top_right, 2, false, &mut phase);
		self.stroke(top_right, bottom_right, 2, false, &mut phase);
		self.stroke(bottom_right, bottom_left, 2, false, &mut phase);
		self.stroke(bottom_left, top_left, 2, false, &mut phase);
	}

	fn draw_ellipse(&mut self, center: Point, size: Point, double: bool) {
		let half = Point::new(size.x / 2.0, size.y / 2.0);
		let left = self.projection.point(Point::new(center.x - half.x, center.y)).x;
		let right = self.projection.point(Point::new(center.x + half.x, center.y)).x;
		let center = self.projection.point(center);
		let top = (center.y - 1).max(0);
		let bottom = (center.y + 1).min(self.projection.rows.saturating_sub(1) as i32);
		let (left, right) = (left.min(right), left.max(right));
		if top >= bottom || left >= right {
			return;
		}

		let (horizontal, vertical, top_left, top_right, bottom_left, bottom_right) =
			if self.ascii {
				('-', '|', '+', '+', '+', '+')
			} else if double {
				('═', '║', '╔', '╗', '╚', '╝')
			} else {
				('─', '│', '╭', '╮', '╰', '╯')
			};
		for x in left + 1..right {
			self.set_char(x, top, horizontal, CellRole::Line, 3);
			self.set_char(x, bottom, horizontal, CellRole::Line, 3);
		}
		for y in top + 1..bottom {
			self.set_char(left, y, vertical, CellRole::Line, 3);
			self.set_char(right, y, vertical, CellRole::Line, 3);
		}
		self.set_char(left, top, top_left, CellRole::Line, 3);
		self.set_char(right, top, top_right, CellRole::Line, 3);
		self.set_char(left, bottom, bottom_left, CellRole::Line, 3);
		self.set_char(right, bottom, bottom_right, CellRole::Line, 3);
	}

	fn draw_arrow(&mut self, path: &[(Point, Point)], dashed: bool, heads: (bool, bool), text: &str) {
		if path.len() < 2 {
			return;
		}
		let mut phase = 0;
		self.orthogonal(path[0].0, path[0].1, path[1].0, path[1].1, dashed, &mut phase);
		let mut endpoint = path[1].1;
		let mut previous_control = path[1].0;
		for &(control, next) in path.iter().skip(2) {
			let reflected = Point::new(
				2.0 * endpoint.x - previous_control.x,
				2.0 * endpoint.y - previous_control.y,
			);
			self.orthogonal(endpoint, reflected, control, next, dashed, &mut phase);
			endpoint = next;
			previous_control = control;
		}

		if heads.0 {
			let direction = Point::new(path[0].0.x - path[0].1.x, path[0].0.y - path[0].1.y);
			self.arrowhead(path[0].0, direction);
		}
		if heads.1 {
			let last = path[path.len() - 1];
			let direction = Point::new(last.1.x - last.0.x, last.1.y - last.0.y);
			self.arrowhead(last.1, direction);
		}
		if !text.is_empty() {
			self.draw_text(path[path.len() / 2].1, text, 5);
		}
	}

	fn orthogonal(
		&mut self,
		start: Point,
		control_a: Point,
		control_b: Point,
		stop: Point,
		dashed: bool,
		phase: &mut usize,
	) {
		let start_axis = horizontal_tangent(start, control_a);
		let stop_axis = horizontal_tangent(control_b, stop);
		let start = self.projection.point(start);
		let stop = self.projection.point(stop);
		let (points, len) = match (start_axis, stop_axis) {
			(true, true) => {
				let middle = (start.x + stop.x) / 2;
				([
					start,
					GridPoint { x: middle, y: start.y },
					GridPoint { x: middle, y: stop.y },
					stop,
				], 4)
			},
			(false, false) => {
				let middle = (start.y + stop.y) / 2;
				([
					start,
					GridPoint { x: start.x, y: middle },
					GridPoint { x: stop.x, y: middle },
					stop,
				], 4)
			},
			(true, false) => ([
				start,
				GridPoint { x: stop.x, y: start.y },
				stop,
				stop,
			], 3),
			(false, true) => ([
				start,
				GridPoint { x: start.x, y: stop.y },
				stop,
				stop,
			], 3),
		};
		for segment in points[..len].windows(2) {
			self.stroke(segment[0], segment[1], 1, dashed, phase);
		}
	}

	fn arrowhead(&mut self, point: Point, direction: Point) {
		let glyph = arrow_glyph(direction, self.ascii);
		let point = self.projection.point(point);
		self.set_char(point.x, point.y, glyph, CellRole::Accent, 4);
	}

	fn draw_text(&mut self, center: Point, text: &str, priority: u8) {
		if text.is_empty() {
			return;
		}
		let center = self.projection.point(center);
		let lines = text.lines().count().max(1) as i32;
		let start_y = center.y - (lines - 1) / 2;
		for (line_index, line) in text.lines().enumerate() {
			let mut x = center.x - i32::from(cell_width(line)) / 2;
			let y = start_y + line_index as i32;
			for grapheme in line.graphemes() {
				let width = usize::from(cell_width(grapheme));
				if width == 0 {
					continue;
				}
				self.set_glyph(x, y, grapheme, CellRole::Text, priority, width);
				x += width as i32;
			}
		}
	}

	fn set_char(&mut self, x: i32, y: i32, glyph: char, role: CellRole, priority: u8) {
		let mut buffer = [0; 4];
		self.set_glyph(x, y, glyph.encode_utf8(&mut buffer), role, priority, 1);
	}

	fn set_glyph(
		&mut self,
		x: i32,
		y: i32,
		glyph: &str,
		role: CellRole,
		priority: u8,
		width: usize,
	) {
		if x < 0 || y < 0 || width == 0 {
			return;
		}
		let (x, y) = (x as usize, y as usize);
		if y >= self.projection.rows || x.saturating_add(width) > self.projection.columns {
			return;
		}
		let index = self.index(x, y);
		if priority < self.cells[index].glyph_priority {
			return;
		}
		self.cells[index].glyph = Some(Str::new(glyph));
		self.cells[index].glyph_role = role;
		self.cells[index].glyph_priority = priority;
		self.cells[index].wide_continuation = false;
		for column in x + 1..x + width {
			let continuation = self.index(column, y);
			self.cells[continuation].glyph = None;
			self.cells[continuation].glyph_priority = priority;
			self.cells[continuation].wide_continuation = true;
		}
	}

	fn stroke(
		&mut self,
		start: GridPoint,
		stop: GridPoint,
		priority: u8,
		dashed: bool,
		phase: &mut usize,
	) {
		if start == stop {
			return;
		}

		let (mut x, mut y) = (start.x, start.y);
		let dx = (stop.x - start.x).abs();
		let sx = if start.x < stop.x { 1 } else { -1 };
		let dy = -(stop.y - start.y).abs();
		let sy = if start.y < stop.y { 1 } else { -1 };
		let mut error = dx + dy;
		loop {
			if x == stop.x && y == stop.y {
				break;
			}
			let twice = 2 * error;
			let mut next_x = x;
			let mut next_y = y;
			if twice >= dy {
				error += dy;
				next_x += sx;
			}
			if twice <= dx {
				error += dx;
				next_y += sy;
			}
			if !dashed || (*phase / 2).is_multiple_of(2) {
				self.connect(GridPoint { x, y }, GridPoint { x: next_x, y: next_y }, priority);
			}
			*phase += 1;
			x = next_x;
			y = next_y;
		}
	}


	fn connect(&mut self, from: GridPoint, to: GridPoint, priority: u8) {
		let dx = to.x - from.x;
		let dy = to.y - from.y;
		let Some((outgoing, incoming)) = direction_bits(dx, dy) else {
			return;
		};
		self.add_line(from, outgoing, priority);
		self.add_line(to, incoming, priority);
	}

	fn add_line(&mut self, point: GridPoint, direction: u8, priority: u8) {
		if point.x < 0 || point.y < 0 {
			return;
		}
		let (x, y) = (point.x as usize, point.y as usize);
		if x >= self.projection.columns || y >= self.projection.rows {
			return;
		}
		let cell = &mut self.cells[y * self.projection.columns + x];
		if priority > cell.line_priority {
			cell.line = direction;
			cell.line_priority = priority;
		} else if priority == cell.line_priority {
			cell.line |= direction;
		}
	}

	fn index(&self, x: usize, y: usize) -> usize {
		y * self.projection.columns + x
	}

	fn emit(&self, width: u16, styles: DiagramStyles, sink: &mut dyn RichSink) -> bool {
		let Some((first_row, last_row)) = self.occupied_rows() else {
			return false;
		};
		for row in first_row..=last_row {
			let mut clip = (&mut *sink).clip(width, None);
			self.emit_row(row, styles, &mut clip);
			clip.newline();
		}
		true
	}

	fn occupied_rows(&self) -> Option<(usize, usize)> {
		let first = (0..self.projection.rows).find(|&row| self.row_end(row).is_some())?;
		let last = (first..self.projection.rows)
			.rev()
			.find(|&row| self.row_end(row).is_some())?;
		Some((first, last))
	}

	fn row_end(&self, row: usize) -> Option<usize> {
		(0..self.projection.columns)
			.rev()
			.find(|&column| self.cell(column, row).is_occupied())
	}

	fn emit_row(&self, row: usize, styles: DiagramStyles, sink: &mut dyn RichSink) {
		let Some(end) = self.row_end(row) else {
			return;
		};
		let mut current_role = CellRole::Text;
		let mut run = String::new();
		for column in 0..=end {
			let cell = self.cell(column, row);
			if cell.wide_continuation {
				continue;
			}
			let (glyph, role) = if let Some(glyph) = &cell.glyph {
				(glyph.as_str(), cell.glyph_role)
			} else if cell.line != 0 {
				let glyph = line_glyph(cell.line, self.ascii);
				if role_changed(current_role, CellRole::Line, &run) {
					sink.run(style_for(current_role, styles), &run);
					run.clear();
					current_role = CellRole::Line;
				}
				run.push(glyph);
				continue;
			} else {
				(" ", CellRole::Text)
			};
			if role_changed(current_role, role, &run) {
				sink.run(style_for(current_role, styles), &run);
				run.clear();
				current_role = role;
			}
			run.push_str(glyph);
		}
		if !run.is_empty() {
			sink.run(style_for(current_role, styles), &run);
		}
	}

	fn cell(&self, column: usize, row: usize) -> &Cell {
		&self.cells[self.index(column, row)]
	}
}

impl Cell {
	fn is_occupied(&self) -> bool {
		self.line != 0 || self.glyph.is_some() || self.wide_continuation
	}
}

fn role_changed(current: CellRole, next: CellRole, run: &str) -> bool {
	!run.is_empty() && current != next
}

const fn style_for(role: CellRole, styles: DiagramStyles) -> crate::Style {
	match role {
		CellRole::Text => styles.text,
		CellRole::Line => styles.line,
		CellRole::Accent => styles.accent,
	}
}

fn concentric(left: Point, right: Point) -> bool {
	(left.x - right.x).abs() < f64::EPSILON && (left.y - right.y).abs() < f64::EPSILON
}

fn horizontal_tangent(from: Point, to: Point) -> bool {
	(to.x - from.x).abs() >= (to.y - from.y).abs()
}


const fn direction_bits(dx: i32, dy: i32) -> Option<(u8, u8)> {
	match (dx.signum(), dy.signum()) {
		(0, -1) => Some((NORTH, SOUTH)),
		(1, -1) => Some((NORTH_EAST, SOUTH_WEST)),
		(1, 0) => Some((EAST, WEST)),
		(1, 1) => Some((SOUTH_EAST, NORTH_WEST)),
		(0, 1) => Some((SOUTH, NORTH)),
		(-1, 1) => Some((SOUTH_WEST, NORTH_EAST)),
		(-1, 0) => Some((WEST, EAST)),
		(-1, -1) => Some((NORTH_WEST, SOUTH_EAST)),
		_ => None,
	}
}

fn line_glyph(mask: u8, ascii: bool) -> char {
	let orthogonal = mask & (NORTH | EAST | SOUTH | WEST);
	let diagonal = mask & (NORTH_EAST | SOUTH_EAST | SOUTH_WEST | NORTH_WEST);
	if ascii {
		return if diagonal == 0 {
			match orthogonal {
				EAST | WEST | HORIZONTAL => '-',
				NORTH | SOUTH | VERTICAL => '|',
				_ => '+',
			}
		} else if orthogonal == 0 && diagonal & (NORTH_EAST | SOUTH_WEST) != 0 {
			'/'
		} else if orthogonal == 0 && diagonal & (NORTH_WEST | SOUTH_EAST) != 0 {
			'\\'
		} else {
			'+'
		};
	}

	if diagonal == 0 {
		return match orthogonal {
			NORTH | SOUTH | VERTICAL => '│',
			EAST | WEST | HORIZONTAL => '─',
			TOP_LEFT => '┌',
			TOP_RIGHT => '┐',
			BOTTOM_LEFT => '└',
			BOTTOM_RIGHT => '┘',
			TEE_RIGHT => '├',
			TEE_LEFT => '┤',
			TEE_DOWN => '┬',
			TEE_UP => '┴',
			CROSS => '┼',
			_ => '─',
		};
	}
	if orthogonal == 0 {
		let rising = diagonal & (NORTH_EAST | SOUTH_WEST) != 0;
		let falling = diagonal & (NORTH_WEST | SOUTH_EAST) != 0;
		return match (rising, falling) {
			(true, false) => '╱',
			(false, true) => '╲',
			_ => '╳',
		};
	}
	if mask.count_ones() > 2 {
		return '┼';
	}

	let (mut x, mut y) = (0_i8, 0_i8);
	for (bit, dx, dy) in [
		(NORTH, 0, -1),
		(NORTH_EAST, 1, -1),
		(EAST, 1, 0),
		(SOUTH_EAST, 1, 1),
		(SOUTH, 0, 1),
		(SOUTH_WEST, -1, 1),
		(WEST, -1, 0),
		(NORTH_WEST, -1, -1),
	] {
		if mask & bit != 0 {
			x += dx;
			y += dy;
		}
	}
	if x.abs() > y.abs() * 2 {
		'─'
	} else if y.abs() > x.abs() * 2 {
		'│'
	} else if x.signum() == y.signum() {
		'╲'
	} else {
		'╱'
	}
}

fn arrow_glyph(direction: Point, ascii: bool) -> char {
	let horizontal = direction.x.abs();
	let vertical = direction.y.abs();
	if horizontal > vertical * 1.5 {
		if direction.x < 0.0 {
			if ascii { '<' } else { '◀' }
		} else if ascii {
			'>'
		} else {
			'▶'
		}
	} else if vertical > horizontal * 1.5 {
		if direction.y < 0.0 {
			if ascii { '^' } else { '▲' }
		} else if ascii {
			'v'
		} else {
			'▼'
		}
	} else if ascii {
		if direction.x < 0.0 { '<' } else { '>' }
	} else {
		match (direction.x.is_sign_negative(), direction.y.is_sign_negative()) {
			(true, true) => '↖',
			(false, true) => '↗',
			(false, false) => '↘',
			(true, false) => '↙',
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn line_masks_select_terminal_junctions() {
		assert_eq!(line_glyph(EAST | WEST, false), '─');
		assert_eq!(line_glyph(NORTH | EAST | SOUTH, false), '├');
		assert_eq!(line_glyph(NORTH_EAST | SOUTH_WEST, false), '╱');
		assert_eq!(line_glyph(NORTH | EAST, true), '+');
	}

	#[test]
	fn arrows_follow_cardinal_and_diagonal_tangents() {
		assert_eq!(arrow_glyph(Point::new(4.0, 0.0), false), '▶');
		assert_eq!(arrow_glyph(Point::new(0.0, -4.0), false), '▲');
		assert_eq!(arrow_glyph(Point::new(-2.0, -2.0), false), '↖');
		assert_eq!(arrow_glyph(Point::new(2.0, 2.0), true), '>');
	}
}
