//! Analytic geometry and a compact bounding-volume hierarchy for ray tracing.

use std::f32::consts::PI;

use super::{Material, Ray, Vec3, vec3};

const BOUNDS_PADDING: f32 = 1.0e-4;
const PARALLEL_EPSILON: f32 = 1.0e-8;
const VECTOR_EPSILON_SQUARED: f32 = PARALLEL_EPSILON * PARALLEL_EPSILON;
const LEAF_SIZE: usize = 4;
const TRAVERSAL_STACK_SIZE: usize = usize::BITS as usize;

/// An axis-aligned bounding box represented by inclusive minimum and maximum
/// corners.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
	/// Smallest coordinate on each axis.
	pub min: Vec3,
	/// Largest coordinate on each axis.
	pub max: Vec3,
}

impl Aabb {
	/// Creates a bounding box from already ordered corners.
	pub const fn new(min: Vec3, max: Vec3) -> Self {
		Self { min, max }
	}

	/// Creates the smallest box containing both points.
	pub const fn from_points(a: Vec3, b: Vec3) -> Self {
		Self {
			min: vec3(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z)),
			max: vec3(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z)),
		}
	}

	/// Returns whether every coordinate is finite and the corners are ordered.
	pub fn is_valid(self) -> bool {
		is_finite_vec(self.min)
			&& is_finite_vec(self.max)
			&& self.min.x <= self.max.x
			&& self.min.y <= self.max.y
			&& self.min.z <= self.max.z
	}

	/// Returns the smallest box containing both input boxes.
	pub const fn union(self, other: Self) -> Self {
		Self {
			min: vec3(
				self.min.x.min(other.min.x),
				self.min.y.min(other.min.y),
				self.min.z.min(other.min.z),
			),
			max: vec3(
				self.max.x.max(other.max.x),
				self.max.y.max(other.max.y),
				self.max.z.max(other.max.z),
			),
		}
	}

	/// Returns the midpoint of the box without overflowing finite coordinates.
	pub fn centroid(self) -> Vec3 {
		self.min * 0.5 + self.max * 0.5
	}

	/// Expands thin axes symmetrically to at least `minimum_extent` wide.
	pub fn padded(self, minimum_extent: f32) -> Self {
		if !self.is_valid() || !minimum_extent.is_finite() || minimum_extent <= 0.0 {
			return self;
		}
		let mut min = self.min;
		let mut max = self.max;
		if max.x - min.x < minimum_extent {
			let padding = (minimum_extent - (max.x - min.x)) * 0.5;
			min.x -= padding;
			max.x += padding;
		}
		if max.y - min.y < minimum_extent {
			let padding = (minimum_extent - (max.y - min.y)) * 0.5;
			min.y -= padding;
			max.y += padding;
		}
		if max.z - min.z < minimum_extent {
			let padding = (minimum_extent - (max.z - min.z)) * 0.5;
			min.z -= padding;
			max.z += padding;
		}
		Self { min, max }
	}

	/// Clips a ray interval against this box using the slab method.
	///
	/// Parallel rays are accepted only when their origin lies inside that slab.
	/// Invalid boxes, rays, or intervals return `None` rather than propagating
	/// NaNs.
	pub fn hit_interval(self, ray: Ray, mut t_min: f32, mut t_max: f32) -> Option<(f32, f32)> {
		if !self.is_valid()
			|| !is_finite_vec(ray.origin)
			|| !is_finite_vec(ray.dir)
			|| !t_min.is_finite()
			|| t_max.is_nan()
			|| t_min > t_max
		{
			return None;
		}

		for (origin, direction, slab_min, slab_max) in [
			(ray.origin.x, ray.dir.x, self.min.x, self.max.x),
			(ray.origin.y, ray.dir.y, self.min.y, self.max.y),
			(ray.origin.z, ray.dir.z, self.min.z, self.max.z),
		] {
			if direction.abs() <= PARALLEL_EPSILON {
				if origin < slab_min || origin > slab_max {
					return None;
				}
				continue;
			}
			let inverse = 1.0 / direction;
			let mut near = (slab_min - origin) * inverse;
			let mut far = (slab_max - origin) * inverse;
			if near > far {
				std::mem::swap(&mut near, &mut far);
			}
			t_min = t_min.max(near);
			t_max = t_max.min(far);
			if t_max < t_min {
				return None;
			}
		}
		Some((t_min, t_max))
	}

	fn longest_axis(self) -> usize {
		let extent = self.max - self.min;
		if extent.x >= extent.y && extent.x >= extent.z {
			0
		} else if extent.y >= extent.z {
			1
		} else {
			2
		}
	}
}

/// Geometry-local intersection data before a material and face orientation are
/// applied.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeometryHit {
	/// Ray parameter at the intersection.
	pub t:                f32,
	/// World-space intersection point.
	pub point:            Vec3,
	/// Outward geometric surface normal.
	pub geometric_normal: Vec3,
	/// Outward smooth or perturbed normal used for shading.
	pub shading_normal:   Vec3,
	/// Surface coordinates in the unit square where the primitive defines them.
	pub uv:               (f32, f32),
}

/// A bounded, thread-safe ray-intersectable shape.
pub trait Geometry: Send + Sync {
	/// Returns a finite world-space bound for the shape.
	fn bounds(&self) -> Aabb;

	/// Returns the closest intersection within the inclusive ray interval.
	fn intersect(&self, ray: Ray, t_min: f32, t_max: f32) -> Option<GeometryHit>;
}

/// An analytic sphere.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sphere {
	/// Sphere center in world space.
	pub center: Vec3,
	/// Sphere radius; non-positive or non-finite radii never intersect.
	pub radius: f32,
}

impl Sphere {
	/// Creates a sphere from its center and radius.
	pub const fn new(center: Vec3, radius: f32) -> Self {
		Self { center, radius }
	}
}

impl Geometry for Sphere {
	fn bounds(&self) -> Aabb {
		if !is_finite_vec(self.center) || !self.radius.is_finite() || self.radius <= 0.0 {
			return invalid_bounds();
		}
		let radius = vec3(self.radius, self.radius, self.radius);
		Aabb::new(self.center - radius, self.center + radius).padded(BOUNDS_PADDING)
	}

	fn intersect(&self, ray: Ray, t_min: f32, t_max: f32) -> Option<GeometryHit> {
		if !valid_query(ray, t_min, t_max)
			|| !is_finite_vec(self.center)
			|| !self.radius.is_finite()
			|| self.radius <= 0.0
		{
			return None;
		}
		let offset = ray.origin - self.center;
		let a = ray.dir.dot(ray.dir);
		if !a.is_finite() || a <= PARALLEL_EPSILON {
			return None;
		}
		let half_b = offset.dot(ray.dir);
		let c = offset.dot(offset) - self.radius * self.radius;
		let discriminant = half_b.mul_add(half_b, -a * c);
		if !discriminant.is_finite() || discriminant < 0.0 {
			return None;
		}
		let root = discriminant.sqrt();
		let mut distance = (-half_b - root) / a;
		if distance < t_min || distance > t_max {
			distance = (-half_b + root) / a;
			if distance < t_min || distance > t_max {
				return None;
			}
		}
		let point = ray.origin + ray.dir * distance;
		let normal = (point - self.center) * (1.0 / self.radius);
		if !is_finite_vec(point) || !is_finite_vec(normal) {
			return None;
		}
		let uv = sphere_uv(normal);
		Some(GeometryHit { t: distance, point, geometric_normal: normal, shading_normal: normal, uv })
	}
}

/// A finite parallelogram spanned from one corner by two edge vectors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quad {
	/// Corner corresponding to UV coordinate `(0, 0)`.
	pub origin: Vec3,
	/// Edge from `(0, 0)` to `(1, 0)`.
	pub u:      Vec3,
	/// Edge from `(0, 0)` to `(0, 1)`.
	pub v:      Vec3,
}

impl Quad {
	/// Creates a finite quad from a corner and two non-collinear edge vectors.
	pub const fn new(origin: Vec3, u: Vec3, v: Vec3) -> Self {
		Self { origin, u, v }
	}
}

impl Geometry for Quad {
	fn bounds(&self) -> Aabb {
		if !is_finite_vec(self.origin) || !is_finite_vec(self.u) || !is_finite_vec(self.v) {
			return invalid_bounds();
		}
		if quad_determinant(self.u, self.v).is_none() {
			return invalid_bounds();
		}
		let opposite = self.origin + self.u + self.v;
		let a = Aabb::from_points(self.origin, opposite);
		let b = Aabb::from_points(self.origin + self.u, self.origin + self.v);
		a.union(b).padded(BOUNDS_PADDING)
	}

	fn intersect(&self, ray: Ray, t_min: f32, t_max: f32) -> Option<GeometryHit> {
		if !valid_query(ray, t_min, t_max)
			|| !is_finite_vec(self.origin)
			|| !is_finite_vec(self.u)
			|| !is_finite_vec(self.v)
		{
			return None;
		}
		let cross = self.u.cross(self.v);
		let determinant = quad_determinant(self.u, self.v)?;
		let normal = unit_vector(cross)?;
		let denominator = normal.dot(ray.dir);
		if !denominator.is_finite() || denominator.abs() <= PARALLEL_EPSILON {
			return None;
		}
		let distance = normal.dot(self.origin - ray.origin) / denominator;
		if !distance.is_finite() || distance < t_min || distance > t_max {
			return None;
		}
		let point = ray.origin + ray.dir * distance;
		if !is_finite_vec(point) {
			return None;
		}
		let relative = point - self.origin;
		let uu = self.u.dot(self.u);
		let uv = self.u.dot(self.v);
		let vv = self.v.dot(self.v);
		let ru = relative.dot(self.u);
		let rv = relative.dot(self.v);
		let s = (ru * vv - rv * uv) / determinant;
		let t = (rv * uu - ru * uv) / determinant;
		if !s.is_finite() || !t.is_finite() || !(0.0..=1.0).contains(&s) || !(0.0..=1.0).contains(&t)
		{
			return None;
		}
		Some(GeometryHit {
			t: distance,
			point,
			geometric_normal: normal,
			shading_normal: normal,
			uv: (s, t),
		})
	}
}

/// A circular disk in an arbitrarily oriented plane.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Disk {
	/// Disk center in world space.
	pub center: Vec3,
	/// Disk plane normal; it is normalized when queried.
	pub normal: Vec3,
	/// Disk radius; non-positive or non-finite radii never intersect.
	pub radius: f32,
}

impl Disk {
	/// Creates an oriented disk from its center, normal, and radius.
	pub const fn new(center: Vec3, normal: Vec3, radius: f32) -> Self {
		Self { center, normal, radius }
	}
}

impl Geometry for Disk {
	fn bounds(&self) -> Aabb {
		let Some(normal) = unit_vector(self.normal) else {
			return invalid_bounds();
		};
		if !is_finite_vec(self.center) || !self.radius.is_finite() || self.radius <= 0.0 {
			return invalid_bounds();
		}
		let extent = vec3(
			self.radius * (1.0 - normal.x * normal.x).max(0.0).sqrt(),
			self.radius * (1.0 - normal.y * normal.y).max(0.0).sqrt(),
			self.radius * (1.0 - normal.z * normal.z).max(0.0).sqrt(),
		);
		Aabb::new(self.center - extent, self.center + extent).padded(BOUNDS_PADDING)
	}

	fn intersect(&self, ray: Ray, t_min: f32, t_max: f32) -> Option<GeometryHit> {
		if !valid_query(ray, t_min, t_max)
			|| !is_finite_vec(self.center)
			|| !self.radius.is_finite()
			|| self.radius <= 0.0
		{
			return None;
		}
		let normal = unit_vector(self.normal)?;
		let denominator = normal.dot(ray.dir);
		if !denominator.is_finite() || denominator.abs() <= PARALLEL_EPSILON {
			return None;
		}
		let distance = normal.dot(self.center - ray.origin) / denominator;
		if !distance.is_finite() || distance < t_min || distance > t_max {
			return None;
		}
		let point = ray.origin + ray.dir * distance;
		if !is_finite_vec(point) {
			return None;
		}
		let radial = point - self.center;
		let scaled_radial =
			vec3(radial.x / self.radius, radial.y / self.radius, radial.z / self.radius);
		let radial_squared = scaled_radial.dot(scaled_radial);
		if !radial_squared.is_finite() || radial_squared > 1.0 {
			return None;
		}
		let (tangent, bitangent) = disk_basis(normal);
		let uv = (0.5 + 0.5 * scaled_radial.dot(tangent), 0.5 + 0.5 * scaled_radial.dot(bitangent));
		Some(GeometryHit { t: distance, point, geometric_normal: normal, shading_normal: normal, uv })
	}
}

/// Common geometry stored inline, with a boxed fallback for user-defined
/// shapes.
pub enum Primitive {
	/// An inline analytic sphere.
	Sphere(Sphere),
	/// An inline finite quad.
	Quad(Quad),
	/// An inline oriented disk.
	Disk(Disk),
	/// A custom shape allocated once when the scene is constructed.
	Custom(Box<dyn Geometry>),
}

impl Primitive {
	/// Boxes a custom shape for storage alongside the built-in primitives.
	pub fn custom(geometry: impl Geometry + 'static) -> Self {
		Self::Custom(Box::new(geometry))
	}
}

impl From<Sphere> for Primitive {
	fn from(value: Sphere) -> Self {
		Self::Sphere(value)
	}
}

impl From<Quad> for Primitive {
	fn from(value: Quad) -> Self {
		Self::Quad(value)
	}
}

impl From<Disk> for Primitive {
	fn from(value: Disk) -> Self {
		Self::Disk(value)
	}
}

impl Geometry for Primitive {
	fn bounds(&self) -> Aabb {
		match self {
			Self::Sphere(shape) => shape.bounds(),
			Self::Quad(shape) => shape.bounds(),
			Self::Disk(shape) => shape.bounds(),
			Self::Custom(shape) => shape.bounds(),
		}
	}

	fn intersect(&self, ray: Ray, t_min: f32, t_max: f32) -> Option<GeometryHit> {
		match self {
			Self::Sphere(shape) => shape.intersect(ray, t_min, t_max),
			Self::Quad(shape) => shape.intersect(ray, t_min, t_max),
			Self::Disk(shape) => shape.intersect(ray, t_min, t_max),
			Self::Custom(shape) => shape.intersect(ray, t_min, t_max),
		}
	}
}

/// A scene primitive paired with its surface material.
pub struct Object {
	/// Shape used for bounds and intersection queries.
	pub primitive: Primitive,
	/// Surface material returned with intersections.
	pub material:  Material,
}

impl Object {
	/// Creates a renderable object from any built-in or explicit [`Primitive`].
	pub fn new(primitive: impl Into<Primitive>, material: Material) -> Self {
		Self { primitive: primitive.into(), material }
	}

	/// Returns the object's world-space bounds.
	pub fn bounds(&self) -> Aabb {
		self.primitive.bounds()
	}

	fn hit(&self, ray: Ray, t_min: f32, t_max: f32) -> Option<Hit<'_>> {
		let geometry_hit = self.primitive.intersect(ray, t_min, t_max)?;
		if !valid_geometry_hit(geometry_hit, t_min, t_max) {
			return None;
		}
		let geometric_outward = unit_vector(geometry_hit.geometric_normal)?;
		let mut shading_outward =
			unit_vector(geometry_hit.shading_normal).unwrap_or(geometric_outward);
		if shading_outward.dot(geometric_outward) < 0.0 {
			shading_outward *= -1.0;
		}
		let front_face = ray.dir.dot(geometric_outward) < 0.0;
		let orientation = if front_face { 1.0 } else { -1.0 };
		let geometric_normal = geometric_outward * orientation;
		let normal = shading_outward * orientation;
		Some(Hit {
			point: geometry_hit.point,
			geometric_normal,
			normal,
			uv: geometry_hit.uv,
			t: geometry_hit.t,
			front_face,
			material: &self.material,
		})
	}
}

/// A fully oriented scene intersection borrowing its object's material.
#[derive(Clone, Copy, Debug)]
pub struct Hit<'a> {
	/// World-space intersection point.
	pub point:            Vec3,
	/// Geometric normal oriented against the incoming ray.
	pub geometric_normal: Vec3,
	/// Shading normal oriented against the incoming ray and geometric normal.
	pub normal:           Vec3,
	/// Surface coordinates supplied by the primitive.
	pub uv:               (f32, f32),
	/// Positive ray parameter at the intersection.
	pub t:                f32,
	/// Whether the ray arrived on the outward-facing side of the surface.
	pub front_face:       bool,
	/// Material owned by the intersected object.
	pub material:         &'a Material,
}

#[derive(Clone, Copy, Debug)]
struct BvhNode {
	bounds: Aabb,
	kind:   BvhNodeKind,
}

#[derive(Clone, Copy, Debug)]
enum BvhNodeKind {
	Leaf { start: usize, len: usize },
	Branch { left: usize, right: usize },
}

/// An owning bounding-volume hierarchy over scene objects.
///
/// Construction discards objects with invalid bounds, orders the remaining
/// objects in place by median centroid splits, and stores no per-ray state.
pub struct Bvh {
	objects: Vec<Object>,
	nodes:   Vec<BvhNode>,
	root:    Option<usize>,
}

impl Bvh {
	/// Builds a balanced hierarchy from owned objects.
	pub fn new(mut objects: Vec<Object>) -> Self {
		objects.retain(|object| object.bounds().is_valid());
		let mut nodes = Vec::with_capacity(objects.len().saturating_mul(2));
		let root = if objects.is_empty() {
			None
		} else {
			let count = objects.len();
			Some(build_node(&mut objects, &mut nodes, 0, count))
		};
		Self { objects, nodes, root }
	}

	/// Returns the objects in BVH leaf order.
	pub fn objects(&self) -> &[Object] {
		&self.objects
	}

	/// Finds the nearest valid intersection inside the inclusive ray interval.
	pub fn hit(&self, ray: Ray, t_min: f32, t_max: f32) -> Option<Hit<'_>> {
		if !valid_query(ray, t_min, t_max) {
			return None;
		}
		let root = self.root?;
		self.nodes[root].bounds.hit_interval(ray, t_min, t_max)?;
		let mut stack = [0usize; TRAVERSAL_STACK_SIZE];
		let mut stack_len = 1;
		stack[0] = root;
		let mut closest = t_max;
		let mut result = None;

		while stack_len > 0 {
			stack_len -= 1;
			let node_index = stack[stack_len];
			let node = self.nodes[node_index];
			if node.bounds.hit_interval(ray, t_min, closest).is_none() {
				continue;
			}
			match node.kind {
				BvhNodeKind::Leaf { start, len } => {
					for object in &self.objects[start..start + len] {
						if let Some(hit) = object.hit(ray, t_min, closest) {
							closest = hit.t;
							result = Some(hit);
						}
					}
				},
				BvhNodeKind::Branch { left, right } => {
					let left_interval = self.nodes[left].bounds.hit_interval(ray, t_min, closest);
					let right_interval = self.nodes[right].bounds.hit_interval(ray, t_min, closest);
					match (left_interval, right_interval) {
						(Some(left_hit), Some(right_hit)) => {
							let (near, far) = if left_hit.0 <= right_hit.0 {
								(left, right)
							} else {
								(right, left)
							};
							push_node(&mut stack, &mut stack_len, far);
							push_node(&mut stack, &mut stack_len, near);
						},
						(Some(_), None) => push_node(&mut stack, &mut stack_len, left),
						(None, Some(_)) => push_node(&mut stack, &mut stack_len, right),
						(None, None) => {},
					}
				},
			}
		}
		result
	}

	/// Returns as soon as any object intersects inside the inclusive interval.
	pub fn occluded(&self, ray: Ray, t_min: f32, t_max: f32) -> bool {
		if !valid_query(ray, t_min, t_max) {
			return false;
		}
		let Some(root) = self.root else {
			return false;
		};
		if self.nodes[root]
			.bounds
			.hit_interval(ray, t_min, t_max)
			.is_none()
		{
			return false;
		}
		let mut stack = [0usize; TRAVERSAL_STACK_SIZE];
		let mut stack_len = 1;
		stack[0] = root;

		while stack_len > 0 {
			stack_len -= 1;
			let node = self.nodes[stack[stack_len]];
			if node.bounds.hit_interval(ray, t_min, t_max).is_none() {
				continue;
			}
			match node.kind {
				BvhNodeKind::Leaf { start, len } => {
					if self.objects[start..start + len]
						.iter()
						.any(|object| object.hit(ray, t_min, t_max).is_some())
					{
						return true;
					}
				},
				BvhNodeKind::Branch { left, right } => {
					let left_interval = self.nodes[left].bounds.hit_interval(ray, t_min, t_max);
					let right_interval = self.nodes[right].bounds.hit_interval(ray, t_min, t_max);
					match (left_interval, right_interval) {
						(Some(left_hit), Some(right_hit)) => {
							let (near, far) = if left_hit.0 <= right_hit.0 {
								(left, right)
							} else {
								(right, left)
							};
							push_node(&mut stack, &mut stack_len, far);
							push_node(&mut stack, &mut stack_len, near);
						},
						(Some(_), None) => push_node(&mut stack, &mut stack_len, left),
						(None, Some(_)) => push_node(&mut stack, &mut stack_len, right),
						(None, None) => {},
					}
				},
			}
		}
		false
	}
}

fn build_node(objects: &mut [Object], nodes: &mut Vec<BvhNode>, start: usize, end: usize) -> usize {
	let bounds = bounds_of(&objects[start..end]);
	let node_index = nodes.len();
	nodes.push(BvhNode { bounds, kind: BvhNodeKind::Leaf { start, len: end - start } });
	if end - start <= LEAF_SIZE {
		return node_index;
	}

	let centroid_bounds = centroid_bounds_of(&objects[start..end]);
	let axis = centroid_bounds.longest_axis();
	objects[start..end].sort_unstable_by(|a, b| {
		component(a.bounds().centroid(), axis).total_cmp(&component(b.bounds().centroid(), axis))
	});
	let middle = start + (end - start) / 2;
	let left = build_node(objects, nodes, start, middle);
	let right = build_node(objects, nodes, middle, end);
	nodes[node_index].kind = BvhNodeKind::Branch { left, right };
	node_index
}

fn bounds_of(objects: &[Object]) -> Aabb {
	let mut bounds = objects[0].bounds();
	for object in &objects[1..] {
		bounds = bounds.union(object.bounds());
	}
	bounds.padded(BOUNDS_PADDING)
}

fn centroid_bounds_of(objects: &[Object]) -> Aabb {
	let first = objects[0].bounds().centroid();
	let mut bounds = Aabb::new(first, first);
	for object in &objects[1..] {
		let centroid = object.bounds().centroid();
		bounds = bounds.union(Aabb::new(centroid, centroid));
	}
	bounds
}

const fn push_node(stack: &mut [usize; TRAVERSAL_STACK_SIZE], len: &mut usize, node: usize) {
	// Median splitting bounds the pending-node count by the machine word width.
	if *len < stack.len() {
		stack[*len] = node;
		*len += 1;
	}
}

const fn component(value: Vec3, axis: usize) -> f32 {
	match axis {
		0 => value.x,
		1 => value.y,
		_ => value.z,
	}
}

fn valid_query(ray: Ray, t_min: f32, t_max: f32) -> bool {
	is_finite_vec(ray.origin)
		&& is_finite_vec(ray.dir)
		&& ray.dir.dot(ray.dir) > VECTOR_EPSILON_SQUARED
		&& t_min.is_finite()
		&& !t_max.is_nan()
		&& t_min > 0.0
		&& t_max >= t_min
}

fn valid_geometry_hit(hit: GeometryHit, t_min: f32, t_max: f32) -> bool {
	hit.t.is_finite()
		&& hit.t >= t_min
		&& hit.t <= t_max
		&& is_finite_vec(hit.point)
		&& is_finite_vec(hit.geometric_normal)
		&& hit.uv.0.is_finite()
		&& hit.uv.1.is_finite()
}

const fn is_finite_vec(value: Vec3) -> bool {
	value.x.is_finite() && value.y.is_finite() && value.z.is_finite()
}

fn unit_vector(value: Vec3) -> Option<Vec3> {
	if !is_finite_vec(value) {
		return None;
	}
	let length_squared = value.dot(value);
	if !length_squared.is_finite() || length_squared <= VECTOR_EPSILON_SQUARED {
		return None;
	}
	Some(value * (1.0 / length_squared.sqrt()))
}

fn quad_determinant(u: Vec3, v: Vec3) -> Option<f32> {
	let uu = u.dot(u);
	let uv = u.dot(v);
	let vv = v.dot(v);
	let determinant = uu.mul_add(vv, -uv * uv);
	if !uu.is_finite()
		|| !vv.is_finite()
		|| !determinant.is_finite()
		|| uu <= VECTOR_EPSILON_SQUARED
		|| vv <= VECTOR_EPSILON_SQUARED
		|| determinant <= f32::EPSILON * uu * vv
	{
		None
	} else {
		Some(determinant)
	}
}

const fn invalid_bounds() -> Aabb {
	Aabb::new(vec3(1.0, 1.0, 1.0), vec3(-1.0, -1.0, -1.0))
}

fn sphere_uv(normal: Vec3) -> (f32, f32) {
	let u = 0.5 + normal.z.atan2(normal.x) / (2.0 * PI);
	let v = 0.5 - normal.y.clamp(-1.0, 1.0).asin() / PI;
	(u, v)
}

fn disk_basis(normal: Vec3) -> (Vec3, Vec3) {
	let helper = if normal.x.abs() < 0.9 {
		vec3(1.0, 0.0, 0.0)
	} else {
		vec3(0.0, 1.0, 0.0)
	};
	let tangent = normal.cross(helper).normalize();
	let bitangent = normal.cross(tangent);
	(tangent, bitangent)
}

#[cfg(test)]
mod tests {
	use super::*;

	const EPSILON: f32 = 1.0e-4;

	fn ray(origin: Vec3, direction: Vec3) -> Ray {
		Ray { origin, dir: direction.normalize() }
	}

	fn assert_near(actual: f32, expected: f32) {
		assert!((actual - expected).abs() < EPSILON, "expected {expected}, got {actual}");
	}

	fn material() -> Material {
		Material::diffuse(vec3(0.7, 0.7, 0.7))
	}

	fn object(center: Vec3, radius: f32) -> Object {
		Object::new(Sphere::new(center, radius), material())
	}

	#[test]
	fn aabb_slabs_handle_parallel_and_bounded_intervals() {
		let bounds = Aabb::new(vec3(-1.0, -1.0, -1.0), vec3(1.0, 1.0, 1.0));
		let interval = bounds
			.hit_interval(ray(vec3(0.0, 0.0, -3.0), vec3(0.0, 0.0, 1.0)), 0.001, 10.0)
			.unwrap();
		assert_near(interval.0, 2.0);
		assert_near(interval.1, 4.0);
		assert!(
			bounds
				.hit_interval(ray(vec3(2.0, 0.0, -3.0), vec3(0.0, 0.0, 1.0)), 0.001, 10.0)
				.is_none()
		);
		assert!(
			bounds
				.hit_interval(ray(vec3(0.0, 0.0, -3.0), vec3(0.0, 0.0, 1.0)), 0.001, 1.5)
				.is_none()
		);
	}

	#[test]
	fn sphere_intersects_from_outside_and_inside() {
		let sphere = Sphere::new(Vec3::ZERO, 1.0);
		let outside = sphere
			.intersect(ray(vec3(0.0, 0.0, -3.0), vec3(0.0, 0.0, 1.0)), 0.001, 100.0)
			.unwrap();
		assert_near(outside.t, 2.0);
		assert_near(outside.point.z, -1.0);
		let inside = sphere
			.intersect(ray(Vec3::ZERO, vec3(1.0, 0.0, 0.0)), 0.001, 100.0)
			.unwrap();
		assert_near(inside.t, 1.0);
		assert!(
			Sphere::new(Vec3::ZERO, 0.0)
				.intersect(ray(vec3(0.0, 0.0, -3.0), vec3(0.0, 0.0, 1.0)), 0.001, 100.0)
				.is_none()
		);
	}

	#[test]
	fn finite_quad_intersects_only_inside_edges() {
		let quad = Quad::new(vec3(-1.0, -1.0, 0.0), vec3(2.0, 0.0, 0.0), vec3(0.0, 2.0, 0.0));
		let hit = quad
			.intersect(ray(vec3(0.0, 0.0, -2.0), vec3(0.0, 0.0, 1.0)), 0.001, 10.0)
			.unwrap();
		assert_near(hit.t, 2.0);
		assert_near(hit.uv.0, 0.5);
		assert_near(hit.uv.1, 0.5);
		assert!(
			quad
				.intersect(ray(vec3(2.0, 0.0, -2.0), vec3(0.0, 0.0, 1.0)), 0.001, 10.0)
				.is_none()
		);
	}

	#[test]
	fn oriented_disk_uses_its_plane_and_radius() {
		let disk = Disk::new(Vec3::ZERO, vec3(0.0, 1.0, 1.0), 2.0);
		let normal = vec3(0.0, 1.0, 1.0).normalize();
		let hit = disk
			.intersect(ray(normal * -3.0, normal), 0.001, 10.0)
			.unwrap();
		assert_near(hit.t, 3.0);
		assert_near(hit.uv.0, 0.5);
		assert_near(hit.uv.1, 0.5);
		let tangent = disk_basis(normal).0;
		assert!(
			disk
				.intersect(ray(tangent * 2.1 - normal * 3.0, normal), 0.001, 10.0)
				.is_none()
		);
	}

	#[test]
	fn subnormal_disk_center_remains_intersectable() {
		let disk = Disk::new(Vec3::ZERO, vec3(0.0, 0.0, 1.0), f32::from_bits(1));
		let hit = disk
			.intersect(ray(vec3(0.0, 0.0, -1.0), vec3(0.0, 0.0, 1.0)), 0.001, 10.0)
			.unwrap();
		assert_eq!(hit.uv, (0.5, 0.5));
	}

	#[test]
	fn object_orients_both_faces_against_the_ray() {
		let object = object(Vec3::ZERO, 1.0);
		let front = object
			.hit(ray(vec3(0.0, 0.0, -3.0), vec3(0.0, 0.0, 1.0)), 0.001, 10.0)
			.unwrap();
		assert!(front.front_face);
		assert_near(front.normal.z, -1.0);
		let back = object
			.hit(ray(Vec3::ZERO, vec3(0.0, 0.0, 1.0)), 0.001, 10.0)
			.unwrap();
		assert!(!back.front_face);
		assert_near(back.geometric_normal.z, -1.0);
		assert!(back.normal.dot(vec3(0.0, 0.0, 1.0)) < 0.0);
	}

	#[test]
	fn bvh_selects_nearest_independent_of_insertion_order() {
		let trace = ray(vec3(0.0, 0.0, -10.0), vec3(0.0, 0.0, 1.0));
		let forward = Bvh::new(vec![object(vec3(0.0, 0.0, 2.0), 1.0), object(Vec3::ZERO, 1.0)]);
		let reverse = Bvh::new(vec![object(Vec3::ZERO, 1.0), object(vec3(0.0, 0.0, 2.0), 1.0)]);
		assert_near(forward.hit(trace, 0.001, f32::INFINITY).unwrap().t, 9.0);
		assert_near(reverse.hit(trace, 0.001, 100.0).unwrap().t, 9.0);
	}

	#[test]
	fn occlusion_respects_maximum_distance() {
		let bvh = Bvh::new(vec![object(Vec3::ZERO, 1.0)]);
		let trace = ray(vec3(0.0, 0.0, -5.0), vec3(0.0, 0.0, 1.0));
		assert!(!bvh.occluded(trace, 0.001, 3.99));
		assert!(bvh.occluded(trace, 0.001, 4.01));
	}

	#[test]
	fn bvh_matches_linear_reference_for_deterministic_rays() {
		let mut objects = Vec::new();
		for z in -2..=2 {
			for x in -3..=3 {
				objects.push(object(vec3(x as f32 * 1.3, (x * z) as f32 * 0.07, z as f32 * 1.4), 0.42));
			}
		}
		let bvh = Bvh::new(objects);
		for index in 0..257 {
			let x = ((index * 73 % 257) as f32 / 128.0 - 1.0) * 0.8;
			let y = ((index * 151 % 263) as f32 / 131.0 - 1.0) * 0.55;
			let trace = ray(vec3(0.0, 0.0, -9.0), vec3(x, y, 1.0));
			let accelerated = bvh.hit(trace, 0.001, 100.0).map(|hit| hit.t);
			let mut linear: Option<f32> = None;
			for object in bvh.objects() {
				let limit = linear.unwrap_or(100.0);
				if let Some(hit) = object.hit(trace, 0.001, limit) {
					linear = Some(hit.t);
				}
			}
			match (accelerated, linear) {
				(Some(a), Some(b)) => assert_near(a, b),
				(None, None) => {},
				pair => panic!("BVH and linear traversal disagree: {pair:?}"),
			}
		}
	}
}
