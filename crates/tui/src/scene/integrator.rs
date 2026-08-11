use std::time::Duration;

use smallvec::SmallVec;

use super::{Bvh, Camera, Hit, Material, Object, Ray, Trace, Vec3, vec3};

const PI: f32 = std::f32::consts::PI;
const INLINE_LIGHTS: usize = 4;

/// An analytic light evaluated without sampling noise.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Light {
	/// An isotropic point source. `intensity` is radiant intensity and therefore
	/// falls off with the inverse square of distance.
	Point {
		/// World-space source position.
		position:  Vec3,
		/// Unit-scale linear RGB source color.
		color:     Vec3,
		/// Non-negative radiant intensity multiplier.
		intensity: f32,
	},
	/// A source at infinity. `direction` is the direction its rays travel.
	Directional {
		/// Unit direction from the light toward the scene.
		direction: Vec3,
		/// Unit-scale linear RGB source color.
		color:     Vec3,
		/// Non-negative incident irradiance multiplier.
		intensity: f32,
	},
}

impl Light {
	/// Creates an inverse-square point light in linear color space.
	pub const fn point(position: Vec3, color: Vec3, intensity: f32) -> Self {
		Self::Point { position, color, intensity }
	}

	/// Creates a directional light whose rays travel along `direction`.
	pub fn directional(direction: Vec3, color: Vec3, intensity: f32) -> Self {
		Self::Directional { direction: direction.normalize(), color, intensity }
	}
}

/// Geometry, analytic lights, and the linear radiance seen by rays that miss.
///
/// Four lights stay inline; larger scenes spill only while the world is built.
pub struct World {
	bvh:         Bvh,
	environment: Vec3,
	lights:      SmallVec<Light, INLINE_LIGHTS>,
}

impl World {
	/// Builds a world from owned objects with no lights and a black environment.
	pub fn new(objects: Vec<Object>) -> Self {
		Self::from_bvh(Bvh::new(objects))
	}

	/// Creates a world from an already-built acceleration structure.
	pub const fn from_bvh(bvh: Bvh) -> Self {
		Self { bvh, environment: Vec3::ZERO, lights: SmallVec::new() }
	}

	/// Adds an analytic light and returns this world.
	///
	/// The first four lights remain inline; overflow storage is allocated only
	/// while constructing the world, never during tracing.
	pub fn with_light(mut self, light: Light) -> Self {
		self.lights.push(light);
		self
	}

	/// Sets the constant linear environment radiance and returns this world.
	pub const fn with_environment(mut self, environment: Vec3) -> Self {
		self.environment = positive_color(environment);
		self
	}

	fn lights(&self) -> impl ExactSizeIterator<Item = &Light> {
		self.lights.iter()
	}
}

/// Path-transport limits and deterministic sampling configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Integrator {
	/// Maximum number of secondary surface bounces after the primary hit.
	pub max_bounces:            u8,
	/// Independent paths averaged for each primary ray; zero is treated as one.
	pub samples_per_ray:        u16,
	/// Secondary-bounce index at which Russian roulette begins.
	pub russian_roulette_start: u8,
	/// Positive distance used for ray origins and minimum intersections.
	pub ray_epsilon:            f32,
	/// Furthest distance considered by primary, secondary, and shadow rays.
	pub max_distance:           f32,
	/// User-controlled salt mixed with every primary ray's exact float bits.
	pub seed:                   u64,
}

impl Integrator {
	/// Returns this configuration with unsafe or non-finite limits repaired.
	pub fn sanitized(self) -> Self {
		let ray_epsilon = if self.ray_epsilon.is_finite() {
			self.ray_epsilon.clamp(1.0e-6, 0.1)
		} else {
			1.0e-4
		};
		Self {
			max_bounces: self.max_bounces.min(64),
			samples_per_ray: self.samples_per_ray.max(1),
			russian_roulette_start: self.russian_roulette_start.min(64),
			ray_epsilon,
			max_distance: if self.max_distance.is_finite() {
				self.max_distance.max(ray_epsilon * 2.0)
			} else {
				1.0e30
			},
			seed: self.seed,
		}
	}
}

impl Default for Integrator {
	fn default() -> Self {
		Self {
			max_bounces:            6,
			samples_per_ray:        1,
			russian_roulette_start: 3,
			ray_epsilon:            1.0e-4,
			max_distance:           1.0e30,
			seed:                   0,
		}
	}
}

/// Deterministic CPU path tracer over an owned [`World`].
pub struct PathTracer {
	world:      World,
	integrator: Integrator,
	camera:     Camera,
}

impl PathTracer {
	/// Creates a tracer with the default camera and [`Integrator`]
	/// configuration.
	pub fn new(world: World) -> Self {
		Self { world, integrator: Integrator::default(), camera: Camera::default() }
	}

	/// Selects the camera returned by [`Trace::advance`].
	pub const fn with_camera(mut self, camera: Camera) -> Self {
		self.camera = camera;
		self
	}

	/// Selects transport limits and deterministic sampling configuration.
	pub fn with_integrator(mut self, integrator: Integrator) -> Self {
		self.integrator = integrator.sanitized();
		self
	}
}

impl Trace for PathTracer {
	fn advance(&mut self, _now: Duration) -> Camera {
		self.camera
	}

	fn shade(&self, ray: Ray) -> (Vec3, f32) {
		let config = self.integrator.sanitized();
		let Some(primary_hit) = self
			.world
			.bvh
			.hit(ray, config.ray_epsilon, config.max_distance)
		else {
			return (display_color(self.world.environment), 0.0);
		};

		let mut radiance = Vec3::ZERO;
		for sample in 0..config.samples_per_ray {
			let mut rng = Rng::for_ray(ray, config.seed, sample);
			radiance += self.trace_path(ray, primary_hit, config, &mut rng);
		}
		(display_color(radiance * (1.0 / f32::from(config.samples_per_ray))), 1.0)
	}
}

impl PathTracer {
	fn trace_path<'w>(
		&'w self,
		mut ray: Ray,
		mut hit: Hit<'w>,
		config: Integrator,
		rng: &mut Rng,
	) -> Vec3 {
		let mut radiance = Vec3::ZERO;
		let mut throughput = vec3(1.0, 1.0, 1.0);

		for depth in 0..=config.max_bounces {
			let material = hit.material.sanitized();
			radiance = bounded_color(radiance + throughput * material.emission);
			radiance = bounded_color(
				radiance + throughput * self.direct_lighting(&hit, ray, material, config),
			);

			if depth == config.max_bounces || is_non_reflecting_emitter(material) {
				break;
			}
			let Some(scatter) = sample_surface(ray, &hit, material, rng) else {
				break;
			};
			throughput = bounded_color(throughput * scatter.weight);
			if throughput.max_component() <= 0.0 {
				break;
			}

			let secondary = depth.saturating_add(1);
			if secondary >= config.russian_roulette_start {
				let survive = throughput.max_component().clamp(0.05, 0.95);
				if rng.next_f32() >= survive {
					break;
				}
				throughput *= 1.0 / survive;
			}

			ray = Ray {
				origin: offset_origin(
					hit.point,
					hit.geometric_normal,
					scatter.direction,
					config.ray_epsilon,
				),
				dir:    scatter.direction.normalize(),
			};
			let Some(next_hit) = self
				.world
				.bvh
				.hit(ray, config.ray_epsilon, config.max_distance)
			else {
				radiance =
					bounded_color(radiance + throughput * positive_color(self.world.environment));
				break;
			};
			hit = next_hit;
		}
		bounded_color(radiance)
	}

	fn direct_lighting(
		&self,
		hit: &Hit<'_>,
		ray: Ray,
		material: Material,
		config: Integrator,
	) -> Vec3 {
		if is_non_reflecting_emitter(material) {
			return Vec3::ZERO;
		}
		let mut result = Vec3::ZERO;
		let view = ray.dir * -1.0;
		for light in self.world.lights() {
			let Some(incident) = incident_light(*light, hit.point, config.max_distance) else {
				continue;
			};
			let n_dot_l = hit.normal.dot(incident.direction).max(0.0);
			if n_dot_l <= 0.0 {
				continue;
			}
			let shadow = Ray {
				origin: offset_origin(
					hit.point,
					hit.geometric_normal,
					incident.direction,
					config.ray_epsilon,
				),
				dir:    incident.direction,
			};
			let shadow_max = (incident.distance - config.ray_epsilon).min(config.max_distance);
			if shadow_max > config.ray_epsilon
				&& self
					.world
					.bvh
					.occluded(shadow, config.ray_epsilon, shadow_max)
			{
				continue;
			}
			let brdf = opaque_brdf(material, hit.normal, view, incident.direction);
			result = bounded_color(result + brdf * incident.radiance * n_dot_l);
		}
		bounded_color(result)
	}
}

#[derive(Clone, Copy)]
struct IncidentLight {
	direction: Vec3,
	distance:  f32,
	radiance:  Vec3,
}

fn incident_light(light: Light, point: Vec3, max_distance: f32) -> Option<IncidentLight> {
	match light {
		Light::Point { position, color, intensity } => {
			let offset = position - point;
			let distance_squared = offset.dot(offset);
			if !distance_squared.is_finite() || distance_squared <= 1.0e-12 {
				return None;
			}
			let distance = distance_squared.sqrt();
			if distance > max_distance {
				return None;
			}
			Some(IncidentLight {
				direction: offset * (1.0 / distance),
				distance,
				radiance: positive_color(color) * (positive(intensity) / distance_squared),
			})
		},
		Light::Directional { direction, color, intensity } => {
			let direction = (direction * -1.0).normalize();
			if direction.dot(direction) < 0.5 {
				return None;
			}
			Some(IncidentLight {
				direction,
				distance: max_distance,
				radiance: positive_color(color) * positive(intensity),
			})
		},
	}
}

fn opaque_brdf(material: Material, normal: Vec3, view: Vec3, light: Vec3) -> Vec3 {
	let n_dot_v = normal.dot(view).max(0.0);
	let n_dot_l = normal.dot(light).max(0.0);
	if n_dot_v <= 0.0 || n_dot_l <= 0.0 {
		return Vec3::ZERO;
	}
	let half = (view + light).normalize();
	let n_dot_h = normal.dot(half).max(0.0);
	let v_dot_h = view.dot(half).max(0.0);
	let f0 = material_f0(material);
	let fresnel = fresnel_schlick(f0, v_dot_h);
	let distribution = ggx_distribution(n_dot_h, material.roughness);
	let geometry = smith_geometry(n_dot_v, n_dot_l, material.roughness);
	let specular = fresnel * (distribution * geometry / (4.0 * n_dot_v * n_dot_l).max(1.0e-8));
	let diffuse_weight = (1.0 - material.metallic) * (1.0 - material.transmission);
	let diffuse = material.base_color * (Vec3::ONE - fresnel) * (diffuse_weight / PI);
	diffuse + specular
}

fn material_f0(material: Material) -> Vec3 {
	let dielectric = ((material.ior - 1.0) / (material.ior + 1.0)).powi(2);
	vec3(dielectric, dielectric, dielectric).lerp(material.base_color, material.metallic)
}

fn fresnel_schlick(f0: Vec3, cosine: f32) -> Vec3 {
	f0 + (Vec3::ONE - f0) * (1.0 - cosine.clamp(0.0, 1.0)).powi(5)
}

fn fresnel_dielectric(cosine: f32, ior: f32) -> f32 {
	let f0 = ((ior - 1.0) / (ior + 1.0)).powi(2);
	f0 + (1.0 - f0) * (1.0 - cosine.clamp(0.0, 1.0)).powi(5)
}

fn ggx_distribution(n_dot_h: f32, roughness: f32) -> f32 {
	let alpha = roughness * roughness;
	let alpha_squared = alpha * alpha;
	let n_squared = n_dot_h.clamp(0.0, 1.0).powi(2);
	let denominator = n_squared.mul_add(alpha_squared, 1.0 - n_squared);
	alpha_squared / (PI * denominator * denominator).max(f32::MIN_POSITIVE)
}

fn smith_geometry(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
	let alpha = roughness * roughness;
	let alpha_squared = alpha * alpha;
	let g1 = |n_dot_x: f32| {
		2.0 * n_dot_x
			/ (n_dot_x + (alpha_squared + (1.0 - alpha_squared) * n_dot_x * n_dot_x).sqrt())
				.max(1.0e-8)
	};
	g1(n_dot_v) * g1(n_dot_l)
}

#[derive(Clone, Copy)]
struct Scatter {
	direction: Vec3,
	weight:    Vec3,
}

fn sample_surface(ray: Ray, hit: &Hit<'_>, material: Material, rng: &mut Rng) -> Option<Scatter> {
	let transmission = material.transmission;
	if transmission > 0.0 && rng.next_f32() < transmission {
		return Some(sample_dielectric(ray, hit, material, rng));
	}

	let mut opaque = material;
	opaque.transmission = 0.0;
	if opaque.metallic >= 0.999 && opaque.roughness <= 0.021 {
		return Some(Scatter {
			direction: ray.dir.reflect(hit.normal).normalize(),
			weight:    opaque.base_color,
		});
	}
	sample_opaque(ray, hit.normal, opaque, rng)
}

fn sample_dielectric(ray: Ray, hit: &Hit<'_>, material: Material, rng: &mut Rng) -> Scatter {
	let incident_cosine = (-ray.dir.dot(hit.normal)).clamp(0.0, 1.0);
	let eta = if hit.front_face {
		1.0 / material.ior
	} else {
		material.ior
	};
	let reflected = ray.dir.reflect(hit.normal).normalize();
	let Some(refracted) = ray.dir.refract(hit.normal, eta) else {
		return Scatter { direction: reflected, weight: vec3(1.0, 1.0, 1.0) };
	};
	if rng.next_f32() < fresnel_dielectric(incident_cosine, material.ior) {
		Scatter { direction: reflected, weight: vec3(1.0, 1.0, 1.0) }
	} else {
		Scatter { direction: refracted, weight: material.base_color * (eta * eta) }
	}
}

fn sample_opaque(ray: Ray, normal: Vec3, material: Material, rng: &mut Rng) -> Option<Scatter> {
	let view = ray.dir * -1.0;
	let f0 = material_f0(material);
	let specular_probability = luminance(f0).clamp(0.1, 0.9);
	let choose_specular = rng.next_f32() < specular_probability;
	let direction = if choose_specular {
		let half = sample_ggx_half(normal, material.roughness, rng);
		let candidate = half * (2.0 * view.dot(half)) - view;
		if normal.dot(candidate) <= 0.0 {
			return None;
		}
		candidate.normalize()
	} else {
		sample_cosine_hemisphere(normal, rng)
	};

	let half = (view + direction).normalize();
	let n_dot_l = normal.dot(direction).max(0.0);
	let n_dot_h = normal.dot(half).max(0.0);
	let v_dot_h = view.dot(half).abs().max(1.0e-8);
	let diffuse_pdf = n_dot_l / PI;
	let specular_pdf = ggx_distribution(n_dot_h, material.roughness) * n_dot_h / (4.0 * v_dot_h);
	let pdf = (1.0 - specular_probability) * diffuse_pdf + specular_probability * specular_pdf;
	if !pdf.is_finite() || pdf <= 1.0e-10 {
		return None;
	}
	let brdf = opaque_brdf(material, normal, view, direction);
	Some(Scatter { direction, weight: bounded_color(brdf * (n_dot_l / pdf)) })
}

fn sample_cosine_hemisphere(normal: Vec3, rng: &mut Rng) -> Vec3 {
	let radius = rng.next_f32().sqrt();
	let phi = 2.0 * PI * rng.next_f32();
	let local_x = radius * phi.cos();
	let local_y = radius * phi.sin();
	let local_z = (1.0 - radius * radius).sqrt();
	let (tangent, bitangent) = basis(normal);
	(tangent * local_x + bitangent * local_y + normal * local_z).normalize()
}

fn sample_ggx_half(normal: Vec3, roughness: f32, rng: &mut Rng) -> Vec3 {
	let alpha = roughness * roughness;
	let alpha_squared = alpha * alpha;
	let u = rng.next_f32().min(1.0 - f32::EPSILON);
	let one_minus_u = 1.0 - u;
	let cos_theta = (one_minus_u / alpha_squared.mul_add(u, one_minus_u)).sqrt();
	let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
	let phi = 2.0 * PI * rng.next_f32();
	let (tangent, bitangent) = basis(normal);
	(tangent * (sin_theta * phi.cos()) + bitangent * (sin_theta * phi.sin()) + normal * cos_theta)
		.normalize()
}

fn basis(normal: Vec3) -> (Vec3, Vec3) {
	let helper = if normal.z.abs() < 0.999 {
		vec3(0.0, 0.0, 1.0)
	} else {
		vec3(0.0, 1.0, 0.0)
	};
	let tangent = helper.cross(normal).normalize();
	(tangent, normal.cross(tangent))
}

fn offset_origin(point: Vec3, normal: Vec3, direction: Vec3, epsilon: f32) -> Vec3 {
	let side = if normal.dot(direction) >= 0.0 {
		1.0
	} else {
		-1.0
	};
	point + normal * (epsilon * side)
}

fn is_non_reflecting_emitter(material: Material) -> bool {
	material.emission.max_component() > 0.0 && material.base_color.max_component() <= 0.0
}

fn luminance(color: Vec3) -> f32 {
	0.2126 * color.x + 0.7152 * color.y + 0.0722 * color.z
}

const fn positive(value: f32) -> f32 {
	if value.is_finite() {
		value.max(0.0)
	} else {
		0.0
	}
}

const fn positive_color(color: Vec3) -> Vec3 {
	vec3(positive(color.x), positive(color.y), positive(color.z))
}

fn bounded_color(color: Vec3) -> Vec3 {
	let channel = |value: f32| {
		if value.is_finite() {
			value.clamp(0.0, 1.0e6)
		} else {
			0.0
		}
	};
	vec3(channel(color.x), channel(color.y), channel(color.z))
}

fn display_color(color: Vec3) -> Vec3 {
	bounded_color(color).clamp01()
}

struct Rng {
	state: u64,
}

impl Rng {
	fn for_ray(ray: Ray, seed: u64, sample: u16) -> Self {
		let mut state = seed ^ 0xa076_1d64_78bd_642f;
		for bits in [
			ray.origin.x.to_bits(),
			ray.origin.y.to_bits(),
			ray.origin.z.to_bits(),
			ray.dir.x.to_bits(),
			ray.dir.y.to_bits(),
			ray.dir.z.to_bits(),
		] {
			state = mix64(state ^ u64::from(bits));
		}
		Self { state: mix64(state ^ u64::from(sample)) }
	}

	fn next_f32(&mut self) -> f32 {
		self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
		let value = mix64(self.state);
		((value >> 40) as u32) as f32 * (1.0 / 16_777_216.0)
	}
}

const fn mix64(mut value: u64) -> u64 {
	value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
	value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
	value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::scene::{Object, Primitive, Sphere};

	fn sphere(center: Vec3, radius: f32, material: Material) -> Object {
		Object::new(Primitive::Sphere(Sphere::new(center, radius)), material)
	}

	fn primary_ray() -> Ray {
		Ray { origin: vec3(0.0, 0.0, 3.0), dir: vec3(0.0, 0.0, -1.0) }
	}

	fn direct_tracer(objects: Vec<Object>) -> PathTracer {
		let world = World::new(objects).with_light(Light::point(
			vec3(0.0, 2.0, 2.0),
			vec3(1.0, 1.0, 1.0),
			30.0,
		));
		PathTracer::new(world).with_integrator(Integrator {
			max_bounces: 0,
			samples_per_ray: 1,
			..Integrator::default()
		})
	}

	#[test]
	fn direct_light_respects_nearest_hit_shadows() {
		let target = || sphere(Vec3::ZERO, 0.5, Material::diffuse(vec3(0.8, 0.8, 0.8)));
		let clear = direct_tracer(vec![target()]).shade(primary_ray()).0;
		let blocker = sphere(vec3(0.0, 1.0, 1.25), 0.35, Material::diffuse(vec3(0.2, 0.2, 0.2)));
		let shadowed = direct_tracer(vec![target(), blocker])
			.shade(primary_ray())
			.0;
		assert!(luminance(clear) > luminance(shadowed) + 0.05);
	}

	#[test]
	fn minimum_trace_distance_keeps_directional_light_visible() {
		let epsilon = 1.0e-4;
		let world = World::new(vec![sphere(Vec3::ZERO, 1.0, Material::diffuse(Vec3::ONE))])
			.with_light(Light::directional(vec3(0.0, 0.0, -1.0), Vec3::ONE, 1.0));
		let tracer = PathTracer::new(world).with_integrator(Integrator {
			max_bounces: 0,
			ray_epsilon: epsilon,
			max_distance: epsilon * 2.0,
			..Integrator::default()
		});
		let ray = Ray { origin: vec3(0.0, 0.0, 1.000_15), dir: vec3(0.0, 0.0, -1.0) };
		assert!(luminance(tracer.shade(ray).0) > 0.0);
	}

	#[test]
	fn schlick_fresnel_preserves_normal_and_grazing_limits() {
		let f0 = vec3(0.04, 0.25, 0.81);
		assert_eq!(fresnel_schlick(f0, 1.0), f0);
		let grazing = fresnel_schlick(f0, 0.0);
		assert!((grazing.x - 1.0).abs() < 1.0e-6);
		assert!((grazing.y - 1.0).abs() < 1.0e-6);
		assert!((grazing.z - 1.0).abs() < 1.0e-6);
	}

	#[test]
	fn normal_incidence_brdf_conserves_fresnel_split() {
		let base = vec3(0.8, 0.6, 0.4);
		let material = Material::diffuse(base);
		let normal = vec3(0.0, 0.0, 1.0);
		let actual = opaque_brdf(material, normal, normal, normal);
		let f0 = Vec3::splat(0.04);
		let expected = base * (Vec3::ONE - f0) * (1.0 / PI) + f0 * (1.0 / (4.0 * PI));
		assert!((actual.x - expected.x).abs() < 1.0e-6);
		assert!((actual.y - expected.y).abs() < 1.0e-6);
		assert!((actual.z - expected.z).abs() < 1.0e-6);
	}

	#[test]
	fn ggx_distribution_normalizes_and_preserves_smooth_peaks() {
		let roughness = 0.2;
		let steps = 100_000;
		let mut integral = 0.0_f64;
		for index in 0..steps {
			let cosine = (index as f32 + 0.5) / steps as f32;
			let density = ggx_distribution(cosine, roughness);
			integral += f64::from(density * cosine) * (2.0 * f64::from(PI) / f64::from(steps));
		}
		assert!((integral - 1.0).abs() < 1.0e-4, "GGX projected-area integral: {integral}");

		let smooth_roughness = 0.02_f32;
		let peak = ggx_distribution(1.0, smooth_roughness);
		let expected = 1.0 / (PI * smooth_roughness.powi(4));
		assert!(((peak - expected) / expected).abs() < 1.0e-5);
	}

	#[test]
	fn mirror_reflection_collects_environment_radiance() {
		let world =
			World::new(vec![sphere(Vec3::ZERO, 1.0, Material::metal(vec3(0.9, 0.8, 0.7), 0.0))])
				.with_environment(vec3(0.3, 0.4, 0.5));
		let tracer = PathTracer::new(world).with_integrator(Integrator {
			max_bounces: 1,
			russian_roulette_start: 2,
			..Integrator::default()
		});
		let (color, coverage) = tracer.shade(primary_ray());
		assert_eq!(coverage, 1.0);
		assert!(color.x > 0.2 && color.y > 0.2 && color.z > 0.2);
	}

	#[test]
	fn dielectric_refraction_and_total_internal_reflection_are_distinct() {
		let entering = vec3(0.0, 0.0, -1.0).refract(vec3(0.0, 0.0, 1.0), 1.0 / 1.5);
		assert!(entering.is_some());
		let grazing_inside = vec3(0.9, 0.0, 0.435_889_9).normalize();
		assert!(grazing_inside.refract(vec3(0.0, 0.0, -1.0), 1.5).is_none());
	}

	#[test]
	fn dielectric_path_refracts_to_emissive_geometry_behind_it() {
		let glass = sphere(Vec3::ZERO, 1.0, Material::dielectric(Vec3::ONE, 1.5));
		let emitter = sphere(vec3(0.0, 0.0, -3.0), 0.75, Material::emissive(Vec3::ONE, 1.0));
		let tracer = PathTracer::new(World::new(vec![glass, emitter])).with_integrator(Integrator {
			max_bounces: 4,
			samples_per_ray: 32,
			russian_roulette_start: 5,
			seed: 11,
			..Integrator::default()
		});
		let (color, coverage) = tracer.shade(primary_ray());
		assert_eq!(coverage, 1.0);
		assert!(luminance(color) > 0.25, "glass must transmit the emitter: {color:?}");
	}

	#[test]
	fn indirect_path_collects_emissive_geometry() {
		let diffuse = sphere(Vec3::ZERO, 1.0, Material::diffuse(vec3(0.8, 0.8, 0.8)));
		let emitter = sphere(Vec3::ZERO, 8.0, Material::emissive(vec3(0.7, 0.5, 0.3), 1.0));
		let tracer =
			PathTracer::new(World::new(vec![diffuse, emitter])).with_integrator(Integrator {
				max_bounces: 1,
				samples_per_ray: 4,
				russian_roulette_start: 2,
				seed: 7,
				..Integrator::default()
			});
		let (color, _) = tracer.shade(primary_ray());
		assert!(luminance(color) > 0.05);
	}

	#[test]
	fn seeded_sampling_is_bitwise_deterministic() {
		let tracer = PathTracer::new(
			World::new(vec![sphere(Vec3::ZERO, 1.0, Material::diffuse(vec3(0.8, 0.7, 0.6)))])
				.with_environment(vec3(0.2, 0.3, 0.4)),
		)
		.with_integrator(Integrator {
			max_bounces: 3,
			samples_per_ray: 8,
			seed: 42,
			..Integrator::default()
		});
		assert_eq!(tracer.shade(primary_ray()), tracer.shade(primary_ray()));
	}

	#[test]
	fn output_stays_finite_and_energy_bounded() {
		let world = World::new(vec![sphere(Vec3::ZERO, 1.0, Material::diffuse(vec3(1.0, 1.0, 1.0)))])
			.with_light(Light::point(
				vec3(0.0, 0.0, 2.0),
				vec3(f32::INFINITY, 1.0e30, -1.0),
				f32::INFINITY,
			));
		let (color, _) = PathTracer::new(world).shade(primary_ray());
		assert!(color.x.is_finite() && color.y.is_finite() && color.z.is_finite());
		assert!(color.max_component() <= 1.0);
	}

	#[test]
	fn primary_miss_has_zero_coverage_even_with_environment() {
		let tracer = PathTracer::new(World::new(Vec::new()).with_environment(vec3(0.2, 0.3, 0.4)));
		let (color, coverage) = tracer.shade(primary_ray());
		assert_eq!(color, vec3(0.2, 0.3, 0.4));
		assert_eq!(coverage, 0.0);
	}
}
