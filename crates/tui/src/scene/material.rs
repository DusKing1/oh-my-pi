use super::{Vec3, vec3};

/// Compact physically based surface parameters in linear color space.
///
/// Call [`sanitized`](Self::sanitized) after changing public fields directly.
/// The convenience constructors always return sanitized values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Material {
	/// Diffuse albedo, metallic reflectance, or dielectric transmission tint.
	pub base_color:   Vec3,
	/// Surface-emitted radiance. Values above one are valid for bright emitters.
	pub emission:     Vec3,
	/// Microfacet roughness for non-transmissive surfaces, in `0.02..=1.0`.
	pub roughness:    f32,
	/// Fraction of the opaque response that is metallic, in `0..=1`.
	pub metallic:     f32,
	/// Fraction of energy assigned to dielectric transmission, in `0..=1`.
	pub transmission: f32,
	/// Index of refraction of the material interior, in `1.0001..=3.0`.
	pub ior:          f32,
}

impl Material {
	/// Creates an opaque Lambertian material with `color` as its linear albedo.
	pub fn diffuse(color: Vec3) -> Self {
		Self {
			base_color:   color,
			emission:     Vec3::ZERO,
			roughness:    1.0,
			metallic:     0.0,
			transmission: 0.0,
			ior:          1.5,
		}
		.sanitized()
	}

	/// Creates an opaque conductor with linear normal-incidence reflectance
	/// `color` and the requested perceptual `roughness`.
	pub fn metal(color: Vec3, roughness: f32) -> Self {
		Self {
			base_color: color,
			emission: Vec3::ZERO,
			roughness,
			metallic: 1.0,
			transmission: 0.0,
			ior: 1.5,
		}
		.sanitized()
	}

	/// Creates an ideal specular dielectric tinted by linear `color`.
	///
	/// `ior` is the material's index of refraction relative to vacuum; common
	/// glass is approximately `1.5`.
	pub fn dielectric(color: Vec3, ior: f32) -> Self {
		Self {
			base_color: color,
			emission: Vec3::ZERO,
			roughness: 0.02,
			metallic: 0.0,
			transmission: 1.0,
			ior,
		}
		.sanitized()
	}

	/// Creates a non-reflecting area emitter of linear `color`.
	///
	/// `strength` scales the emitted radiance and may be greater than one.
	pub fn emissive(color: Vec3, strength: f32) -> Self {
		Self {
			base_color:   Vec3::ZERO,
			emission:     color * finite_or(strength, 0.0).max(0.0),
			roughness:    1.0,
			metallic:     0.0,
			transmission: 0.0,
			ior:          1.5,
		}
		.sanitized()
	}

	/// Returns a finite, energy-bounded copy suitable for transport.
	///
	/// Reflectance parameters are clamped to unit range. Transmission is
	/// reduced by the non-metallic fraction so a surface cannot spend the same
	/// energy on both a conductor and a dielectric lobe. Emission remains HDR,
	/// but negative and non-finite channels become zero.
	pub fn sanitized(self) -> Self {
		let metallic = finite_unit(self.metallic);
		Self {
			base_color: finite_unit_color(self.base_color),
			emission: finite_positive_color(self.emission),
			roughness: finite_or(self.roughness, 1.0).clamp(0.02, 1.0),
			metallic,
			transmission: finite_unit(self.transmission) * (1.0 - metallic),
			ior: finite_or(self.ior, 1.5).clamp(1.0001, 3.0),
		}
	}
}

impl Default for Material {
	fn default() -> Self {
		Self::diffuse(vec3(0.8, 0.8, 0.8))
	}
}

const fn finite_or(value: f32, fallback: f32) -> f32 {
	if value.is_finite() { value } else { fallback }
}

const fn finite_unit(value: f32) -> f32 {
	finite_or(value, 0.0).clamp(0.0, 1.0)
}

const fn finite_unit_color(color: Vec3) -> Vec3 {
	vec3(finite_unit(color.x), finite_unit(color.y), finite_unit(color.z))
}

fn finite_positive_color(color: Vec3) -> Vec3 {
	let channel = |value: f32| {
		if value.is_finite() {
			value.max(0.0)
		} else {
			0.0
		}
	};
	vec3(channel(color.x), channel(color.y), channel(color.z))
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn sanitization_bounds_scattering_energy_and_keeps_hdr_emission() {
		let material = Material {
			base_color:   vec3(-1.0, 2.0, f32::NAN),
			emission:     vec3(4.0, -2.0, f32::INFINITY),
			roughness:    0.0,
			metallic:     0.75,
			transmission: 1.0,
			ior:          99.0,
		}
		.sanitized();
		assert_eq!(material.base_color, vec3(0.0, 1.0, 0.0));
		assert_eq!(material.emission, vec3(4.0, 0.0, 0.0));
		assert_eq!(material.roughness, 0.02);
		assert_eq!(material.transmission, 0.25);
		assert_eq!(material.ior, 3.0);
	}
}
