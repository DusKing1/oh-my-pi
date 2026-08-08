use candle_core::Device;

use crate::{Error, Result};

/// Preferred execution device for a model.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum DevicePreference {
	/// Select the fastest compiled backend and fall back to CPU.
	#[default]
	Auto,
	/// Require CPU execution.
	Cpu,
	/// Require the platform-native GPU backend: Metal on macOS or CUDA
	/// elsewhere.
	Gpu,
	/// Require Apple Metal.
	Metal,
	/// Require NVIDIA CUDA.
	Cuda,
}

/// Backend selected for a loaded model.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Accelerator {
	/// Portable CPU execution.
	Cpu,
	/// Apple Metal execution through Candle.
	Metal,
	/// NVIDIA CUDA execution.
	Cuda,
	/// Apple Core ML execution through ONNX Runtime.
	CoreMl,
}

pub fn candle_device(preference: DevicePreference) -> Result<(Device, Accelerator)> {
	match preference {
		DevicePreference::Cpu => Ok((Device::Cpu, Accelerator::Cpu)),
		DevicePreference::Metal => metal_device(),
		DevicePreference::Cuda => cuda_device(),
		DevicePreference::Gpu => native_gpu(),
		DevicePreference::Auto => native_gpu().or_else(|_| Ok((Device::Cpu, Accelerator::Cpu))),
	}
}

#[cfg(target_os = "macos")]
fn metal_device() -> Result<(Device, Accelerator)> {
	Device::new_metal(0)
		.map(|device| (device, Accelerator::Metal))
		.map_err(|error| Error::backend("metal", error))
}

#[cfg(not(target_os = "macos"))]
fn metal_device() -> Result<(Device, Accelerator)> {
	Err(Error::unavailable("this target was not compiled with Metal"))
}

#[cfg(feature = "cuda")]
fn cuda_device() -> Result<(Device, Accelerator)> {
	Device::new_cuda(0)
		.map(|device| (device, Accelerator::Cuda))
		.map_err(|error| Error::backend("cuda", error))
}

#[cfg(not(feature = "cuda"))]
fn cuda_device() -> Result<(Device, Accelerator)> {
	Err(Error::unavailable("enable the omp-llm-local `cuda` feature"))
}

#[cfg(target_os = "macos")]
fn native_gpu() -> Result<(Device, Accelerator)> {
	metal_device()
}

#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
fn native_gpu() -> Result<(Device, Accelerator)> {
	cuda_device()
}

#[cfg(all(not(target_os = "macos"), not(feature = "cuda")))]
fn native_gpu() -> Result<(Device, Accelerator)> {
	Err(Error::unavailable("no GPU backend is enabled"))
}

pub fn whisper_accelerator(preference: DevicePreference) -> Result<Accelerator> {
	match preference {
		DevicePreference::Cpu => Ok(Accelerator::Cpu),
		DevicePreference::Metal => Ok(require_metal()),
		DevicePreference::Cuda => require_cuda(),
		DevicePreference::Gpu => native_whisper_gpu(),
		DevicePreference::Auto => native_whisper_gpu().or(Ok(Accelerator::Cpu)),
	}
}

#[cfg(target_os = "macos")]
const fn require_metal() -> Accelerator {
	Accelerator::Metal
}

#[cfg(not(target_os = "macos"))]
fn require_metal() -> Result<Accelerator> {
	Err(Error::unavailable("this target was not compiled with Metal"))
}

#[cfg(feature = "cuda")]
fn require_cuda() -> Result<Accelerator> {
	Ok(Accelerator::Cuda)
}

#[cfg(not(feature = "cuda"))]
fn require_cuda() -> Result<Accelerator> {
	Err(Error::unavailable("enable the omp-llm-local `cuda` feature"))
}

#[cfg(target_os = "macos")]
const fn native_whisper_gpu() -> Result<Accelerator> {
	Ok(require_metal())
}

#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
fn native_whisper_gpu() -> Result<Accelerator> {
	require_cuda()
}

#[cfg(all(not(target_os = "macos"), not(feature = "cuda")))]
fn native_whisper_gpu() -> Result<Accelerator> {
	Err(Error::unavailable("no GPU backend is enabled"))
}
