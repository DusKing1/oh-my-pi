use std::path::Path;

use tokio::io::AsyncWriteExt;

use crate::{Error, Result};

/// Interleaved normalized `f32` PCM audio.
#[derive(Clone, Debug, PartialEq)]
pub struct Audio {
	samples:     Vec<f32>,
	sample_rate: u32,
	channels:    u16,
}

impl Audio {
	/// Creates audio after validating its rate, channel count, and frame
	/// alignment.
	pub fn new(samples: Vec<f32>, sample_rate: u32, channels: u16) -> Result<Self> {
		if sample_rate == 0 {
			return Err(Error::invalid("sample rate must be non-zero"));
		}
		if channels == 0 {
			return Err(Error::invalid("channel count must be non-zero"));
		}
		if !samples.len().is_multiple_of(usize::from(channels)) {
			return Err(Error::invalid("interleaved samples do not contain complete frames"));
		}
		Ok(Self { samples, sample_rate, channels })
	}

	pub(crate) const fn mono(samples: Vec<f32>, sample_rate: u32) -> Self {
		Self { samples, sample_rate, channels: 1 }
	}

	/// Interleaved samples in the range normally expected by audio devices,
	/// `-1.0..=1.0`.
	pub fn samples(&self) -> &[f32] {
		&self.samples
	}

	/// Consumes the buffer and returns its interleaved samples.
	pub fn into_samples(self) -> Vec<f32> {
		self.samples
	}

	/// Samples per second in each channel.
	pub const fn sample_rate(&self) -> u32 {
		self.sample_rate
	}

	/// Number of interleaved channels.
	pub const fn channels(&self) -> u16 {
		self.channels
	}

	/// Playback duration in seconds.
	pub fn duration(&self) -> std::time::Duration {
		let frames = self.samples.len() / usize::from(self.channels);
		std::time::Duration::from_secs_f64(frames as f64 / f64::from(self.sample_rate))
	}

	/// Writes IEEE-float WAV audio without blocking the async executor.
	pub async fn write_wav(&self, path: impl AsRef<Path>) -> Result<()> {
		let data_size = self
			.samples
			.len()
			.checked_mul(size_of::<f32>())
			.and_then(|size| u32::try_from(size).ok())
			.ok_or_else(|| Error::invalid("audio is too large for a RIFF WAV file"))?;
		let riff_size = data_size
			.checked_add(36)
			.ok_or_else(|| Error::invalid("audio is too large for a RIFF WAV file"))?;
		let byte_rate = self
			.sample_rate
			.checked_mul(u32::from(self.channels))
			.and_then(|rate| rate.checked_mul(4))
			.ok_or_else(|| Error::invalid("WAV byte rate overflowed"))?;
		let block_align = self
			.channels
			.checked_mul(4)
			.ok_or_else(|| Error::invalid("WAV block alignment overflowed"))?;

		let mut header = [0_u8; 44];
		header[0..4].copy_from_slice(b"RIFF");
		header[4..8].copy_from_slice(&riff_size.to_le_bytes());
		header[8..12].copy_from_slice(b"WAVE");
		header[12..16].copy_from_slice(b"fmt ");
		header[16..20].copy_from_slice(&16_u32.to_le_bytes());
		header[20..22].copy_from_slice(&3_u16.to_le_bytes());
		header[22..24].copy_from_slice(&self.channels.to_le_bytes());
		header[24..28].copy_from_slice(&self.sample_rate.to_le_bytes());
		header[28..32].copy_from_slice(&byte_rate.to_le_bytes());
		header[32..34].copy_from_slice(&block_align.to_le_bytes());
		header[34..36].copy_from_slice(&32_u16.to_le_bytes());
		header[36..40].copy_from_slice(b"data");
		header[40..44].copy_from_slice(&data_size.to_le_bytes());

		let mut file = tokio::fs::File::create(path).await?;
		file.write_all(&header).await?;
		let mut bytes = [0_u8; 8192];
		for samples in self.samples.chunks(bytes.len() / size_of::<f32>()) {
			for (sample, output) in samples.iter().zip(bytes.as_chunks_mut::<4>().0) {
				output.copy_from_slice(&sample.to_le_bytes());
			}
			file.write_all(&bytes[..samples.len() * 4]).await?;
		}
		file.flush().await?;
		Ok(())
	}

	pub(crate) fn into_mono_at(self, target_rate: u32) -> Result<Vec<f32>> {
		if target_rate == 0 {
			return Err(Error::invalid("target sample rate must be non-zero"));
		}
		let mut mono = if self.channels == 1 {
			self.samples
		} else {
			let channels = usize::from(self.channels);
			self
				.samples
				.chunks_exact(channels)
				.map(|frame| frame.iter().sum::<f32>() / f32::from(self.channels))
				.collect()
		};
		if self.sample_rate == target_rate || mono.is_empty() {
			return Ok(mono);
		}

		let output_len = u64::try_from(mono.len())
			.ok()
			.and_then(|length| length.checked_mul(u64::from(target_rate)))
			.and_then(|length| length.checked_add(u64::from(self.sample_rate) / 2))
			.map(|length| length / u64::from(self.sample_rate))
			.and_then(|length| usize::try_from(length).ok())
			.ok_or_else(|| Error::invalid("resampled audio length overflowed"))?;
		let mut output = Vec::with_capacity(output_len);
		let scale = f64::from(self.sample_rate) / f64::from(target_rate);
		for index in 0..output_len {
			let position = index as f64 * scale;
			let left = position.floor() as usize;
			let fraction = (position - left as f64) as f32;
			let a = mono[left.min(mono.len() - 1)];
			let b = mono[(left + 1).min(mono.len() - 1)];
			output.push(f32::mul_add(b - a, fraction, a));
		}
		mono.clear();
		Ok(output)
	}
}

#[cfg(test)]
mod tests {
	use super::Audio;

	#[test]
	fn stereo_downmix_and_resampling_preserve_frame_duration() {
		let audio = Audio::new(vec![1.0, -1.0, 0.5, 0.5], 2, 2).unwrap();
		let mono = audio.into_mono_at(4).unwrap();
		assert_eq!(mono, [0.0, 0.25, 0.5, 0.5]);
	}
}
