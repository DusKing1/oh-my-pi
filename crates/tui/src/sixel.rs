use std::{collections::BTreeMap, fmt::Write as _};

use crate::escape::esc;

const MAX_COLORS: usize = 256;

#[derive(Clone, Copy)]
struct ColorSample {
	rgb:   [u8; 3],
	count: usize,
	first: usize,
}

pub struct SixelImage {
	pixels:  Vec<Vec<[u8; 3]>>,
	palette: Vec<[u8; 3]>,
	indices: Vec<u8>,
}

impl SixelImage {
	pub(crate) fn new(pixels: Vec<Vec<[u8; 3]>>) -> Option<Self> {
		let width = pixels.first()?.len();
		if width == 0 || pixels.iter().any(|row| row.len() != width) {
			return None;
		}
		let height = pixels.len();
		let palette = quantize(&pixels);
		let mut indices = Vec::with_capacity(width.saturating_mul(height));
		for row in &pixels {
			for &pixel in row {
				indices.push(nearest(pixel, &palette) as u8);
			}
		}
		Some(Self { pixels, palette, indices })
	}

	pub(crate) fn from_png(bytes: &[u8]) -> Option<Self> {
		let mut decoder = png::Decoder::new(std::io::Cursor::new(bytes));
		decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
		let mut reader = decoder.read_info().ok()?;
		let mut buffer = vec![0_u8; reader.output_buffer_size()?];
		let info = reader.next_frame(&mut buffer).ok()?;
		let bytes = &buffer[..info.buffer_size()];
		let width = usize::try_from(info.width).ok()?;
		let height = usize::try_from(info.height).ok()?;
		let mut pixels = Vec::with_capacity(height);
		let stride = info.color_type.samples();
		for y in 0..height {
			let mut row = Vec::with_capacity(width);
			for x in 0..width {
				let at = y.checked_mul(width)?.checked_add(x)?.checked_mul(stride)?;
				let rgb = match info.color_type {
					png::ColorType::Rgb | png::ColorType::Rgba => {
						[*bytes.get(at)?, *bytes.get(at + 1)?, *bytes.get(at + 2)?]
					},
					png::ColorType::Grayscale | png::ColorType::GrayscaleAlpha => {
						let value = *bytes.get(at)?;
						[value, value, value]
					},
					png::ColorType::Indexed => return None,
				};
				row.push(rgb);
			}
			pixels.push(row);
		}
		Self::new(pixels)
	}

	/// Encodes scaled pixel rows `[y0, y1)` as one self-contained sixel DCS.
	pub(crate) fn encode_band(
		&self,
		target_width: usize,
		target_height: usize,
		y0: usize,
		y1: usize,
	) -> String {
		let y0 = y0.min(target_height);
		let y1 = y1.min(target_height).max(y0);
		if target_width == 0 || y0 == y1 {
			return String::new();
		}
		let band_height = y1 - y0;
		let mut output = String::new();
		output.push_str(esc!(dcs, "0;1;0q"));
		let _ = write!(output, "\"1;1;{target_width};{band_height}");
		for (index, color) in self.palette.iter().enumerate() {
			let _ = write!(
				output,
				"#{index};2;{};{};{}",
				percent(color[0]),
				percent(color[1]),
				percent(color[2])
			);
		}

		let bands = band_height.div_ceil(6);
		let mut columns = Vec::with_capacity(target_width);
		for band in 0..bands {
			for color in 0..self.palette.len() {
				columns.clear();
				for x in 0..target_width {
					let mut bits = 0_u8;
					for bit in 0..6 {
						let target_y = y0 + band * 6 + bit;
						if target_y >= y1 {
							break;
						}
						if self.scaled_index(x, target_y, target_width, target_height) == color {
							bits |= 0b0000_0001 << bit;
						}
					}
					columns.push(bits + 63);
				}
				while columns.last() == Some(&63) {
					columns.pop();
				}
				let _ = write!(output, "#{color}");
				append_runs(&mut output, &columns);
				if color + 1 < self.palette.len() {
					output.push('$');
				}
			}
			if band + 1 < bands {
				output.push('-');
			}
		}
		output.push_str(esc!(st));
		output
	}

	fn scaled_index(&self, x: usize, y: usize, width: usize, height: usize) -> usize {
		let source_width = self.pixels[0].len();
		let source_x = x.saturating_mul(source_width) / width;
		let source_y = y.saturating_mul(self.pixels.len()) / height;
		usize::from(self.indices[source_y * source_width + source_x])
	}

	#[cfg(test)]
	fn pixels(&self) -> &[Vec<[u8; 3]>] {
		&self.pixels
	}
}

fn append_runs(output: &mut String, columns: &[u8]) {
	let mut at = 0;
	while at < columns.len() {
		let byte = columns[at];
		let mut end = at + 1;
		while end < columns.len() && columns[end] == byte {
			end += 1;
		}
		let count = end - at;
		if count > 1 {
			let _ = write!(output, "!{count}{}", char::from(byte));
		} else {
			output.push(char::from(byte));
		}
		at = end;
	}
}

const fn percent(value: u8) -> u16 {
	(value as u16 * 100 + 127) / 255
}

fn quantize(pixels: &[Vec<[u8; 3]>]) -> Vec<[u8; 3]> {
	let mut unique = BTreeMap::<[u8; 3], (usize, usize)>::new();
	let mut first = 0;
	for row in pixels {
		for &pixel in row {
			let entry = unique.entry(pixel).or_insert_with(|| {
				let order = first;
				first += 1;
				(0, order)
			});
			entry.0 += 1;
		}
	}
	let mut samples: Vec<_> = unique
		.into_iter()
		.map(|(rgb, (count, first))| ColorSample { rgb, count, first })
		.collect();
	if samples.len() <= MAX_COLORS {
		samples.sort_unstable_by_key(|sample| sample.first);
		return samples.into_iter().map(|sample| sample.rgb).collect();
	}

	let mut boxes = vec![(0..samples.len()).collect::<Vec<_>>()];
	while boxes.len() < MAX_COLORS {
		let Some((box_index, channel)) = boxes
			.iter()
			.enumerate()
			.filter_map(|(index, colors)| {
				split_channel(colors, &samples).map(|channel| (index, channel))
			})
			.max_by_key(|(index, channel)| {
				let colors = &boxes[*index];
				let (low, high) = channel_range(colors, &samples, *channel);
				(
					usize::from(high - low) * colors.iter().map(|&i| samples[i].count).sum::<usize>(),
					usize::MAX - *index,
				)
			})
		else {
			break;
		};
		let mut colors = boxes.swap_remove(box_index);
		colors.sort_unstable_by_key(|&index| {
			let sample = samples[index];
			(sample.rgb[channel], sample.rgb, sample.first)
		});
		let total: usize = colors.iter().map(|&index| samples[index].count).sum();
		let mut accumulated = 0;
		let mut split = 1;
		for (position, &index) in colors.iter().enumerate().take(colors.len() - 1) {
			accumulated += samples[index].count;
			if accumulated.saturating_mul(2) >= total {
				split = position + 1;
				break;
			}
		}
		let right = colors.split_off(split);
		boxes.push(colors);
		boxes.push(right);
	}

	boxes
		.iter()
		.map(|colors| {
			let mut sums = [0_usize; 3];
			let mut count = 0;
			for &index in colors {
				let sample = samples[index];
				count += sample.count;
				for (sum, value) in sums.iter_mut().zip(sample.rgb) {
					*sum += usize::from(value) * sample.count;
				}
			}
			sums.map(|sum| ((sum + count / 2) / count) as u8)
		})
		.collect()
}

fn split_channel(colors: &[usize], samples: &[ColorSample]) -> Option<usize> {
	if colors.len() < 2 {
		return None;
	}
	(0..3).max_by_key(|&channel| {
		let (low, high) = channel_range(colors, samples, channel);
		high - low
	})
}

fn channel_range(colors: &[usize], samples: &[ColorSample], channel: usize) -> (u8, u8) {
	colors
		.iter()
		.fold((u8::MAX, u8::MIN), |(low, high), &index| {
			(low.min(samples[index].rgb[channel]), high.max(samples[index].rgb[channel]))
		})
}

fn nearest(pixel: [u8; 3], palette: &[[u8; 3]]) -> usize {
	palette
		.iter()
		.enumerate()
		.min_by_key(|(_, color)| {
			pixel
				.iter()
				.zip(color.iter())
				.map(|(&a, &b)| i32::from(a).saturating_sub(i32::from(b)).pow(2) as u32)
				.sum::<u32>()
		})
		.map_or(0, |(index, _)| index)
}

#[cfg(test)]
mod tests {
	use super::SixelImage;

	fn tiny() -> SixelImage {
		SixelImage::new(vec![vec![[255, 0, 0], [255, 0, 0], [0, 0, 255], [0, 0, 255]], vec![
			[255, 0, 0],
			[255, 0, 0],
			[0, 0, 255],
			[0, 0, 255],
		]])
		.expect("rectangular image")
	}

	#[test]
	fn encodes_tiny_two_color_image() {
		assert_eq!(
			tiny().encode_band(4, 2, 0, 2),
			"\x1bP0;1;0q\"1;1;4;2#0;2;100;0;0#1;2;0;0;100#0!2B$#1!2?!2B\x1b\\"
		);
	}

	#[test]
	fn encodes_only_cropped_lower_half() {
		assert_eq!(
			tiny().encode_band(4, 2, 1, 2),
			"\x1bP0;1;0q\"1;1;4;1#0;2;100;0;0#1;2;0;0;100#0!2@$#1!2?!2@\x1b\\"
		);
	}

	#[test]
	fn retains_source_pixels_with_palette() {
		let image = tiny();
		assert_eq!(image.pixels().len(), 2);
		assert_eq!(image.palette.len(), 2);
	}
}
