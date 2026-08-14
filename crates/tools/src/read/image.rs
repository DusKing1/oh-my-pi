//! Pure image classification and model-boundary normalization.

use std::{fmt, io::Cursor, path::Path};

use bytes::Bytes;
use image::{
	AnimationDecoder, DynamicImage, ImageEncoder, ImageFormat,
	codecs::{
		gif::GifDecoder,
		jpeg::JpegEncoder,
		png::PngEncoder,
		webp::{WebPDecoder, WebPEncoder},
	},
	imageops::FilterType,
};
use omp_core::Str;

/// Largest image accepted by the read tool (20 MiB).
pub const MAX_IMAGE_INPUT_BYTES: usize = 20 * 1024 * 1024;
/// Longest allowed output edge in pixels.
pub const MAX_IMAGE_WIDTH: u32 = 1_568;
/// Longest allowed output edge in pixels.
pub const MAX_IMAGE_HEIGHT: u32 = 1_568;
/// Smallest edge accepted reliably by vision backends.
pub const MIN_IMAGE_DIMENSION: u32 = 200;
/// Preferred encoded output budget (500 KiB).
pub const MAX_IMAGE_OUTPUT_BYTES: usize = 500 * 1024;

const IMAGE_METADATA_HEADER_BYTES: usize = 256 * 1024;
const COMFORTABLE_IMAGE_BYTES: usize = MAX_IMAGE_OUTPUT_BYTES / 4;
const JPEG_QUALITY: u8 = 80;
const QUALITY_STEPS: [u8; 4] = [70, 60, 50, 40];
const SCALE_STEPS: [f64; 5] = [1.0, 0.75, 0.5, 0.35, 0.25];

/// Supported image encoding discovered from file bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageKind {
	/// Portable Network Graphics.
	Png,
	/// Joint Photographic Experts Group image.
	Jpeg,
	/// Graphics Interchange Format image.
	Gif,
	/// WebP image.
	WebP,
}

impl ImageKind {
	/// Model-facing media type for this encoding.
	pub const fn media_type(self) -> &'static str {
		match self {
			Self::Png => "image/png",
			Self::Jpeg => "image/jpeg",
			Self::Gif => "image/gif",
			Self::WebP => "image/webp",
		}
	}

	const fn format(self) -> ImageFormat {
		match self {
			Self::Png => ImageFormat::Png,
			Self::Jpeg => ImageFormat::Jpeg,
			Self::Gif => ImageFormat::Gif,
			Self::WebP => ImageFormat::WebP,
		}
	}
}

/// Header metadata used to classify an image before decoding it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImageMetadata {
	/// Encoding identified by magic bytes.
	pub kind:   ImageKind,
	/// Header width when present and valid.
	pub width:  Option<u32>,
	/// Header height when present and valid.
	pub height: Option<u32>,
}

/// Processed image ready for the executor to place in blob storage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProcessedImage {
	/// Encoded image bytes. An unchanged input reuses the caller's allocation.
	pub data:                Bytes,
	/// Media type matching `data`.
	pub media_type:          Str,
	/// Encoded byte count.
	pub bytes:               usize,
	/// Decoded source width, when decoding succeeded.
	pub original_width:      Option<u32>,
	/// Decoded source height, when decoding succeeded.
	pub original_height:     Option<u32>,
	/// Displayed width, when decoding succeeded.
	pub width:               Option<u32>,
	/// Displayed height, when decoding succeeded.
	pub height:              Option<u32>,
	/// Whether the image was re-encoded by the resize pipeline.
	pub was_resized:         bool,
	/// Whether the source contained multiple animation frames.
	pub was_animated:        bool,
	/// Whether the output retains the source animation.
	pub animation_preserved: bool,
	/// Model-visible text accompanying the blob part.
	pub description:         Str,
}

/// Typed image-processing failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ImageFault {
	/// The encoded input exceeds the hard read limit.
	TooLarge {
		/// Actual encoded byte count.
		bytes:     usize,
		/// Maximum accepted byte count.
		max_bytes: usize,
	},
}

impl ImageFault {
	/// Exact model-facing failure text used by pi.
	pub fn message(&self) -> Str {
		match *self {
			Self::TooLarge { bytes, max_bytes } => Str::from(format!(
				"Image file too large: {} exceeds {} limit.",
				format_bytes(bytes),
				format_bytes(max_bytes)
			)),
		}
	}
}

impl fmt::Display for ImageFault {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter.write_str(self.message().as_ref())
	}
}

impl std::error::Error for ImageFault {}

/// Returns whether a path has one of pi's supported image extensions.
///
/// Byte sniffing remains authoritative: an image may be recognized without one
/// of these extensions, and a file with one of these extensions may not decode
/// as an image.
pub fn is_supported_extension(path: &Path) -> bool {
	path
		.extension()
		.and_then(|extension| extension.to_str())
		.is_some_and(|extension| {
			["png", "jpg", "jpeg", "gif", "webp"]
				.into_iter()
				.any(|supported| extension.eq_ignore_ascii_case(supported))
		})
}

/// Classifies PNG, JPEG, GIF, and WebP bytes.
///
/// Extracts dimensions available in their headers. This intentionally
/// recognizes truncated images after a valid magic signature, matching pi's
/// classification behavior.
pub fn sniff_metadata(header: &[u8]) -> Option<ImageMetadata> {
	parse_png(header)
		.or_else(|| parse_jpeg(header))
		.or_else(|| parse_gif(header))
		.or_else(|| parse_webp(header))
}

/// Normalizes an in-memory image for a model.
///
/// The hard input-size limit is enforced before format sniffing; smaller inputs
/// return `None` when their bytes are not one of the four supported image
/// encodings. Inputs within the dimension bounds and at most one quarter of the
/// output budget are retained verbatim. Other inputs are resized/recompressed
/// using pi's dimension, quality, and scale ladders. GIF/WebP animation is
/// retained on the verbatim path; re-encoding produces the decoded first frame.
pub fn process_image(input: Bytes) -> Result<Option<ProcessedImage>, ImageFault> {
	if input.len() > MAX_IMAGE_INPUT_BYTES {
		return Err(ImageFault::TooLarge {
			bytes:     input.len(),
			max_bytes: MAX_IMAGE_INPUT_BYTES,
		});
	}
	let Some(metadata) = sniff_metadata(&input[..input.len().min(IMAGE_METADATA_HEADER_BYTES)])
	else {
		return Ok(None);
	};

	let exclude_webp = webp_is_excluded();
	let decoded = decode_image(&input, metadata.kind);
	let Ok((image, was_animated)) = decoded else {
		return Ok(Some(unchanged_image(input, metadata, false)));
	};
	let original_width = image.width();
	let original_height = image.height();
	let within_dimensions = original_width >= MIN_IMAGE_DIMENSION
		&& original_height >= MIN_IMAGE_DIMENSION
		&& original_width <= MAX_IMAGE_WIDTH
		&& original_height <= MAX_IMAGE_HEIGHT;
	if within_dimensions
		&& input.len() <= COMFORTABLE_IMAGE_BYTES
		&& !(exclude_webp && metadata.kind == ImageKind::WebP)
	{
		return Ok(Some(unchanged_decoded_image(
			input,
			metadata.kind,
			original_width,
			original_height,
			was_animated,
		)));
	}

	let Some(encoded) = resize_and_encode(&image, exclude_webp) else {
		return Ok(Some(unchanged_decoded_image(
			input,
			metadata.kind,
			original_width,
			original_height,
			was_animated,
		)));
	};
	let bytes = encoded.data.len();
	let dimension_note =
		dimension_note(original_width, original_height, encoded.width, encoded.height);
	let description = match dimension_note {
		Some(note) => Str::from(format!("Read image file [{}]\n{note}", encoded.kind.media_type())),
		None => Str::from(format!("Read image file [{}]", encoded.kind.media_type())),
	};
	Ok(Some(ProcessedImage {
		data: Bytes::from(encoded.data),
		media_type: Str::new_static(encoded.kind.media_type()),
		bytes,
		original_width: Some(original_width),
		original_height: Some(original_height),
		width: Some(encoded.width),
		height: Some(encoded.height),
		was_resized: true,
		was_animated,
		animation_preserved: false,
		description,
	}))
}

struct EncodedImage {
	data:   Vec<u8>,
	kind:   ImageKind,
	width:  u32,
	height: u32,
}

fn unchanged_image(input: Bytes, metadata: ImageMetadata, was_animated: bool) -> ProcessedImage {
	let bytes = input.len();
	ProcessedImage {
		data: input,
		media_type: Str::new_static(metadata.kind.media_type()),
		bytes,
		original_width: metadata.width,
		original_height: metadata.height,
		width: metadata.width,
		height: metadata.height,
		was_resized: false,
		was_animated,
		animation_preserved: was_animated,
		description: Str::from(format!("Read image file [{}]", metadata.kind.media_type())),
	}
}

fn unchanged_decoded_image(
	input: Bytes,
	kind: ImageKind,
	width: u32,
	height: u32,
	was_animated: bool,
) -> ProcessedImage {
	unchanged_image(
		input,
		ImageMetadata { kind, width: Some(width), height: Some(height) },
		was_animated,
	)
}

fn decode_image(input: &[u8], kind: ImageKind) -> image::ImageResult<(DynamicImage, bool)> {
	match kind {
		ImageKind::Gif => {
			let decoder = GifDecoder::new(Cursor::new(input))?;
			let mut frames = decoder.into_frames();
			let first = frames.next().transpose()?.ok_or_else(|| {
				image::ImageError::Decoding(image::error::DecodingError::new(
					ImageFormat::Gif.into(),
					std::io::Error::new(std::io::ErrorKind::InvalidData, "GIF contains no image frames"),
				))
			})?;
			let animated = frames.next().transpose()?.is_some();
			Ok((DynamicImage::ImageRgba8(first.into_buffer()), animated))
		},
		ImageKind::WebP => {
			let decoder = WebPDecoder::new(Cursor::new(input))?;
			let animated = decoder.has_animation();
			if animated {
				let mut frames = decoder.into_frames();
				let first = frames.next().transpose()?.ok_or_else(|| {
					image::ImageError::Decoding(image::error::DecodingError::new(
						ImageFormat::WebP.into(),
						std::io::Error::new(
							std::io::ErrorKind::InvalidData,
							"WebP contains no image frames",
						),
					))
				})?;
				Ok((DynamicImage::ImageRgba8(first.into_buffer()), true))
			} else {
				Ok((DynamicImage::from_decoder(decoder)?, false))
			}
		},
		_ => image::load_from_memory_with_format(input, kind.format()).map(|image| (image, false)),
	}
}

fn resize_and_encode(image: &DynamicImage, exclude_webp: bool) -> Option<EncodedImage> {
	let (target_width, target_height) = target_dimensions(image.width(), image.height());
	let resized = image.resize_exact(target_width, target_height, FilterType::Lanczos3);
	let mut best = encode_smallest(&resized, JPEG_QUALITY, exclude_webp)?;
	if best.0.len() <= MAX_IMAGE_OUTPUT_BYTES {
		return Some(EncodedImage {
			data:   best.0,
			kind:   best.1,
			width:  target_width,
			height: target_height,
		});
	}

	for quality in QUALITY_STEPS {
		best = encode_lossy_smallest(&resized, quality, exclude_webp)?;
		if best.0.len() <= MAX_IMAGE_OUTPUT_BYTES {
			return Some(EncodedImage {
				data:   best.0,
				kind:   best.1,
				width:  target_width,
				height: target_height,
			});
		}
	}

	let mut final_width = target_width;
	let mut final_height = target_height;
	for scale in SCALE_STEPS {
		final_width = ((target_width as f64) * scale).round() as u32;
		final_height = ((target_height as f64) * scale).round() as u32;
		if final_width < 100 || final_height < 100 {
			break;
		}
		let scaled = image.resize_exact(final_width, final_height, FilterType::Lanczos3);
		for quality in QUALITY_STEPS {
			best = encode_lossy_smallest(&scaled, quality, exclude_webp)?;
			if best.0.len() <= MAX_IMAGE_OUTPUT_BYTES {
				return Some(EncodedImage {
					data:   best.0,
					kind:   best.1,
					width:  final_width,
					height: final_height,
				});
			}
		}
	}
	Some(EncodedImage { data: best.0, kind: best.1, width: final_width, height: final_height })
}

fn target_dimensions(original_width: u32, original_height: u32) -> (u32, u32) {
	let mut width = original_width;
	let mut height = original_height;
	if width > MAX_IMAGE_WIDTH {
		height = ((height as f64 * MAX_IMAGE_WIDTH as f64) / width as f64).round() as u32;
		width = MAX_IMAGE_WIDTH;
	}
	if height > MAX_IMAGE_HEIGHT {
		width = ((width as f64 * MAX_IMAGE_HEIGHT as f64) / height as f64).round() as u32;
		height = MAX_IMAGE_HEIGHT;
	}
	if width < MIN_IMAGE_DIMENSION || height < MIN_IMAGE_DIMENSION {
		let short_edge = width.min(height);
		let upscale = (MIN_IMAGE_DIMENSION as f64 / short_edge as f64)
			.min(MAX_IMAGE_WIDTH as f64 / width as f64)
			.min(MAX_IMAGE_HEIGHT as f64 / height as f64);
		if upscale > 1.0 {
			width = (width as f64 * upscale).round() as u32;
			height = (height as f64 * upscale).round() as u32;
		}
		width = width.clamp(MIN_IMAGE_DIMENSION, MAX_IMAGE_WIDTH);
		height = height.clamp(MIN_IMAGE_DIMENSION, MAX_IMAGE_HEIGHT);
	}
	(width, height)
}

fn encode_smallest(
	image: &DynamicImage,
	jpeg_quality: u8,
	exclude_webp: bool,
) -> Option<(Vec<u8>, ImageKind)> {
	let mut candidates = Vec::with_capacity(if exclude_webp { 2 } else { 3 });
	if let Ok(data) = encode_png(image) {
		candidates.push((data, ImageKind::Png));
	}
	if let Ok(data) = encode_jpeg(image, jpeg_quality) {
		candidates.push((data, ImageKind::Jpeg));
	}
	if !exclude_webp && let Ok(data) = encode_webp(image) {
		candidates.push((data, ImageKind::WebP));
	}
	candidates.into_iter().min_by_key(|(data, _)| data.len())
}

fn encode_lossy_smallest(
	image: &DynamicImage,
	jpeg_quality: u8,
	exclude_webp: bool,
) -> Option<(Vec<u8>, ImageKind)> {
	let jpeg = encode_jpeg(image, jpeg_quality)
		.ok()
		.map(|data| (data, ImageKind::Jpeg));
	if exclude_webp {
		return jpeg;
	}
	let webp = encode_webp(image).ok().map(|data| (data, ImageKind::WebP));
	match (jpeg, webp) {
		(Some(jpeg), Some(webp)) => Some(if webp.0.len() < jpeg.0.len() {
			webp
		} else {
			jpeg
		}),
		(Some(jpeg), None) => Some(jpeg),
		(None, Some(webp)) => Some(webp),
		(None, None) => None,
	}
}

fn encode_png(image: &DynamicImage) -> image::ImageResult<Vec<u8>> {
	let rgba = image.to_rgba8();
	let mut output = Vec::new();
	PngEncoder::new(&mut output).write_image(
		rgba.as_raw(),
		rgba.width(),
		rgba.height(),
		image::ExtendedColorType::Rgba8,
	)?;
	Ok(output)
}

fn encode_jpeg(image: &DynamicImage, quality: u8) -> image::ImageResult<Vec<u8>> {
	let rgb = image.to_rgb8();
	let mut output = Vec::new();
	JpegEncoder::new_with_quality(&mut output, quality).write_image(
		rgb.as_raw(),
		rgb.width(),
		rgb.height(),
		image::ExtendedColorType::Rgb8,
	)?;
	Ok(output)
}

fn encode_webp(image: &DynamicImage) -> image::ImageResult<Vec<u8>> {
	let rgba = image.to_rgba8();
	let mut output = Vec::new();
	WebPEncoder::new_lossless(&mut output).write_image(
		rgba.as_raw(),
		rgba.width(),
		rgba.height(),
		image::ExtendedColorType::Rgba8,
	)?;
	Ok(output)
}

fn dimension_note(
	original_width: u32,
	original_height: u32,
	width: u32,
	height: u32,
) -> Option<String> {
	if width == original_width && height == original_height {
		return None;
	}
	let scale = original_width as f64 / width as f64;
	Some(format!(
		"[Image: original {original_width}x{original_height}, displayed at {width}x{height}. \
		 Multiply coordinates by {scale:.2} to map to original image.]"
	))
}

fn webp_is_excluded() -> bool {
	std::env::var("OMP_NO_WEBP")
		.is_ok_and(|value| value.eq_ignore_ascii_case("1") || value.eq_ignore_ascii_case("true"))
}

fn parse_png(header: &[u8]) -> Option<ImageMetadata> {
	const MAGIC: &[u8] = b"\x89PNG\r\n\x1a\n";
	if !header.starts_with(MAGIC) {
		return None;
	}
	let dimensions = (header.len() >= 26 && &header[12..16] == b"IHDR").then(|| {
		(
			u32::from_be_bytes(header[16..20].try_into().unwrap()),
			u32::from_be_bytes(header[20..24].try_into().unwrap()),
		)
	});
	Some(ImageMetadata {
		kind:   ImageKind::Png,
		width:  dimensions.map(|value| value.0),
		height: dimensions.map(|value| value.1),
	})
}

fn parse_jpeg(header: &[u8]) -> Option<ImageMetadata> {
	if header.len() < 3 || header[..3] != [0xff, 0xd8, 0xff] {
		return None;
	}
	let mut offset = 2;
	while offset + 9 < header.len() {
		if header[offset] != 0xff {
			offset += 1;
			continue;
		}
		let mut marker_offset = offset + 1;
		while marker_offset < header.len() && header[marker_offset] == 0xff {
			marker_offset += 1;
		}
		if marker_offset >= header.len() {
			break;
		}
		let marker = header[marker_offset];
		let segment_offset = marker_offset + 1;
		if marker == 0xd8 || marker == 0xd9 || marker == 0x01 || (0xd0..=0xd7).contains(&marker) {
			offset = segment_offset;
			continue;
		}
		if segment_offset + 1 >= header.len() {
			break;
		}
		let length =
			u16::from_be_bytes([header[segment_offset], header[segment_offset + 1]]) as usize;
		if length < 2 {
			break;
		}
		let is_start_of_frame =
			(0xc0..=0xcf).contains(&marker) && !matches!(marker, 0xc4 | 0xc8 | 0xcc);
		if is_start_of_frame {
			if segment_offset + 7 >= header.len() {
				break;
			}
			return Some(ImageMetadata {
				kind:   ImageKind::Jpeg,
				width:  Some(u16::from_be_bytes([
					header[segment_offset + 5],
					header[segment_offset + 6],
				]) as u32),
				height: Some(u16::from_be_bytes([
					header[segment_offset + 3],
					header[segment_offset + 4],
				]) as u32),
			});
		}
		offset = segment_offset.saturating_add(length);
	}
	Some(ImageMetadata { kind: ImageKind::Jpeg, width: None, height: None })
}

fn parse_gif(header: &[u8]) -> Option<ImageMetadata> {
	if !header.starts_with(b"GIF87a") && !header.starts_with(b"GIF89a") {
		return None;
	}
	let dimensions = (header.len() >= 10).then(|| {
		(
			u16::from_le_bytes([header[6], header[7]]) as u32,
			u16::from_le_bytes([header[8], header[9]]) as u32,
		)
	});
	Some(ImageMetadata {
		kind:   ImageKind::Gif,
		width:  dimensions.map(|value| value.0),
		height: dimensions.map(|value| value.1),
	})
}

fn parse_webp(header: &[u8]) -> Option<ImageMetadata> {
	if header.len() < 12 || &header[..4] != b"RIFF" || &header[8..12] != b"WEBP" {
		return None;
	}
	if header.len() < 30 {
		return Some(ImageMetadata { kind: ImageKind::WebP, width: None, height: None });
	}
	let dimensions = if &header[12..16] == b"VP8X" {
		Some((read_u24_le(&header[24..27]) + 1, read_u24_le(&header[27..30]) + 1))
	} else if &header[12..16] == b"VP8L" {
		let bits = u32::from_le_bytes(header[21..25].try_into().unwrap());
		Some(((bits & 0x3fff) + 1, ((bits >> 14) & 0x3fff) + 1))
	} else if &header[12..16] == b"VP8 " {
		Some((
			u16::from_le_bytes([header[26], header[27]]) as u32 & 0x3fff,
			u16::from_le_bytes([header[28], header[29]]) as u32 & 0x3fff,
		))
	} else {
		None
	};
	Some(ImageMetadata {
		kind:   ImageKind::WebP,
		width:  dimensions.map(|value| value.0),
		height: dimensions.map(|value| value.1),
	})
}

fn read_u24_le(bytes: &[u8]) -> u32 {
	u32::from(bytes[0]) | (u32::from(bytes[1]) << 8) | (u32::from(bytes[2]) << 16)
}

fn format_bytes(bytes: usize) -> String {
	if bytes < 1024 {
		format!("{bytes}B")
	} else if bytes < 1024 * 1024 {
		format!("{:.1}KB", bytes as f64 / 1024.0)
	} else if bytes < 1024 * 1024 * 1024 {
		format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
	} else {
		format!("{:.1}GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
	}
}
