//! Anthropic image and document source projection.

use std::fmt;

use omp_llm_types::BlobPart;
use serde::Serialize;

/// Anthropic media types accepted by the Messages API.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MediaKind {
	/// A supported image source.
	Image,
	/// A PDF document source.
	Document,
}

/// Classifies a canonical blob using Anthropic's exact media allowlist.
///
/// MIME matching is ASCII-case-insensitive and ignores surrounding whitespace;
/// `image/jpg` is normalized to Anthropic's canonical `image/jpeg`.
pub(crate) fn media_kind(blob: &BlobPart) -> Result<(MediaKind, &'static str), &'static str> {
	let mime = blob.mime.trim();
	if mime.eq_ignore_ascii_case("image/jpeg") || mime.eq_ignore_ascii_case("image/jpg") {
		Ok((MediaKind::Image, "image/jpeg"))
	} else if mime.eq_ignore_ascii_case("image/png") {
		Ok((MediaKind::Image, "image/png"))
	} else if mime.eq_ignore_ascii_case("image/gif") {
		Ok((MediaKind::Image, "image/gif"))
	} else if mime.eq_ignore_ascii_case("image/webp") {
		Ok((MediaKind::Image, "image/webp"))
	} else if mime.eq_ignore_ascii_case("application/pdf") {
		Ok((MediaKind::Document, "application/pdf"))
	} else {
		Err("Anthropic accepts only JPEG, PNG, GIF, WebP, and PDF media")
	}
}

/// A validated inline media payload ready for wire serialization.
#[derive(Clone, Copy, Debug)]
pub(crate) struct InlineMedia<'a> {
	pub(crate) media_type: &'static str,
	pub(crate) data:       &'a [u8],
}

/// Validates the inline bytes of a supported canonical media blob.
pub(crate) fn inline_media(blob: &BlobPart) -> Result<InlineMedia<'_>, &'static str> {
	let (_, media_type) = media_kind(blob)?;
	if blob.inline.is_empty() {
		return Err("Anthropic base64 media projection requires non-empty resolved inline bytes");
	}
	if blob.size != blob.inline.len() as u64 {
		return Err("Anthropic base64 media size does not match the resolved inline payload");
	}
	Ok(InlineMedia { media_type, data: &blob.inline })
}

/// Anthropic media sources shared by image and document blocks.
#[derive(Clone, Copy, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum MediaSource<'a> {
	Base64 { media_type: &'static str, data: Base64<'a> },
	Url { url: &'a str },
	File { file_id: &'a str },
}

impl<'a> MediaSource<'a> {
	pub(crate) const fn inline(media: InlineMedia<'a>) -> Self {
		Self::Base64 { media_type: media.media_type, data: Base64(media.data) }
	}
}
/// Validates an Anthropic URL media source without resolving the payload.
pub(crate) fn url_source(value: &str) -> Result<&str, &'static str> {
	if value.is_empty() || value.trim() != value {
		return Err("Anthropic URL source must be non-empty and contain no surrounding whitespace");
	}
	let uri = value
		.parse::<http::Uri>()
		.map_err(|_| "Anthropic URL source must be a valid absolute HTTP(S) URL")?;
	if !matches!(uri.scheme_str(), Some("http" | "https")) || uri.authority().is_none() {
		return Err("Anthropic URL source must be a valid absolute HTTP(S) URL");
	}
	Ok(value)
}

/// Validates an Anthropic Files API identifier without resolving the payload.
pub(crate) fn file_source(value: &str) -> Result<&str, &'static str> {
	let Some(suffix) = value.strip_prefix("file_") else {
		return Err("Anthropic file source must use a non-empty `file_` identifier");
	};
	if suffix.is_empty()
		|| !suffix
			.bytes()
			.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
	{
		return Err("Anthropic file source must use a non-empty `file_` identifier");
	}
	Ok(value)
}

/// Borrowed bytes serialized as canonical padded base64 without allocating.
#[derive(Clone, Copy)]
pub(crate) struct Base64<'a>(pub(crate) &'a [u8]);

impl Serialize for Base64<'_> {
	fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
	where
		S: serde::Serializer,
	{
		serializer.collect_str(self)
	}
}

impl fmt::Display for Base64<'_> {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		const ALPHABET: &[u8; 64] =
			b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
		for chunk in self.0.chunks(3) {
			let first = chunk[0];
			let second = chunk.get(1).copied().unwrap_or(0);
			let third = chunk.get(2).copied().unwrap_or(0);
			let encoded = [
				ALPHABET[usize::from(first >> 2)],
				ALPHABET[usize::from(((first & 3) << 4) | (second >> 4))],
				if chunk.len() > 1 {
					ALPHABET[usize::from(((second & 15) << 2) | (third >> 6))]
				} else {
					b'='
				},
				if chunk.len() > 2 {
					ALPHABET[usize::from(third & 63)]
				} else {
					b'='
				},
			];
			formatter.write_str(std::str::from_utf8(&encoded).expect("base64 is ASCII"))?;
		}
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use bytes::Bytes;
	use omp_llm_types::BlobPart;

	use super::{Base64, MediaKind, file_source, inline_media, media_kind, url_source};

	fn blob(mime: &str, bytes: &'static [u8]) -> BlobPart {
		BlobPart::builder()
			.hash([0; 32])
			.mime(mime.into())
			.size(bytes.len() as u64)
			.inline(Bytes::from_static(bytes))
			.build()
	}

	#[test]
	fn validates_exact_anthropic_media_types() {
		let jpeg_blob = blob(" image/JPG ", b"abc");
		let (kind, mime) = media_kind(&jpeg_blob).expect("jpeg");
		assert_eq!(kind, MediaKind::Image);
		assert_eq!(mime, "image/jpeg");
		let pdf_blob = blob("application/pdf", b"%PDF");
		let (kind, mime) = media_kind(&pdf_blob).expect("pdf");
		assert_eq!(kind, MediaKind::Document);
		assert_eq!(mime, "application/pdf");
		assert_eq!(inline_media(&pdf_blob).expect("inline PDF").media_type, "application/pdf");
		assert!(media_kind(&blob("image/svg+xml", b"<svg/>")).is_err());
		assert!(media_kind(&blob("application/octet-stream", b"x")).is_err());
		let mut mismatched = blob("application/pdf", b"%PDF");
		mismatched.size += 1;
		assert!(inline_media(&mismatched).is_err());
		assert!(inline_media(&blob("application/pdf", b"")).is_err());
		assert_eq!(
			url_source("https://example.test/manual.pdf?version=1").unwrap(),
			"https://example.test/manual.pdf?version=1"
		);
		assert!(url_source("/manual.pdf").is_err());
		assert_eq!(file_source("file_012345").unwrap(), "file_012345");
		assert!(file_source("document_012345").is_err());
	}

	#[test]
	fn base64_projection_is_padded_without_an_intermediate_buffer() {
		assert_eq!(Base64(b"hello").to_string(), "aGVsbG8=");
	}
}
