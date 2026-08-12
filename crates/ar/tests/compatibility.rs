//! Legacy-name and ZIP64 compatibility contracts.

mod support;

use omp_ar::{Archive, zip::CompressionMethod};
use support::{Member, fixture, zip64_fixture};

#[test]
fn legacy_names_use_windows_1252_when_utf8_flag_is_clear() {
	let bytes = fixture(&[Member::stored(b"price-\x80.txt", b"ten euros")]);
	let mut archive = Archive::from_bytes(&bytes).unwrap();

	let entry = archive.entry("price-€.txt").unwrap();
	assert_eq!(entry.name(), "price-€.txt");
	assert_eq!(archive.read("price-€.txt").unwrap().as_slice(), b"ten euros");
}

#[test]
fn zip64_directory_and_entry_metadata_are_read() {
	let bytes = zip64_fixture(b"zip64.txt", b"small payload, ZIP64 metadata");
	let mut archive = Archive::from_bytes(&bytes).unwrap();

	let entry = archive.entry("zip64.txt").unwrap();
	assert_eq!(entry.size(), 29);
	assert_eq!(entry.compressed_size(), 29);
	assert_eq!(entry.zip_compression(), Some(CompressionMethod::Stored));
	assert_eq!(archive.read("zip64.txt").unwrap().as_slice(), b"small payload, ZIP64 metadata");
}
