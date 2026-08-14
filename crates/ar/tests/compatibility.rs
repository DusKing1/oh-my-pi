//! Legacy-name and ZIP64 compatibility contracts.

mod support;

use omp_ar::{Archive, Error, zip::CompressionMethod};
use support::{Member, assert_error_kind, fixture, zip64_fixture};

#[test]
fn legacy_names_use_zip_standard_cp437_when_utf8_flag_is_clear() {
	let bytes = fixture(&[Member::stored(b"Cura\x87ao.txt", b"coast")]);
	let mut archive = Archive::from_bytes(&bytes).unwrap();

	let entry = archive.entry("Curaçao.txt").unwrap();
	assert_eq!(entry.name(), "Curaçao.txt");
	assert_eq!(archive.read("Curaçao.txt").unwrap().as_slice(), b"coast");
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

#[test]
fn prepended_data_offsets_are_applied_to_zip32_and_zip64_records() {
	let fixtures = [
		fixture(&[Member::stored(b"ordinary.txt", b"ordinary")]),
		zip64_fixture(b"large-metadata.txt", b"zip64"),
	];
	for bytes in fixtures {
		let mut prefixed = b"MZPK\x01\x02\x90\0self-extracting-stub".to_vec();
		prefixed.extend_from_slice(&bytes);
		let mut archive = Archive::from_bytes(&prefixed).unwrap();
		let entry = archive
			.entries()
			.find(|entry| !entry.is_directory())
			.unwrap();
		let name = entry.name().to_owned();
		let expected = if name == "ordinary.txt" {
			b"ordinary".as_slice()
		} else {
			b"zip64"
		};
		assert_eq!(archive.read(&name).unwrap().as_slice(), expected);
	}
}

#[test]
fn valid_infozip_unicode_path_extra_field_overrides_the_legacy_name() {
	let raw_name = b"resume.txt";
	let unicode_name = "résumé.txt";
	let mut field = Vec::new();
	field.extend_from_slice(&0x7075_u16.to_le_bytes());
	field.extend_from_slice(&u16::try_from(5 + unicode_name.len()).unwrap().to_le_bytes());
	field.push(1);
	field.extend_from_slice(&crc32fast::hash(raw_name).to_le_bytes());
	field.extend_from_slice(unicode_name.as_bytes());
	let bytes = fixture_with_central_extra(raw_name, b"curriculum vitae", &field);

	let mut archive = Archive::from_bytes(&bytes).unwrap();
	assert_eq!(archive.entry(unicode_name).unwrap().name(), unicode_name);
	assert_eq!(archive.read(unicode_name).unwrap().as_slice(), b"curriculum vitae");
}

#[test]
fn invalid_unicode_path_crc_is_ignored_and_truncated_extra_fields_are_rejected() {
	let raw_name = b"plain.txt";
	let mut unicode = Vec::new();
	unicode.extend_from_slice(&0x7075_u16.to_le_bytes());
	unicode.extend_from_slice(&9_u16.to_le_bytes());
	unicode.push(1);
	unicode.extend_from_slice(&0_u32.to_le_bytes());
	unicode.extend_from_slice(b"fake");
	let bytes = fixture_with_central_extra(raw_name, b"plain", &unicode);
	assert!(
		Archive::from_bytes(&bytes)
			.unwrap()
			.entry("plain.txt")
			.is_some()
	);

	let malformed = fixture_with_central_extra(raw_name, b"plain", &[0x75, 0x70, 8, 0, 1]);
	assert_error_kind(Archive::from_bytes(&malformed), |error| {
		matches!(error, Error::InvalidArchive("malformed ZIP extra field"))
	});
}

#[test]
fn malformed_zip64_locator_offsets_are_rejected_without_trusting_wrapped_ranges() {
	let mut bytes = zip64_fixture(b"offset.txt", b"bounded");
	let locator = bytes
		.windows(4)
		.position(|bytes| bytes == b"PK\x06\x07")
		.unwrap();
	bytes[locator + 8..locator + 16].copy_from_slice(&u64::MAX.to_le_bytes());
	assert_error_kind(Archive::from_bytes(&bytes), |error| {
		matches!(error, Error::InvalidArchive(_))
	});
}

#[test]
fn central_directory_may_precede_file_data_and_false_eocd_comment_magic_is_skipped() {
	let original = fixture(&[Member::stored(b"front-indexed.txt", b"payload")]);
	let central = original
		.windows(4)
		.position(|bytes| bytes == b"PK\x01\x02")
		.unwrap();
	let eocd = original
		.windows(4)
		.rposition(|bytes| bytes == b"PK\x05\x06")
		.unwrap();
	let central_record = &original[central..eocd];
	let local_record = &original[..central];
	let mut reordered = Vec::new();
	reordered.extend_from_slice(central_record);
	reordered.extend_from_slice(local_record);
	reordered.extend_from_slice(&original[eocd..]);
	reordered[42..46].copy_from_slice(&u32::try_from(central_record.len()).unwrap().to_le_bytes());
	let new_eocd = central_record.len() + local_record.len();
	reordered[new_eocd + 16..new_eocd + 20].copy_from_slice(&0_u32.to_le_bytes());
	let mut archive = Archive::from_bytes(&reordered).unwrap();
	assert_eq!(archive.read("front-indexed.txt").unwrap().as_slice(), b"payload");

	let mut commented = fixture(&[Member::stored(b"commented.txt", b"safe")]);
	let actual_eocd = commented
		.windows(4)
		.rposition(|bytes| bytes == b"PK\x05\x06")
		.unwrap();
	commented[actual_eocd + 20..actual_eocd + 22].copy_from_slice(&22_u16.to_le_bytes());
	let mut false_eocd = [0_u8; 22];
	false_eocd[..4].copy_from_slice(b"PK\x05\x06");
	false_eocd[8..10].copy_from_slice(&1_u16.to_le_bytes());
	false_eocd[10..12].copy_from_slice(&1_u16.to_le_bytes());
	false_eocd[12..16].copy_from_slice(&46_u32.to_le_bytes());
	false_eocd[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
	commented.extend_from_slice(&false_eocd);
	let mut archive = Archive::from_bytes(&commented).unwrap();
	assert_eq!(archive.read("commented.txt").unwrap().as_slice(), b"safe");
}

#[test]
fn extended_timestamp_and_extra_field_padding_are_observable_and_accepted() {
	let timestamp = 1_700_000_123_u32;
	let mut field = Vec::new();
	field.extend_from_slice(&0x5455_u16.to_le_bytes());
	field.extend_from_slice(&5_u16.to_le_bytes());
	field.push(1);
	field.extend_from_slice(&timestamp.to_le_bytes());
	field.push(0);
	let bytes = fixture_with_central_extra(b"dated.txt", b"then", &field);
	let mut archive = Archive::from_bytes(&bytes).unwrap();
	assert_eq!(
		archive.entry("dated.txt").unwrap().modified_unix_seconds(),
		Some(u64::from(timestamp))
	);
	assert_eq!(archive.read("dated.txt").unwrap().as_slice(), b"then");
}

fn fixture_with_central_extra(name: &[u8], data: &[u8], extra: &[u8]) -> Vec<u8> {
	let mut bytes = fixture(&[Member::stored(name, data)]);
	let central = bytes
		.windows(4)
		.position(|bytes| bytes == b"PK\x01\x02")
		.unwrap();
	let name_len = usize::from(u16::from_le_bytes([bytes[central + 28], bytes[central + 29]]));
	bytes[central + 30..central + 32]
		.copy_from_slice(&u16::try_from(extra.len()).unwrap().to_le_bytes());
	bytes.splice(central + 46 + name_len..central + 46 + name_len, extra.iter().copied());

	let eocd = bytes
		.windows(4)
		.rposition(|bytes| bytes == b"PK\x05\x06")
		.unwrap();
	let old_size = u32::from_le_bytes(bytes[eocd + 12..eocd + 16].try_into().unwrap());
	bytes[eocd + 12..eocd + 16]
		.copy_from_slice(&(old_size + u32::try_from(extra.len()).unwrap()).to_le_bytes());
	bytes
}
