//! Synthetic PowerPoint 97-2003 conversion fixtures.

use std::path::Path;

use omp_tools::read::markit;

const SECTOR_LEN: usize = 512;
const STREAM_LEN: usize = 4096;
const FREE_SECTOR: u32 = 0xffff_ffff;
const END_OF_CHAIN: u32 = 0xffff_fffe;
const FAT_SECTOR: u32 = 0xffff_fffd;

#[test]
fn ppt_preserves_slide_order_and_pairs_speaker_notes() {
	let bytes = presentation(false);
	let conversion = markit::convert(Path::new("ordered.ppt"), &bytes)
		.expect("PPT conversion succeeds")
		.expect("PPT is supported");

	assert_eq!(
		conversion.text.as_str(),
		"## First Slide\n\nAlpha point\n\nBox one\n\n> Speaker one\n\n## Second Slide\n\nBeta \
		 point\n"
	);
	assert!(!conversion.text.contains("macro payload"));
}

#[test]
fn ppt_reports_malformed_ole_without_panicking() {
	let error = markit::convert(Path::new("broken.ppt"), b"not an OLE compound file")
		.expect_err("malformed OLE must fail conversion");
	assert_eq!(error.format(), "ppt");
	assert!(error.message().contains("not an OLE2 compound file"));
}

#[test]
fn ppt_reports_current_user_encryption() {
	let error = markit::convert(Path::new("encrypted.ppt"), &presentation(true))
		.expect_err("encrypted PPT must fail conversion");
	assert_eq!(error.format(), "ppt");
	assert_eq!(error.message(), "document is encrypted");
}

#[test]
fn ppt_skips_a_truncated_record_length_without_allocating_it() {
	let mut stream = vec![0u8; 8];
	put_u32(&mut stream, 4, u32::MAX);
	let bytes = compound_file(&stream, &[0; 20]);
	let conversion = markit::convert(Path::new("truncated.ppt"), &bytes)
		.expect("truncated record is skipped safely")
		.expect("PPT is supported");
	assert!(conversion.text.is_empty());
}

/// Build a compact, persist-resolved binary presentation. The record layout
/// follows [MS-PPT], while `compound_file` emits the minimum CFB v3 container
/// needed for the two required streams.
fn presentation(encrypted: bool) -> Vec<u8> {
	let slide_list = container(
		0x0ff0,
		0,
		&[
			slide_persist(2, 101),
			text_shape(0, "First Slide"),
			text_shape(1, "Alpha point"),
			slide_persist(3, 102),
			text_shape(0, "Second Slide"),
			text_shape(1, "Beta point"),
		]
		.concat(),
	);
	let notes_list = container(0x0ff0, 2, &slide_persist(4, 0));
	let document = container(0x03e8, 0, &[slide_list, notes_list].concat());
	let slide_one = container(
		0x03ee,
		0,
		&[
			slide_atom(),
			record(0, 0x7777, b"macro payload must stay ignored"),
			text_shape(1, "Box one"),
		]
		.concat(),
	);
	let slide_two = container(0x03ee, 0, &slide_atom());
	let mut notes_atom = vec![0; 8];
	put_u32(&mut notes_atom, 0, 101);
	let notes = container(
		0x03f0,
		0,
		&[record(1, 0x03f1, &notes_atom), text_shape(1, "Speaker one")].concat(),
	);

	let mut powerpoint = record(0, 0, &[]);
	let edit_offset = powerpoint.len();
	powerpoint.extend(record(0, 0x0ff5, &[0; 28]));
	let directory_offset = powerpoint.len();
	powerpoint.extend(record(0, 0x1772, &[0; 20]));
	let document_offset = powerpoint.len();
	powerpoint.extend(document);
	let slide_one_offset = powerpoint.len();
	powerpoint.extend(slide_one);
	let slide_two_offset = powerpoint.len();
	powerpoint.extend(slide_two);
	let notes_offset = powerpoint.len();
	powerpoint.extend(notes);

	let edit_body = edit_offset + 8;
	put_u32(&mut powerpoint, edit_body + 12, directory_offset as u32);
	put_u32(&mut powerpoint, edit_body + 16, 1);
	put_u32(&mut powerpoint, edit_body + 20, 5);
	let directory_body = directory_offset + 8;
	put_u32(&mut powerpoint, directory_body, (4 << 20) | 1);
	for (index, offset) in [document_offset, slide_one_offset, slide_two_offset, notes_offset]
		.into_iter()
		.enumerate()
	{
		put_u32(&mut powerpoint, directory_body + 4 + index * 4, offset as u32);
	}

	let mut current_user_body = vec![0; 12];
	put_u32(&mut current_user_body, 0, 20);
	put_u32(&mut current_user_body, 4, if encrypted { 0xf3d1_c4df } else { 0xe391_c05f });
	put_u32(&mut current_user_body, 8, edit_offset as u32);
	compound_file(&powerpoint, &record(0, 0x0ff6, &current_user_body))
}

fn slide_persist(persist_id: u32, slide_id: u32) -> Vec<u8> {
	let mut body = vec![0; 20];
	put_u32(&mut body, 0, persist_id);
	put_u32(&mut body, 12, slide_id);
	record(0, 0x03f3, &body)
}

fn slide_atom() -> Vec<u8> {
	record(2, 0x03ef, &[0; 24])
}

fn text_shape(text_type: u8, text: &str) -> Vec<u8> {
	let header = record(0, 0x0f9f, &[text_type, 0, 0, 0]);
	let utf16 = text
		.encode_utf16()
		.flat_map(u16::to_le_bytes)
		.collect::<Vec<_>>();
	[header, record(0, 0x0fa0, &utf16)].concat()
}

fn container(record_type: u16, instance: u16, body: &[u8]) -> Vec<u8> {
	record((instance << 4) | 0x000f, record_type, body)
}

fn record(version_and_instance: u16, record_type: u16, body: &[u8]) -> Vec<u8> {
	let mut bytes = Vec::with_capacity(8 + body.len());
	bytes.extend(version_and_instance.to_le_bytes());
	bytes.extend(record_type.to_le_bytes());
	bytes.extend((body.len() as u32).to_le_bytes());
	bytes.extend(body);
	bytes
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
	bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn compound_file(powerpoint: &[u8], current_user: &[u8]) -> Vec<u8> {
	assert!(powerpoint.len() <= STREAM_LEN);
	assert!(current_user.len() <= STREAM_LEN);

	// Header + 18 sectors: two eight-sector streams, directory, and FAT.
	let mut bytes = vec![0; SECTOR_LEN * 19];
	bytes[..8].copy_from_slice(&[0xd0, 0xcf, 0x11, 0xe0, 0xa1, 0xb1, 0x1a, 0xe1]);
	put_u16(&mut bytes, 24, 0x003e);
	put_u16(&mut bytes, 26, 3);
	put_u16(&mut bytes, 28, 0xfffe);
	put_u16(&mut bytes, 30, 9);
	put_u16(&mut bytes, 32, 6);
	put_u32(&mut bytes, 44, 1);
	put_u32(&mut bytes, 48, 16);
	put_u32(&mut bytes, 56, STREAM_LEN as u32);
	put_u32(&mut bytes, 60, END_OF_CHAIN);
	put_u32(&mut bytes, 68, END_OF_CHAIN);
	put_u32(&mut bytes, 76, 17);
	for offset in (80..512).step_by(4) {
		put_u32(&mut bytes, offset, FREE_SECTOR);
	}

	let powerpoint_start = sector_offset(0);
	bytes[powerpoint_start..powerpoint_start + powerpoint.len()].copy_from_slice(powerpoint);
	let current_user_start = sector_offset(8);
	bytes[current_user_start..current_user_start + current_user.len()].copy_from_slice(current_user);

	let directory = sector_offset(16);
	write_directory_entry(
		&mut bytes[directory..directory + 128],
		"Root Entry",
		5,
		u32::MAX,
		END_OF_CHAIN,
		0,
	);
	write_directory_entry(
		&mut bytes[directory + 128..directory + 256],
		"Current User",
		2,
		2,
		8,
		STREAM_LEN as u64,
	);
	write_directory_entry(
		&mut bytes[directory + 256..directory + 384],
		"PowerPoint Document",
		2,
		u32::MAX,
		0,
		STREAM_LEN as u64,
	);

	let fat = sector_offset(17);
	for index in 0..128 {
		put_u32(&mut bytes, fat + index * 4, FREE_SECTOR);
	}
	for sector in 0..7 {
		put_u32(&mut bytes, fat + sector * 4, (sector + 1) as u32);
	}
	put_u32(&mut bytes, fat + 7 * 4, END_OF_CHAIN);
	for sector in 8..15 {
		put_u32(&mut bytes, fat + sector * 4, (sector + 1) as u32);
	}
	put_u32(&mut bytes, fat + 15 * 4, END_OF_CHAIN);
	put_u32(&mut bytes, fat + 16 * 4, END_OF_CHAIN);
	put_u32(&mut bytes, fat + 17 * 4, FAT_SECTOR);
	bytes
}

fn write_directory_entry(
	entry: &mut [u8],
	name: &str,
	object_type: u8,
	right_sibling: u32,
	start_sector: u32,
	stream_len: u64,
) {
	let mut name_units = name.encode_utf16().collect::<Vec<_>>();
	name_units.push(0);
	for (index, unit) in name_units.iter().enumerate() {
		entry[index * 2..index * 2 + 2].copy_from_slice(&unit.to_le_bytes());
	}
	put_u16(entry, 64, (name_units.len() * 2) as u16);
	entry[66] = object_type;
	entry[67] = 1;
	put_u32(entry, 68, u32::MAX);
	put_u32(entry, 72, right_sibling);
	put_u32(entry, 76, if object_type == 5 { 1 } else { u32::MAX });
	put_u32(entry, 116, start_sector);
	entry[120..128].copy_from_slice(&stream_len.to_le_bytes());
}

fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
	bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn sector_offset(sector: usize) -> usize {
	SECTOR_LEN + sector * SECTOR_LEN
}
