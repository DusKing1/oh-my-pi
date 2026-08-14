//! Word 97-2003 binary conversion contracts.

use std::path::Path;

use omp_tools::read::markit;

const END_OF_CHAIN: u32 = 0xffff_fffe;
const FAT_SECTOR: u32 = 0xffff_fffd;
const FREE_SECTOR: u32 = 0xffff_ffff;

fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
	bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
	bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
	bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn directory_entry(
	name: &str,
	object_type: u8,
	right_sibling: u32,
	child: u32,
	start_sector: u32,
	stream_size: u64,
) -> [u8; 128] {
	let mut entry = [0; 128];
	let mut name_units = name.encode_utf16();
	let mut unit_count = 0usize;
	for (index, unit) in name_units.by_ref().enumerate() {
		entry[index * 2..index * 2 + 2].copy_from_slice(&unit.to_le_bytes());
		unit_count = index + 1;
	}
	put_u16(&mut entry, 64, ((unit_count + 1) * 2) as u16);
	entry[66] = object_type;
	entry[67] = 1;
	put_u32(&mut entry, 68, FREE_SECTOR);
	put_u32(&mut entry, 72, right_sibling);
	put_u32(&mut entry, 76, child);
	put_u32(&mut entry, 116, start_sector);
	put_u64(&mut entry, 120, stream_size);
	entry
}

/// Build the smallest regular-FAT CFB v3 file holding `WordDocument`.
///
/// This synthetic fixture follows the MIT-licensed anydoc fixture generator's
/// `write_cfb` layout. Padding to the 4096-byte cutoff keeps the stream out of
/// the mini-FAT and makes every sector relationship explicit here.
fn compound_word_document(mut word_document: Vec<u8>) -> Vec<u8> {
	word_document.resize(4096, 0);

	let mut header = [0; 512];
	header[..8].copy_from_slice(&[0xd0, 0xcf, 0x11, 0xe0, 0xa1, 0xb1, 0x1a, 0xe1]);
	put_u16(&mut header, 24, 0x003e);
	put_u16(&mut header, 26, 0x0003);
	put_u16(&mut header, 28, 0xfffe);
	put_u16(&mut header, 30, 9);
	put_u16(&mut header, 32, 6);
	put_u32(&mut header, 44, 1);
	put_u32(&mut header, 48, 8);
	put_u32(&mut header, 56, 4096);
	put_u32(&mut header, 60, END_OF_CHAIN);
	put_u32(&mut header, 68, END_OF_CHAIN);
	put_u32(&mut header, 76, 9);
	for offset in (80..512).step_by(4) {
		put_u32(&mut header, offset, FREE_SECTOR);
	}

	let mut directory = [0; 512];
	directory[..128].copy_from_slice(&directory_entry(
		"Root Entry",
		5,
		FREE_SECTOR,
		1,
		END_OF_CHAIN,
		0,
	));
	directory[128..256].copy_from_slice(&directory_entry(
		"WordDocument",
		2,
		FREE_SECTOR,
		FREE_SECTOR,
		0,
		4096,
	));

	let mut fat = [FREE_SECTOR; 128];
	for (sector, next) in fat.iter_mut().take(7).enumerate() {
		*next = sector as u32 + 1;
	}
	fat[7] = END_OF_CHAIN;
	fat[8] = END_OF_CHAIN;
	fat[9] = FAT_SECTOR;

	let mut compound = Vec::with_capacity(5632);
	compound.extend_from_slice(&header);
	compound.extend_from_slice(&word_document);
	compound.extend_from_slice(&directory);
	for entry in fat {
		compound.extend_from_slice(&entry.to_le_bytes());
	}
	compound
}

fn binary_word(text: &[u8], encrypted: bool) -> Vec<u8> {
	let mut stream = vec![0; 0x400];
	put_u16(&mut stream, 0, 0xa5ec);
	put_u16(&mut stream, 2, 0x00c1);
	put_u16(&mut stream, 6, 0x0419); // Russian: Windows-1251.
	put_u16(&mut stream, 0x0a, if encrypted { 0x0100 } else { 0 });
	put_u32(&mut stream, 0x18, 0x400);
	put_u32(&mut stream, 0x1c, 0x400 + text.len() as u32);
	put_u32(&mut stream, 0x4c, text.len() as u32);
	stream.extend_from_slice(text);
	compound_word_document(stream)
}

#[test]
fn doc_dispatch_decodes_binary_word_codepage_and_paragraphs() {
	let bytes = binary_word(
		b"\xcf\xf0\xe8\xe2\xe5\xf2, \xec\xe8\xf0!\r\xc2\xf2\xee\xf0\xee\xe9 \xe0\xe1\xe7\xe0\xf6 \xef\xee-\xf0\xf3\xf1\xf1\xea\xe8.\r",
		false,
	);
	let conversion = markit::convert(Path::new("report.doc"), &bytes)
		.expect("valid binary Word document converts")
		.expect("doc extension is recognized");

	assert_eq!(conversion.text.as_str(), "Привет, мир!\n\nВторой абзац по-русски.\n");
	assert_eq!(conversion.note, None);
	assert_eq!(conversion.title, None);
}

#[test]
fn malformed_doc_is_a_truthful_doc_conversion_error() {
	let error = markit::convert(Path::new("broken.doc"), b"not an OLE compound file")
		.expect_err("malformed OLE input must fail");
	assert_eq!(error.format(), "doc");
	assert!(error.message().contains("not an OLE2 compound file"), "{error}");
}

#[test]
fn encrypted_doc_is_rejected_without_processing_payloads() {
	let bytes = binary_word(b"secret\r", true);
	let error = markit::convert(Path::new("encrypted.doc"), &bytes)
		.expect_err("encrypted binary Word input must fail");
	assert_eq!(error.format(), "doc");
	assert!(error.message().contains("encrypted"), "{error}");
}
