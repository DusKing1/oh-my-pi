//! Deterministic TAR fixtures built independently of `omp_ar`.

use std::io::Write;

use flate2::{Compression, GzBuilder};

const BLOCK_SIZE: usize = 512;
const MTIME: u64 = 1_700_000_000;

#[derive(Clone, Copy)]
pub struct TarMember<'a> {
	pub path:      &'a str,
	pub data:      &'a [u8],
	pub kind:      u8,
	pub link_name: Option<&'a str>,
	pub prefix:    Option<&'a str>,
}

impl<'a> TarMember<'a> {
	pub const fn file(path: &'a str, data: &'a [u8]) -> Self {
		Self { path, data, kind: b'0', link_name: None, prefix: None }
	}

	pub const fn hard_link(path: &'a str, target: &'a str) -> Self {
		Self { path, data: b"", kind: b'1', link_name: Some(target), prefix: None }
	}

	pub const fn symlink(path: &'a str, target: &'a str) -> Self {
		Self { path, data: b"", kind: b'2', link_name: Some(target), prefix: None }
	}

	pub const fn metadata(kind: u8, data: &'a [u8]) -> Self {
		Self { path: "././@Meta", data, kind, link_name: None, prefix: None }
	}

	pub const fn with_prefix(mut self, prefix: &'a str) -> Self {
		self.prefix = Some(prefix);
		self
	}
}

pub fn fixture(members: &[TarMember<'_>]) -> Vec<u8> {
	let mut output = Vec::new();
	for member in members {
		append_member(&mut output, *member);
	}
	output.resize(output.len() + 2 * BLOCK_SIZE, 0);
	output
}

pub fn v7_fixture(member: TarMember<'_>) -> Vec<u8> {
	let mut output = fixture(&[member]);
	output[257..265].fill(0);
	let header: &mut [u8; BLOCK_SIZE] = (&mut output[..BLOCK_SIZE]).try_into().unwrap();
	checksum(header);
	output
}

pub fn gzip_fixture(members: &[TarMember<'_>]) -> Vec<u8> {
	gzip_bytes(&fixture(members))
}

pub fn gzip_bytes(bytes: &[u8]) -> Vec<u8> {
	let mut encoder = GzBuilder::new()
		.mtime(0)
		.write(Vec::new(), Compression::default());
	encoder.write_all(bytes).unwrap();
	encoder.finish().unwrap()
}

pub fn pax_fixture<K: AsRef<str>, V: AsRef<str>>(
	records: &[(K, V)],
	member: TarMember<'_>,
) -> Vec<u8> {
	let body = pax_records(records);

	let mut output = Vec::new();
	let header = header("PaxHeaders/entry", body.len() as u64, b'x', None, None);
	output.extend_from_slice(&header);
	append_data(&mut output, &body);
	append_member(&mut output, member);
	output.resize(output.len() + 2 * BLOCK_SIZE, 0);
	output
}
pub fn pax_records<K: AsRef<str>, V: AsRef<str>>(records: &[(K, V)]) -> Vec<u8> {
	let mut body = Vec::new();
	for (key, value) in records {
		body.extend_from_slice(&pax_record(key.as_ref(), value.as_ref()));
	}
	body
}

pub fn old_gnu_sparse_fixture() -> Vec<u8> {
	let stored = b"sparse-extent\n";
	let mut sparse_header = header("data/old-sparse.bin", stored.len() as u64, b'S', None, None);
	// Old-GNU headers put the continuation flag at byte 482. The final
	// continuation block uses byte 504 and leaves it clear.
	sparse_header[482] = 1;
	checksum(&mut sparse_header);

	let mut output = Vec::new();
	output.extend_from_slice(&sparse_header);
	output.resize(output.len() + BLOCK_SIZE, 0);
	append_data(&mut output, stored);
	append_member(&mut output, TarMember::file("data/after.txt", b"after sparse\n"));
	output.resize(output.len() + 2 * BLOCK_SIZE, 0);
	output
}

fn append_member(output: &mut Vec<u8>, member: TarMember<'_>) {
	let header =
		header(member.path, member.data.len() as u64, member.kind, member.link_name, member.prefix);
	output.extend_from_slice(&header);
	append_data(output, member.data);
}

fn append_data(output: &mut Vec<u8>, data: &[u8]) {
	output.extend_from_slice(data);
	let remainder = data.len() % BLOCK_SIZE;
	if remainder != 0 {
		output.resize(output.len() + BLOCK_SIZE - remainder, 0);
	}
}

fn header(
	name: &str,
	size: u64,
	kind: u8,
	link_name: Option<&str>,
	prefix: Option<&str>,
) -> [u8; BLOCK_SIZE] {
	let mut header = [0_u8; BLOCK_SIZE];
	write_string(&mut header[0..100], name);
	write_octal(&mut header[100..108], 0o644);
	write_octal(&mut header[108..116], 0);
	write_octal(&mut header[116..124], 0);
	write_octal(&mut header[124..136], size);
	write_octal(&mut header[136..148], MTIME);
	header[156] = kind;
	if let Some(link_name) = link_name {
		write_string(&mut header[157..257], link_name);
	}
	write_string(&mut header[257..263], "ustar");
	write_string(&mut header[263..265], "00");
	if let Some(prefix) = prefix {
		write_string(&mut header[345..500], prefix);
	}
	checksum(&mut header);
	header
}

fn checksum(header: &mut [u8; BLOCK_SIZE]) {
	header[148..156].fill(b' ');
	let sum: u64 = header.iter().map(|byte| u64::from(*byte)).sum();
	let encoded = format!("{sum:06o}");
	header[148..154].copy_from_slice(encoded.as_bytes());
	header[154] = 0;
	header[155] = b' ';
}

fn write_string(field: &mut [u8], value: &str) {
	let bytes = value.as_bytes();
	assert!(bytes.len() <= field.len(), "fixture value exceeds TAR field");
	field[..bytes.len()].copy_from_slice(bytes);
}

fn write_octal(field: &mut [u8], value: u64) {
	let digits = field.len() - 1;
	let encoded = format!("{value:0digits$o}");
	assert_eq!(encoded.len(), digits, "fixture number exceeds TAR field");
	field[..digits].copy_from_slice(encoded.as_bytes());
}

fn pax_record(key: &str, value: &str) -> Vec<u8> {
	let suffix = format!(" {key}={value}\n");
	let mut length = suffix.len() + 1;
	loop {
		let corrected = length.to_string().len() + suffix.len();
		if corrected == length {
			return format!("{length}{suffix}").into_bytes();
		}
		length = corrected;
	}
}
