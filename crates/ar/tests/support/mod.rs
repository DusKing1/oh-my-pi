//! Shared binary fixture builders for the integration-test crates.

#![allow(dead_code, reason = "each integration crate uses a subset of these fixture builders")]

pub mod tar;

pub struct Member<'a> {
	pub name:   &'a [u8],
	pub data:   &'a [u8],
	pub flags:  u16,
	pub method: u16,
	pub crc32:  Option<u32>,
}

impl<'a> Member<'a> {
	pub const fn stored(name: &'a [u8], data: &'a [u8]) -> Self {
		Self { name, data, flags: 0, method: 0, crc32: None }
	}
}

pub fn fixture(members: &[Member<'_>]) -> Vec<u8> {
	let mut zip = Vec::new();
	let mut records = Vec::with_capacity(members.len());
	for member in members {
		let offset = u32::try_from(zip.len()).unwrap();
		let size = u32::try_from(member.data.len()).unwrap();
		let crc32 = member.crc32.unwrap_or_else(|| crc32fast::hash(member.data));
		push_u32(&mut zip, 0x0403_4b50);
		push_u16(&mut zip, 20);
		push_u16(&mut zip, member.flags);
		push_u16(&mut zip, member.method);
		push_u16(&mut zip, 0);
		push_u16(&mut zip, 0x21);
		push_u32(&mut zip, crc32);
		push_u32(&mut zip, size);
		push_u32(&mut zip, size);
		push_u16(&mut zip, u16::try_from(member.name.len()).unwrap());
		push_u16(&mut zip, 0);
		zip.extend_from_slice(member.name);
		zip.extend_from_slice(member.data);
		records.push((member, offset, size, crc32));
	}

	let central_offset = u32::try_from(zip.len()).unwrap();
	for (member, offset, size, crc32) in records {
		push_u32(&mut zip, 0x0201_4b50);
		push_u16(&mut zip, 20);
		push_u16(&mut zip, 20);
		push_u16(&mut zip, member.flags);
		push_u16(&mut zip, member.method);
		push_u16(&mut zip, 0);
		push_u16(&mut zip, 0x21);
		push_u32(&mut zip, crc32);
		push_u32(&mut zip, size);
		push_u32(&mut zip, size);
		push_u16(&mut zip, u16::try_from(member.name.len()).unwrap());
		push_u16(&mut zip, 0);
		push_u16(&mut zip, 0);
		push_u16(&mut zip, 0);
		push_u16(&mut zip, 0);
		push_u32(&mut zip, 0);
		push_u32(&mut zip, offset);
		zip.extend_from_slice(member.name);
	}
	let central_size = u32::try_from(zip.len()).unwrap() - central_offset;
	push_eocd(&mut zip, u16::try_from(members.len()).unwrap(), central_size, central_offset);
	zip
}

pub fn zip64_fixture(name: &[u8], data: &[u8]) -> Vec<u8> {
	let mut zip = Vec::new();
	let size = u32::try_from(data.len()).unwrap();
	let crc32 = crc32fast::hash(data);
	push_u32(&mut zip, 0x0403_4b50);
	push_u16(&mut zip, 45);
	push_u16(&mut zip, 0x0800);
	push_u16(&mut zip, 0);
	push_u16(&mut zip, 0);
	push_u16(&mut zip, 0x21);
	push_u32(&mut zip, crc32);
	push_u32(&mut zip, size);
	push_u32(&mut zip, size);
	push_u16(&mut zip, u16::try_from(name.len()).unwrap());
	push_u16(&mut zip, 0);
	zip.extend_from_slice(name);
	zip.extend_from_slice(data);

	let central_offset = u64::try_from(zip.len()).unwrap();
	push_u32(&mut zip, 0x0201_4b50);
	push_u16(&mut zip, 45);
	push_u16(&mut zip, 45);
	push_u16(&mut zip, 0x0800);
	push_u16(&mut zip, 0);
	push_u16(&mut zip, 0);
	push_u16(&mut zip, 0x21);
	push_u32(&mut zip, crc32);
	push_u32(&mut zip, u32::MAX);
	push_u32(&mut zip, u32::MAX);
	push_u16(&mut zip, u16::try_from(name.len()).unwrap());
	push_u16(&mut zip, 28);
	push_u16(&mut zip, 0);
	push_u16(&mut zip, 0);
	push_u16(&mut zip, 0);
	push_u32(&mut zip, 0);
	push_u32(&mut zip, u32::MAX);
	zip.extend_from_slice(name);
	push_u16(&mut zip, 0x0001);
	push_u16(&mut zip, 24);
	push_u64(&mut zip, u64::from(size));
	push_u64(&mut zip, u64::from(size));
	push_u64(&mut zip, 0);
	let central_size = u64::try_from(zip.len()).unwrap() - central_offset;

	let zip64_eocd_offset = u64::try_from(zip.len()).unwrap();
	push_u32(&mut zip, 0x0606_4b50);
	push_u64(&mut zip, 44);
	push_u16(&mut zip, 45);
	push_u16(&mut zip, 45);
	push_u32(&mut zip, 0);
	push_u32(&mut zip, 0);
	push_u64(&mut zip, 1);
	push_u64(&mut zip, 1);
	push_u64(&mut zip, central_size);
	push_u64(&mut zip, central_offset);
	push_u32(&mut zip, 0x0706_4b50);
	push_u32(&mut zip, 0);
	push_u64(&mut zip, zip64_eocd_offset);
	push_u32(&mut zip, 1);
	push_u32(&mut zip, 0x0605_4b50);
	push_u16(&mut zip, 0);
	push_u16(&mut zip, 0);
	push_u16(&mut zip, u16::MAX);
	push_u16(&mut zip, u16::MAX);
	push_u32(&mut zip, u32::MAX);
	push_u32(&mut zip, u32::MAX);
	push_u16(&mut zip, 0);
	zip
}

pub fn assert_error_kind<T>(
	result: omp_ar::Result<T>,
	predicate: impl FnOnce(&omp_ar::Error) -> bool,
) {
	match result {
		Err(error) if predicate(&error) => {},
		Err(error) => panic!("unexpected archive error: {error:?}"),
		Ok(_) => panic!("operation unexpectedly succeeded"),
	}
}

fn push_eocd(zip: &mut Vec<u8>, entries: u16, central_size: u32, central_offset: u32) {
	push_u32(zip, 0x0605_4b50);
	push_u16(zip, 0);
	push_u16(zip, 0);
	push_u16(zip, entries);
	push_u16(zip, entries);
	push_u32(zip, central_size);
	push_u32(zip, central_offset);
	push_u16(zip, 0);
}

fn push_u16(output: &mut Vec<u8>, value: u16) {
	output.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(output: &mut Vec<u8>, value: u32) {
	output.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(output: &mut Vec<u8>, value: u64) {
	output.extend_from_slice(&value.to_le_bytes());
}
