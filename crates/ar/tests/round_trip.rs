//! Deterministic writer and lazy-reader round-trip contracts.

use omp_ar::{
	Archive,
	zip::{CompressionMethod, Writer},
};

fn deterministic_noise(len: usize) -> Vec<u8> {
	let mut state = 0x6d2b_79f5_u32;
	(0..len)
		.map(|_| {
			state ^= state << 13;
			state ^= state >> 17;
			state ^= state << 5;
			state as u8
		})
		.collect()
}

fn nested_archive(random: &[u8]) -> Vec<u8> {
	let mut writer = Writer::new(Vec::new());
	writer.add_directory("tree/empty").unwrap();
	writer
		.add_file("tree/repeat.txt", &vec![b'A'; 4096])
		.unwrap();
	writer.add_file("tree/random.bin", random).unwrap();
	writer.add_file("tree/deep/note.txt", b"nested").unwrap();
	writer.finish().unwrap()
}

#[test]
fn nested_round_trip_is_deterministic_and_chooses_the_smaller_encoding() {
	let random = deterministic_noise(1024);
	let first = nested_archive(&random);
	let second = nested_archive(&random);
	assert_eq!(first, second);

	let mut archive = Archive::from_bytes(&first).unwrap();
	assert_eq!(
		archive.entry("tree/repeat.txt").unwrap().zip_compression(),
		Some(CompressionMethod::Deflate)
	);
	assert_eq!(
		archive.entry("tree/random.bin").unwrap().zip_compression(),
		Some(CompressionMethod::Stored)
	);
	assert_eq!(archive.read("tree/repeat.txt").unwrap(), vec![b'A'; 4096]);
	assert_eq!(archive.read("tree/random.bin").unwrap(), random);
	assert_eq!(archive.read("tree/deep/note.txt").unwrap().as_slice(), b"nested");

	let files = archive.read_all().unwrap();
	assert_eq!(files.get("tree/random.bin").unwrap(), &random);
	assert_eq!(files.get("tree/deep/note.txt").unwrap().as_slice(), b"nested");
}

#[test]
fn listing_is_direct_and_missing_parents_are_synthesized() {
	let random = deterministic_noise(64);
	let bytes = nested_archive(&random);
	let archive = Archive::from_bytes(&bytes).unwrap();

	let paths: Vec<_> = archive.entries().map(|entry| entry.path()).collect();
	assert_eq!(paths, [
		"tree",
		"tree/deep",
		"tree/deep/note.txt",
		"tree/empty",
		"tree/random.bin",
		"tree/repeat.txt",
	]);
	assert!(archive.entry("tree").unwrap().is_directory());
	assert!(archive.entry("tree/deep").unwrap().is_directory());

	let root: Vec<_> = archive
		.list("")
		.unwrap()
		.into_iter()
		.map(|entry| entry.path())
		.collect();
	assert_eq!(root, ["tree"]);
	let tree: Vec<_> = archive
		.list("tree")
		.unwrap()
		.into_iter()
		.map(|entry| entry.path())
		.collect();
	assert_eq!(tree, ["tree/deep", "tree/empty", "tree/random.bin", "tree/repeat.txt"]);
}
