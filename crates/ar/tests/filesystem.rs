//! File-backed reading and capability-scoped extraction contracts.

mod support;

use std::io::Write;

use cap_std::{ambient_authority, fs::Dir};
use omp_ar::{Archive, Error};
use support::{Member, assert_error_kind, fixture};
use tempfile::{NamedTempFile, tempdir};

#[test]
fn open_indexes_a_file_backed_archive_and_reads_members_on_demand() {
	let bytes = fixture(&[Member::stored(b"disk.txt", b"file-backed"), Member {
		name:   b"corrupt.txt",
		data:   b"deferred failure",
		flags:  0x0800,
		method: 0,
		crc32:  Some(0),
	}]);
	let mut source = NamedTempFile::new().unwrap();
	source.write_all(&bytes).unwrap();
	source.flush().unwrap();

	let mut archive = Archive::open(source.path()).unwrap();
	assert_eq!(archive.entry("disk.txt").unwrap().size(), 11);
	assert_eq!(archive.read("disk.txt").unwrap().as_slice(), b"file-backed");
	assert_error_kind(archive.read("corrupt.txt"), |error| {
		matches!(error, Error::ChecksumMismatch { .. })
	});
}

#[test]
fn extraction_stays_beneath_a_capability_directory_and_preserves_empty_directories() {
	let bytes = fixture(&[
		Member::stored(b"empty/", b""),
		Member::stored(b"nested/one.txt", b"one"),
		Member::stored(b"nested/two.txt", b"two"),
	]);
	let mut archive = Archive::from_bytes(&bytes).unwrap();
	let destination = tempdir().unwrap();
	let directory = Dir::open_ambient_dir(destination.path(), ambient_authority()).unwrap();

	assert_eq!(archive.extract_to(&directory).unwrap(), 2);
	assert!(destination.path().join("empty").is_dir());
	assert_eq!(
		std::fs::read(destination.path().join("nested/one.txt"))
			.unwrap()
			.as_slice(),
		b"one"
	);
	assert_eq!(
		std::fs::read(destination.path().join("nested/two.txt"))
			.unwrap()
			.as_slice(),
		b"two"
	);
}
