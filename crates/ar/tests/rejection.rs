//! Malformed-input and resource-limit rejection contracts.

mod support;

use omp_ar::{Archive, Error, Limits};
use support::{Member, assert_error_kind, fixture};

#[test]
fn corrupted_stored_member_is_rejected_by_crc() {
	let bytes = fixture(&[Member {
		name:   b"payload.txt",
		data:   b"content changed",
		flags:  0x0800,
		method: 0,
		crc32:  Some(0x1234_5678),
	}]);
	let mut archive = Archive::from_bytes(&bytes).unwrap();

	assert_error_kind(archive.read("payload.txt"), |error| {
		matches!(error, Error::ChecksumMismatch { .. })
	});
}

#[test]
fn unsafe_central_paths_are_omitted_without_hiding_safe_members() {
	let bytes = fixture(&[
		Member::stored(b"../escape.txt", b"escape"),
		Member::stored(b"/absolute.txt", b"absolute"),
		Member::stored(b"safe/../../outside.txt", b"outside"),
		Member::stored(b"safe/ok.txt", b"ok"),
	]);
	let mut archive = Archive::from_bytes(&bytes).unwrap();

	let paths: Vec<_> = archive.entries().map(|entry| entry.path()).collect();
	assert_eq!(paths, ["safe", "safe/ok.txt"]);
	assert!(archive.entry("../escape.txt").is_none());
	assert!(archive.entry("/absolute.txt").is_none());
	assert_eq!(archive.read("safe/ok.txt").unwrap().as_slice(), b"ok");
}

#[test]
fn encrypted_and_unsupported_members_are_indexed_but_rejected_on_read() {
	let bytes = fixture(&[
		Member {
			name:   b"secret.txt",
			data:   b"ciphertext",
			flags:  0x0801,
			method: 0,
			crc32:  None,
		},
		Member { name: b"legacy.bin", data: b"opaque", flags: 0x0800, method: 99, crc32: None },
	]);
	let mut archive = Archive::from_bytes(&bytes).unwrap();
	assert!(archive.entry("secret.txt").unwrap().is_encrypted());

	assert_error_kind(archive.read("secret.txt"), |error| matches!(error, Error::Encrypted(_)));
	assert_error_kind(archive.read("legacy.bin"), |error| {
		matches!(error, Error::UnsupportedCompression { method: 99, .. })
	});
}

#[test]
fn member_and_aggregate_materialization_limits_are_independent() {
	let bytes = fixture(&[Member::stored(b"alpha", b"12345"), Member::stored(b"beta", b"67890")]);

	let member_limits = Limits::DEFAULT.with_max_member_size(4);
	let mut member_limited = Archive::from_bytes_with_limits(&bytes, member_limits).unwrap();
	assert_error_kind(member_limited.read("alpha"), |error| {
		matches!(error, Error::MemberTooLarge { actual: 5, limit: 4, .. })
	});

	let aggregate_limits = Limits::DEFAULT.with_max_in_memory_size(9);
	let mut aggregate_limited = Archive::from_bytes_with_limits(&bytes, aggregate_limits).unwrap();
	assert_error_kind(aggregate_limited.read_all(), |error| {
		matches!(error, Error::ArchiveTooLargeInMemory { actual: 10, limit: 9 })
	});
	assert_eq!(aggregate_limited.read("alpha").unwrap().as_slice(), b"12345");
}

#[test]
fn maximum_length_zip_path_is_rejected_before_parent_synthesis() {
	let mut name = "a/".repeat(32_767);
	name.push('a');
	assert_eq!(name.len(), usize::from(u16::MAX));
	let bytes = fixture(&[Member::stored(name.as_bytes(), b"x")]);

	assert_error_kind(Archive::from_bytes(&bytes), |error| {
		matches!(error, Error::PathTooLong { actual: 65_535, limit: 4096 })
	});
}

#[test]
fn synthetic_directories_count_toward_the_index_limit() {
	let bytes = fixture(&[
		Member::stored(b"first/file.txt", b"one"),
		Member::stored(b"second/file.txt", b"two"),
	]);
	let limits = Limits::DEFAULT.with_max_entries(2);

	assert_error_kind(Archive::from_bytes_with_limits(&bytes, limits), |error| {
		matches!(error, Error::TooManyEntries { actual: 3, limit: 2 })
	});
}
