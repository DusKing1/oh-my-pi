//! TAR/TAR.GZ interoperability, alias, and malformed-input contracts.

mod support;

use std::{io::Write, path::Path};

use cap_std::{ambient_authority, fs::Dir};
use omp_ar::{Archive, Error, Format, Limits, tar};
use support::{
	Member, fixture as zip_fixture,
	tar::{
		TarMember, fixture, gzip_bytes, gzip_fixture, old_gnu_sparse_fixture, pax_fixture,
		pax_records, v7_fixture,
	},
};
use tempfile::tempdir;

fn assert_error_contains<T>(result: omp_ar::Result<T>, expected: &str) {
	match result {
		Err(error) => assert!(
			error
				.to_string()
				.to_ascii_lowercase()
				.contains(&expected.to_ascii_lowercase()),
			"expected error containing {expected:?}, got {error:?}",
		),
		Ok(_) => panic!("operation unexpectedly succeeded"),
	}
}
fn rewrite_header_checksum(block: &mut [u8]) {
	block[148..156].fill(b' ');
	let sum: u64 = block.iter().take(512).map(|byte| u64::from(*byte)).sum();
	let encoded = format!("{sum:06o}");
	block[148..154].copy_from_slice(encoded.as_bytes());
	block[154] = 0;
	block[155] = b' ';
}
fn rewrite_signed_header_checksum(block: &mut [u8]) {
	block[148..156].fill(b' ');
	let sum: i64 = block
		.iter()
		.take(512)
		.map(|byte| i64::from(*byte as i8))
		.sum();
	assert!(sum >= 0);
	let encoded = format!("{sum:06o}");
	block[148..154].copy_from_slice(encoded.as_bytes());
	block[154] = 0;
	block[155] = b' ';
}

fn rewrite_octal(field: &mut [u8], value: u64) {
	let digits = field.len() - 1;
	let encoded = format!("{value:0digits$o}");
	field[..digits].copy_from_slice(encoded.as_bytes());
	field[digits] = 0;
}

#[derive(Debug, Eq, PartialEq)]
struct RawTarRecord {
	path: Vec<u8>,
	kind: u8,
	link: Option<Vec<u8>>,
}

fn raw_tar_records(bytes: &[u8]) -> Vec<RawTarRecord> {
	const BLOCK: usize = 512;

	let mut records = Vec::new();
	let mut offset = 0;
	let mut long_name = None;
	let mut long_link = None;
	while let Some(header) = bytes.get(offset..offset + BLOCK) {
		if header.iter().all(|byte| *byte == 0) {
			break;
		}
		let size = tar_octal(&header[124..136]);
		let data_start = offset + BLOCK;
		let data_end = data_start + size;
		let payload = &bytes[data_start..data_end];
		let kind = header[156];
		match kind {
			b'L' => long_name = Some(tar_text(payload)),
			b'K' => long_link = Some(tar_text(payload)),
			_ => records.push(RawTarRecord {
				path: long_name.take().unwrap_or_else(|| tar_text(&header[..100])),
				kind,
				link: long_link.take().or_else(|| {
					let link = tar_text(&header[157..257]);
					(!link.is_empty()).then_some(link)
				}),
			}),
		}
		offset = data_start + size.div_ceil(BLOCK) * BLOCK;
	}
	records
}

fn tar_octal(field: &[u8]) -> usize {
	field
		.iter()
		.take_while(|byte| **byte != 0 && **byte != b' ')
		.fold(0_usize, |value, byte| value * 8 + usize::from(byte - b'0'))
}

fn tar_text(field: &[u8]) -> Vec<u8> {
	let end = field
		.iter()
		.position(|byte| *byte == 0)
		.unwrap_or(field.len());
	field[..end].to_vec()
}

#[test]
fn plain_and_gzip_tar_reads_preserve_unicode_ustar_prefixes() {
	let prefix = "bun-da3851e57ae130c5594d0e208a5da5ba8c13edfb/test/js/node/fixtures/新建文件夹";
	let path = format!("{prefix}/experimental.json");
	let member =
		TarMember::file("experimental.json", br#"{ "type": "module" }"#).with_prefix(prefix);

	for (bytes, format) in
		[(fixture(&[member]), Format::Tar), (gzip_fixture(&[member]), Format::TarGz)]
	{
		let mut archive = Archive::from_bytes(&bytes).unwrap();
		assert_eq!(archive.format(), format);
		assert_eq!(archive.read(&path).unwrap(), br#"{ "type": "module" }"#);
	}
}
#[test]
fn legacy_headers_ignore_tail_padding_and_sniff_without_ustar_magic() {
	let mut bytes = v7_fixture(TarMember::file("legacy.txt", b"legacy"));
	bytes[345..500].fill(b'p');
	rewrite_header_checksum(&mut bytes[..512]);

	assert_eq!(Format::sniff(&bytes), Some(Format::Tar));
	let mut archive = Archive::from_bytes(&bytes).unwrap();
	assert_eq!(archive.read("legacy.txt").unwrap(), b"legacy");
	assert!(
		archive
			.entry(&format!("{}/legacy.txt", "p".repeat(155)))
			.is_none()
	);
}

#[test]
fn negative_binary_mtime_does_not_reject_an_otherwise_readable_member() {
	let mut negative_time = fixture(&[TarMember::file("before-epoch.txt", b"old")]);
	negative_time[136..148].fill(0xff);
	rewrite_header_checksum(&mut negative_time[..512]);
	let mut archive = Archive::from_bytes(&negative_time).unwrap();
	assert_eq!(archive.read("before-epoch.txt").unwrap(), b"old");
	assert_eq!(
		archive
			.entry("before-epoch.txt")
			.unwrap()
			.modified_unix_seconds(),
		None
	);
}

#[test]
fn base256_numbers_and_signed_header_checksums_are_accepted() {
	let mut binary = fixture(&[TarMember::file("binary.txt", b"bin")]);
	binary[124..136].fill(0);
	binary[124] = 0x80;
	binary[135] = 3;
	binary[136..148].fill(0);
	binary[136] = 0x80;
	binary[147] = 42;
	rewrite_header_checksum(&mut binary[..512]);
	let mut archive = Archive::from_bytes(&binary).unwrap();
	assert_eq!(archive.read("binary.txt").unwrap(), b"bin");
	assert_eq!(archive.entry("binary.txt").unwrap().modified_unix_seconds(), Some(42));

	let mut signed = fixture(&[TarMember::file("signed.txt", b"signed")]);
	signed[500] = 0xff;
	rewrite_signed_header_checksum(&mut signed[..512]);
	let mut archive = Archive::from_bytes(&signed).unwrap();
	assert_eq!(archive.read("signed.txt").unwrap(), b"signed");
}

#[test]
fn special_nodes_and_unknown_typeflags_are_not_materialized_as_regular_files() {
	let special = [
		TarMember {
			path:      "character",
			data:      b"special",
			kind:      b'3',
			link_name: None,
			prefix:    None,
		},
		TarMember {
			path:      "block",
			data:      b"special",
			kind:      b'4',
			link_name: None,
			prefix:    None,
		},
		TarMember {
			path:      "fifo",
			data:      b"special",
			kind:      b'6',
			link_name: None,
			prefix:    None,
		},
		TarMember {
			path:      "volume",
			data:      b"special",
			kind:      b'V',
			link_name: None,
			prefix:    None,
		},
	];
	let mut members = special.to_vec();
	members.push(TarMember::file("after.txt", b"after"));
	let bytes = fixture(&members);
	let mut archive = Archive::from_bytes(&bytes).unwrap();
	assert_eq!(archive.read("after.txt").unwrap(), b"after");
	assert!(
		archive
			.entries()
			.all(|entry| { !matches!(entry.path(), "character" | "block" | "fifo" | "volume") })
	);
}

#[test]
fn hard_links_and_safe_relative_file_symlinks_materialize_target_bytes() {
	let bytes = fixture(&[
		TarMember::file("pkg/original.txt", b"shared content\n"),
		TarMember::hard_link("pkg/linked.txt", "pkg/original.txt"),
		TarMember::file("pkg/lib/tool.js", b"export const linked = true;\n"),
		TarMember::symlink("pkg/bin/tool", "../lib/tool.js"),
	]);
	let mut archive = Archive::from_bytes(&bytes).unwrap();

	assert_eq!(archive.read("pkg/linked.txt").unwrap(), b"shared content\n");
	assert_eq!(archive.read("pkg/bin/tool").unwrap(), b"export const linked = true;\n");
}

#[test]
fn directory_aliases_are_resolved_lazily_without_synthetic_subtrees() {
	let bytes = fixture(&[
		TarMember::file("pkg/lib/tool.js", b"tool\n"),
		TarMember::file("pkg/lib/extra.js", b"extra\n"),
		TarMember::symlink("pkg/current-a", "lib"),
		TarMember::symlink("pkg/current-b", "lib"),
	]);
	let mut archive = Archive::from_bytes(&bytes).unwrap();

	assert!(archive.entry("pkg/current-a/tool.js").is_none());
	assert_eq!(
		archive
			.resolve_entry("pkg/current-a/tool.js")
			.unwrap()
			.path(),
		"pkg/lib/tool.js"
	);
	assert_eq!(archive.read("pkg/current-a/tool.js").unwrap(), b"tool\n");
	let listed: Vec<_> = archive
		.list("pkg/current-b")
		.unwrap()
		.into_iter()
		.map(|entry| entry.name())
		.collect();
	assert_eq!(listed, ["extra.js", "tool.js"]);
}

#[test]
fn forward_file_symlink_routes_through_a_later_directory_alias() {
	let bytes = fixture(&[
		TarMember::symlink("pkg/bin/tool", "../current/tool.js"),
		TarMember::file("pkg/lib/tool.js", b"forward alias\n"),
		TarMember::symlink("pkg/current", "lib"),
	]);
	let mut archive = Archive::from_bytes(&bytes).unwrap();

	assert_eq!(archive.read("pkg/bin/tool").unwrap(), b"forward alias\n");
}

#[test]
fn aliases_can_target_the_archive_root() {
	let bytes = fixture(&[
		TarMember::file("top.txt", b"top level\n"),
		TarMember::file("dir/inner.txt", b"inner\n"),
		TarMember::symlink("current", "."),
		TarMember::symlink("dir/up", ".."),
	]);
	let mut archive = Archive::from_bytes(&bytes).unwrap();

	assert_eq!(archive.read("current/top.txt").unwrap(), b"top level\n");
	assert_eq!(archive.read("dir/up/top.txt").unwrap(), b"top level\n");
}

#[test]
fn later_duplicate_member_wins() {
	let bytes = fixture(&[
		TarMember::file("dup/file.txt", b"first\n"),
		TarMember::file("dup/file.txt", b"second\n"),
	]);
	let mut archive = Archive::from_bytes(&bytes).unwrap();

	assert_eq!(archive.read("dup/file.txt").unwrap(), b"second\n");
	assert_eq!(
		archive
			.entries()
			.filter(|entry| entry.path() == "dup/file.txt")
			.count(),
		1
	);
}

#[test]
fn dangling_and_self_cyclic_links_remain_listed_but_unreadable() {
	let dangling = fixture(&[TarMember::symlink("pkg/dangling", "missing-target")]);
	let mut archive = Archive::from_bytes(&dangling).unwrap();
	assert!(archive.entry("pkg/dangling").unwrap().is_link());
	assert_error_contains(archive.read("pkg/dangling"), "cannot be materialized");

	let cyclic =
		fixture(&[TarMember::file("a/b/f.txt", b"still readable\n"), TarMember::symlink("a", "a/b")]);
	let mut archive = Archive::from_bytes(&cyclic).unwrap();
	assert_eq!(archive.read("a/b/f.txt").unwrap(), b"still readable\n");
	assert_error_contains(archive.read("a"), "cannot be materialized");
}

fn alias_chain(length: usize) -> Vec<u8> {
	let names: Vec<_> = (0..length).map(|index| format!("a{index}")).collect();
	let mut members = Vec::with_capacity(length + 1);
	members.push(TarMember::file("real/f.txt", b"deep\n"));
	for index in 0..length {
		let target = if index + 1 == length {
			"real"
		} else {
			names[index + 1].as_str()
		};
		members.push(TarMember::symlink(names[index].as_str(), target));
	}
	fixture(&members)
}

#[test]
fn alias_rewrite_limit_accepts_exactly_40_and_rejects_41() {
	let bytes = alias_chain(40);
	let mut archive = Archive::from_bytes(&bytes).unwrap();
	assert_eq!(archive.read("a0/f.txt").unwrap(), b"deep\n");

	let bytes = alias_chain(41);
	let mut archive = Archive::from_bytes(&bytes).unwrap();
	assert_error_contains(archive.read("a0/f.txt"), "40");
}

#[test]
fn old_gnu_sparse_continuations_are_validated_streamed_and_extracted() {
	let bytes = old_gnu_sparse_fixture();
	let mut archive = Archive::from_bytes(&bytes).unwrap();

	let entry = archive.entry("data/old-sparse.bin").unwrap();
	assert_eq!(entry.size(), 4608);
	assert_eq!(entry.compressed_size(), 2052);
	assert_eq!(archive.read("data/after.txt").unwrap(), b"after sparse\n");
	let expanded = archive.read("data/old-sparse.bin").unwrap();
	assert_eq!(expanded.len(), 4608);
	for (offset, byte) in [(0, b'A'), (1024, b'B'), (2048, b'C'), (3072, b'D')] {
		assert!(
			expanded[offset..offset + 512]
				.iter()
				.all(|value| *value == byte)
		);
		assert!(
			expanded[offset + 512..offset + 1024]
				.iter()
				.all(|value| *value == 0)
		);
	}
	assert_eq!(&expanded[4096..4100], b"tail");
	assert!(expanded[4100..].iter().all(|value| *value == 0));

	let destination = tempdir().unwrap();
	let directory = Dir::open_ambient_dir(destination.path(), ambient_authority()).unwrap();
	assert_eq!(archive.extract_to(&directory).unwrap(), 2);
	assert_eq!(std::fs::read(destination.path().join("data/old-sparse.bin")).unwrap(), expanded);
}

#[test]
fn malformed_old_gnu_sparse_maps_fail_during_bounded_indexing() {
	let mut overlap = old_gnu_sparse_fixture();
	rewrite_octal(&mut overlap[410..422], 256);
	rewrite_header_checksum(&mut overlap[..512]);
	assert_error_contains(Archive::from_bytes(&overlap), "overlap");

	let mut wrong_total = old_gnu_sparse_fixture();
	rewrite_octal(&mut wrong_total[124..136], 2053);
	rewrite_header_checksum(&mut wrong_total[..512]);
	assert_error_contains(Archive::from_bytes(&wrong_total), "stored size");

	let mut invalid_flag = old_gnu_sparse_fixture();
	invalid_flag[482] = 2;
	rewrite_header_checksum(&mut invalid_flag[..512]);
	assert_error_contains(Archive::from_bytes(&invalid_flag), "continuation flag");

	let limits = Limits::DEFAULT.with_max_member_size(4096);
	assert!(matches!(
		Archive::from_bytes_with_limits(&old_gnu_sparse_fixture(), limits),
		Err(Error::MemberTooLarge { actual: 4608, limit: 4096, .. })
	));

	let limits = Limits::DEFAULT.with_max_index_size(64);
	assert!(matches!(
		Archive::from_bytes_with_limits(&old_gnu_sparse_fixture(), limits),
		Err(Error::IndexTooLarge { actual: 80, limit: 64 })
	));
}

#[test]
fn sparse_pax_uses_real_name_and_logical_size_but_rejects_reads() {
	let records = [
		("GNU.sparse.major", "1"),
		("GNU.sparse.minor", "0"),
		("GNU.sparse.name", "data/sparse.bin"),
		("GNU.sparse.realsize", "1048576"),
		("size", "11"),
	];
	let bytes =
		pax_fixture(&records, TarMember::file("GNUSparseFile.0/sparse.bin", b"sparse-map\n"));
	let mut archive = Archive::from_bytes(&bytes).unwrap();

	let entry = archive.entry("data/sparse.bin").unwrap();
	assert_eq!(entry.size(), 1_048_576);
	assert_eq!(entry.compressed_size(), 11);
	assert!(archive.entry("GNUSparseFile.0/sparse.bin").is_none());
	assert_error_contains(archive.read("data/sparse.bin"), "sparse");
}

#[test]
fn sparse_pax_v0_size_sets_logical_size_and_enforces_the_member_limit() {
	for minor in ["0", "1"] {
		let records = [
			("GNU.sparse.major", "0"),
			("GNU.sparse.minor", minor),
			("GNU.sparse.size", "1048576"),
			("GNU.sparse.numblocks", "1"),
			("GNU.sparse.offset", "0"),
			("GNU.sparse.numbytes", "11"),
			("size", "11"),
		];
		let bytes = pax_fixture(&records, TarMember::file("sparse.bin", b"sparse-map\n"));
		let mut archive = Archive::from_bytes(&bytes).unwrap();

		let entry = archive.entry("sparse.bin").unwrap();
		assert_eq!(entry.size(), 1_048_576);
		assert_eq!(entry.compressed_size(), 11);
		assert_error_contains(archive.read("sparse.bin"), "sparse");

		let limits = Limits::DEFAULT.with_max_member_size(1_000_000);
		assert!(matches!(
			Archive::from_bytes_with_limits(&bytes, limits),
			Err(Error::MemberTooLarge { actual: 1_048_576, limit: 1_000_000, .. })
		));
	}
}

#[test]
fn truncated_payloads_fail_while_clean_boundary_eof_and_one_zero_block_terminate() {
	let complete = fixture(&[TarMember::file("big.txt", &vec![b'A'; 2048])]);
	assert_error_contains(Archive::from_bytes(&complete[..512 + 256]), "truncated");

	let complete = fixture(&[TarMember::file("complete.txt", b"complete member\n")]);
	for end in [complete.len() - 1024, complete.len() - 512] {
		let mut archive = Archive::from_bytes(&complete[..end]).unwrap();
		assert_eq!(archive.read("complete.txt").unwrap(), b"complete member\n");
	}

	let orphan = fixture(&[TarMember::metadata(b'L', b"future.txt\0")]);
	assert_error_contains(Archive::from_bytes(&orphan[..orphan.len() - 1024]), "orphaned");

	let not_tar = gzip_bytes(b"hello world\n");
	assert_error_contains(Archive::from_bytes(&not_tar), "valid TAR archive");
}

#[test]
fn concatenated_archives_stop_at_the_first_zero_terminator_by_default() {
	let mut bytes = fixture(&[TarMember::file("first.txt", b"first")]);
	bytes.extend_from_slice(&fixture(&[TarMember::file("second.txt", b"second")]));
	let mut archive = Archive::from_bytes(&bytes).unwrap();
	assert_eq!(archive.read("first.txt").unwrap(), b"first");
	assert!(archive.entry("second.txt").is_none());
}

#[test]
fn pax_path_and_link_limits_accept_4096_bytes_and_reject_4097() {
	let max_path = "p".repeat(4096);
	let records = [("path", max_path.as_str())];
	let bytes = pax_fixture(&records, TarMember::file("placeholder", b"ok"));
	let mut archive = Archive::from_bytes(&bytes).unwrap();
	assert_eq!(archive.read(&max_path).unwrap(), b"ok");

	let too_long_path = "p".repeat(4097);
	let records = [("path", too_long_path.as_str())];
	let bytes = pax_fixture(&records, TarMember::file("placeholder", b"no"));
	assert!(matches!(
		Archive::from_bytes(&bytes),
		Err(Error::PathTooLong { actual: 4097, limit: 4096 })
	));

	let max_target = "t".repeat(4096);
	let records = [("linkpath", max_target.as_str())];
	let bytes = pax_fixture(&records, TarMember::symlink("link", "placeholder"));
	let archive = Archive::from_bytes(&bytes).unwrap();
	assert_eq!(archive.entry("link").unwrap().link_target().unwrap().len(), 4096);

	let too_long_target = "t".repeat(4097);
	let records = [("linkpath", too_long_target.as_str())];
	let bytes = pax_fixture(&records, TarMember::symlink("link", "placeholder"));
	assert!(matches!(
		Archive::from_bytes(&bytes),
		Err(Error::PathTooLong { actual: 4097, limit: 4096 })
	));
}

#[test]
fn gnu_long_records_override_pax_names_and_duplicate_local_metadata_is_rejected() {
	let pax_path = pax_records(&[("path", "pax-name.txt")]);
	let bytes = fixture(&[
		TarMember::metadata(b'x', &pax_path),
		TarMember::metadata(b'L', b"gnu-name.txt\0"),
		TarMember::file("header-name.txt", b"gnu wins"),
	]);
	let mut archive = Archive::from_bytes(&bytes).unwrap();
	assert_eq!(archive.read("gnu-name.txt").unwrap(), b"gnu wins");
	assert!(archive.entry("pax-name.txt").is_none());

	let pax_link = pax_records(&[("linkpath", "pax-target.txt")]);
	let bytes = fixture(&[
		TarMember::file("gnu-target.txt", b"gnu target"),
		TarMember::file("pax-target.txt", b"pax target"),
		TarMember::metadata(b'x', &pax_link),
		TarMember::metadata(b'K', b"gnu-target.txt\0"),
		TarMember::symlink("link.txt", "header-target.txt"),
	]);
	let mut archive = Archive::from_bytes(&bytes).unwrap();
	assert_eq!(archive.read("link.txt").unwrap(), b"gnu target");

	for bytes in [
		fixture(&[
			TarMember::metadata(b'L', b"one\0"),
			TarMember::metadata(b'L', b"two\0"),
			TarMember::file("file.txt", b"file"),
		]),
		fixture(&[
			TarMember::metadata(b'K', b"one\0"),
			TarMember::metadata(b'K', b"two\0"),
			TarMember::symlink("link", "target"),
		]),
		fixture(&[
			TarMember::metadata(b'x', &pax_path),
			TarMember::metadata(b'x', &pax_path),
			TarMember::file("file.txt", b"file"),
		]),
	] {
		assert_error_contains(Archive::from_bytes(&bytes), "multiple");
	}
}

#[test]
fn pax_size_overrides_nonzero_headers_but_never_intermediary_extension_records() {
	let pax_size = pax_records(&[("size", "1024")]);
	let payload = vec![b'P'; 1024];
	let mut bytes = fixture(&[
		TarMember::metadata(b'x', &pax_size),
		TarMember::file("payload.bin", &payload),
		TarMember::file("after.txt", b"after"),
	]);
	rewrite_octal(&mut bytes[1024 + 124..1024 + 136], 8);
	rewrite_header_checksum(&mut bytes[1024..1536]);
	let mut archive = Archive::from_bytes(&bytes).unwrap();
	assert_eq!(archive.read("payload.bin").unwrap(), payload);
	assert_eq!(archive.read("after.txt").unwrap(), b"after");

	let bytes = fixture(&[
		TarMember::metadata(b'x', &pax_size),
		TarMember::metadata(b'L', b"renamed.bin\0"),
		TarMember::file("header.bin", &payload),
		TarMember::file("after.txt", b"after"),
	]);
	let mut archive = Archive::from_bytes(&bytes).unwrap();
	assert_eq!(archive.read("renamed.bin").unwrap(), payload);
	assert_eq!(archive.read("after.txt").unwrap(), b"after");
}

#[test]
fn many_unused_pax_keys_do_not_change_the_effective_member() {
	let mut records: Vec<(String, String)> = (0..2048)
		.map(|index| (format!("vendor.unused.{index}"), "discarded metadata".repeat(4)))
		.collect();
	records.push(("path".into(), "kept/member.txt".into()));
	let bytes = pax_fixture(&records, TarMember::file("placeholder", b"kept\n"));
	let mut archive = Archive::from_bytes(&bytes).unwrap();

	assert_eq!(archive.read("kept/member.txt").unwrap(), b"kept\n");
	assert!(archive.entry("placeholder").is_none());
}

#[test]
fn global_pax_attributes_inherit_update_and_delete() {
	let set_path = pax_records(&[("path", "global.txt")]);
	let clear_path = pax_records(&[("path", "")]);
	let bytes = fixture(&[
		TarMember::metadata(b'g', &set_path),
		TarMember::file("ignored.txt", b"global\n"),
		TarMember::metadata(b'g', &clear_path),
		TarMember::file("literal.txt", b"literal\n"),
	]);
	let mut archive = Archive::from_bytes(&bytes).unwrap();
	assert_eq!(archive.read("global.txt").unwrap(), b"global\n");
	assert_eq!(archive.read("literal.txt").unwrap(), b"literal\n");
	assert!(archive.entry("ignored.txt").is_none());

	let set_global = pax_records(&[("path", "global-name.txt"), ("size", "1024")]);
	let clear_local = pax_records(&[("path", ""), ("size", "")]);
	let bytes = fixture(&[
		TarMember::metadata(b'g', &set_global),
		TarMember::metadata(b'x', &clear_local),
		TarMember::file("header-name.txt", b"header"),
	]);
	let mut archive = Archive::from_bytes(&bytes).unwrap();
	assert_eq!(archive.read("header-name.txt").unwrap(), b"header");
	assert!(archive.entry("global-name.txt").is_none());

	let set_sparse = pax_records(&[("GNU.sparse.major", "1")]);
	let clear_sparse = pax_records(&[("GNU.sparse.major", "")]);
	let bytes = fixture(&[
		TarMember::metadata(b'g', &set_sparse),
		TarMember::file("sparse.bin", b"extent"),
		TarMember::metadata(b'g', &clear_sparse),
		TarMember::file("plain.bin", b"plain"),
	]);
	let mut archive = Archive::from_bytes(&bytes).unwrap();
	assert_error_contains(archive.read("sparse.bin"), "sparse");
	assert_eq!(archive.read("plain.bin").unwrap(), b"plain");
}

#[test]
fn old_gnu_name_records_rename_subtrees_and_hard_link_targets() {
	let rename = b"Rename old to moved/\n";
	let bytes = fixture(&[
		TarMember::file("moved/file.txt", b"stale\n"),
		TarMember::file("old/file.txt", b"renamed\n"),
		TarMember::hard_link("outside-link.txt", "old/file.txt"),
		TarMember::metadata(b'N', rename),
	]);
	let mut archive = Archive::from_bytes(&bytes).unwrap();

	assert!(archive.entry("old/file.txt").is_none());
	assert_eq!(archive.read("moved/file.txt").unwrap(), b"renamed\n");
	assert_eq!(archive.read("outside-link.txt").unwrap(), b"renamed\n");
}

#[test]
fn old_gnu_name_records_unquote_rename_targets() {
	let rename = b"Rename old to moved\\040line\\nset/\n";
	let bytes =
		fixture(&[TarMember::file("old/file.txt", b"escaped\n"), TarMember::metadata(b'N', rename)]);
	let mut archive = Archive::from_bytes(&bytes).unwrap();

	assert!(archive.entry("old/file.txt").is_none());
	assert_eq!(archive.read("moved line\nset/file.txt").unwrap(), b"escaped\n");
}

#[test]
fn tar_and_tar_gzip_writers_are_deterministic_and_round_trip() {
	let files = [("tree/repeat.txt", vec![b'A'; 4096]), ("tree/deep/note.txt", b"nested".to_vec())];
	let first = tar::encode(files.iter().map(|(path, data)| (*path, data.as_slice()))).unwrap();
	let second = tar::encode(files.iter().map(|(path, data)| (*path, data.as_slice()))).unwrap();
	assert_eq!(first, second);
	let mut archive = Archive::from_bytes(&first).unwrap();
	assert_eq!(archive.format(), Format::Tar);
	assert_eq!(archive.read("tree/repeat.txt").unwrap(), vec![b'A'; 4096]);
	assert_eq!(archive.read("tree/deep/note.txt").unwrap(), b"nested");

	let first = tar::encode_gzip(files.iter().map(|(path, data)| (*path, data.as_slice()))).unwrap();
	let second =
		tar::encode_gzip(files.iter().map(|(path, data)| (*path, data.as_slice()))).unwrap();
	assert_eq!(first, second);
	let mut archive = Archive::from_bytes(&first).unwrap();
	assert_eq!(archive.format(), Format::TarGz);
	assert_eq!(archive.read("tree/deep/note.txt").unwrap(), b"nested");

	let long_path = format!("long/{}/file.txt", "segment".repeat(20));
	let mut writer = tar::Writer::new(Vec::new());
	writer.add_directory("empty").unwrap();
	writer.add_file(&long_path, b"long name").unwrap();
	let bytes = writer.finish().unwrap();
	let mut archive = Archive::from_bytes(&bytes).unwrap();
	assert!(archive.entry("empty").unwrap().is_directory());
	assert_eq!(archive.read(&long_path).unwrap(), b"long name");
}

#[test]
fn writer_rejects_mixed_kind_duplicates_and_emits_standard_gnu_long_link_records() {
	let long_target = "target-".repeat(18);
	let long_path = format!("pkg/{long_target}");
	let mut writer = tar::Writer::new(Vec::new());
	writer.add_file("pkg/target.txt", b"first").unwrap();
	assert!(matches!(
		writer.add_symlink("./pkg\\target.txt", "elsewhere"),
		Err(Error::DuplicatePath(path)) if path == "pkg/target.txt"
	));
	writer
		.add_hard_link("pkg/hard.txt", "pkg/target.txt")
		.unwrap();
	writer.add_file(&long_path, b"long target").unwrap();
	writer.add_symlink("pkg/current", &long_target).unwrap();
	let bytes = writer.finish().unwrap();
	let mut second = tar::Writer::new(Vec::new());
	second.add_file("pkg/target.txt", b"first").unwrap();
	second
		.add_hard_link("pkg/hard.txt", "pkg/target.txt")
		.unwrap();
	second.add_file(&long_path, b"long target").unwrap();
	second.add_symlink("pkg/current", &long_target).unwrap();
	assert_eq!(bytes, second.finish().unwrap());

	let mut archive = Archive::from_bytes(&bytes).unwrap();
	assert_eq!(archive.read("pkg/target.txt").unwrap(), b"first");
	assert_eq!(archive.read("pkg/hard.txt").unwrap(), b"first");
	assert_eq!(archive.read("pkg/current").unwrap(), b"long target");

	let records = raw_tar_records(&bytes);
	assert_eq!(
		records
			.iter()
			.filter(|record| record.path.as_slice() == b"pkg/target.txt")
			.count(),
		1
	);
	assert!(records.iter().any(|record| {
		record.path.as_slice() == b"pkg/hard.txt"
			&& record.kind == b'1'
			&& record.link.as_deref() == Some(b"pkg/target.txt".as_slice())
	}));
	assert!(records.iter().any(|record| {
		record.path.as_slice() == b"pkg/current"
			&& record.kind == b'2'
			&& record.link.as_deref() == Some(long_target.as_bytes())
	}));
}

#[test]
fn format_sniffing_and_path_inference_cover_all_supported_formats() {
	let zip = zip_fixture(&[Member::stored(b"zip.txt", b"zip")]);
	let tar = fixture(&[TarMember::file("tar.txt", b"tar")]);
	let tar_gz = gzip_fixture(&[TarMember::file("gzip.txt", b"gzip")]);

	assert_eq!(Format::sniff(&zip), Some(Format::Zip));
	assert_eq!(Format::sniff(&tar), Some(Format::Tar));
	assert_eq!(Format::sniff(&tar_gz), Some(Format::TarGz));
	assert_eq!(Archive::from_bytes(&zip).unwrap().format(), Format::Zip);
	assert_eq!(Archive::from_bytes(&tar).unwrap().format(), Format::Tar);
	assert_eq!(Archive::from_bytes(&tar_gz).unwrap().format(), Format::TarGz);
	let v7 = v7_fixture(TarMember::file("legacy.txt", b"legacy"));
	assert_eq!(Format::sniff(&v7), Some(Format::Tar));

	let open_as = |suffix: &str, bytes: &[u8]| -> omp_ar::Result<Format> {
		let mut file = tempfile::Builder::new().suffix(suffix).tempfile()?;
		file.write_all(bytes)?;
		Ok(Archive::open(file.path())?.format())
	};
	assert_eq!(open_as(".tar", &tar).unwrap(), Format::Tar);
	assert_eq!(open_as(".tgz", &tar_gz).unwrap(), Format::TarGz);
	assert_eq!(open_as(".unknown", &zip).unwrap(), Format::Zip);
	assert_eq!(open_as(".tar", &v7).unwrap(), Format::Tar);
	assert_eq!(open_as(".unknown", &v7).unwrap(), Format::Tar);

	for extension in ["zip", "jar", "war", "ear", "apk"] {
		assert_eq!(Format::from_path(Path::new(&format!("archive.{extension}"))), Some(Format::Zip));
	}
	assert_eq!(Format::from_path(Path::new("archive.TAR")), Some(Format::Tar));
	assert_eq!(Format::from_path(Path::new("archive.tar.gz")), Some(Format::TarGz));
	assert_eq!(Format::from_path(Path::new("archive.TGZ")), Some(Format::TarGz));
	assert_eq!(Format::from_path(Path::new("archive.bin")), None);
	assert_eq!(Format::Zip.extension(), "zip");
	assert_eq!(Format::Tar.extension(), "tar");
	assert_eq!(Format::TarGz.extension(), "tar.gz");
}
