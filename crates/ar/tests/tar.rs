//! TAR/TAR.GZ interoperability, alias, and malformed-input contracts.

mod support;

use std::{io::Write, path::Path};

use omp_ar::{Archive, Error, Format, tar};
use support::{
	Member, fixture as zip_fixture,
	tar::{
		TarMember, fixture, gzip_bytes, gzip_fixture, old_gnu_sparse_fixture, pax_fixture,
		pax_records, v7_fixture,
	},
};

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
fn old_gnu_sparse_continuation_is_skipped_without_hiding_following_members() {
	let bytes = old_gnu_sparse_fixture();
	let mut archive = Archive::from_bytes(&bytes).unwrap();

	assert!(archive.entry("data/old-sparse.bin").is_some());
	assert_eq!(archive.read("data/after.txt").unwrap(), b"after sparse\n");
	assert_error_contains(archive.read("data/old-sparse.bin"), "sparse");
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
fn truncated_member_missing_terminator_and_non_tar_gzip_are_rejected() {
	let complete = fixture(&[TarMember::file("big.txt", &vec![b'A'; 2048])]);
	assert_error_contains(Archive::from_bytes(&complete[..512 + 256]), "truncated");

	let complete = fixture(&[TarMember::file("complete.txt", b"complete member\n")]);
	assert_error_contains(
		Archive::from_bytes(&complete[..complete.len() - 1024]),
		"terminating TAR zero block",
	);

	let not_tar = gzip_bytes(b"hello world\n");
	assert_error_contains(Archive::from_bytes(&not_tar), "valid TAR archive");
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
	assert_eq!(Format::sniff(&v7), None);

	let open_as = |suffix: &str, bytes: &[u8]| -> omp_ar::Result<Format> {
		let mut file = tempfile::Builder::new().suffix(suffix).tempfile()?;
		file.write_all(bytes)?;
		Ok(Archive::open(file.path())?.format())
	};
	assert_eq!(open_as(".tar", &tar).unwrap(), Format::Tar);
	assert_eq!(open_as(".tgz", &tar_gz).unwrap(), Format::TarGz);
	assert_eq!(open_as(".unknown", &zip).unwrap(), Format::Zip);
	assert_eq!(open_as(".tar", &v7).unwrap(), Format::Tar);
	assert!(matches!(open_as(".unknown", &v7), Err(Error::UnknownFormat)));

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
