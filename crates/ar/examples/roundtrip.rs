//! Creates, indexes, and reads ZIP, TAR, and TAR.GZ archives in memory.

use omp_ar::{Archive, tar, zip};

fn main() -> omp_ar::Result<()> {
	let members = [("hello.txt", b"hello".as_slice()), ("nested/data.bin", [0, 1, 2, 3].as_slice())];
	let encoded = [
		("ZIP", zip::encode(members)?),
		("TAR", tar::encode(members)?),
		("TAR.GZ", tar::encode_gzip(members)?),
	];

	for (label, bytes) in encoded {
		let mut archive = Archive::from_bytes(&bytes)?;
		assert_eq!(archive.read("hello.txt")?, b"hello");
		assert_eq!(archive.read("nested/data.bin")?, [0, 1, 2, 3]);
		println!("indexed {} entries from {} {label} bytes", archive.entries().count(), bytes.len());
	}
	Ok(())
}
