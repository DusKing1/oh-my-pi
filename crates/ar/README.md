# omp-ar

`omp-ar` provides bounded, lazy reads and deterministic writes for ZIP, TAR, and gzip-compressed TAR archives. It sniffs in-memory inputs, infers formats from common archive extensions, and indexes seekable sources before reading member payloads on demand.

## Formats

- ZIP reading supports stored and DEFLATE members, CRC-32 verification, ZIP64 metadata, and capability-scoped extraction. ZIP writing emits ordinary deterministic archives and reports inputs that require ZIP64.
- TAR reading supports USTAR, GNU long names and links, PAX path/link/size records, hard links, safe symbolic-link aliases, and old-GNU sparse indexing. Sparse payload expansion is rejected explicitly.
- TAR and TAR.GZ writing emits deterministic USTAR/GNU records. Gzip output fixes the header modification time at zero.

## Safety

Archive paths are normalized once, unsafe paths never enter the index, and limits bound decoded archives, indexes, members, materialized output, path bytes, path depth, entries, and link rewrites. TAR.GZ input is bounded while decompressing; ZIP and plain TAR members stay seek-lazy.

## Example

```rust
use omp_ar::{Archive, tar, zip};

let members = [("hello.txt", b"hello".as_slice())];
for bytes in [zip::encode(members)?, tar::encode(members)?, tar::encode_gzip(members)?] {
	let mut archive = Archive::from_bytes(&bytes)?;
	assert_eq!(archive.read("hello.txt")?, b"hello");
}
# Ok::<(), omp_ar::Error>(())
```
