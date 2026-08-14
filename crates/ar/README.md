# omp-ar

`omp-ar` provides bounded, lazy reads and deterministic writes for ZIP, TAR, and gzip-compressed TAR archives. It sniffs in-memory inputs, infers formats from common archive extensions, and indexes seekable sources before reading member payloads on demand.

## Formats

- ZIP reading supports stored and DEFLATE members, CRC-32 verification, ZIP64 metadata, legacy CP437 and Info-ZIP Unicode names, extended timestamps, prepended archives, and capability-scoped extraction. ZIP writing emits ordinary deterministic archives and reports inputs that require ZIP64.
- TAR reading supports V7, USTAR, GNU long names and links, PAX path/link/size records, hard links, safe symbolic-link aliases, and bounded old-GNU sparse expansion. PAX sparse members remain listable but reject payload reads because tar 0.4.46 does not expand them.
- TAR and TAR.GZ writing emits deterministic USTAR/GNU file, directory, hard-link, and symbolic-link records. Gzip output fixes the header modification time at zero.

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
