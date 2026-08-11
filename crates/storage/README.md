# omp-storage

`omp-storage` provides persistent session storage for omp. It combines a filesystem-backed, BLAKE3-addressed blob store for binary and large payloads with the transcript v4 append-only event journal.

## Structure

- `blob` defines typed blob references and the content-addressed `BlobStore`, including streamed writes, atomic placement, and integrity verification.
- `transcript` implements the event log. Its `block`, `event`, `msg`, and `types` modules define stored data; `codec`, `reader`, and `writer` handle the journal format; `patch`, `replay`, and `capsule` represent later corrections and provider-specific replay data.

## Philosophy

Storage is append-only: existing transcript bytes are not rewritten, and corrections or navigation are recorded as later events. Stable event indexes are preserved even for malformed lines. Payload ownership is explicit—neutral content belongs in transcript blocks, provider-native residue in replay capsules, and large data in the blob store. Blob references are derived from content, making writes idempotent and allowing payloads to be deduplicated across sessions; new files are synchronized and atomically renamed so readers do not observe partial writes.
