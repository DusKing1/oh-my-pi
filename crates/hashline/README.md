# omp-hashline

`omp-hashline` provides disk-free parsing and application of hashline patches over immutable, exact-byte snapshots. It turns patch input into canonical byte edits for a caller-owned transaction coordinator, with conservative recovery when an edit targets a stale snapshot.

## Structure

- `input`, `tokenizer`, and `parser` split patch envelopes, tokenize line-oriented operations, and build parsed patches.
- `apply`, `block`, and `replace` lower exact, syntax-aware block, and fuzzy replacement operations into byte edits.
- `snapshots` retains collision-aware read snapshots and computes revision tags; `recovery` reconciles stale edits against those snapshots.
- `clipboard` manages transaction-local cut/paste registers, while `normalize` preserves BOM and line-ending behavior.
- `format`, `diff_preview`, `loop_guard`, `syntax`, and `types` provide hashline formatting, compact previews, repeated no-op detection, conservative syntax probes, and shared domain types.

## Philosophy

The crate keeps filesystem ownership and transaction coordination outside the patch engine. Operations are evaluated against immutable byte content, and results are expressed as explicit edits rather than performed as hidden I/O. Exact application is preferred; fuzzy replacement and stale recovery are deliberately conservative, and normalization is restored so byte-oriented edits do not silently change file conventions.
