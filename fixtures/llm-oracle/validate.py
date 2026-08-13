#!/usr/bin/env python3
"""Validate the frozen LLM behavioral-oracle corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

LOCK_NAME = "manifest.lock.json"
ROOT_FILES = frozenset({LOCK_NAME, "validate.py"})
ID_RE = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TOKEN_RE = re.compile(rb"[A-Za-z0-9_./+=:-]{8,}")
CANARY_FINGERPRINTS = {
    "6a5f78b8b99dd4025e41ba11bf54c304c6af29f924c5569cc7865b2428ce03a9": "catalog-google-gemini-client-secret",
    "1d2f041093fd95aa8995a038c711d50a7960da09a505381c09a745d6ad0ecc60": "catalog-google-antigravity-client-secret",
    "baa1ada9abfe1f830d159948860833622d90e0ca63a1970e7aa8485342d73d3a": "broker-pkce-access",
    "166a03708adb575d2f6a46ded55167552c97f96dcaf82bf19141db5f4ea1231d": "broker-pkce-refresh",
    "13dfe07f0da5e88e4e122b2343005c77c4c02a03a52c0a3c4f8dadb533c5b350": "broker-device-secret",
    "7a6c309f081e6f7dea13ff24061abb604fa00d71cb685665624bdd07ce8abc08": "broker-expiring-secret",
    "a40a2029c0a57f245aa66a733491b4a44f9c3d46a50f3891eedcf50af198b78c": "broker-device-access",
    "9f3d8bea668860889373563b13b83071db6f974da4dbfb221f1bffb9b3cb524c": "broker-device-refresh",
    "e2864b5f8a6c8732209327f28885ae9fcb9ad1271afffd188833d7dec1935ee5": "broker-copilot-session",
    "d11a661b3c3e0b41c6a383d1d1a6c2edc91a7544a2e68f46d9d6f2516eb03d12": "broker-github-access",
    "633554e879e5a61adb0340b5b44196190e3f19e2351101e98d3c1f4d8f4d0c94": "broker-anthropic-access",
    "b53ced31dc7f7d709f8feb2a376179f0abc59d9bdd04e56840c974ac85126c6a": "broker-anthropic-refresh",
    "698577eb3dbeeae5d7f1c39d057646a4ef69f037bf8683da193bfeefa1991932": "broker-codex-access",
    "b9178c5dbf7365112a74c9e73ff61a4ab42f61e3f7ed725ee293018ae946e816": "broker-codex-refresh",
    "bab3881a9d25c42469bfaa182c73fce57ee1d9845bcffe55f074eb25fb28aac0": "broker-gemini-access",
    "54e0b69ed115972d5709823585835a84ac544811741b8cf3b11a2bd9f6a363ca": "broker-gemini-refresh",
    "5d9d1a37af8c338df35640c163321d5b3c15ee557d9364c9d7d4a6e704dc8f1e": "broker-antigravity-access",
    "540ec1fe0b341464559cc178e25bdd11bcdda6b6cce0c5e7c5c1b7e7d05de207": "broker-antigravity-refresh",
    "a867a21265af638062a4a909273cef867bc6ec4cfbc0151f3789711f70796075": "broker-gitlab-access",
    "43a8d3efb6be8c58e7e54841596573e50fe8defc8d851b7528fac15afc9d4276": "broker-gitlab-refresh",
    "a515b9860c00d7ad022cc199eeefb4642fdeef45062541028b30ac878c2bc7c6": "broker-github-device",
    "743dec1659ac0fdbd9c6d16eab0b86cb48752bb87840288ad06c4feb956a5350": "broker-github-source",
    "a4e6d261933dbc25ff7136a2ab9aaec7f64d533164fc360c77909b6418f55c1d": "broker-xai-device",
    "0d4a761e93da0bb65b58b860a313fcebe473137ad28e41867b65367a88fd17e8": "broker-xai-access",
    "e43ca0ceebdf9690033a75a01d1c64ec283bf7702fbec7ab9717ffc2fde75e82": "broker-xai-refresh",
    "63acf972a7e8634acf6d748ad0b088fa438c2787e1a13ca09d26612675f447e7": "broker-kimi-device",
    "b96e122528b7050b1872e0c93d2c5617746d3a348cf4dba58b2efe0ad310152f": "broker-kimi-access",
    "f0fc08457eca7a6676c9e27aaee5ca598012c223894fa19fa213bf47fba87f76": "broker-kimi-refresh",
}
PRIVATE_KEY_MARKERS = (
    b"-----BEGIN " + b"PRIVATE KEY-----",
    b"-----BEGIN " + b"RSA PRIVATE KEY-----",
    b"-----BEGIN " + b"OPENSSH PRIVATE KEY-----",
)
ENTRY_FIELDS = frozenset(
    {"id", "kind", "path", "provenance_category", "secret_free", "sha256"}
)
CATEGORY_FIELDS = frozenset({"id", "manifest_path", "artifact_count", "indexed_file_count"})


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path, errors: list[str], label: str) -> Any | None:
    try:
        return json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        errors.append(f"{label} is not valid readable JSON: {error}")
        return None


def safe_relative_path(value: object) -> str | None:
    if not isinstance(value, str) or not value or "\\" in value:
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return None
    if path.as_posix() != value or len(path.parts) < 2:
        return None
    return value


def corpus_files(root: Path, errors: list[str]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    if root.is_symlink():
        errors.append(f"corpus root is a symlink: {root}")
        return found
    if not root.is_dir():
        errors.append(f"corpus root is not a directory: {root}")
        return found

    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            children = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as error:
            errors.append(f"cannot scan {directory}: {error}")
            continue
        for child in children:
            path = Path(child.path)
            relative = path.relative_to(root).as_posix()
            if child.is_symlink():
                errors.append(f"symlink is forbidden: {relative}")
            elif child.is_dir(follow_symlinks=False):
                pending.append(path)
            elif child.is_file(follow_symlinks=False):
                found[relative] = path
            else:
                errors.append(f"non-regular corpus entry is forbidden: {relative}")
    return found


def scan_secrets(relative: str, payload: bytes, errors: list[str]) -> None:
    for marker in PRIVATE_KEY_MARKERS:
        if marker in payload:
            errors.append(f"private-key marker found in {relative}")
    for match in TOKEN_RE.finditer(payload):
        candidates = (match.group(), *re.split(rb"[:=/]+", match.group()))
        for candidate in candidates:
            fingerprint = sha256_bytes(candidate)
            label = CANARY_FINGERPRINTS.get(fingerprint)
            if label is not None:
                errors.append(f"archived credential canary {label} found in {relative}")


def aggregate_digest(entries: list[dict[str, Any]]) -> str:
    rows = [
        "\0".join(
            (
                entry["id"],
                entry["path"],
                entry["sha256"],
                entry["provenance_category"],
                "true" if entry["secret_free"] is True else "false",
            )
        )
        for entry in entries
    ]
    return sha256_bytes(("\n".join(rows) + "\n").encode())


def validate(root: Path) -> tuple[list[str], dict[str, Any] | None]:
    errors: list[str] = []
    files = corpus_files(root, errors)
    root_file_names = {path for path in files if "/" not in path}
    if root_file_names != ROOT_FILES:
        errors.append(
            "root governance file mismatch: "
            f"expected {sorted(ROOT_FILES)}, found {sorted(root_file_names)}"
        )

    for relative, path in files.items():
        try:
            scan_secrets(relative, path.read_bytes(), errors)
        except OSError as error:
            errors.append(f"cannot read {relative}: {error}")

    lock = load_json(root / LOCK_NAME, errors, "root lock")
    if not isinstance(lock, dict):
        return errors, None
    if lock.get("schema_version") != 1:
        errors.append("root lock schema_version must be 1")
    if lock.get("corpus") != "llm-oracle":
        errors.append("root lock corpus must be llm-oracle")
    if lock.get("hash_algorithm") != "sha256":
        errors.append("root lock hash_algorithm must be sha256")

    raw_categories = lock.get("categories")
    raw_entries = lock.get("entries")
    if not isinstance(raw_categories, list):
        errors.append("root lock categories must be an array")
        raw_categories = []
    if not isinstance(raw_entries, list):
        errors.append("root lock entries must be an array")
        raw_entries = []

    categories: dict[str, dict[str, Any]] = {}
    for index, category in enumerate(raw_categories):
        label = f"categories[{index}]"
        if not isinstance(category, dict) or set(category) != CATEGORY_FIELDS:
            errors.append(f"{label} must contain exactly {sorted(CATEGORY_FIELDS)}")
            continue
        category_id = category.get("id")
        if not isinstance(category_id, str) or ID_RE.fullmatch(category_id) is None:
            errors.append(f"{label} has non-normalized id")
            continue
        if category_id in categories:
            errors.append(f"duplicate category id: {category_id}")
        categories[category_id] = category

    entries: list[dict[str, Any]] = []
    ids: set[str] = set()
    paths: set[str] = set()
    for index, entry in enumerate(raw_entries):
        label = f"entries[{index}]"
        if not isinstance(entry, dict) or set(entry) != ENTRY_FIELDS:
            errors.append(f"{label} must contain exactly {sorted(ENTRY_FIELDS)}")
            continue
        entry_id = entry.get("id")
        if not isinstance(entry_id, str) or ID_RE.fullmatch(entry_id) is None:
            errors.append(f"{label} has non-normalized id")
        elif entry_id in ids:
            errors.append(f"duplicate entry id: {entry_id}")
        else:
            ids.add(entry_id)
        relative = safe_relative_path(entry.get("path"))
        if relative is None:
            errors.append(f"unsafe path in {label}: {entry.get('path')!r}")
        elif relative in paths:
            errors.append(f"duplicate entry path: {relative}")
        else:
            paths.add(relative)
        category = entry.get("provenance_category")
        if not isinstance(category, str) or ID_RE.fullmatch(category) is None:
            errors.append(f"{label} has non-normalized provenance_category")
        elif relative is not None and PurePosixPath(relative).parts[0] != category:
            errors.append(f"{label} path is outside provenance_category {category}")
        if entry.get("kind") not in {"artifact", "category-manifest"}:
            errors.append(f"{label} has invalid kind")
        if entry.get("secret_free") is not True:
            errors.append(f"{label} is not attested secret-free")
        digest = entry.get("sha256")
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            errors.append(f"{label} has invalid SHA-256")
        entries.append(entry)

    actual_indexed = {path for path in files if "/" in path}
    for relative in sorted(paths - actual_indexed):
        errors.append(f"missing indexed file: {relative}")
    for relative in sorted(actual_indexed - paths):
        errors.append(f"unindexed corpus file: {relative}")

    manifest_entries: dict[str, dict[str, Any]] = {}
    artifact_entries: dict[str, dict[str, Any]] = {}
    for entry in entries:
        relative = safe_relative_path(entry.get("path"))
        if relative is None:
            continue
        path = files.get(relative)
        if path is not None:
            try:
                actual_digest = sha256_bytes(path.read_bytes())
                if actual_digest != entry.get("sha256"):
                    errors.append(f"SHA-256 mismatch: {relative}")
            except OSError as error:
                errors.append(f"cannot hash {relative}: {error}")
        if entry.get("kind") == "category-manifest":
            category = entry.get("provenance_category")
            expected_id = f"oracle.category.{category}.manifest.v1"
            if entry.get("id") != expected_id:
                errors.append(f"category manifest id must be {expected_id}: {relative}")
            if relative != f"{category}/manifest.json":
                errors.append(f"category manifest path mismatch: {relative}")
            if isinstance(category, str):
                manifest_entries[category] = entry
        elif entry.get("kind") == "artifact":
            artifact_entries[relative] = entry

    actual_category_ids = {PurePosixPath(path).parts[0] for path in actual_indexed}
    if set(categories) != actual_category_ids:
        errors.append(
            "category inventory mismatch: "
            f"expected {sorted(categories)}, found {sorted(actual_category_ids)}"
        )
    if set(manifest_entries) != set(categories):
        errors.append("every category must have exactly one indexed category manifest")

    for category_id, summary in categories.items():
        expected_manifest_path = f"{category_id}/manifest.json"
        if summary.get("manifest_path") != expected_manifest_path:
            errors.append(f"category {category_id} has wrong manifest_path")
        manifest_path = files.get(expected_manifest_path)
        if manifest_path is None:
            continue
        manifest = load_json(manifest_path, errors, f"category manifest {category_id}")
        if not isinstance(manifest, dict):
            continue
        if manifest.get("category") != category_id:
            errors.append(f"category manifest {category_id} has wrong category")
        fixtures = manifest.get("fixtures")
        if not isinstance(fixtures, list):
            errors.append(f"category manifest {category_id} fixtures must be an array")
            continue
        if summary.get("artifact_count") != len(fixtures):
            errors.append(f"category {category_id} artifact_count mismatch")
        if summary.get("indexed_file_count") != len(fixtures) + 1:
            errors.append(f"category {category_id} indexed_file_count mismatch")

        declared_paths: set[str] = set()
        declared_ids: set[str] = set()
        for fixture_index, fixture in enumerate(fixtures):
            fixture_label = f"{category_id} fixtures[{fixture_index}]"
            if not isinstance(fixture, dict):
                errors.append(f"{fixture_label} must be an object")
                continue
            fixture_path = safe_relative_path(f"{category_id}/{fixture.get('path')}")
            fixture_id = fixture.get("id")
            if fixture_path is None:
                errors.append(f"unsafe category fixture path in {fixture_label}")
                continue
            if fixture_path in declared_paths:
                errors.append(f"duplicate category fixture path: {fixture_path}")
            declared_paths.add(fixture_path)
            if not isinstance(fixture_id, str) or ID_RE.fullmatch(fixture_id) is None:
                errors.append(f"{fixture_label} has non-normalized id")
            elif fixture_id in declared_ids:
                errors.append(f"duplicate category fixture id: {fixture_id}")
            declared_ids.add(fixture_id)
            indexed = artifact_entries.get(fixture_path)
            if indexed is None:
                errors.append(f"category fixture is not indexed: {fixture_path}")
                continue
            for field in ("id", "sha256", "secret_free"):
                if fixture.get(field) != indexed.get(field):
                    errors.append(f"category/root {field} mismatch: {fixture_path}")
        indexed_category_paths = {
            path
            for path, entry in artifact_entries.items()
            if entry.get("provenance_category") == category_id
        }
        if declared_paths != indexed_category_paths:
            errors.append(f"category/root fixture inventory mismatch: {category_id}")

    expected_counts = {
        "category_count": len(categories),
        "manifest_count": len(manifest_entries),
        "artifact_count": len(artifact_entries),
        "indexed_file_count": len(entries),
        "secret_free_count": sum(entry.get("secret_free") is True for entry in entries),
    }
    for field, expected in expected_counts.items():
        if lock.get(field) != expected:
            errors.append(f"root lock {field} mismatch: expected {expected}")

    if entries != sorted(entries, key=lambda entry: entry.get("path", "")):
        errors.append("root lock entries are not sorted by path")
    if all(isinstance(category, dict) for category in raw_categories):
        if raw_categories != sorted(raw_categories, key=lambda category: category.get("id", "")):
            errors.append("root lock categories are not sorted by id")
    try:
        digest = aggregate_digest(entries)
    except (KeyError, TypeError):
        digest = None
        errors.append("cannot compute aggregate corpus digest from malformed entries")
    if digest is not None and lock.get("corpus_sha256") != digest:
        errors.append("root lock corpus_sha256 mismatch")

    stats = {
        **expected_counts,
        "categories": sorted(categories),
        "corpus_sha256": digest,
    }
    return errors, stats


def snapshot(root: Path) -> dict[str, str]:
    errors: list[str] = []
    files = corpus_files(root, errors)
    if errors:
        raise RuntimeError("; ".join(errors))
    return {relative: sha256_bytes(path.read_bytes()) for relative, path in files.items()}


def assert_rejected(root: Path, mutation: str, expected: str) -> None:
    errors, _ = validate(root)
    if not any(expected in error for error in errors):
        details = "; ".join(errors) if errors else "validation unexpectedly succeeded"
        raise RuntimeError(f"{mutation} was not discriminated by {expected!r}: {details}")
    print(f"[ok] rejected {mutation}")


def self_test(root: Path) -> dict[str, Any]:
    errors, stats = validate(root)
    if errors or stats is None:
        raise RuntimeError("baseline corpus is invalid: " + "; ".join(errors))
    before = snapshot(root)
    with tempfile.TemporaryDirectory(prefix="llm-oracle-validator-") as temporary:
        temporary_root = Path(temporary)

        changed = temporary_root / "changed-byte"
        shutil.copytree(root, changed, symlinks=True)
        first_artifact = next(
            entry["path"]
            for entry in json.loads((changed / LOCK_NAME).read_bytes())["entries"]
            if entry["kind"] == "artifact"
        )
        with (changed / first_artifact).open("ab") as fixture:
            fixture.write(b"\0")
        assert_rejected(changed, "changed byte", "SHA-256 mismatch")

        removed = temporary_root / "removed-fixture"
        shutil.copytree(root, removed, symlinks=True)
        (removed / first_artifact).unlink()
        assert_rejected(removed, "removed fixture", "missing indexed file")

        duplicate = temporary_root / "duplicate-id"
        shutil.copytree(root, duplicate, symlinks=True)
        duplicate_lock_path = duplicate / LOCK_NAME
        duplicate_lock = json.loads(duplicate_lock_path.read_bytes())
        duplicate_lock["entries"][1]["id"] = duplicate_lock["entries"][0]["id"]
        duplicate_lock_path.write_text(json.dumps(duplicate_lock, indent=2) + "\n")
        assert_rejected(duplicate, "duplicate ID", "duplicate entry id")

        unsafe = temporary_root / "unsafe-path"
        shutil.copytree(root, unsafe, symlinks=True)
        unsafe_lock_path = unsafe / LOCK_NAME
        unsafe_lock = json.loads(unsafe_lock_path.read_bytes())
        unsafe_lock["entries"][0]["path"] = "../escape"
        unsafe_lock_path.write_text(json.dumps(unsafe_lock, indent=2) + "\n")
        assert_rejected(unsafe, "unsafe path", "unsafe path")

        symlink = temporary_root / "symlink-escape"
        shutil.copytree(root, symlink, symlinks=True)
        (symlink / "escape").symlink_to(temporary_root)
        assert_rejected(symlink, "symlink escape", "symlink is forbidden")

        canary = temporary_root / "credential-canary"
        shutil.copytree(root, canary, symlinks=True)
        with (canary / first_artifact).open("ab") as fixture:
            fixture.write(b"\n" + b"pkce-" + b"access")
        assert_rejected(canary, "archived credential canary", "archived credential canary")

    if snapshot(root) != before:
        raise RuntimeError("self-test mutated the source corpus")
    return stats


def print_success(stats: dict[str, Any]) -> None:
    print(
        "OK: "
        f"{stats['indexed_file_count']} indexed files "
        f"({stats['manifest_count']} category manifests + {stats['artifact_count']} artifacts), "
        f"{stats['secret_free_count']} secret-free; "
        f"categories: {', '.join(stats['categories'])}; "
        f"corpus_sha256: {stats['corpus_sha256']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        help="corpus root to validate (defaults to the validator's directory)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="also prove rejection of changed, removed, duplicate-ID, and unsafe-path copies",
    )
    arguments = parser.parse_args()
    root = arguments.root if arguments.root is not None else Path(__file__).absolute().parent
    try:
        if arguments.self_test:
            stats = self_test(root)
        else:
            errors, stats = validate(root)
            if errors or stats is None:
                for error in errors:
                    print(f"ERROR: {error}", file=sys.stderr)
                return 1
        print_success(stats)
        return 0
    except (OSError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
