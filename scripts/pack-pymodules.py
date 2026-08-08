"""Packs trees of .py files into an omp-py frozen-modules blob.

Blob format (consumed by crates/py/src/lib.rs): u32 entry count, then per
module a `<HBI>` record — u16 name length (including trailing NUL), u8
is-package, u32 code length — followed by the NUL-terminated dotted name and
the marshalled code object. Must run under the vendored interpreter so the
bytecode format matches the embedded runtime.

Usage: pack-pymodules.py ROOT [ROOT ...] OUT [--prefix P] [--exclude PATTERN ...]

`--prefix` becomes the co_filename prefix (e.g. `<omp-stdlib>`); `--exclude`
drops directories whose basename matches any fnmatch pattern.
"""

import argparse
import fnmatch
import marshal
import os
import struct
import sys

parser = argparse.ArgumentParser()
parser.add_argument("roots", nargs="+", metavar="root")
parser.add_argument("out")
parser.add_argument("--prefix", default="<omp-py>")
parser.add_argument("--exclude", nargs="*", default=[])
args = parser.parse_args()

entries = []
for root in args.roots:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames
            if not any(fnmatch.fnmatch(d, pat) for pat in args.exclude)
        )
        for fname in sorted(filenames):
            if not fname.endswith(".py"):
                continue
            rel = os.path.normpath(
                os.path.join(os.path.relpath(dirpath, root), fname)
            )
            parts = rel[:-3].split(os.sep)
            is_pkg = parts[-1] == "__init__"
            if is_pkg:
                parts = parts[:-1]
            name = ".".join(parts)
            with open(os.path.join(dirpath, fname), "rb") as fh:
                src = fh.read()
            try:
                code = compile(src, f"{args.prefix}/{rel}", "exec")
            except SyntaxError:
                continue  # stdlib ships py2-flavoured template files
            data = marshal.dumps(code)
            nb = name.encode() + b"\x00"
            entries.append(struct.pack("<HBI", len(nb), is_pkg, len(data)) + nb + data)

with open(args.out, "wb") as fh:
    fh.write(struct.pack("<I", len(entries)))
    fh.write(b"".join(entries))
print(
    f"{os.path.basename(args.out)}: {len(entries)} modules, "
    f"{os.path.getsize(args.out)} bytes",
    file=sys.stderr,
)
