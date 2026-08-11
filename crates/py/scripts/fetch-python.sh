#!/usr/bin/env bash
# Fetches the python-build-standalone "full" archive (static libpython +
# stdlib) and generates the build inputs omp-py derives from it:
#   <dest>/python/stdlib.bin       in-memory stdlib blob embedded by omp-py
#   <dest>/python/pyo3-config.txt  static-link config consumed via PYO3_CONFIG_FILE
#
# Usage: fetch-python.sh [dest-dir]
#   dest-dir  directory that receives the `python/` tree; defaults to the
#             repo checkout's `vendor/` when run from a checkout. Consumers
#             building the published crate pass an explicit directory and
#             point PYO3_CONFIG_FILE at <dest>/python/pyo3-config.txt.
#
# Idempotent: re-running regenerates the derived artifacts only.
#
# The pgo+lto variant ships LLVM-22 LTO bitcode: linking requires an
# LLVM-22 lld (scripts/ld64.lld, `brew install lld`) plus -export_dynamic so
# wheels can resolve the full C-API at dlopen. The freethreaded+debug
# variant is the plain Mach-O fallback if that path regresses.
set -euo pipefail

TAG=20260807
VER=3.14.7
TRIPLE=aarch64-apple-darwin
NAME="cpython-${VER}+${TAG}-${TRIPLE}-freethreaded+pgo+lto-full"
URL="https://github.com/astral-sh/python-build-standalone/releases/download/${TAG}/${NAME}.tar.zst"

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CRATE_DIR=$(dirname "$SCRIPT_DIR")

# Explicit destination = consumer mode: the crate's THIRD-PARTY-NOTICES.txt
# already ships in the package, so it is only regenerated in repo mode.
if [ $# -ge 1 ]; then
	DEST=$(mkdir -p "$1" && cd "$1" && pwd)
	REPO_MODE=
else
	DEST="$CRATE_DIR/../../vendor"
	mkdir -p "$DEST"
	DEST=$(cd "$DEST" && pwd)
	REPO_MODE=1
fi
VENDOR="$DEST/python"

if [ ! -e "$VENDOR" ]; then
	echo "fetching ${NAME}..." >&2
	curl -fsSL "$URL" | zstd -d | tar -x -C "$DEST"
fi

# Derive layout facts from the archive: abiflags land in several names
# (python3.14td for freethreaded+debug), so glob instead of hardcoding.
STDLIB_DIR=$(dirname "$(echo "$VENDOR"/install/lib/python3.14*/os.py)")
CONFIG_DIR=$(echo "$STDLIB_DIR"/config-3.14*-darwin)
LIB_NAME=$(basename "$CONFIG_DIR"/libpython3.14*.a .a | sed 's/^lib//')
EXECUTABLE="$VENDOR/install/bin/python3.14td"
[ -x "$EXECUTABLE" ] || EXECUTABLE="$VENDOR/install/bin/python3.14t"

# In-memory stdlib: every module is compiled and marshalled by the vendored
# interpreter itself (guaranteeing a matching bytecode format), packed into
# an uncompressed blob that omp-py embeds and registers wholesale as
# frozen modules — per-interpreter machinery, so sub-interpreters work.
# Uncompressed on purpose: the OS loader mmaps the binary, so modules that
# are never imported are never paged in and startup does no decompression.
# Every C extension is statically linked into the binary; tkinter/dbm (the
# two dynload-only modules) and test/tooling packages are excluded.
# The same packer builds the crate-local module blob in build.rs.
echo "generating stdlib.bin..." >&2
"$EXECUTABLE" "$SCRIPT_DIR/pack-pymodules.py" "$STDLIB_DIR" "$VENDOR/stdlib.bin" \
	--prefix '<omp-stdlib>' \
	--exclude lib-dynload test idlelib tkinter turtledemo ensurepip \
	          site-packages __pycache__ 'config-*'

# Bundled pure-Python packages (requirements.txt): resolved with uv into
# <vendor>/bundled, stamped with the manifest text so omp-py's build script
# can verify freshness without network I/O. Built in a temp dir and swapped
# in atomically — a failed fetch never destroys the cache. In repo mode the
# third-party license texts are collected into the tracked
# THIRD-PARTY-NOTICES.txt; rerun this script after editing the manifest and
# commit the notices file when it changes.
REQ="$CRATE_DIR/requirements.txt"
BUNDLED="$VENDOR/bundled"
if grep -qvE '^[[:space:]]*(#|$)' "$REQ"; then
	if ! cmp -s "$REQ" "$BUNDLED/.requirements.stamp" 2>/dev/null; then
		echo "fetching bundled python packages..." >&2
		TMP=$(mktemp -d "$VENDOR/bundled.XXXXXX")
		trap 'rm -rf "$TMP"' EXIT
		uv pip install --link-mode=copy --python "$EXECUTABLE" --target "$TMP" -r "$REQ"
		NATIVE=$(find "$TMP" -name '*.so' -o -name '*.dylib' -o -name '*.pyd')
		if [ -n "$NATIVE" ]; then
			echo "error: $REQ pulled native extensions; only pure-Python packages can be" >&2
			echo "frozen — install native wheels into site-packages instead:" >&2
			echo "$NATIVE" >&2
			exit 1
		fi
		cp "$REQ" "$TMP/.requirements.stamp"
		rm -rf "$BUNDLED"
		mv "$TMP" "$BUNDLED"
		trap - EXIT
	fi
else
	rm -rf "$BUNDLED"
fi
if [ -n "$REPO_MODE" ]; then
	"$EXECUTABLE" "$SCRIPT_DIR/gen-py-notices.py" "$BUNDLED" "$CRATE_DIR/THIRD-PARTY-NOTICES.txt"
fi

echo "generating pyo3-config.txt..." >&2
cat > "$VENDOR/pyo3-config.txt" <<EOF
implementation=CPython
version=3.14
shared=false
abi3=false
lib_name=${LIB_NAME}
lib_dir=${CONFIG_DIR}
executable=${EXECUTABLE}
pointer_width=64
build_flags=$(case "$LIB_NAME" in *td) echo "Py_DEBUG,Py_GIL_DISABLED";; *) echo "Py_GIL_DISABLED";; esac)
suppress_build_script_link_lines=false
EOF

echo "done: ${VENDOR} (${LIB_NAME})" >&2
