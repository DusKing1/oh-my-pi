#!/usr/bin/env python3
"""Assembles the publishable npm packages for a release.

The Rust omp is distributed on npm codex-style: a launcher package whose
platform binaries ride in per-platform optionalDependencies. Three package
names are FROZEN because every deployed TypeScript-era `omp update` hard-pins
them at the target version (buildVersionedPackageInstallArgs in
oh-my-pi/packages/coding-agent/src/cli/update-cli.ts) — a release that stops
publishing any of them strands those installs with a failed update:

  @oh-my-pi/pi-coding-agent          launcher (bin/omp.js) + optionalDependencies
  @oh-my-pi/pi-natives               lockstep version sentinel (contentless)
  @oh-my-pi/pi-natives-<tag>         the native binary for one platform

Contracts the release MUST keep (see npm/README.md):
  - `omp --version` prints `omp/X.Y.Z` — the updater's post-install
    verification parses exactly that shape and rolls back on mismatch.
  - pi-coding-agent's manifest carries `"omp": {"dist": "npm"}` so hardened
    updaters keep routing bun/npm installs through the package manager
    across the major bump instead of forcing a binary migration.
  - Publish order: leaves first, then pi-natives, then pi-coding-agent —
    the core must never be live while its pinned companions are missing.

Usage: gen-npm-packages.py --version X.Y.Z --binaries DIR [--out dist/npm]

DIR must hold the release binaries under their GitHub asset names
(omp-linux-x64, omp-linux-arm64, omp-darwin-x64, omp-darwin-arm64,
omp-windows-x64.exe). Missing binaries are a hard error: publishing a
partial platform set breaks the pinned-leaf install on the missing tags.
"""

import argparse
import json
import re
import shutil
import stat
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCOPE = "@oh-my-pi"
CORE = f"{SCOPE}/pi-coding-agent"
SENTINEL = f"{SCOPE}/pi-natives"
LICENSE = "MIT"
HOMEPAGE = "https://omp.sh/"
REPOSITORY = "https://github.com/can1357/omp"

# tag -> (npm "os" value, npm "cpu" value, GitHub release asset name)
TARGETS = {
    "linux-x64": ("linux", "x64", "omp-linux-x64"),
    "linux-arm64": ("linux", "arm64", "omp-linux-arm64"),
    "darwin-x64": ("darwin", "x64", "omp-darwin-x64"),
    "darwin-arm64": ("darwin", "arm64", "omp-darwin-arm64"),
    "win32-x64": ("win32", "x64", "omp-windows-x64.exe"),
}

VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?$")


def base_manifest(name: str, version: str, description: str) -> dict:
    return {
        "name": name,
        "version": version,
        "description": description,
        "license": LICENSE,
        "homepage": HOMEPAGE,
        "repository": {"type": "git", "url": f"git+{REPOSITORY}.git"},
    }


def write_package(out_dir: Path, manifest: dict, readme: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "package.json").write_text(json.dumps(manifest, indent="\t") + "\n")
    (out_dir / "README.md").write_text(readme)


def emit_leaf(out: Path, tag: str, version: str, binaries: Path) -> str:
    os_name, cpu, asset = TARGETS[tag]
    name = f"{SENTINEL}-{tag}"
    src = binaries / asset
    if not src.is_file():
        sys.exit(f"error: missing release binary {src} for {name}")
    pkg_dir = out / f"pi-natives-{tag}"
    exe = "omp.exe" if os_name == "win32" else "omp"
    bin_dir = pkg_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, bin_dir / exe)
    (bin_dir / exe).chmod(
        (bin_dir / exe).stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    )

    manifest = base_manifest(name, version, f"omp native binary for {tag}")
    manifest["os"] = [os_name]
    manifest["cpu"] = [cpu]
    manifest["files"] = ["bin/"]
    # No "exports": the launcher resolves `<name>/package.json` through the
    # legacy algorithm and joins bin/ itself.
    write_package(
        pkg_dir,
        manifest,
        f"# {name}\n\nomp native binary for {tag}. Installed as an optional dependency of"
        f" `{CORE}`; not useful on its own.\n\nGenerated at release time by"
        " `scripts/gen-npm-packages.py`.\n",
    )
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", required=True, help="release version (X.Y.Z)")
    parser.add_argument(
        "--binaries", required=True, type=Path, help="directory holding the release binaries"
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "dist" / "npm", help="output directory"
    )
    args = parser.parse_args()

    if not VERSION_RE.match(args.version):
        sys.exit(f"error: invalid version {args.version!r} (expected X.Y.Z)")
    version: str = args.version

    out: Path = args.out
    if out.exists():
        shutil.rmtree(out)

    leaves = [emit_leaf(out, tag, version, args.binaries) for tag in TARGETS]

    # Contentless sentinel: nothing loads it, but deployed updaters install it
    # pinned to the release version, so it must exist at every version.
    write_package(
        out / "pi-natives",
        base_manifest(
            SENTINEL, version, "Lockstep version sentinel for omp platform binary packages"
        ),
        f"# {SENTINEL}\n\nEmpty by design. Deployed `omp update` clients pin this package in"
        f" lockstep with `{CORE}`, so it is published at every version; the actual binaries"
        f" live in the `{SENTINEL}-<platform>-<arch>` packages.\n",
    )

    core = base_manifest(CORE, version, "omp — a coding agent with the IDE wired in")
    core["bin"] = {"omp": "bin/omp.js"}
    core["files"] = ["bin/"]
    core["engines"] = {"node": ">=20"}
    # Read by `omp update` (resolveReleaseDist): "npm" keeps bun/npm-managed
    # installs on their package manager across major bumps; removing this
    # field makes cross-major updaters migrate users to the standalone binary.
    core["omp"] = {"dist": "npm"}
    core["dependencies"] = {SENTINEL: version}
    core["optionalDependencies"] = {leaf: version for leaf in leaves}
    core_dir = out / "pi-coding-agent"
    write_package(
        core_dir,
        core,
        f"# {CORE}\n\nnpm distribution of [omp]({HOMEPAGE}). The `omp` bin entry is a launcher"
        " that runs the native binary installed by the matching platform package.\n\nPrefer the"
        f" standalone install? `curl -fsSL https://omp.sh/install | sh -s -- --binary`\n",
    )
    launcher = REPO_ROOT / "npm" / "pi-coding-agent" / "bin" / "omp.js"
    (core_dir / "bin").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(launcher, core_dir / "bin" / "omp.js")

    print(f"Generated {len(leaves) + 2} packages in {out}")
    print("Publish order: leaves, then pi-natives, then pi-coding-agent:")
    for leaf in leaves:
        print(f"  npm publish --access public {out}/pi-natives-{leaf.rsplit('-', 2)[-2]}-{leaf.rsplit('-', 1)[-1]}")
    print(f"  npm publish --access public {out}/pi-natives")
    print(f"  npm publish --access public {out}/pi-coding-agent")


if __name__ == "__main__":
    main()
