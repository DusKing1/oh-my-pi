#!/usr/bin/env node
"use strict";
// Launcher for npm/bun-managed installs of omp.
//
// The real program is the native binary carried by the platform-specific
// @oh-my-pi/pi-natives-<platform>-<arch> package (an optionalDependency of
// @oh-my-pi/pi-coding-agent). This file only locates it and runs it,
// mirroring the exit status. It MUST stay dependency-free and runnable by
// both Node and Bun: bun-managed global installs execute the `bin` entry of
// this package with whatever runtime the shim picked.
const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const PLATFORM_PACKAGES = {
	"linux-x64": "@oh-my-pi/pi-natives-linux-x64",
	"linux-arm64": "@oh-my-pi/pi-natives-linux-arm64",
	"darwin-x64": "@oh-my-pi/pi-natives-darwin-x64",
	"darwin-arm64": "@oh-my-pi/pi-natives-darwin-arm64",
	"win32-x64": "@oh-my-pi/pi-natives-win32-x64",
};

const INSTALLER_HINT =
	process.platform === "win32"
		? "& ([scriptblock]::Create((irm https://omp.sh/install.ps1))) -Binary"
		: "curl -fsSL https://omp.sh/install | sh -s -- --binary";

const tag = `${process.platform}-${process.arch}`;
const pkg = PLATFORM_PACKAGES[tag];
if (!pkg) {
	console.error(`omp: no prebuilt npm binary for ${tag}.`);
	console.error(`Install the standalone binary instead: ${INSTALLER_HINT}`);
	process.exit(1);
}

const exe = process.platform === "win32" ? "omp.exe" : "omp";
let binary;
try {
	// Resolve via the leaf's package.json: subpath-agnostic, so it keeps
	// working even if an `exports` map is ever added to the leaf.
	binary = path.join(path.dirname(require.resolve(`${pkg}/package.json`)), "bin", exe);
} catch {
	binary = undefined;
}
if (binary === undefined || !fs.existsSync(binary)) {
	console.error(`omp: ${pkg} is not installed (optional dependencies may have been skipped).`);
	console.error("Reinstall with optional dependencies enabled, or use the standalone binary:");
	console.error(`  ${INSTALLER_HINT}`);
	process.exit(1);
}

// stdio: "inherit" hands the tty straight to the binary. No signal forwarding
// is needed: in the TUI the terminal is in raw mode, so Ctrl+C is a keypress
// delivered to the child; in cooked mode (plain subcommands) the signal goes
// to the whole foreground process group, child included.
const result = spawnSync(binary, process.argv.slice(2), { stdio: "inherit" });
if (result.error) {
	console.error(`omp: failed to launch ${binary}: ${result.error.message}`);
	process.exit(1);
}
if (result.signal) {
	// Re-raise so the shell observes the true 128+n termination status.
	process.kill(process.pid, result.signal);
}
process.exit(result.status ?? 1);
