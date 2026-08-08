// Project-local `rust` tool: compile and run scratch Rust directly with
// `rustc`. Imports from workspace crates and `[workspace.dependencies]` use
// linkable artifacts already present under `target/debug`; this tool never
// invokes Cargo, resolves packages, downloads crates, or rebuilds dependencies.
//
// Each invocation writes only its source and executable to a temporary
// directory. A shared symlink index gives rustc one dependency search path.

import {
	mkdir,
	mkdtemp,
	readdir,
	readFile,
	rm,
	stat,
	symlink,
	writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, dirname, join } from "node:path";

/** Path segments that never name an external crate. */
const NOT_CRATES: Record<string, true> = {
	std: true,
	core: true,
	alloc: true,
	crate: true,
	self: true,
	super: true,
	bool: true,
	char: true,
	str: true,
	f16: true,
	f32: true,
	f64: true,
	f128: true,
	u8: true,
	u16: true,
	u32: true,
	u64: true,
	u128: true,
	usize: true,
	i8: true,
	i16: true,
	i32: true,
	i64: true,
	i128: true,
	isize: true,
};

/** Minimal slice of the host schema builder (`pi.zod`) this tool uses. */
interface Schema {
	describe(text: string): Schema;
	optional(): Schema;
}
interface SchemaBuilder {
	object(shape: Record<string, Schema>): Schema;
	string(): Schema;
	boolean(): Schema;
	number(): Schema;
	array(item: Schema): Schema;
}
/** Minimal slice of `CustomToolAPI` this tool uses. */
interface ToolHost {
	cwd: string;
	zod: SchemaBuilder;
}
interface ToolUpdate {
	content: { type: "text"; text: string }[];
	details?: Record<string, unknown>;
}
interface RustParams {
	code: string;
	args?: string[];
	release?: boolean;
	timeout?: number;
}

interface CrateSpec {
	/** Artifact crate name, which may differ from an aliased dependency key. */
	artifact: string;
	/** Cargo package name, used only to locate an existing build artifact. */
	package: string;
}

/**
 * Crates scratch code may import, keyed by their Rust ident.
 *
 * Sources are `[workspace.dependencies]` plus every `crates/*` member.
 */
async function collectCrates(root: string): Promise<Map<string, CrateSpec>> {
	const crates = new Map<string, CrateSpec>();
	const rootToml = await readFile(join(root, "Cargo.toml"), "utf8");
	let inDeps = false;
	for (const raw of rootToml.split("\n")) {
		const line = raw.trim();
		if (line.startsWith("[")) {
			inDeps = line === "[workspace.dependencies]";
			continue;
		}
		if (!inDeps || !line || line.startsWith("#")) continue;
		const match = line.match(/^([A-Za-z0-9_-]+)\s*=\s*(.+)$/);
		if (!match) continue;
		const packageName =
			match[2].match(/\bpackage\s*=\s*"([^"]+)"/)?.[1] ?? match[1];
		crates.set(match[1].replace(/-/g, "_"), {
			artifact: packageName.replace(/-/g, "_"),
			package: packageName,
		});
	}
	for (const dir of await readdir(join(root, "crates"))) {
		let manifest: string;
		try {
			manifest = await readFile(
				join(root, "crates", dir, "Cargo.toml"),
				"utf8",
			);
		} catch {
			continue;
		}
		const name = manifest.match(/^name\s*=\s*"([^"]+)"/m)?.[1];
		if (!name) continue;
		const ident = name.replace(/-/g, "_");
		crates.set(ident, { artifact: ident, package: name });
	}
	return crates;
}

/** Crate idents the snippet references via `use`, `extern crate`, or paths. */
function crateRefs(code: string): Set<string> {
	const refs = new Set<string>();
	const local = new Set<string>();
	for (const m of code.matchAll(/\bmod\s+([a-z_][a-z0-9_]*)/g)) local.add(m[1]);
	for (const m of code.matchAll(
		/\b(?:use|extern\s+crate)\s+([a-z_][a-z0-9_]*)/g,
	))
		refs.add(m[1]);
	for (const m of code.matchAll(/\b([a-z_][a-z0-9_]*)\s*::/g)) refs.add(m[1]);
	for (const ident of refs)
		if (NOT_CRATES[ident] || local.has(ident)) refs.delete(ident);
	return refs;
}

interface UsedCrate extends CrateSpec {
	ident: string;
}

interface Artifact {
	path: string;
	preferred: boolean;
	rlib: boolean;
	mtimeMs: number;
}

interface ArtifactIndex {
	dir: string;
	paths: Set<string>;
}

interface ResolvedCrate {
	crate: UsedCrate;
	artifact: Artifact;
}

const ARTIFACT_GLOBS = [
	"target/debug/build/*/*/out/*.{rlib,dylib,so,dll,a,lib}",
	"target/debug/deps/*.{rlib,dylib,so,dll,a,lib}",
];

function errorCode(error: unknown): string | undefined {
	if (typeof error !== "object" || error === null) return undefined;
	const code = Reflect.get(error, "code");
	return typeof code === "string" ? code : undefined;
}

async function linkArtifact(
	source: string,
	destination: string,
): Promise<void> {
	try {
		await symlink(source, destination);
		return;
	} catch (error) {
		if (errorCode(error) !== "EEXIST") throw error;
	}

	try {
		await stat(destination);
		return;
	} catch (error) {
		if (errorCode(error) !== "ENOENT") throw error;
	}

	await rm(destination, { force: true });
	try {
		await symlink(source, destination);
	} catch (error) {
		// Another tool process may have repaired the same shared link.
		if (errorCode(error) !== "EEXIST") throw error;
	}
}

async function scanArtifactPaths(root: string): Promise<string[]> {
	const paths = new Set<string>();
	for (const pattern of ARTIFACT_GLOBS) {
		const glob = new Bun.Glob(pattern);
		for await (const path of glob.scan({ cwd: root, onlyFiles: true })) {
			paths.add(join(root, path));
		}
	}
	return [...paths];
}

async function buildArtifactIndex(root: string): Promise<ArtifactIndex> {
	const [paths, rustcInfo] = await Promise.all([
		scanArtifactPaths(root),
		readFile(join(root, "target", ".rustc_info.json"), "utf8").catch(() => ""),
	]);
	const key = Bun.hash(`${root}\0${rustcInfo}`).toString(16);
	const dir = join(tmpdir(), `omp-rust-artifacts-${key}`);
	await mkdir(dir, { recursive: true });

	const links = new Map<string, string>();
	for (const path of paths) {
		const name = basename(path);
		if (!links.has(name)) links.set(name, path);
	}
	await Promise.all(
		[...links].map(([name, source]) => linkArtifact(source, join(dir, name))),
	);
	return { dir, paths: new Set(paths) };
}

async function preferredFingerprint(path: string): Promise<boolean> {
	const outDir = dirname(path);
	if (basename(outDir) !== "out") return false;
	const fingerprintDir = join(dirname(outDir), "fingerprint");
	try {
		const name = (await readdir(fingerprintDir)).find(
			(entry) => entry.startsWith("lib-") && entry.endsWith(".json"),
		);
		if (!name) return false;
		const fingerprint = await readFile(join(fingerprintDir, name), "utf8");
		return (
			fingerprint.includes('"compile_kind":0') &&
			!fingerprint.includes("link-arg=")
		);
	} catch {
		return false;
	}
}

async function artifactCandidates(
	root: string,
	crate: UsedCrate,
): Promise<Artifact[]> {
	const extension = "{rlib,dylib,so,dll}";
	const patterns = [
		`target/debug/build/${crate.package}/*/out/*${crate.artifact}*.${extension}`,
		`target/debug/deps/*${crate.artifact}*.${extension}`,
		`target/debug/*${crate.artifact}*.${extension}`,
	];
	const escapedArtifact = crate.artifact.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
	const namePattern = new RegExp(
		`^(?:lib)?${escapedArtifact}(?:-[0-9a-f]+)?\\.(?:rlib|dylib|so|dll)$`,
	);
	const paths = new Set<string>();
	for (const pattern of patterns) {
		const glob = new Bun.Glob(pattern);
		for await (const path of glob.scan({ cwd: root, onlyFiles: true })) {
			const absolute = join(root, path);
			if (namePattern.test(basename(absolute))) paths.add(absolute);
		}
	}

	const artifacts = await Promise.all(
		[...paths].map(async (path): Promise<Artifact> => {
			const [info, preferred] = await Promise.all([
				stat(path),
				preferredFingerprint(path),
			]);
			return {
				path,
				preferred,
				rlib: path.endsWith(".rlib"),
				mtimeMs: info.mtimeMs,
			};
		}),
	);
	return artifacts.sort(
		(a, b) =>
			Number(b.preferred) - Number(a.preferred) ||
			Number(b.rlib) - Number(a.rlib) ||
			b.mtimeMs - a.mtimeMs,
	);
}

async function resolveCrate(
	root: string,
	crate: UsedCrate,
): Promise<ResolvedCrate> {
	const [artifact] = await artifactCandidates(root, crate);
	if (!artifact) {
		throw new Error(
			`no linkable debug artifact for \`${crate.ident}\` under target/debug; ` +
				"this tool never builds dependencies, so the workspace must build it first",
		);
	}
	return { crate, artifact };
}

interface RunResult {
	stdout: string;
	stderr: string;
	code: number;
	timedOut: boolean;
}

/** Spawn with timeout + cancellation; SIGKILL on either. */
async function run(
	argv: string[],
	opts: {
		cwd: string;
		env: Record<string, string>;
		signal?: AbortSignal;
		timeoutMs: number;
	},
): Promise<RunResult> {
	const proc = Bun.spawn(argv, {
		cwd: opts.cwd,
		env: { ...process.env, ...opts.env },
		stdin: "ignore",
		stdout: "pipe",
		stderr: "pipe",
	});
	let timedOut = false;
	const timer = setTimeout(() => {
		timedOut = true;
		proc.kill(9);
	}, opts.timeoutMs);
	const onAbort = () => proc.kill(9);
	opts.signal?.addEventListener("abort", onAbort, { once: true });
	try {
		const [stdout, stderr, code] = await Promise.all([
			new Response(proc.stdout).text(),
			new Response(proc.stderr).text(),
			proc.exited,
		]);
		return { stdout, stderr, code, timedOut };
	} finally {
		clearTimeout(timer);
		opts.signal?.removeEventListener("abort", onAbort);
	}
}

function resultText(result: RunResult): string {
	return [result.stdout, result.stderr && `--- stderr ---\n${result.stderr}`]
		.filter(Boolean)
		.join("\n")
		.trim();
}

/** Creates the project-local direct-rustc scratch tool. */
const factory = (pi: ToolHost) => {
	let artifactIndex: Promise<ArtifactIndex> | undefined;
	const loadArtifactIndex = (refresh = false): Promise<ArtifactIndex> => {
		if (refresh || !artifactIndex) artifactIndex = buildArtifactIndex(pi.cwd);
		return artifactIndex;
	};

	return {
		name: "rust",
		label: "Rust Scratch",
		description:
			"Compile and run a throwaway Rust program directly with rustc (edition 2024, " +
			"optimized by default). Pass a full program or a bare snippet — snippets without " +
			"`fn main` are wrapped in one. Imports of workspace crates (`use omp_core::…`, " +
			"`use omp_tui::…`) and `[workspace.dependencies]` (`use smallvec::…`) reuse " +
			"linkable artifacts already under `target/debug`; Cargo is never invoked and " +
			"missing dependencies are never built or downloaded. Program stdout/stderr is " +
			"returned; non-zero exit, compile errors, and timeouts surface as tool errors.",
		parameters: pi.zod.object({
			code: pi.zod
				.string()
				.describe("Rust source: full program or bare statements"),
			args: pi.zod
				.array(pi.zod.string())
				.optional()
				.describe("argv passed to the program"),
			release: pi.zod
				.boolean()
				.optional()
				.describe("optimize scratch code (default true)"),
			timeout: pi.zod
				.number()
				.optional()
				.describe("compile+run timeout in seconds (default 300)"),
		}),

		async execute(
			_toolCallId: string,
			params: RustParams,
			onUpdate: ((update: ToolUpdate) => void) | undefined,
			_ctx: unknown,
			signal?: AbortSignal,
		) {
			const root = pi.cwd;
			const available = await collectCrates(root);
			const used: UsedCrate[] = [];
			for (const ident of crateRefs(params.code)) {
				const crate = available.get(ident);
				if (crate) used.push({ ident, ...crate });
			}
			const depNames = used.map(({ ident }) => ident);

			const resolved = await Promise.all(
				used.map((crate) => resolveCrate(root, crate)),
			);
			let index: ArtifactIndex | undefined;
			if (resolved.length) {
				index = await loadArtifactIndex();
				const cachedRoot = join(root, "target", "debug");
				if (
					resolved.some(
						({ artifact }) =>
							(artifact.path.startsWith(join(cachedRoot, "build")) ||
								artifact.path.startsWith(join(cachedRoot, "deps"))) &&
							!index?.paths.has(artifact.path),
					)
				) {
					index = await loadArtifactIndex(true);
				}
			}

			let src = params.code;
			if (!/\bfn\s+main\s*\(/.test(src)) src = `fn main() {\n${src}\n}`;
			src = `#![allow(warnings)]\n${src}`;

			const dir = await mkdtemp(join(tmpdir(), "omp-rust-"));
			try {
				const sourcePath = join(dir, "main.rs");
				const binaryPath = join(
					dir,
					process.platform === "win32" ? "scratch.exe" : "scratch",
				);
				await writeFile(sourcePath, src);

				const env: Record<string, string> = {};
				const argv = [
					"rustc",
					"--edition=2024",
					"--crate-name",
					"omp_scratch",
					"--color=never",
					"-Awarnings",
				];
				if (params.release !== false) {
					argv.push(
						"-Copt-level=2",
						"-Cdebug-assertions=off",
						"-Coverflow-checks=off",
					);
				}
				if (index) {
					argv.push(`-Ldependency=${index.dir}`, `-Lnative=${index.dir}`);
				}
				for (const { crate, artifact } of resolved) {
					argv.push("--extern", `${crate.ident}=${artifact.path}`);
				}

				if (depNames.includes("omp_py") || depNames.includes("pyo3")) {
					const configPath = join(root, "vendor/python/pyo3-config.txt");
					const config = await readFile(configPath, "utf8");
					const libDir = config.match(/^lib_dir=(.+)$/m)?.[1];
					if (!libDir) throw new Error(`missing lib_dir in ${configPath}`);
					env.PYO3_CONFIG_FILE = configPath;
					argv.push(
						`-Lnative=${libDir}`,
						`-Clink-arg=--ld-path=${join(root, "scripts/ld64.lld")}`,
						"-Clink-arg=-Wl,-export_dynamic",
					);
				}
				argv.push(sourcePath, "-o", binaryPath);

				onUpdate?.({
					content: [
						{
							type: "text",
							text: `compiling with rustc (deps: ${depNames.join(", ") || "none"})…`,
						},
					],
					details: { deps: depNames },
				});

				const timeoutMs = (params.timeout ?? 300) * 1000;
				const started = performance.now();
				const compiled = await run(argv, { cwd: dir, env, signal, timeoutMs });
				const compileText = resultText(compiled);
				if (compiled.timedOut) {
					throw new Error(
						`timed out after ${params.timeout ?? 300}s\n${compileText}`,
					);
				}
				if (compiled.code !== 0) {
					throw new Error(
						compileText || `rustc exited with code ${compiled.code}`,
					);
				}

				const remainingMs = timeoutMs - (performance.now() - started);
				if (remainingMs <= 0)
					throw new Error(`timed out after ${params.timeout ?? 300}s`);
				const result = await run([binaryPath, ...(params.args ?? [])], {
					cwd: dir,
					env,
					signal,
					timeoutMs: remainingMs,
				});
				const text = resultText(result);
				if (result.timedOut) {
					throw new Error(`timed out after ${params.timeout ?? 300}s\n${text}`);
				}
				if (result.code !== 0)
					throw new Error(text || `exit code ${result.code}`);

				const elapsed = Number(
					((performance.now() - started) / 1000).toFixed(2),
				);
				return {
					content: [{ type: "text", text: text || "(no output)" }],
					details: {
						deps: depNames,
						release: params.release !== false,
						seconds: elapsed,
					},
				};
			} finally {
				await rm(dir, { recursive: true, force: true });
			}
		},
	};
};

export default factory;
