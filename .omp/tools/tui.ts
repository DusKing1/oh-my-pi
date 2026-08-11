// Project-local `tui` tool: run and debug omp-tui apps (examples or cargo
// bins) headlessly. Each session spawns the target on a Bun-native PTY — a
// real controlling terminal, so capability probes, SIGWINCH resizes, and
// immediate-mode hosts all behave as in production — with `OMP_TUI_DEBUG`
// pointed at a unix socket every omp-tui host serves. The wire speaks the
// crate's `TerminalEvent`: injected input rides the terminal's own event
// mailbox, screenshots answer from the renderer's last paint, and
// `frame`/`tree`/`values` are mailbox queries answered by `App` hosts
// (immediate-mode hosts let them time out server-side).
//
// Sessions live in this module for the lifetime of the agent session; the
// child and its terminal are torn down on `stop` or shutdown.

import { mkdtempSync, rmSync } from "node:fs";
import * as net from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";

/** Minimal slice of the host schema builder (`omp.zod`) this tool uses. */
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
interface ExecResult {
	stdout: string;
	stderr: string;
	code: number | null;
	killed?: boolean;
}
/** Minimal slice of `CustomToolAPI` this tool uses. */
interface ToolHost {
	cwd: string;
	zod: SchemaBuilder;
	exec(
		command: string,
		args: string[],
		options?: { cwd?: string; signal?: AbortSignal },
	): Promise<ExecResult>;
}
interface ToolUpdate {
	content: { type: "text"; text: string }[];
	details?: Record<string, unknown>;
}
interface ToolResult {
	content: { type: "text"; text: string }[];
	details?: Record<string, unknown>;
}

interface TuiParams {
	op:
		| "start"
		| "stop"
		| "list"
		| "text"
		| "frame"
		| "tree"
		| "values"
		| "info"
		| "keys"
		| "type"
		| "paste"
		| "mouse"
		| "send"
		| "resize"
		| "raw";
	name?: string;
	example?: string;
	bin?: string;
	args?: string[];
	rows?: number;
	cols?: number;
	build?: boolean;
	keys?: string;
	text?: string;
	x?: number;
	y?: number;
	action?: string;
	peek?: number;
	clear?: boolean;
	timeout?: number;
}

// ─── Sessions ────────────────────────────────────────────────────────────────

interface Waiter {
	resolve(value: Record<string, unknown>): void;
	reject(error: Error): void;
	timer: NodeJS.Timeout;
}

/** The slice of `Bun.spawn`'s PTY handle this tool touches. */
interface TerminalHandle {
	write(data: string | Uint8Array): void;
	resize(cols: number, rows: number): void;
	close(): void;
}

/** The slice of Bun's PTY-backed subprocess this tool touches. */
interface Child {
	pid: number;
	exited: Promise<number>;
	terminal: TerminalHandle;
	kill(signal?: number | NodeJS.Signals): void;
}

interface Session {
	name: string;
	target: string;
	proc: Child;
	cols: number;
	rows: number;
	dir: string;
	sock: net.Socket | null;
	sockBuf: string;
	waiters: Waiter[];
	raw: Buffer[];
	rawBytes: number;
	exit: number | null;
}

const sessions = new Map<string, Session>();

/** Appends one PTY chunk to the session's capped raw capture. */
function capture(session: Session, chunk: Buffer) {
	session.raw.push(chunk);
	session.rawBytes += chunk.length;
	// Cap the capture at 8 MiB, dropping the oldest chunks.
	while (session.rawBytes > 8 * 1024 * 1024 && session.raw.length > 1) {
		session.rawBytes -= session.raw[0].length;
		session.raw.shift();
	}
}

function connectSocket(session: Session, path: string, timeoutMs: number): Promise<boolean> {
	const deadline = Date.now() + timeoutMs;
	const { promise, resolve } = Promise.withResolvers<boolean>();
	const attempt = () => {
		const sock = net.createConnection(path);
		sock.on("connect", () => {
			sock.setEncoding("utf8");
			sock.on("data", (data: string) => {
				session.sockBuf += data;
				for (;;) {
					const index = session.sockBuf.indexOf("\n");
					if (index < 0) return;
					const line = session.sockBuf.slice(0, index);
					session.sockBuf = session.sockBuf.slice(index + 1);
					const waiter = session.waiters.shift();
					if (!waiter) continue;
					clearTimeout(waiter.timer);
					try {
						waiter.resolve(JSON.parse(line));
					} catch (error) {
						waiter.reject(new Error(`bad response line: ${error}`));
					}
				}
			});
			sock.on("close", () => {
				if (session.sock === sock) session.sock = null;
			});
			sock.on("error", () => {});
			session.sock = sock;
			resolve(true);
		});
		sock.on("error", () => {
			sock.destroy();
			if (Date.now() >= deadline || session.exit !== null) resolve(false);
			else setTimeout(attempt, 150);
		});
	};
	attempt();
	return promise;
}

/** Sends one debug request and awaits its response line. */
function request(
	session: Session,
	body: Record<string, unknown>,
	timeoutMs = 10_000,
): Promise<Record<string, unknown>> {
	const sock = session.sock;
	if (!sock) {
		return Promise.reject(
			new Error(
				`session "${session.name}" has no debug socket — the app is not an ` +
					"omp-tui host (or exited). raw/send/resize/stop still work.",
			),
		);
	}
	const { promise, resolve, reject } = Promise.withResolvers<Record<string, unknown>>();
	const timer = setTimeout(() => {
		const index = session.waiters.findIndex((waiter) => waiter.timer === timer);
		if (index >= 0) session.waiters.splice(index, 1);
		reject(new Error(`debug request timed out: ${JSON.stringify(body)}`));
	}, timeoutMs);
	session.waiters.push({ resolve, reject, timer });
	sock.write(`${JSON.stringify(body)}\n`);
	return promise;
}

function need(session: string | undefined): Session {
	const name = session ?? "main";
	const found = sessions.get(name);
	if (!found) {
		const names = [...sessions.keys()].join(", ") || "none";
		throw new Error(`no session "${name}" (running: ${names})`);
	}
	return found;
}

function sleep(ms: number): Promise<null> {
	const { promise, resolve } = Promise.withResolvers<null>();
	setTimeout(() => resolve(null), ms);
	return promise;
}

async function stopSession(session: Session): Promise<number | null> {
	try {
		if (session.sock) {
			await request(session, { op: "quit" }, 2_000);
		} else if (session.exit === null) {
			// Non-omp-tui apps have no debug socket; Ctrl-C is the
			// conventional quit chord.
			session.proc.terminal.write("\x03");
		}
	} catch {
		// Fall through to signals.
	}
	const exited = await Promise.race([session.proc.exited, sleep(2_000)]);
	if (exited === null) {
		session.proc.kill("SIGKILL");
		await session.proc.exited.catch(() => {});
	}
	session.sock?.destroy();
	session.proc.terminal.close();
	rmSync(session.dir, { recursive: true, force: true });
	sessions.delete(session.name);
	return exited ?? -9;
}

// ─── Response narrowing ──────────────────────────────────────────────────────

/** The string rows of a `lines` response field; anything else is empty. */
function stringLines(value: unknown): string[] {
	if (!Array.isArray(value)) return [];
	return value.filter((line): line is string => typeof line === "string");
}

/** Reads one property of an unknown JSON value; `undefined` when absent. */
function field(value: unknown, name: string): unknown {
	if (value && typeof value === "object" && name in value) {
		return Reflect.get(value, name);
	}
	return undefined;
}

// ─── Rendering helpers ───────────────────────────────────────────────────────

function screenshotText(response: Record<string, unknown>): string {
	const lines = stringLines(response.lines);
	const header =
		`── viewport (window_top=${response.window_top}` +
		`${response.alt_screen ? ", alt screen" : ""}` +
		`${response.cursor ? `, cursor=${JSON.stringify(response.cursor)}` : ""}) ──`;
	return `${header}\n${lines.map((line) => `│${line}`).join("\n")}`;
}

/** Renders one `tree` response node (and its children) as outline rows. */
function renderTree(node: unknown, depth: number, out: string[]) {
	if (!node || typeof node !== "object") return;
	const rect = field(node, "rect");
	const id = field(node, "id");
	const flags = [
		field(node, "focused") === true ? "FOCUSED" : "",
		field(node, "focusable") === true ? "focusable" : "",
		field(node, "hidden") === true ? "hidden" : "",
	]
		.filter(Boolean)
		.join(" ");
	out.push(
		"  ".repeat(depth) +
			`${field(node, "kind")}${typeof id === "string" ? `#${id}` : ""}` +
			(Array.isArray(rect) ? ` [${rect[0]},${rect[1]} ${rect[2]}x${rect[3]}]` : "") +
			(flags ? `  ${flags}` : ""),
	);
	const children = field(node, "children");
	if (Array.isArray(children)) {
		for (const child of children) renderTree(child, depth + 1, out);
	}
}

const SEQUENCES: Record<string, string> = {
	alt_enter: "\x1b[?1049h",
	alt_leave: "\x1b[?1049l",
	clear_scrollback: "\x1b[3J",
	sync_begin: "\x1b[?2026h",
	sync_end: "\x1b[?2026l",
	mouse_on: "\x1b[?1003h",
	mouse_off: "\x1b[?1003l",
	cursor_hide: "\x1b[?25l",
	cursor_show: "\x1b[?25h",
};

function count(haystack: Buffer, needle: string): number {
	const bytes = Buffer.from(needle, "latin1");
	let total = 0;
	let from = 0;
	for (;;) {
		const index = haystack.indexOf(bytes, from);
		if (index < 0) return total;
		total += 1;
		from = index + bytes.length;
	}
}

/** Escapes control bytes so raw terminal output is printable. */
function visible(bytes: Buffer): string {
	let out = "";
	for (const byte of bytes) {
		if (byte === 0x1b) out += "\\e";
		else if (byte === 0x0a) out += "\n";
		else if (byte === 0x0d) out += "\\r";
		else if (byte < 0x20 || byte === 0x7f)
			out += `\\x${byte.toString(16).padStart(2, "0")}`;
		else out += String.fromCharCode(byte);
	}
	return out;
}

/** Unescapes `\e`, `\r`, `\n`, `\t`, and `\xNN` in a `send` payload. */
function unescapeBytes(text: string): Buffer {
	const out: number[] = [];
	for (let index = 0; index < text.length; index++) {
		if (text[index] !== "\\") {
			out.push(...Buffer.from(text[index], "utf8"));
			continue;
		}
		const next = text[index + 1];
		if (next === "e") {
			out.push(0x1b);
			index++;
		} else if (next === "r") {
			out.push(0x0d);
			index++;
		} else if (next === "n") {
			out.push(0x0a);
			index++;
		} else if (next === "t") {
			out.push(0x09);
			index++;
		} else if (next === "x") {
			out.push(Number.parseInt(text.slice(index + 2, index + 4), 16));
			index += 3;
		} else {
			out.push(0x5c);
		}
	}
	return Buffer.from(out);
}

// ─── Tool ────────────────────────────────────────────────────────────────────

const factory = (omp: ToolHost) => {
	const startSession = async (
		params: TuiParams,
		onUpdate?: (update: ToolUpdate) => void,
	): Promise<string> => {
		const name = params.name ?? "main";
		if (sessions.has(name)) {
			throw new Error(`session "${name}" already running; stop it first`);
		}
		if (!params.example && !params.bin) {
			throw new Error("start needs `example` or `bin`");
		}
		const target = params.example ?? params.bin ?? "";
		if (params.build !== false) {
			onUpdate?.({ content: [{ type: "text", text: `building ${target}…` }] });
			const kind = params.example ? "--example" : "--bin";
			const built = await omp.exec("cargo", ["build", kind, target], { cwd: omp.cwd });
			if (built.code !== 0) {
				throw new Error(`cargo build failed:\n${built.stderr.slice(-4000)}`);
			}
		}
		const binary = params.example
			? join(omp.cwd, "target", "debug", "examples", target)
			: join(omp.cwd, "target", "debug", target);

		const rows = params.rows ?? 30;
		const cols = params.cols ?? 100;
		const dir = mkdtempSync(join(tmpdir(), `omp-tui-${name}-`));
		const sockPath = join(dir, "debug.sock");
		// The PTY data callback closes over `session`; Bun.spawn returns
		// synchronously and the callback fires on the event loop, so the
		// binding is assigned before the first chunk can arrive.
		let session: Session;
		const proc: Child = Bun.spawn([binary, ...(params.args ?? [])], {
			cwd: omp.cwd,
			env: {
				...process.env,
				OMP_TUI_DEBUG: sockPath,
				TERM: "xterm-256color",
				COLORTERM: "truecolor",
			},
			terminal: {
				cols,
				rows,
				data(_terminal: TerminalHandle, chunk: Buffer) {
					capture(session, chunk);
				},
			},
		});
		session = {
			name,
			target,
			proc,
			cols,
			rows,
			dir,
			sock: null,
			sockBuf: "",
			waiters: [],
			raw: [],
			rawBytes: 0,
			exit: null,
		};
		proc.exited.then((code) => {
			session.exit = code;
		});
		sessions.set(name, session);

		const deadline = Date.now() + (params.timeout ?? 15) * 1000;
		const connected = await connectSocket(
			session,
			sockPath,
			(params.timeout ?? 15) * 1000,
		);
		if (session.exit !== null) {
			const tail = visible(Buffer.concat(session.raw)).slice(-3000);
			await stopSession(session).catch(() => {});
			throw new Error(
				`"${target}" exited immediately (code ${session.exit}).\n` +
					`terminal tail: ${tail || "(empty)"}`,
			);
		}
		let text = `session "${name}": ${target} pid=${proc.pid} pty=${cols}x${rows}`;
		if (connected) {
			// The socket binds at terminal entry, before the first frame
			// paints; retry until the snapshot exists so `start` reliably
			// returns the opening screenshot.
			let shot = await request(session, { op: "text" });
			while (
				shot.ok === false &&
				String(shot.error ?? "").includes("no frame painted yet") &&
				session.exit === null &&
				Date.now() < deadline
			) {
				await sleep(50);
				shot = await request(session, { op: "text" });
			}
			text += `\n${screenshotText(shot)}`;
		} else {
			text +=
				"\n(no debug socket: app is not an omp-tui host; " +
				"raw/send/resize/stop only)";
		}
		return text;
	};

	return {
		name: "tui",
		label: "TUI Debug",
		description:
			"Run and debug omp-tui apps (cargo examples or bins) headlessly on a real " +
			"PTY plus the OMP_TUI_DEBUG socket. Ops: start (example|bin, rows/cols, " +
			"args, build), text (viewport screenshot as text), frame (full document), " +
			"tree (component tree with ids/rects/focus), values (widget values JSON), " +
			"info, keys (spec like \"tab C-c enter 'literal'\"), type (literal text " +
			"through the input decoder), paste (bracketed paste), mouse (x,y,action: " +
			"click|right-click|middle-click|move|drag|release|wheel-up|wheel-down), " +
			"send (raw bytes to the terminal, \\e/\\xNN escapes), resize (cols,rows " +
			"delivered via SIGWINCH), raw (captured terminal byte stream: " +
			"escape-sequence stats + escaped tail via peek, clear resets), stop, list. " +
			"Sessions persist across calls; input injected via keys/type/mouse routes " +
			"through the app's real input path.",
		parameters: omp.zod.object({
			op: omp.zod
				.string()
				.describe(
					"operation: start | stop | text | tree | values | keys | event | mouse | resize | list | logs",
				),
			name: omp.zod.string().optional().describe("session name (default: main)"),
			example: omp.zod.string().optional().describe("start: cargo example name"),
			bin: omp.zod.string().optional().describe("start: cargo bin name"),
			args: omp.zod.array(omp.zod.string()).optional().describe("start: program argv"),
			rows: omp.zod.number().optional().describe("start/resize: pty rows (default 30)"),
			cols: omp.zod.number().optional().describe("start/resize: pty cols (default 100)"),
			build: omp.zod
				.boolean()
				.optional()
				.describe("start: cargo build first (default true)"),
			keys: omp.zod
				.string()
				.optional()
				.describe("keys: spec, e.g. \"tab tab enter C-c pgdn 'hello'\""),
			text: omp.zod.string().optional().describe("type/paste/send: payload text"),
			x: omp.zod.number().optional().describe("mouse: zero-based column"),
			y: omp.zod.number().optional().describe("mouse: zero-based viewport row"),
			action: omp.zod.string().optional().describe("mouse: gesture (default click)"),
			peek: omp.zod
				.number()
				.optional()
				.describe("raw: tail bytes to show (default 2000)"),
			clear: omp.zod.boolean().optional().describe("raw: reset capture after reading"),
			timeout: omp.zod
				.number()
				.optional()
				.describe("start: socket wait seconds (default 15)"),
		}),

		async execute(
			_toolCallId: string,
			params: TuiParams,
			onUpdate?: (update: ToolUpdate) => void,
		): Promise<ToolResult> {
			const reply = (text: string, details?: Record<string, unknown>): ToolResult => ({
				content: [{ type: "text", text }],
				details,
			});

			switch (params.op) {
				case "start":
					return reply(await startSession(params, onUpdate));
				case "list": {
					const rows = [...sessions.values()].map(
						(session) =>
							`${session.name}: ${session.target} pid=${session.proc.pid} ` +
							`${session.cols}x${session.rows} ` +
							`${session.exit === null ? "running" : `exited(${session.exit})`}` +
							`${session.sock ? "" : " (no socket)"}`,
					);
					return reply(rows.join("\n") || "no sessions");
				}
				case "stop": {
					const session = need(params.name);
					const code = await stopSession(session);
					return reply(`stopped "${session.name}" (exit ${code})`);
				}
				case "text": {
					const response = await request(need(params.name), { op: "text" });
					return reply(screenshotText(response), response);
				}
				case "frame": {
					const response = await request(need(params.name), { op: "frame" });
					const lines = stringLines(response.lines);
					return reply(lines.map((line) => `│${line}`).join("\n"), response);
				}
				case "tree": {
					const response = await request(need(params.name), { op: "tree" });
					const out: string[] = [];
					renderTree(field(response.tree, "root"), 0, out);
					const overlays = field(response.tree, "overlays");
					if (Array.isArray(overlays)) {
						for (const layer of overlays) {
							out.push(
								`overlay #${field(layer, "overlay")} ` +
									`band=${JSON.stringify(field(layer, "band"))}` +
									`${field(layer, "hidden") === true ? " hidden" : ""}`,
							);
							renderTree(field(layer, "root"), 1, out);
						}
					}
					return reply(out.join("\n"), response);
				}
				case "values":
				case "info": {
					const response = await request(need(params.name), { op: params.op });
					return reply(JSON.stringify(response, null, 1), response);
				}
				case "keys": {
					if (!params.keys) throw new Error("keys op needs `keys`");
					const response = await request(need(params.name), {
						op: "keys",
						keys: params.keys,
					});
					if (!response.ok) throw new Error(String(response.error));
					return reply(`injected ${response.injected} events`, response);
				}
				case "type": {
					if (params.text === undefined) throw new Error("type op needs `text`");
					const response = await request(need(params.name), {
						op: "bytes",
						data: params.text,
					});
					if (!response.ok) throw new Error(String(response.error));
					return reply(`typed ${params.text.length} chars`, response);
				}
				case "paste": {
					if (params.text === undefined) throw new Error("paste op needs `text`");
					const response = await request(need(params.name), {
						op: "paste",
						text: params.text,
					});
					if (!response.ok) throw new Error(String(response.error));
					return reply("pasted", response);
				}
				case "mouse": {
					if (params.x === undefined || params.y === undefined) {
						throw new Error("mouse op needs `x` and `y`");
					}
					const response = await request(need(params.name), {
						op: "mouse",
						x: params.x,
						y: params.y,
						action: params.action ?? "click",
					});
					if (!response.ok) throw new Error(String(response.error));
					return reply(
						`mouse ${params.action ?? "click"} at ${params.x},${params.y}`,
						response,
					);
				}
				case "send": {
					if (params.text === undefined) throw new Error("send op needs `text`");
					const session = need(params.name);
					const bytes = unescapeBytes(params.text);
					session.proc.terminal.write(bytes);
					return reply(`sent ${bytes.length} bytes to the terminal`);
				}
				case "resize": {
					const session = need(params.name);
					const rows = params.rows ?? session.rows;
					const cols = params.cols ?? session.cols;
					session.proc.terminal.resize(cols, rows);
					session.cols = cols;
					session.rows = rows;
					return reply(
						`resized to ${cols}x${rows}; SIGWINCH delivered ` +
							"(resize settles ≈120ms before the rebuild)",
					);
				}
				case "raw": {
					const session = need(params.name);
					const blob = Buffer.concat(session.raw);
					const stats: Record<string, number> = { bytes: blob.length };
					for (const key in SEQUENCES) {
						const total = count(blob, SEQUENCES[key]);
						if (total > 0) stats[key] = total;
					}
					if (params.clear) {
						session.raw = [];
						session.rawBytes = 0;
					}
					const peek = params.peek ?? 2000;
					const tail = peek > 0 ? visible(blob.subarray(-peek)) : "";
					return reply(
						`${JSON.stringify(stats)}${tail ? `\n── tail ──\n${tail}` : ""}`,
						{ stats },
					);
				}
				default:
					throw new Error(`unknown op ${JSON.stringify(params.op)}`);
			}
		},

		onSession(event: { reason?: string }) {
			if (event.reason === "shutdown") {
				for (const session of sessions.values()) {
					try {
						session.proc.kill("SIGKILL");
						session.proc.terminal.close();
						rmSync(session.dir, { recursive: true, force: true });
					} catch {
						// Best-effort teardown.
					}
				}
				sessions.clear();
			}
		},
	};
};

export default factory;
