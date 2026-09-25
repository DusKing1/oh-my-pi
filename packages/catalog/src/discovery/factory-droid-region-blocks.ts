import { getFactoryDroidRegionBlocklistPath, isEnoent, isRecord, logger } from "@oh-my-pi/pi-utils";

/**
 * Reactive region rejections scoped to the serving Vercel edge PoP. A model
 * denied on one edge must not disappear after the user changes networks. The
 * proxy error's requestId carries the rejected request's first PoP, and
 * discovery obtains the current edge from the feature-flags response.
 *
 * Stored as `{ [edge]: { [modelId]: blockedAtMs } }`. Legacy unscoped
 * entries are ignored: replaying them could hide a model on every network.
 */
type FactoryDroidRegionBlocks = Record<string, Record<string, number>>;

function validEdge(edge: string | undefined): edge is string {
	return edge !== undefined && /^[a-z]{3}\d+$/.test(edge);
}

async function readBlocks(agentDir?: string): Promise<FactoryDroidRegionBlocks> {
	try {
		const raw: unknown = JSON.parse(await Bun.file(getFactoryDroidRegionBlocklistPath(agentDir)).text());
		if (!isRecord(raw)) return {};
		const blocks: FactoryDroidRegionBlocks = {};
		for (const [edge, value] of Object.entries(raw)) {
			if (!validEdge(edge) || !isRecord(value)) continue;
			const models: Record<string, number> = {};
			for (const [modelId, blockedAt] of Object.entries(value)) {
				if (typeof blockedAt === "number" && Number.isFinite(blockedAt)) models[modelId] = blockedAt;
			}
			blocks[edge] = models;
		}
		return blocks;
	} catch (error) {
		if (!isEnoent(error)) {
			logger.debug("factory-droid region blocklist unreadable, ignoring", { error: String(error) });
		}
		return {};
	}
}

/** Model IDs rejected from this serving edge; an unknown edge excludes none. */
export async function readFactoryDroidRegionBlockedIds(
	edge: string | undefined,
	agentDir?: string,
): Promise<readonly string[]> {
	if (!validEdge(edge)) return [];
	return Object.keys((await readBlocks(agentDir))[edge] ?? {});
}

/** Best-effort recording; without a known edge, never persist a global ban. */
export async function recordFactoryDroidRegionBlock(modelId: string, edge?: string, agentDir?: string): Promise<void> {
	if (!validEdge(edge)) return;
	try {
		const blocks = await readBlocks(agentDir);
		const models = (blocks[edge] ??= {});
		models[modelId] ??= Date.now();
		await Bun.write(getFactoryDroidRegionBlocklistPath(agentDir), `${JSON.stringify(blocks)}\n`);
	} catch (error) {
		logger.debug("factory-droid region blocklist write failed", { error: String(error) });
	}
}
