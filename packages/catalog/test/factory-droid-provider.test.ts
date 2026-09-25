import { describe, expect, it } from "bun:test";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { getFactoryDroidRegionBlocklistPath } from "@oh-my-pi/pi-utils";
import { buildModel } from "../src/build";
import { supportsOutputTokenLimit } from "../src/compat/output-limits";
import { buildFactoryDroidModel, fetchFactoryDroidModels } from "../src/discovery/factory-droid";
import { resolveProviderModels } from "../src/model-manager";
import { activeFactoryDroidPromotion } from "../src/types";
import {
	FACTORY_DROID_ANTHROPIC_BASE_URL,
	FACTORY_DROID_COMPLETIONS_BASE_URL,
	FACTORY_DROID_GOOGLE_BASE_URL,
	FACTORY_DROID_MODEL_META,
	FACTORY_DROID_MODELS,
	FACTORY_DROID_RESPONSES_BASE_URL,
	FACTORY_DROID_WIRE_BASE_URLS,
	factoryDroidEdgeRegion,
	factoryDroidPoolForModel,
	factoryDroidServingEdge,
	factoryDroidWireBaseUrl,
	resolveFactoryDroidRotation,
} from "../src/discovery/factory-droid-models";
import {
	readFactoryDroidRegionBlockedIds,
	recordFactoryDroidRegionBlock,
} from "../src/discovery/factory-droid-region-blocks";
import { ANTHROPIC_THINKING, Effort } from "../src/effort";
import { getBundledModel } from "../src/models";
import { resolveModelCacheProviderId } from "../src/provider-models/cache-provider-id";
import { factoryDroidModelManagerOptions } from "../src/provider-models/special";
import type { FetchImpl } from "../src/types";

const zeroCost = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };

function factoryToken(org: string, user: string, exp: number, jti: string): string {
	const payload = Buffer.from(JSON.stringify({ external_org_id: org, sub: user, exp, jti })).toString("base64url");
	return `header.${payload}.signature`;
}

describe("Factory Droid catalog", () => {
	it("strips the off rung while preserving a selectable default thinking effort", () => {
		const model = buildFactoryDroidModel({
			id: "kimi-k3",
			name: "Kimi K3 (Droid Core)",
			wire: "openai-completions",
			contextWindow: 196_608,
			maxTokens: 65_536,
			apiProviders: ["fireworks", "baseten"],
			supportedReasoningEfforts: ["off", Effort.Low, Effort.High, Effort.Max],
			defaultReasoningEffort: Effort.High,
		});
		expect(model.thinking).toEqual({
			mode: "effort",
			efforts: [Effort.Low, Effort.High, Effort.Max],
			requiresEffort: false,
			defaultLevel: Effort.High,
		});
	});

	it("marks text-only models and forces effort when off is unsupported", () => {
		const model = buildFactoryDroidModel({
			id: "text-model",
			name: "Text model",
			wire: "openai-completions",
			contextWindow: 100_000,
			maxTokens: 10_000,
			apiProviders: ["baseten"],
			noImageSupport: true,
			supportedReasoningEfforts: [Effort.High],
			defaultReasoningEffort: Effort.High,
		});

		expect(model.input).toEqual(["text"]);
		expect(model.thinking).toEqual({
			mode: "effort",
			efforts: [Effort.High],
			requiresEffort: true,
			defaultLevel: Effort.High,
		});
	});

	it("omits thinking config for models without a controllable ladder", () => {
		const model = buildFactoryDroidModel({
			id: "glm-4.6",
			name: "GLM-4.6 (Droid Core)",
			wire: "openai-completions",
			contextWindow: 200_000,
			maxTokens: 128_000,
			apiProviders: ["baseten"],
			supportedReasoningEfforts: ["none"],
			defaultReasoningEffort: "none",
			noImageSupport: true,
		});

		expect(model.reasoning).toBe(false);
		expect(model.thinking).toBeUndefined();
	});

	it("does not advertise the retired Inkling model in the offline picker", () => {
		const models = factoryDroidModelManagerOptions().staticModels ?? [];
		expect(models.some(model => model.id === "inkling")).toBe(false);
	});

	it("keeps the offline fallback on the account's EU host and hides unavailable models", () => {
		const manager = factoryDroidModelManagerOptions({ region: "eu" });
		const models = manager.staticModels ?? [];
		expect(models.some(model => model.id === "kimi-k3")).toBe(false);
		const opus = models.find(model => model.id === "claude-opus-5");
		expect(opus?.baseUrl).toBe("https://api.eu.factory.ai/api/llm/a");
		expect(opus?.factoryDroidApiProviders).toEqual(["bedrock_anthropic"]);
	});

	it("filters the static registry by feature flags and org model policy", async () => {
		const flags = Object.fromEntries(
			FACTORY_DROID_MODELS.flatMap(m => (m.featureFlag ? [[m.featureFlag, true]] : [])),
		);
		flags.kimi_k3 = false; // account gate off -> kimi-k3 hidden
		const fetchImpl: FetchImpl = async url => {
			if (String(url).includes("feature-flags")) {
				return new Response(JSON.stringify({ flags }), { status: 200 });
			}
			return new Response(
				JSON.stringify({
					settings: { modelPolicy: { allowAllFactoryModels: false, allowedModelIds: ["kimi-k2.6"] } },
				}),
				{ status: 200 },
			);
		};
		const models = await fetchFactoryDroidModels({ apiKey: "token", fetch: fetchImpl });
		expect(models?.map(model => model.id)).toEqual(["kimi-k2.6"]);
	});

	it("does not expose org-denied models when the allowlist is absent", async () => {
		const flags = Object.fromEntries(
			FACTORY_DROID_MODELS.flatMap(m => (m.featureFlag ? [[m.featureFlag, true]] : [])),
		);
		const fetchImpl: FetchImpl = async url =>
			new Response(
				JSON.stringify(
					String(url).includes("feature-flags")
						? { flags }
						: {
								settings: { modelPolicy: { allowAllFactoryModels: false } },
							},
				),
				{ status: 200 },
			);
		const models = await fetchFactoryDroidModels({ apiKey: "token", fetch: fetchImpl });
		expect(models?.map(model => model.id)).toEqual([]);
	});

	it("rejects unsupported GPT and Gemini output caps while preserving Grok caps", () => {
		for (const id of ["gpt-5.4", "gemini-3.7-flash", "garnet-07-15"]) {
			const model = buildModel(buildFactoryDroidModel(FACTORY_DROID_MODEL_META[id]));
			expect(model.omitMaxOutputTokens, id).toBe(true);
			expect(supportsOutputTokenLimit(model), id).toBe(false);
		}
		const grok = buildModel(buildFactoryDroidModel(FACTORY_DROID_MODEL_META["grok-4.5"]));
		expect(grok.omitMaxOutputTokens).toBeUndefined();
		expect(supportsOutputTokenLimit(grok)).toBe(true);
	});

	it("uses the KDL output clamp for Droid Core Completions without applying it to Responses", () => {
		for (const id of ["glm-5.3", "kimi-k3", "deepseek-v4-flash-0731", "nemotron-3-ultra"]) {
			const model = buildModel<"openai-completions">({
				...buildFactoryDroidModel(FACTORY_DROID_MODEL_META[id]),
				api: "openai-completions",
				remoteCompaction: undefined,
			});
			expect(model.compat.clampOutputToModelMax, id).toBe(true);
		}
		const gpt = buildModel<"openai-responses">({
			...buildFactoryDroidModel(FACTORY_DROID_MODEL_META["gpt-5.4"]),
			api: "openai-responses",
			remoteCompaction: undefined,
		});
		const grok = buildModel<"openai-responses">({
			...buildFactoryDroidModel(FACTORY_DROID_MODEL_META["grok-4.5"]),
			api: "openai-responses",
			remoteCompaction: undefined,
		});
		expect(gpt.compat.clampOutputToModelMax).not.toBe(true);
		expect(grok.compat.clampOutputToModelMax).not.toBe(true);
	});

	it("distinguishes billing pool from Anthropic wire for MiniMax", () => {
		expect(factoryDroidPoolForModel("minimax-m3")).toBe("core");
		expect(factoryDroidPoolForModel("claude-opus-5")).toBe("standard");
		expect(factoryDroidPoolForModel("kimi-k3")).toBe("core");
		expect(factoryDroidPoolForModel("not-in-registry")).toBeUndefined();
	});

	it("keeps refreshed WorkOS tokens in one account cache without crossing org, user or residency", () => {
		const original = factoryToken("org-A", "user-A", 1_000, "first");
		const refreshed = factoryToken("org-A", "user-A", 2_000, "second");
		const account = resolveModelCacheProviderId("factory-droid", { apiKey: original, region: "global" });
		expect(account).toBe(resolveModelCacheProviderId("factory-droid", { apiKey: refreshed, region: "global" }));
		expect(account).not.toBe(
			resolveModelCacheProviderId("factory-droid", { apiKey: factoryToken("org-B", "user-A", 1_000, "first") }),
		);
		expect(account).not.toBe(
			resolveModelCacheProviderId("factory-droid", { apiKey: factoryToken("org-A", "user-B", 1_000, "first") }),
		);
		expect(account).not.toBe(resolveModelCacheProviderId("factory-droid", { apiKey: refreshed, region: "eu" }));
		expect(account).not.toContain(original);
		expect(account).not.toContain("org-A");
		expect(factoryDroidModelManagerOptions({ apiKey: refreshed }).cacheProviderId).toBe(account);
		const opaque = resolveModelCacheProviderId("factory-droid", { apiKey: "secret-A" });
		expect(opaque).not.toBe(resolveModelCacheProviderId("factory-droid", { apiKey: "secret-B" }));
		expect(opaque).not.toContain("secret-A");
	});

	it("restores the account's authoritative roster after WorkOS refresh", async () => {
		const dir = await fs.mkdtemp(path.join(os.tmpdir(), "factory-account-cache-"));
		const cacheDbPath = path.join(dir, "models.db");
		const original = factoryToken("org-A", "user-A", 1_000, "first");
		const refreshed = factoryToken("org-A", "user-A", 2_000, "second");
		try {
			await resolveProviderModels(
				{
					...factoryDroidModelManagerOptions({ apiKey: original }),
					cacheDbPath,
					fetchDynamicModels: async () => [buildFactoryDroidModel(FACTORY_DROID_MODEL_META["glm-5.3"])],
				},
				"online",
			);
			const restored = await resolveProviderModels(
				{ ...factoryDroidModelManagerOptions({ apiKey: refreshed }), cacheDbPath },
				"offline",
			);
			expect(restored.source).toBe("cache");
			expect(restored.models.map(model => model.id)).toEqual(["glm-5.3"]);
		} finally {
			await fs.rm(dir, { recursive: true, force: true });
		}
	});

	it("withdraws fast tiers only when the org explicitly disallows them", async () => {
		const flags = Object.fromEntries(
			FACTORY_DROID_MODELS.flatMap(m => (m.featureFlag ? [[m.featureFlag, true]] : [])),
		);
		const discover = async (modelPolicy: Record<string, unknown>): Promise<string[]> => {
			const fetchImpl: FetchImpl = async url =>
				new Response(
					JSON.stringify(String(url).includes("feature-flags") ? { flags } : { settings: { modelPolicy } }),
					{ status: 200 },
				);
			const models = await fetchFactoryDroidModels({ apiKey: "token", fetch: fetchImpl });
			return models?.map(model => model.id) ?? [];
		};
		// `baseVariant` is what marks an entry as a fast tier; the gate is a
		// class switch, not a per-id policy.
		const fastIds = FACTORY_DROID_MODELS.filter(m => m.baseVariant !== undefined).map(m => m.id);
		expect(fastIds.length).toBeGreaterThan(0);

		// Allow-all is the CLI's default kind, and older servers omit the
		// field entirely: neither may withdraw a fast tier.
		const allowed = await discover({ allowAllFactoryModels: true, isFastModelsAllowed: true });
		const silent = await discover({ allowAllFactoryModels: true });
		for (const id of fastIds) {
			expect(allowed).toContain(id);
			expect(silent).toContain(id);
		}

		// Only an explicit false hides them, and it hides nothing else.
		const denied = await discover({ allowAllFactoryModels: true, isFastModelsAllowed: false });
		for (const id of fastIds) expect(denied).not.toContain(id);
		expect(denied).toContain("gpt-5.4");
		expect(denied.length).toBe(allowed.length - fastIds.length);
	});

	it("applies the live provider_routing config to the model spec", async () => {
		const fetchImpl: FetchImpl = async url => {
			if (String(url).includes("feature-flags")) {
				return new Response(
					JSON.stringify({
						flags: { kimi_k3: true },
						configs: { provider_routing: { version: 1, models: { "kimi-k3": ["baseten", "fireworks"] } } },
					}),
					{ status: 200 },
				);
			}
			return new Response(JSON.stringify({ settings: {} }), { status: 200 });
		};
		const models = await fetchFactoryDroidModels({ apiKey: "token", fetch: fetchImpl });
		const kimi = models?.find(model => model.id === "kimi-k3");
		expect(kimi?.factoryDroidApiProviders).toEqual(["baseten", "fireworks"]);
		// Models without a routing entry keep the registry's static order.
		const glm = models?.find(model => model.id === "glm-5.2");
		expect(glm?.factoryDroidApiProviders).toBeUndefined();
	});

	it("falls back to null without credentials so the static list stays", async () => {
		const fetchImpl: FetchImpl = async () => {
			throw new Error("network down");
		};
		expect(await fetchFactoryDroidModels({ apiKey: "token", fetch: fetchImpl })).toBeNull();
	});

	it("distinguishes an off/none control rung from mandatory reasoning", () => {
		for (const id of ["kimi-k3", "gpt-5.6-sol"]) {
			const thinking = buildFactoryDroidModel(FACTORY_DROID_MODEL_META[id]).thinking;
			expect(thinking?.requiresEffort).toBe(false);
			expect(thinking?.efforts).not.toContain("off");
			expect(thinking?.efforts).not.toContain("none");
		}
		expect(buildFactoryDroidModel(FACTORY_DROID_MODEL_META["gpt-5.1-codex-max"]).thinking?.requiresEffort).toBe(true);
	});

	it("bakes the shared anthropic budget ladder for budget-style thinking", () => {
		for (const id of ["claude-sonnet-4-5-20250929", "claude-opus-4-5-20251101", "claude-haiku-4-5-20251001"]) {
			const thinking = buildFactoryDroidModel(FACTORY_DROID_MODEL_META[id]).thinking;
			expect(thinking?.efforts).toEqual([Effort.Low, Effort.Medium, Effort.High]);
			expect(thinking?.effortBudgets).toEqual(ANTHROPIC_THINKING);
			expect(thinking?.requiresEffort).toBe(false);
			expect(thinking).not.toHaveProperty("defaultLevel");
		}
		// 'none'-only ladders are not reasoning models: no thinking config is built.
		for (const id of ["glm-4.7", "glm-5"]) {
			expect(buildFactoryDroidModel(FACTORY_DROID_MODEL_META[id]).reasoning).toBe(false);
			expect(buildFactoryDroidModel(FACTORY_DROID_MODEL_META[id]).thinking).toBeUndefined();
		}
	});

	it("maps each wire family to its base URL", () => {
		const base = {
			contextWindow: 100_000,
			maxTokens: 10_000,
			apiProviders: ["fireworks"] as const,
			supportedReasoningEfforts: [Effort.Low, Effort.High],
		};
		const cases: Array<
			[string, "openai-completions" | "openai-responses" | "anthropic-messages" | "google-generate", string]
		> = [
			["completions", "openai-completions", FACTORY_DROID_COMPLETIONS_BASE_URL],
			["responses", "openai-responses", FACTORY_DROID_RESPONSES_BASE_URL],
			["anthropic", "anthropic-messages", FACTORY_DROID_ANTHROPIC_BASE_URL],
			["google", "google-generate", FACTORY_DROID_GOOGLE_BASE_URL],
		];
		for (const [label, wire, expected] of cases) {
			const model = buildFactoryDroidModel({ id: `m-${label}`, name: label, wire, ...base });
			expect(model.baseUrl).toBe(expected);
		}
		// The WebSocket Responses transport is a resolved wire the provider
		// dispatches on, never a registry entry's, and shares the Responses
		// namespace on both hosts.
		expect(FACTORY_DROID_WIRE_BASE_URLS["openai-responses-ws"]).toBe(FACTORY_DROID_RESPONSES_BASE_URL);
		expect(factoryDroidWireBaseUrl("openai-responses-ws", "eu")).toBe("https://api.eu.factory.ai/api/llm/o/v1");
		expect(FACTORY_DROID_MODELS.some(model => model.wire === "openai-responses-ws")).toBe(false);
	});

	it("enables summarized thinking display for adaptive Anthropic models", () => {
		const thinking = buildFactoryDroidModel(FACTORY_DROID_MODEL_META["claude-opus-4-7-fast"]).thinking;
		expect(thinking).toMatchObject({ mode: "anthropic-adaptive", supportsDisplay: true });
	});

	it("wires upstream list prices and effective credit rates from the registry", () => {
		// Cost is inherited from the referenced bundled catalog entry, not inlined.
		const kimi = buildFactoryDroidModel(FACTORY_DROID_MODEL_META["kimi-k2.7-code"]);
		expect(kimi.cost).toEqual(getBundledModel("fireworks", "kimi-k2.7-code").cost);
		expect(kimi.factoryDroidCredits).toEqual({ input: 0.38, output: 1.5998 });

		// Cache-read-metered models project the relative multiplier through the input rate.
		const grok = buildFactoryDroidModel(FACTORY_DROID_MODEL_META["grok-4.5"]);
		expect(grok.cost).toEqual(getBundledModel("xai", "grok-4.5").cost);
		expect(grok.factoryDroidCredits).toEqual({
			input: 0.8,
			output: 2.4,
			cacheRead: 0.12,
			promotions: [{ discount: 0.6, expiresAt: "2026-09-05T02:00:00Z", label: ", 60% Off" }],
		});

		// No outputTokenMultiplier -> output billed at the input rate.
		const opus = buildFactoryDroidModel(FACTORY_DROID_MODEL_META["claude-opus-5"]);
		expect(opus.factoryDroidCredits).toEqual({ input: 2, output: 2 });
		expect(opus.cost).toEqual(getBundledModel("anthropic", "claude-opus-5").cost);

		// 0.203.0 repriced GLM-5.2 to 0.56 in / 3.15x out.
		const glm = buildFactoryDroidModel(FACTORY_DROID_MODEL_META["glm-5.2"]);
		expect(glm.factoryDroidCredits).toEqual({ input: 0.56, output: 1.764 });
	});

	it("keeps Factory-only SKUs at zero cost with the credit badge only", () => {
		// Fast tiers are distinct SKUs with no upstream list price.
		const fast = buildFactoryDroidModel(FACTORY_DROID_MODEL_META["claude-opus-4-8-fast"]);
		expect(fast.cost).toEqual(zeroCost);
		expect(fast.factoryDroidCredits).toEqual({ input: 4, output: 4 });

		// Preview codenames have no upstream catalog entry either.
		const atlas = buildFactoryDroidModel(FACTORY_DROID_MODEL_META["atlas-07-21"]);
		expect(atlas.cost).toEqual(zeroCost);
		expect(atlas.factoryDroidCredits).toEqual({ input: 2, output: 2 });
	});

	it("applies the first active promo window, then expires it", () => {
		const sol = buildFactoryDroidModel(FACTORY_DROID_MODEL_META["gpt-5.6-sol"]).factoryDroidCredits!;
		// The CLI applies the first currently-active window: 60% off until
		// 2026-09-05, then the 20% fallback through 2026-11-22, then list.
		expect(activeFactoryDroidPromotion(sol, new Date("2026-09-04T00:00:00Z"))?.discount).toBe(0.6);
		expect(activeFactoryDroidPromotion(sol, new Date("2026-09-06T00:00:00Z"))?.discount).toBe(0.2);
		expect(activeFactoryDroidPromotion(sol, new Date("2026-11-23T00:00:00Z"))).toBeUndefined();
		// Kimi K3's old promo expired and no longer grants an active discount.
		expect(
			activeFactoryDroidPromotion(buildFactoryDroidModel(FACTORY_DROID_MODEL_META["kimi-k3"]).factoryDroidCredits!),
		).toBeUndefined();
	});

	it("ships gemini-3.7-flash on the google ladder with no upstream list price yet", () => {
		// This Factory-only catalog entry has no upstream list price.
		const meta = FACTORY_DROID_MODEL_META["gemini-3.7-flash"];

		const model = buildFactoryDroidModel(meta);
		expect(model.baseUrl).toBe(FACTORY_DROID_GOOGLE_BASE_URL);
		expect(model.cost).toEqual(zeroCost);
		expect(model.thinking).toMatchObject({
			mode: "google-level",
			efforts: [Effort.Low, Effort.Medium, Effort.High],
			defaultLevel: Effort.High,
			requiresEffort: true,
		});
		expect(model.factoryDroidCredits).toEqual({
			input: 0.6,
			output: 3,
			promotions: [
				{
					discount: 0.5,
					startsAt: "2026-08-17T21:10:19Z",
					expiresAt: "2027-01-01T00:00:00Z",
					label: ", 50% Off",
				},
			],
		});
		// The predecessor lost its minimal rung, while 3.7 never exposed it.
		expect(buildFactoryDroidModel(FACTORY_DROID_MODEL_META["gemini-3.6-flash"]).thinking?.efforts).toEqual([
			Effort.Low,
			Effort.Medium,
			Effort.High,
		]);
	});

	it("resolves every registry priceRef in the bundled catalog", () => {
		// A models.json regen that drops a referenced id must fail here, not
		// silently degrade the model's cost display to zero.
		let checked = 0;
		for (const meta of FACTORY_DROID_MODELS) {
			if (!meta.priceRef) continue;
			checked += 1;
			const reference = getBundledModel(meta.priceRef.provider, meta.priceRef.modelId) as unknown;
			expect(reference, `${meta.id} -> ${meta.priceRef.provider}/${meta.priceRef.modelId}`).toBeDefined();
		}
		expect(checked).toBeGreaterThan(30);
	});
});

describe("Factory Droid EU region", () => {
	const allFlagsOn = Object.fromEntries(
		FACTORY_DROID_MODELS.flatMap(m => (m.featureFlag ? [[m.featureFlag, true]] : [])),
	);
	const okJson = (body: unknown) => new Response(JSON.stringify(body), { status: 200 });

	it("resolves rotations from the region: override wins, filter is the fallback, global passes through", () => {
		const opus5 = FACTORY_DROID_MODELS.find(m => m.id === "claude-opus-5")!;
		const sonnet = FACTORY_DROID_MODELS.find(m => m.id === "claude-sonnet-4-5-20250929")!;
		const fable = FACTORY_DROID_MODELS.find(m => m.id === "claude-fable-5")!;
		const kimi = FACTORY_DROID_MODELS.find(m => m.id === "kimi-k3")!;

		// Explicit EU override (the CLI's regionOverrides.eu) wins verbatim.
		expect(resolveFactoryDroidRotation(opus5, "eu")).toEqual(["bedrock_anthropic"]);
		// An empty override means unavailable in the region.
		expect(resolveFactoryDroidRotation(fable, "eu")).toEqual([]);
		// No override: the default rotation is filtered to EU-serving upstreams.
		expect(resolveFactoryDroidRotation(sonnet, "eu")).toEqual(["vertex_anthropic", "bedrock_anthropic"]);
		// fireworks/baseten serve only the global region.
		expect(resolveFactoryDroidRotation(kimi, "eu")).toEqual([]);
		// Global and unknown regions keep the static rotation untouched.
		expect(resolveFactoryDroidRotation(opus5, undefined)).toEqual(opus5.apiProviders);
		expect(resolveFactoryDroidRotation(opus5, "global")).toEqual(opus5.apiProviders);
		expect(resolveFactoryDroidRotation(kimi, "global")).toEqual(["fireworks", "baseten"]);
	});

	it("keeps the 0.203.0 azure and mistral additions out of EU rotations", () => {
		const gpt52 = FACTORY_DROID_MODELS.find(m => m.id === "gpt-5.2")!;
		const gpt54 = FACTORY_DROID_MODELS.find(m => m.id === "gpt-5.4")!;
		const glm52 = FACTORY_DROID_MODELS.find(m => m.id === "glm-5.2")!;

		// azure_openai joined the GPT rotations in second position...
		expect(gpt52.apiProviders).toEqual(["openai", "azure_openai"]);
		expect(resolveFactoryDroidRotation(gpt52, "global")).toEqual(["openai", "azure_openai"]);
		// ...but serves the global region only, so EU accounts never route to it.
		expect(resolveFactoryDroidRotation(gpt52, "eu")).toEqual(["openai"]);
		// An explicit EU override still wins verbatim over that filter.
		expect(gpt54.apiProviders).toEqual(["openai", "azure_openai", "bedrock_openai"]);
		expect(resolveFactoryDroidRotation(gpt54, "eu")).toEqual(["openai"]);
		// mistral is global-only too, so GLM-5.2 stays EU-unavailable.
		expect(glm52.apiProviders).toEqual(["fireworks", "baseten", "mistral"]);
		expect(resolveFactoryDroidRotation(glm52, "eu")).toEqual([]);
	});

	it("queries the EU host and hides models with no EU-serving upstream", async () => {
		const urls: string[] = [];
		const fetchImpl: FetchImpl = async url => {
			urls.push(String(url));
			if (String(url).includes("feature-flags")) return okJson({ flags: allFlagsOn });
			return okJson({ settings: {} });
		};
		const models = await fetchFactoryDroidModels({ apiKey: "token", region: "eu", fetch: fetchImpl });
		expect(models).not.toBeNull();
		const ids = models!.map(model => model.id);

		// Discovery endpoints follow the region.
		expect(urls[0]).toBe("https://api.eu.factory.ai/api/feature-flags");
		expect(urls[1]).toBe("https://api.eu.factory.ai/api/organization/managed-settings");

		// Hidden for EU: Droid Core (fireworks/baseten-only), Gemini (google-only),
		// grok (xai-only), and fable-5 (explicit empty EU override).
		expect(ids).not.toContain("kimi-k3");
		expect(ids).not.toContain("gemini-3.1-pro-preview");
		expect(ids).not.toContain("grok-4.5");
		expect(ids).not.toContain("claude-fable-5");
		// Available with region-resolved rotations and EU wire URLs.
		const opus5 = models!.find(model => model.id === "claude-opus-5")!;
		expect(opus5.factoryDroidApiProviders).toEqual(["bedrock_anthropic"]);
		expect(opus5.baseUrl).toBe("https://api.eu.factory.ai/api/llm/a");
		const sonnet = models!.find(model => model.id === "claude-sonnet-4-5-20250929")!;
		expect(sonnet.factoryDroidApiProviders).toEqual(["vertex_anthropic", "bedrock_anthropic"]);
		const gpt54 = models!.find(model => model.id === "gpt-5.4")!;
		expect(gpt54.factoryDroidApiProviders).toEqual(["openai"]);
		expect(gpt54.baseUrl).toBe("https://api.eu.factory.ai/api/llm/o/v1");
	});

	it("intersects live provider_routing with the EU rotation instead of resurrecting global upstreams", async () => {
		const fetchImpl: FetchImpl = async url => {
			if (String(url).includes("feature-flags")) {
				return okJson({
					flags: allFlagsOn,
					configs: {
						provider_routing: {
							version: 1,
							models: {
								// US-centric entry: no EU upstream survives the intersection,
								// so the region-resolved rotation wins.
								"claude-opus-5": ["anthropic"],
								// Mixed entry narrows to the EU-serving subset.
								"claude-sonnet-4-5-20250929": ["anthropic", "bedrock_anthropic"],
							},
						},
					},
				});
			}
			return okJson({ settings: {} });
		};
		const models = await fetchFactoryDroidModels({ apiKey: "token", region: "eu", fetch: fetchImpl });
		expect(models!.find(model => model.id === "claude-opus-5")?.factoryDroidApiProviders).toEqual([
			"bedrock_anthropic",
		]);
		expect(models!.find(model => model.id === "claude-sonnet-4-5-20250929")?.factoryDroidApiProviders).toEqual([
			"bedrock_anthropic",
		]);
	});

	it("keeps the global path byte-identical when no region is known", async () => {
		const urls: string[] = [];
		const fetchImpl: FetchImpl = async url => {
			urls.push(String(url));
			if (String(url).includes("feature-flags")) return okJson({ flags: allFlagsOn });
			return okJson({ settings: {} });
		};
		const models = await fetchFactoryDroidModels({ apiKey: "token", fetch: fetchImpl });
		expect(urls[0]).toBe("https://api.factory.ai/api/feature-flags");
		const opus5 = models!.find(model => model.id === "claude-opus-5")!;
		// No routing entry: the sparse field stays unset and the wire URL is the global host.
		expect(opus5.factoryDroidApiProviders).toBeUndefined();
		expect(opus5.baseUrl).toBe("https://api.factory.ai/api/llm/a");
		expect(models!.find(model => model.id === "kimi-k3")).toBeDefined();
	});
});

describe("Factory Droid serving edge", () => {
	const allFlagsOn = Object.fromEntries(
		FACTORY_DROID_MODELS.flatMap(m => (m.featureFlag ? [[m.featureFlag, true]] : [])),
	);

	it("parses the serving edge PoP from x-vercel-id", () => {
		expect(factoryDroidEdgeRegion(new Headers({ "x-vercel-id": "cdg1::sfo1::jsvsj-123" }))).toBe("eu");
		expect(factoryDroidEdgeRegion(new Headers({ "x-vercel-id": "FRA1::iad1::x" }))).toBe("eu");
		expect(factoryDroidEdgeRegion(new Headers({ "x-vercel-id": "sfo1::sfo1::x" }))).toBeUndefined();
		expect(factoryDroidEdgeRegion(new Headers({ "x-vercel-id": "cpt1::sfo1::x" }))).toBeUndefined();
		expect(factoryDroidEdgeRegion(new Headers())).toBeUndefined();
		expect(factoryDroidEdgeRegion(new Headers({ "x-vercel-id": "" }))).toBeUndefined();
		expect(factoryDroidServingEdge(new Headers({ "x-vercel-id": "cdg1::sfo1::req-123" }))).toBe("cdg1");
		expect(factoryDroidServingEdge(new Headers({ "x-vercel-id": "not-a-pop" }))).toBeUndefined();
	});

	it("hides global-only-upstream models and resolves EU rotations on an EU edge, keeping the global host", async () => {
		const fetchImpl: FetchImpl = async url => {
			if (String(url).includes("feature-flags")) {
				return new Response(JSON.stringify({ flags: allFlagsOn }), {
					status: 200,
					headers: { "x-vercel-id": "cdg1::sfo1::jsvsj-123" },
				});
			}
			return new Response(JSON.stringify({ settings: {} }), { status: 200 });
		};
		const models = await fetchFactoryDroidModels({ apiKey: "token", fetch: fetchImpl });
		const ids = models!.map(model => model.id);
		// No EU-serving upstream: hidden exactly as for an EU-resident account.
		expect(ids).not.toContain("kimi-k3");
		expect(ids).not.toContain("gemini-3.1-pro-preview");
		expect(ids).not.toContain("grok-4.5");
		expect(ids).not.toContain("claude-fable-5");
		// EU rotation override applies, but the host stays global (account has
		// no residency region; only the serving edge is European).
		const opus5 = models!.find(model => model.id === "claude-opus-5");
		expect(opus5?.factoryDroidApiProviders).toEqual(["bedrock_anthropic"]);
		expect(opus5?.baseUrl).toBe(FACTORY_DROID_ANTHROPIC_BASE_URL);
	});

	it("leaves the model list untouched on a US edge", async () => {
		const fetchImpl: FetchImpl = async url => {
			if (String(url).includes("feature-flags")) {
				return new Response(JSON.stringify({ flags: allFlagsOn }), {
					status: 200,
					headers: { "x-vercel-id": "sfo1::sfo1::jsvsj-123" },
				});
			}
			return new Response(JSON.stringify({ settings: {} }), { status: 200 });
		};
		const models = await fetchFactoryDroidModels({ apiKey: "token", fetch: fetchImpl });
		expect(models!.find(model => model.id === "kimi-k3")).toBeDefined();
	});

	it("lets an explicit account region win over a US edge", async () => {
		const fetchImpl: FetchImpl = async url => {
			if (String(url).includes("feature-flags")) {
				return new Response(JSON.stringify({ flags: allFlagsOn }), {
					status: 200,
					headers: { "x-vercel-id": "sfo1::sfo1::jsvsj-123" },
				});
			}
			return new Response(JSON.stringify({ settings: {} }), { status: 200 });
		};
		const models = await fetchFactoryDroidModels({ apiKey: "token", region: "eu", fetch: fetchImpl });
		const ids = models!.map(model => model.id);
		expect(ids).not.toContain("kimi-k3");
		expect(models!.find(model => model.id === "claude-opus-5")?.baseUrl).toContain("api.eu.factory.ai");
	});

	it("hides models excluded by the region blocklist", async () => {
		const fetchImpl: FetchImpl = async url => {
			if (String(url).includes("feature-flags")) {
				return new Response(JSON.stringify({ flags: allFlagsOn }), { status: 200 });
			}
			return new Response(JSON.stringify({ settings: {} }), { status: 200 });
		};
		const models = await fetchFactoryDroidModels({
			apiKey: "token",
			fetch: fetchImpl,
			excludeModelIds: ["claude-opus-5"],
		});
		const ids = models!.map(model => model.id);
		expect(ids).not.toContain("claude-opus-5");
		expect(ids).toContain("kimi-k3");
	});

	it("keeps a region rejection on its own serving edge, including across discovery", async () => {
		const dir = await fs.mkdtemp(path.join(os.tmpdir(), "fd-region-blocks-"));
		try {
			await recordFactoryDroidRegionBlock("claude-opus-5", "cdg1", dir);
			const discover = (edge: string) =>
				fetchFactoryDroidModels({
					apiKey: "token",
					agentDir: dir,
					fetch: async url =>
						new Response(
							JSON.stringify(String(url).includes("feature-flags") ? { flags: allFlagsOn } : { settings: {} }),
							{
								status: 200,
								headers: { "x-vercel-id": `${edge}::iad1::request-123` },
							},
						),
				});
			const denied = await discover("cdg1");
			const allowed = await discover("sfo1");
			expect(denied?.some(model => model.id === "claude-opus-5")).toBe(false);
			expect(allowed?.some(model => model.id === "claude-opus-5")).toBe(true);
		} finally {
			await fs.rm(dir, { recursive: true, force: true });
		}
	});
});

describe("Factory Droid region blocklist", () => {
	it("round-trips edge-specific blocks and ignores legacy or corrupt global exclusions", async () => {
		const dir = await fs.mkdtemp(path.join(os.tmpdir(), "fd-region-blocks-"));
		try {
			expect(await readFactoryDroidRegionBlockedIds("cdg1", dir)).toEqual([]);
			await recordFactoryDroidRegionBlock("deepseek-v4-flash-0731", "cdg1", dir);
			await recordFactoryDroidRegionBlock("kimi-k3", "cdg1", dir);
			await recordFactoryDroidRegionBlock("kimi-k3", "cdg1", dir);
			await recordFactoryDroidRegionBlock("gpt-5.4", undefined, dir);
			expect([...(await readFactoryDroidRegionBlockedIds("cdg1", dir))].sort()).toEqual([
				"deepseek-v4-flash-0731",
				"kimi-k3",
			]);
			expect(await readFactoryDroidRegionBlockedIds("sfo1", dir)).toEqual([]);
			await Bun.write(getFactoryDroidRegionBlocklistPath(dir), '{"kimi-k3":123}');
			expect(await readFactoryDroidRegionBlockedIds("cdg1", dir)).toEqual([]);
			await Bun.write(getFactoryDroidRegionBlocklistPath(dir), "not json");
			expect(await readFactoryDroidRegionBlockedIds("cdg1", dir)).toEqual([]);
		} finally {
			await fs.rm(dir, { recursive: true, force: true });
		}
	});
});
