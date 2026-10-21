// LowRouter tests cover provider registration and live model refresh, run through
// Pi's own model runtime (key lookup, cache restore and persistence included).
import { createModels, InMemoryModelsStore, type ModelsStore, type Provider } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { afterEach, describe, expect, it, vi } from "vitest";
import { LOWROUTER_CATALOG } from "../src/catalog.generated.js";
import lowrouterExtension, { lowrouterProvider } from "../src/index.js";
import { LOWROUTER_BASE_URL, LOWROUTER_MODELS_PATH } from "../src/models.js";

const liveRows = [
  {
    id: "auto/fixture/live-only",
    name: "Live Only",
    modality: "chat",
    context_length: 200_000,
    max_output_tokens: 64_000,
    capabilities: { function_calling: true, input_modalities: ["text"] },
    auto_routing: { pricing_range: { currency: "USD", prompt_per_1m_tokens: { max: 1 } } },
  },
  { id: "alias/team-coder", modality: "chat", capabilities: { function_calling: true } },
  { id: "auto/fixture/embedder", modality: "embedding", capabilities: { function_calling: false } },
];

const snapshotIds = LOWROUTER_CATALOG.map((model) => model.id);

/** Pi's model runtime with the LowRouter provider, `LOWROUTER_API_KEY` set unless `key` is null. */
function runtime(options: { key?: string | null; modelsStore?: ModelsStore } = {}) {
  const key = options.key === undefined ? "sk-lr-test" : options.key;
  const models = createModels({
    modelsStore: options.modelsStore,
    authContext: {
      env: async (name) => (name === "LOWROUTER_API_KEY" && key !== null ? key : undefined),
      fileExists: async () => false,
    },
  });
  models.setProvider(lowrouterProvider());
  return models;
}

function stubModelsResponse(response: () => Response) {
  const fetchMock = vi.fn(async () => response());
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

const ids = (models: ReturnType<typeof runtime>) => models.getModels("lowrouter").map((model) => model.id);

describe("lowrouterProvider", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("starts from the snapshot", () => {
    const models = runtime();
    expect(ids(models)).toEqual(snapshotIds);
    expect(models.getModels("lowrouter").every((model) => model.api === "openai-completions")).toBe(true);
  });

  it("replaces the snapshot with the live list", async () => {
    const fetchMock = stubModelsResponse(() => Response.json({ data: liveRows }));
    const models = runtime();

    const result = await models.refresh({ providers: ["lowrouter"] });

    expect(result.errors.size).toBe(0);
    expect(fetchMock).toHaveBeenCalledWith(
      `${LOWROUTER_BASE_URL}/${LOWROUTER_MODELS_PATH}`,
      expect.objectContaining({ headers: expect.objectContaining({ authorization: "Bearer sk-lr-test" }) }),
    );
    expect(ids(models)).toEqual(["auto/fixture/live-only", "alias/team-coder"]);
    expect(models.getModel("lowrouter", "auto/fixture/live-only")).toMatchObject({
      name: "Live Only",
      provider: "lowrouter",
      contextWindow: 200_000,
      maxTokens: 64_000,
      cost: { input: 1 },
    });
  });

  it("drops the snapshot when the live list is empty", async () => {
    stubModelsResponse(() => Response.json({ data: [] }));
    const models = runtime();

    const result = await models.refresh({ providers: ["lowrouter"] });

    expect(result.errors.size).toBe(0);
    expect(ids(models)).toEqual([]);
  });

  it("restores the last live list offline", async () => {
    stubModelsResponse(() => Response.json({ data: liveRows }));
    const modelsStore = new InMemoryModelsStore();
    await runtime({ modelsStore }).refresh({ providers: ["lowrouter"] });
    const fetchMock = stubModelsResponse(() => new Response("unreachable", { status: 500 }));

    const offline = runtime({ modelsStore });
    await offline.refresh({ providers: ["lowrouter"], allowNetwork: false });

    expect(fetchMock).not.toHaveBeenCalled();
    expect(ids(offline)).toEqual(["auto/fixture/live-only", "alias/team-coder"]);
  });

  it.each([
    ["without a key", { key: null }, {}],
    ["offline", {}, { allowNetwork: false }],
  ] as const)("keeps the snapshot and makes no request %s", async (_label, runtimeOptions, refreshOptions) => {
    const fetchMock = stubModelsResponse(() => Response.json({ data: liveRows }));
    const models = runtime(runtimeOptions);

    await models.refresh({ providers: ["lowrouter"], ...refreshOptions });

    expect(fetchMock).not.toHaveBeenCalled();
    expect(ids(models)).toEqual(snapshotIds);
  });

  it("keeps the previous list when the refresh fails", async () => {
    stubModelsResponse(() => new Response("down", { status: 503 }));
    const models = runtime();

    const result = await models.refresh({ providers: ["lowrouter"] });

    expect(result.errors.get("lowrouter")?.message).toContain("failed: 503");
    expect(ids(models)).toEqual(snapshotIds);
  });

  it("serves refreshed models from the configured base URL", async () => {
    const fetchMock = stubModelsResponse(() => Response.json({ data: liveRows }));
    const provider = lowrouterProvider();

    await provider.refreshModels?.({
      credential: { type: "api_key", key: "sk-lr-test" },
      baseUrl: "https://proxy.example/v1/",
      allowNetwork: true,
      signal: new AbortController().signal,
      publish: async (publication) => {
        publication.update?.();
        return true;
      },
    });

    expect(fetchMock).toHaveBeenCalledWith(`https://proxy.example/v1/${LOWROUTER_MODELS_PATH}`, expect.anything());
    expect(provider.getModels().find((model) => model.id === "alias/team-coder")?.baseUrl).toBe(
      "https://proxy.example/v1",
    );
  });
});

describe("extension", () => {
  it("registers the LowRouter provider", () => {
    const registered: Provider[] = [];
    lowrouterExtension({ registerProvider: (provider: Provider) => registered.push(provider) } as unknown as ExtensionAPI);
    expect(registered.map((provider) => provider.id)).toEqual(["lowrouter"]);
  });
});
