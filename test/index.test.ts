// LowRouter tests cover provider registration and live model refresh.
import type { ModelsPublication, Provider, RefreshModelsContext } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { afterEach, describe, expect, it, vi } from "vitest";
import { LOWROUTER_CATALOG } from "../src/catalog.generated.js";
import lowrouterExtension, { LOWROUTER_BASE_URL, LOWROUTER_MODELS_PATH, lowrouterProvider } from "../src/index.js";

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

function refreshContext(overrides: Partial<RefreshModelsContext> = {}): {
  context: RefreshModelsContext;
  publications: ModelsPublication[];
} {
  const publications: ModelsPublication[] = [];
  return {
    publications,
    context: {
      allowNetwork: true,
      signal: new AbortController().signal,
      publish: async (publication) => {
        publications.push(publication);
        publication.update?.();
        return true;
      },
      ...overrides,
    },
  };
}

describe("lowrouterProvider", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("starts from the snapshot", () => {
    const provider = lowrouterProvider();
    expect(provider).toMatchObject({ id: "lowrouter", name: "LowRouter", baseUrl: LOWROUTER_BASE_URL });
    expect(provider.getModels().map((model) => model.id)).toEqual(LOWROUTER_CATALOG.map((model) => model.id));
    expect(provider.getModels().every((model) => model.api === "openai-completions")).toBe(true);
  });

  it("reads the key from LOWROUTER_API_KEY", async () => {
    const result = await lowrouterProvider().auth.apiKey?.resolve({
      ctx: { env: async (name: string) => (name === "LOWROUTER_API_KEY" ? "sk-lr-env" : undefined) } as never,
      signal: new AbortController().signal,
    });
    expect(result?.auth.apiKey).toBe("sk-lr-env");
  });

  it("adds the live models over the snapshot and persists them", async () => {
    const fetchMock = vi.fn(async () => Response.json({ data: liveRows }));
    vi.stubGlobal("fetch", fetchMock);
    const provider = lowrouterProvider();
    const { context, publications } = refreshContext({ credential: { type: "api_key", key: "sk-lr-test" } });

    await provider.refreshModels?.(context);

    expect(fetchMock).toHaveBeenCalledWith(
      `${LOWROUTER_BASE_URL}/${LOWROUTER_MODELS_PATH}`,
      expect.objectContaining({ headers: expect.objectContaining({ authorization: "Bearer sk-lr-test" }) }),
    );
    const ids = provider.getModels().map((model) => model.id);
    expect(ids).toContain("auto/fixture/live-only");
    expect(ids).toContain("alias/team-coder");
    expect(ids).not.toContain("auto/fixture/embedder");
    expect(provider.getModels().find((model) => model.id === "auto/fixture/live-only")).toMatchObject({
      name: "Live Only",
      provider: "lowrouter",
      contextWindow: 200_000,
      maxTokens: 64_000,
      cost: { input: 1 },
    });
    expect(publications.at(-1)?.persist?.models.map((model) => model.id)).toEqual([
      "auto/fixture/live-only",
      "alias/team-coder",
    ]);
  });

  it("serves refreshed models from the configured base URL", async () => {
    const fetchMock = vi.fn(async () => Response.json({ data: liveRows }));
    vi.stubGlobal("fetch", fetchMock);
    const provider = lowrouterProvider();
    const { context } = refreshContext({
      credential: { type: "api_key", key: "sk-lr-test" },
      baseUrl: "https://proxy.example/v1",
    });

    await provider.refreshModels?.(context);

    expect(fetchMock).toHaveBeenCalledWith(`https://proxy.example/v1/${LOWROUTER_MODELS_PATH}`, expect.anything());
    expect(provider.getModels().find((model) => model.id === "alias/team-coder")?.baseUrl).toBe(
      "https://proxy.example/v1",
    );
  });

  it("does not call the API without a key", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const provider = lowrouterProvider();

    await provider.refreshModels?.(refreshContext().context);

    expect(fetchMock).not.toHaveBeenCalled();
    expect(provider.getModels()).toHaveLength(LOWROUTER_CATALOG.length);
  });

  it("does not call the API offline", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const provider = lowrouterProvider();

    await provider.refreshModels?.(
      refreshContext({ allowNetwork: false, credential: { type: "api_key", key: "sk-lr-test" } }).context,
    );

    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("keeps the previous list when the refresh fails", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("down", { status: 503 })));
    const provider = lowrouterProvider();
    const { context } = refreshContext({ credential: { type: "api_key", key: "sk-lr-test" } });

    await expect(provider.refreshModels?.(context)).rejects.toThrow("failed: 503");
    expect(provider.getModels()).toHaveLength(LOWROUTER_CATALOG.length);
  });
});

describe("extension", () => {
  it("registers the LowRouter provider", () => {
    const registered: Provider[] = [];
    lowrouterExtension({ registerProvider: (provider: Provider) => registered.push(provider) } as unknown as ExtensionAPI);
    expect(registered.map((provider) => provider.id)).toEqual(["lowrouter"]);
  });
});
