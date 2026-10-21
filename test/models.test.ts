// LowRouter tests cover the snapshot and the live /v1/models projection.
import { afterEach, describe, expect, it, vi } from "vitest";
import { LOWROUTER_CATALOG } from "../src/catalog.generated.js";
import {
  fetchLowRouterModelRows,
  isPickerModelId,
  LOWROUTER_BASE_URL,
  LOWROUTER_MODELS_PATH,
  type LowRouterModelRow,
  projectLowRouterModelRows,
  toLowRouterModel,
} from "../src/models.js";

/** A `/v1/models?currency=usd` row as LowRouter serves it, trimmed to the fields read. */
function liveRow(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    id: "auto/fixture/never-bundled",
    name: "Fixture Model",
    modality: "chat",
    context_length: 131_072,
    max_output_tokens: 32_768,
    supported_parameters: ["tools", "tool_choice"],
    capabilities: { function_calling: true, input_modalities: ["text", "image"] },
    auto_routing: {
      pricing_range: {
        currency: "USD",
        prompt_per_1m_tokens: { min: 0.4, max: 0.45 },
        completion_per_1m_tokens: { min: 2, max: 2.2 },
        cache_read_per_1m_tokens: { min: 0.04, max: 0.045 },
      },
    },
    ...overrides,
  };
}

function snapshotRow(overrides: Partial<LowRouterModelRow> = {}): LowRouterModelRow {
  return {
    id: "auto/fixture/never-bundled",
    name: "auto/fixture/never-bundled",
    reasoning: true,
    input: ["text"],
    cost: { input: 1, output: 3, cacheRead: 0.1, cacheWrite: 1.25 },
    contextWindow: 64_000,
    maxTokens: 16_000,
    ...overrides,
  };
}

describe("isPickerModelId", () => {
  it("keeps auto/<creator>/<model> and alias/<name>", () => {
    expect(isPickerModelId("auto/mistralai/mistral-large-2512")).toBe(true);
    expect(isPickerModelId("alias/my-coder")).toBe(true);
  });

  it("drops pinned routes, scoped auto ids and malformed ids", () => {
    expect(isPickerModelId("scaleway/mistralai/mistral-large-2512")).toBe(false);
    expect(isPickerModelId("auto/mistralai/mistral-large-2512/eu")).toBe(false);
    expect(isPickerModelId("auto/mistralai")).toBe(false);
    expect(isPickerModelId("auto//mistral-large-2512")).toBe(false);
    expect(isPickerModelId("alias/")).toBe(false);
    expect(isPickerModelId("")).toBe(false);
  });
});

describe("projectLowRouterModelRows", () => {
  it("projects a full row", () => {
    expect(projectLowRouterModelRows([liveRow()])).toEqual([
      {
        id: "auto/fixture/never-bundled",
        name: "Fixture Model",
        reasoning: false,
        input: ["text", "image"],
        cost: { input: 0.45, output: 2.2, cacheRead: 0.045, cacheWrite: 0 },
        contextWindow: 131_072,
        maxTokens: 32_768,
      },
    ]);
  });

  it("keeps only tool-capable chat models", () => {
    const rows = [
      liveRow({ id: "auto/a/embed", modality: "embedding" }),
      liveRow({ id: "auto/a/no-tools", capabilities: { function_calling: false } }),
      liveRow({ id: "auto/a/no-caps", capabilities: undefined }),
      liveRow({ id: "auto/a/ok" }),
    ];
    expect(projectLowRouterModelRows(rows).map((model) => model.id)).toEqual(["auto/a/ok"]);
  });

  it("drops pinned and scoped ids, non-objects and duplicates", () => {
    const rows = [
      liveRow({ id: "scaleway/a/pinned" }),
      liveRow({ id: "auto/a/b/eu" }),
      null,
      "auto/a/string",
      liveRow({ id: 42 }),
      liveRow({ id: "  " }),
      liveRow({ id: "alias/mine" }),
      liveRow({ id: "alias/mine", name: "second copy" }),
    ];
    expect(projectLowRouterModelRows(rows)).toEqual([expect.objectContaining({ id: "alias/mine", name: "Fixture Model" })]);
  });

  it("reads reasoning from supported_parameters", () => {
    const [effort] = projectLowRouterModelRows([liveRow({ supported_parameters: ["tools", "reasoning_effort"] })]);
    const [reasoning] = projectLowRouterModelRows([liveRow({ supported_parameters: ["reasoning"] })]);
    expect(effort?.reasoning).toBe(true);
    expect(reasoning?.reasoning).toBe(true);
  });

  it("prices at the top of the USD range and ignores other currencies", () => {
    const eur = liveRow({
      auto_routing: {
        pricing_range: { currency: "EUR", prompt_per_1m_tokens: { min: 1, max: 2 } },
      },
    });
    expect(projectLowRouterModelRows([eur])[0]?.cost).toEqual({ input: 0, output: 0, cacheRead: 0, cacheWrite: 0 });
  });

  it("ignores negative and non-numeric prices", () => {
    const row = liveRow({
      auto_routing: {
        pricing_range: {
          currency: "USD",
          prompt_per_1m_tokens: { max: -1 },
          completion_per_1m_tokens: { max: "2" },
          cache_read_per_1m_tokens: { max: Number.NaN },
        },
      },
    });
    expect(projectLowRouterModelRows([row])[0]?.cost).toEqual({ input: 0, output: 0, cacheRead: 0, cacheWrite: 0 });
  });

  it("falls back to conservative limits when the row publishes none", () => {
    const [model] = projectLowRouterModelRows([
      liveRow({ context_length: null, max_output_tokens: undefined, capabilities: { function_calling: true } }),
    ]);
    expect(model).toMatchObject({ contextWindow: 32_768, maxTokens: 8_192, input: ["text"] });
  });

  it("caps the output limit at the context window", () => {
    const [model] = projectLowRouterModelRows([liveRow({ context_length: 4_096, max_output_tokens: 65_536 })]);
    expect(model?.maxTokens).toBe(4_096);
  });

  it("fills missing live fields from the snapshot row", () => {
    const [model] = projectLowRouterModelRows(
      [
        liveRow({
          name: "",
          context_length: undefined,
          max_output_tokens: undefined,
          supported_parameters: undefined,
          capabilities: { function_calling: true },
          auto_routing: undefined,
        }),
      ],
      [snapshotRow({ input: ["text", "image"] })],
    );
    expect(model).toEqual({
      ...snapshotRow({ input: ["text", "image"] }),
      name: "auto/fixture/never-bundled",
    });
  });

  it("prefers live values over the snapshot row", () => {
    const [model] = projectLowRouterModelRows([liveRow()], [snapshotRow()]);
    expect(model).toMatchObject({
      reasoning: false,
      contextWindow: 131_072,
      cost: { input: 0.45, output: 2.2, cacheRead: 0.045, cacheWrite: 1.25 },
    });
  });
});

describe("toLowRouterModel", () => {
  it("serves the row through LowRouter's chat completions API", () => {
    expect(toLowRouterModel(snapshotRow())).toMatchObject({
      id: "auto/fixture/never-bundled",
      api: "openai-completions",
      provider: "lowrouter",
      baseUrl: LOWROUTER_BASE_URL,
      compat: { supportsStore: false, supportsDeveloperRole: false, supportsUsageInStreaming: true },
    });
    expect(toLowRouterModel(snapshotRow(), "https://proxy.example/v1").baseUrl).toBe("https://proxy.example/v1");
  });
});

describe("snapshot", () => {
  it("is non-empty and holds only public auto/ picker models", () => {
    expect(LOWROUTER_CATALOG.length).toBeGreaterThan(0);
    for (const model of LOWROUTER_CATALOG) {
      expect(model.id.startsWith("auto/") && isPickerModelId(model.id), model.id).toBe(true);
      expect(model.contextWindow, model.id).toBeGreaterThan(0);
      expect(model.maxTokens, model.id).toBeLessThanOrEqual(model.contextWindow);
    }
    expect(new Set(LOWROUTER_CATALOG.map((model) => model.id)).size).toBe(LOWROUTER_CATALOG.length);
  });
});

describe("fetchLowRouterModelRows", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("sends the key and returns the data array", async () => {
    const fetchMock = vi.fn(async () => Response.json({ data: [liveRow()] }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(fetchLowRouterModelRows({ apiKey: "sk-lr-test", baseUrl: "https://proxy.example/v1/" })).resolves.toEqual([
      liveRow(),
    ]);
    expect(fetchMock).toHaveBeenCalledWith(
      `https://proxy.example/v1/${LOWROUTER_MODELS_PATH}`,
      expect.objectContaining({
        headers: { accept: "application/json", authorization: "Bearer sk-lr-test" },
        signal: expect.any(AbortSignal),
      }),
    );
  });

  it("rejects a failed request", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("nope", { status: 401 })));
    await expect(fetchLowRouterModelRows({ apiKey: "sk-lr-bad" })).rejects.toThrow("failed: 401");
  });

  it("rejects a body without a data array", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => Response.json({ object: "list" })));
    await expect(fetchLowRouterModelRows({ apiKey: "sk-lr-test" })).rejects.toThrow("Invalid LowRouter model list");
  });
});
