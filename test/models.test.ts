// LowRouter tests cover the snapshot and the live /v1/models projection.
import { afterEach, describe, expect, it, vi } from "vitest";
import { LOWROUTER_CATALOG } from "../src/catalog.generated.js";
import {
  fetchLowRouterModelRows,
  isAnthropicMessagesModelId,
  isPickerModelId,
  LOWROUTER_BASE_URL,
  LOWROUTER_MODELS_PATH,
  type LowRouterModelRow,
  projectLowRouterModelRows,
  toAnthropicBaseUrl,
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

  it("maps the published effort levels onto Pi's thinking levels", () => {
    const [model] = projectLowRouterModelRows([
      liveRow({
        auto_routing: { reasoning_mode: "adaptive", reasoning_effort: ["low", "medium", "high", "xhigh"] },
      }),
    ]);
    expect(model).toMatchObject({
      reasoning: true,
      thinkingLevelMap: { minimal: null, low: "low", medium: "medium", high: "high", xhigh: "xhigh" },
    });
    expect(model?.thinkingLevelMap).not.toHaveProperty("off");
  });

  it("keeps a Claude model's reasoning mode when Pi's Anthropic client acts on it", () => {
    const mode = (id: string, reasoning_mode: unknown) =>
      projectLowRouterModelRows([liveRow({ id, auto_routing: { reasoning_mode, reasoning_effort: ["low", "high"] } })])[0]
        ?.reasoningMode;
    expect(["adaptive", "budget", "thinking_level", undefined].map((value) => mode("auto/anthropic/claude", value))).toEqual(
      ["adaptive", "budget", undefined, undefined],
    );
    expect(mode("auto/google/gemini", "adaptive")).toBeUndefined();
    const [kept] = projectLowRouterModelRows(
      [liveRow({ id: "auto/anthropic/claude", auto_routing: { reasoning_effort: ["low", "high"] } })],
      [snapshotRow({ id: "auto/anthropic/claude", reasoningMode: "adaptive" })],
    );
    expect(kept?.reasoningMode).toBe("adaptive");
    // An unknown mode, or a row without auto_routing, keeps the snapshot's too.
    const snapshot = [snapshotRow({ id: "auto/anthropic/claude", reasoningMode: "adaptive" })];
    const [unknown] = projectLowRouterModelRows(
      [liveRow({ id: "auto/anthropic/claude", auto_routing: { reasoning_mode: "thinking_level", reasoning_effort: ["low"] } })],
      snapshot,
    );
    expect(unknown?.reasoningMode).toBe("adaptive");
    const [unrouted] = projectLowRouterModelRows(
      [liveRow({ id: "auto/anthropic/claude", auto_routing: undefined, supported_parameters: ["reasoning_effort"] })],
      snapshot,
    );
    expect(unrouted?.reasoningMode).toBe("adaptive");
  });

  it("leaves xhigh unmapped when the routes do not take it", () => {
    const [model] = projectLowRouterModelRows([liveRow({ auto_routing: { reasoning_effort: ["low", "medium", "high"] } })]);
    expect(model?.thinkingLevelMap).toEqual({ minimal: null, low: "low", medium: "medium", high: "high" });
  });

  it("ignores effort values Pi has no level for", () => {
    const [model] = projectLowRouterModelRows([liveRow({ auto_routing: { reasoning_effort: ["none", "max", 3] } })]);
    expect(model).toMatchObject({ reasoning: false });
    expect(model).not.toHaveProperty("thinkingLevelMap");
  });

  it("does not reason when an auto row publishes no levels", () => {
    for (const auto_routing of [{ reasoning_mode: "unsupported" }, {}]) {
      const [model] = projectLowRouterModelRows(
        [liveRow({ auto_routing, supported_parameters: ["tools", "reasoning_effort"] })],
        [snapshotRow({ thinkingLevelMap: { low: "low" } })],
      );
      expect(model).toMatchObject({ reasoning: false });
      expect(model).not.toHaveProperty("thinkingLevelMap");
    }
  });

  it("reads reasoning from supported_parameters on a row without auto_routing", () => {
    const [effort] = projectLowRouterModelRows([
      liveRow({ id: "alias/a", auto_routing: undefined, supported_parameters: ["tools", "reasoning_effort"] }),
    ]);
    const [plain] = projectLowRouterModelRows([liveRow({ id: "alias/b", auto_routing: undefined })]);
    expect(effort?.reasoning).toBe(true);
    expect(plain?.reasoning).toBe(false);
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
      [snapshotRow({ input: ["text", "image"], thinkingLevelMap: { low: "low", high: "high" } })],
    );
    expect(model).toEqual({
      ...snapshotRow({ input: ["text", "image"], thinkingLevelMap: { low: "low", high: "high" } }),
      name: "auto/fixture/never-bundled",
    });
  });

  it("matches a padded live id to its snapshot row", () => {
    const [model] = projectLowRouterModelRows(
      [liveRow({ id: " auto/fixture/never-bundled ", context_length: undefined })],
      [snapshotRow()],
    );
    expect(model).toMatchObject({ id: "auto/fixture/never-bundled", contextWindow: 64_000 });
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
  it("serves a Claude row through LowRouter's Anthropic API", () => {
    const claude = snapshotRow({ id: "auto/anthropic/claude-sonnet-4.6", reasoningMode: "adaptive" });
    const model = toLowRouterModel(claude);
    expect(model).toMatchObject({
      id: "auto/anthropic/claude-sonnet-4.6",
      api: "anthropic-messages",
      provider: "lowrouter",
      baseUrl: LOWROUTER_BASE_URL,
      compat: { forceAdaptiveThinking: true },
    });
    expect(model).not.toHaveProperty("reasoningMode");
    expect(toLowRouterModel({ ...claude, reasoningMode: "budget" }).compat).toEqual({ forceAdaptiveThinking: false });
    expect(toAnthropicBaseUrl("https://proxy.example/v1/")).toBe("https://proxy.example");
    expect(toAnthropicBaseUrl("https://api.lowrouter.ai")).toBe("https://api.lowrouter.ai");
    // An alias's target is not published, so it stays on chat completions.
    expect(toLowRouterModel(snapshotRow({ id: "alias/claude" })).api).toBe("openai-completions");
  });

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
      if (isAnthropicMessagesModelId(model.id) && model.reasoning) {
        expect(model.reasoningMode, model.id).toBeDefined();
      }
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

  it("reads the public catalog without a key", async () => {
    const fetchMock = vi.fn(async () => Response.json({ data: [] }));
    vi.stubGlobal("fetch", fetchMock);

    await fetchLowRouterModelRows({ baseUrl: LOWROUTER_BASE_URL });

    expect(fetchMock).toHaveBeenCalledWith(
      `${LOWROUTER_BASE_URL}/${LOWROUTER_MODELS_PATH}`,
      expect.objectContaining({ headers: { accept: "application/json" } }),
    );
  });

  it("rejects a failed request", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("nope", { status: 401 })));
    await expect(fetchLowRouterModelRows({ baseUrl: LOWROUTER_BASE_URL, apiKey: "sk-lr-bad" })).rejects.toThrow("failed: 401");
  });

  it("rejects a body without a data array", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => Response.json({ object: "list" })));
    await expect(fetchLowRouterModelRows({ baseUrl: LOWROUTER_BASE_URL, apiKey: "sk-lr-test" })).rejects.toThrow("Invalid LowRouter model list");
  });
});
