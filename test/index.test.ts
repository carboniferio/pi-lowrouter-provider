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

function stubFetch(response: () => Response) {
  const fetchMock = vi.fn(async (_url: string, _init?: RequestInit) => response());
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

const ids = (models: ReturnType<typeof runtime>) => models.getModels("lowrouter").map((model) => model.id);

/** A streamed Anthropic Messages response that answers "pong". */
function anthropicStream(): Response {
  const events: Record<string, unknown>[] = [
    {
      type: "message_start",
      message: {
        id: "msg_test",
        type: "message",
        role: "assistant",
        model: "claude",
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: 12, output_tokens: 1, cache_read_input_tokens: 0, cache_creation_input_tokens: 0 },
      },
    },
    { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
    { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "pong" } },
    { type: "content_block_stop", index: 0 },
    { type: "message_delta", delta: { stop_reason: "end_turn", stop_sequence: null }, usage: { output_tokens: 2 } },
    { type: "message_stop" },
  ];
  const body = events.map((event) => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`).join("");
  return new Response(body, { headers: { "content-type": "text/event-stream" } });
}

/** Whether a model-list request carried the key. */
const withKey = (call: [string, RequestInit?]) => new Headers(call[1]?.headers).has("authorization");

describe("lowrouterProvider", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
  });

  it("starts from the snapshot", () => {
    const models = runtime();
    expect(ids(models)).toEqual(snapshotIds);
    expect(models.getModel("lowrouter", "auto/anthropic/claude-sonnet-4.6")?.api).toBe("anthropic-messages");
    expect(models.getModel("lowrouter", "auto/mistralai/mistral-large-2512")?.api).toBe("openai-completions");
  });

  it("replaces the snapshot with the live list", async () => {
    const fetchMock = stubFetch(() => Response.json({ data: liveRows }));
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
    stubFetch(() => Response.json({ data: [] }));
    const models = runtime();

    const result = await models.refresh({ providers: ["lowrouter"] });

    expect(result.errors.size).toBe(0);
    expect(ids(models)).toEqual([]);
  });

  it("restores the last live list offline", async () => {
    stubFetch(() => Response.json({ data: liveRows }));
    const modelsStore = new InMemoryModelsStore();
    await runtime({ modelsStore }).refresh({ providers: ["lowrouter"] });
    const fetchMock = stubFetch(() => new Response("unreachable", { status: 500 }));
    vi.stubEnv("PI_OFFLINE", "1");

    const offline = runtime({ modelsStore });
    await offline.refresh({ providers: ["lowrouter"], allowNetwork: false });

    expect(fetchMock).not.toHaveBeenCalled();
    expect(ids(offline)).toEqual(["auto/fixture/live-only", "alias/team-coder"]);
  });

  it("lists the public list without a key, without waiting for it", async () => {
    let respond!: () => void;
    const fetchMock = vi.fn(
      (_url: string, _init?: RequestInit) =>
        new Promise<Response>((resolve) => {
          respond = () => resolve(Response.json({ data: liveRows }));
        }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const modelsStore = new InMemoryModelsStore();
    const models = runtime({ key: null, modelsStore });

    const result = await models.refresh({ providers: ["lowrouter"] });

    expect(result.errors.size).toBe(0);
    expect(ids(models)).toEqual(snapshotIds);
    respond();
    await vi.waitFor(() => expect(ids(models)).toEqual(["auto/fixture/live-only", "alias/team-coder"]));
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0]![0]).toBe(`${LOWROUTER_BASE_URL}/${LOWROUTER_MODELS_PATH}`);
    expect(withKey(fetchMock.mock.calls[0]!)).toBe(false);
    // Pi's cache holds only keyed lists.
    expect(await modelsStore.read("lowrouter")).toBeUndefined();

    await models.refresh({ providers: ["lowrouter"] });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("keeps the snapshot when the public list fails or is empty", async () => {
    for (const response of [() => new Response("down", { status: 503 }), () => Response.json({ data: [] })]) {
      const fetchMock = stubFetch(response);
      const models = runtime({ key: null });

      await models.refresh({ providers: ["lowrouter"] });
      await vi.waitFor(() => expect(fetchMock).toHaveBeenCalled());
      await new Promise((resolve) => setTimeout(resolve, 0));

      expect(ids(models)).toEqual(snapshotIds);
    }
  });

  it("prefers the keyed list to the public one", async () => {
    const keyedRows = liveRows.filter((row) => row.id === "alias/team-coder");
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) =>
      Response.json({ data: withKey([url, init]) ? keyedRows : liveRows }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const models = runtime();

    await models.refresh({ providers: ["lowrouter"] });
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(fetchMock.mock.calls.map(withKey).sort()).toEqual([false, true]);
    expect(ids(models)).toEqual(["alias/team-coder"]);
  });

  it("prefers Pi's cached list to the public one", async () => {
    const modelsStore = new InMemoryModelsStore();
    await modelsStore.write("lowrouter", {
      models: [{ ...lowrouterProvider().getModels()[0]!, id: "alias/cached" }],
      checkedAt: Date.now(),
    });
    const fetchMock = stubFetch(() => Response.json({ data: liveRows }));
    const models = runtime({ key: null, modelsStore });

    await models.refresh({ providers: ["lowrouter"] });
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalled());
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(ids(models)).toEqual(["alias/cached"]);
  });

  it("keeps the snapshot and makes no request offline", async () => {
    vi.stubEnv("PI_OFFLINE", "1");
    const fetchMock = stubFetch(() => Response.json({ data: liveRows }));
    const models = runtime();

    await models.refresh({ providers: ["lowrouter"], allowNetwork: false });

    expect(fetchMock).not.toHaveBeenCalled();
    expect(ids(models)).toEqual(snapshotIds);
  });

  it("keeps the previous list when the refresh fails", async () => {
    stubFetch(() => new Response("down", { status: 503 }));
    const models = runtime();

    const result = await models.refresh({ providers: ["lowrouter"] });

    expect(result.errors.get("lowrouter")?.message).toContain("failed: 503");
    expect(ids(models)).toEqual(snapshotIds);
  });

  it("serves refreshed models from the configured base URL", async () => {
    const fetchMock = stubFetch(() => Response.json({ data: liveRows }));
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

describe("requests", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it.each([
    // Adaptive: Pi's level goes out as the effort the routes take.
    [
      "auto/anthropic/claude-sonnet-4.6",
      { thinking: { type: "adaptive", display: "summarized" }, output_config: { effort: "medium" } },
    ],
    // Budget: Pi's level goes out as a thinking-token budget.
    ["auto/anthropic/claude-haiku-4.5", { thinking: { type: "enabled", budget_tokens: expect.any(Number) } }],
  ])("sends %s to /v1/messages in Anthropic's shape", async (id, thinking) => {
    const fetchMock = stubFetch(anthropicStream);
    const models = runtime();

    const message = await models.completeSimple(
      models.getModel("lowrouter", id)!,
      {
        systemPrompt: "You are a coding agent.",
        messages: [{ role: "user", content: "Reply with the single word: pong", timestamp: Date.now() }],
      },
      { reasoning: "medium" },
    );

    expect(message.stopReason, message.errorMessage).toBe("stop");
    expect(message.content).toEqual([expect.objectContaining({ type: "text", text: "pong" })]);
    expect(message.usage).toMatchObject({ input: 12, output: 2 });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("https://api.lowrouter.ai/v1/messages?beta=true");
    expect(new Headers(init?.headers).get("x-api-key")).toBe("sk-lr-test");
    const body = JSON.parse(String(init?.body));
    expect(body).toMatchObject({ model: id, stream: true, ...thinking });
    expect(body.system).toEqual([expect.objectContaining({ cache_control: { type: "ephemeral" } })]);
    expect(body.messages.at(-1).content.at(-1)).toMatchObject({ cache_control: { type: "ephemeral" } });
  });

  it("drops /v1 from an overridden base URL for /v1/messages", async () => {
    // A models.json `baseUrl` replaces every model's base URL with the OpenAI-style one.
    const fetchMock = stubFetch(anthropicStream);
    const models = runtime();
    const model = models.getModel("lowrouter", "auto/anthropic/claude-haiku-4.5")!;

    const message = await models.completeSimple(
      { ...model, baseUrl: "https://proxy.example/v1" },
      { messages: [{ role: "user", content: "ping", timestamp: Date.now() }] },
    );

    expect(message.stopReason, message.errorMessage).toBe("stop");
    expect(String(fetchMock.mock.calls[0]![0])).toBe("https://proxy.example/v1/messages?beta=true");
  });

  it("sends other models to /v1/chat/completions", async () => {
    const fetchMock = stubFetch(() => new Response("down", { status: 400 }));
    const models = runtime();

    await models.completeSimple(models.getModel("lowrouter", "auto/mistralai/mistral-large-2512")!, {
      messages: [{ role: "user", content: "ping", timestamp: Date.now() }],
    });

    expect(fetchMock).toHaveBeenCalled();
    expect(String(fetchMock.mock.calls[0]![0])).toBe(`${LOWROUTER_BASE_URL}/chat/completions`);
  });
});

describe("extension", () => {
  it("registers the LowRouter provider", () => {
    const registered: Provider[] = [];
    lowrouterExtension({ registerProvider: (provider: Provider) => registered.push(provider) } as unknown as ExtensionAPI);
    expect(registered.map((provider) => provider.id)).toEqual(["lowrouter"]);
  });
});
