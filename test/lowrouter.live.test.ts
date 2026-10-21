// Live tests against LowRouter's production API. Gated on LOWROUTER_LIVE_TEST=1 and
// LOWROUTER_API_KEY; they spend a few cents of the key's credit.
import { type AssistantMessage, type Context, normalizeContext } from "@earendil-works/pi-ai/compat";
import { describe, expect, it } from "vitest";
import { type LowRouterModel, lowrouterProvider } from "../src/index.js";

const apiKey = process.env.LOWROUTER_API_KEY?.trim();
const live = process.env.LOWROUTER_LIVE_TEST === "1" && Boolean(apiKey);

// One model per family the picker leads with; each must stream and call tools.
const MODELS = [
  "auto/mistralai/mistral-large-2512",
  "auto/mistralai/mistral-small-2603",
  "auto/qwen/qwen3-coder-next",
  "auto/anthropic/claude-haiku-4.5",
  "auto/openai/gpt-5.5",
];

const readFileTool = {
  name: "read_file",
  description: "Read a file from the workspace and return its contents.",
  parameters: {
    type: "object",
    properties: { path: { type: "string", description: "Workspace-relative path" } },
    required: ["path"],
  },
} as Context["tools"] extends (infer T)[] | undefined ? T : never;

async function liveModels(): Promise<{ provider: ReturnType<typeof lowrouterProvider>; models: LowRouterModel[] }> {
  const provider = lowrouterProvider();
  await provider.refreshModels?.({
    credential: { type: "api_key", key: apiKey! },
    allowNetwork: true,
    signal: AbortSignal.timeout(30_000),
    publish: async (publication) => {
      publication.update?.();
      return true;
    },
  });
  return { provider, models: [...provider.getModels()] };
}

async function liveModel(id: string): Promise<LowRouterModel> {
  const model = (await liveModels()).models.find((entry) => entry.id === id);
  if (!model) {
    throw new Error(`${id} is not in the live catalog`);
  }
  return model;
}

async function run(model: LowRouterModel, context: Context): Promise<{ message: AssistantMessage; textDeltas: number }> {
  const provider = lowrouterProvider();
  const stream = provider.streamSimple(model, normalizeContext(context), { apiKey: apiKey! });
  let textDeltas = 0;
  for await (const event of stream) {
    if (event.type === "text_delta") {
      textDeltas++;
    }
  }
  return { message: await stream.result(), textDeltas };
}

describe.skipIf(!live)("LowRouter live", () => {
  it("discovers the account's models", async () => {
    const { models } = await liveModels();
    expect(models.length).toBeGreaterThan(50);
    for (const id of MODELS) {
      expect(models.map((model) => model.id)).toContain(id);
    }
  });

  for (const id of MODELS) {
    it(`${id}: streams text with usage`, { timeout: 120_000 }, async () => {
      const model = await liveModel(id);
      const { message, textDeltas } = await run(model, {
        messages: [{ role: "user", content: "Reply with the single word: pong", timestamp: Date.now() }],
      });
      expect(message.stopReason, message.errorMessage).toBe("stop");
      expect(textDeltas).toBeGreaterThan(0);
      expect(message.usage.input).toBeGreaterThan(0);
      expect(message.usage.output).toBeGreaterThan(0);
    });

    it(`${id}: calls a tool`, { timeout: 120_000 }, async () => {
      const model = await liveModel(id);
      const { message } = await run(model, {
        systemPrompt: "You are a coding agent. Use tools to read files; never guess their contents.",
        messages: [{ role: "user", content: "What does README.md say? Read it first.", timestamp: Date.now() }],
        tools: [readFileTool],
      });
      expect(message.stopReason, message.errorMessage).toBe("toolUse");
      const call = message.content.find((block) => block.type === "toolCall");
      expect(call).toMatchObject({ name: "read_file", arguments: { path: expect.stringContaining("README") } });
    });
  }
});
