// Live tests against LowRouter's production API, through Pi's model runtime. Gated
// on LOWROUTER_LIVE_TEST=1 and LOWROUTER_API_KEY; they spend a few cents of the
// key's credit.
import {
  type Api,
  type AssistantMessage,
  type Context,
  createModels,
  type Model,
  type SimpleStreamOptions,
  type Tool,
  Type,
} from "@earendil-works/pi-ai";
import { beforeAll, describe, expect, it } from "vitest";
import { lowrouterProvider } from "../src/index.js";

const apiKey = process.env.LOWROUTER_API_KEY?.trim();
const live = process.env.LOWROUTER_LIVE_TEST === "1" && Boolean(apiKey);

// One model per family the picker leads with; each must stream and call tools.
// Not auto/qwen/qwen3-coder-30b-a3b: its aws-bedrock br-gru route returns tool
// calls as text about half the time, a LowRouter routing issue, not a Pi one.
const MODELS = [
  "auto/mistralai/mistral-large-2512",
  "auto/mistralai/mistral-small-2603",
  "auto/qwen/qwen3-coder-next",
  "auto/anthropic/claude-haiku-4.5",
  "auto/openai/gpt-5.5",
];

// Models whose default routes publish `reasoning_effort` levels.
const REASONING_MODELS = ["auto/anthropic/claude-sonnet-4.6", "auto/google/gemini-3.1-pro-preview"];

const readFileTool: Tool = {
  name: "read_file",
  description: "Read a file from the workspace and return its contents.",
  parameters: Type.Object({ path: Type.String({ description: "Workspace-relative path" }) }),
};

const models = createModels({
  authContext: {
    env: async (name) => (name === "LOWROUTER_API_KEY" ? apiKey : undefined),
    fileExists: async () => false,
  },
});
models.setProvider(lowrouterProvider());

function liveModel(id: string): Model<Api> {
  const model = models.getModel("lowrouter", id);
  if (!model) {
    throw new Error(`${id} is not in the live catalog`);
  }
  return model;
}

async function run(
  id: string,
  context: Context,
  options?: SimpleStreamOptions,
): Promise<{ message: AssistantMessage; textDeltas: number }> {
  const stream = models.streamSimple(liveModel(id), context, options);
  let textDeltas = 0;
  for await (const event of stream) {
    if (event.type === "text_delta") {
      textDeltas++;
    }
  }
  return { message: await stream.result(), textDeltas };
}

describe.skipIf(!live).concurrent("LowRouter live", () => {
  beforeAll(async () => {
    const { errors } = await models.refresh({ providers: ["lowrouter"] });
    expect(errors.get("lowrouter")).toBeUndefined();
  });

  it("discovers the account's models", () => {
    const ids = models.getModels("lowrouter").map((model) => model.id);
    expect(ids.length).toBeGreaterThan(50);
    expect(ids).toEqual(expect.arrayContaining(MODELS));
  });

  for (const id of MODELS) {
    it(`${id}: streams text with usage`, { timeout: 120_000 }, async () => {
      const { message, textDeltas } = await run(id, {
        messages: [{ role: "user", content: "Reply with the single word: pong", timestamp: Date.now() }],
      });
      expect(message.stopReason, message.errorMessage).toBe("stop");
      expect(textDeltas).toBeGreaterThan(0);
      expect(message.usage.input).toBeGreaterThan(0);
      expect(message.usage.output).toBeGreaterThan(0);
    });

    it(`${id}: calls a tool`, { timeout: 120_000 }, async () => {
      const { message } = await run(id, {
        systemPrompt: "You are a coding agent. Use tools to read files; never guess their contents.",
        messages: [{ role: "user", content: "What does README.md say? Read it first.", timestamp: Date.now() }],
        tools: [readFileTool],
      });
      expect(message.stopReason, message.errorMessage).toBe("toolUse");
      const call = message.content.find((block) => block.type === "toolCall");
      expect(call).toMatchObject({ name: "read_file", arguments: { path: expect.stringContaining("README") } });
    });
  }

  // A model whose routes publish effort levels must take one through Pi's thinking
  // level and still answer.
  for (const id of REASONING_MODELS) {
    it(`${id}: answers with a thinking level`, { timeout: 120_000 }, async () => {
      const model = liveModel(id);
      expect(model.reasoning).toBe(true);
      const { message } = await run(
        id,
        { messages: [{ role: "user", content: "What is 17 * 23? Reply with the number only.", timestamp: Date.now() }] },
        { reasoning: "medium" },
      );
      expect(message.stopReason, message.errorMessage).toBe("stop");
      expect(message.content.some((block) => block.type === "text" && block.text.includes("391"))).toBe(true);
    });
  }
});
