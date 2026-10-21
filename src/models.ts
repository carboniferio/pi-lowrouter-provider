/**
 * LowRouter model catalog: projection of `/v1/models` rows into Pi chat models.
 */
import type { Model, OpenAICompletionsCompat } from "@earendil-works/pi-ai";

/** Pi provider id: models are addressed as `lowrouter/<model id>`. */
export const LOWROUTER_PROVIDER_ID = "lowrouter";
/** Base URL for LowRouter's OpenAI-compatible API. */
export const LOWROUTER_BASE_URL = "https://api.lowrouter.ai/v1";
/**
 * Model-list path. `currency=usd` returns USD price ranges (Pi tracks cost in USD);
 * without it EUR-billed routes are listed in EUR.
 */
export const LOWROUTER_MODELS_PATH = "models?currency=usd";

const MODELS_TIMEOUT_MS = 15_000;

// Used only when /v1/models publishes no window for a model. Deliberately low:
// overstating a window sends prompts the model rejects, understating it only
// makes Pi compact earlier.
const DEFAULT_CONTEXT_WINDOW = 32_768;
// Used when a model publishes no output limit. Pi sends it as the request's
// max tokens, so it stays at a value every listed route accepts.
const DEFAULT_MAX_TOKENS = 8_192;

// LowRouter streams a final usage chunk on every listed model. Some upstream
// hosts reject the `developer` role and `store`.
const LOWROUTER_COMPAT: OpenAICompletionsCompat = {
  supportsStore: false,
  supportsDeveloperRole: false,
  supportsUsageInStreaming: true,
};

export type LowRouterModel = Model<"openai-completions">;

/** The data half of a model; the release-time snapshot stores only this. */
export type LowRouterModelRow = Pick<
  LowRouterModel,
  "id" | "name" | "reasoning" | "input" | "cost" | "contextWindow" | "maxTokens"
>;

type PriceRange = { min?: unknown; max?: unknown };

type LowRouterLiveModelRow = {
  id?: unknown;
  name?: unknown;
  modality?: unknown;
  supported_parameters?: unknown;
  context_length?: unknown;
  max_output_tokens?: unknown;
  capabilities?: { function_calling?: unknown; input_modalities?: unknown };
  auto_routing?: {
    pricing_range?: {
      currency?: unknown;
      prompt_per_1m_tokens?: PriceRange;
      completion_per_1m_tokens?: PriceRange;
      cache_read_per_1m_tokens?: PriceRange;
    };
  };
};

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

function readPositiveInteger(value: unknown): number | undefined {
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0 ? value : undefined;
}

/**
 * Upper bound of a per-1M price range. An `auto/` model is served by several routes
 * at different prices and the response does not say which one will serve, so the
 * dearest route is shown: Pi's cost estimate is an upper bound.
 */
function readRangeMax(range: PriceRange | undefined): number | undefined {
  const max = range?.max;
  return typeof max === "number" && Number.isFinite(max) && max >= 0 ? max : undefined;
}

/**
 * Picker ids: `auto/<creator>/<model>` (LowRouter picks and fails over between
 * routes) and the account's own `alias/<name>` models. Pinned routes
 * (`<host>/<creator>/<model>`) and jurisdiction-scoped ids
 * (`auto/<creator>/<model>/<scope>`) stay out of the list.
 */
export function isPickerModelId(id: string): boolean {
  const parts = id.split("/");
  if (parts.some((part) => part.length === 0)) {
    return false;
  }
  return (parts[0] === "auto" && parts.length === 3) || (parts[0] === "alias" && parts.length === 2);
}

function readInput(value: unknown, fallback: LowRouterModelRow | undefined): LowRouterModelRow["input"] {
  if (!Array.isArray(value)) {
    return fallback?.input ?? ["text"];
  }
  return value.includes("image") ? ["text", "image"] : ["text"];
}

function readName(value: unknown, id: string): string {
  return typeof value === "string" && value.trim() ? value.trim() : id;
}

/**
 * A model reasons when `/v1/models` lists a reasoning control for it. Without
 * `supported_parameters` the snapshot row decides; Pi then sends no effort.
 */
function readReasoning(value: unknown, fallback: LowRouterModelRow | undefined): boolean {
  if (!Array.isArray(value)) {
    return fallback?.reasoning ?? false;
  }
  return value.includes("reasoning_effort") || value.includes("reasoning");
}

function projectRow(
  row: LowRouterLiveModelRow,
  fallback: LowRouterModelRow | undefined,
): LowRouterModelRow | undefined {
  const id = typeof row.id === "string" ? row.id.trim() : "";
  if (!id || !isPickerModelId(id)) {
    return undefined;
  }
  // Pi acts through tools: embeddings, realtime and tool-less chat models would
  // fail on the first turn.
  if (row.modality !== "chat" || asRecord(row.capabilities)?.function_calling !== true) {
    return undefined;
  }

  const pricing = asRecord(row.auto_routing)?.pricing_range as
    | NonNullable<NonNullable<LowRouterLiveModelRow["auto_routing"]>["pricing_range"]>
    | undefined;
  // Pi costs are USD. A range in any other currency is ignored rather than shown
  // under the wrong symbol.
  const usd = pricing?.currency === "USD" ? pricing : undefined;

  const contextWindow =
    readPositiveInteger(row.context_length) ?? fallback?.contextWindow ?? DEFAULT_CONTEXT_WINDOW;
  const maxTokens = Math.min(
    readPositiveInteger(row.max_output_tokens) ?? fallback?.maxTokens ?? DEFAULT_MAX_TOKENS,
    contextWindow,
  );

  return {
    id,
    name: readName(row.name, id),
    reasoning: readReasoning(row.supported_parameters, fallback),
    input: readInput(row.capabilities?.input_modalities, fallback),
    cost: {
      input: readRangeMax(usd?.prompt_per_1m_tokens) ?? fallback?.cost.input ?? 0,
      output: readRangeMax(usd?.completion_per_1m_tokens) ?? fallback?.cost.output ?? 0,
      cacheRead: readRangeMax(usd?.cache_read_per_1m_tokens) ?? fallback?.cost.cacheRead ?? 0,
      cacheWrite: fallback?.cost.cacheWrite ?? 0,
    },
    contextWindow,
    maxTokens,
  };
}

/**
 * Projects `/v1/models` rows into model rows. `fallbacks` (the snapshot) fills
 * fields a live row leaves out; pass none to generate the snapshot itself.
 */
export function projectLowRouterModelRows(
  rows: readonly unknown[],
  fallbacks: readonly LowRouterModelRow[] = [],
): LowRouterModelRow[] {
  const byId = new Map(fallbacks.map((model) => [model.id, model]));
  const seen = new Set<string>();
  const models: LowRouterModelRow[] = [];
  for (const row of rows) {
    const record = asRecord(row);
    if (!record) {
      continue;
    }
    const model = projectRow(record as LowRouterLiveModelRow, byId.get(String(record.id)));
    if (!model || seen.has(model.id)) {
      continue;
    }
    seen.add(model.id);
    models.push(model);
  }
  return models;
}

/** Turns a data row into a Pi chat model served through LowRouter. */
export function toLowRouterModel(row: LowRouterModelRow, baseUrl = LOWROUTER_BASE_URL): LowRouterModel {
  return {
    ...row,
    api: "openai-completions",
    provider: LOWROUTER_PROVIDER_ID,
    baseUrl,
    compat: { ...LOWROUTER_COMPAT },
  };
}

/**
 * Reads LowRouter's `/v1/models` with the user's key and returns its rows. A key
 * restricted to a catalog allowlist gets only the allowed models, and the
 * account's `alias/` models only appear with a key.
 */
export async function fetchLowRouterModelRows(options: {
  apiKey: string;
  baseUrl?: string;
  signal?: AbortSignal;
}): Promise<unknown[]> {
  const baseUrl = (options.baseUrl ?? LOWROUTER_BASE_URL).replace(/\/+$/, "");
  const timeout = AbortSignal.timeout(MODELS_TIMEOUT_MS);
  const response = await fetch(`${baseUrl}/${LOWROUTER_MODELS_PATH}`, {
    headers: { accept: "application/json", authorization: `Bearer ${options.apiKey}` },
    signal: options.signal ? AbortSignal.any([options.signal, timeout]) : timeout,
  });
  if (!response.ok) {
    throw new Error(`LowRouter model list request failed: ${response.status}`);
  }
  const body: unknown = await response.json();
  const data = asRecord(body)?.data;
  if (!Array.isArray(data)) {
    throw new Error("Invalid LowRouter model list response");
  }
  return data;
}
