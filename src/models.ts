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
 * without it EUR-billed routes are listed in EUR. The server keeps only chat
 * models with tool calling, the ones Pi can act through.
 */
export const LOWROUTER_MODELS_PATH = "models?currency=usd&function_calling=yes&modality=chat";

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

/** The Pi chat APIs LowRouter models are served through. */
export type LowRouterApi = "openai-completions" | "anthropic-messages";
export type LowRouterModel = Model<"openai-completions"> | Model<"anthropic-messages">;

/** The data half of a model; the release-time snapshot stores only this. */
export type LowRouterModelRow = Pick<
  LowRouterModel,
  "id" | "name" | "reasoning" | "thinkingLevelMap" | "promptCache" | "input" | "cost" | "contextWindow" | "maxTokens"
> & {
  /**
   * `auto_routing.reasoning_mode` of a Claude model: `adaptive` takes Pi's
   * thinking level as an effort, `budget` as a thinking-token budget.
   */
  reasoningMode?: ReasoningMode;
};

/** `reasoning_mode` values Pi's Anthropic client acts on. */
const REASONING_MODES = ["adaptive", "budget"] as const;
type ReasoningMode = (typeof REASONING_MODES)[number];

/** `reasoning_effort` values LowRouter publishes, in Pi's thinking-level order. */
const EFFORT_LEVELS = ["minimal", "low", "medium", "high", "xhigh"] as const;
type EffortLevel = (typeof EFFORT_LEVELS)[number];

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
    reasoning_effort?: unknown;
    reasoning_mode?: unknown;
    prompt_cache?: { ttl_seconds?: unknown; extended_ttl_seconds?: unknown };
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

/**
 * Claude models, served on LowRouter's Anthropic-native `/v1/messages` so thinking
 * blocks, `cache_control` breakpoints and tool use keep Anthropic's own shape.
 * Among picker ids only `auto/anthropic/` ones are Claude; an `alias/` model's
 * target is not published, so it stays on chat completions. The rows' `endpoints`
 * cannot decide it: every model lists `messages`, Anthropic-native or translated.
 */
export function isAnthropicMessagesModelId(id: string): boolean {
  return id.startsWith("auto/anthropic/");
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
 * Pi's thinking levels mapped onto the `reasoning_effort` values the model's
 * default routes apply. An unpublished level is `null`, so Pi does not offer it.
 * `off` stays unmapped: on chat completions Pi sends no effort and the model keeps
 * its default; on the Anthropic API Pi disables thinking.
 */
function thinkingLevelMap(levels: ReadonlySet<EffortLevel>): NonNullable<LowRouterModelRow["thinkingLevelMap"]> {
  const map: NonNullable<LowRouterModelRow["thinkingLevelMap"]> = {};
  for (const level of EFFORT_LEVELS) {
    if (levels.has(level)) {
      map[level] = level;
    } else if (level !== "xhigh") {
      // Pi offers xhigh only when mapped, and the other levels unless null.
      map[level] = null;
    }
  }
  return map;
}

/**
 * Reasoning support. `auto_routing.reasoning_effort` lists the levels every
 * default route applies as sent; an `auto/` row without it gets no effort, which
 * is the model's default. A row without `auto_routing` (an `alias/` model) falls
 * back to `supported_parameters`, then to the snapshot row.
 */
function readReasoning(
  row: LowRouterLiveModelRow,
  fallback: LowRouterModelRow | undefined,
): Pick<LowRouterModelRow, "reasoning" | "thinkingLevelMap"> {
  const routing = row.auto_routing;
  if (routing) {
    const published = Array.isArray(routing.reasoning_effort) ? routing.reasoning_effort : [];
    const levels = new Set(EFFORT_LEVELS.filter((level) => published.includes(level)));
    return levels.size > 0 ? { reasoning: true, thinkingLevelMap: thinkingLevelMap(levels) } : { reasoning: false };
  }
  const params = row.supported_parameters;
  if (Array.isArray(params) && (params.includes("reasoning_effort") || params.includes("reasoning"))) {
    return { reasoning: true };
  }
  return fallback?.thinkingLevelMap
    ? { reasoning: fallback.reasoning, thinkingLevelMap: fallback.thinkingLevelMap }
    : { reasoning: fallback?.reasoning ?? false };
}

/**
 * How a Claude model takes Pi's thinking level: `auto_routing.reasoning_mode`.
 * Only Claude rows carry a mode. One the row leaves out (or a row without
 * `auto_routing`), or one Pi does not act on, keeps the snapshot's.
 */
function readReasoningMode(
  id: string,
  row: LowRouterLiveModelRow,
  fallback: LowRouterModelRow | undefined,
): ReasoningMode | undefined {
  if (!isAnthropicMessagesModelId(id)) {
    return undefined;
  }
  return REASONING_MODES.find((mode) => mode === row.auto_routing?.reasoning_mode) ?? fallback?.reasoningMode;
}

/**
 * Prompt cache lifetimes of a Claude model, in seconds: `auto_routing.prompt_cache`
 * `ttl_seconds` for Pi's short retention, `extended_ttl_seconds` for long. Pi keeps
 * a cache warm only for a model that declares the lifetime of the retention used.
 * Only Claude rows get one: Pi sends `cache_control` on the Anthropic API alone,
 * and LowRouter adds none on chat completions, so a chat completions model (an
 * `alias/` to Claude among them) writes no cache for Pi to warm, and each refresh
 * would bill the full prompt. A row that leaves the field out states no lifetime,
 * so none is declared, not even the snapshot's.
 */
function readPromptCache(id: string, row: LowRouterLiveModelRow): LowRouterModelRow["promptCache"] {
  if (!isAnthropicMessagesModelId(id)) {
    return undefined;
  }
  const short = readPositiveInteger(row.auto_routing?.prompt_cache?.ttl_seconds);
  const long = readPositiveInteger(row.auto_routing?.prompt_cache?.extended_ttl_seconds);
  return short || long ? { ...(short && { short }), ...(long && { long }) } : undefined;
}

function projectRow(
  row: LowRouterLiveModelRow,
  fallbacks: ReadonlyMap<string, LowRouterModelRow>,
): LowRouterModelRow | undefined {
  const id = typeof row.id === "string" ? row.id.trim() : "";
  if (!id || !isPickerModelId(id)) {
    return undefined;
  }
  // Pi acts through tools: embeddings, realtime and tool-less chat models would
  // fail on the first turn. The query already asks for these; this guards a
  // server that ignores the filter.
  if (row.modality !== "chat" || row.capabilities?.function_calling !== true) {
    return undefined;
  }

  const fallback = fallbacks.get(id);
  const pricing = row.auto_routing?.pricing_range;
  // Pi costs are USD. A range in any other currency is ignored rather than shown
  // under the wrong symbol.
  const usd = pricing?.currency === "USD" ? pricing : undefined;

  const reasoningMode = readReasoningMode(id, row, fallback);
  const promptCache = readPromptCache(id, row);
  const contextWindow =
    readPositiveInteger(row.context_length) ?? fallback?.contextWindow ?? DEFAULT_CONTEXT_WINDOW;
  const maxTokens = Math.min(
    readPositiveInteger(row.max_output_tokens) ?? fallback?.maxTokens ?? DEFAULT_MAX_TOKENS,
    contextWindow,
  );

  return {
    id,
    name: readName(row.name, id),
    ...readReasoning(row, fallback),
    ...(reasoningMode && { reasoningMode }),
    ...(promptCache && { promptCache }),
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
    const model = projectRow(record as LowRouterLiveModelRow, byId);
    if (!model || seen.has(model.id)) {
      continue;
    }
    seen.add(model.id);
    models.push(model);
  }
  return models;
}

/**
 * Base URL of the Anthropic API behind an OpenAI-style one: Pi's Anthropic
 * client appends `/v1/messages`, so a trailing `/v1` is dropped. Applied per
 * request, so a `models.json` `baseUrl` override is converted too.
 */
export function toAnthropicBaseUrl(baseUrl: string): string {
  return baseUrl.replace(/\/+$/, "").replace(/\/v1$/, "");
}

/**
 * Turns a data row into a Pi chat model served through LowRouter: Claude on the
 * Anthropic API, every other model on chat completions. `baseUrl` is the
 * OpenAI-style base the provider is configured with; every model keeps it.
 */
export function toLowRouterModel(row: LowRouterModelRow, baseUrl = LOWROUTER_BASE_URL): LowRouterModel {
  const { reasoningMode, ...data } = row;
  const base = { ...data, provider: LOWROUTER_PROVIDER_ID, baseUrl };
  if (isAnthropicMessagesModelId(row.id)) {
    return {
      ...base,
      api: "anthropic-messages",
      // An adaptive model gets its level as an effort through thinkingLevelMap;
      // the others get a thinking-token budget for it.
      compat: { forceAdaptiveThinking: reasoningMode === "adaptive" },
    };
  }
  return { ...base, api: "openai-completions", compat: { ...LOWROUTER_COMPAT } };
}

/**
 * Reads LowRouter's `/v1/models` and returns its rows. With a key, a catalog
 * allowlist trims the list and the account's `alias/` models are added; without
 * one, the public catalog comes back (what the release snapshot is built from).
 */
export async function fetchLowRouterModelRows(options: {
  baseUrl: string;
  apiKey?: string;
  signal?: AbortSignal;
}): Promise<unknown[]> {
  const timeout = AbortSignal.timeout(MODELS_TIMEOUT_MS);
  const headers: Record<string, string> = { accept: "application/json" };
  if (options.apiKey) {
    headers.authorization = `Bearer ${options.apiKey}`;
  }
  const baseUrl = options.baseUrl.replace(/\/+$/, "");
  const response = await fetch(`${baseUrl}/${LOWROUTER_MODELS_PATH}`, {
    headers,
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
