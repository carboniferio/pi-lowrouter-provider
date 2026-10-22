/**
 * LowRouter provider for Pi.
 *
 * Registers `lowrouter` with the models of the release-time snapshot, then
 * replaces them with the live `/v1/models` list: the public one without a key,
 * the account's own (allowlist applied, `alias/` models added) once a key is
 * configured (`LOWROUTER_API_KEY` or `/login`). Pi keeps the last list for
 * offline starts. Claude models use Pi's Anthropic API, the others chat
 * completions.
 */
// Pi maps a fixed set of host modules into extensions; `compat` is the mapped
// entry that also exports the API implementations.
import {
  anthropicMessagesApi,
  createProvider,
  envApiKeyAuth,
  lazyStream,
  openAICompletionsApi,
  type Provider,
  type ProviderStreams,
} from "@earendil-works/pi-ai/compat";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { LOWROUTER_CATALOG } from "./catalog.generated.js";
import {
  fetchLowRouterModelRows,
  LOWROUTER_BASE_URL,
  LOWROUTER_PROVIDER_ID,
  type LowRouterApi,
  projectLowRouterModelRows,
  toAnthropicBaseUrl,
  toLowRouterModel,
} from "./models.js";

const SNAPSHOT_MODELS = LOWROUTER_CATALOG.map((row) => toLowRouterModel(row));

const MISSING_KEY_MESSAGE =
  "LowRouter needs an API key to send requests. Run /login and pick LowRouter, or set " +
  "LOWROUTER_API_KEY. Get a key at https://lowrouter.ai/app/keys";

/**
 * `LOWROUTER_API_KEY` or a `/login` key. Without one, auth still resolves, with
 * no credential: Pi then counts LowRouter as configured, lists its models and
 * refreshes them from the public `/v1/models`. There is deliberately no `check`,
 * which Pi would run instead of `resolve` to decide whether to list the models.
 */
function lowrouterApiKeyAuth() {
  const auth = envApiKeyAuth("LowRouter API key", ["LOWROUTER_API_KEY"]);
  return {
    ...auth,
    resolve: async (input: Parameters<typeof auth.resolve>[0]) =>
      (await auth.resolve(input)) ?? { auth: {}, source: "no key (public model list)" },
  };
}

function hasHeader(headers: Record<string, string | null> | undefined, name: string): boolean {
  return Object.entries(headers ?? {}).some(([key, value]) => key.toLowerCase() === name && Boolean(value?.trim()));
}

/**
 * Fails a request that carries no key with LowRouter's own message, before any
 * network call. A `models.json` header can carry the key instead, as Pi's APIs
 * accept.
 */
function requireKey(api: ProviderStreams): ProviderStreams {
  const hasKey = (options?: { apiKey?: string; headers?: Record<string, string | null> }) =>
    Boolean(options?.apiKey) || hasHeader(options?.headers, "authorization") || hasHeader(options?.headers, "x-api-key");
  const missingKey = (model: Parameters<ProviderStreams["stream"]>[0]) =>
    lazyStream(model, async () => {
      throw new Error(MISSING_KEY_MESSAGE);
    });
  return {
    stream: (model, context, options) => (hasKey(options) ? api.stream(model, context, options) : missingKey(model)),
    streamSimple: (model, context, options) =>
      hasKey(options) ? api.streamSimple(model, context, options) : missingKey(model),
  };
}

/**
 * Pi's Anthropic API on LowRouter's `/v1/messages`. Every model keeps the
 * OpenAI-style base URL (a `models.json` `baseUrl` replaces it on all of them),
 * and the trailing `/v1` is dropped per request, since Pi's Anthropic client
 * appends `/v1/messages` itself.
 */
function lowrouterAnthropicApi(): ProviderStreams {
  const api = anthropicMessagesApi();
  const onMessages = <T extends { baseUrl: string }>(model: T): T => ({
    ...model,
    baseUrl: toAnthropicBaseUrl(model.baseUrl),
  });
  // Only the two methods that take the converted model: passing the others
  // through would send them to the OpenAI-style base URL.
  return {
    stream: (model, context, options) => api.stream(onMessages(model), context, options),
    streamSimple: (model, context, options) => api.streamSimple(onMessages(model), context, options),
  };
}

/** Builds the LowRouter provider. */
export function lowrouterProvider(): Provider<LowRouterApi> {
  const provider = createProvider<LowRouterApi>({
    id: LOWROUTER_PROVIDER_ID,
    name: "LowRouter",
    baseUrl: LOWROUTER_BASE_URL,
    auth: { apiKey: lowrouterApiKeyAuth() },
    models: [],
    fetchModels: async (context) => {
      const apiKey = context.credential?.type === "api_key" ? context.credential.key : undefined;
      const baseUrl = (context.baseUrl || LOWROUTER_BASE_URL).replace(/\/+$/, "");
      const rows = await fetchLowRouterModelRows({ baseUrl, apiKey, signal: context.signal });
      const models = projectLowRouterModelRows(rows, LOWROUTER_CATALOG).map((row) => toLowRouterModel(row, baseUrl));
      // A key's allowlist can leave nothing usable; the public list cannot.
      // Throwing keeps the previous list, where returning [] would persist an
      // empty one over it.
      if (!apiKey && models.length === 0) {
        throw new Error("LowRouter's public model list is empty");
      }
      return models;
    },
    // Claude models go to /v1/messages, the rest to /v1/chat/completions.
    api: {
      "openai-completions": requireKey(openAICompletionsApi()),
      "anthropic-messages": requireKey(lowrouterAnthropicApi()),
    },
  });
  // The snapshot stands in only until a live list exists, restored or fetched.
  // As createProvider's baseline it would never be dropped, and a key with a
  // catalog allowlist would keep listing models it cannot use. A published live
  // list replaces it even when empty: an allowlist can leave nothing usable.
  let hasLiveList = false;
  const refreshModels = provider.refreshModels!;
  const getAllModels = provider.getAllModels!;
  return {
    ...provider,
    getModels: () => (hasLiveList ? provider.getModels() : [...SNAPSHOT_MODELS]),
    getAllModels: () => (hasLiveList ? getAllModels() : [...SNAPSHOT_MODELS]),
    refreshModels: (context) =>
      refreshModels({
        ...context,
        publish: (publication) =>
          context.publish({
            ...publication,
            update: () => {
              publication.update?.();
              hasLiveList = true;
            },
          }),
      }),
  };
}

export default function lowrouterExtension(pi: ExtensionAPI): void {
  pi.registerProvider(lowrouterProvider());
}
