/**
 * LowRouter provider for Pi.
 *
 * Registers `lowrouter` with the models of the release-time snapshot, then
 * replaces them with the live `/v1/models` list: the public one without a key,
 * the account's own (allowlist applied, `alias/` models added) once a key is
 * configured (`LOWROUTER_API_KEY` or `/login`). Pi keeps the last keyed list for
 * offline starts. Claude models use Pi's Anthropic API, the others chat
 * completions.
 */
// Pi maps a fixed set of host modules into extensions; `compat` is the mapped
// entry that also exports the API implementations.
import {
  anthropicMessagesApi,
  createProvider,
  envApiKeyAuth,
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
  type LowRouterModel,
  projectLowRouterModelRows,
  toAnthropicBaseUrl,
  toLowRouterModel,
} from "./models.js";

const SNAPSHOT_MODELS = LOWROUTER_CATALOG.map((row) => toLowRouterModel(row));

/** The live `/v1/models` list as Pi models; without `apiKey`, the public one. */
async function fetchModelList(
  configuredBaseUrl: string | undefined,
  apiKey?: string,
  signal?: AbortSignal,
): Promise<LowRouterModel[]> {
  const baseUrl = (configuredBaseUrl || LOWROUTER_BASE_URL).replace(/\/+$/, "");
  const rows = await fetchLowRouterModelRows({ baseUrl, apiKey, signal });
  return projectLowRouterModelRows(rows, LOWROUTER_CATALOG).map((row) => toLowRouterModel(row, baseUrl));
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
    auth: { apiKey: envApiKeyAuth("LowRouter API key", ["LOWROUTER_API_KEY"]) },
    models: [],
    fetchModels: async (context) => {
      // Pi fetches only with a resolved key. Throwing keeps the cached list;
      // returning [] would persist an empty one over it.
      if (context.credential?.type !== "api_key") {
        throw new Error("LowRouter model refresh needs an API key");
      }
      return fetchModelList(context.baseUrl, context.credential.key, context.signal);
    },
    // Claude models go to /v1/messages, the rest to /v1/chat/completions.
    api: { "openai-completions": openAICompletionsApi(), "anthropic-messages": lowrouterAnthropicApi() },
  });
  // The snapshot stands in only until a live list exists. As createProvider's
  // baseline it would never be dropped, and a key with a catalog allowlist would
  // keep listing models it cannot use. A list Pi publishes (the keyed one, or
  // Pi's cache of it) wins, even when empty: an allowlist can leave nothing
  // usable. The public list comes next.
  let hasLiveList = false;
  let publicModels: LowRouterModel[] | undefined;
  let publicListRequested = false;
  // Pi's network phase needs a resolved key, so without one Pi never calls
  // fetchModels. The provider reads the public list itself, once, when Pi first
  // restores its models (at registration). It is not awaited: a slow or failed
  // request leaves the snapshot, and Pi's startup never waits on it.
  const loadPublicList = (baseUrl: string | undefined) => {
    if (publicListRequested || process.env.PI_OFFLINE !== undefined) {
      return;
    }
    publicListRequested = true;
    fetchModelList(baseUrl).then(
      (models) => {
        // An empty public list is a server fault, not an allowlist.
        if (models.length > 0) {
          publicModels = models;
        }
      },
      () => {},
    );
  };
  const fallbackModels = () => [...(publicModels ?? SNAPSHOT_MODELS)];
  const refreshModels = provider.refreshModels!;
  const getAllModels = provider.getAllModels!;
  return {
    ...provider,
    getModels: () => (hasLiveList ? provider.getModels() : fallbackModels()),
    getAllModels: () => (hasLiveList ? getAllModels() : fallbackModels()),
    refreshModels: (context) => {
      loadPublicList(context.baseUrl);
      return refreshModels({
        ...context,
        publish: (publication) =>
          context.publish({
            ...publication,
            update: () => {
              publication.update?.();
              hasLiveList = true;
            },
          }),
      });
    },
  };
}

export default function lowrouterExtension(pi: ExtensionAPI): void {
  pi.registerProvider(lowrouterProvider());
}
