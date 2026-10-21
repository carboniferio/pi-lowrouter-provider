/**
 * LowRouter provider for Pi.
 *
 * Registers `lowrouter` with the models of the release-time snapshot, then
 * replaces them with the live `/v1/models` list once a key is configured
 * (`LOWROUTER_API_KEY` or `/login`). Pi keeps the last live list for offline
 * starts.
 */
// Pi maps a fixed set of host modules into extensions; `compat` is the mapped
// entry that also exports the API implementations.
import { createProvider, envApiKeyAuth, openAICompletionsApi, type Provider } from "@earendil-works/pi-ai/compat";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { LOWROUTER_CATALOG } from "./catalog.generated.js";
import {
  fetchLowRouterModelRows,
  LOWROUTER_BASE_URL,
  LOWROUTER_PROVIDER_ID,
  projectLowRouterModelRows,
  toLowRouterModel,
} from "./models.js";

const SNAPSHOT_MODELS = LOWROUTER_CATALOG.map((row) => toLowRouterModel(row));

/** Builds the LowRouter provider. */
export function lowrouterProvider(): Provider<"openai-completions"> {
  const provider = createProvider<"openai-completions">({
    id: LOWROUTER_PROVIDER_ID,
    name: "LowRouter",
    baseUrl: LOWROUTER_BASE_URL,
    auth: { apiKey: envApiKeyAuth("LowRouter API key", ["LOWROUTER_API_KEY"]) },
    models: [],
    fetchModels: async (context) => {
      // Pi refreshes only with a resolved key. Throwing keeps the cached list;
      // returning [] would persist an empty one over it.
      if (context.credential?.type !== "api_key") {
        throw new Error("LowRouter model refresh needs an API key");
      }
      const baseUrl = (context.baseUrl || LOWROUTER_BASE_URL).replace(/\/+$/, "");
      const rows = await fetchLowRouterModelRows({ baseUrl, apiKey: context.credential.key, signal: context.signal });
      return projectLowRouterModelRows(rows, LOWROUTER_CATALOG).map((row) => toLowRouterModel(row, baseUrl));
    },
    api: openAICompletionsApi(),
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
