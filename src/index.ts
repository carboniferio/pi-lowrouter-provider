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

export * from "./models.js";

/** Builds the LowRouter provider. */
export function lowrouterProvider(): Provider<"openai-completions"> {
  return createProvider<"openai-completions">({
    id: LOWROUTER_PROVIDER_ID,
    name: "LowRouter",
    baseUrl: LOWROUTER_BASE_URL,
    auth: { apiKey: envApiKeyAuth("LowRouter API key", ["LOWROUTER_API_KEY"]) },
    models: LOWROUTER_CATALOG.map((row) => toLowRouterModel(row)),
    fetchModels: async (context) => {
      const apiKey = context.credential?.type === "api_key" ? context.credential.key : undefined;
      if (!apiKey) {
        return [];
      }
      const baseUrl = context.baseUrl ?? LOWROUTER_BASE_URL;
      const rows = await fetchLowRouterModelRows({ apiKey, baseUrl, signal: context.signal });
      return projectLowRouterModelRows(rows, LOWROUTER_CATALOG).map((row) => toLowRouterModel(row, baseUrl));
    },
    api: openAICompletionsApi(),
  });
}

export default function lowrouterExtension(pi: ExtensionAPI): void {
  pi.registerProvider(lowrouterProvider());
}
