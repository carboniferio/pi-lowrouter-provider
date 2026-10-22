# LowRouter provider for Pi

Adds [LowRouter](https://lowrouter.ai) as a model provider in [Pi](https://pi.dev).

LowRouter is an OpenAI-compatible router run from the EU. An `auto/<creator>/<model>` ID can be
served by several European or global hosts. LowRouter prefers low-carbon regions and fails over
between them. Maintained by the LowRouter team.

## Install

Requires Pi `1.3.0` or newer and Node.js `22.19` or newer.

```bash
pi install npm:@lowrouter/pi-provider
```

LowRouter's models show up in Pi without a key. To send requests, get a key at
[lowrouter.ai/app/keys](https://lowrouter.ai/app/keys), then either export it or store it with
`/login` inside Pi (pick **LowRouter**):

```bash
export LOWROUTER_API_KEY=sk-lr-...
```

Pick a model:

```bash
pi --list-models lowrouter
pi --model lowrouter/auto/mistralai/mistral-large-2512
```

or `/model` in a session.

## Models

The provider reads `GET /v1/models?currency=usd&function_calling=yes&modality=chat` when Pi
starts, when the model picker refreshes and on `pi update --models`. Without a key it reads the
public list. With a key the key's catalog allowlist applies and the account's `alias/` models are
added. Pi keeps the last list for offline starts. The picker lists:

- `auto/<creator>/<model>` IDs and your account's `alias/<name>` models;
- chat models with tool calling only, since Pi acts through tools.

Pinned routes (`<host>/<creator>/<model>`) and jurisdiction-scoped IDs
(`auto/<creator>/<model>/eu`) stay out of the list. To use one, add it in `models.json` as
described in the [LowRouter guide](https://lowrouter.ai/docs/integrations/pi).

Claude models in the picker (`auto/anthropic/<model>`) go through Pi's Anthropic API on
LowRouter's `/v1/messages`, so `cache_control` breakpoints and tool calls keep Anthropic's own
format. Every other model uses `/v1/chat/completions`.

Context windows, output limits and input types come from `/v1/models`. A model reasons when its
default routes publish `reasoning_effort` levels there; Pi's thinking levels then map onto exactly
those levels, and a level the routes don't take is not offered. Other models get no effort.
A Claude model whose `reasoning_mode` is `adaptive` takes the level as an effort; one in `budget`
mode gets a thinking-token budget for it.
A model without a published context window gets 32,768 tokens, so Pi compacts early rather than
sending a prompt the model rejects. A model without an output limit gets 8,192.

Until a live or cached list is in, Pi uses the snapshot shipped with the release, and it stays
when Pi is offline (`--offline` or `PI_OFFLINE`) with nothing cached. The snapshot is
generated from the same endpoint by `npm run gen:catalog`; nothing in it is written by hand.

## Without a key

Pi lists only providers it counts as configured, so without a key the provider still reports
LowRouter as configured, with no credential. The models show in `/model` and
`pi --list-models`, but a request fails before it is sent, with a message to run `/login` or set
`LOWROUTER_API_KEY`. Two things follow from Pi counting LowRouter as configured:

- If no default model is saved and no configured provider has a Pi default model (a custom
  `models.json` provider has none), a fresh Pi can pick a LowRouter model as its default, and the
  first prompt shows that message. With no other provider at all, it replaces Pi's "No models
  available" notice.
- A session, or Pi's saved default model, on a LowRouter model restores without a key and fails
  at the first prompt, instead of switching to another provider.

## Cost

LowRouter bills in euros, for the route that served the request. Pi tracks cost in US dollars,
so the provider reads USD prices (`?currency=usd`). An `auto/` model has several routes at
different prices, and the provider shows the dearest one, so Pi's estimate is an upper bound for
input and output tokens. Cache tokens are different: `/v1/models` publishes no cache-write price
and a cache-read price for only some models, and Pi counts an unpublished price as zero. Claude
caches the prompt prefix on every turn, so its estimate can come out below the bill. What you
were charged is in each response's `usage.cost` (EUR) and on
[lowrouter.ai/app](https://lowrouter.ai/app).

## Development

```bash
npm install
npm run typecheck
npm test
npm pack --dry-run
```

Try a local checkout without installing it: `pi -e ./dist/index.js` after `npm run build`.

Live tests need a key and explicit opt-in. They run discovery, one streamed completion and one
tool call on each of six models, a thinking-level turn on three, and a two-turn prompt-cache
check on Claude Haiku 4.5:

```bash
LOWROUTER_LIVE_TEST=1 LOWROUTER_API_KEY=sk-lr-... npm run test:live
```

To release:

1. Bump `version` in `package.json` and refresh the snapshot: `npm run gen:catalog`. It reads
   the public catalog, without a key.
2. Push to `main`, then run the **Publish** workflow from the Actions tab, first with `dry_run`
   on. It stages the version on npm with provenance through trusted publishing, with no stored
   token.
3. Approve the staged version with 2FA on npmjs.com, or `npm stage approve <stage-id>`
   (`npm stage list @lowrouter/pi-provider` shows the id). Until then it is not installable.

## License

Apache-2.0
