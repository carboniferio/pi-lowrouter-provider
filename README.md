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

Get a key at [lowrouter.ai/app/keys](https://lowrouter.ai/app/keys), then either export it or
store it with `/login` inside Pi (pick **LowRouter**):

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

With a key, the provider reads `GET /v1/models?currency=usd&function_calling=yes&modality=chat`
when Pi starts, when the model picker refreshes and on `pi update --models`. Pi keeps the last
list for offline starts. The picker lists:

- `auto/<creator>/<model>` IDs and your account's `alias/<name>` models;
- chat models with tool calling only, since Pi acts through tools.

Pinned routes (`<host>/<creator>/<model>`) and jurisdiction-scoped IDs
(`auto/<creator>/<model>/eu`) stay out of the list. To use one, add it in `models.json` as
described in the [LowRouter guide](https://lowrouter.ai/docs/integrations/pi).

Context windows, output limits, input types and reasoning support come from `/v1/models`.
A model without a published context window gets 32,768 tokens, so Pi compacts early rather than
sending a prompt the model rejects. A model without an output limit gets 8,192.

Before the first refresh, Pi uses the snapshot shipped with the release. The snapshot is
generated from the same endpoint by `npm run gen:catalog`; nothing in it is written by hand.

## Cost

LowRouter bills in euros, for the route that served the request. Pi tracks cost in US dollars,
so the provider reads USD prices (`?currency=usd`). An `auto/` model has several routes at
different prices, and the provider shows the dearest one, so Pi's estimate is an upper bound.
What you were charged is in each response's `usage.cost` (EUR) and on
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
tool call on each of five models:

```bash
LOWROUTER_LIVE_TEST=1 LOWROUTER_API_KEY=sk-lr-... npm run test:live
```

To release:

1. Bump `version` in `package.json` and refresh the snapshot: `npm run gen:catalog`. It reads
   the public catalog, without a key.
2. Push to `main`, then run the **Publish** workflow from the Actions tab, first with `dry_run`
   on. It publishes to npm with provenance through trusted publishing, with no stored token.

## License

Apache-2.0
