# Flight tracker live checks

The webpage calls this Cloudflare Worker to obtain source-backed flight records
from public Flightradar24 pages through OpenAI Responses web search. The existing
Codex authoring scripts and published `observations.json` remain available.

The OpenAI API key belongs to the user's Personal / Default project and is kept
in the ignored repository-root `.env.local` and the deployed `OPENAI_API_KEY`
secret. Never print it, put it in `api-config.json`, or embed it in frontend code.

## Deploy

From this directory, with the user's Cloudflare account authenticated:

```sh
npx --yes wrangler@4.156.0 login
npx --yes wrangler@4.156.0 deploy --secrets-file ../../.env.local
```

Set `flight-tracker/api-config.json` to the resulting HTTPS Worker origin. Publish
the static page through the existing GitHub Pages workflow and verify both the
Worker and the webpage. An empty endpoint leaves the live-check button disabled.

The Worker and SQLite Durable Object use resources supported by the Workers Free
plan. API requests consume the OpenAI project's available API credits. The Worker
does not enable paid billing or automatic credit purchases.

## Behavior and bounds

- `POST /check` accepts only a flight number, departure airport, and local date.
  `GET /health` reports readiness without querying OpenAI. Browser origins must
  match `ALLOWED_ORIGINS`; the public UI receives no API key.
- Searches are restricted to `flightradar24.com`. Each flight row must cite an
  actual retrieved flight/aircraft data page. Date, route, timezone, actual versus
  estimated events, duplicates, and impossible timing are validated. Evidence
  quotes are retained with each row for review.
- Airport and weather context additionally require a retrieved airport data page,
  matching origin/date, and a valid observation/forecast time. Unsupported context
  is dropped without losing valid flight records. No delay probability is invented.
- Public history may be incomplete; Gold subscription credentials are not used.
  A result can contain history without a selected-day record, and the UI says so.
- The global Durable Object enforces at most 10 new checks per UTC day and three
  per visitor per hour. Each request permits at most six web-tool calls and 10,000
  output tokens. Failed upstream requests still count toward the daily allowance.
  Visitor addresses are hashed with the current day and kept only in that day's
  counter, rather than stored as raw addresses.
- Successful identical selections share a 15-minute cache. Check times do not
  advance when cached data is reused. Up to 150 prior records per flight/origin
  are retained to extend history across checks, with their original check times.
- API results update the current view and shared backend history. They do not
  commit to GitHub or change the public default flight/date for other visitors.
  Codex publication remains the way to change the bare-page default.

## Verification

From the repository root:

```sh
node --test tests/flight-live-api.test.mjs tests/flight-history-search.test.cjs tests/flight-update.test.cjs
node tests/flight-tracker.test.cjs
python3 scripts/build-search-index.py --check
git diff --check
```

Bundle check, from this directory:

```sh
npx --yes wrangler@4.156.0 deploy --dry-run
```

Exercise the browser at desktop/mobile widths: saved loading, live success,
partial history, unavailable selected date, error preservation, invalid inputs,
and reloading the published snapshot. Initial page load must not call OpenAI.
Verify real production checks separately from mocked browser cases.
