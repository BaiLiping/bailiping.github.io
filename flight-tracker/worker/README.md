# Flight tracker live checks

The webpage calls this Cloudflare Worker to obtain source-backed flight records
from public Flightradar24 pages through OpenAI Responses web search. Every
submitted search fetches new data for the selected flight, departure airport,
and local date, including the previous seven days.

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
  The page starts empty; shared URL parameters only prefill the form.
- The global Durable Object enforces at most 10 new checks per UTC day and three
  per visitor per hour. Each request permits at most six web-tool calls and 10,000
  output tokens. Failed upstream requests still count toward the daily allowance.
  Visitor addresses are hashed with the current day and kept only in that day's
  counter, rather than stored as raw addresses.
- Results are not cached or archived. Every completed search is followed by a new
  API lookup on the next submission. Only simultaneous identical requests share
  an in-flight call. The retired cache/archive is removed on initialization,
  without resetting usage counters.
- The frontend replaces the entire result with the response to the submitted
  inputs. Input edits, errors, and empty lookups clear previous results; there is
  no saved dataset or fallback. No flight/date is selected by default.

## Verification

From the repository root:

```sh
node --test tests/flight-live-api.test.mjs
node tests/flight-tracker.test.cjs
python3 scripts/build-search-index.py --check
git diff --check
```

Bundle check, from this directory:

```sh
npx --yes wrangler@4.156.0 deploy --dry-run
```

Exercise the browser at desktop/mobile widths: empty form, URL prefill without
fetching, button and Enter submission, seven-day history, partial results,
unavailable selected date, input changes, and failure without stale records.
Initial page load must not call OpenAI or load a dataset. Verify a real production
lookup separately from mocked browser cases.
