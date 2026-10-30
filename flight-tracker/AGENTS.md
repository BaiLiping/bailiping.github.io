# Flight tracker

On 30 October 2026, the user requested input-driven API lookups only, removal of
saved observations and their banner/controls, and a Previous 7 days history
window. This supersedes earlier instructions to publish observations.json or
change the bare-page default flight/date. Do not restore that dataset or workflow.

## User flow

- The bare page opens with empty flight, airport, and date fields and hidden
  results. Shared URL parameters may prefill fields, but never trigger a lookup.
- Submitting the form (button or Enter) requests new data from the configured API.
  Clear previous results on input changes and before a new lookup. A failed or
  empty lookup must not fall back to data from an earlier search.
- The frontend does not read a snapshot file, local storage, or saved defaults.
  The backend does not cache or archive flight results. It retains only usage
  counters and the marker for removal of its former cache/archive. Simultaneous
  identical requests may share one in-flight API call.

## Source and data rules

1. Flightradar24 is the sole flight-data source. OpenAI searches and extracts its
   public pages through worker/api.mjs. worker/source-context.mjs additionally
   reads public airport departure/weather pages for timestamped context and a
   current estimate matched to the exact flight, route and scheduled departure.
   That estimate retains its airport-page source link and report time. Do not
   invent records or use another provider. Gold-only data is not available
   through web search; the backend does not use subscriber credentials.
2. Include the selected local departure date and the preceding seven dates.
   For future dates, history ends the day before the source check's local date.
   Keep different departure airports and routes separate, and report missing
   dates explicitly. Missing data is not cancellation or on-time operation.
3. Require retrieved flight/aircraft source URLs, source evidence, check times,
   confirmed IANA timezones, and explicit ISO UTC offsets. Convert UTC source
   tables correctly across midnight and daylight-saving changes. Keep estimates
   separate from actual events, and use null for unavailable fields. Provider
   ATD and landing times are not confirmed gate events.
4. Airport/weather context needs retrieved airport evidence for the same origin
   and date, with a valid observation or forecast time. Omit unsupported context.
   Observations must precede the scheduled departure. Current airport totals
   cannot reconstruct earlier conditions. METARs are observations, not forecasts.
   Do not invent a delay probability from a small sample.
5. Keep API keys in ignored env files and Cloudflare secrets, never public files,
   Git, or logs. Preserve the global daily and per-visitor API limits.

## Validation and publication

Pull first after inspecting the working tree. Run:

```sh
node tests/flight-tracker.test.cjs
node --test tests/flight-live-api.test.mjs
python3 scripts/build-search-index.py --check
git diff --check
```

Check desktop and mobile: empty initial form, no automatic data request, URL
prefill without fetching, button and Enter submission, seven history dates,
source links, missing records, changed selections, failure without stale results,
and no console errors or horizontal overflow. Use fixtures for failure cases;
verify a real production lookup separately. Backend changes require Worker
deployment; frontend changes require GitHub Pages publication and live-file
verification. See worker/README.md for deployment commands.
