# Updating flight observations

This page displays a published JSON snapshot. The user requires Flightradar24
as the data source and explicitly requested a script that searches its historical
data. Use `check-flight.cjs` for future flight checks. It calls Codex's web tools
to search and open public Flightradar24 pages, then uses
`update-observations.cjs` to validate and update `observations.json`.
Keep this workflow free of flight APIs, API keys, and backend workers.
The scripts run during authoring; they are not loaded by the visitor's browser.

## Quick update with Codex

Pull first after checking the working tree. In Codex's JavaScript tool runner,
load this trusted repository script and pass the available tools to it:

```js
const repo = "/home/bailiping/Desktop/bailiping.github.io";
const source = await tools.exec_command({
  cmd: "cat flight-tracker/check-flight.cjs", workdir: repo,
  max_output_tokens: 12000
});
if (source.exit_code !== 0) throw new Error(source.output);
const module = {exports: {}};
new Function("module", "exports", source.output)(module, module.exports);
const result = await module.exports.checkFlight(tools, {
  repo, flight: "UA1219", airport: "RNO", destination: "SFO",
  date: "2026-10-19", dryRun: true
});
text(result);
```

Replace the flight, origin, destination, and local date with the user's request.
For airports not already in the snapshot, also supply confirmed IANA
`originTimeZone` and `destinationTimeZone` values. The script opens the public
flight history, searches the preceding ten local dates, opens discovered
Flightradar24 aircraft history when useful, and filters the requested route.
Review `retrievedDates`, `missingHistoryDates`, `skipped`, source links, and the
preview. Run its returned `applyCommand` in the repository to write that exact
batch without repeating the search. `dryRun: false` applies directly when the
selection and source evidence are already established.

A plain HTTP download can be blocked by Flightradar24. This workflow deliberately
uses Codex's web-search/open tools; it is not a standalone Node web client.
It does not bypass sign-in or subscription access. If no history is retrieved,
inspect the source before using `allowNoHistory: true`; do not silently publish
an empty history panel while accessible records exist.

## Data and publication rules

1. Confirm the flight number, departure airport, and local departure date.
   Use Flightradar24 only. Open the matching source pages and
   read the actual records; never infer current status from an old snapshot.
2. Include the accessible history in the page's previous-ten-day window, not
   only the selected day's record. Report unavailable dates explicitly; missing
   data is not evidence that a flight was cancelled or on time. The history
   window ends the day before the earlier of the selected date and the source
   check's local date. Keep other routes using the same flight number separate.
   Update or append matching entries in `records`, preserving other flights and
   dates. Each entry needs `flight`, known `aliases`, `date`, `origin`, known
   `originAliases`, `destination`, airport names and IANA timezones, a status,
   `sourceUrl`, and the actual `checkedAt` time. Use the updater's `--template`
   and `--input` modes for a manually reviewed correction when needed.
   If a source is unavailable, keep the old observation and its old check time.
3. Use ISO timestamps with an explicit UTC offset or `Z`. The record's `date`
   must match `scheduledDeparture` in `originTimeZone`; use the event date's
   daylight-saving offset. Use `null` for unknown times. Keep estimates separate
   from actual departure and landing times. Keep `departureEvent: "provider-atd"`
   unless the source explicitly confirms a gate departure. A gate delay is any
   positive difference from schedule; provider ATD and landing comparisons are
   not verified gate-to-gate delays.
4. Add airport and weather evidence to matching dated `contexts`, with a source
   URL and check time for each item. Samples must belong to that local date and
   origin. Clearly distinguish observations from forecasts and retrospective
   information from information available before departure. Do not invent a
   calibrated delay probability from a small sample.
5. Whenever the user asks to check a flight on a date, set `defaultFlight`,
   `defaultAirport`, and `defaultDate` to that requested flight, origin, and
   departure date in the same update. This is the user's standing preference:
   `/flight-tracker/` must open on the most recently requested selection. Use
   a supported flight alias if needed. Explicit URL parameters override the
   saved defaults; do not replace the requested date with today's date. The
   default hint is rendered from these fields, so no HTML edit is needed for
   each lookup. Set top-level `checkedAt` to the latest flight-record check and
   `observationDate` to that check's local date at the default origin. A change
   to defaults alone must not advance any observation's check time.
6. Run `node --test tests/flight-update.test.cjs tests/flight-history-search.test.cjs`
   after changing the scripts, and `node tests/flight-tracker.test.cjs` after
   every data update. The tests
   validate the current JSON and exercise lookup behavior with separate fixed
   fixtures, so routine data updates must not require changing expected results.
   Serve the site and check the bare `/flight-tracker/` URL opens on the new
   defaults, including the requested date, and that explicit URL parameters
   still work. Check the historical chart, source links, collapsed history
   details, and mobile layout. Verify that the page requests
   only local assets and `observations.json` until a source link is clicked.
7. Follow the requested publication scope and verify the live JSON after
   publishing. Describe exactly which flights and dates were checked and any
   unavailable information. A successful local edit alone is not publication.

The frontend accepts only the explicitly listed Flightradar24 website hosts.
Preserve the actual source URL, including an aircraft history page when used.
Do not substitute another provider unless the user changes this instruction.
