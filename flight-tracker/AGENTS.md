# Updating flight observations

This page displays a published JSON snapshot. When asked to update flight data,
check the source websites with Codex and edit `observations.json`. Keep this
workflow free of flight APIs, API keys, backend workers, and automated scraping.

1. Confirm the flight number, departure airport, and local departure date.
   The existing dataset uses Flightradar24. Open the matching source pages and
   read the actual records; never infer current status from an old snapshot.
2. Update or append matching entries in `records`, preserving other flights and
   dates. Each entry needs `flight`, known `aliases`, `date`, `origin`, known
   `originAliases`, `destination`, airport names and IANA timezones, a status,
   `sourceUrl`, and the actual `checkedAt` time. Copy a nearby record for shape.
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
5. Set top-level `checkedAt` to the latest flight-record check and
   `observationDate` to that check's local date at the default origin. Keep
   `defaultFlight`, `defaultAirport`, and `defaultDate` valid. The page starts on
   today's Copenhagen date unless the URL specifies a date. Never advance an
   individual record's check time unless that record was checked again.
6. Run `node tests/flight-tracker.test.cjs` from the repository root. The tests
   validate the current JSON and exercise lookup behavior with separate fixed
   fixtures, so routine data updates must not require changing expected results.
   Serve the site and check the selected date, historical chart, source links,
   collapsed history details, and mobile layout. Verify that the page requests
   only local assets and `observations.json` until a source link is clicked.
7. Follow the requested publication scope and verify the live JSON after
   publishing. Describe exactly which flights and dates were checked and any
   unavailable information. A successful local edit alone is not publication.

The frontend accepts Flightradar24 source links. If another source is needed,
update the source labeling and link validation deliberately rather than labeling
that source's data as Flightradar24.
