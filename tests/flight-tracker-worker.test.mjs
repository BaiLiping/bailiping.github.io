import assert from "node:assert/strict";
import worker, { dateAtOrigin, isValidDate, shiftDate, normalizeData, loadFlightData } from "../flight-tracker/worker/worker.mjs";

assert.equal(isValidDate("2026-10-26"), true);
assert.equal(isValidDate("2026-02-31"), false);
assert.equal(shiftDate("2026-10-26", -10), "2026-10-16");
assert.equal(dateAtOrigin("2026-10-25T23:30:00Z", "Europe/Stockholm"), "2026-10-26");
assert.equal(dateAtOrigin("2026-10-26T00:30:00Z", "America/Los_Angeles"), "2026-10-25");

const historyFlight = {
  fa_flight_id: "SAS1431-historical",
  ident: "SAS1431",
  origin: { code_iata: "ARN", timezone: "Europe/Stockholm" },
  destination: { code_iata: "GOT" },
  scheduled_out: "2026-10-25T07:00:00Z",
  actual_out: "2026-10-25T07:20:00Z",
  scheduled_in: "2026-10-25T08:00:00Z",
  actual_in: "2026-10-25T08:35:00Z",
  status: "Arrived"
};
const currentFlight = {
  fa_flight_id: "SAS1431-current",
  ident: "SAS1431",
  origin: { code_iata: "ARN", timezone: "Europe/Stockholm" },
  destination: { code_iata: "GOT" },
  scheduled_out: "2026-10-26T07:00:00Z",
  estimated_out: "2026-10-26T07:15:00Z",
  scheduled_in: "2026-10-26T08:00:00Z",
  estimated_in: "2026-10-26T08:45:00Z",
  status: "Scheduled"
};
const cancelled = {
  fa_flight_id: "SAS1431-cancelled",
  ident: "SAS1431", origin: { timezone: "Europe/Stockholm" },
  scheduled_out: "2026-10-24T07:00:00Z", cancelled: true, status: "Cancelled",
  scheduled_in: "2026-10-24T08:00:00Z"
};
const now = new Date("2026-10-26T12:00:00Z");
const parsed = normalizeData("SK1431", "2026-10-26",
  [historyFlight, currentFlight, cancelled, historyFlight], now);
assert.equal(parsed.history.length, 10);
assert.equal(parsed.history[0].date, "2026-10-16");
assert.equal(parsed.history[9].date, "2026-10-25");
assert.equal(parsed.history[9].flights.length, 1, "Duplicate flight ids should be deduplicated");
assert.equal(parsed.history[9].flights[0].departureDelay, 20);
assert.equal(parsed.history[9].flights[0].arrivalDelay, 35);
assert.equal(parsed.summary.completedArrivals, 1);
assert.equal(parsed.summary.onTimeArrivalPercent, 0);
assert.equal(parsed.summary.cancellations, 1);
assert.equal(parsed.current.length, 1);
assert.equal(parsed.current[0].arrivalDelay, 45);
assert.equal(parsed.current[0].arrivalDelayType, "estimated");

const future = normalizeData("SK1431", "2026-11-07", [historyFlight], now);
assert.equal(future.history[9].date, "2026-10-25");
assert.equal(future.current.length, 0);

let requested = [];
const mockFetch = async (url, options) => {
  requested.push(new URL(url));
  assert.equal(options.headers["x-apikey"], "TEST_KEY");
  if (url.includes("/history/flights/")) return new Response(JSON.stringify({
    flights: [historyFlight, cancelled]
  }), { status: 200 });
  if (url.includes("/flights/")) return new Response(JSON.stringify({
    flights: [currentFlight]
  }), { status: 200 });
  throw new Error("Unexpected URL " + url);
};
const live = await loadFlightData("SK1431", "2026-10-26", "TEST_KEY", {
  fetchImpl: mockFetch, now
});
assert.equal(live.current.length, 1);
assert.equal(live.summary.historyFlights, 2);
assert.ok(requested.length >= 3);
assert.ok(requested.some(url => url.pathname.includes("/history/flights/")));
assert.ok(requested.some(url => url.pathname.includes("/flights/SK1431")));
for (const url of requested.filter(url => url.pathname.includes("/history/"))) {
  const from = new Date(url.searchParams.get("start"));
  const to = new Date(url.searchParams.get("end"));
  assert.ok(to - from <= 7 * 86400000, "History queries must not exceed 7 days");
}

const partial = await loadFlightData("SK1431", "2026-10-26", "TEST_KEY", {
  now,
  fetchImpl: async url => url.includes("/history/flights/")
    ? new Response("{}", { status: 403 })
    : new Response(JSON.stringify({ flights: [currentFlight] }), { status: 200 })
});
assert.ok(partial.historyError);
assert.equal(partial.current.length, 1, "Still show current status if history is unavailable");

const options = await worker.fetch(new Request("https://flight.test/api/flight", {
  method: "OPTIONS", headers: { Origin: "https://bailiping.com" }
}), {});
assert.equal(options.status, 204);
assert.equal(options.headers.get("Access-Control-Allow-Origin"), "https://bailiping.com");
assert.equal(await options.text(), "");
const invalid = await worker.fetch(new Request("https://flight.test/api/flight?flight=SK<script>&date=2026-10-26"), {});
assert.equal(invalid.status, 400);
for (const invalidCode of ["12345", "1A", "A1", "12A123", "U2A123", "ABCD123"]) {
  const result = await worker.fetch(new Request("https://flight.test/api/flight?flight=" + invalidCode + "&date=2026-10-26"), {});
  assert.equal(result.status, 400, "Reject malformed airline designator: " + invalidCode);
}
for (const validCode of ["BA247", "SK1431", "UAL123", "U21234", "6E123", "BA123A"]) {
  const result = await worker.fetch(new Request("https://flight.test/api/flight?flight=" + validCode + "&date=2026-10-26"), {});
  assert.equal(result.status, 503, "Validate code before detecting missing secret: " + validCode);
}

const missingKey = await worker.fetch(new Request("https://flight.test/api/flight?flight=SK1431&date=2026-10-26"), {});
assert.equal(missingKey.status, 503);

console.log("FlightAware data adapter tests passed");
