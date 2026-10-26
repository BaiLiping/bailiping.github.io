// Browser logic regression tests for the embedded flight tracker (no paid API needed).
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const root = path.join(__dirname, "..", "flight-tracker");
const html = fs.readFileSync(path.join(root, "index.html"), "utf8");
const script = fs.readFileSync(path.join(root, "tracker.js"), "utf8");
const styles = fs.readFileSync(path.join(root, "styles.css"), "utf8");

assert.match(html, /01 \/ Historical performance/);
assert.match(html, /02 \/ Current status/);
assert.match(html, /id="history-rows"/);
assert.match(html, /id="current-flights"/);
assert.match(html, /src="tracker.js"/);
assert.match(html, /src="config.js"/);
assert.match(styles, /@media/);
assert.doesNotMatch(html, /https:\/\/(?:www\.)?(?:flightradar24|flightaware)\.com\//, "Results should render on-page, not in an outbound link");

class FixedDate extends Date {
  constructor(...args) { super(...(args.length ? args : ["2026-10-26T13:00:00Z"])); }
  static now() { return new Date("2026-10-26T13:00:00Z").getTime(); }
}

function fixtures() {
  const ids = ["flight-form", "flight-number", "flight-date", "form-error", "results",
    "search-button", "result-flight", "result-date", "refresh", "fetch-status",
    "history-period", "history-metrics", "history-chart", "history-rows",
    "current-period", "current-flights", "provider-name", "provider-updated"];
  const map = new Map();
  for (const id of ids) {
    map.set(id, {
      textContent: "", innerHTML: "", value: "", disabled: false, hidden: id === "results" || id === "form-error",
      events: {}, setAttribute(k, v) { this[k] = v; },
      focus() { this.focused = true; },
      addEventListener(ev, fn) { this.events[ev] = fn; }
    });
  }
  const loc = { href: "https://bailiping.com/flight-tracker/", search: "" };
  let urlChanged = "";
  let requests = [];
  let mockResponse = null;
  const context = {
    Date: FixedDate, String, Number, Array, Intl, URL, URLSearchParams, AbortController,
    location: loc, history: { replaceState(_a, _b, value) { urlChanged = value; } },
    document: { getElementById(id) { assert.ok(map.has(id), "Unrecognized element: " + id); return map.get(id); } },
    window: { FLIGHT_TRACKER_API_URL: "" },
    fetch: async (url, opt) => {
      requests.push({ url, opt });
      return { ok: true, async json() { return mockResponse; } };
    }
  };
  vm.runInNewContext(script, context, { filename: "tracker.js" });
  return {
    byId(id) { return map.get(id); },
    submit() { map.get("flight-form").events.submit({ preventDefault() {} }); },
    refresh() { map.get("refresh").events.click(); },
    setEndpoint(url) { context.window.FLIGHT_TRACKER_API_URL = url; },
    setResponse(value) { mockResponse = value; },
    requests, get urlChanged() { return urlChanged; }
  };
}

function makeData() {
  const dayRecord = {
    flight: "SK1431", status: "Arrived", origin: "ARN", destination: "GOT",
    scheduledOut: "2026-10-25T07:00:00Z", actualOut: "2026-10-25T07:35:00Z",
    scheduledIn: "2026-10-25T08:00:00Z", actualIn: "2026-10-25T08:25:00Z",
    departureDelay: 35, departureDelayType: "actual", arrivalDelay: 25,
    arrivalDelayType: "actual"
  };
  const currentRecord = {
    flight: "SK1431", status: "En Route", origin: "ARN", destination: "GOT",
    scheduledOut: "2026-10-26T07:00:00Z", estimatedOut: "2026-10-26T07:20:00Z",
    scheduledIn: "2026-10-26T08:00:00Z", estimatedIn: "2026-10-26T08:40:00Z",
    departureDelay: 20, departureDelayType: "estimated", arrivalDelay: 40,
    arrivalDelayType: "estimated"
  };
  const history = [];
  for (let i = 16; i <= 25; i++) {
    history.push({ date: "2026-10-" + i, flights: i === 25 ? [dayRecord] : [] });
  }
  return {
    flight: "SK1431", selectedDate: "2026-10-26", history, current: [currentRecord],
    asOf: "2026-10-26T13:00:00Z", source: "FlightAware AeroAPI"
  };
}

async function nextMicrotasks() {
  await new Promise(resolve => setImmediate(resolve));
}

(async function () {
  const f = fixtures();
  f.byId("flight-number").value = " SK 1431 ";
  f.byId("flight-date").value = "2026-10-26";
  f.submit();
  await nextMicrotasks();
  assert.equal(f.byId("flight-number").value, "SK1431");
  assert.equal(f.byId("results").hidden, false);
  assert.match(f.byId("history-rows").innerHTML, /Not available/);
  assert.match(f.byId("current-flights").innerHTML, /Status unavailable/);
  assert.match(f.byId("history-metrics").innerHTML, /No verified records/);
  assert.equal(f.requests.length, 0);
  assert.match(f.urlChanged, /flight=SK1431/);

  f.setEndpoint("https://flight-api.example.workers.dev");
  f.setResponse(makeData());
  f.submit();
  await nextMicrotasks();
  assert.equal(f.requests.length, 1);
  assert.match(f.requests[0].url, /flight=SK1431/);
  assert.match(f.requests[0].url, /date=2026-10-26/);
  assert.match(f.byId("history-chart").innerHTML, /\+25 min/);
  assert.match(f.byId("history-metrics").innerHTML, /0%/, "Arrival 25 minutes late should count as not on time");
  assert.match(f.byId("history-rows").innerHTML, /\+35 min/);
  assert.match(f.byId("current-flights").innerHTML, /En Route/);
  assert.match(f.byId("current-flights").innerHTML, /\+40 min \(est\.\)/);
  assert.match(f.byId("provider-name").textContent, /FlightAware/);

  f.setResponse({ ...makeData(), historyError: "History requires Standard tier" });
  f.refresh();
  await nextMicrotasks();
  assert.match(f.byId("history-chart").innerHTML, /History requires Standard tier/);
  assert.match(f.byId("history-metrics").innerHTML, /History access unavailable/);
  assert.match(f.byId("current-flights").innerHTML, /En Route/, "Current info remains visible");

  f.setResponse({ ...makeData(), current: [{ ...makeData().current[0], status: '<img src=x onerror=alert(1)>' }] });
  f.refresh();
  await nextMicrotasks();
  assert.doesNotMatch(f.byId("current-flights").innerHTML, /<img/);
  assert.match(f.byId("current-flights").innerHTML, /&lt;img/);

  f.byId("flight-number").value = "<script>";
  f.submit();
  await nextMicrotasks();
  assert.equal(f.byId("results").hidden, true);
  assert.equal(f.byId("form-error").hidden, false);
  assert.equal(f.requests.length, 3, "Invalid input must not make a request");

  f.byId("flight-number").value = "SK1431";
  f.byId("flight-date").value = "2026-02-31";
  f.submit();
  await nextMicrotasks();
  assert.match(f.byId("form-error").textContent, /valid departure date/);

  console.log("Embedded flight tracker UI tests passed");
})().catch(error => { console.error(error); process.exitCode = 1; });
