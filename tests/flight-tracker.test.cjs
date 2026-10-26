// Regression checks for the static flight-tracker lookup.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const html = fs.readFileSync(path.join(__dirname, "..", "flight-tracker", "index.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/);
assert.ok(script, "Tracker should include lookup script");

const listeners = {};
const nodes = new Map();
for (const id of [
  "lookup-form", "flight-number", "flight-date", "lookup-error", "results",
  "result-flight", "result-date", "date-reminder", "fr24-link",
  "fa-live-link", "fa-history-link", "results-title"
]) {
  nodes.set(id, {
    value: "",
    textContent: "",
    href: "",
    hidden: id === "results" || id === "lookup-error",
    focus() { this.focused = true; },
    addEventListener(event, fn) { listeners[event] = fn; }
  });
}

let replacedUrl = "";
const location = {
  href: "https://bailiping.com/flight-tracker/?flight=BA247&date=2026-10-26",
  search: "?flight=BA247&date=2026-10-26"
};
const context = {
  document: { getElementById(id) { assert.ok(nodes.has(id), "Unknown element: " + id); return nodes.get(id); } },
  window: { location, history: { replaceState(_a, _b, url) { replacedUrl = url; } } },
  Date, String, Number, Intl, URL, URLSearchParams
};
vm.runInNewContext(script[1], context, { filename: "flight-tracker-inline.js" });

const get = (id) => nodes.get(id);
assert.equal(get("results").hidden, false, "URL query should display flight lookup");
assert.equal(get("result-flight").textContent, "BA247");
assert.equal(get("date-reminder").textContent, "2026-10-26");
assert.equal(get("fr24-link").href, "https://www.flightradar24.com/data/flights/ba247");
assert.equal(get("fa-live-link").href, "https://www.flightaware.com/live/flight/BA247");
assert.equal(get("fa-history-link").href, "https://www.flightaware.com/live/flight/BA247/history");
assert.match(get("result-date").textContent, /26 October 2026/);
assert.ok(get("flight-date").value, "Date should default or load from query");

const submit = () => listeners.submit({ preventDefault() {} });
get("flight-number").value = " SK 1431 ";
get("flight-date").value = "2026-10-27";
submit();
assert.equal(get("flight-number").value, "SK1431", "Flight code should be normalized");
assert.equal(get("results").hidden, false);
assert.match(replacedUrl, /flight=SK1431/);
assert.match(replacedUrl, /date=2026-10-27/);
assert.equal(get("fr24-link").href, "https://www.flightradar24.com/data/flights/sk1431");

for (const flight of ["BA247", "SK1431", "UAL123", "U21234", "6E123", "BA123A"]) {
  get("flight-number").value = flight;
  submit();
  assert.equal(get("results").hidden, false, "Valid airline designator should be accepted: " + flight);
}

for (const flight of ["12345", "1A", "A1", "12A123", "U2A123", "ABCD123"]) {
  get("flight-number").value = flight;
  submit();
  assert.equal(get("results").hidden, true, "Invalid airline designator should be rejected: " + flight);
  assert.equal(get("lookup-error").hidden, false);
}

get("flight-number").value = "<script>alert(1)</script>";
submit();
assert.equal(get("results").hidden, true, "Invalid flight codes cannot create links");
assert.equal(get("lookup-error").hidden, false);
assert.ok(get("flight-number").focused);

get("flight-number").value = "BA247";
get("flight-date").value = "2026-02-31";
submit();
assert.equal(get("results").hidden, true, "Invalid calendar dates should be rejected");
assert.ok(get("flight-date").focused);

console.log("Flight-tracker lookup tests passed");
