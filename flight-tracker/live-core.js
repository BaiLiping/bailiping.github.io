(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory(require("./tracker-core.js"));
  else root.FlightTrackerLive = factory(root.FlightTracker);
})(typeof globalThis !== "undefined" ? globalThis : this, function (core) {
  "use strict";
  function selection(value) {
    var result = {flight: core.normalFlight(value.flight), airport: core.normalAirport(value.airport), date: value.date};
    if (!core.validFlight(result.flight) || !core.validAirport(result.airport) || !core.validDate(result.date)) throw new Error("Enter a valid flight, departure airport, and date.");
    return result;
  }
  function matches(row, query) {
    return (row.flight === query.flight || (row.aliases || []).includes(query.flight)) &&
      (row.origin === query.airport || (row.originAliases || []).includes(query.airport));
  }
  function validateResponse(response, query) {
    query = selection(query);
    if (!response || !response.snapshot) throw new Error("The flight lookup returned no records.");
    var next = core.validateSnapshot(response.snapshot);
    if (!next.records.length) throw new Error("No accessible Flightradar24 records were returned for this search.");
    if (next.defaultFlight !== query.flight || next.defaultAirport !== query.airport || next.defaultDate !== query.date ||
        next.records.length > 40 || next.records.some(function (row) { return !matches(row, query); }) ||
        (next.contexts || []).some(function (row) { return !matches(row, query) || row.date !== query.date; })) throw new Error("The flight lookup returned a different flight, departure airport, or date.");
    if (Date.parse(next.checkedAt) > Date.now() + 60000) throw new Error("The flight lookup returned an invalid check time.");
    var dates = new Set(core.select(next,query.flight,query.airport,query.date).history.map(function(day) { return day.date; }));
    dates.add(query.date);
    if (next.records.some(function(row) { return !dates.has(row.date); })) throw new Error("The flight lookup returned records outside the selected seven-day history window.");
    return next;
  }
  return {selection: selection, matches: matches, validateResponse: validateResponse};
});
