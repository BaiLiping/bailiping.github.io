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
  function key(row) { return [row.flight, row.origin, row.destination || "", row.date].join("|"); }
  function mergeRows(oldRows, newRows) {
    var rows = new Map();
    oldRows.concat(newRows).forEach(function (row) {
      var previous = rows.get(key(row));
      // A freshly retrieved cached schedule must not erase an observed outcome.
      if (previous && ((previous.actualDeparture && !row.actualDeparture) ||
          (previous.landedAt && !row.landedAt) || (previous.cancelled && !row.cancelled && !row.actualDeparture))) return;
      if (!previous || core.instant(row.checkedAt) >= core.instant(previous.checkedAt)) rows.set(key(row), row);
    });
    return Array.from(rows.values());
  }
  function merge(base, response, query) {
    query = selection(query);
    if (!response || !response.snapshot) throw new Error("The live check returned no flight observations.");
    var next = core.validateSnapshot(response.snapshot);
    if (next.defaultFlight !== query.flight || next.defaultAirport !== query.airport || next.defaultDate !== query.date ||
        !next.records.length || next.records.length > 150 || next.records.some(function (row) { return !matches(row, query); }) ||
        (next.contexts || []).some(function (row) { return !matches(row, query) || row.date !== query.date; })) throw new Error("The live check returned a different flight or departure airport.");
    if (Date.parse(next.checkedAt) > Date.now() + 60000) throw new Error("The live check returned an invalid check time.");
    var merged = Object.assign({}, next, {records: mergeRows(base ? base.records : [], next.records)});
    // Preserve dated context and its original timestamp when a new check supplies none.
    var contexts = new Map();
    ((base && base.contexts) || []).concat(next.contexts || []).forEach(function (row) {
      var id = [row.flight, row.origin, row.date].join("|");
      var previous = contexts.get(id);
      var merged = Object.assign({}, previous || {}, row);
      ["airport","weather"].forEach(function(name) {
        var old = previous && previous[name], incoming = row[name];
        merged[name] = !incoming || (old && core.instant(old.checkedAt) > core.instant(incoming.checkedAt)) ? (old || null) : incoming;
      });
      contexts.set(id, merged);
    });
    merged.contexts = Array.from(contexts.values());
    return core.validateSnapshot(merged);
  }
  return {selection: selection, matches: matches, mergeRows: mergeRows, merge: merge};
});
