(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.FlightTracker = factory();
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";
  function normalFlight(value) { return String(value || "").toUpperCase().replace(/[\s-]+/g, ""); }
  function validFlight(value) { return /^(?:[A-Z]{3}|[A-Z][A-Z0-9]|[0-9][A-Z])[0-9]{1,5}[A-Z]?$/.test(value); }
  function normalAirport(value) { return String(value || "").trim().toUpperCase(); }
  function validAirport(value) { return /^[A-Z]{3,4}$/.test(value); }
  function validDate(value) {
    if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
    var date = new Date(value + "T12:00:00Z");
    return Number.isFinite(date.getTime()) && date.toISOString().slice(0, 10) === value;
  }
  function instant(value) {
    return typeof value === "string" && /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})$/.test(value) && Number.isFinite(Date.parse(value)) ? Date.parse(value) : null;
  }
  function difference(scheduled, actual) {
    var a = instant(actual), s = instant(scheduled);
    return a === null || s === null ? null : (a - s) / 60000;
  }
  function gateDelay(row) {
    if (row.cancelled || row.departureEvent !== "gate") return null;
    return difference(row.scheduledDeparture, row.actualDeparture);
  }
  function shiftDay(day, by) {
    var d = new Date(day + "T12:00:00Z"); d.setUTCDate(d.getUTCDate() + by);
    return d.toISOString().slice(0, 10);
  }
  function matches(row, flight, airport) {
    return row && (row.flight === flight || (row.aliases || []).includes(flight)) &&
      (row.origin === airport || (row.originAliases || []).includes(airport));
  }
  function select(data, flight, airport, date) {
    var unique = new Map();
    (data.records || []).filter(function (r) { return matches(r, flight, airport) && validDate(r.date) && instant(r.checkedAt) !== null; }).forEach(function (r) {
      var key = [r.flight, r.origin, r.destination, r.scheduledDeparture].join("|");
      var previous = unique.get(key);
      if (!previous || instant(r.checkedAt) > instant(previous.checkedAt)) unique.set(key, r);
    });
    var records = Array.from(unique.values());
    var datasetDay = validDate(data.observationDate) ? data.observationDate : data.defaultDate;
    var anchor = date > datasetDay ? datasetDay : date;
    var history = [];
    for (var i = 10; i >= 1; i--) {
      var day = shiftDay(anchor, -i);
      history.push({ date: day, flights: records.filter(function (r) { return r.date === day; }) });
    }
    return { flight: flight, airport: airport, date: date, history: history,
      current: records.filter(function (r) { return r.date === date; }),
      context: (data.contexts || []).find(function (r) { return matches(r, flight, airport) && r.date === date; }) || null };
  }
  function stats(rows) {
    var actual = rows.filter(function (r) { return !r.cancelled; });
    var departures = actual.map(function (r) { return difference(r.scheduledDeparture, r.actualDeparture); }).filter(function (v) { return v !== null; });
    var arrivals = actual.map(function (r) { return difference(r.scheduledArrival, r.landedAt); }).filter(function (v) { return v !== null; });
    var gates = actual.map(gateDelay).filter(function (v) { return v !== null; });
    function mean(v) { return v.length ? v.reduce(function (a, b) { return a + b; }, 0) / v.length : null; }
    return { count: rows.length, departureCount: departures.length, arrivalCount: arrivals.length,
      meanDeparture: mean(departures), meanArrival: mean(arrivals), gateCount: gates.length,
      delayedGateCount: gates.filter(function (v) { return v > 0; }).length };
  }
  function outlook(row, now) {
    if (!row) return { title: "No saved observation", reason: "This flight and date have not been manually checked. No delay probability is available." };
    if (row.cancelled) return { title: "Cancelled at last check", reason: "A departure-delay probability is not applicable to a flight reported cancelled." };
    if (instant(row.actualDeparture) !== null) return { title: "Already departed", reason: "Departure had already occurred when this flight was checked. A pre-departure probability would no longer be a forecast." };
    var estimated = difference(row.scheduledDeparture, row.estimatedDeparture);
    if (estimated !== null && instant(row.estimatedDeparture) > now) return {
      title: estimated > 0 ? "Departure estimated " + estimated + " minutes after schedule" : estimated < 0 ? "Departure estimated " + Math.abs(estimated) + " minutes before schedule" : "Departure estimated on schedule",
      reason: "This is Flightradar24’s published departure-time estimate at the last check. It may change. Flightradar24 does not provide a probability percentage in this record."
    };
    if (instant(row.scheduledDeparture) !== null && instant(row.scheduledDeparture) <= now) return { title: "Departure outcome needs checking", reason: "The scheduled departure time has passed. The saved observation does not confirm an actual departure; no fresh status or probability is inferred." };
    return { title: "Insufficient evidence for a percentage", reason: "A reliable estimate needs earlier airport departures, a forecast issued before departure, and a model fitted and tested on comparable records." };
  }
  return { normalFlight: normalFlight, validFlight: validFlight, normalAirport: normalAirport, validAirport: validAirport,
    validDate: validDate, instant: instant, difference: difference, gateDelay: gateDelay, select: select, stats: stats, outlook: outlook };
});
