(function () {
  "use strict";

  var form = document.getElementById("flight-form");
  var flightField = document.getElementById("flight-number");
  var dateField = document.getElementById("flight-date");
  var errorField = document.getElementById("form-error");
  var results = document.getElementById("results");
  var searchButton = document.getElementById("search-button");
  var refreshButton = document.getElementById("refresh");
  var pending = null;
  var requestNumber = 0;

  function node(id) { return document.getElementById(id); }
  function normalFlight(value) { return String(value || "").toUpperCase().replace(/[\s-]+/g, ""); }
  function validFlight(value) { return /^[A-Z0-9]{2,3}[0-9]{1,5}[A-Z]?$/.test(value); }
  function validDate(value) {
    if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
    var date = new Date(value + "T12:00:00Z");
    return Number.isFinite(date.getTime()) && date.toISOString().slice(0, 10) === value;
  }
  function localToday() {
    var d = new Date();
    return [d.getFullYear(), String(d.getMonth() + 1).padStart(2, "0"), String(d.getDate()).padStart(2, "0")].join("-");
  }
  function utcToday() { return new Date().toISOString().slice(0, 10); }
  function shiftDay(day, by) {
    var d = new Date(day + "T12:00:00Z");
    d.setUTCDate(d.getUTCDate() + by);
    return d.toISOString().slice(0, 10);
  }
  function esc(value) {
    return String(value == null ? "" : value).replace(/[&<>"']/g, function (character) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[character];
    });
  }
  function displayDate(date) {
    if (!validDate(date)) return "—";
    return new Intl.DateTimeFormat("en-GB", {
      month: "short", day: "numeric", year: "numeric", timeZone: "UTC"
    }).format(new Date(date + "T12:00:00Z"));
  }
  function displayShort(date) {
    if (!validDate(date)) return "—";
    return new Intl.DateTimeFormat("en-GB", {
      day: "numeric", month: "short", timeZone: "UTC"
    }).format(new Date(date + "T12:00:00Z"));
  }
  function displayTime(value) {
    if (!value || !Number.isFinite(Date.parse(value))) return "—";
    return new Intl.DateTimeFormat("en-GB", {
      day: "numeric", month: "short", hour: "2-digit", minute: "2-digit",
      timeZone: "UTC", hour12: false
    }).format(new Date(value)) + " UTC";
  }
  function displayDelay(value, type) {
    if (typeof value !== "number" || !Number.isFinite(value)) return "—";
    return (value > 0 ? "+" : "") + value + " min" + (type === "estimated" ? " (est.)" : "");
  }
  function delayClass(value) {
    return typeof value !== "number" || !Number.isFinite(value) ? "unknown" : value > 0 ? "late" : "";
  }
  function setError(message) {
    errorField.hidden = false;
    errorField.textContent = message;
    results.hidden = true;
  }
  function resetError() {
    errorField.hidden = true;
    errorField.textContent = "";
  }
  function setLoading(loading) {
    searchButton.disabled = loading;
    refreshButton.disabled = loading;
    searchButton.textContent = loading ? "Loading…" : "Track flight";
    if (loading) node("fetch-status").textContent = "Retrieving flight records…";
  }
  function metric(value, label, sample) {
    return '<div class="metric"><strong>' + esc(value) + '</strong><span>' +
      esc(label) + '</span><small>' + esc(sample) + '</small></div>';
  }
  function historyDates(anchor) {
    var days = [];
    for (var i = 10; i >= 1; i--) days.push(shiftDay(anchor, -i));
    return days;
  }
  function safeList(value) { return Array.isArray(value) ? value : []; }
  function labelRoute(row) {
    return esc(row.origin || "—") + " → " + esc(row.destination || "—");
  }
  function statusText(row) {
    if (row.cancelled === true) return "Cancelled";
    if (row.diverted === true) return "Diverted";
    return esc(row.status || "Status unavailable");
  }
  function renderUnavailable(date, reason) {
    var today = utcToday();
    var anchor = date > today ? today : date;
    var dates = historyDates(anchor);
    node("history-period").textContent = "Requested period: " + displayDate(dates[0]) +
      " – " + displayDate(dates[9]) + ".";
    node("history-metrics").innerHTML =
      metric("—", "Flights recorded", "No verified records") +
      metric("—", "Average arrival delay", "No actual arrival times") +
      metric("—", "On-time arrivals", "No actual arrival times");
    node("history-chart").innerHTML = '<p class="muted">' + esc(reason) + '</p>';
    node("history-chart").setAttribute("aria-label", "No verified flight delay data available");
    node("history-rows").innerHTML = dates.map(function (day) {
      return '<tr><td>' + esc(displayDate(day)) +
        '</td><td class="missing" colspan="4">Not available — no flight data was retrieved</td></tr>';
    }).join("");
    node("current-period").textContent = "Departure date: " + displayDate(date);
    node("current-flights").innerHTML = '<div class="empty"><strong>Status unavailable</strong><p>' +
      esc(reason) + '</p></div>';
    node("provider-name").textContent = "FlightAware (not connected)";
    node("provider-updated").textContent = "Updated: unavailable";
  }

  function renderHistory(data) {
    var rows = safeList(data.history).filter(function (v) { return v && validDate(v.date); });
    var records = rows.flatMap(function (v) { return safeList(v.flights); });
    var finished = records.filter(function (v) {
      return v && !v.cancelled && v.arrivalDelayType === "actual" &&
        typeof v.arrivalDelay === "number" && Number.isFinite(v.arrivalDelay);
    });
    var avg = finished.length ? Math.round(finished.reduce(function (sum, v) {
      return sum + v.arrivalDelay;
    }, 0) / finished.length) : null;
    var onTime = finished.filter(function (v) { return v.arrivalDelay < 15; }).length;
    var onTimePercentage = finished.length ? Math.round(100 * onTime / finished.length) : null;
    var first = rows.length ? rows[0].date : null;
    var last = rows.length ? rows[rows.length - 1].date : null;
    node("history-period").textContent = first && last ?
      displayDate(first) + " – " + displayDate(last) + ", based on available FlightAware records." :
      "FlightAware did not return a usable history window.";
    node("history-metrics").innerHTML =
      metric(records.length, "Flight records", "Across 10 departure dates") +
      metric(avg === null ? "—" : (avg > 0 ? "+" : "") + avg + " min",
        "Average arrival delay", finished.length + " completed arrivals") +
      metric(onTimePercentage === null ? "—" : onTimePercentage + "%",
        "On-time arrivals", finished.length + " completed arrivals");

    var chartPoints = rows.map(function (entry) {
      var actual = safeList(entry.flights).filter(function (flight) {
        return flight && !flight.cancelled && flight.arrivalDelayType === "actual" &&
          typeof flight.arrivalDelay === "number" && Number.isFinite(flight.arrivalDelay);
      });
      return {
        date: entry.date,
        delay: actual.length ? Math.round(actual.reduce(function (sum, flight) {
          return sum + flight.arrivalDelay;
        }, 0) / actual.length) : null
      };
    });
    var upper = Math.max(15, ...chartPoints.map(function (point) { return Math.abs(point.delay || 0); }));
    node("history-chart").innerHTML =
      '<p class="chart-note">Actual arrival delays; a negative value means early arrival. No bar means no actual arrival record.</p>' +
      chartPoints.map(function (point) {
        var delay = point.delay;
        var width = delay === null ? 0 : Math.max(2, Math.round(100 * Math.abs(delay) / upper));
        var color = delay === null ? "missing" : delay > 0 ? "late" : delay < 0 ? "early" : "";
        var label = delay === null ? "No data" : displayDelay(delay, "actual");
        return '<div class="chart-row"><span class="chart-label">' + esc(displayShort(point.date)) +
          '</span><div class="chart-track"><span class="chart-fill ' + color + '" style="width:' +
          width + '%"></span></div><span class="chart-value">' + esc(label) + '</span></div>';
      }).join("");
    node("history-chart").setAttribute("aria-label", "Actual arrival delays by date over the previous 10 days");

    node("history-rows").innerHTML = rows.map(function (entry) {
      var flights = safeList(entry.flights);
      if (!flights.length) {
        return '<tr><td>' + esc(displayDate(entry.date)) +
          '</td><td class="missing" colspan="4">No flight record returned</td></tr>';
      }
      return flights.map(function (flight) {
        return '<tr><td>' + esc(displayDate(entry.date)) + '</td><td>' + labelRoute(flight) +
          '</td><td>' + statusText(flight) + '</td><td class="num">' +
          esc(displayDelay(flight.departureDelay, flight.departureDelayType)) +
          '</td><td class="num">' + esc(displayDelay(flight.arrivalDelay, flight.arrivalDelayType)) +
          '</td></tr>';
      }).join("");
    }).join("");
  }

  function renderHistoryError(data) {
    var rows = safeList(data.history);
    var first = rows[0] && rows[0].date;
    var last = rows[rows.length - 1] && rows[rows.length - 1].date;
    node("history-period").textContent = first && last ?
      displayDate(first) + " – " + displayDate(last) : "Previous 10 days";
    node("history-metrics").innerHTML =
      metric("—", "Flight records", "History access unavailable") +
      metric("—", "Average arrival delay", "Historical data unavailable") +
      metric("—", "On-time arrivals", "Historical data unavailable");
    node("history-chart").innerHTML = '<p class="muted">' + esc(data.historyError) + '</p>';
    node("history-chart").setAttribute("aria-label", "Historical flight data is unavailable");
    node("history-rows").innerHTML = rows.map(function (row) {
      return '<tr><td>' + esc(displayDate(row.date)) +
        '</td><td class="missing" colspan="4">Historical data unavailable — check AeroAPI subscription</td></tr>';
    }).join("");
  }

  function timeItem(label, value) {
    return '<div><dt>' + esc(label) + '</dt><dd>' + esc(displayTime(value)) + '</dd></div>';
  }
  function timeGroup(title, kind, flight) {
    var tail = kind === "departure" ? "Out" : "In";
    return '<div><h3>' + esc(title) + '</h3><dl>' +
      timeItem("Scheduled", flight["scheduled" + tail]) +
      timeItem("Estimated", flight["estimated" + tail]) +
      timeItem("Actual", flight["actual" + tail]) + '</dl></div>';
  }
  function renderCurrent(data) {
    var flights = safeList(data.current);
    var date = data.selectedDate;
    node("current-period").textContent = "Departure " + displayDate(date) +
      ". Gate departure and arrival times are shown in UTC when available.";
    if (!flights.length) {
      node("current-flights").innerHTML =
        '<div class="empty"><strong>No matching flight returned for this date</strong>' +
        '<p>This is not a cancellation confirmation. The flight may not operate daily, ' +
        'the airline code may refer to a codeshare, or the record may be outside the provider’s coverage. ' +
        esc(data.currentNote || "") + '</p></div>';
      return;
    }
    node("current-flights").innerHTML = flights.map(function (flight) {
      var status = statusText(flight);
      var statusClass = flight.cancelled || flight.diverted ? "warning" :
        (/unavailable|unknown|scheduled/i.test(status) ? "neutral" : "");
      return '<article class="flight-card"><div class="flight-top">' +
        '<div class="route">' + labelRoute(flight) + '</div><span class="status ' +
        statusClass + '">' + status + '</span></div><div class="time-grid">' +
        timeGroup("Departure", "departure", flight) + timeGroup("Arrival", "arrival", flight) +
        '</div><div class="delays">' +
        '<span class="delay ' + delayClass(flight.departureDelay) + '">Departure: ' +
        esc(displayDelay(flight.departureDelay, flight.departureDelayType)) + '</span>' +
        '<span class="delay ' + delayClass(flight.arrivalDelay) + '">Arrival: ' +
        esc(displayDelay(flight.arrivalDelay, flight.arrivalDelayType)) + '</span>' +
        '</div></article>';
    }).join("");
  }
  function validResponse(data, flight, date) {
    return data && typeof data === "object" && data.flight === flight &&
      data.selectedDate === date && Array.isArray(data.history) && Array.isArray(data.current);
  }
  function renderResponse(data) {
    if (data.historyError) renderHistoryError(data);
    else renderHistory(data);
    renderCurrent(data);
    node("provider-name").textContent = data.source || "FlightAware AeroAPI";
    node("provider-updated").textContent = "Updated: " + displayTime(data.asOf);
  }

  async function lookup(updateUrl) {
    var flight = normalFlight(flightField.value);
    var date = dateField.value;
    resetError();
    if (!validFlight(flight)) {
      setError("Enter a valid airline code and flight number, for example SK1431 or BA247.");
      flightField.focus();
      return;
    }
    if (!validDate(date)) {
      setError("Choose a valid departure date.");
      dateField.focus();
      return;
    }
    var min = shiftDay(utcToday(), -180);
    var max = shiftDay(utcToday(), 90);
    if (date < min || date > max) {
      setError("Choose a departure date within the past 180 days or next 90 days.");
      dateField.focus();
      return;
    }
    flightField.value = flight;
    node("result-flight").textContent = flight;
    node("result-date").textContent = "Departure " + displayDate(date);
    results.hidden = false;
    if (updateUrl) {
      var address = new URL(location.href);
      address.searchParams.set("flight", flight);
      address.searchParams.set("date", date);
      history.replaceState(null, "", address.toString());
    }
    if (pending) pending.abort();
    var id = ++requestNumber;
    renderUnavailable(date, "Loading flight data…");
    var endpoint = String(window.FLIGHT_TRACKER_API_URL || "").trim().replace(/\/+$/, "");
    if (!/^https:\/\//i.test(endpoint)) {
      renderUnavailable(date, "The secure FlightAware data service has not yet been configured. " +
        "A server-side API connection is required to show historical delays and current status here.");
      node("fetch-status").textContent = "Flight data connection not configured";
      return;
    }
    var url = endpoint + "/api/flight?flight=" + encodeURIComponent(flight) +
      "&date=" + encodeURIComponent(date);
    pending = new AbortController();
    setLoading(true);
    try {
      var response = await fetch(url, { headers: { "Accept": "application/json" }, signal: pending.signal });
      var data = await response.json();
      if (id !== requestNumber) return;
      if (!response.ok) throw new Error(data && data.error ? String(data.error).slice(0, 250) : "Flight data service error.");
      if (!validResponse(data, flight, date)) throw new Error("The data service returned an unexpected response.");
      renderResponse(data);
      node("fetch-status").textContent = "Provider data loaded";
    } catch (reason) {
      if (id !== requestNumber || reason.name === "AbortError") return;
      var message = reason instanceof Error ? reason.message : "Flight data is unavailable.";
      renderUnavailable(date, message);
      node("fetch-status").textContent = "Unable to retrieve flight data";
    } finally {
      if (id === requestNumber) { pending = null; setLoading(false); }
    }
  }

  form.addEventListener("submit", function (event) { event.preventDefault(); lookup(true); });
  refreshButton.addEventListener("click", function () { lookup(false); });
  dateField.value = localToday();
  var query = new URLSearchParams(location.search);
  if (query.has("flight")) {
    flightField.value = query.get("flight") || "";
    if (query.has("date")) dateField.value = query.get("date") || "";
    lookup(false);
  }
})();
