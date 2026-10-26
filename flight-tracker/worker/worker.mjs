/**
 * FlightAware AeroAPI adapter for the static bailiping.com flight tracker.
 *
 * Deploy as a Cloudflare Worker after setting FLIGHTAWARE_API_KEY as a secret.
 * This is a separate backend: never embed FlightAware credentials in GitHub Pages.
 * The provider supplies scheduled, estimated and actual gate-to-gate times.
 */

const PROVIDER = "FlightAware AeroAPI";
const FLIGHT_RE = /^[A-Z0-9]{2,3}[0-9]{1,5}[A-Z]?$/;
const DATE_RE = /^\d{4}-\d{2}-\d{2}$/;

export function isValidDate(value) {
  if (typeof value !== "string" || !DATE_RE.test(value)) return false;
  const d = new Date(value + "T12:00:00Z");
  return Number.isFinite(d.getTime()) && d.toISOString().slice(0, 10) === value;
}

export function shiftDate(date, offset) {
  const d = new Date(date + "T12:00:00Z");
  d.setUTCDate(d.getUTCDate() + offset);
  return d.toISOString().slice(0, 10);
}

export function dateAtOrigin(iso, timeZone) {
  const time = Date.parse(iso || "");
  if (!Number.isFinite(time)) return null;
  if (timeZone) {
    try {
      const parts = new Intl.DateTimeFormat("en", {
        timeZone, year: "numeric", month: "2-digit", day: "2-digit"
      }).formatToParts(new Date(time));
      const fields = Object.fromEntries(parts.map(p => [p.type, p.value]));
      return fields.year + "-" + fields.month + "-" + fields.day;
    } catch (_) {
      // Vendor did not provide a usable IANA timezone: display UTC date instead.
    }
  }
  return new Date(time).toISOString().slice(0, 10);
}

function safeTime(value) {
  if (typeof value !== "string" || !Number.isFinite(Date.parse(value))) return null;
  return new Date(value).toISOString();
}

function minutesLate(scheduled, actual, estimated) {
  if (!scheduled) return null;
  const comparison = actual || estimated;
  if (!comparison) return null;
  const value = (Date.parse(comparison) - Date.parse(scheduled)) / 60000;
  return Number.isFinite(value) ? Math.round(value) : null;
}

function airportCode(value) {
  if (typeof value === "string") return value.slice(0, 12);
  if (!value || typeof value !== "object") return null;
  return (value.code_iata || value.code_icao || value.code || null)?.slice(0, 12) || null;
}

function airportTimeZone(value, explicit) {
  const candidate = explicit || (value && typeof value === "object" ? (value.timezone || value.time_zone || value.tz) : null);
  if (typeof candidate !== "string" || candidate.length > 80) return null;
  try {
    new Intl.DateTimeFormat("en", { timeZone: candidate });
    return candidate;
  } catch (_) {
    return null;
  }
}

function compactRecord(raw) {
  const out = safeTime(raw.actual_out);
  const scheduledOut = safeTime(raw.scheduled_out);
  const estimatedOut = safeTime(raw.estimated_out);
  const scheduledIn = safeTime(raw.scheduled_in);
  const estimatedIn = safeTime(raw.estimated_in);
  const actualIn = safeTime(raw.actual_in);
  const actualOff = safeTime(raw.actual_off);
  const actualOn = safeTime(raw.actual_on);
  const originTZ = airportTimeZone(raw.origin, raw.origin_timezone);
  const date = dateAtOrigin(scheduledOut || out || safeTime(raw.scheduled_off) || safeTime(raw.estimated_out), originTZ);
  if (!date) return null;
  const cancelled = raw.cancelled === true || /\bcancel(l)?ed\b/i.test(String(raw.status || ""));
  const diverted = raw.diverted === true || /\bdiverted\b/i.test(String(raw.status || ""));
  let status = typeof raw.status === "string" ? raw.status.slice(0, 80) : "Status unavailable";
  if (cancelled) status = "Cancelled";
  else if (diverted) status = "Diverted";
  else if (actualIn) status = "Arrived";
  else if (actualOff || actualOn) status = actualOn ? "Landed" : "Airborne";
  else if (out) status = "Departed";
  return {
    id: typeof raw.fa_flight_id === "string" ? raw.fa_flight_id.slice(0, 130) : "",
    flight: typeof raw.ident === "string" ? raw.ident.slice(0, 14) : "",
    departureDate: date,
    dateBasis: originTZ ? "departure-airport local" : "UTC",
    originTZ,
    origin: airportCode(raw.origin),
    destination: airportCode(raw.destination),
    status,
    cancelled,
    diverted,
    scheduledOut,
    estimatedOut,
    actualOut: out,
    scheduledIn,
    estimatedIn,
    actualIn,
    departureDelay: cancelled ? null : minutesLate(scheduledOut, out, estimatedOut),
    arrivalDelay: cancelled ? null : minutesLate(scheduledIn, actualIn, estimatedIn),
    departureDelayType: out ? "actual" : (estimatedOut ? "estimated" : null),
    arrivalDelayType: actualIn ? "actual" : (estimatedIn ? "estimated" : null)
  };
}

function uniqueRecords(raw) {
  const seen = new Set();
  const flights = [];
  for (const item of raw) {
    if (!item || typeof item !== "object") continue;
    const record = compactRecord(item);
    if (!record) continue;
    const id = record.id || [
      record.flight, record.departureDate, record.origin,
      record.destination, record.scheduledOut
    ].join("|");
    if (seen.has(id)) continue;
    seen.add(id);
    flights.push(record);
  }
  return flights.sort((a, b) => (a.scheduledOut || "").localeCompare(b.scheduledOut || ""));
}

export function normalizeData(flight, selectedDate, rawFlights, now = new Date()) {
  const today = now.toISOString().slice(0, 10);
  const historyAnchor = selectedDate > today ? today : selectedDate;
  const records = uniqueRecords(rawFlights);
  const history = [];
  for (let ago = 10; ago >= 1; ago--) {
    const date = shiftDate(historyAnchor, -ago);
    history.push({
      date,
      flights: records.filter(record => record.departureDate === date)
    });
  }
  const current = records.filter(record => record.departureDate === selectedDate);
  const allHistory = history.flatMap(row => row.flights);
  const actualArrivals = allHistory.filter(
    record => record.arrivalDelayType === "actual" && record.arrivalDelay !== null && !record.cancelled
  );
  const knownDeparture = allHistory.filter(
    record => record.departureDelayType === "actual" && record.departureDelay !== null && !record.cancelled
  );
  const totalArrivalDelay = actualArrivals.reduce((sum, row) => sum + row.arrivalDelay, 0);
  const onTime = actualArrivals.filter(row => row.arrivalDelay < 15).length;
  return {
    flight,
    selectedDate,
    historyAnchor,
    history,
    current,
    source: PROVIDER,
    asOf: now.toISOString(),
    dateNote: "Provider timestamps are displayed in UTC. Dates use the origin airport's IANA timezone if supplied, otherwise UTC.",
    summary: {
      historyFlights: allHistory.length,
      completedArrivals: actualArrivals.length,
      completedDepartures: knownDeparture.length,
      cancellations: allHistory.filter(row => row.cancelled).length,
      meanArrivalDelay: actualArrivals.length ? Math.round(totalArrivalDelay / actualArrivals.length) : null,
      onTimeArrivalPercent: actualArrivals.length ? Math.round(100 * onTime / actualArrivals.length) : null
    },
    currentNote: selectedDate > shiftDate(today, 2)
      ? "This departure date may be beyond FlightAware's current flight-query horizon."
      : null
  };
}

function json(body, status, origin, cacheControl) {
  const headers = {
    "Content-Type": "application/json; charset=utf-8",
    "X-Content-Type-Options": "nosniff",
    "Cache-Control": cacheControl || "no-store",
    "Vary": "Origin"
  };
  if (origin === "https://bailiping.com" || origin === "https://www.bailiping.com") {
    headers["Access-Control-Allow-Origin"] = origin;
    headers["Access-Control-Allow-Methods"] = "GET, OPTIONS";
    headers["Access-Control-Allow-Headers"] = "Accept";
  }
  return new Response(JSON.stringify(body), { status, headers });
}

async function fetchProvider(url, apiKey, fetchImpl) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 12000);
  try {
    const response = await fetchImpl(url, {
      method: "GET",
      headers: { "x-apikey": apiKey, "Accept": "application/json" },
      signal: controller.signal
    });
    if (!response.ok) {
      if (response.status === 401 || response.status === 403) {
        throw new Error("FlightAware rejected the API key or this endpoint is not included in the subscription.");
      }
      if (response.status === 429) {
        throw new Error("FlightAware's request limit was reached. Retry later.");
      }
      throw new Error("FlightAware is currently unavailable (HTTP " + response.status + ").");
    }
    const payload = await response.json();
    if (!payload || !Array.isArray(payload.flights)) throw new Error("Unexpected FlightAware response.");
    return payload.flights;
  } finally {
    clearTimeout(timeout);
  }
}

export async function loadFlightData(flight, selectedDate, apiKey, options = {}) {
  const now = options.now || new Date();
  const fetchImpl = options.fetchImpl || fetch;
  const today = now.toISOString().slice(0, 10);
  const anchor = selectedDate > today ? today : selectedDate;
  const oldest = shiftDate(anchor, -11);
  const recentLimit = shiftDate(today, -9);
  const root = "https://aeroapi.flightaware.com/aeroapi";
  let flights = [];

  // AeroAPI's regular flight lookup serves recent history and near-future status.
  if (oldest >= recentLimit && selectedDate <= shiftDate(today, 2)) {
    const u = root + "/flights/" + encodeURIComponent(flight) + "?max_pages=4";
    flights = await fetchProvider(u, apiKey, fetchImpl);
  } else {
    // Most lookups still benefit from the recent flight endpoint. For an older
    // selected date the paid History endpoint is required. Split the requested
    // window into chunks below its seven-day query-window limit.
    if (selectedDate > shiftDate(today, 2)) {
      const u = root + "/flights/" + encodeURIComponent(flight) + "?max_pages=4";
      flights = await fetchProvider(u, apiKey, fetchImpl);
    } else {
      const first = shiftDate(anchor, -11);
      const end = shiftDate(selectedDate, 2);
      for (let cursor = first; cursor < end; cursor = shiftDate(cursor, 6)) {
        const stop = shiftDate(cursor, 6) < end ? shiftDate(cursor, 6) : end;
        const url = new URL(root + "/history/flights/" + encodeURIComponent(flight));
        url.searchParams.set("start", cursor + "T00:00:00Z");
        url.searchParams.set("end", stop + "T00:00:00Z");
        url.searchParams.set("max_pages", "4");
        flights.push(...await fetchProvider(url.toString(), apiKey, fetchImpl));
      }
    }
  }
  return normalizeData(flight, selectedDate, flights, now);
}

export default {
  async fetch(request, env, context) {
    const origin = request.headers.get("Origin");
    if (request.method === "OPTIONS") return json({}, 204, origin);
    if (request.method !== "GET") return json({ error: "Method not allowed." }, 405, origin);
    const u = new URL(request.url);
    if (u.pathname !== "/" && u.pathname !== "/api/flight") {
      return json({ error: "Unknown endpoint." }, 404, origin);
    }
    const flight = (u.searchParams.get("flight") || "").toUpperCase().replace(/[\s-]+/g, "");
    const date = u.searchParams.get("date") || "";
    if (!FLIGHT_RE.test(flight) || !isValidDate(date)) {
      return json({ error: "Enter a valid flight number and departure date." }, 400, origin);
    }
    if (!env || !env.FLIGHTAWARE_API_KEY) {
      return json({ error: "FlightAware API key has not been configured on the data service." }, 503, origin);
    }
    const today = new Date().toISOString().slice(0, 10);
    if (date < shiftDate(today, -180) || date > shiftDate(today, 90)) {
      return json({ error: "Choose a departure date within the past 180 or next 90 days." }, 400, origin);
    }
    try {
      // Cache by flight + date to prevent duplicate billable AeroAPI requests.
      const cache = typeof caches !== "undefined" ? caches.default : null;
      const cacheKey = new Request("https://flight-cache.internal/api/flight?flight=" +
        encodeURIComponent(flight) + "&date=" + encodeURIComponent(date));
      if (cache) {
        const hit = await cache.match(cacheKey);
        if (hit) return json(await hit.json(), 200, origin, "public, max-age=0, s-maxage=120");
      }
      const payload = await loadFlightData(flight, date, env.FLIGHTAWARE_API_KEY);
      if (cache && context && typeof context.waitUntil === "function") {
        context.waitUntil(cache.put(cacheKey, new Response(JSON.stringify(payload), {
          headers: { "Content-Type": "application/json", "Cache-Control": "public, max-age=120" }
        })));
      }
      return json(payload, 200, origin, "public, max-age=0, s-maxage=120");
    } catch (error) {
      return json({ error: error instanceof Error ? error.message : "Data service is temporarily unavailable." }, 502, origin);
    }
  }
};
