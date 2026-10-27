#!/usr/bin/env node
'use strict';

// Local authoring helper. Codex reads Flightradar24; this script makes no network requests.
const fs = require('node:fs');
const path = require('node:path');
const {isDeepStrictEqual} = require('node:util');
const {randomUUID} = require('node:crypto');
const core = require('./tracker-core.js');

const TIMES = ['scheduledDeparture', 'estimatedDeparture', 'actualDeparture', 'scheduledArrival', 'estimatedArrival', 'landedAt'];
const FIELDS = new Set(['flight', 'aliases', 'date', 'origin', 'originAliases', 'destination', 'originName', 'destinationName', 'originTimeZone', 'destinationTimeZone', ...TIMES, 'departureEvent', 'status', 'aircraft', 'registration', 'checkedAt', 'sourceUrl', 'cancelled']);
const HELP = `Update the flight tracker's saved Flightradar24 observations (Node.js 18+).

Create a blank observation to fill after reading Flightradar24:
  node flight-tracker/update-observations.cjs --flight UA1219 --airport RNO --date 2026-10-19 --destination SFO --template > /tmp/flight-check.json

Preview, then apply checked observations and set the page defaults:
  node flight-tracker/update-observations.cjs --flight UA1219 --airport RNO --date 2026-10-19 --input /tmp/flight-check.json --dry-run
  node flight-tracker/update-observations.cjs --flight UA1219 --airport RNO --date 2026-10-19 --input /tmp/flight-check.json

Input: {"records": [...], "contexts": [...]}. Contexts are optional.
Use {"records": []} to select an already saved flight/date without refreshing it.
--snapshot PATH selects a different snapshot, useful for testing.
Times must include their UTC offset or Z. checkedAt is the actual source-check time.
The script validates and writes locally; it does not fetch, commit, push, or deploy.
`;

function check(condition, message) {
  if (!condition) throw new Error(message);
}
function localDate(instant, timeZone) {
  return new Intl.DateTimeFormat('en-CA', {timeZone, year:'numeric', month:'2-digit', day:'2-digit'}).format(new Date(instant));
}
function timestamp(value, label) {
  check(core.instant(value) !== null && Number(value.slice(11,13)) < 24 && Number(value.slice(14,16)) < 60 && Number(value.slice(17,19)) < 60, `${label} needs an ISO timestamp with an explicit offset or Z.`);
}
function checkedTime(value, now) {
  timestamp(value, 'checkedAt');
  check(Date.parse(value) <= now + 60000, 'checkedAt cannot be in the future. Use the time the source was actually checked.');
}
function selectionOf(options) {
  const selection = {flight:core.normalFlight(options.flight), airport:core.normalAirport(options.airport), date:options.date};
  check(core.validFlight(selection.flight) && core.validAirport(selection.airport) && core.validDate(selection.date), 'Supply --flight, --airport, and a valid local --date (YYYY-MM-DD).');
  return selection;
}
function matches(row, selection) {
  return [row.flight, ...(row.aliases || [])].includes(selection.flight) && [row.origin, ...(row.originAliases || [])].includes(selection.airport);
}
function codes(values, normalize) {
  check(values == null || Array.isArray(values), 'Aliases must be arrays.');
  return [...new Set((values || []).map(normalize))];
}
function identity(row) {
  return {...row, flight:core.normalFlight(row.flight), origin:core.normalAirport(row.origin), aliases:codes(row.aliases, core.normalFlight), originAliases:codes(row.originAliases, core.normalAirport)};
}
function recordOf(raw, now) {
  check(raw && typeof raw === 'object' && !Array.isArray(raw), 'Each observation must be an object.');
  for (const key of Object.keys(raw)) check(FIELDS.has(key), `Unknown observation field: ${key}`);
  const row = {...identity(raw), destination:core.normalAirport(raw.destination), departureEvent:raw.departureEvent || 'provider-atd'};
  for (const field of TIMES) {
    row[field] = raw[field] == null ? null : raw[field];
    if (row[field] !== null) timestamp(row[field], field);
  }
  check(row.scheduledDeparture !== null, 'scheduledDeparture is required to establish the local flight date.');
  const day = localDate(row.scheduledDeparture, row.originTimeZone);
  check(raw.date == null || raw.date === day, `The local departure date is ${day}, not ${raw.date}. Check the source timezone.`);
  row.date = day;
  checkedTime(row.checkedAt, now);
  for (const key of ['originName', 'destinationName', 'status']) check(typeof row[key] === 'string' && row[key].trim(), `Fill ${key} from the checked source.`);
  const source = new URL(row.sourceUrl);
  check(core.validSourceUrl(row.sourceUrl), 'sourceUrl must be an HTTPS Flightradar24 link.');
  const sourceFlight = source.pathname.match(/^\/data\/flights\/([^/]+)\/?$/);
  const sourceAircraft = source.pathname.match(/^\/data\/aircraft\/([^/]+)\/?$/);
  check((sourceFlight && [row.flight, ...row.aliases].includes(core.normalFlight(sourceFlight[1]))) || (sourceAircraft && row.registration && sourceAircraft[1].toUpperCase() === row.registration.toUpperCase()), 'sourceUrl must be the matching Flightradar24 flight or aircraft history page.');
  check(row.cancelled == null || typeof row.cancelled === 'boolean', 'cancelled must be a boolean.');
  if (row.actualDeparture && row.landedAt) check(Date.parse(row.landedAt) >= Date.parse(row.actualDeparture), 'Landing cannot precede actual departure. Check the UTC date rollover.');
  if (row.scheduledArrival) check(Date.parse(row.scheduledArrival) >= Date.parse(row.scheduledDeparture), 'Scheduled arrival cannot precede departure. Check the date and offsets.');
  return row;
}
function routeDay(row) {
  return [row.flight, row.origin, row.destination, row.date].join('|');
}
function overlap(left, right) {
  return left.some(code => right.includes(code));
}
function requireNewer(next, previous) {
  check(Date.parse(next.checkedAt) >= Date.parse(previous.checkedAt), 'An older check cannot replace a newer observation.');
  if (Date.parse(next.checkedAt) === Date.parse(previous.checkedAt)) check(isDeepStrictEqual(next, previous), 'Conflicting observations have the same checkedAt. Use the actual time of a new source check.');
}

function updateSnapshot(snapshot, input, selection, now = Date.now()) {
  core.validateSnapshot(snapshot);
  check(input && Array.isArray(input.records), 'Input must contain a records array.');
  check(Object.keys(input).every(key => ['records', 'contexts'].includes(key)), 'Input only accepts records and optional contexts.');
  check(input.contexts == null || Array.isArray(input.contexts), 'contexts must be an array.');
  const next = structuredClone(snapshot);
  const batchKeys = new Set();
  const changes = {added:0, updated:0, unchanged:0, contexts:0};
  for (const raw of input.records) {
    const row = recordOf(raw, now);
    check(matches(row, selection), 'Every supplied observation must match the requested flight and departure airport, including known aliases.');
    const known = next.records.filter(old => old.date === row.date && old.destination === row.destination &&
      overlap([old.flight, ...(old.aliases || [])], [row.flight, ...row.aliases]) &&
      overlap([old.origin, ...(old.originAliases || [])], [row.origin, ...row.originAliases]));
    check(new Set(known.map(old => [old.flight, old.origin].join('|'))).size <= 1, 'Ambiguous flight or airport aliases; resolve the canonical codes before updating.');
    if (known.length) {
      const old = known[0];
      row.aliases = [...new Set([...(old.aliases || []), ...row.aliases, row.flight])].filter(code => code !== old.flight);
      row.originAliases = [...new Set([...(old.originAliases || []), ...row.originAliases, row.origin])].filter(code => code !== old.origin);
      row.flight = old.flight;
      row.origin = old.origin;
    }
    const key = routeDay(row);
    check(!batchKeys.has(key), 'Supply only one observation per flight, route, and local date in a batch.');
    batchKeys.add(key);
    const candidates = next.records.map((old, i) => ({old, i})).filter(({old}) => routeDay(old) === key);
    const exact = candidates.filter(({old}) => Date.parse(old.scheduledDeparture) === Date.parse(row.scheduledDeparture));
    check(candidates.length <= 1 || exact.length === 1, 'Multiple saved departures match this route/date. Identify the exact scheduled departure before updating.');
    const previous = candidates.length <= 1 ? candidates[0] : exact[0];
    if (previous) {
      const normalized = recordOf(previous.old, now);
      if ((normalized.actualDeparture && !row.actualDeparture) || (normalized.landedAt && !row.landedAt)) {
        changes.unchanged++;
        continue; // An older cached schedule must not erase a completed flight or refresh its check time.
      }
      requireNewer(row, normalized);
      if (isDeepStrictEqual(row, normalized)) changes.unchanged++;
      else { next.records[previous.i] = row; changes.updated++; }
    } else { next.records.push(row); changes.added++; }
  }
  for (const raw of input.contexts || []) {
    const context = identity(raw);
    check(matches(context, selection), 'Context must match the requested flight and departure airport.');
    const index = (next.contexts || []).findIndex(old => old.flight === context.flight && old.origin === context.origin && old.date === context.date);
    const previous = index < 0 ? {} : next.contexts[index];
    const merged = {...previous, ...context};
    for (const kind of ['airport', 'weather']) {
      if (!Object.hasOwn(context, kind)) continue;
      check(context[kind] && typeof context[kind] === 'object', `Omit unchecked ${kind} context instead of clearing its previous observation.`);
      checkedTime(context[kind].checkedAt, now);
      if (previous[kind]) requireNewer(context[kind], previous[kind]);
    }
    next.contexts ||= [];
    if (index < 0) next.contexts.push(merged);
    else next.contexts[index] = merged;
    changes.contexts++;
  }
  next.defaultFlight = selection.flight;
  next.defaultAirport = selection.airport;
  next.defaultDate = selection.date;
  const selected = core.select(next, selection.flight, selection.airport, selection.date).current;
  check(selected.length > 0, 'No saved observation matches the requested local date. Check the flight, airport, UTC rollover, and aliases.');
  // Replaying old input or changing defaults must never pretend a new check occurred.
  next.checkedAt = next.records.reduce((latest, row) => Date.parse(row.checkedAt) > Date.parse(latest) ? row.checkedAt : latest, snapshot.checkedAt);
  next.observationDate = localDate(next.checkedAt, selected[0].originTimeZone);
  core.validateSnapshot(next);
  return {snapshot:next, changes};
}

function template(snapshot, selection, destination) {
  destination = core.normalAirport(destination);
  check(core.validAirport(destination), 'Supply --destination for the observation template.');
  function airport(code) {
    for (const row of snapshot.records) {
      if ([row.origin, ...(row.originAliases || [])].includes(code)) return {code:row.origin, aliases:row.originAliases || [], name:row.originName, zone:row.originTimeZone};
      if (row.destination === code) return {code, aliases:[], name:row.destinationName, zone:row.destinationTimeZone};
    }
    return {code, aliases:[], name:null, zone:null};
  }
  const origin = airport(selection.airport), arrival = airport(destination);
  const known = snapshot.records.find(row => matches(row, selection) && row.destination === arrival.code);
  const flight = known ? known.flight : selection.flight;
  return {records:[{
    flight, aliases:known ? known.aliases || [] : [], date:selection.date,
    origin:origin.code, originAliases:origin.aliases, destination:arrival.code,
    originName:origin.name, destinationName:arrival.name,
    originTimeZone:origin.zone, destinationTimeZone:arrival.zone,
    ...Object.fromEntries(TIMES.map(key => [key, null])),
    departureEvent:'provider-atd', status:null, aircraft:null, registration:null,
    checkedAt:null, sourceUrl:`https://www.flightradar24.com/data/flights/${flight.toLowerCase()}`
  }]};
}

function main(args) {
  if (args.includes('--help') || args.length === 0) { process.stdout.write(HELP); return; }
  const options = {};
  const flags = new Set(['template', 'dry-run']);
  const values = new Set(['flight', 'airport', 'date', 'destination', 'input', 'snapshot']);
  for (let i = 0; i < args.length; i++) {
    const key = args[i].slice(2);
    check(args[i].startsWith('--') && (flags.has(key) || values.has(key)) && !Object.hasOwn(options, key), `Unknown or repeated option: ${args[i]}`);
    if (flags.has(key)) options[key] = true;
    else { check(args[i+1] && !args[i+1].startsWith('--'), `Missing value for --${key}`); options[key] = args[++i]; }
  }
  const selection = selectionOf(options);
  const file = path.resolve(options.snapshot || path.join(__dirname, 'observations.json'));
  const before = fs.readFileSync(file, 'utf8');
  const snapshot = core.validateSnapshot(JSON.parse(before));
  if (options.template) {
    check(!options.input && !options['dry-run'], '--template cannot be combined with --input or --dry-run.');
    process.stdout.write(JSON.stringify(template(snapshot, selection, options.destination), null, 2) + '\n');
    return;
  }
  check(options.input && !options.destination, 'Supply --input; --destination is only used with --template.');
  const input = JSON.parse(fs.readFileSync(path.resolve(options.input), 'utf8'));
  const result = updateSnapshot(snapshot, input, selection);
  const changed = !isDeepStrictEqual(snapshot, result.snapshot);
  if (changed && !options['dry-run']) {
    const temporary = `${file}.${randomUUID()}.tmp`;
    try {
      fs.writeFileSync(temporary, JSON.stringify(result.snapshot, null, 2) + '\n', {flag:'wx', mode:fs.statSync(file).mode});
      check(fs.readFileSync(file, 'utf8') === before, 'The snapshot changed during this update; rerun against the latest file.');
      fs.renameSync(temporary, file);
    } finally { if (fs.existsSync(temporary)) fs.unlinkSync(temporary); }
  }
  const history = core.select(result.snapshot, selection.flight, selection.airport, selection.date).history;
  console.log(JSON.stringify({mode:options['dry-run'] ? 'dry-run' : changed ? 'updated locally' : 'unchanged', file, defaults:selection, ...result.changes, checkedAt:result.snapshot.checkedAt, history:{from:history[0].date, to:history.at(-1).date, savedDays:history.filter(day => day.flights.length).length, missingDates:history.filter(day => !day.flights.length).map(day => day.date)}, preview:`https://bailiping.com/flight-tracker/?${new URLSearchParams(selection)}`, next:'Run node tests/flight-tracker.test.cjs, review the diff, then publish and verify the bare live page.'}, null, 2));
}

module.exports = {updateSnapshot, template, selectionOf};
if (require.main === module) {
  try { main(process.argv.slice(2)); }
  catch (error) { console.error(`Flight update failed: ${error.message}`); process.exitCode = 1; }
}
