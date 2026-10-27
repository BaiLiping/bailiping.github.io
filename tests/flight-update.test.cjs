'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {spawnSync} = require('node:child_process');
const core = require('../flight-tracker/tracker-core.js');
const {updateSnapshot, template, selectionOf} = require('../flight-tracker/update-observations.cjs');

const NOW = Date.parse('2026-10-27T22:00:00Z');
const selection = {flight:'UA1219', airport:'RNO', date:'2026-10-19'};
const saved = {
  flight:'UA1219', aliases:['UAL1219'], origin:'RNO', originAliases:['KRNO'], destination:'SFO',
  originName:'Reno–Tahoe', destinationName:'San Francisco',
  originTimeZone:'America/Los_Angeles', destinationTimeZone:'America/Los_Angeles', date:'2026-10-19',
  scheduledDeparture:'2026-10-20T01:39:00Z', estimatedDeparture:null, actualDeparture:null,
  scheduledArrival:'2026-10-20T02:56:00Z', estimatedArrival:null, landedAt:null,
  departureEvent:'provider-atd', status:'Scheduled', aircraft:null, registration:null,
  checkedAt:'2026-10-27T18:00:00Z', sourceUrl:'https://www.flightradar24.com/data/flights/ua1219',
};
const observation = {...saved, status:'Landed', actualDeparture:'2026-10-20T02:39:00Z', landedAt:'2026-10-20T03:50:00Z', checkedAt:'2026-10-27T21:15:18Z'};
function fixture() {
  return {
    schemaVersion:1, provider:'Flightradar24', collectionMethod:'Manual browser observation',
    checkedAt:saved.checkedAt, observationDate:'2026-10-27', defaultFlight:'SK935', defaultAirport:'CPH', defaultDate:'2026-10-19',
    records:[structuredClone(saved), {...saved, flight:'SK935', aliases:['SAS935'], origin:'CPH', originAliases:['EKCH'], originName:'Copenhagen', originTimeZone:'Europe/Copenhagen', scheduledDeparture:'2026-10-19T10:45:00Z', scheduledArrival:'2026-10-19T21:40:00Z', sourceUrl:'https://www.flightradar24.com/data/flights/sk935'}],
    contexts:[{flight:'UA1219', aliases:['UAL1219'], origin:'RNO', originAliases:['KRNO'], date:'2026-10-19',
      airport:{summary:'Previously checked airport sample.', checkedAt:saved.checkedAt, sourceUrl:'https://www.flightradar24.com/data/airports/rno/departures'},
      weather:{summary:'Previously checked weather.', checkedAt:saved.checkedAt, sourceUrl:'https://www.flightradar24.com/data/airports/rno/weather'}}],
  };
}
function update(input = {records:[observation]}, selected = selection, snapshot = fixture()) {
  return updateSnapshot(snapshot, input, selected, NOW);
}

test('updates the checked flight and defaults while preserving unrelated data', () => {
  const original = fixture(), before = structuredClone(original);
  const {snapshot, changes} = update({records:[observation]}, selection, original);
  assert.deepEqual(original, before, 'The input snapshot is never mutated');
  assert.equal(snapshot.defaultFlight, 'UA1219');
  assert.equal(snapshot.defaultAirport, 'RNO');
  assert.equal(snapshot.defaultDate, '2026-10-19');
  assert.equal(snapshot.checkedAt, observation.checkedAt);
  assert.equal(snapshot.observationDate, '2026-10-27');
  assert.equal(snapshot.records[0].status, 'Landed');
  assert.deepEqual(snapshot.records[1], before.records[1]);
  assert.deepEqual(snapshot.contexts, before.contexts);
  assert.deepEqual(changes, {added:0, updated:1, unchanged:0, contexts:0});
  core.validateSnapshot(snapshot);
});

test('derives the local date across UTC midnight and rejects a UTC date used as a local date', () => {
  const {date, ...withoutDate} = observation;
  assert.equal(update({records:[withoutDate]}).snapshot.records[0].date, '2026-10-19');
  assert.throws(() => update({records:[{...observation, date:'2026-10-20'}]}), /local departure date is 2026-10-19/);
  assert.throws(() => update({records:[withoutDate]}, {...selection, date:'2026-10-20'}), /No saved observation/);
});

test('uses the airport timezone after the daylight-saving transition', () => {
  const winter = {...observation, date:'2026-11-01', scheduledDeparture:'2026-11-02T02:39:00Z', scheduledArrival:'2026-11-02T03:56:00Z', actualDeparture:null, landedAt:null, status:'Scheduled'};
  const result = update({records:[winter]}, {...selection, date:winter.date});
  assert.equal(result.snapshot.records.at(-1).date, '2026-11-01');
  assert.equal(new Intl.DateTimeFormat('en-GB', {timeZone:winter.originTimeZone, hour:'2-digit', minute:'2-digit', hour12:false}).format(new Date(winter.scheduledDeparture)), '18:39');
});

test('replays are idempotent and selecting saved data does not refresh check times', () => {
  const once = update().snapshot;
  const replay = update({records:[observation]}, selection, once);
  assert.deepEqual(replay.snapshot, once);
  assert.equal(replay.changes.unchanged, 1);
  const defaults = update({records:[]}).snapshot;
  assert.deepEqual(defaults.records, fixture().records);
  assert.deepEqual(defaults.contexts, fixture().contexts);
  assert.equal(defaults.checkedAt, saved.checkedAt);
});

test('cached schedules cannot erase a previously checked actual outcome', () => {
  const completed = update().snapshot;
  const cachedSchedule = {...saved, checkedAt:'2026-10-27T21:30:00Z'};
  const result = update({records:[cachedSchedule]}, selection, completed);
  assert.deepEqual(result.snapshot, completed);
  assert.equal(result.changes.unchanged, 1);
});

test('known aliases update the canonical record without creating duplicates', () => {
  const alias = {...observation, flight:'UAL1219', aliases:['UA1219'], origin:'KRNO', originAliases:['RNO'], sourceUrl:'https://www.flightradar24.com/data/flights/ual1219'};
  const result = update({records:[alias]}, {...selection, flight:'UAL1219', airport:'KRNO'});
  assert.equal(result.snapshot.records.length, fixture().records.length);
  assert.equal(result.snapshot.records[0].flight, 'UA1219');
  assert.equal(result.snapshot.records[0].origin, 'RNO');
  assert.equal(core.select(result.snapshot, 'UAL1219', 'KRNO', selection.date).current.length, 1);
});

test('schedule corrections replace one record and ambiguous departures are refused', () => {
  const moved = {...observation, scheduledDeparture:'2026-10-20T01:49:00Z'};
  assert.equal(update({records:[moved]}).snapshot.records.length, fixture().records.length);
  const multiple = fixture();
  multiple.records.push({...saved, scheduledDeparture:'2026-10-20T01:59:00Z'});
  assert.throws(() => update({records:[moved]}, selection, multiple), /Multiple saved departures/);
  const exact = update({records:[observation]}, selection, multiple);
  assert.equal(exact.snapshot.records[0].status, 'Landed');
  assert.equal(exact.snapshot.records.at(-1).status, 'Scheduled');
});

test('rejects stale checks, same-time conflicts, future checks, and blank templates', () => {
  assert.throws(() => update({records:[{...observation, checkedAt:'2026-10-27T17:59:59Z'}]}), /older check/);
  assert.throws(() => update({records:[{...observation, checkedAt:saved.checkedAt}]}), /same checkedAt/);
  assert.throws(() => update({records:[{...observation, checkedAt:'2026-10-28T00:00:00Z'}]}), /future/);
  assert.throws(() => update({records:[{...observation, checkedAt:null}]}), /checkedAt/);
  assert.throws(() => update(template(fixture(), selection, 'SFO')), /scheduledDeparture/);
});

test('requires matching Flightradar24 sources and well-formed observations', () => {
  for (const sourceUrl of ['https://example.com/data/flights/ua1219', 'https://evilflightradar24.com/data/flights/ua1219', 'https://www.flightradar24.com/data/flights/ua5264', 'https://user:password@www.flightradar24.com/data/flights/ua1219']) {
    assert.throws(() => update({records:[{...observation, sourceUrl}]}), /source/);
  }
  for (const scheduledDeparture of ['2026-10-19 18:39', '2026-10-20T24:00:00Z', '2026-02-30T01:39:00Z']) {
    assert.throws(() => update({records:[{...observation, scheduledDeparture}]}), /timestamp/);
  }
  assert.throws(() => update({records:[{...observation, originTimeZone:'Invalid/Zone'}]}), /time zone/i);
  assert.throws(() => update({records:[{...observation, status:null}]}), /status/);
  assert.throws(() => update({records:[{...observation, actualDepartue:observation.actualDeparture}]}), /Unknown observation field/);
  assert.throws(() => update({records:[{...observation, landedAt:'2026-10-20T01:00:00Z'}]}), /Landing cannot precede/);
  assert.throws(() => update({records:[{...observation, scheduledArrival:'2026-10-20T00:56:00Z'}]}), /Scheduled arrival cannot precede/);
});

test('rejects duplicate batch entries and observations for another flight or airport', () => {
  assert.throws(() => update({records:[observation, observation]}), /only one observation/);
  assert.throws(() => update({records:[{...observation, flight:'UA5264', aliases:[], sourceUrl:'https://www.flightradar24.com/data/flights/ua5264'}]}), /requested flight/);
  assert.throws(() => update({records:[{...observation, origin:'LAX', originAliases:[]}]}), /departure airport/);
});

test('partial context updates preserve unrefreshed weather and its source-check time', () => {
  const context = {...fixture().contexts[0], airport:{summary:'New checked airport sample.', checkedAt:observation.checkedAt, sourceUrl:'https://www.flightradar24.com/data/airports/rno/departures'}};
  delete context.weather;
  const result = update({records:[], contexts:[context]}).snapshot;
  assert.deepEqual(result.contexts[0].weather, fixture().contexts[0].weather);
  assert.equal(result.contexts[0].airport.summary, 'New checked airport sample.');
  assert.equal(result.checkedAt, saved.checkedAt, 'Context checks do not refresh the flight check time');
  assert.throws(() => update({records:[], contexts:[{...context, weather:null}]}), /Omit unchecked weather/);
  assert.throws(() => update({records:[], contexts:[{...context, airport:{...context.airport, checkedAt:'2026-10-27T17:00:00Z'}}]}), /older check/);
});

test('templates reuse airport metadata but never copy previous status or timestamps', () => {
  const result = template(fixture(), {...selection, flight:'UAL1219', airport:'KRNO'}, 'SFO').records[0];
  assert.equal(result.flight, 'UA1219');
  assert.equal(result.origin, 'RNO');
  assert.equal(result.originTimeZone, 'America/Los_Angeles');
  assert.equal(result.sourceUrl, 'https://www.flightradar24.com/data/flights/ua1219');
  for (const key of ['scheduledDeparture', 'estimatedDeparture', 'actualDeparture', 'scheduledArrival', 'estimatedArrival', 'landedAt', 'checkedAt', 'status', 'aircraft', 'registration']) assert.equal(result[key], null);
  assert.deepEqual(selectionOf({flight:' ua 1219 ', airport:' rno ', date:selection.date}), selection);
  assert.throws(() => selectionOf({...selection, date:'2026-02-30'}), /valid local/);
});

test('CLI previews, writes atomically, replays without changes, and never writes invalid input', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'flight-update-test-'));
  try {
    const file = path.join(dir, 'snapshot.json'), input = path.join(dir, 'input.json');
    const original = JSON.stringify(fixture(), null, 2) + '\n';
    fs.writeFileSync(file, original);
    fs.writeFileSync(input, JSON.stringify({records:[observation]}));
    const script = path.resolve(__dirname, '../flight-tracker/update-observations.cjs');
    const base = [script, '--flight', 'UA1219', '--airport', 'RNO', '--date', '2026-10-19', '--snapshot', file];
    const run = (...args) => spawnSync(process.execPath, [...base, ...args], {encoding:'utf8'});
    const starter = run('--template', '--destination', 'SFO');
    assert.equal(starter.status, 0, starter.stderr);
    assert.equal(JSON.parse(starter.stdout).records[0].checkedAt, null);
    const preview = run('--input', input, '--dry-run');
    assert.equal(preview.status, 0, preview.stderr);
    assert.equal(JSON.parse(preview.stdout).mode, 'dry-run');
    assert.equal(fs.readFileSync(file, 'utf8'), original);
    const applied = run('--input', input);
    assert.equal(applied.status, 0, applied.stderr);
    assert.equal(JSON.parse(applied.stdout).mode, 'updated locally');
    const after = fs.readFileSync(file, 'utf8');
    assert.equal(JSON.parse(after).defaultDate, selection.date);
    const replay = run('--input', input);
    assert.equal(replay.status, 0, replay.stderr);
    assert.equal(JSON.parse(replay.stdout).mode, 'unchanged');
    assert.equal(fs.readFileSync(file, 'utf8'), after);
    fs.writeFileSync(input, JSON.stringify({records:[{...observation, sourceUrl:'https://example.com/data/flights/ua1219'}]}));
    const invalid = run('--input', input);
    assert.equal(invalid.status, 1);
    assert.match(invalid.stderr, /source/);
    assert.equal(fs.readFileSync(file, 'utf8'), after);
    assert.deepEqual(fs.readdirSync(dir).sort(), ['input.json', 'snapshot.json']);
    assert.equal(run('--bogus').status, 1);
    assert.equal(run('--input').status, 1);
  } finally { fs.rmSync(dir, {recursive:true, force:true}); }
});
