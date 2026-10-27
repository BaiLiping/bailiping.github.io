'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const {parseHistory, mergeEvidence, checkFlight} = require('../flight-tracker/check-flight.cjs');
const config = {flight:'UA1219', aliases:['UAL1219'], origin:'RNO', originAliases:[], destination:'SFO', originName:'Reno–Tahoe', destinationName:'San Francisco', originTimeZone:'America/Los_Angeles', destinationTimeZone:'America/Los_Angeles', from:'2026-10-09', to:'2026-10-19', checkedAt:'2026-10-27T21:00:00Z'};
const page = (lines, url = 'https://www.flightradar24.com/data/flights/ua1219', extra = '') => `Flight history (${url})\nCrawled: today;\nTIME UTC\n${extra}\n${lines.join('\n')}\n`;
const rows = [
  'L104: 【12†N840UA】 20 Oct 2026 1:11 Landed 03:50 STD 01:39 ATD 02:39 STA 02:56 FROM Reno 【3†(RNO)】 TO San Francisco 【4†(SFO)】',
  'N841UA 19 Oct 2026 0:55 Landed 03:12 STD 01:39 ATD 02:17 STA 02:56 FROM Reno (RNO) TO San Francisco (SFO)',
  'N17338 19 Oct 2026 0:50 Landed 05:20 STD 04:12 ATD 04:30 STA 05:46 FROM San Francisco (SFO) TO Medford (MFR)',
  'Not available 28 Oct 2026 - Scheduled STD 10:12 ATD — STA 11:50 FROM Cleveland (CLE) TO Chicago (ORD)',
];

test('extracts requested route history, converts UTC dates, and distinguishes events', () => {
  const result = parseHistory(page(rows), config);
  assert.deepEqual(result.records.map(row => row.date), ['2026-10-19','2026-10-18']);
  assert.equal(result.records[0].scheduledDeparture, '2026-10-20T01:39:00Z');
  assert.equal(result.records[0].actualDeparture, '2026-10-20T02:39:00Z');
  assert.equal(result.records[0].landedAt, '2026-10-20T03:50:00Z');
  assert.equal(result.records[0].departureEvent, 'provider-atd');
  assert.equal(result.records[0].registration, 'N840UA');
  assert.equal(result.records[0].estimatedDeparture, null);
  assert.equal(parseHistory(page(rows, undefined, 'AIRCRAFT\n\nREGISTRATION'), config).records[0].aircraft, null, 'Column headings are not aircraft models');
  assert.deepEqual(result.skipped, []);
});

test('opens aircraft evidence without mixing other flight numbers or providers', () => {
  const result = parseHistory(page([
    'UA1219 19 Oct 2026 0:55 Landed 03:12 STD 01:39 ATD 02:17 STA 02:56 FROM Reno (RNO) TO San Francisco (SFO)',
    'UA5264 18 Oct 2026 0:55 Landed 03:12 STD 01:39 ATD 02:17 STA 02:56 FROM Reno (RNO) TO San Francisco (SFO)'
  ], 'https://data.flightradar24.com/data/aircraft/n841ua', 'AIRCRAFT Airbus A319-131'), config);
  assert.equal(result.records.length, 1);
  assert.equal(result.records[0].registration, 'N841UA');
  assert.equal(result.records[0].aircraft, 'Airbus A319-131');
  assert.equal(parseHistory(page(rows, 'https://example.com/data/flights/ua1219'), config).records.length, 0);
  assert.equal(parseHistory(page(rows, 'https://www.flightradar24.com/data/flights/ua5264'), config).records.length, 0);
  assert.equal(parseHistory(page(rows).replace('TIME UTC', 'Local airport time'), config).records.length, 0);
});

test('keeps estimates separate and never upgrades an older schedule to an actual outcome', () => {
  const actual = parseHistory(page(rows), config).records;
  const estimated = parseHistory(page(['N840UA 20 Oct 2026 - Estimated departure 01:45 STD 01:39 ATD — STA 02:56 FROM Reno (RNO) TO San Francisco (SFO)']), config).records;
  assert.equal(estimated[0].estimatedDeparture, '2026-10-20T01:45:00Z');
  assert.equal(estimated[0].actualDeparture, null);
  const merged = mergeEvidence([actual, estimated]);
  assert.equal(merged.find(row => row.date === '2026-10-19').status, 'Landed');
  assert.throws(() => mergeEvidence([actual, [{...actual[0], landedAt:'2026-10-20T03:51:00Z'}]]), /Conflicting completed/);
});

test('handles midnight rollover and refuses ambiguous duration/date combinations', () => {
  const result = parseHistory(page(['N123 19 Oct 2026 0:55 Landed 01:15 STD 23:50 ATD 00:20 STA 01:05 FROM Reno (RNO) TO San Francisco (SFO)']), config);
  assert.equal(result.records[0].actualDeparture, '2026-10-20T00:20:00Z');
  assert.equal(result.records[0].landedAt, '2026-10-20T01:15:00Z');
  assert.equal(result.records[0].scheduledArrival, '2026-10-20T01:05:00Z');
  const ambiguous = parseHistory(page([rows[1].replace('0:55', '12:55')]), config);
  assert.equal(ambiguous.records.length, 0);
  assert.equal(ambiguous.skipped.length, 1);
});

test('runs in the Codex isolate without Node or browser globals', () => {
  const context = vm.createContext({module:{exports:{}}});
  vm.runInContext(fs.readFileSync(require.resolve('../flight-tracker/check-flight.cjs'), 'utf8'), context);
  assert.equal(context.module.exports.parseHistory(page(rows), config).records.length, 2);
});

test('workflow searches missing dates, retrieves history, and passes checked data to the updater', async () => {
  const calls = [], commands = [];
  const tools = {
    clock__curr_time:async () => ({current_time:'2026-10-27 21:00:00 UTC'}),
    web__run:async args => { calls.push(args); return page(rows); },
    exec_command:async args => {
      commands.push(args.cmd);
      if (args.cmd.includes('--template')) return {exit_code:0, output:JSON.stringify({records:[{...config, sourceUrl:'https://www.flightradar24.com/data/flights/ua1219'}]})};
      if (args.cmd.startsWith('printf')) return {exit_code:0, output:''};
      return {exit_code:0, output:JSON.stringify({mode:'dry-run'})};
    },
  };
  const result = await checkFlight(tools, {flight:'UA1219', airport:'RNO', destination:'SFO', date:'2026-10-19', dryRun:true});
  assert.deepEqual(result.retrievedDates, ['2026-10-18', '2026-10-19']);
  assert.equal(result.missingHistoryDates.length, 9);
  assert.ok(calls.some(call => call.open?.some(item => item.ref_id.endsWith('/ua1219'))));
  assert.ok(calls.some(call => call.search_query?.some(item => item.q.includes('09 Oct 2026'))));
  assert.ok(calls.every(call => !call.search_query || call.search_query.every(query => query.domains.includes('flightradar24.com'))));
  assert.ok(commands.at(-1).endsWith('--dry-run'));
  assert.ok(commands.find(command => command.startsWith('printf')).includes('2026-10-18'));
  tools.web__run = async () => 'Access unavailable';
  const before = commands.length;
  await assert.rejects(() => checkFlight(tools, {flight:'UA1219', airport:'RNO', destination:'SFO', date:'2026-10-19'}), /No snapshot was changed/);
  assert.equal(commands.length, before+1, 'Only the read-only template command runs if source access fails');
});
