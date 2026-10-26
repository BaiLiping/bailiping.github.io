// Offline validation of the published snapshot, plus independent behavior fixtures.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const core = require('../flight-tracker/tracker-core.js');
const root = path.join(__dirname, '..', 'flight-tracker');
const publishedData = JSON.parse(fs.readFileSync(path.join(root, 'observations.json'), 'utf8'));
core.validateSnapshot(publishedData);

// Synthetic fixtures stay fixed when Codex publishes newer observations.
const base = {
  flight:'SK935', aliases:['SAS935'], origin:'CPH', originAliases:['EKCH'], destination:'SFO',
  originName:'Copenhagen', destinationName:'San Francisco',
  originTimeZone:'Europe/Copenhagen', destinationTimeZone:'America/Los_Angeles',
  departureEvent:'provider-atd', checkedAt:'2026-10-26T14:04:00Z',
  sourceUrl:'https://www.flightradar24.com/data/flights/sk935',
};
const observed = {...base,date:'2026-10-26',status:'Airborne',scheduledDeparture:'2026-10-26T11:40:00Z',actualDeparture:'2026-10-26T11:56:00Z',scheduledArrival:'2026-10-26T22:50:00Z',estimatedArrival:'2026-10-26T22:23:00Z',landedAt:null};
const data = {
  schemaVersion:1,provider:'Flightradar24',checkedAt:base.checkedAt,observationDate:'2026-10-26',defaultDate:'2026-10-27',defaultFlight:'SK935',defaultAirport:'CPH',
  records:[...Array.from({length:10},(_,i)=>{
    const date='2026-10-'+(16+i);
    return {...base,date,status:'Landed',scheduledDeparture:date+'T10:45:00Z',actualDeparture:date+'T11:00:00Z',scheduledArrival:date+'T21:40:00Z',landedAt:date+'T21:30:00Z'};
  }),observed,{...observed,date:'2026-10-27',status:'Estimated departure',scheduledDeparture:'2026-10-27T11:40:00Z',actualDeparture:null,estimatedDeparture:'2026-10-27T11:50:00Z',scheduledArrival:'2026-10-27T22:50:00Z',estimatedArrival:null}],
  contexts:[],
};
core.validateSnapshot(data);
assert.throws(()=>core.validateSnapshot({...data,observationDate:'2026-02-30'}),/snapshot date/);
assert.throws(()=>core.validateSnapshot({...data,records:[{...observed,checkedAt:null}]}),/check time/);
assert.throws(()=>core.validateSnapshot({...data,records:[{...observed,sourceUrl:'https://evilflightradar24.com/data'}]}),/source link/);
assert.throws(()=>core.validateSnapshot({...data,records:[{...observed,originTimeZone:'Invalid/Zone'}]}),/timezone/);
assert.throws(()=>core.validateSnapshot({...data,records:[{...observed,date:'2026-10-27'}]}),/local departure date/);
assert.throws(()=>core.validateSnapshot({...data,records:[{...observed,actualDeparture:'2026-02-30T12:00:00Z'}]}),/actualDeparture/);

assert.equal(core.normalFlight(' sk 935 '), 'SK935');
for (const code of ['SK935','SAS935','U21234','6E123','BA123A']) assert.ok(core.validFlight(code));
for (const code of ['12345','12A123','ABCD123','<script>']) assert.ok(!core.validFlight(code));
assert.ok(core.validAirport('CPH') && core.validAirport('EKCH'));
assert.ok(!core.validAirport('12X'));
assert.ok(core.validDate('2024-02-29'));
assert.ok(!core.validDate('2026-02-29') && !core.validDate('2026-02-31'));
assert.equal(core.difference(null,'2026-10-26T10:00:00Z'),null);
assert.equal(core.difference('2026-10-26T10:00:00Z',null),null);
assert.equal(core.difference('2026-10-26T10:00:00Z','2026-10-26T10:00:00Z'),0);
assert.equal(core.difference('2026-10-26T10:00:00Z','2026-10-26T10:00:30Z'),0.5);
assert.equal(core.difference('2026-10-26T23:50:00Z','2026-10-27T00:05:00Z'),15);
assert.equal(core.instant('2026-10-26 12:00'),null,'Unzoned times are not accepted');

const view=core.select(data,'SK935','CPH','2026-10-26');
assert.equal(view.current.length,1);
assert.equal(view.history.length,10);
assert.equal(view.history[0].date,'2026-10-16');
assert.equal(view.history[9].date,'2026-10-25');
assert.equal(view.history.flatMap(x=>x.flights).length,10);
assert.equal(core.select(data,'SK935','ARN','2026-10-26').current.length,0,'Never reuse another origin');
assert.equal(core.select(data,'BA247','CPH','2026-10-26').current.length,0,'Never reuse another flight');
assert.equal(core.select(data,'SK935','CPH','2026-10-28').current.length,0,'Never reuse another date');
assert.equal(core.select(data,'SAS935','EKCH','2026-10-26').current.length,1,'Known aliases work');
assert.equal(core.select(data,'SK935','CPH','2026-11-01').history[9].date,'2026-10-25','Future history is anchored to the saved observation day');

const r=view.current[0];
assert.equal(core.difference(r.scheduledDeparture,r.actualDeparture),16);
assert.equal(core.gateDelay(r),null,'Provider ATD is not a confirmed gate event');
const gate={...r,departureEvent:'gate',actualDeparture:'2026-10-26T11:40:30Z'};
assert.equal(core.gateDelay(gate),0.5,'Any positive delay counts, not just 15 minutes');
assert.equal(core.gateDelay({...gate,cancelled:true}),null);
const stats=core.stats([gate,{...gate,actualDeparture:gate.scheduledDeparture},{...gate,actualDeparture:null},{...gate,actualDeparture:'2026-10-26T11:39:00Z'}]);
assert.equal(stats.gateCount,3);
assert.equal(stats.delayedGateCount,1);
assert.equal(core.stats([r]).arrivalCount,0,'Estimated arrival is excluded from actual history');
assert.equal(core.stats([]).meanDeparture,null,'Missing data is not zero delay');
assert.equal(core.outlook(r,Date.parse('2026-10-26T14:00:00Z')).title,'Already departed');
assert.equal(core.outlook({...r,actualDeparture:null},Date.parse('2026-10-26T14:00:00Z')).title,'Departure outcome needs checking');
assert.equal(core.outlook({...r,actualDeparture:null},Date.parse('2026-10-26T10:00:00Z')).title,'Insufficient evidence for a percentage');
const future=core.select(data,'SK935','CPH','2026-10-27').current[0];
assert.match(core.outlook(future,Date.parse('2026-10-26T14:00:00Z')).title,/10 minutes after schedule/);
assert.equal(core.outlook(future,Date.parse('2026-10-27T14:00:00Z')).title,'Departure outcome needs checking','Old estimates do not become current status');

const old={...r,checkedAt:'2026-10-26T12:00:00Z',estimatedArrival:'2026-10-26T23:00:00Z'};
assert.equal(core.select({...data,records:[r,old]},'SK935','CPH',r.date).current[0].estimatedArrival,r.estimatedArrival,'Latest manual observation wins, without duplicate flights');
for(const record of publishedData.records){
  assert.ok(core.validDate(record.date));
  assert.ok(core.instant(record.checkedAt)!==null);
  assert.equal(new Intl.DateTimeFormat('en-CA',{timeZone:record.originTimeZone,year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date(record.scheduledDeparture)),record.date,'Departure date is local to origin');
  assert.ok(new URL(record.sourceUrl).hostname.endsWith('flightradar24.com'));
}
// Copenhagen changed from summer time on 25 October; San Francisco did not.
assert.equal(new Intl.DateTimeFormat('en-GB',{timeZone:'Europe/Copenhagen',hour:'2-digit',minute:'2-digit',hour12:false}).format(new Date(data.records.find(r=>r.date==='2026-10-24').scheduledDeparture)),'12:45');
assert.equal(new Intl.DateTimeFormat('en-GB',{timeZone:'Europe/Copenhagen',hour:'2-digit',minute:'2-digit',hour12:false}).format(new Date(r.scheduledDeparture)),'12:40');
console.log('Flight tracker: published snapshot validation, selection isolation, timestamp arithmetic, event semantics, missing data, deduplication, and forecast abstention passed.');
