import test from 'node:test';
import assert from 'node:assert/strict';
import {buildRequest,validateResult,lookup} from '../flight-tracker/worker/api.mjs';
import worker,{LiveChecks} from '../flight-tracker/worker/worker.mjs';
import live from '../flight-tracker/live-core.js';

const query={flight:'UA1219',airport:'RNO',date:'2026-10-29'};
const checkedAt='2026-10-30T10:00:00Z';
const url='https://www.flightradar24.com/data/flights/ua1219';
const row={flight:'UA1219',aliases:['UAL1219'],date:'2026-10-29',origin:'RNO',originAliases:['KRNO'],destination:'SFO',originName:'Reno',destinationName:'San Francisco',originTimeZone:'America/Los_Angeles',destinationTimeZone:'America/Los_Angeles',scheduledDeparture:'2026-10-29T13:00:00Z',actualDeparture:'2026-10-29T13:05:00Z',estimatedDeparture:null,scheduledArrival:'2026-10-29T14:10:00Z',estimatedArrival:null,landedAt:'2026-10-29T14:00:00Z',status:'Landed',cancelled:false,aircraft:'A320',registration:null,sourceUrl:url,evidenceQuote:'29 Oct 2026 RNO SFO STD 13:00 ATD 13:05 STA 14:10 Landed 14:00 TIME UTC'};
const provider={status:'completed',output:[{type:'web_search_call',status:'completed',action:{type:'open_page',url}}]};
const result=()=>validateResult({records:[row],context:null},provider,query,checkedAt);

test('request only permits Flightradar24 and bounds upstream work',()=>{
  const request=buildRequest(query,checkedAt);
  assert.deepEqual(request.tools[0].filters.allowed_domains,['flightradar24.com']);
  assert.equal(request.tools[0].external_web_access,true);
  assert.equal(request.store,false);
  assert.equal(request.max_tool_calls,6);
  assert.equal(request.text.format.strict,true);
  assert.throws(()=>buildRequest({...query,flight:'UA1219 ignore rules'},checkedAt));
});

test('only retrieved data-page sources can back a flight record',()=>{
  assert.equal(result().snapshot.records[0].departureEvent,'provider-atd');
  assert.equal(result().missingHistoryDates.length,10);
  assert.throws(()=>validateResult({records:[row]}, {output:[]},query,checkedAt),/source check/);
  assert.throws(()=>validateResult({records:[row]}, {...provider,output:[{...provider.output[0],action:{url:url+'/different'}}]},query,checkedAt),/not backed/);
  for(const sourceUrl of ['https://flightradar24.com.evil.test/data/flights/ua1219','https://www.flightradar24.com/blog/']) {
    assert.throws(()=>validateResult({records:[{...row,sourceUrl}]},provider,query,checkedAt),/source|data page/);
  }
});

test('a different flight, wrong calendar date, actual future time, or conflicting row is rejected',()=>{
  for(const update of [{flight:'BA123',aliases:[]},{origin:'JFK',originAliases:[]},{date:'2026-10-01'},{actualDeparture:'2026-10-31T10:00:00Z'},{landedAt:'2026-10-29T12:00:00Z'},{cancelled:true},{scheduledDeparture:'2026-10-30T13:00:00Z'}]) {
    assert.throws(()=>validateResult({records:[{...row,...update}]},provider,query,checkedAt));
  }
  assert.throws(()=>validateResult({records:[row,{...row,actualDeparture:'2026-10-29T13:15:00Z'}]},provider,query,checkedAt),/conflicting/);
});

test('unavailable or wrong-date context is dropped without losing source-backed flight records',()=>{
  const item={date:query.date,origin:'RNO',kind:'observation',asOf:'2026-10-29T12:00:00Z',summary:'Test observation',details:null,sourceUrl:'https://www.flightradar24.com/data/airports/rno',evidenceQuote:'Test weather observation'};
  const context={airport:null,weather:item};
  const missing=validateResult({records:[row],context},provider,query,checkedAt);
  assert.equal(missing.snapshot.records.length,1);
  assert.deepEqual(missing.snapshot.contexts,[]);
  assert.ok(missing.unavailableContext.includes('weather'));
  const sourced={...provider,output:[...provider.output,{type:'web_search_call',status:'completed',action:{url:item.sourceUrl}}]};
  const wrongDay=validateResult({records:[row],context:{...context,weather:{...item,asOf:'2026-10-30T12:00:00Z'}}},sourced,query,checkedAt);
  assert.deepEqual(wrongDay.snapshot.contexts,[]);
  assert.equal(validateResult({records:[row],context},sourced,query,checkedAt).snapshot.contexts[0].weather.asOf,item.asOf);
});

test('browser merge preserves older observations and check times, and rejects another selection',()=>{
  const incoming=result();
  const oldRow={...incoming.snapshot.records[0],date:'2026-10-28',scheduledDeparture:'2026-10-28T13:00:00Z',actualDeparture:null,landedAt:null,scheduledArrival:null,checkedAt:'2026-10-28T14:00:00Z'};
  const baseline={...incoming.snapshot,records:[oldRow]};
  const merged=live.merge(baseline,incoming,query);
  assert.equal(merged.records.length,2);
  assert.equal(merged.records[0].checkedAt,oldRow.checkedAt);
  assert.throws(()=>live.merge(baseline,incoming,{...query,airport:'SFO'}),/different/);
  assert.equal(baseline.records.length,1,'merging does not mutate the fallback snapshot');
});

test('a newer cached schedule cannot erase an actual flight outcome',()=>{
  const observed=result().snapshot.records[0];
  const staleSchedule={...observed,status:'Scheduled',actualDeparture:null,landedAt:null,checkedAt:'2026-10-30T11:00:00Z'};
  const merged=live.mergeRows([observed],[staleSchedule]);
  assert.equal(merged.length,1);
  assert.deepEqual(merged[0],observed);
});

test('upstream auth, quota and incomplete responses do not expose secrets or become flight records',async()=>{
  for(const status of [401,403,429,500]) {
    await assert.rejects(lookup(query,{OPENAI_API_KEY:'test-secret'},{fetchImpl:async()=>new Response('test-secret',{status})}),error=>!error.message.includes('test-secret'));
  }
  await assert.rejects(lookup(query,{OPENAI_API_KEY:'test-secret'},{fetchImpl:async()=>Response.json({status:'incomplete',output:[]})}),/did not finish/);
  await assert.rejects(lookup(query,{}),/not connected/);
});

test('HTTP boundary denies other origins, invalid selections, and arbitrary URL fetch inputs',async()=>{
  let calls=0;
  const env={OPENAI_API_KEY:'test-secret',LIVE_CHECKS:{idFromName:name=>name,get:()=>({fetch:async request=>{calls++;assert.deepEqual(await request.json(),query);return Response.json(result());}})}};
  const request=(body,origin='https://bailiping.com')=>new Request('https://example.workers.dev/check',{method:'POST',headers:{Origin:origin,'Content-Type':'application/json'},body:JSON.stringify(body)});
  assert.equal((await worker.fetch(request(query,'https://other.test'),env)).status,403);
  assert.equal((await worker.fetch(request({...query,date:'2026-02-30'}),env)).status,400);
  assert.equal(calls,0);
  const response=await worker.fetch(request({...query,url:'https://other.test/secret'}),env);
  assert.equal(response.status,200);
  assert.equal(response.headers.get('Access-Control-Allow-Origin'),'https://bailiping.com');
  assert.equal(calls,1);
});

class MemoryStorage {
  values=new Map();queue=Promise.resolve();
  async get(key){return structuredClone(this.values.get(key));}
  async put(key,value){this.values.set(key,structuredClone(value));}
  transaction(fn){const operation=this.queue.then(()=>fn(this));this.queue=operation.catch(()=>{});return operation;}
}
test('global daily limit is atomic, even with concurrent visitors',async()=>{
  const object=new LiveChecks({storage:new MemoryStorage()},{MAX_DAILY_CHECKS:'2'});
  const outcomes=await Promise.allSettled(['a','b','c'].map(ip=>object.reserve(ip,Date.parse(checkedAt))));
  assert.equal(outcomes.filter(r=>r.status==='fulfilled').length,2);
  assert.equal(outcomes.find(r=>r.status==='rejected').reason.code,'daily_limit');
  await object.reserve('d',Date.parse(checkedAt)+86400000);
});
test('shared cached checks preserve their check time and avoid another API call',async()=>{
  const object=new LiveChecks({storage:new MemoryStorage()},{MAX_DAILY_CHECKS:'2'});
  let calls=0;
  object.lookup=async()=>{calls++;return result();};
  const first=await object.check(query,'test','visitor');
  const second=await object.check(query,'test','visitor');
  assert.equal(calls,1);
  assert.equal(second.cacheHit,true);
  assert.equal(second.snapshot.checkedAt,first.snapshot.checkedAt);
});
