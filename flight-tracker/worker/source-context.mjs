import core from '../tracker-core.js';

const MAX_PAGE_BYTES = 2500000;
const HOUR = 3600000;
const sourceOrigin = 'https://www.flightradar24.com';
const dayAt = (time,zone) => new Intl.DateTimeFormat('en-CA',{timeZone:zone,year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date(time));
const iso = time => new Date(time).toISOString().replace(/\.\d{3}Z$/,'Z');
const finite = value => typeof value==='number' && Number.isFinite(value);
const shortText = (value,max=120) => typeof value==='string' && value.trim() && value.length<=max ? value.trim() : null;

export function airportPage(html) {
  // Read only the page's JSON payload; never evaluate its scripts or forward its
  // dispatcher, session fields, account information, or arbitrary URLs.
  const match=html.match(/\bdata-page="([^"]+)"/);
  if(!match) return null;
  const decoded=match[1].replace(/&quot;/g,'"').replace(/&#039;|&apos;/g,"'").replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&amp;/g,'&');
  const props=JSON.parse(decoded).props;
  if(!props || typeof props!=='object') return null;
  return {airport:props.airport,meta:props.meta,flights:props.flights,statistics:props.statistics,latestMetar:props.latestMetar,weatherHistory:props.weatherHistory};
}

function matchesAirport(page,row) {
  const airport=page?.airport;
  const aliases=[row.origin,...(row.originAliases || [])];
  return airport && /^[A-Z]{3}$/.test(airport.iata) && /^[A-Z]{4}$/.test(airport.icao) &&
    aliases.some(code=>code===airport.iata || code===airport.icao) && airport.timezone===row.originTimeZone;
}

function timeWindow(row,checkedAt) {
  const departure=core.instant(row.scheduledDeparture),checked=core.instant(checkedAt);
  if(departure===null || checked===null) return null;
  return {checked,cutoff:Math.min(departure,checked)};
}

export function departureEstimate(page,row,checkedAt,sourceUrl) {
  if(!matchesAirport(page,row) || row.actualDeparture || row.cancelled || !Array.isArray(page.flights)) return null;
  const scheduled=core.instant(row.scheduledDeparture),checked=core.instant(checkedAt),asOf=core.instant(page.meta?.dateModified);
  if(scheduled===null || checked===null || asOf===null || asOf>checked || checked-asOf>HOUR || page.meta?.outsideAllowance) return null;
  const aliases=[row.flight,...(row.aliases || [])];
  const matches=page.flights.filter(flight=>
    (aliases.includes(flight.flightNumber) || aliases.includes(flight.callsign)) &&
    [flight.endpoint?.iata,flight.endpoint?.icao].includes(row.destination) &&
    Number.isInteger(flight.scheduledTime) && flight.scheduledTime*1000===scheduled &&
    flight.status?.name==='estimated' && Number.isInteger(flight.estimatedTime) &&
    Math.abs(flight.estimatedTime*1000-scheduled)<=36*HOUR);
  if(matches.length!==1) return null;
  return {time:iso(matches[0].estimatedTime*1000),source:{sourceUrl,asOf:iso(asOf),checkedAt,flight:matches[0].flightNumber,scheduledDeparture:iso(scheduled)}};
}

export function departureContext(page,row,checkedAt,sourceUrl) {
  if(!matchesAirport(page,row)) return null;
  const window=timeWindow(row,checkedAt),asOf=core.instant(page.meta?.dateModified);
  // The airport's current totals are not an archive of earlier conditions.
  if(!window || asOf===null || asOf>window.cutoff || window.checked-asOf>HOUR ||
      dayAt(asOf,row.originTimeZone)!==row.date || page.meta?.outsideAllowance) return null;
  const live=page.statistics?.live,today=page.statistics?.today,parts=[];
  if(finite(live?.averageDelayMin) && live.averageDelayMin>=0 && live.averageDelayMin<1440) parts.push(`Average departure delay: ${live.averageDelayMin} minutes.`);
  if(Number.isInteger(today?.delayed) && today.delayed>=0 && Number.isInteger(today?.cancelled) && today.cancelled>=0) parts.push(`Today's departure board: ${today.delayed} flights reported delayed; ${today.cancelled} cancelled.`);
  if(!parts.length) return null;
  return {kind:'observation',asOf:iso(asOf),checkedAt,sourceUrl,summary:parts.join(' '),details:'Figures for all flights at this airport, as reported by Flightradar24.',sourceFields:{averageDelayMin:live?.averageDelayMin,delayed:today?.delayed,cancelled:today?.cancelled}};
}

export function weatherContext(page,row,checkedAt,sourceUrl) {
  if(!matchesAirport(page,row)) return null;
  const window=timeWindow(row,checkedAt);
  if(!window) return null;
  const reports=[page.latestMetar,...(Array.isArray(page.weatherHistory)?page.weatherHistory:[])];
  const report=reports.filter(item=>{
    if(!item || !Number.isInteger(item.timeUtc) || !shortText(item.metar,1200)) return false;
    const stamp=item.metar.match(new RegExp(`^(?:(?:METAR|SPECI) )?${page.airport.icao}\\s+(\\d{2})(\\d{2})(\\d{2})Z\\b`));
    if(!stamp) return false;
    const asOf=item.timeUtc*1000;
    const date=new Date(asOf);
    if(date.getUTCDate()!==Number(stamp[1]) || date.getUTCHours()!==Number(stamp[2]) || date.getUTCMinutes()!==Number(stamp[3])) return false;
    return asOf<=window.cutoff && window.cutoff-asOf<=3*HOUR && dayAt(asOf,row.originTimeZone)===row.date;
  }).sort((a,b)=>b.timeUtc-a.timeUtc)[0];
  if(!report) return null;
  const parts=[];
  if(shortText(report.condition?.label,60)) parts.push(report.condition.label);
  if(finite(report.temperature) && ['°C','°F'].includes(report.temperatureUnit)) parts.push(`${report.temperature} ${report.temperatureUnit}`);
  if(finite(report.wind?.speed) && report.wind.speed>=0 && ['kts','km/h','m/s'].includes(report.wind.speedUnit)) parts.push(`wind ${report.wind.speed} ${report.wind.speedUnit}`);
  if(finite(report.visibility) && report.visibility>=0 && ['SM','km','m'].includes(report.visibilityUnit)) parts.push(`visibility ${report.visibility} ${report.visibilityUnit}`);
  return {kind:'observation',asOf:iso(report.timeUtc*1000),checkedAt,sourceUrl,summary:parts.length?parts.join(' · '):'Airport weather observation available.',report:report.metar,details:'METAR weather observation recorded before the selected departure.',evidenceQuote:report.metar};
}

async function readPage(url,fetchImpl) {
  const response=await fetchImpl(url,{headers:{Accept:'text/html','User-Agent':'BailipingFlightTracker/1.0 (+https://bailiping.com/flight-tracker/)','Cache-Control':'no-cache'},redirect:'manual',signal:AbortSignal.timeout(12000)});
  if(!response.ok || !response.headers.get('Content-Type')?.includes('text/html') || Number(response.headers.get('Content-Length'))>MAX_PAGE_BYTES || !response.body) return null;
  const reader=response.body.getReader(),decoder=new TextDecoder();
  let bytes=0,html='';
  try {
    while(true) {
      const {value,done}=await reader.read();
      if(done) break;
      bytes+=value.byteLength;
      if(bytes>MAX_PAGE_BYTES) { await reader.cancel(); return null; }
      html+=decoder.decode(value,{stream:true});
    }
    html+=decoder.decode();
    return airportPage(html);
  } finally { reader.releaseLock(); }
}

export async function addPublicContext(result,query,{fetchImpl=fetch,now=()=>new Date().toISOString().replace(/\.\d{3}Z$/,'Z')}={}) {
  const row=result.snapshot.records.find(item=>item.date===query.date);
  if(!row || core.instant(row.scheduledDeparture)===null) return result;
  const airport=[row.origin,...(row.originAliases || [])].find(code=>/^[A-Z]{3}$/.test(code));
  if(!airport) return result;
  const base=`${sourceOrigin}/data/airports/${airport.toLowerCase()}`;
  const pages=await Promise.allSettled(['departures','weather'].map(path=>readPage(`${base}/${path}`,fetchImpl)));
  const checkedAt=now();
  if(pages[0].status==='fulfilled') {
    try {
      const estimate=departureEstimate(pages[0].value,row,checkedAt,`${base}/departures`);
      if(estimate) {
        row.estimatedDeparture=estimate.time;
        row.departureEstimateSource=estimate.source;
        row.status='Estimated departure';
      }
    } catch { /* Retain the verified flight record if no matching estimate exists. */ }
  }
  const context={...(result.snapshot.contexts[0] || {flight:row.flight,aliases:row.aliases,origin:row.origin,originAliases:row.originAliases,date:row.date})};
  for(const [index,name,builder,path] of [[0,'airport',departureContext,'departures'],[1,'weather',weatherContext,'weather']]) {
    if(pages[index].status!=='fulfilled') continue;
    try {
      const item=builder(pages[index].value,row,checkedAt,`${base}/${path}`);
      if(item) context[name]=item;
    } catch { /* Missing or changed source data must not invalidate flight records. */ }
  }
  if(context.airport || context.weather) result.snapshot.contexts=[context];
  result.unavailableContext=['airport','weather'].filter(name=>!context[name]);
  core.validateSnapshot(result.snapshot);
  return result;
}
