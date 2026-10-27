'use strict';

// Runs inside Codex's tool runner, with tools passed in. No provider API or key.
// See AGENTS.md for the exact invocation. Node can also load the pure parser for tests.
const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
const HOSTS = ['flightradar24.com','www.flightradar24.com','data.flightradar24.com','data-live.flightradar24.com','free.flightradar24.com'];
const DAY = 86400000;
function assert(ok, message) { if (!ok) throw new Error(message); }
function quote(value) { return "'" + String(value).replaceAll("'", "'\\''") + "'"; }
function dayAt(value, zone) { return new Intl.DateTimeFormat('en-CA', {timeZone:zone, year:'numeric', month:'2-digit', day:'2-digit'}).format(new Date(value)); }
function shift(day, count) { return new Date(Date.parse(day + 'T12:00:00Z') + count * DAY).toISOString().slice(0,10); }
function label(day) { const [y,m,d] = day.split('-'); return `${d} ${MONTHS[Number(m)-1]} ${y}`; }
function pageUrl(value) {
  // The Codex JavaScript tool runner has ECMAScript globals, but no browser URL class.
  const match = String(value).match(/^https:\/\/([^/?#]+)(\/[^?#]*)?(?:[?#].*)?$/);
  assert(match && HOSTS.includes(match[1].toLowerCase()), 'Expected a public Flightradar24 page URL.');
  return {hostname:match[1].toLowerCase(), pathname:match[2] || '/'};
}
function textOf(result) {
  if (typeof result === 'string') return result;
  if (Array.isArray(result.content)) return result.content.filter(item => item.type === 'text').map(item => item.text).join('\n');
  throw new Error('Unrecognized web-tool response; no observations were changed.');
}
function sections(result) {
  const text = textOf(result);
  const starts = [...text.matchAll(/^.*\((https:\/\/[^\s)]+)\)\s*$/gm)];
  return starts.map((match, index) => ({url:match[1], text:text.slice(match.index, starts[index+1]?.index ?? text.length)})).filter(section => {
    try { const url = pageUrl(section.url); return /^\/data\/(flights|aircraft)\//.test(url.pathname); }
    catch (_) { return false; }
  });
}
function clean(text) {
  return text.replace(/【[^†\n]+†([^】]+)】/g, '$1').replace(/\[([^\]]+)\]\([^)]*\)/g, '$1').replace(/^L\d+:\s*/gm, '').replace(/\*\*/g, '');
}
function iso(value) { return new Date(value).toISOString().replace('.000Z', 'Z'); }
function timeOn(day, clock) {
  if (!/^\d{2}:\d{2}$/.test(clock)) return null;
  const value = Date.parse(`${day}T${clock}:00Z`);
  assert(Number.isFinite(value) && Number(clock.slice(0,2)) < 24 && Number(clock.slice(3)) < 60 && iso(value).slice(0,10) === day, `Invalid source time: ${day} ${clock}`);
  return value;
}
function onOrAfter(day, clock, anchor) {
  let value = timeOn(day, clock);
  if (value === null) return null;
  while (value < anchor) value += DAY;
  return value;
}
function nearDeparture(day, clock, scheduled) {
  let value = timeOn(day, clock);
  if (value === null) return null;
  // Handle midnight rollover; ambiguous very-early departures need manual review.
  if (value - scheduled > 22 * 3600000) value -= DAY;
  if (value - scheduled < -2 * 3600000) value += DAY;
  return value;
}
function score(row) { return (row.landedAt ? 100 : 0) + (row.actualDeparture ? 50 : 0) + (row.cancelled ? 100 : 0) + (row.estimatedDeparture ? 10 : 0); }

function parseHistory(result, config) {
  const records = [], skipped = [];
  for (const section of sections(result)) {
    const text = clean(section.text);
    if (!/TIME\s+UTC/.test(text)) continue; // Search snippets without a timezone need opening first.
    const source = pageUrl(section.url);
    const flightPage = source.pathname.match(/^\/data\/flights\/([^/]+)\/?$/i);
    const aircraftPage = source.pathname.match(/^\/data\/aircraft\/([^/]+)\/?$/i);
    const identities = [config.flight, ...(config.aliases || [])];
    if (flightPage && !identities.includes(flightPage[1].toUpperCase())) continue;
    const aircraft = text.match(/^AIRCRAFT[ \t]+([^\n]+)/m)?.[1]?.trim() || null;
    for (const line of text.split('\n')) {
      const match = line.match(/^\s*(.*?)\s+(\d{1,2})\s+([A-Z][a-z]{2})\s+(\d{4})\s+(\d+:\d{2}|-)\s+(.+?)\s+STD\s+(\d{2}:\d{2}|—|-)\s+ATD\s+(\d{2}:\d{2}|—|-)\s+STA\s+(\d{2}:\d{2}|—|-)\s+FROM\s+(.+?)\s+\(([A-Z]{3})\)\s+TO\s+(.+?)\s+\(([A-Z]{3})\)\s*$/);
      if (!match) continue;
      const [,prefix,d,m,y,duration,statusText,std,atd,sta,originName,origin,destinationName,destination] = match;
      if (origin !== config.origin || destination !== config.destination) continue;
      if (aircraftPage && !identities.includes(prefix.trim().toUpperCase())) continue;
      const month = MONTHS.indexOf(m) + 1;
      if (!month || std === '—' || std === '-') { skipped.push('Missing scheduled departure: ' + line); continue; }
      const utcDay = `${y}-${String(month).padStart(2,'0')}-${d.padStart(2,'0')}`;
      const scheduled = timeOn(utcDay, std);
      const date = dayAt(scheduled, config.originTimeZone);
      if (date < config.from || date > config.to) continue;
      const actual = nearDeparture(utcDay, atd, scheduled);
      const landedClock = statusText.match(/^Landed\s+(\d{2}:\d{2})/i)?.[1];
      const estimatedClock = statusText.match(/^Estimated departure\s+(\d{2}:\d{2})/i)?.[1];
      const estimatedArrivalClock = statusText.match(/^Estimated\s+(\d{2}:\d{2})/i)?.[1];
      const landed = landedClock ? onOrAfter(utcDay, landedClock, actual ?? scheduled) : null;
      if (landed !== null && actual !== null && duration !== '-') {
        const [hours, minutes] = duration.split(':').map(Number);
        if (Math.abs((landed - actual)/60000 - (hours*60 + minutes)) > 2) { skipped.push(`Ambiguous time rollover for ${date}: ${line}`); continue; }
      }
      const registration = aircraftPage ? aircraftPage[1].toUpperCase() : /^(?:Not available|-)$/.test(prefix.trim()) ? null : prefix.trim();
      const row = {
        flight:config.flight, aliases:config.aliases || [], date, origin, originAliases:config.originAliases || [], destination,
        originName:config.originName || originName, destinationName:config.destinationName || destinationName,
        originTimeZone:config.originTimeZone, destinationTimeZone:config.destinationTimeZone,
        scheduledDeparture:iso(scheduled), actualDeparture:actual === null ? null : iso(actual),
        estimatedDeparture:estimatedClock ? iso(nearDeparture(utcDay, estimatedClock, scheduled)) : null,
        scheduledArrival:sta === '—' || sta === '-' ? null : iso(onOrAfter(utcDay, sta, scheduled)),
        landedAt:landed === null ? null : iso(landed),
        estimatedArrival:estimatedArrivalClock ? iso(onOrAfter(utcDay, estimatedArrivalClock, actual ?? scheduled)) : null,
        departureEvent:'provider-atd', status:landedClock ? 'Landed' : estimatedClock ? 'Estimated departure' : estimatedArrivalClock ? 'Estimated arrival' : statusText.trim(),
        aircraft, registration, checkedAt:config.checkedAt, sourceUrl:section.url,
      };
      if (/cancelled|canceled/i.test(statusText)) row.cancelled = true;
      records.push(row);
    }
  }
  return {records, skipped};
}
function mergeEvidence(groups) {
  const rows = new Map();
  for (const row of groups.flat()) {
    const key = [row.flight,row.origin,row.destination,row.date].join('|');
    const old = rows.get(key);
    if (old && score(old) === score(row) && old.actualDeparture && row.actualDeparture) {
      assert(old.actualDeparture === row.actualDeparture && old.landedAt === row.landedAt, `Conflicting completed flight records for ${row.date}; inspect the sources before importing.`);
    }
    if (!old || score(row) > score(old) || (score(row) === score(old) && row.aircraft && !old.aircraft)) rows.set(key, row);
  }
  return [...rows.values()].sort((a,b) => a.date.localeCompare(b.date));
}

async function checkFlight(tools, options) {
  const repo = options.repo || '/home/bailiping/Desktop/bailiping.github.io';
  const flight = String(options.flight || '').toUpperCase().replace(/[\s-]+/g, '');
  const airport = String(options.airport || '').trim().toUpperCase();
  const destination = String(options.destination || '').trim().toUpperCase();
  const date = options.date;
  assert(/^[A-Z0-9]{2,3}\d{1,5}[A-Z]?$/.test(flight) && /^[A-Z]{3,4}$/.test(airport) && /^[A-Z]{3}$/.test(destination) && /^\d{4}-\d{2}-\d{2}$/.test(date), 'Supply flight, airport, destination, and local date.');
  const base = `node flight-tracker/update-observations.cjs --flight ${quote(flight)} --airport ${quote(airport)} --date ${quote(date)}`;
  const starter = await tools.exec_command({cmd:`${base} --destination ${quote(destination)} --template`, workdir:repo, max_output_tokens:2500});
  assert(starter.exit_code === 0, starter.output || 'Could not prepare flight metadata.');
  const metadata = JSON.parse(starter.output).records[0];
  metadata.originTimeZone = options.originTimeZone || metadata.originTimeZone;
  metadata.destinationTimeZone = options.destinationTimeZone || metadata.destinationTimeZone;
  assert(metadata.originTimeZone && metadata.destinationTimeZone, 'For a new airport, supply originTimeZone and destinationTimeZone as confirmed IANA timezone names.');
  const clock = await tools.clock__curr_time({});
  const checkedAt = clock.current_time.replace(' UTC', 'Z').replace(' ', 'T');
  const anchor = [date, dayAt(checkedAt, metadata.originTimeZone)].sort()[0];
  const days = Array.from({length:10}, (_,i) => shift(anchor, i-10));
  const config = {...metadata, from:days[0], to:date, checkedAt};
  const source = metadata.sourceUrl;
  const responses = [];
  responses.push(await tools.web__run({open:[{ref_id:source, lineno:85},{ref_id:source + '/', lineno:85}], response_length:'long'}));
  responses.push(await tools.web__run({search_query:[{q:`"${metadata.flight}" "${metadata.origin}" "${label(date)}"`, domains:['flightradar24.com']},{q:`"${metadata.flight}" "${label(days[0])}" "${metadata.origin}"`, domains:['flightradar24.com']}], response_length:'long'}));
  const opened = new Set([source, source + '/']);
  async function openDiscoveries() {
    const discovered = [...new Set(responses.flatMap(sections).filter(part => /\bSTD\b/.test(part.text) && part.text.includes(metadata.origin) && part.text.toUpperCase().includes(metadata.flight)).map(part => part.url))].filter(url => !opened.has(url)).slice(0,12);
    if (!discovered.length) return;
    discovered.forEach(url => opened.add(url));
    for (let start = 0; start < discovered.length; start += 3) {
      responses.push(await tools.web__run({open:discovered.slice(start,start+3).map(ref_id => ({ref_id, lineno:ref_id.includes('/aircraft/') ? 107 : 85})), response_length:'long'}));
    }
  }
  await openDiscoveries();
  let parsed = responses.map(result => parseHistory(result, config));
  let records = mergeEvidence(parsed.map(result => result.records));
  // Search the actual missing dates, not just the flight number's latest route.
  const missing = days.filter(day => !records.some(row => row.date === day && row.actualDeparture));
  for (let start = 0; start < missing.length; start += 4) {
    const batch = missing.slice(start, start+4);
    responses.push(await tools.web__run({search_query:batch.map(day => ({q:`"${metadata.flight}" "${metadata.origin}" ("${label(shift(day,-1))}" OR "${label(day)}" OR "${label(shift(day,1))}")`, domains:['flightradar24.com']})), response_length:'long'}));
  }
  await openDiscoveries();
  parsed = responses.map(result => parseHistory(result, config));
  records = mergeEvidence(parsed.map(result => result.records));
  assert(records.some(row => row.date === date), 'Flightradar24 did not expose the requested flight/date. No snapshot was changed.');
  assert(options.allowNoHistory || records.some(row => days.includes(row.date)), 'No earlier records were retrieved. Inspect Flightradar24 history before publishing an empty historical panel; use allowNoHistory only when absence has been checked.');
  const inputFile = `/tmp/flight-check-${flight}-${Date.now()}.json`;
  const input = JSON.stringify({records});
  const saved = await tools.exec_command({cmd:`printf '%s\n' ${quote(input)} > ${quote(inputFile)}`, workdir:repo, max_output_tokens:1000});
  assert(saved.exit_code === 0, 'Could not save the checked observations.');
  const applyCommand = `${base} --input ${quote(inputFile)}`;
  const applied = await tools.exec_command({cmd:applyCommand + (options.dryRun ? ' --dry-run' : ''), workdir:repo, max_output_tokens:3500});
  assert(applied.exit_code === 0, applied.output || 'The observation validation failed.');
  return {result:JSON.parse(applied.output), inputFile, applyCommand, sourcePages:[...new Set(records.map(row => row.sourceUrl))], skipped:parsed.flatMap(result => result.skipped), retrievedDates:records.map(row => row.date), missingHistoryDates:days.filter(day => !records.some(row => row.date === day)), note:'Searches public Flightradar24 pages through Codex. Availability depends on exposed history; no API or key is used. Review, test, publish, and verify the live page.'};
}

module.exports = {checkFlight, parseHistory, mergeEvidence, sections};
