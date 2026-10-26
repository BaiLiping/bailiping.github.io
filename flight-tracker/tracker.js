(function () {
  "use strict";
  var core = window.FlightTracker;
  var node = function (id) { return document.getElementById(id); };
  var form = node("flight-form"), flightField = node("flight-number"), airportField = node("departure-airport"), dateField = node("flight-date");
  var data = null, selection = null, loading = false;
  function esc(value) { return String(value == null ? "" : value).replace(/[&<>"']/g, function (c) { return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]; }); }
  function formatDate(value, short) {
    if (!core.validDate(value)) return "—";
    return new Intl.DateTimeFormat("en-GB", { day:"numeric", month:"short", year:short ? undefined : "numeric", timeZone:"UTC" }).format(new Date(value + "T12:00:00Z"));
  }
  function time(value, zone, full) {
    if (core.instant(value) === null) return "—";
    var options = { hour:"2-digit", minute:"2-digit", hour12:false, timeZone:zone || "UTC", timeZoneName:"short" };
    if (full) { options.day="numeric"; options.month="short"; options.year="numeric"; }
    try { return new Intl.DateTimeFormat("en-GB", options).format(new Date(value)); }
    catch (_) { return time(value, "UTC", full); }
  }
  function delta(value) { if (value === null || !Number.isFinite(value)) return "—"; var rounded=Math.round(value*10)/10; return (rounded>0 ? "+" : "") + rounded + " min"; }
  function metric(value, label, note) { return '<div class="metric"><strong>'+esc(value)+'</strong><span>'+esc(label)+'</span><small>'+esc(note)+'</small></div>'; }
  function sourceUrl(value) {
    try { var url = new URL(value); return url.protocol === "https:" && (url.hostname === "www.flightradar24.com" || url.hostname === "flightradar24.com") ? url.href : "https://www.flightradar24.com/data"; }
    catch (_) { return "https://www.flightradar24.com/data"; }
  }
  function link(value, label) { return '<a href="'+esc(sourceUrl(value))+'" target="_blank" rel="noopener noreferrer">'+esc(label)+' ↗</a>'; }
  function empty(title, reason) { return '<div class="empty"><strong>'+esc(title)+'</strong><p>'+esc(reason)+'</p></div>'; }
  function setBusy(value) {
    loading=value; node("search-button").disabled=value; node("refresh").disabled=value;
    node("results").setAttribute("aria-busy",String(value));
  }
  function formError(reason, field) { node("form-error").textContent=reason; node("form-error").hidden=false; if(field) field.focus(); }
  function drawHistory(view) {
    var rows=view.history, records=rows.flatMap(function (day) { return day.flights; }), stats=core.stats(records);
    node("history-period").textContent=formatDate(rows[0].date)+" – "+formatDate(rows[9].date)+" · "+stats.count+" saved flight records · "+view.airport+" departures";
    node("history-metrics").innerHTML=metric(stats.count,"Flights recorded","Across ten calendar days")+
      metric(delta(stats.meanDeparture),"Mean ATD − STD",stats.departureCount+" actual departure entries")+
      metric(delta(stats.meanArrival),"Mean landing − STA",stats.arrivalCount+" completed flights");
    var points=rows.map(function(day) { return {date:day.date, value:core.stats(day.flights).meanDeparture}; });
    var scale=Math.max(1,...points.map(function(p) { return Math.abs(p.value || 0); }));
    node("history-chart").innerHTML='<p class="chart-note">Reported actual departure minus scheduled departure. These values may include taxi time; they are not a verified gate-delay rate.</p>'+points.map(function(p) {
      var width=p.value===null ? 0 : Math.max(2,100*Math.abs(p.value)/scale);
      return '<div class="chart-row"><span class="chart-label">'+esc(formatDate(p.date,true))+'</span><div class="chart-track"><span class="chart-fill '+(p.value===null ? 'missing' : p.value>0 ? 'late' : 'early')+'" style="width:'+width+'%"></span></div><span class="chart-value">'+esc(p.value===null ? 'No record' : delta(p.value))+'</span></div>';
    }).join("");
    node("history-chart").setAttribute("aria-label",points.map(function(p) { return formatDate(p.date,true)+": "+(p.value===null ? "no record" : delta(p.value)); }).join("; "));
    node("history-rows").innerHTML=rows.map(function(day) {
      if(!day.flights.length) return '<tr><td>'+esc(formatDate(day.date,true))+'</td><td class="missing" colspan="7">Not checked / no saved record</td></tr>';
      return day.flights.map(function(r) {
        return '<tr><td>'+link(r.sourceUrl,formatDate(r.date,true))+'</td><td>'+esc(r.origin+" → "+r.destination)+'</td><td class="num">'+esc(time(r.scheduledDeparture,r.originTimeZone))+'</td><td class="num">'+esc(time(r.actualDeparture,r.originTimeZone))+'</td><td class="num">'+esc(core.difference(r.scheduledDeparture,r.actualDeparture)===null ? r.status : delta(core.difference(r.scheduledDeparture,r.actualDeparture)))+'</td><td class="num">'+esc(time(r.scheduledArrival,r.destinationTimeZone))+'</td><td class="num">'+esc(time(r.landedAt,r.destinationTimeZone))+'</td><td class="num">'+esc(delta(core.difference(r.scheduledArrival,r.landedAt)))+'</td></tr>';
      }).join("");
    }).join("");
  }
  function timeRow(label,value,zone) { return '<div><dt>'+esc(label)+'</dt><dd>'+esc(time(value,zone,true))+'</dd></div>'; }
  function drawCurrent(view) {
    node("current-period").textContent="Departure "+formatDate(view.date)+" · times local to each airport, with timezone labels.";
    if(!view.current.length) {
      node("current-flights").innerHTML=empty("No saved observation for this flight and date","This does not mean the flight is cancelled or on time. A manual check is needed before a record can be shown here.");
      node("provider-updated").textContent="Last checked: no matching observation";
      node("source-flight").href="https://www.flightradar24.com/data/flights/"+encodeURIComponent(view.flight.toLowerCase());
      return;
    }
    node("current-flights").innerHTML=view.current.map(function(r) {
      var dep=core.difference(r.scheduledDeparture,r.actualDeparture || r.estimatedDeparture), arrival=core.difference(r.scheduledArrival,r.landedAt || r.estimatedArrival);
      var age=Date.now()-core.instant(r.checkedAt);
      var snapshotNote=age>3600000 ? "This observation is over an hour old. Status may have changed." : "Status and estimates can change after this check.";
      return '<article class="flight-card"><div class="flight-top"><div class="route">'+esc(r.origin+" → "+r.destination)+'</div><span class="status '+(r.cancelled ? 'warning' : '')+'">'+esc(r.status)+' · at last check</span></div><p class="route-name">'+esc(r.originName+" → "+r.destinationName)+'</p><div class="time-grid"><div><h3>Departure · '+esc(r.origin)+'</h3><dl>'+timeRow("Scheduled",r.scheduledDeparture,r.originTimeZone)+timeRow("Estimated",r.estimatedDeparture,r.originTimeZone)+timeRow("Reported ATD",r.actualDeparture,r.originTimeZone)+'</dl></div><div><h3>Arrival · '+esc(r.destination)+'</h3><dl>'+timeRow("Scheduled",r.scheduledArrival,r.destinationTimeZone)+timeRow("Estimated",r.estimatedArrival,r.destinationTimeZone)+timeRow("Landed",r.landedAt,r.destinationTimeZone)+'</dl></div></div><div class="delays"><span class="delay '+(dep===null?'unknown':dep>0?'late':'')+'">'+(r.actualDeparture ? 'ATD − STD: ' : 'Estimated departure − STD: ')+esc(delta(dep))+'</span><span class="delay '+(arrival===null?'unknown':arrival>0?'late':'')+'">'+(r.landedAt ? 'Landing' : 'Estimated arrival')+' − STA: '+esc(delta(arrival))+'</span></div><p class="record-note">'+esc(r.aircraft || "Aircraft not recorded")+(r.registration ? ' · '+esc(r.registration) : '')+' · Checked '+esc(time(r.checkedAt,"UTC",true))+'</p><p class="record-note">'+esc(snapshotNote)+' Reported ATD and landing are not confirmed gate times.</p></article>';
    }).join("");
    var latest=view.current.slice().sort(function(a,b) { return core.instant(b.checkedAt)-core.instant(a.checkedAt); })[0];
    node("provider-updated").textContent="Last checked: "+time(latest.checkedAt,"UTC",true);
    node("source-flight").href=sourceUrl(latest.sourceUrl);
  }
  function drawOutlook(view) {
    var r=view.current[0], result=core.outlook(r,Date.now()), context=view.context;
    node("delay-outlook").innerHTML='<div class="outlook-label">'+esc(result.title)+'</div>'+((r && !r.actualDeparture && r.estimatedDeparture) ? '<p class="forecast-time">Estimated departure: <strong>'+esc(time(r.estimatedDeparture,r.originTimeZone,true))+'</strong></p>' : '')+'<p>'+esc(result.reason)+'</p>';
    if(context && context.airport) {
      var a=context.airport;
      node("airport-context").innerHTML='<p>'+esc(a.summary)+'</p>'+(a.details?'<p class="record-note">'+esc(a.details)+'</p>':'')+'<p class="record-note">Checked '+esc(time(a.checkedAt,"UTC",true))+'</p>'+link(a.sourceUrl,"Airport departure board");
      if(Array.isArray(a.samples) && a.samples.length) node("airport-context").innerHTML+='<details class="sample-details"><summary>View the '+a.samples.length+' sampled flights</summary><div class="sample-scroll"><table><caption>26 October · Copenhagen local time (CET)</caption><thead><tr><th>Flight</th><th>Scheduled</th><th>Reported departure</th></tr></thead><tbody>'+a.samples.map(function(s) { return '<tr><td>'+esc(s.flight)+'</td><td>'+esc(s.scheduled)+'</td><td>'+esc(s.reported || 'Not resolved')+'</td></tr>'; }).join('')+'</tbody></table></div></details>';
    } else node("airport-context").innerHTML='<p>No earlier-departure sample has been saved for this flight date.</p>';
    if(context && context.weather) {
      var w=context.weather;
      node("weather-context").innerHTML='<p>'+esc(w.summary)+'</p>'+(w.report?'<p class="weather-report">'+esc(w.report)+'</p>':'')+(w.details?'<p class="record-note">'+esc(w.details)+'</p>':'')+'<p class="record-note">Checked '+esc(time(w.checkedAt,"UTC",true))+'</p>'+link(w.sourceUrl,"Airport weather");
    } else node("weather-context").innerHTML='<p>No forecast issued before this departure has been saved. Current weather cannot reconstruct what was forecast beforehand.</p>';
  }
  function render() {
    if(!selection || !data) return;
    var view=core.select(data,selection.flight,selection.airport,selection.date);
    var current=view.current[0];
    node("result-flight").textContent=selection.flight+" · "+selection.airport+(current ? " → "+current.destination : " departures");
    node("result-date").textContent=formatDate(selection.date);
    drawHistory(view);drawCurrent(view);drawOutlook(view);
    node("fetch-status").textContent=view.current.length ? "Saved Flightradar24 observation loaded. Reload does not perform a new provider lookup." : "No matching saved status. Available historical observations are shown below.";
  }
  function submit(updateUrl) {
    var flight=core.normalFlight(flightField.value),airport=core.normalAirport(airportField.value),date=dateField.value;
    node("form-error").hidden=true;
    if(!core.validFlight(flight)) return formError("Enter a valid flight number, for example SK935 or SAS935.",flightField);
    if(!core.validAirport(airport)) return formError("Enter a three-letter IATA or four-letter ICAO departure airport code, for example CPH.",airportField);
    if(!core.validDate(date)) return formError("Choose a valid departure date.",dateField);
    flightField.value=flight;airportField.value=airport;
    selection={flight:flight,airport:airport,date:date};
    if(updateUrl) { var url=new URL(location.href);url.searchParams.set("flight",flight);url.searchParams.set("airport",airport);url.searchParams.set("date",date);history.replaceState(null,"",url.toString()); }
    render();
  }
  async function load(initial) {
    if(loading) return;
    setBusy(true);node("fetch-status").textContent="Loading published observations…";
    try {
      var path=window.FLIGHT_TRACKER_DATA_PATH || "observations.json";
      var url=new URL(path,location.href);
      if(url.origin!==location.origin) throw new Error("The observation file must be hosted on this site.");
      var response=await fetch(url.href,{cache:"no-cache",headers:{Accept:"application/json"}});
      if(!response.ok) throw new Error("The saved observation file could not be loaded.");
      var next=await response.json();
      if(!next || next.schemaVersion!==1 || !Array.isArray(next.records) || core.instant(next.checkedAt)===null || !core.validDate(next.defaultDate)) throw new Error("The saved observation file has an unexpected format.");
      data=next;
      node("snapshot-summary").textContent="Last manual check: "+time(data.checkedAt,"UTC",true)+". Updates are added manually.";
      if(initial) {
        var query=new URLSearchParams(location.search);
        flightField.value=query.get("flight") || data.defaultFlight || "SK935";
        airportField.value=query.get("airport") || data.defaultAirport || "CPH";
        dateField.value=query.get("date") || new Intl.DateTimeFormat("en-CA", { timeZone:"Europe/Copenhagen", year:"numeric", month:"2-digit", day:"2-digit" }).format(new Date());
      }
      submit(false);
    } catch(reason) {
      node("fetch-status").textContent=data ? "Reload failed. The previously loaded snapshot remains visible." : "Saved observations could not be loaded.";
      if(!data) {
        node("snapshot-summary").textContent="Observation file unavailable. Try reloading.";
        node("current-flights").innerHTML=empty("Saved data unavailable",reason.message);
        node("history-chart").innerHTML='<p>'+esc(reason.message)+'</p>';
        node("delay-outlook").innerHTML=empty("No estimate available","The saved data could not be loaded.");
      }
    } finally { setBusy(false); }
  }
  form.addEventListener("submit",function(event) { event.preventDefault();submit(true); });
  node("refresh").addEventListener("click",function() { load(false); });
  load(true);
})();
