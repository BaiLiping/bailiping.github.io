(function () {
  "use strict";
  var core = window.FlightTracker;
  var node = function (id) { return document.getElementById(id); };
  var form = node("flight-form"), flightField = node("flight-number"), airportField = node("departure-airport"), dateField = node("flight-date");
  var data = null, selection = null, loading = false, apiEndpoint = "";
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
    return core.validSourceUrl(value) ? new URL(value).href : "https://www.flightradar24.com/data";
  }
  function link(value, label) { return '<a href="'+esc(sourceUrl(value))+'" target="_blank" rel="noopener noreferrer">'+esc(label)+' ↗</a>'; }
  function empty(title, reason) { return '<div class="empty"><strong>'+esc(title)+'</strong><p>'+esc(reason)+'</p></div>'; }
  function setBusy(value) {
    loading=value;
    node("live-check").disabled=value || !apiEndpoint;
    [flightField,airportField,dateField].forEach(function(field) { field.disabled=value; });
    node("results").setAttribute("aria-busy",String(value));
  }
  function formError(reason, field) { node("form-error").textContent=reason; node("form-error").hidden=false; if(field) field.focus(); return false; }
  function drawHistory(view) {
    var rows=view.history, records=rows.flatMap(function (day) { return day.flights; }), stats=core.stats(records);
    node("history-period").textContent=formatDate(rows[0].date)+" – "+formatDate(rows[rows.length-1].date)+" · "+stats.count+" flight records · "+view.airport+" departures";
    node("history-metrics").innerHTML=metric(stats.count,"Flights recorded","Across seven calendar days")+
      metric(delta(stats.meanDeparture),"Mean ATD − STD",stats.departureCount+" actual departure entries")+
      metric(delta(stats.meanArrival),"Mean landing − STA",stats.arrivalCount+" completed flights");
    var points=rows.map(function(day) {
      var daily=core.stats(day.flights);
      return {date:day.date, departure:daily.meanDeparture, arrival:daily.meanArrival};
    });
    var largest=Math.max(1,...points.flatMap(function(p) { return [Math.abs(p.departure || 0),Math.abs(p.arrival || 0)]; }));
    var scale=Math.ceil(largest/10)*10;
    function series(value,kind,label) {
      var width=value===null ? 0 : 50*Math.abs(value)/scale;
      var left=value!==null && value<0 ? 50-width : 50;
      return '<div class="chart-row"><span class="chart-series-label '+kind+'">'+label+'</span><div class="chart-track"><span class="chart-fill '+kind+(value===null ? ' missing' : value===0 ? ' zero' : '')+'" style="left:'+left+'%;width:'+width+'%"></span></div><span class="chart-value">'+esc(value===null ? 'No record' : delta(value))+'</span></div>';
    }
    node("history-chart").innerHTML='<p class="chart-note">Daily averages: reported departure − scheduled departure, and landing − scheduled arrival. Negative values are early; positive values are late. These comparisons may include taxi time and are not verified gate-to-gate delays.</p><div class="chart-axis"><div class="chart-scale"><span>−'+scale+'</span><span>0</span><span>+'+scale+' min</span></div></div>'+points.map(function(p) {
      return '<div class="chart-day"><span class="chart-label">'+esc(formatDate(p.date,true))+'</span><div class="chart-series">'+series(p.departure,'departure','Departure')+series(p.arrival,'arrival','Arrival')+'</div></div>';
    }).join("");
    node("history-chart").setAttribute("aria-label","Departure and arrival differences from schedule, in minutes. Negative is early; positive is late. "+points.map(function(p) {
      return formatDate(p.date,true)+": departure "+(p.departure===null ? "no record" : delta(p.departure))+", arrival "+(p.arrival===null ? "no record" : delta(p.arrival));
    }).join("; "));
    node("history-rows").innerHTML=rows.map(function(day) {
      if(!day.flights.length) return '<tr><td>'+esc(formatDate(day.date,true))+'</td><td class="missing" colspan="7">No record available from this lookup</td></tr>';
      return day.flights.map(function(r) {
        return '<tr><td>'+link(r.sourceUrl,formatDate(r.date,true))+'</td><td>'+esc(r.origin+" → "+r.destination)+'</td><td class="num">'+esc(time(r.scheduledDeparture,r.originTimeZone))+'</td><td class="num">'+esc(time(r.actualDeparture,r.originTimeZone))+'</td><td class="num">'+esc(core.difference(r.scheduledDeparture,r.actualDeparture)===null ? r.status : delta(core.difference(r.scheduledDeparture,r.actualDeparture)))+'</td><td class="num">'+esc(time(r.scheduledArrival,r.destinationTimeZone))+'</td><td class="num">'+esc(time(r.landedAt,r.destinationTimeZone))+'</td><td class="num">'+esc(delta(core.difference(r.scheduledArrival,r.landedAt)))+'</td></tr>';
      }).join("");
    }).join("");
  }
  function timeRow(label,value,zone) { return '<div><dt>'+esc(label)+'</dt><dd>'+esc(time(value,zone,true))+'</dd></div>'; }
  function drawCurrent(view) {
    node("current-period").textContent="Departure "+formatDate(view.date)+" · times local to each airport, with timezone labels.";
    if(!view.current.length) {
      node("current-flights").innerHTML=empty("No record available for this flight and date","This does not mean the flight is cancelled or on time. A record will appear here when source-backed information is available.");
      node("provider-updated").textContent="Last checked: no matching observation";
      node("source-flight").href="https://www.flightradar24.com/data/flights/"+encodeURIComponent(view.flight.toLowerCase());
      return;
    }
    node("current-flights").innerHTML=view.current.map(function(r) {
      var dep=core.difference(r.scheduledDeparture,r.actualDeparture || r.estimatedDeparture), arrival=core.difference(r.scheduledArrival,r.landedAt || r.estimatedArrival);
      var age=Date.now()-core.instant(r.checkedAt);
      var statusNote=age>3600000 ? "This observation is over an hour old. Status may have changed." : "Status and estimates can change after this check.";
      return '<article class="flight-card"><div class="flight-top"><div class="route">'+esc(r.origin+" → "+r.destination)+'</div><span class="status '+(r.cancelled ? 'warning' : '')+'">'+esc(r.status)+' · at last check</span></div><p class="route-name">'+esc(r.originName+" → "+r.destinationName)+'</p><div class="time-grid"><div><h3>Departure · '+esc(r.origin)+'</h3><dl>'+timeRow("Scheduled",r.scheduledDeparture,r.originTimeZone)+timeRow("Estimated",r.estimatedDeparture,r.originTimeZone)+timeRow("Reported ATD",r.actualDeparture,r.originTimeZone)+'</dl></div><div><h3>Arrival · '+esc(r.destination)+'</h3><dl>'+timeRow("Scheduled",r.scheduledArrival,r.destinationTimeZone)+timeRow("Estimated",r.estimatedArrival,r.destinationTimeZone)+timeRow("Landed",r.landedAt,r.destinationTimeZone)+'</dl></div></div><div class="delays"><span class="delay '+(dep===null?'unknown':dep>0?'late':'')+'">'+(r.actualDeparture ? 'ATD − STD: ' : 'Estimated departure − STD: ')+esc(delta(dep))+'</span><span class="delay '+(arrival===null?'unknown':arrival>0?'late':'')+'">'+(r.landedAt ? 'Landing' : 'Estimated arrival')+' − STA: '+esc(delta(arrival))+'</span></div><p class="record-note">'+esc(r.aircraft || "Aircraft not recorded")+(r.registration ? ' · '+esc(r.registration) : '')+' · Checked '+esc(time(r.checkedAt,"UTC",true))+'</p><p class="record-note">'+esc(statusNote)+' Reported ATD and landing are not confirmed gate times.</p></article>';
    }).join("");
    var latest=view.current.slice().sort(function(a,b) { return core.instant(b.checkedAt)-core.instant(a.checkedAt); })[0];
    node("provider-updated").textContent="Last checked: "+time(latest.checkedAt,"UTC",true);
    node("source-flight").href=sourceUrl(latest.sourceUrl);
  }
  function drawOutlook(view) {
    var r=view.current[0], result=core.outlook(r,Date.now()), context=view.context;
    node("delay-outlook").innerHTML='<div class="outlook-label">'+esc(result.title)+'</div>'+((r && !r.actualDeparture && r.estimatedDeparture) ? '<p class="forecast-time">Estimated departure: <strong>'+esc(time(r.estimatedDeparture,r.originTimeZone,true))+'</strong></p>' : '')+'<p>'+esc(result.reason)+'</p>';
    if(r && !r.actualDeparture && r.departureEstimateSource) node("delay-outlook").innerHTML+='<p class="record-note">Reported '+esc(time(r.departureEstimateSource.asOf,r.originTimeZone,true))+' · '+link(r.departureEstimateSource.sourceUrl,"Departure estimate source")+'</p>';
    if(context && context.airport) {
      var a=context.airport;
      node("airport-context").innerHTML='<p>'+esc(a.summary)+'</p>'+(a.details?'<p class="record-note">'+esc(a.details)+'</p>':'')+'<p class="record-note">'+(a.asOf?'Reported '+esc(time(a.asOf,r?r.originTimeZone:"UTC",true))+' · ':'')+'Checked '+esc(time(a.checkedAt,"UTC",true))+'</p>'+link(a.sourceUrl,"Airport departure board");
      if(Array.isArray(a.samples) && a.samples.length) node("airport-context").innerHTML+='<details class="sample-details"><summary>View the '+a.samples.length+' sampled flights</summary><div class="sample-scroll"><table><caption>'+esc(formatDate(context.date)+' · '+context.origin+' local time'+(a.timeZone ? ' ('+a.timeZone+')' : ''))+'</caption><thead><tr><th>Flight</th><th>Scheduled</th><th>Reported departure</th></tr></thead><tbody>'+a.samples.map(function(s) { return '<tr><td>'+esc(s.flight)+'</td><td>'+esc(s.scheduled)+'</td><td>'+esc(s.reported || 'Not resolved')+'</td></tr>'; }).join('')+'</tbody></table></div></details>';
    } else node("airport-context").innerHTML='<p>No airport conditions recorded before this departure were available.</p>';
    if(context && context.weather) {
      var w=context.weather;
      node("weather-context").innerHTML='<p>'+esc(w.summary)+'</p>'+(w.report?'<p class="weather-report">'+esc(w.report)+'</p>':'')+(w.details?'<p class="record-note">'+esc(w.details)+'</p>':'')+'<p class="record-note">'+(w.asOf?(w.kind==='forecast'?'Forecast issued ':'Observed ')+esc(time(w.asOf,r?r.originTimeZone:"UTC",true))+' · ':'')+'Checked '+esc(time(w.checkedAt,"UTC",true))+'</p>'+link(w.sourceUrl,"Airport weather");
    } else node("weather-context").innerHTML='<p>No weather observation or forecast was available for this departure time.</p>';
  }
  function render() {
    if(!selection || !data) return;
    var view=core.select(data,selection.flight,selection.airport,selection.date);
    var current=view.current[0];
    node("result-flight").textContent=selection.flight+" · "+selection.airport+(current ? " → "+current.destination : " departures");
    node("result-date").textContent=formatDate(selection.date);
    drawHistory(view);drawCurrent(view);drawOutlook(view);
    node("results").hidden=false;
  }
  function clearResults() {
    data=null;selection=null;
    node("results").hidden=true;
    ["history-metrics","history-chart","history-rows","current-flights","delay-outlook","airport-context","weather-context"].forEach(function(id) { node(id).replaceChildren(); });
    node("result-flight").textContent="";
    node("result-date").textContent="";
    node("provider-updated").textContent="Last checked: —";
    node("source-flight").href="https://www.flightradar24.com/data";
    node("history-details").open=false;
  }
  function readSelection() {
    var flight=core.normalFlight(flightField.value),airport=core.normalAirport(airportField.value),date=dateField.value;
    node("form-error").hidden=true;
    if(!core.validFlight(flight)) return formError("Enter a valid flight number, for example SK935 or SAS935.",flightField);
    if(!core.validAirport(airport)) return formError("Enter a three-letter IATA or four-letter ICAO departure airport code, for example CPH.",airportField);
    if(!core.validDate(date)) return formError("Choose a valid departure date.",dateField);
    flightField.value=flight;airportField.value=airport;
    return {flight:flight,airport:airport,date:date};
  }
  async function configureLiveCheck() {
    try {
      var response=await fetch(new URL("api-config.json",location.href),{cache:"no-cache"});
      if(!response.ok) throw new Error("Configuration unavailable");
      var config=await response.json();
      var endpoint=new URL(config.endpoint);
      var local=["127.0.0.1","localhost"].includes(location.hostname) && ["127.0.0.1","localhost"].includes(endpoint.hostname);
      if(endpoint.username || endpoint.password || endpoint.search || endpoint.hash || (endpoint.protocol!=="https:" && !(local && endpoint.protocol==="http:"))) throw new Error("Invalid endpoint");
      apiEndpoint=endpoint.href.replace(/\/$/,"");
    } catch (_) {
      node("live-check-note").textContent="Flight lookup is temporarily unavailable. Please try again later.";
    }
    setBusy(loading);
  }
  async function checkLive() {
    if(loading || !apiEndpoint) return;
    var requested=readSelection();
    if(!requested) return;
    clearResults();
    var url=new URL(location.href);
    ["flight","airport","date"].forEach(function(key) { url.searchParams.set(key,requested[key]); });
    history.replaceState(null,"",url.toString());
    var controller=new AbortController();
    var timeout=setTimeout(function(){ controller.abort(); },165000);
    setBusy(true);
    node("live-check").textContent="Checking Flightradar24…";
    node("fetch-status").textContent="Reading Flightradar24’s records for this flight and its previous seven days. This may take a minute or two…";
    try {
      var response=await fetch(apiEndpoint+"/check",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(requested),signal:controller.signal,cache:"no-store"});
      var result;
      try { result=await response.json(); } catch (_) { throw new Error("The flight lookup returned an unreadable response. Please try again."); }
      if(!response.ok) throw new Error(typeof result.error==="string" ? result.error.slice(0,600) : "The source check could not be completed.");
      data=window.FlightTrackerLive.validateResponse(result,requested);
      selection=requested;
      render();
      var view=core.select(data,requested.flight,requested.airport,requested.date);
      var found=view.history.filter(function(day){return day.flights.length;}).length;
      var missing=core.HISTORY_DAYS-found;
      node("fetch-status").textContent="Flightradar24 checked: "+time(data.checkedAt,"UTC",true)+". "+(!view.current.length ? "The selected day was unavailable in the retrieved pages. " : "")+found+" of "+core.HISTORY_DAYS+" history dates have source-backed records; "+missing+(missing===1 ? " remains" : " remain")+" unavailable.";
    } catch(reason) {
      clearResults();
      node("fetch-status").textContent=(reason.name==="AbortError" ? "The source check timed out. Please try again." : reason.message);
    } finally {
      clearTimeout(timeout);node("live-check").textContent="Check Flightradar24";setBusy(false);
    }
  }
  form.addEventListener("submit",function(event) { event.preventDefault();checkLive(); });
  form.addEventListener("input",function() {
    if(loading) return;
    clearResults();
    node("form-error").hidden=true;
    node("fetch-status").textContent="Submit the form to check these flight details.";
  });
  var query=new URLSearchParams(location.search);
  flightField.value=query.get("flight") || "";
  airportField.value=query.get("airport") || "";
  dateField.value=query.get("date") || "";
  configureLiveCheck();
})();
