import {PRESETS,preset,solveScene} from './math.mjs';
import {sceneSVG,world} from './scene.mjs';
const $=id=>document.getElementById(id),params=new URLSearchParams(location.search);
let presetKey=Object.hasOwn(PRESETS,params.get('preset'))?params.get('preset'):'shared';
let state=preset(presetKey),selectedMeasurement=0,selectedEvent=null,result,dragging=null,announcement;
const percent=v=>`${(100*v).toFixed(1)}%`,num=(v,n=2)=>v.toFixed(n),trace=m=>m[0][0]+m[1][1];
const target=t=>t?'B':'A';
function controlValues(){
  $('preset').value=presetKey;
  for(const key of ['separation','pd','sigma'])$(key).value=state[key];
  $('lambda').value=Math.log10(state.lambda);
  $('measurement').replaceChildren(...state.measurements.map((_,j)=>new Option(`z${j+1}`,j)));
  $('measurement').value=selectedMeasurement;
  positionValues();
}
function positionValues(){
  const z=state.measurements[selectedMeasurement];
  $('zx').value=z[0];$('zy').value=z[1];
  $('zx-out').textContent=`${num(z[0])} m`;$('zy-out').textContent=`${num(z[1])} m`;
}
function renderPlot(){
  const focus=document.activeElement?.getAttribute('data-measurement');
  $('plot').innerHTML=sceneSVG(result,{interactive:true,selectedMeasurement,selectedEvent,showPDA:$('compare').checked,showContours:$('contours').checked,id:'lab-scene'});
  if(focus!==null&&focus!==undefined)$('plot').querySelector(`[data-measurement="${focus}"]`)?.focus({preventScroll:true});
}
function table(rows,{clutter=false,totals=false}={}){
  let html='<table><thead><tr><th scope="col">Target</th><th scope="col">Miss</th>'+state.measurements.map((_,j)=>`<th scope="col">z${j+1}</th>`).join('')+'</tr></thead><tbody>';
  rows.forEach((row,t)=>{html+=`<tr><th scope="row" class="track-${t?'b':'a'}">${target(t)}</th>`+row.map(p=>`<td class="prob-cell" style="--fill:${100*p}%">${percent(p)}</td>`).join('')+'</tr>';});
  if(clutter)html+='<tr class="clutter"><th scope="row">Clutter</th><td>—</td>'+result.clutter.map(p=>`<td>${percent(p)}</td>`).join('')+'</tr>';
  if(totals)html+='<tr><th scope="row">A + B</th><td>—</td>'+state.measurements.map((_,j)=>{const sum=rows.reduce((s,row)=>s+row[j+1],0);return `<td class="${sum>1+1e-9?'overbooked':''}">${percent(sum)}</td>`;}).join('')+'</tr>';
  return html+'</tbody></table>';
}
function renderEvents(){
  const focus=document.activeElement?.dataset.event;
  $('events').innerHTML=[...result.events].sort((a,b)=>b.probability-a.probability).map(e=>`<button type="button" class="event" data-event="${e.id}" aria-pressed="${selectedEvent===e.id}" style="--probability:${100*e.probability}%" aria-label="${e.assignment.map((j,t)=>`${target(t)} ${j?'uses z'+j:'missed'}`).join(', ')}, probability ${percent(e.probability)}"><span>${e.assignment.map((j,t)=>`${target(t)} → ${j?'z'+j:'miss'}`).join(' · ')}</span><b>${percent(e.probability)}</b></button>`).join('');
  $('clear-event').setAttribute('aria-pressed',String(selectedEvent===null));
  if(focus)$('events').querySelector(`[data-event="${focus}"]`)?.focus({preventScroll:true});
}
function render(){
  result=solveScene(state);
  for(const key of ['separation','pd','sigma','lambda'])$(key+'-out').textContent=key==='lambda'?`${num(state[key],3)} m⁻²`:num(state[key],key==='separation'?1:2)+(key==='sigma'||key==='separation'?' m':'');
  $('lambda').setAttribute('aria-valuetext',`${num(state.lambda,3)} per square metre`);
  $('event-count').textContent=result.events.length;
  $('total-probability').textContent=num(result.events.reduce((s,e)=>s+e.probability,0),3);
  renderPlot();renderEvents();
  $('marginals').innerHTML=table(result.beta,{clutter:true});
  $('pda-comparison').hidden=!$('compare').checked;
  $('pda-comparison').innerHTML='<div class="pda-table"><h4>Independent PDA, same likelihoods</h4><div class="table-scroll">'+table(result.independent,{totals:true})+'</div><p>Independent filters normalize separately. Their A + B totals can exceed 100%; these are not joint assignment probabilities.</p></div>';
  const sums=state.measurements.map((_,j)=>result.independent.reduce((s,row)=>s+row[j+1],0)),max=Math.max(...sums),j=sums.indexOf(max);
  $('insight').textContent=max>1.001?`Independent PDA assigns a combined ${percent(max)} to z${j+1}. JPDA gives ${percent(result.beta.reduce((s,row)=>s+row[j+1],0))}: the same return cannot originate from both targets in one event.`:`The most probable event carries ${percent(result.map.probability)}. JPDA keeps all ${result.events.length} feasible alternatives in the update, including missed detections and clutter.`;
  $('updates').innerHTML=result.updates.map((u,t)=>`<div><h4 class="track-${t?'b':'a'}">Target ${target(t)}</h4><dl><dt>Updated position / m</dt><dd>(${u.mean.map(x=>num(x)).join(', ')})</dd><dt>Covariance trace / m²</dt><dd>${num(trace(u.cov),3)}</dd><dt>Within-branch trace / m²</dt><dd>${num(trace(u.base),3)}</dd><dt>Between-branch spread / m²</dt><dd>${num(trace(u.spread),3)}</dd></dl></div>`).join('');
  clearTimeout(announcement);announcement=setTimeout(()=>{$('status').textContent=`Updated: ${result.events.length} events. Target A miss probability ${percent(result.beta[0][0])}; target B miss probability ${percent(result.beta[1][0])}.`;},450);
}
function reset(){state=preset(presetKey);selectedMeasurement=0;selectedEvent=null;controlValues();render();}
$('preset').addEventListener('change',e=>{presetKey=e.target.value;reset();});
$('reset').addEventListener('click',reset);
for(const key of ['separation','pd','lambda','sigma'])$(key).addEventListener('input',e=>{state[key]=key==='lambda'?10**Number(e.target.value):Number(e.target.value);selectedEvent=null;render();});
for(const key of ['compare','contours'])$(key).addEventListener('change',render);
$('measurement').addEventListener('change',e=>{selectedMeasurement=Number(e.target.value);positionValues();renderPlot();});
for(const [axis,id] of ['zx','zy'].entries())$(id).addEventListener('input',e=>{state.measurements[selectedMeasurement][axis]=Number(e.target.value);selectedEvent=null;positionValues();render();});
$('events').addEventListener('click',e=>{const button=e.target.closest('[data-event]');if(!button)return;selectedEvent=button.dataset.event===selectedEvent?null:button.dataset.event;renderPlot();renderEvents();});
$('clear-event').addEventListener('click',()=>{selectedEvent=null;renderPlot();renderEvents();});
$('plot').addEventListener('keydown',e=>{
  const item=e.target.closest('[data-measurement]');if(!item||!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key))return;
  e.preventDefault();e.stopPropagation();selectedMeasurement=Number(item.dataset.measurement);
  const z=state.measurements[selectedMeasurement],step=e.shiftKey ? .5 : .1;
  z[0]=Math.max(-8.5,Math.min(8.5,z[0]+(e.key==='ArrowRight'?step:e.key==='ArrowLeft'?-step:0)));
  z[1]=Math.max(-5,Math.min(5,z[1]+(e.key==='ArrowUp'?step:e.key==='ArrowDown'?-step:0)));
  selectedEvent=null;$('measurement').value=selectedMeasurement;positionValues();render();
});
$('plot').addEventListener('pointerdown',e=>{
  const item=e.target.closest('[data-measurement]');if(!item||e.button!==0)return;
  e.preventDefault();selectedMeasurement=Number(item.dataset.measurement);dragging=e.pointerId;selectedEvent=null;
  $('plot').setPointerCapture(e.pointerId);$('measurement').value=selectedMeasurement;positionValues();renderPlot();
  $('plot').querySelector(`[data-measurement="${selectedMeasurement}"]`)?.focus({preventScroll:true});
});
$('plot').addEventListener('pointermove',e=>{
  if(dragging!==e.pointerId)return;
  const svg=$('plot').querySelector('svg'),point=new DOMPoint(e.clientX,e.clientY).matrixTransform(svg.getScreenCTM().inverse());
  state.measurements[selectedMeasurement]=world([point.x,point.y]);
  positionValues();render();
});
for(const event of ['pointerup','pointercancel','lostpointercapture'])$('plot').addEventListener(event,()=>{dragging=null;});
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&params.get('embed')==='1'){e.preventDefault();parent.postMessage({type:'jpda-back'},location.origin);}});
window.addEventListener('pagehide',()=>clearTimeout(announcement));
controlValues();render();
document.documentElement.dataset.jpdaReady='true';
if(params.get('embed')==='1')parent.postMessage({type:'jpda-ready'},location.origin);
