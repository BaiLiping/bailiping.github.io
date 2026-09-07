/* Real one-scan numerical demo. All three views share the article's solver. */
(function(){
'use strict';
const M=window.AssociationModel,$=id=>document.getElementById(id);
const query=new URLSearchParams(location.search),mode=['assignment','bp','hypotheses'].includes(query.get('demo'))?query.get('demo'):'assignment';
if(query.has('embed'))document.body.classList.add('embed');
$(mode).hidden=false;
const colors=['#2ca02c','#9467bd','#d62728'];
let scene=M.clone(M.DEFAULT),gated=true,R=null,iteration=0,k=5,timer=null;
const percent=x=>(100*x).toFixed(1)+'%',msg=x=>x==null?'–':x.toFixed(2);
function table(A,labels,probability=true){
 let s='<table><thead><tr><th></th>'+labels.map(t=>'<th>'+t+'</th>').join('')+'</tr></thead><tbody>';
 A.forEach((row,i)=>{s+='<tr><th>T'+(i+1)+'</th>'+row.map(x=>'<td>'+ (probability?percent(x):x===0?'0':x.toFixed(3))+'</td>').join('')+'</tr>';});
 return s+'</tbody></table>';
}
function compute(){
 const {L,gate,n,m}=M.buildWeights(scene,gated),events=M.enumerate(L),bp=M.bp(L),exact=M.eventMarginals(events,n,m).marginals;
 R={L,gate,n,m,events,bp,exact,graph:M.topology(L)};k=Math.min(k,events.length);iteration=0;
}
function stop(){if(timer)clearInterval(timer);timer=null;$('play').textContent='Play';}
function refresh(){stop();compute();draw();}
function draw(){
 $('pdOut').textContent=scene.PD.toFixed(2);$('clutterOut').textContent=Math.log10(scene.c).toFixed(3);
 $('status').textContent=`One scan · ${R.events.length} positive-weight assignments · ${R.graph.edges} active edges · ${R.graph.acyclic?'acyclic':'cyclic'} graph. Exact means summation for these weights, not a full PMBM filter.`;
 if(mode==='assignment')drawAssignment();else if(mode==='bp')drawBP();else drawHypotheses();
}
function ellipse(S){const a=S[0][0],b=S[0][1],c=S[1][1],t=(a+c)/2,d=Math.hypot((a-c)/2,b);return{rx:Math.sqrt(M.GATE*(t+d)),ry:Math.sqrt(M.GATE*(t-d)),angle:90*Math.atan2(2*b,a-c)/Math.PI};}
function drawAssignment(){
 let s='<rect width="720" height="420" fill="#fff"/>';
 for(let x=0;x<720;x+=60)s+=`<path d="M${x} 0V420" stroke="#eef2f5"/>`;
 for(let y=0;y<420;y+=60)s+=`<path d="M0 ${y}H720" stroke="#eef2f5"/>`;
 scene.T.forEach((t,i)=>{
  const e=ellipse(t.S);if(gated)s+=`<ellipse cx="${t.x}" cy="${t.y}" rx="${e.rx}" ry="${e.ry}" transform="rotate(${e.angle} ${t.x} ${t.y})" fill="${colors[i]}" fill-opacity=".05" stroke="${colors[i]}" stroke-dasharray="6 4"/>`;
  s+=`<g class="draggable"><circle cx="${t.x}" cy="${t.y}" r="7" fill="${colors[i]}"/><circle cx="${t.x}" cy="${t.y}" r="18" fill="transparent"/><text x="${t.x+10}" y="${t.y-12}" fill="${colors[i]}" font-size="18">T${i+1}</text></g>`;
 });
 scene.Z.forEach((z,j)=>{s+=`<g class="draggable"><path d="M${z.x-7} ${z.y-7}l14 14m-14 0l14-14" stroke="#16222e" stroke-width="3"/><circle cx="${z.x}" cy="${z.y}" r="18" fill="transparent"/><text x="${z.x+10}" y="${z.y+20}" font-size="17">z${j+1}</text></g>`;});
 $('scene').innerHTML=s;$('weights').innerHTML=table(R.L,['∅',...scene.Z.map((_,j)=>'z'+(j+1))],false);
 $('sceneStats').textContent=`After ${R.bp.iterations} sweeps, ${R.bp.converged?'BP stopping tolerance met':'iteration cap reached'}. Maximum BP–exact difference: ${(100*M.maxDifference(R.bp.marginals,R.exact)).toFixed(3)} percentage points.`;
}
function drawBP(){
 const h=R.bp.history[iteration],labels=['∅',...scene.Z.map((_,j)=>'z'+(j+1))];
 $('phase').textContent=h.kind==='init'?'Initialize ν = 1':`Sweep ${h.t}/${R.bp.iterations} · ${h.kind==='mu'?'μ updated; track beliefs unchanged':'ν updated; track beliefs updated'}`;
 $('bpTable').innerHTML=table(h.marg,labels);
 if(!h.bmarg)$('measurementTable').textContent='Available after the first μ half-step.';
 else{
  let s='<table><tr><th></th><th>unassigned</th><th>T1</th><th>T2</th><th>T3</th></tr>';
  h.bmarg.forEach((row,j)=>{s+='<tr><th>z'+(j+1)+'</th>'+row.map(x=>'<td>'+percent(x)+'</td>').join('')+'</tr>';});$('measurementTable').innerHTML=s+'</table>';
 }
 const ty=i=>40+82*i,my=j=>22+67*j;let s='';
 for(let i=0;i<R.n;i++)for(let j=0;j<R.m;j++){
  const active=R.L[i][j+1]>0,y1=ty(i),y2=my(j),fraction=.27+i*.21;
  s+=`<line x1="62" y1="${y1}" x2="475" y2="${y2}" stroke="${active?colors[i]:'#cbd2d9'}" stroke-opacity="${active?.6:.25}" ${active?'':'stroke-dasharray="3 5"'}/>`;
  if(active){const x=62+413*fraction,y=y1+(y2-y1)*fraction;s+=`<text x="${x}" y="${y-4}" font-size="12" fill="${colors[i]}" stroke="white" stroke-width="4" paint-order="stroke" text-anchor="middle">${msg(h.mu?.[i][j])} / ${msg(h.nu[j][i])}</text>`;}
 }
 for(let i=0;i<R.n;i++)s+=`<circle cx="47" cy="${ty(i)}" r="17" fill="white" stroke="${colors[i]}" stroke-width="2"/><text x="47" y="${ty(i)+4}" font-size="13" text-anchor="middle">a${i+1}</text>`;
 for(let j=0;j<R.m;j++)s+=`<circle cx="490" cy="${my(j)}" r="15" fill="white" stroke="#51606e"/><text x="490" y="${my(j)+4}" font-size="13" text-anchor="middle">b${j+1}</text>`;
 $('graph').innerHTML=s;
 $('bpStats').textContent=(iteration===R.bp.history.length-1?(R.bp.converged?'Stopping tolerance met. ':'Iteration cap reached. '):'Intermediate beliefs need not agree across the two orientations. ')+`Final max |Δ log ν| ${R.bp.delta.toExponential(1)}; final edge-consistency residual ${R.bp.dualResidual.toExponential(1)}. Final BP–exact difference ${(100*M.maxDifference(R.bp.marginals,R.exact)).toFixed(3)} pp.`;
 $('back').disabled=iteration===0;$('step').disabled=iteration===R.bp.history.length-1;
}
function drawHypotheses(){
 const top=M.eventMarginals(R.events,R.n,R.m,k);$('keep').max=R.events.length;$('keep').value=k;$('keepOut').textContent=k;
 $('mass').textContent=`Retained ${percent(top.mass)} · discarded ${percent(top.discardedMass)}`;
 let s='<table><tr><th>#</th><th>T1</th><th>T2</th><th>T3</th><th>P(event)</th></tr>';
 R.events.forEach((e,r)=>{s+=`<tr class="${r<k?'kept':'discarded'}"><td>${r+1}</td>`+e.a.map(j=>'<td>'+(j<0?'∅':'z'+(j+1))+'</td>').join('')+'<td>'+percent(e.p)+'</td></tr>';});$('events').innerHTML=s+'</table>';
 const labels=['∅',...scene.Z.map((_,j)=>'z'+(j+1))];$('exactTable').innerHTML=table(R.exact,labels);$('topTable').innerHTML=table(top.marginals,labels);
 $('topStats').textContent=`Top-k maximum marginal error ${(100*M.maxDifference(top.marginals,R.exact)).toFixed(3)} pp; bounded by ${(100*top.discardedMass).toFixed(3)} pp. This is truncation error, not BP error.`;
}
$('pd').addEventListener('input',e=>{scene.PD=+e.target.value;refresh();});
$('clutter').addEventListener('input',e=>{scene.c=10**(+e.target.value);refresh();});
$('gate').addEventListener('change',e=>{gated=e.target.checked;refresh();});
$('preset').addEventListener('change',e=>{
 const PD=scene.PD,c=scene.c;scene=M.clone(M.DEFAULT);scene.PD=PD;scene.c=c;
 if(e.target.value==='separated'){scene.T.forEach((t,i)=>{t.x=120+220*i;t.y=210;scene.Z[i]={x:t.x+5,y:t.y-3};});scene.Z[3]={x:660,y:45};}
 if(e.target.value==='symmetric'){scene.T.forEach(t=>{t.x=320;t.y=210;t.S=[[500,0],[0,500]];});scene.Z=[{x:305,y:200},{x:320,y:225},{x:335,y:200},{x:560,y:60}];}
 gated=true;$('gate').checked=true;refresh();
});
$('keep').addEventListener('input',e=>{k=+e.target.value;drawHypotheses();});
function step(){if(iteration<R.bp.history.length-1){iteration++;drawBP();}else stop();}
$('step').onclick=()=>{stop();step();};$('reset').onclick=()=>{stop();iteration=0;drawBP();};$('back').onclick=()=>{stop();iteration=Math.max(0,iteration-1);drawBP();};$('end').onclick=()=>{stop();iteration=R.bp.history.length-1;drawBP();};
$('play').onclick=()=>{if(timer){stop();return;}if(iteration===R.bp.history.length-1)iteration=0;$('play').textContent='Pause';timer=setInterval(step,450);};
let drag=null;const svg=$('scene');
function point(e){const q=new DOMPoint(e.clientX,e.clientY).matrixTransform(svg.getScreenCTM().inverse());return{x:q.x,y:q.y};}
svg.addEventListener('pointerdown',e=>{const p=point(e);let distance=20**2;for(const name of ['T','Z'])scene[name].forEach((o,i)=>{const d=(o.x-p.x)**2+(o.y-p.y)**2;if(d<distance){distance=d;drag={name,i};}});if(drag){svg.setPointerCapture(e.pointerId);e.preventDefault();}});
svg.addEventListener('pointermove',e=>{if(!drag)return;const p=point(e),o=scene[drag.name][drag.i];o.x=Math.max(15,Math.min(705,p.x));o.y=Math.max(15,Math.min(405,p.y));refresh();});
for(const type of ['pointerup','pointercancel'])svg.addEventListener(type,()=>{drag=null;});
document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
window.addEventListener('message',event=>{if(event.source!==parent)return;if(event.data?.type==='bento-inline-pause'||event.data?.type==='bento-live-pause')stop();});
document.addEventListener('keydown',event=>{
 if(!document.body.classList.contains('embed')||event.target.closest('input,select,button,textarea,a'))return;
 if(['ArrowLeft','ArrowRight','PageUp','PageDown'].includes(event.key)){
  event.preventDefault();parent.postMessage({type:'bento-inline-nav',direction:['ArrowLeft','PageUp'].includes(event.key)?-1:1},'*');
 }
});
window.BPAssociationAudit={snapshot:()=>M.clone({mode,L:R.L,exact:R.exact,bp:R.bp.marginals,iterations:R.bp.iterations,converged:R.bp.converged,graph:R.graph,iteration,k})};
compute();draw();
parent.postMessage({type:'bento-inline-ready'},'*');
})();
