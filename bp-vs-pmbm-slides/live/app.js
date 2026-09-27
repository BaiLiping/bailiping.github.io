// Canonical UI state -> the shared deterministic association model -> renderer.
(function(){
'use strict';
const M=window.AssociationModel,V=window.AssociationRenderer,$=id=>document.getElementById(id);
const query=new URLSearchParams(location.search),mode=['assignment','bp','hypotheses'].includes(query.get('demo'))?query.get('demo'):'assignment';
if(query.has('embed'))document.body.classList.add('embed');
$(mode).hidden=false;$(mode+'-controls').hidden=false;
let scene=M.clone(M.DEFAULT),gated=true,R,iteration=0,k=5,timer=null,selected=[0,0];
function stop(){clearInterval(timer);timer=null;$('play').textContent='Play';}
function compute(){const {L,n,m}=M.buildWeights(scene,gated),events=M.enumerate(L),bp=M.bp(L),exact=M.eventMarginals(events,n,m).marginals;R={L,n,m,events,bp,exact,graph:M.topology(L)};iteration=0;k=Math.min(k,events.length);const options=[];for(let i=0;i<n;i++)for(let j=0;j<m;j++)if(L[i][j+1]>0)options.push([i,j]);if(!options.some(p=>p[0]===selected[0]&&p[1]===selected[1]))selected=options[0]||[0,0];$('edge').innerHTML=options.map(([i,j])=>`<option value="${i},${j}">T${i+1} ↔ z${j+1}</option>`).join('');$('edge').value=selected.join(',');$('edge').disabled=!options.length;}
function refresh(){stop();compute();draw();}
function draw(){
 $('status').textContent=`One synthetic scan · ${R.events.length} compatible assignments · ${R.graph.edges} candidate pairs · certain existing point targets; no undetected PPP.`;
 if(mode==='assignment')drawAssignment();if(mode==='bp')drawBP();if(mode==='hypotheses')drawHypotheses();
}
function drawAssignment(){
 $('scene').innerHTML=V.scene(scene,gated,M.GATE);$('weights').innerHTML=V.table(R.L,{weights:true});$('pdOut').textContent=scene.PD.toFixed(2);$('clutterOut').textContent=scene.c.toExponential(1);
}
function drawBP(){const h=R.bp.history[iteration],i=selected[0],j=selected[1];
 $('phase').textContent=h.kind==='init'?'Initial beliefs':`Sweep ${h.t} · ${h.kind==='mu'?'track → measurement':'measurement → track'}`;
 $('bp-table-title').textContent=$('compare').checked?'BP probabilities / exact reference':'Track association probabilities';$('graph').innerHTML=V.graph(R.L,h,selected);$('bpTable').innerHTML=V.table(h.marg,{reference:$('compare').checked?R.exact:null});
 $('mu-value').textContent=h.mu?h.mu[i][j].toFixed(3):'—';$('nu-value').textContent=h.nu[j][i].toFixed(3);
 $('bpGap').textContent=(100*M.maxDifference(h.marg,R.exact)).toFixed(2);
 const final=iteration===R.bp.history.length-1;
 $('bpStats').textContent=final?`${R.bp.converged?'Stopping tolerance met':'Iteration cap reached'} after ${R.bp.iterations} sweeps. Stabilized messages can still give approximate marginals.`:h.kind==='mu'?'The track message has changed. Track beliefs update after the return message.':h.kind==='nu'?'Return messages change the track beliefs. Continue until the messages stabilize.':'Return messages start at one. Select a pair, then step through both message directions.';
 $('back').disabled=iteration===0;$('step').disabled=final;$('end').disabled=final;
}
function drawHypotheses(){const top=M.eventMarginals(R.events,R.n,R.m,k);$('keep').max=R.events.length;$('keep').value=k;$('keepOut').textContent=k+' / '+R.events.length;$('mass').textContent=V.percent(top.mass);$('topGap').textContent=(100*M.maxDifference(top.marginals,R.exact)).toFixed(2);$('eventChart').innerHTML=V.bars(R.events,k);$('exactTable').innerHTML=V.table(R.exact);$('topTable').innerHTML=V.table(top.marginals);$('topStats').textContent=`${V.percent(top.discardedMass)} probability discarded. Full tail mass is known here because every event is enumerated.`;}
function preset(value){const PD=scene.PD,c=scene.c;scene=M.clone(M.DEFAULT);scene.PD=PD;scene.c=c;
 if(value==='separated'){scene.T.forEach((p,i)=>{p.x=120+220*i;p.y=190;scene.Z[i]={x:p.x+5,y:p.y-3};});scene.Z[3]={x:660,y:45};}
 if(value==='symmetric'){scene.T.forEach(p=>{p.x=320;p.y=190;p.S=[[500,0],[0,500]];});scene.Z=[{x:305,y:180},{x:320,y:205},{x:335,y:180},{x:560,y:60}];}
 gated=true;$('gate').checked=true;refresh();}
$('preset').onchange=e=>preset(e.target.value);
$('pd').oninput=e=>{scene.PD=+e.target.value;refresh();};$('clutter').oninput=e=>{scene.c=10**(+e.target.value);refresh();};$('gate').onchange=e=>{gated=e.target.checked;refresh();};
$('sceneReset').onclick=()=>{scene=M.clone(M.DEFAULT);gated=true;$('preset').value='default';$('pd').value=.9;$('clutter').value=Math.log10(scene.c);$('gate').checked=true;refresh();};
$('edge').onchange=e=>{selected=e.target.value.split(',').map(Number);drawBP();};$('compare').onchange=drawBP;
function step(){if(iteration<R.bp.history.length-1){iteration++;drawBP();}if(iteration===R.bp.history.length-1)stop();}
$('step').onclick=()=>{stop();step();};$('back').onclick=()=>{stop();iteration=Math.max(0,iteration-1);drawBP();};$('end').onclick=()=>{stop();iteration=R.bp.history.length-1;drawBP();};$('reset').onclick=()=>{stop();iteration=0;drawBP();};$('play').onclick=()=>{if(timer){stop();return;}if(iteration===R.bp.history.length-1)iteration=0;timer=setInterval(step,550);$('play').textContent='Pause';};
$('keep').oninput=e=>{k=+e.target.value;drawHypotheses();};$('keepAll').onclick=()=>{k=R.events.length;drawHypotheses();};$('hypReset').onclick=()=>{k=Math.min(5,R.events.length);drawHypotheses();};
let drag=null;const svg=$('scene');
function point(e){return new DOMPoint(e.clientX,e.clientY).matrixTransform(svg.getScreenCTM().inverse());}
svg.onpointerdown=e=>{const p=point(e);let distance=20**2;for(const name of ['T','Z'])scene[name].forEach((v,i)=>{const d=(v.x-p.x)**2+(v.y-p.y)**2;if(d<distance){distance=d;drag={name,i};}});if(drag){svg.setPointerCapture(e.pointerId);e.preventDefault();}};
svg.onpointermove=e=>{if(!drag)return;const p=point(e),v=scene[drag.name][drag.i];v.x=Math.max(20,Math.min(700,p.x));v.y=Math.max(20,Math.min(330,p.y));refresh();};
for(const name of ['pointerup','pointercancel'])svg.addEventListener(name,()=>{drag=null;});
document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});window.addEventListener('bento-live-visibility',e=>{if(e.detail.paused)stop();});
window.BPAssociationAudit={snapshot:()=>M.clone({mode,L:R.L,exact:R.exact,bp:R.bp.marginals,iterations:R.bp.iterations,converged:R.bp.converged,graph:R.graph,iteration,k,scene,gated,playing:!!timer,selected})};
compute();draw();
})();
