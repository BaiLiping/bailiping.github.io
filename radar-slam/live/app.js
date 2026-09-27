(function(){
'use strict';
const M=window.RadarModel,{C,text,axes,pairBoxes,drawMapped}=window.RadarRender,$=id=>document.getElementById(id),activeDemo=window.activeDemo;
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const finite=(v,d=2)=>Number.isFinite(v)?v.toFixed(d):'—';
const stats=(id,items)=>{$(id).innerHTML=items.map(([a,b])=>`<div class="metric"><span>${a}</span><strong>${b}</strong></div>`).join('');};
const num=id=>Number($(id).value),on=id=>$(id).checked;
function input(id,callback){const el=$(id);el.addEventListener('input',()=>{const output=$(id+'-out');if(output)output.textContent=el.value+(el.dataset.unit||'');clearPresets();callback();});}
function set(id,value){const el=$(id);if(el.type==='checkbox')el.checked=value;else el.value=value;const out=$(id+'-out');if(out)out.textContent=el.value+(el.dataset.unit||'');}
function clearPresets(){document.querySelectorAll('[data-preset]').forEach(b=>b.setAttribute('aria-pressed','false'));}
function presets(fn){document.querySelectorAll('[data-preset]').forEach(b=>{b.setAttribute('aria-pressed','false');b.onclick=()=>{clearPresets();fn(b.dataset.preset);b.setAttribute('aria-pressed','true');};});}
function insight(s){$('insight').textContent=s;}
const painters=[];function register(id,paint){const el=$(id);const redraw=()=>{const width=el.clientWidth,height=el.clientHeight;if(!width||!height||document.documentElement.dataset.bentoPaused==='true')return;const dpr=Math.min(devicePixelRatio||1,2);el.width=Math.round(width*dpr);el.height=Math.round(height*dpr);const c=el.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);paint(c,width,height);};new ResizeObserver(redraw).observe(el);painters.push(redraw);return redraw;}
function history(id,values,color=C.green){const valid=values.filter(Number.isFinite);if(!valid.length)return;const max=Math.max(...valid,1e-6),points=valid.map((v,i)=>`${i*560/Math.max(1,valid.length-1)},${35-v/max*30}`).join(' ');$(id).innerHTML=`<polyline points="${points}" fill="none" stroke="${color}" stroke-width="2.2"/><text x="565" y="20" fill="${color}" font-size="12" font-family="sans-serif">${finite(valid.at(-1),2)}</text>`;}
let paused=false,timers=new Set();const repeat=(fn,ms)=>{const id=setInterval(()=>{if(!paused)fn();},ms);timers.add(id);return id;};const stop=id=>{clearInterval(id);timers.delete(id);};
function visibility(value){paused=value;if(!paused)painters.forEach(f=>f());}
addEventListener('bento-live-visibility',e=>visibility(e.detail.paused));document.addEventListener('visibilitychange',()=>visibility(document.hidden));addEventListener('pagehide',()=>{timers.forEach(clearInterval);timers.clear();});
window.radarDebug={demo:activeDemo};

if(activeDemo==='whole-run'){
let missionTimer=null;
let mission=M.makeMission();
function renderMission(c,w,h){const k=num('mission-frame'),boxes=pairBoxes(w,h);text(c,'RADAR FRAME · scan '+k,boxes[0].x,boxes[0].y-12,C.green,10);text(c,'WORLD FRAME · accumulated estimate',boxes[1].x,boxes[1].y-12,C.green,10);const local=axes(c,boxes[0],[-16,16,-16,16],'x in radar (m)','y in radar (m)',true),world=axes(c,boxes[1],[-14,14,-6,24],'world x (m)','world y (m)',true);mission.scans[k].forEach(q=>local.dot(q,C.blue,2.5));local.pose([0,0,0]);if(on('mission-truth')){mission.world.forEach(q=>world.dot(q,C.gray,1.7,true));world.line(mission.truth,C.gray,1.5,[4,4]);}if(on('mission-map'))drawMapped(world,mission,mission.poses,k);world.line(mission.poses.slice(0,k+1),C.green,2.3);world.pose(mission.poses[k]);const t=mission.poses[k];stats('mission-stats',[['Measured returns',String(mission.scans[k].length)],['Pose x, y (m)',`${finite(t[0],1)}, ${finite(t[1],1)}`],['Position error',`${finite(Math.hypot(t[0]-mission.truth[k][0],t[1]-mission.truth[k][1]))} m`],['Gated ICP RMSE',k?`${finite(mission.quality[k])} m`:'bootstrap']]);}
const drawMission=register('mission-canvas',renderMission);['mission-frame','mission-map','mission-truth'].forEach(id=>input(id,drawMission));
function setFrame(k){$('mission-frame').value=clamp(k,0,mission.N);$('mission-frame-out').textContent=$('mission-frame').value+' / 48';$('mission-step').disabled=num('mission-frame')===mission.N;drawMission();}
$('mission-step').onclick=()=>setFrame(num('mission-frame')+1);$('mission-reset').onclick=()=>{if(missionTimer)stop(missionTimer);missionTimer=null;$('mission-play').textContent='Play';setFrame(0);};
$('mission-play').onclick=()=>{if(missionTimer){stop(missionTimer);missionTimer=null;$('mission-play').textContent='Play';return;}if(num('mission-frame')===mission.N)setFrame(0);$('mission-play').textContent='Pause';missionTimer=repeat(()=>{const k=num('mission-frame');if(k>=mission.N){stop(missionTimer);missionTimer=null;$('mission-play').textContent='Play';return;}setFrame(k+1);},330);};


const missionCache=new Map([[.35,mission]]);
input('mission-bias',()=>{const bias=on('mission-bias')?.35:0;if(!missionCache.has(bias))missionCache.set(bias,M.makeMission(bias));mission=missionCache.get(bias);drawMission();insight(bias?'Declared bias: 0.35° is added after every scan match. Compare drift at the revisit.':'No added heading bias. Measurement noise and imperfect correspondences still affect the estimate.');});
presets(name=>{if(missionTimer)stop(missionTimer);missionTimer=null;$('mission-play').textContent='Play';setFrame({start:0,half:24,loop:48}[name]);});
window.radarDebug.getState=()=>({frame:num('mission-frame'),bias:mission.biasDeg,pose:mission.poses[num('mission-frame')]});
setFrame(24);

}

if(activeDemo==='range'){
let rangeData=M.rangeSpectrum();
const drawRange=register('range-canvas',(c,w,h)=>{const top=axes(c,{x:46,y:24,w:w-63,h:88},[0,63,-2.1,2.1],'sample n','amplitude');top.line(rangeData.re.slice(0,64).map((x,i)=>[i,x]),C.green,1.4);top.line(rangeData.im.slice(0,64).map((x,i)=>[i,x]),C.blue,1,[3,2]);text(c,'I (solid)     Q (dashed)',w-20,15,C.muted,10,'right');const bottom=axes(c,{x:46,y:185,w:w-63,h:h-231},[0,12,-65,3],'range (m)','normalized power (dB)');bottom.line(rangeData.spectrum,C.green,2);for(const r of [rangeData.r1,rangeData.r2])bottom.line([[r,-65],[r,3]],C.orange,1,[4,3]);text(c,'RANGE SPECTRUM · dashed markers = injected ranges',46,170,C.green,10);stats('range-stats',[['Nominal resolution',finite(rangeData.resolution,3)+' m'],['True separation',finite(num('range-sep'),2)+' m']]);});
function updateRange(){rangeData=M.rangeSpectrum(num('range-band'),num('range-sep'),on('range-hann'));drawRange();insight(`The reflector spacing spans ${finite(num('range-sep')/rangeData.resolution,2)} nominal range cells. ${on('range-hann')?'Hann windowing broadens the peaks and suppresses sidelobes.':'Without a window, sidelobes rise while the main lobe narrows.'}`);}['range-band','range-sep','range-hann'].forEach(id=>input(id,updateRange));

presets(name=>{set('range-band',name==='merged'?.15:name==='resolved'?1.2:.45);set('range-sep',.45);set('range-hann',true);updateRange();});
window.radarDebug.getState=()=>({resolution:rangeData.resolution,separation:num('range-sep')});

}

if(activeDemo==='cfar'){
let cfarSeed=12,cfarData=M.cfar();
const drawCfar=register('cfar-canvas',(c,w,h)=>{const max=Math.max(...cfarData.power,...cfarData.thresholds.filter(x=>x!==null))*1.13,pl=axes(c,{x:46,y:32,w:w-64,h:h-84},[0,127,0,max],'range-bin index','power (arbitrary units)');const cell=num('cfar-cell');pl.clip(()=>{for(const [lo,hi,color]of [[cell-10.5,cell-2.5,'#deebe2'],[cell+2.5,cell+10.5,'#deebe2'],[cell-2.5,cell-.5,'#f6e7d5'],[cell+.5,cell+2.5,'#f6e7d5']]){c.fillStyle=color;c.fillRect(pl.X(lo),32,pl.X(hi)-pl.X(lo),h-84);}});pl.line([[cell,0],[cell,max]],C.orange,1.5,[4,3]);pl.line(cfarData.power.map((x,i)=>[i,x]),C.blue,1.1);pl.line(cfarData.thresholds.map((x,i)=>[i,x]),C.green,2);cfarData.hits.forEach(k=>pl.dot([k,cfarData.power[k]],C.orange,5,true));cfarData.targets.forEach(k=>pl.line([[k,0],[k,max]],C.gray,1,[2,4]));text(c,'BLUE · power     GREEN · threshold     CIRCLES · detections',46,17,C.muted,10);stats('cfar-stats',[['Design Pfa',(10**(-num('cfar-pfa'))).toExponential(0)],['Detected cells',String(cfarData.hits.length)],['Cell power / threshold',finite(cfarData.power[cell],1)+' / '+finite(cfarData.thresholds[cell],1)],['Cell decision',cfarData.hits.includes(cell)?'Detected':'Below threshold']]);});
function updateCfar(){cfarData=M.cfar(num('cfar-pfa'),on('cfar-edge'),cfarSeed);drawCfar();}['cfar-pfa','cfar-edge'].forEach(id=>input(id,updateCfar));$('cfar-new').onclick=()=>{cfarSeed++;updateCfar();};

input('cfar-cell',drawCfar);$('cfar-reset').onclick=()=>{cfarSeed=12;updateCfar();};
presets(name=>{cfarSeed=12;set('cfar-edge',name==='edge');set('cfar-cell',name==='edge'?74:62);set('cfar-pfa',3);updateCfar();});
window.radarDebug.getState=()=>({seed:cfarSeed,cell:num('cfar-cell'),hits:cfarData.hits,threshold:cfarData.thresholds[num('cfar-cell')]});

}

if(activeDemo==='frames'){
const drawFrames=register('frames-canvas',(c,w,h)=>{const boxes=pairBoxes(w,h),t=[num('frames-x'),num('frames-y'),M.rad(num('frames-yaw'))],a=M.rad(30),q=[6*Math.cos(a),6*Math.sin(a)],point=M.transform(t,q),p=axes(c,boxes[0],[-2,9,-4,7],'radar x (m)','radar y (m)',true),g=axes(c,boxes[1],[-12,12,-12,12],'world x (m)','world y (m)',true);text(c,'SAME LOCAL RETURN',boxes[0].x,boxes[0].y-12,C.green,10);text(c,'PLACEMENT UNDER ASSUMED POSE',boxes[1].x,boxes[1].y-12,C.green,10);p.line([[0,0],q],C.blue,2);p.pose([0,0,0]);p.dot(q,C.blue,5);g.line([t,point],C.blue,1.7);g.pose(t);g.dot(point,C.blue,5);const ellipse=[];for(let i=0;i<=72;i++){const b=i*2*Math.PI/72,dr=2*.15*Math.cos(b),da=2*M.rad(3)*Math.sin(b);const perturb=[Math.cos(a)*dr-6*Math.sin(a)*da,Math.sin(a)*dr+6*Math.cos(a)*da];ellipse.push(M.transform(t,[q[0]+perturb[0],q[1]+perturb[1]]));}g.line(ellipse,C.orange,1.5);text(c,'Two-standard-deviation contour; measurement noise only',15,h-4,C.muted,9);stats('frames-stats',[['Local q (m)',`${finite(q[0],2)}, ${finite(q[1],2)}`],['World point (m)',`${finite(point[0],2)}, ${finite(point[1],2)}`]]);});['frames-x','frames-y','frames-yaw'].forEach(id=>input(id,drawFrames));

presets(name=>{set('frames-x',name==='default'?1.5:0);set('frames-y',name==='default'?-1:0);set('frames-yaw',name==='identity'?0:name==='quarter'?90:35);drawFrames();});
window.radarDebug.getState=()=>({pose:[num('frames-x'),num('frames-y'),M.rad(num('frames-yaw'))]});

}

if(activeDemo==='velocity'){
let velocityData=M.velocity(),ordinaryData=M.velocity(4,1,.25,130,false);
const drawVelocity=register('velocity-canvas',(c,w,h)=>{const v=velocityData,span=num('velocity-span')/2,max=Math.max(3,...v.data.map(p=>Math.abs(p.d)),num('velocity-x')+2),pl=axes(c,{x:48,y:34,w:w-67,h:h-86},[-span,span,-max,max],'bearing (degrees)','radial velocity (m/s)'),ids=new Set(v.inliers),truth=[],estimated=[];for(let i=0;i<=140;i++){const a=M.rad(-span+2*span*i/140);truth.push([M.deg(a),-v.truth[0]*Math.cos(a)-v.truth[1]*Math.sin(a)]);estimated.push([M.deg(a),-v.v[0]*Math.cos(a)-v.v[1]*Math.sin(a)]);}pl.line(truth,C.gray,2,[5,4]);pl.line(truth.map(p=>{const a=M.rad(p[0]);return [p[0],-ordinaryData.v[0]*Math.cos(a)-ordinaryData.v[1]*Math.sin(a)];}),C.orange,1.5,[3,3]);pl.line(estimated,C.green,2.2);v.data.forEach((p,i)=>pl.dot([M.deg(p.a),p.d],ids.has(i)?C.blue:C.orange,ids.has(i)?3:3.6,false,!ids.has(i)));text(c,'Positive = receding · Negative = approaching',48,18,C.muted,10);stats('velocity-stats',[['Estimated vx',finite(v.v[0],2)+' m/s'],['Estimated vy',finite(v.v[1],2)+' m/s'],['Used returns',`${v.inliers.length} / 80`],['Geometry condition',finite(v.condition,1)]]);});
function updateVelocity(){velocityData=M.velocity(num('velocity-x'),1,num('velocity-movers')/100,num('velocity-span'),on('velocity-robust'));ordinaryData=M.velocity(num('velocity-x'),1,num('velocity-movers')/100,num('velocity-span'),false);drawVelocity();insight(velocityData.condition>100?'Weak lateral geometry: even a robust fit has little independent information in this narrow view.':`Velocity error: selected fit ${finite(Math.hypot(velocityData.v[0]-velocityData.truth[0],velocityData.v[1]-1))} m/s; ordinary fit ${finite(Math.hypot(ordinaryData.v[0]-ordinaryData.truth[0],ordinaryData.v[1]-1))} m/s. Truth is evaluation only.`);}['velocity-x','velocity-movers','velocity-span','velocity-robust'].forEach(id=>input(id,updateVelocity));

presets(name=>{set('velocity-x',4);set('velocity-movers',name==='clean'?0:35);set('velocity-span',name==='narrow'?4:130);set('velocity-robust',true);updateVelocity();});
window.radarDebug.getState=()=>({fit:velocityData.v,ordinary:ordinaryData.v,condition:velocityData.condition,inliers:velocityData.inliers.length});

}

if(activeDemo==='icp'){
const icpData=M.makeICP();let icpPose=[0,0,0],icpPairs=[],icpPhase=0,icpIteration=0,icpTimer=null,icpHistory=[];
const drawICP=register('icp-canvas',(c,w,h)=>{const pl=axes(c,{x:43,y:30,w:w-63,h:h-81},[-8,8,-8,8],'reference-frame x (m)','reference-frame y (m)',true);for(const p of icpPairs)pl.line([M.transform(icpPose,icpData.source[p.i]),p.q],'#cad6c2',.8);icpData.target.forEach(p=>pl.dot(p,C.gray,3.7,true));icpData.source.forEach(p=>pl.dot(M.transform(icpPose,p),C.green,2.8));pl.pose(icpPose,C.orange);const pairs=M.associate(icpData.source,icpData.target,icpPose,num('icp-gate'),on('icp-robust')),rmse=pairs.length?Math.sqrt(pairs.reduce((s,p)=>s+p.d*p.d,0)/pairs.length):NaN;stats('icp-stats',[['Updates completed',String(icpIteration)],['Gated pairs',String(pairs.length)],['Estimated tx, ty',`${finite(icpPose[0],2)}, ${finite(icpPose[1],2)}`],['Heading / RMSE',`${finite(M.deg(icpPose[2]),1)}° / ${finite(rmse,2)} m`]]);if(icpHistory.length===icpIteration)icpHistory.push(rmse);else icpHistory[icpIteration]=rmse;history('icp-history',icpHistory);text(c,'Reference ○     Transformed scan ●     Pose ▶',43,16,C.muted,10);});
function stopICP(){if(icpTimer)stop(icpTimer);icpTimer=null;$('icp-run').textContent='Animate';$('icp-step').disabled=false;}
function resetICP(){stopICP();icpHistory=[];icpPose=[0,0,M.rad(num('icp-start'))];icpPairs=[];icpPhase=0;icpIteration=0;$('icp-step').textContent='1 · Find matches';$('insight').textContent='Initial translation is (0, 0). Find tentative nearest neighbors to begin.';drawICP();}
function stepICP(){if(icpPhase===0){icpPairs=M.associate(icpData.source,icpData.target,icpPose,num('icp-gate'),on('icp-robust'));icpPhase=1;$('icp-step').textContent='2 · Update pose';$('insight').textContent=`Association: ${icpPairs.length} tentative pairs pass the ${finite(num('icp-gate'),1)} m gate. Their weights are frozen for the next update.`;}else{const d=M.alignPairs(icpPairs);if(d){icpPose=M.compose(d,icpPose);icpIteration++;$('insight').textContent='Transformation updated from the weighted pairs. Find matches again because geometry has changed.';}else $('insight').textContent='Insufficient or degenerate matches. No pose update was accepted. Increase the gate or improve initialization.';icpPhase=0;$('icp-step').textContent='1 · Find matches';}drawICP();}
$('icp-step').onclick=stepICP;$('icp-reset').onclick=resetICP;$('icp-run').onclick=()=>{if(icpTimer){stopICP();return;}let remaining=40;$('icp-run').textContent='Pause';$('icp-step').disabled=true;icpTimer=repeat(()=>{stepICP();if(--remaining<=0)stopICP();},300);};['icp-start','icp-gate','icp-robust'].forEach(id=>input(id,resetICP));

presets(name=>{set('icp-start',name==='poor'?40:0);set('icp-gate',name==='tight'?.3:2);set('icp-robust',true);resetICP();});
window.radarDebug.getState=()=>({pose:icpPose,phase:icpPhase,iterations:icpIteration,history:icpHistory});
resetICP();

}

if(activeDemo==='optimize'){
let mission=M.makeMission();
let graphPoses=mission.poses.map(p=>p.slice()),graphIteration=0,graphEdges=M.graphEdges(mission,.15),graphCost=M.graphCost(graphPoses,graphEdges),graphInitialCost=graphCost,lastStep=null,graphTimer=null,graphHistory=[graphCost];
const positionRMSE=poses=>Math.sqrt(poses.reduce((sum,p,i)=>sum+(p[0]-mission.truth[i][0])**2+(p[1]-mission.truth[i][1])**2,0)/poses.length);
const drawGraph=register('graph-canvas',(c,w,h)=>{const boxes=pairBoxes(w,h);const before=axes(c,boxes[0],[-16,16,-6,24],'world x (m)','world y (m)',true),pl=axes(c,boxes[1],[-16,16,-6,24],'world x (m)','world y (m)',true);if(on('graph-map')){drawMapped(before,mission,mission.poses,mission.N);drawMapped(pl,mission,graphPoses,mission.N);}before.line(mission.truth,C.gray,1.7,[5,4]);before.line(mission.poses,C.orange,2.2);before.pose(mission.poses.at(-1),C.orange);pl.line(mission.truth,C.gray,1.7,[5,4]);pl.line(graphPoses,C.green,2.2);graphPoses.forEach(p=>pl.dot(p,C.green,2));pl.line([graphPoses[0],graphPoses.at(-1)],C.purple,2,[3,2]);pl.pose(graphPoses[0],C.ink);pl.pose(graphPoses.at(-1),C.purple);text(c,'BEFORE · accumulated odometry',boxes[0].x,boxes[0].y-12,C.orange,10);text(c,'AFTER · current graph estimate',boxes[1].x,boxes[1].y-12,C.green,10);stats('graph-stats',[['Solver updates',String(graphIteration)],['Weighted objective',finite(graphCost,3)],['Position RMSE',finite(positionRMSE(graphPoses))+' m'],['Initial position RMSE',finite(positionRMSE(mission.poses))+' m']]);history('graph-history',graphHistory);const inliers=mission.loop.pairs.filter(p=>p.d<.45),rms=Math.sqrt(inliers.reduce((s,p)=>s+p.d*p.d,0)/inliers.length);$('insight').textContent=on('graph-false')?'A false (4 m, −3 m, 24°) loop has been inserted. Cost reduction alone does not make this measurement correct.':`Supplied loop candidate 0 ↔ 48: ${inliers.length} matches below 0.45 m, in-gate RMSE ${finite(rms,3)} m. ${lastStep?(lastStep.step<1e-6?'Numerically converged for this objective.':'A damped update was accepted.'):'Optimize to reconcile the loop with the odometry chain.'}`;});
function stopGraph(){if(graphTimer)stop(graphTimer);graphTimer=null;$('graph-solve').textContent='Optimize';$('graph-step').disabled=false;}
function resetGraph(){stopGraph();graphEdges=M.graphEdges(mission,num('graph-sigma'),on('graph-false'));graphPoses=mission.poses.map(p=>p.slice());graphIteration=0;lastStep=null;graphCost=M.graphCost(graphPoses,graphEdges,on('graph-robust'));graphInitialCost=graphCost;graphHistory=[graphCost];drawGraph();}
function stepGraph(){lastStep=M.graphStep(graphPoses,graphEdges,on('graph-robust'));if(lastStep.accepted){graphPoses=lastStep.poses;graphIteration++;graphCost=lastStep.cost;graphHistory.push(graphCost);}drawGraph();}
$('graph-step').onclick=stepGraph;$('graph-reset').onclick=resetGraph;$('graph-solve').onclick=()=>{if(graphTimer){stopGraph();return;}let remaining=20;$('graph-solve').textContent='Pause';$('graph-step').disabled=true;graphTimer=repeat(()=>{stepGraph();if(--remaining<=0||!lastStep.accepted||lastStep.step<1e-7)stopGraph();},160);};['graph-sigma','graph-false','graph-robust'].forEach(id=>input(id,resetGraph));input('graph-map',drawGraph);
$('graph-export').onclick=()=>{const record={description:'Synthetic 2D radar-SLAM teaching run. Truth is display-only, not an optimization input.',units:{distance:'m',angle:'rad'},headingBiasDegPerFrame:mission.biasDeg,loopTranslationSigma:num('graph-sigma'),positionRMSE:positionRMSE(graphPoses),costHistory:graphHistory,loopCandidate:'supplied: frames 0 and 48',falseLoop:on('graph-false'),huberLoop:on('graph-robust'),objective:graphCost,iterations:graphIteration,truthDisplayOnly:mission.truth,initialPoses:mission.poses,optimizedPoses:graphPoses,edges:graphEdges,scans:mission.scans};const url=URL.createObjectURL(new Blob([JSON.stringify(record,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='radar-slam-demo-run.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);};

presets(name=>{set('graph-sigma',.15);set('graph-false',name!=='valid');set('graph-robust',name==='robust');resetGraph();});
window.radarDebug.getState=()=>({cost:graphCost,initialCost:graphInitialCost,iterations:graphIteration,rmse:positionRMSE(graphPoses),history:graphHistory,poses:graphPoses});

}

requestAnimationFrame(()=>painters.forEach(f=>f()));
})();
