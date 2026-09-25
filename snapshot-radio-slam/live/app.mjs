import {C, makeSnapshot, rotationFromEulerDegrees, conditionalSolve, stackSystem, yawSweep, refineOrientation, feasibilityMask, degrees, norm, sub} from './math.mjs';
import {Scene3D} from './scene3d.mjs';

const $=id=>document.getElementById(id), params=new URLSearchParams(location.search);
const validViews=['geometry','linear','orientation'];
const defaultSettings={yaw:15,pitch:0,roll:0,clock:10,noise:0,selected:['los',...Array.from({length:6},(_,i)=>`single-${i+1}`)]};
let settings={...defaultSettings,selected:[...defaultSettings.selected]};
try{const saved=JSON.parse(sessionStorage.getItem('snapshot-radio-slam-lab-v1'));if(saved&&['yaw','pitch','roll','clock','noise'].every(k=>Number.isFinite(saved[k]))&&Array.isArray(saved.selected))settings={...settings,...saved};}catch{}
let view=validViews.includes(params.get('lab'))?params.get('lab'):'geometry';
let step='stack', highlighted='single-1', generated, result, scan, scanKey='', trace=[], orientationMessage='';
let queued=false;
const fmt=(x,n=3)=>Number.isFinite(x)?(Math.abs(x)<.5*10**(-n)?'0.'+'0'.repeat(n):x.toFixed(n)):'—';
const sci=x=>!Number.isFinite(x)?'—':x===0?'0':x<.001||x>=1e5?x.toExponential(2):fmt(x,3);
const vec=(a,n=3)=>a?`[${a.map(x=>fmt(x,n)).join(', ')}]`:'—';
const label=id=>id==='los'?'LoS':id.startsWith('single')?'S'+id.split('-')[1]:'D1';
const allIds=['los',...Array.from({length:6},(_,i)=>`single-${i+1}`),'double-1'];
settings.selected=settings.selected.filter(id=>allIds.includes(id));
const scene=new Scene3D($('scene'));
let sceneState={};
function updateScene(partial){sceneState={...sceneState,...partial};scene.setScene(sceneState);}
const plot=$('cost-plot'), plotContext=plot.getContext('2d');

function persist(){try{sessionStorage.setItem('snapshot-radio-slam-lab-v1',JSON.stringify(settings));}catch{}}
function syncInputs(){
  for(const [id,key] of [['yaw','yaw'],['pitch','pitch'],['roll','roll'],['source-clock','clock'],['noise','noise']])$(id).value=settings[key];
  for(const id of ['yaw','pitch','roll'])$(id).step='any';
}
function requestUpdate({clearTrace=true}={}){if(clearTrace){trace=[];orientationMessage='';}persist();if(queued)return;queued=true;requestAnimationFrame(()=>{queued=false;update();});}
function update(){
  try{
    generated=makeSnapshot({clockBiasNs:settings.clock,noise:settings.noise,includeDoubleBounce:true});
    const R=rotationFromEulerDegrees(settings.yaw,settings.pitch,settings.roll);
    result=conditionalSolve(generated.snapshot,R,settings.selected);
    for(const id of ['yaw','pitch','roll'])$(id+'-value').textContent=fmt(settings[id],2)+'°';
    $('source-clock-value').textContent=fmt(settings.clock,0)+' ns';
    $('noise-value').textContent=fmt(settings.noise,1)+' × SD';
    $('noise-description').textContent=`LoS SD: ${fmt(.3*settings.noise,2)} m / ${fmt(settings.noise,1)}°. Single bounce: twice these SDs. Fixed random realization.`;
    $('position').textContent=result.state?vec(result.state.position,2):'Unidentifiable';
    $('clock-estimate').textContent=result.state?fmt(result.state.clockBiasNs,4):'—';
    $('residual').textContent=result.state?sci(result.rmsResidual):'—';
    $('rank').textContent=result.rank+' / 4';
    renderPathControls();
    const paths=generated.paths.map(p=>({...p,label:label(p.id),kind:p.id==='los'?'los':p.id.startsWith('single')?'single':'double'}));
    const allBlocks=stackSystem(generated.snapshot,R).blocks;
    updateScene({bsPosition:generated.snapshot.bsPosition,truePosition:generated.truth.position,estimatedPosition:result.state?.position,
      trueRotation:generated.truth.rotation,candidateRotation:R,paths,selectedIds:settings.selected,highlightId:highlighted,showTruth:true,
      rays:result.state?allBlocks.filter(b=>settings.selected.includes(b.id)).map(b=>({id:b.id,origin:result.state.position,direction:b.v,length:9})):[]});
    renderStatus();renderMatrix();renderMeasurements();
    if(view==='orientation')renderOrientation();
  }catch(error){$('solve-status').textContent='The experiment could not be evaluated: '+error.message;$('solve-status').classList.add('warning');console.error(error);}
}
function renderPathControls(){
  if(!$('path-controls').children.length){
    for(const id of allIds){
      const row=document.createElement('label');row.className=id==='double-1'?'double':'';
      const input=document.createElement('input');input.type='checkbox';input.value=id;input.setAttribute('aria-label',id==='los'?'Include LoS':id==='double-1'?'Include double bounce D1':'Include single bounce '+label(id));
      const name=document.createElement('span');name.textContent=label(id)+(id==='double-1'?' · double bounce':'');row.append(input,name);
      input.addEventListener('change',()=>{settings.selected=input.checked?[...settings.selected,id]:settings.selected.filter(p=>p!==id);requestUpdate();});
      row.addEventListener('mouseenter',()=>{highlighted=id;updateScene({highlightId:id});});
      $('path-controls').append(row);
    }
  }
  for(const input of $('path-controls').querySelectorAll('input'))input.checked=settings.selected.includes(input.value);
}
function renderStatus(){
  const status=$('solve-status');status.classList.toggle('warning',!result.state||result.conditionNumber>1e6);
  if(!result.state){status.textContent=`Rank ${result.rank}/4: these paths do not give a unique position and clock at this orientation. A minimum-norm vector would not resolve the missing information.`;return;}
  const pErr=norm(sub(result.state.position,generated.truth.position)), bErr=result.state.clockBiasNs-generated.truth.clockBiasNs;
  const feasible=feasibilityMask(generated.snapshot,result.state).filter(p=>settings.selected.includes(p.id));
  const passed=feasible.filter(p=>p.feasible).length;
  status.textContent=`Conditional fit · ${settings.selected.length} paths · synthetic position error ${fmt(pErr,3)} m · clock error ${fmt(bErr,3)} ns · geometry checks ${passed}/${feasible.length}. Low residual alone does not certify the pose.`;
}
function renderMatrix(){
  const target=$('matrix-content');
  if(step==='directions'){
    const blocks=stackSystem(generated.snapshot,rotationFromEulerDegrees(settings.yaw,settings.pitch,settings.roll)).blocks;
    const block=blocks.find(b=>b.id===highlighted)||blocks[0];
    target.innerHTML=`<h3>Angles become global bearing vectors</h3><label for="inspect-path">Inspect path <select id="inspect-path">${blocks.map(b=>`<option value="${b.id}" ${b.id===block.id?'selected':''}>${label(b.id)}</option>`).join('')}</select></label><p>u = R<sub>BS</sub> a(AoD), v = R a(AoA). The arrival vector points from the UE toward the last interaction. <a href="https://arxiv.org/html/2607.04847v2#S3.E20" target="_top">Eqs. (20)–(21)</a>.</p><div class="formula">u = ${vec(block.u)}<br>v = ${vec(block.v)}</div><div class="matrix-pair"><div><h3>M · Eq. (10)</h3><pre>${block.M.map(r=>r.map(x=>fmt(x,4).padStart(8)).join(' ')).join('\n')}</pre></div><div><h3>Clock column w = v − u</h3><pre>${block.difference.map(x=>fmt(x,4)).join('\n')}</pre><p>ρ = ${fmt(block.rho,4)} m</p></div></div><p>M = vuᵀ + uvᵀ − (uᵀv + 1)I. Changing candidate R changes this matrix.</p>`;
    $('inspect-path').addEventListener('change',event=>{highlighted=event.target.value;updateScene({highlightId:highlighted});renderMatrix();});
  }else if(step==='stack'){
    target.innerHTML=`<h3>${result.A.length} scalar rows, 4 shared unknowns</h3><div class="formula">[ Mᵢ &nbsp; <span class="orange">vᵢ − uᵢ</span> ] [ tₓ, tᵧ, t<sub>z</sub>, <span class="orange">B</span> ]ᵀ = ρᵢ(vᵢ − uᵢ)</div><p>Three rows per included path. B = cb in metres. This is <a href="https://arxiv.org/html/2607.04847v2#S3.E18" target="_top">Eq. (18)</a> with both clock signs reversed. Three rows need not be three independent constraints.</p><table aria-label="Actual stacked system A and right hand side"><thead><tr><th>Path / row</th><th>tₓ</th><th>tᵧ</th><th>t<sub>z</sub></th><th class="clock-column">B</th><th>RHS</th></tr></thead><tbody>${result.blocks.flatMap(block=>block.block.map((row,j)=>`<tr class="${j===0?'path-start':''}"><td class="path-column">${label(block.id)} · ${['x','y','z'][j]}</td>${row.map((x,k)=>`<td class="${k===3?'clock-column':''}">${fmt(x,4)}</td>`).join('')}<td>${fmt(block.rhs[j],4)}</td></tr>`)).join('')}</tbody></table>`;
  }else if(step==='svd'){
    const max=result.singularValues[0]||1;
    target.innerHTML=`<h3>The pseudoinverse uses singular directions</h3><div class="formula">A = UΣVᵀ<br>x̂ = Σⱼ vⱼ (uⱼᵀb / σⱼ), &nbsp; σⱼ &gt; tolerance</div><p><a href="https://arxiv.org/html/2607.04847v2#S3.E18" target="_top">Eq. (18): x̂ = A†b.</a> The implementation computes an SVD directly. Small singular values expose weakly constrained combinations.</p><div class="singular-values">${result.singularValues.map((s,i)=>`<div><span>${sci(s)}</span><i style="height:${Math.max(2,s/max*55)}px"></i><span>σ${i+1}</span></div>`).join('')}</div><p>Rank ${result.rank}/4 · condition number ${Number.isFinite(result.conditionNumber)?fmt(result.conditionNumber,2):'∞'} · relative cutoff 10⁻¹¹</p>${result.modes?`<table aria-label="SVD projection coefficients"><thead><tr><th>Mode</th><th>σⱼ</th><th>uⱼᵀb</th><th>(uⱼᵀb)/σⱼ</th></tr></thead><tbody>${result.modes.map((m,i)=>`<tr><td>${i+1}</td><td>${sci(m.sigma)}</td><td>${fmt(m.projection,4)}</td><td>${m.retained?fmt(m.coefficient,4):'discarded'}</td></tr>`).join('')}</tbody></table>`:''}${result.state?`<div class="result-vector">${[...result.state.displacement,result.state.clockBiasM].map((x,i)=>`<div class="${i===3?'clock-value':''}"><span>${['tₓ · m','tᵧ · m','t_z · m','B · m'][i]}</span><strong>${fmt(x,4)}</strong></div>`).join('')}</div><div class="formula clock-formula">b̂ = B̂ / c = ${fmt(result.state.clockBiasNs,6)} ns</div><p>Add the known BS position to t̂ to get the UE position.</p>`:'<p><b>A unique four-dimensional solution is unavailable.</b> The displayed rank explains why the position and clock outputs are withheld.</p>'}`;
  }else{
    const scaleMax=Math.max(1,...(result.residuals||[]).map(r=>r.norm));
    target.innerHTML=`<h3>Do the paths agree with one state?</h3><div class="formula">rᵢ = Mᵢt̂ − (ρᵢ − B̂)(vᵢ − uᵢ)</div><p><a href="https://arxiv.org/html/2607.04847v2#S3.E11" target="_top">Eq. (11)</a>. These are unified coarse residual norms in metres. Faded rows are excluded from the fit but still evaluated.</p>${result.state?result.residuals.map(r=>`<div class="residual-row ${r.included?'':'excluded'}"><span>${label(r.id)}${r.included?'':' (out)'}</span><div class="residual-track"><i style="width:${Math.max(.2,r.norm/scaleMax*100)}%;background:${r.id==='double-1'?'#7c4dbe':'#0a6b5e'}"></i></div><b>${sci(r.norm)}</b></div>`).join(''):'<p>Select enough informative paths to evaluate a unique fit.</p>'}<p>RMS = √(Σ included ‖rᵢ‖² / number of included paths). Algorithm 1 uses residual thresholding and a truncated cost across candidate subsets. This lab selects paths manually.</p>`;
  }
}
function renderMeasurements(){
  $('truth-summary').textContent=`Synthetic reference only: UE ${vec(generated.truth.position,1)} m; yaw 30°, pitch 12°, roll −8°; clock ${fmt(settings.clock,1)} ns. BS ${vec(generated.snapshot.bsPosition,1)} m with identity orientation. Angles below are measured in the local array frames.`;
  $('measurement-table').innerHTML=`<table><thead><tr><th>Path</th><th>ρ · m</th><th>τ · ns</th><th>AoD az · °</th><th>AoD el · °</th><th>AoA az · °</th><th>AoA el · °</th></tr></thead><tbody>${generated.snapshot.measurements.map(m=>`<tr class="${settings.selected.includes(m.id)?'':'excluded'}"><td>${label(m.id)}</td><td>${fmt(m.rho,4)}</td><td>${fmt(m.rho/C*1e9,4)}</td>${[...m.aoD,...m.aoA].map(x=>`<td>${fmt(degrees(x),3)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}
function ensureScan(){
  const key=JSON.stringify([settings.pitch,settings.roll,settings.clock,settings.noise,[...settings.selected].sort()]);
  if(key!==scanKey){scan=yawSweep(generated.snapshot,{pitchDeg:settings.pitch,rollDeg:settings.roll,stepDeg:2,selectedIds:settings.selected});scanKey=key;}
}
function renderOrientation(){
  ensureScan();$('scan-context').textContent=`Pitch ${fmt(settings.pitch,2)}° · roll ${fmt(settings.roll,2)}°`;
  $('apply-yaw').disabled=!scan.best;$('refine').disabled=!result.state;
  $('search-status').textContent=orientationMessage||(scan.best?`Best 2° grid point: yaw ${fmt(scan.best.yawDeg,0)}°, cost ${sci(scan.best.cost)} m². This is a fixed-tilt slice of the full 3D objective.`:'No unique linear fit along this yaw scan. Add informative paths.');
  $('iteration-trace').innerHTML=trace.length?`<table aria-label="Local orientation iteration history"><thead><tr><th>Iteration</th><th>Σ ‖rᵢ‖² · m²</th><th>Rotation step · rad</th></tr></thead><tbody>${trace.map(h=>`<tr><td>${h.iteration}</td><td>${sci(h.cost)}</td><td>${h.stepNorm===null?'—':sci(h.stepNorm)}</td></tr>`).join('')}</tbody></table>`:'';
  drawPlot();
}
function drawPlot(){
  if(!scan||view!=='orientation')return;
  const box=plot.getBoundingClientRect(),w=box.width,h=box.height;if(w<1||h<1)return;
  const dpr=Math.min(devicePixelRatio||1,2);plot.width=Math.round(w*dpr);plot.height=Math.round(h*dpr);plotContext.setTransform(dpr,0,0,dpr,0,0);
  const ctx=plotContext,left=57,right=w-17,top=22,bottom=h-34;
  ctx.clearRect(0,0,w,h);ctx.fillStyle='#fff';ctx.fillRect(0,0,w,h);ctx.font='11px Arial';
  const finite=scan.samples.filter(p=>Number.isFinite(p.cost));
  if(!finite.length){ctx.fillStyle='#51606e';ctx.fillText('No identifiable position/clock system for the selected paths.',30,h/2);return;}
  const logs=finite.map(p=>Math.log10(Math.max(1e-12,p.cost)));
  if(result.state)logs.push(Math.log10(Math.max(1e-12,result.cost)));
  const ymin=Math.floor(Math.min(...logs)),ymax=Math.max(ymin+1,Math.ceil(Math.max(...logs)));
  const xx=yaw=>left+(yaw+180)/360*(right-left), yy=cost=>bottom-(Math.log10(Math.max(1e-12,cost))-ymin)/(ymax-ymin)*(bottom-top);
  ctx.strokeStyle='#e1e7ec';ctx.fillStyle='#51606e';ctx.textAlign='right';
  const stride=Math.max(1,Math.ceil((ymax-ymin)/4));
  for(let exponent=ymin;exponent<=ymax;exponent+=stride){const y=bottom-(exponent-ymin)/(ymax-ymin)*(bottom-top);ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();ctx.fillText('10^'+exponent,left-7,y+4);}
  ctx.textAlign='center';for(const angle of [-180,-90,0,90,180])ctx.fillText(angle+'°',xx(angle),bottom+17);
  ctx.fillText('Candidate yaw',w/2,h-4);ctx.textAlign='left';ctx.fillText('Profiled cost · m² · log scale',left,12);
  ctx.strokeStyle='#0a6b5e';ctx.lineWidth=2;ctx.beginPath();let started=false;for(const p of scan.samples){if(!Number.isFinite(p.cost)){started=false;continue;}if(!started){ctx.moveTo(xx(p.yawDeg),yy(p.cost));started=true;}else ctx.lineTo(xx(p.yawDeg),yy(p.cost));}ctx.stroke();
  if(scan.best){ctx.fillStyle='#1874b8';ctx.beginPath();ctx.arc(xx(scan.best.yawDeg),yy(scan.best.cost),4,0,2*Math.PI);ctx.fill();}
  const x=xx(settings.yaw);ctx.strokeStyle='#d76809';ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,bottom);ctx.stroke();ctx.setLineDash([]);
  if(result.state){ctx.fillStyle='#d76809';ctx.beginPath();ctx.arc(x,yy(result.cost),5,0,2*Math.PI);ctx.fill();}
}
function switchView(next){
  view=next;for(const v of validViews){$(v+'-panel').hidden=v!==next;document.querySelector(`[data-view="${v}"]`).setAttribute('aria-pressed',String(v===next));}
  const url=new URL(location.href);url.searchParams.set('lab',next);history.replaceState(null,'',url);document.querySelector('.expand').href='./?lab='+next;
  if(generated&&next==='orientation')renderOrientation();if(next==='geometry')requestAnimationFrame(()=>updateScene({}));
}
for(const button of document.querySelectorAll('[data-view]'))button.addEventListener('click',()=>switchView(button.dataset.view));
for(const button of document.querySelectorAll('[data-step]'))button.addEventListener('click',()=>{step=button.dataset.step;for(const b of document.querySelectorAll('[data-step]'))b.setAttribute('aria-pressed',String(b===button));renderMatrix();});
for(const [id,key] of [['yaw','yaw'],['pitch','pitch'],['roll','roll'],['source-clock','clock'],['noise','noise']])$(id).addEventListener('input',event=>{settings[key]=Number(event.target.value);$('orientation-note').textContent='Candidate R = Rz(yaw) Ry(pitch) Rx(roll).';requestUpdate();});
$('true-orientation').addEventListener('click',()=>{settings.yaw=30;settings.pitch=12;settings.roll=-8;syncInputs();$('orientation-note').textContent='Conditional test: true orientation supplied explicitly. Position and clock are still solved from observations.';requestUpdate();});
$('reset').addEventListener('click',()=>{settings={...defaultSettings,selected:[...defaultSettings.selected]};syncInputs();scene.resetView();$('orientation-note').textContent='Candidate R = Rz(yaw) Ry(pitch) Rx(roll).';requestUpdate();});
$('inliers').addEventListener('click',()=>{settings.selected=[...defaultSettings.selected];requestUpdate();});
$('los-only').addEventListener('click',()=>{settings.selected=['los'];requestUpdate();});
$('reset-camera').addEventListener('click',()=>scene.resetView());
$('apply-yaw').addEventListener('click',()=>{ensureScan();if(!scan.best)return;settings.yaw=scan.best.yawDeg;syncInputs();requestUpdate();});
$('refine').addEventListener('click',()=>{
  $('refine').disabled=true;
  requestAnimationFrame(()=>{
    const solved=refineOrientation(generated.snapshot,rotationFromEulerDegrees(settings.yaw,settings.pitch,settings.roll),{selectedIds:settings.selected,maxIterations:120});
    if(solved.state){const R=solved.state.rotation;settings.pitch=degrees(Math.asin(Math.max(-1,Math.min(1,-R[2][0]))));settings.yaw=degrees(Math.atan2(R[1][0],R[0][0]));settings.roll=degrees(Math.atan2(R[2][1],R[2][2]));syncInputs();trace=solved.history;orientationMessage=`${solved.iterations} local updates · ${solved.converged?'step tolerance reached':'stopped without a convergence certificate'} · final cost ${sci(solved.cost)} m². Each update re-solves position and clock.`;$('orientation-note').textContent='Orientation returned by local SO(3) refinement. Displayed angles parameterize its rotation matrix.';requestUpdate({clearTrace:false});}else{orientationMessage=solved.message;renderOrientation();}
  });
});
plot.addEventListener('click',event=>{const b=plot.getBoundingClientRect();settings.yaw=Math.max(-180,Math.min(180,((event.clientX-b.left-57)/(b.width-74))*360-180));syncInputs();requestUpdate();});
plot.addEventListener('keydown',event=>{if(event.key==='ArrowLeft'||event.key==='ArrowRight'){event.preventDefault();settings.yaw=Math.max(-180,Math.min(180,settings.yaw+(event.key==='ArrowLeft'?-2:2)));syncInputs();requestUpdate();}});
new ResizeObserver(()=>drawPlot()).observe(plot);
syncInputs();switchView(view);update();
