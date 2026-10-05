import {CONFIG,DEFAULT_SENSOR,cloneScene,render,project,covariance,rotation,multiply,transpose,rad,deg,clamp,peaks,TRAIN_INITIAL,TRAIN_TARGET,TRAIN_GRID,trainProfile,trainLoss,trainStep} from './model.mjs';
const $=id=>document.getElementById(id);
const state={scene:cloneScene(),sensor:{...DEFAULT_SENSOR},selected:0,stage:3,mode:'total',view:'top',inspector:'shape',beam:218,threshold:-24,detect:false};
let result,frame=0,drag=null,sceneProjection;
const C={ink:'#1d2923',muted:'#64756a',grid:'#e5ebe3',green:'#2f7d5b',deep:'#215c43',orange:'#b76035',grey:'#83918c',surface:'#fff'};
const stageContent=[
 ['A 3D ellipsoid becomes a polar footprint.','Σ<sub>rθ</sub> = [J Σ<sub>sensor</sub> Jᵀ]<sub>rθ,rθ</sub>','Geometry only: unit-peak Gaussian footprints, before return weights and antenna gain. Rotate or stretch the selected splat to change the footprint.'],
 ['Weight the footprint, then add its power.','I<sub>elev</sub> ∝ r⁻⁴ ∑ᵢ G<sub>φ</sub>(φᵢ)² σᵢ · footprintᵢ','Elevation is integrated out. Returns at the same range and azimuth add, rather than being transparency-composited. The plot shows center samples of the oversampled image.'],
 ['A beam collects neighboring directions.','I<sub>azi</sub> = Conv<sub>θ</sub>(I<sub>elev</sub>; kernel 2Q, stride Q)','The high-resolution azimuth image is convolved and downsampled with circular padding. Q = 4 here; the paper uses Q = 10. Change azimuth beamwidth in the Radar controls.'],
 ['A return occupies more than one range bin.','I = K<sub>r</sub> ∗ I<sub>azi</sub>','The Gaussian range kernel approximates spectral leakage after windowing. Increase its width in the Radar controls; the geometry has not changed.']
];
function selected(){return state.scene[state.selected];}
function source(){return [result.footprint,result.elevation,result.azimuth,result.final][state.stage];}
function fmt(n,d=2){return Number.isFinite(n)?n.toFixed(d):'—';}
function db(n){return Math.max(-70,10*Math.log10(Math.max(n,1e-10)));}
function requestRender(){if(frame)return;frame=requestAnimationFrame(()=>{frame=0;result=render(state.scene,state.sensor,state.mode);drawAll();updateMath();updateStage();});}
function pressed(selector,value,key){document.querySelectorAll(selector).forEach(b=>b.setAttribute('aria-pressed',String(b.dataset[key]===String(value))));}
function rebuildSelect(){
 $('selected-splat').innerHTML=state.scene.map((g,i)=>`<option value="${i}">G${i+1} · ${g.name}</option>`).join('');$('selected-splat').value=state.selected;
}
const shapeFields=[['x','Center x',-38,38,.1,'m'],['y','Center y',-38,38,.1,'m'],['z','Height z',.2,10,.1,'m'],['sx','Scale s₁',.2,3.5,.05,'m'],['sy','Scale s₂',.2,3.5,.05,'m'],['sz','Scale s₃',.2,3.5,.05,'m'],['yaw','Yaw',-180,180,1,'°'],['pitch','Pitch',-80,80,1,'°']];
const returnFields=[['rho','Base reflectivity',.05,3,.05,''],['alpha','Occupancy α',0,1,.01,''],['eta','Noise η',0,1,.01,''],['directional','View dependence',0,.95,.05,''],['facing','Reflectivity direction',-180,180,1,'°']];
const sensorFields=[['x','Radar x',-10,10,.25,'m'],['y','Radar y',-10,10,.25,'m'],['z','Radar height',.5,6,.1,'m'],['yaw','Radar yaw',-180,180,1,'°'],['azWidth','Azimuth width',.45,1.8,.05,'°'],['elWidth','Elevation width',2,35,.5,'°'],['leakage','Range kernel σ',0,1.2,.01,'m']];
function controlFormat(value,step,unit){return `${fmt(value,step>=1?0:step>=.1?1:2)}${unit?' '+unit:''}`;}
function rebuildControls(){
 const fields=state.inspector==='shape'?shapeFields:state.inspector==='return'?returnFields:sensorFields;
 const target=state.inspector==='sensor'?state.sensor:selected();
 $('controls').innerHTML='<div class="control-grid">'+fields.map(([k,label,min,max,step,unit])=>`<label class="slider-label ${state.inspector==='return'?'wide':''}" for="param-${k}">${label}<output id="value-${k}" for="param-${k}">${controlFormat(target[k],step,unit)}</output><input id="param-${k}" data-param="${k}" type="range" min="${min}" max="${max}" step="${step}" value="${target[k]}"></label>`).join('')+'</div>'+(state.inspector==='sensor'?'<label class="check-label"><input id="falloff" type="checkbox" '+(state.sensor.falloff?'checked':'')+'> Apply 1/r⁴ power loss</label>':'');
 fields.forEach(([k,label,min,max,step,unit])=>$(`param-${k}`).addEventListener('input',e=>{target[k]=Number(e.target.value);$(`value-${k}`).textContent=controlFormat(target[k],step,unit);requestRender();}));
 if($('falloff'))$('falloff').addEventListener('change',e=>{state.sensor.falloff=e.target.checked;requestRender();});
 $('parameter-note').textContent=state.inspector==='shape'?'Scales are standard deviations along rotated principal axes. Contours show 1σ, 2σ and 3σ; the point marks the center.':state.inspector==='return'?'The view term is a restricted first-order SH slice. α and η are not camera opacity. Noise-like splats remain smooth basis functions.':'Analytic teaching antenna, not the measured Navtech pattern. Widths are one-way FWHM; elevation gain is squared. Azimuth support is fixed at 2Q samples (1.8°).';
 pressed('[data-inspector]',state.inspector,'inspector');
}
function updateStage(){
 const [title,formula,copy]=stageContent[state.stage];
 $('stage-label').textContent=`STAGE ${state.stage+1} / 4`;$('stage-title').textContent=title;$('stage-formula').innerHTML=formula;$('stage-copy').textContent=copy;
 $('radar-title').textContent=state.stage===0?'Projected footprints':'Rendered radar';
 pressed('[data-stage]',state.stage,'stage');
 $('render-mode').disabled=state.stage===0;
 $('show-detections').disabled=state.stage===0;
}
function matrix(el,m){el.style.gridTemplateColumns=`repeat(${m[0].length},1fr)`;el.innerHTML=m.flat().map(v=>`<span>${Math.abs(v)<.0001&&v!==0?v.toExponential(2):fmt(v,4)}</span>`).join('');}
function updateMath(){
 const g=selected(),p=result.projections[state.selected];matrix($('world-matrix'),covariance(g));
 if(!p){$('projection-values').textContent='The spherical chart is singular on the radar vertical axis. Move the splat off-axis.';$('polar-matrix').textContent='Not defined';$('jacobian-matrix').textContent='Not defined';$('return-values').textContent='No finite projection';return;}
 matrix($('polar-matrix'),p.spherical.slice(0,2).map(r=>r.slice(0,2)));matrix($('jacobian-matrix'),p.J);
 $('projection-values').textContent=`G${state.selected+1}: r = ${fmt(p.r)} m · θ = ${fmt(deg(p.theta),1)}° · φ = ${fmt(deg(p.phi),1)}°`;
 $('return-values').innerHTML=`<span>View reflectivity ρ<strong>${fmt(p.rho,3)}</strong></span><span>Total weight σ<strong>${fmt(p.sigma,3)}</strong></span><span>Occupancy α<strong>${fmt(g.alpha)}</strong></span><span>Noise η<strong>${fmt(g.eta)}</strong></span>`;
 const excess=Math.max(0,g.alpha+g.eta-1);
 $('clipping-note').textContent=excess>1e-8?`α + η = ${fmt(g.alpha+g.eta)} > 1. Total σ is clipped, while ρα + ρη is not. The paper’s excess-probability penalty is ${fmt(excess,3)} here.`:`α + η = ${fmt(g.alpha+g.eta)} ≤ 1. Clipping is inactive: target + noise equals total in linear power. The excess-probability penalty is zero.`;
}
function setupCanvas(id){const canvas=$(id),box=canvas.getBoundingClientRect(),dpr=Math.min(window.devicePixelRatio||1,2);const w=Math.max(1,box.width),h=Math.max(1,box.height);if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);}const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);ctx.font='11px system-ui';ctx.lineWidth=1;ctx.textAlign='left';ctx.textBaseline='alphabetic';return {canvas,ctx,w,h};}
function line(ctx,points,color,width=1,dash=[]){if(!points.length)return;ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);points.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.stroke();ctx.setLineDash([]);}
function ellipse(ctx,cov,center,color,isSelected){const A=cov[0][0],B=cov[0][1],D=cov[1][1],v=Math.sqrt((A-D)**2+4*B*B),a=Math.sqrt(Math.max(.01,(A+D+v)/2)),b=Math.sqrt(Math.max(.01,(A+D-v)/2)),angle=.5*Math.atan2(2*B,A-D);for(let n=3;n>=1;n--){ctx.beginPath();ctx.ellipse(center[0],center[1],a*n,b*n,angle,0,2*Math.PI);ctx.globalAlpha=n===3?.08:n===2?.13:.2;ctx.fillStyle=color;ctx.fill();ctx.globalAlpha=isSelected?.85:.6;ctx.strokeStyle=color;ctx.lineWidth=n===2&&isSelected?1.3:.65;ctx.stroke();}ctx.globalAlpha=1;}
function drawScene(){
 const {ctx,w,h}=setupCanvas('scene-canvas'),scale=Math.min(w-44,h-34)/80,cx=w/2,cy=h/2;
 const oblique=state.view==='oblique';
 const M=oblique?[[.80*scale,-.55*scale,0],[-.25*scale,-.37*scale,-1.7*scale]]:[[scale,0,0],[0,-scale,0]];
 const screen=(x,y,z=0)=>[cx+M[0][0]*x+M[0][1]*y+M[0][2]*z,cy+M[1][0]*x+M[1][1]*y+M[1][2]*z];
 sceneProjection={screen,scale,cx,cy};
 ctx.fillStyle='#fcfdf9';ctx.fillRect(0,0,w,h);
 for(let x=-40;x<=40;x+=10){line(ctx,[screen(x,-40),screen(x,40)],C.grid,.7);line(ctx,[screen(-40,x),screen(40,x)],C.grid,.7);}
 for(const r of [10,20,30,40]){line(ctx,Array.from({length:121},(_,i)=>screen(state.sensor.x+r*Math.cos(i*Math.PI/60),state.sensor.y+r*Math.sin(i*Math.PI/60))),r===20?'#cedbcf':'#e1e8df',.7,[2,4]);}
 if(!oblique){ctx.fillStyle=C.muted;ctx.font='9px ui-monospace,monospace';for(const v of [-30,-10,10,30]){ctx.fillText(String(v),screen(v,0)[0]+2,h-5);}ctx.fillText('x (m)',w-31,h-5);ctx.fillText('y (m)',4,11);}
 const theta=-Math.PI+(state.beam+.5)*2*Math.PI/CONFIG.H+rad(state.sensor.yaw),sensor=state.sensor,ss=screen(sensor.x,sensor.y,sensor.z),end=screen(sensor.x+42*Math.cos(theta),sensor.y+42*Math.sin(theta),sensor.z);
 ctx.save();ctx.beginPath();ctx.rect(0,0,w,h);ctx.clip();line(ctx,[ss,end],'#759781',1,[4,4]);ctx.restore();
 state.scene.map((g,i)=>({g,i})).sort((a,b)=>a.g.y-b.g.y).forEach(({g,i})=>{
  const pos=screen(g.x,g.y,g.z),color=g.alpha>=g.eta?C.green:C.orange;
  if(oblique){line(ctx,[screen(g.x,g.y,0),pos],'#b6c6b7',1,[2,3]);}
  ellipse(ctx,multiply(multiply(M,covariance(g)),transpose(M)),pos,color,i===state.selected);
  ctx.beginPath();ctx.arc(...pos,i===state.selected?4:3,0,2*Math.PI);ctx.fillStyle=color;ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.stroke();
  const lx=clamp(pos[0]+9,6,w-35),ly=clamp(pos[1]-10,14,h-12);
  ctx.font=(i===state.selected?'600 ':'')+'10px system-ui';ctx.fillStyle=color;ctx.fillText('G'+(i+1),lx,ly);
 });
 const heading=screen(sensor.x+3*Math.cos(rad(sensor.yaw)),sensor.y+3*Math.sin(rad(sensor.yaw)),sensor.z);
 ctx.beginPath();ctx.arc(...ss,5,0,2*Math.PI);ctx.fillStyle=C.deep;ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.stroke();line(ctx,[ss,heading],C.deep,2);ctx.fillStyle=C.deep;ctx.font='10px system-ui';ctx.fillText('Radar',ss[0]-14,ss[1]+19);
 if(oblique){ctx.font='10px system-ui';ctx.fillStyle=C.muted;ctx.fillText('3D covariance · oblique view',8,h-8);}
 $('scene-help').textContent=oblique?'Height and all three scales affect the 3D ellipsoid. Switch to Top to drag.':'Drag a center to move it. Use the inspector for height and shape.';
}
const stops=[[0,[19,37,30]],[.3,[40,86,62]],[.55,[69,139,91]],[.8,[162,197,115]],[1,[238,242,180]]];
const palette=Array.from({length:256},(_,i)=>{const x=i/255;let j=1;while(j<stops.length-1&&x>stops[j][0])j++;const [lo,a]=stops[j-1],[hi,b]=stops[j],t=(x-lo)/(hi-lo);return a.map((v,k)=>Math.round(v+(b[k]-v)*t));});
const heat=document.createElement('canvas');heat.width=CONFIG.W;heat.height=CONFIG.H;const heatCtx=heat.getContext('2d');
function radarRect(w,h){return {x:43,y:17,w:w-54,h:h-51};}
function drawRadar(){
 const {ctx,w,h}=setupCanvas('radar-canvas'),rect=radarRect(w,h),data=source(),img=heatCtx.createImageData(CONFIG.W,CONFIG.H);
 for(let i=0;i<data.length;i++){const value=clamp((db(data[i])+50)/60,0,1),color=palette[Math.round(value*255)];img.data[i*4]=color[0];img.data[i*4+1]=color[1];img.data[i*4+2]=color[2];img.data[i*4+3]=255;}
 heatCtx.putImageData(img,0,0);ctx.imageSmoothingEnabled=false;ctx.drawImage(heat,rect.x,rect.y,rect.w,rect.h);ctx.imageSmoothingEnabled=true;
 ctx.font='9px ui-monospace,monospace';ctx.fillStyle=C.muted;ctx.textAlign='right';
 for(const a of [-180,-90,0,90,180]){const y=rect.y+(a+180)/360*rect.h;ctx.fillText(`${a}°`,rect.x-7,y+3);}
 ctx.textAlign='center';for(const r of [10,20,30,40,50]){const x=rect.x+(r-CONFIG.rmin)/(CONFIG.rmax-CONFIG.rmin)*rect.w;ctx.fillText(String(r),x,rect.y+rect.h+16);}
 ctx.textAlign='left';ctx.fillText('azimuth',rect.x,9);ctx.textAlign='right';ctx.fillText('range (m)',w-10,h-1);
 const beamY=rect.y+(state.beam+.5)/CONFIG.H*rect.h;
 line(ctx,[[rect.x,beamY],[rect.x+rect.w,beamY]],'#f8e5b3',.9,[4,3]);
 ctx.beginPath();ctx.moveTo(rect.x-4,beamY);ctx.lineTo(rect.x-9,beamY-4);ctx.lineTo(rect.x-9,beamY+4);ctx.fillStyle=C.orange;ctx.fill();
}
function plotBase(id,{xmin,xmax,ymin,ymax,xTicks,yTicks,xLabel,yLabel}){
 const {ctx,w,h}=setupCanvas(id),p={x:45,y:24,w:w-60,h:h-58},X=x=>p.x+(x-xmin)/(xmax-xmin)*p.w,Y=y=>p.y+(ymax-y)/(ymax-ymin)*p.h;
 ctx.font='10px ui-monospace,monospace';ctx.fillStyle=C.muted;
 for(const y of yTicks){line(ctx,[[p.x,Y(y)],[p.x+p.w,Y(y)]],C.grid,.8);ctx.textAlign='right';ctx.fillText(String(y),p.x-9,Y(y)+3);}
 for(const x of xTicks){line(ctx,[[X(x),p.y],[X(x),p.y+p.h]],'#f0f3ed',.6);ctx.textAlign='center';ctx.fillText(String(x),X(x),p.y+p.h+17);}
 ctx.font='10px system-ui';ctx.textAlign='left';ctx.fillText(yLabel,0,10);ctx.textAlign='right';ctx.fillText(xLabel,w-4,h-2);ctx.textAlign='left';return {ctx,p,X,Y,w,h};
}
function drawBeam(){
 if(!result)return;
 const {W,rmin,rmax,H}=CONFIG,dr=(rmax-rmin)/W,start=state.beam*W,profile=source().subarray(start,start+W),raw=result.elevation.subarray(start,start+W);
 const {ctx,p,X,Y}=plotBase('beam-canvas',{xmin:rmin,xmax:rmax,ymin:-50,ymax:10,xTicks:[10,20,30,40,50],yTicks:[-50,-30,-10,10],xLabel:'range (m)',yLabel:state.stage===0?'unit-peak footprint (dB)':'relative power (dB)'});
 ctx.save();ctx.beginPath();ctx.rect(p.x,p.y,p.w,p.h);ctx.clip();
 if(state.stage!==0)line(ctx,Array.from(raw,(v,i)=>[X(rmin+(i+.5)*dr),Y(clamp(db(v),-50,10))]),C.grey,1.4,[4,3]);
 line(ctx,Array.from(profile,(v,i)=>[X(rmin+(i+.5)*dr),Y(clamp(db(v),-50,10))]),C.green,2);
 const detected=peaks(profile,10**(state.threshold/10));
 if(state.detect&&state.stage!==0){line(ctx,[[p.x,Y(state.threshold)],[p.x+p.w,Y(state.threshold)]],C.orange,1,[6,4]);for(const i of detected){const x=X(rmin+(i+.5)*dr),y=Y(clamp(db(profile[i]),-50,10));ctx.beginPath();ctx.arc(x,y,4,0,2*Math.PI);ctx.fillStyle='#fff';ctx.fill();ctx.strokeStyle=C.orange;ctx.lineWidth=1.8;ctx.stroke();}}
 ctx.restore();
 const theta=-180+(state.beam+.5)*360/H;$('beam-value').textContent=`${fmt(theta,2)}°`;$('beam-angle').value=state.beam;
 $('beam-description').textContent=state.stage===0?'Geometry only: sensor weighting is bypassed.':'Elevation samples compared with the selected output, on one fixed power scale.';
 $('detection-controls').hidden=!state.detect||state.stage===0;$('threshold-value').textContent=`${state.threshold} dB`;$('detection-count').textContent=`${detected.length} peak${detected.length===1?'':'s'} in this beam`;
 let idx=0;profile.forEach((v,i)=>{if(v>profile[idx])idx=i;});
 $('beam-readout').textContent=profile[idx]>1e-10?`Beam ${state.beam} / 399 · maximum at ${fmt(rmin+(idx+.5)*dr)} m · ${fmt(db(profile[idx]),2)} dB · ${fmt(profile[idx],5)} relative units`:`Beam ${state.beam} / 399 · no visible return in the plotted range. Select a row crossing a footprint.`;
}
function drawAll(){if(!result)return;drawScene();drawRadar();drawBeam();}
// Real miniature optimization; no prerecorded trajectory or simulated progress.
const fit={p:{...TRAIN_INITIAL},iteration:0,history:[],occ:1,running:false,remaining:0,raf:0};fit.history=[trainLoss(fit.p,fit.occ)];
function drawTraining(){
 const a=trainProfile(fit.p),target=trainProfile(TRAIN_TARGET);
 const {ctx,p,X,Y}=plotBase('train-canvas',{xmin:10,xmax:35,ymin:-60,ymax:0,xTicks:[10,15,20,25,30,35],yTicks:[-60,-40,-20,0],xLabel:'range (m)',yLabel:'relative power (dB)'});
 ctx.save();ctx.beginPath();ctx.rect(p.x,p.y,p.w,p.h);ctx.clip();line(ctx,Array.from(target,(v,i)=>[X(TRAIN_GRID[i]),Y(clamp(db(v),-60,0))]),C.grey,2,[5,4]);line(ctx,Array.from(a,(v,i)=>[X(TRAIN_GRID[i]),Y(clamp(db(v),-60,0))]),C.orange,2);ctx.restore();
 const maxIt=Math.max(10,fit.iteration),ticks=[0,Math.round(maxIt/2),maxIt];
 const plot=plotBase('loss-canvas',{xmin:0,xmax:maxIt,ymin:-6,ymax:1,xTicks:ticks,yTicks:[-6,-3,0],xLabel:'iteration',yLabel:'log₁₀ objective'});
 line(plot.ctx,fit.history.map((v,i)=>[plot.X(i),plot.Y(clamp(Math.log10(Math.max(1e-8,v)),-6,1))]),C.green,1.8);
 if(fit.history.length===1){plot.ctx.beginPath();plot.ctx.arc(plot.X(0),plot.Y(Math.log10(fit.history[0])),2.5,0,2*Math.PI);plot.ctx.fillStyle=C.green;plot.ctx.fill();}
 $('train-iteration').textContent=`ITERATION ${String(fit.iteration).padStart(3,'0')}`;$('train-loss').textContent=fit.history.at(-1).toExponential(3);
 const names=[['r','Center range','m'],['s','Radial scale','m'],['rho','Reflectivity ρ',''],['alpha','Occupancy α',''],['eta','Noise η','']];
 $('train-values').innerHTML=names.map(([key,label,unit])=>`<div class="train-row"><span>${label}</span><output>${fmt(fit.p[key],3)} ${unit}</output></div>`).join('');
 $('train-run').textContent=fit.running?'Pause':'Run 100 steps';$('train-step').disabled=fit.running;
}
function stopFit(){fit.running=false;cancelAnimationFrame(fit.raf);drawTraining();}
function resetFit(){fit.running=false;cancelAnimationFrame(fit.raf);fit.p={...TRAIN_INITIAL};fit.iteration=0;fit.history=[trainLoss(fit.p,fit.occ)];$('train-status').textContent='Ready. Every step computes a numerical gradient and checks the objective.';drawTraining();}
function stepFit(){const update=trainStep(fit.p,fit.occ);fit.p=update.p;fit.iteration++;fit.history.push(update.loss);$('train-status').textContent=`Step ${fit.iteration} · objective ${update.loss.toExponential(4)} · ${update.rate?'accepted gradient update':'no improving step found'} · dL/dr = ${fmt(update.grad[0],5)} · occupancy weight ${fmt(fit.occ,1)}`;}
function animateFit(){if(!fit.running)return;for(let i=0;i<2&&fit.remaining>0;i++,fit.remaining--)stepFit();if(fit.remaining<=0)fit.running=false;drawTraining();if(fit.running)fit.raf=requestAnimationFrame(animateFit);}
// Interaction wiring.
rebuildSelect();rebuildControls();
$('selected-splat').addEventListener('change',e=>{state.selected=Number(e.target.value);rebuildControls();requestRender();});
$('render-mode').addEventListener('change',e=>{state.mode=e.target.value;requestRender();});
$('beam-angle').addEventListener('input',e=>{state.beam=Number(e.target.value);drawScene();drawRadar();drawBeam();});
$('show-detections').addEventListener('change',e=>{state.detect=e.target.checked;drawBeam();});
$('threshold').addEventListener('input',e=>{state.threshold=Number(e.target.value);drawBeam();});
$('reset-scene').addEventListener('click',()=>{Object.assign(state,{scene:cloneScene(),sensor:{...DEFAULT_SENSOR},selected:0,stage:3,mode:'total',beam:218,threshold:-24,detect:false,view:'top',inspector:'shape'});$('render-mode').value='total';$('show-detections').checked=false;$('threshold').value=-24;pressed('[data-view]',state.view,'view');rebuildSelect();rebuildControls();requestRender();});
for(const el of document.querySelectorAll('[data-stage]'))el.addEventListener('click',()=>{state.stage=Number(el.dataset.stage);updateStage();drawRadar();drawBeam();});
for(const el of document.querySelectorAll('[data-view]'))el.addEventListener('click',()=>{state.view=el.dataset.view;pressed('[data-view]',state.view,'view');drawScene();});
for(const el of document.querySelectorAll('[data-inspector]'))el.addEventListener('click',()=>{state.inspector=el.dataset.inspector;rebuildControls();});
$('radar-canvas').addEventListener('pointerdown',e=>{const box=e.currentTarget.getBoundingClientRect(),rect=radarRect(box.width,box.height);state.beam=clamp(Math.floor((e.clientY-box.top-rect.y)/rect.h*CONFIG.H),0,CONFIG.H-1);drawScene();drawRadar();drawBeam();});
$('scene-canvas').addEventListener('pointerdown',e=>{if(state.view!=='top')return;const box=e.currentTarget.getBoundingClientRect(),x=e.clientX-box.left,y=e.clientY-box.top;let distance=24,idx=-1;state.scene.forEach((g,i)=>{const p=sceneProjection.screen(g.x,g.y,g.z),d=Math.hypot(x-p[0],y-p[1]);if(d<distance){distance=d;idx=i;}});if(idx>=0){state.selected=idx;rebuildSelect();rebuildControls();const g=selected();drag={id:e.pointerId,dx:g.x-(x-sceneProjection.cx)/sceneProjection.scale,dy:g.y+(y-sceneProjection.cy)/sceneProjection.scale};e.currentTarget.setPointerCapture(e.pointerId);requestRender();}});
$('scene-canvas').addEventListener('pointermove',e=>{if(!drag||drag.id!==e.pointerId)return;const box=e.currentTarget.getBoundingClientRect(),x=e.clientX-box.left,y=e.clientY-box.top,g=selected();g.x=clamp((x-sceneProjection.cx)/sceneProjection.scale+drag.dx,-38,38);g.y=clamp(-(y-sceneProjection.cy)/sceneProjection.scale+drag.dy,-38,38);if(state.inspector==='shape'){for(const k of ['x','y']){$(`param-${k}`).value=g[k];$(`value-${k}`).textContent=`${fmt(g[k],1)} m`;}}requestRender();});
for(const name of ['pointerup','pointercancel','lostpointercapture'])$('scene-canvas').addEventListener(name,()=>{drag=null;});
$('train-run').addEventListener('click',()=>{if(fit.running){stopFit();return;}fit.running=true;fit.remaining=100;drawTraining();fit.raf=requestAnimationFrame(animateFit);});
$('train-step').addEventListener('click',()=>{stepFit();drawTraining();});$('train-reset').addEventListener('click',resetFit);
$('occ-weight').addEventListener('input',e=>{fit.occ=Number(e.target.value);$('occ-value').textContent=`${fmt(fit.occ,1)}×`;resetFit();});
document.addEventListener('visibilitychange',()=>{if(document.hidden&&fit.running)stopFit();});
const resize=new ResizeObserver(()=>{if(result)drawAll();drawTraining();});for(const id of ['scene-canvas','radar-canvas','beam-canvas','train-canvas','loss-canvas'])resize.observe($(id));
requestRender();drawTraining();
