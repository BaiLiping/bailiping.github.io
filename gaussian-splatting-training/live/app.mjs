import {
  createTraining, evaluateTraining, stepTraining, inspectPixel,
  createMultiView, evaluateMultiView, stepMultiView,
} from './model.mjs';

const $ = id => document.getElementById(id);
const C = {ink:'#182d33', muted:'#60747a', teal:'#16736e', rust:'#b44f37', line:'#d6dedc', gray:'#98a8a6', paper:'#fffdfa'};
const query = new URLSearchParams(location.search);
const embedded = query.get('embed') === '1';
const number = (value, digits=4) => Number.isFinite(value) ? value.toFixed(digits) : '—';
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const allGroups = ['position','shape','opacity','color'];

let lab = query.get('lab') === 'multiview' ? 'multiview' : 'training';
let training = createTraining({size:32, lambda:.2});
let trainingResult = evaluateTraining(training);
let multi = createMultiView({size:32, views:1});
let multiResult = evaluateMultiView(multi);
let objectiveAnchor = {iteration:0, loss:trainingResult.loss};
let viewAnchor = {iteration:0, trainingLoss:multiResult.trainingLoss, heldOutLoss:multiResult.heldOutLoss};
let objectiveLabel = 'actual optimizer steps';
let viewHistoryLabel = 'Across optimizer steps';
let geometryTrace = [multiResult.center.slice()];
let selectedPixel = {x:17, y:17};
let running = false, timer = null, remaining = 0;

function trainingGroups() {
  return Object.fromEntries(allGroups.map(group => [group, $('train-'+group).checked]));
}

function canTrain() {
  return lab !== 'training' || allGroups.some(group => $('train-'+group).checked);
}

function historySince(history, anchor, fields) {
  return [anchor, ...history.filter(row => row.iteration > anchor.iteration && fields.every(field => Number.isFinite(row[field])))];
}

function heatColor(value) {
  const stops = [
    {at:0, c:[7,21,27]}, {at:.34, c:[22,115,110]},
    {at:.70, c:[231,175,101]}, {at:1, c:[255,236,200]},
  ];
  const t = clamp(value / .5, 0, 1);
  let index = 0;
  while (index < stops.length-2 && t > stops[index+1].at) index++;
  const a = stops[index], b = stops[index+1], u = (t-a.at)/(b.at-a.at);
  return a.c.map((v, i) => Math.round(v+(b.c[i]-v)*u));
}

const buffer = document.createElement('canvas');
function paintImage(canvas, data, size, {residual=false, splats=null, pixel=null}={}) {
  buffer.width = size;
  buffer.height = size;
  const source = buffer.getContext('2d');
  const image = source.createImageData(size, size);
  for (let p=0; p<size*size; p++) {
    const colors = residual ? heatColor((data[p*3]+data[p*3+1]+data[p*3+2])/3) : [0,1,2].map(c => Math.round(clamp(data[p*3+c],0,1)*255));
    for (let c=0;c<3;c++) image.data[p*4+c]=colors[c];
    image.data[p*4+3]=255;
  }
  source.putImageData(image,0,0);
  const ctx=canvas.getContext('2d');
  const w=canvas.width,h=canvas.height;
  ctx.clearRect(0,0,w,h);
  ctx.imageSmoothingEnabled=true;
  ctx.drawImage(buffer,0,0,w,h);
  if(splats) {
    ctx.save();
    ctx.lineWidth=1.35;
    ctx.font='11px Arial';
    ctx.textAlign='center';
    ctx.textBaseline='middle';
    for(let i=0;i<splats.length;i++) {
      const s=splats[i];
      ctx.strokeStyle='rgba(255,253,250,.85)';
      ctx.beginPath();
      ctx.ellipse(s.x*w,s.y*h,s.sx*w,s.sy*h,s.angle,0,2*Math.PI);
      ctx.stroke();
      ctx.fillStyle='rgba(10,30,34,.9)';
      ctx.beginPath();ctx.arc(s.x*w,s.y*h,8,0,2*Math.PI);ctx.fill();
      ctx.fillStyle='#fffdfa';ctx.fillText(String(i+1),s.x*w,s.y*h+.2);
    }
    ctx.restore();
  }
  if(pixel) {
    const x=(pixel.x+.5)/size*w, y=(pixel.y+.5)/size*h;
    ctx.save();
    ctx.strokeStyle='#fffdfa';ctx.lineWidth=1;
    ctx.strokeRect(pixel.x/size*w,pixel.y/size*h,w/size,h/size);
    ctx.beginPath();ctx.moveTo(x-9,y);ctx.lineTo(x-5,y);ctx.moveTo(x+5,y);ctx.lineTo(x+9,y);ctx.moveTo(x,y-9);ctx.lineTo(x,y-5);ctx.moveTo(x,y+5);ctx.lineTo(x,y+9);ctx.stroke();
    ctx.restore();
  }
}

function compact(value) {
  if(value === 0) return '0';
  if(value < .001) return value.toExponential(0);
  return value.toFixed(value < .01 ? 3 : 2).replace(/0+$/,'').replace(/\.$/,'');
}

function chart(id, rows, fields) {
  // Keep long running sessions responsive while preserving actual sampled steps.
  if(rows.length>500){const stride=Math.ceil(rows.length/500);rows=rows.filter((_,i)=>i%stride===0||i===rows.length-1);}
  const svg=$(id);
  const [, , width, height]=svg.getAttribute('viewBox').split(/\s+/).map(Number);
  const margin={left:38,right:8,top:7,bottom:18};
  const plotW=width-margin.left-margin.right, plotH=height-margin.top-margin.bottom;
  const max=Math.max(1e-6,...rows.flatMap(row=>fields.map(f=>row[f.key]||0)))*1.08;
  const first=rows[0]?.iteration||0;
  const last=Math.max(first+1, rows.at(-1)?.iteration||0);
  const X=x=>margin.left+(x-first)/(last-first)*plotW;
  const Y=y=>margin.top+(1-y/max)*plotH;
  let html='';
  for(const fraction of [0,.5,1]) {
    const y=max*fraction;
    html+=`<line x1="${margin.left}" y1="${Y(y)}" x2="${width-margin.right}" y2="${Y(y)}" stroke="${C.line}" stroke-width=".7"/><text x="${margin.left-5}" y="${Y(y)+3}" text-anchor="end" font-size="9">${compact(y)}</text>`;
  }
  html+=`<text x="${margin.left}" y="${height-3}" font-size="9">${first}</text><text x="${width-margin.right}" y="${height-3}" text-anchor="end" font-size="9">${rows.at(-1)?.iteration||0} steps</text>`;
  for(const field of fields) {
    const points=rows.map(row=>`${X(row.iteration).toFixed(2)},${Y(row[field.key]).toFixed(2)}`).join(' ');
    html+=`<polyline points="${points}" fill="none" stroke="${field.color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
    if(rows.length===1)html+=`<circle cx="${X(rows[0].iteration)}" cy="${Y(rows[0][field.key])}" r="2.5" fill="${field.color}"/>`;
  }
  svg.innerHTML=html;
}

function renderPixel() {
  if(embedded || lab!=='training')return;
  const p=inspectPixel(training,selectedPixel.x,selectedPixel.y);
  // Keep an in-progress numeric edit intact while other parts of the lab render.
  // Both typing and native steppers update the selected pixel on input below.
  if(document.activeElement!==$('pixel-x'))$('pixel-x').value=p.x;
  if(document.activeElement!==$('pixel-y'))$('pixel-y').value=p.y;
  const rgb=a=>a.map(v=>number(v,3)).join(', ');
  const swatch=a=>`<i class="rgb-swatch" style="background:rgb(${a.map(v=>Math.round(clamp(v,0,1)*255)).join(',')})" aria-hidden="true"></i>`;
  $('pixel-rows').innerHTML=p.contributions.map(c=>`<tr><th scope="row">Splat ${c.index+1}</th><td>${number(c.opacity,3)}</td><td>${number(c.gaussian,3)}</td><td>${number(c.alpha,3)}</td><td>${number(c.transmittance,3)}</td><td>${number(c.weight,3)}</td><td>${swatch(c.contribution)}${rgb(c.contribution)}</td></tr>`).join('')+`<tr class="background-row"><th scope="row">Background</th><td>—</td><td>—</td><td>—</td><td>${number(p.remainingTransmittance,3)}</td><td>${number(p.remainingTransmittance,3)}</td><td>${swatch(p.background)}${rgb(p.background)}</td></tr>`;
  $('pixel-rendered').textContent=rgb(p.prediction);
  $('pixel-target').textContent=rgb(p.target);
  const offset=3*(p.y*training.size+p.x);
  $('pixel-gradient').textContent=Array.from(trainingResult.pixelGradient.slice(offset,offset+3)).map(v=>(v>=0?'+':'')+v.toExponential(2)).join(', ');
}

function renderTraining() {
  const r=trainingResult, lambda=training.lambda;
  $('iteration-label').textContent='Iteration '+r.iteration;
  $('result-title').textContent='Five splats, one target image';
  $('lambda-value').value=number(lambda,2);
  $('l1-value').textContent=number(r.l1);
  $('dssim-value').textContent=number(r.dssim);
  $('loss-value').textContent=number(r.loss);
  const improvement=objectiveAnchor.loss>0?(1-r.loss/objectiveAnchor.loss)*100:0;
  $('loss-change').textContent=r.iteration===objectiveAnchor.iteration?'Initial state':(improvement>=0?'−':'+')+number(Math.abs(improvement),1)+'%';
  $('loss-change').title='Change relative to the start of the current objective.';
  $('splat-count').textContent=$('footprints').checked?'1σ outlines · fixed layer order':r.splats.length+' trainable 2D splats';
  const marker=embedded?null:selectedPixel;
  paintImage($('target-image'),r.target,r.size,{pixel:marker});
  paintImage($('render-image'),r.image,r.size,{splats:$('footprints').checked?r.splats:null,pixel:marker});
  paintImage($('residual-image'),r.residual,r.size,{residual:true,pixel:marker});
  $('loss-equation').textContent=`${number(1-lambda,2)} × ${number(r.l1,4)} + ${number(lambda,2)} × ${number(r.dssim,4)} = ${number(r.loss,4)}`;
  const enabled=allGroups.filter(group=>$('train-'+group).checked);
  const frozen=allGroups.filter(group=>!$('train-'+group).checked);
  if(enabled.length===0) $('training-caption').textContent='All parameter groups are frozen. Enable a group to train.';
  else if(frozen.length) $('training-caption').textContent='Frozen: '+frozen.join(', ')+'. Other groups still receive image gradients.';
  else $('training-caption').textContent=r.iteration===0?'The target supplies pixels. Adam changes the splat parameters.':'A smaller image loss can still admit different splat parameters.';
  const rows=historySince(r.history,objectiveAnchor,['loss']);
  chart('loss-chart',rows,[{key:'loss',color:C.teal}]);
  $('history-caption').textContent=objectiveLabel;
  renderPixel();
  updateButtons();
}

function geometry() {
  const r=multiResult, [x,,z]=r.center, [rx,,rz]=r.reference;
  const svg=$('geometry-chart'), show=$('show-reference').checked;
  const left=45, top=17, width=238, height=174;
  const X=v=>left+v*width, Z=v=>top+(1-v)*height;
  let html=`<rect x="${left}" y="${top}" width="${width}" height="${height}" rx="3" fill="#f6f8f5" stroke="${C.line}"/>`;
  for(const t of [0,.25,.5,.75,1]) {
    html+=`<line x1="${X(t)}" y1="${top}" x2="${X(t)}" y2="${top+height}" stroke="${C.line}" stroke-width=".7"/><line x1="${left}" y1="${Z(t)}" x2="${left+width}" y2="${Z(t)}" stroke="${C.line}" stroke-width=".7"/>`;
    if(t===0||t===.5||t===1)html+=`<text x="${X(t)}" y="${top+height+14}" text-anchor="middle" font-size="10">${t}</text><text x="${left-8}" y="${Z(t)+3}" text-anchor="end" font-size="10">${t}</text>`;
  }
  // These lines illustrate the constraints inferred from the target images.
  // They are explanatory geometry only and do not enter any parameter update.
  html+=`<line x1="${X(rx)}" y1="${top}" x2="${X(rx)}" y2="${top+height}" stroke="${C.gray}" stroke-width="2" stroke-dasharray="5 4"/><text x="${X(rx)+5}" y="${top+11}" font-size="9">camera 1</text>`;
  if(r.views===2)html+=`<line x1="${left}" y1="${Z(rz)}" x2="${left+width}" y2="${Z(rz)}" stroke="${C.gray}" stroke-width="2" stroke-dasharray="5 4"/><text x="${left+5}" y="${Z(rz)-5}" font-size="9">camera 2</text>`;
  if(geometryTrace.length>1)html+=`<polyline points="${geometryTrace.map(c=>X(c[0]).toFixed(2)+','+Z(c[2]).toFixed(2)).join(' ')}" fill="none" stroke="${C.teal}" stroke-opacity=".35" stroke-width="1.8"/>`;
  if(show)html+=`<circle cx="${X(rx)}" cy="${Z(rz)}" r="8" fill="none" stroke="${C.rust}" stroke-width="1.8" stroke-dasharray="3 2"/>`;
  html+=`<circle cx="${X(x)}" cy="${Z(z)}" r="5.5" fill="${C.teal}" stroke="${C.paper}" stroke-width="1.5"/><text x="${left+width}" y="${top+height+30}" text-anchor="end" font-size="10">x position</text><text x="12" y="${top+height/2}" transform="rotate(-90 12 ${top+height/2})" text-anchor="middle" font-size="10">z depth</text>`;
  svg.innerHTML=html;
  $('reference-legend').hidden=!show;
  $('geometry-note').textContent=r.views===1?`Camera 1 cannot change z = ${number(z,3)}. All depths on its constraint line share the same projection.`:`Both image constraints now act on the same center. Estimated depth z = ${number(z,3)}.`;
}

function initializeCameraRows() {
  $('camera-rows').innerHTML=[['Camera 1','x / y'],['Camera 2','z / y'],['Held-out','45°']].map(([label,axes],i)=>`<div class="camera-row" id="camera-row-${i}"><div class="camera-label">${label}<small>${axes}</small><em id="camera-use-${i}">Evaluation</em></div><canvas id="camera-target-${i}" width="96" height="96" role="img" aria-label="${esc(label)} target image"></canvas><canvas id="camera-render-${i}" width="96" height="96" role="img" aria-label="${esc(label)} current rendered image"></canvas><span class="camera-loss" id="camera-loss-${i}">—</span></div>`).join('');
}

function renderMultiView() {
  const r=multiResult;
  $('iteration-label').textContent='Iteration '+r.iteration;
  $('result-title').textContent='The same 3D center must explain every training view';
  $('view-loss').textContent=number(r.trainingLoss);
  $('heldout-loss').textContent=number(r.heldOutLoss);
  $('training-view-count').textContent=r.views===1?'camera 1':'mean over 2 cameras';
  $('center-value').textContent=r.center.map(v=>number(v,3)).join(', ');
  $('camera-hint').textContent=r.views===1?'Camera 1 sees x and y. Its pixels give no gradient for z.':'Camera 2 sees z and y. Its error now supplies a depth gradient.';
  for(const [i,c] of r.cameras.entries()) {
    const row=$('camera-row-'+i);
    row.classList.toggle('held-out',i===2);
    row.classList.toggle('inactive',i===1&&!c.training);
    $('camera-use-'+i).textContent=c.training?'Training':i===2?'Never trained':'Evaluation';
    $('camera-loss-'+i).textContent=number(c.loss,4);
    paintImage($('camera-target-'+i),c.target,r.size);
    paintImage($('camera-render-'+i),c.image,r.size);
  }
  geometry();
  chart('view-chart',historySince(r.history,viewAnchor,['trainingLoss','heldOutLoss']),[
    {key:'trainingLoss',color:C.teal},{key:'heldOutLoss',color:C.rust},
  ]);
  $('view-history-label').textContent=viewHistoryLabel;
  updateButtons();
}

function updateButtons() {
  for(const prefix of ['train','view']) {
    const active=(prefix==='train'?lab==='training':lab==='multiview')&&running;
    $(prefix+'-play').textContent=active?'Pause':'Play';
    $(prefix+'-play').setAttribute('aria-pressed',String(active));
    $(prefix+'-batch').disabled=active;
  }
  const frozen=allGroups.every(group=>!$('train-'+group).checked);
  $('train-play').disabled=frozen;
  $('train-step').disabled=frozen;
  $('train-batch').disabled=frozen||(lab==='training'&&running);
}

function announce(message) {$('status').textContent=message;}

function stop() {
  if(timer!==null)clearTimeout(timer);
  timer=null;running=false;remaining=0;
  updateButtons();
}

function advance(count) {
  if(lab==='training') {
    trainingResult=stepTraining(training,{steps:count,lambda:Number($('lambda').value),train:trainingGroups()});
    renderTraining();
  } else {
    multiResult=stepMultiView(multi,{steps:count,views:multi.views});
    geometryTrace.push(multiResult.center.slice());
    if(geometryTrace.length>700)geometryTrace=[geometryTrace[0],...geometryTrace.slice(-599)];
    renderMultiView();
  }
}

function tick() {
  if(!running)return;
  if(document.hidden || document.documentElement.dataset.bentoPaused==='true' || !canTrain()) {stop();return;}
  const count=Number.isFinite(remaining)?Math.min(5,remaining):2;
  advance(count);
  remaining-=count;
  if(remaining<=0) {
    stop();
    announce(lab==='training'?`Iteration ${training.iteration}. Combined image loss ${number(trainingResult.loss)}.`:`Iteration ${multi.iteration}. Training loss ${number(multiResult.trainingLoss)}. Held-out loss ${number(multiResult.heldOutLoss)}.`);
    return;
  }
  // A requested Play action is the only source of continuous motion.
  timer=setTimeout(tick,Number.isFinite(remaining)?20:70);
}

function start(count=Infinity) {
  if(!canTrain())return;
  stop();running=true;remaining=count;updateButtons();
  timer=setTimeout(tick,0);
}

function singleStep() {
  stop();
  if(!canTrain())return;
  advance(1);
  announce(lab==='training'?`One optimizer step. Loss ${number(trainingResult.loss)}.`:`One optimizer step. Training loss ${number(multiResult.trainingLoss)}.`);
}

function resetTraining() {
  stop();
  training=createTraining({size:32,lambda:Number($('lambda').value)});
  trainingResult=evaluateTraining(training);
  objectiveAnchor={iteration:0,loss:trainingResult.loss};
  objectiveLabel='actual optimizer steps';
  renderTraining();
  announce('The five splats have returned to the same initial parameters. Parameter freezes and loss weight are preserved.');
}

function resetMultiView() {
  stop();
  multi=createMultiView({size:32,views:multi.views});
  multiResult=evaluateMultiView(multi);
  geometryTrace=[multiResult.center.slice()];
  viewAnchor={iteration:0,trainingLoss:multiResult.trainingLoss,heldOutLoss:multiResult.heldOutLoss};
  viewHistoryLabel='Across optimizer steps';
  renderMultiView();
  announce('The 3D center has reset. The selected training cameras are preserved.');
}

function changeLambda() {
  stop();
  training.lambda=Number($('lambda').value);
  trainingResult=evaluateTraining(training);
  objectiveAnchor={iteration:training.iteration,loss:trainingResult.loss};
  objectiveLabel='since λ changed';
  renderTraining();
}

function setLab(name) {
  stop();
  lab=name==='multiview'?'multiview':'training';
  for(const button of document.querySelectorAll('[data-lab]'))button.setAttribute('aria-pressed',String(button.dataset.lab===lab));
  for(const candidate of ['training','multiview']) {
    $(candidate+'-controls').hidden=candidate!==lab;
    $(candidate+'-results').hidden=candidate!==lab;
    $(candidate+'-explanation').hidden=candidate!==lab;
  }
  $('pixel-inspector').hidden=lab!=='training';
  $('scope').textContent=lab==='training'?'Computed 2D teaching model: anisotropic footprints, fixed depth order, constant RGB colors. This is not the full 3DGS rasterizer.':'Computed 3D-center example: known orthographic cameras and fixed appearance. Only training images enter the update; the 45° image evaluates it.';
  $('expand').href='./?lab='+lab;
  const url=new URL(location.href);url.searchParams.set('lab',lab);
  history.replaceState(null,'',url.pathname+url.search+url.hash);
  if(lab==='training')renderTraining();else renderMultiView();
}

for(const button of document.querySelectorAll('[data-lab]'))button.addEventListener('click',()=>setLab(button.dataset.lab));
for(const prefix of ['train','view']) {
  $(prefix+'-play').addEventListener('click',()=>running?stop():start());
  $(prefix+'-step').addEventListener('click',singleStep);
  $(prefix+'-batch').addEventListener('click',()=>start(50));
}
$('train-reset').addEventListener('click',resetTraining);
$('view-reset').addEventListener('click',resetMultiView);
$('lambda').addEventListener('input',changeLambda);
$('paper-weight').addEventListener('click',()=>{$('lambda').value=.2;changeLambda();});
$('footprints').addEventListener('change',renderTraining);
for(const group of allGroups)$('train-'+group).addEventListener('change',()=>{
  stop();objectiveAnchor={iteration:training.iteration,loss:trainingResult.loss};
  objectiveLabel='since freeze changed';renderTraining();
});
for(const radio of document.querySelectorAll('input[name="views"]'))radio.addEventListener('change',()=>{
  if(!radio.checked)return;
  stop();multi.views=Number(radio.value);multiResult=evaluateMultiView(multi);
  viewAnchor={iteration:multi.iteration,trainingLoss:multiResult.trainingLoss,heldOutLoss:multiResult.heldOutLoss};
  viewHistoryLabel='Since camera change';
  renderMultiView();
  announce(multi.views===2?'The second camera now contributes to training. The current center is preserved.':'Only camera 1 now contributes to training. Depth is unobserved.');
});
$('show-reference').addEventListener('change',geometry);

function selectPixel(x,y) {
  selectedPixel={x:clamp(Math.round(x)||0,0,training.size-1),y:clamp(Math.round(y)||0,0,training.size-1)};
  renderTraining();
}
for(const axis of ['x','y']) {
  const input=$('pixel-'+axis);
  const update=()=>{
    if(input.value.trim()==='' || !Number.isFinite(input.valueAsNumber))return;
    selectPixel(axis==='x'?input.valueAsNumber:selectedPixel.x,axis==='y'?input.valueAsNumber:selectedPixel.y);
  };
  input.addEventListener('input',update);
  input.addEventListener('change',()=>{update();input.value=selectedPixel[axis];});
}
if(!embedded)for(const id of ['target-image','render-image','residual-image']) {
  const canvas=$(id);
  canvas.classList.add('inspectable');
  canvas.addEventListener('click',event=>{
    const bounds=canvas.getBoundingClientRect();
    selectPixel(Math.floor((event.clientX-bounds.left)/bounds.width*training.size),Math.floor((event.clientY-bounds.top)/bounds.height*training.size));
  });
}

window.addEventListener('bento-live-visibility',event=>{if(event.detail?.paused)stop();});
document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
window.addEventListener('pagehide',stop);
window.addEventListener('error',()=>{stop();announce('An unexpected error stopped training. Reload the lab to restart.');});

initializeCameraRows();
setLab(lab);
