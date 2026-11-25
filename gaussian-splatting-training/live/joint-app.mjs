import {createJoint, evaluateJoint, stepJoint} from './joint-model.mjs';

const $ = id => document.getElementById(id);
const embedded = new URLSearchParams(location.search).get('embed') === '1';
const C = {ink:'#182d33',muted:'#60747a',teal:'#16736e',rust:'#b44f37',line:'#d6dedc',paper:'#fffdfa'};
const clamp = (v,lo,hi) => Math.max(lo,Math.min(hi,v));
const number = (v,n=4) => Number.isFinite(v) ? v.toFixed(n) : '—';
const norm = values => Math.sqrt(values.reduce((sum,v)=>sum+v*v,0));
const buffer = document.createElement('canvas');
let state = createJoint({size:24});
let result = evaluateJoint(state);
let selected = {view:1,x:13,y:12};
let running = false, timer = null, remaining = 0;

function heatColor(value) {
  const stops = [[0,[7,21,27]],[.34,[22,115,110]],[.70,[231,175,101]],[1,[255,236,200]]];
  const t=clamp(value/.5,0,1);
  let i=0;
  while(i<2&&t>stops[i+1][0])i++;
  const [a,ac]=stops[i], [b,bc]=stops[i+1], u=(t-a)/(b-a);
  return ac.map((v,j)=>Math.round(v+u*(bc[j]-v)));
}

function image(canvas,data,{residual=false,marker=false}={}) {
  const size=result.size;
  buffer.width=size; buffer.height=size;
  const source=buffer.getContext('2d'), pixels=source.createImageData(size,size);
  for(let p=0;p<size*size;p++) {
    const rgb=residual?heatColor((data[3*p]+data[3*p+1]+data[3*p+2])/3):[0,1,2].map(c=>Math.round(255*clamp(data[3*p+c],0,1)));
    for(let c=0;c<3;c++)pixels.data[4*p+c]=rgb[c];
    pixels.data[4*p+3]=255;
  }
  source.putImageData(pixels,0,0);
  const ctx=canvas.getContext('2d'), w=canvas.width,h=canvas.height;
  ctx.clearRect(0,0,w,h); ctx.imageSmoothingEnabled=true;
  ctx.drawImage(buffer,0,0,w,h);
  if(marker&&!embedded) {
    ctx.strokeStyle=C.paper;ctx.lineWidth=1.4;
    ctx.strokeRect(selected.x/size*w,selected.y/size*h,w/size,h/size);
  }
}

function historyChart() {
  let rows=result.history;
  if(rows.length>400) {const stride=Math.ceil(rows.length/400);rows=rows.filter((_,i)=>i%stride===0||i===rows.length-1);}
  const max=Math.max(1e-8,...rows.map(r=>r.loss))*1.08;
  const start=rows[0]?.iteration||0, last=rows.at(-1)?.iteration||0;
  const X=x=>42+308*(x-start)/Math.max(1,last-start),Y=y=>8+62*(1-y/max);
  const compact=v=>v<.001?v.toExponential(0):v.toFixed(3);
  let svg='';
  for(const f of [0,.5,1]) {
    const v=max*f;
    svg+=`<line x1="42" x2="350" y1="${Y(v)}" y2="${Y(v)}" stroke="${C.line}" stroke-width=".8"/><text x="37" y="${Y(v)+3}" text-anchor="end" font-size="9">${compact(v)}</text>`;
  }
  svg+=`<text x="42" y="86" font-size="9">${start}</text><text x="350" y="86" text-anchor="end" font-size="9">${last} updates</text>`;
  svg+=`<polyline points="${rows.map(r=>`${X(r.iteration)},${Y(r.loss)}`).join(' ')}" fill="none" stroke="${C.teal}" stroke-width="2" stroke-linejoin="round"/>`;
  if(rows.length===1)svg+=`<circle cx="${X(start)}" cy="${Y(rows[0].loss)}" r="2.5" fill="${C.teal}"/>`;
  $('loss-chart').innerHTML=svg;
}

function mapChart() {
  if(embedded)return;
  const left=40,top=24,side=220,X=x=>left+x*side,Y=y=>top+y*side;
  let svg=`<rect x="${left}" y="${top}" width="${side}" height="${side}" fill="#f6f8f5" stroke="${C.line}"/>`;
  for(const v of [0,.25,.5,.75,1]) {
    svg+=`<line x1="${X(v)}" x2="${X(v)}" y1="${top}" y2="${top+side}" stroke="${C.line}" stroke-width=".6"/><line x1="${left}" x2="${left+side}" y1="${Y(v)}" y2="${Y(v)}" stroke="${C.line}" stroke-width=".6"/>`;
    if(v===0||v===.5||v===1)svg+=`<text x="${X(v)}" y="${top+side+17}" font-size="10" text-anchor="middle">${v}</text><text x="${left-8}" y="${Y(v)+3}" font-size="10" text-anchor="end">${v}</text>`;
  }
  for(const [i,g] of result.splats.entries()) {
    const color=`rgb(${g.color.map(c=>Math.round(c*255)).join(',')})`;
    svg+=`<ellipse cx="${X(g.x)}" cy="${Y(g.y)}" rx="${g.sx*side}" ry="${g.sy*side}" transform="rotate(${g.angle*180/Math.PI} ${X(g.x)} ${Y(g.y)})" fill="${color}" fill-opacity=".28" stroke="${color}" stroke-width="1.7"/><circle cx="${X(g.x)}" cy="${Y(g.y)}" r="3" fill="${C.ink}"/><text x="${X(g.x)+6}" y="${Y(g.y)-6}" font-size="11">${i+1}</text>`;
  }
  const poses=[[0,0,0],result.pose];
  for(const [i,[tx,ty,a]] of poses.entries()) {
    const c=Math.cos(a),s=Math.sin(a),x=.5-c*tx-s*ty,y=.5+s*tx-c*ty;
    const color=i===0?C.muted:C.teal;
    const ax=.12*c,ay=-.12*s,bx=.12*s,by=.12*c;
    svg+=`<circle cx="${X(x)}" cy="${Y(y)}" r="4" fill="${C.paper}" stroke="${color}" stroke-width="2"/><path d="M${X(x)},${Y(y)} L${X(x+ax)},${Y(y+ay)} M${X(x)},${Y(y)} L${X(x+bx)},${Y(y+by)}" fill="none" stroke="${color}" stroke-width="2" ${i===0?'stroke-dasharray="3 3"':''}/><text x="${X(x+ax)+3}" y="${Y(y+ay)-3}" font-size="9">x${i}</text><text x="${X(x+bx)+3}" y="${Y(y+by)+5}" font-size="9">y${i}</text>`;
  }
  svg+=`<text x="292" y="48" font-size="12" font-weight="600">Current estimates</text><text x="292" y="74" font-size="11">1–4 · shared splats</text><line x1="292" x2="310" y1="100" y2="100" stroke="${C.muted}" stroke-width="2" stroke-dasharray="3 3"/><text x="319" y="104" font-size="11">Camera 0 · fixed</text><line x1="292" x2="310" y1="127" y2="127" stroke="${C.teal}" stroke-width="2"/><text x="319" y="131" font-size="11">Camera 1 · estimated</text><text x="292" y="170" font-size="10">Axes show image-right</text><text x="292" y="185" font-size="10">and image-down directions</text><text x="292" y="200" font-size="10">in the world frame.</text><text x="${left+side}" y="277" text-anchor="end" font-size="10">world x → · world y ↓</text>`;
  $('map-chart').innerHTML=svg;
  $('scene-gradient').textContent=norm(result.gradients.centers).toExponential(3);
  $('pose-gradient').textContent=norm(result.gradients.pose).toExponential(3);
}

function pixelInspector() {
  if(embedded)return;
  const r=result.cameras[selected.view],offset=3*(selected.y*result.size+selected.x);
  const rgb=data=>Array.from(data.slice(offset,offset+3)).map(v=>number(v,5)).join(', ');
  $('pixel-target').textContent=rgb(r.target);
  $('pixel-rendered').textContent=rgb(r.image);
  $('pixel-absolute').textContent=rgb(r.residual);
  $('pixel-l1').textContent=number((r.residual[offset]+r.residual[offset+1]+r.residual[offset+2])/3,6);
  for(const axis of ['x','y'])if(document.activeElement!==$('pixel-'+axis))$('pixel-'+axis).value=selected[axis];
  $('pixel-view').value=selected.view;
}

function buttons() {
  $('play').textContent=running?'Pause':'Run';
  $('play').setAttribute('aria-pressed',String(running));
  $('batch').disabled=running;
  $('step').disabled=running;
}

function render() {
  const r=result,lambda=r.lambda;
  $('iteration').textContent='Iteration '+r.iteration;
  $('lambda-value').value=number(lambda,2);
  $('l1-value').textContent=number(r.l1,5);
  $('ssim-value').textContent=number(r.ssim,5);
  $('loss-value').textContent=number(r.loss,5);
  const first=r.history[0]?.loss||r.loss;
  $('loss-change').textContent=r.history.length<=1?'Initial state':`−${number(Math.max(0,(1-r.loss/first)*100),1)}%`;
  $('loss-equation').textContent=`${number(1-lambda,2)} × ${number(r.l1,5)} + ${number(lambda,2)} × ${number(1-r.ssim,5)} = ${number(r.loss,5)}`;
  $('pose-translation').textContent=r.pose.slice(0,2).map(v=>number(v,4)).join(', ');
  $('pose-angle').textContent=number(r.pose[2]*180/Math.PI,2)+'°';
  $('schedule-note').textContent=r.mode==='joint'?(r.schedule==='simultaneous'?'Both gradient blocks use the same current estimates.':'Pose updates first; then splats use a recomputed residual.'):(r.mode==='pose'?'Only camera 1 moves. The shared map stays frozen.':'Only shared splat centers move. Both camera poses stay frozen.');
  for(const [i,camera] of r.cameras.entries()) {
    image($('target-'+i),camera.target,{marker:selected.view===i});
    image($('render-'+i),camera.image,{marker:selected.view===i});
    image($('residual-'+i),camera.residual,{residual:true,marker:selected.view===i});
    $('view-loss-'+i).textContent='Loss '+number(camera.loss,4);
  }
  historyChart();mapChart();pixelInspector();buttons();
}

function stop(message) {
  running=false;
  if(timer!==null)clearTimeout(timer);
  timer=null;buttons();
  if(message)$('status').textContent=message;
}

function advance() {
  result=stepJoint(state);
  render();
  if(!result.accepted) {
    stop('No improving step found at the current precision. Change the objective, unfreeze a block, or reset to continue.');
    $('history-note').textContent='No improving step found at current precision. You can reset or change controls.';
    return false;
  }
  return true;
}

function tick() {
  if(!running)return;
  if(!advance())return;
  remaining--;
  if(remaining<=0) {stop(`Run finished at iteration ${result.iteration}. Loss ${number(result.loss,5)}.`);return;}
  timer=setTimeout(tick,20);
}

function start(count=150) {
  stop();running=true;remaining=count;
  $('history-note').textContent='Measured over both images. No true pose enters the loss.';
  buttons();timer=setTimeout(tick,0);
}

function optionsChanged() {
  stop();
  result=stepJoint(state,{steps:0,mode:$('mode').value,schedule:$('schedule').value,lambda:Number($('lambda').value)});
  $('history-note').textContent='New history segment; current estimates and observed images are retained.';
  render();
}

function selectPixel(view,x,y) {
  selected={view:clamp(Math.round(view),0,1),x:clamp(Math.round(x)||0,0,result.size-1),y:clamp(Math.round(y)||0,0,result.size-1)};
  render();
}

$('camera-rows').innerHTML=[0,1].map(i=>`<div class="joint-camera-row"><div class="camera-label">Camera ${i}<small>${i===0?'Pose anchored':'Pose estimated'}</small><em id="view-loss-${i}">—</em></div><figure><canvas id="target-${i}" width="192" height="192" role="img" aria-label="Camera ${i} fixed observed image"></canvas></figure><figure><canvas id="render-${i}" width="192" height="192" role="img" aria-label="Camera ${i} current predicted image"></canvas></figure><figure><canvas id="residual-${i}" width="192" height="192" role="img" aria-label="Camera ${i} absolute RGB residual, scale zero to 0.5"></canvas></figure></div>`).join('');
$('play').addEventListener('click',()=>running?stop():start());
$('step').addEventListener('click',()=>{stop();advance();$('status').textContent=`One optimizer update. Loss ${number(result.loss,5)}.`;});
$('batch').addEventListener('click',()=>start(50));
$('reset').addEventListener('click',()=>{
  stop();state=createJoint({size:24,lambda:Number($('lambda').value),mode:$('mode').value,schedule:$('schedule').value});
  result=evaluateJoint(state);$('history-note').textContent='Estimates reset; the same fixed photographs supervise every run.';render();
  $('status').textContent='Camera pose and splat centers reset. The selected objective and update choices are preserved.';
});
for(const id of ['mode','schedule'])$(id).addEventListener('change',optionsChanged);
$('lambda').addEventListener('input',optionsChanged);
$('paper-weight').addEventListener('click',()=>{$('lambda').value=.2;optionsChanged();});
$('pixel-view').addEventListener('change',()=>selectPixel(Number($('pixel-view').value),selected.x,selected.y));
for(const axis of ['x','y']) {
  $('pixel-'+axis).max=result.size-1;
  $('pixel-'+axis).addEventListener('input',()=>{
    const input=$('pixel-'+axis);
    if(input.value!==''&&Number.isFinite(input.valueAsNumber))selectPixel(selected.view,axis==='x'?input.valueAsNumber:selected.x,axis==='y'?input.valueAsNumber:selected.y);
  });
  $('pixel-'+axis).addEventListener('change',()=>{$('pixel-'+axis).value=selected[axis];});
}
if(!embedded)for(const view of [0,1])for(const kind of ['target','render','residual']) {
  const canvas=$(kind+'-'+view);
  canvas.addEventListener('click',event=>{
    const rect=canvas.getBoundingClientRect();
    selectPixel(view,Math.floor((event.clientX-rect.left)/rect.width*result.size),Math.floor((event.clientY-rect.top)/rect.height*result.size));
  });
}
window.addEventListener('bento-live-visibility',event=>{if(event.detail?.paused)stop();});
document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
window.addEventListener('pagehide',()=>stop());
window.addEventListener('error',()=>{stop('An unexpected error stopped the demo. Reload to restart.');});
render();
