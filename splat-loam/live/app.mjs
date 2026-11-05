import {DEG,clamp,add,mul,sphericalRay,makeRenderScene,renderRay,rangeImage,pixelAngles,pixelForAngles,makeMap,makeScan,transformPoint,registerStep,evaluateRegistration,poseError} from './model.mjs';
import {rangeImageRgba,rangeColor} from './range-image.mjs';
const $=id=>document.getElementById(id);
const params=new URLSearchParams(location.search),lab=params.get('lab')==='registration'?'registration':'rendering',embedded=params.get('embed')==='1';
const colors={paper:'#fffcf6',ink:'#183a3c',muted:'#627675',rule:'#d7ddd3',teal:'#167d8d',orange:'#c66b3d',sage:'#647d65',grey:'#a5b3ad'};
for(const [id,name] of [['render-tab','rendering'],['register-tab','registration']]){const el=$(id);el.href=`?lab=${name}${embedded?'&embed=1':''}`;if(name===lab)el.setAttribute('aria-current','page');}
$('rendering-lab').hidden=lab!=='rendering';$('registration-lab').hidden=lab!=='registration';
$('lab-number').textContent=lab==='rendering'?'LAB 01':'LAB 02';$('lab-title').textContent=lab==='rendering'?'LiDAR rays → range cells':'Geometric registration';
$('scope').innerHTML=lab==='rendering'
  ?'Splat-LOAM uses geometry, not RGB. Synthetic angular window · exact ray–plane intersections · paper D and code-style D/O · median mixture omitted. <a href="https://arxiv.org/html/2503.17491v1#S3.SS2" target="_blank" rel="noopener">§3.2 ↗</a>'
  :'Synthetic SE(2) reduction · geometric term only. The paper combines SE(3) point-to-plane geometry with range-image warping; its map update is separate. <a href="https://arxiv.org/html/2503.17491v1#S3.SS3.SSS3" target="_blank" rel="noopener">§3.3.3 ↗</a>';

function canvas(id){const el=$(id),r=el.getBoundingClientRect(),d=Math.min(devicePixelRatio||1,2);if(r.width<1||r.height<1)return null;const w=r.width,h=r.height;if(el.width!==Math.round(w*d)||el.height!==Math.round(h*d)){el.width=Math.round(w*d);el.height=Math.round(h*d);}const c=el.getContext('2d');c.setTransform(d,0,0,d,0,0);c.clearRect(0,0,w,h);return {c,w,h};}
function line(c,points,color,width=1,dash=[]){if(!points.length)return;c.beginPath();c.moveTo(...points[0]);points.slice(1).forEach(p=>c.lineTo(...p));c.strokeStyle=color;c.lineWidth=width;c.setLineDash(dash);c.stroke();c.setLineDash([]);}
function circle(c,p,r,fill,stroke=null,width=1){c.beginPath();c.arc(...p,r,0,Math.PI*2);if(fill){c.fillStyle=fill;c.fill();}if(stroke){c.strokeStyle=stroke;c.lineWidth=width;c.stroke();}}
function text(c,str,x,y,{size=10,color=colors.muted,align='left',weight=400}={}){c.font=`${weight} ${size}px system-ui,sans-serif`;c.fillStyle=color;c.textAlign=align;c.fillText(str,x,y);c.textAlign='left';}
function arrow(c,p,q,color,width=1.5,size=5){line(c,[p,q],color,width);const a=Math.atan2(q[1]-p[1],q[0]-p[0]);line(c,[[q[0]-size*Math.cos(a-.45),q[1]-size*Math.sin(a-.45)],q,[q[0]-size*Math.cos(a+.45),q[1]-size*Math.sin(a+.45)]],color,width);}
const fmt=(v,n=2)=>Number.isFinite(v)?v.toFixed(n).replace('-', '−'):'—';
let paused=document.hidden,animation=0,lastTick=0;

const render={tilt:25,scale:1.15,opacity:.82,yaw:0,azimuth:0,elevation:0,mode:'expectedDepth',colorMode:'gray',width:72,height:32,col:36,row:16,splats:[],image:null,result:null};
const texture=document.createElement('canvas');
function updateRender(recompute=true){
  render.splats=makeRenderScene(render);
  if(recompute)render.image=rangeImage(render.splats,{yaw:render.yaw,width:render.width,height:render.height});
  Object.assign(render,pixelForAngles(render.image,render.azimuth,render.elevation));
  Object.assign(render,pixelAngles(render.image,render.col,render.row));
  const da=(render.image.azMax-render.image.azMin)/render.width,de=(render.image.elMax-render.image.elMin)/render.height;
  for(const [name,min,max,step] of [['azimuth',render.image.azMin+da/2,render.image.azMax-da/2,da],['elevation',render.image.elMin+de/2,render.image.elMax-de/2,de]]){
    Object.assign($(name),{min:String(min),max:String(max),step:String(step),value:String(render[name])});
  }
  const ray=sphericalRay(render.azimuth*DEG,render.elevation*DEG,render.yaw*DEG);render.result=renderRay(ray,render.splats);
  for(const name of ['tilt','scale','opacity','yaw','azimuth','elevation']){
    const suffix=['scale'].includes(name)?' m':['opacity'].includes(name)?'':'°';const decimals=['scale','opacity'].includes(name)?2:['azimuth','elevation'].includes(name)?1:0;
    $(`${name}-out`).textContent=fmt(render[name],decimals)+suffix;
  }
  $('selected-cell').textContent=`Cell ${render.col+1}, ${render.row+1} · one sampled direction`;
  $('footprint-count').textContent=`${render.image.footprintCounts.A.toLocaleString()} / ${(render.width*render.height).toLocaleString()}`;
  $('footprint-detail').textContent=`${render.width} × ${render.height} rays · chosen 3σ boundary`;
  $('accumulated-depth').textContent=fmt(render.result.depth)+' m';
  $('expected-depth').textContent=render.result.expectedDepth===null?'undefined':fmt(render.result.expectedDepth)+' m';
  $('ray-opacity').textContent=fmt(render.result.opacity,3);
  const rows=render.result.hits.filter(h=>h.alpha>=0.0005);
  $('hit-rows').innerHTML=rows.length?rows.map(h=>`<tr><td><i class="hit-dot" style="background:${h.color}"></i>${h.id}</td><td>${fmt(h.range)} m</td><td>${fmt(h.alpha,3)}</td><td>${fmt(h.weight,3)}</td></tr>`).join(''):'<tr><td colspan="4" style="text-align:left;color:#627675">No appreciable Gaussian support.</td></tr>';
  // Keep a stable layout when a selected ray has fewer useful hits.
  for(let i=rows.length||1;i<3;i++)$('hit-rows').insertAdjacentHTML('beforeend','<tr aria-hidden="true"><td>·</td><td>—</td><td>—</td><td>—</td></tr>');
  drawRendering();
}
function drawRendering(){if(lab!=='rendering'||paused)return;drawRenderScene();drawRangeImage();}
function drawRenderScene(){const target=canvas('render-scene');if(!target)return;const {c,w,h}=target;
  const s=Math.min(w/12.7,h/7.5),origin=[w*.17,h*.61];
  const project=p=>[origin[0]+(p[0]*.91-p[1]*.40)*s,origin[1]+(p[0]*.10+p[1]*.20-p[2]*.90)*s];
  c.save();
  for(let x=0;x<=10;x+=2)line(c,[project([x,-4,-2.2]),project([x,4,-2.2])],'#e5e9df');
  for(let y=-4;y<=4;y+=2)line(c,[project([0,y,-2.2]),project([10,y,-2.2])],'#e5e9df');
  // The faint fan samples the selected elevation row; it is not a reflection lobe.
  const sensor=project([0,0,0]);
  for(let col=0;col<render.width;col+=Math.max(1,Math.round(render.width/12))){
    const a=pixelAngles(render.image,col,render.row),ray=sphericalRay(a.azimuth*DEG,a.elevation*DEG,render.yaw*DEG);
    line(c,[sensor,project(mul(ray,9))],'#c8d4ca',.7);
  }
  // Fixed oblique view. Ellipses are true world-space contours projected onto the canvas.
  for(const splat of [...render.splats].sort((a,b)=>b.center[0]-a.center[0])){
    for(const radius of splat.id==='A'?[3,2,1.5,1,.5]:[2,1.5,1,.5]){
      const points=Array.from({length:65},(_,i)=>{const a=i/64*Math.PI*2;return project(add(add(splat.center,mul(splat.u,Math.cos(a)*splat.scaleU*radius)),mul(splat.v,Math.sin(a)*splat.scaleV*radius)));});
      c.beginPath();c.moveTo(...points[0]);points.slice(1).forEach(p=>c.lineTo(...p));c.closePath();c.setLineDash(radius===3?[4,3]:[]);c.fillStyle=splat.color;c.globalAlpha=(.055+.07*Math.exp(-radius*radius/2))*splat.opacity;c.fill();c.globalAlpha=.45;c.strokeStyle=splat.color;c.lineWidth=.8;c.stroke();c.setLineDash([]);c.globalAlpha=1;
    }
    const p=project(splat.center);circle(c,p,2.8,splat.color);text(c,`${splat.id}`,p[0]+7,p[1]-6,{size:12,color:splat.color,weight:700});
    if(splat.id==='A'){const q=project(add(splat.center,mul(splat.normal,1)));arrow(c,p,q,splat.color,1,4);text(c,'n',q[0]+4,q[1]-1,{color:splat.color,size:10});}
  }
  const ray=sphericalRay(render.azimuth*DEG,render.elevation*DEG,render.yaw*DEG),o=project([0,0,0]);
  line(c,[o,project(mul(ray,10))],'#55776f',1.4,[5,4]);
  const useful=render.result.hits.filter(hit=>hit.alpha>.01&&hit.range<14);
  useful.forEach((hit,index)=>{const p=project(hit.point);circle(c,p,4.2,'#fffcf6',hit.color,1.8);text(c,`${index+1}`,p[0]+7,p[1]+13,{color:hit.color,size:9,weight:700});});
  const sensorEnd=project(sphericalRay(0,0,render.yaw*DEG).map(x=>x*.85));arrow(c,o,sensorEnd,colors.ink,2,6);circle(c,o,4,colors.ink);text(c,'LiDAR',o[0]-3,o[1]+19,{size:10,color:colors.ink});
  arrow(c,project([0,-3,-2.2]),project([1,-3,-2.2]),colors.muted,1,4);text(c,'x',...project([1.2,-3,-2.2]),{size:9});
  const altitude=project([0,-3,-1.2]);arrow(c,project([0,-3,-2.2]),altitude,colors.muted,1,4);text(c,'z',altitude[0]-5,altitude[1]-5,{size:9});
  text(c,'Dashed ellipse: A’s chosen 3σ footprint',10,16,{size:9});
  text(c,'Thin lines: sample beams in the selected row',w/2,h-7,{size:9,align:'center'});
  c.restore();
}
function imageRect(w,h){return {x:37,y:9,w:w-46,h:h-37};}
function drawRangeImage(){const target=canvas('range-image');if(!target||!render.image)return;const {c,w,h}=target,r=imageRect(w,h),img=render.image;
  texture.width=img.width;texture.height=img.height;const tc=texture.getContext('2d'),pixels=tc.createImageData(img.width,img.height);
  pixels.data.set(rangeImageRgba(img,render.mode,render.colorMode));
  tc.putImageData(pixels,0,0);c.imageSmoothingEnabled=false;c.drawImage(texture,r.x,r.y,r.w,r.h);
  // Pixel centres are the same directions used for numerical ray rendering.
  const dx=r.w/img.width,dy=r.h/img.height;
  if(dx>=3&&dy>=2){
    for(let x=0;x<=img.width;x++)line(c,[[r.x+x*dx,r.y],[r.x+x*dx,r.y+r.h]],'rgba(255,252,246,.13)',.5);
    for(let y=0;y<=img.height;y++)line(c,[[r.x,r.y+y*dy],[r.x+r.w,r.y+y*dy]],'rgba(255,252,246,.13)',.5);
  }
  for(const a of [-40,0,40]){const x=r.x+(a-img.azMin)/(img.azMax-img.azMin)*r.w;text(c,`${a}°`,x,r.y+r.h+11,{size:8,align:'center'});}
  for(const el of [-20,0,20]){const y=r.y+(img.elMax-el)/(img.elMax-img.elMin)*r.h;text(c,`${el}°`,r.x-4,y+3,{size:8,align:'right'});}
  text(c,'azimuth',r.x+r.w/2,h-2,{size:8,align:'center'});
  c.save();c.translate(9,r.y+r.h/2);c.rotate(-Math.PI/2);text(c,'elevation',0,0,{size:8,align:'center'});c.restore();
  const px=r.x+(render.col+.5)*dx,py=r.y+(render.row+.5)*dy;
  c.strokeStyle='#fff';c.lineWidth=3;c.strokeRect(r.x+render.col*dx,r.y+render.row*dy,dx,dy);
  c.strokeStyle=colors.orange;c.lineWidth=1.4;c.strokeRect(r.x+render.col*dx,r.y+render.row*dy,dx,dy);
  line(c,[[px-10,py],[px-5,py]],colors.orange,1.2);line(c,[[px+5,py],[px+10,py]],colors.orange,1.2);
  const isOpacity=render.mode==='opacity';
  $('image-scale').textContent=isOpacity?'O = 0':'0 m';$('image-scale-end').textContent=isOpacity?'O = 1':'12+ m';
  $('range-gradient').style.background=`linear-gradient(to right,${Array.from({length:13},(_,i)=>`rgb(${rangeColor(isOpacity?i/12:i,render.mode,render.colorMode).join(',')})`).join(',')})`;
  $('range-colors').disabled=isOpacity;
  $('image-caption').textContent=isOpacity?'Each cell = one direction · value is accumulated opacity':render.colorMode==='gray'?'Each cell = one direction · shade shows range, not RGB':'False color displays range · no camera color is measured';
}
function selectPixel(col,row){
  Object.assign(render,pixelAngles(render.image,clamp(col,0,render.width-1),clamp(row,0,render.height-1)));
  updateRender(false);
}
for(const id of ['tilt','scale','opacity','yaw','azimuth','elevation'])$(id).addEventListener('input',e=>{render[id]=+e.target.value;updateRender(!['azimuth','elevation'].includes(id));});
$('range-mode').addEventListener('change',e=>{render.mode=e.target.value;drawRangeImage();});
$('range-colors').addEventListener('change',e=>{render.colorMode=e.target.value;drawRangeImage();});
$('scan-grid').addEventListener('change',e=>{render.width=+e.target.value;render.height=render.width===72?32:64;updateRender();});
$('render-reset').addEventListener('click',()=>{
  Object.assign(render,{tilt:25,scale:1.15,opacity:.82,yaw:0,azimuth:0,elevation:0,width:72,height:32,mode:'expectedDepth',colorMode:'gray'});
  for(const id of ['tilt','scale','opacity','yaw','azimuth','elevation'])$(id).value=render[id];
  $('scan-grid').value='72';$('range-mode').value=render.mode;$('range-colors').value=render.colorMode;updateRender();
});
$('range-image').addEventListener('pointerdown',e=>{
  const box=e.currentTarget.getBoundingClientRect(),r=imageRect(box.width,box.height),x=e.clientX-box.left-r.x,y=e.clientY-box.top-r.y;
  if(x<0||x>r.w||y<0||y>r.h)return;
  e.currentTarget.focus();selectPixel(Math.floor(x/r.w*render.width),Math.floor(y/r.h*render.height));
});
$('range-image').addEventListener('keydown',e=>{const d={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]}[e.key];if(!d)return;e.preventDefault();e.stopPropagation();selectPixel(render.col+d[0],render.row+d[1]);});

const registration={kind:'corner',initial:[.7,-.6,9*DEG],pose:[.7,-.6,9*DEG],noise:.015,map:[],scan:[],history:[],running:false,iteration:0,showMatches:false,lastResult:null};
function readInitial(){return [+ $('initial-x').value,+ $('initial-y').value,+ $('initial-yaw').value*DEG];}
function resetRegistration(){registration.running=false;registration.initial=readInitial();registration.pose=[...registration.initial];registration.noise=+$('noise').value;registration.kind=$('scene-kind').value;registration.map=makeMap(registration.kind);registration.scan=makeScan(registration.map,{noise:registration.noise});registration.iteration=0;registration.lastResult=null;
  const evaln=evaluateRegistration(registration.scan,registration.map,registration.pose),error=poseError(registration.pose);registration.history=[{iteration:0,rmse:evaln.rmse,error:error.translation,yaw:error.yaw}];
  for(const [id,n,s] of [['initial-x',2,' m'],['initial-y',2,' m'],['initial-yaw',1,'°'],['noise',3,' m']])$(`${id}-out`).textContent=fmt(+$(id).value,n)+s;
  $('registration-status').textContent='Ready · press Run or Step.';$('registration-status').classList.remove('warn');updateRegistration();
}
function stepRegistration(){
  const r=registerStep(registration.scan,registration.map,registration.pose);registration.lastResult=r;registration.pose=r.pose;registration.iteration++;
  const err=poseError(r.pose);registration.history.push({iteration:registration.iteration,rmse:r.after.rmse,error:err.translation,yaw:err.yaw});
  const smallStep=Math.hypot(r.step[0],r.step[1],r.step[2])<1e-6,done=smallStep||registration.iteration>=20;
  if(done)registration.running=false;
  const status=$('registration-status');status.classList.toggle('warn',r.unobservableX);
  status.textContent=r.unobservableX?`Step ${registration.iteration} · x is unobservable; its ${fmt(Math.abs(registration.pose[0]))} m offset remains.`:done?`Converged · ${registration.iteration} steps · the remaining residual reflects the data.`:`Step ${registration.iteration} · recomputed matches and updated the pose.`;
  updateRegistration();
}
function updateRegistration(){const last=registration.history.at(-1);$('run').textContent=registration.running?'Pause':'Run';$('residual-value').textContent=fmt(last.rmse,3)+' m';$('position-error').textContent=fmt(last.error,3)+' m';$('yaw-error').textContent=fmt(last.yaw,2)+'°';
  $('insight').classList.toggle('warn',registration.kind==='corridor');$('insight').innerHTML=registration.kind==='corridor'?'<b>A corridor cannot resolve motion along itself.</b> Parallel normals constrain y and yaw. The residual can vanish while the x error survives.':'<b>Corner geometry constrains the pose.</b> Surfaces with different normals resolve both translations and yaw.';
  drawRegistration();
}
function drawRegistration(){if(lab!=='registration'||paused)return;drawAlignment();drawConvergence();}
function drawAlignment(){const target=canvas('registration-scene');if(!target)return;const {c,w,h}=target,cor=registration.kind==='corridor';const bounds=cor?[-5.6,7.6,-4,4]:[-4.7,5.5,-3.7,4.1];const s=Math.min((w-35)/(bounds[1]-bounds[0]),(h-30)/(bounds[3]-bounds[2]));const mx=(bounds[0]+bounds[1])/2,my=(bounds[2]+bounds[3])/2;const p=point=>[w/2+(point[0]-mx)*s,h/2-(point[1]-my)*s];
  for(let x=Math.ceil(bounds[0]);x<=bounds[1];x++){line(c,[p([x,bounds[2]]),p([x,bounds[3]])],'#e9ece3',.6);if(x%2===0)text(c,`${x}`,p([x,bounds[2]])[0],h-1,{size:8,align:'center'});}
  for(let y=Math.ceil(bounds[2]);y<=bounds[3];y++){line(c,[p([bounds[0],y]),p([bounds[1],y])],'#e9ece3',.6);}
  for(const mapline of registration.map)line(c,[p(mapline.a),p(mapline.b)],colors.teal,3);
  registration.scan.forEach((point,i)=>{if(i%2===0)circle(c,p(transformPoint(point,registration.initial)),1.4,'#bfc7bd');});
  if(registration.showMatches){const e=evaluateRegistration(registration.scan,registration.map,registration.pose);e.matches.forEach((m,i)=>{if(i%3===0)line(c,[p(m.p),p([m.p[0]-m.residual*m.normal[0],m.p[1]-m.residual*m.normal[1]])],'#c8a17c',.8);});}
  registration.scan.forEach(point=>circle(c,p(transformPoint(point,registration.pose)),2.1,colors.orange));
  const truth=p([0,0]),now=p(registration.pose);circle(c,truth,5,null,colors.teal,1.2);arrow(c,truth,p([.8,0]),colors.teal,1.2,4);
  arrow(c,now,p([registration.pose[0]+.9*Math.cos(registration.pose[2]),registration.pose[1]+.9*Math.sin(registration.pose[2])]),colors.orange,2,5);circle(c,now,4,colors.orange);text(c,'pose',now[0]+6,now[1]+14,{size:9,color:colors.orange});
  if(cor){const a=p([-3.6,.1]),b=p([4.7,.1]);arrow(c,a,b,'#bdad8d',1,4);arrow(c,b,a,'#bdad8d',1,4);text(c,'no geometric constraint along x',w/2,p([0,.1])[1]-9,{size:9,color:'#977747',align:'center'});}
  text(c,'top-down view · metres',8,13,{size:9});text(c,'Truth ⊕ is evaluation only',w-8,13,{size:9,align:'right'});
}
function drawConvergence(){const target=canvas('convergence-chart');if(!target)return;const {c,w,h}=target,left=32,top=12,bottom=h-23,right=w-8,H=bottom-top,W=right-left;const history=registration.history,maxX=Math.max(6,registration.iteration),maxY=Math.max(.1,...history.flatMap(x=>[x.rmse,x.error]))*1.1;const p=(x,y)=>[left+x/maxX*W,bottom-y/maxY*H];
  for(let j=0;j<=3;j++){const y=maxY*j/3;line(c,[p(0,y),p(maxX,y)],'#e2e7dc',.8);text(c,fmt(y,2),left-5,p(0,y)[1]+3,{size:8,align:'right'});}
  for(let i=0;i<=maxX;i+=maxX>10?4:2)text(c,`${i}`,p(i,0)[0],bottom+14,{size:8,align:'center'});
  text(c,'m',left-19,9,{size:8});text(c,'iteration',right,bottom+14,{size:8,align:'right'});
  for(const [key,color] of [['error',colors.orange],['rmse',colors.teal]]){line(c,history.map(row=>p(row.iteration,row[key])),color,2);history.forEach(row=>circle(c,p(row.iteration,row[key]),2.7,colors.paper,color,1.3));}
}
function tick(time){animation=0;if(paused||!registration.running)return;if(time-lastTick>330){lastTick=time;stepRegistration();}if(registration.running)animation=requestAnimationFrame(tick);}
function startLoop(){if(!paused&&registration.running&&!animation){lastTick=performance.now()-350;animation=requestAnimationFrame(tick);}}
$('run').addEventListener('click',()=>{registration.running=!registration.running;updateRegistration();startLoop();});
$('step').addEventListener('click',()=>{registration.running=false;stepRegistration();});
$('registration-reset').addEventListener('click',resetRegistration);
for(const id of ['initial-x','initial-y','initial-yaw','noise'])$(id).addEventListener('input',resetRegistration);
$('scene-kind').addEventListener('change',resetRegistration);
$('show-matches').addEventListener('change',e=>{registration.showMatches=e.target.checked;drawAlignment();});
window.addEventListener('bento-live-visibility',e=>{paused=Boolean(e.detail?.paused);if(paused&&animation){cancelAnimationFrame(animation);animation=0;}if(!paused){drawRendering();drawRegistration();startLoop();}});
document.addEventListener('visibilitychange',()=>{paused=document.hidden;if(!paused){drawRendering();drawRegistration();startLoop();}});
let resizeFrame=0;new ResizeObserver(()=>{cancelAnimationFrame(resizeFrame);resizeFrame=requestAnimationFrame(()=>{drawRendering();drawRegistration();});}).observe(document.querySelector('.laboratory:not([hidden])'));
// Read-only numerical summary for presentation QA and keyboard-accessible controls.
window.splatLab={get snapshot(){return lab==='rendering'?{lab,depth:render.result?.depth,expectedDepth:render.result?.expectedDepth,opacity:render.result?.opacity,selectedAzimuth:render.azimuth,selectedElevation:render.elevation,selectedColumn:render.col,selectedRow:render.row,grid:{width:render.width,height:render.height},footprintCount:render.image?.footprintCounts.A,footprintSigma:render.image?.footprintSigma,mode:render.mode,colorMode:render.colorMode,selectedImageRange:render.image?.expectedDepth[render.row*render.width+render.col]}:{lab,kind:registration.kind,pose:[...registration.pose],iteration:registration.iteration,running:registration.running,paused,history:registration.history.map(x=>({...x}))};}};
if(lab==='rendering')updateRender();else resetRegistration();

