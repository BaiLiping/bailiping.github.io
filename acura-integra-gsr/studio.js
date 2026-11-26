import * as THREE from '../honda-civic-type-r/vendor/three-r170.js?v=31c84364';
import {createStudioFloor} from '../honda-civic-type-r/studio-floor.js';

const $ = selector => document.querySelector(selector);
const canvas = $('#stage');
const rect = $('#image-rect');
const room = $('#room');
const overlay = $('#overlay');
const params = new URLSearchParams(location.search);
const capture = params.has('capture');
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
document.body.classList.toggle('capture', capture);
const frames = await fetch('./views.json').then(response => response.json());
const frameList = $('#photo-angle');
frames.forEach(frame => frameList.add(new Option(frame.title, frame.id)));
let active = Math.max(0, frames.findIndex(frame => frame.id === params.get('frame')));
let matched = null;
let scene, renderer, camera, model, floor;
let ready = false;
let pending = false;
let exteriorOrbit = true;
let lastAspect;
const exteriorRadius = aspect => Math.max(6.9,1.45+7.5/aspect);
let radius = 6, yaw = 0, elevation = .23;
const target = new THREE.Vector3(0,.6,0);
const panels = {hood:false,doors:false,hatch:false,passengerOnly:false};
const pivots = {};
const original = {};
const position = new THREE.Vector3();
const drag = new Map();
let gesture;

// Move complete GLB assemblies so trim, glass and hinges travel with their panels.
const assemblyOffsets = {
  Body: [0, 1.22, 0],
  Hood: [-.88, 2.12, 0],
  Liftback: [1.05, 1.92, 0],
  DoorLeft: [0, .63, 1.40],
  DoorRight: [-.45, .90, -2.05],
  Powertrain: [-1.17, .18, 0],
  Interior: [1.42, .08, .35],
  WheelFrontLeft: [-.27, 0, 1.05], WheelFrontRight: [-.27, 0, -1.05],
  WheelRearLeft: [.24, 0, 1.05], WheelRearRight: [.24, 0, -1.05],
};
const assemblyParts = [];
const assemblyCenter = new THREE.Vector3();
const assemblyFocus = new THREE.Vector3();
const assemblyCorners = [];
const assemblyDirection = new THREE.Vector3(-.78, .51, .85).normalize();
const assemblyRight = new THREE.Vector3(0, 1, 0).cross(assemblyDirection).normalize();
const assemblyUp = assemblyDirection.clone().cross(assemblyRight);
let assemblyDistance = 0;
let explosion = 0;
let intro = null;

function prepareAssembly() {
  const boxes = [];
  model.updateMatrixWorld(true);
  for (const [name, offset] of Object.entries(assemblyOffsets)) {
    const node = model.getObjectByName(name);
    const travel = new THREE.Vector3(...offset);
    assemblyParts.push({node, rest: node.position.clone(), travel});
    node.traverse(mesh => {
      if (!mesh.isMesh) return;
      if (!mesh.geometry.boundingBox) mesh.geometry.computeBoundingBox();
      boxes.push(mesh.geometry.boundingBox.clone().applyMatrix4(mesh.matrixWorld).translate(travel));
    });
  }
  const bounds = boxes[0].clone();
  for (const box of boxes) bounds.union(box);
  bounds.getCenter(assemblyCenter);
  for (const {min, max} of boxes) {
    for (const x of [min.x, max.x]) for (const y of [min.y, max.y]) for (const z of [min.z, max.z]) {
      const corner = new THREE.Vector3(x, y, z).sub(assemblyCenter);
      assemblyCorners.push(new THREE.Vector3(corner.dot(assemblyRight), corner.dot(assemblyUp), corner.dot(assemblyDirection)));
    }
  }
}
function fitAssembly() {
  const area = room.getBoundingClientRect();
  const bottomSpace = area.bottom - $('#bar').getBoundingClientRect().top + 12;
  const slope = Math.tan(THREE.MathUtils.degToRad(16)); // half the exterior field of view
  const side = .90 * slope * area.width / area.height;
  const top = .88 * slope;
  const bottom = Math.max(.25, 1 - 2 * bottomSpace / area.height) * slope;
  let leftEdge = Infinity, rightEdge = -Infinity, lowEdge = Infinity, highEdge = -Infinity;
  // Fit each edge of the perspective view, leaving the toolbar below the lowest wheel.
  for (const p of assemblyCorners) {
    leftEdge = Math.min(leftEdge, p.x - side * p.z);
    rightEdge = Math.max(rightEdge, p.x + side * p.z);
    lowEdge = Math.min(lowEdge, p.y - bottom * p.z);
    highEdge = Math.max(highEdge, p.y + top * p.z);
  }
  assemblyDistance = Math.max((rightEdge - leftEdge) / (2 * side), (highEdge - lowEdge) / (top + bottom));
  assemblyFocus.copy(assemblyCenter)
    .addScaledVector(assemblyRight, (leftEdge + rightEdge) / 2)
    .addScaledVector(assemblyUp, (lowEdge + highEdge + (bottom - top) * assemblyDistance) / 2);
}
function setExplosion(amount) {
  if (amount === explosion) return;
  explosion = amount;
  for (const {node, rest, travel} of assemblyParts) node.position.copy(rest).addScaledVector(travel, amount);
  $('#exploded-view').setAttribute('aria-pressed', String(amount > 0));
  floor.update();
}
function frameExterior(rear = false) {
  const aspect = Math.max(.40, room.clientWidth / room.clientHeight);
  const distance = THREE.MathUtils.lerp(exteriorRadius(aspect), assemblyDistance, explosion);
  target.set(-.16, .62, 0).lerp(assemblyFocus, explosion);
  camera.position.set(rear ? .75 : -.78, .26, .85).normalize()
    .lerp(assemblyDirection, explosion).normalize().multiplyScalar(distance).add(target);
  camera.lookAt(target);
  rememberOrbit();
}
function showExplosion(animate) {
  setExplosion(1);
  frameExterior();
  if (animate && !reducedMotion.matches) intro = {elapsed: 0, previous: null};
  invalidate();
}
function draw(now) {
  pending = false;
  if (intro && !document.hidden) {
    if (intro.previous !== null) intro.elapsed += now - intro.previous;
    intro.previous = now;
    const amount = 1 - THREE.MathUtils.smoothstep(intro.elapsed, 700, 3300);
    if (amount !== explosion) {
      setExplosion(amount);
      frameExterior();
    }
    if (amount === 0 || reducedMotion.matches) intro = null;
    else invalidate();
  }
  if (renderer && camera) renderer.render(scene,camera);
}
document.addEventListener('visibilitychange', () => {
  if (intro) { intro.previous = null; invalidate(); }
});
function invalidate() {
  if (!pending) { pending = true; requestAnimationFrame(draw); }
}
function redrawOrbit() {
  camera.up.set(0,1,0);
  camera.position.set(target.x + radius*Math.cos(elevation)*Math.cos(yaw), target.y + radius*Math.sin(elevation), target.z + radius*Math.cos(elevation)*Math.sin(yaw));
  camera.lookAt(target);
  invalidate();
}
function rememberOrbit() {
  position.copy(camera.position).sub(target);
  radius = position.length();
  yaw = Math.atan2(position.z,position.x);
  elevation = Math.asin(position.y/radius);
}
function layout() {
  if (!camera) return;
  const available = room.getBoundingClientRect();
  let width = available.width, height = available.height;
  if (matched) {
    const aspect = matched.width / matched.height;
    width = Math.min(width,height*aspect); height = width/aspect;
  }
  rect.style.width = width+'px'; rect.style.height = height+'px';
  camera.aspect = width/height;
  renderer.setSize(Math.round(width),Math.round(height),false);
  camera.updateProjectionMatrix();
  invalidate();
}
function setPanels(changes, angles = {}) {
  Object.assign(panels,changes);
  if (pivots.Hood) {
    for (const [name,axis,angle] of [['Hood','z',panels.hood ? -(angles.hood ?? 1.00) : 0],['Liftback','z',panels.hatch ? (angles.hatch ?? .93) : 0],['DoorLeft','y',panels.doors && !panels.passengerOnly ? -(angles.doors ?? .98) : 0],['DoorRight','y',panels.doors ? (angles.doors ?? .98) : 0]]) {
      pivots[name].quaternion.copy(original[name]);
      if (axis==='z') pivots[name].rotateZ(angle); else pivots[name].rotateY(angle);
    }
    floor.update();
  }
  document.querySelectorAll('[data-panel]').forEach(button => button.setAttribute('aria-pressed',String(panels[button.dataset.panel])));
  invalidate();
}
function clearMatch() {
  if (!matched) return;
  matched = null;
  overlay.hidden = true;
  $('#overlay-toggle').setAttribute('aria-pressed','false');
  $('#overlay-opacity').hidden = true;
  $('#photo-tools-3d').hidden = true;
  layout();
}
function setView(name) {
  if (!ready) return;
  intro = null;
  setExplosion(0);
  clearMatch();
  exteriorOrbit = name === 'front' || name === 'rear';
  if (name === 'front' || name === 'rear') {
    camera.fov = 32;
    frameExterior(name === 'rear');
  } else if (name === 'engine') {
    setPanels({hood:true}); target.set(-1.36,.55,.03);camera.position.set(-2.98,2.16,1.06);camera.fov=38;
  } else if (name === 'wheel') {
    target.set(-1.285,.30,.80);camera.position.set(-1.54,.49,2.10);camera.fov=35;
  } else {
    setPanels({doors:true,passengerOnly:true});target.set(-.42,.73,.07);camera.position.set(.16,1.00,-.52);camera.fov=66;
  }
  camera.up.set(0,1,0);camera.lookAt(target);camera.updateProjectionMatrix();rememberOrbit();layout();
}
function gallery() {return location.hash === '#integra-views';}
function selectPhoto(index) {
  active = (index+frames.length)%frames.length;
  const frame=frames[active];
  frameList.value=frame.id;
  $('#photo').alt=frame.title+' · Flamenco Black Pearl 2000 Acura Integra GS-R';
  $('#photo-error').hidden=true;
  $('#photo').src=frame.url;
  $('#full-photo').href=frame.url;
  $('#frame-count').textContent=(active+1)+' / '+frames.length+' · '+frame.width+' × '+frame.height;
}
function setURL(index,hash) {
  const url=new URL(location.href);
  url.searchParams.set('frame',frames[index].id);url.hash=hash;history.pushState(null,'',url);route();
}
function match(index) {
  if (!ready) return;
  const frame=frames[index];
  clearMatch();matched=frame;
  setPanels({hood:['open','engine'].includes(frame.state),doors:['open','cabin'].includes(frame.state),hatch:frame.state==='open',passengerOnly:frame.state==='cabin'},frame.angles);
  camera.position.fromArray(frame.camera.position);target.fromArray(frame.camera.target);
  camera.up.fromArray(frame.camera.up);camera.lookAt(target);camera.fov=frame.camera.fov;
  rememberOrbit();layout();
  $('#match-label').textContent=frame.title;$('#photo-tools-3d').hidden=false;
}
function route({opening = false} = {}) {
  intro = null;
  setExplosion(0);
  const url=new URL(location.href);
  active=Math.max(0,frames.findIndex(f=>f.id===url.searchParams.get('frame')));
  const photos=gallery();
  document.documentElement.classList.toggle('show-photos',photos);
  $('#photo-room').hidden=!photos;
  $('#model-link').setAttribute('aria-current',photos?'false':'page');
  $('#gallery-link').setAttribute('aria-current',photos?'page':'false');
  if (photos) selectPhoto(active);
  else if (url.searchParams.has('frame')) match(active);
  else if (ready) {
    setPanels({hood:false,doors:false,hatch:false,passengerOnly:false});
    setView('front');
    if (!capture && (opening || url.hash === '#assembly')) showExplosion(true);
  }
}
function baseURL(hash = 'model') {
  const url=new URL(location.href);url.searchParams.delete('frame');url.hash=hash;
  if (url.href!==location.href) history.pushState(null,'',url);
}
function leaveCalibration() {
  intro = null;
  if (matched) {baseURL();clearMatch();}
}
function initGL() {
  renderer=new THREE.WebGLRenderer({canvas,antialias:true,alpha:false,preserveDrawingBuffer:capture,powerPreference:'high-performance'});
  renderer.setPixelRatio(capture?1:Math.min(devicePixelRatio,2));
  renderer.outputColorSpace=THREE.SRGBColorSpace;
  renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.11;
  scene=new THREE.Scene();scene.background=new THREE.Color(0xe7e8ea);
  // Long reflection banks follow the existing car studios and describe the crowned sheet metal.
  const pmrem=new THREE.PMREMGenerator(renderer);
  const panorama=document.createElement('canvas');panorama.width=512;panorama.height=256;
  const context=panorama.getContext('2d');const pixels=context.createImageData(512,256);
  const smooth=THREE.MathUtils.smoothstep;
  for (let y=0;y<256;y++) {
    const lat=(.5-(y+.5)/256)*Math.PI;
    const high=smooth(lat,.24,.40)*(1-smooth(lat,.59,.83));
    const low=smooth(lat,-.18,-.06)*(1-smooth(lat,.035,.15));
    for (let x=0;x<512;x++) {
      const az=((x+.5)/512-.5)*2*Math.PI;
      const bank=smooth(Math.abs(Math.sin(az+.20)),.27,.85);
      const light=.17+.28*smooth(lat,.86,1.4)+high*(.24+2.9*bank)+low*(.12+1.25*bank);
      const k=4*(y*512+x);pixels.data[k]=light*64;pixels.data[k+1]=light*66;pixels.data[k+2]=light*69;pixels.data[k+3]=255;
    }
  }
  context.putImageData(pixels,0,0);const studio=new THREE.CanvasTexture(panorama);
  scene.environment=pmrem.fromEquirectangular(studio).texture;studio.dispose();pmrem.dispose();
  scene.environmentIntensity=1.38;
  scene.add(new THREE.HemisphereLight(0xf2f6fd,0x53575e,1.13));
  const key=new THREE.DirectionalLight(0xfff8ef,1.43);key.position.set(-3,6,3);scene.add(key);
  renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;
  key.castShadow=true;key.shadow.mapSize.set(2048,2048);
  Object.assign(key.shadow.camera,{left:-3,right:3,top:3,bottom:-3,near:.1,far:16});key.shadow.camera.updateProjectionMatrix();
  key.shadow.bias=-.00024;key.shadow.normalBias=.008;
  const edge=new THREE.DirectionalLight(0xe3edff,.63);edge.position.set(3,3,-4);scene.add(edge);
  camera=new THREE.PerspectiveCamera(32,1,.015,140);
  floor=createStudioFloor(scene,renderer,0xe7e8ea);
  layout();
  return new THREE.GLTFLoader().loadAsync('./integra.glb');
}
$('#previous-view').onclick=()=>setURL((active+frames.length-1)%frames.length,'integra-views');
$('#next-view').onclick=()=>setURL((active+1)%frames.length,'integra-views');
frameList.onchange=()=>setURL(frameList.selectedIndex,'integra-views');
$('#match-view').onclick=()=>setURL(active,'model');
$('#photo').onerror=()=>{$('#photo-error').hidden=false;};
$('#gallery-link').onclick=event=>{event.preventDefault();setURL(active,'integra-views');};
$('#model-link').onclick=event=>{event.preventDefault();baseURL();route();};
$('#overlay-toggle').onclick=()=>{
  if (!matched) return;
  const show=overlay.hidden;overlay.hidden=!show;
  if (show) overlay.src=matched.url;
  $('#overlay-toggle').setAttribute('aria-pressed',String(show));$('#overlay-opacity').hidden=!show;
};
$('#overlay-opacity').oninput=event=>{overlay.style.opacity=event.target.value/100;};
$('#reset').onclick=()=>{baseURL();route();$('#more').open=false;};
$('#exploded-view').onclick = () => {
  const assemble = explosion > 0;
  baseURL(); route();
  if (!assemble) showExplosion(false);
  $('#more').open = false;
};
$('#replay-assembly').onclick = event => {
  event.preventDefault(); baseURL('assembly'); route(); $('#more').open = false;
};
document.querySelectorAll('[data-panel]').forEach(button=>button.onclick=()=>{
  leaveCalibration();
  if (explosion) setView('front');
  const key=button.dataset.panel;setPanels({[key]:!panels[key],passengerOnly:false});
});
document.querySelectorAll('[data-view]').forEach(button=>button.onclick=()=>{
  leaveCalibration();setView(button.dataset.view);$('#more').open=false;
});
function gestures() {
  const a=[...drag.values()];return {x:a[0][0],y:a[0][1],gap:a.length>1?Math.hypot(a[1][0]-a[0][0],a[1][1]-a[0][1]):0};
}
canvas.addEventListener('pointerdown',e=>{
  if (!ready || e.button!==0) return;
  intro = null;
  canvas.setPointerCapture(e.pointerId);drag.set(e.pointerId,[e.clientX,e.clientY]);gesture=gestures();
});
canvas.addEventListener('pointermove',e=>{
  if (!drag.has(e.pointerId)) return;
  drag.set(e.pointerId,[e.clientX,e.clientY]);const next=gestures();
  if (Math.abs(next.x-gesture.x)+Math.abs(next.y-gesture.y)+Math.abs(next.gap-gesture.gap)<.15) return;
  leaveCalibration();
  if (next.gap&&gesture.gap) radius=Math.min(22,Math.max(.20,radius*gesture.gap/next.gap));
  else {yaw+=(next.x-gesture.x)*.007;elevation=Math.min(1.48,Math.max(-.10,elevation+(next.y-gesture.y)*.006));}
  gesture=next;redrawOrbit();
});
function release(e) {drag.delete(e.pointerId);if (drag.size) gesture=gestures();}
canvas.addEventListener('pointerup',release);canvas.addEventListener('pointercancel',release);
canvas.addEventListener('wheel',e=>{e.preventDefault();if (!ready) return;leaveCalibration();radius=Math.max(.20,Math.min(22,radius*Math.exp(Math.sign(e.deltaY)*.10)));redrawOrbit();},{passive:false});
canvas.addEventListener('keydown',e=>{
  if (!ready || !['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','=','-','Home'].includes(e.key)) return;
  e.preventDefault();leaveCalibration();
  if (e.key==='Home') {$('#reset').click();return;}
  if (e.key==='ArrowLeft') yaw-=.10;
  if (e.key==='ArrowRight') yaw+=.10;
  if (e.key==='ArrowUp') elevation=Math.min(1.48,elevation+.08);
  if (e.key==='ArrowDown') elevation=Math.max(-.10,elevation-.08);
  if (['+','='].includes(e.key)) radius=Math.max(.20,radius*.9);
  if (e.key==='-') radius=Math.min(22,radius/.9);
  redrawOrbit();
});
window.addEventListener('popstate',route);window.addEventListener('hashchange',route);
const observer=new ResizeObserver(()=>{
  const aspect=Math.max(.40,room.clientWidth/Math.max(1,room.clientHeight));
  const oldAssemblyDistance = assemblyDistance;
  const oldFocus = assemblyFocus.clone();
  if (ready) fitAssembly();
  if (ready && exteriorOrbit && !matched && !gallery() && lastAspect) {
    const previous = THREE.MathUtils.lerp(exteriorRadius(lastAspect), oldAssemblyDistance, explosion);
    radius *= THREE.MathUtils.lerp(exteriorRadius(aspect), assemblyDistance, explosion) / previous;
    target.addScaledVector(oldFocus.sub(assemblyFocus), -explosion);
    redrawOrbit();
  }
  lastAspect=aspect;layout();
});observer.observe(room);
route();$('#match-view').disabled=true;
try {
  const gltf=await initGL(); model=gltf.scene;scene.add(model);
  model.traverse(node=>{
    if (!node.isMesh) return;
    floor.addCaster(node);
    node.receiveShadow=true;
    node.castShadow=!node.material.transparent;
    if (node.parent?.name==='Interior' || node.material.name==='Inner wheel well') {
      node.material=node.material.clone();node.material.envMapIntensity=.30;
    }
  });
  for (const name of ['Hood','Liftback','DoorLeft','DoorRight']) {pivots[name]=model.getObjectByName(name);original[name]=pivots[name].quaternion.clone();}
  prepareAssembly();fitAssembly();
  ready=true;floor.update();$('#loader').hidden=true;$('#bar').inert=false;$('#match-view').disabled=false;
  setView('front');route({opening:true});renderer.render(scene,camera);
  document.documentElement.dataset.modelReady='true';
} catch (error) {
  console.error(error);
  $('#load-status').replaceChildren('3D is unavailable in this browser. ');
  const link=document.createElement('a');link.href='./frame-by-frame/';link.textContent='View the photo comparisons';$('#load-status').append(link);
  $('#bar').hidden=true;
}
