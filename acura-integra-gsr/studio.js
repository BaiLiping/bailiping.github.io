import * as THREE from '../honda-civic-type-r/vendor/three-r170.js?v=31c84364';
import {createStudioFloor} from '../honda-civic-type-r/studio-floor.js';
import {bindPartInteraction} from './part-interaction.js?v=8cb3fe9cf2';

const $ = selector => document.querySelector(selector);
const canvas = $('#stage');
const rect = $('#image-rect');
const room = $('#room');
const overlay = $('#overlay');
const params = new URLSearchParams(location.search);
const capture = params.has('capture');
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
document.body.classList.toggle('capture', capture);
let frames = [];
const frameList = $('#photo-angle');
let active = 0;
let matched = null;
let scene, renderer, camera, model, floor;
let ready = false;
let pending = false;
let floorDirty = false;
let exteriorOrbit = true;
let currentView = 'front';
let lastAspect;
const exteriorRadius = aspect => Math.max(6.9,1.45+7.5/aspect);
const gearboxRadius = aspect => Math.max(.70, .82 / aspect);
const cabinFov = aspect => THREE.MathUtils.clamp(THREE.MathUtils.radToDeg(2 * Math.atan(.47 / aspect)), 72, 96);
let radius = 6, yaw = 0, elevation = .23;
const target = new THREE.Vector3(0,.6,0);
const panelSpec = {
  hood: ['Hood', 'z', -1], hatch: ['Liftback', 'z', .93],
  leftDoor: ['DoorLeft', 'y', -.98], rightDoor: ['DoorRight', 'y', .98],
};
const panels = {hood:false, hatch:false, leftDoor:false, rightDoor:false};
const panelAngles = {hood:0, hatch:0, leftDoor:0, rightDoor:0};
const panelTargets = {...panelAngles};
let hingeMotion;
const pivots = {};
const original = {};
const position = new THREE.Vector3();
const frontCenter = new THREE.Vector3(-.16,.62,0);
let partInteraction;
let detachedCount = 0;
let steering = false;
let spinning = false;
let spinAngle = 0;
let lastSpinTime;
const frontWheels = [];
const rollingMeshes = [];

// Move complete GLB assemblies so trim, glass and hinges travel with their panels.
const assemblyOffsets = {
  Body: [0, 1.22, 0],
  Hood: [-.88, 2.12, 0],
  Liftback: [1.05, 1.92, 0],
  DoorLeft: [0, .63, 1.40],
  DoorRight: [-.45, .90, -2.05],
  Powertrain: [-1.17, .18, 0],
  Transmission: [-1.10, .32, -1.30],
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
let spreadMotion = null;
let firstModelVisit = true;
let lastRoute = '';

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
  explosion = amount;
  partInteraction?.setSeparation(amount);
  updateInspectionTools();
}
function updateInspectionTools() {
  const active = ready && location.hash === '#exploded' && !matched && !gallery();
  $('#exploded-view').setAttribute('aria-pressed', String(active || (spreadMotion ? spreadMotion.to > 0 : explosion > 0)));
  $('#separation-control').hidden = !active;
  const percent = Math.round(explosion * 100);
  $('#separation').value = percent;
  $('#separation-control output').value = percent + '%';
  $('#reattach').hidden = detachedCount === 0;
  $('#reattach').textContent = detachedCount === 1 ? 'Reattach 1 part' : 'Reattach ' + detachedCount + ' parts';
  $('#inspection-tools').hidden = !active && !detachedCount;
}
function frameExterior(rear = false) {
  const aspect = Math.max(.40, room.clientWidth / room.clientHeight);
  const distance = THREE.MathUtils.lerp(exteriorRadius(aspect), assemblyDistance, explosion);
  target.copy(frontCenter).lerp(assemblyFocus, explosion);
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
  if (document.hidden || gallery()) return;
  if (intro && !document.hidden) {
    if (intro.previous !== null) {
      const elapsed = now - intro.previous;
      // A slow first GPU frame can consume the hold. Still show the full assembly motion.
      intro.elapsed = intro.elapsed < 1200 ? Math.min(1200, intro.elapsed + elapsed) : intro.elapsed + Math.min(elapsed, 120);
    }
    intro.previous = now;
    const amount = 1 - THREE.MathUtils.smoothstep(intro.elapsed, 1200, 3300);
    if (amount !== explosion) {
      setExplosion(amount);
      frameExterior();
    }
    if (amount === 0 || reducedMotion.matches) intro = null;
    else invalidate();
  }
  if (spreadMotion) {
    const t = THREE.MathUtils.smoothstep(now - spreadMotion.start, 0, 430);
    setExplosion(THREE.MathUtils.lerp(spreadMotion.from, spreadMotion.to, t));
    frameExterior();
    if (t === 1) spreadMotion = null;
    else invalidate();
  }
  if (hingeMotion) {
    const t = THREE.MathUtils.smoothstep(now - hingeMotion.start, 0, 310);
    for (const key of Object.keys(panelSpec)) applyHinge(key, THREE.MathUtils.lerp(hingeMotion.from[key], panelTargets[key], t));
    floorDirty = true;
    if (t === 1) hingeMotion = null;
    else invalidate();
  }
  if (spinning) {
    if (lastSpinTime !== undefined) spinAngle = (spinAngle + Math.min(65, now - lastSpinTime) * .0028) % (2 * Math.PI);
    lastSpinTime = now;
    for (const mesh of rollingMeshes) mesh.rotation.z = spinAngle;
    invalidate();
  }
  if (floorDirty && floor) { floor.update(); floorDirty = false; }
  if (renderer && camera) renderer.render(scene,camera);
}
document.addEventListener('visibilitychange', () => {
  if (intro) intro.previous = null;
  lastSpinTime = undefined;
  if (document.hidden) partInteraction?.stopGesture();
  else invalidate();
});
reducedMotion.addEventListener('change', () => {
  if (!reducedMotion.matches || !ready) return;
  if (intro) {
    intro = null;
    showExplosion(false);
  } else if (spreadMotion) {
    setExplosion(spreadMotion.to);
    spreadMotion = null;
    frameExterior();
    updateInspectionTools();
    invalidate();
  }
});
function invalidate() {
  if (!pending) { pending = true; requestAnimationFrame(draw); }
}
function geometryChanged() { floorDirty = true; invalidate(); }
function redrawOrbit() {
  camera.up.set(0,1,0);
  position.set(Math.cos(elevation)*Math.cos(yaw), Math.sin(elevation), Math.cos(elevation)*Math.sin(yaw)).multiplyScalar(radius);
  if (currentView === 'cabin') target.copy(camera.position).sub(position);
  else camera.position.copy(target).add(position);
  camera.lookAt(target);
  invalidate();
}
function orbitBy(dx, dy) {
  leaveCalibration();
  yaw += dx * .007;
  const inCabin = currentView === 'cabin';
  elevation = THREE.MathUtils.clamp(elevation + dy * .006, inCabin ? -1.23 : -.10, inCabin ? 1.23 : 1.48);
  redrawOrbit();
}
function zoomBy(scale) {
  leaveCalibration();
  if (currentView === 'cabin') {
    camera.fov = THREE.MathUtils.clamp(camera.fov * scale, 36, 98);
    camera.updateProjectionMatrix();
    invalidate();
  } else {
    radius = THREE.MathUtils.clamp(radius * scale, .20, Math.max(22, assemblyDistance * 1.2));
    redrawOrbit();
  }
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
function applyHinge(key, angle) {
  panelAngles[key] = angle;
  const [name, axis] = panelSpec[key];
  const node = pivots[name];
  node.quaternion.copy(original[name]);
  if (axis === 'z') node.rotateZ(angle);
  else node.rotateY(angle);
}
function setPanels(changes, angles = {}, animate = false) {
  for (const [key, open] of Object.entries(changes)) {
    const standard = panelSpec[key][2];
    const amount = angles[key] ?? (key.endsWith('Door') ? angles.doors : undefined) ?? Math.abs(standard);
    panels[key] = open;
    panelTargets[key] = open ? Math.sign(standard) * amount : 0;
  }
  if (animate && !capture && !reducedMotion.matches) hingeMotion = { start: performance.now(), from: {...panelAngles} };
  else {
    hingeMotion = null;
    for (const key of Object.keys(panelSpec)) applyHinge(key, panelTargets[key]);
  }
  for (const button of document.querySelectorAll('[data-panel]')) {
    const key = button.dataset.panel;
    const value = key === 'doors' ? (panels.leftDoor === panels.rightDoor ? panels.leftDoor : 'mixed') : panels[key];
    button.setAttribute('aria-pressed', String(value));
  }
  geometryChanged();
}
function togglePanel(key) {
  leaveCalibration();
  if (key === 'doors') {
    const open = !(panels.leftDoor && panels.rightDoor);
    setPanels({leftDoor:open, rightDoor:open}, {}, true);
  } else setPanels({[key]:!panels[key]}, {}, true);
}
function setWheels(steer, spin, resetAngle = false) {
  steering = steer;
  spinning = spin;
  lastSpinTime = undefined;
  for (const wheel of frontWheels) wheel.rotation.y = steer ? .48 : 0;
  if (resetAngle) {
    spinAngle = 0;
    for (const mesh of rollingMeshes) mesh.rotation.z = 0;
  }
  $('#steer').setAttribute('aria-pressed', String(steer));
  $('#spin').setAttribute('aria-pressed', String(spin));
  geometryChanged();
}
function clearMatch() {
  if (!matched) return;
  matched = null;
  overlay.hidden = true;
  $('#overlay-toggle').setAttribute('aria-pressed','false');
  $('#overlay-opacity').hidden = true;
  $('#photo-tools-3d').hidden = true;
  layout();
  updateInspectionTools();
}
function showGearboxCutaway(visible) {
  for (const {node} of assemblyParts) {
    node.visible = !visible || node.name === 'Transmission';
    if (node.name === 'Transmission') {
      for (const piece of node.children) {
        if (piece.userData.caseCover || piece.userData.name === 'Driveshaft joints and shift linkage') piece.visible = !visible;
      }
    }
  }
  $('#gearbox-caption').hidden = !visible;
  $('#bar').classList.toggle('in-gearbox', visible);
  floorDirty = true;
}
function setView(name) {
  if (!ready) return;
  intro = null;
  spreadMotion = null;
  setExplosion(0);
  clearMatch();
  currentView = name;
  showGearboxCutaway(name === 'gearbox');
  $('#bar').classList.toggle('in-cabin', name === 'cabin');
  $('#cabin-view').setAttribute('aria-pressed', String(name === 'cabin'));
  exteriorOrbit = name === 'front' || name === 'rear';
  if (name === 'front' || name === 'rear') {
    camera.fov = 32;
    frameExterior(name === 'rear');
  } else if (name === 'engine') {
    setPanels({hood:true}); target.set(-1.36,.55,.03);camera.position.set(-2.98,2.16,1.06);camera.fov=38;
  } else if (name === 'wheel') {
    target.set(-1.285,.30,.80);camera.position.set(-1.54,.49,2.10);camera.fov=35;
  } else if (name === 'gearbox') {
    target.set(-1.34,.405,-.397);
    const distance = gearboxRadius(camera.aspect);
    camera.position.set(-.83,.68,-.36).normalize().multiplyScalar(distance).add(target);
    camera.fov = 37;
  } else {
    target.set(-.48,.76,.29);camera.position.set(.25,1.08,.365);camera.fov=cabinFov(camera.aspect);
  }
  camera.up.set(0,1,0);camera.lookAt(target);camera.updateProjectionMatrix();rememberOrbit();layout();
}
function gallery() {return location.hash === '#integra-views';}
async function loadPhotos() {
  try {
    const response = await fetch('./views.json?v=41069a1d4d');
    if (!response.ok) throw new Error('Photo catalogue unavailable');
    const photos = await response.json();
    if (!Array.isArray(photos) || !photos.length) throw new Error('Empty photo catalogue');
    frames = photos;
    frames.forEach(frame => frameList.add(new Option(frame.title, frame.id)));
    const frameId = new URLSearchParams(location.search).get('frame');
    active = Math.max(0, frames.findIndex(frame => frame.id === frameId));
    if (gallery()) selectPhoto(active);
    else if (ready && frameId) match(active);
  } catch {
    $('#photo').hidden = true;
    $('#full-photo').hidden = true;
    const link = document.createElement('a');
    link.href = './frame-by-frame/'; link.textContent = 'Open the photo comparisons';
    $('#photo-error').replaceChildren('The reference gallery could not load. ', link);
    $('#photo-error').hidden = false;
  }
  for (const selector of ['#previous-view','#next-view','#photo-angle']) $(selector).disabled = !frames.length;
  $('#match-view').disabled = !ready || !frames.length;
}
function selectPhoto(index) {
  if (!frames.length) return;
  active = (index+frames.length)%frames.length;
  const frame=frames[active];
  frameList.value=frame.id;
  $('#photo').alt=frame.title+' · Flamenco Black Pearl 2000 Acura Integra GS-R';
  $('#photo-error').hidden=true;
  $('#photo').src=frame.url;
  $('#full-photo').href=frame.url;
  const credit = $('#photo-credit');
  credit.href = frame.source || 'https://bringatrailer.com/listing/2000-acura-integra-36/';
  credit.textContent = frame.credit || 'rmwcali · Bring a Trailer lot 263,334';
  $('#frame-count').textContent=(active+1)+' / '+frames.length+' · '+frame.width+' × '+frame.height;
}
function setURL(index,hash) {
  if (!frames.length) return;
  const url=new URL(location.href);
  url.searchParams.set('frame',frames[index].id);url.hash=hash;history.pushState(null,'',url);route();
}
function match(index) {
  if (!ready) return;
  const frame=frames[index];
  intro = null; spreadMotion = null;
  partInteraction?.reset();setWheels(false,false,true);setExplosion(0);
  showGearboxCutaway(false);
  clearMatch();matched=frame;currentView='photo';exteriorOrbit=false;$('#bar').classList.remove('in-cabin');
  $('#cabin-view').setAttribute('aria-pressed', 'false');
  setPanels({hood:['open','engine'].includes(frame.state),leftDoor:frame.state==='open',rightDoor:['open','cabin'].includes(frame.state),hatch:frame.state==='open'},frame.angles);
  camera.position.fromArray(frame.camera.position);target.fromArray(frame.camera.target);
  camera.up.fromArray(frame.camera.up);camera.lookAt(target);camera.fov=frame.camera.fov;
  rememberOrbit();layout();
  $('#match-label').textContent=frame.title;$('#photo-tools-3d').hidden=false;
  if (capture) { floor.update(); floorDirty = false; }
}
function route() {
  intro = null;
  spreadMotion = null;
  const url=new URL(location.href);
  lastRoute = url.href;
  active=Math.max(0,frames.findIndex(f=>f.id===url.searchParams.get('frame')));
  const photos=gallery();
  document.documentElement.classList.toggle('show-photos',photos);
  $('#photo-room').hidden=!photos;
  $('#model-link').setAttribute('aria-current',photos?'false':'page');
  $('#gallery-link').setAttribute('aria-current',photos?'page':'false');
  $('#more').open = false;
  if (photos) {
    partInteraction?.stopGesture();
    lastSpinTime = undefined;
    selectPhoto(active);
  }
  else if (url.searchParams.has('frame')) {
    firstModelVisit = false;
    if (frames.length) match(active);
  }
  else if (ready) {
    partInteraction.reset(); setWheels(false,false,true);
    setPanels({hood:false,leftDoor:false,rightDoor:false,hatch:false});
    const view = url.hash.slice(1);
    setView(['cabin','engine','wheel','rear','gearbox'].includes(view) ? view : 'front');
    if (url.hash === '#exploded') showExplosion(false);
    else if (!capture && (view === 'assembly' || (firstModelVisit && (!view || view === 'model')))) showExplosion(true);
    firstModelVisit = false;
  }
  updateInspectionTools();
}
function baseURL(hash = 'model') {
  const url=new URL(location.href);url.searchParams.delete('frame');url.hash=hash;
  if (url.href!==location.href) history.pushState(null,'',url);
  lastRoute = url.href;
}
function stopIntro() {
  if (!intro && !spreadMotion) return;
  intro = null;
  spreadMotion = null;
  if (explosion > 0) {
    const url = new URL(location.href); url.hash = 'exploded'; history.replaceState(null, '', url);
    lastRoute = url.href;
    updateInspectionTools();
  }
}
function leaveCalibration(assemble = false) {
  stopIntro();
  if (matched || (!frames.length && new URLSearchParams(location.search).has('frame'))) {baseURL();clearMatch();}
  if (assemble && (explosion || location.hash === '#exploded' || currentView === 'gearbox')) { baseURL(); setView('front'); }
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
      const endWall = Math.pow(Math.abs(Math.cos(az - .45)), 12) * smooth(lat, -.12, .22);
      const light=.25+.72*smooth(lat,.64,1.36)+high*(.34+2.45*bank)+low*(.15+.98*bank)+.82*endWall;
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
  const decoder = new THREE.DRACOLoader().setDecoderPath('../honda-civic-type-r/vendor/');
  const loader = new THREE.GLTFLoader().setDRACOLoader(decoder);
  return loader.loadAsync('./integra.glb?v=851e0b0ce1').finally(() => decoder.dispose());
}
$('#previous-view').onclick=()=>setURL((active+frames.length-1)%frames.length,'integra-views');
$('#next-view').onclick=()=>setURL((active+1)%frames.length,'integra-views');
frameList.onchange=()=>setURL(frameList.selectedIndex,'integra-views');
$('#match-view').onclick=()=>setURL(active,'model');
$('#photo').onerror=()=>{$('#photo-error').hidden=false;};
$('#gallery-link').onclick=event=>{if (frames.length) {event.preventDefault();setURL(active,'integra-views');}};
$('#model-link').onclick=event=>{event.preventDefault();baseURL();route();};
$('#overlay-toggle').onclick=()=>{
  if (!matched) return;
  const show=overlay.hidden;overlay.hidden=!show;
  if (show) overlay.src=matched.url;
  $('#overlay-toggle').setAttribute('aria-pressed',String(show));$('#overlay-opacity').hidden=!show;
};
$('#overlay-opacity').oninput=event=>{overlay.style.opacity=event.target.value/100;};
$('#reset').onclick=()=>{baseURL();route();$('#more').open=false;};
$('#cabin-view').onclick = () => { baseURL(currentView === 'cabin' ? 'model' : 'cabin'); route(); };
$('#more > summary').addEventListener('click', stopIntro);
$('#exploded-view').onclick = () => {
  const from = explosion;
  const open = spreadMotion ? spreadMotion.to === 0 : from === 0 && location.hash !== '#exploded';
  baseURL(open ? 'exploded' : 'model');
  route();
  if (!capture && !reducedMotion.matches && from !== explosion) {
    spreadMotion = {from, to:explosion, start:performance.now()};
    setExplosion(from);
    frameExterior();
    invalidate();
  }
};
$('#separation').oninput = event => {
  if (!ready) return;
  spreadMotion = null;
  const amount = Number(event.target.value) / 100;
  const aspect = Math.max(.4,room.clientWidth / room.clientHeight);
  const restRadius = exteriorRadius(aspect);
  const previousDistance = THREE.MathUtils.lerp(restRadius, assemblyDistance, explosion);
  const nextDistance = THREE.MathUtils.lerp(restRadius, assemblyDistance, amount);
  radius *= nextDistance / previousDistance;
  target.addScaledVector(position.copy(assemblyFocus).sub(frontCenter), amount - explosion);
  setExplosion(amount);
  redrawOrbit();
};
$('#replay-assembly').onclick = event => {
  event.preventDefault(); baseURL('assembly'); route();
};
document.querySelectorAll('[data-panel]').forEach(button => button.onclick = () => {
  leaveCalibration(true);
  togglePanel(button.dataset.panel);
});
document.querySelectorAll('[data-view]').forEach(button => button.onclick = () => {
  baseURL(button.dataset.view); route();
});
function leaveCutaway() {
  if (currentView === 'gearbox') { baseURL(); route(); }
}
$('#xray').onclick = () => { leaveCutaway(); partInteraction?.toggleXray(); };
$('#steer').onclick = () => { leaveCutaway(); leaveCalibration(); setWheels(!steering, spinning); };
$('#spin').onclick = () => { leaveCutaway(); leaveCalibration(); setWheels(steering, !spinning); };
function shortcuts(event) {
  if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey ||
      event.target.closest('input,select,textarea,[contenteditable="true"]')) return;
  if (gallery()) {
    if (frames.length && ['ArrowLeft','ArrowRight'].includes(event.key)) {
      event.preventDefault();
      setURL((active + frames.length + (event.key === 'ArrowLeft' ? -1 : 1)) % frames.length, 'integra-views');
    }
    return;
  }
  if (!ready) return;
  if (event.key === 'Escape') {
    if ($('#more').open) { $('#more').open = false; $('#more > summary').focus(); }
    partInteraction.stopGesture(); return;
  }
  const key = event.key.toLowerCase();
  const commands = { e: '#exploded-view', x: '#xray', r: '#reset', f: '[data-panel="hood"]',
    d: '[data-panel="doors"]', k: '[data-panel="hatch"]', '1': '[data-panel="leftDoor"]',
    '2': '[data-panel="rightDoor"]', s: '#steer', w: '#spin', i: '#cabin-view' };
  if (commands[key]) {
    event.preventDefault();
    if (!event.repeat) $(commands[key]).click();
    return;
  }
  if (event.target !== canvas) return;
  if (!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','=','-','Home'].includes(event.key)) return;
  event.preventDefault();
  if (event.key === 'Home') { $('#reset').click(); return; }
  if (event.key.startsWith('Arrow')) {
    const x = event.key === 'ArrowLeft' ? -15 : event.key === 'ArrowRight' ? 15 : 0;
    const y = event.key === 'ArrowUp' ? 13 : event.key === 'ArrowDown' ? -13 : 0;
    orbitBy(x,y);
  } else zoomBy(event.key === '-' ? 1 / .9 : .9);
}
window.addEventListener('keydown', shortcuts);
let photoSwipe;
const photoFigure = $('#photo-room figure');
photoFigure.addEventListener('pointerdown', event => {
  if (!frames.length || !event.isPrimary || event.button !== 0 || event.target.closest('a')) return;
  photoSwipe = {id:event.pointerId, x:event.clientX, y:event.clientY};
  photoFigure.setPointerCapture(event.pointerId);
});
photoFigure.addEventListener('pointerup', event => {
  if (event.pointerId !== photoSwipe?.id) return;
  const dx = event.clientX - photoSwipe.x, dy = event.clientY - photoSwipe.y;
  photoSwipe = null;
  if (Math.abs(dx) > Math.max(45, Math.abs(dy))) setURL((active + frames.length + (dx < 0 ? 1 : -1)) % frames.length, 'integra-views');
});
photoFigure.addEventListener('pointercancel', () => { photoSwipe = null; });
$('#photo').draggable = false;
document.addEventListener('pointerdown', event => {
  if ($('#more').open && !$('#more').contains(event.target)) $('#more').open = false;
});
function followHistory() { if (location.href !== lastRoute) route(); }
window.addEventListener('popstate', followHistory);
window.addEventListener('hashchange', followHistory);
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
  if (ready && lastAspect && currentView === 'gearbox') {
    radius *= gearboxRadius(aspect) / gearboxRadius(lastAspect);
    redrawOrbit();
  }
  if (ready && lastAspect && currentView === 'cabin') {
    camera.fov = THREE.MathUtils.clamp(camera.fov * cabinFov(aspect) / cabinFov(lastAspect), 36, 98);
  }
  lastAspect=aspect;layout();
});observer.observe(room);
new ResizeObserver(() => {
  document.documentElement.style.setProperty('--toolbar-height', (innerHeight - $('#bar').getBoundingClientRect().top) + 'px');
}).observe($('#bar'));
route();$('#match-view').disabled=true;
loadPhotos();
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
  for (const { node } of assemblyParts) {
    if (!node.name.startsWith('Wheel')) continue;
    if (node.name.startsWith('WheelFront')) frontWheels.push(node);
    // Each rolling mesh is authored in hub coordinates. The fixed brake casting stays with the upright.
    node.traverse(mesh => { if (mesh.isMesh && !/Brake[ _]casting/i.test(mesh.name)) rollingMeshes.push(mesh); });
  }
  partInteraction = bindPartInteraction(canvas, camera, model, assemblyParts, {
    orbit: orbitBy, zoom: zoomBy, stopIntro, leavePhoto: leaveCalibration, showCar: leaveCutaway,
    lookAround: () => currentView === 'cabin' || currentView === 'gearbox', panel: togglePanel, isPanelOpen: key => panels[key],
    render: invalidate, moved: geometryChanged, detached: count => { detachedCount = count; updateInspectionTools(); },
  });
  ready=true;floor.update();$('#loader').hidden=true;$('#bar').inert=false;$('#match-view').disabled=!frames.length;
  setView('front');route();renderer.render(scene,camera);
  document.documentElement.dataset.modelReady='true';
} catch (error) {
  console.error(error);
  $('#load-status').replaceChildren('3D is unavailable in this browser. ');
  const link=document.createElement('a');link.href='./frame-by-frame/';link.textContent='View the photo comparisons';$('#load-status').append(link);
  $('#bar').hidden=true;
}
