import {createAssemblyPicker,detailDirection} from './assembly.js?v=d85c6708';
import * as T from '/honda-civic-type-r/vendor/three-r170.js';
import {createStudioFloor} from '/honda-civic-type-r/studio-floor.js';
const $ = id => document.getElementById(id);
const vec = a => new T.Vector3(...a);
const canvas = $('stage');
const reduced = matchMedia('(prefers-reduced-motion: reduce)');
const params = new URLSearchParams(location.search);
let cfg, renderer, scene, camera, floor, model, exploded, assembled, matched, motion, cabin = false;
let assemblyPicker, focusedAssembly, explodedWide, explodedTall, portraitLayout;
let amount = 0, frame = 0, aspect = 1;
let inspection = null, inspectionPose = null, inspectionNav = null;
const hiddenMeshes = [];
const photoState = [];
function focusAssembly(name) {
  reset();
  setModelUrl(null, 'exploded', name);
  setSpread(1);
  assemblyPicker.select(name);
  if (name) {
    focusedAssembly=name;
    camera.fov=32;camera.updateProjectionMatrix();
    inspectionPose=measurePose(detailDirection(focusedAssembly));
    target.copy(inspectionPose.center);
    camera.position.copy(fit(inspectionPose));
    camera.lookAt(target);
  } else frameCamera();
  invalidate();
}
function clearInspection(){
  for(const object of hiddenMeshes)object.visible=true;
  hiddenMeshes.length=0;inspection=null;inspectionPose=null;
  const strip=$('inspection-strip');if(strip)strip.hidden=true;
  if(inspectionNav){inspectionNav.removeAttribute('aria-current');$('model-link').setAttribute('aria-current','page');}
  floor.visible=true;dirtyGround=true;
}
function inspect(id){
  const record=cfg.inspections?.find(v=>v.id===id);
  if(!record)return false;
  reset();inspection=record;
  const allowed=new Set(record.show),omit=new Set(record.hide||[]);
  model.traverse(o=>{
    if(!o.isMesh||!o.visible)return;
    let allowedAncestor=false,omitted=false;
    for(let parent=o;parent;parent=parent.parent){
      if(allowed.has(parent.name))allowedAncestor=true;
      if(omit.has(parent.name))omitted=true;
    }
    if(!allowedAncestor||omitted){hiddenMeshes.push(o);o.visible=false;}
  });
  inspectionPose=measurePose(record.direction);
  target.copy(inspectionPose.center);camera.position.copy(fit(inspectionPose));camera.lookAt(target);
  floor.visible=false;
  $('inspection-strip').hidden=false;$('inspection-choice').value=id;$('inspection-caption').textContent=record.note;
  $('inspection-source').href=record.source;
  inspectionNav.setAttribute('aria-current','page');$('model-link').removeAttribute('aria-current');
  $('panel').hidden=true;$('controls').setAttribute('aria-expanded','false');resize();return true;
}
function buildInspectionMenu(){
  if(!cfg.inspections?.length)return;
  const first = cfg.inspections[0];
  const label = cfg.inspectionLabel || 'Powertrain';
  inspectionNav = document.createElement('a');
  inspectionNav.id = 'inspection-link';
  inspectionNav.href = '#' + first.id;
  inspectionNav.textContent = label;
  $('review').before(inspectionNav);
  const menu=document.createElement('section');menu.id='inspection-strip';menu.hidden=true;
  menu.setAttribute('aria-label',label+' inspection');
  menu.innerHTML='<div><select id="inspection-choice" aria-label="Component view"></select> <a id="inspection-source">Technical reference</a></div><p id="inspection-caption"></p>';
  $('room').after(menu);
  for(const record of cfg.inspections){
    const option=document.createElement('option');option.value=record.id;option.textContent=record.label;
    $('inspection-choice').add(option);
    const button=document.createElement('button');button.textContent=record.label;
    button.onclick=()=>{if(location.hash==='#'+record.id)inspect(record.id);else location.hash=record.id;};
    $('actions').append(button);
  }
  $('inspection-choice').onchange=e=>{location.hash=e.target.value;};
  inspectionNav.onclick=e=>{if(location.hash===inspectionNav.hash){e.preventDefault();inspect(first.id);}};
}
const target = new T.Vector3(), parts = [], actions = [];
const pointers = new Map();
let updateGround,dirtyGround=true;
const pressed = (id,on) => $(id).setAttribute('aria-pressed',String(on));

function draw() {
  frame = 0;
  if (motion) {
    const t = T.MathUtils.clamp((performance.now()-motion.start)/motion.duration,0,1);
    const u = t*t*(3-2*t);
    setSpread(motion.from+(motion.to-motion.from)*u);
    frameCamera();
    if (t >= 1) motion = null;
    else invalidate();
  }
  if(dirtyGround && updateGround){updateGround();dirtyGround=false;}
  renderer.render(scene,camera);
}
function invalidate() { if (!frame) frame=requestAnimationFrame(draw); }
function setSpread(a) {
  amount=a;dirtyGround=true;
  floor.visible=a<.001 && !inspection && !matched?.noFloor;
  assemblyPicker?.setAmount(a);
  for (const p of parts) p.object.position.copy(p.original).addScaledVector(p.offset,a);
  pressed('explode',a>.001);
}
// Fit the actual corners of each part along the initial direction of view.
function measurePose(viewDirection) {
  const direction = viewDirection ? vec(viewDirection).normalize() : vec(cfg.home).sub(vec(cfg.target)).normalize();
  const right = vec([0,1,0]).cross(direction).normalize(), up=direction.clone().cross(right);
  const points = [];
  model.updateMatrixWorld(true);
  model.traverse(o => {
    if (!o.isMesh) return;
    for(let parent=o; parent; parent=parent.parent) if(!parent.visible) return;
    if (!viewDirection) {
      const vertices = o.geometry.attributes.position;
      const stride = Math.max(1, Math.floor(vertices.count / 96));
      for (let i = 0; i < vertices.count; i += stride)
        points.push(new T.Vector3().fromBufferAttribute(vertices,i).applyMatrix4(o.matrixWorld));
      return;
    }
    if (!o.geometry.boundingBox) o.geometry.computeBoundingBox();
    const {min,max}=o.geometry.boundingBox;
    for (const x of [min.x,max.x]) for (const y of [min.y,max.y]) for (const z of [min.z,max.z])
      points.push(new T.Vector3(x,y,z).applyMatrix4(o.matrixWorld));
  });
  const mids=[right,up,direction].map(axis=>{
    let lo=Infinity,hi=-Infinity;
    for(const p of points){const q=p.dot(axis);lo=Math.min(q,lo);hi=Math.max(q,hi);}
    return (lo+hi)/2;
  });
  const center=right.clone().multiplyScalar(mids[0]).addScaledVector(up,mids[1]).addScaledVector(direction,mids[2]);
  return {points,center,up,right,direction,fitAspect:0,eye:new T.Vector3()};
}
function fit(pose) {
  if (pose.fitAspect === aspect) return pose.eye;
  const {points,center,up,right,direction}=pose;
  const v = new T.Vector3();
  const tangent=Math.tan(T.MathUtils.degToRad(32)*.5);
  let d=0;
  for(const point of points){
    v.subVectors(point,center);
    const across=Math.max(Math.abs(v.dot(up)),Math.abs(v.dot(right))/aspect);
    d=Math.max(d,v.dot(direction)+1.09*across/tangent);
  }
  if (pose === assembled) {
    // Center the projected car with equal margins. A front bumper sits closer
    // to the camera than the rear quarter and occupies more of the frame.
    let cx = 0, cy = 0;
    for (let pass = 0; pass < 3; pass++) {
      let left = Infinity, rightEdge = -Infinity, bottom = Infinity, top = -Infinity;
      for (const point of points) {
        v.subVectors(point,center);
        const depth = d - v.dot(direction);
        const x = (v.dot(right) - cx) / depth;
        const y = (v.dot(up) - cy) / depth;
        left = Math.min(left,x); rightEdge = Math.max(rightEdge,x);
        bottom = Math.min(bottom,y); top = Math.max(top,y);
      }
      cx += .5 * (left + rightEdge) * d;
      cy += .5 * (bottom + top) * d;
      d = 0;
      for (const point of points) {
        v.subVectors(point,center);
        const span = Math.max(Math.abs(v.dot(up)-cy), Math.abs(v.dot(right)-cx)/aspect);
        d = Math.max(d, v.dot(direction) + 1.17*span/tangent);
      }
    }
    pose.framedCenter = center.clone().addScaledVector(right,cx).addScaledVector(up,cy);
  }
  pose.fitAspect=aspect;
  return pose.eye.copy(pose.framedCenter || center).addScaledVector(direction,d);
}
function setExteriorNear() {
  // A long-lens parts drawing places the eye far away. Keep enough depth
  // precision to resolve the paint skin from the liner a few millimetres below.
  camera.near=.012 + (focusedAssembly || inspection ? 0 : .05*amount*camera.position.distanceTo(target));
}
function frameCamera() {
  camera.position.copy(fit(assembled)).lerp(fit(exploded),amount);
  target.copy(assembled.framedCenter).lerp(exploded.center,amount);
  // Use a long lens for the parts drawing. Stepping the eye back by the same
  // factor keeps framing stable while removing perspective overlap at the edges.
  const dolly=1+5*amount;
  camera.position.sub(target).multiplyScalar(dolly).add(target);
  camera.fov=T.MathUtils.radToDeg(2*Math.atan(Math.tan(T.MathUtils.degToRad(16))/dolly));
  setExteriorNear();
  camera.updateProjectionMatrix();
  camera.lookAt(target);
}
function resize() {
  const w=canvas.clientWidth, h=canvas.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w,h,false);
  aspect=w/h;
  const portrait=aspect<.87;
  if (explodedTall && portrait!==portraitLayout) {
    portraitLayout=portrait;
    exploded=portrait ? explodedTall : explodedWide;
    for(const part of parts) part.offset=portrait ? part.tall : part.wide;
    setSpread(amount);
    if(focusedAssembly) {
      const oldCenter=inspectionPose.center;
      inspectionPose=measurePose(detailDirection(focusedAssembly));
      if(userOrbit) {
        const delta=inspectionPose.center.clone().sub(oldCenter);
        target.add(delta);
        camera.position.add(delta);
      }
    } else if(amount===1 && !inspection) userOrbit=false;
  }
  applyViewport(w,h);
  if((inspection || focusedAssembly) && !userOrbit){target.copy(inspectionPose.center);camera.position.copy(fit(inspectionPose));camera.lookAt(target);}
  else if(model && !cabin && !matched && !userOrbit) frameCamera();
  invalidate();
}
let photoCrop = null;
function applyViewport(w,h) {
  const ar = matched && (matched.aspect || (matched.imageSize && matched.imageSize[0]/matched.imageSize[1]) || (matched.width/matched.height));
  let vw=w,vh=h;
  if(ar>0){vw=Math.min(w,h*ar);vh=vw/ar;}
  camera.aspect=vw/vh;
  camera.updateProjectionMatrix();
  if(matched?.principalPoint && matched.imageSize){
    const [iw,ih]=matched.imageSize, [cx,cy]=matched.principalPoint;
    camera.projectionMatrix.elements[8]=1-2*cx/iw;
    camera.projectionMatrix.elements[9]=2*cy/ih-1;
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
  }
  if(photoCrop && matched) {
    // Crop the projected image after the camera intrinsics, including its principal point.
    const {zoom,left,top}=photoCrop;
    const matrix=camera.projectionMatrix.elements;
    matrix[0]*=zoom;matrix[5]*=zoom;
    matrix[8]=zoom*matrix[8]-(zoom-1-2*zoom*left);
    matrix[9]=zoom*matrix[9]-(1-zoom+2*zoom*top);
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
  }
  renderer.setViewport((w-vw)/2,(h-vh)/2,vw,vh);
}
function stop() { motion=null; }
let userOrbit=false;
function unmatch() {
  for (const {object, rotation, visible} of photoState.splice(0)) {
    object.rotation.copy(rotation);
    object.visible = visible;
  }
  matched=null;camera.clearViewOffset();camera.up.set(0,1,0);
  $('angles').value='';$('ref-line').hidden=true;applyViewport(canvas.clientWidth,canvas.clientHeight);
}
function clearActions() {
  for(const a of actions){
    for(const b of a.nodes){b.object.rotation.copy(b.original);b.object.visible=b.visible;}
    a.active=false;a.button.setAttribute('aria-pressed','false');
  }
}
function reset() {
  assemblyPicker?.clear();
  stop();if(inspection)clearInspection();
  if(focusedAssembly){assemblyPicker.select('');focusedAssembly=null;inspectionPose=null;}
  unmatch();cabin=false;userOrbit=false;pressed('cabin',false);
  camera.fov=32;camera.updateProjectionMatrix();setSpread(0);clearActions();frameCamera();invalidate();
}
function match(id){
  const v=cfg.views.find(record=>String(record.id)===String(id));
  if(!v)return false;
  reset();matched=v;userOrbit=true;
  if(v.actions){matched=null;for(const a of actions)if(v.actions.includes(a.label))a.button.click();matched=v;}
  // A calibrated detail view includes the photographed panel position and any occluders.
  const remember = object => photoState.push({object, rotation:object.rotation.clone(), visible:object.visible});
  for (const [name, degrees] of Object.entries(v.rotate || {})) {
    const object = model.getObjectByName(name);
    if (!object) continue;
    remember(object);
    object.rotation.set(...degrees.map(T.MathUtils.degToRad));
  }
  if (v.only?.length) {
    const allowed = new Set(v.only);
    model.traverse(object => {
      if (!object.isMesh) return;
      for (let parent=object; parent; parent=parent.parent) if (allowed.has(parent.name)) return;
      remember(object);
      object.visible=false;
    });
  }
  for (const name of v.hide || []) {
    const object = model.getObjectByName(name);
    if (!object) continue;
    remember(object);
    object.visible=false;
  }
  if(v.noFloor){remember(floor);floor.visible=false;}
  camera.fov=v.fov||32;camera.position.copy(vec(v.position));target.copy(vec(v.target));
  camera.up.copy(vec(v.up||[0,1,0]));camera.lookAt(target);
  if(v.shiftX || v.shiftY) {
    const w=v.width||1920,h=v.height||1080;
    camera.setViewOffset(w,h,v.shiftX||0,v.shiftY||0,w,h);
  }
  applyViewport(canvas.clientWidth,canvas.clientHeight);
  $('angles').value=v.id;
  $('source').href=v.source||cfg.review;$('ref-line').hidden=!v.source;invalidate();return true;
}
function setModelUrl(id, hash = 'model', component) {
  const url = new URL(location.href);
  url.searchParams.delete('frame');
  url.searchParams.delete('view');
  url.searchParams.delete('part');
  if(component) url.searchParams.set('part',component);
  if (id) url.searchParams.set('frame', id);
  url.hash = hash;
  if (url.href !== location.href) history.replaceState(null, '', url);
}
function resetView() {
  reset();
  setModelUrl();
}
function toggleSpread(){
  const dest=motion ? 1-motion.to : (amount>.5?0:1);
  setModelUrl(null, dest ? 'exploded' : 'model');
  stop(); if(cabin||matched||inspection||focusedAssembly)reset();userOrbit=false;
  clearActions();
  if(reduced.matches){setSpread(dest);frameCamera();}
  else motion={from:amount,to:dest,start:performance.now(),duration:700};
  invalidate();
}
function toggleCabin(){
  setModelUrl();
  if(cabin){reset();return;}
  reset();cabin=true;pressed('cabin',true);camera.fov=72;camera.updateProjectionMatrix();
  camera.position.copy(vec(cfg.eye));target.copy(vec(cfg.look));camera.lookAt(target);invalidate();
}
function interact(){
  stop();if(matched){unmatch();setModelUrl();}userOrbit=true;
}
function orbit(dx,dy){
  interact();
  const v=cabin ? target.clone().sub(camera.position) : camera.position.clone().sub(target);
  const r=v.length(), yaw=Math.atan2(v.x,v.z)+(cabin?1:-1)*dx*.007;
  const pitch=T.MathUtils.clamp(Math.asin(v.y/r)+dy*.006,cabin||inspection||amount>.01?-1.45:.025,1.48);
  v.set(Math.sin(yaw)*Math.cos(pitch),Math.sin(pitch),Math.cos(yaw)*Math.cos(pitch)).multiplyScalar(r);
  if(cabin)target.copy(camera.position).add(v);else camera.position.copy(target).add(v);
  camera.lookAt(target);invalidate();
}
function zoom(factor){
  interact();
  if(cabin){camera.fov=T.MathUtils.clamp(camera.fov*factor,37,105);camera.updateProjectionMatrix();}
  else{
    const d=camera.position.clone().sub(target);
    const distance=T.MathUtils.clamp(d.length()*factor,.35,focusedAssembly ? 65 : 65*(1+5*amount));
    camera.position.copy(target).addScaledVector(d.normalize(),distance);
    setExteriorNear();camera.updateProjectionMatrix();
  }
  invalidate();
}
function input(){
  canvas.onpointerdown=e=>{interact();canvas.setPointerCapture(e.pointerId);pointers.set(e.pointerId,[e.clientX,e.clientY]);};
  canvas.onpointermove=e=>{
    if(!pointers.has(e.pointerId))return;
    const old=pointers.get(e.pointerId), now=[e.clientX,e.clientY];pointers.set(e.pointerId,now);
    if(pointers.size===1)orbit(now[0]-old[0],now[1]-old[1]);
    if(pointers.size===2){
      const other=[...pointers].find(([k])=>k!==e.pointerId)[1];
      const distance=p=>Math.hypot(p[0]-other[0],p[1]-other[1]);
      if(distance(now)>1 && distance(old)>1)zoom(distance(old)/distance(now));
    }
  };
  canvas.onlostpointercapture=e=>pointers.delete(e.pointerId);
  canvas.onwheel=e=>{e.preventDefault();zoom(Math.exp(T.MathUtils.clamp(e.deltaY,-500,500)*.001));};
  canvas.onkeydown=e=>{
    const key=e.key.toLowerCase();
    const op={arrowleft:()=>orbit(8,0),arrowright:()=>orbit(-8,0),arrowup:()=>orbit(0,8),arrowdown:()=>orbit(0,-8),'+':()=>zoom(.9),'=':()=>zoom(.9),'-':()=>zoom(1.1),r:resetView};
    if(op[key]){e.preventDefault();op[key]();}
  };
  $('explode').onclick=toggleSpread;$('cabin').onclick=toggleCabin;
  $('reset').onclick=resetView;
  $('controls').onclick=()=>{stop();$('panel').hidden=!$('panel').hidden;$('controls').setAttribute('aria-expanded',String(!$('panel').hidden));};
  document.addEventListener('keydown',e=>{if(e.key==='Escape'&&!$('panel').hidden){$('controls').click();$('controls').focus();}});
  $('angles').onchange=e=>{
    if (e.target.value) { match(e.target.value); setModelUrl(e.target.value); }
    else resetView();
  };
  $('model-link').onclick=e=>{
    if (location.hash==='#honda-views') {
      e.preventDefault();
      const url=new URL(location.href);
      url.searchParams.delete('frame');url.searchParams.delete('compare');
      url.hash='model';location.assign(url.href);
    } else {e.preventDefault();resetView();}
  };
  window.addEventListener('hashchange',()=>{
    if(location.hash==='#exploded') openExplodedLink();
    else if(!inspect(location.hash.slice(1)) && location.hash==='#model') {
      if(!match(new URLSearchParams(location.search).get('frame'))) reset();
      $('review').removeAttribute('aria-current');
      $('model-link').setAttribute('aria-current','page');
    }
  });
}
function openExplodedLink() {
  const requested = new URL(location.href).searchParams.get('part');
  focusAssembly(cfg.parts.some(part => part.node === requested) ? requested : '');
}
async function init(){
  cfg=await (await fetch('studio.json?v=d85c6708')).json();
  $('spec').textContent=cfg.subtitle;
  renderer=new T.WebGLRenderer({canvas,antialias:true,alpha:true,preserveDrawingBuffer:params.has('capture')});
  renderer.setPixelRatio(params.has('capture')?1:Math.min(devicePixelRatio,2));
  renderer.toneMapping=T.ACESFilmicToneMapping;renderer.toneMappingExposure=1.03;
  scene=new T.Scene();scene.background=new T.Color(0xe8ebed);
  // Studio Small 09 by Sergej Majboroda (Poly Haven, CC0). The texture
  // encodes linear radiance divided by 32 before conversion to sRGB.
  const studio = await new T.TextureLoader().loadAsync('studio-light.webp');
  studio.colorSpace = T.SRGBColorSpace;
  const pmrem = new T.PMREMGenerator(renderer);
  scene.environment = pmrem.fromEquirectangular(studio).texture;
  studio.dispose();
  pmrem.dispose();
  scene.environmentIntensity = 32 * .73;
  scene.environmentRotation.y = -1.03;
  scene.add(new T.HemisphereLight(0xf4f7ff, 0x606066, .83));
  const key = new T.DirectionalLight(0xfff6eb, 1.15);
  key.position.set(-3,6,3);
  scene.add(key);
  renderer.shadowMap.enabled=true;renderer.shadowMap.type=T.PCFSoftShadowMap;
  key.castShadow=true;key.shadow.mapSize.set(1024,1024);Object.assign(key.shadow.camera,{left:-5,right:5,top:5,bottom:-5,near:.1,far:22});key.shadow.camera.updateProjectionMatrix();key.shadow.normalBias=.008;
  const fill = new T.DirectionalLight(0xebf1ff, .52);
  fill.position.set(-5,2,-4);
  scene.add(fill);
  const ground=createStudioFloor(scene,renderer,0xe8ebed);
  floor=ground.floor;
  camera=new T.PerspectiveCamera(32,1,.012,1200);
  const draco=new T.DRACOLoader().setDecoderPath('/honda-civic-type-r/vendor/');
  const gltf=await new T.GLTFLoader().setDRACOLoader(draco).loadAsync(cfg.model).finally(()=>draco.dispose());
  model=gltf.scene;scene.add(model);model.traverse(o=>{if(o.isMesh){o.castShadow=!o.material.transparent;if(o.castShadow)ground.addCaster(o);}});
  for(const p of cfg.parts){const object=model.getObjectByName(p.node);if(!object)throw Error('Missing assembly '+p.node);const wide=vec(p.offset),tall=vec(p.portraitOffset||p.offset);parts.push({object,original:object.position.clone(),wide,tall,offset:wide});}
  for(const a of cfg.actions||[]){
    const button=document.createElement('button');button.textContent=a.label;button.setAttribute('aria-pressed','false');$('actions').append(button);
    const nodes=a.nodes.map(n=>{const object=model.getObjectByName(n.node);if(!object)throw Error('Missing pivot '+n.node);return {...n,object,original:object.rotation.clone(),visible:object.visible};});
    const state={...a,nodes,button,active:false};actions.push(state);
    button.onclick=()=>{stop();if(amount || matched || inspection)resetView();state.active=!state.active;
      for(const n of nodes){
        if(n.hidden)n.object.visible=state.active?false:n.visible;
        else n.object.rotation[n.axis]=n.original[n.axis]+(state.active?T.MathUtils.degToRad(n.degrees):0);
      }
      button.setAttribute('aria-pressed',String(state.active));dirtyGround=true;invalidate();};
  }
  for(const v of cfg.views||[]){const option=document.createElement('option');option.value=v.id;option.textContent=v.label;$('angles').add(option);}
  buildInspectionMenu();
  assemblyPicker=createAssemblyPicker(T,model,cfg,canvas,camera,focusAssembly,{
    invalidate, stop,
    open(name) {
      const action=actions.find(a=>a.nodes.some(n=>n.object.name===name && n.axis));
      if (!action || inspection || matched || cabin) return;
      const n=action.nodes.find(n=>n.object.name===name);
      const opening=Math.abs(n.object.rotation[n.axis]-n.original[n.axis]) < .001;
      n.object.rotation[n.axis]=n.original[n.axis]+(opening?T.MathUtils.degToRad(n.degrees):0);
      action.active=action.nodes.every(b=>Math.abs(b.object.rotation[b.axis]-b.original[b.axis])>.001);
      action.button.setAttribute('aria-pressed',String(action.active));
      dirtyGround=true;invalidate();
    },
  });
  assembled=measurePose();
  setSpread(1);explodedWide=measurePose(cfg.explodedDirection);
  for(const part of parts) part.offset=part.tall;
  setSpread(1);explodedTall=measurePose(cfg.explodedDirection);
  resize();input();
  ground.update();updateGround=ground.update;
  const componentEntry=cfg.inspections?.some(item=>item.id===location.hash.slice(1));
  if(location.hash==='#honda-views')reset();
  else if(location.hash==='#exploded')openExplodedLink();
  else if(componentEntry)inspect(location.hash.slice(1));
  else if(!match(params.get('frame')||params.get('view')||'')){
    if(params.has('capture'))reset();
    else if(!reduced.matches)motion={from:1,to:0,start:performance.now()+1200,duration:2100};
  }
  $('status').hidden=true;$('tools').inert=false;new ResizeObserver(resize).observe(canvas);draw();return true;
}
let normalMaterial;
function capture(width,height,transparent=false,normals=false){
  stop();
  const oldRatio=renderer.getPixelRatio();renderer.setPixelRatio(1);renderer.setSize(width,height,false);
  aspect=width/height; applyViewport(width,height);
  if(!matched && !inspection && !focusedAssembly && !cabin && !userOrbit) frameCamera();
  const bg=scene.background, floorVisible=floor.visible;
  if(transparent){scene.background=null;floor.visible=false;renderer.setClearColor(0,0);}
  const previousMaterial=scene.overrideMaterial;
  if(normals) scene.overrideMaterial=normalMaterial ??= new T.ShaderMaterial({
    side:T.DoubleSide,
    vertexShader:`varying vec3 surfaceNormal;
      void main() {
        surfaceNormal=normalMatrix*normal;
        gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);
      }`,
    fragmentShader:`varying vec3 surfaceNormal;
      void main() { gl_FragColor=vec4(normalize(surfaceNormal)*0.5+0.5,1.0); }`
  });
  renderer.render(scene,camera);const data=canvas.toDataURL('image/png');
  scene.overrideMaterial=previousMaterial;
  scene.background=bg;floor.visible=floorVisible;renderer.setPixelRatio(oldRatio);resize();return data;
}
function setPhotoCrop(crop) {
  photoCrop=crop;
  if(model && camera && canvas.clientWidth && canvas.clientHeight) {
    applyViewport(canvas.clientWidth,canvas.clientHeight);
    invalidate();
  }
}
window.carStudio={match,inspect,reset,capture,setPhotoCrop,get pose(){return {spread:amount,cabin,inspection:inspection?.id,component:focusedAssembly,matched:matched?.id,position:camera?.position.toArray(),target:target.toArray()};},ready:init().catch(e=>{
  console.error(e);
  const poster=document.createElement('img');
  poster.className='fallback-poster';
  poster.alt=cfg?.title || document.querySelector('h1').textContent;
  poster.src=cfg?.poster || 'poster.webp';
  canvas.after(poster);
  $('status').textContent='The interactive model could not load. Open Honda views to see the reference photographs.';
  throw e;
})};
