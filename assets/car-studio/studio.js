import * as T from '/honda-civic-type-r/vendor/three-r170.js';
import {createStudioFloor} from '/honda-civic-type-r/studio-floor.js';
const $ = id => document.getElementById(id);
const vec = a => new T.Vector3(...a);
const canvas = $('stage');
const reduced = matchMedia('(prefers-reduced-motion: reduce)');
const params = new URLSearchParams(location.search);
function studioAsset(path) {
  const version = new URL(import.meta.url).searchParams.get('v');
  const url = new URL(path, document.baseURI);
  if (version) url.searchParams.set('v', version);
  return url.href;
}
let cfg, renderer, scene, camera, floor, model, exploded, assembled, matched, motion, cabin = false;
let assemblyStudy, assemblyAspect;
let amount = 0, frame = 0, aspect = 1, introTimer = 0;
let inspection = null, inspectionPose = null, inspectionNav = null;
const hiddenMeshes = [];
let drawing, drawingFocus, portraitOffsets;
function holdDrawing() {
  if (motion?.intro || amount === 1) { stop(); setModelUrl(null, 'exploded'); }
}
function focusDrawing(selected, direction) {
  userOrbit=false;
  drawingFocus=selected ? measurePose(direction || cfg.assemblyDrawing.direction) : null;
  frameCamera();invalidate();
}
function enterDrawing() {
  reset();setSpread(1);frameCamera();invalidate();
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
  if (motion && performance.now() < motion.start && (assemblyStudy || drawing)) {
    if (!introTimer) introTimer=setTimeout(()=>{introTimer=0;invalidate();}, motion.start-performance.now());
  } else if (motion) {
    const t = T.MathUtils.clamp((performance.now()-motion.start)/motion.duration,0,1);
    const u = t*t*(3-2*t);
    setSpread(motion.from+(motion.to-motion.from)*u);
    frameCamera();
    if (t >= 1) motion = null;
    else invalidate();
  }
  if(dirtyGround && updateGround && (!(assemblyStudy || drawing) || floor.visible)){updateGround();dirtyGround=false;}
  renderer.render(scene,camera);
}
function invalidate() { if (!frame) frame=requestAnimationFrame(draw); }
function setSpread(a) {
  amount=a;dirtyGround=true;
  if(assemblyStudy){assemblyStudy.apply(a);floor.visible=a<.001 && !inspection;}
  for (const p of parts) p.object.position.copy(p.original).addScaledVector(p.offset,a);
  if(drawing) {
    drawing.setActive(a >= .999);
    floor.visible = !inspection && a < .012;
  }
  pressed('explode',a>.001);
}
// Fit the actual corners of each part along the initial direction of view.
function measurePose(viewDirection, wholeDrawing = false) {
  const direction = viewDirection ? vec(viewDirection).normalize() : vec(cfg.home).sub(vec(cfg.target)).normalize();
  const right = vec([0,1,0]).cross(direction).normalize(), up=direction.clone().cross(right);
  const points = [];
  model.updateMatrixWorld(true);
  if (drawing && (wholeDrawing || (!inspection && !drawing.separated))) {
    for (const part of parts) {
      if (!part.object.visible) continue;
      const delta=part.object.position.clone().sub(part.original);
      for (const corner of part.hull) points.push(corner.clone().add(delta));
    }
  } else model.traverseVisible(o => {
    if (!o.isMesh) return;
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
  const tangent=Math.tan(T.MathUtils.degToRad(cfg.exteriorFov || 32)*.5);
  let d=0;
  for(const point of points){
    v.subVectors(point,center);
    const across=Math.max(Math.abs(v.dot(up)),Math.abs(v.dot(right))/aspect);
    d=Math.max(d,v.dot(direction)+1.09*across/tangent);
  }
  pose.fitAspect=aspect;
  return pose.eye.copy(center).addScaledVector(direction,d);
}
function frameCamera() {
  if(drawingFocus) {
    target.copy(drawingFocus.center);camera.position.copy(fit(drawingFocus));camera.lookAt(target);return;
  }
  target.copy(assembled.center).lerp(exploded.center,amount);
  camera.position.copy(fit(assembled)).lerp(fit(exploded),amount);
  if (assemblyStudy) {
    // A longer lens limits perspective overlap between separated components.
    // Match its distance to the original framing throughout the animation.
    const fov = 32 - amount * 20;
    const distanceScale = Math.tan(T.MathUtils.degToRad(16)) / Math.tan(T.MathUtils.degToRad(fov / 2));
    camera.position.sub(target).multiplyScalar(distanceScale).add(target);
    camera.fov=fov; camera.updateProjectionMatrix();
  }
  camera.lookAt(target);
}
function layoutAssembly() {
  const spread=amount;
  assemblyAspect=aspect;
  assemblyStudy.layout(aspect); setSpread(1);
  exploded=measurePose(assemblyStudy.direction); setSpread(spread);
}
function selectAssembly(group, piece) {
  stop(); userOrbit=false;
  assemblyStudy.select(group, piece); setModelUrl(null,'exploded');
  camera.near = piece ? .0001 : .012; camera.updateProjectionMatrix();
  layoutAssembly(); setSpread(1); frameCamera(); invalidate();
}
function resize() {
  const w=canvas.clientWidth, h=canvas.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w,h,false);
  aspect=w/h;
  if (drawing && portraitOffsets !== (aspect < cfg.assemblyDrawing.narrowAspect)) {
    portraitOffsets=aspect < cfg.assemblyDrawing.narrowAspect;
    for (const part of parts) part.offset.copy(portraitOffsets ? part.narrow : part.wide);
    const saved=amount, visibility=parts.map(p=>p.object.visible);
    for(const part of parts)part.object.visible=true;
    // Measure the full drawing even when a single part is being inspected.
    for(const part of parts)part.object.position.copy(part.original).add(part.offset);
    exploded=measurePose(cfg.assemblyDrawing.direction, true);
    parts.forEach((part,i)=>{part.object.visible=visibility[i];part.object.position.copy(part.original).addScaledVector(part.offset,saved);});
    if(drawingFocus)drawingFocus=measurePose(drawingFocus.direction.toArray());
  }
  if (assemblyStudy && !inspection && Math.abs(assemblyAspect-aspect) > .02) {
    layoutAssembly();
    if (amount && !cabin && !matched) userOrbit=false;
  }
  applyViewport(w,h);
  if(inspection && !userOrbit){target.copy(inspectionPose.center);camera.position.copy(fit(inspectionPose));camera.lookAt(target);}
  else if(model && !cabin && !matched && !userOrbit) frameCamera();
  invalidate();
}
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
  renderer.setViewport((w-vw)/2,(h-vh)/2,vw,vh);
}
function stop() { motion=null; clearTimeout(introTimer); introTimer=0; }
let userOrbit=false;
function unmatch() {
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
  stop();if(inspection)clearInspection();
  if(assemblyStudy?.detailed){assemblyStudy.select('','');layoutAssembly();}
  if(drawing?.selected)drawing.clear();
  drawingFocus=null;
  unmatch();cabin=false;userOrbit=false;pressed('cabin',false);
  camera.near=.012;camera.fov=cfg.exteriorFov || 32;camera.updateProjectionMatrix();setSpread(0);clearActions();frameCamera();invalidate();
}
function match(id){
  const v=cfg.views.find(record=>String(record.id)===String(id));
  if(!v)return false;
  reset();matched=v;userOrbit=true;
  if(v.actions){matched=null;for(const a of actions)if(v.actions.includes(a.label))a.button.click();matched=v;}
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
function setModelUrl(id, hash = 'model') {
  const url = new URL(location.href);
  url.searchParams.delete('frame');
  url.searchParams.delete('view');
  if (id) url.searchParams.set('frame', id);
  url.hash = hash;
  if (url.href !== location.href) history.replaceState(null, '', url);
}
function resetView() {
  reset();
  setModelUrl();
}
function toggleSpread(){
  let dest = motion && !motion.intro ? 1-motion.to : (amount>.5?0:1);
  if(drawing && motion?.intro)dest=1;
  if(drawing?.selected)drawing.clear();
  drawingFocus=null;
  if(assemblyStudy?.detailed)selectAssembly('','');
  setModelUrl(null, (assemblyStudy || drawing) && dest ? 'exploded' : 'model');
  stop(); if(cabin||matched||inspection)reset();userOrbit=false;
  clearActions();
  if(reduced.matches){setSpread(dest);frameCamera();}
  else motion={from:amount,to:dest,start:performance.now(),duration:assemblyStudy?1200:700};
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
  const pitch=T.MathUtils.clamp(Math.asin(v.y/r)+dy*.006,cabin||inspection||((assemblyStudy || drawing) && amount>.001)?-1.45:.025,1.48);
  v.set(Math.sin(yaw)*Math.cos(pitch),Math.sin(pitch),Math.cos(yaw)*Math.cos(pitch)).multiplyScalar(r);
  if(cabin)target.copy(camera.position).add(v);else camera.position.copy(target).add(v);
  camera.lookAt(target);invalidate();
}
function zoom(factor){
  interact();
  if(cabin){camera.fov=T.MathUtils.clamp(camera.fov*factor,37,105);camera.updateProjectionMatrix();}
  else{
    const d=camera.position.clone().sub(target);
    const distance=T.MathUtils.clamp(d.length()*factor,assemblyStudy?.detailed ? .0004 : drawingFocus ? .035 : .35,(assemblyStudy || drawing) && amount ? 260 : 65);
    camera.position.copy(target).addScaledVector(d.normalize(),distance);
  }
  invalidate();
}
function input(){
  const assemblyPicker=$('assembly-picker');
  if(assemblyPicker){
    for(const event of ['focusin','pointerdown']) assemblyPicker.addEventListener(event,()=>{if(motion?.intro)stop();});
  }
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
  $('model-link').onclick=e=>{e.preventDefault();resetView();};
  window.addEventListener('hashchange',()=>{if(drawing && location.hash==='#exploded')enterDrawing();
    else if(assemblyStudy && location.hash==='#exploded'){reset();selectAssembly('','');}
    else if(!inspect(location.hash.slice(1)) && location.hash==='#model')reset();});
}
async function init(){
  cfg=await (await fetch(studioAsset('studio.json'))).json();
  $('review').href=cfg.review;$('spec').textContent=cfg.subtitle;
  renderer=new T.WebGLRenderer({canvas,antialias:true,alpha:true,preserveDrawingBuffer:params.has('capture')});
  renderer.setPixelRatio(params.has('capture')?1:Math.min(devicePixelRatio,2));
  renderer.toneMapping=T.ACESFilmicToneMapping;renderer.toneMappingExposure=1.03;
  scene=new T.Scene();scene.background=new T.Color(0xe8ebed);
  // Long reflection banks follow the existing car studios and describe the crowned sheet metal.
  const assemblyModule=document.querySelector('script[data-assembly]')?.dataset.assembly;
  const studioExtension=assemblyModule ? await import(studioAsset(assemblyModule)) : null;
  const lighting=studioExtension?.lighting;
  if (cfg.lighting) {
    const {studioEnvironment}=await import(studioAsset('studio-light.js'));
    scene.environment=await studioEnvironment(renderer,studioAsset(cfg.lighting.image));
    scene.environmentRotation.y=cfg.lighting.rotation;
  } else {
    const pmrem=new T.PMREMGenerator(renderer);
    const panorama=document.createElement('canvas');panorama.width=512;panorama.height=256;
    const context=panorama.getContext('2d');const pixels=context.createImageData(512,256);
    const smooth=T.MathUtils.smoothstep;
    for (let y=0;y<256;y++) {
      const lat=(.5-(y+.5)/256)*Math.PI;
      const high=smooth(lat,.24,.40)*(1-smooth(lat,.59,.83));
      const low=smooth(lat,-.18,-.06)*(1-smooth(lat,.035,.15));
      for (let x=0;x<512;x++) {
        const az=((x+.5)/512-.5)*2*Math.PI;
        const bank=smooth(Math.abs(Math.sin(az+.20)),.27,.85);
        const endWall = Math.pow(Math.abs(Math.cos(az - .45)), 12) * smooth(lat, -.12, .22);
        const light=.25+.72*smooth(lat,.64,1.36)+high*(.34+2.45*bank)+low*(.15+.98*bank)+.82*endWall;
        const k=4*(y*512+x);
        if(lighting){
          const intensity=lighting.strip(az,lat,smooth);
          pixels.data[k]=intensity;pixels.data[k+1]=intensity*.99;pixels.data[k+2]=intensity;
        }else{pixels.data[k]=light*64;pixels.data[k+1]=light*66;pixels.data[k+2]=light*69;}
        pixels.data[k+3]=255;
      }
    }
    context.putImageData(pixels,0,0);const studio=new T.CanvasTexture(panorama);
    scene.environment=pmrem.fromEquirectangular(studio).texture;studio.dispose();pmrem.dispose();
  }
  scene.environmentIntensity=cfg.lighting?.intensity ?? lighting?.environment ?? 1.38;
  scene.add(new T.HemisphereLight(0xf2f6fd,0x53575e,cfg.lighting?.ambient ?? lighting?.hemisphere ?? 1.13));
  const key=new T.DirectionalLight(0xfff8ef,cfg.lighting?.key ?? lighting?.key ?? 1.43);key.position.set(...(lighting?.keyPosition ?? [-3,6,3]));scene.add(key);
  renderer.shadowMap.enabled=true;renderer.shadowMap.type=T.PCFSoftShadowMap;
  key.castShadow=true;key.shadow.mapSize.set(1024,1024);Object.assign(key.shadow.camera,{left:-5,right:5,top:5,bottom:-5,near:.1,far:22});key.shadow.camera.updateProjectionMatrix();key.shadow.normalBias=.008;
  const fill=new T.DirectionalLight(0xeef4ff,cfg.lighting?.fill ?? lighting?.fill ?? 1.1);fill.position.set(-5,2,-4);scene.add(fill);
  const ground=createStudioFloor(scene,renderer,0xe8ebed);
  floor=ground.floor;
  camera=new T.PerspectiveCamera(cfg.exteriorFov || 32,1,.012,300);
  const draco=new T.DRACOLoader().setDecoderPath('/honda-civic-type-r/vendor/');
  const gltf=await new T.GLTFLoader().setDRACOLoader(draco).loadAsync(studioAsset(cfg.model)).finally(()=>draco.dispose());
  model=gltf.scene;scene.add(model);model.traverse(o=>{if(o.isMesh){o.castShadow=!o.material.transparent;if(o.castShadow)ground.addCaster(o);}});
  if(studioExtension){
    const {createAssemblyStudy}=studioExtension;
    assemblyStudy=createAssemblyStudy(T,gltf,camera,selectAssembly);
  }
  else {
    const hulls=cfg.assemblyDrawing ? await (await fetch(studioAsset(cfg.assemblyDrawing.envelopes))).json() : null;
    for(const p of cfg.parts){
      const object=model.getObjectByName(p.node);
      if(!object)throw Error('Missing assembly '+p.node);
      const part={...p,object,original:object.position.clone(),offset:vec(p.offset)};
      if(hulls){part.wide=part.offset.clone();part.narrow=vec(p.portraitOffset);part.hull=hulls[p.node].map(vec);}
      parts.push(part);
    }
    if(cfg.assemblyDrawing){
      const {attachDrawing}=await import(studioAsset('assembly-drawing.js'));
      drawing=attachDrawing({parts,canvas,camera,focus:focusDrawing,hold:holdDrawing});
    }
  }
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
  assembled=measurePose();
  if(assemblyStudy){aspect=canvas.clientWidth/canvas.clientHeight;layoutAssembly();}
  else {setSpread(1);exploded=measurePose(cfg.assemblyDrawing?.direction);}
  resize();input();
  ground.update();updateGround=ground.update;
  const componentEntry=cfg.inspections?.some(item=>item.id===location.hash.slice(1));
  if(componentEntry)inspect(location.hash.slice(1));
  else if(!match(params.get('frame')||params.get('view')||'')){
    if(drawing && location.hash === '#exploded')enterDrawing();
    else if(assemblyStudy && location.hash === '#exploded') selectAssembly('', '');
    else if(params.has('capture'))reset();
    else {
      if(assemblyStudy){setSpread(1);frameCamera();}
      if(!reduced.matches)motion={from:1,to:0,intro:true,
        start:performance.now()+(cfg.assemblyDrawing?.introHold ?? (assemblyStudy?3200:1200)),
        duration:cfg.assemblyDrawing?.introDuration ?? 2100};
    }
  }
  $('status').hidden=true;$('tools').inert=false;new ResizeObserver(resize).observe(canvas);draw();return true;
}
function capture(width,height,transparent=false){
  stop();
  const oldRatio=renderer.getPixelRatio();renderer.setPixelRatio(1);renderer.setSize(width,height,false);applyViewport(width,height);
  const bg=scene.background, floorVisible=floor.visible;
  if(transparent){scene.background=null;floor.visible=false;renderer.setClearColor(0,0);}
  if(dirtyGround && updateGround && floor.visible){updateGround();dirtyGround=false;}
  renderer.render(scene,camera);const data=canvas.toDataURL('image/png');
  scene.background=bg;floor.visible=floorVisible;renderer.setPixelRatio(oldRatio);resize();return data;
}
window.carStudio={match,inspect,reset,capture,get pose(){return {spread:amount,cabin,separated:drawing?.separated,component:drawing?.separated ? $('assembly-component').value : undefined,assembly:drawing?.selected ? $('assembly-part').value || $('assembly-system').value : undefined,inspection:inspection?.id,matched:matched?.id,position:camera?.position.toArray(),target:target.toArray()};},ready:init().catch(e=>{
  console.error(e);
  const poster=document.createElement('img');
  poster.className='fallback-poster';
  poster.alt=cfg?.title || document.querySelector('h1').textContent;
  poster.src=cfg?.poster || 'poster.webp';
  canvas.after(poster);
  $('status').textContent='The interactive model could not load. Open Photo comparisons to see the matched views.';
  throw e;
})};
