import * as T from '/honda-civic-type-r/vendor/three-r170.js';
import {createStudioFloor} from '/honda-civic-type-r/studio-floor.js';
const $ = id => document.getElementById(id);
const vec = a => new T.Vector3(...a);
const assetRevision = new URL(import.meta.url).searchParams.get('v');
function studioAsset(name) {
  const url = new URL(name, document.baseURI);
  if (assetRevision && url.origin === location.origin) url.searchParams.set('v', assetRevision);
  return url.href;
}
const canvas = $('stage');
const reduced = matchMedia('(prefers-reduced-motion: reduce)');
const params = new URLSearchParams(location.search);
let cfg, renderer, scene, camera, floor, model, exploded, assembled, matched, motion, cabin = false;
let assemblyStudy, assemblyAspect, gestures, detailFrame;
const viewRect={left:0,top:0,width:1,height:1};
let amount = 0, frame = 0, aspect = 1;
let inspection = null, inspectionPose = null, inspectionNav = null;
const hiddenMeshes = [];
const hiddenForPhoto = [];
const steeredForPhoto = new Map();
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
  $('factory-link').before(inspectionNav);
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
const target = new T.Vector3(), actions = [];

const defaultNear = .012;
let updateGround,dirtyGround=true;
const pressed = (id,on) => $(id).setAttribute('aria-pressed',String(on));

function draw() {
  frame = 0;
  if (motion && motion.pausedAt==null && Number.isFinite(motion.start)) {
    const t = T.MathUtils.clamp((performance.now()-motion.start)/motion.duration,0,1);
    const u = t*t*(3-2*t);
    setSpread(motion.from+(motion.to-motion.from)*u);
    frameCamera();
    if (t >= 1) {
      if(motion.opening && location.hash==='#'+assemblyStudy.hash)setModelUrl();
      motion = null;
    }
    else invalidate();
  }
  if(dirtyGround && updateGround){updateGround();dirtyGround=false;}
  renderer.render(scene,camera);
}
function invalidate() { if (!frame) frame=requestAnimationFrame(draw); }
function setSpread(a) {
  amount=a;dirtyGround=true;
  assemblyStudy.apply(a);
  floor.visible=a<.001 && !inspection;
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
    for (let p=o; p; p=p.parent) if (!p.visible) return;
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
  pose.fitAspect=aspect;
  return pose.eye.copy(center).addScaledVector(direction,d);
}
function frameCamera() {
  const progress=assemblyStudy.cameraAmount(amount);
  target.copy(assembled.center).lerp(exploded.center,progress);
  camera.position.copy(fit(assembled)).lerp(fit(exploded),progress);
  camera.lookAt(target);
}
function layoutAssembly() {
  const spread=amount;
  assemblyAspect=aspect;
  assemblyStudy.layout(aspect); setSpread(1);
  exploded=measurePose(assemblyStudy.direction); setSpread(spread);
}
function selectAssembly(group, piece) {
  stop(); gestures?.restore(); userOrbit=false;
  assemblyStudy.select(group, piece);
  layoutAssembly(); setSpread(1); frameCamera();
  camera.near=piece!=='' ? Math.min(defaultNear,camera.position.distanceTo(target)/100) : defaultNear;
  camera.updateProjectionMatrix(); invalidate();
}
function resize() {
  const w=canvas.clientWidth, h=canvas.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w,h,false);
  aspect=w/h;
  if(!inspection && !matched && !cabin && !gestures?.moved && !actions.some(a=>a.active) && Math.abs(assemblyAspect-aspect)>.02) layoutAssembly();
  applyViewport(w,h);
  if(inspection && !userOrbit){target.copy(inspectionPose.center);camera.position.copy(fit(inspectionPose));camera.lookAt(target);}
  else if(model && !cabin && !matched && !userOrbit) frameCamera();
  invalidate();
}
function applyViewport(w,h) {
  const framing=matched || detailFrame;
  const ar = framing && (framing.aspect || (framing.imageSize && framing.imageSize[0]/framing.imageSize[1]) || (framing.width/framing.height));
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
  Object.assign(viewRect,{left:(w-vw)/2,top:(h-vh)/2,width:vw,height:vh});
  renderer.setViewport(viewRect.left,viewRect.top,vw,vh);
}
function stop() { motion=null; }
let userOrbit=false;
function unmatch() {
  if (steeredForPhoto.size || hiddenForPhoto.length) dirtyGround=true;
  for (const [node, yaw] of steeredForPhoto) node.rotation.y = yaw;
  steeredForPhoto.clear();
  for (const node of hiddenForPhoto) node.visible=true;
  hiddenForPhoto.length=0;
  matched=detailFrame=null;camera.near=defaultNear;camera.clearViewOffset();camera.up.set(0,1,0);
  $('angles').value='';$('ref-line').hidden=true;applyViewport(canvas.clientWidth,canvas.clientHeight);
}
function clearActions() {
  for(const a of actions){
    for(const b of a.nodes){b.object.rotation.copy(b.original);b.object.position.copy(b.position);b.object.visible=b.visible;}
    a.active=false;a.button.setAttribute('aria-pressed','false');
  }
}
function reset() {
  stop();gestures?.restore();if(inspection)clearInspection();
  if(assemblyStudy.detailed){assemblyStudy.select('','');layoutAssembly();}
  unmatch();cabin=false;userOrbit=false;pressed('cabin',false);
  camera.fov=32;camera.updateProjectionMatrix();setSpread(0);clearActions();
  if(Math.abs(assemblyAspect-aspect)>.02)layoutAssembly();
  frameCamera();invalidate();
}
function match(id){
  const v=cfg.views.find(record=>String(record.id)===String(id));
  if(!v)return false;
  reset();matched=v;userOrbit=true;
  if(v.actions){matched=null;for(const a of actions)if(v.actions.includes(a.label))a.button.click();matched=v;}
  if (v.steeringDegrees) {
    for (const name of cfg.steeringPivots) {
      const node = model.getObjectByName(name);
      steeredForPhoto.set(node, node.rotation.y);
      node.rotation.y += T.MathUtils.degToRad(v.steeringDegrees);
    }
    dirtyGround = true;
  }
  for (const name of v.hide || []) {
    const node=model.getObjectByName(name);
    if (node?.visible) { hiddenForPhoto.push(node); node.visible=false; }
  }
  camera.near=v.near || defaultNear;camera.fov=v.fov||32;camera.position.copy(vec(v.position));target.copy(vec(v.target));
  camera.up.copy(vec(v.up||[0,1,0]));camera.lookAt(target);
  if(v.shiftX || v.shiftY) {
    const w=v.width||1920,h=v.height||1080;
    camera.setViewOffset(w,h,v.shiftX||0,v.shiftY||0,w,h);
  }
  applyViewport(canvas.clientWidth,canvas.clientHeight);
  $('angles').value=v.id;
  $('source').href=v.source||cfg.review;$('ref-line').hidden=!v.source;invalidate();return true;
}
function setModelUrl(id,hash='model') {
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
function openSpread(){
  reset();setSpread(1);frameCamera();invalidate();
}
function toggleSpread(){
  const dest=motion ? 1-motion.to : (amount>.5?0:1);
  setModelUrl(null,dest ? assemblyStudy.hash : 'model');
  if(assemblyStudy.detailed)selectAssembly('','');
  stop(); gestures.restore(); if(cabin||matched||detailFrame||inspection)reset();userOrbit=false;
  clearActions();
  if(reduced.matches){setSpread(dest);frameCamera();}
  else motion={from:amount,to:dest,start:performance.now(),duration:assemblyStudy.transitionDuration};
  invalidate();
}
function toggleCabin(){
  setModelUrl();
  if(cabin){reset();return;}
  reset();cabin=true;pressed('cabin',true);camera.fov=72;camera.updateProjectionMatrix();
  camera.position.copy(vec(cfg.eye));target.copy(vec(cfg.look));camera.lookAt(target);invalidate();
}
function interact(){
  stop();
  if(matched){
    if(matched.id.startsWith('badge-')){
      // Keep the inspection cutaway and close-focus lens as the visitor moves.
      detailFrame=matched;matched=null;$('angles').value='';$('ref-line').hidden=true;
    }else unmatch();
    setModelUrl();
  }
  userOrbit=true;
}
function orbit(dx,dy){
  interact();
  const v=cabin ? target.clone().sub(camera.position) : camera.position.clone().sub(target);
  const r=v.length(), yaw=Math.atan2(v.x,v.z)+(cabin?1:-1)*dx*.007;
  const pitch=T.MathUtils.clamp(Math.asin(v.y/r)+dy*.006,cabin||inspection||detailFrame||assemblyStudy.detailed?-1.45:.025,1.48);
  v.set(Math.sin(yaw)*Math.cos(pitch),Math.sin(pitch),Math.cos(yaw)*Math.cos(pitch)).multiplyScalar(r);
  if(cabin)target.copy(camera.position).add(v);else camera.position.copy(target).add(v);
  camera.lookAt(target);invalidate();
}
function zoom(factor){
  interact();
  if(cabin){camera.fov=T.MathUtils.clamp(camera.fov*factor,37,105);camera.updateProjectionMatrix();}
  else{
    const d=camera.position.clone().sub(target);
    let minimum=.35;
    if(detailFrame)minimum=camera.near*3;
    else if(assemblyStudy.detailed)minimum=Math.min(.35,fit(exploded).distanceTo(exploded.center)*.1);
    const distance=T.MathUtils.clamp(d.length()*factor,minimum,65);
    camera.position.copy(target).addScaledVector(d.normalize(),distance);
  }
  invalidate();
}
function input(){
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
    $('panel').hidden=true;$('controls').setAttribute('aria-expanded','false');canvas.focus();
  };
  $('model-link').onclick=e=>{e.preventDefault();resetView();};
  window.addEventListener('hashchange',()=>{
    const id=location.hash.slice(1);
    if(id===assemblyStudy.hash)openSpread();
    else if(!inspect(id) && id==='model')reset();
  });
}
async function init(){
  const readJSON = async name => {
    const response=await fetch(studioAsset(name));
    if(!response.ok)throw Error(name+' returned '+response.status);
    return response.json();
  };
  const [config,factoryCameras,badgeViews]=await Promise.all([readJSON('studio.json'),
    readJSON('factory-360/cameras.json').catch(error=>{console.warn(error);return [];}),readJSON('badge-views.json')]);
  cfg=config;
  const referenceViews=cfg.views;
  cfg.views=[...referenceViews,...badgeViews,...factoryCameras];
  $('review').href=cfg.review;$('spec').textContent=cfg.subtitle;
  renderer=new T.WebGLRenderer({canvas,antialias:true,alpha:true,preserveDrawingBuffer:params.has('capture')});
  renderer.setPixelRatio(params.has('capture')?1:Math.min(devicePixelRatio,2));
  renderer.toneMapping=T.ACESFilmicToneMapping;renderer.toneMappingExposure=1.03;
  scene=new T.Scene();scene.background=new T.Color(0xe8ebed);
  // Long reflection banks follow the existing car studios and describe the crowned sheet metal.
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
      let light=.25+.72*smooth(lat,.64,1.36)+high*(.34+2.45*bank)+low*(.15+.98*bank)+.82*endWall;
      if (cfg.sideSoftboxes) {
        // Long, diffuse side banks reflect across the rocker, waist and front apron.
        const side = .50 + .50 * Math.abs(Math.cos(az - .18));
        const waist = smooth(lat, -.32, -.16) * (1 - smooth(lat, -.04, .07));
        const shoulder = smooth(lat, .19, .30) * (1 - smooth(lat, .44, .58));
        // A smaller overhead panel catches the convex windscreen and near fender.
        const overhead = smooth(az, -1.94, -1.84) * (1 - smooth(az, -1.37, -1.17))
                       * smooth(lat, .57, .71) * (1 - smooth(lat, 1.30, 1.46));
        light = .23 + .61 * smooth(lat, .60, 1.36) + side * (1.28 * waist + 3.8 * shoulder)
              + .65 * endWall + 3.9 * overhead;
      }
      const k=4*(y*512+x);pixels.data[k]=light*64;pixels.data[k+1]=light*66;pixels.data[k+2]=light*69;pixels.data[k+3]=255;
    }
  }
  context.putImageData(pixels,0,0);const studio=new T.CanvasTexture(panorama);
  scene.environment=pmrem.fromEquirectangular(studio).texture;studio.dispose();pmrem.dispose();
  scene.environmentIntensity=1.38;
  scene.add(new T.HemisphereLight(0xf2f6fd,0x53575e,1.13));
  const key=new T.DirectionalLight(0xfff8ef,1.43);key.position.set(-3,6,3);scene.add(key);
  renderer.shadowMap.enabled=true;renderer.shadowMap.type=T.PCFSoftShadowMap;
  key.castShadow=true;key.shadow.mapSize.set(1024,1024);Object.assign(key.shadow.camera,{left:-5,right:5,top:5,bottom:-5,near:.1,far:22});key.shadow.camera.updateProjectionMatrix();key.shadow.normalBias=.008;
  const fill=new T.DirectionalLight(0xeef4ff,1.1);fill.position.set(-5,2,-4);scene.add(fill);
  if (cfg.sideSoftboxes) {
    key.intensity = 1.12;
    const sideKey = new T.DirectionalLight(0xf0f4ff, .62);
    sideKey.position.set(5, 2, 3);
    scene.add(sideKey);
  }
  const ground=createStudioFloor(scene,renderer,0xe8ebed);
  floor=ground.floor;
  camera=new T.PerspectiveCamera(32,1,defaultNear,300);
  const draco=new T.DRACOLoader().setDecoderPath('/honda-civic-type-r/vendor/');
  const gltf=await new T.GLTFLoader().setDRACOLoader(draco).loadAsync(studioAsset(cfg.model)).finally(()=>draco.dispose());
  model=gltf.scene;scene.add(model);model.traverse(o=>{if(o.isMesh){o.castShadow=!o.material.transparent;if(o.castShadow)ground.addCaster(o);}});
  const [{createAssemblyStudy},{bindModelGestures}]=await Promise.all([
    import(studioAsset('assembly.js')),import(studioAsset('interaction.js'))
  ]);
  assemblyStudy=createAssemblyStudy(T,model,selectAssembly,{stop});
  gestures=bindModelGestures(T,{canvas,camera,model,viewRect,assembly:assemblyStudy,interact,orbit,zoom,
    cabin:()=>cabin,repaint:()=>{dirtyGround=true;invalidate();},
    companions:group=>amount===0 ? (actions.find(a=>a.hinge && a.group===group.key)?.nodes.slice(1) || []) : [],
    activate(owner){
      if(assemblyStudy.pick(owner))return;
      if(amount===0 && !inspection){
        const object=owner.group.nodes[owner.index].object;
        const action=actions.find(a=>a.group===owner.group.key || a.nodes.some(n=>n.object===object));
        if(action)action.button.click();
      }
    }
  });
  for(const a of cfg.actions||[]){
    const button=document.createElement('button');button.textContent=a.label;button.setAttribute('aria-pressed','false');$('actions').append(button);
    const nodes=a.nodes.map(n=>{const object=model.getObjectByName(n.node);if(!object)throw Error('Missing pivot '+n.node);return {...n,object,original:object.rotation.clone(),position:object.position.clone(),visible:object.visible};});
    const state={...a,nodes,button,active:false};actions.push(state);
    button.onclick=()=>{stop();gestures.restore();if(amount || matched || detailFrame || inspection)resetView();state.active=!state.active;
      for(const n of nodes){
        if(n.hidden)n.object.visible=state.active?false:n.visible;
        else {
          const angle=state.active?T.MathUtils.degToRad(n.degrees):0;
          n.object.rotation.copy(n.original);
          n.object.rotation[n.axis]+=angle;
          if(a.hinge){
            const pivot=vec(a.hinge), axis=new T.Vector3();
            axis[n.axis]=1;
            n.object.position.copy(n.position).sub(pivot).applyAxisAngle(axis,angle).add(pivot);
          }
        }
      }
      button.setAttribute('aria-pressed',String(state.active));dirtyGround=true;invalidate();};
  }
  for(const [label,views] of [['Photo comparisons',referenceViews],['Badge details',badgeViews],['Factory 360°',factoryCameras]]){
    if(!views.length)continue;
    const group=document.createElement('optgroup');group.label=label;
    for(const v of views)group.append(new Option(v.label,v.id));
    $('angles').append(group);
  }
  buildInspectionMenu();
  assembled=measurePose();
  aspect=canvas.clientWidth/canvas.clientHeight;
  layoutAssembly();
  resize();input();
  ground.update();updateGround=ground.update;
  const componentEntry=cfg.inspections?.some(item=>item.id===location.hash.slice(1));
  if(componentEntry)inspect(location.hash.slice(1));
  else if(!match(params.get('frame')||params.get('view')||'')){
    if(params.has('capture') || reduced.matches)reset();
    else {
      setSpread(1);frameCamera();
      const hold=assemblyStudy.intro.hold;
      const intro=motion={from:1,to:0,start:Infinity,duration:assemblyStudy.intro.duration,opening:true};
      // Start the hold after the first WebGL frame has reached the screen.
      requestAnimationFrame(()=>requestAnimationFrame(()=>{
        if(motion!==intro)return;
        intro.start=performance.now()+hold;invalidate();
      }));
    }
  }
  document.addEventListener('visibilitychange',()=>{
    if(!motion?.opening || !Number.isFinite(motion.start))return;
    if(document.hidden)motion.pausedAt=performance.now();
    else if(motion.pausedAt!=null){
      motion.start+=performance.now()-motion.pausedAt;delete motion.pausedAt;invalidate();
    }
  });
  reduced.addEventListener('change',()=>{
    if(reduced.matches && motion){const dest=motion.to;stop();setSpread(dest);frameCamera();invalidate();}
  });
  $('status').hidden=true;$('tools').inert=false;new ResizeObserver(resize).observe(canvas);draw();return true;
}
function capture(width,height,transparent=false){
  stop();
  const oldRatio=renderer.getPixelRatio();renderer.setPixelRatio(1);renderer.setSize(width,height,false);applyViewport(width,height);
  const bg=scene.background, floorVisible=floor.visible;
  if(transparent){scene.background=null;floor.visible=false;renderer.setClearColor(0,0);}
  renderer.render(scene,camera);const data=canvas.toDataURL('image/png');
  scene.background=bg;floor.visible=floorVisible;renderer.setPixelRatio(oldRatio);resize();return data;
}
window.carStudio={match,inspect,reset,capture,get pose(){return {spread:amount,cabin,inspection:inspection?.id,matched:matched?.id,position:camera?.position.toArray(),target:target.toArray()};},ready:init().catch(e=>{
  console.error(e);
  const poster=document.createElement('img');
  poster.className='fallback-poster';
  poster.alt=cfg?.title || document.querySelector('h1').textContent;
  poster.src=cfg?.poster || 'poster.webp';
  canvas.after(poster);
  $('status').textContent='The interactive model could not load. Open Photo comparisons to see the matched views.';
  throw e;
})};
