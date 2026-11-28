// Pull in the plane of the screen so a gesture follows the pointer at any orbit or zoom.
export function bindModelGestures(T, {canvas, model, camera, viewRect, assembly, interact, orbit, zoom, repaint, cabin, companions, activate}) {
  const ray = new T.Raycaster(), pointer = new T.Vector2(), intersections = [];
  const plane = new T.Plane(), normal = new T.Vector3(), world = new T.Vector3();
  const touches = new Map(), parked = new Map();
  const label = document.getElementById('part-label');
  let press, pinch = 0, lastHover = -Infinity;
  const hidden = object => {
    for (let p = object; p; p = p.parent) if (!p.visible) return true;
    return false;
  };
  function aim(x,y) {
    const rect = canvas.getBoundingClientRect();
    pointer.set((x-rect.left-viewRect.left)*2/viewRect.width-1, 1-(y-rect.top-viewRect.top)*2/viewRect.height);
    ray.setFromCamera(pointer,camera);
  }
  function pick(x,y) {
    aim(x,y);
    if(Math.max(Math.abs(pointer.x),Math.abs(pointer.y))>1)return null;
    intersections.length = 0;
    ray.intersectObject(model,true,intersections);
    for (const {object,point} of intersections) {
      const owner = assembly.ownerOf(object);
      if (owner && !hidden(object)) return {owner,point};
    }
    return null;
  }
  function identify(hit,x,y) {
    label.hidden = !hit;
    canvas.classList.toggle('over-part',Boolean(hit));
    if (!hit) return;
    const {group,index} = hit.owner;
    const node = group.nodes[index].object;
    const name = (node.userData.name || node.name).replaceAll('_',' ').replace(/\.\d+$/,'');
    label.textContent = name === group.label ? name : group.label + ' · ' + name;
    const rect = canvas.getBoundingClientRect();
    label.style.left = Math.max(4, Math.min(rect.width-label.offsetWidth-4, x-rect.left+12))+'px';
    label.style.top = Math.max(4, Math.min(rect.height-label.offsetHeight-4, y-rect.top+14))+'px';
  }
  function restore() {
    for (const [object,start] of parked) object.position.copy(start);
    parked.clear();
    identify(null);
  }
  function startPull(single) {
    const {group,index} = press.hit.owner;
    const nodes = single || assembly.detailed ? [group.nodes[index]] : [...group.nodes,...companions(group)];
    press.nodes = nodes.map(({object}) => {
      const start = object.position.clone();
      if (!parked.has(object)) parked.set(object,start);
      // Inverse linear transform converts displacement, including imported scale.
      const local = object.parent.matrixWorld.clone().invert().setPosition(0,0,0);
      return {object,start,local};
    });
    plane.setFromNormalAndCoplanarPoint(camera.getWorldDirection(normal),press.hit.point);
    press.mode = 'pull';
  }
  function touchDistance() {
    const [a,b] = touches.values();
    return Math.hypot(b.x-a.x,b.y-a.y);
  }
  function cancelDrag() {
    if (press?.mode === 'pull') {
      for (const {object,start} of press.nodes) object.position.copy(start);
      repaint();
    }
    press = null;
  }
  canvas.addEventListener('contextmenu',e=>e.preventDefault());
  canvas.addEventListener('pointerdown',e=>{
    interact();
    canvas.setPointerCapture(e.pointerId);
    if(e.pointerType==='touch') {
      touches.set(e.pointerId,{x:e.clientX,y:e.clientY});
      if(touches.size>1){cancelDrag();pinch=touchDistance();identify(null);return;}
    }
    if(press) return;
    const hit=e.button===0 ? pick(e.clientX,e.clientY) : null;
    press={id:e.pointerId,x:e.clientX,y:e.clientY,lastX:e.clientX,lastY:e.clientY,
      time:performance.now(),mode:'pending',hit};
    identify(hit,e.clientX,e.clientY);
  });
  canvas.addEventListener('pointermove',e=>{
    const x=e.clientX,y=e.clientY;
    if(touches.has(e.pointerId))touches.set(e.pointerId,{x,y});
    if(touches.size>=2) {
      const distance=touchDistance();
      if(pinch>1 && distance>1)zoom(pinch/distance);
      pinch=distance;
      return;
    }
    if(!press) {
      if(e.pointerType==='mouse' && !e.buttons && performance.now()-lastHover>=65) {
        lastHover=performance.now();
        identify(pick(x,y),x,y);
      }
      return;
    }
    if(press.id!==e.pointerId)return;
    if(press.mode==='pending'){
      if (Math.hypot(x-press.x,y-press.y)<5)return;
      const touch=e.pointerType==='touch', held=performance.now()-press.time>=350;
      if(press.hit && (!cabin() || e.altKey || (touch && held)) && (!touch || held))startPull(e.altKey || touch);
      else {press.mode='orbit';identify(null);}
    }
    if(press.mode==='pull'){
      aim(x,y);
      if(ray.ray.intersectPlane(plane,world)){
        const delta=world.sub(press.hit.point);
        for(const {object,start,local} of press.nodes)object.position.copy(delta).applyMatrix4(local).add(start);
        identify(press.hit,x,y);repaint();
      }
    }else orbit(x-press.lastX,y-press.lastY);
    press.lastX=x;press.lastY=y;
  });
  function end(e,cancelled) {
    const wasPinch=touches.size>=2;
    touches.delete(e.pointerId);
    if(wasPinch && touches.size===1) {
      const [id,p]=touches.entries().next().value;
      press={id,x:p.x,y:p.y,lastX:p.x,lastY:p.y,mode:'orbit'};
    }
    if(press?.id!==e.pointerId)return;
    if(cancelled) cancelDrag();
    else {
      if(press.mode==='pending' && press.hit)activate(press.hit.owner);
      press=null;
    }
    if(e.pointerType==='touch' || cancelled)identify(null);
  }
  canvas.addEventListener('pointerup',e=>end(e,false));
  canvas.addEventListener('pointercancel',e=>end(e,true));
  canvas.addEventListener('lostpointercapture',e=>end(e,true));
  canvas.addEventListener('pointerleave',()=>{if(!press)identify(null);});
  canvas.addEventListener('wheel',e=>{
    e.preventDefault();identify(null);
    zoom(Math.exp(T.MathUtils.clamp(e.deltaY,-500,500)*.001));
  },{passive:false});
  return {restore,get moved(){return parked.size>0;}};
}
