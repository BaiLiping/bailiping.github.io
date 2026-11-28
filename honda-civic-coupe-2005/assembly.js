function category(name) {
  if (/^(Wheel|Tire|Brake|SpareWheel)/.test(name)) return 'Wheels and brakes';
  if (/^(Suspension|Subframes|Exhaust|FuelTank)/.test(name)) return 'Undercarriage';
  if (/^(Engine|OilPan|CylinderHead|ValveCover|AccessoryDrive|Ignition|Transaxle|Halfshaft|Battery|Intake|Radiator)/.test(name)) return 'Engine bay and driveline';
  if (/^(Seat|Interior|Console|Dashboard|Steering|RearBench|DoorTrim|LuggageLiner|ParcelShelf)/.test(name)) return 'Interior';
  if (/Glass|Windshield|Moonroof|Mirror|Headlamp|Taillamp/.test(name)) return 'Glass, mirrors and lighting';
  return 'Body and closures';
}

export function detailDirection(name) {
  const side = name.endsWith('Left') ? 1 : -1;
  if (/^(Wheel|Tire|Brake|Fender|DoorGlass|QuarterGlass)/.test(name)) return [-.34,.34,side];
  if (name.startsWith('DoorTrim')) return [.18,.29,-side];
  if (name.startsWith('Door')) return [-.32,.47,side];
  if (name.startsWith('Mirror')) return [1,.38,.5*side];
  if (name.startsWith('Suspension')) return [-.62,.39,side];
  if (name.startsWith('Taillamp') || ['Trunk','RearBumper','BackGlass'].includes(name)) return [1,.57,.47*side];
  if (name.startsWith('Headlamp') || name==='FrontBumper') return [-1,.34,.47*side];
  if (name==='CylinderHead') return [-.46,-1.55,.48];
  if (name==='OilPan') return [-1.08,-.75,.49];
  if (name==='Engine') return [-.55,1.55,-.72];
  if (name==='ValveCover') return [-.69,1.6,.54];
  if (name==='AccessoryDrive') return [-.2,.27,1.4];
  if (name==='RearBench') return [-1.4,.54,.56];
  if (name==='LuggageLiner') return [1,1.55,.52];
  if (name==='SpareWheel') return [-.40,1.40,.75];
  if (name==='Dashboard' || name==='Steering') return [1,.52,.18];
  if (name==='Exhaust' || name==='Subframes' || name==='FuelTank') return [-.36,-.90,.72];
  return [-.58,.92,1.45];
}

// Each GLB group owns its attached trim. Nested groups, such as the glass in
// a door, have a second removal path relative to their parent.
export function createAssemblyPicker(T, model, config, canvas, camera, onSelect, controls) {
  const parts = new Map(config.parts.map(part => [model.getObjectByName(part.node), part]));
  const meshes = [];
  model.traverse(mesh => {
    if (!mesh.isMesh) return;
    for (let owner = mesh; owner; owner = owner.parent) {
      if (!parts.has(owner)) continue;
      meshes.push({ mesh, owner: parts.get(owner) });
      break;
    }
  });

  const picker = document.createElement('label');
  picker.id = 'assembly-picker';
  picker.textContent = 'Inspect component';
  const select = document.createElement('select');
  select.add(new Option('All ' + parts.size + ' assemblies', ''));
  const categories = new Map();
  for (const part of parts.values()) {
    const title=category(part.node);
    if (!categories.has(title)) {
      const group=document.createElement('optgroup');
      group.label=title;
      categories.set(title,group);
    }
    categories.get(title).append(new Option(part.label, part.node));
  }
  for (const title of [...categories.keys()].sort()) select.append(categories.get(title));
  select.onchange = () => onSelect(select.value);
  picker.append(select);
  picker.hidden = true;
  canvas.parentElement.append(picker);

  const lookup = new Map(meshes.map(({mesh,owner}) => [mesh, owner]));
  const ray = new T.Raycaster(), point = new T.Vector2();
  const label = document.createElement('span');
  label.className = 'assembly-tooltip';
  label.hidden = true;
  canvas.parentElement.append(label);

  let spread = 0, pending = null, hoverFrame = 0, clickTimer = 0;
  let xray = false;
  const originals = new Map(), pulled = new Map();
  function hit(event) {
    const rect = canvas.getBoundingClientRect();
    point.set(2*(event.clientX-rect.left)/rect.width-1, 1-2*(event.clientY-rect.top)/rect.height);
    ray.setFromCamera(point,camera);
    for (const {object} of ray.intersectObject(model, true)) {
      if (!lookup.has(object)) continue;
      let visible = true;
      for (let parent=object; parent; parent=parent.parent) if (!parent.visible) visible=false;
      if (visible) return lookup.get(object);
    }
  }
  function showHover(event) {
    const part = hit(event);
    label.hidden = !part;
    if (!part) return;
    label.textContent = part.label;
    const rect = canvas.getBoundingClientRect();
    label.style.left = Math.max(2, Math.min(event.clientX-rect.left+12,rect.width-label.offsetWidth-2))+'px';
    label.style.top = Math.max(2, Math.min(event.clientY-rect.top+12,rect.height-label.offsetHeight-2))+'px';
  }
  function setXray(value) {
    if (xray === value) return;
    xray=value;
    canvas.dataset.xray=String(value);
    for (const {mesh,owner} of meshes) {
      if (!/^(Body|Door|Fender|Hood|Trunk|BackGlass|Windshield|Moonroof|QuarterGlass|FrontBumper|RearBumper)/.test(owner.node)) continue;
      if (value) {
        const ghost = m => {
          const copy=m.clone();
          copy.transparent=true; copy.opacity=.10; copy.depthWrite=false;
          return copy;
        };
        originals.set(mesh,mesh.material);
        mesh.material=Array.isArray(mesh.material)?mesh.material.map(ghost):ghost(mesh.material);
      } else if(originals.has(mesh)) {
        for(const material of [].concat(mesh.material)) material.dispose();
        mesh.material=originals.get(mesh);
      }
    }
    if (!value) originals.clear();
    controls.invalidate();
  }
  function clearPulls() {
    for (const [object,rest] of pulled) object.position.copy(rest);
    pulled.clear();
    canvas.dataset.pulled='0';
  }
  function clear() {
    clearTimeout(clickTimer); cancelAnimationFrame(hoverFrame); hoverFrame=0;
    pending=null; label.hidden=true; setXray(false); clearPulls();
  }
  function closure(name) {
    if (/^(DoorGlass|DoorTrim|Mirror)(Left|Right)$/.test(name)) return 'Door'+(name.endsWith('Left')?'Left':'Right');
    return /^(DoorLeft|DoorRight|Trunk|Hood)$/.test(name) ? name : null;
  }
  canvas.addEventListener('pointerdown', event => {
    if (!event.isPrimary || event.button!==0) { pending=null; return; }
    const part=hit(event);
    if (!part) return;
    const pullable=!select.value && part.node!=='Subframes' && (event.shiftKey || spread>=.999);
    pending={id:event.pointerId,x:event.clientX,y:event.clientY,part,pullable,drag:false};
    if (pullable) {
      event.stopImmediatePropagation();
      canvas.setPointerCapture(event.pointerId);
    }
  },true);
  canvas.addEventListener('pointermove', event => {
    if (pending && pending.id===event.pointerId) {
      const dx=event.clientX-pending.x, dy=event.clientY-pending.y;
      if (Math.hypot(dx,dy)<5 && !pending.drag) return;
      if (!pending.pullable) {pending=null;label.hidden=true;return;}
      event.stopImmediatePropagation(); label.hidden=true;
      const item=model.getObjectByName(pending.part.node);
      if(!pending.drag) {
        controls.stop();
        pending.drag=true;
        if(!pulled.has(item)) pulled.set(item,item.position.clone());
        pending.base=item.position.clone();
      }
      const dist=item.getWorldPosition(new T.Vector3()).distanceTo(camera.position);
      const scale=2*dist*Math.tan(T.MathUtils.degToRad(camera.fov/2))/canvas.clientHeight;
      const delta=new T.Vector3(dx*scale,-dy*scale,0).applyQuaternion(camera.quaternion);
      const inverse=item.parent.getWorldQuaternion(new T.Quaternion()).invert();
      item.position.copy(pending.base).add(delta.applyQuaternion(inverse));
      canvas.dataset.pulled=String(pulled.size);controls.invalidate();
      return;
    }
    if(event.buttons || event.pointerType==='touch' || hoverFrame)return;
    hoverFrame=requestAnimationFrame(()=>{hoverFrame=0;showHover(event);});
  },true);
  canvas.addEventListener('pointerleave',()=>{cancelAnimationFrame(hoverFrame);hoverFrame=0;label.hidden=true;});
  canvas.addEventListener('pointercancel',()=>{pending=null;label.hidden=true;});
  canvas.addEventListener('pointerup',event=>{
    if(!pending || pending.id!==event.pointerId)return;
    const {drag,part,pullable}=pending; pending=null;
    if(pullable)event.stopImmediatePropagation();
    if(drag)return;
    clearTimeout(clickTimer);
    clickTimer=setTimeout(()=>{
      const item=model.getObjectByName(part.node);
      if(pulled.has(item)) {
        item.position.copy(pulled.get(item));
        pulled.delete(item); canvas.dataset.pulled=String(pulled.size);
        controls.invalidate();
        return;
      }
      if(spread>=.999 && !select.value)onSelect(part.node);
      else if(spread<.001 && !select.value && closure(part.node))controls.open(closure(part.node));
    },240);
  },true);
  canvas.addEventListener('dblclick',event=>{
    event.preventDefault();clearTimeout(clickTimer);controls.stop();setXray(!xray);
  });
  return {
    clear,
    setAmount(amount) {
      if (amount!==spread) clearPulls();
      spread=amount; picker.hidden=amount<.999;
    },
    select(name) {
      select.value=name;
      for (const {mesh,owner} of meshes) mesh.visible=!name || owner.node===name;
    },
  };
}
