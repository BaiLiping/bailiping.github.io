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
export function createAssemblyPicker(T, model, config, canvas, camera, onSelect) {
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

  // A tap on an assembly is a shortcut to the same close-up as the native menu.
  let start;
  canvas.addEventListener('pointerdown', event => {
    start = picker.hidden || select.value || !event.isPrimary ? null
      : {id:event.pointerId, x:event.clientX, y:event.clientY};
  });
  canvas.addEventListener('pointermove', event => {
    if (start && Math.hypot(event.clientX-start.x,event.clientY-start.y)>5) start=null;
  });
  canvas.addEventListener('pointercancel', () => { start=null; });
  canvas.addEventListener('pointerup', event => {
    if (!start || start.id !== event.pointerId) return;
    start=null;
    const rect=canvas.getBoundingClientRect();
    const point=new T.Vector2(2*(event.clientX-rect.left)/rect.width-1,1-2*(event.clientY-rect.top)/rect.height);
    const ray = new T.Raycaster();
    ray.setFromCamera(point, camera);
    const hit=ray.intersectObject(model, true).find(entry=>entry.object.isMesh && entry.object.visible);
    const part=hit && meshes.find(record=>record.mesh === hit.object);
    if (part) onSelect(part.owner.node);
  });
  return {
    setAmount(amount) { picker.hidden=amount<.999; },
    select(name) {
      select.value=name;
      for (const {mesh,owner} of meshes) mesh.visible=!name || owner.node===name;
    },
  };
}
