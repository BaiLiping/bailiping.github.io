// The coordinates below locate related assemblies around the bodyshell as seen
// from the front left. All stay at the GLB's metric scale and orientation.

const body = 'Body and glazing', interior = 'Interior', chassis = 'Chassis and driveline';
const running = 'Wheels and brakes', power = 'T-Hybrid powertrain';
const catalog = {
  Bodywork: [body, 'Bodyshell', 0, .4],
  Bonnet: [body, 'Bonnet', -3.4, 2.25],
  Left_Door: [body, 'Left door', .85, -1.3],
  Right_Door: [body, 'Right door', -.9, 2.75],
  Front_Bumper: [body, 'Front bumper', -4.9, .5],
  Rear_Bumper: [body, 'Rear bumper', 4.65, 1.55],
  Left_Headlight: [body, 'Left headlamp', -3.95, -.2],
  Right_Headlight: [body, 'Right headlamp', -4.15, 1.65],
  Windscreen: [body, 'Windscreen', -1.7, 2.65],
  Rear_Light_Bar: [body, 'Rear light bar', 4, 2.75],
  Rear_Spoiler: [body, 'Rear spoiler', 2.45, 3.15],
  Soft_Top: [body, 'Fabric roof and rear window', .45, 3.7],
  sideGlass: [body, 'Side glazing', 1.65, 2.45],
  engineDeck: [body,'Engine deck and grille', 2.65, 2.4],
  tonneau: [body,'Convertible top compartment lid', 1.15, 2.7],
  undertray: [chassis, 'Undertray', .15, -2.9],
  cabinFloor: [interior,'Cabin floor and trim', -.1, -1.05],
  Driver_Seat: [interior,'Driver seat', .65, 1.65],
  Passenger_Seat: [interior,'Passenger seat', -.1, 2.2],
  restraints: [interior,'Seat belts and buckles', .35, 1.95],
  rearSeats: [interior,'Rear seats', 1.85, 1.5],
  console: [interior,'Centre console', -.55, 1.6],
  dash: [interior,'Dashboard and instruments', -1.65, 1.9],
  wheel: [interior,'Steering wheel', -1.95, 1.2],
  mirror: [interior,'Interior mirror', -1.65, 3.25],
  trim: [interior,'Door trim', .25, -1.8],
  Front_Axle: [chassis,'Front axle', -2.45, -1.45],
  Rear_Axle: [chassis,'Rear axle', 1.45, -2.05],
  Propeller_Shaft: [chassis,'Propeller shaft', -.75, -2.2],
  HV_Battery: [chassis,'High-voltage battery', -.95, -2.9],
  Boxer_Engine: [power,'Flat-six engine', 3.9, -.9],
  Left_eTurbo: [power,'Left electric turbocharger', 4.35, -1.95],
  Right_eTurbo: [power,'Right electric turbocharger', 4.65, -.15],
  PDK_Hybrid: [power,'PDK transmission', 3.4, -2],
  PDK_Motor_Envelope: [power,'PDK electric motor envelope', 3.25, -2.75],
};
const cabinet = [
  [/interior mirror|auto.dimming/, 'mirror'],
  [/^steering/, 'wheel'],
  [/^dash|digitalcluster|pcmscreen|clock|chrono/, 'dash'],
  [/console|selector|pdk|switch|transmission tunnel/, 'console'],
  [/carpet|toe board/, 'cabinFloor'],
  [/seatbelt|buckle|shoulder belt|lap belt/, 'restraints'],
  [/rear (backrest|seat|cushion|insert)/, 'rearSeats'],
  [/door|belt molding/, 'trim'],
];
const skin = [
  [/undertray/, 'undertray'],
  [/engine deck|engine lid|deck slat/, 'engineDeck'],
  [/soft.top storage lid|tonneau/, 'tonneau'],
];
const asWords = name => name.replaceAll('_', ' ').replace(/\.[0-9]+$/, '').toLowerCase();
function matchRule(rules, name, fallback) {
  return rules.find(([pattern]) => pattern.test(name))?.[1] || fallback;
}

export function createAssemblyStudy(T, model, onSelect, {canvas,camera,stop}) {
  const direction = new T.Vector3(1.32, .88, .85).normalize();
  const right = new T.Vector3(0,1,0).cross(direction).normalize();
  const up = direction.clone().cross(right);
  const origin = new T.Vector3(0, .65, 0);
  const sample=model.getObjectByProperty('isMesh',true).geometry;
  if (!sample.boundingBox) sample.computeBoundingBox();
  const emptyBounds=()=>sample.boundingBox.clone().makeEmpty();
  const groups = new Map();
  let selected = null, piece = null;
  const meshOwner = new WeakMap();
  const boundsPoint = new T.Vector3();
  const add = (key, object, description) => {
    if (!groups.has(key)) {
      const [category,label,x,y] = description;
      groups.set(key,{key,category,label,x,y,nodes:[],bounds:emptyBounds()});
    }
    const group = groups.get(key);
    const localFromWorld = object.parent.matrixWorld.clone().invert().setPosition(0,0,0);
    const bounds = emptyBounds(), projected = new T.Vector3();
    const index=group.nodes.length;
    object.traverse(node => {
      if (!node.isMesh) return;
      meshOwner.set(node,{group,index});
      const pos = node.geometry.getAttribute('position');
      for (let i=0; i<pos.count; i++) {
        boundsPoint.fromBufferAttribute(pos,i).applyMatrix4(node.matrixWorld);
        bounds.expandByPoint(projected.set(boundsPoint.dot(right),boundsPoint.dot(up),boundsPoint.dot(direction)));
      }
    });
    if(bounds.isEmpty())return;
    group.nodes.push({object, bounds, start:object.position.clone(), visible:object.visible, localFromWorld, delta:new T.Vector3()});
    group.bounds.union(bounds);
  };
  model.updateMatrixWorld(true);
  for (const root of model.children) {
    const name = root.name;
    if (name.endsWith('_Wheel') || name.endsWith('_Brake')) {
      const side = name.includes('_Left_') ? 1 : -1;
      const front = name.startsWith('Front');
      const axleX = front ? -2.25 : 1.8;
      const axleY = front ? -.75 : -.1;
      const prefix = name.replaceAll('_',' ').replace(/ (Wheel|Brake)$/,'').toLowerCase();
      for (const node of root.children) {
        const n = asWords(node.userData.name || node.name);
        const isBrake = name.endsWith('_Brake');
        const isTyre = /\b(tire|tyre)\b/.test(n) && !n.includes('valve');
        const kind = isBrake ? 'brake' : isTyre ? 'tyre' : 'rim';
        const shift = isBrake ? 1.35 : isTyre ? 2.65 : 2;
        add(name+kind, node, [running, prefix[0].toUpperCase()+prefix.slice(1)+' '+kind, axleX+side*shift*.76, axleY-side*shift*1.02]);
      }
    } else if (name==='Bodywork' || name==='Cabin' || name==='Soft_Top') {
      for (const node of root.children) {
        const n = asWords(node.userData.name || node.name);
        let key;
        if (name==='Cabin') key=matchRule(cabinet,n,'cabinFloor');
        else if (name==='Bodywork') key=matchRule(skin,n,'Bodywork');
        else key=/frameless|side glass/.test(n) ? 'sideGlass' : 'Soft_Top';
        add(key,node,catalog[key]);
      }
    } else {
      const description=catalog[name] || [body,name.replaceAll('_',' '),2,1];
      for(const node of root.children.length ? root.children : [root])add(name,node,description);
    }
  }
  const items = [...groups.values()].filter(g=>!g.bounds.isEmpty());
  for (const g of items) {
    g.center=g.bounds.getCenter(new T.Vector3());
    g.size=g.bounds.getSize(new T.Vector3());
    // The fixed shell is the visual datum for every removal direction.
    g.mobility=g.key==='Bodywork' ? 0 : 1/(.6+g.size.x*g.size.y);
  }

  // Relax overlaps between tight projected mesh bounds. Preserve the order of
  // neighboring parts while allowing the same plate to fit portrait or landscape.
  function layout(aspect) {
    if (selected) {
      // Lay out the named pieces of one assembly in reading order. Keep their
      // original orientation, material surfaces and mounting detail intact.
      const pieces = piece ? [piece] : selected.nodes;
      const cells = pieces.map(n=>({n,size:n.bounds.getSize(new T.Vector3()),center:n.bounds.getCenter(new T.Vector3())}));
      const largest=Math.max(...cells.map(c=>Math.max(c.size.x,c.size.y)));
      const gap=Math.max(.008,largest*.095);
      const area=cells.reduce((a,c)=>a+(c.size.x+gap)*(c.size.y+gap),0);
      const width=Math.max(...cells.map(c=>c.size.x),Math.sqrt(area*aspect));
      cells.sort((a,b)=>b.center.y-a.center.y || a.center.x-b.center.x);
      const rows=[];let row={cells:[],width:0,height:0};
      for(const c of cells){
        if(row.cells.length && row.width+gap+c.size.x>width){rows.push(row);row={cells:[],width:0,height:0};}
        c.x=row.width+(row.cells.length?gap:0)+c.size.x/2;
        row.width=c.x+c.size.x/2;row.height=Math.max(row.height,c.size.y);row.cells.push(c);
      }
      rows.push(row);
      let top=rows.reduce((h,r)=>h+r.height+gap,-gap)/2;
      for(const r of rows){
        const y=top-r.height/2;
        for(const c of r.cells){
          c.n.delta.copy(origin).addScaledVector(right,c.x-r.width/2-c.center.x)
            .addScaledVector(up,y-c.center.y).addScaledVector(direction,-c.center.z).applyMatrix4(c.n.localFromWorld);
        }
        top-=r.height+gap;
      }
      return;
    }
    const stretch = T.MathUtils.clamp(Math.pow(aspect/1.4, aspect>1.4 ? .86 : .26), .69, 2.15);
    const clearance = .17;
    // Account for perspective at the outer edge of the plate. Deep pieces near
    // a corner project farther out than their parallel-projection bounds.
    const paddedHalf=(center,length,depth)=>length*.53+(Math.abs(center)+1+length/2)*depth/22;
    for (const g of items) {
      g.u=g.x*stretch;
      g.v=g.y/stretch;
    }
    for (let iteration=0; iteration<480; iteration++) {
      let overlaps=0;
      for (let i=0; i<items.length; i++) {
        const a=items[i];
        for (let j=i+1; j<items.length; j++) {
          const b=items[j], dx=b.u-a.u, dy=b.v-a.v;
          const ox=paddedHalf(a.u,a.size.x,a.size.z)+paddedHalf(b.u,b.size.x,b.size.z)+clearance-Math.abs(dx);
          const oy=paddedHalf(a.v,a.size.y,a.size.z)+paddedHalf(b.v,b.size.y,b.size.z)+clearance-Math.abs(dy);
          if (Math.min(ox,oy)<=.0002) continue;
          overlaps++;
          const weight=a.mobility/(a.mobility+b.mobility);
          if (ox/Math.sqrt(aspect)<oy) {
            const shift=(Math.sign(dx)||1)*ox*1.01;
            a.u-=shift*weight; b.u+=shift*(1-weight);
          } else {
            const shift=(Math.sign(dy)||1)*oy*1.01;
            a.v-=shift*weight; b.v+=shift*(1-weight);
          }
        }
      }
      if (!overlaps) break;
    }
    for (const g of items) {
      const delta=origin.clone().addScaledVector(right,g.u-g.center.x)
        .addScaledVector(up,g.v-g.center.y).addScaledVector(direction,-g.center.z);
      for (const n of g.nodes) n.delta.copy(delta).applyMatrix4(n.localFromWorld);
    }
  }

  function apply(amount) {
    for (const g of items) {
      // During reassembly, seat the driveline first and close the body last.
      const start=g.category===power || g.category===chassis ? .22 : g.category===interior ? .10 : 0;
      const end=g.category===body ? .84 : 1;
      const t=T.MathUtils.smoothstep(amount,start,end);
      for (const n of g.nodes) n.object.position.copy(n.start).addScaledVector(n.delta,t);
    }
  }
  const menu=document.getElementById('porsche-assembly-picker');
  const groupSelect=document.getElementById('assembly-group'), partSelect=document.getElementById('assembly-piece');
  menu.addEventListener('focusin',stop);
  menu.addEventListener('pointerdown',stop);
  const categories=new Map();
  for(const g of items){
    if(!categories.has(g.category)){
      const group=document.createElement('optgroup');group.label=g.category;
      categories.set(g.category,group);groupSelect.append(group);
    }
    categories.get(g.category).append(new Option(g.label,g.key));
  }
  groupSelect.onchange=()=>onSelect(groupSelect.value,'');
  partSelect.onchange=()=>onSelect(groupSelect.value,partSelect.value);
  function select(key,index){
    const group=groups.get(key);
    if(selected!==group){
      partSelect.replaceChildren(new Option('All parts',''));
      for(const [i,n] of (group?.nodes || []).entries()){
        const raw=n.object.userData.name || n.object.name;
        partSelect.add(new Option(raw.replaceAll('_',' ').replace(/\.[0-9]+$/, ''),String(i)));
      }
    }
    selected=group;piece=index===''?null:group?.nodes[Number(index)];
    for(const g of items)for(const n of g.nodes){
      n.object.visible=n.visible && (!selected || g===selected) && (!piece || n===piece);
    }
    groupSelect.value=key;partSelect.value=index;
    partSelect.parentElement.hidden=!selected;
  }
  // Picking a visible component opens the same hierarchy as the keyboard menus.
  // A drag, pinch or cancelled touch must never select an assembly.
  const ray=new T.Raycaster(),pointer=new T.Vector2();
  let press=null;
  canvas.addEventListener('pointerdown',event=>{
    press=event.isPrimary && !menu.hidden ? [event.clientX,event.clientY] : null;
  });
  canvas.addEventListener('pointermove',event=>{
    if(press && Math.hypot(event.clientX-press[0],event.clientY-press[1])>6)press=null;
  });
  canvas.addEventListener('pointercancel',()=>{press=null;});
  canvas.addEventListener('pointerup',event=>{
    if(!press || menu.hidden || piece)return;
    press=null;
    const rect=canvas.getBoundingClientRect();
    pointer.set((event.clientX-rect.left)*2/rect.width-1,1-(event.clientY-rect.top)*2/rect.height);
    ray.setFromCamera(pointer,camera);
    const hits=ray.intersectObject(model);
    for(const hit of hits){
      const owner=meshOwner.get(hit.object);
      if(!owner || (selected && owner.group!==selected))continue;
      onSelect(owner.group.key,selected ? String(owner.index) : '');
      break;
    }
  });
  return {
    direction:direction.toArray(),layout,select,hash:'exploded',intro:{hold:1700,duration:3100},transitionDuration:1450,
    // The first body panels reach their targets before the interior. Follow
    // their progress to keep those outer pieces in frame during the transition.
    cameraAmount(amount){return Math.max(amount,T.MathUtils.smoothstep(amount,0,.84));},
    get detailed(){return Boolean(selected);},
    apply(amount){apply(amount);menu.hidden=amount<.999;}
  };
}
