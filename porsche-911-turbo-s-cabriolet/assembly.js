// Removal paths are measured from each component's installed position in metres.
// +X is vehicle left, +Y is up, and +Z points toward the front bumper.

const body = 'Body and glazing', interior = 'Interior', chassis = 'Chassis and driveline';
const running = 'Wheels and brakes', power = 'T-Hybrid powertrain';
const catalog = {
  Bodywork: [body, 'Bodyshell', 0, 1.05, 0],
  Bonnet: [body, 'Bonnet', 0, 1.135, 0.06],
  Left_Door: [body, 'Left door', 0.19, 1.06, 0],
  Right_Door: [body, 'Right door', -0.19, 1.06, 0],
  Front_Bumper: [body, 'Front bumper', 0, 1.025, 0.13],
  Rear_Bumper: [body, 'Rear bumper', 0, 1.03, -0.13],
  Left_Headlight: [body, 'Left headlamp', 0.035, 1.10, 0.07],
  Right_Headlight: [body, 'Right headlamp', -0.035, 1.10, 0.07],
  Windscreen: [body, 'Windscreen', 0, 1.12, 0],
  Rear_Light_Bar: [body, 'Rear light bar', 0, 1.11, -0.10],
  Rear_Spoiler: [body, 'Rear spoiler', 0, 1.56, -0.24],
  Soft_Top: [body, 'Fabric roof and rear window', 0, 1.27, -0.02],
  sideGlass: [body, 'Side glazing', 0, 1.15, 0],
  engineDeck: [body,'Engine deck and grille', 0, 1.15, -0.06],
  tonneau: [body,'Convertible top compartment lid', 0, 1.135, -0.02],
  undertray: [chassis, 'Undertray', 0, -0.08, 0],
  cabinFloor: [interior,'Cabin floor and trim', 0.15, 0.1, 0],
  Driver_Seat: [interior,'Driver seat', 0.29, 0.35, 0.01],
  Passenger_Seat: [interior,'Passenger seat', 0.03, 0.35, 0.01],
  restraints: [interior,'Seat belts and buckles', 0.13, 0.3, 0],
  rearSeats: [interior,'Rear seats', 0.16, 0.28, -0.12],
  console: [interior,'Centre console', 0.15, 0.3, 0.03],
  dash: [interior,'Dashboard and instruments', 0.15, 0.44, 0.17],
  wheel: [interior,'Steering wheel', 0.36, 0.36, 0.28],
  mirror: [interior,'Interior mirror', 0, 1.12, 0],
  trim: [interior,'Door trim', 0.17, 0.2, 0],
  Front_Axle: [chassis,'Front axle', 0, 0.03, 0.06],
  Rear_Axle: [chassis,'Rear axle', 0, 0.03, -0.06],
  Propeller_Shaft: [chassis,'Propeller shaft', 0, -0.035, 0.04],
  HV_Battery: [chassis,'High-voltage battery', 0.23, 0.07, 0.28],
  Boxer_Engine: [power,'Flat-six engine', 0.12, 0.13, -0.56],
  Left_eTurbo: [power,'Left electric turbocharger', 0.38, 0.13, -0.58],
  Right_eTurbo: [power,'Right electric turbocharger', -0.18, 0.13, -0.6],
  PDK_Hybrid: [power,'PDK transmission', 0.15, 0.12, -0.28],
  PDK_Motor_Envelope: [power,'PDK electric motor envelope', 0.16, 0.14, 0.03],
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

export function createAssemblyStudy(T, model, onSelect, {stop}) {
  const direction = new T.Vector3(1.12, .61, 1.32).normalize();
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
      const [category,label,x,y,z] = description;
      groups.set(key,{key,category,label,travel:new T.Vector3(x,y,z),nodes:[],bounds:emptyBounds()});
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
      const prefix = name.replaceAll('_',' ').replace(/ (Wheel|Brake)$/,'').toLowerCase();
      for (const node of root.children) {
        const n = asWords(node.userData.name || node.name);
        const isBrake = name.endsWith('_Brake');
        const isTyre = /\b(tire|tyre)\b/.test(n) && !n.includes('valve');
        const kind = isBrake ? 'brake' : isTyre ? 'tyre' : 'rim';
        const shift = isBrake ? .07 : isTyre ? .45 : .40;
        add(name+kind, node, [running, prefix[0].toUpperCase()+prefix.slice(1)+' '+kind, side*shift, 0, 0]);
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
      const description=catalog[name] || [body,name.replaceAll('_',' '),0,0,0];
      for(const node of root.children.length ? root.children : [root])add(name,node,description);
    }
  }
  const items = [...groups.values()].filter(g=>!g.bounds.isEmpty());
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
    // The shell and its outer panels lift together. Wheels pull along their
    // axles and the engine withdraws aft; orbit reveals their mounting faces.
    for (const group of items) {
      for (const node of group.nodes) {
        node.delta.copy(group.travel).applyMatrix4(node.localFromWorld);
      }
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
  function pick(owner) {
    if (!owner || menu.hidden || piece || (selected && owner.group!==selected)) return false;
    onSelect(owner.group.key, selected ? String(owner.index) : '');
    return true;
  }
  return {
    direction:direction.toArray(),layout,select,pick,ownerOf:mesh=>meshOwner.get(mesh),
    hash:'exploded',intro:{hold:1200,duration:2100},transitionDuration:1450,
    // The first body panels reach their targets before the interior. Follow
    // their progress to keep those outer pieces in frame during the transition.
    cameraAmount(amount){return Math.max(amount,T.MathUtils.smoothstep(amount,0,.84));},
    get detailed(){return Boolean(selected);},
    apply(amount){apply(amount);menu.hidden=amount<.999;}
  };
}
