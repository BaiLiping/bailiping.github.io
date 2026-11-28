// Studio reflection strips illuminate the rear haunch and the break in the lower door.
export const lighting = {
  environment: 2.2,
  hemisphere: .40,
  key: 1.25,
  keyPosition: [-2, 5, 4],
  fill: .34,
  strip(azimuth, elevation, smooth) {
    const band = (center, width, roll) => 1 - smooth(Math.abs(elevation-center), width, width+roll);
    const side = Math.abs(Math.sin(azimuth));
    const ceiling = .11 + .24*smooth(elevation, -.02, 1.4);
    const doors = 4*band(.49,.105,.105)*smooth(side,.16,.86);
    const sills = 2.2*band(.09,.035,.06)*smooth(side,.35,.90);
    const crown = 4.6*band(1.09,.19,.18)*(.32+.68*Math.cos(azimuth-.5)**2);
    const nose = 2*band(.37,.12,.12)*smooth(Math.cos(azimuth-.38),.86,.997);
    return (ceiling+doors+sills+crown+nose)*41;
  }
};
// Offsets in vehicle coordinates (metres): longitudinal, up and left.
const groups = {
  Front_fascia: ['Front fascia and lamps', [-2.5, .7, 0]],
  Front_closure: ['Hood and extractors', [-1.35, 3.6, 0]],
  Windshield: ['Windshield and wipers', [-.65, 3.4, 0]],
  Roof_panel: ['Removable roof panel', [0, 4.8, 0]],
  Body: ['Fixed body shell', [0, 2.15, 0]],
  Underfloor: ['Underfloor panel', [0, -2.1, 0]],
  Front_suspension: ['Front suspension', [-1.7, -1.4, -.35]],
  Rear_suspension: ['Rear suspension', [1.7, -1.4, -.35]],
  Door_L: ['Left door and mirror', [0, 1.65, 2.35]],
  Door_R: ['Right door and mirror', [0, 1.65, -2.35]],
  Rear_closure: ['Engine hatch', [1.6, 3.65, 0]],
  Carbon_Aero: ['Rear wing', [2.5, 3.25, 0]],
  Rear_fascia: ['Rear fascia and diffuser', [2.35, 1, 0]],
  Cockpit: ['Cabin structure and trim', [0, .6, 0]],
  Seats: ['Seats', [.5, 1.3, 1.15]],
  Dashboard: ['Instrument panel and displays', [-1.25, 1.2, -.45]],
  Steering: ['Steering wheel and controls', [-1.5, 1.2, 1.4]],
  Console: ['Center console', [-.4, .8, 1.3]],
  Front_electric_drive: ['Front electric drive', [-1.1, -.4, 0]],
  Tunnel_battery: ['Tunnel battery', [-.25, -.8, .75]],
  LT7_engine: ['LT7 twin-turbo V8', [1.15, -.45, .15]],
  Rear_transaxle: ['Dual-clutch transaxle', [2.2, -.45, 0]],
};

for (const axle of ['Front', 'Rear']) {
  for (const side of ['left', 'right']) {
    const id = axle + '_' + side;
    const name = axle + ' ' + side;
    const x = axle === 'Front' ? -.95 : .95;
    const sign = side === 'left' ? 1 : -1;
    groups[id] = [name + ' wheel and tire', [x, 0, 2.8 * sign]];
    groups[id + '_brake'] = [name + ' brake', [x, .05, 1.5 * sign]];
  }
}

const label = object => (object.userData.name || object.name).replaceAll('_', ' ');
function groupId(root, object) {
  const name = label(object).toLowerCase();
  if (/^(Front|Rear)_(left|right)$/.test(root.name)) {
    return root.name + (/rotor|caliper|brake/.test(name) ? '_brake' : '');
  }
  if (root.name === 'Cockpit') {
    if (/seat|bucket|head restraint/.test(name)) return 'Seats';
    if (/steering|spoke|horn|paddle shift|column shift|flag|roller|upshift/.test(name)) return 'Steering';
    if (/dashboard|instrument|screen perimeter|driver display|hybrid readout|infotainment/.test(name)) return 'Dashboard';
    if (/console|climate|selector|cupholder|mode switch|tunnel trim/.test(name)) return 'Console';
  }
  if (root.name === 'Powertrain') {
    if (groups[object.name]) return object.name;
    if (/battery/.test(name)) return 'Tunnel_battery';
    if (/front drive/.test(name)) return 'Front_electric_drive';
    if (/transaxle/.test(name)) return 'Rear_transaxle';
    return 'LT7_engine';
  }
  return root.name;
}

export function createAssemblyStudy(T, gltf, camera, onSelect) {
  const model = gltf.scene;
  const homeDirection = new T.Vector3(-.62, .52, 1).normalize();
  const direction = homeDirection.clone();
  const right = new T.Vector3(0, 1, 0).cross(direction).normalize();
  const up = direction.clone().cross(right);
  const axes = [right, up, direction];
  const owner = new WeakMap();
  const byGroup = new Map(), records = [], offset = new T.Vector3();
  const menu = document.getElementById('assembly-picker');
  const groupSelect = document.getElementById('assembly-group');
  const pieceSelect = document.getElementById('assembly-piece');
  let group = '', piece = '', active = false, spread = 0;
  model.updateMatrixWorld(true);

  function register(id, object, section = '') {
    const nodeIndex = gltf.parser.associations.get(object)?.nodes;
    if (!object.isMesh && gltf.parser.json.nodes[nodeIndex]?.mesh == null) {
      const heading = section || label(object);
      for (const child of object.children) register(id, child, heading);
      return;
    }
    // A multi-material glTF mesh may load as a group of primitives. Keep its
    // material surfaces together and move the original named component once.
    let box;
    const corners = [];
    object.traverse(mesh => {
      if (!mesh.isMesh) return;
      if (!mesh.geometry.boundingBox) mesh.geometry.computeBoundingBox();
      if (!box) box = mesh.geometry.boundingBox.clone().makeEmpty();
      const {min, max} = mesh.geometry.boundingBox;
      for (const x of [min.x, max.x]) for (const y of [min.y, max.y]) for (const z of [min.z, max.z]) {
        const world = new T.Vector3(x, y, z).applyMatrix4(mesh.matrixWorld);
        corners.push(world);
        box.expandByPoint(new T.Vector3(...axes.map(axis => axis.dot(world))));
      }
    });
    if (!box || box.isEmpty()) return;
    if (!byGroup.has(id)) byGroup.set(id, []);
    const inverse = object.parent.matrixWorld.clone().invert();
    const record = {
      id, object, section, box, corners, original: object.position.clone(), visible: object.visible,
      inverse: object.normalMatrix.clone().setFromMatrix4(inverse), delta: new T.Vector3()
    };
    object.traverse(mesh => { if (mesh.isMesh) owner.set(mesh, record); });
    byGroup.get(id).push(record);
    records.push(record);
  }
  for (const root of model.children) {
    if (root.isMesh) { register(root.name, root); continue; }
    for (const object of root.children) {
      const id = groupId(root, object);
      if (root.name === 'Powertrain' && groups[object.name] && !object.isMesh) {
        for (const child of object.children) register(id, child);
      } else register(id, object);
    }
  }
  const groupLabel = id => groups[id]?.[0] || id.replaceAll('_', ' ');
  const categories = new Map();
  for (const id of Object.keys(groups).concat([...byGroup.keys()].filter(id => !groups[id]))) {
    if (!byGroup.has(id)) continue;
    let category = 'Body and glass';
    if (['Cockpit', 'Seats', 'Steering', 'Dashboard', 'Console'].includes(id)) category = 'Cabin';
    if (['LT7_engine', 'Rear_transaxle', 'Front_electric_drive', 'Tunnel_battery'].includes(id)) category = 'Hybrid powertrain';
    if (id.includes('suspension') || /^(Front|Rear)_(left|right)/.test(id)) category = 'Suspension, wheels and brakes';
    if (!categories.has(category)) {
      const heading = document.createElement('optgroup');
      heading.label = category;
      categories.set(category, heading);
    }
    categories.get(category).append(new Option(groupLabel(id), id));
  }
  groupSelect.append(...categories.values());
  groupSelect.options[0].textContent = 'Whole vehicle (' + byGroup.size + ' assemblies)';
  groupSelect.onchange = () => onSelect(groupSelect.value, '');
  pieceSelect.onchange = () => onSelect(group, pieceSelect.value);

  // The menus provide the same inspection path as selecting geometry.
  const canvas = document.getElementById('stage');
  const hint = document.getElementById('part-name');
  const ray = new T.Raycaster(), pointer = new T.Vector2();
  let tap;
  function hit(e) {
    if (spread !== 1) return;
    const rect = canvas.getBoundingClientRect();
    pointer.set(2*(e.clientX-rect.left)/rect.width-1, 1-2*(e.clientY-rect.top)/rect.height);
    ray.setFromCamera(pointer, camera);
    for (const intersection of ray.intersectObject(model, true)) {
      const record = owner.get(intersection.object);
      if (record?.object.visible) return record;
    }
  }
  canvas.addEventListener('pointerdown', e => {
    tap = e.isPrimary && spread === 1 ? [e.clientX, e.clientY] : null;
    hint.hidden = true;
  });
  canvas.addEventListener('pointermove', e => {
    if (tap && Math.hypot(e.clientX-tap[0],e.clientY-tap[1]) > 5) tap = null;
    if (e.buttons || e.pointerType !== 'mouse' || spread !== 1) return;
    const record = hit(e);
    hint.hidden = !record;
    if (record) {
      hint.textContent = group ? (record.section && !piece ? record.section : label(record.object)) : groupLabel(record.id);
      const rect = canvas.getBoundingClientRect();
      hint.style.left = Math.max(4, Math.min(rect.width-hint.offsetWidth-4, e.clientX-rect.left+12)) + 'px';
      hint.style.top = Math.max(4, Math.min(rect.height-hint.offsetHeight-4, e.clientY-rect.top+12)) + 'px';
    }
  });
  canvas.addEventListener('pointerleave', () => { hint.hidden = true; });
  canvas.addEventListener('pointercancel', () => { tap = null; });
  canvas.addEventListener('pointerup', e => {
    if (!tap || spread !== 1) return;
    tap = null;
    const record = hit(e);
    if (!record) return;
    if (!group) onSelect(record.id, '');
    else if (!piece && record.section) onSelect(group, 'section:' + record.section);
    else onSelect(group, String(byGroup.get(group).indexOf(record)));
  });
  function select(id, index) {
    hint.hidden = true;
    if (group === id && piece === index) return;
    if (group !== id) {
      direction.copy(homeDirection);
      if (id.includes('right') || id === 'Door_R') direction.z *= -1;
      if (['Rear_fascia', 'Rear_closure','Rear_transaxle','Steering','Dashboard'].includes(id)) direction.x = .95;
      if (id === 'Underfloor') direction.y = -.8;
      direction.normalize();
      right.set(0,1,0).cross(direction).normalize();
      up.copy(direction).cross(right);
      for (const r of (id ? byGroup.get(id) : records)) {
        r.box.makeEmpty();
        for (const world of r.corners) r.box.expandByPoint(new T.Vector3(...axes.map(axis => axis.dot(world))));
      }
      const count = byGroup.get(id)?.length || 0;
      pieceSelect.replaceChildren(new Option('All component parts (' + count + ')', ''));
      const headings = new Map();
      for (const [i, record] of (byGroup.get(id) || []).entries()) {
        if (record.section && !headings.has(record.section)) {
          const heading = document.createElement('optgroup');
          heading.label = record.section;
          heading.append(new Option('Explode ' + record.section, 'section:' + record.section));
          headings.set(record.section, heading);
          pieceSelect.append(heading);
        }
        const container = headings.get(record.section) || pieceSelect;
        container.append(new Option(label(record.object), String(i)));
      }
    }
    group = id;
    piece = index;
    groupSelect.value = id;
    pieceSelect.value = index;
    pieceSelect.parentElement.hidden = !group;
    for (const record of records) {
      record.object.visible = record.visible && (!id || record.id === id)
        && (!index || index === 'section:' + record.section || record === byGroup.get(id)[Number(index)]);
    }
  }

  function layout(aspect) {
    const clusters = [];
    function cluster(members, shift) {
      const box = members[0].box.clone();
      for (const record of members) box.union(record.box);
      const center = box.getCenter(new T.Vector3());
      const size = box.getSize(new T.Vector3());
      clusters.push({members, center, size, wanted: center.clone().add(shift)});
    }
    if (group) {
      const members = byGroup.get(group);
      const bundles = new Map();
      for (const record of members) {
        if (!record.object.visible) continue;
        const key = (!piece && record.section) || record;
        if (!bundles.has(key)) bundles.set(key, []);
        bundles.get(key).push(record);
      }
      for (const bundle of bundles.values()) cluster(bundle, new T.Vector3());
    } else {
      for (const [id, members] of byGroup) {
        const shift = new T.Vector3(...(groups[id]?.[1] || [0, 1, 0]));
        cluster(members, new T.Vector3(...axes.map(axis => shift.dot(axis))));
      }
    }
    const biggest = Math.max(...clusters.map(c => Math.max(c.size.x, c.size.y)));
    const gap = biggest * (group ? .045 : .115);
    const portrait = Math.min(1.6, Math.max(.67, Math.sqrt(aspect)));
    const placed = [];
    // Pack projected bounds around each part's physical removal direction. The
    // outward shift opens up the body; packing keeps the far side legible too.
    clusters.sort((a,b) => b.size.x*b.size.y - a.size.x*a.size.y);
    for (const c of clusters) {
      const hw = c.size.x / 2 + gap / 2, hh = c.size.y / 2 + gap / 2;
      const x0 = c.wanted.x * portrait, y0 = c.wanted.y / portrait;
      const xEdges = [x0], yEdges = [y0];
      for (const p of placed) {
        xEdges.push(p.xmin-hw, p.xmax+hw);
        yEdges.push(p.ymin-hh, p.ymax+hh);
      }
      xEdges.sort((a,b) => Math.abs(a-x0)-Math.abs(b-x0));
      yEdges.sort((a,b) => Math.abs(a-y0)-Math.abs(b-y0));
      let winner, best = Infinity;
      for (const x of xEdges) {
        const dx2 = ((x-x0)/portrait)**2;
        if (dx2 >= best) break;
        for (const y of yEdges) {
          const distance = dx2 + ((y-y0)*portrait)**2;
          if (distance >= best) break;
          const rect = {xmin:x-hw, xmax:x+hw, ymin:y-hh, ymax:y+hh};
          if (placed.some(p => rect.xmin < p.xmax-1e-7 && rect.xmax > p.xmin+1e-7 &&
                               rect.ymin < p.ymax-1e-7 && rect.ymax > p.ymin+1e-7)) continue;
          winner = {x,y,rect}; best = distance;
        }
      }
      placed.push(winner.rect);
      // A shallow relief preserves the 3D order with room for perspective
      // foreshortening. In a detail view the small parts share a display plane.
      const depth = group ? 0 : c.wanted.z * .25;
      offset.copy(right).multiplyScalar(winner.x-c.center.x)
        .addScaledVector(up, winner.y-c.center.y)
        .addScaledVector(direction, depth-c.center.z);
      for (const record of c.members) record.delta.copy(offset).applyMatrix3(record.inverse);
    }
  }
  function apply(amount) {
    spread = amount;
    if (amount !== 1) hint.hidden = true;
    for (const r of records) r.object.position.copy(r.original).addScaledVector(r.delta, amount);
    if (active === (amount > .001)) return;
    active = amount > .001;
    menu.hidden = !active;
    document.getElementById('explode').textContent = active ? 'Assemble' : 'Exploded view';
  }
  return {get direction() { return direction.toArray(); }, layout, apply, select, get detailed(){return Boolean(group);}};
}
