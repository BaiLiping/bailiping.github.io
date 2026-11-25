// Use the same exported meshes, materials and WebGL context as the complete car.
const views = {
  cutaway: {
    label: 'Cutaway', title: 'Inside the transaxle',
    description: 'Follow engine torque from the dry clutch through the six-speed gear train to the differential. Internal dimensions are reconstructed.',
    include: name => /^(Gear\d|Reverse_|Clutch_|InputShaft|OutputShaft|Bearing_|Selector_|Final_)/.test(name), angle: [-1.25, 1.05],
  },
  pedals: {
    label: 'Pedal box', title: 'The clutch pedal',
    description: 'The left footrest is fixed. The next pad is the clutch, followed by the brake and accelerator. Press the clutch pedal to move its pushrod.',
    include: name => /^(Pedal_|Hydraulic_(Master|Clevis))/.test(name), angle: [-3.00, 1.24],
  },
  hydraulics: {
    label: 'Pedal linkage', title: 'From the pedal to the clutch',
    description: 'The pedal pushes the master piston. Hydraulic pressure travels along the line to the rear slave cylinder, which moves the fork and release bearing.',
    include: name => /^(Pedal_(Clutch|Rubber_Clutch|Support)|Hydraulic_|Clutch_)/.test(name), angle: [-1.36, 1.08],
  },
  shifter: {
    label: 'Gear lever', title: 'Six-speed H-pattern gear lever',
    description: 'The short lever pivots in the centre console. Its H-pattern places reverse to the left of first gear.',
    include: name => /^(ManualShifter_|Linkage_(Gate|Cabin))/.test(name), angle: [-2.30, .85],
  },
  shiftlinkage: {
    label: 'Shift cables', title: 'From the gear lever to the selector',
    description: 'One cable selects the shift gate. The other engages the gear. Both connect the lever in the centre console to the selector on the transaxle.',
    include: name => /^(ManualShifter_|Linkage_)/.test(name), angle: [-1.15, 1.05],
  },
  gears: {
    label: 'Gear pairs', title: 'Six forward ratios and reverse',
    description: 'Choose a ratio to see its mating teeth in close-up.',
    include: name => /^(Gear\d|Reverse_|InputShaft|OutputShaft|Bearing_|Selector_)/.test(name), angle: [-1.25, .92],
  },
  shafts: {
    label: 'Shafts', title: 'Input and output shafts',
    description: 'Both clutch discs engage the splines on one input shaft. A selected gear pair transmits torque to the parallel output shaft.',
    include: name => /^(InputShaft|OutputShaft|Bearing_|Clutch_FrictionDisc[12]_SplinedHub)/.test(name), angle: [-1.18, 1.0],
  },
  clutch: {
    label: 'Clutch', title: 'Flywheel and dry clutch',
    description: 'Press the pedal to move the release fork and bearing. The diaphragm spring lifts the pressure plate, freeing the two discs on the gearbox input shaft.',
    include: name => /^(Clutch_|InputShaft$)/.test(name), angle: [-.65, 1.04],
  },
  final: {
    label: 'Final drive', title: 'Rear differential and final drive',
    description: 'The pinion turns the crown wheel and differential carrier. Porsche specifies a 4.30:1 rear-axle ratio.',
    include: name => /^Final_/.test(name), angle: [-1.04, .98],
  },
  axles: {
    label: 'Half-shafts', title: 'Drive to the rear wheels',
    description: 'One half-shaft drives each rear wheel through its CV joints. The front wheels steer and roll freely. This rear-wheel-drive GT3 has no front driveshaft.',
    include: name => /^(Final_|Axle_)/.test(name), angle: [.35, .96],
  },
  wheels: {
    label: 'Wheel layout', title: 'Which wheels does the gearbox drive?',
    description: 'The two rear half-shafts run from the differential to the wheel hubs. The front wheels are undriven. The tyres and rims are transparent so you can trace the connection.',
    include: name => /^(Wheel_|Axle_|Gear\d|Clutch_|InputShaft|OutputShaft|Final_)/.test(name), angle: [-1.17, .57],
  },
  cv: {
    label: 'CV joint', title: 'Wheel-side constant-velocity joint',
    description: 'A section through the left rear joint exposes its six balls, cage and grooved races. The splined stub connects to the hub.',
    include: name => /^Axle_JointL_(?!CupSector)/.test(name), angle: [.53, 1.2],
  },
  housing: {
    label: 'Housing', title: 'Manual transaxle housing',
    description: 'The bellhousing encloses the flywheel and clutch. The selector mechanism sits on the ribbed gearbox casting.',
    include: name => /^(Case_|Clutch|Final_|InputShaft|Linkage_(SelectorTower|SelectorShaft|SelectorSeal|ShiftArm|SelectArm))/.test(name), angle: [2.16, 1.12],
  },
};

const ratios = ['3.75', '2.38', '1.72', '1.34', '1.08', '0.88'];
const ordinals = ['First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth'];
const shortName = name => name.replace(/^GT3_(MT_)?/, '').split('__')[0];
export function transmissionPartName(name) {
  const part = shortName(name);
  const gear = part.match(/^Gear0([1-6])_(Driven|Drive)$/);
  if (gear) return ordinals[Number(gear[1]) - 1] + ' gear · ' + (gear[2] === 'Driven' ? 'output wheel' : 'input wheel');
  if (part.startsWith('Wheel_')) {
    const [, corner, piece] = part.match(/^Wheel_(FL|FR|RL|RR)_(.+)$/);
    const position = {FL: 'Left front', FR: 'Right front', RL: 'Left rear', RR: 'Right rear'}[corner];
    return position + ' ' + {Tire: 'tyre', YSpoke_Rim: 'rim', CenterLockHub: 'wheel hub'}[piece];
  }
  const labels = {
    InputShaft: 'Splined gearbox input shaft', OutputShaft: 'Gearbox output shaft',
    Clutch_Flywheel: 'Flywheel', Clutch_FrictionDisc1: 'Engine-side driven disc',
    Clutch_FrictionDisc2: 'Gearbox-side driven disc', Clutch_IntermediatePlate: 'Intermediate plate',
    Clutch_PressurePlate: 'Pressure plate', Clutch_DiaphragmSpring: 'Diaphragm spring',
    Clutch_ReleaseBearing: 'Release bearing', Clutch_ReleaseFork: 'Release fork',
    Pedal_Clutch: 'Clutch pedal', Pedal_Brake: 'Brake pedal', Pedal_Accelerator: 'Accelerator',
    Pedal_Footrest: 'Driver’s footrest',
    Final_Crown: 'Differential crown wheel', Final_Pinion: 'Final-drive pinion',
    Reverse_Drive: 'Reverse input pinion', Reverse_Driven: 'Reverse output wheel',
    Reverse_Idler: 'Reverse idler gear', Reverse_IdlerStud: 'Reverse idler spindle',
    Axle_JointL_Balls: 'CV joint · six drive balls', Axle_JointL_Cage: 'CV joint · ball cage',
    Axle_JointL_InnerRace: 'CV joint · grooved inner star', Axle_JointL_OuterRace: 'CV joint · outer race',
    Axle_JointL_HubSpline: 'Splined wheel-end stub', Axle_JointL_ShaftNose: 'Splined shaft nose',
    Case_WebArc: 'Transverse casting rib', Case_WebLong: 'Longitudinal casting rib',
    Case_Bell: 'Engine-side bellhousing', Case_Centre: 'Centre housing', Case_Tail: 'Transmission tail housing',
  };
  if (labels[part]) return labels[part];
  return part.replace(/^Case_/, 'Housing · ').replace(/^Final_/, 'Final drive · ')
    .replace(/^Axle_/, 'Half-shaft · ').replace(/([a-z])([A-Z])/g, '$1 $2')
    .replaceAll('_', ' ').replace(/CV(?=[A-Z])/g, 'CV ').trim();
}

export function createTransmissionInspector(THREE, renderer, environment, callbacks) {
  const dialog = document.getElementById('transmission');
  const viewport = document.getElementById('mt-viewport');
  const launch = document.getElementById('mt-open');
  const tabRow = document.getElementById('mt-views');
  const compact = document.getElementById('mt-compact');
  const gears = document.getElementById('mt-gear');
  const readout = document.getElementById('mt-readout');
  const canvas = renderer.domElement;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xeef1f3);
  scene.environment = environment;
  scene.environmentIntensity = .83;
  let softboxTexture;
  scene.add(new THREE.HemisphereLight(0xf4f7ff, 0x39414c, .38));
  const keyLight = new THREE.DirectionalLight(0xfff3df, 2.1);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.set(2048, 2048);
  keyLight.shadow.bias = -.00015;
  scene.add(keyLight, keyLight.target);
  const rim = new THREE.DirectionalLight(0xdaeaff, 1.1);
  rim.position.set(3, 4, 3);
  scene.add(rim);

  // Dark walls and broad reflective strips make the curvature of steel visible.
  // The environment is generated once, on the first transmission inspection.
  function lightStudio() {
    const room = new THREE.Scene();
    room.background = new THREE.Color(0x58616a);
    const cards = [
      [[-3, 5, -2], 8, 2.8, 0xfff5df, 4],
      [[2, 2, 5], 8, 1.4, 0xe0edff, 4],
      [[4, 1, -3], 1.6, 7, 0xf2f6ff, 2.8],
      [[-5, -.5, 2], 4, 4, 0xb5c0ca, 1.2],
      [[0, -5, 0], 8, 6, 0x78848c, .7],
    ];
    for (const [position, width, height, color, power] of cards) {
      const material = new THREE.MeshStandardMaterial({color: 0, emissive: color, emissiveIntensity: power});
      const card = new THREE.Mesh(new THREE.PlaneGeometry(width, height), material);
      card.position.fromArray(position);
      card.lookAt(0, 0, 0);
      room.add(card);
    }
    const generator = new THREE.PMREMGenerator(renderer);
    softboxTexture = generator.fromScene(room, .035);
    scene.environment = softboxTexture.texture;
    generator.dispose();
    for (const card of room.children) {
      card.geometry.dispose();
      card.material.dispose();
    }
  }

  const camera = new THREE.PerspectiveCamera(30, 1, .05, 2000);
  const center = new THREE.Vector3();
  const eye = new THREE.Vector3();
  const right = new THREE.Vector3();
  const up = new THREE.Vector3();
  const ray = new THREE.Raycaster();
  const parts = [];
  const visible = [];
  let bounds = [];
  let selection = 'cutaway';
  let theta, phi, baseDistance = 1, zoom = 1;
  let previousWidth = 0, previousHeight = 0;
  let frame = 0, idle = 0, moving = false;

  const clutchButtons = [...document.querySelectorAll('[data-clutch]')];
  const motions = [];
  const diaphragmMeshes = [];
  let pedalDown = false, stroke = 0, pedalFrame = 0;

  function bindMotion(pivot) {
    const {clutchChannel: channel, clutchAxis: axis, clutchTravel: travel} = pivot.userData;
    motions.push({pivot, channel, axis, travel, rest: pivot[channel][axis]});
  }

  function setClutch(down) {
    if (pedalDown === down) return;
    pedalDown = down;
    for (const button of clutchButtons) {
      button.setAttribute('aria-pressed', String(down));
      button.textContent = down ? 'Release clutch pedal' : 'Press clutch pedal';
    }
    cancelAnimationFrame(pedalFrame);
    const from = stroke, to = Number(down), start = performance.now();
    const duration = matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : 370 * Math.abs(to - from);
    function advance(now) {
      const t = duration ? Math.min(1, (now - start) / duration) : 1;
      stroke = from + (to - from) * t * t * (3 - 2 * t);
      for (const {pivot, channel, axis, travel, rest} of motions) pivot[channel][axis] = rest + stroke * travel;
      for (const mesh of diaphragmMeshes) mesh.morphTargetInfluences[0] = stroke;
      renderer.shadowMap.needsUpdate = true;
      if (dialog.open) redraw(); else callbacks.poseChanged();
      if (t < 1) pedalFrame = requestAnimationFrame(advance);
    }
    advance(start);
  }
  for (const button of clutchButtons) button.addEventListener('click', () => setClutch(!pedalDown));

  function register(meshes) {
    const inspectionPivots = new Map();
    const wheelOutline = new THREE.MeshStandardMaterial({color: 0x394b56, roughness: .7, transparent: true, opacity: .15, depthWrite: false});
    for (const mesh of meshes) {
      const wheel = /^GT3_(Tire|YSpoke_Rim|CenterLockHub)_(FL|FR|RL|RR)_/.exec(mesh.name);
      const name = mesh.userData.transmissionPart || (wheel && 'Wheel_' + wheel[2] + '_' + wheel[1]);
      if (!name) continue;
      const translucent = wheel && wheel[1] !== 'CenterLockHub';
      const part = new THREE.Mesh(mesh.geometry, translucent ? wheelOutline : mesh.material);
      part.name = name;
      part.castShadow = part.receiveShadow = !translucent;
      part.matrixAutoUpdate = false;
      const parent = mesh.parent;
      if (parent.userData.clutchChannel) {
        let pivot = inspectionPivots.get(parent);
        if (!pivot) {
          pivot = parent.clone(false);
          parent.matrixWorld.decompose(pivot.position, pivot.quaternion, pivot.scale);
          scene.add(pivot);
          inspectionPivots.set(parent, pivot);
          bindMotion(parent);
          bindMotion(pivot);
        }
        part.matrix.copy(mesh.matrix);
        pivot.add(part);
      } else {
        part.matrix.copy(mesh.matrixWorld);
        scene.add(part);
      }
      if (name === 'Clutch_DiaphragmSpring' && mesh.morphTargetInfluences?.length) {
        mesh.morphTargetInfluences[0] = part.morphTargetInfluences[0] = stroke;
        diaphragmMeshes.push(mesh, part);
      }
      parts.push(part);
    }
    for (const button of clutchButtons) button.disabled = !motions.length;
    launch.disabled = !parts.length;
  }

  function eyeDirection(t, p) {
    return eye.set(Math.sin(p) * Math.cos(t), Math.cos(p), Math.sin(p) * Math.sin(t));
  }

  function fitDistance() {
    // Fit the projected bounds of individual parts rather than a sphere around the axles.
    const [t, p] = currentAngle();
    eyeDirection(t, p);
    orientCamera(eye, 1);
    right.set(1, 0, 0).applyQuaternion(camera.quaternion);
    up.set(0, 1, 0).applyQuaternion(camera.quaternion);
    const tanV = Math.tan(camera.fov * Math.PI / 360) * .87;
    const tanH = tanV * camera.aspect;
    baseDistance = 1;
    for (const point of bounds) {
      const depth = point.dot(eye);
      baseDistance = Math.max(baseDistance, depth + Math.abs(point.dot(right)) / tanH,
        depth + Math.abs(point.dot(up)) / tanV);
    }
  }

  function orientCamera(direction, distance) {
    camera.position.copy(center).addScaledVector(direction, distance);
    camera.up.set(0, 1, 0);
    camera.lookAt(center);
    if (camera.aspect < .95 && ['cutaway', 'shafts', 'axles', 'housing', 'gears', 'hydraulics', 'shiftlinkage', 'wheels'].includes(selection)
        && !(selection === 'gears' && gears.value)) {
      camera.rotateZ(-THREE.MathUtils.clamp((.95 - camera.aspect) * 3.5, 0, 1.10));
    }
  }

  function draw() {
    frame = 0;
    if (!dialog.open) return;
    const w = viewport.clientWidth, h = viewport.clientHeight;
    if (!w || !h) return;
    const ratio = Math.min(devicePixelRatio, moving ? 1.25 : 3, Math.sqrt(12000000 / (w * h)));
    if (w !== previousWidth || h !== previousHeight) {
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h, false);
      previousWidth = w;
      previousHeight = h;
      fitDistance();
    }
    if (renderer.getPixelRatio() !== ratio) renderer.setPixelRatio(ratio);
    orientCamera(eyeDirection(theta, phi), baseDistance * zoom);
    renderer.toneMappingExposure = .98;
    renderer.render(scene, camera);
  }
  function redraw(interacting = false) {
    if (!dialog.open) return;
    if (interacting) {
      moving = true;
      clearTimeout(idle);
      idle = setTimeout(() => { moving = false; redraw(); }, 120);
    }
    if (!frame) frame = requestAnimationFrame(draw);
  }
  const observer = new ResizeObserver(() => redraw());
  observer.observe(viewport);

  function currentAngle() {
    return selection === 'gears' && gears.value ? [-.48, 1.25] : views[selection].angle;
  }
  function fit() {
    [theta, phi] = currentAngle();
    zoom = 1;
    fitDistance();
    redraw();
  }

  function show(key) {
    selection = key;
    const view = views[key];
    compact.value = key;
    const reference = document.getElementById('mt-reference');
    const [referenceName, referenceUrl] = {
      housing: ['Manual gearbox photograph', 'https://i.ebayimg.com/images/g/c0MAAeSwcFNqrSbU/s-l1600.jpg'],
      shifter: ['Porsche shifter photograph', 'https://content-hub.imgix.net/1xG1WBxs1p33uKAXCwRv4h/4f30739a1fdb792be458c4401be01a18/2025_911_GT3_4.jpg?w=2000'],
      pedals: ['Porsche manual cockpit', 'https://newsroom.porsche.com/dam/jcr%3Aa09c5ee8-5353-4c1e-9460-5bb72bb80f51/Vehicle%20information%20911%20GT3%20TP%20S-GO%205007.pdf#page=13'],
      clutch: ['Clutch and fork diagram', 'https://www.boutiqueporschepoitiers.fr/pieces/911/GT3-4/2025/301-200/'],
      hydraulics: ['Clutch hydraulics diagram', 'https://www.boutiqueporschepoitiers.fr/pieces/911/GT3-4/2025/703-000/'],
      shiftlinkage: ['Gear lever and cables diagram', 'https://www.boutiqueporschepoitiers.fr/pieces/911/GT3-4/2025/701-000/'],
      cv: ['CV joint drawing', 'https://patents.google.com/patent/US7147565B2/en'],
    }[key] || ['Porsche technical data', 'https://newsroom.porsche.com/dam/jcr:46cb0e24-ad5a-489c-a404-52c678071d03/pag-911-gt3-mt-en.pdf#page=2'];
    reference.textContent = referenceName;
    reference.href = referenceUrl;
    gears.hidden = key !== 'gears';
    document.getElementById('mt-clutch').hidden = !['cutaway', 'pedals', 'hydraulics', 'clutch'].includes(key);
    for (const button of tabRow.children) button.setAttribute('aria-pressed', String(button.value === key));
    let predicate = view.include, title = view.title, caption = view.description;
    if (key === 'gears' && gears.value) {
      const reverse = gears.value === 'R';
      const number = Number(gears.value);
      const prefix = reverse ? 'Reverse_' : 'Gear0' + number + '_';
      predicate = name => name.startsWith(prefix);
      title = reverse ? 'Reverse gear' : ordinals[number - 1] + ' gear';
      caption = reverse ? 'Porsche ratio 3.42:1. The intermediate idler reverses the direction of rotation.'
        : 'Porsche ratio ' + ratios[number - 1] + ':1 · Input-shaft turns per output-shaft turn.';
    }

    visible.length = 0;
    bounds = [];
    center.set(0, 0, 0);
    const min = new THREE.Vector3(Infinity, Infinity, Infinity);
    const max = new THREE.Vector3(-Infinity, -Infinity, -Infinity);
    scene.updateMatrixWorld(true);
    for (const part of parts) {
      part.visible = predicate(shortName(part.name));
      if (!part.visible) continue;
      visible.push(part);
      if (key === 'clutch' && part.name === 'InputShaft') continue;
      const geo = part.geometry;
      const positions = geo.getAttribute('position');
      const stride = Math.max(1, Math.floor(positions.count / 900));
      for (let i = 0; i < positions.count; i += stride) {
        const point = new THREE.Vector3().fromBufferAttribute(positions, i).applyMatrix4(part.matrixWorld);
        bounds.push(point);
        min.min(point);
        max.max(point);
      }
    }
    if (bounds.length) center.addVectors(min, max).multiplyScalar(.5);
    for (const point of bounds) point.sub(center);
    const span = Math.max(1, max.distanceTo(min));
    keyLight.target.position.copy(center);
    keyLight.position.set(-.55, 1, -.7).multiplyScalar(span).add(center);
    const shadowCamera = keyLight.shadow.camera;
    shadowCamera.left = shadowCamera.bottom = -.52 * span;
    shadowCamera.right = shadowCamera.top = .52 * span;
    shadowCamera.near = .35 * span;
    shadowCamera.far = 2.6 * span;
    keyLight.shadow.normalBias = span * .00015;
    shadowCamera.updateProjectionMatrix();
    renderer.shadowMap.needsUpdate = true;
    document.getElementById('mt-part-title').textContent = title;
    document.getElementById('mt-description').textContent = caption;
    readout.textContent = 'Drag to rotate · Select a part to identify it';
    fit();
  }

  for (const [key, view] of Object.entries(views)) {
    const button = document.createElement('button');
    button.type = 'button';
    button.value = key;
    button.textContent = view.label;
    button.addEventListener('click', () => show(key));
    tabRow.append(button);
    compact.add(new Option(view.label, key));
  }
  compact.addEventListener('change', () => show(compact.value));
  gears.add(new Option('All gear pairs', ''));
  for (const [i, ratio] of ratios.entries()) gears.add(new Option(ordinals[i] + ' · ' + ratio + ':1', String(i + 1)));
  gears.add(new Option('Reverse · 3.42:1', 'R'));
  gears.addEventListener('change', () => show('gears'));

  const returnPoint = document.createComment('Car canvas');
  function open(initialView = 'cutaway') {
    if (!parts.length || dialog.open) return;
    callbacks.enter();
    if (!softboxTexture) lightStudio();
    canvas.before(returnPoint);
    viewport.append(canvas);
    dialog.showModal();
    previousWidth = previousHeight = 0;
    gears.value = '';
    show(initialView);
    history.replaceState(history.state, '', initialView === 'pedals' ? '#clutch-pedal' : '#transmission');
  }
  launch.addEventListener('click', () => open());
  document.getElementById('mt-close').addEventListener('click', () => dialog.close());
  dialog.addEventListener('close', () => {
    cancelAnimationFrame(frame);
    clearTimeout(idle);
    frame = 0;
    moving = false;
    pointers.clear();
    returnPoint.replaceWith(canvas);
    if (['#transmission', '#clutch-pedal'].includes(location.hash)) history.replaceState(history.state, '', location.pathname + location.search);
    callbacks.leave();
    launch.focus({preventScroll: true});
  });
  window.addEventListener('hashchange', () => {
    if (['#transmission', '#clutch-pedal'].includes(location.hash)) {
      const key = location.hash === '#clutch-pedal' ? 'pedals' : 'cutaway';
      if (dialog.open) show(key); else open(key);
    }
    else if (dialog.open) dialog.close();
  });

  function scale(factor) {
    zoom = THREE.MathUtils.clamp(zoom * factor, .14, 4);
    redraw(true);
  }
  function orbit(dx, dy) {
    theta -= dx * .008;
    phi = THREE.MathUtils.clamp(phi - dy * .008, .05, Math.PI - .05);
    redraw(true);
  }
  function identify(x, y) {
    const rect = viewport.getBoundingClientRect();
    const pointer = new THREE.Vector2((x - rect.left) / rect.width * 2 - 1, 1 - (y - rect.top) / rect.height * 2);
    ray.setFromCamera(pointer, camera);
    const hit = ray.intersectObjects(visible, false)[0];
    readout.textContent = hit ? transmissionPartName(hit.object.name) : 'Drag to rotate · Select a part to identify it';
  }
  const pointers = new Map();
  let pressX, pressY, dragged, pinch = 0;
  function separation() {
    const [a, b] = pointers.values();
    return Math.hypot(a.x - b.x, a.y - b.y);
  }
  viewport.addEventListener('pointerdown', e => {
    if (e.button !== 0) return;
    viewport.focus({preventScroll: true});
    viewport.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.size === 1) {
      pressX = e.clientX; pressY = e.clientY; dragged = false;
    } else {
      dragged = true;
      pinch = separation();
    }
  });
  viewport.addEventListener('pointermove', e => {
    const last = pointers.get(e.pointerId);
    if (!last) return;
    const dx = e.clientX - last.x, dy = e.clientY - last.y;
    last.x = e.clientX; last.y = e.clientY;
    if (pointers.size === 2) {
      const distance = separation();
      if (pinch > 0 && distance > 0) scale(pinch / distance);
      pinch = distance;
    } else if (pointers.size === 1) {
      if (Math.hypot(last.x - pressX, last.y - pressY) > 4) dragged = true;
      if (dragged) orbit(dx, dy);
    }
  });
  viewport.addEventListener('pointerup', e => {
    if (pointers.delete(e.pointerId) && !dragged) identify(e.clientX, e.clientY);
  });
  viewport.addEventListener('lostpointercapture', e => pointers.delete(e.pointerId));
  viewport.addEventListener('wheel', e => {
    e.preventDefault();
    scale(Math.exp(THREE.MathUtils.clamp(e.deltaY * .0012, -1, 1)));
  }, {passive: false});
  viewport.addEventListener('keydown', e => {
    const movements = { ArrowLeft: [-9, 0], ArrowRight: [9, 0], ArrowUp: [0, -9], ArrowDown: [0, 9] };
    if (movements[e.key]) orbit(...movements[e.key]);
    else if (['+', '='].includes(e.key)) scale(.8);
    else if (e.key === '-') scale(1.25);
    else if (e.key === 'Home') fit();
    else return;
    e.preventDefault();
  });
  document.getElementById('mt-in').addEventListener('click', () => scale(.8));
  document.getElementById('mt-out').addEventListener('click', () => scale(1.25));
  document.getElementById('mt-fit').addEventListener('click', fit);
  return { register, open, setClutch, get available() { return parts.length > 0; }, get active() { return dialog.open; } };
}
