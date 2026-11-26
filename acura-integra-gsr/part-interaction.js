import * as THREE from '../honda-civic-type-r/vendor/three-r170.js?v=31c84364';

const assemblyNames = {
  Body: 'Body shell', Hood: 'Hood', Liftback: 'Liftback', DoorLeft: 'Driver’s door',
  DoorRight: 'Passenger door', Powertrain: 'B18C1 engine and gearbox', Interior: 'Cabin',
  WheelFrontLeft: 'Left front wheel', WheelFrontRight: 'Right front wheel',
  WheelRearLeft: 'Left rear wheel', WheelRearRight: 'Right rear wheel',
};
const panelNames = { Hood: 'hood', Liftback: 'hatch', DoorLeft: 'leftDoor', DoorRight: 'rightDoor' };
const partLabels = [
  [/cast face|polished gs-r alloy/i, 'Six-spoke alloy wheel'], [/lug nuts/i, 'Wheel nuts'],
  [/cap insignia/i, 'Wheel centre cap'], [/carcass|tire sidewall/i, 'Tire'], [/shoulder blocks|tread grooves/i, 'Tire tread'],
  [/swept brake surface/i, 'Brake disc'], [/photographed red b18c1 cover/i, 'B18C1 valve cover'],
  [/washer reservoir/i, 'Washer reservoir'], [/mirror glass seal/i, 'Door mirror seal'], [/mirror glass/i, 'Door mirror glass'],
  [/mirror housing/i, 'Door mirror housing'], [/windshield wiper/i, 'Windshield wiper'],
  [/tinted moonroof/i, 'Moonroof glass'], [/rear window heating/i, 'Rear window defroster'],
  [/bumper indicator|amber indicator/i, 'Front indicator'], [/rear combination lamps|tail lamps/i, 'Rear combination lamp'],
  [/reverse lenses/i, 'Reverse lamp'], [/condenser|projector lens/i, 'Headlamp projector'],
  [/reflector flutes/i, 'Headlamp reflector'], [/embossed integra/i, 'Integra bumper lettering'],
];
const pointer = new THREE.Vector2();
const raycaster = new THREE.Raycaster();

export function bindPartInteraction(canvas, camera, model, assemblies, actions) {
  const info = document.querySelector('#callout');
  const title = info.querySelector('.title');
  const hint = info.querySelector('.sub');
  const status = document.querySelector('#part-status');
  const chooser = document.querySelector('#part-choice');
  const moveButton = document.querySelector('#move-part');
  const assemblyByNode = new Map(assemblies.map(part => [part.node, part]));
  const meshInfo = new Map();
  const home = new Map();
  const offsets = new Map();
  const ghosts = new Map();
  const highlights = new Map();
  const pointers = new Map();
  const direction = new THREE.Vector3();
  const point = new THREE.Vector3();
  let separation = 0;
  let xray = false;
  let hover = null;
  let hoverFrame = 0, hoverX = 0, hoverY = 0;
  let active = null;
  let pinchGap = 0;
  let holdTimer, labelTimer, pendingTap;

  for (const { node, rest } of assemblies) {
    home.set(node, rest);
    chooser.add(new Option(assemblyNames[node.name], node.name));
    const shell = node.name === 'Body' || Object.hasOwn(panelNames, node.name);
    node.traverse(mesh => {
      if (!mesh.isMesh) return;
      home.set(mesh, mesh.position.clone());
      const meshName = (mesh.userData.name || mesh.name).replaceAll('_', ' ');
      const label = partLabels.find(([test]) => test.test(meshName))?.[1] || assemblyNames[node.name];
      meshInfo.set(mesh, { assembly: node, label, shell, material: mesh.material, castShadow: mesh.castShadow });
    });
  }
  const meshes = [...meshInfo.keys()];
  chooser.value = 'WheelFrontLeft';

  function rayAt(x, y) {
    const rect = canvas.getBoundingClientRect();
    pointer.set((x - rect.left) * 2 / rect.width - 1, 1 - (y - rect.top) * 2 / rect.height);
    camera.updateMatrixWorld();
    raycaster.setFromCamera(pointer, camera);
    return raycaster.ray;
  }
  function pick(x, y) {
    model.updateMatrixWorld(true);
    rayAt(x, y);
    return raycaster.intersectObjects(meshes, false).find(hit => !(xray && meshInfo.get(hit.object).shell));
  }
  function applyPosition(node) {
    node.position.copy(home.get(node));
    const assembly = assemblyByNode.get(node);
    if (assembly) node.position.addScaledVector(assembly.travel, separation);
    if (offsets.has(node)) node.position.add(offsets.get(node));
  }
  function setSeparation(amount) {
    separation = amount;
    for (const { node } of assemblies) applyPosition(node);
    actions.moved();
  }
  function updateCount() {
    const part = model.getObjectByName(chooser.value);
    moveButton.textContent = offsets.has(part) ? 'Reattach' : 'Detach';
    actions.detached(offsets.size);
  }
  function reattach(node) {
    offsets.delete(node);
    applyPosition(node);
    updateCount();
    actions.moved();
  }
  function reattachAll() {
    for (const node of offsets.keys()) {
      offsets.delete(node);
      applyPosition(node);
    }
    updateCount();
    actions.moved();
  }
  function ghost(base) {
    if (!ghosts.has(base)) {
      const m = base.clone();
      m.transparent = true;
      m.opacity = base.transparent ? .045 : .115;
      m.depthWrite = false;
      m.metalness = .08;
      m.roughness = .55;
      m.envMapIntensity = .34;
      m.color.set(0x90adba);
      m.side = THREE.FrontSide;
      ghosts.set(base, m);
    }
    return ghosts.get(base);
  }
  function materialFor(mesh) {
    const { material, shell } = meshInfo.get(mesh);
    return xray && shell ? ghost(material) : material;
  }
  function clearHover() {
    clearTimeout(labelTimer);
    cancelAnimationFrame(hoverFrame);
    hoverFrame = 0;
    if (hover) {
      hover.material = materialFor(hover);
      hover = null;
      actions.render();
    }
    info.classList.remove('visible');
    canvas.classList.remove('hover-part');
  }
  function setXray(value) {
    if (xray === value) return;
    clearHover();
    xray = value;
    for (const [mesh, data] of meshInfo) {
      if (!data.shell) continue;
      mesh.material = materialFor(mesh);
      mesh.castShadow = !xray && data.castShadow;
    }
    document.querySelector('#xray').setAttribute('aria-pressed', String(xray));
    actions.moved();
  }

  function showPart(hit, x, y, touch = false) {
    if (!hit) { clearHover(); return; }
    clearTimeout(labelTimer);
    const mesh = hit.object;
    const { assembly, label } = meshInfo.get(mesh);
    if (hover !== mesh) {
      clearHover();
      const base = materialFor(mesh);
      if (!highlights.has(base)) {
        const m = base.clone();
        if (m.emissive) { m.emissive.set(0xdb5649); m.emissiveIntensity = .28; }
        highlights.set(base, m);
      }
      mesh.material = highlights.get(base);
      hover = mesh;
      actions.render();
    }
    title.textContent = label;
    const detached = offsets.has(mesh) || offsets.has(assembly);
    const verb = touch ? 'Tap' : 'Click';
    let gesture = touch ? 'Hold and drag to detach' : 'Drag to detach';
    if (detached) gesture = verb + ' to reattach';
    else if (panelNames[assembly.name]) gesture = verb + (actions.isPanelOpen(panelNames[assembly.name]) ? ' to close' : ' to open');
    hint.textContent = label === assemblyNames[assembly.name] ? gesture : assemblyNames[assembly.name] + ' · ' + gesture;
    info.classList.add('visible');
    if (!touch) {
      info.style.left = Math.max(0, Math.min(x, innerWidth - info.offsetWidth - 20)) + 'px';
      info.style.top = Math.max(18, Math.min(y, innerHeight - info.offsetHeight + 5)) + 'px';
    }
    canvas.classList.add('hover-part');
  }
  function queueHover(x, y) {
    hoverX = x; hoverY = y;
    if (!hoverFrame) hoverFrame = requestAnimationFrame(() => {
      hoverFrame = 0;
      if (!pointers.size) showPart(pick(hoverX, hoverY), hoverX, hoverY);
    });
  }

  function preparePull(press) {
    // Leave photographic letterboxing first, then back-project both ends of the gesture
    // onto the same plane. World-to-local also handles a steered or hinged parent.
    actions.leavePhoto();
    const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(camera.getWorldDirection(direction), press.hit.point);
    const intersection = rayAt(press.startX, press.startY).intersectPlane(plane, point);
    if (!intersection) return false;
    press.plane = plane;
    press.anchor = press.node.parent.worldToLocal(intersection.clone());
    press.saved = offsets.get(press.node)?.clone();
    return true;
  }
  function dragPart(press, x, y) {
    if (!rayAt(x, y).intersectPlane(press.plane, point)) return;
    press.node.parent.worldToLocal(point).sub(press.anchor);
    if (press.saved) point.add(press.saved);
    const offset = offsets.get(press.node) || new THREE.Vector3();
    offset.copy(point);
    offsets.set(press.node, offset);
    applyPosition(press.node);
    actions.moved();
  }
  function cancelPull() {
    if (active?.mode !== 'pull') return;
    if (active.saved) offsets.set(active.node, active.saved);
    else offsets.delete(active.node);
    applyPosition(active.node);
    actions.moved();
    updateCount();
  }
  function clearTap() {
    clearTimeout(pendingTap?.timer);
    pendingTap = null;
  }
  function stopGesture() {
    cancelPull();
    clearTimeout(holdTimer);
    active = null;
    for (const id of pointers.keys()) if (canvas.hasPointerCapture(id)) canvas.releasePointerCapture(id);
    pointers.clear();
    pinchGap = 0;
    clearTap();
    clearHover();
    canvas.classList.remove('grabbing');
  }
  function activate(hit, x, y, touch) {
    const mesh = hit.object, assembly = meshInfo.get(mesh).assembly;
    const detached = offsets.has(mesh) ? mesh : offsets.has(assembly) ? assembly : null;
    if (detached) {
      reattach(detached);
      status.textContent = assemblyNames[assembly.name] + ' reattached';
    } else if (panelNames[assembly.name]) {
      actions.panel(panelNames[assembly.name]);
      status.textContent = assemblyNames[assembly.name] + (actions.isPanelOpen(panelNames[assembly.name]) ? ' open' : ' closed');
    } else status.textContent = assemblyNames[assembly.name];
    showPart(hit, x, y, touch);
    if (touch) labelTimer = setTimeout(clearHover, 1700);
  }
  function tap(press, event) {
    const { clientX: x, clientY: y } = event;
    const previous = pendingTap;
    clearTap();
    if (previous && Math.hypot(x - previous.x, y - previous.y) < 12 && previous.type === event.pointerType) {
      actions.leavePhoto();
      setXray(!xray);
      return;
    }
    if (previous) previous.activate();
    if (!press.hit) { clearHover(); return; }
    showPart(press.hit, x, y, press.touch);
    const perform = () => activate(press.hit, x, y, press.touch);
    pendingTap = { x, y, type: event.pointerType, activate: perform, timer: setTimeout(() => { pendingTap = null; perform(); }, 275) };
  }
  function gap() {
    const [a, b] = pointers.values();
    return Math.hypot(a.x - b.x, a.y - b.y);
  }

  canvas.addEventListener('contextmenu', event => event.preventDefault());
  canvas.addEventListener('pointerdown', event => {
    if (![0, 1, 2].includes(event.button)) return;
    if (event.button === 1) event.preventDefault();
    actions.stopIntro();
    canvas.setPointerCapture(event.pointerId);
    pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    clearTimeout(holdTimer);
    if (pointers.size > 1) {
      cancelPull(); clearTap(); clearHover(); active = null; pinchGap = gap(); return;
    }
    const touch = event.pointerType === 'touch';
    const hit = event.button === 0 && !event.shiftKey ? pick(event.clientX, event.clientY) : null;
    const node = hit && (event.altKey || offsets.has(hit.object) ? hit.object : meshInfo.get(hit.object).assembly);
    active = { id: event.pointerId, startX: event.clientX, startY: event.clientY, x: event.clientX, y: event.clientY,
      mode: 'pressed', touch, hit, node, held: false };
    if (touch && hit && !actions.inCabin()) {
      holdTimer = setTimeout(() => { if (active?.mode === 'pressed') { active.held = true; showPart(hit, active.x, active.y, true); } }, 350);
    }
  });
  canvas.addEventListener('pointermove', event => {
    const { clientX: x, clientY: y } = event;
    if (!pointers.has(event.pointerId)) {
      if (!pointers.size && event.pointerType !== 'touch') queueHover(x, y);
      return;
    }
    pointers.set(event.pointerId, { x, y });
    if (pointers.size > 1) {
      const next = Math.max(1, gap());
      if (pinchGap > 0) actions.zoom(pinchGap / next);
      pinchGap = next;
      return;
    }
    if (active?.id !== event.pointerId) return;
    if (active.mode === 'pressed') {
      if (Math.hypot(x - active.startX, y - active.startY) < (active.touch ? 8 : 5)) return;
      clearTimeout(holdTimer); clearTap(); clearHover();
      const pull = active.hit && (!active.touch || active.held) && (!actions.inCabin() || event.altKey);
      active.mode = pull && preparePull(active) ? 'pull' : 'orbit';
      canvas.classList.add('grabbing');
    }
    if (active.mode === 'pull') dragPart(active, x, y);
    else actions.orbit(x - active.x, y - active.y);
    active.x = x; active.y = y;
  });
  function release(event, cancelled) {
    if (!pointers.delete(event.pointerId)) return;
    clearTimeout(holdTimer);
    if (active?.id === event.pointerId) {
      if (cancelled) { cancelPull(); clearHover(); }
      else if (active.mode === 'pull') {
        if (offsets.get(active.node)?.length() < .035) reattach(active.node);
        else { updateCount(); status.textContent = assemblyNames[meshInfo.get(active.hit.object).assembly.name] + ' detached'; }
      } else if (active.mode === 'pressed') tap(active, event);
    }
    active = null;
    if (pointers.size === 1) {
      const [[id, p]] = pointers;
      active = { id, x: p.x, y: p.y, mode: 'orbit' };
    }
    pinchGap = pointers.size > 1 ? gap() : 0;
    if (!pointers.size) canvas.classList.remove('grabbing');
  }
  canvas.addEventListener('pointerup', event => release(event, false));
  canvas.addEventListener('pointercancel', event => release(event, true));
  canvas.addEventListener('lostpointercapture', event => release(event, true));
  canvas.addEventListener('pointerleave', () => { if (!active) clearHover(); });
  canvas.addEventListener('wheel', event => {
    event.preventDefault(); actions.stopIntro(); clearHover();
    const pixels = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? canvas.clientHeight : 1);
    actions.zoom(Math.exp(THREE.MathUtils.clamp(pixels * .0015, -.9, .9)));
  }, { passive: false });

  chooser.addEventListener('change', updateCount);
  moveButton.addEventListener('click', () => {
    actions.stopIntro(); actions.leavePhoto(); clearHover();
    const node = model.getObjectByName(chooser.value);
    if (offsets.has(node)) reattach(node);
    else {
      offsets.set(node, assemblyByNode.get(node).travel.clone().normalize().multiplyScalar(.82));
      applyPosition(node); actions.moved(); updateCount();
    }
    status.textContent = assemblyNames[node.name] + (offsets.has(node) ? ' detached' : ' reattached');
  });
  document.querySelector('#reattach').addEventListener('click', () => {
    stopGesture(); reattachAll(); status.textContent = 'All parts reattached';
  });
  updateCount();
  return {
    setSeparation, stopGesture,
    toggleXray() { stopGesture(); actions.leavePhoto(); setXray(!xray); },
    reset() { stopGesture(); reattachAll(); setXray(false); status.textContent = ''; },
  };
}
