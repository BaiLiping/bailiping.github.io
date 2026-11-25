// Both models use the same axes and metres. Draw their DOM-defined viewports with one
// renderer and a common orthographic scale so corresponding parts line up as the user orbits.
import * as T from './vendor/three.bundle.min.js';
import { createOpening, remainingOffset } from './assembly.js?v=e816bcea50';
import { createFlow, KINDS } from './flow.js?v=e816bcea50';

const MODELS = {
  turbo: ['./engine930.glb?v=e816bcea50', './engine930.json?v=e816bcea50'],
  sc: ['./engineSC.glb?v=e816bcea50', './engineSC.json?v=e816bcea50'],
};
const HOME = { yaw: 3.86, pitch: 0.40, zoom: 1 };
const MODES = ['assembled', 'gas', 'intake', 'exhaust', 'cutaway', 'exploded'];
const OUTER_SYSTEMS = new Set(['induction', 'intercooler', 'cooling', 'fuel', 'ignition', 'turbo', 'exhaust']);
const CUT_PLANE = new T.Plane(new T.Vector3(0, -1, 0), 0.001);
const GAS = {
  gas: new Set(['ambient', 'boost', 'charge', 'exhaust', 'exhaust_out', 'wastegate', 'recirc', 'cooling']),
  intake: new Set(['ambient', 'boost', 'charge']),
  exhaust: new Set(['exhaust', 'exhaust_out', 'wastegate']),
};
const clamp = T.MathUtils.clamp;

function belongsToSystem(part, mode) {
  if (mode === 'intake') {
    return (part.group === 'induction' && !/^(Air_Pump|Cam_Pulley)/.test(part.name)) ||
      part.group === 'intercooler' || ['Turbo_Compressor_Housing', 'Turbo_Rotor', 'Turbo_Centre_Housing'].includes(part.name);
  }
  if (mode === 'exhaust') {
    return part.group === 'exhaust' || ['Turbo_Turbine_Housing', 'Turbo_Rotor', 'Turbo_Centre_Housing'].includes(part.name);
  }
  return true;
}

// The two clips describe a four-stroke cycle. Fan and auxiliary shafts use the
// drive ratios in the model, and each timing chain follows its own closed pitch line.
function prepareMotion(model, info, clip) {
  const mixer = new T.AnimationMixer(model);
  mixer.clipAction(clip).play();
  const rotors = [];
  model.traverse(mesh => {
    const spin = mesh.userData.spin;
    if (!spin) return;
    rotors.push({ mesh, rest: mesh.quaternion.clone(),
      axis: new T.Vector3(spin[0], spin[2], -spin[1]).normalize(),
      ratio: spin[3], turbo: mesh.name === 'Turbo_Rotor' });
  });
  const chains = Object.entries(info.chains).map(([side, c]) => {
    const path = c.pts.map(([y, z]) => new T.Vector3(c.x, z, -y).multiplyScalar(0.001));
    const distance = [0];
    for (let i = 0; i < path.length; i++) distance.push(distance[i] + path[i].distanceTo(path[(i + 1) % path.length]));
    const links = Array.from({ length: c.links }, (_, i) => model.getObjectByName('Chain_' + side + '_' + String(i).padStart(3, '0')));
    return { path, distance, links, pitch: c.pitch * 0.001, ratio: c.pitches_per_rev };
  });
  const point = (chain, index, out) => {
    const { path, distance, pitch } = chain;
    const d = index * pitch % distance[path.length];
    let lo = 0, hi = path.length;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (distance[mid] <= d) lo = mid;
      else hi = mid;
    }
    return out.copy(path[lo]).lerp(path[hi % path.length], (d - distance[lo]) / (distance[hi] - distance[lo]));
  };
  const q = new T.Quaternion(), a = new T.Vector3(), b = new T.Vector3();
  return degrees => {
    mixer.setTime((degrees % 720) / 720 * clip.duration);
    for (const rotor of rotors) {
      // The turbo has its own shaft. The illustrative wheel speed is independent of the fan ratio.
      const phase = rotor.turbo ? degrees * 5 : degrees * rotor.ratio;
      rotor.mesh.quaternion.copy(rotor.rest).premultiply(q.setFromAxisAngle(rotor.axis, T.MathUtils.degToRad(phase)));
    }
    for (const c of chains) {
      const base = (degrees % 360) / 360 * c.ratio;
      point(c, base, a);
      for (let i = 0; i < c.links.length; i++) {
        point(c, i + 1 + base, b);
        const link = c.links[i];
        link.position.copy(a);
        link.rotation.set(Math.atan2(b.y - a.y, a.z - b.z), 0, 0);
        a.copy(b);
      }
    }
    model.updateMatrixWorld(true);
  };
}

// Instance repeated chain links, fasteners and cylinders. Keep a record for each original
// part so picking and exploded offsets still address individual pieces.
function prepareModel(model, info, clip, environment, pane, label) {
  const scene = new T.Scene();
  scene.environment = environment;
  scene.environmentIntensity = 0.66;
  scene.add(new T.HemisphereLight(0xfffaf3, 0xc0b8aa, 0.65));
  const light = new T.DirectionalLight(0xfff6e7, 2.8);
  light.position.set(-1.3, 3.6, 1.15);
  light.castShadow = true;
  const shadowSize = matchMedia('(pointer: coarse)').matches ? 768 : 1024;
  light.shadow.mapSize.set(shadowSize, shadowSize);
  Object.assign(light.shadow.camera, { left: -0.96, right: 0.96, top: 0.96, bottom: -0.96, near: 1.3, far: 7 });
  light.shadow.radius = 2.5;
  light.shadow.normalBias = 0.0015;
  light.shadow.bias = -0.00025;
  light.shadow.autoUpdate = false;
  scene.add(light);
  const rim = new T.DirectionalLight(0xf2f5ff, 0.85);
  rim.position.set(-1, 1, -2);
  scene.add(rim);

  const floor = new T.Mesh(new T.PlaneGeometry(4, 4), new T.ShadowMaterial({ opacity: 0.085, depthWrite: false }));
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -0.289;
  floor.receiveShadow = true;
  scene.add(floor);
  model.updateMatrixWorld(true);
  const groups = new Map(), byName = new Map(), parts = [];
  model.traverse((mesh) => {
    if (!mesh.isMesh) return;
    let owner = mesh;
    while (owner.parent && !owner.userData.g) owner = owner.parent;
    const { g, cut, ex = [0, 0, 0] } = owner.userData;
    const part = {
      name: owner.name, group: g, label: owner.userData.label || owner.name,
      mesh, geometry: mesh.geometry, matrix: mesh.matrixWorld.clone(), cut: cut === 1 || OUTER_SYSTEMS.has(g), detached: false,
      explode: new T.Vector3(ex[0], ex[2], -ex[1]).multiplyScalar(0.001),
      bounds: new T.Box3(), visible: true,
    };
    if (!mesh.geometry.boundingBox) mesh.geometry.computeBoundingBox();
    part.bounds.copy(mesh.geometry.boundingBox).applyMatrix4(part.matrix);
    const key = mesh.geometry.uuid + mesh.material.uuid + part.cut;
    if (!groups.has(key)) groups.set(key, { geometry: mesh.geometry, material: mesh.material, parts: [], cut: part.cut });
    groups.get(key).parts.push(part);
    parts.push(part);
    byName.set(part.name, part);
  });
  const reach = Math.max(...parts.map(part => part.explode.length()));
  for (const part of parts) part.relativeReach = reach ? part.explode.length() / reach : 0;
  const batches = [];
  for (const group of groups.values()) {
    group.material.side = T.FrontSide;
    const mesh = new T.InstancedMesh(group.geometry, group.material, group.parts.length);
    mesh.userData.slots = [];
    mesh.castShadow = mesh.receiveShadow = true;
    mesh.frustumCulled = false;
    scene.add(mesh);
    const cutMaterial = group.cut ? group.material.clone() : null;
    if (cutMaterial) {
      cutMaterial.clippingPlanes = [CUT_PLANE];
      cutMaterial.clipShadows = true;
      cutMaterial.side = T.DoubleSide;
      cutMaterial.onBeforeCompile = shader => {
        shader.fragmentShader = shader.fragmentShader.replace('#include <color_fragment>',
          '#include <color_fragment>\nif (!gl_FrontFacing) diffuseColor.rgb = vec3(0.55, 0.08, 0.045);');
      };
    }
    batches.push({ mesh, parts: group.parts, material: group.material, cutMaterial });
  }
  const highlight = new T.Mesh(undefined, new T.MeshBasicMaterial({
    color: 0x29a47d, transparent: true, opacity: 0.55, depthWrite: false,
    polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2,
  }));
  highlight.visible = false;
  highlight.matrixAutoUpdate = false;
  scene.add(highlight);
  const camera = new T.OrthographicCamera(-1, 1, 1, -1, 0.05, 12);
  const overlay = new T.Scene();
  const gas = info.gas.filter(p => GAS.gas.has(p.kind));
  const flow = createFlow({ ...info, gas }, overlay, 1);
  // A steady, loaded example. The 930 wastegate regulates boost while the
  // throttle is open. Its recirculation valve opens on lift-off and stays shut here.
  const flowState = { crank: 0, rpm: 4000, throttle: 0.9, boost: info.spec.boost_bar, spec: info.spec,
    wastegate: info.spec.boost_bar > 0 ? 0.55 : 0, bypass: 0 };
  // Keep depth testing and opaque silhouettes while muting the hardware behind
  // the streams. Transparency would pile up hundreds of overlapping internals.
  const fadedMaterial = new T.MeshStandardMaterial({
    color: 0xa7aaa5, emissive: 0x8c8e85, roughness: 0.95, metalness: 0,
  });
  const pose = prepareMotion(model, info, clip);
  return { scene, overlay, camera, pane, label, parts, batches, byName, highlight, flow, flowState, fadedMaterial, pose, light, floor };
}

function applyMode(view, mode, refreshShadow = true, opening = null) {
  view.mode = mode;
  if (refreshShadow) view.light.shadow.needsUpdate = true;
  view.floor.visible = mode === 'assembled' || mode === 'cutaway';
  for (const { mesh, parts, material, cutMaterial } of view.batches) {
    mesh.material = mode === 'cutaway' && cutMaterial ? cutMaterial : material;
    if (mode === 'gas') mesh.material = view.fadedMaterial;
    const slots = mesh.userData.slots;
    slots.length = 0;
    for (const part of parts) {
      part.visible = belongsToSystem(part, mode);
      if (!part.visible) continue;
      part.matrix.copy(part.mesh.matrixWorld);
      if (mode === 'exploded' || (part.detached && (mode === 'assembled' || mode === 'cutaway'))) {
        const e = part.matrix.elements;
        const fraction = opening ? remainingOffset(opening, part.relativeReach) : 1;
        e[12] += part.explode.x * fraction;
        e[13] += part.explode.y * fraction;
        e[14] += part.explode.z * fraction;
        if (part.detached && part.explode.lengthSq() === 0) e[13] += 0.28;
      }
      part.bounds.copy(part.geometry.boundingBox).applyMatrix4(part.matrix);
      if (mode === 'cutaway' && part.cut) {
        if (part.bounds.min.y > CUT_PLANE.constant) { part.visible = false; continue; }
        part.bounds.max.y = Math.min(part.bounds.max.y, CUT_PLANE.constant);
      }
      mesh.setMatrixAt(slots.length, part.matrix);
      slots.push(part);
    }
    mesh.count = slots.length;
    mesh.visible = Boolean(slots.length);
    mesh.instanceMatrix.needsUpdate = true;
    mesh.boundingSphere = null; // picking must use the current exploded position
  }
  view.flow.object.visible = Boolean(GAS[mode]);
}

function selectPart(views, part) {
  for (const view of views) {
    const match = part && view.byName.get(part.name);
    view.highlight.visible = Boolean(match && match.visible);
    if (view.highlight.visible) {
      view.highlight.material.clippingPlanes = view.mode === 'cutaway' && match.cut ? [CUT_PLANE] : null;
      view.highlight.geometry = match.geometry;
      view.highlight.matrix.copy(match.matrix);
      view.highlight.matrixWorldNeedsUpdate = true;
    }
    if (view.label && view.label.textContent !== (match?.label || '')) view.label.textContent = match?.label || '';
  }
}

// Project each visible part's world-space bounds onto the camera axes. Both cameras
// frame the union, including when removing the rest of the engine exposes the plumbing.
function framePair(views, orbit, rects) {
  const { yaw } = orbit;
  const pitch = clamp(orbit.pitch + (views[0].mode === 'cutaway' ? 0.49 : 0), -1.48, 1.48);
  const direction = new T.Vector3(Math.cos(pitch) * Math.cos(yaw), Math.sin(pitch), Math.cos(pitch) * Math.sin(yaw));
  const right = new T.Vector3(direction.z, 0, -direction.x).normalize();
  const up = new T.Vector3().crossVectors(direction, right);
  const center = new T.Vector3(), size = new T.Vector3();
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const view of views) for (const p of view.parts) {
    if (!p.visible) continue;
    p.bounds.getCenter(center);
    p.bounds.getSize(size).multiplyScalar(0.5);
    const x = center.dot(right), y = center.dot(up);
    const rx = Math.abs(right.x) * size.x + Math.abs(right.z) * size.z;
    const ry = Math.abs(up.x) * size.x + Math.abs(up.y) * size.y + Math.abs(up.z) * size.z;
    x0 = Math.min(x0, x - rx); x1 = Math.max(x1, x + rx);
    y0 = Math.min(y0, y - ry); y1 = Math.max(y1, y + ry);
  }
  const target = right.clone().multiplyScalar((x1 + x0) / 2).addScaledVector(up, (y1 + y0) / 2);
  const scale = Math.min(...rects.map(r => Math.min(r.width / (x1 - x0), r.height / (y1 - y0)))) * 0.90 * orbit.zoom;
  views.forEach(({ camera }, i) => {
    camera.left = -rects[i].width / (2 * scale); camera.right = -camera.left;
    camera.top = rects[i].height / (2 * scale); camera.bottom = -camera.top;
    camera.position.copy(target).addScaledVector(direction, 4);
    camera.lookAt(target);
    camera.updateMatrixWorld();
    camera.updateProjectionMatrix();
  });
}

export async function mountComparison(root) {
  const canvas = root.querySelector('[data-compare-canvas]');
  const status = root.querySelector('[data-compare-status]');
  const orbit = { ...HOME }, views = [], pointers = new Map();
  const still = matchMedia('(prefers-reduced-motion: reduce)');
  const events = new AbortController(), listener = { signal: events.signal };
  const observer = new ResizeObserver(() => invalidate());
  let renderer, environment, frame = 0, mode = 'assembled', previous = 0, reframe = true;
  let running = false, crank = 0, selected = null;
  const opening = createOpening();
  const runButton = root.querySelector('[data-action=run]');
  const detachButton = root.querySelector('[data-action=detach]');
  let width = 0, height = 0, active = true, onScreen = true;
  const visibility = new IntersectionObserver(([entry]) => {
    onScreen = entry.isIntersecting;
    if (onScreen) invalidate();
    else previous = 0;
  });

  function invalidate() {
    reframe = true;
    if (active && !frame && onScreen && !document.hidden) frame = requestAnimationFrame(draw);
  }

  function draw(now) {
    frame = 0;
    if (!active || !onScreen || document.hidden || views.length !== 2) return;
    const area = canvas.getBoundingClientRect();
    const rects = views.map(v => v.pane.getBoundingClientRect());
    if (area.width < 1 || area.height < 1 || rects.some(r => r.width < 1 || r.height < 1)) return;
    if (area.width !== width || area.height !== height) {
      width = area.width; height = area.height;
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2, Math.sqrt(2.5e6 / (width * height))));
      renderer.setSize(width, height, false);
      reframe = true;
    }
    const elapsed = previous ? Math.max(0, (now - previous) / 1000) : 0;
    const dt = Math.min(elapsed, 0.1);
    if (opening.on) {
      const lastTime = opening.t;
      opening.t += elapsed;
      if (opening.t >= opening.hold + opening.span + opening.settle) {
        finishOpening();
      } else if (opening.t > opening.hold) {
        const refreshShadow = Math.floor(opening.t * 10) !== Math.floor(lastTime * 10);
        for (const view of views) applyMode(view, mode, refreshShadow, opening);
        reframe = true;
      }
    }
    if (reframe) framePair(views, orbit, rects);
    reframe = false;
    if (running) {
      crank += dt * 180; // one crank revolution in two seconds, matched across both models
      for (const view of views) {
        view.pose(crank);
        applyMode(view, mode, Math.floor(crank / 35) !== Math.floor((crank - dt * 180) / 35));
      }
      if (selected) selectPart(views, selected);
    }
    previous = now;
    renderer.setScissorTest(false);
    renderer.clear();
    renderer.setScissorTest(true);
    views.forEach((view, i) => {
      const r = rects[i], x = r.left - area.left, y = area.bottom - r.bottom;
      renderer.setViewport(x, y, r.width, r.height);
      renderer.setScissor(x, y, r.width, r.height);
      renderer.render(view.scene, view.camera);
      if (GAS[mode]) {
        view.flowState.crank = running ? crank % 720 : (view.flowState.crank + (still.matches ? 0 : dt * 180)) % 720;
        view.flow.material.uniforms.uScale.value = renderer.getPixelRatio() * clamp(r.width / 450, 0.7, 1.4) * 1.85;
        view.flow.update(view.flowState, still.matches && !running ? 0 : dt, 1, GAS[mode]);
        renderer.render(view.overlay, view.camera);
      }
    });
    if (root.dataset.ready !== 'true') {
      if (status) status.textContent = '';
      root.dataset.ready = 'true';
    }
    if (!frame && (opening.on || running || (GAS[mode] && !still.matches))) frame = requestAnimationFrame(draw);
  }

  function canDetach() {
    return Boolean(selected && (mode === 'assembled' || mode === 'cutaway'));
  }
  function updateDetach() {
    if (!detachButton) return;
    detachButton.disabled = !canDetach();
    detachButton.textContent = selected?.detached ? 'Refit selected part' : 'Move selected part';
  }
  function detach() {
    if (!canDetach()) return;
    selected.detached = !selected.detached;
    for (const view of views) applyMode(view, mode);
    selectPart(views, selected);
    updateDetach();
    invalidate();
  }
  function setRunning(value) {
    running = value;
    previous = 0;
    if (runButton) {
      runButton.setAttribute('aria-pressed', String(running));
      runButton.textContent = running ? 'Pause engines' : 'Run engines';
    }
    invalidate();
  }
  function setMode(next) {
    if (!MODES.includes(next)) return;
    if (opening.on) {
      opening.on = false;
      delete root.dataset.opening;
    }
    mode = next;
    previous = 0;
    selected = null;
    updateDetach();
    selectPart(views, null);
    for (const view of views) applyMode(view, mode);
    for (const button of root.querySelectorAll('[data-mode]')) {
      button.setAttribute('aria-pressed', String(mode === button.dataset.mode));
    }
    root.dataset.currentMode = mode;
    root.dispatchEvent(new CustomEvent('comparemodechange', { detail: { mode } }));
    invalidate();
  }

  function finishOpening() {
    if (opening.on) setMode('assembled');
  }

  function reset() {
    Object.assign(orbit, HOME);
    setRunning(false);
    crank = 0;
    for (const view of views) {
      for (const part of view.parts) part.detached = false;
      view.pose(crank);
    }
    setMode('assembled');
  }
  function zoom(factor) {
    orbit.zoom = clamp(orbit.zoom * factor, 0.45, 3.5);
    invalidate();
  }
  function turn(dx, dy) {
    orbit.yaw -= dx;
    orbit.pitch = clamp(orbit.pitch + dy, -1.48, 1.48);
    invalidate();
  }
  function pick(x, y) {
    const ray = new T.Raycaster();
    for (const view of views) {
      const r = view.pane.getBoundingClientRect();
      if (x < r.left || x > r.right || y < r.top || y > r.bottom) continue;
      ray.setFromCamera(new T.Vector2(2 * (x - r.left) / r.width - 1, 1 - 2 * (y - r.top) / r.height), view.camera);
      const meshes = view.batches.map(b => b.mesh).filter(mesh => mesh.visible);
      const hit = ray.intersectObjects(meshes, false).find(h => {
        const part = h.object.userData.slots[h.instanceId];
        return mode !== 'cutaway' || !part.cut || CUT_PLANE.distanceToPoint(h.point) >= 0;
      });
      selected = hit?.object.userData.slots[hit.instanceId] || null;
      selectPart(views, selected);
      updateDetach();
      invalidate();
      break;
    }
  }

  function bindInput() {
    const capture = { ...listener, capture: true };
    root.addEventListener('pointerdown', finishOpening, capture);
    root.addEventListener('click', finishOpening, capture);
    window.addEventListener('keydown', finishOpening, capture);
    for (const label of root.querySelectorAll('.gas-key [data-kind]')) {
      label.style.setProperty('--tone', KINDS[label.dataset.kind].color);
    }
    for (const button of root.querySelectorAll('[data-mode]')) {
      button.addEventListener('click', () => setMode(button.dataset.mode), listener);
    }
    const actions = { reset, detach, run: () => setRunning(!running), 'zoom-in': () => zoom(1.2), 'zoom-out': () => zoom(1 / 1.2) };
    for (const button of root.querySelectorAll('[data-action]')) {
      const action = actions[button.dataset.action];
      if (action) button.addEventListener('click', action, listener);
    }
    canvas.addEventListener('dblclick', e => {
      pick(e.clientX, e.clientY);
      detach();
    }, listener);
    canvas.addEventListener('pointerdown', e => {
      if (e.button !== 0) return;
      canvas.focus({ preventScroll: true });
      canvas.setPointerCapture(e.pointerId);
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY, sx: e.clientX, sy: e.clientY, dragged: false });
      if (pointers.size > 1) for (const p of pointers.values()) p.dragged = true;
    }, listener);
    canvas.addEventListener('pointermove', e => {
      const p = pointers.get(e.pointerId);
      if (!p) return;
      const dx = e.clientX - p.x, dy = e.clientY - p.y;
      if (pointers.size === 2) {
        const other = [...pointers.values()].find(v => v !== p);
        const before = Math.hypot(p.x - other.x, p.y - other.y);
        const after = Math.hypot(e.clientX - other.x, e.clientY - other.y);
        if (before > 2 && after > 2) zoom(after / before);
      } else if (pointers.size === 1 && (p.dragged || Math.hypot(e.clientX - p.sx, e.clientY - p.sy) > 4)) {
        p.dragged = true;
        turn(dx * 0.009, dy * 0.009);
      }
      p.x = e.clientX; p.y = e.clientY;
    }, listener);
    canvas.addEventListener('pointerup', e => {
      const p = pointers.get(e.pointerId);
      if (p && !p.dragged) pick(e.clientX, e.clientY);
      pointers.delete(e.pointerId);
    }, listener);
    for (const name of ['pointercancel', 'lostpointercapture']) {
      canvas.addEventListener(name, e => pointers.delete(e.pointerId), listener);
    }
    canvas.addEventListener('wheel', e => {
      finishOpening();
      e.preventDefault();
      const units = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? height : 1;
      zoom(Math.exp(-clamp(e.deltaY * units, -300, 300) * 0.0015));
    }, { ...listener, passive: false });
    canvas.addEventListener('keydown', e => {
      const actions = {
        ArrowLeft: () => turn(-0.13, 0), ArrowRight: () => turn(0.13, 0),
        ArrowUp: () => turn(0, 0.13), ArrowDown: () => turn(0, -0.13),
        '+': () => zoom(1.15), '=': () => zoom(1.15), '-': () => zoom(1 / 1.15), '0': reset,
      };
      if (e.metaKey || e.ctrlKey || e.altKey || !actions[e.key]) return;
      e.preventDefault();
      actions[e.key]();
    }, listener);
    still.addEventListener('change', () => {
      if (still.matches) { finishOpening(); setRunning(false); }
      else invalidate();
    }, listener);
    document.addEventListener('visibilitychange', () => {
      previous = 0;
      invalidate();
    }, listener);
    observer.observe(canvas);
    for (const view of views) observer.observe(view.pane);
    visibility.observe(canvas);
  }

  function dispose() {
    active = false;
    events.abort();
    observer.disconnect();
    visibility.disconnect();
    cancelAnimationFrame(frame);
    const resources = new Set();
    for (const view of views) {
      view.light.shadow.dispose();
      resources.add(view.fadedMaterial);
      for (const b of view.batches) {
        resources.add(b.material);
        if (b.cutMaterial) resources.add(b.cutMaterial);
      }
      for (const scene of [view.scene, view.overlay]) scene.traverse(o => {
        if (o.geometry) resources.add(o.geometry);
        if (o.material) resources.add(o.material);
        if (o.isInstancedMesh) resources.add(o);
      });
    }
    for (const res of resources) res.dispose();
    environment?.dispose();
    renderer?.dispose();
  }

  try {
    if (status) status.textContent = 'Loading both engines…';
    renderer = new T.WebGLRenderer({ canvas, alpha: true, antialias: true, powerPreference: 'high-performance' });
    renderer.setClearColor(0, 0);
    renderer.autoClear = false;
    renderer.localClippingEnabled = true;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = T.PCFShadowMap;
    renderer.toneMapping = T.ACESFilmicToneMapping;
    const generator = new T.PMREMGenerator(renderer), room = new T.RoomEnvironment();
    try { environment = generator.fromScene(room, 0.04); }
    finally { room.dispose(); generator.dispose(); }
    const loader = new T.GLTFLoader();
    const models = await Promise.all(Object.entries(MODELS).map(async ([name, [modelUrl, dataUrl]]) => {
      const [gltf, info] = await Promise.all([
        loader.loadAsync(modelUrl),
        fetch(dataUrl).then(r => {
          if (!r.ok) throw new Error(`Could not load ${name} model data (${r.status})`);
          return r.json();
        }),
      ]);
      return { name, root: gltf.scene, clip: gltf.animations[0], info };
    }));
    for (const model of models) {
      const pane = root.querySelector(`[data-engine="${model.name}"]`);
      views.push(prepareModel(model.root, model.info, model.clip, environment.texture, pane, root.querySelector(`[data-part="${model.name}"]`)));
    }
    bindInput();
    setMode(still.matches ? 'assembled' : 'exploded');
    if (!still.matches) {
      opening.on = true;
      root.dataset.opening = 'true';
    }
    return { dispose, reset };
  } catch (err) {
    dispose();
    if (status) status.textContent = 'The 3D models could not load. Enable WebGL 2 and reload the page.';
    throw err;
  }
}
