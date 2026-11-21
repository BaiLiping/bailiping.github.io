// app.js: the page. One canvas, one engine, everything done by hand: grab a part and pull it, drag the empty
// stage to turn it, hold the throttle and watch boost arrive late. The model is engine930.glb, built by the
// script in model/; this file poses it, runs the small engine model in sim.js and wires the keys.
import * as T from './vendor/three.bundle.min.js';
import { createStage } from './stage.js';
import { createSim, step, nudge } from './sim.js';
import { createFlow, createMotes } from './flow.js';
import { createSound } from './sound.js';

const canvas = document.getElementById('stage');
const statusEl = document.getElementById('status');
const HOME_VIEW = { yaw: 3.86, pitch: 0.36, dist: 3.0 };
const PULL_OFF = 0.07;          // metres: pull a part further than this and it stays off

const view = { ...HOME_VIEW, vyaw: 0, vpitch: 0, target: new T.Vector3(-0.17, 0.04, 0) };
const ui = { explode: 0, explodeTo: 0, gas: 0, gasTo: 0, cut: -1, hidden: false, explain: false, sound: null, soundOn: false, demandKey: 0, demandTouch: 0, hover: null };
const pointers = new Map();
let drag = null, stage = null, sim = null, flow = null, motes = null, info = null, tracks = [], chains = [];
let pickAt = null, last = performance.now();

/**
 * Format a number of revs with a thin space between thousands.
 * Takes the number. Returns a string.
 */
function revs(n) {
  return String(Math.round(n / 10) * 10).replace(/\B(?=(\d{3})+$)/g, ' ');
}

/**
 * Write the one-line readout.
 * Takes nothing. Returns nothing.
 */
function showStatus() {
  if (!sim) return;
  const crank = Math.floor(((sim.crank % 720) + 720) % 720);
  const bits = ['930/60', `${Math.round(info.spec.displacement_cc)} cm³`, `${revs(sim.rpm)} rpm`, `boost ${sim.boost.toFixed(2)} bar`, `crank ${String(crank).padStart(3, '0')}°`];
  if (sim.wastegate > 0.05) bits.push('wastegate open');
  if (ui.cut >= 0) bits.push(`cut: ${stage.cutNames[ui.cut]}`);
  const head = ui.hover ? `${ui.hover.label}  ·  ` : '';
  statusEl.textContent = head + bits.join(' · ');
}

/**
 * Point on a closed chain path.
 * Takes a chain (with pts, cum and total) and a distance along it. Returns [y, z] in model millimetres.
 */
function chainPoint(chain, s) {
  s = ((s % chain.total) + chain.total) % chain.total;
  let lo = 0, hi = chain.cum.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (chain.cum[mid] <= s) lo = mid; else hi = mid;
  }
  const a = chain.pts[lo], b = chain.pts[(lo + 1) % chain.pts.length];
  const f = (s - chain.cum[lo]) / Math.max(1e-9, chain.cum[lo + 1] - chain.cum[lo]);
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
}

/**
 * Sample the baked crank cycle and drive everything it cannot hold.
 * Takes nothing; reads the engine state. Returns nothing.
 */
function pose() {
  for (const p of stage.parts) {
    p.mesh.position.copy(p.restPos);
    p.mesh.quaternion.copy(p.restQuat);
  }
  const t = ((((sim.crank % 720) + 720) % 720) / 720) * stage.clip.duration;
  for (const tr of tracks) tr.target[tr.prop].fromArray(tr.interpolant.evaluate(Math.min(t, stage.clip.duration - 1e-5)));
  const crankRad = sim.crank * Math.PI / 180;
  const q = new T.Quaternion(), v = new T.Vector3();
  for (const p of stage.parts) {
    if (p.spin) {
      const angle = p.param && p.param.p === 'turbo' ? sim.turbo : crankRad * p.spin.ratio;
      p.mesh.quaternion.premultiply(q.setFromAxisAngle(p.spin.axis, angle));
    } else if (p.param) {
      const value = sim[p.param.p] || 0;
      if (p.param.moveV) p.mesh.position.addScaledVector(p.param.moveV, value);
      if (p.param.axisV) {
        q.setFromAxisAngle(p.param.axisV, T.MathUtils.degToRad(p.param.deg) * value);
        p.mesh.quaternion.premultiply(q);
        if (p.param.pivotV) p.mesh.position.add(p.param.pivotV).sub(v.copy(p.param.pivotV).applyQuaternion(q));
      }
    }
  }
  // chain links ride their path; the chain advances a whole number of pitches per crank turn
  for (const c of chains) {
    const phase = (((sim.crank % 360) + 360) % 360) / 360 * c.pitches_per_rev;
    let a = chainPoint(c, phase * c.pitch);
    for (let k = 0; k < c.links; k++) {
      const b = chainPoint(c, (k + 1 + phase) * c.pitch);
      stage.setLink(c.tag, k, c.x * 0.001, a[1] * 0.001, -a[0] * 0.001, Math.atan2(b[1] - a[1], b[0] - a[0]));
      a = b;
    }
  }
  stage.commitLinks();
}

/**
 * Move every part toward where it should be: home, pulled by hand, or out in the exploded view.
 * Takes the time step. Returns nothing.
 */
function settle(dt) {
  const want = new T.Vector3(), local = new T.Vector3(), q = new T.Quaternion();
  for (const p of stage.parts) {
    want.copy(p.manual).addScaledVector(p.explode, ui.explode);
    if (p.held) {
      p.offset.copy(want); p.vel.set(0, 0, 0);
    } else {                    // a damped spring, so parts glide home without bouncing forever
      p.vel.addScaledVector(want.sub(p.offset), 90 * dt).multiplyScalar(Math.max(0, 1 - 17 * dt));
      p.offset.addScaledVector(p.vel, dt);
    }
    if (p.offset.lengthSq() < 1e-10) continue;
    local.copy(p.offset);
    const parent = p.mesh.parent;
    if (parent && parent.userData.part) local.applyQuaternion(parent.getWorldQuaternion(q).invert());
    p.mesh.position.add(local);
  }
}

/**
 * Place the camera from the orbit state.
 * Takes the time step, for the turn's inertia. Returns nothing.
 */
function aim(dt) {
  if (!pointers.size) {
    view.yaw += view.vyaw * dt; view.pitch += view.vpitch * dt;
    const fade = Math.max(0, 1 - 4.5 * dt);
    view.vyaw *= fade; view.vpitch *= fade;
  }
  view.pitch = T.MathUtils.clamp(view.pitch, 0.015, 1.5);
  const c = Math.cos(view.pitch), dist = view.dist * (1 + 0.32 * ui.explode);        // step back as the parts spread out
  stage.camera.position.set(view.target.x + dist * c * Math.cos(view.yaw), view.target.y + dist * Math.sin(view.pitch),
    view.target.z + dist * c * Math.sin(view.yaw));
  stage.camera.lookAt(view.target);
  stage.camera.updateMatrixWorld();
}

/**
 * One frame.
 * Takes the timestamp from requestAnimationFrame. Returns nothing.
 */
function frame(now) {
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  step(sim, dt, Math.max(ui.demandKey, ui.demandTouch));
  ui.explode += (ui.explodeTo - ui.explode) * Math.min(1, dt * 5);
  ui.gas += (ui.gasTo - ui.gas) * Math.min(1, dt * 5);
  aim(dt);
  pose();
  settle(dt);
  if (pickAt && !drag) {
    const hit = stage.pick(pickAt[0], pickAt[1]);
    ui.hover = hit ? hit.part : null;
    stage.setHover(ui.hover);
    canvas.style.cursor = hit ? 'grab' : 'default';
    pickAt = null;
  }
  flow.update(sim, dt, ui.gas);
  if (motes) motes.update(now / 1000);
  if (ui.sound && ui.soundOn) ui.sound.update(sim);
  if (!ui.explain) stage.render(ui.gas);          // the explainer covers the stage while it is open
  showStatus();
  requestAnimationFrame(frame);
}

/**
 * Carry out one of the page's commands.
 * Takes the command name. Returns nothing.
 */
function act(name) {
  if (!stage) return;
  if (name === 'explode') ui.explodeTo = ui.explodeTo ? 0 : 1;
  if (name === 'gas') ui.gasTo = ui.gasTo ? 0 : 1;
  if (name === 'cut') {
    ui.cut = ui.cut + 1 >= stage.cutNames.length ? -1 : ui.cut + 1;
    stage.setCut(ui.cut < 0 ? null : stage.cutNames[ui.cut]);
  }
  if (name === 'reassemble') {
    ui.explodeTo = 0;
    for (const p of stage.parts) p.manual.set(0, 0, 0);
  }
  if (name === 'pause') sim.running = !sim.running;
  if (name === 'sound') {
    if (!ui.sound) ui.sound = createSound();
    ui.soundOn = !ui.soundOn;
    ui.sound.setOn(ui.soundOn);
  }
  if (name === 'hide') {
    ui.hidden = !ui.hidden;
    document.body.classList.toggle('quiet', ui.hidden);
  }
  if (name === 'home') Object.assign(view, HOME_VIEW);
  if (name === 'explain') {
    ui.explain = !ui.explain;
    const sheet = document.getElementById('explain'), frameEl = sheet.querySelector('iframe');
    if (ui.explain && !frameEl.src) frameEl.src = frameEl.dataset.src;      // loaded the first time it is asked for
    sheet.hidden = !ui.explain;
    for (const b of document.querySelectorAll('[data-act="explain"][aria-expanded]')) b.setAttribute('aria-expanded', String(ui.explain));
    (ui.explain ? sheet.querySelector('button') : canvas).focus();
  }
  for (const b of document.querySelectorAll('[data-act]')) {
    const on = { explode: ui.explodeTo === 1, gas: ui.gasTo === 1, cut: ui.cut >= 0, pause: !sim.running, sound: ui.soundOn }[b.dataset.act];
    if (on !== undefined) b.setAttribute('aria-pressed', String(on));
  }
}

/**
 * Where a pointer's ray meets the drag plane.
 * Takes the pointer event and the plane. Returns a Vector3, or null if the ray runs parallel to the plane.
 */
function onPlane(e, plane) {
  const rect = canvas.getBoundingClientRect();
  const ray = new T.Raycaster();
  ray.setFromCamera(new T.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1), stage.camera);
  return ray.ray.intersectPlane(plane, new T.Vector3());
}

/**
 * Distance between the two fingers of a pinch.
 * Takes nothing. Returns CSS pixels.
 */
function pinchSpan() {
  const [a, b] = [...pointers.values()];
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/**
 * Wire the pointer, wheel and keys.
 * Takes nothing. Returns nothing.
 */
function bind() {
  canvas.addEventListener('pointerdown', (e) => {
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.size === 2) {
      if (drag && drag.part) drag.part.held = false;
      drag = { pinch: pinchSpan(), dist: view.dist };
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const hit = stage.pick(e.clientX - rect.left, e.clientY - rect.top);
    if (!hit) { drag = { orbit: true }; return; }
    const normal = stage.camera.getWorldDirection(new T.Vector3()).negate();
    drag = { part: hit.part, plane: new T.Plane().setFromNormalAndCoplanarPoint(normal, hit.point), from: hit.point, manual: hit.part.manual.clone() };
    hit.part.held = true;
    canvas.style.cursor = 'grabbing';
  });
  canvas.addEventListener('pointermove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const prev = pointers.get(e.pointerId);
    if (!prev) { pickAt = [e.clientX - rect.left, e.clientY - rect.top]; return; }
    const dx = e.clientX - prev.x, dy = e.clientY - prev.y;
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (!drag) return;
    if (drag.pinch) {
      view.dist = T.MathUtils.clamp(drag.dist * drag.pinch / Math.max(20, pinchSpan()), 1.1, 9);
    } else if (drag.part) {
      const p = onPlane(e, drag.plane);
      if (p) drag.part.manual.copy(drag.manual).add(p.sub(drag.from));
    } else {
      view.yaw += dx * 0.006; view.pitch += dy * 0.006;
      view.vyaw = dx * 0.22; view.vpitch = dy * 0.22;
    }
  });
  const release = (e) => {
    pointers.delete(e.pointerId);
    if (drag && drag.part) {
      drag.part.held = false;
      if (drag.part.manual.length() < PULL_OFF) drag.part.manual.set(0, 0, 0);       // not pulled hard enough: it springs home
    }
    drag = null;
    canvas.style.cursor = 'default';
  };
  canvas.addEventListener('pointerup', release);
  canvas.addEventListener('pointercancel', release);
  canvas.addEventListener('pointerleave', () => { ui.hover = null; stage.setHover(null); });
  canvas.addEventListener('dblclick', (e) => {
    const rect = canvas.getBoundingClientRect();
    const hit = stage.pick(e.clientX - rect.left, e.clientY - rect.top);
    if (!hit) return;
    const p = hit.part;
    if (p.manual.lengthSq() > 1e-5) { p.manual.set(0, 0, 0); return; }
    if (p.explode.lengthSq() > 1e-6) p.manual.copy(p.explode).multiplyScalar(1 - ui.explode);
    else p.manual.copy(stage.camera.position).sub(hit.point).normalize().multiplyScalar(0.22);
  });
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    view.dist = T.MathUtils.clamp(view.dist * Math.exp(e.deltaY * 0.0011), 1.1, 9);
  }, { passive: false });
  window.addEventListener('keydown', (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key;
    if (ui.explain) { if (k === 'Escape') act('explain'); return; }
    if (k === 'ArrowUp') { ui.demandKey = 1; e.preventDefault(); return; }
    if (k === 'ArrowLeft' || k === 'ArrowRight') {
      if (!sim.running) nudge(sim, (k === 'ArrowRight' ? 1 : -1) * (e.shiftKey ? 1 : 6));
      e.preventDefault();
      return;
    }
    if (k === '+' || k === '=') { view.dist = Math.max(1.1, view.dist * 0.88); return; }
    if (k === '-' || k === '_') { view.dist = Math.min(9, view.dist / 0.88); return; }
    const name = { x: 'explode', c: 'cut', g: 'gas', r: 'reassemble', ' ': 'pause', m: 'sound', h: 'hide', i: 'explain', 0: 'home' }[k.toLowerCase()];
    if (!name || e.repeat || (k === ' ' && e.target.tagName === 'BUTTON')) return;
    act(name);
    e.preventDefault();
  });
  window.addEventListener('keyup', (e) => { if (e.key === 'ArrowUp') ui.demandKey = 0; });
  window.addEventListener('blur', () => { ui.demandKey = 0; ui.demandTouch = 0; });
  for (const b of document.querySelectorAll('[data-act]')) {
    if (b.dataset.act !== 'throttle') { b.addEventListener('click', () => act(b.dataset.act)); continue; }
    b.addEventListener('pointerdown', (e) => { ui.demandTouch = 1; b.setPointerCapture(e.pointerId); });
    for (const ev of ['pointerup', 'pointercancel', 'lostpointercapture']) b.addEventListener(ev, () => { ui.demandTouch = 0; });
    b.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') ui.demandTouch = 1; });
    b.addEventListener('keyup', () => { ui.demandTouch = 0; });
  }
  new ResizeObserver(() => stage.resize()).observe(canvas);
}

/**
 * Load the model and start.
 * Takes nothing. Returns nothing.
 */
async function main() {
  statusEl.textContent = 'loading the engine';
  info = await (await fetch('./engine930.json')).json();
  stage = await createStage(canvas, './engine930.glb', (f) => { statusEl.textContent = `loading the engine  ${Math.round(f * 100)}%`; });
  sim = createSim(info.spec);
  const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
  sim.running = !still;
  tracks = stage.clip.tracks.map((tr) => {
    const dot = tr.name.lastIndexOf('.');
    return { target: stage.byName.get(tr.name.slice(0, dot)).mesh, prop: tr.name.slice(dot + 1), interpolant: tr.createInterpolant() };
  });
  chains = Object.entries(info.chains).map(([tag, c]) => {
    const cum = [0];
    for (let i = 0; i < c.pts.length; i++) {
      const a = c.pts[i], b = c.pts[(i + 1) % c.pts.length];
      cum.push(cum[i] + Math.hypot(b[0] - a[0], b[1] - a[1]));
    }
    return { ...c, tag, cum, total: cum[cum.length - 1] };
  });
  const scale = stage.renderer.getPixelRatio() * canvas.clientHeight / 1000;
  flow = createFlow(info, stage.overlay, scale * 3.4);
  motes = still ? null : createMotes(stage.overlay, scale * 3.4);
  bind();
  act('none');
  aim(0);
  pose();
  stage.calibrate();
  document.body.classList.add('ready');
  window.engineLab = { stage, sim, ui, view, info, act };          // for the browser checks
  requestAnimationFrame((t) => { last = t; frame(t); });
}

main().catch((err) => {
  console.error(err);
  statusEl.textContent = `the engine could not be shown: ${err.message}`;
});
