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

// The exhaust-energy view: what stays solid, what turns to glass, which gas is drawn, and the name tags.
const EXHAUST = {
  groups: new Set(['exhaust', 'turbo']),
  extra: ['Charge_Pipe', 'Alternator', 'ExhaustValve_1', 'ExhaustValve_2', 'ExhaustValve_3', 'ExhaustValve_4', 'ExhaustValve_5', 'ExhaustValve_6'],
  glass: new Set(['Turbo_Turbine_Housing', 'Turbo_Compressor_Housing', 'Turbo_Centre_Housing']),
  kinds: new Set(['exhaust', 'exhaust_out', 'wastegate', 'boost']),
  view: { yaw: 4.5, pitch: 0.72, dist: 2.3, target: new T.Vector3(-0.3, -0.08, -0.04) },
  tags: [
    ['HeatExchanger_L', 'heat exchanger', 'gathers the exhaust of cylinders 1 to 3'],
    ['HeatExchanger_R', 'heat exchanger', 'gathers the exhaust of cylinders 4 to 6'],
    ['Crossover_Pipe', 'crossover pipe', 'brings the right bank’s exhaust across'],
    ['Turbine_Inlet_Pipe', 'pipe to the turbine', 'the exhaust of all six cylinders, joined'],
    ['Turbo_Turbine_Housing', 'turbine', 'the exhaust spins this wheel'],
    ['Turbo_Centre_Housing', 'shaft', 'carries the turbine’s power to the compressor'],
    ['Turbo_Compressor_Housing', 'compressor', 'driven by the shaft; packs more air into the engine'],
    ['Charge_Pipe', 'charge pipe', 'the boosted air, on its way to the intercooler'],
    ['Wastegate_Body', 'wastegate', 'at 0.8 bar it lets surplus exhaust pass round the turbine'],
    ['Muffler', 'muffler', ''],
    ['Tailpipe_R', 'tailpipe', 'exhaust that has been through the turbine'],
    ['Tailpipe_L', 'tailpipe', 'wastegate gas only'],
    ['Alternator', 'alternator', 'makes the electricity; the fan belt drives it, not the exhaust'],
  ],
};

const view = { ...HOME_VIEW, vyaw: 0, vpitch: 0, target: new T.Vector3(-0.17, 0.04, 0) };
const ui = { explode: 0, explodeTo: 0, gas: 0, gasTo: 0, cut: -1, hidden: false, explain: false, focus: false, focusFade: 0, spread: 0, sound: null, soundOn: false, demandKey: 0, demandTouch: 0, hover: null };
const pointers = new Map();
let drag = null, stage = null, sim = null, flow = null, motes = null, info = null, tracks = [], chains = [];
let pickAt = null, last = performance.now(), tags = [], savedView = null;
const intro = { on: false, t: 0, hold: 1.0, span: 2.4, settle: 1.0 };

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
  if (ui.focus) bits.push('exhaust energy');
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
    want.copy(p.manual).addScaledVector(p.explode, intro.on ? introSpread(p) : ui.explode);
    if (p.held || intro.on) {
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
 * How far out a part still is during the opening assembly: parts nearest the engine go home first.
 * Takes a part record. Returns 1 (fully out) to 0 (home).
 */
function introSpread(p) {
  const u = T.MathUtils.clamp((intro.t - intro.hold - p.introDelay * intro.span) / intro.settle, 0, 1);
  return 1 - u * u * (3 - 2 * u);
}

/**
 * End the opening assembly early; the parts finish going home on their springs.
 * Takes nothing. Returns nothing.
 */
function endIntro() {
  if (!intro.on) return;
  intro.on = false;
  ui.explode = ui.spread;
  ui.explodeTo = 0;
}

/**
 * Lay out the name tags of the exhaust-energy view: a column on each side of the page, each tag level with its
 * part where there is room, with a leader back to the part.
 * Takes nothing. Returns nothing.
 */
function placeTags() {
  const w = canvas.clientWidth, sides = { left: [], right: [] };
  for (const t of tags) {
    const p = stage.screenPoint(t.part);
    t.p = p;
    t.el.hidden = !p.front;
    if (p.front) sides[p.x < w * 0.5 ? 'left' : 'right'].push(t);
  }
  let lines = '';
  for (const [side, list] of Object.entries(sides)) {
    if (!list.length) continue;
    list.sort((a, b) => a.p.y - b.p.y);
    // one column per side, standing clear of the parts it names
    const edge = side === 'left' ? Math.min(...list.map((t) => t.p.x)) - 70 : Math.max(...list.map((t) => t.p.x)) + 70;
    let floor = -1e9;
    for (const t of list) {
      const y = Math.max(t.p.y - 8, floor);
      floor = y + t.el.offsetHeight + 6;
      const x = side === 'left' ? Math.max(8, edge - t.el.offsetWidth) : Math.min(w - t.el.offsetWidth - 8, edge);
      t.el.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
      const lx = side === 'left' ? x + t.el.offsetWidth + 3 : x - 3;
      lines += `<line x1="${t.p.x.toFixed(1)}" y1="${t.p.y.toFixed(1)}" x2="${lx.toFixed(1)}" y2="${(y + 9).toFixed(1)}"/><circle cx="${t.p.x.toFixed(1)}" cy="${t.p.y.toFixed(1)}" r="2.6"/>`;
    }
  }
  document.getElementById('leaders').innerHTML = lines;
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
  if (view.goal) {              // gliding to a chosen view
    const k = Math.min(1, dt * 3.5), g = view.goal;
    view.yaw += (g.yaw - view.yaw) * k; view.pitch += (g.pitch - view.pitch) * k; view.dist += (g.dist - view.dist) * k;
    view.target.lerp(g.target, k);
    if (Math.abs(g.yaw - view.yaw) + Math.abs(g.pitch - view.pitch) + Math.abs(g.dist - view.dist) < 0.004) view.goal = null;
  }
  view.pitch = T.MathUtils.clamp(view.pitch, 0.015, 1.5);
  const c = Math.cos(view.pitch), dist = view.dist * (1 + 0.32 * ui.spread);         // step back as the parts spread out
  stage.camera.position.set(view.target.x + dist * c * Math.cos(view.yaw), view.target.y + dist * Math.sin(view.pitch),
    view.target.z + dist * c * Math.sin(view.yaw));
  stage.camera.lookAt(view.target);
  stage.camera.updateMatrixWorld();
}

/**
 * Advance everything by one frame and draw it.
 * Takes the time now in milliseconds. Returns nothing.
 */
function advance(now) {
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  step(sim, dt, Math.max(ui.demandKey, ui.demandTouch));
  ui.explode += (ui.explodeTo - ui.explode) * Math.min(1, dt * 5);
  ui.gas += (ui.gasTo - ui.gas) * Math.min(1, dt * 5);
  ui.focusFade += ((ui.focus ? 1 : 0) - ui.focusFade) * Math.min(1, dt * 5);
  if (intro.on) {
    intro.t += dt;
    const u = T.MathUtils.clamp((intro.t - intro.hold) / (intro.span + intro.settle), 0, 1);
    ui.spread = 1 - u * u * (3 - 2 * u);
    if (u >= 1) { intro.on = false; ui.explode = 0; }
  } else ui.spread = ui.explode;
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
  flow.update(sim, dt, Math.max(ui.gas, ui.focusFade), ui.focus && !ui.gasTo ? EXHAUST.kinds : null);
  if (motes) motes.update(now / 1000);
  if (ui.sound && ui.soundOn) ui.sound.update(sim);
  if (!ui.explain) stage.render(ui.gas);          // the explainer covers the stage while it is open
  if (ui.focus) placeTags();
  showStatus();
}

/**
 * One frame of the running page.
 * Takes the timestamp from requestAnimationFrame. Returns nothing.
 */
function frame(now) {
  advance(now);
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
  if (name === 'home') { view.goal = null; Object.assign(view, HOME_VIEW); }
  if (name === 'exhaust') {
    ui.focus = !ui.focus;
    stage.setFocus(ui.focus ? new Set(stage.parts.filter((p) => EXHAUST.groups.has(p.group) || EXHAUST.extra.includes(p.name)).map((p) => p.name)) : null, EXHAUST.glass);
    if (ui.focus) {             // remember where the visitor was looking, and glide to a view of the turbo
      savedView = { yaw: view.yaw, pitch: view.pitch, dist: view.dist, target: view.target.clone() };
      view.goal = EXHAUST.view;
      ui.explodeTo = 0;
    } else if (savedView) view.goal = savedView;
    if (ui.focus) document.body.classList.add('coached');
    document.getElementById('tags').hidden = !ui.focus;
    document.getElementById('caption').hidden = !ui.focus;
  }
  if (name === 'explain') {
    ui.explain = !ui.explain;
    const sheet = document.getElementById('explain'), frameEl = sheet.querySelector('iframe');
    if (ui.explain && !frameEl.src) frameEl.src = frameEl.dataset.src;      // loaded the first time it is asked for
    sheet.hidden = !ui.explain;
    for (const b of document.querySelectorAll('[data-act="explain"][aria-expanded]')) b.setAttribute('aria-expanded', String(ui.explain));
    (ui.explain ? sheet.querySelector('button') : canvas).focus();
  }
  for (const b of document.querySelectorAll('[data-act]')) {
    const on = { explode: ui.explodeTo === 1, gas: ui.gasTo === 1, cut: ui.cut >= 0, pause: !sim.running, sound: ui.soundOn, exhaust: ui.focus }[b.dataset.act];
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
 * Keep the particles the same size on the page whatever the drawing resolution.
 * Takes nothing. Returns nothing.
 */
function sizeParticles() {
  const scale = stage.renderer.getPixelRatio() * canvas.clientHeight / 1000 * 3.4;
  flow.material.uniforms.uScale.value = scale;
  if (motes) motes.object.material.uniforms.uScale.value = scale;
}

/**
 * Wire the pointer, wheel and keys.
 * Takes nothing. Returns nothing.
 */
function bind() {
  canvas.addEventListener('pointerdown', (e) => {
    endIntro();
    view.goal = null;
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
      else document.body.classList.add('coached');                                  // they have found it: the prompt can go
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
    view.goal = null;
    view.dist = T.MathUtils.clamp(view.dist * Math.exp(e.deltaY * 0.0011), 1.1, 9);
  }, { passive: false });
  window.addEventListener('keydown', (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key;
    if (ui.explain) { if (k === 'Escape') act('explain'); return; }
    endIntro();
    if (k === 'ArrowUp') { ui.demandKey = 1; e.preventDefault(); return; }
    if (k === 'ArrowLeft' || k === 'ArrowRight') {
      if (!sim.running) nudge(sim, (k === 'ArrowRight' ? 1 : -1) * (e.shiftKey ? 1 : 6));
      e.preventDefault();
      return;
    }
    if (k === '+' || k === '=') { view.dist = Math.max(1.1, view.dist * 0.88); return; }
    if (k === '-' || k === '_') { view.dist = Math.min(9, view.dist / 0.88); return; }
    const name = { x: 'explode', c: 'cut', g: 'gas', r: 'reassemble', ' ': 'pause', m: 'sound', h: 'hide', i: 'explain', e: 'exhaust', 0: 'home' }[k.toLowerCase()];
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
    b.addEventListener('keyup', (e) => { if (e.key === 'Enter' || e.key === ' ') ui.demandTouch = 0; });
  }
  new ResizeObserver(() => { stage.resize(); sizeParticles(); }).observe(canvas);
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
  flow = createFlow(info, stage.overlay, 1);
  motes = still ? null : createMotes(stage.overlay, 1);
  // the opening: every part out where the exploded view puts it, then home, the nearest first
  const reach = Math.max(...stage.parts.map((p) => p.explode.length()));
  for (const p of stage.parts) p.introDelay = p.explode.length() / reach;
  intro.on = !still;
  ui.spread = intro.on ? 1 : 0;
  setTimeout(() => document.body.classList.add('coached'), 22000);
  const layer = document.getElementById('tags');
  tags = EXHAUST.tags.filter(([name]) => stage.byName.has(name)).map(([name, title, text]) => {
    const el = document.createElement('p');
    el.className = 'line tag';
    el.innerHTML = `<b>${title}</b>${text ? ` ${text}` : ''}`;
    layer.append(el);
    return { part: stage.byName.get(name), el };
  });
  bind();
  act('none');
  aim(0);
  pose();
  stage.calibrate();
  sizeParticles();
  document.body.classList.add('ready');
  // for browser checks: the page's state, and tick() to step frames by hand where the browser will not
  window.engineLab = { stage, sim, ui, view, info, act, intro, tick: (ms) => advance(last + ms) };
  requestAnimationFrame((t) => { last = t; frame(t); });
}

main().catch((err) => {
  console.error(err);
  statusEl.textContent = `the engine could not be shown: ${err.message}`;
});
