// flow.js: the gas made visible. Particles ride the paths exported with the model: ambient air to the compressor,
// hot boosted air up to the intercooler, cooled charge to the six ports, exhaust pulses to the turbine, and the
// wastegate's surplus round it. Each cylinder's runner flows only while its valve is open.
import * as T from './vendor/three.bundle.min.js';
import { valveOpen } from './sim.js?v=6f007a4e69';

const KINDS = {
  // kind: colour, particles per metre, base speed in m/s as drawn, point size
  ambient: { color: '#5f9ec4', density: 150, speed: 0.55, size: 9 },
  boost: { color: '#e09a2b', density: 240, speed: 0.75, size: 10 },
  charge: { color: '#2f9bd6', density: 170, speed: 0.7, size: 9 },
  exhaust: { color: '#dc4f2a', density: 170, speed: 0.95, size: 10 },
  exhaust_out: { color: '#9c6a52', density: 150, speed: 0.6, size: 11 },
  wastegate: { color: '#e6b43c', density: 200, speed: 0.9, size: 9 },
  recirc: { color: '#7d6fd0', density: 240, speed: 0.9, size: 8 },
  cooling: { color: '#9fb4bd', density: 55, speed: 0.5, size: 14 },
  // inside the turbocharger: each stream changes colour where it meets its wheel
  turbine_swirl: { color: '#e2491f', to: '#8f6a58', at: 0.62, density: 230, speed: 0.3, size: 2.5 },
  compressor_swirl: { color: '#4f9fd0', to: '#e09a2b', at: 0.3, density: 230, speed: 0.3, size: 2.5 },
};

const VERT = /* glsl */`
attribute vec4 color;
attribute float size;
uniform float uScale;
varying vec4 vColor;
void main() {
  vColor = color;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_PointSize = size * uScale / max(0.4, -mv.z);
  gl_Position = projectionMatrix * mv;
}`;

const FRAG = /* glsl */`
varying vec4 vColor;
void main() {
  vec2 d = gl_PointCoord - 0.5;
  float a = smoothstep(0.5, 0.12, length(d)) * vColor.a;
  if (a < 0.01) discard;
  gl_FragColor = vec4(vColor.rgb, a);
  #include <colorspace_fragment>
}`;

/**
 * Prepare a path: scene coordinates and running length.
 * Takes a gas path from engine930.json. Returns { pts, cum, total, ...path }.
 */
function preparePath(path) {
  const pts = path.pts.map((p) => new T.Vector3(p[0], p[2], -p[1]).multiplyScalar(0.001));
  const cum = [0];
  for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + pts[i].distanceTo(pts[i - 1]));
  return { ...path, pts, cum, total: cum[cum.length - 1], radius: path.r * 0.001 };
}

/**
 * Point at a distance along a prepared path.
 * Takes the path, the distance and a Vector3 to write into. Returns that vector, and leaves the segment it
 * fell in and the fraction along it in `seg` for the caller.
 */
const seg = { lo: 0, hi: 0, f: 0 };
function along(path, s, out) {
  let lo = 0, hi = path.cum.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (path.cum[mid] <= s) lo = mid; else hi = mid;
  }
  const f = (s - path.cum[lo]) / Math.max(1e-6, path.cum[hi] - path.cum[lo]);
  seg.lo = lo; seg.hi = hi; seg.f = Math.min(1, Math.max(0, f));
  return out.copy(path.pts[lo]).lerp(path.pts[hi], seg.f);
}

/**
 * Find, for every point of a gas path, the part the gas is inside there, so the stream can move with that part
 * when it is pulled off or the engine is exploded.
 * Takes the prepared paths and the part records (with the engine assembled). Returns nothing.
 */
function findCarriers(paths, parts) {
  const groups = new Set(['induction', 'intercooler', 'turbo', 'exhaust', 'heads', 'cooling']);
  const boxes = parts.filter((p) => groups.has(p.group) && p.mesh.geometry).map((p) => {
    const box = new T.Box3().setFromObject(p.mesh).expandByScalar(0.012), size = box.getSize(new T.Vector3());
    return { part: p, box, volume: size.x * size.y * size.z };
  }).sort((a, b) => a.volume - b.volume);
  for (const path of paths) {
    path.carriers = path.pts.map((pt) => {
      const inside = boxes.find((b) => b.box.containsPoint(pt));           // the smallest part that holds the point
      if (inside) return inside.part;
      let best = null, near = 0.08;
      for (const b of boxes) { const d = b.box.distanceToPoint(pt); if (d < near) { near = d; best = b.part; } }
      return best;
    });
  }
}

/**
 * Build the particle system.
 * Takes the info from engine930.json, the scene to add the points to, the pixel scale and the part records.
 * Returns { update, object, material, count }.
 */
export function createFlow(info, scene, pixelScale, parts) {
  const paths = info.gas.map(preparePath);
  findCarriers(paths, parts);
  const cyls = new Map(info.cylinders.map((c) => [c.n, c]));
  const particles = [];
  for (const path of paths) {
    const kind = KINDS[path.kind];
    const count = Math.max(6, Math.round(path.total * kind.density));
    for (let i = 0; i < count; i++) {
      const a = Math.random() * Math.PI * 2, r = Math.sqrt(Math.random()) * path.radius * 0.8;
      particles.push({ path, kind, s: Math.random() * path.total, jitter: new T.Vector3(Math.cos(a) * r, (Math.random() - 0.5) * r * 1.4, Math.sin(a) * r),
        pace: 0.75 + Math.random() * 0.5, phase: Math.random() });
    }
  }
  const n = particles.length;
  const position = new Float32Array(n * 3), color = new Float32Array(n * 4), size = new Float32Array(n);
  const tint = new T.Color();
  const tones = {};
  for (const [name, k] of Object.entries(KINDS)) tones[name] = [new T.Color(k.color), k.to ? new T.Color(k.to) : null];
  particles.forEach((p, i) => {
    tint.set(p.kind.color);
    color.set([tint.r, tint.g, tint.b, 0], i * 4);
    size[i] = p.kind.size * (0.7 + Math.random() * 0.6);
  });
  const geometry = new T.BufferGeometry();
  geometry.setAttribute('position', new T.BufferAttribute(position, 3).setUsage(T.DynamicDrawUsage));
  geometry.setAttribute('color', new T.BufferAttribute(color, 4).setUsage(T.DynamicDrawUsage));
  geometry.setAttribute('size', new T.BufferAttribute(size, 1));
  const material = new T.ShaderMaterial({ vertexShader: VERT, fragmentShader: FRAG, transparent: true, depthTest: false, depthWrite: false,
    uniforms: { uScale: { value: pixelScale } } });
  const object = new T.Points(geometry, material);
  object.frustumCulled = false;
  object.visible = false;
  scene.add(object);
  const tmp = new T.Vector3();

  /**
   * Move the particles and set how visible each stream is.
   * Takes the engine state from sim.js, the time step, the overall fade (0 to 1) and the set of gas kinds to
   * show. Returns nothing.
   */
  function update(sim, dt, fade, only) {
    object.visible = fade > 0.01;
    if (!object.visible) return;
    const load = 0.3 + 0.7 * (sim.rpm / sim.spec.redline) * (0.4 + 0.6 * sim.throttle + sim.boost);
    const moving = sim.rpm > 60 ? 1 : 0;
    for (let i = 0; i < n; i++) {
      const p = particles[i], path = p.path;
      let rate = load * moving, alpha = 0.85;
      if (path.cyl) {
        const cyl = cyls.get(path.cyl);
        const open = valveOpen(path.kind === 'exhaust' ? cyl.exhaust : cyl.intake, sim.crank);
        // in the runner the gas moves only with the valve; nearer the common pipe it keeps drifting
        const near = path.kind === 'exhaust' ? p.s / path.total : 1 - p.s / path.total;
        rate *= open * (path.kind === 'exhaust' ? 1.9 : 1.3) + 0.22 * near;
        alpha = 0.3 + 0.6 * Math.max(open, near * 0.7);
      } else if (path.kind === 'wastegate') {
        rate *= sim.wastegate * 1.4; alpha = Math.min(1, sim.wastegate * 3);
      } else if (path.kind === 'recirc') {
        rate = sim.bypass * 1.6; alpha = Math.min(1, sim.bypass * 2.5);
      } else if (path.kind === 'boost') {
        rate *= 0.6 + sim.boost; alpha = 0.45 + 0.6 * Math.min(1, sim.boost / sim.spec.boost_bar);
      } else if (path.kind === 'cooling') {
        rate = (0.25 + sim.rpm / sim.spec.redline) * moving; alpha = 0.3;
      } else if (p.kind.to) {
        rate = (0.35 + 1.5 * Math.min(1.2, sim.turboSpeed)) * moving; alpha = 0.85;
        const [from, to] = tones[path.kind], u = Math.min(1, Math.max(0, (p.s / path.total - p.kind.at) * 9 + 0.5));
        color[i * 4] = from.r + (to.r - from.r) * u; color[i * 4 + 1] = from.g + (to.g - from.g) * u; color[i * 4 + 2] = from.b + (to.b - from.b) * u;
      }
      p.s += p.kind.speed * p.pace * rate * dt;
      if (p.s > path.total) p.s -= path.total;
      along(path, p.s, tmp).add(p.jitter);
      const a = path.carriers[seg.lo], b = path.carriers[seg.hi];             // the gas goes where its pipe goes
      if (a) tmp.addScaledVector(a.offset, 1 - seg.f);
      if (b) tmp.addScaledVector(b.offset, seg.f);
      position[i * 3] = tmp.x; position[i * 3 + 1] = tmp.y; position[i * 3 + 2] = tmp.z;
      const ends = Math.min(1, p.s / 0.03, (path.total - p.s) / 0.03);       // fade in and out at the ends
      color[i * 4 + 3] = only.has(path.kind) ? alpha * fade * Math.max(0, ends) : 0;
    }
    geometry.attributes.position.needsUpdate = true;
    geometry.attributes.color.needsUpdate = true;
  }

  return { update, object, material, count: n };
}

/**
 * Dust motes drifting round the stage, the far ones larger and fainter.
 * Takes the scene to add them to and the pixel scale. Returns { update }.
 */
export function createMotes(scene, pixelScale) {
  const n = 70;
  const position = new Float32Array(n * 3), color = new Float32Array(n * 4), size = new Float32Array(n);
  const seeds = [];
  for (let i = 0; i < n; i++) {
    seeds.push({ x: (Math.random() - 0.5) * 4.4, y: -0.2 + Math.random() * 1.5, z: (Math.random() - 0.5) * 4.4, a: Math.random() * 6.28, w: 0.15 + Math.random() * 0.3 });
    color.set([0.32, 0.3, 0.27, 0.1 + Math.random() * 0.16], i * 4);
    size[i] = 3 + Math.random() * 9;
  }
  const geometry = new T.BufferGeometry();
  geometry.setAttribute('position', new T.BufferAttribute(position, 3).setUsage(T.DynamicDrawUsage));
  geometry.setAttribute('color', new T.BufferAttribute(color, 4));
  geometry.setAttribute('size', new T.BufferAttribute(size, 1));
  const object = new T.Points(geometry, new T.ShaderMaterial({ vertexShader: VERT, fragmentShader: FRAG, transparent: true, depthTest: false, depthWrite: false,
    uniforms: { uScale: { value: pixelScale } } }));
  object.frustumCulled = false;
  scene.add(object);

  /**
   * Drift the motes.
   * Takes the time in seconds. Returns nothing.
   */
  function update(time) {
    for (let i = 0; i < n; i++) {
      const s = seeds[i];
      position[i * 3] = s.x + Math.sin(time * s.w + s.a) * 0.12;
      position[i * 3 + 1] = s.y + Math.sin(time * s.w * 0.7 + s.a * 2) * 0.08;
      position[i * 3 + 2] = s.z + Math.cos(time * s.w * 0.9 + s.a) * 0.12;
    }
    geometry.attributes.position.needsUpdate = true;
  }
  return { update, object };
}
