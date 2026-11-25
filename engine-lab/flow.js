// Particles trace each engine’s exported intake, exhaust and fan-cooling paths.
import * as T from './vendor/three.bundle.min.js';

export const KINDS = {
  // kind: colour, particles per metre, base speed in m/s as drawn, point size
  ambient: { color: '#5f9ec4', density: 150, speed: 0.55, size: 9 },
  boost: { color: '#e09a2b', density: 240, speed: 0.75, size: 10 },
  charge: { color: '#2f9bd6', density: 170, speed: 0.7, size: 9 },
  exhaust: { color: '#dc4f2a', density: 170, speed: 0.95, size: 10 },
  exhaust_out: { color: '#9c6a52', density: 150, speed: 0.6, size: 11 },
  wastegate: { color: '#e6b43c', density: 200, speed: 0.9, size: 9 },
  recirc: { color: '#7d6fd0', density: 240, speed: 0.9, size: 8 },
  cooling: { color: '#9fb4bd', density: 55, speed: 0.5, size: 14 },
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

// Lift follows the intake or exhaust timing window over a 720-degree cycle.
function valveOpen(window, crank) {
  const centre = (window[0] + window[1]) / 2, half = (window[1] - window[0]) / 2;
  const d = ((((crank - centre) % 720) + 1080) % 720) - 360;
  return Math.abs(d) >= half ? 0 : 0.5 * (1 + Math.cos(Math.PI * d / half));
}

// Convert model millimetres to scene metres and tabulate the distance down each pipe.
function preparePath(path) {
  const pts = path.pts.map((p) => new T.Vector3(p[0], p[2], -p[1]).multiplyScalar(0.001));
  const cum = [0];
  for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + pts[i].distanceTo(pts[i - 1]));
  return { ...path, pts, cum, total: cum[cum.length - 1], radius: path.r * 0.001 };
}

function along(path, s, out) {
  let lo = 0, hi = path.cum.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (path.cum[mid] <= s) lo = mid; else hi = mid;
  }
  const f = (s - path.cum[lo]) / Math.max(1e-6, path.cum[hi] - path.cum[lo]);
  return out.copy(path.pts[lo]).lerp(path.pts[hi], Math.min(1, Math.max(0, f)));
}

export function createFlow(info, scene, pixelScale) {
  const paths = info.gas.map(preparePath);
  const cyls = new Map(info.cylinders.map((c) => [c.n, c]));
  const particles = [];
  for (const path of paths) {
    const kind = KINDS[path.kind];
    const count = Math.max(6, Math.round(path.total * kind.density));
    for (let i = 0; i < count; i++) {
      const a = Math.random() * Math.PI * 2, r = Math.sqrt(Math.random()) * path.radius * 0.8;
      particles.push({ path, kind, s: Math.random() * path.total, jitter: new T.Vector3(Math.cos(a) * r, (Math.random() - 0.5) * r * 1.4, Math.sin(a) * r),
        pace: 0.75 + Math.random() * 0.5 });
    }
  }
  const n = particles.length;
  const position = new Float32Array(n * 3), color = new Float32Array(n * 4), size = new Float32Array(n);
  const tint = new T.Color();
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
   * The engine state supplies crank angle, load and valve openings for the selected streams.
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
      }
      p.s += p.kind.speed * p.pace * rate * dt;
      if (p.s > path.total) p.s -= path.total;
      along(path, p.s, tmp).add(p.jitter);
      position[i * 3] = tmp.x; position[i * 3 + 1] = tmp.y; position[i * 3 + 2] = tmp.z;
      const ends = Math.min(1, p.s / 0.03, (path.total - p.s) / 0.03);       // fade in and out at the ends
      color[i * 4 + 3] = only.has(path.kind) ? alpha * fade * Math.max(0, ends) : 0;
    }
    geometry.attributes.position.needsUpdate = true;
    geometry.attributes.color.needsUpdate = true;
  }

  return { update, object, material };
}
