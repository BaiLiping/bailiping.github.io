// Sectioned teaching models. X follows the gearbox shafts, Y is up, and the axle runs along Z.
// Each rigid assembly is batched by material before it reaches the renderer.
import * as T from '../engine-lab/vendor/three.bundle.min.js';
import { createOpening, remainingOffset } from '../engine-lab/assembly.js';
import { clearanceAt, engagementAt } from './linkage.js?v=c918544ca383';
import { selectorPose } from './selector.js?v=756fd58ab53a';

const TAU = 2 * Math.PI;
// Keep the near shaft below the forks on the manual and the twin clutch faces open on the PDK.
const VIEW_ANGLES = {
  selector: { manual: [-.20, .99], pdk: [-.41, .81] },
  overhead: { manual: [0, 1.49], pdk: [0, 1.49] },
  clutch: { manual: [-.66, .41], pdk: [-.70, .42] },
};
const START = { manualGear: 1, manualClutch: 1, oddGear: 1, evenGear: 2, oddClutch: 1, evenClutch: 0, moving: true, running: false };
const clamp = T.MathUtils.clamp;
const v = (x, y, z) => new T.Vector3(x, y, z);

function geometry(points, indices, normals) {
  const result = new T.BufferGeometry();
  result.setAttribute('position', new T.Float32BufferAttribute(points, 3));
  if (indices) result.setIndex(indices);
  if (normals) result.setAttribute('normal', new T.Float32BufferAttribute(normals, 3));
  else result.computeVertexNormals();
  return result;
}

function roundedBlock(w, h, d, bevel = 0.035) {
  const half = [w / 2, h / 2, d / 2];
  const r = Math.min(bevel, ...half.map(s => s * 0.8));
  const points = [], normals = [], indices = [];
  for (let axis = 0; axis < 3; axis++) {
    const u = (axis + 1) % 3, t = (axis + 2) % 3;
    const samples = s => [-s, -s + r * 0.36, -s + r, s - r, s - r * 0.36, s];
    const us = samples(half[u]), ts = samples(half[t]);
    for (const sign of [-1, 1]) {
      const start = points.length / 3;
      for (const a of us) for (const b of ts) {
        const p = [0, 0, 0];
        p[axis] = sign * half[axis]; p[u] = a; p[t] = b;
        const core = p.map((coord, i) => clamp(coord, r - half[i], half[i] - r));
        const delta = p.map((coord, i) => coord - core[i]);
        const length = Math.hypot(...delta);
        const n = delta.map(coord => coord / length);
        points.push(...core.map((coord, i) => coord + r * n[i]));
        normals.push(...n);
      }
      for (let i = 0; i < us.length - 1; i++) for (let j = 0; j < ts.length - 1; j++) {
        const a = start + i * ts.length + j, b = a + ts.length;
        if (sign > 0) indices.push(a, b, a + 1, b, b + 1, a + 1);
        else indices.push(a, a + 1, b, b, a + 1, b + 1);
      }
    }
  }
  return geometry(points, indices, normals);
}

// Closed radial profile as [axial coordinate, radius], revolved around X. A missing
// sector lays open the clutch basket and the casting without a clipping shader.
function turn(profile, segments = 48, from = 0, sweep = TAU) {
  const points = [], indices = [];
  for (let j = 0; j < profile.length; j++) {
    const a = profile[j], b = profile[(j + 1) % profile.length], start = points.length / 3;
    for (let i = 0; i <= segments; i++) {
      const angle = from + i / segments * sweep, cy = Math.cos(angle), cz = Math.sin(angle);
      points.push(a[0], a[1] * cy, a[1] * cz, b[0], b[1] * cy, b[1] * cz);
    }
    for (let i = 0; i < segments; i++) {
      const k = start + 2 * i;
      indices.push(k, k + 2, k + 1, k + 2, k + 3, k + 1);
    }
  }
  if (sweep < TAU - 0.0001) {
    const center = profile.reduce((p, a) => [p[0] + a[0] / profile.length, p[1] + a[1] / profile.length], [0, 0]);
    for (const [end, angle] of [[false, from], [true, from + sweep]]) {
      const start = points.length / 3, cy = Math.cos(angle), cz = Math.sin(angle);
      for (const [x, radius] of [center, ...profile]) points.push(x, radius * cy, radius * cz);
      for (let j = 0; j < profile.length; j++) {
        const a = start + j + 1, b = start + (j + 1) % profile.length + 1;
        if (end) indices.push(start, b, a);
        else indices.push(start, a, b);
      }
    }
  }
  return geometry(points, indices);
}

function ring(length, outer, inner = 0, segments = 48, from = 0, sweep = TAU, bevel = 0.012) {
  const end = length / 2, b = Math.min(bevel, end * 0.65, (outer - inner) * 0.35);
  return turn([[-end, inner], [-end, outer - b], [-end + b, outer],
    [end - b, outer], [end, outer - b], [end, inner]], segments, from, sweep);
}

function toothRing(pitch, width, count, helix, inside, depth) {
  const vertices = [], faces = [];
  const tooth = depth || (2 * pitch / count);
  const outline = [[0, -1.18], [.14, -1.18], [.20, -.98], [.28, -.30],
    [.36, .57], [.40, .87], [.60, .87], [.64, .57], [.72, -.30], [.80, -.98], [.86, -1.18]];
  const total = count * outline.length;
  const levels = [[-0.5, -0.11], [-0.40, 0], [0.40, 0], [.5, -.11]];
  const sample = (i, level, inner = false) => {
    const [p, offset] = outline[i % outline.length];
    const f = (Math.floor(i / outline.length) + p) / count;
    const a = f * TAU + helix * width * level[0] / pitch;
    const r = inner ? inside : pitch + tooth * (offset + level[1]);
    return [width * level[0], r * Math.cos(a), r * Math.sin(a)];
  };
  const quad = (a, b, c, d) => {
    const i = vertices.length / 3;
    vertices.push(...a, ...b, ...c, ...d);
    faces.push(i, i + 1, i + 2, i, i + 2, i + 3);
  };
  for (let i = 0; i < total; i++) {
    const next = (i + 1) % total;
    levels.slice(0, -1).forEach((level, j) => quad(sample(i, level), sample(next, level), sample(next, levels[j + 1]), sample(i, levels[j + 1])));
    const first = levels[0], last = levels[3];
    quad(sample(i, first, true), sample(next, first, true), sample(next, first), sample(i, first));
    quad(sample(i, last), sample(next, last), sample(next, last, true), sample(i, last, true));
    quad(sample(i, first, true), sample(i, last, true), sample(next, last, true), sample(next, first, true));
  }
  return geometry(vertices, faces);
}

function bevelTeeth(pitch, width, count, helix, inside, taper) {
  const result = toothRing(pitch, width, count, helix, inside);
  const pos = result.getAttribute('position');
  for (let i = 0; i < pos.count; i++) {
    const scale = 1 + pos.getX(i) / width * taper;
    pos.setXYZ(i, pos.getX(i), pos.getY(i) * scale, pos.getZ(i) * scale);
  }
  result.computeVertexNormals();
  return result;
}

// Punched material between the hub and rim. Build the faces around real openings
// so the next gear is still visible through the blank as the view turns.
function perforatedWeb(radius, width, holes = 6, bore = radius * .28) {
  const points = [], indices = [];
  const outer = radius * .84, inner = bore, drill = radius * .116;
  const mid = (outer + inner) / 2;
  for (let h = 0; h < holes; h++) {
    const a = h * TAU / holes, half = Math.PI / holes;
    const path = [];
    for (let i = 0; i <= 8; i++) path.push([outer, a - half + 2 * half * i / 8]);
    path.push([mid, a + half]);
    for (let i = 8; i >= 0; i--) path.push([inner, a - half + 2 * half * i / 8]);
    path.push([mid, a - half]);
    const cy = mid * Math.cos(a), cz = mid * Math.sin(a);
    const bounds = path.map(([r, theta]) => {
      const y = r * Math.cos(theta), z = r * Math.sin(theta), norm = Math.hypot(y - cy, z - cz);
      return [y, z, cy + (y - cy) * drill / norm, cz + (z - cz) * drill / norm];
    });
    for (let i = 0; i < bounds.length; i++) {
      const [y1, z1, h1, k1] = bounds[i], [y2, z2, h2, k2] = bounds[(i + 1) % bounds.length];
      const left = -width / 2, right = -left, off = points.length / 3;
      points.push(left, y1, z1, left, y2, z2, left, h2, k2, left, h1, k1,
        right, y1, z1, right, y2, z2, right, h2, k2, right, h1, k1,
        left, h1, k1, left, h2, k2, right, h2, k2, right, h1, k1);
      for (const k of [0, 4, 8]) {
        if (k === 4) indices.push(off + k, off + k + 1, off + k + 2, off + k, off + k + 2, off + k + 3);
        else indices.push(off + k, off + k + 2, off + k + 1, off + k, off + k + 3, off + k + 2);
      }
    }
  }
  return geometry(points, indices);
}

function metal(color, roughness = .36, amount = .71) {
  return new T.MeshStandardMaterial({ color, metalness: amount, roughness, side: T.DoubleSide });
}

function palette() {
  return {
    steel: metal(0x697375), turned: metal(0xb5bfbd, .29, .79), shaded: metal(0x363d3e, .39),
    casting: metal(0x737b79, .62, .50), edge: metal(0xadb4b1, .44, .62),
    black: metal(0x242826, .73, .06), rubber: metal(0x151918, .92, .025), grip: metal(0x383a36, .77, .045),
    bronze: metal(0xc79b5d, .35), graphite: metal(0x393f40, .54, .45), ivory: metal(0xede8d8, .58, .14),
    copper: metal(0xb36642, .37, .56), teal: metal(0x467e78, .36, .57), friction: metal(0x65564b, .82, .12),
  };
}

// Bake small features into one draw call per rigid body and material.
function workshop() {
  const batches = new Map();
  const root = new T.Group();
  const materials = palette();
  const pairs = [], layers = [];
  const group = (parent = root, x = 0, y = 0, z = 0) => {
    const item = new T.Group();
    item.position.set(x, y, z); parent.add(item);
    return item;
  };
  // Each holder translates in gearbox coordinates. Shift actuators and rotating
  // parts retain their own local transforms under it throughout the assembly.
  function layer(parent, dx, dy, dz) {
    const node = group(parent);
    layers.push({ node, offset: v(dx, dy, dz), samples: [] });
    return node;
  }
  function part(parent, material, source, x = 0, y = 0, z = 0, angles = [0, 0, 0]) {
    const key = parent.id + '/' + material.id;
    if (!batches.has(key)) batches.set(key, { parent, material, positions: [], normals: [], indices: [] });
    const batch = batches.get(key);
    const position = source.getAttribute('position'), normal = source.getAttribute('normal');
    const rotation = angles.isQuaternion ? angles : new T.Quaternion().setFromEuler(new T.Euler(...angles));
    const offset = batch.positions.length / 3, vector = new T.Vector3();
    for (let i = 0; i < position.count; i++) {
      vector.fromBufferAttribute(position, i).applyQuaternion(rotation);
      batch.positions.push(vector.x + x, vector.y + y, vector.z + z);
      vector.fromBufferAttribute(normal, i).applyQuaternion(rotation);
      batch.normals.push(vector.x, vector.y, vector.z);
    }
    const index = source.index;
    if (index) for (let i = 0; i < index.count; i++) batch.indices.push(offset + index.getX(i));
    else for (let i = 0; i < position.count; i++) batch.indices.push(offset + i);
    source.dispose();
  }
  const block = (parent, mat, x, y, z, w, h, d, bevel, rot) =>
    part(parent, mat, roundedBlock(w, h, d, bevel), x, y, z, rot);
  const bush = (parent, mat, x, y, z, length, outer, inner = 0, segments = 40, rot) =>
    part(parent, mat, ring(length, outer, inner, segments), x, y, z, rot);
  function rod(parent, mat, a, b, radius, segments = 12) {
    const dir = v(...b).sub(v(...a));
    const rotation = new T.Quaternion().setFromUnitVectors(v(1, 0, 0), dir.clone().normalize());
    bush(parent, mat, (a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2, dir.length(), radius, 0, segments, rotation);
  }
  function pipe(parent, mat, points, radius) {
    for (let i = 1; i < points.length; i++) rod(parent, mat, points[i - 1], points[i], radius);
  }
  function finish() {
    for (const b of batches.values()) {
      const mesh = new T.Mesh(geometry(b.positions, b.indices, b.normals), b.material);
      mesh.geometry.computeBoundingBox();
      mesh.castShadow = mesh.receiveShadow = true;
      b.parent.add(mesh);
    }
    batches.clear();
  }
  return { root, materials, group, layer, part, block, bush, rod, pipe, pairs, layers, finish };
}

const UP = [0, 0, Math.PI / 2], AXLE = [0, -Math.PI / 2, 0];
const AXES = { input: [.12, -.14], lay: [-.06, .87] };
const distance = Math.hypot(AXES.input[0] - AXES.lay[0], AXES.input[1] - AXES.lay[1]);

function gearPair(w, n, x, toothCounts, pdk) {
  const { group, part, bush, materials: m } = w;
  const rim = metal(0x7b8383, .41, .76);
  const axial = .42 + x * .76;
  const first = group(w.layer(w.root, axial, .74, -.82), x, ...AXES.input);
  const second = group(w.layer(w.root, axial, .75, 1.13), x, ...AXES.lay);
  const c = toothCounts[0], d = toothCounts[1], face = pdk ? .235 : .265;
  for (const [shaft, teeth, direction] of [[first, c, 1], [second, d, -1]]) {
    const r = distance * teeth / (c + d);
    const onHollowInput = pdk && shaft === first && n % 2 === 0;
    const bore = onHollowInput ? .145 : .065;
    part(shaft, rim, toothRing(r, face, teeth, direction * .37, r * .76));
    if (r > .31) part(shaft, m.steel, perforatedWeb(r, face * .55, r > .62 ? 7 : 5, Math.max(r * .28, bore)));
    else bush(shaft, m.steel, 0, 0, 0, face * .71, r * .81, onHollowInput ? bore : .06);
    bush(shaft, m.turned, 0, 0, 0, face * 1.13, Math.max(bore + .041, r * .31), bore);
    bush(shaft, m.steel, -face * .42, 0, 0, face * .16, r * .78, r * .69);
  }
  // Opposite helices and inverse tooth-count speeds keep each pair in mesh.
  const contact = Math.atan2(AXES.lay[1] - AXES.input[1], AXES.lay[0] - AXES.input[0]);
  second.rotation.x = contact * (1 + c / d) + Math.PI + Math.PI / d;
  w.pairs.push({ n, first, second, c, d, rim, x });
}

function castingAndAxle(w, pdk) {
  const { root, group, part, block, bush, materials: m } = w;
  const frame = w.layer(root, .10, -.48, -.22);
  block(frame, m.casting, .17, -.68, .33, 4.48, .16, 1.44, .07);
  for (const z of [-.55, 1.31]) {
    block(frame, m.casting, .08, -.55, z, 4.64, .20, .14, .045);
    block(frame, m.edge, .10, -.43, z, 4.62, .06, .21, .027);
    for (let i = 0; i < 6; i++) {
      const x = -1.96 + .81 * i;
      block(frame, m.casting, x, -.61, z + (z > 0 ? -.065 : .065), .12, .36, .18, .035, [.12, 0, 0]);
      bush(frame, m.shaded, x, -.386, z, .043, .072, .030, 6, UP);
    }
  }
  for (const x of [-2.24, -1.31, 2.18]) {
    const radius = x < -1.5 ? 1.08 : .90;
    part(frame, m.casting, ring(.17, radius, radius - .11, 38, 1.74, 3.40, .022), x, ...AXES.input);
    part(frame, m.edge, ring(.045, radius + .025, radius - .12, 35, 1.74, 3.40), x - .08, ...AXES.input);
  }
  block(frame, m.casting, 2.22, -.35, .38, .17, .30, 1.74, .09);
  for (const axis of Object.values(AXES)) {
    const input = axis === AXES.input;
    if (!(pdk && input)) {
      const shaft = w.layer(root, input ? .95 : 1.34, .07, input ? -.39 : .49);
      bush(shaft, m.turned, .21, ...axis, 4.22, .105);
    }
    const endBearing = w.layer(root, input ? 1.91 : 2.18, .07, input ? -.39 : .49);
    bush(endBearing, m.shaded, 2.20, ...axis, .23, .204, .100);
    bush(endBearing, m.turned, 2.31, ...axis, .062, .171, .078);
    bush(endBearing, m.graphite, 2.35, ...axis, .040, .127, .063, 12);
  }
  if (pdk) {
    // The short hollow input stops at the end of the even gear train. Its
    // opened wall reveals the longer shaft reaching the four odd gears.
    const oddShaft = w.layer(root, .58, .19, -.60);
    const evenShaft = w.layer(root, -.56, .19, -.60);
    bush(oddShaft, m.copper, -.045, ...AXES.input, 4.49, .106);
    bush(oddShaft, m.turned, 2.26, ...AXES.input, .16, .069);
    part(evenShaft, m.teal, ring(1.80, .150, .110, 44, 1.65, TAU-1.10), -.72, ...AXES.input);
  }
  const pinion = group(w.layer(root, -.44, -.01, .52), -1.20, ...AXES.lay);
  part(pinion, m.turned, bevelTeeth(.206,.224,10,.35,.095,.25));
  const diff = group(w.layer(root, -.74, -.18, 1.70), -1.49, -.32, .93);
  diff.rotation.y = -Math.PI / 2;
  // The ring drives the carrier and two transverse half-shaft stubs.
  part(diff, m.turned, bevelTeeth(.61, .16, pdk ? 45 : 43, .40, .48, -.14), .13);
  part(diff, m.shaded, perforatedWeb(.64, .075, 7), .13);
  part(diff, m.steel, turn([[-.34, .095], [-.34, .17], [-.27, .24], [-.15, .34],
    [.11, .40], [.22, .36], [.31, .19], [.33, .105]], 42));
  for (const x of [-.31, .32]) bush(diff, m.turned, x, 0, 0, .074, .218, .095);
  bush(diff, m.shaded, .13, 0, 0, 1.42, .103);
  for (const x of [-.57, .78]) {
    bush(diff, m.steel, x, 0, 0, .17, .20, .078);
    for (let j = 0; j < 6; j++) {
      const a = j * TAU / 6;
      bush(diff, m.shaded, x + (x > 0 ? .09 : -.09), .148 * Math.cos(a), .148 * Math.sin(a), .035, .028, .010, 6);
    }
  }
  for (let i = 0; i < 8; i++) {
    const a = i * TAU / 8;
    bush(diff, m.turned, .224, .43 * Math.cos(a), .43 * Math.sin(a), .055, .050, 0, 6);
  }
  const gearCounts = pdk ? [[14,52],[20,47],[25,43],[29,39],[32,36],[35,33],[37,31]]
    : [[14,52],[20,47],[25,43],[29,39],[33,35],[36,32]];
  const positions = pdk ? [.36,-.98,.78,-.55,1.25,-.10,1.73] : [-.94,-.40,.14,.68,1.22,1.76];
  gearCounts.forEach((pair, i) => gearPair(w, i + 1, positions[i], pair, pdk));
  return { diff, pinion, frame };
}

function synchronizers(w, pdk) {
  const { root, group, bush, part, block, pipe, materials: m } = w;
  // The two PDK branches have independent selectors, including for 1st/2nd.
  const collars = pdk ? [[-.765,2,4],[.080,6,0],[.57,1,3],[1.49,5,7]] : [[-.67,1,2],[.41,3,4],[1.49,5,6]];
  const slides = [];
  const rail = w.layer(root, .62, 1.04, .01);
  bush(rail, m.turned, .16, .54, .07, 3.59, .047);
  for (const [x, left, right] of collars) {
    const slide = group(w.layer(root, .42 + .76 * x, 1.34, .42), x, ...AXES.lay);
    const forkMetal = pdk ? m.steel : metal(0x929c98, .38, .61);
    // A fork meets the groove from behind and above the output gear train.
    bush(slide, m.bronze, -.050, 0, 0, .077, .231, .114);
    bush(slide, m.bronze,  .050, 0, 0, .077, .231, .114);
    bush(slide, m.shaded, 0, 0, 0, .123, .217, .113);
    for (const dx of [-.035, .035]) bush(slide, m.turned, dx, 0, 0, .023, .246, .207);
    part(slide, forkMetal, ring(.055, .296, .246, 18, 4.07, 2.76), 0);
    pipe(slide, forkMetal, [[0,.20,-.17],[0,.45,-.44],[0,.59,-.77]], .046);
    bush(slide, m.steel, 0, .59, -.80, .14, .095, .039);
    block(slide, m.shaded, 0, .46, -.49, .082, .11, .13, .025);
    slides.push({ slide, x, left, right, forkMetal });
  }
  return slides;
}

// Release travel and disc clearance are exaggerated to make the action readable.
function manualClutch(w) {
  const { root, group, part, bush, block, pipe, materials: m } = w;
  const [y, z] = AXES.input;
  const flywheel = group(w.layer(root, -2.45, .37, -.19), -2.05, y, z);
  const disc = group(w.layer(root, -1.65, .37, -.19), -1.93, y, z);
  const cover = group(w.layer(root, -.94, .37, -.19), -1.80, y, z);
  const pressure = group(cover);
  const actuator = group(w.layer(root, -.40, .37, -.19), -1.21, y, z);
  const release = group(actuator, -.27);
  part(flywheel, m.turned, toothRing(.833, .12, 86, 0, .724, .025));
  part(flywheel, m.steel, ring(.111, .754, .588));
  part(flywheel, m.shaded, perforatedWeb(.736, .070, 8));
  bush(flywheel, m.turned, -.044, 0, 0, .16, .234, .110);
  const opening = 1.59, span = TAU - 1.12;
  part(flywheel, m.turned, ring(.032, .747, .465, 44, opening, span), .064);
  part(disc, m.friction, ring(.085, .752, .468, 47, opening, span));
  part(disc, m.bronze, ring(.035, .703, .439, 44, opening, span));
  part(disc, m.shaded, perforatedWeb(.601, .069, 6));
  bush(disc, m.turned, 0, 0, 0, .175, .187, .097);
  for (let i = 0; i < 14; i++) {
    const a = i * TAU / 14;
    if (a > .48 && a < opening) continue;
    bush(disc, m.bronze, -.048, .604 * Math.cos(a), .604 * Math.sin(a), .027, .023, .010, 10);
  }
  part(pressure, m.casting, turn([[-.068,.55],[-.068,.74],[-.028,.782],[.014,.785],[.059,.654],[.059,.547]],
    45, opening, span));
  part(pressure, m.turned, ring(.044, .766, .591, 42, opening, span), -.068);
  // The cover stays anchored to the flywheel. Its fulcrum lets the inner
  // spring fingers and the outer pressure-plate contact move in opposite directions.
  part(cover, m.steel, ring(.043, .566, .492, 39, opening, span), .15);
  const fingers = [];
  for (let i = 0; i < 20; i++) {
    const a = i * TAU / 20;
    if (a > .48 && a < opening) continue;
    const sector = group(cover);
    sector.rotation.x = a;
    const finger = group(sector, .15, .51, 0);
    finger.rotation.z = .34;
    block(finger, m.turned, 0, -.088, 0, .039, .49, .067, .013);
    block(finger, m.steel, 0, .096, 0, .049, .085, .088, .013);
    fingers.push(finger);
    if (i % 3 === 0) {
      block(sector, m.casting, .064, .706, 0, .13,.083,.088,.014);
      bush(sector, m.shaded, .137, .706, 0, .048,.052,.015,6);
    }
  }
  // A cutaway fixed cylinder exposes the telescoping piston and bearing collar.
  part(actuator, m.graphite, ring(.238,.240,.140,42,1.76,TAU-1.45));
  part(actuator, m.turned, ring(.043,.260,.138,40,1.76,TAU-1.45), .105);
  bush(actuator, m.bronze, -.112, 0, 0, .046,.202,.145);
  bush(release, m.turned, .153, 0, 0, .31,.15,.109);
  bush(release, m.teal, .012, 0, 0, .083,.236,.117);
  bush(release, m.turned, -.044, 0, 0, .059,.216,.113);
  bush(release, m.shaded, .069, 0, 0, .043,.204,.111);
  const port = [0,.335,.26];
  pipe(actuator, m.teal, [[.028,.151,.099], [.028,.252,.204], port], .048);
  bush(actuator, m.bronze, ...port, .109,.081,.034,6,UP);
  function setRelease(stroke, grip) {
    // The spring can reduce normal force while the linings still touch. A
    // section can show daylight only after the clutch has finished releasing.
    const clearance = grip > .00001 ? 0 : .142 * clearanceAt(stroke);
    release.position.x = -.27 - .219 * stroke;
    pressure.position.x = clearance;
    disc.position.x = -1.93 + .38 * clearance;
    for (const finger of fingers) finger.rotation.z = .34 - .68 * stroke;
  }
  return { flywheel, disc, cover, release, actuator, setRelease };
}

function cockpit(w) {
  const { root, group, part, block, bush, rod, pipe, materials: m } = w;
  // Cockpit pieces are lifted clear of the assembly so both control paths are visible.
  const bracket = group(w.layer(root, -1.05, .94, -.39), -.34,.27,-.07), pivot = [-2.58, 1.98, -.12];
  const pedal = group(bracket, ...pivot);
  for (const dx of [-.22, .22]) {
    block(bracket, m.casting, pivot[0] + dx, 1.91, -.24, .065, .38, .28, .035, [.20, 0, 0]);
    bush(bracket, m.turned, pivot[0] + dx, pivot[1], pivot[2], .080, .112, .055, 18);
  }
  bush(pedal, m.shaded, 0, 0, 0, .57, .060, 0, 20);
  block(pedal, m.steel, -.01, -.35, .09, .081, .72, .134, .034, [-.20, 0, -.04]);
  block(pedal, m.turned, -.050, -.65, .20, .133, .40, .108, .026, [-.53, 0, -.15]);
  const face = group(pedal, -.07, -.83, .26);
  face.rotation.x = -.33;
  block(face, m.shaded, 0, 0, .016, .44, .47, .09, .040);
  block(face, m.rubber, 0, 0, .079, .48, .44, .065, .030);
  for (let j = 0; j < 5; j++) {
    const y = (j - 2) * .074;
    block(face, m.grip, 0, y, .124, .37, .025, .023, .009);
  }
  const master = group(bracket, -2.76, 1.615, -.59);
  const hydraulic = metal(0x416f68, .37, .50);
  const unpressurised = new T.Color(0x456a63), pressurised = new T.Color(0x078d80);
  // A window in the master cylinder shows the piston as the pedal pushes its rod.
  part(master, m.graphite, ring(.48,.149,.100,36,1.76,TAU-1.73), 0,0,0, AXLE);
  for (const z of [-.218,.219]) bush(master, m.turned, 0,0,z, .058,.151,.090,24,AXLE);
  bush(master, m.bronze, 0,0,-.284, .126,.099,.035,6,AXLE);
  const piston = group(master,0,0,.16);
  bush(piston, m.teal, 0,0,0,.14,.096,.036,24,AXLE);
  bush(piston, m.turned, 0,0,.072,.037,.111,.044,24,AXLE);
  const clevis = group(pedal,-.074,-.38,.082);
  bush(clevis, m.bronze,0,0,0,.165,.067,.026,16);
  const pushrod = group(bracket);
  bush(pushrod, m.turned, 0,0,0,1,.042,0,14);
  const path = [[-2.76,1.615,-.90],[-2.50,1.615,-.90],[-2.30,1.55,-.84],[-2.15,1.40,-.73],
    [-1.99,1.10,-.54],[-1.78,.86,-.34],[-1.54,.70,-.17],[-1.20,.54,.025],[-.88,.40,.19],[-.87,.185,.19]];
  // This teaching layout lifts the pedal and brings the pressure line into view.
  // Its final fitting meets the port on the fixed release cylinder in the assembled pose.
  const tubing = w.layer(root, -.56,.53,-.30);
  const line = group(tubing,-.34,.27,-.07);
  pipe(line, hydraulic, path, .044);
  for (const index of [1,5]) {
    const a = path[index-1], b = path[index];
    rod(line, m.bronze, a.map((p,i) => p + .65*(b[i]-p)), a.map((p,i) => p + .87*(b[i]-p)), .059);
  }
  const pin = new T.Vector3(), pistonEnd = new T.Vector3(), direction = new T.Vector3();
  const rodAxis = v(1,0,0);
  function setPedal(stroke) {
    pedal.rotation.x = .66 * stroke;
    piston.position.z = .16 - .203 * stroke;
    pin.copy(clevis.position).applyQuaternion(pedal.quaternion).add(pedal.position);
    pistonEnd.copy(master.position).add(piston.position);
    pistonEnd.z += .083;
    direction.subVectors(pistonEnd, pin);
    pushrod.position.copy(pin).addScaledVector(direction, .5);
    pushrod.scale.x = direction.length();
    pushrod.quaternion.setFromUnitVectors(rodAxis, direction.normalize());
    hydraulic.color.copy(unpressurised).lerp(pressurised,stroke);
    hydraulic.emissive.setRGB(.002*stroke,.054*stroke,.038*stroke);
  }
  setPedal(0);

  const base = group(w.layer(root, .79, 1.05, -.65), .70, 1.08, -.69);
  block(base, m.graphite, 0, -.07, 0, .77, .15, .66, .065);
  block(base, m.turned, 0, .012, 0, .72, .043, .60, .021);
  for (let j = 0; j < 4; j++) {
    const r = .338 - j * .049;
    const boot = turn([[-.047, r * .70],[-.041, r *.92],[0, r],[.035,r * .88],[.05,r * .67]], 8, Math.PI / 8);
    part(base, j % 2 ? m.black : m.grip, boot, 0, .052 + j * .074, 0, UP);
  }
  const lever = group(base, 0, .20, 0);
  rod(lever, m.turned, [0,-.12,0],[.06,.70,.018], .055, 24);
  bush(lever, m.copper, .057, .628, .018, .065, .088, .042, 24, UP);
  part(lever, m.black, turn([[-.16,.061],[-.147,.109],[-.095,.170],[.020,.206],
    [.105,.196],[.144,.168],[.169,.121],[.171,0]], 40), .066, .787, .015, UP);
  bush(lever, m.turned, .066, .951, .015, .020, .122, 0, 36, UP);
  bush(lever, m.graphite, .066, .962, .015, .012, .112, 0, 36, UP);
  // H pattern on the knob. The extra forward-left branch is the reverse gate.
  const kx = .066, kz = .015, ky = .971;
  for (const x of [-.047, .009, .065]) rod(lever, m.ivory, [kx+x,ky,kz-.051],[kx+x,ky,kz+.051], .006, 6);
  rod(lever, m.ivory, [kx-.091,ky,kz],[kx+.065,ky,kz], .006, 6);
  rod(lever, m.ivory, [kx-.091,ky,kz],[kx-.091,ky,kz-.040], .006, 6);
  const cables = w.layer(root, .49, .64, -.45);
  for (const dx of [-.205,.205]) {
    pipe(cables, m.graphite, [[.70+dx,1.02,-.69],[.67+dx,.87,-.69],[.56+dx,.71,-.60],
      [.44+dx,.63,-.46],[.40+dx,.57,-.15],[.38+dx,.54,.07]], .031);
    bush(cables, m.bronze, .44 + dx, .63, -.48, .081, .058, .024, 8);
  }
  return { lever, pedalPoint: face, stickPoint: lever, linePoint: line, setPedal };
}

function wetClutches(w) {
  const { root, group, part, bush, materials: m } = w;
  const basket = group(root, -1.86, ...AXES.input);
  const oddMat = metal(0xb66d4b, .40, .56), evenMat = metal(0x398780, .38, .55);
  const oddDiscs = [], evenDiscs = [];
  const start = 1.80, range = TAU - 1.61;
  // Only the rear/bottom of the shell is retained, exposing both concentric packs.
  const shell = w.layer(basket, -2.78, .27, -.37);
  part(shell, m.edge, turn([[-.355,.78],[-.355,.897],[-.284,.925],[.34,.918],[.364,.873],[.32,.845],[-.264,.852]], 45, start, range));
  for (let i = 0; i < 10; i++) {
    const a = start + range * i / 10;
    bush(shell, m.shaded, -.361, .842 * Math.cos(a), .842 * Math.sin(a), .03, .041, .017, 9);
  }
  for (let i = 0; i < 7; i++) {
    const pack = group(w.layer(basket, -2.30 + .225 * i, .85, -.88), -.25 + i * .077);
    part(pack, oddMat, ring(.036, .837, .630, 42, 1.42, TAU-1.03));
    part(pack, m.friction, ring(.016, .814, .645, 42, 1.42, TAU-1.03), -.023);
    const inner = group(w.layer(basket, -2.36 + .215 * i, .12, .65), -.33 + i * .082);
    part(inner, evenMat, ring(.040, .595, .354, 43, 1.18, TAU-.62));
    part(inner, m.turned, ring(.018, .575, .376, 44, 1.18, TAU-.62), .029);
    oddDiscs.push(pack); evenDiscs.push(inner);
  }
  // The outer pack returns to the central shaft on the engine side. The inner
  // pack returns to the hollow shaft on the gearbox side, clear of the K1 web.
  const oddCarrier = group(w.layer(basket, -2.89, .85, -.88), -.398);
  const evenCarrier = group(w.layer(basket, -.54, .12, .65), .247);
  const oddHub = group(oddCarrier), evenHub = group(evenCarrier);
  part(oddHub, oddMat, perforatedWeb(.77, .052, 8));
  bush(oddHub, oddMat, 0, 0, 0, .072, .230, .099);
  part(evenHub, evenMat, perforatedWeb(.437, .050, 7, .145));
  bush(evenHub, evenMat, 0, 0, 0, .064, .180, .145);
  // Section the carrier walls on the same viewing side as the plates. Their
  // raised splines stay backed by the drum even when the stack is compressed.
  for (const [carrier, material, x, length, radius] of [
    [oddCarrier, oddMat, .314, .635, .627],
    [evenCarrier, evenMat, -.291, .592, .351],
  ]) {
    const from = 1.98, sweep = 4.02;
    part(carrier, material, ring(length, radius, radius - .025, 36, from, sweep), x);
    for (let i = 0; i < 7; i++) {
      const a = from + .12 + i * .61;
      part(carrier, material, ring(length, radius + .012, radius - .006, 3, a, .105, .003), x);
    }
  }
  return { oddHub, evenHub, oddDiscs, evenDiscs, oddMat, evenMat };
}

function mechatronics(w, frame) {
  const { root, group, block, bush, rod, pipe, materials: m } = w;
  const valve = group(root, -.47, 1.41, -.73);
  const body = w.layer(valve, -.21, .98, -.79);
  let labelledSolenoid;
  block(body, m.casting, 0, 0, 0, 1.56, .235, .70, .073);
  block(body, m.graphite, 0, .109, 0, 1.53, .042, .675, .020);
  block(body, m.edge, 0, .185, 0, 1.52, .123, .67, .052);
  for (let i = 0; i < 5; i++) {
    const x = -.592 + i * .295;
    const solenoid = w.layer(valve, -.21 + .44 * x, .98, .05);
    if (i === 1) labelledSolenoid = solenoid;
    bush(solenoid, m.turned, x, .094, .327, .25, .112, .039, 28, AXLE);
    bush(solenoid, m.shaded, x, .094, .476, .16, .104, 0, 24, AXLE);
    bush(solenoid, m.bronze, x, .094, .566, .058, .097, .040, 10, AXLE);
    bush(solenoid, m.graphite, x, .094, .599, .029, .050, 0, 10, AXLE);
    bush(body, m.shaded, x-.045, .25, -.26, .028, .043, .019, 8, UP);
    bush(body, m.shaded, x, .25, .23, .028, .043, .019, 8, UP);
    pipe(body, m.steel, [[x-.045,.249,-.26],[x-.045,.278,-.26],[x-.045,.278,-.13],
      [x+.076,.278,-.13],[x+.076,.278,.088],[x,.278,.088],[x,.278,.23],[x,.249,.23]], .023);
  }
  for (const x of [-.69,.69]) for (const z of [-.245,.245]) {
    bush(body, m.shaded, x, .246, z, .040, .051, .020, 6, UP);
  }
  const ecu = group(w.layer(root, 1.18, 1.13, -.65), 1.39, 1.58, -.76);
  block(ecu, m.shaded, 0, 0, 0, 1.26, .190, .80, .068);
  block(ecu, m.edge, 0, .095, 0, 1.27, .090, .81, .040);
  block(ecu, m.graphite, 0, .153, -.011, 1.17, .089, .702, .033);
  for (let i = 0; i < 9; i++) {
    const x = - .487 + .122 * i;
    block(ecu, m.edge, x, .222, -.041, .043, .118, .49, .020);
  }
  for (const x of [-.35, .31]) {
    block(ecu, m.black, x, .028, .426, .44, .176, .199, .039);
    block(ecu, m.grip, x, .015, .542, .42, .16, .11, .028);
    for (let i = 0; i < 5; i++) bush(ecu, m.bronze, x-.146+.072*i, .014, .604, .020, .018, 0, 7, AXLE);
  }
  for (const x of [-.561,.561]) for (const z of [-.330,.330]) bush(ecu, m.shaded, x, .166, z, .04, .035, .014, 6, UP);
  // The loom plugs into stationary housings at both ends.
  block(ecu, m.black, -.35, .015, .65, .38, .144, .141, .033);
  block(body, m.black, .785, .072, .10, .13, .151, .175, .027);
  const harness = w.layer(root, .52, .73, -.45);
  pipe(harness, m.black, [[1.04,1.595,-.06],[1.02,1.57,.02],[.85,1.49,.075],
    [.66,1.41,.025],[.48,1.42,-.16],[.45,1.48,-.57],[.35,1.482,-.63]], .043);

  // A fixed gallery on the clutch-end casting receives both clutch-pressure feeds.
  rod(frame, m.casting, [-1.31,.445,-.94], [-1.36,.78,-.51], .107);
  block(frame, m.casting, -1.36, .78, -.51, .37, .14, .22, .035);
  const pressureLines = w.layer(root, -.13, .52, -.55);
  for (const dx of [-.087, .087]) {
    bush(body, m.bronze, -.48+dx, -.126, -.06, .073, .064, .034, 6, UP);
    bush(frame, m.bronze, -1.36+dx, .862, -.51, .070, .064, .034, 6, UP);
    pipe(pressureLines, m.turned, [[-.95+dx,1.284,-.79],[-.95+dx,1.16,-.79],
      [-1.11+dx,1.05,-.76],[-1.31+dx,.99,-.58],[-1.36+dx,.94,-.51],[-1.36+dx,.86,-.51]], .037);
  }
  const mounts = w.layer(root, .21, .46, -.62);
  for (const x of [-.98, .07, .93, 1.83]) {
    const underside = x < .4 ? 1.293 : 1.485;
    block(mounts, m.edge, x, -.39, -.55, .16, .052, .16, .013);
    rod(mounts, m.casting, [x,-.398,-.55], [x,underside,-.81], .051, 8);
    block(mounts, m.edge, x, underside-.013, -.81, .15,.055,.16,.012);
  }
  return { valve: labelledSolenoid, ecu, pressureLines };
}

// Per-layer bounds follow the actual staggered pose even while outer assemblies
// remain spread and inner parts are already seated. Reproject vertices on orbit
// only; the opening moves each projected extent by a translation.
function assemblyRig(root, layers) {
  const fixed = { node: new T.Group(), offset: v(0,0,0), samples: [] };
  const parts = [fixed, ...layers];
  const owners = new Map(layers.map(part => [part.node, part]));
  const reach = Math.max(...layers.map(part => part.offset.length()));
  root.traverse(mesh => {
    if (!mesh.isMesh) return;
    let owner = mesh.parent;
    while (owner && !owners.has(owner)) owner = owner.parent;
    const bucket = (owners.get(owner) || fixed).samples;
    const pos = mesh.geometry.getAttribute('position');
    const stride = Math.max(1, Math.ceil(pos.count / 90));
    for (let i = 0; i < pos.count; i += stride) bucket.push(new T.Vector3().fromBufferAttribute(pos, i).applyMatrix4(mesh.matrixWorld));
  });
  for (const part of parts) {
    part.relativeReach = part.offset.length() / reach;
    part.low = part.samples.reduce((y, point) => Math.min(y,point.y), Infinity);
  }
  const seatedLow = Math.min(...parts.map(part => part.low));
  const lastRight = new T.Vector3(), lastUp = new T.Vector3();
  function bounds(right, up) {
    if (!lastRight.equals(right) || !lastUp.equals(up)) {
      lastRight.copy(right); lastUp.copy(up);
      for (const part of parts) {
        let left = Infinity, rightEdge = -Infinity, bottom = Infinity, top = -Infinity;
        for (const point of part.samples) {
          const x = point.dot(right), y = point.dot(up);
          left = Math.min(left, x); rightEdge = Math.max(rightEdge, x);
          bottom = Math.min(bottom, y); top = Math.max(top, y);
        }
        part.bounds = [left,rightEdge,bottom,top];
      }
    }
    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (const part of parts) {
      const dx = part.node.position.dot(right), dy = part.node.position.dot(up);
      xMin = Math.min(xMin, part.bounds[0] + dx); xMax = Math.max(xMax, part.bounds[1] + dx);
      yMin = Math.min(yMin, part.bounds[2] + dy); yMax = Math.max(yMax, part.bounds[3] + dy);
    }
    return [xMin,xMax,yMin,yMax];
  }
  function pose(opening) {
    let low = seatedLow, spread = 0;
    for (const part of parts) {
      const fraction = remainingOffset(opening, part.relativeReach);
      part.node.position.copy(part.offset).multiplyScalar(fraction);
      spread = Math.max(spread, fraction * part.relativeReach);
      low = Math.min(low, part.low + part.node.position.y);
    }
    return { floorY: -.982 + (low - seatedLow), spread };
  }
  return { bounds, pose };
}

const smooth = u => { const t = clamp(u,0,1); return t*t*(3-2*t); };

function makeModel(side) {
  const w = workshop(), pdk = side === 'pdk';
  const { diff, pinion, frame } = castingAndAxle(w, pdk);
  const slides = synchronizers(w, pdk);
  const clutch = pdk ? wetClutches(w) : manualClutch(w);
  const controls = pdk ? mechatronics(w, frame) : cockpit(w);
  w.finish();
  w.root.updateMatrixWorld(true);
  const gearLabels = pdk
    ? { odd: { dock: 'bottom-right', point: v(0,0,0) }, even: { dock: 'bottom-left', point: v(0,0,0) } }
    : { manual: { dock: 'bottom-right', point: v(0,.30,-.28) } };
  const anchors = pdk
    ? [{ label: 'Solenoid bank', object: controls.valve, point: v(-.33,.12,.51) },
       { label: 'Control unit', object: controls.ecu, point: v(.25,.27,-.02) },
       { label: 'Clutch feeds', dock: 'side', assembledOnly: true, object: controls.pressureLines, point: v(-1.397,.99,-.58) },
       gearLabels.even, gearLabels.odd]
    : [{ label: 'Clutch pedal', object: controls.pedalPoint, point: v(0,.08,.10) },
       { label: 'Gear stick', object: controls.stickPoint, point: v(.07,.88,.015) },
       { label: 'Pressure line', dock: 'side', assembledOnly: true, object: controls.linePoint, point: v(-1.78,.86,-.34) },
       { label: 'Release bearing', dock: 'bottom-left', assembledOnly: true, object: clutch.release, point: v(-.047,.132,.156) },
       gearLabels.manual];
  const assembly = assemblyRig(w.root, w.layers);
  const coupled = { manual: START.manualClutch, pedal: 0, odd: START.oddClutch, even: START.evenClutch };
  const selected = { manual: START.manualGear, odd: START.oddGear, even: START.evenGear };
  const weights = Array(8).fill(0);
  const steel = new T.Color(0x788382), copper = new T.Color(0x975b41), teal = new T.Color(0x376f6c);
  const oddOpen = new T.Color(0xad765b), oddClosed = new T.Color(0xcf7144);
  const evenOpen = new T.Color(0x5e8c87), evenClosed = new T.Color(0x269993);
  const forkTravel = .078;
  let state = { ...START, manualPedal: 0 }, transition = null;
  function colorGears() {
    for (const pair of w.pairs) {
      const branch = pdk ? (pair.n % 2 ? 'odd' : 'even') : 'manual';
      const accent = branch === 'even' ? teal : copper;
      pair.rim.color.copy(steel).lerp(accent, weights[pair.n] * (.37 + .63 * coupled[branch]));
    }
  }
  // The gate diagram, stick and fork all use the same instantaneous selector position.
  function setSelector({ lane, rail, stroke, gear }) {
    controls.lever.rotation.z = (1 - lane) * .23 - .035;
    controls.lever.rotation.x = stroke * .24;
    selected.manual = gear;
    const aligned = lane === rail;
    weights.fill(0);
    slides.forEach(({ slide, x, left, right, forkMetal }, i) => {
      const travel = i === rail ? stroke : 0;
      slide.position.x = x + forkTravel * travel;
      weights[travel < 0 ? left : right] = Math.abs(travel);
      forkMetal.color.setHex(i === rail && aligned ? 0xc16c3f : 0x929c98);
    });
    const fork = slides[rail];
    gearLabels.manual.object = aligned ? fork.slide : null;
    gearLabels.manual.label = fork.left + '–' + fork.right + ' selector fork';
    colorGears();
  }
  function update(next, animate = false) {
    const changingGear = pdk ? next.oddGear !== state.oddGear || next.evenGear !== state.evenGear :
      next.manualGear !== state.manualGear;
    const changingClutch = pdk ? next.oddClutch !== state.oddClutch || next.evenClutch !== state.evenClutch : next.manualClutch !== state.manualClutch || next.manualPedal !== state.manualPedal;
    const prepareFirst = pdk && ((next.oddGear !== state.oddGear && next.oddClutch > coupled.odd) ||
      (next.evenGear !== state.evenGear && next.evenClutch > coupled.even));
    state = next;
    if (animate && !changingGear && !changingClutch) return;
    const start = { ...coupled };
    const gears = [state.oddGear,state.evenGear];
    const from = pdk && slides.map(({slide,x}) => slide.position.x - x);
    const to = pdk && slides.map(({left,right}) => gears.includes(left) ? -forkTravel : gears.includes(right) ? forkTravel : 0);
    const unloadFirst = changingGear && state.manualClutch < start.manual;
    const selectFirst = changingGear && state.manualClutch > start.manual;
    const followsBiteCurve = !pdk && Math.abs(engagementAt(start.pedal) - start.manual) < .00001 &&
      Math.abs(engagementAt(state.manualPedal) - state.manualClutch) < .00001;
    function pose(u) {
      const pressureProgress = smooth(pdk ? (prepareFirst ? (u-.52)/.48 : u) :
        unloadFirst ? u/.49 : selectFirst ? (u-.55)/.45 : u);
      if (pdk) {
        coupled.odd = T.MathUtils.lerp(start.odd, state.oddClutch, pressureProgress);
        coupled.even = T.MathUtils.lerp(start.even, state.evenClutch, pressureProgress);
        clutch.oddDiscs.forEach((disc,i) => { disc.position.x = -.25 + i * (.077 - .024 * coupled.odd); });
        clutch.evenDiscs.forEach((disc,i) => { disc.position.x = -.33 + i * (.082 - .025 * coupled.even); });
        clutch.oddMat.color.copy(oddOpen).lerp(oddClosed,coupled.odd);
        clutch.evenMat.color.copy(evenOpen).lerp(evenClosed,coupled.even);
        const shiftProgress = smooth(prepareFirst ? u/.55 : u);
        weights.fill(0);
        selected.odd = 0; selected.even = 0;
        slides.forEach(({slide,x,left,right},i) => {
          let travel = to[i];
          if (u < 1 && Math.abs(from[i] - to[i]) > .00001) {
            // An outgoing sleeve reaches neutral before a new gear can be locked.
            travel = from[i] * (1 - smooth(shiftProgress/.46)) + to[i] * smooth((shiftProgress-.46)/.54);
          }
          slide.position.x = x + travel;
          const gear = travel < 0 ? left : right;
          if (gear) {
            weights[gear] = Math.abs(travel) / forkTravel;
            if (weights[gear] > .80) selected[gear % 2 ? 'odd' : 'even'] = gear;
          }
        });
        for (const branch of ['odd', 'even']) {
          const pair = w.pairs[selected[branch] - 1], label = gearLabels[branch];
          label.object = pair?.second.parent;
          if (!pair) continue;
          const radius = distance * pair.d / (pair.c + pair.d);
          label.point.copy(pair.second.position);
          label.point.y += .7 * radius; label.point.z += .6 * radius;
          label.label = (branch === 'odd' ? 'K1' : 'K2') + ' · gear ' + pair.n;
        }
      } else {
        coupled.pedal = T.MathUtils.lerp(start.pedal, state.manualPedal, pressureProgress);
        coupled.manual = followsBiteCurve ? engagementAt(coupled.pedal) : T.MathUtils.lerp(start.manual, state.manualClutch, pressureProgress);
        controls.setPedal(coupled.pedal);
        clutch.setRelease(coupled.pedal, coupled.manual);
      }
      colorGears();
    }
    if (animate) transition = { elapsed: 0, pose };
    else { transition = null; pose(1); }
  }
  function tick(dt, elapsed) {
    if (transition) {
      transition.elapsed += elapsed;
      const fraction = Math.min(1, transition.elapsed / 1.10);
      transition.pose(fraction);
      if (fraction === 1) transition = null;
    }
    const out = state.outputSpeed ?? (state.moving ? .99 : 0);
    function shaftSpeed(gear, freeSpeed) {
      const pair = w.pairs[gear - 1];
      return pair ? out * pair.d / pair.c : freeSpeed;
    }
    const freeEngine = 1.3;
    const manualInput = shaftSpeed(selected.manual, coupled.manual ? freeEngine : .6);
    const oddInput = pdk ? shaftSpeed(selected.odd, 0) : 0;
    const evenInput = pdk ? shaftSpeed(selected.even, 0) : 0;
    for (const pair of w.pairs) {
      const inSpeed = pdk ? (pair.n % 2 ? oddInput : evenInput) : manualInput;
      pair.first.rotation.x += dt * inSpeed;
      pair.second.rotation.x -= dt * inSpeed * pair.c / pair.d;
    }
    pinion.rotation.x -= dt * out;
    diff.rotateX(dt * out / (pdk ? 4.54 : 4.30));
    if (pdk) {
      // The sectioned pack faces stay exposed while the shaft hubs turn.
      clutch.oddHub.rotation.x += dt * oddInput;
      clutch.evenHub.rotation.x += dt * evenInput;
    } else {
      // The disc belongs to the gearbox input, the pressure plate and flywheel to the engine.
      const engineSpeed = manualInput * coupled.manual + freeEngine * (1 - coupled.manual);
      clutch.flywheel.rotation.x += dt * engineSpeed;
      clutch.cover.rotation.x += dt * engineSpeed;
      clutch.disc.rotation.x += dt * manualInput;
    }
  }
  return { model: w.root, assembly, anchors, setSelector, update, tick };
}

function labels(node, anchors) {
  const ns = 'http://www.w3.org/2000/svg';
  const element = tag => document.createElementNS(ns, tag);
  const svg = element('svg');
  svg.setAttribute('aria-hidden','true');
  svg.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;pointer-events:none;overflow:hidden';
  const items = anchors.map(anchor => {
    const leader = element('path'), dot = element('circle'), label = element('text');
    leader.setAttribute('fill','none'); leader.setAttribute('stroke','#757d73'); leader.setAttribute('stroke-width','1');
    dot.setAttribute('r','2.7'); dot.setAttribute('fill','#434e48'); dot.setAttribute('stroke','#f4f2e9'); dot.setAttribute('stroke-width','1');
    label.setAttribute('font-family','"IBM Plex Mono",ui-monospace,monospace');
    label.setAttribute('fill','#37433d'); label.setAttribute('stroke','#f5f3ed'); label.setAttribute('stroke-width','3');
    label.setAttribute('paint-order','stroke');
    if (anchor.dock) {
      const accent = anchor.dock === 'bottom-right' ? '#944b34' : '#32685d';
      leader.setAttribute('stroke',accent);
      dot.setAttribute('fill',accent);
      label.setAttribute('fill',accent);
    }
    svg.append(leader, dot, label);
    return { anchor, leader, dot, labelNode: label, world: new T.Vector3() };
  });
  node.append(svg);
  let connected = true;
  function draw(camera, width, height) {
    items.forEach(item => {
      const { object, point } = item.anchor;
      if (!object) return;
      const projected = object.localToWorld(item.world.copy(point)).project(camera);
      item.x = (projected.x + 1) * width / 2;
      item.y = (1 - projected.y) * height / 2;
    });
    const top = items.filter(item => !item.anchor.dock).sort((a,b) => a.x - b.x);
    items.forEach(item => {
      const { label, dock, object, assembledOnly } = item.anchor;
      const outside = item.x < 2 || item.y < 2 || item.x > width - 2 || item.y > height - 2;
      const hidden = outside || !object || (assembledOnly && !connected);
      for (const el of [item.leader, item.dot, item.labelNode]) el.style.visibility = hidden ? 'hidden' : '';
      if (hidden) return;
      if (item.labelNode.textContent !== label) item.labelNode.textContent = label;
      const left = dock ? dock !== 'bottom-right' : item === top[0], gap = clamp(width * .08, 14, 50);
      let x = left ? gap : width - gap, y = clamp(height * .10, 16, 34);
      if (dock) {
        x = clamp(width * .025, 6, 17);
        if (!left) x = width - x;
        y = dock === 'side' ? height * .57 : height - 8;
      }
      const size = width < 340 || (dock && width < 430) ? 10 : 11;
      const tagWidth = label.length * size * .59;
      const elbow = left ? x + tagWidth + 6 : x - tagWidth - 6;
      const leadEnd = left ? Math.max(elbow + 6, (elbow + item.x) / 2) : Math.min(elbow - 6, (elbow + item.x) / 2);
      item.leader.setAttribute('d', 'M' + elbow + ' ' + (y-3) + ' H' + leadEnd + ' L' + item.x + ' ' + item.y);
      item.dot.setAttribute('cx',item.x); item.dot.setAttribute('cy',item.y);
      item.labelNode.setAttribute('x',x); item.labelNode.setAttribute('y',y);
      item.labelNode.setAttribute('text-anchor',left ? 'start' : 'end');
      item.labelNode.setAttribute('font-size',size);
    });
  }
  return { svg, draw, setConnected(value) { connected = value; } };
}

function stage(side, node) {
  const canvas = document.createElement('canvas');
  canvas.setAttribute('aria-hidden','true');
  canvas.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;display:block;touch-action:pan-y pinch-zoom;cursor:grab';
  const attributes = { alpha: true, antialias: true, powerPreference: 'low-power' };
  const context = canvas.getContext('webgl2', attributes);
  if (!context) throw new Error('WebGL 2 is unavailable');
  const renderer = new T.WebGLRenderer({ canvas, context, ...attributes });
  let env, model, overlay, floor, light, contact;
  const scene = new T.Scene();
  const oldPosition = node.style.position, oldTabIndex = node.getAttribute('tabindex'), oldLabel = node.getAttribute('aria-label');
  function dispose() {
    const materials = new Set(Object.values(model?.materials || {}));
    scene.traverse(object => {
      if (object.isMesh) { object.geometry.dispose(); materials.add(object.material); }
    });
    for (const material of materials) material.dispose();
    light?.shadow.dispose(); env?.dispose(); renderer.dispose();
    canvas.remove(); overlay?.svg.remove();
    node.style.position = oldPosition;
    if (oldTabIndex === null) node.removeAttribute('tabindex');
    if (oldLabel === null) node.removeAttribute('aria-label');
  }
  try {
    renderer.setClearColor(0x000000, 0);
    renderer.toneMapping = T.ACESFilmicToneMapping;
    renderer.toneMappingExposure = .96;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = T.PCFShadowMap;
    const generator = new T.PMREMGenerator(renderer), room = new T.RoomEnvironment();
    try { env = generator.fromScene(room, .035); }
    finally { room.dispose(); generator.dispose(); }
    scene.environment = env.texture;
    scene.environmentIntensity = .64;
    scene.add(new T.HemisphereLight(0xfff9eb, 0xb0b7b7, .63));
    light = new T.DirectionalLight(0xfff3df, 2.24);
    light.position.set(-3.2, 6.5, 3.8);
    light.castShadow = true;
    light.shadow.mapSize.set(768, 768);
    Object.assign(light.shadow.camera, { left: -3.8, right: 3.8, top: 3.7, bottom: -3.7, near: .2, far: 16 });
    light.shadow.bias = -.00055; light.shadow.normalBias = .009;
    scene.add(light);
    const fill = new T.DirectionalLight(0xeaf3ff, .86);
    fill.position.set(3.4, 2.3, -5); scene.add(fill);
    model = makeModel(side);
    scene.add(model.model);
    floor = new T.Mesh(new T.PlaneGeometry(18, 18), new T.ShadowMaterial({ opacity: .105, depthWrite: false }));
    floor.rotation.x = -Math.PI/2; floor.position.y = -.982; floor.receiveShadow = true; floor.renderOrder = -1;
    scene.add(floor);
    contact = new T.Mesh(new T.PlaneGeometry(6.8,3.6), new T.ShaderMaterial({
      transparent:true, depthWrite:false, uniforms: { strength: { value: 1 } },
      vertexShader:'varying vec2 q;void main(){q=2.*uv-1.;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}',
      fragmentShader:'uniform float strength;varying vec2 q;void main(){float r=dot(q,q);float a=.16*strength*exp(-4.*r)*(1.-smoothstep(.65,1.,r));gl_FragColor=vec4(.16,.18,.16,a);}',
    }));
    contact.rotation.x = -Math.PI/2; contact.position.set(-.05,-.979,.38); contact.renderOrder = -2;
    scene.add(contact);
    if (getComputedStyle(node).position === 'static') node.style.position = 'relative';
    if (oldTabIndex === null) node.tabIndex = 0;
    if (oldLabel === null) node.setAttribute('aria-label',(side === 'manual' ? 'Manual' : 'PDK') + ' transmission cutaway. Use the arrow keys to rotate both views.');
    node.append(canvas);
    overlay = labels(node, model.anchors);
  } catch (error) { dispose(); throw error; }
  function assemble(mode, opening) {
    const { floorY, spread } = model.assembly.pose(opening);
    overlay.setConnected(mode === 'assembled');
    floor.position.y = floorY; contact.position.y = floorY + .003;
    contact.material.uniforms.strength.value = 1 - .65 * spread;
    const shadow = light.shadow.camera;
    shadow.right = 3.8 + 1.6 * spread; shadow.left = -shadow.right;
    shadow.top = 3.7 + 1.3 * spread; shadow.bottom = -shadow.top;
    shadow.updateProjectionMatrix();
  }
  const rect = node.getBoundingClientRect();
  const visible = rect.width > 0 && rect.height > 0 && rect.top < innerHeight && rect.bottom > 0 && rect.left < innerWidth && rect.right > 0;
  const [yaw, pitch] = VIEW_ANGLES.selector[side];
  return { side, yaw, pitch, node, canvas, renderer, scene, camera: new T.OrthographicCamera(-1,1,1,-1,.05,40),
    model, overlay, assemble, dispose, width: 0, height: 0, visible, dpr: 0 };
}

export function mountMechanisms({ manual, pdk, initialAssembly = 'assembled', onAssemblyChange, onFirstDraw, onOrbit }) {
  const stages = [];
  try {
    stages.push(stage('manual', manual));
    stages.push(stage('pdk', pdk));
  } catch (error) { stages.forEach(s => s.dispose()); throw error; }
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const handlers = new AbortController(), options = { signal: handlers.signal };
  const look = new T.Vector3(), right = new T.Vector3(), up = new T.Vector3(), focus = new T.Vector3();
  let zoom = 1, current = START, frame = 0, previous = 0, disposed = false;
  const opening = createOpening();
  const seatedTime = opening.hold + opening.span + opening.settle;
  let assemblyMode, targetTime, firstDraw = false;
  function fit(s) {
    const { width, height, camera, yaw, pitch } = s;
    const aspect = width / height;
    look.set(Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), Math.cos(yaw) * Math.cos(pitch));
    right.set(Math.cos(yaw), 0, -Math.sin(yaw));
    up.crossVectors(look, right).normalize();
    const [xMin,xMax,yMin,yMax] = s.model.assembly.bounds(right,up);
    const span = Math.max((xMax - xMin) / (aspect * .91), (yMax - yMin) / .84) / zoom;
    focus.copy(right).multiplyScalar((xMax + xMin) / 2).addScaledVector(up, (yMax + yMin) / 2 + span * .015);
    camera.left = -span * aspect / 2; camera.right = -camera.left;
    camera.top = span / 2; camera.bottom = -camera.top;
    camera.position.copy(focus).addScaledVector(look, 11);
    camera.lookAt(focus); camera.updateProjectionMatrix(); camera.updateMatrixWorld(true);
  }
  function applyAssembly() {
    for (const s of stages) {
      s.assemble(assemblyMode, opening);
      if (s.width && s.height) fit(s);
    }
  }
  function setAssembly(mode, immediate = false) {
    if (mode !== 'opening' && mode !== 'exploded' && mode !== 'assembled') return assemblyMode;
    if (mode === 'opening' && reduced.matches) mode = 'assembled';
    if (mode === assemblyMode && !immediate) return mode;
    const intro = assemblyMode === 'opening';
    const initial = !assemblyMode;
    assemblyMode = mode;
    targetTime = mode === 'exploded' ? opening.hold : seatedTime;
    if (mode === 'opening') opening.t = 0;
    else if (initial || immediate || reduced.matches || (intro && mode === 'assembled')) opening.t = targetTime;
    else opening.t = Math.max(opening.t, opening.hold);
    opening.on = opening.t !== targetTime;
    previous = 0;
    applyAssembly();
    onAssemblyChange?.(mode);
    requestDraw();
    return mode;
  }
  function redraw(now) {
    frame = 0;
    const onscreen = stages.some(s => s.visible && s.width && s.height);
    if (disposed || document.hidden || !onscreen) { previous = 0; return; }
    const elapsed = previous ? Math.max(0,(now - previous) / 1000) : 0;
    if (opening.on && elapsed) {
      if (assemblyMode === 'opening') {
        opening.t += elapsed;
        if (opening.t >= seatedTime) setAssembly('assembled');
        else if (opening.t > opening.hold) applyAssembly();
      } else {
        // Both directions use the same assembly path. Reversing midway starts at the current pose.
        const step = elapsed * (seatedTime - opening.hold) / .75;
        opening.t += clamp(targetTime - opening.t, -step, step);
        opening.on = opening.t !== targetTime;
        applyAssembly();
      }
    }
    const animate = current.running && !reduced.matches && assemblyMode !== 'opening';
    const dt = Math.min(elapsed, .08);
    for (const s of stages) {
      if (animate && dt) s.model.tick(dt, elapsed);
      if (!s.visible || !s.width || !s.height) continue;
      s.renderer.render(s.scene, s.camera);
      s.overlay.draw(s.camera, s.width, s.height);
    }
    if (!firstDraw) { firstDraw = true; onFirstDraw?.(); }
    // Start each visible interval after its first render, so shader setup and
    // background time cannot consume the one-second exploded hold.
    previous = opening.on || animate ? (previous ? now : performance.now()) : 0;
    if (opening.on || animate) requestDraw();
  }
  function requestDraw() {
    if (!disposed && !frame && !document.hidden) frame = requestAnimationFrame(redraw);
  }
  function resize() {
    for (const s of stages) {
      const width = s.node.clientWidth, height = s.node.clientHeight;
      const pixelRatio = clamp(devicePixelRatio || 1, 1.25, 1.75);
      if (s.width !== width || s.height !== height || s.dpr !== pixelRatio) {
        s.width = width; s.height = height; s.dpr = pixelRatio;
        if (width && height) {
          s.renderer.setPixelRatio(pixelRatio);
          s.renderer.setSize(width, height, false);
        }
      }
      if (width && height) fit(s);
    }
    requestDraw();
  }
  function setView(name) {
    if (!Object.hasOwn(VIEW_ANGLES, name)) return;
    for (const s of stages) [s.yaw, s.pitch] = VIEW_ANGLES[name][s.side];
    resize();
  }
  function orbit(dx, dy) {
    for (const s of stages) {
      s.yaw += dx; s.pitch = clamp(s.pitch + dy, -.26, 1.54);
      if (s.width && s.height) fit(s);
    }
    onOrbit?.();
    requestDraw();
  }
  for (const s of stages) {
    let drag;
    s.canvas.addEventListener('pointerdown', e => {
      if (!e.isPrimary || e.button !== 0) return;
      drag = { id:e.pointerId, x:e.clientX, y:e.clientY, lastX:e.clientX, lastY:e.clientY, moved:false };
      s.canvas.setPointerCapture(e.pointerId);
      s.canvas.style.cursor = 'grabbing';
    }, options);
    s.canvas.addEventListener('pointermove', e => {
      if (!drag || e.pointerId !== drag.id) return;
      if (Math.hypot(e.clientX - drag.x, e.clientY - drag.y) > 4) drag.moved = true;
      if (drag.moved) orbit((drag.lastX - e.clientX) * .009, (e.clientY - drag.lastY) * .007);
      drag.lastX = e.clientX; drag.lastY = e.clientY;
    }, options);
    const release = e => {
      if (!drag || e.pointerId !== drag.id) return;
      drag = null; s.canvas.style.cursor = 'grab';
    };
    s.canvas.addEventListener('pointerup', release, options);
    s.canvas.addEventListener('pointercancel', release, options);
    s.canvas.addEventListener('lostpointercapture', release, options);
    s.node.addEventListener('keydown', e => {
      if (e.target !== s.node) return;
      const moves = { ArrowLeft:[.12,0], ArrowRight:[-.12,0], ArrowUp:[0,.085], ArrowDown:[0,-.085] };
      if (Object.hasOwn(moves, e.key)) { e.preventDefault(); orbit(...moves[e.key]); }
    }, options);
    s.canvas.addEventListener('webglcontextrestored', resize, options);
  }
  const observer = new ResizeObserver(resize);
  const intersection = new IntersectionObserver(entries => {
    for (const entry of entries) {
      const s = stages.find(s => s.node === entry.target);
      if (s) s.visible = entry.isIntersecting;
    }
    if (!stages.some(s => s.visible)) previous = 0;
    requestDraw();
  });
  stages.forEach(s => { observer.observe(s.node); intersection.observe(s.node); });
  document.addEventListener('visibilitychange', () => { previous = 0; requestDraw(); }, options);
  reduced.addEventListener('change', () => {
    if (reduced.matches) {
      stages.forEach(s => s.model.update(current));
      if (opening.on) setAssembly(assemblyMode === 'opening' ? 'assembled' : assemblyMode, true);
    }
    requestDraw();
  }, options);
  function setState(state) {
    current = { ...START, ...state };
    current.manualPedal = clamp(state.manualPedal ?? (1 - current.manualClutch), 0, 1);
    const gear = current.manualGear;
    stages[0].model.setSelector(current.selector ?? (gear ? selectorPose(gear, gear, 0) : { lane: 1, rail: 1, stroke: 0, gear: 0 }));
    const animate = current.running && !reduced.matches && !document.hidden && assemblyMode !== 'opening' && stages.some(s => s.visible);
    stages.forEach(s => s.model.update(current, animate));
    requestDraw();
  }
  setState(START);
  setAssembly(initialAssembly);
  resize();
  return {
    setState, setAssembly, setView, resize,
    setSelector(pose) { stages[0].model.setSelector(pose); requestDraw(); },
    zoomBy(factor) {
      if (Number.isFinite(factor) && factor > 0) {
        zoom = clamp(zoom * factor, .65, 2.6);
        resize();
      }
      return zoom;
    },
    reset() { zoom = 1; setView('selector'); },
    dispose() {
      disposed = true;
      handlers.abort(); observer.disconnect(); intersection.disconnect();
      cancelAnimationFrame(frame);
      stages.forEach(s => s.dispose());
    },
  };
}
