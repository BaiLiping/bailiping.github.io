// Deterministic teaching reductions of LiDAR-GS-SLAM. This is not the authors' SLAM system.
// Equations (2)–(6), (9) and Section 3.5: https://arxiv.org/html/2607.04127v1
export const clamp = (x, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, x));
export const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
export const norm = a => Math.sqrt(dot(a, a));
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export function seededRandom(seed = 7) {
  let state = seed >>> 0 || 7;
  return () => { state ^= state << 13; state ^= state >>> 17; state ^= state << 5; return ((state >>> 0) + .5) / 4294967296; };
}
function normalRandom(random) { return Math.sqrt(-2 * Math.log(random())) * Math.cos(2 * Math.PI * random()); }
export function quantile(values, q) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b), p = clamp(q) * (sorted.length - 1), i = Math.floor(p);
  return sorted[i] + (p - i) * (sorted[Math.min(i + 1, sorted.length - 1)] - sorted[i]);
}
export function sampleCovariance(points) {
  const mean = [0, 0, 0], matrix = Array.from({length: 3}, () => [0, 0, 0]);
  if (points.length < 2) throw new RangeError('At least two 3D samples are needed.');
  points.forEach(p => p.forEach((v, j) => mean[j] += v / points.length));
  for (const p of points) for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) matrix[i][j] += (p[i] - mean[i]) * (p[j] - mean[j]) / (points.length - 1);
  return {mean, matrix};
}
export function symmetricEigen3(matrix) {
  const a = matrix.map(row => [...row]), v = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let step = 0; step < 40; step++) {
    let p = 0, q = 1;
    for (const [i, j] of [[0, 2], [1, 2]]) if (Math.abs(a[i][j]) > Math.abs(a[p][q])) { p = i; q = j; }
    if (Math.abs(a[p][q]) < 1e-13) break;
    const phi = .5 * Math.atan2(2 * a[p][q], a[q][q] - a[p][p]), c = Math.cos(phi), s = Math.sin(phi);
    const app = a[p][p], aqq = a[q][q], apq = a[p][q];
    a[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
    a[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
    a[p][q] = a[q][p] = 0;
    for (let k = 0; k < 3; k++) {
      if (k !== p && k !== q) {
        const akp = a[k][p], akq = a[k][q];
        a[k][p] = a[p][k] = c * akp - s * akq;
        a[k][q] = a[q][k] = s * akp + c * akq;
      }
      const vp = v[k][p], vq = v[k][q]; v[k][p] = c * vp - s * vq; v[k][q] = s * vp + c * vq;
    }
  }
  const pairs = [0, 1, 2].map(i => ({value: Math.max(0, a[i][i]), vector: v.map(row => row[i])})).sort((a, b) => b.value - a.value);
  for (const p of pairs) {
    const axis = p.vector.reduce((best, n, i, arr) => Math.abs(n) > Math.abs(arr[best]) ? i : best, 0);
    if (p.vector[axis] < 0) p.vector = p.vector.map(x => -x);
  }
  if (dot(cross(pairs[0].vector, pairs[1].vector), pairs[2].vector) < 0) pairs[2].vector = pairs[2].vector.map(x => -x);
  return {values: pairs.map(p => p.value), vectors: pairs.map(p => p.vector)};
}
export function geometryFeatures(lambda) {
  const [a, b, c] = [...lambda].sort((x, y) => y - x);
  return {linearity: (a - b) / (a + 1e-9), curvature: c / (a + b + c + 1e-9), normalGap: (b - c) / (a + 1e-9)};
}
function rotateCloud(p) {
  const a = .47, b = -.36, c = .40;
  const x = p[0] * Math.cos(a) - p[1] * Math.sin(a), y = p[0] * Math.sin(a) + p[1] * Math.cos(a);
  const xx = x * Math.cos(b) + p[2] * Math.sin(b), z = -x * Math.sin(b) + p[2] * Math.cos(b);
  return [xx, y * Math.cos(c) - z * Math.sin(c), y * Math.sin(c) + z * Math.cos(c)];
}
export const covarianceDefaults = {
  plane: {tangent: .88, normal: .035, noise: .035},
  edge: {tangent: .10, normal: .10, noise: .035},
  scattered: {tangent: 1, normal: 1, noise: .035}
};
export function initialScales(std, range, factor = .04, maximum = 2) {
  return std.slice(0, 2).map(s => clamp(factor * Math.max(range, 1) * s, 0, maximum));
}
export function covarianceLab({shape = 'plane', tangent, noise = .035, range = 12, seed = 7} = {}) {
  const preset = covarianceDefaults[shape] || covarianceDefaults.plane;
  tangent ??= preset.tangent;
  const random = seededRandom(seed), scales = [1, tangent, preset.normal];
  const points = Array.from({length: 180}, () => rotateCloud(scales.map(s => s * normalRandom(random) + noise * normalRandom(random))));
  const {mean, matrix} = sampleCovariance(points), eigen = symmetricEigen3(matrix), features = geometryFeatures(eigen.values);
  const normalizedStd = eigen.values.map(v => Math.sqrt(v / Math.max(eigen.values[0], 1e-12)));
  const diskScales = initialScales(normalizedStd, range);
  const normalStatus = features.curvature > .18 ? 'No dominant surface normal' : features.normalGap > .25 ? 'Well-separated normal' : features.normalGap > .07 ? 'Weakly separated normal' : 'Normal is ambiguous';
  return {shape, tangent, noise, range, seed, points, mean, matrix, eigen, features, normalizedStd, diskScales, normal: eigen.vectors[2], normalStatus};
}

// Tracking uses the paper's min–max ratio normalization, with epsilon = .001
// chosen for this teaching example. It does not use map-control score c_i.
export function trackingWeights(primitives, enabled = true, epsilon = .001) {
  const ratios = primitives.map(g => Math.max(...g.scales) / (Math.min(...g.scales) + 1e-9));
  const min = Math.min(...ratios), max = Math.max(...ratios), span = max - min;
  return ratios.map(ratio => {
    const normalized = span > 1e-10 ? clamp((ratio - min) / span) : 0;
    return {ratio, normalized, weight: enabled ? Math.min(1 / (normalized + epsilon), 10) : 1};
  });
}
export function transformPoint(p, pose) {
  const c = Math.cos(pose[2]), s = Math.sin(pose[2]); return [c * p[0] - s * p[1] + pose[0], s * p[0] + c * p[1] + pose[1]];
}
function inversePoint(p, pose) {
  const c = Math.cos(pose[2]), s = Math.sin(pose[2]), x = p[0] - pose[0], y = p[1] - pose[1]; return [c * x + s * y, -s * x + c * y];
}
function covariance2(major, minor, angle) {
  const c = Math.cos(angle), s = Math.sin(angle), a = major * major, b = minor * minor;
  return [[a * c * c + b * s * s, (a - b) * c * s], [(a - b) * c * s, a * s * s + b * c * c]];
}
function inverse2(a) {
  const det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
  if (!(det > 0)) throw new RangeError('A positive definite covariance is required.');
  return [[a[1][1] / det, -a[0][1] / det], [-a[1][0] / det, a[0][0] / det]];
}
function multiply2(a, v) { return [dot(a[0], v), dot(a[1], v)]; }
export function makeTrackingScene({bias = .45, noise = .025, seed = 31} = {}) {
  const random = seededRandom(seed), truth = [.45, .25, 5 * Math.PI / 180], points = [];
  const add = (target, type, angle, i) => {
    const unstable = type === 'mixed', ratio = unstable ? 3.2 + 2.8 * random() : 1.02 + .2 * random();
    const scales = [.28 * ratio, .28];
    const targetCovariance = covariance2(unstable ? .10 : .32, unstable ? .09 : .07, angle);
    // Isotropic source covariance makes R C_s R^T = C_s exactly in this toy.
    const totalCovariance = targetCovariance.map((r, j) => r.map((x, k) => x + (j === k ? .04 ** 2 : 0)));
    const world = target.map((v, j) => v + noise * normalRandom(random) + (unstable ? bias * (j === 0 ? 1 : .35) : 0));
    points.push({id: points.length, target, source: inversePoint(world, truth), type, scales, targetCovariance, totalCovariance, precision: inverse2(totalCovariance), angle, sample: i});
  };
  for (let i = 0; i < 15; i++) add([-3 + .38 * i, -1.2], 'plane', 0, i);
  for (let i = 0; i < 11; i++) add([2.55, -.82 + .30 * i], 'plane', Math.PI / 2, i);
  for (let i = 0; i < 16; i++) add([-1.8 + .20 * i, 1.20 + .32 * Math.sin(.70 * i)], 'mixed', 0, i);
  return {points, truth, bias, noise, seed};
}
export function trackingObjective(scene, pose, weighted = true) {
  const weights = trackingWeights(scene.points, weighted);
  return scene.points.reduce((sum, p, i) => {
    const q = transformPoint(p.source, pose), d = q.map((x, j) => x - p.target[j]);
    return sum + weights[i].weight * dot(d, multiply2(p.precision, d));
  }, 0);
}
function solve3(a, b) {
  const aug = a.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < 3; col++) {
    let pivot = col; for (let i = col + 1; i < 3; i++) if (Math.abs(aug[i][col]) > Math.abs(aug[pivot][col])) pivot = i;
    if (Math.abs(aug[pivot][col]) < 1e-14) return [0, 0, 0];
    [aug[pivot], aug[col]] = [aug[col], aug[pivot]];
    const divisor = aug[col][col]; for (let j = col; j < 4; j++) aug[col][j] /= divisor;
    for (let i = 0; i < 3; i++) if (i !== col) { const f = aug[i][col]; for (let j = col; j < 4; j++) aug[i][j] -= f * aug[col][j]; }
  }
  return aug.map(row => row[3]);
}
export function trackingStep(scene, pose, weighted = true) {
  const weights = trackingWeights(scene.points, weighted), hessian = Array.from({length: 3}, () => [0, 0, 0]), gradient = [0, 0, 0];
  const c = Math.cos(pose[2]), s = Math.sin(pose[2]);
  scene.points.forEach((p, i) => {
    const q = transformPoint(p.source, pose), d = q.map((x, j) => x - p.target[j]);
    const jacobian = [[1, 0], [0, 1], [-s * p.source[0] - c * p.source[1], c * p.source[0] - s * p.source[1]]];
    for (let j = 0; j < 3; j++) {
      gradient[j] += weights[i].weight * dot(jacobian[j], multiply2(p.precision, d));
      for (let k = 0; k < 3; k++) hessian[j][k] += weights[i].weight * dot(jacobian[j], multiply2(p.precision, jacobian[k]));
    }
  });
  for (let j = 0; j < 3; j++) hessian[j][j] += 1e-8;
  const delta = solve3(hessian, gradient), before = trackingObjective(scene, pose, weighted);
  for (let factor = 1; factor >= 1 / 1024; factor /= 2) {
    const next = pose.map((v, i) => v - factor * delta[i]), after = trackingObjective(scene, next, weighted);
    if (after <= before + 1e-10) return {pose: next, before, after, factor, converged: norm(delta) * factor < 1e-7, accepted: true};
  }
  return {pose: [...pose], before, after: before, factor: 0, converged: true, accepted: false};
}
export function trackingError(scene, pose) {
  return {translation: Math.hypot(pose[0] - scene.truth[0], pose[1] - scene.truth[1]), angle: Math.abs(Math.atan2(Math.sin(pose[2] - scene.truth[2]), Math.cos(pose[2] - scene.truth[2]))) * 180 / Math.PI};
}
export class TrackingLab {
  constructor(options = {}) { this.reset(options); }
  reset({weighted = true, bias = .45, heading = 18, noise = .025} = {}) {
    this.scene = makeTrackingScene({bias, noise}); this.weighted = weighted; this.heading = heading;
    this.pose = [-.25, .62, this.scene.truth[2] + heading * Math.PI / 180]; this.iteration = 0; this.converged = false;
    this.trace = [trackingObjective(this.scene, this.pose, this.weighted)];
    return this.state();
  }
  step() {
    if (this.converged) return this.state();
    const result = trackingStep(this.scene, this.pose, this.weighted);
    this.pose = result.pose; this.converged = result.converged; this.iteration++; this.trace.push(result.after);
    return this.state();
  }
  solve(limit = 24) { for (let i = 0; i < limit && !this.converged; i++) this.step(); return this.state(); }
  state() { return {scene: this.scene, pose: [...this.pose], weighted: this.weighted, iteration: this.iteration, converged: this.converged, trace: [...this.trace], error: trackingError(this.scene, this.pose), weights: trackingWeights(this.scene.points, this.weighted)}; }
}

// These normalization choices follow the released feature computation: 5th–95th
// percentile clipping for geometry, residual divided by its 95th percentile.
// The population and map-management budgets below are deliberately small toy values.
export function controlScores(items) {
  const features = items.map(g => geometryFeatures(g.lambda));
  const loL = quantile(features.map(f => f.linearity), .05), hiL = quantile(features.map(f => f.linearity), .95);
  const loC = quantile(features.map(f => f.curvature), .05), hiC = quantile(features.map(f => f.curvature), .95);
  const hiR = quantile(items.map(g => g.residual), .95);
  return items.map((g, i) => {
    const raw = features[i], l = clamp((raw.linearity - loL) / Math.max(hiL - loL, 1e-9)), c = clamp((raw.curvature - loC) / Math.max(hiC - loC, 1e-9)), r = clamp(g.residual / Math.max(hiR, 1e-9));
    return {...g, features: raw, normalized: {linearity: l, curvature: c, residual: r}, score: clamp(.55 * l + .30 * c + .15 * r)};
  });
}
function pointsOnPath(vertices, step) {
  const out = [];
  for (let i = 1; i < vertices.length; i++) {
    const a = vertices[i - 1], b = vertices[i], length = Math.hypot(b[0] - a[0], b[1] - a[1]), count = Math.ceil(length / step), angle = Math.atan2(b[1] - a[1], b[0] - a[0]);
    for (let j = 0; j < count; j++) { const t = j / count; out.push({x: a[0] + t * (b[0] - a[0]), y: a[1] + t * (b[1] - a[1]), angle, corner: j === 0}); }
  }
  const last = vertices[vertices.length - 1]; out.push({x: last[0], y: last[1], angle: Math.PI, corner: true}); return out;
}
export function makeBudgetScene(seed = 19) {
  const random = seededRandom(seed), raw = [];
  const add = (x, y, angle, type, scales) => {
    const prescribed = type === 'plane' ? [1, .94, .003] : type === 'corner' ? [1, .62, .27] : [1, .025, .020];
    const samples = Array.from({length: 48}, () => prescribed.map(v => Math.sqrt(v) * normalRandom(random)));
    const lambda = symmetricEigen3(sampleCovariance(samples).matrix).values;
    raw.push({id: raw.length, x, y, angle, type, scales, lambda, opacity: .72 + .25 * random(), residual: (type === 'plane' ? .008 : .04) + .025 * random(), operation: 'unchanged'});
  };
  for (let i = 0; i < 11; i++) for (let j = 0; j < 10; j++) add(-3.4 + .36 * i + .015 * (random() - .5), -1.62 + .36 * j + .015 * (random() - .5), 0, 'plane', [.29, .28]);
  const path = [[1, -1.45], [2.2, -1.45], [2.2, -.3], [3.2, -.3], [3.2, 1.45], [1.2, 1.45]];
  const detail = pointsOnPath(path, .25);
  detail.forEach(p => add(p.x, p.y, p.angle, p.corner ? 'corner' : 'edge', [.19, .10]));
  const samples = [];
  for (let x = -3.58; x <= .39 + 1e-8; x += .11) for (let y = -1.80; y <= 1.80 + 1e-8; y += .11) samples.push({x, y, type: 'plane'});
  pointsOnPath(path, .04).forEach(p => samples.push({...p, type: 'detail'}));
  return {gaussians: controlScores(raw), samples, path, seed};
}
export function contains(g, p, slack = 1e-9) {
  const dx = p.x - g.x, dy = p.y - g.y, c = Math.cos(g.angle), s = Math.sin(g.angle);
  return ((dx * c + dy * s) / g.scales[0]) ** 2 + ((-dx * s + dy * c) / g.scales[1]) ** 2 <= 1 + slack;
}
export function ellipseBoundary(g, n = 32) {
  const c = Math.cos(g.angle), s = Math.sin(g.angle);
  return Array.from({length: n}, (_, i) => { const t = 2 * Math.PI * i / n, x = Math.cos(t) * g.scales[0], y = Math.sin(t) * g.scales[1]; return {x: g.x + c * x - s * y, y: g.y + s * x + c * y}; });
}
function coveringRepresentative(representative, members) {
  const c = Math.cos(representative.angle), s = Math.sin(representative.angle);
  let sx = representative.scales[0], sy = representative.scales[1];
  // Each ellipse is contained in its exact axis-aligned support bounds in the
  // representative frame. sqrt(2) encloses that rectangle in a larger ellipse.
  for (const g of members) {
    const a = g.angle - representative.angle, ca = Math.cos(a), sa = Math.sin(a), dx = g.x - representative.x, dy = g.y - representative.y;
    const extentX = Math.hypot(g.scales[0] * ca, g.scales[1] * sa), extentY = Math.hypot(g.scales[0] * sa, g.scales[1] * ca);
    sx = Math.max(sx, Math.abs(dx * c + dy * s) + extentX); sy = Math.max(sy, Math.abs(-dx * s + dy * c) + extentY);
  }
  return {...representative, scales: [Math.SQRT2 * sx, Math.SQRT2 * sy], operation: 'expanded', coveredIds: members.map(g => g.id)};
}
export function manageBudget(scene, {prune = true, split = true, cell = .9, splitQuantile = .82, maxPruneRatio = .35, maxSplitRatio = .12} = {}) {
  const originals = scene.gaussians, valid = originals.filter(g => g.score > 0 && g.opacity >= .5);
  const planeThreshold = quantile(valid.map(g => g.score), .35), splitThreshold = quantile(valid.map(g => g.score), splitQuantile);
  let gaussians = originals.map(g => ({...g, scales: [...g.scales], operation: 'unchanged'}));
  const removed = [], expanded = [], splitParents = [], groups = new Map(), pruneBudget = Math.floor(originals.length * maxPruneRatio);
  if (prune) {
    for (const g of valid) if (g.score <= planeThreshold) {
      const key = Math.floor((g.x + 4) / cell) + ':' + Math.floor((g.y + 2.1) / cell);
      if (!groups.has(key)) groups.set(key, []); groups.get(key).push(g);
    }
    for (const group of [...groups.values()].sort((a, b) => b.length - a.length)) {
      if (group.length < 2 || removed.length >= pruneBudget) continue;
      const ordered = [...group].sort((a, b) => b.opacity - a.opacity), representative = ordered[0];
      const redundant = ordered.slice(1, 1 + Math.min(ordered.length - 1, pruneBudget - removed.length));
      const replacement = coveringRepresentative(representative, [representative, ...redundant]);
      // A toy size safeguard: never expand a support radius beyond 1.5 metres.
      if (Math.max(...replacement.scales) > 1.5) continue;
      const ids = new Set(redundant.map(g => g.id));
      gaussians = gaussians.filter(g => !ids.has(g.id)).map(g => g.id === representative.id ? replacement : g);
      removed.push(...redundant); expanded.push(replacement);
    }
  }
  if (split) {
    const candidates = gaussians.filter(g => g.score > 0 && g.score >= splitThreshold && g.opacity >= .5 && g.operation !== 'expanded').sort((a, b) => b.score - a.score).slice(0, Math.floor(originals.length * maxSplitRatio));
    const ids = new Set(candidates.map(g => g.id));
    gaussians = gaussians.filter(g => !ids.has(g.id));
    for (const g of candidates) {
      splitParents.push(g);
      for (const sign of [-1, 1]) gaussians.push({...g, id: `${g.id}:${sign}`, parentId: g.id, x: g.x + sign * .45 * g.scales[0] * Math.cos(g.angle), y: g.y + sign * .45 * g.scales[0] * Math.sin(g.angle), scales: g.scales.map(s => .72 * s), operation: 'child'});
    }
  }
  const cover = population => scene.samples.filter(p => population.some(g => contains(g, p))).length / scene.samples.length;
  const beforeCoverage = cover(originals), coverage = cover(gaussians), area = gaussians.reduce((sum, g) => sum + Math.PI * g.scales[0] * g.scales[1], 0);
  return {originals, gaussians, removed, expanded, splitParents, planeThreshold, splitThreshold, beforeCoverage, coverage, area, pruneBudget, splitBudget: Math.floor(originals.length * maxSplitRatio), settings: {prune, split, cell, splitQuantile}};
}
