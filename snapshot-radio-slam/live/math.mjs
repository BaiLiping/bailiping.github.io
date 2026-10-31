/**
 * Browser-safe educational radio-SLAM mathematics.
 * Based on arXiv:2607.04847v2, Eqs. (9)--(11), (18), (25)--(27).
 *
 * This is a conditional position/clock solver and orientation exploration,
 * not the paper's exhaustive robust Algorithm 1 or its IRLS/QAIC refinement.
 * Clock is parameterized B=c*b in metres: [M, v-u] [t; B] = rho (v-u).
 * Rotations map local vectors to the global frame. Euler controls use
 * R = Rz(yaw) Ry(pitch) Rx(roll). All measured angles are radians.
 */

export const C = 299792458;
export const radians = (degrees) => degrees * Math.PI / 180;
export const degrees = (angle) => angle * 180 / Math.PI;
export const dot = (a, b) => a.reduce((sum, x, i) => sum + x * b[i], 0);
export const norm = (a) => Math.hypot(...a);
export const add = (a, b) => a.map((x, i) => x + b[i]);
export const sub = (a, b) => a.map((x, i) => x - b[i]);
export const scale = (a, s) => a.map((x) => x * s);
export const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const identity = (n = 3) => Array.from({length: n}, (_, i) => Array.from({length: n}, (_, j) => +(i === j)));
export const transpose = (A) => A[0].map((_, j) => A.map((row) => row[j]));
export const matVec = (A, v) => A.map((row) => dot(row, v));
export const matMul = (A, B) => A.map((row) => transpose(B).map((col) => dot(row, col)));
export const skew = ([x, y, z]) => [[0, -z, y], [z, 0, -x], [-y, x, 0]];
const wrap = (angle) => ((angle + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;

export function rotationFromEulerDegrees(yawDeg, pitchDeg = 0, rollDeg = 0) {
  const y = radians(yawDeg), p = radians(pitchDeg), r = radians(rollDeg);
  const [cy, sy, cp, sp, cr, sr] = [Math.cos(y), Math.sin(y), Math.cos(p), Math.sin(p), Math.cos(r), Math.sin(r)];
  return [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
          [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
          [-sp, cp * sr, cp * cr]];
}

export function expSO3(delta) {
  const angle = norm(delta), K = skew(delta), K2 = matMul(K, K);
  const a = angle < 1e-8 ? 1 - angle * angle / 6 : Math.sin(angle) / angle;
  const b = angle < 1e-8 ? 0.5 - angle * angle / 24 : (1 - Math.cos(angle)) / (angle * angle);
  return identity().map((row, i) => row.map((x, j) => x + a * K[i][j] + b * K2[i][j]));
}

export function directionFromAngles(azimuth, elevation) {
  return [Math.cos(elevation) * Math.cos(azimuth), Math.cos(elevation) * Math.sin(azimuth), Math.sin(elevation)];
}

export function anglesFromDirection(v) {
  if (norm(v) === 0) throw new Error("A zero vector has no direction.");
  return [Math.atan2(v[1], v[0]), Math.atan2(v[2], Math.hypot(v[0], v[1]))];
}

/** One-sided Jacobi SVD. No normal equations, external library or truth input. */
export function svdLeastSquares(A, b, {relativeTolerance = 1e-11, maxSweeps = 80, columns = 4} = {}) {
  if (A.length !== b.length) throw new Error("A and b must have the same row count.");
  const m = A.length, n = m ? A[0].length : columns;
  if (!n || A.some((row) => row.length !== n || row.some((x) => !Number.isFinite(x))) || b.some((x) => !Number.isFinite(x))) {
    throw new Error("The linear system must be rectangular and finite.");
  }
  let magnitude = 0;
  for (const row of A) for (const x of row) magnitude = Math.max(magnitude, Math.abs(x));
  if (!m || magnitude === 0) return {solution: Array(n).fill(0), rank: 0, singularValues: Array(n).fill(0),
    modes: identity(n).map((vector) => ({sigma: 0, projection: 0, coefficient: 0, vector, retained: false, contribution: Array(n).fill(0)})),
    residual: b.map((x) => -x), cost: dot(b, b), conditionNumber: Infinity, sweeps: 0, converged: true};
  const B = A.map((row) => row.map((x) => x / magnitude));
  const V = identity(n);
  let sweeps = 0, converged = false;
  for (; sweeps < maxSweeps; sweeps++) {
    let changed = false;
    for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
      let alpha = 0, beta = 0, gamma = 0;
      for (let i = 0; i < m; i++) { alpha += B[i][p] ** 2; beta += B[i][q] ** 2; gamma += B[i][p] * B[i][q]; }
      if (alpha === 0 || beta === 0 || Math.abs(gamma) <= 2e-15 * Math.sqrt(alpha * beta)) continue;
      const tau = (beta - alpha) / (2 * gamma);
      const t = (tau >= 0 ? 1 : -1) / (Math.abs(tau) + Math.hypot(1, tau));
      const c = 1 / Math.hypot(1, t), s = c * t;
      for (let i = 0; i < m; i++) { const a = B[i][p], d = B[i][q]; B[i][p] = c * a - s * d; B[i][q] = s * a + c * d; }
      for (let i = 0; i < n; i++) { const a = V[i][p], d = V[i][q]; V[i][p] = c * a - s * d; V[i][q] = s * a + c * d; }
      changed = true;
    }
    if (!changed) { converged = true; break; }
  }
  const lengths = Array.from({length: n}, (_, j) => Math.hypot(...B.map((row) => row[j])));
  const order = Array.from({length: n}, (_, i) => i).sort((a, b) => lengths[b] - lengths[a]);
  const singularValues = order.map((j) => lengths[j] * magnitude);
  const cutoff = relativeTolerance * singularValues[0];
  const solution = Array(n).fill(0);
  const modes = [];
  let rank = 0;
  for (const j of order) {
    const sigma = lengths[j] * magnitude;
    let projection = 0;
    if (lengths[j] > 0) for (let i = 0; i < m; i++) projection += B[i][j] / lengths[j] * b[i];
    const retained = sigma > cutoff;
    const coefficient = sigma > 0 ? projection / sigma : 0;
    const vector = V.map((row) => row[j]);
    const contribution = vector.map((x) => retained ? x * coefficient : 0);
    modes.push({sigma, projection, coefficient, vector, retained, contribution});
    if (!retained) continue;
    rank++;
    for (let k = 0; k < n; k++) solution[k] += contribution[k];
  }
  const residual = matVec(A, solution).map((x, i) => x - b[i]);
  return {solution, rank, singularValues, modes, residual, cost: dot(residual, residual),
          conditionNumber: rank === n ? singularValues[0] / singularValues[n - 1] : Infinity,
          sweeps, converged};
}

/** Eq. (10), with global AoD u and global receiver-to-source AoA v. */
export function constraintMatrix(u, v) {
  const diagonal = dot(u, v) + 1;
  return u.map((_, i) => u.map((__, j) => v[i] * u[j] + u[i] * v[j] - (i === j ? diagonal : 0)));
}

function selectedMeasurements(snapshot, selectedIds) {
  if (selectedIds == null) return snapshot.measurements;
  const ids = new Set(selectedIds);
  return snapshot.measurements.filter((measurement) => ids.has(measurement.id));
}

/** Return the actual blocks shown in the teaching UI, not a surrogate system. */
export function stackSystem(snapshot, rotation, selectedIds = null) {
  const selected = selectedMeasurements(snapshot, selectedIds);
  const A = [], b = [], blocks = [];
  for (const measured of selected) {
    const u = matVec(snapshot.bsRotation, directionFromAngles(...measured.aoD));
    const vLocal = directionFromAngles(...measured.aoA);
    const v = matVec(rotation, vLocal);
    const M = constraintMatrix(u, v), difference = sub(v, u);
    const block = M.map((row, i) => [...row, difference[i]]);
    const rhs = scale(difference, measured.rho), firstRow = A.length;
    A.push(...block); b.push(...rhs);
    blocks.push({id: measured.id, u, v, vLocal, M, difference, block, rhs, rho: measured.rho,
                 rows: [firstRow, firstRow + 1, firstRow + 2]});
  }
  return {A, b, blocks, selectedIds: selected.map((m) => m.id)};
}

export function unifiedResiduals(snapshot, state) {
  const t = sub(state.position, snapshot.bsPosition);
  return stackSystem(snapshot, state.rotation).blocks.map((block) => {
    const vector = sub(matVec(block.M, t), scale(block.difference, block.rho - state.clockBiasM));
    return {id: block.id, vector, norm: norm(vector)};
  });
}

/**
 * Position and clock are estimated only from measured data and candidate R.
 * cost = sum_i ||r_i||^2 over selected paths, in square metres.
 * rmsResidual = sqrt(cost / N_selected_paths), the per-path vector-norm RMS.
 * It is not sqrt(cost / (3*N)), which would average scalar residual rows.
 * Full linear rank does not guarantee geometric feasibility, correct path
 * identity, a correct supplied rotation, or global nonlinear identifiability.
 */
export function conditionalSolve(snapshot, rotation, selectedIds = null) {
  const system = stackSystem(snapshot, rotation, selectedIds);
  const linear = svdLeastSquares(system.A, system.b);
  if (linear.rank < 4) return {...system, ...linear, state: null, residuals: [],
                              message: "Position and clock are not identifiable at this orientation with these selected paths."};
  const t = linear.solution.slice(0, 3), clockBiasM = linear.solution[3];
  const state = {position: add(snapshot.bsPosition, t), displacement: t, clockBiasM,
                 clockBiasNs: clockBiasM / C * 1e9, rotation: rotation.map((row) => [...row])};
  const selected = new Set(system.selectedIds);
  const residuals = unifiedResiduals(snapshot, state).map((r) => ({...r, included: selected.has(r.id)}));
  return {...system, ...linear, state, residuals, rmsResidual: Math.sqrt(linear.cost / Math.max(1, system.selectedIds.length)),
          message: "Conditional least-squares solution for the supplied orientation. Full linear rank does not guarantee geometric feasibility or global nonlinear identifiability."};
}

/** Eq. (27), derivative for Exp(delta) R with position and clock fixed. */
export function orientationJacobian(snapshot, state, selectedIds = null) {
  const t = sub(state.position, snapshot.bsPosition);
  const blocks = stackSystem(snapshot, state.rotation, selectedIds).blocks;
  return blocks.map(({u, v, rho}) => {
    const diagonal = dot(u, t) - (rho - state.clockBiasM);
    const G = u.map((_, i) => u.map((__, j) => (i === j ? diagonal : 0) + u[i] * t[j] - t[i] * u[j]));
    return matMul(G, skew(v)).map((row) => row.map((x) => -x));
  });
}

export function feasibilityMask(snapshot, state, {collinearityTolerance = 0.1, coplanarityCosine = 0.8} = {}) {
  const t = sub(state.position, snapshot.bsPosition), length = norm(t);
  if (length < 1e-12) return snapshot.measurements.map((m) => ({id: m.id, feasible: false, collinear: false, singleBounce: false}));
  const tHat = scale(t, 1 / length);
  return stackSystem(snapshot, state.rotation).blocks.map(({id, u, v}) => {
    const d1 = cross(tHat, v), d2 = cross(tHat, u), n1 = norm(d1), n2 = norm(d2);
    const collinear = n1 < collinearityTolerance && n2 < collinearityTolerance;
    const cosine = n1 * n2 > 1e-14 ? dot(d1, d2) / (n1 * n2) : null;
    const singleBounce = cosine !== null && cosine > coplanarityCosine && dot(tHat, v) < dot(tHat, u);
    return {id, feasible: collinear || singleBounce, collinear, singleBounce, cosine};
  });
}

/** Conditional yaw scan at fixed pitch/roll. This is not a full SO(3) search. */
export function yawSweep(snapshot, {pitchDeg = 0, rollDeg = 0, stepDeg = 2, selectedIds = null} = {}) {
  if (!(stepDeg > 0 && stepDeg <= 360)) throw new Error("Yaw step must lie in (0, 360].");
  const samples = [];
  let best = null;
  for (let yawDeg = -180; yawDeg < 180 - 1e-10; yawDeg += stepDeg) {
    const result = conditionalSolve(snapshot, rotationFromEulerDegrees(yawDeg, pitchDeg, rollDeg), selectedIds);
    samples.push({yawDeg, cost: result.state ? result.cost : Infinity, rank: result.rank});
    if (result.state && (!best || result.cost < best.cost)) best = {...result, yawDeg, pitchDeg, rollDeg};
  }
  return {samples, best, scope: "Yaw sweep with pitch and roll held fixed, using only the selected measured paths."};
}

/** Alternating Eq. (18) and Eq. (26). Local method, with no global guarantee. */
export function refineOrientation(snapshot, initialRotation, {selectedIds = null, maxIterations = 120, tolerance = 1e-9} = {}) {
  let result = conditionalSolve(snapshot, initialRotation, selectedIds);
  const history = result.state ? [{iteration: 0, cost: result.cost, stepNorm: null}] : [];
  if (!result.state) return {...result, history, iterations: 0, converged: false};
  const ids = result.selectedIds;
  let converged = false, iteration = 0;
  for (; iteration < maxIterations; iteration++) {
    const J = orientationJacobian(snapshot, result.state, ids).flat();
    const residual = result.blocks.flatMap(({M, difference, rho}) => sub(matVec(M, result.state.displacement), scale(difference, rho - result.state.clockBiasM)));
    const update = svdLeastSquares(J, residual.map((x) => -x), {columns: 3});
    const stepNorm = norm(update.solution);
    if (!Number.isFinite(stepNorm) || update.rank < 3) break;
    if (stepNorm < tolerance) { converged = true; break; }
    const next = conditionalSolve(snapshot, matMul(expSO3(update.solution), result.state.rotation), ids);
    if (!next.state) break;
    result = next;
    history.push({iteration: iteration + 1, cost: result.cost, stepNorm});
  }
  return {...result, history, iterations: history.length - 1, converged,
          scope: "Local alternating orientation refinement, without exhaustive subset search or IRLS/QAIC."};
}

function randomGenerator(seed) {
  let value = seed >>> 0;
  return () => { value += 0x6D2B79F5; let t = value; t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}

function normalPair(random) {
  const radius = Math.sqrt(-2 * Math.log(Math.max(random(), 1e-15))), phase = 2 * Math.PI * random();
  return [radius * Math.cos(phase), radius * Math.sin(phase)];
}

/**
 * Fixed diverse 3D teaching fixture, inspired by the independently verified
 * Python tests. Truth constructs the observations and is returned separately.
 * noise is the multiplier on standard deviations 0.3 m and 1 degree.
 * It is deliberately NOT called the paper's gamma (which scales covariance).
 * A path-specific random seed keeps its noisy samples stable as toggles change.
 */
export function makeSnapshot({yawDeg = 30, pitchDeg = 12, rollDeg = -8, clockBiasNs = 10,
                              noise = 0, seed = 12, includeLos = true, includeDoubleBounce = false} = {}) {
  if (![yawDeg, pitchDeg, rollDeg, clockBiasNs, noise].every(Number.isFinite) || noise < 0) throw new Error("Scenario controls must be finite and noise nonnegative.");
  const bsPosition = [0, 0, 2], bsRotation = identity(), position = [6, 4, 3.5];
  const rotation = rotationFromEulerDegrees(yawDeg, pitchDeg, rollDeg), clockBiasM = clockBiasNs * 1e-9 * C;
  const scatteringPoints = [[-3, 6, 7], [11, -6, 1], [14, 8, 10], [-5, -8, 4], [3, 11, 0], [9, 14, 9]];
  const paths = [];
  if (includeLos) paths.push({id: "los", kind: "LoS", noiseKey: 101, points: [bsPosition, position]});
  scatteringPoints.forEach((p, i) => paths.push({id: `single-${i + 1}`, kind: "Single bounce", noiseKey: i + 201, points: [bsPosition, p, position]}));
  if (includeDoubleBounce) paths.push({id: "double-1", kind: "Double bounce", noiseKey: 401,
                                      points: [bsPosition, [-6, 10, 10], [15, -5, 1], position]});
  const clean = {};
  const measurements = paths.map((path) => {
    const points = path.points;
    const outgoing = sub(points[1], bsPosition), arriving = sub(points[points.length - 2], position);
    const rho = points.slice(1).reduce((length, point, i) => length + norm(sub(point, points[i])), 0) + clockBiasM;
    const aoD = anglesFromDirection(matVec(transpose(bsRotation), outgoing));
    const aoA = anglesFromDirection(matVec(transpose(rotation), arriving));
    clean[path.id] = {rho, aoD: [...aoD], aoA: [...aoA]};
    const random = randomGenerator((seed ^ Math.imul(path.noiseKey, 2654435761)) >>> 0);
    const errors = [...normalPair(random), ...normalPair(random), ...normalPair(random)];
    const factor = path.kind === "LoS" ? 1 : path.kind === "Single bounce" ? 2 : 4;
    const rangeNoise = noise * factor * 0.3, angularNoise = noise * factor * radians(1);
    return {id: path.id, rho: rho + errors[0] * rangeNoise,
            aoD: [wrap(aoD[0] + errors[1] * angularNoise), aoD[1] + errors[2] * angularNoise],
            aoA: [wrap(aoA[0] + errors[3] * angularNoise), aoA[1] + errors[4] * angularNoise]};
  });
  return {snapshot: {bsPosition, bsRotation, measurements},
          truth: {position, rotation, clockBiasM, clockBiasNs, trueEuler: {yawDeg, pitchDeg, rollDeg}, cleanMeasurements: clean},
          paths, noiseDescription: "Standard-deviation multiplier on 0.3 m range and 1 degree angle noise, with path-class multipliers 1/2/4."};
}
