const { test } = require('node:test');
const assert = require('node:assert/strict');
const M = require('../live/extent-model.js');
const { add, scale, multiply, transpose, inverse2, sqrt2, sandwich, identity } = M.math;
const near = (a, b, tol = 1e-9) => assert.ok(Math.abs(a - b) < tol, a + ' != ' + b);
const nearMatrix = (A, B, tol = 1e-9) => A.forEach((r, i) => r.forEach((v, j) => near(v, B[i][j], tol)));
function positiveDefinite(A) {
  const L = A.map(row => row.map(() => 0));
  for (let i = 0; i < A.length; i++) for (let j = 0; j <= i; j++) {
    let s = A[i][j];
    for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
    if (i === j) { assert.ok(s > 0 && Number.isFinite(s), 'positive Cholesky pivot'); L[i][j] = Math.sqrt(s); }
    else L[i][j] = s / L[j][j];
    near(A[i][j], A[j][i]);
  }
}

test('sources are uniform over physical ellipse area, with covariance X/4', () => {
  const data = M.sampleScan({ count: 80000, length: 10, width: 4, angle: 35, noise: 0.6, seed: 73 }, 1);
  const inverse = inverse2(data.truth.X);
  let squaredRadius = 0, nx = 0, ny = 0;
  for (let i = 0; i < data.sources.length; i++) {
    const v = data.sources[i].map((x, j) => x - data.truth.center[j]);
    const r2 = v[0] ** 2 * inverse[0][0] + 2 * v[0] * v[1] * inverse[0][1] + v[1] ** 2 * inverse[1][1];
    assert.ok(r2 <= 1 + 1e-12);
    squaredRadius += r2;
    nx += (data.detections[i][0] - data.sources[i][0]) ** 2;
    ny += (data.detections[i][1] - data.sources[i][1]) ** 2;
  }
  near(squaredRadius / data.sources.length, 0.5, 0.006);
  const stats = M.measurementStatistics(data.sources);
  nearMatrix(scale(stats.scatter, 1 / stats.count), scale(data.truth.X, 0.25), 0.06);
  near(nx / data.sources.length, 0.36, 0.006); near(ny / data.sources.length, 0.36, 0.006);
  // Real sampled clouds are not forced to have their mean at the true center.
  assert.ok(Math.hypot(...M.measurementStatistics(data.detections).centroid.map((x, i) => x - data.truth.center[i])) > 1e-5);
});

test('hand-calculated diagonal random-matrix update checks geometry scaling and IW convention', () => {
  const prior = { mean: [0, 0, 0, 0], covariance: [[0.25, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], nu: 10, V: [[16, 0], [0, 4]] };
  const posterior = M.update(prior, [[2, 0], [-2, 0], [0, 1], [0, -1]], [[0.25, 0], [0, 0.25]]);
  assert.deepEqual(posterior.mean, [0, 0, 0, 0]);
  assert.equal(posterior.nu, 14);
  nearMatrix(posterior.V, [[41.6, 0], [0, 8]]);
  nearMatrix(M.extentMean(posterior), [[5.2, 0], [0, 1]]);
  near(posterior.covariance[0][0], 5 / 36);
  near(posterior.covariance[1][1], 0.1);
});

test('clouds with identical centroids teach different sizes and orientations', () => {
  const prior = M.initialState(); prior.mean = [0, 0, 0, 0];
  const R = scale(identity(2), 0.09);
  const horizontal = [[-3, 0], [3, 0], [0, -0.5], [0, 0.5]];
  const vertical = horizontal.map(([x, y]) => [-y, x]);
  const wider = horizontal.map(([x, y]) => [x * 1.5, y]);
  const a = M.update(prior, horizontal, R), b = M.update(prior, vertical, R), c = M.update(prior, wider, R);
  assert.deepEqual(a.mean, b.mean); assert.deepEqual(a.mean, c.mean);
  near(M.ellipseParameters(M.extentMean(a)).angle, 0);
  near(M.ellipseParameters(M.extentMean(b)).angle, 90);
  assert.ok(M.ellipseParameters(M.extentMean(c)).length > M.ellipseParameters(M.extentMean(a)).length);
});

test('matrix square roots and filter update are rotation-equivariant', () => {
  const t = 0.73, U = [[Math.cos(t), -Math.sin(t)], [Math.sin(t), Math.cos(t)]];
  const T = [[...U[0], 0, 0], [...U[1], 0, 0], [0, 0, ...U[0]], [0, 0, ...U[1]]];
  const X = [[5, 1.2], [1.2, 2]], root = sqrt2(X);
  nearMatrix(multiply(root, root), X);
  const prior = M.predict(M.initialState()), data = M.sampleScan({}, 3);
  const rotate = (A, x) => A.map(row => row.reduce((s, v, i) => s + v * x[i], 0));
  const rotatedPrior = { mean: rotate(T, prior.mean), covariance: sandwich(T, prior.covariance), nu: prior.nu, V: sandwich(U, prior.V) };
  const post = M.update(prior, data.detections, data.sensorCovariance);
  const rotated = M.update(rotatedPrior, data.detections.map(z => rotate(U, z)), sandwich(U, data.sensorCovariance));
  rotated.mean.forEach((v, i) => near(v, rotate(T, post.mean)[i]));
  nearMatrix(rotated.V, sandwich(U, post.V), 1e-8);
  nearMatrix(rotated.covariance, sandwich(T, post.covariance));
});

test('prediction propagates motion and forgets extent evidence without changing mean size', () => {
  const prior = M.initialState(), predicted = M.predict(prior, 2);
  near(predicted.mean[0], prior.mean[0] + 2 * prior.mean[2]);
  near(predicted.mean[1], prior.mean[1] + 2 * prior.mean[3]);
  assert.ok(predicted.nu < prior.nu && predicted.nu > 6);
  nearMatrix(M.extentMean(prior), M.extentMean(predicted));
  assert.ok(predicted.covariance[0][0] > prior.covariance[0][0]);
});

test('filter can be replayed with measurements alone; changing truth does not change its prior', () => {
  const run = M.simulate({ length: 14, width: 2.5, angle: 120, noise: 0.5 }, 15);
  let state = M.initialState();
  for (const frame of run.history) {
    state = M.update(M.predict(state), structuredClone(frame.detections), structuredClone(frame.sensorCovariance));
    assert.deepEqual(state, frame.posterior);
  }
  assert.deepEqual(M.simulate({}, 1).frame.prior, M.simulate({ length: 16, width: 1.5, angle: 170 }, 1).frame.prior);
});

test('repeated scans learn motion and extent across independent seeds', () => {
  const initial = [0, 0, 0, 0], final = [0, 0, 0, 0];
  for (let seed = 1; seed <= 20; seed++) {
    const run = M.simulate({ seed }, 40);
    for (const [i, key] of ['center', 'length', 'width', 'angle'].entries()) {
      initial[i] += run.history[0].errors[key]; final[i] += run.frame.errors[key];
    }
  }
  final.forEach((v, i) => assert.ok(v < initial[i], 'average error decreases: ' + i));
  assert.ok(final[0] / 20 < 0.7); assert.ok(final[1] / 20 < 0.7);
  assert.ok(final[2] / 20 < 0.25); assert.ok(final[3] / 20 < 3);
});

test('known sensor noise is not mistaken for physical width', () => {
  let correct = 0, ignored = 0;
  for (let seed = 1; seed <= 12; seed++) {
    let state = M.initialState(), wrong = M.initialState();
    for (let scan = 1; scan <= 40; scan++) {
      const data = M.sampleScan({ width: 3, noise: 1.2, seed, count: 30 }, scan);
      state = M.update(M.predict(state), data.detections, data.sensorCovariance);
      wrong = M.update(M.predict(wrong), data.detections, scale(identity(2), 1e-6));
    }
    correct += Math.abs(M.ellipseParameters(M.extentMean(state)).width - 3);
    ignored += Math.abs(M.ellipseParameters(M.extentMean(wrong)).width - 3);
  }
  assert.ok(correct < ignored * 0.4);
});

test('all control extremes keep kinematic and extent matrices positive definite', () => {
  for (const length of [6, 16]) for (const width of [1.5, 5.5]) for (const angle of [0, 90, 170]) for (const noise of [0.05, 1.2]) for (const count of [4, 40]) {
    const run = M.simulate({ length, width, angle, noise, count }, 40);
    for (const f of run.history) {
      positiveDefinite(f.posterior.covariance); positiveDefinite(f.posterior.V);
      assert.ok(f.posterior.mean.every(Number.isFinite));
      assert.ok(f.posterior.nu > 6);
    }
  }
});

test('empty scans retain predictions, inputs are not mutated, and replay is deterministic', () => {
  const prior = M.predict(M.initialState()), before = structuredClone(prior);
  assert.deepEqual(M.update(prior, [], identity(2)), prior);
  M.update(prior, [[1, 2], [2, 3]], identity(2));
  assert.deepEqual(prior, before);
  assert.deepEqual(M.simulate({}, 3), M.simulate({}, 3));
  assert.throws(() => M.simulate({ width: -1 }), RangeError);
  assert.throws(() => M.simulate({}, 0), RangeError);
});
