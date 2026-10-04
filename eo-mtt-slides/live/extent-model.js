/* Factorized Gaussian / inverse-Wishart random-matrix extended-object filter.
 * Independent implementation of Tables IV and IX in Granstrom, Baum & Reuter,
 * arXiv:1604.00970 (Feldmann et al. update and forgetting prediction).
 * X is the geometric ellipse matrix: source covariance = X/4.
 * IW convention: E[X] = V/(nu - 2*d - 2), d = 2.
 * Gaussian approximation to uniform elliptical sources; symmetric SPD roots.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.ExtendedTargetModel = api;
})(typeof globalThis === 'object' ? globalThis : this, function () {
  'use strict';
  const defaults = Object.freeze({ length: 10, width: 4, angle: 35, noise: 0.35, count: 20, seed: 17 });
  const MAX_SCANS = 40;
  const ZETA = 0.25;
  const matrix = (n, m = n) => Array.from({ length: n }, () => Array(m).fill(0));
  const identity = n => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => +(i === j)));
  const transpose = A => A[0].map((_, j) => A.map(row => row[j]));
  const add = (A, B) => A.map((row, i) => row.map((x, j) => x + B[i][j]));
  const scale = (A, s) => A.map(row => row.map(x => x * s));
  const multiply = (A, B) => A.map(row => B[0].map((_, j) => row.reduce((sum, x, k) => sum + x * B[k][j], 0)));
  const mv = (A, x) => A.map(row => row.reduce((sum, a, i) => sum + a * x[i], 0));
  const outer = x => x.map(a => x.map(b => a * b));
  const sym = A => scale(add(A, transpose(A)), 0.5);
  const positionCov = P => P.slice(0, 2).map(row => row.slice(0, 2));

  function inverse2(A) {
    const determinant = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    if (!(determinant > 0) || !Number.isFinite(determinant)) throw new RangeError('Matrix must be positive definite.');
    return scale([[A[1][1], -A[0][1]], [-A[1][0], A[0][0]]], 1 / determinant);
  }
  function sqrt2(A) {
    const det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    if (!(det > 0) || A[0][0] <= 0) throw new RangeError('Matrix must be positive definite.');
    const s = Math.sqrt(det), t = Math.sqrt(A[0][0] + A[1][1] + 2 * s);
    return scale(add(A, scale(identity(2), s)), 1 / t);
  }
  const sandwich = (A, B) => sym(multiply(multiply(A, B), transpose(A)));
  const extentMean = state => scale(state.V, 1 / (state.nu - 6));
  const clone = s => ({ mean: [...s.mean], covariance: s.covariance.map(r => [...r]), nu: s.nu, V: s.V.map(r => [...r]) });

  function ellipseMatrix(length, width, angleDegrees) {
    const a = length / 2, b = width / 2, t = angleDegrees * Math.PI / 180;
    const rotation = [[Math.cos(t), -Math.sin(t)], [Math.sin(t), Math.cos(t)]];
    return sandwich(rotation, [[a * a, 0], [0, b * b]]);
  }
  function ellipseParameters(X) {
    const a = X[0][0], b = (X[0][1] + X[1][0]) / 2, d = X[1][1];
    const gap = Math.hypot(a - d, 2 * b), major = (a + d + gap) / 2, minor = (a + d - gap) / 2;
    if (!(minor > 0)) throw new RangeError('Extent must be positive definite.');
    const angle = ((Math.atan2(2 * b, a - d) * 90 / Math.PI) % 180 + 180) % 180;
    return { length: 2 * Math.sqrt(major), width: 2 * Math.sqrt(minor), angle, identifiable: major / minor > 1.15 };
  }

  // Fixed across all truth-size, truth-angle, and noise settings.
  function initialState() {
    return { mean: [-17.5, -2.5, 0.6, 0.15], covariance: [[4, 0, 0, 0], [0, 4, 0, 0], [0, 0, 0.4, 0], [0, 0, 0, 0.4]], nu: 10, V: scale(ellipseMatrix(7, 7, 0), 4) };
  }
  function predict(state, dt = 1, { accelerationSigma = 0.08, forgettingTime = 20 } = {}) {
    if (!(dt > 0) || !(forgettingTime > 0) || accelerationSigma < 0) throw new RangeError('Invalid dynamics.');
    const F = [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]];
    const G = [[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]];
    const Q = scale(multiply(G, transpose(G)), accelerationSigma ** 2);
    const decay = Math.exp(-dt / forgettingTime);
    return { mean: mv(F, state.mean), covariance: add(sandwich(F, state.covariance), Q), nu: 6 + decay * (state.nu - 6), V: scale(state.V, decay) };
  }
  function measurementStatistics(detections) {
    if (!detections.length) return { count: 0, centroid: null, scatter: matrix(2) };
    if (!detections.every(z => z.length === 2 && z.every(Number.isFinite))) throw new RangeError('Detections must be finite 2D positions.');
    const centroid = [0, 1].map(j => detections.reduce((s, z) => s + z[j], 0) / detections.length);
    const scatter = detections.reduce((S, z) => add(S, outer(z.map((v, j) => v - centroid[j]))), matrix(2));
    return { count: detections.length, centroid, scatter };
  }

  // Truth-free estimator: only prior, observed detections, and known sensor R.
  function update(prior, detections, sensorCovariance) {
    const stats = measurementStatistics(detections);
    if (!stats.count) return clone(prior);
    inverse2(sensorCovariance);
    const X = extentMean(prior), Y = add(scale(X, ZETA), sensorCovariance);
    const centroidNoise = scale(Y, 1 / stats.count);
    const S = add(positionCov(prior.covariance), centroidNoise);
    const residual = stats.centroid.map((z, j) => z - prior.mean[j]);
    const cross = prior.covariance.map(row => row.slice(0, 2));
    const K = multiply(cross, inverse2(S));
    const correction = mv(K, residual);
    const mean = prior.mean.map((x, i) => x + correction[i]);
    const A = identity(4);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 2; j++) A[i][j] -= K[i][j];
    const covariance = add(sandwich(A, prior.covariance), sandwich(K, centroidNoise));
    const rootX = sqrt2(X);
    const centroidTransform = multiply(rootX, inverse2(sqrt2(S)));
    const scatterTransform = multiply(rootX, inverse2(sqrt2(Y)));
    const Nhat = sandwich(centroidTransform, outer(residual));
    const Zhat = sandwich(scatterTransform, stats.scatter);
    return { mean, covariance, nu: prior.nu + stats.count, V: add(add(prior.V, Nhat), Zhat) };
  }

  function randomGenerator(seed) {
    let a = seed | 0;
    return () => {
      a = a + 0x6D2B79F5 | 0;
      let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function settings(parameters = {}) {
    const p = { ...defaults, ...parameters };
    if (!Object.values(p).every(Number.isFinite) || p.length <= 0 || p.width <= 0 || p.noise <= 0 || !Number.isInteger(p.count) || p.count < 0 || !Number.isInteger(p.seed)) throw new RangeError('Invalid simulation parameters.');
    return p;
  }
  function sampleScan(parameters, scan) {
    const p = settings(parameters);
    const random = randomGenerator((p.seed ^ Math.imul(scan + 1, 0x9e3779b9)) >>> 0);
    const normalPair = () => {
      const r = Math.sqrt(-2 * Math.log(Math.max(1e-12, random()))), t = 2 * Math.PI * random();
      return [r * Math.cos(t), r * Math.sin(t)];
    };
    const center = [-16 + 0.7 * (scan - 1), -4 + 0.2 * (scan - 1)];
    const angle = p.angle * Math.PI / 180, c = Math.cos(angle), s = Math.sin(angle);
    const sources = [], detections = [];
    for (let i = 0; i < p.count; i++) {
      // sqrt(U) is essential for a uniform AREA density.
      const r = Math.sqrt(random()), t = 2 * Math.PI * random();
      const u = p.length / 2 * r * Math.cos(t), v = p.width / 2 * r * Math.sin(t);
      const source = [center[0] + c * u - s * v, center[1] + s * u + c * v];
      const noise = normalPair();
      sources.push(source);
      detections.push(source.map((x, j) => x + p.noise * noise[j]));
    }
    return { scan, truth: { center, velocity: [0.7, 0.2], length: p.length, width: p.width, angle: p.angle, X: ellipseMatrix(p.length, p.width, p.angle) }, sources, detections, sensorCovariance: scale(identity(2), p.noise ** 2) };
  }
  function errorMetrics(state, truth) {
    const estimate = ellipseParameters(extentMean(state));
    const delta = Math.abs(estimate.angle - ((truth.angle % 180 + 180) % 180));
    return { center: Math.hypot(state.mean[0] - truth.center[0], state.mean[1] - truth.center[1]), length: Math.abs(estimate.length - truth.length), width: Math.abs(estimate.width - truth.width), angle: Math.min(delta, 180 - delta) };
  }
  function simulate(parameters = {}, scans = 1) {
    const p = settings(parameters);
    if (!Number.isInteger(scans) || scans < 1 || scans > MAX_SCANS) throw new RangeError('Scan must be between 1 and 40.');
    let state = initialState();
    const history = [];
    for (let k = 1; k <= scans; k++) {
      const prior = predict(state), data = sampleScan(p, k);
      state = update(prior, data.detections, data.sensorCovariance);
      history.push({ ...data, prior, posterior: state, stats: measurementStatistics(data.detections), estimate: ellipseParameters(extentMean(state)), errors: errorMetrics(state, data.truth) });
    }
    return { settings: p, history, frame: history.at(-1) };
  }
  return Object.freeze({ defaults, MAX_SCANS, ZETA, initialState, predict, update, extentMean, ellipseMatrix, ellipseParameters, measurementStatistics, sampleScan, simulate, errorMetrics, math: { identity, add, scale, multiply, transpose, inverse2, sqrt2, sandwich, positionCov } });
});
