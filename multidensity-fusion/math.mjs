/** Small, deterministic numerical kernels used by the teaching labs.
 * Variances, not standard deviations, are used unless a name says otherwise.
 * A null density means that the requested normalized pool does not exist.
 */
export function positive(x, name = 'value') {
  if (!Number.isFinite(x) || x <= 0) throw new RangeError(`${name} must be positive and finite`);
  return x;
}
export function weight(w) {
  if (!Number.isFinite(w) || w < 0 || w > 1) throw new RangeError('weight must lie in [0,1]');
  return w;
}
export function logNormal(x, mean, variance) {
  positive(variance, 'variance');
  return -.5 * (Math.log(2 * Math.PI * variance) + (x - mean) ** 2 / variance);
}
export const normal = (x, mean, variance) => Math.exp(logNormal(x, mean, variance));
export function gaussianPool(m1, p1, m2, p2, w = .5) {
  positive(p1); positive(p2); weight(w);
  const variance = 1 / (w / p1 + (1 - w) / p2);
  return { mean: variance * (w * m1 / p1 + (1 - w) * m2 / p2), variance };
}
export function mixtureMoments(m1, p1, m2, p2, w = .5) {
  positive(p1); positive(p2); weight(w);
  const mean = w * m1 + (1 - w) * m2;
  return { mean, variance: w * (p1 + (m1 - mean) ** 2) + (1 - w) * (p2 + (m2 - mean) ** 2) };
}
/** Independent Gaussian likelihoods with one shared N(0, priorVariance) prior. */
export function sharedPrior(priorVariance, sensors, measurement, noiseVariance = 1) {
  positive(priorVariance); positive(noiseVariance);
  if (!Number.isInteger(sensors) || sensors < 1) throw new RangeError('sensors must be a positive integer');
  const j0 = 1 / priorVariance, jm = 1 / noiseVariance;
  const correctVariance = 1 / (j0 + sensors * jm);
  const localVariance = 1 / (j0 + jm);
  return {
    correct: { mean: correctVariance * sensors * jm * measurement, variance: correctVariance },
    product: { mean: localVariance * jm * measurement, variance: localVariance / sensors },
    gci: { mean: localVariance * jm * measurement, variance: localVariance }
  };
}
/** Scalar BLUE and CI, with a valid joint error covariance (|rho| < 1). */
export function correlation(p1, p2, rho, w = .5) {
  positive(p1); positive(p2); weight(w);
  if (!Number.isFinite(rho) || Math.abs(rho) >= 1) throw new RangeError('|rho| must be less than 1');
  const cross = rho * Math.sqrt(p1 * p2);
  const actual = a => a * a * p1 + (1 - a) ** 2 * p2 + 2 * a * (1 - a) * cross;
  const alpha = p2 / (p1 + p2);
  const oracleAlpha = (p2 - cross) / (p1 + p2 - 2 * cross);
  const bound = gaussianPool(0, p1, 0, p2, w).variance;
  const ciAlpha = bound * w / p1;
  return { naiveReported: p1 * p2 / (p1 + p2), naiveActual: actual(alpha),
    oracle: actual(oracleAlpha), ciReported: bound, ciActual: actual(ciAlpha), alpha, ciAlpha, oracleAlpha };
}
/** A symmetric 2x2 matrix is stored as [a,b,d] for [[a,b],[b,d]]. */
export const det2 = ([a, b, d]) => a * d - b * b;
export function inverse2(p) {
  const determinant = det2(p);
  if (p.length !== 3 || !p.every(Number.isFinite) || p[0] <= 0 || determinant <= 0)
    throw new RangeError('matrix must be symmetric positive definite');
  return [p[2] / determinant, -p[1] / determinant, p[0] / determinant];
}
export function rotateCov(major, minor, angle) {
  positive(major); positive(minor);
  const c = Math.cos(angle), s = Math.sin(angle);
  return [major * c * c + minor * s * s, (major - minor) * c * s, major * s * s + minor * c * c];
}
export function ci2(p1, p2, w) {
  weight(w); const a = inverse2(p1), b = inverse2(p2);
  return inverse2(a.map((v, i) => w * v + (1 - w) * b[i]));
}
export function optimalCI(p1, p2, steps = 1000) {
  if (!Number.isInteger(steps) || steps < 1) throw new RangeError('steps must be positive');
  let best = { weight: 0, covariance: ci2(p1, p2, 0), objective: Infinity };
  for (let i = 0; i <= steps; i++) {
    const covariance = ci2(p1, p2, i / steps), objective = Math.log(det2(covariance));
    if (objective < best.objective - 1e-12) best = { weight: i / steps, covariance, objective };
  }
  return best;
}
/** Trapezoidal quadrature on an explicitly finite, uniform grid. */
export function integrate(values, dx) {
  positive(dx, 'grid spacing');
  if (values.length < 2) throw new RangeError('at least two grid points are required');
  return dx * values.reduce((sum, v, i) => sum + v * (i === 0 || i === values.length - 1 ? .5 : 1), 0);
}
export function normalizeLogs(logs, dx) {
  if (logs.length < 2 || logs.some(v => Number.isNaN(v) || v === Infinity)) throw new RangeError('invalid log density');
  const max = Math.max(...logs);
  if (max === -Infinity) return { density: null, logZ: -Infinity };
  const scaled = logs.map(v => Math.exp(v - max)), total = integrate(scaled, dx);
  return { density: scaled.map(v => v / total), logZ: max + Math.log(total) };
}
export function pools(log1, log2, dx, w) {
  weight(w);
  if (log1.length !== log2.length) throw new RangeError('grids must match');
  const a = normalizeLogs(log1, dx), b = normalizeLogs(log2, dx);
  if (!a.density || !b.density) throw new RangeError('each input must have positive mass on the grid');
  // Subtract log normalizers, not log(exp(log p)), to retain tiny tails.
  const la = log1.map(v => v - a.logZ), lb = log2.map(v => v - b.logZ);
  const combined = la.map((v, i) => w === 0 ? lb[i] : w === 1 ? v : w * v + (1 - w) * lb[i]);
  const geo = normalizeLogs(combined, dx), product = normalizeLogs(la.map((v, i) => v + lb[i]), dx);
  return { p1: a.density, p2: b.density, aa: a.density.map((v, i) => w * v + (1 - w) * b.density[i]),
    gci: geo.density, product: product.density, logZ: geo.logZ };
}
/** Bernoulli RFS fusion; eta is the spatial overlap integral, not a likelihood. */
export function bernoulli(r1, r2, w, eta) {
  weight(r1); weight(r2); weight(w);
  if (!Number.isFinite(eta) || eta < 0 || eta > 1 + 1e-10) throw new RangeError('overlap must be in [0,1]');
  if (w === 0) return { aa: r2, gci: r2 };
  if (w === 1) return { aa: r1, gci: r1 };
  const logA = w * Math.log1p(-r1) + (1 - w) * Math.log1p(-r2);
  const logB = w * Math.log(r1) + (1 - w) * Math.log(r2) + Math.log(eta);
  const aa = w * r1 + (1 - w) * r2;
  if (logA === -Infinity && logB === -Infinity) return { aa, gci: null };
  if (logB === -Infinity) return { aa, gci: 0 };
  if (logA === -Infinity) return { aa, gci: 1 };
  return { aa, gci: 1 / (1 + Math.exp(logA - logB)) };
}
export function equalGaussianOverlap(separation, variance, w) {
  positive(variance); weight(w);
  return Math.exp(-w * (1 - w) * separation ** 2 / (2 * variance));
}
