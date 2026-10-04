/* Joint Bayesian center/extent inference for a finite 1D extended object.
 * z_i = c + u_i + v_i, u_i ~ Uniform[-L/2,L/2], v_i ~ N(0,sigma^2).
 * Independent implementation of the spatial-source convolution principle in
 * Granstrom, Baum & Reuter, arXiv:1604.00970, Eq. (6), specialized to a segment.
 * Midpoint quadrature in (c, log L), retaining center/size dependence.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.TwoReturnModel = api;
})(typeof globalThis === 'object' ? globalThis : this, function () {
  'use strict';
  const defaults = Object.freeze({ priorMean: 0, priorSigma: 1.2, extentMedian: 4, trueCenter: 0, trueLength: 6, noise: 0.3, side: 1 });
  const extentPrior = Object.freeze({ median: 4, logSigma: 0.5 });
  const fractions = Object.freeze([0.25, 0.85]);
  const gaussian = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
  // Stable positive-tail erfc approximation (relative error about 1e-7).
  function erfcPositive(x) {
    const t = 1 / (1 + x / 2);
    return t * Math.exp(-x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  }
  function normalInterval(lo, hi) {
    const s = Math.SQRT2;
    if (lo >= 0) return Math.max(0, 0.5 * (erfcPositive(lo / s) - erfcPositive(hi / s)));
    if (hi <= 0) return Math.max(0, 0.5 * (erfcPositive(-hi / s) - erfcPositive(-lo / s)));
    return Math.max(0, 1 - 0.5 * (erfcPositive(hi / s) + erfcPositive(-lo / s)));
  }
  function likelihood(z, center, length, noise) {
    if (!(length >= 0) || !(noise >= 0) || ![z, center, length, noise].every(Number.isFinite)) throw new RangeError('Invalid observation model.');
    if (noise === 0) {
      if (length === 0) throw new RangeError('A noiseless point has no ordinary density.');
      return Math.abs(z - center) < length / 2 ? 1 / length : 0;
    }
    if (length / noise < 1e-5) return gaussian(z, center, noise);
    return normalInterval((z - center - length / 2) / noise, (z - center + length / 2) / noise) / length;
  }
  function quantile(axis, masses, step, probability, logarithmic = false) {
    let sum = 0;
    for (let i = 0; i < masses.length; i++) {
      if (sum + masses[i] >= probability && masses[i] > 0) {
        const value = axis[i] - step / 2 + step * (probability - sum) / masses[i];
        return logarithmic ? Math.exp(value) : value;
      }
      sum += masses[i];
    }
    return logarithmic ? Math.exp(axis.at(-1) + step / 2) : axis.at(-1) + step / 2;
  }
  // The estimator has no true center, true length, side, or source-location inputs.
  function infer({ detections, noise, priorMean = defaults.priorMean, priorSigma = defaults.priorSigma,
    extentMedian = extentPrior.median, extentLogSigma = extentPrior.logSigma }, { resolution = 1 } = {}) {
    if (!Array.isArray(detections) || detections.length !== 2 || !detections.every(Number.isFinite) ||
        ![noise, priorMean, priorSigma, extentMedian, extentLogSigma, resolution].every(Number.isFinite) ||
        noise <= 0 || priorSigma <= 0 || extentMedian <= 0 || extentLogSigma <= 0 || resolution <= 0) throw new RangeError('Invalid inference inputs.');
    const logMedian = Math.log(extentMedian);
    // Retain a long-size tail even when a confident center prior strongly
    // disagrees with the returns: those observations may favor a much longer body.
    const requiredLength = 2 * Math.max(...detections.map(z => Math.abs(z - priorMean))) + 4 * noise + 16 * priorSigma;
    const qlo = logMedian - 9 * extentLogSigma;
    const qhi = Math.max(logMedian + 9 * extentLogSigma, Math.log(requiredLength) + 3 * extentLogSigma);
    const nl = Math.ceil((qhi - qlo) / 0.0175 * resolution), dq = (qhi - qlo) / nl;
    const q = Array.from({ length: nl }, (_, j) => qlo + (j + 0.5) * dq), lengths = q.map(Math.exp);
    const extentReach = extentMedian * Math.exp(3.3 * extentLogSigma) / 2 + 5 * noise;
    const xlo = Math.min(priorMean - 8 * priorSigma, Math.min(...detections) - extentReach);
    const xhi = Math.max(priorMean + 8 * priorSigma, Math.max(...detections) + extentReach);
    const desiredDx = Math.min(0.055, noise / 2.5, priorSigma / 18) / resolution;
    const nx = Math.ceil((xhi - xlo) / desiredDx), dx = (xhi - xlo) / nx;
    const centers = Array.from({ length: nx }, (_, i) => xlo + (i + 0.5) * dx);
    const pc = centers.map(x => gaussian(x, priorMean, priorSigma));
    const pq = q.map(v => gaussian(v, logMedian, extentLogSigma));
    const joint = new Float64Array(nx * nl);
    const centerMass = new Float64Array(nx), lengthMass = new Float64Array(nl);
    const centerLikelihood = new Float64Array(nx), lengthLikelihood = new Float64Array(nl);
    let evidence = 0, meanProduct = 0;
    for (let j = 0; j < nl; j++) {
      for (let i = 0; i < nx; i++) {
        const ell = likelihood(detections[0], centers[i], lengths[j], noise) * likelihood(detections[1], centers[i], lengths[j], noise);
        const mass = ell * pc[i] * pq[j] * dx * dq;
        joint[j * nx + i] = mass;
        centerMass[i] += mass; lengthMass[j] += mass; evidence += mass;
        meanProduct += centers[i] * lengths[j] * mass;
        centerLikelihood[i] += ell * pq[j] * dq;
        lengthLikelihood[j] += ell * pc[i] * dx;
      }
    }
    if (!(evidence > 0) || !Number.isFinite(evidence)) throw new RangeError('The posterior could not be normalized.');
    for (let k = 0; k < joint.length; k++) joint[k] /= evidence;
    for (let i = 0; i < nx; i++) centerMass[i] /= evidence;
    for (let j = 0; j < nl; j++) lengthMass[j] /= evidence;
    const mean = (axis, mass) => axis.reduce((sum, value, i) => sum + value * mass[i], 0);
    const variance = (axis, mass, m) => axis.reduce((sum, value, i) => sum + (value - m) ** 2 * mass[i], 0);
    const centerMean = mean(centers, centerMass), lengthMean = mean(lengths, lengthMass);
    const centerVariance = variance(centers, centerMass, centerMean), lengthVariance = variance(lengths, lengthMass, lengthMean);
    const centerInterval = [0.025, 0.975].map(p => quantile(centers, centerMass, dx, p));
    const lengthInterval = [0.025, 0.975].map(p => quantile(q, lengthMass, dq, p, true));
    return {
      observations: [...detections], noise, priorMean, priorSigma, extentMedian, extentLogSigma,
      centers, lengths, q, dx, dq, nx, nl, xlo, xhi, qlo, qhi, joint, evidence,
      center: { prior: pc, likelihood: centerLikelihood, posterior: Array.from(centerMass, p => p / dx), mass: centerMass, mean: centerMean, sigma: Math.sqrt(centerVariance), interval: centerInterval },
      extent: { prior: pq.map((p, j) => p / lengths[j]), likelihood: lengthLikelihood, posterior: Array.from(lengthMass, (p, j) => p / (dq * lengths[j])), mass: lengthMass, mean: lengthMean, sigma: Math.sqrt(lengthVariance), interval: lengthInterval },
      correlation: (meanProduct / evidence - centerMean * lengthMean) / Math.sqrt(centerVariance * lengthVariance),
      edgeMass: centerMass[0] + centerMass.at(-1) + lengthMass[0] + lengthMass.at(-1)
    };
  }
  function example(parameters = {}, options) {
    const p = { ...defaults, ...parameters };
    if (![p.priorMean, p.priorSigma, p.trueCenter, p.trueLength, p.noise, p.side].every(Number.isFinite) ||
        p.trueLength <= 0 || ![-1, 1].includes(p.side)) throw new RangeError('Invalid example settings.');
    // Deliberately chosen observations, not random draws or known correspondences.
    // Noise sets the likelihood uncertainty; moving it does not resample the pair.
    const detections = fractions.map(f => p.trueCenter + p.side * f * p.trueLength / 2);
    const posterior = infer({ detections, noise: p.noise, priorMean: p.priorMean, priorSigma: p.priorSigma, extentMedian: p.extentMedian }, options);
    return { settings: p, detections, posterior };
  }
  return Object.freeze({ defaults, extentPrior, fractions, gaussian, normalInterval, likelihood, infer, example });
});
