/* A scalar center update conditional on known Gaussian extent and association.
 * z_i = x + e_i + v_i, e_i ~ N(0, sigma_e^2), v_i ~ N(0, sigma_r^2).
 * All returns are conditionally independent. The centroid has variance
 * (sigma_e^2 + sigma_r^2) / n. Extent is supplied, not estimated here.
 * Spatial model: Granstrom, Baum & Reuter, arXiv:1604.00970, Eq. (14).
 */
(function (root, factory) {
  const model = factory();
  if (typeof module === 'object' && module.exports) module.exports = model;
  else root.ExtendedTargetModel = model;
})(typeof globalThis === 'object' ? globalThis : this, function () {
  'use strict';

  const defaults = Object.freeze({ priorMean: -1.2, priorSigma: 1.35, centroid: 2.1, noiseSigma: 0.75, extentSigma: 1.5, count: 8 });

  function posterior(parameters = {}) {
    const p = { ...defaults, ...parameters };
    if (!Object.values(p).every(Number.isFinite) || p.priorSigma <= 0 || p.noiseSigma <= 0 || p.extentSigma < 0 || !Number.isInteger(p.count) || p.count < 1) {
      throw new RangeError('Use positive prior/noise deviations, nonnegative extent, and a positive integer return count.');
    }
    const priorVar = p.priorSigma ** 2;
    const returnVar = p.extentSigma ** 2 + p.noiseSigma ** 2;
    const centroidVar = returnVar / p.count;
    const gain = priorVar / (priorVar + centroidVar);
    const postMean = p.priorMean + gain * (p.centroid - p.priorMean);
    const postVar = priorVar * centroidVar / (priorVar + centroidVar);
    return { ...p, priorVar, returnVar, centroidVar, gain, postMean, postVar, postSigma: Math.sqrt(postVar) };
  }

  // Fixed, symmetric display points hold the centroid still as spread changes.
  // They illustrate one cloud; they are not a Monte Carlo validation dataset.
  function displayReturns(state) {
    const offsets = [-1.534, -0.887, -0.489, -0.157, 0.157, 0.489, 0.887, 1.534];
    const scale = Math.sqrt(offsets.reduce((sum, x) => sum + x * x, 0) / offsets.length);
    return offsets.map(x => state.centroid + x / scale * Math.sqrt(state.returnVar));
  }

  function gaussian(x, mean, variance) {
    return Math.exp(-0.5 * (x - mean) ** 2 / variance) / Math.sqrt(2 * Math.PI * variance);
  }

  return Object.freeze({ defaults, posterior, displayReturns, gaussian });
});
