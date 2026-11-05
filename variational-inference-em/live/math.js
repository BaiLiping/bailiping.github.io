/* Small, dependency-free kernels shared by the browser and Node tests. */
(function (root) {
  'use strict';
  const LOG2PI = Math.log(2 * Math.PI);
  function logsumexp(a) {
    const m = Math.max(...a);
    if (m === -Infinity) return m;
    return m + Math.log(a.reduce((s, v) => s + Math.exp(v - m), 0));
  }
  function logNormal(x, mean, variance) {
    if (!(variance > 0)) throw new Error('Variance must be positive');
    return -.5 * (LOG2PI + Math.log(variance) + (x - mean) ** 2 / variance);
  }
  function gaussianKL(rho, mean, variance) {
    if (!(Math.abs(rho) < 1) || variance.some(v => !(v > 0))) throw new Error('Invalid Gaussian covariance');
    const d = 1 - rho * rho;
    return .5 * ((variance[0] + variance[1] + mean[0] ** 2 + mean[1] ** 2 - 2 * rho * mean[0] * mean[1]) / d - 2 + Math.log(d / (variance[0] * variance[1])));
  }
  function cavi(rho, mean, variance, coordinate) {
    if (coordinate !== 0 && coordinate !== 1) throw new Error('Coordinate must be 0 or 1');
    gaussianKL(rho, mean, variance);
    const m = mean.slice(), v = variance.slice();
    m[coordinate] = rho * m[1 - coordinate];
    v[coordinate] = 1 - rho * rho;
    return { mean: m, variance: v };
  }
  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a += 0x6D2B79F5;
      let t = a;
      t = Math.imul(t ^ t >>> 15, t | 1);
      t ^= t + Math.imul(t ^ t >>> 7, t | 61);
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function sampleData(seed = 7, separation = 3.5, n = 120) {
    if (!(n > 0 && Number.isInteger(n) && separation >= 0)) throw new Error('Invalid data configuration');
    const random = rng(seed), x = [];
    for (let i = 0; i < n; i++) {
      const component = random() < .4 ? -1 : 1;
      const normal = Math.sqrt(-2 * Math.log(Math.max(1e-15, random()))) * Math.cos(2 * Math.PI * random());
      x.push(component * separation / 2 + .8 * normal);
    }
    return x.sort((a, b) => a - b);
  }
  function initialModel(kind = 'spread', separation = 3.5) {
    const mean = kind === 'identical' ? [0, 0] : kind === 'poor' ? [-separation / 2 - .6, -separation / 2 + .2] : [-.5, .5];
    return { weight: [.5, .5], mean, variance: [.64, .64] };
  }
  function logScores(x, model) {
    return model.weight.map((p, k) => Math.log(p) + logNormal(x, model.mean[k], model.variance[k]));
  }
  function expectation(data, model) {
    return data.map(x => {
      const a = logScores(x, model), norm = logsumexp(a);
      return a.map(v => Math.exp(v - norm));
    });
  }
  // Exact maximizer of Q with fixed variances, or with variance >= floor.
  // Empty components keep their old mean/variance and receive zero weight.
  function maximization(data, responsibilities, model, learnVariance = false, floor = .09) {
    if (!data.length || responsibilities.length !== data.length || !(floor > 0)) throw new Error('Invalid M-step input');
    const kCount = model.weight.length, n = data.length;
    const next = { weight: [], mean: [], variance: [] };
    for (let k = 0; k < kCount; k++) {
      const count = responsibilities.reduce((s, row) => s + row[k], 0);
      next.weight[k] = count / n;
      if (count === 0) {
        next.mean[k] = model.mean[k];
        next.variance[k] = model.variance[k];
        continue;
      }
      const mean = data.reduce((s, x, i) => s + responsibilities[i][k] * x, 0) / count;
      next.mean[k] = mean;
      next.variance[k] = learnVariance ? Math.max(floor, data.reduce((s, x, i) => s + responsibilities[i][k] * (x - mean) ** 2, 0) / count) : model.variance[k];
    }
    return next;
  }
  function metrics(data, model, responsibilities) {
    let likelihood = 0, elbo = 0, gap = 0;
    data.forEach((x, i) => {
      const a = logScores(x, model), norm = logsumexp(a);
      likelihood += norm;
      responsibilities[i].forEach((r, k) => {
        if (r > 0) {
          elbo += r * (a[k] - Math.log(r));
          gap += r * (Math.log(r) - a[k] + norm);
        }
      });
    });
    return { likelihood, elbo, gap };
  }
  const api = { logsumexp, logNormal, gaussianKL, cavi, rng, sampleData, initialModel, expectation, maximization, metrics };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.VIMath = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
