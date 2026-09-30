/* Pure numerical core. No DOM, network, or third-party dependencies. */
(function (root) {
  'use strict';
  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function normalGenerator(seed) {
    const uniform = rng(seed); let spare = null;
    return function () {
      if (spare !== null) { const z = spare; spare = null; return z; }
      const r = Math.sqrt(-2 * Math.log(1 - uniform()));
      const t = 2 * Math.PI * uniform();
      spare = r * Math.sin(t); return r * Math.cos(t);
    };
  }
  function gaussianTrial(n, sigma, mu, normal) {
    if (!Number.isInteger(n) || n < 1 || !(sigma > 0)) throw new Error('Invalid Gaussian model');
    const data = Array.from({ length: n }, () => mu + sigma * normal());
    return { data, mean: data.reduce((a, b) => a + b, 0) / n, first: data[0] };
  }
  function statistics(xs, truth) {
    if (!xs.length) return { count: 0, mean: NaN, variance: NaN, bias: NaN, mse: NaN };
    const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
    const ss = xs.reduce((a, x) => a + (x - mean) ** 2, 0);
    return { count: xs.length, mean, variance: xs.length > 1 ? ss / (xs.length - 1) : NaN,
      bias: mean - truth, mse: xs.reduce((a, x) => a + (x - truth) ** 2, 0) / xs.length };
  }
  function shrinkage(alpha, mu, sigma, n, anchor = 0) {
    const v = sigma * sigma / n, bias = (1 - alpha) * (anchor - mu);
    return { expectation: alpha * mu + (1 - alpha) * anchor, bias,
      variance: alpha * alpha * v, mse: alpha * alpha * v + bias * bias,
      unbiasedCRB: v, biasedVarianceBound: alpha * alpha * v };
  }
  function eigen2(a, b, c) {
    const d = Math.hypot(a - c, 2 * b), hi = Math.max(0, (a + c + d) / 2);
    const det = a * c - b * b;
    const lo = hi > 0 ? Math.max(0, det / hi) : 0;
    return { hi, lo, angle: 0.5 * Math.atan2(2 * b, a - c) };
  }
  function rangeFisher(anchors, target, sigma, unknownBias = false) {
    if (!(sigma > 0) || anchors.length < 1) throw new Error('Invalid range model');
    let a = 0, b = 0, c = 0, ux = 0, uy = 0;
    for (const p of anchors) {
      const dx = target[0] - p[0], dy = target[1] - p[1], r = Math.hypot(dx, dy);
      if (r < 0.08) return { invalid: true, reason: 'Target too close to an anchor: range derivative undefined at coincidence.' };
      const x = dx / r, y = dy / r; a += x*x; b += x*y; c += y*y; ux += x; uy += y;
    }
    if (unknownBias) { a -= ux * ux / anchors.length; b -= ux * uy / anchors.length; c -= uy * uy / anchors.length; }
    const w = 1 / (sigma * sigma); a *= w; b *= w; c *= w;
    const e = eigen2(a,b,c), tol = Math.max(1e-12, e.hi * 1e-10);
    const rank = (e.hi > tol ? 1 : 0) + (e.lo > tol ? 1 : 0);
    const det = a * c - b * b;
    return { invalid: false, a, b, c, ...e, rank,
      covariance: rank === 2 ? [c / det, -b / det, a / det] : null,
      peb: rank === 2 ? Math.sqrt(1 / e.hi + 1 / e.lo) : Infinity,
      condition: rank === 2 ? e.hi / e.lo : Infinity };
  }
  const api = { rng, normalGenerator, gaussianTrial, statistics, shrinkage, eigen2, rangeFisher };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.CRBMath = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
