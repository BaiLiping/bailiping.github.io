/* Fixed-scale SVG views for comparing Bayes updates as true extent changes. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.TwoReturnView = factory();
})(typeof globalThis === 'object' ? globalThis : this, function () {
  'use strict';
  const C = { prior: '#496e87', likelihood: '#b16f35', posterior: '#2f6b4f', truth: '#75638f', ink: '#203129', muted: '#66756e', line: '#e3e7df' };
  const fmt = (x, n = 2) => Number(x.toFixed(n)).toString();
  const start = (w, h, title) => '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ' + w + ' ' + h + '" role="img" aria-label="' + title + '"><rect width="100%" height="100%" rx="8" fill="#fffefb"/><g font-family="Inter,Arial,sans-serif" font-size="10" fill="' + C.ink + '">';
  const end = '</g></svg>';
  const line = (x1, y1, x2, y2, color, dash = '', width = 1) => '<path d="M' + fmt(x1) + ',' + fmt(y1) + 'L' + fmt(x2) + ',' + fmt(y2) + '" fill="none" stroke="' + color + '" stroke-width="' + width + '" stroke-dasharray="' + dash + '"/>';
  const ranges = Object.freeze({ center: Object.freeze([-5, 5, 0.7]), extent: Object.freeze([0, 16, 0.4]) });
  function interpolate(axis, values, value) {
    if (value <= axis[0]) return values[0];
    if (value >= axis.at(-1)) return values.at(-1);
    let lo = 0, hi = axis.length - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (axis[mid] > value) hi = mid; else lo = mid; }
    return values[lo] + (values[hi] - values[lo]) * (value - axis[lo]) / (axis[hi] - axis[lo]);
  }
  function curveData(run, kind) {
    const p = run.posterior, data = p[kind], axis = kind === 'center' ? p.centers : p.lengths;
    const [lo, hi, ymax] = ranges[kind], count = 640, step = (hi - lo) / count;
    const xs = Array.from({ length: count + 1 }, (_, i) => lo + i * step);
    // Evaluate the prior on an invariant plotting grid so it is pixel-identical
    // across true sizes, independent of adaptive posterior quadrature bounds.
    const normal = (v, m, sigma) => Math.exp(-0.5 * ((v - m) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI));
    const prior = xs.map(v => kind === 'center' ? normal(v, p.priorMean, p.priorSigma) : v > 0 ? normal(Math.log(v), Math.log(p.extentMedian), p.extentLogSigma) / v : 0);
    const posterior = xs.map(v => kind === 'extent' && v === 0 ? 0 : interpolate(axis, data.posterior, v));
    const rawLikelihood = xs.map(v => interpolate(axis, data.likelihood, v));
    const likelihoodArea = step * rawLikelihood.reduce((sum, v, i) => sum + v * (i === 0 || i === count ? 0.5 : 1), 0);
    // Unit area over the same visible interval, not peak-height matching. This
    // display normalization does not enter inference or posterior normalization.
    const likelihood = rawLikelihood.map(v => v / likelihoodArea);
    const modeIndex = likelihood.reduce((best, v, i) => v > likelihood[best] ? i : best, 0);
    return { xs, prior, likelihood, posterior, lo, hi, ymax, step, likelihoodArea, likelihoodMode: xs[modeIndex] };
  }
  function bodySVG(run, width = 238, height = 122) {
    const { settings: s, detections: z } = run, half = s.trueLength / 2;
    const x = value => 17 + (value + 4.5) / 9 * (width - 34);
    let out = start(width, height, 'True target, ' + fmt(s.trueLength) + ' metres long, with two detections in its right half. The metre scale stays fixed.');
    out += '<text x="10" y="15" fill="' + C.truth + '" font-weight="700">TRUE BODY · ' + fmt(s.trueLength) + ' m</text>';
    const left = x(s.trueCenter - half), center = x(s.trueCenter), right = x(s.trueCenter + half);
    out += '<rect id="true-body" x="' + fmt(left) + '" y="48" width="' + fmt(right - left) + '" height="22" rx="3" fill="#f0ecf5" stroke="' + C.truth + '"/>';
    out += '<rect x="' + fmt(center) + '" y="49" width="' + fmt((right - left) / 2 - 1) + '" height="20" fill="#f6eee2"/>';
    out += line(center, 44, center, 78, C.truth, '3 2', 1.5);
    out += '<text x="' + fmt(center) + '" y="88" text-anchor="middle" fill="' + C.truth + '">c* = 0</text>';
    z.forEach((value, i) => {
      const labelY = i ? 41 : 29;
      out += line(x(value), labelY + 2, x(value), 52, C.likelihood);
      out += '<text x="' + fmt(x(value)) + '" y="' + labelY + '" text-anchor="middle" fill="' + C.likelihood + '">z' + (i ? '₂' : '₁') + '</text>';
      out += '<circle class="target-detection" data-position="' + value + '" cx="' + fmt(x(value)) + '" cy="59" r="4" fill="' + C.likelihood + '" stroke="white" stroke-width="1"/>';
    });
    out += line(x(-4.5), 97, x(4.5), 97, C.line);
    for (const tick of [-4, 0, 4]) out += line(x(tick), 94, x(tick), 100, C.muted) + '<text x="' + fmt(x(tick)) + '" y="113" text-anchor="middle" fill="' + C.muted + '">' + tick + ' m</text>';
    return out + end;
  }
  function densitySVG(run, kind, width = 840, height = 290, suffix = '') {
    const d = curveData(run, kind), compact = height < 180;
    const left = compact ? 31 : 44, right = width - 16, top = compact ? 25 : 33, bottom = height - (compact ? 31 : 39);
    const x = v => left + (v - d.lo) / (d.hi - d.lo) * (right - left);
    const y = v => bottom - v / d.ymax * (bottom - top);
    const clip = 'curve-clip-' + kind + suffix;
    let out = start(width, height, (kind === 'center' ? 'Center' : 'Extent') + ' prior, likelihood and posterior with fixed axes. True extent is ' + fmt(run.settings.trueLength) + ' metres.');
    out += '<text x="12" y="16" font-size="' + (compact ? 10 : 11) + '" font-weight="700">' + (kind === 'center' ? 'CENTER c' : 'EXTENT L') + '</text>';
    out += '<text x="' + (width - 12) + '" y="16" text-anchor="end" fill="' + C.posterior + '" font-size="' + (compact ? 9 : 10) + '">posterior mean ' + fmt(run.posterior[kind].mean) + ' m</text>';
    for (let tick = kind === 'center' ? -4 : 0; tick <= d.hi; tick += kind === 'center' ? 2 : 4) {
      out += line(x(tick), top, x(tick), bottom, C.line);
      out += '<text x="' + fmt(x(tick)) + '" y="' + (bottom + 14) + '" text-anchor="middle" fill="' + C.muted + '">' + tick + '</text>';
    }
    for (let i = 0; i <= 3; i++) {
      const value = i * (kind === 'center' ? 0.2 : 0.1);
      out += line(left, y(value), right, y(value), C.line);
      out += '<text x="' + (left - 5) + '" y="' + fmt(y(value) + 3) + '" text-anchor="end" font-size="9" fill="' + C.muted + '">' + fmt(value, 1) + '</text>';
    }
    out += '<defs><clipPath id="' + clip + '"><rect x="' + left + '" y="' + top + '" width="' + (right - left) + '" height="' + (bottom - top) + '"/></clipPath></defs>';
    out += '<g clip-path="url(#' + clip + ')" data-x-min="' + d.lo + '" data-x-max="' + d.hi + '" data-y-max="' + d.ymax + '">';
    const trueValue = kind === 'center' ? run.settings.trueCenter : run.settings.trueLength;
    out += line(x(trueValue), top, x(trueValue), bottom, C.truth, '3 3', 1.4);
    const path = key => d.xs.map((value, i) => (i ? 'L' : 'M') + fmt(x(value)) + ',' + fmt(y(d[key][i]))).join(' ');
    out += '<path d="' + path('posterior') + 'L' + fmt(x(d.hi)) + ',' + bottom + 'L' + fmt(x(d.lo)) + ',' + bottom + 'Z" fill="' + C.posterior + '" fill-opacity=".07"/>';
    for (const [key, color, dash] of [['prior', C.prior, ''], ['likelihood', C.likelihood, '6 4'], ['posterior', C.posterior, '']]) {
      out += '<path class="' + kind + '-' + key + '" d="' + path(key) + '" fill="none" stroke="' + color + '" stroke-width="' + (key === 'posterior' ? 3 : 2) + '" stroke-dasharray="' + dash + '"/>';
    }
    out += '</g><text x="' + width / 2 + '" y="' + (height - 5) + '" text-anchor="middle" fill="' + C.muted + '" font-size="' + (compact ? 9 : 10) + '">Possible ' + (kind === 'center' ? 'center c' : 'extent L') + ' (m)</text>';
    return out + end;
  }
  function overviewSVG(runs, width = 1088, height = 398) {
    const gap = 16, chartW = (width - gap) / 2, chartH = (height - 72) / 2;
    let out = start(width, height, 'Compare true extents of 2 and 8 metres. Fixed priors; changed likelihoods and posteriors for both center and extent.') + '</g>';
    runs.forEach((run, i) => {
      const offset = i * (chartW + gap);
      out += '<g transform="translate(' + offset + ' 0)" font-family="Arial,sans-serif"><text x="12" y="19" font-size="15" font-weight="700" fill="' + C.truth + '">TRUE EXTENT ' + run.settings.trueLength + ' m</text><text x="12" y="39" font-size="12" fill="' + C.muted + '">Two right-side detections: ' + run.detections.map(v => fmt(v) + ' m').join(', ') + '</text></g>';
      for (const [row, kind] of ['center', 'extent'].entries()) {
        out += densitySVG(run, kind, chartW, chartH, '-print-' + i).replace('<svg ', '<svg x="' + offset + '" y="' + (52 + row * chartH) + '" width="' + chartW + '" height="' + chartH + '" ');
      }
    });
    out += '<text x="' + width / 2 + '" y="' + (height - 3) + '" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" fill="' + C.muted + '">Blue: fixed prior · Orange dashed: likelihood (unit area in plot) · Green: posterior · Purple dotted: truth</text>';
    return out + '</svg>';
  }
  return Object.freeze({ ranges, curveData, bodySVG, densitySVG, overviewSVG });
});
