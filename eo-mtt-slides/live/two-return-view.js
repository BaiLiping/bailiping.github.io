/* Pure SVG views shared by the live two-return experiment and print fallback. */
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
  function bodySVG(run, width = 840, height = 110) {
    const { settings: s, detections: z, posterior: p } = run, half = s.trueLength / 2, estimatedHalf = p.extent.mean / 2;
    const lo = Math.min(s.trueCenter - half, p.center.mean - estimatedHalf) - 0.8;
    const hi = Math.max(s.trueCenter + half, p.center.mean + estimatedHalf) + 0.8;
    const x = value => 22 + (value - lo) / (hi - lo) * (width - 44);
    let out = start(width, height, 'Two detections on the ' + (s.side > 0 ? 'right' : 'left') + ' half of a finite target. Purple is the true body; green is the posterior mean body.');
    out += '<text x="12" y="16" fill="' + C.muted + '" font-size="' + (width < 420 ? 9 : 10) + '">Two ' + (s.side > 0 ? 'right' : 'left') + '-side detections: z₁ = ' + fmt(z[0]) + ' m, z₂ = ' + fmt(z[1]) + ' m</text>';
    const left = x(s.trueCenter - half), center = x(s.trueCenter), right = x(s.trueCenter + half);
    out += '<rect id="true-body" x="' + fmt(left) + '" y="45" width="' + fmt(right - left) + '" height="22" rx="4" fill="#f0ecf5" stroke="' + C.truth + '"/>';
    out += '<rect x="' + fmt(s.side > 0 ? center : left) + '" y="45" width="' + fmt((right - left) / 2) + '" height="22" fill="#f6eee2" fill-opacity=".85"/>';
    out += line(center, 40, center, 72, C.truth, '3 2', 1.5);
    out += '<text x="' + fmt(center) + '" y="78" text-anchor="middle" fill="' + C.truth + '">c*</text>';
    z.forEach((value, i) => {
      out += '<circle class="target-detection" data-position="' + value + '" cx="' + fmt(x(value)) + '" cy="56" r="5" fill="' + C.likelihood + '" stroke="white" stroke-width="1.5"/>';
      const labelY = i ? 40 : 30;
      out += line(x(value), labelY + 2, x(value), 49, C.likelihood);
      out += '<text x="' + fmt(x(value)) + '" y="' + labelY + '" text-anchor="middle" fill="' + C.likelihood + '">z' + (i ? '₂' : '₁') + '</text>';
    });
    out += line(x(p.center.mean - estimatedHalf), 83, x(p.center.mean + estimatedHalf), 83, C.posterior, '', 4);
    out += line(x(p.center.mean), 77, x(p.center.mean), 89, C.posterior, '', 2);
    out += '<text x="' + width / 2 + '" y="103" text-anchor="middle" font-size="' + (width < 420 ? 8 : 9) + '" fill="' + C.muted + '">Purple: true body · Green: posterior mean body · c*: true center</text>';
    return out + end;
  }
  function ranges(run) {
    const p = run.posterior, s = run.settings;
    return {
      center: [Math.min(p.priorMean - 3.7 * p.priorSigma, p.center.interval[0] - 0.5, s.trueCenter - 0.8, ...run.detections.map(z => z - 0.8)),
        Math.max(p.priorMean + 3.7 * p.priorSigma, p.center.interval[1] + 0.5, s.trueCenter + 0.8, ...run.detections.map(z => z + 0.8))],
      extent: [0, Math.ceil(Math.max(s.trueLength * 1.15, p.extent.interval[1] * 1.25, p.extentMedian * Math.exp(2.6 * p.extentLogSigma)) / 2) * 2]
    };
  }
  function stepFor(span) {
    const rough = span / 5, power = 10 ** Math.floor(Math.log10(rough));
    return [1, 2, 5, 10].find(n => n * power >= rough) * power;
  }
  function densitySVG(run, kind, width = 414, height = 232) {
    const p = run.posterior, data = p[kind], axis = kind === 'center' ? p.centers : p.lengths;
    const [lo, hi] = ranges(run)[kind], left = 38, right = width - 13, top = 32, bottom = height - 43;
    const visible = axis.map((value, i) => value >= lo && value <= hi ? i : -1).filter(i => i >= 0);
    const maxDensity = Math.max(...visible.map(i => Math.max(data.prior[i], data.posterior[i])));
    const maxLikelihood = Math.max(...visible.map(i => data.likelihood[i]));
    const likelihoodScale = maxDensity * 0.9 / maxLikelihood, ymax = maxDensity * 1.15;
    const x = v => left + (v - lo) / (hi - lo) * (right - left);
    const y = v => bottom - v / ymax * (bottom - top);
    const clip = 'clip-' + kind;
    let out = start(width, height, (kind === 'center' ? 'Center' : 'Extent') + ' prior, nuisance-marginalized likelihood, and joint-posterior marginal.');
    out += '<text x="12" y="17" font-size="11" font-weight="700">' + (kind === 'center' ? 'CENTER c (m)' : 'EXTENT L (m)') + '</text>';
    out += '<text x="' + (width - 12) + '" y="17" text-anchor="end" fill="' + C.posterior + '">mean ' + fmt(data.mean) + ' m</text>';
    const step = stepFor(hi - lo);
    for (let tick = Math.ceil(lo / step) * step; tick <= hi; tick += step) {
      out += line(x(tick), top, x(tick), bottom, C.line);
      out += '<text x="' + fmt(x(tick)) + '" y="' + (bottom + 15) + '" text-anchor="middle" fill="' + C.muted + '">' + fmt(tick, 1) + '</text>';
    }
    for (let i = 0; i <= 3; i++) {
      const v = ymax * i / 3;
      out += line(left, y(v), right, y(v), C.line);
      out += '<text x="' + (left - 5) + '" y="' + fmt(y(v) + 3) + '" text-anchor="end" font-size="9" fill="' + C.muted + '">' + fmt(v) + '</text>';
    }
    out += '<defs><clipPath id="' + clip + '"><rect x="' + left + '" y="' + top + '" width="' + (right - left) + '" height="' + (bottom - top) + '"/></clipPath></defs><g clip-path="url(#' + clip + ')">';
    const trueValue = kind === 'center' ? run.settings.trueCenter : run.settings.trueLength;
    out += line(x(trueValue), top, x(trueValue), bottom, C.truth, '3 3', 1.6);
    for (const [key, color, scale, dash] of [['prior', C.prior, 1, ''], ['likelihood', C.likelihood, likelihoodScale, '5 3'], ['posterior', C.posterior, 1, '']]) {
      const path = axis.map((value, i) => (i ? 'L' : 'M') + fmt(x(value)) + ',' + fmt(y(data[key][i] * scale))).join(' ');
      out += '<path class="' + kind + '-' + key + '" d="' + path + '" fill="none" stroke="' + color + '" stroke-width="' + (key === 'posterior' ? 2.8 : 1.8) + '" stroke-dasharray="' + dash + '"/>';
    }
    out += '</g><text x="' + width / 2 + '" y="' + (height - 9) + '" text-anchor="middle" fill="' + C.posterior + '" font-size="10">95% credible interval [' + data.interval.map(v => fmt(v)).join(', ') + '] m</text>';
    return out + end;
  }
  function jointSVG(run, width = 840, height = 232) {
    const p = run.posterior, range = ranges(run);
    const [lo, hi] = range.center, maxL = range.extent[1];
    const compact = width < 540;
    const left = 43, right = width - 16, top = compact ? 47 : 30, bottom = height - 32;
    const x = v => left + (v - lo) / (hi - lo) * (right - left);
    const y = v => bottom - v / maxL * (bottom - top);
    const cols = 110, rows = 55, dx = (hi - lo) / cols, dl = maxL / rows;
    const values = [];
    let max = 0;
    for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) {
      const c = lo + (i + 0.5) * dx, L = (j + 0.5) * dl;
      const ci = Math.floor((c - p.xlo) / p.dx), li = Math.floor((Math.log(L) - p.qlo) / p.dq);
      const value = ci >= 0 && ci < p.nx && li >= 0 && li < p.nl ? p.joint[li * p.nx + ci] / (p.dx * p.dq * p.lengths[li]) : 0;
      values.push(value); max = Math.max(max, value);
    }
    let out = start(width, height, 'Joint posterior over center and extent. Darker green indicates greater probability density.');
    out += '<text x="12" y="17" font-size="11" font-weight="700">JOINT POSTERIOR p(c,L | z₁,z₂)</text><text x="' + (compact ? 12 : width - 12) + '" y="' + (compact ? 33 : 17) + '" text-anchor="' + (compact ? 'start' : 'end') + '" fill="' + C.muted + '" font-size="9">Darker = higher density · ρ = ' + fmt(p.correlation) + '</text>';
    values.forEach((value, k) => {
      if (value < max * 0.001) return;
      const shade = Math.round(Math.sqrt(value / max) * 18) / 18;
      const rgb = [244, 249, 244].map((v, i) => Math.round(v + ([47, 107, 79][i] - v) * shade));
      const i = k % cols, j = Math.floor(k / cols);
      out += '<rect x="' + fmt(x(lo + i * dx)) + '" y="' + fmt(y((j + 1) * dl)) + '" width="' + fmt((right - left) / cols + 0.2) + '" height="' + fmt((bottom - top) / rows + 0.2) + '" fill="rgb(' + rgb.join(',') + ')"/>';
    });
    const xtick = stepFor(hi - lo), ytick = stepFor(maxL);
    for (let v = Math.ceil(lo / xtick) * xtick; v <= hi; v += xtick) out += '<text x="' + fmt(x(v)) + '" y="' + (bottom + 15) + '" text-anchor="middle" fill="' + C.muted + '">' + fmt(v, 1) + '</text>';
    for (let v = 0; v <= maxL; v += ytick) out += '<text x="' + (left - 7) + '" y="' + fmt(y(v) + 3) + '" text-anchor="end" fill="' + C.muted + '">' + fmt(v, 1) + '</text>';
    out += line(left, top, left, bottom, C.line) + line(left, bottom, right, bottom, C.line);
    out += '<circle cx="' + fmt(x(run.settings.trueCenter)) + '" cy="' + fmt(y(run.settings.trueLength)) + '" r="5" fill="none" stroke="' + C.truth + '" stroke-width="2"/>';
    out += line(x(p.center.mean) - 5, y(p.extent.mean), x(p.center.mean) + 5, y(p.extent.mean), '#203129', '', 2);
    out += line(x(p.center.mean), y(p.extent.mean) - 5, x(p.center.mean), y(p.extent.mean) + 5, '#203129', '', 2);
    out += '<text x="' + (width / 2) + '" y="' + (height - 3) + '" text-anchor="middle" fill="' + C.muted + '">Center c (m) · ○ truth · + posterior mean</text><text transform="translate(11 ' + ((top + bottom) / 2) + ') rotate(-90)" text-anchor="middle" fill="' + C.muted + '">Extent L (m)</text>';
    return out + end;
  }
  function overviewSVG(run, width = 1088, height = 398) {
    const gap = 10, chartY = 137, chartW = (width - gap) / 2;
    return start(width, height, 'Two same-side detections: joint inference of center and extent.') + '</g>' +
      bodySVG(run, width, 110).replace('<svg ', '<svg x="0" y="0" width="' + width + '" height="110" ') +
      '<text x="12" y="128" font-family="Arial,sans-serif" font-size="11" fill="' + C.muted + '">Blue: prior · Orange dashed: marginalized likelihood (scaled) · Green: posterior · Purple dotted: truth</text>' +
      densitySVG(run, 'center', chartW, height - chartY).replace('<svg ', '<svg x="0" y="' + chartY + '" width="' + chartW + '" height="' + (height - chartY) + '" ') +
      densitySVG(run, 'extent', chartW, height - chartY).replace('<svg ', '<svg x="' + (chartW + gap) + '" y="' + chartY + '" width="' + chartW + '" height="' + (height - chartY) + '" ') + '</svg>';
  }
  return Object.freeze({ bodySVG, densitySVG, jointSVG, overviewSVG });
});
