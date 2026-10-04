/* Shared scene renderer for the interactive lab and its generated print figure. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('./extent-model.js'));
  else root.ExtendedTargetView = factory(root.ExtendedTargetModel);
})(typeof globalThis === 'object' ? globalThis : this, function (model) {
  'use strict';
  const colors = { truth: '#75638f', prior: '#496e87', posterior: '#2f6b4f', returns: '#b16f35', muted: '#66756e', line: '#e3e7df' };
  const fmt = x => Number(x.toFixed(2));
  function sceneSVG(run, { width = 760, height = 350, showTruth = true, showSources = false, showUncertainty = true } = {}) {
    const { frame, history } = run, { prior, posterior, truth, detections } = frame;
    const legend = [
      ...(showTruth ? [['true body', colors.truth, '5 3']] : []),
      ['predicted extent', colors.prior, '6 3'], ['updated extent', colors.posterior, ''], ['detections', colors.returns, 'dot'],
      ...(showUncertainty ? [['95% center region', colors.muted, '2 3']] : [])
    ];
    let lx = 12, ly = 16;
    const legendItems = legend.map(([label, color, dash]) => {
      const itemWidth = label.length * 5.4 + 28;
      if (lx + itemWidth > width - 8) { lx = 12; ly += 17; }
      const item = { label, color, dash, x: lx, y: ly };
      lx += itemWidth;
      return item;
    });
    const Xprior = model.extentMean(prior), Xpost = model.extentMean(posterior);
    const centerRegion = model.math.scale(model.math.positionCov(posterior.covariance), 5.991464547);
    const bounds = [...detections, ...history.slice(-9).map(f => f.posterior.mean.slice(0, 2))];
    for (const [center, X] of [[prior.mean, Xprior], [posterior.mean, Xpost], ...(showTruth ? [[truth.center, truth.X]] : []), ...(showUncertainty ? [[posterior.mean, centerRegion]] : [])]) {
      const rx = Math.sqrt(X[0][0]), ry = Math.sqrt(X[1][1]);
      bounds.push([center[0] - rx, center[1] - ry], [center[0] + rx, center[1] + ry]);
    }
    const xmin = Math.min(...bounds.map(p => p[0])), xmax = Math.max(...bounds.map(p => p[0]));
    const ymin = Math.min(...bounds.map(p => p[1])), ymax = Math.max(...bounds.map(p => p[1]));
    const cx = (xmin + xmax) / 2, cy = (ymin + ymax) / 2;
    const top = ly + 14, left = 47, right = 15, bottom = 32;
    const plotW = width - left - right, plotH = height - top - bottom;
    const unit = Math.min(plotW / Math.max(20, xmax - xmin + 4), plotH / Math.max(13, ymax - ymin + 4));
    const x = v => left + plotW / 2 + (v - cx) * unit, y = v => top + plotH / 2 - (v - cy) * unit;
    const lowX = cx - plotW / (2 * unit), highX = cx + plotW / (2 * unit);
    const lowY = cy - plotH / (2 * unit), highY = cy + plotH / (2 * unit);
    const tick = highX - lowX > 35 ? 5 : 2;
    const path = points => points.map((p, i) => `${i ? 'L' : 'M'}${fmt(x(p[0]))},${fmt(y(p[1]))}`).join(' ');
    const ellipse = (id, center, X, color, dash, opacity, strokeWidth = 2) => {
      const S = model.math.sqrt2(X);
      const points = Array.from({ length: 81 }, (_, i) => {
        const t = i * 2 * Math.PI / 80, c = Math.cos(t), s = Math.sin(t);
        return [center[0] + S[0][0] * c + S[0][1] * s, center[1] + S[1][0] * c + S[1][1] * s];
      });
      return `<path id="${id}" d="${path(points)} Z" fill="${color}" fill-opacity="${opacity}" stroke="${color}" stroke-width="${strokeWidth}" ${dash ? `stroke-dasharray="${dash}"` : ''}/>`;
    };
    let out = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="scene-title scene-desc"><title id="scene-title">Extended-target tracking, scan ${frame.scan}</title><desc id="scene-desc">${detections.length} noisy detections from a moving ellipse. Estimated length ${fmt(frame.estimate.length)} meters, width ${fmt(frame.estimate.width)} meters, orientation ${fmt(frame.estimate.angle)} degrees. Physical extent and center uncertainty are separate ellipses.</desc><rect width="100%" height="100%" rx="9" fill="#fffefb"/><g font-family="Inter,Arial,sans-serif" font-size="10">`;
    for (const { label, color, dash, x: lx, y: ly } of legendItems) {
      out += dash === 'dot' ? `<circle cx="${lx + 6}" cy="${ly - 3}" r="3" fill="${color}"/>` : `<path d="M${lx},${ly - 3}h16" stroke="${color}" stroke-width="2" stroke-dasharray="${dash}"/>`;
      out += `<text x="${lx + 21}" y="${ly}" fill="${color}">${label}</text>`;
    }
    for (let v = Math.ceil(lowX / tick) * tick; v <= highX; v += tick) out += `<path d="M${fmt(x(v))},${top}V${height - bottom}" stroke="${colors.line}"/><text x="${fmt(x(v))}" y="${height - 17}" text-anchor="middle" fill="${colors.muted}">${v}</text>`;
    for (let v = Math.ceil(lowY / tick) * tick; v <= highY; v += tick) out += `<path d="M${left},${fmt(y(v))}H${width - right}" stroke="${colors.line}"/><text x="${left - 8}" y="${fmt(y(v) + 3)}" text-anchor="end" fill="${colors.muted}">${v}</text>`;
    out += `<text x="${width - right}" y="${height - 3}" text-anchor="end" fill="${colors.muted}">world x (m) · equal axis scale</text><text transform="translate(12 ${(top + height - bottom) / 2}) rotate(-90)" text-anchor="middle" fill="${colors.muted}">world y (m)</text>`;
    if (showTruth) out += `<path d="${path(history.slice(-9).map(f => f.truth.center))}" fill="none" stroke="${colors.truth}" stroke-opacity=".5" stroke-dasharray="2 3"/>` + ellipse('true-extent', truth.center, truth.X, colors.truth, '5 3', 0.09);
    out += `<path d="${path(history.slice(-9).map(f => f.posterior.mean))}" fill="none" stroke="${colors.posterior}" stroke-opacity=".45" stroke-width="1.5"/>`;
    out += ellipse('predicted-extent', prior.mean, Xprior, colors.prior, '7 4', 0, 1.8);
    out += ellipse('estimated-extent', posterior.mean, Xpost, colors.posterior, '', 0.05, 2.8);
    if (showUncertainty) out += ellipse('center-uncertainty', posterior.mean, centerRegion, colors.muted, '2 3', 0.1, 1.5);
    detections.forEach((z, i) => {
      if (showSources) { const s = frame.sources[i]; out += `<path d="M${fmt(x(s[0]))},${fmt(y(s[1]))}L${fmt(x(z[0]))},${fmt(y(z[1]))}" stroke="${colors.returns}" stroke-opacity=".5"/><circle class="source-point" cx="${fmt(x(s[0]))}" cy="${fmt(y(s[1]))}" r="2.8" fill="white" stroke="${colors.truth}"/>`; }
      out += `<circle class="detection" cx="${fmt(x(z[0]))}" cy="${fmt(y(z[1]))}" r="3.3" fill="${colors.returns}" stroke="#fffefb" stroke-width="1"/>`;
    });
    const cross = (center, color, size) => `<path d="M${fmt(x(center[0]) - size)},${fmt(y(center[1]))}h${size * 2}M${fmt(x(center[0]))},${fmt(y(center[1]) - size)}v${size * 2}" stroke="${color}" stroke-width="2"/>`;
    if (showTruth) out += cross(truth.center, colors.truth, 5);
    out += cross(posterior.mean, colors.posterior, 6);
    return out + '</g></svg>';
  }
  return Object.freeze({ sceneSVG, colors });
});
