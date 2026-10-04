(() => {
  'use strict';
  const model = window.ExtendedTargetModel;
  const NS = 'http://www.w3.org/2000/svg';
  const controls = { priorMean: 'prior-mean', priorSigma: 'prior-sigma', centroid: 'measurement', noiseSigma: 'measurement-sigma', extentSigma: 'target-extent' };
  const colors = { prior: '#496e87', likelihood: '#b16f35', posterior: '#2f6b4f', extent: '#75638f', muted: '#66756e', line: '#d8ded7' };
  const fmt = (x, digits = 3) => Number(x.toFixed(digits)).toString();
  let state;

  function svg(parent, tag, attributes, text) {
    const element = document.createElementNS(NS, tag);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
    if (text !== undefined) element.textContent = text;
    parent.appendChild(element);
    return element;
  }

  function draw() {
    if (!state) return;
    const root = document.getElementById('density-drawing');
    const cloud = document.getElementById('cloud-drawing');
    const plot = document.getElementById('density-plot');
    const cloudPlot = document.getElementById('cloud-plot');
    const width = plot.clientWidth || 560, height = plot.clientHeight || 270;
    const cloudHeight = cloudPlot.clientHeight || 92;
    plot.setAttribute('viewBox', `0 0 ${width} ${height}`);
    cloudPlot.setAttribute('viewBox', `0 0 ${width} ${cloudHeight}`);
    root.replaceChildren(); cloud.replaceChildren();
    const returns = model.displayReturns(state);
    const limit = Math.ceil(Math.max(6, Math.abs(state.priorMean) + 3 * state.priorSigma, Math.abs(state.centroid) + 3 * Math.sqrt(state.centroidVar), ...returns.map(x => Math.abs(x) + 0.5)));
    const X = x => 32 + (x + limit) / (2 * limit) * (width - 50);
    const base = height - 40;
    const peak = Math.max(model.gaussian(state.priorMean, state.priorMean, state.priorVar), model.gaussian(state.centroid, state.centroid, state.centroidVar), model.gaussian(state.postMean, state.postMean, state.postVar));
    const Y = y => base - y / peak * (base - 35);
    const tickStep = limit <= 8 ? 2 : 4;
    for (let x = Math.ceil(-limit / tickStep) * tickStep; x <= limit; x += tickStep) {
      svg(root, 'line', { x1: X(x), x2: X(x), y1: 25, y2: base, stroke: colors.line, 'stroke-width': 0.8 });
      svg(root, 'text', { x: X(x), y: base + 17, fill: colors.muted, 'font-size': 10, 'text-anchor': 'middle' }, x);
      svg(cloud, 'text', { x: X(x), y: cloudHeight - 7, fill: colors.muted, 'font-size': 10, 'text-anchor': 'middle' }, x);
    }
    svg(root, 'line', { x1: 32, x2: width - 18, y1: base, y2: base, stroke: colors.muted });
    svg(root, 'text', { x: 32, y: 16, fill: colors.muted, 'font-size': 9 }, 'density · likelihood normalized over x');
    svg(root, 'text', { x: width / 2, y: height - 6, fill: colors.muted, 'font-size': 10, 'text-anchor': 'middle' }, 'object center x');
    const curves = [
      { key: 'prior', mean: state.priorMean, variance: state.priorVar, label: 'm⁻' },
      { key: 'likelihood', mean: state.centroid, variance: state.centroidVar, label: 'z̄' },
      { key: 'posterior', mean: state.postMean, variance: state.postVar, label: 'm⁺' }
    ];
    // Include dense samples around each mean so very narrow densities are visible.
    for (const curve of curves) {
      const xs = Array.from({ length: 561 }, (_, i) => -limit + i / 560 * 2 * limit);
      for (let j = -100; j <= 100; j++) xs.push(curve.mean + j / 20 * Math.sqrt(curve.variance));
      const points = xs.filter(x => Math.abs(x) <= limit).sort((a, b) => a - b);
      const path = points.map((x, i) => `${i ? 'L' : 'M'}${X(x).toFixed(2)},${Y(model.gaussian(x, curve.mean, curve.variance)).toFixed(2)}`).join(' ');
      if (curve.key === 'posterior') svg(root, 'path', { d: `${path} L${X(points.at(-1))},${base} L${X(points[0])},${base} Z`, fill: colors.posterior, opacity: 0.08 });
      svg(root, 'path', { id: `${curve.key}-curve`, d: path, fill: 'none', stroke: colors[curve.key], 'stroke-width': curve.key === 'posterior' ? 3 : 2.2, ...(curve.key === 'likelihood' ? { 'stroke-dasharray': '6 4' } : {}) });
      svg(root, 'line', { x1: X(curve.mean), x2: X(curve.mean), y1: 25, y2: base, stroke: colors[curve.key], 'stroke-width': 1.2, 'stroke-dasharray': '3 4', opacity: 0.6 });
    }
    const cloudY = (cloudHeight + 6) / 2;
    svg(cloud, 'line', { x1: 32, x2: width - 18, y1: cloudY, y2: cloudY, stroke: colors.line });
    const left = X(state.centroid - state.extentSigma), right = X(state.centroid + state.extentSigma);
    svg(cloud, 'rect', { id: 'extent-band', x: left, y: cloudY - 16, width: Math.max(0, right - left), height: 32, rx: 5, fill: colors.extent, opacity: 0.17 });
    svg(cloud, 'line', { x1: left, x2: right, y1: cloudY - 16, y2: cloudY - 16, stroke: colors.extent, 'stroke-width': 2 });
    svg(cloud, 'text', { x: X(state.centroid), y: 15, fill: colors.extent, 'font-size': 10, 'text-anchor': 'middle' }, state.extentSigma ? `extent band: z̄ ± ${fmt(state.extentSigma, 2)}` : 'zero spatial extent');
    returns.forEach((z, i) => svg(cloud, 'circle', { cx: X(z), cy: cloudY + (i % 2 ? -4 : 4), r: 4, fill: colors.likelihood, stroke: '#fffefb', 'stroke-width': 1.5 }));
    svg(cloud, 'line', { x1: X(state.centroid), x2: X(state.centroid), y1: cloudY - 13, y2: cloudY + 16, stroke: colors.posterior, 'stroke-width': 2 });
  }

  function update() {
    const parameters = {};
    for (const [key, id] of Object.entries(controls)) {
      parameters[key] = Number(document.getElementById(id).value);
      document.getElementById(`${id}-out`).textContent = fmt(parameters[key], 2);
    }
    state = model.posterior(parameters);
    document.getElementById('gain').textContent = fmt(state.gain, 5);
    document.getElementById('posterior').textContent = `m⁺ = ${fmt(state.postMean, 4)}\nσ⁺ = ${fmt(state.postSigma, 4)}`;
    document.getElementById('centroid-variance').textContent = fmt(state.centroidVar, 5);
    document.getElementById('return-variance').textContent = `${fmt(state.extentSigma ** 2, 4)} + ${fmt(state.noiseSigma ** 2, 4)}`;
    document.getElementById('density-description').textContent = `Prior mean ${fmt(state.priorMean)}, measurement centroid ${fmt(state.centroid)}, posterior mean ${fmt(state.postMean, 4)} and standard deviation ${fmt(state.postSigma, 4)}. Kalman gain ${fmt(state.gain, 5)}.`;
    document.getElementById('takeaway').textContent = state.extentSigma === 0
      ? 'Zero extent: the 8 returns share one source location; sensor noise remains. Rc = σr² / 8.'
      : 'Hold the other sliders fixed: larger extent → broader centroid likelihood → lower gain and a wider center posterior.';
    draw();
  }

  for (const id of Object.values(controls)) document.getElementById(id).addEventListener('input', update);
  document.getElementById('reset').addEventListener('click', () => {
    for (const [key, id] of Object.entries(controls)) document.getElementById(id).value = model.defaults[key];
    update();
  });
  document.getElementById('zero-extent').addEventListener('click', () => {
    document.getElementById('target-extent').value = 0;
    update();
  });
  // Event-driven SVG only: no animation loop survives slide navigation.
  const observer = new ResizeObserver(draw);
  const observe = () => observer.observe(document.querySelector('.visuals'));
  observe();
  window.addEventListener('pagehide', () => observer.disconnect());
  window.addEventListener('pageshow', observe);
  window.ExtendedTargetLab = Object.freeze({ getState: () => ({ ...state }) });
  update();
})();
