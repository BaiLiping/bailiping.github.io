(() => {
  'use strict';
  const model = window.TwoReturnModel, view = window.TwoReturnView, el = id => document.getElementById(id);
  const fmt = x => Number(x.toFixed(2)).toString();
  let run, kind = 'center', pending = null;
  function draw() {
    if (!run) return;
    const body = el('body-scene'), chart = el('density-chart');
    body.innerHTML = view.bodySVG(run, Math.max(220, body.clientWidth), 122);
    chart.innerHTML = view.densitySVG(run, kind, Math.max(260, chart.clientWidth), Math.max(190, chart.clientHeight));
    for (const name of ['center', 'extent']) el('show-' + name).setAttribute('aria-pressed', String(kind === name));
    const p = run.posterior, data = view.curveData(run, kind), isCenter = kind === 'center';
    const logVariance = p.extentLogSigma ** 2;
    const priorMean = isCenter ? p.priorMean : p.extentMedian * Math.exp(logVariance / 2);
    const priorSigma = isCenter ? p.priorSigma : priorMean * Math.sqrt(Math.expm1(logVariance));
    el('prior-summary').textContent = 'μ ' + fmt(priorMean) + ' m · σ ' + fmt(priorSigma) + ' m';
    el('likelihood-summary').textContent = 'Peak at ' + fmt(data.likelihoodMode) + ' m';
    el('posterior-summary').textContent = 'μ ' + fmt(p[kind].mean) + ' m · σ ' + fmt(p[kind].sigma) + ' m';
    el('takeaway').textContent = 'L* = ' + fmt(run.settings.trueLength) + ' m: the blue prior is unchanged. ' + (isCenter ? 'The two right-side detections pull the center posterior right.' : 'The two detections update the size distribution; they do not identify the whole body.') + ' 95% interval: [' + p[kind].interval.map(v => fmt(v)).join(', ') + '] m.';
  }
  function update() {
    pending = null;
    const trueLength = Number(el('target-extent').value);
    el('target-extent-out').textContent = fmt(trueLength) + ' m';
    run = model.example({ trueLength });
    el('observations').textContent = 'z₁ = ' + fmt(run.detections[0]) + ' m · z₂ = ' + fmt(run.detections[1]) + ' m';
    draw();
  }
  function queueUpdate() {
    if (pending !== null) cancelAnimationFrame(pending);
    pending = requestAnimationFrame(update);
  }
  el('target-extent').addEventListener('input', queueUpdate);
  for (const [id, size] of [['small-extent', 2], ['large-extent', 8]]) el(id).addEventListener('click', () => { el('target-extent').value = size; queueUpdate(); });
  el('reset').addEventListener('click', () => { kind = 'center'; el('target-extent').value = model.defaults.trueLength; queueUpdate(); });
  for (const name of ['center', 'extent']) el('show-' + name).addEventListener('click', () => { kind = name; draw(); });
  const observer = new ResizeObserver(draw);
  const observe = () => { observer.observe(el('body-scene')); observer.observe(el('density-chart')); };
  const cleanup = () => { if (pending !== null) cancelAnimationFrame(pending); pending = null; observer.disconnect(); };
  window.addEventListener('pagehide', cleanup);
  window.addEventListener('pageshow', observe);
  window.addEventListener('bento-live-visibility', event => { if (event.detail?.paused) cleanup(); else { observe(); queueUpdate(); } });
  window.TwoReturnLab = Object.freeze({
    getState: () => ({ settings: { ...run.settings }, detections: [...run.detections], kind,
      center: { mean: run.posterior.center.mean, sigma: run.posterior.center.sigma, interval: [...run.posterior.center.interval] },
      extent: { mean: run.posterior.extent.mean, sigma: run.posterior.extent.sigma, interval: [...run.posterior.extent.interval] },
      correlation: run.posterior.correlation })
  });
  observe(); update();
})();
