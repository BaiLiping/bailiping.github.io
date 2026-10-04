(() => {
  'use strict';
  const model = window.TwoReturnModel, view = window.TwoReturnView, el = id => document.getElementById(id);
  const controls = { priorMean: 'prior-mean', priorSigma: 'prior-sigma', extentMedian: 'extent-median', trueLength: 'target-extent', trueCenter: 'target-center', noise: 'sensor-noise' };
  const fmt = x => Number(x.toFixed(2)).toString();
  let run, side = 1, jointVisible = false, pending = null;
  function draw() {
    if (!run) return;
    const body = el('body-scene');
    body.innerHTML = view.bodySVG(run, Math.max(280, body.clientWidth), 110);
    const panels = el('density-panels'), joint = el('joint-density');
    panels.hidden = jointVisible; joint.hidden = !jointVisible;
    if (jointVisible) joint.innerHTML = view.jointSVG(run, Math.max(280, joint.clientWidth), Math.max(200, joint.clientHeight));
    else for (const kind of ['center', 'extent']) {
      const panel = el(kind + '-density');
      panel.innerHTML = view.densitySVG(run, kind, Math.max(260, panel.clientWidth), Math.max(190, panel.clientHeight));
    }
    el('toggle-joint').setAttribute('aria-pressed', String(jointVisible));
    el('toggle-joint').textContent = jointVisible ? 'Marginal curves' : 'Joint view';
  }
  function update() {
    pending = null;
    const parameters = { side };
    for (const [key, id] of Object.entries(controls)) {
      parameters[key] = Number(el(id).value);
      el(id + '-out').textContent = fmt(parameters[key]) + ' m';
    }
    run = model.example(parameters);
    const p = run.posterior;
    el('flip-side').textContent = side > 0 ? 'Move pair left' : 'Move pair right';
    el('takeaway').textContent = 'Posterior mean: center ' + fmt(p.center.mean) + ' m · extent ' + fmt(p.extent.mean) + ' m. Two same-side returns leave a range of plausible centers and sizes.';
    draw();
  }
  function queueUpdate() {
    if (pending !== null) cancelAnimationFrame(pending);
    pending = requestAnimationFrame(update);
  }
  for (const id of Object.values(controls)) el(id).addEventListener('input', queueUpdate);
  el('flip-side').addEventListener('click', () => { side *= -1; queueUpdate(); });
  el('reset').addEventListener('click', () => {
    side = 1; jointVisible = false;
    for (const [key, id] of Object.entries(controls)) el(id).value = model.defaults[key];
    queueUpdate();
  });
  el('toggle-joint').addEventListener('click', () => { jointVisible = !jointVisible; draw(); });
  const observer = new ResizeObserver(draw);
  const observe = () => { observer.observe(el('body-scene')); observer.observe(el('density-panels')); observer.observe(el('joint-density')); };
  const cleanup = () => { if (pending !== null) cancelAnimationFrame(pending); pending = null; observer.disconnect(); };
  window.addEventListener('pagehide', cleanup);
  window.addEventListener('pageshow', observe);
  window.addEventListener('bento-live-visibility', event => { if (event.detail?.paused) cleanup(); else { observe(); queueUpdate(); } });
  window.TwoReturnLab = Object.freeze({
    getState: () => ({ settings: { ...run.settings }, detections: [...run.detections],
      center: { mean: run.posterior.center.mean, sigma: run.posterior.center.sigma, interval: [...run.posterior.center.interval] },
      extent: { mean: run.posterior.extent.mean, sigma: run.posterior.extent.sigma, interval: [...run.posterior.extent.interval] },
      correlation: run.posterior.correlation, jointVisible })
  });
  observe(); update();
})();
