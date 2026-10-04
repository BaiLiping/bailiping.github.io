(() => {
  'use strict';
  const model = window.ExtendedTargetModel, view = window.ExtendedTargetView;
  const controls = { length: 'target-length', width: 'target-width', angle: 'target-angle', noise: 'sensor-noise', count: 'return-count' };
  const units = { length: ' m', width: ' m', angle: '°', noise: ' m', count: '' };
  const el = id => document.getElementById(id);
  const fmt = (x, digits = 2) => Number(x.toFixed(digits)).toString();
  let run, scan = 1, timer = null;
  function stop() {
    if (timer !== null) clearInterval(timer);
    timer = null;
    el('play').textContent = scan === model.MAX_SCANS ? 'Replay' : 'Play';
    el('play').setAttribute('aria-pressed', 'false');
  }
  function draw() {
    if (!run) return;
    const scene = el('scene');
    scene.innerHTML = view.sceneSVG(run, {
      width: Math.max(280, scene.clientWidth), height: Math.max(210, scene.clientHeight),
      showTruth: el('show-truth').checked, showSources: el('show-sources').checked,
      showUncertainty: el('show-uncertainty').checked
    });
  }
  function render() {
    const parameters = {};
    for (const [key, id] of Object.entries(controls)) {
      parameters[key] = Number(el(id).value);
      el(id + '-out').textContent = fmt(parameters[key]) + units[key];
    }
    run = model.simulate(parameters, scan);
    const f = run.frame, truthVisible = el('show-truth').checked;
    el('scan-index').value = scan;
    el('scan-index-out').textContent = scan + ' / ' + model.MAX_SCANS;
    el('scan-badge').textContent = 'SCAN ' + scan + ' / ' + model.MAX_SCANS;
    el('step').disabled = scan === model.MAX_SCANS;
    for (const name of ['length', 'width', 'angle']) {
      el('true-' + name).textContent = truthVisible ? fmt(f.truth[name], 1) + units[name] : '—';
      el('estimated-' + name).textContent = name === 'angle' && !f.estimate.identifiable ? 'near circle' : fmt(f.estimate[name], name === 'angle' ? 1 : 2) + units[name];
    }
    el('position-estimate').textContent = '(' + fmt(f.posterior.mean[0]) + ', ' + fmt(f.posterior.mean[1]) + ') m';
    el('velocity-estimate').textContent = '(' + fmt(f.posterior.mean[2]) + ', ' + fmt(f.posterior.mean[3]) + ') m/s';
    el('center-error').textContent = truthVisible ? 'Center error: ' + fmt(f.errors.center) + ' m' : 'Ground truth hidden';
    el('measurement-count').textContent = f.detections.length + ' returns this scan';
    el('takeaway').textContent = scan === 1
      ? 'The circular prior becomes an oriented ellipse after one cloud. Advance scans to refine motion and extent.'
      : 'The centroid updates motion; the cloud scatter updates length, width, and orientation. Extent is learned from the detections.';
    if (timer === null) el('play').textContent = scan === model.MAX_SCANS ? 'Replay' : 'Play';
    draw();
  }
  for (const id of Object.values(controls)) el(id).addEventListener('input', () => { stop(); render(); });
  el('scan-index').addEventListener('input', () => { stop(); scan = Number(el('scan-index').value); render(); });
  el('step').addEventListener('click', () => { stop(); scan = Math.min(model.MAX_SCANS, scan + 1); render(); });
  el('reset').addEventListener('click', () => {
    stop(); scan = 1;
    for (const [key, id] of Object.entries(controls)) el(id).value = model.defaults[key];
    el('show-truth').checked = true; el('show-sources').checked = false; el('show-uncertainty').checked = true;
    render();
  });
  el('play').addEventListener('click', () => {
    if (timer !== null) { stop(); return; }
    if (scan === model.MAX_SCANS) { scan = 1; render(); }
    el('play').textContent = 'Pause'; el('play').setAttribute('aria-pressed', 'true');
    timer = setInterval(() => {
      if (document.hidden) { stop(); return; }
      scan = Math.min(model.MAX_SCANS, scan + 1); render();
      if (scan === model.MAX_SCANS) stop();
    }, 550);
  });
  ['show-truth', 'show-sources', 'show-uncertainty'].forEach(id => el(id).addEventListener('change', render));
  const observer = new ResizeObserver(draw);
  const observe = () => observer.observe(el('scene'));
  observe();
  document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
  window.addEventListener('bento-live-visibility', event => { if (event.detail?.paused) stop(); });
  window.addEventListener('pagehide', () => { stop(); observer.disconnect(); });
  window.addEventListener('pageshow', observe);
  window.ExtendedTargetLab = Object.freeze({ getRun: () => structuredClone(run), isPlaying: () => timer !== null });
  render();
})();
