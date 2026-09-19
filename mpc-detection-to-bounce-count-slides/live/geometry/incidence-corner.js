(() => {
  'use strict';
  if (new URLSearchParams(location.search).get('construction') !== 'incidence-points') return;
  document.documentElement.classList.add('incidence-points-construction');
  const model = window.IncidenceCornerModel, drawing = window.IncidenceCornerDrawing;
  const { bs, ue: initialUE, wallA, wallB } = model.scene;
  const panel = document.getElementById('udouble');
  panel.classList.add('incidence-corner');
  panel.querySelector('h3').innerHTML = '<span class="no">3.2</span>Double bounce: infer wall A, then intersect the reflected ray';
  document.querySelector('#unknown-tab-double small').textContent = 'reflect → intersect';
  document.getElementById('udouble-demo').innerHTML = `<div class="body split">
    <div class="corner-stage">
      <svg id="corner-scene" viewBox="0 0 760 340" role="img" tabindex="0" aria-keyshortcuts="ArrowLeft ArrowRight ArrowUp ArrowDown" aria-label="Corner with two bounces. Infer the first wall from a single-bounce path, then reflect and intersect measured rays. Drag the UE or use arrow keys to move it."></svg>
      <div class="corner-steps" role="group" aria-label="Construction steps">
        <button type="button" data-corner-step="1" aria-pressed="true">1 · Infer wall A</button>
        <button type="button" data-corner-step="2" aria-pressed="false">2 · Reflect on A</button>
        <button type="button" data-corner-step="3" aria-pressed="false">3 · Locate P2</button>
      </div>
      <p id="corner-caption" aria-live="polite"></p>
    </div>
    <div class="rail">
      <div class="corner-measurements" role="group" aria-label="Measurement to edit">
        <button type="button" data-corner-measurement="single" aria-pressed="true">Single bounce</button>
        <button type="button" data-corner-measurement="double" aria-pressed="false">Double bounce</button>
      </div>
      <div class="ctl">
        <label for="corner-aod"><span>BS departure angle (AoD)</span><output id="corner-aod-value"></output></label>
        <input id="corner-aod" type="range" min="-35" max="35" step="0.1" value="0">
        <label for="corner-aoa"><span>UE arrival angle (AoA)</span><output id="corner-aoa-value"></output></label>
        <input id="corner-aoa" type="range" min="-35" max="35" step="0.1" value="0">
        <label for="corner-length"><span>Measured length · check only</span><output id="corner-length-value"></output></label>
        <input id="corner-length" class="meas" type="range" min="-3" max="3" step="0.05" value="0">
      </div>
      <button type="button" class="reset" id="corner-reset">Reset both measurements</button>
      <label class="ck"><input type="checkbox" id="corner-reference"> Show reference walls</label>
      <div class="corner-result" aria-live="polite" aria-atomic="true">
        <strong id="corner-status"></strong>
        <div id="corner-position"></div>
        <div id="corner-check"></div>
      </div>
      <div class="hint">The single path identifies the first wall of the double path. Drag UE for fresh data. Delays check lengths only.</div>
    </div>
  </div><div class="figcap"><b>Read it.</b> Intersect the single-bounce AoD and AoA rays at S; the reflection law gives wall A through S. The double-bounce AoD ray hits this inferred wall at P1. Reflect that ray on A and intersect it with the double-bounce UE AoA ray to obtain P2 on wall B. S and P1 generally differ. Both paths must be associated with the same first wall; reference walls are only synthetic ground truth, never estimator inputs.</div>`;
  const $ = name => document.getElementById(`corner-${name}`);
  const zero = () => ({ aod: 0, aoa: 0, length: 0 });
  const state = { ue: [...initialUE], single: zero(), double: zero(), step: 1, measurement: 'single', reference: false };
  const format = value => Math.abs(value) < .005 ? '0.00' : Math.abs(value) >= 10000 ? value.toExponential(2) : value.toFixed(2);
  const angle = value => (((value + 180) % 360 + 360) % 360 - 180).toFixed(1) + '°';
  const coords = p => `(${format(p[0])}, ${format(p[1])}) m`;
  const captions = [
    'Single bounce: the AoD and AoA rays meet at S. Their directions determine the wall normal, fixing wall A.',
    'Double bounce: project the BS AoD ray to wall A at P1, then reflect it using the inferred wall normal.',
    'Project the UE AoA ray. Its forward intersection with the reflected ray is P2 on the second wall.'
  ];
  const reasons = {
    'first-wall-unresolved': 'The single-bounce rays do not determine wall A.',
    'parallel-wall': 'The double-bounce AoD ray is parallel to wall A.',
    'behind-wall': 'The double-bounce AoD ray points away from wall A.',
    'second-parallel': 'The reflected AoD and UE AoA rays are parallel.',
    'second-collinear': 'The reflected and arrival rays have no unique intersection.',
    'second-behind': 'The second intersection is behind a ray origin.',
    invalid: 'Choose finite angle measurements.'
  };
  let camera;
  function render() {
    const reference = model.referenceMeasurements(bs, state.ue);
    const measurements = {};
    for (const path of ['single', 'double']) measurements[path] = Object.fromEntries(['aod', 'aoa', 'length'].map(key => [key, reference[path][key] + state[path][key]]));
    const result = model.construction(bs, state.ue, measurements.single, measurements.double);
    const frame = drawing.render({ bs, ue: state.ue, result, step: state.step, reference: { ...reference, wallA, wallB }, showReference: state.reference });
    camera = frame.camera;
    $('scene').innerHTML = frame.markup;
    $('caption').textContent = captions[state.step - 1];
    for (const button of panel.querySelectorAll('[data-corner-step]')) button.setAttribute('aria-pressed', String(Number(button.dataset.cornerStep) === state.step));
    for (const button of panel.querySelectorAll('[data-corner-measurement]')) button.setAttribute('aria-pressed', String(button.dataset.cornerMeasurement === state.measurement));
    for (const key of ['aod', 'aoa', 'length']) {
      const value = measurements[state.measurement][key];
      const text = key === 'length' ? `${format(value)} m` : angle(value);
      $(key).value = state[state.measurement][key];
      $(key).setAttribute('aria-valuetext', `${state.measurement} bounce: ${text}`);
      $(`${key}-value`).textContent = text;
    }
    const firstValid = result.single.status === 'ok';
    const visibleValid = state.step === 1 ? firstValid : state.step === 2 ? !!result.p1 : result.status === 'ok';
    $('status').textContent = visibleValid ? ['Wall A inferred', 'Departure ray reflected', 'Second incidence point located'][state.step - 1] : 'No forward construction';
    $('position').textContent = visibleValid ? state.step === 1 ? `S: ${coords(result.single.point)}` : state.step === 2 ? `P1: ${coords(result.p1)}` : `P2: ${coords(result.p2)}` : reasons[result.status];
    const checks = [];
    if (firstValid) checks.push(`Single residual: ${format(result.single.residual)} m`);
    if (state.step === 3 && result.status === 'ok') checks.push(`Double residual: ${format(result.residual)} m`);
    const mismatch = firstValid && Math.abs(result.single.residual) > .15 || state.step === 3 && result.status === 'ok' && Math.abs(result.residual) > .15;
    const sensitive = firstValid && result.single.crossingAngle < 2 || state.step === 3 && result.status === 'ok' && result.crossingAngle < 2;
    if (mismatch) checks.push('Delay mismatch.');
    if (sensitive) checks.push('Nearly parallel rays: angle sensitive.');
    $('check').textContent = checks.join(' · ');
    $('status').parentElement.classList.toggle('is-warning', !visibleValid || mismatch || sensitive);
  }
  for (const key of ['aod', 'aoa', 'length']) $(key).addEventListener('input', event => { state[state.measurement][key] = Number(event.target.value); render(); });
  for (const button of panel.querySelectorAll('[data-corner-step]')) button.addEventListener('click', () => {
    state.step = Number(button.dataset.cornerStep); state.measurement = state.step === 1 ? 'single' : 'double'; render();
  });
  for (const button of panel.querySelectorAll('[data-corner-measurement]')) button.addEventListener('click', () => { state.measurement = button.dataset.cornerMeasurement; render(); });
  const resetMeasurements = () => { state.single = zero(); state.double = zero(); };
  $('reset').addEventListener('click', () => { resetMeasurements(); render(); });
  $('reference').addEventListener('change', event => { state.reference = event.target.checked; render(); });
  const clampUE = p => [Math.max(14, Math.min(21, p[0])), Math.max(-21, Math.min(-13, p[1]))];
  function localPoint(event) {
    const point = $('scene').createSVGPoint(); point.x = event.clientX; point.y = event.clientY;
    return point.matrixTransform($('scene').getScreenCTM().inverse());
  }
  let dragging = false;
  $('scene').addEventListener('pointerdown', event => {
    const p = localPoint(event), u = [camera.x + camera.scale * state.ue[0], camera.y - camera.scale * state.ue[1]];
    if (Math.hypot(p.x - u[0], p.y - u[1]) > 24) return;
    dragging = true; $('scene').setPointerCapture(event.pointerId); $('scene').focus(); event.preventDefault();
  });
  $('scene').addEventListener('pointermove', event => {
    if (!dragging) return;
    const p = localPoint(event);
    state.ue = clampUE([(p.x - camera.x) / camera.scale, (camera.y - p.y) / camera.scale]);
    resetMeasurements(); render();
  });
  for (const event of ['pointerup', 'pointercancel', 'lostpointercapture']) $('scene').addEventListener(event, () => { dragging = false; });
  $('scene').addEventListener('keydown', event => {
    const delta = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] }[event.key];
    if (!delta) return;
    state.ue = clampUE(state.ue.map((v, i) => v + delta[i] * (event.shiftKey ? 1 : .25)));
    resetMeasurements(); render(); event.preventDefault();
  });
  render();
})();
