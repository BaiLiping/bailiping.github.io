(() => {
  'use strict';
  if (new URLSearchParams(location.search).get('construction') !== 'incidence-points') return;
  const model = window.IncidenceSingleModel, drawing = window.IncidenceSingleDrawing;
  const { bs, ue: initialUE, wallPoint, wallTangent } = model.scene;
  const panel = document.getElementById('usingle');
  panel.classList.add('incidence-single');
  panel.querySelector('h3').innerHTML = '<span class="no">3.1</span>Single bounce: intersect the AoD and AoA rays';
  document.querySelector('#unknown-tab-single small').textContent = 'ray intersection';
  const demo = document.getElementById('usingle-demo');
  demo.innerHTML = `<div class="body split">
    <svg id="incidence-scene" viewBox="0 0 760 400" role="img" tabindex="0" aria-keyshortcuts="ArrowLeft ArrowRight ArrowUp ArrowDown" aria-label="Single-bounce incidence point from BS AoD and UE AoA rays. Drag the UE or use arrow keys to move it."></svg>
    <div class="rail">
      <h4>Angle measurements · world frame</h4>
      <div class="ctl">
        <label for="incidence-aod"><span>BS departure angle (AoD)</span><output id="incidence-aod-value"></output></label>
        <input id="incidence-aod" type="range" min="-35" max="35" step="0.1" value="0">
        <label for="incidence-aoa"><span>UE arrival angle (AoA)</span><output id="incidence-aoa-value"></output></label>
        <input id="incidence-aoa" type="range" min="-35" max="35" step="0.1" value="0">
        <label for="incidence-length"><span>Measured path length · check only</span><output id="incidence-length-value"></output></label>
        <input id="incidence-length" class="meas" type="range" min="-3" max="3" step="0.05" value="0">
      </div>
      <button class="reset" id="incidence-reset">Reset measurements</button>
      <label class="ck"><input id="incidence-reference" type="checkbox"> Show reference wall</label>
      <div class="incidence-result" aria-live="polite" aria-atomic="true">
        <strong id="incidence-status"></strong>
        <div id="incidence-position"></div>
        <div id="incidence-length-check"></div>
      </div>
      <div class="hint">Drag the UE to sample fresh measurements. Delay checks the path length; it does not move the point.</div>
    </div>
  </div><div class="figcap"><b>Read it.</b> Project the BS ray along AoD and the UE ray toward the arrival direction. Their forward intersection is the incidence point. Under a specular single-bounce hypothesis, the two directions also determine the wall normal. The wall map is not an input; the optional reference wall only illustrates the synthetic scene.</div>`;
  const $ = name => document.getElementById(`incidence-${name}`);
  const state = { ue: [...initialUE], aod: 0, aoa: 0, length: 0, reference: false };
  let camera;
  const format = value => Math.abs(value) < .005 ? '0.00' : Math.abs(value) >= 10000 ? value.toExponential(2) : value.toFixed(2);
  const angle = value => (((value + 180) % 360 + 360) % 360 - 180).toFixed(1) + '°';
  const reasons = {
    parallel: 'Parallel rays do not determine an incidence point.',
    collinear: 'Collinear rays do not determine a unique incidence point.',
    behind: 'The lines meet behind a ray origin; reject the single-bounce candidate.',
    invalid: 'Choose finite angle measurements.'
  };
  function render() {
    const reference = model.referenceMeasurements(bs, state.ue, wallPoint, wallTangent);
    const aod = reference.aod + state.aod, aoa = reference.aoa + state.aoa, length = reference.length + state.length;
    const result = model.construction(bs, state.ue, aod, aoa, length);
    const frame = drawing.render({ bs, ue: state.ue, result, reference: { point: reference.point, tangent: wallTangent }, showReference: state.reference });
    camera = frame.camera;
    $('scene').innerHTML = frame.markup;
    for (const [key, value, text] of [['aod', state.aod, angle(aod)], ['aoa', state.aoa, angle(aoa)], ['length', state.length, format(length) + ' m']]) {
      $(key).value = value;
      $(key).setAttribute('aria-valuetext', text);
      $(`${key}-value`).textContent = text;
    }
    const valid = result.status === 'ok';
    $('status').textContent = valid ? 'Forward-ray intersection' : 'No incidence point';
    $('position').textContent = valid ? `P: x ${format(result.point[0])} m · y ${format(result.point[1])} m` : reasons[result.status];
    $('length-check').textContent = valid ? `Length from rays: ${format(result.length)} m. Residual: ${format(result.residual)} m.` : 'Delay cannot resolve this bearing geometry.';
    if (valid && result.crossingAngle < 2) $('length-check').textContent += ' Nearly parallel: sensitive to angle errors.';
    else if (valid && Math.abs(result.residual) > .15) $('length-check').textContent += ' Delay mismatch.';
    $('status').parentElement.classList.toggle('is-warning', !valid || Math.abs(result.residual) > .15 || result.crossingAngle < 2);
  }
  for (const key of ['aod', 'aoa', 'length']) $(key).addEventListener('input', event => { state[key] = Number(event.target.value); render(); });
  $('reference').addEventListener('change', event => { state.reference = event.target.checked; render(); });
  const resetMeasurements = () => { state.aod = 0; state.aoa = 0; state.length = 0; };
  $('reset').addEventListener('click', () => { resetMeasurements(); render(); });
  const clampUE = p => [Math.max(-4, Math.min(10, p[0])), Math.max(-15, Math.min(-3, p[1]))];
  function localPoint(event) {
    const point = $('scene').createSVGPoint();
    point.x = event.clientX; point.y = event.clientY;
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
