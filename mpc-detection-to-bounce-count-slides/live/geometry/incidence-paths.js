(() => {
  'use strict';
  if (new URLSearchParams(location.search).get('construction') !== 'incidence-points') return;
  const model = window.IncidencePathsModel, drawing = window.IncidencePathsDrawing;
  const format = v => Math.abs(v) < .005 ? '0.00' : Math.abs(v) >= 10000 ? v.toExponential(2) : v.toFixed(2);
  const angle = v => (((v + 180) % 360 + 360) % 360 - 180).toFixed(1) + '°';
  const coords = p => `(${format(p[0])}, ${format(p[1])}) m`;
  const view = (path, hits, arrival, walls) => ({ path, hits, arrival, walls });
  const configs = [
    { key: 'corner3', panel: 'utriple', tab: 'triple', number: '3.3', title: 'Corner ×3: infer two walls, then locate the third incidence point', detail: 'two reflections → P3', names: ['A', 'B', 'C'],
      steps: ['1 · Infer A', '2 · Infer B', '3 · Hit A', '4 · Reflect B', '5 · Locate P3'],
      views: [view(0, 1, true, 1), view(1, 2, true, 2), view(2, 1, false, 2), view(2, 2, false, 2), view(2, 3, true, 3)],
      captions: [
        'Single bounce: intersect its BS AoD and UE AoA rays at S. The reflection law fixes wall A.',
        'Double bounce: reflect its AoD ray on A. Its intersection with the UE AoA ray fixes a point and wall B.',
        'Triple bounce: start with this path’s own AoD. It hits inferred wall A at P1; reflect the ray there.',
        'The reflected ray hits inferred wall B at P2. Reflect it again using the normal of B.',
        'Intersect the twice-reflected ray with the triple-bounce UE AoA ray. This gives P3 and wall C.'
      ], hint: 'The single and double paths infer the same first two walls used by the triple path. Drag UE for fresh data.' },
    { key: 'corridor2', panel: 'ucorridor', tab: 'corridor', number: '3.4', title: 'Corridor ×2: infer the first wall, reflect, then intersect', detail: 'infer R → locate L', names: ['R', 'L'], corridor: true,
      steps: ['1 · Infer R', '2 · Reflect on R', '3 · Locate P2'],
      views: [view(0, 1, true, 1), view(1, 1, false, 1), view(1, 2, true, 2)],
      captions: [
        'The single-bounce AoD and AoA rays meet at S and determine the first corridor wall R.',
        'The double-bounce AoD ray hits inferred R at P1. Reflect it toward the opposite side of the corridor.',
        'Intersect the reflected ray with the UE AoA ray to locate P2 and wall L. Check that R and L are parallel.'
      ], hint: 'Infer R from the associated single path. Reflect before intersecting: the original endpoint rays are parallel.' },
    { key: 'corridor3', panel: 'ucorridor3', tab: 'corridor3', number: '3.5', title: 'Corridor ×3: reconstruct R → L → the same R', detail: 'return to the same R', names: ['R', 'L'], corridor: true, repeated: true,
      steps: ['1 · Infer R', '2 · Infer L', '3 · Hit R', '4 · Reflect L', '5 · Return R'],
      views: [view(0, 1, true, 1), view(1, 2, true, 2), view(2, 1, false, 2), view(2, 2, false, 2), view(2, 3, true, 2)],
      captions: [
        'Use the single-bounce measurement to locate S and infer the first wall R.',
        'Use the double-bounce measurement and inferred R to locate the second wall L. Check their parallelism.',
        'Project the triple-bounce BS AoD ray to R at P1, then reflect it toward L.',
        'The ray reaches L at P2. Reflect it on L; the next hit must return to the original wall R.',
        'Intersect with the triple-bounce UE AoA ray at P3. Verify both incidence on the original R and its reflection law.'
      ], hint: 'R and L come from the shorter paths. A third wall cannot replace the original R.' }
  ];
  const reasons = {
    invalid: 'Invalid angle measurement.', 'parallel-wall': 'The departure ray is parallel to the next inferred wall.',
    'behind-wall': 'The departure ray points away from the next inferred wall.',
    parallel: 'The final reflected and arrival rays are parallel.', collinear: 'The final rays overlap; the last hit is not unique.',
    behind: 'The final intersection is behind a ray origin.'
  };
  function initPath(config) {
    const scene = model.scenes[config.key], panel = document.getElementById(config.panel), total = scene.routes.length;
    panel.classList.add('incidence-paths');
    panel.querySelector('h3').innerHTML = `<span class="no">${config.number}</span>${config.title}`;
    document.querySelector(`#unknown-tab-${config.tab} small`).textContent = config.detail;
    const prefix = `ip-${config.key}`, $ = key => document.getElementById(`${prefix}-${key}`);
    panel.querySelector('.fig').innerHTML = `<div class="body split">
      <div class="ip-stage">
        <svg id="${prefix}-scene" viewBox="0 0 760 340" role="img" tabindex="0" aria-keyshortcuts="ArrowLeft ArrowRight ArrowUp ArrowDown" aria-label="${config.title}. Drag the UE or use arrow keys to move it."></svg>
        <div class="ip-steps" role="group" aria-label="Construction steps">${config.steps.map((label, i) => `<button type="button" data-ip-step="${i}" aria-pressed="${i === 0}">${label}</button>`).join('')}</div>
        <p id="${prefix}-caption" aria-live="polite"></p>
      </div>
      <div class="rail">
        <div class="ip-measurements" role="group" aria-label="Measurement to edit">${['Single', 'Double', 'Triple'].slice(0, total).map((label, i) => `<button type="button" data-ip-measurement="${i}" aria-pressed="${i === 0}">${label} bounce</button>`).join('')}</div>
        <div class="ctl">
          <label for="${prefix}-aod"><span>BS departure angle (AoD)</span><output id="${prefix}-aod-value"></output></label>
          <input id="${prefix}-aod" type="range" min="-35" max="35" step="0.1" value="0">
          <label for="${prefix}-aoa"><span>UE arrival angle (AoA)</span><output id="${prefix}-aoa-value"></output></label>
          <input id="${prefix}-aoa" type="range" min="-35" max="35" step="0.1" value="0">
          <label for="${prefix}-length"><span>Measured length · check only</span><output id="${prefix}-length-value"></output></label>
          <input id="${prefix}-length" class="meas" type="range" min="-3" max="3" step="0.05" value="0">
        </div>
        <button class="reset" type="button" id="${prefix}-reset">Reset all measurements</button>
        <label class="ck"><input type="checkbox" id="${prefix}-reference"> Show reference walls</label>
        <div class="ip-result" aria-live="polite" aria-atomic="true"><strong id="${prefix}-status"></strong><div id="${prefix}-position"></div><div id="${prefix}-check"></div></div>
        <div class="hint">${config.hint} Delays check lengths only.</div>
      </div>
    </div><div class="figcap"><b>Read it.</b> ${config.captions.join(' ')} Each associated path has its own incidence points. Reference walls only generate the synthetic observations; they are not reconstruction inputs.</div>`;
    const zero = () => scene.routes.map(() => ({ aod: 0, aoa: 0, length: 0 }));
    const state = { ue: [...scene.ue], offsets: zero(), step: 0, measurement: 0, reference: false };
    let camera;
    function render() {
      const reference = model.referenceMeasurements(scene, state.ue);
      const measurements = reference.map((m, i) => Object.fromEntries(['aod', 'aoa', 'length'].map(k => [k, m[k] + state.offsets[i][k]])));
      const result = model.reconstruct(scene.bs, state.ue, measurements, config);
      const view = config.views[state.step], current = result.paths[view.path];
      const frame = drawing.render({ bs: scene.bs, ue: state.ue, result, view, names: config.names, maxOrder: total, reference: { walls: scene.walls, measurements: reference }, showReference: state.reference });
      camera = frame.camera; $('scene').innerHTML = frame.markup; $('caption').textContent = config.captions[state.step];
      for (const button of panel.querySelectorAll('[data-ip-step]')) button.setAttribute('aria-pressed', String(Number(button.dataset.ipStep) === state.step));
      for (const button of panel.querySelectorAll('[data-ip-measurement]')) button.setAttribute('aria-pressed', String(Number(button.dataset.ipMeasurement) === state.measurement));
      for (const key of ['aod', 'aoa', 'length']) {
        const value = measurements[state.measurement][key], text = key === 'length' ? `${format(value)} m` : angle(value);
        $(key).value = state.offsets[state.measurement][key]; $(key).setAttribute('aria-valuetext', `${['Single', 'Double', 'Triple'][state.measurement]} bounce: ${text}`); $(`${key}-value`).textContent = text;
      }
      const valid = current && (view.arrival ? current.status === 'ok' : current.hits.length >= view.hits);
      const final = state.step === config.steps.length - 1;
      const mismatch = result.paths.slice(0, view.path + 1).some(p => p.status === 'ok' && Math.abs(p.residual) > .15);
      const sensitive = result.paths.slice(0, view.path + 1).some(p => p.crossingAngle < 2);
      const corridorMismatch = config.corridor && view.walls >= 2 && result.parallelError > 2;
      const repeatedMismatch = final && result.closure && (result.closure.distance > .15 || result.closure.orientation > 2);
      let status = view.path === 0 ? `Wall ${config.names[0]} inferred` : !view.arrival ? `Ray reflected at ${config.names[view.hits - 1]}` : `Wall ${config.names[view.path] || 'R'} located`;
      if (final && config.repeated) status = repeatedMismatch ? 'Return to R rejected' : 'Third hit matches original R';
      if (corridorMismatch) status = 'Parallel-wall condition fails';
      $('status').textContent = valid ? status : 'No forward construction';
      $('position').textContent = valid ? `${view.path === 0 ? 'S' : view.path === 1 && total === 3 ? `Q${view.hits}` : `P${view.hits}`}: ${coords(current.hits[view.hits - 1])}` : `Path ${(result.failedPath ?? view.path) + 1}: ${reasons[result.paths.at(-1)?.status] || 'A shorter path must first determine its wall.'}`;
      const residuals = result.paths.slice(0, view.path + 1).filter(p => p.status === 'ok').map(p => format(p.residual));
      const checks = residuals.length ? [`Length residual${residuals.length > 1 ? 's' : ''} (m): ${residuals.join(' / ')}`] : [];
      if (view.walls >= 2 && result.parallelError !== undefined) checks.push(`Parallel error: ${format(result.parallelError)}°`);
      if (final && result.closure) checks.push(`R gap: ${format(result.closure.distance)} m · normal: ${format(result.closure.orientation)}°`);
      if (mismatch) checks.push('Delay mismatch.'); if (sensitive) checks.push('Nearly parallel rays: angle sensitive.');
      $('check').textContent = checks.join(' · ');
      $('status').parentElement.classList.toggle('is-warning', Boolean(!valid || mismatch || sensitive || corridorMismatch || repeatedMismatch));
    }
    for (const key of ['aod', 'aoa', 'length']) $(key).addEventListener('input', e => { state.offsets[state.measurement][key] = Number(e.target.value); render(); });
    for (const button of panel.querySelectorAll('[data-ip-step]')) button.addEventListener('click', () => { state.step = Number(button.dataset.ipStep); state.measurement = config.views[state.step].path; render(); });
    for (const button of panel.querySelectorAll('[data-ip-measurement]')) button.addEventListener('click', () => { state.measurement = Number(button.dataset.ipMeasurement); render(); });
    $('reset').addEventListener('click', () => { state.offsets = zero(); render(); });
    $('reference').addEventListener('change', e => { state.reference = e.target.checked; render(); });
    const clamp = p => [Math.max(scene.bounds[0], Math.min(scene.bounds[1], p[0])), Math.max(scene.bounds[2], Math.min(scene.bounds[3], p[1]))];
    function localPoint(event) { const p = $('scene').createSVGPoint(); p.x = event.clientX; p.y = event.clientY; return p.matrixTransform($('scene').getScreenCTM().inverse()); }
    let dragging = false;
    $('scene').addEventListener('pointerdown', e => {
      const p = localPoint(e), u = [camera.x + camera.scale * state.ue[0], camera.y - camera.scale * state.ue[1]];
      if (Math.hypot(p.x - u[0], p.y - u[1]) > 24) return;
      dragging = true; $('scene').setPointerCapture(e.pointerId); $('scene').focus(); e.preventDefault();
    });
    $('scene').addEventListener('pointermove', e => {
      if (!dragging) return; const p = localPoint(e);
      state.ue = clamp([(p.x - camera.x) / camera.scale, (camera.y - p.y) / camera.scale]); state.offsets = zero(); render();
    });
    for (const event of ['pointerup', 'pointercancel', 'lostpointercapture']) $('scene').addEventListener(event, () => { dragging = false; });
    $('scene').addEventListener('keydown', e => {
      const delta = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] }[e.key]; if (!delta) return;
      state.ue = clamp(state.ue.map((v, i) => v + delta[i] * (e.shiftKey ? 1 : .25))); state.offsets = zero(); render(); e.preventDefault();
    });
    render();
  }
  configs.forEach(initPath);

  function initAmbiguity() {
    const scene = model.ambiguityScene, panel = document.getElementById('uestimate');
    panel.classList.add('incidence-paths');
    panel.querySelector('h3').innerHTML = '<span class="no">3.6</span>Ambiguity: fixed rays, moving incidence points';
    panel.querySelector('.lede').textContent = 'Two double-bounce MPCs at known UE poses can fit a family of wall pairs. Rotate the candidate first wall, reflect each measured departure ray, and intersect with its arrival ray. The resulting incidence points move while both measured lengths remain unchanged.';
    panel.querySelector('.accuracy').textContent = 'An associated single-bounce measurement locates a point and the normal of wall A. Once A is fixed, each double-bounce measurement determines its two incidence points.';
    document.querySelector('#unknown-tab-estimate small').textContent = 'fixed rays, moving hits';
    const $ = key => document.getElementById(`ip-ambiguity-${key}`);
    panel.querySelector('.fig').innerHTML = `<div class="body split"><div class="ip-stage">
      <svg id="ip-ambiguity-scene" viewBox="0 0 760 340" role="img" aria-label="Fixed measured rays at two known UE poses fit moving incidence points and a family of two-wall maps."></svg>
      <div class="ip-steps" role="group" aria-label="Ambiguity construction steps">
        <button type="button" data-ambiguity-step="1" aria-pressed="false">1 · Measured rays</button>
        <button type="button" data-ambiguity-step="2" aria-pressed="false">2 · Reflect / intersect</button>
        <button type="button" data-ambiguity-step="3" aria-pressed="true">3 · Two-pose check</button>
      </div><p id="ip-ambiguity-caption" aria-live="polite"></p></div>
      <div class="rail">
        <div class="ctl"><label for="ip-ambiguity-rotation"><span>Candidate wall A rotation</span><output id="ip-ambiguity-rotation-value"></output></label>
          <input id="ip-ambiguity-rotation" type="range" min="-18" max="12" step="0.2" value="0"></div>
        <label class="ck"><input type="checkbox" id="ip-ambiguity-prefix"> Add single-bounce measurement</label>
        <label class="ck"><input type="checkbox" id="ip-ambiguity-reference"> Show original walls</label>
        <button class="reset" type="button" id="ip-ambiguity-reset">Reset family</button>
        <div class="ip-fixed-data"><strong>Fixed double-bounce measurements</strong><div id="ip-ambiguity-data"></div></div>
        <div class="ip-result" aria-live="polite" aria-atomic="true"><strong id="ip-ambiguity-status"></strong><div id="ip-ambiguity-check"></div></div>
      </div></div><div class="figcap"><b>Read it.</b> Each double-bounce path starts with its measured BS AoD, reflects on the candidate first wall, and intersects its UE AoA ray. A common-rotation family of wall pairs preserves both poses’ geometric measurements. Add the associated single-bounce measurement to determine wall A from its own two rays and remove this family. No wall map is supplied to that reconstruction.</div>`;
    const measurements = model.ambiguityMeasurements(scene);
    const single = model.referencePath(scene.bs, scene.ues[0], [scene.walls[0]]);
    const baseline = model.ambiguity(scene.bs, scene.ues, measurements, scene.walls[0]);
    const candidate = degrees => model.wall(scene.pivot, [Math.cos(degrees * Math.PI / 180), Math.sin(degrees * Math.PI / 180)]);
    const fitPoints = [];
    for (let a = -18; a <= 12; a += 1) fitPoints.push(...model.ambiguity(scene.bs, scene.ues, measurements, candidate(a)).paths.flatMap(p => p.hits));
    const state = { rotation: 0, prefix: false, reference: false, step: 3 };
    const captions = [
      'Each pose provides a double-bounce AoD, AoA and length. The endpoint rays alone do not identify the first wall.',
      'Choose a candidate wall A. Reflect the BS AoD there and intersect with UE a’s AoA to locate P2 and wall B.',
      'Both known UE poses fit the same rotating wall pair. The points move while the measured rays and lengths stay fixed.'
    ];
    function render() {
      const result = model.ambiguity(scene.bs, scene.ues, measurements, candidate(state.rotation), state.prefix ? single : null);
      const frame = drawing.renderAmbiguity({ bs: scene.bs, ues: scene.ues, result, baseline, step: state.step, fitPoints, showReference: state.reference });
      $('scene').innerHTML = frame.markup;
      $('rotation').disabled = state.prefix; $('rotation').value = state.rotation;
      $('rotation').setAttribute('aria-valuetext', state.prefix ? 'Fixed by the single-bounce measurement' : angle(state.rotation));
      $('rotation-value').textContent = state.prefix ? 'fixed by single path' : angle(state.rotation);
      $('caption').textContent = state.prefix ? 'The single path fixes S and wall A. Reflecting each double-bounce AoD on that wall now determines both paths’ incidence points.' : captions[state.step - 1];
      for (const b of panel.querySelectorAll('[data-ambiguity-step]')) b.setAttribute('aria-pressed', String(Number(b.dataset.ambiguityStep) === state.step));
      $('data').innerHTML = '<table><thead><tr><th>Pose</th><th>AoD</th><th>AoA</th><th>Length</th></tr></thead><tbody>' + measurements.map((m, i) => `<tr><td>UE ${i ? 'b' : 'a'}</td><td>${angle(m.aod)}</td><td>${angle(m.aoa)}</td><td>${format(m.length)} m</td></tr>`).join('') + '</tbody></table>';
      const motion = result.status === 'ok' ? Math.max(...result.paths.flatMap((p, j) => p.hits.map((q, i) => Math.hypot(q[0] - baseline.paths[j].hits[i][0], q[1] - baseline.paths[j].hits[i][1])))) : NaN;
      $('status').textContent = result.status !== 'ok' ? 'No forward construction' : state.prefix ? 'Wall A fixed by single path' : 'Two poses still admit a wall family';
      $('check').textContent = result.status !== 'ok' ? 'The candidate does not fit both forward paths.' : `Largest length residual: ${format(result.maxResidual)} m. Point motion: ${format(motion)} m. Both poses share wall B.`;
      $('status').parentElement.classList.toggle('is-warning', result.status !== 'ok' || !state.prefix);
    }
    $('rotation').addEventListener('input', e => { state.rotation = Number(e.target.value); render(); });
    $('prefix').addEventListener('change', e => { state.prefix = e.target.checked; state.step = 3; render(); });
    $('reference').addEventListener('change', e => { state.reference = e.target.checked; render(); });
    $('reset').addEventListener('click', () => { state.rotation = 0; state.prefix = false; $('prefix').checked = false; render(); });
    for (const b of panel.querySelectorAll('[data-ambiguity-step]')) b.addEventListener('click', () => { state.step = Number(b.dataset.ambiguityStep); render(); });
    render();
  }
  initAmbiguity();
})();
