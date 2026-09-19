(function () {
  'use strict';
  const M = window.ASRModel;
  const params = new URLSearchParams(location.search);
  const mode = ['manifold', 'spline', 'gp'].includes(params.get('demo')) ? params.get('demo') : 'manifold';
  const embedded = params.get('embed') === 'region';
  document.body.classList.toggle('embedded', embedded);

  const $ = id => document.getElementById(id);
  const fmt = (value, digits = 3) => Number(value).toFixed(digits);
  const colors = { ink: '#182d33', muted: '#60747a', rule: '#d6dedc', teal: '#16736e', soft: '#e4f0ed', coral: '#c2573f', warm: '#f7e8e1', blue: '#3f6f91', cool: '#e5edf4', amber: '#a87820', panel: '#fffdfa' };
  const metric = (id, label) => `<div class="metric"><span>${label}</span><strong id="${id}">—</strong></div>`;
  const nav = '<div class="mobile-nav"><button type="button" data-nav="-1">← concept</button><button type="button" data-nav="1">next slide →</button></div>';
  const esc = value => String(value).replace(/[&<>"']/g, character => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[character]);
  const line = (x1, y1, x2, y2, stroke = colors.rule, width = 1.5, dash = '') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${stroke}" stroke-width="${width}" ${dash ? `stroke-dasharray="${dash}"` : ''}/>`;
  const text = (x, y, value, size = 11, extra = '') => `<text x="${x}" y="${y}" font-size="${size}" ${extra}>${esc(value)}</text>`;
  const circle = (x, y, radius, fill, stroke = 'white', width = 1.5) => `<circle cx="${x}" cy="${y}" r="${radius}" fill="${fill}" stroke="${stroke}" stroke-width="${width}"/>`;
  const path = (points, map, stroke, width = 2.5, dash = '', fill = 'none', opacity = 1) => {
    if (!points.length) return '';
    const d = points.map((point, i) => `${i ? 'L' : 'M'}${map(point).join(' ')}`).join(' ');
    return `<path d="${d}" fill="${fill}" stroke="${stroke}" stroke-width="${width}" stroke-linejoin="round" stroke-linecap="round" opacity="${opacity}" ${dash ? `stroke-dasharray="${dash}"` : ''}/>`;
  };
  let state = null;
  let step = () => {};

  function setupManifold() {
    $('app').innerHTML = `<div class="lab"><section class="controls"><div class="eyebrow">Live 01 · Retraction</div><h1>Walk on the group.<br>Do not leave it.</h1><p>Apply the same tangent increment repeatedly. The exponential map stays on the rotation manifold; a first-order matrix step drifts away.</p><label class="field" for="rot-delta">Increment <output id="rot-delta-value">12°</output></label><input id="rot-delta" type="range" min="-20" max="20" step="1" value="12"><label class="field" for="rot-steps">Repeated updates <output id="rot-steps-value">6</output></label><input id="rot-steps" type="range" min="1" max="8" step="1" value="6"><div class="buttons"><button type="button" id="rot-step" class="primary">One more</button><button type="button" id="rot-reset">Reset</button></div><div class="param">Start angle: −35°<br>Exact: compose with a rotation<br>Euler: multiply by first-order step</div><p class="hint">The Euler column length exposes loss of orthonormality. This is an integration illustration, not an optimizer benchmark.</p>${nav}</section><section class="stage"><div class="chart"><h2>Exact retraction versus first-order matrix accumulation</h2><svg id="rot-plot" role="img" aria-label="Exact and Euler rotation updates compared on two coordinate frames"></svg></div><div class="bottom"><div class="metrics">${metric('rot-exact-det','exact determinant')}${metric('rot-euler-det','Euler determinant')}${metric('rot-exact-orth','exact orthogonality')}${metric('rot-euler-orth','Euler orthogonality')}</div><div class="message" id="rot-status" aria-live="polite"></div></div></section></div>`;
    function draw() {
      state = M.rotationExperiment(Number($('rot-delta').value), Number($('rot-steps').value), -35);
      const panel = (cx, title, matrix, points, exact) => {
        const scale = 83;
        let svg = `<rect x="${cx - 156}" y="18" width="312" height="250" rx="11" fill="${exact ? colors.soft : colors.warm}" stroke="${exact ? colors.teal : colors.coral}"/>`;
        svg += text(cx, 41, title, 12, 'font-weight="800" text-anchor="middle"');
        svg += circle(cx, 147, scale, 'none', colors.rule, 1.2) + line(cx - 108, 147, cx + 108, 147) + line(cx, 39, cx, 255);
        if (exact) {
          for (const angle of points) svg += circle(cx + Math.cos(angle) * scale, 147 - Math.sin(angle) * scale, 3.2, colors.teal, colors.panel, 1);
        } else {
          for (const point of points) svg += circle(cx + point[0] * scale, 147 - point[1] * scale, 3.2, colors.coral, colors.panel, 1);
        }
        const ex = [matrix[0][0], matrix[1][0]], ey = [matrix[0][1], matrix[1][1]];
        svg += line(cx, 147, cx + ex[0] * scale, 147 - ex[1] * scale, exact ? colors.teal : colors.coral, 4);
        svg += line(cx, 147, cx + ey[0] * scale, 147 - ey[1] * scale, colors.blue, 4);
        svg += text(cx + ex[0] * scale + 7, 147 - ex[1] * scale - 5, 'x-axis', 10);
        svg += text(cx + ey[0] * scale + 7, 147 - ey[1] * scale - 5, 'y-axis', 10);
        return svg;
      };
      const svg = panel(190, 'Exponential-map update', state.exact, state.exactPath, true) + panel(570, 'First-order Euler update', state.euler, state.eulerPath, false);
      $('rot-plot').setAttribute('viewBox', '0 0 760 282');
      $('rot-plot').innerHTML = svg;
      $('rot-delta-value').value = `${state.incrementDeg}°`;
      $('rot-steps-value').value = state.steps;
      $('rot-exact-det').textContent = fmt(state.exactDeterminant, 4);
      $('rot-euler-det').textContent = fmt(state.eulerDeterminant, 4);
      $('rot-exact-orth').textContent = state.exactOrthogonality.toExponential(1);
      $('rot-euler-orth').textContent = fmt(state.eulerOrthogonality, 3);
      const severe = state.eulerOrthogonality > 0.5;
      $('rot-status').classList.toggle('warning', severe);
      $('rot-status').innerHTML = `<div class="legend"><span><i style="background:${colors.teal}"></i>first column</span><span><i style="background:${colors.blue}"></i>second column</span></div>${severe ? '<b>The shortcut has visibly left the rotation group.</b> ' : ''}Composition through the exponential map keeps determinant one and orthogonal columns after every update.`;
    }
    step = () => { $('rot-steps').value = Math.min(8, Number($('rot-steps').value) + 1); draw(); };
    $('rot-delta').addEventListener('input', draw);
    $('rot-steps').addEventListener('input', draw);
    $('rot-step').addEventListener('click', step);
    $('rot-reset').addEventListener('click', () => { $('rot-delta').value = 12; $('rot-steps').value = 6; draw(); });
    draw();
  }

  function setupSpline() {
    $('app').innerHTML = `<div class="lab"><section class="controls"><div class="eyebrow">Live 02 · Local support</div><h1>Move one control.<br>Change one neighborhood.</h1><p>A compact-support spline uses only a few coefficients at the query time. Move one control point and watch the rest of the trajectory stay put.</p><label class="field" for="spline-degree">Basis</label><select id="spline-degree"><option value="3">Cubic · at most four active</option><option value="1">Linear · at most two active</option></select><label class="field" for="spline-query">Query time <output id="spline-query-value">0.54</output></label><input id="spline-query" type="range" min="0" max="1" step="0.01" value="0.54"><label class="field" for="spline-control">Control point <output id="spline-control-value">c5</output></label><input id="spline-control" type="range" min="0" max="10" step="1" value="5"><label class="field" for="spline-shift">Vertical move <output id="spline-shift-value">+0.18</output></label><input id="spline-shift" type="range" min="-0.35" max="0.35" step="0.01" value="0.18"><div class="buttons"><button type="button" id="spline-reset">Reset</button></div><p class="hint">The dashed curve is the unchanged reference. Bars show the basis weights at the query.</p>${nav}</section><section class="stage"><div class="chart"><h2>Trajectory, control polygon, and active basis weights</h2><svg id="spline-plot" role="img" aria-label="Spline curve with control points, local deformation, query point, and basis weights"></svg></div><div class="bottom"><div class="metrics">${metric('spline-active','active coefficients')}${metric('spline-weight','weights sum')}${metric('spline-speed','query speed')}${metric('spline-span','changed interval')}</div><div class="message" id="spline-status" aria-live="polite"></div></div></section></div>`;
    function draw() {
      state = M.splineExperiment({ degree: Number($('spline-degree').value), query: Number($('spline-query').value), selected: Number($('spline-control').value), shift: Number($('spline-shift').value) });
      const map = point => [44 + point[0] * 500, 262 - point[1] * 222];
      let svg = `<rect x="25" y="19" width="540" height="252" rx="10" fill="${colors.panel}" stroke="${colors.rule}"/>`;
      for (let i = 0; i <= 5; i += 1) svg += line(44 + i * 100, 40, 44 + i * 100, 262, colors.rule, 1, '3 5');
      svg += path(state.reference, map, colors.muted, 2, '5 5', 'none', 0.7);
      svg += path(state.points, map, colors.rule, 1.4, '4 4');
      svg += path(state.curve, map, colors.teal, 4);
      state.points.forEach((point, i) => {
        const [x, y] = map(point), active = state.active.some(item => item.i === i), selected = i === state.selected;
        svg += circle(x, y, selected ? 8 : 6, selected ? colors.coral : active ? colors.amber : colors.panel, selected ? colors.coral : active ? colors.amber : colors.muted, 2);
        svg += text(x, y - 12, `c${i}`, 10, 'text-anchor="middle"');
      });
      const [qx, qy] = map(state.queryPoint);
      svg += line(qx, 36, qx, 263, colors.coral, 1.4, '4 4') + circle(qx, qy, 7, colors.coral, colors.panel, 2) + text(qx + 9, qy - 10, 'query', 10);
      svg += `<rect x="586" y="19" width="151" height="252" rx="10" fill="${colors.soft}" stroke="${colors.rule}"/>` + text(604, 43, 'Basis weights now', 11, 'font-weight="800"');
      state.weights.forEach((weight, i) => {
        const y = 54 + i * 19;
        svg += text(604, y + 10, `c${i}`, 9) + `<rect x="628" y="${y}" width="90" height="13" rx="4" fill="${colors.panel}"/>` + `<rect x="628" y="${y}" width="${90 * weight}" height="13" rx="4" fill="${weight > 1e-8 ? colors.teal : colors.rule}"/>` + text(722, y + 10, weight.toFixed(2), 8.5, 'text-anchor="end"');
      });
      $('spline-plot').setAttribute('viewBox', '0 0 760 282');
      $('spline-plot').innerHTML = svg;
      $('spline-query-value').value = state.query.toFixed(2);
      $('spline-control-value').value = `c${state.selected}`;
      $('spline-shift-value').value = `${state.shift >= 0 ? '+' : ''}${state.shift.toFixed(2)}`;
      $('spline-active').textContent = `${state.active.length} / ${state.points.length}`;
      $('spline-weight').textContent = state.weights.reduce((a, b) => a + b, 0).toFixed(3);
      $('spline-speed').textContent = state.speed.toFixed(2);
      $('spline-span').textContent = state.influenceStart === null ? 'none' : `${state.influenceStart.toFixed(2)}–${state.influenceEnd.toFixed(2)}`;
      $('spline-status').innerHTML = `<div class="legend"><span><i style="background:${colors.teal}"></i>edited curve</span><span><i style="background:${colors.muted}"></i>reference</span><span><i style="background:${colors.amber}"></i>active control</span></div><b>${state.active.length} coefficients</b> determine the queried pose. Compact support keeps both evaluation and factor connectivity local.`;
    }
    step = () => { $('spline-query').value = Math.min(1, Number($('spline-query').value) + 0.08); draw(); };
    ['spline-degree', 'spline-query', 'spline-control', 'spline-shift'].forEach(id => $(id).addEventListener(id === 'spline-degree' ? 'change' : 'input', draw));
    $('spline-reset').addEventListener('click', () => { $('spline-degree').value = 3; $('spline-query').value = 0.54; $('spline-control').value = 5; $('spline-shift').value = 0.18; draw(); });
    draw();
  }

  function setupGP() {
    $('app').innerHTML = `<div class="lab"><section class="controls"><div class="eyebrow">Live 03 · Sparse GP</div><h1>Fit time.<br>Keep it local.</h1><p>A scalar random-walk GP connects neighboring control states. Asynchronous measurements interpolate between them; the information matrix stays banded.</p><label class="field" for="gp-query">Query time <output id="gp-query-value">3.65 s</output></label><input id="gp-query" type="range" min="0" max="7" step="0.05" value="3.65"><label class="field" for="gp-process">Process variance <output id="gp-process-value">0.18</output></label><input id="gp-process" type="range" min="0.02" max="1.2" step="0.02" value="0.18"><label class="field" for="gp-noise">Measurement noise <output id="gp-noise-value">0.22</output></label><input id="gp-noise" type="range" min="0.06" max="0.9" step="0.02" value="0.22"><div class="buttons"><button type="button" id="gp-reset">Reset</button></div><div class="param">8 control states · 8 measurements<br>measurement times differ from control times<br>query time chosen independently</div><p class="hint">This scalar random-walk model isolates sparsity and regularization. A pose-valued GP uses local Lie-algebra coordinates.</p>${nav}</section><section class="stage"><div class="chart"><h2>Posterior trajectory and sparse information pattern</h2><svg id="gp-plot" role="img" aria-label="Gaussian-process trajectory estimate, uncertainty band, asynchronous measurements, and sparse information matrix"></svg></div><div class="bottom"><div class="metrics">${metric('gp-estimate','query estimate')}${metric('gp-sigma','query uncertainty')}${metric('gp-rough','path roughness')}${metric('gp-nnz','nonzero entries')}</div><div class="message" id="gp-status" aria-live="polite"></div></div></section></div>`;
    function draw() {
      state = M.gpExperiment({ query: Number($('gp-query').value), processVariance: Number($('gp-process').value), measurementSigma: Number($('gp-noise').value) });
      const plotMap = point => [44 + point[0] / 7 * 500, 258 - point[1] / 3.2 * 208];
      const upper = state.curve.map(item => [item.t, item.mean + 2 * item.sigma]);
      const lower = state.curve.slice().reverse().map(item => [item.t, item.mean - 2 * item.sigma]);
      let svg = `<rect x="25" y="19" width="540" height="252" rx="10" fill="${colors.panel}" stroke="${colors.rule}"/>`;
      for (let i = 0; i < 8; i += 1) svg += line(44 + i / 7 * 500, 42, 44 + i / 7 * 500, 258, colors.rule, 1, '3 5') + text(44 + i / 7 * 500, 276, `${i}s`, 9, 'text-anchor="middle"');
      const bandPoints = [...upper, ...lower];
      svg += path(bandPoints, plotMap, 'none', 0, '', colors.cool, 0.95);
      svg += path(state.curve.map(item => [item.t, item.mean]), plotMap, colors.teal, 3.5);
      state.mean.forEach((value, i) => { const [x, y] = plotMap([i, value]); svg += circle(x, y, 5, colors.panel, colors.teal, 2); });
      state.measurements.forEach(item => { const [x, y] = plotMap([item.t, item.z]); svg += `<rect x="${x - 4}" y="${y - 4}" width="8" height="8" fill="${colors.coral}" transform="rotate(45 ${x} ${y})"/>`; });
      const [qx, qy] = plotMap([state.query, state.queryMean]);
      svg += line(qx, 36, qx, 259, colors.coral, 1.4, '4 4') + circle(qx, qy, 7, colors.coral, colors.panel, 2);
      svg += `<rect x="586" y="19" width="151" height="252" rx="10" fill="${colors.soft}" stroke="${colors.rule}"/>` + text(604, 43, 'Information matrix', 11, 'font-weight="800"');
      const cell = 15, x0 = 601, y0 = 59, max = Math.max(...state.information.flat().map(Math.abs));
      for (let i = 0; i < state.count; i += 1) for (let j = 0; j < state.count; j += 1) {
        const value = Math.abs(state.information[i][j]);
        svg += `<rect x="${x0 + j * cell}" y="${y0 + i * cell}" width="12" height="12" rx="2" fill="${value < 1e-10 ? colors.panel : colors.teal}" opacity="${value < 1e-10 ? 1 : 0.25 + 0.75 * value / max}"/>`;
      }
      svg += text(604, 198, 'Only the diagonal', 10) + text(604, 213, 'and neighbors fill.', 10) + text(604, 240, 'motion factors:', 9, 'fill="#60747a"') + text(604, 255, 'x0—x1—…—x7', 10, 'font-weight="700"');
      $('gp-plot').setAttribute('viewBox', '0 0 760 282');
      $('gp-plot').innerHTML = svg;
      $('gp-query-value').value = `${state.query.toFixed(2)} s`;
      $('gp-process-value').value = state.processVariance.toFixed(2);
      $('gp-noise-value').value = state.measurementSigma.toFixed(2);
      $('gp-estimate').textContent = state.queryMean.toFixed(3);
      $('gp-sigma').textContent = `±${(2 * state.querySigma).toFixed(3)}`;
      $('gp-rough').textContent = state.roughness.toFixed(3);
      $('gp-nnz').textContent = `${state.nonzeros} / 64`;
      $('gp-status').innerHTML = `<div class="legend"><span><i style="background:${colors.teal}"></i>posterior mean</span><span><i style="background:${colors.blue}"></i>≈95% interpolation band</span><span><i style="background:${colors.coral}"></i>measurements</span></div>The Markov prior produces a <b>tridiagonal information pattern</b>. Change process variance to trade smoothness against data fit without making the graph dense. The band is for deterministic interpolation of uncertain controls, not the full continuous-time GP.`;
    }
    step = () => { $('gp-query').value = Math.min(7, Number($('gp-query').value) + 0.5); draw(); };
    ['gp-query', 'gp-process', 'gp-noise'].forEach(id => $(id).addEventListener('input', draw));
    $('gp-reset').addEventListener('click', () => { $('gp-query').value = 3.65; $('gp-process').value = 0.18; $('gp-noise').value = 0.22; draw(); });
    draw();
  }

  try {
    if (!M) throw new Error('Teaching model unavailable');
    ({ manifold: setupManifold, spline: setupSpline, gp: setupGP })[mode]();
  } catch (error) {
    $('app').innerHTML = '<p class="error">The experiment could not initialize. Please reload the page.</p>';
    console.error(error);
  }
  document.querySelectorAll('[data-nav]').forEach(button => button.addEventListener('click', () => parent.postMessage({ type: 'bento-inline-nav', direction: Number(button.dataset.nav) }, '*')));
  window.ASRLab = { mode, getState: () => state, step: () => step() };
})();
