(() => {
  'use strict';

  const model = window.VariationalEMModel;
  const app = document.querySelector('#app');
  const SIGMA = 0.72;
  const TAU = 2 * Math.PI;
  const defaults = Object.freeze({ separation: 3.2, center: 0.45 });
  const colors = Object.freeze({
    blue: '#39708C',
    rust: '#A95736',
    violet: '#6557A7',
    teal: '#2D7A70',
    ink: '#202B33',
    muted: '#66737D',
    faint: '#8A949B',
    rule: '#D8DEE2',
    paper: '#FBFAF6'
  });

  function tex(strings, ...values) {
    const source = String.raw(strings, ...values);
    return `<span class="math-tex math-inline">\\(${source}\\)</span>`;
  }

  function texBlock(strings, ...values) {
    const source = String.raw(strings, ...values);
    return `<span class="math-tex math-display">\\[${source}\\]</span>`;
  }

  const state = {
    separation: defaults.separation,
    center: defaults.center,
    data: [],
    params: null,
    responsibilities: [],
    phase: 'E',
    iteration: 0,
    history: [],
    lastAction: 'Ready for the first E-step.',
    paused: false
  };

  app.innerHTML = `
    <aside class="control-panel" aria-label="EM experiment controls">
      <p class="eyebrow">Deterministic teaching model</p>
      <h1>Gaussian mixture lab</h1>
      <p class="intro">Two components, fixed ${tex`\sigma=0.72`}. Alternate responsibilities and parameter updates without variance collapse.</p>

      <div class="control-stack">
        <div class="control">
          <div class="control-head">
            <label for="separation">True component separation</label>
            <output id="separation-value" for="separation">3.20</output>
          </div>
          <input id="separation" data-testid="separation-slider" type="range" min="1.2" max="5.0" step="0.1" value="3.2">
          <p class="control-hint">Lower values create more ambiguous assignments.</p>
        </div>

        <div class="control">
          <div class="control-head">
            <label for="center">Initial center bias</label>
            <output id="center-value" for="center">0.45</output>
          </div>
          <input id="center" data-testid="center-slider" type="range" min="-1.2" max="1.2" step="0.05" value="0.45">
          <p class="control-hint">Moves both starting means before iteration.</p>
        </div>

        <div class="button-grid" aria-label="EM steps">
          <button class="primary" data-testid="e-step" type="button">E-step</button>
          <button data-testid="m-step" type="button">M-step</button>
          <button data-testid="converge" type="button">Run to convergence</button>
          <button data-testid="reset" type="button">Reset</button>
        </div>
      </div>

      <div class="step-formulas" aria-label="Current EM equations">
        <section id="e-formula" class="step-formula">
          <span>E-STEP</span>
          ${texBlock`r_{nk}\propto\pi_k\mathcal N(x_n\mid\mu_k,\sigma^2)`}
        </section>
        <section id="m-formula" class="step-formula" hidden>
          <span>M-STEP</span>
          ${texBlock`\mu_k\leftarrow\frac{\sum_n r_{nk}x_n}{\sum_n r_{nk}}`}
        </section>
      </div>

      <p class="keyboard-hint">Page Up / Page Down navigate · Escape returns focus</p>
    </aside>

    <section class="stage-panel" aria-label="EM model state">
      <header class="stage-bar">
        <div>
          <span class="stage-k">NEXT COORDINATE</span>
          <strong id="phase-title">Infer assignments · E-step</strong>
        </div>
        <p class="status"><i></i><span id="status-text">Ready</span></p>
      </header>

      <div class="stage-body">
        <div class="visual-card">
          <canvas id="em-canvas" role="img" aria-label="Gaussian mixture densities, responsibility-colored observations, component means, and log-likelihood history"></canvas>
          <div class="legend" aria-hidden="true">
            <span class="component-one"><i></i>component 1</span>
            <span class="component-two"><i></i>component 2</span>
            <span class="ambiguous"><i></i>ambiguous responsibility</span>
          </div>
        </div>

        <aside class="metric-rail" aria-label="EM metrics">
          <section class="metric-card accent">
            <span class="metric-k">Iteration / phase</span>
            <strong class="metric-v" id="iteration-value">0 · E next</strong>
          </section>
          <section class="metric-card">
            <span class="metric-k">Component parameters</span>
            <div class="parameter-list">
              <span>mean 1 <code id="mean-one">−0.45</code></span>
              <span>mean 2 <code id="mean-two">1.35</code></span>
              <span>weight 1 <code id="weight-one">0.50</code></span>
              <span>weight 2 <code id="weight-two">0.50</code></span>
            </div>
          </section>
          <section class="metric-card">
            <span class="metric-k">Observed log likelihood</span>
            <strong class="metric-v" id="likelihood-value">—</strong>
            <p class="metric-copy" id="likelihood-delta">Initial value</p>
          </section>
          <section class="metric-card">
            <span class="metric-k">Variational gap</span>
            <strong class="metric-v" id="gap-value">—</strong>
            <p class="metric-copy">Closes after an exact E-step.</p>
          </section>
          <section class="metric-card">
            <span class="metric-k">Ambiguous points</span>
            <strong class="metric-v" id="ambiguous-value">—</strong>
            <p class="metric-copy">Largest responsibility below 0.75.</p>
          </section>
          <section class="metric-card action-card">
            <span class="metric-k">Last action</span>
            <p class="action-copy" id="action-copy">Ready for the first E-step.</p>
          </section>
        </aside>
      </div>
    </section>
  `;

  const nodes = {
    separation: app.querySelector('#separation'),
    separationValue: app.querySelector('#separation-value'),
    center: app.querySelector('#center'),
    centerValue: app.querySelector('#center-value'),
    eButton: app.querySelector('[data-testid="e-step"]'),
    mButton: app.querySelector('[data-testid="m-step"]'),
    convergeButton: app.querySelector('[data-testid="converge"]'),
    resetButton: app.querySelector('[data-testid="reset"]'),
    eFormula: app.querySelector('#e-formula'),
    mFormula: app.querySelector('#m-formula'),
    phaseTitle: app.querySelector('#phase-title'),
    statusText: app.querySelector('#status-text'),
    iteration: app.querySelector('#iteration-value'),
    meanOne: app.querySelector('#mean-one'),
    meanTwo: app.querySelector('#mean-two'),
    weightOne: app.querySelector('#weight-one'),
    weightTwo: app.querySelector('#weight-two'),
    likelihood: app.querySelector('#likelihood-value'),
    likelihoodDelta: app.querySelector('#likelihood-delta'),
    gap: app.querySelector('#gap-value'),
    ambiguous: app.querySelector('#ambiguous-value'),
    action: app.querySelector('#action-copy'),
    canvas: app.querySelector('#em-canvas')
  };

  function format(value, digits = 3) {
    if (!Number.isFinite(value)) return '—';
    return value.toFixed(digits).replace('-', '−');
  }

  function resetAlgorithm(reason = 'Reset to the deterministic initial state.') {
    state.data = model.generateDataset({ separation: state.separation, sigma: SIGMA });
    state.params = model.initialParams(state.center);
    state.responsibilities = model.uniformResponsibilities(state.data);
    state.phase = 'E';
    state.iteration = 0;
    state.history = [model.logLikelihood(state.data, state.params, SIGMA)];
    state.lastAction = reason;
    update();
  }

  function runEStep() {
    if (state.phase !== 'E') return;
    state.responsibilities = model.eStep(state.data, state.params, SIGMA);
    state.phase = 'M';
    state.lastAction = 'E-step: responsibilities now equal the exact latent posterior for the current parameters.';
    update();
  }

  function runMStep() {
    if (state.phase !== 'M') return;
    const previousLikelihood = state.history.at(-1);
    state.params = model.mStep(state.data, state.responsibilities);
    state.iteration += 1;
    state.phase = 'E';
    const nextLikelihood = model.logLikelihood(state.data, state.params, SIGMA);
    state.history.push(nextLikelihood);
    state.lastAction = `M-step: weighted statistics raised log likelihood by ${format(nextLikelihood - previousLikelihood, 4)}.`;
    update();
  }

  function runToConvergence() {
    let cycles = 0;
    let change = Infinity;
    if (state.phase === 'M') runMStep();
    while (cycles < 40 && change > 1e-7) {
      const before = state.params;
      const responsibilities = model.eStep(state.data, before, SIGMA);
      const after = model.mStep(state.data, responsibilities);
      change = model.maxParameterChange(before, after);
      state.responsibilities = responsibilities;
      state.params = after;
      state.iteration += 1;
      state.history.push(model.logLikelihood(state.data, state.params, SIGMA));
      cycles += 1;
    }
    state.responsibilities = model.eStep(state.data, state.params, SIGMA);
    state.phase = 'M';
    state.lastAction = `Converged after ${cycles} additional EM cycle${cycles === 1 ? '' : 's'}; maximum parameter change ${change.toExponential(1)}.`;
    update();
  }

  function mixColor(firstResponsibility) {
    const a = [57, 112, 140];
    const b = [169, 87, 54];
    const t = 1 - firstResponsibility;
    const channel = index => Math.round(a[index] * (1 - t) + b[index] * t);
    return `rgb(${channel(0)} ${channel(1)} ${channel(2)})`;
  }

  function drawLine(context, points, stroke, width = 2, dash = []) {
    if (!points.length) return;
    context.save();
    context.strokeStyle = stroke;
    context.lineWidth = width;
    context.setLineDash(dash);
    context.beginPath();
    points.forEach(([x, y], index) => index ? context.lineTo(x, y) : context.moveTo(x, y));
    context.stroke();
    context.restore();
  }

  function drawCanvas() {
    if (state.paused) return;
    const canvas = nodes.canvas;
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const ratio = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.round(rect.width * ratio);
    canvas.height = Math.round(rect.height * ratio);
    const context = canvas.getContext('2d');
    context.scale(ratio, ratio);
    const width = rect.width;
    const height = rect.height;
    context.clearRect(0, 0, width, height);
    context.fillStyle = colors.paper;
    context.fillRect(0, 0, width, height);

    const allX = state.data.map(item => item.x).concat(state.params.means);
    const xMin = Math.min(-3.8, Math.min(...allX) - 0.75);
    const xMax = Math.max(3.8, Math.max(...allX) + 0.75);
    const left = 42;
    const right = width - 18;
    const top = 28;
    const densityBottom = Math.max(190, height * 0.61);
    const axisY = densityBottom;
    const plotWidth = right - left;
    const xPixel = value => left + (value - xMin) / (xMax - xMin) * plotWidth;
    const densityScale = Math.max(100, (densityBottom - top) * 0.78 / 0.56);

    context.strokeStyle = colors.rule;
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(left, axisY + 0.5);
    context.lineTo(right, axisY + 0.5);
    context.stroke();

    const samples = 180;
    const firstCurve = [];
    const secondCurve = [];
    const totalCurve = [];
    for (let index = 0; index <= samples; index += 1) {
      const x = xMin + (xMax - xMin) * index / samples;
      const first = state.params.weights[0] * model.density(x, state.params.means[0], SIGMA);
      const second = state.params.weights[1] * model.density(x, state.params.means[1], SIGMA);
      const px = xPixel(x);
      firstCurve.push([px, axisY - first * densityScale]);
      secondCurve.push([px, axisY - second * densityScale]);
      totalCurve.push([px, axisY - (first + second) * densityScale]);
    }
    drawLine(context, firstCurve, colors.blue, 3);
    drawLine(context, secondCurve, colors.rust, 3);
    drawLine(context, totalCurve, colors.violet, 2, [7, 5]);

    state.params.means.forEach((mean, index) => {
      const px = xPixel(mean);
      context.save();
      context.strokeStyle = index ? colors.rust : colors.blue;
      context.lineWidth = 1.5;
      context.setLineDash([4, 4]);
      context.beginPath();
      context.moveTo(px, top + 8);
      context.lineTo(px, axisY + 2);
      context.stroke();
      context.fillStyle = index ? colors.rust : colors.blue;
      context.font = '700 10px ui-monospace, monospace';
      context.textAlign = 'center';
      context.fillText(`mean ${index + 1}`, px, top);
      context.restore();
    });

    state.data.forEach((item, index) => {
      const radius = 4.5;
      const x = xPixel(item.x);
      const y = axisY + 11 + (index % 4) * 13;
      context.beginPath();
      context.arc(x, y, radius, 0, TAU);
      context.fillStyle = mixColor(state.responsibilities[index][0]);
      context.fill();
      context.lineWidth = 1;
      context.strokeStyle = '#FFFEFB';
      context.stroke();
    });

    context.fillStyle = colors.faint;
    context.font = '700 9px ui-monospace, monospace';
    context.textAlign = 'left';
    context.fillText(format(xMin, 1), left, axisY + 73);
    context.textAlign = 'right';
    context.fillText(format(xMax, 1), right, axisY + 73);
    context.textAlign = 'center';
    context.fillText('observed x', (left + right) / 2, axisY + 73);

    const historyTop = Math.min(height - 96, axisY + 88);
    const historyBottom = height - 24;
    if (historyBottom - historyTop > 28) {
      context.strokeStyle = colors.rule;
      context.beginPath();
      context.moveTo(left, historyBottom + 0.5);
      context.lineTo(right, historyBottom + 0.5);
      context.stroke();
      const minHistory = Math.min(...state.history);
      const maxHistory = Math.max(...state.history);
      const spread = Math.max(1e-6, maxHistory - minHistory);
      const historyPoints = state.history.map((value, index) => [
        left + index / Math.max(1, state.history.length - 1) * plotWidth,
        historyBottom - 8 - (value - minHistory) / spread * Math.max(12, historyBottom - historyTop - 16)
      ]);
      drawLine(context, historyPoints, colors.teal, 2.5);
      historyPoints.forEach(([x, y]) => {
        context.beginPath();
        context.arc(x, y, 2.7, 0, TAU);
        context.fillStyle = colors.teal;
        context.fill();
      });
      context.fillStyle = colors.faint;
      context.font = '800 9px ui-monospace, monospace';
      context.textAlign = 'left';
      context.fillText('OBSERVED LOG LIKELIHOOD BY M-STEP', left, historyTop - 4);
    }
  }

  function update() {
    const likelihood = model.logLikelihood(state.data, state.params, SIGMA);
    const bound = model.elbo(state.data, state.params, SIGMA, state.responsibilities);
    const gap = Math.max(0, likelihood - bound);
    const summary = model.summarizeResponsibilities(state.responsibilities);
    const previous = state.history.length > 1 ? state.history.at(-2) : null;
    const latest = state.history.at(-1);

    nodes.separationValue.textContent = format(state.separation, 2);
    nodes.centerValue.textContent = format(state.center, 2);
    nodes.eButton.disabled = state.phase !== 'E';
    nodes.mButton.disabled = state.phase !== 'M';
    nodes.eFormula.hidden = state.phase !== 'E';
    nodes.mFormula.hidden = state.phase !== 'M';
    nodes.phaseTitle.textContent = state.phase === 'E' ? 'Infer assignments · E-step' : 'Refit parameters · M-step';
    nodes.statusText.textContent = state.paused ? 'Paused while hidden' : (state.phase === 'E' ? 'Parameters fixed' : 'Responsibilities fixed');
    nodes.iteration.textContent = `${state.iteration} · ${state.phase} next`;
    nodes.meanOne.textContent = format(state.params.means[0]);
    nodes.meanTwo.textContent = format(state.params.means[1]);
    nodes.weightOne.textContent = format(state.params.weights[0]);
    nodes.weightTwo.textContent = format(state.params.weights[1]);
    nodes.likelihood.textContent = format(likelihood, 4);
    nodes.likelihoodDelta.textContent = previous === null ? 'Initial value' : `Last M-step: +${format(latest - previous, 4)}`;
    nodes.gap.textContent = gap < 5e-7 ? '0.0000 · tight' : format(gap, 4);
    nodes.ambiguous.textContent = `${summary.ambiguous} / ${state.data.length}`;
    nodes.action.textContent = state.lastAction;
    drawCanvas();
  }

  nodes.separation.addEventListener('input', event => {
    state.separation = Number(event.currentTarget.value);
    resetAlgorithm('Changed component separation; algorithm state reset.');
  });
  nodes.center.addEventListener('input', event => {
    state.center = Number(event.currentTarget.value);
    resetAlgorithm('Changed initialization bias; algorithm state reset.');
  });
  nodes.eButton.addEventListener('click', runEStep);
  nodes.mButton.addEventListener('click', runMStep);
  nodes.convergeButton.addEventListener('click', runToConvergence);
  nodes.resetButton.addEventListener('click', () => {
    state.separation = defaults.separation;
    state.center = defaults.center;
    nodes.separation.value = String(defaults.separation);
    nodes.center.value = String(defaults.center);
    resetAlgorithm();
  });

  window.addEventListener('resize', drawCanvas);
  window.addEventListener('bento-live-visibility', event => {
    state.paused = Boolean(event.detail?.paused);
    update();
  });
  new ResizeObserver(drawCanvas).observe(nodes.canvas);
  resetAlgorithm('Ready for the first E-step.');
})();
