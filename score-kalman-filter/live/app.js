import { DEFAULTS, quarticParameters, referenceQuadrature, referenceDensity, score, fitScore, steinClosure, gaussianObservationUpdate } from './model.mjs';

const $ = id => document.getElementById(id);
const colors = { green: '#146b56', orange: '#b55727', blue: '#3b6fa0', axis: '#acb9af', text: '#566961', grid: '#e8ede6' };
const sub = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈'];
const params = new URLSearchParams(location.search);
const state = { ...DEFAULTS, lab: ['score', 'closure', 'update'].includes(params.get('lab')) ? params.get('lab') : 'score', applied: true };
let snapshot;
let pending = false;

function number(value, digits = 3) {
  if (Math.abs(value) < 5e-12) return '0';
  const magnitude = Math.abs(value);
  if (magnitude >= 1e4 || magnitude < 1e-3) return value.toExponential(2).replace('-', '−');
  return value.toFixed(digits).replace('-', '−');
}
function scientific(value) { return value === 0 ? '0' : value.toExponential(1).replace('-', '−'); }
function legend(items) { return items.map(([label, style = '']) => `<span><i class="${style}" aria-hidden="true"></i>${label}</span>`).join(''); }
function metric(label, value) { return `<div><span>${label}</span><strong>${value}</strong></div>`; }

function compute() {
  const trueLambda = quarticParameters(state);
  const prior = referenceQuadrature(trueLambda);
  const fit = fitScore(prior.moments.slice(0, 7));
  const fitted = referenceQuadrature(fit.lambda);
  const closure = steinClosure(prior.moments.slice(0, 7), fit.lambda);
  const posteriorLambda = gaussianObservationUpdate(fit.lambda, state.y, state.R);
  const posterior = referenceQuadrature(posteriorLambda);
  snapshot = { trueLambda, prior, fit, fitted, closure, posteriorLambda, posterior };
}

function updateControls() {
  for (const key of ['a', 'b', 'c', 'y', 'R']) {
    $(key).value = state[key];
    $(`${key}-value`).textContent = number(state[key], 2);
  }
  $('measurement-controls').hidden = state.lab !== 'update';
  $('apply-update').textContent = state.applied ? 'Show prior only' : 'Apply one observation';
  document.documentElement.classList.toggle('lab-update', state.lab === 'update');
  $('shape-description').textContent = state.b < 0
    ? 'Negative b allows two wells. The tilt changes their relative mass.'
    : 'For b ≥ 0, this quartic prior has one mode. It still need not be Gaussian.';
  document.querySelectorAll('[data-lab]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.lab === state.lab)));
  $('expand').href = `./?lab=${state.lab}`;
  $('input-moments').innerHTML = snapshot.prior.moments.slice(0, 7).map((value, k) => `<span>m${sub[k]} <b>${number(value)}</b></span>`).join('');
}

function renderScore() {
  const { trueLambda, fit } = snapshot;
  const error = Math.max(...fit.lambda.map((value, i) => Math.abs(value - trueLambda[i])));
  $('view-title').textContent = 'Fit four coefficients from seven moments';
  $('equation-link').textContent = 'Eqs. (8)–(9) ↗';
  $('equation-link').href = 'https://arxiv.org/html/2605.16644v1#S3.E8';
  $('density-title').textContent = 'Density · reference and fitted family';
  $('second-title').textContent = 'Score · derivative of log density';
  $('density-legend').innerHTML = legend([['Reference'], ['Fitted family', 'dashed']]);
  $('second-legend').innerHTML = legend([['Reference'], ['Fitted score', 'dashed']]);
  $('second-plot').setAttribute('aria-label', 'Reference score and fitted score. They agree for these consistent synthetic moments.');
  $('inspector').innerHTML = `
    <h3>Build A and d, then solve Aλ = d</h3>
    <div class="formula">A<sub>jk</sub> = jk m<sub>j+k−2</sub><br>d<sub>j</sub> = j(j−1) m<sub>j−2</sub></div>
    <p class="note">j,k = 1,…,4 and d₁ = 0. The right-hand side is named d here to distinguish it from the prior’s b.</p>
    <div class="table-scroll"><table class="matrix" aria-label="Score matching matrix A and right-hand side d"><thead><tr><th colspan="4">A · numeric entries</th><th class="rhs">d</th></tr></thead><tbody>
    ${fit.A.map((row, i) => `<tr>${row.map(value => `<td>${number(value, 2)}</td>`).join('')}<td class="rhs">${number(fit.rhs[i], 2)}</td></tr>`).join('')}
    </tbody></table></div>
    <p class="note">Fitted λ = [λ₁, λ₂, λ₃, λ₄]</p>
    <div class="parameters">${fit.lambda.map((value, k) => `<div><span>λ${sub[k + 1]}</span><strong>${number(value, 3)}</strong></div>`).join('')}</div>
    <div class="metrics">${metric('max |Aλ − d|', scientific(fit.maxResidual))}${metric('max coefficient error', scientific(error))}</div>
    <p class="note">Reference λ = [−c, b, 0, a]. Consistent moments recover these parameters. The fit uses a scaled pivoted linear solve, with no normalization integral.</p>`;
}

function renderClosure() {
  const { prior, fit, closure } = snapshot;
  $('view-title').textContent = 'Recover two higher moments with Stein closure';
  $('equation-link').textContent = 'Eq. (12) ↗';
  $('equation-link').href = 'https://arxiv.org/html/2605.16644v1#S4.E12';
  $('density-title').textContent = 'The same fitted quartic family';
  $('second-title').textContent = 'Withheld moments · closure vs. reference';
  $('density-legend').innerHTML = legend([['Reference'], ['Fitted family', 'dashed']]);
  $('second-legend').innerHTML = legend([['Quadrature reference'], ['Stein closure', 'orange']]);
  $('second-plot').setAttribute('aria-label', 'Bar comparison of moments m7 and m8 from the Stein recurrence with the independently computed quadrature reference.');
  const maxError = Math.max(...[7, 8].map(k => Math.abs(closure.moments[k] - prior.moments[k])));
  $('inspector').innerHTML = `
    <h3>Known: m₀,…,m₆ and fitted λ</h3>
    <p class="note">m₇ and m₈ are withheld from the recurrence. They appear only in the reference comparison.</p>
    <div class="formula">λ₁m<sub>β</sub> + 2λ₂m<sub>β+1</sub> + 3λ₃m<sub>β+2</sub><br>+ 4λ₄m<sub>β+3</sub> = βm<sub>β−1</sub></div>
    <div class="step"><div class="step-title">1 · Set β = 4 to get m₇</div><div class="formula">m₇ = (4m₃ − λ₁m₄ − 2λ₂m₅<br>− 3λ₃m₆) / (4λ₄)<br><b>= ${number(closure.moments[7], 5)}</b></div></div>
    <div class="step"><div class="step-title">2 · Set β = 5 to get m₈</div><div class="formula">m₈ = (5m₄ − λ₁m₅ − 2λ₂m₆<br>− 3λ₃m₇) / (4λ₄)<br><b>= ${number(closure.moments[8], 5)}</b></div></div>
    <table aria-label="Stein closure compared with reference moments"><thead><tr><th>Moment</th><th>Reference</th><th>Closure</th></tr></thead><tbody>${[7, 8].map(k => `<tr><td>m${sub[k]}</td><td>${number(prior.moments[k], 5)}</td><td>${number(closure.moments[k], 5)}</td></tr>`).join('')}</tbody></table>
    <div class="metrics">${metric('max absolute error', scientific(maxError))}${metric('4λ₄ · denominator', number(4 * fit.lambda[3]))}</div>
    <p class="note">The second step uses the computed m₇. This quartic example has λ₃ ≈ 0, so that term vanishes. The general recurrence retains it. No Gaussian moment formula is substituted.</p>`;
}

function renderUpdate() {
  const { fit, prior, posteriorLambda, posterior } = snapshot;
  const shown = state.applied ? posterior : prior;
  const shownLambda = state.applied ? posteriorLambda : fit.lambda;
  const likelihood = [-state.y / state.R, 1 / (2 * state.R), 0, 0];
  const variance = shown.moments[2] - shown.moments[1] ** 2;
  $('view-title').textContent = state.applied ? 'Add one likelihood score to the prior' : 'Prior only · apply an observation to compare';
  $('equation-link').textContent = 'Eqs. (13)–(14) ↗';
  $('equation-link').href = 'https://arxiv.org/html/2605.16644v1#S5.E13';
  $('density-title').textContent = state.applied ? 'Density · prior and one-update posterior' : 'Density · prior before observation';
  $('second-title').textContent = state.applied ? 'Score · prior + likelihood = posterior' : 'Score · prior and candidate likelihood';
  $('density-legend').innerHTML = legend(state.applied ? [['Prior'], ['Posterior', 'orange']] : [['Prior']]);
  $('second-legend').innerHTML = legend(state.applied ? [['Prior'], ['Likelihood', 'blue'], ['Posterior', 'orange']] : [['Prior'], ['Likelihood', 'blue']]);
  $('second-plot').setAttribute('aria-label', 'Score curves: the posterior score equals the prior score plus the Gaussian likelihood score.');
  $('inspector').innerHTML = `
    <h3>One measurement factor · v ∼ N(0,R)</h3>
    <div class="formula">s⁺(x) = s⁻(x) + (y − x)/R<br>λ₁⁺ = λ₁⁻ − y/R<br>λ₂⁺ = λ₂⁻ + 1/(2R)</div>
    <table aria-label="Prior, likelihood contribution and posterior coefficients"><thead><tr><th>λ</th><th>Prior</th><th>Δ likelihood</th><th>${state.applied ? 'Posterior' : 'Prior only'}</th></tr></thead><tbody>
      ${fit.lambda.map((value, k) => `<tr class="${k > 1 ? 'unchanged' : ''}"><td>λ${sub[k + 1]}</td><td>${number(value)}</td><td>${number(state.applied ? likelihood[k] : 0)}</td><td>${number(shownLambda[k])}</td></tr>`).join('')}
    </tbody></table>
    <div class="metrics">${metric(`${state.applied ? 'Posterior' : 'Prior'} mean · reference`, number(shown.moments[1], 4))}${metric(`${state.applied ? 'Posterior' : 'Prior'} variance · reference`, number(variance, 4))}</div>
    <p><b>Likelihood factors applied: ${state.applied ? '1' : '0'}</b></p>
    <p class="note">y = ${number(state.y, 2)}, R = ${number(state.R, 2)}. The coefficient update is analytic. These density plots, mean, and variance use quadrature as a reference. They do not implement the paper’s truncated posterior moment recovery in Eq. (15).</p>`;
}

function axisNumber(value) {
  if (Math.abs(value) < 1e-9) return '0';
  if (Math.abs(value) >= 1000) return value.toExponential(0);
  if (Math.abs(value) < .1) return value.toFixed(2);
  return Number(value.toPrecision(2)).toString().replace('-', '−');
}

function setupCanvas(canvas) {
  const bounds = canvas.getBoundingClientRect();
  const width = Math.max(1, bounds.width);
  const height = Math.max(1, bounds.height);
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.font = '11px Arial, sans-serif';
  return { ctx, width, height };
}

function plot(canvas, curves, xRange, options = {}) {
  const { ctx, width, height } = setupCanvas(canvas);
  if (width < 30 || height < 30) return;
  const pad = { left: 38, right: 13, top: 9, bottom: 23 };
  const graphWidth = width - pad.left - pad.right;
  const graphHeight = height - pad.top - pad.bottom;
  const samples = 320;
  const series = curves.map(curve => Array.from({ length: samples + 1 }, (_, i) => {
    const x = xRange[0] + (xRange[1] - xRange[0]) * i / samples;
    return [x, curve.fn(x)];
  }));
  const values = series.flatMap(points => points.map(point => point[1]));
  let yMin = options.nonnegative ? 0 : Math.min(0, ...values);
  let yMax = Math.max(0, ...values);
  const span = yMax - yMin || 1;
  if (!options.nonnegative) yMin -= .06 * span;
  yMax += .08 * span;
  const px = x => pad.left + graphWidth * (x - xRange[0]) / (xRange[1] - xRange[0]);
  const py = y => pad.top + graphHeight * (yMax - y) / (yMax - yMin);
  ctx.lineWidth = 1;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let i = 0; i <= 3; i += 1) {
    const y = yMin + (yMax - yMin) * i / 3;
    ctx.strokeStyle = colors.grid;
    ctx.beginPath(); ctx.moveTo(pad.left, py(y)); ctx.lineTo(width - pad.right, py(y)); ctx.stroke();
    ctx.fillStyle = colors.text;
    ctx.fillText(axisNumber(y), pad.left - 5, py(y));
  }
  ctx.strokeStyle = colors.axis;
  ctx.beginPath(); ctx.moveTo(pad.left, pad.top); ctx.lineTo(pad.left, height - pad.bottom); ctx.lineTo(width - pad.right, height - pad.bottom); ctx.stroke();
  if (yMin < 0 && yMax > 0) {
    ctx.beginPath(); ctx.moveTo(pad.left, py(0)); ctx.lineTo(width - pad.right, py(0)); ctx.stroke();
  }
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let i = 0; i <= 4; i += 1) {
    const x = xRange[0] + (xRange[1] - xRange[0]) * i / 4;
    ctx.fillText(axisNumber(x), px(x), height - pad.bottom + 5);
  }
  ctx.textAlign = 'right';
  ctx.fillText('x', width - 1, height - pad.bottom + 5);

  ctx.save();
  ctx.beginPath(); ctx.rect(pad.left, pad.top, graphWidth, graphHeight); ctx.clip();
  curves.forEach((curve, k) => {
    const points = series[k];
    if (curve.fill) {
      ctx.beginPath(); ctx.moveTo(px(points[0][0]), py(0));
      points.forEach(([x, y]) => ctx.lineTo(px(x), py(y)));
      ctx.lineTo(px(points.at(-1)[0]), py(0)); ctx.closePath();
      ctx.fillStyle = curve.fill; ctx.fill();
    }
    ctx.strokeStyle = curve.color;
    ctx.lineWidth = curve.dash ? 2 : 2.5;
    ctx.setLineDash(curve.dash || []);
    ctx.beginPath(); points.forEach(([x, y], i) => i ? ctx.lineTo(px(x), py(y)) : ctx.moveTo(px(x), py(y))); ctx.stroke();
  });
  ctx.setLineDash([]);
  if (options.observation !== undefined) {
    const x = px(options.observation);
    ctx.strokeStyle = colors.blue;
    ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, height - pad.bottom); ctx.stroke();
    ctx.setLineDash([]); ctx.fillStyle = colors.blue; ctx.textAlign = 'left'; ctx.fillText('y', x + 4, pad.top + 1);
  }
  ctx.restore();
}

function plotClosure() {
  const { ctx, width, height } = setupCanvas($('second-plot'));
  const { prior, closure } = snapshot;
  const pad = { left: 38, right: 13, top: 18, bottom: 25 };
  const numbers = [prior.moments[7], closure.moments[7], prior.moments[8], closure.moments[8]];
  let lo = Math.min(0, ...numbers);
  let hi = Math.max(0, ...numbers);
  const span = hi - lo || 1;
  hi += .18 * span;
  if (lo < 0) lo -= .12 * span;
  const graphWidth = width - pad.left - pad.right;
  const graphHeight = height - pad.top - pad.bottom;
  const py = value => pad.top + graphHeight * (hi - value) / (hi - lo);
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.lineWidth = 1;
  for (let i = 0; i <= 3; i += 1) {
    const value = lo + (hi - lo) * i / 3;
    ctx.strokeStyle = colors.grid; ctx.beginPath(); ctx.moveTo(pad.left, py(value)); ctx.lineTo(width - pad.right, py(value)); ctx.stroke();
    ctx.fillStyle = colors.text; ctx.fillText(axisNumber(value), pad.left - 5, py(value));
  }
  ctx.strokeStyle = colors.axis; ctx.beginPath(); ctx.moveTo(pad.left, py(0)); ctx.lineTo(width - pad.right, py(0)); ctx.stroke();
  [7, 8].forEach((order, group) => {
    const middle = pad.left + graphWidth * (.25 + group * .5);
    const barWidth = Math.min(28, graphWidth * .12);
    [prior.moments[order], closure.moments[order]].forEach((value, series) => {
      const x = middle + (series ? 3 : -barWidth - 3);
      ctx.fillStyle = series ? colors.orange : colors.green;
      ctx.fillRect(x, Math.min(py(value), py(0)), barWidth, Math.max(1, Math.abs(py(value) - py(0))));
    });
    ctx.fillStyle = colors.text; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText(number(closure.moments[order], 2), middle, py(Math.max(0, closure.moments[order])) - 5);
    ctx.textBaseline = 'top'; ctx.fillText(`m${sub[order]}`, middle, height - pad.bottom + 7);
  });
}

function plotRange() {
  const ref = snapshot.prior;
  const n = 400;
  const points = Array.from({ length: n + 1 }, (_, k) => {
    const x = -ref.radius + 2 * ref.radius * k / n;
    return [x, referenceDensity(ref, x)];
  });
  const peak = Math.max(...points.map(point => point[1]));
  const visible = points.filter(point => point[1] > peak * .001);
  let low = visible[0][0] - .3;
  let high = visible.at(-1)[0] + .3;
  if (state.lab === 'update') { low = Math.min(low, state.y - .3); high = Math.max(high, state.y + .3); }
  return [low, high];
}

function draw() {
  if (!snapshot) return;
  const { prior, fitted, trueLambda, fit, posterior, posteriorLambda } = snapshot;
  const range = plotRange();
  if (state.lab === 'update') {
    const densityCurves = [{ fn: x => referenceDensity(prior, x), color: colors.green, fill: '#e4efe5' }];
    if (state.applied) densityCurves.push({ fn: x => referenceDensity(posterior, x), color: colors.orange, fill: '#e3a17a25' });
    plot($('density-plot'), densityCurves, range, { nonnegative: true, observation: state.y });
    const scoreCurves = [{ fn: x => score(fit.lambda, x), color: colors.green }, { fn: x => (state.y - x) / state.R, color: colors.blue, dash: [4, 4] }];
    if (state.applied) scoreCurves.push({ fn: x => score(posteriorLambda, x), color: colors.orange });
    plot($('second-plot'), scoreCurves, range);
  } else {
    plot($('density-plot'), [{ fn: x => referenceDensity(prior, x), color: colors.green, fill: '#e4efe5' }, { fn: x => referenceDensity(fitted, x), color: colors.orange, dash: [5, 5] }], range, { nonnegative: true });
    if (state.lab === 'closure') plotClosure();
    else plot($('second-plot'), [{ fn: x => score(trueLambda, x), color: colors.green }, { fn: x => score(fit.lambda, x), color: colors.orange, dash: [5, 5] }], range);
  }
}

function render() {
  try {
    compute();
    updateControls();
    if (state.lab === 'score') renderScore();
    else if (state.lab === 'closure') renderClosure();
    else renderUpdate();
    $('status').classList.remove('error');
    $('status').textContent = state.lab === 'update'
      ? 'Teaching example. Quadrature supplies reference density plots and posterior moments. The paper’s truncated posterior moment recovery is not run.'
      : 'Teaching example, not a paper experiment. Quadrature supplies synthetic input/reference moments and density plots. Score fitting and closure use algebra.';
    draw();
    document.documentElement.dataset.ready = 'true';
  } catch (error) {
    $('status').textContent = `This setting could not be evaluated: ${error.message}`;
    $('status').classList.add('error');
    console.error(error);
  }
}

function scheduleRender() {
  if (pending) return;
  pending = true;
  requestAnimationFrame(() => { pending = false; render(); });
}

for (const key of ['a', 'b', 'c', 'y', 'R']) $(key).addEventListener('input', event => { state[key] = Number(event.target.value); scheduleRender(); });
document.querySelectorAll('[data-lab]').forEach(button => button.addEventListener('click', () => {
  state.lab = button.dataset.lab;
  const url = new URL(location.href);
  url.searchParams.set('lab', state.lab);
  history.replaceState(null, '', url);
  $('inspector').scrollTop = 0;
  document.querySelector('.controls').scrollTop = 0;
  scheduleRender();
}));
$('reset').addEventListener('click', () => { Object.assign(state, DEFAULTS, { applied: true }); scheduleRender(); });
$('two-wells').addEventListener('click', () => { Object.assign(state, { a: .35, b: -1.3, c: 0 }); scheduleRender(); });
$('one-well').addEventListener('click', () => { Object.assign(state, { a: .35, b: .8, c: .25 }); scheduleRender(); });
$('apply-update').addEventListener('click', () => { state.applied = !state.applied; scheduleRender(); });
new ResizeObserver(() => requestAnimationFrame(draw)).observe(document.querySelector('.plots-column'));
render();
