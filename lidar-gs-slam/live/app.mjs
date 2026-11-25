import {covarianceLab, covarianceDefaults, TrackingLab, transformPoint, makeBudgetScene, manageBudget, ellipseBoundary, clamp} from './model.mjs';

const $ = id => document.getElementById(id);
const colors = {teal: '#16736e', rust: '#b44f37', blue: '#276991', ink: '#182d33', gray: '#8c9c98', line: '#d6dedc', paper: '#fffdfa'};
const format = (x, n = 2) => Number.isFinite(x) ? x.toFixed(n) : '—';
const metric = (label, value, detail = '') => `<div class="metric"><span>${label}</span><strong>${value}</strong>${detail ? `<small>${detail}</small>` : ''}</div>`;
const title = (x, y, content, cls = 'figure-title', anchor = 'start') => `<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${content}</text>`;
const path = points => points.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join(' ');
const circle = (x, y, r, fill, extra = '') => `<circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="${r}" fill="${fill}" ${extra}/>`;
let lab = ['covariance', 'tracking', 'budget'].includes(new URLSearchParams(location.search).get('lab')) ? new URLSearchParams(location.search).get('lab') : 'covariance';
let covarianceState, tracking = new TrackingLab(), budgetScene = makeBudgetScene(), budgetState, selectedId = 127;

function projected(p, cx, cy, scale) {
  return [cx + scale * (.84 * p[0] - .54 * p[1]), cy + scale * (.29 * p[0] + .45 * p[1] - .85 * p[2])];
}
function vector(v, scale = 1) { return v.map(x => x * scale); }
function projectedEllipse(center, u, v, a, b, project) {
  return Array.from({length: 65}, (_, i) => {
    const t = i * 2 * Math.PI / 64;
    return project(center.map((x, j) => x + u[j] * a * Math.cos(t) + v[j] * b * Math.sin(t)));
  });
}
function drawArrow(start, end, color, dashed = false, label = '') {
  const dx = end[0] - start[0], dy = end[1] - start[1], length = Math.hypot(dx, dy), ux = dx / Math.max(length, .001), uy = dy / Math.max(length, .001);
  return `<path d="${path([start, end])}" stroke="${color}" stroke-width="1.8" fill="none"${dashed ? ' stroke-dasharray="4 3"' : ''}/><path d="${path([[end[0] - 7 * ux + 3 * uy, end[1] - 7 * uy - 3 * ux], end, [end[0] - 7 * ux - 3 * uy, end[1] - 7 * uy + 3 * ux]])}" stroke="${color}" stroke-width="1.8" fill="none"/>${label ? `<text x="${end[0] + 7}" y="${end[1] + 4}" style="fill:${color};font-size:11px">${label}</text>` : ''}`;
}
function renderCovariance() {
  covarianceState = covarianceLab({shape: $('shape').value, tangent: Number($('tangent').value), noise: Number($('noise').value), range: Number($('range').value)});
  const state = covarianceState, [u, v, normal] = state.eigen.vectors;
  const centered = state.points.map(p => p.map((x, i) => x - state.mean[i]));
  const projectedUnits = centered.map(p => projected(p, 0, 0, 1));
  const cloudScale = Math.min(42, 163 / Math.max(...projectedUnits.map(p => Math.abs(p[0]))), 101 / Math.max(...projectedUnits.map(p => Math.abs(p[1]))));
  const cloudProject = p => projected(p, 194, 154, cloudScale), diskProject = p => projected(p, 590, 154, 52);
  const ambiguous = state.normalStatus !== 'Well-separated normal';
  let html = `<title>Covariance from ${state.shape} samples and the corresponding initialized Gaussian disk</title>`;
  html += title(18, 23, 'Normalized 3D neighborhood') + title(414, 23, 'Range-adaptive surface disk');
  html += `<line x1="390" y1="36" x2="390" y2="272" stroke="${colors.line}"/>`;
  // Projected coordinate axes stay fixed while the eigensystem changes.
  for (const axis of [[2.4, 0, 0], [0, 2.4, 0], [0, 0, 1.8]]) html += `<path d="${path([cloudProject(axis.map(x => -.30 * x)), cloudProject(axis)])}" stroke="${colors.line}" stroke-width="1"/>`;
  html += `<path d="${path(projectedEllipse([0, 0, 0], u, v, Math.sqrt(state.eigen.values[0]), Math.sqrt(state.eigen.values[1]), cloudProject))}Z" fill="${colors.teal}" fill-opacity=".10" stroke="${colors.teal}" stroke-opacity=".28"/>`;
  const points = centered.sort((a, b) => (.45 * a[0] + .78 * a[1] + .5 * a[2]) - (.45 * b[0] + .78 * b[1] + .5 * b[2]));
  points.forEach(p => { const xy = cloudProject(p); html += circle(xy[0], xy[1], 2, colors.teal, 'opacity=".57"'); });
  html += drawArrow(cloudProject([0, 0, 0]), cloudProject(vector(u, Math.sqrt(state.eigen.values[0]) * 1.5)), colors.teal, false, 'e₀');
  html += drawArrow(cloudProject([0, 0, 0]), cloudProject(vector(v, Math.sqrt(state.eigen.values[1]) * 1.5)), colors.blue, false, 'e₁');
  html += drawArrow(cloudProject([0, 0, 0]), cloudProject(vector(normal, 1.6)), colors.rust, ambiguous, 'n');
  html += title(194, 267, `λ = [${state.eigen.values.map(x => format(x, 3)).join(', ')}]`, 'figure-caption', 'middle');
  html += title(194, 284, 'Samples and eigenvalues in normalized local coordinates', 'figure-caption', 'middle');
  const grid = [-1.5, -1, -.5, 0, .5, 1, 1.5];
  for (const tick of grid) {
    html += `<path d="${path([diskProject([tick, -1.4, 0]), diskProject([tick, 1.4, 0])])}" stroke="${colors.line}" stroke-opacity=".65" stroke-width=".8"/>`;
    html += `<path d="${path([diskProject([-1.4, tick, 0]), diskProject([1.4, tick, 0])])}" stroke="${colors.line}" stroke-opacity=".65" stroke-width=".8"/>`;
  }
  for (const [factor, opacity] of [[1, .13], [.66, .16], [.33, .21]]) html += `<path d="${path(projectedEllipse([0, 0, 0], u, v, state.diskScales[0] * factor, state.diskScales[1] * factor, diskProject))}Z" fill="${colors.teal}" fill-opacity="${opacity}" stroke="${colors.teal}" stroke-opacity=".6" stroke-width="${factor === 1 ? 1.7 : .7}"/>`;
  html += `<path d="${path([diskProject(vector(u, -state.diskScales[0])), diskProject(vector(u, state.diskScales[0]))])}" stroke="${colors.teal}" stroke-width="1.5"/>`;
  html += `<path d="${path([diskProject(vector(v, -state.diskScales[1])), diskProject(vector(v, state.diskScales[1]))])}" stroke="${colors.blue}" stroke-width="1.5"/>`;
  html += drawArrow(diskProject([0, 0, 0]), diskProject(vector(normal, .88)), colors.rust, ambiguous, 'n');
  html += circle(590, 154, 3, colors.ink);
  html += title(590, 267, `r = ${state.range} m · scale factor 0.04 · cap 2 m`, 'figure-caption', 'middle');
  html += title(590, 284, ambiguous ? 'Dashed n: orientation lacks a strong surface constraint' : 'Disk axes follow the covariance tangent plane', 'figure-caption', 'middle');
  $('covariance-plot').innerHTML = html;
  $('covariance-metrics').innerHTML = metric('Raw linearity L', format(state.features.linearity, 3)) + metric('Raw curvature C', format(state.features.curvature, 4)) + metric('Initial scales s₀ × s₁', `${format(state.diskScales[0])} × ${format(state.diskScales[1])} m`);
  $('normal-status').textContent = `${state.normalStatus}. Relative normal eigengap (λ₁ − λ₂)/λ₀ = ${format(state.features.normalGap, 3)}. This is a shape diagnostic, not a probability.`;
  $('normal-status').classList.toggle('warning', ambiguous);
  $('tangent-value').value = format(state.tangent); $('noise-value').value = format(state.noise, 3); $('range-value').value = state.range + ' m';
  $('result-tag').textContent = '180 seeded 3D samples';
}

function coordinateFrame({xmin, xmax, ymin, ymax, left, top, width, height}) {
  const scale = Math.min(width / (xmax - xmin), height / (ymax - ymin)), ox = left + (width - scale * (xmax - xmin)) / 2, oy = top + (height - scale * (ymax - ymin)) / 2;
  const X = x => ox + (x - xmin) * scale, Y = y => oy + (ymax - y) * scale;
  let svg = '';
  for (let x = Math.ceil(xmin); x <= Math.floor(xmax); x++) {
    svg += `<line x1="${X(x)}" y1="${Y(ymin)}" x2="${X(x)}" y2="${Y(ymax)}" stroke="${colors.line}" stroke-width="${x === 0 ? 1 : .6}" stroke-opacity=".8"/>`;
    svg += `<text x="${X(x)}" y="${Y(ymin) + 15}" text-anchor="middle" class="axis-label">${x}</text>`;
  }
  for (let y = Math.ceil(ymin); y <= Math.floor(ymax); y++) {
    svg += `<line x1="${X(xmin)}" y1="${Y(y)}" x2="${X(xmax)}" y2="${Y(y)}" stroke="${colors.line}" stroke-width="${y === 0 ? 1 : .6}" stroke-opacity=".8"/>`;
    svg += `<text x="${X(xmin) - 8}" y="${Y(y) + 4}" text-anchor="end" class="axis-label">${y}</text>`;
  }
  svg += `<text x="${X(xmax) + 6}" y="${Y(ymin) + 15}" class="axis-label">x [m]</text><text x="${X(xmin)}" y="${Y(ymax) - 8}" class="axis-label">y [m]</text>`;
  return {X, Y, scale, svg};
}
function lossPlot(trace) {
  const x = i => 576 + i / Math.max(6, trace.length - 1) * 176, y = value => 160 - clamp(value / trace[0]) * 93;
  let html = title(563, 36, 'Objective / start', 'figure-title');
  for (const p of [0, .5, 1]) html += `<line x1="576" y1="${160 - p * 93}" x2="752" y2="${160 - p * 93}" stroke="${colors.line}"/><text x="569" y="${164 - p * 93}" text-anchor="end" style="font-size:10px">${p * 100}%</text>`;
  for (let i = 0; i <= Math.max(6, trace.length - 1); i += 2) html += `<text x="${x(i)}" y="177" text-anchor="middle" style="font-size:10px">${i}</text>`;
  html += `<path d="${path(trace.map((v, i) => [x(i), y(v)]))}" fill="none" stroke="${colors.teal}" stroke-width="2.5"/>`;
  trace.forEach((v, i) => html += circle(x(i), y(v), 2.8, colors.teal));
  html += title(752, 192, 'Gauss–Newton iteration', 'figure-caption', 'end');
  return html;
}
function renderTracking() {
  const state = tracking.state(), frame = coordinateFrame({xmin: -4.2, xmax: 3.7, ymin: -2.3, ymax: 3.0, left: 25, top: 35, width: 495, height: 228}), {X, Y, scale} = frame;
  let html = `<title>Registration iteration ${state.iteration}: translation error ${format(state.error.translation * 100, 1)} centimetres</title>` + frame.svg;
  state.scene.points.forEach((p, i) => {
    const t = p.target, q = transformPoint(p.source, state.pose), color = p.type === 'plane' ? colors.teal : colors.rust;
    const [a, b, c] = [p.targetCovariance[0][0], p.targetCovariance[0][1], p.targetCovariance[1][1]], root = Math.sqrt((a - c) ** 2 + 4 * b * b), major = Math.sqrt((a + c + root) / 2), minor = Math.sqrt((a + c - root) / 2), angle = .5 * Math.atan2(2 * b, a - c);
    html += `<ellipse cx="${X(t[0])}" cy="${Y(t[1])}" rx="${major * scale}" ry="${minor * scale}" transform="rotate(${-angle * 180 / Math.PI} ${X(t[0])} ${Y(t[1])})" fill="${color}" fill-opacity=".07" stroke="${color}" stroke-opacity=".18"/>`;
    html += `<path d="${path([[X(t[0]), Y(t[1])], [X(q[0]), Y(q[1])]])}" fill="none" stroke="${colors.gray}" stroke-width=".8" stroke-dasharray="2 3" opacity=".7"/>`;
    html += circle(X(t[0]), Y(t[1]), 3.1, colors.paper, `stroke="${colors.gray}" stroke-width="1.2"`);
    const px = X(q[0]), py = Y(q[1]), radius = 3.2;
    html += `<path d="M${px},${py - radius}l${radius},${radius * 1.7}h${-2 * radius}Z" fill="${color}" opacity=".9"><title>Match ${i + 1}: κ̄=${format(state.weights[i].normalized)}, w=${format(state.weights[i].weight, 1)}</title></path>`;
  });
  html += `<line x1="535" y1="24" x2="535" y2="279" stroke="${colors.line}"/>` + lossPlot(state.trace);
  html += title(563, 218, `t = (${format(state.pose[0])}, ${format(state.pose[1])}) m`, 'figure-caption');
  html += title(563, 237, `θ = ${format(state.pose[2] * 180 / Math.PI, 1)}°`, 'figure-caption');
  html += title(563, 259, 'Cost comparison: same weights only', 'figure-caption');
  html += title(563, 278, 'Compare pose errors across settings', 'figure-caption');
  $('tracking-plot').innerHTML = html;
  const loss = state.trace[state.trace.length - 1];
  $('tracking-metrics').innerHTML = metric('Objective / its initial value', `${format(100 * loss / state.trace[0], 1)}%`, `Raw objective ${format(loss, 1)}`) + metric('Translation error', `${format(100 * state.error.translation, 1)} cm`, 'Against synthetic truth') + metric('Heading error', `${format(state.error.angle, 2)}°`, 'Against synthetic truth');
  $('tracking-status').textContent = state.converged ? `Converged after ${state.iteration} steps. Toggle geometry weights to restart the same observations and compare final pose error.` : state.iteration ? `Accepted step ${state.iteration}. The actual objective fell from ${format(state.trace[state.iteration - 1], 1)} to ${format(loss, 1)}.` : 'Step updates the pose once. Solve iterates to convergence. Sliders and weighting restart the same seeded observations.';
  $('result-tag').textContent = `Iteration ${state.iteration} · ${state.weighted ? 'geometry weights' : 'uniform weights'}`;
  const groupMean = type => { const w = state.weights.filter((_, i) => state.scene.points[i].type === type); return w.reduce((sum, v) => sum + v.weight, 0) / w.length; };
  $('plane-weight').textContent = format(groupMean('plane'), 1); $('mixed-weight').textContent = format(groupMean('mixed'), 1);
  $('bias-value').value = format(Number($('bias').value)) + ' m'; $('heading-value').value = $('heading').value + '°';
  $('track-step').disabled = state.converged; $('track-solve').disabled = state.converged;
}
function resetTracking() { tracking.reset({weighted: $('weighted').checked, bias: Number($('bias').value), heading: Number($('heading').value)}); renderTracking(); }

function scoreColor(score) {
  const a = [22, 115, 110], b = [180, 79, 55], t = clamp(score);
  return `rgb(${a.map((v, i) => Math.round(v + t * (b[i] - v))).join(',')})`;
}
function scoreHistogram(state) {
  const bins = Array.from({length: 10}, () => 0); state.originals.forEach(g => bins[Math.min(9, Math.floor(g.score * 10))]++);
  const max = Math.max(...bins), left = 571, width = 181, base = 169, height = 95, X = score => left + width * score;
  let html = title(563, 36, 'Initial map scores', 'figure-title');
  for (let i = 0; i < bins.length; i++) html += `<rect x="${left + i * width / 10 + 1}" y="${base - bins[i] / max * height}" width="${width / 10 - 3}" height="${bins[i] / max * height}" fill="${scoreColor((i + .5) / 10)}" fill-opacity=".8"><title>c in [${format(i / 10, 1)}, ${format((i + 1) / 10, 1)}]: ${bins[i]} Gaussians</title></rect>`;
  html += `<line x1="${left}" y1="${base}" x2="${left + width}" y2="${base}" stroke="${colors.gray}"/>`;
  for (const value of [0, .5, 1]) html += `<text x="${X(value)}" y="186" text-anchor="middle" style="font-size:10px">${value}</text>`;
  for (const [value, color] of [[state.planeThreshold, colors.teal], [state.splitThreshold, colors.rust]]) html += `<line x1="${X(value)}" y1="61" x2="${X(value)}" y2="172" stroke="${color}" stroke-width="1.4" stroke-dasharray="4 3"/>`;
  html += title(563, 213, `Plane threshold c ≤ ${format(state.planeThreshold)}`, 'figure-caption');
  html += title(563, 232, `Split threshold c ≥ ${format(state.splitThreshold)}`, 'figure-caption');
  html += title(563, 257, `−${state.removed.length} redundant primitives`, 'figure-caption');
  html += title(563, 277, `+${state.splitParents.length} net after splitting`, 'figure-caption');
  return html;
}
function drawBudget() {
  const state = budgetState, frame = coordinateFrame({xmin: -4.25, xmax: 3.7, ymin: -2.6, ymax: 2.6, left: 24, top: 35, width: 495, height: 228}), {X, Y} = frame;
  let html = `<title>Budget pass: ${state.gaussians.length} Gaussians from ${state.originals.length}, ${format(state.coverage * 100, 1)}% sampled footprint coverage</title>` + frame.svg;
  budgetScene.samples.forEach((p, i) => { if (i % 3 === 0) html += circle(X(p.x), Y(p.y), .6, colors.gray, 'opacity=".4"'); });
  const ordered = [...state.gaussians].sort((a, b) => Number(b.operation === 'expanded') - Number(a.operation === 'expanded'));
  ordered.forEach(g => {
    const p = ellipseBoundary(g, 32).map(p => [X(p.x), Y(p.y)]), selected = String(g.id) === String(selectedId), color = scoreColor(g.score);
    html += `<path d="${path(p)}Z" class="splat" data-gaussian="${g.id}" role="button" tabindex="${selected ? 0 : -1}" aria-label="Gaussian ${g.id}, ${g.type}, score ${format(g.score, 3)}${g.operation === 'unchanged' ? '' : ', ' + g.operation}" fill="${color}" fill-opacity="${selected ? .35 : g.operation === 'expanded' ? .10 : .22}" stroke="${selected ? colors.ink : color}" stroke-width="${selected ? 2 : g.operation === 'expanded' ? 1.3 : .7}"><title>${g.type} #${g.id} · c=${format(g.score, 3)} · ${g.operation}</title></path>`;
    html += circle(X(g.x), Y(g.y), g.operation === 'child' ? 1.9 : 1.3, color, 'pointer-events="none"');
  });
  html += `<line x1="535" y1="24" x2="535" y2="279" stroke="${colors.line}"/>` + scoreHistogram(state);
  html += title(263, 22, 'Finite 1σ footprints on a synthetic surface patch', 'figure-caption', 'middle');
  $('budget-plot').innerHTML = html;
  $('budget-metrics').innerHTML = metric('Gaussian count', `${state.gaussians.length} / ${state.originals.length}`, `${format((state.gaussians.length / state.originals.length - 1) * 100, 1)}% change`) + metric('Sample coverage', `${format(state.coverage * 100, 1)}%`, `Before ${format(state.beforeCoverage * 100, 1)}%`) + metric('Removed / split parents', `${state.removed.length} / ${state.splitParents.length}`, 'One illustrative pass');
  $('budget-status').textContent = `Click a footprint to inspect cᵢ. Splitting creates capacity; later map optimization would fit the children. Counts are not runtime or VRAM estimates.`;
  $('result-tag').textContent = `${state.gaussians.length} Gaussians · ${state.removed.length} removed`;
  renderSelected();
}
function renderSelected() {
  const state = budgetState;
  let g = state.gaussians.find(g => String(g.id) === String(selectedId));
  if (!g) g = state.gaussians.find(g => g.type !== 'plane') || state.gaussians[0];
  selectedId = g.id;
  $('selected-label').textContent = `${g.type[0].toUpperCase() + g.type.slice(1)} #${g.id}`;
  $('selected-score').textContent = `c = ${format(g.score, 3)}`;
  $('score-breakdown').innerHTML = [['Linearity L̃', g.normalized.linearity, .55], ['Curvature C̃', g.normalized.curvature, .30], ['Residual r̃', g.normalized.residual, .15]].map(([label, value, weight]) => `<div class="score-row"><span>${label}</span><div><i style="width:${value * 100}%" aria-hidden="true"></i></div><span>${format(value)}</span></div>`).join('');
}
function calculateBudget() {
  budgetState = manageBudget(budgetScene, {prune: $('prune').checked, split: $('split').checked, cell: Number($('cell').value), splitQuantile: Number($('split-quantile').value)});
  $('cell-value').value = format(Number($('cell').value)) + ' m'; $('split-quantile-value').value = Math.round(Number($('split-quantile').value) * 100) + '%';
  if (!budgetState.gaussians.some(g => String(g.id) === String(selectedId))) selectedId = budgetState.gaussians.find(g => g.type !== 'plane')?.id ?? budgetState.gaussians[0].id;
  drawBudget();
}
function setLab(name) {
  lab = name;
  document.querySelectorAll('[data-lab]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.lab === lab)));
  for (const key of ['covariance', 'tracking', 'budget']) for (const suffix of ['view', 'controls', 'explanation']) $(key + '-' + suffix).hidden = key !== lab;
  $('result-title').textContent = {covariance: 'A neighborhood supplies axes and a candidate normal', tracking: 'Registration uses the refined surface geometry', budget: 'Compression and splitting spend different parts of the budget'}[lab];
  $('scope').textContent = {covariance: 'Synthetic 3D neighborhood. Eigenvalues are measured from the displayed samples. Range scaling uses stated teaching constants.', tracking: 'Computed SE(2) teaching reduction. Fixed matches and synthetic biased returns. Compare pose error, not raw costs across weighting settings.', budget: 'Synthetic map-control illustration. Coverage counts points inside finite ellipses, not mesh F-score. Splitting is initialization only.'}[lab];
  $('expand').href = './?lab=' + lab;
  const url = new URL(location.href); url.searchParams.set('lab', lab); history.replaceState(null, '', url.pathname + url.search);
  if (lab === 'covariance') renderCovariance(); else if (lab === 'tracking') renderTracking(); else calculateBudget();
}

for (const button of document.querySelectorAll('[data-lab]')) button.addEventListener('click', () => setLab(button.dataset.lab));
for (const id of ['tangent', 'noise', 'range']) $(id).addEventListener('input', renderCovariance);
$('shape').addEventListener('change', () => { const preset = covarianceDefaults[$('shape').value]; $('tangent').value = preset.tangent; $('noise').value = preset.noise; renderCovariance(); });
$('cov-reset').addEventListener('click', () => { $('shape').value = 'plane'; $('tangent').value = .88; $('noise').value = .035; $('range').value = 12; renderCovariance(); });
$('weighted').addEventListener('change', resetTracking); for (const id of ['bias', 'heading']) $(id).addEventListener('input', resetTracking);
$('track-reset').addEventListener('click', resetTracking);
$('track-step').addEventListener('click', () => { tracking.step(); renderTracking(); });
$('track-solve').addEventListener('click', () => { tracking.solve(); renderTracking(); });
for (const id of ['prune', 'split']) $(id).addEventListener('change', calculateBudget); for (const id of ['cell', 'split-quantile']) $(id).addEventListener('input', calculateBudget);
$('budget-reset').addEventListener('click', () => { $('prune').checked = true; $('split').checked = true; $('cell').value = .9; $('split-quantile').value = .82; selectedId = 127; calculateBudget(); });
$('budget-plot').addEventListener('click', event => { const item = event.target.closest('[data-gaussian]'); if (!item) return; selectedId = item.dataset.gaussian; drawBudget(); });
$('budget-plot').addEventListener('keydown', event => {
  const current = budgetState.gaussians.findIndex(g => String(g.id) === String(selectedId));
  if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key) && event.target.closest('[data-gaussian]')) {
    event.preventDefault(); event.stopPropagation(); const step = ['ArrowLeft', 'ArrowUp'].includes(event.key) ? -1 : 1;
    selectedId = budgetState.gaussians[(current + step + budgetState.gaussians.length) % budgetState.gaussians.length].id; drawBudget();
    $('budget-plot').querySelector('[tabindex="0"]')?.focus();
  } else if (event.key === 'Enter' || event.key === ' ') { const item = event.target.closest('[data-gaussian]'); if (item) { event.preventDefault(); selectedId = item.dataset.gaussian; drawBudget(); $('budget-plot').querySelector('[tabindex="0"]')?.focus(); } }
});
// All updates are user-driven. No timers or animation continue in hidden slides.
setLab(lab);
