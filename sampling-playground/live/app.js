/* Sampling lab: one seeded sampler per ?demo= value. All numerical work lives in ../model.js;
 * this file owns controls, canvas drawing and the readouts derived from the model's records.
 */
(function () {
'use strict';
const S = window.SamplingModel;
const $ = id => document.getElementById(id);
const C = { ink: '#203129', muted: '#66756e', rule: '#d8ded7', grid: '#e9ede7', green: '#2f6b4f', soft: '#e7f0ea',
  rust: '#a94f2a', blue: '#496e87', amber: '#986b22', paper: '#fbfaf6' };
const DEMOS = ['gibbs', 'mh', 'hmc', 'slice', 'rejection', 'importance', 'smc'];
const CHAINS = ['gibbs', 'mh', 'hmc', 'slice'];
const params = new URLSearchParams(location.search);
const demo = DEMOS.includes(params.get('demo')) ? params.get('demo') : 'gibbs';
const isChain = CHAINS.includes(demo);
const SUB = { x: '₁', y: '₂' };
// fmt(v, d): fixed-point text with a true minus sign; '—' for non-finite values.
const fmt = (v, d = 2) => Number.isFinite(v) ? (Number(v.toFixed(d)) === 0 ? (0).toFixed(d) : v.toFixed(d)).replace('-', '−') : '—';
const pct = v => Number.isFinite(v) ? Math.round(v * 100) + '%' : '—';

/* ---------------- per-demo copy and controls ---------------- */
const rho = { id: 'rho', label: 'Target correlation ρ', min: .1, max: .95, step: .01, value: S.CHAIN.rho, format: v => fmt(v) };
const META = {
  gibbs: { kicker: 'Chain · exact conditionals', title: 'Gibbs sampling', heading: 'Correlated Gaussian · Gibbs',
    copy: 'Each move redraws one coordinate from its exact conditional (green bell). Nothing is rejected; correlation keeps the moves short.',
    controls: [rho], step: 'Step', batch: 1, ms: 450,
    legend: [[C.rust, 'target contours (39%, 86%)', 'line'], [C.blue, 'chain', 'dot'], [C.green, 'conditional being sampled', 'line'], [C.amber, 'start', 'dot']] },
  mh: { kicker: 'Chain · propose, then accept or stay', title: 'Random-walk Metropolis–Hastings', heading: 'Correlated Gaussian · Metropolis–Hastings',
    copy: 'A Gaussian step of size σ is proposed (dashed circle: 2σ). Uphill moves are kept; downhill moves are kept with probability π(x′)/π(x).',
    controls: [rho, { id: 'sigma', label: 'Proposal step σ', min: .1, max: 2.5, step: .01, value: S.CHAIN.mh.sigma, format: v => fmt(v) }],
    step: 'Step', batch: 1, ms: 450,
    legend: [[C.rust, 'target contours', 'line'], [C.blue, 'chain', 'dot'], [C.rust, 'rejected proposal', 'ring'], [C.green, '2σ proposal circle', 'dash']] },
  hmc: { kicker: 'Chain · gradient trajectories', title: 'Hamiltonian Monte Carlo', heading: 'Correlated Gaussian · HMC',
    copy: 'A fresh momentum and L = 9 leapfrog steps (amber path) carry the state along the ridge. The energy error ΔH sets the acceptance.',
    controls: [rho, { id: 'eps', label: 'Leapfrog step ε', min: .02, max: .8, step: .01, value: S.CHAIN.hmc.eps, format: v => fmt(v) }],
    step: 'Step', batch: 1, ms: 520,
    legend: [[C.rust, 'target contours', 'line'], [C.blue, 'chain', 'dot'], [C.amber, 'leapfrog path', 'line'], [C.rust, 'rejected endpoint', 'ring']] },
  slice: { kicker: 'Chain · uniform under the curve', title: 'Slice sampling', heading: 'Correlated Gaussian · slice',
    copy: 'Pick a height under π, step a width-w bracket out past the slice (green band), then shrink it toward the current point after each miss (×).',
    controls: [rho, { id: 'w', label: 'Initial bracket width w', min: .2, max: 4, step: .1, value: S.CHAIN.slice.w, format: v => fmt(v, 1) }],
    step: 'Step', batch: 1, ms: 520,
    legend: [[C.rust, 'target contours', 'line'], [C.blue, 'chain', 'dot'], [C.green, 'slice (the admissible set)', 'line'], [C.rust, 'miss → shrink', 'ring']] },
  rejection: { kicker: 'Exact · independent draws', title: 'Rejection sampling', heading: 'Two-bump target · rejection',
    copy: 'Draw x from q and a height uniformly under M·q(x); keep it if it lands under π. Kept draws are exact while M·q ≥ π everywhere.',
    controls: [{ id: 's', label: 'Proposal width s', min: .8, max: 3, step: .05, value: S.POP.proposal.s, format: v => fmt(v) },
      { id: 'slack', label: 'Envelope slack  M / sup(π/q)', min: .6, max: 3, step: .05, value: 1, format: v => fmt(v) }],
    step: 'Draw 25', batch: 25, ms: 260, initial: 40,
    legend: [[C.rust, 'target π', 'line'], [C.amber, 'envelope M·q', 'dash'], [C.blue, 'kept', 'dot'], ['#d9a58c', 'discarded', 'dot']] },
  importance: { kicker: 'Weighted · independent draws', title: 'Importance sampling', heading: 'Two-bump target · importance weights',
    copy: 'Keep every draw from q, weighted by w = π/q (stems). The weighted histogram approximates π; the Kish ESS says what the weights are worth.',
    controls: [{ id: 'm', label: 'Proposal mean', min: -2, max: 2, step: .05, value: S.POP.proposal.m, format: v => fmt(v) },
      { id: 's', label: 'Proposal width s', min: .3, max: 3, step: .05, value: S.POP.proposal.s, format: v => fmt(v) }],
    step: 'Draw 40', batch: 40, ms: 260, initial: 25,
    legend: [[C.rust, 'target π', 'line'], [C.amber, 'proposal q', 'dash'], [C.blue, 'weights / weighted histogram', 'line']] },
  smc: { kicker: 'Particles · sequential', title: 'Bootstrap particle filter', heading: 'Linear-Gaussian tracking · 60 observations',
    copy: 'Propagate the particles, weight them by the newest observation, and resample by the chosen rule. The exact Kalman mean is the reference.',
    controls: [{ id: 'N', label: 'Particles N', min: 10, max: 200, step: 2, value: S.SSM.N, format: v => String(v) },
      { id: 'rule', label: 'Resample', options: [['half', 'ESS < N/2'], ['always', 'every step'], ['never', 'never']], value: 'half' }],
    step: 'Advance t', batch: 1, ms: 360, initial: 20,
    legend: [[C.ink, 'hidden state (simulator)', 'line'], [C.amber, 'observations', 'ring'], [C.blue, 'particle mean · particles', 'line'], [C.rust, 'exact Kalman mean ± 2 sd', 'dash']] }
};
const meta = META[demo];

/* ---------------- state ---------------- */
const values = Object.fromEntries(meta.controls.map(c => [c.id, c.value]));
let seed = null;          // null keeps the model's documented default seed
let autoTimer = null;
let st = {};              // demo-specific run state

// build(count): recomputes the run for the current controls and seed, advanced to `count` steps.
function build(count) {
  if (isChain) {
    const o = { rho: values.rho, sigma: values.sigma, eps: values.eps, w: values.w };
    if (seed !== null) o.seed = seed;
    st = { run: S.runChain(demo, o), step: Math.min(count, S.CHAIN.steps) };
  } else if (demo === 'rejection') {
    st = { sampler: S.makeRejection({ s: values.s, slack: values.slack, seed: seed ?? undefined }), presses: 0 };
    for (let i = 0; i < count; i++) press();
  } else if (demo === 'importance') {
    st = { sampler: S.makeImportance({ m: values.m, s: values.s, seed: seed ?? undefined }), presses: 0,
      ew2: S.weightSecondMoment(values.m, values.s) };
    for (let i = 0; i < count; i++) press();
  } else {
    st = { filter: S.makeFilter({ N: values.N, rule: values.rule, seed: seed ?? undefined }) };
    for (let i = 0; i < count; i++) st.filter.step();
  }
}
// progress(): how many steps the current run has taken.
function progress() {
  if (isChain) return st.step;
  if (demo === 'smc') return st.filter.t;
  return st.presses;
}
// atEnd(): whether Step can advance further.
function atEnd() {
  if (isChain) return st.step >= S.CHAIN.steps;
  if (demo === 'smc') return st.filter.t >= S.SSM.T;
  return st.sampler.draws.length >= 12000;
}
// press(): one Step press for the active demo.
function press() {
  if (atEnd()) return;
  if (isChain) st.step += 1;
  else if (demo === 'smc') st.filter.step();
  else { st.sampler.draw(meta.batch); st.presses += 1; }
}

/* ---------------- controls ---------------- */
function buildControls() {
  const box = $('controls');
  box.innerHTML = '';
  for (const c of meta.controls) {
    if (c.options) {
      const label = document.createElement('div');
      label.className = 'range-label';
      label.innerHTML = `<span>${c.label}</span>`;
      const group = document.createElement('div');
      group.className = 'segmented';
      group.setAttribute('role', 'group');
      group.setAttribute('aria-label', c.label);
      for (const [v, text] of c.options) {
        const b = document.createElement('button');
        b.type = 'button';
        b.textContent = text;
        b.dataset.value = v;
        b.setAttribute('aria-pressed', String(values[c.id] === v));
        b.addEventListener('click', () => {
          values[c.id] = v;
          group.querySelectorAll('button').forEach(x => x.setAttribute('aria-pressed', String(x.dataset.value === v)));
          rebuild();
        });
        group.append(b);
      }
      box.append(label, group);
      continue;
    }
    const wrap = document.createElement('div');
    wrap.innerHTML = `<label class="range-label" for="ctl-${c.id}"><span>${c.label}</span><output id="out-${c.id}" for="ctl-${c.id}">${c.format(c.value)}</output></label>` +
      `<input id="ctl-${c.id}" type="range" min="${c.min}" max="${c.max}" step="${c.step}" value="${c.value}">`;
    box.append(wrap);
    wrap.querySelector('input').addEventListener('input', e => {
      values[c.id] = Number(e.target.value);
      $('out-' + c.id).textContent = c.format(values[c.id]);
      rebuild();
    });
  }
}
// rebuild(): a control changed; keep the step count and recompute from the same seed.
function rebuild() { build(progress()); draw(); }

function stopAuto() {
  if (autoTimer) clearInterval(autoTimer);
  autoTimer = null;
  $('auto').setAttribute('aria-pressed', 'false');
  $('auto').textContent = 'Auto';
}
function startAuto() {
  if (atEnd()) return;
  $('auto').setAttribute('aria-pressed', 'true');
  $('auto').textContent = 'Pause';
  autoTimer = setInterval(() => {
    press(); draw();
    if (atEnd()) stopAuto();
  }, meta.ms);
}

/* ---------------- canvas ---------------- */
const canvas = $('stage');
const ctx = canvas.getContext('2d');
let W = 700, H = 420;
function resize() {
  const r = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  W = Math.max(200, r.width); H = Math.max(160, r.height);
  canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}
const clampPx = v => Math.max(-1e4, Math.min(1e4, v));
function font(size, weight = 500, family = 'Inter, ui-sans-serif, system-ui, sans-serif') { ctx.font = `${weight} ${size}px ${family}`; }
function label(text, x, y, color = C.muted, size = 11, align = 'left', weight = 600) {
  font(size, weight); ctx.fillStyle = color; ctx.textAlign = align; ctx.textBaseline = 'middle'; ctx.fillText(text, x, y);
}
function dot(x, y, r, fill, alpha = 1) { ctx.globalAlpha = alpha; ctx.beginPath(); ctx.arc(x, y, r, 0, S.TAU); ctx.fillStyle = fill; ctx.fill(); ctx.globalAlpha = 1; }
function ring(x, y, r, color, width = 1.6, alpha = 1) { ctx.globalAlpha = alpha; ctx.beginPath(); ctx.arc(x, y, r, 0, S.TAU); ctx.strokeStyle = color; ctx.lineWidth = width; ctx.stroke(); ctx.globalAlpha = 1; }
function cross(x, y, r, color, width = 1.8) { ctx.beginPath(); ctx.moveTo(x - r, y - r); ctx.lineTo(x + r, y + r); ctx.moveTo(x + r, y - r); ctx.lineTo(x - r, y + r); ctx.strokeStyle = color; ctx.lineWidth = width; ctx.stroke(); }
function polyline(pts, color, width = 2, dash = [], alpha = 1) {
  if (pts.length < 2) return;
  ctx.save(); ctx.globalAlpha = alpha; ctx.setLineDash(dash); ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin = 'round';
  ctx.beginPath(); pts.forEach(([x, y], i) => i ? ctx.lineTo(clampPx(x), clampPx(y)) : ctx.moveTo(clampPx(x), clampPx(y))); ctx.stroke(); ctx.restore();
}

/* ---- chain view ---- */
function drawChain() {
  const { run } = st, k = st.step, r = values.rho;
  const lim = 3.2, size = Math.min(W - 20, H - 20), scale = size / (2 * lim);
  const cx = W / 2, cy = H / 2;
  const X = x => cx + x * scale, Y = y => cy - y * scale;
  const P = p => [X(p.x), Y(p.y)];

  // grid and axes
  ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
  for (let v = -3; v <= 3; v++) {
    ctx.beginPath(); ctx.moveTo(X(v), Y(-lim)); ctx.lineTo(X(v), Y(lim)); ctx.moveTo(X(-lim), Y(v)); ctx.lineTo(X(lim), Y(v)); ctx.stroke();
  }
  ctx.strokeStyle = '#c9d1c9'; ctx.beginPath(); ctx.moveTo(X(-lim), Y(0)); ctx.lineTo(X(lim), Y(0)); ctx.moveTo(X(0), Y(-lim)); ctx.lineTo(X(0), Y(lim)); ctx.stroke();
  for (let v = -3; v <= 3; v++) if (v) { label(String(v).replace('-', '−'), X(v), Y(0) + 10, '#8b9690', 9.5, 'center', 500); label(String(v).replace('-', '−'), X(0) - 8, Y(v), '#8b9690', 9.5, 'right', 500); }
  label('x₁', X(lim) - 4, Y(0) - 11, C.muted, 12, 'right', 700);
  label('x₂', X(0) + 8, Y(lim) + 6, C.muted, 12, 'left', 700);

  // Mahalanobis contours: radius 1 holds 39% of the mass, radius 2 holds 86%.
  const s = Math.sqrt(1 - r * r);
  for (const m of [.5, 1, 1.5, 2, 2.5]) {
    const pts = [];
    for (let i = 0; i <= 96; i++) { const t = i / 96 * S.TAU; pts.push([X(m * Math.cos(t)), Y(m * (r * Math.cos(t) + s * Math.sin(t)))]); }
    const main = m === 1 || m === 2;
    polyline(pts, C.rust, main ? 1.4 : .8, [], main ? .75 : .3);
    if (main) {
      // label where the ellipse reaches farthest along the ridge
      let best = null;
      for (let i = 0; i < 96; i++) { const t = i / 96 * S.TAU, px = m * Math.cos(t), py = m * (r * Math.cos(t) + s * Math.sin(t)); if (!best || px + py > best[0] + best[1]) best = [px, py]; }
      label(m === 1 ? '39%' : '86%', X(best[0]) + 6, Y(best[1]) - 6, C.rust, 10.5, 'left', 700);
    }
  }

  const pts = run.points.slice(0, k + 1), moves = run.moves.slice(0, k);

  // rejected proposals, faint, for every visible move
  moves.forEach(mv => { if (!mv.accepted && mv.proposal) ring(...P(mv.proposal), 3.5, C.rust, 1.2, .35); });

  // path
  polyline(pts.map(P), C.blue, 1.6, [], .55);

  // latest move construction
  const last = moves[moves.length - 1];
  if (last) drawConstruction(last, X, Y, P, scale, r);

  // samples
  pts.forEach((p, i) => {
    if (i === 0) return;
    const mv = moves[i - 1];
    if (!mv.accepted) return;
    const [x, y] = P(p);
    if (i <= S.CHAIN.shown) {
      dot(x, y, 7, C.blue); font(8.5, 800); ctx.fillStyle = '#fff'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(String(i), x, y + .5);
    } else dot(x, y, 3.6, C.blue, .85);
  });
  const [sx, sy] = P(pts[0]);
  dot(sx, sy, 7.5, C.amber); label('start', sx - 11, sy + 2, C.amber, 11, 'right', 700);
  const [ex, ey] = P(pts[pts.length - 1]);
  ring(ex, ey, 11, C.blue, 1.6, .6);
}

// drawConstruction(mv, …): draws what produced the latest move — the conditional, proposal, trajectory or slice.
function drawConstruction(mv, X, Y, P, scale, r) {
  const [fx, fy] = P(mv.from), [tx, ty] = P(mv.to);
  if (demo === 'gibbs' || demo === 'slice') {
    // the line along which this coordinate moves
    const alongX = mv.axis === 'x';
    ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = C.green; ctx.globalAlpha = .45; ctx.lineWidth = 1.2; ctx.beginPath();
    if (alongX) { ctx.moveTo(X(-3.2), fy); ctx.lineTo(X(3.2), fy); } else { ctx.moveTo(fx, Y(-3.2)); ctx.lineTo(fx, Y(3.2)); }
    ctx.stroke(); ctx.restore();
    const at = v => alongX ? [X(v), fy] : [fx, Y(v)];
    const off = (pt, d) => alongX ? [pt[0], pt[1] - d] : [pt[0] + d, pt[1]];
    if (demo === 'gibbs') {
      const peak = 1 / (mv.sd * Math.sqrt(S.TAU)), pts = [];
      for (let i = 0; i <= 80; i++) { const v = mv.mean + (i / 40 - 1) * 3.5 * mv.sd; pts.push(off(at(v), 48 * S.normalPdf(v, mv.mean, mv.sd) / peak)); }
      ctx.save(); ctx.beginPath(); pts.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.closePath(); ctx.fillStyle = C.green; ctx.globalAlpha = .13; ctx.fill(); ctx.restore();
      polyline(pts, C.green, 1.8);
      const top = off(at(mv.mean), 56), other = alongX ? 'x₂' : 'x₁', own = alongX ? 'x₁' : 'x₂';
      label(`π(${own} | ${other} = ${fmt(alongX ? mv.from.y : mv.from.x)})`, top[0] + (alongX ? 0 : 6), top[1] - (alongX ? 4 : 0), C.green, 11, alongX ? 'center' : 'left', 700);
    } else {
      // exact slice along the line: the conditional is Gaussian, so the admissible set is an interval
      const sd = Math.sqrt(1 - r * r), mean = r * (alongX ? mv.from.y : mv.from.x);
      const peakLog = S.logPi(alongX ? { x: mean, y: mv.from.y } : { x: mv.from.x, y: mean }, r);
      const hw = sd * Math.sqrt(Math.max(0, 2 * (peakLog - mv.level)));
      polyline([at(mean - hw), at(mean + hw)], C.green, 9, [], .25);
      const b0 = off(at(mv.bracket[0]), -14), b1 = off(at(mv.bracket[1]), -14);
      polyline([b0, b1], C.muted, 1.2);
      for (const b of [b0, b1]) polyline(alongX ? [[b[0], b[1] - 5], [b[0], b[1] + 5]] : [[b[0] - 5, b[1]], [b[0] + 5, b[1]]], C.muted, 1.2);
      label('bracket after stepping out', ...off(b1, alongX ? -12 : -14), C.muted, 10, alongX ? 'right' : 'left', 600);
      polyline([off(at(mv.final[0]), -7), off(at(mv.final[1]), -7)], C.green, 2.4);
      mv.misses.forEach(v => cross(...at(v), 4, C.rust));
    }
    polyline([[fx, fy], [tx, ty]], C.blue, 2.2);
  } else if (demo === 'mh') {
    ctx.save(); ctx.setLineDash([5, 4]); ring(fx, fy, 2 * values.sigma * scale, C.green, 1.2, .7); ctx.restore();
    const [px, py] = P(mv.proposal);
    polyline([[fx, fy], [px, py]], mv.accepted ? C.blue : C.rust, 2, mv.accepted ? [] : [5, 4]);
    if (!mv.accepted) ring(px, py, 6, C.rust, 2);
    label(`α = ${fmt(mv.alpha)}`, px + 10, py - 10, mv.accepted ? C.blue : C.rust, 11.5, 'left', 700);
  } else if (demo === 'hmc') {
    const path = mv.path.map(P);
    polyline(path, C.amber, 2);
    path.forEach(([x, y], i) => { if (i) dot(x, y, 2.4, C.amber); });
    const [px, py] = path[path.length - 1];
    if (!mv.accepted) ring(px, py, 6, C.rust, 2);
    label(`ΔH = ${fmt(mv.dH, 3)}`, px + 14, py + 16, mv.accepted ? C.amber : C.rust, 11.5, 'left', 700);
  }
}

/* ---- 1-D population views ---- */
const XMIN = S.POP.xmin, XMAX = S.POP.xmax;
function band(top, bottom) {
  const left = 34, right = W - 14;
  return { top, bottom, X: x => left + (x - XMIN) / (XMAX - XMIN) * (right - left), left, right };
}
function curve(b, f, ymax, color, width = 2, dash = []) {
  const pts = [];
  for (let i = 0; i <= 240; i++) { const x = XMIN + i / 240 * (XMAX - XMIN); pts.push([b.X(x), b.bottom - Math.min(f(x) / ymax, 1.04) * (b.bottom - b.top)]); }
  polyline(pts, color, width, dash);
}
function baseline(b, text) {
  ctx.strokeStyle = '#b9c2ba'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(b.left, b.bottom); ctx.lineTo(b.right, b.bottom); ctx.stroke();
  if (text) label(text, b.left, b.top + 2, C.muted, 10.5, 'left', 700);
}
function ticks(b) { for (let v = -4; v <= 4; v += 2) label(String(v).replace('-', '−'), b.X(v), b.bottom + 9, '#8b9690', 9.5, 'center', 500); }
// histogram(b, xs, ws, ymax): density-normalized (weighted) histogram in bins of 0.2.
function histogram(b, xs, ws, ymax, color) {
  const bw = .2, nb = Math.round((XMAX - XMIN) / bw), h = new Array(nb).fill(0);
  let total = 0;
  xs.forEach((x, i) => { const j = Math.floor((x - XMIN) / bw), w = ws ? ws[i] : 1; total += w; if (j >= 0 && j < nb) h[j] += w; });
  if (!total) return;
  ctx.fillStyle = color; ctx.globalAlpha = .32;
  h.forEach((c, j) => {
    const d = c / total / bw, hp = Math.min(d / ymax, 1.04) * (b.bottom - b.top);
    const x0 = b.X(XMIN + j * bw), x1 = b.X(XMIN + (j + 1) * bw);
    ctx.fillRect(x0 + .5, b.bottom - hp, x1 - x0 - 1, hp);
  });
  ctx.globalAlpha = 1;
}
const PI_MAX = (() => { let m = 0; for (let x = XMIN; x <= XMAX; x += .01) m = Math.max(m, S.piX(x)); return m; })();

function drawRejection() {
  const sm = st.sampler, top = band(26, H * .6), bot = band(H * .67, H - 20);
  const env = x => sm.M * S.qX(x, sm.m, sm.s);
  const envPeak = env(sm.m), ymax = Math.min(Math.max(envPeak, PI_MAX), 3.2 * PI_MAX) * 1.06;
  baseline(top, 'proposals (x, height) under the envelope');
  // where the envelope dips below π the sampler can no longer produce π
  ctx.fillStyle = C.rust; ctx.globalAlpha = .22;
  for (let i = 0; i < 240; i++) {
    const x = XMIN + (i + .5) / 240 * (XMAX - XMIN), p = S.piX(x), e = env(x);
    if (p > e) { const y0 = top.bottom - Math.min(p / ymax, 1) * (top.bottom - top.top), y1 = top.bottom - Math.min(e / ymax, 1) * (top.bottom - top.top); ctx.fillRect(top.X(x) - (top.right - top.left) / 480, y0, (top.right - top.left) / 240, y1 - y0); }
  }
  ctx.globalAlpha = 1;
  const shown = sm.draws.slice(-3000);
  shown.forEach(d => { if (d.u <= ymax * 1.04) dot(top.X(d.x), top.bottom - d.u / ymax * (top.bottom - top.top), d.ok ? 2.5 : 2.1, d.ok ? C.blue : '#d9a58c', d.ok ? .85 : .55); });
  curve(top, S.piX, ymax, C.rust, 2.2);
  curve(top, env, ymax, C.amber, 2, [6, 4]);
  if (envPeak > ymax) label(`M·q continues above (peak ${fmt(envPeak, 1)})`, top.X(sm.m) + 8, top.top + 14, C.amber, 10.5, 'left', 700);
  label('π', top.X(-1.35) - 16, top.bottom - S.piX(-1.35) / ymax * (top.bottom - top.top) - 8, C.rust, 13, 'center', 700);
  const kept = sm.draws.filter(d => d.ok).map(d => d.x);
  const hmax = PI_MAX * 1.5;
  baseline(bot, `kept draws (${kept.length}) as a histogram, against π`);
  histogram(bot, kept, null, hmax, C.blue);
  curve(bot, S.piX, hmax, C.rust, 2);
  ticks(bot);
}

function drawImportance() {
  const sm = st.sampler, top = band(26, H * .38), mid = band(H * .44, H * .6), bot = band(H * .67, H - 20);
  let qmax = 0; for (let x = XMIN; x <= XMAX; x += .01) qmax = Math.max(qmax, S.qX(x, sm.m, sm.s));
  const ymax = Math.max(PI_MAX, qmax) * 1.08;
  baseline(top, 'target π and proposal q');
  curve(top, S.piX, ymax, C.rust, 2.2);
  curve(top, x => S.qX(x, sm.m, sm.s), ymax, C.amber, 2, [6, 4]);
  sm.draws.slice(-3000).forEach(d => { ctx.fillStyle = C.amber; ctx.globalAlpha = .35; ctx.fillRect(top.X(d.x) - .5, top.bottom - 5, 1, 5); });
  ctx.globalAlpha = 1;
  baseline(mid, 'weights w = π/q (stem height relative to the largest)');
  const wmax = sm.draws.reduce((a, d) => Math.max(a, d.w), 1e-12);
  ctx.strokeStyle = C.blue; ctx.lineWidth = 1.2;
  sm.draws.slice(-3000).forEach(d => { ctx.globalAlpha = .55; ctx.beginPath(); ctx.moveTo(top.X(d.x), mid.bottom); ctx.lineTo(top.X(d.x), mid.bottom - d.w / wmax * (mid.bottom - mid.top - 12)); ctx.stroke(); });
  ctx.globalAlpha = 1;
  const hmax = PI_MAX * 1.5;
  baseline(bot, 'weighted histogram of the draws, against π');
  histogram(bot, sm.draws.map(d => d.x), sm.draws.map(d => d.w), hmax, C.blue);
  curve(bot, S.piX, hmax, C.rust, 2);
  ticks(bot);
}

function drawSMC() {
  const f = st.filter, w = f.world, h = f.history;
  const all = w.truth.concat(w.obs, f.exact.map(e => e.mean + 2 * e.sd), f.exact.map(e => e.mean - 2 * e.sd));
  const lo = Math.min(...all) - .3, hi = Math.max(...all) + .3;
  const left = 40, right = W - 14, top = 24, bottom = H * .72, eTop = H * .84, eBottom = H - 12;
  const T = x => left + x / S.SSM.T * (right - left), Y = v => bottom - (v - lo) / (hi - lo) * (bottom - top);
  ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
  for (let v = Math.ceil(lo); v <= hi; v++) { ctx.beginPath(); ctx.moveTo(left, Y(v)); ctx.lineTo(right, Y(v)); ctx.stroke(); label(String(v).replace('-', '−'), left - 8, Y(v), '#8b9690', 9.5, 'right', 500); }
  for (let t = 0; t <= S.SSM.T; t += 10) label('t = ' + t, T(t), bottom + 10, '#8b9690', 9.5, 'center', 500);
  if (h.length) {
    // exact posterior band and mean from the Kalman filter
    ctx.fillStyle = C.rust; ctx.globalAlpha = .09; ctx.beginPath();
    h.forEach((e, i) => { const x = T(e.t), y = Y(e.exact + 2 * e.exactSd); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    for (let i = h.length - 1; i >= 0; i--) ctx.lineTo(T(h[i].t), Y(h[i].exact - 2 * h[i].exactSd));
    ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1;
    // particles after weighting, dot area ∝ weight (r = 3 at w = 1/N)
    h.forEach(e => e.particles.forEach(p => dot(T(e.t), Y(p.x), Math.min(6, .8 + 2.2 * Math.sqrt(p.w * f.N)), C.blue, .22)));
    polyline(h.map(e => [T(e.t), Y(e.truth)]), C.ink, 1.6);
    h.forEach(e => ring(T(e.t), Y(e.y), 2.6, C.amber, 1.3, .9));
    polyline(h.map(e => [T(e.t), Y(e.mean)]), C.blue, 2.2);
    polyline(h.map(e => [T(e.t), Y(e.exact)]), C.rust, 2, [6, 4]);
  }
  // ESS strip with the resampling threshold and the steps that resampled
  ctx.strokeStyle = '#b9c2ba'; ctx.beginPath(); ctx.moveTo(left, eBottom); ctx.lineTo(right, eBottom); ctx.stroke();
  label('ESS / N after weighting' + (f.rule === 'half' ? ' · dashed: resampling threshold 1/2' : '') + ' · green: resampled', left, eTop - 9, C.muted, 10.5, 'left', 700);
  const bw = (right - left) / S.SSM.T * .6;
  h.forEach(e => {
    const hp = e.ess / f.N * (eBottom - eTop);
    ctx.fillStyle = e.resampled ? C.green : C.blue; ctx.globalAlpha = e.resampled ? .75 : .45;
    ctx.fillRect(T(e.t) - bw / 2, eBottom - hp, bw, hp);
  });
  ctx.globalAlpha = 1;
  if (f.rule === 'half') polyline([[left, (eTop + eBottom) / 2], [right, (eTop + eBottom) / 2]], C.green, 1.2, [4, 4]);
}

/* ---------------- readouts ---------------- */
function metric(name, value, note = '') { return `<div><dt>${name}</dt><dd>${value}${note ? ` <small>${note}</small>` : ''}</dd></div>`; }
function readouts() {
  let status = '', warn = false, rows = [];
  if (isChain) {
    const { run } = st, k = st.step, moves = run.moves.slice(0, k), pts = run.points.slice(0, k + 1);
    const acc = moves.filter(m => m.accepted).length, last = moves[k - 1];
    const work = moves.reduce((a, m) => a + m.evals, 0);
    const unit = { gibbs: ['conditional draw', 'conditional draws'], mh: ['density', 'densities'], hmc: ['gradient', 'gradients'], slice: ['density', 'densities'] }[demo][work === 1 ? 0 : 1];
    const cost = `${work} <small>${unit}</small>`;
    rows = [
      metric('Moves', `${k} <small>/ ${S.CHAIN.steps}</small>`),
      metric('Acceptance', k ? pct(acc / k) : '—', demo === 'gibbs' || demo === 'slice' ? 'by construction' : 'of proposals'),
      metric('Lag-1 ESS proxy', `${S.essLag1(pts.map(p => p.x))} <small>/ ${pts.length} points</small>`),
      metric('Work so far', cost)
    ];
    if (!last) status = 'Start at (−2.35, −2.05). Press Step.';
    else if (demo === 'gibbs') status = `Move ${k}: x${SUB[last.axis]} ← N(${fmt(last.mean)}, ${fmt(last.sd)}²) · accepted`;
    else if (demo === 'mh') status = `Move ${k}: α = ${fmt(last.alpha)} · ${last.accepted ? 'accepted' : 'rejected: the state repeats'}`;
    else if (demo === 'hmc') { status = `Move ${k}: ${S.CHAIN.hmc.L} leapfrog steps · ΔH = ${fmt(last.dH, 3)} · ${last.accepted ? 'accepted' : 'rejected'}`; warn = !last.accepted && Math.abs(last.dH) > 5; }
    else status = `Move ${k}: x${SUB[last.axis]} slice · ${last.misses.length} miss${last.misses.length === 1 ? '' : 'es'} before landing inside`;
    if (demo === 'hmc' && values.eps > 2 * Math.sqrt(1 - values.rho)) { status += ' · ε beyond the leapfrog stability limit'; warn = true; }
  } else if (demo === 'rejection') {
    const sm = st.sampler, s = sm.stats();
    rows = [
      metric('Proposals', String(s.n)),
      metric('Acceptance', s.n ? pct(s.acceptance) : '—', `1/M = ${pct(s.predicted)}`),
      metric('Kept draws', String(s.n ? Math.round(s.acceptance * s.n) : 0)),
      metric('Envelope M', fmt(sm.M), `sup π/q = ${fmt(sm.sup)}`)
    ];
    if (sm.slack < 1) { status = 'Envelope below π (shaded): kept draws follow min(π, M·q)'; warn = true; }
    else status = sm.slack === 1 ? 'Tightest envelope: every kept draw is exact and independent' : 'Looser envelope: still exact, more proposals wasted';
  } else if (demo === 'importance') {
    const s = st.sampler.stats(), ew2 = st.ew2;
    const asym = Number.isFinite(ew2) ? 1 / ew2 : 0;
    rows = [
      metric('Draws', String(s.n || 0)),
      metric('Kish ESS', s.n ? `${Math.round(s.ess)} <small>/ ${s.n}</small>` : '—', `→ ${asym < .005 ? '≈ 0%' : pct(asym)} of n`),
      metric('Estimate of E[x]', s.n ? `${fmt(s.est, 3)}` : '—', s.n ? `± ${fmt(s.se, 3)} · true ${fmt(S.TRUE_MEAN, 3)}` : ''),
      metric('Largest weight', s.n ? pct(s.maxShare) : '—', 'of the total')
    ];
    if (values.s <= .6 / Math.SQRT2) { status = 'q too narrow: E_q[w²] is infinite; a few draws carry the weight'; warn = true; }
    else if (values.s <= .6) { status = 'q narrower than a target bump: weights are unbounded'; warn = true; }
    else status = 'Every draw is kept; weights repair the q–π mismatch';
  } else {
    const f = st.filter, s = f.stats(), last = f.history[f.history.length - 1];
    rows = [
      metric('Time', `${f.t} <small>/ ${S.SSM.T}</small>`),
      metric('ESS now', last ? `${fmt(last.ess, 1)} <small>/ ${f.N}</small>` : '—'),
      metric('Resamples', f.t ? String(s.resamples) : '—', s.resamples ? `${pct(s.distinct)} distinct parents` : ''),
      metric('Error vs exact mean', f.t ? fmt(s.mcError) : '—', f.t ? `RMS · tracking ${fmt(s.rmse)} (exact ${fmt(s.exactRmse)})` : '')
    ];
    if (!last) status = 'Particles start from the prior N(0, 0.8²). Advance t.';
    else { status = `t = ${last.t} · ESS ${fmt(last.ess, 1)} / ${f.N}${last.resampled ? ` · resampled (${last.distinct} distinct parents)` : ''}`; warn = f.rule === 'never' && last.ess < 2; }
  }
  $('metrics').innerHTML = rows.join('');
  $('status').textContent = status;
  $('status').classList.toggle('warn', warn);
  $('step').disabled = atEnd();
}
function draw() {
  ctx.clearRect(0, 0, W, H);
  if (isChain) drawChain();
  else if (demo === 'rejection') drawRejection();
  else if (demo === 'importance') drawImportance();
  else drawSMC();
  readouts();
}

/* ---------------- boot ---------------- */
function init() {
  document.title = `${meta.title} · Sampling Lab · Bai Liping`;
  $('kicker').textContent = meta.kicker;
  $('method-title').textContent = meta.title;
  $('method-copy').textContent = meta.copy;
  $('stage-heading').textContent = meta.heading;
  $('step').textContent = meta.step;
  $('back-link').href = '../#' + demo;
  canvas.setAttribute('aria-label', meta.heading + ' plot');
  document.querySelectorAll('.method-tabs a').forEach(a => { if (a.dataset.demo === demo) a.setAttribute('aria-current', 'page'); });
  $('legend').innerHTML = meta.legend.map(([color, text, kind]) => `<span><i class="${kind}" style="background:${color};color:${color}"></i>${text}</span>`).join('');
  buildControls();
  build(isChain ? S.CHAIN.shown : meta.initial);

  $('step').addEventListener('click', () => { press(); draw(); if (atEnd()) stopAuto(); });
  $('auto').addEventListener('click', () => autoTimer ? stopAuto() : startAuto());
  $('reset').addEventListener('click', () => { stopAuto(); build(0); draw(); });
  $('reseed').addEventListener('click', () => { stopAuto(); seed = 1 + Math.floor(Math.random() * 1e9); build(progress()); draw(); });
  window.addEventListener('bento-live-visibility', e => { if (e.detail.paused) stopAuto(); });
  new ResizeObserver(resize).observe(canvas);
  resize();
}
init();
})();
