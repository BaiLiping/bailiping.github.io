'use strict';
// Numerical checks run the deployed model (sampling-playground/model.js), the code the live labs execute.
const path = require('node:path');
const assert = require('node:assert/strict');
const {test} = require('node:test');
const S = require(path.resolve(__dirname, '../sampling-playground/model.js'));

const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
function moments(points) {
  const xs = points.map(p => p.x), ys = points.map(p => p.y), mx = mean(xs), my = mean(ys);
  const vx = mean(xs.map(v => (v - mx) ** 2)), vy = mean(ys.map(v => (v - my) ** 2));
  const c = mean(xs.map((v, i) => (v - mx) * (ys[i] - my))) / Math.sqrt(vx * vy);
  return {mx, my, vx, vy, c};
}

test('all four chain samplers leave the correlated Gaussian invariant', () => {
  for (const method of ['gibbs', 'mh', 'hmc', 'slice']) {
    const run = S.runChain(method, {steps: 41000});
    const m = moments(run.points.slice(1001));
    assert.ok(Math.abs(m.mx) < .12 && Math.abs(m.my) < .12, `${method} mean ${m.mx}, ${m.my}`);
    assert.ok(Math.abs(m.vx - 1) < .1 && Math.abs(m.vy - 1) < .1, `${method} variance ${m.vx}, ${m.vy}`);
    assert.ok(Math.abs(m.c - S.CHAIN.rho) < .02, `${method} correlation ${m.c}`);
  }
});

test('the default chains reproduce the original page on its 72-move horizon', () => {
  const acc = m => {const r = S.runChain(m, {steps: 72}); return r.moves.filter(v => v.accepted).length / 72;};
  const ess = m => S.essLag1(S.runChain(m, {steps: 72}).points.map(p => p.x));
  assert.equal(acc('gibbs'), 1);
  assert.equal(Math.round(100 * acc('mh')), 42);
  assert.equal(Math.round(100 * acc('hmc')), 99);
  assert.deepEqual(['gibbs', 'mh', 'hmc'].map(ess), [4, 4, 28]);
});

test('move records are consistent with each algorithm', () => {
  const mh = S.runChain('mh');
  for (const m of mh.moves) assert.deepEqual(m.to, m.accepted ? m.proposal : m.from);
  const hmc = S.runChain('hmc');
  for (const m of hmc.moves) {assert.equal(m.path.length, S.CHAIN.hmc.L + 1); assert.equal(m.evals, S.CHAIN.hmc.L);}
  const slice = S.runChain('slice');
  for (const m of slice.moves) {
    assert.ok(S.logPi(m.to, S.CHAIN.rho) >= m.level, 'slice move landed outside its slice');
    const v = m.axis === 'x' ? m.to.x : m.to.y;
    assert.ok(v >= m.final[0] - 1e-12 && v <= m.final[1] + 1e-12);
    for (const miss of m.misses) assert.ok(miss >= m.bracket[0] && miss <= m.bracket[1]);
  }
  const gibbs = S.runChain('gibbs');
  gibbs.moves.forEach((m, i) => assert.equal(m.axis, i % 2 ? 'y' : 'x'));
});

test('HMC energy error shrinks with the leapfrog step and diverges past the stability limit', () => {
  const meanAbsDH = eps => mean(S.runChain('hmc', {eps, steps: 200}).moves.map(m => Math.abs(m.dH)));
  assert.ok(meanAbsDH(.04) < meanAbsDH(.16) && meanAbsDH(.16) < meanAbsDH(.5));
  const limit = 2 * Math.sqrt(1 - S.CHAIN.rho);
  const beyond = S.runChain('hmc', {eps: limit * 1.1, steps: 100});
  assert.ok(beyond.moves.filter(m => m.accepted).length < 10, 'leapfrog should be unstable beyond 2·sqrt(1−ρ)');
});

test('Geyer ESS agrees with the AR(1) value for an AR(1) series', () => {
  const r = S.rng(5), a = .6, xs = [0];
  for (let i = 1; i < 50000; i++) xs.push(a * xs[i - 1] + Math.sqrt(1 - a * a) * S.randn(r));
  const expected = xs.length * (1 - a) / (1 + a);
  assert.ok(Math.abs(S.essGeyer(xs) / expected - 1) < .1);
});

test('rejection envelope is tight and acceptance matches 1/M', () => {
  const sup = S.supRatio();
  for (let x = -6; x <= 6; x += .001) assert.ok(S.piX(x) <= sup * S.qX(x) * (1 + 1e-9));
  const run = S.makeRejection().draw(40000).stats();
  assert.ok(Math.abs(run.acceptance - 1 / sup) < .01, `${run.acceptance} vs ${1 / sup}`);
  assert.equal(S.supRatio(0, .6), Infinity);
  // Kept draws are exact: their mean matches the mixture mean.
  const kept = S.makeRejection().draw(40000).draws.filter(d => d.ok).map(d => d.x);
  assert.ok(Math.abs(mean(kept) - S.TRUE_MEAN) < .03);
  // A broken envelope clips the peaks, so the kept draws no longer follow π (Kolmogorov–Smirnov distance).
  const clipped = S.makeRejection({slack: .6}).draw(40000).draws.filter(d => d.ok).map(d => d.x);
  const grid = [], cdf = [];
  let acc = 0;
  for (let x = -6; x <= 6; x += .001) {acc += S.piX(x) * .001; grid.push(x); cdf.push(acc);}
  const ks = a => {
    const s = [...a].sort((p, q) => p - q);
    let d = 0, j = 0;
    s.forEach((x, i) => {while (j < grid.length - 1 && grid[j] < x) j++; d = Math.max(d, Math.abs((i + 1) / s.length - cdf[j]));});
    return d;
  };
  assert.ok(ks(kept) < .02, `exact draws KS ${ks(kept)}`);
  assert.ok(ks(clipped) > .04, `clipped draws KS ${ks(clipped)}`);
});

test('importance sampling: weight moments, Kish ESS and calibrated standard errors', () => {
  assert.ok(Math.abs(1 / S.weightSecondMoment() - .652) < .002);
  assert.equal(S.weightSecondMoment(0, .4), Infinity);
  const big = S.makeImportance().draw(100000).stats();
  assert.ok(Math.abs(big.ess / big.n - 1 / S.weightSecondMoment()) < .02);
  assert.ok(Math.abs(big.meanWeight - 1) < .01, 'unnormalized weights should average Z = 1');
  let outside = 0;
  for (let seed = 1; seed <= 400; seed++) {
    const s = S.makeImportance({seed}).draw(1000).stats();
    if (Math.abs(s.est - S.TRUE_MEAN) / s.se > 2) outside++;
  }
  assert.ok(outside / 400 > .02 && outside / 400 < .09, `coverage outside ±2 SE: ${outside / 400}`);
});

test('particle filter converges to the exact Kalman posterior', () => {
  const kf = S.kalman(S.makeWorld().obs);
  assert.ok(Math.abs(kf[kf.length - 1].sd - S.kalmanSteadySd()) < 1e-6);
  const errors = [44, 400, 4000].map(N => S.makeFilter({N}).run().stats().mcError);
  assert.ok(errors[0] > errors[1] && errors[1] > errors[2] && errors[2] < .02, errors.join(', '));
  const never = S.makeFilter({rule: 'never'}).run().stats();
  assert.ok(never.ess < 1.5 && never.mcError > errors[0]);
  const always = S.makeFilter({rule: 'always'}).run().stats();
  assert.equal(always.resamples, S.SSM.T);
});

test('deck: numbers, live pairs and contents are generated from the model', async () => {
  const {deck, inlineLiveMap, numbers} = await import('../sampling-playground/bento-deck.mjs');
  const ids = deck.slides.map(s => s.id);
  assert.equal(inlineLiveMap.length, 7);
  for (const e of inlineLiveMap) assert.equal(ids.indexOf(e.introSlide), e.slideIndex - 1);
  const cover = deck.slides[0].elements.find(e => e.id === 'cover-boundary').html;
  assert.ok(cover.startsWith(`${deck.slides.length} SLIDES · 7 LIVE LABS`));
  const html = JSON.stringify(deck);
  assert.ok(html.includes(`M=\\\\sup\\\\pi/q=${numbers.rej.sup.toFixed(2)}`));
  assert.ok(numbers.long.hmc.essPerK > numbers.long.gibbs.essPerK && numbers.long.gibbs.essPerK > numbers.long.mh.essPerK);
  assert.ok(!/\\\\[\(\[][^\\]*<[A-Za-z]/.test(html), 'a < inside math would be parsed as an HTML tag');
});
