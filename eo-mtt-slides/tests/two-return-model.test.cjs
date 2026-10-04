const { test } = require('node:test');
const assert = require('node:assert/strict');
const M = require('../live/two-return-model.js');
const near = (a, b, eps = 1e-7) => assert.ok(Math.abs(a - b) < eps, a + ' != ' + b);
const sum = xs => xs.reduce((a, b) => a + b, 0);
const simpson = (f, a, b, n = 4000) => {
  const h = (b - a) / n; let s = f(a) + f(b);
  for (let i = 1; i < n; i++) s += (i % 2 ? 4 : 2) * f(a + i * h);
  return s * h / 3;
};

test('closed-form source convolution agrees with independent source integration', () => {
  for (const [z, c, L, noise] of [[0, 0, 6, .3], [2.9, 0, 6, .3], [4, 0, 6, .6], [-2.8, 0, 6, .1], [.3, .5, 1, 1]]) {
    const integral = simpson(u => M.gaussian(z, c + u, noise) / L, -L / 2, L / 2);
    near(M.likelihood(z, c, L, noise), integral, 7e-8);
  }
});

test('likelihood is normalized and its zero-extent limit is the point measurement model', () => {
  for (const [L, noise] of [[1, .1], [6, .3], [8, 1]]) {
    near(simpson(z => M.likelihood(z, .7, L, noise), .7 - L / 2 - 8 * noise, .7 + L / 2 + 8 * noise), 1, 2e-7);
  }
  for (const z of [-2, 0, 1, 4]) {
    near(M.likelihood(z, .4, 0, .6), M.gaussian(z, .4, .6), 1e-12);
    near(M.likelihood(z, .4, 1e-8, .6), M.gaussian(z, .4, .6), 1e-12);
  }
});

test('finite-body support uses both returns, including their separation', () => {
  const pair = [1, 3];
  for (let c = -2; c <= 6; c += .05) {
    assert.equal(M.likelihood(pair[0], c, 1.5, 0) * M.likelihood(pair[1], c, 1.5, 0), 0);
  }
  near(M.likelihood(1, 2, 4, 0) * M.likelihood(3, 2, 4, 0), 1 / 16);
  assert.equal(M.likelihood(3, 0, 4, 0), 0);
});

test('joint posterior and both marginals normalize; center and extent remain dependent', () => {
  const p = M.example().posterior;
  near(sum(p.joint), 1, 1e-10);
  near(sum(p.center.posterior) * p.dx, 1, 1e-10);
  near(sum(p.extent.posterior.map((v, j) => v * p.lengths[j])) * p.dq, 1, 1e-10);
  near(sum(p.center.prior) * p.dx, 1, 1e-10);
  near(sum(p.extent.prior.map((v, j) => v * p.lengths[j])) * p.dq, 1, 1e-10);
  assert.ok(p.correlation < -.3, 'the posterior must retain center-size coupling');
  assert.ok(p.center.interval[1] - p.center.interval[0] > 2, 'two returns should not imply a known center');
  assert.ok(p.extent.interval[1] - p.extent.interval[0] > 5, 'two returns should not imply a known size');
  // Each plotted marginal must obey its own prior times the nuisance-integrated likelihood.
  for (let i = 0; i < p.nx; i += 19) near(p.center.posterior[i], p.center.prior[i] * p.center.likelihood[i] / p.evidence, 1e-10);
  for (let j = 0; j < p.nl; j += 11) near(p.extent.posterior[j], p.extent.prior[j] * p.extent.likelihood[j] / p.evidence, 1e-10);
});

test('equal-centroid pairs with different separation infer different sizes', () => {
  const a = M.infer({ detections: [-.3, .3], noise: .15 });
  const b = M.infer({ detections: [-2, 2], noise: .15 });
  near(a.center.mean, 0, 1e-10); near(b.center.mean, 0, 1e-10);
  assert.ok(b.extent.mean > a.extent.mean + 2);
});

test('inference is translation/reflection equivariant and has no truth inputs', () => {
  const run = M.example(), p = run.posterior;
  const manual = M.infer({ detections: run.detections, noise: run.settings.noise, priorMean: run.settings.priorMean, priorSigma: run.settings.priorSigma, extentMedian: run.settings.extentMedian });
  near(manual.center.mean, p.center.mean, 1e-12); near(manual.extent.mean, p.extent.mean, 1e-12);
  const shifted = M.infer({ detections: run.detections.map(z => z + 3), priorMean: 3, noise: .3 });
  near(shifted.center.mean, p.center.mean + 3, 1e-9); near(shifted.extent.mean, p.extent.mean, 1e-9);
  const mirrored = M.example({ side: -1 }).posterior;
  near(mirrored.center.mean, -p.center.mean, 1e-9);
  near(mirrored.extent.mean, p.extent.mean, 1e-9);
  near(mirrored.correlation, -p.correlation, 1e-9);
});

test('illustrated pair stays strictly within one half of the physical body', () => {
  for (const trueLength of [1, 6, 8]) for (const trueCenter of [-3, 3]) for (const side of [-1, 1]) {
    const a = M.example({ trueLength, trueCenter, side, noise: .1 });
    for (const z of a.detections) assert.ok(side * (z - trueCenter) > 0 && side * (z - trueCenter) < trueLength / 2);
    assert.equal(a.posterior.extentMedian, M.extentPrior.median);
  }
  assert.deepEqual(M.example({ noise: .1 }).detections, M.example({ noise: 1 }).detections);
});

test('refined quadrature agrees with the interactive grid, including a narrow-noise case', () => {
  for (const parameters of [{}, { priorSigma: 2.5, noise: .1, trueLength: 1 }, { extentMedian: 1, trueLength: 8, priorMean: -3, priorSigma: .3, trueCenter: 3, noise: .1 }]) {
    const a = M.example(parameters).posterior, b = M.example(parameters, { resolution: 2 }).posterior;
    for (const key of ['center', 'extent']) {
      near(a[key].mean, b[key].mean, .004); near(a[key].sigma, b[key].sigma, .006);
      a[key].interval.forEach((v, i) => near(v, b[key].interval[i], .03));
    }
    near(a.correlation, b.correlation, .002);
  }
});

test('control extremes keep the posterior finite and inside the integration domain', () => {
  for (const parameters of [
    { priorMean: -3, priorSigma: .3, extentMedian: 1, trueLength: 1, trueCenter: -3, noise: .1 },
    { priorMean: 3, priorSigma: 2.5, extentMedian: 8, trueLength: 8, trueCenter: 3, noise: 1 },
    { priorMean: 3, priorSigma: .3, extentMedian: 1, trueLength: 8, trueCenter: -3, noise: .1 },
    { priorMean: -3, priorSigma: .3, extentMedian: 8, trueLength: 1, trueCenter: 3, noise: 1 }
  ]) {
    const p = M.example(parameters).posterior;
    for (const key of ['center', 'extent']) {
      assert.ok(Number.isFinite(p[key].mean)); assert.ok(p[key].sigma > 0);
      assert.ok(p[key].interval[0] < p[key].interval[1]);
      assert.ok(p[key].posterior.every(v => Number.isFinite(v) && v >= 0));
    }
    assert.ok(p.edgeMass < 1e-8); assert.ok(Math.abs(p.correlation) <= 1);
  }
  assert.throws(() => M.infer({ detections: [1], noise: .3 }), RangeError);
  assert.throws(() => M.example({ trueLength: 0 }), RangeError);
  assert.throws(() => M.example({ noise: 0 }), RangeError);
});
