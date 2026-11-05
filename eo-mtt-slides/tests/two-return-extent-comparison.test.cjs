const test = require('node:test');
const assert = require('node:assert/strict');
const model = require('../live/two-return-model.js');
const view = require('../live/two-return-view.js');
const runs = [1, 2, 4, 6, 8].map(trueLength => model.example({ trueLength }));

test('changing true extent leaves the prior and display axes fixed, while both marginal updates change', () => {
  for (const kind of ['center', 'extent']) {
    const data = runs.map(run => view.curveData(run, kind));
    const priorPaths = runs.map(run => view.densitySVG(run, kind).match(new RegExp('class="' + kind + '-prior" d="([^"]+)"'))[1]);
    for (let i = 1; i < runs.length; i++) {
      assert.deepEqual(data[i].prior, data[0].prior);
      assert.deepEqual(data[i].xs, data[0].xs);
      assert.equal(data[i].ymax, data[0].ymax);
      assert.equal(priorPaths[i], priorPaths[0]);
      assert.notDeepEqual(data[i].likelihood, data[i - 1].likelihood);
      assert.notDeepEqual(data[i].posterior, data[i - 1].posterior);
      assert.ok(runs[i].posterior[kind].mean > runs[i - 1].posterior[kind].mean);
    }
  }
});

test('likelihood display has unit area over a common interval and preserves prior-times-likelihood shape', () => {
  for (const run of runs) for (const kind of ['center', 'extent']) {
    const d = view.curveData(run, kind), integrate = values => d.step * values.reduce((sum, v, i) => sum + v * (i === 0 || i === values.length - 1 ? .5 : 1), 0);
    assert.ok(Math.abs(integrate(d.likelihood) - 1) < 1e-12);
    const product = d.prior.map((v, i) => v * d.likelihood[i]), scale = integrate(d.posterior) / integrate(product);
    assert.ok(Math.max(...product.map((v, i) => Math.abs(scale * v - d.posterior[i]))) < 0.001);
    for (const curve of ['prior', 'likelihood', 'posterior']) assert.ok(Math.max(...d[curve]) < d.ymax, kind + ': curves fit fixed vertical limits');
    if (kind === 'center') assert.ok(Math.abs(d.likelihoodMode - (run.detections[0] + run.detections[1]) / 2) < d.step);
  }
});

test('the physical drawing grows proportionally with true extent, keeping both detections in the right half', () => {
  const widths = runs.map(run => {
    for (const z of run.detections) assert.ok(z > 0 && z < run.settings.trueLength / 2);
    const svg = view.bodySVG(run), rect = svg.match(/id="true-body"[^>]+/)[0];
    assert.equal((svg.match(/class="target-detection"/g) || []).length, 2);
    return Number(rect.match(/width="([^"]+)"/)[1]);
  });
  for (let i = 1; i < runs.length; i++) assert.ok(Math.abs(widths[i] / widths[0] - runs[i].settings.trueLength) < 0.003);
});
