'use strict';
const assert = require('node:assert/strict');
const M = require('./model.js');

const close = (a, b, tolerance = 1e-9) => assert.ok(Math.abs(a - b) <= tolerance, `${a} != ${b}`);

const rotation = M.rotationExperiment(15, 7, -35);
close(rotation.exactDeterminant, 1, 1e-10);
assert.ok(rotation.exactOrthogonality < 1e-10);
assert.ok(rotation.eulerDeterminant > 1.4);
assert.ok(rotation.eulerOrthogonality > 0.5);

for (const degree of [1, 3]) {
  for (const t of [0, 0.13, 0.5, 0.87, 1]) {
    const weights = M.basisWeights(t, M.BASE_CONTROL_POINTS.length, degree);
    close(weights.reduce((a, b) => a + b, 0), 1, 1e-9);
    assert.ok(weights.filter(value => value > 1e-9).length <= degree + 1);
  }
}
const local = M.splineExperiment({ degree: 3, selected: 5, shift: 0.3, query: 0.54 });
assert.equal(local.active.length, 4);
assert.ok(local.influenceStart > 0.2 && local.influenceEnd < 0.8, 'a central cubic control should have bounded support');
close(local.curve[0][0], local.reference[0][0]);
close(local.curve[0][1], local.reference[0][1]);
close(local.curve.at(-1)[0], local.reference.at(-1)[0]);
close(local.curve.at(-1)[1], local.reference.at(-1)[1]);

const gp = M.gpExperiment({ processVariance: 0.18, measurementSigma: 0.22, query: 3.65 });
assert.equal(gp.mean.length, 8);
assert.ok(Number.isFinite(gp.queryMean) && gp.querySigma > 0);
assert.ok(gp.nonzeros <= 3 * gp.count - 2, 'information matrix should remain tridiagonal');
for (let i = 0; i < gp.count; i += 1) {
  assert.ok(gp.covariance[i][i] > 0);
  for (let j = 0; j < gp.count; j += 1) close(gp.covariance[i][j], gp.covariance[j][i], 1e-8);
}
const smooth = M.gpExperiment({ processVariance: 0.03, measurementSigma: 0.22 });
const flexible = M.gpExperiment({ processVariance: 0.9, measurementSigma: 0.22 });
assert.ok(smooth.roughness < flexible.roughness, 'larger process variance should permit a rougher estimate');

console.log('Advanced-state teaching models: all checks passed.');
