const { test } = require('node:test');
const assert = require('node:assert/strict');
const { posterior, displayReturns, gaussian } = require('../live/extent-model.js');
const near = (a, b, tolerance = 1e-10) => assert.ok(Math.abs(a - b) < tolerance, `${a} != ${b}`);

test('centroid update equals eight individual Gaussian measurement updates', () => {
  for (const extentSigma of [0, 0.05, 1.5, 3]) for (const noiseSigma of [0.2, 0.75, 3]) {
    const result = posterior({ extentSigma, noiseSigma });
    const detections = displayReturns(result);
    let mean = result.priorMean, variance = result.priorVar;
    for (const z of detections) {
      const k = variance / (variance + result.returnVar);
      mean += k * (z - mean);
      variance *= 1 - k;
    }
    near(mean, result.postMean);
    near(variance, result.postVar);
    near(detections.reduce((sum, x) => sum + x, 0) / 8, result.centroid);
  }
});

test('larger extent lowers gain and increases center uncertainty at fixed count', () => {
  let previous = posterior({ extentSigma: 0 });
  for (const extentSigma of [0.05, 0.5, 1, 1.5, 2, 3]) {
    const next = posterior({ extentSigma });
    assert.ok(next.gain < previous.gain);
    assert.ok(next.postVar > previous.postVar);
    assert.ok(Math.abs(next.postMean - next.priorMean) < Math.abs(previous.postMean - previous.priorMean));
    assert.ok(next.postVar < next.priorVar);
    previous = next;
  }
});

test('zero extent and one return recover the scalar point-measurement update', () => {
  const result = posterior({ extentSigma: 0, count: 1 });
  near(result.gain, result.priorVar / (result.priorVar + result.noiseSigma ** 2));
  near(result.postMean, 1.321698113207547);
  near(posterior({ extentSigma: 0 }).centroidVar, result.noiseSigma ** 2 / 8);
});

test('independent additional returns reduce center uncertainty', () => {
  const one = posterior({ count: 1 }), eight = posterior({ count: 8 });
  near(eight.centroidVar, one.centroidVar / 8);
  assert.ok(eight.postVar < one.postVar);
});

test('normalized prior times full cloud likelihood has the calculated posterior moments', () => {
  const result = posterior();
  const detections = displayReturns(result);
  const dx = 0.002;
  let mass = 0, first = 0, second = 0;
  for (let x = -12; x <= 12; x += dx) {
    let density = gaussian(x, result.priorMean, result.priorVar);
    for (const z of detections) density *= gaussian(z, x, result.returnVar);
    mass += density * dx; first += x * density * dx; second += x * x * density * dx;
  }
  near(first / mass, result.postMean, 1e-8);
  near(second / mass - (first / mass) ** 2, result.postVar, 1e-8);
});

test('invalid model parameters are rejected', () => {
  for (const p of [{ count: 0 }, { count: 1.5 }, { extentSigma: -1 }, { noiseSigma: 0 }, { priorSigma: 0 }, { centroid: NaN }]) assert.throws(() => posterior(p), RangeError);
});
