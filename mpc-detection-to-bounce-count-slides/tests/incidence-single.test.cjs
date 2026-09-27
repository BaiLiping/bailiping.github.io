const { test } = require('node:test');
const assert = require('node:assert/strict');
const model = require('../live/geometry/incidence-single-model.js');
const drawing = require('../live/geometry/incidence-single-render.js');
const near = (a, b, tolerance = 1e-9) => assert.ok(Math.abs(a - b) < tolerance, `${a} != ${b}`);

test('two measured bearings recover the independent triangle geometry', () => {
  const result = model.intersectRays([0, 0], [4, 0], 45, 135);
  assert.equal(result.status, 'ok');
  near(result.point[0], 2); near(result.point[1], 2);
  near(result.bsDistance, Math.sqrt(8)); near(result.ueDistance, Math.sqrt(8));
  near(result.length, 4 * Math.sqrt(2));
  near(result.wallNormal[0], 0); near(Math.abs(result.wallNormal[1]), 1);
});

test('delay is only a residual and cannot move the ray intersection or inferred wall', () => {
  const a = model.construction([0, 0], [4, 0], 45, 135, 4 * Math.sqrt(2));
  const b = model.construction([0, 0], [4, 0], 45, 135, 4 * Math.sqrt(2) + 3);
  assert.deepEqual(a.point, b.point); assert.deepEqual(a.wallNormal, b.wallNormal);
  near(a.residual, 0); near(b.residual, 3);
});

test('parallel, collinear, backward, and zero-length paths never fabricate a bounce', () => {
  for (const [bs, ue, aod, aoa, status] of [
    [[0, 0], [0, 2], 0, 0, 'parallel'],
    [[0, 0], [4, 0], 0, 0, 'collinear'],
    [[0, 0], [4, 0], 0, 180, 'collinear'],
    [[0, 0], [4, 0], 135, 45, 'behind'],
    [[0, 0], [4, 2], 0, 45, 'behind'],
    [[0, 0], [0, 2], 0, -90, 'behind'],
    [[0, 0], [4, 0], NaN, 45, 'invalid']
  ]) {
    const result = model.intersectRays(bs, ue, aod, aoa);
    assert.equal(result.status, status); assert.equal(result.point, undefined);
    assert.ok(!drawing.render({ bs, ue, result }).svg.match(/NaN|Infinity|<ellipse|virtual anchor/i));
  }
});

test('reference measurements across the draggable region satisfy wall incidence and reflection', () => {
  const { bs, wallPoint, wallTangent } = model.scene;
  const normal = [-wallTangent[1], wallTangent[0]];
  for (const x of [-4, 0, 6, 10]) for (const y of [-15, -10, -3]) {
    const ue = [x, y], reference = model.referenceMeasurements(bs, ue, wallPoint, wallTangent);
    const result = model.construction(bs, ue, reference.aod, reference.aoa, reference.length);
    assert.equal(result.status, 'ok'); near(result.residual, 0);
    near((result.point[0] - wallPoint[0]) * normal[0] + (result.point[1] - wallPoint[1]) * normal[1], 0);
    const dot = result.departure[0] * result.wallNormal[0] + result.departure[1] * result.wallNormal[1];
    for (let i = 0; i < 2; i++) near(result.departure[i] - 2 * dot * result.wallNormal[i], -result.arrival[i]);
  }
});

test('near-parallel forward rays remain finite and expose poor conditioning', () => {
  const result = model.intersectRays([0, 0], [0, -1], 0, .01);
  assert.equal(result.status, 'ok'); assert.ok(result.crossingAngle < 2);
  near(result.point[1], 0); assert.ok(result.point[0] > 5000);
  assert.ok(!drawing.render({ bs: [0, 0], ue: [0, -1], result }).svg.match(/NaN|Infinity/));
});
