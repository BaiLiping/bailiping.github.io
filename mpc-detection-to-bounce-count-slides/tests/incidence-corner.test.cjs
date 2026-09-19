const { test } = require('node:test');
const assert = require('node:assert/strict');
const model = require('../live/geometry/incidence-corner-model.js');
const drawing = require('../live/geometry/incidence-corner-render.js');
const near = (a, b, tolerance = 1e-8) => assert.ok(Math.abs(a - b) < tolerance, `${a} != ${b}`);
const pointNear = (a, b) => a.forEach((v, i) => near(v, b[i]));
const sub = (a, b) => a.map((v, i) => v - b[i]);
const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
const unit = v => v.map(x => x / Math.hypot(...v));
const degrees = r => r * 180 / Math.PI;
const bs = [0, 0], ue = [4, 0];
// Independently specified physical route: A is y=4, B is x=10.
// Single hit (2,4); double hits (8,4) and (10,3).
const one = { aod: degrees(Math.atan2(4, 2)), aoa: degrees(Math.atan2(4, -2)), length: Math.sqrt(80) };
const two = { aod: degrees(Math.atan2(4, 8)), aoa: degrees(Math.atan2(3, 6)), length: Math.sqrt(320) };

test('single-bounce wall determines two distinct double-bounce hits from measured bearings', () => {
  const r = model.construction(bs, ue, one, two);
  assert.equal(r.status, 'ok');
  pointNear(r.single.point, [2, 4]); pointNear(r.p1, [8, 4]); pointNear(r.p2, [10, 3]);
  near(dot(sub(r.p1, r.wallA.point), r.wallA.normal), 0);
  near(Math.abs(r.wallA.normal[1]), 1); near(Math.abs(r.wallB.normal[0]), 1);
  near(r.single.residual, 0); near(r.residual, 0);
  assert.notDeepEqual(r.p1, r.single.point);
});

test('the construction respects global rotations and translations of known poses', () => {
  const theta = .71, shift = [-7, 13];
  const transform = p => [p[0] * Math.cos(theta) - p[1] * Math.sin(theta) + shift[0], p[0] * Math.sin(theta) + p[1] * Math.cos(theta) + shift[1]];
  const rotateMeasurement = m => ({ ...m, aod: m.aod + degrees(theta), aoa: m.aoa + degrees(theta) });
  const r = model.construction(transform(bs), transform(ue), rotateMeasurement(one), rotateMeasurement(two));
  assert.equal(r.status, 'ok'); pointNear(r.p1, transform([8, 4])); pointNear(r.p2, transform([10, 3]));
  near(r.residual, 0);
});

test('both delays are checks only; neither delay moves either wall or incidence point', () => {
  const a = model.construction(bs, ue, one, two);
  const b = model.construction(bs, ue, { ...one, length: one.length + 2 }, { ...two, length: two.length - 3 });
  for (const key of ['wallA', 'wallB', 'p1', 'p2', 'reflected']) assert.deepEqual(a[key], b[key]);
  near(b.single.residual, 2); near(b.residual, -3);
});

test('changing the single-path bearing propagates through the first wall to the second point', () => {
  const a = model.construction(bs, ue, one, two);
  const b = model.construction(bs, ue, { ...one, aod: one.aod + 2 }, two);
  assert.equal(b.status, 'ok');
  assert.ok(Math.hypot(...sub(a.p2, b.p2)) > .01);
  assert.ok(Math.hypot(...sub(a.wallA.normal, b.wallA.normal)) > .001);
  near(dot(sub(b.p1, b.wallA.point), b.wallA.normal), 0);
  pointNear(unit(sub(b.p2, b.p1)), b.reflected);
});

test('parallel, backward, and unresolved paths do not invent the missing hit', () => {
  const cases = [
    [{ ...one, aoa: one.aod }, two, 'first-wall-unresolved'],
    [one, { ...two, aod: 0 }, 'parallel-wall'],
    [one, { ...two, aod: -10 }, 'behind-wall'],
    [one, { ...two, aoa: -two.aod }, 'second-parallel'],
    [one, { ...two, aod: one.aod, aoa: one.aoa }, 'second-collinear'],
    [one, { ...two, aoa: 180 }, 'second-behind'],
    [one, { ...two, aod: NaN }, 'invalid']
  ];
  for (const [first, second, status] of cases) {
    const r = model.construction(bs, ue, first, second);
    assert.equal(r.status, status); assert.equal(r.p2, undefined);
    for (const step of [1, 2, 3]) assert.doesNotMatch(drawing.render({ bs, ue, result: r, step }).svg, /NaN|Infinity|<ellipse|virtual anchor/i);
  }
});

test('draggable scene satisfies both wall incidence and specular reflection throughout the region', () => {
  const { bs, wallA, wallB } = model.scene;
  for (const x of [14, 16, 18, 21]) for (const y of [-21, -17, -13]) {
    const ue = [x, y], reference = model.referenceMeasurements(bs, ue);
    const r = model.construction(bs, ue, reference.single, reference.double);
    assert.equal(r.status, 'ok'); near(r.single.residual, 0); near(r.residual, 0);
    for (const [p, wall] of [[r.single.point, wallA], [r.p1, wallA], [r.p2, wallB]]) {
      near(dot(sub(p, wall.point), [-wall.tangent[1], wall.tangent[0]]), 0);
    }
    const route = [bs, r.p1, r.p2, ue];
    for (const i of [1, 2]) {
      const incoming = unit(sub(route[i], route[i - 1])), outgoing = unit(sub(route[i + 1], route[i]));
      const n = (i === 1 ? r.wallA : r.wallB).normal;
      pointNear(incoming.map((v, j) => v - 2 * dot(incoming, n) * n[j]), outgoing);
    }
    for (const step of [1, 2, 3]) assert.doesNotMatch(drawing.render({ bs, ue, result: r, step, reference: { ...reference, wallA, wallB }, showReference: true }).svg, /NaN|Infinity|<ellipse|virtual anchor/i);
  }
});
