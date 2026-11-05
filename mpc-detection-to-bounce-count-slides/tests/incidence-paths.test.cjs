const { test } = require('node:test');
const assert = require('node:assert/strict');
const m = require('../live/geometry/incidence-paths-model.js');
const drawing = require('../live/geometry/incidence-paths-render.js');
const single = require('../live/geometry/incidence-single-model.js');
const near = (a, b, t = 1e-7) => assert.ok(Math.abs(a - b) < t, `${a} != ${b}`);
const pointNear = (a, b) => a.forEach((v, i) => near(v, b[i]));
const sub = (a, b) => a.map((v, i) => v - b[i]);
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const unit = v => v.map(x => x / Math.hypot(...v));
const bearing = v => Math.atan2(v[1], v[0]) * 180 / Math.PI;
function measurement(bs, ue, hits) {
  const nodes = [bs, ...hits, ue];
  return { aod: bearing(sub(hits[0], bs)), aoa: bearing(sub(hits.at(-1), ue)), length: nodes.slice(1).reduce((s, q, i) => s + Math.hypot(...sub(q, nodes[i])), 0) };
}

test('three associated MPCs recover independently specified A/B/C incidence points', () => {
  const bs = [0, 0], ue = [2, -2.5];
  const hits = [[[16 / 21, 4]], [[48 / 7, 4], [10, 13 / 6]], [[16 / 3, 4], [10, .5], [4, -4]]];
  const zs = hits.map(h => measurement(bs, ue, h));
  const r = m.reconstruct(bs, ue, zs);
  assert.equal(r.status, 'ok'); assert.equal(r.walls.length, 3);
  for (let j = 0; j < 3; j++) {
    r.paths[j].hits.forEach((p, i) => pointNear(p, hits[j][i])); near(r.paths[j].residual, 0);
  }
  near(r.paths[2].length, 22.5);
  near(Math.abs(r.walls[0].normal[1]), 1); near(Math.abs(r.walls[1].normal[0]), 1); near(Math.abs(r.walls[2].normal[1]), 1);
});

test('parallel endpoint rays in a two-bounce corridor become identifiable after reflecting on inferred R', () => {
  const bs = [0, 0], ue = [2, 6];
  const a = measurement(bs, ue, [[6, 3.6]]), b = measurement(bs, ue, [[6, 18 / 11], [-4, 48 / 11]]);
  assert.equal(single.intersectRays(bs, ue, b.aod, b.aoa).status, 'parallel');
  const r = m.reconstruct(bs, ue, [a, b], { corridor: true });
  assert.equal(r.status, 'ok'); assert.equal(r.consistent, true); near(r.parallelError, 0);
  pointNear(r.paths[1].hits[0], [6, 18 / 11]); pointNear(r.paths[1].hits[1], [-4, 48 / 11]);
});

test('corridor triple path returns to the original wall; a shifted third wall is rejected', () => {
  const bs = [0, 0], ue = [2, 6];
  const zs = [measurement(bs, ue, [[6, 3.6]]), measurement(bs, ue, [[6, 18 / 11], [-4, 48 / 11]]), measurement(bs, ue, [[6, 1.2], [-4, 3.2], [6, 5.2]])];
  const r = m.reconstruct(bs, ue, zs, { corridor: true, repeated: true });
  assert.equal(r.status, 'ok'); assert.equal(r.consistent, true); assert.equal(r.walls.length, 2);
  pointNear(r.paths[2].hits[2], [6, 5.2]); near(r.closure.distance, 0); near(r.closure.orientation, 0, 2e-6);
  const changed = zs.map(z => ({ ...z })); changed[2].aoa += 4;
  const bad = m.reconstruct(bs, ue, changed, { corridor: true, repeated: true });
  assert.equal(bad.status, 'ok'); assert.equal(bad.consistent, false);
  assert.ok(bad.closure.distance > .15 || bad.closure.orientation > 2);
  assert.equal(bad.walls.length, 2, 'do not invent a third corridor wall');
});

test('all three delays remain independent checks; prefix bearing errors propagate to later hits', () => {
  const s = m.scenes.corner3, zs = m.referenceMeasurements(s);
  const a = m.reconstruct(s.bs, s.ue, zs);
  const b = m.reconstruct(s.bs, s.ue, zs.map((z, i) => ({ ...z, length: z.length + i + 1 })));
  assert.deepEqual(a.walls, b.walls);
  for (let i = 0; i < 3; i++) { assert.deepEqual(a.paths[i].hits, b.paths[i].hits); near(b.paths[i].residual, i + 1); }
  const c = m.reconstruct(s.bs, s.ue, zs.map((z, i) => ({ ...z, aod: z.aod + (i === 1 ? 1 : 0) })));
  assert.equal(c.status, 'ok'); assert.ok(Math.hypot(...sub(a.paths[2].hits[2], c.paths[2].hits[2])) > .1);
});

test('draggable regions preserve physical reflection and incidence for every reference path', () => {
  for (const [key, s] of Object.entries(m.scenes)) {
    for (const x of [s.bounds[0], s.ue[0], s.bounds[1]]) for (const y of [s.bounds[2], s.ue[1], s.bounds[3]]) {
      const ue = [x, y], zs = m.referenceMeasurements(s, ue);
      assert.ok(zs.every(z => z.status === 'ok'), `${key} at ${ue}`);
      const r = m.reconstruct(s.bs, ue, zs, { corridor: key.startsWith('corridor'), repeated: key === 'corridor3' });
      assert.equal(r.status, 'ok'); assert.equal(r.consistent, true);
      for (let j = 0; j < zs.length; j++) {
        const route = [s.bs, ...r.paths[j].hits, ue]; near(r.paths[j].residual, 0);
        for (let i = 1; i < route.length - 1; i++) {
          const w = s.walls[s.routes[j][i - 1]];
          near(dot(sub(route[i], w.point), w.normal), 0);
          const incoming = unit(sub(route[i], route[i - 1])), outgoing = unit(sub(route[i + 1], route[i]));
          pointNear(incoming.map((v, k) => v - 2 * dot(incoming, w.normal) * w.normal[k]), outgoing);
        }
      }
      const svg = drawing.render({ bs: s.bs, ue, result: r, view: { path: zs.length - 1, hits: zs.length, arrival: true, walls: r.walls.length }, names: key.startsWith('corridor') ? ['R', 'L'] : ['A', 'B', 'C'] }).svg;
      assert.doesNotMatch(svg, /NaN|Infinity|<ellipse|virtual anchor/i);
    }
  }
});

test('failed first or later forward intersections stop the dependent wall construction', () => {
  const s = m.scenes.corner3, zs = m.referenceMeasurements(s);
  for (const [path, changes] of [[0, { aoa: zs[0].aod }], [1, { aod: -20 }], [2, { aod: -10 }], [2, { aod: NaN }]]) {
    const r = m.reconstruct(s.bs, s.ue, zs.map((z, i) => i === path ? { ...z, ...changes } : z));
    assert.equal(r.status, 'unresolved'); assert.equal(r.failedPath, path); assert.equal(r.walls.length, path);
    assert.doesNotMatch(drawing.render({ bs: s.bs, ue: s.ue, result: r, view: { path: 2, hits: 3, arrival: true, walls: 3 } }).svg, /NaN|Infinity|<ellipse|virtual anchor/i);
  }
});

test('two known UE poses share moving wall hits while both measured angle pairs and lengths remain fixed', () => {
  const s = m.ambiguityScene, zs = m.ambiguityMeasurements(s), baseline = m.ambiguity(s.bs, s.ues, zs, s.walls[0]);
  for (const deg of [-18, -10, 0, 7, 12]) {
    const a = deg * Math.PI / 180, r = m.ambiguity(s.bs, s.ues, zs, m.wall(s.pivot, [Math.cos(a), Math.sin(a)]));
    assert.equal(r.status, 'ok'); near(r.maxResidual, 0); near(r.sharedError, 0); near(r.sharedAngle, 0, 2e-6);
    r.paths.forEach((p, i) => {
      const reconstructed = measurement(s.bs, s.ues[i], p.hits);
      near(reconstructed.aod, zs[i].aod); near(reconstructed.aoa, zs[i].aoa); near(reconstructed.length, zs[i].length);
    });
    if (deg) assert.ok(Math.hypot(...sub(r.paths[0].hits[0], baseline.paths[0].hits[0])) > .1);
    for (const step of [1, 2, 3]) assert.doesNotMatch(drawing.renderAmbiguity({ bs: s.bs, ues: s.ues, result: r, baseline, step }).svg, /NaN|Infinity|<ellipse|virtual anchor/i);
  }
});

test('an associated single-bounce MPC removes the demonstrated ambiguity independently of the candidate angle', () => {
  const s = m.ambiguityScene, zs = m.ambiguityMeasurements(s), one = m.referencePath(s.bs, s.ues[0], [s.walls[0]]);
  const baseline = m.ambiguity(s.bs, s.ues, zs, s.walls[0]);
  for (const deg of [-18, 0, 12]) {
    const a = deg * Math.PI / 180, r = m.ambiguity(s.bs, s.ues, zs, m.wall(s.pivot, [Math.cos(a), Math.sin(a)]), one);
    assert.equal(r.status, 'ok'); near(r.maxResidual, 0);
    r.paths.forEach((p, j) => p.hits.forEach((q, i) => pointNear(q, baseline.paths[j].hits[i])));
  }
});
