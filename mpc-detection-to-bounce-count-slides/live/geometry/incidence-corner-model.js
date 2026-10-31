(function (root) {
  'use strict';
  const single = typeof module === 'object' && module.exports ? require('./incidence-single-model.js') : root.IncidenceSingleModel;
  const add = (a, b) => a.map((v, i) => v + b[i]);
  const sub = (a, b) => a.map((v, i) => v - b[i]);
  const mul = (v, s) => v.map(x => x * s);
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
  const norm = v => Math.hypot(...v);
  const unit = v => mul(v, 1 / norm(v));
  const bearing = v => Math.atan2(v[1], v[0]) * 180 / Math.PI;
  const reflect = (direction, normal) => sub(direction, mul(normal, 2 * dot(direction, normal)));
  const scene = {
    bs: [0, 0], ue: [16, -17],
    wallA: { point: [0, 7.5], tangent: [1, 0] },
    wallB: { point: [24, 7.5], tangent: [Math.sin(Math.PI / 9), -Math.cos(Math.PI / 9)] }
  };

  function hitWall(origin, direction, point, normal) {
    const denominator = dot(direction, normal);
    if (Math.abs(denominator) < 1e-8) return { status: 'parallel-wall' };
    const distance = dot(sub(point, origin), normal) / denominator;
    if (distance <= 1e-8) return { status: 'behind-wall' };
    return { status: 'ok', point: add(origin, mul(direction, distance)), distance };
  }

  // Synthetic observations only. The reconstruction below receives no reference map.
  function referenceMeasurements(bs, ue, wallA = scene.wallA, wallB = scene.wallB) {
    const normalB = unit([-wallB.tangent[1], wallB.tangent[0]]);
    const reflectedUE = sub(ue, mul(normalB, 2 * dot(sub(ue, wallB.point), normalB)));
    const first = single.referenceMeasurements(bs, ue, wallA.point, wallA.tangent);
    const unfolded = single.referenceMeasurements(bs, reflectedUE, wallA.point, wallA.tangent);
    const p1 = unfolded.point, departure = unit(sub(p1, bs));
    const normalA = unit([-wallA.tangent[1], wallA.tangent[0]]);
    const hit = hitWall(p1, reflect(departure, normalA), wallB.point, normalB);
    if (hit.status !== 'ok') throw new RangeError('The synthetic route must hit wall B after wall A.');
    const p2 = hit.point;
    return {
      single: first,
      double: { aod: bearing(sub(p1, bs)), aoa: bearing(sub(p2, ue)), length: norm(sub(p1, bs)) + norm(sub(p2, p1)) + norm(sub(ue, p2)) },
      points: { single: first.point, p1, p2 }
    };
  }

  // Path 1 is associated with a single reflection on path 2's FIRST wall.
  // Bearings are outward, in the known global BS/UE orientation frame.
  function construction(bs, ue, firstMeasurement, secondMeasurement) {
    const first = single.construction(bs, ue, firstMeasurement.aod, firstMeasurement.aoa, firstMeasurement.length);
    const base = { single: first };
    if (first.status !== 'ok') return { ...base, status: 'first-wall-unresolved' };
    const wallA = { point: first.point, normal: first.wallNormal, tangent: first.wallTangent };
    const a = { ...base, wallA };
    if (![secondMeasurement.aod, secondMeasurement.aoa].every(Number.isFinite)) return { ...a, status: 'invalid' };
    const departure = single.direction(secondMeasurement.aod), arrival = single.direction(secondMeasurement.aoa);
    const hit = hitWall(bs, departure, wallA.point, wallA.normal);
    const b = { ...a, departure, arrival };
    if (hit.status !== 'ok') return { ...b, status: hit.status };
    const reflected = reflect(departure, wallA.normal), p1 = hit.point;
    const second = single.intersectRays(p1, ue, bearing(reflected), secondMeasurement.aoa);
    const c = { ...b, p1, reflected, firstDistance: hit.distance, second };
    if (second.status !== 'ok') return { ...c, status: `second-${second.status}` };
    const length = hit.distance + second.length;
    return {
      ...c, status: 'ok', p2: second.point,
      wallB: { point: second.point, normal: second.wallNormal, tangent: second.wallTangent },
      middleDistance: second.bsDistance, lastDistance: second.ueDistance,
      length, measuredLength: secondMeasurement.length, residual: secondMeasurement.length - length,
      crossingAngle: second.crossingAngle
    };
  }

  const api = { scene, referenceMeasurements, construction };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.IncidenceCornerModel = api;
})(globalThis);
