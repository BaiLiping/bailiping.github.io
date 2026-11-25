(function (root) {
  'use strict';
  const add = (a, b) => a.map((v, i) => v + b[i]);
  const sub = (a, b) => a.map((v, i) => v - b[i]);
  const mul = (v, s) => v.map(x => x * s);
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
  const cross = (a, b) => a[0] * b[1] - a[1] * b[0];
  const norm = v => Math.hypot(...v);
  const direction = degrees => [Math.cos(degrees * Math.PI / 180), Math.sin(degrees * Math.PI / 180)];
  const bearing = v => Math.atan2(v[1], v[0]) * 180 / Math.PI;
  const scene = {
    bs: [0, 0], ue: [6, -10], wallPoint: [19, 13],
    wallTangent: [Math.sin(Math.PI / 10), -Math.cos(Math.PI / 10)]
  };

  // Synthetic measurements only. The estimator below never receives a wall.
  // Minimize the two physical segment lengths along the reference wall.
  function referenceMeasurements(bs, ue, wallPoint, wallTangent) {
    const tangent = mul(wallTangent, 1 / norm(wallTangent));
    const normal = [-tangent[1], tangent[0]];
    const db = dot(sub(bs, wallPoint), normal), du = dot(sub(ue, wallPoint), normal);
    if (db * du <= 0) throw new RangeError('Both endpoints must be on the same side of the reflecting wall.');
    const footB = sub(bs, mul(normal, db)), footU = sub(ue, mul(normal, du));
    const point = mul(add(mul(footB, Math.abs(du)), mul(footU, Math.abs(db))), 1 / (Math.abs(db) + Math.abs(du)));
    return { point, aod: bearing(sub(point, bs)), aoa: bearing(sub(point, ue)), length: norm(sub(point, bs)) + norm(sub(point, ue)) };
  }

  // Both bearings point OUTWARD from their endpoint toward the scatterer.
  // BS + t * departure = UE + s * arrival, with t > 0 and s > 0.
  function intersectRays(bs, ue, aod, aoa) {
    if (![...bs, ...ue, aod, aoa].every(Number.isFinite)) return { status: 'invalid' };
    const departure = direction(aod), arrival = direction(aoa), delta = sub(ue, bs);
    const determinant = cross(departure, arrival);
    const base = { departure, arrival };
    if (Math.abs(determinant) < 1e-8) {
      const collinear = Math.abs(cross(delta, departure)) < 1e-8 * Math.max(1, norm(delta));
      return { ...base, status: collinear ? 'collinear' : 'parallel' };
    }
    const bsDistance = cross(delta, arrival) / determinant;
    const ueDistance = cross(delta, departure) / determinant;
    if (bsDistance <= 1e-8 || ueDistance <= 1e-8) return { ...base, status: 'behind', bsDistance, ueDistance };
    const point = add(bs, mul(departure, bsDistance));
    const normalSum = add(departure, arrival);
    const normal = mul(normalSum, 1 / norm(normalSum));
    return {
      ...base, status: 'ok', point, bsDistance, ueDistance,
      length: bsDistance + ueDistance, wallNormal: normal, wallTangent: [-normal[1], normal[0]],
      crossingAngle: Math.asin(Math.min(1, Math.abs(determinant))) * 180 / Math.PI
    };
  }

  function construction(bs, ue, aod, aoa, measuredLength) {
    const result = intersectRays(bs, ue, aod, aoa);
    return { ...result, measuredLength, residual: result.status === 'ok' ? measuredLength - result.length : null };
  }

  const api = { scene, direction, referenceMeasurements, intersectRays, construction };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.IncidenceSingleModel = api;
})(globalThis);
