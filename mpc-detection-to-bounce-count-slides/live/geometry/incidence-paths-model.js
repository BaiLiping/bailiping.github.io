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
  const reflect = (v, n) => sub(v, mul(n, 2 * dot(v, n)));
  const wall = (point, tangent) => ({ point, tangent: unit(tangent), normal: unit([-tangent[1], tangent[0]]) });
  const wallAngle = (a, b) => Math.acos(Math.min(1, Math.abs(dot(a.normal, b.normal)))) * 180 / Math.PI;
  const scenes = {
    corner3: {
      bs: [0, 0], ue: [16, -17], bounds: [14, 21, -21, -13],
      walls: [wall([0, 7.5], [1, 0]), wall([24, 7.5], [Math.sin(Math.PI / 9), -Math.cos(Math.PI / 9)]), wall([-3, -28.5], [Math.cos(8 * Math.PI / 180), Math.sin(8 * Math.PI / 180)])],
      routes: [[0], [0, 1], [0, 1, 2]]
    },
    corridor2: {
      bs: [0, 0], ue: [10, 25], bounds: [1, 15, 8, 28],
      walls: [wall([17, 0], [0, 1]), wall([-5, 0], [0, 1])], routes: [[0], [0, 1]]
    },
    corridor3: {
      bs: [0, 0], ue: [10, 25], bounds: [1, 15, 8, 28],
      walls: [wall([17, 0], [0, 1]), wall([-5, 0], [0, 1])], routes: [[0], [0, 1], [0, 1, 0]]
    }
  };

  function hitWall(origin, direction, w) {
    const denominator = dot(direction, w.normal);
    if (Math.abs(denominator) < 1e-8) return { status: 'parallel-wall' };
    const distance = dot(sub(w.point, origin), w.normal) / denominator;
    if (distance <= 1e-8) return { status: 'behind-wall' };
    return { status: 'ok', point: add(origin, mul(direction, distance)), distance };
  }

  // Test-scene observation generator only. No reference walls enter reconstruct().
  // Unfold the receiver to generate a physical departure bearing, then ray-trace
  // every ordered reflection and check that the last forward leg reaches the UE.
  function referencePath(bs, ue, orderedWalls) {
    let target = [...ue];
    for (const w of [...orderedWalls].reverse()) target = sub(target, mul(w.normal, 2 * dot(sub(target, w.point), w.normal)));
    const departure = unit(sub(target, bs));
    let origin = bs, direction = departure, length = 0;
    const hits = [];
    for (const w of orderedWalls) {
      const hit = hitWall(origin, direction, w);
      if (hit.status !== 'ok') return { status: hit.status };
      hits.push(hit.point); length += hit.distance;
      origin = hit.point; direction = reflect(direction, w.normal);
    }
    const last = sub(ue, origin), distance = norm(last);
    if (distance < 1e-8 || dot(last, direction) <= 0 || norm(sub(unit(last), direction)) > 1e-7) return { status: 'invalid-route' };
    return { status: 'ok', aod: bearing(departure), aoa: bearing(sub(origin, ue)), length: length + distance, hits };
  }
  function referenceMeasurements(scene, ue = scene.ue) {
    return scene.routes.map(route => referencePath(scene.bs, ue, route.map(i => scene.walls[i])));
  }

  // Reflect the measured AoD at already inferred walls. The last hit is the
  // forward intersection with measured AoA, never a point read from a map.
  function advance(bs, ue, measurement, knownWalls) {
    if (![...bs, ...ue, measurement.aod, measurement.aoa].every(Number.isFinite)) return { status: 'invalid', hits: [], directions: [] };
    const departure = single.direction(measurement.aod), arrival = single.direction(measurement.aoa);
    let origin = bs, direction = departure, spent = 0;
    const hits = [], directions = [departure];
    const base = { departure, arrival, hits, directions, measuredLength: measurement.length };
    for (let i = 0; i < knownWalls.length; i++) {
      const hit = hitWall(origin, direction, knownWalls[i]);
      if (hit.status !== 'ok') return { ...base, status: hit.status, failedWall: i };
      hits.push(hit.point); spent += hit.distance; origin = hit.point;
      direction = reflect(direction, knownWalls[i].normal); directions.push(direction);
    }
    const last = single.intersectRays(origin, ue, bearing(direction), measurement.aoa);
    if (last.status !== 'ok') return { ...base, status: last.status };
    hits.push(last.point);
    const length = spent + last.length;
    return { ...base, status: 'ok', length, residual: measurement.length - length,
      lastWall: { point: last.point, normal: last.wallNormal, tangent: last.wallTangent },
      crossingAngle: last.crossingAngle, lastDistance: last.ueDistance };
  }

  function reconstruct(bs, ue, measurements, { corridor = false, repeated = false } = {}) {
    const paths = [], walls = [];
    for (let i = 0; i < measurements.length; i++) {
      const path = advance(bs, ue, measurements[i], walls.slice(0, i));
      paths.push(path);
      if (path.status !== 'ok') return { status: 'unresolved', failedPath: i, paths, walls };
      if (i < 2 || !repeated) walls.push(path.lastWall);
    }
    const result = { status: 'ok', paths, walls, consistent: true };
    if (corridor && walls.length >= 2) {
      result.parallelError = wallAngle(walls[0], walls[1]);
      result.consistent = result.parallelError <= 2;
    }
    if (repeated && paths.length === 3) {
      const last = paths[2], point = last.hits[2];
      const distance = Math.abs(dot(sub(point, walls[0].point), walls[0].normal));
      const orientation = wallAngle(last.lastWall, walls[0]);
      result.closure = { distance, orientation, expected: hitWall(last.hits[1], last.directions[2], walls[0]) };
      result.consistent = result.consistent && distance <= .15 && orientation <= 2 && result.closure.expected.status === 'ok';
    }
    return result;
  }

  const ambiguityScene = {
    bs: [0, 0], ues: [[15, -14], [20.5, -20.5]], pivot: [24, 7.5],
    walls: scenes.corner3.walls.slice(0, 2), rotationBounds: [-18, 12]
  };
  function ambiguityMeasurements(scene = ambiguityScene) {
    return scene.ues.map(ue => referencePath(scene.bs, ue, scene.walls));
  }
  function ambiguity(bs, ues, measurements, candidateWallA, singleMeasurement = null) {
    let firstWall = candidateWallA, singlePath = null;
    if (singleMeasurement) {
      singlePath = advance(bs, ues[0], singleMeasurement, []);
      if (singlePath.status !== 'ok') return { status: 'unresolved', singlePath, paths: [] };
      firstWall = singlePath.lastWall;
    }
    const paths = ues.map((ue, i) => advance(bs, ue, measurements[i], [firstWall]));
    if (paths.some(p => p.status !== 'ok')) return { status: 'unresolved', wallA: firstWall, singlePath, paths };
    const wallB = paths[0].lastWall;
    const sharedError = Math.abs(dot(sub(paths[1].hits[1], wallB.point), wallB.normal));
    return { status: 'ok', wallA: firstWall, wallB, paths, singlePath, sharedError,
      sharedAngle: wallAngle(wallB, paths[1].lastWall), maxResidual: Math.max(...paths.map(p => Math.abs(p.residual))) };
  }
  const api = { scenes, wall, referencePath, referenceMeasurements, advance, reconstruct, ambiguityScene, ambiguityMeasurements, ambiguity };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.IncidencePathsModel = api;
})(globalThis);
