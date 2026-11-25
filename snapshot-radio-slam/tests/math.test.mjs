import assert from 'node:assert/strict';
import {C, makeSnapshot, rotationFromEulerDegrees, conditionalSolve, svdLeastSquares, unifiedResiduals,
        orientationJacobian, matMul, expSO3, yawSweep, refineOrientation, norm, sub} from '../live/math.mjs';

function near(actual, expected, tolerance = 1e-9) {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} differs from ${expected} by more than ${tolerance}`);
}

// Independent dense least-squares fixture with exact answer.
{
  const A = [[1, 2, 0, 1], [0, -1, 3, 2], [2, 0, 1, -1], [3, 2, -2, 0], [1, 0, 2, 3]];
  const truth = [2, -3, 4, 1], b = A.map(row => row.reduce((s, x, i) => s + x * truth[i], 0));
  const fit = svdLeastSquares(A, b);
  assert.equal(fit.rank, 4); fit.solution.forEach((x, i) => near(x, truth[i]));
  assert.equal(fit.modes.length, 4);
  fit.modes.forEach((mode, j) => {
    assert.ok(mode.retained); near(mode.sigma, fit.singularValues[j]);
    near(mode.coefficient, mode.projection / mode.sigma);
    near(norm(mode.vector), 1);
  });
  fit.solution.forEach((x, k) => near(x, fit.modes.reduce((sum, mode) => sum + mode.contribution[k], 0)));
  assert.ok(fit.cost < 1e-20);
}

const scene = makeSnapshot(), R = rotationFromEulerDegrees(30, 12, -8);
const exact = conditionalSolve(scene.snapshot, R);
assert.equal(exact.rank, 4);
assert.ok(norm(sub(exact.state.position, scene.truth.position)) < 1e-10);
near(exact.state.clockBiasNs, 10, 1e-10);
assert.ok(exact.cost < 1e-20);
assert.equal(exact.A.length, 21); assert.equal(exact.blocks.length, 7);
assert.deepEqual(exact.blocks[1].rows, [3, 4, 5]);
assert.ok(unifiedResiduals(scene.snapshot, scene.truth).every(r => r.norm < 1e-12));

// Delay-reference shifts change only clock, with no truth used by the solve.
{
  const shift = 7.5;
  const changed = {...scene.snapshot, measurements: scene.snapshot.measurements.map(m => ({...m, rho: m.rho + shift}))};
  const result = conditionalSolve(changed, R);
  assert.ok(norm(sub(result.state.position, exact.state.position)) < 1e-10);
  near(result.state.clockBiasM - exact.state.clockBiasM, shift, 1e-10);
}

// Repeated copies of one LoS path do not identify four conditional unknowns.
{
  const los = scene.snapshot.measurements.find(m => m.id === 'los');
  const snapshot = {...scene.snapshot, measurements: Array.from({length: 6}, (_, i) => ({...los, id: `copy-${i}`}))};
  const fit = conditionalSolve(snapshot, R);
  assert.equal(fit.rank, 1); assert.equal(fit.state, null);
  assert.equal(fit.modes.filter(mode => mode.retained).length, 1);
  assert.ok(fit.modes.filter(mode => !mode.retained).every(mode => mode.contribution.every(x => x === 0)));
  assert.equal(conditionalSolve(scene.snapshot, R, []).rank, 0);
}

// Physical double-bounce geometry violates the assumed unified inlier model.
{
  const contaminated = makeSnapshot({includeDoubleBounce: true});
  const all = conditionalSolve(contaminated.snapshot, R);
  near(all.rmsResidual, Math.sqrt(all.cost / contaminated.snapshot.measurements.length));
  const ids = contaminated.snapshot.measurements.filter(m => m.id !== 'double-1').map(m => m.id);
  const filtered = conditionalSolve(contaminated.snapshot, R, ids);
  assert.ok(all.cost > 1);
  assert.ok(norm(sub(filtered.state.position, contaminated.truth.position)) < 1e-10);
  assert.ok(filtered.residuals.find(r => r.id === 'double-1').norm > 1);
}

// Noise is deterministic, independent by component, and stable under toggles.
{
  const a = makeSnapshot({noise: 0.1, includeLos: true});
  const b = makeSnapshot({noise: 0.1, includeLos: false, includeDoubleBounce: true});
  for (const m of b.snapshot.measurements.filter(m => m.id.startsWith('single-'))) assert.deepEqual(m, a.snapshot.measurements.find(k => k.id === m.id));
}

// Independent central differences verify the left-SO(3) orientation derivative.
{
  const candidate = conditionalSolve(scene.snapshot, rotationFromEulerDegrees(33, 9, -6));
  const J = orientationJacobian(scene.snapshot, candidate.state);
  for (let k = 0; k < 3; k++) {
    const step = [0, 0, 0]; step[k] = 1e-6;
    const plus = unifiedResiduals(scene.snapshot, {...candidate.state, rotation: matMul(expSO3(step), candidate.state.rotation)});
    const minus = unifiedResiduals(scene.snapshot, {...candidate.state, rotation: matMul(expSO3(step.map(x => -x)), candidate.state.rotation)});
    plus.forEach((rp, i) => rp.vector.forEach((x, j) => near((x - minus[i].vector[j]) / 2e-6, J[i][j][k], 2e-8)));
  }
}

// The true Euler slice is sampled exactly by this 2-degree yaw sweep.
{
  const sweep = yawSweep(scene.snapshot, {pitchDeg: 12, rollDeg: -8, stepDeg: 2});
  near(sweep.best.yawDeg, 30); assert.ok(sweep.best.cost < 1e-20);
  const refined = refineOrientation(scene.snapshot, rotationFromEulerDegrees(33, 9, -6), {maxIterations: 800});
  assert.ok(refined.cost < 1e-12, `Local refinement cost ${refined.cost}`);
  assert.ok(norm(sub(refined.state.position, scene.truth.position)) < 1e-5);
}

console.log('All 8 math checks passed: SVD, conditional pose/clock, delay shift, degeneracy, outlier exclusion, deterministic noise, analytic Jacobian, yaw/local orientation.');
