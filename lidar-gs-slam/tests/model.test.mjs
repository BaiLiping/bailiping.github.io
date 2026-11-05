import test from 'node:test';
import assert from 'node:assert/strict';
import {
  sampleCovariance, symmetricEigen3, dot, geometryFeatures, covarianceLab,
  initialScales, trackingWeights, makeTrackingScene, trackingObjective,
  trackingStep, TrackingLab, controlScores, makeBudgetScene, manageBudget,
  contains, ellipseBoundary
} from '../live/model.mjs';

const near = (a, b, epsilon = 1e-9) => assert.ok(Math.abs(a - b) <= epsilon, `${a} differs from ${b}`);

test('symmetric eigendecomposition reconstructs a rotated covariance with orthonormal axes', () => {
  const matrix = [[2.7, .6, -.3], [.6, 1.5, .2], [-.3, .2, .4]], eigen = symmetricEigen3(matrix);
  assert.ok(eigen.values[0] >= eigen.values[1] && eigen.values[1] >= eigen.values[2]);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    near(dot(eigen.vectors[i], eigen.vectors[j]), i === j ? 1 : 0);
    const reconstructed = eigen.values.reduce((sum, value, k) => sum + value * eigen.vectors[k][i] * eigen.vectors[k][j], 0);
    near(reconstructed, matrix[i][j]);
  }
});

test('planar sample covariance has a null normal direction, while a line has two null directions', () => {
  const plane = symmetricEigen3(sampleCovariance([[-2,-1,0], [-2,1,0], [2,-1,0], [2,1,0]]).matrix);
  near(plane.values[2], 0); near(Math.abs(plane.vectors[2][2]), 1);
  assert.ok(geometryFeatures(plane.values).normalGap > .2);
  const line = symmetricEigen3(sampleCovariance([[-2,0,0], [-1,0,0], [1,0,0], [2,0,0]]).matrix);
  near(line.values[1], 0); near(line.values[2], 0); near(geometryFeatures(line.values).normalGap, 0);
});

test('covariance presets remain deterministic and identify normal ambiguity', () => {
  assert.deepEqual(covarianceLab({shape:'plane'}), covarianceLab({shape:'plane'}));
  assert.equal(covarianceLab({shape:'plane'}).normalStatus, 'Well-separated normal');
  assert.equal(covarianceLab({shape:'edge'}).normalStatus, 'Normal is ambiguous');
  assert.equal(covarianceLab({shape:'scattered'}).normalStatus, 'No dominant surface normal');
});

test('range initialization is linear before clipping and respects the scale cap', () => {
  assert.deepEqual(initialScales([1,.5,.01], 10), [.4,.2]);
  assert.deepEqual(initialScales([1,.5,.01], 20), [.8,.4]);
  assert.deepEqual(initialScales([1,.5,.01], 100), [2,2]);
  assert.deepEqual(initialScales([1,.5,.01], 0), initialScales([1,.5,.01], 1));
});

test('paper geometry weights favour near-round in-plane scales and remain capped', () => {
  const w = trackingWeights([{scales:[1,1]}, {scales:[3,1]}, {scales:[5,1]}]);
  near(w[0].normalized, 0); near(w[1].normalized, .5); near(w[2].normalized, 1);
  near(w[0].weight, 10); near(w[1].weight, 1/.501); near(w[2].weight, 1/1.001);
  assert.deepEqual(trackingWeights([{scales:[1,1]}, {scales:[5,1]}], false).map(x=>x.weight), [1,1]);
  assert.ok(trackingWeights([{scales:[2,1]}, {scales:[2,1]}]).every(x => x.weight === 10 && Number.isFinite(x.normalized)));
});

test('a noiseless fixed-correspondence problem recovers the true SE(2) pose', () => {
  for (const weighted of [false,true]) {
    const lab = new TrackingLab({weighted, bias:0, noise:0, heading:25}), state = lab.solve();
    assert.ok(state.error.translation < 1e-7); assert.ok(state.error.angle < 1e-6);
    assert.ok(state.trace.at(-1) < 1e-8);
  }
});

test('accepted Gauss–Newton updates never increase their own weighting objective', () => {
  for (const weighted of [true,false]) for (const heading of [-25,0,25]) for (const bias of [0,.45,.8]) {
    const lab = new TrackingLab({weighted,heading,bias}), state = lab.solve();
    for (let i=1;i<state.trace.length;i++) assert.ok(state.trace[i] <= state.trace[i-1]+1e-8);
    assert.ok(state.pose.every(Number.isFinite));
  }
});

test('inverse covariance penalizes a surface-normal displacement more than a tangent displacement', () => {
  const scene = makeTrackingScene({bias:0,noise:0}), point = scene.points.find(p=>p.type==='plane' && p.angle===0);
  const smallScene = {...scene, points:[point]}, base = [...scene.truth];
  const tangential = trackingObjective(smallScene,[base[0]+.1,base[1],base[2]],false);
  const normal = trackingObjective(smallScene,[base[0],base[1]+.1,base[2]],false);
  assert.ok(normal > tangential * 10);
});

test('the stated synthetic biased-return example benefits from the planar weights', () => {
  const weighted = new TrackingLab({weighted:true}).solve(), uniform = new TrackingLab({weighted:false}).solve();
  assert.ok(weighted.error.translation < uniform.error.translation);
  assert.ok(weighted.error.angle < uniform.error.angle);
  // This checks this deliberately constructed scene, not a general performance claim.
});

test('map-score normalization and combination stay finite in degenerate populations', () => {
  const same = Array.from({length:30},(_,id)=>({id,lambda:[1,1,0],residual:0}));
  for(const point of controlScores(same)) { assert.equal(point.score,0); assert.ok(Object.values(point.normalized).every(Number.isFinite)); }
  const scene=makeBudgetScene();
  for(const g of scene.gaussians) {
    assert.ok(g.score>=0&&g.score<=1); near(g.score,.55*g.normalized.linearity+.30*g.normalized.curvature+.15*g.normalized.residual);
  }
});

test('plane pruning encloses the original finite ellipses before removing their representatives', () => {
  const scene=makeBudgetScene(), result=manageBudget(scene,{prune:true,split:false});
  assert.ok(result.removed.length>0); assert.ok(result.removed.length<=result.pruneBudget);
  assert.equal(result.gaussians.length,scene.gaussians.length-result.removed.length);
  for(const expanded of result.expanded) for(const id of expanded.coveredIds) {
    const original=scene.gaussians.find(g=>g.id===id);
    for(const p of ellipseBoundary(original,128)) assert.ok(contains(expanded,p,1e-8));
  }
  assert.ok(result.coverage>=result.beforeCoverage-1e-12);
});

test('splits are symmetric, obey their cap and add exactly one net Gaussian per parent', () => {
  const scene=makeBudgetScene(), result=manageBudget(scene,{prune:false,split:true});
  assert.ok(result.splitParents.length>0); assert.ok(result.splitParents.length<=result.splitBudget);
  assert.equal(result.gaussians.length,scene.gaussians.length+result.splitParents.length);
  for(const parent of result.splitParents) {
    const children=result.gaussians.filter(g=>g.parentId===parent.id);assert.equal(children.length,2);
    near((children[0].x+children[1].x)/2,parent.x);near((children[0].y+children[1].y)/2,parent.y);
    assert.ok(children.every(g=>g.scales[0]<parent.scales[0]&&g.score>=result.splitThreshold));
  }
});

test('budget comparisons restart from an immutable map and preserve count identities', () => {
  const scene=makeBudgetScene(), snapshot=JSON.stringify(scene), baseline=manageBudget(scene,{prune:false,split:false}), full=manageBudget(scene);
  assert.equal(baseline.gaussians.length,140);assert.equal(full.gaussians.length,140-full.removed.length+full.splitParents.length);
  assert.equal(JSON.stringify(scene),snapshot);assert.deepEqual(full,manageBudget(scene));
  assert.ok(full.coverage>=0&&full.coverage<=1);
  assert.deepEqual(baseline.gaussians.map(g=>g.id),scene.gaussians.map(g=>g.id));
});
