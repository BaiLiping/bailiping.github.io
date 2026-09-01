import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import vm from 'node:vm';
import { deck, inlineLiveMap } from './bento-deck.mjs';

const output = new URL('./review-results/', import.meta.url);
await mkdir(output, { recursive: true });
assert.equal(deck.slides.length, 21);
assert.equal(new Set(deck.slides.map(s => s.id)).size, 21);
for (const entry of inlineLiveMap) {
  assert.equal(deck.slides[entry.slideIndex].id, entry.slide);
  assert.equal(deck.slides[entry.slideIndex - 1].id, entry.introSlide);
}
for (const slide of deck.slides) {
  for (const element of slide.elements) {
    if (element.type !== 'text') continue;
    assert.ok(!/<\/?(?:sup|sub)\b/i.test(element.html), `${slide.id}/${element.id}: hand-built math`);
    assert.ok(element.x >= 0 && element.y >= 0 && element.x + element.w <= 1280 && element.y + element.h <= 720,
      `${slide.id}/${element.id}: outside slide`);
  }
}

const context = vm.createContext({ window: { location: { search: '' } }, document: { getElementById: () => null }, URLSearchParams, console });
vm.runInContext(await readFile(new URL('./live/model.js', import.meta.url), 'utf8'), context);
let app = await readFile(new URL('./live/app.js', import.meta.url), 'utf8');
const mounts = "  if (demo === 'geometry') mountGeometry()\n  else if (demo === 'equivalence') mountEquivalence()\n  else mountScalar()";
assert.ok(app.includes(mounts), 'Cannot isolate the numerical library for testing');
app = app.replace(mounts, '  window.ReviewMath = { makeOps, generateProblem, covarianceMethod, informationMethod, qrMethod, jacobiEigenvalues, transpose, eye }');
vm.runInContext(app, context);
const { makeOps, generateProblem, covarianceMethod, informationMethod, qrMethod, jacobiEigenvalues, transpose, eye } = context.window.ReviewMath;
const { scalarPosterior, covarianceGeometry } = context.window.KalmanModel;
const ops = makeOps(16);
const maxdiff = (A, B) => Math.max(...A.flat().map((v, i) => Math.abs(v - B.flat()[i])));
let tests = 0, maximumRouteDifference = 0;
function close(A, B, tolerance = 2e-9, message = 'matrix identity') {
  const difference = maxdiff(A, B);
  assert.ok(difference < tolerance, `${message}: difference ${difference}`);
  tests++;
  return difference;
}

for (let n = 2; n <= 4; n++) for (let m = 1; m <= n; m++) for (const seed of [17, 1949, 9868, 77777]) {
  const p = generateProblem(n, m, 2, seed);
  const covariance = covarianceMethod(p, ops);
  const information = informationMethod(p, ops);
  const joseph = covarianceMethod(p, ops, true);
  const qr = qrMethod(p, ops);
  for (const r of [information, joseph, qr]) {
    maximumRouteDifference = Math.max(maximumRouteDifference, close(covariance.covariance, r.covariance));
    close([covariance.mean], [r.mean]);
  }
  const reduction = ops.sub(p.P, covariance.covariance);
  assert.ok(jacobiEigenvalues(reduction)[0] > -1e-10);
  assert.ok(jacobiEigenvalues(covariance.covariance)[0] > 0);
  tests += 2;
  // The arbitrary-gain optimality certificate is independent of Gaussianity.
  const S = ops.add(ops.mul(ops.mul(p.H, p.P), transpose(p.H)), p.R);
  const trialK = covariance.K.map(row => row.map(value => value + .13));
  const A = ops.sub(eye(n), ops.mul(trialK, p.H));
  const trialP = ops.add(ops.mul(ops.mul(A, p.P), transpose(A)), ops.mul(ops.mul(trialK, p.R), transpose(trialK)));
  const dK = ops.sub(trialK, covariance.K);
  close(ops.sub(trialP, covariance.covariance), ops.mul(ops.mul(dK, S), transpose(dK)), 2e-9, 'gain optimality certificate');
  // Precision Hessian and QR covariance order: T^-1 T^-T, not reversed.
  const J = ops.add(ops.inverse(p.P), ops.mul(ops.mul(transpose(p.H), ops.inverse(p.R)), p.H));
  close(ops.mul(J, qr.covariance), eye(n), 2e-8, 'inverse Hessian');
}
for (const sx of [.4, 1.8, 2.8]) for (const sy of [.35, 1, 2.2]) for (const rho of [-.9, 0, .9]) for (const angle of [0, 28, 90, 180]) {
  const args = { sx, sy, rho, angleDeg: angle, z: 1.7, measurementSigma: .45 };
  const g = covarianceGeometry(args);
  const r = ops.matVec(g.P, g.h);
  close(ops.sub(g.P, g.Pp), r.map(ri => r.map(rj => ri * rj / g.S)), 1e-10, 'rank-one covariance reduction');
  close(g.Pp, covarianceGeometry({ ...args, z: -3 }).Pp, 1e-12, 'covariance independent of observed value');
  assert.ok(g.areaRatio > 0 && g.areaRatio < 1);
  tests++;
}
const example = covarianceMethod({ P: [[4, 1], [1, 1]], R: [[1]], H: [[1, 0]], priorMean: [0, 0], z: [1] }, ops);
close(example.K, [[.8], [.2]], 1e-12, 'slide gain example');
close(example.covariance, [[.8, .2], [.2, .8]], 1e-12, 'slide covariance example');
for (const priorSigma of [.2, 1.35, 3]) for (const measurementSigma of [.2, .75, 3]) {
  const r = scalarPosterior({ priorMean: -1.2, priorSigma, measurement: 2.1, measurementSigma });
  assert.ok(r.delta < 1e-12 && r.K > 0 && r.K < 1 && r.postVar > 0);
  tests++;
}
// Low-precision QR must round its final covariance products and sums too.
let rounds = 0;
const counted = { ...makeOps(7), round: value => { rounds++; return makeOps(7).round(value); } };
qrMethod(generateProblem(3, 1, 2, 1949), counted);
assert.ok(rounds >= 36, 'Final QR covariance reconstruction is not using simulated rounding');
tests++;
const report = { passed: true, tests, maximumRouteDifference, slides: deck.slides.map((s, i) => ({ number: i + 1, id: s.id })), liveRegions: inlineLiveMap.length, numericalScope: 'Deterministic consistency and matrix-identity checks, not a proof of every claim or a universal solver benchmark.' };
await writeFile(new URL('numerical-checks.json', output), JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));
