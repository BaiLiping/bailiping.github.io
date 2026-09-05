/* Regression checks for the 2026-09-05 mathematical corrections.
 * Content checks exercise deployed HTML/build sources. Independent numerical
 * checks below test the corrected identities, not all website solver branches.
 * Run: node --test tests/math-audit.test.cjs
 */
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const read = p => fs.readFileSync(path.join(root, p), 'utf8');
const near = (a,b,t=1e-9) => assert.ok(Math.abs(a-b)<t, `${a} != ${b}`);

test('PMBM contains both disjoint-set convolutions and normalized weights', () => {
  const s=read('bp-vs-pmbm/index.html');
  assert.ok(s.includes('(4a) undetected Poisson'));
  assert.ok(s.includes('(4b) sum over disjoint component-set decompositions'));
  assert.ok(s.includes('Both disjoint-set sums'));
});
test('association mixture is a marginal, not a joint posterior', () => {
  const s=read('jvs-slam/index.html');
  assert.ok(s.includes('marginal trajectory–map posterior'));
  assert.ok(!s.includes('<i>M</i>, <i>A</i><sub>1:k</sub> |'));
  assert.ok(s.includes('and the weights sum to one'));
});
test('registration states zero-noise, cell-domain, and Hessian conditions', () => {
  const s=read('frame-registration/index.html');
  for(const q of ['conditional on an inlier assignment','only piecewise analytic','nonsingular pose Hessian']) assert.ok(s.includes(q),q);
  assert.ok(!s.includes('so the worst seams cancel'));
});
test('ray tracing distinguishes fixed geometry from differentiable inputs', () => {
  const s=read('differentiable-ray-tracing/index.html');
  assert.ok(s.includes('∂ℓ/∂h = 2(2h'));
  assert.ok(s.includes('These are examples, not an exhaustive list'));
  assert.ok(!s.includes('geometry you specified is not on it'));
});
test('radio hypotheses, residual units, mixture scope and sphere domain survive build', () => {
  const s=read('mpc-detection-to-bounce-count/index.html');
  for(const q of ['Each entry of \\(A\\) selects a complete path hypothesis','fixed-association geometric back-end objective','data-math-audit="mixture-scope"','including all cross-covariances','At the antipode','\\Delta\\tau\\sim1/B']) assert.ok(s.includes(q),q);
  assert.equal((s.match(/data-math-audit="mixture-scope"/g)||[]).length,2);
  assert.ok(!s.includes('a_{ts\\ell}\\in\\{0,1,\\ldots,J\\}'));
  for(const q of ['id="problem-formulation-2d"','id="problem-formulation-3d"','Direct wall-state measurement model','Direct planar-wall measurement model']) assert.ok(s.includes(q),q);
  for(const p of ['scripts/add-radio-slam-problem-formulation.mjs','scripts/upgrade-radio-slam-formulation-3d.mjs']) assert.ok(read(p).includes('data-math-audit="mixture-scope"'),p);
});
test('advanced-state canonical and generated deck state rank and interpolation scope', () => {
  const source=read('advanced-state-representations/bento-deck.mjs');
  assert.ok(source.includes('For a positive-definite kernel'));
  assert.ok(source.includes('inverse-free form remains valid'));
  const s=read('advanced-state-representations/index.html');
  const match=s.match(/<script[^>]*id="bento-doc"[^>]*>([\s\S]*?)<\/script>/);
  assert.ok(match);
  const deck=JSON.parse(match[1]);
  const kernel=JSON.stringify(deck.slides.find(x=>x.id==='kernel-trick'));
  assert.ok(kernel.includes('positive-definite'));
  assert.ok(kernel.includes('inverse-free'));
  const gp=JSON.stringify(deck.slides.find(x=>x.id==='gp-live'));
  assert.ok(gp.includes('not full continuous-time GP uncertainty'));
  assert.ok(read('advanced-state-representations/live/app.js').includes('≈95% interpolation band'));
});
test('gauge explanation and ESS interpretation are qualified', () => {
  assert.ok(read('graph-slam/index.html').includes('not by monocular gauge freedom:'));
  assert.ok(read('sampling-playground/index.html').includes('not a general function-specific standard-error guarantee'));
});
test('CPD positive-outlier limit differs from conditional-inlier limit', () => {
  const p=(sigma,w)=>{const g=[.2,.5].map(d=>Math.exp(-d*d/(2*sigma*sigma)));const out=2*Math.PI*sigma*sigma*w/(1-w);return [...g.map(x=>x/(g[0]+g[1]+out)),out/(g[0]+g[1]+out)];};
  assert.ok(p(.03,.1)[0]<1e-6);
  assert.ok(p(.03,.1)[2]>.999999);
  assert.ok(p(.03,0)[0]>.999999);
});
test('wall-height path derivative agrees with central differences', () => {
  const L=h=>Math.hypot(3,2*h-3),h=5,e=1e-5;
  near((L(h+e)-L(h-e))/(2*e),2*(2*h-3)/L(h),1e-9);
});
test('inverse-free finite-feature solve works for singular kernel', () => {
  const K=[[1,2],[2,4]], B=[[2,2],[2,5]],det=B[0][0]*B[1][1]-B[0][1]*B[1][0];
  assert.equal(K[0][0]*K[1][1]-K[0][1]*K[1][0],0);
  const v=[B[1][1]/det,-B[1][0]/det];
  const delta=K.map(row=>row[0]*v[0]+row[1]*v[1]);
  near(delta[0],1/6);near(delta[1],1/3);
});
test('seconds-to-metres covariance conversion preserves Mahalanobis cost', () => {
  const quad=(r,S)=>{const det=S[0][0]*S[1][1]-S[0][1]*S[1][0];return (S[1][1]*r[0]**2-2*S[0][1]*r[0]*r[1]+S[0][0]*r[1]**2)/det;};
  near(quad([2,1],[[4,.5],[.5,2]]),quad([6,1],[[36,1.5],[1.5,2]]));
});
test('continuous Wiener bridge remains uncertain between known endpoints', () => {
  const q=.4,t=.5,bridge=q*t*(1-t);
  near(bridge,.1);assert.ok(bridge>0);
});
