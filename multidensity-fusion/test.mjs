import test from 'node:test';
import assert from 'node:assert/strict';
import { gaussianPool, mixtureMoments, sharedPrior, correlation, ci2, rotateCov, det2, optimalCI, logNormal,
  pools, integrate, normalizeLogs, bernoulli, equalGaussianOverlap } from './math.mjs';
import { deck, inlineLiveMap } from './bento-deck.mjs';
const near=(a,b,tol=1e-10)=>assert.ok(Math.abs(a-b)<=tol,`${a} != ${b}`);
test('shared prior: precision is counted once, including M=1',()=>{
  const a=sharedPrior(1,4,2);near(a.correct.variance,.2);near(a.correct.mean,1.6);near(a.product.variance,.125);near(a.product.mean,1);
  assert.deepEqual(sharedPrior(1,1,2).correct,sharedPrior(1,1,2).product);
});
test('Gaussian GCI: idempotence and endpoints',()=>{
  const identical=gaussianPool(3,2,3,2,.2); near(identical.mean,3); near(identical.variance,2);
  assert.deepEqual(gaussianPool(0,1,2,4,0),{mean:2,variance:4});
  assert.deepEqual(gaussianPool(0,1,2,4,1),{mean:0,variance:1});
});
test('AA moments retain between-mean spread',()=>{const a=mixtureMoments(-2,1,2,1);near(a.mean,0);near(a.variance,5);});
test('correlation-aware optimum and CI bound across a parameter sweep',()=>{
  for(const p of [.25,.5,1,2,4])for(const rho of [-.95,-.5,0,.5,.95])for(const w of [0,.1,.5,.9,1]){
    const r=correlation(1,p,rho,w);
    assert.ok(r.oracle>=0);assert.ok(r.oracle<=r.naiveActual+1e-10);assert.ok(r.ciActual<=r.ciReported+1e-10);
    if(rho===0)near(r.naiveActual,r.naiveReported);
  }
  near(correlation(1,1,.8).naiveActual,.9);near(correlation(1,1,.8).naiveReported,.5);
});
test('2D CI: endpoint, symmetry, SPD, and orthogonal optimum',()=>{
  const a=[4,0,.25],b=rotateCov(4,.25,Math.PI/2);
  near(det2(ci2(a,b,1)),1);near(det2(ci2(a,b,0)),1);near(optimalCI(a,b).weight,.5);
  for(let i=0;i<=100;i++)assert.ok(det2(ci2(a,b,i/100))>0);
  const c=ci2(a,b,.3),d=ci2(b,a,.7);c.forEach((v,i)=>near(v,d[i]));
});
test('log-domain pools agree with Gaussian CI and preserve normalizers',()=>{
  const x=Array.from({length:2001},(_,i)=>-10+i*.01),dx=.01;
  const p=pools(x.map(v=>logNormal(v,-1,1)),x.map(v=>logNormal(v,1,1)),dx,.5);
  for(const key of ['p1','p2','aa','gci','product'])near(integrate(p[key],dx),1);
  near(Math.exp(p.logZ),Math.exp(-.5));
  near(integrate(p.gci.map((v,i)=>v*x[i]),dx),0);
  near(integrate(p.gci.map((v,i)=>v*x[i]**2),dx),1);
});
test('zero overlap remains undefined, but weight endpoints are well defined',()=>{
  const a=[0,0,-Infinity,-Infinity],b=[-Infinity,-Infinity,0,0];
  const p=pools(a,b,1,.5);assert.equal(p.gci,null);assert.equal(p.product,null);near(integrate(p.aa,1),1);
  assert.deepEqual(pools(a,b,1,0).gci,pools(a,b,1,0).p2);
  assert.deepEqual(pools(a,b,1,1).gci,pools(a,b,1,1).p1);
});
test('very small log values normalize without underflow',()=>{
  const r=normalizeLogs([-10000,-10001,-10002],.1);near(integrate(r.density,.1),1);assert.ok(Number.isFinite(r.logZ));
});
test('Bernoulli set normalization, conflict, zero masses, endpoints',()=>{
  near(bernoulli(.9,.9,.5,1).gci,.9);
  const eta=equalGaussianOverlap(4,1,.5);near(eta,Math.exp(-2));assert.ok(bernoulli(.9,.9,.5,eta).gci<.9);
  assert.equal(bernoulli(.9,.9,.5,0).gci,0);
  assert.equal(bernoulli(1,1,.5,0).gci,null);
  assert.equal(bernoulli(0,1,.5,1).gci,null);
  near(bernoulli(.3,.8,0,0).gci,.8);near(bernoulli(.3,.8,1,0).gci,.3);
});
test('reject invalid input rather than silently produce NaN',()=>{
  assert.throws(()=>gaussianPool(0,0,0,1));assert.throws(()=>gaussianPool(0,1,0,1,2));
  assert.throws(()=>correlation(1,1,1));assert.throws(()=>ci2([1,2,1],[1,0,1],.5));
  assert.throws(()=>sharedPrior(1,1.5,1));assert.throws(()=>normalizeLogs([0,NaN],1));
});
test('29 valid Bento slides, six aligned inline labs, no out-of-bounds elements',()=>{
  assert.equal(deck.slides.length,29);assert.equal(inlineLiveMap.length,6);
  const ids=new Set(deck.slides.map(s=>s.id));assert.equal(ids.size,deck.slides.length);
  for(const s of deck.slides){
    const elements=new Set();assert.ok(s.notes.length>30);
    for(const e of s.elements){
      assert.ok(!elements.has(e.id),`${s.id}: duplicate ${e.id}`);elements.add(e.id);
      assert.ok(e.x>=0&&e.y>=0&&e.w>0&&e.h>0&&e.x+e.w<=1280.01&&e.y+e.h<=720,`${s.id}/${e.id}: out of bounds`);
      if(e.link&&!e.link.startsWith('https://'))assert.ok(ids.has(e.link),`missing target ${e.link}`);
      if(e.type==='text')assert.ok(!/<\/?(?:sup|sub)\b/i.test(e.html));
    }
  }
  for(const entry of inlineLiveMap){assert.equal(deck.slides[entry.slideIndex].id,entry.slide);assert.ok(entry.src.includes('demo='));}
});
