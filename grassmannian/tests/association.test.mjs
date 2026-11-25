import test from 'node:test';
import assert from 'node:assert/strict';
import * as M from '../model.mjs';
import * as A from '../association.mjs';
const near=(a,b,t=1e-7)=>assert.ok(Math.abs(a-b)<t,`${a} != ${b}`);
test('affine embedding is orthonormal and ignores displacement within the object',()=>{
 const o=A.makeScene().source[3],E=A.affineEmbedding(o);near(M.orthError(E),0);
 const moved={...o,b:o.b.map((x,i)=>x+3*o.A[i][0])};
 near(M.norm(M.add(M.projector(E),M.projector(A.affineEmbedding(moved)),-1)),0);
});
test('unequal-dimensional containment gives zero in the paper dissimilarity',()=>{
 near(Math.hypot(...M.principalAngles([[1],[0],[0]],[[1,0],[0,1],[0,0]])),0);
 assert.equal(M.principalAngles([[1],[0],[0]],[[1,0],[0,1],[0,0]]).length,1);
});
test('Proposition 1: shifting corresponding anchors preserves each mixed pair under SE3',()=>{
 const {source,R,t}=A.makeScene();
 for(let i=0;i<source.length;i++)for(let j=i+1;j<source.length;j++) {
  const a=source[i],b=source[j];
  near(A.objectDistance(a,b),A.objectDistance(A.transformObject(a,R,t),A.transformObject(b,R,t)));
 }
 // The raw embedding is NOT translation invariant.
 const a=source[0],b=source[1];
 assert.ok(Math.abs(A.objectDistance(a,b,{mode:'raw'})-A.objectDistance(A.transformObject(a,R,t),A.transformObject(b,R,t),{mode:'raw'}))>.1);
});
test('scaling positions and rho together preserves the shifted score',()=>{
 const [a,b]=A.makeScene().source;
 const grow=o=>({...o,b:o.b.map(x=>x*10)});
 near(A.objectDistance(a,b,{rho:3}),A.objectDistance(grow(a),grow(b),{rho:30}));
});
test('noiseless mixed-object matching recovers all six despite clutter and arbitrary pose',()=>{
 for(const angle of [-145,0,55,139])for(const translation of [0,7,18]){
  const scene=A.makeScene({angle,translation}),r=A.associate(scene);
  assert.equal(r.candidates.length,24);assert.equal(r.correct,6);assert.equal(r.matches.length,6);near(r.density,6);
  assert.equal(new Set(r.matches.map(m=>m.i)).size,6);assert.equal(new Set(r.matches.map(m=>m.j)).size,6);
 }
});
test('changing evaluation IDs never changes the selected matching',()=>{
 const s=A.makeScene({noise:.03}),r=A.associate(s);
 const noTruth={...s,source:s.source.map((x,i)=>({...x,id:'a'+i})),target:s.target.map((x,i)=>({...x,id:'b'+i}))};
 assert.deepEqual(A.associate(noTruth).selected,r.selected);assert.equal(A.associate(noTruth).correct,0);
 assert.equal(r.correct,6);
});
test('weighted density can prefer a smaller clique; solver evaluates nonmaximal sets',()=>{
 const W=[[1,1,.1],[1,1,.1],[.1,.1,1]],r=A.densestClique(W);
 assert.deepEqual(r.selected,[0,1]);near(r.density,2);
 const C=[[1,.9,0],[.9,1,.8],[0,.8,1]];assert.deepEqual(A.densestClique(C).selected,[0,1]);
});
