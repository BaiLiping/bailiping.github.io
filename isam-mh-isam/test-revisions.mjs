import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {deck,inlineLiveMap} from './bento-deck.mjs';
import {applyTeachingRevisions} from './teaching-revisions.mjs';
const require=createRequire(import.meta.url),M=require('./live/dp-math.js');
applyTeachingRevisions(deck,inlineLiveMap);
assert.equal(deck.slides.length,25);assert.equal(inlineLiveMap.length,4);
const ids=new Set(deck.slides.map(s=>s.id));assert.equal(ids.size,25);
for(const s of deck.slides){
  assert.ok(s.notes);
  assert.equal(new Set(s.elements.map(e=>e.id)).size,s.elements.length,'Duplicate element on '+s.id);
  for(const e of s.elements)assert.ok(e.x>=0&&e.y>=0&&e.x+e.w<=1281&&e.y+e.h<=721,`${s.id}/${e.id} leaves canvas`);
}
for(const lab of inlineLiveMap){assert.equal(deck.slides[lab.slideIndex].id,lab.slide);assert.equal(deck.slides[lab.slideIndex-1].id,lab.introSlide);}
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-10,`${a} != ${b}`);
for(let z=-1;z<=5;z+=.1){
  const {u,s,cost}=M.solve(z);
  close(2*u-1-s,0);close(2*s-u-z,0);
  close(cost,M.child(u,s)+M.external(s,z));
  for(const separator of [-1,0,1,3,5])close(M.summary(separator),M.child(M.recover(separator),separator));
}
assert.deepEqual(M.solve(4),{z:4,s:3,u:2,cost:1.5});
assert.throws(()=>M.solve(NaN),TypeError);
applyTeachingRevisions(deck,inlineLiveMap);assert.equal(deck.slides.length,25);
console.log('PASS: 25 native slides, 4 correctly placed labs, unique IDs, canvas bounds, and exact quadratic DP identities.');
