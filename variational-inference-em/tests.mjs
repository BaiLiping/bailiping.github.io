import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {deck,inlineLiveMap} from './bento-deck.mjs';
const require=createRequire(import.meta.url),V=require('./live/math.js');
let checks=0;
function near(a,b,tolerance=1e-8){assert.ok(Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<=tolerance,`${a} != ${b} (tolerance ${tolerance})`);checks++;}
function ge(a,b,tolerance=1e-8){assert.ok(a>=b-tolerance,`${a} decreased below ${b}`);checks++;}
near(V.logsumexp([1000,1001]),1001+Math.log1p(Math.exp(-1)));
assert.deepEqual(V.sampleData(7),V.sampleData(7));checks++;
for(const rho of [-.95,-.8,-.3,0,.3,.8,.95]){
 const d=1-rho*rho,optimum=-.5*Math.log(d);
 near(V.gaussianKL(rho,[0,0],[d,d]),optimum);
 let mean=[-1.5,1.5],variance=[1,1],prev=V.gaussianKL(rho,mean,variance);
 for(let t=0;t<500;t++){
  const j=t%2,oldM=mean.slice(),oldV=variance.slice();
  ({mean,variance}=V.cavi(rho,mean,variance,j));
  near(mean[1-j],oldM[1-j]);near(variance[1-j],oldV[1-j]);near(variance[j],d);
  const kl=V.gaussianKL(rho,mean,variance);ge(prev,kl);ge(kl,optimum);prev=kl;
 }
 near(prev,optimum,1e-8);near(mean[0],0,1e-8);near(mean[1],0,1e-8);
}
let fits=0,halfSteps=0;
for(const seed of [1,7,42])for(const separation of [.5,3.5,6])for(const kind of ['spread','poor','identical'])for(const learn of [false,true]){
 const data=V.sampleData(seed,separation);let model=V.initialModel(kind,separation),q=data.map(()=>[.5,.5]);
 for(let t=0;t<40;t++){
  const before=V.metrics(data,model,q);q=V.expectation(data,model);
  for(const row of q){near(row.reduce((a,b)=>a+b),1);assert.ok(row.every(r=>r>=0&&r<=1));}
  const afterE=V.metrics(data,model,q);near(before.likelihood,afterE.likelihood);near(afterE.elbo,afterE.likelihood,1e-7);near(afterE.gap,0,1e-7);ge(afterE.elbo,before.elbo,1e-7);
  model=V.maximization(data,q,model,learn,.09);const afterM=V.metrics(data,model,q);
  ge(afterM.likelihood,afterE.likelihood,1e-7);ge(afterM.elbo,afterE.elbo,1e-7);ge(afterM.gap,0,1e-7);near(afterM.likelihood-afterM.elbo,afterM.gap,1e-7);
  near(model.weight.reduce((a,b)=>a+b),1);assert.ok(model.variance.every(v=>v>=.09));
  if(!learn)model.variance.forEach(v=>near(v,.64));
  if(kind==='identical'){near(model.mean[0],model.mean[1]);near(model.variance[0],model.variance[1]);}
  halfSteps+=2;
 }
 fits++;
}
assert.equal(deck.slides.length,26);assert.equal(new Set(deck.slides.map(s=>s.id)).size,26);
const ids=new Set(deck.slides.map(s=>s.id));
for(const slide of deck.slides){assert.ok(slide.notes);for(const e of slide.elements){if(e.link&&!e.link.startsWith('http'))assert.ok(ids.has(e.link),'Unknown route '+e.link);}}
for(const entry of inlineLiveMap){assert.equal(deck.slides[entry.slideIndex].id,entry.slide);assert.equal(entry.sandbox,'allow-scripts');}
console.log(JSON.stringify({status:'passed',checks,gaussianCorrelations:7,mixtureFits:fits,emHalfSteps:halfSteps,slides:deck.slides.length,liveLabs:inlineLiveMap.length},null,2));
