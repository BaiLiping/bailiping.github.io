import test from 'node:test';
import assert from 'node:assert/strict';
import {steering,covarianceMatrix,complexGaussianLogLikelihood,covarianceLab,LearningLab} from '../live/model.mjs';
const close=(a,b,t=1e-9)=>assert.ok(Math.abs(a-b)<t,`${a} != ${b}`);
test('Complex covariance is Hermitian and a unit-norm source adds its stated total power',()=>{
  const c=covarianceMatrix([{u:.31,power:2},{u:-.2,power:.7}],.4,6);
  let tr=0;for(let i=0;i<6;i++){tr+=c.re[i][i];for(let j=0;j<6;j++){close(c.re[i][j],c.re[j][i]);close(c.im[i][j],-c.im[j][i]);}}
  close(tr,6*.4+2+.7);
});
test('Complex Gaussian likelihood agrees with scalar CN density and is phase invariant',()=>{
  const ll=complexGaussianLogLikelihood([{re:[2],im:[1]}],{re:[[3]],im:[[0]]});close(ll,-Math.log(3*Math.PI)-5/3);
  const c=covarianceMatrix([{u:.2,power:1}],.3,4),z={re:[.2,1,-.5,.1],im:[.4,-.2,.8,-.3]},rot={re:z.im.map(v=>-v),im:z.re};
  close(complexGaussianLogLikelihood([z],c),complexGaussianLogLikelihood([rot],c));
});
test('Cholesky rank-one likelihood agrees with analytic Woodbury result',()=>{
  const u=.17,g=1.3,eta=.5,m=4,z={re:[.4,.2,-.8,.6],im:[-.5,.2,.3,.1]},a=steering(u,m);
  let r=0,i=0,e=0;for(let k=0;k<m;k++){r+=a.re[k]*z.re[k]+a.im[k]*z.im[k];i+=a.re[k]*z.im[k]-a.im[k]*z.re[k];e+=z.re[k]**2+z.im[k]**2;}
  const expected=-m*Math.log(Math.PI)-(m-1)*Math.log(eta)-Math.log(eta+g)-e/eta+g*(r*r+i*i)/(eta*(eta+g));
  close(complexGaussianLogLikelihood([z],covarianceMatrix([{u,power:g}],eta,m)),expected);
});
test('Covariance lab normalizes posteriors, retains deterministic data, and diagonal model is bearing invariant',()=>{
  const a=covarianceLab({seed:9,snr:12,snapshots:64,separation:18}),b=covarianceLab({seed:9,snr:12,snapshots:64,separation:18});
  assert.deepEqual(a.posterior,b.posterior);close(a.posterior.reduce((s,v)=>s+v,0),1);assert.ok(Math.abs(a.mapAngle-9)<2);
  assert.ok(a.intensityPosterior.every(v=>v===a.intensityPosterior[0]));
  for(const u of [-.8,0,.8]){const c=covarianceMatrix([{u,power:1}],.2,6);for(let i=0;i<6;i++)close(c.re[i][i],.2+1/6);}
});
test('Learning disabled gives exactly identical filters on the same synthetic measurements',()=>{
  const lab=new LearningLab({learning:false,seed:4});let s;for(let k=0;k<45;k++)s=lab.step();
  close(s.xLearned,s.xFixed);close(s.rmseLearned,s.rmseFixed);assert.equal(s.trainingCount,0);assert.deepEqual(s.posteriorLearned,s.posteriorFixed);
});
test('Learning update has no access to hidden state or truth field values',()=>{
  const a=new LearningLab({seed:3}),b=new LearningLab({seed:3});a.xTruth=-.9;b.xTruth=.9;
  const precedingMean=a.xLearned;
  const z=[{re:[.4,.5,-.1,.2],im:[.1,-.3,.2,.1]},{re:[.1,.3,.4,-.2],im:[-.5,.2,.4,.1]}];
  a.updateFromSamples(z,1);b.updateFromSamples(z,1);assert.deepEqual(a.beta,b.beta);assert.deepEqual(a.pLearned,b.pLearned);close(a.context,b.context);close(a.context,precedingMean);
});
test('Sequential learning remains normalized, finite, repeatable and updates conditional coefficients',()=>{
  const a=new LearningLab({seed:7}),b=new LearningLab({seed:7});let s,t;for(let k=0;k<165;k++){s=a.step();t=b.step();}
  assert.equal(s.frame,165);assert.equal(s.cycle,3);assert.equal(s.trainingCount,165);assert.deepEqual(s.trace,t.trace);
  close(s.posteriorLearned.reduce((a,b)=>a+b,0),1);close(s.posteriorFixed.reduce((a,b)=>a+b,0),1);
  assert.ok(s.beta.slice(1).some(v=>Math.abs(v)>.05));assert.ok(s.field.every(p=>Number.isFinite(p.learned)&&p.learned>0));assert.ok(Number.isFinite(s.rmseLearned));
  const initial=a.reset();assert.equal(initial.frame,0);assert.equal(initial.trainingCount,0);assert.equal(initial.trace.length,0);
});
