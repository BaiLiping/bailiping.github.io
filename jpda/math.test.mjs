import test from 'node:test';
import assert from 'node:assert/strict';
import {solveJPDA,solveScene,preset,enumerateAssignments,eye,gaussianLogDensity,det,multiply,transpose,add,scale,outer,zero} from './math.mjs';
const close=(a,b,tolerance=1e-10)=>assert.ok(Math.abs(a-b)<tolerance,`${a} != ${b}`);
const track=(x,pd=0.9)=>({mean:[x,0],cov:eye(1),pd});
test('enumeration includes misses and excludes repeated nonzero assignments',()=>{
  [1,3,7,13].forEach((count,m)=>{
    const events=enumerateAssignments(2,m);assert.equal(events.length,count);
    events.forEach(a=>assert.equal(new Set(a.filter(Boolean)).size,a.filter(Boolean).length));
  });
});
test('one target reduces to the analytic PDA formula',()=>{
  const t=track(0,0.8),z=[[0.3,-0.2]],noise=eye(1),lambda=0.03;
  const result=solveJPDA({tracks:[t],measurements:z,noise,lambda});
  const L=0.8*Math.exp(-(0.3**2+0.2**2)/4)/(4*Math.PI*lambda);
  close(result.beta[0][1],L/(0.2+L));close(result.beta[0][0],0.2/(0.2+L));
});
test('two symmetric targets share one return without double assignment',()=>{
  const result=solveJPDA({tracks:[track(-1),track(1)],measurements:[[0,0]],noise:eye(1),lambda:0.01});
  const L=0.9*Math.exp(-0.25)/(4*Math.PI*0.01);
  close(result.beta[0][1],L/(0.1+2*L));close(result.beta[1][1],result.beta[0][1]);
  assert.ok(result.beta[0][1]+result.beta[1][1]<1);
  assert.ok(result.independent[0][1]+result.independent[1][1]>1);
});
test('ratio weights agree with the independently expressed Poisson clutter model',()=>{
  const state=preset('ambiguous'),r=solveScene(state),m=r.measurements.length;
  const raw=r.events.map(e=>{
    const assigned=e.assignment.filter(Boolean).length;
    return state.lambda**(m-assigned)*e.assignment.reduce((v,j,t)=>v*(j?state.pd*Math.exp(gaussianLogDensity(r.measurements[j-1],r.tracks[t].mean,r.updates[t].S)):1-state.pd),1);
  });
  const total=raw.reduce((a,b)=>a+b,0);r.events.forEach((e,i)=>close(e.probability,raw[i]/total));
});
test('marginal rows normalize; measurement columns never exceed one',()=>{
  for(const key of ['shared','separated','ambiguous','missed']){
    const r=solveScene(preset(key));close(r.events.reduce((s,e)=>s+e.probability,0),1);
    r.beta.forEach(row=>close(row.reduce((a,b)=>a+b,0),1));
    r.measurements.forEach((_,j)=>{const sum=r.beta.reduce((s,row)=>s+row[j+1],0);assert.ok(sum<=1+1e-12);close(sum+r.clutter[j],1);});
  }
});
test('target and measurement permutations preserve the inference',()=>{
  const s=preset('ambiguous'),r=solveScene(s);
  const perm=solveJPDA({...r,tracks:[...r.tracks].reverse(),measurements:[...r.measurements].reverse()});
  r.beta.forEach((row,t)=>row.forEach((v,j)=>close(v,perm.beta[1-t][j===0?0:r.measurements.length-j+1])));
  r.updates.forEach((u,t)=>u.mean.forEach((v,i)=>close(v,perm.updates[1-t].mean[i])));
});
test('moment matching equals the PDA covariance formula, including innovation spread',()=>{
  const r=solveScene(preset('ambiguous'));
  r.updates.forEach((u,t)=>{
    const innovations=r.measurements.map(z=>z.map((x,i)=>x-r.tracks[t].mean[i]));
    const mean=[0,1].map(i=>innovations.reduce((s,v,j)=>s+r.beta[t][j+1]*v[i],0));
    let second=zero();innovations.forEach((v,j)=>{second=add(second,scale(outer(v),r.beta[t][j+1]));});
    const spread=multiply(multiply(u.K,add(second,scale(outer(mean),-1))),transpose(u.K));
    const expected=add(add(scale(r.tracks[t].cov,r.beta[t][0]),scale(u.corrected,1-r.beta[t][0])),spread);
    u.cov.forEach((row,i)=>row.forEach((v,j)=>close(v,expected[i][j])));
    assert.ok(det(u.cov)>0);assert.ok(u.spread[0][0]>=-1e-12&&det(u.spread)>=-1e-12);
  });
});
test('empty scan and zero detection probability retain predictions',()=>{
  for(const measurements of [[],[[0,0],[2,1]]]){
    const r=solveJPDA({tracks:[track(-1,0),track(1,0)],measurements,noise:eye(1),lambda:0.02});
    r.updates.forEach((u,t)=>{assert.deepEqual(u.mean,r.tracks[t].mean);assert.deepEqual(u.cov,r.tracks[t].cov);close(r.beta[t][0],1);});
  }
});
test('rare likelihoods remain finite; impossible observations and invalid inputs fail clearly',()=>{
  const r=solveJPDA({tracks:[track(0)],measurements:[[1e4,1e4]],noise:eye(1),lambda:1e-8});
  close(r.beta[0][0],1);assert.ok(r.events.every(e=>Number.isFinite(e.probability)));
  assert.throws(()=>solveJPDA({tracks:[track(0,1)],measurements:[],noise:eye(1),lambda:0.1}),/No feasible event/);
  assert.throws(()=>solveJPDA({tracks:[track(0)],measurements:[],noise:eye(1),lambda:0}),/Invalid/);
});
