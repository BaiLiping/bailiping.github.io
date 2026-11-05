import test from 'node:test';
import assert from 'node:assert/strict';
import {createJoint,evaluateJoint,stepJoint,projectJointSplats} from '../live/joint-model.mjs';

const close=(a,b,t=1e-8,label='')=>assert.ok(Math.abs(a-b)<=t,`${label}: ${a} versus ${b}`);

test('joint loss averages two measured-image L1 + SSIM terms with no pose labels',()=>{
  const s=createJoint({size:16}),r=evaluateJoint(s);
  close(r.loss,.8*r.l1+.2*(1-r.ssim));
  for(const c of r.cameras){
    close(c.loss,.8*c.l1+.2*c.dssim);
    close(c.dssim,1-c.ssim);
    const l1=c.image.reduce((sum,v,i)=>sum+Math.abs(v-c.target[i]),0)/c.image.length;
    close(c.l1,l1);
  }
  assert.deepEqual(r.cameras[0].pose,[0,0,0]);
  assert.equal(r.cameras[0].anchored,true);
  assert.ok(!('reference' in s));
  assert.ok(!('truePose' in s));
  assert.ok(!('trueCenters' in s));
});

test('both analytic gradient blocks match central differences for L1, mixture, and SSIM alone',()=>{
  const h=1e-6;
  for(const lambda of [0,.2,1]){
    const s=createJoint({size:16,lambda}),a=evaluateJoint(s).gradients;
    for(const block of ['centers','pose'])for(let i=0;i<s[block].length;i++){
      const x=s[block][i];
      s[block][i]=x+h; const plus=evaluateJoint(s).loss;
      s[block][i]=x-h; const minus=evaluateJoint(s).loss;
      s[block][i]=x;
      close(a[block][i],(plus-minus)/(2*h),3e-7,`lambda ${lambda}, ${block}[${i}]`);
    }
  }
});

test('pose rotation changes anisotropic covariance as well as the projected center',()=>{
  const s=createJoint({size:16}),p0=projectJointSplats(s.centers,[0,0,0]);
  const p1=projectJointSplats(s.centers,[0,0,.3]);
  for(let i=0;i<p0.length;i++){
    close(p1[i].angle-p0[i].angle,.3);
    close(p1[i].sx,p0[i].sx);
    close(p1[i].sy,p0[i].sy);
    assert.notEqual(p1[i].sx,p1[i].sy);
  }
});

test('simultaneous and alternating updates fit both unknown blocks using image loss',()=>{
  for(const schedule of ['simultaneous','alternating']){
    const s=createJoint({size:16,schedule}),initial=evaluateJoint(s);
    const r=stepJoint(s,{steps:100});
    assert.ok(r.loss<initial.loss*.01,`${schedule}: ${initial.loss} -> ${r.loss}`);
    assert.ok(r.pose.some((v,i)=>Math.abs(v-s.initialPose[i])>.05));
    assert.ok(r.centers.some((v,i)=>Math.abs(v-s.initialCenters[i])>.03));
    assert.deepEqual(r.cameras[0].pose,[0,0,0]);
    for(let i=1;i<s.history.length;i++)assert.ok(s.history[i].loss<=s.history[i-1].loss+1e-14);
    close(s.history.at(-1).loss,evaluateJoint(s).loss);
  }
});

test('pose-only and splat-only modes preserve the other block exactly',()=>{
  for(const mode of ['pose','splats']){
    const s=createJoint({size:16,mode}),initial=evaluateJoint(s);
    const centers=Array.from(s.centers),pose=Array.from(s.pose);
    const r=stepJoint(s,{steps:15});
    if(mode==='pose')assert.deepEqual(Array.from(s.centers),centers);
    else assert.deepEqual(Array.from(s.pose),pose);
    assert.ok(r.loss<initial.loss);
    assert.deepEqual(r.cameras[0].pose,[0,0,0]);
  }
});

test('observations remain fixed and fabricated truth fields cannot affect optimization',()=>{
  const a=createJoint({size:16}),b=createJoint({size:16});
  const before=a.targets.map(x=>Float64Array.from(x));
  b.reference={centers:Array(8).fill(100),pose:[99,-82,3]};
  b.truePose=[1,2,3]; b.trueCenters=Array(8).fill(-100);
  const ar=stepJoint(a,{steps:15}),br=stepJoint(b,{steps:15});
  assert.deepEqual(ar.centers,br.centers);
  assert.deepEqual(ar.pose,br.pose);
  assert.deepEqual(ar.gradients,br.gradients);
  close(ar.loss,br.loss);
  assert.deepEqual(a.targets,before);
});

test('control changes retain the fit and start a comparable loss-history phase',()=>{
  const s=createJoint({size:16});stepJoint(s,{steps:4});
  const centers=Array.from(s.centers),pose=Array.from(s.pose);
  const r=stepJoint(s,{steps:0,lambda:0,mode:'pose',schedule:'alternating'});
  assert.deepEqual(r.centers,centers);assert.deepEqual(r.pose,pose);
  close(r.loss,r.l1);assert.equal(s.history.length,1);assert.equal(s.phase,1);
  assert.equal(s.iteration,4);assert.equal(r.mode,'pose');
});
