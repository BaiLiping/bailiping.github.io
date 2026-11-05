const {test}=require('node:test');
const assert=require('node:assert/strict');
const M=require('./model.js');
const rmse=(poses,truth)=>Math.sqrt(poses.reduce((s,p,i)=>s+(p[0]-truth[i][0])**2+(p[1]-truth[i][1])**2,0)/poses.length);
test('coordinate transforms round-trip and composition preserves conventions',()=>{
 const a=[2,-1,.7],b=[-.3,4,-.2],q=[5,-2];const p=M.transform(M.inverse(a),M.transform(a,q));assert.ok(Math.hypot(p[0]-q[0],p[1]-q[1])<1e-10);
 const one=M.transform(M.compose(a,b),q),two=M.transform(a,M.transform(b,q));assert.ok(Math.hypot(one[0]-two[0],one[1]-two[1])<1e-10);
});
test('bandwidth changes nominal resolution while preserving injected reflector positions',()=>{
 const a=M.rangeSpectrum(.15,.45),b=M.rangeSpectrum(1.2,.45);assert.equal(a.resolution/b.resolution,8);assert.equal(a.r1,b.r1);assert.equal(a.r2,b.r2);assert.deepEqual(a,M.rangeSpectrum(.15,.45));assert.ok(a.spectrum.every(p=>Number.isFinite(p[1])));
});
test('CFAR repeatable noise and nested decisions under stricter false-alarm settings',()=>{
 const a=M.cfar(2),b=M.cfar(6);assert.deepEqual(a.power,b.power);assert.ok(b.hits.every(i=>a.hits.includes(i)));assert.notDeepEqual(a.power,M.cfar(2,false,13).power);assert.equal(a.thresholds[0],null);assert.deepEqual(a,M.cfar(2));
});
test('robust velocity comparison uses identical observations and improves outlier resistance',()=>{
 const robust=M.velocity(4,1,.35,130,true),ordinary=M.velocity(4,1,.35,130,false);assert.deepEqual(robust.data,ordinary.data);const err=v=>Math.hypot(v[0]-4,v[1]-1);assert.ok(err(robust.v)<err(ordinary.v)/3);assert.ok(M.velocity(4,1,.35,4).condition>robust.condition*100);
});
test('near-initialized ICP recovers the synthetic transform without identities',()=>{
 const d=M.makeICP(),fit=M.icp(d.source,d.target,[0,0,0],40,2,true);assert.ok(Math.hypot(fit.t[0]-d.truth[0],fit.t[1]-d.truth[1])<.15);assert.ok(Math.abs(M.wrap(fit.t[2]-d.truth[2]))<M.rad(2));assert.deepEqual(fit,M.icp(d.source,d.target,[0,0,0],40,2,true));
});
test('loop optimization decreases its own objective, fixes the first pose, and re-places scans',()=>{
 const m=M.makeMission(),edges=M.graphEdges(m,.15);let poses=m.poses,cost=M.graphCost(poses,edges);const initial=rmse(poses,m.truth),point=M.transform(poses[24],m.scans[24][0]);
 for(let i=0;i<12;i++){const step=M.graphStep(poses,edges);assert.ok(step.cost<=cost+1e-8);poses=step.poses;cost=step.cost;}
 assert.deepEqual(poses[0],[0,0,0]);assert.ok(rmse(poses,m.truth)<initial*.5);assert.notDeepEqual(M.transform(poses[24],m.scans[24][0]),point);
 console.log({initialPositionRMSE:initial,optimizedPositionRMSE:rmse(poses,m.truth),cost});
});
test('heading-bias comparison preserves every simulated scan',()=>{
 const a=M.makeMission(0),b=M.makeMission(.35);assert.deepEqual(a.scans,b.scans);assert.deepEqual(a.truth,b.truth);assert.notDeepEqual(a.poses,b.poses);
});
