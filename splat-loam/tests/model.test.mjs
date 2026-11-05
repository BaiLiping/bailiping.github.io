import test from 'node:test';
import assert from 'node:assert/strict';
import {DEG,makeSplat,sphericalRay,intersectSplat,compositeHits,renderRay,makeRenderScene,makeMap,makeScan,registerStep,evaluateRegistration,poseError,rangeImage,pixelAngles,pixelForAngles} from '../live/model.mjs';
const near=(a,b,eps=1e-9)=>assert.ok(Math.abs(a-b)<eps,`${a} != ${b}`);
const front=makeSplat({id:'front',center:[5,0,0],tangentU:[0,1,0],tangentV:[0,0,1],scaleU:1,scaleV:1,opacity:0.5});
test('spherical directions are unit vectors, including nonzero sensor yaw',()=>{
  for(const a of [-120,0,60])for(const e of [-35,0,42]){const r=sphericalRay(a*DEG,e*DEG,25*DEG);near(r.reduce((s,x)=>s+x*x,0),1);}
});
test('exact ray-plane intersection preserves metric range and Gaussian local coordinates',()=>{
  const hit=intersectSplat(sphericalRay(Math.atan(1/5),0),front);
  near(hit.range,Math.sqrt(26));near(hit.point[0],5);near(hit.u,1);near(hit.gaussian,Math.exp(-0.5));
  assert.equal(intersectSplat([0,1,0],front),null);assert.equal(intersectSplat([-1,0,0],front),null);
});
test('slanted splat intersects its center and tilted off-axis plane exactly',()=>{
  const splat=makeRenderScene({tilt:43})[0], center=intersectSplat([1,0,0],splat);
  near(center.range,5);near(center.u,0);near(center.v,0);
  const ray=sphericalRay(12*DEG,8*DEG),h=intersectSplat(ray,splat);
  near(splat.normal.reduce((sum,n,i)=>sum+n*(h.point[i]-splat.center[i]),0),0);
});
test('range-sorted alpha compositing remains explicitly unnormalised',()=>{
  const r=compositeHits([{range:8,alpha:0.5,id:'back'},{range:5,alpha:0.5,id:'front'}]);
  assert.deepEqual(r.hits.map(h=>h.id),['front','back']);near(r.hits[0].weight,0.5);near(r.hits[1].weight,0.25);
  near(r.depth,4.5);near(r.expectedDepth,6);near(r.opacity,0.75);near(r.transmission,0.25);
  assert.notEqual(r.depth,r.depth/r.opacity);
});
test('opaque first intersection fully occludes following splats',()=>{
  const r=compositeHits([{range:3,alpha:1},{range:9,alpha:0.9}]);near(r.depth,3);near(r.opacity,1);near(r.hits[1].weight,0);
  const empty=renderRay([-1,0,0],[front]);near(empty.depth,0);near(empty.opacity,0);assert.equal(empty.expectedDepth,null);
});
test('a selected cell identifies the same ray and range at either teaching resolution',()=>{
  for(const [width,height] of [[72,32],[144,64]]) {
    const scene=makeRenderScene(), image=rangeImage(scene,{width,height,yaw:17});
    for(const [col,row] of [[0,0],[width-1,height-1],[23,19]]) {
      const angles=pixelAngles(image,col,row);
      assert.deepEqual(pixelForAngles(image,angles.azimuth,angles.elevation),{col,row});
      const ray=sphericalRay(angles.azimuth*DEG,angles.elevation*DEG,image.yaw*DEG);
      const hit=renderRay(ray,scene),index=row*width+col;
      near(image.depth[index],hit.depth,1e-5);
      if(hit.expectedDepth===null)assert.ok(Number.isNaN(image.expectedDepth[index]));
      else near(image.expectedDepth[index],hit.expectedDepth,1e-5);
    }
    assert.deepEqual(pixelForAngles(image,-100,100),{col:0,row:0});
    assert.deepEqual(pixelForAngles(image,100,-100),{col:width-1,row:height-1});
  }
});
test('footprint count is geometric, independent of opacity and occlusion',()=>{
  const options={width:3,height:1,azMin:-45,azMax:45,elMin:-1,elMax:1};
  // Three horizontal rays at -30°, 0°, 30° intersect x=5 at |y| <= 2.887.
  const image=rangeImage([front],options);
  assert.equal(image.footprintCounts.front,3);
  const transparent=rangeImage([{...front,opacity:0}],options);
  assert.equal(transparent.footprintCounts.front,3);
  assert.ok(transparent.expectedDepth.every(Number.isNaN));
  const blocker={...front,id:'blocker',center:[2,0,0],opacity:1,scaleU:100,scaleV:100};
  assert.equal(rangeImage([blocker,front],options).footprintCounts.front,3);
  assert.equal(rangeImage([{...front,scaleU:0.5}],options).footprintCounts.front,1);
});
test('3σ count does not truncate Gaussian tails in the renderer',()=>{
  const image=rangeImage([{...front,scaleU:0.5}],{width:3,height:1,azMin:-45,azMax:45,elMin:-1,elMax:1});
  assert.equal(image.footprintCounts.front,1);
  assert.ok(image.opacity[0]>0,'an outside-footprint ray still has a Gaussian tail weight');
  assert.ok(image.depth[0]>0);
});
test('geometric Gauss–Newton converges from a perturbed pose in corner geometry',()=>{
  const map=makeMap('corner'),scan=makeScan(map,{noise:0,truth:[0,0,0]});let pose=[0.7,-0.6,9*DEG];
  const initial=evaluateRegistration(scan,map,pose).cost;
  for(let i=0;i<12;i++){const r=registerStep(scan,map,pose);assert.ok(r.after.cost<=r.before.cost+1e-12);pose=r.pose;}
  assert.ok(evaluateRegistration(scan,map,pose).cost<initial/10000);assert.ok(poseError(pose).translation<1e-5);assert.ok(poseError(pose).yaw<1e-5);
});
test('optimizer can recover an unknown nonzero pose without being passed ground truth',()=>{
  const truth=[0.25,-0.4,3*DEG],map=makeMap(),scan=makeScan(map,{noise:0,truth});let pose=[0.85,-0.85,11*DEG];
  for(let i=0;i<12;i++)pose=registerStep(scan,map,pose).pose;
  assert.ok(poseError(pose,truth).translation<1e-5);assert.ok(poseError(pose,truth).yaw<1e-5);
});
test('corridor leaves translation along the corridor unobservable even when residual converges',()=>{
  const map=makeMap('corridor'),scan=makeScan(map,{noise:0});let pose=[0.9,-0.6,9*DEG],result;
  for(let i=0;i<12;i++){result=registerStep(scan,map,pose);pose=result.pose;}
  assert.equal(result.unobservableX,true);near(pose[0],0.9);assert.ok(Math.abs(pose[1])<1e-5);assert.ok(Math.abs(pose[2])<1e-5);
  assert.ok(result.after.rmse<1e-5);assert.ok(poseError(pose).translation>0.89);
});
test('noise is deterministic and residual settles at a physically plausible noise floor',()=>{
  const map=makeMap(),scan=makeScan(map,{noise:0.03});assert.deepEqual(scan,makeScan(map,{noise:0.03}));let pose=[0.7,-0.6,9*DEG];
  for(let i=0;i<12;i++)pose=registerStep(scan,map,pose).pose;
  const e=evaluateRegistration(scan,map,pose);assert.ok(e.rmse>0.01&&e.rmse<0.06);assert.ok(poseError(pose).translation<0.025);
});

