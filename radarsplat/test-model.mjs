import assert from 'node:assert/strict';
import {CONFIG,DEFAULT_SENSOR,INITIAL_SCENE,cloneScene,project,covariance,rotation,multiply,transpose,render,azimuthConvolve,rangeConvolve,peaks,wrap,TRAIN_INITIAL,trainLoss,trainStep} from './model.mjs';
let checks=0;
function test(name,fn){fn();checks++;console.log(`PASS ${name}`);}
function near(a,b,tol=1e-7){assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);}
function arrayNear(a,b,tol=1e-6){assert.equal(a.length,b.length);for(let i=0;i<a.length;i++)near(a[i],b[i],tol);}
const sensor={...DEFAULT_SENSOR};
test('covariance is symmetric and positive definite',()=>{for(const g of cloneScene()){const S=covariance(g);for(let i=0;i<3;i++)for(let j=0;j<3;j++)near(S[i][j],S[j][i]);for(const v of [[1,0,0],[0,1,0],[0,0,1],[1,-2,3]])assert.ok(v.reduce((sum,a,i)=>sum+a*S[i].reduce((s,b,j)=>s+b*v[j],0),0)>0);}});
test('spherical Jacobian agrees with central differences',()=>{const g={...INITIAL_SCENE[0],z:4.2},s={...sensor,x:2,y:-1,yaw:0},p=project(g,s),eps=1e-4;for(const [j,key] of ['x','y','z'].entries()){const a=project({...g,[key]:g[key]+eps},s),b=project({...g,[key]:g[key]-eps},s),values=[(a.r-b.r)/(2*eps),wrap(a.theta-b.theta)/(2*eps),(a.phi-b.phi)/(2*eps)];for(let i=0;i<3;i++)near(values[i],p.J[i][j],1e-8);}});
test('joint world rotation preserves sensor projection and reflectivity',()=>{const g={...INITIAL_SCENE[0]},R=rotation(67),p=project(g,sensor),xyz=multiply(R,[[g.x],[g.y],[g.z]]).map(r=>r[0]),q=project({...g,x:xyz[0],y:xyz[1],z:xyz[2],yaw:g.yaw+67,facing:g.facing+67},{...sensor,yaw:67});near(p.r,q.r);near(p.theta,q.theta);near(p.phi,q.phi);near(p.rho,q.rho);arrayNear(p.spherical.flat(),q.spherical.flat());});
test('radar origin singularity is handled without NaN',()=>{assert.equal(project({...INITIAL_SCENE[0],x:0,y:0,z:sensor.z},sensor),null);});
test('view gain and height change physical return weights',()=>{const g={...INITIAL_SCENE[0]},a=project(g,sensor),b=project({...g,z:8},sensor);assert.ok(b.gain<a.gain);assert.ok(a.rho>=0);});
const small={...CONFIG,H:100,W:100,Q:4};
test('the rendering model adds splats without alpha compositing',()=>{const a=render([INITIAL_SCENE[0]],sensor,'total',small),b=render([INITIAL_SCENE[1]],sensor,'total',small),both=render(INITIAL_SCENE.slice(0,2),sensor,'total',small);arrayNear(both.final,Float32Array.from(a.final,(v,i)=>v+b.final[i]));});
test('target plus noise equals total when clipping is inactive',()=>{const g={...INITIAL_SCENE[0],alpha:.6,eta:.2},t=render([g],sensor,'target',small),n=render([g],sensor,'noise',small),s=render([g],sensor,'total',small);arrayNear(s.final,Float32Array.from(t.final,(v,i)=>v+n.final[i]));});
test('clipped total is not equal to unclipped target plus noise',()=>{const g={...INITIAL_SCENE[0],alpha:.8,eta:.8},p=project(g,sensor);near(p.sigma,p.rho);assert.ok(p.rho*(g.alpha+g.eta)>p.sigma);});
test('circular azimuth convolution crosses the angular seam',()=>{const H=8,Q=4,W=1,input=new Float32Array(H*Q);input[input.length-1]=1;const out=azimuthConvolve(input,H,W,Q,45);assert.ok(out[0]>0);assert.ok(out[H-1]>0);});
test('range leakage preserves interior energy and zero width is identity',()=>{const a=new Float32Array(101);a[50]=1;arrayNear(a,rangeConvolve(a,1,101,0,.1));const b=rangeConvolve(a,1,101,.17,.1);near(b.reduce((s,x)=>s+x,0),1,1e-6);assert.ok(b[49]>0&&b[50]<1);near(b[49],b[51]);});
test('local peak detector uses an explicit threshold',()=>{assert.deepEqual(peaks([0,.1,0,.3,.3,0],.2),[3]);});
test('all renderer stages are finite, nonnegative and correctly sized',()=>{const r=render(cloneScene(),sensor);for(const name of ['footprint','elevation','azimuth','final']){assert.equal(r[name].length,CONFIG.H*CONFIG.W);assert.ok(r[name].every(v=>Number.isFinite(v)&&v>=0));}});
test('actual inverse updates monotonically reduce the toy objective',()=>{let p={...TRAIN_INITIAL},loss=trainLoss(p),initial=loss;for(let i=0;i<200;i++){const next=trainStep(p);assert.ok(next.loss<=loss+1e-10);p=next.p;loss=next.loss;}assert.ok(loss<initial*1e-4);near(p.r,21,.01);near(p.s,.65,.01);console.log(`  objective ${initial} -> ${loss}`);});
test('the experiment also fits with occupancy supervision disabled',()=>{let p={...TRAIN_INITIAL};const initial=trainLoss(p,0);for(let i=0;i<150;i++)p=trainStep(p,0).p;assert.ok(trainLoss(p,0)<initial*.01);});
console.log(`${checks} model checks passed.`);
