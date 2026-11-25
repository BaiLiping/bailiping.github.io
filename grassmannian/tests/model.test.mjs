import test from 'node:test';
import assert from 'node:assert/strict';
import * as M from '../model.mjs';
const near=(a,b,t=1e-10)=>assert.ok(Math.abs(a-b)<t,`${a} != ${b}`);
test('rotating, reflecting, or flipping a basis preserves its orthogonal projector',()=>{
 const Q=M.planeBasis(.4),P=M.projector(Q);
 for(const O of [M.rotation(.8),[[-1,0],[0,1]],[[-1,0],[0,-1]]]){
  const Y=M.mul(Q,O);near(M.norm(M.add(P,M.projector(Y),-1)),0);near(M.orthError(Y),0);
  near(M.distances(Q,Y).projection,0,5e-8);
 }
 near(M.norm(M.add(M.mul(P,P),P,-1)),0);
});
test('principal angles and distances agree with independent analytic Gr(2,4) cases',()=>{
 for(const [a,b] of [[0,0],[0,Math.PI/3],[.2,.8],[Math.PI/2,Math.PI/2]]){
  const v=M.anglePair(a,b,.71);
  v.angles.forEach((x,i)=>near(x,[a,b][i],5e-8));
  near(v.geodesic,Math.hypot(a,b),5e-8);
  near(v.projection,M.norm(M.add(M.projector(v.Q),M.projector(v.Y),-1))/Math.sqrt(2),5e-8);
 }
});
test('two planes in R3 necessarily share a direction',()=>{
 const Q=M.planeBasis(.1),Y=M.planeBasis(.9),theta=M.principalAngles(Q,Y);
 near(theta[0],0,3e-8);near(theta[1],.8);
});
test('symmetric eigensolver satisfies its residual and orthogonality',()=>{
 const A=[[4,1,-.4],[1,2,.5],[-.4,.5,1]],e=M.eigenSymmetric(A);
 near(M.orthError(e.vectors),0);
 const EV=e.vectors.map(r=>r.map((v,j)=>v*e.values[j]));
 near(M.norm(M.add(M.mul(A,e.vectors),EV,-1)),0);
});
test('exponential has correct tangent, stays orthonormal, and follows analytic principal rotations',()=>{
 const Q=[[1,0],[0,1],[0,0],[0,0]],D=[[0,0],[0,0],[.3,0],[0,.7]];
 for(const t of [0,.2,.5,1]) {
  const Y=M.exponential(Q,D,t),expected=M.anglePair(.3*t,.7*t).Y;
  near(M.orthError(Y),0);near(M.norm(M.add(Y,expected,-1)),0);
  near(M.distances(Q,Y).geodesic,t*M.norm(D),4e-8);
 }
 const h=1e-6,fd=M.scale(M.add(M.exponential(Q,D,h),M.exponential(Q,D,-h),-1),1/(2*h));
 near(M.norm(M.add(fd,D,-1)),0,1e-9);
 // Rank-deficient tangent, including n < 2k.
 const Q3=M.planeBasis(.2),D3=M.horizontal(Q3,[[.3,.2],[.1,0],[.4,.8]]);
 near(M.orthError(M.exponential(Q3,D3)),0,1e-9);
});
test('line geodesics identify antipodal representatives and retain endpoints',()=>{
 const u=[1,0,0],v=[Math.cos(.8),Math.sin(.8),0];
 for(const t of [0,.5,1]) {
  const a=M.lineGeodesic(u,v,t),b=M.lineGeodesic(u,v.map(x=>-x),t);
  near(Math.hypot(...a.point),1);near(M.norm(M.add([a.point],[b.point],-1)),0);
  near(M.principalAngles(M.transpose([u]),M.transpose([a.point]))[0],t*.8,2e-8);
 }
 near(M.lineGeodesic(u,[-1,0,0],.5).theta,0);
 near(M.lineGeodesic(u,[0,1,0],1).theta,Math.PI/2);
});
test('PCA gradient matches a finite-difference directional derivative',()=>{
 const {C}=M.pcaData(),Q=M.initialPcaBasis(),G=M.pcaGradient(Q,C);
 const D=M.horizontal(Q,[[.1,.8],[-.3,.2],[.2,-.4]]),h=1e-6;
 const fd=(-M.pcaScore(M.exponential(Q,D,h),C)+M.pcaScore(M.exponential(Q,D,-h),C))/(2*h);
 near(fd,M.dot(G.flat(),D.flat()),1e-7);near(M.norm(M.mul(M.transpose(Q),G)),0);
});
test('PCA uses actual plotted covariance, improves monotonically, and reaches the eigenspace',()=>{
 const data=M.pcaData();data.points[0].forEach((_,j)=>near(data.points.reduce((s,p)=>s+p[j],0),0));
 const e=M.eigenSymmetric(data.C);e.values.forEach((v,i)=>near(v,data.lambdas[i]));
 let Q=M.initialPcaBasis(),previous=M.pcaScore(Q,data.C);
 for(let k=0;k<160;k++){const r=M.pcaStep(Q,data.C);assert.ok(r.score>=previous-1e-12);Q=r.Q;previous=r.score;near(M.orthError(Q),0);}
 near(previous,data.optimum,1e-9);
 near(M.norm(M.add(M.projector(Q),M.projector(data.target),-1)),0,1e-6);
});
test('a repeated PCA boundary eigenvalue admits distinct optimal subspaces',()=>{
 const {C,E,optimum}=M.pcaData(1);
 const Q=E.map(r=>[r[0],r[1]]),Y=E.map(r=>[r[0],r[2]]);
 near(M.pcaScore(Q,C),optimum);near(M.pcaScore(Y,C),optimum);
 near(M.distances(Q,Y).projection,1,1e-7);
});
