'use strict';
const assert = require('node:assert/strict');
const M = require('../src/math.js');
let checks = 0;
const close=(a,b,tol=1e-10)=>{checks++;assert.ok(Math.abs(a-b)<=tol,`${a} != ${b} (tol ${tol})`);};
const ok=(v,message)=>{checks++;assert.ok(v,message);};
const r1=M.normalGenerator(42),r2=M.normalGenerator(42);
for(let i=0;i<20;i++)close(r1(),r2(),0);
const normal=M.normalGenerator(2026),means=[],firsts=[];
for(let i=0;i<100000;i++){const d=M.gaussianTrial(16,2,1,normal);means.push(d.mean);firsts.push(d.first);}
const a=M.statistics(means,1),b=M.statistics(firsts,1);
close(a.mean,1,.006);close(a.variance,.25,.004);close(b.variance,4,.07);
close(a.mse,(a.count-1)/a.count*a.variance+a.bias*a.bias,1e-12);
const near=M.shrinkage(.55,.35,2,16);close(near.variance,.075625);close(near.mse,.10043125);
close(M.shrinkage(.55,2,2,16).mse,.885625);
close(M.shrinkage(1,3,2,16).mse,.25);close(M.shrinkage(0,3,2,16).mse,9);
const anchors=[[-4,-3],[4,-3],[4,3],[-4,3]];
const g=M.rangeFisher(anchors,[0,0],.6);
close(g.a,64/9);close(g.b,0);close(g.c,4);close(g.peb,.625);ok(g.rank===2);
close(g.covariance[0],.140625);close(g.covariance[2],.25);
close(M.rangeFisher(anchors,[0,0],1.2).peb,1.25);
close(M.rangeFisher(anchors,[0,0],.6,true).peb,.625);
const angle=.72,R=p=>[p[0]*Math.cos(angle)-p[1]*Math.sin(angle),p[0]*Math.sin(angle)+p[1]*Math.cos(angle)];
const rotated=M.rangeFisher(anchors.map(R),[0,0],.6);close(rotated.peb,g.peb);close(rotated.hi,g.hi);close(rotated.lo,g.lo);
const shift=p=>[p[0]+1.4,p[1]-2.1];close(M.rangeFisher(anchors.map(shift),shift([0,0]),.6).peb,g.peb);
const line=M.rangeFisher([[-5,0],[-2,0],[2,0],[5,0]],[0,0],.6);
ok(line.rank===1&&line.covariance===null&&line.peb===Infinity);
ok(M.rangeFisher([[0,0]],[0,0],.6).invalid);
const cluster=[[-4,-1.3],[-4.8,-.3],[-4.4,.65],[-3.9,1.3]];
const known=M.rangeFisher(cluster,[0,0],.6),unknown=M.rangeFisher(cluster,[0,0],.6,true);
ok(unknown.peb>known.peb,'Unknown nuisance cannot improve position information');
// Numerical derivatives of each range recover the analytical Fisher matrix.
const p=[.4,.8],eps=1e-5;let aa=0,bb=0,cc=0;
for(const anchor of anchors){const range=q=>Math.hypot(q[0]-anchor[0],q[1]-anchor[1]);
 const x=(range([p[0]+eps,p[1]])-range([p[0]-eps,p[1]]))/(2*eps);
 const y=(range([p[0],p[1]+eps])-range([p[0],p[1]-eps]))/(2*eps);
 aa+=x*x/.36;bb+=x*y/.36;cc+=y*y/.36;}
const f=M.rangeFisher(anchors,p,.6);close(f.a,aa,1e-8);close(f.b,bb,1e-8);close(f.c,cc,1e-8);
// Random scenes: Schur information loss is positive semidefinite; inverse is valid.
const u=M.rng(88);
for(let k=0;k<60;k++){
 const as=Array.from({length:4},()=>{let a=2*Math.PI*u(),r=2+4*u();return[r*Math.cos(a),r*Math.sin(a)];});
 const f=M.rangeFisher(as,[0,0],.6),e=M.rangeFisher(as,[0,0],.6,true);
 const da=f.a-e.a,db=f.b-e.b,dc=f.c-e.c;
 ok(da>=-1e-10&&dc>=-1e-10&&da*dc-db*db>=-1e-8);
 if(e.rank===2)ok(e.peb>=f.peb-1e-9);
 if(f.rank===2){const [x,y,z]=f.covariance;close(f.a*x+f.b*y,1,1e-9);close(f.a*y+f.b*z,0,1e-9);close(f.b*y+f.c*z,1,1e-9);}
}
console.log(`${checks} numerical assertions passed. 100,000 Gaussian trials: mean variance ${a.variance.toFixed(6)} (theory 0.25), first-observation variance ${b.variance.toFixed(6)} (theory 4).`);
