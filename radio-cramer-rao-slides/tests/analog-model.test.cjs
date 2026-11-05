'use strict';
const assert=require('node:assert/strict'),A=require('../analog/model.js');
const close=(a,b,t=1e-7)=>assert.ok(Math.abs(a-b)<=t*Math.max(1e-20,Math.abs(a),Math.abs(b)),`${a} != ${b}`);
let r=A.evaluate();assert.equal(r.symbols,3200);assert.equal(r.samples,10560000);close(r.usefulMs,26.4);assert.equal(r.rank,7);close(r.noise,3.8330638305071264e-15);
close(r.bounds[0],1e9/(2*Math.PI*r.beta*Math.sqrt(2*r.Gamma)));
for(const [schedule,count,rank] of [['fixed',1,3],['bs-only',100,5],['ue-only',32,5]]){const x=A.evaluate({schedule});assert.equal(x.symbols,count);assert.equal(x.rank,rank);}
for(const j of [1,2,3,4])assert.equal(A.evaluate({schedule:'fixed'}).bounds[j],Infinity);
assert.equal(A.evaluate({K:1}).bounds[0],Infinity);
assert.equal(A.evaluate({nty:1,ntz:1,nry:1,nrz:1}).rank,3);
let high=A.evaluate({powerDbm:A.defaults.powerDbm+10});r.bounds.forEach((b,i)=>close(high.bounds[i],b/Math.sqrt(10)));
// Independent raw-I/Q central differences. No engine steering/derivative helper is used.
const c={...A.defaults,nty:4,ntz:3,nry:3,nrz:2,bsAz:3,bsEl:3,ueAz:3,ueEl:2,K:7,BMHz:30};r=A.evaluate(c);
const times=(a,b)=>[a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]];
function steer(ny,nz,az,el){const a=az*Math.PI/180,e=el*Math.PI/180,out=[];for(let y=0;y<ny;y++)for(let z=0;z<nz;z++){const p=2*Math.PI*c.spacing*((y-(ny-1)/2)*Math.cos(e)*Math.sin(a)+(z-(nz-1)/2)*Math.sin(e));out.push([Math.cos(p),Math.sin(p)]);}return out;}
function dotH(a,b){let re=0,im=0;for(let i=0;i<a.length;i++){re+=a[i][0]*b[i][0]+a[i][1]*b[i][1];im+=a[i][0]*b[i][1]-a[i][1]*b[i][0];}return [re,im];}
const weights=(ny,nz,b)=>steer(ny,nz,...b).map(v=>v.map(x=>x/Math.sqrt(ny*nz)));
const vb=r.bs.map(b=>weights(c.nty,c.ntz,b)),wb=r.ue.map(b=>weights(c.nry,c.nrz,b));
function means(p){const ar=steer(c.nry,c.nrz,p[1],p[2]),at=steer(c.nty,c.ntz,p[3],p[4]),amp=10**(-p[5]/20)*Math.sqrt(10**((c.powerDbm-30)/10)/c.K),out=[];for(const [m,n] of r.pairs){const g=times(dotH(wb[n],ar),dotH(at,vb[m]));for(let k=0;k<c.K;k++){const phase=p[6]-2*Math.PI*(k-Math.floor(c.K/2))*r.df*p[0]*1e-9+(m+n+k)*Math.PI/2;out.push(times(g,[amp*Math.cos(phase),amp*Math.sin(phase)]));}}return out;}
const p=[17.3,c.rxAz,c.rxEl,c.txAz,c.txEl,c.lossDb,.27],h=[.001,.00001,.00001,.00001,.00001,.0001,.00001];
const D=p.map((_,j)=>{const plus=p.slice(),minus=p.slice();plus[j]+=h[j];minus[j]-=h[j];const a=means(plus),b=means(minus);return a.map((x,i)=>[(x[0]-b[i][0])/(2*h[j]),(x[1]-b[i][1])/(2*h[j])]);});
let worst=0;for(let i=0;i<7;i++)for(let j=0;j<7;j++){let x=0;for(let s=0;s<D[i].length;s++)x+=2/r.noise*(D[i][s][0]*D[j][s][0]+D[i][s][1]*D[j][s][1]);const err=Math.abs(x-r.J[i][j])/Math.sqrt(r.J[i][i]*r.J[j][j]);worst=Math.max(worst,err);assert.ok(err<2e-6,`Raw FIM mismatch ${i},${j}: ${err}`);}
for(let i=0;i<7;i++)for(let j=0;j<7;j++){let v=0;for(let k=0;k<7;k++)v+=r.J[i][k]*r.covariance[k][j];assert.ok(Math.abs(v-(i===j?1:0))<1e-6);}
assert.throws(()=>A.evaluate({K:0}));assert.throws(()=>A.evaluate({bsAz:1.5}));assert.throws(()=>A.evaluate({schedule:'unknown'}));
console.log(JSON.stringify({status:'passed',defaultSymbols:3200,defaultUsefulMs:26.4,fullRank:7,fixedPairRank:3,maxNormalizedRawFIMError:worst}));
