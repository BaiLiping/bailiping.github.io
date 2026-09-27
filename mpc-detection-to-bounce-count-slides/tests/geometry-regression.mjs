/** Independent numerical regressions plus execution of corrected production helpers. */
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
import {fileURLToPath} from 'node:url';
import {dirname, join} from 'node:path';
const root=join(dirname(fileURLToPath(import.meta.url)),'..');
const read=p=>readFileSync(join(root,p),'utf8');
let count=0;
const check=(ok,label)=>{assert.ok(ok,label);count++;};
const close=(a,b,t=1e-7)=>Math.abs(a-b)<t;
const add=(a,b)=>a.map((v,i)=>v+b[i]);
const sub=(a,b)=>a.map((v,i)=>v-b[i]);
const scale=(a,t)=>a.map(v=>v*t);
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
const norm=a=>Math.hypot(...a);
const unit=a=>scale(a,1/norm(a));
const direction=a=>[Math.cos(a),Math.sin(a)];
const mirror=(p,w)=>sub(p,scale(w.n,2*(dot(w.n,p)-w.d)));
const reflect=(d,n)=>sub(d,scale(n,2*dot(d,n)));
const wrap=a=>Math.atan2(Math.sin(a),Math.cos(a));
function hitRayLine(o,d,q,n){const den=dot(d,n);if(Math.abs(den)<1e-9)return null;const t=dot(sub(q,o),n)/den;return {p:add(o,scale(d,t)),t};}
const BS=[0,0];
function path(x,chain){
  const p=x.slice(0,2),walls=[{n:direction(x[3]),d:x[4]},{n:direction(x[5]),d:x[6]}];
  const images=[BS]; for(const j of chain) images.push(mirror(images.at(-1),walls[j]));
  let q=p;const reverse=[];
  for(let k=chain.length-1;k>=0;k--){
    const wall=walls[chain[k]],v=sub(images[k+1],q);
    const t=(wall.d-dot(wall.n,q))/dot(wall.n,v);
    assert.ok(t>0&&t<1,'each unfolded intersection is forward');
    q=add(q,scale(v,t));reverse.push(q);
  }
  const points=[BS,...reverse.reverse(),p];
  const L=norm(sub(p,images.at(-1))),u=unit(sub(points.at(-2),p)),d=unit(sub(points[1],BS));
  const physical=points.slice(1).reduce((v,q,i)=>v+norm(sub(q,points[i])),0);
  assert.ok(close(L,physical),'unfolded distance equals physical path length');
  for(let k=1;k<points.length-1;k++){
    const incoming=unit(sub(points[k],points[k-1])),outgoing=unit(sub(points[k+1],points[k]));
    assert.ok(norm(sub(reflect(incoming,walls[chain[k-1]].n),outgoing))<1e-7,'reflection law');
  }
  return {L,u,d,points,z:[L,wrap(Math.atan2(u[1],u[0])-x[2]),Math.atan2(d[1],d[0])]};
}
function jacobian(f,x){const h=1e-5,base=f(x);return base.map((_,r)=>x.map((_,i)=>{const a=x.slice(),b=x.slice();a[i]+=h;b[i]-=h;return (f(a)[r]-f(b)[r])/(2*h);}));}
function rank(a,tolerance=1e-6){
  const A=a.map(r=>r.slice());let r=0;
  for(let c=0;c<A[0].length&&r<A.length;c++){
    let pivot=r;for(let i=r+1;i<A.length;i++)if(Math.abs(A[i][c])>Math.abs(A[pivot][c]))pivot=i;
    if(Math.abs(A[pivot][c])<tolerance)continue;
    [A[r],A[pivot]]=[A[pivot],A[r]];const d=A[r][c];for(let j=c;j<A[0].length;j++)A[r][j]/=d;
    for(let i=r+1;i<A.length;i++){const v=A[i][c];for(let j=c;j<A[0].length;j++)A[i][j]-=v*A[r][j];}r++;
  }return r;
}
const x=[2.5,4.7,0.1,0,4,0,-1];
const two=t=>[...path(t,[0]).z,...path(t,[0,1]).z];
const three=t=>[...two(t),...path(t,[0,1,0]).z];
check(rank(jacobian(two,x))===5,'single-R plus double-RL model: rank five, seven unknowns');
check(rank(jacobian(three,x))===7,'adding repeated-wall RLR: locally full rank at nondegenerate physical scene');
const [p1,p2,p3]=[[0],[0,1],[0,1,0]].map(c=>path(x,c));
const recovered=sub(add(scale(p1.u,p1.L),scale(p2.u,-p2.L)),scale(p3.u,p3.L));
check(norm(sub(recovered,x.slice(0,2)))<1e-9,'parallel-corridor identity fixes position with world-frame AoA');
function functionSource(text,name,start=0){
  const a=text.indexOf('function '+name+'(',start);assert.ok(a>=0,name+' function exists');
  let i=text.indexOf('{',a),depth=1,j=i+1;
  for(;depth&&j<text.length;j++){if(text[j]==='{')depth++;if(text[j]==='}')depth--;}
  assert.equal(depth,0);return text.slice(a,j);
}
const geometry=read('live/geometry/unknown-map.js');
const corridorStart=geometry.indexOf('/* ---- section 4.5:');
const stripSource=functionSource(geometry,'strip3',corridorStart);
function candidate(a,heading=0){
  const u1=direction(Math.atan2(p1.u[1],p1.u[0])+heading),u2=direction(Math.atan2(p2.u[1],p2.u[0])+heading);
  const Pw=add(BS,scale(p1.d,a*p1.L)),U1=sub(Pw,scale(u1,(1-a)*p1.L));
  const nA=unit(add(p1.d,u1));const H=hitRayLine(BS,p2.d,Pw,nA);assert.ok(H?.t>0);
  const e=reflect(p2.d,nA),rem=p2.L-H.t,v=add(e,u2),o=sub(H.p,scale(u2,rem));
  const m=dot(sub(U1,o),v)/dot(v,v);assert.ok(m>0&&m<rem);
  const P2s=add(H.p,scale(e,m)),E2=add(H.p,scale(e,rem)),nB=unit(sub(U1,E2));
  const pp={...p3,u:direction(Math.atan2(p3.u[1],p3.u[0])+heading)};
  return runInNewContext(`(${stripSource})(pp)`,{doubleFeasible:true,BS,Pw,U1,nA,P2s,nB,pp,hitRayLine,unit,Math,Number});
}
const trueFraction=norm(sub(p1.points[1],BS))/p1.L;
check(candidate(trueFraction).feasible,'actual corrected demo accepts the reference repeated-wall path');
const falseSlice=candidate(trueFraction+0.03);
check(!falseSlice.feasible&&falseSlice.reason.includes('ORIGINAL'),'actual demo rejects sliding slice that invents a new third wall');
check(geometry.includes('dist(U1,Xraw)<1e-5')&&geometry.includes('dist(U1,X2raw)<1e-5')&&geometry.includes('dist(U1,X3raw)<1e-5'),'corner demos enforce a common UE');
console.log(`PASS: ${count} geometry regression checks.`);
