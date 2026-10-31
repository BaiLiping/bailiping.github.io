// Lusk & How, arXiv:2205.08556v1, Eqs. (2)-(5), (8)-(9), Proposition 1.
// The browser solves the tiny discrete density problem exhaustively, not with CLIPPER.
import * as M from './model.mjs';
export function affineEmbedding(object,origin=[0,0,0],rho=1) {
  if(!(rho>0))throw Error('rho must be positive');
  const b=object.b.map((x,i)=>(x-origin[i])/rho),P=M.projector(object.A);
  const perpendicular=b.map((x,i)=>x-M.dot(P[i],b));
  const s=Math.hypot(1,...perpendicular);
  return [...object.A.map((row,i)=>[...row,perpendicular[i]/s]),[...Array(object.A[0].length).fill(0),1/s]];
}
export function objectDistance(a,b,{rho=3,mode='shifted'}={}) {
  if(mode==='directions')return Math.hypot(...M.principalAngles(a.A,b.A));
  // Both affine coordinates are shifted BEFORE perpendicular displacements
  // are computed. The same first-correspondence anchor is used in both scans.
  const origin=mode==='shifted'?a.b:[0,0,0];
  return Math.hypot(...M.principalAngles(affineEmbedding(a,origin,rho),affineEmbedding(b,origin,rho)));
}
export function transformObject(o,R,t) {
  return {...o,A:M.mul(R,o.A),b:R.map((r,i)=>M.dot(r,o.b)+t[i])};
}
export function axisRotation(axis,angle) {
  const [x,y,z]=M.unit(axis),c=Math.cos(angle),s=Math.sin(angle),a=1-c;
  return [[c+x*x*a,x*y*a-z*s,x*z*a+y*s],[y*x*a+z*s,c+y*y*a,y*z*a-x*s],[z*x*a-y*s,z*y*a+x*s,c+z*z*a]];
}
function basisFromNormal(normal) {
  const n=M.unit(normal),u=M.unit(M.cross(n,Math.abs(n[0])<.8?[1,0,0]:[0,1,0]));
  return M.transpose([u,M.cross(n,u)]);
}
function line(id,b,d){return {id,type:'line',b,A:M.transpose([M.unit(d)])};}
function plane(id,b,n){return {id,type:'plane',b,A:basisFromNormal(n)};}
function random(seed) { let x=seed>>>0;return()=>{x=(1664525*x+1013904223)>>>0;return x/4294967296;}; }
export function makeScene({angle=55,translation=7,noise=0,clutter=true,seed=19}={}) {
  const source=[line('L1',[-3,-1,0],[.1,.2,1]),line('L2',[1.1,-2,.3],[.2,-.1,1]),line('L3',[2.2,2,0],[-.2,.3,1]),
    plane('P1',[0,0,-1.4],[.1,.2,1]),plane('P2',[-3.8,0,1],[1,.1,.1]),plane('P3',[0,3.4,.7],[.2,1,-.1])];
  const R=axisRotation([.25,-.15,1],M.rad(angle)),t=[translation,-.35*translation,.2*translation],rng=random(seed);
  let target=source.map(o=>transformObject(o,R,t)).map(o=>{
    const N=axisRotation([rng()+.1,rng()+.1,rng()+.1],(rng()-.5)*noise);
    return {...o,A:M.mul(N,o.A),b:o.b.map(x=>x+2*noise*(rng()-.5))};
  });
  if(clutter)target.push(transformObject(line('outlier-line',[-.8,3.9,.5],[.5,.3,1]),R,t),transformObject(plane('outlier-plane',[1.5,-3,2],[.6,-.4,1]),R,t));
  const order=clutter?[4,1,6,3,7,0,5,2]:[4,1,3,0,5,2];
  target=order.map(i=>target[i]);
  return {source,target,R,t};
}
export function buildConsistency(source,target,{rho=3,mode='shifted',epsilon=.16,sigma=.05}={}) {
  const candidates=[];
  source.forEach((a,i)=>target.forEach((b,j)=>{if(a.type===b.type)candidates.push({i,j});}));
  const n=candidates.length,weights=M.eye(n),residual=M.zeros(n,n),pairs=[];
  for(let u=0;u<n;u++)for(let v=u+1;v<n;v++){
    const a=candidates[u],b=candidates[v];
    if(a.i===b.i||a.j===b.j){residual[u][v]=residual[v][u]=null;continue;}
    const d1=objectDistance(source[a.i],source[b.i],{rho,mode});
    const d2=objectDistance(target[a.j],target[b.j],{rho,mode});
    const c=Math.abs(d1-d2),w=c<epsilon?Math.exp(-c*c/(2*sigma*sigma)):0;
    weights[u][v]=weights[v][u]=w;residual[u][v]=residual[v][u]=c;
    pairs.push({u,v,d1,d2,c,w});
  }
  return {candidates,weights,residual,pairs};
}
export function densestClique(weights) {
  // Enumerate EVERY feasible clique, including nonmaximal cliques: adding a weak
  // node can lower uᵀMu/uᵀu. Zero off-diagonals are hard incompatibilities.
  let best=[],density=0,visited=0;
  function visit(chosen,available,sum) {
    if(chosen.length){visited++;const score=sum/chosen.length;
      if(score>density+1e-12||(Math.abs(score-density)<1e-12&&chosen.length>best.length)){best=[...chosen];density=score;}}
    for(let p=0;p<available.length;p++){
      const u=available[p],next=available.slice(p+1).filter(v=>weights[u][v]>0);
      visit([...chosen,u],next,sum+weights[u][u]+2*chosen.reduce((s,v)=>s+weights[u][v],0));
    }
  }
  visit([],Array.from({length:weights.length},(_,i)=>i),0);
  return {selected:best,density,visited};
}
export function associate(scene,options={}) {
  // Truth IDs are deliberately absent from graph construction and optimization.
  const graph=buildConsistency(scene.source,scene.target,options),solution=densestClique(graph.weights);
  const matches=solution.selected.map(u=>graph.candidates[u]);
  const correct=matches.filter(({i,j})=>scene.source[i].id===scene.target[j].id).length;
  return {...graph,...solution,matches,correct};
}
