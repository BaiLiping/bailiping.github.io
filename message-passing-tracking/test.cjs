const {test}=require('node:test'),assert=require('node:assert/strict'),M=require('./model.js');
const close=(a,b,tol=1e-11)=>assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);
test('Every consistency factor enforces a=m if and only if b=j',()=>{
 for(let j=0;j<2;j++)for(let m=0;m<2;m++)for(let a=0;a<3;a++)for(let b=0;b<3;b++)assert.equal(M.psi(j,m,a,b),Number((a===m+1)===(b===j+1)));
});
test('Prediction, measurement weights, and existence branches follow the finite equations',()=>{
 const r=M.run();r.alpha.forEach(v=>close(M.sum(v),1));close(r.alpha[0][0],.145);close(r.alpha[1][0],.24);
 for(let j=0;j<2;j++)close(r.beta[j][0],r.alpha[j][0]+(1-r.alpha[j][0])*(1-r.p.pD));
 for(let m=0;m<2;m++){close(r.xi[m][0],1+M.sum(r.v[m].slice(1).map(v=>v[0])));assert.deepEqual(r.xi[m].slice(1),[1,1]);}
});
test('Exact association enumeration matches independent full-state enumeration',()=>{
 for(const parameters of [{},{z2:.3,pD:.99,birth:1},{z2:1.2,pD:0,birth:0}]){
  const r=M.run(parameters),old=Array.from({length:2},()=>[0,0,0]),fresh=Array.from({length:2},()=>[0,0,0]);let total=0;
  for(let y1=0;y1<3;y1++)for(let y2=0;y2<3;y2++)for(let n1=0;n1<3;n1++)for(let n2=0;n2<3;n2++)for(let a1=0;a1<3;a1++)for(let a2=0;a2<3;a2++){
   if(a1&&a1===a2)continue;const a=[a1,a2],y=[y1,y2],n=[n1,n2],b=[1,2].map(m=>a.indexOf(m)+1);
   const w=M.product(y.map((s,j)=>r.alpha[j][s]*r.q[j][s][a[j]]))*M.product(n.map((s,m)=>r.v[m][s][b[m]]));
   total+=w;y.forEach((s,j)=>old[j][s]+=w);n.forEach((s,m)=>fresh[m][s]+=w);
  }
  close(total,r.exact.z);assert.equal(r.exact.rows.length,7);
  old.forEach((v,j)=>v.forEach((w,s)=>close(w/total,r.exact.legacy[j][s])));fresh.forEach((v,m)=>v.forEach((w,s)=>close(w/total,r.exact.newTargets[m][s])));
 }
});
test('BP is exact on a one-target association tree',()=>{
 const beta=[[.2,1.3,.7]],xi=[[1.4,1],[1.2,1]],bp=M.infer(beta,xi,4),exact=M.exactAssociations(beta,xi);
 bp.aBeliefs.forEach((v,j)=>v.forEach((p,a)=>close(p,exact.aBeliefs[j][a])));bp.bBeliefs.forEach((v,m)=>v.forEach((p,b)=>close(p,exact.bBeliefs[m][b])));
});
test('The vector updates agree with the paper’s scalar ratio recursions',()=>{
 const r=M.run({iterations:10});
 for(const round of r.da.rounds)for(let j=0;j<2;j++)for(let m=0;m<2;m++){
  close(round.nu[j][m][m+1],r.xi[m][j+1]/(r.xi[m][0]+r.xi[m][2-j]*round.previousPhi[1-j][m][2-j]));
  close(round.phi[j][m][j+1],r.beta[j][m+1]/(r.beta[j][0]+r.beta[j][2-m]*round.nu[j][1-m][2-m]));
 }
});
test('Bounds, normalization and causal effects hold across controls',()=>{
 for(const z2 of [-.2,.3,1.2])for(const pD of [0,.85,1])for(const birth of [0,.3,2]){
  const r=M.run({z2,pD,birth,iterations:8});for(const row of [...r.legacy,...r.newTargets,...r.da.aBeliefs,...r.da.bBeliefs]){close(M.sum(row),1);row.forEach(v=>assert.ok(Number.isFinite(v)&&v>=0&&v<=1));}
  if(!birth)r.newTargets.forEach(v=>close(v[0],1));
 }
 const a=M.run(),b=M.run({z2:1});assert.notDeepEqual(a.beta,b.beta);assert.deepEqual(a.alpha,b.alpha);
 assert.ok(M.run({iterations:15}).da.rounds.at(-1).delta<1e-9);
});
