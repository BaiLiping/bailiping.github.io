import test from 'node:test';
import assert from 'node:assert/strict';
import {withOptimality,optimalitySlideIds} from './optimality.mjs';
const near=(a,b,t=1e-10)=>assert.ok(Math.abs(a-b)<t,`${a} != ${b}`);
const sum=a=>a.reduce((x,y)=>x+y,0);
const kl=(p,q)=>sum(p.map((x,i)=>x*Math.log(x/q[i])));
const norm=a=>{const z=sum(a);return a.map(x=>x/z);};
const weights=[.3,.7], inputs=[[.2,.3,.5],[.5,.4,.1]],q=[.4,.25,.35];
const aa=inputs[0].map((_,k)=>sum(inputs.map((p,i)=>weights[i]*p[k])));
const rawG=inputs[0].map((_,k)=>Math.exp(sum(inputs.map((p,i)=>weights[i]*Math.log(p[k])))));
const geo=norm(rawG);
test('AA and GCI objective-gap identities',()=>{
 const JA=q=>sum(inputs.map((p,i)=>weights[i]*kl(p,q)));
 const JG=q=>sum(inputs.map((p,i)=>weights[i]*kl(q,p)));
 near(JA(q)-JA(aa),kl(aa,q));near(JG(q),kl(q,geo)-Math.log(sum(rawG)));
 assert.ok(JA(q)>JA(aa));assert.ok(JG(q)>JG(geo));
});
test('Bayesian free-energy identity counts the prior once',()=>{
 const p0=[.1,.4,.5], L=[[.3,.8,.4],[.9,.2,.6]],raw=p0.map((v,k)=>v*L[0][k]*L[1][k]),post=norm(raw);
 const J=kl(q,p0)-sum(L.map(l=>sum(q.map((v,k)=>v*Math.log(l[k])))));
 near(J,kl(q,post)-Math.log(sum(raw)));
 const local=L.map(l=>norm(p0.map((v,k)=>v*l[k])));
 const corrected=norm(p0.map((v,k)=>local[0][k]*local[1][k]/v));
 corrected.forEach((v,k)=>near(v,post[k]));
});
test('known-correlation scalar covariance square and CI bound',()=>{
 for(const p1 of [.4,1,3])for(const p2 of [.7,2,5])for(const rho of [-.99,-.5,0,.5,.99]){
  const c=rho*Math.sqrt(p1*p2),S=p1+p2-2*c,B=p1-c,W=B/S;
  const cov=w=>p1-2*w*B+w*w*S;
  for(const w of [-.2,0,.2,.7,1,1.2])near(cov(w),cov(W)+(w-W)**2*S);
  for(const w of [0,.2,.5,.8,1]){const P=1/(w/p1+(1-w)/p2),a=P*w/p1,b=P*(1-w)/p2;assert.ok(a*a*p1+b*b*p2+2*a*b*c<=P+1e-12);}
 }
});
test('Gaussian moment projection cross-entropy gap',()=>{
 const xs=[-2,.5,4],p=[.2,.5,.3],mu=sum(xs.map((x,i)=>p[i]*x)),V=sum(xs.map((x,i)=>p[i]*(x-mu)**2));
 const H=(m,P)=>.5*Math.log(2*Math.PI*P)+sum(xs.map((x,i)=>.5*p[i]*(x-m)**2/P));
 const m=1.7,P=4.3;
 near(H(m,P)-H(mu,V),.5*(Math.log(P/V)+(V+(mu-m)**2)/P-1));
});
test('one-dimensional quantile square and Gaussian standard deviation',()=>{
 const Qs=[[0,1,3],[1,2,5]],Q=[.2,1.8,4],bar=Q.map((_,k)=>sum(Qs.map((x,i)=>weights[i]*x[k])));
 const objective=X=>sum(Qs.map((x,i)=>weights[i]*sum(x.map((v,k)=>(v-X[k])**2))));
 near(objective(Q)-objective(bar),sum(Q.map((v,k)=>(v-bar[k])**2)));
 const mus=[-2,3],sigmas=[1,4],m=sum(mus.map((v,i)=>weights[i]*v)),s=sum(sigmas.map((v,i)=>weights[i]*v));
 const J=(a,b)=>sum(mus.map((v,i)=>weights[i]*((a-v)**2+(b-sigmas[i])**2)));
 near(J(.5,2)-J(m,s),(.5-m)**2+(2-s)**2);
});
test('Bernoulli decomposition, existence odds, and AA spatial weights',()=>{
 const r=[.75,.9],f=inputs.map((p,i)=>[1-r[i],...p.map(x=>r[i]*x)]),cand=[.4,...q.map(x=>.6*x)];
 const dB=(a,b)=>a*Math.log(a/b)+(1-a)*Math.log((1-a)/(1-b));
 near(kl(cand,f[0]),dB(.6,r[0])+.6*kl(q,inputs[0]));
 const pooled=norm(f[0].map((_,k)=>Math.exp(sum(f.map((p,i)=>weights[i]*Math.log(p[k]))))));
 const logOdds=sum(r.map((v,i)=>weights[i]*Math.log(v/(1-v))))+Math.log(sum(rawG));
 near(1-pooled[0],1/(1+Math.exp(-logOdds)));
 const rA=sum(r.map((v,i)=>weights[i]*v)),pA=q.map((_,k)=>sum(inputs.map((p,i)=>weights[i]*r[i]*p[k]))/rA);
 const fA=f[0].map((_,k)=>sum(f.map((p,i)=>weights[i]*p[k])));
 near(1-fA[0],rA);pA.forEach((v,k)=>near(v,fA[k+1]/rA));
});
test('PPP reverse-KL geometric intensity and forward-KL projection gaps',()=>{
 const ds=[[.4,2,1],[2,.5,3]],D=[1,1,2],A=D.map((_,k)=>sum(ds.map((d,i)=>weights[i]*d[k]))),G=D.map((_,k)=>Math.exp(sum(ds.map((d,i)=>weights[i]*Math.log(d[k])))));
 const gkl=(a,b)=>sum(a.map((v,k)=>v*Math.log(v/b[k])-v+b[k]));
 const JG=d=>sum(ds.map((v,i)=>weights[i]*gkl(d,v)));
 const JA=d=>sum(ds.map((v,i)=>weights[i]*gkl(v,d)));
 near(JG(D)-JG(G),gkl(D,G));near(JA(D)-JA(A),gkl(A,D));
});
test('geometric consensus reaches stationary-weight KL center',()=>{
 const A=[[.8,.2],[.4,.6]],pi=[2/3,1/3];let beliefs=structuredClone(inputs);
 for(let t=0;t<80;t++)beliefs=A.map(row=>norm(q.map((_,k)=>Math.exp(sum(row.map((w,j)=>w*Math.log(beliefs[j][k])))))));
 const target=norm(q.map((_,k)=>Math.exp(sum(pi.map((w,j)=>w*Math.log(inputs[j][k]))))));
 beliefs.forEach(p=>p.forEach((v,k)=>near(v,target[k])));
});
// In the repository this imports the real authoring source. Local rendering can
// opt into an explicitly identified fixture; no remote browser claim is made.
const base=await import(process.env.OPTIMALITY_BASE||'./bento-deck.mjs');
test('40-slide integration preserves originals, links, and six live indexes',()=>{
 const before=JSON.stringify(base);const out=withOptimality(base.deck,base.inlineLiveMap);
 assert.equal(JSON.stringify(base),before);assert.equal(out.deck.slides.length,base.deck.slides.length+11);
 assert.equal(out.deck.slides.length,40);assert.equal(out.inlineLiveMap.length,6);
 const ids=new Set(out.deck.slides.map(s=>s.id));assert.equal(ids.size,40);
 optimalitySlideIds.forEach(id=>assert.ok(ids.has(id)));
 for(const [i,s] of out.deck.slides.entries()){
  assert.equal(new Set(s.elements.map(e=>e.id)).size,s.elements.length);
  assert.equal(s.elements.find(e=>e.id==='footer-page').html,`${String(i+1).padStart(2,'0')} / 40`);
  for(const e of s.elements)if(e.link&&!e.link.startsWith('http'))assert.ok(ids.has(e.link),e.link);
 }
 for(const lab of out.inlineLiveMap)assert.equal(out.deck.slides[lab.slideIndex].id,lab.slide);
 assert.throws(()=>withOptimality(out.deck,out.inlineLiveMap),/already/);
});
