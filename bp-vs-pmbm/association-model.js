/* Shared, dependency-free one-scan association benchmark.
 * Williams & Lau (2014), equations (20)-(23), with unnormalised miss weights.
 * This is NOT a complete PMBM tracker. See index.html for the model boundary.
 * UMD: browser global AssociationModel; CommonJS import for build and tests.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.AssociationModel = factory();
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const GATE = -2 * Math.log(0.01); // Exact 99% chi-square quantile, two dimensions.
  const DEFAULT = {
    T: [{x:285,y:205,S:[[520,140],[140,340]]},
        {x:352,y:232,S:[[460,-120],[-120,480]]},
        {x:318,y:158,S:[[620,0],[0,300]]}],
    Z: [{x:318,y:198},{x:322,y:215},{x:314,y:186},{x:560,y:120}],
    PD: 0.9, c: 5e-5
  };
  const clone = x => JSON.parse(JSON.stringify(x));
  const cp = A => A.map(row => row.slice());
  const sum = a => a.reduce((s,x) => s+x,0);
  function dimensions(L) {
    if (!Array.isArray(L) || !L.length || !Array.isArray(L[0]) || !L[0].length)
      throw new TypeError('At least one track and a missed-detection column are required.');
    const m = L[0].length-1;
    if (L.some(row => row.length !== m+1 || !(row[0]>0) || row.some(x => !Number.isFinite(x) || x<0)))
      throw new RangeError('Weights must be finite, nonnegative, rectangular, with strictly positive miss weights.');
    return {n:L.length,m};
  }
  function gaussian2(dx,dy,S) {
    const det=S[0][0]*S[1][1]-S[0][1]*S[1][0];
    if (!(det>0 && S[0][0]>0) || Math.abs(S[0][1]-S[1][0])>1e-10)
      throw new RangeError('The covariance must be symmetric positive definite.');
    const d2=(dx*(S[1][1]*dx-S[0][1]*dy)+dy*(-S[1][0]*dx+S[0][0]*dy))/det;
    return {pdf:Math.exp(-0.5*d2)/(2*Math.PI*Math.sqrt(det)),d2};
  }
  function buildWeights(scene, gated=true) {
    if (!(scene.PD>=0 && scene.PD<1 && scene.c>0 && Number.isFinite(scene.c)))
      throw new RangeError('This benchmark requires 0 <= PD < 1 and positive clutter intensity.');
    const gate=[];
    const L=scene.T.map((track,i)=>{
      gate[i]=[];
      return [1-scene.PD,...scene.Z.map((z,j)=>{
        const g=gaussian2(z.x-track.x,z.y-track.y,track.S);
        gate[i][j]=!gated || g.d2<=GATE;
        return gate[i][j]?scene.PD*g.pdf/scene.c:0;
      })];
    });
    return {L,gate,n:scene.T.length,m:scene.Z.length};
  }
  function targetMarginals(L,nu) {
    return L.map((row,i)=>{
      const r=[row[0],...row.slice(1).map((v,j)=>v*nu[j][i])];
      const z=sum(r); return r.map(v=>v/z);
    });
  }
  function measurementMarginals(mu,m=mu[0].length) {
    return Array.from({length:m},(_,j)=>{
      const r=[1,...mu.map(row=>row[j])],z=sum(r);
      return r.map(v=>v/z);
    });
  }
  // Prefix/suffix sums avoid cancellation in total-minus-own-message and stay O(nm).
  function excluding(values,baseline) {
    const out=Array(values.length),suffix=Array(values.length+1).fill(0);
    for(let j=values.length-1;j>=0;j--) suffix[j]=suffix[j+1]+values[j];
    let prefix=baseline;
    for(let j=0;j<values.length;j++){out[j]=prefix+suffix[j+1];prefix+=values[j];}
    return out;
  }
  function consistency(A,B) {
    let d=0;
    for(let i=0;i<A.length;i++) for(let j=0;j<B.length;j++) d=Math.max(d,Math.abs(A[i][j+1]-B[j][i+1]));
    return d;
  }
  function bp(L,{maxIterations=10000,tolerance=1e-10,history=true}={}) {
    const {n,m}=dimensions(L);
    if (!Number.isInteger(maxIterations) || maxIterations<1 || !(tolerance>0 && Number.isFinite(tolerance)))
      throw new RangeError('Use a positive integer iteration cap and a positive finite tolerance.');
    // Per-row scaling leaves the posterior and scalar BP messages unchanged.
    const W=L.map(row=>{const s=Math.max(...row);return row.map(x=>x/s);});
    if(W.some(row=>row[0]===0)) throw new RangeError('Weight dynamic range exceeds floating-point precision.');
    let nu=Array.from({length:m},()=>Array(n).fill(1));
    let mu=Array.from({length:n},()=>Array(m).fill(0));
    let marginal=targetMarginals(W,nu),bmarg=null,delta=Infinity,dualResidual=Infinity;
    const hist=history?[{kind:'init',t:0,mu:null,nu:cp(nu),marg:marginal,bmarg:null,delta:null}]:[];
    let converged=false,t=0;
    for(t=1;t<=maxIterations;t++) {
      mu=W.map((row,i)=>{
        const den=excluding(row.slice(1).map((v,j)=>v*nu[j][i]),row[0]);
        return row.slice(1).map((v,j)=>v/den[j]);
      });
      bmarg=measurementMarginals(mu,m);
      if(history) hist.push({kind:'mu',t,mu:cp(mu),nu:cp(nu),marg:marginal,bmarg,delta:null});
      delta=0;
      const next=Array.from({length:m},(_,j)=>{
        const den=excluding(mu.map(row=>row[j]),1);
        return den.map((v,i)=>{const q=1/v;delta=Math.max(delta,Math.abs(Math.log(q)-Math.log(nu[j][i])));return q;});
      });
      nu=next;marginal=targetMarginals(W,nu);dualResidual=consistency(marginal,bmarg);
      if(!Number.isFinite(delta+dualResidual)) throw new RangeError('Non-finite BP messages.');
      if(history) hist.push({kind:'nu',t,mu:cp(mu),nu:cp(nu),marg:marginal,bmarg,delta,dualResidual});
      // Numerical stopping diagnostic, not a bound on error relative to the exact posterior.
      if(delta<=tolerance && dualResidual<=tolerance){converged=true;break;}
    }
    return {marginals:marginal,measurementMarginals:bmarg,mu,nu,history:hist,
      iterations:Math.min(t,maxIterations),converged,delta,dualResidual,tolerance};
  }
  function enumerate(L,{maxEvents=200000}={}) {
    const {n,m}=dimensions(L),events=[],a=Array(n).fill(-1),used=new Set();
    function visit(i,logWeight) {
      if(i===n){
        if(events.length>=maxEvents) throw new RangeError('Exact enumeration cap reached; use a smaller benchmark.');
        events.push({a:a.slice(),logWeight});return;
      }
      a[i]=-1;visit(i+1,logWeight+Math.log(L[i][0]));
      for(let j=0;j<m;j++) if(!used.has(j) && L[i][j+1]>0){
        a[i]=j;used.add(j);visit(i+1,logWeight+Math.log(L[i][j+1]));used.delete(j);
      }
      a[i]=-1;
    }
    visit(0,0);
    const max=events.reduce((v,e)=>Math.max(v,e.logWeight),-Infinity);
    const z=sum(events.map(e=>Math.exp(e.logWeight-max)));
    events.forEach(e=>{e.p=Math.exp(e.logWeight-max)/z;});
    events.sort((a,b)=>b.p-a.p);
    return events;
  }
  function eventMarginals(events,n,m,k=events.length) {
    if(!Number.isInteger(k) || k<1 || k>events.length) throw new RangeError('Invalid retained-hypothesis count.');
    const chosen=events.slice(0,k),mass=sum(chosen.map(e=>e.p));
    const marginals=Array.from({length:n},()=>Array(m+1).fill(0));
    const measurement=Array.from({length:m},()=>Array(n+1).fill(0));
    for(const e of chosen){
      for(let i=0;i<n;i++) marginals[i][e.a[i]+1]+=e.p;
      for(let j=0;j<m;j++) measurement[j][e.a.indexOf(j)+1]+=e.p;
    }
    return {marginals:marginals.map(r=>r.map(x=>x/mass)),measurementMarginals:measurement.map(r=>r.map(x=>x/mass)),
      mass,discardedMass:Math.max(0,1-mass)};
  }
  function topology(L) {
    const {n,m}=dimensions(L),parent=Array.from({length:n+m},(_,i)=>i);
    const find=i=>parent[i]===i?i:(parent[i]=find(parent[i]));
    let edges=0,cycles=0;
    for(let i=0;i<n;i++)for(let j=0;j<m;j++)if(L[i][j+1]>0){
      edges++;const a=find(i),b=find(n+j);if(a===b)cycles++;else parent[a]=b;
    }
    return {edges,cycles,acyclic:cycles===0,components:new Set(parent.map((_,i)=>find(i))).size};
  }
  function maxDifference(A,B){let d=0;for(let i=0;i<A.length;i++)for(let j=0;j<A[i].length;j++)d=Math.max(d,Math.abs(A[i][j]-B[i][j]));return d;}
  // Conditional-parent PMBM association weights, before any gating approximation.
  // eta[i][j] = integral PD(x) l(z_j|x) p_i(x) dx; pDbar[i] = integral PD(x) p_i(x) dx.
  function pmbmWeights(r,pDbar,eta,clutter,evidence) {
    const n=r.length,m=clutter.length;
    if(!n || pDbar.length!==n || eta.length!==n || evidence.length!==m || eta.some(row=>row.length!==m))
      throw new RangeError('Inconsistent PMBM evidence dimensions.');
    if(r.concat(pDbar).some(x=>!Number.isFinite(x)||x<0||x>1) || clutter.concat(evidence,...eta).some(x=>!Number.isFinite(x)||x<0))
      throw new RangeError('Invalid PMBM probabilities or evidence.');
    const q=clutter.map((c,j)=>c+evidence[j]);
    if(q.some(x=>x<=0))throw new RangeError('The normalized formulation requires c(z)+e(z)>0.');
    const miss=r.map((v,i)=>1-v*pDbar[i]);
    const L=r.map((v,i)=>[miss[i],...eta[i].map((x,j)=>v*x/q[j])]);
    dimensions(L);
    return {L,q,newExistence:q.map((v,j)=>evidence[j]/v),
      missedExistence:r.map((v,i)=>v*(1-pDbar[i])/miss[i])};
  }
  return {GATE,DEFAULT,clone,gaussian2,buildWeights,targetMarginals,measurementMarginals,bp,enumerate,eventMarginals,topology,maxDifference,pmbmWeights};
});
