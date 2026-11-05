/* Sampling teaching models shared by the live lab, the deck build and the tests.
 * Every sampler is seeded, so a remount, the static fallback and the numbers on the slides agree.
 * Chain samplers target a standard bivariate Gaussian with correlation rho; the population
 * methods use a two-component 1-D mixture and a linear-Gaussian tracking model.
 */
(function(root){
'use strict';
const TAU=2*Math.PI;

// rng(seed): mulberry32 PRNG. Takes an integer seed, returns a function yielding floats in [0, 1).
function rng(seed){
 let t=seed>>>0;
 return function(){
  t+=0x6D2B79F5;
  let r=Math.imul(t^t>>>15,t|1);
  r^=r+Math.imul(r^r>>>7,r|61);
  return ((r^r>>>14)>>>0)/4294967296;
 };
}
// randn(r): one standard-normal draw by Box–Muller. Takes a PRNG, returns a float.
function randn(r){const a=Math.max(r(),1e-9),b=Math.max(r(),1e-9);return Math.sqrt(-2*Math.log(a))*Math.cos(TAU*b);}
function normalPdf(x,mu,sd){const z=(x-mu)/sd;return Math.exp(-.5*z*z)/(sd*Math.sqrt(TAU));}

/* ---------------- chain target: correlated bivariate Gaussian ---------------- */
const CHAIN={steps:120,shown:12,start:{x:-2.35,y:-2.05},rho:.88,
 seeds:{gibbs:431,mh:1428,hmc:2425,slice:3422},
 mh:{sigma:.72},hmc:{eps:.16,L:9},slice:{w:1.7,m:20}};

// logPi(p, rho): unnormalized log density. Takes {x, y} and rho, returns a float.
function logPi(p,rho){return -(p.x*p.x-2*rho*p.x*p.y+p.y*p.y)/(2*(1-rho*rho));}
// gradLogPi(p, rho): gradient of logPi. Takes {x, y} and rho, returns {x, y}.
function gradLogPi(p,rho){const n=1-rho*rho;return {x:-(p.x-rho*p.y)/n,y:-(p.y-rho*p.x)/n};}
// mahalanobis(p, rho): distance of p from the origin in units of the target's covariance.
function mahalanobis(p,rho){return Math.sqrt(-2*logPi(p,rho));}

// runChain(method, options): runs one chain sampler from the fixed start and records every move with
// the construction that produced it and its work (evals: conditional draws, densities or gradients). Options: rho, steps, seed, sigma (MH), eps and L (HMC),
// w and m (slice), start. Returns {points, moves, cost}; points[0] is the start.
function runChain(method,o={}){
 const rho=o.rho??CHAIN.rho,steps=o.steps??CHAIN.steps,seed=o.seed??CHAIN.seeds[method];
 const sigma=o.sigma??CHAIN.mh.sigma,eps=o.eps??CHAIN.hmc.eps,L=o.L??CHAIN.hmc.L,w=o.w??CHAIN.slice.w,m=o.m??CHAIN.slice.m;
 const r=rng(seed),condSd=Math.sqrt(1-rho*rho);
 let cur={...(o.start||CHAIN.start)},curLog=logPi(cur,rho),curGrad=gradLogPi(cur,rho);
 const points=[{...cur}],moves=[],cost={density:0,gradient:0,conditional:0};
 for(let n=0;n<steps;n++){
  let move;
  if(method==='gibbs'){
   // Exact conditional x_j | x_-j ~ N(rho·x_-j, 1 − rho²), alternating coordinates.
   const axis=n%2?'y':'x',mean=rho*(axis==='x'?cur.y:cur.x),v=mean+condSd*randn(r);
   const to=axis==='x'?{x:v,y:cur.y}:{x:cur.x,y:v};
   cost.conditional++;
   move={axis,mean,sd:condSd,from:cur,to,accepted:true,evals:1};
   cur=to;curLog=logPi(cur,rho);
  }else if(method==='mh'){
   // Symmetric Gaussian random walk; the Hastings ratio reduces to a density ratio.
   const prop={x:cur.x+sigma*randn(r),y:cur.y+sigma*randn(r)},lp=logPi(prop,rho);
   cost.density++;
   const logA=lp-curLog,accepted=Math.log(Math.max(r(),1e-9))<logA;
   move={from:cur,proposal:prop,alpha:Math.min(1,Math.exp(logA)),accepted,to:accepted?prop:cur,evals:1};
   if(accepted){cur=prop;curLog=lp;}
  }else if(method==='hmc'){
   // Leapfrog with unit mass: half momentum step, L position steps, final half step.
   let q={...cur},g=curGrad;
   const p0={x:randn(r),y:randn(r)},p={...p0},path=[{...q}];
   p.x+=eps/2*g.x;p.y+=eps/2*g.y;
   for(let l=0;l<L;l++){
    q={x:q.x+eps*p.x,y:q.y+eps*p.y};
    g=gradLogPi(q,rho);cost.gradient++;
    path.push({...q});
    if(l<L-1){p.x+=eps*g.x;p.y+=eps*g.y;}
   }
   p.x+=eps/2*g.x;p.y+=eps/2*g.y;
   const lq=logPi(q,rho);cost.density++;
   const h0=-curLog+(p0.x*p0.x+p0.y*p0.y)/2,h1=-lq+(p.x*p.x+p.y*p.y)/2;
   const dH=h1-h0,accepted=Number.isFinite(dH)&&Math.log(Math.max(r(),1e-9))<-dH;
   move={from:cur,proposal:q,path,dH,alpha:Number.isFinite(dH)?Math.min(1,Math.exp(-dH)):0,accepted,to:accepted?q:cur,evals:L};
   if(accepted){cur=q;curLog=lq;curGrad=g;}
  }else if(method==='slice'){
   // Neal (2003): stepping out with a total budget of m widths split at random, then shrinkage.
   const axis=n%2?'y':'x',x0=axis==='x'?cur.x:cur.y,before=cost.density;
   const at=v=>{cost.density++;return logPi(axis==='x'?{x:v,y:cur.y}:{x:cur.x,y:v},rho);};
   const level=curLog+Math.log(Math.max(r(),1e-9));
   let lo=x0-w*r(),hi=lo+w,J=Math.floor(m*r()),K=m-1-J;
   while(J>0&&at(lo)>level){lo-=w;J--;}
   while(K>0&&at(hi)>level){hi+=w;K--;}
   const bracket=[lo,hi],misses=[];
   let v=x0;
   for(let k=0;k<200;k++){
    const t=lo+r()*(hi-lo);
    if(at(t)>=level){v=t;break;}
    misses.push(t);
    if(t<x0)lo=t;else hi=t;
   }
   const to=axis==='x'?{x:v,y:cur.y}:{x:cur.x,y:v};
   move={axis,from:cur,to,level,bracket,final:[lo,hi],misses,accepted:true,evals:cost.density-before};
   cur=to;curLog=logPi(cur,rho);
  }else throw new Error('Unknown chain method '+method);
  moves.push(move);
  points.push({...cur});
 }
 return {method,rho,seed,points,moves,cost};
}

// essLag1(xs): the lab's quick mixing hint n(1−a)/(1+a) from the lag-1 autocorrelation a, clamped to ±0.95.
function essLag1(xs){
 if(xs.length<4)return xs.length;
 const mean=xs.reduce((s,v)=>s+v,0)/xs.length;
 let v=0,c=0;
 for(let i=0;i<xs.length;i++){v+=(xs[i]-mean)**2;if(i)c+=(xs[i]-mean)*(xs[i-1]-mean);}
 if(v<1e-12)return 1;
 const a=Math.max(-.95,Math.min(.95,c/v));
 return Math.max(1,Math.round(xs.length*(1-a)/(1+a)));
}
// essGeyer(xs): effective sample size with Geyer's initial monotone sequence estimator. Returns a float.
function essGeyer(xs){
 const n=xs.length,mean=xs.reduce((s,v)=>s+v,0)/n,d=xs.map(v=>v-mean);
 const gamma=k=>{let s=0;for(let i=0;i+k<n;i++)s+=d[i]*d[i+k];return s/n;};
 const g0=gamma(0);
 let sum=0,prev=Infinity;
 for(let k=0;2*k+1<n;k++){
  let G=gamma(2*k)+gamma(2*k+1);
  if(G<=0)break;
  G=Math.min(G,prev);prev=G;sum+=G;
 }
 const tau=Math.max(1e-9,(-g0+2*sum)/g0);
 return n/tau;
}

/* ---------------- population target: a two-bump mixture ---------------- */
const MIX=[{w:.55,mu:-1.35,sd:.6},{w:.45,mu:1.15,sd:.45}];
const POP={seed:1913,proposal:{m:0,s:1.5},xmin:-4.2,xmax:4.2};
function piX(x){return MIX.reduce((s,c)=>s+c.w*normalPdf(x,c.mu,c.sd),0);}
const TRUE_MEAN=MIX.reduce((s,c)=>s+c.w*c.mu,0);
function qX(x,m=POP.proposal.m,s=POP.proposal.s){return normalPdf(x,m,s);}

// supRatio(m, s): the tightest envelope constant sup π/q, found on a fine grid and refined by
// golden-section search. Infinite when q's tails are lighter than a mixture component's.
function supRatio(m=POP.proposal.m,s=POP.proposal.s){
 if(s<=Math.max(...MIX.map(c=>c.sd)))return Infinity;
 const f=x=>piX(x)/qX(x,m,s);
 let best=-1,at=0;
 for(let x=-12;x<=12;x+=.002){const v=f(x);if(v>best){best=v;at=x;}}
 let a=at-.004,b=at+.004;const phi=(Math.sqrt(5)-1)/2;
 for(let i=0;i<60;i++){const c=b-phi*(b-a),d=a+phi*(b-a);if(f(c)>f(d))b=d;else a=c;}
 return Math.max(best,f((a+b)/2));
}
// weightSecondMoment(m, s): E_q[w²] = ∫π²/q by quadrature; Infinity when the integral diverges.
function weightSecondMoment(m=POP.proposal.m,s=POP.proposal.s){
 if(s<=Math.max(...MIX.map(c=>c.sd))/Math.SQRT2)return Infinity;
 // Log-space integrand 2·log π − log q avoids 0·∞ in the far tails.
 const logPiX=x=>{const t=MIX.map(c=>Math.log(c.w)-.5*((x-c.mu)/c.sd)**2-Math.log(c.sd*Math.sqrt(TAU)));const top=Math.max(...t);return top+Math.log(t.reduce((a,v)=>a+Math.exp(v-top),0));};
 const logQ=x=>-.5*((x-m)/s)**2-Math.log(s*Math.sqrt(TAU));
 let sum=0;const h=.001;
 for(let x=-60;x<=60;x+=h)sum+=Math.exp(2*logPiX(x)-logQ(x))*h;
 return sum;
}

// makeRejection(options): rejection sampler with envelope M = slack · sup π/q. Options: m, s, slack, seed.
// Returns an object whose draw(k) adds k proposals; slack < 1 deliberately breaks the envelope.
function makeRejection(o={}){
 const m=o.m??POP.proposal.m,s=o.s??POP.proposal.s,slack=o.slack??1,seed=o.seed??POP.seed;
 const r=rng(seed),sup=supRatio(m,s),M=slack*sup,draws=[];
 return {m,s,slack,sup,M,draws,
  draw(k){for(let i=0;i<k;i++){const x=m+s*randn(r),u=r()*M*qX(x,m,s);draws.push({x,u,ok:u<=piX(x)});}return this;},
  stats(){const kept=draws.filter(d=>d.ok).length;return {n:draws.length,kept,acceptance:draws.length?kept/draws.length:NaN,predicted:1/M};}};
}

// makeImportance(options): self-normalized importance sampler. Options: m, s, seed.
function makeImportance(o={}){
 const m=o.m??POP.proposal.m,s=o.s??POP.proposal.s,seed=o.seed??POP.seed;
 const r=rng(seed),draws=[];
 return {m,s,draws,
  draw(k){for(let i=0;i<k;i++){const x=m+s*randn(r);draws.push({x,w:piX(x)/qX(x,m,s)});}return this;},
  stats(){
   if(!draws.length)return {n:0};
   const W=draws.reduce((a,d)=>a+d.w,0),W2=draws.reduce((a,d)=>a+d.w*d.w,0);
   const est=draws.reduce((a,d)=>a+d.w*d.x,0)/W;
   // Delta-method standard error of the self-normalized estimate.
   const se=Math.sqrt(draws.reduce((a,d)=>a+(d.w/W)**2*(d.x-est)**2,0));
   const wmax=draws.reduce((a,d)=>Math.max(a,d.w),0);
   return {n:draws.length,ess:W*W/W2,est,se,maxShare:wmax/W,meanWeight:W/draws.length};
  }};
}

/* ---------------- sequential Monte Carlo: bootstrap particle filter ---------------- */
const SSM={a:.92,q:.45,r:.5,m0:0,s0:.8,T:60,worldSeed:2718,filterSeed:1913,N:44};

// makeWorld(options): simulates the hidden state and its observations once, independent of the filter.
function makeWorld(o={}){
 const r=rng(o.seed??SSM.worldSeed),T=o.T??SSM.T,truth=[],obs=[];
 let x=SSM.m0+SSM.s0*randn(r);
 for(let t=1;t<=T;t++){x=SSM.a*x+SSM.q*randn(r);truth.push(x);obs.push(x+SSM.r*randn(r));}
 return {truth,obs,T};
}
// kalman(obs): the exact posterior means and standard deviations for this linear-Gaussian model.
function kalman(obs){
 let mean=SSM.m0,P=SSM.s0**2;const out=[];
 for(const y of obs){
  mean=SSM.a*mean;P=SSM.a*SSM.a*P+SSM.q**2;
  const K=P/(P+SSM.r**2);mean+=K*(y-mean);P*=1-K;
  out.push({mean,sd:Math.sqrt(P)});
 }
 return out;
}
// kalmanSteadySd(): stationary posterior standard deviation of the exact filter.
function kalmanSteadySd(){let P=1;for(let i=0;i<500;i++){const Pm=SSM.a*SSM.a*P+SSM.q**2;P=Pm*SSM.r**2/(Pm+SSM.r**2);}return Math.sqrt(P);}

// makeFilter(options): bootstrap particle filter with systematic resampling.
// Options: N, rule ('half' | 'always' | 'never'), seed, world. step() advances one observation.
function makeFilter(o={}){
 const N=o.N??SSM.N,rule=o.rule??'half',world=o.world??makeWorld(),r=rng(o.seed??SSM.filterSeed);
 const exact=kalman(world.obs);
 let xs=Array.from({length:N},()=>SSM.m0+SSM.s0*randn(r)),logw=new Array(N).fill(0);
 const history=[];
 const f={N,rule,world,exact,history,get t(){return history.length;},
  step(){
   const t=history.length;if(t>=world.T)return false;
   const y=world.obs[t];
   xs=xs.map(x=>SSM.a*x+SSM.q*randn(r));
   logw=logw.map((l,i)=>l-.5*((y-xs[i])/SSM.r)**2);
   const top=Math.max(...logw);let ws=logw.map(l=>Math.exp(l-top));const sum=ws.reduce((a,b)=>a+b,0);ws=ws.map(v=>v/sum);
   const ess=1/ws.reduce((a,v)=>a+v*v,0),mean=ws.reduce((a,v,i)=>a+v*xs[i],0);
   const snapshot=xs.map((x,i)=>({x,w:ws[i]}));
   const resample=rule==='always'||(rule==='half'&&ess<N/2);
   let distinct=N;
   if(resample){
    const u0=r()/N,picked=[],seen=new Set();let c=ws[0],i=0;
    for(let k=0;k<N;k++){const u=u0+k/N;while(u>c&&i<N-1){i++;c+=ws[i];}picked.push(xs[i]);seen.add(i);}
    xs=picked;logw=new Array(N).fill(0);distinct=seen.size;
   }else logw=ws.map(v=>Math.log(Math.max(v,1e-300)));
   history.push({t:t+1,y,truth:world.truth[t],mean,ess,resampled:resample,distinct,particles:snapshot,exact:exact[t].mean,exactSd:exact[t].sd});
   return true;
  },
  run(){while(this.step());return this;},
  stats(){
   const h=history;if(!h.length)return {t:0};
   const rms=a=>Math.sqrt(a.reduce((s,v)=>s+v*v,0)/a.length);
   const res=h.filter(e=>e.resampled);
   return {t:h.length,ess:h[h.length-1].ess,resamples:res.length,
    distinct:res.length?res.reduce((s,e)=>s+e.distinct,0)/res.length/N:NaN,
    rmse:rms(h.map(e=>e.mean-e.truth)),exactRmse:rms(h.map(e=>e.exact-e.truth)),mcError:rms(h.map(e=>e.mean-e.exact))};
  }};
 return f;
}

const api={TAU,rng,randn,normalPdf,CHAIN,logPi,gradLogPi,mahalanobis,runChain,essLag1,essGeyer,
 MIX,POP,piX,TRUE_MEAN,qX,supRatio,weightSecondMoment,makeRejection,makeImportance,
 SSM,makeWorld,kalman,kalmanSteadySd,makeFilter};
if(typeof module!=='undefined')module.exports=api;
root.SamplingModel=api;
})(typeof window!=='undefined'?window:globalThis);
