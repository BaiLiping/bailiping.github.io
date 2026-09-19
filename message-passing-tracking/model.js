(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.TrackingBP=api;})(typeof globalThis==='object'?globalThis:this,function(){
  'use strict';
  // A finite-state teaching instance of Fig. 4 / Sections VI and IX-A.
  // The absent state aggregates the paper's normalized dummy spatial density.
  const defaults=Object.freeze({z2:.62,pD:.85,birth:.3,iterations:3});
  const priors=[[.1,.8,.1],[.2,.15,.65]],survival=.95,sigma=.35,clutter=.5;
  const transition=[[1,0,0],[1-survival,.85*survival,.15*survival],[1-survival,.15*survival,.85*survival]];
  const sum=v=>v.reduce((a,b)=>a+b,0),normalize=v=>{const z=sum(v);if(!(z>0))throw new Error('Zero message mass');return v.map(x=>x/z);};
  const matrix=(n,m,fn)=>Array.from({length:n},(_,j)=>Array.from({length:m},(_,k)=>fn(j,k)));
  const likelihood=(z,x)=>Math.exp(-.5*((z-x)/sigma)**2)/(Math.sqrt(2*Math.PI)*sigma);
  const psi=(j,m,a,b)=>(a===m+1)===(b===j+1)?1:0;
  const product=v=>v.reduce((a,b)=>a*b,1);
  function cavityA(beta,nu,j,m){return beta[j].map((w,a)=>w*product(nu[j].filter((_,n)=>n!==m).map(v=>v[a])));}
  function cavityB(xi,phi,j,m){return xi[m].map((w,b)=>w*product(phi.filter((_,i)=>i!==j).map(row=>row[m][b])));}
  function infer(beta,xi,iterations){
    const J=beta.length,M=xi.length;
    // Dimension is passed explicitly to the local update; no hidden run state.
    const message=(c,j,m,toB)=>{
      const raw=Array.from({length:toB?J+1:M+1},(_,v)=>sum(c.map((w,s)=>w*(toB?psi(j,m,s,v):psi(j,m,v,s)))));
      return {raw,values:raw.map(w=>w/raw[0]),scale:raw[0]};
    };
    const initial=matrix(J,M,(j,m)=>message(beta[j],j,m,true));
    let phi=initial.map(row=>row.map(v=>v.values)),nu;
    const rounds=[];
    for(let iteration=1;iteration<=iterations;iteration++){
      const bCavity=matrix(J,M,(j,m)=>cavityB(xi,phi,j,m));
      const nuDetail=matrix(J,M,(j,m)=>message(bCavity[j][m],j,m,false));
      nu=nuDetail.map(row=>row.map(v=>v.values));
      const aCavity=matrix(J,M,(j,m)=>cavityA(beta,nu,j,m));
      const phiDetail=matrix(J,M,(j,m)=>message(aCavity[j][m],j,m,true));
      const nextPhi=phiDetail.map(row=>row.map(v=>v.values));
      const delta=Math.max(...nextPhi.flatMap((row,j)=>row.flatMap((v,m)=>v.map((x,b)=>Math.abs(x-phi[j][m][b])))));
      rounds.push({iteration,previousPhi:phi,bCavity,nuDetail,nu,aCavity,phiDetail,phi:nextPhi,delta});phi=nextPhi;
    }
    const kappa=beta.map((v,j)=>v.map((_,a)=>product(nu[j].map(msg=>msg[a]))));
    const iota=xi.map((v,m)=>v.map((_,b)=>product(phi.map(row=>row[m][b]))));
    const aBeliefs=beta.map((v,j)=>normalize(v.map((x,a)=>x*kappa[j][a])));
    const bBeliefs=xi.map((v,m)=>normalize(v.map((x,b)=>x*iota[m][b])));
    return {initial,rounds,phi,nu,kappa,iota,aBeliefs,bBeliefs};
  }
  function exactAssociations(beta,xi){
    const J=beta.length,M=xi.length,rows=[];
    function visit(a){
      if(a.length<J){for(let m=0;m<=M;m++){if(m===0||!a.includes(m))visit([...a,m]);}return;}
      const b=Array.from({length:M},(_,m)=>{const j=a.indexOf(m+1);return j<0?0:j+1;});
      rows.push({a,b,weight:product(a.map((m,j)=>beta[j][m]))*product(b.map((j,m)=>xi[m][j]))});
    }
    visit([]);const z=sum(rows.map(r=>r.weight));rows.forEach(r=>r.probability=r.weight/z);
    return {rows,z,aBeliefs:beta.map((v,j)=>v.map((_,a)=>sum(rows.filter(r=>r.a[j]===a).map(r=>r.probability)))),bBeliefs:xi.map((v,m)=>v.map((_,b)=>sum(rows.filter(r=>r.b[m]===b).map(r=>r.probability))))};
  }
  function run(input={}){
    const p={...defaults,...input};
    if(!Number.isFinite(p.z2)||p.z2<-.2||p.z2>1.2||!Number.isFinite(p.pD)||p.pD<0||p.pD>1||!Number.isFinite(p.birth)||p.birth<0||p.birth>2||!Number.isInteger(p.iterations)||p.iterations<1||p.iterations>50)throw new RangeError('Invalid teaching parameters');
    const z=[.3,p.z2],L=z.map(v=>[likelihood(v,0),likelihood(v,1)]);
    const alpha=priors.map(prior=>[0,1,2].map(y=>sum(prior.map((w,old)=>w*transition[old][y]))));
    const q=alpha.map(()=>[0,1,2].map(y=>[0,1,2].map(a=>y===0?Number(a===0):a===0?1-p.pD:p.pD*L[a-1][y-1]/clutter)));
    const v=z.map((_,m)=>[0,1,2].map(y=>[0,1,2].map(b=>y===0?1:b===0?p.birth*.5*L[m][y-1]/clutter:0)));
    const beta=q.map((factor,j)=>[0,1,2].map(a=>sum(factor.map((row,y)=>alpha[j][y]*row[a]))));
    const xi=v.map(factor=>[0,1,2].map(b=>sum(factor.map(row=>row[b]))));
    const da=infer(beta,xi,p.iterations),exact=exactAssociations(beta,xi);
    const gamma=q.map((factor,j)=>factor.map(row=>sum(row.map((w,a)=>w*da.kappa[j][a]))));
    const zeta=v.map((factor,m)=>factor.map(row=>sum(row.map((w,b)=>w*da.iota[m][b]))));
    const legacy=alpha.map((row,j)=>normalize(row.map((w,y)=>w*gamma[j][y]))),newTargets=zeta.map(normalize);
    exact.legacy=alpha.map((row,j)=>row.map((w,y)=>sum(beta[j].map((weight,a)=>exact.aBeliefs[j][a]*(weight? w*q[j][y][a]/weight:0)))));
    exact.newTargets=v.map((factor,m)=>factor.map(row=>sum(row.map((w,b)=>exact.bBeliefs[m][b]*w/xi[m][b]))));
    const maxError=Math.max(...legacy.flatMap((row,j)=>row.map((w,y)=>Math.abs(w-exact.legacy[j][y]))),...newTargets.flatMap((row,m)=>row.map((w,y)=>Math.abs(w-exact.newTargets[m][y]))));
    return {p,z,L,alpha,q,v,beta,xi,da,gamma,zeta,legacy,newTargets,exact,maxError};
  }
  function steps(iterations=3){
    return [
      {id:'prior',group:0,title:'Send the previous posterior',symbol:String.raw`\widetilde f_-^j`},
      {id:'predict',group:0,title:'Predict motion and survival',symbol:String.raw`\alpha_j`},
      {id:'copy',group:0,title:'Pass the prediction into the likelihood',symbol:String.raw`\alpha_j`},
      {id:'beta',group:1,title:'Evaluate the legacy association weights',symbol:String.raw`\beta_j`},
      {id:'xi',group:1,title:'Evaluate the new-target branch',symbol:String.raw`\xi_m`},
      {id:'initial',group:2,title:'Initialize target-to-measurement messages',symbol:String.raw`\varphi^{[0]}_{j,m}`},
      ...Array.from({length:iterations},(_,i)=>[{id:'nu',group:2,iteration:i+1,title:`Iteration ${i+1}: measurement to target`,symbol:String.raw`\nu^{[${i+1}]}_{m,j}`},{id:'phi',group:2,iteration:i+1,title:`Iteration ${i+1}: target to measurement`,symbol:String.raw`\varphi^{[${i+1}]}_{j,m}`}]).flat(),
      {id:'kappa',group:3,title:'Return the association evidence to each legacy factor',symbol:String.raw`\kappa_j`},
      {id:'iota',group:3,title:'Return the association evidence to each new-target factor',symbol:String.raw`\iota_m`},
      {id:'gamma',group:3,title:'Update the legacy state',symbol:String.raw`\gamma_j`},
      {id:'zeta',group:3,title:'Update the possible new target',symbol:String.raw`\varsigma_m`},
      {id:'belief',group:4,title:'Normalize the state and existence beliefs',symbol:String.raw`\widetilde f`}
    ];
  }
  return {defaults,priors,survival,sigma,clutter,transition,sum,normalize,product,likelihood,psi,cavityA,cavityB,infer,exactAssociations,run,steps};
});
