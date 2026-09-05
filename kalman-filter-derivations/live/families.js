/* Deterministic teaching models shared by browser experiments and Node tests.
 * No random sampling, network calls, or animation timers.
 */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else{root.KalmanFamilies=api;api.mount();}})(typeof globalThis!=='undefined'?globalThis:this,function(){
'use strict';
const positive=(x,name)=>{if(!(Number.isFinite(x)&&x>0))throw new RangeError(name+' must be positive');return x;};
const normal=(x,m,v)=>Math.exp(-.5*(x-m)**2/positive(v,'variance'))/Math.sqrt(2*Math.PI*v);
function scalar(m,P,z,R){positive(P,'P');positive(R,'R');const S=P+R,K=P/S;return{mean:m+K*(z-m),variance:P*R/S,K,logEvidence:-.5*(Math.log(2*Math.PI*S)+(z-m)**2/S)};}
function bayes({z=0,R=1,separation=2,shape='mixture'}={}){
 const P=separation**2+.36,prior=shape==='gaussian'?[{m:0,v:P,w:1}]:[{m:-separation,v:.36,w:.5},{m:separation,v:.36,w:.5}];
 const parts=prior.map(p=>{const post=scalar(p.m,p.v,z,R);return{...post,logWeight:Math.log(p.w)+post.logEvidence};});
 const max=Math.max(...parts.map(p=>p.logWeight)),total=parts.reduce((s,p)=>s+Math.exp(p.logWeight-max),0),posterior=parts.map(p=>({...p,w:Math.exp(p.logWeight-max)/total}));
 const mean=posterior.reduce((s,p)=>s+p.w*p.mean,0),variance=posterior.reduce((s,p)=>s+p.w*(p.variance+(p.mean-mean)**2),0);
 return{prior,posterior,mean,variance,logEvidence:max+Math.log(total),affine:scalar(0,P,z,R),P,priorDensity:x=>prior.reduce((s,p)=>s+p.w*normal(x,p.m,p.v),0),density:x=>posterior.reduce((s,p)=>s+p.w*normal(x,p.mean,p.variance),0)};
}
function mse({K=.3,R=1,P=4}={}){positive(P,'P');positive(R,'R');const optimum=P/(P+R),risk=k=>(1-k)**2*P+k**2*R;return{K,R,P,optimum,risk,value:risk(K),minimum:risk(optimum),cross:P-K*(P+R),excess:(P+R)*(K-optimum)**2};}
function wls({x=0,alpha=1,R=.5625,m=-1.2,P=1.8225,z=2.1}={}){positive(alpha,'alpha');positive(P,'P');positive(R,'R');const J=alpha/P+1/R,h=alpha*m/P+z/R,optimum=h/J,priorCost=t=>.5*alpha*(t-m)**2/P,dataCost=t=>.5*(z-t)**2/R,cost=t=>priorCost(t)+dataCost(t);return{x,alpha,J,h,optimum,variance:1/J,gradient:J*x-h,value:cost(x),priorCost,dataCost,cost,newton:x-(J*x-h)/J};}
function variational({mu=0,sigma=1.6,m=-1.2,P=1.8225,z=2.1,R=.5625}={}){positive(sigma,'sigma');const post=scalar(m,P,z,R),v=sigma**2,priorKL=.5*((v+(mu-m)**2)/P-1+Math.log(P/v)),expectedNLL=.5*(Math.log(2*Math.PI*R)+((z-mu)**2+v)/R),meanGap=.5*(mu-post.mean)**2/post.variance,varianceGap=.5*(v/post.variance-1+Math.log(post.variance/v));return{mu,sigma,post,priorKL,expectedNLL,freeEnergy:priorKL+expectedNLL,gap:meanGap+varianceGap,meanGap,varianceGap};}
function solve(A,b){const n=b.length,W=A.map((r,i)=>[...r,b[i]]);for(let k=0;k<n;k++){let pivot=k;for(let i=k+1;i<n;i++)if(Math.abs(W[i][k])>Math.abs(W[pivot][k]))pivot=i;if(Math.abs(W[pivot][k])<1e-14)throw new Error('Singular teaching system');[W[k],W[pivot]]=[W[pivot],W[k]];const d=W[k][k];for(let j=k;j<=n;j++)W[k][j]/=d;for(let i=0;i<n;i++)if(i!==k){const f=W[i][k];for(let j=k;j<=n;j++)W[i][j]-=f*W[k][j];}}return W.map(r=>r[n]);}
function chain({last=2.5,Q=.6,R=.5,eliminated=0}={}){
 positive(Q,'Q');positive(R,'R');const z=[1,.2,1.1,last],n=z.length,J=Array.from({length:n},()=>Array(n).fill(0)),h=z.map(v=>v/R);J[0][0]=1;
 for(let i=0;i<n;i++)J[i][i]+=1/R;
 for(let i=1;i<n;i++){J[i][i]+=1/Q;J[i-1][i-1]+=1/Q;J[i][i-1]=J[i-1][i]=-1/Q;}
 const batch=solve(J,h),covariance=Array.from({length:n},(_,j)=>solve(J,Array.from({length:n},(_,i)=>+(i===j))));
 const filtered=[],variances=[],predicted=[],predVars=[];let m=0,P=1;
 for(let i=0;i<n;i++){if(i)P+=Q;predicted.push(m);predVars.push(P);const p=scalar(m,P,z[i],R);m=p.mean;P=p.variance;filtered.push(m);variances.push(P);}
 const smoothed=filtered.slice(),smoothVars=variances.slice();
 for(let i=n-2;i>=0;i--){const G=variances[i]/predVars[i+1];smoothed[i]+=G*(smoothed[i+1]-predicted[i+1]);smoothVars[i]+=G*G*(smoothVars[i+1]-predVars[i+1]);}
 const d=[],g=[];for(let i=0;i<n;i++){d[i]=J[i][i]-(i?J[i][i-1]**2/d[i-1]:0);g[i]=h[i]-(i?J[i][i-1]*g[i-1]/d[i-1]:0);}
 const k=Math.min(3,Math.max(0,Math.floor(eliminated))),active=J.slice(k).map(r=>r.slice(k)),info=h.slice(k);active[0][0]=d[k];info[0]=g[k];
 return{z,J,h,batch,covariance,filtered,variances,smoothed,smoothVars,d,g,active,info,eliminated:k,lastMean:g[n-1]/d[n-1],lastVariance:1/d[n-1]};
}
const COLORS={prior:'#496E87',exact:'#2F6B4F',other:'#A94F2A',light:'#D8DED7',ink:'#203129'};
const fmt=(v,n=3)=>Number.isFinite(v)?Math.abs(v)>1e4?v.toExponential(2):v.toFixed(n):'—';
const lin=(a,b,n=180)=>Array.from({length:n},(_,i)=>a+(b-a)*i/(n-1));
const txt=(x,y,s,extra='')=>`<text x="${x}" y="${y}" font-size="11" fill="${COLORS.ink}" ${extra}>${s}</text>`;
function plot(series,xrange,yrange,markers=[],labels=['state x','value']){
 const[a,b]=xrange,[c,d]=yrange,X=x=>52+640*(x-a)/(b-a),Y=y=>212-181*(y-c)/(d-c);
 let svg=`<svg class="plot" viewBox="0 0 730 255" role="img" aria-label="${labels.join(' versus ')}"><defs><clipPath id="curve-clip"><rect x="52" y="31" width="640" height="181"/></clipPath></defs>`;
 lin(a,b,6).forEach(x=>{svg+=`<path d="M${X(x)} 31V212" stroke="${COLORS.light}"/>`+txt(X(x),231,fmt(x,1),'text-anchor="middle"');});
 lin(c,d,4).forEach(y=>{svg+=`<path d="M52 ${Y(y)}H692" stroke="${COLORS.light}"/>`+txt(44,Y(y)+4,fmt(y,1),'text-anchor="end"');});
 svg+=txt(55,17,labels[1])+txt(690,251,labels[0],'text-anchor="end"')+'<g clip-path="url(#curve-clip)">';
 for(const s of series){const pts=s.points||lin(a,b).map(x=>[x,s.f(x)]);svg+=`<path d="${pts.map(([x,y],i)=>`${i?'L':'M'}${X(x).toFixed(2)},${Y(y).toFixed(2)}`).join(' ')}" fill="none" stroke="${s.color}" stroke-width="2.6" ${s.dash?'stroke-dasharray="6 4"':''}/>`;}
 for(const m of markers){if(m.line)svg+=`<path d="M${X(m.x)} 31V212" stroke="${m.color}" stroke-dasharray="4 4"/>`;else svg+=`<circle cx="${X(m.x)}" cy="${Y(m.y)}" r="5" fill="${m.color}" stroke="white"/>`;}
 return svg+'</g></svg>';
}
function mount(){
 if(typeof document==='undefined')return;
 const params=new URLSearchParams(location.search),mode=params.get('demo')||'bayes';if(params.has('embed'))document.body.classList.add('embed');
 const app=document.getElementById('app');
 const settings={bayes:{z:0,R:1,separation:2,shape:'mixture'},mse:{K:.3,R:1,shape:'gaussian'},wls:{x:0,alpha:1,R:.5625},kl:{mu:0,sigma:1.6},graphs:{last:2.5,Q:.6,eliminated:0}};
 const titles={bayes:['Probability','A posterior is more than its mean.','Switch shapes while retaining the same prior mean and variance.'],mse:['Projection','Choose the gain; remove error correlation.','The risk uses exact second moments, not a sample estimate.'],wls:['Quadratic objectives','Find the bottom of the cost.','One Newton step solves this scalar quadratic exactly.'],kl:['Variational updating','Correct mean ≠ correct belief.','Move location and spread independently. The gap is an exact Gaussian KL.'],graphs:['Recursive inference','Eliminate the past, retain a message.','All estimates use the same four-state random-walk model.']};
 if(!settings[mode]){app.innerHTML='<p class="error">Unknown experiment. Use bayes, mse, wls, kl, or graphs.</p>';return;}
 const state={...settings[mode]};let result;
 const range=(key,label,min,max,step)=>`<div class="ctrl"><label for="${key}">${label}<output id="out-${key}"></output></label><input id="${key}" type="range" min="${min}" max="${max}" step="${step}" value="${state[key]}"></div>`;
 const shape=text=>`<div class="ctrl"><label for="shape">${text}</label><select id="shape"><option value="${mode==='bayes'?'mixture':'gaussian'}">${mode==='bayes'?'Two Gaussian components':'Gaussian variables'}</option><option value="${mode==='bayes'?'gaussian':'two-point'}">${mode==='bayes'?'Moment-matched Gaussian':'Symmetric two-point variables'}</option></select></div>`;
 const controls={bayes:shape('Prior shape')+range('z','Observation z',-4,4,.05)+range('R','Noise variance R',.1,4,.05)+range('separation','Mode separation a',0,3,.05),mse:shape('Same second moments')+range('K','Candidate gain K',-.25,1.5,.01)+range('R','Noise variance R',.1,4,.05)+'<button class="primary" data-action="optimum">Set optimal gain</button>',wls:range('x','Candidate state x',-4,4,.01)+range('alpha','Prior multiplier α',.2,4,.05)+range('R','Noise variance R',.1,3,.05)+'<button class="primary" data-action="newton">One Newton step</button>',kl:range('mu','Candidate mean μ',-4,4,.01)+range('sigma','Candidate standard deviation σ',.15,3.5,.01)+'<div class="buttons"><button data-action="mean">Match mean only</button><button class="primary" data-action="posterior">Match posterior</button></div>',graphs:range('last','Final measurement z3',-2,4,.05)+range('Q','Process variance Q',.1,2,.05)+'<div class="buttons"><button class="primary" data-action="eliminate">Eliminate next state</button><button data-action="all">Eliminate to last</button></div>'};
 app.innerHTML=`<aside><p class="eyebrow">GROUP EXPERIMENT · ${titles[mode][0]}</p><h1>${titles[mode][1]}</h1><p class="intro">${titles[mode][2]}</p>${controls[mode]}<div class="buttons"><button data-action="reset">Reset</button></div><p class="note" id="side-note"></p><footer><button data-nav="-1">← Summary</button><button data-nav="1">Equations →</button></footer></aside><section class="stage ${mode==='graphs'?'graph-stage':''}"><div class="toolbar"><h2 id="stage-title"></h2><span class="eyebrow">COMPUTED LIVE</span></div><div class="legend" id="legend"></div><div id="chart"></div><div class="metrics" id="metrics" aria-live="polite"></div><div id="extra"></div><div class="explain" id="explain"></div></section>`;
 const $=id=>document.getElementById(id),metrics=items=>items.map(([l,v])=>`<div class="metric"><small>${l}</small><strong>${v}</strong></div>`).join(''),legends=items=>items.map(([s,c])=>`<span style="--mark:${c}">${s}</span>`).join('');
 const sync=()=>{for(const[key,v]of Object.entries(state)){const el=$(key);if(el)el.value=v;const out=$('out-'+key);if(out)out.value=fmt(v,2);}};
 function draw(){
  sync();$('extra').innerHTML='';
  if(mode==='bayes'){
   result=bayes(state);const r=result;$('stage-title').textContent='Exact density versus a Kalman Gaussian';$('legend').innerHTML=legends([['Prior',COLORS.prior],['Exact posterior',COLORS.exact],['Kalman Gaussian',COLORS.other]]);
   const fs=[r.priorDensity,r.density,x=>normal(x,r.affine.mean,r.affine.variance)],ymax=Math.max(...lin(-7,7).flatMap(x=>fs.map(f=>f(x))))*1.15;
   $('chart').innerHTML=plot(fs.map((f,i)=>({f,color:[COLORS.prior,COLORS.exact,COLORS.other][i],dash:i===2})),[-7,7],[0,ymax],[{line:true,x:state.z,color:COLORS.ink}],['state x','density; dashed vertical = observation']);
   $('metrics').innerHTML=metrics([['Exact posterior mean',fmt(r.mean)],['Exact posterior variance',fmt(r.variance)],['Kalman mean / variance',`${fmt(r.affine.mean,2)} / ${fmt(r.affine.variance,2)}`]]);
   $('side-note').textContent=`Both shapes have prior mean 0 and variance ${fmt(r.P)}. Mixture centers: ±a; component variance: 0.36. All statistics are analytic.`;
   $('explain').textContent=state.shape==='gaussian'?'The prior is Gaussian: the exact posterior and Kalman Gaussian coincide.':'Exact Bayes updates and reweights both components. At z = 0 symmetry makes the two means agree, but the densities can still differ. Move z away from zero to expose the difference between the affine estimate and the conditional mean.';
  }else if(mode==='mse'){
   result=mse(state);const r=result;$('stage-title').textContent='Gain risk and orthogonality';$('legend').innerHTML=legends([['Exact expected risk',COLORS.prior],['Optimal gain',COLORS.exact],['Candidate gain',COLORS.other]]);
   $('chart').innerHTML=plot([{f:r.risk,color:COLORS.prior}],[-.25,1.5],[0,Math.max(r.risk(-.25),r.risk(1.5))*1.08],[{x:r.optimum,y:r.minimum,color:COLORS.exact},{x:state.K,y:r.value,color:COLORS.other}],['gain K','E[(e− − Kν)²]']);
   $('metrics').innerHTML=metrics([['Expected squared error',fmt(r.value)],['Error–innovation covariance',fmt(r.cross)],['Excess over minimum',fmt(r.excess)]]);
   $('side-note').textContent=`P = 4; R = ${fmt(state.R)}. K* = ${fmt(r.optimum)}. Risks and covariances are exact, not Monte Carlo estimates.`;
   if(state.shape==='two-point'){const cases=[-2,2].flatMap(e=>[-Math.sqrt(state.R),Math.sqrt(state.R)].map(v=>[e,v,(1-state.K)*e-state.K*v]));$('extra').innerHTML='<div class="note">Four equally probable (x, v, error) outcomes: '+cases.map(row=>'('+row.map(x=>fmt(x,2)).join(', ')+')').join(' · ')+'</div>';}
   $('explain').textContent='The optimum is exactly where the error becomes uncorrelated with the innovation. Switching Gaussian variables to two-point variables leaves this risk curve unchanged: only P and R enter. This proves affine optimality, not an unrestricted Bayesian posterior.';
  }else if(mode==='wls'){
   result=wls(state);const r=result;$('stage-title').textContent='Prior penalty + data penalty = total cost';$('legend').innerHTML=legends([['Prior penalty',COLORS.prior],['Measurement penalty',COLORS.other],['Total',COLORS.exact]]);
   $('chart').innerHTML=plot([{f:r.priorCost,color:COLORS.prior},{f:r.dataCost,color:COLORS.other},{f:r.cost,color:COLORS.exact}],[-4,4],[0,Math.max(r.cost(-4),r.cost(4))*1.04],[{x:state.x,y:r.value,color:COLORS.other},{x:r.optimum,y:r.cost(r.optimum),color:COLORS.exact}],['candidate state x','half weighted squared residual']);
   $('metrics').innerHTML=metrics([['Gradient Jx − h',fmt(r.gradient)],['Optimal state',fmt(r.optimum)],['Curvature J',fmt(r.J)]]);
   $('side-note').textContent='m− = −1.2; P = 1.8225; z = 2.1. Inverse curvature is the Gaussian posterior variance for this model. The candidate x does not change that curvature.';
   $('explain').textContent=`x ← x − (Jx − h)/J reaches the minimizer in one step. At α = 1 the weights match the shared Kalman example. Changing α to ${fmt(state.alpha,2)} changes the effective prior variance to P/α: a statistical-model change, not a new algebraic implementation.`;
  }else if(mode==='kl'){
   result=variational(state);const r=result,lo=Math.log(.15),hi=Math.log(3.5);let s='<svg id="heatmap" class="plot" viewBox="0 0 730 255" role="img" aria-label="KL gap over candidate mean and standard deviation"><title>Click to choose candidate mean and spread</title>';
   for(let j=0;j<36;j++)for(let i=0;i<90;i++){const gap=variational({mu:-4+(i+.5)*8/90,sigma:Math.exp(hi-(j+.5)*(hi-lo)/36)}).gap,t=Math.min(1,Math.log1p(gap)/Math.log(35)),rr=Math.round(231-143*t),gg=Math.round(240-155*t),bb=Math.round(234-129*t);s+=`<rect x="${52+i*640/90}" y="${31+j*181/36}" width="7.2" height="5.2" fill="rgb(${rr},${gg},${bb})"/>`;}
   const X=x=>52+(x+4)*80,Y=sigma=>31+181*(hi-Math.log(sigma))/(hi-lo);s+=txt(53,17,'Lighter = lower KL gap; σ axis is logarithmic');[-4,-2,0,2,4].forEach(x=>s+=txt(X(x),230,x,'text-anchor="middle"'));[.15,.3,.7,1.5,3.5].forEach(v=>s+=txt(44,Y(v)+4,v,'text-anchor="end"'));
   s+=`<circle cx="${X(r.post.mean)}" cy="${Y(Math.sqrt(r.post.variance))}" r="7" fill="none" stroke="#A94F2A" stroke-width="3"/><path d="M${X(state.mu)-6} ${Y(state.sigma)}h12 M${X(state.mu)} ${Y(state.sigma)-6}v12" stroke="white" stroke-width="4"/><path d="M${X(state.mu)-6} ${Y(state.sigma)}h12 M${X(state.mu)} ${Y(state.sigma)-6}v12" stroke="#203129" stroke-width="2"/>`+txt(690,251,'candidate mean μ','text-anchor="end"')+'</svg>';
   $('chart').innerHTML=s;$('stage-title').textContent='Free-energy gap F(q) + log p(z)';$('legend').innerHTML=legends([['Circle: exact posterior',COLORS.other],['Cross: candidate q',COLORS.ink]]);
   $('metrics').innerHTML=metrics([['Mean-mismatch term',fmt(r.meanGap)],['Spread / entropy term',fmt(r.varianceGap)],['Total KL gap',fmt(r.gap)]]);
   $('side-note').textContent=`Target: mean ${fmt(r.post.mean)}, standard deviation ${fmt(Math.sqrt(r.post.variance))}. F(q) = ${fmt(r.freeEnergy)}; −log p(z) = ${fmt(-r.post.logEvidence)}. Units: nats.`;
   $('explain').textContent='Match the mean only: a positive gap remains unless the variance also matches. MAP optimizes a point; this optimizes a distribution. Entropy prevents variance collapse. With unrestricted q and negative log-likelihood loss, this is Bayes in variational form.';
   $('heatmap').addEventListener('click',e=>{const svg=e.currentTarget,p=svg.createSVGPoint();p.x=e.clientX;p.y=e.clientY;const local=p.matrixTransform(svg.getScreenCTM().inverse()),px=local.x,py=local.y;if(px<52||px>692||py<31||py>212)return;state.mu=(px-52)/80-4;state.sigma=Math.exp(hi-(py-31)*(hi-lo)/181);draw();});
  }else{
   result=chain(state);const r=result;$('stage-title').textContent=`Gaussian chain: ${r.eliminated} of 3 old states eliminated`;$('legend').innerHTML=legends([['Filtered (past + present)',COLORS.prior],['Smoothed / batch (all data)',COLORS.exact],['Observations',COLORS.other]]);
   const pts=a=>a.map((v,i)=>[i,v]);$('chart').innerHTML=plot([{points:pts(r.filtered),color:COLORS.prior},{points:pts(r.smoothed),color:COLORS.exact},{points:pts(r.z),color:COLORS.other,dash:true}],[0,3],[-2.3,4.3],[],['state index','state estimate']);
   $('metrics').innerHTML=metrics([['Last filtered mean',fmt(r.filtered[3])],['Last batch / Schur mean',fmt(r.lastMean)],['Last marginal variance',fmt(r.lastVariance)]]);
   $('extra').innerHTML='<div class="graph-extra"><div><b>Remaining precision J</b><table class="matrix">'+r.active.map(row=>'<tr>'+row.map(v=>`<td>${fmt(v,2)}</td>`).join('')+'</tr>').join('')+'</table></div><div><b>Information h</b><table class="matrix">'+r.info.map(v=>`<tr><td>${fmt(v,2)}</td></tr>`).join('')+'</table></div><div>Active states: '+Array.from({length:4-r.eliminated},(_,i)=>'x'+(i+r.eliminated)).join(' → ')+'</div></div>';
   $('side-note').textContent='x0 ~ N(0,1); xi = xi−1 + wi; R = 0.5. Forward Schur elimination gives the final marginal. Back substitution recovers the batch mean. Checked against RTS smoothing.';
   $('explain').textContent='Move only the last observation. Earlier filtered estimates stay fixed; earlier smoothed estimates change because they use future data. The last filtered and batch estimates always agree. These are different information sets, not competing answers to the same question.';
  }
 }
 app.addEventListener('input',e=>{if(e.target.id in state){state[e.target.id]=e.target.id==='shape'?e.target.value:Number(e.target.value);draw();}});
 app.addEventListener('click',e=>{const el=e.target.closest('button');if(!el)return;if(el.dataset.nav){parent.postMessage({type:'bento-inline-nav',direction:Number(el.dataset.nav)},'*');return;}const a=el.dataset.action;if(a==='reset')Object.assign(state,settings[mode]);else if(a==='optimum')state.K=mse(state).optimum;else if(a==='newton')state.x=wls(state).newton;else if(a==='mean')state.mu=variational(state).post.mean;else if(a==='posterior'){const p=variational(state).post;state.mu=p.mean;state.sigma=Math.sqrt(p.variance);}else if(a==='eliminate')state.eliminated=Math.min(3,state.eliminated+1);else if(a==='all')state.eliminated=3;else return;draw();});
 try{draw();window.KalmanFamilyLab={mode,getState:()=>({...state}),getResult:()=>result,setState:patch=>{Object.assign(state,patch);draw();}}catch(e){app.innerHTML='<p class="error">The experiment could not initialize. Reload this page.</p>';console.error(e);}
}
return{normal,scalar,bayes,mse,wls,variational,chain,solve,mount};
});
