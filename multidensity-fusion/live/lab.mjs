import { normal, logNormal, sharedPrior, correlation, rotateCov, ci2, det2, optimalCI,
  pools, bernoulli, equalGaussianOverlap } from '../math.mjs';

const $ = id => document.getElementById(id);
const color = { a:'#496e87', b:'#a94f2a', aa:'#71638b', gci:'#2f6b4f', product:'#92702e', gray:'#89938d' };
const range = (label,min,max,step,value) => ({label,min,max,step,value});
const select = (label,options,value) => ({label,options,value});
const definitions = {
  prior:{title:'One prior, not M priors',intro:'All sensors observe the same scalar x, with independent noise of variance 1. For this example, all observed values equal z.',
    controls:{p:range('Prior variance',.1,8,.1,1),m:range('Number of sensors M',1,8,1,4),z:range('Observed value z',0,4,.1,2)}},
  correlation:{title:'Reported ≠ actual error',intro:'Unbiased scalar estimates: P₁ = 1 and cross-covariance C = ρ√P₂. The errors, not the observed values, are correlated.',
    controls:{p:range('P₂',.25,4,.05,1),rho:range('Error correlation ρ',-.95,.95,.01,.8),w:range('CI weight on source 1',0,1,.01,.5)}},
  geometry:{title:'Choose the CI weight',intro:'Zero-mean Gaussians. Ellipses have Mahalanobis radius 1 (not 95% probability). Optimize log det P on a 1001-point grid.',
    controls:{angle:range('Angle between long axes',0,90,1,90),ratio:range('Second covariance scale',.4,3,.1,1),w:range('CI weight on source 1',0,1,.01,.5)}},
  pooling:{title:'Overlap or alternatives?',intro:'AA retains alternatives. GCI pools log densities. The unweighted product has no common-information correction.',
    controls:{preset:select('Input densities',{'gaussian':'Two Gaussians','mixture':'Conflicting mixtures','disjoint':'Disjoint compact supports'},'gaussian'),d:range('Separation / shift',0,6,.1,3),s:range('Source 2 standard deviation',.4,2,.1,1),w:range('Weight on source 1',0,1,.01,.5)}},
  bernoulli:{title:'Does an object exist?',intro:'One possible object: existence r and Gaussian location conditional on existence. Both spatial variances equal 1.',
    controls:{r1:range('Existence r₁',.01,.99,.01,.9),r2:range('Existence r₂',.01,.99,.01,.9),d:range('Spatial mean separation',0,8,.1,4),w:range('Weight on source 1',0,1,.01,.5)}},
  rumors:{title:'Copies are not evidence',intro:'M unbiased estimates all report variance 1. Compare perfect copies with genuinely independent errors.',
    controls:{m:range('Number of messages M',1,20,1,5),model:select('Actual error model',{'copies':'Perfectly correlated copies','independent':'Independent errors'},'copies')}}
};
const requested = new URLSearchParams(location.search).get('demo') || 'prior';
const mode = Object.hasOwn(definitions, requested) ? requested : 'prior';
const def = definitions[mode], state = {};
$('title').textContent = def.title; $('intro').textContent = def.intro;
if (window.parent !== window) document.body.classList.add('embedded');
for (const [id,c] of Object.entries(def.controls)) {
  state[id] = c.value;
  const label = document.createElement('label'); label.className = 'control'; label.htmlFor = id;
  const caption = document.createElement('span'); caption.textContent = c.label; label.append(caption);
  const input = document.createElement(c.options ? 'select' : 'input'); input.id = id;
  if (c.options) for (const [value,title] of Object.entries(c.options)) {
    const option = document.createElement('option'); option.value = value; option.textContent = title; input.append(option);
  } else {
    Object.assign(input,{type:'range',min:c.min,max:c.max,step:c.step});
    const out = document.createElement('output'); out.id = `${id}-out`; out.htmlFor = id; caption.append(out);
  }
  input.value = c.value; label.append(input); $('inputs').append(label);
  input.addEventListener('input',()=>{state[id] = c.options ? input.value : Number(input.value); render();});
}
$('reset').onclick = () => {for (const [id,c] of Object.entries(def.controls)) {state[id]=c.value;$(id).value=c.value;} render();};
const fmt = n => n === null ? 'undefined' : !Number.isFinite(n) ? '−∞' : Math.abs(n) > 0 && Math.abs(n) < .001 ? n.toExponential(2) : n.toFixed(3);
function metrics(items) { $('metrics').replaceChildren(...items.map(([name,value])=>{
  const div=document.createElement('div');div.className='metric';const small=document.createElement('small');small.textContent=name;
  const strong=document.createElement('strong');strong.textContent=typeof value==='string'?value:fmt(value);div.append(small,strong);return div;
})); }
function legend(items) { $('legend').innerHTML=items.map(([name,c,dash])=>`<span><i style="border-color:${c};${dash?'border-top-style:dashed':''}"></i>${name}</span>`).join(''); }
const grid = (lo,hi,n=501) => Array.from({length:n},(_,i)=>lo+(hi-lo)*i/(n-1));
function chart(xs,series,{xlabel='state x',ylabel='density',ymax,description=''}={}) {
  const x0=xs[0],x1=xs.at(-1),max=ymax??Math.max(.01,...series.flatMap(s=>s.values??[]))*1.14;
  const X=x=>56+(x-x0)/(x1-x0)*668, Y=y=>260-y/max*226;
  let content=`<title id="plot-title">${def.title}</title><desc id="plot-desc">${description}</desc>`;
  for(let i=0;i<=4;i++){const y=i*max/4;content+=`<path d="M56 ${Y(y)}H724" stroke="#d8ded7" fill="none"/><text x="48" y="${Y(y)+4}" text-anchor="end">${y.toFixed(2)}</text>`;}
  for(let i=0;i<=4;i++){const x=x0+(x1-x0)*i/4;content+=`<text x="${X(x)}" y="280" text-anchor="middle">${x.toFixed(1)}</text>`;}
  content+=`<text x="390" y="302" text-anchor="middle">${xlabel}</text><text x="56" y="17">${ylabel}</text>`;
  for (const s of series) if(s.values) content+=`<path d="${s.values.map((v,i)=>`${i?'L':'M'}${X(xs[i]).toFixed(2)} ${Y(v).toFixed(2)}`).join(' ')}" fill="none" stroke="${s.color}" stroke-width="${s.width??2.5}" ${s.dash?'stroke-dasharray="7 5"':''}/>`;
  $('plot').innerHTML=content;
}
function bars(entries,description) {
  const max=Math.max(...entries.map(e=>e[1]))*1.25, width=600/entries.length;
  let html=`<title id="plot-title">${def.title}</title><desc id="plot-desc">${description}</desc><text x="55" y="22">Error variance / reported bound</text>`;
  for(let i=0;i<=4;i++){const v=max*i/4,y=250-v/max*200;html+=`<path d="M55 ${y}H735" stroke="#d8ded7"/><text x="45" y="${y+4}" text-anchor="end">${v.toFixed(2)}</text>`;}
  entries.forEach(([label,value,c],i)=>{const x=78+i*width,h=value/max*200;html+=`<rect x="${x}" y="${250-h}" width="${width-22}" height="${h}" rx="4" fill="${c}"/><text x="${x+(width-22)/2}" y="${243-h}" text-anchor="middle">${fmt(value)}</text><text x="${x+(width-22)/2}" y="276" text-anchor="middle">${label}</text>`;});
  $('plot').innerHTML=html;
}
function explain(text,warn=false){$('explain').textContent=text;$('explain').classList.toggle('warning',warn);}
function render(){
  for(const [id,c] of Object.entries(def.controls)) if(!c.options) $(`${id}-out`).value=Number(state[id]).toFixed(c.step>=1?0:2);
  const s=state;
  if(mode==='prior'){
    const f=sharedPrior(s.p,s.m,s.z), xs=grid(-4,7);
    legend([['Correct Bayes',color.gci],['Uncorrected product',color.product],['GCI (equal weights)',color.aa]]);
    chart(xs,[['correct',color.gci],['product',color.product],['gci',color.aa]].map(([key,c])=>({color:c,values:xs.map(x=>normal(x,f[key].mean,f[key].variance))})),{description:'Correct posterior compared with a product that repeats the prior and a tempered logarithmic pool.'});
    metrics([['Correct variance',f.correct.variance],['Product variance',f.product.variance],['Correct mean',f.correct.mean],['Product mean',f.product.mean]]);
    explain('The correct posterior adds the prior precision once. The product adds it M times and pulls the answer toward the prior mean 0. Equal-weight GCI keeps the common prior once but tempers each new likelihood by 1/M.');
  }else if(mode==='correlation'){
    const r=correlation(1,s.p,s.rho,s.w);legend([]);
    bars([['Naive: reported',r.naiveReported,color.product],['Naive: actual',r.naiveActual,color.b],['Known C',r.oracle,color.a],['CI: bound',r.ciReported,color.gci],['CI: actual',r.ciActual,color.aa]],'Compare assumed and true variances using the specified joint error covariance.');
    metrics([['Naive actual / reported',r.naiveActual/r.naiveReported],['CI actual / bound',r.ciActual/r.ciReported],['Oracle weight on estimate 1',r.oracleAlpha]]);
    explain('Positive correlation can make the independence-based variance too small. CI is an upper bound here because the marginal error variances are valid. The oracle uses known C; its linear weight may lie outside [0,1].',r.naiveActual>r.naiveReported+1e-9);
  }else if(mode==='geometry'){
    const a=[4,0,.25],b=rotateCov(4*s.ratio,.25*s.ratio,s.angle*Math.PI/180),p=ci2(a,b,s.w),best=optimalCI(a,b);
    legend([['Source 1',color.a],['Source 2',color.b],['CI',color.gci]]);
    let html='<title id="plot-title">CI covariance geometry and log determinant</title><desc id="plot-desc">Left: unit Mahalanobis contours. Right: log determinant versus fusion weight.</desc>';
    for(const [cov,c] of [[a,color.a],[b,color.b],[p,color.gci]]){
      const l=Math.sqrt(cov[0]),m=cov[1]/l,n=Math.sqrt(cov[2]-m*m);
      const points=grid(0,2*Math.PI,121).map((t,i)=>`${i?'L':'M'}${(200+48*l*Math.cos(t)).toFixed(2)} ${(153-48*(m*Math.cos(t)+n*Math.sin(t))).toFixed(2)}`).join(' ');
      html+=`<path d="${points}Z" stroke="${c}" stroke-width="3" fill="none"/>`;
    }
    html+='<path d="M25 153H375M200 15V290" stroke="#d8ded7"/><text x="32" y="300">x₁ →</text><text x="208" y="20">x₂</text>';
    const ws=grid(0,1,101),ys=ws.map(w=>Math.log(det2(ci2(a,b,w)))),lo=Math.min(...ys)-.2,hi=Math.max(...ys)+.2;
    const xx=w=>450+260*w,yy=y=>255-(y-lo)/(hi-lo)*195;
    html+=`<text x="450" y="30">log det P(ω)</text><path d="M450 60V255H710" stroke="#89938d" fill="none"/><path d="${ys.map((v,i)=>`${i?'L':'M'}${xx(ws[i])} ${yy(v)}`).join(' ')}" stroke="${color.gci}" stroke-width="3" fill="none"/><circle cx="${xx(s.w)}" cy="${yy(Math.log(det2(p)))}" r="6" fill="${color.b}"/><text x="450" y="278">0</text><text x="710" y="278">1</text><text x="560" y="300">weight ω</text>`;
    $('plot').innerHTML=html;metrics([['Selected det P',det2(p)],['Grid-optimal ω',best.weight],['Minimum det P',det2(best.covariance)]]);
    explain('Different well-observed directions can make an interior CI weight useful. Coincident covariances give a flat objective; the grid reports its first minimizer. A covariance-only objective cannot diagnose disagreement between means.');
  }else if(mode==='pooling'){
    const xs=grid(-12,12,1201),dx=xs[1]-xs[0];let la,lb;
    const compact=s.preset==='disjoint';$('d').disabled=compact;$('s').disabled=compact;
    if(compact){la=xs.map(x=>x>-4&&x<-1?0:-Infinity);lb=xs.map(x=>x>1&&x<4?0:-Infinity);}
    else if(s.preset==='mixture'){
      la=xs.map(x=>Math.log(.8*normal(x,-2,1)+.2*normal(x,2,1)));
      lb=xs.map(x=>Math.log(.2*normal(x,-2+s.d-3,s.s*s.s)+.8*normal(x,2+s.d-3,s.s*s.s)));
    }else {la=xs.map(x=>logNormal(x,-s.d/2,1));lb=xs.map(x=>logNormal(x,s.d/2,s.s*s.s));}
    const f=pools(la,lb,dx,s.w);
    legend([['p₁',color.a,true],['p₂',color.b,true],['AA',color.aa],['GCI',color.gci],['Product',color.product]]);
    chart(xs,[['p1',color.a,true],['p2',color.b,true],['aa',color.aa],['product',color.product],['gci',color.gci]].map(([key,c,dash])=>({values:f[key],color:c,dash})),{description:'Numerically normalized input and fused densities on [-12,12]. Undefined pools are omitted.'});
    metrics([['Spatial overlap Z',Math.exp(f.logZ)],['GCI',f.gci?'normalized':'undefined'],['Product',f.product?'normalized':'undefined']]);
    explain(f.gci?'Curves use trapezoidal quadrature on [-12,12], with log-domain normalization. With Gaussian inputs, GCI covariance does not grow when the means move apart; small Z warns of conflict.':'For positive weights, these inputs have no common support. Z = 0, so normalized GCI does not exist. AA remains valid. At ω = 0 or 1, GCI returns the active source exactly.',!f.gci);
  }else if(mode==='bernoulli'){
    const eta=equalGaussianOverlap(s.d,1,s.w),r=bernoulli(s.r1,s.r2,s.w,eta),ds=grid(0,8);
    legend([['AA existence',color.aa],['GCI existence',color.gci]]);
    chart(ds,[{color:color.aa,values:ds.map(()=>r.aa)},{color:color.gci,values:ds.map(d=>bernoulli(s.r1,s.r2,s.w,equalGaussianOverlap(d,1,s.w)).gci)}],{xlabel:'spatial mean separation d',ylabel:'existence probability',ymax:1,description:'Existence after full Bernoulli density fusion, as spatial disagreement grows.'});
    metrics([['Selected separation',s.d],['Spatial overlap η',eta],['AA existence',r.aa],['GCI existence',r.gci]]);
    explain('η = exp[−ω(1−ω)d²/2]. The GCI normalizer includes both the empty set and the singleton set. Spatial disagreement can lower existence even when both sensors report high r; averaging only r misses this effect.');
  }else{
    const ms=grid(1,20,20),actual=s.model==='copies'?1:1/s.m;
    legend([['Product: reported',color.product],['Actual mean-estimate error',color.b,true],['GCI / AA density variance',color.gci]]);
    chart(ms,[{color:color.product,values:ms.map(m=>1/m)},{color:color.gci,values:ms.map(()=>1)},{color:color.b,dash:true,values:ms.map(m=>s.model==='copies'?1:1/m)}],{xlabel:'number of messages M',ylabel:'variance',ymax:1.1,description:'Repeated pooling compared with the actual error variance of the arithmetic mean estimate.'});
    metrics([['Selected M',String(s.m)],['Product: reported',1/s.m],['Actual error of mean',actual],['AA / GCI density',1]]);
    explain('The density curves assume every received density is N(0,1). The actual-error curve is a separate model calculation for the arithmetic mean of unbiased estimates. Identical reported densities alone cannot tell you whether their estimation errors are independent.',s.model==='copies'&&s.m>1);
  }
}
render();
// Only this trusted, first-party frame sends these messages. Preserve slider keys.
function notify(type,extra={}){if(parent!==window)parent.postMessage({type,...extra},location.origin);}
addEventListener('keydown',event=>{
  if(event.key==='Escape'){notify('bento-inline-focus');return;}
  if(event.target.closest('input,select,button,textarea,a'))return;
  if(event.key==='PageDown'||event.key==='PageUp'){event.preventDefault();notify('bento-inline-nav',{direction:event.key==='PageDown'?1:-1});}
});
notify('bento-inline-ready');
