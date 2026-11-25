// Sampling deck in the site's shared Bento slide system (see radar-slam/bento-deck.mjs).
// Every number quoted on a slide is computed here from model.js, the same code the live labs run.
import S from './model.js';
const C={paper:'#F7F5EF',panel:'#FFFEFB',ink:'#203129',muted:'#66756E',rule:'#D8DED7',green:'#2F6B4F',soft:'#E7F0EA',rust:'#A94F2A',warm:'#F5E8DF',blue:'#496E87',cool:'#E7EEF3',amber:'#986B22'};
const serif="Georgia, 'Times New Roman', serif",sans="Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",mono="'SFMono-Regular', Consolas, monospace";
const R=String.raw;
const M=s=>`<span class="math-tex math-display">\\[${s}\\]</span>`;
const I=s=>`<span class="math-tex math-inline">\\(${s}\\)</span>`;
function text(id,x,y,w,h,html,o={}){return {id,type:'text',x,y,w,h,rotation:0,opacity:1,html,fontSize:18,fontFamily:sans,fontWeight:400,color:C.ink,align:'left',valign:'top',lineHeight:1.4,...o};}
function box(id,x,y,w,h,fill=C.panel,stroke=C.rule,o={}){return {id,type:'shape',shape:'rect',x,y,w,h,fill,stroke,strokeWidth:1,radius:14,rotation:0,opacity:1,...o};}
function panel(id,x,y,w,h,label,html,o={}){return [box(id+'-bg',x,y,w,h,o.fill||C.panel,o.stroke||C.rule),text(id+'-label',x+22,y+17,w-44,23,label,{fontFamily:mono,fontSize:11,fontWeight:800,color:o.accent||C.green,letterSpacing:.8}),text(id+'-body',x+22,y+(o.top||50),w-44,h-(o.top||50)-14,html,{fontSize:o.size||18,lineHeight:1.45,...(o.body||{})})];}
function nativeTable(id,x,y,w,h,headers,rows,o={}){return {id,type:'table',x,y,w,h,rotation:0,opacity:1,header:true,columns:(o.columns||headers.map(()=>1)).map(w=>({w})),rows:[{cells:headers.map(html=>({html}))},...rows.map(row=>({cells:row.map(html=>({html}))}))],style:{headerBg:o.headerBg||C.green,headerColor:'#FFFFFF',zebra:o.zebra||C.soft,borderColor:C.rule,borderWidth:1,cellPadX:o.padX||13,cellPadY:o.padY||10,fontSize:o.fontSize||16,color:C.ink,fontFamily:sans,radius:12}};}
function callout(id,x,y,w,h,html,o={}){return [box(id+'-bg',x,y,w,h,o.fill||C.warm,o.stroke||C.rust,{radius:10}),text(id+'-text',x+18,y+12,w-36,h-24,html,{fontSize:o.size||16,fontWeight:o.weight||600,color:C.ink,valign:'middle',lineHeight:1.4})];}

/* ---------------- references (footer links) ---------------- */
const refs={
 GG84:['Geman & Geman 1984','https://doi.org/10.1109/TPAMI.1984.4767596','Stochastic relaxation, Gibbs distributions, and the Bayesian restoration of images. IEEE TPAMI 6(6).'],
 MH53:['Metropolis et al. 1953','https://doi.org/10.1063/1.1699114','Equation of state calculations by fast computing machines. J. Chem. Phys. 21.'],
 H70:['Hastings 1970','https://doi.org/10.1093/biomet/57.1.97','Monte Carlo sampling methods using Markov chains and their applications. Biometrika 57(1).'],
 N11:['Neal 2011','https://arxiv.org/abs/1206.1901','MCMC using Hamiltonian dynamics. Handbook of Markov Chain Monte Carlo, ch. 5.'],
 HG14:['Hoffman & Gelman 2014','https://arxiv.org/abs/1111.4246','The No-U-Turn Sampler: adaptively setting path lengths in Hamiltonian Monte Carlo. JMLR 15.'],
 N03:['Neal 2003','https://arxiv.org/abs/physics/0009028','Slice sampling. Annals of Statistics 31(3).'],
 O13:['Owen 2013, ch. 9','https://artowen.su.domains/mc/Ch-var-is.pdf','Monte Carlo theory, methods and examples, ch. 9: importance sampling.'],
 GSS93:['Gordon, Salmond & Smith 1993','https://doi.org/10.1049/ip-f-2.1993.0015','Novel approach to nonlinear/non-Gaussian Bayesian state estimation. IEE Proc. F 140(2).'],
 DJ11:['Doucet & Johansen 2011','https://www.stats.ox.ac.uk/~doucet/doucet_johansen_tutorialPF2011.pdf','A tutorial on particle filtering and smoothing: fifteen years later.'],
 V21:['Vehtari et al. 2021','https://arxiv.org/abs/1903.08008','Rank-normalization, folding, and localization: an improved R-hat for assessing convergence of MCMC. Bayesian Analysis 16(2).'],
 SP:['Original Sampling Playground','https://sampling-playground.sugary-book-2058.chatgpt.site/','The interactive page this deck was adapted from; samplers re-implemented and re-verified here.']
};
const source=(...keys)=>keys.map(k=>`<a href="${refs[k][1]}" target="_blank" rel="noopener">${refs[k][0]}</a>`).join(' · ');

/* ---------------- numbers from the model ---------------- */
const f2=v=>v.toFixed(2),f1=v=>v.toFixed(1),pct=v=>Math.round(100*v)+'%',neg=s=>String(s).replace('-','−');
const rho=S.CHAIN.rho;
const geo={ridge:Math.sqrt(1+rho),across:Math.sqrt(1-rho),cond:Math.sqrt(1-rho*rho),start:S.mahalanobis(S.CHAIN.start,rho),stable:2*Math.sqrt(1-rho)};
const BURN=1000,LONG=20000;
const long=Object.fromEntries(['gibbs','mh','hmc','slice'].map(m=>{
 const run=S.runChain(m,{steps:BURN+LONG}),moves=run.moves.slice(BURN),xs=run.points.slice(BURN+1).map(p=>p.x);
 const ess=S.essGeyer(xs),evals=moves.reduce((a,v)=>a+v.evals,0)+(m==='hmc'?moves.length:0);
 return [m,{acc:moves.filter(v=>v.accepted).length/LONG,essPerK:1000*ess/LONG,evalsPerMove:evals/LONG,essPerKEval:1000*ess/evals}];
}));
const rej={sup:S.supRatio(),lab:S.makeRejection().draw(1000).stats()};
const imp={frac:1/S.weightSecondMoment(),narrow:S.weightSecondMoment(0,.5),lab:S.makeImportance().draw(1000).stats()};
imp.z=(imp.lab.est-S.TRUE_MEAN)/imp.lab.se;
// Two-sided normal tail probability via the Abramowitz–Stegun erfc approximation.
const tail=z=>{const x=Math.abs(z)/Math.SQRT2,k=1/(1+.3275911*x);return k*(.254829592+k*(-.284496736+k*(1.421413741+k*(-1.453152027+k*1.061405429))))*Math.exp(-x*x);};
imp.p=tail(imp.z);
// Calibration of the ± interval over 400 other seeds of the same 1,000-draw run.
imp.outside=Array.from({length:400},(_,i)=>{const s=S.makeImportance({seed:i+1}).draw(1000).stats();return Math.abs(s.est-S.TRUE_MEAN)/s.se>2;}).filter(Boolean).length/400;
const smc={steady:S.kalmanSteadySd(),half:S.makeFilter().run().stats(),never:S.makeFilter({rule:'never'}).run().stats(),big:S.makeFilter({N:200}).run().stats()};
export const numbers={geo,long,rej,imp,smc};

/* ---------------- slide scaffolding ---------------- */
function chrome(src){return [box('chrome-rule',72,669,1136,1,C.rule,C.rule,{radius:0}),text('chrome-home',72,684,278,20,'BAI LIPING · ESTIMATION NOTES',{fontFamily:mono,fontSize:10,color:C.muted,link:'https://bailiping.com/'}),text('chrome-source',350,684,580,21,src,{fontFamily:mono,fontSize:10,color:C.muted,align:'center'}),text('chrome-index',1032,684,176,20,'CONTENTS ↗',{fontFamily:mono,fontSize:10,color:C.green,align:'right',link:'contents'})];}
const slides=[];
const toc={};
function add(id,section,title,sub,elements,notes,src,tocTitle){
 slides.push({id,background:C.paper,transition:'none',notes,elements:[
  text('eyebrow',72,35,1110,23,section.toUpperCase(),{fontFamily:mono,fontSize:11,fontWeight:800,color:C.rust,letterSpacing:1.3}),
  text('heading',72,70,1136,58,title,{fontFamily:serif,fontSize:38,fontWeight:700,lineHeight:1.05}),
  text('subtitle',72,129,1136,44,sub,{fontSize:16,color:C.muted,lineHeight:1.35}),
  ...elements,...chrome(src)]});
 if(tocTitle)toc[id]=tocTitle;
}
// Equation band across the top, then two explanatory panels.
function eqTwo(id,section,title,sub,[eqLabel,eqHtml,eqH=150],[lLabel,left],[rLabel,right],notes,src,tocTitle,o={}){
 const y=186+eqH+18,h=633-y;
 add(id,section,title,sub,[
  ...panel(id+'-eq',72,186,1136,eqH,eqLabel,eqHtml,{fill:C.soft,size:o.eqSize||21,top:42,body:{align:'center',color:C.ink}}),
  ...panel(id+'-left',72,y,558,h,lLabel,left,{size:o.size||20}),
  ...panel(id+'-right',650,y,558,h,rLabel,right,{size:o.size||20,accent:C.rust})
 ],notes,src,tocTitle);
}
function four(id,section,title,sub,cards,notes,src,tocTitle,o={}){
 add(id,section,title,sub,cards.flatMap(([label,body,fill,accent],i)=>panel(id+'-'+i,72+(i%2)*578,186+Math.floor(i/2)*228,558,210,label,body,{fill:fill||(i===0?C.soft:C.panel),accent:accent||C.green,size:o.size||20})),notes,src,tocTitle);
}
// A live experiment replaces the exact rectangle of its static fallback.
const bounds={x:72,y:180,width:1136,height:475};
const live=[];
function liveSlide(intro,section,title,prompt,fallbackTitle,notes,src,plainTitle=title){
 const id=intro+'-live';
 add(id,section+' · experiment',title,prompt,[
  box('fallback-bg',72,180,1136,475,C.panel,C.rule),
  {id:'fallback',type:'image',x:73,y:181,w:1134,h:473,src:`fallback/${intro}.png`,fit:'contain',alt:`${fallbackTitle}: initial state of the live lab`},
  box('live-demo-mount',72,180,1136,475,'rgba(255,255,255,0)','rgba(255,255,255,0)',{opacity:0})
 ],`${fallbackTitle}. ${notes} Direct lab: /sampling-playground/live/?demo=${intro}. Page Up / Page Down navigate from inside the lab; Escape returns focus to the presentation.`,src);
 live.push({introSlide:intro,slide:id,demo:intro,title:plainTitle});
}

/* ---------------- 1 · cover ---------------- */
add('overview','Sampling / a field guide','See how samplers think.','Seven Monte Carlo methods, each running live on a target you can see.',[
 text('hero',72,200,690,120,`How do I draw from ${I(R`\pi`)}?<br><span style="color:#2F6B4F">What does each draw cost?</span>`,{fontFamily:serif,fontSize:48,fontWeight:700,lineHeight:1.1}),
 text('cover-copy',74,340,680,92,'Four chain samplers on one correlated Gaussian, then rejection, importance weights and a particle filter—each checked against a known answer.',{fontSize:21,color:C.muted}),
 ...[['01','Chains','gibbs'],['02','Weights','importance'],['03','Choosing','compare']].flatMap(([n,t,link],i)=>[box('cover-card-'+i,72+i*227,452,210,104,i===1?C.cool:C.soft),text('cover-card-'+i+'-label',90+i*227,470,176,68,n+'<br><b>'+t+'</b>',{fontSize:18,link})]),
 box('cover-route',790,188,418,445,C.panel,C.rule),
 ...[['01','The idea & the targets','idea'],['02','Gibbs & Metropolis–Hastings','gibbs'],['03','HMC & slice sampling','hmc'],['04','Four chains, measured','scorecard'],['05','Rejection, importance, SMC','rejection'],['06','Choose & diagnose','compare']].map(([n,t,link],i)=>text('route-'+n,816,218+i*62,363,38,`<span style="color:#A94F2A">${n}</span> &nbsp; ${t} →`,{fontSize:18,link})),
 text('cover-boundary',74,582,690,20,'SLIDE COUNT',{fontSize:12,fontFamily:mono,color:C.green}),
 text('cover-companion',74,608,690,24,`Bai Liping · adapted from the <a href="${refs.SP[1]}" target="_blank" rel="noopener">original Sampling Playground</a> · reviewed 24 Sep 2026`,{fontSize:14,color:C.muted})
],'Every lab is the real algorithm with a fixed seed, so the fallback image, a remount and the numbers on the slides agree. Advance normally: each live lab loads on the slide after its introduction.','SAMPLING · A FIELD GUIDE');

/* ---------------- 2 · contents (entries added after every slide exists) ---------------- */
add('contents','Contents','What’s inside.','Every entry opens its slide. Each live lab follows its introduction.',[],'Every entry is a link, and the number is the page you will land on.','SAMPLING · CONTENTS');

/* ---------------- 3–5 · idea, families, chain target ---------------- */
eqTwo('idea','01 · the idea','A sampler turns an integral into an average.','The cloud of samples stands in for the distribution; methods differ in how they produce it.',
 ['MONTE CARLO IN ONE LINE',M(R`\mathbb E_\pi[f]=\int f(x)\,\pi(x)\,dx\;\approx\;\frac1N\sum_{n=1}^{N}f\big(x^{(n)}\big)\quad\text{or}\quad\sum_{n=1}^{N}\bar w_n\,f\big(x^{(n)}\big)`),150],
 ['WHAT YOU USUALLY HAVE',`An unnormalized density ${I(R`\tilde\pi(x)=Z\,\pi(x)`)} whose constant ${I('Z')} is unknown.<br><br>Every method here touches ${I(R`\tilde\pi`)} only through ratios, conditionals, gradients or self-normalized weights, so ${I('Z')} cancels or never appears.`],
 ['WHAT COUNTS AS A SAMPLE',`<b>Chain</b> · correlated draws; average after burn-in.<br><b>Exact draws</b> · independent, equally weighted.<br><b>Weighted draws</b> · independent, unequal weights ${I(R`\bar w_n`)}.<br><b>Particles</b> · a weighted population moved through time.`],
 'The two estimators on the band: plain averages for chains and exact draws, self-normalized weighted averages for importance sampling and particle filters. The Markov-chain average is consistent under ergodicity; the self-normalized estimator is consistent but slightly biased for finite N.',source('O13'),'A sampler turns an integral into an average');

add('families','01 · the atlas','Seven methods, two families.','They return different objects, so they fail in different ways.',[
 ...panel('chain',72,186,558,447,'CHAIN SAMPLERS · ONE STATE AT A TIME',`<b>Gibbs</b> — exact conditional draws, one coordinate at a time.<br><br><b>Metropolis–Hastings</b> — propose, then accept or stay.<br><br><b>Hamiltonian Monte Carlo</b> — gradient-guided trajectories.<br><br><b>Slice</b> — uniform draws under the density curve.<br><br><span style="color:#66756E">Output: a correlated Markov chain whose stationary distribution is ${I(R`\pi.`)}</span>`,{fill:C.soft,size:21}),
 ...panel('population',650,186,558,447,'POPULATION METHODS · MANY DRAWS AT ONCE',`<b>Rejection</b> — keep proposals that land under the target.<br><br><b>Importance</b> — keep every proposal and reweight it.<br><br><b>Sequential Monte Carlo</b> — propagate, weight and resample a particle population.<br><br><span style="color:#66756E">Output: exact independent draws, weighted draws, or a weighted population that tracks a sequence of targets.</span>`,{size:21,accent:C.rust})
],'Chain samplers are judged by mixing: how quickly correlated draws forget their start and cover the target. Population methods are judged by weight balance or acceptance. Both families end with diagnostics.',source('MH53','GG84','O13'),'Seven methods, two families');

eqTwo('target','02 · the chain target','One narrow ridge for all four chains.','A correlated Gaussian is easy to check against—and hard for coordinate-wise moves.',
 ['THE CHAIN TARGET',M(R`\pi(x)\propto\exp\!\left(-\tfrac12\,x^{\mathsf T}\Sigma^{-1}x\right),\qquad \Sigma=\begin{bmatrix}1&\rho\\ \rho&1\end{bmatrix},\qquad \rho=${rho}`),150],
 ['WHAT EACH SAMPLER USES',`Gibbs uses the conditionals ${I(R`x_1\mid x_2\sim\mathcal N(\rho x_2,\,1-\rho^2)`)}.<br>HMC uses the gradient ${I(R`\nabla\log\pi(x)=-\Sigma^{-1}x`)}.<br>Metropolis–Hastings and slice sampling only evaluate ${I(R`\tilde\pi`)}.<br><br>Contours in the labs are Mahalanobis radii 1 and 2: they enclose 39% and 86% of the mass.`],
 ['WHY IT IS HARD',`Along the ridge the standard deviation is ${I(R`\sqrt{1+\rho}=${f2(geo.ridge)}`)}; across it only ${I(R`\sqrt{1-\rho}=${f2(geo.across)}`)}.<br><br>Axis-aligned moves have conditional width ${I(R`\sqrt{1-\rho^2}=${f2(geo.cond)}`)}, so they must zig-zag.<br><br>Every chain starts at ${I('(-2.35,\\,-2.05)')}, ${f2(geo.start)} Mahalanobis units out along the ridge.`],
 'Marginal variances are one; the correlation is set by the slider in every chain lab. A Mahalanobis radius r in two dimensions encloses probability 1 − exp(−r²/2).',source('GG84','N11'),'One narrow ridge for all four chains',{size:18});

/* ---------------- 6–8 · Gibbs ---------------- */
eqTwo('gibbs','03 · gibbs sampling','Gibbs: condition, then draw.','Replace one coordinate at a time with an exact draw from its full conditional.',
 ['ONE MOVE = ONE EXACT CONDITIONAL DRAW',M(R`x_1'\sim\mathcal N\!\left(\rho\,x_2,\;1-\rho^2\right),\qquad x_2'\sim\mathcal N\!\left(\rho\,x_1',\;1-\rho^2\right)`),140],
 ['KEY IDEA',`Freeze every coordinate but one and redraw it from its full conditional ${I(R`\pi(x_j\mid x_{-j}).`)}<br><br>Nothing is rejected and there is no step size to tune. The lab alternates ${I('x_1')} and ${I('x_2')} (a systematic scan).`],
 ['WHAT TO WATCH NEXT',`Moves are horizontal or vertical, about ${I(R`\pm2\sqrt{1-\rho^2}=\pm${f2(2*geo.cond)}`)} wide. The green bell is the conditional being sampled.<br><br>Raise ${I(R`\rho`)} to 0.95: the bell narrows to ${I(R`\pm${f2(2*Math.sqrt(1-.95*.95))}`)} while the ridge stays long, so progress slows even though every draw is accepted.`],
 'Gibbs needs full conditionals that are available and cheap to sample. Blocking correlated coordinates or reparameterizing removes the zig-zag. The lab numbers the first twelve moves.',source('GG84'),'Gibbs: condition, then draw');
liveSlide('gibbs','03 · gibbs sampling','Gibbs: condition, then draw.',`Step the chain and watch the green conditional it draws from; then drag ${I(R`\rho`)} toward 0.95.`,'Gibbs sampling on the correlated Gaussian','Twelve moves are shown at the start; Step adds one coordinate update. Work is counted in conditional draws.',source('GG84'));

add('gibbs-mh','03 · gibbs sampling','Gibbs is Metropolis–Hastings that never rejects.','Propose from the exact conditional and the acceptance ratio cancels to one.',[
 ...panel('gm-eq',72,186,1136,214,'ONE CANCELLATION',M(R`q(x'\mid x)=\pi(x_j'\mid x_{-j})\,\mathbb 1[x'_{-j}=x_{-j}],\qquad \pi(x)=\pi(x_j\mid x_{-j})\,\pi(x_{-j})`)+M(R`\frac{\pi(x')\,q(x\mid x')}{\pi(x)\,q(x'\mid x)}=\frac{\pi(x_j'\mid x_{-j})\,\pi(x_{-j})\;\pi(x_j\mid x_{-j})}{\pi(x_j\mid x_{-j})\,\pi(x_{-j})\;\pi(x_j'\mid x_{-j})}=1`),{fill:C.soft,size:19,top:42,body:{align:'center'}}),
 ...panel('gm-left',72,418,558,215,'WHY IT MATTERS','The exact conditional already proposes in the right proportion, so there is nothing left to correct. That is the whole difference between Gibbs and a general Metropolis–Hastings update.',{size:20}),
 ...panel('gm-right',650,418,558,215,'THE PRICE','The full conditional must be available and sampleable. Correlation is not removed—only rejection is. When conditionals are awkward, a Metropolis step inside Gibbs is common.',{size:20,accent:C.rust})
],'The proposal changes only coordinate j; the indicator keeps the others fixed. Using the factorization of π, every term cancels. Metropolis-within-Gibbs replaces an unavailable conditional draw with an MH update for that coordinate.',source('H70','GG84'),'Gibbs is Metropolis–Hastings that never rejects');

/* ---------------- 9–10 · MH ---------------- */
eqTwo('mh','04 · metropolis–hastings','Metropolis–Hastings: propose, then judge.','Any proposal becomes a valid sampler once an acceptance step corrects it.',
 ['ACCEPT OR STAY',M(R`\alpha(x\to x')=\min\!\left(1,\;\frac{\tilde\pi(x')\,q(x\mid x')}{\tilde\pi(x)\,q(x'\mid x)}\right),\qquad x'=x+\sigma\varepsilon,\;\;\varepsilon\sim\mathcal N(0,I)`),140],
 ['KEY IDEA',`For the symmetric random walk the ${I('q')} terms cancel: uphill moves are always accepted, downhill moves with probability ${I(R`\tilde\pi(x')/\tilde\pi(x)`)}.<br><br>A rejection repeats the current state—that repeat is part of the sample. Only ratios of ${I(R`\tilde\pi`)} are needed.`],
 ['WHAT TO WATCH NEXT',`The dashed circle is ${I(R`2\sigma`)} around the current state; rust rings are rejected proposals.<br><br>Shrink ${I(R`\sigma`)} to 0.15: nearly everything is accepted, but the chain crawls. Grow it to 2.5: most proposals leave the ridge and are rejected. Default ${I(R`\sigma=${S.CHAIN.mh.sigma}`)}.`],
 'Acceptance in the lab is the fraction of proposals accepted; the start point is not counted. For a random-walk proposal the best acceptance rate depends on dimension; there is no universal target.',source('MH53','H70'),'Metropolis–Hastings: propose, then judge');
liveSlide('mh','04 · metropolis–hastings','Metropolis–Hastings: propose, then judge.',`Step until you see a rejection, then compare ${I(R`\sigma=0.15`)} with ${I(R`\sigma=2.5`)}.`,'Random-walk Metropolis–Hastings on the correlated Gaussian','Each move costs one density evaluation. Rejected proposals stay on the plot as faint rings.',source('MH53','H70'));

/* ---------------- 11–12 · HMC ---------------- */
eqTwo('hmc','05 · hamiltonian monte carlo','HMC: follow the gradient, then correct.','Momentum and simulated dynamics turn many gradient evaluations into one distant proposal.',
 ['SIMULATE, THEN ACCEPT BY ENERGY',M(R`H(x,p)=-\log\tilde\pi(x)+\tfrac12\,p^{\mathsf T}p,\qquad p\sim\mathcal N(0,I),\qquad \alpha=\min\!\left(1,\;e^{-\Delta H}\right)`),140],
 ['LEAPFROG',`${M(R`p\leftarrow p+\tfrac{\varepsilon}{2}\nabla\log\tilde\pi(x),\;\;x\leftarrow x+\varepsilon p,\;\;p\leftarrow p+\tfrac{\varepsilon}{2}\nabla\log\tilde\pi(x)`)}Repeated ${I('L')} times. Leapfrog is reversible and volume-preserving, so only the energy error ${I(R`\Delta H`)} enters the test. Lab: ${I(R`\varepsilon=${S.CHAIN.hmc.eps},\;L=${S.CHAIN.hmc.L}`)}.`],
 ['WHAT TO WATCH NEXT',`The amber path is the latest trajectory; its ${I(R`\Delta H`)} is printed at the end.<br><br>Raise ${I(R`\varepsilon`)}: the energy error grows and rejections appear. Beyond ${I(R`\varepsilon=2\sqrt{1-\rho}\approx${f2(geo.stable)}`)} leapfrog is unstable on this target. NUTS chooses the path length automatically.`],
 'Stability limit: for a Gaussian with unit mass, leapfrog is stable when ε is below twice the smallest standard deviation, here 2·sqrt(1−ρ). The lab keeps L fixed at 9; NUTS (Hoffman & Gelman) removes the need to choose L.',source('N11','HG14'),'HMC: follow the gradient, then correct');
liveSlide('hmc','05 · hamiltonian monte carlo','HMC: follow the gradient, then correct.',`Step and watch each trajectory, then push ${I(R`\varepsilon`)} past ${f2(geo.stable)}.`,'Hamiltonian Monte Carlo on the correlated Gaussian','Each move costs nine gradient evaluations and one density. The status line warns when ε exceeds the leapfrog stability limit.',source('N11','HG14'));

/* ---------------- 13–14 · slice ---------------- */
eqTwo('slice','06 · slice sampling','Slice: sample uniformly under the curve.','An auxiliary height turns density evaluation into a uniform draw.',
 ['TWO UNIFORM DRAWS',M(R`u\sim\mathcal U\big(0,\,\tilde\pi(x)\big),\qquad x'\sim\mathcal U\big\{x:\tilde\pi(x)\ge u\big\}`),130],
 ['STEP OUT, THEN SHRINK',`Place a width-${I('w')} bracket at random around ${I('x')}; step it out until both ends leave the slice, with a budget of ${I(`m=${S.CHAIN.slice.m}`)} widths split at random between the sides.<br><br>Draw uniformly inside; after each miss, shrink the bracket toward ${I('x')}. The lab updates one coordinate at a time, ${I(`w=${S.CHAIN.slice.w}`)}.`],
 ['WHAT TO WATCH NEXT',`The green band is the exact slice; the grey bracket is what stepping out found; rust crosses are misses.<br><br>Nothing is rejected: misses cost density evaluations instead. Try ${I('w=0.3')} (more stepping out) and ${I('w=4')} (more shrinking). The move adapts; the zig-zag remains.`],
 'Stepping out follows Neal (2003), Fig. 3: J = floor(mV), K = (m − 1) − J. The random split keeps the update reversible when the budget binds. Shrinkage needs no extra acceptance test with stepping out.',source('N03'),'Slice: sample uniformly under the curve');
liveSlide('slice','06 · slice sampling','Slice: sample uniformly under the curve.',`Step and read each bracket and miss; then compare ${I('w=0.3')} with ${I('w=4')}.`,'Coordinate-wise slice sampling on the correlated Gaussian','Work is counted in density evaluations, including those spent stepping out and shrinking.',source('N03'));

/* ---------------- 15 · scorecard ---------------- */
const L=long;
add('scorecard','07 · four chains, measured','High acceptance is not fast mixing.',`Seeded runs at ${I(`\\rho=${rho}`)}: ${LONG.toLocaleString('en-US')} moves after ${BURN.toLocaleString('en-US')} burn-in. Effective sample size of ${I('x_1')} by Geyer’s initial monotone sequence.`,[
 nativeTable('score',72,186,1136,262,['Sampler','One move','Work per move','Acceptance','ESS per 1,000 moves','ESS per 1,000 evaluations'],[
  ['Gibbs','one coordinate','1 conditional draw',pct(L.gibbs.acc),Math.round(L.gibbs.essPerK),Math.round(L.gibbs.essPerKEval)],
  ['Metropolis–Hastings','both coordinates','1 density',pct(L.mh.acc),Math.round(L.mh.essPerK),Math.round(L.mh.essPerKEval)],
  ['HMC (ε = 0.16, L = 9)','both coordinates','9 gradients + 1 density',pct(L.hmc.acc),Math.round(L.hmc.essPerK),Math.round(L.hmc.essPerKEval)],
  ['Slice (w = 1.7)','one coordinate',`${f1(L.slice.evalsPerMove)} densities (average)`,pct(L.slice.acc),Math.round(L.slice.essPerK),Math.round(L.slice.essPerKEval)]
 ].map(r=>r.map(String)),{columns:[1.35,1.05,1.45,.8,1,1.1],fontSize:16,padY:12}),
 ...panel('score-move',72,466,558,167,'PER MOVE',`HMC mixes about ${Math.round(L.hmc.essPerK/L.gibbs.essPerK)}× faster than Gibbs and ${Math.round(L.hmc.essPerK/L.mh.essPerK)}× faster than random-walk Metropolis. Gibbs and slice accept everything and still zig-zag.`,{fill:C.soft,size:18}),
 ...panel('score-work',650,466,558,167,'PER EVALUATION · AND IN HIGH DIMENSION',`In 2-D a gradient costs about one density, and Gibbs leads per evaluation. The gradient pays off as dimension grows: random-walk Metropolis work scales as ${I('d^2')}, HMC as ${I('d^{5/4}.')}`,{size:18,accent:C.rust})
],'The lab shows a lag-1 ESS proxy on a short chain; this table uses long runs and Geyer’s estimator instead, so the numbers differ. Evaluation counts treat a conditional draw, a density and a gradient as one unit each, a fair approximation for this 2-D Gaussian only. The scaling statement is Neal (2011), section 4.4, for targets of independent Gaussian-like components.',source('N11','V21'),'High acceptance is not fast mixing');

/* ---------------- 16–17 · rejection ---------------- */
eqTwo('rejection','08 · rejection sampling',`Rejection: keep what lands under ${I(R`\pi`)}.`,'Exact, independent draws—as long as the envelope really covers the target.',
 ['KEEP A DRAW IF IT LANDS UNDER THE TARGET',M(R`x\sim q,\quad u\sim\mathcal U(0,1),\quad\text{keep }x\text{ if }u\le\frac{\tilde\pi(x)}{M\,q(x)},\qquad\text{valid when }\tilde\pi(x)\le M\,q(x)\;\;\forall x`),130],
 ['THE POPULATION TARGET',`${M(R`\pi(x)=0.55\,\mathcal N(-1.35,0.6^2)+0.45\,\mathcal N(1.15,0.45^2)`)}Proposal ${I(R`q=\mathcal N(0,1.5^2)`)}. The tightest constant is ${I(R`M=\sup\pi/q=${f2(rej.sup)}`)}, so the acceptance rate is ${I(R`1/M=${pct(1/rej.sup).replace('%','\\%')}.`)}`],
 ['WHAT TO WATCH NEXT',`Blue points are kept; their histogram matches ${I(R`\pi`)}. The lab starts after 1,000 proposals: ${pct(rej.lab.acceptance)} kept.<br><br>Loosen the envelope: acceptance falls as ${I('1/M')}, exactness stays. Push the slack below 1: the envelope dips under ${I(R`\pi`)} (shaded) and the kept draws follow ${I(R`\min(\pi,Mq)`)} instead.`],
 'With a normalized target the acceptance probability is exactly 1/M; with an unnormalized one it is Z/M. The narrowest proposal allowed in the lab is s = 0.8: below the widest component standard deviation 0.6 no finite envelope exists.',source('O13'),`Rejection: keep what lands under ${I(R`\pi`)}`);
liveSlide('rejection','08 · rejection sampling',`Rejection: keep what lands under ${I(R`\pi`)}.`,`Loosen the envelope, then push its slack below 1 and compare the histogram with ${I(R`\pi`)}.`,'Rejection sampling on the two-bump target','Draw 25 adds 25 proposals. The shaded region appears only when the envelope is invalid.',source('O13'),'Rejection: keep what lands under π.');

/* ---------------- 18–19 · importance ---------------- */
eqTwo('importance','09 · importance sampling','Importance sampling: reweight every draw.','Weights repair the mismatch between an easy proposal and the target.',
 ['WEIGHT, SELF-NORMALIZE, CHECK THE BALANCE',M(R`w_n=\frac{\tilde\pi(x_n)}{q(x_n)},\;\;x_n\sim q,\qquad \hat\mu=\frac{\sum_n w_n f(x_n)}{\sum_n w_n},\qquad \mathrm{ESS}=\frac{\left(\sum_n w_n\right)^2}{\sum_n w_n^2}`),140],
 ['KEY IDEA',`Every draw is kept. Self-normalizing removes ${I('Z')}, at the cost of a small finite-sample bias.<br><br>For ${I(R`q=\mathcal N(0,1.5^2)`)} the large-sample ESS fraction is ${I(R`1/\mathbb E_q[w^2]=${pct(imp.frac).replace('%','\\%')}`)}: a weight-degeneracy diagnostic, not a general function-specific standard-error guarantee.`],
 ['WHAT TO WATCH NEXT',`After 1,000 draws: ESS ${Math.round(imp.lab.ess)}, ${I(R`\hat\mu=${neg(imp.lab.est.toFixed(3))}\pm${imp.lab.se.toFixed(3)}`)} vs ${neg(S.TRUE_MEAN.toFixed(3))}—${f1(imp.z)} standard errors off, a ${(100*imp.p).toFixed(1)}% event (${pct(imp.outside)} of 400 other seeds land beyond 2). Press New seed.<br><br>Narrow ${I('q')} below ${I(R`s=0.6`)}: weights become unbounded; ${I(R`\mathbb E_q[w^2]\approx${(imp.narrow/1e5).toFixed(1)}\times10^5`)} at ${I('s=0.5')}, infinite below ${I(R`0.6/\sqrt2.`)}`],
 'The ± value is the delta-method standard error of the self-normalized estimate (Owen, eq. 9.9). The default seed is kept from the original page; its unlucky first 1,000 draws are left in on purpose. Kish ESS is a weight-balance diagnostic, not a guarantee for every function f (Owen, eq. 9.13–9.14). The proposal needs tails at least as heavy as the target’s.',source('O13'),'Importance sampling: reweight every draw',{size:18});
liveSlide('importance','09 · importance sampling','Importance sampling: reweight every draw.',`Narrow the proposal below ${I('s=0.6')}, then shift its mean; watch the stems and the ESS.`,'Importance sampling on the two-bump target','Draw 40 adds 40 weighted draws. Stem height is relative to the largest weight.',source('O13'));

/* ---------------- 20–21 · SMC ---------------- */
eqTwo('smc','10 · sequential monte carlo','SMC: propagate, weight, resample.','A weighted particle population follows a posterior that changes with every observation.',
 ['BOOTSTRAP PARTICLE FILTER',M(R`x_t^{(i)}\sim p\big(x_t\mid x_{t-1}^{(i)}\big),\qquad w_t^{(i)}\propto w_{t-1}^{(i)}\,p\big(y_t\mid x_t^{(i)}\big),\qquad\text{resample when }\mathrm{ESS}_t\lt N/2`),130],
 ['THIS MODEL HAS AN EXACT ANSWER',`${M(R`x_t=0.92\,x_{t-1}+0.45\,\varepsilon_t,\qquad y_t=x_t+0.5\,\nu_t`)}Linear and Gaussian, so the Kalman filter gives the exact posterior (steady-state sd ${f2(smc.steady)}). The particle filter can be checked against it: that gap is pure Monte Carlo error.`],
 ['WHAT TO WATCH NEXT',`Rust dashes: exact Kalman mean. Blue: the ${S.SSM.N}-particle mean. Default run: ${smc.half.resamples} resamplings in 60 steps, error vs the exact mean ${f2(smc.half.mcError)}.<br><br><b>Never resample</b>: ESS collapses to ${f1(smc.never.ess)} and the error grows to ${f2(smc.never.mcError)}. <b>200 particles</b>: ${f2(smc.big.mcError)}.`],
 `Systematic resampling. The world (hidden state and observations) has its own seed, so changing N or the resampling rule changes only the filter. Resampling adds noise at the current step but prevents weight degeneracy later; ESS-triggered resampling is a common compromise. Each resampling keeps about ${pct(smc.half.distinct)} distinct parents here; process noise restores diversity. Tracking RMSE ${f2(smc.half.rmse)} vs ${f2(smc.half.exactRmse)} for the exact filter on this run.`,source('GSS93','DJ11'),'SMC: propagate, weight, resample');
liveSlide('smc','10 · sequential monte carlo','SMC: propagate, weight, resample.',`Run to ${I('t=60')}, then switch to “never” resample; vary ${I('N')} and compare with the exact mean.`,'Bootstrap particle filter on a linear-Gaussian model','Advance t processes one observation. Green ESS bars mark steps that resampled.',source('GSS93','DJ11'));

/* ---------------- 22–24 · compare, choose, diagnose ---------------- */
add('compare','11 · compare','Seven samplers, side by side.','What each needs, what it returns, and how it typically fails.',[
 nativeTable('compare-table',72,186,1136,440,['Method','Needs','Returns','Signature strength','Typical failure'],[
  ['Gibbs','Sampleable full conditionals','Correlated chain','No rejection, no step size','Slow zig-zag under correlation'],
  ['Metropolis–Hastings','Density ratios','Correlated chain','Works with almost any proposal','Random-walk crawl or many rejections'],
  ['HMC / NUTS','Density and gradients','Correlated chain','Long, informed moves in high dimension','Bad geometry, divergences'],
  ['Slice','Density evaluations','Correlated chain','Adapts its own step scale','Costly stepping out and shrinking'],
  ['Rejection','An envelope M·q ≥ π','Exact independent draws','Exact and simple','Acceptance vanishes with a loose envelope or dimension'],
  ['Importance','Proposal density and target','Weighted independent draws','Parallel and reusable','Weight collapse from light-tailed q'],
  ['SMC','Transition, likelihood, resampling','Weighted particle population','Online and multimodal','Degeneracy, particle impoverishment']
 ],{columns:[1.15,1.3,1.2,1.35,1.5],fontSize:15,padY:11})
],'Correlated chains need burn-in and mixing diagnostics. Exact draws need none. Weighted draws and particles need weight diagnostics.',source('O13','N11','DJ11'),'Seven samplers, side by side');

add('choose','11 · choose','Choose by what your problem gives you.','Start from the access you have to the target, not from which sampler sounds sophisticated.',[
 nativeTable('choose-table',72,186,1136,384,['If you have…','Try first','Why'],[
  ['Closed-form conditionals (e.g. a conjugate hierarchical model)','Gibbs','Exact conditionals need no tuning; block or reparameterize if the chain zig-zags.'],
  ['A smooth, differentiable posterior in many dimensions','NUTS / HMC','Gradients avoid random-walk behaviour; NUTS sets the path length.'],
  ['Only an unnormalized density','Metropolis–Hastings or slice','Needs only density evaluations; improve the proposal from structure.'],
  ['A hidden state with observations arriving over time','Sequential Monte Carlo','Particles propagate, reweight and resample with each observation.'],
  ['One expectation and a proposal with heavier tails than the target','Importance sampling','Independent weighted draws parallelize; monitor the weight ESS.']
 ],{columns:[1.6,.95,2],fontSize:16,padY:11}),
 ...callout('choose-note',72,586,1136,50,`Whatever you choose, verify it: several chains, trace plots, effective sample size, ${I(R`\widehat R`)}—and weight balance for weighted methods.`,{fill:C.cool,stroke:C.blue,size:17})
],'These are starting points, not rules. Many practical samplers combine them: Metropolis-within-Gibbs, HMC inside SMC samplers, importance-weighted resampling.',source('HG14','V21'),'Choose by what your problem gives you');

four('diagnostics','12 · diagnose','Never stop at “it ran”.','Four checks that matter after sampling.',[
 ['01 · BURN-IN','Early draws remember the arbitrary start. Discarding them removes initialization bias—but cannot repair a chain that never mixed.'],
 ['02 · MIXING','Run several chains from dispersed starts. Slow wandering, sticky runs or chains that disagree mean the sample is incomplete.'],
 ['03 · EFFECTIVE SAMPLE SIZE',`Converts correlated or weighted draws into an approximate count of independent ones. Vehtari et al. recommend ESS above 400 for reliable summaries.`],
 ['04 · CONVERGENCE',`Rank-normalized ${I(R`\widehat R`)} below 1.01 supports—but does not prove—that the chains reached the same target.`,C.warm,C.rust]
],'The thresholds are those recommended by Vehtari et al. (2021): rank-normalized split-R-hat below 1.01 and bulk and tail ESS above 400, using at least four chains. For importance sampling and particle filters, also watch the largest normalized weight.',source('V21'),'Never stop at “it ran”');

/* ---------------- 25–27 · connections, takeaways, references ---------------- */
add('connections','13 · on this site','Where these samplers show up.','Three other notes use the same ideas; the links open in this tab.',[
 ...[['EXTENDED-OBJECT TRACKING','/eo-mtt/','MCMC over partitions','The partition toolbox runs a Metropolis–Hastings-style split/merge chain and a Gibbs-style sampler over measurement partitions. Mixing and multiple-chain checks apply unchanged.'],
     ['BP VS PMBM','/bp-vs-pmbm/','Weighted hypotheses','PMBM keeps a weighted set of global association hypotheses; practical implementations generate high-weight children by ranked assignment or sampling. These are hypothesis weights, not importance weights.'],
     ['FRAME REGISTRATION','/frame-registration/','Annealing as continuation','The annealed soft-target heuristic widens, then narrows, a Gaussian kernel—an optimization cousin of the tempering sequences used by SMC samplers.']
 ].flatMap(([label,href,t,body],i)=>[...panel('conn-'+i,72+i*382,186,358,447,label,`<b>${t}</b><br><br>${body}`,{fill:i===0?C.soft:C.panel,size:19}),text('conn-link-'+i,94+i*382,585,314,26,`<a href="${href}">Open ${href} →</a>`,{fontSize:15,fontWeight:700,color:C.green})])
],'These are connections of technique, not claims that the other pages implement the samplers in this deck.','SAMPLING · CONNECTIONS','Where these samplers show up');

four('takeaways','14 · takeaways','What to keep.','Four distinctions to carry into any sampling problem.',[
 ['1 · MATCH THE METHOD TO YOUR ACCESS','Conditionals → Gibbs. Gradients → HMC/NUTS. Densities only → Metropolis–Hastings or slice. A good proposal → importance sampling. A sequence → SMC.'],
 ['2 · ACCEPTANCE IS NOT EFFICIENCY',`Gibbs and slice accept every move and still zig-zag. Compare effective sample size per unit of work.`],
 ['3 · WEIGHTS NEED TAILS','An importance proposal lighter-tailed than the target gives unbounded or infinite-variance weights. Watch the ESS and the largest weight.'],
 ['4 · CHECK AGAINST A KNOWN ANSWER','Test a sampler where the answer is exact—a Gaussian, a mixture, a Kalman filter—before trusting it where it is not.',C.warm,C.rust]
],'Each takeaway maps to a lab: the scorecard for point 2, the importance lab for point 3, and the exact Kalman reference in the particle-filter lab for point 4.','SAMPLING · SUMMARY','What to keep');

add('references','Reference desk','Read the methods. Know the simplifications.','Primary sources for the algorithms; the labs are independent teaching implementations.',[
 ...Object.entries(refs).map(([k,r],i)=>text('ref-'+k,72+(i%2)*578,184+Math.floor(i/2)*62,548,56,`<b>${r[0]}</b> · ${r[2]} ↗`,{fontSize:13.5,lineHeight:1.3,link:r[1]})),
 ...panel('scope',72,554,1136,88,'SCOPE',`Implemented: Gibbs, random-walk MH, fixed-length HMC, stepping-out slice, rejection, self-normalized importance sampling and a bootstrap particle filter with an exact Kalman reference. Not implemented: NUTS, adaptation, multiple-chain ${I(R`\widehat R`)}.`,{size:14,top:38})
],'All labs use fixed seeds; New seed draws a fresh one. The deck was adapted from the original Sampling Playground; its chain samplers keep that page’s constants, and the slice sampler now follows Neal’s randomized stepping-out budget.',source('SP'),'References');

/* ---------------- assembly ---------------- */
const doc={format:'bento/slides',version:1,docId:'sampling-bento',title:'See how samplers think',readonly:true,meta:{author:'Bai Liping',subject:'Monte Carlo sampling methods with seven interactive labs',company:'bailiping.com'},size:{width:1280,height:720},theme:{background:C.paper,color:C.ink,accent:C.green,fontFamily:sans},slides};
const all=doc.slides;
const entries=Object.entries(toc).filter(([id])=>id!=='overview');
const half=Math.ceil(entries.length/2);
const contents=all.find(s=>s.id==='contents');
contents.elements.splice(3,0,...entries.flatMap(([id,title],k)=>{
 const col=k<half?0:1,row=col?k-half:k,x=72+col*578,y=180+row*46;
 const page=String(all.findIndex(s=>s.id===id)+1).padStart(2,'0');
 const hasLab=all.some(s=>s.id===id+'-live');
 return [box('toc-rule-'+k,x,y+42,558,1,C.rule,C.rule,{radius:0}),
  text('toc-'+k,x,y+9,558,32,`<span style="font-family:${mono};font-size:13px;color:${C.rust}">${page}</span>&nbsp;&nbsp;&nbsp;${title}${hasLab?` <span style="font-family:${mono};font-size:11px;color:${C.green}">· LIVE LAB</span>`:''}`,{fontFamily:serif,fontSize:17,link:id})];
}));
all.find(s=>s.id==='overview').elements.find(e=>e.id==='cover-boundary').html=`${all.length} SLIDES · ${live.length} LIVE LABS · SEEDED BROWSER SAMPLERS`;

export const deck=doc;
export const inlineLiveMap=live.map(({introSlide,slide,demo,title})=>({introSlide,slide,slideIndex:all.findIndex(s=>s.id===slide),inline:true,layout:'region',bounds,src:`./live/?demo=${demo}&embed=region`,source:`./live/?demo=${demo}`,title,sandbox:'allow-scripts',hideSource:true,readyMessage:true,unloadWhenHidden:true}));
