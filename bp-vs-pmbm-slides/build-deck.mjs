// Authoring source. Rebuild content and local figures; preserve the checked-in Bento runtime.
import fs from 'node:fs';
import Model from '../bp-vs-pmbm/association-model.js';
import {C,buildFigures} from './figures.mjs';
const FONT='Arial, Helvetica, sans-serif', t=String.raw;
const inline=s=>`<span class="math-tex math-inline">\\(${s}\\)</span>`;
const display=s=>`<span class="math-tex math-display">\\[${s}\\]</span>`;
const T=(id,x,y,w,h,html,size=24,color=C.ink,weight=400,align='left')=>({id,type:'text',x,y,w,h,html,fontSize:size,fontFamily:FONT,fontWeight:weight,color,align,valign:'top',lineHeight:1.25,rotation:0,opacity:1});
const R=(id,x,y,w,h,fill=C.wash,stroke='none',radius=8)=>({id,type:'shape',shape:'rect',x,y,w,h,fill,stroke,strokeWidth:stroke==='none'?0:1,radius,rotation:0,opacity:1});
const I=(id,name,x,y,w,h,alt)=>({id,type:'image',src:`./assets/${name}?v=20260918-redesign`,x,y,w,h,alt,fit:'contain',rotation:0,opacity:1});
const E=(id,x,y,w,h,latex,size=28,color=C.ink)=>T(id,x,y,w,h,display(latex),size,color);
const link=(id,x,y,w,label,href,size=18)=>T(id,x,y,w,Math.max(30,size*1.4),`<a class="lesson-link" href="${href}">${label}</a>`,size,C.green,600);
const refs={bp:['Williams & Lau · Association BP (2014)','https://arxiv.org/abs/1209.6299'],pm:['García-Fernández et al. · PMBM (2018)','https://arxiv.org/abs/1703.04264'],mb:['Williams · Marginal multi-Bernoulli filters (2015)','https://arxiv.org/abs/1203.2995']};
function slide(id,kicker,title,body,notes,ref){return {id,background:'#ffffff',transition:'none',notes,elements:[T('kicker',72,35,730,22,kicker.toUpperCase(),12,C.green,700),link('return',864,35,344,'← All topics','/',12),T('slide-title',72,73,1136,82,title,36,C.ink,700),R('rule',72,165,1136,1,C.line,'none',0),...body,...(ref?[link('reference',72,650,1080,refs[ref][0],refs[ref][1],11)]:[]),T('footer',72,682,1000,18,'Belief-Propagation MTT · Liping Bai',11,C.muted),T('page',1090,682,118,18,'{{page:2}} / {{pages:2}}',11,C.muted,400,'right')]};}
const bounds={x:72,y:205,width:1136,height:435};
function live(id,intro,kicker,title,prompt,mode){const s=slide(id,kicker,title,[T('prompt',72,175,1136,26,prompt,17,C.muted),I('fallback',`${mode}-fallback.png`,bounds.x,bounds.y,bounds.width,bounds.height,`Static initial state of the ${mode} experiment`),R('live-demo-mount',bounds.x,bounds.y,bounds.width,bounds.height,'transparent')],`Apply the immediately preceding explanation. ${prompt} The deterministic one-scan benchmark uses three certain existing point targets, Gaussian predicted measurements, Poisson clutter, and no undetected PPP. These are association calculations, not full tracking results. Page Up / Down navigate from controls; Escape returns focus to the deck.`,'bp');s.demo={introSlide:intro,slide:id,inline:true,layout:'region',bounds,src:`./live/?demo=${mode}&embed=region&v=20260918-redesign`,source:`./live/?demo=${mode}`,title,sandbox:'allow-scripts',hideSource:true,readyMessage:true,unloadWhenHidden:true};return s;}
const {L}=Model.buildWeights(Model.DEFAULT),events=Model.enumerate(L),exact=Model.eventMarginals(events,3,4).marginals,bp=Model.bp(L,{history:false}),error=100*Model.maxDifference(bp.marginals,exact);
if(!bp.converged)throw Error('Benchmark BP did not converge');buildFigures(L);
const slides=[];
slides.push(slide('s-cover','Multi-target tracking / Point targets','Belief-Propagation MTT',[
 T('subtitle',72,177,1050,40,'Scalable data association, and its connection to PMBM',26,C.muted),
 T('cover-claim',72,284,490,110,'Track uncertainty includes<br>uncertain assignments.',38,C.ink,700),
 T('cover-copy',72,413,475,94,'BP estimates the probability of each match by passing local messages.',25,C.muted),
 I('cover-figure','cover.svg',600,223,608,382,'Two predicted tracks compete for detections'),
 T('roadmap',72,595,1100,42,'01  Association model     →     02  BP inference     →     03  PMBM connection',20,C.green,600)
], 'Introduce belief propagation as a method for association inference within a tracker. Establish the point-target setting: at most one detection per target in a scan. The presentation later connects association marginals to a state update and to PMBM. BP is an inference method; PMBM is a posterior family.'));
slides.push(slide('s-question','01 / The association problem','A measurement can belong to only one target.',[
 T('intro',72,191,1136,62,'Nearby predictions can support several matches. The assignments must agree across all targets.',24,C.muted),
 I('assignments','constraints.svg',72,277,1136,314,'Two compatible matchings and an incompatible double claim'),
 T('miss-and-new',72,608,1136,31,'A track may be missed; an unassigned detection may be clutter or a newly detected target.',19,C.muted),
], 'For standard point-target measurements, each target generates at most one measurement and each measurement originates from at most one target. A missed detection is permitted; an unassigned measurement may be clutter or a newly detected target. The figure isolates competition between two existing tracks.','bp'));
slides.push(slide('s-weights','01 / From geometry to evidence','Score each possible match, including a miss.',[
 T('matrix-head',72,205,536,32,'The shared example: 3 tracks, 4 detections',23,C.ink,700),
 {id:'weights',type:'table',x:72,y:263,w:536,h:236,rotation:0,opacity:1,header:true,columns:Array(6).fill({w:1}),rows:[{cells:['Track','Miss','z1','z2','z3','z4'].map(html=>({html}))},...L.map((row,i)=>({cells:[{html:`T${i+1}`,bold:true},...row.map(v=>({html:v.toFixed(2)}))]}))],style:{headerBg:C.wash,headerColor:C.ink,borderColor:C.line,borderWidth:1,cellPadX:10,cellPadY:12,fontSize:20,color:C.ink}},
 T('matrix-note',72,519,528,66,'A zero removes a candidate pair.<br>These weights are not probabilities.',21,C.muted),
 R('formula-panel',664,204,544,385),
 T('miss-head',692,229,480,26,'MISSED DETECTION',13,C.green,700),E('miss',692,264,480,54,t`\ell_{i0}=1-P_D`,29),
 T('match-head',692,339,480,26,'ASSIGNED DETECTION',13,C.green,700),E('match',692,372,480,96,t`\ell_{ij}=\frac{P_D\,\mathcal N(z_j;\hat z_i,S_i)}{c(z_j)}`,29),
 T('formula-note',692,481,480,80,'More plausible detections get more weight; stronger clutter reduces their evidence.',21,C.muted),
], 'Assumptions for all three experiments: one scan, certain existing tracks, constant detection probability, Gaussian predicted-measurement densities, homogeneous Poisson clutter, and zero undetected-target PPP. Unassigned measurements have normalized weight one. The optional 99% gate sets pair weights to zero without adjusting the miss probability; it is a numerical truncation. The next slide changes geometry and these weights.','bp'));
slides.push(live('s-weights-live','s-weights','Explore / Association weights','Move a detection. Watch the candidate matches change.','Compare overlapping and separated tracks, then drag a detection across a gate.','assignment'));
slides.push(slide('s-constraint','02 / The factor graph','Local consistency factors couple the assignments.',[
 I('graph','factor-graph.svg',72,202,660,390,'Multi-track association graph with unary weights and pairwise consistency factors'),
 T('a-definition',778,211,430,64,inline(t`a_i=j`)+': track '+inline(t`i`)+ ' chooses detection '+inline(t`j`)+'.',22),
 T('b-definition',778,289,430,64,inline(t`b_j=i`)+': detection '+inline(t`j`)+ ' chooses track '+inline(t`i`)+'.',22),
 R('consistency',770,377,438,106),T('consistency-title',790,392,398,24,'CONSISTENCY RULE',12,C.green,700),E('constraint',790,425,398,40,t`(a_i=j)\Leftrightarrow(b_j=i)`,26),
 T('graph-copy',778,506,420,85,'Each factor is one when these statements agree, and zero otherwise.',22,C.muted),
 E('unary-definition',72,603,660,39,t`w_i(a_i)=\ell_{i,a_i}`,22),
 T('zero-state',778,609,430,31,'0 means missed / unassigned.',18,C.muted),
], 'Unary weight w_i(a_i)=ell_i,a_i includes the missed-detection state a_i=0. Measurement state b_j=0 means unassigned. Each pairwise psi_ij equals one when the Boolean statements a_i=j and b_j=i agree, and zero otherwise. The diagram shows active gated pairs; states for excluded pairs are disallowed, so omitted factors are inert. The isolated fourth measurement can only be unassigned in this example. Factor boxes are dependencies, not additional random variables.','bp'));
slides.push(slide('s-bp-messages','02 / How BP works','Each message accounts for the other possible matches.',[
 R('mu-panel',72,207,544,215),R('nu-panel',664,207,544,215),
 T('mu-label',100,229,488,28,'TRACK → MEASUREMENT',14,C.green,700),E('mu',96,275,496,88,t`\mu_{i\to j}=\frac{\ell_{ij}}{\ell_{i0}+\sum_{k\ne j}\ell_{ik}\nu_{k\to i}}`,28),
 T('mu-meaning',100,377,488,30,'How strong are this track’s other options?',19,C.muted),
 T('nu-label',692,229,488,28,'MEASUREMENT → TRACK',14,C.blue,700),E('nu',688,275,496,88,t`\nu_{j\to i}=\frac{1}{1+\sum_{h\ne i}\mu_{h\to j}}`,28),
 T('nu-meaning',692,377,488,30,'How much competition comes from other tracks?',19,C.muted),
 T('belief-label',72,467,370,32,'Normalize to obtain track beliefs',22,C.ink,700),E('belief',440,448,768,98,t`\beta_{ij}\propto\begin{cases}\ell_{i0},&j=0,\\\ell_{ij}\nu_{j\to i},&j>0.\end{cases}`,27),
 T('repeat',72,569,1136,50,'Initialize return messages to one. Alternate the two updates until they stabilize.',22,C.muted)
], 'Williams–Lau scalar association BP. Each denominator excludes the recipient edge; messages are ratios, not probabilities. Normalize beta over the miss and all candidate detections for each track. Under the positive miss/unassigned assumptions of this construction, association BP converges; on loopy graphs the converged beliefs are generally approximate. This is not a convergence theorem for arbitrary loopy BP.','bp'));
slides.push(live('s-bp-messages-live','s-bp-messages','Explore / Message passing','Follow the messages into association probabilities.','Step through one update, then compare the final BP beliefs with exact enumeration.','bp'));
slides.push(slide('s-bp-marginals','02 / From association to tracking','Association probabilities weight the state update.',[
 ...[['Predict','Carry each track’s state distribution into the new scan.'],['Associate','Use BP to estimate the probability of each candidate match.'],['Update','Combine the miss and measurement-conditioned state densities.']].flatMap(([head,body],i)=>[R('step-rule-'+i,72+i*390,219,4,190,i===1?C.green:C.line),T('step-head-'+i,94+i*390,218,335,37,head,28,i===1?C.green:C.ink,700),T('step-body-'+i,94+i*390,278,323,116,body,23,C.muted)]),
 E('update',110,426,1050,116,t`f_i^+(x)=\sum_{j=0}^{m}\beta_{ij}\,f_i(x\mid a_i=j,Z)`,32),
 T('update-note',110,553,720,56,'The example assumes certain existing tracks. A complete tracker also handles existence and birth.',20,C.muted),
 T('bp-error-number',930,540,250,42,error.toFixed(2)+' pp',31,C.green,700,'center'),T('bp-error-caption',908,589,300,37,'BP–exact gap in this example',15,C.muted,400,'center')
], 'For certain tracks, the state update is a mixture of the missed-detection and measurement-conditioned densities with association-marginal weights. Full marginal multi-target trackers also propagate existence and new-target components. A single Gaussian approximation to the mixture is an additional choice, not part of the BP association algorithm. The displayed gap is the maximum absolute marginal difference for this one synthetic scene, after the actual stopping criterion; it is not a full tracking benchmark.','mb'));
slides.push(slide('s-joint-events','03 / Joint hypotheses','An exact marginal sums over compatible assignments.',[
 ...events.slice(0,3).flatMap((ev,k)=>{const y=218+k*107;return [R('event-'+k,72,y,708,86,k===0?'#fdf3e8':C.wash),T('rank-'+k,94,y+20,45,32,'0'+(k+1),21,C.orange,700),T('story-'+k,159,y+21,455,34,ev.a.map((j,i)=>`T${i+1} → ${j<0?'miss':'z'+(j+1)}`).join(' &nbsp; · &nbsp; '),21),T('prob-'+k,635,y+20,118,34,(ev.p*100).toFixed(1)+'%',24,C.orange,700,'right')];}),
 T('event-count',72,558,708,56,`${events.length} positive-weight assignments in this example.<br>Each row is one compatible joint decision.`,21,C.muted),
 R('exact-panel',826,218,382,337),T('exact-label',850,245,334,28,'SUM FIRST, THEN NORMALIZE',13,C.orange,700),E('event-weight',846,301,342,73,t`w(a)\propto\prod_i\ell_{i,a_i}`,28),E('exact-marginal',840,409,352,89,t`\beta_{ij}=\sum_{a:\,a_i=j}P(a\mid Z)`,25),
], 'The normalized posterior over legal assignments is proportional to the product of selected local weights. Misses are allowed; a nonzero detection index can appear at most once. Exact enumeration is feasible here because the problem is tiny. Practical hypothesis trackers use gating and selected assignments, then prune. The next slide isolates how retaining only the top k events changes the marginals.','bp'));
slides.push(live('s-pruning-live','s-joint-events','Explore / Retaining hypotheses','Keep fewer assignments. Measure what is discarded.','Change the number retained and compare the resulting marginals with the full sum.','hypotheses'));
slides.push(slide('s-scaling','03 / Why scale matters','BP avoids listing the global assignments.',[
 I('growth','scaling.svg',72,211,741,426,'Logarithmic comparison of pair weights and compatible assignment counts'),
 T('seven',873,212,335,30,'7 TRACKS · 7 DETECTIONS',14,C.muted,700),T('edge-count',873,272,335,65,'49',52,C.green,700),T('edge-label',873,340,335,56,'pair weights<br>per dense BP sweep',22,C.muted),
 T('event-count',873,436,335,65,'130,922',46,C.orange,700),T('event-label',873,505,335,61,'compatible assignments<br>with misses allowed',22,C.muted),
], 'Counts for complete gating: number of partial injections is sum_{r=0}^{min(n,m)} choose(n,r) choose(m,r) r!. Dense Williams–Lau updates cost O(nm) per sweep using aggregate sums, or O(Tnm) for T sweeps. The plot compares combinatorial objects, not measured runtime. Production PMBM implementations use ranked assignment and pruning rather than enumerating every event.','bp'));
slides.push(slide('s-dependence','03 / What marginals cannot retain','The marginals alone do not encode joint dependence.',[
 R('joint',72,212,544,313),T('joint-head',100,239,488,34,'Two legal assignments',25,C.orange,700),E('joint-one',96,303,496,58,t`P(a_1=1,a_2=2)=\tfrac12`,28),E('joint-two',96,387,496,58,t`P(a_1=2,a_2=1)=\tfrac12`,28),T('joint-foot',100,467,488,38,'Each marginal is 50% / 50%.',21,C.muted),
 T('product-head',672,220,536,36,'Multiplying those marginals gives',24,C.ink,700),
 ...[[1,1,false],[1,2,true],[2,1,true],[2,2,false]].flatMap(([a,b,legal],k)=>{const x=672+(k%2)*280,y=280+Math.floor(k/2)*122;return [R('outcome-'+k,x,y,256,103,legal?'#edf7f3':'#fdf3e8'),E('outcome-math-'+k,x+12,y+8,232,44,t`(${a},${b})\quad\tfrac14`,24),T('outcome-status-'+k,x+12,y+61,232,27,legal?'Compatible':'Double claim',17,legal?C.green:C.orange,600,'center')];}),
 T('distinction',72,564,1136,57,'The loss here comes from assuming independence—even when the marginals are exact.',24,C.ink,600)
], 'This illustrative distribution has two certain tracks, two detections, and no misses. It is a separate discrete example, not the positive-miss BP experiment. Marginalization, product-of-marginals projection, and Gaussian moment matching are distinct operations. Marginals do not specify a joint association distribution; BP does not instruct us to sample them independently.','mb'));
slides.push(slide('s-pmbm','04 / The PMBM connection','PMBM represents undetected targets and joint alternatives.',[
 R('ppp',72,218,364,317),T('ppp-label',100,244,310,28,'UNDETECTED TARGETS',14,C.orange,700),T('ppp-name',100,298,310,45,'Poisson process',28,C.ink,700),E('ppp-math',100,369,310,65,t`\lambda^u(x)`,34),T('ppp-copy',100,453,306,65,'Represents targets that have not yet been detected.',21,C.muted),
 T('plus',451,341,70,70,'+',42,C.muted,400,'center'),
 R('mbm',534,218,674,317),T('mbm-label',562,244,618,28,'DETECTED TARGETS',14,C.green,700),T('mbm-name',562,287,618,45,'Multi-Bernoulli mixture',28,C.ink,700),
 ...[0,1,2].flatMap(k=>[R('hyp-'+k,562,353+k*50,618,39,'#fff'),E('hyp-math-'+k,571,355+k*50,100,34,`w^{${k+1}}`,21,C.orange),T('hyp-text-'+k,699,360+k*50,462,27,`Hypothesis ${k+1}: a compatible set of tracks`,18)]),
 T('pmbm-note',72,563,1136,57,'Each Bernoulli has an existence probability and a state density. The mixture retains alternatives.',23,C.muted)
], 'PMBM is a posterior family, not a competing message-passing algorithm. Under standard point-target assumptions it consists of an independent undetected-target PPP and a detected-target MBM. The sum over hypotheses carries cross-track dependence. A full PMBM update generates new Bernoulli components and updates existing existence/state densities; an assignment table alone is not a PMBM tracker.','pm'));
slides.push(slide('s-pmbm-weights','04 / PMBM association evidence','The assignment structure remains; the weights change.',[
 R('existing',72,213,544,345),T('existing-head',100,237,488,29,'EXISTING BERNOULLI TRACK',14,C.green,700),E('rho0',96,283,496,58,t`\rho_{i0}=1-r_i+r_i\int(1-p_D(x))p_i(x)\,dx`,22),E('rhoj',96,365,496,68,t`\rho_{ij}=r_i\int p_D(x)g(z_j\mid x)p_i(x)\,dx`,23),E('norm-old',96,469,496,55,t`\ell_{i0}=\rho_{i0},\qquad\ell_{ij}=\rho_{ij}/q_j`,25),
 R('new',664,213,544,345),T('new-head',692,237,488,29,'CLUTTER OR A NEW DETECTION',14,C.orange,700),E('ej',688,283,496,62,t`e_j=\int p_D(x)g(z_j\mid x)\lambda^u(x)\,dx`,25),E('qj',688,373,496,58,t`q_j=c(z_j)+e_j`,28),E('new-r',688,453,496,87,t`r_j^{\rm new}=\frac{e_j}{q_j}`,28),
 T('condition',72,578,1136,48,'Condition on one predicted parent hypothesis; the new-target existence is conditional on an unassigned detection.',19,C.muted)
], 'Here r_i is predicted existence, p_i is conditional state density, g is measurement likelihood, c is clutter intensity, and lambda^u is undetected PPP intensity. Before gating, a child weight is proportional to parent weight times product_i rho_i,a_i times product over unassigned detections q_j. Divide by the common product over all q_j to obtain the same normalized assignment structure. New means newly detected, not necessarily born this scan. Marginal new existence also multiplies the probability that the detection is unassigned. All parent hypotheses and full state/existence updates are needed in PMBM.','pm'));
slides.push(slide('s-bridge','04 / A specific bridge','TOMB/P uses marginals to simplify the detected-target mixture.',[
 ...[['Multi-Bernoulli mixture','Alternative joint associations',C.orange],['Association marginals','Exact summation or BP',C.green],['One multi-Bernoulli','Track-oriented approximation',C.blue]].flatMap(([head,body,col],i)=>{const x=72+i*395;return [R('bridge-'+i,x,258,346,229),T('bridge-head-'+i,x+24,288,298,72,head,28,col,700),T('bridge-body-'+i,x+24,399,298,57,body,21,C.muted),...(i<2?[T('arrow-'+i,x+350,342,42,50,'→',31,C.muted,400,'center')]:[])];}),
 R('ppp-retained',72,529,1136,68,'#edf7f3'),T('ppp-retained-text',96,550,1088,35,'The undetected-target PPP is retained throughout.',24,C.green,600,'center')
], 'TOMB/P approximates the association distribution in a track-oriented way and collapses the detected-object MBM to one MB, while retaining the PPP. MOMB/P uses a different grouping. BP may supply approximate association marginals, but other marginal computations are possible. This is a specific connection, not an equivalence between all BP-based trackers and PMBM.','mb'));
slides.push(slide('s-takeaways','Takeaways / Choosing the approximation','Separate the inference method from the posterior you retain.',[
 ...[['01','BP computes association marginals.','Local messages reduce association work to repeated passes over candidate pairs.',C.green],['02','Joint hypotheses retain dependence.','Ranked assignment and pruning trade retained alternatives against computation.',C.orange],['03','PMBM defines a full posterior representation.','BP can support a marginal approximation; existence, state, and new targets still need updates.',C.blue]].flatMap(([n,head,body,col],i)=>{const y=220+i*130;return [T('num-'+i,72,y,64,48,n,30,col,700),T('head-'+i,159,y,1049,37,head,27,C.ink,700),T('body-'+i,159,y+52,1000,58,body,23,C.muted)];})
], 'The audience should leave with three distinct objects: the local assignment model, the algorithm used to estimate association probabilities, and the representation carried into the next scan. The worked experiments compare association quantities under one shared model, not end-to-end tracker accuracy. The companion article contains the extended derivations.'));
slides.push(slide('s-extensions','Extensions / Sources','Continue the derivation or explore all topics.',[
 link('article',72,222,1110,'Extended notes & association experiments →','/bp-vs-pmbm/',29),T('article-copy',72,272,1070,52,'Full derivations, additional numerical experiments, and model assumptions.',22,C.muted),
 link('handover',72,353,1110,'All topics →','/',29),T('handover-copy',72,403,1070,52,'Explore the other presentations and interactive explanations.',22,C.muted),
 R('source-rule',72,486,1136,1,C.line,'none',0),T('source-label',72,512,1136,26,'PRIMARY SOURCES',13,C.green,700),
 ...Object.entries(refs).map(([key,[label,url]],i)=>link('source-'+key,72,550+i*31,1110,label+' ↗',url,17))
], 'The extended article remains subordinate to this slide deck in the site navigation. Primary sources: Williams and Lau (2014), García-Fernández et al. (2018), and Williams (2015). Return to the homepage for all topics.'));
const demos=slides.filter(s=>s.demo).map((s)=>({...s.demo,slideIndex:slides.indexOf(s)}));
const doc={format:'bento/slides',version:1,docId:'bp-vs-pmbm-data-association-deck',title:'Belief-Propagation MTT',readonly:true,meta:{author:'Liping Bai',subject:'Association inference and its connection to PMBM',company:'bailiping.com'},size:{width:1280,height:720},theme:{background:'#ffffff',color:C.ink,accent:C.green,fontFamily:FONT},slides:slides.map(({demo,...s})=>s)};
const json=x=>JSON.stringify(x,null,1).replaceAll('<','\\u003c');
const path=new URL('./index.html',import.meta.url);let html=fs.readFileSync(path,'utf8');
html=html.replace(/\s*<!-- bp-host-start -->[\s\S]*?<!-- bp-host-end -->/g,'').replace(/\s*<script type="module" src="\.\.\/assets\/deck-extensions.js"><\/script>/g,'').replace(/\s*<meta name="description"[^>]*>/g,'').replace(/\s*<link rel="canonical"[^>]*>/g,'');
html=html.replace(/(<script type="application\/bento\+json" id="bento-doc">\s*)[\s\S]*?(\s*<\/script>)/,(_,a,b)=>a+json(doc)+b).replace(/(<script type="application\/json" id="bento-inline-live-map">\s*)[\s\S]*?(\s*<\/script>)/,(_,a,b)=>a+json(demos)+b).replace(/<title>[^<]*<\/title>/,'<title>Belief-Propagation MTT | Bai Liping</title>').replace('</head>',`<!-- bp-host-start -->
<meta name="description" content="Belief-propagation multi-target tracking: association weights, factor graphs, message passing, and the connection to PMBM, with three interactive experiments.">
<link rel="canonical" href="https://bailiping.com/bp-vs-pmbm-slides/">
<link rel="stylesheet" href="./slides.css?v=20260918-redesign">
<script defer src="./navigation.js?v=20260918-redesign"></script>
<script src="./assets/math-config.js"></script>
<script defer src="./assets/mathjax-3.2.2-tex-svg-full.js"></script>
<script defer src="../assets/mathjax-dynamic.js"></script>
<!-- bp-host-end --></head>`);
fs.writeFileSync(path,html);console.log(`Built ${slides.length} slides with ${demos.length} introduction/live pairs.`);
