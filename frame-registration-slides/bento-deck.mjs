// Frame registration deck in the site's shared Bento slide system (see radar-slam/bento-deck.mjs).
import {applyDeckExtensions} from '../assets/deck-extensions.mjs';
const C={paper:'#F7F5EF',panel:'#FFFEFB',ink:'#203129',muted:'#66756E',rule:'#D8DED7',green:'#2F6B4F',soft:'#E7F0EA',rust:'#A94F2A',warm:'#F5E8DF',blue:'#496E87',cool:'#E7EEF3',amber:'#986B22'};
const serif="Georgia, 'Times New Roman', serif",sans="Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",mono="'SFMono-Regular', Consolas, monospace";
const R=String.raw;
const M=s=>`<span class="math-tex math-display">\\[${s}\\]</span>`;
const I=s=>`<span class="math-tex math-inline">\\(${s}\\)</span>`;
const small=s=>`<span style="font-size:.82em;color:${C.muted}">${s}</span>`;
function text(id,x,y,w,h,html,o={}){return {id,type:'text',x,y,w,h,rotation:0,opacity:1,html,fontSize:18,fontFamily:sans,fontWeight:400,color:C.ink,align:'left',valign:'top',lineHeight:1.4,...o};}
function box(id,x,y,w,h,fill=C.panel,stroke=C.rule,o={}){return {id,type:'shape',shape:'rect',x,y,w,h,fill,stroke,strokeWidth:1,radius:14,rotation:0,opacity:1,...o};}
function line(id,x,y,dx,dy,color=C.rule,width=2,o={}){const length=Math.hypot(dx,dy);return {id,type:'shape',shape:'line',x:x+dx/2-length/2,y:y+dy/2-width/2,w:length,h:width,fill:color,stroke:'none',strokeWidth:0,radius:0,rotation:Math.atan2(dy,dx)*180/Math.PI,opacity:1,...o};}
function circle(id,x,y,size,fill=C.panel,stroke=C.green,o={}){return {id,type:'shape',shape:'ellipse',x,y,w:size,h:size,fill,stroke,strokeWidth:2,rotation:0,opacity:1,...o};}
function panel(id,x,y,w,h,label,html,o={}){return [box(id+'-bg',x,y,w,h,o.fill||C.panel,o.stroke||C.rule),text(id+'-label',x+22,y+17,w-44,23,label,{fontFamily:mono,fontSize:11,fontWeight:800,color:o.accent||C.green,letterSpacing:.8}),text(id+'-body',x+22,y+(o.top||50),w-44,h-(o.top||50)-14,html,{fontSize:o.size||18,lineHeight:1.45,...(o.body||{})})];}
function chrome(source){return [box('chrome-rule',72,669,1136,1,C.rule,C.rule,{radius:0}),text('chrome-home',72,684,278,20,'BAI LIPING · ESTIMATION NOTES',{fontFamily:mono,fontSize:10,color:C.muted,link:'https://bailiping.com/'}),text('chrome-source',350,684,580,21,source,{fontFamily:mono,fontSize:10,color:C.muted,align:'center'}),text('chrome-index',1032,684,176,20,'CONTENTS ↗',{fontFamily:mono,fontSize:10,color:C.green,align:'right',link:'contents'})];}
function nativeTable(id,x,y,w,h,headers,rows,o={}){return {id,type:'table',x,y,w,h,rotation:0,opacity:1,header:true,columns:(o.columns||headers.map(()=>1)).map(w=>({w})),rows:[{cells:headers.map(html=>({html}))},...rows.map(row=>({cells:row.map(html=>({html}))}))],style:{headerBg:o.headerBg||C.green,headerColor:'#FFFFFF',zebra:o.zebra||C.soft,borderColor:C.rule,borderWidth:1,cellPadX:o.padX||13,cellPadY:o.padY||10,fontSize:o.fontSize||16,color:C.ink,fontFamily:sans,radius:12}};}

const slides=[];
const toc={};
function add(id,section,title,sub,elements,notes,source,tocTitle){
  slides.push({id,background:C.paper,transition:'none',notes,elements:[
    text('eyebrow',72,35,1110,23,section.toUpperCase(),{fontFamily:mono,fontSize:11,fontWeight:800,color:C.rust,letterSpacing:1.3}),
    text('heading',72,70,1136,58,title,{fontFamily:serif,fontSize:38,fontWeight:700,lineHeight:1.05}),
    text('subtitle',72,129,1136,44,sub,{fontSize:16,color:C.muted,lineHeight:1.35}),
    ...elements,...chrome(source)]});
  if(tocTitle)toc[id]=tocTitle;
}
// Equation band across the top, then two explanatory panels.
function eqTwo(id,section,title,sub,[eqLabel,eqHtml,eqH=150],[lLabel,left],[rLabel,right],notes,source,tocTitle,o={}){
  const y=186+eqH+18,h=633-y;
  add(id,section,title,sub,[
    ...panel(id+'-eq',72,186,1136,eqH,eqLabel,eqHtml,{fill:C.soft,size:o.eqSize||19,top:42,body:{align:'center',color:C.ink}}),
    ...panel(id+'-left',72,y,558,h,lLabel,left,{size:o.size||19}),
    ...panel(id+'-right',650,y,558,h,rLabel,right,{size:o.size||19,accent:C.rust})
  ],notes,source,tocTitle);
}
// A live experiment replaces the exact rectangle of its static fallback.
const bounds={x:72,y:180,width:1136,height:475};
const live=[];
function liveSlide(intro,demo,section,title,prompt,fallbackTitle,fallbackBody,notes,source){
  const id=intro+'-live';
  add(id,section+' · experiment',title,prompt,[
    box('fallback-bg',72,180,1136,475,C.panel,C.rule),
    {id:'fallback',type:'image',x:73,y:181,w:1134,h:473,src:`fallback/${demo}.png`,fit:'contain',alt:`${fallbackTitle}: initial state of the live lab`},
    box('live-demo-mount',72,180,1136,475,'rgba(255,255,255,0)','rgba(255,255,255,0)',{opacity:0})
  ],`${fallbackTitle}. ${fallbackBody} ${notes} Direct lab: /frame-registration-slides/live/?demo=${demo}. Page Up / Page Down navigate from inside the lab; Escape returns focus to the presentation.`,source);
  live.push({introSlide:intro,slide:id,demo,title});
}

// 1 · Cover
add('overview','Frame registration / a teaching guide','Rigid frame registration.','Separate the correspondence model, the alignment objective, and the optimizer.',[
  text('hero',72,200,690,120,'Which point is which?<br><span style="color:#2F6B4F">How far did it move?</span>',{fontFamily:serif,fontSize:48,fontWeight:700,lineHeight:1.1}),
  text('cover-copy',74,340,680,92,'A ladder of correspondence assumptions—known, corrupted, unknown, cell-based—followed by soft and correspondence-free alternatives.',{fontSize:21,color:C.muted}),
  ...[['01','Known pairs','kabsch'],['02','Unknown pairs','icp'],['03','No pairs at all','families']].flatMap(([n,t,link],i)=>[box('cover-card-'+i,72+i*227,452,210,104,i===1?C.cool:C.soft),text('cover-card-'+i+'-label',90+i*227,470,176,68,n+'<br><b>'+t+'</b>',{fontSize:18,link})]),
  box('cover-route',790,188,418,445,C.panel,C.rule),
  ...[['01','The problem & the ladder','problem'],['02','Closed form & RANSAC','kabsch'],['03','ICP & structured residuals','icp'],['04','NDT score landscape','ndt'],['05','Soft & correspondence-free','families'],['06','Compare, evaluate, verify','comparison']].map(([n,t,link],i)=>text('route-'+n,816,218+i*62,363,38,`<span style="color:#A94F2A">${n}</span> &nbsp; ${t} →`,{fontSize:18,link})),
  text('cover-boundary',74,582,690,20,'SLIDE COUNT · 3 LIVE LABS · PLANAR SE(2) TEACHING MODELS',{fontSize:12,fontFamily:mono,color:C.green}),
  text('cover-companion',74,608,690,24,'Bai Liping · <a href="/frame-registration/">companion article: bailiping.com/frame-registration</a> · reviewed 24 Sep 2026',{fontSize:14,color:C.muted})
],'A planar teaching guide, not an exhaustive survey or an official benchmark reproduction. Follow the companion article for derivations, implementation boundaries, primary references and numerical tests. Advance normally: each live lab loads on the slide after its introduction.','FRAME REGISTRATION · PLANAR TEACHING MODELS');

// 2 · Contents (filled after every slide exists, so titles and page numbers cannot drift)
add('contents','Contents','What’s inside.','Every entry opens its slide. Each live lab follows its introduction.',[],'Every entry is a link, and the number is the page you will land on. Use it to skip ahead or to answer a question out of order.','FRAME REGISTRATION · CONTENTS');

// 3 · Problem
add('problem','01 · the problem','Two clouds, one rigid motion.','Estimate the transform that maps source coordinates into target coordinates.',[
  ...panel('given',72,186,676,447,'IF THE PAIRS WERE KNOWN',M(R`(\hat R,\hat t)=\arg\min_{R\in SO(d),\;t\in\mathbb R^d}\;\sum_i\lVert Rp_i+t-q_i\rVert^2`)+`Source → target: ${I(R`T(p)=Rp+t`)}. A proper rotation ${I(R`R\in SO(d)`)} excludes reflection and scale.<br><br>The demos use ${I(R`SE(2)`)}: three pose parameters. 3-D rigid registration uses ${I(R`SE(3)`)}: six. A similarity transform additionally estimates scale.`,{fill:C.soft,size:21}),
  ...panel('catch',772,186,436,447,'THE CATCH',`The sum assumes we know which target point belongs to each source point.<br><br>With unknown pairs, first choose an association or distribution model.<br><br><b>Low objective, numerical stopping, and correct pose are three different claims.</b>`,{size:21,accent:C.rust})
],'Convention: T maps source to target. R is a proper rotation; reflections are prohibited. The demos are SE(2), except the similarity option in the article’s CPD walkthrough. Three-dimensional observability differs from the planar case.','FRAME REGISTRATION · NOTATION','Two clouds, one rigid motion');

// 4 · Ladder
const rungs=[['1','Known correspondences','Kabsch–Umeyama · exact optimum of paired least squares','kabsch'],['2','Corrupted correspondences','RANSAC · minimal fits compete for consensus','ransac'],['3','Unknown correspondences','ICP · assign nearest, solve, repeat','icp'],['3½','Structured residuals','Point-to-plane · GICP · robust kernels','structured'],['4','Point-to-cell association','NDT · target geometry as Gaussian cells','ndt']];
add('ladder','01 · roadmap','The correspondence ladder.','Different assumptions about matches lead to different objectives. Complexity is not a strict ladder.',
  rungs.flatMap(([n,name,method,link],i)=>{const y=186+i*90;return [
    box('rung-'+i,72,y,1136,78,i%2?C.panel:C.soft,C.rule,{radius:12}),
    circle('rung-'+i+'-dot',94,y+17,44,C.panel,C.green),
    text('rung-'+i+'-n',94,y+27,44,24,n,{fontFamily:mono,fontSize:14,fontWeight:800,align:'center',color:C.green}),
    text('rung-'+i+'-name',160,y+22,420,34,name+' →',{fontFamily:serif,fontSize:23,fontWeight:700,link}),
    text('rung-'+i+'-method',600,y+26,584,30,method,{fontSize:17,color:C.muted})];}),
'The ladder is a teaching order, not a theorem about information or complexity. NDT still performs point-to-cell association. Known-pair Kabsch is exact only for its stated least-squares objective. Each rung name links to its slide.','FRAME REGISTRATION · ROADMAP','The correspondence ladder');

// 5 · Kabsch
eqTwo('kabsch','02 · rung one · known correspondences','Kabsch–Umeyama: the closed form.','With fixed pairs, the best rigid fit is a singular value decomposition away.',
  ['CENTRE, CORRELATE, DECOMPOSE',M(R`H=\sum_i\tilde p_i\tilde q_i^{\mathsf T}=U\Lambda V^{\mathsf T},\qquad \hat R=VCU^{\mathsf T},\qquad \hat t=\bar q-\hat R\bar p`)+`${I(R`\tilde p_i=p_i-\bar p,\;\tilde q_i=q_i-\bar q`)}; &nbsp;${I(R`C=\operatorname{diag}\!\big(1,\ldots,1,\det(VU^{\mathsf T})\big)`)} prevents a reflection.`,150],
  ['KEY IDEA',`Fixed paired least squares has a closed-form global optimum over proper rotations and translation. No starting pose is required.<br><br><b>The optimum need not equal the true motion</b> under noise or wrong pairs.`],
  ['WHY IT MATTERS',`Two distinct 2-D pairs, or three noncollinear 3-D pairs, determine a noise-free rigid pose. Collinear 3-D points leave rotation about their line undetermined.<br><br>Point-to-point ICP reuses this solve on its current pairs.`],
  'H = sum of centred p times centred q transpose. With H = U Lambda V^T, R = V C U^T and C = diag(1, ..., det(V U^T)); t = qbar − R pbar. This excludes reflections. Degenerate geometry can make the solution nonunique.','Kabsch 1976 · Arun et al. 1987 · Umeyama 1991','Kabsch–Umeyama: the closed form');

// 6–7 · RANSAC
eqTwo('ransac','03 · rung two · corrupted correspondences','RANSAC: consensus over trust.','Fit minimal samples, count agreeing pairs, keep the best, refit.',
  ['HOW MANY SAMPLES?',M(R`N=\left\lceil\frac{\log(1-p)}{\log\!\left(1-w^{s}\right)}\right\rceil`)+`Confidence ${I('p')}, inlier ratio ${I('w')}, sample size ${I('s')}; independent trials and usable all-inlier samples. ${I('w^s')} is the usual approximation.`,164],
  ['KEY IDEA',`Gross outliers can strongly distort least squares, though not every wrong match ruins it. Fit small subsets, score distance-threshold consensus, then refit.<br><br>Two distinct pairs give a planar hypothesis; three noncollinear pairs are the usual 3-D minimum.`],
  ['TRADE-OFFS',`No initial pose is needed, but <b>sampling confidence is not a correctness certificate</b>.<br><br>Thresholds, degenerate samples, repeated geometry and iteration caps all matter. Next: raise the outlier rate and watch how long consensus takes.`],
  'With independent trials, the probability of at least one all-inlier sample is 1 − (1 − a)^N; a is usually approximated by w^s, and sampling without replacement uses a combinatorial ratio. Correct-model consensus and nondegenerate geometry are further assumptions. RANSAC does not certify a global optimum.','Fischler & Bolles 1981','RANSAC: consensus over trust');
liveSlide('ransac','ransac','03 · rung two','RANSAC: consensus over trust.','Raise the outlier rate, then test batches of 36 pair hypotheses. Green points support the best pose so far.','Pair-congruence search','Hypotheses come from geometrically compatible point pairs; consensus still needs verification.','This lab searches geometric pair congruences rather than running textbook RANSAC on a supplied correspondence list. It tests batches of 36 hypotheses. The sampling bound on the previous slide does not directly certify this proposal mechanism.','Fischler & Bolles 1981');

// 8–9 · ICP
eqTwo('icp','04 · rung three · unknown correspondences','ICP: alternate guess and solve.','Freeze tentative matches, solve the closed form, rematch at the new pose.',
  ['TWO SUBPROBLEMS, ALTERNATED',M(R`c_i\leftarrow\arg\min_j\lVert Rp_i+t-q_j\rVert\quad\rightleftarrows\quad(R,t)\leftarrow\arg\min_{R,\,t}\sum_i\lVert Rp_i+t-q_{c_i}\rVert^2`)+'Assign nearest neighbours, run the rung-one fit on those pairs, repeat.',150],
  ['KEY IDEA',`Chicken and egg: the transform reveals the matches; the matches give the transform.<br><br>ICP takes nearest neighbours as provisional matches, solves the closed form, and repeats. <b>The start pose chooses the basin.</b>`],
  ['TRADE-OFFS',`With a fixed source set, exact nearest neighbours and exact paired fitting, <b>the objective never increases</b> and its values converge. That is not a global or correct-pose guarantee.<br><br>A gated or trimmed RMS can change its contributing set; fixed-count or capped objectives keep descent.`],
  'Exact vanilla point-to-point ICP does not increase a fixed nearest-neighbour squared cost. This proves convergence of cost values, not global pose recovery. Fixed-count trimmed and capped-distance variants can also be monotone; a variable gated or trimmed RMS can change its denominator.','Besl & McKay 1992 · Chetverikov et al. 2002','ICP: alternate guess and solve');
liveSlide('icp','icp','04 · rung three','ICP: alternate guess and solve.','Drag the blue scan to choose a start; Shift-drag rotates. Compare the residual with the true pose error.','Gated point-to-point ICP','The lab trims the worst 16% of surviving pairs; stopping does not certify correct alignment.','This lab gates matches, trims about 16 percent of the surviving pairs, and stops on a small RMS change or after 35 steps. Its truth-based status is a simulation diagnostic, not a deployable acceptance test.','Besl & McKay 1992');

// 10 · Structured residuals
add('structured','05 · rung three½ · structured residuals','Reshape the cost, not the loop.','The alternation stays; the residual, or its weighting, changes.',[
  ...[['POINT-TO-PLANE',M(R`\sum_i\big(n_{c_i}^{\mathsf T}(Rp_i+t-q_{c_i})\big)^2`)+'Penalize only the normal component. One plane leaves tangential motion unobservable; varied geometry is needed. The demo takes a local linearized step.<br><br>'+small('Chen & Medioni 1992'),C.soft],
      ['GENERALIZED ICP',M(R`\sum_i d_i^{\mathsf T}\big(C^{B}_{c_i}+RC^{A}_iR^{\mathsf T}\big)^{-1}d_i`)+`${I(R`d_i=q_{c_i}-(Rp_i+t)`)}. Covariances encode local surface orientation, not automatic confidence. The demo freezes the rotation-dependent covariance within each step.<br><br>`+small('Segal et al. 2009'),C.panel],
      ['ROBUST KERNELS',M(R`\sum_i\rho\big(\lVert r_i\rVert\big)`)+'Huber has <b>linear, unbounded tails</b>; Tukey has bounded loss and zero weight beyond a cutoff. Neither makes pose estimation globally convex.<br><br>'+small('M-estimators · IRLS'),C.warm]
  ].flatMap(([label,body,fill],i)=>panel('residual-'+i,72+i*382,186,358,447,label,body,{fill,size:18,accent:i===2?C.rust:C.green}))
],'Point-to-plane degeneracy: one plane leaves two tangential translations and the normal-axis rotation unconstrained in 3-D. GICP uses prescribed anisotropic surface covariances; the browser freezes them during a step. Huber loss is unbounded; Tukey loss is bounded.','Chen & Medioni 1992 · Segal et al. 2009','Reshape the cost, not the loop');

// 11–12 · NDT
eqTwo('ndt','06 · rung four · point-to-cell association','NDT: match against a Gaussian field.','Summarize each target cell by a Gaussian, then score every transformed source point.',
  ['SCORE A POSE AGAINST GAUSSIAN CELLS',M(R`s(\theta)=\sum_i\exp\!\Big(-\tfrac12\,\tilde q_i^{\mathsf T}\Sigma_{c(i)}^{-1}\tilde q_i\Big),\qquad \tilde q_i=T_\theta(p_i)-\mu_{c(i)}`)+`Unnormalized score, maximized with Newton steps; empty cells contribute zero. ${I(R`\theta`)} collects the pose parameters.`,150],
  ['KEY IDEA',`A Gaussian summarizes the target points in each cell. This removes raw point matching, <b>not association</b>: each transformed point still selects a cell.<br><br>Covariance regularization prevents singular inverses; one Gaussian can blur mixed surfaces.`],
  ['TRADE-OFFS',`The hard-cell objective is <b>piecewise analytic and can jump at cell boundaries</b>. Newton derivatives hold only while cell assignments are unchanged.<br><br>Shifted grids reduce artifacts without guaranteeing continuity. Coarse-to-fine helps; it is not globally reliable.`],
  'The displayed unnormalized score is smooth only within unchanged hard cell assignments. Covariance determinants are omitted from this score. Shifted grids do not guarantee continuity. Empty cells contribute zero. NDT variants differ; the article derives the fixed-cell gradient and Hessian.','Biber & Straßer 2003','NDT: match against a Gaussian field');
liveSlide('ndt','ndt','06 · rung four','NDT: match against a Gaussian field.','Fix the rotation with the slider, then let a direct search climb the score in translation.','Translation-only direct search on an NDT score','Four shifted grids; this is not the article’s Newton method, and a stopped search is not a certified maximum.','Do not call this Newton: the lab tests eight translation directions and shrinks its step when none improves. It holds rotation fixed, uses four grids of cell size 1.3, and adds 0.026 to covariance diagonals. The companion article implements damped Newton.','Biber & Straßer 2003');

// 13 · Families
add('families','07 · beyond hard matches','What should replace correspondences?','Commit to one match, average over candidates, or compare whole distributions.',[
  ...[['HARD · COMMIT','<b>ICP family</b><br><br>Reassign nearest neighbours, then fit. Smooth for fixed pairs; assignment switches create kinks. Indexed search is often '+I(R`O(M\log N)`)+'; the browser uses brute force.',C.soft,C.green],
      ['SOFT · AVERAGE','<b>CPD · FilterReg</b><br><br>Mixture responsibilities express alternatives. Exact EM never decreases a fixed model’s likelihood; schedules and approximate steps need separate care. Gaussian filtering can accelerate the sums.',C.cool,C.blue],
      ['DISTRIBUTIONS · COMPARE','<b>GMMReg · MMD-Reg</b><br><br>Compare mixture densities or kernel mean embeddings without discrete pair assignments. NDT differs: it selects grid cells and need not be globally smooth.',C.warm,C.rust]
  ].flatMap(([label,body,fill,accent],i)=>panel('family-'+i,72+i*382,186,358,392,label,body,{fill,accent,size:19})),
  text('family-note',72,596,1136,40,`As ${I(R`\sigma\to0`)}, the nearest component dominates for an inlier; with positive clutter weight, clutter may instead take all the mass.`,{fontSize:16,color:C.muted})
],'Do not confuse a smooth fixed-bandwidth likelihood with an exact EM update, or distribution matching with point-to-cell NDT. GMMReg minimizes a density distance without an E-step. The zero-noise nearest-neighbour limit needs an inlier condition when clutter has positive weight.','Jian & Vemuri 2011 · Myronenko & Song 2010','What should replace correspondences?');

// 14 · CPD
eqTwo('cpd','07 · soft correspondences · I','CPD: registration as mixture fitting.','Gaussians ride on the transformed source points; target points are the observations.',
  ['E-STEP: RESPONSIBILITIES WITH A CLUTTER TERM',M(R`P(m\mid x_n)=\frac{g_{mn}}{\sum_{k=1}^{M}g_{kn}+c},\qquad g_{mn}=\exp\!\Big(-\frac{\lVert x_n-T(y_m)\rVert^2}{2\sigma^2}\Big)`)+`${I(R`c=(2\pi\sigma^2)^{d/2}\,\tfrac{w}{1-w}\,\tfrac{M}{V}`)} for clutter density ${I(R`1/V`)} (the paper’s ${I(R`1/N`)} gives ${I(R`M/N`)}). CPD letters: ${I('y_m')} source, ${I('x_n')} target.`,176],
  ['KEY IDEA',`The E-step normalizes each observation’s responsibilities, clutter included. The M-step is a weighted Procrustes fit.<br><br>The paper’s “rigid” CPD already estimates a scale, ${I(R`T(y)=sRy+t`)}; fix ${I('s=1')} for strictly rigid motion.`],
  ['WHAT STANDS OUT',`The variance has a closed-form update but <b>need not shrink every step</b>; this is not deterministic annealing.<br><br>A naive E-step costs ${I(R`O(MN)`)}. Blocked sufficient sums avoid storing the full matrix; fast Gauss transforms accelerate it.`],
  'Responsibilities are normalized with clutter in the denominator. The spatial-window clutter convention differs from the paper’s 1/N convention. The paper defines its rigid transform as sRy + t, so the published rigid algorithm includes scale; the article’s walkthrough estimates s, and s = 1 gives strictly rigid motion. Variance optimization is not monotone annealing. Source: Myronenko and Song 2010, arXiv:0905.2635.','Myronenko & Song 2010','CPD: registration as mixture fitting',{size:18});

// 15 · FilterReg
add('filterreg','07 · soft correspondences · II','FilterReg: the E-step becomes a filter.','Reverse the mixture: Gaussians sit on the fixed cloud, and the moving points query it.',[
  ...panel('filterreg-eq',72,186,780,176,'FILTERED MOMENTS FOR EACH MOVING POINT',M(R`M^0_i=\sum_j\mathcal N\big(Tp_i;\,q_j,\sigma^2 I\big),\qquad M^1_i=\sum_j\mathcal N\big(Tp_i;\,q_j,\sigma^2 I\big)\,q_j`)+`Soft target ${I(R`M^1_i/M^0_i`)}; confidence from ${I(R`M^0_i`)} against an outlier constant. The M-step is a local, twist-based Gauss–Newton step.`,{fill:C.soft,size:18,top:42,body:{align:'center'}}),
  box('filterreg-stat',876,186,332,176,C.panel,C.rule),
  text('filterreg-stat-n',876,206,332,80,I(R`\sigma^2`),{fontSize:54,align:'center',color:C.rust}),
  text('filterreg-stat-cap',900,292,284,50,'may be optimized analytically (Section 3.3)',{fontSize:14,color:C.muted,align:'center',fontFamily:mono}),
  ...panel('filterreg-left',72,380,558,253,'KEY IDEA',`Because the fixed cloud carries the mixture, the filter index can be built once and reused while the bandwidth stays fixed.<br><br>FilterReg uses a customized permutohedral lattice and can add features or point-to-plane residuals.`,{size:18}),
  ...panel('filterreg-right',650,380,558,253,'THE TRADE',`<b>The variance update is not lost:</b> Section 3.3 permits analytic optimization, at the cost of rebuilding the filter when the bandwidth changes.<br><br>Filtering is approximate; exact-EM monotonicity does not automatically transfer to a filtered, linearized step.`,{size:18,accent:C.rust})
],'FilterReg reverses CPD: the observation (fixed) cloud induces the mixture, and model points query it. Section 3.3 (Optimized Variance) states that an isotropic variance can be optimized analytically, as in CPD. A fixed bandwidth allows index reuse; changing it can require a rebuild. The article’s walkthrough uses a Cartesian sparse grid, not the paper’s permutohedral implementation. Source: Gao and Tedrake 2019, arXiv:1811.10136.','Gao & Tedrake 2019','FilterReg: the E-step becomes a filter');

// 16 · MMD-Reg
eqTwo('mmd','08 · correspondence-free','MMD-Reg: clouds as distributions.','Compare feature means of the two clouds: no E-step, no discrete pairs.',
  ['KERNEL MEAN EMBEDDINGS WITH RANDOM FEATURES',M(R`\mathrm{MMD}^2(P,Q)=\lVert\mu_P-\mu_Q\rVert_{\mathcal H}^2\;\approx\;F(\theta)=\Big\lVert\frac1M\sum_i z(T_\theta p_i)-\frac1N\sum_j z(q_j)\Big\rVert^2`)+`${I(R`k(x,y)\approx z(x)^{\mathsf T}z(y)`)} with ${I('D')} Gaussian frequencies, i.e. ${I('2D')} sine/cosine features; linear cost in the number of points for fixed ${I('D')}.`,164],
  ['KEY IDEA',`Registration becomes nonlinear least squares on a <b>${I('2D')}-dimensional residual</b>.<br><br>Finite random features approximate a characteristic kernel; the finite embedding need not identify a distribution uniquely.`],
  ['WHAT IT BUYS · WHAT IT COSTS',`Smooth but nonconvex. Differentiating the selected optimum by the implicit function theorem needs an <b>isolated solution with a nonsingular pose Hessian</b>.<br><br>Partial overlap, density imbalance and outliers can shift the optimum; learned weights may mitigate, not remove, these effects.`],
  'D sine/cosine frequency pairs give 2D features. Exact Gaussian-kernel population MMD identifies probability measures; a finite random-feature approximation need not. Implicit differentiation is local and requires a nonsingular Hessian at an isolated optimum. Sources: Crane et al., ICML 2026, arXiv:2606.27818; Gretton et al. 2012.','Crane et al. 2026 · Gretton et al. 2012','MMD-Reg: clouds as distributions');

// 17 · Comparison
add('comparison','09 · comparison','Representations and implementation choices.','Costs describe computational routes, not universal lower bounds.',[
  nativeTable('cmp-table',72,186,1136,440,['','CPD','FilterReg','MMD-Reg'],[
    ['<b>Intermediate state</b>','responsibilities or sufficient sums','filtered moments; soft targets','feature means; residual of length 2D'],
    ['<b>Evaluation cost</b>','O(MN) naive; accelerations exist','near-linear approximate filtering','O((M+N)D); target term can be cached'],
    ['<b>Outlier model</b>','uniform component','uniform component; confidence','none in the basic objective'],
    ['<b>Bandwidth</b>','closed-form variance update','fixed, scheduled, or optimized','fixed or scheduled kernel length scale'],
    ['<b>Solution derivative</b>','requires regularity','depends on approximations','implicit function theorem; nonsingular Hessian'],
    ['<b>Transform family</b>','similarity (the paper’s “rigid”), affine, nonrigid','rigid, articulated, deformable','rigid in the demo; model-dependent']
  ],{columns:[1.05,1.15,1.15,1.15],fontSize:16})
],'Costs refer to computational routes, not universal lower bounds. CPD can stream sufficient sums; FilterReg supports optimized variance. The table does not claim these methods uniquely own differentiability or suitability for a sensor.','Myronenko & Song 2010 · Gao & Tedrake 2019 · Crane et al. 2026','Representations and implementation choices');

// 18 · Benchmark
add('benchmark','09 · evaluation','Measure both rotation and translation.','Planar teaching scenes, arbitrary world units, configurable noise, outliers, rotation and offset. No universal ranking.',[
  ...[['RUN IT',`Use the companion article’s <b>Benchmark 25 seeds</b> button.<br>Seeds ${I(R`40+13k,\ k=0,\ldots,24`)}; 70 steps per solver. Same clouds and local starts; RANSAC ignores initialization.`],
      ['REPORT BOTH ERRORS',`Rotation error (RRE) and translation error (RTE).<br>Joint success: ${I(R`\mathrm{RRE}<5^\circ`)} <b>and</b> ${I(R`\mathrm{RTE}<0.2`)} world units.`],
      ['CHANGE ONE THING','Vary noise, outliers and starting pose one at a time, so each failure has a single cause.'],
      ['WHAT IT IS NOT','These toy solvers are not official paper implementations. A low objective is not ground truth, a small pose step is not success, and an iteration count is not runtime.']
  ].flatMap(([label,body],i)=>panel('bench-'+i,72+(i%2)*578,186+Math.floor(i/2)*228,558,210,label,body,{fill:i===0?C.soft:i===3?C.warm:C.panel,accent:i===3?C.rust:C.green,size:19})),
],'An earlier hard-coded success-rate chart mixed rotation-only success with unsupported claims such as a universal NDT capture angle; use the reproducible 25-seed benchmark in the companion article instead. It reports rotation-only and joint pose success. No superiority claim follows from these educational implementations.','Companion article · reproducible benchmark','Measure both rotation and translation');

// 19 · Pipeline
add('pipeline','10 · in practice','Initialize, refine, then independently verify.','Global proposal, local refinement and verification are separate jobs.',[
  ...panel('pipe-a',72,186,520,176,'1 · PROPOSE AN INITIAL POSE','Features + RANSAC, a learned model, or an odometry motion prediction. No proposal is automatically basin-safe.',{fill:C.soft,size:20}),
  line('pipe-arrow',604,274,72,0,C.rust,2),
  text('pipe-arrow-head',664,258,30,30,'▶',{fontSize:16,color:C.rust}),
  ...panel('pipe-b',688,186,520,176,'2 · REFINE, THEN VERIFY','ICP, GICP or NDT refine locally. Then check overlap, geometry, motion consistency and independent support.',{size:20}),
  ...panel('rap',72,380,780,253,'A DIFFERENT ROUTE · LEARNED FLOW','RAP treats registration as conditional generation: a learned point-wise velocity field transports points to a registered scene, with rigidity enforced at test time and a rigid pose read out. It needs no initial pose, but that is <b>not a global correctness guarantee</b>.<br><br>The companion article’s small planar MLP is an analogy, not RAP or its evaluation.',{size:19,fill:C.cool,accent:C.blue}),
  box('verify-bg',876,380,332,253,C.warm,C.rust),
  text('verify-word',876,430,332,70,'Verify',{fontFamily:serif,fontSize:56,fontWeight:700,color:C.rust,align:'center'}),
  text('verify-cap',900,516,284,60,'INITIALIZATION-FREE ≠ FAILURE-FREE',{fontFamily:mono,fontSize:12,fontWeight:800,color:C.rust,align:'center',letterSpacing:.6})
],'Global proposal, local refinement and verification is common, but odometry can initialize locally from a motion prediction. KISS-ICP is point-to-point, not point-to-plane. RAP (Pan et al., arXiv:2512.01850) provides learned initialization-free inference, not failure-free registration. Do not infer RAP behaviour from the toy MLP.','KISS-ICP (Vizzo et al. 2023) · RAP (Pan et al. 2025)','Initialize, refine, then independently verify');

// 20 · Takeaways
add('takeaways','10 · takeaways','What the ladder teaches.','Four distinctions to keep when reading any registration paper.',[
  ...[['1 · SEPARATE MODEL FROM OPTIMIZER','Fixed pairs, soft associations, point-to-cell scores and distribution distances encode different assumptions. Newton, EM and LM solve different subproblems.'],
      ['2 · KNOBS ARE NOT INTERCHANGEABLE',`Gates select pairs; cell size changes geometry; ${I(R`\sigma`)} and ${I(R`\ell`)} change interactions. Coarse-to-fine can help, but these controls do different things.`],
      ['3 · SMOOTH OBJECTIVE ≠ DIFFERENTIABLE SOLUTION','Nearest-neighbour costs have kinks; hard-grid NDT can jump. Smooth MMD still needs a nonsingular Hessian for a local solution derivative.'],
      ['4 · STOPPING IS NOT SUCCESS','Validate rotation and translation in simulation. On real data, check overlap, conditioning, motion consistency and independent evidence.']
  ].flatMap(([label,body],i)=>panel('take-'+i,72+(i%2)*578,186+Math.floor(i/2)*206,558,190,label,body,{fill:i===0?C.soft:C.panel,accent:i===3?C.rust:C.green,size:19})),
  text('timeline',72,604,1136,24,'1976 KABSCH · 1981 RANSAC · 1987 ARUN · 1992 ICP & POINT-TO-PLANE · 2003 NDT · 2009 GICP · 2010 CPD · 2019 FILTERREG · 2025 RAP · 2026 MMD-REG',{fontFamily:mono,fontSize:11,color:C.muted,align:'center',letterSpacing:.4})
],'Distinguish low cost, optimizer stopping and correct registration. Scale schedules, robust losses and local geometry have separate roles. Use synthetic ground truth only for evaluation; deployed checks must rely on independent evidence and consistency.','FRAME REGISTRATION · SUMMARY','What the ladder teaches');

// Contents entries, cover count and extension appendix are derived from the finished slide list.
const doc={format:'bento/slides',version:1,docId:'frame-registration-bento',title:'Rigid frame registration',readonly:true,meta:{author:'Bai Liping',subject:'Frame registration methods with three interactive labs',company:'bailiping.com'},size:{width:1280,height:720},theme:{background:C.paper,color:C.ink,accent:C.green,fontFamily:sans},slides};
applyDeckExtensions(doc,'frame-registration-slides');
const all=doc.slides;
toc['s-extensions']='Continue beyond the slides';
const entries=Object.entries(toc);
const half=Math.ceil(entries.length/2);
const contents=all.find(s=>s.id==='contents');
contents.elements.splice(3,0,...entries.flatMap(([id,title],k)=>{
  const col=k<half?0:1,row=col?k-half:k,x=72+col*578,y=186+row*56;
  const page=String(all.findIndex(s=>s.id===id)+1).padStart(2,'0');
  const hasLab=all.some(s=>s.id===id+'-live');
  return [box('toc-rule-'+k,x,y+50,558,1,C.rule,C.rule,{radius:0}),
    text('toc-'+k,x,y+12,558,34,`<span style="font-family:${mono};font-size:13px;color:${C.rust}">${page}</span>&nbsp;&nbsp;&nbsp;${title}${hasLab?` <span style="font-family:${mono};font-size:11px;color:${C.green}">· LIVE LAB</span>`:''}`,{fontFamily:serif,fontSize:19,link:id})];
}));
all.find(s=>s.id==='overview').elements.find(e=>e.id==='cover-boundary').html=`${all.length} SLIDES · ${live.length} LIVE LABS · PLANAR SE(2) TEACHING MODELS`;

export const deck=doc;
export const inlineLiveMap=live.map(({introSlide,slide,demo,title})=>({introSlide,slide,slideIndex:all.findIndex(s=>s.id===slide),inline:true,layout:'region',bounds,src:`./live/?demo=${demo}&embed=region`,source:`./live/?demo=${demo}`,title,sandbox:'allow-scripts',hideSource:true,readyMessage:true,unloadWhenHidden:true}));
