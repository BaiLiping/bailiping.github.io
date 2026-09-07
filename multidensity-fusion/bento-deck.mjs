/** Authoring source: regular Bento elements, not a second slide engine.
 * Coordinates are in the reference deck's 1280 x 720 canvas.
 * Equations use MathJax; working experiments use the existing inline-live host.
 */
const C={paper:'#F7F5EF',panel:'#FFFEFB',ink:'#203129',muted:'#66756E',rule:'#D8DED7',green:'#2F6B4F',soft:'#E7F0EA',blue:'#496E87',rust:'#A94F2A'};
const SANS="Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
const SERIF="Georgia, 'Times New Roman', serif";
const MONO="'SFMono-Regular', Consolas, monospace";
const raw=String.raw;
const inline=s=>`<span class="math-tex math-inline">\\(${s}\\)</span>`;
const display=s=>`<span class="math-tex math-display">\\[${s}\\]</span>`;
const text=(id,x,y,w,h,html,size=18,options={})=>({id,type:'text',x,y,w,h,html,fontSize:size,fontFamily:SANS,fontWeight:400,color:C.ink,lineHeight:1.35,align:'left',valign:'top',rotation:0,opacity:1,...options});
const rect=(id,x,y,w,h,fill=C.panel)=>({id,type:'shape',shape:'rect',x,y,w,h,fill,stroke:C.rule,strokeWidth:1,radius:13,rotation:0,opacity:1});
const formula=(id,x,y,w,h,latex,size=23)=>text(id,x,y,w,h,display(latex),size,{align:'center',valign:'middle'});
const panel=(id,x,y,w,h,title,body,size=18,fill=C.panel)=>[
  rect(`${id}-panel`,x,y,w,h,fill),
  text(`${id}-label`,x+20,y+17,w-40,26,title,12,{fontFamily:MONO,fontWeight:800,color:C.green,letterSpacing:.7}),
  text(`${id}-body`,x+20,y+58,w-40,h-76,body,size)
];
const callout=(html)=>[rect('takeaway-bg',72,584,1136,54,C.soft),text('takeaway',90,597,1100,36,html,16,{color:C.green,fontWeight:600})];
const refs={
  prior:{label:'Wu et al. · shared priors',url:'https://arxiv.org/abs/2212.07311',title:'P. Wu, T. Imbiriba, V. Elvira & P. Closas. Bayesian data fusion with shared priors. arXiv:2212.07311 (2022).'},
  ci:{label:'Julier & Uhlmann · CI (1997)',url:'https://ieeexplore.ieee.org/document/609105/',title:'S. J. Julier & J. K. Uhlmann. A Non-divergent Estimation Algorithm in the Presence of Unknown Correlations. ACC (1997), 2369–2373.'},
  rumor:{label:'Campbell & Ahmed (2016)',url:'https://ieeexplore.ieee.org/document/7515322/',title:'M. Campbell & N. R. Ahmed. Distributed Data Fusion: Neighbors, Rumors, and the Art of Collective Knowledge. IEEE Control Systems Magazine 36(4), 83–109 (2016).'},
  kl:{label:'Abbas · KL pools (2009)',url:'https://pubsonline.informs.org/doi/10.1287/deca.1080.0133',title:'A. E. Abbas. A Kullback-Leibler View of Linear and Log-Linear Pools. Decision Analysis 6(1), 25–37 (2009).'},
  transport:{label:'Agueh & Carlier (2011)',url:'https://epubs.siam.org/doi/10.1137/100805741',title:'M. Agueh & G. Carlier. Barycenters in the Wasserstein Space. SIAM Journal on Mathematical Analysis (2011). DOI: 10.1137/100805741.'},
  rfs:{label:'Clark et al. · multi-object GCI (2010)',url:'https://udrc.eng.ed.ac.uk/sites/udrc.eng.ed.ac.uk/files/publications/0017.pdf',title:'D. Clark, S. Julier, R. Mahler & B. Ristić. Robust Multi-Object Sensor Fusion with Unknown Correlations. Sensor Signal Processing for Defence (2010).'},
  aa:{label:'Li et al. · AA, Part I (2024)',url:'https://arxiv.org/abs/2110.01440',title:'T. Li, Y. Song, E. Song & H. Fan. Arithmetic Average Density Fusion — Part I: Some Statistic and Information-theoretic Results. Information Fusion 104, 102199 (2024).'},
  aaRfs:{label:'Li · AA, Part II (2024)',url:'https://arxiv.org/abs/2209.10433',title:'T. Li. Arithmetic Average Density Fusion — Part II: Unified Derivation for Unlabeled and Labeled RFS Fusion. IEEE TAES 60(3), 3255–3268 (2024).'},
  consensus:{label:'Battistelli & Chisci (2014)',url:'https://www.sciencedirect.com/science/article/pii/S0005109813005591',title:'G. Battistelli & L. Chisci. Kullback–Leibler average, consensus on probability densities, and distributed state estimation with guaranteed stability. Automatica 50(3), 707–718 (2014).'}
};
const slides=[];
function add(id,section,title,subtitle,elements,sources=[],notes=''){
  slides.push({id,background:C.paper,transition:slides.length?'morph':'none',
    notes:`${notes}\nSources: ${sources.map(k=>refs[k].title+' '+refs[k].url).join('\n')}`,
    elements:[
      text('eyebrow',72,34,1120,22,`ESTIMATION NOTES / ${section.toUpperCase()}`,11,{fontFamily:MONO,color:C.green,fontWeight:800,letterSpacing:1.4}),
      text('heading',72,69,1136,57,title,38,{fontFamily:SERIF,fontWeight:700,lineHeight:1.1}),
      text('subtitle',74,128,1132,44,subtitle,17,{color:C.muted}),...elements,
      ...sources.map((k,i)=>text(`source-${i}`,74+i*365,648,350,20,refs[k].label+' ↗',10,{color:C.muted,link:refs[k].url})),
      rect('footer-rule',72,675,1136,1,C.rule),
      text('footer-site',74,688,290,18,'BAI LIPING · MULTIDENSITY FUSION',10,{color:C.muted,fontFamily:MONO}),
      text('footer-map',925,688,180,18,'CHAPTER MAP ↗',10,{color:C.green,align:'right',link:'map'})
    ]});
}
function eqSlide(id,section,title,subtitle,equations,leftTitle,leftBody,rightTitle,rightBody,sources,notes=''){
  add(id,section,title,subtitle,[rect('equation-panel',72,184,1136,139,C.soft),
    formula('main-equation',92,198,1096,111,equations,24),
    ...panel('left',72,344,554,283,leftTitle,leftBody,18),
    ...panel('right',646,344,562,283,rightTitle,rightBody,18)],sources,notes);
}
function lab(id,title,subtitle,demo,sources){
  add(id,'live experiment',title,subtitle,[rect('live-placeholder',72,181,1136,455)],sources,
    `Deterministic teaching experiment. Open live/?demo=${demo} for a standalone responsive view. All formulas are implemented in math.mjs and covered by test.mjs. These are not empirical benchmarks or a full tracking system.`);
  live.push({slide:id,slideIndex:slides.length-1,inline:true,layout:'region',bounds:{x:74,y:181,width:1132,height:457},src:`./live/?demo=${demo}&embed=region`,source:`./live/?demo=${demo}`,title,hideSource:true,readyMessage:true,unloadWhenHidden:false});
}
const live=[];

// 01 — Cover. Connected cards and a moving line echo the sample deck.
add('overview','probability · information · random finite sets','Multidensity fusion','Single-state probability densities → multi-object densities. Choose a rule by what the sources share.',[
  text('hero-a',74,196,700,70,'Many beliefs.',60,{fontFamily:SERIF,fontWeight:700,lineHeight:1}),
  text('hero-b',74,270,830,74,'One fused density.',60,{fontFamily:SERIF,fontWeight:700,color:C.green,lineHeight:1}),
  text('hero-explainer',78,359,1040,57,'The difficult part is not averaging numbers. It is knowing which information is new, which is shared, and which hypotheses must survive.',22,{color:C.muted}),
  {id:'flow',type:'shape',shape:'line',x:80,y:438,w:1118,h:1,fill:'none',stroke:C.green,strokeWidth:2,lineEnd:'arrow',rotation:0,opacity:1,fx:{loop:{type:'dash-march'}}},
  ...[['01','Evidence','bayes'],['02','Density pools','gci'],['03','Multi-object','sets'],['04','Networks','consensus']].flatMap(([n,t,target],i)=>[
    rect(`cover-card-${i}`,72+i*288,473,272,94,i===0?C.soft:C.panel),
    text(`cover-n-${i}`,92+i*288,490,45,22,n,12,{color:C.green,fontWeight:800,fontFamily:MONO,link:target}),
    text(`cover-link-${i}`,92+i*288,518,240,32,t+' ↗',21,{fontFamily:SERIF,fontWeight:700,link:target})]),
  text('cover-nav',76,603,1120,24,'Arrow keys: navigate · Chapter map: jump · Six live labs: change assumptions, not just numbers.',15,{color:C.muted})
],['prior','kl','rfs'],'Scope: fusion of multiple probability densities for the same unknown state, then multi-object (random-finite-set) density fusion. This is not kernel density estimation or a ranking of universally best fusion algorithms.');

// 02 — Definitions before formulas.
add('notation','start here','Align the meaning before the math','Every density must describe the same state, time, frame, units, and reference measure.',[
  ...panel('definition',72,185,554,377,'SINGLE-STATE DENSITIES',
    `${inline(raw`x\in\mathbb R^d`)}: the common unknown state.<br><br>${inline(raw`p_i(x)`)}: source ${inline('i')}'s normalized density, with ${inline(raw`\int p_i\,d\mu=1`)}.<br><br>${inline(raw`\mu_i,P_i`)}: mean and covariance of that density.<br><br>${inline(raw`w_i\ge0,\quad\sum_iw_i=1`)}: pooling weights; omit zero-weight sources.`,19),
  ...panel('meaning',646,185,562,377,'DO NOT CONFUSE THESE OBJECTS',
    '<b>Likelihood:</b> evidence about x, not usually a density in x.<br><br><b>Posterior:</b> prior plus already-assimilated evidence.<br><br><b>Error cross-covariance:</b> dependence between estimators; marginal covariances do not reveal it.<br><br><b>Intensity:</b> expected objects per unit volume; it need not integrate to one.',18),
  ...callout('Frame registration, time propagation, and object association come before density fusion.')
],['rumor','rfs'],'Integrals use a common reference measure μ. Multi-object integrals later use the corresponding finite-set reference measure. Coordinate changes require consistent Jacobians.');

// 03 — High-level group map.
add('map','chapter map','Four questions, four different answers','These rules solve different problems. Similar-looking formulas do not make them interchangeable.',[
  ...[['01 · KNOWN EVIDENCE','Bayesian fusion','Combine independent new likelihoods; subtract known common information.','bayes'],
      ['02 · RETAIN ALTERNATIVES','Arithmetic pool (AA)','Mix the source densities; preserve modes and between-source spread.','aa'],
      ['03 · UNKNOWN DEPENDENCE','Logarithmic pool / GCI','Pool log densities. For Gaussians, obtain covariance intersection (CI).','gci'],
      ['04 · TRANSPORT GEOMETRY','Wasserstein barycenter','Average distributions through a transport metric, not an evidence model.','wasserstein']].flatMap(([label,title,body,target],i)=>{
        const x=72+(i%2)*576,y=185+Math.floor(i/2)*203;
        return [...panel(`route-${i}`,x,y,560,185,label,`<b>${title}</b><br><br>${body}`,18),text(`jump-${i}`,x+435,y+17,100,22,'OPEN ↗',11,{link:target,color:C.green,align:'right'})];
      }),
  text('map-rfs',76,604,1118,28,'Continue to random finite sets ↗',18,{link:'sets',color:C.green,fontWeight:600})
],['prior','kl','transport'],'CI is a special Gaussian case of the normalized weighted geometric pool. CI covariance consistency has assumptions; the general GCI density does not inherit every Gaussian covariance guarantee.');

// 04 — Correct evidence accounting.
eqSlide('bayes','01 · known evidence','Count the common prior once','Assume a common prior and likelihoods conditionally independent given the state.',
  raw`p_i(x)\propto p_0(x)L_i(x),\qquad p_F(x)\propto p_0(x)\prod_{i=1}^M L_i(x)=\frac{\prod_{i=1}^M p_i(x)}{p_0(x)^{M-1}}`,
  'WHY DIVIDE?',`Multiplying ${inline(raw`M`)} local posteriors repeats the prior ${inline(raw`M`)} times. Divide out the extra ${inline(raw`M-1`)} copies.<br><br>The equality is on the relevant support, where ${inline(raw`p_0>0`)}. Normalize the result.`,
  'WHEN THIS IS NOT ENOUGH','Shared measurements, earlier messages, or loops create common information beyond the initial prior.<br><br>Subtract the correct common-information factor, not automatically p₀. Unknown dependence cannot be reconstructed from the marginals alone.',
  ['prior','rumor'],'For two sources with a known common-information density pc and conditionally independent private information, pF ∝ p1 p2 / pc. For multiple overlapping histories, simple powers of one common density are not generally sufficient.');

// 05
lab('prior-lab','Live: the cost of repeating a prior','Move the prior variance and the number of sensors. Compare exact Bayes, an uncorrected product, and GCI.','prior',['prior']);

// 06
eqSlide('known-correlation','01 · known dependence','Use the cross-covariance when it is known','Two unbiased estimates of the same state; valid joint error covariance and invertible S.',
  raw`\begin{aligned}S&=P_1+P_2-C-C^{\mathsf T},& W&=(P_1-C)S^{-1}\\ \hat x_F&=\hat x_1+W(\hat x_2-\hat x_1),&P_F&=P_1-(P_1-C)S^{-1}(P_1-C^{\mathsf T})\end{aligned}`,
  'WHAT IS C?',`${inline(raw`e_i=\hat x_i-x`)} and ${inline(raw`C=\mathbb E[e_1e_2^{\mathsf T}]`)}.<br><br>The formula minimizes error covariance within the linear unbiased fusion class. Setting ${inline('C=0')} is a model assumption, not a harmless default.`,
  'A DIFFERENT PROBLEM','This combines estimators with a specified joint error model. It is not a formula for a general Bayesian posterior from two marginal densities.<br><br>Gaussianity is not needed for this linear minimum-variance result.',
  ['rumor','ci'],'Derive W by writing error eF = e1 + W(e2−e1), expanding its covariance, and completing the matrix square. The error-difference covariance S must be invertible for this displayed form.');

// 07
lab('correlation-lab','Live: confidence without independence','The actual error variance follows the joint covariance. A reported variance may tell a different story.','correlation',['ci','rumor']);

// 08
eqSlide('ci','03 · unknown cross-correlation','Covariance intersection: a guarded Gaussian merge','Choose fixed nonnegative weights summing to one; every input covariance must be a valid error bound.',
  raw`P_{\mathrm{CI}}^{-1}=\sum_iw_iP_i^{-1},\qquad \hat x_{\mathrm{CI}}=P_{\mathrm{CI}}\sum_iw_iP_i^{-1}\hat x_i`,
  'THE GUARANTEE — WITH ASSUMPTIONS','For unbiased estimates with individually consistent covariance bounds, CI gives a consistent error-covariance bound without knowing their cross-correlations.<br><br>It does not repair a biased or overconfident input model.',
  'THE WEIGHT IS NOT A VOTE ON THE MEAN',`A common choice minimizes ${inline(raw`\log\det P_{\mathrm{CI}}`)} or ${inline(raw`\operatorname{tr}P_{\mathrm{CI}}`)} over the simplex.<br><br>Weights act on <b>information matrices</b>. For unequal scalar variances, these objectives select the smaller-variance endpoint.`,
  ['ci'],'Assume SPD covariance matrices and fixed weights (or a setting where the requisite bounds hold conditionally). Avoid claiming arbitrary data-dependent selection preserves the same unconditional guarantee. General non-Gaussian GCI is discussed separately.');

// 09
lab('geometry-lab','Live: which directions carry information?','Rotate the covariance ellipses and compare the selected weight with the log-determinant optimum.','geometry',['ci']);

// 10
eqSlide('gci','03 · logarithmic pooling','GCI is a normalized geometric mean','Also called the logarithmic opinion pool or exponential-mixture density.',
  raw`q_G(x)=\frac{\prod_i p_i(x)^{w_i}}{Z_w},\qquad Z_w=\int\prod_i p_i(x)^{w_i}\,d\mu(x)`,
  'WHAT THE NORMALIZER SAYS',`For normalized inputs, ${inline(raw`0\le Z_w\le1`)}. If ${inline(raw`Z_w=0`)}, the normalized pool is undefined.<br><br>For positive weights, only the common support can survive. Zero-weight inputs are omitted.`,
  'GAUSSIAN SPECIAL CASE','Taking weighted sums of Gaussian log densities yields a quadratic. Completing the square gives exactly the CI mean and covariance.<br><br>Gaussian means can strongly disagree while the fused covariance stays unchanged.',
  ['kl','ci'],'Hölder’s inequality gives Zw ≤ 1. For two distinct strictly positive weights, equality requires equality of the normalized densities almost everywhere. GCI is idempotent: pooling identical densities returns that density, not a sharpened product.');

// 11
eqSlide('gci-derivation','03 · variational view','The direction of KL determines the pool','Define KL(a ∥ b) = ∫ a log(a/b) dμ. Optimize over normalized densities, not over a Gaussian approximation.',
  raw`\sum_i w_i\operatorname{KL}(q\Vert p_i)=\operatorname{KL}(q\Vert q_G)-\log Z_w`,
  'DERIVATION IN THREE LINES',`${inline(raw`\sum_iw_i=1`)} leaves one entropy term ${inline(raw`\int q\log q`)}.<br><br>The remaining term is ${inline(raw`-\int q\sum_iw_i\log p_i`)}.<br><br>Insert ${inline(raw`\log q_G=\sum_iw_i\log p_i-\log Z_w`)}.`,
  'RESULT AND DOMAIN',`${inline(raw`q_G=\arg\min_q\sum_iw_i\operatorname{KL}(q\Vert p_i)`)}.<br><br>Require ${inline(raw`Z_w>0`)} and the relevant integrability. A candidate assigning mass where an active input is zero has infinite KL.<br><br>The minimum value is ${inline(raw`-\log Z_w`)}.`,
  ['kl'],'The identity follows directly by substitution. Where KL quantities are infinite, use the common-support domain and avoid subtracting two infinities. This is an unrestricted-density minimization.');

// 12
eqSlide('aa','02 · arithmetic pooling','Reverse the KL arguments, obtain a mixture','AA = arithmetic average = linear opinion pool. Keep the full mixture before considering approximation.',
  raw`q_A(x)=\sum_iw_ip_i(x),\qquad q_A=\arg\min_q\sum_iw_i\operatorname{KL}(p_i\Vert q)`,
  'WHY IT IS THE MINIMIZER',`${inline(raw`\sum_iw_i\operatorname{KL}(p_i\Vert q)`)} differs from ${inline(raw`\operatorname{KL}(q_A\Vert q)`)} only by a constant independent of q.<br><br>The cross-entropy term is ${inline(raw`-\int q_A\log q`)} in both expressions.`,
  'WHAT IS PRESERVED','The active inputs’ support is united, not intersected. Distinct alternatives can remain as separate modes.<br><br>AA is a density mixture. It is not the same operation as averaging independent measurements or random estimates.',
  ['kl','aa'],'Assume the relevant quantities are finite when using a constant-difference identity. More generally the mixture minimizes the expected log loss. Restricting q to a parametric family can change the answer.');

// 13
lab('pooling-lab','Live: a mixture, an intersection, or a product?','Try conflicting mixtures and disjoint supports. No epsilon floor is used to hide a zero normalizer.','pooling',['kl','aa']);

// 14
eqSlide('moments','02 · density versus estimator','A mixture remembers disagreement','The covariance of a mixture has both within-source uncertainty and between-source spread.',
  raw`\mu_A=\sum_iw_i\mu_i,\qquad P_A=\sum_iw_i\left[P_i+(\mu_i-\mu_A)(\mu_i-\mu_A)^{\mathsf T}\right]`,
  'MOMENT MATCHING','Replacing AA by one Gaussian with these moments is an approximation to the density. It can erase multiple modes.<br><br>Averaging the covariances alone drops the disagreement term and is not moment matching.',
  'ESTIMATOR AVERAGING IS DIFFERENT',`For ${inline(raw`\hat x=\sum_iw_i\hat x_i`)}, the error covariance is ${inline(raw`\sum_{i,j}w_iw_jC_{ij}`)}, where ${inline(raw`C_{ij}=\mathbb E[e_ie_j^{\mathsf T}]`)}.<br><br>This depends on error correlations. It is not the AA density covariance above.`,
  ['aa','rumor'],'Apply the law of total covariance to a latent source index I with P(I=i)=wi. The mixture describes selecting a component then sampling x. The estimator formula instead averages jointly random errors.');

// 15
add('support','02–03 · limits','Agreement is not the same as truth','A fusion rule can behave exactly as designed and still express the wrong physical conclusion.',[
  ...panel('overlap',72,184,554,390,'SUPPORT AND CONFLICT',
    '<b>AA:</b> retains alternatives, including a false mode from one source.<br><br><b>GCI:</b> can suppress a true hypothesis absent from another source.<br><br><b>Gaussian GCI:</b> disagreement shifts the mean but does not inflate covariance. Inspect overlap and consistency, not only ellipse size.',20),
  ...panel('update',646,184,562,390,'COMMUTING WITH A COMMON UPDATE',
    `With fixed weights, logarithmic pooling commutes with applying the <b>same likelihood</b> L to all sources:<br><br>${inline(raw`G(\{Lp_i\})\propto L\,G(\{p_i\})`)}.<br><br>Each updated input is normalized first; constants cancel. Fixed-weight AA does not generally have this property.`,19),
  ...callout('External Bayesianity is an algebraic property, not a claim that every source contains independent data.')
],['kl','rumor'],'For GCI, ∏(L pi / ci)^wi ∝ L^(∑wi)∏pi^wi = L∏pi^wi. For AA, updating a mixture changes its component weights by the component evidences; keeping weights fixed generally breaks commutation. Assume all required normalizers are positive.');

// 16
eqSlide('wasserstein','04 · transport geometry','A barycenter moves mass instead of multiplying it','A useful alternative when the desired notion of “average” is geometric transport.',
  raw`q_W\in\arg\min_q\sum_iw_iW_2^2(q,p_i),\qquad \mu_W=\sum_iw_i\mu_i,\quad\sigma_W=\sum_iw_i\sigma_i\ \ (\text{1D Gaussian})`,
  'ONE-DIMENSIONAL GAUSSIANS',`The standard deviations ${inline(raw`\sigma_i`)} are averaged, not the variances.<br><br>The squared 2-Wasserstein distance is ${inline(raw`(\mu_1-\mu_2)^2+(\sigma_1-\sigma_2)^2`)}. This explains the scalar formula.`,
  'NOT AN EVIDENCE COMBINER','A transport barycenter is neither an independent-likelihood product nor a CI covariance guarantee.<br><br>In multiple dimensions, covariance barycenters generally require a matrix equation; averaging standard-deviation matrices is not a general shortcut.',
  ['transport'],'Finite second moments are required. In one dimension, W2 barycenters average quantile functions; Gaussian quantiles give the displayed mean and standard deviation. Multidimensional noncommuting covariance matrices require additional care.');

// 17
add('representations','implementation','The representation changes what is tractable','Separate the exact pooling rule from the approximation used to store or evaluate it.',[
  ...panel('mixtures',72,184,554,390,'GAUSSIAN MIXTURES',
    '<b>AA:</b> concatenate components and multiply each component weight by its source weight. This is exact before pruning or merging.<br><br><b>GCI:</b> fractional powers of mixtures are not generally Gaussian mixtures. Componentwise powering needs an explicit approximation.',20),
  ...panel('particles',646,184,562,390,'PARTICLES AND GRIDS',
    'Distinct empirical particle clouds do not automatically provide overlapping, evaluable continuous densities.<br><br>Use a common proposal, justified density reconstruction, or a dedicated fusion algorithm.<br><br>On grids, include quadrature weights and check domain truncation.',20),
  ...callout('A fractional power of a sum is not the sum of fractional powers. Do not silently change the algorithm.')
],['rfs','aa'],'For 0<w<1, (Σk ak φk)^w is not generally equal to Σk ak^w φk^w. A product of Dirac distributions is not a generic continuous-density fusion recipe. The included labs use analytic Gaussian formulas or explicit one-dimensional grid integration.');

// 18
eqSlide('sets','multi-object fusion','Now the unknown is a finite set','A random finite set X represents both an unknown number of objects and their states.',
  raw`\int f(X)\,\delta X=1,\qquad \int h(X)\,\delta X=h(\varnothing)+\sum_{n\ge1}\frac{1}{n!}\int h(\{x_1,\ldots,x_n\})\,dx_{1:n}`,
  'FULL MULTI-OBJECT DENSITY',`${inline(raw`f(X)`)} describes cardinality and joint object states. AA and GCI can be defined on this set space using the same variational ideas.<br><br>Every source must refer to the same physical object population and state space.`,
  'INTENSITY IS ONLY A FIRST MOMENT',`The probability hypothesis density ${inline(raw`D(x)`)} is an intensity:<br><br>${inline(raw`\int D(x)\,dx=\mathbb E[|X|]`)}.<br><br>Fusing D alone does not generally specify the full cardinality distribution or object dependence.`,
  ['rfs','aaRfs'],'The displayed finite-set integral is the usual FISST convention for an unlabeled continuous state space. For labeled RFSs, also sum over labels with a compatible reference measure and distinct-label constraint.');

// 19
eqSlide('bernoulli','multi-object · one possible object','Bernoulli fusion couples existence and location','For source i: fᵢ(∅) = 1 − rᵢ and fᵢ({x}) = rᵢpᵢ(x). Let η be the spatial geometric-overlap integral.',
  raw`\eta=\int p_1^w p_2^{1-w}\,dx,\qquad r_G=\frac{r_1^w r_2^{1-w}\eta}{(1-r_1)^w(1-r_2)^{1-w}+r_1^w r_2^{1-w}\eta}`,
  'GCI: NORMALIZE THE WHOLE SET DENSITY',`Conditional spatial density: ${inline(raw`p_G=p_1^wp_2^{1-w}/\eta`)} when ${inline(raw`\eta>0`)}.<br><br>The empty-set mass is part of the denominator. Spatial disagreement changes the fused probability of existence.`,
  'AA: AN EXACT BERNOULLI MIXTURE',`${inline(raw`r_A=wr_1+(1-w)r_2`)}.<br><br>${inline(raw`p_A=\frac{wr_1p_1+(1-w)r_2p_2}{r_A}`)} when ${inline(raw`r_A>0`)}.<br><br>The spatial mixture weights depend on existence, not just w.`,
  ['rfs','aaRfs'],'For 0<w<1, define a=(1−r1)^w(1−r2)^(1−w), b=r1^w r2^(1−w)η. If a+b=0, GCI is undefined. If η=0 but a>0, rG=0 and the conditional location density is irrelevant. At weight endpoints, return the active source rather than evaluating 0^0.');

// 20
lab('bernoulli-lab','Live: spatial conflict changes existence','Even two high-existence Bernoulli inputs can produce a low-existence geometric pool.','bernoulli',['rfs']);

// 21
eqSlide('poisson','multi-object · Poisson','Poisson GCI is closed in the Poisson family','A Poisson point process (PPP) is determined by its intensity, not by a unit-normalized spatial density alone.',
  raw`f_i(X)=e^{-\lambda_i}\prod_{x\in X}D_i(x),\qquad D_G(x)=\prod_iD_i(x)^{w_i},\quad\lambda_G=\int D_G(x)\,dx`,
  'THE OVERLAP CHANGES THE COUNT',`Write ${inline(raw`D_i=\lambda_ip_i`)} with normalized ${inline(raw`p_i`)} and positive rates. Then<br><br>${inline(raw`\lambda_G=\left(\prod_i\lambda_i^{w_i}\right)\int\prod_ip_i^{w_i}\,dx`)}.<br><br>Do not normalize ${inline(raw`D_G`)} to integrate to one.`,
  'WHY CLOSURE HOLDS','The unnormalized geometric product has a constant times a product of pointwise intensities. Set normalization turns that constant into exp(−λG).<br><br>If the intensity overlap vanishes, the result can be the valid empty PPP, unlike a zero-overlap single-state pool.',
  ['rfs'],'The set normalizer is exp(−Σi wi λi + λG), which is positive for finite rates even if λG=0. Weighted Hölder bounds λG for integrable intensities. Zero-rate and zero-weight cases need endpoint conventions.');

// 22
eqSlide('poisson-aa','multi-object · projection','AA of Poisson densities is usually not Poisson','The arithmetic intensity average is exact as a first moment, not as a full-density closure claim.',
  raw`f_A=\sum_iw_if_i,\quad D_A=\sum_iw_iD_i,\quad \operatorname{Var}_{f_A}(|X|)=\bar\lambda+\sum_iw_i(\lambda_i-\bar\lambda)^2`,
  'A MIXTURE OF PPPs',`${inline(raw`\bar\lambda=\sum_iw_i\lambda_i`)} is the mean count. A latent source choice also contributes between-source count variation.<br><br>The full mixture generally has object dependencies absent from a single PPP.`,
  'AN INTENSITY-MATCHED PPP','Approximating the mixture by a PPP with intensity DA preserves its first moment.<br><br>Its count variance becomes λ̄. This drops the additional mixture variance and does not preserve every feature of fA.',
  ['aaRfs'],'Use the law of total variance with N|I=i ∼ Poisson(λi). Even equal λi do not generally make the full mixture a PPP when spatial intensities differ; matching count moments alone is insufficient.');

// 23
add('association','multi-object · practical boundary','A fusion rule cannot solve identity by itself','Track labels are local bookkeeping unless the system explicitly makes them globally meaningful.',[
  ...panel('identity',72,184,554,390,'ASSOCIATE AND ALIGN',
    'Two sources may assign different labels to the <b>same</b> object, or the same label to <b>different</b> objects.<br><br>Align coordinates, synchronize time, and account for association uncertainty.<br><br>Do not merge two different targets as though they were two estimates of one target.',20),
  ...panel('visibility',646,184,562,390,'MODEL THE FIELD OF VIEW',
    '“Not detected” is not equivalent to “does not exist,” especially outside a sensor’s field of view.<br><br>GCI can suppress a track supported only by one source. AA can retain that track, but can also retain clutter alternatives.<br><br>The remedy depends on the sensing model.',19),
  ...callout('Fuse existence, location, cardinality, and identity consistently—not only Gaussian means.')
],['rfs','aaRfs'],'This slide identifies modeling requirements, not a complete labeled-RFS association algorithm. Field-of-view-aware fusion may require regional models or other explicit treatment of unseen states.');

// 24
eqSlide('consensus','networks','Consensus reaches agreement, not new evidence','Repeated local pooling redistributes existing beliefs across the communication graph.',
  raw`q_i^{(t+1)}(x)\propto\prod_j\left[q_j^{(t)}(x)\right]^{a_{ij}},\qquad q_\infty(x)\propto\prod_j\left[q_j^{(0)}(x)\right]^{\pi_j}`,
  'WHEN THE LIMIT HAS THIS FORM',`For a fixed primitive row-stochastic matrix ${inline(raw`A=[a_{ij}]`)}, ${inline(raw`A^t\to\mathbf1\pi^{\mathsf T}`)}.<br><br>The stationary vector ${inline(raw`\pi`)} sets the final source weights. A doubly stochastic A gives uniform weights under these assumptions.`,
  'WHAT A ROUND DOES NOT DO','Another communication round is not another independent measurement.<br><br>Geometric consensus is idempotent on identical beliefs. An uncorrected product of recycled posteriors can instead create unjustified confidence.',
  ['consensus','rumor'],'Assume common support and finite nonzero normalizers. In log-density ratios relative to a reference state, normalization constants cancel and the iteration is linear in A. Changing graphs, finite rounds, and new measurements require their own analysis.');

// 25
lab('rumors-lab','Live: twenty messages, one piece of information','Switch the true error model while keeping the reported marginal variances unchanged.','rumors',['rumor','ci']);

// 26
add('numerics','implementation checklist','Make the numerical result mean what it claims','The six experiments are deterministic illustrations—not a benchmark, a full tracker, or an empirical calibration study.',[
  ...panel('numerical',72,184,554,390,'NUMERICAL DISCIPLINE',
    'Use linear solves or Cholesky factors for matrix equations in production code.<br><br>Evaluate geometric products in the log domain; include integration weights.<br><br>Report zero support overlap and truncation explicitly. Never add an unreported density floor.',20),
  ...panel('model-check',646,184,562,390,'MODEL AND VALIDATION',
    'Record data provenance, common priors, and recycled messages.<br><br>Check normalization, idempotence, weight endpoints, and covariance positivity.<br><br>Validate uncertainty against ground truth under the intended error model—not just visually pleasing ellipses.',20),
  ...callout('Every demo has resettable controls; its math is separate from drawing code and covered by executable tests.')
],['prior','ci','aa'],'Small explicit 2x2 inverses in the geometry lab are transparent teaching formulas, not the recommended large-matrix implementation. The density lab uses a fixed uniform grid [-12,12] with 1201 samples and trapezoidal integration.');

// 27
add('takeaways','decision guide','Start with what is known about the information','There is no single rule that simultaneously recovers all private evidence, ignores unknown overlap, and preserves every alternative.',[
  ...[['KNOWN COMMON INFORMATION','Use the Bayesian factorization; remove the actual shared contribution.','bayes'],
      ['KNOWN JOINT ERROR COVARIANCE','Use correlation-aware linear fusion within the estimator model.','known-correlation'],
      ['UNKNOWN CROSS-CORRELATION','CI is an option for unbiased, individually consistent estimates.','ci'],
      ['ALTERNATIVES MUST REMAIN VISIBLE','Keep the AA mixture; label any moment-matched approximation.','aa'],
      ['THE UNKNOWN IS A SET OF OBJECTS','Fuse the full multi-object model or identify the exact projection.','sets']].flatMap(([title,body,target],i)=>[
        rect(`decision-${i}`,72,182+i*89,1136,77,i%2?C.panel:C.soft),
        text(`decision-label-${i}`,91,195+i*89,315,53,title,12,{fontFamily:MONO,fontWeight:800,color:C.green,link:target}),
        text(`decision-text-${i}`,420,199+i*89,760,45,body+' ↗',18,{link:target})])
],['prior','ci','aaRfs'],'Transport barycenters answer a different geometric averaging question; see the separate chapter. Unknown dependence makes exact evidence fusion unidentifiable from marginal densities alone without additional structure.');

// 28–29 — Clickable primary references, including explanatory provenance.
function referenceSlide(id,title,keys){
  add(id,'reading and reproducibility',title,'Click a reference to open the primary paper or its author-hosted preprint.',keys.flatMap((k,i)=>[
    rect(`ref-${k}`,72,183+i*85,1136,74),
    text(`ref-number-${k}`,92,201+i*85,35,25,String(i+1).padStart(2,'0'),14,{fontFamily:MONO,color:C.green,fontWeight:800}),
    text(`ref-text-${k}`,145,196+i*85,1035,56,refs[k].title,16,{link:refs[k].url,lineHeight:1.35})
  ]),[],keys.map(k=>refs[k].url).join('\n'));
}
referenceSlide('references','Foundations: information, correlation, transport',['prior','ci','rumor','kl','transport']);
referenceSlide('references-rfs','Multi-object densities and distributed consensus',['rfs','aa','aaRfs','consensus']);
slides.at(-1).elements.push(text('source-code',93,553,1090,53,'Readable source, numerical kernels, and tests ↗',20,{color:C.green,link:'https://github.com/BaiLiping/bailiping.github.io/tree/main/multidensity-fusion'}));

for(const [i,slide] of slides.entries()){
  slide.elements.unshift({...rect('progress-track',0,0,1280,4,C.rule),strokeWidth:0,radius:0},
    {...rect('progress-fill',0,0,1280*(i+1)/slides.length,4,C.green),strokeWidth:0,radius:0});
  slide.elements.push(text('footer-page',1120,687,86,20,`${String(i+1).padStart(2,'0')} / ${slides.length}`,11,{fontFamily:MONO,color:C.muted,align:'right'}));
}
export const deck={format:'bento/slides',version:1,docId:'multidensity-fusion-bento',title:'Multidensity Fusion',readonly:true,
  meta:{author:'Bai Liping',subject:'Probability and multi-object density fusion: principles, assumptions, and six interactive experiments',company:'bailiping.com'},
  size:{width:1280,height:720},theme:{background:C.paper,color:C.ink,accent:C.green,fontFamily:SANS},slides};
export const inlineLiveMap=live;
