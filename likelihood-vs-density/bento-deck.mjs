/** Original teaching content; same 1280×720 Bento schema as the other Random Thoughts decks. */
const C={paper:'#F7F5EF',panel:'#FFFEFB',ink:'#203129',muted:'#66756E',rule:'#D8DED7',green:'#2F6B4F',soft:'#E7F0EA',rust:'#A94F2A'};
const sans="Inter, ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif",serif="Georgia, 'Times New Roman', serif",mono="Consolas, monospace",r=String.raw;
const I=s=>`<span class="math-tex math-inline">\\(${s}\\)</span>`,D=s=>`<span class="math-tex math-display">\\[${s}\\]</span>`;
const T=(id,x,y,w,h,html,size=20,o={})=>({id,type:'text',x,y,w,h,html,fontSize:size,fontFamily:sans,fontWeight:400,color:C.ink,lineHeight:1.35,align:'left',valign:'top',rotation:0,opacity:1,...o});
const R=(id,x,y,w,h,fill=C.panel)=>({id,type:'shape',shape:'rect',x,y,w,h,fill,stroke:C.rule,strokeWidth:1,radius:13,rotation:0,opacity:1});
const E=(id,x,y,w,h,tex,size=26)=>T(id,x,y,w,h,D(tex),size,{align:'center',valign:'middle'});
const box=(id,x,y,w,h,label,body,size=20,fill=C.panel)=>[R(id+'-bg',x,y,w,h,fill),T(id+'-label',x+22,y+17,w-44,26,label,12,{fontFamily:mono,fontWeight:800,color:C.green}),T(id+'-body',x+22,y+59,w-44,h-76,body,size)];
const call=s=>[R('take-bg',72,585,1136,53,C.soft),T('take',92,599,1096,29,s,16,{color:C.green,fontWeight:600})];
export const refs={
 density:{label:'Piech · continuous distributions',url:'https://chrispiech.github.io/probabilityForComputerScientists/en/part2/continuous/',title:'Chris Piech, Stanford University. Probability for Computer Scientists: Continuous Distribution.'},
 mle:{label:'Piech · maximum likelihood',url:'https://chrispiech.github.io/probabilityForComputerScientists/en/part5/mle/',title:'Chris Piech, Stanford University. Probability for Computer Scientists: Maximum Likelihood Estimation.'},
 bayes:{label:'Piech · MAP and priors',url:'https://chrispiech.github.io/probabilityForComputerScientists/en/part5/map/',title:'Chris Piech, Stanford University. Probability for Computer Scientists: Maximum A Posteriori.'},
 transform:{label:'Stan · change of variables',url:'https://mc-stan.org/docs/stan-users-guide/reparameterization.html',title:'Stan Development Team. Stan User’s Guide: Reparameterization and Change of Variables.'},
 fusion:{label:'Wu et al. · shared priors',url:'https://arxiv.org/abs/2212.07311',title:'Peng Wu, Tales Imbiriba, Víctor Elvira & Pau Closas. Bayesian Data Fusion With Shared Priors. IEEE Transactions on Signal Processing 72, 275–288 (2024); arXiv:2212.07311.'}
};
const slides=[],live=[];
function add(id,section,title,sub,content,sources=[],notes=''){
 slides.push({id,background:C.paper,transition:slides.length?'morph':'none',notes:notes+'\nSources:\n'+sources.map(k=>refs[k].title+' '+refs[k].url).join('\n'),elements:[
 T('eyebrow',72,34,1136,22,'RANDOM THOUGHTS / '+section.toUpperCase(),11,{fontFamily:mono,color:C.green,fontWeight:800,letterSpacing:1.3}),
 T('heading',72,69,1136,59,title,38,{fontFamily:serif,fontWeight:700,lineHeight:1.1}),
 T('subtitle',74,133,1132,42,sub,17,{color:C.muted}),...content,
 ...sources.map((k,i)=>T('ref-'+k,74+365*i,650,350,20,refs[k].label+' ↗',10,{color:C.muted,link:refs[k].url})),
 R('footer-rule',72,676,1136,1,C.rule),T('footer-brand',74,689,500,18,'BAI LIPING · LIKELIHOOD & DENSITY',10,{color:C.muted,fontFamily:mono}),
 T('footer-map',916,689,184,18,'CHAPTER MAP ↗',10,{align:'right',color:C.green,link:'map'})]});
}
function two(id,section,title,sub,lTitle,lBody,rTitle,rBody,take,sources=[],notes=''){
 add(id,section,title,sub,[...box('a',72,187,552,373,lTitle,lBody),...box('b',646,187,562,373,rTitle,rBody),...call(take)],sources,notes);
}
function eq(id,section,title,sub,tex,lTitle,lBody,rTitle,rBody,sources=[],notes=''){
 add(id,section,title,sub,[R('eq-bg',72,188,1136,137,C.soft),E('eq',94,203,1092,107,tex,25),...box('a',72,346,552,288,lTitle,lBody,19),...box('b',646,346,562,288,rTitle,rBody,19)],sources,notes);
}
function lab(id,title,sub,demo){
 add(id,'live experiment',title,sub,[R('lab-bg',72,185,1136,450)],['mle','bayes'],'Deterministic illustration. No statistical calibration claim. The controls change a model or an observed data summary; no data are randomly resampled. Open live/?demo='+demo+' directly.');
 live.push({slide:id,slideIndex:slides.length-1,inline:true,layout:'region',bounds:{x:74,y:186,width:1132,height:448},src:'./live/?demo='+demo+'&embed=region',source:'./live/?demo='+demo,title,hideSource:true,readyMessage:true,unloadWhenHidden:false});
}
add('overview','a probability primer','Likelihood & Density','One formula. Two directions. A third object—the posterior—requires a prior.',[
 T('hero',75,211,1080,155,'Same formula.<br>Different question.',63,{fontFamily:serif,fontWeight:700,lineHeight:1.05}),
 T('hero-eq',78,391,1070,58,I(r`f(y\mid\theta)`)+ ' does not tell you which argument is varying.',28,{color:C.muted}),
 ...[['01','Possible data','density'],['02','Candidate parameters','likelihood'],['03','Posterior beliefs','bayes']].flatMap(([n,s,link],i)=>[R('cover-'+i,72+i*386,486,365,91,i===1?C.soft:C.panel),T('covertext-'+i,93+i*386,504,325,59,n+'<br><b>'+s+' ↗</b>',18,{link})]),
 T('guide',78,605,1100,30,'Arrow keys: navigate · Three live experiments · Core explanation first; technical extensions near the end.',15,{color:C.muted})
],['density','mle','bayes'],'Prerequisites: basic integrals and conditional probability. Lowercase y is observed data, uppercase Y is the random observation; theta is the unknown parameter or state. All algebraic numerical examples are original derivations.');
add('map','chapter map','Keep track of the variable','The same expression can be used in different mathematical roles.',[
 ...[['01 · DENSITY','Fix θ. Vary the data y.','Where can a future observation fall?','density'],['02 · LIKELIHOOD','Fix the observed y. Vary θ.','Which candidate parameters fit these data better?','likelihood'],['03 · POSTERIOR','Fix y. Put a distribution on θ.','What are the probabilities after incorporating a prior?','bayes'],['04 · CONNECTIONS','Estimation, sensors, and fusion.','Why likelihood factors are not posterior densities.','fusion']].flatMap(([a,b,c,link],i)=>box('map-'+i,72+(i%2)*576,187+Math.floor(i/2)*205,560,188,a,'<b>'+b+'</b><br><br>'+c+' <a href="#'+link+'">↗</a>',18)),
 T('appendix-link',80,612,1110,27,'Deeper checks: begin with parameter transformations ↗',16,{color:C.green,link:'transform'})
],['density','mle','bayes']);
eq('density','01 · probability density','Probability is area, not height','Start with a continuous observation Y and a fixed parameter θ.',
 r`\Pr_\theta(a\le Y\le b)=\int_a^b f(y\mid\theta)\,dy,\qquad \int_{\mathbb R} f(y\mid\theta)\,dy=1`,
 'WHAT THE DENSITY DOES','It allocates probability across possible values of <b>Y</b>.<br><br>For a small interval of width Δ, its probability is approximately density × Δ, when the density varies little there.',
 'HEIGHTS CAN EXCEED ONE',`${I(r`Y\sim\mathrm{Uniform}(0,0.2)`)} has density 5 inside its support.<br><br>${I(r`\Pr(0.05\le Y\le0.10)=5(0.05)=0.25`)}.<br><br>The area is a probability; the height is not.`,['density'],
 'A nonnegative Lebesgue density integrates to one. Under an absolutely continuous model P(Y=y)=0 for each exact y. The small-bin approximation assumes local regularity; finite-resolution observations can instead be modeled by their bin probabilities.');
two('discrete','01 · discrete versus continuous','Likelihood works for counts and measurements','A probability mass function (PMF) is not the same as a continuous probability density (PDF).',
 'DISCRETE: SUM PROBABILITIES',`${I(r`K\sim\mathrm{Binomial}(n,\theta)`)}<br><br>${I(r`\Pr(K=k\mid\theta)=\binom nk\theta^k(1-\theta)^{n-k}`)}<br><br>${I(r`\sum_{k=0}^{n}\Pr(K=k\mid\theta)=1`)}.<br><br>An exact count can have positive probability.`,
 'CONTINUOUS: INTEGRATE A DENSITY',`${I(r`Y\mid\theta\sim\mathcal N(\theta,\sigma^2)`)}<br><br>${I(r`\int f(y\mid\theta)\,dy=1`)}.<br><br>Exact points have zero probability; intervals have probability.<br><br>A density has units inverse to its data variable.`,
 'Both a PMF and a PDF become a likelihood when the observed data are fixed and the parameter varies.',['density','mle']);
eq('likelihood','02 · the central distinction','Freeze the data. Let the parameter move.','Observing y changes how we use the sampling model; it does not reverse the conditional probability.',
 r`\underbrace{y\mapsto f(y\mid\theta_0)}_{\text{density in data}}\qquad\qquad\underbrace{\theta\mapsto L(\theta;y_{\rm obs})=f(y_{\rm obs}\mid\theta)}_{\text{likelihood in parameter}}`,
 'DENSITY QUESTION','Assuming θ = θ₀, how is probability distributed across possible measurements y?<br><br>The data argument varies. Normalization is over the data space.',
 'LIKELIHOOD QUESTION','For the measurement already observed, how do candidate θ values compare?<br><br>The parameter argument varies. There is <b>no requirement</b> that its integral over θ be one.',['mle'],
 'L(theta;y) is not p(theta|y). The semicolon emphasizes that y is fixed. In a frequentist model f_theta(y) is also valid notation; writing a conditional does not by itself give theta a prior distribution. A likelihood can accidentally have unit integral in a particular coordinate without automatically becoming a posterior.');
add('coin-table','02 · a finite view','Read across for probabilities; down for likelihoods','Illustrative model: two independent coin flips. θ is the chance of heads; K is the number of heads.',[
 R('table-bg',72,187,686,374),
 T('table',100,207,630,329,`<table style="width:100%;border-collapse:collapse;text-align:center;font-size:23px"><thead><tr><th style="padding:13px">θ</th><th>K = 0</th><th>K = 1</th><th style="background:#e7f0ea">K = 2</th></tr></thead><tbody>${[[.2,.64,.32,.04],[.5,.25,.50,.25],[.8,.04,.32,.64]].map(row=>'<tr>'+row.map((x,j)=>`<td style="padding:20px 10px;border-top:1px solid #d8ded7;${j===3?'background:#e7f0ea':''}">${x.toFixed(2)}</td>`).join('')+'</tr>').join('')}</tbody></table>`,22),
 ...box('table-explain',784,187,424,374,'NOW OBSERVE K = 2','Each <b>row</b> sums to 1.<br><br>The highlighted column gives likelihoods: 0.04, 0.25, 0.64.<br><br>These sum to 0.93—not 1.<br><br>They are not posterior probabilities for the three θ values.',20),
 ...call('Normalizing that column gives a posterior only after assuming equal prior mass on exactly these candidates.')
],['mle','bayes'],'The table lists only three possible parameter values, not a prior distribution. If the parameter space is restricted to these three candidates with equal prior mass, the posterior is (4,25,64)/93. A continuous uniform prior on [0,1] is a different model.');
eq('coin','02 · coin example','Seven heads out of ten: a curve over θ','Assume K | θ ~ Binomial(10, θ), and observe K = 7.',
 r`L(\theta;7)=120\theta^7(1-\theta)^3,\quad \widehat\theta_{\rm ML}=0.7,\quad \int_0^1 L(\theta;7)\,d\theta=\frac1{11}`,
 'WHERE THE PEAK COMES FROM',`${I(r`\ell(\theta)=\log120+7\log\theta+3\log(1-\theta)`)}.<br><br>Set ${I(r`\ell'(\theta)=7/\theta-3/(1-\theta)=0`)}.<br><br>The solution is 0.7; the second derivative is negative in the interior.`,
 'WHAT THE PEAK DOES NOT SAY','It does <b>not</b> say there is a 70% posterior probability that θ is correct.<br><br>θ = 0.7 is the best-fitting candidate under this model. Uncertainty about θ needs an inferential procedure, not just a peak.',['mle'],
 'Integral: binom(10,7) B(8,4)=120·7!·3!/11!=1/11. The likelihood for one particular ordered 7-head, 3-tail sequence omits 120; it has the same shape but a different scale.');
lab('coin-lab','Live: change the data or change the model','The left plot fixes a candidate θ. The right plot fixes the observed head count.','coin');
eq('gaussian','02 · a Gaussian sensor','A bell shape can hide the distinction','Known sensor model: Y | θ ~ N(bθ, σ²), with b > 0 and σ > 0.',
 r`f(y\mid\theta)=\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(y-b\theta)^2}{2\sigma^2}},\qquad \int f(y\mid\theta)dy=1,\quad\int L(\theta;y)d\theta=\frac1b`,
 'VARY THE MEASUREMENT','Hold θ fixed. The density in y is centered at bθ with standard deviation σ.<br><br>Changing θ moves the model’s predicted measurement.',
 'VARY THE STATE','Hold y fixed. The likelihood peaks at y/b and has width σ/b in θ.<br><br>Its area is 1/b. For b = 1, the area happens to be one; that is not a general likelihood property.',['density','mle'],
 'Substitute u=(y−bθ)/σ and use dθ=(σ/b)du. For b<0 the integral is 1/|b|; for b=0 the likelihood is constant in theta and has infinite integral over R. A flat prior in theta yields a normalized Gaussian posterior for b nonzero, but the prior choice must be stated.');
lab('gaussian-lab','Live: the same sensor, two different axes','Move the sensor gain b. The density stays normalized in y; the likelihood area in θ changes.','gaussian');
eq('ratios','02 · relative evidence','A likelihood ratio is not a posterior probability','Compare two specified candidates under the same data model and reference measure.',
 r`\frac{L(\theta_a;y)}{L(\theta_b;y)}\quad\text{compares fit};\qquad\frac{\Pr(\theta_a\mid y)}{\Pr(\theta_b\mid y)}=\frac{L(\theta_a;y)}{L(\theta_b;y)}\frac{\Pr(\theta_a)}{\Pr(\theta_b)}`,
 'COIN EXAMPLE: 7 HEADS / 10',`${I(r`L(0.7;7)/L(0.5;7)\approx2.28`)}.<br><br>The observed result favors θ = 0.7 over θ = 0.5 by that likelihood ratio.<br><br>This is not a probability of 2.28 or 228%.`,
 'PRIOR ODDS STILL MATTER','With only these two candidates and equal prior odds, posterior odds are about 2.28 : 1.<br><br>With prior odds 1 : 9, posterior odds are about 0.253 : 1. The same evidence need not reverse the prior preference.',['mle','bayes'],
 'Posterior-odds formula here is for discrete candidate hypotheses of positive prior mass. For continuous theta, compare posterior densities at points, not probabilities of singleton points. The numerical likelihood ratio is (0.7/0.5)^7(0.3/0.5)^3=2.2769316864.');
eq('constants','02 · scale and log likelihood','“Up to a constant” means constant in θ','Keep the observed data and statistical model fixed when dropping multiplicative factors.',
 r`\widetilde L(\theta;y)=c(y)L(\theta;y),\ c(y)>0\quad\Longrightarrow\quad \frac{\widetilde L(\theta_a;y)}{\widetilde L(\theta_b;y)}=\frac{L(\theta_a;y)}{L(\theta_b;y)}`,
 'SAFE FOR PARAMETER COMPARISONS','A positive, parameter-independent factor cancels in likelihood ratios, MLE, and a normalized posterior with a fixed prior.<br><br>Use log likelihoods for products: multiplication becomes addition.',
 'NOT EVERY NORMALIZER IS CONSTANT','For a Gaussian with <b>unknown σ</b>, the factor 1/σ depends on the parameter and must stay.<br><br>Evidence values and comparisons across models also require the appropriate constants.',['mle'],
 'Do not discard parameter-dependent support indicators either. Example: Uniform(0,theta) likelihood is theta^(−n) times 1{theta>=max y_i} for nonnegative data; that indicator changes the feasible parameter values. A density need not integrate to one after arbitrary rescaling, although the rescaled likelihood is equivalent for within-model parameter inference.');
eq('bayes','03 · posterior density','A prior turns relative fit into posterior belief','The likelihood still describes the data model. Bayes’ rule creates a different density, over θ.',
 r`\pi(\theta\mid y)=\frac{L(\theta;y)\pi(\theta)}{m(y)},\qquad m(y)=\int L(u;y)\pi(u)\,du`,
 'THREE DIFFERENT ROLES',`<b>Prior</b> ${I(r`\pi(\theta)`)}: parameter uncertainty before these data.<br><br><b>Likelihood</b> ${I(r`L(\theta;y)`)}: data fit for each θ.<br><br><b>Posterior</b> ${I(r`\pi(\theta\mid y)`)}: updated parameter uncertainty.`,
 'THE NORMALIZATION IS OVER θ',`${I(r`\int\pi(\theta\mid y)\,d\theta=1`)} when ${I(r`0<m(y)<\infty`)}.<br><br>m(y) is the prior-predictive density or probability mass of the data—not the posterior density of θ.`,['bayes'],
 'For continuous observations, m(y) is a density value, not P(Y=y). For a discrete parameter space replace the parameter integral by a sum. Assume compatible dominating measures and a proper prior unless an improper-prior calculation is explicitly justified.');
eq('beta','03 · conjugate example','Same likelihood. Different prior. Different posterior.','For a coin, a Beta prior makes the update visible in its exponents.',
 r`\pi(\theta)\propto\theta^{\alpha-1}(1-\theta)^{\beta-1}\quad\Longrightarrow\quad\theta\mid K=k\sim\mathrm{Beta}(\alpha+k,\beta+n-k)`,
 'UNIFORM PRIOR: BETA(1,1)','After 7 heads and 3 tails:<br><br><b>Posterior: Beta(8,4).</b><br><br>Posterior mode = 0.7.<br>Posterior mean = 8/12 ≈ 0.667.',
 'CENTERED PRIOR: BETA(10,10)','After the same 7 heads and 3 tails:<br><br><b>Posterior: Beta(17,13).</b><br><br>Posterior mode = 16/28 ≈ 0.571.<br>Posterior mean = 17/30 ≈ 0.567.',['bayes'],
 'Beta(a,b) density is theta^(a−1)(1−theta)^(b−1)/B(a,b), for theta in (0,1). Parameters alpha,beta>0. Mean=a/(a+b); unique interior mode=(a−1)/(a+b−2) requires a,b>1. All posterior examples here have an interior mode.');
lab('prior-lab','Live: a likelihood is not yet a posterior','Change the Beta prior while holding the observed heads and tails fixed. Watch which curve moves.','prior');
eq('estimates','03 · density versus point estimate','MLE, MAP, and the posterior mean differ','The first two select peaks. The third averages under a posterior distribution.',
 r`\widehat\theta_{\rm ML}\in\arg\max_\theta L(\theta;y),\qquad \widehat\theta_{\rm MAP}\in\arg\max_\theta \pi(\theta\mid y),\qquad \widehat\theta_{\rm mean}=\int\theta\pi(\theta\mid y)d\theta`,
 'WHERE THE PRIOR ENTERS','MLE optimizes the likelihood alone.<br><br>MAP optimizes likelihood × prior density in the chosen coordinates.<br><br>The posterior mean uses the full posterior, not only its maximum.',
 'WHAT “OPTIMAL” CAN MEAN',`${I(r`\mathbb E[(\Theta-a)^2\mid y]=\operatorname{Var}(\Theta\mid y)+(a-\mathbb E[\Theta\mid y])^2`)}.<br><br>Thus the posterior mean minimizes posterior expected squared error, when the second moment is finite.`,['mle','bayes'],
 'This is a loss-specific Bayes action, not a universal best estimate. A continuous MAP is not generally invariant under smooth reparameterization, whereas MLE maximizers map equivariantly under a bijection when no parameter Jacobian is introduced. An estimator’s sampling properties require separate analysis.');
eq('independence','04 · repeated evidence','Multiply likelihoods only with the right factorization','Conditional independence given θ is enough; unconditional independence is not required.',
 r`L(\theta;y_{1:n})=f(y_{1:n}\mid\theta)=\prod_{i=1}^{n}f_i(y_i\mid\theta)\quad\text{under joint conditional independence}`,
 'GAUSSIAN REPEATS: KNOWN VARIANCE',`${I(r`Y_i\mid\theta\sim\mathcal N(\theta,\sigma^2)`)} independently.<br><br>${I(r`\sum_i(y_i-\theta)^2=\sum_i(y_i-\bar y)^2+n(\theta-\bar y)^2`)}.<br><br>The likelihood peaks at ${I(r`\bar y`)} and its Gaussian-kernel width is ${I(r`\sigma/\sqrt n`)}.`,
 'DUPLICATING A FILE IS NOT A NEW SAMPLE','For dependent observations use the joint likelihood—or a valid chain of conditional factors.<br><br>Multiplying the same evidence repeatedly changes the model and can create artificial precision.',['mle','fusion'],
 'Joint conditional independence, not merely pairwise independence, is needed for a full product over n>2. The general chain rule is f(y1:n|theta)=product_i f(yi|y1:i−1,theta). The Gaussian kernel width is not automatically a posterior standard deviation for arbitrary priors.');
eq('fusion','04 · back to density fusion','Posterior densities already contain a prior','This is why “multiply the densities” can count the same information twice.',
 r`p_i(\theta)\propto\pi_0(\theta)L_i(\theta)\quad\Longrightarrow\quad p_F(\theta)\propto\pi_0(\theta)\prod_i L_i(\theta)\propto\frac{\prod_i p_i(\theta)}{\pi_0(\theta)^{M-1}}`,
 'WHEN THIS IDENTITY APPLIES','The M sources share a common prior and have conditionally independent private observations given θ.<br><br>Work on the relevant support where the prior is positive; normalize the fused posterior.',
 'DO NOT CONFUSE FACTORS WITH BELIEFS','A likelihood is an evidence factor in θ.<br><br>A posterior is a normalized belief about θ.<br><br>AA and geometric density pooling answer other fusion objectives; they do not automatically reproduce independent evidence fusion.',['fusion'],
 'More shared information than the initial prior, such as shared measurements or recirculated messages, requires accounting for the actual common information. The displayed power of pi0 is not valid for arbitrary overlapping data histories. Related deck: ../multidensity-fusion/.');
eq('tracking','04 · state estimation','A measurement density is not a state density','In tracking notation, the unknown parameter becomes a state x and the observation becomes z.',
 r`g(z\mid x)=\mathcal N(z;Hx,R),\qquad p(x\mid z)\propto g(z\mid x)p^-(x)`,
 'A LIKELIHOOD MAY HAVE AN INFINITE AREA',`Let ${I(r`x=(x_1,x_2)`)} and ${I(r`z=x_1+v`)}.<br><br>${I(r`g(z\mid x)=\mathcal N(z;x_1,R)`)} is constant in x₂.<br><br>${I(r`\int_{\mathbb R^2}g(z\mid x)\,dx=\infty`)}.<br>The sensor alone leaves one direction unconstrained.`,
 'THE PRIOR PROVIDES MISSING STRUCTURE','A proper prior over both state components can produce a proper posterior.<br><br>For Gaussian priors, the precision update is:<br><br>'+I(r`P_+^{-1}=P_-^{-1}+H^{\mathsf T}R^{-1}H`),['mle','bayes'],
 'Assume R is positive definite and known, and Pminus is positive definite for the inverse form. g is normalized over measurement space z for each fixed x. Under a fixed finite-variance Gaussian sensor, the likelihood is positive and bounded, so a proper prior gives finite positive evidence.');
eq('transform','deeper check · coordinates','Reparameterization: where the Jacobian belongs','Here we reparameterize θ, not the observed data. Let φ = g(θ) be a smooth bijection.',
 r`L_\phi(\phi;y)=L_\theta(g^{-1}(\phi);y),\qquad \pi_\phi(\phi\mid y)=\pi_\theta(g^{-1}(\phi)\mid y)\left|\frac{d\theta}{d\phi}\right|`,
 'LIKELIHOOD: RELABEL CANDIDATES','The same physical model gets a new parameter name.<br><br>Its data fit does not acquire a parameter-volume factor.<br><br>Adding a θ-to-φ Jacobian to the likelihood would change the parameter comparison.',
 'POSTERIOR: PRESERVE PROBABILITY MASS','Probability in a region must be unchanged.<br><br>'+I(r`\pi_\theta(\theta\mid y)d\theta=\pi_\phi(\phi\mid y)d\phi`)+'.<br><br>Density heights—and generally MAP locations—depend on coordinates.',['transform','mle'],
 'For a parameter-independent invertible DATA transformation z=h(y), the sampling density does acquire |dy/dz|. At fixed observed data this is independent of theta, so likelihood ratios are unchanged. For vector parameters use the absolute determinant of the Jacobian. Distinguish this from reparameterizing theta.');
two('normalization','deeper check · priors','Why not just normalize the likelihood?','It may be possible numerically, but the interpretation and choice of parameter measure still matter.',
 'WHEN THE INTEGRAL IS FINITE',`${I(r`q(\theta)=L(\theta;y)/\int L(u;y)du`)}<br><br>This defines a density relative to dθ.<br><br>It agrees with a formal flat-prior posterior in <b>that coordinate</b>, when the posterior is proper.<br><br>A flat prior in θ is not generally flat in g(θ).`,
 'WHEN THE INTEGRAL DIVERGES',`One exponential observation y > 0 with unknown scale s > 0:<br><br>${I(r`L(s;y)=s^{-1}e^{-y/s}`)}.<br><br>As s grows, ${I(r`L(s;y)\sim1/s`)}. Its integral over s diverges.<br><br>There is no normalized likelihood in ds here.`,
 'Normalization is a mathematical operation; a prior and a reference measure provide its inferential meaning.',['transform','bayes'],
 'For the exponential example the MLE s=y still exists. With rate lambda=1/s, L(lambda;y)=lambda exp(−lambda y) has integral 1/y², which is finite. Normalizing in ds and in dlambda is not a parameterization-invariant prescription. A constant prior over R or R+ is improper, and evidence-based model comparison is not defined by its unspecified scaling.');
add('summary','takeaway','Before using p(·), ask: density in what?','These objects can share algebraic factors while answering different questions.',[
 ...[['SAMPLING DENSITY',r`y\mapsto f(y\mid\theta)`,'θ fixed; probability is allocated across data y.'],['LIKELIHOOD',r`\theta\mapsto f(y_{\rm obs}\mid\theta)`,'Observed y fixed; compare candidate θ values.'],['POSTERIOR DENSITY',r`\theta\mapsto\pi(\theta\mid y)`,'Observed y fixed; normalized belief over θ after a prior.']].flatMap(([a,b,c],i)=>[R('row-'+i,72,187+i*123,1136,106,i===1?C.soft:C.panel),T('rowlabel-'+i,94,205+i*123,280,64,a,14,{fontFamily:mono,color:C.green,fontWeight:800}),E('roweq-'+i,370,204+i*123,320,69,b,24),T('rowbody-'+i,737,206+i*123,443,64,c,19)]),
 ...call('Write the variable, the fixed quantity, and the integration measure. Most confusion disappears.')
],['density','mle','bayes']);
two('check','test your understanding','Four checks before moving on','Answer first; the reasoning is in the right-hand panel and the speaker notes.',
 'QUESTIONS','1. Can a Gaussian density value exceed 1?<br><br>2. If L(θ; y) integrates to 1, is it automatically the posterior?<br><br>3. Should an unknown Gaussian σ keep its 1/σ factor?<br><br>4. May two local posteriors always be multiplied?',
 'ANSWERS','1. Yes. A density height is not a probability.<br><br>2. No. A posterior also requires a prior model and a parameter measure.<br><br>3. Yes. The factor varies with the parameter being fitted.<br><br>4. No. Shared prior and shared evidence must be accounted for.',
 'For estimation and fusion: distinguish the observation model from the distribution over the unknown.',['density','mle','fusion'],
 'Example for question 1: Normal(0,0.1²) has density about 3.989 at zero. For question 2 the function might coincide with a posterior under a stated flat prior, but unit integral alone does not identify that model.');
eq('variance','appendix · a useful warning','Dropping the Gaussian normalizer changes the optimum','Known mean μ; n observations; unknown positive variance v = σ².',
 r`\ell(v)=-\frac n2\log(2\pi v)-\frac{S}{2v},\quad S=\sum_i(y_i-\mu)^2>0,\quad \widehat v=\frac Sn`,
 'WITH THE NORMALIZER',`${I(r`\ell'(v)=-n/(2v)+S/(2v^2)`)}.<br><br>It is positive below S/n and negative above S/n.<br><br>The likelihood trades residual size against the spread of the predictive distribution.`,
 'WITHOUT THE NORMALIZER',`Keeping only ${I(r`-S/(2v)`)} makes the objective increase as v grows.<br><br>The resulting supremum is approached at ${I(r`v\to\infty`)}—a different, incorrect fitting problem.<br><br>“Constant” always means independent of <b>every fitted parameter</b>.`,['mle'],
 'If S=0, the Gaussian likelihood is unbounded as v approaches zero, so no finite positive-variance maximum exists. That boundary case is deliberately excluded from the displayed derivation. If the mean is estimated too, replace it by the sample mean before solving for v.');
add('references','sources and reproducibility','Read the definitions alongside the examples','Primary teaching sources and documentation. The numerical examples and step-by-step algebra are original.',[
 ...Object.entries(refs).map(([k,v],i)=>[R('refbox-'+k,72,187+i*85,1136,73),T('refnum-'+k,93,207+i*85,40,30,String(i+1).padStart(2,'0'),15,{fontFamily:mono,color:C.green}),T('reftext-'+k,150,201+i*85,1028,53,v.title+' ↗',16,{link:v.url})]).flat()
],[],
 'All source links checked during preparation. Sources support the definitions and modeling principles; custom numerical illustrations are derived in this deck and tested in test.mjs. Live plots are pedagogical calculations, not empirical evidence. Readable source and tests live in likelihood-vs-density/.');
for(const [i,s] of slides.entries()){
 s.elements.unshift({...R('progress-bg',0,0,1280,4,C.rule),strokeWidth:0,radius:0},{...R('progress',0,0,1280*(i+1)/slides.length,4,C.green),strokeWidth:0,radius:0});
 s.elements.push(T('footer-page',1120,688,86,20,`${String(i+1).padStart(2,'0')} / ${slides.length}`,11,{fontFamily:mono,color:C.muted,align:'right'}));
}
export const deck={format:'bento/slides',version:1,docId:'likelihood-vs-density-bento',title:'Likelihood & Density',readonly:true,meta:{author:'Bai Liping',subject:'Likelihood versus density: definitions, inference, and sensor fusion',company:'bailiping.com'},size:{width:1280,height:720},theme:{background:C.paper,color:C.ink,accent:C.green,fontFamily:sans},slides};
export const inlineLiveMap=live;
