import {preset,solveScene} from '../jpda/math.mjs';
const C={paper:'#ffffff',ink:'#16273e',muted:'#596d80',rule:'#d8e1e9',wash:'#f2f6fa',green:'#087f68',blue:'#2766b1',orange:'#b96815',purple:'#7854a3'};
const FONT='Arial, Helvetica, sans-serif';
const tex=(s,block=false)=>`<span class="math-tex math-${block?'display':'inline'}">${block?'\\[':'\\('}${s}${block?'\\]':'\\)'}</span>`;
const eq=s=>tex(s,true);
const T=(id,x,y,w,h,html,size=24,color=C.ink,weight=400)=>({id,type:'text',x,y,w,h,html,fontSize:size,fontFamily:FONT,fontWeight:weight,color,align:'left',valign:'top',lineHeight:1.3,rotation:0,opacity:1});
const R=(id,x,y,w,h,fill=C.wash,radius=8)=>({id,type:'shape',shape:'rect',x,y,w,h,fill,stroke:'none',strokeWidth:0,radius,rotation:0,opacity:1});
const I=(id,name,x,y,w,h,alt)=>({id,type:'image',src:`./assets/${name}.svg`,x,y,w,h,alt,fit:'contain',rotation:0,opacity:1});
const link=(label,href,live=false)=>`<a class="${live?'try-live':'deck-link'}" href="${href}"><span>${label}</span><span aria-hidden="true"> →</span></a>`;
const P=p=>`${(100*p).toFixed(1)}%`;
function slide(id,kicker,title,body,notes){return {id:'s-'+id,background:C.paper,transition:'none',notes,elements:[T('kicker',72,34,1136,22,kicker.toUpperCase(),12,C.green,700),T('title',72,75,1136,82,title,37,C.ink,700),R('rule',72,165,1136,1,C.rule,0),...body,T('footer',72,682,970,18,'JPDA · Joint Probabilistic Data Association',11,C.muted),T('page',1120,682,88,18,'{{page:2}} / {{pages:2}}',11,C.muted)]};}
function build(){
  const r=solveScene(preset('shared')),amb=solveScene(preset('ambiguous'));
  const s=[];
  s.push(slide('cover','From the likelihood ratio to joint association','Joint Probabilistic Data Association',[
    T('headline',76,209,550,115,'Reason jointly.<br>Update softly.',45,C.green,700),
    T('description',78,356,480,96,'One return can support several hypotheses. A feasible joint event gives it only one origin.',26),
    T('sources',78,490,505,76,'Fortmann, Bar-Shalom &amp; Scheffe · 1983<br>Bar-Shalom, Daum &amp; Huang · 2009',17,C.muted),
    T('lab',78,596,520,38,link('Open the interactive lab','/jpda/'),21,C.green,700),
    I('shared-scene','shared',591,199,626,383,'Two predicted targets and one shared return, with JPDA and independent PDA updates.'),
    T('scene-caption',620,582,570,60,'○ prediction · ◆ JPDA mean · □ independent PDA<br>Dashed: innovation contours · black dot: return',16,C.muted)
  ],'This companion introduces point-target JPDA using the two requested primary references. The interactive example is a synthetic Cartesian one-scan model, not a reproduction of the original sonar experiments. All figures and numerical values are generated from the same implementation used by the lab. Arrow keys navigate; page 8 and page 10 offer a focused live lab with Back and Escape.'));
  s.push(slide('competition','01 / Why joint?','A shared return couples the association decisions.',[
    I('scene','shared',70,203,651,398,'A symmetric shared return pulls both target estimates inward.'),
    T('pda-title',778,222,415,37,'Independent PDA',25,C.blue,700),
    T('pda-number',778,272,415,51,`${P(r.independent[0][1])} + ${P(r.independent[1][1])}`,36,C.blue,700),
    T('pda-copy',778,329,407,69,'Each target normalizes its own hypotheses.',21,C.muted),
    T('jpda-title',778,443,415,34,'Joint PDA',25,C.green,700),
    T('jpda-number',778,489,415,50,`${P(r.beta[0][1])} + ${P(r.beta[1][1])}`,36,C.green,700),
    T('jpda-copy',778,547,410,69,`The remaining ${P(r.clutter[0])} is clutter.`,21,C.muted),
    T('takeaway',92,623,1072,35,'The marginal probabilities come from mutually exclusive joint explanations.',21,C.ink,600)
  ],'At the default shared-return settings, independently normalized PDA filters each assign about 96.9 percent to the same return. Those are local hypotheses, not a coherent joint assignment distribution. JPDA gives about 49.2 percent per target and 1.6 percent clutter. A single point-target return cannot come from both targets in one feasible event. Both JPDA means can still move toward the return because they summarize different mutually exclusive possibilities.'));
  s.push(slide('model','02 / Assumptions','Fix the model before assigning probabilities.',[
    ...[['Established point targets','Each target generates zero or one return in a scan. No birth, death, or track management.'],['Gaussian predictions','Independent predicted target states. Linear position measurements with H = I.'],['Poisson clutter','Uniform spatial intensity λ. Unassigned returns are clutter; λ is a density, not a probability.'],['All returns are eligible',tex('P_G=1')+'. Dashed ellipses show uncertainty; they do not reject measurements.']].flatMap(([h,b],i)=>{const x=92+(i%2)*576,y=211+Math.floor(i/2)*205;return [R('card'+i,x-20,y-10,540,178),T('head'+i,x,y+8,492,34,h,25,i===3?C.purple:C.green,700),T('body'+i,x,y+60,492,90,b,22,C.muted)];}),
    T('source',92,637,1096,25,'Large-gate simplification: Fortmann et al. (1983), Section III and footnote 5.',16,C.muted)
  ],'The demo uses two 2D position states with means plus or minus half the separation, covariance P-minus = 1.21 I square metres, and R = sigma squared I. Detection events and Poisson clutter are independent. P_G=1 follows the large-gate simplification explicitly described in footnote 5 on printed page 177 of the 1983 reference. All displayed measurements remain eligible even outside the drawn innovation contour. The viewport is not a surveillance boundary used to truncate Gaussian likelihoods. This is a one-scan teaching model, not a full trajectory tracker.'));
  s.push(slide('ratio','03 / Local evidence','Start with the PDA likelihood ratio.',[
    T('formula',106,227,1068,135,eq(String.raw`\ell_{tj}=\frac{\bbox[3px,border:2px solid #b96815]{P_{D,t}}\,\bbox[3px,border:2px solid #2766b1]{\mathcal N(z_j;\mu_t,S_t)}}{\bbox[3px,border:2px solid #7854a3]{\lambda}}`),39),
    {...T('definitions',92,374,1096,39,tex(String.raw`\mu_t=H\hat x_t^-`)+': predicted measurement · '+tex('S_t')+': innovation covariance',21,C.muted),align:'center'},
    ...[['Detection','How likely is target t to generate a return?',C.orange],['Measurement fit','How plausible is zⱼ under the predicted target distribution?',C.blue],['Clutter competition','How much background clutter is expected per unit area?',C.purple]].flatMap(([h,b,c],i)=>[R('stripe'+i,82+i*389,430,4,146,c,0),T('head'+i,103+i*389,429,334,36,h,25,c,700),T('body'+i,103+i*389,480,327,100,b,22,C.muted)]),
    T('source',92,627,1096,36,'Bar-Shalom, Daum &amp; Huang (2009), equation (38); target index t added.',17,C.muted)
  ],'The original equation (38) is the single-target PDA likelihood ratio, calligraphic L_i(k). Here time is suppressed and a target index is added. P_D is deliberately placed before the Gaussian density as in the handover deck. For the linear Gaussian model the innovation covariance is S_t = H P_t^- H^T + R, with H=I in the lab. Gaussian density and lambda have the same spatial units, so the ratio is dimensionless. This is an unnormalized evidence ratio, not an association probability. The probability requires missed-detection hypotheses and global exclusivity.'));
  s.push(slide('events','04 / Feasible joint events','Two targets + one return = three explanations.',[
    ...[[[1,0],'A generates z₁','B is missed',r.events.find(e=>e.id==='1-0').probability,C.green],[[0,1],'B generates z₁','A is missed',r.events.find(e=>e.id==='0-1').probability,C.orange],[[0,0],'z₁ is clutter','Both targets are missed',r.events.find(e=>e.id==='0-0').probability,C.purple]].flatMap(([a,h,b,p,c],i)=>[R('card'+i,72+i*386,216,364,276),T('assignment'+i,95+i*386,243,315,39,tex(`a=(${a.join(',')})`),25,c,700),T('name'+i,95+i*386,311,315,34,h,25,C.ink,700),T('meaning'+i,95+i*386,359,310,51,b,21,C.muted),T('probability'+i,95+i*386,425,315,47,P(p),32,c,700)]),
    T('invalid',92,547,1096,43,tex(String.raw`a=(1,1)`)+ ' is invalid: one return cannot have two target origins.',26,C.ink,600),
    T('zero',92,604,1096,45,'aₜ = 0 denotes a miss. Positive entries must be distinct. Unassigned returns are clutter.',20,C.muted)
  ],'Assignment a is target-oriented: a_t=0 means no measurement from target t, while a_t=j means measurement j originates from t. This is a reindexing of the feasible joint events in the references. For two targets and one measurement, there are exactly three feasible events. The event assigning z1 to both targets is excluded before normalization. The plotted numeric values use the default shared-return preset. Section III of the 1983 reference and equation (45) of the 2009 tutorial define these exclusivity constraints.'));
  s.push(slide('weights','05 / Score, then normalize','Multiply evidence within each feasible event.',[
    T('weight',86,215,1108,136,eq(String.raw`w(a)=\prod_{t:a_t=0}(1-P_{D,t})\;\prod_{t:a_t>0}\ell_{t,a_t}`),35),
    T('normalize',86,378,1108,119,eq(String.raw`p(a\mid Z)=\frac{w(a)}{\displaystyle\sum_{a'\in\mathcal A}w(a')}`),35),
    T('intuition',108,535,1044,72,'Missed target → multiply by '+tex('1-P_D')+'.<br>Assigned return → multiply by its target-to-clutter likelihood ratio.',23,C.muted),
    T('source',92,631,1096,34,'Poisson form: Fortmann et al. (1983), (3.18); Bar-Shalom et al. (2009), (47).',17,C.muted)
  ],'These weights assume P_G=1, independent Gaussian predicted target states, independent Bernoulli detections, and homogeneous Poisson clutter. Starting with lambda to the power m-minus-number-assigned times the product of P_D Gaussian terms and missed-detection terms, divide by the common lambda to the power m. This produces the displayed likelihood-ratio form. The clutter-count factorial cancels in the parametric Poisson derivation; do not insert the factorial from the nonparametric alternative in equations (49)-(50). The lab uses log weights and log-sum-exp normalization to avoid numerical underflow. A is the set of all feasible assignments for the current scan.'));
  s.push(slide('marginals','06 / Sum consistent events','A target’s probability is a sum over joint events.',[
    T('marginal',95,211,1090,100,eq(String.raw`\beta_{tj}=\sum_{a\in\mathcal A:\,a_t=j}p(a\mid Z)`),37),
    R('table-background',92,356,520,233),
    ...[['Target','Miss','z₁'],['A',P(r.beta[0][0]),P(r.beta[0][1])],['B',P(r.beta[1][0]),P(r.beta[1][1])],['Clutter','—',P(r.clutter[0])]].flatMap((row,i)=>row.map((v,j)=>T('cell'+i+j,114+j*161,373+i*50,154,35,v,i?23:18,i===0?C.muted:j===0&&i<3?i===1?C.green:C.orange:C.ink,i>0&&j===0?700:400))),
    T('row',696,377,474,62,tex(String.raw`\sum_{j=0}^{m}\beta_{tj}=1`),30,C.green),
    T('row-label',704,436,474,40,'Every target has one outcome.',22,C.muted),
    T('column',696,489,474,62,tex(String.raw`\sum_t\beta_{tj}\leq 1\quad(j>0)`),30,C.blue),
    T('column-label',704,548,478,57,'The remainder is measurement j’s clutter probability.',22,C.muted),
    T('source',92,637,1096,28,'Marginals: 1983 (3.19)–(3.20); 2009 (51). Values shown for the shared-return scene.',16,C.muted)
  ],'Sum the probabilities of all joint events in which target t is associated with measurement j. Include j=0 as the missed-detection marginal. The row sum is one because the outcomes partition all events. The sum over target origins for each real measurement is at most one, with the remaining probability belonging to clutter. No analogous bound applies to the column of misses: many targets may be missed together. Displayed values are rounded, but the underlying calculations satisfy the constraints at floating-point precision.'));
  s.push(slide('explore','07 / Interactive experiment','Move the return. Watch the explanations change.',[
    I('scene','shared',70,201,700,428,'Static default shared-return JPDA scene, available without opening the live lab.'),
    T('prompt',816,215,358,52,'TRY THREE CHANGES',15,C.green,700),
    T('step1',816,265,360,82,'1   Move z₁ toward A.<br>Which event gains weight?',24),
    T('step2',816,373,360,82,'2   Increase λ.<br>Does clutter gain support?',24),
    T('step3',816,481,360,82,'3   Lower '+tex('P_D')+'.<br>How does “miss” change?',24),
    T('try-live',806,586,386,47,link('TRY LIVE · shared return','/jpda/?preset=shared',true),21,C.green,700),
    T('caption',96,625,663,35,'Static baseline: A → z₁ 49.2% · B → z₁ 49.2% · clutter 1.6%.',18,C.muted)
  ],'Open the focused lab using TRY LIVE. The iframe is created only on demand and removed when closed. Use Back to slides or Escape to return with focus restored. Measurements can be dragged, moved with arrow keys, or positioned using labelled range controls. First move the return toward A, then increase the clutter intensity, then lower the detection probability. Inspect the event list and the marginal table after each change. The static figure and baseline numbers remain in the slide for print and noninteractive reading.'));
  s.push(slide('mean','08 / State update','Update with a weighted innovation.',[
    T('mean-formula',91,212,1100,113,eq(String.raw`\hat x_t^+=\hat x_t^-+K_t\bar\nu_t,\qquad \bar\nu_t=\sum_{j=1}^{m}\beta_{tj}(z_j-\mu_t)`),32),
    T('gain',91,347,1100,74,eq(String.raw`K_t=P_t^-H^\top S_t^{-1},\qquad S_t=HP_t^-H^\top+R`),27,C.muted),
    R('example-background',92,461,1096,143),
    T('example-head',118,481,1044,35,'Shared-return example: target A',24,C.green,700),
    T('example',118,529,1044,48,tex(String.raw`(-1,0)\ \longrightarrow\ (-0.635,\;0.091)\;\mathrm m`)+ '   with '+tex(String.raw`\beta_{A1}\approx0.492`),27),
    T('source',92,634,1096,29,'Bar-Shalom et al. (2009), (39)–(41). The miss branch contributes zero innovation.',17,C.muted)
  ],'For each target, compute the Kalman gain from its predicted covariance and the measurement noise. The weighted innovation sums only real measurements, since the missed-detection branch retains the prior and has zero innovation. The example uses A prior mean (-1,0), z1=(0,0.25), P-minus=1.21 I, sigma=0.65, and beta A1 approximately 0.492232. Its posterior mean is approximately (-0.635160,0.091210). The mean is a summary of alternatives, not a decision that a particular return was generated by this target.'));
  s.push(slide('covariance','09 / Uncertainty','Averaging positions does not remove ambiguity.',[
    T('covariance',82,209,1116,117,eq(String.raw`P_t^+=\sum_{j=0}^{m}\beta_{tj}\!\left[P_{tj}+(\hat x_{tj}-\hat x_t^+)(\hat x_{tj}-\hat x_t^+)^\top\right]`),29),
    I('scene','ambiguous',73,349,584,282,'Overlapping targets and three returns yield spread between association alternatives.'),
    T('branch',714,367,466,72,'Within-branch covariance<br>+ between-branch spread',26,C.green,700),
    T('explain',714,465,466,81,'Keep uncertainty about which return belongs to the target.',23,C.muted),
    T('try-live',706,579,495,47,link('TRY LIVE · overlapping targets','/jpda/?preset=ambiguous',true),20,C.green,700),
    T('source',92,639,1096,26,'Equivalent mixture form of 2009 (42)–(44). j = 0 is the unchanged prior branch.',16,C.muted)
  ],`The missed-detection branch has mean x-minus and covariance P-minus. Every real-measurement branch has the ordinary Kalman posterior mean and covariance P_c=P-minus-KSK-transpose. The covariance of the mixture equals the weighted branch covariances plus the spread of branch means around the mixture mean. This is algebraically equivalent to the PDA covariance equations (42)-(44). JPDA substitutes its jointly consistent marginal association probabilities. The overlapping-target preset has ${amb.events.length} feasible events. Its ellipses are contours of the moment-matched Gaussian, not contours of the full non-Gaussian mixture. The lab reports within-branch and between-branch traces separately.`));
  s.push(slide('limits','10 / What is exact here?','Exact event sums. Approximate recursive filtering.',[
    ...[['One scan','All feasible joint events are enumerated. The event sums are exact for the stated prediction model.',C.green],['State compression','JPDAF keeps Gaussian marginal moments. It does not retain the full mixture of joint track histories.',C.blue],['Scale & ambiguity','Event counts grow rapidly. Persistent ambiguity can pull tracks together: coalescence.',C.purple]].flatMap(([h,b,c],i)=>[R('stripe'+i,82+i*389,229,4,275,c,0),T('head'+i,105+i*389,226,327,47,h,27,c,700),T('body'+i,105+i*389,304,325,184,b,25,C.muted)]),
    R('distinction-background',92,549,1096,99),T('distinction',116,569,1050,67,'This lab uses decoupled JPDAF updates. JPDACF also maintains cross-covariances between target states.',23,C.ink)
  ],'Do not confuse exact enumeration in a small single-scan association problem with exact recursive Bayesian multitarget filtering. JPDAF uses Gaussian approximations and independent target marginal predictions. Coupling through association events is included, while a full coupled target covariance is not carried into the next scan in this lab. The 2009 tutorial distinguishes JPDAF from JPDACF, which includes cross-covariances. Association ambiguity and Gaussian compression can contribute to track coalescence. The number of feasible events for T targets and m eligible returns is the sum over k from zero to min(T,m) of binomial(T,k) binomial(m,k) k factorial. For two targets, it is 1 + 2m + m(m-1). This lab deliberately keeps examples small and transparent.'));
  s.push(slide('connection','11 / Back to extended targets','Similar factor roles. Different measurement models.',[
    T('point-head',92,223,510,37,'Point-target JPDA',27,C.blue,700),
    T('point-copy',92,292,510,169,'One target → at most one return.<br><br>'+tex('P_D')+' is a detection probability.<br>λ is clutter intensity per unit area.',24),
    R('divider',638,226,1,344,C.rule,0),
    T('extended-head',692,223,500,37,'Grouped extended-target likelihood',25,C.green,700),
    T('extended-copy',692,292,495,217,'One target → a group of returns.<br><br>'+tex(String.raw`\mu_m`)+' is a mean measurement count.<br>The count exponential and group clutter terms also matter.',24),
    T('roles',92,556,1096,46,'The matching colors compare detection, measurement-fit, and clutter roles.',22,C.muted),
    T('return',92,616,1096,40,link('Return to Intuitive Intepretation','/et-handover/#/3'),22,C.green,700)
  ],'The orange, blue, and purple boxes in the handover deck compare analogous factor roles. The scalar mu_m in the grouped extended-target likelihood is a Poisson mean count, while P_D in point-target PDA/JPDA is a Bernoulli detection probability. The grouped likelihood contains an exponential count term outside the product and a group clutter normalization. A spatial clutter pdf f_c is distinct from the intensity lambda. These similarities are useful for interpretation but are not a direct substitution proving the models equivalent. The current handover formulation conditions on a fixed external partition; JPDA here associates individual point measurements. The return link navigates in the same tab to page 4.'));
  s.push(slide('references','Primary sources / Read further','Two papers behind this deck and lab.',[
    T('1983-year',92,209,130,42,'1983',30,C.green,700),
    T('1983-title',232,207,953,70,'Sonar Tracking of Multiple Targets Using<br>Joint Probabilistic Data Association',27,C.ink,700),
    T('1983-authors',232,296,953,64,'Thomas E. Fortmann, Yaakov Bar-Shalom &amp; Molly Scheffe<br>IEEE Journal of Oceanic Engineering · OE-8(3), 173–184 · July 1983',17,C.muted),
    T('1983-link',232,367,953,33,link('Read PDF · joint events (3.18)–(3.20)','/et-handover/papers/55JOE.pdf'),19,C.green,600),
    R('divider',92,422,1096,1,C.rule,0),
    T('2009-year',92,454,130,42,'2009',30,C.blue,700),
    T('2009-title',232,452,953,42,'The Probabilistic Data Association Filter',27,C.ink,700),
    T('2009-authors',232,514,953,65,'Yaakov Bar-Shalom, Fred Daum &amp; Jim Huang<br>IEEE Control Systems Magazine · 82–100 · December 2009',17,C.muted),
    T('2009-link',232,583,953,33,link('Read PDF · likelihood (38), updates (39)–(44), JPDA (45)–(51)','/et-handover/papers/358CSM.pdf'),19,C.blue,600),
    T('lab-link',92,641,506,28,link('Open the standalone interactive lab','/jpda/'),18,C.green,600),
    T('back-link',705,641,480,28,link('Back to the handover deck','/et-handover/#/3'),18,C.green,600)
  ],'The local reference PDFs were supplied by the user and are hosted alongside the existing handover deck. Fortmann, Bar-Shalom, and Scheffe (1983) establish the joint event formulation for multiple targets. Bar-Shalom, Daum, and Huang (2009), DOI 10.1109/MCS.2009.934469, provide the PDA tutorial and JPDA review. Printed pages 90-93 contain the likelihood ratio, moment update, event probabilities, and JPDAF/JPDACF discussion. All numeric plots in this new deck are generated by the educational model, not copied from the papers or claimed to reproduce their experiments.'));
  return s;
}
export default {docId:'jpda-companion-20260929',title:'JPDA — Joint Probabilistic Data Association',description:'A source-grounded introduction to JPDA with an interactive two-target association lab, based on Fortmann et al. (1983) and Bar-Shalom et al. (2009).',build,...C,fontFamily:FONT};
