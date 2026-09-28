// EO-specific authoring source. The local builder preserves the Bento runtime.
const C = { paper:'#ffffff', ink:'#16273e', muted:'#596d80', rule:'#d8e1e9', wash:'#f2f6fa', green:'#087f68', blue:'#2766b1', orange:'#b96815', purple:'#7854a3', red:'#c62828' };
const FONT = 'Arial, Helvetica, sans-serif';
const TITLE = 'Scalable Extended-Target Handover in Distributed Integrated Sensing and Communication';
const tex = (s, block=false) => `<span class="math-tex math-${block?'display':'inline'}">${block?'\\[':'\\('}${s}${block?'\\]':'\\)'}</span>`;
const eq = s => tex(s,true);
const T = (id,x,y,w,h,html,size=24,color=C.ink,weight=400) => ({id,type:'text',x,y,w,h,html,fontSize:size,fontFamily:FONT,fontWeight:weight,color,align:'left',valign:'top',lineHeight:1.25,rotation:0,opacity:1});
const R = (id,x,y,w,h,fill=C.wash,radius=0) => ({id,type:'shape',shape:'rect',x,y,w,h,fill,stroke:'none',strokeWidth:0,radius,rotation:0,opacity:1});
const I = (id,src,x,y,w,h,alt) => ({id,type:'image',src,x,y,w,h,alt,fit:'contain',rotation:0,opacity:1});
const multiTargetGraphs = new Set(['local_grbp','distributed','centralized','centralized_parallel']);
const asset = n => `./assets/${n}.svg${multiTargetGraphs.has(n)?'?v=20260918-multitarget-prediction':''}`;
function slide(id,kicker,title,body,notes) {
  return {id:'s-'+id,background:C.paper,transition:'none',notes,elements:[
    T('kicker',72,35,1080,22,kicker.toUpperCase(),12,C.green,700),
    T('title',72,73,1136,85,title,36,C.ink,700),
    R('rule',72,165,1136,1,C.rule), ...body,
    T('footer',72,682,960,18,'GrBP · Extended-target handover',11,C.muted),
    T('page',1110,682,98,18,'{{page:2}} / {{pages:2}}',11,C.muted)
  ]};
}
function figure(id,kicker,title,name,caption,notes,captionColor=C.muted) {
  return slide(id,kicker,title,[
    I('diagram',asset(name),72,186,1136,425,title),
    ...(caption ? [T('caption',92,625,1096,43,caption,19,captionColor)] : [])
  ],notes);
}
function live(id,intro,title,prompt,selector,fallback,notes) {
  const s = slide(id,'Explore · '+(id==='score-live'?'the trigger':'the protocol'),title,[
    T('prompt',72,123,1136,32,prompt,17,C.muted),
    I('static-fallback',`./assets/${fallback}.png`,40,178,1200,486,title+' — deterministic initial state'),
    R('live-demo-mount',40,178,1200,486,'transparent')
  ],notes);
  s.elements.find(e=>e.id==='title').h=44;
  s.__demo={introSlide:'s-'+intro,slide:s.id,inline:true,layout:'region',bounds:{x:40,y:178,width:1200,height:486},
    src:`./live/?slide-embed=%23${selector}&v=20260927-loading`,source:`./live/#${selector}`,title,
    sandbox:'allow-scripts allow-same-origin',hideSource:true,readyMessage:true,unloadWhenHidden:true};
  return s;
}
function build() {
  const slides = [];
  slides.push({id:'s-cover',background:C.paper,transition:'none',notes:'Introduce grouped-measurement belief propagation (GrBP). This talk separates the local method, its processing schedule, and track-level handover. Figure 1 is reproduced unchanged from the manuscript: gray discs denote sensing fields of view, blue is the target trajectory, black dots are incidence points, and red highlights handover. The central unit depicts the coordinated baseline.',elements:[
    T('eyebrow',72,35,1136,24,'DISTRIBUTED ISAC / EXTENDED-TARGET TRACKING',13,C.green,700),
    T('cover-title',72,70,1136,108,TITLE,42,C.ink,700),
    T('point-target-reference',72,194,640,26,'<a class="cover-reference" href="/target-handover-slides/"><span>extention of point-target handover</span><span aria-hidden="true"> →</span></a>',17,C.green,600),
    {...T('paper-reference',752,194,456,26,'<a class="cover-reference" href="https://arxiv.org/abs/2609.25737"><span>Paper · arXiv:2609.25737</span><span aria-hidden="true"> →</span></a>',17,C.green,600),align:'right'},
    R('rule',72,228,1136,1,C.rule),
    I('manuscript-figure-1','./assets/manuscript-figure-1.png',220,244,840,414,'Figure 1: extended-target handover in a DISAC network, with overlapping sensing regions, a target trajectory, and a red handover arrow.'),
    T('author',72,682,420,18,'Current manuscript companion',13,C.muted),
    T('figure-caption',590,682,618,18,'Fig. 1 · Extended-target handover in a DISAC network.',13,C.muted)
  ]});
  slides.push(slide('scalability','Motivation / DISAC','Scalability is core design objective for DISAC',[
    {...R('scalability-tracking-box',72,276,544,248,C.wash,12),stroke:C.rule,strokeWidth:2},
    {...R('scalability-fusion-box',664,276,544,248,C.wash,12),stroke:C.rule,strokeWidth:2},
    {...T('scalability-tracking',72,276,544,248,'<a class="scalability-topic" href="/eo-mtt-slides/"><span>Extended-Target Tracking</span><span class="topic-arrow" aria-hidden="true"> →</span></a>',30,C.ink,700),align:'center',valign:'middle'},
    {...T('scalability-fusion',664,276,544,248,'<a class="scalability-topic" href="/multidensity-fusion/"><span>Density Fusion</span><span class="topic-arrow" aria-hidden="true"> →</span></a>',30,C.ink,700),align:'center',valign:'middle'},
    {...T('scalability-tracking-solution',72,552,544,40,'Solution: GrBP',25,C.green,600),align:'center'},
    {...T('scalability-fusion-solution',664,552,544,40,'Solution: Target Handover',25,C.green,600),align:'center'},
    {...T('scalability-bp-mtt',72,608,544,30,'<a class="cover-reference" href="/bp-vs-pmbm-slides/"><span>Belief-Propagation MTT</span><span aria-hidden="true"> →</span></a>',19,C.green,600),align:'center'}
  ],'Introduce scalability as a central requirement for DISAC. The two boxes link to the Extended-Target Tracking and Density Fusion presentations. Both links navigate in the current tab; use the browser Back button to return to this presentation. A supporting Belief-Propagation MTT link appears beneath Solution: GrBP and opens its presentation in the current tab.'));
  slides.push(slide('factorization','01 / The local method','GrBP factorization and likelihood functions',[
    T('factorization-context',92,187,1096,28,'Fixed measurement groups; time and BS indices suppressed. '+tex(String.raw`\bm y=(\bm x,\bm E,r),\quad r\in\{0,1\}.`),18,C.muted),
    R('factorization-background',72,220,1136,98,C.wash,6),
    T('joint-factorization',88,225,1104,88,eq(String.raw`f(\underline{\bm Y},\overline{\bm Y},\bm a,\bm b,\bm Z^G)\approx C\,\bm\psi(\bm a,\bm b)\prod_{p=1}^{n^t}\!\left[f(\underline{\bm y}^{p})\,\underline l(\underline{\bm y}^{p},a^p;\bm Z^G)\right]\prod_{j=1}^{n^g}\!\overline l(\overline{\bm y}^{j},b^j;\bm Z^j).`),23),
    T('factorization-key',92,319,1096,25,tex(String.raw`f(\underline{\bm y}^{p})`)+': predicted prior · '+tex(String.raw`\bm\psi`)+': one-to-one association consistency · '+tex('C')+': fixed-scan constant',16,C.muted),
    T('legacy-likelihood-heading',92,361,530,28,'Legacy likelihood factor',21,C.green,700),
    T('newborn-likelihood-heading',678,361,530,28,'Newborn likelihood factor · group '+tex('j'),21,C.purple,700),
    R('likelihood-divider',639,363,1,281,C.rule),
    T('legacy-likelihood',80,404,544,242,eq(String.raw`\underline l(\bm y,a;\bm Z^G)=\begin{cases}\begin{aligned}[t]&\dfrac{e^{-\mu_m(\bm x,\bm E)}}{\mu_g}\\[-3pt]&\times\!\prod_{\bm z\in\bm Z^a}\!\dfrac{\mu_m(\bm x,\bm E)f(\bm z\mid\bm x,\bm E)}{f_c(\bm z)},\end{aligned}&r=1,\ a>0,\\[5pt]e^{-\mu_m(\bm x,\bm E)},&r=1,\ a=0,\\[3pt]1,&r=0,\ a=0,\\[3pt]0,&r=0,\ a>0.\end{cases}`),20),
    T('newborn-likelihood',662,404,546,242,eq(String.raw`\overline l(\bm y,b;\bm Z^j)=\begin{cases}\begin{aligned}[t]&\dfrac{\mu_n f_n(\bm x,\bm E)e^{-\mu_m(\bm x,\bm E)}}{\mu_g[1-e^{-\mu_m(\bm x,\bm E)}]}\\[-3pt]&\times\!\prod_{\bm z\in\bm Z^j}\!\dfrac{\mu_m(\bm x,\bm E)f(\bm z\mid\bm x,\bm E)}{f_c(\bm z)},\end{aligned}&r=1,\ b=0,\\[5pt]0,&r=1,\ b>0,\\[3pt]f_d(\bm x,\bm E),&r=0.\end{cases}`),20),
    T('likelihood-key',92,651,1096,28,tex(String.raw`\mu_m`)+': mean detections · '+tex(String.raw`\mu_g`)+': mean clutter groups · '+tex(String.raw`\mu_n f_n`)+': detected-newborn intensity · '+tex('f_d')+': dummy density',15,C.muted)
  ],'The joint density factorizes under the approximate model conditioned on a fixed external measurement partition. Time, base-station indices, and conditioning on past data are suppressed. The augmented state y contains kinematics x, extent E, and existence r. There are n^t legacy components and n^g group-indexed newborn candidates. The prior f of each legacy state is the predicted marginal, also denoted alpha in the processing schedules. The consistency factor psi is the product of pairwise indicators enforcing a^p=j if and only if b^j=p; zero denotes no legacy-to-group assignment. C depends only on the fixed observations and fixed model parameters, so it drops out when normalizing the posterior. Both likelihood factors explicitly show their dependence on x and E. The assigned-existing legacy branch is exp(-mu_m(x,E))/mu_g times the product of mu_m(x,E)f(z|x,E)/f_c(z) over the assigned group. The exponential and inverse clutter-group mean each occur once per target-generated group; the remaining cases encode missed detection and absence. The newborn factor includes the detected-newborn intensity mu_n f_n(x,E) and the nonempty-count normalization 1-exp(-mu_m(x,E)). A group claimed by a legacy target cannot also create an existing newborn. The absence branch carries a normalized dummy density f_d(x,E). The displayed likelihood factors are local factors, not separately normalized state densities. There is no additional newborn prior factor. Fixed-partition modeling and loopy belief propagation are separate approximations.'));
  slides.push(slide('grbp','01 / The local method','GrBP, Group-measurement Belief Propagation',[
    I('local-graph',asset('local_grbp'),92,191,1096,320,'GrBP factor graph with multiple legacy targets and group-indexed newborn candidates, coupled through the shared association-consistency factor'),
    ...[['01','Group once','One fixed external partition per scan.'],['02','Associate','BP links groups to legacy and newborn hypotheses.'],['03','Update','Return legacy and newborn marginals.']].flatMap(([n,t,b],i)=>[
      T('step'+i,80+i*390,542,50,32,n,22,C.green,700),T('head'+i,135+i*390,542,300,35,t,23,C.ink,700),T('body'+i,135+i*390,582,300,66,b,18,C.muted)
    ]),
    T('grbp-derivation',92,642,1096,27,'<a class="cover-reference" href="../eo-derivation/#grbp"><span>Derivation for GrBP</span><span aria-hidden="true"> →</span></a>',18,C.green,600)
  ],'GrBP means grouped-measurement belief propagation. The augmented state contains kinematics, extent, and existence. The fixed clustering partition is input to the association problem; it is not inferred jointly. Adapted from the manuscript factor graph: show legacy chains 1 through n^t and group/newborn chains 1 through n^g, with separate ellipses. These counts need not be equal; rows do not prescribe target-to-group matches. All a and b variables meet at psi, which abbreviates the product of pairwise consistency factors. The first factor is the predicted density f with its argument abbreviated by a dot. Underlined l(dot) and overlined l(dot) abbreviate the full legacy and newborn local factors. Variable nodes retain their component indices. The newborn factor already contains its density and clutter terms. Time indices are suppressed. The same multi-target graph is expanded in each later processing diagram.'));
  slides.push(slide('architectures','02 / Processing choices','Different level of coordination.',[
    ...[['Distributed','GrBP-D','Least accurate but inexpensive.',C.muted],['Handover','GrBP-H / HM / HL / HP','Event triggered.',C.green],['Coordinated','GrBP-CS / GrBP-CP','Most accurate but expensive.',C.blue]].flatMap(([h,sub,b,col],i)=>[
      R('stripe'+i,76+i*398,219,4,310,col),T('a-head'+i,98+i*398,218,345,45,h,27,col,700),
      T('a-sub'+i,98+i*398,284,330,54,sub,23,C.ink,700),T('a-body'+i,98+i*398,378,312,112,b,24,C.muted)
    ]),
  ],'Do not equate coordinated scheduling with one fixed deployment topology. A physical fusion node collecting network-wide measurements carries at least linear load; the schedules themselves may also be implemented without that node.'));
  slides.push(figure('distributed','03 / GrBP-D','Distributed: independent GrBP, no exchanged beliefs.','distributed','Local load stays local; track continuity must be rediscovered at a boundary.','Each station predicts, groups, associates, and updates independently. Each full local graph shows first and last legacy and group/newborn chains with ellipses. Its counts are local legacy count n_{s,k}^t and group count n_{s,k}^g; k is suppressed inside the graph. Dark edges are factor dependencies and arrows at card boundaries are processing inputs and outputs. There are no inter-BS communication links. Local duplicates may occur in overlaps; this is distinct from the individual-measurement BP-based ETT reference.'));
  slides.push(figure('centralized','04 / GrBP-CS','Sequential: the updated belief moves forward.','centralized','One way to implement sequential processing','Each stage contains the full multi-target association graph, with separate incoming-target and local-group counts. The first stage starts with the common prediction. Later stages receive the predecessor’s updated legacy and newborn beliefs; n_s^t counts all incoming components, including earlier newborn proposals. The original prediction is not multiplied again. A parenthesized index denotes the completed update stage. Dark edges are factor dependencies; blue arrows pass the set of updated beliefs between stages. Logical stages can run at a physical fusion node or across a coordinated network.',C.red));
  slides.push(figure('parallel','05 / GrBP-CP','Parallel: local evidence meets at one combiner.','centralized_parallel','Every local update starts from the same prediction. The prior enters the final product once.','Each of the three representative BSs contains the full multi-target association graph: a common predicted legacy set of size n^t and its own n_s^g group/newborn candidates. The combiner receives a collection of likelihood messages, one for each matched legacy target i. Its posterior is proportional to the predicted density f times the product of local gamma messages. This order-invariant product is not a naive product of local posteriors. Newborn proposals remain separate and may duplicate without cross-BS association. Dark edges are factor dependencies; colored arrows connect processing stages. Time indices are suppressed inside the graphs.'));
  slides.push(figure('handover-variants','06 / The four paper variants','Event-triggered target handover','handover_variants','','Orange is the event-triggered prior, green is measurement evidence, red is likelihood evidence, blue is posterior evidence, and purple is the fused posterior. HM sends a selected non-null associated measurement group. HL sends likelihood information. HP sends a local posterior. The same pairwise exchange repeats over an owner-centered star.'));
  slides.push(slide('fusion','07 / Information accounting','Likelihood fusion and posterior fusion are different.',[
    T('hl-name',80,217,510,36,'GrBP-HL · likelihood messages',25,C.green,700),
    T('hl-eq',72,291,535,115,eq(String.raw`\widetilde f^\ell\propto\alpha^\ell\prod_{s\in S_\ell}\gamma_s^\ell`),28),
    T('hl-copy',92,445,487,100,'Multiply likelihood information.<br>Include the common prior once.',23,C.muted),
    R('split',638,220,1,352,C.rule),
    T('hp-name',684,217,525,36,'GrBP-HP · local posteriors',25,C.purple,700),
    T('hp-eq',669,291,535,115,eq(String.raw`\widetilde f^\ell\propto\prod_{s\in S_\ell}(\widetilde f_s^\ell)^{1/|S_\ell|}`),28),
    T('hp-copy',690,445,484,100,'Use equal-weight GCI.<br>Do not multiply correlated posteriors directly.',23,C.muted),
    T('scope',92,597,1096,58,'HL’s factorization is a fixed-track approximation, not an exact centralized multi-object posterior.',19,C.muted)
  ],'HL removes the common alpha from local posterior information to form gamma, then uses alpha times the gamma product at the owner. HP conservatively pools local posteriors using equal-weight geometric fusion, known as GCI. Both return the fused posterior to the relevant shadows.'));
  slides.push(slide('sensing','08 / Extended-target sensing','Visibility changes the number of detections.',[
    T('mu-eq',92,228,1096,113,eq(String.raw`\mu_{m,s}(\bm x,\bm E)=\rho_s(\bm x)\,A_\kappa(\bm E)\,\delta_s(\bm x,\bm E)`),38),
    ...[[String.raw`\rho_s(\bm x)`,'Range','Scatter density decreases with distance.'],[String.raw`A_\kappa(\bm E)`,'Extent','A larger footprint contributes more expected detections.'],[String.raw`\delta_s(\bm x,\bm E)`,'Visibility','Only the visible fraction contributes near a field-of-view edge.']].flatMap(([m,t,b],i)=>[
      T('mu-symbol'+i,92+i*388,392,330,46,tex(m),26,C.green,700),T('mu-title'+i,92+i*388,453,330,40,t,25,C.ink,700),T('mu-copy'+i,92+i*388,511,304,98,b,21,C.muted)
    ])
  ],'The paper models an expected count for an extended target. It is not a Bernoulli point-target detection probability. Kappa controls the footprint and is distinct from the scatter spread parameter. Partial visibility softens the field-of-view boundary.'));
  slides.push(slide('score','09 / Trigger → experiment','Ask what the neighboring receiver will see.',[
    T('trigger-eq',88,235,1104,130,eq(String.raw`\Lambda_{t\to r}=\mathbf 1\{p_t\ge p_{\rm th}\}\int\mu_{m,r}(\bm x,\bm E)f_t(\bm x,\bm E)\,d\bm x\,d\bm E`),30),
    T('trigger-condition',104,415,460,63,tex(String.raw`\Lambda_{t\to r}\ge\Lambda_{\rm th}`)+' → send a prior',26,C.green,700),
    T('trigger-explain',660,416,508,106,'Existence-gated.<br>Receiver-specific.<br>Averaged over the predicted density.',24,C.muted),
    R('watch-bg',92,577,1096,68,C.wash,6),T('watch',114,597,1052,37,'Next: drag the target and change its extent. Watch partial visibility change the score.',20,C.ink)
  ],'This is the introduction to the next live slide. Prediction can initialize a shadow before local detections arrive. The paper uses the density integral and hysteresis; the teaching widget evaluates a single predicted pose with the existence gate on, so it is an approximation.'));
  slides.push(live('score-live','score','Explore the handover trigger','Drag the target. Increase the extent. Compare a grazing target with one entering the field of view.','handover-score-demo','score-fallback','The widget is a deterministic illustration of the mean count at one predicted pose, not the full posterior-density integral. Dragging and the extent slider use the same model as the source page. Page Up/Down navigate; Escape returns focus to the deck.'));
  slides.push(slide('protocol','10 / Continuity → experiment','Replicate first. Transfer ownership later.',[
    ...[['01','Predict visibility','The owner scores neighboring receivers.'],['02','Replicate a prior','A shadow keeps the same owner–UID label.'],['03','Exchange evidence','HM / HL / HP send evidence and fused returns.'],['04','Transfer authority','A requests the transfer. C becomes owner only after the acknowledgment reaches A.']].flatMap(([n,h,b],i)=>[
      T('p-num'+i,92,204+i*96,64,43,n,31,C.green,700),T('p-head'+i,180,208+i*96,370,41,h,27,C.ink,700),T('p-copy'+i,596,209+i*96,572,66,b,22,C.muted)
    ]),
    T('p-watch',92,618,1096,38,'Next: play the six events and switch all four GrBP handover variants.',20,C.green,700)
  ],'This is the introduction to the next live slide. Replication and transfer are different events. The old owner retains authority until the two-message transfer is acknowledged. If no shadow sees the target, it retains ownership until recovery or pruning. The old (owner, UID) label is replaced only when the new owner issues its UID.'));
  slides.push(live('protocol-live','protocol','Follow one track across three stations','Event 5: target exits BS A → request to C → acknowledgment to A → BS C owns.','toy','timeline-fallback','Use the existing event controls or timeline to inspect event 5. The highlighted transfer arrow appears only during the request and acknowledgment. Playback remains at the normal speed. A remains the owner with label (A, 3) while the request and acknowledgment travel. Only when the acknowledgment reaches A does C become owner and the label change to (C, 7). The target center crossing A’s field-of-view boundary triggers this schematic event; part of its extent can still be visible. The six events are birth, two prior replications, shadow pruning, ownership transfer, and final pruning. Geometry and deterministic dots are schematic rather than actual GrBP estimates. Belief payloads are 196 B, measurements 16 B. The illustrative 24 B controls are additional toy bookkeeping, excluded from the benchmark payload table.'));
  const environment = slide('simulation-environment','11 / Simulation environment','Simulation environment',[
    I('simulation-animation','./assets/simulation-environment.gif?v=20260923-no-legend',36,30,630,630,'Animated GrBP tracking trial mc_0088 across seven base stations and their overlapping sensing regions; the legend is alongside the animation'),
    // Replace the baked-in static heading; the adjacent animated frame counter stays visible.
    R('simulation-heading-background',219.75,38.4,210,14.7,C.paper),
    {...T('simulation-heading',219.75,38.8,209,13,'GrBP Tracking | mc_0088 |',10.5,'#000000'),align:'right'},
    I('environment-legend-symbols','./assets/simulation-legend-symbols.svg',724,240,48,342,'Legend symbols matching the animation: dashed green line, green pentagon, gray cross, dashed gray line, black dot, blue line, blue dot, dotted blue line, and solid blue line'),
    ...['BS sensing range','Base station','Measurements','True trajectory','True position','Estimated trajectory','Estimated position','Estimated position covariance','Estimated extent'].map((label,i)=>
      T('environment-legend-label-'+i,790,246+i*38,418,30,label,21,C.ink)
    ),
    T('environment-source',724,611,464,45,'GrBP tracking · trial mc_0088',16,C.muted)
  ],'Simulation-environment introduction, using the user-supplied simulation animation with its static heading relabeled GrBP Tracking in the slide. The animation is rendered from the original archived mc_0088 results with the in-plot legend removed. The legend beneath the slide title identifies sensing range, base stations, measurements, true trajectories and positions, estimated trajectories and positions, estimated position covariance, and estimated extent. The GIF contains 100 frames, plays at 10 frames per second, and loops; its animated frame counter remains visible. Introduce the overlapping surveillance regions before advancing to the aggregate results.');
  Object.assign(environment.elements.find(e=>e.id==='kicker'),{x:724,y:73,w:464});
  Object.assign(environment.elements.find(e=>e.id==='title'),{x:724,y:112,w:464,h:92});
  Object.assign(environment.elements.find(e=>e.id==='rule'),{x:724,y:213,w:464});
  slides.push(environment);
  const results=[['GrBP-CS',2.02,468,C.blue],['GrBP-CP',1.99,468,C.blue],['GrBP-D',3.50,0,C.muted],['GrBP-H',2.80,32,C.green],['GrBP-HM',2.79,161,C.green],['GrBP-HL',2.65,343,C.green],['GrBP-HP',2.59,344,C.green]];
  slides.push(slide('results','12 / Current manuscript evidence','Lower traffic and lower error are different objectives.',[
    T('table-method',92,205,255,26,'METHOD',12,C.muted,700),T('table-gospa',347,205,160,26,'GOSPA ↓',12,C.muted,700),T('table-kb',519,205,520,26,'PAYLOAD PER TRIAL · KB',12,C.muted,700),
    ...results.flatMap(([m,g,k,col],i)=>[T('m'+i,92,245+i*48,238,33,m,22,col,700),T('g'+i,347,245+i*48,135,33,g.toFixed(2),22),R('bar'+i,519,249+i*48,k/468*525,19,col,2),T('kb'+i,1060,245+i*48,110,32,''+k,22,C.ink)]),
    T('results-caption',92,604,1096,62,'7 BSs · 100 trials · first 100 frames. GOSPA: mean across all BSs. Payload: rounded KB (bytes / 1024); headers excluded.',18,C.muted)
  ],'Source: current manuscript per_bs_results_summary.tex and handover_counting_summary.tex. The rounded table shows CS 2.02/468 KB, CP 1.99/468, D 3.50/0, H 2.80/32, HM 2.79/161, HL 2.65/343 and HP 2.59/344. H uses 6.9% of coordinated payload from unrounded source data; dividing displayed rounded KB values gives only an approximate ratio. HP is best among non-oracle handover methods. Oracle and individual-measurement ETT comparisons are omitted here, not renamed as GrBP.'));
  slides.push(slide('failure-extent-expansion','13 / Failure analysis','Extent Expansion',[
    T('extent-case-1',80,182,548,24,'MC_0002 · BS5',15,C.muted,600),
    T('extent-case-2',664,182,548,24,'MC_0008 · BS6',15,C.muted,600),
    I('extent-animation-1','./assets/failure-extent-expansion-1.gif',56,214,572,444,'Extent expansion example from PowerPoint page 11: trial MC_0002 at BS5, with nearby tracks and estimated extents'),
    I('extent-animation-2','./assets/failure-extent-expansion-2.gif',652,214,572,444,'Extent expansion example from PowerPoint page 12: trial MC_0008 at BS6, with nearby tracks and estimated extents'),
    R('extent-divider',639,214,1,444,C.rule)
  ],'Failure analysis: extent expansion. The two animations are taken from pages 11 and 12 of the user-provided Sep 11.pptx (MC_0002 BS5 and MC_0008 BS6). They retain all 60 and 61 source frames, respectively, at 10 frames per second. The embedded titles are removed, and each frame is translated onto a fixed plotting canvas to eliminate the source layout jitter. Track positions, extent shapes, trajectories, and annotations are preserved; no temporal smoothing is applied. These are illustrative failure cases, not additional aggregate benchmark results.'));
  slides.push(slide('failure-multiple-initiation','13 / Failure analysis','Multiple Initiation',[
    {...T('initiation-case',72,181,1136,24,'MC_0006 · BS4',15,C.muted,600),align:'center'},
    I('initiation-animation','./assets/failure-multiple-initiation.gif',350,209,580,452,'Multiple initiation example from PowerPoint page 13: trial MC_0006 at BS4, showing overlapping estimated tracks')
  ],'Failure analysis: multiple initiation. The animation is taken from page 13 of the user-provided Sep 11.pptx (MC_0006 BS4). All 62 source frames and the original 10-frame-per-second playback are retained. The embedded title is removed and the plotting canvas is stabilized using integer translations, without altering track identities or smoothing the estimates. Use this example to discuss multiple track initiations around an object.'));
  const multimodalFigures = [
    [6,'BS5',182],
    [7,'BS5',91],
    [8,'BS3',139]
  ];
  multimodalFigures.forEach(([page,station,frame],index)=>{
    const s = slide('multimodal-slide-'+page,'14 / Failure analysis · '+(index+1)+' / 3','Underlying Multimodal distribution',[
      I('multimodal-figure',`./assets/multimodal-slide-${page}.svg`,40,152,1200,516,`${station}, frame ${frame}/200: distributed EOT trajectories on the left and particle clouds on the right, with the original red highlight; from PowerPoint slide ${page}`)
    ],`Underlying Multimodal distribution. Picture from slide ${page} of the user-provided Oct 2.pptx: ${station} distributed EOT, frame ${frame}/200. The left panel shows the tracking scene; the right panel shows particle clouds. The original red annotation highlights the example. Plot and annotation PNG bytes are embedded unchanged, with the annotation positioned using its PowerPoint coordinates. A cropped SVG viewport removes both embedded panel titles while retaining the axes, trajectories, particle clouds, and red highlight. These are individual illustrated cases, not aggregate benchmark results.`);
    Object.assign(s.elements.find(e=>e.id==='title'),{h:52});
    Object.assign(s.elements.find(e=>e.id==='rule'),{y:139});
    slides.push(s);
  });
  slides.push(slide('failure-clustering-error','14 / Failure analysis','Clustering Error Type I',[
    T('legacy-existence-heading',92,187,1096,30,'Legacy tracks · updating the existence belief '+tex(String.raw`P(\underline r^p=1)`),23,C.green,700),
    T('legacy-state-belief',92,232,1096,54,eq(String.raw`\widetilde f^{p}(\bm x,\bm E,r)\propto\alpha^{p}(\bm x,\bm E,r)\,\gamma^{p}(\bm x,\bm E,r)`),27),
    T('legacy-belief-key',92,293,1096,28,tex(String.raw`\alpha^{p}`)+': predicted state-and-existence density · '+tex(String.raw`\gamma^{p}`)+': grouped-measurement likelihood message',18,C.muted),
    R('existence-update-background',72,335,1136,141,C.wash,6),
    T('legacy-existence-update',88,346,1104,118,eq(String.raw`\widetilde P(\underline r^p=1)=\frac{\iint\alpha^{p}(\bm x,\bm E,1)\,\gamma^{p}(\bm x,\bm E,1)\,d\bm x\,d\bm E}{\sum_{r\in\{0,1\}}\iint\alpha^{p}(\bm x,\bm E,r)\,\gamma^{p}(\bm x,\bm E,r)\,d\bm x\,d\bm E}`),26),
    T('legacy-group-message-heading',92,494,1096,28,'The fixed groups enter through the legacy likelihood factor',21,C.green,700),
    T('legacy-group-message',92,533,1096,69,eq(String.raw`\gamma^{p}(\bm x,\bm E,r)=\sum\nolimits_{a=0}^{n^g}\underline l\bigl((\bm x,\bm E,r),a;\bm Z^G\bigr)\,m_{a^p\to\underline l^p}(a)`),23),
    T('legacy-association-message-key',92,609,1096,29,tex(String.raw`m_{a^p\to\underline l^p}(a)`)+': incoming BP association message. Time and BS indices are suppressed.',17,C.muted),
    T('clustering-existence-effect',92,645,1096,29,'Type I error: the estimated cluster contains fewer elements than the true cluster.',18,C.muted)
  ],'Clustering Error Type I means that the estimated cluster contains fewer elements than the true cluster. This page gives the existence-probability update for legacy tracks only. The legacy component index is p; x and E are the kinematic and extent integration variables. Time and base-station indices are suppressed, as is conditioning on past measurements and the fixed current partition. Alpha is the predicted normalized augmented-state density, so its r=1 integral is the predicted existence probability. Gamma is the likelihood-factor-to-state message after BP association. The normalized state belief is proportional to alpha times gamma. Integrating its r=1 branch gives the displayed existence belief; the denominator sums both existence branches and integrates over x and E, including the normalized dummy-state representation for r=0. The tilde marks a BP belief approximating the posterior, not an exact marginal of the full multi-target model. The message m from a^p to the legacy local factor contains the association information from the consistency subgraph and is not a posterior association probability. Summing the product of that incoming message and the legacy factor over all group assignments produces gamma; this avoids feeding a posterior association probability back into the likelihood and counting local evidence twice. For r=0, only a=0 contributes, so gamma(x,E,0)=m(0), independent of x and E. For r=1, the null-assignment term includes exp(-mu_m(x,E)) and each non-null term uses the explicit grouped legacy factor on the factorization slide. A Type I clustering error leaves fewer elements in the target cluster, changing its group factor and potentially the BP association messages. It can reduce support for an existing track, but no monotonic decrease is claimed for every Type I error. The scope is existing legacy tracks, with no newborn update or birth intensity. The normalization and marginalization follow the legacy belief calculation in Meyer et al., Message Passing Algorithms for Scalable Multitarget Tracking, Sec. IX-A5–6, with extent included in the augmented state.'));
  const clusteringExampleRows = Array.from({length:6},(_,count)=>{
    const evidence = Math.exp(-5) * (count === 0 ? 1 : 1 + 5 ** count);
    const posterior = 0.9 * evidence / (0.1 + 0.9 * evidence);
    const highlight = count === 5 ? {bg:C.green,color:C.paper,bold:true} : {};
    return {cells:[
      {html:count === 5 ? '5 · truth' : String(count),...highlight},
      {html:(100 * posterior).toFixed(2)+'%',align:'right',...highlight}
    ]};
  });
  slides.push(slide('failure-clustering-example','14 / Failure analysis · Numerical example','Clustering Error Type I',[
    T('clustering-example-assumptions',92,181,1096,64,eq(String.raw`P^{-}(\underline r^p=1)=0.9,\qquad\mu_m(\bm x,\bm E)=5,\qquad\mu_g=1,\qquad\frac{f(\bm z\mid\bm x,\bm E)}{f_c(\bm z)}=1`),21),
    T('clustering-example-state',92,252,1096,28,'Type I error: fewer elements than the true cluster size of 5. Use the same fixed '+tex(String.raw`(\bm x,\bm E)`)+'.',17,C.muted),
    T('clustering-example-messages',92,291,1096,32,'Fixed BP messages: '+tex(String.raw`m_{a^p\to\underline l^p}(0)=1`)+', and '+tex(String.raw`m_{a^p\to\underline l^p}(1)=1`)+ ' when a group exists.',17,C.muted),
    T('clustering-example-update-heading',92,335,552,28,'Legacy-track existence update',21,C.green,700),
    T('clustering-example-update',80,373,568,76,eq(String.raw`\widetilde P(\underline r^p=1)=\frac{0.9\,\gamma^{p}(\bm x,\bm E,1)}{0.1+0.9\,\gamma^{p}(\bm x,\bm E,1)}`),22),
    T('clustering-example-nonempty-label',92,470,552,24,'1–5 detections · one candidate group',17,C.muted),
    T('clustering-example-evidence',80,498,568,42,eq(String.raw`\gamma^{p}(\bm x,\bm E,1)=e^{-5}\bigl(1+5^{|\bm Z^1|}\bigr)`),22),
    T('clustering-example-zero-label',92,558,552,24,'0 detections · null assignment only',17,C.muted),
    T('clustering-example-zero-evidence',80,586,568,42,eq(String.raw`\gamma^{p}(\bm x,\bm E,1)=e^{-5}`),22),
    {id:'clustering-example-table',type:'table',x:674,y:335,w:534,h:294,rotation:0,opacity:1,header:true,
      columns:[{w:1.1},{w:1}],
      rows:[{cells:[{html:'Retained detections'},{html:'Posterior existence',align:'right'}]},...clusteringExampleRows],
      style:{headerBg:C.wash,headerColor:C.ink,zebra:'#f8fafc',borderColor:C.rule,borderWidth:1,cellPadX:18,cellPadY:6,fontSize:21,color:C.ink,fontFamily:FONT,radius:6}},
    T('clustering-example-takeaway',92,647,1096,28,'For every row, '+tex(String.raw`\gamma^{p}(\bm x,\bm E,0)=1`)+'. Counts 0–4 illustrate Type I error; 5 is the truth.',17,C.muted)
  ],'Controlled numerical example of Clustering Error Type I for one legacy track, following the preceding existence-belief update. Type I means fewer elements in the estimated cluster than in the true cluster. Counts zero through four illustrate Type I error; the row with five is the true cluster and is not an error. The true measurement group contains five detections. The table compares retaining zero through five of those detections; five is explicitly labeled truth and highlighted. These are numbers of detections retained in the target group, not numbers of targets or groups. Condition on one fixed kinematic state x and extent E, equivalently a point prediction given existence. Hold the predicted existence probability at 0.9, the expected detection count mu_m(x,E) at 5, and the mean clutter-group count mu_g at 1 for every row. For every included point, the spatial density ratio f(z|x,E)/f_c(z) is 1, so each point contributes the rate-weighted ratio 5. For a positive retained count, there is one candidate group and the null assignment. Hold the unnormalized incoming BP messages for a=0 and, when present, a=1 at 1; these are messages, not posterior association probabilities. The existing branch sums the null-assignment contribution exp(-5) and the assigned-group contribution exp(-5) times 5 raised to the group size. Thus gamma(x,E,1)=exp(-5)(1+5^|Z^1|) for one through five detections. Zero retained detections means there is no nonempty candidate group, so only a=0 is possible and gamma(x,E,1)=exp(-5). Do not substitute zero into the nonempty-group formula: adding an assigned empty group would duplicate the null hypothesis. The absent branch permits only a=0 and has gamma(x,E,0)=1 for all rows. Keeping the null assignment is essential: conditioning on a non-null assignment would instead force existence. Substitution into the legacy existence update gives 0.9 gamma1/(0.1+0.9 gamma1). The posterior probabilities for retained counts zero through five are respectively 0.057174381426, 0.266781074113, 0.611903629439, 0.884270402211, 0.974333698575, and 0.994752457710, displayed as 5.72, 26.68, 61.19, 88.43, 97.43, and 99.48 percent. The predicted count parameter remains five in every row. Any excluded detections are not supplied to this legacy candidate; no additional group is introduced. This isolates group-size loss with fixed association messages. A full BP rerun with other split groups or competing tracks can also change the incoming messages and need not reproduce these numbers. The numerical assumptions are illustrative, with no newborn update or simulated-data claim.'));
  slides.push(slide('takeaways','15 / What to remember','Keep the method, schedule, and protocol distinct.',[
    ...[['GrBP is the local method.','A fixed group partition feeds belief-propagation association.'],['CS and CP schedule evidence differently.','Sequential updates reuse updated beliefs; parallel fusion counts the prior once.'],['Handover preserves local track continuity.','H sends priors. HM / HL / HP add evidence and fused posterior returns.']].flatMap(([h,b],i)=>[
      T('end-num'+i,80,230+i*132,56,46,''+(i+1),34,C.green,700),T('end-head'+i,165,231+i*132,1000,46,h,29,C.ink,700),T('end-copy'+i,165,289+i*132,1000,52,b,22,C.muted)
    ]),
    T('conditions',92,631,1096,32,'Scalability assumes bounded degree, bounded local count moments, and finite message representations.',16,C.muted)
  ],'Close with the scope of the claim. The per-node load result requires bounded local moments and representation size as well as bounded neighborhood degree. The webpage includes the editable TikZ diagrams, PDF exports, variants table, and the same live demonstrations.'));
  slides.forEach((s,i)=>{if(s.__demo)s.__demo.slideIndex=i;});
  return slides;
}
export default {
  target:'et-handover/index.html',docId:'eo-handover-grbp-2026-09',
  title:TITLE,subject:'GrBP processing architectures and owner-centered handover',
  description:'Scalable extended-target handover in distributed integrated sensing and communication (ISAC): GrBP diagrams, four handover variants, and interactive examples.',
  footer:'GrBP · Extended-target handover',ink:C.ink,paper:C.paper,accent:C.green,
  fontFamily:FONT,build
};
