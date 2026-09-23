// EO-specific authoring source. The local builder preserves the Bento runtime.
const C = { paper:'#ffffff', ink:'#16273e', muted:'#596d80', rule:'#d8e1e9', wash:'#f2f6fa', green:'#087f68', blue:'#2766b1', orange:'#b96815', purple:'#7854a3', red:'#c62828' };
const FONT = 'Arial, Helvetica, sans-serif';
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
    src:`./live/?slide-embed=%23${selector}&v=${id==='protocol-live'?'grbp-transfer-compact-2026-09':'grbp-2026-09'}`,source:`./live/#${selector}`,title,
    sandbox:'allow-scripts allow-same-origin',hideSource:true,readyMessage:true,unloadWhenHidden:true};
  return s;
}
function build() {
  const slides = [];
  slides.push({id:'s-cover',background:C.paper,transition:'none',notes:'Introduce grouped-measurement belief propagation (GrBP). This talk separates the local method, its processing schedule, and track-level handover. Figure 1 is reproduced unchanged from the manuscript: gray discs denote sensing fields of view, blue is the target trajectory, black dots are incidence points, and red highlights handover. The central unit depicts the coordinated baseline.',elements:[
    T('eyebrow',72,35,1136,24,'DISTRIBUTED ISAC / EXTENDED-TARGET TRACKING',13,C.green,700),
    T('cover-title',72,79,1136,76,'Scalable Extended-Target Handover',52,C.ink,700),
    T('point-target-reference',72,151,640,26,'<a class="cover-reference" href="/target-handover-slides/"><span>extention of point-target handover</span><span aria-hidden="true"> →</span></a>',17,C.green,600),
    {...T('paper-reference',752,151,456,26,'<a class="cover-reference" href="https://arxiv.org/abs/2609.25737"><span>Paper · arXiv:2609.25737</span><span aria-hidden="true"> →</span></a>',17,C.green,600),align:'right'},
    R('rule',72,185,1136,1,C.rule),
    I('manuscript-figure-1','./assets/manuscript-figure-1.png',220,200,840,458,'Figure 1: extended-target handover in a DISAC network, with overlapping sensing regions, a target trajectory, and a red handover arrow.'),
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
    I('simulation-animation','./assets/simulation-environment.gif',36,30,630,630,'Animated GrBP tracking trial mc_0088 across seven base stations and their overlapping sensing regions'),
    // Replace the baked-in static heading; the adjacent animated frame counter stays visible.
    R('simulation-heading-background',219.75,38.4,210,14.7,C.paper),
    {...T('simulation-heading',219.75,38.8,209,13,'GrBP Tracking | mc_0088 |',10.5,'#000000'),align:'right'},
    T('environment-network',724,244,464,72,'Seven base stations with<br>overlapping sensing regions',26,C.ink,700),
    T('environment-motion',724,342,464,104,'Targets move across the network while base stations collect measurements.',24,C.muted),
    T('environment-watch',724,482,464,88,'Watch targets cross coverage boundaries and move between stations.',23,C.green,700),
    T('environment-source',724,611,464,45,'GrBP tracking · trial mc_0088',16,C.muted)
  ],'Simulation-environment introduction, using the user-supplied simulation animation with its static heading relabeled GrBP Tracking in the slide. Seven base stations are shown as green pentagons; dashed green circles show sensing range. The original animation legend identifies measurements, true trajectories and positions, estimated trajectories and positions, estimated position covariance, and estimated extent. The trial is mc_0088. The GIF contains 100 frames, plays at 10 frames per second, and loops; its animated frame counter remains visible. Introduce the overlapping surveillance regions before advancing to the aggregate results.');
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
  slides.push(slide('takeaways','13 / What to remember','Keep the method, schedule, and protocol distinct.',[
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
  title:'Scalable Extended-Target Handover',subject:'GrBP processing architectures and owner-centered handover',
  description:'Paper-aligned GrBP diagrams, four handover variants, and interactive examples.',
  footer:'GrBP · Extended-target handover',ink:C.ink,paper:C.paper,accent:C.green,
  fontFamily:FONT,build
};
