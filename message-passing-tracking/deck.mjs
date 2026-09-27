import {createRequire} from 'node:module';
const require=createRequire(import.meta.url),BP=require('./model.js'),G=require('./graph.js'),r=BP.run();
const R=String.raw,I=s=>`<span class="math-tex math-inline">\\(${s}\\)</span>`,M=s=>`<span class="math-tex math-display">\\[${s}\\]</span>`;
const C={paper:'#F4F7FA',white:'#FFFFFF',ink:'#203446',muted:'#617181',line:'#D8E1E8',blue:'#126CAA',tint:'#E8F2F9',orange:'#C16023',teal:'#168078'},sans='Arial, Helvetica, sans-serif',serif="Georgia, 'Times New Roman', serif";
const T=(id,x,y,w,h,html,fontSize=22,opts={})=>({id,type:'text',x,y,w,h,html,fontSize,fontFamily:sans,fontWeight:400,color:C.ink,lineHeight:1.36,align:'left',valign:'top',rotation:0,opacity:1,...opts});
const B=(id,x,y,w,h,fill=C.white,opts={})=>({id,type:'shape',shape:'rect',x,y,w,h,fill,stroke:'none',strokeWidth:0,rotation:0,opacity:1,...opts});
function L(id,x1,y1,x2,y2,color=C.line,width=2){const len=Math.hypot(x2-x1,y2-y1);return B(id,(x1+x2-len)/2,(y1+y2-width)/2,len,width,color,{rotation:Math.atan2(y2-y1,x2-x1)*180/Math.PI});}
export const bounds={x:72,y:176,width:1136,height:488};
function graph(x,y,w,step='none'){
 const scale=w/760,out=[];
 for(const [a,b] of G.edges)out.push(L('edge-'+a+'-'+b,x+G.nodes[a].x*scale,y+G.nodes[a].y*scale,x+G.nodes[b].x*scale,y+G.nodes[b].y*scale,C.line,2));
 for(const [a,b] of G.paths(step,0,0)){const p=G.segment(a,b),angle=Math.atan2(p.y2-p.y1,p.x2-p.x1),k=a+'-'+b;out.push(L('message-'+k,x+p.x1*scale,y+p.y1*scale,x+p.x2*scale,y+p.y2*scale,C.orange,3));for(const side of [-1,1])out.push(L('arrow-'+k+'-'+side,x+p.x2*scale,y+p.y2*scale,x+(p.x2-10*Math.cos(angle+side*.5))*scale,y+(p.y2-10*Math.sin(angle+side*.5))*scale,C.orange,3));}
 for(const [id,n] of Object.entries(G.nodes)){
  const circle=n.kind==='variable',sz=(circle?36:n.kind==='constraint'?12:28)*scale,cx=x+n.x*scale,cy=y+n.y*scale;
  out.push(B('node-'+id,cx-sz/2,cy-sz/2,sz,sz,n.kind==='transition'?'#CAE2CE':circle?'#FFFFFF':'#F4D3D2',{shape:circle?'ellipse':'rect',stroke:'#405769',strokeWidth:1.4,radius:3}));
  out.push(T('label-'+id,cx-39*scale,cy+(n.kind==='constraint'?-35:-13)*scale,78*scale,34*scale,I(n.label),(n.kind==='constraint'?17:20)*scale,{align:'center'}));
 }
 return out;
}
const source='Meyer et al. (2018)',paper='https://doi.org/10.1109/JPROC.2018.2789427';
function slide(id,section,title,subtitle,elements,notes,page='221–259'){
 return {id,background:C.paper,transition:'none',notes:`${notes}\nSource: Meyer et al., Message Passing Algorithms for Scalable Multitarget Tracking, Proceedings of the IEEE 106(2), pp. ${page}. ${paper}`,elements:[T('section',72,29,1080,22,section.toUpperCase(),12,{color:C.blue,fontWeight:700,letterSpacing:1.2}),T('title',72,66,1136,58,title,38,{fontFamily:serif,fontWeight:700,lineHeight:1.1}),T('subtitle',72,128,1136,38,subtitle,17,{color:C.muted}),...elements,B('footer-rule',72,676,1136,1,C.line),T('home',72,690,310,18,'BAI LIPING · RANDOM THOUGHTS',11,{color:C.muted,link:'https://bailiping.com/#group-random-thoughts'}),T('source',388,689,640,20,`${source} · p. ${page}`,11,{color:C.muted,align:'center',link:paper}),T('pages',1090,689,118,20,'{{page}} / {{pages}}',11,{color:C.muted,align:'right'})]};
}
const E=(id,x,y,w,h,latex,size=25)=>T(id,x,y,w,h,M(latex),size,{align:'center',valign:'middle'});
const S=(id,x,y,w,h,text,size=20)=>T(id,x,y,w,h,text,size,{color:C.muted});
const mount=()=>B('live-demo-mount',bounds.x,bounds.y,bounds.width,bounds.height,'rgba(255,255,255,0)',{opacity:0});
const pct=x=>(100*x).toFixed(2)+'%';
export const slides=[];
slides.push(slide('overview','Paper notes · Figure 4','Message passing for multitarget tracking','A guided reading of Meyer, Kropfreiter, Williams, Lau, Hlawatsch, Braca and Win (2018).',[
 T('claim',72,218,550,157,'From uncertain matches<br>to uncertain tracks',43,{fontFamily:serif,fontWeight:700,lineHeight:1.15}),
 T('cover-copy',72,402,515,104,'Follow one scan through prediction, association, and the birth or survival of possible targets.',25),
 ...graph(630,230,578,'nu'),
 B('cover-rule',72,537,1136,1,C.line),
 ...[['01','Read the graph','graph'],['02','Walk every message','schedule'],['03','Check the beliefs','beliefs']].flatMap(([n,text,id],i)=>[T('toc-number-'+i,72+i*386,567,40,30,n,16,{color:C.blue,fontWeight:700,link:id}),T('toc-'+i,119+i*386,561,325,38,text,23,{fontFamily:serif,link:id})]),
 S('cover-nav',72,625,1136,26,'Arrow keys advance · Page Up / Page Down navigate from the demo · Escape returns focus to the slides',15)
],'The focus is Figure 4, printed p. 242 (PDF p. 22), and the single-sensor algorithm in Section IX-A. The graph is instantiated with two legacy potential targets and two measurements. The two-cell state model and numerical data are original teaching choices.','241–243'));
slides.push(slide('graph','01 · Figure 4, redrawn','One scan contains two kinds of uncertainty','Target state and existence sit outside; measurement ownership is resolved in the middle.',[
 T('previous-label',72,194,137,29,'Previous',17,{color:C.muted}),T('legacy-label',237,194,450,29,'Legacy targets at the current scan',18,{color:C.blue}),T('new-label',900,194,305,29,'Possible new targets',18,{color:C.blue}),
 ...graph(80,220,1120),
 S('legend',72,631,1136,33,'Circles: variables · Green squares: transitions · Red squares: local factors and association constraints',17)
],'This shows the complete current-scan topology of Figure 4 for n_p = n_m = 2. The previous scan is collapsed into its two posterior inputs y_-; the preceding association subgraph has already been processed. Every a variable is connected to every b variable through one Psi factor. A new potential target is attached to each measurement.','242'));
slides.push(slide('variables','01 · The variables','A potential target may not exist','An association choice and an existence indicator answer different questions.',[
 E('state',72,203,539,100,R`y=(x,r),\qquad r\in\{0,1\}`,30),
 T('legacy-meaning',83,327,501,87,`${I(R`\underline y^j`)}: a legacy candidate carried from the previous scan.`,24),
 T('new-meaning',83,441,501,116,`${I(R`\overline y^m`)}: a possible new target introduced by measurement ${I(R`m`)}.`,24),
 B('variable-rule',640,208,1,367,C.line),
 E('association-domain',697,205,487,83,R`a^j\in\{0,1,2\},\quad b^m\in\{0,1,2\}`,25),
 T('a-def',695,332,481,95,`${I(R`a^j=m`)}: target ${I(R`j`)} produced measurement ${I(R`m`)}.<br>${I(R`a^j=0`)}: no detection from that target.`,22),
 T('b-def',695,462,481,112,`${I(R`b^m=j`)}: measurement ${I(R`m`)} came from legacy target ${I(R`j`)}.<br>${I(R`b^m=0`)}: new target <em>or</em> clutter.`,22),
 S('candidate-not-target',72,617,1136,30,'A new candidate is created for each measurement; its posterior existence may remain very small.',19)
],'PT means potential target, not a declared target. A zero target-oriented association allows both missed detection and nonexistence. A zero measurement-oriented association excludes legacy ownership but does not distinguish a new target from clutter. The v factor performs that distinction.','236–243'));
slides.push(slide('teaching-model','01 · A concrete numerical instance','Two cells make every sum inspectable','The figure and update rules come from the paper. These numerical choices are for teaching.',[
 E('discrete-states',72,193,525,91,R`y\in\{\varnothing,L,R\},\quad x_L=0,\ x_R=1`,27),
 E('priors',72,310,535,147,R`\begin{aligned}\widetilde f_-^1&=[0.10,\ 0.80,\ 0.10]\\\widetilde f_-^2&=[0.20,\ 0.15,\ 0.65]\end{aligned}`,25),
 S('state-order',87,483,509,92,'Each vector lists absent, left, right. The absent entry aggregates the paper’s normalized dummy state density.',22),
 E('transition',663,190,538,200,R`f(y\mid y_-)=\begin{pmatrix}1&0&0\\0.05&0.8075&0.1425\\0.05&0.1425&0.8075\end{pmatrix}`,26),
 S('matrix-order',685,402,490,48,'Rows: previous state. Columns: current state.',18),
 E('sensing',669,478,529,104,R`\begin{aligned}z_1&=0.30,\quad z_2=0.62,\quad p_D=0.85\\L_m(x)&=\mathcal N(z_m;x,0.35^2)\end{aligned}`,24),
 S('model-boundary',72,619,1136,31,'Clutter intensity = 0.5 · Expected new detections = 0.3 · New-target location mass = [0.5, 0.5]',17)
],'The model replaces continuous-state integrals by finite sums, with two position cells and one aggregated absent state. Survival is 0.95. Conditional on survival, staying in the same cell has probability 0.85. Gaussian measurement densities are evaluated at the observed z. Clutter intensity means mu_c f_c(z) = 0.5 in the observation region. mu_n = 0.3 is the expected number of newly detected targets, as in the paper, so no extra detection factor is inserted into v.','236–243'));
slides.push(slide('weights','02 · Before the association loop','Local factors turn states into association weights','Prediction is a probability vector. The outgoing measurement weights are generally unnormalized.',[
 E('q-definition',72,192,591,175,R`q^j(y,a)=\begin{cases}\mathbf1[a=0]&y=\varnothing\\1-p_D&y\ne\varnothing,\ a=0\\p_D L_a(x)/\lambda_c&y\ne\varnothing,\ a>0\end{cases}`,25),
 E('beta-rule',693,202,482,133,R`\beta_j(a)=\sum_y q^j(y,a)\alpha_j(y)`,27),
 E('beta-default',678,358,514,91,R`\beta_1\approx[0.2733,\ 0.9371,\ 0.4758]`,22),
 B('weight-rule',72,464,1136,1,C.line),
 E('xi-rule',72,487,599,127,R`\xi_m(0)=1+B_m,\qquad\xi_m(j)=1`,27),
 E('birth-evidence',702,477,472,90,R`B_m=\sum_x\frac{\mu_n f_n(x)L_m(x)}{\lambda_c}`,25),
 S('xi-copy',703,579,470,60,'The 1 is the absent-new-target contribution. The extra mass supports a new detection.',19)
],'q is the finite-state counterpart of (72), using (58), (64) and (65). lambda_c abbreviates mu_c f_c(z). The message beta is (78). v comes from (73) and (74), so xi(0)=1+B and xi(j)=1 for every legacy target index j. These are local association weights, not normalized marginal probabilities.','240–243'));
slides.push(slide('schedule','02 · The Figure 4 walkthrough','Follow the information out and back','Only the association block repeats. The rest of the scan runs in one forward-time schedule.',[
 ...graph(72,210,787,'nu'),
 ...[['1','Predict','previous posterior → transition → state'],['2','Evaluate','state → local weights for a and b'],['3','Associate','iterate through the consistency factors'],['4','Return','association evidence → state factors'],['5','Form beliefs','normalize state and existence weights']].flatMap(([n,h,body],i)=>[T('stage-n-'+n,900,192+i*75,29,29,n,21,{color:C.blue,fontWeight:700}),T('stage-h-'+n,943,192+i*75,250,30,h,22,{fontFamily:serif}),S('stage-body-'+n,943,227+i*75,261,39,body,14)]),
 B('watch-bg',72,599,1136,60,C.tint),T('watch',93,613,1090,34,'Next: advance one message, select its output entry, then inspect the sum that produced it.',20)
],'The demo exposes both parts of each association pass: the variable multiplies other inputs and the Psi factor sums compatible assignments. Pairs update in parallel within each half-sweep, but the orange path shows one selected pair. The schedule follows prediction, measurement evaluation, iterative DA, measurement update, and belief calculation in IX-A. It begins with the previous posterior and uses three association sweeps by default.','242–243'));
slides.push(slide('schedule-live','02 · Interactive Figure 4','Walk the graph, one message at a time','Use Next message, or choose a stage. Select a track, measurement, and output entry to inspect.',[
 B('fallback-bg',72,176,1136,488,C.white,{stroke:C.line,strokeWidth:1,radius:8}),
 T('fallback-heading',94,192,1060,34,'1 / 17 · Send the previous posterior',21,{color:C.blue,fontWeight:700}),
 ...graph(89,262,707,'prior'),
 T('fallback-inspector',832,257,342,66,'Track 1 → transition factor',25,{fontFamily:serif}),
 E('fallback-vector',831,335,342,110,R`\widetilde f_-^1=[0.10,\ 0.80,\ 0.10]`,22),
 S('fallback-explain',832,477,341,120,'The entries are absent, left, and right. The next message will sum this posterior through the transition matrix.',21),
 S('fallback-next',94,606,701,30,'Next: prediction → local weights → association loop → state beliefs.',18),mount()
],'The direct route is /message-passing-tracking/live/. The embedded walkthrough starts at the first message. Seventeen steps are present with three complete nu/phi sweeps. Every arithmetic table uses the actual finite-state model; no illustrative values are hard-coded into the renderer. Click graph nodes or selectors to inspect either legacy candidate or measurement. Click existence cards to jump to final beliefs. Sliders change measurement 2, detection probability and expected newly detected targets. Printed slides preserve this initial state.','242–243'));
slides.push(slide('association','03 · Inside the association block','One message leaves out one neighbor','The consistency factor prevents two targets from claiming the same measurement.',[
 E('psi-rule',72,189,1136,104,R`\Psi^{j,m}(a,b)=\mathbf1\big[(a=m)\Longleftrightarrow(b=j)\big]`,30),
 E('nu-rule',72,323,1136,84,R`\nu^{[\ell]}_{m,j}(a)\propto\sum_b\Psi^{j,m}(a,b)\,\xi_m(b)\!\prod_{i\ne j}\varphi^{[\ell-1]}_{i,m}(b)`,26),
 E('phi-rule',72,436,1136,84,R`\varphi^{[\ell]}_{j,m}(b)\propto\sum_a\Psi^{j,m}(a,b)\,\beta_j(a)\!\prod_{n\ne m}\nu^{[\ell]}_{n,j}(a)`,26),
 S('excluded-meaning',72,574,549,79,'Excluding the recipient avoids immediately sending its own information back to it.',21),
 S('scaling-meaning',696,574,512,79,'Each message has two distinct values. The demo rescales nonmatching entries to 1.',21)
],'Equations (27) and (28) combine a variable-to-factor product and a factor-to-variable sum. They are applied in Section IX-A3 using beta from (78) and xi from (79). Initialization here follows the vector formula (29), then rescales each message. This is not the alternative scalar initialization beta(m)/beta(0) stated after (31). Scaling a whole message by a positive constant does not change normalized beliefs.','232, 243'));
slides.push(slide('beliefs','03 · What comes out of the graph','Stable messages can still give approximate beliefs','The demo checks its small loopy graph against all seven valid association matchings.',[
 E('legacy-belief',72,187,559,96,R`\widetilde f(\underline y^j)\propto\alpha_j(\underline y^j)\gamma_j(\underline y^j)`,27),
 E('new-belief',691,187,504,96,R`\widetilde f(\overline y^m)\propto\varsigma_m(\overline y^m)`,27),
 T('belief-heading-name',105,328,280,31,'Candidate',22,{fontWeight:700}),T('belief-heading-bp',487,328,280,31,'BP existence',22,{fontWeight:700}),T('belief-heading-exact',850,328,280,31,'Exact existence',22,{fontWeight:700}),
 ...[...r.legacy.map((v,j)=>['Legacy '+(j+1),v,r.exact.legacy[j]]),...r.newTargets.map((v,m)=>['New '+(m+1),v,r.exact.newTargets[m]])].flatMap(([name,bp,exact],i)=>[T('belief-name-'+i,105,379+i*50,271,35,name,24),T('belief-bp-'+i,487,379+i*50,228,35,pct(1-bp[0]),24,{color:C.blue}),T('belief-exact-'+i,850,379+i*50,260,35,pct(1-exact[0]),24,{color:C.teal})]),
 S('belief-boundary',72,604,1136,50,'Defaults, after 3 sweeps. Exact means exhaustive inference for this two-cell teaching model, not ground-truth tracking.',18)
],'Existence equals the sum of the present left and right states, or one minus the absent-state belief. The exact reference enumerates the seven valid one-to-one matchings and marginalizes all target states. It has also been verified by full enumeration of the independent finite joint distribution. Finite BP iterates are approximate; the shared loop can leave nonzero marginal error even after messages stabilize.','243'));
slides.push(slide('next-scan','04 · Across time','Today’s new candidates become tomorrow’s legacy set','Messages travel forward through time; there is no backward smoothing pass in this schedule.',[
 T('legacy-box',92,224,452,75,'Updated legacy candidates',29,{fontFamily:serif}),
 E('legacy-outputs',103,319,434,76,R`\widetilde f(\underline y_k^1),\quad\widetilde f(\underline y_k^2)`,30),
 T('new-box',92,427,452,61,'Updated new candidates',29,{fontFamily:serif}),
 E('new-outputs',103,496,434,76,R`\widetilde f(\overline y_k^1),\quad\widetilde f(\overline y_k^2)`,30),
 T('forward-arrow',576,356,97,92,'→',50,{color:C.blue,align:'center'}),
 B('next-tint',719,221,489,336,C.tint),T('next-title',747,246,437,46,'Prediction at the next scan',28,{fontFamily:serif}),
 E('next-states',743,330,435,70,R`y_{k+1}^j\sim f(y_{k+1}^j\mid y_k^j)`,25),
 S('next-explain',747,435,427,93,'After optional pruning, both groups enter the same legacy-target prediction step.',22),
 S('declaration',72,616,1136,34,'Declaring and pruning targets uses existence thresholds; the demo keeps all candidates so their beliefs stay visible.',18)
],'The paper concatenates updated legacy and new potential targets into the state for the next time step. Section IX-A6 describes declaration, state estimation and pruning thresholds. The demo studies one scan and deliberately does not simulate tracks or apply an arbitrary threshold. Its previous inputs are an already-computed filtering posterior, not another repeated association graph.','236–237, 243–244'));
slides.push(slide('scalability','04 · Why message passing helps','Local updates replace a global assignment sum','Each sweep visits target–measurement pairs and reuses compact summaries.',[
 T('local-head',72,220,543,54,'What scales',31,{fontFamily:serif}),
 T('local-copy',72,302,531,175,'The two-valued consistency messages admit scalar ratios. With shared sums, one association sweep costs work proportional to the number of candidate pairs.',24),
 E('pair-cost',82,511,510,82,R`O(n_p n_m)\quad\text{per DA sweep}`,28),
 B('scale-rule',649,218,1,382,C.line),
 T('limits-head',698,220,497,54,'What the figure does not promise',29,{fontFamily:serif}),
 T('limits-copy',698,302,488,257,'A fixed number of sweeps does not certify exact posterior marginals.<br><br>Continuous-state integration, particle counts, gating, and the number of potential targets also affect total tracking cost.',23)
],'Section VI-B derives scalar ratio messages (30) and (31). With precomputed shared sums, each DA sweep is linear in the number of target-measurement pairs, not in the number of joint assignments. This does not claim that arbitrary loopy BP is exact or that an entire tracker has a cost independent of state representation and iteration count.','232–234, 242–244'));
slides.push(slide('references','Sources & scope','Read alongside Figure 4 and Section IX-A','The diagram is redrawn; the source PDF is linked, not redistributed.',[
 T('paper-title',72,202,1114,93,'Message Passing Algorithms for<br>Scalable Multitarget Tracking',33,{fontFamily:serif,fontWeight:700,link:paper}),
 S('authors',72,321,1136,77,'Florian Meyer, Thomas Kropfreiter, Jason L. Williams, Roslyn A. Lau,<br>Franz Hlawatsch, Paolo Braca and Moe Z. Win.',21),
 S('journal',72,419,1136,34,'Proceedings of the IEEE, 106(2), 221–259, February 2018.',21),
 T('doi-link',72,483,1120,35,'DOI: 10.1109/JPROC.2018.2789427 ↗',20,{color:C.blue,link:paper}),
 T('paper-copy',72,536,1120,35,'Paper in the Australian National University repository ↗',20,{color:C.blue,link:'https://openresearch-repository.anu.edu.au/items/7bffe07d-bd73-4fcb-ab87-18ba3e100d40'}),
 T('read-static',72,615,490,29,'Static reading / print view ↗',18,{color:C.blue,link:'https://bailiping.com/message-passing-tracking/study.html'}),
 T('bp-primer',679,615,521,29,'Companion: factor graphs & sum-product ↗',18,{color:C.blue,link:'https://bailiping.com/factor-graphs/'})
],'Primary source: the user-supplied paper, checked visually at Figure 4 and equations (27)–(31), (73)–(86). ANU provides an institutional repository copy. Figures, controls, calculations and example parameters in this lesson are newly authored. The two-cell finite model is not a reported experiment from the paper.','221–259'));
export const deck={format:'bento/slides',version:1,docId:'message-passing-tracking-fig4',title:'Message Passing for Multitarget Tracking',readonly:true,meta:{author:'Bai Liping',subject:'A step-by-step reading of Figure 4 in Meyer et al. (2018)',company:'bailiping.com'},size:{width:1280,height:720},theme:{background:C.paper,color:C.ink,accent:C.blue,fontFamily:sans},slides};
export const inlineLiveMap=[{introSlide:'schedule',slide:'schedule-live',slideIndex:slides.findIndex(s=>s.id==='schedule-live'),inline:true,layout:'region',bounds,src:'./live/?embed=region',source:'./live/',title:'Figure 4: every message in one tracking scan',sandbox:'allow-scripts',hideSource:true,readyMessage:true,unloadWhenHidden:true}];
