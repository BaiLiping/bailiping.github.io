import {createRequire} from 'node:module';
const require=createRequire(import.meta.url),BP=require('./model.js'),G=require('./graph.js');
const result=BP.run();
const C={paper:'#F4F7FA',white:'#FFFFFF',ink:'#203446',muted:'#617181',line:'#D8E1E8',blue:'#126CAA',tint:'#E8F2F9',orange:'#CA6218',teal:'#1E766D'};
const sans='Arial, Helvetica, sans-serif',serif="Georgia, 'Times New Roman', serif";
const R=String.raw,I=s=>`<span class="math-tex math-inline">\\(${s}\\)</span>`,M=s=>`<span class="math-tex math-display">\\[${s}\\]</span>`;
export const bounds={x:72,y:180,width:1136,height:480};
const T=(id,x,y,w,h,html,size=22,opts={})=>({id,type:'text',x,y,w,h,html,fontSize:size,fontFamily:sans,fontWeight:400,color:C.ink,lineHeight:1.36,align:'left',valign:'top',rotation:0,opacity:1,...opts});
const B=(id,x,y,w,h,fill=C.white,opts={})=>({id,type:'shape',shape:'rect',x,y,w,h,fill,stroke:'none',strokeWidth:0,rotation:0,opacity:1,...opts});
function L(id,x1,y1,x2,y2,color=C.line,width=2){const len=Math.hypot(x2-x1,y2-y1);return B(id,(x1+x2-len)/2,(y1+y2-width)/2,len,width,color,{rotation:Math.atan2(y2-y1,x2-x1)*180/Math.PI});}
function graph(x,y,w,phase=0){
  const scale=w/720,shapes=[];
  for(const [f,v] of BP.edges){const a=G.nodes[f],b=G.nodes[v];shapes.push(L(`edge-${f}-${v}`,x+a[0]*scale,y+a[1]*scale,x+b[0]*scale,y+b[1]*scale,C.line,3));}
  for(const m of Object.values(result.messages).filter(m=>m.phase===phase)){
    const p=G.segment(m.from,m.to,5),dx=p.x2-p.x1,dy=p.y2-p.y1,angle=Math.atan2(dy,dx),k=m.from+'-'+m.to;
    shapes.push(L(`message-${k}`,x+p.x1*scale,y+p.y1*scale,x+p.x2*scale,y+p.y2*scale,C.blue,3));
    for(const side of [-1,1])shapes.push(L(`arrow-${k}-${side}`,x+p.x2*scale,y+p.y2*scale,x+(p.x2-12*Math.cos(angle+side*.5))*scale,y+(p.y2-12*Math.sin(angle+side*.5))*scale,C.blue,3));
  }
  for(const [n,[nx,ny]] of Object.entries(G.nodes)){
    const isX=n.startsWith('x'),sz=(isX?52:34)*scale,cx=x+nx*scale,cy=y+ny*scale;
    shapes.push(B('node-'+n,cx-sz/2,cy-sz/2,sz,sz,isX?C.white:C.ink,{shape:isX?'ellipse':'rect',stroke:isX?C.ink:'none',strokeWidth:2,radius:3}));
    shapes.push(T('label-'+n,cx-35*scale,cy+(isX?-18:23)*scale,70*scale,40*scale,I(G.tex(n)),Math.max(17,24*scale),{align:'center'}));
  }
  return shapes;
}
const source='Kschischang, Frey & Loeliger (2001)';
const paperURL='https://doi.org/10.1109/18.910572';
function slide(id,section,title,sub,elements,notes,page='498–519'){
  return {id,background:C.paper,transition:'none',notes:`${notes}\nSource: ${source}, Factor Graphs and the Sum-Product Algorithm, IEEE Transactions on Information Theory 47(2), pp. ${page}. ${paperURL}`,elements:[
    T('section',72,29,900,22,section.toUpperCase(),12,{color:C.blue,fontWeight:700,letterSpacing:1.3}),
    T('title',72,66,1136,58,title,39,{fontFamily:serif,fontWeight:700,lineHeight:1.08}),
    T('subtitle',72,128,1136,40,sub,17,{color:C.muted}),...elements,
    B('footer-rule',72,676,1136,1,C.line),T('home',72,690,300,18,'BAI LIPING · RANDOM THOUGHTS',11,{color:C.muted,link:'https://bailiping.com/#group-random-thoughts'}),
    T('source',376,689,660,20,`${source} · p. ${page}`,11,{color:C.muted,align:'center',link:paperURL}),
    T('pages',1090,689,118,20,'{{page}} / {{pages}}',11,{color:C.muted,align:'right'})]};
}
const equation=(id,x,y,w,h,latex,size=27)=>T(id,x,y,w,h,M(latex),size,{align:'center',valign:'middle'});
const small=(id,x,y,w,h,html,size=18)=>T(id,x,y,w,h,html,size,{color:C.muted});
const mount=()=>B('live-demo-mount',bounds.x,bounds.y,bounds.width,bounds.height,'rgba(255,255,255,0)',{opacity:0});
export const slides=[];
slides.push(slide('overview','Paper notes · 2001','Factor graphs & the sum-product algorithm','Frank R. Kschischang · Brendan J. Frey · Hans-Andrea Loeliger',[
 T('cover-claim',72,218,565,126,'Belief propagation,<br>one message at a time',44,{fontFamily:serif,fontWeight:700,lineHeight:1.14}),
 T('cover-copy',72,369,520,104,'A reading of the paper’s opening example: how local sums and products recover global marginals.',24),
 ...graph(654,214,554,3),
 B('cover-line',72,494,1136,1,C.line),
 ...[['01','The factorization','factor-graph'],['02','The five-step example','schedule'],['03','Inside one message','factor-message']].flatMap(([n,label,id],j)=>[
 T('toc-n-'+j,72+j*386,527,48,31,n,16,{color:C.blue,fontWeight:700,link:id}),
 T('toc-t-'+j,119+j*386,522,320,36,label,22,{fontFamily:serif,link:id})]),
 small('cover-nav',72,607,1136,27,'Arrow keys advance · Page Up / Page Down navigate from a demo · Escape returns focus to the slides',15)
],'The paper presents sum-product on general factor graphs. We focus on its Example 1 and Section II-D. The topology, factor arguments and five-phase schedule are from the paper; binary alphabets and numerical factor values are explicitly introduced for this lesson.','498–503'));
slides.push(slide('marginal-problem','01 · The computational problem','A marginal sums over everything else','The global function is a product; the question concerns just one of its variables.',[
 equation('product',105,192,1070,86,R`g(x_1,\ldots,x_n)=\prod_{a\in F} f_a(\mathbf x_a)`,31),
 equation('marginal',105,317,1070,94,R`g_i(x_i)=\sum_{\mathbf x\setminus x_i}\prod_{a\in F}f_a(\mathbf x_a)`,31),
 B('norm-tint',72,451,1136,174,C.tint),
 T('prob-note',99,475,494,115,'For nonnegative factors with finite, positive total mass, normalize the marginal to obtain a probability.',22),
 equation('norm',635,470,535,134,R`p_i(x_i)=\frac{g_i(x_i)}{Z},\quad Z=\sum_{x_i}g_i(x_i)`,27)
],'The paper solves the marginalize-product-of-functions problem. g need not already be a probability distribution. For our finite nonnegative example Z is positive. The not-sum notation in the paper names variables retained; this deck uses ordinary sums over variables eliminated.','499'));
slides.push(slide('factor-graph','01 · Example 1, equation (2)','The paper’s five-variable factor graph','A circle represents a variable; a square represents one local function.',[
 equation('example-product',72,186,1136,75,R`g(\mathbf x)=f_A(x_1)f_B(x_2)f_C(x_1,x_2,x_3)f_D(x_3,x_4)f_E(x_3,x_5)`,25),
 ...graph(240,262,800),
 small('topology-note',72,646,1136,26,'Five variables · Five factors · Nine edges · A connected tree, redrawn in the orientation of Figure 7',17)
],'Example 1, equation (2) and Figure 1 establish the exact scopes used here. Figure 7 redraws the same tree with the central x3–fC edge horizontal. No observation factors or edges have been added.','499–500, 503'));
slides.push(slide('distributivity','01 · Why the graph saves work','The distributive law exposes reusable sums','For the central variable, the three branches can be summarized independently.',[
 equation('g3-decomp',72,179,1136,204,R`\begin{aligned}g_3(x_3)=&\underbrace{\sum_{x_1,x_2}f_A(x_1)f_B(x_2)f_C(x_1,x_2,x_3)}_{\text{left branch}}\\[-2pt]&\times\underbrace{\sum_{x_4}f_D(x_3,x_4)}_{\text{upper branch}}\times\underbrace{\sum_{x_5}f_E(x_3,x_5)}_{\text{lower branch}}\end{aligned}`,24),
 ...graph(72,365,610,3),
 T('branch-meaning',790,410,399,134,'Each bracket is a function of only the variable at the branch boundary.',25,{fontFamily:serif}),
 small('branch-reuse',790,555,399,78,'Store that function as a message. Reuse it when computing other marginals.',19)
],'This is equation (4), with explicit summation variables replacing the paper’s summary notation. Cutting the central edge separates disjoint sets of factors. The distributive law gives exact arithmetic on the tree.','500'));
slides.push(slide('update-rules','01 · Equations (5) and (6)','Two local message rules','Every outgoing message excludes the message arriving from its recipient.',[
 T('var-head',72,204,450,35,'Variable → factor',26,{fontFamily:serif,color:C.blue}),
 small('var-copy',72,251,380,85,'Multiply the other incoming messages, point by point.',22),
 equation('var-eq',479,202,729,139,R`\mu_{x\to f}(x)=\prod_{h\in N(x)\setminus\{f\}}\mu_{h\to x}(x)`,27),
 B('rules-line',72,373,1136,1,C.line),
 T('factor-head',72,415,450,35,'Factor → variable',26,{fontFamily:serif,color:C.blue}),
 small('factor-copy',72,462,380,126,'Multiply by the local factor, then sum out every variable except the recipient.',22),
 equation('factor-eq',479,396,729,174,R`\mu_{f\to x}(x)=\sum_{\mathbf x_f\setminus x} f(\mathbf x_f)\!\prod_{y\in N(f)\setminus\{x\}}\mu_{y\to f}(y)`,25),
 small('leaf-note',72,615,1136,34,`Leaf variable: ${I(R`\mu=1`)}. Leaf factor: send the factor itself. A message is a function, not a single belief.`,18)
],'The update rule is Figure 6 and equations (5)–(6). All messages on an edge have the variable associated with that edge as their sole argument. Omitting the recipient prevents immediate reuse of the information it sent.','502–503'));
slides.push(slide('numerical-example','02 · A numerical version of Example 1','Concrete factor values for the demo','The paper leaves these functions symbolic. The following binary model is a teaching choice.',[
 T('binary',72,191,1136,41,`${I(R`x_i\in\{0,1\}`)} for every variable. Factor arguments and graph topology remain exactly those of Example 1.`,21),
 equation('unary',72,259,524,97,R`f_A=[0.7,\;0.3],\quad f_B=[0.4,\;0.6]`,24),
 equation('xor',72,384,524,138,R`f_C=\begin{cases}q&x_3=x_1\oplus x_2\\1-q&\text{otherwise}\end{cases}\quad q=0.85`,25),
 small('xor-def',82,546,504,65,'The XOR of two bits is 1 when they differ. The central factor favors that relation.',19),
 B('values-rule',632,261,1,347,C.line),
 equation('D',680,254,500,119,R`f_D=\begin{pmatrix}3&1\\1&2\end{pmatrix}`,29),
 equation('E',680,410,500,119,R`f_E=\begin{pmatrix}2&1\\1&4\end{pmatrix}`,29),
 small('matrix-note',681,561,500,62,`Rows index ${I(R`x_3`)}; columns index ${I(R`x_4`)} or ${I(R`x_5`)}. These are nonnegative weights.`,18)
],'The original worked example is symbolic. The numerical factors here are original teaching data, not numerical results reported in the paper. Unary vectors are in state order [0,1]. The fixed D and E matrices are not normalized conditional distributions. The global product must be normalized by Z.','499–503'));
slides.push(slide('schedule','02 · Figure 7 and Section II-D','The five-step message schedule','Messages move inward, meet across the central edge, then move outward.',[
 ...graph(72,206,716,3),
 ...[['1','Leaves send their initial messages'],['2','The outer branches summarize'],['3','The central edge carries both directions'],['4','The centre sends outward'],['5','The leaf marginals are available']].flatMap(([n,copy],i)=>[
 T('phase-number-'+n,839,206+i*71,32,33,n,23,{color:C.blue,fontWeight:700}),T('phase-copy-'+n,882,206+i*71,320,58,copy,19)]),
 B('observe-strip',72,574,1136,74,C.tint),T('observe',94,591,1090,45,'Next: compare the two central messages at step 3. Select step 1 to replay the full schedule.',21)
],'This is the schedule in Figure 7, p. 503. There are 4, 4, 2, 4 and 4 messages in the five steps, for 18 total. The central marginal has all three incoming factor messages at step 3; x1 and x2 at step 4; x4 and x5 at step 5. No partially informed belief is labeled exact.','503'));
slides.push(slide('schedule-live','02 · Interactive example','How each message is computed','Click an arrow for its arithmetic, or a variable for the messages that form its belief.',[
 B('fallback-bg',72,180,1136,480,C.white,{stroke:C.line,strokeWidth:1,radius:8}),
 T('fallback-stage',97,199,1000,30,'STEP 3 / 5 · The two halves exchange summaries',20,{color:C.blue,fontWeight:700}),
 ...graph(94,264,558,3),
 equation('fallback-calculation',690,251,492,209,R`\begin{aligned}\mu_{f_C\to x_3}(1)&=0.15(0.7)(0.4)\\&\quad+0.85(0.7)(0.6)\\&\quad+0.85(0.3)(0.4)\\&\quad+0.15(0.3)(0.6)\\&=0.528\end{aligned}`,23),
 equation('fallback-output',688,477,492,65,R`\mu_{f_C\to x_3}=[0.472,\;0.528]`,24),
 small('fallback-excluded',98,548,527,70,`The left message summarizes ${I(R`f_A,f_B,f_C`)}. It excludes the return message from ${I(R`x_3`)}.`,19),
 small('fallback-note',690,560,492,64,`Combining all three inputs at ${I(R`x_3`)} gives ${I(R`p(x_3=1)=0.5830`)} after normalization.`,19),mount()
],'Begin at step 3 to inspect the central sum. Green nodes and arrows mark the factors and inputs used by the selected outgoing message. Choose either retained state to inspect its summands. The opposite message x3 to fC depends only on fD and fE. Click a variable or the Variable belief tab to inspect its incoming vectors, product and normalization. Partial beliefs use only messages received at the current phase and are explicitly not the final marginals. Choose step 1 or Reset to replay the paper’s five phases.','503'));
slides.push(slide('factor-message','03 · Anatomy of a message','Inside the central factor’s sum','Hold the destination variable fixed. Enumerate only the other factor arguments.',[
 equation('central-message',72,190,1136,103,R`\mu_{f_C\to x_3}(x_3)=\sum_{x_1,x_2}f_A(x_1)f_B(x_2)f_C(x_1,x_2,x_3)`,28),
 T('term-label',72,335,540,35,`For ${I(R`x_3=1`)}, the four contributions are`,23),
 equation('four-terms',72,388,616,145,R`\begin{aligned}0.7\cdot0.4\cdot0.15&=0.042\\0.7\cdot0.6\cdot0.85&=0.357\\0.3\cdot0.4\cdot0.85&=0.102\\0.3\cdot0.6\cdot0.15&=0.027\end{aligned}`,25),
 equation('sum-output',791,343,385,151,R`\mu_{f_C\to x_3}(1)=0.528`,29),
 small('no-return',798,503,387,90,`The recipient’s message ${I(R`\mu_{x_3\to f_C}`)} does not appear in this sum.`,22),
 small('factor-observe',72,615,1136,30,`Next: change the factor inputs. At ${I(R`q=0.5`)}, the outgoing message is uniform.`,19)
],'This is step 3 of Section II-D, evaluated with our teaching factors. x1 and x2 messages are fA and fB because each corresponding variable has degree two. The four summands for output one are 0.042, 0.357, 0.102 and 0.027; their sum is 0.528. For output zero the result is 0.472.','503'));
slides.push(slide('factor-message-live','03 · Interactive calculation','Each output value gets its own sum','Choose the retained value, then vary the two unary weights and the XOR agreement.',[
 B('factor-fallback-bg',72,180,1136,480,C.white,{stroke:C.line,strokeWidth:1,radius:8}),
 T('fallback-retained',97,207,1040,45,`Retained value: ${I(R`x_3=1`)}`,24),
 equation('factor-fallback-rows',97,282,670,233,R`\begin{array}{cc|ccc|c}x_1&x_2&f_A&f_B&f_C&\text{product}\\\hline0&0&0.7&0.4&0.15&0.042\\0&1&0.7&0.6&0.85&0.357\\1&0&0.3&0.4&0.85&0.102\\1&1&0.3&0.6&0.15&0.027\end{array}`,23),
 equation('factor-fallback-value',812,296,345,139,R`\mu_{f_C\to x_3}=[0.472,\;0.528]`,25),
 small('factor-fallback-note',812,461,345,100,'Every row multiplies local evidence. The final addition eliminates the two input variables.',21),
 small('factor-fallback-foot',97,610,1070,32,'The numerical values are illustrative; the update is the paper’s exact sum-product rule.',18),mount()
],'The live calculation expands the message from fC to x3. Switching retained value recomputes each fC entry; changing a, b or q changes the four contributions. The message is not the posterior at x3: that posterior also includes the two right-hand branches.','502–503'));
slides.push(slide('beliefs','04 · Termination and normalization','Messages combine into exact marginals','A marginal uses every incoming factor message; an outgoing message excludes one.',[
 equation('belief-eq',72,185,1136,102,R`g_i(x_i)=\prod_{f\in N(x_i)}\mu_{f\to x_i}(x_i),\qquad p_i(x_i)=\frac{g_i(x_i)}{\sum_{a\in\{0,1\}}g_i(a)}`,27),
 equation('default-g3',72,319,609,130,R`\begin{aligned}g_3&=[0.472,\;0.528]\odot[4,\;3]\odot[3,\;5]\\&=[5.664,\;7.920]\\Z&=13.584\end{aligned}`,25),
 small('normalization-note',84,498,575,97,`Normalize once to obtain ${I(R`p_3(1)\approx0.583039`)}. The central message alone would give 0.528.`,23),
 ...BP.variables.flatMap((v,i)=>[T('belief-label-'+v,754,320+i*57,97,30,I(R`p(${G.tex(v)}=1)`),21),B('belief-track-'+v,885,323+i*57,229,19,'#DCE8EE'),B('belief-bar-'+v,885,323+i*57,229*result.beliefs[v][1],19,C.teal),T('belief-number-'+v,1127,318+i*57,82,28,result.beliefs[v][1].toFixed(4),19,{color:C.teal})])
],'The termination equations are on p. 503. These are results of the stated teaching factors. Raw marginals at all five variables sum to the same Z=13.584. Independent enumeration over 32 configurations agrees with all normalized BP marginals to floating-point precision. Elementwise multiplication is denoted by the circled dot.','503'));
slides.push(slide('tree-exactness','04 · Why a tree is special','One cut separates the information','Every message summarizes the factors on its sender’s side of an edge.',[
 ...graph(72,221,716,3),
 L('cut',423,233,423,560,C.orange,2),T('cut-label',368,573,159,34,'CUT ONE EDGE',13,{color:C.orange,fontWeight:700,align:'center'}),
 T('cut-meaning',844,207,348,205,'On a tree, the two sides share only the boundary variable. Their factors are disjoint.',27,{fontFamily:serif}),
 small('cut-proof',844,432,348,185,'Sum each side separately, then multiply the two messages to recover the exact marginal. Two messages per edge suffice.',21),
 small('complexity',72,622,720,31,'Local cost still grows with the number of states in a factor’s arguments.',17)
],'The cut-edge interpretation is stated in Section II-B and proved in Appendix A. The visual cut crosses the unique central edge between fC and x3. A tree with nine edges requires 18 directed messages for all marginals. This is not a universal linear-cost claim: summation within a high-arity factor can be exponential in its arity.','501–503, 516–517'));
slides.push(slide('beyond-trees','05 · The wider paper','The same rule appears in many algorithms','Graph structure and message representation determine the familiar special case.',[
 ...[['Chain / hidden Markov model','Forward–backward uses finite state message vectors.'],['Linear Gaussian state model','Gaussian messages give Kalman filtering and smoothing.'],['Alternative sum/product operations','The semiring view connects related dynamic programs.']].flatMap(([head,body],i)=>[T('family-head-'+i,72,203+i*118,552,37,head,25,{fontFamily:serif}),small('family-body-'+i,72,250+i*118,552,60,body,21)]),
 B('loopy-background',685,195,523,411,C.tint),
 T('loop-head',715,222,462,42,'What changes with cycles?',28,{fontFamily:serif,color:C.blue}),
 T('loop-copy',715,289,451,254,'Messages are reused iteratively, and the separated-subtree argument no longer applies.<br><br>Convergence and exact marginals are not guaranteed in general.<br><br>The five-step exact result belongs to this tree.',22)
],'Sections IV and V discuss trellis/state-space algorithms and factor graphs with cycles. The sum-product algorithm becomes loopy BP under iterative scheduling. For Viterbi the operations change to max-product or equivalent min-sum; it is not ordinary sum-product marginalization. The present demo deliberately retains the original tree.','506–514'));
slides.push(slide('references','Sources & reading','The paper and the worked example','The source supplies the symbolic example; this lesson supplies its numerical instantiation.',[
 T('citation-title',72,208,1100,69,'Factor Graphs and the Sum-Product Algorithm',34,{fontFamily:serif,fontWeight:700,link:paperURL}),
 small('citation-authors',72,291,1136,70,'Frank R. Kschischang, Brendan J. Frey and Hans-Andrea Loeliger.<br>IEEE Transactions on Information Theory, 47(2), 498–519, February 2001.',22),
 T('doi-link',72,380,1100,35,'DOI: 10.1109/18.910572 ↗',21,{color:C.blue,link:paperURL}),
 T('author-copy',72,433,1100,35,'Author-hosted paper at ETH Zürich ↗',21,{color:C.blue,link:'https://www.isiweb.ee.ethz.ch/papers/arch/aloe-2001-1.pdf'}),
 B('reading-rule',72,495,1136,1,C.line),
 T('reading-guide',72,519,1110,90,'Example 1 & Figure 1: pp. 499–500 · Update rules: p. 502<br>Five-step example & Figure 7: p. 503 · Cyclic graphs: Section V',20),
 T('read-static',72,617,520,29,'Static reading / print view ↗',18,{color:C.blue,link:'https://bailiping.com/factor-graphs/study.html'})
],'Primary reference only. The linked ETH copy is hosted by one of the authors’ institutions. This deck does not redistribute the user’s downloaded licensed PDF. All diagrams are redrawn for teaching with the paper’s factor scopes, and the binary model is clearly disclosed as an original illustrative choice.'));
export const deck={format:'bento/slides',version:1,docId:'factor-graphs-sum-product',title:'Factor Graphs & the Sum-Product Algorithm',readonly:true,meta:{author:'Bai Liping',subject:'The opening example of Kschischang, Frey & Loeliger (2001)',company:'bailiping.com'},size:{width:1280,height:720},theme:{background:C.paper,color:C.ink,accent:C.blue,fontFamily:sans},slides};
export const inlineLiveMap=[['schedule','schedule-live','schedule','The five-step sum-product example'],['factor-message','factor-message-live','factor','Inside the central factor message']].map(([introSlide,id,demo,title])=>({introSlide,slide:id,slideIndex:slides.findIndex(s=>s.id===id),inline:true,layout:'region',bounds,src:`./live/?demo=${demo}&embed=region`,source:`./live/?demo=${demo}`,title,sandbox:'allow-scripts',hideSource:true,readyMessage:true,unloadWhenHidden:true}));
