// Companion source for the 25-slide revision. Applied by build-bento.mjs.
// Keep the existing native Bento document and its three numerical labs intact.
const R=String.raw;
const M=s=>`<span class="math-tex math-display">\\[${s}\\]</span>`;
const I=s=>`<span class="math-tex math-inline">\\(${s}\\)</span>`;
const sans="Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
const mono="'SFMono-Regular', Consolas, monospace";
function txt(id,x,y,w,h,html,size=18){return {id,type:'text',x,y,w,h,rotation:0,opacity:1,html,fontSize:size,fontFamily:sans,fontWeight:400,color:'#203129',align:'left',valign:'top',lineHeight:1.35};}
function rect(id,x,y,w,h,fill='#FFFEFB'){return {id,type:'shape',shape:'rect',x,y,w,h,rotation:0,opacity:1,fill,stroke:'#D8DED7',strokeWidth:1,radius:12};}

export function applyTeachingRevisions(deck,liveMap){
  if(deck.slides.some(s=>s.id==='dp-demo'))return; // Idempotent when rebuilding in memory.
  const get=id=>{const s=deck.slides.find(s=>s.id===id);if(!s)throw new Error('Missing base slide: '+id);return s;};
  const element=(sid,id)=>{const e=get(sid).elements.find(e=>e.id===id);if(!e)throw new Error('Missing base element: '+sid+'/'+id);return e;};
  const set=(sid,id,html,size)=>{const e=element(sid,id);e.html=html;if(size)e.fontSize=size;e.lineHeight=1.32;};
  set('objective','objective-left-body',`${M(R`p(\Theta\mid Z)\propto\prod_k f_k(\Theta_k)`)}${M(R`f_k(\Theta_k)\propto\exp\!\left[-\tfrac12\lVert h_k(\Theta_k)-z_k\rVert_{\Sigma_k}^{2}\right]`)}Each factor touches only a few variables; a prior is another factor.<br><br><b>Gaussian measurement errors do not make a nonlinear posterior globally Gaussian.</b>`,18);
  set('objective','objective-right-body',`${M(R`\hat\Theta=\arg\min_\Theta\tfrac12\sum_k\lVert e_k(\Theta_k)\rVert_{\Sigma_k}^{2}`)}${M(R`e_k=h_k(\Theta_k)-z_k,\quad\lVert e\rVert_\Sigma^2=e^{\mathsf T}\Sigma^{-1}e`)}<b>Mahalanobis squared norm:</b> residuals are weighted by inverse covariance.${M(R`\Sigma=\operatorname{diag}(\sigma_i^2)\ \Rightarrow\ \lVert e\rVert_\Sigma^2=\sum_i(e_i/\sigma_i)^2`)}Larger variance means less weight; reliable measurements count more.`,17);
  get('objective').notes+=' The covariance-weighted norm is explicitly defined in K11 Eq. (3). The diagonal-covariance expression is a teaching expansion of that definition.';
  set('linearize','linearize-left-body',`${M(R`e_k(\bar\Theta\oplus\delta)\approx e_k(\bar\Theta)+J_k\delta_k`)}${M(R`A_k=W_kJ_k,\qquad b_k=-W_ke_k(\bar\Theta)`)}${M(R`W_k^{\mathsf T}W_k=\Sigma_k^{-1}`)}${I(R`W_k`)} is a <b>whitening matrix</b>, often denoted ${I(R`\Sigma_k^{-1/2}`)}.<br><br>Stack the blocks into ${I('A')} and ${I('b')}. The retraction ${I(R`\oplus`)} applies a tangent-space increment.`,18);
  set('isam','isam-left-body',`${M(R`\begin{bmatrix}R_t&d_t\\A_{\mathrm{new}}&b_{\mathrm{new}}\end{bmatrix}\xrightarrow{\mathrm{Givens}}\begin{bmatrix}R_{t+1}&d_{t+1}\\0&r_\perp\end{bmatrix}`)}<b>Append:</b> add measurement rows and any new variable columns.<br><br><b>Rotate:</b> restore triangular form.<br><br><b>Solve:</b> back-substitute for the trajectory.<br><br>For the <b>same current linearized system</b>, this gives the same least-squares solution as fresh batch QR, while reusing the old factorization.`,17);
  const bayes=get('bayes-tree');
  bayes.notes+=' The following original quadratic example explains separator-dependent summaries and reconstruction as a dynamic-programming analogy. It is not an additional algorithm claimed by the source papers.';
  const dp={id:'dp-demo',background:'#F7F5EF',transition:'none',notes:'Original quadratic teaching example. E_L(u,s)=0.5(u-1)^2+0.5(s-u)^2; minimizing over u yields m_L(s)=0.25(s-1)^2 and u*(s)=(s+1)/2. Adding E_R(s;z)=0.5(s-z)^2 gives s*=(1+2z)/3. This is a min-sum / dynamic-programming explanation, not a full iSAM2 implementation. For this fixed-curvature Gaussian example, marginalizing exp(-E_L) yields exp(-m_L(s)) times a constant independent of s. Do not equate minimization with probabilistic integration for arbitrary models. The root has frontal s and an empty separator; s is the separator of its child. Moving z leaves the child summary and conditional unchanged but changes both estimates. K11 Section IV and Algorithms 3-4 support caching subtree marginals and downward state recovery; the numerical example is original.',elements:[]};
  dp.elements=bayes.elements.filter(e=>['eyebrow','heading','subtitle'].includes(e.id)||e.id.startsWith('chrome-')).map(e=>structuredClone(e));
  dp.elements.find(e=>e.id==='eyebrow').html='INTERACTIVE LAB 02 · AN ORIGINAL QUADRATIC EXAMPLE';
  dp.elements.find(e=>e.id==='heading').html='Reuse a summary. Recover a new estimate.';
  dp.elements.find(e=>e.id==='subtitle').html='A dynamic-programming view: cache a function of the separator, not one frozen child estimate.';
  dp.elements.find(e=>e.id==='chrome-source').html='[K11] §IV, Algs. 3–4 · original min-sum analogy';
  dp.elements.push(rect('dp-fallback',72,180,1136,475),
    txt('dp-fallback-label',96,202,1088,24,'STATIC / PRINT VIEW · MOVE z IN THE LIVE LAB',11),
    rect('dp-left-panel',96,248,520,330,'#E7F0EA'),
    txt('dp-left-body',119,268,474,282,`<b>1 · Summarize the child once</b>${M(R`E_L(u,s)=\tfrac12(u-1)^2+\tfrac12(s-u)^2`)}${M(R`m_L(s)=\min_uE_L(u,s)=\tfrac14(s-1)^2`)}${M(R`u^\star(s)=\tfrac12(s+1)`)}The summary and reconstruction rule remain valid when only external evidence changes.`,17),
    rect('dp-right-panel',638,248,546,330,'#F5E8DF'),
    txt('dp-right-body',661,268,500,282,`<b>2 · Add evidence, then reconstruct</b>${M(R`E_R(s;z)=\tfrac12(s-z)^2`)}${M(R`s^\star=\arg\min_s[m_L(s)+E_R(s;z)]=\tfrac{1+2z}{3}`)}For ${I('z=1')}: ${I(R`s^\star=u^\star=1`)}.<br><br>For ${I('z=4')}: ${I(R`s^\star=3,\ u^\star=2`)}.<br><br><b>Same cached child; different estimates.</b>`,17),
    txt('dp-scope',98,597,1080,33,'This lab explains reuse; it does not implement a numerical incremental Bayes-tree solver.',14),
    {...rect('live-demo-mount',72,180,1136,475),fill:'rgba(255,255,255,0)',stroke:'rgba(255,255,255,0)',opacity:0});
  deck.slides.splice(deck.slides.indexOf(bayes)+1,0,dp);
  set('overview','cover-labs','4 LIVE LABS · QR ↗ · SUMMARY REUSE ↗ · TREE ↗ · BRANCHING ↗',12);
  set('references','refs-labs','4 LABS · incremental QR ↗ · separator summaries ↗ · Bayes-tree locality ↗ · delayed disambiguation ↗',12);
  set('tree-demo','eyebrow','INTERACTIVE LAB 03',11);
  set('mh-demo','eyebrow','INTERACTIVE LAB 04',11);
  for(const s of deck.slides){
    s.notes=s.notes.replace(/three (browser|live) experiments/g,'four $1 experiments');
    for(const e of s.elements)if(e.type==='text')e.html=e.html.replace('In Lab 3:','In Lab 4:');
  }
  liveMap.push({introSlide:'bayes-tree',slide:'dp-demo',slideIndex:0,inline:true,layout:'region',bounds:{x:72,y:180,width:1136,height:475},src:'./live/dp.html?embed=region',source:'./live/dp.html',title:'Bayes-tree reuse as dynamic programming',sandbox:'allow-scripts',hideSource:true,readyMessage:true,unloadWhenHidden:true});
  for(const entry of liveMap)entry.slideIndex=deck.slides.findIndex(s=>s.id===entry.slide);
  liveMap.sort((a,b)=>a.slideIndex-b.slideIndex);
  deck.meta.subject='Incremental smoothing, Bayes-tree reuse, discrete ambiguity, and four interactive teaching labs';
}
